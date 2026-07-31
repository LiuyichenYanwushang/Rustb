//! Electronic occupations and thermal energy scales.
//!
//! Response functions and self-consistent solvers use the same occupation
//! convention through [`Occupation`].  Temperatures are expressed in kelvin;
//! Hamiltonian energies and smearing widths are expressed in electronvolts.

use crate::error::{Result, TbError};

/// Boltzmann constant in electronvolts per kelvin.
pub const BOLTZMANN_CONSTANT_EV_PER_K: f64 = 8.617_333_262_145e-5;

/// Electronic occupation used in Brillouin-zone integrations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Occupation {
    /// Exact zero-temperature step function.
    #[default]
    ZeroTemperature,

    /// Physical Fermi-Dirac distribution at a temperature in kelvin.
    FermiDirac {
        /// Temperature in kelvin. Must be finite and strictly positive.
        temperature_kelvin: f64,
    },

    /// Fermi-function smearing specified directly as an energy width.
    ///
    /// This is useful for metallic calculations whose physical temperature is
    /// zero but whose Brillouin-zone integration needs a smooth occupation.
    FermiSmearing {
        /// Smearing energy in electronvolts. Must be finite and positive.
        width: f64,
    },
}

impl Occupation {
    /// Validate the thermodynamic parameters.
    pub fn validate(self) -> Result<()> {
        match self {
            Self::ZeroTemperature => Ok(()),
            Self::FermiDirac { temperature_kelvin } => {
                validate_positive("temperature_kelvin", temperature_kelvin)
            }
            Self::FermiSmearing { width } => validate_positive("width", width),
        }
    }

    /// Thermal energy width in electronvolts.
    ///
    /// Zero is returned for [`Occupation::ZeroTemperature`].
    pub fn energy_width(self) -> Result<f64> {
        self.validate()?;
        Ok(match self {
            Self::ZeroTemperature => 0.0,
            Self::FermiDirac { temperature_kelvin } => {
                temperature_kelvin * BOLTZMANN_CONSTANT_EV_PER_K
            }
            Self::FermiSmearing { width } => width,
        })
    }

    /// Occupation of a state with energy `energy` at chemical potential `mu`.
    pub fn value(self, energy: f64, mu: f64) -> Result<f64> {
        validate_finite("energy", energy)?;
        validate_finite("chemical_potential", mu)?;
        Ok(fermi_from_width(energy, mu, self.energy_width()?))
    }

    /// The positive Fermi window `-df/dE`.
    ///
    /// An exact zero-temperature derivative is a Dirac delta and cannot be
    /// represented as a scalar. Energy-cut algorithms handle that limit
    /// explicitly; direct sums should use [`Occupation::FermiSmearing`].
    pub fn minus_derivative(self, energy: f64, mu: f64) -> Result<f64> {
        validate_finite("energy", energy)?;
        validate_finite("chemical_potential", mu)?;
        let width = self.energy_width()?;
        if width == 0.0 {
            return Err(TbError::InvalidThermodynamicParameter {
                parameter: "occupation",
                message: "the zero-temperature derivative is a Dirac delta; use an energy-cut integration or finite smearing".into(),
            });
        }
        Ok(fermi_derivative_from_width(energy, mu, width))
    }

    pub(crate) fn value_unchecked(self, energy: f64, mu: f64) -> f64 {
        let width = match self {
            Self::ZeroTemperature => 0.0,
            Self::FermiDirac { temperature_kelvin } => {
                temperature_kelvin * BOLTZMANN_CONSTANT_EV_PER_K
            }
            Self::FermiSmearing { width } => width,
        };
        fermi_from_width(energy, mu, width)
    }
}

/// Numerically stable Fermi function parameterized by an energy width.
#[inline]
pub(crate) fn fermi_from_width(energy: f64, mu: f64, width: f64) -> f64 {
    if width == 0.0 {
        return if energy < mu {
            1.0
        } else if energy > mu {
            0.0
        } else {
            0.5
        };
    }
    let x = (energy - mu) / width;
    if x > 50.0 {
        0.0
    } else if x < -50.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// Numerically stable positive Fermi window `-df/dE`.
#[inline]
pub(crate) fn fermi_derivative_from_width(energy: f64, mu: f64, width: f64) -> f64 {
    debug_assert!(width > 0.0);
    let occupation = fermi_from_width(energy, mu, width);
    occupation * (1.0 - occupation) / width
}

fn validate_positive(parameter: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TbError::InvalidThermodynamicParameter {
            parameter,
            message: "must be finite and strictly positive".into(),
        });
    }
    Ok(())
}

fn validate_finite(parameter: &'static str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(TbError::InvalidThermodynamicParameter {
            parameter,
            message: "must be finite".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_temperature_is_a_step() {
        let occupation = Occupation::ZeroTemperature;
        assert_eq!(occupation.value(-1.0, 0.0).unwrap(), 1.0);
        assert_eq!(occupation.value(0.0, 0.0).unwrap(), 0.5);
        assert_eq!(occupation.value(1.0, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn thermal_and_smearing_widths_are_equivalent() {
        let temperature = 300.0;
        let thermal = Occupation::FermiDirac {
            temperature_kelvin: temperature,
        };
        let smeared = Occupation::FermiSmearing {
            width: temperature * BOLTZMANN_CONSTANT_EV_PER_K,
        };
        let a = thermal.value(0.02, 0.0).unwrap();
        let b = smeared.value(0.02, 0.0).unwrap();
        assert!((a - b).abs() < 1e-14);
    }

    #[test]
    fn zero_temperature_derivative_requires_energy_cut() {
        assert!(
            Occupation::ZeroTemperature
                .minus_derivative(0.0, 0.0)
                .is_err()
        );
    }
}
