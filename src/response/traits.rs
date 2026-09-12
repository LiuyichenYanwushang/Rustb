//! Reusable band-resolved Berry-curvature interface.

use ndarray::prelude::*;
use ndarray::{ArrayBase, Data};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::math::anti_comm;
use crate::velocity::Velocity;
use crate::{Gauge, Model, RMatrixData, SpinDirection};

use super::config::{Conditions, direction_matrix, validate_broadening, validate_direction_values};

/// Berry curvature and energies of every band at one k-point.
#[derive(Clone, Debug, PartialEq)]
pub struct BandBerryCurvature {
    /// Berry curvature of every band.
    pub berry_curvature: Array1<f64>,
    /// Band energies in eV.
    pub energies: Array1<f64>,
}

/// Berry-curvature methods shared by tight-binding-like model types.
///
/// None of these methods integrates over the Brillouin zone. The band-resolved
/// methods depend only on the supplied k-point, so they take no thermodynamic
/// state at all; the occupation-weighted methods take the one fixed DC
/// [`Conditions`] point they weight the bands with, and never a k-mesh.
pub trait BerryCurvature<const DIM: usize>: Velocity {
    /// Evaluate the charge or spin Berry curvature of every band at one k-point.
    ///
    /// `directions[0]` and `directions[1]` are the two tensor indices, `eta_ev`
    /// broadens the denominator and `spin` selects the spin current. Any
    /// occupation weighting is applied by the caller, so no temperature,
    /// chemical potential or k-mesh is involved.
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<BandBerryCurvature>;

    /// Sum band Berry curvatures with the electronic occupation selected by the
    /// fixed temperature and chemical potential of `conditions`.
    ///
    /// `conditions` must pin all three axes, with `omega_ev = 0`; it is not a
    /// k-mesh configuration and no mesh is validated here.
    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        conditions: &Conditions,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<f64>;

    /// Evaluate occupation-weighted Berry curvature on multiple k-points.
    ///
    /// See [`Self::occupied_berry_curvature_at`] for the `conditions` contract.
    fn occupied_berry_curvature_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        conditions: &Conditions,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<Array1<f64>>;
}

impl<const DIM: usize, R: RMatrixData> BerryCurvature<DIM> for Model<false, DIM, R> {
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<BandBerryCurvature> {
        self.validate()?;
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        if let Some(direction) = spin {
            return Err(TbError::SpinNotAllowed(direction));
        }
        let direction = direction_matrix(&directions);
        self.berry_curvature_at_impl(k, &direction, None, eta_ev)
    }

    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        conditions: &Conditions,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<f64> {
        let resolved = conditions.resolve()?;
        resolved.require_fixed_dc()?;
        let (_, chemical_potential, _) = resolved.point(0);
        let occupation = resolved.occupation(0);
        let bands = self.berry_curvature_at(k, directions, eta_ev, spin)?;
        Ok(bands
            .berry_curvature
            .iter()
            .zip(&bands.energies)
            .map(|(&berry, &energy)| berry * occupation.value_unchecked(energy, chemical_potential))
            .sum())
    }

    fn occupied_berry_curvature_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        conditions: &Conditions,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<Array1<f64>> {
        self.validate()?;
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "Berry-curvature k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        if let Some(direction) = spin {
            return Err(TbError::SpinNotAllowed(direction));
        }
        // Validate once up front, then reuse the unvalidated kernel per k-point.
        let resolved = conditions.resolve()?;
        resolved.require_fixed_dc()?;
        let direction = direction_matrix(&directions);
        let (_, chemical_potential, _) = resolved.point(0);
        let occupation = resolved.occupation(0);
        let values: Vec<Result<f64>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| {
                let bands = self.berry_curvature_at_impl(&k, &direction, None, eta_ev)?;
                Ok(bands
                    .berry_curvature
                    .iter()
                    .zip(&bands.energies)
                    .map(|(&berry, &energy)| {
                        berry * occupation.value_unchecked(energy, chemical_potential)
                    })
                    .sum())
            })
            .collect();
        Ok(Array1::from_vec(
            values.into_iter().collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl<const DIM: usize, R: RMatrixData> BerryCurvature<DIM> for Model<true, DIM, R> {
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<BandBerryCurvature> {
        self.validate()?;
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        let spin = spin.map(|direction| self.build_spin_matrix(direction));
        let direction = direction_matrix(&directions);
        self.berry_curvature_at_impl(k, &direction, spin.as_ref(), eta_ev)
    }

    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        conditions: &Conditions,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<f64> {
        let resolved = conditions.resolve()?;
        resolved.require_fixed_dc()?;
        let (_, chemical_potential, _) = resolved.point(0);
        let occupation = resolved.occupation(0);
        let bands = self.berry_curvature_at(k, directions, eta_ev, spin)?;
        Ok(bands
            .berry_curvature
            .iter()
            .zip(&bands.energies)
            .map(|(&berry, &energy)| berry * occupation.value_unchecked(energy, chemical_potential))
            .sum())
    }

    fn occupied_berry_curvature_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        conditions: &Conditions,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
        spin: Option<SpinDirection>,
    ) -> Result<Array1<f64>> {
        self.validate()?;
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "Berry-curvature k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        // Validate once up front, then reuse the unvalidated kernel per k-point.
        let resolved = conditions.resolve()?;
        resolved.require_fixed_dc()?;
        let direction = direction_matrix(&directions);
        let (_, chemical_potential, _) = resolved.point(0);
        let occupation = resolved.occupation(0);
        let spin = spin.map(|direction| self.build_spin_matrix(direction));
        let values: Vec<Result<f64>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| {
                let bands = self.berry_curvature_at_impl(&k, &direction, spin.as_ref(), eta_ev)?;
                Ok(bands
                    .berry_curvature
                    .iter()
                    .zip(&bands.energies)
                    .map(|(&berry, &energy)| {
                        berry * occupation.value_unchecked(energy, chemical_potential)
                    })
                    .sum())
            })
            .collect();
        Ok(Array1::from_vec(
            values.into_iter().collect::<Result<Vec<_>>>()?,
        ))
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Band-resolved Berry-curvature kernel without input validation.
    ///
    /// Callers must have already validated the direction rank, the optional
    /// spin matrix against the model, and `eta`. Entry points construct the
    /// spin matrix once with `Model<true>::build_spin_matrix`, then reuse it
    /// per k-point; the public trait methods are independent
    /// boundaries and validate before delegating here.
    pub(crate) fn berry_curvature_at_impl<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        direction: &Array2<f64>,
        spin: Option<&Array2<Complex<f64>>>,
        eta: f64,
    ) -> Result<BandBerryCurvature> {
        let (projected_velocity, hamiltonian) = self.gen_v_projected(k, Gauge::Atom, direction);
        #[cfg(test)]
        super::config::counters::count_eigen_decomposition();
        let (energies, eigenvectors) = hamiltonian.eigh(UPLO::Lower)?;

        let current: Array2<Complex<f64>> = if let Some(spin_matrix) = spin {
            anti_comm(spin_matrix, &projected_velocity.index_axis(Axis(0), 0)) * 0.5
        } else {
            projected_velocity.index_axis(Axis(0), 0).to_owned()
        };
        let second_velocity = projected_velocity.index_axis(Axis(0), 1);
        let bra = eigenvectors.t();
        let ket = eigenvectors.mapv(|value| value.conj());
        let current_band = bra.dot(&current.dot(&ket));
        let velocity_band = bra.dot(&second_velocity.dot(&ket));
        let kernel = current_band * velocity_band.reversed_axes();
        let eta_squared = eta * eta;
        let mut berry_curvature = Array1::<f64>::zeros(self.nsta());

        for band in 0..self.nsta() {
            let mut value = 0.0;
            for other in 0..self.nsta() {
                if band == other {
                    continue;
                }
                let difference = energies[band] - energies[other];
                value += -2.0 * kernel[[band, other]].im / (difference * difference + eta_squared);
            }
            berry_curvature[band] = value;
        }
        Ok(BandBerryCurvature {
            berry_curvature,
            energies,
        })
    }
}
