//! Shared configuration types for response calculations.
//!
//! A response calculation fixes a thermodynamic point and may sample one of
//! its three physical axes. Temperature, chemical potential and photon
//! frequency enter a response only through the occupation factors and the
//! frequency denominators, so the expensive k-mesh preparation — eigenstates,
//! velocity kernels and band tracking — is performed once and reused by every
//! sample of the sampled axis.

use ndarray::array;
use ndarray::{Array1, Array2, ArrayView1};

use crate::error::{Result, TbError};
use crate::thermodynamics::Occupation;

/// Brillouin-zone integration algorithm shared by all response entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integration {
    /// Uniform k-mesh summation.
    Direct,
    /// Band-tracked simplex quadrature (quantum geometry, optical conductivity).
    Simplex,
    /// Band-tracked energy-cut integration (Hall, nonlinear Hall).
    EnergyCut,
}

/// Whether the two external-field indices of the extrinsic nonlinear Hall
/// response are explicitly symmetrized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldSymmetry {
    /// Return the ordered kernel `S[current, field_1; field_2]`.
    Ordered,
    /// Return `(S[current, field_1; field_2] +
    /// S[current, field_2; field_1]) / 2`.
    Symmetrized,
}

/// One physical axis of a response: either pinned to a single value or sampled
/// on a series of values.
#[derive(Clone, Debug, PartialEq)]
pub enum Sampling {
    /// A single value; this axis is not sampled.
    Fixed(f64),
    /// The values the response is evaluated at.
    Values(Array1<f64>),
}

impl Sampling {
    /// The sampled coordinates, or `None` for a fixed axis.
    pub fn values(&self) -> Option<&Array1<f64>> {
        match self {
            Self::Fixed(_) => None,
            Self::Values(values) => Some(values),
        }
    }

    /// Number of response evaluations this axis requests.
    pub fn len(&self) -> usize {
        match self {
            Self::Fixed(_) => 1,
            Self::Values(values) => values.len(),
        }
    }

    /// `true` only for an empty `Values` series, which validation rejects.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether this axis is pinned instead of sampled.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// The value this axis takes at sample `index`; a fixed axis ignores
    /// `index`.
    pub fn value_at(&self, index: usize) -> f64 {
        match self {
            Self::Fixed(value) => *value,
            Self::Values(values) => values[index],
        }
    }

    fn validate(&self, parameter: &'static str) -> Result<()> {
        match self {
            Self::Fixed(value) => {
                if !value.is_finite() {
                    return Err(TbError::InvalidResponseParameter {
                        parameter,
                        message: "must be finite".into(),
                    });
                }
            }
            Self::Values(values) => {
                if values.is_empty() {
                    return Err(TbError::InvalidResponseParameter {
                        parameter,
                        message: "must contain at least one value".into(),
                    });
                }
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(TbError::InvalidResponseParameter {
                        parameter,
                        message: "all values must be finite".into(),
                    });
                }
            }
        }
        Ok(())
    }
}

/// The three physical axes of a response calculation.
///
/// At most one axis may be sampled; the result is then indexed by that axis.
/// Sampling several axes at once would need a multi-dimensional result and is
/// rejected before any k-mesh work happens.
#[derive(Clone, Debug, PartialEq)]
pub struct Conditions {
    /// Temperature in kelvin. `0` selects the exact zero-temperature step
    /// function; a positive value selects Fermi-Dirac occupation.
    pub t_kelvin: Sampling,
    /// Chemical potential in eV.
    pub mu_ev: Sampling,
    /// Photon / perturbation frequency in eV.
    pub omega_ev: Sampling,
}

impl Conditions {
    /// Every axis pinned at one explicit value.
    pub fn fixed(t_kelvin: f64, mu_ev: f64, omega_ev: f64) -> Self {
        Self {
            t_kelvin: Sampling::Fixed(t_kelvin),
            mu_ev: Sampling::Fixed(mu_ev),
            omega_ev: Sampling::Fixed(omega_ev),
        }
    }

    /// Resolve the sampled axis and the pinned values of the other two.
    ///
    /// A sampled series is packed into contiguous storage here: callers may
    /// hand in a strided `Array1`, while the energy-cut and simplex kernels
    /// index packed slices.
    pub(crate) fn resolve(&self) -> Result<ResolvedConditions> {
        self.t_kelvin.validate("t_kelvin")?;
        self.mu_ev.validate("mu_ev")?;
        self.omega_ev.validate("omega_ev")?;
        // Only the temperature axis is a temperature: a chemical potential and a
        // photon frequency are legitimately negative. Every axis already had its
        // finiteness checked above.
        for index in 0..self.t_kelvin.len() {
            validate_temperature(self.t_kelvin.value_at(index))?;
        }
        let mut sampled: Vec<(&'static str, ResponseAxis)> = Vec::new();
        if let Sampling::Values(values) = &self.t_kelvin {
            sampled.push(("t_kelvin", ResponseAxis::Temperature(pack(values))));
        }
        if let Sampling::Values(values) = &self.mu_ev {
            sampled.push(("mu_ev", ResponseAxis::ChemicalPotential(pack(values))));
        }
        if let Sampling::Values(values) = &self.omega_ev {
            sampled.push(("omega_ev", ResponseAxis::Frequency(pack(values))));
        }
        let axis = match sampled.len() {
            0 => ResponseAxis::Fixed,
            1 => sampled.pop().expect("one sample").1,
            _ => {
                return Err(TbError::InvalidResponseParameter {
                    parameter: sampled[1].0,
                    message: "at most one axis may be sampled; pin the others with Sampling::Fixed"
                        .into(),
                });
            }
        };
        Ok(ResolvedConditions {
            axis,
            t_kelvin: self.t_kelvin.value_at(0),
            mu_ev: self.mu_ev.value_at(0),
            omega_ev: self.omega_ev.value_at(0),
        })
    }
}

/// The axis a response result is indexed by.
#[derive(Clone, Debug, PartialEq)]
pub enum ResponseAxis {
    /// A single evaluation at the fixed conditions.
    Fixed,
    /// One entry per sampled temperature in kelvin.
    Temperature(Array1<f64>),
    /// One entry per sampled chemical potential in eV.
    ChemicalPotential(Array1<f64>),
    /// One entry per sampled frequency in eV.
    Frequency(Array1<f64>),
}

impl ResponseAxis {
    /// The sampled coordinates, or `None` for a single fixed evaluation.
    pub fn sampled_values(&self) -> Option<&Array1<f64>> {
        match self {
            Self::Fixed => None,
            Self::Temperature(values)
            | Self::ChemicalPotential(values)
            | Self::Frequency(values) => Some(values),
        }
    }

    /// Number of entries the corresponding result carries.
    pub fn len(&self) -> usize {
        self.sampled_values().map_or(1, |values| values.len())
    }

    /// `true` only for an empty sampled series, which validation rejects.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the result is a single evaluation at the fixed conditions.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }
}

/// Pack a possibly strided series into contiguous storage.
///
/// Energy-cut and simplex kernels index packed slices, so a caller-supplied
/// strided `Array1` must never reach them.
fn pack(values: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(values.iter().copied())
}

/// The sampled axis together with the pinned values of the other two axes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedConditions {
    /// The sampled axis; the result is indexed by it.
    pub axis: ResponseAxis,
    t_kelvin: f64,
    mu_ev: f64,
    omega_ev: f64,
}

impl ResolvedConditions {
    /// The three axis values at sample `index`.
    ///
    /// The stored scalar of the sampled axis is never read: each arm takes the
    /// sampled value from the axis itself.
    pub(crate) fn point(&self, index: usize) -> (f64, f64, f64) {
        match &self.axis {
            ResponseAxis::Fixed => (self.t_kelvin, self.mu_ev, self.omega_ev),
            ResponseAxis::Temperature(values) => (values[index], self.mu_ev, self.omega_ev),
            ResponseAxis::ChemicalPotential(values) => {
                (self.t_kelvin, values[index], self.omega_ev)
            }
            ResponseAxis::Frequency(values) => (self.t_kelvin, self.mu_ev, values[index]),
        }
    }

    /// Number of samples.
    pub(crate) fn len(&self) -> usize {
        self.axis.len()
    }

    /// The occupation selected by the temperature at sample `index`.
    pub(crate) fn occupation(&self, index: usize) -> Occupation {
        occupation_for(self.point(index).0)
    }

    /// The chemical potentials an energy-cut integrator should sweep: the
    /// sampled series, or the single pinned potential.
    pub(crate) fn chemical_potentials(&self) -> Array1<f64> {
        match &self.axis {
            ResponseAxis::ChemicalPotential(values) => values.clone(),
            _ => array![self.mu_ev],
        }
    }

    /// Require an explicitly fixed zero frequency for a DC response.
    ///
    /// Nonzero fixed frequencies and frequency scans would mislabel a static
    /// result as a frequency-dependent response. Both signed zeros are valid.
    pub(crate) fn require_dc(&self) -> Result<()> {
        if matches!(self.axis, ResponseAxis::Frequency(_)) || self.omega_ev != 0.0 {
            return Err(TbError::InvalidResponseParameter {
                parameter: "omega_ev",
                message: "this DC response requires omega_ev = Sampling::Fixed(0.0); sample t_kelvin or mu_ev instead".into(),
            });
        }
        Ok(())
    }

    /// Require one fixed DC state for a per-k-point Berry/geometry method.
    pub(crate) fn require_fixed_dc(&self) -> Result<()> {
        if self.axis.is_fixed() {
            return self.require_dc();
        }
        Err(TbError::InvalidResponseParameter {
            parameter: "conditions",
            message: "this method evaluates a single state; sample the axis with a Brillouin-zone entry point instead".into(),
        })
    }

    /// Every sample must have a finite Fermi-window peak. Direct Fermi-surface
    /// integration samples `-df/dE`, whose maximum is `0.25 / (k_B T)`.
    /// Reject the whole call if this overflows, including zero thermal width.
    pub(crate) fn require_positive_temperature(&self) -> Result<()> {
        for index in 0..self.len() {
            let width = self.occupation(index).energy_width()?;
            // Some subnormal widths still have a finite peak and remain valid.
            if !(0.25 / width).is_finite() {
                return Err(TbError::InvalidThermodynamicParameter {
                    parameter: "t_kelvin",
                    message: "direct Fermi-surface integration requires a positive thermal energy k_B T with a finite Fermi-window peak 0.25 / (k_B T) at every sample; use Integration::EnergyCut for the exact zero-temperature Fermi surface".into(),
                });
            }
        }
        Ok(())
    }
}

/// Fully specified Brillouin-zone input of a response calculation.
///
/// One structure is shared by `hall_conductivity`, `quantum_geometry`,
/// `optical_conductivity`, `optical_conductivity_tensor`,
/// `extrinsic_nonlinear_hall` and `intrinsic_nonlinear_hall`. It holds only
/// what those methods genuinely share: the thermodynamic conditions, the
/// k-mesh and the integration algorithm. Every field is required; no
/// constructor fills a physical choice silently.
///
/// Choices read by a single method are function parameters of that method —
/// the projection directions, `eta_ev`, `spin` and `field_symmetry` — so a
/// caller never states a value the method ignores.
#[derive(Clone, Debug)]
pub struct Parameters<const DIM: usize> {
    /// The three physical axes; at most one is sampled.
    pub conditions: Conditions,
    /// Number of uniform samples along each reciprocal-lattice direction.
    pub kmesh: [usize; DIM],
    /// Brillouin-zone integration algorithm. The accepted algorithms are
    /// stated by each entry point: Hall and both nonlinear Hall responses
    /// accept `Direct`/`EnergyCut`, while optical conductivity and quantum
    /// geometry accept `Direct`/`Simplex`.
    pub integration: Integration,
}

impl<const DIM: usize> Parameters<DIM> {
    /// Validate the fields every Brillouin-zone response reads and resolve the
    /// axes.
    ///
    /// Projection directions are method parameters and are validated by the
    /// entry point that receives them; this call validates the k-mesh and the
    /// thermodynamic conditions only.
    pub(crate) fn validate_grid_response(&self) -> Result<ResolvedConditions> {
        validate_k_mesh(&self.kmesh)?;
        self.conditions.resolve()
    }
}

/// Build an `Array2` direction matrix from const-generic row vectors.
pub(crate) fn direction_matrix<const N: usize, const DIM: usize>(
    rows: &[[f64; DIM]; N],
) -> Array2<f64> {
    let mut matrix = Array2::<f64>::zeros((N, DIM));
    for (index, row) in rows.iter().enumerate() {
        matrix.row_mut(index).assign(&ArrayView1::from(row));
    }
    matrix
}

/// Convert a temperature in kelvin into the internal occupation semantics.
pub(crate) fn occupation_for(t_kelvin: f64) -> Occupation {
    if t_kelvin <= 0.0 {
        Occupation::ZeroTemperature
    } else {
        Occupation::FermiDirac {
            temperature_kelvin: t_kelvin,
        }
    }
}

pub(crate) fn validate_temperature(temperature: f64) -> Result<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(TbError::InvalidResponseParameter {
            parameter: "t_kelvin",
            message: "must be finite and non-negative".into(),
        });
    }
    Ok(())
}

pub(crate) fn validate_direction_values<const N: usize, const DIM: usize>(
    directions: &[[f64; DIM]; N],
) -> Result<()> {
    for row in directions {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(TbError::InvalidResponseParameter {
                parameter: "direction",
                message: "all components must be finite".into(),
            });
        }
        if row.iter().all(|value| *value == 0.0) {
            return Err(TbError::InvalidResponseParameter {
                parameter: "direction",
                message: "each direction row must not be the zero vector".into(),
            });
        }
    }
    Ok(())
}

/// Diagnostics reported by simplex-based response integrations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegrationDiagnostics {
    /// Number of simplices whose minimum inter-band gap was below the safety
    /// threshold used by the interpolation algorithm.
    ///
    /// With a sampled axis this is the largest count over the samples: no
    /// single quadrature produced it.
    pub unsafe_simplex_count: usize,
}

pub(crate) fn validate_k_mesh<const DIM: usize>(k_mesh: &[usize; DIM]) -> Result<()> {
    if !(1..=3).contains(&DIM) {
        return Err(TbError::InvalidDimension {
            dim: DIM,
            supported: vec![1, 2, 3],
        });
    }
    if k_mesh.contains(&0) {
        return Err(TbError::InvalidKmeshDimensions(Array1::from_vec(
            k_mesh.to_vec(),
        )));
    }
    Ok(())
}

pub(crate) fn mesh_array<const DIM: usize>(k_mesh: &[usize; DIM]) -> Array1<usize> {
    Array1::from_vec(k_mesh.to_vec())
}

pub(crate) fn validate_broadening(broadening: f64) -> Result<()> {
    if !broadening.is_finite() || broadening < 0.0 {
        return Err(TbError::InvalidResponseParameter {
            parameter: "eta_ev",
            message: "must be finite and non-negative".into(),
        });
    }
    Ok(())
}

pub(crate) fn validate_sorted(values: &Array1<f64>, parameter: &'static str) -> Result<()> {
    if values
        .iter()
        .zip(values.iter().skip(1))
        .any(|(left, right)| left > right)
    {
        return Err(TbError::InvalidResponseParameter {
            parameter,
            message: "must be sorted in ascending order".into(),
        });
    }
    Ok(())
}

/// Test-only instrumentation for the shared-preparation contract.
///
/// The counters are thread-local on purpose. A contract test runs one whole
/// response call inside a single-threaded rayon pool, so every counted site
/// executes on that one thread and a concurrently running test cannot
/// contribute to the count. The counters stay active in release test builds,
/// where the numerical suite actually runs.
#[cfg(test)]
pub(crate) mod counters {
    use std::cell::Cell;

    thread_local! {
        static EIGEN: Cell<usize> = const { Cell::new(0) };
        static TRACKING: Cell<usize> = const { Cell::new(0) };
    }

    /// Record one band-structure diagonalization.
    pub(crate) fn count_eigen_decomposition() {
        EIGEN.with(|count| count.set(count.get() + 1));
    }

    /// Record one band-tracking pass.
    pub(crate) fn count_band_tracking() {
        TRACKING.with(|count| count.set(count.get() + 1));
    }

    /// Run `body` on one dedicated thread and report the work it performed as
    /// `(diagonalizations, band-tracking passes, body result)`.
    pub(crate) fn measure<R: Send>(body: impl FnOnce() -> R + Send) -> (usize, usize, R) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a single-threaded measurement pool");
        pool.install(|| {
            EIGEN.with(|count| count.set(0));
            TRACKING.with(|count| count.set(0));
            let value = body();
            (EIGEN.with(Cell::get), TRACKING.with(Cell::get), value)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn sorted_validation_handles_strided_arrays() {
        let descending = array![0.0, 1.0, 2.0].slice_move(s![..;-1]);
        assert!(validate_sorted(&descending, "values").is_err());
    }

    #[test]
    fn direction_components_must_be_finite_and_nonzero() {
        // Rank and width are compile-time properties now, so these two value
        // checks are the whole runtime direction validation.
        assert!(validate_direction_values(&[[1.0, 0.0], [0.0, 1.0]]).is_ok());
        for malformed in [
            [[f64::NAN, 0.0], [0.0, 1.0]],
            [[f64::INFINITY, 0.0], [0.0, 1.0]],
            [[f64::NEG_INFINITY, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ] {
            assert!(
                matches!(
                    validate_direction_values(&malformed),
                    Err(TbError::InvalidResponseParameter {
                        parameter: "direction",
                        ..
                    })
                ),
                "{malformed:?} must be rejected"
            );
        }
        // One dimension has only one axis, which is still a legal direction.
        assert!(validate_direction_values(&[[1.0], [1.0]]).is_ok());
    }

    #[test]
    fn parameter_structure_holds_only_shared_grid_input() {
        let parameters = Parameters::<2> {
            conditions: Conditions::fixed(300.0, 0.0, 0.0),
            kmesh: [4, 4],
            integration: Integration::Direct,
        };
        let resolved = parameters.validate_grid_response().unwrap();
        assert!(resolved.axis.is_fixed());
        assert_eq!(resolved.point(0), (300.0, 0.0, 0.0));

        let invalid = Parameters::<2> {
            conditions: Conditions::fixed(300.0, 0.0, 0.0),
            kmesh: [0, 4],
            integration: Integration::Direct,
        };
        assert!(matches!(
            invalid.validate_grid_response(),
            Err(TbError::InvalidKmeshDimensions(_))
        ));
    }

    #[test]
    fn fixed_conditions_resolve_to_a_single_point() {
        let resolved = Conditions::fixed(0.0, -1.25, 0.5).resolve().unwrap();
        assert!(resolved.axis.is_fixed());
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved.point(0), (0.0, -1.25, 0.5));
        assert!(resolved.axis.sampled_values().is_none());
        assert_eq!(resolved.chemical_potentials(), array![-1.25]);
        assert!(matches!(
            resolved.occupation(0),
            Occupation::ZeroTemperature
        ));
    }

    #[test]
    fn a_sampled_axis_keeps_the_other_two_fixed() {
        #[derive(Clone, Copy, PartialEq)]
        enum Axis {
            Temperature,
            ChemicalPotential,
            Frequency,
        }
        let cases = [
            (
                Conditions {
                    t_kelvin: Sampling::Values(array![10.0, 20.0]),
                    mu_ev: Sampling::Fixed(-0.5),
                    omega_ev: Sampling::Fixed(0.25),
                },
                Axis::Temperature,
                array![10.0, 20.0],
            ),
            (
                Conditions {
                    t_kelvin: Sampling::Fixed(300.0),
                    mu_ev: Sampling::Values(array![-1.0, 0.0, 1.0]),
                    omega_ev: Sampling::Fixed(0.25),
                },
                Axis::ChemicalPotential,
                array![-1.0, 0.0, 1.0],
            ),
            (
                Conditions {
                    t_kelvin: Sampling::Fixed(300.0),
                    mu_ev: Sampling::Fixed(-0.5),
                    omega_ev: Sampling::Values(array![0.0, 0.7, 2.0]),
                },
                Axis::Frequency,
                array![0.0, 0.7, 2.0],
            ),
        ];
        for (conditions, expected, values) in cases {
            let resolved = conditions.resolve().unwrap();
            assert_eq!(resolved.axis.sampled_values(), Some(&values));
            assert_eq!(resolved.len(), values.len());
            match expected {
                Axis::Temperature => assert!(matches!(resolved.axis, ResponseAxis::Temperature(_))),
                Axis::ChemicalPotential => {
                    assert!(matches!(resolved.axis, ResponseAxis::ChemicalPotential(_)))
                }
                Axis::Frequency => assert!(matches!(resolved.axis, ResponseAxis::Frequency(_))),
            }
            for (index, &value) in values.iter().enumerate() {
                let (t, mu, omega) = resolved.point(index);
                match expected {
                    Axis::Temperature => {
                        assert_eq!(t, value);
                        assert_eq!((mu, omega), (-0.5, 0.25));
                    }
                    Axis::ChemicalPotential => {
                        assert_eq!((t, omega), (300.0, 0.25));
                        assert_eq!(mu, value);
                    }
                    Axis::Frequency => {
                        assert_eq!((t, mu), (300.0, -0.5));
                        assert_eq!(omega, value);
                    }
                }
            }
            if expected != Axis::Temperature {
                assert!(matches!(
                    resolved.occupation(0),
                    Occupation::FermiDirac { temperature_kelvin } if temperature_kelvin == 300.0
                ));
            }
        }
    }

    #[test]
    fn a_sampled_series_is_packed_for_the_kernels() {
        // A strided series is legal input; the resolved axis must be
        // contiguous because the energy-cut and simplex kernels index slices.
        let strided = array![-1.0, 99.0, 0.0, 99.0, 1.0].slice_move(s![..;2]);
        assert!(strided.as_slice().is_none());
        let resolved = Conditions {
            t_kelvin: Sampling::Fixed(0.0),
            mu_ev: Sampling::Values(strided),
            omega_ev: Sampling::Fixed(0.0),
        }
        .resolve()
        .unwrap();
        let sampled = resolved.axis.sampled_values().unwrap();
        assert_eq!(sampled.as_slice().unwrap(), &[-1.0, 0.0, 1.0]);
        assert_eq!(
            resolved.chemical_potentials().as_slice().unwrap(),
            &[-1.0, 0.0, 1.0]
        );
        assert_eq!(resolved.point(2), (0.0, 1.0, 0.0));
    }

    #[test]
    fn sampling_two_axes_is_rejected() {
        let conditions = Conditions {
            t_kelvin: Sampling::Values(array![10.0, 20.0]),
            mu_ev: Sampling::Values(array![0.0, 1.0]),
            omega_ev: Sampling::Fixed(0.0),
        };
        assert!(matches!(
            conditions.resolve(),
            Err(TbError::InvalidResponseParameter {
                parameter: "mu_ev",
                ..
            })
        ));
    }

    #[test]
    fn malformed_axes_are_rejected() {
        for (conditions, parameter) in [
            (
                Conditions {
                    t_kelvin: Sampling::Fixed(f64::NAN),
                    mu_ev: Sampling::Fixed(0.0),
                    omega_ev: Sampling::Fixed(0.0),
                },
                "t_kelvin",
            ),
            (
                Conditions {
                    t_kelvin: Sampling::Fixed(0.0),
                    mu_ev: Sampling::Values(Array1::from_vec(Vec::new())),
                    omega_ev: Sampling::Fixed(0.0),
                },
                "mu_ev",
            ),
            (
                Conditions {
                    t_kelvin: Sampling::Fixed(0.0),
                    mu_ev: Sampling::Fixed(0.0),
                    omega_ev: Sampling::Values(array![0.0, f64::INFINITY]),
                },
                "omega_ev",
            ),
        ] {
            assert!(matches!(
                conditions.resolve(),
                Err(TbError::InvalidResponseParameter {
                    parameter: found,
                    ..
                }) if found == parameter
            ));
        }
    }

    #[test]
    fn invalid_temperatures_are_rejected_for_fixed_and_sampled_axes() {
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let fixed = Parameters::<2> {
                conditions: Conditions::fixed(invalid, 0.0, 0.0),
                kmesh: [4, 4],
                integration: Integration::Direct,
            };
            let sampled = Parameters::<2> {
                conditions: Conditions {
                    t_kelvin: Sampling::Values(array![50.0, invalid]),
                    mu_ev: Sampling::Fixed(0.0),
                    omega_ev: Sampling::Fixed(0.0),
                },
                kmesh: [4, 4],
                integration: Integration::EnergyCut,
            };
            for rejected in [
                fixed.validate_grid_response(),
                sampled.validate_grid_response(),
            ] {
                assert!(matches!(
                    rejected,
                    Err(TbError::InvalidResponseParameter {
                        parameter: "t_kelvin",
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn broadening_is_validated_where_a_method_receives_it() {
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                validate_broadening(invalid),
                Err(TbError::InvalidResponseParameter {
                    parameter: "eta_ev",
                    ..
                })
            ));
        }
        assert!(validate_broadening(0.0).is_ok());
        assert!(validate_broadening(2.5e-2).is_ok());
    }

    #[test]
    fn per_state_methods_reject_a_sampled_axis() {
        let sampled = Conditions {
            t_kelvin: Sampling::Values(array![10.0, 20.0]),
            mu_ev: Sampling::Fixed(0.0),
            omega_ev: Sampling::Fixed(0.0),
        }
        .resolve()
        .unwrap();
        assert!(sampled.require_fixed_dc().is_err());

        let fixed = Conditions::fixed(300.0, 0.0, 0.0).resolve().unwrap();
        assert!(fixed.require_fixed_dc().is_ok());
        assert!(fixed.require_positive_temperature().is_ok());
        assert!(
            Conditions::fixed(0.0, 0.0, 0.0)
                .resolve()
                .unwrap()
                .require_positive_temperature()
                .is_err()
        );
    }
}
