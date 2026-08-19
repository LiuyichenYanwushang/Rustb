//! Shared configuration types for response calculations.

use ndarray::array;
use ndarray::{Array1, Array2, ArrayView1};

use crate::SpinDirection;
use crate::error::{Result, TbError};
use crate::thermodynamics::Occupation;

/// Brillouin-zone integration algorithm shared by all response entry points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Integration {
    /// Uniform k-mesh summation.
    #[default]
    Direct,
    /// Band-tracked simplex quadrature (quantum geometry, optical conductivity).
    Simplex,
    /// Band-tracked energy-cut integration (Hall, nonlinear Hall).
    EnergyCut,
}

/// Whether the two external-field indices of the extrinsic nonlinear Hall
/// response are explicitly symmetrized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FieldSymmetry {
    /// Return the ordered kernel `S[current, field_1; field_2]`.
    Ordered,
    /// Return `(S[current, field_1; field_2] +
    /// S[current, field_2; field_1]) / 2`.
    #[default]
    Symmetrized,
}

/// Unified physical parameters for every response calculation.
///
/// One structure is shared by `hall_conductivity`, `quantum_geometry`,
/// `optical_conductivity`, `extrinsic_nonlinear_hall` and
/// `intrinsic_nonlinear_hall`. Each method reads only the fields it needs;
/// the remaining fields are silently ignored.
#[derive(Clone, Debug)]
pub struct Parameters<const DIM: usize> {
    /// Temperature in kelvin. `T[0] == 0.0` selects the exact zero-temperature
    /// step function; `T[0] > 0.0` selects Fermi-Dirac occupation at that
    /// temperature. Only `T[0]` is currently used.
    #[allow(non_snake_case)]
    pub T: Array1<f64>,
    /// Chemical potential(s) in eV. Methods that expect a single chemical
    /// potential (optical conductivity) use `mu[0]` and require one element.
    pub mu: Array1<f64>,
    /// Non-negative energy-denominator regularization in eV.
    pub eta: f64,
    /// Number of uniform samples along each reciprocal-lattice direction.
    pub kmesh: [usize; DIM],
    /// Photon / perturbation frequency(ies) in eV. Only `optical_conductivity`
    /// scans all supplied frequencies; DC response methods ignore this field.
    pub omega: Array1<f64>,
    /// Spin-current polarization. `None` selects the charge current.
    pub spin: Option<SpinDirection>,
    /// Direction matrix with shape `(rank, DIM)`:
    /// - `rank = 2` for rank-two tensors (Hall, quantum geometry, optical
    ///   component): row 0 = first tensor index, row 1 = second tensor index.
    /// - `rank = 3` for rank-three tensors (nonlinear Hall):
    ///   row 0 = current direction, rows 1-2 = field directions.
    pub direction: Array2<f64>,
    /// Brillouin-zone integration algorithm.
    pub integration: Integration,
    /// Ordering convention for the two field indices of the extrinsic
    /// nonlinear Hall response. Ignored by all other methods.
    pub field_symmetry: FieldSymmetry,
}

impl<const DIM: usize> Parameters<DIM> {
    /// Construct a zero-temperature charge-response parameter set with
    /// `1e-3` eV broadening, DC frequency and direct integration.
    pub fn new(kmesh: [usize; DIM], direction: Array2<f64>, mu: Array1<f64>) -> Self {
        Self {
            T: array![0.0],
            mu,
            eta: 1e-3,
            kmesh,
            omega: array![0.0],
            spin: None,
            direction,
            integration: Integration::Direct,
            field_symmetry: FieldSymmetry::Symmetrized,
        }
    }

    /// Convenience constructor for a single chemical potential.
    pub fn at_mu(kmesh: [usize; DIM], direction: Array2<f64>, mu: f64) -> Self {
        Self::new(kmesh, direction, array![mu])
    }

    /// Rank-two charge response from two const-generic direction vectors.
    pub fn rank2(
        kmesh: [usize; DIM],
        direction_a: [f64; DIM],
        direction_b: [f64; DIM],
        mu: Array1<f64>,
    ) -> Self {
        Self::new(kmesh, direction_matrix(&[direction_a, direction_b]), mu)
    }

    /// Rank-three charge response from const-generic current and field
    /// vectors. Row 0 is the current, rows 1-2 the fields.
    pub fn rank3(
        kmesh: [usize; DIM],
        current: [f64; DIM],
        field_1: [f64; DIM],
        field_2: [f64; DIM],
        mu: Array1<f64>,
    ) -> Self {
        Self::new(kmesh, direction_matrix(&[current, field_1, field_2]), mu)
    }

    /// Set the temperature in kelvin.
    pub fn with_temperature(mut self, kelvin: f64) -> Self {
        self.T = array![kelvin];
        self
    }

    /// Select the spin-current polarization.
    pub fn with_spin(mut self, spin: SpinDirection) -> Self {
        self.spin = Some(spin);
        self
    }

    /// Set a single frequency in eV.
    pub fn with_frequency(mut self, omega: f64) -> Self {
        self.omega = array![omega];
        self
    }

    /// Select the Brillouin-zone integration algorithm.
    pub fn with_integration(mut self, integration: Integration) -> Self {
        self.integration = integration;
        self
    }

    /// Validate the fields read by every rank-two response method.
    pub(crate) fn validate_rank2(&self) -> Result<()> {
        validate_k_mesh(&self.kmesh)?;
        validate_direction_matrix(&self.direction, 2, DIM)?;
        validate_chemical_potentials(&self.mu)?;
        validate_broadening(self.eta)?;
        validate_temperature(&self.T)
    }

    /// Validate the fields read by every rank-three response method.
    ///
    /// `eta` is intentionally not validated here: the intrinsic nonlinear
    /// Hall response never reads it.
    pub(crate) fn validate_rank3(&self) -> Result<()> {
        validate_k_mesh(&self.kmesh)?;
        validate_direction_matrix(&self.direction, 3, DIM)?;
        validate_chemical_potentials(&self.mu)?;
        validate_temperature(&self.T)
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

/// Convert the temperature convention into the internal occupation semantics.
pub(crate) fn parameters_occupation<const DIM: usize>(params: &Parameters<DIM>) -> Occupation {
    if params.T[0] <= 0.0 {
        Occupation::ZeroTemperature
    } else {
        Occupation::FermiDirac {
            temperature_kelvin: params.T[0],
        }
    }
}

pub(crate) fn validate_temperature(values: &Array1<f64>) -> Result<()> {
    if values.is_empty() {
        return Err(TbError::InvalidResponseParameter {
            parameter: "T",
            message: "must contain at least one value".into(),
        });
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(TbError::InvalidResponseParameter {
            parameter: "T",
            message: "all values must be finite and non-negative".into(),
        });
    }
    Ok(())
}

pub(crate) fn validate_direction_matrix(
    direction: &Array2<f64>,
    rank: usize,
    dim: usize,
) -> Result<()> {
    if direction.nrows() != rank {
        return Err(TbError::DimensionMismatch {
            context: "direction".into(),
            expected: rank,
            found: direction.nrows(),
        });
    }
    if direction.ncols() != dim {
        return Err(TbError::DimensionMismatch {
            context: "direction".into(),
            expected: dim,
            found: direction.ncols(),
        });
    }
    for row in direction.rows() {
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
            parameter: "broadening",
            message: "must be finite and non-negative".into(),
        });
    }
    Ok(())
}

pub(crate) fn validate_chemical_potentials(values: &Array1<f64>) -> Result<()> {
    if values.is_empty() {
        return Err(TbError::InvalidResponseParameter {
            parameter: "chemical_potentials",
            message: "must contain at least one value".into(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(TbError::InvalidResponseParameter {
            parameter: "chemical_potentials",
            message: "all values must be finite".into(),
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
    fn direction_matrix_builds_rows_and_rank3_validation() {
        let matrix = direction_matrix::<3, 2>(&[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        assert_eq!(matrix.dim(), (3, 2));
        assert!(validate_direction_matrix(&matrix, 3, 2).is_ok());
        assert!(validate_direction_matrix(&matrix, 2, 2).is_err());
    }

    #[test]
    fn parameters_rank2_rank3_construct_direction_rows() {
        let rank2 = Parameters::rank2([4, 4], [1.0, 0.0], [0.0, 1.0], array![0.0]);
        assert_eq!(rank2.direction.dim(), (2, 2));
        assert!(rank2.validate_rank2().is_ok());

        let rank3 = Parameters::rank3([4, 4], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], array![0.0]);
        assert_eq!(rank3.direction.dim(), (3, 2));
        assert!(rank3.validate_rank3().is_ok());
        // A rank-2 validation must reject the rank-3 direction matrix.
        assert!(rank3.validate_rank2().is_err());
    }
}
