//! Shared configuration types for response calculations.

use ndarray::Array1;

use crate::SpinDirection;
use crate::error::{Result, TbError};

/// Ordered pair of real-space directions defining a rank-two response.
///
/// The const-generic array representation makes a direction with the wrong
/// spatial dimension unrepresentable at the public high-level API.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionPair<const DIM: usize> {
    /// First tensor index.
    pub first: [f64; DIM],
    /// Second tensor index.
    pub second: [f64; DIM],
}

impl<const DIM: usize> DirectionPair<DIM> {
    pub const fn new(first: [f64; DIM], second: [f64; DIM]) -> Self {
        Self { first, second }
    }

    /// Construct a pair of Cartesian unit vectors.
    pub fn cartesian(first: usize, second: usize) -> Result<Self> {
        if first >= DIM {
            return Err(TbError::InvalidDirectionIndex {
                index: first,
                dim: DIM,
            });
        }
        if second >= DIM {
            return Err(TbError::InvalidDirectionIndex {
                index: second,
                dim: DIM,
            });
        }
        let mut a = [0.0; DIM];
        let mut b = [0.0; DIM];
        a[first] = 1.0;
        b[second] = 1.0;
        Ok(Self::new(a, b))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_direction("first_direction", &self.first)?;
        validate_direction("second_direction", &self.second)
    }

    pub(crate) fn as_arrays(&self) -> (Array1<f64>, Array1<f64>) {
        (
            Array1::from_vec(self.first.to_vec()),
            Array1::from_vec(self.second.to_vec()),
        )
    }
}

/// Operator carried by the current vertex of a response tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurrentOperator {
    /// Charge current.
    #[default]
    Charge,
    /// Spin current polarized along the selected spin direction.
    Spin(SpinDirection),
}

impl CurrentOperator {
    pub(crate) const fn spin_direction(self) -> Option<SpinDirection> {
        match self {
            Self::Charge => None,
            Self::Spin(direction) => Some(direction),
        }
    }
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

fn validate_direction<const DIM: usize>(
    parameter: &'static str,
    direction: &[f64; DIM],
) -> Result<()> {
    if direction.iter().any(|value| !value.is_finite()) {
        return Err(TbError::InvalidResponseParameter {
            parameter,
            message: "all components must be finite".into(),
        });
    }
    if direction.iter().all(|value| *value == 0.0) {
        return Err(TbError::InvalidResponseParameter {
            parameter,
            message: "must not be the zero vector".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn cartesian_pair_is_dimension_checked() {
        assert_eq!(
            DirectionPair::<2>::cartesian(0, 1).unwrap(),
            DirectionPair::new([1.0, 0.0], [0.0, 1.0])
        );
        assert!(DirectionPair::<2>::cartesian(2, 0).is_err());
    }

    #[test]
    fn sorted_validation_handles_strided_arrays() {
        let descending = array![0.0, 1.0, 2.0].slice_move(s![..;-1]);
        assert!(validate_sorted(&descending, "values").is_err());
    }
}
