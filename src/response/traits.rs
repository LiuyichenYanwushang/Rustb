//! Reusable band-resolved Berry-curvature interface.

use ndarray::prelude::*;
use ndarray::{ArrayBase, Data};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::math::anti_comm;
use crate::thermodynamics::Occupation;
use crate::velocity::Velocity;
use crate::{Gauge, Model, RMatrixData};

use super::config::{CurrentOperator, DirectionPair, validate_broadening};
use super::helpers::build_spin_matrix;

/// Configuration of the band-resolved Berry-curvature kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BerryCurvatureParams<const DIM: usize> {
    /// Ordered tensor directions `(current, velocity)`.
    pub directions: DirectionPair<DIM>,
    /// Charge current or spin-current polarization at the first vertex.
    pub current: CurrentOperator,
    /// Non-negative energy-denominator regularization in eV.
    pub broadening: f64,
}

impl<const DIM: usize> BerryCurvatureParams<DIM> {
    /// Construct a charge Berry-curvature calculation with `1e-3` eV
    /// denominator broadening.
    pub const fn new(directions: DirectionPair<DIM>) -> Self {
        Self {
            directions,
            current: CurrentOperator::Charge,
            broadening: 1e-3,
        }
    }

    fn validate(&self) -> Result<()> {
        self.directions.validate()?;
        validate_broadening(self.broadening)
    }
}

/// Berry curvature and energies of every band at one k-point.
#[derive(Clone, Debug, PartialEq)]
pub struct BandBerryCurvature {
    /// Berry curvature of every band.
    pub berry_curvature: Array1<f64>,
    /// Band energies in eV.
    pub energies: Array1<f64>,
}

/// Berry-curvature methods shared by tight-binding-like model types.
pub trait BerryCurvature<const DIM: usize>: Velocity {
    /// Evaluate the charge or spin Berry curvature of every band at one k-point.
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &BerryCurvatureParams<DIM>,
    ) -> Result<BandBerryCurvature>;

    /// Sum band Berry curvatures with the requested electronic occupation.
    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &BerryCurvatureParams<DIM>,
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<f64>;

    /// Evaluate occupation-weighted Berry curvature on multiple k-points.
    fn occupied_berry_curvature_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        params: &BerryCurvatureParams<DIM>,
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<Array1<f64>>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> BerryCurvature<DIM>
    for Model<SPIN, DIM, R>
{
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &BerryCurvatureParams<DIM>,
    ) -> Result<BandBerryCurvature> {
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        params.validate()?;
        let spin = params.current.spin_direction();
        if !SPIN && let Some(direction) = spin {
            return Err(TbError::SpinNotAllowed(direction));
        }

        let mut directions = Array2::<f64>::zeros((2, DIM));
        directions
            .row_mut(0)
            .assign(&ArrayView1::from(&params.directions.first));
        directions
            .row_mut(1)
            .assign(&ArrayView1::from(&params.directions.second));
        let (projected_velocity, hamiltonian) = self.gen_v_projected(k, Gauge::Atom, &directions);
        let (energies, eigenvectors) = hamiltonian.eigh(UPLO::Lower)?;

        let current: Array2<Complex<f64>> = if SPIN && spin.is_some() {
            let spin_matrix = build_spin_matrix(self.norb(), spin);
            anti_comm(&spin_matrix, &projected_velocity.index_axis(Axis(0), 0)) * 0.5
        } else {
            projected_velocity.index_axis(Axis(0), 0).to_owned()
        };
        let second_velocity = projected_velocity.index_axis(Axis(0), 1);
        let bra = eigenvectors.t();
        let ket = eigenvectors.mapv(|value| value.conj());
        let current_band = bra.dot(&current.dot(&ket));
        let velocity_band = bra.dot(&second_velocity.dot(&ket));
        let kernel = current_band * velocity_band.reversed_axes();
        let eta_squared = params.broadening * params.broadening;
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

    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &BerryCurvatureParams<DIM>,
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<f64> {
        if !chemical_potential.is_finite() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "chemical_potential",
                message: "must be finite".into(),
            });
        }
        occupation.validate()?;
        let bands = self.berry_curvature_at(k, params)?;
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
        params: &BerryCurvatureParams<DIM>,
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<Array1<f64>> {
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "Berry-curvature k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        let values: Vec<Result<f64>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.occupied_berry_curvature_at(&k, params, chemical_potential, occupation))
            .collect();
        Ok(Array1::from_vec(
            values.into_iter().collect::<Result<Vec<_>>>()?,
        ))
    }
}
