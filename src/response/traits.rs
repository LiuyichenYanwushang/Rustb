//! Reusable band-resolved Berry-curvature interface.

use ndarray::prelude::*;
use ndarray::{ArrayBase, Data};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::math::anti_comm;
use crate::velocity::Velocity;
use crate::{Gauge, Model, RMatrixData};

use super::config::{
    Parameters, parameters_occupation, validate_broadening, validate_chemical_potentials,
    validate_direction_matrix, validate_temperature,
};
use super::helpers::build_spin_matrix;

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
    ///
    /// Reads `direction` (rank 2), `spin` and `eta` from the parameter set;
    /// all other fields are ignored.
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<BandBerryCurvature>;

    /// Sum band Berry curvatures with the electronic occupation selected by
    /// `params.T` at the chemical potential `params.mu[0]`.
    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<f64>;

    /// Evaluate occupation-weighted Berry curvature on multiple k-points.
    fn occupied_berry_curvature_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        params: &Parameters<DIM>,
    ) -> Result<Array1<f64>>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> BerryCurvature<DIM>
    for Model<SPIN, DIM, R>
{
    fn berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<BandBerryCurvature> {
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        validate_direction_matrix(&params.direction, 2, DIM)?;
        validate_broadening(params.eta)?;
        let spin = params.spin;
        if !SPIN && let Some(direction) = spin {
            return Err(TbError::SpinNotAllowed(direction));
        }
        self.berry_curvature_at_impl(k, params)
    }

    fn occupied_berry_curvature_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<f64> {
        validate_chemical_potentials(&params.mu)?;
        if params.mu.len() != 1 {
            return Err(TbError::InvalidResponseParameter {
                parameter: "mu",
                message: "occupied_berry_curvature_at expects a single chemical potential".into(),
            });
        }
        validate_temperature(&params.T)?;
        let occupation = parameters_occupation(params);
        let chemical_potential = params.mu[0];
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
        params: &Parameters<DIM>,
    ) -> Result<Array1<f64>> {
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "Berry-curvature k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        // Validate once up front, then reuse the unvalidated kernel per k-point.
        validate_chemical_potentials(&params.mu)?;
        if params.mu.len() != 1 {
            return Err(TbError::InvalidResponseParameter {
                parameter: "mu",
                message: "occupied_berry_curvature_on expects a single chemical potential".into(),
            });
        }
        validate_temperature(&params.T)?;
        validate_direction_matrix(&params.direction, 2, DIM)?;
        validate_broadening(params.eta)?;
        if !SPIN && let Some(direction) = params.spin {
            return Err(TbError::SpinNotAllowed(direction));
        }
        let occupation = parameters_occupation(params);
        let chemical_potential = params.mu[0];
        let values: Vec<Result<f64>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| {
                let bands = self.berry_curvature_at_impl(&k, params)?;
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
    /// Callers must have already validated `direction` (rank 2), `spin`
    /// against the model and `eta`. The high-level entry points validate
    /// once and then call this per k-point; the public trait methods are
    /// independent boundaries and validate before delegating here.
    pub(crate) fn berry_curvature_at_impl<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<BandBerryCurvature> {
        let spin = params.spin;
        let (projected_velocity, hamiltonian) =
            self.gen_v_projected(k, Gauge::Atom, &params.direction);
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
        let eta_squared = params.eta * params.eta;
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
