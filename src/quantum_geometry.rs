//! Occupation-weighted quantum metric and Berry curvature.
//!
//! For band `n`, the quantum geometric tensor is
//!
//! ```math
//! G^{ab}_n(k) = \sum_{m\ne n}
//! \frac{v^a_{nm}(k)v^b_{mn}(k)}{(E_n-E_m)^2+\eta^2}
//! = g^{ab}_n(k)-\frac{i}{2}\Omega^{ab}_n(k).
//! ```
//!
//! [`QuantumGeometry`] exposes reusable band-resolved kernels. The high-level
//! [`Model::quantum_geometry`] method performs the Brillouin-zone integration
//! from one [`Parameters`] value.

use ndarray::Data;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::response::config::{
    Integration, IntegrationDiagnostics, Parameters, mesh_array, parameters_occupation,
};
use crate::response::linear::integrate_occupied_geometry;
use crate::response::{VertexKernel, global_band_track};
use crate::velocity::Velocity;
use crate::{Gauge, Model, RMatrixData};

/// Band-resolved quantum geometry at one k-point.
#[derive(Clone, Debug, PartialEq)]
pub struct BandQuantumGeometry {
    /// Quantum metric of every band.
    pub metric: Array1<f64>,
    /// Berry curvature of every band.
    pub berry_curvature: Array1<f64>,
    /// Band energies in eV.
    pub energies: Array1<f64>,
}

/// Band-resolved quantum geometry on a list of k-points.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGeometryMap {
    /// Shape `(number_of_k_points, number_of_states)`.
    pub metric: Array2<f64>,
    /// Shape `(number_of_k_points, number_of_states)`.
    pub berry_curvature: Array2<f64>,
    /// Shape `(number_of_k_points, number_of_states)`.
    pub energies: Array2<f64>,
}

/// Occupation-weighted Brillouin-zone quantum geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGeometryResult {
    /// Chemical potentials copied from the input configuration.
    pub chemical_potentials: Array1<f64>,
    /// Occupation-weighted quantum metric at every chemical potential.
    pub metric: Array1<f64>,
    /// Occupation-weighted Berry curvature at every chemical potential.
    pub berry_curvature: Array1<f64>,
    /// Present only for simplex integration.
    pub diagnostics: Option<IntegrationDiagnostics>,
}

/// Reusable band-resolved quantum-geometry kernels.
pub trait QuantumGeometry<const DIM: usize>: Velocity {
    /// Evaluate every band at one k-point.
    ///
    /// Reads `direction` (rank 2) and `eta` from the parameter set; all
    /// other fields are ignored.
    fn quantum_geometry_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<BandQuantumGeometry>;

    /// Evaluate every band on a list of k-points in parallel.
    fn quantum_geometry_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        params: &Parameters<DIM>,
    ) -> Result<QuantumGeometryMap>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> QuantumGeometry<DIM>
    for Model<SPIN, DIM, R>
{
    fn quantum_geometry_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<BandQuantumGeometry> {
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        crate::response::config::validate_direction_matrix(&params.direction, 2, DIM)?;
        crate::response::config::validate_broadening(params.eta)?;
        self.quantum_geometry_at_impl(k, params)
    }

    fn quantum_geometry_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        params: &Parameters<DIM>,
    ) -> Result<QuantumGeometryMap> {
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "quantum geometry k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        // Validate once up front, then reuse the unvalidated kernel per k-point.
        crate::response::config::validate_direction_matrix(&params.direction, 2, DIM)?;
        crate::response::config::validate_broadening(params.eta)?;
        let rows: Vec<Result<BandQuantumGeometry>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.quantum_geometry_at_impl(&k, params))
            .collect();
        let rows: Vec<BandQuantumGeometry> = rows.into_iter().collect::<Result<_>>()?;
        let number_of_k_points = rows.len();
        let mut metric = Array2::<f64>::zeros((number_of_k_points, self.nsta()));
        let mut berry_curvature = Array2::<f64>::zeros((number_of_k_points, self.nsta()));
        let mut energies = Array2::<f64>::zeros((number_of_k_points, self.nsta()));
        for (index, row) in rows.into_iter().enumerate() {
            metric.row_mut(index).assign(&row.metric);
            berry_curvature.row_mut(index).assign(&row.berry_curvature);
            energies.row_mut(index).assign(&row.energies);
        }
        Ok(QuantumGeometryMap {
            metric,
            berry_curvature,
            energies,
        })
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Band-resolved quantum-geometry kernel without input validation.
    ///
    /// Callers must have already validated `direction` (rank 2) and `eta`.
    /// The high-level entry point and `quantum_geometry_on` validate once and
    /// then reuse this per k-point; the public trait method is an independent
    /// boundary and validates before delegating here.
    pub(crate) fn quantum_geometry_at_impl<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        params: &Parameters<DIM>,
    ) -> Result<BandQuantumGeometry> {
        let (projected_velocity, hamiltonian) =
            self.gen_v_projected(k, Gauge::Atom, &params.direction);
        let (energies, eigenvectors) = hamiltonian.eigh(UPLO::Lower)?;
        let bra = eigenvectors.t();
        let ket = eigenvectors.mapv(|value| value.conj());

        let velocity_a = projected_velocity.index_axis(Axis(0), 0);
        let velocity_b = projected_velocity.index_axis(Axis(0), 1);
        let a_band = bra.dot(&velocity_a.dot(&ket));
        let b_band = bra.dot(&velocity_b.dot(&ket));
        let kernel = a_band * b_band.reversed_axes();
        let eta_squared = params.eta * params.eta;
        let mut metric = Array1::<f64>::zeros(self.nsta());
        let mut berry_curvature = Array1::<f64>::zeros(self.nsta());

        for band in 0..self.nsta() {
            let mut tensor = Complex::new(0.0, 0.0);
            for other in 0..self.nsta() {
                if band == other {
                    continue;
                }
                let difference = energies[band] - energies[other];
                tensor += kernel[[band, other]] / (difference * difference + eta_squared);
            }
            metric[band] = tensor.re;
            berry_curvature[band] = -2.0 * tensor.im;
        }

        Ok(BandQuantumGeometry {
            metric,
            berry_curvature,
            energies,
        })
    }

    /// Integrate occupation-weighted quantum geometry over the Brillouin zone.
    ///
    /// Reads `kmesh`, `direction` (rank 2), `mu`, `T` and `eta` from the
    /// parameter set; `omega`, `spin` and `field_symmetry` are ignored. Both
    /// algorithms return the same named result and use Cartesian
    /// reciprocal-space normalization. Simplex mode additionally reports the
    /// number of small-gap simplices encountered during band tracking.
    pub fn quantum_geometry(&self, params: &Parameters<DIM>) -> Result<QuantumGeometryResult> {
        params.validate_rank2()?;
        if params.integration == Integration::EnergyCut {
            return Err(TbError::InvalidResponseParameter {
                parameter: "integration",
                message: "quantum_geometry supports Integration::Direct or Simplex, not EnergyCut"
                    .into(),
            });
        }
        if params.integration == Integration::Simplex && DIM == 1 {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![2, 3],
            });
        }
        let k_mesh = mesh_array(&params.kmesh);
        let determinant = self.lat.det()?;
        let occupation = parameters_occupation(params);

        let (metric, berry_curvature, diagnostics) = match params.integration {
            Integration::Direct => {
                let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
                let geometry = self.quantum_geometry_on(&k_points, params)?;
                let normalization = 1.0 / k_points.nrows() as f64 / determinant;
                let values: Vec<(f64, f64)> = params
                    .mu
                    .par_iter()
                    .map(|&mu| {
                        let mut metric_sum = 0.0;
                        let mut berry_sum = 0.0;
                        for k in 0..k_points.nrows() {
                            for band in 0..self.nsta() {
                                let occ =
                                    occupation.value_unchecked(geometry.energies[[k, band]], mu);
                                metric_sum += geometry.metric[[k, band]] * occ;
                                berry_sum += geometry.berry_curvature[[k, band]] * occ;
                            }
                        }
                        (metric_sum * normalization, berry_sum * normalization)
                    })
                    .collect();
                let (metric, berry): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                (Array1::from_vec(metric), Array1::from_vec(berry), None)
            }
            Integration::Simplex => {
                let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
                let direction_a = params.direction.row(0).to_owned();
                let direction_b = params.direction.row(1).to_owned();
                let vertices: Vec<Result<VertexKernel>> = (0..k_points.nrows())
                    .into_par_iter()
                    .map(|index| {
                        self.compute_velocity_kernel(
                            &k_points.row(index).to_owned(),
                            &direction_a,
                            &direction_b,
                            None,
                            Gauge::Atom,
                            None,
                        )
                    })
                    .collect();
                let mut vertices: Vec<VertexKernel> =
                    vertices.into_iter().collect::<Result<_>>()?;
                global_band_track(&mut vertices, &params.kmesh);
                let (metric, berry, unsafe_simplex_count) = integrate_occupied_geometry(
                    &vertices, &k_mesh, params.eta, &params.mu, occupation,
                );
                (
                    metric / determinant,
                    berry / determinant,
                    Some(IntegrationDiagnostics {
                        unsafe_simplex_count,
                    }),
                )
            }
            Integration::EnergyCut => unreachable!("rejected during validation"),
        };

        Ok(QuantumGeometryResult {
            chemical_potentials: params.mu.clone(),
            metric,
            berry_curvature,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn massive_dirac_model() -> Model<false, 2> {
        let mut model = Model::<false, 2>::tb_model(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
            None,
        )
        .unwrap();
        model.set_onsite(&array![-0.5, 0.5], None);
        model
    }

    fn qwz_model(mass: f64) -> Model<false, 2> {
        let mut model = Model::<false, 2>::tb_model(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
            None,
        )
        .unwrap();
        model.add_onsite(&array![mass, -mass], None);
        model.add_hop(Complex::new(0.0, -0.5), 0, 1, &array![1, 0], None);
        model.add_hop(Complex::new(0.0, 0.5), 0, 1, &array![-1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);
        model.add_hop(0.5, 0, 1, &array![0, -1], None);
        for displacement in [array![1, 0], array![-1, 0], array![0, 1], array![0, -1]] {
            model.add_hop(0.5, 0, 0, &displacement, None);
            model.add_hop(-0.5, 1, 1, &displacement, None);
        }
        model
    }

    #[test]
    fn named_band_result_has_real_components() {
        let model = massive_dirac_model();
        let params = Parameters::rank2([1, 1], [1.0, 0.0], [0.0, 1.0], array![0.0]);
        let geometry = model
            .quantum_geometry_at(&array![0.0, 0.0], &params)
            .unwrap();
        assert_eq!(geometry.metric.len(), model.nsta());
        assert_eq!(geometry.berry_curvature.len(), model.nsta());
        assert_eq!(geometry.energies.len(), model.nsta());
    }

    #[test]
    fn direct_and_simplex_integrate_the_same_geometry() {
        let model = qwz_model(-1.0);
        let mut params = Parameters::rank2([31, 31], [1.0, 0.0], [0.0, 1.0], array![0.0]);
        params.eta = 0.1;
        let direct = model.quantum_geometry(&params).unwrap();

        params.integration = Integration::Simplex;
        let simplex = model.quantum_geometry(&params).unwrap();
        assert!(simplex.diagnostics.is_some());
        assert!((direct.metric[0] - simplex.metric[0]).abs() < 5e-3);
        assert!((direct.berry_curvature[0] - simplex.berry_curvature[0]).abs() < 5e-3);
    }
}
