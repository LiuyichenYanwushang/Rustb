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
//! from one [`QuantumGeometryParams`] value.

use ndarray::Data;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::response::config::{
    DirectionPair, IntegrationDiagnostics, mesh_array, validate_broadening,
    validate_chemical_potentials, validate_k_mesh,
};
use crate::response::linear::integrate_occupied_geometry;
use crate::response::{VertexKernel, global_band_track};
use crate::thermodynamics::Occupation;
use crate::velocity::Velocity;
use crate::{Gauge, Model, RMatrixData};

/// Integration algorithm for occupation-weighted quantum geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantumGeometryIntegration {
    /// Evaluate the band-resolved tensor on every point of a uniform mesh.
    #[default]
    Direct,
    /// Interpolate gauge-invariant velocity kernels inside triangles or
    /// tetrahedra and use symmetric simplex quadrature.
    Simplex,
}

/// Configuration for a Brillouin-zone quantum-geometry calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantumGeometryParams<const DIM: usize> {
    /// Number of uniform samples along each reciprocal-lattice direction.
    pub k_mesh: [usize; DIM],
    /// Ordered directions of the quantum geometric tensor.
    pub directions: DirectionPair<DIM>,
    /// Chemical potentials in eV.
    pub chemical_potentials: Array1<f64>,
    /// Electronic occupation or smearing convention.
    pub occupation: Occupation,
    /// Non-negative energy-denominator regularization in eV.
    pub broadening: f64,
    /// Brillouin-zone integration algorithm.
    pub integration: QuantumGeometryIntegration,
}

impl<const DIM: usize> QuantumGeometryParams<DIM> {
    /// Construct a zero-temperature direct-sum calculation.
    pub fn new(
        k_mesh: [usize; DIM],
        directions: DirectionPair<DIM>,
        chemical_potentials: Array1<f64>,
    ) -> Self {
        Self {
            k_mesh,
            directions,
            chemical_potentials,
            occupation: Occupation::ZeroTemperature,
            broadening: 1e-3,
            integration: QuantumGeometryIntegration::Direct,
        }
    }

    fn validate(&self) -> Result<()> {
        validate_k_mesh(&self.k_mesh)?;
        self.directions.validate()?;
        validate_chemical_potentials(&self.chemical_potentials)?;
        self.occupation.validate()?;
        validate_broadening(self.broadening)?;
        if self.integration == QuantumGeometryIntegration::Simplex && DIM == 1 {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![2, 3],
            });
        }
        Ok(())
    }
}

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
    fn quantum_geometry_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: &DirectionPair<DIM>,
        broadening: f64,
    ) -> Result<BandQuantumGeometry>;

    /// Evaluate every band on a list of k-points in parallel.
    fn quantum_geometry_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        directions: &DirectionPair<DIM>,
        broadening: f64,
    ) -> Result<QuantumGeometryMap>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> QuantumGeometry<DIM>
    for Model<SPIN, DIM, R>
{
    fn quantum_geometry_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: &DirectionPair<DIM>,
        broadening: f64,
    ) -> Result<BandQuantumGeometry> {
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        directions.validate()?;
        validate_broadening(broadening)?;

        let mut direction_matrix = Array2::<f64>::zeros((2, DIM));
        direction_matrix
            .row_mut(0)
            .assign(&ArrayView1::from(&directions.first));
        direction_matrix
            .row_mut(1)
            .assign(&ArrayView1::from(&directions.second));
        let (projected_velocity, hamiltonian) =
            self.gen_v_projected(k, Gauge::Atom, &direction_matrix);
        let (energies, eigenvectors) = hamiltonian.eigh(UPLO::Lower)?;
        let bra = eigenvectors.t();
        let ket = eigenvectors.mapv(|value| value.conj());

        let velocity_a = projected_velocity.index_axis(Axis(0), 0);
        let velocity_b = projected_velocity.index_axis(Axis(0), 1);
        let a_band = bra.dot(&velocity_a.dot(&ket));
        let b_band = bra.dot(&velocity_b.dot(&ket));
        let kernel = a_band * b_band.reversed_axes();
        let eta_squared = broadening * broadening;
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

    fn quantum_geometry_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        directions: &DirectionPair<DIM>,
        broadening: f64,
    ) -> Result<QuantumGeometryMap> {
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "quantum geometry k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        let rows: Vec<Result<BandQuantumGeometry>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.quantum_geometry_at(&k, directions, broadening))
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
    /// Integrate occupation-weighted quantum geometry over the Brillouin zone.
    ///
    /// Both algorithms return the same named result and use Cartesian
    /// reciprocal-space normalization. Simplex mode additionally reports the
    /// number of small-gap simplices encountered during band tracking.
    pub fn quantum_geometry(
        &self,
        params: &QuantumGeometryParams<DIM>,
    ) -> Result<QuantumGeometryResult> {
        params.validate()?;
        let k_mesh = mesh_array(&params.k_mesh);
        let determinant = self.lat.det()?;

        let (metric, berry_curvature, diagnostics) = match params.integration {
            QuantumGeometryIntegration::Direct => {
                let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
                let geometry =
                    self.quantum_geometry_on(&k_points, &params.directions, params.broadening)?;
                let normalization = 1.0 / k_points.nrows() as f64 / determinant;
                let values: Vec<(f64, f64)> = params
                    .chemical_potentials
                    .par_iter()
                    .map(|&mu| {
                        let mut metric_sum = 0.0;
                        let mut berry_sum = 0.0;
                        for k in 0..k_points.nrows() {
                            for band in 0..self.nsta() {
                                let occupation = params
                                    .occupation
                                    .value_unchecked(geometry.energies[[k, band]], mu);
                                metric_sum += geometry.metric[[k, band]] * occupation;
                                berry_sum += geometry.berry_curvature[[k, band]] * occupation;
                            }
                        }
                        (metric_sum * normalization, berry_sum * normalization)
                    })
                    .collect();
                let (metric, berry): (Vec<_>, Vec<_>) = values.into_iter().unzip();
                (Array1::from_vec(metric), Array1::from_vec(berry), None)
            }
            QuantumGeometryIntegration::Simplex => {
                let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
                let (direction_a, direction_b) = params.directions.as_arrays();
                let mut vertices: Vec<VertexKernel> = (0..k_points.nrows())
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
                global_band_track(&mut vertices, &params.k_mesh);
                let (metric, berry, unsafe_simplex_count) = integrate_occupied_geometry(
                    &vertices,
                    &k_mesh,
                    params.broadening,
                    &params.chemical_potentials,
                    params.occupation,
                );
                (
                    metric / determinant,
                    berry / determinant,
                    Some(IntegrationDiagnostics {
                        unsafe_simplex_count,
                    }),
                )
            }
        };

        Ok(QuantumGeometryResult {
            chemical_potentials: params.chemical_potentials.clone(),
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
        let geometry = model
            .quantum_geometry_at(
                &array![0.0, 0.0],
                &DirectionPair::new([1.0, 0.0], [0.0, 1.0]),
                1e-3,
            )
            .unwrap();
        assert_eq!(geometry.metric.len(), model.nsta());
        assert_eq!(geometry.berry_curvature.len(), model.nsta());
        assert_eq!(geometry.energies.len(), model.nsta());
    }

    #[test]
    fn direct_and_simplex_integrate_the_same_geometry() {
        let model = qwz_model(-1.0);
        let mut params = QuantumGeometryParams::new(
            [31, 31],
            DirectionPair::new([1.0, 0.0], [0.0, 1.0]),
            array![0.0],
        );
        params.broadening = 0.1;
        let direct = model.quantum_geometry(&params).unwrap();

        params.integration = QuantumGeometryIntegration::Simplex;
        let simplex = model.quantum_geometry(&params).unwrap();
        assert!(simplex.diagnostics.is_some());
        assert!((direct.metric[0] - simplex.metric[0]).abs() < 5e-3);
        assert!((direct.berry_curvature[0] - simplex.berry_curvature[0]).abs() < 5e-3);
    }
}
