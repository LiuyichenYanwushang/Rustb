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

use crate::ndarray_lapack::eigh_full;
use ndarray::Data;
use ndarray::array;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::response::config::{
    Integration, IntegrationDiagnostics, Parameters, ResponseAxis, direction_matrix, mesh_array,
    occupation_for, validate_broadening, validate_direction_values,
};
use crate::response::global_band_track;
use crate::response::linear::integrate_occupied_geometry;
use crate::thermodynamics::Occupation;
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
    /// The axis the result is indexed by; [`ResponseAxis::Fixed`] for a single
    /// evaluation at the fixed conditions.
    pub axis: ResponseAxis,
    /// Occupation-weighted quantum metric at every sample of that axis.
    pub metric: Array1<f64>,
    /// Occupation-weighted Berry curvature at every sample of that axis.
    pub berry_curvature: Array1<f64>,
    /// Present only for simplex integration.
    pub diagnostics: Option<IntegrationDiagnostics>,
}

/// Reusable band-resolved quantum-geometry kernels.
///
/// Neither method integrates over the Brillouin zone, so neither takes a
/// thermodynamic state or a k-mesh: they depend only on the supplied k-points,
/// the two projection directions and the broadening. They are charge-response
/// methods, so no spin current can be requested.
pub trait QuantumGeometry<const DIM: usize>: Velocity {
    /// Evaluate every band at one k-point.
    fn quantum_geometry_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
    ) -> Result<BandQuantumGeometry>;

    /// Evaluate every band on a list of k-points in parallel.
    fn quantum_geometry_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
    ) -> Result<QuantumGeometryMap>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> QuantumGeometry<DIM>
    for Model<SPIN, DIM, R>
{
    fn quantum_geometry_at<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
    ) -> Result<BandQuantumGeometry> {
        self.validate()?;
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        let direction = direction_matrix(&directions);
        self.quantum_geometry_at_impl(k, &direction, eta_ev)
    }

    fn quantum_geometry_on<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
    ) -> Result<QuantumGeometryMap> {
        self.validate()?;
        if k_points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "quantum geometry k-points".into(),
                expected: DIM,
                found: k_points.ncols(),
            });
        }
        // Validate once up front, then reuse the unvalidated kernel per k-point.
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        let direction = direction_matrix(&directions);
        self.quantum_geometry_map_impl(k_points, &direction, eta_ev)
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Band-resolved quantum-geometry kernel without input validation.
    ///
    /// Callers must have already validated the direction rank and `eta`. The
    /// high-level entry points validate once and then reuse this per k-point;
    /// the public trait methods are independent boundaries and validate before
    /// delegating here.
    pub(crate) fn quantum_geometry_at_impl<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
        direction: &Array2<f64>,
        eta: f64,
    ) -> Result<BandQuantumGeometry> {
        if k.iter().any(|value| !value.is_finite()) {
            return Err(TbError::InvalidResponseParameter {
                parameter: "k",
                message: "quantum geometry k-points must be finite".into(),
            });
        }
        let (projected_velocity, hamiltonian) = self.gen_v_projected(k, Gauge::Atom, direction);
        if projected_velocity
            .iter()
            .chain(hamiltonian.iter())
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                message: "Bloch Hamiltonian or projected velocities are nonfinite".into(),
            });
        }
        #[cfg(test)]
        crate::response::config::counters::count_eigen_decomposition();
        let (energies, eigenvectors) = eigh_full(&hamiltonian, UPLO::Lower)?;
        let bra = eigenvectors.mapv(|value| value.conj());
        let ket = eigenvectors.t();

        let velocity_a = projected_velocity.index_axis(Axis(0), 0);
        let velocity_b = projected_velocity.index_axis(Axis(0), 1);
        let a_band = bra.dot(&velocity_a.dot(&ket));
        let b_band = bra.dot(&velocity_b.dot(&ket));
        if energies.iter().any(|value| !value.is_finite())
            || a_band
                .iter()
                .chain(b_band.iter())
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                message: "band energies or band-projected velocities are nonfinite".into(),
            });
        }
        let mut metric = Array1::<f64>::zeros(self.nsta());
        let mut berry_curvature = Array1::<f64>::zeros(self.nsta());

        for band in 0..self.nsta() {
            let mut tensor = Complex::new(0.0, 0.0);
            for other in 0..self.nsta() {
                if band == other {
                    continue;
                }
                let difference = energies[band] - energies[other];
                if !difference.is_finite() {
                    return Err(TbError::InvalidResponseParameter {
                        parameter: "quantum_geometry",
                        message: format!("energy gap between bands {band} and {other} overflows"),
                    });
                }
                let scale = difference.abs().max(eta);
                if scale == 0.0 {
                    return Err(TbError::InvalidResponseParameter {
                        parameter: "quantum_geometry",
                        message: format!(
                            "bands {band} and {other} are degenerate with zero broadening"
                        ),
                    });
                }
                // Divide each velocity by sqrt(gap² + eta²) before multiplying.
                // The scaled hypot also avoids forming an overflowing norm.
                let norm = (difference / scale).hypot(eta / scale);
                let original_a = a_band[[band, other]];
                let original_b = b_band[[other, band]];
                let a = original_a / scale / norm;
                let b = original_b / scale / norm;
                // An asymmetric pair can have a representable product even
                // when one scaled factor underflows. Do not silently lose it.
                if ((((a.re == 0.0 || a.re.is_subnormal()) && original_a.re != 0.0)
                    || ((a.im == 0.0 || a.im.is_subnormal()) && original_a.im != 0.0))
                    && original_b != Complex::new(0.0, 0.0))
                    || ((((b.re == 0.0 || b.re.is_subnormal()) && original_b.re != 0.0)
                        || ((b.im == 0.0 || b.im.is_subnormal()) && original_b.im != 0.0))
                        && original_a != Complex::new(0.0, 0.0))
                {
                    return Err(TbError::InvalidResponseParameter {
                        parameter: "quantum_geometry",
                        message: format!(
                            "scaled velocity for bands {band}, {other} underflowed; geometry cannot be certified"
                        ),
                    });
                }
                tensor += a * b;
            }
            metric[band] = tensor.re;
            berry_curvature[band] = -2.0 * tensor.im;
            if !metric[band].is_finite() || !berry_curvature[band].is_finite() {
                return Err(TbError::InvalidResponseParameter {
                    parameter: "quantum_geometry",
                    message: format!("geometry of band {band} is not representable as finite f64"),
                });
            }
        }

        Ok(BandQuantumGeometry {
            metric,
            berry_curvature,
            energies,
        })
    }

    /// Band-resolved quantum geometry on a k-mesh without input validation.
    ///
    /// Callers must have already validated the direction rank and `eta`.
    pub(crate) fn quantum_geometry_map_impl<S: Data<Elem = f64> + Sync>(
        &self,
        k_points: &ArrayBase<S, Ix2>,
        direction: &Array2<f64>,
        eta: f64,
    ) -> Result<QuantumGeometryMap> {
        let rows: Vec<Result<BandQuantumGeometry>> = k_points
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.quantum_geometry_at_impl(&k, direction, eta))
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

    /// Integrate occupation-weighted quantum geometry over the Brillouin zone.
    ///
    /// Reads `conditions` (at most one axis sampled), `kmesh` and `integration`
    /// from `params`. The rank-2 `directions` and broadening `eta_ev` are explicit
    /// arguments. This is a charge-only response. `conditions.omega_ev` must be
    /// `Sampling::Fixed(0.0)` because the response is DC. Both algorithms use
    /// Cartesian reciprocal-space normalization and prepare eigenstates,
    /// velocity kernels and band tracking once. Weighting and integration still
    /// run for each sample; temperature scans repeat simplex quadrature on the
    /// shared vertices. Simplex mode additionally reports the number of
    /// small-gap simplices.
    pub fn quantum_geometry(
        &self,
        params: &Parameters<DIM>,
        directions: [[f64; DIM]; 2],
        eta_ev: f64,
    ) -> Result<QuantumGeometryResult> {
        let resolved = params.validate_grid_response()?;
        resolved.require_dc()?;
        self.validate()?;
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        let direction = direction_matrix(&directions);
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
        let eta = eta_ev;
        let k_mesh = mesh_array(&params.kmesh);
        let determinant = self.lat.det()?;
        if !determinant.is_finite() || determinant == 0.0 {
            return Err(TbError::InvalidResponseParameter {
                parameter: "lattice_volume",
                message: "quantum geometry needs a finite, nonzero lattice determinant".into(),
            });
        }
        let samples = resolved.len();

        let (metric, berry_curvature, diagnostics) = match params.integration {
            Integration::Direct => {
                let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
                let geometry = self.quantum_geometry_map_impl(&k_points, &direction, eta)?;
                let normalization = 1.0 / k_points.nrows() as f64 / determinant;
                let values: Vec<(f64, f64)> = (0..samples)
                    .into_par_iter()
                    .map(|index| {
                        let (t_kelvin, mu, _) = resolved.point(index);
                        let occupation = occupation_for(t_kelvin);
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
                let direction_a = direction.row(0).to_owned();
                let direction_b = direction.row(1).to_owned();
                let vertices: Vec<Result<_>> = (0..k_points.nrows())
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
                let (mut vertices, mut eigenvectors): (Vec<_>, Vec<_>) = vertices
                    .into_iter()
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .unzip();
                // Tracked once; every sample reuses the labelled vertices.
                global_band_track(&mut eigenvectors, &params.kmesh, |index, permutation| {
                    vertices[index] =
                        crate::response::permute_vertex(&vertices[index], permutation);
                });
                drop(eigenvectors);
                // ponytail: Simplex divides unscaled interpolated primitives.
                // Certify denominator bounds for every convex vertex mixture;
                // use Direct outside this range rather than silently clipping to zero.
                for band in 0..self.nsta() {
                    for other in band + 1..self.nsta() {
                        let mut minimum = f64::INFINITY;
                        let mut maximum = f64::NEG_INFINITY;
                        for vertex in &vertices {
                            let gap = vertex.band[band] - vertex.band[other];
                            minimum = minimum.min(gap);
                            maximum = maximum.max(gap);
                        }
                        let smallest = if minimum <= 0.0 && maximum >= 0.0 {
                            0.0
                        } else {
                            minimum.abs().min(maximum.abs())
                        };
                        let largest = minimum.abs().max(maximum.abs());
                        // NV <= 4: leave room for weighted-gap accumulation,
                        // squaring and addition rounding at both hard boundaries.
                        let roundoff = 64.0 * f64::EPSILON;
                        let upper = largest * largest + eta * eta;
                        if smallest * smallest + eta * eta < 1e-30 * (1.0 + roundoff)
                            || !upper.is_finite()
                            || upper > f64::MAX * (1.0 - roundoff)
                        {
                            return Err(TbError::InvalidResponseParameter {
                                parameter: "quantum_geometry",
                                message: "Simplex quantum geometry cannot represent the interpolated denominator; use Direct".into(),
                            });
                        }
                    }
                }
                let integrate = |mu_values: &Array1<f64>, occupation: Occupation| {
                    integrate_occupied_geometry(&vertices, &k_mesh, eta, mu_values, occupation)
                };
                let (metric, berry, unsafe_simplex_count) = match &resolved.axis {
                    ResponseAxis::ChemicalPotential(values) => {
                        integrate(values, resolved.occupation(0))
                    }
                    // Temperature samples redo the quadrature on the shared
                    // simplex decomposition.
                    _ => {
                        let mut metric = Array1::<f64>::zeros(samples);
                        let mut berry = Array1::<f64>::zeros(samples);
                        let mut unsafe_count = 0usize;
                        for index in 0..samples {
                            let (_, mu, _) = resolved.point(index);
                            let (sample_metric, sample_berry, count) =
                                integrate(&array![mu], resolved.occupation(index));
                            metric[index] = sample_metric[0];
                            berry[index] = sample_berry[0];
                            unsafe_count = unsafe_count.max(count);
                        }
                        (metric, berry, unsafe_count)
                    }
                };
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

        if metric
            .iter()
            .chain(berry_curvature.iter())
            .any(|value| !value.is_finite())
        {
            return Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                message: "integrated geometry is not representable as finite f64".into(),
            });
        }
        Ok(QuantumGeometryResult {
            axis: resolved.axis,
            metric,
            berry_curvature,
            diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::config::{Conditions, Sampling};
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

    fn atomic_dimer(t: f64) -> Model<false, 1> {
        let mut model =
            Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], None).unwrap();
        model.add_hop(t, 0, 1, &array![0], None);
        model
    }

    #[test]
    fn atomic_dimer_geometry_is_scale_invariant_at_on_and_direct() {
        let directions = [[1.0], [1.0]];
        let points = array![[0.0], [0.173], [0.49]];
        let params = Parameters::<1> {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [3],
            integration: Integration::Direct,
        };
        // The largest t with eta/|t|=1.75 also overflows an unscaled hypot.
        for magnitude in [1e-300, 1e-200, 1.0, 1e200, 1e300, 7e307] {
            for sign in [-1.0, 1.0] {
                let model = atomic_dimer(sign * magnitude);
                for eta_ratio in [0.0, 0.75, 1.75] {
                    let eta = eta_ratio * magnitude;
                    // Equal-weight dimer kets have position variance (0.5)²/4.
                    // Broadening multiplies it by 4t²/(4t² + eta²).
                    let expected = (1.0 / 16.0) / (1.0 + eta_ratio * eta_ratio / 4.0);
                    for point in points.rows() {
                        let geometry = model.quantum_geometry_at(&point, directions, eta).unwrap();
                        for band in 0..2 {
                            assert!((geometry.metric[band] - expected).abs() < 2e-13);
                            assert!(geometry.berry_curvature[band].abs() < 2e-13);
                            let energy = if band == 0 { -1.0 } else { 1.0 };
                            assert!((geometry.energies[band] / magnitude - energy).abs() < 2e-13);
                        }
                    }
                    let map = model.quantum_geometry_on(&points, directions, eta).unwrap();
                    assert!(
                        map.metric
                            .iter()
                            .all(|value| (value - expected).abs() < 2e-13)
                    );
                    assert!(map.berry_curvature.iter().all(|value| value.abs() < 2e-13));
                    assert!(map.energies.iter().all(|value| value.is_finite()));
                    // At zero chemical potential only the lower band is occupied.
                    let integrated = model.quantum_geometry(&params, directions, eta).unwrap();
                    assert!((integrated.metric[0] - expected).abs() < 2e-13);
                    assert!(integrated.berry_curvature[0].abs() < 2e-13);
                }
            }
        }
    }

    #[test]
    fn undefined_and_unrepresentable_dimer_geometry_returns_structured_errors() {
        let params = Parameters::<1> {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [1],
            integration: Integration::Direct,
        };
        for (t, direction, reason) in [
            (0.0, 1.0, "degenerate"),
            (1.0, 1e200, "not representable"),
            (1e200, 1e200, "projected velocities"),
            (1e308, 1.0, "energy gap"),
        ] {
            let model = atomic_dimer(t);
            let directions = [[direction], [direction]];
            let errors = [
                model
                    .quantum_geometry_at(&array![0.0], directions, 0.0)
                    .unwrap_err(),
                model
                    .quantum_geometry_on(&array![[0.0]], directions, 0.0)
                    .unwrap_err(),
                model
                    .quantum_geometry(&params, directions, 0.0)
                    .unwrap_err(),
            ];
            for error in errors {
                match error {
                    TbError::InvalidResponseParameter {
                        parameter: "quantum_geometry",
                        message,
                    } => assert!(message.contains(reason), "{message}"),
                    other => panic!("unexpected geometry error: {other:?}"),
                }
            }
        }
        let model = atomic_dimer(1.0);
        for k in [f64::NAN, f64::INFINITY] {
            for error in [
                model
                    .quantum_geometry_at(&array![k], [[1.0], [1.0]], 0.0)
                    .unwrap_err(),
                model
                    .quantum_geometry_on(&array![[k]], [[1.0], [1.0]], 0.0)
                    .unwrap_err(),
            ] {
                assert!(matches!(
                    error,
                    TbError::InvalidResponseParameter { parameter: "k", .. }
                ));
            }
        }
        // Positive broadening makes the zero-velocity degenerate dimer finite.
        let regularized = atomic_dimer(0.0)
            .quantum_geometry_at(&array![0.0], [[1.0], [1.0]], 1e-200)
            .unwrap();
        assert!(regularized.metric.iter().all(|value| *value == 0.0));
        assert!(
            regularized
                .berry_curvature
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn quantum_geometry_rejects_berry_curvature_overflow() {
        // At k=0 this model has gap 2 and unit off-diagonal x/y velocities.
        // G_xy is finite (|Im G_xy| = 1e308), but -2 Im G_xy overflows.
        let model = qwz_model(-3.0);
        assert!(matches!(
            model.quantum_geometry_at(&array![0.0, 0.0], [[2e154, 0.0], [0.0, 2e154]], 0.0),
            Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                ..
            })
        ));
    }

    #[test]
    fn simplex_preserves_small_gaps_at_nonzero_energy_origin() {
        let gap = f64::from_bits(8.0_f64.to_bits() + 1) - 8.0;
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![8.0, 8.0 + gap], None);
        model.add_hop(Complex::new(0.0, -gap / 8.0), 0, 1, &array![1, 0], None);
        model.add_hop(Complex::new(0.0, gap / 8.0), 0, 1, &array![-1, 0], None);
        for mu in [8.0, 8.0 + gap, 10.0] {
            let expected = (Occupation::ZeroTemperature.value_unchecked(8.0, mu)
                + Occupation::ZeroTemperature.value_unchecked(8.0 + gap, mu))
                / 16.0;
            for integration in [Integration::Direct, Integration::Simplex] {
                let params = Parameters {
                    conditions: Conditions {
                        t_kelvin: Sampling::Fixed(0.0),
                        mu_ev: Sampling::Fixed(mu),
                        omega_ev: Sampling::Fixed(0.0),
                    },
                    kmesh: [1, 1],
                    integration,
                };
                let result = model
                    .quantum_geometry(&params, [[1.0, 0.0], [1.0, 0.0]], 0.0)
                    .unwrap();
                assert!(
                    (result.metric[0] - expected).abs() < 1e-12,
                    "mu={mu}, {integration:?}: {result:?}"
                );
            }
        }
    }

    #[test]
    fn geometry_rejects_asymmetric_scaled_velocity_underflow() {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![-5e199, 5e199], None);
        for (amplitude, r) in [(1e-200, array![1, 0]), (1e300, array![0, 1])] {
            model.add_hop(Complex::new(0.0, -amplitude / 2.0), 0, 1, &r, None);
            model.add_hop(Complex::new(0.0, amplitude / 2.0), 0, 1, &(-&r), None);
        }
        // True g_xy is about 1e-300, not zero; an uncertifiable ratio must
        // fail explicitly rather than losing its small but finite factor.
        assert!(matches!(
            model.quantum_geometry_at(&array![0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 0.0),
            Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                ..
            })
        ));
        // Also reject loss of just the real component while an imaginary
        // scaled component survives (vx gains a unit sigma_y term).
        model.add_hop(-0.5, 0, 1, &array![1, 0], None);
        model.add_hop(0.5, 0, 1, &array![-1, 0], None);
        assert!(matches!(
            model.quantum_geometry_at(&array![0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 0.0),
            Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                ..
            })
        ));
    }

    #[test]
    fn simplex_rejects_roundoff_near_both_denominator_boundaries() {
        for (gap, eta) in [
            (1.9100000000000002e-16, 9.815900366242518e-16),
            (7.3742943614684275e152, 1.3387513260980761e154),
        ] {
            let mut model =
                Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
            model.set_onsite(&array![0.0, gap], None);
            model.add_hop(Complex::new(0.0, -gap / 8.0), 0, 1, &array![1, 0], None);
            model.add_hop(Complex::new(0.0, gap / 8.0), 0, 1, &array![-1, 0], None);
            let params = Parameters {
                conditions: Conditions {
                    t_kelvin: Sampling::Fixed(0.0),
                    mu_ev: Sampling::Fixed(gap * 2.0),
                    omega_ev: Sampling::Fixed(0.0),
                },
                kmesh: [1, 1],
                integration: Integration::Simplex,
            };
            let direct = model
                .quantum_geometry_at(&array![0.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], eta)
                .unwrap();
            assert!(direct.metric.iter().all(|g| g.is_finite() && *g > 0.0));
            assert!(matches!(
                model.quantum_geometry(&params, [[1.0, 0.0], [1.0, 0.0]], eta),
                Err(TbError::InvalidResponseParameter {
                    parameter: "quantum_geometry",
                    ..
                })
            ));
        }
    }

    #[test]
    fn simplex_rejects_partial_raw_velocity_product_underflow() {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![0.0, 1e-15], None);
        for (r, amplitude) in [
            (array![1, 0], Complex::new(-0.5, -0.5e-165)),
            (array![-1, 0], Complex::new(0.5, 0.5e-165)),
            (array![0, 1], Complex::new(0.0, -0.5e-165)),
            (array![0, -1], Complex::new(0.0, 0.5e-165)),
        ] {
            model.add_hop(amplitude, 0, 1, &r, None);
        }
        let k = array![0.0, 0.0];
        let direct = model
            .quantum_geometry_at(&k, [[1.0, 0.0], [0.0, 1.0]], 0.0)
            .unwrap();
        assert!((direct.metric[0] / 1e-300 - 1.0).abs() < 1e-12);
        assert!(matches!(
            model.compute_velocity_kernel(
                &k,
                &array![1.0, 0.0],
                &array![0.0, 1.0],
                None,
                Gauge::Atom,
                None
            ),
            Err(TbError::InvalidResponseParameter {
                parameter: "velocity_kernel",
                ..
            })
        ));
    }

    #[test]
    fn direct_rejects_subnormal_scaled_factors_with_normal_raw_quotient() {
        let gap = 3.0 * 2.0_f64.powi(500);
        let va = 2.0_f64.powi(-570);
        let vb = 3.0 * 2.0_f64.powi(1000);
        let raw_quotient = (va * vb) / (gap * gap);
        assert!((raw_quotient / (va / 3.0) - 1.0).abs() < 1e-12);
        assert!((va * vb).is_normal() && (gap * gap).is_normal() && raw_quotient.is_normal());
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![0.0, gap], None);
        for (amplitude, r) in [(va, array![1, 0]), (vb, array![0, 1])] {
            model.add_hop(Complex::new(0.0, -amplitude / 2.0), 0, 1, &r, None);
            model.add_hop(Complex::new(0.0, amplitude / 2.0), 0, 1, &(-&r), None);
        }
        assert!(matches!(
            model.quantum_geometry_at(&array![0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 0.0),
            Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                ..
            })
        ));
    }

    #[test]
    fn simplex_rejects_subnormal_raw_products_with_normal_geometry() {
        let gap = 2.0_f64.powi(-48);
        let velocity = 2.0_f64.powi(-536);
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![0.0, gap], None);
        model.add_hop(
            Complex::new(0.0, -velocity / 2.0),
            0,
            1,
            &array![1, 0],
            None,
        );
        model.add_hop(
            Complex::new(0.0, velocity / 2.0),
            0,
            1,
            &array![-1, 0],
            None,
        );
        let k = array![0.0, 0.0];
        let direct = model
            .quantum_geometry_at(&k, [[1.0, 0.0], [1.0, 0.0]], 0.0)
            .unwrap();
        let expected = 2.0_f64.powi(-976);
        assert!(expected.is_normal());
        assert!((direct.metric[0] / expected - 1.0).abs() < 1e-12);
        let params = Parameters {
            conditions: Conditions::fixed(0.0, gap * 2.0, 0.0),
            kmesh: [1, 1],
            integration: Integration::Simplex,
        };
        assert!(matches!(
            model.quantum_geometry(&params, [[1.0, 0.0], [1.0, 0.0]], 0.0),
            Err(TbError::InvalidResponseParameter {
                parameter: "velocity_kernel",
                ..
            })
        ));
    }

    #[test]
    fn quantum_geometry_rejects_nonfinite_integrated_outputs() {
        let mut dimer = atomic_dimer(1.0);
        dimer.lat[[0, 0]] = 1e-200;
        let directions = [[1e300], [1e300]];
        let bands = dimer
            .quantum_geometry_at(&array![0.0], directions, 0.0)
            .unwrap();
        assert!(bands.metric.iter().all(|value| value.is_finite()));
        let params = Parameters::<1> {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [1],
            integration: Integration::Direct,
        };
        // The Cartesian normalization makes the otherwise finite metric overflow.
        assert!(matches!(
            dimer.quantum_geometry(&params, directions, 0.0),
            Err(TbError::InvalidResponseParameter {
                parameter: "quantum_geometry",
                ..
            })
        ));

        let mut large =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0], [0.5, 0.0]], None)
                .unwrap();
        large.add_hop(1e200, 0, 1, &array![0, 0], None);
        let params = Parameters::<2> {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [2, 2],
            integration: Integration::Simplex,
        };
        // Simplex still uses raw products/denominators: unsupported scales
        // must be errors, including finite-but-wrong zeros and huge eta².
        for (hopping, eta) in [(1e200, 0.0), (1e-200, 0.0), (1e-20, 0.0), (1e150, 1e200)] {
            large.set_hop(hopping, 0, 1, &array![0, 0], None);
            assert!(matches!(
                large.quantum_geometry(&params, [[1.0, 0.0], [1.0, 0.0]], eta),
                Err(TbError::InvalidResponseParameter {
                    parameter: "quantum_geometry" | "velocity_kernel",
                    ..
                })
            ));
        }
        // A genuine zero projected velocity must not count as underflow.
        large.set_hop(1e-200, 0, 1, &array![0, 0], None);
        let zero = large
            .compute_velocity_kernel(
                &array![0.0, 0.0],
                &array![1.0, 0.0],
                &array![0.0, 1.0],
                None,
                Gauge::Atom,
                None,
            )
            .unwrap()
            .0;
        assert!(zero.k_ab.iter().all(|z| *z == Complex::new(0.0, 0.0)));
        // An invertible lattice can still have an under/overflowing determinant.
        for scale in [1e-200, 1e200] {
            large.lat = Array2::eye(2) * scale;
            assert!(matches!(
                large.quantum_geometry(&params, [[1.0, 0.0], [1.0, 0.0]], 0.0),
                Err(TbError::InvalidResponseParameter {
                    parameter: "lattice_volume",
                    ..
                })
            ));
        }
    }

    #[test]
    fn named_band_result_has_real_components() {
        let model = massive_dirac_model();
        let geometry = model
            .quantum_geometry_at(&array![0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 1e-3)
            .unwrap();
        assert_eq!(geometry.metric.len(), model.nsta());
        assert_eq!(geometry.berry_curvature.len(), model.nsta());
        assert_eq!(geometry.energies.len(), model.nsta());
    }

    #[test]
    fn direct_and_simplex_integrate_the_same_geometry() {
        let model = qwz_model(-1.0);
        let mut params = Parameters::<2> {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [31, 31],
            integration: Integration::Direct,
        };
        let directions = [[1.0, 0.0], [0.0, 1.0]];
        let direct = model.quantum_geometry(&params, directions, 0.1).unwrap();

        params.integration = Integration::Simplex;
        let simplex = model.quantum_geometry(&params, directions, 0.1).unwrap();
        assert!(simplex.diagnostics.is_some());
        assert!((direct.metric[0] - simplex.metric[0]).abs() < 5e-3);
        assert!((direct.berry_curvature[0] - simplex.berry_curvature[0]).abs() < 5e-3);
    }
}
