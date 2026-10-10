//! # Nonlinear response: Berry dipole, intrinsic & extrinsic NLH
//!
//! ## Extrinsic NLH — Berry curvature dipole (BCD)
//!
//! ```math
//! \chi^{\rm ext}_{abc}(\mu,T) =
//!   \sum_n \int_{\rm BZ} \left(-\frac{\partial f}{\partial E_n}\right)
//!   v^c_n(\mathbf{k})\Omega^{ab}_n(\mathbf{k})\,d\mathbf{k}
//! ```
//!
//! The BCD is **TR‑even** ($D\_{TR}=D$) — survives in TR‑symmetric, P‑broken systems.
//! Under time reversal: $v^c\to -v^c$, $\Omega^{ab}\to -\Omega^{ab}$, so the
//! product $v^c\Omega^{ab}$ is invariant.
//!
//! ## Intrinsic NLH — Berry connection dipole
//!
//! ```math
//! \sigma^{ab;c}_{\rm int}(\mu,T) = -\frac{e^3}{\hbar}
//!   \sum_n \int_{\rm BZ} (-\partial f/\partial E_n)
//!   \bigl[2v^c_n G^{ab}_n - \tfrac12(v^a_n G^{bc}_n + v^b_n G^{ac}_n)\bigr]\,d\mathbf{k}
//! ```
//!
//! where $G^{ij}\_n = \operatorname{Re}\sum\_{m\ne n} K^{ij}\_{nm} / (E\_n-E\_m)^3$.
//! The intrinsic NLH is **TR‑odd** ($\sigma\_{TR}=-\sigma$) — requires both
//! $\mathcal P$ and $\mathcal T$ breaking.
//!
//! ## API
//!
//! | Method | Path | Formula |
//! |--------|------|---------|
//! | `extrinsic_nonlinear_hall` | direct sum or energy cut | $\chi^{\rm ext}$ |
//! | `intrinsic_nonlinear_hall` | direct sum or energy cut | $\sigma\_{\rm int}$ |

use crate::ndarray_lapack::eigh_full;
use ndarray::array;
use ndarray::prelude::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::SpinDirection;
use crate::error::{Result, TbError};
#[cfg(test)]
use crate::math::anti_comm;
use crate::thermodynamics::fermi_derivative_from_width;

use super::config::{
    FieldSymmetry, Integration, IntegrationDiagnostics, Parameters, ResponseAxis, direction_matrix,
    mesh_array, occupation_for, validate_broadening, validate_direction_values, validate_sorted,
};
use super::energy_cut::integrate_dipole_energy_cut_2d;
use super::kernel::intrinsic_inverse_gap;
use super::tracking::global_band_track;
use super::types::VertexKernel;

/// Nonlinear Hall conductivity on the sampled axis, or at the fixed conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearHallResult {
    /// The axis the result is indexed by; [`ResponseAxis::Fixed`] for a single
    /// evaluation at the fixed conditions.
    pub axis: ResponseAxis,
    /// Nonlinear Hall response at every sample of that axis.
    pub conductivity: Array1<f64>,
    /// Algorithm diagnostics when exposed by the selected energy-cut path.
    pub diagnostics: Option<IntegrationDiagnostics>,
}

impl NonlinearHallResult {
    fn checked(
        axis: ResponseAxis,
        conductivity: Array1<f64>,
        diagnostics: Option<IntegrationDiagnostics>,
    ) -> Result<Self> {
        if let Some(index) = conductivity.iter().position(|value| !value.is_finite()) {
            return Err(TbError::InvalidResponseParameter {
                parameter: "nonlinear_hall",
                message: format!(
                    "nonlinear Hall conductivity sample {index} is nonfinite; check velocity, energy and temperature scales"
                ),
            });
        }
        Ok(Self {
            axis,
            conductivity,
            diagnostics,
        })
    }

    /// The scalar value of a single-point calculation.
    ///
    /// `None` for a sampled axis, including a sampled series that happens to
    /// hold exactly one value: only [`ResponseAxis::Fixed`] is a single point.
    pub fn single(&self) -> Option<f64> {
        self.conductivity
            .first()
            .copied()
            .filter(|_| self.axis.is_fixed())
    }
}

impl<const DIM: usize, R: RMatrixData> Model<true, DIM, R> {
    /// Evaluate the Berry-curvature-dipole nonlinear Hall response.
    ///
    /// `directions` rows are `(current, field_1, field_2)`. In the internal
    /// kernel this maps to `Ω^{current, field_1} v^{field_2}`: `current` and
    /// `field_1` are the two Berry-curvature indices and `field_2` is the
    /// Fermi-surface velocity index. `eta_ev` broadens the Berry-curvature
    /// denominator, `spin: None` selects the charge current and a spin direction
    /// selects the corresponding spin current, and `field_symmetry` fixes the
    /// ordering convention of the two field indices:
    /// [`FieldSymmetry::Symmetrized`] averages the two field permutations,
    /// [`FieldSymmetry::Ordered`] returns the raw ordered kernel.
    ///
    /// This is a DC response, so `params.conditions.omega_ev` must be
    /// `Sampling::Fixed(0.0)`. Eigenstates, velocity kernels and band tracking
    /// are prepared once and reused by every sample of the sampled axis.
    ///
    /// Direct integration samples `-df/dE` on k-points, so every sample must
    /// have a positive thermal energy `k_B T` with a finite Fermi-window peak
    /// `0.25 / (k_B T)`. Zero widths or overflowing peaks reject the whole call
    /// before any k-mesh work; subnormal widths with finite peaks are accepted.
    /// Energy-cut integration supports the exact zero-temperature limit.
    pub fn extrinsic_nonlinear_hall(
        &self,
        params: &Parameters<DIM>,
        directions: [[f64; DIM]; 3],
        eta_ev: f64,
        spin: Option<SpinDirection>,
        field_symmetry: FieldSymmetry,
    ) -> Result<NonlinearHallResult> {
        self.extrinsic_nonlinear_hall_impl(
            params,
            directions,
            eta_ev,
            spin,
            field_symmetry,
            |direction| Ok(self.build_spin_matrix(direction)),
        )
    }
}

impl<const DIM: usize, R: RMatrixData> Model<false, DIM, R> {
    /// Evaluate the Berry-curvature-dipole nonlinear Hall response.
    ///
    /// Identical to the spinful entry point except that a requested spin
    /// current returns [`TbError::SpinNotAllowed`]: a model without spin has no
    /// spin current. See
    /// [`Model::<true, DIM, R>::extrinsic_nonlinear_hall`] for the remaining
    /// argument contract, including the `(current, field_1, field_2)` row order
    /// of `directions` and the two `field_symmetry` conventions.
    pub fn extrinsic_nonlinear_hall(
        &self,
        params: &Parameters<DIM>,
        directions: [[f64; DIM]; 3],
        eta_ev: f64,
        spin: Option<SpinDirection>,
        field_symmetry: FieldSymmetry,
    ) -> Result<NonlinearHallResult> {
        self.extrinsic_nonlinear_hall_impl(
            params,
            directions,
            eta_ev,
            spin,
            field_symmetry,
            |direction| Err(TbError::SpinNotAllowed(direction)),
        )
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Computes the unsymmetrized Berry-curvature-dipole kernel for each band at a
    /// single k-point.
    ///
    /// This computes:
    /// $$ \pdv{\varepsilon_{n\mathbf k}}{k_\gamma} \Omega_{n,\alpha\beta} $$
    ///
    /// The energy derivative is obtained using the diagonal elements of the velocity operator:
    /// $$ \pdv{\varepsilon_{\mathbf k}}{\mathbf k} = \text{diag}(v_{\mathbf k}) $$
    /// This follows from the relation $\varepsilon_{\mathbf k} = U^\dagger H_{\mathbf k} U$ and
    /// the observation that the commutator term $[\varepsilon_{\mathbf k}, U^\dagger\partial_{\mathbf k}U]$
    /// does not contribute to diagonal elements.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates.
    /// * `current_dir` - First Berry-curvature index $\alpha$ of $\Omega_{n,\alpha\beta}$.
    /// * `dir_2` - Second Berry-curvature index $\beta$.
    /// * `dir_3` - Velocity / Fermi-surface index $\gamma$.
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// `(omega_n, band)` where `omega_n` contains $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$
    /// for each band, and `band` contains the band energies.
    #[cfg(test)]
    pub(crate) fn berry_curvature_dipole_n_onek(
        &self,
        k_vec: &Array1<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin_matrix: Option<&Array2<Complex<f64>>>,
        eta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if k_vec.len() != self.dim_r() {
            return Err(TbError::KVectorLengthMismatch {
                expected: self.dim_r(),
                actual: k_vec.len(),
            });
        }
        // Build direction matrix: [current_dir, dir_2, dir_3]
        let directions = {
            let mut d = Array2::<f64>::zeros((3, self.dim_r()));
            d.row_mut(0).assign(current_dir);
            d.row_mut(1).assign(dir_2);
            d.row_mut(2).assign(dir_3);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(k_vec, Gauge::Atom, &directions);
        // v_proj[0] = Σ_d current_dir[d] * v_raw[d]  → J
        // v_proj[1] = Σ_d dir_2[d] * v_raw[d]        → v
        // v_proj[2] = Σ_d dir_3[d] * v_raw[d]        → v0
        let J: Array2<Complex<f64>> = if let Some(matrix) = spin_matrix {
            anti_comm(matrix, &v_proj.slice(s![0, .., ..])) * 0.5
        } else {
            v_proj.slice(s![0, .., ..]).to_owned()
        };
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();
        let v0: Array2<Complex<f64>> = v_proj.slice(s![2, .., ..]).to_owned();

        #[cfg(test)]
        super::config::counters::count_eigen_decomposition();
        let (band, evec) = eigh_full(&hamk, UPLO::Lower)?;
        let evec_conj = evec.mapv(|x| x.conj());
        let evec = evec.t();

        let v0 = v0.dot(&evec);
        let v0 = &evec_conj.dot(&v0);
        let partial_ve = v0.diag().map(|x| x.re);
        let A1 = J.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec);
        let A2 = &evec_conj.dot(&A2);
        let mut U0 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                if i != j {
                    U0[[i, j]] =
                        Complex::new(1.0 / ((band[[i]] - band[[j]]).powi(2) + eta * eta), 0.0);
                } else {
                    U0[[i, j]] = Complex::new(0.0, 0.0);
                }
            }
        }
        let mut omega_n = Array1::<f64>::zeros(self.nsta());
        let A1 = A1 * U0;
        for i in 0..self.nsta() {
            omega_n[[i]] = -2.0 * A1.slice(s![i, ..]).dot(&A2.slice(s![.., i])).im;
        }

        let omega_n: Array1<f64> = omega_n * partial_ve;
        Ok((omega_n, band))
    }

    /// Computes the Berry curvature dipole for each band at multiple k-points in parallel.
    ///
    /// This is a parallelized version of [`Self::berry_curvature_dipole_n_onek`] for computing
    /// the Berry curvature dipole over a k-point set.
    ///
    /// The extrinsic nonlinear Hall conductivity is related to this quantity via:
    /// $$ \sigma_{\alpha\beta\gamma} = \tau \int \dd\mathbf k \sum_n
    ///    \partial_\gamma \varepsilon_{n\mathbf k} \Omega_{n,\alpha\beta}
    ///    \left. \pdv{f_{\mathbf k}}{\varepsilon} \right\rvert_{E=\varepsilon_{n\mathbf k}}. $$
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `current_dir`, `dir_2` - Direction vectors for the Berry curvature indices $\alpha, \beta$.
    /// * `dir_3` - Direction vector for the energy derivative index $\gamma$.
    /// * `spin_matrix` - Prebuilt spin matrix shared by all k-points; `None` = charge current.
    /// * `eta` - Broadening parameter.
    ///
    /// # Returns
    ///
    /// `(omega, band)` where `omega` has shape `(nk, nsta)` containing
    /// $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$ for each k-point and band,
    /// and `band` has the band energies with the same shape.
    #[cfg(test)]
    pub(crate) fn berry_curvature_dipole_n(
        &self,
        k_vec: &Array2<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin_matrix: Option<&Array2<Complex<f64>>>,
        eta: f64,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        for (name, dir) in [
            ("current_dir", current_dir),
            ("dir_2", dir_2),
            ("dir_3", dir_3),
        ] {
            if dir.len() != self.dim_r() {
                return Err(TbError::DimensionMismatch {
                    context: name.into(),
                    expected: self.dim_r(),
                    found: dir.len(),
                });
            }
        }
        if k_vec.ncols() != self.dim_r() {
            return Err(TbError::DimensionMismatch {
                context: "k_vec".into(),
                expected: self.dim_r(),
                found: k_vec.ncols(),
            });
        }
        let nk = k_vec.len_of(Axis(0));
        let results: Vec<Result<_>> = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                self.berry_curvature_dipole_n_onek(
                    &x.to_owned(),
                    current_dir,
                    dir_2,
                    dir_3,
                    spin_matrix,
                    eta,
                )
            })
            .collect();
        let results: Vec<_> = results.into_iter().collect::<Result<_>>()?;
        let (omega, band): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        let omega =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), omega.into_iter().flatten().collect())
                .map_err(|e| TbError::Shape(e))?;
        let band =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), band.into_iter().flatten().collect())
                .map_err(|e| TbError::Shape(e))?;
        Ok((omega, band))
    }

    /// Shared validation and integration for the concrete spinful/spinless entry points.
    fn extrinsic_nonlinear_hall_impl(
        &self,
        params: &Parameters<DIM>,
        directions: [[f64; DIM]; 3],
        eta_ev: f64,
        spin: Option<SpinDirection>,
        field_symmetry: FieldSymmetry,
        build_spin: impl FnOnce(SpinDirection) -> Result<Array2<Complex<f64>>>,
    ) -> Result<NonlinearHallResult> {
        let resolved = params.validate_grid_response()?;
        resolved.require_dc()?;
        self.validate()?;
        validate_direction_values(&directions)?;
        validate_broadening(eta_ev)?;
        let eta = eta_ev;
        let eta_squared = eta * eta;
        if !eta_squared.is_finite() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "eta_ev",
                message: "squared nonlinear Hall broadening overflows".into(),
            });
        }
        if !SPIN && let Some(direction) = spin {
            return Err(TbError::SpinNotAllowed(direction));
        }
        if params.integration == Integration::Simplex {
            return Err(TbError::InvalidResponseParameter {
                parameter: "integration",
                message: "extrinsic_nonlinear_hall supports Integration::Direct or EnergyCut, not Simplex".into(),
            });
        }
        if params.integration == Integration::EnergyCut {
            validate_sorted(&resolved.chemical_potentials(), "mu_ev")?;
            if DIM != 2 {
                return Err(TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2],
                });
            }
        } else {
            // Reject the whole call, before any k-mesh work, if any sample
            // has zero thermal width or an overflowing Fermi-window peak.
            resolved.require_positive_temperature()?;
        }
        let samples = resolved.len();
        let widths: Vec<f64> = (0..samples)
            .map(|index| occupation_for(resolved.point(index).0).energy_width())
            .collect::<Result<Vec<_>>>()?;
        let spin_matrix = spin.map(build_spin).transpose()?;
        let direction = direction_matrix(&directions);
        let current = direction.row(0).to_owned();
        let field_1 = direction.row(1).to_owned();
        let field_2 = direction.row(2).to_owned();
        let k_mesh = mesh_array(&params.kmesh);
        let determinant = self.lat.det()?;
        if !determinant.is_normal() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "lat",
                message:
                    "nonlinear Hall normalization requires a finite, nonzero normal lattice volume"
                        .into(),
            });
        }
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let symmetrized = field_symmetry == FieldSymmetry::Symmetrized && field_1 != field_2;
        let compute = |k: ArrayView1<'_, f64>| {
            let data = self.compute_velocity_kernel(
                &k.to_owned(),
                &current,
                &field_1,
                Some(&field_2),
                Gauge::Atom,
                spin_matrix.as_ref(),
            )?;
            // Sorted vertex energies bound every pair gap, even after tracking.
            // Leave headroom for NV <= 4 energy-cut interpolation rounding.
            let band = &data.0.band;
            let spread = band.last().unwrap() - band.first().unwrap();
            let denominator = spread * spread + eta_squared;
            if !denominator.is_finite() || denominator > f64::MAX * (1.0 - 64.0 * f64::EPSILON) {
                return Err(TbError::InvalidResponseParameter {
                    parameter: "nonlinear_hall", message: "extrinsic nonlinear Hall squared gaps overflow; the kernel cannot be certified".into(),
                });
            }
            Ok(data)
        };
        if params.integration == Integration::Direct {
            // Reduce each point immediately: direct integration retains only
            // band energies and one averaged kernel, not all dense matrices.
            let data: Vec<Result<_>> = k_points
                .outer_iter()
                .into_par_iter()
                .map(|k| {
                    let (vertex, _) = compute(k)?;
                    let evaluate = |kernel, diagonal| {
                        let kernel: &Array2<Complex<f64>> = kernel;
                        // Preserve the direct extrinsic denominator, including
                        // finite gaps below the simplex Berry-kernel cutoff.
                        let berry = Array1::from_shape_fn(self.nsta(), |n| {
                            -2.0 * (0..self.nsta())
                                .filter(|&m| m != n)
                                .map(|m| {
                                    let gap = vertex.band[n] - vertex.band[m];
                                    kernel[[n, m]] / (gap * gap + eta * eta)
                                })
                                .sum::<Complex<f64>>()
                                .im
                        });
                        berry * diagonal
                    };
                    let nonlinear = vertex.nonlinear.as_ref().unwrap();
                    let first = evaluate(&vertex.k_ab, &nonlinear.vdiag);
                    let values = if symmetrized {
                        (first + evaluate(&nonlinear.k_ac, &nonlinear.vdiag_b)) * 0.5
                    } else {
                        first
                    };
                    Ok((vertex.band, values))
                })
                .collect();
            let data = data.into_iter().collect::<Result<Vec<_>>>()?;
            let values: Vec<f64> = (0..samples)
                .into_par_iter()
                .map(|index| {
                    let (_, mu, _) = resolved.point(index);
                    let width = widths[index];
                    data.iter()
                        .flat_map(|(band, kernel)| {
                            kernel.iter().zip(band).map(move |(&value, &energy)| {
                                value * fermi_derivative_from_width(energy, mu, width)
                            })
                        })
                        .sum::<f64>()
                        / data.len() as f64
                        / determinant
                })
                .collect();
            return NonlinearHallResult::checked(resolved.axis, Array1::from_vec(values), None);
        }
        let vertices: Vec<Result<_>> = k_points.outer_iter().into_par_iter().map(compute).collect();
        let (mut vertices, mut eigenvectors): (Vec<_>, Vec<_>) = vertices
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .unzip();
        // Tracked once; every sample reuses the labelled vertices.
        global_band_track(&mut eigenvectors, &params.kmesh, |index, permutation| {
            vertices[index] =
                crate::response::tracking::permute_vertex(&vertices[index], permutation);
        });
        drop(eigenvectors);
        let integrate = |vertices: &[VertexKernel],
                         chemical_potentials: &Array1<f64>,
                         width: f64| {
            let (conductivity, unsafe_simplex_count) =
                integrate_dipole_energy_cut_2d(vertices, &k_mesh, chemical_potentials, width, eta);
            (
                conductivity / determinant,
                IntegrationDiagnostics {
                    unsafe_simplex_count,
                },
            )
        };
        let scan =
            |vertices: &mut [VertexKernel]| -> Result<(Array1<f64>, IntegrationDiagnostics)> {
                match &resolved.axis {
                    ResponseAxis::ChemicalPotential(values) => {
                        Ok(integrate(vertices, values, widths[0]))
                    }
                    // Temperature samples redo the cut and the convolution on the
                    // shared vertices.
                    _ => {
                        let mut conductivity = Array1::<f64>::zeros(samples);
                        let mut diagnostics = IntegrationDiagnostics::default();
                        for index in 0..samples {
                            let (_, mu, _) = resolved.point(index);
                            let (value, sample) = integrate(vertices, &array![mu], widths[index]);
                            conductivity[index] = value[0];
                            diagnostics.unsafe_simplex_count = diagnostics
                                .unsafe_simplex_count
                                .max(sample.unsafe_simplex_count);
                        }
                        Ok((conductivity, diagnostics))
                    }
                }
            };
        let (first, diagnostics) = scan(&mut vertices)?;
        let conductivity = if symmetrized {
            // The same eigenbasis already contains K^{ac} and v^b. Exchange
            // only the two consumed kernels, preserving the tracked band order.
            for vertex in &mut vertices {
                let nonlinear = vertex.nonlinear.as_mut().unwrap();
                std::mem::swap(&mut vertex.k_ab, &mut nonlinear.k_ac);
                std::mem::swap(&mut nonlinear.vdiag, &mut nonlinear.vdiag_b);
            }
            let (second, _) = scan(&mut vertices)?;
            (first + second) * 0.5
        } else {
            first
        };
        NonlinearHallResult::checked(resolved.axis, conductivity, Some(diagnostics))
    }

    /// Computes the Berry connection dipole at a single k-point.
    ///
    /// This computes the charge intrinsic NLH kernel `-Q^{ab;c}` with the
    /// argument order `(a, b, c)`.
    ///
    /// ```text
    /// Q^{ab;c}_n = 2 v^c_n G^{ab}_n
    ///             - 1/2 (v^a_n G^{bc}_n + v^b_n G^{ac}_n)
    /// G^{ij}_n = Re sum_{m != n} v^i_nm v^j_mn / (E_n - E_m)^3
    /// ```
    ///
    /// Spin-current dressing would need a spin matrix and a
    /// $\partial_{h_i} G_{jk}$ branch; the public intrinsic entry point is
    /// charge-only, so no such branch exists here.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates.
    /// * `dir_a` - Direction vector for the first field index `a`.
    /// * `dir_b` - Direction vector for the second field index `b`.
    /// * `dir_c` - Direction vector for the current/output index `c`.
    ///
    /// # Returns
    ///
    /// `(omega, band)` where:
    /// - `omega`: `-Q^{ab;c}` per band.
    /// - `band`: Band energies.
    ///
    /// The three direction vectors `(dir_a, dir_b, dir_c)` are treated as
    /// field indices `(a, b, c)` of the intrinsic NLH kernel.
    ///
    /// Callers must pass directions in `(dir_a, dir_b, dir_c)` order.
    /// [`Model::intrinsic_nonlinear_hall`] maps its current-first input
    /// `(current=c, field_1=a, field_2=b)` to this internal order.
    pub(crate) fn berry_connection_dipole_onek(
        &self,
        k_vec: &Array1<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if k_vec.len() != self.dim_r() {
            return Err(TbError::KVectorLengthMismatch {
                expected: self.dim_r(),
                actual: k_vec.len(),
            });
        }
        // Build direction matrix: [dir_a, dir_b, dir_c]
        let directions = {
            let mut d = Array2::<f64>::zeros((3, self.dim_r()));
            d.row_mut(0).assign(dir_a);
            d.row_mut(1).assign(dir_b);
            d.row_mut(2).assign(dir_c);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(k_vec, Gauge::Atom, &directions);
        // v_proj[0] = Σ_d dir_a[d] * v_raw[d]  →  v^a
        // v_proj[1] = Σ_d dir_b[d] * v_raw[d]  →  v^b
        // v_proj[2] = Σ_d dir_c[d] * v_raw[d]  →  v^c

        #[cfg(test)]
        super::config::counters::count_eigen_decomposition();
        let (band, evec) = eigh_full(&hamk, UPLO::Lower)?;
        if band
            .first()
            .zip(band.last())
            .is_some_and(|(low, high)| !(high - low).is_finite())
        {
            return Err(TbError::InvalidResponseParameter {
                parameter: "nonlinear_hall",
                message: "nonlinear Hall band-gap subtraction overflows".into(),
            });
        }
        let ut = evec.mapv(|x| x.conj());
        let uc = evec.t();
        let to_band = |op: &Array2<Complex<f64>>| -> Array2<Complex<f64>> { ut.dot(&op.dot(&uc)) };

        // Transform projected matrices to eigenbasis in one shot per projection.
        let v0: Array2<Complex<f64>> = v_proj.slice(s![0, .., ..]).to_owned();
        let v1: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();
        let v2: Array2<Complex<f64>> = v_proj.slice(s![2, .., ..]).to_owned();
        let v_1 = to_band(&v0); // v^a  (dir_a)
        let v_2 = to_band(&v1); // v^b  (dir_b)
        let v_3 = to_band(&v2); // v^c  (dir_c)
        let mut U0 = Array2::<f64>::zeros((self.nsta(), self.nsta()));
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                U0[[i, j]] = intrinsic_inverse_gap(band[[i]] - band[[j]]);
            }
        }

        let partial_ve_1 = v_1.diag().map(|x| x.re);
        let partial_ve_2 = v_2.diag().map(|x| x.re);
        let partial_ve_3 = v_3.diag().map(|x| x.re);

        // —— SM Eq. (43): charge intrinsic nonlinear Hall ——
        // σ^{ab;c}_{int} = -e³/ħ Σ_n ∫_k f_n
        //   [2 ∂_c G^{ab}_n − 1/2 (∂_a G^{bc}_n + ∂_b G^{ac}_n)]
        //
        // After ibp → integrand:
        //   Q^{ab;c}_n = 2 v^c_n G^{ab}_n − ½ (v^a_n G^{bc}_n + v^b_n G^{ac}_n)
        //
        // With v_1=v^a, v_2=v^b, v_3=v^c and G_12=G^{ab}, G_13=G^{ac}, G_23=G^{bc}:
        //   omega = 2·v_3·G_12 − ½(v_1·G_23 + v_2·G_13)
        //         = 2·v^c·G^{ab} − ½(v^a·G^{bc} + v^b·G^{ac})
        //         = Q^{ab;c}
        // Return −omega = −Q^{ab;c} (overall −e³/ħ factor separate).
        let calc_G = |va: &Array2<Complex<f64>>, vb: &Array2<Complex<f64>>| -> Array1<f64> {
            let U3 = U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0));
            let A = va * &U3;
            let mut G = Array1::<f64>::zeros(self.nsta());
            for i in 0..self.nsta() {
                G[[i]] = A.slice(s![i, ..]).dot(&vb.slice(s![.., i])).re;
            }
            G
        };

        let G_12 = calc_G(&v_1, &v_2); // G^{ab}
        let G_13 = calc_G(&v_1, &v_3); // G^{ac}
        let G_23 = calc_G(&v_2, &v_3); // G^{bc}

        let omega =
            &partial_ve_3 * &G_12 * 2.0 - (&partial_ve_1 * &G_23 + &partial_ve_2 * &G_13) * 0.5;
        Ok((-omega, band))
    }

    /// Parallel version of [`Self::berry_connection_dipole_onek`].
    ///
    /// The three direction vectors `(dir_a, dir_b, dir_c)` are passed directly
    /// to the one‑k‑point kernel — see its docstring for the index convention.
    pub(crate) fn berry_connection_dipole(
        &self,
        k_vec: &Array2<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        for (name, dir) in [("dir_a", dir_a), ("dir_b", dir_b), ("dir_c", dir_c)] {
            if dir.len() != self.dim_r() {
                return Err(TbError::DimensionMismatch {
                    context: name.into(),
                    expected: self.dim_r(),
                    found: dir.len(),
                });
            }
        }
        if k_vec.ncols() != self.dim_r() {
            return Err(TbError::DimensionMismatch {
                context: "k_vec".into(),
                expected: self.dim_r(),
                found: k_vec.ncols(),
            });
        }
        let nk = k_vec.len_of(Axis(0));

        let results: Vec<Result<_>> = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| self.berry_connection_dipole_onek(&x.to_owned(), dir_a, dir_b, dir_c))
            .collect();
        let results: Vec<_> = results.into_iter().collect::<Result<_>>()?;

        let mut omega_arrays = Vec::with_capacity(nk);
        let mut band_arrays = Vec::with_capacity(nk);
        for (omega_one, band_one) in results {
            omega_arrays.push(omega_one);
            band_arrays.push(band_one);
        }

        let from_vecs = |vecs: Vec<Array1<f64>>| -> Result<Array2<f64>> {
            Array2::from_shape_vec((nk, self.nsta()), vecs.into_iter().flatten().collect())
                .map_err(TbError::from)
        };
        let omega = from_vecs(omega_arrays)?;
        let band = from_vecs(band_arrays)?;
        Ok((omega, band))
    }

    /// Evaluate current-first intrinsic nonlinear Hall conductivity.
    ///
    /// `directions` rows are `(current, field_1, field_2)`. The response is
    /// charge-current only and broadens no denominator, so neither `spin` nor
    /// `eta_ev` appears in this signature, and `field_symmetry` does not
    /// apply. `params` carries the thermodynamic `conditions` (at most one
    /// axis sampled), the `kmesh` and `integration`, and
    /// `conditions.omega_ev` must be `Sampling::Fixed(0.0)` because the
    /// response is DC.
    /// Both paths omit interband gaps at or below `1e-10` eV.
    ///
    /// Direct integration samples `-df/dE` on k-points, so every sample must
    /// have a positive thermal energy `k_B T` with a finite Fermi-window peak
    /// `0.25 / (k_B T)`. Zero widths or overflowing peaks reject the whole call
    /// before any k-mesh work; subnormal widths with finite peaks are accepted.
    /// Energy-cut mode evaluates the
    /// zero-temperature Fermi surface exactly within the simplex interpolation
    /// and also accepts finite thermal widths. Eigenstates, velocity kernels and
    /// band tracking are prepared once and reused by every sample.
    pub fn intrinsic_nonlinear_hall(
        &self,
        params: &Parameters<DIM>,
        directions: [[f64; DIM]; 3],
    ) -> Result<NonlinearHallResult> {
        let resolved = params.validate_grid_response()?;
        resolved.require_dc()?;
        self.validate()?;
        validate_direction_values(&directions)?;
        if params.integration == Integration::Simplex {
            return Err(TbError::InvalidResponseParameter {
                parameter: "integration",
                message: "intrinsic_nonlinear_hall supports Integration::Direct or EnergyCut, not Simplex".into(),
            });
        }
        if params.integration == Integration::EnergyCut {
            validate_sorted(&resolved.chemical_potentials(), "mu_ev")?;
            if DIM != 2 && DIM != 3 {
                return Err(TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2, 3],
                });
            }
        } else {
            // Reject the whole call, before any k-mesh work, if any sample
            // has zero thermal width or an overflowing Fermi-window peak.
            resolved.require_positive_temperature()?;
        }
        let samples = resolved.len();
        let widths: Vec<f64> = (0..samples)
            .map(|index| occupation_for(resolved.point(index).0).energy_width())
            .collect::<Result<Vec<_>>>()?;
        let k_mesh = mesh_array(&params.kmesh);
        let determinant = self.lat.det()?;
        if !determinant.is_normal() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "lat",
                message:
                    "nonlinear Hall normalization requires a finite, nonzero normal lattice volume"
                        .into(),
            });
        }
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let direction = direction_matrix(&directions);
        let current = direction.row(0).to_owned();
        let field_1 = direction.row(1).to_owned();
        let field_2 = direction.row(2).to_owned();

        let conductivity = match params.integration {
            Integration::Direct => {
                let (kernel, energies) =
                    self.berry_connection_dipole(&k_points, &field_1, &field_2, &current)?;
                let values: Vec<f64> = (0..samples)
                    .into_par_iter()
                    .map(|index| {
                        let (_, mu, _) = resolved.point(index);
                        let width = widths[index];
                        kernel
                            .iter()
                            .zip(&energies)
                            .map(|(&value, &energy)| {
                                value * fermi_derivative_from_width(energy, mu, width)
                            })
                            .sum::<f64>()
                            / k_points.nrows() as f64
                            / determinant
                    })
                    .collect();
                Array1::from_vec(values)
            }
            Integration::EnergyCut => {
                let vertices: Vec<Result<_>> = (0..k_points.nrows())
                    .into_par_iter()
                    .map(|index| {
                        let data = self.compute_velocity_kernel(
                            &k_points.row(index).to_owned(),
                            &field_1,
                            &field_2,
                            Some(&current),
                            Gauge::Atom,
                            None,
                        )?;
                        if data
                            .0
                            .band
                            .first()
                            .zip(data.0.band.last())
                            .is_some_and(|(low, high)| !(high - low).is_finite())
                        {
                            return Err(TbError::InvalidResponseParameter {
                                parameter: "nonlinear_hall",
                                message: "nonlinear Hall band-gap subtraction overflows".into(),
                            });
                        }
                        Ok(data)
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
                        crate::response::tracking::permute_vertex(&vertices[index], permutation);
                });
                drop(eigenvectors);
                let integrate = |chemical_potentials: &Array1<f64>, width: f64| -> Array1<f64> {
                    let values = match DIM {
                        2 => super::energy_cut::integrate_intrinsic_cut_2d(
                            &vertices,
                            &k_mesh,
                            chemical_potentials,
                            width,
                        ),
                        3 => super::energy_cut::integrate_intrinsic_cut_3d(
                            &vertices,
                            &k_mesh,
                            chemical_potentials,
                            width,
                        ),
                        _ => unreachable!("validated before energy-cut integration"),
                    };
                    values / determinant
                };
                match &resolved.axis {
                    ResponseAxis::ChemicalPotential(values) => integrate(values, widths[0]),
                    // Temperature samples redo the cut and the convolution on
                    // the shared vertices.
                    _ => {
                        let mut conductivity = Array1::<f64>::zeros(samples);
                        for index in 0..samples {
                            let (_, mu, _) = resolved.point(index);
                            conductivity[index] = integrate(&array![mu], widths[index])[0];
                        }
                        conductivity
                    }
                }
            }
            Integration::Simplex => unreachable!("rejected during validation"),
        };

        NonlinearHallResult::checked(resolved.axis, conductivity, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::config::Conditions;

    fn tilted_qwz() -> Model<false, 2> {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![1.0, -1.0], None);
        for r in [array![1, 0], array![0, 1]] {
            model.add_hop(0.5, 0, 0, &r, None);
            model.add_hop(-0.5, 1, 1, &r, None);
        }
        for state in 0..2 {
            model.add_hop(Complex::new(0.0, -0.15), state, state, &array![1, 0], None);
        }
        for (r, amplitude) in [
            (array![1, 0], Complex::new(0.0, -0.5)),
            (array![-1, 0], Complex::new(0.0, 0.5)),
            (array![0, 1], Complex::new(-0.5, 0.0)),
            (array![0, -1], Complex::new(0.5, 0.0)),
        ] {
            model.add_hop(amplitude, 0, 1, &r, None);
        }
        model
    }

    #[test]
    fn nonlinear_hall_rejects_nonfinite_direct_and_cut_outputs() {
        let model = tilted_qwz();
        for (integration, temperature, mu, mesh, scale) in [
            (Integration::Direct, 300.0, 3.0, [1, 1], 1e104),
            (Integration::Direct, 30.0, 3.0, [1, 1], 1e103),
            (Integration::EnergyCut, 0.0, 1.5, [4, 4], 1e104),
        ] {
            let params = Parameters {
                conditions: Conditions::fixed(temperature, mu, 0.0),
                kmesh: mesh,
                integration,
            };
            let x = [scale, 0.0];
            let y = [0.0, scale];
            // All raw primitives are finite; overflow occurs later in the
            // dipole, Fermi weighting or integration, not the existing guard.
            let (vertex, _) = model
                .compute_velocity_kernel(
                    &array![0.0, 0.0],
                    &array![scale, 0.0],
                    &array![0.0, scale],
                    Some(&array![scale, 0.0]),
                    Gauge::Atom,
                    None,
                )
                .unwrap();
            assert!(
                vertex
                    .k_ab
                    .iter()
                    .all(|v| v.re.is_finite() && v.im.is_finite())
            );
            let intrinsic = model.intrinsic_nonlinear_hall(&params, [x, y, y]);
            for symmetry in [FieldSymmetry::Ordered, FieldSymmetry::Symmetrized] {
                let extrinsic =
                    model.extrinsic_nonlinear_hall(&params, [x, y, x], 0.0, None, symmetry);
                eprintln!(
                    "{integration:?}, T={temperature}, scale={scale}: extrinsic={extrinsic:?}, intrinsic={intrinsic:?}"
                );
                assert!(matches!(
                    extrinsic,
                    Err(TbError::InvalidResponseParameter {
                        parameter: "nonlinear_hall",
                        ..
                    })
                ));
            }
            assert!(matches!(
                intrinsic,
                Err(TbError::InvalidResponseParameter {
                    parameter: "nonlinear_hall",
                    ..
                })
            ));
        }
    }

    #[test]
    fn nonlinear_hall_rejects_unreliable_volume_and_overflowed_denominators() {
        let params = Parameters {
            conditions: Conditions::fixed(300.0, 3.0, 0.0),
            kmesh: [1, 1],
            integration: Integration::Direct,
        };
        for lattice_scale in [1e-200, 1e-160, 1e200] {
            let mut model = tilted_qwz();
            model.lat *= lattice_scale;
            model.validate().unwrap(); // Finite inverse does not certify volume.
            assert!(matches!(
                model.extrinsic_nonlinear_hall(
                    &params,
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    0.0,
                    None,
                    FieldSymmetry::Ordered
                ),
                Err(TbError::InvalidResponseParameter {
                    parameter: "lat",
                    ..
                })
            ));
            assert!(matches!(
                model.intrinsic_nonlinear_hall(&params, [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
                Err(TbError::InvalidResponseParameter {
                    parameter: "lat",
                    ..
                })
            ));
        }
        let model = tilted_qwz();
        // Large but representable eta² must still produce the finite analytic
        // answer, rather than a blanket rejection of large scales.
        let s = 1e104_f64;
        let eta = 1e154_f64;
        let value = model
            .extrinsic_nonlinear_hall(
                &params,
                [[s, 0.0], [0.0, s], [s, 0.0]],
                eta,
                None,
                FieldSymmetry::Ordered,
            )
            .unwrap()
            .single()
            .unwrap();
        let width = occupation_for(300.0).energy_width().unwrap();
        let window = fermi_derivative_from_width(-3.0, 3.0, width)
            - fermi_derivative_from_width(3.0, 3.0, width);
        let expected = 0.6 * (s / eta).powi(2) * s * window;
        assert!((value / expected - 1.0).abs() < 1e-12);
        for integration in [Integration::Direct, Integration::EnergyCut] {
            let mut p = params.clone();
            p.integration = integration;
            assert!(matches!(
                model.extrinsic_nonlinear_hall(
                    &p,
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    1e155,
                    None,
                    FieldSymmetry::Ordered
                ),
                Err(TbError::InvalidResponseParameter {
                    parameter: "eta_ev",
                    ..
                })
            ));
            let mut huge = tilted_qwz();
            huge.ham.mapv_inplace(|value| value * 1e155);
            p.conditions = Conditions::fixed(1e159, 3e155, 0.0);
            let x = [1e-52, 0.0];
            let y = [0.0, 1e-52];
            assert!(matches!(
                huge.extrinsic_nonlinear_hall(&p, [x, y, x], 0.0, None, FieldSymmetry::Ordered),
                Err(TbError::InvalidResponseParameter {
                    parameter: "nonlinear_hall",
                    ..
                })
            ));
            huge = tilted_qwz();
            huge.ham.mapv_inplace(|value| value * 5e307);
            p.conditions = Conditions::fixed(300.0, 1.5e308, 0.0);
            let x = [1e-206, 0.0];
            let y = [0.0, 1e-206];
            assert!(matches!(
                huge.intrinsic_nonlinear_hall(&p, [x, y, y]),
                Err(TbError::InvalidResponseParameter {
                    parameter: "nonlinear_hall",
                    ..
                })
            ));
        }
    }

    #[test]
    fn nonlinear_hall_rejects_a_later_overflowed_sample() {
        let model = tilted_qwz();
        let s = 1e103;
        let x = [s, 0.0];
        let y = [0.0, s];
        let mut params = Parameters {
            conditions: Conditions::fixed(300.0, 3.0, 0.0),
            kmesh: [1, 1],
            integration: Integration::Direct,
        };
        assert!(
            model
                .extrinsic_nonlinear_hall(&params, [x, y, x], 0.0, None, FieldSymmetry::Ordered)
                .unwrap()
                .single()
                .unwrap()
                .is_finite()
        );
        assert!(
            model
                .intrinsic_nonlinear_hall(&params, [x, y, y])
                .unwrap()
                .single()
                .unwrap()
                .is_finite()
        );
        params.conditions.t_kelvin = super::super::config::Sampling::Values(array![300.0, 30.0]);
        for result in [
            model.extrinsic_nonlinear_hall(&params, [x, y, x], 0.0, None, FieldSymmetry::Ordered),
            model.intrinsic_nonlinear_hall(&params, [x, y, y]),
        ] {
            assert!(
                matches!(result, Err(TbError::InvalidResponseParameter { parameter: "nonlinear_hall", message }) if message.contains("sample 1"))
            );
        }
    }

    #[test]
    fn intrinsic_three_dimensional_cut_preserves_finite_scaling_and_rejects_overflow() {
        let source = tilted_qwz();
        let mut model =
            Model::<false, 3>::tb_model(Array2::eye(3), Array2::zeros((2, 3)), None).unwrap();
        model.ham = source.ham;
        model.hamR = Array2::from_shape_fn((source.hamR.nrows(), 3), |(row, col)| {
            if col < 2 { source.hamR[[row, col]] } else { 0 }
        });
        let params = Parameters {
            conditions: Conditions::fixed(0.0, 1.5, 0.0),
            kmesh: [4, 4, 2],
            integration: Integration::EnergyCut,
        };
        let directions = |s| [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, s, 0.0]];
        let base = model
            .intrinsic_nonlinear_hall(&params, directions(1.0))
            .unwrap()
            .single()
            .unwrap();
        assert!(base.is_finite() && base != 0.0);
        let large = model
            .intrinsic_nonlinear_hall(&params, directions(1e100))
            .unwrap()
            .single()
            .unwrap();
        assert!((large / (base * 1e300) - 1.0).abs() < 1e-10);
        assert!(matches!(
            model.intrinsic_nonlinear_hall(&params, directions(1e104)),
            Err(TbError::InvalidResponseParameter {
                parameter: "nonlinear_hall",
                ..
            })
        ));
    }

    #[test]
    fn nonlinear_hall_preserves_finite_cubic_scaling() {
        let model = tilted_qwz();
        let params = Parameters {
            conditions: Conditions::fixed(300.0, 3.0, 0.0),
            kmesh: [1, 1],
            integration: Integration::Direct,
        };
        let width = occupation_for(300.0).energy_width().unwrap();
        let window = fermi_derivative_from_width(-3.0, 3.0, width)
            - fermi_derivative_from_width(3.0, 3.0, width);
        for scale in [1.0_f64, 1e100] {
            let x = [scale, 0.0];
            let y = [0.0, scale];
            let cubic = scale.powi(3);
            for (symmetry, weight) in [
                (FieldSymmetry::Ordered, 1.0),
                (FieldSymmetry::Symmetrized, 0.5),
            ] {
                let value = model
                    .extrinsic_nonlinear_hall(&params, [x, y, x], 0.0, None, symmetry)
                    .unwrap()
                    .single()
                    .unwrap();
                let expected = window * (0.3 / 18.0) * cubic * weight;
                assert!(
                    (value / expected - 1.0).abs() < 1e-12,
                    "{value} vs {expected}"
                );
            }
            let value = model
                .intrinsic_nonlinear_hall(&params, [x, y, y])
                .unwrap()
                .single()
                .unwrap();
            let expected = window * (0.6 / 216.0) * cubic;
            assert!(
                (value / expected - 1.0).abs() < 1e-12,
                "{value} vs {expected}"
            );
        }
    }
}
