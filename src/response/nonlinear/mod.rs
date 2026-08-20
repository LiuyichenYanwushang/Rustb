//! # Nonlinear response: Berry dipole, intrinsic & extrinsic NLH
//!
//! ## Extrinsic NLH — Berry curvature dipole (BCD)
//!
//! $$\chi^{\rm ext}_{abc}(\mu,T) =
//!   \sum_n \int_{\rm BZ} \left(-\frac{\partial f}{\partial E_n}\right)
//!   v^c_n(\mathbf{k})\Omega^{ab}_n(\mathbf{k})d\mathbf{k}$$
//!
//! The BCD is **TR‑even** ($D_{TR}=D$) — survives in TR‑symmetric, P‑broken systems.
//! Under time reversal: $v^c\to -v^c$, $\Omega^{ab}\to -\Omega^{ab}$, so the
//! product $v^c\Omega^{ab}$ is invariant.
//!
//! ## Intrinsic NLH — Berry connection dipole
//!
//! $$\sigma^{ab;c}_{\rm int}(\mu,T) = -\frac{e^3}{\hbar}
//!   \sum_n \int_{\rm BZ} (-\partial f/\partial E_n)
//!   \bigl[2v^c_n G^{ab}_n - \tfrac12(v^a_n G^{bc}_n + v^b_n G^{ac}_n)\bigr]d\mathbf{k}$$
//!
//! where $G^{ij}_n = \operatorname{Re}\sum_{m\ne n} K^{ij}_{nm} / (E_n-E_m)^3$.
//! The intrinsic NLH is **TR‑odd** ($\sigma_{TR}=-\sigma$) — requires both
//! $\mathcal P$ and $\mathcal T$ breaking.
//!
//! ## API
//!
//! | Method | Path | Formula |
//! |--------|------|---------|
//! | `extrinsic_nonlinear_hall` | direct sum or energy cut | $\chi^{\rm ext}$ |
//! | `intrinsic_nonlinear_hall` | direct sum or energy cut | $\sigma_{\rm int}$ |

use ndarray::prelude::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::SpinDirection;
use crate::error::{Result, TbError};
use crate::math::anti_comm;
use crate::thermodynamics::fermi_derivative_from_width;

use super::config::{
    FieldSymmetry, Integration, IntegrationDiagnostics, Parameters, mesh_array,
    parameters_occupation, validate_broadening, validate_sorted,
};
use super::energy_cut::integrate_dipole_energy_cut_2d;
use super::helpers::build_spin_matrix;
use super::tracking::global_band_track;
use super::types::VertexKernel;

/// Conductivity evaluated on a chemical-potential grid.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearHallResult {
    /// Chemical potentials copied from the input configuration.
    pub chemical_potentials: Array1<f64>,
    /// Nonlinear Hall response at every chemical potential.
    pub conductivity: Array1<f64>,
    /// Algorithm diagnostics when exposed by the selected energy-cut path.
    pub diagnostics: Option<IntegrationDiagnostics>,
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
    /// * `spin` - `Option<SpinDirection>`; `None` = charge current.
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// `(omega_n, band)` where `omega_n` contains $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$
    /// for each band, and `band` contains the band energies.
    pub(crate) fn berry_curvature_dipole_n_onek(
        &self,
        k_vec: &Array1<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: Option<SpinDirection>,
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
        let J: Array2<Complex<f64>> = if SPIN {
            let X = build_spin_matrix(self.norb(), spin);
            anti_comm(&X, &v_proj.slice(s![0, .., ..])) * 0.5
        } else {
            if let Some(direction) = spin {
                return Err(TbError::SpinNotAllowed(direction));
            }
            v_proj.slice(s![0, .., ..]).to_owned()
        };
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();
        let v0: Array2<Complex<f64>> = v_proj.slice(s![2, .., ..]).to_owned();

        let (band, evec) = hamk.eigh(UPLO::Lower)?;
        let evec_conj = evec.t();
        let evec = evec.map(|x| x.conj());

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
    /// This is a parallelized version of [`berry_curvature_dipole_n_onek`] for computing
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
    /// * `spin` - `Option<SpinDirection>`; `None` = charge current.
    /// * `eta` - Broadening parameter.
    ///
    /// # Returns
    ///
    /// `(omega, band)` where `omega` has shape `(nk, nsta)` containing
    /// $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$ for each k-point and band,
    /// and `band` has the band energies with the same shape.
    pub(crate) fn berry_curvature_dipole_n(
        &self,
        k_vec: &Array2<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: Option<SpinDirection>,
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
                    spin,
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

    /// Evaluate the Berry-curvature-dipole nonlinear Hall response.
    ///
    /// Reads `kmesh`, `direction` (rank 3), `mu`, `T`, `eta` and `spin` from
    /// the parameter set. This is a DC response: `omega` is ignored.
    /// Direct integration requires a finite thermal width because `-df/dE`
    /// is sampled on k-points. Energy-cut integration supports the exact
    /// zero-temperature limit.
    ///
    /// Direction rows are `(current, field_1, field_2)`. In the internal
    /// kernel this maps to `Ω^{current, field_1} v^{field_2}`: `current` and
    /// `field_1` are the two Berry-curvature indices and `field_2` is the
    /// Fermi-surface velocity index. `FieldSymmetry::Symmetrized` averages
    /// the two orderings of `field_1`/`field_2`.
    ///
    /// The two field indices are combined according to `params.field_symmetry`:
    /// [`FieldSymmetry::Symmetrized`] (the default) averages the two field
    /// permutations, [`FieldSymmetry::Ordered`] returns the raw ordered
    /// kernel.
    pub fn extrinsic_nonlinear_hall(
        &self,
        params: &Parameters<DIM>,
    ) -> Result<NonlinearHallResult> {
        params.validate_rank3()?;
        validate_broadening(params.eta)?;
        let spin = params.spin;
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
            validate_sorted(&params.mu, "mu")?;
            if DIM != 2 {
                return Err(TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2],
                });
            }
        }
        let current = params.direction.row(0).to_owned();
        let field_1 = params.direction.row(1).to_owned();
        let field_2 = params.direction.row(2).to_owned();
        let (first, first_diagnostics) =
            self.extrinsic_nonlinear_hall_component(params, &current, &field_1, &field_2, spin)?;
        let (conductivity, diagnostics) = if params.field_symmetry == FieldSymmetry::Symmetrized
            && field_1 != field_2
        {
            let (second, second_diagnostics) = self
                .extrinsic_nonlinear_hall_component(params, &current, &field_2, &field_1, spin)?;
            let diagnostics = match (first_diagnostics, second_diagnostics) {
                (Some(a), Some(b)) => Some(IntegrationDiagnostics {
                    unsafe_simplex_count: a.unsafe_simplex_count.max(b.unsafe_simplex_count),
                }),
                (a, b) => a.or(b),
            };
            ((first + second) * 0.5, diagnostics)
        } else {
            (first, first_diagnostics)
        };
        Ok(NonlinearHallResult {
            chemical_potentials: params.mu.clone(),
            conductivity,
            diagnostics,
        })
    }

    fn extrinsic_nonlinear_hall_component(
        &self,
        params: &Parameters<DIM>,
        current: &Array1<f64>,
        field_1: &Array1<f64>,
        field_2: &Array1<f64>,
        spin: Option<SpinDirection>,
    ) -> Result<(Array1<f64>, Option<IntegrationDiagnostics>)> {
        let k_mesh = mesh_array(&params.kmesh);
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let width = parameters_occupation(params).energy_width()?;
        let determinant = self.lat.det()?;
        match params.integration {
            Integration::Direct => {
                if width == 0.0 {
                    return Err(TbError::InvalidThermodynamicParameter {
                        parameter: "T",
                        message: "direct nonlinear Hall integration requires a finite temperature"
                            .into(),
                    });
                }
                let (kernel, energies) = self.berry_curvature_dipole_n(
                    &k_points, current, field_1, field_2, spin, params.eta,
                )?;
                let values: Vec<f64> = params
                    .mu
                    .par_iter()
                    .map(|&mu| {
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
                Ok((Array1::from_vec(values), None))
            }
            Integration::EnergyCut => {
                let chemical_potentials = Array1::from_iter(params.mu.iter().copied());
                let vertices: Vec<Result<VertexKernel>> = (0..k_points.nrows())
                    .into_par_iter()
                    .map(|index| {
                        self.compute_velocity_kernel(
                            &k_points.row(index).to_owned(),
                            current,
                            field_1,
                            Some(field_2),
                            Gauge::Atom,
                            spin,
                        )
                    })
                    .collect();
                let mut vertices: Vec<VertexKernel> =
                    vertices.into_iter().collect::<Result<_>>()?;
                global_band_track(&mut vertices, &params.kmesh);
                let (conductivity, unsafe_simplex_count) = integrate_dipole_energy_cut_2d(
                    &vertices,
                    &k_mesh,
                    &chemical_potentials,
                    width,
                    params.eta,
                );
                Ok((
                    conductivity / determinant,
                    Some(IntegrationDiagnostics {
                        unsafe_simplex_count,
                    }),
                ))
            }
            Integration::Simplex => unreachable!("rejected during validation"),
        }
    }

    /// Computes the Berry connection dipole at a single k-point.
    ///
    /// For spinless models, this computes the charge intrinsic NLH kernel
    /// `-Q^{ab;c}` with the argument order `(a, b, c)`.
    ///
    /// ```text
    /// Q^{ab;c}_n = 2 v^c_n G^{ab}_n
    ///             - 1/2 (v^a_n G^{bc}_n + v^b_n G^{ac}_n)
    /// G^{ij}_n = Re sum_{m != n} v^i_nm v^j_mn / (E_n - E_m)^3
    /// ```
    ///
    /// For spinful models (when `spin.is_some()`), this additionally computes
    /// $\partial_{h_i} G_{jk}$, the derivative with respect to the spin field.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates.
    /// * `current_dir` - Direction vector for the first field index `a`.
    /// * `dir_2` - Direction vector for the second field index `b`.
    /// * `dir_3` - Direction vector for the current/output index `c`.
    /// * `spin` - `Option<SpinDirection>`; `None` = charge current.
    ///
    /// # Returns
    ///
    /// `(omega, band, partial_G)` where:
    /// - `omega`: `-Q^{ab;c}` per band for the charge branch.
    /// - `band`: Band energies.
    /// - `partial_G`: $\partial_{h} G$ per band (only `Some` for spinful models, `None` for spinless).
    /// Compute Berry connection dipole integrand at one k-point.
    ///
    /// The three direction vectors `(dir_a, dir_b, dir_c)` are treated as
    /// field indices `(a, b, c)` of the intrinsic NLH kernel.  The charge
    /// branch returns `−Q^{ab;c}` where
    ///
    /// ```text
    /// Q^{ab;c}_n = 2 v^c_n G^{ab}_n − ½(v^a_n G^{bc}_n + v^b_n G^{ac}_n)
    /// G^{ij}_n    = Re Σ_{m≠n} v^i_{nm} v^j_{mn} / (E_n−E_m)³
    /// ```
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
        spin: Option<SpinDirection>,
    ) -> Result<(Array1<f64>, Array1<f64>, Option<Array1<f64>>)> {
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

        let (band, evec) = hamk.eigh(UPLO::Lower)?;
        let ut = evec.t();
        let uc = evec.map(|x| x.conj());
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
                if (band[[i]] - band[[j]]).abs() < 1e-5 {
                    U0[[i, j]] = 0.0;
                } else {
                    U0[[i, j]] = 1.0 / (band[[i]] - band[[j]]);
                }
            }
        }

        let partial_ve_1 = v_1.diag().map(|x| x.re);
        let partial_ve_2 = v_2.diag().map(|x| x.re);
        let partial_ve_3 = v_3.diag().map(|x| x.re);

        // Only enter spin branch when model is spinful AND spin requested
        if SPIN && spin.is_some() {
            // Anti-commute on projected raw matrices (once each, not per-direction)
            let X = build_spin_matrix(self.norb(), spin);
            let s_1_raw = anti_comm(&X, &v_proj.slice(s![0, .., ..])) * 0.5;
            let s_2_raw = anti_comm(&X, &v_proj.slice(s![1, .., ..])) * 0.5;
            let s_3_raw = anti_comm(&X, &v_proj.slice(s![2, .., ..])) * 0.5;
            // Transform to eigenbasis
            let s_1 = to_band(&s_1_raw);
            let s_2 = to_band(&s_2_raw);
            let s_3 = to_band(&s_3_raw);
            let G_23: Array1<f64> = {
                let A = &v_2 * (U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0)));
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&v_3.slice(s![.., i])).re * 2.0
                }
                G
            };
            let G_13_h: Array1<f64> = {
                let A = &s_1 * (U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0)));
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&v_3.slice(s![.., i])).re * 2.0
                }
                G
            };
            let partial_s_1 = s_1.diag().map(|x| x.re);
            let _partial_s_2 = s_2.diag().map(|x| x.re);
            let _partial_s_3 = s_3.diag().map(|x| x.re);
            let partial_G: Array1<f64> = {
                let mut A = Array1::<Complex<f64>>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    for j in 0..self.nsta() {
                        A[[i]] += 3.0
                            * (partial_s_1[[i]] - partial_s_1[[j]])
                            * v_2[[i, j]]
                            * v_3[[j, i]]
                            * U0[[i, j]].powi(4);
                    }
                }
                let mut B = Array1::<Complex<f64>>::zeros(self.nsta());
                for n in 0..self.nsta() {
                    for n1 in 0..self.nsta() {
                        for n2 in 0..self.nsta() {
                            B[[n]] += s_1[[n, n2]]
                                * (v_2[[n2, n1]] * v_3[[n1, n]] + v_3[[n2, n1]] * v_2[[n1, n]])
                                * U0[[n, n1]].powi(3)
                                * U0[[n, n2]];
                        }
                    }
                }
                let mut C = Array1::<Complex<f64>>::zeros(self.nsta());
                for n in 0..self.nsta() {
                    for n1 in 0..self.nsta() {
                        for n2 in 0..self.nsta() {
                            C[[n]] += s_1[[n1, n2]]
                                * (v_2[[n2, n]] * v_3[[n, n1]] + v_3[[n2, n]] * v_2[[n, n1]])
                                * U0[[n, n1]].powi(3)
                                * U0[[n1, n2]];
                        }
                    }
                }
                2.0 * (A - B - C).map(|x| x.re)
            };
            return Ok((
                partial_s_1 * G_23 - partial_ve_2 * G_13_h,
                band,
                Some(partial_G),
            ));
        } else {
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
            return Ok((-omega, band, None));
        }
    }

    /// Parallel version of [`berry_connection_dipole_onek`].
    ///
    /// The three direction vectors `(dir_a, dir_b, dir_c)` are passed directly
    /// to the one‑k‑point kernel — see its docstring for the index convention.
    pub(crate) fn berry_connection_dipole(
        &self,
        k_vec: &Array2<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        spin: Option<SpinDirection>,
    ) -> Result<(Array2<f64>, Array2<f64>, Option<Array2<f64>>)> {
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
            .map(|x| self.berry_connection_dipole_onek(&x.to_owned(), dir_a, dir_b, dir_c, spin))
            .collect();
        let results: Vec<_> = results.into_iter().collect::<Result<_>>()?;

        let mut omega_arrays = Vec::with_capacity(nk);
        let mut band_arrays = Vec::with_capacity(nk);
        let mut partial_g_arrays = Vec::with_capacity(nk);
        let mut has_partial_g = false;
        for (omega_one, band_one, partial_g) in results {
            omega_arrays.push(omega_one);
            band_arrays.push(band_one);
            if let Some(partial_g) = partial_g {
                has_partial_g = true;
                partial_g_arrays.push(partial_g);
            }
        }

        let from_vecs = |vecs: Vec<Array1<f64>>| -> Result<Array2<f64>> {
            Array2::from_shape_vec((nk, self.nsta()), vecs.into_iter().flatten().collect())
                .map_err(TbError::from)
        };
        let omega = from_vecs(omega_arrays)?;
        let band = from_vecs(band_arrays)?;
        let partial_g = if has_partial_g {
            Some(from_vecs(partial_g_arrays)?)
        } else {
            None
        };
        Ok((omega, band, partial_g))
    }

    /// Evaluate current-first intrinsic nonlinear Hall conductivity.
    ///
    /// Reads `kmesh`, `direction` (rank 3), `mu` and `T` from the parameter
    /// set; `eta`, `omega`, `spin` and `field_symmetry` are ignored. The
    /// response is charge-current only. Direct integration requires a finite
    /// thermal width. Energy-cut mode evaluates the zero-temperature Fermi
    /// surface exactly within the simplex interpolation and also accepts
    /// finite thermal widths.
    pub fn intrinsic_nonlinear_hall(
        &self,
        params: &Parameters<DIM>,
    ) -> Result<NonlinearHallResult> {
        params.validate_rank3()?;
        if params.spin.is_some() {
            eprintln!(
                "Warning: intrinsic_nonlinear_hall is charge-only; the requested spin direction is ignored."
            );
        }
        if params.integration == Integration::Simplex {
            return Err(TbError::InvalidResponseParameter {
                parameter: "integration",
                message: "intrinsic_nonlinear_hall supports Integration::Direct or EnergyCut, not Simplex".into(),
            });
        }
        if params.integration == Integration::EnergyCut {
            validate_sorted(&params.mu, "mu")?;
            if DIM != 2 && DIM != 3 {
                return Err(TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2, 3],
                });
            }
        }
        let k_mesh = mesh_array(&params.kmesh);
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let width = parameters_occupation(params).energy_width()?;
        let determinant = self.lat.det()?;
        let current = params.direction.row(0).to_owned();
        let field_1 = params.direction.row(1).to_owned();
        let field_2 = params.direction.row(2).to_owned();

        let conductivity = match params.integration {
            Integration::Direct => {
                if width == 0.0 {
                    return Err(TbError::InvalidThermodynamicParameter {
                        parameter: "T",
                        message: "direct nonlinear Hall integration requires a finite temperature"
                            .into(),
                    });
                }
                let (kernel, energies, _) =
                    self.berry_connection_dipole(&k_points, &field_1, &field_2, &current, None)?;
                let values: Vec<f64> = params
                    .mu
                    .par_iter()
                    .map(|&mu| {
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
                let chemical_potentials = Array1::from_iter(params.mu.iter().copied());
                let vertices: Vec<Result<VertexKernel>> = (0..k_points.nrows())
                    .into_par_iter()
                    .map(|index| {
                        self.compute_velocity_kernel(
                            &k_points.row(index).to_owned(),
                            &field_1,
                            &field_2,
                            Some(&current),
                            Gauge::Atom,
                            None,
                        )
                    })
                    .collect();
                let mut vertices: Vec<VertexKernel> =
                    vertices.into_iter().collect::<Result<_>>()?;
                global_band_track(&mut vertices, &params.kmesh);
                let values = match DIM {
                    2 => super::energy_cut::integrate_intrinsic_cut_2d(
                        &vertices,
                        &k_mesh,
                        &chemical_potentials,
                        width,
                    ),
                    3 => super::energy_cut::integrate_intrinsic_cut_3d(
                        &vertices,
                        &k_mesh,
                        &chemical_potentials,
                        width,
                    ),
                    _ => unreachable!("validated before energy-cut integration"),
                };
                values / determinant
            }
            Integration::Simplex => unreachable!("rejected during validation"),
        };

        Ok(NonlinearHallResult {
            chemical_potentials: params.mu.clone(),
            conductivity,
            diagnostics: None,
        })
    }
}
