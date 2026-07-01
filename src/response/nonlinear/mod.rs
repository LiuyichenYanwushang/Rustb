//! # Nonlinear response: Berry dipole, intrinsic & extrinsic NLH
//!
//! ## Extrinsic NLH — Berry curvature dipole
//!
//! $$\chi^{\rm ext}_{abc}(\mu,T) =
//!   \sum_n \int_{\rm BZ} \left(-\frac{\partial f}{\partial E_n}\right)
//!   v^c_n(\mathbf{k})\Omega^{ab}_n(\mathbf{k})d\mathbf{k}$$
//!
//! where $f(E)=1/(1+e^{\beta(E-\mu)})$, $\Omega^{ab}_n = -2\operatorname{Im}G^{ab}_n$.
//! The symmetrised form is $\chi^{\rm ext}_{c,ab}=\frac12(S_{ab;c}+S_{ac;b})$.
//!
//! ## Intrinsic NLH — Berry connection dipole
//!
//! $$\sigma^{ab;c}_{\rm int}(\mu,T) = -\frac{e^3}{\hbar}
//!   \sum_n \int_{\rm BZ} f_n
//!   \bigl[2\partial_c G^{ab}_n - \tfrac12(\partial_a G^{bc}_n + \partial_b G^{ac}_n)\bigr]d\mathbf{k}$$
//!
//! After integration by parts $\int f\partial_i G = -\int(\partial_i f)G$, the kernel is
//!
//! $$Q^{ab;c}_n = 2v^c_n G^{ab}_n - \tfrac12\bigl(v^a_n G^{bc}_n + v^b_n G^{ac}_n\bigr)$$
//!
//! The **Berry curvature dipole** (energy‑cut path) computes $D^{ab;c}$ via
//! analytic iso‑energy line cuts on 2D triangles.  Works at any $T$,
//! including $T=0$.  Requires `dir_c` so that `v^c_n$ is available.
//!
//! ## API
//!
//! | Method | Path | Formula |
//! |--------|------|---------|
//! | `berry_curvature_dipole_energy_cut` | energy‑cut | $D^{ab;c}$ (2D only) |
//! | `Nonlinear_Hall_conductivity_Extrinsic` | direct sum | $\chi^{\rm ext}$ |
//! | `Nonlinear_Hall_conductivity_Extrinsic_sym` | direct sum | symmetrised $\chi^{\rm ext}$ |
//! | `Nonlinear_Hall_conductivity_Intrinsic` | direct sum | $\sigma_{\rm int}$ |
//! | `berry_connection_dipole` | per‑k‑point | $Q^{ab;c}_n$ at each k |

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::SpinDirection;
use crate::error::Result;
use crate::math::anti_comm;

use super::energy_cut::integrate_dipole_energy_cut_2d;
use super::helpers::build_spin_matrix;
use super::tracking::global_band_track;
use super::traits::BerryCurvature;
use super::types::VertexKernel;

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Berry-curvature dipole via analytic 2D energy-cut integration.
    ///
    /// This avoids volume quadrature over the narrow finite-temperature Fermi
    /// window.  Inside each triangle, `E_n(k)` and
    /// `A_n(k)=v^c_n(k)Omega^{ab}_n(k)` are linearly interpolated; the
    /// `delta(E_n-mu)` line cut is evaluated analytically and convolved with
    /// `beta f(1-f)` for finite `T`.
    ///
    /// Currently implemented for 2D k-meshes only.
    pub fn berry_curvature_dipole_energy_cut(
        &self,
        k_mesh: &Array1<usize>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        eta: f64,
    ) -> Result<(Array1<f64>, usize)> {
        assert_eq!(
            k_mesh.len(),
            2,
            "berry_curvature_dipole_energy_cut currently supports 2D only"
        );
        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let gauge = Gauge::Atom;

        let mut all_pts: Vec<VertexKernel> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_velocity_kernel(&kv, dir_a, dir_b, Some(dir_c), gauge, None)
            })
            .collect();
        global_band_track(&mut all_pts, k_mesh.as_slice().unwrap());

        let (dipole, unsafe_count) = integrate_dipole_energy_cut_2d(&all_pts, k_mesh, mu, T, eta);
        let det = self.lat.det().unwrap();
        Ok((dipole / det, unsafe_count))
    }

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
    /// * `og` - Frequency $\omega$ (for the energy denominator).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// `(omega_n, band)` where `omega_n` contains $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$
    /// for each band, and `band` contains the band energies.
    pub fn berry_curvature_dipole_n_onek(
        &self,
        k_vec: &Array1<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();
        // Build direction matrix: [current_dir, dir_2, dir_3]
        let directions = {
            let mut d = Array2::<f64>::zeros((3, self.dim_r()));
            d.row_mut(0).assign(current_dir);
            d.row_mut(1).assign(dir_2);
            d.row_mut(2).assign(dir_3);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(&k_vec, Gauge::Atom, &directions);
        // v_proj[0] = Σ_d current_dir[d] * v_raw[d]  → J
        // v_proj[1] = Σ_d dir_2[d] * v_raw[d]        → v
        // v_proj[2] = Σ_d dir_3[d] * v_raw[d]        → v0
        let J: Array2<Complex<f64>> = if SPIN {
            let X = build_spin_matrix(self.norb(), spin);
            anti_comm(&X, &v_proj.slice(s![0, .., ..])) * 0.5
        } else {
            if spin.is_some() {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            v_proj.slice(s![0, .., ..]).to_owned()
        };
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();
        let v0: Array2<Complex<f64>> = v_proj.slice(s![2, .., ..]).to_owned();

        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
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
                    U0[[i, j]] = 1.0 / ((band[[i]] - band[[j]]).powi(2) - (og + li * eta).powi(2));
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
        (omega_n, band)
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
    /// * `og` - Frequency $\omega$.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter.
    ///
    /// # Returns
    ///
    /// `(omega, band)` where `omega` has shape `(nk, nsta)` containing
    /// $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$ for each k-point and band,
    /// and `band` has the band energies with the same shape.
    ///
    /// # Panics
    ///
    /// Panics if any of `current_dir`, `dir_2`, or `dir_3` has length different from `self.dim_r()`.
    pub fn berry_curvature_dipole_n(
        &self,
        k_vec: &Array2<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        if current_dir.len() != self.dim_r()
            || dir_2.len() != self.dim_r()
            || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));
        let (omega, band): (Vec<_>, Vec<_>) = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let (omega_one, band) = self.berry_curvature_dipole_n_onek(
                    &x.to_owned(),
                    &current_dir,
                    &dir_2,
                    &dir_3,
                    og,
                    spin,
                    eta,
                );
                (omega_one, band)
            })
            .collect();
        let omega =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), omega.into_iter().flatten().collect())
                .unwrap();
        let band =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), band.into_iter().flatten().collect())
                .unwrap();
        (omega, band)
    }

    /// Computes the unsymmetrized extrinsic nonlinear Hall kernel via the Berry
    /// curvature dipole.
    ///
    /// This integrates the Berry-curvature-dipole kernel over the Brillouin zone:
    /// $$ S_{\alpha\beta;\gamma} = \int \dd\mathbf k \sum_n
    ///    \left(-\pdv{f_n}{\varepsilon}\right) \partial_\gamma \varepsilon_{n\mathbf k}
    ///    \Omega_{n,\alpha\beta} $$
    ///
    /// This is not the fully field-symmetrized current-first tensor
    /// $\chi^{\rm ext}_{abc}$.  Use
    /// [`Nonlinear_Hall_conductivity_Extrinsic_sym`] for
    /// $\chi^{\rm ext}_{abc}=\frac12(S_{ab;c}+S_{ac;b})$.
    ///
    /// The energy derivative of the Fermi-Dirac distribution is:
    /// $$ -\pdv{f_n}{\varepsilon} = \beta \f{e^{\beta(\varepsilon_n-\mu)}}{(e^{\beta(\varepsilon_n-\mu)}+1)^2}
    ///    = \beta f_n(1-f_n) $$
    ///
    /// **T>0**: direct k‑point sum with Fermi window.
    /// **T=0**: uses a mesh‑broadened Fermi window
    /// `T_eff = max(1, 1/(n_per_dim·k_B))` — not a true δ‑function limit.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction.
    /// * `current_dir`, `dir_2` - Direction vectors for the Berry curvature indices $\alpha, \beta$.
    /// * `dir_3` - Direction vector for the velocity / Fermi-surface index $\gamma$.
    /// * `mu` - Array of chemical potential values (in eV).
    /// * `T` - Temperature (in K).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// The extrinsic nonlinear Hall conductivity for each $\mu$ value.
    ///
    pub fn Nonlinear_Hall_conductivity_Extrinsic(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        if current_dir.len() != self.dim_r()
            || dir_2.len() != self.dim_r()
            || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let (omega, band) =
            self.berry_curvature_dipole_n(&kvec, &current_dir, &dir_2, &dir_3, og, spin, eta);
        let omega = omega.into_raw_vec();
        let band = band.into_raw_vec();
        let n_e = mu.len();
        let mut conductivity = Array1::<f64>::zeros(n_e);
        if T != 0.0 {
            let beta = 1.0 / T / (8.617e-5);
            let use_iter = band.iter().zip(omega.iter()).par_bridge();
            conductivity = use_iter
                .fold(
                    || Array1::<f64>::zeros(n_e),
                    |acc, (energy, omega0)| {
                        let f = 1.0 / (beta * (mu - *energy)).mapv(|x| x.exp() + 1.0);
                        acc + &f * (1.0 - &f) * beta * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        } else {
            // T=0: use low-T Fermi window matching k-mesh resolution
            let nk_per_dim = (nk as f64).powf(1.0 / self.dim_r() as f64);
            let T_eff = (1.0 / (nk_per_dim * 8.617e-5)).max(1.0);
            let beta_eff = 1.0 / (T_eff * 8.617e-5);
            let use_iter = band.iter().zip(omega.iter()).par_bridge();
            conductivity = use_iter
                .fold(
                    || Array1::<f64>::zeros(n_e),
                    |acc, (energy, omega0)| {
                        let f = 1.0 / (beta_eff * (mu - *energy)).mapv(|x| x.exp() + 1.0);
                        acc + &f * (1.0 - &f) * beta_eff * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        }
        Ok(conductivity)
    }

    /// Current-first, field-symmetrized extrinsic nonlinear Hall tensor.
    ///
    /// For `j_a = chi_{abc} E_b E_c`, this returns
    /// ```text
    /// chi_ext[a,b,c] = 1/2 * (S_{ab;c} + S_{ac;b})
    /// S_{ab;c} = ∫ (-df/dE) v_c Omega_{ab} dk
    /// ```
    ///
    /// The same physical prefactors omitted by
    /// [`Nonlinear_Hall_conductivity_Extrinsic`] are omitted here.
    pub fn Nonlinear_Hall_conductivity_Extrinsic_sym(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        field_dir_1: &Array1<f64>,
        field_dir_2: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        let term_1 = self.Nonlinear_Hall_conductivity_Extrinsic(
            k_mesh,
            current_dir,
            field_dir_1,
            field_dir_2,
            mu,
            T,
            og,
            spin,
            eta,
        )?;
        let term_2 = self.Nonlinear_Hall_conductivity_Extrinsic(
            k_mesh,
            current_dir,
            field_dir_2,
            field_dir_1,
            mu,
            T,
            og,
            spin,
            eta,
        )?;
        Ok((term_1 + term_2) * 0.5)
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
    /// For spinful models (when `spin != 0`), this additionally computes
    /// $\partial_{h_i} G_{jk}$, the derivative with respect to the spin field.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates.
    /// * `current_dir` - Direction vector for the first field index `a`.
    /// * `dir_2` - Direction vector for the second field index `b`.
    /// * `dir_3` - Direction vector for the current/output index `c`.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
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
    /// E.g. `Nonlinear_Hall_conductivity_Intrinsic` maps its public
    /// `(current=c, field_1=a, field_2=b)` → `(dir_a=a, dir_b=b, dir_c=c)`.
    pub fn berry_connection_dipole_onek(
        &self,
        k_vec: &Array1<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        spin: Option<SpinDirection>,
    ) -> (Array1<f64>, Array1<f64>, Option<Array1<f64>>) {
        // Build direction matrix: [dir_a, dir_b, dir_c]
        let directions = {
            let mut d = Array2::<f64>::zeros((3, self.dim_r()));
            d.row_mut(0).assign(dir_a);
            d.row_mut(1).assign(dir_b);
            d.row_mut(2).assign(dir_c);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(&k_vec, Gauge::Atom, &directions);
        // v_proj[0] = Σ_d dir_a[d] * v_raw[d]  →  v^a
        // v_proj[1] = Σ_d dir_b[d] * v_raw[d]  →  v^b
        // v_proj[2] = Σ_d dir_c[d] * v_raw[d]  →  v^c

        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
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
            let partial_s_2 = s_2.diag().map(|x| x.re);
            let partial_s_3 = s_3.diag().map(|x| x.re);
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
            return (
                (partial_s_1 * G_23 - partial_ve_2 * G_13_h),
                band,
                Some(partial_G),
            );
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
            return (-omega, band, None);
        }
    }

    /// Parallel version of [`berry_connection_dipole_onek`].
    ///
    /// The three direction vectors `(dir_a, dir_b, dir_c)` are passed directly
    /// to the one‑k‑point kernel — see its docstring for the index convention.
    pub fn berry_connection_dipole(
        &self,
        k_vec: &Array2<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        spin: Option<SpinDirection>,
    ) -> (Array2<f64>, Array2<f64>, Option<Array2<f64>>) {
        if dir_a.len() != self.dim_r() || dir_b.len() != self.dim_r() || dir_c.len() != self.dim_r()
        {
            panic!(
                "Wrong, the dir_a or dir_b you input has wrong length, it must equal to dim_r={}, but you input {}, {} and {}",
                self.dim_r(),
                dir_a.len(),
                dir_b.len(),
                dir_c.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));

        if SPIN && spin.is_some() {
            let ((omega, band), partial_G): ((Vec<_>, Vec<_>), Vec<_>) = k_vec
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|x| {
                    let (omega_one, band, partial_G) =
                        self.berry_connection_dipole_onek(&x.to_owned(), dir_a, dir_b, dir_c, spin);
                    let partial_G = partial_G.expect("SPIN && spin.is_some() must return Some");
                    ((omega_one, band), partial_G)
                })
                .collect();

            let omega = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                omega.into_iter().flatten().collect(),
            )
            .unwrap();
            let band = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                band.into_iter().flatten().collect(),
            )
            .unwrap();
            let partial_G = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                partial_G.into_iter().flatten().collect(),
            )
            .unwrap();

            return (omega, band, Some(partial_G));
        } else {
            let (omega, band): (Vec<_>, Vec<_>) = k_vec
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|x| {
                    let (omega_one, band, partial_G) =
                        self.berry_connection_dipole_onek(&x.to_owned(), dir_a, dir_b, dir_c, spin);
                    (omega_one, band)
                })
                .collect();
            let omega = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                omega.into_iter().flatten().collect(),
            )
            .unwrap();
            let band = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                band.into_iter().flatten().collect(),
            )
            .unwrap();
            return (omega, band, None);
        }
    }

    /// Current-first charge intrinsic nonlinear Hall conductivity (k-point
    /// sum).
    ///
    /// ```text
    /// chi_int[c,a,b](μ,T) = σ_int^{ab;c}(μ,T)
    /// σ^{ab;c}_{int}(μ,T) = Σ_n ∫_BZ (−∂f/∂E_n) (−Q^{ab;c}_n(k)) dk
    /// Q^{ab;c}_n = 2 v^c_n G^{ab}_n − ½(v^a_n G^{bc}_n + v^b_n G^{ac}_n)
    /// G^{ij}_n = Re Σ_{m≠n} v^i_{nm} v^j_{mn} / (E_n−E_m)³
    /// ```
    ///
    /// Argument order is current-first: `(current c, field a, field b)`.
    /// Internally this maps to the `sigma^{ab;c}` kernel.
    ///
    /// **T>0**: direct k‑point sum with Fermi window.
    /// **T=0**: uses a mesh‑broadened Fermi window
    /// `T_eff = max(1, 1/(n_per_dim·k_B))` — not a true δ‑function limit.
    ///
    /// Spinful / partial_G branch is not yet correctly implemented.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction.
    /// * `current_dir` - Current/output direction `c`.
    /// * `dir_2`, `dir_3` - Electric-field directions `a,b`.
    /// * `mu` - Array of chemical potential values (in eV).
    /// * `T` - Temperature (in K).
    pub fn Nonlinear_Hall_conductivity_Intrinsic(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
    ) -> Result<Array1<f64>> {
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        // public API (current=c, field_1=a, field_2=b) → helper (dir_a=a, dir_b=b, dir_c=c)
        let (omega, band, _partial_G) =
            self.berry_connection_dipole(&kvec, &dir_2, &dir_3, &current_dir, None);
        let omega = omega.into_raw_vec();
        let omega = Array1::from(omega);
        let band = band.into_raw_vec();
        let band = Array1::from(band);
        let n_e = mu.len();
        let mut conductivity = Array1::<f64>::zeros(n_e);
        if T != 0.0 {
            let beta = 1.0 / T / 8.617e-5;
            let use_iter = band.iter().zip(omega.iter()).par_bridge();
            conductivity = use_iter
                .fold(
                    || Array1::<f64>::zeros(n_e),
                    |acc, (energy, omega0)| {
                        let f = 1.0 / ((beta * (*energy - mu)).mapv(|x| x.exp() + 1.0));
                        acc + &f * (1.0 - &f) * beta * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        } else {
            // T=0: use low-T Fermi window matching k-mesh resolution
            let nk_per_dim = (nk as f64).powf(1.0 / self.dim_r() as f64);
            let T_eff = (1.0 / (nk_per_dim * 8.617e-5)).max(1.0);
            let beta_eff = 1.0 / (T_eff * 8.617e-5);
            let use_iter = band.iter().zip(omega.iter()).par_bridge();
            conductivity = use_iter
                .fold(
                    || Array1::<f64>::zeros(n_e),
                    |acc, (energy, omega0)| {
                        let f = 1.0 / ((beta_eff * (*energy - mu)).mapv(|x| x.exp() + 1.0));
                        acc + &f * (1.0 - &f) * beta_eff * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        }
        Ok(conductivity)
    }

    /// Current-first charge intrinsic NLH via K‑quadrature energy‑cut (2D).
    ///
    /// Returns the same signed result as [`Nonlinear_Hall_conductivity_Intrinsic`]
    /// (i.e. integrates $-Q^{ab;c}_n$).
    ///
    /// ```text
    /// Q^{ab;c}_n = 2 v^c_n G^{ab}_n − ½(v^a_n G^{bc}_n + v^b_n G^{ac}_n)
    /// G^{ij}_n = Re Σ_{m≠n} K^{ij}_{nm} / (E_n−E_m)³
    /// ```
    ///
    /// Uses K‑quadrature along the $E_n=\mu$ Fermi‑surface (line in 2D,
    /// surface in 3D) inside each simplex.  Requires `dir_c` so diagonal
    /// velocity fields are available.
    #[allow(non_snake_case)]
    pub fn Nonlinear_Hall_conductivity_Intrinsic_ec(
        &self,
        k_mesh: &Array1<usize>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        eta: f64,
    ) -> Result<Array1<f64>> {
        assert!(
            k_mesh.len() == 2 || k_mesh.len() == 3,
            "Intrinsic EC: only 2D/3D supported"
        );
        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let gauge = Gauge::Atom;

        let mut all_pts: Vec<VertexKernel> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_velocity_kernel(&kv, dir_a, dir_b, Some(dir_c), gauge, None)
            })
            .collect();
        global_band_track(&mut all_pts, k_mesh.as_slice().unwrap());

        let sigma = match k_mesh.len() {
            2 => super::energy_cut::integrate_intrinsic_cut_2d(&all_pts, k_mesh, mu, T),
            3 => super::energy_cut::integrate_intrinsic_cut_3d(&all_pts, k_mesh, mu, T),
            _ => unreachable!(),
        };
        let det = self.lat.det().unwrap();
        Ok(sigma / det)
    }
}
