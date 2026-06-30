//! Public traits for response calculations.
//!
//! This module provides the `BerryCurvature` trait and related per‑k‑point
//! kernels that were previously in `conductivity.rs`.

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::SpinDirection;
use crate::math::anti_comm;
use crate::velocity::Velocity;

use super::helpers::build_spin_matrix;

/// Trait providing Berry curvature calculations.
///
/// This trait requires the [`Velocity`] trait and provides methods to compute
/// the Berry curvature and spin Berry curvature at individual k-points or over k-point sets.
///
/// The spin parameter selects the spin operator:
/// - `0`: $\sigma_0$ (identity, charge Berry curvature)
/// - `1`: $\sigma_x$ (spin Berry curvature, x-component)
/// - `2`: $\sigma_y$ (spin Berry curvature, y-component)
/// - `3`: $\sigma_z$ (spin Berry curvature, z-component)
///
/// If the model is spinless, the spin parameter is ignored and spin=0 is used.
pub trait BerryCurvature: Velocity {
    /// Computes the Berry curvature for each band at a single k-point.
    ///
    /// Returns `(omega_n, band)` where `omega_n` contains the Berry curvature for each band
    /// and `band` contains the band energies.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates (in fractional reciprocal coordinates).
    /// * `current_dir` - Current direction (direction vector for the first index $\alpha$ of $\Omega_{n,\alpha\beta}$).
    ///   Must have length equal to `self.dim_r()`.
    /// * `dir_2` - Direction vector for the second index $\beta$ of $\Omega_{n,\alpha\beta}$.
    /// * `spin` - Spin operator index (0, 1, 2, 3 for $\sigma_0, \sigma_x, \sigma_y, \sigma_z$).
    /// * `eta` - Broadening parameter $\eta$ for the energy denominator.
    fn berry_curvature_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>);

    /// Computes the temperature-dependent Berry curvature at a single k-point.
    ///
    /// The formula computed is:
    /// $$ \sum_n f_n\Omega_{n,\alpha\beta}^\gamma(\mathbf k) =
    ///    \sum_n \f{1}{e^{(\varepsilon_{n\mathbf k}-\mu)/(k_B T)}+1}
    ///    \sum_{m\neq n} \f{J_{\alpha,nm}^\gamma v_{\beta,mn}}
    ///    {(\varepsilon_{n\mathbf k}-\varepsilon_{m\mathbf k})^2 + \eta^2} $$
    /// where $J_\alpha^\gamma = \{s_\gamma, v_\alpha\}$ is the anti-commutator of the
    /// spin and velocity operators.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates (in fractional reciprocal coordinates).
    /// * `current_dir` - Current direction (direction vector for the first index $\alpha$ of $\Omega_{\alpha\beta}$).
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K). If `T=0`, a step function is used for the Fermi-Dirac distribution.
    /// * `spin` - Spin operator index (0, 1, 2, 3 for $\sigma_0, \sigma_x, \sigma_y, \sigma_z$).
    ///   Ignored if the model is spinless.
    /// * `eta` - Broadening parameter $\eta$ (in eV).
    fn berry_curvature_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> f64;

    /// Computes the Berry curvature at multiple k-points in parallel.
    ///
    /// This is useful for plotting Berry curvature along band structures or generating heat maps.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `current_dir` - Direction vector for the first index $\alpha$.
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// An `Array1<f64>` of length `nk` containing $\sum_n f_n \Omega_{n,\alpha\beta}$ at each k-point.
    ///
    /// # Panics
    ///
    /// Panics if `current_dir.len()` or `dir_2.len()` does not equal `self.dim_r()`.
    #[allow(non_snake_case)]
    fn berry_curvature<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Array1<f64>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> BerryCurvature for Model<SPIN, DIM, R> {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn berry_curvature_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();
        // Build direction matrix: [current_dir, dir_2]
        let directions = {
            let mut d = Array2::<f64>::zeros((2, self.dim_r()));
            d.row_mut(0).assign(current_dir);
            d.row_mut(1).assign(dir_2);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(&k_vec, Gauge::Atom, &directions);
        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let J: Array2<Complex<f64>> = if SPIN && spin.is_some() {
            let sp = build_spin_matrix(self.norb(), spin);
            anti_comm(&sp, &v_proj.slice(s![0, .., ..])) * 0.5
        } else {
            if !SPIN && spin.is_some() {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            v_proj.slice(s![0, .., ..]).to_owned()
        };
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();

        let evec_conj = evec.t();
        // Lazy map: no heap allocation for the conjugated copy
        let evec = evec.map(|x| x.conj());
        let A1 = J.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec);
        let A2 = evec_conj.dot(&A2);
        let A2 = A2.reversed_axes();
        let AA = A1 * A2;
        let Complex { re, im } = AA.view().split_complex();
        let im = im.mapv(|x| -2.0 * x);
        // Fused: compute omega_n directly without allocating UU[nsta,nsta]
        let mut omega_n = Array1::<f64>::zeros(self.nsta());
        for i in 0..self.nsta() {
            let im_row = im.row(i);
            let mut sum = 0.0f64;
            for j in 0..self.nsta() {
                if i != j {
                    let a = band[[i]] - band[[j]];
                    sum += im_row[[j]] / (a.powi(2) + eta.powi(2));
                }
            }
            omega_n[[i]] = sum;
        }
        (omega_n, band)
    }

    #[allow(non_snake_case)]
    fn berry_curvature_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> f64 {
        let (omega_n, band) = self.berry_curvature_n_onek(&k_vec, &current_dir, &dir_2, spin, eta);
        let mut omega: f64 = 0.0;
        let fermi_dirac = if T == 0.0 {
            band.mapv(|x| if x > mu { 0.0 } else { 1.0 })
        } else {
            let beta = 1.0 / T / 8.617e-5;
            band.mapv(|x| ((beta * (x - mu)).exp() + 1.0).recip())
        };
        let omega = omega_n.dot(&fermi_dirac);
        omega
    }
    #[allow(non_snake_case)]
    fn berry_curvature<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Array1<f64> {
        if current_dir.len() != self.dim_r() || dir_2.len() != self.dim_r() {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));
        let omega: Vec<f64> = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let omega_one = self.berry_curvature_onek(
                    &x.to_owned(),
                    &current_dir,
                    &dir_2,
                    mu,
                    T,
                    spin,
                    eta,
                );
                omega_one
            })
            .collect();
        let omega = arr1(&omega);
        omega
    }
}
