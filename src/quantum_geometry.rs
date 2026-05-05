//! Quantum geometry computations: quantum metric and Berry curvature.
//!
//! The quantum geometric tensor for band $n$ is defined as
//!
//! $$
//! G_{n,\alpha\beta}(\mathbf{k}) = \sum_{m\neq n}
//! \frac{\bra{n}v_\alpha\ket{m}\bra{m}v_\beta\ket{n}}
//!      {(E_n - E_m)^2 + \eta^2},
//! $$
//!
//! where $v_\alpha = \frac{1}{\hbar}\partial_{k_\alpha} H(\mathbf{k})$ is the velocity
//! operator.  The quantum metric $g_{n,\alpha\beta}$ (real, symmetric) and the Berry
//! curvature $\Omega_{n,\alpha\beta}$ (real, antisymmetric) are obtained from the
//! real and imaginary parts:
//!
//! $$
//! \begin{aligned}
//! g_{n,\alpha\beta} &= \operatorname{Re}\bigl[G_{n,\alpha\beta}\bigr], \\[4pt]
//! \Omega_{n,\alpha\beta} &= -2\,\operatorname{Im}\bigl[G_{n,\alpha\beta}\bigr].
//! \end{aligned}
//! $$
//!
//! The two quantities are related through
//!
//! $$
//! G_{\alpha\beta} = g_{\alpha\beta} - \frac{i}{2}\,\Omega_{\alpha\beta}.
//! $$
//!
//! # Broadening
//!
//! The parameter $\eta$ is a Lorentzian broadening that regularises the energy
//! denominator when bands are nearly degenerate.  The limit $\eta\to 0$ recovers
//! the exact zero-temperature expression.

use crate::error::Result;
use crate::kpoints::gen_kmesh;
use crate::velocity::*;
use crate::{Gauge, Model};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

/// Trait for computing the quantum geometric tensor and its real/imaginary
/// components at the band-resolved level.
///
/// The two functions differ in the k‑point argument:
/// - `*_onek` evaluates a single k‑point.
/// - `*_n` evaluates a list of k‑points in parallel.
pub trait QuantumGeometry: Velocity {
    /// Band-resolved quantum geometry at **one** k‑point.
    ///
    /// # Arguments
    ///
    /// * `k_vec` — k‑point coordinates (fractional reciprocal coordinates).
    /// * `dir_1` — Direction vector for the first index $\alpha$ of
    ///   $G_{n,\alpha\beta}$ (must have length `self.dim_r()`).
    /// * `dir_2` — Direction vector for the second index $\beta$ of
    ///   $G_{n,\alpha\beta}$.
    /// * `eta`  — Broadening parameter $\eta$ (in eV).
    ///
    /// # Returns
    ///
    /// `(metric_n, omega_n, band)` where
    ///
    /// | field      | type                  | meaning                                      |
    /// |------------|-----------------------|----------------------------------------------|
    /// | `metric_n` | `Array1<Complex<f64>>` | $g_{n,\alpha\beta}$ for each band $n$        |
    /// | `omega_n`  | `Array1<Complex<f64>>` | $\Omega_{n,\alpha\beta}$ for each band $n$   |
    /// | `band`     | `Array1<f64>`          | band energies $E_{n\mathbf{k}}$              |
    ///
    /// (`omega_n` is stored as `Complex` for convenience; its imaginary part is
    /// always zero.)
    fn quantum_geometry_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        eta: f64,
    ) -> (Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<f64>);

    /// Band-resolved quantum geometry at **multiple** k‑points (parallel).
    ///
    /// # Arguments
    ///
    /// * `k_vec` — Array of k‑points, shape `(nk, dim_r)`.
    /// * `dir_1`, `dir_2`, `eta` — same meaning as in
    ///   [`quantum_geometry_n_onek`].
    ///
    /// # Returns
    ///
    /// `(metric, omega, bands)` where
    ///
    /// | field    | type                        | shape          |
    /// |----------|-----------------------------|----------------|
    /// | `metric` | `Array2<Complex<f64>>`      | `(nk, nsta)`   |
    /// | `omega`  | `Array2<Complex<f64>>`      | `(nk, nsta)`   |
    /// | `bands`  | `Array2<f64>`               | `(nk, nsta)`   |
    fn quantum_geometry_n<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        eta: f64,
    ) -> (Array2<Complex<f64>>, Array2<Complex<f64>>, Array2<f64>);
}

impl<const SPIN: bool> QuantumGeometry for Model<SPIN> {
    #[inline(always)]
    fn quantum_geometry_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        eta: f64,
    ) -> (Array1<Complex<f64>>, Array1<Complex<f64>>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();

        // Velocity operator v_a(k) at this k-point
        let (v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
            self.gen_v(&k_vec, Gauge::Atom);

        // Project velocity along dir_1 (α direction): v_α = Σ_d dir_1[d] · v_d
        let v_alpha = v
            .outer_iter()
            .zip(dir_1.iter())
            .fold(
                Array2::zeros((self.nsta(), self.nsta())),
                |acc, (v_d, &d)| acc + &v_d * (d + 0.0 * li),
            );

        // Project velocity along dir_2 (β direction): v_β = Σ_d dir_2[d] · v_d
        let v_beta = v
            .outer_iter()
            .zip(dir_2.iter())
            .fold(
                Array2::zeros((self.nsta(), self.nsta())),
                |acc, (v_d, &d)| acc + &v_d * (d + 0.0 * li),
            );

        // Diagonalize H(k)
        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            panic!("Diagonalization failed in quantum_geometry_n_onek");
        };

        // Transform velocity matrices to the eigenstate basis
        // evec[:, n] = |ψ_n⟩,  evec_conj[n, :] = ⟨ψ_n|
        let evec_conj = evec.t();
        let evec = evec.mapv(|x| x.conj());

        // A1[n,m] = ⟨n|v_α|m⟩
        let A1 = v_alpha.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        // A2[m,n] = ⟨m|v_β|n⟩ (transposed so element-wise product works)
        let A2 = v_beta.dot(&evec);
        let A2 = evec_conj.dot(&A2);
        let A2 = A2.reversed_axes();

        // Element-wise: AA[n,m] = ⟨n|v_α|m⟩ · ⟨m|v_β|n⟩
        let AA = A1 * A2;
        let Complex { re, im } = AA.view().split_complex();

        // Quantum metric:    g_n = Σ_{m≠n} Re[⟨n|v_α|m⟩⟨m|v_β|n⟩] / (ΔE² + η²)
        // Berry curvature:   Ω_n = Σ_{m≠n} -2 Im[⟨n|v_α|m⟩⟨m|v_β|n⟩] / (ΔE² + η²)
        let nsta = self.nsta();
        let h_eta = eta * eta; // η² for the denominator

        let mut metric_n = Array1::<Complex<f64>>::zeros(nsta);
        let mut omega_n = Array1::<Complex<f64>>::zeros(nsta);

        for i in 0..nsta {
            let mut g_sum = 0.0f64;
            let mut o_sum = 0.0f64;
            for j in 0..nsta {
                if i == j {
                    continue;
                }
                let de = band[i] - band[j];
                let inv_denom = 1.0 / (de * de + h_eta);
                g_sum += re[[i, j]] * inv_denom;
                o_sum += im[[i, j]] * inv_denom;
            }
            metric_n[i] = Complex::new(g_sum, 0.0);
            // Ω_n = -2 Im[G_n], convention matching berry_curvature_n_onek
            omega_n[i] = Complex::new(-2.0 * o_sum, 0.0);
        }

        (metric_n, omega_n, band)
    }

    fn quantum_geometry_n<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        eta: f64,
    ) -> (Array2<Complex<f64>>, Array2<Complex<f64>>, Array2<f64>) {
        let nk = k_vec.len_of(Axis(0));
        let nsta = self.nsta();

        // Collect results in parallel
        let results: Vec<_> = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.quantum_geometry_n_onek(&k.to_owned(), dir_1, dir_2, eta))
            .collect();

        let mut metric = Array2::<Complex<f64>>::zeros((nk, nsta));
        let mut omega = Array2::<Complex<f64>>::zeros((nk, nsta));
        let mut bands = Array2::<f64>::zeros((nk, nsta));

        for (ik, (m, o, b)) in results.into_iter().enumerate() {
            metric.row_mut(ik).assign(&m);
            omega.row_mut(ik).assign(&o);
            bands.row_mut(ik).assign(&b);
        }

        (metric, omega, bands)
    }
}

// ── Fermi‑Dirac‑weighted quantum geometry over a k‑mesh ─────────────────

impl<const SPIN: bool> Model<SPIN> {
    /// Fermi–Dirac weighted quantum metric and Berry curvature as a function of
    /// chemical potential.
    ///
    /// This is the quantum‑geometry analogue of
    /// [`Hall_conductivity_mu`](crate::conductivity::Model::Hall_conductivity_mu):
    /// it first computes the band‑resolved quantum metric $g_{n,\alpha\beta}$
    /// and Berry curvature $\Omega_{n,\alpha\beta}$ at every k‑point of a
    /// uniform mesh, then evaluates the Fermi–Dirac‑weighted Brillouin‑zone sum
    /// for each chemical potential in `mu`.  Re‑using the band‑resolved data
    /// avoids recomputing the expensive velocity‑matrix diagonalisation for
    /// every $\mu$.
    ///
    /// $$
    /// \begin{aligned}
    /// g_{\alpha\beta}(\mu) &=
    ///   \frac{1}{N} \sum_{\mathbf{k},n}
    ///   f\bigl(E_{n\mathbf{k}}; \mu, T\bigr)\,
    ///   g_{n,\alpha\beta}(\mathbf{k}), \\[4pt]
    /// \Omega_{\alpha\beta}(\mu) &=
    ///   \frac{1}{N} \sum_{\mathbf{k},n}
    ///   f\bigl(E_{n\mathbf{k}}; \mu, T\bigr)\,
    ///   \Omega_{n,\alpha\beta}(\mathbf{k}),
    /// \end{aligned}
    /// $$
    ///
    /// where $f$ is the Fermi–Dirac distribution and $N$ is the number of
    /// k‑points.  Both quantities are returned in **natural units** (the
    /// $e^2/\hbar$ prefactor is omitted, consistent with the rest of the
    /// crate).
    ///
    /// # Arguments
    ///
    /// * `k_mesh` — Number of k‑points along each direction, e.g.
    ///   `arr1(&[nk, nk])` for 2D.
    /// * `dir_1` — Direction vector for the first index $\alpha$.
    /// * `dir_2` — Direction vector for the second index $\beta$.
    /// * `mu` — Array of chemical‑potential values $\mu$ (in eV).
    /// * `T` — Temperature (in K).  `T = 0` uses a step function.
    /// * `eta` — Broadening parameter $\eta$ (in eV).
    ///
    /// # Returns
    ///
    /// `Ok((metric, omega))` where `metric` (`Array1<f64>`) and `omega`
    /// (`Array1<f64>`) are the Brillouin‑zone sums for each $\mu$ in `mu`.
    /// The quantum metric has units of $\text{Å}^2$ (the geometric length‑squared
    /// of the wavefunction in real space); the Berry curvature is dimensionless
    /// (in units of the reciprocal‑lattice cell).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ndarray::{arr1, Array1};
    /// # use rustb::Model;
    /// # fn example(model: &Model) -> Result<(), rustb::error::TbError> {
    /// let kmesh = arr1(&[31, 31]);
    /// let dir_1 = arr1(&[1.0, 0.0]);
    /// let dir_2 = arr1(&[0.0, 1.0]);
    /// let mu = Array1::linspace(-2.0, 2.0, 101);
    /// let (metric, omega) = model.quantum_geometry(&kmesh, &dir_1, &dir_2, &mu, 0.0, 1e-3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn quantum_geometry(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        eta: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let nsta = self.nsta();

        // Compute band‑resolved quantum geometry at every k‑point
        let all: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| {
                let (m, o, b) =
                    self.quantum_geometry_n_onek(&k.to_owned(), dir_1, dir_2, eta);
                let m_real: Vec<f64> = m.iter().map(|c| c.re).collect();
                let o_real: Vec<f64> = o.iter().map(|c| c.re).collect();
                (m_real, o_real, b.to_vec())
            })
            .collect();

        // Flatten into (nk, nsta) arrays
        let (metric_flat, omega_flat, band_flat): (Vec<f64>, Vec<f64>, Vec<f64>) = {
            let cap = nk * nsta;
            let mut mf = Vec::with_capacity(cap);
            let mut of = Vec::with_capacity(cap);
            let mut bf = Vec::with_capacity(cap);
            for (m, o, b) in all {
                mf.extend(m);
                of.extend(o);
                bf.extend(b);
            }
            (mf, of, bf)
        };
        let metric_n = Array2::<f64>::from_shape_vec((nk, nsta), metric_flat).unwrap();
        let omega_n = Array2::<f64>::from_shape_vec((nk, nsta), omega_flat).unwrap();
        let band = Array2::<f64>::from_shape_vec((nk, nsta), band_flat).unwrap();

        let n_mu = mu.len();
        let norm = 1.0 / (nk as f64);

        // Fermi‑Dirac weighted BZ sum — single pass over mu, computing metric
        // and omega together to avoid redundant work and intermediate allocations.
        let metric_mu: Vec<(f64, f64)> = if T == 0.0 {
            // T = 0: step function, parallel over mu
            mu.into_par_iter()
                .map(|&mu_i| {
                    let mut g_sum = 0.0f64;
                    let mut o_sum = 0.0f64;
                    for ik in 0..nk {
                        for ib in 0..nsta {
                            if band[[ik, ib]] <= mu_i {
                                g_sum += metric_n[[ik, ib]];
                                o_sum += omega_n[[ik, ib]];
                            }
                        }
                    }
                    (g_sum * norm, o_sum * norm)
                })
                .collect::<Vec<(f64, f64)>>()
        } else {
            // T > 0: Fermi‑Dirac distribution, parallel over mu
            let beta = 1.0 / (T * 8.617e-5);
            mu.into_par_iter()
                .map(|&mu_i| {
                    let mut g_sum = 0.0f64;
                    let mut o_sum = 0.0f64;
                    for ik in 0..nk {
                        for ib in 0..nsta {
                            let e = band[[ik, ib]];
                            let fd = 1.0 / ((beta * (e - mu_i)).exp() + 1.0);
                            g_sum += metric_n[[ik, ib]] * fd;
                            o_sum += omega_n[[ik, ib]] * fd;
                        }
                    }
                    (g_sum * norm, o_sum * norm)
                })
                .collect::<Vec<(f64, f64)>>()
        };

        let n_mu = mu.len();
        let mut g_arr = Array1::<f64>::zeros(n_mu);
        let mut o_arr = Array1::<f64>::zeros(n_mu);
        for (i, (g, o)) in metric_mu.into_iter().enumerate() {
            g_arr[i] = g;
            o_arr[i] = o;
        }

        Ok((g_arr, o_arr))
    }
}
