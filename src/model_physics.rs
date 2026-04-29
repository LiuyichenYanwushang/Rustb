//! Physics calculation methods for tight-binding models
use crate::Gauge;
use crate::Dimension;
use crate::Model;
use crate::error::{Result, TbError};
use crate::kpoints::gen_kmesh;
use crate::solve_ham::solve;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

impl Model {
    #[allow(non_snake_case)]
    #[inline(always)]
    #[cfg_attr(doc, katexit::katexit)]
    ///Performs Fourier transform, converting real-space Hamiltonian to reciprocal-space Hamiltonian.
    ///
    ///There are two gauge choices: lattice gauge and atomic gauge, corresponding to `Gauge::Lattice` and `Gauge::Atom`.
    ///
    ///For the atomic gauge, the transformation between real-space wavefunction $\ket{n\bm R}$ and reciprocal-space wavefunction $\ket{u_{\bm k,n}}$ is:
    ///
    ///$$\ket{u_{n\bm k}(\bm r)}=\sum_{\bm R} e^{i\bm k\cdot(\bm R+\bm\tau_n)}\ket{n\bm R}$$
    ///
    ///satisfying $\ket{u_{i\bm k}(\bm r+\bm R)}=\ket{u_{i\bm k}(\bm r)}$.
    ///
    ///For the Hamiltonian, we have:
    ///$$
    ///H_{mn,\bm k}=\bra{u_{m\bm k}}\hat H\ket{u_{n\bm k}}=\sum_{\bm R^\prime}\sum_{\bm R} \bra{m\bm R^\prime}\hat H\ket{n\bm R}e^{-i(\bm R'-\bm R+\bm\tau_m-\bm \tau_n)\cdot\bm k}.
    ///$$
    ///Due to translational symmetry, only $\bm R'-\bm R$ matters, thus:
    ///$$
    ///H_{mn,\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R-\bm\tau_m+\bm \tau_n)\cdot\bm k}
    ///$$
    ///
    ///For the lattice gauge, we have $$\ket{\phi_{n\bm k}}=\sum_{\bm R} e^{i\bm k\cdot\bm R}\ket{n\bm R},$$ so:
    ///$$
    ///H_{mn,\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R)\cdot\bm k}
    ///$$
    ///
    ///Here $\ket{\psi_{n\bm k}}$ is periodic in reciprocal space: $\ket{\phi_{n\bm k}(\bm r)}=\ket{\phi_{n\bm k+\bm G}(\bm r)}$.
    pub fn gen_ham<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> Array2<Complex<f64>> {
        assert!(
            kvec.len() == self.dim_r(),
            "Wrong, the k-vector's length must equal to the dimension of model."
        );

        let nsta = self.nsta();
        let mut hamk = Array2::<Complex<f64>>::zeros((nsta, nsta));

        // Dimension-dispatched: each arm the compiler sees a compile-time
        // constant loop bound, so the inner phase loop is fully unrolled
        // and per-element bounds checks are eliminated.
        match self.dim_r {
            Dimension::one => {
                Zip::from(self.ham.outer_iter())
                    .and(self.hamR.outer_iter())
                    .for_each(|hm, hamr_row| {
                        let phase = hamr_row[0] as f64 * kvec[0];
                        let u = Complex::new(0.0, 2.0 * PI * phase).exp();
                        //Zip::from(&mut hamk).and(&hm).for_each(|a, &b| *a += b * u);
                        hamk.scaled_add(u, &hm);
                    });
            }
            Dimension::two => {
                Zip::from(self.ham.outer_iter())
                    .and(self.hamR.outer_iter())
                    .for_each(|hm, hamr_row| {
                        let phase =
                            hamr_row[0] as f64 * kvec[0] + hamr_row[1] as f64 * kvec[1];
                        let u = Complex::new(0.0, 2.0 * PI * phase).exp();
                        //Zip::from(&mut hamk).and(&hm).for_each(|a, &b| *a += b * u);
                        hamk.scaled_add(u, &hm);
                    });
            }
            Dimension::three => {
                Zip::from(self.ham.outer_iter())
                    .and(self.hamR.outer_iter())
                    .for_each(|hm, hamr_row| {
                        let phase = hamr_row[0] as f64 * kvec[0]
                            + hamr_row[1] as f64 * kvec[1]
                            + hamr_row[2] as f64 * kvec[2];
                        let u = Complex::new(0.0, 2.0 * PI * phase).exp();
                        //Zip::from(&mut hamk).and(&hm).for_each(|a, &b| *a += b * u);
                        hamk.scaled_add(u, &hm);
                    });
            }
        }

        match gauge {
            Gauge::Lattice => hamk,
            Gauge::Atom => {
                // Dimension-dispatched τ·k phase factors
                let orb_phase: Vec<Complex<f64>> = match self.dim_r {
                    Dimension::one => self
                        .orb
                        .outer_iter()
                        .map(|tau| Complex::new(0.0, 2.0 * PI * tau[0] * kvec[0]).exp())
                        .collect(),
                    Dimension::two => self
                        .orb
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(
                                0.0,
                                2.0 * PI * (tau[0] * kvec[0] + tau[1] * kvec[1]),
                            )
                            .exp()
                        })
                        .collect(),
                    Dimension::three => self
                        .orb
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(
                                0.0,
                                2.0 * PI
                                    * (tau[0] * kvec[0] + tau[1] * kvec[1] + tau[2] * kvec[2]),
                            )
                            .exp()
                        })
                        .collect(),
                };
                let norb = self.norb();
                let orb_phase = Array1::from_vec(orb_phase);
                // Build gauge phase vector: for spinful, duplicate orbital phases
                let mut U0 = Array1::<Complex<f64>>::zeros(if self.spin { 2 * norb } else { norb });
                U0.slice_mut(s![..norb]).assign(&orb_phase);
                if self.spin {
                    U0.slice_mut(s![norb..]).assign(&orb_phase);
                }
                // Gauge transform: H'[m,n] = conj(U0[m]) * H[m,n] * U0[n]
                for m in 0..nsta {
                    let mut row = hamk.slice_mut(s![m, ..]);
                    let conj_pm = U0[m].conj();
                    Zip::from(&mut row)
                        .and(&U0)
                        .for_each(|h, &pn| *h *= conj_pm * pn);
                }
                hamk
            }
        }
    }

    /// Computes the density of states $\rho(E)$ using Gaussian smearing.
    ///
    /// The DOS is defined as:
    ///
    /// $$\rho(E) = \frac{1}{N_k} \sum_{n,\mathbf{k}} \delta(E - E_{n\mathbf{k}})$$
    ///
    /// The delta function is approximated by a Gaussian of width $\sigma$:
    ///
    /// $$\delta(x) \approx \frac{1}{\sqrt{2\pi}\,\sigma}\, e^{-x^2 / (2\sigma^2)}$$
    ///
    /// # Algorithm
    ///
    /// 1. Generate a uniform k-mesh from `k_mesh`
    /// 2. Diagonalize $H(\mathbf{k})$ at every k-point in parallel
    /// 3. Convolve eigenvalues with the Gaussian kernel and sum
    ///
    /// The smoothness depends on both the k-point density and $\sigma$.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` — k-points along each direction, e.g. `[51, 51]`
    /// * `E_min`, `E_max` — Energy range
    /// * `E_n` — Number of energy bins
    /// * `sigma` — Gaussian smearing width (same units as energy)
    ///
    /// # Returns
    ///
    /// `(energies, dos)` — energy grid and corresponding DOS.
    #[allow(non_snake_case)]
    pub fn dos(
        &self,
        k_mesh: &Array1<usize>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        sigma: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk = kvec.len_of(Axis(0));
        let eigenvalues = self.solve_band_all_parallel(&kvec);
        let E = Array1::linspace(E_min, E_max, E_n);
        let dim: usize = k_mesh.len();
        let centre = eigenvalues.into_raw_vec().into_par_iter();
        let sigma0 = 1.0 / sigma;
        let pi0 = 1.0 / (2.0 * PI).sqrt();
        let dos = Array1::<f64>::zeros(E_n);
        let dos = centre
            .fold(
                || Array1::<f64>::zeros(E_n),
                |acc, x| {
                    let A: Array1<f64> = (&E - x) * sigma0;
                    let f: Array1<f64> = (-&A * &A / 2.0).mapv(|x: f64| x.exp()) * sigma0 * pi0;
                    acc + &f
                },
            )
            .reduce(|| Array1::<f64>::zeros(E_n), |acc, x| acc + x);
        let dos = dos / (nk as f64);
        Ok((E, dos))
    }
}
