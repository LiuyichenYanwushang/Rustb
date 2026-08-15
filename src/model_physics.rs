//! Physics calculation methods for tight-binding models
use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::kpoints::gen_kmesh;
use crate::solve_ham::Solve;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
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

        // Precompute phase factors exp(i 2π k·R) for each R vector.
        // Dimension-dispatched: compile-time constant loop bound for R·k dot.
        let Us: Vec<Complex<f64>> = match DIM {
            1 => self
                .hamR
                .outer_iter()
                .map(|r| Complex::new(0.0, 2.0 * PI * r[0] as f64 * kvec[0]).exp())
                .collect(),
            2 => self
                .hamR
                .outer_iter()
                .map(|r| {
                    Complex::new(
                        0.0,
                        2.0 * PI * (r[0] as f64 * kvec[0] + r[1] as f64 * kvec[1]),
                    )
                    .exp()
                })
                .collect(),
            3 => self
                .hamR
                .outer_iter()
                .map(|r| {
                    Complex::new(
                        0.0,
                        2.0 * PI
                            * (r[0] as f64 * kvec[0]
                                + r[1] as f64 * kvec[1]
                                + r[2] as f64 * kvec[2]),
                    )
                    .exp()
                })
                .collect(),
            _ => unreachable!(),
        };

        let hamk_slice = hamk.as_slice_mut().unwrap();
        for (iR, &u) in Us.iter().enumerate() {
            let hm = self.ham.index_axis(Axis(0), iR);
            crate::ndarray_lapack::zaxpy(u, hm.as_slice().unwrap(), hamk_slice);
        }

        match gauge {
            Gauge::Lattice => hamk,
            Gauge::Atom => {
                // Dimension-dispatched τ·k phase factors
                let orb_phase: Vec<Complex<f64>> = match DIM {
                    1 => self
                        .orb
                        .outer_iter()
                        .map(|tau| Complex::new(0.0, 2.0 * PI * tau[0] * kvec[0]).exp())
                        .collect(),
                    2 => self
                        .orb
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(0.0, 2.0 * PI * (tau[0] * kvec[0] + tau[1] * kvec[1]))
                                .exp()
                        })
                        .collect(),
                    3 => self
                        .orb
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(
                                0.0,
                                2.0 * PI * (tau[0] * kvec[0] + tau[1] * kvec[1] + tau[2] * kvec[2]),
                            )
                            .exp()
                        })
                        .collect(),
                    _ => unreachable!(),
                };
                let norb = self.norb();
                let orb_phase = Array1::from_vec(orb_phase);
                // Build gauge phase vector: for spinful, duplicate orbital phases
                let mut U0 = Array1::<Complex<f64>>::zeros(if SPIN { 2 * norb } else { norb });
                U0.slice_mut(s![..norb]).assign(&orb_phase);
                if SPIN {
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
        if E_min >= E_max {
            return Err(TbError::InvalidEnergyRange {
                min: E_min,
                max: E_max,
            });
        }
        if E_n == 0 {
            return Err(TbError::InvalidDosParameter {
                parameter: "E_n",
                message: "number of energy bins must be at least 1".to_string(),
            });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(TbError::InvalidDosParameter {
                parameter: "sigma",
                message: "Gaussian smearing width must be finite and positive".to_string(),
            });
        }
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk = kvec.len_of(Axis(0));
        let eigenvalues = self.solve_band_all_parallel(&kvec);
        let E = Array1::linspace(E_min, E_max, E_n);
        let _dim: usize = k_mesh.len();
        let centre = eigenvalues.into_raw_vec_and_offset().0.into_par_iter();
        let sigma0 = 1.0 / sigma;
        let pi0 = 1.0 / (2.0 * PI).sqrt();
        let _dos = Array1::<f64>::zeros(E_n);
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
