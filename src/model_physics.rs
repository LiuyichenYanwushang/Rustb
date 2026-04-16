//! Physics calculation methods for tight-binding models
use crate::Gauge;
use crate::Model;
use crate::error::{Result, TbError};
use crate::kpoints::gen_kmesh;
use crate::solve_ham::solve;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::Inverse;
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

        let Us = (self.hamR.mapv(|x| x as f64))
            .dot(kvec)
            .mapv(|x| Complex::<f64>::new(x, 0.0));
        let Us = Us * Complex::new(0.0, 2.0 * PI);
        let Us = Us.mapv(Complex::exp);
        let hamk: Array2<Complex<f64>> = self
            .ham
            .outer_iter()
            .zip(Us.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (hm, u)| {
                acc + &hm * *u
            });

        let hamk = match gauge {
            Gauge::Lattice => hamk,
            Gauge::Atom => {
                let U0: Array1<f64> = self.orb.dot(kvec);
                let U0: Array1<Complex<f64>> = U0.mapv(|x| Complex::<f64>::new(x, 0.0));
                let U0 = U0 * Complex::new(0.0, 2.0 * PI);
                let mut U0: Array1<Complex<f64>> = U0.mapv(Complex::exp);
                let U0 = if self.spin {
                    let U0 = concatenate![Axis(0), U0, U0];
                    U0
                } else {
                    U0
                };
                let U = Array2::from_diag(&U0);
                let U_dag = Array2::from_diag(&U0.map(|x: &Complex<f64>| x.conj()));
                // Next two steps fill in the phase due to orbital coordinates
                let hamk: Array2<Complex<f64>> = U_dag.dot(&hamk);
                let re_ham = hamk.dot(&U);
                re_ham
            }
        };
        hamk
    }

    #[allow(non_snake_case)]
    pub fn dos(
        &self,
        k_mesh: &Array1<usize>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        sigma: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        //! 我这里用的算法是高斯算法, 其算法过程如下
        //!
        //! 首先, 根据 k_mesh 算出所有的能量 $\ve_n$, 然后, 按照定义
        //! $$\rho(\ve)=\sum_N\int\dd\bm k \delta(\ve_n-\ve)$$
        //! 我们将 $\delta(\ve_n-\ve)$ 做了替换, 换成了 $\f{1}{\sqrt{2\pi}\sigma}e^{-\f{(\ve_n-\ve)^2}{2\sigma^2}}$
        //!
        //! 然后, 计算方法是先算出所有的能量, 再将能量乘以高斯分布, 就能得到态密度.
        //!
        //! 态密度的光滑程度和k点密度以及高斯分布的展宽有关
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
