//! # Optical conductivity
//!
//! ## Theory
//!
//! The optical conductivity tensor is
//!
//! $$\sigma_{\alpha\beta}(\omega) =
//!   \frac{2ie^2\hbar}{V}\sum_{\mathbf{k}}\sum_n f_n
//!   \Bigl(g_{n,\alpha\beta} + \frac{i}{2}\Omega_{n,\alpha\beta}\Bigr)$$
//!
//! where the quantum metric $g$ and Berry curvature $\Omega$ contributions are
//!
//! $$g_{n,\alpha\beta} = \sum_{m\neq n}
//!   \frac{\omega-i\eta}{E_n-E_m}\,
//!   \frac{\operatorname{Re}[v^\alpha_{nm} v^\beta_{mn}]}
//!        {(E_n-E_m)^2 - (\omega-i\eta)^2}$$
//!
//! $$\Omega_{n,\alpha\beta} = \sum_{m\neq n}
//!   \frac{\operatorname{Re}[v^\alpha_{nm} v^\beta_{mn}]}
//!        {(E_n-E_m)^2 - (\omega-i\eta)^2}$$
//!
//! In the **simplex path**, the equivalent form is used:
//!
//! $$\sigma^{ab}(\omega,\mu,T) = \sum_{n\neq m} \int_{\rm BZ}
//!   \frac{(f_n-f_m)\,K^{ab}_{nm}}
//!        {(E_n-E_m)^2 - (\omega+i\eta)^2}\,d\mathbf{k}$$
//!
//! where $K^{ab}_{nm}=v^a_{nm}v^b_{mn}$ is the gauge‑invariant kernel.
//!
//! ## API
//!
//! | Method | Path | Formula |
//! |--------|------|---------|
//! | `optical_conductivity` | direct sum | $\sigma(\omega)$ per frequency |
//! | `optical_conductivity_T` | direct sum | $\sigma(\omega,\mu,T)$ |
//! | `optical_conductivity_all_direction` | direct sum | all $\alpha\beta$ components |
//! | `optical_conductivity_simplex` | simplex | $\sigma^{ab}(\omega)$ via quadrature |

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::Result;
use crate::velocity::Velocity;

use super::kernel::quadrature_optical_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d};
use super::types::VertexKernel;

// ── Old direct‑sum OpticalGeometry trait ──────────────────────────────────

pub trait OpticalGeometry: Velocity {
    fn optical_geometry_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        og: &Array1<f64>,
        eta: f64,
    ) -> (Array2<Complex<f64>>, Array2<Complex<f64>>, Array1<f64>);
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> OpticalGeometry for Model<SPIN, DIM, R> {
    #[inline(always)]
    fn optical_geometry_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        og: &Array1<f64>,
        eta: f64,
    ) -> (Array2<Complex<f64>>, Array2<Complex<f64>>, Array1<f64>) {
        //! This function calculates $g_{n,\ap\bt}$ and $\og_{n\ap\bt}$
        //!
        //! `og` represents the frequency
        //!
        //! `eta` is a small quantity

        let li: Complex<f64> = 1.0 * Complex::i();

        // Build direction matrix: [dir_1, dir_2]
        let directions = {
            let mut d = Array2::<f64>::zeros((2, self.dim_r()));
            d.row_mut(0).assign(dir_1);
            d.row_mut(1).assign(dir_2);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(&k_vec, Gauge::Atom, &directions);

        let J: Array2<Complex<f64>> = v_proj.slice(s![0, .., ..]).to_owned();
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();

        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec_conj = evec.t();
        let evec = evec.mapv(|x| x.conj());

        let A1 = J.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec);
        let A2 = evec_conj.dot(&A2);
        let A2 = A2.reversed_axes();
        let AA = A1 * A2;

        let Complex { re, im } = AA.view().split_complex();
        let re = re.mapv(|x| Complex::new(2.0 * x, 0.0));
        let im = im.mapv(|x| Complex::new(0.0, -2.0 * x));

        let n_og = og.len();
        assert_eq!(
            band.len(),
            self.nsta(),
            "this is strange for band's length is not equal to self.nsta()"
        );

        let mut U0 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        let mut Us = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

        // Calculate the energy differences and their inverses
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                let a = band[[i]] - band[[j]];
                U0[[i, j]] = Complex::new(a, 0.0);
                Us[[i, j]] = if a.abs() > 1e-6 {
                    Complex::new(1.0 / a, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
            }
        }

        let mut matric_n = Array2::zeros((n_og, self.nsta()));
        let mut omega_n = Array2::zeros((n_og, self.nsta()));

        // Calculate the matrices for each frequency
        Zip::from(omega_n.outer_iter_mut())
            .and(matric_n.outer_iter_mut())
            .and(og.view())
            .for_each(|mut omega, mut matric, a0| {
                let li_eta = a0 + li * eta;
                let UU = U0.mapv(|x| (x * x - li_eta * li_eta).finv());
                let U1 = &UU * &Us * li_eta;

                let o = im
                    .outer_iter()
                    .zip(UU.outer_iter())
                    .map(|(a, b)| a.dot(&b))
                    .collect();
                let m = re
                    .outer_iter()
                    .zip(U1.outer_iter())
                    .map(|(a, b)| a.dot(&b))
                    .collect();
                let o = Array1::from_vec(o);
                let m = Array1::from_vec(m);
                omega.assign(&o);
                matric.assign(&m);
            });

        (matric_n, omega_n, band)
    }
}

// ── Old direct‑sum Model<false, DIM> methods ──────────────────────────────

impl<const DIM: usize> Model<false, DIM> {
    pub fn optical_conductivity(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        T: f64,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array1<Complex<f64>>, Array1<Complex<f64>>)> {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let (matric_sum, omega_sum) = kvec
            .outer_iter()
            .into_par_iter()
            .map(|k| {
                let (matric_n, omega_n, band) =
                    self.optical_geometry_n_onek(&k, dir_1, dir_2, og, eta);
                let fermi_dirac = if T == 0.0 {
                    band.mapv(|x| if x > mu { 0.0 } else { 1.0 })
                } else {
                    let beta = 1.0 / T / 8.617e-5;
                    band.mapv(|x| ((beta * (x - mu)).exp() + 1.0).recip())
                };
                let fermi_dirac = fermi_dirac.mapv(|x| Complex::new(x, 0.0));
                let matric = matric_n.dot(&fermi_dirac);
                let omega = omega_n.dot(&fermi_dirac);
                (matric, omega)
            })
            .reduce(
                || (Array1::zeros(n_og), Array1::zeros(n_og)),
                |(matric_acc, omega_acc), (matric, omega)| (matric_acc + matric, omega_acc + omega),
            );
        let matric_sum = li * matric_sum / self.lat.det().unwrap() / (nk as f64);
        let omega_sum = li * omega_sum / self.lat.det().unwrap() / (nk as f64);
        Ok((matric_sum, omega_sum))
    }

    pub fn optical_conductivity_T(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        T: &Array1<f64>,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let n_T = T.len();
        let (matric_sum, omega_sum) = kvec
            .outer_iter()
            .into_par_iter()
            .map(|k| {
                let (matric_n, omega_n, band) =
                    self.optical_geometry_n_onek(&k, dir_1, dir_2, og, eta);
                let beta = T.mapv(|x| 1.0 / x / 8.617e-5);
                let nsta = band.len();
                let n_T = beta.len();
                let mut fermi_dirac: Array2<Complex<f64>> = Array2::zeros((nsta, n_T));
                Zip::from(fermi_dirac.outer_iter_mut())
                    .and(band.view())
                    .for_each(|mut f0, e0| {
                        let a = beta
                            .map(|x0| Complex::new(((x0 * (e0 - mu)).exp() + 1.0).recip(), 0.0));
                        f0.assign(&a);
                    });
                let matric = matric_n.dot(&fermi_dirac);
                let omega = omega_n.dot(&fermi_dirac);
                (matric, omega)
            })
            .reduce(
                || (Array2::zeros((n_og, n_T)), Array2::zeros((n_og, n_T))),
                |(matric_acc, omega_acc), (matric, omega)| (matric_acc + matric, omega_acc + omega),
            );
        let matric_sum = li * matric_sum / self.lat.det().unwrap() / (nk as f64);
        let omega_sum = li * omega_sum / self.lat.det().unwrap() / (nk as f64);
        Ok((matric_sum, omega_sum))
    }

    ///Computes the optical conductivity for the six components xx, yy, zz, xy, yz, xz
    ///directly, separated into symmetric and antisymmetric parts.
    ///Output format is ($\sigma_{ab}^S$, $\sigma_{ab}^A$), where S and A denote
    ///symmetry and antisymmetry.
    ///$\sigma_{ab}^S$ has shape $6\times n_\omega$.
    ///For 2D systems, only the three components xx, yy, xy are produced.
    pub fn optical_conductivity_all_direction(
        &self,
        k_mesh: &Array1<usize>,
        T: f64,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let (matric,omega):(Vec<_>,Vec<_>)=kvec.outer_iter().into_par_iter()
            .map(|k| {
                let (mut v, hamk): (Array3<Complex<f64>>,Array2<Complex<f64>>) = self.gen_v(&k,Gauge::Atom);
                let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
                    (eigvals, eigvecs)
                } else {
                    todo!()
                };
                let evec_conj=evec.t();
                let evec= evec.mapv(|x| x.conj());

                let mut A = Array3::zeros((self.dim_r(),self.nsta(),self.nsta()));
                Zip::from(A.outer_iter_mut()).and(v.outer_iter()).for_each(|mut a,v| a.assign(&evec_conj.dot(&v.dot(&evec))));

                // Calculate the energy differences and their inverses
                let mut U0=Array2::zeros((self.nsta(),self.nsta()));
                let mut Us=Array2::zeros((self.nsta(),self.nsta()));
                for i in 0..self.nsta() {
                    for j in 0..self.nsta() {
                        let a = band[[i]] - band[[j]];
                        U0[[i, j]] = Complex::new(a, 0.0);
                        Us[[i, j]] = if a.abs() > 1e-6 {
                            Complex::new(1.0 / a, 0.0)
                        } else {
                            Complex::new(0.0, 0.0)
                        };
                    }
                }

                let fermi_dirac=if T==0.0{
                    band.mapv(|x| if x>mu {0.0} else {1.0})
                }else{
                    let beta=1.0/T/8.617e-5;
                    band.mapv(|x| {((beta*(x-mu)).exp()+1.0).recip()})
                };
                let fermi_dirac=fermi_dirac.mapv(|x| Complex::new(x,0.0));

                let n_og=og.len();
                assert_eq!(band.len(), self.nsta(), "this is strange for band's length is not equal to self.nsta()");

                // Precompute U0^2 once: avoids per-frequency x*x
                let U0_sq: Array2<f64> = U0.mapv(|x| x.re * x.re);

                let (matric_n,omega_n)=match self.dim_r(){
                    3=>{
                        let mut matric_n=Array2::zeros((6,n_og));
                        let mut omega_n=Array2::zeros((3,n_og));
                        let A_xx=&A.slice(s![0,..,..])*&A.slice(s![0,..,..]).t();
                        let A_yy=&A.slice(s![1,..,..])*&A.slice(s![1,..,..]).t();
                        let A_zz=&A.slice(s![2,..,..])*&A.slice(s![2,..,..]).t();
                        let A_xy=&A.slice(s![0,..,..])*&A.slice(s![1,..,..]).t();
                        let A_yz=&A.slice(s![1,..,..])*&A.slice(s![2,..,..]).t();
                        let A_xz=&A.slice(s![0,..,..])*&A.slice(s![2,..,..]).t();
                        let re_xx:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_xx;
                        let re_yy:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_yy;
                        let re_zz:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_zz;
                        let Complex { re, im } = A_xy.view().split_complex();
                        let re_xy:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_xy:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));
                        let Complex { re, im } = A_yz.view().split_complex();
                        let re_yz:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_yz:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));
                        let Complex { re, im } = A_xz.view().split_complex();
                        let re_xz:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_xz:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));

                        #[inline]
                        fn accumulate(mat: &Array2<Complex<f64>>, kernel: &Array2<Complex<f64>>, fermi: &Array1<Complex<f64>>) -> Complex<f64> {
                            let mut sum = Complex::new(0.0, 0.0);
                            for (i, (mr, kr)) in mat.outer_iter().zip(kernel.outer_iter()).enumerate() {
                                sum += mr.dot(&kr) * fermi[i];
                            }
                            sum
                        }

                        Zip::from(omega_n.axis_iter_mut(Axis(1)))
                            .and(matric_n.axis_iter_mut(Axis(1)))
                            .and(og.view())
                            .par_for_each(|mut omega, mut matric, a0| {
                                let li_eta = a0 + li * eta;
                                let li_eta_sq = li_eta * li_eta;
                                let UU = U0_sq.mapv(|x| (Complex::new(x, 0.0) - li_eta_sq).finv());
                                let U1: Array2<Complex<f64>> = &UU * &Us * li_eta;

                                matric[[0]] = accumulate(&re_xx, &U1, &fermi_dirac);
                                matric[[1]] = accumulate(&re_yy, &U1, &fermi_dirac);
                                matric[[2]] = accumulate(&re_zz, &U1, &fermi_dirac);

                                omega[[0]]  = accumulate(&im_xy, &UU, &fermi_dirac);
                                matric[[3]] = accumulate(&re_xy, &U1, &fermi_dirac);
                                omega[[1]]  = accumulate(&im_yz, &UU, &fermi_dirac);
                                matric[[4]] = accumulate(&re_yz, &U1, &fermi_dirac);
                                omega[[2]]  = accumulate(&im_xz, &UU, &fermi_dirac);
                                matric[[5]] = accumulate(&re_xz, &U1, &fermi_dirac);
                            });
                        (matric_n,omega_n)
                    },
                    2=>{
                        let mut matric_n=Array2::zeros((3,n_og));
                        let mut omega_n=Array2::zeros((1,n_og));
                        let A_xx=&A.slice(s![0,..,..])*&(A.slice(s![0,..,..]).reversed_axes());
                        let A_yy=&A.slice(s![1,..,..])*&(A.slice(s![1,..,..]).reversed_axes());
                        let A_xy=&A.slice(s![0,..,..])*&(A.slice(s![1,..,..]).reversed_axes());
                        let re_xx:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_xx;
                        let re_yy:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_yy;
                        let Complex { re, im } = A_xy.view().split_complex();
                        let re_xy:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_xy:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));

                        #[inline]
                        fn accumulate(mat: &Array2<Complex<f64>>, kernel: &Array2<Complex<f64>>, fermi: &Array1<Complex<f64>>) -> Complex<f64> {
                            let mut sum = Complex::new(0.0, 0.0);
                            for (i, (mr, kr)) in mat.outer_iter().zip(kernel.outer_iter()).enumerate() {
                                sum += mr.dot(&kr) * fermi[i];
                            }
                            sum
                        }

                        Zip::from(omega_n.axis_iter_mut(Axis(1)))
                            .and(matric_n.axis_iter_mut(Axis(1)))
                            .and(og.view())
                            .par_for_each(|mut omega, mut matric, a0| {
                                let li_eta = a0 + li * eta;
                                let li_eta_sq = li_eta * li_eta;
                                let UU = U0_sq.mapv(|x| (Complex::new(x, 0.0) - li_eta_sq).finv());
                                let U1: Array2<Complex<f64>> = &UU * &Us * li_eta;

                                matric[[0]] = accumulate(&re_xx, &U1, &fermi_dirac);
                                matric[[1]] = accumulate(&re_yy, &U1, &fermi_dirac);

                                omega[[0]]  = accumulate(&im_xy, &UU, &fermi_dirac);
                                matric[[2]] = accumulate(&re_xy, &U1, &fermi_dirac);
                            });
                        (matric_n,omega_n)
                    },
                    _=>panic!("Wrong, self.dim_r must be 2 or 3 for using optical_conductivity_all_direction")
                };
                (matric_n,omega_n)
            }).collect();
        let (matric_sum, omega_sum) = match self.dim_r() {
            3 => {
                let omega = omega
                    .into_iter()
                    .fold(Array2::zeros((3, n_og)), |omega_acc, omega| {
                        omega_acc + omega
                    });
                let matric = matric
                    .into_iter()
                    .fold(Array2::zeros((6, n_og)), |matric_acc, matric| {
                        matric_acc + matric
                    });
                (matric, omega)
            }
            2 => {
                let omega = omega
                    .into_iter()
                    .fold(Array2::zeros((1, n_og)), |omega_acc, omega| {
                        omega_acc + omega
                    });
                let matric = matric
                    .into_iter()
                    .fold(Array2::zeros((3, n_og)), |matric_acc, matric| {
                        matric_acc + matric
                    });
                (matric, omega)
            }
            _ => panic!(
                "Wrong, self.dim_r must be 2 or 3 for using optical_conductivity_all_direction"
            ),
        };
        let matric_sum = li * matric_sum / self.lat.det().unwrap() / (nk as f64);
        let omega_sum = li * omega_sum / self.lat.det().unwrap() / (nk as f64);
        Ok((matric_sum, omega_sum))
    }
}

// ── Simplex quadrature ────────────────────────────────────────────────────

/// Integrate the optical conductivity kernel over the BZ.
///
/// Returns the complex conductivity `σ^{ab}` in fractional‑coordinate
/// volume.  Divide by `det(lat)` for Cartesian.
pub fn integrate(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    omega: f64,
    eta: f64,
    mu: f64,
    T: f64,
) -> Complex<f64> {
    let dim = k_mesh.len();
    let beta = if T > 0.0 {
        1.0 / (T * 8.617333262e-5)
    } else {
        0.0
    };
    let mut total = Complex::new(0.0, 0.0);

    match dim {
        2 => {
            let (nx, ny) = (k_mesh[0], k_mesh[1]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for sim in &build_triangles_2d(ix, iy, nx, ny, inv_nx, inv_ny, all_pts) {
                        total += quadrature_optical_simplex(sim, omega, eta, mu, beta);
                    }
                }
            }
        }
        3 => {
            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            let inv_nz = 1.0 / nz as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        for sim in &build_tetrahedra_3d(
                            ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts,
                        ) {
                            total += quadrature_optical_simplex(sim, omega, eta, mu, beta);
                        }
                    }
                }
            }
        }
        _ => panic!("optical::integrate: only dim=2,3 supported"),
    }

    total
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Optical conductivity via simplex quadrature.
    pub fn optical_conductivity_simplex(
        &self,
        k_mesh: &Array1<usize>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        omega: f64,
        eta: f64,
        mu: f64,
        T: f64,
    ) -> Result<Complex<f64>> {
        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let gauge = Gauge::Atom;

        let all_pts: Vec<VertexKernel> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_velocity_kernel(&kv, dir_a, dir_b, None, gauge, None)
            })
            .collect();

        let sigma = integrate(&all_pts, k_mesh, omega, eta, mu, T);
        let det = self.lat.det().unwrap();
        Ok(sigma / det)
    }
}
