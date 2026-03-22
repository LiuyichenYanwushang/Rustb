//! This module calculates the optical conductivity
/// The adopted definition is
/// $$\sigma_{\ap\bt}=\f{2ie^2\hbar}{V}\sum_{\bm k}\sum_{n} f_n (g_{n,\ap\bt}+\f{i}{2}\Og_{n,\ap\bt})$$
///
/// Where
/// $$\\begin{aligned}
/// g_{n\ap\bt}&=\sum_{m=\not n}\f{\og-i\eta}{\ve_{n\bm k}-\ve_{m\bm k}}\f{\text{Re} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og-i\eta)^2}\\\\
/// \Og_{n\ap\bt}&=\sum_{m=\not n}\f{\text{Re} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og-i\eta)^2}
/// \\end{aligned}
/// $$
///
use crate::error::{Result, TbError};
use crate::kpoints::{gen_kmesh, gen_krange};
use crate::math::*;
use crate::phy_const::mu_B;
use crate::solve_ham::solve;
use crate::velocity::*;
use crate::{Gauge, Model};
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::AddAssign;
use std::ops::MulAssign;

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

impl OpticalGeometry for Model {
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
        //let (band, evec) = self.solve_onek(&k_vec);

        let (mut v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
            self.gen_v(&k_vec, Gauge::Atom); //这是速度算符
        let mut J = v.view();

        // Project the velocity operator onto the direction dir_1
        let J = J
            .outer_iter()
            .zip(dir_1.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                acc + &x * (*d + 0.0 * li)
            });

        // Project the velocity operator onto the direction dir_2
        let v = v
            .outer_iter()
            .zip(dir_2.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                acc + &x * (*d + 0.0 * li)
            });

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

impl Model {
    pub fn optical_conductivity(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        T: f64,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array1<Complex<f64>>, Array1<Complex<f64>>)>
//针对单个的
    {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
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
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
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

    ///直接计算 xx, yy, zz, xy, yz, xz 这六个量的光电导, 分为对称和反对称部分.
    ///输出格式为 ($\sigma_{ab}^S$, $\sigma_{ab}^A), 这里 S 和 A 表示 symmetry and antisymmetry.
    ///$sigma_{ab}^S$ 是 $6\times n_\omega$
    ///如果是二维系统, 那么输出 xx yy xy 这三个分量
    pub fn optical_conductivity_all_direction(
        &self,
        k_mesh: &Array1<usize>,
        T: f64,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let (matric,omega):(Vec<_>,Vec<_>)=kvec.outer_iter().into_par_iter()
            .map(|k| {
                //let (band, evec) = self.solve_onek(&k);
                let (mut v, hamk): (Array3<Complex<f64>>,Array2<Complex<f64>>) = self.gen_v(&k,Gauge::Atom); //这是速度算符
                let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
                    (eigvals, eigvecs)
                } else {
                    todo!()
                };
                let evec_conj=evec.t();
                let evec= evec.mapv(|x| x.conj());

                let mut A = Array3::zeros((self.dim_r(),self.nsta(),self.nsta()));
                //transfrom the basis into bolch state
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
                        // Calculate the matrices for each frequency
                        Zip::from(omega_n.axis_iter_mut(Axis(1)))
                            .and(matric_n.axis_iter_mut(Axis(1)))
                            .and(og.view())
                            .par_for_each(|mut omega, mut matric, a0| {
                                let li_eta = a0 + li * eta;
                                let UU = U0.mapv(|x| (x*x - li_eta*li_eta).finv());
                                let U1:Array2<Complex<f64>> = &UU * &Us * li_eta;

                                let m = re_xx.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[0]]=m;
                                let m = re_yy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[1]]=m;
                                let m = re_zz.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[2]]=m;

                                let o = im_xy.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_xy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[0]]=o;
                                matric[[3]]=m;
                                let o = im_yz.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_yz.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[1]]=o;
                                matric[[4]]=m;
                                let o = im_xz.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_xz.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[2]]=o;
                                matric[[5]]=m;
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
                        // Calculate the matrices for each frequency
                        Zip::from(omega_n.axis_iter_mut(Axis(1)))
                            .and(matric_n.axis_iter_mut(Axis(1)))
                            .and(og.view())
                            .par_for_each(|mut omega, mut matric, a0| {
                                let li_eta = a0 + li * eta;
                                let UU = U0.mapv(|x| (x*x - li_eta*li_eta).finv());
                                let U1:Array2<Complex<f64>> = &UU * &Us * li_eta;

                                let m = re_xx.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[0]]=m;
                                let m = re_yy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[1]]=m;

                                let o = im_xy.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_xy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[0]]=o;
                                matric[[2]]=m;
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
