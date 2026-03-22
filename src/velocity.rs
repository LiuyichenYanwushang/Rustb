//!这个模块是用来产生某个k点的速度算符
use crate::Gauge;
use crate::Model;
use crate::comm;
use crate::solve_ham::*;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
/// This function generates the velocity operator, i.e., $\bra{m\bm k}\p_\ap H_{\bm k}\ket{n\bm k},$
/// The basis functions are Bloch wavefunctions.
///
/// The velocity operator formula uses the tight-binding model,
/// where the Fourier transform includes atomic positions.
///
/// Thus we have
///
/// $$
/// \\begin\{aligned\}
/// \\bra{m\bm k}\p_\ap H_{\bm k}\ket{n\bm k}&=\p_\ap\left(\bra{m\bm k} H\ket{n\bm k}\rt)-\p_\ap\left(\bra{m\bm k}\rt) H\ket{n\bm k}-\bra{m\bm k} H\p_\ap\ket{n\bm k}\\\\
/// &=\sum_{\bm R} i(\bm R-\bm\tau_m+\bm\tau_n)H_{mn}(\bm R) e^{i\bm k\cdot(\bm R-\bm\tau_m+\bm\tau_n)}-\lt[H_{\bm k},\\mathcal A_{\bm k,\ap}\rt]_{mn}
/// \\end\{aligned\}
/// $$
///
/// Here $\\mathcal A_{\bm k}$ is defined as $$\\mathcal A_{\bm k,\ap,mn}=-i\sum_{\bm R}r_{mn,\ap}(\bm R)e^{i\bm k\cdot(\bm R-\bm\tau_m+\bm\tau_{n})}+i\tau_{n\ap}\dt_{mn}$$
/// where $\bm r_{mn}$ can be provided by wannier90 by setting write_rmn=true
/// Here, all $\bm R$, $\bm r$, and $\bm \tau$ are in real-space coordinates.
///
use num_complex::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::ops::AddAssign;

pub trait Velocity {
    fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>);
}

impl Velocity for Model {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        // We use lattice gauge rather than atomic gauge
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length {} must equal to the dimension of model {}.",
            kvec.len(),
            self.dim_r()
        );

        let Us = (self.hamR.mapv(|x| x as f64))
            .dot(kvec)
            .mapv(|x| Complex::<f64>::new(x, 0.0));
        let Us = Us * Complex::new(0.0, 2.0 * PI);
        let Us = Us.mapv(Complex::exp); // Us is exp(i k R)
        // Define an initialized velocity matrix
        let mut v = Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
        let R0 = &self.hamR.mapv(|x| Complex::<f64>::new(x as f64, 0.0));
        // R0 is the real-space hamR
        let R0 = R0.dot(&self.lat.mapv(|x| Complex::new(x, 0.0)));
        let hamk: Array2<Complex<f64>> = self
            .ham
            .outer_iter()
            .zip(Us.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (hm, u)| {
                acc + &hm * *u
            });
        let (v, hamk) = match gauge {
            Gauge::Atom => {
                let orb_sta = if self.spin {
                    let orb0 = concatenate(Axis(0), &[self.orb.view(), self.orb.view()]).unwrap();
                    orb0
                } else {
                    self.orb.to_owned()
                };
                let U0 = orb_sta.dot(kvec);
                let U0 = U0.mapv(|x| Complex::<f64>::new(x, 0.0));
                let U0 = U0 * Complex::new(0.0, 2.0 * PI);
                let mut U0 = U0.mapv(Complex::exp);
                // U0 is the phase factor
                let U = Array2::from_diag(&U0);
                let U_conj = Array2::from_diag(&U0.mapv(|x| x.conj()));
                let orb_real = orb_sta.dot(&self.lat);
                // Start constructing -orb_real[[i,r]]+orb_real[[j,r]];-----------------
                let mut UU = Array3::<f64>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                let A = orb_real.view().insert_axis(Axis(2));
                let A = A
                    .broadcast((self.nsta(), self.dim_r(), self.nsta()))
                    .unwrap()
                    .permuted_axes([1, 0, 2]);
                let mut B = A.view().permuted_axes([0, 2, 1]);
                let UU = &B - &A;
                let UU = UU.mapv(|x| Complex::<f64>::new(0.0, x)); //UU[i,j]=i(-tau[i]+tau[j])
                // Define an initialized velocity matrix
                Zip::from(v.outer_iter_mut())
                    .and(R0.axis_iter(Axis(1)))
                    .and(UU.outer_iter())
                    .for_each(|mut v0, r, det_tau| {
                        let vv: Array2<Complex<f64>> =
                            self.ham.outer_iter().zip(Us.iter().zip(r.iter())).fold(
                                Array2::zeros((self.nsta(), self.nsta())),
                                |acc, (ham, (us, r0))| acc + &ham * *us * *r0 * Complex::i(),
                            );
                        let vv: Array2<Complex<f64>> = &vv + &hamk * &det_tau;
                        let vv = &U_conj.dot(&vv);
                        let vv = vv.dot(&U); // Next two steps fill in the phase due to orbital coordinates
                        v0.assign(&vv);
                    });
                // At this point, we have computed sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)}
                // Next, compute Berry connection A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                let hamk = U_conj.dot(&hamk.dot(&U)); // Don't forget to add the phase to hamk
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let mut rk =
                        Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                    let mut rk = self
                        .rmatrix
                        .axis_iter(Axis(0))
                        .zip(Us.iter())
                        .fold(rk, |acc, (ham, us)| acc + &ham * *us);
                    for i in 0..3 {
                        let mut r0: ArrayViewMut2<Complex<f64>> = rk.slice_mut(s![i, .., ..]);
                        let r_new = r0.dot(&U);
                        let r_new = U_conj.dot(&r_new);
                        r0.assign(&r_new);
                        let mut dig = r0.diag_mut();
                        //dig.assign(&(&dig - &orb_real.column(i)));
                        dig.assign(&Array1::zeros(self.nsta()));
                        let A = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&A);
                    }
                }
                (v, hamk)
            }
            Gauge::Lattice => {
                // Use lattice gauge
                Zip::from(v.outer_iter_mut())
                    .and(R0.axis_iter(Axis(1)))
                    .for_each(|mut v0, r| {
                        let vv: Array2<Complex<f64>> =
                            self.ham.outer_iter().zip(Us.iter().zip(r.iter())).fold(
                                Array2::zeros((self.nsta(), self.nsta())),
                                |acc, (ham, (us, r0))| acc + &ham * *us * *r0 * Complex::i(),
                            );
                        v0.assign(&vv);
                    });
                // At this point, we have computed sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)}
                // Next, compute Berry connection A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let mut rk =
                        Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                    let mut rk = self
                        .rmatrix
                        .axis_iter(Axis(0))
                        .zip(Us.iter())
                        .fold(rk, |acc, (ham, us)| acc + &ham * *us);
                    for i in 0..3 {
                        let mut r0: ArrayViewMut2<Complex<f64>> = rk.slice_mut(s![i, .., ..]);
                        //let mut dig = r0.diag_mut();
                        //dig.assign(&Array1::zeros(self.nsta()));
                        let A = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&A);
                    }
                }
                (v, hamk)
            }
        };
        (v, hamk)
    }
}
