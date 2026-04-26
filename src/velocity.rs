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
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length {} must equal to the dimension of model {}.",
            kvec.len(),
            self.dim_r()
        );

        let n_r = self.hamR.nrows();
        let dim = self.dim_r();
        let nsta = self.nsta();

        // Fused phase factor: R·k → exp(i 2π R·k) in one pass
        let Us: Vec<Complex<f64>> = (0..n_r)
            .map(|i| {
                let mut phase = 0.0f64;
                for d in 0..dim {
                    phase += self.hamR[[i, d]] as f64 * kvec[d];
                }
                Complex::new(0.0, 2.0 * PI * phase).exp()
            })
            .collect();

        let mut v = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));
        let R0 = &self.hamR.mapv(|x| Complex::<f64>::new(x as f64, 0.0));
        let R0 = R0.dot(&self.lat.mapv(|x| Complex::new(x, 0.0)));

        // Mutable accumulate (no intermediate per-R allocations)
        let mut hamk = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for i in 0..n_r {
            let u = Us[i];
            let hm = self.ham.slice(s![i, .., ..]);
            Zip::from(&mut hamk).and(&hm).for_each(|a, &b| *a += b * u);
        }
        let (v, hamk) = match gauge {
            Gauge::Atom => {
                let orb_sta = if self.spin {
                    let orb0 = concatenate(Axis(0), &[self.orb.view(), self.orb.view()]).unwrap();
                    orb0
                } else {
                    self.orb.to_owned()
                };
                // Fused: τ·k → exp(i 2π τ·k) in one pass
                let orb_phase: Vec<Complex<f64>> = orb_sta
                    .outer_iter()
                    .map(|tau| {
                        let mut phase = 0.0f64;
                        for d in 0..dim {
                            phase += tau[d] * kvec[d];
                        }
                        Complex::new(0.0, 2.0 * PI * phase).exp()
                    })
                    .collect();
                let U = Array2::from_diag(&Array1::from(orb_phase.clone()));
                let U_conj = Array2::from_diag(&Array1::from(orb_phase.iter().map(|x| x.conj()).collect::<Vec<_>>()));
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
                // Mutable accumulate velocity per direction (no intermediate allocations)
                for d in 0..dim {
                    let mut vv = Array2::<Complex<f64>>::zeros((nsta, nsta));
                    for i_r in 0..n_r {
                        let factor = Us[i_r] * R0[[i_r, d]] * Complex::i();
                        let hm = self.ham.slice(s![i_r, .., ..]);
                        Zip::from(&mut vv).and(&hm).for_each(|a, &b| *a += b * factor);
                    }
                    let det_tau = UU.slice(s![d, .., ..]);
                    let vv = &vv + &hamk * &det_tau;
                    let vv = &U_conj.dot(&vv);
                    let vv = vv.dot(&U);
                    v.slice_mut(s![d, .., ..]).assign(&vv);
                }
                // At this point, we have computed sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)}
                // Next, compute Berry connection A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                let hamk = U_conj.dot(&hamk.dot(&U)); // Don't forget to add the phase to hamk
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let n_rmat = self.rmatrix.len_of(Axis(0));
                    let mut rk = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));
                    for i_r in 0..n_rmat {
                        let u = Us[i_r];
                        let rm = self.rmatrix.slice(s![i_r, .., .., ..]);
                        Zip::from(&mut rk).and(&rm).for_each(|a, &b| *a += b * u);
                    }
                    for i in 0..dim {
                        let mut r0 = rk.slice_mut(s![i, .., ..]);
                        let r_new = r0.dot(&U);
                        let r_new = U_conj.dot(&r_new);
                        r0.assign(&r_new);
                        let mut dig = r0.diag_mut();
                        dig.assign(&Array1::zeros(nsta));
                        let a_comm = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&a_comm);
                    }
                }
                (v, hamk)
            }
            Gauge::Lattice => {
                // Mutable accumulate velocity per direction
                for d in 0..dim {
                    let mut vv = Array2::<Complex<f64>>::zeros((nsta, nsta));
                    for i_r in 0..n_r {
                        let factor = Us[i_r] * R0[[i_r, d]] * Complex::i();
                        let hm = self.ham.slice(s![i_r, .., ..]);
                        Zip::from(&mut vv).and(&hm).for_each(|a, &b| *a += b * factor);
                    }
                    v.slice_mut(s![d, .., ..]).assign(&vv);
                }
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let n_rmat = self.rmatrix.len_of(Axis(0));
                    let mut rk = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));
                    for i_r in 0..n_rmat {
                        let u = Us[i_r];
                        let rm = self.rmatrix.slice(s![i_r, .., .., ..]);
                        Zip::from(&mut rk).and(&rm).for_each(|a, &b| *a += b * u);
                    }
                    for i in 0..dim {
                        let r0 = rk.slice(s![i, .., ..]);
                        let a_comm = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&a_comm);
                    }
                }
                (v, hamk)
            }
        };
        (v, hamk)
    }
}
