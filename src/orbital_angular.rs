//! This module calculates the orbital Hall conductivity and the orbital angular moment
/// The calculation using the orbial magnetism , refer to PHYSICAL REVIEW B 106, 104414 (2022).
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
pub trait OrbitalAngular: Velocity {
    //! Computes the orbital angular momentum at a single k-point.
    //! The orbital angular momentum is defined as
    //! $$\bra{u_{m\bm k}}\bm L\ket{u_{n\bm k}}=\frac{1}{4i
    //! g_L\mu_B}\sum_{\ell=\not m,n}\f{2\ve_{\ell\bm k}-\ve_{m\bm k}-\ve_{n\bm k}}{(\ve_{m\bm
    //! k}-\ve_{\ell\bm k})(\ve_{n\bm k}-\ve_{\ell\bm k})}\bra{u_{m\bm k}}\p_{\bm k} H_{\bm k}\ket{u_{\ell\bm k}}\times\bra{u_{\ell\bm k}}\p_{\bm k} H_{\bm k}\ket{u_{n\bm k}}$$
    fn orbital_angular_momentum_onek(&self, kvec: &Array1<f64>) -> Array3<Complex<f64>>;
}
impl<const SPIN: bool> OrbitalAngular for Model<SPIN> {
    fn orbital_angular_momentum_onek(&self, kvec: &Array1<f64>) -> Array3<Complex<f64>> {
        let li = Complex::<f64>::new(0.0, 1.0);
        let (v, hamk) = self.gen_v(kvec, Gauge::Atom);
        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let mut L = Array3::zeros((self.dim_r(), self.nsta(), self.nsta()));
        // m,n,l
        let mut U = Array3::zeros((self.nsta(), self.nsta(), self.nsta()));
        for (i, e1) in evec.iter().enumerate() {
            for (j, e2) in evec.iter().enumerate() {
                for (k, e3) in evec.iter().enumerate() {
                    U[[i, j, k]] = (2.0 * e3 - e1 - e2) / (e1 - e3) / (e2 - e3);
                }
            }
        }
        //g_L 是朗德g因子, 这个朗德g因子也是随着轨道而变化的
        let g_L = 1.0;
        for r in 0..self.dim_r() {
            for i in 0..self.nsta() {
                for j in 0..self.nsta() {
                    L[[r, i, j]] = -li / 4.0 / g_L / mu_B;
                }
            }
        }
        L
    }
}
