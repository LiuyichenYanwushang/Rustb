//!这个模块是用来求解哈密顿量的
use crate::Gauge;
use crate::Model;
use crate::error::{Result, TbError};
use crate::ndarray_lapack::{eigh_r, eigvalsh_r, eigvalsh_v};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use std::f64::consts::PI;
/// 这个模块主要用于求解模型的哈密顿量, 目前针对的是TB模型, 日后会增加Hubbard model的平均场方法
///
///

pub trait solve {
    /// Solve energy bands at a single k-point
    fn solve_band_onek<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>) -> Array1<f64>;
    /// Solve energy bands at a single k-point with a range
    fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> Array1<f64>;
    /// Solve energy bands at all given k-points
    fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Array2<f64>;
    /// Solve energy bands at all given k-points with parallel method, i.e., rayon
    fn solve_band_all_parallel<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>)
    -> Array2<f64>;
    /// Solve energy bands and eigenvectors at a single k-point
    fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> (Array1<f64>, Array2<Complex<f64>>);
    fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> (Array1<f64>, Array2<Complex<f64>>);
    fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>);
    fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>);
}
impl solve for Model {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn solve_band_onek<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>) -> Array1<f64> {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(kvec, Gauge::Atom);
        let eval = eigvalsh_v(&hamk, UPLO::Upper);
        eval
    }

    fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> Array1<f64> {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        let eval = eigvalsh_r(&hamk, range, epsilon, UPLO::Upper);
        eval
    }
    fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Array2<f64> {
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .for_each(|x, mut a| {
                let eval = self.solve_band_onek(&x);
                a.assign(&eval);
            });
        band
    }
    #[allow(non_snake_case)]
    fn solve_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Array2<f64> {
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .par_for_each(|x, mut a| {
                let eval = self.solve_band_onek(&x);
                a.assign(&eval);
            });
        band
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> (Array1<f64>, Array2<Complex<f64>>) {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        let (eval, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec = conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>>(&evec);
        (eval, evec)
    }
    fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> (Array1<f64>, Array2<Complex<f64>>) {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        let (eval, evec) = eigh_r(&hamk, range, epsilon, UPLO::Upper);
        let evec = evec.mapv(|x| x.conj());
        (eval, evec)
    }

    #[allow(non_snake_case)]
    fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        let mut vectors = Array3::<Complex<f64>>::zeros((nk, self.nsta(), self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .and(vectors.outer_iter_mut())
            .for_each(|x, mut a, mut b| {
                let (eval, evec) = self.solve_onek(&x);
                a.assign(&eval);
                b.assign(&evec);
            });
        (band, vectors)
    }
    #[allow(non_snake_case)]
    fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        let mut vectors = Array3::<Complex<f64>>::zeros((nk, self.nsta(), self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .and(vectors.outer_iter_mut())
            .par_for_each(|x, mut a, mut b| {
                let (eval, evec) = self.solve_onek(&x);
                a.assign(&eval);
                b.assign(&evec);
            });
        (band, vectors)
    }
}
