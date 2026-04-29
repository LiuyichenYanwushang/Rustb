//! Partial eigensolver bindings to LAPACK routines (zheevx, zheevr, zheev).
//!
//! This module provides functions for solving Hermitian eigenvalue problems
//! in a specified energy window, using LAPACK's `zheevx` (expert driver with
//! eigenvalue range selection), `zheevr` (relative robust representation),
//! and `zheev` (full diagonalization).
//!
//! # Backend selection
//!
//! The LAPACK backend is chosen via Cargo features:
//! - `intel-mkl-static` / `intel-mkl-system`: Intel MKL.
//! - `openblas-static` / `openblas-system`: OpenBLAS.
//! - `netlib-static` / `netlib-system`: reference netlib LAPACK.

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src as _src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src as _src;

#[cfg(any(feature = "netlib-system", feature = "netlib-static"))]
extern crate netlib_src as _src;

use lapack::{cheevx, zheev, zheevr, zheevr_2stage, zheevx};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::EigValsh;
use ndarray_linalg::UPLO;
use num_complex::Complex;
use std::ffi::c_char;

/// Compute selected eigenvalues and eigenvectors of a complex Hermitian matrix
/// using LAPACK's `zheevx` (expert driver).
///
/// # Parameters
///
/// - `x`: the input Hermitian matrix.
/// - `range`: `(v_low, v_high)` -- eigenvalue range to search.
/// - `epsilon`: absolute tolerance for eigenvalue convergence.
/// - `uplo`: whether the upper or lower triangle of `x` is stored.
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` where eigenvalues is `Array1<f64>` and
/// eigenvectors is `Array2<Complex<f64>>` of shape `(n_found, n)`.
/// Each row of the eigenvector matrix is an eigenvector.
///
/// # Panics
///
/// Panics if `zheevx` returns a non-zero info code.
pub fn eigh_x<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> (Array1<f64>, Array2<Complex<f64>>)
where
    S: Data<Elem = Complex<f64>>,
{
    let n = x.shape()[0] as i32;
    let mut a: Vec<_> = x.iter().cloned().collect();
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut m = 0;
    let mut info = 0;
    let mut ifail = vec![0; n as usize];
    let mut work = vec![Complex::new(0.0, 0.0); (2 * n) as usize];
    let mut rwork = vec![0.0; (7 * n) as usize];
    let mut iwork = vec![0; (5 * n) as usize];
    let job1 = b'V'; // compute eigenvectors
    let job2 = b'V'; // eigenvalues in range
    let job3 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };

    unsafe {
        zheevx(
            job1,
            job2,
            job3,
            n,
            &mut a,
            n,
            range.0,
            range.1,
            0,
            n,
            epsilon,
            &mut m,
            &mut w,
            &mut z,
            n,
            &mut work,
            2 * n,
            &mut rwork,
            &mut iwork,
            &mut ifail,
            &mut info,
        );
    }
    if info == 0 {
        (
            Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect()),
            Array2::<Complex<f64>>::from_shape_vec(
                [m as usize, n as usize],
                z.into_iter().take((n * m) as usize).collect(),
            )
            .unwrap(),
        )
    } else {
        panic!("zheevx failed with info = {}", info);
    }
}

/// Compute selected eigenvalues only (no eigenvectors) of a complex Hermitian
/// matrix using LAPACK's `zheevx`.
///
/// # Parameters
///
/// - `x`: the input Hermitian matrix.
/// - `range`: `(v_low, v_high)` -- eigenvalue range to search.
/// - `epsilon`: absolute tolerance for eigenvalue convergence.
/// - `uplo`: whether the upper or lower triangle of `x` is stored.
///
/// # Returns
///
/// `Array1<f64>` of eigenvalues in the specified range.
///
/// # Panics
///
/// Panics if `zheevx` returns a non-zero info code.
pub fn eigvalsh_x<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> Array1<f64>
where
    S: Data<Elem = Complex<f64>>,
{
    let n = x.shape()[0] as i32;
    let mut a: Vec<_> = x.iter().cloned().collect();
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut m = 0;
    let mut info = 0;
    let mut ifail = vec![0; n as usize];
    let mut work = vec![Complex::new(0.0, 0.0); (2 * n) as usize];
    let mut rwork = vec![0.0; (7 * n) as usize];
    let mut iwork = vec![0; (5 * n) as usize];
    let job1 = b'N'; // eigenvalues only
    let job2 = b'V'; // eigenvalues in range
    let job3 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };

    unsafe {
        zheevx(
            job1,
            job2,
            job3,
            n,
            &mut a,
            n,
            range.0,
            range.1,
            0,
            n,
            epsilon,
            &mut m,
            &mut w,
            &mut z,
            n,
            &mut work,
            2 * n,
            &mut rwork,
            &mut iwork,
            &mut ifail,
            &mut info,
        );
    }
    if info == 0 {
        Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect())
    } else {
        panic!("zheevx failed with info = {}", info);
    }
}

/// Compute selected eigenvalues and eigenvectors of a complex Hermitian matrix
/// using LAPACK's `zheevr` (relative robust representation).
///
/// This is generally faster than `zheevx` for large matrices when only a subset
/// of eigenvalues is needed.
///
/// # Parameters
///
/// - `x`: the input Hermitian matrix.
/// - `range`: `(v_low, v_high)` -- eigenvalue range to search.
/// - `epsilon`: absolute tolerance for eigenvalue convergence.
/// - `uplo`: whether the upper or lower triangle of `x` is stored.
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` where eigenvalues is `Array1<f64>` and
/// eigenvectors is `Array2<Complex<f64>>` of shape `(n_found, n)`.
pub fn eigh_r<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> (Array1<f64>, Array2<Complex<f64>>)
where
    S: Data<Elem = Complex<f64>>,
{
    let job1 = b'V'; // compute eigenvectors
    let job2 = b'V'; // eigenvalues in range
    let job3 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };
    let n = x.shape()[0] as i32;
    let mut a: Vec<_> = x.iter().cloned().collect();
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut isuppz = vec![0; 2 * n as usize];
    let mut m = 0;
    let mut info = 0;
    let lwork = n * 33 as i32;
    let liwork = n * 10 as i32;
    let lrwork = n * 24 as i32;
    let mut work = vec![Complex::new(0.0, 0.0); lwork as usize];
    let mut rwork = vec![0.0; lrwork as usize];
    let mut iwork = vec![0; liwork as usize];

    unsafe {
        zheevr(
            job1,
            job2,
            job3,
            n,
            &mut a,
            n,
            range.0,
            range.1,
            0,
            n,
            epsilon,
            &mut m,
            &mut w,
            &mut z,
            n,
            &mut isuppz,
            &mut work,
            lwork,
            &mut rwork,
            lrwork,
            &mut iwork,
            liwork,
            &mut info,
        );
    }

    if info == 0 {
        (
            Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect()),
            Array2::<Complex<f64>>::from_shape_vec(
                [m as usize, n as usize],
                z.into_iter().take((n * m) as usize).collect(),
            )
            .unwrap(),
        )
    } else {
        panic!("zheevr failed with info = {}", info);
    }
}

/// Compute selected eigenvalues only (no eigenvectors) of a complex Hermitian
/// matrix using LAPACK's `zheevr`.
///
/// # Parameters
///
/// - `x`: the input Hermitian matrix.
/// - `range`: `(v_low, v_high)` -- eigenvalue range to search.
/// - `epsilon`: absolute tolerance for eigenvalue convergence.
/// - `uplo`: whether the upper or lower triangle of `x` is stored.
///
/// # Returns
///
/// `Array1<f64>` of eigenvalues in the specified range.
pub fn eigvalsh_r<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> Array1<f64>
where
    S: Data<Elem = Complex<f64>>,
{
    let n = x.shape()[0] as i32;
    let mut a: Vec<_> = x.iter().cloned().collect();
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut isuppz = vec![0; 2 * n as usize];
    let mut m = 0;
    let mut info = 0;
    // Workspace query
    let mut work = vec![Complex::new(0.0, 0.0); 1 as usize];
    let mut rwork = vec![0.0; 1 as usize];
    let mut iwork = vec![0; 1 as usize];
    let job1 = b'N'; // eigenvalues only
    let job2 = b'V'; // eigenvalues in range
    let job3 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };

    unsafe {
        zheevr(
            job1,
            job2,
            job3,
            n,
            &mut a,
            n,
            range.0,
            range.1,
            0,
            n,
            epsilon,
            &mut m,
            &mut w,
            &mut z,
            n,
            &mut isuppz,
            &mut work,
            -1,
            &mut rwork,
            -1,
            &mut iwork,
            -1,
            &mut info,
        );
    }

    let lwork = work[0].re as i32;
    let liwork = iwork[0] as i32;
    let lrwork = rwork[0] as i32;
    let mut work = vec![Complex::new(0.0, 0.0); lwork as usize];
    let mut rwork = vec![0.0; lrwork as usize];
    let mut iwork = vec![0; liwork as usize];

    unsafe {
        zheevr(
            job1,
            job2,
            job3,
            n,
            &mut a,
            n,
            range.0,
            range.1,
            0,
            n,
            epsilon,
            &mut m,
            &mut w,
            &mut z,
            n,
            &mut isuppz,
            &mut work,
            lwork,
            &mut rwork,
            lrwork,
            &mut iwork,
            liwork,
            &mut info,
        );
    }
    if info == 0 {
        Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect())
    } else {
        panic!("zheevr failed with info = {}", info);
    }
}

/// Compute all eigenvalues of a complex Hermitian matrix using LAPACK's `zheev`
/// (simple driver for full diagonalization).
///
/// # Parameters
///
/// - `x`: the input Hermitian matrix.
/// - `uplo`: whether the upper or lower triangle of `x` is stored.
///
/// # Returns
///
/// `Array1<f64>` of all eigenvalues.
///
/// # Panics
///
/// Panics if `zheev` returns a non-zero info code.
pub fn eigvalsh_v<S>(x: &ArrayBase<S, Ix2>, uplo: UPLO) -> Array1<f64>
where
    S: Data<Elem = Complex<f64>>,
{
    let n = x.shape()[0] as i32;
    let mut a: Vec<_> = x.iter().cloned().collect();
    let mut w = vec![0.0; n as usize];
    let mut m = 0;
    let mut info = 0;
    // Workspace query
    let mut work = vec![Complex::new(0.0, 0.0); 1 as usize];
    let mut rwork = vec![0.0; (3 * n - 2) as usize];
    let job1 = b'N'; // eigenvalues only
    let job2 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };

    unsafe {
        zheev(
            job1, job2, n, &mut a, n, &mut w, &mut work, -1, &mut rwork, &mut info,
        );
    }
    let lwork = work[0].re as i32;
    work = vec![Complex::new(0.0, 0.0); lwork as usize];

    unsafe {
        zheev(
            job1, job2, n, &mut a, n, &mut w, &mut work, lwork, &mut rwork, &mut info,
        );
    }
    if info == 0 {
        Array1::<f64>::from_vec(w.into_iter().collect())
    } else {
        panic!("zheev failed with info = {}", info);
    }
}
