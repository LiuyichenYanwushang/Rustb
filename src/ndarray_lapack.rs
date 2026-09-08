//! Partial eigensolver bindings to LAPACK routines (zheevx, zheevr, zheev).
//!
//! This module provides functions for solving Hermitian eigenvalue problems
//! in a specified energy window, using LAPACK's `zheevx` (expert driver with
//! eigenvalue range selection), `zheevr` (relative robust representation),
//! and `zheev` (full diagonalization).
//!
//! All eigensolvers return [`Result`], accept any ndarray layout, and interpret
//! `uplo` in the logical input matrix. Empty square matrices return empty results.
//! Selected eigenvalue ranges are half-open `(v_low, v_high]`.
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

use crate::error::{Result, TbError};
use lapack::{zheev, zheevr, zheevx};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::UPLO;
use num_complex::Complex;

/// Safe wrapper around BLAS `zaxpy`: `y += alpha * x` for `Complex<f64>` slices.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
#[inline]
pub fn zaxpy(alpha: Complex<f64>, x: &[Complex<f64>], y: &mut [Complex<f64>]) {
    assert_eq!(x.len(), y.len(), "zaxpy: x and y must have the same length");
    let n = i32::try_from(x.len()).expect("zaxpy: vector length exceeds LAPACK integer range");
    unsafe { blas::zaxpy(n, alpha, x, 1, y, 1) };
}

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
/// Each row contains the ket coefficients of an eigenvector of `x`.
/// Inputs may have any ndarray layout; only the triangle selected by `uplo` is read.
///
/// # Errors
///
/// Returns an error for non-square inputs, invalid ranges/tolerances, dimensions
/// exceeding LAPACK integer limits, or a non-zero LAPACK info code.
pub fn eigh_x<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> Result<(Array1<f64>, Array2<Complex<f64>>)>
where
    S: Data<Elem = Complex<f64>>,
{
    let (n, mut a) = prepare_matrix(x)?;
    validate_range(range, epsilon)?;
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); n as usize * n as usize];
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
        Ok((
            Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect()),
            Array2::<Complex<f64>>::from_shape_vec(
                [m as usize, n as usize],
                z.into_iter().take(n as usize * m as usize).collect(),
            )?,
        ))
    } else {
        Err(TbError::Lapack {
            routine: "zheevx",
            info,
        })
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
/// # Errors
///
/// Returns an error for non-square inputs, invalid ranges/tolerances, dimensions
/// exceeding LAPACK integer limits, or a non-zero LAPACK info code.
pub fn eigvalsh_x<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> Result<Array1<f64>>
where
    S: Data<Elem = Complex<f64>>,
{
    let (n, mut a) = prepare_matrix(x)?;
    validate_range(range, epsilon)?;
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); n as usize * n as usize];
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
        Ok(Array1::<f64>::from_vec(
            w.into_iter().take(m as usize).collect(),
        ))
    } else {
        Err(TbError::Lapack {
            routine: "zheevx",
            info,
        })
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
) -> Result<(Array1<f64>, Array2<Complex<f64>>)>
where
    S: Data<Elem = Complex<f64>>,
{
    let job1 = b'V'; // compute eigenvectors
    let job2 = b'V'; // eigenvalues in range
    let job3 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };
    let (n, mut a) = prepare_matrix(x)?;
    validate_range(range, epsilon)?;
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); n as usize * n as usize];
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
        Ok((
            Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect()),
            Array2::<Complex<f64>>::from_shape_vec(
                [m as usize, n as usize],
                z.into_iter().take(n as usize * m as usize).collect(),
            )?,
        ))
    } else {
        Err(TbError::Lapack {
            routine: "zheevr",
            info,
        })
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
) -> Result<Array1<f64>>
where
    S: Data<Elem = Complex<f64>>,
{
    let (n, mut a) = prepare_matrix(x)?;
    validate_range(range, epsilon)?;
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut w = vec![0.0; n as usize];
    let mut z = vec![Complex::new(0.0, 0.0); n as usize * n as usize];
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

    check_info("workspace query", info)?;
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
        Ok(Array1::<f64>::from_vec(
            w.into_iter().take(m as usize).collect(),
        ))
    } else {
        Err(TbError::Lapack {
            routine: "zheevr",
            info,
        })
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
/// # Errors
///
/// Returns an error for non-square inputs, dimensions exceeding LAPACK integer
/// limits, or a non-zero LAPACK info code.
pub fn eigvalsh_v<S>(x: &ArrayBase<S, Ix2>, uplo: UPLO) -> Result<Array1<f64>>
where
    S: Data<Elem = Complex<f64>>,
{
    let (n, mut a) = prepare_matrix(x)?;
    if n == 0 {
        return Ok(Array1::zeros(0));
    }
    let mut w = vec![0.0; n as usize];
    let _m = 0;
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
    check_info("workspace query", info)?;
    let lwork = work[0].re as i32;
    work = vec![Complex::new(0.0, 0.0); lwork as usize];

    unsafe {
        zheev(
            job1, job2, n, &mut a, n, &mut w, &mut work, lwork, &mut rwork, &mut info,
        );
    }
    if info == 0 {
        Ok(Array1::<f64>::from_vec(w))
    } else {
        Err(TbError::Lapack {
            routine: "zheev",
            info,
        })
    }
}

// Keep every i32 workspace expression below representable before entering FFI.
fn lapack_dimension(n: usize) -> Result<i32> {
    i32::try_from(n)
        .ok()
        .filter(|&n| n.checked_mul(33).is_some())
        .ok_or_else(|| {
            TbError::Other("matrix dimension exceeds LAPACK workspace integer limits".into())
        })
}

fn prepare_matrix<S: Data<Elem = Complex<f64>>>(
    x: &ArrayBase<S, Ix2>,
) -> Result<(i32, Vec<Complex<f64>>)> {
    if x.nrows() != x.ncols() {
        return Err(TbError::InvalidArrayShape {
            expected: vec![x.nrows(), x.nrows()],
            found: x.shape().to_vec(),
        });
    }
    let n = lapack_dimension(x.nrows())?;
    // LAPACK consumes columns. ndarray iterates in logical row order even for views.
    Ok((n, x.t().iter().copied().collect()))
}

fn validate_range(range: (f64, f64), epsilon: f64) -> Result<()> {
    if !range.0.is_finite() || !range.1.is_finite() || range.0 >= range.1 {
        return Err(TbError::Other(
            "eigenvalue range must have finite bounds with low < high".into(),
        ));
    }
    if !epsilon.is_finite() {
        return Err(TbError::Other("eigenvalue tolerance must be finite".into()));
    }
    Ok(())
}

fn check_info(routine: &'static str, info: i32) -> Result<()> {
    if info == 0 {
        Ok(())
    } else {
        Err(TbError::Lapack { routine, info })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ShapeBuilder, array, s};

    #[test]
    fn eigensolvers_reject_rectangular_inputs() {
        for shape in [(3, 2), (2, 3), (0, 2)] {
            let x = Array2::zeros(shape);
            assert!(eigh_x(&x, (-10.0, 10.0), 0.0, UPLO::Upper).is_err());
            assert!(eigh_r(&x, (-10.0, 10.0), 0.0, UPLO::Upper).is_err());
            assert!(eigvalsh_x(&x, (-10.0, 10.0), 0.0, UPLO::Upper).is_err());
            assert!(eigvalsh_r(&x, (-10.0, 10.0), 0.0, UPLO::Upper).is_err());
            assert!(eigvalsh_v(&x, UPLO::Upper).is_err());
        }
        assert!(lapack_dimension(i32::MAX as usize).is_err());
        assert!(lapack_dimension(usize::MAX).is_err());
    }

    #[test]
    fn eigenvectors_respect_input_triangle_and_layout() {
        let c = Complex::new;
        let h = array![
            [c(1.0, 0.0), c(0.2, 0.7), c(-0.3, 0.1)],
            [c(0.2, -0.7), c(2.0, 0.0), c(0.4, -0.5)],
            [c(-0.3, -0.1), c(0.4, 0.5), c(4.0, 0.0)],
        ];
        for uplo in [UPLO::Upper, UPLO::Lower] {
            let mut stored = h.clone();
            for i in 0..3 {
                for j in 0..3 {
                    if (matches!(uplo, UPLO::Upper) && i > j)
                        || (matches!(uplo, UPLO::Lower) && i < j)
                    {
                        stored[[i, j]] = c(f64::NAN, f64::NAN);
                    }
                }
            }
            let mut fortran = Array2::zeros((3, 3).f());
            fortran.assign(&stored);
            let mut padded = Array2::zeros((6, 6));
            padded.slice_mut(s![..;2, ..;2]).assign(&stored);
            for x in [stored.view(), fortran.view(), padded.slice(s![..;2, ..;2])] {
                let reference = eigvalsh_v(&x, uplo).unwrap();
                for (energies, vectors) in [
                    eigh_x(&x, (-10.0, 10.0), 0.0, uplo).unwrap(),
                    eigh_r(&x, (-10.0, 10.0), 0.0, uplo).unwrap(),
                ] {
                    assert_eq!(vectors.dim(), (3, 3));
                    for n in 0..3 {
                        assert!((energies[n] - reference[n]).abs() < 1e-12);
                        let ket = vectors.row(n);
                        let residual = h.dot(&ket) - &ket * energies[n];
                        assert!(residual.iter().all(|z| z.norm() < 1e-12));
                        assert!(
                            (ket.iter().map(|z| z.norm_sqr()).sum::<f64>() - 1.0).abs() < 1e-12
                        );
                    }
                }
                for energies in [
                    eigvalsh_x(&x, (-10.0, 10.0), 0.0, uplo).unwrap(),
                    eigvalsh_r(&x, (-10.0, 10.0), 0.0, uplo).unwrap(),
                ] {
                    assert!(
                        energies
                            .iter()
                            .zip(&reference)
                            .all(|(a, b)| (a - b).abs() < 1e-12)
                    );
                }
            }
        }
    }

    #[test]
    fn eigensolvers_handle_empty_scalar_and_invalid_ranges() {
        let empty = Array2::<Complex<f64>>::zeros((0, 0));
        for uplo in [UPLO::Upper, UPLO::Lower] {
            assert_eq!(
                eigh_x(&empty, (-1.0, 1.0), 0.0, uplo).unwrap().1.dim(),
                (0, 0)
            );
            assert_eq!(
                eigh_r(&empty, (-1.0, 1.0), 0.0, uplo).unwrap().1.dim(),
                (0, 0)
            );
            assert!(
                eigvalsh_x(&empty, (-1.0, 1.0), 0.0, uplo)
                    .unwrap()
                    .is_empty()
            );
            assert!(
                eigvalsh_r(&empty, (-1.0, 1.0), 0.0, uplo)
                    .unwrap()
                    .is_empty()
            );
            assert!(eigvalsh_v(&empty, uplo).unwrap().is_empty());
            let scalar = array![[Complex::new(2.0, 0.0)]];
            assert_eq!(eigvalsh_v(&scalar, uplo).unwrap(), array![2.0]);
            for solve in [eigh_x, eigh_r] {
                assert_eq!(
                    solve(&scalar, (1.0, 2.0), 0.0, uplo).unwrap().0,
                    array![2.0]
                );
                assert_eq!(
                    solve(&scalar, (2.0, 3.0), 0.0, uplo).unwrap().1.dim(),
                    (0, 1)
                );
                assert!(solve(&scalar, (2.0, 1.0), 0.0, uplo).is_err());
                assert!(solve(&scalar, (1.0, 3.0), f64::NAN, uplo).is_err());
            }
        }
    }
}
