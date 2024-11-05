//!这个模块是用来求解大矩阵的部分本征值的模块, 用的lapackc的 cheevx 等函数求解.
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

pub fn eigh_x<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> (Array1<f64>, Array2<Complex<f64>>)
where
    S: Data<Elem = Complex<f64>>,
{
    // 获取矩阵的阶数
    let n = x.shape()[0] as i32;
    // 创建一个可变的副本，用于存储特征向量或被销毁
    let mut a: Vec<_> = x.iter().cloned().collect();
    // 创建一个可变的向量，用于存储特征值
    let mut w = vec![0.0; n as usize];
    // 创建一个可变的向量，用于存储特征向量
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    // 创建一个可变的变量，用于存储特征值的个数
    let mut m = 0;
    // 创建一个可变的变量，用于存储函数的返回状态
    let mut info = 0;
    // 创建一个可变的向量，用于存储失败的特征值的索引
    let mut ifail = vec![0; n as usize];
    // 创建一个可变的向量，用于作为工作空间
    let mut work = vec![Complex::new(0.0, 0.0); (2 * n) as usize];
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; (7 * n) as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let mut iwork = vec![0; (5 * n) as usize];
    // 调用 cheevx 函数，使用 'N' 表示不计算特征向量，使用 'V' 表示计算范围内的特征值，使用 'U' 表示输入的矩阵是上三角
    // Assuming range is a tuple like (f64, f64)
    let job1 = b'V';
    let job2 = b'V';
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
    // 检查函数的返回状态，如果是 0，表示成功，否则表示有问题
    if info == 0 {
        // 将特征值向量转换为一维数组并返回
        (
            Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect()),
            Array2::<Complex<f64>>::from_shape_vec(
                [m as usize, n as usize],
                z.into_iter().take((n * m) as usize).collect(),
            )
            .unwrap(),
        )
    } else {
        // 报告错误信息
        panic!("cheevx failed with info = {}", info);
    }
}

pub fn eigvalsh_x<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> Array1<f64>
where
    S: Data<Elem = Complex<f64>>,
{
    // 获取矩阵的阶数
    let n = x.shape()[0] as i32;
    // 创建一个可变的副本，用于存储特征向量或被销毁
    let mut a: Vec<_> = x.iter().cloned().collect();
    // 创建一个可变的向量，用于存储特征值
    let mut w = vec![0.0; n as usize];
    // 创建一个可变的向量，用于存储特征向量
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    // 创建一个可变的变量，用于存储特征值的个数
    let mut m = 0;
    // 创建一个可变的变量，用于存储函数的返回状态
    let mut info = 0;
    // 创建一个可变的向量，用于存储失败的特征值的索引
    let mut ifail = vec![0; n as usize];
    // 创建一个可变的向量，用于存储失败的特征值的索引
    let mut work = vec![Complex::new(0.0, 0.0); (2 * n) as usize];
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; (7 * n) as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let mut iwork = vec![0; (5 * n) as usize];
    // 调用 cheevx 函数，使用 'N' 表示不计算特征向量，使用 'V' 表示计算范围内的特征值，使用 'U' 表示输入的矩阵是上三角
    // Assuming range is a tuple like (f64, f64)
    let job1 = b'N';
    let job2 = b'V';
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
    // 检查函数的返回状态，如果是 0，表示成功，否则表示有问题
    if info == 0 {
        // 将特征值向量转换为一维数组并返回
        Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect())
    } else {
        // 报告错误信息
        panic!("cheevx failed with info = {}", info);
    }
}

pub fn eigh_r<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> (Array1<f64>, Array2<Complex<f64>>)
where
    S: Data<Elem = Complex<f64>>,
{
    let job1 = b'V';
    let job2 = b'V';
    let job3 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };
    // 获取矩阵的阶数
    let n = x.shape()[0] as i32;
    // 创建一个可变的副本，用于存储特征向量或被销毁
    let mut a: Vec<_> = x.iter().cloned().collect();
    // 创建一个可变的向量，用于存储特征值
    let mut w = vec![0.0; n as usize];
    // 创建一个可变的向量，用于存储特征向量
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut isuppz = vec![0; 2 * n as usize];
    // 创建一个可变的变量，用于存储特征值的个数
    let mut m = 0;
    // 创建一个可变的变量，用于存储函数的返回状态
    let mut info = 0;
    /*
    // 创建一个可变的向量，用于作为工作空间
    let mut work = vec![Complex::new(0.0,0.0); 1 as usize];
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; 1 as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let mut iwork = vec![0; 1 as usize];
    // 调用 cheevx 函数，使用 'N' 表示不计算特征向量，使用 'V' 表示计算范围内的特征值，使用 'U' 表示输入的矩阵是上三角
    // Assuming range is a tuple like (f64, f64)

    unsafe {
         zheevr(
            job1,
            job2,
            job3,
            n ,
            &mut a,
            n ,
            range.0,
            range.1,
            0 ,
            n ,
            epsilon ,
            &mut m,
            &mut w,
            &mut z,
            n ,
            &mut isuppz,
            &mut work,
            -1 ,
            &mut rwork,
            -1,
            &mut iwork,
            -1,
            &mut info,
        );
    }

    let lwork=work[0].re as i32;
    let liwork=iwork[0] as i32;
    let lrwork=rwork[0] as i32;
    */
    let lwork = n * 33 as i32;
    let liwork = n * 10 as i32;
    let lrwork = n * 24 as i32;
    let mut work = vec![Complex::new(0.0, 0.0); lwork as usize];
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; lrwork as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let mut iwork = vec![0; liwork as usize];
    // 调用 cheevx 函数，使用 'N' 表示不计算特征向量，使用 'V' 表示计算范围内的特征值，使用 'U' 表示输入的矩阵是上三角

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

    // 检查函数的返回状态，如果是 0，表示成功，否则表示有问题
    if info == 0 {
        // 将特征值向量转换为一维数组并返回
        (
            Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect()),
            Array2::<Complex<f64>>::from_shape_vec(
                [m as usize, n as usize],
                z.into_iter().take((n * m) as usize).collect(),
            )
            .unwrap(),
        )
    } else {
        // 报告错误信息
        panic!("cheevx failed with info = {}", info);
    }
}

pub fn eigvalsh_r<S>(
    x: &ArrayBase<S, Ix2>,
    range: (f64, f64),
    epsilon: f64,
    uplo: UPLO,
) -> Array1<f64>
where
    S: Data<Elem = Complex<f64>>,
{
    // 获取矩阵的阶数
    let n = x.shape()[0] as i32;
    // 创建一个可变的副本，用于存储特征向量或被销毁
    let mut a: Vec<_> = x.iter().cloned().collect();
    // 创建一个可变的向量，用于存储特征值
    let mut w = vec![0.0; n as usize];
    // 创建一个可变的向量，用于存储特征向量
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut isuppz = vec![0; 2 * n as usize];
    // 创建一个可变的变量，用于存储特征值的个数
    let mut m = 0;
    // 创建一个可变的变量，用于存储函数的返回状态
    let mut info = 0;
    // 创建一个可变的向量，用于作为工作空间
    let mut work = vec![Complex::new(0.0, 0.0); 1 as usize];
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; 1 as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let mut iwork = vec![0; 1 as usize];
    // 调用 cheevx 函数，使用 'N' 表示不计算特征向量，使用 'V' 表示计算范围内的特征值，使用 'U' 表示输入的矩阵是上三角
    // Assuming range is a tuple like (f64, f64)
    let job1 = b'N';
    let job2 = b'V';
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
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; lrwork as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let mut iwork = vec![0; liwork as usize];
    // 调用 cheevx 函数，使用 'N' 表示不计算特征向量，使用 'V' 表示计算范围内的特征值，使用 'U' 表示输入的矩阵是上三角

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
    // 检查函数的返回状态，如果是 0，表示成功，否则表示有问题
    if info == 0 {
        // 将特征值向量转换为一维数组并返回
        Array1::<f64>::from_vec(w.into_iter().take(m as usize).collect())
    } else {
        // 报告错误信息
        panic!("cheevx failed with info = {}", info);
    }
}

pub fn eigvalsh_v<S>(x: &ArrayBase<S, Ix2>, uplo: UPLO) -> Array1<f64>
where
    S: Data<Elem = Complex<f64>>,
{
    // 获取矩阵的阶数
    let n = x.shape()[0] as i32;
    // 创建一个可变的副本，用于存储特征向量或被销毁
    let mut a: Vec<_> = x.iter().cloned().collect();
    // 创建一个可变的向量，用于存储特征值
    let mut w = vec![0.0; n as usize];
    // 创建一个可变的向量，用于存储特征向量
    let mut z = vec![Complex::new(0.0, 0.0); (n * n) as usize];
    let mut isuppz = vec![0; 2 * n as usize];
    // 创建一个可变的变量，用于存储特征值的个数
    let mut m = 0;
    // 创建一个可变的变量，用于存储函数的返回状态
    let mut info = 0;
    // 创建一个可变的向量，用于作为工作空间
    let mut work = vec![Complex::new(0.0, 0.0); 1 as usize];
    // 创建一个可变的向量，用于作为实数工作空间
    let mut rwork = vec![0.0; (3 * n - 2) as usize];
    // 创建一个可变的向量，用于作为整数工作空间
    let job1 = b'N';
    let job2 = match uplo {
        UPLO::Upper => b'U',
        UPLO::Lower => b'L',
    };

    unsafe {
        zheev(
            job1, job2, n, &mut a, n, &mut w, &mut work, -1, &mut rwork, &mut info,
        );
    }
    // 获取推荐的工作空间大小
    let lwork = work[0].re as i32;
    // 重新分配工作空间
    work = vec![Complex::new(0.0, 0.0); lwork as usize];

    // 第二次调用 zheev，实际计算特征值和特征向量
    unsafe {
        zheev(
            job1, job2, n, &mut a, n, &mut w, &mut work, lwork, &mut rwork, &mut info,
        );
    }
    // 检查函数的返回状态，如果是 0，表示成功，否则表示有问题
    if info == 0 {
        // 将特征值向量转换为一维数组并返回
        Array1::<f64>::from_vec(w.into_iter().collect())
    } else {
        // 报告错误信息
        panic!("zheev failed with info = {}", info);
    }
}
