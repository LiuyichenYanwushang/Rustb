//! Small mathematical helpers shared by the numerical modules.
//!
//! [`comm`] and [`anti_comm`] are matrix commutators and anti-commutators for
//! any ndarray element type implementing `LinalgScalar`; [`gauss`] is the
//! normalized Gaussian kernel used for density-of-states-style broadening (the
//! `dos` implementation inlines the same kernel and does not call it).
//!
//! # Examples
//!
//! ```rust
//! use ndarray::arr2;
//! use Rustb::*;
//!
//! let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
//! let b = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
//! let commutator = comm(&a, &b);
//! let anticommutator = anti_comm(&a, &b);
//! assert_eq!(commutator + anticommutator, a.dot(&b) * 2.0);
//! assert!(gauss(0.0, 0.1) > gauss(0.1, 0.1));
//! ```

use ndarray::{Array2, ArrayBase, Data, Ix2, LinalgScalar};
use std::f64::consts::PI;
pub fn gauss(x: f64, eta: f64) -> f64 {
    //高斯函数
    let a = x / eta;
    let g = (-a * a / 2.0).exp();
    let g = 1.0 / (2.0 * PI).sqrt() / eta * g;
    g
}

#[allow(non_snake_case)]
#[inline(always)]
pub fn comm<A, B, T>(A: &ArrayBase<A, Ix2>, B: &ArrayBase<B, Ix2>) -> Array2<T>
where
    A: Data<Elem = T>,
    B: Data<Elem = T>,
    T: LinalgScalar, // 约束条件：T 必须实现 LinalgScalar trait
{
    //! Compute the commutator $[A, B] = AB - BA$.
    A.dot(B) - B.dot(A)
}
#[allow(non_snake_case)]
#[inline(always)]
pub fn anti_comm<A, B, T>(A: &ArrayBase<A, Ix2>, B: &ArrayBase<B, Ix2>) -> Array2<T>
where
    A: Data<Elem = T>,
    B: Data<Elem = T>,
    T: LinalgScalar, // 约束条件：T 必须实现 LinalgScalar trait
{
    //! Compute the anti-commutator $\lbrace A, B\rbrace = AB + BA$.
    A.dot(B) + B.dot(A)
}
