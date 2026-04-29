use ndarray::{Array2, ArrayBase, Data, Ix2, LinalgScalar};
use std::f64::consts::PI;
pub fn gauss(x: f64, eta: f64) -> f64 {
    //高斯函数
    let a = (x / eta);
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
    //! Compute the anti-commutator $\{A, B\} = AB + BA$.
    A.dot(B) + B.dot(A)
}
