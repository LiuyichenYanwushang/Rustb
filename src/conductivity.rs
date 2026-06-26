//! Calculation of linear and nonlinear conductivity tensors using Kubo formalism.
//!
//! This module implements various conductivity calculations including:
//! - Anomalous Hall conductivity
//! - Spin Hall conductivity
//! - Nonlinear Hall conductivity
//! - Berry curvature and orbital magnetization
//!
//! The implementations are based on the Kubo formula and semiclassical wave-packet
//! dynamics, providing both intrinsic and extrinsic contributions to transport.

//! ## Derivation of nonlinear Hall effect using Niu-Qian equations
//!
//! This section derives formulas for linear and nonlinear Hall conductivities using the Niu-Qian formalism.
//! Starting from the current density formula:
//! $$\bm J=-e\int_\tx{BZ}\dd\bm k\sum_n f_n\bm v_n$$
//! Here $n$ labels bands, $f_n$ is the Fermi-Dirac distribution. The velocity operator according to Niu-Qian is:
//! $$\bm v=\f{1}{\hbar}\f{\p\ve_n}{\p\bm k}-\f{e}{\hbar}\bm E\times\bm\Og_n$$
//! The $n$-th order Hall conductivity is defined as:
//! $$\sg_{\ap_1,\ap_2,\cdots,\ap_n;d}=\f{1}{n!}\left\.\f{\p^n J_d}{\p E_{\ap_1}\cdots\p E_{\ap_n}}\right\vert_{\bm E=0}$$
//! To obtain its expression, we define series expansions:
//! $$\lt\\\{\\begin{aligned}
//! f_n=f_n^{(0)}+f_n^{(1)}+f_n^{(2)}\cdots\\\\
//! \bm v_n=\bm v_n^{(0)}+\bm v_n^{(1)}+\bm v_n^{(2)}\cdots\\\\
//! \\end{aligned}\rt\.$$
//! This gives:
//! $$ \\begin{aligned}\bm J^{(0)}&=-e\int_\tx{BZ}\dd\bm k\sum_n f_n^{(0)}\bm v_n^{(0)}\\\\
//! \bm J^{(1)}&=-e\int_\tx{BZ}\dd\bm k\sum_n f_n^{(1)}\bm v_n^{(0)}+f_n^{(0)}\bm v_n^{(1)}\\\\
//! \bm J^{(2)}&=-e\int_\tx{BZ}\dd\bm k\sum_n f_n^{(2)}\bm v_n^{(0)}+f_n^{(1)}\bm v_n^{(1)}+f_n^{(0)}\bm v_n^{(2)}\\\\
//! \\end{aligned}$$
//!
//! Now consider the corrections to $f$. Using the Boltzmann equation:
//! $$\p_t f-\f{e}{\hbar}\bm E\cdot\nb_{\bm k} f=-\f{f-f_0}{\tau}$$
//! Setting $f=\sum_{s=1}e^{is\og t} f_n^{(s)}$, we have:
//! $$\\begin{aligned} is\og\sum_{s=1}f_n^{(s)}-\f{e}{\hbar}\bm E\cdot\nb_{\bm k}\sum_{s=0} f_n^{(s)}=-\f{1}{\tau}\sum_{s=1} f_n^{(s)}\\\\
//! \Rightarrow (is\og+\f{1}{\tau})\sum_{s=1} f_n^{(i)}-\f{e}{\hbar}\bm E\cdot\nb_{\bm k}\sum_{i=0} f_n^{(i)}=0\\\\
//! \\end{aligned}$$
//! Finally, we obtain the higher-order Fermi distribution:
//! $$f_n^{(l)}=\f{e}{\hbar} \f{\bm E\nb_{\bm k} f_n^{(l-1)}}{i l \og+1/\tau}=\lt(\f{e/\hbar}{i\og+1/\tau}\rt)\bm E^l\nb^l_{\bm k} f_n^{(0)}$$
//! Taking the zero-frequency limit: $$\lim_{\og\to 0} f_n^{(l)}\approx \lt(\f{e\tau}{\hbar}\rt)^l \bm E^l\nb^l_{\bm k} f_n^{(0)}$$
//!
//! For the Fermi velocity $\bm v_n=\f{1}{\hbar}\pdv{\ve_n}{\bm k}+\f{e}{\hbar}\bm E\times\bm \Og_n$,
//! we define order-by-order expansions:
//! $$\\begin{aligned}
//! \bm v_n^{(0)}&=\f{1}{\hbar}\pdv{\ve_n^{(0)}}{\bm k}\\\\
//! \bm v_n^{(1)}&=\f{1}{\hbar}\pdv{\ve_n^{(1)}}{\bm k}+\f{e}{\hbar}\bm E\times\bm \Og_n^{(0)}\\\\
//! \bm v_n^{(2)}&=\f{1}{\hbar}\pdv{\ve_n^{(2)}}{\bm k}+\f{e}{\hbar}\bm E\times\bm \Og_n^{(1)}\\\\
//! \\end{aligned}$$
//! Next, starting from the Hamiltonian under an electric field:
//! $$H_{\bm k}=\sum_{mn}\lt(\ve_n^{(0)}\dt_{nm}-e\bm E\cdot\bra{\psi_n}\bm r\ket{\psi_n}\rt)\ket{\psi_n}\bra{\psi_m}$$
//! We split it into two parts: the diagonal part and the off-diagonal part:
//! $$\\begin{aligned}
//! H_{\bm k}^{(0)}&=\sum_{n}\lt(\ve_{n\bm k}^{(0)}-e\bm E\cdot\bm A_n\rt)\dyad{\psi_n}\\\\
//! H_{\bm k}^{(1)}&=\sum_{n=\not m}\lt(-e\bm E\cdot\bm A_{mn}\rt)\ket{\psi_m}\bra{\psi_n}\\\\
//! \\end{aligned}$$
//! where $\bm A_{mn}=\bra{\psi_m}\bm r\ket{\psi_n}=i\bra{\psi_m}\p_{\bm k}\ket{\psi_n}$.
//!
//! Clearly, we have the formula:
//! $$e^{\hat S}\hat{\mathcal{O}}e^{-\hat S}=\mathcal{O}+\lt[\hat S,\hat{\mcl{O}}\rt]+\f{1}{2}\lt[\hat S,\lt[\hat S,\hat{\mcl{O}}\rt]\rt]+\f{1}{6}\lt[\hat S,\lt[\hat S,\lt[\hat S,\hat{\mcl{O}}\rt]\rt]\rt]\cdots$$
//! For computational convenience, we choose $\hat S$ such that $H_{\bm k}^{(1)}+\lt[\hat S,\hat H_{\bm k}^{(0)}\rt]=0$, giving:
//! $$\\begin{aligned}
//! H^\prime_{\bm k}&=e^{\hat S}H_{\bm k} e^{-\hat S}=H_{\bm k}^{(0)}+\lt(H_{\bm k}^{(1)}+\lt[\hat S,\hat H_{\bm k}^{(0)}\rt]\rt)+\lt(\lt[\hat S,\hat H_{\bm k}^{(1)}\rt]+\f{1}{2}\lt[\hat S,\lt[\hat S,\hat H_{\bm k}^{(0)}\rt]\rt]\rt)\cdots\\\\
//! &=H_{\bm k}^{(0)}+\f{1}{2}\lt[S,H_{\bm k}^{(1)}\rt]+\f{1}{3}\lt[S,\lt[S,H_{\bm k}^{(1)}\rt]\rt]\cdots
//! \\end{aligned}$$
//! To satisfy the condition, we choose:
//! $$S_{nn}=0,\ S_{nm}=\f{-e\bm E\cdot \bm A_{nm}}{\ve_{nm}-e\bm E\cdot \bm A_{nm}}$$
//!
//! Because we have:
//! $$\\begin{aligned} \lt[S,H_{\bm k}^{(0)}\rt]&=SH_{\bm k}^{(0)}-H_{\bm k}^{(0)}S=\sum_{j=\not m} S_{mj}H_{\bm k,jn}^{(0)}-\sum_{j=\not n }H_{\bm k,mj}^{(0)}S_{jn}\\\\
//! &=\sum_{j=\not m}\f{-e\bm E\cdot \bm A_{mj}\lt(\ve_j^{(0)}-e\bm E\cdot\bm A_j\rt)\dt_{jn}}{\ve_{mj}-e\bm E\cdot\lt(\bm A_m-\bm A_j\rt)}-\sum_{j=\not n}\f{-e\lt(\ve_j^{(0)}-e\bm E\cdot\bm A_j\rt)\lt(\bm E\cdot \bm A_{jn}\rt)\dt_{mj}}{\ve_{jn}-e\bm E\cdot\lt(\bm A_j-\bm A_n\rt)}\\\\
//! &=\f{e\lt(\bm E\cdot\bm A_{mn}\rt)\lt[\ve_{mn}- e\bm E\cdot\lt(\bm A_m-\bm A_n\rt)\rt]}{\ve_{mn}-e\bm E\cdot(\bm A_m-\bm A_n)}=-H_{\bm k}^{(1)}
//! \\end{aligned}$$
//! This verifies our result. We simplify and expand $\hat S$ to obtain:
//! $$S_{nm}\approx \f{-e\bm E\cdot\bm A_{nm}}{\ve_n^{(0)}-\ve_m^{(0)}}-\f{ e^2\lt(\bm E\cdot\bm A_{nm}\rt)\lt(\bm E\cdot\lt(\bm A_n-\bm A_m\rt)\rt)}{\lt(\ve_n^{(0)}-\ve_m^{(0)}\rt)^2}$$
//! Thus we obtain the band perturbations at each order:
//! $$\\begin{aligned}
//! \ve_n^{(1)}&=-e\bm E\cdot\bm A_n\\\\
//! \ve_n^{(2)}&=\f{e^2}{2}E_a E_b \sum_{m=\not n}\f{A_{nm}^a A_{mn}^b+A_{mn}^a A_{nm}^b}{\ve_n-\ve_m}=e^2 G_n^{ab}E_a E_b\\\\
//! \ve_n^{(3)}&=-e^3E_a E_b E_c \lt( \sum_{m=\not n}\sum_{l=\not m,n}\f{A_{nl}^a A_{lm}^b A_{mn}^c}{(\ve_n-\ve_m)(\ve_n-\ve_l)}\rt)+e^3 E_a E_b E_c\lt( \sum_{m=\not n}\sum_{l=\not m,n}\f{A_{nm}^a A_{mn}^b (A_n^c-A_m^c)}{(\ve_n-\ve_m)^2}\rt)\\\\
//! \\end{aligned}$$
//! where $$G_n^{ab}=\sum_{m=\not n}\f{A_{nm}^a A_{mn}^b+A_{mn}^a A_{nm}^b}{\ve_n-\ve_m}=\sum_{m=\not n} 2\tx{Re}\f{v_{nm}^a v_{mn}^b}{(\ve_n-\ve_m)^3}$$
//! At this point, we have obtained the band perturbations. However, there is a problem: $\bm A$ is a gauge-dependent quantity and is therefore not unique.
//! Meanwhile, for the intra-band contribution $\bm A_{n}=i\bra{\psi_{n\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}$, there is no simple way to compute it because $\bm A=-e\bra{\psi_n}\bm r\ket{\psi_n}$ breaks translational symmetry. But we can always choose a gauge.
//! Here we choose $-e\bm E\cdot\bm A_n=0$, a gauge that makes $\ve_n^{(1)}=0$. Physically, this gauge means the Berry connection is perpendicular to the direction of the electric field. For the higher-order term of the Berry curvature, using
//! $\bm A\to\bm A^\prime=A+\lt[\hat S,\bm A\rt]+\f{1}{2}\lt[\hat S,\lt[\hat S,\bm A\rt]\rt]\cdots$, we have:
//! $$\\begin{aligned}
//! \lt(A_n^b\rt)^{(1)}&=-e\bm E_a G_n^{ab}\\\\
//! \lt(A_n^c\rt)^{(2)}&=e^2 E_a E_b \lt( \sum_{m=\not n}\sum_{l=\not m,n}\f{A_{nl}^a A_{lm}^b A_{mn}^c}{(\ve_n-\ve_m)(\ve_n-\ve_l)}\rt)+e^2 E_a E_b\lt( \sum_{m=\not n}\f{A_{nm}^a A_{mn}^b (A_n^c-A_m^c)}{(\ve_n-\ve_m)^2}\rt)\\\\
//! &=e^2 E_a E_b\lt(S_n^{abc}-F_n^{abc}\rt)
//! \\end{aligned}$$
//!
//! Next, using the Berry curvature formula $\Og_n^{ab}=\p_a A_n^b -\p_b A_n^a$, we obtain:
//! $$\\begin{aligned}
//! \lt(\Og_n^{ab}\rt)^{(1)}&=-e E_c\lt(\p_a G_n^{bc}-\p_b G_n^{ac}\rt)\\\\
//! \lt(\Og_n^{ab}\rt)^{(2)}&=e^2 E_{\ap}E_{\bt}\lt(\p_a S^{\ap\bt b}-\p_b S^{\ap\bt a}-\p_a F^{\ap\bt b}+\p_b F^{\ap\bt a}\rt)
//! \\end{aligned}$$
//! Finally, substituting into the conductivity formula, we obtain:
//! $$\begin{aligned}
//! \sigma_{ab}=&-\f{e^2}{\hbar}\int_\tx{BZ} \f{\dd\bm k}{(2\pi)^3}\sum_n f_n\Og_n^{ab}+\f{e^2\tau}{\hbar^2}\sum_n \int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3}\f{\p^2\ve_n}{\p k_a\p k_b}\\\\
//! \sigma_{abc}=&-\f{e^3\tau^2}{\hbar^3}\sum_n\int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3}\f{\p^3\ve_n}{\p k_a \p k_b \p k_c}
//! +\f{e^3\tau}{\hbar^2}\sum_n \int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3} \f{1}{2} f_n \lt(\p_a\Og_n^{bc}+\p_b\Og_n^{ac}\rt)\\\\
//! &-\f{e^3}{\hbar}\sum_n\int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3} f_n\lt(2\p_c G_n^{ab}-\f{1}{2}\lt(\p_a G_n^{bc}+\p_b G_n^{ac}\rt)\rt)
//! \end{aligned}$$
//!
//! ## Simplification of the Berry connection
//!
//! For practical computations, we need to modify the Berry connection form. First, by the chain rule, we have:
//! $$\p_{\bm k}\lt(H_{\bm k}\ket{\psi_{n\bm k}}\rt)=\lt(\p_{\bm k}H_{\bm k}+H_{\bm k}\p_{\bm k}\rt)\ket{\psi_{n\bm k}}$$
//! Since $H_{\bm k}\ket{\psi_{n\bm k}}=\ve_{n\bm k}\ket{\psi_{n\bm k}}$, we also have:
//! $$\p_{\bm k}\lt(H_{\bm k}\ket{\psi_{n\bm k}}\rt)=\p_{\bm k}\ve_{n\bm k}\ket{\psi_{n\bm k}}+\ve_{n\bm k}\p_{\bm k}\ket{\psi_{n\bm k}}$$
//! Therefore:
//! $$\\begin{aligned}
//! \p_{\bm k}H_{\bm k}\ket{\psi_{n\bm k}}+H_{\bm k}\p_{\bm k}\ket{\psi_{n\bm k}}=\p_{\bm k}\ve_{n\bm k}\ket{\psi_{n\bm k}}+\ve_{n\bm k}\p_{\bm k}\ket{\psi_{n\bm k}}
//! \\end{aligned}$$
//! Inserting a complete set of states $\sum_m \dyad{\psi_{m\bm k}}$ on the left of both sides yields:
//! $$\sum_m\lt[\bra{\psi_{m\bm k}}\p_{\bm k}H_{\bm k}\ket{\psi_{n\bm k}}+\lt(\ve_{m\bm k}-\ve_{n\bm k}\rt)\bra{\psi_{m\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}\rt]\ket{\psi_{m\bm k}}=\bra{\psi_{n\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}\ket{\psi_{n\bm k}}$$
//! From the above, we easily see that when $m\neq n$:
//! $$\bra{\psi_{m\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}=\f{\bra{\psi_{m\bm k}}\p_{\bm k}H_{\bm k}\ket{\psi_{n\bm k}}}{\ve_{n\bm k}-\ve_{m\bm k}}$$
//! That is, we obtain the final expression:
//! $$\bm A_{mn}=i\f{\bra{\psi_{m\bm k}}\p_{\bm k}H_{\bm k}\ket{\psi_{n\bm k}}}{\ve_{n\bm k}-\ve_{m\bm k}}$$

use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::kpoints::{gen_kmesh, gen_krange};
use crate::math::*;
use crate::phy_const::mu_B;
use crate::solve_ham::solve;
use crate::velocity::*;
use crate::{Gauge, Model, SpinDirection};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::AddAssign;
use std::ops::MulAssign;

/// Directly construct spin Pauli matrix σ⊗I_{norb}/2 without kron.
/// Only sets 2*norb non-zero elements (O(norb)) instead of O(nsta²).
#[inline]
fn build_spin_matrix(norb: usize, spin: Option<SpinDirection>) -> Array2<Complex<f64>> {
    let nsta = 2 * norb;
    let mut m = Array2::<Complex<f64>>::zeros((nsta, nsta));
    let half = Complex::new(0.5, 0.0);
    let i_half = Complex::new(0.0, 0.5);
    match spin {
        None => {
            m = Array2::<Complex<f64>>::eye(2 * norb);
        }
        Some(SpinDirection::X) => {
            // σ_x ⊗ I: [0 I; I 0] / 2
            for i in 0..norb {
                m[[i, i + norb]] = half;
                m[[i + norb, i]] = half;
            }
        }
        Some(SpinDirection::Y) => {
            // σ_y ⊗ I: [0 -iI; iI 0] / 2
            for i in 0..norb {
                m[[i, i + norb]] = -i_half;
                m[[i + norb, i]] = i_half;
            }
        }
        Some(SpinDirection::Z) => {
            // σ_z ⊗ I: [I 0; 0 -I] / 2
            for i in 0..norb {
                m[[i, i]] = half;
                m[[i + norb, i + norb]] = -half;
            }
        }
    }
    m
}

/// Adaptive integration algorithm over an n-dimensional hyperrectangle.
///
/// For an integral in $n$ dimensions, the integration domain is partitioned into $(n+1)$-simplices,
/// and linear interpolation is used to approximate the integral over each simplex.
///
/// Given the integrand $f(x_1,x_2,...,x_n)$, let there be $n+1$ vertices
/// $(y_{01},y_{02},\cdots y_{0n})\cdots(y_{n1},y_{n2}\cdots y_{nn})$ with corresponding function
/// values $z_0,z_1,...,z_n$. The approximate integral over one simplex is:
/// $$ \f{1}{(n+1)!}\times\sum_{i=0}^n z_i \cdot \dd V,$$
/// where $\dd V$ is the volume of the $(n+1)$-simplex.
///
/// **In 1D:** linear interpolation is equivalent to the trapezoidal rule. Between two adjacent data
/// points $(x_1, f_1)$ and $(x_2, f_2)$, the integral is $\Delta = \f{f_1+f_2}{2}\cdot(x_2-x_1)$.
///
/// **In 2D:** triangular elements are used. For any small triangle, the integral is
/// $\Delta = S\sum_{i=1}^3 f_i / 3!$.
///
/// **In 3D:** tetrahedral elements are used. The linear interpolation result is
/// $\Delta = V\sum_{i=1}^4 f_i / 4!$.
///
/// The algorithm recursively subdivides simplices until the error estimate satisfies the tolerance.
///
/// # Arguments
///
/// * `f0` - The integrand function $f(\mathbf{k})$, taking an `Array1<f64>` and returning `f64`.
/// * `k_range` - A `(dim, 2)` array specifying the integration domain. Each row `[k_min, k_max]`
///   defines the range along one dimension.
/// * `re_err` - Relative error tolerance.
/// * `ab_err` - Absolute error tolerance.
///
/// # Returns
///
/// The approximate value of the integral.
///
/// # Panics
///
/// Panics if `k_range.len_of(Axis(0))` (the dimension) is not 1, 2, or 3.
///
/// # Examples
///
/// ```ignore
/// use ndarray::{arr1, arr2};
/// use rustb::conductivity::adapted_integrate_quick;
///
/// // Integrate sin(k1 + k2) over [0, pi] x [0, pi]
/// let f = |k: &ndarray::Array1<f64>| (k[0] + k[1]).sin();
/// let k_range = arr2(&[[0.0, std::f64::consts::PI], [0.0, std::f64::consts::PI]]);
/// let result = adapted_integrate_quick(&f, &k_range, 1.0, 1e-4);
/// assert!((result - 4.0).abs() < 1e-3);
/// ```

#[inline(always)]
pub fn adapted_integrate_quick(
    f0: &dyn Fn(&Array1<f64>) -> f64,
    k_range: &Array2<f64>,
    re_err: f64,
    ab_err: f64,
) -> f64 {
    let dim = k_range.len_of(Axis(0));
    match dim {
        1 => {
            //对于一维情况, 我们就是用梯形算法的 (a+b)*h/2, 这里假设的是函数的插值为线性插值.
            let mut use_range = vec![(k_range.clone(), re_err, ab_err)];
            let mut result = 0.0;
            while let Some((k_range, re_err, ab_err)) = use_range.pop() {
                let kvec_l: Array1<f64> = arr1(&[k_range[[0, 0]]]);
                let kvec_r: Array1<f64> = arr1(&[k_range[[0, 1]]]);
                let kvec_m: Array1<f64> = arr1(&[(k_range[[0, 1]] + k_range[[0, 0]]) / 2.0]);
                let dk: f64 = k_range[[0, 1]] - k_range[[0, 0]];
                let y_l: f64 = f0(&kvec_l);
                let y_r: f64 = f0(&kvec_r);
                let y_m: f64 = f0(&kvec_m);
                let all: f64 = (y_l + y_r) * dk / 2.0;
                let all_1 = (y_l + y_m) * dk / 4.0;
                let all_2 = (y_r + y_m) * dk / 4.0;
                let err = all_1 + all_2 - all;
                let abs_err = if ab_err > all * re_err {
                    ab_err
                } else {
                    all * re_err
                };
                if err < abs_err {
                    result += all_1 + all_2;
                } else {
                    let k_range_l = arr2(&[[kvec_l[[0]], kvec_m[[0]]]]);
                    let k_range_r = arr2(&[[kvec_m[[0]], kvec_r[[0]]]]);
                    use_range.push((k_range_l, re_err, ab_err / 2.0));
                    use_range.push((k_range_r, re_err, ab_err / 2.0));
                }
            }
            return result;
        }
        2 => {
            //对于二维, 我们依旧假设线性插值, 这样我们考虑的就是二维平面上的三角形上的任意一点的值是到其余三个点的距离的加权系数的平均值, 我们将四边形变成两个三角形来考虑.
            let area_1: Array2<f64> = arr2(&[
                [k_range.row(0)[0], k_range.row(1)[0]],
                [k_range.row(0)[1], k_range.row(1)[0]],
                [k_range.row(0)[0], k_range.row(1)[1]],
            ]); //第一个三角形
            let area_2: Array2<f64> = arr2(&[
                [k_range.row(0)[1], k_range.row(1)[1]],
                [k_range.row(0)[1], k_range.row(1)[0]],
                [k_range.row(0)[0], k_range.row(1)[1]],
            ]); //第二个三角形
            #[inline(always)]
            fn adapt_integrate_triangle(
                f0: &dyn Fn(&Array1<f64>) -> f64,
                kvec: &Array2<f64>,
                re_err: f64,
                ab_err: f64,
                s1: f64,
                s2: f64,
                s3: f64,
                S: f64,
            ) -> f64 {
                //这个函数是用来进行自适应算法的
                let mut result = 0.0;
                let mut use_kvec = vec![(kvec.clone(), re_err, ab_err, s1, s2, s3, S)];
                while let Some((kvec, re_err, ab_err, s1, s2, s3, S)) = use_kvec.pop() {
                    let kvec_m = kvec.mean_axis(Axis(0)).unwrap();
                    let sm: f64 = f0(&kvec_m.to_owned());

                    let mut new_kvec = kvec.to_owned();
                    new_kvec.push_row(kvec_m.view());
                    let kvec_1 = new_kvec.select(Axis(0), &[0, 1, 3]);
                    let kvec_2 = new_kvec.select(Axis(0), &[0, 3, 2]);
                    let kvec_3 = new_kvec.select(Axis(0), &[3, 1, 2]);
                    let all: f64 = (s1 + s2 + s3) * S / 6.0;
                    let all_new: f64 = all / 3.0 * 2.0 + sm * S / 6.0;
                    let abs_err: f64 = if ab_err > all * re_err {
                        ab_err
                    } else {
                        all * re_err
                    };
                    if (all_new - all).abs() > abs_err && S > 1e-8 {
                        let S1 = S / 3.0;
                        use_kvec.push((kvec_1, re_err, ab_err / 3.0, s1, s2, sm, S1));
                        use_kvec.push((kvec_2, re_err, ab_err / 3.0, s1, sm, s3, S1));
                        use_kvec.push((kvec_3, re_err, ab_err / 3.0, sm, s2, s3, S1));
                    } else {
                        result += all_new;
                    }
                }
                result
            }
            let S = (k_range[[0, 1]] - k_range[[0, 0]]) * (k_range[[1, 1]] - k_range[[1, 0]]);
            let s1 = f0(&arr1(&[k_range.row(0)[0], k_range.row(1)[0]]));
            let s2 = f0(&arr1(&[k_range.row(0)[1], k_range.row(1)[0]]));
            let s3 = f0(&arr1(&[k_range.row(0)[0], k_range.row(1)[1]]));
            let s4 = f0(&arr1(&[k_range.row(0)[1], k_range.row(1)[1]]));
            let all_1 = adapt_integrate_triangle(f0, &area_1, re_err, ab_err / 2.0, s1, s2, s3, S);
            let all_2 = adapt_integrate_triangle(f0, &area_2, re_err, ab_err / 2.0, s4, s2, s3, S);
            return all_1 + all_2;
        }
        3 => {
            //对于三位情况, 需要用到四面体, 所以需要先将6面体变成6个四面体
            #[inline(always)]
            fn adapt_integrate_tetrahedron(
                f0: &dyn Fn(&Array1<f64>) -> f64,
                kvec: &Array2<f64>,
                re_err: f64,
                ab_err: f64,
                s1: f64,
                s2: f64,
                s3: f64,
                s4: f64,
                S: f64,
            ) -> f64 {
                //这个函数是用来进行自适应算法的
                let mut result = 0.0;
                let mut use_kvec = vec![(kvec.clone(), re_err, ab_err, s1, s2, s3, s4, S)];
                while let Some((kvec, re_err, ab_err, s1, s2, s3, s4, S)) = use_kvec.pop() {
                    let kvec_m = kvec.mean_axis(Axis(0)).unwrap();
                    let sm = f0(&kvec_m.to_owned());
                    let mut new_kvec = kvec.to_owned();
                    new_kvec.push_row(kvec_m.view());
                    let kvec_1 = new_kvec.select(Axis(0), &[0, 1, 2, 4]);
                    let kvec_2 = new_kvec.select(Axis(0), &[0, 1, 4, 3]);
                    let kvec_3 = new_kvec.select(Axis(0), &[0, 4, 2, 3]);
                    let kvec_4 = new_kvec.select(Axis(0), &[4, 1, 2, 3]);

                    let all = (s1 + s2 + s3 + s4) * S / 24.0;
                    let all_new = all * 0.75 + sm * S / 24.0;
                    let S1 = S * 0.25;
                    let abs_err = if ab_err > all * re_err {
                        ab_err
                    } else {
                        all * re_err
                    };
                    if (all_new - all).abs() > abs_err && S > 1e-9 {
                        use_kvec.push((kvec_1, re_err, ab_err * 0.25, s1, s2, s3, sm, S1));
                        use_kvec.push((kvec_2, re_err, ab_err * 0.25, s1, s2, sm, s4, S1));
                        use_kvec.push((kvec_3, re_err, ab_err * 0.25, s1, sm, s3, s4, S1));
                        use_kvec.push((kvec_4, re_err, ab_err * 0.25, sm, s2, s3, s4, S1));
                    } else {
                        result += all_new;
                    }
                }
                result
            }

            let k_points: Array2<f64> = arr2(&[
                [k_range.row(0)[0], k_range.row(1)[0], k_range.row(2)[0]],
                [k_range.row(0)[1], k_range.row(1)[0], k_range.row(2)[0]],
                [k_range.row(0)[0], k_range.row(1)[1], k_range.row(2)[0]],
                [k_range.row(0)[1], k_range.row(1)[1], k_range.row(2)[0]],
                [k_range.row(0)[0], k_range.row(1)[0], k_range.row(2)[1]],
                [k_range.row(0)[1], k_range.row(1)[0], k_range.row(2)[1]],
                [k_range.row(0)[0], k_range.row(1)[1], k_range.row(2)[1]],
                [k_range.row(0)[1], k_range.row(1)[1], k_range.row(2)[1]],
            ]); //六面体的顶点

            let area_1 = k_points.select(Axis(0), &[0, 1, 2, 5]);
            let area_2 = k_points.select(Axis(0), &[0, 2, 4, 5]);
            let area_3 = k_points.select(Axis(0), &[6, 2, 4, 5]);
            let area_4 = k_points.select(Axis(0), &[1, 2, 3, 5]);
            let area_5 = k_points.select(Axis(0), &[7, 2, 3, 5]);
            let area_6 = k_points.select(Axis(0), &[7, 2, 6, 5]);
            let s0 = f0(&k_points.row(0).to_owned());
            let s1 = f0(&k_points.row(1).to_owned());
            let s2 = f0(&k_points.row(2).to_owned());
            let s3 = f0(&k_points.row(3).to_owned());
            let s4 = f0(&k_points.row(4).to_owned());
            let s5 = f0(&k_points.row(5).to_owned());
            let s6 = f0(&k_points.row(6).to_owned());
            let s7 = f0(&k_points.row(7).to_owned());
            let V = (k_range[[0, 1]] - k_range[[0, 0]])
                * (k_range[[1, 1]] - k_range[[1, 0]])
                * (k_range[[2, 1]] - k_range[[2, 0]]);
            let all_1 =
                adapt_integrate_tetrahedron(f0, &area_1, re_err, ab_err / 6.0, s0, s1, s2, s5, V);
            let all_2 =
                adapt_integrate_tetrahedron(f0, &area_2, re_err, ab_err / 6.0, s0, s2, s4, s5, V);
            let all_3 =
                adapt_integrate_tetrahedron(f0, &area_3, re_err, ab_err / 6.0, s6, s2, s4, s5, V);
            let all_4 =
                adapt_integrate_tetrahedron(f0, &area_4, re_err, ab_err / 6.0, s1, s2, s3, s5, V);
            let all_5 =
                adapt_integrate_tetrahedron(f0, &area_5, re_err, ab_err / 6.0, s7, s2, s3, s5, V);
            let all_6 =
                adapt_integrate_tetrahedron(f0, &area_5, re_err, ab_err / 6.0, s7, s2, s6, s5, V);
            return all_1 + all_2 + all_3 + all_4 + all_5 + all_6;
        }
        _ => {
            panic!(
                "wrong, the row_dim if k_range must be 1,2 or 3, but you's give {}",
                dim
            );
        }
    }
}

/// Trait providing Berry curvature calculations.
///
/// This trait requires the [`Velocity`] trait and provides methods to compute
/// the Berry curvature and spin Berry curvature at individual k-points or over k-point sets.
///
/// The spin parameter selects the spin operator:
/// - `0`: $\sigma_0$ (identity, charge Berry curvature)
/// - `1`: $\sigma_x$ (spin Berry curvature, x-component)
/// - `2`: $\sigma_y$ (spin Berry curvature, y-component)
/// - `3`: $\sigma_z$ (spin Berry curvature, z-component)
///
/// If the model is spinless, the spin parameter is ignored and spin=0 is used.
pub trait BerryCurvature: Velocity {
    /// Computes the Berry curvature for each band at a single k-point.
    ///
    /// Returns `(omega_n, band)` where `omega_n` contains the Berry curvature for each band
    /// and `band` contains the band energies.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates (in fractional reciprocal coordinates).
    /// * `current_dir` - Current direction (direction vector for the first index $\alpha$ of $\Omega_{n,\alpha\beta}$).
    ///   Must have length equal to `self.dim_r()`.
    /// * `dir_2` - Direction vector for the second index $\beta$ of $\Omega_{n,\alpha\beta}$.
    /// * `spin` - Spin operator index (0, 1, 2, 3 for $\sigma_0, \sigma_x, \sigma_y, \sigma_z$).
    /// * `eta` - Broadening parameter $\eta$ for the energy denominator.
    fn berry_curvature_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>);

    /// Computes the temperature-dependent Berry curvature at a single k-point.
    ///
    /// The formula computed is:
    /// $$ \sum_n f_n\Omega_{n,\alpha\beta}^\gamma(\mathbf k) =
    ///    \sum_n \f{1}{e^{(\varepsilon_{n\mathbf k}-\mu)/(k_B T)}+1}
    ///    \sum_{m\neq n} \f{J_{\alpha,nm}^\gamma v_{\beta,mn}}
    ///    {(\varepsilon_{n\mathbf k}-\varepsilon_{m\mathbf k})^2 + \eta^2} $$
    /// where $J_\alpha^\gamma = \{s_\gamma, v_\alpha\}$ is the anti-commutator of the
    /// spin and velocity operators.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates (in fractional reciprocal coordinates).
    /// * `current_dir` - Current direction (direction vector for the first index $\alpha$ of $\Omega_{\alpha\beta}$).
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K). If `T=0`, a step function is used for the Fermi-Dirac distribution.
    /// * `spin` - Spin operator index (0, 1, 2, 3 for $\sigma_0, \sigma_x, \sigma_y, \sigma_z$).
    ///   Ignored if the model is spinless.
    /// * `eta` - Broadening parameter $\eta$ (in eV).
    fn berry_curvature_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> f64;

    /// Computes the Berry curvature at multiple k-points in parallel.
    ///
    /// This is useful for plotting Berry curvature along band structures or generating heat maps.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `current_dir` - Direction vector for the first index $\alpha$.
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// An `Array1<f64>` of length `nk` containing $\sum_n f_n \Omega_{n,\alpha\beta}$ at each k-point.
    ///
    /// # Panics
    ///
    /// Panics if `current_dir.len()` or `dir_2.len()` does not equal `self.dim_r()`.
    #[allow(non_snake_case)]
    fn berry_curvature<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Array1<f64>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> BerryCurvature for Model<SPIN, DIM, R> {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn berry_curvature_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();
        // Build direction matrix: [current_dir, dir_2]
        let directions = {
            let mut d = Array2::<f64>::zeros((2, self.dim_r()));
            d.row_mut(0).assign(current_dir);
            d.row_mut(1).assign(dir_2);
            d
        };
        let (v_proj, hamk) =
            self.gen_v_projected(&k_vec, Gauge::Atom, &directions);
        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let J: Array2<Complex<f64>> = if SPIN && spin.is_some() {
            let sp = build_spin_matrix(self.norb(), spin);
            anti_comm(&sp, &v_proj.slice(s![0, .., ..])) * 0.5
        } else {
            if !SPIN && spin.is_some() {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            v_proj.slice(s![0, .., ..]).to_owned()
        };
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();

        let evec_conj = evec.t();
        // Lazy map: no heap allocation for the conjugated copy
        let evec = evec.map(|x| x.conj());
        let A1 = J.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec);
        let A2 = evec_conj.dot(&A2);
        let A2 = A2.reversed_axes();
        let AA = A1 * A2;
        let Complex { re, im } = AA.view().split_complex();
        let im = im.mapv(|x| -2.0 * x);
        // Fused: compute omega_n directly without allocating UU[nsta,nsta]
        let mut omega_n = Array1::<f64>::zeros(self.nsta());
        for i in 0..self.nsta() {
            let im_row = im.row(i);
            let mut sum = 0.0f64;
            for j in 0..self.nsta() {
                if i != j {
                    let a = band[[i]] - band[[j]];
                    sum += im_row[[j]] / (a.powi(2) + eta.powi(2));
                }
            }
            omega_n[[i]] = sum;
        }
        (omega_n, band)
    }

    #[allow(non_snake_case)]
    fn berry_curvature_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> f64 {
        let (omega_n, band) = self.berry_curvature_n_onek(&k_vec, &current_dir, &dir_2, spin, eta);
        let mut omega: f64 = 0.0;
        let fermi_dirac = if T == 0.0 {
            band.mapv(|x| if x > mu { 0.0 } else { 1.0 })
        } else {
            let beta = 1.0 / T / 8.617e-5;
            band.mapv(|x| ((beta * (x - mu)).exp() + 1.0).recip())
        };
        let omega = omega_n.dot(&fermi_dirac);
        omega
    }
    #[allow(non_snake_case)]
    fn berry_curvature<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Array1<f64> {
        if current_dir.len() != self.dim_r() || dir_2.len() != self.dim_r() {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));
        let omega: Vec<f64> = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let omega_one = self.berry_curvature_onek(
                    &x.to_owned(),
                    &current_dir,
                    &dir_2,
                    mu,
                    T,
                    spin,
                    eta,
                );
                omega_one
            })
            .collect();
        let omega = arr1(&omega);
        omega
    }
}

#[allow(non_snake_case)]
impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Methods for computing conductivity tensors including the anomalous Hall conductivity,
    /// spin Hall conductivity, and nonlinear Hall conductivity.
    ///
    /// The Hall conductivity is computed by integrating the Berry curvature over the Brillouin zone:
    /// $$ \sigma_{\alpha\beta}^\gamma = \f{e^2}{\hbar} \int_{BZ} \f{\dd\mathbf k}{(2\pi)^d}
    ///    \sum_n f_n \Omega_{n,\alpha\beta}^\gamma(\mathbf k) $$
    /// where $d$ is the spatial dimension, $f_n$ is the Fermi-Dirac distribution, and
    /// $\Omega_{n,\alpha\beta}^\gamma$ is the (spin) Berry curvature.
    ///
    /// The output is in units of $e^2/\hbar$ per length (in Angstrom) in 3D, or $e^2/\hbar$ in 2D.
    ///
    /// The `spin` parameter selects the spin operator:
    /// - `0`: $\sigma_0$ (identity, charge Hall conductivity)
    /// - `1`: $\sigma_x$ (spin Hall conductivity, x-component)
    /// - `2`: $\sigma_y$ (spin Hall conductivity, y-component)
    /// - `3`: $\sigma_z$ (spin Hall conductivity, z-component)
    ///
    /// If the model is spinless, the spin parameter is ignored and spin=0 is used.

    /// Computes the anomalous Hall conductivity at a given chemical potential and temperature.
    ///
    /// Uses a uniform k-mesh and direct summation:
    /// $$ \sigma_{\alpha\beta}^\gamma = \f{1}{N (2\pi)^d V} \sum_{\mathbf k} \Omega_{\alpha\beta}^\gamma(\mathbf k), $$
    /// where $N$ is the number of k-points, $d$ is the dimension, and $V$ is the unit cell volume.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction, e.g. `arr1(&[nk, nk])` for 2D.
    /// * `current_dir` - Direction vector for the first index $\alpha$ of $\sigma_{\alpha\beta}$.
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K). Use `T=0` for the zero-temperature step function.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$ (in eV).
    ///
    /// # Returns
    ///
    /// The Hall conductivity $\sigma_{\alpha\beta}$ in units of $e^2/\hbar/\AA$ (3D) or $e^2/\hbar$ (2D).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ndarray::arr1;
    /// # use rustb::Model;
    /// # fn example(model: &Model) -> Result<(), rustb::error::TbError> {
    /// let kmesh = arr1(&[31, 31]);
    /// let current_dir = arr1(&[1.0, 0.0]);
    /// let dir_2 = arr1(&[0.0, 1.0]);
    /// let sigma_xy = model.Hall_conductivity(&kmesh, &current_dir, &dir_2, 0.0, 0.0, None, 1e-3)?;
    /// println!("Hall conductivity = {}", sigma_xy);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(non_snake_case)]
    pub fn Hall_conductivity(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<f64> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let omega = self.berry_curvature(&kvec, &current_dir, &dir_2, mu, T, spin, eta);
        //目前求积分的方法上, 还是直接求和最有用, 其他的典型积分方法, 如gauss 法等,
        //都因为存在间断点而效率不高.
        //对于非零温的, 使用梯形法应该效果能好一些.
        let conductivity: f64 = omega.sum() / (nk as f64) / self.lat.det().unwrap();
        Ok(conductivity)
    }

    /// Computes the Hall conductivity using an adaptive integration algorithm.
    ///
    /// This method uses [`adapted_integrate_quick`] to refine the integration mesh adaptively,
    /// which can produce more accurate results with fewer k-points compared to uniform sampling.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of subdomain divisions along each direction.
    /// * `current_dir`, `dir_2` - Direction vectors for the conductivity tensor indices.
    /// * `mu` - Chemical potential (in eV).
    /// * `T` - Temperature (in K).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter (in eV).
    /// * `re_err` - Relative error tolerance for the adaptive integrator. Recommended: `1.0`.
    /// * `ab_err` - Absolute error tolerance for the adaptive integrator. Recommended: `0.01`.
    ///
    /// # Returns
    ///
    /// The Hall conductivity in units of $e^2/\hbar/\AA$ (3D) or $e^2/\hbar$ (2D).
    #[allow(non_snake_case)]
    pub fn Hall_conductivity_adapted(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
        re_err: f64,
        ab_err: f64,
    ) -> Result<f64> {
        let mut k_range = gen_krange(k_mesh)?; //将要计算的区域分成小块
        let n_range = k_range.len_of(Axis(0));
        let ab_err = ab_err / (n_range as f64);
        let use_fn = |k0: &Array1<f64>| {
            self.berry_curvature_onek(k0, &current_dir, &dir_2, mu, T, spin, eta)
        };
        let inte = |k_range| adapted_integrate_quick(&use_fn, &k_range, re_err, ab_err);
        let omega: Vec<f64> = k_range
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| inte(x.to_owned()))
            .collect();
        let omega: Array1<f64> = arr1(&omega);
        let conductivity: f64 = omega.sum() / self.lat.det().unwrap();
        Ok(conductivity)
    }

    /// Computes the Hall conductivity for multiple chemical potential values efficiently.
    ///
    /// This method first computes $\Omega_n$ (the Berry curvature per band) at each k-point,
    /// then evaluates the Fermi-Dirac-weighted sum for each $\mu$. This avoids repeatedly
    /// computing $\Omega_n$, making it much faster than calling [`Hall_conductivity`] for each $\mu$.
    /// However, it uses more memory and cannot use adaptive integration.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction.
    /// * `current_dir`, `dir_2` - Direction vectors for the conductivity tensor indices.
    /// * `mu` - Array of chemical potential values (in eV).
    /// * `T` - Temperature (in K).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter (in eV).
    ///
    /// # Returns
    ///
    /// An `Array1<f64>` of Hall conductivity values, one for each $\mu$, in units of $e^2/\hbar/\AA$ (3D) or $e^2/\hbar$ (2D).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ndarray::Array1;
    /// # use rustb::Model;
    /// # fn example(model: &Model) -> Result<(), rustb::error::TbError> {
    /// let kmesh = ndarray::arr1(&[31, 31]);
    /// let current_dir = ndarray::arr1(&[1.0, 0.0]);
    /// let dir_2 = ndarray::arr1(&[0.0, 1.0]);
    /// let mu = Array1::linspace(-2.0, 2.0, 101);
    /// let sigma_vs_mu = model.Hall_conductivity_mu(&kmesh, &current_dir, &dir_2, &mu, 0.0, None, 1e-3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn Hall_conductivity_mu(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let (omega_n, band): (Vec<_>, Vec<_>) = kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let (omega_n, band) =
                    self.berry_curvature_n_onek(&x.to_owned(), &current_dir, &dir_2, spin, eta);
                (omega_n, band)
            })
            .collect();
        let omega_n = Array2::<f64>::from_shape_vec(
            (nk, self.nsta()),
            omega_n.into_iter().flatten().collect(),
        )
        .unwrap();
        let band =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), band.into_iter().flatten().collect())
                .unwrap();
        let n_mu: usize = mu.len();
        let conductivity = if T == 0.0 {
            let conductivity_new: Vec<f64> = mu
                .into_par_iter()
                .map(|x| {
                    let mut omega = Array1::<f64>::zeros(nk);
                    for k in 0..nk {
                        for i in 0..self.nsta() {
                            omega[[k]] += if band[[k, i]] > *x {
                                0.0
                            } else {
                                omega_n[[k, i]]
                            };
                        }
                    }
                    omega.sum() / self.lat.det().unwrap() / (nk as f64)
                })
                .collect();
            Array1::<f64>::from_vec(conductivity_new)
        } else {
            let beta = 1.0 / (T * 8.617e-5);
            let conductivity_new: Vec<f64> = mu
                .into_par_iter()
                .map(|x| {
                    let fermi_dirac = band.mapv(|x0| 1.0 / ((beta * (x0 - x)).exp() + 1.0));
                    let omega: Vec<f64> = omega_n
                        .axis_iter(Axis(0))
                        .zip(fermi_dirac.axis_iter(Axis(0)))
                        .map(|(a, b)| (&a * &b).sum())
                        .collect();
                    let omega: Array1<f64> = arr1(&omega);
                    omega.sum() / self.lat.det().unwrap() / (nk as f64)
                })
                .collect();
            Array1::<f64>::from_vec(conductivity_new)
        };
        Ok(conductivity)
    }

    /// Computes the Berry curvature dipole for each band at a single k-point.
    ///
    /// This computes:
    /// $$ \pdv{\varepsilon_{n\mathbf k}}{k_\gamma} \Omega_{n,\alpha\beta} $$
    ///
    /// The energy derivative is obtained using the diagonal elements of the velocity operator:
    /// $$ \pdv{\varepsilon_{\mathbf k}}{\mathbf k} = \text{diag}(v_{\mathbf k}) $$
    /// This follows from the relation $\varepsilon_{\mathbf k} = U^\dagger H_{\mathbf k} U$ and
    /// the observation that the commutator term $[\varepsilon_{\mathbf k}, U^\dagger\partial_{\mathbf k}U]$
    /// does not contribute to diagonal elements.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates.
    /// * `current_dir` - Current direction (direction vector for the first index $\alpha$ of $\Omega_{n,\alpha\beta}$).
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `dir_3` - Direction vector for the derivative index $\gamma$.
    /// * `og` - Frequency $\omega$ (for the energy denominator).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// `(omega_n, band)` where `omega_n` contains $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$
    /// for each band, and `band` contains the band energies.
    pub fn berry_curvature_dipole_n_onek(
        &self,
        k_vec: &Array1<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();
        // Build direction matrix: [current_dir, dir_2, dir_3]
        let directions = {
            let mut d = Array2::<f64>::zeros((3, self.dim_r()));
            d.row_mut(0).assign(current_dir);
            d.row_mut(1).assign(dir_2);
            d.row_mut(2).assign(dir_3);
            d
        };
        let (v_proj, hamk) =
            self.gen_v_projected(&k_vec, Gauge::Atom, &directions);
        // v_proj[0] = Σ_d current_dir[d] * v_raw[d]  → J
        // v_proj[1] = Σ_d dir_2[d] * v_raw[d]        → v
        // v_proj[2] = Σ_d dir_3[d] * v_raw[d]        → v0
        let J: Array2<Complex<f64>> = if SPIN {
            let X = build_spin_matrix(self.norb(), spin);
            anti_comm(&X, &v_proj.slice(s![0, .., ..])) * 0.5
        } else {
            if spin.is_some() {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            v_proj.slice(s![0, .., ..]).to_owned()
        };
        let v: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();
        let v0: Array2<Complex<f64>> = v_proj.slice(s![2, .., ..]).to_owned();

        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec_conj = evec.t();
        let evec = evec.map(|x| x.conj());

        let v0 = v0.dot(&evec.t());
        let v0 = &evec_conj.dot(&v0);
        let partial_ve = v0.diag().map(|x| x.re);
        let A1 = J.dot(&evec.t());
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec.t());
        let A2 = &evec_conj.dot(&A2);
        let mut U0 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                if i != j {
                    U0[[i, j]] = 1.0 / ((band[[i]] - band[[j]]).powi(2) - (og + li * eta).powi(2));
                } else {
                    U0[[i, j]] = Complex::new(0.0, 0.0);
                }
            }
        }
        //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        let mut omega_n = Array1::<f64>::zeros(self.nsta());
        let A1 = A1 * U0;
        for i in 0..self.nsta() {
            omega_n[[i]] = -2.0 * A1.slice(s![i, ..]).dot(&A2.slice(s![.., i])).im;
        }

        //let (omega_n,band)=self.berry_curvature_n_onek(&k_vec,&current_dir,&dir_2,og,spin,eta);
        let omega_n: Array1<f64> = omega_n * partial_ve;
        (omega_n, band) //最后得到的 D
    }

    /// Computes the Berry curvature dipole for each band at multiple k-points in parallel.
    ///
    /// This is a parallelized version of [`berry_curvature_dipole_n_onek`] for computing
    /// the Berry curvature dipole over a k-point set.
    ///
    /// The extrinsic nonlinear Hall conductivity is related to this quantity via:
    /// $$ \sigma_{\alpha\beta\gamma} = \tau \int \dd\mathbf k \sum_n
    ///    \partial_\gamma \varepsilon_{n\mathbf k} \Omega_{n,\alpha\beta}
    ///    \left. \pdv{f_{\mathbf k}}{\varepsilon} \right\rvert_{E=\varepsilon_{n\mathbf k}}. $$
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `current_dir`, `dir_2` - Direction vectors for the Berry curvature indices $\alpha, \beta$.
    /// * `dir_3` - Direction vector for the energy derivative index $\gamma$.
    /// * `og` - Frequency $\omega$.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter.
    ///
    /// # Returns
    ///
    /// `(omega, band)` where `omega` has shape `(nk, nsta)` containing
    /// $\partial_\gamma\varepsilon_n \Omega_{n,\alpha\beta}$ for each k-point and band,
    /// and `band` has the band energies with the same shape.
    ///
    /// # Panics
    ///
    /// Panics if any of `current_dir`, `dir_2`, or `dir_3` has length different from `self.dim_r()`.
    pub fn berry_curvature_dipole_n(
        &self,
        k_vec: &Array2<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        if current_dir.len() != self.dim_r()
            || dir_2.len() != self.dim_r()
            || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));
        let (omega, band): (Vec<_>, Vec<_>) = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let (omega_one, band) = self.berry_curvature_dipole_n_onek(
                    &x.to_owned(),
                    &current_dir,
                    &dir_2,
                    &dir_3,
                    og,
                    spin,
                    eta,
                );
                (omega_one, band)
            })
            .collect();
        let omega =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), omega.into_iter().flatten().collect())
                .unwrap();
        let band =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), band.into_iter().flatten().collect())
                .unwrap();
        (omega, band)
    }

    /// Computes the extrinsic nonlinear Hall conductivity via the Berry curvature dipole.
    ///
    /// This integrates the Berry curvature dipole over the Brillouin zone:
    /// $$ \mathcal D_{\alpha\beta\gamma} = \int \dd\mathbf k \sum_n
    ///    \left(-\pdv{f_n}{\varepsilon}\right) \partial_\gamma \varepsilon_{n\mathbf k}
    ///    \Omega_{n,\alpha\beta} $$
    ///
    /// The energy derivative of the Fermi-Dirac distribution is:
    /// $$ -\pdv{f_n}{\varepsilon} = \beta \f{e^{\beta(\varepsilon_n-\mu)}}{(e^{\beta(\varepsilon_n-\mu)}+1)^2}
    ///    = \beta f_n(1-f_n) $$
    ///
    /// **Note:** This function currently only supports $T \neq 0$. The $T=0$ case is not yet implemented.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction.
    /// * `current_dir`, `dir_2` - Direction vectors for the Berry curvature indices $\alpha, \beta$.
    /// * `dir_3` - Direction vector for the energy derivative index $\gamma$.
    /// * `mu` - Array of chemical potential values (in eV).
    /// * `T` - Temperature (in K). Must be non-zero.
    /// * `og` - Frequency $\omega$ (use 0 for the DC limit).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$.
    ///
    /// # Returns
    ///
    /// The extrinsic nonlinear Hall conductivity for each $\mu$ value.
    ///
    /// # Panics
    ///
    /// Panics if `T == 0` (not yet supported).
    pub fn Nonlinear_Hall_conductivity_Extrinsic(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        og: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        if current_dir.len() != self.dim_r()
            || dir_2.len() != self.dim_r()
            || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        //为了节省内存, 本来是可以直接算完求和, 但是为了方便, 我是先存下来再算, 让程序结构更合理
        let (omega, band) =
            self.berry_curvature_dipole_n(&kvec, &current_dir, &dir_2, &dir_3, og, spin, eta);
        let omega = omega.into_raw_vec();
        let band = band.into_raw_vec();
        let n_e = mu.len();
        let mut conductivity = Array1::<f64>::zeros(n_e);
        if T != 0.0 {
            let beta = 1.0 / T / (8.617e-5);
            let use_iter = band.iter().zip(omega.iter()).par_bridge();
            conductivity = use_iter
                .fold(
                    || Array1::<f64>::zeros(n_e),
                    |acc, (energy, omega0)| {
                        let f = 1.0 / (beta * (mu - *energy)).mapv(|x| x.exp() + 1.0);
                        acc + &f * (1.0 - &f) * beta * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        } else {
            // T=0: Blochl tetrahedron integration
            let omega_2d =
                Array2::<f64>::from_shape_vec((nk, self.nsta()), omega).unwrap();
            let band_2d =
                Array2::<f64>::from_shape_vec((nk, self.nsta()), band).unwrap();
            conductivity =
                crate::tetrahedron::tetrahedron_integrate(&band_2d, &omega_2d, k_mesh, mu)
                    / self.lat.det().unwrap();
        }
        Ok(conductivity)
    }

    /// Computes the Berry connection dipole at a single k-point.
    ///
    /// For spinless models, this computes:
    /// $$ v_\alpha G_{\beta\gamma} - v_\beta G_{\alpha\gamma} $$
    /// where
    /// $$ G_{ij} = -2\,\text{Re} \sum_{m\neq n} \f{v_{i,nm} v_{j,mn}}{(\varepsilon_n - \varepsilon_m)^3} $$
    ///
    /// For spinful models (when `spin != 0`), this additionally computes
    /// $\partial_{h_i} G_{jk}$, the derivative with respect to the spin field.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - k-point coordinates.
    /// * `current_dir` - Direction vector for the first index $\alpha$.
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `dir_3` - Direction vector for the third index $\gamma$.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    ///
    /// # Returns
    ///
    /// `(omega, band, partial_G)` where:
    /// - `omega`: $v_\alpha G_{\beta\gamma} - v_\beta G_{\alpha\gamma}$ per band.
    /// - `band`: Band energies.
    /// - `partial_G`: $\partial_{h} G$ per band (only `Some` for spinful models, `None` for spinless).
    pub fn berry_connection_dipole_onek(
        &self,
        k_vec: &Array1<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: Option<SpinDirection>,
    ) -> (Array1<f64>, Array1<f64>, Option<Array1<f64>>) {
        // Build direction matrix: [current_dir, dir_2, dir_3]
        let directions = {
            let mut d = Array2::<f64>::zeros((3, self.dim_r()));
            d.row_mut(0).assign(current_dir);
            d.row_mut(1).assign(dir_2);
            d.row_mut(2).assign(dir_3);
            d
        };
        let (v_proj, hamk) =
            self.gen_v_projected(&k_vec, Gauge::Atom, &directions);
        // v_proj[0] = Σ_d current_dir[d] * v_raw[d]
        // v_proj[1] = Σ_d dir_2[d] * v_raw[d]
        // v_proj[2] = Σ_d dir_3[d] * v_raw[d]

        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec_conj = evec.t();
        let evec = evec.map(|x| x.conj());
        // Precompute transposed evec once — avoids per-direction clone in original
        let evec_t: Array2<Complex<f64>> = evec.t().to_owned();

        // Transform projected matrices to eigenbasis in one shot per projection
        let v0: Array2<Complex<f64>> = v_proj.slice(s![0, .., ..]).to_owned();
        let v1: Array2<Complex<f64>> = v_proj.slice(s![1, .., ..]).to_owned();
        let v2: Array2<Complex<f64>> = v_proj.slice(s![2, .., ..]).to_owned();
        let v_1 = evec_conj.dot(&v0.dot(&evec_t));
        let v_2 = evec_conj.dot(&v1.dot(&evec_t));
        let v_3 = evec_conj.dot(&v2.dot(&evec_t));
        //三个方向的速度算符都得到了
        let mut U0 = Array2::<f64>::zeros((self.nsta(), self.nsta()));
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                if (band[[i]] - band[[j]]).abs() < 1e-5 {
                    U0[[i, j]] = 0.0;
                } else {
                    U0[[i, j]] = 1.0 / (band[[i]] - band[[j]]);
                }
            }
        }
        //这样U0[[i,j]]=1/(E_i-E_j), 这样就可以省略判断, 减少计算量

        //开始计算能带的导数, 详细的公式请看 berry_curvature_dipole_onek 的公式
        //其实就是速度算符的对角项
        //开始计算速度的偏导项, 这里偏导来自实空间
        let partial_ve_1 = v_1.diag().map(|x| x.re);
        let partial_ve_2 = v_2.diag().map(|x| x.re);
        let partial_ve_3 = v_3.diag().map(|x| x.re);

        //开始最后的计算
        // Only enter spin branch when model is spinful AND spin requested
        if SPIN && spin.is_some() {
            // Anti-commute on projected raw matrices (once each, not per-direction)
            let X = build_spin_matrix(self.norb(), spin);
            let s_1_raw = anti_comm(&X, &v_proj.slice(s![0, .., ..])) * 0.5;
            let s_2_raw = anti_comm(&X, &v_proj.slice(s![1, .., ..])) * 0.5;
            let s_3_raw = anti_comm(&X, &v_proj.slice(s![2, .., ..])) * 0.5;
            // Transform to eigenbasis
            let s_1 = evec_conj.dot(&s_1_raw.dot(&evec_t));
            let s_2 = evec_conj.dot(&s_2_raw.dot(&evec_t));
            let s_3 = evec_conj.dot(&s_3_raw.dot(&evec_t));
            let G_23: Array1<f64> = {
                //用来计算  beta gamma 的 G
                let A = &v_2 * (U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0)));
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&v_3.slice(s![.., i])).re * 2.0
                }
                G
            };
            let G_13_h: Array1<f64> = {
                //用来计算 alpha gamma 的 G
                let A = &s_1 * (U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0)));
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&v_3.slice(s![.., i])).re * 2.0
                }
                G
            };
            let partial_s_1 = s_1.diag().map(|x| x.re);
            let partial_s_2 = s_2.diag().map(|x| x.re);
            let partial_s_3 = s_3.diag().map(|x| x.re);
            //开始计算partial G
            let partial_G: Array1<f64> = {
                let mut A = Array1::<Complex<f64>>::zeros(self.nsta()); //第一项
                for i in 0..self.nsta() {
                    for j in 0..self.nsta() {
                        A[[i]] += 3.0
                            * (partial_s_1[[i]] - partial_s_1[[j]])
                            * v_2[[i, j]]
                            * v_3[[j, i]]
                            * U0[[i, j]].powi(4);
                    }
                }
                let mut B = Array1::<Complex<f64>>::zeros(self.nsta()); //第二项
                for n in 0..self.nsta() {
                    for n1 in 0..self.nsta() {
                        for n2 in 0..self.nsta() {
                            B[[n]] += s_1[[n, n2]]
                                * (v_2[[n2, n1]] * v_3[[n1, n]] + v_3[[n2, n1]] * v_2[[n1, n]])
                                * U0[[n, n1]].powi(3)
                                * U0[[n, n2]];
                        }
                    }
                }
                let mut C = Array1::<Complex<f64>>::zeros(self.nsta()); //第三项
                for n in 0..self.nsta() {
                    for n1 in 0..self.nsta() {
                        for n2 in 0..self.nsta() {
                            C[[n]] += s_1[[n1, n2]]
                                * (v_2[[n2, n]] * v_3[[n, n1]] + v_3[[n2, n]] * v_2[[n, n1]])
                                * U0[[n, n1]].powi(3)
                                * U0[[n1, n2]];
                        }
                    }
                }
                2.0 * (A - B - C).map(|x| x.re)
            };
            //计算结束
            //开始最后的输出
            return (
                (partial_s_1 * G_23 - partial_ve_2 * G_13_h),
                band,
                Some(partial_G),
            );
        } else {
            // —— SM Eq. (43): charge intrinsic nonlinear Hall ——
            // Original: σ^{ab;c}_{int} = -e³/ħ Σ_n ∫_k f_n
            //   [2 ∂_c G^{ab}_n − 1/2 (∂_a G^{bc}_n + ∂_b G^{ac}_n)]
            //
            // Integration by parts (surface term vanishes):
            //   ∫ f_n ∂_i G = −∫ (∂_i f_n) G = ∫ (−∂f/∂ε_n) v_i G
            //
            // After ibp → kernel:
            //   Q^{ab;c}_n = 2 v_c G^{ab}_n − ½ (v_a G^{bc}_n + v_b G^{ac}_n)
            //
            // Conductivity (+ overall −e³/ħ sign):
            //   σ = −e³/ħ · Σ_{k,n} (−∂f/∂ε_n) · Q_n / (N_k · det(lat))
            //
            // G^{ij}_n = Re Σ_{m≠n} v^i_{nm} v^j_{mn} / (ε_n − ε_m)³
            let calc_G = |va: &Array2<Complex<f64>>,
                          vb: &Array2<Complex<f64>>|
             -> Array1<f64> {
                let U3 = U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0));
                let A = va * &U3;
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&vb.slice(s![.., i])).re;
                }
                G
            };

            let G_12 = calc_G(&v_1, &v_2); // G_{ab}
            let G_13 = calc_G(&v_1, &v_3); // G_{ac}
            let G_23 = calc_G(&v_2, &v_3); // G_{bc}

            // Q^{ab;c}_n = 2 v_c G_{ab} − ½ (v_a G_{bc} + v_b G_{ac})
            // v_1 = v_a,  v_2 = v_b,  v_3 = v_c
            let omega = &partial_ve_3 * &G_12 * 2.0
                - (&partial_ve_1 * &G_23 + &partial_ve_2 * &G_13) * 0.5;
            // −e³/ħ overall factor (only the minus sign; physical prefactor
            // left to the caller)
            return (-omega, band, None);
        }
    }

    /// Computes the Berry connection dipole at multiple k-points in parallel.
    ///
    /// This is a parallelized version of [`berry_connection_dipole_onek`].
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `current_dir`, `dir_2`, `dir_3` - Direction vectors for the three indices.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    ///
    /// # Returns
    ///
    /// `(omega, band, partial_G)` where `omega` and `band` have shape `(nk, nsta)`.
    /// For spinful models, `partial_G` is `Some` with shape `(nk, nsta)`.
    /// For spinless models, `partial_G` is `None`.
    pub fn berry_connection_dipole(
        &self,
        k_vec: &Array2<f64>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: Option<SpinDirection>,
    ) -> (Array2<f64>, Array2<f64>, Option<Array2<f64>>) {
        if current_dir.len() != self.dim_r()
            || dir_2.len() != self.dim_r()
            || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the current_dir or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {} and {}",
                self.dim_r(),
                current_dir.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));

        if SPIN {
            let ((omega, band), partial_G): ((Vec<_>, Vec<_>), Vec<_>) = k_vec
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|x| {
                    let (omega_one, band, partial_G) = self.berry_connection_dipole_onek(
                        &x.to_owned(),
                        &current_dir,
                        &dir_2,
                        &dir_3,
                        spin,
                    );
                    let partial_G = partial_G.unwrap();
                    ((omega_one, band), partial_G)
                })
                .collect();

            let omega = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                omega.into_iter().flatten().collect(),
            )
            .unwrap();
            let band = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                band.into_iter().flatten().collect(),
            )
            .unwrap();
            let partial_G = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                partial_G.into_iter().flatten().collect(),
            )
            .unwrap();

            return (omega, band, Some(partial_G));
        } else {
            let (omega, band): (Vec<_>, Vec<_>) = k_vec
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|x| {
                    let (omega_one, band, partial_G) = self.berry_connection_dipole_onek(
                        &x.to_owned(),
                        &current_dir,
                        &dir_2,
                        &dir_3,
                        spin,
                    );
                    (omega_one, band)
                })
                .collect();
            let omega = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                omega.into_iter().flatten().collect(),
            )
            .unwrap();
            let band = Array2::<f64>::from_shape_vec(
                (nk, self.nsta()),
                band.into_iter().flatten().collect(),
            )
            .unwrap();
            return (omega, band, None);
        }
    }

    /// Computes the intrinsic nonlinear Hall conductivity.
    ///
    /// The intrinsic nonlinear Hall conductivity arises from the correction of the Berry connection
    /// by electric and magnetic fields [PRL 112, 166601 (2014)]. The modified Berry curvature is:
    /// $$ \tilde{\bm\Omega}_{\mathbf k} = \nabla_{\mathbf k} \times (\bm A_{\mathbf k} + \bm A_{\mathbf k}^\prime) $$
    /// where $\bm A_{i,\mathbf k}^\prime = F_{ij} B_j + G_{ij} E_j$, with
    /// $$
    /// \begin{aligned}
    /// F_{ij} &= \text{Im} \sum_{m\neq n} \f{v_{i,nm} \omega_{j,mn}}{(\varepsilon_n - \varepsilon_m)^2} \\
    /// G_{ij} &= 2\,\text{Re} \sum_{m\neq n} \f{v_{i,nm} v_{j,mn}}{(\varepsilon_n - \varepsilon_m)^3} \\
    /// \omega_{\alpha,mn} &= -i \varepsilon_{\alpha\beta\gamma} \sum_{l\neq n}
    ///    \f{(v_{\beta,ml} + \partial_\beta \varepsilon_{\mathbf k} \delta_{ml}) v_{\gamma,ln}}{\varepsilon_l - \varepsilon_n}
    /// \end{aligned}
    /// $$
    ///
    /// The current response is:
    /// $$
    /// \begin{aligned}
    /// \f{\partial^2 j_\alpha^\prime}{\partial E_\beta \partial E_\gamma}
    ///    &= \int \f{\dd\mathbf k}{(2\pi)^3}
    ///       (\partial_\alpha \varepsilon_{\mathbf k} G_{\beta\gamma} -
    ///        \partial_\beta \varepsilon_{\mathbf k} G_{\alpha\gamma})
    ///       \pdv{f_{\mathbf k}}{\varepsilon} \\
    /// \f{\partial^2 j_\alpha^\prime}{\partial E_\beta \partial B_\gamma}
    ///    &= \int \f{\dd\mathbf k}{(2\pi)^3}
    ///       (\partial_\alpha \varepsilon_{\mathbf k} F_{\beta\gamma} -
    ///        \partial_\beta \varepsilon_{\mathbf k} F_{\alpha\gamma} +
    ///        \varepsilon_{\alpha\beta\ell} \Omega_\ell m_\gamma)
    ///       \pdv{f_{\mathbf k}}{\varepsilon}
    /// \end{aligned}
    /// $$
    ///
    /// Because of the $\partial f_{\mathbf k}/\partial\varepsilon$ factor, it is recommended to
    /// use $T \neq 0$. The $T=0$ case (using Gauss's theorem to integrate over the Fermi surface)
    /// is not yet implemented.
    ///
    /// For spin Hall conductivity, the formula is [PRL 112, 166601 (2014)]:
    /// $$ \sigma_{\alpha\beta\gamma}^i = -\int \dd\mathbf k \left[
    ///    \f{1}{2} f_{\mathbf k} \pdv{G_{\beta\gamma}}{h_\alpha} +
    ///    \pdv{f_{\mathbf k}}{\varepsilon}
    ///    (\partial_\alpha s_{\mathbf k}^i G_{\beta\gamma} -
    ///     \partial_\beta \varepsilon_{\mathbf k} G_{\alpha\gamma}^h) \right] $$
    /// where
    /// $$ \f{\partial G_{\beta\gamma,n}}{\partial h_\alpha} =
    ///    2\,\text{Re} \sum_{n'\neq n}
    ///    \f{3 (s_{\alpha,n}^i - s_{\alpha,n_1}^i) v_{\beta,nn_1} v_{\gamma,n'n}}
    ///      {(\varepsilon_n - \varepsilon_{n'})^4}
    ///    - 2\,\text{Re} \sum_{n_1\neq n} \sum_{n_2\neq n}
    ///      \left[ \f{s_{\alpha,nn_2}^i v_{\beta,n_2n_1} v_{\gamma,n_1n}}
    ///              {(\varepsilon_n - \varepsilon_{n_1})^3 (\varepsilon_n - \varepsilon_{n_2})}
    ///      + (\beta \leftrightarrow \gamma) \right]
    ///    - 2\,\text{Re} \sum_{n_1\neq n} \sum_{n_2\neq n_1}
    ///      \left[ \f{s_{\alpha,n_1n_2}^i v_{\beta,n_2n} v_{\gamma,nn_1}}
    ///              {(\varepsilon_n - \varepsilon_{n_1})^3 (\varepsilon_{n_1} - \varepsilon_{n_2})}
    ///      + (\beta \leftrightarrow \gamma) \right] $$
    /// and
    /// $$
    /// \begin{aligned}
    /// G_{\alpha\beta}   &= 2\,\text{Re} \sum_{m\neq n} \f{v_{\alpha,nm} v_{\beta,mn}}{(\varepsilon_n - \varepsilon_m)^3} \\
    /// G_{\alpha\beta}^h &= 2\,\text{Re} \sum_{m\neq n} \f{s_{\alpha,nm}^i v_{\beta,mn}}{(\varepsilon_n - \varepsilon_m)^3}
    /// \end{aligned}
    /// $$
    /// where $s_{\alpha,mn}^i = \{ \hat{s}^i, v_\alpha \}$ is the anti-commutator of the spin and velocity operators.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction.
    /// * `current_dir`, `dir_2`, `dir_3` - Direction vectors for the three tensor indices.
    /// * `mu` - Array of chemical potential values (in eV).
    /// * `T` - Temperature (in K). Must be non-zero.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    ///
    /// # Returns
    ///
    /// The intrinsic nonlinear Hall conductivity for each $\mu$ value.
    ///
    /// # Panics
    ///
    /// Panics if `T == 0` (not yet supported).
    pub fn Nonlinear_Hall_conductivity_Intrinsic(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
    ) -> Result<Array1<f64>> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let (omega, band, _partial_G) =
            self.berry_connection_dipole(&kvec, &current_dir, &dir_2, &dir_3, None);
        let omega = omega.into_raw_vec();
        let omega = Array1::from(omega);
        let band0 = band.clone();
        let band = band.into_raw_vec();
        let band = Array1::from(band);
        let n_e = mu.len();
        let mut conductivity = Array1::<f64>::zeros(n_e);
        if T != 0.0 {
            let beta = 1.0 / T / 8.617e-5;
            let use_iter = band.iter().zip(omega.iter()).par_bridge();
            conductivity = use_iter
                .fold(
                    || Array1::<f64>::zeros(n_e),
                    |acc, (energy, omega0)| {
                        let f = 1.0 / ((beta * (*energy - mu)).mapv(|x| x.exp() + 1.0));
                        acc + &f * (1.0 - &f) * beta * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        } else {
            // T=0: Blochl tetrahedron integration
            let omega_2d = omega.into_shape((nk, self.nsta())).unwrap();
            conductivity =
                crate::tetrahedron::tetrahedron_integrate(&band0, &omega_2d, k_mesh, mu)
                    / self.lat.det().unwrap();
        }
        Ok(conductivity)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tetrahedron quadrature for Hall-effect integrals
// ═══════════════════════════════════════════════════════════════════════
//
// ## Assumptions
//
// 1. **Band energies `E_n(k)` are linearly interpolated** inside each simplex
//    (triangle in 2D, tetrahedron in 3D).
//
// 2. **Gauge-invariant velocity kernels are linearly interpolated**:
//    `K^ab_nm(k) = v^a_nm(k) · v^b_mn(k)` is stored at each vertex and
//    interpolated to quadrature points.  This quantity is invariant under
//    independent U(1) phase rotations of bands n and m (when bands are
//    isolated), so it is smooth across the BZ and safe to interpolate.
//
// 3. **Response formulas are evaluated at quadrature points** from
//    interpolated primitives.  The Berry curvature / quantum‑geometry
//    scalar with its singular denominators `1/(E_n−E_m)^p` is assembled
//    *after* interpolation — the singularity structure is preserved,
//    not averaged away at the vertices.
//
// 4. **This section provides BZ *integrals*** — it yields `∫ dk …` for
//    the desired response.  The caller receives `∫ δ(ε−μ) Ω_n(k) dk`
//    (or the step‑function / Fermi‑Dirac weighted counterpart), NOT
//    per‑k‑point values.
//
// 5. **Quadrature rules** (initial implementation):
//    - 2D: degree‑2 symmetric triangle rule (3‑point).
//    - 3D: 4‑point degree‑2 tetrahedron rule.
//
// The primitive data computed below (`TetraKPoint`) is the foundation
// for all tetrahedron‑based Hall effect implementations (linear AHC,
// nonlinear extrinsic / intrinsic).


/// Per‑k‑point primitive data for tetrahedron quadrature.
///
/// All velocity quantities are in the band eigenbasis.  `k_ab` is
/// gauge‑invariant and safe to interpolate within a simplex.
///
/// | Field | Formula | Used for |
/// |-------|---------|----------|
/// | `k_ab` | `v^a_{nm} · v^b_{mn}` | Berry curvature Ω_{ab} |
/// | `vdiag` | `v^v_n = <n|v_v|n>` | energy derivative (dipoles) |
#[derive(Clone)]
pub struct TetraKPoint {
    /// Band energies `ε_n`, length `nsta`.
    pub band: Array1<f64>,
    /// Eigenvectors `U[:, n]`, shape `(nsta, nsta)` — for band tracking.
    pub evec: Array2<Complex<f64>>,
    /// `K^{ab}_{nm} = v^a_{nm} · v^b_{mn}`, shape `(nsta, nsta)`.
    pub k_ab: Array2<Complex<f64>>,
    /// Diagonal velocity `v^v_n = <n|v_v|n>`, length `nsta`.
    pub vdiag: Array1<f64>,
}

/// Compute per‑k‑point primitive data for tetrahedron quadrature.
///
/// # Arguments
/// * `k_vec` — single k‑point in fractional reciprocal coordinates.
/// * `dir_a` — Berry curvature first index α (length `DIM`).
/// * `dir_b` — Berry curvature second index β.
/// * `dir_v` — velocity / energy‑derivative direction (for dipoles).
/// * `gauge` — `Gauge::Atom` or `Gauge::Lattice`.
/// * `spin` — optional spin direction for spin‑current evaluation.
impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    pub fn compute_tetra_primitives(
        &self,
        k_vec: &Array1<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_v: &Array1<f64>,
        gauge: Gauge,
        spin: Option<SpinDirection>,
    ) -> TetraKPoint {
        let nsta = self.nsta();

        let directions = {
            let mut d = Array2::<f64>::zeros((3, DIM));
            d.row_mut(0).assign(dir_a);
            d.row_mut(1).assign(dir_b);
            d.row_mut(2).assign(dir_v);
            d
        };

        let (v_proj, hamk) = self.gen_v_projected(k_vec, gauge, &directions);
        let (band, evec) = hamk.eigh(UPLO::Lower).unwrap();
        // Convention: U^T · O · U^*  (matches berry_curvature_n_onek).
        // U^T = evec.t(),  U^* = evec.map(|x| x.conj()).
        let ut = evec.t();
        let uc = evec.map(|x| x.conj());

        let to_band = |d: usize, spin_dress: bool| -> Array2<Complex<f64>> {
            let v_raw = v_proj.slice(s![d, .., ..]).to_owned();
            if spin_dress && SPIN && spin.is_some() {
                let x = build_spin_matrix(self.norb(), spin);
                let s = anti_comm(&x, &v_raw) * 0.5;
                ut.dot(&s.dot(&uc))
            } else {
                ut.dot(&v_raw.dot(&uc))
            }
        };

        // Only dir_a (Berry curvature α index) gets spin‑dressed.
        // dir_b and dir_v are plain velocities — same convention as
        // berry_curvature_n_onek and berry_curvature_dipole_n_onek.
        let va = to_band(0, true);
        let vb = to_band(1, false);
        let vv = to_band(2, false);

        let vdiag = vv.diag().map(|x| x.re).to_owned();

        let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for n in 0..nsta {
            for m in 0..nsta {
                k_ab[[n, m]] = va[[n, m]] * vb[[m, n]];
            }
        }

        TetraKPoint {
            band,
            evec,
            k_ab,
            vdiag,
        }
    }
}

/// `Φ_η(x)` — the function whose 4th derivative is `1/(x²+η²)`.
#[inline]
fn phi(x: f64, eta: f64) -> f64 {
    if eta > 0.0 {
        let x_over_e = x / eta;
        (x.powi(3) / (6.0 * eta) - eta * x / 2.0) * x_over_e.atan()
            + (eta.powi(2) / 12.0 - x.powi(2) / 4.0) * (x.powi(2) + eta.powi(2)).ln()
    } else {
        if x.abs() < 1e-14 {
            0.0
        } else {
            -0.5 * x.powi(2) * x.abs().ln()
        }
    }
}

/// `Φ'_η(x)`.
#[inline]
fn phi_prime(x: f64, eta: f64) -> f64 {
    if eta > 0.0 {
        -eta / 2.0 * (x / eta).atan()
            - x / 2.0 * (x.powi(2) + eta.powi(2)).ln()
            - x / 3.0
            + x.powi(2) / (2.0 * eta) * (x / eta).atan()
    } else {
        if x.abs() < 1e-14 {
            0.0
        } else {
            -x * x.abs().ln() - x / 2.0
        }
    }
}

/// `Φ''_η(x)`.
#[inline]
fn phi_double_prime(x: f64, eta: f64) -> f64 {
    if eta > 0.0 {
        -0.5 * (x.powi(2) + eta.powi(2)).ln() - 5.0 / 6.0 + x / eta * (x / eta).atan()
    } else {
        if x.abs() < 1e-14 {
            -1.5  // regularized cutoff at x≈0 (true limit is +∞)
        } else {
            -x.abs().ln() - 1.5
        }
    }
}

/// `Φ'''_η(x)`.
#[inline]
fn phi_triple_prime(x: f64, eta: f64) -> f64 {
    if eta > 0.0 {
        (x / eta).atan() / eta
    } else {
        if x.abs() < 1e-14 {
            f64::NEG_INFINITY
        } else {
            -1.0 / x
        }
    }
}

/// Newton divided difference of `Φ_η` over nodes `xs` (must be non‑empty).
///
/// Repeated nodes are handled using the analytic derivatives above —
/// no numerical differentiation.
fn divided_diff_phi(xs: &[f64], eta: f64) -> f64 {
    let n = xs.len();
    debug_assert!(n > 0);

    if n == 1 {
        return phi(xs[0], eta);
    }

    // Sort (DD is symmetric in its arguments)
    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let x_min = sorted[0];
    let x_max = sorted[n - 1];

    // All nodes equal (within tolerance): use the (n−1)‑th derivative
    let tol = 1e-12 * (1.0 + x_max.abs().max(x_min.abs()));
    if x_max - x_min < tol {
        let x_mid = (x_min + x_max) / 2.0;
        let p = n - 1;
        // Φ^{(p)}(x) / p!
        let deriv = match p {
            0 => phi(x_mid, eta),
            1 => phi_prime(x_mid, eta),
            2 => phi_double_prime(x_mid, eta),
            3 => phi_triple_prime(x_mid, eta),
            4 => {
                // 1/(x²+η²)
                let denom = x_mid.powi(2) + eta.powi(2);
                if denom < 1e-30 { f64::INFINITY } else { 1.0 / denom }
            }
            _ => panic!("divided_diff_phi: p={p} not supported"),
        };
        // Divide by p!
        let fact: f64 = match p {
            0 => 1.0,
            1 => 1.0,
            2 => 2.0,
            3 => 6.0,
            4 => 24.0,
            _ => (1..=p).product::<usize>() as f64,
        };
        return deriv / fact;
    }

    // Standard recursive formula
    let a = divided_diff_phi(&sorted[1..], eta);
    let b = divided_diff_phi(&sorted[..n - 1], eta);
    (a - b) / (x_max - x_min)
}

/// Compute vertex weight vector `W_r` (r=0..3) for the analytic
/// tetrahedron Berry‑curvature integral.
///
/// `d[4]` — energy differences `E_n−E_m` at the 4 vertices.
///
/// Returns `[w0, w1, w2, w3]` where `w_r = 6 · Φ_η[d₀,…,d_r,d_r,…,d₃]`.
fn compute_weights_omega(d: &[f64; 4], eta: f64) -> [f64; 4] {
    let mut w = [0.0f64; 4];
    for r in 0..4 {
        let mut nodes = vec![d[0], d[1], d[2], d[3]];
        nodes.insert(r, d[r]); // duplicate d_r at position r → 5 nodes
        w[r] = 6.0 * divided_diff_phi(&nodes, eta);
    }
    w
}

/// Check whether `d(k)` crosses zero inside the tetrahedron.
///
/// Uses a small tolerance so that d=0 exactly at a vertex/edge/face
/// is also flagged (dangerous at `η=0`).
#[inline]
fn gap_crosses_zero(d: &[f64; 4], tol: f64) -> bool {
    let dmin = d.iter().cloned().fold(f64::INFINITY, f64::min);
    let dmax = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    dmin <= tol && dmax >= -tol
}

// ── Hall conductivity via tetrahedron integration ─────────────────────
//
// Blochl sub‑tetrahedron decomposition for T=0, thermal convolution for T>0.

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Hall conductivity using analytic tetrahedron integration.
    ///
    /// T=0: Blochl sub‑tet decomposition for partial occupation.
    /// T>0: thermal convolution of the T=0 result.
    pub fn Hall_conductivity_tetra(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        let dim = k_mesh.len();
        assert!(dim == 2 || dim == 3);

        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let nsta = self.nsta();
        let gauge = Gauge::Atom;
        let dir_v = Array1::<f64>::zeros(DIM);

        let all_pts: Vec<TetraKPoint> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_tetra_primitives(
                    &kv, current_dir, dir_2, &dir_v, gauge, spin,
                )
            })
            .collect();

        let all_band: Array2<f64> = {
            let mut a = Array2::zeros((nk, nsta));
            for ik in 0..nk { a.row_mut(ik).assign(&all_pts[ik].band); }
            a
        };

        let n_mu = mu.len();
        let det = self.lat.det().unwrap();

        let mut result_t0 = Array1::<f64>::zeros(n_mu);

        match dim {
            2 => {
                let (nx, ny) = (k_mesh[0], k_mesh[1]);
                let ncell = nx * ny;
                result_t0 = (0..ncell)
                    .into_par_iter()
                    .fold(
                        || Array1::<f64>::zeros(n_mu),
                        |mut acc, cell_id| {
                            let ix = cell_id / ny;
                            let iy = cell_id % ny;
                            accum_hall_cell_2d(
                                ix, iy, nx, ny, nsta, n_mu, eta,
                                mu, &all_band, &all_pts, &mut acc,
                            );
                            acc
                        },
                    )
                    .reduce(
                        || Array1::<f64>::zeros(n_mu),
                        |a, b| a + b,
                    );
            }
            3 => {
                let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
                let ncell = nx * ny * nz;
                result_t0 = (0..ncell)
                    .into_par_iter()
                    .fold(
                        || Array1::<f64>::zeros(n_mu),
                        |mut acc, cell_id| {
                            let ix = cell_id / (ny * nz);
                            let rem = cell_id % (ny * nz);
                            let iy = rem / nz;
                            let iz = rem % nz;
                            accum_hall_cell_3d(
                                ix, iy, iz, nx, ny, nz,
                                nsta, n_mu, eta, mu, &all_band, &all_pts, &mut acc,
                            );
                            acc
                        },
                    )
                    .reduce(
                        || Array1::<f64>::zeros(n_mu),
                        |a, b| a + b,
                    );
            }
            _ => unreachable!(),
        }

        let raw = result_t0 / det;

        if T > 0.0 && n_mu > 1 {
            // T>0: thermal convolution. Requires strictly increasing, uniformly
            // spaced mu.  Boundary truncation (|x| > 50) assumes the mu window
            // is wide enough to capture the full thermal kernel; a narrow
            // window loses normalization.
            let mu_slice = mu.as_slice().unwrap();
            if mu_slice.windows(2).any(|w| w[1] <= w[0]) {
                return Err(TbError::Other(
                    "mu must be strictly increasing for thermal convolution".into(),
                ));
            }
            let dmu = mu[1] - mu[0];
            let tol = 1e-12 * (1.0 + dmu.abs());
            if mu_slice.windows(2).any(|w| (w[1] - w[0] - dmu).abs() > tol) {
                return Err(TbError::Other(
                    "mu must have uniform spacing for thermal convolution".into(),
                ));
            }
            let beta = 1.0 / (T * 8.617e-5);
            let mut conv = Array1::<f64>::zeros(n_mu);
            for i in 0..n_mu {
                let mut s = 0.0;
                for j in 0..n_mu {
                    let x = beta * (mu[[j]] - mu[[i]]);
                    if x.abs() > 50.0 { continue; }
                    let f = 1.0 / (x.exp() + 1.0);
                    s += raw[[j]] * beta * f * (1.0 - f) * dmu;
                }
                conv[[i]] = s;
            }
            Ok(conv)
        } else {
            Ok(raw)
        }
    }
}

// ── helpers for Hall_conductivity_tetra ────────────────────────────────

#[inline]
fn sort_by_energy(e: &[f64; 4]) -> [usize; 4] {
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap());
    idx
}

#[inline]
fn interp_t(mu: f64, ei: f64, ej: f64) -> f64 {
    let d = ej - ei;
    if d.abs() < 1e-14 { 0.5 } else { (mu - ei) / d }
}

#[inline]
fn full_tet_one_pair(p: &[f64; 4], d: &[f64; 4], vt: f64, eta: f64) -> f64 {
    if eta == 0.0 && gap_crosses_zero(d, 1e-12) { return 0.0; }
    let w = compute_weights_omega(d, eta);
    vt * (p[0]*w[0] + p[1]*w[1] + p[2]*w[2] + p[3]*w[3])
}

fn sub_tet_omega(
    all_pts: &[TetraKPoint], corners: &[usize; 4], srt: &[usize; 4],
    n: usize, nsta: usize, v_sub: f64, eta: f64,
    verts: &[(usize, usize, Option<f64>)],
) -> f64 {
    let mut total = 0.0;
    for m in 0..nsta {
        if m == n { continue; }
        let mut p_sub = [0.0f64; 4];
        let mut d_sub = [0.0f64; 4];
        for (vi, &(from, to, t)) in verts.iter().enumerate() {
            let from_pt = corners[srt[from]];
            if let Some(tv) = t {
                let to_pt = corners[srt[to]];
                let p_from = all_pts[from_pt].k_ab[[n,m]].im;
                let p_to = all_pts[to_pt].k_ab[[n,m]].im;
                let d_from = all_pts[from_pt].band[[n]] - all_pts[from_pt].band[[m]];
                let d_to = all_pts[to_pt].band[[n]] - all_pts[to_pt].band[[m]];
                p_sub[vi] = p_from + tv * (p_to - p_from);
                d_sub[vi] = d_from + tv * (d_to - d_from);
            } else {
                p_sub[vi] = all_pts[from_pt].k_ab[[n,m]].im;
                d_sub[vi] = all_pts[from_pt].band[[n]] - all_pts[from_pt].band[[m]];
            }
        }
        total += full_tet_one_pair(&p_sub, &d_sub, v_sub, eta);
    }
    -2.0 * total
}

fn compute_occ_omega(
    all_pts: &[TetraKPoint], corners: &[usize; 4],
    n: usize, nsta: usize, vt: f64, eta: f64,
    e_unsorted: &[f64; 4], srt: &[usize; 4], mu: f64, fully_occ: bool,
) -> f64 {
    let es = [e_unsorted[srt[0]], e_unsorted[srt[1]], e_unsorted[srt[2]], e_unsorted[srt[3]]];

    if fully_occ || mu >= es[3] {
        let mut total = 0.0;
        for m in 0..nsta {
            if m == n { continue; }
            let p = [
                all_pts[corners[0]].k_ab[[n,m]].im, all_pts[corners[1]].k_ab[[n,m]].im,
                all_pts[corners[2]].k_ab[[n,m]].im, all_pts[corners[3]].k_ab[[n,m]].im,
            ];
            let d = [
                e_unsorted[0]-all_pts[corners[0]].band[[m]], e_unsorted[1]-all_pts[corners[1]].band[[m]],
                e_unsorted[2]-all_pts[corners[2]].band[[m]], e_unsorted[3]-all_pts[corners[3]].band[[m]],
            ];
            total += full_tet_one_pair(&p, &d, vt, eta);
        }
        return -2.0 * total;
    }
    if mu <= es[0] { return 0.0; }

    if mu < es[1] {
        let t01 = interp_t(mu, es[0], es[1]);
        let t02 = interp_t(mu, es[0], es[2]);
        let t03 = interp_t(mu, es[0], es[3]);
        sub_tet_omega(all_pts, corners, srt, n, nsta, vt*t01*t02*t03, eta,
            &[(0,0,None),(0,1,Some(t01)),(0,2,Some(t02)),(0,3,Some(t03))])
    } else if mu < es[2] {
        let t02 = interp_t(mu, es[0], es[2]);
        let t03 = interp_t(mu, es[0], es[3]);
        let t12 = interp_t(mu, es[1], es[2]);
        let t13 = interp_t(mu, es[1], es[3]);
        let v1 = t02 * t03;
        let v2 = t12 * t03 * (1.0 - t02);
        let v3 = t12 * t13 * (1.0 - t03);
        sub_tet_omega(all_pts,corners,srt,n,nsta,vt*v1,eta,&[(0,0,None),(1,1,None),(0,2,Some(t02)),(0,3,Some(t03))])
        + sub_tet_omega(all_pts,corners,srt,n,nsta,vt*v2,eta,&[(1,1,None),(0,2,Some(t02)),(1,2,Some(t12)),(0,3,Some(t03))])
        + sub_tet_omega(all_pts,corners,srt,n,nsta,vt*v3,eta,&[(1,1,None),(1,2,Some(t12)),(0,3,Some(t03)),(1,3,Some(t13))])
    } else {
        let full = {
            let mut t = 0.0;
            for m in 0..nsta {
                if m == n { continue; }
                let p = [
                    all_pts[corners[0]].k_ab[[n,m]].im, all_pts[corners[1]].k_ab[[n,m]].im,
                    all_pts[corners[2]].k_ab[[n,m]].im, all_pts[corners[3]].k_ab[[n,m]].im,
                ];
                let d = [
                    e_unsorted[0]-all_pts[corners[0]].band[[m]], e_unsorted[1]-all_pts[corners[1]].band[[m]],
                    e_unsorted[2]-all_pts[corners[2]].band[[m]], e_unsorted[3]-all_pts[corners[3]].band[[m]],
                ];
                t += full_tet_one_pair(&p, &d, vt, eta);
            }
            -2.0 * t
        };
        let t03 = interp_t(mu, es[0], es[3]);
        let t13 = interp_t(mu, es[1], es[3]);
        let t23 = interp_t(mu, es[2], es[3]);
        let cv = (1.0-t03)*(1.0-t13)*(1.0-t23);
        let cap = sub_tet_omega(all_pts,corners,srt,n,nsta,vt*cv,eta,
            &[(3,3,None),(0,3,Some(t03)),(1,3,Some(t13)),(2,3,Some(t23))]);
        full - cap
    }
}

// ── Extrinsic NLH via tetrahedron Fermi-surface integration ──────────

/// Lightweight vertex data for a Fermi-surface triangle cut vertex.
struct CutVertex {
    u: f64,       // diagonal velocity v^c_n
    p: f64,       // Im K^{ab}_{nm}
    d: f64,       // E_n - E_m
    k: [f64; 3],  // fractional k-space coordinate
}

/// Compute |∇_k e| from tetrahedron vertex positions and band energies.
/// Uses Cramer's rule on the 3×3 system: (r_{i+1}−r_0)·g = e_{i+1}−e_0.
/// Returns 0.0 for degenerate (flat) tetrahedra.
fn grad_linear_tet_3d(coords: &[[f64; 3]; 4], e: &[f64; 4]) -> f64 {
    let m = [
        [coords[1][0]-coords[0][0], coords[1][1]-coords[0][1], coords[1][2]-coords[0][2]],
        [coords[2][0]-coords[0][0], coords[2][1]-coords[0][1], coords[2][2]-coords[0][2]],
        [coords[3][0]-coords[0][0], coords[3][1]-coords[0][1], coords[3][2]-coords[0][2]],
    ];
    let b = [e[1]-e[0], e[2]-e[0], e[3]-e[0]];
    let det = m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    if det.abs() < 1e-30 { return 0.0; }
    let g0 = (b[0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            - m[0][1]*(b[1]*m[2][2]-m[1][2]*b[2])
            + m[0][2]*(b[1]*m[2][1]-m[1][1]*b[2])) / det;
    let g1 = (m[0][0]*(b[1]*m[2][2]-m[1][2]*b[2])
            - b[0]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            + m[0][2]*(m[1][0]*b[2]-b[1]*m[2][0])) / det;
    let g2 = (m[0][0]*(m[1][1]*b[2]-b[1]*m[2][1])
            - m[0][1]*(m[1][0]*b[2]-b[1]*m[2][0])
            + b[0]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])) / det;
    (g0*g0 + g1*g1 + g2*g2).sqrt()
}

/// K_αβ weight for the triangle surface integral.
///
/// ```text
/// K_αβ =     Φ_η[d₀,d₁,d₂,dα,dβ]   if α≠β  (5‑node DD)
/// K_αβ = 2 · Φ_η[d₀,d₁,d₂,dα,dα]   if α=β
/// ```
/// where Φ_η''''(x) = 1/(x²+η²).
fn k_weight_triangle(alpha: usize, beta: usize, d: &[f64; 3], eta: f64) -> f64 {
    let mut nodes = vec![d[0], d[1], d[2], d[alpha], d[beta]];
    let dd = divided_diff_phi(&nodes, eta);
    if alpha == beta { 2.0 * dd } else { dd }
}

/// Analytic surface integral over one triangle:
///
/// ```text
/// ∫_Δ dS  u(q) P(q) / (d(q)²+η²)  =  2 A_Δ · Σ_{α,β} u_α P_β K_αβ
/// ```
///
/// This function returns `Σ u_α P_β K_αβ` (without the 2·A_Δ prefactor).
fn triangle_surface_integral(u: &[f64; 3], p: &[f64; 3], d: &[f64; 3], eta: f64) -> f64 {
    let mut sum = 0.0;
    for alpha in 0..3 {
        for beta in 0..3 {
            sum += u[alpha] * p[beta] * k_weight_triangle(alpha, beta, d, eta);
        }
    }
    sum
}

/// Triangle area in 3D.
fn triangle_area_3d(q: &[[f64; 3]; 3]) -> f64 {
    let v1 = [q[1][0]-q[0][0], q[1][1]-q[0][1], q[1][2]-q[0][2]];
    let v2 = [q[2][0]-q[0][0], q[2][1]-q[0][1], q[2][2]-q[0][2]];
    let cx = v1[1]*v2[2] - v1[2]*v2[1];
    let cy = v1[2]*v2[0] - v1[0]*v2[2];
    let cz = v1[0]*v2[1] - v1[1]*v2[0];
    0.5 * (cx*cx + cy*cy + cz*cz).sqrt()
}

/// Sort indices by 3 energy values.
#[inline]
fn sort3_by_energy(e: &[f64; 3]) -> [usize; 3] {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap());
    idx
}

/// Compute |∇_k e| for a 2D triangle.
fn grad_linear_tri_2d(coords: &[[f64; 2]; 3], e: &[f64; 3]) -> f64 {
    let dx1 = coords[1][0] - coords[0][0];
    let dy1 = coords[1][1] - coords[0][1];
    let dx2 = coords[2][0] - coords[0][0];
    let dy2 = coords[2][1] - coords[0][1];
    let det = dx1 * dy2 - dy1 * dx2;
    if det.abs() < 1e-30 { return 0.0; }
    let de1 = e[1] - e[0];
    let de2 = e[2] - e[0];
    let gx = (de1 * dy2 - dy1 * de2) / det;
    let gy = (dx1 * de2 - de1 * dx2) / det;
    (gx * gx + gy * gy).sqrt()
}

/// Analytic 1D integral ∫₀¹ u(ξ)P(ξ)/(d(ξ)²+η²) dξ along a segment.
///
/// u(ξ) = (1−ξ)u₀ + ξu₁,  P(ξ) = (1−ξ)P₀ + ξP₁,
/// d(ξ) = (1−ξ)d₀ + ξd₁ = d₀ + ξ(d₁−d₀).
///
/// Uses separate analytic branches for η>0 (atan/log) and η=0 (ln, 1/d).
/// Caller must ensure d₀,d₁ have the same sign when η=0.
fn segment_integral_1d(
    u0: f64, u1: f64, p0: f64, p1: f64, d0: f64, d1: f64, eta: f64,
) -> f64 {
    let dd = d1 - d0;
    if dd.abs() < 1e-14 {
        let denom = d0 * d0 + eta * eta;
        // ∫₀¹ ((1-ξ)u₀+ξu₁)((1-ξ)P₀+ξP₁) dξ = (2u₀P₀+u₀P₁+u₁P₀+2u₁P₁)/6
        return (2.0 * u0 * p0 + u0 * p1 + u1 * p0 + 2.0 * u1 * p1) / (6.0 * denom);
    }
    let a = (u1 - u0) * (p1 - p0);
    let b = u0 * p1 + u1 * p0 - 2.0 * u0 * p0;
    let c = u0 * p0;

    if eta == 0.0 {
        // d₀,d₁ same-sign guaranteed by caller (zero-crossing pre-filtered).
        // ψ₀ = ∫₀¹ 1/(d₀+ξΔ)² dξ = 1/(d₀d₁)
        let psi0 = 1.0 / (d0 * d1);
        // ψ₁ = (ln|d₁/d₀| + d₀/d₁ − 1) / Δ²
        let ln_ratio = (d1 / d0).abs().ln();
        let psi1 = (ln_ratio + d0 / d1 - 1.0) / (dd * dd);
        // ψ₂ = (Δ − 2d₀ ln|d₁/d₀| − d₀²(1/d₁ − 1/d₀)) / Δ³
        let psi2 = (dd - 2.0 * d0 * ln_ratio - d0 * d0 * (1.0 / d1 - 1.0 / d0))
            / (dd * dd * dd);
        return a * psi2 + b * psi1 + c * psi0;
    }

    // η > 0
    let at0 = (d0 / eta).atan();
    let at1 = (d1 / eta).atan();
    let dat = at1 - at0;
    let psi0 = dat / (eta * dd);

    let ln0 = (d0 * d0 + eta * eta).ln();
    let ln1 = (d1 * d1 + eta * eta).ln();
    let dln = ln1 - ln0;
    let psi1 = dln / (2.0 * dd * dd) - d0 * psi0 / dd;

    // ψ₂: Antiderivative F(y) = y − d₀ln(y²+η²) + (d₀²/η−η)atan(y/η)
    let coeff = d0 * d0 / eta - eta;
    let f1 = d1 - d0 * ln1 + coeff * at1;
    let f0 = d0 - d0 * ln0 + coeff * at0;
    let psi2 = (f1 - f0) / (dd * dd * dd);

    a * psi2 + b * psi1 + c * psi0
}

/// Segment length in 2D fractional k-space.
fn segment_length_2d(q0: &[f64; 2], q1: &[f64; 2]) -> f64 {
    let dx = q1[0] - q0[0];
    let dy = q1[1] - q0[1];
    (dx * dx + dy * dy).sqrt()
}

/// Cut a 2D triangle (3 vertices, sorted energies e0≤e1≤e2) at energy μ.
/// Returns 0 or 1 segment as two edge-interpolation specs.
fn cut_tri_fermi_contour_2d(
    e_sorted: &[f64; 3], mu: f64,
) -> Option<[(usize, usize, f64); 2]> {
    if mu <= e_sorted[0] || mu >= e_sorted[2] { return None; }
    if mu < e_sorted[1] {
        let t01 = (mu - e_sorted[0]) / (e_sorted[1] - e_sorted[0]);
        let t02 = (mu - e_sorted[0]) / (e_sorted[2] - e_sorted[0]);
        Some([(0, 1, t01), (0, 2, t02)])
    } else {
        let t02 = (mu - e_sorted[0]) / (e_sorted[2] - e_sorted[0]);
        let t12 = (mu - e_sorted[1]) / (e_sorted[2] - e_sorted[1]);
        Some([(0, 2, t02), (1, 2, t12)])
    }
}

/// Interpolate a scalar value at edge intersection.
#[inline]
fn interp_scalar(vals: &[f64], srt: &[usize], i: usize, j: usize, t: f64) -> f64 {
    let vi = vals[srt[i]];
    let vj = vals[srt[j]];
    vi + t * (vj - vi)
}

/// Interpolate a 2D coordinate at edge intersection.
#[inline]
fn interp_coord_2d(coords: &[[f64; 2]], srt: &[usize], i: usize, j: usize, t: f64) -> [f64; 2] {
    let ci = coords[srt[i]];
    let cj = coords[srt[j]];
    [ci[0] + t * (cj[0] - ci[0]), ci[1] + t * (cj[1] - ci[1])]
}

/// Sort indices by 4 energy values.
#[inline]
fn sort4_by_energy(e: &[f64; 4]) -> [usize; 4] {
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap());
    idx
}

/// Edge interpolation specification: (sorted_vertex_from, sorted_vertex_to, t).
type EdgeSpec = (usize, usize, f64);

/// Cut a tetrahedron by the energy plane e(k)=μ.
/// Returns a list of triangles, each as 3 edge specs.
/// Does NOT depend on per-pair quantities (u, P, d) — only on band energies.
fn cut_tet_fermi_surface_edges(
    e_sorted: &[f64; 4], mu: f64,
) -> Vec<[EdgeSpec; 3]> {
    let mut tris = Vec::new();
    if mu <= e_sorted[0] || mu >= e_sorted[3] { return tris; }

    if mu < e_sorted[1] {
        let t01 = (mu - e_sorted[0]) / (e_sorted[1] - e_sorted[0]);
        let t02 = (mu - e_sorted[0]) / (e_sorted[2] - e_sorted[0]);
        let t03 = (mu - e_sorted[0]) / (e_sorted[3] - e_sorted[0]);
        tris.push([(0,1,t01), (0,2,t02), (0,3,t03)]);
    } else if mu < e_sorted[2] {
        let t02 = (mu - e_sorted[0]) / (e_sorted[2] - e_sorted[0]);
        let t03 = (mu - e_sorted[0]) / (e_sorted[3] - e_sorted[0]);
        let t12 = (mu - e_sorted[1]) / (e_sorted[2] - e_sorted[1]);
        let t13 = (mu - e_sorted[1]) / (e_sorted[3] - e_sorted[1]);
        tris.push([(0,2,t02), (1,2,t12), (0,3,t03)]);
        tris.push([(1,2,t12), (1,3,t13), (0,3,t03)]);
    } else {
        let t03 = (mu - e_sorted[0]) / (e_sorted[3] - e_sorted[0]);
        let t13 = (mu - e_sorted[1]) / (e_sorted[3] - e_sorted[1]);
        let t23 = (mu - e_sorted[2]) / (e_sorted[3] - e_sorted[2]);
        tris.push([(0,3,t03), (1,3,t13), (2,3,t23)]);
    }
    tris
}

/// Build a triangle of CutVertex from edge specs and per-vertex data.
fn make_cut_tri_from_specs(
    coords: &[[f64; 3]; 4], u: &[f64; 4], p: &[f64; 4], d: &[f64; 4],
    srt: &[usize; 4], edges: &[EdgeSpec; 3],
) -> [CutVertex; 3] {
    let cv = |i: usize, j: usize, t: f64| -> CutVertex {
        let ii = srt[i]; let jj = srt[j];
        CutVertex {
            u: u[ii] + t * (u[jj] - u[ii]),
            p: p[ii] + t * (p[jj] - p[ii]),
            d: d[ii] + t * (d[jj] - d[ii]),
            k: [
                coords[ii][0] + t * (coords[jj][0] - coords[ii][0]),
                coords[ii][1] + t * (coords[jj][1] - coords[ii][1]),
                coords[ii][2] + t * (coords[jj][2] - coords[ii][2]),
            ],
        }
    };
    [cv(edges[0].0,edges[0].1,edges[0].2), cv(edges[1].0,edges[1].1,edges[1].2), cv(edges[2].0,edges[2].1,edges[2].2)]
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Extrinsic nonlinear Hall conductivity via tetrahedron Fermi‑surface
    /// integration.
    ///
    /// ```text
    /// σ^{abc}(μ,T) = Σ_n ∫_BZ (-∂f/∂E_n) v^c_n(k) Ω^{ab}_n(k) d³k
    /// ```
    ///
    /// **2D**: each rectangular cell is split into two triangles; the μ‑plane
    /// intersects each triangle in 0 or 1 line segment.  The 1D integral
    /// of u·P/(d²+η²) along the segment is evaluated analytically
    /// (elementary antiderivatives for η>0, ln/1/d for η=0).
    ///
    /// **3D**: each cubic cell is split into 5 tetrahedra; the μ‑plane
    /// intersection of each tetrahedron is cut into 0–2 triangles.  The
    /// 2D surface integral uses divided‑difference triangle weights.
    ///
    /// **T>0**: thermal convolution of the T=0 result.
    ///
    /// `current_dir` = a, `dir_2` = b for Ω^{ab}, `dir_c` = c for v^c.
    pub fn Nonlinear_Hall_conductivity_Extrinsic_tetra(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_c: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        let dim = k_mesh.len();
        assert!(dim == 2 || dim == 3, "only dim=2,3 supported, got dim={dim}");

        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let nsta = self.nsta();
        let gauge = Gauge::Atom;

        // Compute TetraKPoint at every k‑point (dir_v = dir_c for vdiag)
        let all_pts: Vec<TetraKPoint> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_tetra_primitives(
                    &kv, current_dir, dir_2, dir_c, gauge, spin,
                )
            })
            .collect();

        let n_mu = mu.len();
        let det = self.lat.det().unwrap();
        let mut result_t0 = Array1::<f64>::zeros(n_mu);

        match dim {
            2 => {
                let (nx, ny) = (k_mesh[0], k_mesh[1]);
                let inv_nx = 1.0 / nx as f64;
                let inv_ny = 1.0 / ny as f64;
                let ncell = nx * ny;

                result_t0 = (0..ncell)
                    .into_par_iter()
                    .fold(
                        || Array1::<f64>::zeros(n_mu),
                        |mut acc, cell_id| {
                            let ix = cell_id / ny;
                            let iy = cell_id % ny;
                            accum_extrinsic_cell_2d(
                                ix, iy, nx, ny, inv_nx, inv_ny,
                                nsta, n_mu, eta, mu, &all_pts, &mut acc,
                            );
                            acc
                        },
                    )
                    .reduce(
                        || Array1::<f64>::zeros(n_mu),
                        |a, b| a + b,
                    );
            }
            3 => {
                let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
                let inv_nx = 1.0 / nx as f64;
                let inv_ny = 1.0 / ny as f64;
                let inv_nz = 1.0 / nz as f64;
                let ncell = nx * ny * nz;

                result_t0 = (0..ncell)
                    .into_par_iter()
                    .fold(
                        || Array1::<f64>::zeros(n_mu),
                        |mut acc, cell_id| {
                            let ix = cell_id / (ny * nz);
                            let rem = cell_id % (ny * nz);
                            let iy = rem / nz;
                            let iz = rem % nz;
                            accum_extrinsic_cell_3d(
                                ix, iy, iz, nx, ny, nz,
                                inv_nx, inv_ny, inv_nz,
                                nsta, n_mu, eta, mu, &all_pts, &mut acc,
                            );
                            acc
                        },
                    )
                    .reduce(
                        || Array1::<f64>::zeros(n_mu),
                        |a, b| a + b,
                    );
            }
            _ => unreachable!(),
        }

        let raw = result_t0 / det;

        if T > 0.0 && n_mu > 1 {
            let mu_slice = mu.as_slice().unwrap();
            if mu_slice.windows(2).any(|w| w[1] <= w[0]) {
                return Err(TbError::Other(
                    "mu must be strictly increasing for thermal convolution".into(),
                ));
            }
            let dmu = mu[1] - mu[0];
            let tol = 1e-12 * (1.0 + dmu.abs());
            if mu_slice.windows(2).any(|w| (w[1] - w[0] - dmu).abs() > tol) {
                return Err(TbError::Other(
                    "mu must have uniform spacing for thermal convolution".into(),
                ));
            }
            let beta = 1.0 / (T * 8.617e-5);
            let mut conv = Array1::<f64>::zeros(n_mu);
            for i in 0..n_mu {
                let mut s = 0.0;
                for j in 0..n_mu {
                    let x = beta * (mu[[j]] - mu[[i]]);
                    if x.abs() > 50.0 { continue; }
                    let f = 1.0 / (x.exp() + 1.0);
                    s += raw[[j]] * beta * f * (1.0 - f) * dmu;
                }
                conv[[i]] = s;
            }
            Ok(conv)
        } else {
            Ok(raw)
        }
    }
}
// ── Parallel cell accumulation helpers ──────────────────────────────────

/// Thread-local accumulator for one 2D cell (extrinsic NLH).
fn accum_extrinsic_cell_2d(
    ix: usize, iy: usize, nx: usize, ny: usize,
    inv_nx: f64, inv_ny: f64,
    nsta: usize, n_mu: usize, eta: f64,
    mu: &Array1<f64>, all_pts: &[TetraKPoint],
    acc: &mut Array1<f64>,
) {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;

    let corners_frac: [[f64; 2]; 4] = [
        [ix as f64 * inv_nx, iy as f64 * inv_ny],
        [(ix + 1) as f64 * inv_nx, iy as f64 * inv_ny],
        [(ix + 1) as f64 * inv_nx, (iy + 1) as f64 * inv_ny],
        [ix as f64 * inv_nx, (iy + 1) as f64 * inv_ny],
    ];
    let coord_of = |idx: usize| -> [f64; 2] {
        if idx == i00 { corners_frac[0] }
        else if idx == i10 { corners_frac[1] }
        else if idx == i11 { corners_frac[2] }
        else { corners_frac[3] }
    };

    for &(v0, v1, v2) in &[(i00, i10, i01), (i11, i10, i01)] {
        let cr = [v0, v1, v2];
        let tri_coords = [coord_of(v0), coord_of(v1), coord_of(v2)];

        for n in 0..nsta {
            let e_raw: [f64; 3] = [
                all_pts[cr[0]].band[[n]], all_pts[cr[1]].band[[n]],
                all_pts[cr[2]].band[[n]],
            ];
            let srt = sort3_by_energy(&e_raw);
            let es = [e_raw[srt[0]], e_raw[srt[1]], e_raw[srt[2]]];
            let grad = grad_linear_tri_2d(&tri_coords, &e_raw);
            if grad < 1e-14 { continue; }

            let udata: [f64; 3] = [
                all_pts[cr[0]].vdiag[[n]], all_pts[cr[1]].vdiag[[n]],
                all_pts[cr[2]].vdiag[[n]],
            ];

            for im in 0..n_mu {
                let mu_val = mu[[im]];
                if mu_val <= es[0] || mu_val >= es[2] { continue; }
                let seg = match cut_tri_fermi_contour_2d(&es, mu_val) {
                    Some(s) => s, None => continue,
                };
                let (i0, j0, t0) = seg[0];
                let (i1, j1, t1) = seg[1];
                let q0 = interp_coord_2d(&tri_coords, &srt, i0, j0, t0);
                let q1 = interp_coord_2d(&tri_coords, &srt, i1, j1, t1);
                let seg_len = segment_length_2d(&q0, &q1);
                let u0 = interp_scalar(&udata, &srt, i0, j0, t0);
                let u1 = interp_scalar(&udata, &srt, i1, j1, t1);

                for m in 0..nsta {
                    if m == n { continue; }
                    let pdata: [f64; 3] = [
                        all_pts[cr[0]].k_ab[[n, m]].im,
                        all_pts[cr[1]].k_ab[[n, m]].im,
                        all_pts[cr[2]].k_ab[[n, m]].im,
                    ];
                    let ddata: [f64; 3] = [
                        all_pts[cr[0]].band[[n]] - all_pts[cr[0]].band[[m]],
                        all_pts[cr[1]].band[[n]] - all_pts[cr[1]].band[[m]],
                        all_pts[cr[2]].band[[n]] - all_pts[cr[2]].band[[m]],
                    ];
                    let p0 = interp_scalar(&pdata, &srt, i0, j0, t0);
                    let p1 = interp_scalar(&pdata, &srt, i1, j1, t1);
                    let d0 = interp_scalar(&ddata, &srt, i0, j0, t0);
                    let d1 = interp_scalar(&ddata, &srt, i1, j1, t1);
                    if eta == 0.0 {
                        let dmin = d0.min(d1);
                        let dmax = d0.max(d1);
                        if dmin <= 1e-12 && dmax >= -1e-12 { continue; }
                    }
                    let seg_int = segment_integral_1d(u0, u1, p0, p1, d0, d1, eta);
                    acc[[im]] += -2.0 * seg_len * seg_int / grad;
                }
            }
        }
    }
}

/// Thread-local accumulator for one 3D cell (extrinsic NLH).
fn accum_extrinsic_cell_3d(
    ix: usize, iy: usize, iz: usize,
    nx: usize, ny: usize, nz: usize,
    inv_nx: f64, inv_ny: f64, inv_nz: f64,
    nsta: usize, n_mu: usize, eta: f64,
    mu: &Array1<f64>, all_pts: &[TetraKPoint],
    acc: &mut Array1<f64>,
) {
    const TETS: [[usize; 4]; 5] = [
        [0, 1, 2, 4], [3, 1, 2, 7], [5, 1, 4, 7], [6, 2, 4, 7], [1, 2, 4, 7],
    ];
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let izp = (iz + 1) % nz;
    let idx3 = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let c = [
        idx3(ix, iy, iz), idx3(ixp, iy, iz),
        idx3(ix, iyp, iz), idx3(ixp, iyp, iz),
        idx3(ix, iy, izp), idx3(ixp, iy, izp),
        idx3(ix, iyp, izp), idx3(ixp, iyp, izp),
    ];
    let corners_frac: [[f64; 3]; 8] = [
        [ix as f64 * inv_nx, iy as f64 * inv_ny, iz as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, iy as f64 * inv_ny, iz as f64 * inv_nz],
        [ix as f64 * inv_nx, (iy + 1) as f64 * inv_ny, iz as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, (iy + 1) as f64 * inv_ny, iz as f64 * inv_nz],
        [ix as f64 * inv_nx, iy as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, iy as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
        [ix as f64 * inv_nx, (iy + 1) as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, (iy + 1) as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
    ];

    for &[v0, v1, v2, v3] in TETS.iter() {
        let cr = [c[v0], c[v1], c[v2], c[v3]];
        let coords = [corners_frac[v0], corners_frac[v1], corners_frac[v2], corners_frac[v3]];

        for n in 0..nsta {
            let e_raw: [f64; 4] = [
                all_pts[cr[0]].band[[n]], all_pts[cr[1]].band[[n]],
                all_pts[cr[2]].band[[n]], all_pts[cr[3]].band[[n]],
            ];
            let srt = sort4_by_energy(&e_raw);
            let es = [e_raw[srt[0]], e_raw[srt[1]], e_raw[srt[2]], e_raw[srt[3]]];
            let grad = grad_linear_tet_3d(&coords, &e_raw);
            if grad < 1e-14 { continue; }

            let udata: [f64; 4] = [
                all_pts[cr[0]].vdiag[[n]], all_pts[cr[1]].vdiag[[n]],
                all_pts[cr[2]].vdiag[[n]], all_pts[cr[3]].vdiag[[n]],
            ];

            for im in 0..n_mu {
                let mu_val = mu[[im]];
                if mu_val <= es[0] || mu_val >= es[3] { continue; }
                let edge_tris = cut_tet_fermi_surface_edges(&es, mu_val);
                if edge_tris.is_empty() { continue; }

                for m in 0..nsta {
                    if m == n { continue; }
                    let pdata: [f64; 4] = [
                        all_pts[cr[0]].k_ab[[n, m]].im, all_pts[cr[1]].k_ab[[n, m]].im,
                        all_pts[cr[2]].k_ab[[n, m]].im, all_pts[cr[3]].k_ab[[n, m]].im,
                    ];
                    let ddata: [f64; 4] = [
                        all_pts[cr[0]].band[[n]] - all_pts[cr[0]].band[[m]],
                        all_pts[cr[1]].band[[n]] - all_pts[cr[1]].band[[m]],
                        all_pts[cr[2]].band[[n]] - all_pts[cr[2]].band[[m]],
                        all_pts[cr[3]].band[[n]] - all_pts[cr[3]].band[[m]],
                    ];

                    for edges in &edge_tris {
                        let tri = make_cut_tri_from_specs(
                            &coords, &udata, &pdata, &ddata, &srt, edges,
                        );
                        let u_tri = [tri[0].u, tri[1].u, tri[2].u];
                        let p_tri = [tri[0].p, tri[1].p, tri[2].p];
                        let d_tri = [tri[0].d, tri[1].d, tri[2].d];
                        let q_tri = [tri[0].k, tri[1].k, tri[2].k];

                        if eta == 0.0 {
                            let dmin = d_tri.iter().cloned().fold(f64::INFINITY, f64::min);
                            let dmax = d_tri.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                            if dmin <= 1e-12 && dmax >= -1e-12 { continue; }
                        }
                        let surf_int = triangle_surface_integral(&u_tri, &p_tri, &d_tri, eta);
                        let area = triangle_area_3d(&q_tri);
                        acc[[im]] += -4.0 * area * surf_int / grad;
                    }
                }
            }
        }
    }
}

/// Thread-local accumulator for one 2D cell (Hall conductivity).
fn accum_hall_cell_2d(
    ix: usize, iy: usize, nx: usize, ny: usize,
    nsta: usize, n_mu: usize, eta: f64,
    mu: &Array1<f64>, all_band: &Array2<f64>, all_pts: &[TetraKPoint],
    acc: &mut Array1<f64>,
) {
    let cell_area = 1.0 / (nx * ny) as f64;
    let tri_area = cell_area / 2.0;
    let alpha = 1.0 / 6.0;
    let beta = 2.0 / 3.0;
    let bary = [[alpha, alpha, beta], [alpha, beta, alpha], [beta, alpha, alpha]];
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;

    for &(v0, v1, v2) in &[(i00, i10, i01), (i11, i10, i01)] {
        for n in 0..nsta {
            for im in 0..n_mu {
                let muv = mu[[im]];
                let mut omega_n = 0.0;
                for m in 0..nsta {
                    if m == n { continue; }
                    let mut ok = true;
                    let mut pair_contrib = 0.0;
                    for q in 0..3 {
                        let (l0, l1, l2) = (bary[q][0], bary[q][1], bary[q][2]);
                        let eq = l0 * all_band[[v0, n]]
                               + l1 * all_band[[v1, n]]
                               + l2 * all_band[[v2, n]];
                        if eq > muv { continue; }
                        let dq = l0 * (all_band[[v0, n]] - all_band[[v0, m]])
                               + l1 * (all_band[[v1, n]] - all_band[[v1, m]])
                               + l2 * (all_band[[v2, n]] - all_band[[v2, m]]);
                        let pq = l0 * all_pts[v0].k_ab[[n, m]].im
                               + l1 * all_pts[v1].k_ab[[n, m]].im
                               + l2 * all_pts[v2].k_ab[[n, m]].im;
                        if eta == 0.0 && dq.abs() < 1e-12 { ok = false; break; }
                        pair_contrib += pq / (dq.powi(2) + eta.powi(2));
                    }
                    if !ok { continue; }
                    omega_n -= 2.0 * pair_contrib / 3.0;
                }
                acc[[im]] += omega_n * tri_area;
            }
        }
    }
}

/// Thread-local accumulator for one 3D cell (Hall conductivity).
fn accum_hall_cell_3d(
    ix: usize, iy: usize, iz: usize,
    nx: usize, ny: usize, nz: usize,
    nsta: usize, n_mu: usize, eta: f64,
    mu: &Array1<f64>, all_band: &Array2<f64>, all_pts: &[TetraKPoint],
    acc: &mut Array1<f64>,
) {
    const TETS: [[usize; 4]; 5] = [
        [0, 1, 2, 4], [3, 1, 2, 7], [5, 1, 4, 7], [6, 2, 4, 7], [1, 2, 4, 7],
    ];
    const TVF: [f64; 5] = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0];
    let cube_vol = 1.0 / (nx * ny * nz) as f64;
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let izp = (iz + 1) % nz;
    let idx3 = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let c = [
        idx3(ix, iy, iz), idx3(ixp, iy, iz),
        idx3(ix, iyp, iz), idx3(ixp, iyp, iz),
        idx3(ix, iy, izp), idx3(ixp, iy, izp),
        idx3(ix, iyp, izp), idx3(ixp, iyp, izp),
    ];

    for (ti, &[v0, v1, v2, v3]) in TETS.iter().enumerate() {
        let vt = cube_vol * TVF[ti];
        let cr = [c[v0], c[v1], c[v2], c[v3]];
        for n in 0..nsta {
            let e = [
                all_band[[cr[0], n]], all_band[[cr[1], n]],
                all_band[[cr[2], n]], all_band[[cr[3], n]],
            ];
            let mut idx = [0usize, 1, 2, 3];
            idx.sort_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap());
            let srt = idx;
            let es0 = e[srt[0]];
            let es3 = e[srt[3]];
            for im in 0..n_mu {
                if mu[[im]] <= es0 { continue; }
                acc[[im]] += compute_occ_omega(
                    all_pts, &cr, n, nsta, vt, eta,
                    &e, &srt, mu[[im]], mu[[im]] >= es3,
                );
            }
        }
    }
}

// ── Intrinsic NLH tetrahedron integration ────────────────────────────

/// Per‑k‑point primitives for intrinsic NLH.
struct IntrinsicTetraPoint {
    band: Array1<f64>,       // nsta
    v_a_diag: Array1<f64>,   // diag(v^a) in eigenbasis
    v_b_diag: Array1<f64>,   // diag(v^b)
    v_c_diag: Array1<f64>,   // diag(v^c)
    p_ab: Array2<f64>,       // Re[v^a_nm * v^b_mn], (nsta,nsta)
    p_ac: Array2<f64>,       // Re[v^a_nm * v^c_mn]
    p_bc: Array2<f64>,       // Re[v^b_nm * v^c_mn]
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Compute per‑k‑point primitives for intrinsic NLH tetra integration.
    fn compute_intrinsic_tetra_primitives(
        &self,
        k_vec: &Array1<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        gauge: Gauge,
    ) -> IntrinsicTetraPoint {
        let directions = {
            let mut d = Array2::<f64>::zeros((3, DIM));
            d.row_mut(0).assign(dir_a);
            d.row_mut(1).assign(dir_b);
            d.row_mut(2).assign(dir_c);
            d
        };
        let (v_proj, hamk) = self.gen_v_projected(k_vec, gauge, &directions);
        let (band, evec) = hamk.eigh(UPLO::Lower).unwrap();
        let nsta = self.nsta();

        // U^T · v · U^* convention
        let ut = evec.t();
        let uc = evec.map(|x| x.conj());
        let to_band = |d: usize| -> Array2<Complex<f64>> {
            let v_raw = v_proj.slice(s![d, .., ..]).to_owned();
            ut.dot(&v_raw.dot(&uc))
        };

        let va = to_band(0);
        let vb = to_band(1);
        let vc = to_band(2);

        let v_a_diag = va.diag().map(|x| x.re).to_owned();
        let v_b_diag = vb.diag().map(|x| x.re).to_owned();
        let v_c_diag = vc.diag().map(|x| x.re).to_owned();

        let mut p_ab = Array2::<f64>::zeros((nsta, nsta));
        let mut p_ac = Array2::<f64>::zeros((nsta, nsta));
        let mut p_bc = Array2::<f64>::zeros((nsta, nsta));
        for n in 0..nsta {
            for m in 0..nsta {
                p_ab[[n, m]] = (va[[n, m]] * vb[[m, n]]).re;
                p_ac[[n, m]] = (va[[n, m]] * vc[[m, n]]).re;
                p_bc[[n, m]] = (vb[[n, m]] * vc[[m, n]]).re;
            }
        }

        IntrinsicTetraPoint {
            band,
            v_a_diag,
            v_b_diag,
            v_c_diag,
            p_ab,
            p_ac,
            p_bc,
        }
    }
}

// ── φ₃ divided differences (φ₃'''' = 1/x³) ─────────────────────────────

/// φ₃(x) = ½ x ln|x|  (satisfies φ₃''''(x) = 1/x³).
#[inline]
fn phi3(x: f64) -> f64 {
    if x.abs() < 1e-14 { 0.0 } else { 0.5 * x * x.abs().ln() }
}

/// φ₃'(x) = ½ ln|x| + ½.
#[inline]
fn phi3_prime(x: f64) -> f64 {
    if x.abs() < 1e-14 { f64::NEG_INFINITY } else { 0.5 * x.abs().ln() + 0.5 }
}

/// φ₃''(x) = 1/(2x).
#[inline]
fn phi3_double_prime(x: f64) -> f64 {
    if x.abs() < 1e-14 { f64::INFINITY } else { 0.5 / x }
}

/// φ₃'''(x) = −1/(2x²).
#[inline]
fn phi3_triple_prime(x: f64) -> f64 {
    if x.abs() < 1e-14 { f64::NEG_INFINITY } else { -0.5 / (x * x) }
}

/// Newton divided difference of φ₃ over nodes `xs`.
fn divided_diff_phi3(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 1 { return phi3(xs[0]); }

    let mut sorted: Vec<f64> = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let x_min = sorted[0];
    let x_max = sorted[n - 1];
    let tol = 1e-12 * (1.0 + x_max.abs().max(x_min.abs()));

    if x_max - x_min < tol {
        let x_mid = (x_min + x_max) / 2.0;
        let p = n - 1;
        let deriv = match p {
            0 => phi3(x_mid),
            1 => phi3_prime(x_mid),
            2 => phi3_double_prime(x_mid),
            3 => phi3_triple_prime(x_mid),
            4 => {
                // φ₃''''(x) = 1/x³
                let a = x_mid.abs();
                if a < 1e-14 { f64::INFINITY } else { 1.0 / (x_mid * x_mid * x_mid) }
            }
            _ => panic!("divided_diff_phi3: p={p} not supported"),
        };
        let fact: f64 = match p {
            0 => 1.0, 1 => 1.0, 2 => 2.0, 3 => 6.0, 4 => 24.0,
            _ => panic!(),
        };
        return deriv / fact;
    }

    let a = divided_diff_phi3(&sorted[1..]);
    let b = divided_diff_phi3(&sorted[..n - 1]);
    (a - b) / (x_max - x_min)
}

/// K³_{αβ} weight for triangle surface integral with 1/d³ denominator.
fn k3_weight_triangle(alpha: usize, beta: usize, d: &[f64; 3]) -> f64 {
    let mut nodes = vec![d[0], d[1], d[2], d[alpha], d[beta]];
    let dd = divided_diff_phi3(&nodes);
    if alpha == beta { 2.0 * dd } else { dd }
}

/// Surface integral B_Δ[u, P, d] over one triangle (without 2A_Δ prefactor).
fn triangle_integral_d3(u: &[f64; 3], p: &[f64; 3], d: &[f64; 3]) -> f64 {
    let mut sum = 0.0;
    for alpha in 0..3 {
        for beta in 0..3 {
            sum += u[alpha] * p[beta] * k3_weight_triangle(alpha, beta, d);
        }
    }
    sum
}

// ── 2D segment integral with 1/d³ denominator ────────────────────────

/// ∫₀¹ u(ξ)P(ξ)/d(ξ)³ dξ for a line segment.
///
/// Returns `None` if d crosses or touches zero (integral diverges).
fn segment_integral_d3(
    u0: f64, u1: f64, p0: f64, p1: f64, d0: f64, d1: f64,
) -> Option<f64> {
    // Divergence check: d must not cross or touch zero
    if d0 * d1 <= 0.0 { return None; }

    let dd = d1 - d0;
    let a = (u1 - u0) * (p1 - p0);
    let b = u0 * p1 + u1 * p0 - 2.0 * u0 * p0;
    let c = u0 * p0;

    if dd.abs() < 1e-14 * (1.0 + d0.abs()) {
        // Constant d limit: ∫₀¹ (aξ²+bξ+c)/d³ dξ = (c + b/2 + a/3)/d³
        return Some((c + 0.5 * b + a / 3.0) / (d0 * d0 * d0));
    }

    // χ₀ = (d₀+d₁) / (2 d₀² d₁²)
    let chi0 = (d0 + d1) / (2.0 * d0 * d0 * d1 * d1);
    // χ₁ = [1/(2d₀) − 1/d₁ + d₀/(2d₁²)] / Δ²
    let chi1 = (0.5 / d0 - 1.0 / d1 + 0.5 * d0 / (d1 * d1)) / (dd * dd);
    // χ₂ = [ln|d₁/d₀| + 2d₀/d₁ − d₀²/(2d₁²) − 3/2] / Δ³
    let ln_ratio = (d1 / d0).abs().ln();
    let chi2 = (ln_ratio + 2.0 * d0 / d1 - 0.5 * d0 * d0 / (d1 * d1) - 1.5)
        / (dd * dd * dd);

    Some(a * chi2 + b * chi1 + c * chi0)
}

// ── Cell accumulators ──────────────────────────────────────────────────

/// Thread-local accumulator for one 2D cell (intrinsic NLH).
fn accum_intrinsic_cell_2d(
    ix: usize, iy: usize, nx: usize, ny: usize,
    inv_nx: f64, inv_ny: f64,
    nsta: usize, n_mu: usize,
    mu: &Array1<f64>, all_pts: &[IntrinsicTetraPoint],
    acc: &mut Array1<f64>,
) {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;

    let corners_frac: [[f64; 2]; 4] = [
        [ix as f64 * inv_nx, iy as f64 * inv_ny],
        [(ix + 1) as f64 * inv_nx, iy as f64 * inv_ny],
        [(ix + 1) as f64 * inv_nx, (iy + 1) as f64 * inv_ny],
        [ix as f64 * inv_nx, (iy + 1) as f64 * inv_ny],
    ];
    let coord_of = |idx: usize| -> [f64; 2] {
        if idx == i00 { corners_frac[0] }
        else if idx == i10 { corners_frac[1] }
        else if idx == i11 { corners_frac[2] }
        else { corners_frac[3] }
    };

    for &(v0, v1, v2) in &[(i00, i10, i01), (i11, i10, i01)] {
        let cr = [v0, v1, v2];
        let tri_coords = [coord_of(v0), coord_of(v1), coord_of(v2)];

        for n in 0..nsta {
            let e_raw: [f64; 3] = [
                all_pts[cr[0]].band[[n]], all_pts[cr[1]].band[[n]],
                all_pts[cr[2]].band[[n]],
            ];
            let srt = sort3_by_energy(&e_raw);
            let es = [e_raw[srt[0]], e_raw[srt[1]], e_raw[srt[2]]];
            let grad = grad_linear_tri_2d(&tri_coords, &e_raw);
            if grad < 1e-14 { continue; }

            let va_d: [f64; 3] = [all_pts[cr[0]].v_a_diag[[n]], all_pts[cr[1]].v_a_diag[[n]], all_pts[cr[2]].v_a_diag[[n]]];
            let vb_d: [f64; 3] = [all_pts[cr[0]].v_b_diag[[n]], all_pts[cr[1]].v_b_diag[[n]], all_pts[cr[2]].v_b_diag[[n]]];
            let vc_d: [f64; 3] = [all_pts[cr[0]].v_c_diag[[n]], all_pts[cr[1]].v_c_diag[[n]], all_pts[cr[2]].v_c_diag[[n]]];

            for im in 0..n_mu {
                let mu_val = mu[[im]];
                if mu_val <= es[0] || mu_val >= es[2] { continue; }
                let seg = match cut_tri_fermi_contour_2d(&es, mu_val) {
                    Some(s) => s, None => continue,
                };
                let (i0, j0, t0) = seg[0];
                let (i1, j1, t1) = seg[1];
                let q0 = interp_coord_2d(&tri_coords, &srt, i0, j0, t0);
                let q1 = interp_coord_2d(&tri_coords, &srt, i1, j1, t1);
                let seg_len = segment_length_2d(&q0, &q1);

                for m in 0..nsta {
                    if m == n { continue; }

                    let pab: [f64; 3] = [all_pts[cr[0]].p_ab[[n,m]], all_pts[cr[1]].p_ab[[n,m]], all_pts[cr[2]].p_ab[[n,m]]];
                    let pac: [f64; 3] = [all_pts[cr[0]].p_ac[[n,m]], all_pts[cr[1]].p_ac[[n,m]], all_pts[cr[2]].p_ac[[n,m]]];
                    let pbc: [f64; 3] = [all_pts[cr[0]].p_bc[[n,m]], all_pts[cr[1]].p_bc[[n,m]], all_pts[cr[2]].p_bc[[n,m]]];
                    let d_raw: [f64; 3] = [
                        all_pts[cr[0]].band[[n]] - all_pts[cr[0]].band[[m]],
                        all_pts[cr[1]].band[[n]] - all_pts[cr[1]].band[[m]],
                        all_pts[cr[2]].band[[n]] - all_pts[cr[2]].band[[m]],
                    ];

                    let u0_ab = interp_scalar(&pab, &srt, i0, j0, t0);
                    let u1_ab = interp_scalar(&pab, &srt, i1, j1, t1);
                    let u0_ac = interp_scalar(&pac, &srt, i0, j0, t0);
                    let u1_ac = interp_scalar(&pac, &srt, i1, j1, t1);
                    let u0_bc = interp_scalar(&pbc, &srt, i0, j0, t0);
                    let u1_bc = interp_scalar(&pbc, &srt, i1, j1, t1);
                    let d0_v = interp_scalar(&d_raw, &srt, i0, j0, t0);
                    let d1_v = interp_scalar(&d_raw, &srt, i1, j1, t1);

                    // Block B(v_c, P_ab, d)
                    let vc0 = interp_scalar(&vc_d, &srt, i0, j0, t0);
                    let vc1 = interp_scalar(&vc_d, &srt, i1, j1, t1);
                    let b1 = match segment_integral_d3(vc0, vc1, u0_ab, u1_ab, d0_v, d1_v) {
                        Some(v) => v, None => continue,
                    };
                    // Block B(v_a, P_bc, d)
                    let va0 = interp_scalar(&va_d, &srt, i0, j0, t0);
                    let va1 = interp_scalar(&va_d, &srt, i1, j1, t1);
                    let b2 = match segment_integral_d3(va0, va1, u0_bc, u1_bc, d0_v, d1_v) {
                        Some(v) => v, None => continue,
                    };
                    // Block B(v_b, P_ac, d)
                    let vb0 = interp_scalar(&vb_d, &srt, i0, j0, t0);
                    let vb1 = interp_scalar(&vb_d, &srt, i1, j1, t1);
                    let b3 = match segment_integral_d3(vb0, vb1, u0_ac, u1_ac, d0_v, d1_v) {
                        Some(v) => v, None => continue,
                    };

                    // Reference integrates −Q = −2 v_c·G_ab + ½ v_a·G_bc + ½ v_b·G_ac
                    let q = -2.0 * b1 + 0.5 * b2 + 0.5 * b3;
                    acc[[im]] += q * seg_len / grad;
                }
            }
        }
    }
}

/// Thread-local accumulator for one 3D cell (intrinsic NLH).
fn accum_intrinsic_cell_3d(
    ix: usize, iy: usize, iz: usize,
    nx: usize, ny: usize, nz: usize,
    inv_nx: f64, inv_ny: f64, inv_nz: f64,
    nsta: usize, n_mu: usize,
    mu: &Array1<f64>, all_pts: &[IntrinsicTetraPoint],
    acc: &mut Array1<f64>,
) {
    const TETS: [[usize; 4]; 5] = [
        [0, 1, 2, 4], [3, 1, 2, 7], [5, 1, 4, 7], [6, 2, 4, 7], [1, 2, 4, 7],
    ];
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let izp = (iz + 1) % nz;
    let idx3 = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let c = [
        idx3(ix, iy, iz), idx3(ixp, iy, iz),
        idx3(ix, iyp, iz), idx3(ixp, iyp, iz),
        idx3(ix, iy, izp), idx3(ixp, iy, izp),
        idx3(ix, iyp, izp), idx3(ixp, iyp, izp),
    ];
    let corners_frac: [[f64; 3]; 8] = [
        [ix as f64 * inv_nx, iy as f64 * inv_ny, iz as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, iy as f64 * inv_ny, iz as f64 * inv_nz],
        [ix as f64 * inv_nx, (iy + 1) as f64 * inv_ny, iz as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, (iy + 1) as f64 * inv_ny, iz as f64 * inv_nz],
        [ix as f64 * inv_nx, iy as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, iy as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
        [ix as f64 * inv_nx, (iy + 1) as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
        [(ix + 1) as f64 * inv_nx, (iy + 1) as f64 * inv_ny, (iz + 1) as f64 * inv_nz],
    ];

    for &[v0, v1, v2, v3] in TETS.iter() {
        let cr = [c[v0], c[v1], c[v2], c[v3]];
        let coords = [corners_frac[v0], corners_frac[v1], corners_frac[v2], corners_frac[v3]];

        for n in 0..nsta {
            let e_raw: [f64; 4] = [
                all_pts[cr[0]].band[[n]], all_pts[cr[1]].band[[n]],
                all_pts[cr[2]].band[[n]], all_pts[cr[3]].band[[n]],
            ];
            let srt = sort4_by_energy(&e_raw);
            let es = [e_raw[srt[0]], e_raw[srt[1]], e_raw[srt[2]], e_raw[srt[3]]];
            let grad = grad_linear_tet_3d(&coords, &e_raw);
            if grad < 1e-14 { continue; }

            let va_d: [f64; 4] = [all_pts[cr[0]].v_a_diag[[n]], all_pts[cr[1]].v_a_diag[[n]], all_pts[cr[2]].v_a_diag[[n]], all_pts[cr[3]].v_a_diag[[n]]];
            let vb_d: [f64; 4] = [all_pts[cr[0]].v_b_diag[[n]], all_pts[cr[1]].v_b_diag[[n]], all_pts[cr[2]].v_b_diag[[n]], all_pts[cr[3]].v_b_diag[[n]]];
            let vc_d: [f64; 4] = [all_pts[cr[0]].v_c_diag[[n]], all_pts[cr[1]].v_c_diag[[n]], all_pts[cr[2]].v_c_diag[[n]], all_pts[cr[3]].v_c_diag[[n]]];

            for im in 0..n_mu {
                let mu_val = mu[[im]];
                if mu_val <= es[0] || mu_val >= es[3] { continue; }
                let edge_tris = cut_tet_fermi_surface_edges(&es, mu_val);
                if edge_tris.is_empty() { continue; }

                for m in 0..nsta {
                    if m == n { continue; }

                    let pab: [f64; 4] = [all_pts[cr[0]].p_ab[[n,m]], all_pts[cr[1]].p_ab[[n,m]], all_pts[cr[2]].p_ab[[n,m]], all_pts[cr[3]].p_ab[[n,m]]];
                    let pac: [f64; 4] = [all_pts[cr[0]].p_ac[[n,m]], all_pts[cr[1]].p_ac[[n,m]], all_pts[cr[2]].p_ac[[n,m]], all_pts[cr[3]].p_ac[[n,m]]];
                    let pbc: [f64; 4] = [all_pts[cr[0]].p_bc[[n,m]], all_pts[cr[1]].p_bc[[n,m]], all_pts[cr[2]].p_bc[[n,m]], all_pts[cr[3]].p_bc[[n,m]]];
                    let d_raw: [f64; 4] = [
                        all_pts[cr[0]].band[[n]] - all_pts[cr[0]].band[[m]],
                        all_pts[cr[1]].band[[n]] - all_pts[cr[1]].band[[m]],
                        all_pts[cr[2]].band[[n]] - all_pts[cr[2]].band[[m]],
                        all_pts[cr[3]].band[[n]] - all_pts[cr[3]].band[[m]],
                    ];

                    // Check for d crossing zero at vertices (unsafe for 1/d³)
                    if d_raw.iter().any(|&x| x.abs() < 1e-12) { continue; }
                    if d_raw.iter().any(|&x| x < 0.0) && d_raw.iter().any(|&x| x > 0.0) { continue; }

                    for edges in &edge_tris {
                        let u_tri = [
                            interp_scalar(&vc_d, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&vc_d, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&vc_d, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let pab_tri = [
                            interp_scalar(&pab, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&pab, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&pab, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let pac_tri = [
                            interp_scalar(&pac, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&pac, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&pac, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let pbc_tri = [
                            interp_scalar(&pbc, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&pbc, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&pbc, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let va_tri = [
                            interp_scalar(&va_d, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&va_d, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&va_d, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let vb_tri = [
                            interp_scalar(&vb_d, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&vb_d, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&vb_d, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let d_tri = [
                            interp_scalar(&d_raw, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_scalar(&d_raw, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_scalar(&d_raw, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        // Check d doesn't cross zero on cut triangle
                        let dmin = d_tri.iter().cloned().fold(f64::INFINITY, f64::min);
                        let dmax = d_tri.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        if dmin * dmax <= 0.0 { continue; }

                        let q_tri = [
                            interp_coord_3d(&coords, &srt, edges[0].0, edges[0].1, edges[0].2),
                            interp_coord_3d(&coords, &srt, edges[1].0, edges[1].1, edges[1].2),
                            interp_coord_3d(&coords, &srt, edges[2].0, edges[2].1, edges[2].2),
                        ];
                        let area = triangle_area_3d(&q_tri);

                        // B(v_c, P_ab)
                        let b1 = triangle_integral_d3(&u_tri, &pab_tri, &d_tri);
                        // B(v_a, P_bc)
                        let b2 = triangle_integral_d3(&va_tri, &pbc_tri, &d_tri);
                        // B(v_b, P_ac)
                        let b3 = triangle_integral_d3(&vb_tri, &pac_tri, &d_tri);

                        // Reference integrates −Q = −2 v_c·G_ab + ½ v_a·G_bc + ½ v_b·G_ac
                        let q = -2.0 * b1 + 0.5 * b2 + 0.5 * b3;
                        acc[[im]] += 2.0 * area * q / grad;
                    }
                }
            }
        }
    }
}

/// Interpolate 3D coordinate at edge intersection.
#[inline]
fn interp_coord_3d(coords: &[[f64; 3]], srt: &[usize], i: usize, j: usize, t: f64) -> [f64; 3] {
    let ci = coords[srt[i]];
    let cj = coords[srt[j]];
    [ci[0] + t * (cj[0] - ci[0]), ci[1] + t * (cj[1] - ci[1]), ci[2] + t * (cj[2] - ci[2])]
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Intrinsic nonlinear Hall conductivity via tetrahedron integration.
    ///
    /// ```text
    /// σ^{ab;c}_{int}(μ,T) = Σ_n ∫_BZ (−∂f/∂E_n) Q^{ab;c}_n(k) dk
    /// Q^{ab;c}_n = 2 v^c_n G^{ab}_n − ½(v^a_n G^{bc}_n + v^b_n G^{ac}_n)
    /// G^{ij}_n = Σ_{m≠n} Re[v^i_{nm} v^j_{mn}] / (E_n−E_m)³
    /// ```
    ///
    /// **2D**: line‑segment Fermi‑surface cut with analytic 1/d³ integral.
    /// **3D**: triangle Fermi‑surface cut with divided‑difference K³ weights.
    /// **T>0**: thermal convolution of T=0 result.
    ///
    /// Only charge (spinless) branch is supported; spinful partial_G term
    /// requires occupied‑volume tetrahedron integration (future work).
    pub fn Nonlinear_Hall_conductivity_Intrinsic_tetra(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
    ) -> Result<Array1<f64>> {
        let dim = k_mesh.len();
        assert!(dim == 2 || dim == 3, "only dim=2,3 supported, got dim={dim}");

        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let nsta = self.nsta();
        let gauge = Gauge::Atom;

        let all_pts: Vec<IntrinsicTetraPoint> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_intrinsic_tetra_primitives(
                    &kv, current_dir, dir_2, dir_3, gauge,
                )
            })
            .collect();

        let n_mu = mu.len();
        let det = self.lat.det().unwrap();
        let mut result_t0 = Array1::<f64>::zeros(n_mu);

        match dim {
            2 => {
                let (nx, ny) = (k_mesh[0], k_mesh[1]);
                let inv_nx = 1.0 / nx as f64;
                let inv_ny = 1.0 / ny as f64;
                let ncell = nx * ny;
                result_t0 = (0..ncell)
                    .into_par_iter()
                    .fold(
                        || Array1::<f64>::zeros(n_mu),
                        |mut acc, cell_id| {
                            let ix = cell_id / ny;
                            let iy = cell_id % ny;
                            accum_intrinsic_cell_2d(
                                ix, iy, nx, ny, inv_nx, inv_ny,
                                nsta, n_mu, mu, &all_pts, &mut acc,
                            );
                            acc
                        },
                    )
                    .reduce(
                        || Array1::<f64>::zeros(n_mu),
                        |mut a, b| { a += &b; a },
                    );
            }
            3 => {
                let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
                let inv_nx = 1.0 / nx as f64;
                let inv_ny = 1.0 / ny as f64;
                let inv_nz = 1.0 / nz as f64;
                let ncell = nx * ny * nz;
                result_t0 = (0..ncell)
                    .into_par_iter()
                    .fold(
                        || Array1::<f64>::zeros(n_mu),
                        |mut acc, cell_id| {
                            let ix = cell_id / (ny * nz);
                            let rem = cell_id % (ny * nz);
                            let iy = rem / nz;
                            let iz = rem % nz;
                            accum_intrinsic_cell_3d(
                                ix, iy, iz, nx, ny, nz,
                                inv_nx, inv_ny, inv_nz,
                                nsta, n_mu, mu, &all_pts, &mut acc,
                            );
                            acc
                        },
                    )
                    .reduce(
                        || Array1::<f64>::zeros(n_mu),
                        |mut a, b| { a += &b; a },
                    );
            }
            _ => unreachable!(),
        }

        let raw = result_t0 / det;

        if T > 0.0 && n_mu > 1 {
            let mu_slice = mu.as_slice().unwrap();
            if mu_slice.windows(2).any(|w| w[1] <= w[0]) {
                return Err(TbError::Other(
                    "mu must be strictly increasing for thermal convolution".into(),
                ));
            }
            let dmu = mu[1] - mu[0];
            let tol = 1e-12 * (1.0 + dmu.abs());
            if mu_slice.windows(2).any(|w| (w[1] - w[0] - dmu).abs() > tol) {
                return Err(TbError::Other(
                    "mu must have uniform spacing for thermal convolution".into(),
                ));
            }
            let beta = 1.0 / (T * 8.617e-5);
            let mut conv = Array1::<f64>::zeros(n_mu);
            for i in 0..n_mu {
                let mut s = 0.0;
                for j in 0..n_mu {
                    let x = beta * (mu[[j]] - mu[[i]]);
                    if x.abs() > 50.0 { continue; }
                    let f = 1.0 / (x.exp() + 1.0);
                    s += raw[[j]] * beta * f * (1.0 - f) * dmu;
                }
                conv[[i]] = s;
            }
            Ok(conv)
        } else {
            Ok(raw)
        }
    }
}
