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

use crate::error::{Result, TbError};
use crate::kpoints::{gen_kmesh, gen_krange};
use crate::math::*;
use crate::phy_const::mu_B;
use crate::solve_ham::solve;
use crate::velocity::*;
use crate::{Gauge, Model};
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
fn build_spin_matrix(norb: usize, spin: usize) -> Array2<Complex<f64>> {
    let nsta = 2 * norb;
    let mut m = Array2::<Complex<f64>>::zeros((nsta, nsta));
    let half = Complex::new(0.5, 0.0);
    let i_half = Complex::new(0.0, 0.5);
    match spin {
        1 => {
            // σ_x ⊗ I: [0 I; I 0] / 2
            for i in 0..norb {
                m[[i, i + norb]] = half;
                m[[i + norb, i]] = half;
            }
        }
        2 => {
            // σ_y ⊗ I: [0 -iI; iI 0] / 2
            for i in 0..norb {
                m[[i, i + norb]] = -i_half;
                m[[i + norb, i]] = i_half;
            }
        }
        3 => {
            // σ_z ⊗ I: [I 0; 0 -I] / 2
            for i in 0..norb {
                m[[i, i]] = half;
                m[[i + norb, i + norb]] = -half;
            }
        }
        _ => {}
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
    /// * `dir_1` - Direction vector for the first index $\alpha$ of $\Omega_{n,\alpha\beta}$.
    ///   Must have length equal to `self.dim_r()`.
    /// * `dir_2` - Direction vector for the second index $\beta$ of $\Omega_{n,\alpha\beta}$.
    /// * `spin` - Spin operator index (0, 1, 2, 3 for $\sigma_0, \sigma_x, \sigma_y, \sigma_z$).
    /// * `eta` - Broadening parameter $\eta$ for the energy denominator.
    fn berry_curvature_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        spin: usize,
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
    /// * `dir_1` - Direction vector for the first index $\alpha$ of $\Omega_{\alpha\beta}$.
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K). If `T=0`, a step function is used for the Fermi-Dirac distribution.
    /// * `spin` - Spin operator index (0, 1, 2, 3 for $\sigma_0, \sigma_x, \sigma_y, \sigma_z$).
    ///   Ignored if the model is spinless.
    /// * `eta` - Broadening parameter $\eta$ (in eV).
    fn berry_curvature_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> f64;

    /// Computes the Berry curvature at multiple k-points in parallel.
    ///
    /// This is useful for plotting Berry curvature along band structures or generating heat maps.
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `dir_1` - Direction vector for the first index $\alpha$.
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
    /// Panics if `dir_1.len()` or `dir_2.len()` does not equal `self.dim_r()`.
    #[allow(non_snake_case)]
    fn berry_curvature<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> Array1<f64>;
}

impl BerryCurvature for Model {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn berry_curvature_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        spin: usize,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();
        //let (band, evec) = self.solve_onek(&k_vec);
        let (mut v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
            self.gen_v(&k_vec, Gauge::Atom); //这是速度算符
        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let mut J = v.view();
        let (J, v) = if self.spin {
            let J = match spin {
                0 => {
                    let mut jmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                    for d in 0..self.dim_r() {
                        let w = dir_1[d];
                        if w != 0.0 {
                            Zip::from(&mut jmat)
                                .and(&J.slice(s![d, .., ..]))
                                .for_each(|a, &b| *a += b * w);
                        }
                    }
                    jmat
                }
                1 => {
                    let sp = build_spin_matrix(self.norb(), 1);
                    let mut jmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                    for d in 0..self.dim_r() {
                        let w = dir_1[d];
                        if w != 0.0 {
                            let ac = anti_comm(&sp, &J.slice(s![d, .., ..]));
                            Zip::from(&mut jmat)
                                .and(&ac)
                                .for_each(|a, &b| *a += b * w * 0.5);
                        }
                    }
                    jmat
                }
                2 => {
                    let sp = build_spin_matrix(self.norb(), 2);
                    let mut jmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                    for d in 0..self.dim_r() {
                        let w = dir_1[d];
                        if w != 0.0 {
                            let ac = anti_comm(&sp, &J.slice(s![d, .., ..]));
                            Zip::from(&mut jmat)
                                .and(&ac)
                                .for_each(|a, &b| *a += b * w * 0.5);
                        }
                    }
                    jmat
                }
                3 => {
                    let sp = build_spin_matrix(self.norb(), 3);
                    let mut jmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                    for d in 0..self.dim_r() {
                        let w = dir_1[d];
                        if w != 0.0 {
                            let ac = anti_comm(&sp, &J.slice(s![d, .., ..]));
                            Zip::from(&mut jmat)
                                .and(&ac)
                                .for_each(|a, &b| *a += b * w * 0.5);
                        }
                    }
                    jmat
                }
                _ => panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}", spin),
            };
            let v = {
                let mut vmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                for d in 0..self.dim_r() {
                    let w = dir_2[d];
                    if w != 0.0 {
                        Zip::from(&mut vmat)
                            .and(&v.slice(s![d, .., ..]))
                            .for_each(|a, &b| *a += b * w);
                    }
                }
                vmat
            };
            (J, v)
        } else {
            if spin != 0 {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }

            let J = {
                let mut jmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                for d in 0..self.dim_r() {
                    let w = dir_1[d];
                    if w != 0.0 {
                        Zip::from(&mut jmat)
                            .and(&J.slice(s![d, .., ..]))
                            .for_each(|a, &b| *a += b * w);
                    }
                }
                jmat
            };
            let v = {
                let mut vmat = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
                for d in 0..self.dim_r() {
                    let w = dir_2[d];
                    if w != 0.0 {
                        Zip::from(&mut vmat)
                            .and(&v.slice(s![d, .., ..]))
                            .for_each(|a, &b| *a += b * w);
                    }
                }
                vmat
            };
            (J, v)
        };

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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> f64 {
        let (omega_n, band) = self.berry_curvature_n_onek(&k_vec, &dir_1, &dir_2, spin, eta);
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> Array1<f64> {
        if dir_1.len() != self.dim_r() || dir_2.len() != self.dim_r() {
            panic!(
                "Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",
                self.dim_r(),
                dir_1.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));
        let omega: Vec<f64> = k_vec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let omega_one =
                    self.berry_curvature_onek(&x.to_owned(), &dir_1, &dir_2, mu, T, spin, eta);
                omega_one
            })
            .collect();
        let omega = arr1(&omega);
        omega
    }
}

#[allow(non_snake_case)]
impl Model {
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
    /// * `dir_1` - Direction vector for the first index $\alpha$ of $\sigma_{\alpha\beta}$.
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
    /// let dir_1 = arr1(&[1.0, 0.0]);
    /// let dir_2 = arr1(&[0.0, 1.0]);
    /// let sigma_xy = model.Hall_conductivity(&kmesh, &dir_1, &dir_2, 0.0, 0.0, 0, 1e-3)?;
    /// println!("Hall conductivity = {}", sigma_xy);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(non_snake_case)]
    pub fn Hall_conductivity(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> Result<f64> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let omega = self.berry_curvature(&kvec, &dir_1, &dir_2, mu, T, spin, eta);
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
    /// * `dir_1`, `dir_2` - Direction vectors for the conductivity tensor indices.
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
        re_err: f64,
        ab_err: f64,
    ) -> Result<f64> {
        let mut k_range = gen_krange(k_mesh)?; //将要计算的区域分成小块
        let n_range = k_range.len_of(Axis(0));
        let ab_err = ab_err / (n_range as f64);
        let use_fn =
            |k0: &Array1<f64>| self.berry_curvature_onek(k0, &dir_1, &dir_2, mu, T, spin, eta);
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
    /// * `dir_1`, `dir_2` - Direction vectors for the conductivity tensor indices.
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
    /// let dir_1 = ndarray::arr1(&[1.0, 0.0]);
    /// let dir_2 = ndarray::arr1(&[0.0, 1.0]);
    /// let mu = Array1::linspace(-2.0, 2.0, 101);
    /// let sigma_vs_mu = model.Hall_conductivity_mu(&kmesh, &dir_1, &dir_2, &mu, 0.0, 0, 1e-3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn Hall_conductivity_mu(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> Result<Array1<f64>> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let (omega_n, band): (Vec<_>, Vec<_>) = kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let (omega_n, band) =
                    self.berry_curvature_n_onek(&x.to_owned(), &dir_1, &dir_2, spin, eta);
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
    /// * `dir_1` - Direction vector for the first index $\alpha$ of $\Omega_{n,\alpha\beta}$.
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        og: f64,
        spin: usize,
        eta: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        let li: Complex<f64> = 1.0 * Complex::i();
        //let (band, evec) = self.solve_onek(&k_vec);
        let (mut v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
            self.gen_v(&k_vec, Gauge::Atom); //这是速度算符
        let mut J: Array3<Complex<f64>> = v.clone();
        let mut v0 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta())); //这个是速度项, 对应的dir_3 的速度
        for r in 0..self.dim_r() {
            v0 = v0 + v.slice(s![r, .., ..]).to_owned() * dir_3[[r]];
        }
        if self.spin {
            let X = build_spin_matrix(self.norb(), spin);
            for i in 0..self.dim_r() {
                let j = J.slice(s![i, .., ..]).to_owned();
                let j = anti_comm(&X, &j) / 2.0; //这里做反对易
                J.slice_mut(s![i, .., ..]).assign(&(j * dir_1[[i]]));
                v.slice_mut(s![i, .., ..])
                    .mul_assign(Complex::new(dir_2[[i]], 0.0));
            }
        } else {
            if spin != 0 {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r() {
                J.slice_mut(s![i, .., ..])
                    .mul_assign(Complex::new(dir_1[[i]], 0.0));
                v.slice_mut(s![i, .., ..])
                    .mul_assign(Complex::new(dir_2[[i]], 0.0));
            }
        };

        let J: Array2<Complex<f64>> = J.sum_axis(Axis(0));
        let v: Array2<Complex<f64>> = v.sum_axis(Axis(0));

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

        //let (omega_n,band)=self.berry_curvature_n_onek(&k_vec,&dir_1,&dir_2,og,spin,eta);
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
    /// * `dir_1`, `dir_2` - Direction vectors for the Berry curvature indices $\alpha, \beta$.
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
    /// Panics if any of `dir_1`, `dir_2`, or `dir_3` has length different from `self.dim_r()`.
    pub fn berry_curvature_dipole_n(
        &self,
        k_vec: &Array2<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        og: f64,
        spin: usize,
        eta: f64,
    ) -> (Array2<f64>, Array2<f64>) {
        if dir_1.len() != self.dim_r() || dir_2.len() != self.dim_r() || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",
                self.dim_r(),
                dir_1.len(),
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
                    &dir_1,
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
    /// * `dir_1`, `dir_2` - Direction vectors for the Berry curvature indices $\alpha, \beta$.
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        og: f64,
        spin: usize,
        eta: f64,
    ) -> Result<Array1<f64>> {
        if dir_1.len() != self.dim_r() || dir_2.len() != self.dim_r() || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",
                self.dim_r(),
                dir_1.len(),
                dir_2.len()
            )
        }
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        //为了节省内存, 本来是可以直接算完求和, 但是为了方便, 我是先存下来再算, 让程序结构更合理
        let (omega, band) =
            self.berry_curvature_dipole_n(&kvec, &dir_1, &dir_2, &dir_3, og, spin, eta);
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
            //采用四面体积分法, 或者对于二维体系, 采用三角形积分法
            //积分的思路是, 通过将一个六面体变成5个四面体, 然后用线性插值的方法, 得到费米面,
            //以及费米面上的数, 最后, 通过积分算出来结果
            panic!("When T=0, the algorithm have not been writed, please wait for next version");
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
    /// * `dir_1` - Direction vector for the first index $\alpha$.
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: usize,
    ) -> (Array1<f64>, Array1<f64>, Option<Array1<f64>>) {
        let (mut v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
            self.gen_v(&k_vec, Gauge::Atom); //这是速度算符
        let mut J = v.clone();
        //let (band, evec) = self.solve_onek(&k_vec); //能带和本征值
        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec_conj = evec.t();
        let evec = evec.map(|x| x.conj());
        for i in 0..self.dim_r() {
            let v_s = v.slice(s![i, .., ..]).to_owned();
            let v_s = evec_conj
                .clone()
                .dot(&(v_s.dot(&evec.clone().reversed_axes()))); //变换到本征态基函数
            v.slice_mut(s![i, .., ..]).assign(&v_s); //将 v 变换到以本征态为基底
        }
        //现在速度算符已经是以本征态为基函数
        let mut v_1 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta())); //三个方向的速度算符
        let mut v_2 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        let mut v_3 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        for i in 0..self.dim_r() {
            v_1 = v_1.clone() + v.slice(s![i, .., ..]).to_owned() * Complex::new(dir_1[[i]], 0.0);
            v_2 = v_2.clone() + v.slice(s![i, .., ..]).to_owned() * Complex::new(dir_2[[i]], 0.0);
            v_3 = v_3.clone() + v.slice(s![i, .., ..]).to_owned() * Complex::new(dir_3[[i]], 0.0);
        }
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
        if self.spin {
            //如果考虑自旋, 我们就计算 \partial_h G_{ij}
            let X = if spin == 0 {
                Array2::eye(self.nsta())
            } else {
                build_spin_matrix(self.norb(), spin)
            };
            let mut S = Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
            for i in 0..self.dim_r() {
                let v0 = J.slice(s![i, .., ..]).to_owned();
                let v0 = anti_comm(&X, &v0) / 2.0;
                let v0 = evec_conj
                    .clone()
                    .dot(&(v0.dot(&evec.clone().reversed_axes()))); //变换到本征态基函数
                S.slice_mut(s![i, .., ..]).assign(&v0);
            }
            let mut s_1 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta())); //三个方向的速度算符
            let mut s_2 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            let mut s_3 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            for i in 0..self.dim_r() {
                s_1 =
                    s_1.clone() + S.slice(s![i, .., ..]).to_owned() * Complex::new(dir_1[[i]], 0.0);
                s_2 =
                    s_2.clone() + S.slice(s![i, .., ..]).to_owned() * Complex::new(dir_2[[i]], 0.0);
                s_3 =
                    s_3.clone() + S.slice(s![i, .., ..]).to_owned() * Complex::new(dir_3[[i]], 0.0);
            }
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
            //开始计算partial_s
            let partial_s_1 = s_1.clone().diag().map(|x| x.re);
            let partial_s_2 = s_2.clone().diag().map(|x| x.re);
            let partial_s_3 = s_3.clone().diag().map(|x| x.re);
            let mut partial_s = Array2::<f64>::zeros((self.dim_r(), self.nsta()));
            for r in 0..self.dim_r() {
                let s0 = S.slice(s![r, .., ..]).to_owned();
                partial_s
                    .slice_mut(s![r, ..])
                    .assign(&s0.diag().map(|x| x.re)); //\p_i s算符的对角项
            }
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
            //开始计算 G_{ij}
            //G_{ij}=2Re\sum_{m\neq n} v_{i,nm}v_{j,mn}/(E_n-E_m)^3
            let G_23: Array1<f64> = {
                //用来计算  beta gamma 的 G
                let A = &v_2 * (U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0)));
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&v_3.slice(s![.., i])).re * 2.0
                }
                G
            };
            let G_13: Array1<f64> = {
                //用来计算 alpha gamma 的 G
                let A = &v_1 * (U0.map(|x| Complex::<f64>::new(x.powi(3), 0.0)));
                let mut G = Array1::<f64>::zeros(self.nsta());
                for i in 0..self.nsta() {
                    G[[i]] = A.slice(s![i, ..]).dot(&v_3.slice(s![.., i])).re * 2.0
                }
                G
            };
            return (partial_ve_1 * G_23 - partial_ve_2 * G_13, band, None);
        }
    }

    /// Computes the Berry connection dipole at multiple k-points in parallel.
    ///
    /// This is a parallelized version of [`berry_connection_dipole_onek`].
    ///
    /// # Arguments
    ///
    /// * `k_vec` - Array of k-points, shape `(nk, dim_r)`.
    /// * `dir_1`, `dir_2`, `dir_3` - Direction vectors for the three indices.
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: usize,
    ) -> (Array2<f64>, Array2<f64>, Option<Array2<f64>>) {
        if dir_1.len() != self.dim_r() || dir_2.len() != self.dim_r() || dir_3.len() != self.dim_r()
        {
            panic!(
                "Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",
                self.dim_r(),
                dir_1.len(),
                dir_2.len()
            )
        }
        let nk = k_vec.len_of(Axis(0));

        if self.spin {
            let ((omega, band), partial_G): ((Vec<_>, Vec<_>), Vec<_>) = k_vec
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|x| {
                    let (omega_one, band, partial_G) = self.berry_connection_dipole_onek(
                        &x.to_owned(),
                        &dir_1,
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
                        &dir_1,
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
    /// * `dir_1`, `dir_2`, `dir_3` - Direction vectors for the three tensor indices.
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
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        spin: usize,
    ) -> Result<Array1<f64>> {
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let (omega, band, mut partial_G): (Array2<f64>, Array2<f64>, Option<Array2<f64>>) =
            self.berry_connection_dipole(&kvec, &dir_1, &dir_2, &dir_3, spin);
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
                        let f = 1.0 / ((beta * (mu - *energy)).mapv(|x| x.exp() + 1.0));
                        acc + &f * (1.0 - &f) * beta * *omega0
                    },
                )
                .reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            if self.spin {
                let partial_G = partial_G.unwrap();
                let conductivity_new: Vec<f64> = mu
                    .into_par_iter()
                    .map(|x| {
                        let f = band0.map(|x0| 1.0 / ((beta * (x - x0)).exp() + 1.0));
                        let mut omega = Array1::<f64>::zeros(nk);
                        for i in 0..nk {
                            omega[[i]] = (partial_G.row(i).to_owned() * f.row(i).to_owned()).sum();
                        }
                        omega.sum() / 2.0
                    })
                    .collect();
                let conductivity_new = Array1::<f64>::from_vec(conductivity_new);
                conductivity = conductivity.clone() + conductivity_new;
            }
            conductivity = conductivity.clone() / (nk as f64) / self.lat.det().unwrap();
        } else {
            //采用四面体积分法, 或者对于二维体系, 采用三角形积分法
            //积分的思路是, 通过将一个六面体变成5个四面体, 然后用线性插值的方法, 得到费米面,
            //以及费米面上的数, 最后, 通过积分算出来结果
            panic!("the code can not support for T=0");
        }
        Ok(conductivity)
    }
}
