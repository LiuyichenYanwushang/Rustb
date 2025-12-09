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

//!# Niu qian 方程推导非线性霍尔效应
//!以下是用niuqian 方程来推导各阶线性和非线性霍尔效应的公式过程

//!出发点是如下公式
//!$$\bm J=-e\int_\tx{BZ}\dd\bm k\sum_n f_n\bm v_n$$
//!这里 n 表示能带, 而 $f_n$ 是feimi-dirac distribution. 这里速度算符的定义按照 niuqian
//!老师的定义为 $$\bm v=\f{1}{\hbar}\f{\p\ve_n}{\p\bm k}-\f{e}{\hbar}\bm E\times\bm\Og_n$$
//!我们设第 $n$ 阶霍尔电导的定义为
//!$$\sg_{\ap_1,\ap_2,\cdots,\ap_n;d}=\f{1}{n!}\left\.\f{\p^n J_d}{\p E_{\ap_1}\cdots\p E_{\ap_n}}\right\vert_{\bm E=0}$$
//!为了得到其表达式, 我们定义级数展开
//!$$\lt\\\{\\begin{aligned}
//!f_n=f_n^{(0)}+f_n^{(1)}+f_n^{(2)}\cdots\\\\
//!\bm v_n=\bm v_n^{(0)}+\bm v_n^{(1)}+\bm v_n^{(2)}\cdots\\\\
//!\\end{aligned}\rt\.$$
//!这样我们有
//!$$ \\begin{aligned}\bm J^{(0)}&=-e\int_\tx{BZ}\dd\bm k\sum_n f_n^{(0)}\bm v_n^{(0)}\\\\
//!\bm J^{(1)}&=-e\int_\tx{BZ}\dd\bm k\sum_n f_n^{(1)}\bm v_n^{(0)}+f_n^{(0)}\bm v_n^{(1)}\\\\
//!\bm J^{(2)}&=-e\int_\tx{BZ}\dd\bm k\sum_n f_n^{(2)}\bm v_n^{(0)}+f_n^{(1)}\bm v_n^{(1)}+f_n^{(0)}\bm v_n^{(2)}\\\\
//!\\end{aligned}$$

//!接下来我们考虑 $f$ 的各阶修正. 利用玻尔兹曼方程, 我们有
//!$$\p_t f-\f{e}{\hbar}\bm E\cdot\nb_{\bm k} f=-\f{f-f_0}{\tau}$$
//!令 $f=\sum_{s=1}e^{is\og t} f_n^{(s)}$, 我们有
//!$$\\begin{aligned} is\og\sum_{s=1}f_n^{(s)}-\f{e}{\hbar}\bm E\cdot\nb_{\bm k}\sum_{s=0} f_n^{(s)}=-\f{1}{\tau}\sum_{s=1} f_n^{(s)}\\\\
//!\Rightarrow (is\og+\f{1}{\tau})\sum_{s=1} f_n^{(i)}-\f{e}{\hbar}\bm E\cdot\nb_{\bm k}\sum_{i=0} f_n^{(i)}=0\\\\
//!\\end{aligned}$$
//!最终, 我们能够得到高阶的费米分布, 为
//!$$f_n^{(l)}=\f{e}{\hbar} \f{\bm E\nb_{\bm k} f_n^{(l-1)}}{i l \og+1/\tau}=\lt(\f{e/\hbar}{i\og+1/\tau}\rt)\bm E^l\nb^l_{\bm k} f_n^{(0)}$$
//!取零频极限, 我们有 $$\lim_{\og\to 0} f_n^{(l)}\approx \lt(\f{e\tau}{\hbar}\rt)^l \bm E^l\nb^l_{\bm k} f_n^{(0)}$$

//!关于费米速度 $\bm v_n=\f{1}{\hbar}\pdv{\ve_n}{\bm k}+\f{e}{\hbar}\bm E\times\bm \Og_n$,
//!我们可以定义各阶展开
//!$$\\begin{aligned}
//!\bm v_n^{(0)}&=\f{1}{\hbar}\pdv{\ve_n^{(0)}}{\bm k}\\\\
//!\bm v_n^{(1)}&=\f{1}{\hbar}\pdv{\ve_n^{(1)}}{\bm k}+\f{e}{\hbar}\bm E\times\bm \Og_n^{(0)}\\\\
//!\bm v_n^{(2)}&=\f{1}{\hbar}\pdv{\ve_n^{(2)}}{\bm k}+\f{e}{\hbar}\bm E\times\bm \Og_n^{(1)}\\\\
//!\\end{aligned}$$
//!对于接下来我们的出发点是电场下的哈密顿量
//!$$H_{\bm k}=\sum_{mn}\lt(\ve_n^{(0)}\dt_{nm}-e\bm E\cdot\bra{\psi_n}\bm r\ket{\psi_n}\rt)\ket{\psi_n}\bra{\psi_m}$$
//!我们将其拆成两部分, 对角部分和非对角部分
//!$$\\begin{aligned}
//!H_{\bm k}^{(0)}&=\sum_{n}\lt(\ve_{n\bm k}^{(0)}-e\bm E\cdot\bm A_n\rt)\dyad{\psi_n}\\\\
//!H_{\bm k}^{(1)}&=\sum_{n=\not m}\lt(-e\bm E\cdot\bm A_{mn}\rt)\ket{\psi_m}\bra{\psi_n}\\\\
//!\\end{aligned}$$
//!这里 $\bm A_{mn}=\bra{\psi_m}\bm r\ket{\psi_n}=i\bra{\psi_m}\p_{\bm k}\ket{\psi_n}$

//!显然, 我们知道公式
//!$$e^{\hat S}\hat{\mathcal{O}}e^{-\hat S}=\mathcal{O}+\lt[\hat S,\hat{\mcl{O}}\rt]+\f{1}{2}\lt[\hat S,\lt[\hat S,\hat{\mcl{O}}\rt]\rt]+\f{1}{6}\lt[\hat S,\lt[\hat S,\lt[\hat S,\hat{\mcl{O}}\rt]\rt]\rt]\cdots$$
//! 为了方便计算, 我们可以选择一个 $\hat S$, 让 $H_{\bm k}^{(1)}+\lt[\hat S,\hat H_{\bm k}^{(0)}\rt]=0$, 我们有
//!$$\\begin{aligned}
//!H^\prime_{\bm k}&=e^{\hat S}H_{\bm k} e^{-\hat S}=H_{\bm k}^{(0)}+\lt(H_{\bm k}^{(1)}+\lt[\hat S,\hat H_{\bm k}^{(0)}\rt]\rt)+\lt(\lt[\hat S,\hat H_{\bm k}^{(1)}\rt]+\f{1}{2}\lt[\hat S,\lt[\hat S,\hat H_{\bm k}^{(0)}\rt]\rt]\rt)\cdots\\\\
//!&=H_{\bm k}^{(0)}+\f{1}{2}\lt[S,H_{\bm k}^{(1)}\rt]+\f{1}{3}\lt[S,\lt[S,H_{\bm k}^{(1)}\rt]\rt]\cdots
//!\\end{aligned}$$
//!为了满足条件, 我们选择 $$S_{nn}=0,\ S_{nm}=\f{-e\bm E\cdot \bm A_{nm}}{\ve_{nm}-e\bm E\cdot \bm A_{nm}}$$

//!因为我们有
//!$$\\begin{aligned} \lt[S,H_{\bm k}^{(0)}\rt]&=SH_{\bm k}^{(0)}-H_{\bm k}^{(0)}S=\sum_{j=\not m} S_{mj}H_{\bm k,jn}^{(0)}-\sum_{j=\not n }H_{\bm k,mj}^{(0)}S_{jn}\\\\
//!&=\sum_{j=\not m}\f{-e\bm E\cdot \bm A_{mj}\lt(\ve_j^{(0)}-e\bm E\cdot\bm A_j\rt)\dt_{jn}}{\ve_{mj}-e\bm E\cdot\lt(\bm A_m-\bm A_j\rt)}-\sum_{j=\not n}\f{-e\lt(\ve_j^{(0)}-e\bm E\cdot\bm A_j\rt)\lt(\bm E\cdot \bm A_{jn}\rt)\dt_{mj}}{\ve_{jn}-e\bm E\cdot\lt(\bm A_j-\bm A_n\rt)}\\\\
//!&=\f{e\lt(\bm E\cdot\bm A_{mn}\rt)\lt[\ve_{mn}- e\bm E\cdot\lt(\bm A_m-\bm A_n\rt)\rt]}{\ve_{mn}-e\bm E\cdot(\bm A_m-\bm A_n)}=-H_{\bm k}^{(1)}
//!\\end{aligned}$$
//!这样我们就验证了我们的结果, 我们将 $\hat S$ 进行化简和展开有
//!$$S_{nm}\approx \f{-e\bm E\cdot\bm A_{nm}}{\ve_n^{(0)}-\ve_m^{(0)}}-\f{ e^2\lt(\bm E\cdot\bm A_{nm}\rt)\lt(\bm E\cdot\lt(\bm A_n-\bm A_m\rt)\rt)}{\lt(\ve_n^{(0)}-\ve_m^{(0)}\rt)^2}$$
//!这样我们就能得到能带的各阶扰动
//!$$\\begin{aligned}
//!\ve_n^{(1)}&=-e\bm E\cdot\bm A_n\\\\
//!\ve_n^{(2)}&=\f{e^2}{2}E_a E_b \sum_{m=\not n}\f{A_{nm}^a A_{mn}^b+A_{mn}^a A_{nm}^b}{\ve_n-\ve_m}=e^2 G_n^{ab}E_a E_b\\\\
//!\ve_n^{(3)}&=-e^3E_a E_b E_c \lt( \sum_{m=\not n}\sum_{l=\not m,n}\f{A_{nl}^a A_{lm}^b A_{mn}^c}{(\ve_n-\ve_m)(\ve_n-\ve_l)}\rt)+e^3 E_a E_b E_c\lt( \sum_{m=\not n}\sum_{l=\not m,n}\f{A_{nm}^a A_{mn}^b (A_n^c-A_m^c)}{(\ve_n-\ve_m)^2}\rt)\\\\
//!\\end{aligned}$$
//!这里 $$G_n^{ab}=\sum_{m=\not n}\f{A_{nm}^a A_{mn}^b+A_{mn}^a A_{nm}^b}{\ve_n-\ve_m}=\sum_{m=\not n} 2\tx{Re}\f{v_{nm}^a v_{mn}^b}{(\ve_n-\ve_m)^3}$$
//!到这里, 我们将能带的扰动得到了. 但是有一个问题, 就是 $\bm A$ 是一个规范变换, 所以并不是唯一的,
//!同时, 对于带内的贡献 $\bm A_{n}=i\bra{\psi_{n\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}$, 没有好的求解方法,因为 $\bm A=-e\bra{\psi_n}\bm r\ket{\psi_n}$ 破坏了平移对称性. 但是我们总是可以选择一个规范.
//!在这里我们选择 $-e\bm E\cdot\bm A_n=0$, 这个规范令 $\ve_n^{(1)}=0$. 这种规范在物理的意义上, 可以理解为贝利联络和电场的方向垂直. 对于贝利曲率的高阶项, 利用 $\bm A\to\bm A^\prime=A+\lt[\hat S,\bm A\rt]+\f{1}{2}\lt[\hat S,\lt[\hat S,\bm A\rt]\rt]\cdots$, 我们有
//!$$\\begin{aligned}
//!\lt(A_n^b\rt)^{(1)}&=-e\bm E_a G_n^{ab}\\\\
//!\lt(A_n^c\rt)^{(2)}&=e^2 E_a E_b \lt( \sum_{m=\not n}\sum_{l=\not m,n}\f{A_{nl}^a A_{lm}^b A_{mn}^c}{(\ve_n-\ve_m)(\ve_n-\ve_l)}\rt)+e^2 E_a E_b\lt( \sum_{m=\not n}\f{A_{nm}^a A_{mn}^b (A_n^c-A_m^c)}{(\ve_n-\ve_m)^2}\rt)\\\\
//!&=e^2 E_a E_b\lt(S_n^{abc}-F_n^{abc}\rt)
//!\\end{aligned}$$

//!这样利用贝利曲率公式 $\Og_n^{ab}=\p_a A_n^b -\p_b A_n^a$
//!我们有 $$\\begin{aligned}
//!\lt(\Og_n^{ab}\rt)^{(1)}&=-e E_c\lt(\p_a G_n^{bc}-\p_b G_n^{ac}\rt)\\\\
//!\lt(\Og_n^{ab}\rt)^{(2)}&=e^2 E_{\ap}E_{\bt}\lt(\p_a S^{\ap\bt b}-\p_b S^{\ap\bt a}-\p_a F^{\ap\bt b}+\p_b F^{\ap\bt a}\rt)
//!\\end{aligned}$$
//!最终我们带入到电导率公式中, 有
//!$$\begin{aligned}
//!\sigma_{ab}=&-\f{e^2}{\hbar}\int_\tx{BZ} \f{\dd\bm k}{(2\pi)^3}\sum_n f_n\Og_n^{ab}+\f{e^2\tau}{\hbar^2}\sum_n \int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3}\f{\p^2\ve_n}{\p k_a\p k_b}\\\\
//!\sigma_{abc}=&-\f{e^3\tau^2}{\hbar^3}\sum_n\int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3}\f{\p^3\ve_n}{\p k_a \p k_b \p k_c}
//!+\f{e^3\tau}{\hbar^2}\sum_n \int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3} \f{1}{2} f_n \lt(\p_a\Og_n^{bc}+\p_b\Og_n^{ac}\rt)\\\\
//!&-\f{e^3}{\hbar}\sum_n\int_\tx{BZ}\f{\dd\bm k}{(2\pi)^3} f_n\lt(2\p_c G_n^{ab}-\f{1}{2}\lt(\p_a G_n^{bc}+\p_b G_n^{ac}\rt)\rt)
//!\end{aligned}$$

//! ## Berry connection 的化简

//!为了实际的计算, 我们需要将 Berry connection 的形式修改一下, 我们首先按照微分的定理有 $$\p_{\bm k}\lt(H_{\bm k}\ket{\psi_{n\bm k}}\rt)=\lt(\p_{\bm k}H_{\bm k}+H_{\bm k}\p_{\bm k}\rt)\ket{\psi_{n\bm k}}$$
//!然后我们又因为 $H_{\bm k}\ket{\psi_{n\bm k}}=\ve_{n\bm k}\ket{\psi_{n\bm k}}$, 所以 $$\p_{\bm k}\lt(H_{\bm k}\ket{\psi_{n\bm k}}\rt)=\p_{\bm k}\ve_{n\bm k}\ket{\psi_{n\bm k}}+\ve_{n\bm k}\p_{\bm k}\ket{\psi_{n\bm k}}$$
//!所以我们有 $$\\begin{aligned}
//!\p_{\bm k}H_{\bm k}\ket{\psi_{n\bm k}}+H_{\bm k}\p_{\bm k}\ket{\psi_{n\bm k}}=\p_{\bm k}\ve_{n\bm k}\ket{\psi_{n\bm k}}+\ve_{n\bm k}\p_{\bm k}\ket{\psi_{n\bm k}}
//!\\end{aligned}$$
//!显然我们将上式等号两边的左侧插入一个完备算符 $\sum_m \dyad{\psi_{m\bm k}}$ 有 $$\sum_m\lt[\bra{\psi_{m\bm k}}\p_{\bm k}H_{\bm k}\ket{\psi_{n\bm k}}+\lt(\ve_{m\bm k}-\ve_{n\bm k}\rt)\bra{\psi_{m\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}\rt]\ket{\psi_{m\bm k}}=\bra{\psi_{n\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}\ket{\psi_{n\bm k}} $$
//!根据上面的式子, 我们很容易得到当 $m=\not n$ 时 $$\bra{\psi_{m\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}=\f{\bra{\psi_{m\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}}{\ve_{n\bm k}-\ve_{m\bm k}}$$
//!也就是说, 我们能够最终得到 $$\bm A_{mn}=i\f{\bra{\psi_{m\bm k}}\p_{\bm k}\ket{\psi_{n\bm k}}}{\ve_{n\bm k}-\ve_{m\bm k}}$$

use crate::error::{Result, TbError};
use crate::kpoints::{gen_kmesh, gen_krange};
use crate::math::*;
use crate::phy_const::mu_B;
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

/**
这个函数是用来做自适应积分算法的

对于任意维度的积分 n, 我们的将区域刨分成 n+1面体的小块, 然后用线性插值来近似这个n+1的积分结果

设被积函数为 $f(x_1,x_2,...,x_n)$, 存在 $n+1$ 个点 $(y_{01},y_{02},\cdots y_{0n})\cdots(y_{n1},y_{n2}\cdots y_{nn})$, 对应的值为 $z_0,z_1,...,z_n$

这样我们就能得到这一块积分的近似值为 $$ \f{1}{(n+1)!}\times\sum_{i=0}^n z_i *\dd V.$$ 其中$\dd V$ 是正 $n+1$ 面体的体积.

在这里, 对于一维体系, 线性插值积分等价于梯形积分. 在两个相邻的数据点 ($x_1$,$f_1$) 和 ($x_2,$f_2$), 其积分结果为$\Delta=\f{f_1+f_2}{2}*(x_2-x_2)$.

对于二维系统, 用三角形进行近似, 对于任意的小三角形得到的积分结果为 $\Delta=S\sum_{i=1}^3 f_i/3!$

对于三维系统, 线性插值的结果为 $\Delta=S\sum_{i=1}^4 f_i/4!$
*/

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

#[allow(non_snake_case)]
impl Model {
    //! 这个模块是用来提供电导率张量的, 包括自旋霍尔电导率和霍尔电导率, 以及非线性霍尔电导率.
    //!
    //!
    //!
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn berry_curvature_n_onek<S: Data<Elem = f64>>(
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
                    let J = J
                        .outer_iter()
                        .zip(dir_1.iter())
                        .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                            acc + &x * (*d + 0.0 * li)
                        });
                    J
                }
                1 => {
                    let pauli = arr2(&[
                        [0.0 + 0.0 * li, 1.0 + 0.0 * li],
                        [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                    ]) / 2.0;
                    let mut X: Array2<Complex<f64>> = Array2::eye(self.nsta());
                    X = kron(&pauli, &Array2::eye(self.norb()));
                    let J = J
                        .outer_iter()
                        .zip(dir_1.iter())
                        .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                            acc + &anti_comm(&X, &x) * (*d * 0.5 + 0.0 * li)
                        });
                    J
                }
                2 => {
                    let pauli = arr2(&[
                        [0.0 + 0.0 * li, 0.0 - 1.0 * li],
                        [0.0 + 1.0 * li, 0.0 + 0.0 * li],
                    ]) / 2.0;
                    let mut X: Array2<Complex<f64>> = Array2::eye(self.nsta());
                    X = kron(&pauli, &Array2::eye(self.norb()));
                    let J = J
                        .outer_iter()
                        .zip(dir_1.iter())
                        .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                            acc + &anti_comm(&X, &x) * (*d * 0.5 + 0.0 * li)
                        });
                    J
                }
                3 => {
                    let pauli = arr2(&[
                        [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                        [0.0 + 0.0 * li, -1.0 + 0.0 * li],
                    ]) / 2.0;
                    let mut X: Array2<Complex<f64>> = Array2::eye(self.nsta());
                    X = kron(&pauli, &Array2::eye(self.norb()));
                    let J = J
                        .outer_iter()
                        .zip(dir_1.iter())
                        .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                            acc + &anti_comm(&X, &x) * (*d * 0.5 + 0.0 * li)
                        });
                    J
                }
                _ => panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}", spin),
            };
            let v = v
                .outer_iter()
                .zip(dir_2.iter())
                .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                    acc + &x * (*d + 0.0 * li)
                });
            (J, v)
        } else {
            if spin != 0 {
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }

            let J = J
                .outer_iter()
                .zip(dir_1.iter())
                .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                    acc + &x * (*d + 0.0 * li)
                });
            let v = v
                .outer_iter()
                .zip(dir_2.iter())
                .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                    acc + &x * (*d + 0.0 * li)
                });
            (J, v)
        };

        let evec_conj = evec.t();
        let evec = evec.mapv(|x| x.conj());
        let A1 = J.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec);
        let A2 = evec_conj.dot(&A2);
        let A2 = A2.reversed_axes();
        let AA = A1 * A2;
        let Complex { re, im } = AA.view().split_complex();
        let im = im.mapv(|x| -2.0 * x);
        assert_eq!(
            band.len(),
            self.nsta(),
            "this is strange for band's length is not equal to self.nsta()"
        );
        let mut UU = Array2::<f64>::zeros((self.nsta(), self.nsta()));
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                let a = band[[i]] - band[[j]];
                //这里用η进行展宽
                UU[[i, j]] = 1.0 / (a.powi(2) + eta.powi(2));
                /*
                if a.abs() < 1e-8 {
                    UU[[i, j]] = 0.0;
                } else {
                    UU[[i, j]] = 1.0 / (a.powi(2)+eta.powi(2));
                }
                */
            }
        }
        let omega_n = im
            .outer_iter()
            .zip(UU.outer_iter())
            .map(|(a, b)| a.dot(&b))
            .collect();
        let omega_n = Array1::from_vec(omega_n);
        (omega_n, band)
    }

    #[allow(non_snake_case)]
    pub fn berry_curvature_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> f64 {
        //!给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$,
        //!mu=$\mu$ 为费米能级, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
        //!当体系不存在自旋的时候无论如何输入spin都默认 spin=0
        //!eta=$\eta$ 是一个小量
        //! 这个函数返回的是
        //! $$ \sum_n f_n\Omega_{n,\ap\bt}^\gm(\bm k)=\sum_n \f{1}{e^{(\ve_{n\bm k}-\mu)/T/k_B}+1} \sum_{m=\not n}\f{J_{\ap,nm}^\gm v_{\bt,mn}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og+i\eta)^2}$$
        //! 其中 $J_\ap^\gm=\\{s_\gm,v_\ap\\}$
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
    pub fn berry_curvature<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix2>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: usize,
        eta: f64,
    ) -> Array1<f64> {
        //!这个是用来并行计算大量k点的贝利曲率
        //!这个可以用来画能带上的贝利曲率, 或者画一个贝利曲率的热图
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
    ///这个是计算某个费米能级, 某个温度下的Hall conductivity 的, 输出的单位为 $e^2/\hbar/\AA$.
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
        //!这个是用来计算霍尔电导的.
        //!这里采用的是均匀撒点的方法, 利用 berry_curvature, 我们有
        //!$$\sg_{\ap\bt}^\gm=\f{1}{N(2\pi)^r V}\sum_{\bm k} \Og_{\ap\bt}(\bm k),$$ 其中 $N$ 是 k 点数目,
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let omega = self.berry_curvature(&kvec, &dir_1, &dir_2, mu, T, spin, eta);
        //目前求积分的方法上, 还是直接求和最有用, 其他的典型积分方法, 如gauss 法等,
        //都因为存在间断点而效率不高.
        //对于非零温的, 使用梯形法应该效果能好一些.
        let conductivity: f64 = omega.sum() / (nk as f64) / self.lat.det().unwrap();
        Ok(conductivity)
    }
    #[allow(non_snake_case)]
    ///这个是采用自适应积分算法来计算霍尔电导的, 一般来说, 我们建议 re_err 设置为 1, 而 ab_err 设置为 0.01
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
    ///用来计算多个 $\mu$ 值的, 这个函数是先求出 $\Omega_n$, 然后再分别用不同的费米能级来求和, 这样速度更快, 因为避免了重复求解 $\Omega_n$, 但是相对来说更耗内存, 而且不能做到自适应积分算法.
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
        //! 这个是用来计算 $$\pdv{\ve_{n\bm k}}{k_\gm}\Og_{n,\ap\bt}$$
        //!
        //!这里需要注意的一点是, 一般来说对于 $\p_\ap\ve_{\bm k}$, 需要用差分法来求解, 我这里提供了一个算法.
        //!$$ \ve_{\bm k}=U^\dag H_{\bm k} U\Rightarrow \pdv{\ve_{\bm k}}{\bm k}=U^\dag\pdv{H_{\bm k}}{\bm k}U+\pdv{U^\dag}{\bm k} H_{\bm k}U+U^\dag H_{\bm k}\pdv{U}{\bm k}$$
        //!因为 $U^\dag U=1\Rightarrow \p_{\bm k}U^\dag U=-U^\dag\p_{\bm k}U$, $\p_{\bm k}H_{\bm k}=v_{\bm k}$我们有
        //!$$\pdv{\ve_{\bm k}}{\bm k}=v_{\bm k}+\lt[\ve_{\bm k},U^\dag\p_{\bm k}U\rt]$$
        //!而这里面唯一比较难求的项是 $D_{\bm k}=U^\dag\p_{\bm k}U$. 按照 vanderbilt 2008 年的论文中的公式, 用微扰论有
        //!$$D_{mn,\bm k}=\left\\{\\begin{aligned}\f{v_{mn,\bm k}}{\ve_n-\ve_m} \quad &\text{if}\\ m\\ =\not n\\\ 0 \quad \quad &\text{if}\\ m\\ = n\\end{aligned}\right\.$$
        //!我们观察到第二项对对角部分没有贡献, 所以我们可以直接设置为
        //!$$\pdv{\ve_{\bm k}}{\bm k}=\text{diag}\lt(v_{\bm k}\rt)$$
        //我们首先求解 omega_n 和 U^\dag j

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
            let mut X: Array2<Complex<f64>> = Array2::eye(self.nsta());
            let pauli: Array2<Complex<f64>> = match spin {
                0 => arr2(&[
                    [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                    [0.0 + 0.0 * li, 1.0 + 0.0 * li],
                ]),
                1 => {
                    arr2(&[
                        [0.0 + 0.0 * li, 1.0 + 0.0 * li],
                        [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                    ]) / 2.0
                }
                2 => {
                    arr2(&[
                        [0.0 + 0.0 * li, 0.0 - 1.0 * li],
                        [0.0 + 1.0 * li, 0.0 + 0.0 * li],
                    ]) / 2.0
                }
                3 => {
                    arr2(&[
                        [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                        [0.0 + 0.0 * li, -1.0 + 0.0 * li],
                    ]) / 2.0
                }
                _ => panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}", spin),
            };
            X = kron(&pauli, &Array2::eye(self.norb()));
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
        let evec = evec.mapv(|x| x.conj());

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
        //这个是在 onek的基础上进行并行计算得到一系列k点的berry curvature dipole
        //!This function performs parallel computation based on the onek function to obtain a series of Berry curvature dipoles at different k-points.
        //!这个方法用的是对费米分布的修正, 因为高阶的dipole 修正导致的非线性霍尔电导为 $$\sg_{\ap\bt\gm}=\tau\int\dd\bm k\sum_n\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}\lt\.\pdv{f_{\bm k}}{\ve}\rt\rvert_{E=\ve_{n\bm k}}.$$ 所以我们这里输出的是
        //!$$\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}.$$
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
        //这个是用 berry curvature dipole 对整个布里渊去做积分得到非线性霍尔电导, 是extrinsic 的
        //!This function calculates the extrinsic nonlinear Hall conductivity by integrating the Berry curvature dipole over the entire Brillouin zone. The Berry curvature dipole is first computed at a series of k-points using parallel computation based on the onek function.

        //! 我们基于 berry_curvature_n_dipole 来并行得到所有 k 点的 $\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$,
        //! 但是我们最后的公式为
        //! $$\\mathcal D_{\ap\bt\gm}=\int \dd\bm k \sum_n\lt(-\pdv{f_{n}}{\ve}\rt)\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$$
        //! 然而,
        //! $$-\pdv{f_{n}}{\ve}=\beta\f{e^{beta(\ve_n-\mu)}}{(e^{beta(\ve_n-\mu)}+1)^2}=\beta f_n(1-f_n)$$
        //! 对于 T=0 的情况, 我们将采用四面体积分来替代, 这个需要很高的k点密度, 不建议使用
        //! 对于 T!=0 的情况, 我们会采用类似 Dos 的方法来计算

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

    pub fn berry_connection_dipole_onek(
        &self,
        k_vec: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: usize,
    ) -> (Array1<f64>, Array1<f64>, Option<Array1<f64>>) {
        //!这个是根据 Nonlinear_Hall_conductivity_intrinsic 的注释, 当不存在自旋的时候提供
        //!$$v_\ap G_{\bt\gm}-v_\bt G_{\ap\gm}$$
        //!其中 $$ G_{ij}=-2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3} $$
        //!如果存在自旋, 即spin不等于0, 则还存在 $\p_{h_i} G_{jk}$ 项, 具体请看下面的非线性霍尔部分
        //!我们这里暂时不考虑磁场, 只考虑电场
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
        let evec = evec.mapv(|x| x.conj());
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
            let mut S: Array2<Complex<f64>> = Array2::eye(self.nsta());
            let li = Complex::<f64>::new(0.0, 1.0);
            let pauli: Array2<Complex<f64>> = match spin {
                0 => Array2::<Complex<f64>>::eye(2),
                1 => {
                    arr2(&[
                        [0.0 + 0.0 * li, 1.0 + 0.0 * li],
                        [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                    ]) / 2.0
                }
                2 => {
                    arr2(&[
                        [0.0 + 0.0 * li, 0.0 - 1.0 * li],
                        [0.0 + 1.0 * li, 0.0 + 0.0 * li],
                    ]) / 2.0
                }
                3 => {
                    arr2(&[
                        [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                        [0.0 + 0.0 * li, -1.0 + 0.0 * li],
                    ]) / 2.0
                }
                _ => panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}", spin),
            };
            let X = kron(&pauli, &Array2::eye(self.norb()));
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
    pub fn berry_connection_dipole(
        &self,
        k_vec: &Array2<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        dir_3: &Array1<f64>,
        spin: usize,
    ) -> (Array2<f64>, Array2<f64>, Option<Array2<f64>>) {
        //! 这个是基于 onek 的, 进行关于 k 点并行求解
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
        //! The Intrinsic Nonlinear Hall Conductivity arises from the correction of the Berry connection by the electric and magnetic fields [PRL 112, 166601 (2014)]. The formula employed is:
        //!$$\tilde\bm\Og_{\bm k}=\nb_{\bm k}\times\lt(\bm A_{\bm k}+\bm A_{\bm k}^\prime\rt)$$
        //!and the $\bm A_{i,\bm k}^\prime=F_{ij}B_j+G_{ij}E_j$, where
        //!$$
        //!\\begin{aligned}
        //!F_{ij}&=\text{Im}\sum_{m=\not n}\f{v_{i,nm}\og_{j,mn}}{\lt(\ve_{n}-\ve_m\rt)^2}\\\\
        //!G_{ij}&=2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
        //!\og_{\ap,mn}&=-i\ep_{\ap\bt\gm}\sum_{l=\not n}\f{\lt(v_{\bt,ml}+\p_\bt \ve_{\bm k}\dt_{ml}\rt)v_{\gm,ln}}{\ve_l-\ve_n}
        //!\\end{aligned}
        //!$$
        //!最后我们有
        //!$$
        //!\bm j^\prime=\bm E\times\int\f{\dd\bm k}{(2\pi)^3}\lt[\p_{\bm k}\ve_{\bm k}\times\bm A^\prime+\bm\Og\lt(\bm B\cdot\bm m\rt)\rt]\pdv{f_{\bm k}}{\ve}
        //!$$
        //!对其对电场和磁场进行偏导, 有
        //!$$
        //!\\begin{aligned}
        //!\f{\p^2 j_{\ap}^\prime}{\p E_\bt\p E_\gm}&=\int\f{\dd\bm k}{(2\pi)^3}\lt(\p_\ap\ve_{\bm k} G_{\bt\gm}-\p_\bt\ve_{\bm k} G_{\ap\gm}\rt)\pdv{f_{\bm k}}{\ve}\\\\
        //!\f{\p^2 j_{\ap}^\prime}{\p E_\bt\p B_\gm}&=\int\f{\dd\bm k}{(2\pi)^3}\lt(\p_\ap\ve_{\bm k} F_{\bt\gm}-\p_\bt\ve_{\bm k} F_{\ap\gm}+\ep_{\ap\bt\ell}\Og_{\ell} m_\gm\rt)\pdv{f_{\bm k}}{\ve}
        //!\\end{aligned}
        //!$$
        //!由于存在 $\pdv{f_{\bm k}}{\ve}$, 不建议将温度 T=0
        //!
        //!可以考虑当 T=0 时候, 利用高斯公式, 将费米面内的部分进行积分, 得到精确解. 但是我现在还没办法很好的求解费米面, 所以暂时不考虑这个算法.而且对于二维体系, 公式还不一样, 还得分步讨论, 后面有时间再考虑这个程序.
        //!
        //!对于自旋霍尔效应, 按照文章 [PRL 112, 166601 (2014)], 非线性自旋霍尔电导为
        //!$$\sg_{\ap\bt\gm}^i=-\int\dd\bm k \lt[\f{1}{2}f_{\bm k}\pdv{G_{\bt\gm}}{h_\ap}+\pdv{f_{\bm k}}{\ve}\lt(\p_{\ap}s_{\bm k}^i G_{\bt\gm}-\p_\bt\ve_{\bm k}G_{\ap\gm}^h\rt)\rt]$$
        //!其中
        //!$$\f{\p G_{\bt\gm,n}}{\p h_\ap}=2\text{Re}\sum_{n^\pr =\not n}\f{3\lt(s^i_{\ap,n}-s^i_{\ap,n_1}\rt)v_{\bt,nn_1} v_{\gm,n^\pr n}}{\lt(\ve_n-\ve_{n^\pr}\rt)^4}-2\text{Re}\sum_{n_1=\not n}\sum_{n_2=\not n}\lt[\f{s^i_{\ap,nn_2} v_{\bt,n_2n_1} v_{\gm,n_1 n}}{\lt(\ve_n-\ve_{n_1}\rt)^3(\ve_n-\ve_{n_2})}+(\bt \leftrightarrow \gm)\rt]-2\text{Re}\sum_{n_1=\not n}\sum_{n_2=\not n_1}\lt[\f{s^i_{\ap,n_1n_2} v_{\bt,n_2n} v_{\gm,n n_1}}{\lt(\ve_n-\ve_{n_1}\rt)^3(\ve_{n_1}-\ve_{n_2})}+(\bt \leftrightarrow \gm)\rt]$$
        //!以及
        //!$$
        //!\lt\\\{\\begin{aligned}
        //!G_{\ap\bt}&=2\text{Re}\sum_{m=\not n}\f{v_{\ap,nm}v_{\bt,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
        //!G_{\ap\bt}^h&=2\text{Re}\sum_{m=\not n}\f{s^i_{\ap,nm}v_{\bt,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
        //!\\end{aligned}\rt\.
        //!$$
        //!
        //!这里 $s^i_{\ap,mn}$ 的具体形式, 原文中没有明确给出, 但是我根据霍尔效应的类比, 我猜是
        //!$\\\{\hat s^i,v_\ap\\\}$

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
impl Model {
    //! This module calculates the orbital Hall conductivity
    //!
    //! The calculation using the orbial magnetism , refer to PHYSICAL REVIEW B 106, 104414 (2022).
    //!
    pub fn orbital_angular_momentum_onek(&self, kvec: &Array1<f64>) -> Array3<Complex<f64>> {
        //! 这个函数是用来计算单个 k 点的轨道角动量的
        //! 轨道角动量的定义为 $$\bra{u_{m\bm k}}\bm L\ket{u_{n\bm k}}=\frac{1}{4i
        //! g_L\mu_B}\sum_{\ell=\not m,n}\f{2\ve_{\ell\bm k}-\ve_{m\bm k}-\ve_{n\bm k}}{(\ve_{m\bm
        //! k}-\ve_{\ell\bm k})(\ve_{n\bm k}-\ve_{\ell\bm k})}\bra{u_{m\bm k}}\p_{\bm k} H_{\bm k}\ket{u_{\ell\bm k}}\times\bra{u_{\ell\bm k}}\p_{\bm k} H_{\bm k}\ket{u_{n\bm k}}$$

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

impl Model {
    //! This module calculates the optical conductivity
    //! The adopted definition is
    //! $$\sigma_{\ap\bt}=\f{2ie^2\hbar}{V}\sum_{\bm k}\sum_{n} f_n (g_{n,\ap\bt}+\f{i}{2}\Og_{n,\ap\bt})$$
    //!
    //! Where
    //! $$\\begin{aligned}
    //! g_{n\ap\bt}&=\sum_{m=\not n}\f{\og-i\eta}{\ve_{n\bm k}-\ve_{m\bm k}}\f{\text{Re} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og-i\eta)^2}\\\\
    //! \Og_{n\ap\bt}&=\sum_{m=\not n}\f{\text{Re} \bra{\psi_{n\bm k}}\p_\ap H\ket{\psi_{m\bm k}}\bra{\psi_{m\bm k}}\p_\bt H\ket{\psi_{n\bm k}}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og-i\eta)^2}
    //! \\end{aligned}
    //! $$

    #[inline(always)]
    pub fn optical_geometry_n_onek<S: Data<Elem = f64>>(
        &self,
        k_vec: &ArrayBase<S, Ix1>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        og: &Array1<f64>,
        eta: f64,
    ) -> (Array2<Complex<f64>>, Array2<Complex<f64>>, Array1<f64>) {
        //! This function calculates $g_{n,\ap\bt}$ and $\og_{n\ap\bt}$
        //!
        //! `og` represents the frequency
        //!
        //! `eta` is a small quantity

        let li: Complex<f64> = 1.0 * Complex::i();
        //let (band, evec) = self.solve_onek(&k_vec);

        let (mut v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
            self.gen_v(&k_vec, Gauge::Atom); //这是速度算符
        let mut J = v.view();

        // Project the velocity operator onto the direction dir_1
        let J = J
            .outer_iter()
            .zip(dir_1.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                acc + &x * (*d + 0.0 * li)
            });

        // Project the velocity operator onto the direction dir_2
        let v = v
            .outer_iter()
            .zip(dir_2.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (x, d)| {
                acc + &x * (*d + 0.0 * li)
            });

        let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec_conj = evec.t();
        let evec = evec.mapv(|x| x.conj());

        let A1 = J.dot(&evec);
        let A1 = &evec_conj.dot(&A1);
        let A2 = v.dot(&evec);
        let A2 = evec_conj.dot(&A2);
        let A2 = A2.reversed_axes();
        let AA = A1 * A2;

        let Complex { re, im } = AA.view().split_complex();
        let re = re.mapv(|x| Complex::new(2.0 * x, 0.0));
        let im = im.mapv(|x| Complex::new(0.0, -2.0 * x));

        let n_og = og.len();
        assert_eq!(
            band.len(),
            self.nsta(),
            "this is strange for band's length is not equal to self.nsta()"
        );

        let mut U0 = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        let mut Us = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

        // Calculate the energy differences and their inverses
        for i in 0..self.nsta() {
            for j in 0..self.nsta() {
                let a = band[[i]] - band[[j]];
                U0[[i, j]] = Complex::new(a, 0.0);
                Us[[i, j]] = if a.abs() > 1e-6 {
                    Complex::new(1.0 / a, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
            }
        }

        let mut matric_n = Array2::zeros((n_og, self.nsta()));
        let mut omega_n = Array2::zeros((n_og, self.nsta()));

        // Calculate the matrices for each frequency
        Zip::from(omega_n.outer_iter_mut())
            .and(matric_n.outer_iter_mut())
            .and(og.view())
            .for_each(|mut omega, mut matric, a0| {
                let li_eta = a0 + li * eta;
                let UU = U0.mapv(|x| (x * x - li_eta * li_eta).finv());
                let U1 = &UU * &Us * li_eta;

                let o = im
                    .outer_iter()
                    .zip(UU.outer_iter())
                    .map(|(a, b)| a.dot(&b))
                    .collect();
                let m = re
                    .outer_iter()
                    .zip(U1.outer_iter())
                    .map(|(a, b)| a.dot(&b))
                    .collect();
                let o = Array1::from_vec(o);
                let m = Array1::from_vec(m);
                omega.assign(&o);
                matric.assign(&m);
            });

        (matric_n, omega_n, band)
    }

    pub fn optical_conductivity(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        T: f64,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array1<Complex<f64>>, Array1<Complex<f64>>)>
//针对单个的
    {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let (matric_sum, omega_sum) = kvec
            .outer_iter()
            .into_par_iter()
            .map(|k| {
                let (matric_n, omega_n, band) =
                    self.optical_geometry_n_onek(&k, dir_1, dir_2, og, eta);
                let fermi_dirac = if T == 0.0 {
                    band.mapv(|x| if x > mu { 0.0 } else { 1.0 })
                } else {
                    let beta = 1.0 / T / 8.617e-5;
                    band.mapv(|x| ((beta * (x - mu)).exp() + 1.0).recip())
                };
                let fermi_dirac = fermi_dirac.mapv(|x| Complex::new(x, 0.0));
                let matric = matric_n.dot(&fermi_dirac);
                let omega = omega_n.dot(&fermi_dirac);
                (matric, omega)
            })
            .reduce(
                || (Array1::zeros(n_og), Array1::zeros(n_og)),
                |(matric_acc, omega_acc), (matric, omega)| (matric_acc + matric, omega_acc + omega),
            );
        let matric_sum = li * matric_sum / self.lat.det().unwrap() / (nk as f64);
        let omega_sum = li * omega_sum / self.lat.det().unwrap() / (nk as f64);
        Ok((matric_sum, omega_sum))
    }

    pub fn optical_conductivity_T(
        &self,
        k_mesh: &Array1<usize>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        T: &Array1<f64>,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let n_T = T.len();
        let (matric_sum, omega_sum) = kvec
            .outer_iter()
            .into_par_iter()
            .map(|k| {
                let (matric_n, omega_n, band) =
                    self.optical_geometry_n_onek(&k, dir_1, dir_2, og, eta);
                let beta = T.mapv(|x| 1.0 / x / 8.617e-5);
                let nsta = band.len();
                let n_T = beta.len();
                let mut fermi_dirac: Array2<Complex<f64>> = Array2::zeros((nsta, n_T));
                Zip::from(fermi_dirac.outer_iter_mut())
                    .and(band.view())
                    .for_each(|mut f0, e0| {
                        let a = beta
                            .map(|x0| Complex::new(((x0 * (e0 - mu)).exp() + 1.0).recip(), 0.0));
                        f0.assign(&a);
                    });
                let matric = matric_n.dot(&fermi_dirac);
                let omega = omega_n.dot(&fermi_dirac);
                (matric, omega)
            })
            .reduce(
                || (Array2::zeros((n_og, n_T)), Array2::zeros((n_og, n_T))),
                |(matric_acc, omega_acc), (matric, omega)| (matric_acc + matric, omega_acc + omega),
            );
        let matric_sum = li * matric_sum / self.lat.det().unwrap() / (nk as f64);
        let omega_sum = li * omega_sum / self.lat.det().unwrap() / (nk as f64);
        Ok((matric_sum, omega_sum))
    }

    ///直接计算 xx, yy, zz, xy, yz, xz 这六个量的光电导, 分为对称和反对称部分.
    ///输出格式为 ($\sigma_{ab}^S$, $\sigma_{ab}^A), 这里 S 和 A 表示 symmetry and antisymmetry.
    ///$sigma_{ab}^S$ 是 $6\times n_\omega$
    ///如果是二维系统, 那么输出 xx yy xy 这三个分量
    pub fn optical_conductivity_all_direction(
        &self,
        k_mesh: &Array1<usize>,
        T: f64,
        mu: f64,
        og: &Array1<f64>,
        eta: f64,
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        let li: Complex<f64> = 1.0 * Complex::i();
        let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let n_og = og.len();
        let (matric,omega):(Vec<_>,Vec<_>)=kvec.outer_iter().into_par_iter()
            .map(|k| {
                //let (band, evec) = self.solve_onek(&k);
                let (mut v, hamk): (Array3<Complex<f64>>,Array2<Complex<f64>>) = self.gen_v(&k,Gauge::Atom); //这是速度算符
                let (band, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
                    (eigvals, eigvecs)
                } else {
                    todo!()
                };
                let evec_conj=evec.t();
                let evec= evec.mapv(|x| x.conj());

                let mut A = Array3::zeros((self.dim_r(),self.nsta(),self.nsta()));
                //transfrom the basis into bolch state
                Zip::from(A.outer_iter_mut()).and(v.outer_iter()).for_each(|mut a,v| a.assign(&evec_conj.dot(&v.dot(&evec))));

                // Calculate the energy differences and their inverses
                let mut U0=Array2::zeros((self.nsta(),self.nsta()));
                let mut Us=Array2::zeros((self.nsta(),self.nsta()));
                for i in 0..self.nsta() {
                    for j in 0..self.nsta() {
                        let a = band[[i]] - band[[j]];
                        U0[[i, j]] = Complex::new(a, 0.0);
                        Us[[i, j]] = if a.abs() > 1e-6 {
                            Complex::new(1.0 / a, 0.0)
                        } else {
                            Complex::new(0.0, 0.0)
                        };
                    }
                }

                let fermi_dirac=if T==0.0{
                    band.mapv(|x| if x>mu {0.0} else {1.0})
                }else{
                    let beta=1.0/T/8.617e-5;
                    band.mapv(|x| {((beta*(x-mu)).exp()+1.0).recip()})
                };
                let fermi_dirac=fermi_dirac.mapv(|x| Complex::new(x,0.0));

                let n_og=og.len();
                assert_eq!(band.len(), self.nsta(), "this is strange for band's length is not equal to self.nsta()");

                let (matric_n,omega_n)=match self.dim_r(){
                    3=>{
                        let mut matric_n=Array2::zeros((6,n_og));
                        let mut omega_n=Array2::zeros((3,n_og));
                        let A_xx=&A.slice(s![0,..,..])*&A.slice(s![0,..,..]).t();
                        let A_yy=&A.slice(s![1,..,..])*&A.slice(s![1,..,..]).t();
                        let A_zz=&A.slice(s![2,..,..])*&A.slice(s![2,..,..]).t();
                        let A_xy=&A.slice(s![0,..,..])*&A.slice(s![1,..,..]).t();
                        let A_yz=&A.slice(s![1,..,..])*&A.slice(s![2,..,..]).t();
                        let A_xz=&A.slice(s![0,..,..])*&A.slice(s![2,..,..]).t();
                        let re_xx:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_xx;
                        let re_yy:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_yy;
                        let re_zz:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_zz;
                        let Complex { re, im } = A_xy.view().split_complex();
                        let re_xy:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_xy:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));
                        let Complex { re, im } = A_yz.view().split_complex();
                        let re_yz:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_yz:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));
                        let Complex { re, im } = A_xz.view().split_complex();
                        let re_xz:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_xz:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));
                        // Calculate the matrices for each frequency
                        Zip::from(omega_n.axis_iter_mut(Axis(1)))
                            .and(matric_n.axis_iter_mut(Axis(1)))
                            .and(og.view())
                            .par_for_each(|mut omega, mut matric, a0| {
                                let li_eta = a0 + li * eta;
                                let UU = U0.mapv(|x| (x*x - li_eta*li_eta).finv());
                                let U1:Array2<Complex<f64>> = &UU * &Us * li_eta;

                                let m = re_xx.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[0]]=m;
                                let m = re_yy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[1]]=m;
                                let m = re_zz.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[2]]=m;

                                let o = im_xy.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_xy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[0]]=o;
                                matric[[3]]=m;
                                let o = im_yz.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_yz.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[1]]=o;
                                matric[[4]]=m;
                                let o = im_xz.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_xz.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[2]]=o;
                                matric[[5]]=m;
                            });
                        (matric_n,omega_n)
                    },
                    2=>{
                        let mut matric_n=Array2::zeros((3,n_og));
                        let mut omega_n=Array2::zeros((1,n_og));
                        let A_xx=&A.slice(s![0,..,..])*&(A.slice(s![0,..,..]).reversed_axes());
                        let A_yy=&A.slice(s![1,..,..])*&(A.slice(s![1,..,..]).reversed_axes());
                        let A_xy=&A.slice(s![0,..,..])*&(A.slice(s![1,..,..]).reversed_axes());
                        let re_xx:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_xx;
                        let re_yy:Array2<Complex<f64>> = Complex::new(2.0,0.0)*A_yy;
                        let Complex { re, im } = A_xy.view().split_complex();
                        let re_xy:Array2<Complex<f64>> = re.mapv(|x| Complex::new(2.0*x, 0.0));
                        let im_xy:Array2<Complex<f64>> = im.mapv(|x| Complex::new(0.0, -2.0*x));
                        // Calculate the matrices for each frequency
                        Zip::from(omega_n.axis_iter_mut(Axis(1)))
                            .and(matric_n.axis_iter_mut(Axis(1)))
                            .and(og.view())
                            .par_for_each(|mut omega, mut matric, a0| {
                                let li_eta = a0 + li * eta;
                                let UU = U0.mapv(|x| (x*x - li_eta*li_eta).finv());
                                let U1:Array2<Complex<f64>> = &UU * &Us * li_eta;

                                let m = re_xx.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[0]]=m;
                                let m = re_yy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                matric[[1]]=m;

                                let o = im_xy.outer_iter().zip(UU.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let m = re_xy.outer_iter().zip(U1.outer_iter()).map(|(a, b)| a.dot(&b)).collect();
                                let o = Array1::from_vec(o).dot(&fermi_dirac);
                                let m = Array1::from_vec(m).dot(&fermi_dirac);
                                omega[[0]]=o;
                                matric[[2]]=m;
                            });
                        (matric_n,omega_n)
                    },
                    _=>panic!("Wrong, self.dim_r must be 2 or 3 for using optical_conductivity_all_direction")
                };
                (matric_n,omega_n)
            }).collect();
        let (matric_sum, omega_sum) = match self.dim_r() {
            3 => {
                let omega = omega
                    .into_iter()
                    .fold(Array2::zeros((3, n_og)), |omega_acc, omega| {
                        omega_acc + omega
                    });
                let matric = matric
                    .into_iter()
                    .fold(Array2::zeros((6, n_og)), |matric_acc, matric| {
                        matric_acc + matric
                    });
                (matric, omega)
            }
            2 => {
                let omega = omega
                    .into_iter()
                    .fold(Array2::zeros((1, n_og)), |omega_acc, omega| {
                        omega_acc + omega
                    });
                let matric = matric
                    .into_iter()
                    .fold(Array2::zeros((3, n_og)), |matric_acc, matric| {
                        matric_acc + matric
                    });
                (matric, omega)
            }
            _ => panic!(
                "Wrong, self.dim_r must be 2 or 3 for using optical_conductivity_all_direction"
            ),
        };
        let matric_sum = li * matric_sum / self.lat.det().unwrap() / (nk as f64);
        let omega_sum = li * omega_sum / self.lat.det().unwrap() / (nk as f64);
        Ok((matric_sum, omega_sum))
    }
}
