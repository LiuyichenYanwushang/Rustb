#![allow(warnings)]
use num_complex::Complex;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray_linalg::*;
use std::f64::consts::PI;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_linalg::conjugate;
use rayon::prelude::*;
use std::io::Write;
use std::fs::File;
use std::ops::AddAssign;
use std::ops::MulAssign;

pub trait conductivity<'a>{
    fn berry_curvature_n_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array1::<f64>,Array1::<f64>);
    ///给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$, 
    ///mu=$\mu$ 为费米能级, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
    ///当体系不存在自旋的时候无论如何输入spin都默认 spin=0
    ///eta=$\eta$ 是一个小量
    /// 这个函数返回的是 
    /// $$ \sum_n f_n\Omega_{n,\ap\bt}^\gm(\bm k)=\sum_n \f{1}{e^{(\ve_{n\bm k}-\mu)/T/k_B}+1} \sum_{m=\not n}\f{J_{\ap,nm}^\gm v_{\bt,mn}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og+i\eta)^2}$$
    /// 其中 $J_\ap^\gm=\\{s_\gm,v_\ap\\}$
    fn berry_curvature_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64;
    ///这个是用来并行计算大量k点的贝利曲率
    ///这个可以用来画能带上的贝利曲率, 或者画一个贝利曲率的热图
    fn berry_curvature(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->Array1::<f64>;
    ///这个是用来计算霍尔电导的.
    ///这里采用的是均匀撒点的方法, 利用 berry_curvature, 我们有
    ///$$\sg_{\ap\bt}^\gm=\f{1}{N(2\pi)^r V}\sum_{\bm k} \Og_{\ap\bt}(\bm k),$$ 其中 $N$ 是 k 点数目, 
    fn Hall_conductivity(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64;
    ///这个是采用自适应积分算法来计算霍尔电导的, 一般来说, 我们建议 re_err 设置为 1, 而 ab_err 设置为 0.01
    fn Hall_conductivity_adapted(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64,re_err:f64,ab_err:f64)->f64;
    ///用来计算多个 $\mu$ 值的, 这个函数是先求出 $\Omega_n$, 然后再分别用不同的费米能级来求和, 这样速度更快, 因为避免了重复求解 $\Omega_n$, 但是相对来说更耗内存, 而且不能做到自适应积分算法.
    fn Hall_conductivity_mu(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:&Array1::<f64>,spin:usize,eta:f64)->Array1::<f64>;

    /// 这个是用来计算 $$\pdv{\ve_{n\bm k}}{k_\gm}\Og_{n,\ap\bt}$$
    ///
    ///这里需要注意的一点是, 一般来说对于 $\p_\ap\ve_{\bm k}$, 需要用差分法来求解, 我这里提供了一个算法. 
    ///$$ \ve_{\bm k}=U^\dag H_{\bm k} U\Rightarrow \pdv{\ve_{\bm k}}{\bm k}=U^\dag\pdv{H_{\bm k}}{\bm k}U+\pdv{U^\dag}{\bm k} H_{\bm k}U+U^\dag H_{\bm k}\pdv{U}{\bm k}$$
    ///因为 $U^\dag U=1\Rightarrow \p_{\bm k}U^\dag U=-U^\dag\p_{\bm k}U$, $\p_{\bm k}H_{\bm k}=v_{\bm k}$我们有
    ///$$\pdv{\ve_{\bm k}}{\bm k}=v_{\bm k}+\lt[\ve_{\bm k},U^\dag\p_{\bm k}U\rt]$$
    ///而这里面唯一比较难求的项是 $D_{\bm k}=U^\dag\p_{\bm k}U$. 按照 vanderbilt 2008 年的论文中的公式, 用微扰论有 
    ///$$D_{mn,\bm k}=\left\\{\\begin{aligned}\f{v_{mn,\bm k}}{\ve_n-\ve_m} \quad &\text{if}\\ m\\ =\not n\\\ 0 \quad \quad &\text{if}\\ m\\ = n\\end{aligned}\right\.$$
    ///我们观察到第二项对对角部分没有贡献, 所以我们可以直接设置为
    ///$$\pdv{\ve_{\bm k}}{\bm k}=\text{diag}\lt(v_{\bm k}\rt)$$
    fn berry_curvature_dipole_n_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array1<f64>,Array1<f64>);
    ///This function performs parallel computation based on the onek function to obtain a series of Berry curvature dipoles at different k-points.
    ///这个方法用的是对费米分布的修正, 因为高阶的dipole 修正导致的非线性霍尔电导为 $$\sg_{\ap\bt\gm}=\tau\int\dd\bm k\sum_n\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}\lt\.\pdv{f_{\bm k}}{\ve}\rt\rvert_{E=\ve_{n\bm k}}.$$ 所以我们这里输出的是 
    ///$$\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}.$$ 
    fn berry_curvature_dipole_n(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array2::<f64>,Array2::<f64>);
    //这个是用 berry curvature dipole 对整个布里渊去做积分得到非线性霍尔电导, 是extrinsic 的 
    ///This function calculates the extrinsic nonlinear Hall conductivity by integrating the Berry curvature dipole over the entire Brillouin zone. The Berry curvature dipole is first computed at a series of k-points using parallel computation based on the onek function.

    /// 我们基于 berry_curvature_n_dipole 来并行得到所有 k 点的 $\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$,
    /// 但是我们最后的公式为
    /// $$\\mathcal D_{\ap\bt\gm}=\int \dd\bm k \sum_n\lt(-\pdv{f_{n}}{\ve}\rt)\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$$
    /// 然而, 
    /// $$-\pdv{f_{n}}{\ve}=\beta\f{e^{beta(\ve_n-\mu)}}{(e^{beta(\ve_n-\mu)}+1)^2}=\beta f_n(1-f_n)$$
    /// 对于 T=0 的情况, 我们将采用四面体积分来替代, 这个需要很高的k点密度, 不建议使用
    /// 对于 T!=0 的情况, 我们会采用类似 Dos 的方法来计算
    fn Nonlinear_Hall_conductivity_Extrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,mu:&Array1<f64>,T:f64,og:f64,spin:usize,eta:f64)->Array1<f64>;
    ///这个是根据 Nonlinear_Hall_conductivity_intrinsic 的注释, 当不存在自旋的时候提供
    ///$$v_\ap G_{\bt\gm}-v_\bt G_{\ap\gm}$$
    ///其中 $$ G_{ij}=2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3} $$
    ///如果存在自旋, 即spin不等于0, 则还存在 $\p_{h_i} G_{jk}$ 项, 具体请看下面的非线性霍尔部分
    ///我们这里暂时不考虑磁场, 只考虑电场
    fn berry_connection_dipole_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,spin:usize)->(Array1::<f64>,Array1::<f64>,Option<Array1<f64>>);
    /// 这个是基于 onek 的, 进行关于 k 点并行求解

    fn berry_connection_dipole(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,spin:usize)->(Array2<f64>,Array2<f64>,Option<Array2<f64>>);

    /// The Intrinsic Nonlinear Hall Conductivity arises from the correction of the Berry connection by the electric and magnetic fields [PRL 112, 166601 (2014)]. The formula employed is:
    ///$$\tilde\bm\Og_{\bm k}=\nb_{\bm k}\times\lt(\bm A_{\bm k}+\bm A_{\bm k}^\prime\rt)$$
    ///and the $\bm A_{i,\bm k}^\prime=F_{ij}B_j+G_{ij}E_j$, where
    ///$$ 
    ///\\begin{aligned}
    ///F_{ij}&=\text{Im}\sum_{m=\not n}\f{v_{i,nm}\og_{j,mn}}{\lt(\ve_{n}-\ve_m\rt)^2}\\\\
    ///G_{ij}&=2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
    ///\og_{\ap,mn}&=-i\ep_{\ap\bt\gm}\sum_{l=\not n}\f{\lt(v_{\bt,ml}+\p_\bt \ve_{\bm k}\dt_{ml}\rt)v_{\gm,ln}}{\ve_l-\ve_n}
    ///\\end{aligned}
    ///$$
    ///最后我们有
    ///$$
    ///\bm j^\prime=\bm E\times\int\f{\dd\bm k}{(2\pi)^3}\lt[\p_{\bm k}\ve_{\bm k}\times\bm A^\prime+\bm\Og\lt(\bm B\cdot\bm m\rt)\rt]\pdv{f_{\bm k}}{\ve}
    ///$$
    ///对其对电场和磁场进行偏导, 有
    ///$$
    ///\\begin{aligned}
    ///\f{\p^2 j_{\ap}^\prime}{\p E_\bt\p E_\gm}&=\int\f{\dd\bm k}{(2\pi)^3}\lt(\p_\ap\ve_{\bm k} G_{\bt\gm}-\p_\bt\ve_{\bm k} G_{\ap\gm}\rt)\pdv{f_{\bm k}}{\ve}\\\\
    ///\f{\p^2 j_{\ap}^\prime}{\p E_\bt\p B_\gm}&=\int\f{\dd\bm k}{(2\pi)^3}\lt(\p_\ap\ve_{\bm k} F_{\bt\gm}-\p_\bt\ve_{\bm k} F_{\ap\gm}+\ep_{\ap\bt\ell}\Og_{\ell} m_\gm\rt)\pdv{f_{\bm k}}{\ve}
    ///\\end{aligned}
    ///$$
    ///由于存在 $\pdv{f_{\bm k}}{\ve}$, 不建议将温度 T=0
    ///
    ///可以考虑当 T=0 时候, 利用高斯公式, 将费米面内的部分进行积分, 得到精确解. 但是我现在还没办法很好的求解费米面, 所以暂时不考虑这个算法.而且对于二维体系, 公式还不一样, 还得分步讨论, 后面有时间再考虑这个程序.
    ///
    ///对于自旋霍尔效应, 按照文章 [PRL 112, 166601 (2014)], 非线性自旋霍尔电导为
    ///$$\sg_{\ap\bt\gm}^i=-\int\dd\bm k \lt[\f{1}{2}f_{\bm k}\pdv{G_{\bt\gm}}{h_\ap}+\pdv{f_{\bm k}}{\ve}\lt(\p_{\ap}s_{\bm k}^i G_{\bt\gm}-\p_\bt\ve_{\bm k}G_{\ap\gm}^h\rt)\rt]$$
    ///其中
    ///$$\f{\p G_{\bt\gm,n}}{\p h_\ap}=2\text{Re}\sum_{n^\pr =\not n}\f{3\lt(s^i_{\ap,n}-s^i_{\ap,n_1}\rt)v_{\bt,nn_1} v_{\gm,n^\pr n}}{\lt(\ve_n-\ve_{n^\pr}\rt)^4}-2\text{Re}\sum_{n_1=\not n}\sum_{n_2=\not n}\lt[\f{s^i_{\ap,nn_2} v_{\bt,n_2n_1} v_{\gm,n_1 n}}{\lt(\ve_n-\ve_{n_1}\rt)^3(\ve_n-\ve_{n_2})}+(\bt \leftrightarrow \gm)\rt]-2\text{Re}\sum_{n_1=\not n}\sum_{n_2=\not n_1}\lt[\f{s^i_{\ap,n_1n_2} v_{\bt,n_2n} v_{\gm,n n_1}}{\lt(\ve_n-\ve_{n_1}\rt)^3(\ve_{n_1}-\ve_{n_2})}+(\bt \leftrightarrow \gm)\rt]$$
    ///以及
    ///$$
    ///\lt\\\{\\begin{aligned}
    ///G_{\ap\bt}&=2\text{Re}\sum_{m=\not n}\f{v_{\ap,nm}v_{\bt,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
    ///G_{\ap\bt}^h&=2\text{Re}\sum_{m=\not n}\f{s^i_{\ap,nm}v_{\bt,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
    ///\\end{aligned}\rt\.
    ///$$
    ///
    ///这里 $s^i_{\ap,mn}$ 的具体形式, 原文中没有明确给出, 但是我根据霍尔效应的类比, 我猜是
    ///$\\\{\hat s^i,v_\ap\\\}$
    fn Nonlinear_Hall_conductivity_Intrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,mu:&Array1<f64>,T:f64,spin:usize)->Array1<f64>;
}
