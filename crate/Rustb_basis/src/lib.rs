#![allow(warnings)]
use num_complex::Complex;
use ndarray::prelude::*;
use std::ops::AddAssign;
use std::ops::MulAssign;
pub trait basis<'a>{
    fn tb_model(
        dim_r:usize,
        lat:Array2::<f64>,
        orb:Array2::<f64>,
        spin:bool,
        atom:Option<Array2::<f64>>,
        atom_list:Option<Vec<usize>>
        )->Self;
    /// This function is used to add hopping to the model. The "set" indicates that it can be used to override previous hopping.
    ///
    /// - tmp: the parameters for hopping
    ///
    /// - ind_i and ind_j: the orbital indices in the Hamiltonian, representing hopping from i to j
    ///
    /// - R: the position of the target unit cell for hopping
    ///
    /// - pauli: can take the values of 0, 1, 2, or 3, representing $\sigma_0$, $\sigma_x$, $\sigma_y$, $\sigma_z$.
    ///
    /// In general, this function is used to set $\bra{i\bm 0}\hat H\ket{j\bm R}=$tmp.
    fn set_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:&Array1::<isize>,pauli:isize);
    ///参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp 
    fn add_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:&Array1::<isize>,pauli:isize);
    ///相比于add_hop, 相当于直接对哈密顿量进行操作, 更加灵活
    fn add_element(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:&Array1::<isize>);
    /// 直接对对角项进行设置
    fn set_onsite(&mut self, tmp:Array1::<f64>,pauli:isize);
    fn set_onsite_one(&mut self, tmp:f64,ind:usize,pauli:isize);
    fn del_hop(&mut self,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize);
    ///根据高对称点来生成高对称路径, 画能带图
    fn k_path(&self,path:&Array2::<f64>,nk:usize)->(Array2::<f64>,Array1::<f64>,Array1::<f64>);
    ///这个是做傅里叶变换, 将实空间的哈密顿量变换到倒空间的哈密顿量
    ///
    ///具体来说, 就是
    ///$$H_{mn,\bm k}=\bra{m\bm k}\hat H\ket{n\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{-i(\bm R-\bm\tau_i+\bm \tau_j)\cdot\bm k}$$
    fn gen_ham(&self,kvec:&Array1::<f64>)->Array2::<Complex<f64>>;
    ///和 gen_ham 类似, 将 $\hat{\bm r}$ 进行傅里叶变换
    ///
    ///$$\bm r_{mn,\bm k}=\bra{m\bm k}\hat{\bm r}\ket{n\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat{\bm r}\ket{n\bm R}e^{-i(\bm R-\bm\tau_i+\bm \tau_j)\cdot\bm k}$$
    fn gen_r(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>;
    ///这个函数是用来生成速度算符的, 即 $\bra{m\bm k}\p_\ap H_{\bm k}\ket{n\bm k},$
    ///这里的基函数是布洛赫波函数
    ///
    /// 这里速度算符的计算公式, 我们在程序中采用 tight-binding 模型,
    /// 即傅里叶变换的时候考虑原子位置. 
    ///
    /// 这样我们就有
    ///
    /// $$
    /// \\begin\{aligned\}
    /// \\bra{m\bm k}\p_\ap H_{\bm k}\ket{n\bm k}&=\p_\ap\left(\bra{m\bm k} H\ket{n\bm k}\rt)-\p_\ap\left(\bra{m\bm k}\rt) H\ket{n\bm k}-\bra{m\bm k} H\p_\ap\ket{n\bm k}\\\\
    /// &=\sum_{\bm R} i(\bm R-\bm\tau_m+\bm\tau_n)H_{mn}(\bm R) e^{i\bm k\cdot(\bm R-\bm\tau_m+\bm\tau_n)}-\lt[H_{\bm k},\\mathcal A_{\bm k,\ap}\rt]_{mn}
    /// \\end\{aligned\}
    /// $$
    ///
    ///这里的 $\\mathcal A_{\bm k}$ 的定义为 $$\\mathcal A_{\bm k,\ap,mn}=-i\sum_{\bm R}r_{mn,\ap}(\bm R)e^{i\bm k\cdot(\bm R+\bm\tau_m-\bm\tau_{n})}+i\tau_{n\ap}\dt_{mn}$$
    fn gen_v(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>;
    ///求解单个k点的能带值
    fn solve_band_onek(&self,kvec:&Array1::<f64>)->Array1::<f64>;
    fn solve_band_all(&self,kvec:&Array2::<f64>)->Array2::<f64>;
    ///并行求解多个k点的能带值
    fn solve_band_all_parallel(&self,kvec:&Array2::<f64>)->Array2::<f64>;
    fn solve_onek(&self,kvec:&Array1::<f64>)->(Array1::<f64>,Array2::<Complex<f64>>);
    fn solve_all(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>);
    ///这个函数是用来将model的某个方向进行截断的
    ///
    ///num:截出多少个原胞
    ///
    ///dir:方向
    ///
    ///返回一个model, 其中 dir 和输入的model是一致的, 但是轨道数目和原子数目都会扩大num倍, 沿着dir方向没有胞间hopping.
    fn solve_all_parallel(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>);
    /// This function is used to truncate a certain direction of a model.
    ///
    /// Parameters:
    /// - num: number of unit cells to truncate.
    /// - dir: the direction to be truncated.
    ///
    /// Returns a new model with the same direction as the input model, but with the number of orbitals and atoms increased by a factor of "num". There is no inter-cell hopping along the "dir" direction.
    fn cut_piece(&self,num:usize,dir:usize)->Self;
    fn cut_dot(&self,num:usize,shape:usize,dir:Option<Vec<usize>>)->Self;
        ///This function is used to transform the model, where the new basis after transformation is given by $L' = UL$.
    fn make_supercell(&self,U:&Array2::<f64>)->Self;
    fn remove_orb(&mut self,orb_list:usize);
    fn remove_atom(&mut self,atom_list:usize);
    ///这个函数是用来删除某个轨道的
    fn unfold(&self,U:&Array2::<f64>,kvec:&Array2::<f64>,E_min:f64,E_max:f64,E_n:usize)->Array2::<f64>;
    /// 能带反折叠算法, 用来计算能带反折叠后的能带.
    fn shift_to_zero(&mut self);
    /// 我这里用的算法是高斯算法, 其算法过程如下
    /// 首先, 根据 k_mesh 算出所有的能量 $\ve_n$, 然后, 按照定义
    /// $$\rho(\ve)=\sum_N\int\dd\bm k \delta(\ve_n-\ve)$$
    /// 我们将 $\delta(\ve_n-\ve)$ 做了替换, 换成了 $\f{1}{\sqrt{2\pi}\sigma}e^{-\f{(\ve_n-\ve)^2}{2\sigma^2}}$
    /// 然后, 计算方法是先算出所有的能量, 再将能量乘以高斯分布, 就能得到态密度.
    /// 态密度的光滑程度和k点密度以及高斯分布的展宽有关
    fn dos(&self,k_mesh:&Array1::<usize>,E_min:f64,E_max:f64,E_n:usize,sigma:f64)->(Array1::<f64>,Array1::<f64>);
    ///这个函数是用来快速画能带图的, 用python画图, 因为Rust画图不太方便.
    fn show_band(&self,path:&Array2::<f64>,label:&Vec<&str>,nk:usize,name:&str)-> std::io::Result<()>;
    fn from_hr(path:&str,file_name:&str,zero_energy:f64)->Self;
}
