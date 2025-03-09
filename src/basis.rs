//! 这个 impl 是给 tight-binding 模型提供基础的函数.
use crate::atom_struct::{Atom, AtomType, OrbProj};
use crate::generics::hop_use;
use crate::ndarray_lapack::{eigh_r, eigvalsh_r, eigvalsh_v};
use crate::{Model, comm, gen_kmesh, Gauge};
use ndarray::concatenate;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use ndarray_linalg::{Eigh, UPLO};
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::ops::AddAssign;


pub fn find_R<A: Data<Elem = T>, B: Data<Elem = T>, T: std::cmp::PartialEq>(
    hamR: &ArrayBase<A, Ix2>,
    R: &ArrayBase<B, Ix1>,
) -> bool {
    //!用来寻找 R 在hamR 中是否存在
    let n_R: usize = hamR.len_of(Axis(0));
    let dim_R: usize = hamR.len_of(Axis(1));
    for i in 0..(n_R) {
        let mut a = true;
        for j in 0..(dim_R) {
            a = a && (hamR[[i, j]] == R[[j]]);
        }
        if a {
            return true;
        }
    }
    false
}
#[allow(non_snake_case)]
#[inline(always)]
pub fn index_R<A: Data<Elem = T>, B: Data<Elem = T>, T: std::cmp::PartialEq>(
    hamR: &ArrayBase<A, Ix2>,
    R: &ArrayBase<B, Ix1>,
) -> usize {
    //!如果 R 在 hamR 中存在, 返回 R 在hamR 中的位置
    let n_R: usize = hamR.len_of(Axis(0));
    let dim_R: usize = hamR.len_of(Axis(1));
    for i in 0..n_R {
        let mut a = true;
        for j in 0..dim_R {
            a = a && (hamR[[i, j]] == R[[j]]);
        }
        if a {
            return i;
        }
    }
    panic!("Wrong, not find");
}
#[inline(always)]
fn remove_row<T: Copy>(array: Array2<T>, row_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.nrows()).filter(|&r| r != row_to_remove).collect();
    array.select(Axis(0), &indices)
}
#[inline(always)]
fn remove_col<T: Copy>(array: Array2<T>, col_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.ncols()).filter(|&r| r != col_to_remove).collect();
    array.select(Axis(1), &indices)
}

macro_rules! update_hamiltonian {
    //这个代码是用来更新哈密顿量的, 判断是否有自旋, 以及要更新的 ind_i, ind_j,
    //输入一个哈密顿量, 返回一个新的哈密顿量
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                0 => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = $tmp;
                }
                1 => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] = $tmp;
                }
                2 => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] = -$tmp * Complex::<f64>::i();
                }
                3 => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = -$tmp;
                }
                _ => todo!(),
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] = $tmp;
        }
        $new_ham
    }};
}

macro_rules! add_hamiltonian {
    //这个代码是用来更新哈密顿量的, 判断是否有自旋, 以及要更新的 ind_i, ind_j,
    //输入一个哈密顿量, 返回一个新的哈密顿量
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                0 => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] += $tmp;
                }
                1 => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] += $tmp;
                }
                2 => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] -= $tmp * Complex::<f64>::i();
                }
                3 => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] -= $tmp;
                }
                _ => todo!(),
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] += $tmp;
        }
        $new_ham
    }};
}

impl Model {
    //! 这个 impl 是给 tight-binding 模型提供基础的函数.
    /// #Examples
    /// ```
    ///use ndarray::*;
    ///use ndarray::prelude::*;
    ///use num_complex::Complex;
    ///use Rustb::*;
    /// //set the graphene model
    ///let lat=array![[1.0,0.0],[-1.0/2.0,3_f64.sqrt()/2.0]];
    ///let orb=array![[1.0/3.0,2.0/3.0],[2.0/3.0,1.0/3.0]];
    ///let spin=false;
    ///let mut graphene_model=Model::tb_model(2,lat,orb,spin,None);
    ///
    ///```
    pub fn tb_model(
        dim_r: usize,
        lat: Array2<f64>,
        orb: Array2<f64>,
        spin: bool,
        atom: Option<Vec<Atom>>,
    ) -> Model {
        /*
        //!这个函数是用来初始化一个 Model, 需要输入的变量意义为
        //!
        //!模型维度 dim_r,
        //!
        //!轨道数目 norb,
        //!
        //!晶格常数 lat,
        //!
        //!轨道 orb,
        //!
        //!是否考虑自旋 spin,
        //!
        //!原子数目 natom, 可以选择 None,
        //!
        //!原子位置坐标 atom, 可以选择 None,
        //!
        //!每个原子的轨道数目, atom_list, 可以选择 None.
        //!
        //! 注意, 如果原子部分存在 None, 那么最好统一都是None.
         */
        //! This function is used to initialize a Model. The variables that need to be input are as follows:
        //!
        //! - dim_r: the dimension of the model
        //!
        //! - lat: the lattice constant
        //!
        //! - orb: the orbital coordinates
        //!
        //! - spin: whether to consider spin
        //!
        //! - atom: the atomic coordinates, which can be None
        //!
        //! - atom_list: the number of orbitals for each atom, which can be None.
        //!
        //! Note that if any of the atomic variables are None, it is better to make them all None.
        //!
        //!
        let norb: usize = orb.len_of(Axis(0));
        let mut nsta: usize = norb;
        if spin {
            nsta *= 2;
        }
        let mut new_atom_list: Vec<usize> = vec![1];
        let mut new_atom = Array2::<f64>::zeros((0, dim_r));
        if lat.len_of(Axis(1)) != dim_r {
            panic!("Wrong, the lat's second dimension's length must equal to dim_r")
        }
        if lat.len_of(Axis(0)) != lat.len_of(Axis(1)) {
            panic!(
                "Wrong, the lat's second dimension's length must less than first dimension's length"
            )
        }
        let new_atom = match atom {
            Some(atom0) => atom0,
            None => {
                //通过判断轨道是不是离得太近而判定是否属于一个原子,
                //这种方法只适用于wannier90不开最局域化
                let mut new_atom = Vec::new();
                new_atom.push(Atom::new(orb.row(0).to_owned(), 1, AtomType::H));
                for i in 1..norb {
                    if (orb.row(i).to_owned() - new_atom[new_atom.len() - 1].position()).norm_l2()
                        > 1e-2
                    {
                        let use_atom = Atom::new(orb.row(i).to_owned(), 1, AtomType::H);
                        new_atom.push(use_atom);
                    } else {
                        let n = new_atom.len();
                        new_atom[n - 1].push_orb();
                    }
                }
                new_atom
            }
        };
        let natom = new_atom.len();
        let ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
        let hamR = Array2::<isize>::zeros((1, dim_r));
        let mut rmatrix = Array4::<Complex<f64>>::zeros((1, dim_r, nsta, nsta));
        for i in 0..norb {
            for r in 0..dim_r {
                rmatrix[[0, r, i, i]] = Complex::<f64>::from(orb[[i, r]]);
                if spin {
                    rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                }
            }
        }
        let orb_projection = vec![OrbProj::s; norb];
        let mut model = Model {
            dim_r,
            spin,
            lat,
            orb,
            orb_projection,
            atoms: new_atom,
            ham,
            hamR,
            rmatrix,
        };
        model
    }
    pub fn set_projection(&mut self, proj: &Vec<OrbProj>) {
        //! 这个函数是用来设置tb模型的projection 的
        self.orb_projection = proj.clone();
    }
    #[allow(non_snake_case)]
    pub fn set_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: isize,
    ) {
        /*
        //! 这个是用来给模型添加 hopping 的, "set" 表示可以用来覆盖之前的hopping
        //!
        //! tmp: hopping 的参数
        //!
        //! ind_i,ind_j: 哈密顿量中的轨道序数, 表示从 i-> j 的hopping
        //!
        //! R: 表示hopping 到的原胞位置
        //!
        //! pauli:可以取0,1,2,3, 分别表示 $\sg_0$, $\sg_x$, $\sg_y$, $\sg_z$.
        //!
        //! 总地来说, 这个函数是让 $\bra{i\bm 0}\hat H\ket{j\bm R}=$tmp
         */
        //! This function is used to add hopping to the model. The "set" indicates that it can be used to override previous hopping.
        //!
        //! - tmp: the parameters for hopping
        //!
        //! - ind_i and ind_j: the orbital indices in the Hamiltonian, representing hopping from i to j
        //!
        //! - R: the position of the target unit cell for hopping
        //!
        //! - pauli: can take the values of 0, 1, 2, or 3, representing $\sigma_0$, $\sigma_x$, $\sigma_y$, $\sigma_z$.
        //!
        //! In general, this function is used to set $\bra{i\bm 0}\hat H\ket{j\bm R}=$tmp.
        //!
        //!
        //! #Examples
        //! ```
        //!use ndarray::*;
        //!use ndarray::prelude::*;
        //!use num_complex::Complex;
        //!use Rustb::*;
        //! //set the graphene model
        //!let lat=array![[1.0,0.0],[-1.0/2.0,3_f64.sqrt()/2.0]];
        //!let orb=array![[1.0/3.0,2.0/3.0],[2.0/3.0,1.0/3.0]];
        //!let spin=false;
        //!let mut graphene_model=Model::tb_model(2,lat,orb,spin,None);
        //! let t=1.0; //the nearst hopping
        //! graphene_model.set_hop(t,0,1,&array![0,0],0);
        //! // t is the hopping, 0, 1 is the orbital ,array![0,0] is the unit cell
        //! graphene_model.set_hop(t,0,1,&arr1(&[1,0]),0);
        //! graphene_model.set_hop(t,0,1,&arr1(&vec![0,-1]),0);
        //!
        //! ```

        let tmp: Complex<f64> = tmp.to_complex();
        if pauli != 0 && self.spin == false {
            eprintln!("Wrong, if spin is True and pauli is not zero, the pauli is not use")
        }
        assert!(
            R.len() == self.dim_r(),
            "Wrong, the R length should equal to dim_r"
        );
        assert!(
            ind_i < self.norb() && ind_j < self.norb(),
            "Wrong, ind_i and ind_j must be less than norb, here norb is {}, but ind_i={} and ind_j={}",
            self.norb(),
            ind_i,
            ind_j
        );

        let R_exist = find_R(&self.hamR, &R);
        let negative_R = &(-R);
        let norb = self.norb();
        if R_exist {
            let index = index_R(&self.hamR, &R);
            let index_inv = index_R(&self.hamR, &negative_R);
            if self.ham[[index, ind_i, ind_j]] != Complex::new(0.0, 0.0) {
                eprintln!(
                    "Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",
                    self.ham[[index, ind_i, ind_j]]
                )
            }
            update_hamiltonian!(
                self.spin,
                pauli,
                tmp,
                self.ham.slice_mut(s![index, .., ..]),
                ind_i,
                ind_j,
                norb
            );
            if index != 0 || ind_i != ind_j {
                update_hamiltonian!(
                    self.spin,
                    pauli,
                    tmp.conj(),
                    self.ham.slice_mut(s![index_inv, .., ..]),
                    ind_j,
                    ind_i,
                    norb
                );
            }
            assert!(
                !(ind_i == ind_j && tmp.im != 0.0 && (pauli == 0 || pauli == 3) && index == 0),
                "Wrong, the onsite hopping must be real, but here is {}",
                tmp
            );
        } else {
            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

            let new_ham = update_hamiltonian!(self.spin, pauli, tmp, new_ham, ind_i, ind_j, norb);
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), R.view()).unwrap();
            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

            let new_ham =
                update_hamiltonian!(self.spin, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), negative_R.view()).unwrap();
        }
    }

    #[allow(non_snake_case)]
    pub fn add_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: isize,
    ) {
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli != 0 && self.spin == false {
            eprintln!("Wrong, if spin is True and pauli is not zero, the pauli is not use")
        }
        assert!(
            R.len() == self.dim_r(),
            "Wrong, the R length should equal to dim_r"
        );
        assert!(
            ind_i < self.norb() && ind_j < self.norb(),
            "Wrong, ind_i and ind_j must be less than norb, here norb is {}, but ind_i={} and ind_j={}",
            self.norb(),
            ind_i,
            ind_j
        );
        let R_exist = find_R(&self.hamR, &R);
        let negative_R = &(-R);
        let norb = self.norb();
        if R_exist {
            let index = index_R(&self.hamR, &R);
            let index_inv = index_R(&self.hamR, &negative_R);
            add_hamiltonian!(
                self.spin,
                pauli,
                tmp,
                self.ham.slice_mut(s![index, .., ..]),
                ind_i,
                ind_j,
                norb
            );
            if index != 0 || ind_i != ind_j {
                add_hamiltonian!(
                    self.spin,
                    pauli,
                    tmp.conj(),
                    self.ham.slice_mut(s![index_inv, .., ..]),
                    ind_j,
                    ind_i,
                    norb
                );
            }
            assert!(
                !(ind_i == ind_j && tmp.im != 0.0 && (pauli == 0 || pauli == 3) && index == 0),
                "Wrong, the onsite hopping must be real, but here is {}",
                tmp
            );
        } else {
            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            let new_ham = add_hamiltonian!(self.spin, pauli, tmp, new_ham, ind_i, ind_j, norb);
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), R.view()).unwrap();
            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

            let new_ham =
                add_hamiltonian!(self.spin, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), negative_R.view()).unwrap();
        }
    }

    #[allow(non_snake_case)]
    pub fn add_element(
        &mut self,
        tmp: Complex<f64>,
        ind_i: usize,
        ind_j: usize,
        R: &Array1<isize>,
    ) {
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp , 不考虑自旋, 直接添加参数
        if R.len() != self.dim_r() {
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i >= self.nsta() || ind_j >= self.nsta() {
            panic!(
                "Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",
                self.norb(),
                ind_i,
                ind_j
            )
        }
        let R_exist = find_R(&self.hamR, &R);
        let negative_R = (-R);
        if R_exist {
            let index = index_R(&self.hamR, &R);
            let index_inv = index_R(&self.hamR, &negative_R);
            self.ham[[index, ind_i, ind_j]] = tmp;
            if index != 0 && ind_i != ind_j {
                self.ham[[index_inv, ind_j, ind_i]] = tmp.conj();
            }
            if ind_i == ind_j && tmp.im != 0.0 && index == 0 {
                panic!(
                    "Wrong, the onsite hopping must be real, but here is {}",
                    tmp
                )
            }
        } else {
            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            new_ham[[ind_i, ind_j]] = tmp;
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), R.view()).unwrap();

            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            new_ham[[ind_j, ind_i]] = tmp.conj();
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), negative_R.view()).unwrap();
        }
    }

    #[allow(non_snake_case)]
    pub fn set_onsite(&mut self, tmp: &Array1<f64>, pauli: isize) {
        //! 直接对对角项进行设置
        if tmp.len() != self.norb() {
            panic!(
                "Wrong, the norb is {}, however, the onsite input's length is {}",
                self.norb(),
                tmp.len()
            )
        }
        for (i, item) in tmp.iter().enumerate() {
            self.set_onsite_one(*item, i, pauli);
        }
    }

    #[allow(non_snake_case)]
    pub fn add_onsite(&mut self, tmp: &Array1<f64>, pauli: isize) {
        //! 直接对对角项进行设置
        if tmp.len() != self.norb() {
            panic!(
                "Wrong, the norb is {}, however, the onsite input's length is {}",
                self.norb(),
                tmp.len()
            )
        }
        let R = Array1::zeros(self.dim_r());
        for (i, item) in tmp.iter().enumerate() {
            //self.set_onsite_one(*item,i,pauli)
            self.add_hop(Complex::new(*item, 0.0), i, i, &R, pauli)
        }
    }
    #[allow(non_snake_case)]
    pub fn set_onsite_one(&mut self, tmp: f64, ind: usize, pauli: isize) {
        //!对  $\bra{i\bm 0}\hat H\ket{i\bm 0}$ 进行设置
        let R = Array1::<isize>::zeros(self.dim_r());
        self.set_hop(Complex::new(tmp, 0.0), ind, ind, &R, pauli)
    }
    pub fn del_hop(&mut self, ind_i: usize, ind_j: usize, R: &Array1<isize>, pauli: isize) {
        //! 删除 $\bra{i\bm 0}\hat H\ket{j\bm R}$
        if R.len() != self.dim_r() {
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i >= self.norb() || ind_j >= self.norb() {
            panic!(
                "Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",
                self.norb(),
                ind_i,
                ind_j
            )
        }
        self.set_hop(Complex::new(0.0, 0.0), ind_i, ind_j, &R, pauli);
    }

    #[allow(non_snake_case)]
    pub fn k_path(&self, path: &Array2<f64>, nk: usize) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        //!根据高对称点来生成高对称路径, 画能带图
        if self.dim_r() == 0 {
            panic!("the k dimension of the model is 0, do not use k_path")
        }
        let n_node: usize = path.len_of(Axis(0));
        if self.dim_r() != path.len_of(Axis(1)) {
            panic!("Wrong, the path's length along 1 dimension must equal to the model's dimension")
        }
        let k_metric = (self.lat.dot(&self.lat.t())).inv().unwrap();
        let mut k_node = Array1::<f64>::zeros(n_node);
        for n in 1..n_node {
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk = path.row(n).to_owned() - path.slice(s![n - 1, ..]).to_owned();
            let a = k_metric.dot(&dk);
            let dklen = dk.dot(&a).sqrt();
            k_node[[n]] = k_node[[n - 1]] + dklen;
        }
        let mut node_index: Vec<usize> = vec![0];
        for n in 1..n_node - 1 {
            let frac = k_node[[n]] / k_node[[n_node - 1]];
            let a = (frac * ((nk - 1) as f64).round()) as usize;
            node_index.push(a)
        }
        node_index.push(nk - 1);
        let mut k_dist = Array1::<f64>::zeros(nk);
        let mut k_vec = Array2::<f64>::zeros((nk, self.dim_r()));
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&path.row(0));
        for n in 1..n_node {
            let n_i = node_index[n - 1];
            let n_f = node_index[n];
            let kd_i = k_node[[n - 1]];
            let kd_f = k_node[[n]];
            let k_i = path.row(n - 1);
            let k_f = path.row(n);
            for j in n_i..n_f + 1 {
                let frac: f64 = ((j - n_i) as f64) / ((n_f - n_i) as f64);
                k_dist[[j]] = kd_i + frac * (kd_f - kd_i);
                k_vec
                    .row_mut(j)
                    .assign(&((1.0 - frac) * k_i.to_owned() + frac * k_f.to_owned()));
            }
        }
        (k_vec, k_dist, k_node)
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn gen_ham<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>,gauge:Gauge) -> Array2<Complex<f64>> {
        //!这个是做傅里叶变换, 将实空间的哈密顿量变换到倒空间的哈密顿量
        //!
        //!具体来说, 就是
        //!
        //!对于wannier 基函数 $\ket{\bm R,i}$, 我们和布洛赫函数 $\ket{\bm k,i}$ 的变换形式为
        //!$$\ket{\bm k,i}=\sum_{\bm R} e^{i\bm k\cdot(\bm R+\bm\tau_i)}\ket{\bm R}$$
        //!这样, 我们有
        //!$$\\begin{aligned}H_{mn,\bm k}&=\bra{m\bm k}\hat H\ket{n\bm k}=\sum_{\bm R^\prime}\sum_{\bm R} \bra{m\bm R^\prime}\hat H\ket{n\bm R}e^{-i(\bm R'-\bm R+\bm\tau_m-\bm \tau_n)\cdot\bm k}\\\\
        //!&=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R-\bm\tau_m+\bm \tau_n)\cdot\bm k}
        //!\\end{aligned}$$
        assert!(
            kvec.len() == self.dim_r(),
            "Wrong, the k-vector's length must equal to the dimension of model."
        );

        let Us = (self.hamR.mapv(|x| x as f64))
            .dot(kvec)
            .mapv(|x| Complex::<f64>::new(x, 0.0));
        let Us = Us * Complex::new(0.0, 2.0 * PI);
        let Us = Us.mapv(Complex::exp);
        let hamk: Array2<Complex<f64>> = self
            .ham
            .outer_iter()
            .zip(Us.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (hm, u)| {
                acc + &hm * *u
            });

        let hamk=match gauge{
            Gauge::Lattice => hamk,
            Gauge::Atom => {
                let U0: Array1<f64> = self.orb.dot(kvec);
                let U0: Array1<Complex<f64>> = U0.mapv(|x| Complex::<f64>::new(x, 0.0));
                let U0 = U0 * Complex::new(0.0, 2.0 * PI);
                let mut U0: Array1<Complex<f64>> = U0.mapv(Complex::exp);
                let U0 = if self.spin {
                    let U0 = concatenate![Axis(0), U0, U0];
                    U0
                } else {
                    U0
                };
                let U = Array2::from_diag(&U0);
                let U_dag = Array2::from_diag(&U0.map(|x| x.conj()));
                //接下来两步填上轨道坐标导致的相位
                let hamk: Array2<Complex<f64>> = U_dag.dot(&hamk);
                let re_ham = hamk.dot(&U); 
                re_ham
            },
        };
        hamk
    }
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
    ///这里的 $\\mathcal A_{\bm k}$ 的定义为 $$\\mathcal A_{\bm k,\ap,mn}=-i\sum_{\bm R}r_{mn,\ap}(\bm R)e^{i\bm k\cdot(\bm R-\bm\tau_m+\bm\tau_{n})}+i\tau_{n\ap}\dt_{mn}$$
    ///其中 $\bm r_{mn}$ 可以由 wannier90 给出, 只需要设定 write_rmn=ture
    ///在这里, 所有的 $\bm R$, $\bm r$, 以及 $\bm \tau$ 都是以实空间为坐标系.
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    gauge:Gauge) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        //我们采用的不是原子规范, 而是晶格规范
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length {} must equal to the dimension of model {}.",
            kvec.len(),
            self.dim_r()
        );

        let Us = (self.hamR.mapv(|x| x as f64))
            .dot(kvec)
            .mapv(|x| Complex::<f64>::new(x, 0.0));
        let Us = Us * Complex::new(0.0, 2.0 * PI);
        let Us = Us.mapv(Complex::exp); //Us 就是 exp(i k R)
        //定义一个初始化的速度矩阵
        let mut v = Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta())); 
        let R0 = &self.hamR.mapv(|x| Complex::<f64>::new(x as f64, 0.0));
        //R0 是实空间的 hamR
        let R0 = R0.dot(&self.lat.mapv(|x| Complex::new(x, 0.0)));
        let hamk: Array2<Complex<f64>> = self
            .ham
            .outer_iter()
            .zip(Us.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (hm, u)| {
                acc + &hm * *u
            });
        let (v,hamk)=match gauge{
            Gauge::Atom =>{
                let orb_sta = if self.spin {
                    let orb0 = concatenate(Axis(0), &[self.orb.view(), self.orb.view()]).unwrap();
                    orb0
                } else {
                    self.orb.to_owned()
                };
                let U0 = orb_sta.dot(kvec);
                let U0 = U0.mapv(|x| Complex::<f64>::new(x, 0.0));
                let U0 = U0 * Complex::new(0.0, 2.0 * PI);
                let mut U0 = U0.mapv(Complex::exp);
                //U0 是相位因子
                let U = Array2::from_diag(&U0);
                let U_conj = Array2::from_diag(&U0.mapv(|x| x.conj()));
                let orb_real = orb_sta.dot(&self.lat);
                //开始构建 -orb_real[[i,r]]+orb_real[[j,r]];-----------------
                let mut UU = Array3::<f64>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                let A = orb_real.view().insert_axis(Axis(2));
                let A = A
                    .broadcast((self.nsta(), self.dim_r(), self.nsta()))
                    .unwrap()
                    .permuted_axes([1, 0, 2]);
                let mut B = A.view().permuted_axes([0, 2, 1]);
                let UU = &B - &A;
                let UU = UU.mapv(|x| Complex::<f64>::new(0.0, x)); //UU[i,j]=i(-tau[i]+tau[j])
                //定义一个初始化的速度矩阵
                Zip::from(v.outer_iter_mut())
                    .and(R0.axis_iter(Axis(1)))
                    .and(UU.outer_iter())
                    .for_each(|mut v0, r, det_tau| {
                        let vv: Array2<Complex<f64>> =
                            self.ham.outer_iter().zip(Us.iter().zip(r.iter())).fold(
                                Array2::zeros((self.nsta(), self.nsta())),
                                |acc, (ham, (us, r0))| acc + &ham * *us * *r0 * Complex::i(),
                            );
                        let vv: Array2<Complex<f64>> = &vv + &hamk * &det_tau;
                        let vv = &U_conj.dot(&vv);
                        let vv = vv.dot(&U); //接下来两步填上轨道坐标导致的相位
                        v0.assign(&vv);
                    });
                //到这里, 我们完成了 sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)} 的计算
                //接下来, 我们计算贝利联络 A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                let hamk = U_conj.dot(&hamk.dot(&U)); //这一步别忘了把hamk 的相位加上
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let mut rk = Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                    let mut rk = self
                        .rmatrix
                        .axis_iter(Axis(0))
                        .zip(Us.iter())
                        .fold(rk, |acc, (ham, us)| acc + &ham * *us);
                    for i in 0..3 {
                        let mut r0: ArrayViewMut2<Complex<f64>> = rk.slice_mut(s![i, .., ..]);
                        let r_new = r0.dot(&U);
                        let r_new = U_conj.dot(&r_new);
                        r0.assign(&r_new);
                        let mut dig = r0.diag_mut();
                        //dig.assign(&(&dig - &orb_real.column(i)));
                        dig.assign(&Array1::zeros(self.nsta()));
                        let A = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&A);
                    }
                }
                (v, hamk)
            },
            Gauge::Lattice =>{
                Zip::from(v.outer_iter_mut())
                    .and(R0.axis_iter(Axis(1)))
                    .for_each(|mut v0, r| {
                        let vv: Array2<Complex<f64>> =
                            self.ham.outer_iter().zip(Us.iter().zip(r.iter())).fold(
                                Array2::zeros((self.nsta(), self.nsta())),
                                |acc, (ham, (us, r0))| acc + &ham * *us * *r0 * Complex::i(),
                            );
                        v0.assign(&vv);
                    });
                //到这里, 我们完成了 sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)} 的计算
                //接下来, 我们计算贝利联络 A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let mut rk = Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                    let mut rk = self .rmatrix .axis_iter(Axis(0)) .zip(Us.iter()) .fold(rk, |acc, (ham, us)| acc + &ham * *us);
                    for i in 0..3 {
                        let mut r0: ArrayViewMut2<Complex<f64>> = rk.slice_mut(s![i, .., ..]);
                        let mut dig=r0.diag_mut();
                        dig.assign(&Array1::zeros(self.nsta()));
                        let A = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&A);
                    }
                }
                (v, hamk)
            },
        };
        (v,hamk)
    }

    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn solve_band_onek<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>) -> Array1<f64> {
        //!求解单个k点的能带值
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(kvec,Gauge::Atom);
        //let eval = if let Ok(eigvals) = hamk.eigvalsh(UPLO::Lower) { eigvals } else { todo!() };
        let eval = eigvalsh_v(&hamk, UPLO::Upper);
        eval
    }

    pub fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> Array1<f64> {
        ///这个是用来求解部分能带的算法, 可以加快求解速度, 尤其是求解角态或者hing state.
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec,Gauge::Atom);
        let eval = eigvalsh_r(&hamk, range, epsilon, UPLO::Upper);
        eval
    }
    pub fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Array2<f64> {
        //!求解多个k点的能带值
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .for_each(|x, mut a| {
                let eval = self.solve_band_onek(&x);
                a.assign(&eval);
            });
        band
    }
    #[allow(non_snake_case)]
    pub fn solve_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Array2<f64> {
        //!并行求解多个k点的能带值
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .par_for_each(|x, mut a| {
                let eval = self.solve_band_onek(&x);
                a.assign(&eval);
            });
        band
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> (Array1<f64>, Array2<Complex<f64>>) {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec,Gauge::Atom);
        let (eval, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) {
            (eigvals, eigvecs)
        } else {
            todo!()
        };
        let evec =
            conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>, OwnedRepr<Complex<f64>>>(&evec);
        (eval, evec)
    }
    pub fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> (Array1<f64>, Array2<Complex<f64>>) {
        ///这个是用来求解部分能带的算法, 可以加快求解速度, 尤其是求解角态或者hing state.
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec,Gauge::Atom);
        let (eval, evec) = eigh_r(&hamk, range, epsilon, UPLO::Upper);
        let evec = evec.mapv(|x| x.conj());
        (eval, evec)
    }

    #[allow(non_snake_case)]
    pub fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        let mut vectors = Array3::<Complex<f64>>::zeros((nk, self.nsta(), self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .and(vectors.outer_iter_mut())
            .for_each(|x, mut a, mut b| {
                let (eval, evec) = self.solve_onek(&x);
                a.assign(&eval);
                b.assign(&evec);
            });
        (band, vectors)
    }
    #[allow(non_snake_case)]
    pub fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        let nk = kvec.len_of(Axis(0));
        let mut band = Array2::<f64>::zeros((nk, self.nsta()));
        let mut vectors = Array3::<Complex<f64>>::zeros((nk, self.nsta(), self.nsta()));
        Zip::from(kvec.outer_iter())
            .and(band.outer_iter_mut())
            .and(vectors.outer_iter_mut())
            .par_for_each(|x, mut a, mut b| {
                let (eval, evec) = self.solve_onek(&x);
                a.assign(&eval);
                b.assign(&evec);
            });
        (band, vectors)
    }

    ///这个函数是用来将model的某个方向进行截断的
    ///
    ///num:截出多少个原胞
    ///
    ///dir:方向
    ///
    ///返回一个model, 其中 dir 和输入的model是一致的, 但是轨道数目和原子数目都会扩大num倍, 沿着dir方向没有胞间hopping.
    pub fn cut_piece(&self, num: usize, dir: usize) -> Model {
        //! This function is used to truncate a certain direction of a model.
        //!
        //! Parameters:
        //! - num: number of unit cells to truncate.
        //! - dir: the direction to be truncated.
        //!
        //! Returns a new model with the same direction as the input model, but with the number of orbitals and atoms increased by a factor of "num". There is no inter-cell hopping along the "dir" direction.
        if num < 1 {
            panic!("Wrong, the num={} is less than 1", num);
        }
        if dir > self.dim_r() {
            panic!("Wrong, the dir is larger than dim_r");
        }
        let mut new_orb = Array2::<f64>::zeros((self.norb() * num, self.dim_r())); //定义一个新的轨道
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new(); //定义一个新的原子
        let new_norb = self.norb() * num;
        let new_nsta = self.nsta() * num;
        let new_natom = self.natom() * num;
        let mut new_lat = self.lat.clone();
        new_lat
            .row_mut(dir)
            .assign(&(self.lat.row(dir).to_owned() * (num as f64)));
        for i in 0..num {
            for n in 0..self.norb() {
                let mut use_orb = self.orb.row(n).to_owned();
                use_orb[[dir]] += i as f64;
                use_orb[[dir]] = use_orb[[dir]] / (num as f64);
                new_orb.row_mut(i * self.norb() + n).assign(&use_orb);
                new_orb_proj.push(self.orb_projection[n]);
            }
            for n in 0..self.natom() {
                let mut use_atom_position = self.atoms[n].position();
                use_atom_position[[dir]] += i as f64;
                use_atom_position[[dir]] *= 1.0 / (num as f64);
                let use_atom = Atom::new(
                    use_atom_position,
                    self.atoms[n].norb(),
                    self.atoms[n].atom_type(),
                );
                new_atom.push(use_atom);
            }
        }
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, new_nsta, new_nsta));
        let mut new_rmatrix = Array4::<Complex<f64>>::zeros((1, self.dim_r(), new_nsta, new_nsta));
        let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
        //新的轨道和原子构造完成, 开始构建哈密顿量
        //let new_model=tb_model(self.dim_r(),new_norb,new_lat,new_orb,self.spin,Some(new_natom),Some(new_atom),Some(atom_list));
        //先尝试构建位置函数
        let exist_r = self.rmatrix.len_of(Axis(0)) != 1;
        if exist_r == false {
            for n in 0..num {
                for i in 0..new_norb {
                    for r in 0..self.dim_r() {
                        new_rmatrix[[0, r, i, i]] = Complex::new(new_orb[[i, r]], 0.0);
                        if self.spin {
                            new_rmatrix[[0, r, i + new_norb, i + new_norb]] =
                                Complex::new(new_orb[[i, r]], 0.0);
                        }
                    }
                }
            }

            //接下来, 我们创建一个新的哈密顿量, 如果 hamR[[dir]]<0, 那么就将哈密顿量反号,
            //保证沿着切的方向的哈密顿量是正方向hopping的
            let n_R = self.hamR.len_of(Axis(0));
            let mut using_ham = self.ham.clone();
            let mut using_hamR = self.hamR.clone();
            /*
            Zip::from(using_ham.outer_iter_mut())
                .and(using_hamR.outer_iter_mut())
                .par_for_each(|mut ham, mut R| {
                    if R[[dir]] < 0 {
                        let h0 = conjugate::<
                            Complex<f64>,
                            OwnedRepr<Complex<f64>>,
                            OwnedRepr<Complex<f64>>,
                        >(&ham.to_owned());
                        ham.assign(&h0);
                        R.assign(&(-&R));
                    }
                });
            */
            for n in 0..num {
                for (i0, (ind_R, ham)) in using_hamR
                    .outer_iter()
                    .zip(using_ham.outer_iter())
                    .enumerate()
                {
                    let ind: usize = (ind_R[[dir]] + (n as isize)) as usize;
                    let mut ind_R = ind_R.to_owned();
                    let ham = ham.to_owned();
                    ind_R[[dir]] = 0;
                    if ind < num {
                        //开始构建哈密顿量
                        let R_exist = find_R(&new_hamR, &ind_R);
                        let mut use_ham = Array2::<Complex<f64>>::zeros((new_nsta, new_nsta));
                        if self.spin {
                            //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);

                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 =
                                ham.slice(s![self.norb()..self.nsta(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            /*
                            if R_exist {
                                let index = index_R(&new_hamR, &ind_R);
                                if index == 0 && ind != 0 {
                                    let ham = conjugate::<
                                        Complex<f64>,
                                        OwnedRepr<Complex<f64>>,
                                        OwnedRepr<Complex<f64>>,
                                    >(&ham);
                                    let mut s = use_ham.slice_mut(s![
                                        ind * self.norb()..(ind + 1) * self.norb(),
                                        n * self.norb()..(n + 1) * self.norb()
                                    ]);
                                    let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                                    s.assign(&ham0);

                                    let mut s = use_ham.slice_mut(s![
                                        new_norb + ind * self.norb()
                                            ..new_norb + (ind + 1) * self.norb(),
                                        n * self.norb()..(n + 1) * self.norb()
                                    ]);
                                    let ham0 =
                                        ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                                    s.assign(&ham0);
                                    let mut s = use_ham.slice_mut(s![
                                        ind * self.norb()..(ind + 1) * self.norb(),
                                        new_norb + n * self.norb()
                                            ..new_norb + (n + 1) * self.norb()
                                    ]);
                                    let ham0 =
                                        ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                                    s.assign(&ham0);
                                    let mut s = use_ham.slice_mut(s![
                                        new_norb + ind * self.norb()
                                            ..new_norb + (ind + 1) * self.norb(),
                                        new_norb + n * self.norb()
                                            ..new_norb + (n + 1) * self.norb()
                                    ]);
                                    let ham0 = ham.slice(s![
                                        self.norb()..self.nsta(),
                                        self.norb()..self.nsta()
                                    ]);
                                    s.assign(&ham0);
                                }
                            }
                            */
                        } else {
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);
                            /*
                            if R_exist {
                                let index = index_R(&new_hamR, &ind_R);
                                if index == 0 && ind != 0 {
                                    let ham = conjugate::<
                                        Complex<f64>,
                                        OwnedRepr<Complex<f64>>,
                                        OwnedRepr<Complex<f64>>,
                                    >(&ham);
                                    let mut s = use_ham.slice_mut(s![
                                        ind * self.norb()..(ind + 1) * self.norb(),
                                        n * self.norb()..(n + 1) * self.norb()
                                    ]);
                                    let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                                    s.assign(&ham0);
                                }
                            }
                            */
                        }
                        if R_exist {
                            let index = index_R(&new_hamR, &ind_R);
                            new_ham.slice_mut(s![index, .., ..]).add_assign(&use_ham);
                            /*
                            } else if negative_R_exist {
                                let index = index_R(&new_hamR, &negative_R);
                                new_ham
                                    .slice_mut(s![index, .., ..])
                                    .add_assign(&use_ham.t().map(|x| x.conj()));
                                */
                        } else {
                            new_ham.push(Axis(0), use_ham.view());
                            new_hamR.push(Axis(0), ind_R.view());
                        }
                    } else {
                        continue;
                    }
                }
            }
        } else {
            //接下来, 我们创建一个新的哈密顿量, 如果 hamR[[dir]]<0, 那么就将哈密顿量反号,
            //保证沿着切的方向的哈密顿量是正方向hopping的
            let n_R = self.hamR.len_of(Axis(0));
            let mut using_ham = self.ham.clone();
            let mut using_hamR = self.hamR.clone();
            let mut using_rmatrix = self.rmatrix.clone();
            Zip::from(using_ham.outer_iter_mut())
                .and(using_hamR.outer_iter_mut())
                .and(using_rmatrix.outer_iter_mut())
                .par_for_each(|mut ham, mut R, mut rmatrix| {
                    if R[[dir]] < 0 {
                        let h0 = conjugate::<
                            Complex<f64>,
                            OwnedRepr<Complex<f64>>,
                            OwnedRepr<Complex<f64>>,
                        >(&ham.to_owned());
                        ham.assign(&h0);
                        R.assign(&(-&R));
                        let mut r0 = rmatrix.map(|x| x.conj());
                        r0.to_owned().swap_axes(1, 2);
                        rmatrix.assign(&r0);
                    }
                });
            for n in 0..num {
                for (i0, (ind_R, (ham, rmatrix))) in using_hamR
                    .outer_iter()
                    .zip(using_ham.outer_iter().zip(using_rmatrix.outer_iter()))
                    .enumerate()
                {
                    let ind: usize = (ind_R[[dir]] + (n as isize)) as usize;
                    let mut ind_R = ind_R.to_owned();
                    let ham = ham.to_owned();
                    let rmatrix = rmatrix.to_owned();
                    ind_R[[dir]] = 0;
                    if ind < num {
                        //开始构建哈密顿量
                        let R_exist = find_R(&new_hamR, &ind_R);
                        let negative_R = -ind_R.clone();
                        let negative_R_exist = find_R(&new_hamR, &negative_R);
                        let mut use_ham = Array2::<Complex<f64>>::zeros((new_nsta, new_nsta));
                        if self.spin {
                            //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);

                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 =
                                ham.slice(s![self.norb()..self.nsta(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            /*
                            if R_exist {
                                let index = index_R(&new_hamR, &ind_R);
                                if index == 0 && ind != 0 {
                                    let ham = conjugate::<
                                        Complex<f64>,
                                        OwnedRepr<Complex<f64>>,
                                        OwnedRepr<Complex<f64>>,
                                    >(&ham);
                                    let mut s = use_ham.slice_mut(s![
                                        ind * self.norb()..(ind + 1) * self.norb(),
                                        n * self.norb()..(n + 1) * self.norb()
                                    ]);
                                    let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                                    s.assign(&ham0);

                                    let mut s = use_ham.slice_mut(s![
                                        new_norb + ind * self.norb()
                                            ..new_norb + (ind + 1) * self.norb(),
                                        n * self.norb()..(n + 1) * self.norb()
                                    ]);
                                    let ham0 =
                                        ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                                    s.assign(&ham0);
                                    let mut s = use_ham.slice_mut(s![
                                        ind * self.norb()..(ind + 1) * self.norb(),
                                        new_norb + n * self.norb()
                                            ..new_norb + (n + 1) * self.norb()
                                    ]);
                                    let ham0 =
                                        ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                                    s.assign(&ham0);
                                    let mut s = use_ham.slice_mut(s![
                                        new_norb + ind * self.norb()
                                            ..new_norb + (ind + 1) * self.norb(),
                                        new_norb + n * self.norb()
                                            ..new_norb + (n + 1) * self.norb()
                                    ]);
                                    let ham0 = ham.slice(s![
                                        self.norb()..self.nsta(),
                                        self.norb()..self.nsta()
                                    ]);
                                    s.assign(&ham0);
                                }
                            }
                            */
                        } else {
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);
                            /*
                            if R_exist {
                                let index = index_R(&new_hamR, &ind_R);
                                if index == 0 && ind != 0 {
                                    let ham = conjugate::<
                                        Complex<f64>,
                                        OwnedRepr<Complex<f64>>,
                                        OwnedRepr<Complex<f64>>,
                                    >(&ham);
                                    let mut s = use_ham.slice_mut(s![
                                        ind * self.norb()..(ind + 1) * self.norb(),
                                        n * self.norb()..(n + 1) * self.norb()
                                    ]);
                                    let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                                    s.assign(&ham0);
                                }
                            }
                            */
                        }
                        //开始对 r_matrix 进行操作
                        let mut use_rmatrix =
                            Array3::<Complex<f64>>::zeros((self.dim_r(), new_nsta, new_nsta));
                        if exist_r {
                            if self.spin {
                                //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    n * self.norb()..(n + 1) * self.norb(),
                                    ind * self.norb()..(ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., 0..self.norb(), 0..self.norb()]);
                                s.assign(&rmatrix0);

                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                    ind * self.norb()..(ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., self.norb()..self.nsta(), 0..self.norb()]);
                                s.assign(&rmatrix0);
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    n * self.norb()..(n + 1) * self.norb(),
                                    new_norb + ind * self.norb()
                                        ..new_norb + (ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., 0..self.norb(), self.norb()..self.nsta()]);
                                s.assign(&rmatrix0);
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                    new_norb + ind * self.norb()
                                        ..new_norb + (ind + 1) * self.norb()
                                ]);
                                let rmatrix0 = rmatrix.slice(s![
                                    ..,
                                    self.norb()..self.nsta(),
                                    self.norb()..self.nsta()
                                ]);
                                s.assign(&rmatrix0);
                                /*
                                if R_exist {
                                    let index = index_R(&new_hamR, &ind_R);
                                    if index == 0 && ind != 0 {
                                        let mut rmatrix = rmatrix.mapv(|x| x.conj());
                                        rmatrix.swap_axes(1, 2);
                                        let mut s = use_rmatrix.slice_mut(s![
                                            ..,
                                            ind * self.norb()..(ind + 1) * self.norb(),
                                            n * self.norb()..(n + 1) * self.norb()
                                        ]);
                                        let rmatrix0 =
                                            rmatrix.slice(s![.., 0..self.norb(), 0..self.norb()]);
                                        s.assign(&rmatrix0);

                                        let mut s = use_rmatrix.slice_mut(s![
                                            ..,
                                            new_norb + ind * self.norb()
                                                ..new_norb + (ind + 1) * self.norb(),
                                            n * self.norb()..(n + 1) * self.norb()
                                        ]);
                                        let rmatrix0 = rmatrix.slice(s![
                                            ..,
                                            self.norb()..self.nsta(),
                                            0..self.norb()
                                        ]);
                                        s.assign(&rmatrix0);
                                        let mut s = use_rmatrix.slice_mut(s![
                                            ..,
                                            ind * self.norb()..(ind + 1) * self.norb(),
                                            new_norb + n * self.norb()
                                                ..new_norb + (n + 1) * self.norb()
                                        ]);
                                        let rmatrix0 = rmatrix.slice(s![
                                            ..,
                                            0..self.norb(),
                                            self.norb()..self.nsta()
                                        ]);
                                        s.assign(&rmatrix0);
                                        let mut s = use_rmatrix.slice_mut(s![
                                            ..,
                                            new_norb + ind * self.norb()
                                                ..new_norb + (ind + 1) * self.norb(),
                                            new_norb + n * self.norb()
                                                ..new_norb + (n + 1) * self.norb()
                                        ]);
                                        let rmatrix0 = rmatrix.slice(s![
                                            ..,
                                            self.norb()..self.nsta(),
                                            self.norb()..self.nsta()
                                        ]);
                                        s.assign(&rmatrix0);
                                    }
                                }
                                */
                            } else {
                                for i in 0..self.norb() {
                                    for j in 0..self.norb() {
                                        for r in 0..self.dim_r() {
                                            use_rmatrix
                                                [[r, i + n * self.norb(), j + ind * self.norb()]] =
                                                rmatrix[[r, i, j]];
                                        }
                                    }
                                }
                                /*
                                if R_exist {
                                    let index = index_R(&new_hamR, &ind_R);
                                    if index == 0 && ind != 0 {
                                        for i in 0..self.norb() {
                                            for j in 0..self.norb() {
                                                for r in 0..self.dim_r() {
                                                    use_rmatrix[[
                                                        r,
                                                        j + ind * self.norb(),
                                                        i + n * self.norb(),
                                                    ]] = rmatrix[[r, i, j]].conj();
                                                }
                                            }
                                        }
                                    }
                                }
                                */
                            }
                        }
                        if R_exist {
                            let index = index_R(&new_hamR, &ind_R);
                            //let addham=new_ham.slice(s![index,..,..]).to_owned();
                            new_ham.slice_mut(s![index, .., ..]).add_assign(&use_ham);
                            //let addr=new_rmatrix.slice(s![index,..,..,..]).to_owned();
                            new_rmatrix
                                .slice_mut(s![index, .., .., ..])
                                .add_assign(&use_rmatrix);
                            /*
                            } else if negative_R_exist {
                                let index = index_R(&new_hamR, &negative_R);
                                //let addham=new_ham.slice(s![index,..,..]).to_owned();
                                new_ham
                                    .slice_mut(s![index, .., ..])
                                    .add_assign(&use_ham.t().map(|x| x.conj()));
                                //let mut addr=new_rmatrix.slice(s![index,..,..,..]).to_owned();
                                use_rmatrix.swap_axes(1, 2);
                                new_rmatrix
                                    .slice_mut(s![index, .., .., ..])
                                    .add_assign(&use_rmatrix.map(|x| x.conj()));
                                */
                        } else {
                            new_ham.push(Axis(0), use_ham.view());
                            new_hamR.push(Axis(0), ind_R.view());
                            new_rmatrix.push(Axis(0), use_rmatrix.view());
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
        let mut model = Model {
            dim_r: self.dim_r(),
            spin: self.spin,
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_proj,
            atoms: new_atom,
            ham: new_ham,
            hamR: new_hamR,
            rmatrix: new_rmatrix,
        };
        model
    }
    pub fn shift_to_atom(&mut self) {
        //!这个是将轨道移动到原子位置上
        let mut a = 0;
        for (i, atom) in self.atoms.iter().enumerate() {
            for j in 0..atom.norb() {
                self.orb.row_mut(a).assign(&atom.position());
                a += 1;
            }
        }
    }
    pub fn cut_dot(&self, num: usize, shape: usize, dir: Option<Vec<usize>>) -> Model {
        //! 这个是用来且角态或者切棱态的

        match self.dim_r() {
            3 => {
                let dir = if dir == None {
                    eprintln!(
                        "Wrong!, the dir is None, but model's dimension is 3, here we use defult 0,1 direction"
                    );
                    let dir = vec![0, 1];
                    dir
                } else {
                    dir.unwrap()
                };
                let (old_model, use_orb_item, use_atom_item) = {
                    let model_1 = self.cut_piece(num + 1, dir[0]);
                    let model_2 = model_1.cut_piece(num + 1, dir[1]);
                    let mut use_atom_item = Vec::<usize>::new();
                    let mut use_orb_item = Vec::<usize>::new(); //这个是确定要保留哪些轨道
                    let mut a: usize = 0;
                    match shape {
                        3 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[dir[0]]] + atom_position[[dir[1]]]
                                    > (num as f64) / (num as f64 + 1.0)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        4 => {
                            let num0 = num as f64;
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[dir[0]]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                    || atom_position[[dir[1]]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        6 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[0]] - atom_position[[dir[1]]] > 0.5)
                                    || (atom_position[[dir[0]]] - atom_position[[1]] < -0.5)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        8 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[dir[1]]] - atom_position[[dir[0]]] + 0.5 < 0.0)
                                    || (atom_position[[dir[0]]] - atom_position[[dir[1]]] + 0.5
                                        < 0.0)
                                    || (atom_position[[dir[1]]] + atom_position[[dir[0]]] < 0.5)
                                    || (atom_position[[dir[1]]] - atom_position[[dir[0]]] > 0.5)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        _ => {
                            panic!("Wrong, the shape only can be 3,4, 6,8");
                            todo!();
                        }
                    };
                    (model_2, use_orb_item, use_atom_item)
                };
                let natom = use_atom_item.len();
                let norb = use_orb_item.len();
                let mut new_atom = Vec::new();
                let mut new_orb = Array2::<f64>::zeros((norb, self.dim_r()));
                let mut new_orb_proj = Vec::new();
                for (i, use_i) in use_atom_item.iter().enumerate() {
                    new_atom.push(old_model.atoms[*use_i].clone());
                }
                for (i, use_i) in use_orb_item.iter().enumerate() {
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                    new_orb_proj.push(old_model.orb_projection[*use_i])
                }
                let mut new_model = Model::tb_model(
                    self.dim_r(),
                    old_model.lat.clone(),
                    new_orb,
                    self.spin,
                    Some(new_atom),
                );
                new_model.orb_projection = new_orb_proj;
                let n_R = old_model.hamR.len_of(Axis(0));
                let mut new_ham =
                    Array3::<Complex<f64>>::zeros((n_R, new_model.nsta(), new_model.nsta()));
                let mut new_hamR = Array2::<isize>::zeros((0, self.dim_r()));
                let norb = new_model.norb();

                if self.spin {
                    let norb2 = old_model.norb();
                    for (r, R) in old_model.hamR.axis_iter(Axis(0)).enumerate() {
                        new_hamR.push_row(R);
                        for (i, use_i) in use_orb_item.iter().enumerate() {
                            for (j, use_j) in use_orb_item.iter().enumerate() {
                                new_ham[[r, i, j]] = old_model.ham[[r, *use_i, *use_j]];
                                new_ham[[r, i + norb, j + norb]] =
                                    old_model.ham[[r, *use_i + norb2, *use_j + norb2]];
                                new_ham[[r, i + norb, j]] =
                                    old_model.ham[[r, *use_i + norb2, *use_j]];
                                new_ham[[r, i, j + norb]] =
                                    old_model.ham[[r, *use_i, *use_j + norb2]];
                            }
                        }
                    }
                } else {
                    for (r, R) in old_model.hamR.axis_iter(Axis(0)).enumerate() {
                        new_hamR.push_row(R);
                        for (i, use_i) in use_orb_item.iter().enumerate() {
                            for (j, use_j) in use_orb_item.iter().enumerate() {
                                new_ham[[r, i, j]] = old_model.ham[[r, *use_i, *use_j]];
                            }
                        }
                    }
                }
                new_model.ham = new_ham;
                new_model.hamR = new_hamR;
                let nsta = new_model.nsta();
                if self.rmatrix.len_of(Axis(0)) == 1 {
                    for r in 0..self.dim_r() {
                        let mut use_rmatrix = Array2::<Complex<f64>>::zeros((nsta, nsta));
                        for i in 0..norb {
                            use_rmatrix[[i, i]] = Complex::new(new_model.orb[[i, r]], 0.0);
                        }
                        if new_model.spin {
                            for i in 0..norb {
                                use_rmatrix[[i + norb, i + norb]] =
                                    Complex::new(new_model.orb[[i, r]], 0.0);
                            }
                        }
                        new_model
                            .rmatrix
                            .slice_mut(s![0, r, .., ..])
                            .assign(&use_rmatrix);
                    }
                } else {
                    let mut new_rmatrix = Array4::<Complex<f64>>::zeros((
                        n_R,
                        self.dim_r(),
                        new_model.nsta(),
                        new_model.nsta(),
                    ));
                    if old_model.spin {
                        let norb2 = old_model.norb();
                        for r in 0..n_R {
                            for dim in 0..self.dim_r() {
                                for (i, use_i) in use_orb_item.iter().enumerate() {
                                    for (j, use_j) in use_orb_item.iter().enumerate() {
                                        new_rmatrix[[r, dim, i, j]] =
                                            old_model.rmatrix[[r, dim, *use_i, *use_j]];
                                        new_rmatrix[[r, dim, i + norb, j + norb]] = old_model
                                            .rmatrix[[r, dim, *use_i + norb2, *use_j + norb2]];
                                        new_rmatrix[[r, dim, i + norb, j]] =
                                            old_model.rmatrix[[r, dim, *use_i + norb2, *use_j]];
                                        new_rmatrix[[r, dim, i, j + norb]] =
                                            old_model.rmatrix[[r, dim, *use_i, *use_j + norb2]];
                                    }
                                }
                            }
                        }
                    } else {
                        for r in 0..n_R {
                            for dim in 0..self.dim_r() {
                                for (i, use_i) in use_orb_item.iter().enumerate() {
                                    for (j, use_j) in use_orb_item.iter().enumerate() {
                                        new_rmatrix[[r, dim, i, j]] =
                                            old_model.rmatrix[[r, dim, *use_i, *use_j]];
                                    }
                                }
                            }
                        }
                    }
                    new_model.rmatrix = new_rmatrix;
                }
                return new_model;
            }
            2 => {
                if dir != None {
                    eprintln!(
                        "Wrong!, the dimension of model is 2, but the dir is not None, you should give None!, here we use 0,1 direction"
                    );
                }

                let (old_model, use_orb_item, use_atom_item) = {
                    let model_1 = self.cut_piece(num + 1, 0);
                    let model_2 = model_1.cut_piece(num + 1, 1);
                    let mut use_atom_item = Vec::<usize>::new();
                    let mut use_orb_item = Vec::<usize>::new(); //这个是确定要保留哪些轨道
                    let mut a: usize = 0;
                    match shape {
                        3 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[0]] + atom_position[[1]]
                                    > (num as f64) / (num as f64 + 1.0)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        4 => {
                            let num0 = num as f64;
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[0]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                    || atom_position[[1]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        6 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[1]] - atom_position[[0]] + 0.5 < 0.0)
                                    || (atom_position[[0]] - atom_position[[1]] + 0.5 < 0.0)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        8 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[1]] - atom_position[[0]] + 0.5 < 0.0)
                                    || (atom_position[[0]] - atom_position[[1]] + 0.5 < 0.0)
                                    || (atom_position[[1]] + atom_position[[0]] < 0.5)
                                    || (atom_position[[1]] - atom_position[[0]] > 0.5)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        _ => {
                            panic!("Wrong, the shape only can be 3,4, 6,8");
                            todo!();
                        }
                    };
                    (model_2, use_orb_item, use_atom_item)
                };
                let natom = use_atom_item.len();
                let norb = use_orb_item.len();
                let mut new_atom = Vec::new();
                let mut new_orb = Array2::<f64>::zeros((norb, self.dim_r()));
                let mut new_orb_proj = Vec::new();
                for (i, use_i) in use_atom_item.iter().enumerate() {
                    new_atom.push(old_model.atoms[*use_i].clone());
                }
                for (i, use_i) in use_orb_item.iter().enumerate() {
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                    new_orb_proj.push(old_model.orb_projection[*use_i])
                }
                let mut new_model = Model::tb_model(
                    self.dim_r(),
                    old_model.lat.clone(),
                    new_orb,
                    self.spin,
                    Some(new_atom),
                );
                new_model.orb_projection = new_orb_proj;
                let n_R = new_model.hamR.len_of(Axis(0));
                let mut new_ham =
                    Array3::<Complex<f64>>::zeros((n_R, new_model.nsta(), new_model.nsta()));
                let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
                let norb = new_model.norb();
                let nsta = new_model.nsta();

                if self.spin {
                    let norb2 = old_model.norb();
                    for (i, use_i) in use_orb_item.iter().enumerate() {
                        for (j, use_j) in use_orb_item.iter().enumerate() {
                            new_ham[[0, i, j]] = old_model.ham[[0, *use_i, *use_j]];
                            new_ham[[0, i + norb, j + norb]] =
                                old_model.ham[[0, *use_i + norb2, *use_j + norb2]];
                            new_ham[[0, i + norb, j]] = old_model.ham[[0, *use_i + norb2, *use_j]];
                            new_ham[[0, i, j + norb]] = old_model.ham[[0, *use_i, *use_j + norb2]];
                        }
                    }
                } else {
                    for (i, use_i) in use_orb_item.iter().enumerate() {
                        for (j, use_j) in use_orb_item.iter().enumerate() {
                            new_ham[[0, i, j]] = old_model.ham[[0, *use_i, *use_j]];
                        }
                    }
                }
                new_model.ham = new_ham;
                new_model.hamR = new_hamR;
                if self.rmatrix.len_of(Axis(0)) == 1 {
                    for r in 0..self.dim_r() {
                        let mut use_rmatrix = Array2::<Complex<f64>>::zeros((nsta, nsta));
                        for i in 0..norb {
                            use_rmatrix[[i, i]] = Complex::new(new_model.orb[[i, r]], 0.0);
                        }
                        if new_model.spin {
                            for i in 0..norb {
                                use_rmatrix[[i + norb, i + norb]] =
                                    Complex::new(new_model.orb[[i, r]], 0.0);
                            }
                        }
                        new_model
                            .rmatrix
                            .slice_mut(s![0, r, .., ..])
                            .assign(&use_rmatrix);
                    }
                } else {
                    let mut new_rmatrix = Array4::<Complex<f64>>::zeros((
                        n_R,
                        self.dim_r(),
                        new_model.nsta(),
                        new_model.nsta(),
                    ));
                    if old_model.spin {
                        let norb2 = old_model.norb();
                        for dim in 0..self.dim_r() {
                            for (i, use_i) in use_orb_item.iter().enumerate() {
                                for (j, use_j) in use_orb_item.iter().enumerate() {
                                    new_rmatrix[[0, dim, i, j]] =
                                        old_model.rmatrix[[0, dim, *use_i, *use_j]];
                                    new_rmatrix[[0, dim, i + norb, j + norb]] =
                                        old_model.rmatrix[[0, dim, *use_i + norb2, *use_j + norb2]];
                                    new_rmatrix[[0, dim, i + norb, j]] =
                                        old_model.rmatrix[[0, dim, *use_i + norb2, *use_j]];
                                    new_rmatrix[[0, dim, i, j + norb]] =
                                        old_model.rmatrix[[0, dim, *use_i, *use_j + norb2]];
                                }
                            }
                        }
                    } else {
                        for dim in 0..self.dim_r() {
                            for (i, use_i) in use_orb_item.iter().enumerate() {
                                for (j, use_j) in use_orb_item.iter().enumerate() {
                                    new_rmatrix[[0, dim, i, j]] =
                                        old_model.rmatrix[[0, dim, *use_i, *use_j]];
                                }
                            }
                        }
                    }
                    new_model.rmatrix = new_rmatrix;
                }
                return new_model;
            }
            _ => {
                panic!("Wrong, only dim_r=2,3 can using this function!");
                todo!();
            }
        }
    }

    pub fn move_to_atom(&mut self) {
        ///This function moves the orbital position to the atomic position
        let mut a = 0;
        for i in 0..self.natom() {
            for j in 0..self.atoms[i].norb() {
                self.orb.row_mut(a).assign(&self.atoms[i].position());
                a += 1;
            }
        }
    }
    pub fn remove_orb(&mut self, orb_list: &Vec<usize>) {
        let mut use_orb_list = orb_list.clone();
        use_orb_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates = { use_orb_list.windows(2).any(|window| window[0] == window[1]) };
        if has_duplicates {
            panic!("Wrong, make sure no duplicates in orb_list");
        }
        let mut index: Vec<_> = (0..=self.norb() - 1)
            .filter(|&num| !use_orb_list.contains(&num))
            .collect(); //要保留下来的元素
        let delete_n = orb_list.len();
        self.orb = self.orb.select(Axis(0), &index);
        let mut new_orb_proj = Vec::new();
        for i in index.iter() {
            new_orb_proj.push(self.orb_projection[*i])
        }
        self.orb_projection = new_orb_proj;

        let mut b = 0;
        for (i, a) in self.atoms.clone().iter().enumerate() {
            b += a.norb();
            while b > use_orb_list[0] {
                self.atoms[i].remove_orb();
                let _ = use_orb_list.remove(0);
            }
        }
        self.atoms.retain(|x| x.norb() != 0);
        //开始计算nsta
        if self.spin {
            let index_add: Vec<_> = index.iter().map(|x| *x + self.norb()).collect();
            index.extend(index_add);
        }
        let mut b = 0;
        //开始操作哈密顿量
        let new_ham = self.ham.select(Axis(1), &index);
        let new_ham = new_ham.select(Axis(2), &index);
        self.ham = new_ham;
        //开始操作rmatrix
        let new_rmatrix = self.rmatrix.select(Axis(2), &index);
        let new_rmatrix = new_rmatrix.select(Axis(3), &index);
        self.rmatrix = new_rmatrix;
    }

    pub fn remove_atom(&mut self, atom_list: &Vec<usize>) {
        //----------判断是否存在重复, 并给出保留的index
        let mut use_atom_list = atom_list.clone();
        use_atom_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates = {
            use_atom_list
                .windows(2)
                .any(|window| window[0] == window[1])
        };
        if has_duplicates {
            panic!("Wrong, make sure no duplicates in orb_list");
        }

        let mut atom_index: Vec<_> = (0..=self.natom() - 1)
            .filter(|&num| !use_atom_list.contains(&num))
            .collect(); //要保留下来的元素

        let new_atoms = {
            let mut new_atoms = Vec::new();
            for i in atom_index.iter() {
                new_atoms.push(self.atoms[*i].clone());
            }
            new_atoms
        }; //选出需要的原子以及需要的轨道
        //接下来选择需要的轨道

        let mut b = 0;
        let mut orb_index = Vec::new(); //要保留下来的轨道
        let atom_list = self.atom_list();
        let mut int_atom_list = Array1::zeros(self.natom());
        int_atom_list[[0]] = 0;
        for i in 1..self.natom() {
            int_atom_list[[i]] = int_atom_list[[i - 1]] + atom_list[i - 1];
        }
        for i in atom_index.iter() {
            for j in 0..self.atoms[*i].norb() {
                orb_index.push(int_atom_list[[*i]] + j);
            }
        }
        let norb = self.norb(); //保留之前的norb
        self.orb = self.orb.select(Axis(0), &orb_index);
        self.atoms = new_atoms;

        let mut new_orb_proj = Vec::new();
        for i in orb_index.iter() {
            new_orb_proj.push(self.orb_projection[*i])
        }
        self.orb_projection = new_orb_proj;
        if self.spin {
            let index_add: Vec<_> = orb_index.iter().map(|x| *x + norb).collect();
            orb_index.extend(index_add);
        }
        //开始操作哈密顿量
        let new_ham = self.ham.select(Axis(1), &orb_index);
        let new_ham = new_ham.select(Axis(2), &orb_index);
        self.ham = new_ham;
        //开始操作rmatrix
        let new_rmatrix = self.rmatrix.select(Axis(2), &orb_index);
        let new_rmatrix = new_rmatrix.select(Axis(3), &orb_index);
        self.rmatrix = new_rmatrix;
    }

    pub fn unfold(
        &self,
        U: &Array2<f64>,
        path: &Array2<f64>,
        nk: usize,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        eta: f64,
        precision: f64,
    ) -> Array2<f64> {
        //! 能带反折叠算法, 用来计算能带反折叠后的能带. 可以用来计算合金以及一些超胞
        //! 算法参考
        //!
        //! 首先, 我们定义超胞布里渊区下的哈密顿量 $H_{\\bm K}$ 以及其格林函数 $$G(\og,\bm K)=(\og+i\eta-H_{\bm K})^{-1}$$
        //!
        //! 这里 $H_{\bm k}$ 是超胞的哈密顿量. 其本征值和本征态为 $\ve_{N\bm K}$ 和 $\bra{\psi_{N\bm K}}$
        //!
        //! 故我们可以在本征态下将格林函数写为 $$G(\og,\bm K)=\sum_{N}\f{\dyad{\psi_{N\bm K}}}{\og+i\eta-\ve_{N\bm K}}$$
        //!
        //! 再利用普函数定理, 有 $A(\og,\bm K)=-\f{1}{\pi}\Im G(\og,\bm K)$, 对其求trace, 我们就能画超胞的能谱.
        //!
        //! 但是, 我们希望得到的是原胞的能谱, 所以我们需要得到原胞的基, 即 $\ket{n\bm k}$.
        //!
        //! 反折叠后的能谱为 $$A_{nn}(\og,\bm k)=\sum_{N\bm K}\lt\\vert \braket{n\bm k}{\psi_{N\bm K}}\rt\\vert^2 A_{NN}(\og,\bm K)$$
        //!
        //!接下来我们计算 $\braket{n\bm k}{\psi_{N\bm K}}$
        //!
        //!首先, 我们有$$ \lt\\{
        //!\\begin{aligned}
        //!\ket{N\bm K}&=\f{1}{\sqrt{V}}\sum_{\bm R}e^{-i\bm K\cdot(\bm R+\bm\tau_N)}\ket{N\bm R}\\\\
        //!\ket{n\bm k}&=\f{1}{\sqrt{v}}\sum_{\bm r}e^{-i\bm k\cdot(\bm r+\bm\tau_n)}\ket{n\bm r}\\\\
        //!\\end{aligned}\rt\.$$
        let li: Complex<f64> = Complex::i();
        let E = Array1::<f64>::linspace(E_min, E_max, E_n);
        let mut A0 = Array2::<f64>::zeros((E_n, nk));
        let inv_U = U.inv().unwrap();
        let unfold_lat = &inv_U.dot(&self.lat);
        let V = self.lat.det().unwrap();
        let unfold_V = unfold_lat.det().unwrap();
        let U_det = U.det().unwrap();
        if U_det <= 1.0 {
            panic!("wrong!, the det(U) must larger than 1");
        }
        //我们先根据path计算一下k点
        let (kvec, kdist, knode) = {
            let n_node: usize = path.len_of(Axis(0));
            let k_metric = (&unfold_lat.dot(&unfold_lat.t())).inv().unwrap();
            let mut k_node = Array1::<f64>::zeros(n_node);
            for n in 1..n_node {
                //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
                let dk = path.row(n).to_owned() - path.slice(s![n - 1, ..]).to_owned();
                let a = k_metric.dot(&dk);
                let dklen = dk.dot(&a).sqrt();
                k_node[[n]] = k_node[[n - 1]] + dklen;
            }
            let mut node_index: Vec<usize> = vec![0];
            for n in 1..n_node - 1 {
                let frac = k_node[[n]] / k_node[[n_node - 1]];
                let a = (frac * ((nk - 1) as f64).round()) as usize;
                node_index.push(a)
            }
            node_index.push(nk - 1);
            let mut k_dist = Array1::<f64>::zeros(nk);
            let mut k_vec = Array2::<f64>::zeros((nk, self.dim_r()));
            //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
            k_vec.row_mut(0).assign(&path.row(0));
            for n in 1..n_node {
                let n_i = node_index[n - 1];
                let n_f = node_index[n];
                let kd_i = k_node[[n - 1]];
                let kd_f = k_node[[n]];
                let k_i = path.row(n - 1);
                let k_f = path.row(n);
                for j in n_i..n_f + 1 {
                    let frac: f64 = ((j - n_i) as f64) / ((n_f - n_i) as f64);
                    k_dist[[j]] = kd_i + frac * (kd_f - kd_i);
                    k_vec
                        .row_mut(j)
                        .assign(&((1.0 - frac) * k_i.to_owned() + frac * k_f.to_owned()));
                }
            }
            (k_vec, k_dist, k_node)
        };

        //我们先unfold一下k点
        let fold_k = &kvec.dot(&U.t()); // fold_k 是要求解的本征值和本征态
        let (eval, evec) = self.solve_all_parallel(&fold_k); //开始求解本征态和本征值
        let eval = eval.mapv(|x| Complex::new(x, 0.0));
        let mut G = Array3::<Complex<f64>>::zeros((E_n, nk, self.nsta()));
        //Zip::from(G.outer_iter_mut()).and(E.view()).par_for_each(|mut g,og| {g.assign(&(Complex::new(1.0,0.0)/(*og+li*eta-&eval)));});
        for (e, og) in E.iter().enumerate() {
            for (k, vec) in eval.outer_iter().enumerate() {
                for i in 0..self.nsta() {
                    G[[e, k, i]] = 1.0 / (*og + eta * li - vec[[i]]);
                }
            }
        }
        let mut G = G.mapv(|x| -x.im() / PI);
        G.swap_axes(0, 1);
        //接下来我们计算原胞的原子位置和轨道位置
        let mut unit_atom = Array2::<f64>::zeros((0, self.dim_r()));
        let mut unit_orb = Array2::<f64>::zeros((0, self.dim_r()));
        let atom_position = self.atom_position();
        let unfold_atom = &atom_position.dot(U).map(|x| {
            if (x.fract() - 1.0).abs() < precision || x.fract().abs() < precision {
                0.0
            } else if x.fract() < 0.0 {
                x.fract() + 1.0
            } else {
                x.fract()
            }
        });
        let unfold_orb = &self.orb.dot(U).map(|x| {
            if (x.fract() - 1.0).abs() < precision || x.fract().abs() < precision {
                0.0
            } else if x.fract() < 0.0 {
                x.fract() + 1.0
            } else {
                x.fract()
            }
        });
        let mut match_atom_list = Array1::<usize>::zeros(self.natom());
        let mut match_orb_list = Array1::<usize>::zeros(self.norb());
        let mut unit_atom_orb_match = Vec::<usize>::new(); //这个是原胞中, 原子的第一个轨道对应的轨道list中的位置
        //接下来我们计算原胞内存在哪些原子, 然后给出原胞和超胞的对应关系, 以及原胞轨道和超胞轨道的对应关系
        let mut a = 0;
        let mut b = 0;
        let mut orb_index = 0;
        for (i, u_atom) in unfold_atom.outer_iter().enumerate() {
            let (exist, index): (bool, Option<usize>) = {
                let mut exist = false;
                let mut index = None;
                for (j, u_atom_one) in unit_atom.outer_iter().enumerate() {
                    let mut a0 = true;
                    a0 = a0 && ((&u_atom - &u_atom_one).norm() < 5e-2);
                    if a0 {
                        exist = true;
                        index = Some(j);
                        break;
                    }
                }
                (exist, index)
            };
            if exist && a != 0 {
                let index = index.unwrap();
                match_atom_list[[i]] = index;
                for i0 in
                    unit_atom_orb_match[index]..unit_atom_orb_match[index] + self.atoms[i].norb()
                {
                    match_orb_list[[b]] = i0;
                    b += 1;
                }
            } else {
                unit_atom.push_row(u_atom);
                match_atom_list[[i]] = a;
                unit_atom_orb_match.push(orb_index);
                for i0 in 0..self.atoms[i].norb() {
                    unit_orb.push_row(unfold_orb.row(b));
                    match_orb_list[[b]] = i0 + orb_index;
                    b += 1;
                }
                orb_index += self.atoms[i].norb();
                a += 1;
            }
        }
        if unit_atom.nrows() != self.natom() / (U_det as usize) {
            panic!("Wrong, the unit cell's atoms number is wrong! please check your atom position");
        }
        //好了, 接下来让我们计算权重
        let mut weight = Array2::<Complex<f64>>::zeros((nk, self.nsta()));
        let mut B = Array3::<f64>::zeros((nk, E_n, unit_orb.nrows()));
        if self.spin {
            for k0 in 0..nk {
                let mut r = &self.orb.dot(&fold_k.row(k0)) - &self.orb.dot(U).dot(&kvec.row(k0));
                let r0 = r.clone();
                r.append(Axis(0), r0.view()).unwrap();
                weight
                    .slice_mut(s![k0, ..])
                    .assign(&(-PI * 2.0 * r).mapv(|x| Complex::new(0.0, x).exp()));
            }
            let A = match_orb_list.clone();
            match_orb_list.append(Axis(0), A.view()).unwrap();
            Zip::from(B.outer_iter_mut())
                .and(G.outer_iter())
                .and(weight.outer_iter())
                .and(evec.outer_iter())
                .par_for_each(|mut b, g, w, vec| {
                    for (i0, mut b0) in b.axis_iter_mut(Axis(1)).enumerate() {
                        let mut A = Array1::<Complex<f64>>::zeros(self.nsta());
                        A = match_orb_list
                            .iter()
                            .enumerate()
                            .zip(w.iter().zip(vec.axis_iter(Axis(1))))
                            .fold(A.clone(), |mut acc, ((io, orb_index), (w0, vec0))| {
                                if *orb_index == i0 {
                                    acc + *w0 * &vec0
                                } else {
                                    acc
                                }
                            });
                        let A = A.mapv(|x| x.norm_sqr());
                        b0.assign(&g.dot(&A));
                    }
                });
        } else {
            for k0 in 0..nk {
                let weight1 = (-PI
                    * 2.0
                    * (&self.orb.dot(&fold_k.row(k0)) - &self.orb.dot(U).dot(&kvec.row(k0))))
                    .mapv(|x| Complex::new(0.0, x).exp());
                weight.slice_mut(s![k0, ..]).assign(&weight1);
            }
            Zip::from(B.outer_iter_mut())
                .and(G.outer_iter())
                .and(weight.outer_iter())
                .and(evec.outer_iter())
                .par_for_each(|mut b, g, w, vec| {
                    for (i0, mut b0) in b.axis_iter_mut(Axis(1)).enumerate() {
                        let mut A = Array1::<Complex<f64>>::zeros(self.nsta());
                        A = match_orb_list
                            .iter()
                            .enumerate()
                            .zip(w.iter().zip(vec.axis_iter(Axis(1))))
                            .fold(A.clone(), |mut acc, ((io, orb_index), (w0, vec0))| {
                                if *orb_index == i0 {
                                    acc + *w0 * &vec0
                                } else {
                                    acc
                                }
                            });
                        let A = A.mapv(|x| x.norm_sqr());
                        b0.assign(&g.dot(&A));
                    }
                });
        }
        A0 = B.sum_axis(Axis(2));
        A0.reversed_axes()
    }

    pub fn make_supercell(&self, U: &Array2<f64>) -> Model {
        //这个函数是用来对模型做变换的, 变换前后模型的基矢 $L'=UL$.
        //!This function is used to transform the model, where the new basis after transformation is given by $L' = UL$.
        if self.dim_r() != U.len_of(Axis(0)) {
            panic!("Wrong, the imput U's dimension must equal to self.dim_r()")
        }
        //新的lattice
        let new_lat = U.dot(&self.lat);
        //体积的扩大倍数
        let U_det = U.det().unwrap() as isize;
        if U_det < 0 {
            panic!(
                "Wrong, the U_det is {}, you should using right hand axis",
                U_det
            );
        } else if U_det == 0 {
            panic!("Wrong, the U_det is {}", U_det);
        }
        let U_inv = U.inv().unwrap();
        //开始判断是否存在小数
        for i in 0..U.len_of(Axis(0)) {
            for j in 0..U.len_of(Axis(1)) {
                if U[[i, j]].fract() > 1e-8 {
                    panic!(
                        "Wrong, the U's element must be integer, but your given is {} at [{},{}]",
                        U[[i, j]],
                        i,
                        j
                    );
                }
            }
        }

        //开始构建新的轨道位置和原子位置
        //新的轨道
        let mut use_orb = self.orb.dot(&U_inv);
        //新的原子位置
        let use_atom_position = self.atom_position().dot(&U_inv);
        //新的atom_list
        let mut use_atom_list: Vec<usize> = Vec::new();
        let mut orb_list: Vec<usize> = Vec::new();
        let mut new_orb = Array2::<f64>::zeros((0, self.dim_r()));
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new();
        let mut a = 0;
        for i in 0..self.natom() {
            use_atom_list.push(a);
            a += self.atoms[i].norb();
        }

        match self.dim_r() {
            3 => {
                for i in -U_det..U_det {
                    for j in -U_det..U_det {
                        for k in -U_det..U_det {
                            for n in 0..self.natom() {
                                let mut atoms = use_atom_position.row(n).to_owned()
                                    + (i as f64) * U_inv.row(0).to_owned()
                                    + (j as f64) * U_inv.row(1).to_owned()
                                    + (k as f64) * U_inv.row(2).to_owned(); //原子的位置在新的坐标系下的坐标
                                atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                                    0.0
                                } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                                    1.0
                                } else {
                                    atoms[[0]]
                                };
                                atoms[[1]] = if atoms[[1]].abs() < 1e-8 {
                                    0.0
                                } else if (atoms[[1]] - 1.0).abs() < 1e-8 {
                                    1.0
                                } else {
                                    atoms[[1]]
                                };
                                atoms[[2]] = if atoms[[2]].abs() < 1e-8 {
                                    0.0
                                } else if (atoms[[2]] - 1.0).abs() < 1e-8 {
                                    1.0
                                } else {
                                    atoms[[2]]
                                };
                                if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                                    //判断是否在原胞内
                                    new_atom.push(Atom::new(
                                        atoms,
                                        self.atoms[n].norb(),
                                        self.atoms[n].atom_type(),
                                    ));
                                    for n0 in
                                        use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb()
                                    {
                                        //开始根据原子位置开始生成轨道
                                        let mut orbs = use_orb.row(n0).to_owned()
                                            + (i as f64) * U_inv.row(0).to_owned()
                                            + (j as f64) * U_inv.row(1).to_owned()
                                            + (k as f64) * U_inv.row(2).to_owned(); //新的轨道的坐标
                                        new_orb.push_row(orbs.view());
                                        new_orb_proj.push(self.orb_projection[n0]);
                                        orb_list.push(n0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            2 => {
                let U_det = U_det * 2;
                for i in -U_det..U_det {
                    for j in -U_det..U_det {
                        for n in 0..self.natom() {
                            let mut atoms = use_atom_position.row(n).to_owned()
                                + (i as f64) * U_inv.row(0).to_owned()
                                + (j as f64) * U_inv.row(1).to_owned(); //原子的位置在新的坐标系下的坐标
                            atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                                0.0
                            } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                                1.0
                            } else {
                                atoms[[0]]
                            };
                            atoms[[1]] = if atoms[[1]].abs() < 1e-8 {
                                0.0
                            } else if (atoms[[1]] - 1.0).abs() < 1e-8 {
                                1.0
                            } else {
                                atoms[[1]]
                            };
                            if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                                //判断是否在原胞内
                                new_atom.push(Atom::new(
                                    atoms,
                                    self.atoms[n].norb(),
                                    self.atoms[n].atom_type(),
                                ));
                                for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb()
                                {
                                    //开始根据原子位置开始生成轨道
                                    let mut orbs = use_orb.row(n0).to_owned()
                                        + (i as f64) * U_inv.row(0).to_owned()
                                        + (j as f64) * U_inv.row(1).to_owned(); //新的轨道的坐标
                                    new_orb.push_row(orbs.view());
                                    new_orb_proj.push(self.orb_projection[n0]);
                                    orb_list.push(n0);
                                    //orb_list_R.push_row(&arr1(&[i,j]));
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                for i in -U_det..U_det {
                    for n in 0..self.natom() {
                        let mut atoms = use_atom_position.row(n).to_owned()
                            + (i as f64) * U_inv.row(0).to_owned(); //原子的位置在新的坐标系下的坐标
                        atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                            0.0
                        } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                            1.0
                        } else {
                            atoms[[0]]
                        };
                        if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                            //判断是否在原胞内
                            new_atom.push(Atom::new(
                                atoms,
                                self.atoms[n].norb(),
                                self.atoms[n].atom_type(),
                            ));
                            for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb() {
                                //开始根据原子位置开始生成轨道
                                let mut orbs = use_orb.row(n0).to_owned()
                                    + (i as f64) * U_inv.row(0).to_owned(); //新的轨道的坐标
                                new_orb.push_row(orbs.view());
                                new_orb_proj.push(self.orb_projection[n0]);
                                orb_list.push(n0);
                                //orb_list_R.push_row(&arr1(&[i]));
                            }
                        }
                    }
                }
            }
            _ => todo!(),
        }
        //轨道位置和原子位置构建完成, 接下来我们开始构建哈密顿量
        let norb = new_orb.len_of(Axis(0));
        let nsta = if self.spin { 2 * norb } else { norb };
        let natom = new_atom.len();
        let n_R = self.hamR.len_of(Axis(0));
        let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r())); //超胞准备用的hamR
        let mut use_hamR = Array2::<isize>::zeros((1, self.dim_r())); //超胞的hamR的可能, 如果这个hamR没有对应的hopping就会被删除
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta)); //超胞准备用的ham
        //超胞准备用的rmatrix
        let mut new_rmatrix = Array4::<Complex<f64>>::zeros((1, self.dim_r(), nsta, nsta));
        let max_use_hamR = self.hamR.mapv(|x| x as f64);
        let max_use_hamR = max_use_hamR.dot(&U.inv().unwrap());
        let mut max_hamR =
            max_use_hamR
                .outer_iter()
                .fold(Array1::zeros(self.dim_r()), |mut acc, x| {
                    for i in 0..self.dim_r() {
                        acc[[i]] = if acc[[i]] > x[[i]].abs() {
                            acc[[i]]
                        } else {
                            x[[i]].abs()
                        };
                    }
                    acc
                });
        let max_R = max_hamR.mapv(|x| (x.ceil() as isize) + 1);
        //let mut max_R=Array1::<isize>::zeros(self.dim_r());
        //let max_R:isize=U_det.abs()*(self.dim_r() as isize);
        //let max_R=Array1::<isize>::ones(self.dim_r())*max_R;
        //用来产生可能的hamR
        match self.dim_r() {
            1 => {
                for i in -max_R[[0]]..max_R[[0]] + 1 {
                    if i != 0 {
                        use_hamR.push_row(array![i].view());
                    }
                }
            }
            2 => {
                for j in -max_R[[1]]..max_R[[1]] + 1 {
                    for i in -max_R[[0]]..max_R[[0]] + 1 {
                        if i != 0 || j != 0 {
                            use_hamR.push_row(array![i, j].view());
                        }
                    }
                }
            }
            3 => {
                for k in -max_R[[2]]..max_R[[2]] + 1 {
                    for i in -max_R[[0]]..max_R[[0]] + 1 {
                        for j in -max_R[[1]]..max_R[[1]] + 1 {
                            if i != 0 || j != 0 || k != 0 {
                                use_hamR.push_row(array![i, j, k].view());
                            }
                        }
                    }
                }
            }
            _ => todo!(),
        }
        let use_n_R = use_hamR.len_of(Axis(0));
        let mut gen_rmatrix: bool = false;
        if self.rmatrix.len_of(Axis(0)) == 1 {
            for i in 0..self.dim_r() {
                for s in 0..norb {
                    new_rmatrix[[0, i, s, s]] = Complex::new(new_orb[[s, i]], 0.0);
                }
            }
            if self.spin {
                for i in 0..self.dim_r() {
                    for s in 0..norb {
                        new_rmatrix[[0, i, s + norb, s + norb]] =
                            Complex::new(new_orb[[s, i]], 0.0);
                    }
                }
            }
        } else {
            gen_rmatrix = true;
        }
        if self.spin && gen_rmatrix {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                let mut use_rmatrix = Array3::<Complex<f64>>::zeros((self.dim_r(), nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64); //超胞的 R 在原始原胞的 R
                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        let R0_exit = find_R(&self.hamR, &R0);
                        if R0_exit {
                            let index = index_R(&self.hamR, &R0);
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                            useham[[int_i + norb, int_j]] =
                                self.ham[[index, *use_i + self.norb(), *use_j]];
                            useham[[int_i, int_j + norb]] =
                                self.ham[[index, *use_i, *use_j + self.norb()]];
                            useham[[int_i + norb, int_j + norb]] =
                                self.ham[[index, *use_i + self.norb(), *use_j + self.norb()]];
                            for r in 0..self.dim_r() {
                                use_rmatrix[[r, int_i, int_j]] =
                                    self.rmatrix[[index, r, *use_i, *use_j]];
                                use_rmatrix[[r, int_i + norb, int_j]] =
                                    self.rmatrix[[index, r, *use_i + self.norb(), *use_j]];
                                use_rmatrix[[r, int_i, int_j + norb]] =
                                    self.rmatrix[[index, r, *use_i, *use_j + self.norb()]];
                                use_rmatrix[[r, int_i + norb, int_j + norb]] = self.rmatrix
                                    [[index, r, *use_i + self.norb(), *use_j + self.norb()]];
                            }
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_hamR.push_row(use_R.view());
                    new_rmatrix.push(Axis(0), use_rmatrix.view());
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                    new_rmatrix
                        .slice_mut(s![0, .., .., ..])
                        .assign(&use_rmatrix);
                }
            }
        } else if gen_rmatrix && !self.spin {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((norb, norb));
                let mut use_rmatrix = Array3::<Complex<f64>>::zeros((self.dim_r(), norb, norb));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64); //超胞的 R 在原始原胞的 R
                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        let R0_exit = find_R(&self.hamR, &R0);
                        if R0_exit {
                            let index = index_R(&self.hamR, &R0);
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                            for r in 0..self.dim_r() {
                                use_rmatrix[[r, int_i, int_j]] =
                                    self.rmatrix[[index, r, *use_i, *use_j]]
                            }
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_rmatrix.push(Axis(0), use_rmatrix.view());
                    new_hamR.push_row(use_R);
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                    new_rmatrix
                        .slice_mut(s![0, .., .., ..])
                        .assign(&use_rmatrix);
                }
            }
        } else if self.spin {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> =
                            &new_orb.row(int_j) - &new_orb.row(int_i) + &use_R.map(|x| *x as f64); //超胞的 R 在原始原胞的 R

                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        let R0_exist = find_R(&self.hamR, &R0);
                        if R0_exist {
                            let index = index_R(&self.hamR, &R0);
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                            useham[[int_i + norb, int_j]] =
                                self.ham[[index, *use_i + self.norb(), *use_j]];
                            useham[[int_i, int_j + norb]] =
                                self.ham[[index, *use_i, *use_j + self.norb()]];
                            useham[[int_i + norb, int_j + norb]] =
                                self.ham[[index, *use_i + self.norb(), *use_j + self.norb()]];
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_hamR.push_row(use_R.view());
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                }
            }
        } else {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64); //超胞的 R 在原始原胞的 R
                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        let R0_exit = find_R(&self.hamR, &R0);
                        if R0_exit {
                            let index = index_R(&self.hamR, &R0);
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_hamR.push_row(use_R);
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                }
            }
        }
        let mut model = Model {
            dim_r: self.dim_r(),
            spin: self.spin,
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_proj,
            atoms: new_atom,
            ham: new_ham,
            hamR: new_hamR,
            rmatrix: new_rmatrix,
        };
        model
    }
    #[allow(non_snake_case)]
    pub fn dos(
        &self,
        k_mesh: &Array1<usize>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        sigma: f64,
    ) -> (Array1<f64>, Array1<f64>) {
        //! 我这里用的算法是高斯算法, 其算法过程如下
        //!
        //! 首先, 根据 k_mesh 算出所有的能量 $\ve_n$, 然后, 按照定义
        //! $$\rho(\ve)=\sum_N\int\dd\bm k \delta(\ve_n-\ve)$$
        //! 我们将 $\delta(\ve_n-\ve)$ 做了替换, 换成了 $\f{1}{\sqrt{2\pi}\sigma}e^{-\f{(\ve_n-\ve)^2}{2\sigma^2}}$
        //!
        //! 然后, 计算方法是先算出所有的能量, 再将能量乘以高斯分布, 就能得到态密度.
        //!
        //! 态密度的光滑程度和k点密度以及高斯分布的展宽有关
        let kvec: Array2<f64> = gen_kmesh(&k_mesh);
        let nk = kvec.len_of(Axis(0));
        let band = self.solve_band_all_parallel(&kvec);
        let E = Array1::linspace(E_min, E_max, E_n);
        let mut dos = Array1::<f64>::zeros(E_n);
        let dim: usize = k_mesh.len();
        let centre = band.into_raw_vec().into_par_iter();
        let sigma0 = 1.0 / sigma;
        let pi0 = 1.0 / (2.0 * PI).sqrt();
        let dos = Array1::<f64>::zeros(E_n);
        let dos = centre
            .fold(
                || Array1::<f64>::zeros(E_n),
                |acc, x| {
                    let A = (&E - x) * sigma0;
                    let f = (-&A * &A / 2.0).mapv(|x| x.exp()) * sigma0 * pi0;
                    acc + &f
                },
            )
            .reduce(|| Array1::<f64>::zeros(E_n), |acc, x| acc + x);
        let dos = dos / (nk as f64);
        (E, dos)
    }
    ///这个函数是用来快速画能带图的, 用python画图, 因为Rust画图不太方便.
    #[allow(non_snake_case)]
    pub fn show_band(
        &self,
        path: &Array2<f64>,
        label: &Vec<&str>,
        nk: usize,
        name: &str,
    ) -> std::io::Result<()> {
        use gnuplot::AutoOption::*;
        use gnuplot::AxesCommon;
        use gnuplot::Tick::*;
        use gnuplot::{Caption, Color, Figure, Font, LineStyle, Solid};
        use std::fs::create_dir_all;
        use std::path::Path;
        if path.len_of(Axis(0)) != label.len() {
            panic!(
                "Error, the path's length {} and label's length {} must be equal!",
                path.len_of(Axis(0)),
                label.len()
            )
        }
        let (k_vec, k_dist, k_node) = self.k_path(&path, nk);
        let eval = self.solve_band_all_parallel(&k_vec);
        create_dir_all(name).expect("can't creat the file");
        let mut name0 = String::new();
        name0.push_str("./");
        name0.push_str(&name);
        let name = name0;
        let mut band_name = name.clone();
        band_name.push_str("/BAND.dat");
        let band_name = Path::new(&band_name);
        let mut file = File::create(band_name).expect("Unable to BAND.dat");
        for i in 0..nk {
            let mut s = String::new();
            let aa = format!("{:.6}", k_dist[[i]]);
            s.push_str(&aa);
            for j in 0..self.nsta() {
                if eval[[i, j]] >= 0.0 {
                    s.push_str("     ");
                } else {
                    s.push_str("    ");
                }
                let aa = format!("{:.6}", eval[[i, j]]);
                s.push_str(&aa);
            }
            writeln!(file, "{}", s)?;
        }
        let mut k_name = name.clone();
        k_name.push_str("/KLABELS");
        let k_name = Path::new(&k_name);
        let mut file = File::create(k_name).expect("Unable to create KLBAELS"); //写下高对称点的位置
        for i in 0..path.len_of(Axis(0)) {
            let mut s = String::new();
            let aa = format!("{:.6}", k_node[[i]]);
            s.push_str(&aa);
            s.push_str("      ");
            s.push_str(&label[i]);
            writeln!(file, "{}", s)?;
        }
        let mut py_name = name.clone();
        py_name.push_str("/print.py");
        let py_name = Path::new(&py_name);
        let mut file = File::create(py_name).expect("Unable to create print.py");
        writeln!(
            file,
            "import numpy as np\nimport matplotlib.pyplot as plt\ndata=np.loadtxt('BAND.dat')\nk_nodes=[]\nlabel=[]\nf=open('KLABELS')\nfor i in f.readlines():\n    k_nodes.append(float(i.split()[0]))\n    label.append(i.split()[1])\nfig,ax=plt.subplots()\nax.plot(data[:,0],data[:,1:],c='b')\nfor x in k_nodes:\n    ax.axvline(x,c='k')\nax.set_xticks(k_nodes)\nax.set_xticklabels(label)\nax.set_xlim([0,k_nodes[-1]])\nfig.savefig('band.pdf')"
        );
        //开始绘制pdf图片
        let mut fg = Figure::new();
        let x: Vec<f64> = k_dist.to_vec();
        let axes = fg.axes2d();
        for i in 0..self.nsta() {
            let y: Vec<f64> = eval.slice(s![.., i]).to_owned().to_vec();
            axes.lines(&x, &y, &[Color("black"), LineStyle(Solid)]);
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len() - 1]]));
        let label = label.clone();
        let mut show_ticks = Vec::new();
        for i in 0..k_node.len() {
            let A = k_node[[i]];
            let B = label[i];
            show_ticks.push(Major(A, Fix(B)));
        }
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );

        let k_node = k_node.to_vec();
        let mut pdf_name = name.clone();
        pdf_name.push_str("/plot.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn from_hr(path: &str, file_name: &str, zero_energy: f64) -> Model {
        //! 这个函数是从 wannier90 中读取 TB 文件的
        //!
        //! 这里 path 表示 文件的位置, 可以使用绝对路径, 即 "/" 开头的路径, 也可以使用相对路径, 即运行 cargo run 时
        //! 候的文件夹作为起始路径.
        //!
        //! file_name 就是 wannier90 中的 seedname, 文件可以读取
        //! seedname.win, seedname_centres.xyz, seedname_hr.dat, 以及可选的 seedname_r.dat.
        //!
        //! 这里 seedname_centres.xyz 需要在 wannier90 中设置 write_xyz=true, 而
        //! seedname_hr.dat 需要设置 write_hr=true.
        //! 如果想要计算输运性质, 则需要 write_rmn=true, 能够给出一个seedname_r.dat.
        //!
        //! 此外, 对于高版本的wannier90, 如果想要保持较好的对称性, 建议将wannier90_wsvec.dat
        //! 也给出, 这样能够得到较好的对称结果

        use std::fs::File;
        use std::io::BufRead;
        use std::io::BufReader;
        use std::path::Path;

        let mut file_path = path.to_string();
        file_path.push_str(file_name);
        let mut hr_path = file_path.clone();
        hr_path.push_str("_hr.dat");

        let path = Path::new(&hr_path);
        let hr = File::open(path).expect(&format!(
            "Unable to open the file {:?}, please check if hr file is present\n",
            path
        ));
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();

        // 读取文件行
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }

        // 获取轨道数和R点数
        let nsta = reads[1].trim().parse::<usize>().unwrap();
        let n_R = reads[2].trim().parse::<usize>().unwrap();
        let mut weights: Vec<usize> = Vec::new();
        let mut n_line: usize = 0;

        // 解析文件数据以获取权重
        for i in 3..reads.len() {
            if reads[i].contains(".") {
                n_line = i;
                break;
            }
            let string = reads[i].trim().split_whitespace();
            let string: Vec<_> = string.map(|x| x.parse::<usize>().unwrap()).collect();
            weights.extend(string.clone());
        }

        // 初始化哈密顿量矩阵
        let mut hamR = Array2::<isize>::zeros((1, 3));
        let mut ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));

        // 遍历每个R点并填充哈密顿量
        for i in 0..n_R {
            let mut string = reads[i * nsta * nsta + n_line].trim().split_whitespace();
            let a = string.next().unwrap().parse::<isize>().unwrap();
            let b = string.next().unwrap().parse::<isize>().unwrap();
            let c = string.next().unwrap().parse::<isize>().unwrap();

            if a == 0 && b == 0 && c == 0 {
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).unwrap().parse::<f64>().unwrap();
                        let im = string.next().unwrap().parse::<f64>().unwrap();
                        ham[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                    }
                }
            } else {
                let mut matrix = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).unwrap().parse::<f64>().unwrap();
                        let im = string.next().unwrap().parse::<f64>().unwrap();
                        matrix[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                        // wannier90 里面是按照纵向排列的矩阵
                    }
                }
                ham.append(Axis(0), matrix.view()).unwrap();
                hamR.append(Axis(0), arr2(&[[a, b, c]]).view()).unwrap();
            }
        }

        // 调整哈密顿量以匹配能量零点
        for i in 0..nsta {
            ham[[0, i, i]] -= Complex::new(zero_energy, 0.0);
        }
        //开始读取 .win 文件
        let mut reads: Vec<String> = Vec::new();
        let mut win_path = file_path.clone();
        win_path.push_str(".win"); //文件的位置
        let path = Path::new(&win_path); //转化为路径格式
        let hr = File::open(path).expect("Unable open the file, please check if have win file");
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }
        let mut read_iter = reads.iter();
        let mut lat = Array2::<f64>::zeros((3, 3)); //晶格轨道坐标初始化
        let mut spin: bool = false; //体系自旋初始化
        let mut natom: usize = 0; //原子位置初始化
        let mut atom = Vec::new(); //原子位置坐标初始化
        let mut orb_proj = Vec::new();
        let mut proj_name = Vec::new();
        let mut proj_list: Vec<usize> = Vec::new();
        let mut atom_list: Vec<usize> = Vec::new();
        let mut atom_name: Vec<&str> = Vec::new();
        let mut atom_pos = Array2::<f64>::zeros((0, 3));
        let mut atom_proj = Vec::new();
        let mut norb = 0;
        loop {
            let a = read_iter.next();
            if a == None {
                break;
            } else {
                let a = a.unwrap();
                if a.contains("begin unit_cell_cart") {
                    let mut lat1 = read_iter.next().unwrap().trim().split_whitespace(); //将数字放到
                    let mut lat2 = read_iter.next().unwrap().trim().split_whitespace();
                    let mut lat3 = read_iter.next().unwrap().trim().split_whitespace();
                    for i in 0..3 {
                        lat[[0, i]] = lat1.next().unwrap().parse::<f64>().unwrap();
                        lat[[1, i]] = lat2.next().unwrap().parse::<f64>().unwrap();
                        lat[[2, i]] = lat3.next().unwrap().parse::<f64>().unwrap();
                    }
                } else if a.contains("spinors") && (a.contains("T") || a.contains("t")) {
                    spin = true;
                } else if a.contains("begin projections") {
                    loop {
                        let string = read_iter.next().unwrap();
                        if string.contains("end projections") {
                            break;
                        } else {
                            let prj: Vec<&str> = string
                                .split(|c| c == ',' || c == ';' || c == ':')
                                .map(|x| x.trim())
                                .collect();
                            let mut atom_orb_number: usize = 0;
                            let mut proj_orb = Vec::new();
                            for item in prj[1..].iter() {
                                let (aa, use_proj_orb): (usize, Vec<_>) = match (*item).trim() {
                                    "s" => (1, vec![OrbProj::s]),
                                    "p" => (3, vec![OrbProj::pz, OrbProj::px, OrbProj::py]),
                                    "d" => (
                                        5,
                                        vec![
                                            OrbProj::dz2,
                                            OrbProj::dxz,
                                            OrbProj::dyz,
                                            OrbProj::dx2y2,
                                            OrbProj::dxy,
                                        ],
                                    ),
                                    "f" => (
                                        7,
                                        vec![
                                            OrbProj::fz3,
                                            OrbProj::fxz2,
                                            OrbProj::fyz2,
                                            OrbProj::fzx2y2,
                                            OrbProj::fxyz,
                                            OrbProj::fxx23y2,
                                            OrbProj::fy3x2y2,
                                        ],
                                    ),
                                    "sp3" => (
                                        4,
                                        vec![
                                            OrbProj::sp3_1,
                                            OrbProj::sp3_2,
                                            OrbProj::sp3_3,
                                            OrbProj::sp3_4,
                                        ],
                                    ),
                                    "sp2" => {
                                        (3, vec![OrbProj::sp2_1, OrbProj::sp2_2, OrbProj::sp2_3])
                                    }
                                    "sp" => (2, vec![OrbProj::sp_1, OrbProj::sp_2]),
                                    "sp3d" => (
                                        5,
                                        vec![
                                            OrbProj::sp3d_1,
                                            OrbProj::sp3d_2,
                                            OrbProj::sp3d_3,
                                            OrbProj::sp3d_4,
                                            OrbProj::sp3d_5,
                                        ],
                                    ),
                                    "sp3d2" => (
                                        6,
                                        vec![
                                            OrbProj::sp3d2_1,
                                            OrbProj::sp3d2_2,
                                            OrbProj::sp3d2_3,
                                            OrbProj::sp3d2_4,
                                            OrbProj::sp3d2_5,
                                            OrbProj::sp3d2_6,
                                        ],
                                    ),
                                    "px" => (1, vec![OrbProj::px]),
                                    "py" => (1, vec![OrbProj::py]),
                                    "pz" => (1, vec![OrbProj::pz]),
                                    "dxy" => (1, vec![OrbProj::dxy]),
                                    "dxz" => (1, vec![OrbProj::dxz]),
                                    "dyz" => (1, vec![OrbProj::dyz]),
                                    "dz2" => (1, vec![OrbProj::dz2]),
                                    "dx2-y2" => (1, vec![OrbProj::dx2y2]),
                                    &_ => panic!(
                                        "Wrong, no matching, please check the projection field in seedname.win. There are some projections that can not be identified"
                                    ),
                                };
                                atom_orb_number += aa;
                                proj_orb.extend(use_proj_orb);
                            }
                            proj_list.push(atom_orb_number);
                            atom_proj.push(proj_orb);
                            let proj_type = AtomType::from_str(prj[0]);
                            proj_name.push(proj_type);
                        }
                    }
                } else if a.contains("begin atoms_cart") {
                    loop {
                        let string = read_iter.next().unwrap();
                        if string.contains("end atoms_cart") {
                            break;
                        } else {
                            let prj: Vec<&str> = string.split_whitespace().collect();
                            atom_name.push(prj[0]);
                            let a1 = prj[1].parse::<f64>().unwrap();
                            let a2 = prj[2].parse::<f64>().unwrap();
                            let a3 = prj[3].parse::<f64>().unwrap();
                            let a = array![a1, a2, a3];
                            atom_pos.push_row(a.view()); //这里我们不用win 里面的, 因为这个和orb没法对应, 如果没有xyz文件才考虑用这个
                        }
                    }
                }
            }
        }
        //开始读取 seedname_centres.xyz 文件
        let mut reads: Vec<String> = Vec::new();
        let mut xyz_path = file_path.clone();
        xyz_path.push_str("_centres.xyz");
        let path = Path::new(&xyz_path);
        let hr = File::open(path);
        let orb = if let Ok(hr) = hr {
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            //let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
            let norb = if spin { nsta / 2 } else { nsta };
            let mut orb = Array2::<f64>::zeros((norb, 3));
            for i in 0..norb {
                let a: Vec<&str> = reads[i + 2].trim().split_whitespace().collect();
                orb[[i, 0]] = a[1].parse::<f64>().unwrap();
                orb[[i, 1]] = a[2].parse::<f64>().unwrap();
                orb[[i, 2]] = a[3].parse::<f64>().unwrap();
            }
            orb = orb.dot(&lat.inv().unwrap());
            let mut new_atom_pos = Array2::<f64>::zeros((reads.len() - 2 - nsta, 3));
            let mut new_atom_name = Vec::new();
            for i in 0..reads.len() - 2 - nsta {
                let a: Vec<&str> = reads[i + 2 + nsta].trim().split_whitespace().collect();
                new_atom_pos[[i, 0]] = a[1].parse::<f64>().unwrap();
                new_atom_pos[[i, 1]] = a[2].parse::<f64>().unwrap();
                new_atom_pos[[i, 2]] = a[3].parse::<f64>().unwrap();
                new_atom_name.push(AtomType::from_str(a[0]));
            }
            //接下来如果wannier90.win 和 .xyz 文件的原子顺序不一致, 那么我们以xyz的原子顺序为准, 调整 atom_list

            for (i, name) in new_atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    if name == j_name {
                        let use_pos = new_atom_pos.row(i).dot(&lat.inv().unwrap());
                        let use_atom = Atom::new(use_pos, proj_list[j], *name);
                        atom.push(use_atom);
                        orb_proj.extend(atom_proj[j].clone());
                    }
                }
            }
            orb
        } else {
            let mut orb = Array2::<f64>::zeros((0, 3));
            let atom_pos = atom_pos.dot(&lat.inv().unwrap());
            for (i, name) in atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    let name = AtomType::from_str(name);
                    if name == *j_name {
                        let use_atom = Atom::new(atom_pos.row(i).to_owned(), proj_list[j], name);
                        orb_proj.extend(atom_proj[j].clone());
                        atom.push(use_atom.clone());
                        for _ in 0..proj_list[j] {
                            orb.push_row(use_atom.position().view());
                        }
                    }
                }
            }
            orb
        };
        //开始尝试读取 _r.dat 文件
        let mut reads: Vec<String> = Vec::new();
        let mut r_path = file_path.clone();
        r_path.push_str("_r.dat");
        let path = Path::new(&r_path);
        let hr = File::open(path);
        let mut have_r = false;
        let mut rmatrix = if hr.is_ok() {
            have_r = true;
            let hr = hr.unwrap();
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            let n_R = reads[2].trim().parse::<usize>().unwrap();
            let mut rmatrix = Array4::<Complex<f64>>::zeros((hamR.nrows(), 3, nsta, nsta));
            for i in 0..n_R {
                let mut string = reads[i * nsta * nsta + 3].trim().split_whitespace();
                let a = string.next().unwrap().parse::<isize>().unwrap();
                let b = string.next().unwrap().parse::<isize>().unwrap();
                let c = string.next().unwrap().parse::<isize>().unwrap();
                let R0 = array![a, b, c];
                let index = index_R(&hamR, &R0);
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let string = &reads[i * nsta * nsta + ind_i * nsta + ind_j + 3];
                        let mut string = string.trim().split_whitespace();
                        string.nth(4);
                        for r in 0..3 {
                            let re = string.next().unwrap().parse::<f64>().unwrap();
                            let im = string.next().unwrap().parse::<f64>().unwrap();
                            rmatrix[[index, r, ind_j, ind_i]] =
                                Complex::new(re, im) / (weights[i] as f64);
                        }
                    }
                }
            }
            rmatrix
        } else {
            let mut rmatrix = Array4::<Complex<f64>>::zeros((1, 3, nsta, nsta));
            for i in 0..norb {
                for r in 0..3 {
                    rmatrix[[0, r, i, i]] = Complex::<f64>::from(orb[[i, r]]);
                    if spin {
                        rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                    }
                }
            }
            rmatrix
        };

        //最后判断有没有wannier90_wsvec.dat-----------------------------------
        let mut ws_path = file_path.clone();
        ws_path.push_str("_wsvec.dat");
        let path = Path::new(&ws_path); //转化为路径格式
        let ws = File::open(path);
        let mut have_ws = false;
        if ws.is_ok() {
            have_ws = true;
            let ws = ws.unwrap();
            let reader = BufReader::new(ws);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            //开始针对ham, hamR 以及 rmatrix 进行修改
            //我们先考虑有rmatrix的情况
            if have_r {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                let mut new_rmatrix = Array4::zeros((1, 3, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().unwrap().parse::<isize>().unwrap();
                    let b = string.next().unwrap().parse::<isize>().unwrap();
                    let c = string.next().unwrap().parse::<isize>().unwrap();
                    let int_i = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    let int_j = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    let R = array![a, b, c];
                    let index = index_R(&hamR, &R);
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);
                    let hop_x = rmatrix[[index, 0, int_i, int_j]] / (weight as f64);
                    let hop_y = rmatrix[[index, 1, int_i, int_j]] / (weight as f64);
                    let hop_z = rmatrix[[index, 2, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if find_R(&new_hamR, &new_R) {
                            let index0 = index_R(&new_hamR, &new_R);
                            new_ham[[index0, int_i, int_j]] += hop;
                            new_rmatrix[[index0, 0, int_i, int_j]] += hop_x;
                            new_rmatrix[[index0, 1, int_i, int_j]] += hop_y;
                            new_rmatrix[[index0, 2, int_i, int_j]] += hop_z;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            let mut use_rmatrix = Array3::zeros((3, nsta, nsta));
                            use_ham[[int_i, int_j]] += hop;
                            use_rmatrix[[0, int_i, int_j]] += hop_x;
                            use_rmatrix[[1, int_i, int_j]] += hop_y;
                            use_rmatrix[[2, int_i, int_j]] += hop_z;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                            new_rmatrix.push(Axis(0), use_rmatrix.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
                rmatrix = new_rmatrix;
            } else {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().unwrap().parse::<isize>().unwrap();
                    let b = string.next().unwrap().parse::<isize>().unwrap();
                    let c = string.next().unwrap().parse::<isize>().unwrap();
                    let int_i = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    let int_j = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    let R = array![a, b, c];
                    let index = index_R(&hamR, &R);
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if find_R(&new_hamR, &new_R) {
                            let index0 = index_R(&new_hamR, &new_R);
                            new_ham[[index0, int_i, int_j]] += hop;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            use_ham[[int_i, int_j]] = hop;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
            }
        }
        //最后一步, 将rmatrix 变成厄密的

        if have_r {
            for r in 0..hamR.nrows() - 1 {
                let R = hamR.row(r);
                let R_inv = -&R;
                if find_R(&hamR, &R_inv) {
                    let index = index_R(&hamR, &R_inv);
                    for i in 0..nsta {
                        for j in 0..nsta {
                            rmatrix[[r, 0, i, j]] =
                                (rmatrix[[r, 0, i, j]] + rmatrix[[index, 0, j, i]].conj()) / 2.0;
                            rmatrix[[r, 1, i, j]] =
                                (rmatrix[[r, 1, i, j]] + rmatrix[[index, 1, j, i]].conj()) / 2.0;
                            rmatrix[[r, 2, i, j]] =
                                (rmatrix[[r, 2, i, j]] + rmatrix[[index, 2, j, i]].conj()) / 2.0;
                            rmatrix[[index, 0, j, i]] = rmatrix[[r, 0, i, j]].conj();
                            rmatrix[[index, 1, j, i]] = rmatrix[[r, 1, i, j]].conj();
                            rmatrix[[index, 2, j, i]] = rmatrix[[r, 2, i, j]].conj();
                        }
                    }
                } else {
                    panic!("Wrong!, the R has no -R, it's strange");
                }
            }
        }

        let mut model = Model {
            dim_r: 3,
            spin,
            lat,
            orb,
            orb_projection: orb_proj,
            atoms: atom,
            ham,
            hamR,
            rmatrix,
        };
        model
    }

    #[allow(non_snake_case)]
    pub fn from_tb(path: &str, file_name: &str, zero_energy: f64) -> Model {
        //! 这个函数是从 wannier90 中读取 TB 文件的
        //!
        //! 这里 path 表示 文件的位置, 可以使用绝对路径, 即 "/" 开头的路径, 也可以使用相对路径, 即运行 cargo run 时
        //! 候的文件夹作为起始路径.
        //!
        //! file_name 就是 wannier90 中的 seedname, 文件可以读取
        //! seedname.win, seedname_centres.xyz, seedname_hr.dat, 以及可选的 seedname_r.dat 以及 seedname_wsved.dat
        //!
        //! 这里 seedname_centres.xyz 需要在 wannier90 中设置 write_xyz=true, 而
        //! seedname_tb.dat 需要设置 write_tb=true.
        //! 如果想要计算输运性质, 则需要 write_rmn=true, 能够给出一个seedname_r.dat.
        //!
        //! 此外, 对于高版本的wannier90, 如果想要保持较好的对称性, 建议将wannier90_wsvec.dat
        //! 也给出, 这样能够得到较好的对称结果

        use std::fs::File;
        use std::io::BufRead;
        use std::io::BufReader;
        use std::path::Path;

        let mut file_path = path.to_string();
        file_path.push_str(file_name);
        let mut hr_path = file_path.clone();
        hr_path.push_str("_hr.dat");

        let path = Path::new(&hr_path);
        let hr = File::open(path).expect(&format!(
            "Unable to open the file {:?}, please check if hr file is present\n",
            path
        ));
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();

        // 读取文件行
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }

        // 获取轨道数和R点数
        let nsta = reads[1].trim().parse::<usize>().unwrap();
        let n_R = reads[2].trim().parse::<usize>().unwrap();
        let mut weights: Vec<usize> = Vec::new();
        let mut n_line: usize = 0;

        // 解析文件数据以获取权重
        for i in 3..reads.len() {
            if reads[i].contains(".") {
                n_line = i;
                break;
            }
            let string = reads[i].trim().split_whitespace();
            let string: Vec<_> = string.map(|x| x.parse::<usize>().unwrap()).collect();
            weights.extend(string.clone());
        }

        // 初始化哈密顿量矩阵
        let mut hamR = Array2::<isize>::zeros((1, 3));
        let mut ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));

        // 遍历每个R点并填充哈密顿量
        for i in 0..n_R {
            let mut string = reads[i * nsta * nsta + n_line].trim().split_whitespace();
            let a = string.next().unwrap().parse::<isize>().unwrap();
            let b = string.next().unwrap().parse::<isize>().unwrap();
            let c = string.next().unwrap().parse::<isize>().unwrap();

            if a == 0 && b == 0 && c == 0 {
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).unwrap().parse::<f64>().unwrap();
                        let im = string.next().unwrap().parse::<f64>().unwrap();
                        ham[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                    }
                }
            } else {
                let mut matrix = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let mut string = reads[i * nsta * nsta + ind_i * nsta + ind_j + n_line]
                            .trim()
                            .split_whitespace();
                        let re = string.nth(5).unwrap().parse::<f64>().unwrap();
                        let im = string.next().unwrap().parse::<f64>().unwrap();
                        matrix[[0, ind_j, ind_i]] = Complex::new(re, im) / (weights[i] as f64);
                        // wannier90 里面是按照纵向排列的矩阵
                    }
                }
                ham.append(Axis(0), matrix.view()).unwrap();
                hamR.append(Axis(0), arr2(&[[a, b, c]]).view()).unwrap();
            }
        }

        // 调整哈密顿量以匹配能量零点
        for i in 0..nsta {
            ham[[0, i, i]] -= Complex::new(zero_energy, 0.0);
        }
        //开始读取 .win 文件
        let mut reads: Vec<String> = Vec::new();
        let mut win_path = file_path.clone();
        win_path.push_str(".win"); //文件的位置
        let path = Path::new(&win_path); //转化为路径格式
        let hr = File::open(path).expect("Unable open the file, please check if have win file");
        let reader = BufReader::new(hr);
        let mut reads: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }
        let mut read_iter = reads.iter();
        let mut lat = Array2::<f64>::zeros((3, 3)); //晶格轨道坐标初始化
        let mut spin: bool = false; //体系自旋初始化
        let mut natom: usize = 0; //原子位置初始化
        let mut atom = Vec::new(); //原子位置坐标初始化
        let mut orb_proj = Vec::new();
        let mut proj_name = Vec::new();
        let mut proj_list: Vec<usize> = Vec::new();
        let mut atom_list: Vec<usize> = Vec::new();
        let mut atom_name: Vec<&str> = Vec::new();
        let mut atom_pos = Array2::<f64>::zeros((0, 3));
        let mut atom_proj = Vec::new();
        let mut norb = 0;
        loop {
            let a = read_iter.next();
            if a == None {
                break;
            } else {
                let a = a.unwrap();
                if a.contains("begin unit_cell_cart") {
                    let mut lat1 = read_iter.next().unwrap().trim().split_whitespace(); //将数字放到
                    let mut lat2 = read_iter.next().unwrap().trim().split_whitespace();
                    let mut lat3 = read_iter.next().unwrap().trim().split_whitespace();
                    for i in 0..3 {
                        lat[[0, i]] = lat1.next().unwrap().parse::<f64>().unwrap();
                        lat[[1, i]] = lat2.next().unwrap().parse::<f64>().unwrap();
                        lat[[2, i]] = lat3.next().unwrap().parse::<f64>().unwrap();
                    }
                } else if a.contains("spinors") && (a.contains("T") || a.contains("t")) {
                    spin = true;
                } else if a.contains("begin projections") {
                    loop {
                        let string = read_iter.next().unwrap();
                        if string.contains("end projections") {
                            break;
                        } else {
                            let prj: Vec<&str> = string
                                .split(|c| c == ',' || c == ';' || c == ':')
                                .map(|x| x.trim())
                                .collect();
                            let mut atom_orb_number: usize = 0;
                            let mut proj_orb = Vec::new();
                            for item in prj[1..].iter() {
                                let (aa, use_proj_orb): (usize, Vec<_>) = match (*item).trim() {
                                    "s" => (1, vec![OrbProj::s]),
                                    "p" => (3, vec![OrbProj::pz, OrbProj::px, OrbProj::py]),
                                    "d" => (
                                        5,
                                        vec![
                                            OrbProj::dz2,
                                            OrbProj::dxz,
                                            OrbProj::dyz,
                                            OrbProj::dx2y2,
                                            OrbProj::dxy,
                                        ],
                                    ),
                                    "f" => (
                                        7,
                                        vec![
                                            OrbProj::fz3,
                                            OrbProj::fxz2,
                                            OrbProj::fyz2,
                                            OrbProj::fzx2y2,
                                            OrbProj::fxyz,
                                            OrbProj::fxx23y2,
                                            OrbProj::fy3x2y2,
                                        ],
                                    ),
                                    "sp3" => (
                                        4,
                                        vec![
                                            OrbProj::sp3_1,
                                            OrbProj::sp3_2,
                                            OrbProj::sp3_3,
                                            OrbProj::sp3_4,
                                        ],
                                    ),
                                    "sp2" => {
                                        (3, vec![OrbProj::sp2_1, OrbProj::sp2_2, OrbProj::sp2_3])
                                    }
                                    "sp" => (2, vec![OrbProj::sp_1, OrbProj::sp_2]),
                                    "sp3d" => (
                                        5,
                                        vec![
                                            OrbProj::sp3d_1,
                                            OrbProj::sp3d_2,
                                            OrbProj::sp3d_3,
                                            OrbProj::sp3d_4,
                                            OrbProj::sp3d_5,
                                        ],
                                    ),
                                    "sp3d2" => (
                                        6,
                                        vec![
                                            OrbProj::sp3d2_1,
                                            OrbProj::sp3d2_2,
                                            OrbProj::sp3d2_3,
                                            OrbProj::sp3d2_4,
                                            OrbProj::sp3d2_5,
                                            OrbProj::sp3d2_6,
                                        ],
                                    ),
                                    "px" => (1, vec![OrbProj::px]),
                                    "py" => (1, vec![OrbProj::py]),
                                    "pz" => (1, vec![OrbProj::pz]),
                                    "dxy" => (1, vec![OrbProj::dxy]),
                                    "dxz" => (1, vec![OrbProj::dxz]),
                                    "dyz" => (1, vec![OrbProj::dyz]),
                                    "dz2" => (1, vec![OrbProj::dz2]),
                                    "dx2-y2" => (1, vec![OrbProj::dx2y2]),
                                    &_ => panic!(
                                        "Wrong, no matching, please check the projection field in seedname.win. There are some projections that can not be identified"
                                    ),
                                };
                                atom_orb_number += aa;
                                proj_orb.extend(use_proj_orb);
                            }
                            proj_list.push(atom_orb_number);
                            atom_proj.push(proj_orb);
                            let proj_type = AtomType::from_str(prj[0]);
                            proj_name.push(proj_type);
                        }
                    }
                } else if a.contains("begin atoms_cart") {
                    loop {
                        let string = read_iter.next().unwrap();
                        if string.contains("end atoms_cart") {
                            break;
                        } else {
                            let prj: Vec<&str> = string.split_whitespace().collect();
                            atom_name.push(prj[0]);
                            let a1 = prj[1].parse::<f64>().unwrap();
                            let a2 = prj[2].parse::<f64>().unwrap();
                            let a3 = prj[3].parse::<f64>().unwrap();
                            let a = array![a1, a2, a3];
                            atom_pos.push_row(a.view()); //这里我们不用win 里面的, 因为这个和orb没法对应, 如果没有xyz文件才考虑用这个
                        }
                    }
                }
            }
        }
        //开始读取 seedname_centres.xyz 文件
        let mut reads: Vec<String> = Vec::new();
        let mut xyz_path = file_path.clone();
        xyz_path.push_str("_centres.xyz");
        let path = Path::new(&xyz_path);
        let hr = File::open(path);
        let orb = if let Ok(hr) = hr {
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            //let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
            let norb = if spin { nsta / 2 } else { nsta };
            let mut orb = Array2::<f64>::zeros((norb, 3));
            for i in 0..norb {
                let a: Vec<&str> = reads[i + 2].trim().split_whitespace().collect();
                orb[[i, 0]] = a[1].parse::<f64>().unwrap();
                orb[[i, 1]] = a[2].parse::<f64>().unwrap();
                orb[[i, 2]] = a[3].parse::<f64>().unwrap();
            }
            orb = orb.dot(&lat.inv().unwrap());
            let mut new_atom_pos = Array2::<f64>::zeros((reads.len() - 2 - nsta, 3));
            let mut new_atom_name = Vec::new();
            for i in 0..reads.len() - 2 - nsta {
                let a: Vec<&str> = reads[i + 2 + nsta].trim().split_whitespace().collect();
                new_atom_pos[[i, 0]] = a[1].parse::<f64>().unwrap();
                new_atom_pos[[i, 1]] = a[2].parse::<f64>().unwrap();
                new_atom_pos[[i, 2]] = a[3].parse::<f64>().unwrap();
                new_atom_name.push(AtomType::from_str(a[0]));
            }
            //接下来如果wannier90.win 和 .xyz 文件的原子顺序不一致, 那么我们以xyz的原子顺序为准, 调整 atom_list

            for (i, name) in new_atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    if name == j_name {
                        let use_pos = new_atom_pos.row(i).dot(&lat.inv().unwrap());
                        let use_atom = Atom::new(use_pos, proj_list[j], *name);
                        atom.push(use_atom);
                        orb_proj.extend(atom_proj[j].clone());
                    }
                }
            }
            orb
        } else {
            let mut orb = Array2::<f64>::zeros((0, 3));
            let atom_pos = atom_pos.dot(&lat.inv().unwrap());
            for (i, name) in atom_name.iter().enumerate() {
                for (j, j_name) in proj_name.iter().enumerate() {
                    let name = AtomType::from_str(name);
                    if name == *j_name {
                        let use_atom = Atom::new(atom_pos.row(i).to_owned(), proj_list[j], name);
                        orb_proj.extend(atom_proj[j].clone());
                        atom.push(use_atom.clone());
                        for _ in 0..proj_list[j] {
                            orb.push_row(use_atom.position().view());
                        }
                    }
                }
            }
            orb
        };
        //开始尝试读取 _r.dat 文件
        let mut reads: Vec<String> = Vec::new();
        let mut r_path = file_path.clone();
        r_path.push_str("_r.dat");
        let path = Path::new(&r_path);
        let hr = File::open(path);
        let mut have_r = false;
        let mut rmatrix = if hr.is_ok() {
            have_r = true;
            let hr = hr.unwrap();
            let reader = BufReader::new(hr);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            let n_R = reads[2].trim().parse::<usize>().unwrap();
            let mut rmatrix = Array4::<Complex<f64>>::zeros((hamR.nrows(), 3, nsta, nsta));
            for i in 0..n_R {
                let mut string = reads[i * nsta * nsta + 3].trim().split_whitespace();
                let a = string.next().unwrap().parse::<isize>().unwrap();
                let b = string.next().unwrap().parse::<isize>().unwrap();
                let c = string.next().unwrap().parse::<isize>().unwrap();
                let R0 = array![a, b, c];
                let index = index_R(&hamR, &R0);
                for ind_i in 0..nsta {
                    for ind_j in 0..nsta {
                        let string = &reads[i * nsta * nsta + ind_i * nsta + ind_j + 3];
                        let mut string = string.trim().split_whitespace();
                        string.nth(4);
                        for r in 0..3 {
                            let re = string.next().unwrap().parse::<f64>().unwrap();
                            let im = string.next().unwrap().parse::<f64>().unwrap();
                            rmatrix[[index, r, ind_j, ind_i]] =
                                Complex::new(re, im) / (weights[i] as f64);
                        }
                    }
                }
            }
            rmatrix
        } else {
            let mut rmatrix = Array4::<Complex<f64>>::zeros((1, 3, nsta, nsta));
            for i in 0..norb {
                for r in 0..3 {
                    rmatrix[[0, r, i, i]] = Complex::<f64>::from(orb[[i, r]]);
                    if spin {
                        rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                    }
                }
            }
            rmatrix
        };

        //最后判断有没有wannier90_wsvec.dat-----------------------------------
        let mut ws_path = file_path.clone();
        ws_path.push_str("_wsvec.dat");
        let path = Path::new(&ws_path); //转化为路径格式
        let ws = File::open(path);
        let mut have_ws = false;
        if ws.is_ok() {
            have_ws = true;
            let ws = ws.unwrap();
            let reader = BufReader::new(ws);
            let mut reads: Vec<String> = Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            //开始针对ham, hamR 以及 rmatrix 进行修改
            //我们先考虑有rmatrix的情况
            if have_r {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                let mut new_rmatrix = Array4::zeros((1, 3, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().unwrap().parse::<isize>().unwrap();
                    let b = string.next().unwrap().parse::<isize>().unwrap();
                    let c = string.next().unwrap().parse::<isize>().unwrap();
                    let int_i = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    let int_j = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    let R = array![a, b, c];
                    let index = index_R(&hamR, &R);
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);
                    let hop_x = rmatrix[[index, 0, int_i, int_j]] / (weight as f64);
                    let hop_y = rmatrix[[index, 1, int_i, int_j]] / (weight as f64);
                    let hop_z = rmatrix[[index, 2, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if find_R(&new_hamR, &new_R) {
                            let index0 = index_R(&new_hamR, &new_R);
                            new_ham[[index0, int_i, int_j]] += hop;
                            new_rmatrix[[index0, 0, int_i, int_j]] += hop_x;
                            new_rmatrix[[index0, 1, int_i, int_j]] += hop_y;
                            new_rmatrix[[index0, 2, int_i, int_j]] += hop_z;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            let mut use_rmatrix = Array3::zeros((3, nsta, nsta));
                            use_ham[[int_i, int_j]] += hop;
                            use_rmatrix[[0, int_i, int_j]] += hop_x;
                            use_rmatrix[[1, int_i, int_j]] += hop_y;
                            use_rmatrix[[2, int_i, int_j]] += hop_z;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                            new_rmatrix.push(Axis(0), use_rmatrix.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
                rmatrix = new_rmatrix;
            } else {
                let mut i = 0;
                let mut new_hamR = Array2::zeros((1, 3));
                let mut new_ham = Array3::zeros((1, nsta, nsta));
                while (i < reads.len() - 1) {
                    i += 1;
                    let line = &reads[i];
                    let mut string = line.trim().split_whitespace();
                    let a = string.next().unwrap().parse::<isize>().unwrap();
                    let b = string.next().unwrap().parse::<isize>().unwrap();
                    let c = string.next().unwrap().parse::<isize>().unwrap();
                    let int_i = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    let int_j = string.next().unwrap().parse::<usize>().unwrap() - 1;
                    //接下来判断是否在我们的hamR 中
                    i += 1;
                    let mut weight = reads[i]
                        .trim()
                        .split_whitespace()
                        .next()
                        .unwrap()
                        .parse::<usize>()
                        .unwrap();
                    let R = array![a, b, c];
                    let index = index_R(&hamR, &R);
                    let hop = ham[[index, int_i, int_j]] / (weight as f64);

                    for i0 in 0..weight {
                        i += 1;
                        let line = &reads[i];
                        let mut string = line.trim().split_whitespace();
                        let a = string.next().unwrap().parse::<isize>().unwrap();
                        let b = string.next().unwrap().parse::<isize>().unwrap();
                        let c = string.next().unwrap().parse::<isize>().unwrap();
                        let new_R = array![R[[0]] + a, R[[1]] + b, R[[2]] + c];
                        if find_R(&new_hamR, &new_R) {
                            let index0 = index_R(&new_hamR, &new_R);
                            new_ham[[index0, int_i, int_j]] += hop;
                        } else {
                            let mut use_ham = Array2::zeros((nsta, nsta));
                            use_ham[[int_i, int_j]] = hop;
                            new_hamR.push_row(new_R.view());
                            new_ham.push(Axis(0), use_ham.view());
                        }
                    }
                }
                hamR = new_hamR;
                ham = new_ham;
            }
        }
        //最后一步, 将rmatrix 变成厄密的

        if have_r {
            for r in 0..hamR.nrows() - 1 {
                let R = hamR.row(r);
                let R_inv = -&R;
                if find_R(&hamR, &R_inv) {
                    let index = index_R(&hamR, &R_inv);
                    for i in 0..nsta {
                        for j in 0..nsta {
                            rmatrix[[r, 0, i, j]] =
                                (rmatrix[[r, 0, i, j]] + rmatrix[[index, 0, j, i]].conj()) / 2.0;
                            rmatrix[[r, 1, i, j]] =
                                (rmatrix[[r, 1, i, j]] + rmatrix[[index, 1, j, i]].conj()) / 2.0;
                            rmatrix[[r, 2, i, j]] =
                                (rmatrix[[r, 2, i, j]] + rmatrix[[index, 2, j, i]].conj()) / 2.0;
                            rmatrix[[index, 0, j, i]] = rmatrix[[r, 0, i, j]].conj();
                            rmatrix[[index, 1, j, i]] = rmatrix[[r, 1, i, j]].conj();
                            rmatrix[[index, 2, j, i]] = rmatrix[[r, 2, i, j]].conj();
                        }
                    }
                } else {
                    panic!("Wrong!, the R has no -R, it's strange");
                }
            }
        }

        let mut model = Model {
            dim_r: 3,
            spin,
            lat,
            orb,
            orb_projection: orb_proj,
            atoms: atom,
            ham,
            hamR,
            rmatrix,
        };
        model
    }
}

pub fn gauss(x: f64, eta: f64) -> f64 {
    //高斯函数
    let a = (x / eta);
    let g = (-a * a / 2.0).exp();
    let g = 1.0 / (2.0 * PI).sqrt() / eta * g;
    g
}
