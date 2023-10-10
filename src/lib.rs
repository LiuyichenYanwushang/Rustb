#![allow(warnings)]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

use gnuplot::Major;
use num_complex::Complex;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::concatenate;
use ndarray_linalg::*;
use std::f64::consts::PI;
use ndarray_linalg::{Eigh, UPLO};
use ndarray_linalg::conjugate;
use rayon::prelude::*;
use std::io::Write;
use std::fs::File;
use std::ops::AddAssign;
use std::ops::MulAssign;

pub use Rustb_basis::basis;
pub use Rustb_conductivity::conductivity;
/// This cate is used to perform various calculations on the TB model, currently including:
///
/// - Calculate the band structure
///
/// - Expand the cell and calculate the surface state
///
/// - Calculate the first-order anomalous Hall conductivity and spin Hall conductivity
///
#[allow(non_snake_case)]
#[derive(Clone,Debug)]
pub struct Model{
    /// - The real space dimension of the model.
    pub dim_r:usize,
    /// - The number of orbitals in the model.
    pub norb:usize,
    /// - The number of states in the model. If spin is enabled, nsta=norb$\times$2
    pub nsta:usize,
    /// - The number of atoms in the model. The atom and atom_list at the back are used to store the positions of the atoms, and the number of orbitals corresponding to each atom.
    pub natom:usize,
    /// - Whether the model has spin enabled. If enabled, spin=true
    pub spin:bool,
    /// - The lattice vector of the model, a dim_r$\times$dim_r matrix, the axis0 direction stores a 1$\times$dim_r lattice vector.
    pub lat:Array2::<f64>,
    /// - The position of the orbitals in the model. We use fractional coordinates uniformly.
    pub orb:Array2::<f64>,
    /// - The position of the atoms in the model, also in fractional coordinates.
    pub atom:Array2::<f64>,
    /// - The number of orbitals in the atoms, in the same order as the atom positions.
    pub atom_list:Vec<usize>,
    /// - The Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub ham:Array3::<Complex<f64>>,
    /// - The distance between the unit cell hoppings, i.e. R in $\bra{m0}\hat H\ket{nR}$.
    pub hamR:Array2::<isize>,
    /// - The position matrix, i.e. $\bra{m0}\hat{\bm r}\ket{nR}$.
    pub rmatrix:Array4::<Complex<f64>>,
}

pub struct surf_Green{
    /// - The real space dimension of the model.
    pub dim_r:usize,
    /// - The number of orbitals in the model.
    pub norb:usize,
    /// - The number of states in the model. If spin is enabled, nsta=norb$\times$2
    pub nsta:usize,
    /// - The number of atoms in the model. The atom and atom_list at the back are used to store the positions of the atoms, and the number of orbitals corresponding to each atom.
    pub natom:usize,
    /// - Whether the model has spin enabled. If enabled, spin=true
    pub spin:bool,
    /// - The lattice vector of the model, a dim_r$\times$dim_r matrix, the axis0 direction stores a 1$\times$dim_r lattice vector.
    pub lat:Array2::<f64>,
    /// - The position of the orbitals in the model. We use fractional coordinates uniformly.
    pub orb:Array2::<f64>,
    /// - The position of the atoms in the model, also in fractional coordinates.
    pub atom:Array2::<f64>,
    /// - The number of orbitals in the atoms, in the same order as the atom positions.
    pub atom_list:Vec<usize>,
    /// - The bulk Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub eta:f64,
    pub ham_bulk:Array3::<Complex<f64>>,
    /// - The distance between the unit cell hoppings, i.e. R in $\bra{m0}\hat H\ket{nR}$.
    pub ham_bulkR:Array2::<isize>,
    /// - The bulk Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub ham_hop:Array3::<Complex<f64>>,
    pub ham_hopR:Array2::<isize>,
}
#[allow(non_snake_case)]
pub fn find_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->bool{
    //!用来寻找 R 在hamR 中是否存在
    let n_R:usize=hamR.len_of(Axis(0));
    let dim_R:usize=hamR.len_of(Axis(1));
    for i in 0..(n_R){
        let mut a=true;
        for j in 0..(dim_R){
            a=a&&( hamR[[i,j]]==R[[j]]);
        }
        if a{
            return true
        }
    }
    false
}
#[allow(non_snake_case)]
pub fn index_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->usize{
    //!如果 R 在 hamR 中存在, 返回 R 在hamR 中的位置
    let n_R:usize=hamR.len_of(Axis(0));
    let dim_R:usize=hamR.len_of(Axis(1));
    for i in 0..n_R{
        let mut a=true;
        for j in 0..dim_R{
            a=a&&(hamR[[i,j]]==R[[j]]);
        }
        if a{
            return i
        }
    }
    panic!("Wrong, not find");
}
fn remove_row<T: Copy>(array: Array2<T>, row_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.nrows()).filter(|&r| r != row_to_remove).collect();
    array.select(Axis(0), &indices)
}
fn remove_col<T: Copy>(array: Array2<T>, col_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.ncols()).filter(|&r| r != col_to_remove).collect();
    array.select(Axis(1), &indices)
}
#[allow(non_snake_case)]
pub fn gen_kmesh(k_mesh:&Array1::<usize>)->Array2::<f64>{
    let dim:usize=k_mesh.len();
    let mut nk:usize=1;
    for i in 0..dim{
        nk*=k_mesh[[i]];
    }
    fn gen_kmesh_arr(k_mesh:&Array1::<usize>,r0:usize,mut usek:Array1::<f64>)->Array2::<f64>{
        let dim:usize=k_mesh.len();
        let mut kvec=Array2::<f64>::zeros((0,dim));
        if r0==0{
            for i in 0..(k_mesh[[r0]]){
               let mut usek=Array1::<f64>::zeros(dim);
               usek[[r0]]=(i as f64)/((k_mesh[[r0]]) as f64);
               let k0:Array2::<f64>=gen_kmesh_arr(&k_mesh,r0+1,usek);
               kvec.append(Axis(0),k0.view()).unwrap();
            }
            return kvec
        }else if r0<k_mesh.len()-1{
            for i in 0..(k_mesh[[r0]]){
               let mut kk=usek.clone();
               kk[[r0]]=(i as f64)/((k_mesh[[r0]]) as f64);
               let k0:Array2::<f64>=gen_kmesh_arr(&k_mesh,r0+1,kk);
               kvec.append(Axis(0),k0.view()).unwrap();
            }
            return kvec
        }else{
            for i in 0..(k_mesh[[r0]]){
               usek[[r0]]=(i as f64)/((k_mesh[[r0]]) as f64);
               kvec.push_row(usek.view()).unwrap();
            }
            return kvec
        }
    }
    let mut usek=Array1::<f64>::zeros(dim);
    gen_kmesh_arr(&k_mesh,0,usek)
}
#[allow(non_snake_case)]
pub fn gen_krange(k_mesh:&Array1::<usize>)->Array3::<f64>{
    let dim_r=k_mesh.len();
    let mut k_range=Array3::<f64>::zeros((0,dim_r,2));
    if dim_r==1{
        for i in 0..k_mesh[[0]]-1{
            let mut k=Array2::<f64>::zeros((dim_r,2));
            k[[0,0]]=(i as f64)/((k_mesh[[0]]-1) as f64);
            k[[0,1]]=((i+1) as f64)/((k_mesh[[0]]-1) as f64);
            k_range.push(Axis(0),k.view()).unwrap();
        }
    }else if dim_r==2{
        for i in 0..k_mesh[[0]]-1{
            for j in 0..k_mesh[[1]]-1{
                let mut k=Array2::<f64>::zeros((dim_r,2));
                k[[0,0]]=(i as f64)/((k_mesh[[0]]-1) as f64);
                k[[0,1]]=((i+1) as f64)/((k_mesh[[0]]-1) as f64);
                k[[1,0]]=(j as f64)/((k_mesh[[1]]-1) as f64);
                k[[1,1]]=((j+1) as f64)/((k_mesh[[1]]-1) as f64);
                k_range.push(Axis(0),k.view()).unwrap();
            }
        }
    }else if dim_r==3{
        for i in 0..k_mesh[[0]]-1{
            for j in 0..k_mesh[[1]]-1{
                for ks in 0..k_mesh[[2]]-1{
                    let mut k=Array2::<f64>::zeros((dim_r,2));
                    k[[0,0]]=(i as f64)/((k_mesh[[0]]-1) as f64);
                    k[[0,1]]=((i+1) as f64)/((k_mesh[[0]]-1) as f64);
                    k[[1,0]]=(j as f64)/((k_mesh[[1]]-1) as f64);
                    k[[1,1]]=((j+1) as f64)/((k_mesh[[1]]-1) as f64);
                    k[[2,0]]=(ks as f64)/((k_mesh[[1]]-1) as f64);
                    k[[2,1]]=((ks+1) as f64)/((k_mesh[[1]]-1) as f64);
                    k_range.push(Axis(0),k.view()).unwrap();
                }
            }
        }
    }else{
        panic!("Wrong, the dim should be 1,2 or 3, but you give {}",dim_r);
    }
    k_range
}

#[allow(non_snake_case)]
pub fn comm(A:&Array2::<Complex<f64>>,B:&Array2::<Complex<f64>>)->Array2::<Complex<f64>>{
    //! 做 $[A,B]$ 对易操作
    //let A0=A.clone();
    //let B0=B.clone();
    //let C=&A.dot(B);
    //let D=&B.dot(A);
    //C-D
    &A.dot(B)-&B.dot(A)
}
#[allow(non_snake_case)]
pub fn anti_comm(A:&Array2::<Complex<f64>>,B:&Array2::<Complex<f64>>)->Array2::<Complex<f64>>{
    //! 做 $\\\{A,B\\\}$ 反对易操作
    //let A0=A.clone();
    //let B0=B.clone();
    A.dot(B)+B.dot(A)
}
pub fn draw_heatmap(data: Array2<f64>,name:&str) {
    //!这个函数是用来画热图的, 给定一个二维矩阵, 会输出一个像素图片
    use gnuplot::{Figure, AxesCommon, AutoOption::Fix,HOT,RAINBOW};
    let mut fg = Figure::new();
    let (width, height):(usize,usize) = (data.shape()[1], data.shape()[0]);
    let mut heatmap_data = vec![];

    for i in 0..height {
        for j in 0..width {
            heatmap_data.push(data[(i, j)]);
        }
    }
    let axes = fg.axes2d();
    axes.set_title("Heatmap", &[]);
    axes.set_cb_label("Values", &[]);
    axes.set_palette(RAINBOW);
    axes.image(heatmap_data.iter(), width, height,None, &[]);
    let size=data.shape();
    let axes=axes.set_x_range(Fix(0.0), Fix((size[0]-1) as f64));
    let axes=axes.set_y_range(Fix(0.0), Fix((size[1]-1) as f64));
    let axes=axes.set_aspect_ratio(Fix(1.0));
    fg.set_terminal("pdfcairo",name);
    fg.show().expect("Unable to draw heatmap");
}

pub fn adapted_integrate_quick(f0:&dyn Fn(&Array1::<f64>)->f64,k_range:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
    //!对于任意维度的积分 n, 我们的将区域刨分成 n+1面体的小块, 然后用线性插值来近似这个n+1的积分结果
    //!设被积函数为 $f(x_1,x_2,...,x_n)$, 存在 $n+1$ 个点 $(y_{01},y_{02},\cdots y_{0n})\cdots(y_{n1},y_{n2}\cdots y_{nn})$, 对应的值为 $z_0,z_1,...,z_n$
    //!这样我们就能得到这一块积分的近似值为 $$ \f{1}{(n+1)!}\times\sum_{i=0}^n z_i *\dd V.$$ 其中$\dd V$ 是正 $n+1$ 面体的体积.

    let dim=k_range.len_of(Axis(0));
    if dim==1{
        //对于一维情况, 我们就是用梯形算法的 (a+b)*h/2, 这里假设的是函数的插值为线性插值.
        let mut use_range=vec![(k_range.clone(),re_err,ab_err)];
        let mut result=0.0;
        while let Some((k_range,re_err,ab_err))=use_range.pop() {
            let kvec_l:Array1::<f64>=arr1(&[k_range[[0,0]]]);
            let kvec_r:Array1::<f64>=arr1(&[k_range[[0,1]]]);
            let kvec_m:Array1::<f64>=arr1(&[(k_range[[0,1]]+k_range[[0,0]])/2.0]);
            let dk:f64=k_range[[0,1]]-k_range[[0,0]];
            let y_l:f64=f0(&kvec_l);
            let y_r:f64=f0(&kvec_r);
            let y_m:f64=f0(&kvec_m);
            let all:f64=(y_l+y_r)*dk/2.0;
            let all_1=(y_l+y_m)*dk/4.0;
            let all_2=(y_r+y_m)*dk/4.0;
            let err=all_1+all_2-all;
            let abs_err= if ab_err>all*re_err{ab_err} else {all*re_err};
            if err< abs_err{
                result+=all_1+all_2;
            }else{
                let k_range_l=arr2(&[[kvec_l[[0]],kvec_m[[0]]]]);
                let k_range_r=arr2(&[[kvec_m[[0]],kvec_r[[0]]]]);
                use_range.push((k_range_l.clone(),re_err,ab_err/2.0));
                use_range.push((k_range_r.clone(),re_err,ab_err/2.0));
            }
        }
        return result;
    }else if dim==2{
    //对于二维, 我们依旧假设线性插值, 这样我们考虑的就是二维平面上的三角形上的任意一点的值是到其余三个点的距离的加权系数的平均值, 我们将四边形变成两个三角形来考虑.
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第一个三角形
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第二个三角形
        fn adapt_integrate_triangle(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64,s1:f64,s2:f64,s3:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut result=0.0;
            let mut use_kvec=vec![(kvec.clone(),re_err,ab_err,s1,s2,s3)];
            while let Some((kvec,re_err,ab_err,s1,s2,s3))=use_kvec.pop() {
                let S:f64=((kvec[[1,0]]*kvec[[2,1]]-kvec[[2,0]]*kvec[[1,1]])-(kvec[[0,0]]*kvec[[2,1]]-kvec[[0,1]]*kvec[[2,0]])+(kvec[[0,0]]*kvec[[1,1]]-kvec[[1,0]]*kvec[[0,1]])).abs();
                let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                let sm:f64=f0(&kvec_m.to_owned());

                let mut kvec_1=Array2::<f64>::zeros((0,2));
                kvec_1.push_row(kvec.row(0));
                kvec_1.push_row(kvec.row(1));
                kvec_1.push_row(kvec_m.view());

                let mut kvec_2=Array2::<f64>::zeros((0,2));
                kvec_2.push_row(kvec.row(0));
                kvec_2.push_row(kvec_m.view());
                kvec_2.push_row(kvec.row(2));

                let mut kvec_3=Array2::<f64>::zeros((0,2));
                kvec_3.push_row(kvec_m.view());
                kvec_3.push_row(kvec.row(1));
                kvec_3.push_row(kvec.row(2));

                let all:f64=(s1+s2+s3)*S/6.0;
                let all_new:f64=all/3.0*2.0+sm*S/6.0;
                let abs_err:f64= if ab_err>all*re_err{ab_err} else {all*re_err};
                if (all_new-all).abs() > abs_err && S>1e-8{
                   use_kvec.push((kvec_1.clone(),re_err,ab_err/3.0,s1,s2,sm));
                   use_kvec.push((kvec_2.clone(),re_err,ab_err/3.0,s1,sm,s3));
                   use_kvec.push((kvec_3.clone(),re_err,ab_err/3.0,sm,s2,s3));
                }else{
                    result+=all_new; 
                }
            }
            result
        }
        let s1=f0(&arr1(&[k_range.row(0)[0],k_range.row(1)[0]]));
        let s2=f0(&arr1(&[k_range.row(0)[1],k_range.row(1)[0]]));
        let s3=f0(&arr1(&[k_range.row(0)[0],k_range.row(1)[1]]));
        let s4=f0(&arr1(&[k_range.row(0)[1],k_range.row(1)[1]]));
        let all_1=adapt_integrate_triangle(f0,&area_1,re_err,ab_err/2.0,s1,s2,s3);
        let all_2=adapt_integrate_triangle(f0,&area_2,re_err,ab_err/2.0,s4,s2,s3);
        return all_1+all_2;
    }else if dim==3{
    //对于三位情况, 需要用到四面体, 所以需要先将6面体变成6个四面体
        fn adapt_integrate_tetrahedron(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64,s1:f64,s2:f64,s3:f64,s4:f64,S:f64)->f64{
            //这个函数是用来进行自适应算法的
            let mut result=0.0;
            let mut use_kvec=vec![(kvec.clone(),re_err,ab_err,s1,s2,s3,s4,S)];
            while let Some((kvec,re_err,ab_err,s1,s2,s3,s4,S))=use_kvec.pop() {
                let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                let sm=f0(&kvec_m.to_owned());
                /////////////////////////
                let mut kvec_1=Array2::<f64>::zeros((0,3));
                kvec_1.push_row(kvec.row(0));
                kvec_1.push_row(kvec.row(1));
                kvec_1.push_row(kvec.row(2));
                kvec_1.push_row(kvec_m.view());
                
                let mut kvec_2=Array2::<f64>::zeros((0,3));
                kvec_2.push_row(kvec.row(0));
                kvec_2.push_row(kvec.row(1));
                kvec_2.push_row(kvec_m.view());
                kvec_2.push_row(kvec.row(3));

                let mut kvec_3=Array2::<f64>::zeros((0,3));
                kvec_3.push_row(kvec.row(0));
                kvec_3.push_row(kvec_m.view());
                kvec_3.push_row(kvec.row(2));
                kvec_3.push_row(kvec.row(3));

                let mut kvec_4=Array2::<f64>::zeros((0,3));
                kvec_4.push_row(kvec_m.view());
                kvec_4.push_row(kvec.row(1));
                kvec_4.push_row(kvec.row(2));
                kvec_4.push_row(kvec.row(3));

                let all=(s1+s2+s3+s4)*S/24.0;
                let all_new=all/4.0*3.0+sm*S/24.0;
                let S1=S/4.0;
                let abs_err= if ab_err>all*re_err{ab_err} else {all*re_err};
                if (all_new-all).abs()> abs_err && S > 1e-9{
                    use_kvec.push((kvec_1.clone(),re_err,ab_err*0.25,s1,s2,s3,sm,S1));
                    use_kvec.push((kvec_2.clone(),re_err,ab_err*0.25,s1,s2,sm,s4,S1));
                    use_kvec.push((kvec_3.clone(),re_err,ab_err*0.25,s1,sm,s3,s4,S1));
                    use_kvec.push((kvec_4.clone(),re_err,ab_err*0.25,sm,s2,s3,s4,S1));
                }else{
                    result+=all_new;
                }
            }
            result
        }
        let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]]]);//第一个四面体
        let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第二个四面体
        let area_3:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第三个四面体
        let area_4:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第四个四面体
        let area_5:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                        [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]],
                                        [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                        [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]]]);//第五个四面体
        let s1=f0(&area_1.row(0).to_owned());
        let s2=f0(&area_1.row(1).to_owned());
        let s3=f0(&area_2.row(0).to_owned());
        let s4=f0(&area_1.row(2).to_owned());
        let s5=f0(&area_1.row(3).to_owned());
        let s6=f0(&area_3.row(0).to_owned());
        let s7=f0(&area_2.row(3).to_owned());
        let s8=f0(&area_4.row(0).to_owned());
        let V=(k_range[[0,1]]-k_range[[0,0]])*(k_range[[1,1]]-k_range[[1,0]])*(k_range[[2,1]]-k_range[[2,0]]);
        let all_1=adapt_integrate_tetrahedron(f0,&area_1,re_err,ab_err/6.0,s1,s2,s4,s5,V/6.0);
        let all_2=adapt_integrate_tetrahedron(f0,&area_2,re_err,ab_err/6.0,s3,s2,s4,s7,V/6.0);
        let all_3=adapt_integrate_tetrahedron(f0,&area_3,re_err,ab_err/6.0,s6,s2,s5,s7,V/6.0);
        let all_4=adapt_integrate_tetrahedron(f0,&area_4,re_err,ab_err/6.0,s8,s5,s4,s7,V/6.0);
        let all_5=adapt_integrate_tetrahedron(f0,&area_5,re_err,ab_err/3.0,s5,s7,s4,s2,V/3.0);
        return all_1+all_2+all_3+all_4+all_5
    }else{
        panic!("wrong, the row_dim if k_range must be 1,2 or 3, but you's give {}",dim);
    }
}
/*
pub fn tetrahedral_integrate(k_mesh:Array1<f64>,data:Array2<f64>){
}
pub fn tetrahedral_integrate(f0:&dyn Fn(&Array1::<f64>)->Array1::<f64>,k_mash:&Array1::<f64>)->f64{
    0
}
*/
macro_rules! update_hamiltonian {
    //这个代码是用来更新哈密顿量的, 判断是否有自旋, 以及要更新的 ind_i, ind_j,
    //输入一个哈密顿量, 返回一个新的哈密顿量
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {
        {if $spin {
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
                /*
                4 => {
                    $new_ham[[$ind_i, $ind_j + $norb]] = $tmp;
                }
                5 => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp;
                }
                */
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
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {
        {if $spin {
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
                /*
                4 => {
                    $new_ham[[$ind_i, $ind_j + $norb]] += $tmp;
                }
                5 => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp;
                }
                */
                _ => todo!(),
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] += $tmp;
        }
        $new_ham
    }};
}
#[macro_export]
macro_rules! plot{
    ($x0:expr,$y0:expr,$name:expr)=>{{
        use gnuplot::{Figure, Caption, Color};
        use gnuplot::{AxesCommon};
        use gnuplot::AutoOption::*;
        use gnuplot::Tick::*;
        let mut fg = Figure::new();
        let x:Vec<f64>=$x0.to_vec();
        let axes=fg.axes2d();
        /*
        let n=x.len();
        if $y0.ndim()==2{
            let n0=$y0.len_of(Axis(1));
            for i in 0..n0{
                let y:Vec<f64>=$y0.slice(s![..,i]).to_owned().to_vec();
                axes.lines(&x, &y, &[Color("black")]);
            }
        }else{
            let y:Vec<f64>=$y0.to_vec();
            axes.lines(&x, &y, &[Color("black")]);
        }
        */

        let y:Vec<f64>=$y0.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        fg.set_terminal("pdfcairo", &$name);
        fg.show();
    }};
}



///An example
///
///```
///use gnuplot::{Color,Figure, AxesCommon, AutoOption::Fix,HOT};
///use gnuplot::Major;
///use ndarray::*;
///use ndarray::prelude::*;
///use num_complex::Complex;
///use Rustb::*;
///
///fn graphene(){
///    let li:Complex<f64>=1.0*Complex::i();
///    let t1=1.0+0.0*li;
///    let t2=0.1+0.0*li;
///    let t3=0.0+0.0*li;
///    let delta=0.5;
///    let dim_r:usize=2;
///    let norb:usize=2;
///    let lat=arr2(&[[3.0_f64.sqrt(),-1.0],[3.0_f64.sqrt(),1.0]]);
///    let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
///    let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
///    model.set_onsite(arr1(&[delta,-delta]),0);
///    model.add_hop(t1,0,1,&array![0,0],0);
///    model.add_hop(t1,0,1,&array![-1,0],0);
///    model.add_hop(t1,0,1,&array![0,-1],0);
///    model.add_hop(t2,0,0,&array![1,0],0);
///    model.add_hop(t2,1,1,&array![1,0],0);
///    model.add_hop(t2,0,0,&array![0,1],0);
///    model.add_hop(t2,1,1,&array![0,1],0);
///    model.add_hop(t2,0,0,&array![1,-1],0);
///    model.add_hop(t2,1,1,&array![1,-1],0);
///    model.add_hop(t3,0,1,&array![1,-1],0);
///    model.add_hop(t3,0,1,&array![-1,1],0);
///    model.add_hop(t3,0,1,&array![-1,-1],0);
///    let nk:usize=1001;
///    let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[0.0,0.0]];
///    let path=arr2(&path);
///    let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
///    let (eval,evec)=model.solve_all_parallel(&k_vec);
///    let label=vec!["G","K","M","G"];
///    let (k_vec,k_dist,k_node)=model.k_path(&path,nk); //generate the k vector 
///    let eval=model.solve_band_all_parallel(&k_vec);  //calculate the bands
///    let mut fg = Figure::new();
///    let x:Vec<f64>=k_dist.to_vec();
///    let axes=fg.axes2d();
///    for i in 0..model.nsta{
///        let y:Vec<f64>=eval.slice(s![..,i]).to_owned().to_vec();
///        axes.lines(&x, &y, &[Color("black")]);
///    }
///    let axes=axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len()-1]]));
///    let label=label.clone();
///    let mut show_ticks=Vec::new();
///    for i in 0..k_node.len(){
///        let A=k_node[[i]];
///        let B=label[i];
///        show_ticks.push(Major(A,Fix(B)));
///    }
///    axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
///    let k_node=k_node.to_vec();
///    let mut jpg_name=String::new();
///    jpg_name.push_str("band.jpg");
///    fg.set_terminal("jpeg", &jpg_name);
///    fg.show();
///
///    //start to draw the band structure
///    //Starting to calculate the edge state, first is the zigzag state
///    let nk:usize=501;
///    let U=arr2(&[[1.0,1.0],[-1.0,1.0]]);
///    let super_model=model.make_supercell(&U);
///    let zig_model=super_model.cut_piece(100,0);
///    let path=[[0.0,0.0],[0.0,0.5],[0.0,1.0]];
///    let path=arr2(&path);
///    let label=vec!["G","M","G"];
///    zig_model.show_band(&path,&label,nk,"graphene_zig");
///    //Starting to calculate the DOS of graphene
///    let nk:usize=101;
///    let kmesh=arr1(&[nk,nk]);
///    let E_min=-3.0;
///    let E_max=3.0;
///    let E_n=1000;
///    let (E0,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-2);
///    //start to show DOS
///    let mut fg = Figure::new();
///    let x:Vec<f64>=E0.to_vec();
///    let axes=fg.axes2d();
///    let y:Vec<f64>=dos.to_vec();
///    axes.lines(&x, &y, &[Color("black")]);
///    let mut show_ticks=Vec::<String>::new();
///    let mut pdf_name=String::new();
///    pdf_name.push_str("dos.jpg");
///    fg.set_terminal("pdfcairo", &pdf_name);
///    fg.show();
///}
///```
///
///
impl basis<'_> for Model{
    fn tb_model(dim_r:usize,lat:Array2::<f64>,orb:Array2::<f64>,spin:bool,atom:Option<Array2::<f64>>,atom_list:Option<Vec<usize>>)->Model{
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
        let norb:usize=orb.len_of(Axis(0));
        let mut nsta:usize=norb;
        if spin{
            nsta*=2;
        }
        let mut new_atom_list:Vec<usize>=vec![1];
        let mut new_atom=Array2::<f64>::zeros((0,dim_r));
        if lat.len_of(Axis(1)) != dim_r{
            panic!("Wrong, the lat's second dimension's length must equal to dim_r") 
        }
        if lat.len_of(Axis(0)) != lat.len_of(Axis(1)) {
            panic!("Wrong, the lat's second dimension's length must less than first dimension's length") 
        }
        let mut natom:usize=0;
        if atom !=None && atom_list !=None{
            new_atom=atom.unwrap();
            natom=new_atom.len_of(Axis(0));
            new_atom_list=atom_list.unwrap();
        }else if atom_list !=None || atom != None{
            panic!("Wrong, the atom and atom_list is not all None, please correspondence them");
        }else if atom_list==None && atom==None{
            //通过判断轨道是不是离得太近而判定是否属于一个原子,
            //这种方法只适用于wannier90不开最局域化
            new_atom.push_row(orb.row(0));
            natom+=1;
            for i in 1..norb{
                if (orb.row(i).to_owned()-new_atom.row(new_atom.nrows()-1).to_owned()).norm_l2()>1e-2{
                    new_atom.push_row(orb.row(i));
                    new_atom_list.push(1);
                    natom+=1;
                }else{
                    let last=new_atom_list.pop().unwrap();
                    new_atom_list.push(last+1);
                }
            }
        }
        let ham=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
        let hamR=Array2::<isize>::zeros((1,dim_r));
        let mut rmatrix=Array4::<Complex<f64>>::zeros((1,dim_r,nsta,nsta));
        for i in 0..norb {
            for r in 0..dim_r{
                rmatrix[[0,r,i,i]]=Complex::<f64>::from(orb[[i,r]]);
                if spin{
                    rmatrix[[0,r,i+norb,i+norb]]=Complex::<f64>::from(orb[[i,r]]);
                }
            }
        }
        let mut model=Model{
            dim_r,
            norb,
            nsta,
            natom,
            spin,
            lat,
            orb,
            atom:new_atom,
            atom_list:new_atom_list,
            ham,
            hamR,
            rmatrix,
        };
        model
    }
    #[allow(non_snake_case)]
    fn set_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:&Array1::<isize>,pauli:isize){
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

        if pauli != 0 && self.spin==false{
            println!("Wrong, if spin is Ture and pauli is not zero, the pauli is not use")
        }
        if R.len()!=self.dim_r{
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i>=self.norb ||ind_j>=self.norb{
            panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
        }
        let R_exist=find_R(&self.hamR,&R);
        let negative_R=-R.clone();
        let negative_R_exist=find_R(&self.hamR,&negative_R);
        if R_exist {
            let index=index_R(&self.hamR,&R);
            if self.ham[[index,ind_i,ind_j]]!=Complex::new(0.0,0.0){
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_i,ind_j]])
            }
            update_hamiltonian!(self.spin,pauli,tmp,self.ham.slice_mut(s![index,..,..]),ind_i,ind_j,self.norb);
            if negative_R_exist && ind_i != ind_j{
                update_hamiltonian!(self.spin,pauli,tmp.conj(),self.ham.slice_mut(s![0,..,..]),ind_j,ind_i,self.norb);
            }
            if ind_i==ind_j && tmp.im !=0.0 && (pauli==0 ||pauli==3) && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            if self.ham[[index,ind_j,ind_i]]!=Complex::new(0.0,0.0){
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_j,ind_i]])
            }
            update_hamiltonian!(self.spin,pauli,tmp.conj(),self.ham.slice_mut(s![index,..,..]),ind_j,ind_i,self.norb);

        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));

            let new_ham=update_hamiltonian!(self.spin,pauli,tmp,new_ham,ind_i,ind_j,self.norb);
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }


    #[allow(non_snake_case)]
    fn add_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:&Array1::<isize>,pauli:isize){
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp 
        if pauli != 0 && self.spin==false{
            println!("Wrong, if spin is Ture and pauli is not zero, the pauli is not use")
        }
        if R.len()!=self.dim_r{
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i>=self.norb ||ind_j>=self.norb{
            panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
        }
        let R_exist=find_R(&self.hamR,&R);
        let negative_R=-R.clone();
        let negative_R_exist=find_R(&self.hamR,&negative_R);
        if R_exist {
            let index=index_R(&self.hamR,&R);
            add_hamiltonian!(self.spin,pauli,tmp,self.ham.slice_mut(s![index,..,..]),ind_i,ind_j,self.norb);
            if negative_R_exist && ind_i !=ind_j{
                add_hamiltonian!(self.spin,pauli,tmp.conj(),self.ham.slice_mut(s![0,..,..]),ind_j,ind_i,self.norb);
            }
            if ind_i==ind_j && tmp.im !=0.0 && (pauli==0 ||pauli==3) && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            add_hamiltonian!(self.spin,pauli,tmp.conj(),self.ham.slice_mut(s![index,..,..]),ind_j,ind_i,self.norb);
        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let new_ham=add_hamiltonian!(self.spin,pauli,tmp,new_ham,ind_i,ind_j,self.norb);
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }


    #[allow(non_snake_case)]
    fn add_element(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:&Array1::<isize>){
        //!参数和 set_hop 一致, 但是 $\bra{i\bm 0}\hat H\ket{j\bm R}$+=tmp 
        if R.len()!=self.dim_r{
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i>=self.nsta ||ind_j>=self.nsta{
            panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
        }
        let R_exist=find_R(&self.hamR,&R);
        let negative_R=-R.clone();
        let negative_R_exist=find_R(&self.hamR,&negative_R);
        if R_exist {
            let index=index_R(&self.hamR,&R);
            self.ham[[index,ind_i,ind_j]]=tmp;
            if negative_R_exist && ind_i !=ind_j{
                self.ham[[index,ind_j,ind_i]]=tmp.conj();
            }
            if ind_i==ind_j && tmp.im !=0.0 && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            self.ham[[index,ind_j,ind_i]]=tmp.conj();
        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            new_ham[[ind_i,ind_j]]=tmp;
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }

    #[allow(non_snake_case)]
    fn set_onsite(&mut self, tmp:Array1::<f64>,pauli:isize){
        //! 直接对对角项进行设置
        if tmp.len()!=self.norb{
            panic!("Wrong, the norb is {}, however, the onsite input's length is {}",self.norb,tmp.len())
        }
        for (i,item) in tmp.iter().enumerate(){
            self.set_onsite_one(*item,i,pauli)
        }
    }
    #[allow(non_snake_case)]
    fn set_onsite_one(&mut self, tmp:f64,ind:usize,pauli:isize){
        //!对  $\bra{i\bm 0}\hat H\ket{i\bm 0}$ 进行设置
        let R=Array1::<isize>::zeros(self.dim_r);
        self.set_hop(Complex::new(tmp,0.0),ind,ind,&R,pauli)
    }
    fn del_hop(&mut self,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize) {
        //! 删除 $\bra{i\bm 0}\hat H\ket{j\bm R}$
        if R.len()!=self.dim_r{
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i>=self.norb ||ind_j>=self.norb{
            panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
        }
        self.set_hop(Complex::new(0.0,0.0),ind_i,ind_j,&R,pauli);
    }

    #[allow(non_snake_case)]
    fn k_path(&self,path:&Array2::<f64>,nk:usize)->(Array2::<f64>,Array1::<f64>,Array1::<f64>){
        //!根据高对称点来生成高对称路径, 画能带图
        if self.dim_r==0{
            panic!("the k dimension of the model is 0, do not use k_path")
        }
        let n_node:usize=path.len_of(Axis(0));
        if self.dim_r != path.len_of(Axis(1)){
            panic!("Wrong, the path's length along 1 dimension must equal to the model's dimension")
        }
        let k_metric=(self.lat.dot(&self.lat.t())).inv().unwrap();
        let mut k_node=Array1::<f64>::zeros(n_node);
        for n in 1..n_node{
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk=path.row(n).to_owned()-path.slice(s![n-1,..]).to_owned();
            let a=k_metric.dot(&dk);
            let dklen=dk.dot(&a).sqrt();
            k_node[[n]]=k_node[[n-1]]+dklen;
        }
        let mut node_index:Vec<usize>=vec![0];
        for n in 1..n_node-1{
            let frac=k_node[[n]]/k_node[[n_node-1]];
            let a=(frac*((nk-1) as f64).round()) as usize;
            node_index.push(a)
        }
        node_index.push(nk-1);
        let mut k_dist=Array1::<f64>::zeros(nk);
        let mut k_vec=Array2::<f64>::zeros((nk,self.dim_r));
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&path.row(0));
        for n in 1..n_node {
            let n_i=node_index[n-1];
            let n_f=node_index[n];
            let kd_i=k_node[[n-1]];
            let kd_f=k_node[[n]];
            let k_i=path.row(n-1);
            let k_f=path.row(n);
            for j in n_i..n_f+1{
                let frac:f64= ((j-n_i) as f64)/((n_f-n_i) as f64);
                k_dist[[j]]=kd_i + frac*(kd_f-kd_i);
                k_vec.row_mut(j).assign(&((1.0-frac)*k_i.to_owned() +frac*k_f.to_owned()));

            }
        }
        (k_vec,k_dist,k_node)
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    fn gen_ham(&self,kvec:&Array1::<f64>)->Array2::<Complex<f64>>{
        //!这个是做傅里叶变换, 将实空间的哈密顿量变换到倒空间的哈密顿量
        //!
        //!具体来说, 就是
        //!$$H_{mn,\bm k}=\bra{m\bm k}\hat H\ket{n\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{-i(\bm R-\bm\tau_i+\bm \tau_j)\cdot\bm k}$$
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0:Array1::<f64>=self.orb.dot(kvec);
        let U0:Array1::<Complex<f64>>=U0.mapv(|x| Complex::<f64>::new(x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0:Array1::<Complex<f64>>=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.mapv(|x| x as f64)).dot(kvec).mapv(|x| Complex::<f64>::new(x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        let ham0=self.ham.slice(s![0,..,..]);
        /*
        for i in 1..nR{
            hamk=hamk+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]];
        }
        */
        hamk=self.ham.axis_iter(Axis(0)).skip(1).zip(Us.iter().skip(1)).fold(hamk,|acc,(A,us)| {acc+&A**us});
        hamk=&ham0+&hamk.mapv(|x| x.conj()).t()+&hamk;
        hamk=hamk.dot(&U);
        let re_ham=U.mapv(|x| x.conj()).t().dot(&hamk);
        re_ham
    }
    #[allow(non_snake_case)]
    fn gen_r(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>{
        //!和 gen_ham 类似, 将 $\hat{\bm r}$ 进行傅里叶变换
        //!
        //!$$\bm r_{mn,\bm k}=\bra{m\bm k}\hat{\bm r}\ket{n\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat{\bm r}\ket{n\bm R}e^{-i(\bm R-\bm\tau_i+\bm \tau_j)\cdot\bm k}$$
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0=self.orb.dot(kvec);
        let U0=U0.mapv(|x| Complex::<f64>::new(x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.mapv(|x| x as f64)).dot(kvec).mapv(|x| Complex::<f64>::new(x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut rk=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));
        let r0=self.rmatrix.slice(s![0,..,..,..]).to_owned();
        if self.rmatrix.len_of(Axis(0))==1{
            return self.rmatrix.slice(s![0,..,..,..]).to_owned()
        }else{
            rk=self.rmatrix.axis_iter(Axis(0)).skip(1).zip(Us.iter().skip(1)).fold(rk,|acc,(ham,us)|{acc+&ham**us});
            for i in 0..self.dim_r{
                let use_rk=rk.slice(s![i,..,..]);
                let use_rk:Array2::<Complex<f64>>=&r0.slice(s![i,..,..])+&use_rk.mapv(|x| x.conj()).t()+&use_rk;
                //接下来向位置算符添加轨道的位置项
                let use_rk=use_rk.dot(&U); //
                rk.slice_mut(s![i,..,..]).assign(&(U.mapv(|x| x.conj()).t().dot(&use_rk)));
            }
            return rk
        }
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
    ///这里的 $\\mathcal A_{\bm k}$ 的定义为 $$\\mathcal A_{\bm k,\ap,mn}=-i\sum_{\bm R}r_{mn,\ap}(\bm R)e^{i\bm k\cdot(\bm R+\bm\tau_m-\bm\tau_{n})}+i\tau_{n\ap}\dt_{mn}$$
    #[allow(non_snake_case)]
    #[inline(always)]
    fn gen_v(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0=self.orb.dot(kvec);
        let U0=U0.mapv(|x| Complex::<f64>::new(x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.mapv(|x| x as f64)).dot(kvec).mapv(|x| Complex::<f64>::new(x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut UU=Array3::<f64>::zeros((self.dim_r,self.nsta,self.nsta));
        let orb_real=self.orb.dot(&self.lat);
        if self.spin{
            for r in 0..self.dim_r{
                for i in 0..self.norb{
                    for j in 0..self.norb{
                        UU[[r,i,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i+self.norb,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i,j+self.norb]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i+self.norb,j+self.norb]]=-orb_real[[i,r]]+orb_real[[j,r]];
                    }
                }
            }
        }else{
            for r in 0..self.dim_r{
                for i in 0..self.norb{
                    for j in 0..self.norb{
                        UU[[r,i,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                    }
                }
            }
        }
        let UU=UU.mapv(|x| Complex::<f64>::new(0.0,x)); //UU[i,j]=-tau[i]+tau[j] 
        let mut v=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));//定义一个初始化的速度矩阵
        let R0=self.hamR.clone().mapv(|x| Complex::<f64>::new(x as f64,0.0));
        let R0=R0.dot(&self.lat.mapv(|x| Complex::new(x,0.0)));
        for i0 in 0..self.dim_r{
            let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let mut vv=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            (vv,hamk)=self.ham.axis_iter(Axis(0)).skip(1).zip(Us.iter().skip(1).zip(R0.column(i0).iter().skip(1))).fold((vv,hamk),|(acc1,acc2),(ham,(us,r))|{ (acc1+&ham**us**r*Complex::i(),acc2+&ham**us)});

            vv=vv.clone().reversed_axes().mapv(|x| x.conj())+vv;
            let hamk0=hamk.clone();
            let hamk=hamk+&self.ham.slice(s![0,..,..])+hamk0.mapv(|x| x.conj()).t();
            vv=vv+hamk.clone()*&UU.slice(s![i0,..,..]);
            let vv=vv.dot(&U); //接下来两步填上轨道坐标导致的相位
            let vv=U.mapv(|x| x.conj()).t().dot(&vv);
            v.slice_mut(s![i0,..,..]).assign(&vv);
        }
        //到这里, 我们完成了 sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)} 的计算
        //接下来, 我们计算贝利联络 A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
        if self.rmatrix.len_of(Axis(0))!=1 {
            let hamk=self.gen_ham(&kvec);
            let rk=self.gen_r(&kvec); 
            for i in 0..self.dim_r{
                let mut UU=self.orb.slice(s![..,i]).to_owned().clone(); //我们首先提取出alpha方向的轨道的位置
                if self.spin{
                    let UUU=UU.clone();
                    UU.append(Axis(0),UUU.view()).unwrap();
                }
                let mut UU=Array2::from_diag(&UU); //将其化为矩阵
                let A=&rk.slice(s![i,..,..])-&UU;
                let A=comm(&hamk,&A)*Complex::i();
                //let vv=v.slice(s![i,..,..]).to_owned().clone();
                v.slice_mut(s![i,..,..]).add_assign(&A);
            }
        }
        v
    }
    #[allow(non_snake_case)]
    fn solve_band_onek(&self,kvec:&Array1::<f64>)->Array1::<f64>{
        //!求解单个k点的能带值
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
        } 
        let hamk=self.gen_ham(&kvec);
        /*
        let hamk_conj=hamk.clone().map(|x| x.conj());
        let hamk_conj=hamk_conj.t();
        let sum0=(hamk.clone()-hamk_conj).sum();
        if sum0.im()> 1e-8 || sum0.re() >1e-8{
            panic!("Wrong, hamiltonian is not hamilt");
        }
        */
        let eval = if let Ok(eigvals) = hamk.eigvalsh(UPLO::Lower) { eigvals } else { todo!() };
        eval
    }
    fn solve_band_all(&self,kvec:&Array2::<f64>)->Array2::<f64>{
        //!求解多个k点的能带值
        let nk=kvec.len_of(Axis(0));
        let eval:Vec<_>=kvec.axis_iter(Axis(0)).map(|x| {
            let eval=self.solve_band_onek(&x.to_owned()); 
            eval.to_vec()
            }).collect();
        let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
        band
    }
    #[allow(non_snake_case)]
    fn solve_band_all_parallel(&self,kvec:&Array2::<f64>)->Array2::<f64>{
        //!并行求解多个k点的能带值
        let nk=kvec.len_of(Axis(0));
        let eval:Vec<_>=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let eval=self.solve_band_onek(&x.to_owned()); 
            eval.to_vec()
            }).collect();
        let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
        band
    }
    #[allow(non_snake_case)]
    fn solve_onek(&self,kvec:&Array1::<f64>)->(Array1::<f64>,Array2::<Complex<f64>>){
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
        } 
        let hamk=self.gen_ham(&kvec);
        let (eval, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) { (eigvals, eigvecs) } else { todo!() };
        let evec=evec.reversed_axes().map(|x| x.conj());
        (eval,evec)
    }
    #[allow(non_snake_case)]
    fn solve_all(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
        let nk=kvec.len_of(Axis(0));
        let (eval,evec):(Vec<_>,Vec<_>)=kvec
            .axis_iter(Axis(0))
            .map(|x| {
                let (eval, evec) =self.solve_onek(&x.to_owned()); 
                (eval.to_vec(),evec.into_raw_vec())
                }).unzip();
        let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
        let vectors=Array3::from_shape_vec((nk, self.nsta,self.nsta), evec.into_iter().flatten().collect()).unwrap();
        (band,vectors)
    }
    #[allow(non_snake_case)]
    fn solve_all_parallel(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
        let nk=kvec.len_of(Axis(0));
        let (eval,evec):(Vec<_>,Vec<_>)=kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let (eval, evec) =self.solve_onek(&x.to_owned()); 
                (eval.to_vec(),evec.into_raw_vec())
                }).collect();
        let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
        let vectors=Array3::from_shape_vec((nk, self.nsta,self.nsta), evec.into_iter().flatten().collect()).unwrap();
        (band,vectors)
    }

    ///这个函数是用来将model的某个方向进行截断的
    ///
    ///num:截出多少个原胞
    ///
    ///dir:方向
    ///
    ///返回一个model, 其中 dir 和输入的model是一致的, 但是轨道数目和原子数目都会扩大num倍, 沿着dir方向没有胞间hopping.
    fn cut_piece(&self,num:usize,dir:usize)->Model{
        //! This function is used to truncate a certain direction of a model.
        //!
        //! Parameters:
        //! - num: number of unit cells to truncate.
        //! - dir: the direction to be truncated.
        //!
        //! Returns a new model with the same direction as the input model, but with the number of orbitals and atoms increased by a factor of "num". There is no inter-cell hopping along the "dir" direction.
        if num<1{
            panic!("Wrong, the num={} is less than 1",num);
        }
        if dir> self.dim_r{
            panic!("Wrong, the dir is larger than dim_r");
        }
        let mut new_orb=Array2::<f64>::zeros((self.norb*num,self.dim_r));//定义一个新的轨道
        let mut new_atom=Array2::<f64>::zeros((self.natom*num,self.dim_r));//定义一个新的原子
        let new_norb=self.norb*num;
        let new_nsta=self.nsta*num;
        let new_natom=self.natom*num;
        let mut new_atom_list:Vec<usize>=Vec::new();
        let mut new_lat=self.lat.clone();
        new_lat.row_mut(dir).assign(&(self.lat.row(dir).to_owned()*(num as f64)));
        for i in 0..num{
            for n in 0..self.norb{
                let mut use_orb=self.orb.row(n).to_owned();
                use_orb[[dir]]+=i as f64;
                use_orb[[dir]]=use_orb[[dir]]/(num as f64);
                new_orb.row_mut(i*self.norb+n).assign(&use_orb);
            }
            for n in 0..self.natom{
                let mut use_atom=self.atom.row(n).to_owned();
                use_atom[[dir]]+=i as f64;
                use_atom[[dir]]*=1.0/(num as f64);
                new_atom.row_mut(i*self.natom+n).assign(&use_atom);
                new_atom_list.push(self.atom_list[n]);
            }
        }
        let mut new_ham=Array3::<Complex<f64>>::zeros((1,new_nsta,new_nsta));
        let mut new_rmatrix=Array4::<Complex<f64>>::zeros((1,self.dim_r,new_nsta,new_nsta));
        let mut new_hamR=Array2::<isize>::zeros((1,self.dim_r));
        //新的轨道和原子构造完成, 开始构建哈密顿量
        //let new_model=tb_model(self.dim_r,new_norb,new_lat,new_orb,self.spin,Some(new_natom),Some(new_atom),Some(atom_list));
        //先尝试构建位置函数
        let exist_r=self.rmatrix.len_of(Axis(0))!=1;
        if self.rmatrix.len_of(Axis(0))==1{
            for i in 0..new_norb{
                for r in 0..self.dim_r{
                    new_rmatrix[[0,r,i,i]]=Complex::new(new_orb[[i,r]],0.0);
                    if self.spin{
                        new_rmatrix[[0,r,i+new_norb,i+new_norb]]=Complex::new(new_orb[[i,r]],0.0);
                    }
                }
            }

        }
        //开始正式构建哈密顿量
        let n_R=self.hamR.len_of(Axis(0));
        for n in 0..num{
            for i0 in 0..n_R{
                let mut ind_R:Array1::<isize>=self.hamR.row(i0).to_owned();
                let mut rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,new_nsta,new_nsta));
                let ham=if ind_R[[dir]]<0{//如果这个方向的ind_R 小于0, 将其变成大于0
                    ind_R*=-1;
                    let h0=self.ham.slice(s![i0,..,..]).mapv(|x| x.conj()).t().to_owned();
                    if exist_r{
                        rmatrix=self.rmatrix.slice(s![i0,..,..,..]).mapv(|x| x.conj());
                        rmatrix.swap_axes(1,2);
                    }
                    h0
                }else{
                    if exist_r{
                        rmatrix=self.rmatrix.slice(s![i0,..,..,..]).to_owned();
                    }
                    self.ham.slice(s![i0,..,..]).to_owned()
                };
                let ind:usize=(ind_R[[dir]]+(n as isize)) as usize;
                ind_R[[dir]]=0;
                if ind<num{
                    //开始构建哈密顿量
                    let R_exist=find_R(&new_hamR,&ind_R);
                    let negative_R=-ind_R.clone();
                    let negative_R_exist=find_R(&new_hamR,&negative_R);
                    let mut use_ham=Array2::<Complex<f64>>::zeros((new_nsta,new_nsta));
                    if self.spin{ //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                        let mut s=use_ham.slice_mut(s![n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                        let ham0=ham.slice(s![0..self.norb,0..self.norb]);
                        s.assign(&ham0);
                        let mut s=use_ham.slice_mut(s![(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),ind*self.norb..(ind+1)*self.norb]);
                        let ham0=ham.slice(s![self.norb..2*self.norb,0..self.norb]);
                        s.assign(&ham0);
                        let mut s=use_ham.slice_mut(s![n*self.norb..(n+1)*self.norb,(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                        let ham0=ham.slice(s![0..self.norb,self.norb..2*self.norb]);
                        s.assign(&ham0);
                        let mut s=use_ham.slice_mut(s![(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                        let ham0=ham.slice(s![self.norb..2*self.norb,self.norb..2*self.norb]);
                        s.assign(&ham0);

                        if R_exist{
                            let index=index_R(&new_hamR,&ind_R);
                            if index==0 && ind !=0{
                                let ham=ham.mapv(|x| x.conj()).reversed_axes();
                                let mut s=use_ham.slice_mut(s![n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                                let ham0=ham.slice(s![0..self.norb,0..self.norb]);
                                s.assign(&ham0);
                                let mut s=use_ham.slice_mut(s![(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),ind*self.norb..(ind+1)*self.norb]);
                                let ham0=ham.slice(s![self.norb..2*self.norb,0..self.norb]);
                                s.assign(&ham0);
                                let mut s=use_ham.slice_mut(s![n*self.norb..(n+1)*self.norb,(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                                let ham0=ham.slice(s![0..self.norb,self.norb..2*self.norb]);
                                s.assign(&ham0);
                                let mut s=use_ham.slice_mut(s![(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                                let ham0=ham.slice(s![self.norb..2*self.norb,self.norb..2*self.norb]);
                                s.assign(&ham0);
                            }
                        }
                    }else{
                        let mut s=use_ham.slice_mut(s![n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                        s.assign(&ham);
                        if R_exist{
                            let index=index_R(&new_hamR,&ind_R);
                            if index==0 && ind !=0{
                                let mut s=use_ham.slice_mut(s![ind*self.norb..(ind+1)*self.norb,n*self.norb..(n+1)*self.norb]);
                                s.assign(&(ham.mapv(|x| x.conj()).t()));
                            }
                        }
                    }
                    //开始对 r_matrix 进行操作
                    let mut use_rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,new_nsta,new_nsta));
                    if exist_r{
                        if self.spin{ //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                            let mut s=use_rmatrix.slice_mut(s![..,n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                            let rmatrix0=rmatrix.slice(s![..,0..self.norb,0..self.norb]);
                            s.assign(&rmatrix0);
                            let mut s=use_rmatrix.slice_mut(s![..,(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),ind*self.norb..(ind+1)*self.norb]);
                            let rmatrix0=rmatrix.slice(s![..,self.norb..2*self.norb,0..self.norb]);
                            s.assign(&rmatrix0);
                            let mut s=use_rmatrix.slice_mut(s![..,n*self.norb..(n+1)*self.norb,(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                            let rmatrix0=rmatrix.slice(s![..,0..self.norb,self.norb..2*self.norb]);
                            s.assign(&rmatrix0);
                            let mut s=use_rmatrix.slice_mut(s![..,(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                            let rmatrix0=rmatrix.slice(s![..,self.norb..2*self.norb,self.norb..2*self.norb]);
                            s.assign(&rmatrix0);

                            if R_exist{
                                let index=index_R(&new_hamR,&ind_R);
                                if index==0 && ind !=0{
                                    let mut rmatrix=rmatrix.mapv(|x| x.conj());
                                    rmatrix.swap_axes(1,2);
                                    let mut s=use_rmatrix.slice_mut(s![..,n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                                    let rmatrix0=rmatrix.slice(s![..,0..self.norb,0..self.norb]);
                                    s.assign(&rmatrix0);
                                    let mut s=use_rmatrix.slice_mut(s![..,(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),ind*self.norb..(ind+1)*self.norb]);
                                    let rmatrix0=rmatrix.slice(s![..,self.norb..2*self.norb,0..self.norb]);
                                    s.assign(&rmatrix0);
                                    let mut s=use_rmatrix.slice_mut(s![..,n*self.norb..(n+1)*self.norb,(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                                    let rmatrix0=rmatrix.slice(s![..,0..self.norb,self.norb..2*self.norb]);
                                    s.assign(&rmatrix0);
                                    let mut s=use_rmatrix.slice_mut(s![..,(new_norb+n*self.norb)..(new_norb+(n+1)*self.norb),(new_norb+ind*self.norb)..(new_norb+(ind+1)*self.norb)]);
                                    let rmatrix0=rmatrix.slice(s![..,self.norb..2*self.norb,self.norb..2*self.norb]);
                                    s.assign(&rmatrix0);
                                }
                            }
                        }else{
                            let mut s=use_rmatrix.slice_mut(s![..,n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                            s.assign(&rmatrix);
                            if R_exist{
                                let index=index_R(&new_hamR,&ind_R);
                                if index==0 && ind !=0{
                                    let mut rmatrix=rmatrix.mapv(|x| x.conj());
                                    rmatrix.swap_axes(1,2);
                                    let mut s=use_rmatrix.slice_mut(s![..,n*self.norb..(n+1)*self.norb,ind*self.norb..(ind+1)*self.norb]);
                                    s.assign(&rmatrix);
                                }
                            }
                        }
                    }
                    if R_exist{
                        let index=index_R(&new_hamR,&ind_R);
                        //let addham=new_ham.slice(s![index,..,..]).to_owned();
                        new_ham.slice_mut(s![index,..,..]).add_assign(&use_ham);
                        //let addr=new_rmatrix.slice(s![index,..,..,..]).to_owned();
                        new_rmatrix.slice_mut(s![index,..,..,..]).add_assign(&use_rmatrix);
                    }else if negative_R_exist{
                        let index=index_R(&new_hamR,&negative_R);
                        //let addham=new_ham.slice(s![index,..,..]).to_owned();
                        new_ham.slice_mut(s![index,..,..]).add_assign(&use_ham.t().map(|x| x.conj()));
                        //let mut addr=new_rmatrix.slice(s![index,..,..,..]).to_owned();
                        use_rmatrix.swap_axes(1,2);
                        new_rmatrix.slice_mut(s![index,..,..,..]).add_assign(&use_rmatrix.map(|x| x.conj()));
                    }else{
                        new_ham.push(Axis(0),use_ham.view());
                        new_hamR.push(Axis(0),ind_R.view());
                        new_rmatrix.push(Axis(0),use_rmatrix.view());
                    }
                }else{
                    continue
                }
            }
        }

        let mut model=Model{
            dim_r:self.dim_r,
            norb:new_norb,
            nsta:new_nsta,
            natom:new_natom,
            spin:self.spin,
            lat:new_lat,
            orb:new_orb,
            atom:new_atom,
            atom_list:new_atom_list,
            ham:new_ham,
            hamR:new_hamR,
            rmatrix:new_rmatrix,
        };
        model
    }

    fn cut_dot(&self,num:usize,shape:usize,dir:Option<Vec<usize>>)->Model{
        //! 这个是用来且角态或者切棱态的

        match self.dim_r{
            3 => {
                let dir =if dir == None{
                    println!("Wrong!, the dir is None, but model's dimension is 3, here we use defult 0,1 direction");
                    let dir=vec![0,1];
                    dir
                }else{
                    dir.unwrap()
                };
                let (old_model,use_orb_item,use_atom_item)=match shape{
                    3 =>{
                        let model_1=self.cut_piece(num+1,dir[0]);
                        let model_2=model_1.cut_piece(num+1,dir[1]);
                        let mut use_atom_item=Vec::<usize>::new();
                        let mut use_orb_item=Vec::<usize>::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if atom_position[[dir[0]]]+atom_position[[dir[1]]] > (num as f64)/(num as f64+1.0) {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    4 => {
                        let model_1=self.cut_piece(num+1,dir[0]);
                        let model_2=model_1.cut_piece(num+1,dir[1]);
                        let mut use_atom_item:Vec<usize>=Vec::new();
                        let mut use_orb_item:Vec<usize>=Vec::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            let num0=num as f64;
                            if atom_position[[dir[0]]]*(num0+1.0)/num0> 1.0+1e-5 
                                || atom_position[[dir[1]]]*(num0+1.0)/num0> 1.0+1e-5  {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    6 => {
                        let model_1=self.cut_piece(2*num,dir[0]);
                        let model_2=model_1.cut_piece(2*num,dir[1]);
                        let mut use_atom_item=Vec::<usize>::new();
                        let mut use_orb_item=Vec::<usize>::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if (atom_position[[1]]-atom_position[[dir[0]]] + 0.5 < 0.0) || (atom_position[[dir[0]]]-atom_position[[1]] + 0.5 < 0.0) {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    8 => {
                        let model_1=self.cut_piece(2*num,dir[0]);
                        let model_2=model_1.cut_piece(2*num,dir[1]);
                        let mut use_atom_item=Vec::<usize>::new();
                        let mut use_orb_item=Vec::<usize>::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if (atom_position[[dir[1]]]-atom_position[[dir[0]]] + 0.5 < 0.0) || 
                                (atom_position[[dir[0]]]-atom_position[[dir[1]]] + 0.5 < 0.0) ||
                                    (atom_position[[dir[1]]]+atom_position[[dir[0]]] < 0.5) || 
                                    (atom_position[[dir[1]]]-atom_position[[dir[0]]] > 0.5) {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    _=>{
                        panic!("Wrong, the shape only can be 3,4, 6,8");
                        todo!();
                    }
                };
                let natom=use_atom_item.len();
                let norb=use_orb_item.len();
                let mut new_atom=Array2::<f64>::zeros((natom,self.dim_r));
                let mut new_atom_list=Vec::<usize>::new();
                let mut new_orb=Array2::<f64>::zeros((norb,self.dim_r));
                for (i,use_i) in use_atom_item.iter().enumerate(){
                    new_atom.row_mut(i).assign(&old_model.atom.row(*use_i));
                    new_atom_list.push(old_model.atom_list[*use_i]);
                }
                for (i,use_i) in use_orb_item.iter().enumerate(){
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                }
                let mut new_model=Model::tb_model(self.dim_r,old_model.lat,new_orb,self.spin,Some(new_atom),Some(new_atom_list));
                let n_R=old_model.hamR.len_of(Axis(0));
                let mut new_ham=Array3::<Complex<f64>>::zeros((n_R,new_model.nsta,new_model.nsta));
                let mut new_hamR=Array2::<isize>::zeros((0,self.dim_r));
                let norb=new_model.norb;

                if self.spin{
                    let norb2=old_model.norb;
                    for (r,R) in old_model.hamR.axis_iter(Axis(0)).enumerate(){
                        new_hamR.push_row(R);
                        for (i,use_i) in use_orb_item.iter().enumerate(){
                            for (j,use_j) in use_orb_item.iter().enumerate(){
                                new_ham[[r,i,j]]=old_model.ham[[r,*use_i,*use_j]];
                                new_ham[[r,i+norb,j+norb]]=old_model.ham[[r,*use_i+norb2,*use_j+norb2]];
                                new_ham[[r,i+norb,j]]=old_model.ham[[r,*use_i+norb2,*use_j]];
                                new_ham[[r,i,j+norb]]=old_model.ham[[r,*use_i,*use_j+norb2]];
                            }
                        }
                    }
                }else{
                    for (r,R) in old_model.hamR.axis_iter(Axis(0)).enumerate(){
                        new_hamR.push_row(R);
                        for (i,use_i) in use_orb_item.iter().enumerate(){
                            for (j,use_j) in use_orb_item.iter().enumerate(){
                                new_ham[[r,i,j]]=old_model.ham[[r,*use_i,*use_j]];
                            }
                        }
                    }
                }
                new_model.ham=new_ham;
                new_model.hamR=new_hamR;
                let nsta=new_model.nsta;
                if self.rmatrix.len_of(Axis(0))==1{
                    for r in 0..self.dim_r{
                        let mut use_rmatrix=Array2::<Complex<f64>>::zeros((nsta,nsta));
                        for i in 0..norb{
                            use_rmatrix[[i,i]]=Complex::new(new_model.orb[[i,r]],0.0);
                        }
                        if new_model.spin{
                            for i in 0..norb{
                                use_rmatrix[[i+norb,i+norb]]=Complex::new(new_model.orb[[i,r]],0.0);
                            }
                        }
                        new_model.rmatrix.slice_mut(s![0,r,..,..]).assign(&use_rmatrix);
                    }
                }else{
                    let mut new_rmatrix=Array4::<Complex<f64>>::zeros((n_R,self.dim_r,new_model.nsta,new_model.nsta));
                    if old_model.spin{
                        let norb2=old_model.norb;
                        for r in 0..n_R{
                            for dim in 0..self.dim_r{
                                for (i,use_i) in use_orb_item.iter().enumerate(){
                                    for (j,use_j) in use_orb_item.iter().enumerate(){
                                        new_rmatrix[[r,dim,i,j]]=old_model.rmatrix[[r,dim,*use_i,*use_j]];
                                        new_rmatrix[[r,dim,i+norb,j+norb]]=old_model.rmatrix[[r,dim,*use_i+norb2,*use_j+norb2]];
                                        new_rmatrix[[r,dim,i+norb,j]]=old_model.rmatrix[[r,dim,*use_i+norb2,*use_j]];
                                        new_rmatrix[[r,dim,i,j+norb]]=old_model.rmatrix[[r,dim,*use_i,*use_j+norb2]];
                                    }
                                }
                            }
                        }
                    }else{
                        for r in 0..n_R{
                            for dim in 0..self.dim_r{
                                for (i,use_i) in use_orb_item.iter().enumerate(){
                                    for (j,use_j) in use_orb_item.iter().enumerate(){
                                        new_rmatrix[[r,dim,i,j]]=old_model.rmatrix[[r,dim,*use_i,*use_j]];
                                    }
                                }
                            }
                        }
                    }
                    new_model.rmatrix=new_rmatrix;
                }
                return new_model;
            }
            2=>{
                if dir != None{
                    println!("Wrong!, the dimension of model is 2, but the dir is not None, you should give None!, here we use 0,1 direction");
                }
                let (old_model,use_orb_item,use_atom_item)=match shape{
                    3 =>{
                        let model_1=self.cut_piece(num+1,0);
                        let model_2=model_1.cut_piece(num+1,1);
                        let mut use_atom_item:Vec<usize>=Vec::new();
                        let mut use_orb_item:Vec<usize>=Vec::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if atom_position[[0]]+atom_position[[1]] > (num as f64)/(num as f64+1.0) {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    4 => {
                        let model_1=self.cut_piece(num+1,0);
                        let model_2=model_1.cut_piece(num+1,1);
                        let mut use_atom_item=Vec::<usize>::new();
                        let mut use_orb_item=Vec::<usize>::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if atom_position[[1]]> (num as f64)/(num as f64+1.0) 
                                || atom_position[[1]]> (num as f64)/(num as f64+1.0)  {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    6 => {
                        let model_1=self.cut_piece(2*num,0);
                        let model_2=model_1.cut_piece(2*num,1);
                        let mut use_atom_item=Vec::<usize>::new();
                        let mut use_orb_item=Vec::<usize>::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if (atom_position[[0]] > 0.5 && atom_position[[1]]-atom_position[[0]] + 0.5 < 0.0) || (atom_position[[1]] > 0.5 && atom_position[[0]]-atom_position[[1]] + 0.5 < 0.0) {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    8 => {
                        let model_1=self.cut_piece(2*num,0);
                        let model_2=model_1.cut_piece(2*num,1);
                        let mut use_atom_item=Vec::<usize>::new();
                        let mut use_orb_item=Vec::<usize>::new();//这个是确定要保留哪些轨道
                        let mut a:usize=0;
                        for i in 0..model_2.natom{
                            let atom_position=model_2.atom.row(i).to_owned();
                            if (atom_position[[1]]-atom_position[[0]] + 0.5 < 0.0) || 
                                (atom_position[[0]]-atom_position[[1]] + 0.5 < 0.0) ||
                                    (atom_position[[1]]+atom_position[[0]] < 0.5) || 
                                    (atom_position[[1]]-atom_position[[0]] > 0.5) {
                                a+=model_2.atom_list[i];
                            }else{
                                use_atom_item.push(i);
                                for i in 0..model_2.atom_list[i]{
                                    use_orb_item.push(a);
                                    a+=1;
                                }
                            }
                        }
                        (model_2,use_orb_item,use_atom_item)
                    },
                    _=>{
                        panic!("Wrong, the shape only can be 3,4, 6,8");
                        todo!();
                    }
                };
                let natom=use_atom_item.len();
                let norb=use_orb_item.len();
                let mut new_atom=Array2::<f64>::zeros((natom,self.dim_r));
                let mut new_atom_list=Vec::<usize>::new();
                let mut new_orb=Array2::<f64>::zeros((norb,self.dim_r));
                for (i,use_i) in use_atom_item.iter().enumerate(){
                    new_atom.row_mut(i).assign(&old_model.atom.row(*use_i));
                    new_atom_list.push(old_model.atom_list[*use_i]);
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                }
                let mut new_model=Model::tb_model(self.dim_r,old_model.lat,new_orb,self.spin,Some(new_atom),Some(new_atom_list));
                let n_R=new_model.hamR.len_of(Axis(0));
                let mut new_ham=Array3::<Complex<f64>>::zeros((n_R,new_model.nsta,new_model.nsta));
                let mut new_hamR=Array2::<isize>::zeros((0,self.dim_r));
                let norb=new_model.norb;
                let nsta=new_model.nsta;

                if self.spin{
                    let norb2=old_model.norb;
                    for (i,use_i) in use_orb_item.iter().enumerate(){
                        for (j,use_j) in use_orb_item.iter().enumerate(){
                            new_ham[[0,i,j]]=old_model.ham[[0,*use_i,*use_j]];
                            new_ham[[0,i+norb,j+norb]]=old_model.ham[[0,*use_i+norb2,*use_j+norb2]];
                            new_ham[[0,i+norb,j]]=old_model.ham[[0,*use_i+norb2,*use_j]];
                            new_ham[[0,i,j+norb]]=old_model.ham[[0,*use_i,*use_j+norb2]];
                        }
                    }
                }else{
                    for (i,use_i) in use_orb_item.iter().enumerate(){
                        for (j,use_j) in use_orb_item.iter().enumerate(){
                            new_ham[[0,i,j]]=old_model.ham[[0,*use_i,*use_j]];
                        }
                    }
                }
                new_model.ham=new_ham;
                new_model.hamR=new_hamR;
                if self.rmatrix.len_of(Axis(0))==1{
                    for r in 0..self.dim_r{
                        let mut use_rmatrix=Array2::<Complex<f64>>::zeros((nsta,nsta));
                        for i in 0..norb{
                            use_rmatrix[[i,i]]=Complex::new(new_model.orb[[i,r]],0.0);
                        }
                        if new_model.spin{
                            for i in 0..norb{
                                use_rmatrix[[i+norb,i+norb]]=Complex::new(new_model.orb[[i,r]],0.0);
                            }
                        }
                        new_model.rmatrix.slice_mut(s![0,r,..,..]).assign(&use_rmatrix);
                    }
                }else{
                    let mut new_rmatrix=Array4::<Complex<f64>>::zeros((n_R,self.dim_r,new_model.nsta,new_model.nsta));
                    if old_model.spin{
                        let norb2=old_model.norb;
                        for dim in 0..self.dim_r{
                            for (i,use_i) in use_orb_item.iter().enumerate(){
                                for (j,use_j) in use_orb_item.iter().enumerate(){
                                    new_rmatrix[[0,dim,i,j]]=old_model.rmatrix[[0,dim,*use_i,*use_j]];
                                    new_rmatrix[[0,dim,i+norb,j+norb]]=old_model.rmatrix[[0,dim,*use_i+norb2,*use_j+norb2]];
                                    new_rmatrix[[0,dim,i+norb,j]]=old_model.rmatrix[[0,dim,*use_i+norb2,*use_j]];
                                    new_rmatrix[[0,dim,i,j+norb]]=old_model.rmatrix[[0,dim,*use_i,*use_j+norb2]];
                                }
                            }
                        }
                    }else{
                        for dim in 0..self.dim_r{
                            for (i,use_i) in use_orb_item.iter().enumerate(){
                                for (j,use_j) in use_orb_item.iter().enumerate(){
                                    new_rmatrix[[0,dim,i,j]]=old_model.rmatrix[[0,dim,*use_i,*use_j]];
                                }
                            }
                        }
                    }
                    new_model.rmatrix=new_rmatrix;
                }
                return new_model;
            },
            _=>{
                panic!("Wrong, only dim_r=2,3 can using this function!");
                todo!();
            }
        }

    }

    fn remove_orb(&mut self,orb_list:&Vec<usize>){
        let mut use_orb_list=orb_list.clone();
        use_orb_list.sort_by(|a, b|a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates={
            use_orb_list.windows(2).any(|window| window[0] == window[1])
        };
        if has_duplicates{
            panic!("Wrong, make sure no duplicates in orb_list");
        }
        let mut index: Vec<_> = (0..=self.norb-1)
        .filter(|&num| !use_orb_list.contains(&num))
        .collect();//要保留下来的元素
        let delete_n=orb_list.len();
        self.norb-=delete_n;
        self.orb=self.orb.clone().select(Axis(0),&index);

        //开始删除atom_list
        let mut b=0;
        for (i,a) in self.atom_list.clone().iter().enumerate(){
            b+=*a;
            while b>use_orb_list[0]{
                self.atom_list[i]-=1;
                let _=use_orb_list.remove(0);
                
                if self.atom_list[i]==0{
                    let A=self.atom.clone();
                    self.atom=remove_row(A,i);
                    self.natom-=1;
                }
            }
        }
        self.atom_list.retain(|&x| x!=0);
        //开始计算nsta
        if self.spin{
            self.nsta=self.norb*2;
            let index_add:Vec<_>=index.iter().map(|x| *x+self.norb).collect();
            index.extend(index_add);
        }else{
            self.nsta=self.norb;
        }
        let mut b=0;
        //开始操作哈密顿量
        let new_ham=self.ham.select(Axis(1),&index);
        let new_ham=new_ham.select(Axis(2),&index);
        self.ham=new_ham
    }

    fn remove_atom(&mut self,atom_list:&Vec<usize>){

        //----------判断是否存在重复, 并给出保留的index
        let mut use_atom_list=atom_list.clone();
        use_atom_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates={
            use_atom_list.windows(2).any(|window| window[0] == window[1])
        };
        if has_duplicates{
            panic!("Wrong, make sure no duplicates in orb_list");
        }

        let mut atom_index: Vec<_> = (0..=self.natom-1)
        .filter(|&num| !use_atom_list.contains(&num))
        .collect();//要保留下来的元素
        //--------------------------
        self.natom-=atom_list.len();//让原子数保持一致
        self.atom=self.atom.select(Axis(0),&atom_index);//选出需要的原子
        //接下来选择需要的轨道
        let mut b=0;
        let mut orb_index=Vec::new();
        let mut new_atom_list=Vec::new();
        for (i,a) in self.atom_list.iter().enumerate(){
            if atom_index.contains(&i){
                for j in 0..*a{
                    orb_index.push(b+j);
                }
                new_atom_list.push(*a);
            }
            b+=a;
        }
        self.orb=self.orb.select(Axis(0),&orb_index);
        self.norb=self.orb.len_of(Axis(0));
        self.atom_list=new_atom_list;
        if self.spin{
            self.nsta=self.norb*2;
            let index_add:Vec<_>=orb_index.iter().map(|x| *x+self.norb).collect();
            orb_index.extend(index_add);
        }else{
            self.nsta=self.norb;
        }
        //开始操作哈密顿量
        let new_ham=self.ham.select(Axis(1),&orb_index);
        let new_ham=new_ham.select(Axis(2),&orb_index);
        self.ham=new_ham
    }


    fn unfold(&self,U:&Array2::<f64>,kvec:&Array2::<f64>,E_min:f64,E_max:f64,E_n:usize)->Array2::<f64>{
    //! 能带反折叠算法, 用来计算能带反折叠后的能带.
        let nk=kvec.nrows();
        let E=Array1::<f64>::linspace(E_min,E_max,E_n);
        let mut A0=Array2::<f64>::zeros((E_n,nk));
        A0
    }

    fn make_supercell(&self,U:&Array2::<f64>)->Model{
        //这个函数是用来对模型做变换的, 变换前后模型的基矢 $L'=UL$.
        //!This function is used to transform the model, where the new basis after transformation is given by $L' = UL$.
        if self.dim_r!=U.len_of(Axis(0)){
            panic!("Wrong, the imput U's dimension must equal to self.dim_r")
        }
        let new_lat=U.dot(&self.lat);
        let U_det=U.det().unwrap() as isize;
        if U_det <0{
            panic!("Wrong, the U_det is {}, you should using right hand axis",U_det);
        }else if U_det==0{
            panic!("Wrong, the U_det is {}",U_det);
        }
        let U_inv=U.inv().unwrap();
        //开始判断是否存在小数
        for i in 0..U.len_of(Axis(0)){
            for j in 0..U.len_of(Axis(1)){
                if U[[i,j]].fract()> 1e-8{
                    panic!("Wrong, the U's element must be integer, but your given is {} at [{},{}]",U[[i,j]],i,j);
                }
            }
        }

        //开始构建新的轨道位置和原子位置
        let mut use_orb=self.orb.dot(&U_inv);
        let use_atom=self.atom.dot(&U_inv);
        let mut use_atom_list:Vec<usize>=Vec::new();
        let mut orb_list:Vec<usize>=Vec::new();
        let mut new_orb=Array2::<f64>::zeros((0,self.dim_r));
        let mut new_atom=Array2::<f64>::zeros((0,self.dim_r));
        let mut new_atom_list:Vec<usize>=Vec::new();//新的模型的 atom_list
        let mut a=0;
        for i in 0..self.natom{
            use_atom_list.push(a);
            a+=self.atom_list[i];
        }
        match self.dim_r{
            3=>{
                for i in -U_det..U_det{
                    for j in -U_det..U_det{
                        for k in -U_det..U_det{
                            for n in 0..self.natom{
                                let mut atoms=use_atom.row(n).to_owned()
                                    +(i as f64)*U_inv.row(0).to_owned()
                                    +(j as f64)*U_inv.row(1).to_owned()
                                    +(k as f64)*U_inv.row(2).to_owned(); //原子的位置在新的坐标系下的坐标
                                atoms[[0]]=if atoms[[0]].abs()<1e-8{ 0.0}
                                    else if (atoms[[0]]-1.0).abs()<1e-8 {1.0}  else {atoms[[0]]};
                                atoms[[1]]=if atoms[[1]].abs()<1e-8{ 0.0}
                                    else if (atoms[[1]]-1.0).abs()<1e-8 {1.0} else {atoms[[1]]};
                                atoms[[2]]=if atoms[[2]].abs()<1e-8{ 0.0}
                                    else if (atoms[[2]]-1.0).abs()<1e-8 {1.0} else {atoms[[2]]};
                                if atoms.iter().all(|x| *x>=0.0 && *x < 1.0){ //判断是否在原胞内
                                    new_atom.push_row(atoms.view()); 
                                    new_atom_list.push(self.atom_list[n]);
                                    for n0 in use_atom_list[n]..use_atom_list[n]+self.atom_list[n]{
                                        //开始根据原子位置开始生成轨道
                                        let mut orbs=use_orb.row(n0).to_owned()+(i as f64)*U_inv.row(0).to_owned()+(j as f64)*U_inv.row(1).to_owned()+(k as f64)*U_inv.row(2).to_owned(); //新的轨道的坐标
                                        new_orb.push_row(orbs.view());
                                        orb_list.push(n0);
                                        //orb_list_R.push_row(&arr1(&[i,j,k]));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            2=>{
                for i in -U_det..U_det{
                    for j in -U_det..U_det{
                        for n in 0..self.natom{
                            let mut atoms=use_atom.row(n).to_owned()
                                +(i as f64)*U_inv.row(0).to_owned()
                                +(j as f64)*U_inv.row(1).to_owned(); //原子的位置在新的坐标系下的坐标
                            atoms[[0]]=if atoms[[0]].abs()<1e-8{ 0.0}
                                else if (atoms[[0]]-1.0).abs()<1e-8 {1.0}
                                else {atoms[[0]]};
                            atoms[[1]]=if atoms[[1]].abs()<1e-8{ 0.0}
                                else if (atoms[[1]]-1.0).abs()<1e-8 {1.0}
                                else {atoms[[1]]};
                            if atoms.iter().all(|x| *x>=0.0 && *x < 1.0){ //判断是否在原胞内
                                new_atom.push_row(atoms.view()); 
                                new_atom_list.push(self.atom_list[n]);
                                for n0 in use_atom_list[n]..use_atom_list[n]+self.atom_list[n]{
                                    //开始根据原子位置开始生成轨道
                                    let mut orbs=use_orb.row(n0).to_owned()+(i as f64)*U_inv.row(0).to_owned()+(j as f64)*U_inv.row(1).to_owned(); //新的轨道的坐标
                                    new_orb.push_row(orbs.view());
                                    orb_list.push(n0);
                                    //orb_list_R.push_row(&arr1(&[i,j]));
                                }
                            }
                        }
                    }
                }
            }
            1=>{
                for i in -U_det..U_det{
                    for n in 0..self.natom{
                        let mut atoms=use_atom.row(n).to_owned()+(i as f64)*U_inv.row(0).to_owned(); //原子的位置在新的坐标系下的坐标
                        atoms[[0]]=if atoms[[0]].abs()<1e-8{ 0.0}else if (atoms[[0]]-1.0).abs()<1e-8 {1.0}  else {atoms[[0]]};
                        if atoms.iter().all(|x| *x>=0.0 && *x < 1.0){ //判断是否在原胞内
                            new_atom.push_row(atoms.view()); 
                            new_atom_list.push(self.atom_list[n]);
                            for n0 in use_atom_list[n]..use_atom_list[n]+self.atom_list[n]{
                                //开始根据原子位置开始生成轨道
                                let mut orbs=use_orb.row(n0).to_owned()+(i as f64)*U_inv.row(0).to_owned(); //新的轨道的坐标
                                new_orb.push_row(orbs.view());
                                orb_list.push(n0);
                                //orb_list_R.push_row(&arr1(&[i]));
                            }
                        }
                    }
                }
            }
            _=>todo!()
        }
        //轨道位置和原子位置构建完成, 接下来我们开始构建哈密顿量
        let norb=new_orb.len_of(Axis(0));
        let mut nsta=norb;
        if self.spin{
            nsta*=2;
        }
        let natom=new_atom.len_of(Axis(0));
        let n_R=self.hamR.len_of(Axis(0));
        let mut new_hamR=Array2::<isize>::zeros((1,self.dim_r));//超胞准备用的hamR
        let mut use_hamR=Array2::<isize>::zeros((1,self.dim_r));//超胞的hamR的可能, 如果这个hamR没有对应的hopping就会被删除
        let mut new_ham=Array3::<Complex<f64>>::zeros((1,nsta,nsta));//超薄准备用的ham
        let mut new_rmatrix=Array4::<Complex<f64>>::zeros((1,self.dim_r,nsta,nsta));//超薄准备用的rmatrix
        let mut max_R=Array1::<isize>::zeros(self.dim_r);
        /*
        for (j,cow) in self.hamR.axis_iter(Axis(1)).enumerate(){
            let abs_cow=cow.map(|x| x.abs());
            max_R[[j]]=*abs_cow.to_vec().iter().max().unwrap();
        }
        */
        let max_R:isize=U_det.abs()*(self.dim_r as isize);
        let max_R=Array1::<isize>::ones(self.dim_r)*max_R;
        //用来产生可能的hamR
        match self.dim_r{
            1=>{
                for i in 1..max_R[[0]]+1{
                    use_hamR.push_row(array![i].view());
                }
            }
            2=>{
                for j in 1..max_R[[1]]+1{
                    for i in -max_R[[0]]..max_R[[0]]+1{
                        use_hamR.push_row(array![i,j].view());
                    }
                }
                for i in 1..max_R[[0]]+1{
                    use_hamR.push_row(array![i,0].view());
                }
            }
            3=>{
                for k in 1..max_R[[2]]+1{
                    for i in -max_R[[0]]..max_R[[0]]+1{
                        for j in -max_R[[1]]..max_R[[1]]+1{
                            use_hamR.push_row(array![i,j,k].view());
                        }
                    }
                }
                for j in 1..max_R[[1]]+1{
                    for i in -max_R[[0]]..max_R[[0]]+1{
                        use_hamR.push_row(array![i,j,0].view());
                    }
                }
                for i in 1..max_R[[0]]+1{
                    use_hamR.push_row(array![i,0,0].view());
                }
            }
             _ => todo!()
        }
        let use_n_R=use_hamR.len_of(Axis(0));
        let mut gen_rmatrix:bool=false;
        if self.rmatrix.len_of(Axis(0))==1{
            for i in 0..self.dim_r{
                for s in 0..norb{
                    new_rmatrix[[0,i,s,s]]=Complex::new(new_orb[[s,i]],0.0);
                }
            }
            if self.spin{
                for i in 0..self.dim_r{
                    for s in 0..norb{
                        new_rmatrix[[0,i,s+norb,s+norb]]=Complex::new(new_orb[[s,i]],0.0);
                    }
                }
            }
        }else{
            gen_rmatrix=true;
        }
        if self.spin && gen_rmatrix{
            for R in 0..use_n_R{
                let mut add_R:bool=false;
                let mut useham=Array2::<Complex<f64>>::zeros((nsta,nsta));
                let mut use_rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,nsta,nsta));
                let mut use_R=use_hamR.row(R); //超胞的R
                for i in 0..norb{
                    for j in 0..norb{
                        let int_i:usize=i; //超胞中的 <i|
                        let use_i:usize=orb_list[i]; //对应到原胞中的 <i|
                        let int_j:usize=j; //超胞中的 |j>
                        let use_j:usize=orb_list[j]; //超胞中的 |j>
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0:Array1::<f64>=new_orb.row(j).to_owned()-new_orb.row(i).to_owned()+use_R.map(|x| *x as f64); //超胞的 R 在原始原胞的 R
                        let R0:Array1::<isize>=(R0.dot(U)-self.orb.row(use_j)+self.orb.row(use_i)).map(|x| if x.fract().abs()<1e-8 || x.fract().abs()>1.0-1e-8{x.round() as isize} else {x.floor() as isize}); 
                        let R0_inv=-R0.clone();
                        let R0_exit=find_R(&self.hamR,&R0);
                        let R0_inv_exit=find_R(&self.hamR,&R0_inv);
                        if R0_exit{
                            let index=index_R(&self.hamR,&R0);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_i,use_j]];
                            useham[[int_i+norb,int_j]]=self.ham[[index,use_i+self.norb,use_j]];
                            useham[[int_i,int_j+norb]]=self.ham[[index,use_i,use_j+self.norb]];
                            useham[[int_i+norb,int_j+norb]]=self.ham[[index,use_i+self.norb,use_j+self.norb]];
                            for r in 0..self.dim_r{
                                use_rmatrix[[r,int_i,int_j]]=self.rmatrix[[index,r,use_i,use_j]];
                                use_rmatrix[[r,int_i+norb,int_j]]=self.rmatrix[[index,r,use_i+self.norb,use_j]];
                                use_rmatrix[[r,int_i,int_j+norb]]=self.rmatrix[[index,r,use_i,use_j+self.norb]];
                                use_rmatrix[[r,int_i+norb,int_j+norb]]=self.rmatrix[[index,r,use_i+self.norb,use_j+self.norb]];
                            }
                        }else if R0_inv_exit{
                            let index=index_R(&self.hamR,&R0_inv);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_j,use_i]].conj();
                            useham[[int_i+norb,int_j]]=self.ham[[index,use_j,use_i+self.norb]].conj();
                            useham[[int_i,int_j+norb]]=self.ham[[index,use_j+self.norb,use_i]].conj();
                            useham[[int_i+norb,int_j+norb]]=self.ham[[index,use_j+self.norb,use_i+self.norb]].conj();
                            for r in 0..self.dim_r{
                                use_rmatrix[[r,int_i,int_j]]=self.rmatrix[[index,r,use_j,use_i]].conj();
                                use_rmatrix[[r,int_i+norb,int_j]]=self.rmatrix[[index,r,use_j,use_i+self.norb]].conj();
                                use_rmatrix[[r,int_i,int_j+norb]]=self.rmatrix[[index,r,use_j+self.norb,use_i]].conj();
                                use_rmatrix[[r,int_i+norb,int_j+norb]]=self.rmatrix[[index,r,use_j+self.norb,use_i+self.norb]].conj();
                            }
                        }else{
                            continue
                        }
                    }
                }
                if add_R && R != 0{
                    new_ham.push(Axis(0),useham.view());
                    new_hamR.push_row(use_R.view());
                    new_rmatrix.push(Axis(0),use_rmatrix.view());
                }else if R==0{
                    new_ham.slice_mut(s![0,..,..]).assign(&useham);
                    new_rmatrix.slice_mut(s![0,..,..,..]).assign(&use_rmatrix);
                }
            }
        }else if gen_rmatrix && !self.spin{
            for R in 0..use_n_R{
                let mut add_R:bool=false;
                let mut useham=Array2::<Complex<f64>>::zeros((nsta,nsta));
                let mut use_rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,nsta,nsta));
                let mut use_R=use_hamR.row(R); //超胞的R
                for i in 0..norb{
                    for j in 0..norb{
                        let int_i:usize=i; //超胞中的 <i|
                        let use_i:usize=orb_list[i]; //对应到原胞中的 <i|
                        let int_j:usize=j; //超胞中的 |j>
                        let use_j:usize=orb_list[j]; //超胞中的 |j>
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0:Array1::<f64>=new_orb.row(j).to_owned()-new_orb.row(i).to_owned()+use_R.map(|x| *x as f64); //超胞的 R 在原始原胞的 R
                        let R0:Array1::<isize>=(R0.dot(U)-self.orb.row(use_j)+self.orb.row(use_i)).map(|x| if x.fract().abs()<1e-8 || x.fract().abs()>1.0-1e-8{x.round() as isize} else {x.floor() as isize}); 
                        let R0_inv=-R0.clone();
                        let R0_exit=find_R(&self.hamR,&R0);
                        let R0_inv_exit=find_R(&self.hamR,&R0_inv);
                        if R0_exit{
                            let index=index_R(&self.hamR,&R0);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_i,use_j]];
                            for r in 0..self.dim_r{
                                use_rmatrix[[r,int_i,int_j]]=self.rmatrix[[index,r,use_i,use_j]]
                            }
                        }else if R0_inv_exit{
                            let index=index_R(&self.hamR,&R0_inv);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_j,use_i]].conj();
                            for r in 0..self.dim_r{
                                use_rmatrix[[r,int_i,int_j]]=self.rmatrix[[index,r,use_j,use_i]].conj()
                            }
                        }else{
                            continue
                        }
                    }
                }
                if add_R && R != 0{
                    new_ham.push(Axis(0),useham.view());
                    new_rmatrix.push(Axis(0),use_rmatrix.view());
                    new_hamR.push_row(use_R);
                }else if R==0{
                    new_ham.slice_mut(s![0,..,..]).assign(&useham);
                    new_rmatrix.slice_mut(s![0,..,..,..]).assign(&use_rmatrix);
                }
            }
        }else if self.spin{
            for R in 0..use_n_R{
                let mut add_R:bool=false;
                let mut useham=Array2::<Complex<f64>>::zeros((nsta,nsta));
                let mut use_R=use_hamR.row(R); //超胞的R
                for i in 0..norb{
                    for j in 0..norb{
                        let int_i:usize=i; //超胞中的 <i|
                        let use_i:usize=orb_list[i]; //对应到原胞中的 <i|
                        let int_j:usize=j; //超胞中的 |j>
                        let use_j:usize=orb_list[j]; //超胞中的 |j>
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0:Array1::<f64>=new_orb.row(j).to_owned()-new_orb.row(i).to_owned()+use_R.map(|x| *x as f64); //超胞的 R 在原始原胞的 R
                        let R0:Array1::<isize>=(R0.dot(U)-self.orb.row(use_j)+self.orb.row(use_i)).map(|x| if x.fract().abs()<1e-8 || x.fract().abs()>1.0-1e-8{x.round() as isize} else {x.floor() as isize}); 
                        let R0_inv=-R0.clone();
                        let R0_exit=find_R(&self.hamR,&R0);
                        let R0_inv_exit=find_R(&self.hamR,&R0_inv);
                        if R0_exit{
                            let index=index_R(&self.hamR,&R0);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_i,use_j]];
                            useham[[int_i+norb,int_j]]=self.ham[[index,use_i+self.norb,use_j]];
                            useham[[int_i,int_j+norb]]=self.ham[[index,use_i,use_j+self.norb]];
                            useham[[int_i+norb,int_j+norb]]=self.ham[[index,use_i+self.norb,use_j+self.norb]];
                        }else if R0_inv_exit{
                            let index=index_R(&self.hamR,&R0_inv);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_j,use_i]].conj();
                            useham[[int_i+norb,int_j]]=self.ham[[index,use_j,use_i+self.norb]].conj();
                            useham[[int_i,int_j+norb]]=self.ham[[index,use_j+self.norb,use_i]].conj();
                            useham[[int_i+norb,int_j+norb]]=self.ham[[index,use_j+self.norb,use_i+self.norb]].conj();
                        }else{
                            continue
                        }
                    }
                } 
                if add_R && R != 0{
                    new_ham.push(Axis(0),useham.view());
                    new_hamR.push_row(use_R.view());
                }else if R==0{
                    new_ham.slice_mut(s![0,..,..]).assign(&useham);
                }
            }
        }else{
            for R in 0..use_n_R{
                let mut add_R:bool=false;
                let mut useham=Array2::<Complex<f64>>::zeros((nsta,nsta));
                let mut use_R=use_hamR.row(R); //超胞的R
                for i in 0..norb{
                    for j in 0..norb{
                        let int_i:usize=i; //超胞中的 <i|
                        let use_i:usize=orb_list[i]; //对应到原胞中的 <i|
                        let int_j:usize=j; //超胞中的 |j>
                        let use_j:usize=orb_list[j]; //超胞中的 |j>
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0:Array1::<f64>=new_orb.row(j).to_owned()-new_orb.row(i).to_owned()+use_R.map(|x| *x as f64); //超胞的 R 在原始原胞的 R
                        let R0:Array1::<isize>=(R0.dot(U)-self.orb.row(use_j)+self.orb.row(use_i)).map(|x| if x.fract().abs()<1e-8 || x.fract().abs()>1.0-1e-8{x.round() as isize} else {x.floor() as isize}); 
                        let R0_inv=-R0.clone();
                        let R0_exit=find_R(&self.hamR,&R0);
                        let R0_inv_exit=find_R(&self.hamR,&R0_inv);
                        if R0_exit{
                            let index=index_R(&self.hamR,&R0);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_i,use_j]];
                        }else if R0_inv_exit{
                            let index=index_R(&self.hamR,&R0_inv);
                            add_R=true;
                            useham[[int_i,int_j]]=self.ham[[index,use_j,use_i]].conj();
                        }else{
                            continue
                        }
                    }
                }
                if add_R && R != 0{
                    new_ham.push(Axis(0),useham.view());
                    new_hamR.push_row(use_R);
                }else if R==0{
                    new_ham.slice_mut(s![0,..,..]).assign(&useham);
                }
            }
        }
        let mut model=Model{
            dim_r:self.dim_r,
            norb:norb,
            nsta:nsta,
            natom:natom,
            spin:self.spin,
            lat:new_lat,
            orb:new_orb,
            atom:new_atom,
            atom_list:new_atom_list,
            ham:new_ham,
            hamR:new_hamR,
            rmatrix:new_rmatrix,
        };
        model
    }
    fn shift_to_zero(&mut self){
        //! 这个是把超出单元格的原子移动回原本的位置的代码
        //! 这个可以用来重新构造原胞原子的位置.
        //! 我们看一下 SSH model 的结果

        //首先, 先确定哪些原子需要移动
        let mut move_R=Array2::<isize>::zeros((0,self.dim_r));//对于给
        let mut new_atom=Array2::<f64>::zeros((0,self.dim_r));
        let mut new_orb=Array2::<f64>::zeros((0,self.dim_r));
        let mut new_ham=Array3::<Complex<f64>>::zeros((1,self.nsta,self.nsta));
        let mut new_rmatrix=Array4::<Complex<f64>>::zeros((1,self.dim_r,self.nsta,self.nsta));
        let mut new_hamR=Array2::<isize>::zeros((1,self.dim_r));
        let mut a:usize=0;
        for (i,atom) in self.atom.axis_iter(Axis(0)).enumerate(){
            let mut atom=atom.to_owned();
            atom.map(|x| if (x.fract()-1.0).abs() < 1e-8 || x.fract().abs()<1e-8{x.round()} else {*x}); 
            // 这是先把原子中靠近整数晶格的原子移动到整数晶格上, 方便判断原子属于哪个原胞
            let R=atom.map(|x| x.trunc() as isize);
            let use_atom=atom.map(|x| x.fract());
            let R0=use_atom.clone().map(|x| if *x<0.0 {-1 as isize}else{0 as isize});
            let R=R+&R0;
            let use_atom=use_atom.map(|x| if *x<0.0 {*x+1.0}else{*x});
            new_atom.push_row(use_atom.view());
            for j in 0..self.atom_list[i] {
                //self.orb.row_mut(i).add_assign(&(R.map(|x| *x as f64)));
                new_orb.push_row((&self.orb.row_mut(a)-R.map(|x| *x as f64)).view());
                move_R.push_row(R.view());
                a+=1;
            }
        }
        let nr=self.rmatrix.len_of(Axis(0));
        if nr==1{
            for i in 0..self.dim_r{
                for j in 0..self.norb{
                    self.rmatrix[[0,i,j,j]]=Complex::new(self.orb[[j,i]],0.0);
                    if self.spin{
                        self.rmatrix[[0,i,j+self.norb,j+self.norb]]=Complex::new(self.orb[[j,i]],0.0);
                    }
                }
            }
        }
        for i in 0..self.norb{
            for j in 0..self.norb{
                for (r,hopR) in self.hamR.axis_iter(Axis(0)).enumerate(){
                    if r==0 && i>j{
                        continue
                    }
                    let useR=hopR.to_owned()-move_R.row(i)+move_R.row(j);
                    let R_exist=find_R(&new_hamR,&useR);
                    let negative_R=-useR.clone();
                    let negative_R_exist=find_R(&new_hamR,&negative_R);
                    if R_exist{
                        let index=index_R(&new_hamR,&useR);
                        if self.spin{
                            new_ham[[index,i,j]]=self.ham[[r,i,j]];
                            new_ham[[index,i+self.norb,j]]=self.ham[[r,i+self.norb,j]];
                            new_ham[[index,i,j+self.norb]]=self.ham[[r,i,j+self.norb]];
                            new_ham[[index,i+self.norb,j+self.norb]]=self.ham[[r,i+self.norb,j+self.norb]];
                        }else{
                            new_ham[[index,i,j]]=self.ham[[r,i,j]];
                        }
                        if negative_R_exist{
                            if self.spin{
                                new_ham[[index,j,i]]=self.ham[[r,i,j]].conj();
                                new_ham[[index,j+self.norb,i]]=self.ham[[r,i+self.norb,j]].conj();
                                new_ham[[index,j,i+self.norb]]=self.ham[[r,i,j+self.norb]].conj();
                                new_ham[[index,j+self.norb,i+self.norb]]=self.ham[[r,i+self.norb,j+self.norb]].conj();
                            }else{
                                new_ham[[index,j,i]]=self.ham[[r,i,j]].conj();
                            }
                        }
                        if nr !=1{
                            if self.spin{
                                for s in 0..self.dim_r{
                                    new_rmatrix[[index,s,i,j]]=self.rmatrix[[r,s,i,j]];
                                    new_rmatrix[[index,s,i+self.norb,j]]=self.rmatrix[[r,s,i+self.norb,j]];
                                    new_rmatrix[[index,s,i,j+self.norb]]=self.rmatrix[[r,s,i,j+self.norb]];
                                    new_rmatrix[[index,s,i+self.norb,j+self.norb]]=self.rmatrix[[r,s,i+self.norb,j+self.norb]];
                                }
                            }else{
                                for s in 0..self.dim_r{
                                    new_rmatrix[[index,s,i,j]]=self.rmatrix[[r,s,i,j]];
                                }
                            }

                            if negative_R_exist{
                                if self.spin{
                                    for s in 0..self.dim_r{
                                        new_rmatrix[[index,s,j,i]]=self.rmatrix[[r,s,i,j]];
                                        new_rmatrix[[index,s,j+self.norb,i]]=self.rmatrix[[r,s,i+self.norb,j]];
                                        new_rmatrix[[index,s,j,i+self.norb]]=self.rmatrix[[r,s,i,j+self.norb]];
                                        new_rmatrix[[index,s,j+self.norb,i+self.norb]]=self.rmatrix[[r,s,i+self.norb,j+self.norb]];
                                    }
                                }else{
                                    for s in 0..self.dim_r{
                                        new_rmatrix[[index,s,j,i]]=self.rmatrix[[r,s,i,j]].conj();
                                    }
                                }
                            }
                        }
                    }else if negative_R_exist{
                        let index=index_R(&new_hamR,&negative_R);
                        if self.spin{
                            new_ham[[index,i,j]]=self.ham[[r,i,j]];
                            new_ham[[index,i+self.norb,j]]=self.ham[[r,i+self.norb,j]];
                            new_ham[[index,i,j+self.norb]]=self.ham[[r,i,j+self.norb]];
                            new_ham[[index,i+self.norb,j+self.norb]]=self.ham[[r,i+self.norb,j+self.norb]];
                        }else{
                            new_ham[[index,i,j]]=self.ham[[r,i,j]];
                        }
                        if nr !=1{
                            if self.spin{
                                for s in 0..self.dim_r{
                                    new_rmatrix[[index,s,i,j]]=self.rmatrix[[r,s,i,j]];
                                    new_rmatrix[[index,s,i+self.norb,j]]=self.rmatrix[[r,s,i+self.norb,j]];
                                    new_rmatrix[[index,s,i,j+self.norb]]=self.rmatrix[[r,s,i,j+self.norb]];
                                    new_rmatrix[[index,s,i+self.norb,j+self.norb]]=self.rmatrix[[r,s,i+self.norb,j+self.norb]];
                                }
                            }else{
                                for s in 0..self.dim_r{
                                    new_rmatrix[[index,s,i,j]]=self.rmatrix[[r,s,i,j]];
                                }
                            }
                        }
                    }else{
                        new_hamR.push_row(useR.view());
                        let mut use_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
                        if self.spin{
                            use_ham[[i,j]]=self.ham[[r,i,j]];
                            use_ham[[i+self.norb,j]]=self.ham[[r,i+self.norb,j]];
                            use_ham[[i,j+self.norb]]=self.ham[[r,i,j+self.norb]];
                            use_ham[[i+self.norb,j+self.norb]]=self.ham[[r,i+self.norb,j+self.norb]];
                        }else{
                            use_ham[[i,j]]=self.ham[[r,i,j]];
                        }
                        new_ham.push(Axis(0),use_ham.view());
                        if nr !=1{
                            let mut use_rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));
                            if self.spin{
                                for s in 0..self.dim_r{
                                    use_rmatrix[[s,i,j]]=self.rmatrix[[r,s,i,j]];
                                    use_rmatrix[[s,i+self.norb,j]]=self.rmatrix[[r,s,i+self.norb,j]];
                                    use_rmatrix[[s,i,j+self.norb]]=self.rmatrix[[r,s,i,j+self.norb]];
                                    use_rmatrix[[s,i+self.norb,j+self.norb]]=self.rmatrix[[r,s,i+self.norb,j+self.norb]];
                                }
                            }else{
                                for s in 0..self.dim_r{
                                    use_rmatrix[[s,i,j]]=self.rmatrix[[r,s,i,j]];
                                }
                            }
                            new_rmatrix.push(Axis(0),use_rmatrix.view());
                        }
                    }
                }
            }
        }
        self.atom=new_atom;
        self.orb=new_orb;
        self.hamR=new_hamR;
        self.ham=new_ham;
        self.rmatrix=new_rmatrix;
    }
    #[allow(non_snake_case)]
    fn dos(&self,k_mesh:&Array1::<usize>,E_min:f64,E_max:f64,E_n:usize,sigma:f64)->(Array1::<f64>,Array1::<f64>){
        //! 我这里用的算法是高斯算法, 其算法过程如下
        //! 首先, 根据 k_mesh 算出所有的能量 $\ve_n$, 然后, 按照定义
        //! $$\rho(\ve)=\sum_N\int\dd\bm k \delta(\ve_n-\ve)$$
        //! 我们将 $\delta(\ve_n-\ve)$ 做了替换, 换成了 $\f{1}{\sqrt{2\pi}\sigma}e^{-\f{(\ve_n-\ve)^2}{2\sigma^2}}$
        //! 然后, 计算方法是先算出所有的能量, 再将能量乘以高斯分布, 就能得到态密度.
        //! 态密度的光滑程度和k点密度以及高斯分布的展宽有关
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk=kvec.len_of(Axis(0));
        let band=self.solve_band_all_parallel(&kvec);
        let E0=Array1::linspace(E_min,E_max,E_n);
        let E=E0.clone();
        let mut dos=Array1::<f64>::zeros(E_n);
        let dim:usize=k_mesh.len();
        let centre=band.into_raw_vec().into_par_iter();
        let sigma0=1.0/sigma;
        let pi0=1.0/(2.0*PI).sqrt();
        let mut dos=Array1::<f64>::zeros(E_n);
        dos=centre.fold(|| dos.clone(),|acc,x|{
                let A=(&E0-x)*sigma0;
                let f=(-&A*&A/2.0).mapv(|x| x.exp())*sigma0*pi0;
                acc+&f
            }).reduce(|| dos.clone(), |acc, x| acc + x);
        /*
        for i in centre.iter(){
            //dos.map(|x| x+pi0*sigma0*(-((x-i)*sigma0).powi(2)/2.0).exp());
            dos=dos+E0.clone().map(|x| pi0*sigma0*(-((x-i)*sigma0).powi(2)/2.0).exp());
        }
        */
        dos=dos/(nk as f64);
        (E,dos)
    }
    ///这个函数是用来快速画能带图的, 用python画图, 因为Rust画图不太方便.
    #[allow(non_snake_case)]
    fn show_band(&self,path:&Array2::<f64>,label:&Vec<&str>,nk:usize,name:&str)-> std::io::Result<()>{
        use std::fs::create_dir_all;
        use std::path::Path;
        use gnuplot::{Figure, Caption, Color,LineStyle,Solid};
        use gnuplot::{AxesCommon};
        use gnuplot::AutoOption::*;
        use gnuplot::Tick::*;
        if path.len_of(Axis(0))!=label.len(){
            panic!("Error, the path's length {} and label's length {} must be equal!",path.len_of(Axis(0)),label.len())
        }
        let (k_vec,k_dist,k_node)=self.k_path(&path,nk);
        let eval=self.solve_band_all_parallel(&k_vec);
        create_dir_all(name).expect("can't creat the file");
        let mut name0=String::new();
        name0.push_str("./");
        name0.push_str(&name);
        let name=name0;
        let mut band_name=name.clone();
        band_name.push_str("/BAND.dat");
        let band_name=Path::new(&band_name);
        let mut file=File::create(band_name).expect("Unable to BAND.dat");
        for i in 0..nk{
            let mut s = String::new();
            let aa= format!("{:.6}", k_dist[[i]]);
            s.push_str(&aa);
            for j in 0..self.nsta{
                if eval[[i,j]]>=0.0 {
                    s.push_str("     ");
                }else{
                    s.push_str("    ");
                }
                let aa= format!("{:.6}", eval[[i,j]]);
                s.push_str(&aa);
            }
            writeln!(file,"{}",s)?;
        }
        let mut k_name=name.clone();
        k_name.push_str("/KLABELS");
        let k_name=Path::new(&k_name);
        let mut file=File::create(k_name).expect("Unable to create KLBAELS");//写下高对称点的位置
        for i in 0..path.len_of(Axis(0)){
            let mut s=String::new();
            let aa= format!("{:.6}", k_node[[i]]);
            s.push_str(&aa);
            s.push_str("      ");
            s.push_str(&label[i]);
            writeln!(file,"{}",s)?;
        }
        let mut py_name=name.clone();
        py_name.push_str("/print.py");
        let py_name=Path::new(&py_name);
        let mut file=File::create(py_name).expect("Unable to create print.py");
        writeln!(file,"import numpy as np\nimport matplotlib.pyplot as plt\ndata=np.loadtxt('BAND.dat')\nk_nodes=[]\nlabel=[]\nf=open('KLABELS')\nfor i in f.readlines():\n    k_nodes.append(float(i.split()[0]))\n    label.append(i.split()[1])\nfig,ax=plt.subplots()\nax.plot(data[:,0],data[:,1:],c='b')\nfor x in k_nodes:\n    ax.axvline(x,c='k')\nax.set_xticks(k_nodes)\nax.set_xticklabels(label)\nax.set_xlim([0,k_nodes[-1]])\nfig.savefig('band.pdf')");
        //开始绘制pdf图片
        let mut fg = Figure::new();
        let x:Vec<f64>=k_dist.to_vec();
        let axes=fg.axes2d();
        for i in 0..self.nsta{
            let y:Vec<f64>=eval.slice(s![..,i]).to_owned().to_vec();
            axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
        }
        let axes=axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len()-1]]));
        let label=label.clone();
        let mut show_ticks=Vec::new();
        for i in 0..k_node.len(){
            let A=k_node[[i]];
            let B=label[i];
            show_ticks.push(Major(A,Fix(B)));
        }
        axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
        
        let k_node=k_node.to_vec();
        let mut pdf_name=name.clone();
        pdf_name.push_str("/plot.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        Ok(())
    }
    #[allow(non_snake_case)]
    fn from_hr(path:&str,file_name:&str,zero_energy:f64)->Model{
        use std::io::BufReader;
        use std::io::BufRead;
        use std::fs::File;
        use std::path::Path;
        let mut file_path = path.to_string();
        file_path.push_str(file_name);
        let mut hr_path=file_path.clone();
        hr_path.push_str("_hr.dat");
        let path=Path::new(&hr_path);
        let hr=File::open(path).expect("Unable open the file, please check if have hr file");
        let reader = BufReader::new(hr);
        let mut reads:Vec<String>=Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }
        let nsta=reads[1].trim().parse::<usize>().unwrap();
        let n_R=reads[2].trim().parse::<usize>().unwrap();
        let mut weights:Vec<usize>=Vec::new();
        let mut n_line:usize=0;
        for i in 3..reads.len(){
            //if string.clone().count() !=15{
            if reads[i].contains("."){
                n_line=i;
                break
            }
            let string=reads[i].trim().split_whitespace();
            let string:Vec<_>=string.map(|x| x.parse::<usize>().unwrap()).collect();
            weights.extend(string.clone());
        }
        let mut hamR=Array2::<isize>::zeros((1,3));
        let mut ham=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
        for i in 0..n_R{
            let mut string=reads[i*nsta*nsta+n_line].trim().split_whitespace();
            let a=string.next().unwrap().parse::<isize>().unwrap();
            let b=string.next().unwrap().parse::<isize>().unwrap();
            let c=string.next().unwrap().parse::<isize>().unwrap();
            if (c>0) || (c==0 && b>0) ||(c==0 && b==0 && a>=0){
                if a==0 && b==0 && c==0{
                    for ind_i in 0..nsta{
                        for ind_j in 0..nsta{
                            let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+n_line].trim().split_whitespace();
                            let re=string.nth(5).unwrap().parse::<f64>().unwrap();
                            let im=string.next().unwrap().parse::<f64>().unwrap();
                            ham[[0,ind_j,ind_i]]=Complex::new(re,im)/(weights[i] as f64);
                        }
                    }
                }else{
                    let mut matrix=Array3::<Complex<f64>>::zeros((1,nsta,nsta));
                    for ind_i in 0..nsta{
                        for ind_j in 0..nsta{
                            let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+n_line].trim().split_whitespace();
                            let re=string.nth(5).unwrap().parse::<f64>().unwrap();
                            let im=string.next().unwrap().parse::<f64>().unwrap();
                            matrix[[0,ind_j,ind_i]]=Complex::new(re,im)/(weights[i] as f64); //wannier90 里面是按照纵向排列的矩阵
                        }
                    }
                    ham.append(Axis(0),matrix.view()).unwrap();
                    hamR.append(Axis(0),arr2(&[[a,b,c]]).view()).unwrap();
                }
            }
        }
        for i in 0..nsta{
            ham[[0,i,i]]-=Complex::new(zero_energy,0.0);
        }

        //开始读取 .win 文件
        let mut reads:Vec<String>=Vec::new();
        let mut win_path=file_path.clone();
        win_path.push_str(".win"); //文件的位置
        let path=Path::new(&win_path); //转化为路径格式
        let hr=File::open(path).expect("Unable open the file, please check if have hr file");
        let reader = BufReader::new(hr);
        let mut reads:Vec<String>=Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }
        let mut read_iter=reads.iter();
        let mut lat=Array2::<f64>::zeros((3,3)); //晶格轨道坐标初始化
        let mut spin:bool=false; //体系自旋初始化
        let mut natom:usize=0; //原子位置初始化
        let mut atom=Array2::<f64>::zeros((0,3)); //原子位置坐标初始化
        let mut proj_name:Vec<&str>=Vec::new();
        let mut proj_list:Vec<usize>=Vec::new();
        let mut atom_list:Vec<usize>=Vec::new();
        let mut atom_name:Vec<&str>=Vec::new();
        let mut atom_pos=Array2::<f64>::zeros((0,3));
        loop{
            let a=read_iter.next();
            if a==None{
                break;
            }else{
                let a=a.unwrap();
                if a.contains("begin unit_cell_cart") {
                    let mut lat1=read_iter.next().unwrap().trim().split_whitespace(); //将数字放到
                    let mut lat2=read_iter.next().unwrap().trim().split_whitespace();
                    let mut lat3=read_iter.next().unwrap().trim().split_whitespace();
                    for i in 0..3{
                        lat[[0,i]]=lat1.next().unwrap().parse::<f64>().unwrap();
                        lat[[1,i]]=lat2.next().unwrap().parse::<f64>().unwrap();
                        lat[[2,i]]=lat3.next().unwrap().parse::<f64>().unwrap();
                    }
                } else if a.contains("spinors") && (a.contains("T") || a.contains("t")){
                    spin=true;
                }else if a.contains("begin projections"){
                    loop{
                        let string=read_iter.next().unwrap();
                        if string.contains("end projections"){
                            break
                        }else{ 
                            let prj:Vec<&str>=string.split(|c| c==',' || c==';' || c==':').collect();
                            let mut atom_orb_number:usize=0;
                            for item in prj[1..].iter(){
                                let aa:usize=match (*item).trim(){
                                    "s"=>1,
                                    "p"=>3,
                                    "d"=>5,
                                    "f"=>7,
                                    "sp3"=>4,
                                    "sp2"=>3,
                                    "sp"=>2,
                                    "sp3d2"=>6,
                                    "px"=>1,
                                    "py"=>1,
                                    "pz"=>1,
                                    "dxy"=>1,
                                    "dxz"=>1,
                                    "dxz"=>1,
                                    "dz2"=>1,
                                    "dx2-y2"=>1,
                                    &_=>panic!("Wrong, no matching"),
                                };
                                atom_orb_number+=aa;
                            }
                            proj_list.push(atom_orb_number);
                            proj_name.push(prj[0])
                        }
                    }
                }else if a.contains("begin atoms_cart"){
                    loop{
                        let string=read_iter.next().unwrap();
                        if string.contains("end atoms_cart"){
                            break
                        }else{       
                            let prj:Vec<&str>=string.split_whitespace().collect();
                            atom_name.push(prj[0]);
                            let a1=prj[1].parse::<f64>().unwrap();
                            let a2=prj[2].parse::<f64>().unwrap();
                            let a3=prj[3].parse::<f64>().unwrap();
                            let a=array![a1,a2,a3];
                            atom_pos.push_row(a.view());

                        }
                    }
                }
            }
        }
        for (i,name) in atom_name.iter().enumerate(){
            for (j,j_name) in proj_name.iter().enumerate(){
                if j_name==name{
                    atom_list.push(proj_list[j]);
                    atom.push_row(atom_pos.row(i));
                    natom+=1;
                }
            }
        }
        //开始读取 seedname_centres.xyz 文件
        let mut reads:Vec<String>=Vec::new();
        let mut xyz_path=file_path.clone();
        xyz_path.push_str("_centres.xyz");
        let path=Path::new(&xyz_path);
        let hr=File::open(path).expect("Unable open the file, please check if have hr file");
        let reader = BufReader::new(hr);
        let mut reads:Vec<String>=Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            reads.push(line.clone());
        }
        //let nsta=reads[0].trim().parse::<usize>().unwrap()-natom;
        let norb=if spin{nsta/2}else{nsta};
        let mut orb=Array2::<f64>::zeros((norb,3));
        for i in 0..norb{
            let a:Vec<&str>=reads[i+2].trim().split_whitespace().collect();
            orb[[i,0]]=a[1].parse::<f64>().unwrap();
            orb[[i,1]]=a[2].parse::<f64>().unwrap();
            orb[[i,2]]=a[3].parse::<f64>().unwrap();
        }
        /*
        for i in 0..natom{
            let a:Vec<&str>=reads[i+2+nsta].trim().split_whitespace().collect();
            atom[[i,0]]=a[1].parse::<f64>().unwrap();
            atom[[i,1]]=a[2].parse::<f64>().unwrap();
            atom[[i,2]]=a[3].parse::<f64>().unwrap();
        }
        */
        orb=orb.dot(&lat.inv().unwrap());
        atom=atom.dot(&lat.inv().unwrap());
        //开始尝试读取 _r.dat 文件
        let mut reads:Vec<String>=Vec::new();
        let mut r_path=file_path.clone();
        r_path.push_str("_r.dat");
        let mut rmatrix=Array4::<Complex<f64>>::zeros((1,3,nsta,nsta));
        let path=Path::new(&r_path);
        let hr=File::open(path);
         
        if hr.is_ok(){
            let hr=hr.unwrap();
            let reader = BufReader::new(hr);
            let mut reads:Vec<String>=Vec::new();
            for line in reader.lines() {
                let line = line.unwrap();
                reads.push(line.clone());
            }
            let n_R=reads[2].trim().parse::<usize>().unwrap();
            for i in 0..n_R{
                let mut string=reads[i*nsta*nsta+3].trim().split_whitespace();
                let a=string.next().unwrap().parse::<isize>().unwrap();
                let b=string.next().unwrap().parse::<isize>().unwrap();
                let c=string.next().unwrap().parse::<isize>().unwrap();
                if (c>0) || (c==0 && b>0) ||(c==0 && b==0 && a>=0){
                    if a==0 && b==0 && c==0{
                        for ind_i in 0..nsta{
                            for ind_j in 0..nsta{
                                let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+3].trim().split_whitespace();
                                string.nth(4);
                                for r in 0..3{
                                    let re=string.next().unwrap().parse::<f64>().unwrap();
                                    let im=string.next().unwrap().parse::<f64>().unwrap();
                                    rmatrix[[0,r,ind_i,ind_j]]=Complex::new(re,im);
                                }
                            }
                        }
                    }else{
                        let mut matrix=Array4::<Complex<f64>>::zeros((1,3,nsta,nsta));
                        for ind_i in 0..nsta{
                            for ind_j in 0..nsta{
                                let mut string=reads[i*nsta*nsta+ind_i*nsta+ind_j+3].trim().split_whitespace();
                                string.nth(4);
                                for r in 0..3{
                                    let re=string.next().unwrap().parse::<f64>().unwrap();
                                    let im=string.next().unwrap().parse::<f64>().unwrap();
                                    matrix[[0,r,ind_i,ind_j]]=Complex::new(re,im);
                                }
                            }
                        }
                        rmatrix.append(Axis(0),matrix.view()).unwrap();
                    }
                }
            }
        }else{
           for i in 0..norb {
                for r in 0..3{
                    rmatrix[[0,r,i,i]]=Complex::<f64>::from(orb[[i,r]]);
                    if spin{
                        rmatrix[[0,r,i+norb,i+norb]]=Complex::<f64>::from(orb[[i,r]]);
                    }
                }
            }
        }
        let mut model=Model{
            dim_r:3,
            norb,
            nsta,
            natom,
            spin,
            lat,
            orb,
            atom,
            atom_list,
            ham,
            hamR,
            rmatrix,
        };
        model
    }
}

impl conductivity<'_> for Model{

    #[allow(non_snake_case)]
    fn berry_curvature_n_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array1::<f64>,Array1::<f64>){
        //!给定一个k点, 返回 $\Omega_n(\bm k)$
        //返回 $Omega_{n,\ap\bt}, \ve_{n\bm k}$
        let li:Complex<f64>=1.0*Complex::i();
        let (band,evec)=self.solve_onek(&k_vec);
        let mut v:Array3::<Complex<f64>>=self.gen_v(k_vec);
        let mut J:Array3::<Complex<f64>>=v.clone();
        if self.spin {
            let mut X:Array2::<Complex<f64>>=Array2::eye(self.nsta);
            let pauli:Array2::<Complex<f64>>= match spin{
                0=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,1.0+0.0*li]]),
                1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
            };
            X=kron(&pauli,&Array2::eye(self.norb));
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned();
                let j=anti_comm(&X,&j)/2.0; //这里做反对易
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                J.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_1[[i]],0.0));
                v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
            }
        };

        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=&evec_conj.dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=&evec_conj.dot(&A2);
        let mut U0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        for i in 0..self.nsta{
            for j in 0..self.nsta{
                if i != j{
                    U0[[i,j]]=1.0/((band[[i]]-band[[j]]).powi(2)-(og+li*eta).powi(2));
                }else{
                    U0[[i,j]]=Complex::new(0.0,0.0);
                }
            }
        }
        //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        let mut omega_n=Array1::<f64>::zeros(self.nsta);
        let A1=A1*U0;
        let A1=A1.axis_iter(Axis(0));
        let A2=A2.axis_iter(Axis(1));
        let omega_n=A1.zip(A2).map(|(x1,x2)| x1.dot(&x2).im*2.0).collect();
        let omega_n=Array1::from_vec(omega_n);
        /*
        for i in 0..self.nsta{
            omega_n[[i]]=2.0*A1.slice(s![i,..]).dot(&A2.slice(s![..,i])).im;
        }
        */
        (omega_n,band)
    }

    #[allow(non_snake_case)]
    fn berry_curvature_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        //!给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$, 
        //!mu=$\mu$ 为费米能级, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
        //!当体系不存在自旋的时候无论如何输入spin都默认 spin=0
        //!eta=$\eta$ 是一个小量
        //! 这个函数返回的是 
        //! $$\Og_{\ap\bt}^\gm= \sum_n f_n\Omega_{n,\ap\bt}^\gm(\bm k)=\sum_n \f{1}{e^{(\ve_{n\bm k}-\mu)/T/k_B}+1} \sum_{m=\not n}\f{J_{\ap,nm}^\gm v_{\bt,mn}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og+i\eta)^2}$$
        //! 其中 $J_\ap^\gm=\\{s_\gm,v_\ap\\}$
        let (omega_n,band)=self.berry_curvature_n_onek(&k_vec,&dir_1,&dir_2,og,spin,eta);
        let mut omega:f64=0.0;
        if T==0.0{
            for i in 0..self.nsta{
                omega+= if band[[i]]> mu {0.0} else {omega_n[[i]]};
            }
        }else{
            let beta=1.0/(T*8.617e-5);
            let fermi_dirac=band.map(|x| 1.0/((beta*(x-mu)).exp()+1.0));
            omega=(omega_n*fermi_dirac).sum();
        }
        omega
    }
    #[allow(non_snake_case)]
    fn berry_curvature(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->Array1::<f64>{
    //!这个是用来并行计算大量k点的贝利曲率
    //!这个可以用来画能带上的贝利曲率, 或者画一个贝利曲率的热图
        if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r{
            panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
        }
        let nk=k_vec.len_of(Axis(0));
        let omega:Vec<f64>=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let omega_one=self.berry_curvature_onek(&x.to_owned(),&dir_1,&dir_2,T,og,mu,spin,eta); 
            omega_one
            }).collect();
        let omega=arr1(&omega);
        omega
    }
    #[allow(non_snake_case)]
    fn Hall_conductivity(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
    //!这个是用来计算霍尔电导的.
    //!这里采用的是均匀撒点的方法, 利用 berry_curvature, 我们有
    //!$$\sg_{\ap\bt}^\gm=\f{1}{N(2\pi)^r V}\sum_{\bm k} \Og_{\ap\bt}^\gm(\bm k),$$ 其中 $N$ 是 k 点数目, 
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let omega=self.berry_curvature(&kvec,&dir_1,&dir_2,T,og,mu,spin,eta);
        //let min=omega.iter().fold(f64::NAN, |a, &b| a.min(b));
        let conductivity:f64=omega.sum()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        conductivity
    }
    #[allow(non_snake_case)]
    ///这个是采用自适应积分算法来计算霍尔电导的, 一般来说, 我们建议 re_err 设置为 1, 而 ab_err 设置为 0.01
    fn Hall_conductivity_adapted(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64,re_err:f64,ab_err:f64)->f64{
        let mut k_range=gen_krange(k_mesh);//将要计算的区域分成小块
        let n_range=k_range.len_of(Axis(0));
        let ab_err=ab_err/(n_range as f64);
        let use_fn=|k0:&Array1::<f64>| self.berry_curvature_onek(k0,&dir_1,&dir_2,T,og,mu,spin,eta);
        let inte=|k_range| adapted_integrate_quick(&use_fn,&k_range,re_err,ab_err);
        let omega:Vec<f64>=k_range.axis_iter(Axis(0)).into_par_iter().map(|x| { inte(x.to_owned())}).collect();
        let omega:Array1::<f64>=arr1(&omega);
        let conductivity:f64=omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        conductivity
    }
    ///用来计算多个 $\mu$ 值的, 这个函数是先求出 $\Omega_n$, 然后再分别用不同的费米能级来求和, 这样速度更快, 因为避免了重复求解 $\Omega_n$, 但是相对来说更耗内存, 而且不能做到自适应积分算法.
    fn Hall_conductivity_mu(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:&Array1::<f64>,spin:usize,eta:f64)->Array1::<f64>{
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let (omega_n,band):(Vec<_>,Vec<_>)=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let (omega_n,band)=self.berry_curvature_n_onek(&x.to_owned(),&dir_1,&dir_2,og,spin,eta); 
            (omega_n,band)
            }).collect();
        let omega_n=Array2::<f64>::from_shape_vec((nk, self.nsta),omega_n.into_iter().flatten().collect()).unwrap();
        let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
        let n_mu:usize=mu.len();
        let conductivity=if T==0.0{
            let conductivity_new:Vec<f64>=mu.into_par_iter().map(|x| {
                let mut omega=Array1::<f64>::zeros(nk);
                for k in 0..nk{
                    for i in 0..self.nsta{
                        omega[[k]]+= if band[[k,i]]> *x {0.0} else {omega_n[[k,i]]};
                    }
                }
                omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap()/(nk as f64)
            }).collect();
            Array1::<f64>::from_vec(conductivity_new)
        }else{
            let beta=1.0/(T*8.617e-5);
            let conductivity_new:Vec<f64>=mu.into_par_iter().map(|x| {
                let mut omega=Array1::<f64>::zeros(nk);
                let fermi_dirac=band.map(|x0| 1.0/((beta*(x0-x)).exp()+1.0));
                for i in 0..nk{
                    omega[[i]]=(omega_n.row(i).to_owned()*fermi_dirac.row(i).to_owned()).sum();
                }
                omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap()/(nk as f64)
            }).collect();
            Array1::<f64>::from_vec(conductivity_new)
        };
        conductivity
    }

    fn berry_curvature_dipole_n_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array1<f64>,Array1<f64>){
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

        let li:Complex<f64>=1.0*Complex::i();
        let (band,evec)=self.solve_onek(&k_vec);
        let mut v:Array3::<Complex<f64>>=self.gen_v(k_vec);
        let mut J:Array3::<Complex<f64>>=v.clone();
        let mut v0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//这个是速度项, 对应的dir_3 的速度
        for r in 0..self.dim_r{
            v0=v0+v.slice(s![r,..,..]).to_owned()*dir_3[[r]];
        }
        if self.spin {
            let mut X:Array2::<Complex<f64>>=Array2::eye(self.nsta);
            let pauli:Array2::<Complex<f64>>= match spin{
                0=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,1.0+0.0*li]]),
                1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
            };
            X=kron(&pauli,&Array2::eye(self.norb));
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned();
                let j=anti_comm(&X,&j)/2.0; //这里做反对易
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                J.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_1[[i]],0.0));
                v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
            }
        };

        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let v0=v0.dot(&evec.clone().reversed_axes());
        let v0=&evec_conj.dot(&v0);
        let partial_ve=v0.diag().map(|x| x.re);
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=&evec_conj.dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=&evec_conj.dot(&A2);
        let mut U0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        for i in 0..self.nsta{
            for j in 0..self.nsta{
                if i != j{
                    U0[[i,j]]=1.0/((band[[i]]-band[[j]]).powi(2)-(og+li*eta).powi(2));
                }else{
                    U0[[i,j]]=Complex::new(0.0,0.0);
                }
            }
        }
        //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        let mut omega_n=Array1::<f64>::zeros(self.nsta);
        let A1=A1*U0;

        let A1=A1.axis_iter(Axis(0));
        let A2=A2.axis_iter(Axis(1));
        let omega_n=A1.zip(A2).map(|(x1,x2)| x1.dot(&x2).im*2.0).collect();
        let omega_n=Array1::from_vec(omega_n);
        /*
        for i in 0..self.nsta{
            omega_n[[i]]=2.0*A1.slice(s![i,..]).dot(&A2.slice(s![..,i])).im;
        }
        */
        
        //let (omega_n,band)=self.berry_curvature_n_onek(&k_vec,&dir_1,&dir_2,og,spin,eta);
        let omega_n:Array1::<f64>=omega_n*partial_ve;
        (omega_n,band) //最后得到的 D
    }
    fn berry_curvature_dipole_n(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array2::<f64>,Array2::<f64>){
        //这个是在 onek的基础上进行并行计算得到一系列k点的berry curvature dipole
        //!This function performs parallel computation based on the onek function to obtain a series of Berry curvature dipoles at different k-points.
        //!这个方法用的是对费米分布的修正, 因为高阶的dipole 修正导致的非线性霍尔电导为 $$\sg_{\ap\bt\gm}=\tau\int\dd\bm k\sum_n\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}\lt\.\pdv{f_{\bm k}}{\ve}\rt\rvert_{E=\ve_{n\bm k}}.$$ 所以我们这里输出的是 
        //!$$\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}.$$ 
        if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
            panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
        }
        let nk=k_vec.len_of(Axis(0));
        let (omega,band):(Vec<_>,Vec<_>)=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let (omega_one,band)=self.berry_curvature_dipole_n_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,og,spin,eta);
            (omega_one,band)
            }).collect();
        let omega=Array2::<f64>::from_shape_vec((nk, self.nsta),omega.into_iter().flatten().collect()).unwrap();
        let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
        (omega,band)
    }
    fn Nonlinear_Hall_conductivity_Extrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,mu:&Array1<f64>,T:f64,og:f64,spin:usize,eta:f64)->Array1<f64>{
        //这个是用 berry curvature dipole 对整个布里渊去做积分得到非线性霍尔电导, 是extrinsic 的 
        //!This function calculates the extrinsic nonlinear Hall conductivity by integrating the Berry curvature dipole over the entire Brillouin zone. The Berry curvature dipole is first computed at a series of k-points using parallel computation based on the onek function.

        //! 我们基于 berry_curvature_n_dipole 来并行得到所有 k 点的 $\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$,
        //! 但是我们最后的公式为
        //! $$\\mathcal D_{\ap\bt\gm}=\int \dd\bm k \sum_n\lt(-\pdv{f_{n}}{\ve}\rt)\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$$
        //! 然而, 
        //! $$-\pdv{f_{n}}{\ve}=\beta\f{e^{beta(\ve_n-\mu)}}{(e^{beta(\ve_n-\mu)}+1)^2}=\beta f_n(1-f_n)$$
        //! 对于 T=0 的情况, 我们将采用四面体积分来替代, 这个需要很高的k点密度, 不建议使用
        //! 对于 T!=0 的情况, 我们会采用类似 Dos 的方法来计算


        if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
            panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
        }
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        //为了节省内存, 本来是可以直接算完求和, 但是为了方便, 我是先存下来再算, 让程序结构更合理
        let (omega,band)=self.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,og,spin,eta);
        let omega=omega.into_raw_vec();
        let band=band.into_raw_vec();
        let n_e=mu.len();
        let mut conductivity=Array1::<f64>::zeros(n_e);
        if T !=0.0{
            let beta=1.0/T/(8.617e-5);
            let use_iter=band.iter().zip(omega.iter()).par_bridge();
            conductivity=use_iter.fold(|| conductivity.clone(),|acc,(energy,omega0)|{
                let f=1.0/(beta*(mu-*energy)).mapv(|x| x.exp()+1.0);
                acc+&f*(1.0-&f)*beta**omega0
            }).reduce(|| conductivity.clone(), |acc, x| acc + x);
            conductivity=conductivity.clone()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        }else{
            //采用四面体积分法, 或者对于二维体系, 采用三角形积分法
            //积分的思路是, 通过将一个六面体变成5个四面体, 然后用线性插值的方法, 得到费米面,
            //以及费米面上的数, 最后, 通过积分算出来结果
        }
        return conductivity
    }

    fn berry_connection_dipole_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,spin:usize)->(Array1::<f64>,Array1::<f64>,Option<Array1<f64>>){
        //!这个是根据 Nonlinear_Hall_conductivity_intrinsic 的注释, 当不存在自旋的时候提供
        //!$$v_\ap G_{\bt\gm}-v_\bt G_{\ap\gm}$$
        //!其中 $$ G_{ij}=2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3} $$
        //!如果存在自旋, 即spin不等于0, 则还存在 $\p_{h_i} G_{jk}$ 项, 具体请看下面的非线性霍尔部分
        //!我们这里暂时不考虑磁场, 只考虑电场
        let mut v:Array3::<Complex<f64>>=self.gen_v(&k_vec);//这是速度算符
        let mut J=v.clone();
        let (band,evec)=self.solve_onek(&k_vec);//能带和本征值
        let evec_conj=evec.clone().map(|x| x.conj());//本征值的复共轭
        for i in 0..self.dim_r{
            let v_s=v.slice(s![i,..,..]).to_owned();
            let v_s=evec_conj.clone().dot(&(v_s.dot(&evec.clone().reversed_axes())));//变换到本征态基函数
            v.slice_mut(s![i,..,..]).assign(&v_s);//将 v 变换到以本征态为基底
        }
        //现在速度算符已经是以本征态为基函数
        let mut v_1=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//三个方向的速度算符
        let mut v_2=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        let mut v_3=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        for i in 0..self.dim_r{
            v_1=v_1.clone()+v.slice(s![i,..,..]).to_owned()*Complex::new(dir_1[[i]],0.0);
            v_2=v_2.clone()+v.slice(s![i,..,..]).to_owned()*Complex::new(dir_2[[i]],0.0);
            v_3=v_3.clone()+v.slice(s![i,..,..]).to_owned()*Complex::new(dir_3[[i]],0.0);
        }
        //三个方向的速度算符都得到了
        let mut U0=Array2::<f64>::zeros((self.nsta,self.nsta));
        for i in 0..self.nsta{
            for j in 0..self.nsta{
                if (band[[i]]-band[[j]]).abs() < 1e-5{
                    U0[[i,j]]=0.0;
                }else{
                    U0[[i,j]]=1.0/(band[[i]]-band[[j]]);
                }
            }
        }
        //这样U0[[i,j]]=1/(E_i-E_j), 这样就可以省略判断, 减少计算量

        //开始计算能带的导数, 详细的公式请看 berry_curvature_dipole_onek 的公式
        //其实就是速度算符的对角项
        //开始计算速度的偏导项, 这里偏导来自实空间
        let partial_ve_1=v_1.diag().map(|x| x.re);
        let partial_ve_2=v_2.diag().map(|x| x.re);
        let partial_ve_3=v_3.diag().map(|x| x.re);

        //开始最后的计算
        if self.spin{//如果考虑自旋, 我们就计算 \partial_h G_{ij}
            let mut S:Array2::<Complex<f64>>=Array2::eye(self.nsta);
            let li=Complex::<f64>::new(0.0,1.0);
            let pauli:Array2::<Complex<f64>>= match spin{
                0=> Array2::<Complex<f64>>::eye(2),
                1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
            };
            let X=kron(&pauli,&Array2::eye(self.norb));
            let mut S=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));
            for i in 0..self.dim_r{
                let v0=J.slice(s![i,..,..]).to_owned();
                let v0=anti_comm(&X,&v0)/2.0;
                let v0=evec_conj.clone().dot(&(v0.dot(&evec.clone().reversed_axes())));//变换到本征态基函数
                S.slice_mut(s![i,..,..]).assign(&v0);
            }
            let mut s_1=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//三个方向的速度算符
            let mut s_2=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let mut s_3=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            for i in 0..self.dim_r{
                s_1=s_1.clone()+S.slice(s![i,..,..]).to_owned()*Complex::new(dir_1[[i]],0.0);
                s_2=s_2.clone()+S.slice(s![i,..,..]).to_owned()*Complex::new(dir_2[[i]],0.0);
                s_3=s_3.clone()+S.slice(s![i,..,..]).to_owned()*Complex::new(dir_3[[i]],0.0);
            }
            let G_23:Array1::<f64>={//用来计算  beta gamma 的 G 
                let A=&v_2*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                let mut G=Array1::<f64>::zeros(self.nsta);
                for i in 0..self.nsta{
                    G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                }
                G
            };
            let G_13_h:Array1::<f64>={//用来计算 alpha gamma 的 G 
                let A=&s_1*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                let mut G=Array1::<f64>::zeros(self.nsta);
                for i in 0..self.nsta{
                    G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                }
                G
            };
            //开始计算partial_s
            let partial_s_1=s_1.clone().diag().map(|x| x.re);
            let partial_s_2=s_2.clone().diag().map(|x| x.re);
            let partial_s_3=s_3.clone().diag().map(|x| x.re);
            let mut partial_s=Array2::<f64>::zeros((self.dim_r,self.nsta));
            for r in 0..self.dim_r{
                let s0=S.slice(s![r,..,..]).to_owned();
                partial_s.slice_mut(s![r,..]).assign(&s0.diag().map(|x| x.re));//\p_i s算符的对角项
            }
            //开始计算partial G
            let partial_G:Array1::<f64>={
                let mut A=Array1::<Complex<f64>>::zeros(self.nsta);//第一项
                for i in 0..self.nsta{
                    for j in 0..self.nsta{
                        A[[i]]+=3.0*(partial_s_1[[i]]-partial_s_1[[j]])*v_2[[i,j]]*v_3[[j,i]]*U0[[i,j]].powi(4);
                    }
                }
                let mut B=Array1::<Complex<f64>>::zeros(self.nsta);//第二项
                for n in 0..self.nsta{
                    for n1 in 0..self.nsta{
                        for n2 in 0..self.nsta{
                            B[[n]]+=s_1[[n,n2]]*(v_2[[n2,n1]]*v_3[[n1,n]]+v_3[[n2,n1]]*v_2[[n1,n]])*U0[[n,n1]].powi(3)*U0[[n,n2]];
                        }
                    }
                }
                let mut C=Array1::<Complex<f64>>::zeros(self.nsta);//第三项
                for n in 0..self.nsta{
                    for n1 in 0..self.nsta{
                        for n2 in 0..self.nsta{
                            C[[n]]+=s_1[[n1,n2]]*(v_2[[n2,n]]*v_3[[n,n1]]+v_3[[n2,n]]*v_2[[n,n1]])*U0[[n,n1]].powi(3)*U0[[n1,n2]];
                        }
                    }
                }
                2.0*(A-B-C).map(|x| x.re)
            };
            //计算结束
            //开始最后的输出
            //println!("{},{}",&partial_s_1*&G_23,&partial_ve_2*&G_13_h);
            return ((partial_s_1*G_23-partial_ve_2*G_13_h),band,Some(partial_G))
        }else{
            //开始计算 G_{ij}
            //G_{ij}=2Re\sum_{m\neq n} v_{i,nm}v_{j,mn}/(E_n-E_m)^3
            let G_23:Array1::<f64>={//用来计算  beta gamma 的 G 
                let A=&v_2*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                let mut G=Array1::<f64>::zeros(self.nsta);
                for i in 0..self.nsta{
                    G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                }
                G
            };
            let G_13:Array1::<f64>={//用来计算 alpha gamma 的 G 
                let A=&v_1*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                let mut G=Array1::<f64>::zeros(self.nsta);
                for i in 0..self.nsta{
                    G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                }
                G
            };
            return (partial_ve_1*G_23-partial_ve_2*G_13,band,None)
        }
    }
    fn berry_connection_dipole(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,spin:usize)->(Array2<f64>,Array2<f64>,Option<Array2<f64>>){
        //! 这个是基于 onek 的, 进行关于 k 点并行求解
        if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
            panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
        }
        let nk=k_vec.len_of(Axis(0));
        
        if self.spin{
            let ((omega,band),partial_G):((Vec<_>,Vec<_>),Vec<_>)=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                let (omega_one,band,partial_G)=self.berry_connection_dipole_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,spin); 
                let partial_G=partial_G.unwrap();
                ((omega_one,band),partial_G)
                }).collect();

            let omega=Array2::<f64>::from_shape_vec((nk, self.nsta),omega.into_iter().flatten().collect()).unwrap();
            let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
            let partial_G=Array2::<f64>::from_shape_vec((nk, self.nsta),partial_G.into_iter().flatten().collect()).unwrap();

            return (omega,band,Some(partial_G))
        }else{
            let (omega,band):(Vec<_>,Vec<_>)=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                let (omega_one,band,partial_G)=self.berry_connection_dipole_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,spin); 
                (omega_one,band)
                }).collect();
            let omega=Array2::<f64>::from_shape_vec((nk, self.nsta),omega.into_iter().flatten().collect()).unwrap();
            let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
            return (omega,band,None)
        }
    }
    fn Nonlinear_Hall_conductivity_Intrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,mu:&Array1<f64>,T:f64,spin:usize)->Array1<f64>{
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


        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let (omega,band,mut partial_G):(Array2<f64>,Array2<f64>,Option<Array2<f64>>)=self.berry_connection_dipole(&kvec,&dir_1,&dir_2,&dir_3,spin);
        let omega=omega.into_raw_vec();
        let omega=Array1::from(omega);
        let band0=band.clone();
        let band=band.into_raw_vec();
        let band=Array1::from(band);
        let n_e=mu.len();
        let mut conductivity=Array1::<f64>::zeros(n_e);
        if T !=0.0{
            let beta=1.0/T/8.617e-5;
            let use_iter=band.iter().zip(omega.iter()).par_bridge();
            conductivity=use_iter.fold(|| Array1::<f64>::zeros(n_e),|acc,(energy,omega0)|{
                let f=1.0/((beta*(mu-*energy)).mapv(|x| x.exp()+1.0));
                acc+&f*(1.0-&f)*beta**omega0
            }).reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
            if self.spin{
                let partial_G=partial_G.unwrap();
                let conductivity_new:Vec<f64>=mu.into_par_iter().map(|x| {
                    let f=band0.map(|x0| 1.0/((beta*(x-x0)).exp()+1.0));
                    let mut omega=Array1::<f64>::zeros(nk);
                    for i in 0..nk{
                        omega[[i]]=(partial_G.row(i).to_owned()*f.row(i).to_owned()).sum();
                    }
                    omega.sum()/2.0
                }).collect();
                let conductivity_new=Array1::<f64>::from_vec(conductivity_new);
                conductivity=conductivity.clone()+conductivity_new;
            }
            conductivity=conductivity.clone()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        }else{
            //采用四面体积分法, 或者对于二维体系, 采用三角形积分法
            //积分的思路是, 通过将一个六面体变成5个四面体, 然后用线性插值的方法, 得到费米面,
            //以及费米面上的数, 最后, 通过积分算出来结果
            panic!("the code can not support for T=0");
        }
        return conductivity
    }
}

///这个模块是用 wilson loop 的方法来计算 berry phase 等数值的
impl Model{
    pub fn berry_loop(&self,kvec:&Array2<f64>)->Array1<f64>{
        let nk=kvec.len_of(Axis(0));
        let berry_phase=Array1::<f64>::zeros(nk);
        berry_phase
    }
}

///这个模块是用来求解表面格林函数的一个模块.
impl surf_Green{
    ///从 Model 中构建一个 surf_green 的结构体
    ///
    ///dir表示要看哪方向的表面态
    ///
    ///eta表示小虚数得取值
    ///
    ///对于非晶格矢量得方向, 需要用 model.make_supercell 先扩胞
    pub fn from_Model(model:&Model,dir:usize,eta:f64)->surf_Green{
        if dir > model.dim_r{
            panic!("Wrong, the dir must smaller than model's dim_r")
        }
        let mut R_max:usize=0;
        for R0 in model.hamR.rows(){
            if R_max < R0[[dir]].abs() as usize{
                R_max=R0[[dir]].abs() as usize;
            }
        }
        let mut U=Array2::<f64>::eye(model.dim_r);
        U[[dir,dir]]=R_max as f64;
        let model=model.make_supercell(&U);
        let mut ham0=Array3::<Complex<f64>>::zeros((0,model.nsta,model.nsta));
        let mut hamR0=Array2::<isize>::zeros((0,model.dim_r));
        let mut hamR=Array3::<Complex<f64>>::zeros((0,model.nsta,model.nsta));
        let mut hamRR=Array2::<isize>::zeros((0,model.dim_r));
        let use_hamR=model.hamR.rows();
        let use_ham=model.ham.axis_iter(Axis(0));
        for (ham,R) in use_ham.zip(use_hamR){
            let ham=ham.clone();
            let R=R.clone();
            if R[[dir]]==0{
                ham0.push(Axis(0),ham.view());
                hamR0.push_row(R.view());
            }else if R[[dir]] > 0{
                hamR.push(Axis(0),ham.view());
                hamRR.push_row(R.view());
            }else{
                hamR.push(Axis(0),ham.map(|x| x.conj()).t().view());
                hamRR.push_row(R.map(|x| -x).view());
            }
        }
        let new_lat=remove_row(model.lat,dir);
        let new_orb=remove_col(model.orb,dir);
        let new_atom=remove_col(model.atom,dir);
        let new_hamR0=remove_col(hamR0,dir);
        let new_hamRR=remove_col(hamRR,dir);
        let mut green:surf_Green=surf_Green{
            dim_r:model.dim_r-1,
            norb:model.norb,
            nsta:model.nsta,
            natom:model.natom,
            spin:model.spin,
            lat:new_lat,
            orb:new_orb,
            atom:new_atom,
            atom_list:model.atom_list,
            ham_bulk:ham0,
            ham_bulkR:new_hamR0,
            ham_hop:hamR,
            ham_hopR:new_hamRR,
            eta,
        };
        green
    }
    pub fn k_path(&self,path:&Array2::<f64>,nk:usize)->(Array2::<f64>,Array1::<f64>,Array1::<f64>){
        //!根据高对称点来生成高对称路径, 画能带图
        if self.dim_r==0{
            panic!("the k dimension of the model is 0, do not use k_path")
        }
        let n_node:usize=path.len_of(Axis(0));
        if self.dim_r != path.len_of(Axis(1)){
            panic!("Wrong, the path's length along 1 dimension must equal to the model's dimension")
        }
        let k_metric=(self.lat.dot(&self.lat.t())).inv().unwrap();
        let mut k_node=Array1::<f64>::zeros(n_node);
        for n in 1..n_node{
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk=path.row(n).to_owned()-path.slice(s![n-1,..]).to_owned();
            let a=k_metric.dot(&dk);
            let dklen=dk.dot(&a).sqrt();
            k_node[[n]]=k_node[[n-1]]+dklen;
        }
        let mut node_index:Vec<usize>=vec![0];
        for n in 1..n_node-1{
            let frac=k_node[[n]]/k_node[[n_node-1]];
            let a=(frac*((nk-1) as f64).round()) as usize;
            node_index.push(a)
        }
        node_index.push(nk-1);
        let mut k_dist=Array1::<f64>::zeros(nk);
        let mut k_vec=Array2::<f64>::zeros((nk,self.dim_r));
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&path.row(0));
        for n in 1..n_node {
            let n_i=node_index[n-1];
            let n_f=node_index[n];
            let kd_i=k_node[[n-1]];
            let kd_f=k_node[[n]];
            let k_i=path.row(n-1);
            let k_f=path.row(n);
            for j in n_i..n_f+1{
                let frac:f64= ((j-n_i) as f64)/((n_f-n_i) as f64);
                k_dist[[j]]=kd_i + frac*(kd_f-kd_i);
                k_vec.row_mut(j).assign(&((1.0-frac)*k_i.to_owned() +frac*k_f.to_owned()));

            }
        }
        (k_vec,k_dist,k_node)
    }

    #[inline(always)]
    pub fn gen_ham_onek(&self,kvec:&Array1<f64>)->(Array2<Complex<f64>>,Array2<Complex<f64>>){
        let mut ham0k=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        let mut hamRk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.ham_bulkR.len_of(Axis(0));
        let nRR:usize=self.ham_hopR.len_of(Axis(0));
        let U0:Array1::<f64>=self.orb.dot(kvec);
        let U0:Array1::<Complex<f64>>=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0:Array1::<Complex<f64>>=U0.mapv(Complex::exp);//关于轨道的 guage
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();//因为自旋, 把坐标扩大一倍
        }
        let U=Array2::from_diag(&U0);
        //对体系作傅里叶变换
        let U0=(self.ham_bulkR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let U0=U0.mapv(Complex::exp);
        //对 ham_hop 作傅里叶变换
        let UR=(self.ham_hopR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let UR=UR*Complex::new(0.0,2.0*PI);
        let UR=UR.mapv(Complex::exp);
        //先对 ham_bulk 中的 [0,0] 提取出来
        let mut ham0=self.ham_bulk.slice(s![0,..,..]).to_owned();
        for i in 1..nR{
            ham0k=ham0k+self.ham_bulk.slice(s![i,..,..]).to_owned()*U0[[i]];
        }
        for i in 0..nRR{
            hamRk=hamRk+self.ham_hop.slice(s![i,..,..]).to_owned()*UR[[i]];
        }
        ham0k=&ham0+&ham0k.map(|x| x.conj()).t()+&ham0k;
        //hamRk=&hamRk.map(|x| x.conj()).t()+&hamRk;
        //作规范变换
        ham0k=ham0k.dot(&U);
        let ham0k=U.map(|x| x.conj()).t().dot(&ham0k);
        hamRk=hamRk.dot(&U);
        let hamRk=U.map(|x| x.conj()).t().dot(&hamRk);
        (ham0k,hamRk)

    }
    pub fn surf_green_one(&self,kvec:&Array1<f64>,Energy:f64)->(f64,f64,f64){
        let (hamk,hamRk)=self.gen_ham_onek(kvec);
        let hamRk_conj:Array2<Complex<f64>>=hamRk.clone().map(|x| x.conj()).reversed_axes();
        let I0=Array2::<Complex<f64>>::eye(self.nsta);
        let accurate:f64=1e-8;
        let epsilon=Complex::new(Energy,self.eta)*&I0;
        let mut epi=hamk.clone();
        let mut eps=hamk.clone();
        let mut eps_t=hamk.clone();
        let mut ap=hamRk.clone();
        let mut bt=hamRk_conj.clone();

        for _ in 0..100{
            let g0=(&epsilon-&epi).inv().unwrap();
            let mat_1=&ap.dot(&g0);
            let mat_2=&bt.dot(&g0);
            let g0=&mat_1.dot(&bt);
            epi=epi+g0;
            eps=eps+g0;
            let g0=&mat_2.dot(&ap);
            epi=epi+g0;
            eps_t=eps_t+g0;
            ap=mat_1.dot(&ap);
            bt=mat_2.dot(&bt);
            //println!("{}",ap.map(|x| x.norm()).sum());
            if ap.sum().norm() < accurate{
                break
            }
        }
        let g_LL=(&epsilon-eps).inv().unwrap();
        let g_RR=(&epsilon-eps_t).inv().unwrap();
        let g_B=(&epsilon-epi).inv().unwrap();
        let N_R=-1.0/(PI)*g_RR.into_diag().sum().im;
        let N_L=-1.0/(PI)*g_LL.into_diag().sum().im;
        let N_B=-1.0/(PI)*g_B.into_diag().sum().im;
        (N_R,N_L,N_B)
    }

    pub fn surf_green_onek(&self,kvec:&Array1<f64>,Energy:&Array1<f64>)->(Array1<f64>,Array1<f64>,Array1<f64>){
        let (hamk,hamRk)=self.gen_ham_onek(kvec);
        let hamRk_conj:Array2<Complex<f64>>=hamRk.clone().map(|x| x.conj()).reversed_axes();
        let I0=Array2::<Complex<f64>>::eye(self.nsta);
        let accurate:f64=1e-16;
        let ((N_R,N_L),N_B)=Energy.into_par_iter().map(|e| {
            let epsilon=Complex::new(*e,self.eta)*&I0;
            let mut epi=hamk.clone();
            let mut eps=hamk.clone();
            let mut eps_t=hamk.clone();
            let mut ap=hamRk.clone();
            let mut bt=hamRk_conj.clone();
            for _ in 0..100{
                let g0=(&epsilon-&epi).inv().unwrap();
                let mat_1=&ap.dot(&g0);
                let mat_2=&bt.dot(&g0);
                let g0=&mat_1.dot(&bt);
                epi=epi+g0;
                eps=eps+g0;
                let g0=&mat_2.dot(&ap);
                epi=epi+g0;
                eps_t=eps_t+g0;
                ap=mat_1.dot(&ap);
                bt=mat_2.dot(&bt);
                if ap.map(|x| x.norm()).sum() < accurate{
                    break
                }
            }
            let g_LL=(&epsilon-eps).inv().unwrap();
            let g_RR=(&epsilon-eps_t).inv().unwrap();
            let g_B=(&epsilon-epi).inv().unwrap();
            let NR:f64=-1.0/(PI)*g_RR.into_diag().sum().im;
            let NL:f64=-1.0/(PI)*g_LL.into_diag().sum().im;
            let NB:f64=-1.0/(PI)*g_B.into_diag().sum().im;
            ((NR,NL),NB)
         }).collect();
        let N_R=Array1::from_vec(N_R);
        let N_L=Array1::from_vec(N_L);
        let N_B=Array1::from_vec(N_B);
        (N_R,N_L,N_B)
    }
    pub fn surf_green_path(&self,kvec:&Array2<f64>,E_min:f64,E_max:f64,E_n:usize)->(Array2<f64>,Array2<f64>,Array2<f64>){
        let Energy=Array1::<f64>::linspace(E_min,E_max,E_n);
        let ((N_R,N_L),N_B):((Vec<_>,Vec<_>),Vec<_>)=kvec.axis_iter(Axis(0)).into_par_iter().map( |k| {
            let (NR,NL,NB)=self.surf_green_onek(&k.to_owned(),&Energy);
            ((NR.to_vec(),NL.to_vec()),NB.to_vec())
        }).collect();
        let nk=kvec.nrows();
        let N_L = Array2::from_shape_vec((nk, E_n), N_L.into_iter().flatten().collect()).unwrap();
        let N_R = Array2::from_shape_vec((nk, E_n), N_R.into_iter().flatten().collect()).unwrap();
        let N_B = Array2::from_shape_vec((nk, E_n), N_B.into_iter().flatten().collect()).unwrap();
        (N_L,N_R,N_B)
    }
    pub fn show_surf_state(&self,name:&str,kpath:&Array2::<f64>,label:&Vec<&str>,nk:usize,E_min:f64,E_max:f64,E_n:usize){
        use std::io::{BufWriter, Write};
        use std::fs::create_dir_all;
        create_dir_all(name).expect("can't creat the file");
        let (kvec,kdist,knode)=self.k_path(kpath, nk);
        let Energy=Array1::<f64>::linspace(E_min,E_max,E_n);
        let (N_L,N_R,N_B)=self.surf_green_path(&kvec,E_min,E_max,E_n);

        //绘制 left_dos------------------------
        let mut left_name:String=String::new();
        left_name.push_str(&name.clone());
        left_name.push_str("/dos.surf_l");
        let mut file=File::create(left_name).expect("Unable to dos.surf_l.dat");
        let mut writer = BufWriter::new(file);
        for i in 0..nk{
            for j in 0..E_n{
                let mut s = String::new();
                let aa= format!("{:.6}", kdist[[i]]);
                s.push_str(&aa);
                let bb:String=format!("{:.6}",Energy[[j]]);
                if Energy[[j]]>=0.0{
                    s.push_str("    ");
                }else{
                    s.push_str("   ");
                }
                s.push_str(&bb);
                let cc:String=format!("{:.6}",N_L[[i,j]].ln());
                if N_L[[i,j]]>=0.0{
                    s.push_str("    ");
                }else{
                    s.push_str("   ");
                }
                s.push_str(&cc);
                //writeln!(file,"{}",s);

                writer.write(s.as_bytes()).unwrap();
            }
            //writeln!(file,"\n");
            writer.write(b"\n").unwrap();
        }
        let _=file;

        //接下来我们绘制表面态 
        use gnuplot::{Figure, AxesCommon, AutoOption::Fix,HOT,RAINBOW};
        let mut fg = Figure::new();
        let width:usize=nk;
        let height:usize=E_n;
        let mut heatmap_data = vec![];
        for i in 0..height {
            for j in 0..width {
                heatmap_data.push(N_L[[j, i]].ln());
            }
        }
        let axes = fg.axes2d();
        axes.set_palette(RAINBOW);
        axes.image(heatmap_data.iter(), width, height,None, &[]);
        let axes=axes.set_x_range(Fix(0.0), Fix(nk as f64));
        let axes=axes.set_y_range(Fix(0.0), Fix(E_n as f64));
        let axes=axes.set_aspect_ratio(Fix(1.0));
        let mut show_ticks=Vec::new();
        for i in 0..knode.len(){
            let A=knode[[i]];
            let B=label[i];
            show_ticks.push(Major(A,Fix(B)));
        }
        //axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
        let mut pdfname=String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_l.pdf");
        fg.set_terminal("pdfcairo",&pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _=fg;

        //绘制右表面态----------------------
        let mut left_name:String=String::new();
        left_name.push_str(&name.clone());
        left_name.push_str("/dos.surf_r");
        let mut file=File::create(left_name).expect("Unable to dos.surf_l.dat");
        let mut writer = BufWriter::new(file);
        for i in 0..nk{
            for j in 0..E_n{
                let mut s = String::new();
                let aa= format!("{:.6}", kdist[[i]]);
                s.push_str(&aa);
                let bb:String=format!("{:.6}",Energy[[j]]);
                if Energy[[j]]>=0.0{
                    s.push_str("    ");
                }else{
                    s.push_str("   ");
                }
                s.push_str(&bb);
                let cc:String=format!("{:.6}",N_R[[i,j]].ln());
                if N_L[[i,j]]>=0.0{
                    s.push_str("    ");
                }else{
                    s.push_str("   ");
                }
                s.push_str(&cc);
                //writeln!(file,"{}",s);

                writer.write(s.as_bytes()).unwrap();
            }
            //writeln!(file,"\n");
            writer.write(b"\n").unwrap();
        }
        let _=file;

        //接下来我们绘制表面态 
        let mut fg = Figure::new();
        let width:usize=nk;
        let height:usize=E_n;
        let mut heatmap_data = vec![];
        for i in 0..height {
            for j in 0..width {
                heatmap_data.push(N_R[[j, i]].ln());
            }
        }
        let axes = fg.axes2d();
        axes.set_palette(RAINBOW);
        axes.image(heatmap_data.iter(), width, height,None, &[]);
        let axes=axes.set_x_range(Fix(0.0), Fix(nk as f64));
        let axes=axes.set_y_range(Fix(0.0), Fix(E_n as f64));
        let axes=axes.set_aspect_ratio(Fix(1.0));
        let mut show_ticks=Vec::new();
        for i in 0..knode.len(){
            let A=knode[[i]];
            let B=label[i];
            show_ticks.push(Major(A,Fix(B)));
        }
        //axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
        let mut pdfname=String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_r.pdf");
        fg.set_terminal("pdfcairo",&pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _=fg;
        //绘制体态----------------------
        let mut left_name:String=String::new();
        left_name.push_str(&name.clone());
        left_name.push_str("/dos.surf_bulk");
        let mut file=File::create(left_name).expect("Unable to dos.surf_l.dat");
        let mut writer = BufWriter::new(file);
        for i in 0..nk{
            for j in 0..E_n{
                let mut s = String::new();
                let aa= format!("{:.6}", kdist[[i]]);
                s.push_str(&aa);
                let bb:String=format!("{:.6}",Energy[[j]]);
                if Energy[[j]]>=0.0{
                    s.push_str("    ");
                }else{
                    s.push_str("   ");
                }
                s.push_str(&bb);
                let cc:String=format!("{:.6}",N_B[[i,j]].ln());
                if N_L[[i,j]]>=0.0{
                    s.push_str("    ");
                }else{
                    s.push_str("   ");
                }
                s.push_str(&cc);
                //writeln!(file,"{}",s);

                writer.write(s.as_bytes()).unwrap();
            }
            //writeln!(file,"\n");
            writer.write(b"\n").unwrap();
        }
        let _=file;

        //接下来我们绘制表面态 
        let mut fg = Figure::new();
        let width:usize=nk;
        let height:usize=E_n;
        let mut heatmap_data = vec![];
        for i in 0..height {
            for j in 0..width {
                heatmap_data.push(N_B[[j, i]].ln());
            }
        }
        let axes = fg.axes2d();
        axes.set_palette(RAINBOW);
        axes.image(heatmap_data.iter(), width, height,None, &[]);
        let axes=axes.set_x_range(Fix(0.0), Fix(nk as f64));
        let axes=axes.set_y_range(Fix(0.0), Fix(E_n as f64));
        let axes=axes.set_aspect_ratio(Fix(1.0));
        let mut show_ticks=Vec::new();
        for i in 0..knode.len(){
            let A=knode[[i]];
            let B=label[i];
            show_ticks.push(Major(A,Fix(B)));
        }
        //axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
        let mut pdfname=String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_b.pdf");
        fg.set_terminal("pdfcairo",&pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _=fg;
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use num_complex::Complex;
    use super::*;
    use gnuplot::Major;
    use ndarray::prelude::*;
    use ndarray::*;
    use std::time::{Duration, Instant};
    use gnuplot::Figure;
    use gnuplot::Color;
    use gnuplot::PointSymbol;
    use gnuplot::AutoOption;
    use gnuplot::Fix;
    use gnuplot::AxesCommon;
    #[test]
    fn anti_comm_test(){
        let li:Complex<f64>=1.0*Complex::i();
        let t=-1.0+0.0*li;
        let t2=-1.0+0.0*li;
        let delta=0.7;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
        model.set_onsite(arr1(&[-delta,delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t,0,1,&R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,0,0,&R,0);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,1,1,&R,0);
        }

        let kvec=array![0.1,0.1];
        let (eval,evec)=model.solve_onek(&kvec);
        let v=model.gen_v(&kvec);
        let v1:Array2<Complex<f64>>=v.slice(s![0,..,..]).to_owned();
        let v2:Array2<Complex<f64>>=v.slice(s![1,..,..]).to_owned();
        println!("{}",v1.dot(&v2)-v2.dot(&v1));

    }
    #[test]
    fn Haldan_model(){
        let li:Complex<f64>=1.0*Complex::i();
        let t=-1.0+0.0*li;
        let t2=-1.0+0.0*li;
        let delta=0.7;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
        model.set_onsite(arr1(&[-delta,delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t,0,1,&R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,0,0,&R,0);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,1,1,&R,0);
        }
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","K'","G"];
        model.show_band(&path,&label,nk,"tests/Haldan");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=50;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let dir_3=arr1(&[0.0,1.0]);
        let spin:usize=0;
        let kmesh=arr1(&[nk,nk]);

        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("quantom_Hall_effect={}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间

        let nk:usize=50;
        let kmesh=arr1(&[nk,nk]);
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta,0.01,0.01);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("霍尔电导率{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
        //测试一下速度算符的对角项是否等于能带本征值的导数
        let kvec=array![1.0/2.0,1.0/2.0];
        let (band,evec)=model.solve_onek(&kvec);
        let evec_conj=evec.clone().map(|x| x.conj());
        let (omega_n,band)=model.berry_curvature_n_onek(&kvec,&dir_1,&dir_2,og,0,1e-5);
        let mut partial_ve=Array1::<f64>::zeros(model.nsta);
        let v=model.gen_v(&kvec);
        for i in 0..model.dim_r{
            let vs=v.slice(s![i,..,..]).to_owned(); 
            let vs=evec_conj.dot(&(vs.dot(&evec.clone().reversed_axes()))).diag().map(|x| x.re);
            partial_ve=partial_ve.clone()+vs*dir_3[[i]];
        }
        println!("{}",partial_ve);
        println!("{}",omega_n);
        //画一下3000k的时候的费米导数分布
        let T=100.0;
        let nk:usize=101;
        let kmesh=arr1(&[nk,nk]);
        println!("{}",kmesh);
        let E_min=-3.0;
        let E_max=3.0;
        let E_n=1000;
        let mu=Array1::linspace(E_min,E_max,E_n);
        let beta:f64=1.0/T/(8.617e-5);
        let f:Array1<f64>=1.0/((beta*mu.clone()).mapv(f64::exp)+1.0);
        let par_f=beta*f.clone()*(1.0-f);
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=par_f.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/par_f.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        //画一下omega_n 随能量的分布
        let kvec:Array2::<f64>=gen_kmesh(&kmesh);
        let nk:usize=kvec.len_of(Axis(0));
        let (omega,band)=model.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,og,spin,eta);
        let omega=omega.into_raw_vec();
        let omega=Array1::from(omega);
        let band=band.into_raw_vec();
        let band=Array1::from(band);
        let mut fg = Figure::new();
        let x:Vec<f64>=band.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=omega.to_vec();
        axes.points( x.iter(),y.iter(),&[Color("black"),PointSymbol((".").chars().next().unwrap())]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/omega_energy.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


        //开始算非线性霍尔电导

        let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,0,1e-5);

        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/nonlinear_ex.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        let (E0,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-2);
        //开始绘制dos
        let mut fg = Figure::new();
        let x:Vec<f64>=E0.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }
    #[test]
    fn graphene(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.1+0.0*li;
        let t3=0.0+0.0*li;
        let delta=0.0;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[3.0_f64.sqrt(),-1.0],[3.0_f64.sqrt(),1.0]]);
        let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        model.add_hop(t1,0,1,&array![0,0],0);
        model.add_hop(t1,0,1,&array![-1,0],0);
        model.add_hop(t1,0,1,&array![0,-1],0);
        model.add_hop(t2,0,0,&array![1,0],0);
        model.add_hop(t2,1,1,&array![1,0],0);
        model.add_hop(t2,0,0,&array![0,1],0);
        model.add_hop(t2,1,1,&array![0,1],0);
        model.add_hop(t2,0,0,&array![1,-1],0);
        model.add_hop(t2,1,1,&array![1,-1],0);
        model.add_hop(t3,0,1,&array![1,-1],0);
        model.add_hop(t3,0,1,&array![-1,1],0);
        model.add_hop(t3,0,1,&array![-1,-1],0);
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","G"];
        model.show_band(&path,&label,nk,"tests/graphene");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=11;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let spin:usize=0;
        let kmesh=arr1(&[nk,nk]);
        let (eval,evec)=model.solve_onek(&arr1(&[0.3,0.5]));
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        //println!("{}",conductivity/(2.0*PI));
        //开始计算边缘态, 首先是zigsag态
        let nk:usize=501;
        let U=arr2(&[[1.0,1.0],[-1.0,1.0]]);
        let super_model=model.make_supercell(&U);
        let zig_model=super_model.cut_piece(100,0);
        let path=[[0.0,0.0],[0.0,0.5],[0.0,1.0]];
        //let path=[[0.0,0.0],[0.5,0.0],[1.0,0.0]];
        //let path=[[0.0,0.0],[0.5,0.0],[0.5,0.5],[0.0,0.5],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=super_model.k_path(&path,nk);
        let (eval,evec)=super_model.solve_all_parallel(&k_vec);
        //let label=vec!["G","X","M","Y","G"];
        let label=vec!["G","M","G"];
        zig_model.show_band(&path,&label,nk,"tests/graphene_zig");

        //开始计算石墨烯的态密度
        let nk:usize=101;
        let kmesh=arr1(&[nk,nk]);
        let E_min=-3.0;
        let E_max=3.0;
        let E_n=1000;
        let (E0,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-2);
        //开始绘制dos
        let mut fg = Figure::new();
        let x:Vec<f64>=E0.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/graphene");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();



        //开始计算非线性霍尔电导
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let dir_3=arr1(&[1.0,0.0]);
        let og=0.0;
        let mu=Array1::linspace(E_min,E_max,E_n);
        let T=300.0;
        let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,0,1e-5);

        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/graphene");
        pdf_name.push_str("/nonlinear_ex.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }

    #[test]
    fn kane_mele(){
        let li:Complex<f64>=1.0*Complex::i();
        let delta=0.0;
        let t=-1.0+0.0*li;
        let alter=0.0+0.0*li;
        //let soc=-1.0+0.0*li;
        let soc=0.24+0.0*li;
        let rashba=-0.00+0.0*li;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(t,0,1,&R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(soc*li,0,0,&R,3);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(soc*li,1,1,&R,3);
        }
        //加入rashba项
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for  (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            let r0=R.map(|x| *x as f64).dot(&model.lat);
            model.add_hop(rashba*li*r0[[1]],0,0,&R,1);
            model.add_hop(rashba*li*r0[[0]],0,0,&R,2);
        }
        
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for  (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            let r0=R.map(|x| *x as f64).dot(&model.lat);
            model.add_hop(-rashba*li*r0[[1]],1,1,&R,1);
            model.add_hop(-rashba*li*r0[[0]],1,1,&R,2);
        }
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","K'","G"];
        model.show_band(&path,&label,nk,"tests/kane");
        //开始计算超胞

        let super_model=model.cut_piece(50,0);
        let path=[[0.0,0.0],[0.0,0.5],[0.0,1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        super_model.show_band(&path,&label,nk,"tests/kane_super");
        //开始计算表面态
        let green=surf_Green::from_Model(&model,0,1e-3);
        let E_min=-3.0;
        let E_max=3.0;
        let E_n=nk.clone();
        //let (hamR,hamRR)=green.gen_ham_onek(&array![0.5]);
        //println!("{:.3}, \n{:.3}",hamR,hamRR);
        let (N_L,N_R,N_B)=green.surf_green_one(&array![0.5],0.0);
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        green.show_surf_state("tests/kane",&path,&label,nk,E_min,E_max,E_n);

        /////开始计算体系的霍尔电导率//////
        let nk:usize=101;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        //let dir_1=arr1(&[3.0_f64.sqrt()/2.0,-0.5]);
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let spin:usize=3;
        let kmesh=arr1(&[nk,nk]);
        let (eval,evec)=model.solve_onek(&arr1(&[0.3,0.5]));
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
        let nk:usize=31;
        let kmesh=arr1(&[nk,nk]);
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta,0.01,0.01);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间



 

        //开始算非线性霍尔电导
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[1.0,0.0]);
        let dir_3=arr1(&[1.0,0.0]);
        let nk:usize=200;
        let kmesh=arr1(&[nk,nk]);
        let E_min=-0.2;
        let E_max=0.2;
        let E_n=1000;
        let og=0.0;
        let mu=Array1::linspace(E_min,E_max,E_n);
        let T=10.0;
        let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,1,1e-5);
        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/kane");
        pdf_name.push_str("/nonlinear_ex.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        let (E0,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-2);
        //开始绘制dos
        let mut fg = Figure::new();
        let x:Vec<f64>=E0.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/kane");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        //绘制非线性霍尔电导的平面图
        
        //画一下贝利曲率的分布
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[1.0,0.0]);
        let dir_3=arr1(&[1.0,0.0]);
        let T=1000.0;
        let nk:usize=1000;
        let kmesh=arr1(&[nk,nk]);
        let kvec=gen_kmesh(&kmesh);
        let kvec=kvec*2.0-0.1;
        let kvec=model.lat.dot(&(kvec.reversed_axes()));
        let kvec=kvec.reversed_axes();
        let (berry_curv,band)=model.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,0.0,1,1e-3);
        ///////////////////////////////////////////
        let beta=1.0/T/(8.617e-5);
        let f:Array2::<f64>=band.clone().map(|x| 1.0/((beta*x).exp()+1.0));
        let f=beta*&f*(1.0-&f);
        let berry_curv=(berry_curv.clone()*f).sum_axis(Axis(1));
        let data=berry_curv.into_shape((nk,nk)).unwrap();
        draw_heatmap(data,"nonlinear.pdf");

       //画一下贝利曲率的分布
        let nk:usize=1000;
        let kmesh=arr1(&[nk,nk]);
        let kvec=gen_kmesh(&kmesh);
        //let kvec=kvec-0.5;
        let kvec=kvec*2.0;
        let kvec=model.lat.dot(&(kvec.reversed_axes()));
        let kvec=kvec.reversed_axes();
        let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,0.0,0.0,1,1e-3);
        let data=berry_curv.into_shape((nk,nk)).unwrap();
        draw_heatmap((-data).map(|x| (x+1.0).log(10.0)),"heat_map.pdf");



    }

    #[test]
    fn Enonlinear(){
        //!arxiv 1706.07702
        //!用来测试非线性外在的霍尔电导
        let li:Complex<f64>=1.0*Complex::i();
        let delta=0.;
        let t1=1.0+0.0*li;
        let t2=0.2*t1;
        let t3=0.2*t1;
        let dim_r:usize=3;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0,0.0],[0.5,3.0_f64.sqrt()/2.0,0.0],[0.0,0.0,1.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0,0.0],[2.0/3.0,2.0/3.0,0.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0,0],[-1,0,0],[0,-1,0]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(t1,0,1,&R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0,1],[-1,1,1],[0,-1,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(t2,0,0,&R,0);
        }       
        let R0:Array2::<isize>=arr2(&[[1,0,-1],[-1,1,-1],[0,-1,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(t2,1,1,&R,0);
        }       
        let R=arr1(&[0,0,1]);
        model.set_hop(t3,0,0,&R,0);
        model.set_hop(t3,1,1,&R,0);
        let path=array![[0.0,0.0,0.0],[1.0/3.0,2.0/3.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0],[1.0/3.0,2.0/3.0,0.0],[1.0/3.0,2.0/3.0,0.5],[0.0,0.0,0.0],[0.0,0.0,0.5],[1.0/3.0,2.0/3.0,0.5],[0.5,0.5,0.5],[0.0,0.0,0.5]];
        let label=vec!["G","K","M","G","K","H","G","A","H","L","A"];
        let nk=1001;
        model.show_band(&path,&label,nk,"tests/Enonlinear");


        //开始计算非线性霍尔电导
        let dir_1=arr1(&[1.0,0.0,0.0]);
        let dir_2=arr1(&[0.0,1.0,0.0]);
        let dir_3=arr1(&[0.0,0.0,1.0]);
        let nk:usize=100;
        let kmesh=arr1(&[nk,nk,nk]);
        let E_min=-3.0;
        let E_max=3.0;
        let E_n=1000;
        let og=0.0;
        let mu=Array1::linspace(E_min,E_max,E_n);
        let T=30.0;
        let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,0,1e-5);

        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        axes.set_y_range(Fix(-10.0),Fix(10.0));
        axes.set_x_range(Fix(E_min),Fix(E_max));
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Enonlinear");
        pdf_name.push_str("/nonlinear_ex.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Intrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,3);
        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        axes.set_y_range(Fix(-10.0),Fix(10.0));
        axes.set_x_range(Fix(E_min),Fix(E_max));
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Enonlinear");
        pdf_name.push_str("/nonlinear_in.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


        let (E0,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-2);
        //开始绘制dos
        let mut fg = Figure::new();
        let x:Vec<f64>=E0.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Enonlinear");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


    }
    #[test]
    fn kagome(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.1+0.0*li;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[3.0_f64.sqrt(),-1.0],[3.0_f64.sqrt(),1.0]]);
        let orb=arr2(&[[0.0,0.0],[1.0/3.0,0.0],[0.0,1.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
        //最近邻hopping
        model.add_hop(t1,0,1,&array![0,0],0);
        model.add_hop(t1,2,0,&array![0,0],0);
        model.add_hop(t1,1,2,&array![0,0],0);
        model.add_hop(t1,0,2,&array![0,-1],0);
        model.add_hop(t1,0,1,&array![-1,0],0);
        model.add_hop(t1,2,1,&array![-1,1],0);
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.],[0.0,0.0]];
        let path=arr2(&path);
        let label=vec!["G","K","M","G"];
        model.show_band(&path,&label,nk,"tests/kagome/");
        //start to draw the band structure
        //Starting to calculate the edge state, first is the zigzag state
        let nk:usize=501;
        let U=arr2(&[[1.0,1.0],[-1.0,1.0]]);
        let super_model=model.make_supercell(&U);
        let zig_model=super_model.cut_piece(30,0);
        let path=[[0.0,0.0],[0.0,0.5],[0.0,1.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=super_model.k_path(&path,nk);
        let (eval,evec)=super_model.solve_all_parallel(&k_vec);
        let label=vec!["G","M","G"];
        zig_model.show_band(&path,&label,nk,"tests/kagome_zig/");
        //Starting to calculate the DOS of kagome
        let nk:usize=101;
        let kmesh=arr1(&[nk,nk]);
        let E_min=-3.0;
        let E_max=3.0;
        let E_n=1000;
        let (E0,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-2);
        //start to show DOS
        let mut fg = Figure::new();
        let x:Vec<f64>=E0.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/kagome/");
        pdf_name.push_str("dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }

    #[test]
    fn SSH(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.1+0.0*li;
        let dim_r:usize=1;
        let norb:usize=2;
        let lat=arr2(&[[1.0]]);
        let orb=arr2(&[[0.0],[0.5]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
        model.add_hop(t1,0,1,&array![0],0);
        model.add_hop(t2,0,1,&array![-1],0);
        let nk:usize=1001;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        model.show_band(&path,&label,nk,"tests/SSH/");
        println!("atom={}",model.atom);
        println!("{}",model.ham);
        println!("{}",model.hamR);
        model.orb-=0.25;
        model.atom-=0.25;
        println!("orb_old={}",model.orb);
        model.shift_to_zero();
        println!("orb_new={}",model.orb);
        println!("{}",model.ham);
        println!("{}",model.hamR);
        model.show_band(&path,&label,nk,"tests/SSH/new/");
    }
}
