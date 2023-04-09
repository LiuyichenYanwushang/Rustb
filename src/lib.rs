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
/// This cate is used to perform various calculations on the TB model, currently including:
///
/// 1: Calculate the band structure
///
/// 2: Expand the cell and calculate the surface state
///
/// 3: Calculate the first-order anomalous Hall conductivity and spin Hall conductivity
///
#[allow(non_snake_case)]
pub struct Model{
/// The real space dimension of the model.
pub dim_r:usize,
/// The number of orbitals in the model.
pub norb:usize,
/// The number of states in the model. If spin is enabled, nsta=norb$\times$2
pub nsta:usize,
/// The number of atoms in the model. The atom and atom_list at the back are used to store the positions of the atoms, and the number of orbitals corresponding to each atom.
pub natom:usize,
/// Whether the model has spin enabled. If enabled, spin=true
pub spin:bool,
/// The lattice vector of the model, a dim_r$\times$dim_r matrix, the axis0 direction stores a 1$\times$dim_r lattice vector.
pub lat:Array2::<f64>,
/// The position of the orbitals in the model. We use fractional coordinates uniformly.
pub orb:Array2::<f64>,
/// The position of the atoms in the model, also in fractional coordinates.
pub atom:Array2::<f64>,
/// The number of orbitals in the atoms, in the same order as the atom positions.
pub atom_list:Vec<usize>,
/// The Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
pub ham:Array3::<Complex<f64>>,
/// The distance between the unit cell hoppings, i.e. R in $\bra{m0}\hat H\ket{nR}$.
pub hamR:Array2::<isize>,
/// The position matrix, i.e. $\bra{m0}\hat{\bm r}\ket{nR}$.
pub rmatrix:Array4::<Complex<f64>>,
}
#[allow(non_snake_case)]
pub fn find_R(hamR:&Array2::<isize>,R:&Array1::<isize>)->bool{
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
    let A0=A.clone();
    let B0=B.clone();
    let C=A0.dot(&B0);
    let D=B0.dot(&A0);
    C-D
}
#[allow(non_snake_case)]
pub fn anti_comm(A:&Array2::<Complex<f64>>,B:&Array2::<Complex<f64>>)->Array2::<Complex<f64>>{
    let A0=A.clone();
    let B0=B.clone();
    let C=A0.dot(&B0)+B0.dot(&A0);
    C
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


#[allow(non_snake_case)]
impl Model{

    pub fn tb_model(dim_r:usize,norb:usize,lat:Array2::<f64>,orb:Array2::<f64>,spin:bool,natom:Option<usize>,atom:Option<Array2::<f64>>,atom_list:Option<Vec<usize>>)->Model{
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
        //! - norb: the number of orbitals
        //!
        //! - lat: the lattice constant
        //!
        //! - orb: the orbital coordinates
        //!
        //! - spin: whether to consider spin
        //!
        //! - natom: the number of atoms, which can be None
        //!
        //! - atom: the atomic coordinates, which can be None
        //!
        //! - atom_list: the number of orbitals for each atom, which can be None.
        //!
        //! Note that if any of the atomic variables are None, it is better to make them all None.
        let mut nsta:usize=norb;
        if spin{
            nsta*=2;
        }
        let mut new_natom:usize=0;
        let mut new_atom_list:Vec<usize>=vec![1];
        let mut new_atom=Array2::<f64>::zeros((0,dim_r));
        if lat.len_of(Axis(1)) != dim_r{
            panic!("Wrong, the lat's second dimension's length must equal to dim_r") 
        }
        if lat.len_of(Axis(0))<lat.len_of(Axis(1)) {
            panic!("Wrong, the lat's second dimension's length must less than first dimension's length") 
        }
        if natom==None{
           if atom !=None && atom_list !=None{
                let use_natom:usize=atom.as_ref().unwrap().len_of(Axis(0)).try_into().unwrap();
                if use_natom != atom_list.as_ref().unwrap().len().try_into().unwrap(){
                    panic!("Wrong, the length of atom_list is not equal to the natom");
                }
                new_natom=use_natom;
            }else if atom_list !=None || atom != None{
                panic!("Wrong, the atom and atom_list is not all None, please correspondence them");
            }else if atom_list==None && atom==None{
                //通过判断轨道是不是离得太近而判定是否属于一个原子,
                //这种方法只适用于wannier90不开最局域化
                new_atom.push_row(orb.row(0));
                for i in 0..norb{
                    if (orb.row(i).to_owned()-new_atom.row(new_atom.nrows()-1).to_owned()).norm_l2()>1e-2{
                        new_atom.push_row(orb.row(i));
                        new_atom_list.push(1);
                        new_natom+=1;
                    }else{
                        let last=new_atom_list.pop().unwrap();
                        new_atom_list.push(last+1);
                    }
                }
            } else{
                new_natom=norb.clone();
            };
        }else{
            new_natom=natom.unwrap();
            if atom_list==None || atom==None{
                panic!("Wrong, the atom and atom_list is None but natom is not none")
            }else{
                new_atom=atom.unwrap();
                new_atom_list=atom_list.unwrap();
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
            natom:new_natom,
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
    pub fn set_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize){
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
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_i,ind_j]]=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]=tmp;},
                    1=>{self.ham[[index,ind_i+self.norb,ind_j]]=tmp; self.ham[[index,ind_i,ind_j+self.norb]]=tmp;},
                    2=>{self.ham[[index,ind_i+self.norb,ind_j]]=tmp*Complex::<f64>::i(); self.ham[[index,ind_i,ind_j+self.norb]]=-tmp*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]=-tmp;},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_i,ind_j]]=tmp;
            }
            if index==0 && ind_i != ind_j{
                if self.spin{
                    match pauli{
                        0=>{self.ham[[0,ind_j,ind_i]]=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i+self.norb]]=tmp.conj();},
                        1=>{self.ham[[0,ind_j,ind_i+self.norb]]=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i]]=tmp.conj();},
                        2=>{self.ham[[0,ind_j,ind_i+self.norb]]=tmp.conj()*Complex::<f64>::i(); self.ham[[0,ind_j+self.norb,ind_i]]=-tmp.conj()*Complex::<f64>::i();},
                        3=>{self.ham[[0,ind_i,ind_j]]=tmp.conj(); self.ham[[0,ind_i+self.norb,ind_j+self.norb]]=-tmp.conj();},
                        _ => todo!()
                    }
                }else{
                    self.ham[[0,ind_j,ind_i]]=tmp.conj();
                }
            }
            if ind_i==ind_j && tmp.im !=0.0 && (pauli==0 ||pauli==3) && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            if self.ham[[index,ind_j,ind_i]]!=Complex::new(0.0,0.0){
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_j,ind_i]])
            }
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_j,ind_i]]=tmp; self.ham[[index,ind_j+self.norb,ind_i+self.norb]]=tmp.conj();},
                    1=>{self.ham[[index,ind_j,ind_i+self.norb]]=tmp.conj(); self.ham[[index,ind_j+self.norb,ind_i]]=tmp.conj();},
                    2=>{self.ham[[index,ind_j,ind_i+self.norb]]=tmp.conj()*Complex::<f64>::i(); self.ham[[index,ind_j+self.norb,ind_i]]=-tmp.conj()*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]=tmp.conj(); self.ham[[index,ind_i+self.norb,ind_j+self.norb]]=-tmp.conj();},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_j,ind_i]]=tmp.conj();
            }
        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            if self.spin{
                match pauli{
                    0=>{new_ham[[ind_i,ind_j]]=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]=tmp;},
                    1=>{new_ham[[ind_i+self.norb,ind_j]]=tmp; new_ham[[ind_i,ind_j+self.norb]]=tmp;},
                    2=>{new_ham[[ind_i+self.norb,ind_j]]=tmp*Complex::<f64>::i(); new_ham[[ind_i,ind_j+self.norb]]=-tmp*Complex::<f64>::i();},
                    3=>{new_ham[[ind_i,ind_j]]=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]=-tmp;},
                    _ => todo!()
                }
            }else{
                new_ham[[ind_i,ind_j]]=tmp;
            }
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }

    #[allow(non_snake_case)]
    pub fn add_hop(&mut self,tmp:Complex<f64>,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize){
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
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_i,ind_j]]+=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]+=tmp;},
                    1=>{self.ham[[index,ind_i+self.norb,ind_j]]+=tmp; self.ham[[index,ind_i,ind_j+self.norb]]+=tmp;},
                    2=>{self.ham[[index,ind_i+self.norb,ind_j]]+=tmp*Complex::<f64>::i(); self.ham[[index,ind_i,ind_j+self.norb]]-=tmp*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]+=tmp; self.ham[[index,ind_i+self.norb,ind_j+self.norb]]-=tmp;},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_i,ind_j]]+=tmp;
            }
            if index==0 && ind_i !=ind_j{
                if self.spin{
                    match pauli{
                        0=>{self.ham[[0,ind_j,ind_i]]+=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i+self.norb]]+=tmp.conj();},
                        1=>{self.ham[[0,ind_j,ind_i+self.norb]]+=tmp.conj(); self.ham[[0,ind_j+self.norb,ind_i]]+=tmp.conj();},
                        2=>{self.ham[[0,ind_j,ind_i+self.norb]]+=tmp.conj()*Complex::<f64>::i(); self.ham[[0,ind_j+self.norb,ind_i]]-=tmp.conj()*Complex::<f64>::i();},
                        3=>{self.ham[[0,ind_i,ind_j]]+=tmp.conj(); self.ham[[0,ind_i+self.norb,ind_j+self.norb]]-=tmp.conj();},
                        _ => todo!()
                    }
                }else{
                    self.ham[[0,ind_j,ind_i]]+=tmp.conj();
                }
            }
            if ind_i==ind_j && tmp.im !=0.0 && (pauli==0 ||pauli==3) && index==0{
                panic!("Wrong, the onsite hopping must be real, but here is {}",tmp)
            }
        }else if negative_R_exist {
            let index=index_R(&self.hamR,&negative_R);
            if self.ham[[index,ind_j,ind_i]]!=Complex::new(0.0,0.0){
                println!("Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",self.ham[[index,ind_j,ind_i]])
            }
            if self.spin{
                match pauli{
                    0=>{self.ham[[index,ind_j,ind_i]]+=tmp; self.ham[[index,ind_j+self.norb,ind_i+self.norb]]+=tmp.conj();},
                    1=>{self.ham[[index,ind_j,ind_i+self.norb]]+=tmp.conj(); self.ham[[index,ind_j+self.norb,ind_i]]+=tmp.conj();},
                    2=>{self.ham[[index,ind_j,ind_i+self.norb]]+=tmp.conj()*Complex::<f64>::i(); self.ham[[index,ind_j+self.norb,ind_i]]-=tmp.conj()*Complex::<f64>::i();},
                    3=>{self.ham[[index,ind_i,ind_j]]+=tmp.conj(); self.ham[[index,ind_i+self.norb,ind_j+self.norb]]-=tmp.conj();},
                    _ => todo!()
                }
            }else{
                self.ham[[index,ind_j,ind_i]]+=tmp.conj();
            }
        }else{
            let mut new_ham=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            if self.spin{
                match pauli{
                    0=>{new_ham[[ind_i,ind_j]]+=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]+=tmp;},
                    1=>{new_ham[[ind_i+self.norb,ind_j]]+=tmp; new_ham[[ind_i,ind_j+self.norb]]+=tmp;},
                    2=>{new_ham[[ind_i+self.norb,ind_j]]+=tmp*Complex::<f64>::i(); new_ham[[ind_i,ind_j+self.norb]]-=tmp*Complex::<f64>::i();},
                    3=>{new_ham[[ind_i,ind_j]]+=tmp; new_ham[[ind_i+self.norb,ind_j+self.norb]]-=tmp;},
                    _ => todo!()
                }
            }else{
                new_ham[[ind_i,ind_j]]+=tmp;
            }
            self.ham.push(Axis(0),new_ham.view()).unwrap();
            self.hamR.push(Axis(0),R.view()).unwrap();
        }
    }
    #[allow(non_snake_case)]
    pub fn set_onsite(&mut self, tmp:Array1::<f64>,pauli:isize){
        //! 直接对对角项进行设置
        if tmp.len()!=self.norb{
            panic!("Wrong, the norb is {}, however, the onsite input's length is {}",self.norb,tmp.len())
        }
        for (i,item) in tmp.iter().enumerate(){
            self.set_onsite_one(*item,i,pauli)
        }
    }
    #[allow(non_snake_case)]
    pub fn set_onsite_one(&mut self, tmp:f64,ind:usize,pauli:isize){
        //!对  $\bra{i\bm 0}\hat H\ket{i\bm 0}$ 进行设置
        let R=Array1::<isize>::zeros(self.dim_r);
        self.set_hop(Complex::new(tmp,0.0),ind,ind,R,pauli)
    }
    pub fn del_hop(&mut self,ind_i:usize,ind_j:usize,R:Array1::<isize>,pauli:isize) {
        //! 删除 $\bra{i\bm 0}\hat H\ket{j\bm R}$
        if R.len()!=self.dim_r{
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i>=self.norb ||ind_j>=self.norb{
            panic!("Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",self.norb,ind_i,ind_j)
        }
        self.set_hop(Complex::new(0.0,0.0),ind_i,ind_j,R,pauli);
    }

    #[allow(non_snake_case)]
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
    #[allow(non_snake_case)]
    pub fn gen_ham(&self,kvec:&Array1::<f64>)->Array2::<Complex<f64>>{
        //!这个是做傅里叶变换, 将实空间的哈密顿量变换到倒空间的哈密顿量
        //!
        //!具体来说, 就是
        //!$$H_{mn,\bm k}=\bra{m\bm k}\hat H\ket{n\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{-i(\bm R-\bm\tau_i+\bm \tau_j)\cdot\bm k}$$
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0:Array1::<f64>=self.orb.dot(kvec);
        let U0:Array1::<Complex<f64>>=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0:Array1::<Complex<f64>>=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        let ham0=self.ham.slice(s![0,..,..]).to_owned();
        for i in 1..nR{
            hamk=hamk+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]];
        }
        hamk=ham0+hamk.map(|x| x.conj()).t()+hamk;
        hamk=hamk.dot(&U);
        let re_ham=U.map(|x| x.conj()).t().dot(&hamk);
        re_ham
    }
    #[allow(non_snake_case)]
    pub fn gen_r(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>{
        //!和 gen_ham 类似, 将 $\hat{\bm r}$ 进行傅里叶变换
        //!
        //!$$\bm r_{mn,\bm k}=\bra{m\bm k}\hat{\bm r}\ket{n\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat{\bm r}\ket{n\bm R}e^{-i(\bm R-\bm\tau_i+\bm \tau_j)\cdot\bm k}$$
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0=self.orb.dot(kvec);
        let U0=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut rk=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));
        let r0=self.rmatrix.slice(s![0,..,..,..]).to_owned();
        if self.rmatrix.len_of(Axis(0))==1{
            return self.rmatrix.slice(s![0,..,..,..]).to_owned()
        }else{
            for i in 1..nR{
                rk=rk+self.rmatrix.slice(s![i,..,..,..]).to_owned()*Us[[i]];
            }
            for i in 0..self.dim_r{
                let use_rk=rk.slice(s![i,..,..]).to_owned();
                let use_rk:Array2::<Complex<f64>>=r0.slice(s![i,..,..]).to_owned()+use_rk.map(|x| x.conj()).t()+use_rk;
                //接下来向位置算符添加轨道的位置项
                let use_rk=use_rk.dot(&U); //
                rk.slice_mut(s![i,..,..]).assign(&(U.map(|x| x.conj()).t().dot(&use_rk)));
            }
            return rk
        }
    }
    ///这个函数是用来生成速度算符的, 即 $\bra{u_{m\bm k}}\p_\ap H_{\bm k}\ket{u_{n\bm k}}$
    #[allow(non_snake_case)]
    pub fn gen_v(&self,kvec:&Array1::<f64>)->Array3::<Complex<f64>>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        let nR:usize=self.hamR.len_of(Axis(0));
        let U0=self.orb.dot(kvec);
        let U0=U0.map(|x| Complex::<f64>::new(*x,0.0));
        let U0=U0*Complex::new(0.0,2.0*PI);
        let mut U0=U0.mapv(Complex::exp);
        if self.spin{
            let UU=U0.clone();
            U0.append(Axis(0),UU.view()).unwrap();
        }
        let U=Array2::from_diag(&U0);
        let Us=(self.hamR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
        let Us=Us*Complex::new(0.0,2.0*PI);
        let Us=Us.mapv(Complex::exp);
        let mut UU=Array3::<f64>::zeros((self.dim_r,self.nsta,self.nsta));
        let orb_real=self.orb.dot(&self.lat);
        for r in 0..self.dim_r{
            for i in 0..self.norb{
                for j in 0..self.norb{
                    UU[[r,i,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                    if self.spin{
                        UU[[r,i+self.norb,j]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i,j+self.norb]]=-orb_real[[i,r]]+orb_real[[j,r]];
                        UU[[r,i+self.norb,j+self.norb]]=-orb_real[[i,r]]+orb_real[[j,r]];
                    }
                }
            }
        }
        let UU=UU.map(|x| Complex::<f64>::new(0.0,*x)); //UU[i,j]=-tau[i]+tau[j] 
        let mut v=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));//定义一个初始化的速度矩阵
        let ham0=self.ham.slice(s![0,..,..]).to_owned();
        let R0=self.hamR.clone().map(|x| Complex::<f64>::new((*x) as f64,0.0));
        let R0=R0.dot(&self.lat.map(|x| Complex::new(*x,0.0)));
        for i0 in 0..self.dim_r{
            let mut hamk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let mut vv=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            for i in 1..nR{
                vv=vv+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]]*R0[[i,i0]]*Complex::i(); //这一步对 R 求和
                hamk=hamk+self.ham.slice(s![i,..,..]).to_owned()*Us[[i]];
            }
            vv=vv.clone().reversed_axes().map(|x| x.conj())+vv;
            let hamk0=hamk.clone();
            let hamk=hamk+self.ham.slice(s![0,..,..]).to_owned()+hamk0.map(|x| x.conj()).t();
            vv=vv+hamk.clone()*UU.slice(s![i0,..,..]).to_owned();
            let vv=vv.dot(&U); //加下来两步填上轨道坐标导致的相位
            let vv=U.map(|x| x.conj()).t().dot(&vv);
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
                let A=rk.slice(s![i,..,..]).to_owned()-UU;
                let A=comm(&hamk,&A)*Complex::i();
                let vv=v.slice(s![i,..,..]).to_owned().clone();
                v.slice_mut(s![i,..,..]).assign(&(vv+A));
            }
        }
        v
    }

    #[allow(non_snake_case)]
    pub fn solve_band_onek(&self,kvec:&Array1::<f64>)->Array1::<f64>{
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
        } 
        let hamk=self.gen_ham(&kvec);
        let hamk_conj=hamk.clone().map(|x| x.conj());
        let hamk_conj=hamk_conj.t();
        let sum0=(hamk.clone()-hamk_conj).sum();
        if sum0.im()> 1e-8 || sum0.re() >1e-8{
            panic!("Wrong, hamiltonian is not hamilt");
        }
        let eval = if let Ok(eigvals) = hamk.eigvalsh(UPLO::Lower) { eigvals } else { todo!() };
        eval
    }
    pub fn solve_band_all(&self,kvec:&Array2::<f64>)->Array2::<f64>{
        let nk=kvec.len_of(Axis(0));
        let mut band=Array2::<f64>::zeros((nk,self.nsta));
        for i in 0..nk{
            //let k=kvec.slice(s![i,..]).to_owned();
            let k=kvec.row(i).to_owned();
            let eval=self.solve_band_onek(&k);
            band.slice_mut(s![i,..]).assign(&eval);
        }
        band
    }
    #[allow(non_snake_case)]
    pub fn solve_band_all_parallel(&self,kvec:&Array2::<f64>)->Array2::<f64>{
        let nk=kvec.len_of(Axis(0));
        let eval:Vec<_>=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let eval=self.solve_band_onek(&x.to_owned()); 
            eval.to_vec()
            }).collect();
        let band = Array2::from_shape_vec((nk, self.nsta), eval.into_iter().flatten().collect()).unwrap();
        band
    }
    #[allow(non_snake_case)]
    pub fn solve_onek(&self,kvec:&Array1::<f64>)->(Array1::<f64>,Array2::<Complex<f64>>){
        if kvec.len() !=self.dim_r{
            panic!("Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",kvec.len(),self.dim_r)
        } 
        let hamk=self.gen_ham(&kvec);
        let hamk_conj=hamk.clone().map(|x| x.conj());
        let hamk_conj=hamk_conj.t();
        let sum0=(hamk.clone()-hamk_conj).sum();
        if sum0.im()> 1e-8 || sum0.re() >1e-8{
            panic!("Wrong, hamiltonian is not hamilt");
        }
        let (eval, evec) = if let Ok((eigvals, eigvecs)) = hamk.eigh(UPLO::Lower) { (eigvals, eigvecs) } else { todo!() };
        let evec=evec.reversed_axes().map(|x| x.conj());
        (eval,evec)
    }
    #[allow(non_snake_case)]
    pub fn solve_all(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
        let nk=kvec.len_of(Axis(0));
        let mut band=Array2::<f64>::zeros((nk,self.nsta));
        let mut vectors=Array3::<Complex<f64>>::zeros((nk,self.nsta,self.nsta));
        for i in 0..nk{
            //let k=kvec.slice(s![i,..]).to_owned();
            let k=kvec.row(i).to_owned();
            let (eval,evec)=self.solve_onek(&k);
            band.slice_mut(s![i,..]).assign(&eval);
            vectors.slice_mut(s![i,..,..]).assign(&evec);
        }
        (band,vectors)
    }
    #[allow(non_snake_case)]
    pub fn solve_all_parallel(&self,kvec:&Array2::<f64>)->(Array2::<f64>,Array3::<Complex<f64>>){
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
    /*
    ///这个函数是用来将model的某个方向进行截断的
    ///
    ///num:截出多少个原胞
    ///
    ///dir:方向
    ///
    ///返回一个model, 其中 dir 和输入的model是一致的, 但是轨道数目和原子数目都会扩大num倍, 沿着dir方向没有胞间hopping.
    */
    
    pub fn cut_piece(&self,num:usize,dir:usize)->Model{
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
        let mut new_orb=Array2::<f64>::zeros((self.norb*num,self.dim_r));//定义一个新的轨道
        let mut new_atom=Array2::<f64>::zeros((self.natom*num,self.dim_r));//定义一个新的原子
        let new_norb=self.norb*num;
        let new_nsta=self.nsta*num;
        let new_natom=self.nsta*num;
        let mut new_atom_list:Vec<usize>=Vec::new();
        let mut new_lat=self.lat.clone();
        new_lat.row_mut(dir).assign(&(self.lat.row(dir).to_owned()*(num as f64)));
        for i in 0..num{
            for n in 0..self.norb{
                let mut use_orb=self.orb.row(n).to_owned();
                use_orb[[dir]]=use_orb[[dir]]/(num as f64);
                new_orb.row_mut(i*self.norb+n).assign(&use_orb);
            }
            for n in 0..self.natom{
                let mut use_atom=self.atom.row(n).to_owned();
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
            for n in 0..num{
                for i in 0..new_norb{
                    for r in 0..self.dim_r{
                        new_rmatrix[[0,r,i,i]]=Complex::new(new_orb[[i,r]],0.0);
                        if self.spin{
                            new_rmatrix[[0,r,i+new_norb,i+new_norb]]=Complex::new(new_orb[[i,r]],0.0);
                        }
                    }
                }
            }
        }
        let n_R=self.hamR.len_of(Axis(0));
        for n in 0..num{
            for i0 in 0..n_R{
                let mut ind_R:Array1::<isize>=self.hamR.row(i0).to_owned();
                let mut rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,new_nsta,new_nsta));
                let ham=if ind_R[[dir]]<0{//如果这个方向的ind_R 小于0, 将其变成大于0
                    ind_R*=-1;
                    let h0=self.ham.slice(s![i0,..,..]).map(|x| x.conj()).t().to_owned();
                    if exist_r{
                        rmatrix=self.rmatrix.slice(s![i0,..,..,..]).map(|x| x.conj());
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
                        for i in 0..self.norb{
                            for j in 0..self.norb{
                                use_ham[[i+n*self.norb,j+ind*self.norb]]=ham[[i,j]];
                                use_ham[[i+n*self.norb+new_norb,j+ind*self.norb]]=ham[[i+self.norb,j]];
                                use_ham[[i+n*self.norb,j+ind*self.norb+new_norb]]=ham[[i,j+self.norb]];
                                use_ham[[i+n*self.norb+new_norb,j+ind*self.norb+new_norb]]=ham[[i+self.norb,j+self.norb]];
                            }
                        }
                        if R_exist{
                            let index=index_R(&new_hamR,&ind_R);
                            if index==0 && ind !=0{
                                for i in 0..self.norb{
                                    for j in 0..self.norb{
                                       use_ham[[j+ind*self.norb,i+n*self.norb]]=ham[[i,j]].conj();
                                       use_ham[[j+ind*self.norb,i+n*self.norb+new_norb]]=ham[[i+self.norb,j]].conj();
                                       use_ham[[j+ind*self.norb+new_norb,i+n*self.norb]]=ham[[i,j+self.norb]].conj();
                                       use_ham[[j+ind*self.norb+new_norb,i+n*self.norb+new_norb]]=ham[[i+self.norb,j+self.norb]].conj();
                                    }
                                }
                            }
                        }
                    }else{
                        for i in 0..self.norb{
                            for j in 0..self.norb{
                               use_ham[[i+n*self.norb,j+ind*self.norb]]=ham[[i,j]];
                            }
                        }
                        if R_exist{
                            let index=index_R(&new_hamR,&ind_R);
                            if index==0 && ind !=0{
                                for i in 0..self.norb{
                                    for j in 0..self.norb{
                                       use_ham[[j+ind*self.norb,i+n*self.norb]]=ham[[i,j]].conj();
                                    }
                                }
                            }
                        }
                    }
                    //开始对 r_matrix 进行操作
                    let mut use_rmatrix=Array3::<Complex<f64>>::zeros((self.dim_r,new_nsta,new_nsta));
                    if exist_r{
                        if self.spin{ //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                            for i in 0..self.norb{
                                for j in 0..self.norb{
                                    for r in 0..self.dim_r{
                                        use_rmatrix[[r,i+n*self.norb,j+ind*self.norb]]=rmatrix[[r,i,j]];
                                        use_rmatrix[[r,i+n*self.norb+new_norb,j+ind*self.norb]]=rmatrix[[r,i+self.norb,j]];
                                        use_rmatrix[[r,i+n*self.norb,j+ind*self.norb+new_norb]]=rmatrix[[r,i,j+self.norb]];
                                        use_rmatrix[[r,i+n*self.norb+new_norb,j+ind*self.norb+new_norb]]=rmatrix[[r,i+self.norb,j+self.norb]];
                                    }
                                }
                            }
                            if R_exist{
                                let index=index_R(&new_hamR,&ind_R);
                                if index==0 && ind !=0{
                                    for i in 0..self.norb{
                                        for j in 0..self.norb{
                                            for r in 0..self.dim_r{
                                               use_rmatrix[[r,j+ind*self.norb,i+n*self.norb]]=rmatrix[[r,i,j]].conj();
                                               use_rmatrix[[r,j+ind*self.norb,i+n*self.norb+new_norb]]=rmatrix[[r,i+self.norb,j]].conj();
                                               use_rmatrix[[r,j+ind*self.norb+new_norb,i+n*self.norb]]=rmatrix[[r,i,j+self.norb]].conj();
                                               use_rmatrix[[r,j+ind*self.norb+new_norb,i+n*self.norb+new_norb]]=rmatrix[[r,i+self.norb,j+self.norb]].conj();
                                            }
                                        }
                                    }
                                }
                            }
                        }else{
                            for i in 0..self.norb{
                                for j in 0..self.norb{
                                    for r in 0..self.dim_r{
                                       use_rmatrix[[r,i+n*self.norb,j+ind*self.norb]]=rmatrix[[r,i,j]];
                                    }
                                }
                            }
                            if R_exist{
                                let index=index_R(&new_hamR,&ind_R);
                                if index==0 && ind !=0{
                                    for i in 0..self.norb{
                                        for j in 0..self.norb{
                                            for r in 0..self.dim_r{
                                               use_rmatrix[[r,j+ind*self.norb,i+n*self.norb]]=rmatrix[[r,i,j]].conj();
                                           }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if R_exist{
                        let index=index_R(&new_hamR,&ind_R);
                        let addham=new_ham.slice(s![index,..,..]).to_owned();
                        new_ham.slice_mut(s![index,..,..]).assign(&(addham+use_ham));
                        let addr=new_rmatrix.slice(s![index,..,..,..]).to_owned();
                        new_rmatrix.slice_mut(s![index,..,..,..]).assign(&(addr+use_rmatrix));
                    }else if negative_R_exist{
                        let index=index_R(&new_hamR,&negative_R);
                        let addham=new_ham.slice(s![index,..,..]).to_owned();
                        new_ham.slice_mut(s![index,..,..]).assign(&(addham+use_ham.t().map(|x| x.conj())));
                        let mut addr=new_rmatrix.slice(s![index,..,..,..]).to_owned();
                        use_rmatrix.swap_axes(1,2);
                        new_rmatrix.slice_mut(s![index,..,..,..]).assign(&(addr+use_rmatrix.map(|x| x.conj())));
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

    pub fn make_supercell(&self,U:&Array2::<f64>)->Model{
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
        let mut use_atom=self.atom.dot(&U_inv);
        let mut orb_list:Vec<usize>=Vec::new();
        //let mut orb_list_R=Array2::<isize>::zeros((0,self.dim_r));
        let mut new_orb=Array2::<f64>::zeros((0,self.dim_r));
        let mut new_atom=Array2::<f64>::zeros((0,self.dim_r));
        let mut new_atom_list:Vec<usize>=Vec::new();//新的模型的 atom_list
        let mut use_atom_list:Vec<usize>=Vec::new();
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
                                let mut atoms=use_atom.row(n).to_owned()+(i as f64)*U_inv.row(0).to_owned()+(j as f64)*U_inv.row(1).to_owned()+(k as f64)*U_inv.row(2).to_owned(); //原子的位置在新的坐标系下的坐标
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
                            let mut atoms=use_atom.row(n).to_owned()+(i as f64)*U_inv.row(0).to_owned()+(j as f64)*U_inv.row(1).to_owned(); //原子的位置在新的坐标系下的坐标
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
    #[allow(non_snake_case)]
    pub fn dos(&self,k_mesh:&Array1::<usize>,E_min:f64,E_max:f64,E_n:usize,sigma:f64)->(Array1::<f64>,Array1::<f64>){
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let band=self.solve_band_all_parallel(&kvec);
        let E0=Array1::linspace(E_min,E_max,E_n);
        let mut dos=Array1::<f64>::zeros(E_n);
        let mut nk:usize=1;
        let dim:usize=k_mesh.len();
        for i in 0..dim{
            nk*=k_mesh[[i]]-1;
        }
        let mut centre=Array1::<f64>::zeros(0);
        for i in 0..nk{
            centre.append(Axis(0),band.row(i)).unwrap();
        }
        let sigma0=1.0/sigma;
        let pi0=1.0/(2.0*PI).sqrt();
        for i in centre.iter(){
            dos.map(|x| x+pi0*sigma0*(-((x-i)*sigma0).powi(2)/2.0).exp());
        }
        dos=dos/(nk as f64);
        (E0,dos)
    }

    #[allow(non_snake_case)]
    pub fn berry_curvature_n_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,og:f64,spin:usize,eta:f64)->Array1::<f64>{
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
                let v0=v.slice(s![i,..,..]).to_owned();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned();
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                let v0=v.slice(s![i,..,..]).to_owned();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        };

        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=evec_conj.clone().dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=evec_conj.dot(&A2);
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
        let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        omega_n
    }

    #[allow(non_snake_case)]
    pub fn berry_curvature_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
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
                let v0=v.slice(s![i,..,..]).to_owned();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned();
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                let v0=v.slice(s![i,..,..]).to_owned();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        };

        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=evec_conj.clone().dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=evec_conj.dot(&A2);
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
        let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
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
    ///这个是用来并行计算大量k点的贝利曲率
    #[allow(non_snake_case)]
    pub fn berry_curvature(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->Array1::<f64>{
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
    ///这个是用来计算霍尔电导的.
    #[allow(non_snake_case)]
    pub fn Hall_conductivity(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let omega=self.berry_curvature(&kvec,&dir_1,&dir_2,T,og,mu,spin,eta);
        let conductivity:f64=omega.sum()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        conductivity
    }
    #[allow(non_snake_case)]
    ///这个是采用自适应积分算法来计算霍尔电导的, 一般来说, 我们建议 re_err 设置为 1, 而 ab_err 设置为 0.01
    pub fn Hall_conductivity_adapted(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64,re_err:f64,ab_err:f64)->f64{
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
    pub fn Hall_conductivity_mu(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,T:f64,og:f64,mu:Array1::<f64>,spin:usize,eta:f64)->Array1::<f64>{
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let omega_n:Vec<_>=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let omega_n=self.berry_curvature_n_onek(&x.to_owned(),&dir_1,&dir_2,og,spin,eta); 
            omega_n
            }).collect();
        let omega_n=Array2::<f64>::from_shape_vec((nk, self.nsta),omega_n.into_iter().flatten().collect()).unwrap();
        let n_mu:usize=mu.len();
        let mut conductivity=Array1::zeros(n_mu);
        let band=self.solve_band_all_parallel(&kvec);
        if T==0.0{
            for s in 0..n_mu{
                let mut omega=Array1::<f64>::zeros(nk);
                for k in 0..nk{
                    for i in 0..self.nsta{
                        omega[[k]]+= if band[[k,i]]> mu[[s]] {0.0} else {omega_n[[k,i]]};
                    }
                }
                conductivity[[s]]=omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
            }
        }else{
            let beta=1.0/(T*8.617e-5);
            for s in 0..n_mu{
                let fermi_dirac=band.map(|x| 1.0/((beta*(x-mu[[s]])).exp()+1.0));
                let mut omega=Array1::<f64>::zeros(nk);
                for i in 0..nk{
                    omega[[i]]=(omega_n.row(i).to_owned()*fermi_dirac.row(i).to_owned()).sum();
                }
                conductivity[[s]]=omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
            }
        }
        conductivity
    }
    ///这个方法用的是对费米分布的修正, 因为高阶的dipole 修正导致的非线性霍尔电导为 $$\sg_{\ap\bt\gm}=\int\dd\bm k\sum_n\p_\gm\ve_{n\bm k}\Og_{nn,\ap\bt}\lt\.\pdv{f}{E}\rt\rvert_{E=\ve_{n\bm k}}.$$ 所以我们这里输出的是 
    ///$$\\mathcal D_{\ap\bt\gm}=\sum_n\p_\gm\ve_{n\bm k}\Og_{nn,\ap\bt}\lt\.\pdv{f}{E}\rt\rvert_{E=\ve_{n\bm k}}.$$ 这里需要注意的一点是, 一般来说对于 $\p_\ap\ve_{\bm k}$, 需要用差分法来求解, 我这里提供了一个算法. 
    ///$$ \ve_{\bm k}=U^\dag H_{\bm k} U\Rightarrow \pdv{\ve_{\bm k}}{\bm k}=U^\dag\pdv{H_{\bm k}}{\bm k}U+\pdv{U^\dag}{\bm k} H_{\bm k}U+U^\dag H_{\bm k}\pdv{U}{\bm k}$$
    ///因为 $U^\dag U=1\Rightarrow \p_{\bm k}U^\dag U=-U^\dag\p_{\bm k}U$, $\p_{\bm k}H_{\bm k}=v_{\bm k}$我们有
    ///$$\pdv{\ve_{\bm k}}{\bm k}=U^\dag v_{\bm k}U+\lt[\ve_{\bm k},U^\dag\p_{\bm k}U\rt]$$
    ///而这里面唯一比较难求的项是 $D_{\bm k}=U^\dag\p_{\bm k}U$. 按照 vanderbilt 2008 年的论文中的公式, 用微扰论有 $$D_{mn,\bm k}=\left\\{\\begin{aligned}\f{v_{mn,\bm k}}{\ve_n-\ve_m} \quad &\text{if}\\ m\\ \not= n\\\ 0 \quad \quad &\text{if}\\ m\\ = n\\end{aligned}\right\.$$
    ///需要特别注意的是, 最好不要将温度设置为0, 因为这样只有能量完全等于费米能级的时候才会有贡献, 我这里设置的是能量和费米能级小于 $10^{-3}$ 的时候有贡献.
    pub fn berry_curvature_dipole_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        //我们首先求解 omega_n 和 U^\dag j
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
                let v0=v.slice(s![i,..,..]).to_owned();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned();
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                let v0=v.slice(s![i,..,..]).to_owned();
                v.slice_mut(s![i,..,..]).assign(&(v0*dir_2[[i]]));
            }
        };
        let mut D=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//这个是求解本征值倒数的对易项
        let mut v0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//这个是速度项
        for r in 0..self.dim_r{
            for i in 0..self.nsta{
                for j in 0..self.nsta{
                    if i != j{
                        D[[i,j]]=v[[r,i,j]]*dir_3[[r]]/(band[[j]]-band[[i]]);
                    }
                }
            }
            let vs=v.slice(s![r,..,..]).to_owned()*dir_3[[r]];
            v0=v0+vs;
        }
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let ve=Array2::<Complex<f64>>::from_diag(&band.map(|x| Complex::new(*x,0.0)));//将能带变成对角形式
        let partial_ve:Array1::<f64>=(evec_conj.clone().dot(&(v0.dot(&evec.clone().reversed_axes())))+comm(&ve,&D)).diag().map(|x| x.re);//关于速度的偏导项
        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=evec_conj.clone().dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=evec_conj.dot(&A2);
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
        let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        let mut partial_f=Array1::<f64>::zeros(self.nsta);
        if T==0.0{
            println!("you set temperature is zero, this may cause wrong, because no bands lie on the fermi_energy");
            for i in 0..self.nsta{
                if (band[[i]]-mu).abs()<1e-3{
                    partial_f[[i]]=1.0;
                }
            }
        }else{
            let beta=1.0/(T*8.617e-5);
            let a:Array1::<f64>=((band-mu)*beta).map(|x| x.exp());
            partial_f=(a.clone()/(1.0+a)).map(|x| x.powi(2))*beta;
        }
        let omega=(partial_f*omega_n*partial_ve).sum();
        omega //最后得到的 D
        
    }
    pub fn berry_curvature_dipole(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->Array1::<f64>{
        //这个是在 onek的基础上进行并行计算得到一系列k点的berry curvature dipole
        //!This function performs parallel computation based on the onek function to obtain a series of Berry curvature dipoles at different k-points.
        if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
            panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
        }
        let nk=k_vec.len_of(Axis(0));
        let omega:Vec<f64>=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
            let omega_one=self.berry_curvature_dipole_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,T,og,mu,spin,eta); 
            omega_one
            }).collect();
        let omega=arr1(&omega);
        omega
    }
    pub fn Nonlinear_Hall_conductivity_Extrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        //这个是用 berry curvature dipole 对整个布里渊去做积分得到非线性霍尔电导, 是extrinsic 的 
        //!This function calculates the extrinsic nonlinear Hall conductivity by integrating the Berry curvature dipole over the entire Brillouin zone. The Berry curvature dipole is first computed at a series of k-points using parallel computation based on the onek function.
        let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
        let nk:usize=kvec.len_of(Axis(0));
        let omega=self.berry_curvature_dipole(&kvec,&dir_1,&dir_2,&dir_3,T,og,mu,spin,eta);
        let nonlinear_conductivity:f64=omega.sum()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
        nonlinear_conductivity
    }
    
    /*
    pub fn Nonlinear_Hall_conductivity_intrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,T:f64,og:f64,mu:f64,spin:usize,eta:f64)->f64{
        //!这个是来自 PRL 112, 166601 (2014) 这篇论文, 使用的公式为
        //!$$

    }
    */

    ///这个函数是用来快速画能带图的, 用python画图, 因为Rust画图不太方便.
    #[allow(non_snake_case)]
    pub fn show_band(&self,path:&Array2::<f64>,label:&Vec<&str>,nk:usize,name:&str)-> std::io::Result<()>{
        use std::fs::create_dir_all;
        use std::path::Path;
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
        Ok(())
    }
    #[allow(non_snake_case)]
    pub fn from_hr(path:&str,file_name:&str,zero_energy:f64)->Model{
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
            let string:Vec<usize>=reads[i].trim().split_whitespace().map(|x| x.parse::<usize>().unwrap()).collect();
            weights.extend(string.clone());
            if string.len() !=15{
                n_line=i+1;
                break
            }
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
        let mut atom=Array2::<f64>::zeros((1,3)); //原子位置坐标初始化
        let mut proj_name:Vec<&str>=Vec::new();
        let mut proj_list:Vec<usize>=Vec::new();
        let mut atom_list:Vec<usize>=Vec::new();
        let mut atom_name:Vec<&str>=Vec::new();
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
                            atom_name.push(prj[0])
                        }
                    }
                }
            }
        }
        for name in atom_name.iter(){
            for (j,j_name) in proj_name.iter().enumerate(){
                if j_name==name{
                    atom_list.push(proj_list[j])
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
        let mut atom=Array2::<f64>::zeros((natom,3));
        for i in 0..norb{
            let a:Vec<&str>=reads[i+2].trim().split_whitespace().collect();
            orb[[i,0]]=a[1].parse::<f64>().unwrap();
            orb[[i,1]]=a[2].parse::<f64>().unwrap();
            orb[[i,2]]=a[3].parse::<f64>().unwrap();
        }
        for i in 0..natom{
            let a:Vec<&str>=reads[i+2+nsta].trim().split_whitespace().collect();
            atom[[i,0]]=a[1].parse::<f64>().unwrap();
            atom[[i,1]]=a[2].parse::<f64>().unwrap();
            atom[[i,2]]=a[3].parse::<f64>().unwrap();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use ndarray::*;
    use std::time::{Duration, Instant};
    #[test]
    fn anti_comm_test(){
        let a=array![[1.0,2.0,3.0],[0.0,1.0,0.0],[0.0,0.0,0.0]];
        let b=array![[1.0,0.0,0.0],[1.0,1.0,0.0],[2.0,0.0,1.0]];
        let c=a.dot(&b)+b.dot(&a);
        println!("{}",c)
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
        let mut model=Model::tb_model(dim_r,norb,lat,orb,false,None,None,None);
        model.set_onsite(arr1(&[-delta,delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t,0,1,R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,0,0,R,0);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.add_hop(t2*li,1,1,R,0);
        }
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","K'","G"];
        model.show_band(&path,&label,nk,"tests/Haldan");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=101;
        let T:f64=0.0;
        let eta:f64=0.001;
        let og:f64=0.0;
        let mu:f64=0.0;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let spin:usize=0;
        let kmesh=arr1(&[nk,nk]);

        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间

        let nk:usize=11;
        let kmesh=arr1(&[nk,nk]);
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,T,og,mu,spin,eta,0.01,0.01);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
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
        let mut model=Model::tb_model(dim_r,norb,lat,orb,false,None,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        model.add_hop(t1,0,1,array![0,0],0);
        model.add_hop(t1,0,1,array![-1,0],0);
        model.add_hop(t1,0,1,array![0,-1],0);
        model.add_hop(t2,0,0,array![1,0],0);
        model.add_hop(t2,1,1,array![1,0],0);
        model.add_hop(t2,0,0,array![0,1],0);
        model.add_hop(t2,1,1,array![0,1],0);
        model.add_hop(t2,0,0,array![1,-1],0);
        model.add_hop(t2,1,1,array![1,-1],0);
        model.add_hop(t3,0,1,array![1,-1],0);
        model.add_hop(t3,0,1,array![-1,1],0);
        model.add_hop(t3,0,1,array![-1,-1],0);
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
        zig_model.show_band(&path,&label,nk,"tests/graphene");
    }

    #[test]
    fn kane_mele(){
        let li:Complex<f64>=1.0*Complex::i();
        let delta=0.7;
        let t=-1.0+0.0*li;
        let rashba=0.0+0.0*li;
        let soc=-1.0+0.0*li;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,norb,lat,orb,true,None,None,None);
        model.set_onsite(arr1(&[delta,-delta]),0);
        let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(t,0,1,R,0);
        }
        let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(soc*li,0,0,R,3);
        }
        let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
        for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
            let R=R.to_owned();
            model.set_hop(soc*li,1,1,R,3);
        }
        let nk:usize=1001;
        let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
        let path=arr2(&path);
        let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
        let (eval,evec)=model.solve_all_parallel(&k_vec);
        let label=vec!["G","K","M","K'","G"];
        model.show_band(&path,&label,nk,"tests/kane");
        /////开始计算体系的霍尔电导率//////
        let nk:usize=41;
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
    }
}
