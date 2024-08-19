#![allow(warnings)]
pub mod basis;
pub mod conductivity;
pub mod surfgreen;
pub mod geometry;
pub mod ndarray_lapack;
pub mod generics;
pub mod output;
pub mod atom_struct;
pub mod model_struct;
pub mod symm;
pub mod phy_const;
use gnuplot::Major;
use num_complex::Complex;
use num_traits::identities::Zero;
use ndarray::linalg::kron;
use ndarray::*;
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
use std::ops::Deref;
use std::time::Instant;
use crate::generics::usefloat;
use crate::atom_struct::{Atom,OrbProj};
#[doc(hidden)]
pub use crate::surfgreen::surf_Green;

/// This cate is used to perform various calculations on the TB model, currently including:
///
/// - Calculate the band structure
///
/// - Expand the cell and calculate the surface state
///
/// - Calculate the first-order anomalous Hall conductivity and spin Hall conductivity
///


///这个是 tight-binding 模型的基本单位
#[derive(Clone,Debug)]
pub struct Model{
    /// - The real space dimension of the model.
    pub dim_r:usize,
    /// - The number of orbitals in the model.
    pub spin:bool,
    /// - The lattice vector of the model, a dim_r$\times$dim_r matrix, the axis0 direction stores a 1$\times$dim_r lattice vector.
    pub lat:Array2::<f64>,
    /// - The position of the orbitals in the model. We use fractional coordinates uniformly.
    pub orb:Array2::<f64>,
    /// - The position of the atoms in the model, also in fractional coordinates.
    pub orb_projection:Vec<OrbProj>,
    /// - The projection of the orbs, like px,py,pz, etc.
    pub atoms:Vec<Atom>,
    /// - The number of orbitals in the atoms, in the same order as the atom positions.
    pub ham:Array3::<Complex<f64>>,
    /// - The distance between the unit cell hoppings, i.e. R in $\bra{m0}\hat H\ket{nR}$.
    pub hamR:Array2::<isize>,
    /// - The position matrix, i.e. $\bra{m0}\hat{\bm r}\ket{nR}$.
    pub rmatrix:Array4::<Complex<f64>>,
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
#[allow(non_snake_case)]
#[inline(always)]
pub fn gen_kmesh<T>(k_mesh:&Array1::<usize>)->Array2::<T>
    where T:usefloat+ std::ops::Div<Output = T>,
          {
    let dim:usize=k_mesh.len();
    let mut nk:usize=1;
    for i in 0..dim{
        nk*=k_mesh[[i]];
    }
    fn gen_kmesh_arr<T>(k_mesh:&Array1::<usize>,r0:usize,mut usek:Array1::<T>)->Array2::<T>
        where T:usefloat+ std::ops::Div<Output = T>,
        {
        let dim:usize=k_mesh.len();
        let mut kvec=Array2::<T>::zeros((0,dim));
        if r0==0{
            for i in 0..(k_mesh[[r0]]){
               let mut usek=Array1::<T>::zeros(dim);
               usek[[r0]]=T::from(i)/T::from(k_mesh[[r0]]);
               let k0:Array2::<T>=gen_kmesh_arr(&k_mesh,r0+1,usek);
               kvec.append(Axis(0),k0.view()).unwrap();
            }
            return kvec
        }else if r0<k_mesh.len()-1{
            for i in 0..(k_mesh[[r0]]){
               let mut kk=usek.clone();
               kk[[r0]]=T::from(i)/T::from(k_mesh[[r0]]);
               let k0:Array2::<T>=gen_kmesh_arr(&k_mesh,r0+1,kk);
               kvec.append(Axis(0),k0.view()).unwrap();
            }
            return kvec
        }else{
            for i in 0..(k_mesh[[r0]]){
               usek[[r0]]=T::from(i)/T::from(k_mesh[[r0]]);
               kvec.push_row(usek.view()).unwrap();
            }
            return kvec
        }
    }
    let mut usek=Array1::<T>::zeros(dim);
    gen_kmesh_arr(&k_mesh,0,usek)
}
#[allow(non_snake_case)]
#[inline(always)]
pub fn gen_krange<T>(k_mesh:&Array1::<usize>)->Array3::<T>
        where T:usefloat+ std::ops::Div<Output = T>,
{
    let dim_r=k_mesh.len();
    let mut k_range=Array3::<T>::zeros((0,dim_r,2));
    match dim_r{
        1=>{
            for i in 0..k_mesh[[0]]{
                let mut k=Array2::<T>::zeros((dim_r,2));
                k[[0,0]]=T::from(i)/T::from(k_mesh[[0]]);
                k[[0,1]]=T::from(i+1)/T::from(k_mesh[[0]]);
                k_range.push(Axis(0),k.view()).unwrap();
            }
        },
        2=>{
            for i in 0..k_mesh[[0]]{
                for j in 0..k_mesh[[1]]{
                    let mut k=Array2::<T>::zeros((dim_r,2));
                    k[[0,0]]=T::from(i)/T::from(k_mesh[[0]]);
                    k[[0,1]]=T::from(i+1)/T::from(k_mesh[[0]]);
                    k[[1,0]]=T::from(j)/T::from(k_mesh[[1]]);
                    k[[1,1]]=T::from(j+1)/T::from(k_mesh[[1]]);
                    k_range.push(Axis(0),k.view()).unwrap();
                }
            }
        },
        3=>{
            for i in 0..k_mesh[[0]]{
                for j in 0..k_mesh[[1]]{
                    for ks in 0..k_mesh[[2]]{
                        let mut k=Array2::<T>::zeros((dim_r,2));
                        k[[0,0]]=T::from(i)/T::from(k_mesh[[0]]);
                        k[[0,1]]=T::from(i+1)/T::from(k_mesh[[0]]);
                        k[[1,0]]=T::from(j)/T::from(k_mesh[[1]]);
                        k[[1,1]]=T::from(j+1)/T::from(k_mesh[[1]]);
                        k[[2,0]]=T::from(ks)/T::from(k_mesh[[2]]);
                        k[[2,1]]=T::from(ks+1)/T::from(k_mesh[[2]]);
                        k_range.push(Axis(0),k.view()).unwrap();
                    }
                }
            }
        },
        _=>{
            panic!("Wrong, the dim should be 1,2 or 3, but you give {}",dim_r);
        }
    };
    k_range
}

#[allow(non_snake_case)]
#[inline(always)]
pub fn comm<A,B,T>(A: &ArrayBase<A, Ix2>, B: &ArrayBase<B, Ix2>) -> Array2<T>
where  
    A: Data<Elem = T>,
    B: Data<Elem = T>,
    T: LinalgScalar, // 约束条件：T 必须实现 LinalgScalar trait
{
    //! 做 $\\\{A,B\\\}$ 对易操作
    A.dot(B)-B.dot(A)
}
#[allow(non_snake_case)]
#[inline(always)]
pub fn anti_comm<A,B,T>(A: &ArrayBase<A, Ix2>, B: &ArrayBase<B, Ix2>) -> Array2<T>
where  
    A: Data<Elem = T>,
    B: Data<Elem = T>,
    T: LinalgScalar, // 约束条件：T 必须实现 LinalgScalar trait
{
    //! 做 $\\\{A,B\\\}$ 反对易操作
    A.dot(B)+B.dot(A)
}
pub fn draw_heatmap<A:Data<Elem=f64>>(data: &ArrayBase<A, Ix2>,name:&str) {
    //!这个函数是用来画热图的, 给定一个二维矩阵, 会输出一个像素图片
    use gnuplot::{Figure, AxesCommon, AutoOption::Fix,HOT,RAINBOW};
    let mut fg = Figure::new();
    let (height,width):(usize,usize) = (data.shape()[0],data.shape()[1]);
    let mut heatmap_data = vec![];

    for j in 0..width {
        for i in 0..height {
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


pub fn  write_txt<T:usefloat>(data:&Array2<T>,output:&str)-> std::io::Result<()>
{
    use std::fs::File;
    use std::io::Write;
    let mut file=File::create(output).expect("Unable to BAND.dat");
    let n=data.len_of(Axis(0));
    let s=data.len_of(Axis(1));
    let mut s0=String::new();
    for i in 0..n{
        for j in 0..s{
            if data[[i,j]]>=T::from(0.0){
                s0.push_str("     ");
            }else{
                s0.push_str("    ");
            }
            let aa= format!("{:.6}", data[[i,j]]);
            s0.push_str(&aa);
        }
        s0.push_str("\n");
    }
    writeln!(file,"{}",s0)?;
    Ok(())
}

pub fn  write_txt_1<T:usefloat>(data:&Array1<T>,output:&str)-> std::io::Result<()>
{
    use std::fs::File;
    use std::io::Write;
    let mut file=File::create(output).expect("Unable to BAND.dat");
    let n=data.len_of(Axis(0));
    let mut s0=String::new();
    for i in 0..n{
        if data[[i]]>=T::from(0.0){
            s0.push_str(" ");
        }
        let aa= format!("{:.6}\n", data[[i]]);
        s0.push_str(&aa);
    }
    writeln!(file,"{}",s0)?;
    Ok(())
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
///    let mut model=Model::tb_model(dim_r,lat,orb,false,None);
///    model.set_onsite(&arr1(&[delta,-delta]),0);
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


#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use num_complex::Complex;
    use super::*;
    use ndarray::prelude::*;
    use ndarray::*;
    use std::time::{Duration, Instant};
    use gnuplot::{Major,Figure,Color,PointSymbol,AutoOption,Fix,AxesCommon,LineStyle,Solid,Font,TextOffset,Rotate};


    fn  write_txt(data:Array2<f64>,output:&str)-> std::io::Result<()>{
        use std::fs::File;
        use std::io::Write;
        let mut file=File::create(output).expect("Unable to BAND.dat");
        let n=data.len_of(Axis(0));
        let s=data.len_of(Axis(1));
        let mut s0=String::new();
        for i in 0..n{
            for j in 0..s{
                if data[[i,j]]>=0.0{
                    s0.push_str("     ");
                }else{
                    s0.push_str("    ");
                }
                let aa= format!("{:.6}", data[[i,j]]);
                s0.push_str(&aa);
            }
            s0.push_str("\n");
        }
        writeln!(file,"{}",s0)?;
        Ok(())
    }

    fn  write_txt_1(data:Array1<f64>,output:&str)-> std::io::Result<()>{
        use std::fs::File;
        use std::io::Write;
        let mut file=File::create(output).expect("Unable to BAND.dat");
        let n=data.len_of(Axis(0));
        let mut s0=String::new();
        for i in 0..n{
            if data[[i]]>=0.0{
                s0.push_str(" ");
            }
            let aa= format!("{:.6}\n", data[[i]]);
            s0.push_str(&aa);
        }
        writeln!(file,"{}",s0)?;
        Ok(())
    }
    #[test]
    fn test_gen_v(){

        //判断两个Array1<f64> 是否足够接近
        fn are_arrays_close(a: &Array1<f64>, b: &Array1<f64>, tolerance: f64) -> bool {
            a.iter().zip(b.iter()).all(|(&x, &y)| (x - y).abs() < tolerance)
        }

        //判断两个Array2<Compelx<f64>> 是否足够接近
        fn are_complex_arrays_close(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>, tolerance: f64) -> bool {
            a.iter().zip(b.iter()).all(|(&x, &y)| {
                (x.re - y.re).abs() < tolerance && (x.im - y.im).abs() < tolerance
            })
        }
        let li:Complex<f64>=1.0*Complex::i();
        let t=1.0+0.0*li;
        let t2=-0.0+0.0*li;
        let delta=0.0;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_onsite(&arr1(&[-delta,delta]),0);
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
        assert_eq!(model.solve_band_onek(&array![0.0,0.0]),array![-3.0,3.0]);
        let result=model.solve_band_onek(&array![1.0/3.0,2.0/3.0]);
        assert!(are_arrays_close(&result,&array![0.0,0.0],1e-5),"wrong!, the solve_band_onek get wrong result! please check it!");
        let result=model.gen_v(&array![1.0/3.0,1.0/3.0]);
        let resulty=array![[0.0*li,-0.4698463103929542-0.17101007166283436*li],[-0.4698463103929542+0.17101007166283436*li,0.0*li]];
        let resultx=array![[0.0*li,-0.8137976813493737-0.2961981327260237*li],[-0.8137976813493737+0.2961981327260237*li,0.0*li]];
        assert!(are_complex_arrays_close(&result.slice(s![0,..,..]).to_owned(),&resultx,1e-8),"Wrong! the gen_v is get wrong results! please check it!");
        assert!(are_complex_arrays_close(&result.slice(s![1,..,..]).to_owned(),&resulty,1e-8),"Wrong! the gen_v is get wrong results! please check it!");

        let kvec=array![1.0/3.0,1.0/3.0];
        let (band,evec)=model.solve_onek(&kvec);
        let ham=model.gen_ham(&kvec);
        //let evec=evec.reversed_axes();
        //let evec_conj=conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>,OwnedRepr<Complex<f64>>>(&evec);
        let evec_conj=evec.map(|x| x.conj());
        let evec=evec.t();
        let ham=ham.dot(&evec);
        let ham=evec_conj.dot(&ham);
        let new_band=ham.diag().map(|x| x.re);
        assert!(are_arrays_close(&new_band,&band,1e-5),"wrong!, the solve_onek get wrong result! please check it!");
    }
    #[test]
    fn conductivity_test(){
        //这个是用 Haldan 模型来测试
        let li:Complex<f64>=1.0*Complex::i();
        let t=-1.0+0.0*li;
        let t2=-1.0+0.0*li;
        let delta=0.7;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_onsite(&arr1(&[-delta,delta]),0);
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
        let k_vec=array![1.0/3.0,2.0/3.0];
        let dir_1=array![1.0,0.0];
        let dir_2=array![0.0,1.0];
        let mu=0.0;
        let T=0.0;
        let og=0.0;
        let spin=0;
        let eta=1e-3;
        let result1=model.berry_curvature_onek(&k_vec,&dir_1,&dir_2,mu,T,og,spin,eta)*(2.0*PI).powi(2)/3_f64.sqrt()/PI;

        let mut k_list=Array2::zeros((9,2));
        let dk=0.0001;
        k_list.row_mut(0).assign(&(&k_vec+dk*&dir_1));
        k_list.row_mut(1).assign(&(&k_vec+dk*&dir_1+dk*&dir_2));
        k_list.row_mut(2).assign(&(&k_vec+dk*&dir_2));
        k_list.row_mut(3).assign(&(&k_vec-dk*&dir_1+dk*&dir_2));
        k_list.row_mut(4).assign(&(&k_vec-dk*&dir_1));
        k_list.row_mut(5).assign(&(&k_vec-dk*&dir_1-dk*&dir_2));
        k_list.row_mut(6).assign(&(&k_vec-dk*&dir_2));
        k_list.row_mut(7).assign(&(&k_vec+dk*&dir_1-dk*&dir_2));
        k_list.row_mut(8).assign(&(&k_vec+dk*&dir_1));
        let result2=model.berry_loop(&k_list,&vec![0]);
        let result2=result2[[0]]/(dk.powi(2))/4.0/(2.0*PI);
        assert!((result2-result1).abs()<1e-4,"Wrong!, the berry_curvature or berry_flux mut be false");

        
    }
    #[test]
    fn gen_v_speed_test(){
        println!("开始测试各个函数的运行速度, 用次近邻的石墨烯模型");
        let li:Complex<f64>=1.0*Complex::i();
        let t=2.0+0.0*li;
        let t2=-1.0+0.0*li;
        let delta=0.7;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_onsite(&arr1(&[-delta,delta]),0);
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
        println!("{:?}",model.atom_list());
        let U=array![[3.0,0.0],[0.0,3.0]];
        let model=model.make_supercell(&U);

        let nk=101;
        let k_mesh=array![nk,nk];
        let kvec=gen_kmesh(&k_mesh);

        {
        println!("开始计算 gen_v 的耗时速度, 为了平均, 我们单线程求解gen_v");
        let start = Instant::now();   // 开始计时
        let A:Vec<_>=kvec.outer_iter().into_par_iter().map(|x| model.gen_v(&x.to_owned())).collect();
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("run gen_v {} times took {} seconds", kvec.nrows(), duration.as_secs_f64());   // 输出执行时间

        }

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
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_onsite(&arr1(&[-delta,delta]),0);
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
        let nk:usize=100;
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
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,mu,T,og,spin,eta);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("quantom_Hall_effect={}",conductivity*(2.0*PI));
        assert!((conductivity*(2.0*PI)-1.0).abs()<1e-3,"Wrong!, the Hall conductivity is wrong!");
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间

        
        let mu=Array1::linspace(-2.0,2.0,1001);
        let start = Instant::now();   // 开始计时
        let conductivity_mu=model.Hall_conductivity_mu(&kmesh,&dir_1,&dir_2,&mu,T,og,spin,eta);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("quantom_Hall_effect={}",conductivity_mu[[1]]*(2.0*PI));
        assert!((conductivity_mu[[500]]-conductivity).abs()<1e-3,"Wrong!, the Hall conductivity is wrong!, Hall_mu's result is {}, but Hall conductivity is {}",conductivity_mu[[1000]],conductivity);
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,-2.0,T,og,spin,eta);
        assert!((conductivity_mu[[0]]-conductivity).abs()<1e-3,"Wrong!, the Hall conductivity is wrong!, Hall_mu's result is {}, but Hall conductivity is {}",conductivity_mu[[0]],conductivity);
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,2.0,T,og,spin,eta);
        assert!((conductivity_mu[[1000]]-conductivity).abs()<1e-3,"Wrong!, the Hall conductivity is wrong!, Hall_mu's result is {}, but Hall conductivity is {}",conductivity_mu[[1000]],conductivity);
        //开始绘图
        let mut fg = Figure::new();
        let x:Vec<f64>=mu.to_vec();
        let axes=fg.axes2d();
        let y:Vec<f64>=(conductivity_mu*2.0*PI).to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/hall_mu.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


        let mu=0.0;
        let nk:usize=1000;
        let kmesh=arr1(&[nk,nk]);
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,mu,T,og,spin,eta,0.01,0.0001);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("霍尔电导率{}",conductivity*(2.0*PI));
        assert!((conductivity*(2.0*PI)-1.0).abs()<1e-5,"Wrong!, the Hall conductivity is wrong!");
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
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
        let f:Array1<f64>=1.0/((beta*&mu).mapv(f64::exp)+1.0);
        let par_f=beta*&f*(1.0-&f);
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


        //画一下表面态
        let nk=1001;
        let green=surf_Green::from_Model(&model,0,1e-3,None);
        let E_min=-3.0;
        let E_max=3.0;
        let E_n=nk;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        green.show_surf_state("tests/Haldan/surf",&path,&label,nk,E_min,E_max,E_n,0);


        //开始算非线性霍尔电导

        /*
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
        */
        //-----算一下wilson loop 的结果-----------------------
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let occ=vec![0];
        let wcc=model.wannier_centre(&occ,&array![0.0,0.0],&dir_1,&dir_2,101,101);
        let nocc=occ.len();

        let mut fg = Figure::new();
        let x:Vec<f64>=Array1::<f64>::linspace(0.0,1.0,101).to_vec();
        let axes=fg.axes2d();
        for i in 0..nocc{
            let y:Vec<f64>=wcc.row(i).to_vec();
            axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
        }
        let axes=axes.set_x_range(Fix(0.0), Fix(1.0));
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/Haldan/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }
    #[test]
    fn graphene(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.0+0.0*li;
        let t3=0.0+0.0*li;
        let delta=0.0;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[3.0_f64.sqrt(),-1.0],[3.0_f64.sqrt(),1.0]]);
        let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_onsite(&arr1(&[delta,-delta]),0);
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

        // 开始计算两个本征态
        let k1=array![1.0/3.0-0.002,2.0/3.0];
        let k2=array![1.0/3.0+0.001,2.0/3.0];
        let (eval1,evec1)=model.solve_onek(&k1);
        let (eval2,evec2)=model.solve_onek(&k2);
        let evec1=evec1.reversed_axes();
        let evec2=evec2.mapv(|x| x.conj());
        println!("{},{}",eval1,eval2);
        println!("{}",evec2.dot(&evec1).mapv(|x| x.norm().round()));

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
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,mu,T,og,spin,eta);
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
        let nk:usize=201;
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
        let t=-1.0;
        let delta=0.0;
        let alter=0.0+0.0*li;
        let soc=0.06*t;
        let rashba=0.0*t;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
        let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,true,None);
        model.set_onsite(&arr1(&[delta,-delta]),0);
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
        let nk:usize=101;
        println!("aaa");
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
        let nk=1001;
        let green=surf_Green::from_Model(&model,0,1e-3,None);
        let E_min=-1.0;
        let E_max=1.0;
        let E_n=nk;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        green.show_surf_state("tests/kane",&path,&label,nk,E_min,E_max,E_n,0);


        //-----算一下wilson loop 的结果-----------------------
        let n=101;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let occ=vec![0,1];
        let wcc=model.wannier_centre(&occ,&array![0.0,0.0],&dir_1,&dir_2,n,n);
        let nocc=occ.len();
        let mut fg = Figure::new();
        let x:Vec<f64>=Array1::<f64>::linspace(0.0,1.0,n).to_vec();
        let axes=fg.axes2d();
        for j in -1..2{
            for i in 0..nocc{
                let a=wcc.row(i).to_owned()+(j as f64)*2.0*PI;
                let y:Vec<f64>=a.to_vec();
                axes.points(&x, &y, &[Color("black"),gnuplot::PointSymbol('O')]);
            }
        }
        let axes=axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes=axes.set_y_range(Fix(0.0), Fix(2.0*PI));
        let show_ticks=vec![Major(0.0,Fix("0")),Major(0.5,Fix("π")),Major(1.0,Fix("2π"))];
        axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",32.0)]);
        let show_ticks=vec![Major(0.0,Fix("0")),Major(PI,Fix("π")),Major(2.0*PI,Fix("2π"))];
        axes.set_y_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",32.0)]);
        axes.set_x_label("k_x",&[Font("Times New Roman",32.0),TextOffset(0.0,-0.5)]);
        axes.set_y_label("WCC",&[Font("Times New Roman",32.0),Rotate(90.0),TextOffset(-1.0,0.0)]);
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/kane/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


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
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,mu,T,og,spin,eta);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间
        let nk:usize=31;
        let kmesh=arr1(&[nk,nk]);
        let start = Instant::now();   // 开始计时
        let conductivity=model.Hall_conductivity_adapted(&kmesh,&dir_1,&dir_2,mu,T,og,spin,eta,0.01,0.01);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}",conductivity/(2.0*PI));
        println!("function_a took {} seconds", duration.as_secs_f64());   // 输出执行时间

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
        let nk:usize=100;
        let kmesh=arr1(&[nk,nk]);
        let kvec=gen_kmesh(&kmesh);
        //let kvec=kvec-0.5;
        let kvec=kvec*2.0;
        let kvec=model.lat.dot(&(kvec.reversed_axes()));
        let kvec=kvec.reversed_axes();
        let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,0.0,0.0,1,1e-3);
        let data=berry_curv.into_shape((nk,nk)).unwrap();
        draw_heatmap(&(-data).map(|x| (x+1.0).log(10.0)),"./tests/kane/berry_curvature_distribution.pdf");

        //开始考虑磁场, 加入磁性
        let B=0.1+0.0*li;
        let tha=0.0/180.0*PI;

        model.add_hop(B*tha.cos(),0,0,&array![0,0],1);
        model.add_hop(B*tha.cos(),1,1,&array![0,0],1);
        model.add_hop(B*tha.sin(),0,0,&array![0,0],2);
        model.add_hop(B*tha.sin(),1,1,&array![0,0],2);
        //考虑添加onsite 项破坏空间反演和mirror

        let green=surf_Green::from_Model(&model,0,1e-3,None);
        let E_min=-1.0;
        let E_max=1.0;
        let E_n=nk;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        green.show_surf_state("tests/kane/magnetic",&path,&label,nk,E_min,E_max,E_n,0);

        //-----算一下wilson loop 的结果-----------------------
        let n=101;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let occ=vec![0,1];
        let wcc=model.wannier_centre(&occ,&array![0.0,0.0],&dir_1,&dir_2,n,n);
        let nocc=occ.len();
        let mut fg = Figure::new();
        let x:Vec<f64>=Array1::<f64>::linspace(0.0,1.0,n).to_vec();
        let axes=fg.axes2d();
        for j in -1..2{
            for i in 0..nocc{
                let a=wcc.row(i).to_owned()+(j as f64)*2.0*PI;
                let y:Vec<f64>=a.to_vec();
                axes.points(&x, &y, &[Color("black"),gnuplot::PointSymbol('O')]);
            }
        }
        let axes=axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes=axes.set_y_range(Fix(0.0), Fix(2.0*PI));
        let show_ticks=vec![Major(0.0,Fix("0")),Major(0.5,Fix("π")),Major(1.0,Fix("2π"))];
        axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",32.0)]);
        let show_ticks=vec![Major(0.0,Fix("0")),Major(PI,Fix("π")),Major(2.0*PI,Fix("2π"))];
        axes.set_y_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",32.0)]);
        axes.set_x_label("k_x",&[Font("Times New Roman",32.0),TextOffset(0.0,-0.5)]);
        axes.set_y_label("WCC",&[Font("Times New Roman",32.0),Rotate(90.0),TextOffset(-1.0,0.0)]);
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/kane/magnetic/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


        //开始计算角态
        let model=model.make_supercell(&array![[0.0,-1.0],[1.0,0.0]]);
        let num=10;
        /*
        let model_1=model.cut_piece(num,0);
        let new_model=model_1.cut_piece(num,1);
        */
        let new_model=model.cut_dot(num,6,None);
        let mut s=0;
        let start = Instant::now();
        let (band,evec)=new_model.solve_range_onek(&arr1(&[0.0,0.0]),(-0.3,0.3),1e-5);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("solve_band_all took {} seconds", duration.as_secs_f64());   // 输出执行时间
        let nresults=band.len();
        let show_evec=evec.to_owned().map(|x| x.norm_sqr());
        let mut size=Array2::<f64>::zeros((new_model.nsta(),new_model.natom()));
        let norb=new_model.norb();
        for i in 0..nresults{
            let mut s=0;
            for j in 0..new_model.natom(){
                for k in 0..new_model.atoms[j].norb(){
                    size[[i,j]]+=show_evec[[i,s]]+show_evec[[i,s+new_model.norb()]];
                    s+=1;
                }
            }
        }

        let show_str=new_model.atom_position().dot(&model.lat);
        let show_str=show_str.slice(s![..,0..2]).to_owned();
        let show_size=size.row(new_model.norb()).to_owned();
        use std::fs::create_dir_all;
        create_dir_all("tests/kane/magnetic").expect("can't creat the file");
        write_txt_1(band,"tests/kane/magnetic/band.txt");
        write_txt(size,"tests/kane/magnetic/evec.txt");
        write_txt(show_str,"tests/kane/magnetic/structure.txt");
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
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.set_onsite(&arr1(&[delta,-delta]),0);
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
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
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


        let green=surf_Green::from_Model(&super_model,0,1e-3,None);
        let E_min=-2.0;
        let E_max=4.0;
        let E_n=nk;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        green.show_surf_state("tests/kagome_zig",&path,&label,nk,E_min,E_max,E_n,0);

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
        let t1=0.1+0.0*li;
        let t2=1.0+0.0*li;
        let dim_r:usize=1;
        let norb:usize=2;
        let lat=arr2(&[[1.0]]);
        let orb=arr2(&[[0.0],[0.5],[0.0],[0.5]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.add_hop(t1,0,1,&array![0],0);
        model.add_hop(t2,0,1,&array![-1],0);
        model.add_hop(t1,2,3,&array![0],0);
        model.add_hop(t2,2,3,&array![-1],0);
        let t_hop_1=0.0+0.0*li;
        let t_hop_2=0.0+0.0*li;
        model.add_hop(t_hop_1,0,2,&array![0],0);
        model.add_hop(t_hop_1,1,3,&array![0],0);
        /*
        model.add_hop(t_hop_1,0,3,array![0],0);
        model.add_hop(t_hop_1,1,2,array![0],0);
        */

        let nk:usize=1001;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","M","G"];
        model.show_band(&path,&label,nk,"tests/SSH/");
        let mut super_model=model.cut_piece(5,0);

        let (band,evec)=super_model.solve_onek(&array![0.0]);
        println!("{}",band);
    }
    #[test]
    fn unfold_test(){
        use std::fs::create_dir_all;
        let li:Complex<f64>=1.0*Complex::i();
        let t1=1.0+0.0*li;
        let t2=0.1+0.0*li;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[3.0_f64.sqrt(),-1.0],[3.0_f64.sqrt(),1.0]]);
        let orb=arr2(&[[0.,0.],[1.0/3.0,0.0],[0.0,1.0/3.0]]);
        let mut model=Model::tb_model(dim_r,lat,orb,true,None);
        //最近邻hopping
        model.add_hop(t1,0,1,&array![0,0],0);
        model.add_hop(t1,2,0,&array![0,0],0);
        model.add_hop(t1,1,2,&array![0,0],0);
        model.add_hop(t1,0,2,&array![0,-1],0);
        model.add_hop(t1,0,1,&array![-1,0],0);
        model.add_hop(t1,2,1,&array![-1,1],0);

        let nk:usize=1001;
        let path=array![[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.],[0.0,0.0]];
        let label=vec!["G","K","M","G"];
        let (kvec,kdist,knode)=model.k_path(&path,nk);
        let U=array![[1.0,1.0],[-5.0,4.0]];

        let start = Instant::now();   // 开始计时
        let super_model=model.make_supercell(&U);
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("make_supercell took {} seconds", duration.as_secs_f64());   // 输出执行时间
        let A=super_model.unfold(&U,&path,nk,-3.0,5.0,nk,1e-2,1e-3);
        let name="./tests/unfold_test/";
        create_dir_all(&name).expect("can't creat the file");
        draw_heatmap(&A.reversed_axes(),"./tests/unfold_test/unfold_band.pdf");
        super_model.show_band(&path,&label,nk,name);
    }
    #[test]
    fn BBH_model(){
        let li:Complex<f64>=1.0*Complex::i();
        let t1=0.1+0.0*li;
        let t2=1.0+0.0*li;
        let i0=-1.0;
        let dim_r:usize=2;
        let norb:usize=2;
        let lat=arr2(&[[1.0,0.0],[0.0,1.0]]);
        let orb=arr2(&[[0.0,0.0],[0.5,0.0],[0.5,0.5],[0.0,0.5]]);
        let mut model=Model::tb_model(dim_r,lat,orb,false,None);
        model.add_hop(t1,0,1,&array![0,0],0);
        model.add_hop(t1,1,2,&array![0,0],0);
        model.add_hop(t1,2,3,&array![0,0],0);
        model.add_hop(i0*t1,3,0,&array![0,0],0);
        model.add_hop(t2,0,1,&array![-1,0],0);
        model.add_hop(i0*t2,0,3,&array![0,-1],0);
        model.add_hop(t2,2,3,&array![1,0],0);
        model.add_hop(t2,2,1,&array![0,1],0);
        let nk:usize=1001;
        let path=[[0.0,0.0],[0.5,0.0],[0.5,0.5],[0.0,0.0]];
        let path=arr2(&path);
        let label=vec!["G","X","M","G"];
        model.show_band(&path,&label,nk,"tests/BBH/");
        model.output_hr("tests/BBH/","wannier90");


        //算一下wilson loop
        let n=301;
        let dir_1=arr1(&[1.0,0.0]);
        let dir_2=arr1(&[0.0,1.0]);
        let occ=vec![0,1];
        let wcc=model.wannier_centre(&occ,&array![0.0,0.0],&dir_1,&dir_2,n,n);
        let nocc=occ.len();
        let mut fg = Figure::new();
        let x:Vec<f64>=Array1::<f64>::linspace(0.0,1.0,n).to_vec();
        let axes=fg.axes2d();
        for j in -1..2{
            for i in 0..nocc{
                let a=wcc.row(i).to_owned()+(j as f64)*2.0*PI;
                let y:Vec<f64>=a.to_vec();
                axes.points(&x, &y, &[Color("black"),gnuplot::PointSymbol('O')]);
            }
        }
        let axes=axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes=axes.set_y_range(Fix(0.0), Fix(2.0*PI));
        let show_ticks=vec![Major(0.0,Fix("0")),Major(0.5,Fix("π")),Major(1.0,Fix("2π"))];
        axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
        let show_ticks=vec![Major(0.0,Fix("0")),Major(PI,Fix("π")),Major(2.0*PI,Fix("2π"))];
        axes.set_y_ticks_custom(show_ticks.into_iter(),&[],&[]);
        let mut pdf_name=String::new();
        pdf_name.push_str("tests/BBH/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        //算一下边界态
        let green=surf_Green::from_Model(&model,0,1e-3,None);
        let E_min=-2.0;
        let E_max=2.0;
        let E_n=nk;
        let path=[[0.0],[0.5],[1.0]];
        let path=arr2(&path);
        let label=vec!["G","X","G"];
        green.show_surf_state("tests/BBH",&path,&label,nk,E_min,E_max,E_n,0);


        //算一下corner state
        let num=10;
        let model_1=model.cut_piece(num,0);
        let new_model=model_1.cut_piece(2*num,1);
        let mut s=0;
        let start = Instant::now();
        let (band,evec)=new_model.solve_onek(&arr1(&[0.0,0.0]));
        println!("band shape is {:?}, evec shape is {:?}",band.shape(),evec.shape());
        let end = Instant::now();    // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("solve_band_all took {} seconds", duration.as_secs_f64());   // 输出执行时间
        let nresults=band.len();
        let show_evec=evec.to_owned().map(|x| x.norm_sqr());
        let norb=new_model.norb();
        let size=show_evec;
        let show_str=new_model.atom_position().dot(&model.lat);
        use std::fs::create_dir_all;
        create_dir_all("tests/BBH/corner").expect("can't creat the file");
        write_txt_1(band,"tests/BBH/corner/band.txt");
        write_txt(size,"tests/BBH/corner/evec.txt");
        write_txt(show_str,"tests/BBH/corner/structure.txt");
    }
}
