use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
fn main(){
    //!这个是 bismuthene的模型, 只考虑px, py
    //!轨道的基函数为 $\\{s^A,p_x^A,p_y^A,p_z^A,S^B,p_x^B,p_y^B,p_z^B,s_H^A,s_H^B\\}$
    let li:Complex<f64>=1.0*Complex::i();
    let E_p=0.0;
    let V_sg=-2.0;//sigma 键
    let V_pi=-0.1;//Pi 键, 次近邻
    let V_sg_2=0.0;//sigma 键, 次近邻
    let V_pi_2=0.0;//Pi 键
    let lambda=0.0;

    let V_sg=Complex::new(V_sg,0.0);//sigma 键
    let V_pi=Complex::new(V_pi,0.0);//Pi 键, 次近邻
    let V_sg_2=Complex::new(V_sg_2,0.0);//sigma 键, 次近邻
    let V_pi_2=Complex::new(V_pi_2,0.0);//Pi 键
    let h=0.7957;
    let a=5.3;
    let c=10.0;
    let lat=array![[a,0.0,0.0],[-a/2.0,a*3.0_f64.sqrt()/2.0,0.0],[0.0,0.0,c]];
    let orb=array![[1.0/3.0,2.0/3.0,0.0],[1.0/3.0,2.0/3.0,0.0],[2.0/3.0,1.0/3.0,h/c],[2.0/3.0,1.0/3.0,h/c]];
    let atom=array![[1.0/3.0,1.0/3.0,0.0],[2.0/3.0,2.0/3.0,h/c]];
    let atom_list=vec![2,2];
    let mut model=Model::tb_model(3,lat,orb,true,Some(atom),Some(atom_list));
    let theta=(a/3.0_f64.sqrt()).atan();
    let hop0=arr2(&[[0,0,0],[0,1,0],[-1,0,0]]);//最近邻
    model.set_onsite(array![0.0,0.0,0.0,0.0],0);
    for (r,R) in hop0.axis_iter(Axis(0)).enumerate(){
        for i in 0..2{
            for j in 0..2{
                let R=R.to_owned();
                let orb_j=model.orb.row(j+2).to_owned();
                let orb_i=model.orb.row(i).to_owned();
                let theta=(orb_j-orb_i+R.clone().map(|x| *x as f64)).dot(&model.lat);
                let theta=theta.clone()/theta.norm();
                let tmp=-theta[[i]]*theta[[j]]*V_sg ;
                model.add_hop(tmp,i,j+2,&R,0);
                let mut vec_1=-theta[[i]]*theta.clone();
                let mut vec_2=-theta[[j]]*theta.clone();
                vec_1[[i]]+=1.0;
                vec_2[[j]]+=1.0;
                println!("{}",theta);
                println!("{},{}",vec_1,vec_2);
                let tmp=-vec_1.dot(&vec_2)*V_pi ;
                println!("{}",tmp);
                model.add_hop(tmp,i,j+2,&R,0);
            }
        }
    }
    let hop0=arr2(&[[1,-1,0],[0,-1,0],[-1,0,0]]);//次近邻
    for (r,R) in hop0.axis_iter(Axis(0)).enumerate(){
        for i in 0..2{
            for j in 0..2{
                let R=R.to_owned();
                let theta=R.clone().map(|x| *x as f64).dot(&model.lat);
                let theta=theta.clone()/theta.norm();
                let tmp=-theta[[i]]*theta[[j]]*V_sg_2 ;
                model.add_hop(tmp,i,j,&R,0);
                let tmp=-theta[[i]]*theta[[j]]*V_sg_2 ;
                model.add_hop(tmp,i+2,j+2,&(-R.clone()),0);

                let mut vec_1=-theta[[i]]*theta.clone();
                let mut vec_2=-theta[[j]]*theta.clone();
                vec_1[[i]]+=1.0;
                vec_2[[j]]+=1.0;
                let tmp=-vec_1.dot(&vec_2)*V_pi ;
                model.add_hop(tmp,i,j,&R,0);

                let mut vec_1=-theta[[i]]*theta.clone();
                let mut vec_2=-theta[[j]]*theta;
                vec_1[[i]]+=1.0;
                vec_2[[j]]+=1.0;
                let tmp=-vec_1.dot(&vec_2)*V_pi ;
                model.add_hop(tmp,i,j,&(-R),0);
            }
        }
    }
    //开始加上自旋轨道耦合 lambda*L.S
    let H_soc=li/4.0*kron(&array![[1.0,0.0],[0.0,-1.0]],&kron(&arr2(&[[1.0,0.0],[0.0,1.0]]),&arr2(&[[0.0,1.0],[-1.0,0.0]]))).map(|x| Complex::new(*x,0.0));
    //let H_soc:Array2::<Complex<f64>>=arr2(&[[0.0*li,li/4.0,0.0*li,0.0*li],[-li/4.0,0.0*li,0.0*li,0.0*li],[0.0*li,0.0*li,0.0*li,-li/4.0],[0.0*li,0.0*li,li/4.0,0.0*li]]);
    model.ham.slice_mut(s![0,..,..]).add_assign(&H_soc.map(|x| lambda*x));

    let path=array![[0.0,0.0,0.0],[2.0/3.0,1.0/3.0,0.0],[0.5,0.5,0.0],[1.0/3.0,2.0/3.0,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","K","M","K'","G"];
    let nk=301;
    model.show_band(&path,&label,nk,"examples/bismuthene");
}
