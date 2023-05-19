use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon};
fn main(){
    //!这个是 bismuthene的模型, 只考虑px, py
    //!轨道的基函数为 $\\{s^A,p_x^A,p_y^A,p_z^A,S^B,p_x^B,p_y^B,p_z^B,s_H^A,s_H^B\\}$
    let li:Complex<f64>=1.0*Complex::i();
    let t1=1.0+0.0*li;
    let th=0.2+0.0*li;
    let delta=0.0;
    let dim_r:usize=3;
    let norb:usize=2;
    let lat=arr2(&[[3.0_f64.sqrt(),-1.0,0.0],[3.0_f64.sqrt(),1.0,0.0],[0.0,0.0,2.0]]);
    let orb=arr2(&[[0.0,0.0,0.0],[1.0/3.0,1.0/3.0,0.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    model.set_onsite(arr1(&[delta,-delta]),0);
    model.add_hop(t1,0,1,&array![0,0,0],1);
    model.add_hop(t1,0,1,&array![-1,0,0],1);
    model.add_hop(t1,0,1,&array![0,-1,0],1);
    model.add_hop(th,0,0,&array![0,0,1],0);
    model.add_hop(th,1,1,&array![0,0,1],0);
    let nk:usize=1001;
    let path=[[0.0,0.0,0.0],[2.0/3.0,1.0/3.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]];
    let path=arr2(&path);
    let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
    let (eval,evec)=model.solve_all_parallel(&k_vec);
    let label=vec!["G","K","M","G"];
    model.show_band(&path,&label,nk,"./");



    //开始计算非线性霍尔电导
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let dir_3=arr1(&[0.0,0.0,1.0]);
    let nk:usize=210;
    let kmesh=arr1(&[nk,nk,nk]);
    let E_min=-3.0;
    let E_max=3.0;
    let E_n=1000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let T=30.0;
    let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,1,1e-5);

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
    pdf_name.push_str("nonlinear_ex.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


}
