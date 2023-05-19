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
    let t=1.0;
    let v=1.0;
    let ap=1.0;
    let eta=-1.0;
    let m=0.2;
    let path=array![[-0.5,0.0],[0.0,0.0],[0.5,0.0]];
    let label=vec!["M","G","M"];
    let nk=1001;
    let model=gen_model(t,v,ap,eta,m);
    model.show_band(&path,&label,nk,"./");


    //开始计算非线性霍尔电导
    let dir_1=arr1(&[0.0,1.0]);
    let dir_2=arr1(&[1.0,0.0]);
    let dir_3=arr1(&[1.0,0.0]);
    let nk:usize=3000;
    let kmesh=arr1(&[nk,nk]);
    let E_min=-0.5;
    let E_max=0.5;
    let E_n=2000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let T=5.0;
    let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,0,1e-5);
    let sigma=sigma/(2.0*PI).powi(2);

    //开始绘制非线性电导
    let mut fg = Figure::new();
    let x:Vec<f64>=mu.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=sigma.to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    axes.set_y_range(Fix(-0.3),Fix(0.3));
    axes.set_x_range(Fix(E_min),Fix(E_max));
    let mut show_ticks=Vec::<String>::new();
    let mut pdf_name=String::new();
    pdf_name.push_str("nonlinear_ex.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();

/*
    let nk:usize=500;
    let kmesh=arr1(&[nk,nk]);
    let m=0.01;
    let T=100.0;
    let t_n=100;
    let E_min=-0.5;
    let E_max=0.5;
    let E_n=100;
    let t=Array1::linspace(0.0,2.0*v,t_n);
    let mut omega=Array1::<f64>::zeros(E_n);
    for (i,t0) in t.iter().enumerate(){
        let model=gen_model(*t0,v,ap,eta,m);
        let sigma=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,0,1e-5);
        omega[[i]]=sigma.iter().fold(f64::NAN, |a,&b| a.min(b))*model.lat.det().unwrap();
    }
    let mut fg = Figure::new();
    let x:Vec<f64>=t.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=omega.to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    axes.set_y_range(Fix(-8.0),Fix(0.0));
    axes.set_x_range(Fix(0.0),Fix(2.0*v));
    let mut show_ticks=Vec::<String>::new();
    let mut pdf_name=String::new();
    pdf_name.push_str("nonlinear_t.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
*/
}


fn gen_model(t:f64,v:f64,ap:f64,eta:f64,m:f64)-> Model{
    let li:Complex<f64>=1.0*Complex::i();
    let dim_r:usize=2;
    let norb:usize=2;
    let lat=arr2(&[[1.0,0.0],[0.0,1.0]]);
    let orb=arr2(&[[0.0,0.0],[0.0,0.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,false,None,None);
    model.set_onsite(array![m/2.0-4.0*ap,-m/2.0+4.0*ap],0);
    let t=Complex::new(t,0.0);
    let v=Complex::new(v,0.0);
    let ap=Complex::new(ap,0.0);
    let eta=Complex::new(eta,0.0);
    model.add_hop(li*t/2.0,0,0,&array![1,0],0);
    model.add_hop(ap,0,0,&array![1,0],0);
    model.add_hop(ap,0,0,&array![0,1],0);
    model.add_hop(li*t/2.0,1,1,&array![1,0],0);
    model.add_hop(-ap,1,1,&array![1,0],0);
    model.add_hop(-ap,1,1,&array![0,1],0);
    model.add_hop(li*v/2.0,0,1,&array![0,1],0);
    model.add_hop(-li*v/2.0,0,1,&array![0,-1],0);
    model.add_hop(eta*v/2.0,0,1,&array![1,0],0);
    model.add_hop(-eta*v/2.0,0,1,&array![-1,0],0);
    model
}
