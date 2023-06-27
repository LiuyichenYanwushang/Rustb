use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon};
fn main(){
    //!来自 PHYSICAL REVIEW X 12, 031042 (2022) 的 RuO2 模型
    //! $$\\mathcal{H}=t(\cos k_x+\cos k_y)\pm \Delta(\cos k_x-\cos k_y)\pm J\sigma_z\tau_z$$
    let li:Complex<f64>=1.0*Complex::i();
    let t1=0.1+0.0*li;
    let J=0.5+0.0*li;
    let delta=0.075+0.0*li;
    let dim_r:usize=2;
    let norb:usize=2;
    let a0=0.5;
    let lat=arr2(&[[1.0,0.0],[0.0,1.0]])*a0;
    let orb=arr2(&[[0.0,0.0],[0.0,0.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    model.add_hop(t1,0,0,&array![1,0],0);
    model.add_hop(t1,0,0,&array![0,1],0);
    model.add_hop(t1,1,1,&array![1,0],0);
    model.add_hop(t1,1,1,&array![0,1],0);

    model.add_hop(delta,0,0,&array![1,0],0);
    model.add_hop(-delta,0,0,&array![0,1],0);
    model.add_hop(-delta,1,1,&array![1,0],0);
    model.add_hop(delta,1,1,&array![0,1],0);

    model.add_hop(J,0,0,&array![0,0],3);
    model.add_hop(-J,1,1,&array![0,0],3);

    let nk:usize=1001;
    let path=array![[0.0,0.0],[0.5,0.0],[0.5,0.5],[0.0,0.5],[0.0,0.0]];
    let label=vec!["G","X","M","Y","G"];
    model.show_band(&path,&label,nk,"examples/RuO2");



    //画一下贝利曲率的分布
    let dir_1=arr1(&[1.0,0.0]);
    let dir_2=arr1(&[0.0,1.0]);
    let dir_3=arr1(&[0.0,1.0]);
    let T=100.0;
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk]);
    /*
    let kvec=gen_kmesh(&kmesh);
    let lat_inv=model.lat.inv().unwrap();
    let kvec=PI*model.lat.dot(&(kvec.reversed_axes()));
    let kvec=kvec.reversed_axes();
    let (berry_curv,band)=model.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,0.0,0,1e-3);
    ///////////////////////////////////////////
    let beta=1.0/T/(8.617e-5);
    let f:Array2::<f64>=band.clone().map(|x| 1.0/((beta*x).exp()+1.0));
    let f=beta*&f*(1.0-&f);
    println!("{:?}",berry_curv.shape());
    let berry_curv=(berry_curv.clone()*f).sum_axis(Axis(1));
    let data=berry_curv.into_shape((nk,nk)).unwrap();
    draw_heatmap(data,"./examples/RuO2/nonlinear.pdf");
    */

   //画一下贝利曲率的分布
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk]);
    let kvec=gen_kmesh(&kmesh);
    let kvec=PI*model.lat.dot(&(kvec.reversed_axes()));
    //let kvec=model.lat.dot(&(kvec.reversed_axes()));
    let kvec=kvec.reversed_axes();
    let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,0.0,0.0,0,1e-3);
    let data=berry_curv.clone().into_shape((nk,nk)).unwrap();
    draw_heatmap(data.map(|x| {let a:f64=if *x >= 0.0 {(x+1.0).log(10.0)} else {-(-x+1.0).log(10.0)}; a}),"./examples/RuO2/heat_map.pdf");
    let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,0.0,0.0,0.0,0,1e-3);
    println!("{}",conductivity/(2.0*PI));
        

    let E_min=-1.0;
    let E_max=1.0;
    let E_n=2000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Extrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,og,0,1e-5);
    //开始绘制非线性电导
    let mut fg = Figure::new();
    let x:Vec<f64>=mu.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=sigma.to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    //axes.set_y_range(Fix(-10.0),Fix(10.0));
    axes.set_x_range(Fix(E_min),Fix(E_max));
    let mut show_ticks=Vec::<String>::new();
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/RuO2/nonlinear_ex.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


    let E_min=-1.0;
    let E_max=1.0;
    let E_n=2000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Intrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,0);
    //开始绘制非线性电导
    let mut fg = Figure::new();
    let x:Vec<f64>=mu.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=sigma.to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    //axes.set_y_range(Fix(-10.0),Fix(10.0));
    axes.set_x_range(Fix(E_min),Fix(E_max));
    let mut show_ticks=Vec::<String>::new();
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/RuO2/nonlinear_in.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
}
