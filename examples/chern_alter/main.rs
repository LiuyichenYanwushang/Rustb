use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon};
fn main(){
    let li:Complex<f64>=1.0*Complex::i();
    let t=0.8+0.0*li;
    let lm_so=0.5+0.0*li;
    let J=3.0;
    let delta=2.0;
    let dim_r:usize=2;
    let a0=1.0;
    let lat=arr2(&[[1.0,0.0],[0.0,1.0]])*a0;//正方晶格
    let orb=arr2(&[[0.0,0.0],[0.5,0.5]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    model.set_onsite(arr1(&[J,-J]),3);
    //底层最近邻hopping
    model.add_hop(t+delta,0,0,&array![-1,0],0);
    model.add_hop(t-delta,0,0,&array![0,-1],0);
    model.add_hop(t-delta,1,1,&array![-1,0],0);
    model.add_hop(t+delta,1,1,&array![0,-1],0);
    //添加SOC项
    model.add_hop(lm_so,0,1,&array![0,0],3);
    model.add_hop(lm_so,0,1,&array![0,-1],3);
    model.add_hop(lm_so,0,1,&array![-1,0],3);
    model.add_hop(lm_so,0,1,&array![-1,-1],3);




    let nk:usize=1001;
    let path=array![[0.0,0.0],[0.0,0.5],[0.5,0.5],[0.0,0.5],[0.0,0.0]];
    let label=vec!["G","X","M","Y'","G"];
    model.show_band(&path,&label,nk,"examples/chern_alter");



    //画一下贝利曲率的分布
    let dir_1=arr1(&[1.0,0.0]);
    let dir_2=arr1(&[0.0,1.0]);
    let dir_3=arr1(&[0.0,1.0]);
    let T=100.0;
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk]);
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
    draw_heatmap(data,"./examples/chern_alter/nonlinear.pdf");

   //画一下贝利曲率的分布
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk]);
    let kvec=gen_kmesh(&kmesh);
    let kvec=PI*model.lat.dot(&(kvec.reversed_axes()));
    //let kvec=model.lat.dot(&(kvec.reversed_axes()));
    let kvec=kvec.reversed_axes();
    let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,0.0,0.0,0,1e-3);
    let data=berry_curv.clone().into_shape((nk,nk)).unwrap();
    draw_heatmap(data.map(|x| {let a:f64=if *x >= 0.0 {(x+1.0).log(10.0)} else {-(-x+1.0).log(10.0)}; a}),"./examples/chern_alter/heat_map.pdf");
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
    pdf_name.push_str("./examples/chern_alter/nonlinear_ex.pdf");
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
    pdf_name.push_str("./examples/chern_alter/nonlinear_in.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
}
