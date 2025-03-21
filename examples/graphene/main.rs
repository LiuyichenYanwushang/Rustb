#![allow(warnings)]
use Rustb::*;
use gnuplot::{AxesCommon, Color, Figure, Fix};
use ndarray::linalg::kron;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use std::f64::consts::PI;
use std::ops::AddAssign;
fn main() {
    let li: Complex<f64> = 1.0 * Complex::i();
    let t1 = -2.85 + 0.0 * li;
    let soc = 0.2;
    let delta = 0.5;
    let J = 0.1;
    let dim_r: usize = 2;
    let norb: usize = 2;
    let a0 = 1.0;
    let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]) * a0;
    let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
    let mut model = Model::tb_model(dim_r, lat, orb, true, None);
    model.add_onsite(&arr1(&[delta, -delta]), spin_direction::z);
    model.add_onsite(&arr1(&[J, J]), spin_direction::z);
    model.add_hop(t1, 0, 1, &array![0, 0], spin_direction::None);
    model.add_hop(t1, 0, 1, &array![-1, 0], spin_direction::None);
    model.add_hop(t1, 0, 1, &array![0, -1], spin_direction::None);
    model.add_hop(li * soc, 0, 0, &array![1, 0], spin_direction::z);
    model.add_hop(-li * soc, 1, 1, &array![1, 0], spin_direction::z);
    model.add_hop(li * soc, 0, 0, &array![0, 1], spin_direction::z);
    model.add_hop(-li * soc, 1, 1, &array![0, 1], spin_direction::z);
    model.add_hop(li * soc, 0, 0, &array![1, -1], spin_direction::z);
    model.add_hop(-li * soc, 1, 1, &array![1, -1], spin_direction::z);
    println!("{}", model.ham);
    println!("{}", model.hamR);
    /*
    model.add_hop(t3,0,1,&array![1,-1],spin_direction::None);
    model.add_hop(t3,0,1,&array![-1,1],spin_direction::None);
    model.add_hop(t3,0,1,&array![-1,-1],spin_direction::None);
    */
    let path = array![
        [0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [0.0, 0.0]
    ];
    let nk = 2001;
    let label = vec!["G", "K", "K'", "G"];
    let name = "./examples/graphene/";
    model.show_band(&path, &label, nk, name);

    /*
     //画一下贝利曲率的分布
     let dir_1=arr1(&[1.0,0.0]);
     let dir_2=arr1(&[0.0,1.0]);
     let dir_3=arr1(&[1.0,0.0]);
     let T=1000.0;
     let nk:usize=1000;
     let kmesh=arr1(&[nk,nk]);
     let kvec=gen_kmesh(&kmesh);
     let lat_inv=model.lat.inv().unwrap();
     let kvec=model.lat.dot(&(kvec.reversed_axes()));
     let kvec=kvec.reversed_axes();
     let (berry_curv,band)=model.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,0.0,0,1e-3);
     ///////////////////////////////////////////
     let beta=1.0/T/(8.617e-5);
     let f:Array2::<f64>=band.clone().map(|x| 1.0/((beta*x).exp()+1.0));
     let f=beta*&f*(1.0-&f);
     println!("{:?}",berry_curv.shape());
     let berry_curv=(berry_curv.clone()*f).sum_axis(Axis(1));
     let data=berry_curv.into_shape((nk,nk)).unwrap();
     draw_heatmap(&data,"./examples/graphene/nonlinear.pdf");

    //画一下贝利曲率的分布
     let nk:usize=1000;
     let kmesh=arr1(&[nk,nk]);
     let kvec=gen_kmesh(&kmesh);
     let kvec=model.lat.dot(&(kvec.reversed_axes()));
     //let kvec=model.lat.dot(&(kvec.reversed_axes()));
     let kvec=kvec.reversed_axes();
     let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,0.0,0.0,0,1e-3);
     let data=berry_curv.into_shape((nk,nk)).unwrap();
     draw_heatmap(&data.map(|x| {let a:f64=if *x >= 0.0 {(x+1.0).log(10.0)} else {-(-x+1.0).log(10.0)}; a}),"./examples/graphene/heat_map.pdf");

     let E_min=-3.0;
     let E_max=3.0;
     let E_n=1000;
     let og=0.0;
     let mu=Array1::linspace(E_min,E_max,E_n);
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
     pdf_name.push_str("./examples/graphene/nonlinear_ex.pdf");
     fg.set_terminal("pdfcairo", &pdf_name);
     fg.show();
     */
}
