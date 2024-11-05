#![allow(warnings)]
use gnuplot::{AxesCommon, Color, Figure, Fix};
use ndarray::linalg::kron;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::ParallelIterator;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use std::f64::consts::PI;
use std::ops::AddAssign;
use Rustb::*;
///主要参考这篇文章 10.1103/PhysRevLett.127.277202
///
///kp模型为
///
///$$H=wk_x+v_xk_x\tau_x+v_yk_y\tau_y\sigma_x+\Delta\tau_z$$
///
///对应的 TB 模型为
///
///$$H=w\sin(k_x)+v_x \sin(k_x)\tau_x+v_y\sin(k_y)\tau_y\sigma_x+\Delta\tau_z$$
fn main() {
    let vx = 1.;
    let vy = vx.clone();
    let w = 0.4 * vx;
    let m = 0.04;
    let path = array![[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]];
    let label = vec!["M", "G", "M"];
    let nk = 1001;
    let model = gen_model(w, vx, vy, m);
    model.show_band(&path, &label, nk, "./examples/Intrinsic_nonlinear/result/");

    //开始计算非线性霍尔电导
    let dir_1 = arr1(&[1.0, 0.0]);
    let dir_2 = arr1(&[0.0, 1.0]);
    let dir_3 = arr1(&[0.0, 1.0]);
    let nk: usize = 1000;
    let kmesh = arr1(&[nk, nk]);
    let E_min = -0.22;
    let E_max = 0.22;
    let E_n = 2000;
    let og = 0.0;
    let mu = Array1::linspace(E_min, E_max, E_n);
    let T = 30.0;
    /*
    let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Intrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&mu,T,0);
    //let sigma=sigma/(2.0*PI).powi(2);

    //开始绘制非线性电导
    let mut fg = Figure::new();
    let x:Vec<f64>=mu.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=sigma.to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    //axes.set_y_range(Fix(-0.3),Fix(0.3));
    axes.set_x_range(Fix(E_min),Fix(E_max));
    let mut show_ticks=Vec::<String>::new();
    let mut pdf_name=String::new();
    pdf_name.push_str("nonlinear_in.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();

    */
    let k = array![0.0, 0.01 / 2.0 / PI];
    let mu0 = -0.04;
    let (omega_one, band, partial_G) =
        model.berry_connection_dipole_onek(&k, &dir_1, &dir_2, &dir_3, 0);
    let beta = 1.0 / T / 8.617e-5;
    let f = 1.0 / (beta * (&band - mu0)).map(|x| x.exp() + 1.0);
    let pf = &f * (1.0 - &f) * beta;
    println!("band={}", band);
    println!("omega={},pf={}", omega_one, pf);
    println!("{}", (omega_one * pf).sum());
    //let sigma:Array1<f64>=model.Nonlinear_Hall_conductivity_Intrinsic(&kmesh,&dir_1,&dir_2,&dir_3,&array![mu0],T,0);
    //println!("{}",sigma);
    let kvec = gen_kmesh(&kmesh);
    let ratio = 0.1;
    let kvec = (kvec - 0.5) * ratio;
    let kvec = model.lat.dot(&(kvec.reversed_axes()));
    let kvec = kvec.reversed_axes();
    let (berry_curv, band, _) = model.berry_connection_dipole(&kvec, &dir_1, &dir_2, &dir_3, 0);
    let berry_curv = berry_curv.into_shape((nk, nk, model.nsta())).unwrap();
    let data = berry_curv
        .slice(s![.., .., 0..2])
        .to_owned()
        .sum_axis(Axis(2));
    draw_heatmap(
        &data.clone().reversed_axes(),
        "examples/Intrinsic_nonlinear/result/heat_map.pdf",
    );
    let band = band.into_shape((nk, nk, model.nsta())).unwrap();
    let f: Array3<f64> = 1.0 / (beta * (&band - mu0)).map(|x| x.exp() + 1.0);
    let pf = &f * (1.0 - &f) * beta;
    draw_heatmap(
        &pf.slice(s![.., .., 0]).to_owned().reversed_axes(),
        "examples/Intrinsic_nonlinear/result/f_map.pdf",
    );
    let a = (&berry_curv * &pf).sum_axis(Axis(2));
    draw_heatmap(
        &a.clone().reversed_axes(),
        "examples/Intrinsic_nonlinear/result/result_map.pdf",
    );
    println!("{}", a.sum() / (nk.pow(2) as f64) * ratio.powi(2));

    let mut conductivity = Vec::<f64>::new();
    let mu = Array1::linspace(E_min, E_max, E_n);
    let conductivity = mu
        .par_iter()
        .map(|x| {
            let f: Array3<f64> = 1.0 / (beta * (&band - *x)).map(|x| x.exp() + 1.0);
            let pf = -&f * (1.0 - &f) * beta;
            let a =
                (&berry_curv * &pf).sum_axis(Axis(2)).sum() / (nk.pow(2) as f64) * ratio.powi(2);
            a / PI / 2.0
        })
        .collect();
    let mut fg = Figure::new();
    let x: Vec<f64> = mu.to_vec();
    let axes = fg.axes2d();
    let y: Vec<f64> = conductivity;
    axes.lines(&x, &y, &[Color("black")]);
    //axes.set_y_range(Fix(-0.3),Fix(0.3));
    axes.set_x_range(Fix(E_min), Fix(E_max));
    let mut show_ticks = Vec::<String>::new();
    let mut pdf_name = String::new();
    pdf_name.push_str("./examples/Intrinsic_nonlinear/result/nonlinear_in.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
}

///$$H=w\sin(k_x)+v_x \sin(k_x)\tau_x+v_y\sin(k_y)\tau_y\sigma_x+\Delta\tau_z$$
fn gen_model(w: f64, vx: f64, vy: f64, m: f64) -> Model {
    let li: Complex<f64> = 1.0 * Complex::i();
    let dim_r: usize = 2;
    let norb: usize = 2;
    let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let orb = arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    let mut model = Model::tb_model(dim_r, lat, orb, false, None);
    model.set_onsite(&array![m, m, -m, -m], 0);
    let w = Complex::new(w, 0.0);
    let vx = Complex::new(vx, 0.0);
    let vy = Complex::new(vy, 0.0);
    for i in 0..4 {
        model.set_hop(-w * li / 2.0, i, i, &array![1, 0], 0);
    }
    model.add_hop(-vx * li / 2.0, 0, 2, &array![1, 0], 0);
    model.add_hop(vx * li / 2.0, 0, 2, &array![-1, 0], 0);
    model.add_hop(-vx * li / 2.0, 1, 3, &array![1, 0], 0);
    model.add_hop(vx * li / 2.0, 1, 3, &array![-1, 0], 0);
    model.add_hop(vy / 2.0, 0, 3, &array![0, 1], 0);
    model.add_hop(-vy / 2.0, 0, 3, &array![0, -1], 0);
    model.add_hop(vy / 2.0, 2, 1, &array![0, -1], 0);
    model.add_hop(-vy / 2.0, 2, 1, &array![0, 1], 0);
    model
}
