#![allow(warnings)]
use Rustb::*;
use gnuplot::{AxesCommon, Color, Figure, Fix};
use ndarray::linalg::kron;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::ParallelIterator;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use std::f64::consts::PI;
use std::ops::AddAssign;
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
    let nk: usize = 1000;
    let E_min = -0.22;
    let E_max = 0.22;
    let E_n = 2000;
    let mu = Array1::linspace(E_min, E_max, E_n);
    let T = 30.0;
    let params = IntrinsicNonlinearHallParams::new(
        [nk, nk],
        NonlinearHallDirections::new([1.0, 0.0], [0.0, 1.0], [0.0, 1.0]),
        mu.clone(),
        Occupation::FermiDirac {
            temperature_kelvin: T,
        },
    );
    let sigma = model
        .intrinsic_nonlinear_hall(&params)
        .unwrap()
        .conductivity;

    //开始绘制非线性电导
    let mut fg = Figure::new();
    let x: Vec<f64> = mu.to_vec();
    let axes = fg.axes2d();
    let y: Vec<f64> = sigma.to_vec();
    axes.lines(&x, &y, &[Color("black")]);
    //axes.set_y_range(Fix(-0.3),Fix(0.3));
    axes.set_x_range(Fix(E_min), Fix(E_max));
    fg.set_terminal("pdfcairo", "nonlinear_in.pdf");
    fg.show();
}

///$$H=w\sin(k_x)+v_x \sin(k_x)\tau_x+v_y\sin(k_y)\tau_y\sigma_x+\Delta\tau_z$$
fn gen_model(w: f64, vx: f64, vy: f64, m: f64) -> Model<false, 2> {
    let li: Complex<f64> = 1.0 * Complex::i();
    let norb: usize = 2;
    let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
    let orb = arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]);
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_onsite(&array![m, m, -m, -m], None);
    let w = Complex::new(w, 0.0);
    let vx = Complex::new(vx, 0.0);
    let vy = Complex::new(vy, 0.0);
    for i in 0..4 {
        model.set_hop(-w * li / 2.0, i, i, &array![1, 0], None);
    }
    model.add_hop(-vx * li / 2.0, 0, 2, &array![1, 0], None);
    model.add_hop(vx * li / 2.0, 0, 2, &array![-1, 0], None);
    model.add_hop(-vx * li / 2.0, 1, 3, &array![1, 0], None);
    model.add_hop(vx * li / 2.0, 1, 3, &array![-1, 0], None);
    model.add_hop(vy / 2.0, 0, 3, &array![0, 1], None);
    model.add_hop(-vy / 2.0, 0, 3, &array![0, -1], None);
    model.add_hop(vy / 2.0, 2, 1, &array![0, -1], None);
    model.add_hop(-vy / 2.0, 2, 1, &array![0, 1], None);
    model
}
