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
    let t1 = 1.0 + 0.0 * li;
    let lm_so = 0.2 + 0.0 * li;
    let J = -0.5 + 0.0 * li;
    let delta = 0.2;
    let norb: usize = 2;
    let a0 = 0.5;
    let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]) * a0;
    let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
    let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    model.set_onsite(&arr1(&[delta, delta]), SpinDirection::Z);
    //最近邻hopping
    model.add_hop(t1, 0, 1, &array![0, 0], None);
    model.add_hop(t1, 0, 1, &array![-1, 0], None);
    model.add_hop(t1, 0, 1, &array![0, -1], None);
    //Rashba
    let d1 = model.orb.row(1).to_owned() - model.orb.row(0).to_owned();
    let d1 = d1.dot(&model.lat);
    println!("{}", d1);
    model.add_hop(-lm_so * li * d1[[0]], 0, 1, &array![0, 0], SpinDirection::Y);
    model.add_hop(lm_so * li * d1[[1]], 0, 1, &array![0, 0], SpinDirection::X);
    let d2 = model.orb.row(1).to_owned() - model.orb.row(0).to_owned() + &array![-1.0, 0.0];
    let d2 = d2.dot(&model.lat);
    println!("{}", d2);
    model.add_hop(
        -lm_so * li * d2[[0]],
        0,
        1,
        &array![-1, 0],
        SpinDirection::Y,
    );
    model.add_hop(lm_so * li * d2[[1]], 0, 1, &array![-1, 0], SpinDirection::X);
    let d3 = model.orb.row(1).to_owned() - model.orb.row(0).to_owned() + &array![0.0, -1.0];
    let d3 = d3.dot(&model.lat);
    println!("{}", d3);
    model.add_hop(
        -lm_so * li * d3[[0]],
        0,
        1,
        &array![0, -1],
        SpinDirection::Y,
    );
    model.add_hop(lm_so * li * d3[[1]], 0, 1, &array![0, -1], SpinDirection::X);
    //最后加上d+id 项
    model.add_hop(J, 0, 1, &array![0, 0], SpinDirection::X);
    model.add_hop(
        J * (-PI * 4.0 / 3.0 * li).exp(),
        0,
        1,
        &array![-1, 0],
        SpinDirection::X,
    );
    model.add_hop(
        J * (-PI * 8.0 / 3.0 * li).exp(),
        0,
        1,
        &array![0, -1],
        SpinDirection::X,
    );
    /*
    model.add_hop(J,0,0,&array![1,0],SpinDirection::Z);
    model.add_hop(J,1,1,&array![1,0],SpinDirection::Z);
    model.add_hop(J*(-PI*4.0/3.0*li).exp(),0,0,&array![0,1],SpinDirection::Z);
    model.add_hop(J*(-PI*4.0/3.0*li).exp(),1,1,&array![0,1],SpinDirection::Z);
    model.add_hop(J*(-PI*8.0/3.0*li).exp(),0,0,&array![-1,1],SpinDirection::Z);
    model.add_hop(J*(-PI*8.0/3.0*li).exp(),1,1,&array![-1,1],SpinDirection::Z);
    */
    let nk: usize = 1001;
    let path = array![
        [0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.5],
        [1.0 / 3.0, 2.0 / 3.0],
        [0.0, 0.0]
    ];
    let label = vec!["G", "K", "M", "K'", "G"];
    model.show_band(&path, &label, nk, "examples/yuxuan_try");

    //画一下贝利曲率的分布
    let T = 100.0;
    let nk: usize = 1000;
    let kmesh = arr1(&[nk, nk]);
    let kvec = gen_kmesh(&kmesh).unwrap();
    let kvec = PI * model.lat.dot(&(kvec.reversed_axes()));
    //let kvec=model.lat.dot(&(kvec.reversed_axes()));
    let kvec = kvec.reversed_axes();
    let mut berry_params = Parameters::rank2([1, 1], [1.0, 0.0], [0.0, 1.0], array![0.0]);
    berry_params.T = array![T];
    let berry_curv = model.occupied_berry_curvature_on(&kvec, &berry_params).unwrap();
    let data = berry_curv.clone().into_shape((nk, nk)).unwrap();
    draw_heatmap(
        &data.map(|x| {
            let a: f64 = if *x >= 0.0 {
                (x + 1.0).log(10.0)
            } else {
                -(-x + 1.0).log(10.0)
            };
            a
        }),
        "./examples/yuxuan_try/heat_map.pdf",
    );
    let hall_params = Parameters::rank2([nk, nk], [1.0, 0.0], [0.0, 1.0], array![0.0]);
    let conductivity = model
        .hall_conductivity(&hall_params)
        .unwrap()
        .single()
        .unwrap();
    println!("{}", conductivity / (2.0 * PI));
}
