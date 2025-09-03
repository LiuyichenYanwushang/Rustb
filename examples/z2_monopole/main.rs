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
    let th1 = 0.2 + 0.0 * li;
    let th2 = 0.4 + 0.0 * li;
    let delta = 0.5 + 0.0 * li;
    let dim_r: usize = 3;
    let lat = arr2(&[
        [3.0_f64.sqrt(), -1.0, 0.0],
        [3.0_f64.sqrt(), 1.0, 0.0],
        [0.0, 0.0, 2.0],
    ]);
    let orb = arr2(&[
        [1.0 / 3.0, 0.0, 0.],
        [2.0 / 3.0, 0.0, 0.],
        [0.0, 1.0 / 3.0, 0.],
        [0.0, 2.0 / 3.0, 0.],
        [2.0 / 3.0, 1.0 / 3.0, 0.],
        [1.0 / 3.0, 2.0 / 3.0, 0.],
    ]);
    let mut model = Model::tb_model(dim_r, lat, orb, true, None).unwrap();
    //onsite hopping
    model.add_hop(
        delta * 3.0_f64.sqrt(),
        0,
        0,
        &array![0, 0, 0],
        SpinDirection::x,
    );
    model.add_hop(-delta, 0, 0, &array![0, 0, 0], SpinDirection::y);
    model.add_hop(
        -delta * 3.0_f64.sqrt(),
        3,
        3,
        &array![0, 0, 0],
        SpinDirection::x,
    );
    model.add_hop(-delta, 3, 3, &array![0, 0, 0], SpinDirection::y);
    model.add_hop(delta, 4, 4, &array![0, 0, 0], SpinDirection::y);

    /*
    model.add_hop(-delta*3.0_f64.sqrt(),1,1,&array![0,0,0],SpinDirection::x);
    model.add_hop(delta,1,1,&array![0,0,0],SpinDirection::y);
    model.add_hop(delta*3.0_f64.sqrt(),2,2,&array![0,0,0],SpinDirection::x);
    model.add_hop(delta,2,2,&array![0,0,0],SpinDirection::y);
    model.add_hop(-delta,5,5,&array![0,0,0],SpinDirection::y);
    */

    model.add_hop(t1, 0, 1, &array![0, 0, 0], SpinDirection::None);
    model.add_hop(t1, 0, 2, &array![0, 0, 0], SpinDirection::None);
    model.add_hop(t1, 0, 5, &array![0, -1, 0], SpinDirection::None);
    model.add_hop(t1, 1, 4, &array![0, 0, 0], SpinDirection::None);
    model.add_hop(t1, 1, 3, &array![1, -1, 0], SpinDirection::None);
    model.add_hop(t1, 2, 3, &array![0, 0, 0], SpinDirection::None);
    model.add_hop(t1, 2, 4, &array![-1, 0, 0], SpinDirection::None);
    model.add_hop(t1, 3, 5, &array![0, 0, 0], SpinDirection::None);
    model.add_hop(t1, 4, 5, &array![0, 0, 0], SpinDirection::None);

    //    model.add_hop(t1,2,3,&array![0,0,0],SpinDirection::None);
    //    model.add_hop(t1,2,3,&array![-1,0,0],SpinDirection::None);
    //    model.add_hop(t1,2,3,&array![0,-1,0],SpinDirection::None);
    //    model.add_hop(th1,0,2,&array![0,0,0],SpinDirection::None);
    //    model.add_hop(th2,1,3,&array![0,0,0],SpinDirection::None);
    //    model.add_hop(th2,0,2,&array![0,0,-1],SpinDirection::None);
    //    model.add_hop(th1,1,3,&array![0,0,-1],SpinDirection::None);
    let nk: usize = 1001;
    //let path=[[0.0,0.0,0.0],[2.0/3.0,1.0/3.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0],[0.0,0.0,0.5],[2.0/3.0,1.0/3.0,0.5],[2.0/3.0,1.0/3.0,0.0]];
    //let label=vec!["G","K","M","G","H","K0","K"];
    let path = [
        [0.0, 0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let label = vec!["G", "K", "M", "G"];
    let path = arr2(&path);
    let (k_vec, k_dist, k_node) = model.k_path(&path, nk).unwrap();
    let (eval, evec) = model.solve_all_parallel(&k_vec);
    model.show_band(&path, &label, nk, "./examples/z2_monopole/result/");
    println!(
        "ham0={}",
        model.gen_ham(&array![0.0, 0.0, 0.0], Gauge::Atom)
    );
}
