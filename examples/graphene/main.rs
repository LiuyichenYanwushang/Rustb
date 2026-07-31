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
    let norb: usize = 2;
    let a0 = 1.0;
    let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]) * a0;
    let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
    let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    model.add_onsite(&arr1(&[delta, -delta]), SpinDirection::Z);
    model.add_onsite(&arr1(&[J, J]), SpinDirection::Z);
    model.add_hop(t1, 0, 1, &array![0, 0], None);
    model.add_hop(t1, 0, 1, &array![-1, 0], None);
    model.add_hop(t1, 0, 1, &array![0, -1], None);
    model.add_hop(li * soc, 0, 0, &array![1, 0], SpinDirection::Z);
    model.add_hop(-li * soc, 1, 1, &array![1, 0], SpinDirection::Z);
    model.add_hop(li * soc, 0, 0, &array![0, 1], SpinDirection::Z);
    model.add_hop(-li * soc, 1, 1, &array![0, 1], SpinDirection::Z);
    model.add_hop(li * soc, 0, 0, &array![1, -1], SpinDirection::Z);
    model.add_hop(-li * soc, 1, 1, &array![1, -1], SpinDirection::Z);
    println!("{}", model.ham);
    println!("{}", model.hamR);
    /*
    model.add_hop(t3,0,1,&array![1,-1],None);
    model.add_hop(t3,0,1,&array![-1,1],None);
    model.add_hop(t3,0,1,&array![-1,-1],None);
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
}
