#![allow(warnings)]
use Rustb::*;
use gnuplot::AxesCommon;
use gnuplot::{Auto, Caption, Color, Figure, Fix, LineStyle, Solid};
use ndarray::linalg::kron;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::ops::AddAssign;
use std::ops::MulAssign;
fn main() {
    //!来自 PHYSICAL REVIEW X 12, 031042 (2022) 的 RuO2 模型
    //! $$\\mathcal{H}=t(\cos k_x+\cos k_y)\pm \Delta(\cos k_x-\cos k_y)\pm J\sigma_z\tau_z$$
    let li: Complex<f64> = 1.0 * Complex::i();
    let t1 = -1.0 + 0.0 * li;
    let J = 0.5 + 0.0 * li;
    let delta = -0.25 + 0.0 * li;
    let dim_r: usize = 2;
    let norb: usize = 2;
    let a0 = 1.0;
    let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]) * a0;
    let orb = arr2(&[[0.0, 0.0], [0.0, 0.0]]);
    let mut model = Model::tb_model(dim_r, lat, orb, true, None);
    model.add_hop(t1, 0, 0, &array![1, 0], spin_direction::None);
    model.add_hop(t1, 0, 0, &array![0, 1], spin_direction::None);
    model.add_hop(t1, 1, 1, &array![1, 0], spin_direction::None);
    model.add_hop(t1, 1, 1, &array![0, 1], spin_direction::None);

    model.add_hop(delta, 0, 0, &array![1, 0], spin_direction::None);
    model.add_hop(-delta, 0, 0, &array![0, 1], spin_direction::None);
    model.add_hop(-delta, 1, 1, &array![1, 0], spin_direction::None);
    model.add_hop(delta, 1, 1, &array![0, 1], spin_direction::None);

    model.add_hop(J, 0, 0, &array![0, 0], spin_direction::z);
    model.add_hop(-J, 1, 1, &array![0, 0], spin_direction::z);

    let nk: usize = 1001;
    let path = array![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.0, 0.0]];
    let label = vec!["G", "X", "M", "Y", "G"];
    model.show_band(&path, &label, nk, "examples/RuO2");

    //画一下贝利曲率的分布
    /*
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
        draw_heatmap(&data,"./examples/RuO2/nonlinear.pdf");

       //画一下贝利曲率的分布
        let E_min=-1.0;
        let E_max=1.0;
        let E_n=2000;
        let og=0.0;
        let mu=Array1::linspace(E_min,E_max,E_n);
        let nk:usize=1000;
        let kmesh=arr1(&[nk,nk]);
        let kvec=gen_kmesh(&kmesh);
        let kvec=PI*model.lat.dot(&(kvec.reversed_axes()));
        //let kvec=model.lat.dot(&(kvec.reversed_axes()));
        let kvec=kvec.reversed_axes();
        let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,T,0.0,0.0,3,1e-3);
        let data=berry_curv.clone().into_shape((nk,nk)).unwrap();
        draw_heatmap(&data.map(|x| {let a:f64=if *x >= 0.0 {(x+1.0).log(10.0)} else {-(-x+1.0).log(10.0)}; a}),"./examples/RuO2/heat_map.pdf");
        let conductivity=model.Hall_conductivity_mu(&kmesh,&dir_1,&dir_2,0.0,0.0,&mu,3,1e-3);
    //    println!("{}",conductivity/(2.0*PI));

        let mut fg = Figure::new();
        let axes=fg.axes2d();
        let x:Vec<f64>=mu.to_vec();
        let y:Vec<f64>=conductivity.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        //axes.set_y_range(Fix(-10.0),Fix(10.0));
        axes.set_x_range(Fix(E_min),Fix(E_max));
        let mut show_ticks=Vec::<String>::new();
        let mut pdf_name=String::new();
        pdf_name.push_str("./examples/RuO2/spin_Hall_z.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();


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
        */
    //---------------计算新的---------------------
    let k = array![0.25, 0.25] / 2.0;
    let (a, b) = model.solve_onek(&k);

    let nk = 301;
    let k_mesh = arr1(&[nk, nk]);
    let dir_1 = arr1(&[1.0, 0.0]);
    let dir_2 = arr1(&[0.0, 1.0]);
    let mu = 0.5;
    let spin = 3;
    let eta = 1e-1;
    let n_theta = 361;
    let theta = Array1::<f64>::linspace(0.0, 2.0 * PI, n_theta);
    let result: Vec<f64> = theta
        .iter()
        .map(|x| {
            let dir_1 = arr1(&[x.cos(), x.sin(), 0.0]);
            let dir_2 = arr1(&[-x.sin(), x.cos(), 0.0]);
            let cond_xx = conductivity_all(&model, &k_mesh, &dir_1, &dir_1, mu, spin, eta);
            let cond_xy = conductivity_all(&model, &k_mesh, &dir_1, &dir_2, mu, spin, eta);
            cond_xy
        })
        .collect();
    //let result=Array1::from_vec(result);
    let mut fg = Figure::new();
    let x: Vec<f64> = theta
        .iter()
        .zip(result.iter())
        .map(|(x, y)| x.cos() * y.abs())
        .collect();
    let y: Vec<f64> = theta
        .iter()
        .zip(result.iter())
        .map(|(x, y)| x.sin() * y.abs())
        .collect();
    let axes = fg.axes2d();
    axes.lines(&x, &y, &[Color("black"), LineStyle(Solid)]);
    axes.set_aspect_ratio(Fix(1.0));
    let mut pdf_name = String::new();
    pdf_name.push_str("./examples/RuO2/spin_current.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
    //---------------------画出来布里渊区内的分布----------------------------------
    let nk = 501;
    let k_mesh = arr1(&[nk, nk]);
    let x = PI / 4.0;
    let dir_1 = arr1(&[x.cos(), x.sin(), 0.0]);
    let dir_2 = arr1(&[-x.sin(), x.cos(), 0.0]);
    let k_vec = gen_kmesh(&k_mesh);
    let omega: Vec<f64> = k_vec
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|x| {
            let omega_one = conductivity_onek(&model, &x.to_owned(), &dir_1, &dir_2, mu, spin, eta);
            omega_one
        })
        .collect();
    let data = arr1(&omega);
    let data = data.into_shape((nk, nk)).unwrap();
    draw_heatmap(
        &data.map(|x| {
            let a: f64 = if *x >= 0.0 {
                (x + 1.0).log(10.0)
            } else {
                -(-x + 1.0).log(10.0)
            };
            a
        }),
        "./examples/RuO2/spin_current_BZ.pdf",
    );
    let cond_xy = conductivity_all(&model, &k_mesh, &dir_1, &dir_2, mu, spin, eta);
    println!("{}", cond_xy);
    println!(
        "{}",
        conductivity_onek(&model, &k, &dir_1, &dir_2, mu, spin, eta)
    );
}
#[inline(always)]
fn conductivity_onek(
    model: &Model,
    k_vec: &Array1<f64>,
    dir_1: &Array1<f64>,
    dir_2: &Array1<f64>,
    mu: f64,
    spin: usize,
    eta: f64,
) -> f64 {
    //!给定一个k点, 返回 $\Omega_n(\bm k)$
    //返回 $Omega_{n,\ap\bt}, \ve_{n\bm k}$
    let li: Complex<f64> = 1.0 * Complex::i();
    let (band, evec) = model.solve_onek(&k_vec);
    let (mut v, hamk): (Array3<Complex<f64>>, Array2<Complex<f64>>) =
        model.gen_v(k_vec, Gauge::Atom);
    let mut J: Array3<Complex<f64>> = v.clone();
    if model.spin {
        let mut X: Array2<Complex<f64>> = Array2::eye(model.nsta());
        let pauli: Array2<Complex<f64>> = match spin {
            0 => arr2(&[
                [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                [0.0 + 0.0 * li, 1.0 + 0.0 * li],
            ]),
            1 => {
                arr2(&[
                    [0.0 + 0.0 * li, 1.0 + 0.0 * li],
                    [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                ]) / 2.0
            }
            2 => {
                arr2(&[
                    [0.0 + 0.0 * li, 0.0 - 1.0 * li],
                    [0.0 + 1.0 * li, 0.0 + 0.0 * li],
                ]) / 2.0
            }
            3 => {
                arr2(&[
                    [1.0 + 0.0 * li, 0.0 + 0.0 * li],
                    [0.0 + 0.0 * li, -1.0 + 0.0 * li],
                ]) / 2.0
            }
            _ => panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}", spin),
        };
        X = kron(&pauli, &Array2::eye(model.norb()));
        for i in 0..model.dim_r {
            let j = J.slice(s![i, .., ..]).to_owned();
            let j = anti_comm(&X, &j) / 2.0; //这里做反对易
            J.slice_mut(s![i, .., ..]).assign(&(j * dir_1[[i]]));
            v.slice_mut(s![i, .., ..])
                .mul_assign(Complex::new(dir_2[[i]], 0.0));
        }
    } else {
        if spin != 0 {
            println!("Warning, the model haven't got spin, so the spin input will be ignord");
        }
        for i in 0..model.dim_r {
            J.slice_mut(s![i, .., ..])
                .mul_assign(Complex::new(dir_1[[i]], 0.0));
            v.slice_mut(s![i, .., ..])
                .mul_assign(Complex::new(dir_2[[i]], 0.0));
        }
    };

    let J: Array2<Complex<f64>> = J.sum_axis(Axis(0));
    let v: Array2<Complex<f64>> = v.sum_axis(Axis(0));
    let evec_conj: Array2<Complex<f64>> = evec.clone().map(|x| x.conj()).to_owned();
    let A1 = J.dot(&evec.clone().reversed_axes());
    let A1 = &evec_conj.dot(&A1);
    let A2 = v.dot(&evec.reversed_axes());
    let A2 = &evec_conj.dot(&A2);
    let mut U0 = Array2::<Complex<f64>>::zeros((model.nsta(), model.nsta()));
    let eta0 = eta.powi(2);
    for i in 0..model.nsta() {
        for j in 0..model.nsta() {
            U0[[i, j]] = 1.0
                / (((mu - band[[i]]).powi(2) + eta0) * ((mu - band[[j]]).powi(2) + eta0))
                + 0.0 * li;
        }
    }
    //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
    //let mut omega_n=Array1::<f64>::zeros(model.nsta());
    let A1 = A1 * U0;
    let A1 = A1.axis_iter(Axis(0));
    let A2 = A2.axis_iter(Axis(1));
    let omega_one = eta0 * A1.zip(A2).map(|(x, y)| x.dot(&y).re).sum::<f64>();
    omega_one
}
fn conductivity_all(
    model: &Model,
    k_mesh: &Array1<usize>,
    dir_1: &Array1<f64>,
    dir_2: &Array1<f64>,
    mu: f64,
    spin: usize,
    eta: f64,
) -> f64 {
    let k_vec = gen_kmesh(&k_mesh);
    let nk = k_vec.len_of(Axis(0));
    let omega: Vec<f64> = k_vec
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|x| {
            let omega_one = conductivity_onek(&model, &x.to_owned(), &dir_1, &dir_2, mu, spin, eta);
            omega_one
        })
        .collect();
    let omega = arr1(&omega);
    let V = model.lat.det().unwrap();
    omega.sum() / (nk as f64) * (2.0 * PI).powi(3) / V
}
