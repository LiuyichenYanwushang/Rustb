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
    let t1=1.0+0.0*li;
    let th1=0.2+0.0*li;
    let th2=0.4+0.0*li;
    let delta=0.0;
    let dim_r:usize=3;
    let lat=arr2(&[[1.0,0.0,0.0],[1.0/2.0,3.0_f64.sqrt()/2.0,0.0],[0.0,0.0,2.0]]);
    let orb=arr2(&[[1.0/3.0,1.0/3.0,0.1],[2.0/3.0,2.0/3.0,0.9],[1.0/3.0,1.0/3.0,0.4],[2.0/3.0,2.0/3.0,0.6]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    model.set_onsite(arr1(&[delta,-delta,delta,-delta]),3);
    model.add_hop(t1,0,1,&array![0,0,0],0);
    model.add_hop(t1,0,1,&array![-1,0,0],0);
    model.add_hop(t1,0,1,&array![0,-1,0],0);
    model.add_hop(t1,2,3,&array![0,0,0],0);
    model.add_hop(t1,2,3,&array![-1,0,0],0);
    model.add_hop(t1,2,3,&array![0,-1,0],0);

    /*
    model.add_hop(t1,0,1,&array![0,0,0],1);
    model.add_hop(t1,0,1,&array![-1,0,0],1);
    model.add_hop(t1,0,1,&array![0,-1,0],1);
    model.add_hop(t1,2,3,&array![0,0,0],1);
    model.add_hop(t1,2,3,&array![-1,0,0],1);
    model.add_hop(t1,2,3,&array![0,-1,0],1);

    model.add_hop(li*t1,0,1,&array![0,0,0],2);
    model.add_hop(li*t1,0,1,&array![-1,0,0],2);
    model.add_hop(li*t1,0,1,&array![0,-1,0],2);
    model.add_hop(li*t1,2,3,&array![0,0,0],2);
    model.add_hop(li*t1,2,3,&array![-1,0,0],2);
    model.add_hop(li*t1,2,3,&array![0,-1,0],2);
    */

    model.add_hop(th1,0,2,&array![0,0,0],0);
    model.add_hop(th2,1,3,&array![0,0,0],0);
    model.add_hop(th2,0,2,&array![0,0,-1],0);
    model.add_hop(th1,1,3,&array![0,0,-1],0);
    let nk:usize=1001;
    let path=[[0.0,0.0,0.0],[2.0/3.0,1.0/3.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0],[0.0,0.0,0.5],[2.0/3.0,1.0/3.0,0.5],[2.0/3.0,1.0/3.0,0.0]];
    let path=arr2(&path);
    let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
    let (eval,evec)=model.solve_all_parallel(&k_vec);
    let label=vec!["G","K","M","G","H","K0","K"];
    model.show_band(&path,&label,nk,"./examples/z2_AF");

    println!("ham0={}",model.gen_ham(&array![0.0,0.0,0.0]));


    //开始计算非线性霍尔电导
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let dir_3=arr1(&[0.0,1.0,0.0]);
    let nk:usize=250;
    let kmesh=arr1(&[nk,nk,nk]);
    let E_min=-0.1;
    let E_max=0.1;
    let E_n=1000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let T=30.0;
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
    pdf_name.push_str("nonlinear_in.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let dir_3=arr1(&[0.0,1.0,0.0]);
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk,1]);
    let E_min=-3.0;
    let E_max=3.0;
    let E_n=1000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let T=100.0;
    let beta=1.0/T/8.617e-5;
    let mu0=-0.0;


    let kvec=gen_kmesh(&kmesh);
    //let kvec=kvec+array![0.0,0.0,0.0];
    //let kvec=model.lat.dot(&(kvec.reversed_axes()));
    //let kvec=kvec.reversed_axes();

    let (berry_curv,band,G)=model.berry_connection_dipole(&kvec,&dir_1,&dir_2,&dir_3,0);
    let G=G.unwrap();
    let G=G.into_shape((nk,nk,model.nsta)).unwrap();
    let berry_curv=berry_curv.into_shape((nk,nk,model.nsta)).unwrap();
    //let data=berry_curv.slice(s![..,..,0..4]).to_owned().sum_axis(Axis(2));
    let data=berry_curv.slice(s![..,..,3]).to_owned();
    let data=data.map(|x| if *x>0.0 {(x+1.0).ln()} else {-(-x+1.0).ln()});
    draw_heatmap(&data.reversed_axes(),"heat_map.pdf");
    let band=band.into_shape((nk,nk,model.nsta)).unwrap();
    let f:Array3::<f64>=1.0/(beta*(&band-mu0)).map(|x| x.exp()+1.0);
    let pf=&f*(1.0-&f)*beta;
    draw_heatmap(&pf.slice(s![..,..,4]).reversed_axes(),"f_map.pdf");
    let a=(&berry_curv*&pf).sum_axis(Axis(2));
    //let a=a.map(|x| if *x>0.0 {(x+1.0).ln()} else {-(-x+1.0).ln()});
    draw_heatmap(&a.clone().reversed_axes(),"result_map.pdf");
    let a=(&G*&f).sum_axis(Axis(2));
    //let a=a.map(|x| if *x>0.0 {(x+1.0).ln()} else {-(-x+1.0).ln()});
    draw_heatmap(&a.clone().reversed_axes(),"result_map_1.pdf");

    let mut dir_1=arr1(&[0.0,0.0,0.0]);
    let mut dir_2=arr1(&[0.0,0.0,0.0]);
    let nk:usize=200;
    let kmesh=arr1(&[nk,nk,nk]);
    let og=0.0;
    let mu0=0.0;
    let T=30.0;
    for i in 0..3{
        dir_1[[i]]=1.0;
        for j in 0..3{
            dir_2[[j]]=1.0;
            for s in 0..4{
                let sigma=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,T,og,mu0,s,1e-5);
                println!("{},{},{},sigma={}",s,i,j,sigma);
            }
            dir_2[[j]]=0.0;
        }
        dir_1[[i]]=0.0;
    }


    /*
    let (berry_curv,band)=model.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,og,0,1e-5);
    let berry_curv=berry_curv.into_shape((nk,nk,model.nsta)).unwrap();
    let data=berry_curv.slice(s![..,..,0..4]).to_owned().sum_axis(Axis(2));
    draw_heatmap(data.clone().reversed_axes(),"heat_map.pdf");
    let band=band.into_shape((nk,nk,model.nsta)).unwrap();
    let f:Array3::<f64>=1.0/(beta*(&band-mu0)).map(|x| x.exp()+1.0);
    let pf=&f*(1.0-&f)*beta;
    draw_heatmap(pf.slice(s![..,..,4]).to_owned().reversed_axes(),"f_map.pdf");
    let a=(&berry_curv*&pf).sum_axis(Axis(2));
    draw_heatmap(a.clone().reversed_axes(),"result_map.pdf");
    //println!("{}",a.sum()/(nk.pow(2) as f64)*ratio.powi(2));
    */
}
