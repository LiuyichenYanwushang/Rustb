use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon,Major};
fn main(){
    let li:Complex<f64>=1.0*Complex::i();
    let t1=1.0+0.0*li;
    let J=-0.5+0.0*li;
    let delta=0.2;
    let dim_r:usize=2;
    let norb:usize=2;
    let a0=0.5;
    //六角晶格
    //计算下层的哈密顿量
    let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
    let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    model.set_onsite(arr1(&[delta,delta]),3);
    //最近邻hopping
    model.add_hop(t1,0,1,&array![0,0],0);
    model.add_hop(t1,0,1,&array![-1,0],0);
    model.add_hop(t1,0,1,&array![0,-1],0);
    let model_dn=model.make_supercell(&array![[3.0,1.0],[-1.0,2.0]]);

    //计算上层的哈密顿量
    let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]]);
    let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    model.set_onsite(arr1(&[-delta,-delta]),3);
    //最近邻hopping
    model.add_hop(t1,0,1,&array![0,0],0);
    model.add_hop(t1,0,1,&array![-1,0],0);
    model.add_hop(t1,0,1,&array![0,-1],0);
    let model_up=model.make_supercell(&array![[2.0,-1.0],[1.0,3.0]]);

    let lat=model_up.lat.clone();
    let orb=model_up.orb.clone();

    let nk:usize=1001;
    let path=array![[0.0,0.0],[2.0/3.0,1.0/3.0],[-1.0/3.0,1.0/3.0],[0.0,0.0]];
    let label=vec!["G","K","K'","G"];
    let (kvec,k_dist,k_node)=model_all.k_path(&path,nk);
    let band=model_all.solve_band_all_parallel(&kvec);


    let mut fg = Figure::new();
    let x:Vec<f64>=k_dist.to_vec();
    let axes=fg.axes2d();
    for i in 0..model_dn.nsta{
        let y:Vec<f64>=band_dn.slice(s![..,i]).to_owned().to_vec();
        axes.lines(&x, &y, &[Color("blue")]);
        let y:Vec<f64>=band_up.slice(s![..,i]).to_owned().to_vec();
        axes.lines(&x, &y, &[Color("red")]);
    }
    let axes=axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len()-1]]));
    let label=label.clone();
    let mut show_ticks=Vec::new();
    for i in 0..k_node.len(){
        let A=k_node[[i]];
        let B=label[i];
        show_ticks.push(Major(A,Fix(B)));
    }
    axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);

    let name=String::from("./examples/alter_twist");
    let mut pdf_name=name.clone();
    pdf_name.push_str("/plot.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
    //正方晶格
    
    //长方晶格
}

