use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon};
use gnuplot::Major;
use Rustb::*;
fn main(){
    let li:Complex<f64>=1.0*Complex::i();
    let t_sg=3.0+0.0*li;
    let t_pi=1.0+0.0*li;
    let delta=0.5;
    let dim_r:usize=3;
    let norb:usize=2;
    let a0=2.0;
    let h=0.3;
    let lat=arr2(&[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])*a0;
    let orb=arr2(&[[0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,h],[0.0,0.5,h]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    let R0:Array2::<isize>=arr2(&[[0,0,0],[1,0,0],[0,-1,0],[1,-1,0]]);
    let cos_t2=1.0-h.powi(2)/2.0;
    let t_sg=t_sg*cos_t2;
    //我们只考虑最近邻
    model.add_hop(-t_sg/2.0,0,2,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi/2.0,0,2,&R0.row(0).to_owned(),0);
    model.add_hop(t_sg/2.0,0,3,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi/2.0,0,3,&R0.row(0).to_owned(),0);
    model.add_hop(t_sg/2.0,1,2,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi/2.0,1,2,&R0.row(0).to_owned(),0);
    model.add_hop(-t_sg/2.0,1,3,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi/2.0,1,3,&R0.row(0).to_owned(),0);
    //第二个
    model.add_hop(-t_sg/2.0,0,2,&R0.row(1).to_owned(),0);
    model.add_hop(t_pi/2.0,0,2,&R0.row(1).to_owned(),0);
    model.add_hop(-t_sg/2.0,0,3,&R0.row(1).to_owned(),0);
    model.add_hop(-t_pi/2.0,0,3,&R0.row(1).to_owned(),0);
    model.add_hop(-t_sg/2.0,1,2,&R0.row(1).to_owned(),0);
    model.add_hop(-t_pi/2.0,1,2,&R0.row(1).to_owned(),0);
    model.add_hop(-t_sg/2.0,1,3,&R0.row(1).to_owned(),0);
    model.add_hop(t_pi/2.0,1,3,&R0.row(1).to_owned(),0);
    //第三个
    model.add_hop(-t_sg/2.0,0,2,&R0.row(2).to_owned(),0);
    model.add_hop(t_pi/2.0,0,2,&R0.row(2).to_owned(),0);
    model.add_hop(-t_sg/2.0,0,3,&R0.row(2).to_owned(),0);
    model.add_hop(-t_pi/2.0,0,3,&R0.row(2).to_owned(),0);
    model.add_hop(-t_sg/2.0,1,2,&R0.row(2).to_owned(),0);
    model.add_hop(-t_pi/2.0,1,2,&R0.row(2).to_owned(),0);
    model.add_hop(-t_sg/2.0,1,3,&R0.row(2).to_owned(),0);
    model.add_hop(t_pi/2.0,1,3,&R0.row(2).to_owned(),0);
    //第四个
    model.add_hop(-t_sg/2.0,0,2,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi/2.0,0,2,&R0.row(3).to_owned(),0);
    model.add_hop(t_sg/2.0,0,3,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi/2.0,0,3,&R0.row(3).to_owned(),0);
    model.add_hop(t_sg/2.0,1,2,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi/2.0,1,2,&R0.row(3).to_owned(),0);
    model.add_hop(-t_sg/2.0,1,3,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi/2.0,1,3,&R0.row(3).to_owned(),0);
    //开始加上自旋轨道耦合
    let lm=0.0+0.0*li;
    model.add_hop(lm*li,0,1,&array![0,0,0],3);
    model.add_hop(lm*li,2,3,&array![0,0,0],3);
    //最后一项
    let t_eff=0.0+0.0*li;
    //接下来是类似于Rashba 自旋轨道耦合项
    model.add_hop(t_eff,0,3,&R0.row(0).to_owned(),2);
    model.add_hop(t_eff,1,2,&R0.row(0).to_owned(),2);
    model.add_hop(t_eff,0,3,&R0.row(0).to_owned(),2);
    model.add_hop(t_eff,1,2,&R0.row(0).to_owned(),2);
    //开始计算体能带
    let path=array![[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.5,0.0],[0.0,0.0,0.0],[0.5,0.5,0.0]];
    let label=vec!["G","X","M","Y","G","M"];
    let nk=1001;
    model.show_band(&path,&label,nk,"examples/BiF_square/band");
    //加入altermagnetics

    model.set_onsite(array![delta,delta,-delta,-delta],3);
    let path=array![[0.25,-0.5,0.0],[0.25,0.0,0.0],[0.25,0.5,0.0]];
    let (kvec,kdist,knode)=model.k_path(&path,nk);
    let (eval,evec):(Array2<f64>,Array3<Complex<f64>>)=model.solve_all(&kvec);
    let label=vec!["M1","X1","M2"];
    let nk=1001;
    let evec:Array3<f64>=evec.map(|x| x.norm_sqr());

    let mut fg = Figure::new();
    let x:Vec<f64>=kdist.to_vec();
    let axes=fg.axes2d();
    for i in 0..model.nsta{
        let y:Vec<f64>=eval.slice(s![..,i]).to_owned().to_vec();
        axes.lines(&x, &y, &[Color("black")]);
    }
    let axes=axes.set_x_range(Fix(0.0), Fix(knode[[knode.len()-1]]));
    let label=label.clone();
    let mut show_ticks=Vec::new();
    for i in 0..knode.len(){
        let A=knode[[i]];
        let B=label[i];
        show_ticks.push(Major(A,Fix(B)));
    }
    axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
    let knode=knode.to_vec();
    let mut pdf_name=String::new();
    pdf_name.push_str("examples/BiF_square/alter/plot.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
}
