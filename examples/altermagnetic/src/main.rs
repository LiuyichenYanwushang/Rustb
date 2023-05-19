use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon};
///在 kane-mele 上加入了 altermagnetic 项来研究这一项对拓扑的影响
///
///采用的哈密顿量为
///
///$$ H(\bm k)=\sum_{i,\sg}\Delta (-1)^i c_{i\sg}^\dag c_{i\sg}+\sum_{\<ij\>} t c_{i\sg}^\dag c_{j\sg}+\lambda_{soc}\sum_{\<\<ij\>\>}\sum_{\sg=\not \sg^\prime}\nu_{ij} c_{i\sg}^\dag c_{j\sg^\prime} $$
///
///我们又加上了altermagnetic 项, 具体形式为 $d(\bm k)\sg_z$, 表现为
///
fn main() {
    let li:Complex<f64>=1.0*Complex::i();
    let delta=0.0;
    let t=-1.+0.0*li;
    let alter=0.25+0.0*li;
    //let soc=-1.0+0.0*li;
    let soc=-0.24+0.0*li;
    let rashba=-0.0+0.0*li;
    let dim_r:usize=2;
    let norb:usize=2;
    let lat=arr2(&[[3.0_f64.sqrt()/2.0,-0.5],[3.0_f64.sqrt()/2.0,0.5]]);
    let orb=arr2(&[[1.0/3.0,1.0/3.0],[2.0/3.0,2.0/3.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None,None);
    //onsite 
    model.set_onsite(arr1(&[delta,-delta]),0);
    //nearest hopping
    let R0:Array2::<isize>=arr2(&[[0,0],[-1,0],[0,-1]]);
    for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
        let R=R.to_owned();
        model.set_hop(t,0,1,&R,0);
    }
    let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
    //soc
    for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
        let R=R.to_owned();
        model.set_hop(soc*li,0,0,&R,3);
    }
    let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
    for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
        let R=R.to_owned();
        model.set_hop(soc*li,1,1,&R,3);
    }
    //rashba
    let R0:Array2::<isize>=arr2(&[[1,0],[-1,1],[0,-1]]);
    for  (i,R) in R0.axis_iter(Axis(0)).enumerate(){
        let R=R.to_owned();
        let r0=R.map(|x| *x as f64).dot(&model.lat);
        model.add_hop(rashba*li*r0[[1]],0,0,&R,1);
        model.add_hop(rashba*li*r0[[0]],0,0,&R,2);
    }
    
    let R0:Array2::<isize>=arr2(&[[-1,0],[1,-1],[0,1]]);
    for  (i,R) in R0.axis_iter(Axis(0)).enumerate(){
        let R=R.to_owned();
        let r0=R.map(|x| *x as f64).dot(&model.lat);
        model.add_hop(-rashba*li*r0[[1]],1,1,&R,1);
        model.add_hop(-rashba*li*r0[[0]],1,1,&R,2);
    }
    //altermagnetic
    let R0:Array2::<isize>=arr2(&[[0,-1],[1,-1],[1,0]]);
    for (i,R) in R0.axis_iter(Axis(0)).enumerate(){
        let R=R.to_owned();
        model.add_hop(alter*li,0,0,&R,3);
        model.add_hop(-alter*li,1,1,&R,3);
    }
    


    let nk:usize=1001;
    let path=[[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0]];
    let path=arr2(&path);
    let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
    let (eval,evec)=model.solve_all_parallel(&k_vec);
    let label=vec!["G","K","M","K'","G"];
    model.show_band(&path,&label,nk,"band");
    //开始计算超胞

    let super_model=model.make_supercell(&array![[1.0,1.0],[-1.0,1.0],]);
    let super_model=super_model.cut_piece(50,0);
    let path=[[0.0,0.0],[0.0,0.5],[0.0,1.0]];
    let path=arr2(&path);
    let label=vec!["G","M","G"];
    super_model.show_band(&path,&label,nk,"super");
}
