use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon};
use std::io::Write;
use std::fs::File;
fn main(){
    let li:Complex<f64>=1.0*Complex::i();
    let t1=1.0+0.0*li;
    let lm_so=0.2+0.0*li;
    let delta=0.0;
    let delta1=0.9+0.0*li;
    let delta2=0.22+0.0*li;
    let dim_r:usize=2;
    let norb:usize=2;
    let a0=3.0_f64.sqrt();
    let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]])*a0;
    let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
    let model=gen_model(dim_r,&lat,&orb,t1,lm_so,delta1,delta2,delta);
    //let delta1=Array1::linspace(0.0,1.0,100);
    //let delta2=Array1::linspace(0.0,1.0,100);

    //println!("{}",model.gen_ham(&array![0.5,0.5]));
    let nk:usize=1001;
    let path=array![[0.0,0.0],[2.0/3.0,1.0/3.0],[0.5,0.5],[1.0/3.0,2.0/3.0],[0.0,0.0],[1.0/2.0,1.0/2.0]];
    let label=vec!["G","K","M","K'","G","M"];
    model.show_band(&path,&label,nk,"examples/alterhexagonal");

    let edge_model=model.cut_piece(100,0);
    let path=array![[0.0,0.0],[0.0,1.0/2.0],[0.0,1.0]];
    let label=vec!["G","M","G"];
    edge_model.show_band(&path,&label,nk,"examples/alterhexagonal/edge_band");



    //画一下贝利曲率的分布
    let dir_1=arr1(&[1.0,0.0]);
    let dir_2=arr1(&[0.0,1.0]);
    let dir_3=arr1(&[0.0,1.0]);
    let T=0.0;
   //画一下贝利曲率的分布
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk]);
    let kvec=gen_kmesh(&kmesh);
    let E_min=-3.2;
    let E_max=3.6;
    let E_n=500;

    let nx:usize=50;
    let ny:usize=50;
    let delta1=Array1::<f64>::linspace(0.0,1.0,nx);
    let delta2=Array1::<f64>::linspace(0.0,1.0,ny);
    let mut use_conductivity=Array2::<f64>::zeros((nx,ny));
    for (i,d1) in delta1.iter().enumerate(){
        for (j,d2) in delta1.iter().enumerate(){

            let lat=arr2(&[[1.0,0.0],[0.5,3.0_f64.sqrt()/2.0]])*a0;
            let orb=arr2(&[[0.0,0.0],[1.0/3.0,1.0/3.0]]);
            let d1=d1+0.0*li;
            let d2=d2+0.0*li;
            let model=gen_model(dim_r,&lat,&orb,t1,lm_so,d1,d2,delta);
            let (E,dos)=model.dos(&kmesh,E_min,E_max,E_n,1e-3);
            let mut a=0.0;
            let mut mu=0.0;
            let dE=(E_max-E_min)/(E_n as f64);
            for (i,d) in dos.iter().enumerate(){
                a+=d*dE;
                if ((a-2.0).abs()<1e-2 && d < &1e-5) || a> 2.0{
                    mu=E[[i]];
                    break
                }
            }
            //println!("{}",mu);
            //plot!(E,dos,"examples/alterhexagonal/dos.pdf");

            
            let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,0.0,0.0,mu,0,1e-3);
            /*
            println!("{},{}",mu,conductivity/(2.0*PI));
            if (conductivity/2.0/PI-1.0).abs()<1e-2{
                use_conductivity[[i,j]]=1.0;
            }else{
                use_conductivity[[i,j]]=0.0;
            }
            */
            use_conductivity[[i,j]]=conductivity/2.0/PI;
        }
    }
    draw_heatmap(&use_conductivity,"./examples/alterhexagonal/heat_map1.pdf");
        
    /*
    let E_min=-1.0;
    let E_max=1.0;
    let E_n=2000;
    let og=0.0;
    let mu=Array1::linspace(E_min,E_max,E_n);
    let eta=1e-5;
    let conductivity=model.Hall_conductivity_mu(&kmesh,&dir_1,&dir_2,T,og,&mu,0,eta);
    let mut file=File::create("conductivity.dat").expect("Unable to BAND.dat");
    for i in 0..E_n{
        let mut s = String::new();
        let aa= format!("{:.6}", mu[[i]]);
        s.push_str(&aa);
        if conductivity[[i]]>=0.0 {
            s.push_str("     ");
        }else{
            s.push_str("    ");
        }
        let aa= format!("{:.6}", conductivity[[i]]);
        s.push_str(&aa);
        writeln!(file,"{}",s).expect("Can't write");
    }
    */
}


fn gen_model(dim_r:usize,lat:&Array2<f64>,orb:&Array2<f64>,t1:Complex<f64>,lm_so:Complex<f64>,delta1:Complex<f64>,delta2:Complex<f64>,delta:f64)->Model{

    let li:Complex<f64>=1.0*Complex::i();
    let mut model=Model::tb_model(dim_r,lat.clone(),orb.clone(),true,None,None);
    model.set_onsite(arr1(&[delta,delta]),3);
    //最近邻hopping
    model.add_hop(t1,0,1,&array![0,0],0);
    model.add_hop(t1,0,1,&array![-1,0],0);
    model.add_hop(t1,0,1,&array![0,-1],0);
    //Rashba
    let d1=model.orb.row(1).to_owned()-model.orb.row(0).to_owned();
    let d1=d1.dot(&model.lat);
    model.add_hop(-lm_so*li*d1[[0]],0,1,&array![0,0],2);
    model.add_hop(lm_so*li*d1[[1]],0,1,&array![0,0],1);
    let d2=model.orb.row(1).to_owned()-model.orb.row(0).to_owned()+&array![-1.0,0.0];
    let d2=d2.dot(&model.lat);
    model.add_hop(-lm_so*li*d2[[0]],0,1,&array![-1,0],2);
    model.add_hop(lm_so*li*d2[[1]],0,1,&array![-1,0],1);
    let d3=model.orb.row(1).to_owned()-model.orb.row(0).to_owned()+&array![0.0,-1.0];
    let d3=d3.dot(&model.lat);
    model.add_hop(-lm_so*li*d3[[0]],0,1,&array![0,-1],2);
    model.add_hop(lm_so*li*d3[[1]],0,1,&array![0,-1],1);
    //最后加上altermagnetism 项

    let theta=0.0*PI;
    let phi=0.*2.0*PI;
    //最近邻项
    //z方向
    model.add_hop(theta.cos()*delta1/2.0,0,1,&array![0,0],3);
    model.add_hop(theta.cos()*delta1/2.0,0,1,&array![-1,0],3);
    model.add_hop(-theta.cos()*delta1,0,1,&array![0,-1],3);

    //x方向
    model.add_hop(theta.sin()*phi.cos()*delta1/2.0,0,1,&array![0,0],1);
    model.add_hop(theta.sin()*phi.cos()*delta1/2.0,0,1,&array![-1,0],1);
    model.add_hop(-theta.sin()*phi.cos()*delta1,0,1,&array![0,-1],1);
    //y 方向
    model.add_hop(theta.sin()*phi.sin()*delta1/2.0,0,1,&array![0,0],2);
    model.add_hop(theta.sin()*phi.sin()*delta1/2.0,0,1,&array![-1,0],2);
    model.add_hop(-theta.sin()*phi.sin()*delta1,0,1,&array![0,-1],2);
    //次近邻项
    //z方向
    model.add_hop(theta.cos()*delta2,0,0,&array![1,0],3);
    model.add_hop(theta.cos()*delta2,1,1,&array![1,0],3);
    model.add_hop(-0.5*theta.cos()*delta2,0,0,&array![0,1],3);
    model.add_hop(-0.5*theta.cos()*delta2,1,1,&array![0,1],3);
    model.add_hop(-0.5*theta.cos()*delta2,0,0,&array![-1,1],3);
    model.add_hop(-0.5*theta.cos()*delta2,1,1,&array![-1,1],3);

    //x方向
    model.add_hop(theta.sin()*phi.cos()*delta2,0,0,&array![1,0],1);
    model.add_hop(theta.sin()*phi.cos()*delta2,1,1,&array![1,0],1);
    model.add_hop(-0.5*theta.sin()*phi.cos()*delta2,0,0,&array![0,1],1);
    model.add_hop(-0.5*theta.sin()*phi.cos()*delta2,1,1,&array![0,1],1);
    model.add_hop(-0.5*theta.sin()*phi.cos()*delta2,0,0,&array![-1,1],1);
    model.add_hop(-0.5*theta.sin()*phi.cos()*delta2,1,1,&array![-1,1],1);

    //y 方向

    model.add_hop(theta.sin()*phi.sin()*delta2,0,0,&array![1,0],2);
    model.add_hop(theta.sin()*phi.sin()*delta2,1,1,&array![1,0],2);
    model.add_hop(-0.5*theta.sin()*phi.sin()*delta2,0,0,&array![0,1],2);
    model.add_hop(-0.5*theta.sin()*phi.sin()*delta2,1,1,&array![0,1],2);
    model.add_hop(-0.5*theta.sin()*phi.sin()*delta2,0,0,&array![-1,1],2);
    model.add_hop(-0.5*theta.sin()*phi.sin()*delta2,1,1,&array![-1,1],2);
    model
}
