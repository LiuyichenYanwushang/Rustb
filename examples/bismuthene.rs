use ndarray::*;
use ndarray_linalg::*;
use Rustb::*;
use num_complex::Complex;
use std::f64::consts::PI;
fn main(){
    //!这个是 bismuthene的模型
    //!我们采用的轨道和 PHYSICAL REVIEW B 90, 085431 (2014) 这篇文章一致
    //!轨道的基函数为 $\\{s^A,p_x^A,p_y^A,p_z^A,S^B,p_x^B,p_y^B,p_z^B,s_H^A,s_H^B\\}$
    let li:Complex<f64>=1.0*Complex::i();
    let E_p=0.0;
    let E_s=-10.0;
    let E_sH=-3.0;
    let V_ss=1.0;
    let V_sp=0.0;
    let V_sh=0.0;
    let V_sg=0.0;//sigma 键
    let V_pi=3.0;//Pi 键, 次近邻
    let V_sg_2=0.0;//sigma 键, 次近邻
    let V_pi_2=0.0;//Pi 键

    let V_ss=Complex::new(V_ss,0.0);
    let V_sp=Complex::new(V_sp,0.0);
    let V_sh=Complex::new(V_sh,0.0);
    let V_sg=Complex::new(V_sg,0.0);//sigma 键
    let V_pi=Complex::new(V_pi,0.0);//Pi 键, 次近邻
    let V_sg_2=Complex::new(V_sg_2,0.0);//sigma 键, 次近邻
    let V_pi_2=Complex::new(V_pi_2,0.0);//Pi 键
    let h=0.00;
    let a=5.3;
    let c=10.0;
    let lat=array![[a,0.0,0.0],[a/2.0,a*3.0_f64.sqrt()/2.0,0.0],[0.0,0.0,c]];
    let orb=array![[1.0/3.0,1.0/3.0,0.0],[1.0/3.0,1.0/3.0,0.0],[1.0/3.0,1.0/3.0,0.0],[1.0/3.0,1.0/3.0,0.0],[1.0/3.0,1.0/3.0,0.0],[2.0/3.0,2.0/3.0,h/c],[2.0/3.0,2.0/3.0,h/c],[2.0/3.0,2.0/3.0,h/c],[2.0/3.0,2.0/3.0,h/c],[2.0/3.0,2.0/3.0,h/c]];
    let atom=array![[1.0/3.0,1.0/3.0,0.0],[2.0/3.0,2.0/3.0,h/c]];
    let atom_list=vec![5,5];
    let mut model=Model::tb_model(3,lat,orb,true,Some(atom),Some(atom_list));
    let theta=(a/3.0_f64.sqrt()).atan();
    let hop0=arr2(&[[0,0,0],[0,-1,0],[-1,0,0]]);
    model.set_onsite(array![E_s,E_p,E_p,E_p,E_s,E_p,E_p,E_p,E_sH,E_sH],0);

    
    for (r,R) in hop0.axis_iter(Axis(0)).enumerate(){
        for i in 0..4{
            for j in 0..4{
                let R=R.to_owned();
                let orb_j=model.orb.row(j+5).to_owned();
                let orb_i=model.orb.row(i).to_owned();
                let theta=(orb_j-orb_i+R.clone().map(|x| *x as f64)).dot(&model.lat);
                let theta=theta.clone()/theta.norm();
                println!("{},{},{}",i,j,theta);
                match (i,j){
                    (0,0) =>   {model.add_hop(V_ss ,i,j,&R,0);},
                    (0,_) =>   { model.add_hop(-V_sp*theta[[j-1]],i,j+5,&R,0); },
                    (_,0) =>   { model.add_hop(V_sp*theta[[i-1]],i,j+5,&R,0); },
                    (_,_) =>   {
                        model.add_hop(V_sg*theta[[i-1]]*theta[[j-1]],i,j+5,&R,0);//sigma键
                        let theta_0=theta.map(|x| (1.0-x.powi(2)).sqrt());
                        model.add_hop(V_pi*theta_0[[i-1]]*theta_0[[j-1]],i,j+5,&R,0);
                    },
                }
            }
        }
    }
    //开始加上自旋轨道耦合 lambda*L.S
    let path=array![[0.0,0.0,0.0],[2.0/3.0,1.0/3.0,0.0],[0.5,0.0,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","K","M","G"];
    let nk=301;
    model.show_band(&path,&label,nk,"examples/bismuthene");

}
