#![allow(warnings)]
use std::cmp::Ordering;
use std::time::Instant;
use std::ops::AddAssign;
use std::fs::create_dir_all;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon,LineStyle,Solid,Font,MarginBottom,MarginLeft,MarginRight,Rotate,TextOffset,Axes2D};
use gnuplot::Major;
use Rustb::*;
fn main(){
    let li:Complex<f64>=1.0*Complex::i();
    let dim_r:usize=3;
    let norb:usize=2;
    let a0=1.0;
    let h=0.0;
    let lat=arr2(&[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])*a0;
    let orb=arr2(&[[0.5,0.0,0.0],[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.5,0.0]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None);
    let R0:Array2::<isize>=arr2(&[[0,0,0],[1,0,0],[0,-1,0],[1,-1,0]]);
    //------开始添加hopping----------------
    let lm=1.0+0.0*li;
    let t1=1.0+0.0*li;
    let t2=1.0+0.0*li;
    let r1=0.8+0.0*li;
    let r2=0.0+0.0*li;

    //SOC 项
    model.set_hop(li*lm,0,1,&R0.row(0).to_owned(),3);
    model.set_hop(li*lm,2,3,&R0.row(0).to_owned(),3);
    //hopping 项
    model.set_hop(t2,0,2,&R0.row(0).to_owned(),0); 
    model.set_hop(t2,1,3,&R0.row(0).to_owned(),0); 
    model.set_hop(t2,0,2,&R0.row(1).to_owned(),0); 
    model.set_hop(t2,1,3,&R0.row(1).to_owned(),0); 
    model.set_hop(t2,0,2,&R0.row(2).to_owned(),0); 
    model.set_hop(t2,1,3,&R0.row(2).to_owned(),0); 
    model.set_hop(t2,0,2,&R0.row(3).to_owned(),0); 
    model.set_hop(t2,1,3,&R0.row(3).to_owned(),0); 

    model.set_hop(-t1,1,2,&R0.row(0).to_owned(),0); 
    model.set_hop(-t1,0,3,&R0.row(0).to_owned(),0); 
    model.set_hop(t1,1,2,&R0.row(1).to_owned(),0); 
    model.set_hop(t1,0,3,&R0.row(1).to_owned(),0); 
    model.set_hop(t1,1,2,&R0.row(2).to_owned(),0); 
    model.set_hop(t1,0,3,&R0.row(2).to_owned(),0); 
    model.set_hop(-t1,1,2,&R0.row(3).to_owned(),0); 
    model.set_hop(-t1,0,3,&R0.row(3).to_owned(),0); 

    //次近邻项
    let R0:Array2::<isize>=arr2(&[[1,0,0],[0,-1,0],[1,-1,0]]);
    model.set_hop(-li*r1,1,1,&R0.row(0).to_owned(),2);
    model.set_hop(li*r1,3,3,&R0.row(0).to_owned(),2);
    model.set_hop(-li*r1,0,0,&R0.row(1).to_owned(),1);
    model.set_hop(li*r1,2,2,&R0.row(1).to_owned(),1);

    model.set_hop(-li*r2,0,0,&R0.row(0).to_owned(),2);
    model.set_hop(li*r2,2,2,&R0.row(0).to_owned(),2);
    model.set_hop(-li*r2,1,1,&R0.row(1).to_owned(),1);
    model.set_hop(li*r2,3,3,&R0.row(1).to_owned(),1);
    let mut model_bar_xy=model.clone();


    //开始计算体能带
    let path=array![[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","X","M","G"];
    let nk=1001;
    model.show_band(&path,&label,nk,"examples/Bi2F2/band");

    //-----算一下wilson loop 的结果-----------------------
    let n=301;
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let occ=vec![0,1,2,3];
    show_wilson_loop(&model,&dir_1,&dir_2,&occ,n,n,"examples/Bi2F2/wcc.pdf");


    let nk:usize=501;
    let green=surf_Green::from_Model(&model,0,1e-3,None);
    let E_min=-1.0;
    let E_max=1.0;
    let E_n=nk.clone();
    let path=[[-0.5,-0.5],[0.0,0.0],[0.5,0.5]];
    let path=arr2(&path);
    let label=vec!["X","{/symbol G}","X"];
    green.show_surf_state("examples/Bi2F2/surf",&path,&label,nk,E_min,E_max,E_n,0);

    let super_model=model.cut_piece(50,0);
    let path=[[0.0,-0.5,0.0],[0.0,0.0,0.0],[0.0,0.5,0.0]];
    let path=arr2(&path);
    let label=vec!["M","G","M"];
    super_model.show_band(&path,&label,nk,"examples/Bi2F2/super_band");

    //接下来我们计算altermagnetism 下的结果
    
    let J=0.05;
    let t3=0.0;
    let t4=0.2;
    let t5=0.05;
    let t6=0.0;
    let model_xy=add_altermagnetism_1(model.clone(),J,true);
    //let model_xy=add_altermagnetism_3(model,J,t3,t4,t5,t6,true);
    show_alter(&model_xy,"examples/Bi2F2/xy/alter");

    let path=array![[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","X","M","G"];
    let nk=1001;
    model_xy.show_band(&path,&label,nk,"examples/Bi2F2/xy/band");

    let nk:usize=501;
    let green=surf_Green::from_Model(&model_xy,0,1e-3,None);
    let E_min=-1.0;
    let E_max=1.0;
    let E_n=nk.clone();
    let path=[[-0.5,-0.0],[0.0,0.0],[0.5,0.0]];
    let path=arr2(&path);
    let label=vec!["X","G","X"];
    green.show_surf_state("examples/Bi2F2/xy/surf",&path,&label,nk,E_min,E_max,E_n,0);

    let num=20;
    let name="examples/Bi2F2/xy";

    let start = Instant::now();
    cut(&model_xy,num,1,name);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("solve xy cut took {} seconds", duration.as_secs_f64());   // 输出执行时间

    let start = Instant::now();
    cut(&model_xy,num,2,name);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("solve xy cut took {} seconds", duration.as_secs_f64());   // 输出执行时间

    //-------------------- 沿着 -xy 方向的Neel 矢量


    let J=0.05;
    let t3=0.0;
    let t4=0.2;
    let t5=0.05;
    let t6=0.0;
    let model_xy=add_altermagnetism_1(model.clone(),J,false);
    //let model_xy=add_altermagnetism_3(model,J,t3,t4,t5,t6,false);
    show_alter(&model_xy,"examples/Bi2F2/bar_xy/alter");

    let path=array![[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","X","M","G"];
    let nk=1001;
    model_xy.show_band(&path,&label,nk,"examples/Bi2F2/bar_xy/band");

    let nk:usize=501;
    let green=surf_Green::from_Model(&model_xy,0,1e-3,None);
    let E_min=-1.0;
    let E_max=1.0;
    let E_n=nk.clone();
    let path=[[-0.5,-0.0],[0.0,0.0],[0.5,0.0]];
    let path=arr2(&path);
    let label=vec!["X","G","X"];
    green.show_surf_state("examples/Bi2F2/bar_xy/surf",&path,&label,nk,E_min,E_max,E_n,0);

    let num=20;
    let name="examples/Bi2F2/bar_xy";
    let start = Instant::now();
    cut(&model_xy,num,1,name);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("solve xy cut took {} seconds", duration.as_secs_f64());   // 输出执行时间

    let start = Instant::now();
    cut(&model_xy,num,2,name);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("solve xy cut took {} seconds", duration.as_secs_f64());   // 输出执行时间


}



fn  write_txt(data:Array2<f64>,output:&str)-> std::io::Result<()>{
    use std::fs::File;
    use std::io::Write;
    let mut file=File::create(output).expect("Unable to BAND.dat");
    let n=data.len_of(Axis(0));
    let s=data.len_of(Axis(1));
    let mut s0=String::new();
    for i in 0..n{
        for j in 0..s{
            if data[[i,j]]>=0.0{
                s0.push_str("     ");
            }else{
                s0.push_str("    ");
            }
            let aa= format!("{:.6}", data[[i,j]]);
            s0.push_str(&aa);
        }
        s0.push_str("\n");
    }
    writeln!(file,"{}",s0)?;
    Ok(())
}

fn  write_txt_1(data:Array1<f64>,output:&str)-> std::io::Result<()>{
    use std::fs::File;
    use std::io::Write;
    let mut file=File::create(output).expect("Unable to BAND.dat");
    let n=data.len_of(Axis(0));
    let mut s0=String::new();
    for i in 0..n{
        if data[[i]]>=0.0{
            s0.push_str(" ");
        }
        let aa= format!("{:.6}\n", data[[i]]);
        s0.push_str(&aa);
    }
    writeln!(file,"{}",s0)?;
    Ok(())
}



fn add_altermagnetism_1(mut model:Model,J:f64,direction_xy:bool)->Model{
    //接下来开始加入 altermagnetism 项
    //首先, 我们设定磁矩为 J, 加在 onsite 项上, 看能否得到altermagnetism, 目前加入的是 (cos kx- cos ky)(a sx+b xy)
    let J=Complex::new(J,0.0);
    if direction_xy{
        model.add_hop(J,0,0,&array![1,0,0],1);
        model.add_hop(J,1,1,&array![1,0,0],1);
        model.add_hop(-J,2,2,&array![1,0,0],1);
        model.add_hop(-J,3,3,&array![1,0,0],1);
        model.add_hop(-J,0,0,&array![0,1,0],1);
        model.add_hop(-J,1,1,&array![0,1,0],1);
        model.add_hop(J,2,2,&array![0,1,0],1);
        model.add_hop(J,3,3,&array![0,1,0],1);

        model.add_hop(J,0,0,&array![1,0,0],2);
        model.add_hop(J,1,1,&array![1,0,0],2);
        model.add_hop(-J,2,2,&array![1,0,0],2);
        model.add_hop(-J,3,3,&array![1,0,0],2);
        model.add_hop(-J,0,0,&array![0,1,0],2);
        model.add_hop(-J,1,1,&array![0,1,0],2);
        model.add_hop(J,2,2,&array![0,1,0],2);
        model.add_hop(J,3,3,&array![0,1,0],2);
        return model
    }else{
        model.add_hop(J,0,0,&array![1,0,0],1);
        model.add_hop(J,1,1,&array![1,0,0],1);
        model.add_hop(-J,2,2,&array![1,0,0],1);
        model.add_hop(-J,3,3,&array![1,0,0],1);
        model.add_hop(-J,0,0,&array![0,1,0],1);
        model.add_hop(-J,1,1,&array![0,1,0],1);
        model.add_hop(J,2,2,&array![0,1,0],1);
        model.add_hop(J,3,3,&array![0,1,0],1);
        let J=-J;
        model.add_hop(-J,0,0,&array![1,0,0],2);
        model.add_hop(-J,1,1,&array![1,0,0],2);
        model.add_hop(J,2,2,&array![1,0,0],2);
        model.add_hop(J,3,3,&array![1,0,0],2);
        model.add_hop(J,0,0,&array![0,1,0],2);
        model.add_hop(J,1,1,&array![0,1,0],2);
        model.add_hop(-J,2,2,&array![0,1,0],2);
        model.add_hop(-J,3,3,&array![0,1,0],2);
        return model
    }
}
fn add_altermagnetism_2(mut model:Model,J:f64,t3:f64,t4:f64,direction_xy:bool)->Model{
    //目前, 根据magneticTB, 我们能够得到的altermagnetism 可以考虑为如下形式:
    let li:Complex<f64>=1.0*Complex::i();
    let t3=Complex::new(t3,0.0);
    let t4=Complex::new(t4,0.0);
    if direction_xy{
        model.set_onsite(&array![J,J,-J,-J],1);
        model.set_onsite(&array![J,J,-J,-J],2);
        let R0:Array2::<isize>=arr2(&[[1,0,0],[0,-1,0]]);
        //hopping 项
        model.add_hop(t3,0,0,&R0.row(0).to_owned(),0);
        model.add_hop(t4,1,1,&R0.row(0).to_owned(),0);
        model.add_hop(t3,2,2,&R0.row(0).to_owned(),0);
        model.add_hop(t4,3,3,&R0.row(0).to_owned(),0);
        model.add_hop(-t3,0,0,&R0.row(1).to_owned(),0);
        model.add_hop(-t4,1,1,&R0.row(1).to_owned(),0);
        model.add_hop(-t3,2,2,&R0.row(1).to_owned(),0);
        model.add_hop(-t4,3,3,&R0.row(1).to_owned(),0);
        return model
    }else{
        model.set_onsite(&array![J,J,-J,-J],1);
        model.set_onsite(&array![-J,-J,J,J],2);
        let R0:Array2::<isize>=arr2(&[[1,0,0],[0,-1,0]]);
        //hopping 项
        model.add_hop(t3,0,0,&R0.row(0).to_owned(),0);
        model.add_hop(t4,1,1,&R0.row(0).to_owned(),0);
        model.add_hop(t3,2,2,&R0.row(0).to_owned(),0);
        model.add_hop(t4,3,3,&R0.row(0).to_owned(),0);
        model.add_hop(-t3,0,0,&R0.row(1).to_owned(),0);
        model.add_hop(-t4,1,1,&R0.row(1).to_owned(),0);
        model.add_hop(-t3,2,2,&R0.row(1).to_owned(),0);
        model.add_hop(-t4,3,3,&R0.row(1).to_owned(),0);
        return model
    }
}
fn add_altermagnetism_3(mut model:Model,J:f64,t3:f64,t4:f64,t5:f64,t6:f64,direction_xy:bool)->Model{
    let li:Complex<f64>=1.0*Complex::i();
    let t3=Complex::new(t3,0.0);
    let t4=Complex::new(t4,0.0);
    if direction_xy{
        model.set_onsite(&array![J,-J,-J,J],1);
        model.set_onsite(&array![J,-J,-J,J],2);
        //model.set_onsite(&array![J,J,-J,-J],1);
        //model.set_onsite(&array![J,J,-J,-J],2);
        let R0:Array2::<isize>=arr2(&[[0,0,0],[1,0,0],[0,-1,0],[1,-1,0]]);
        model.add_hop(t3,0,2,&R0.row(0).to_owned(),1);
        model.add_hop(-t3,0,2,&R0.row(1).to_owned(),1);
        model.add_hop(-t3,0,2,&R0.row(2).to_owned(),1);
        model.add_hop(t3,0,2,&R0.row(3).to_owned(),1);
        model.add_hop(-t3,1,3,&R0.row(0).to_owned(),1);
        model.add_hop(t3,1,3,&R0.row(1).to_owned(),1);
        model.add_hop(t3,1,3,&R0.row(2).to_owned(),1);
        model.add_hop(-t3,1,3,&R0.row(3).to_owned(),1);


        model.add_hop(t3,0,2,&R0.row(0).to_owned(),2);
        model.add_hop(-t3,0,2,&R0.row(1).to_owned(),2);
        model.add_hop(-t3,0,2,&R0.row(2).to_owned(),2);
        model.add_hop(t3,0,2,&R0.row(3).to_owned(),2);
        model.add_hop(-t3,1,3,&R0.row(0).to_owned(),2);
        model.add_hop(t3,1,3,&R0.row(1).to_owned(),2);
        model.add_hop(t3,1,3,&R0.row(2).to_owned(),2);
        model.add_hop(-t3,1,3,&R0.row(3).to_owned(),2);

        model.add_hop(t4,1,2,&R0.row(0).to_owned(),1);
        model.add_hop(t4,1,2,&R0.row(1).to_owned(),1);
        model.add_hop(t4,1,2,&R0.row(2).to_owned(),1);
        model.add_hop(t4,1,2,&R0.row(3).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(0).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(1).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(2).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(3).to_owned(),1);


        model.add_hop(t4,1,2,&R0.row(0).to_owned(),2);
        model.add_hop(t4,1,2,&R0.row(1).to_owned(),2);
        model.add_hop(t4,1,2,&R0.row(2).to_owned(),2);
        model.add_hop(t4,1,2,&R0.row(3).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(0).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(1).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(2).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(3).to_owned(),2);


        model.add_hop(-li*t5,1,2,&R0.row(0).to_owned(),3);
        model.add_hop(li*t5,1,2,&R0.row(1).to_owned(),3);
        model.add_hop(li*t5,1,2,&R0.row(2).to_owned(),3);
        model.add_hop(-li*t5,1,2,&R0.row(3).to_owned(),3);
        model.add_hop(-li*t6,0,3,&R0.row(0).to_owned(),3);
        model.add_hop(li*t6,0,3,&R0.row(1).to_owned(),3);
        model.add_hop(li*t6,0,3,&R0.row(2).to_owned(),3);
        model.add_hop(-li*t6,0,3,&R0.row(3).to_owned(),3);

        return model
    }else{
        model.set_onsite(&array![J,-J,-J,J],1);
        model.set_onsite(&array![-J,J,J,-J],2);
        //model.set_onsite(&array![J,J,-J,-J],1);
        //model.set_onsite(&array![-J,-J,J,J],2);
        let R0:Array2::<isize>=arr2(&[[0,0,0],[1,0,0],[0,-1,0],[1,-1,0]]);
        //hopping 项
        model.add_hop(t3,0,2,&R0.row(0).to_owned(),1);
        model.add_hop(-t3,0,2,&R0.row(1).to_owned(),1);
        model.add_hop(-t3,0,2,&R0.row(2).to_owned(),1);
        model.add_hop(t3,0,2,&R0.row(3).to_owned(),1);
        model.add_hop(-t3,1,3,&R0.row(0).to_owned(),1);
        model.add_hop(t3,1,3,&R0.row(1).to_owned(),1);
        model.add_hop(t3,1,3,&R0.row(2).to_owned(),1);
        model.add_hop(-t3,1,3,&R0.row(3).to_owned(),1);
        let t3=-t3;
        model.add_hop(t3,0,2,&R0.row(0).to_owned(),2);
        model.add_hop(-t3,0,2,&R0.row(1).to_owned(),2);
        model.add_hop(-t3,0,2,&R0.row(2).to_owned(),2);
        model.add_hop(t3,0,2,&R0.row(3).to_owned(),2);
        model.add_hop(-t3,1,3,&R0.row(0).to_owned(),2);
        model.add_hop(t3,1,3,&R0.row(1).to_owned(),2);
        model.add_hop(t3,1,3,&R0.row(2).to_owned(),2);
        model.add_hop(-t3,1,3,&R0.row(3).to_owned(),2);

        model.add_hop(t4,1,2,&R0.row(0).to_owned(),1);
        model.add_hop(t4,1,2,&R0.row(1).to_owned(),1);
        model.add_hop(t4,1,2,&R0.row(2).to_owned(),1);
        model.add_hop(t4,1,2,&R0.row(3).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(0).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(1).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(2).to_owned(),1);
        model.add_hop(t4,0,3,&R0.row(3).to_owned(),1);
        let t4=-t4;
        model.add_hop(t4,1,2,&R0.row(0).to_owned(),2);
        model.add_hop(t4,1,2,&R0.row(1).to_owned(),2);
        model.add_hop(t4,1,2,&R0.row(2).to_owned(),2);
        model.add_hop(t4,1,2,&R0.row(3).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(0).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(1).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(2).to_owned(),2);
        model.add_hop(t4,0,3,&R0.row(3).to_owned(),2);



        model.add_hop(-li*t5,1,2,&R0.row(0).to_owned(),3);
        model.add_hop(li*t5,1,2,&R0.row(1).to_owned(),3);
        model.add_hop(li*t5,1,2,&R0.row(2).to_owned(),3);
        model.add_hop(-li*t5,1,2,&R0.row(3).to_owned(),3);
        model.add_hop(-li*t6,0,3,&R0.row(0).to_owned(),3);
        model.add_hop(li*t6,0,3,&R0.row(1).to_owned(),3);
        model.add_hop(li*t6,0,3,&R0.row(2).to_owned(),3);
        model.add_hop(-li*t6,0,3,&R0.row(3).to_owned(),3);
        return model
    }

}

fn show_wilson_loop(model:&Model,dir_1:&Array1<f64>,dir_2:&Array1<f64>,occ:&Vec<usize>,n1:usize,n2:usize,name:&str){
    let wcc=model.wannier_centre(occ,&array![0.0,0.0,0.0],dir_1,dir_2,n1,n2);
    let nocc=occ.len();
    let mut fg = Figure::new();
    let x:Vec<f64>=Array1::<f64>::linspace(0.0,1.0,n2).to_vec();
    let axes=fg.axes2d();
    for j in -1..2{
        for i in 0..nocc{
            let a=wcc.row(i).to_owned()+(j as f64)*2.0*PI;
            let y:Vec<f64>=a.to_vec();
            axes.points(&x, &y, &[Color("black"),gnuplot::PointSymbol('O'),gnuplot::PointSize(0.2)]);
        }
    }
    let axes=axes.set_x_range(Fix(0.0), Fix(1.0));
    let axes=axes.set_y_range(Fix(0.0), Fix(2.0*PI));
    let show_ticks=vec![Major(0.0,Fix("0")),Major(0.5,Fix("π")),Major(1.0,Fix("2π"))];
    axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",32.0)]);
    let show_ticks=vec![Major(0.0,Fix("0")),Major(PI,Fix("π")),Major(2.0*PI,Fix("2π"))];
    axes.set_y_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",32.0)]);
    axes.set_x_label("k_x",&[Font("Times New Roman",32.0),TextOffset(0.0,-0.5)]);
    axes.set_y_label("WCC",&[Font("Times New Roman",32.0),Rotate(90.0),TextOffset(-1.0,0.0)]);
    //axes.set_margins(&[MarginLeft(0.1),MarginBottom(0.2),MarginRight(0.0)]);
    axes.set_aspect_ratio(Fix(0.8));
    fg.set_terminal("pdfcairo", name);
    fg.show();
}

fn calculate_M(model:&Model){
        //开始计算三个方向的占据态的磁矩
        let kvec=gen_kmesh(&array![101,101,1]);
        let (band,evec)=model.solve_all_parallel(&kvec);
        let I0=Array1::<Complex<f64>>::ones(model.norb);
        let I0=Array2::from_diag(&I0);
        let sx=array![[0.0,1.0],[1.0,0.0]];
        let sx=sx.mapv(|x| Complex::new(x,0.0));
        let sy=array![[0.0,-1.0],[1.0,0.0]];
        let sy=sy.mapv(|x| Complex::new(0.0,x));
        let sz=array![[1.0,0.0],[0.0,-1.0]];
        let sz=sz.mapv(|x| Complex::new(x,0.0));
        let sx=kron(&sx,&I0);
        let sy=kron(&sy,&I0);
        let sz=kron(&sz,&I0);
        println!("{}",sx);
        let nk=kvec.len_of(Axis(0));
        let evec_conj=evec.clone().mapv(|x| x.conj());
        let band_sx:Vec<_>=evec_conj.outer_iter().zip(evec.outer_iter()).map(|(x,y)| {
            let tmp=x.dot(&sx.dot(&y.t()));
            let a=tmp.diag().to_owned();
            a}).collect();
        let band_sy:Vec<_>=evec_conj.outer_iter().zip(evec.outer_iter()).map(|(x,y)| {
            let tmp=x.dot(&sy.dot(&y.t()));
            let a=tmp.diag().to_owned();
            a}).collect();
        let band_sz:Vec<_>=evec_conj.outer_iter().zip(evec.outer_iter()).map(|(x,y)| {
            let tmp=x.dot(&sz.dot(&y.t()));
            let a=tmp.diag().to_owned();
            a}).collect();
        let band_s0:Vec<_>=evec_conj.outer_iter().zip(evec.outer_iter()).map(|(x,y)| {
            let tmp=x.dot(&y.t());
            let a=tmp.diag().to_owned();
            a}).collect();
        let band_sx=Array2::<Complex<f64>>::from_shape_vec((nk,model.nsta),band_sx.into_iter().flatten().collect()).unwrap();
        let band_sy=Array2::<Complex<f64>>::from_shape_vec((nk,model.nsta),band_sy.into_iter().flatten().collect()).unwrap();
        let band_sz=Array2::<Complex<f64>>::from_shape_vec((nk,model.nsta),band_sz.into_iter().flatten().collect()).unwrap();
        let band_s0=Array2::<Complex<f64>>::from_shape_vec((nk,model.nsta),band_s0.into_iter().flatten().collect()).unwrap();
        let band_sx=band_sx.slice(s![..,0..4]).to_owned();
        let band_sy=band_sy.slice(s![..,0..4]).to_owned();
        let band_sz=band_sz.slice(s![..,0..4]).to_owned();
        let band_s0=band_s0.slice(s![..,0..4]).to_owned();
        println!("{:.3},{:.3},{:.3},{:.3}",band_sx.sum()/(nk as f64)/4.0,band_sy.sum()/(nk as f64)/4.0,band_sz.sum()/(nk as f64)/4.0,band_s0.sum()/(nk as f64)/4.0);
}
fn calculate_C(model:&Model,n:usize)->f64{
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let occ=vec![0,1,2,3];
    let C=model.berry_flux(&occ,&array![0.0,0.0,0.0],&dir_1,&dir_2,n,n).sum()/2.0/PI/4.0;
    C
}

fn cut(model:&Model,num:usize,cut_type:usize,name:&str){
    //普通的切法
    let new_model=match cut_type{
        0=>{
            let model_1=model.cut_piece(num,0);
            let new_model=model_1.cut_piece(num,1);
            new_model
        },
        1=>{

            let mut new_model=model.cut_dot(num,4,Some(vec![0,1]));
            let mut del_atom=Vec::new();
            let num0=num as f64;
            for (i,a) in new_model.atom_position().outer_iter().enumerate(){
                if a[[0]].abs()<1e-3 || (a[[0]]-num0/(num0+1.0)).abs() < 1e-3{
                    del_atom.push(i)
                }else if a[[1]] > (num0-1.0)/(num0+1.0){
                    del_atom.push(i)
                }
            }
            new_model.remove_atom(&del_atom);
            new_model
        },
        2=>{
            let mut new_model=model.cut_dot(num,4,Some(vec![0,1]));
            new_model
        },
        _=> todo!(),
            
    };
    let mut s=0;
    let start = Instant::now();
    let (band,evec)=new_model.solve_range_onek(&arr1(&[0.0,0.0,0.0]),(-0.5,0.5),1e-8);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("solve_band_all took {} seconds", duration.as_secs_f64());   // 输出执行时间
    let show_evec=evec.to_owned().map(|x| x.norm_sqr());
    let mut size=Array2::<f64>::zeros((new_model.nsta,new_model.natom));
    let norb=new_model.norb;
    let nresult=evec.shape()[0];
    for i in 0..nresult{
        let mut s=0;
        for j in 0..new_model.natom{
            for k in 0..new_model.atoms[j].norb(){
                size[[i,j]]+=show_evec[[i,s]]+show_evec[[i,s+new_model.norb]];
                s+=1;
            }
        }
    }

    let show_str=new_model.atom_position().dot(&model.lat);
    let show_str=show_str.slice(s![..,0..2]).to_owned();
    let show_size=size.row(new_model.norb).to_owned();
    let dir_name=match cut_type{
        0=>{ 
            let mut dir_name=String::new();
            dir_name.push_str(name);
            dir_name.push_str("/normal");
            dir_name
        },
        1=>{ 
            let mut dir_name=String::new();
            dir_name.push_str(name);
            dir_name.push_str("/atom_1");
            dir_name
        },
        2=>{ 
            let mut dir_name=String::new();
            dir_name.push_str(name);
            dir_name.push_str("/atom_2");
            dir_name
        },
        _=>todo!(),
    };

    println!("{}",dir_name);
    create_dir_all(&dir_name).expect("can't creat the file");
    let mut band_name=String::new();
    band_name.push_str(&dir_name);
    band_name.push_str("/band.txt");
    write_txt_1(band,&band_name);
    let mut size_name=String::new();
    size_name.push_str(&dir_name);
    size_name.push_str("/evec.txt");
    write_txt(size,&size_name);
    let mut structure_name=String::new();
    structure_name.push_str(&dir_name);
    structure_name.push_str("/structure.txt");
    write_txt(show_str,&structure_name);
}


fn show_alter(model:&Model,name:&str){
    let nk=1001;
    let path=array![[-0.5,0.25,0.0],[0.5,0.25,0.0]];
    let (kvec,kdist,knode)=model.k_path(&path,nk);
    let (eval,evec):(Array2<f64>,Array3<Complex<f64>>)=model.solve_all_parallel(&kvec);
    let label=vec!["X1","X2"];
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
    pdf_name.push_str(name);
    create_dir_all(&pdf_name).expect("can't creat the file");
    pdf_name.push_str("/plot.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
}
