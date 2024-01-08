#![allow(warnings)]
use std::arch::x86_64::*;
use std::fs::create_dir_all;
use std::str::FromStr;
use std::ops::MulAssign;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use ndarray::linalg::kron;
use Rustb::*;
use std::f64::consts::PI;
use gnuplot::{Figure, Caption, Color,Fix,LineStyle,Solid,Auto,Dash,Major};
use gnuplot::AxesCommon;
use num_complex::Complex;
use std::fs::File;
use std::io::{BufWriter,Write};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use std::ops::AddAssign;
fn main() {
    let li=Complex::i();
    let zero_energy:f64=0.;
    let dim_r=3;
    let lat=array![[3.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,40.0]];
    let orb=array![[0.0,0.0,0.0]];
    let spin=false;
    let atom=orb.clone();
    let atom_list=vec![1];
    let mut model_1=Model::tb_model(dim_r,lat,orb,spin,Some(atom),Some(atom_list));
    let t1=-2.0+0.0*li;
    let t2=-1.5+0.0*li;
    let J=1.0;
    //model.set_onsite(&array![J],3);
    model_1.add_hop(t1,0,0,&array![-1,0,0],0);
    model_1.add_hop(t2,0,0,&array![0,-1,0],0);


    let lat=array![[4.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,40.0]];
    let orb=array![[0.0,0.0,0.5]];
    let spin=false;
    let atom=orb.clone();
    let atom_list=vec![1];
    let mut model_2=Model::tb_model(dim_r,lat,orb,spin,Some(atom),Some(atom_list));
    let t1=-2.0+0.0*li;
    let t2=-1.5+0.0*li;
    let J=1.0;
    //model.set_onsite(&array![J],3);
    model_2.add_hop(t2,0,0,&array![-1,0,0],0);
    model_2.add_hop(t1,0,0,&array![0,-1,0],0);


    let path=[[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.5,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","X","M","Y","G"];
    let path=arr2(&path);
    let nk=100;

    //开始转角扩胞

    let U1=array![[4.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,1.0]];
    let U2=array![[3.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,1.0]];
    let model_1=model_1.make_supercell(&U1);
    let model_2=model_2.make_supercell(&U2);
    let orb= concatenate![Axis(0), model_1.orb, model_2.orb];
    let atom= concatenate![Axis(0), model_1.atom, model_2.atom];
    let mut atom_list=model_1.atom_list.clone();
    let mut atom_list_1=model_2.atom_list.clone();
    atom_list.append(&mut atom_list_1);


    let U0=array![[(PI/4.0).cos(),-(PI/4.0).sin(),0.0],[(PI/4.0).sin(),(PI/4.0).cos(),0.0],[0.0,0.0,1.0]];
    let lat=U0.dot(&model_1.lat);
    let mut new_model=Model::tb_model(dim_r,lat.clone(),orb.clone(),true,Some(atom.clone()),Some(atom_list.clone()));
    let mut onsite=Array1::zeros(new_model.norb);
    for  i in 0..model_1.norb{
        onsite[[i]]=J;
        onsite[[i+model_1.norb]]=-J;
    }
    new_model.set_onsite(&onsite,3);
    for i in 0..model_1.norb{
        for j in 0..model_1.norb{
            for (r1,R1) in model_1.hamR.axis_iter(Axis(0)).enumerate(){
                if r1==0 && i>j{continue}
                let t1=model_1.ham[[r1,i,j]];
                new_model.add_hop(t1,i,j,&R1.to_owned(),0);
            }
            for (r2,R2) in model_2.hamR.axis_iter(Axis(0)).enumerate(){
                if r2==0 && i>j{continue}
                let t2=model_2.ham[[r2,i,j]];
                new_model.add_hop(t2,i+model_1.norb,j+model_1.norb,&R2.to_owned(),0);
            }
        }
    }
    new_model.show_band(&path,&label,nk,"./examples/alter_twist");

    let (evec,eval)=new_model.solve_onek(&array![0.0,0.0,0.0]);
    println!("{}",evec);
    
    let mut model_up=Model::tb_model(dim_r,lat.clone(),orb.clone(),false,Some(atom.clone()),Some(atom_list.clone()));
    let mut onsite=Array1::zeros(model_up.norb);
    for  i in 0..model_1.norb{
        onsite[[i]]=J;
        onsite[[i+model_1.norb]]=-J;
    }
    model_up.set_onsite(&onsite,0);
    for i in 0..model_1.norb{
        for j in 0..model_1.norb{
            for (r1,R1) in model_1.hamR.axis_iter(Axis(0)).enumerate(){
                if r1==0 && i>j{continue}
                let t1=model_1.ham[[r1,i,j]];
                model_up.add_hop(t1,i,j,&R1.to_owned(),0);
            }
            for (r2,R2) in model_2.hamR.axis_iter(Axis(0)).enumerate(){
                if r2==0 && i>j{continue}
                let t2=model_2.ham[[r2,i,j]];
                model_up.add_hop(t2,i+model_1.norb,j+model_1.norb,&R2.to_owned(),0);
            }
        }
    }
    let mut model_dn=Model::tb_model(dim_r,lat,orb,false,Some(atom),Some(atom_list));
    let mut onsite=Array1::zeros(model_dn.norb);
    for  i in 0..model_1.norb{
        onsite[[i]]=J;
        onsite[[i+model_1.norb]]=-J;
    }
    model_dn.set_onsite(&(-onsite),0);
    for i in 0..model_1.norb{
        for j in 0..model_1.norb{
            for (r1,R1) in model_1.hamR.axis_iter(Axis(0)).enumerate(){
                if r1==0 && i>j{continue}
                let t1=model_1.ham[[r1,i,j]];
                model_dn.add_hop(t1,i,j,&R1.to_owned(),0);
            }
            for (r2,R2) in model_2.hamR.axis_iter(Axis(0)).enumerate(){
                if r2==0 && i>j{continue}
                let t2=model_2.ham[[r2,i,j]];
                model_dn.add_hop(t2,i+model_1.norb,j+model_1.norb,&R2.to_owned(),0);
            }
        }
    }
    let (k_vec,k_dist,k_node)=model_up.k_path(&path,nk);
    let band_up=model_up.solve_band_all_parallel(&k_vec);
    let band_dn=model_dn.solve_band_all_parallel(&k_vec);
    let name=String::from_str("./examples/alter_twist/band_alter").unwrap();
    create_dir_all(name.clone()).expect("can't creat the file");
    let mut fg = Figure::new();
    let x:Vec<f64>=k_dist.to_vec();
    let axes=fg.axes2d();
    for i in 0..model_up.nsta{
        let y:Vec<f64>=band_up.slice(s![..,i]).to_owned().to_vec();
        axes.lines(&x, &y, &[Color("red"),LineStyle(Solid)]);
    }

    for i in 0..model_up.nsta{
        let y:Vec<f64>=band_dn.slice(s![..,i]).to_owned().to_vec();
        axes.lines(&x, &y, &[Color("blue"),LineStyle(Solid)]);
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

    let k_node=k_node.to_vec();
    let mut pdf_name=name.clone();
    pdf_name.push_str("/plot.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();

    //-------------开始计算霍尔电导--------------------


    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let mu=1.0;
    let spin=3;
    let eta=1e-2;
    //检测收敛性
    for i in (100..600).step_by(100){
        let k_mesh=arr1(&[i,i,1]);
        let cond_xx=conductivity(&new_model,&k_mesh,&dir_1,&dir_1,mu,0,eta);
        let cond_xy=conductivity(&new_model,&k_mesh,&dir_1,&dir_2,mu,0,eta);
        let cond_xyz=conductivity(&new_model,&k_mesh,&dir_1,&dir_2,mu,spin,eta);
        println!("{},{},{},{}",i,cond_xx,cond_xy,cond_xyz);
    }
    let nk=100;
    let k_mesh=arr1(&[nk,nk,1]);
    let n_theta=361;
    let theta=Array1::<f64>::linspace(0.0,2.0*PI,n_theta);
    let (result_x,result_y):(Vec<_>,Vec<_>)=theta.iter().map(|x| {
        let dir_1=arr1(&[x.cos(),x.sin(),0.0]);
        let dir_2=arr1(&[-x.sin(),x.cos(),0.0]);
        let cond_xx=conductivity(&new_model,&k_mesh,&dir_1,&dir_1,mu,0,eta);
        let cond_xy=conductivity(&new_model,&k_mesh,&dir_1,&dir_2,mu,spin,eta);
        (cond_xx,cond_xy)
    }).unzip();
    //let result=Array1::from_vec(result);
    let mut fg = Figure::new();
    let x:Vec<f64>=theta.iter().zip(result_x.iter()).map(|(x,y)| x.cos()*y.abs()).collect();
    let y:Vec<f64>=theta.iter().zip(result_x.iter()).map(|(x,y)| x.sin()*y.abs()).collect();
    let axes=fg.axes2d();
    axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
    axes.set_aspect_ratio(Fix(1.0));
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/alter_twist/plot_xx.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


    let mut fg = Figure::new();
    let x:Vec<f64>=theta.iter().zip(result_y.iter()).map(|(x,y)| x.cos()*y.abs()).collect();
    let y:Vec<f64>=theta.iter().zip(result_y.iter()).map(|(x,y)| x.sin()*y.abs()).collect();
    let axes=fg.axes2d();
    axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
    axes.set_aspect_ratio(Fix(1.0));
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/alter_twist/plot_xy.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


    let result:Vec<_>=result_x.iter().zip(result_y.iter()).map(|(x,y)| y/(x.powi(2)+y.powi(2)).sqrt()).collect();
    let mut fg = Figure::new();
    let x:Vec<f64>=theta.iter().zip(result.iter()).map(|(x,y)| x.cos()*y.abs()).collect();
    let y:Vec<f64>=theta.iter().zip(result.iter()).map(|(x,y)| x.sin()*y.abs()).collect();
    let axes=fg.axes2d();
    axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
    axes.set_aspect_ratio(Fix(1.0));
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/alter_twist/plot_odd.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


    let result:Vec<_>=result_x.iter().zip(result_y.iter()).map(|(x,y)| (x.powi(2)+y.powi(2)).sqrt()).collect();
    let mut fg = Figure::new();
    let x:Vec<f64>=theta.iter().zip(result.iter()).map(|(x,y)| x.cos()*y.abs()).collect();
    let y:Vec<f64>=theta.iter().zip(result.iter()).map(|(x,y)| x.sin()*y.abs()).collect();
    let axes=fg.axes2d();
    axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
    axes.set_aspect_ratio(Fix(1.0));
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/alter_twist/plot_all.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();


    let result:Vec<_>=result_x.iter().zip(result_y.iter()).map(|(x,y)| y/x).collect();
    let mut fg = Figure::new();
    let x:Vec<f64>=theta.iter().zip(result.iter()).map(|(x,y)| x.cos()*y.abs()).collect();
    let y:Vec<f64>=theta.iter().zip(result.iter()).map(|(x,y)| x.sin()*y.abs()).collect();
    let axes=fg.axes2d();
    axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
    axes.set_aspect_ratio(Fix(1.0));
    let mut pdf_name=String::new();
    pdf_name.push_str("./examples/alter_twist/plot_ratio.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();

        let mut name=String::new();
    name.push_str("./examples/alter_twist/data.txt");
    let mut file=File::create(name).expect("Unable to data.dat");
    let mut writer = BufWriter::new(file);
    for i in 0..n_theta{
        let mut s = String::new();
        let aa= format!("{:.6}", theta[[i]]);
        s.push_str(&aa);
        let bb:String=format!("{:.6}",result_x[i]);
        if result_x[i]>=0.0{
            s.push_str("    ");
        }else{
            s.push_str("   ");
        }
        s.push_str(&bb);
        let cc:String=format!("{:.6}",result_y[i]);
        if result_y[i]>=0.0{
            s.push_str("    ");
        }else{
            s.push_str("   ");
        }
        s.push_str(&cc);
        writer.write(s.as_bytes()).unwrap();
        writer.write(b"\n").unwrap();
    }
    let _=file;
}

fn conductivity_onek(model:&Model,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:f64,spin:usize,eta:f64)->f64{
    //!给定一个k点, 返回 $\Omega_n(\bm k)$
    //返回 $Omega_{n,\ap\bt}, \ve_{n\bm k}$
    let li:Complex<f64>=1.0*Complex::i();
    let (band,evec)=model.solve_onek(&k_vec);
    let mut v:Array3::<Complex<f64>>=model.gen_v(k_vec);
    let mut J:Array3::<Complex<f64>>=v.clone();
    if model.spin {
        let mut X:Array2::<Complex<f64>>=Array2::eye(model.nsta);
        let pauli:Array2::<Complex<f64>>= match spin{
            0=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,1.0+0.0*li]]),
            1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
            2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
            3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
            _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
        };
        X=kron(&pauli,&Array2::eye(model.norb));
        for i in 0..model.dim_r{
            let j=J.slice(s![i,..,..]).to_owned();
            let j=anti_comm(&X,&j)/2.0; //这里做反对易
            J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
            v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
        }
    }else{ 
        if spin !=0{
            println!("Warning, the model haven't got spin, so the spin input will be ignord");
        }
        for i in 0..model.dim_r{
            J.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_1[[i]],0.0));
            v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
        }
    };

    let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
    let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
    let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
    let A1=J.dot(&evec.clone().reversed_axes());
    let A1=&evec_conj.dot(&A1);
    let A2=v.dot(&evec.reversed_axes());
    let A2=&evec_conj.dot(&A2);
    let mut U0=Array2::<Complex<f64>>::zeros((model.nsta,model.nsta));
    for i in 0..model.nsta{
        for j in 0..model.nsta{
            U0[[i,j]]=1.0/(((mu-band[[i]]).powi(2)+eta.powi(2))*((mu-band[[j]]).powi(2)+eta.powi(2)))+0.0*li;
        }
    }
    //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
    //let mut omega_n=Array1::<f64>::zeros(model.nsta);
    let A1=A1*U0;
    let eta0=eta.powi(2);
    let A1=A1.axis_iter(Axis(0));
    let A2=A2.axis_iter(Axis(1));
    let omega_one=A1.zip(A2).map(|(x,y)| eta0*x.dot(&y).re).sum();
    omega_one 
}
fn conductivity(model:&Model,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:f64,spin:usize,eta:f64)->f64{
    let k_vec=gen_kmesh(&k_mesh);
    let nk=k_vec.len_of(Axis(0));
    let omega:Vec<f64>=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
        let omega_one=conductivity_onek(&model,&x.to_owned(),&dir_1,&dir_2,mu,spin,eta);
        omega_one
        }).collect();
    let omega=arr1(&omega);
    let V=model.lat.det().unwrap();
    omega.sum()/(nk as f64)*(2.0*PI).powi(3)/V
}
