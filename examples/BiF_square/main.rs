#![allow(warnings)]
use std::fs::create_dir_all;
use std::time::Instant;
use std::ops::AddAssign;
use ndarray::*;
use ndarray::linalg::kron;
use ndarray_linalg::*;
use num_complex::Complex;
use std::f64::consts::PI;
use gnuplot::{Color,Fix,Figure,AxesCommon,LineStyle,Solid};
use gnuplot::Major;
use Rustb::*;
fn main(){
    let li:Complex<f64>=1.0*Complex::i();
    let t_sg=1.0+0.0*li;
    let t_pi=-0.2+0.0*li;
    let lm=1.2+0.0*li;
    let t_eff=0.2+0.0*li;
    let delta=0.0;
    let dim_r:usize=3;
    let norb:usize=2;
    let a0=2.0;
    let h=0.1;
    let lat=arr2(&[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,3.0]])*a0;
    let orb=arr2(&[[0.5,0.0,0.5+h],[0.5,0.0,0.5+h],[0.0,0.5,0.5-h],[0.0,0.5,0.5-h]]);
    let mut model=Model::tb_model(dim_r,lat,orb,true,None);
    let R0:Array2::<isize>=arr2(&[[0,0,0],[1,0,0],[0,-1,0],[1,-1,0]]);
    let cos_t2=1.0-h.powi(2)/2.0;
    let t_sg=t_sg*cos_t2;
    //我们只考虑最近邻
    model.add_hop(-t_sg,0,2,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi,0,2,&R0.row(0).to_owned(),0);
    model.add_hop(t_sg,0,3,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi,0,3,&R0.row(0).to_owned(),0);
    model.add_hop(t_sg,1,2,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi,1,2,&R0.row(0).to_owned(),0);
    model.add_hop(-t_sg,1,3,&R0.row(0).to_owned(),0);
    model.add_hop(t_pi,1,3,&R0.row(0).to_owned(),0);
    //第二个
    model.add_hop(-t_sg,0,2,&R0.row(1).to_owned(),0);
    model.add_hop(t_pi,0,2,&R0.row(1).to_owned(),0);
    model.add_hop(-t_sg,0,3,&R0.row(1).to_owned(),0);
    model.add_hop(-t_pi,0,3,&R0.row(1).to_owned(),0);
    model.add_hop(-t_sg,1,2,&R0.row(1).to_owned(),0);
    model.add_hop(-t_pi,1,2,&R0.row(1).to_owned(),0);
    model.add_hop(-t_sg,1,3,&R0.row(1).to_owned(),0);
    model.add_hop(t_pi,1,3,&R0.row(1).to_owned(),0);
    //第三个
    model.add_hop(-t_sg,0,2,&R0.row(2).to_owned(),0);
    model.add_hop(t_pi,0,2,&R0.row(2).to_owned(),0);
    model.add_hop(-t_sg,0,3,&R0.row(2).to_owned(),0);
    model.add_hop(-t_pi,0,3,&R0.row(2).to_owned(),0);
    model.add_hop(-t_sg,1,2,&R0.row(2).to_owned(),0);
    model.add_hop(-t_pi,1,2,&R0.row(2).to_owned(),0);
    model.add_hop(-t_sg,1,3,&R0.row(2).to_owned(),0);
    model.add_hop(t_pi,1,3,&R0.row(2).to_owned(),0);
    //第四个
    model.add_hop(-t_sg,0,2,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi,0,2,&R0.row(3).to_owned(),0);
    model.add_hop(t_sg,0,3,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi,0,3,&R0.row(3).to_owned(),0);
    model.add_hop(t_sg,1,2,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi,1,2,&R0.row(3).to_owned(),0);
    model.add_hop(-t_sg,1,3,&R0.row(3).to_owned(),0);
    model.add_hop(t_pi,1,3,&R0.row(3).to_owned(),0);
    //开始加上自旋轨道耦合
    model.add_hop(lm*li,0,1,&array![0,0,0],3);
    model.add_hop(lm*li,2,3,&array![0,0,0],3);
    //最后一项
    model.add_hop(li*t_eff,0,3,&R0.row(0).to_owned(),2);
    model.add_hop(-li*t_eff,1,2,&R0.row(0).to_owned(),2);
    model.add_hop(li*t_eff,0,3,&R0.row(1).to_owned(),1);
    model.add_hop(-li*t_eff,1,2,&R0.row(1).to_owned(),1);
    model.add_hop(-li*t_eff,0,3,&R0.row(2).to_owned(),1);
    model.add_hop(li*t_eff,1,2,&R0.row(2).to_owned(),1);
    model.add_hop(-li*t_eff,0,3,&R0.row(3).to_owned(),2);
    model.add_hop(li*t_eff,1,2,&R0.row(3).to_owned(),2);
    //开始计算体能带
    let path=array![[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","X","M","G"];
    let nk=1001;
    model.show_band(&path,&label,nk,"examples/BiF_square/band");
    //

    //首先算一下wilson loop

    let n=301;
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let occ=vec![0,1,2,3];
    let wcc=model.wannier_centre(&occ,&array![0.0,0.0,0.0],&dir_1,&dir_2,n,n);
    let nocc=occ.len();


    let mut fg = Figure::new();
    let x:Vec<f64>=Array1::<f64>::linspace(0.0,1.0,n).to_vec();
    let axes=fg.axes2d();
    for j in -1..2{
        for i in 0..nocc{
            let a=wcc.row(i).to_owned()+(j as f64)*2.0*PI;
            let y:Vec<f64>=a.to_vec();
            axes.points(&x, &y, &[Color("black"),gnuplot::PointSymbol('O')]);
        }
    }
    let axes=axes.set_x_range(Fix(0.0), Fix(1.0));
    let axes=axes.set_y_range(Fix(0.0), Fix(2.0*PI));
    let show_ticks=vec![Major(0.0,Fix("0")),Major(0.5,Fix("π")),Major(1.0,Fix("2π"))];
    axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
    let mut pdf_name=String::new();
    pdf_name.push_str("examples/BiF_square/wcc.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();

    //接下来开始加入 altermagnetism 项
    //首先, 我们设定磁矩为 J, 加在 onsite 项上, 看能否得到altermagnetism

    let J=0.1+0.0*li;
    model.add_hop(J,0,0,&array![1,0,0],1);
    model.add_hop(J,1,1,&array![1,0,0],1);
    model.add_hop(J,2,2,&array![1,0,0],1);
    model.add_hop(J,3,3,&array![1,0,0],1);
    model.add_hop(-J,0,0,&array![0,1,0],1);
    model.add_hop(-J,1,1,&array![0,1,0],1);
    model.add_hop(-J,2,2,&array![0,1,0],1);
    model.add_hop(-J,3,3,&array![0,1,0],1);

    let J=0.1+0.0*li;
    model.add_hop(J,0,0,&array![1,0,0],2);
    model.add_hop(J,1,1,&array![1,0,0],2);
    model.add_hop(J,2,2,&array![1,0,0],2);
    model.add_hop(J,3,3,&array![1,0,0],2);
    model.add_hop(-J,0,0,&array![0,1,0],2);
    model.add_hop(-J,1,1,&array![0,1,0],2);
    model.add_hop(-J,2,2,&array![0,1,0],2);
    model.add_hop(-J,3,3,&array![0,1,0],2);

    let path=array![[0.5,0.0,0.0],[0.0,0.5,0.0]];
    let (kvec,kdist,knode)=model.k_path(&path,nk);
    let (eval,evec):(Array2<f64>,Array3<Complex<f64>>)=model.solve_all(&kvec);
    let label=vec!["X","Y"];
    let nk=1001;
    let evec:Array3<f64>=evec.map(|x| x.norm_sqr());

    let mut fg = Figure::new();
    let x:Vec<f64>=kdist.to_vec();
    let axes=fg.axes2d();
    for i in 0..model.nsta(){
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
    pdf_name.push_str("examples/BiF_square/alter");
    create_dir_all(&pdf_name).expect("can't creat the file");
    pdf_name.push_str("/plot.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();

    let nk:usize=301;
    let T:f64=0.0;
    let eta:f64=0.001;
    let og:f64=0.0;
    let mu:f64=0.0;
    let dir_1=arr1(&[1.0,0.0,0.0]);
    let dir_2=arr1(&[0.0,1.0,0.0]);
    let spin:usize=3;
    let kmesh=arr1(&[nk,nk,1]);
    let start = Instant::now();   // 开始计时
    let conductivity=model.Hall_conductivity(&kmesh,&dir_1,&dir_2,mu,T,og,spin,eta);
    let end = Instant::now();    // 结束计时
    let duration = end.duration_since(start); // 计算执行时间
    println!("Hall_conductivity is {}, took {} seconds", conductivity, duration.as_secs_f64());   // 输出执行时间

    let mu=Array1::<f64>::linspace(-1.0,1.0,1001);
    let conductivity=model.Hall_conductivity_mu(&kmesh,&dir_1,&dir_2,&mu,T,og,spin,eta);

    let mu:f64=0.0;

    let nk:usize=1001;
    let green=surf_Green::from_Model(&model,0,1e-3,None);
    let E_min=-1.0;
    let E_max=1.0;
    let E_n=nk.clone();
    let path=[[-0.5,-0.5],[0.0,0.0],[0.5,0.5]];
    let path=arr2(&path);
    let label=vec!["M","G","M"];
    green.show_surf_state("examples/BiF_square/surf",&path,&label,nk,E_min,E_max,E_n,0);


    let super_model=model.cut_piece(20,0);
    let path=[[0.0,-0.5,0.0],[0.0,0.0,0.0],[0.0,0.5,0.0]];
    let path=arr2(&path);
    let label=vec!["M","G","M"];
    super_model.show_band(&path,&label,nk,"examples/BiF_square/super_band");


    //画一下贝利曲率的分布
    let nk:usize=1000;
    let kmesh=arr1(&[nk,nk,1]);
    let kvec=gen_kmesh(&kmesh);
    //let kvec=kvec-0.5;
    let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,mu,T,og,spin,eta);
    let data=berry_curv.into_shape((nk,nk)).unwrap();
    draw_heatmap(&data,"examples/BiF_square/surf/berry_curvature_distribution.pdf");

    let path=array![[0.0,0.0,0.0],[0.5,0.0,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0]];
    let label=vec!["G","X","M","G"];
    let (k_vec,k_dist,k_node)=model.k_path(&path,nk);
    let berry_curv=model.berry_curvature(&kvec,&dir_1,&dir_2,mu,T,og,spin,eta);

    let mut fg = Figure::new();
    let x:Vec<f64>=k_dist.to_vec();
    let axes=fg.axes2d();
    let y:Vec<f64>=berry_curv.to_vec();
    axes.lines(&x, &y, &[Color("black"),LineStyle(Solid)]);
    let axes=axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len()-1]]));
    let label=label.clone();
    let mut show_ticks=Vec::new();
    for i in 0..k_node.len(){
        let A=k_node[[i]];
        let B=label[i];
        show_ticks.push(Major(A,Fix(B)));
    }
    axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[]);
    
    let mut name=String::new();
    let k_node=k_node.to_vec();
    let mut pdf_name=name.clone();
    pdf_name.push_str("examples/BiF_square/berry_curv.pdf");
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show();
    //开始计算角态
    let show_str=model.atom_position().clone().dot(&model.lat);
    let show_str=show_str.slice(s![..,0..2]).to_owned();
    let num=30;
    let model_1=model.cut_piece(num,0);
    let new_model=model_1.cut_piece(num,1);
    /*
    let mut new_model=model.cut_dot(num,4,Some(vec![0,1]));
    let mut del_atom=Vec::new();
    let num0=num as f64;
    for (i,a) in new_model.atom.outer_iter().enumerate(){
        if a[[0]].abs()<1e-3 || (a[[0]]-num0/(num0+1.0)).abs() < 1e-3{
            del_atom.push(i)
        }else if a[[1]] > (num0-1.0)/(num0+1.0){
            del_atom.push(i)
        }
    }
    new_model.remove_atom(&del_atom);
    */

    let mut s=0;
    let (band,evec)=new_model.solve_onek(&arr1(&[0.0,0.0,0.0]));
    let show_evec=evec.to_owned().map(|x| x.norm_sqr());
    let mut size=Array2::<f64>::zeros((new_model.nsta(),new_model.natom()));
    let norb=new_model.norb();
    for i in 0..new_model.nsta(){
        let mut s=0;
        for j in 0..new_model.natom(){
            for k in 0..new_model.atoms[j].norb(){
                size[[i,j]]+=show_evec[[i,s]]+show_evec[[i,s+new_model.norb()]];
                s+=1;
            }
        }
    }

    let show_str=new_model.atom_position().dot(&model.lat);
    let show_str=show_str.slice(s![..,0..2]).to_owned();
    let show_size=size.row(new_model.norb()).to_owned();

    create_dir_all("examples/BiF_square/corner").expect("can't creat the file");
    write_txt_1(band,"examples/BiF_square/corner/band.txt");
    write_txt(size,"examples/BiF_square/corner/evec.txt");
    write_txt(show_str,"examples/BiF_square/corner/structure.txt");
}



fn  write_txt(data:Array2<f64>,output:&str)-> std::io::Result<()>{
    use std::fs::File;
    use std::io::Write;
    let mut file=File::create(output).expect("Unable to BAND.dat");
    let n=data.len_of(Axis(0));
    let s=data.len_of(Axis(1));
    for i in 0..n{
        let mut s0=String::new();
        for j in 0..s{
            if data[[i,j]]>=0.0{
                s0.push_str("     ");
            }else{
                s0.push_str("    ");
            }
            let aa= format!("{:.6}", data[[i,j]]);
            s0.push_str(&aa);
        }
        writeln!(file,"{}",s0)?;
    }
    Ok(())
}

fn  write_txt_1(data:Array1<f64>,output:&str)-> std::io::Result<()>{
    use std::fs::File;
    use std::io::Write;
    let mut file=File::create(output).expect("Unable to BAND.dat");
    let n=data.len_of(Axis(0));
    for i in 0..n{
        let mut s0=String::new();
        if data[[i]]>=0.0{
            s0.push_str(" ");
        }
        let aa= format!("{:.6}", data[[i]]);
        s0.push_str(&aa);
        writeln!(file,"{}",s0)?;
    }
    Ok(())
}
