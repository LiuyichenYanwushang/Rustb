///这个模块是用来求解表面格林函数的一个模块.
pub mod surfgreen{
    use gnuplot::{Figure, AxesCommon, AutoOption::Fix,HOT,RAINBOW,Font,Auto,Custom};
    use crate::{surf_Green,Model,remove_col,remove_row,gen_kmesh};
    use gnuplot::Major;
    use num_complex::Complex;
    use ndarray::*;
    use ndarray::prelude::*;
    use ndarray::concatenate;
    use ndarray_linalg::*;
    use ndarray::linalg::kron;
    use std::f64::consts::PI;
    use ndarray_linalg::conjugate;
    use rayon::prelude::*;
    use std::io::{Write};
    use std::fs::File;
    use std::ops::AddAssign;
    use std::ops::MulAssign;
    impl surf_Green{
        ///从 Model 中构建一个 surf_green 的结构体
        ///
        ///dir表示要看哪方向的表面态
        ///
        ///eta表示小虚数得取值
        ///
        ///对于非晶格矢量得方向, 需要用 model.make_supercell 先扩胞
        pub fn from_Model(model:&Model,dir:usize,eta:f64)->surf_Green{
            if dir > model.dim_r{
                panic!("Wrong, the dir must smaller than model's dim_r")
            }
            let mut R_max:usize=0;
            for R0 in model.hamR.rows(){
                if R_max < R0[[dir]].abs() as usize{
                    R_max=R0[[dir]].abs() as usize;
                }
            }
            let mut U=Array2::<f64>::eye(model.dim_r);
            U[[dir,dir]]=R_max as f64;
            let model=model.make_supercell(&U);
            let mut ham0=Array3::<Complex<f64>>::zeros((0,model.nsta,model.nsta));
            let mut hamR0=Array2::<isize>::zeros((0,model.dim_r));
            let mut hamR=Array3::<Complex<f64>>::zeros((0,model.nsta,model.nsta));
            let mut hamRR=Array2::<isize>::zeros((0,model.dim_r));
            let use_hamR=model.hamR.rows();
            let use_ham=model.ham.outer_iter();
            for (ham,R) in use_ham.zip(use_hamR){
                let ham=ham.clone();
                let R=R.clone();
                if R[[dir]]==0{
                    ham0.push(Axis(0),ham.view());
                    hamR0.push_row(R.view());
                }else if R[[dir]] > 0{
                    hamR.push(Axis(0),ham.view());
                    hamRR.push_row(R.view());
                }else{
                    hamR.push(Axis(0),ham.mapv(|x| x.conj()).t().view());
                    hamRR.push_row(R.map(|x| -x).view());
                }
            }
            let new_lat=remove_row(model.lat,dir);
            let new_orb=remove_col(model.orb,dir);
            let new_atom=remove_col(model.atom,dir);
            let new_hamR0=remove_col(hamR0,dir);
            let new_hamRR=remove_col(hamRR,dir);
            let mut green:surf_Green=surf_Green{
                dim_r:model.dim_r-1,
                norb:model.norb,
                nsta:model.nsta,
                natom:model.natom,
                spin:model.spin,
                lat:new_lat,
                orb:new_orb,
                atom:new_atom,
                atom_list:model.atom_list,
                ham_bulk:ham0,
                ham_bulkR:new_hamR0,
                ham_hop:hamR,
                ham_hopR:new_hamRR,
                eta,
            };
            green
        }
        pub fn k_path(&self,path:&Array2::<f64>,nk:usize)->(Array2::<f64>,Array1::<f64>,Array1::<f64>){
            //!根据高对称点来生成高对称路径, 画能带图
            if self.dim_r==0{
                panic!("the k dimension of the model is 0, do not use k_path")
            }
            let n_node:usize=path.len_of(Axis(0));
            if self.dim_r != path.len_of(Axis(1)){
                panic!("Wrong, the path's length along 1 dimension must equal to the model's dimension")
            }
            let k_metric=(self.lat.dot(&self.lat.t())).inv().unwrap();
            let mut k_node=Array1::<f64>::zeros(n_node);
            for n in 1..n_node{
                //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
                let dk=path.row(n).to_owned()-path.slice(s![n-1,..]).to_owned();
                let a=k_metric.dot(&dk);
                let dklen=dk.dot(&a).sqrt();
                k_node[[n]]=k_node[[n-1]]+dklen;
            }
            let mut node_index:Vec<usize>=vec![0];
            for n in 1..n_node-1{
                let frac=k_node[[n]]/k_node[[n_node-1]];
                let a=(frac*((nk-1) as f64).round()) as usize;
                node_index.push(a)
            }
            node_index.push(nk-1);
            let mut k_dist=Array1::<f64>::zeros(nk);
            let mut k_vec=Array2::<f64>::zeros((nk,self.dim_r));
            //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
            k_vec.row_mut(0).assign(&path.row(0));
            for n in 1..n_node {
                let n_i=node_index[n-1];
                let n_f=node_index[n];
                let kd_i=k_node[[n-1]];
                let kd_f=k_node[[n]];
                let k_i=path.row(n-1);
                let k_f=path.row(n);
                for j in n_i..n_f+1{
                    let frac:f64= ((j-n_i) as f64)/((n_f-n_i) as f64);
                    k_dist[[j]]=kd_i + frac*(kd_f-kd_i);
                    k_vec.row_mut(j).assign(&((1.0-frac)*k_i.to_owned() +frac*k_f.to_owned()));

                }
            }
            (k_vec,k_dist,k_node)
        }

        #[inline(always)]
        pub fn gen_ham_onek(&self,kvec:&Array1<f64>)->(Array2<Complex<f64>>,Array2<Complex<f64>>){
            let mut ham0k=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let mut hamRk=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            if kvec.len() !=self.dim_r{
                panic!("Wrong, the k-vector's length must equal to the dimension of model.")
            }
            let nR:usize=self.ham_bulkR.len_of(Axis(0));
            let nRR:usize=self.ham_hopR.len_of(Axis(0));
            let U0:Array1::<f64>=self.orb.dot(kvec);
            let U0:Array1::<Complex<f64>>=U0.map(|x| Complex::<f64>::new(*x,0.0));
            let U0=U0*Complex::new(0.0,2.0*PI);
            let mut U0:Array1::<Complex<f64>>=U0.mapv(Complex::exp);//关于轨道的 guage
            if self.spin{
                let UU=U0.clone();
                U0.append(Axis(0),UU.view()).unwrap();//因为自旋, 把坐标扩大一倍
            }
            let U=Array2::from_diag(&U0);
            //对体系作傅里叶变换
            let U0=(self.ham_bulkR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
            let U0=U0*Complex::new(0.0,2.0*PI);
            let U0=U0.mapv(Complex::exp);
            //对 ham_hop 作傅里叶变换
            let UR=(self.ham_hopR.map(|x| *x as f64)).dot(kvec).map(|x| Complex::<f64>::new(*x,0.0));
            let UR=UR*Complex::new(0.0,2.0*PI);
            let UR=UR.mapv(Complex::exp);
            //先对 ham_bulk 中的 [0,0] 提取出来
            let mut ham0=self.ham_bulk.slice(s![0,..,..]).to_owned();
            for i in 1..nR{
                ham0k=ham0k+self.ham_bulk.slice(s![i,..,..]).to_owned()*U0[[i]];
            }
            for i in 0..nRR{
                hamRk=hamRk+self.ham_hop.slice(s![i,..,..]).to_owned()*UR[[i]];
            }
            ham0k=&ham0+&ham0k.map(|x| x.conj()).t()+&ham0k;
            //hamRk=&hamRk.map(|x| x.conj()).t()+&hamRk;
            //作规范变换
            ham0k=ham0k.dot(&U);
            let ham0k=U.map(|x| x.conj()).t().dot(&ham0k);
            hamRk=hamRk.dot(&U);
            let hamRk=U.map(|x| x.conj()).t().dot(&hamRk);
            (ham0k,hamRk)

        }
        #[inline(always)]
        pub fn surf_green_one(&self,kvec:&Array1<f64>,Energy:f64,spin:usize)->(f64,f64,f64){
            let (hamk,hamRk)=self.gen_ham_onek(kvec);
            let hamRk_conj:Array2<Complex<f64>>=conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>,OwnedRepr<Complex<f64>>>(&hamRk);
            let I0=Array2::<Complex<f64>>::eye(self.nsta);
            let accurate:f64=1e-8;
            let epsilon=Complex::new(Energy,self.eta)*&I0;
            let mut epi=hamk.clone();
            let mut eps=hamk.clone();
            let mut eps_t=hamk.clone();
            let mut ap=hamRk.clone();
            let mut bt=hamRk_conj.clone();

            for _ in 0..100{
                let g0=(&epsilon-&epi).inv().unwrap();
                let mat_1=&ap.dot(&g0);
                let mat_2=&bt.dot(&g0);
                let g0=&mat_1.dot(&bt);
                epi=epi+g0;
                eps=eps+g0;
                let g0=&mat_2.dot(&ap);
                epi=epi+g0;
                eps_t=eps_t+g0;
                ap=mat_1.dot(&ap);
                bt=mat_2.dot(&bt);
                if ap.sum().norm() < accurate{
                    break
                }
            }
            let g_LL=(&epsilon-eps).inv().unwrap();
            let g_RR=(&epsilon-eps_t).inv().unwrap();
            let g_B=(&epsilon-epi).inv().unwrap();
            let ((N_R,N_L),N_B)=if self.spin{
                let ((NR,NL),NB)=match spin{
                    0=>{
                        let NR:f64=-1.0/(PI)*g_RR.into_diag().sum().im;
                        let NL:f64=-1.0/(PI)*g_LL.into_diag().sum().im;
                        let NB:f64=-1.0/(PI)*g_B.into_diag().sum().im;
                        ((NR,NL),NB)
                    },
                    1=>{
                        let s=array![[Complex::new(0.0,0.0),Complex::new(1.0,0.0)],[Complex::new(1.0,0.0),Complex::new(0.0,0.0)]];
                        let s=kron(&Array2::from_diag(&Array1::ones(self.norb).mapv(|x| Complex::new(x,0.0))),&s);
                        let NR:f64=-1.0/(PI)*(g_RR.dot(&s)).into_diag().sum().im;
                        let NL:f64=-1.0/(PI)*(g_LL.dot(&s)).into_diag().sum().im;
                        let NB:f64=-1.0/(PI)*(g_B.dot(&s)).into_diag().sum().im;
                        ((NR,NL),NB)
                    },
                    2=>{
                        let s=array![[Complex::new(0.0,0.0),Complex::new(0.0,-1.0)],[Complex::new(0.0,1.0),Complex::new(0.0,0.0)]];
                        let s=kron(&Array2::from_diag(&Array1::ones(self.norb).mapv(|x| Complex::new(x,0.0))),&s);
                        let NR:f64=-1.0/(PI)*(g_RR.dot(&s)).into_diag().sum().im;
                        let NL:f64=-1.0/(PI)*(g_LL.dot(&s)).into_diag().sum().im;
                        let NB:f64=-1.0/(PI)*(g_B.dot(&s)).into_diag().sum().im;
                        ((NR,NL),NB)
                    },
                    3=>{
                        let s=array![[Complex::new(1.0,0.0),Complex::new(0.0,0.0)],[Complex::new(0.0,0.0),Complex::new(-1.0,0.0)]];
                        let s=kron(&Array2::from_diag(&Array1::ones(self.norb).mapv(|x| Complex::new(x,0.0))),&s);
                        let NR:f64=-1.0/(PI)*(g_RR.dot(&s)).into_diag().sum().im;
                        let NL:f64=-1.0/(PI)*(g_LL.dot(&s)).into_diag().sum().im;
                        let NB:f64=-1.0/(PI)*(g_B.dot(&s)).into_diag().sum().im;
                        ((NR,NL),NB)
                    },
                    _=>todo!(),
                };
                ((NR,NL),NB)
            }else{
                let NR:f64=-1.0/(PI)*g_RR.into_diag().sum().im;
                let NL:f64=-1.0/(PI)*g_LL.into_diag().sum().im;
                let NB:f64=-1.0/(PI)*g_B.into_diag().sum().im;
                ((NR,NL),NB)
            };
            (N_R,N_L,N_B)
        }

        #[inline(always)]
        pub fn surf_green_onek(&self,kvec:&Array1<f64>,Energy:&Array1<f64>,spin:usize)->(Array1<f64>,Array1<f64>,Array1<f64>){
            let (hamk,hamRk)=self.gen_ham_onek(kvec);
            //let hamRk_conj:Array2<Complex<f64>>=hamRk.clone().map(|x| x.conj()).reversed_axes();
            let hamRk_conj:Array2<Complex<f64>>=conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>,OwnedRepr<Complex<f64>>>(&hamRk);
            let I0=Array2::<Complex<f64>>::eye(self.nsta);
            let accurate:f64=1e-16;
            let ((N_R,N_L),N_B)=Energy.into_par_iter().map(|e| {
                let epsilon=Complex::new(*e,self.eta)*&I0;
                let mut epi=hamk.clone();
                let mut eps=hamk.clone();
                let mut eps_t=hamk.clone();
                let mut ap=hamRk.clone();
                let mut bt=hamRk_conj.clone();
                for _ in 0..100{
                    let g0=(&epsilon-&epi).inv().unwrap();
                    let mat_1=&ap.dot(&g0);
                    let mat_2=&bt.dot(&g0);
                    let g0=&mat_1.dot(&bt);
                    epi=epi+g0;
                    eps=eps+g0;
                    let g0=&mat_2.dot(&ap);
                    epi=epi+g0;
                    eps_t=eps_t+g0;
                    ap=mat_1.dot(&ap);
                    bt=mat_2.dot(&bt);
                    if ap.map(|x| x.norm()).sum() < accurate{
                        break
                    }
                }
                let g_LL=(&epsilon-eps).inv().unwrap();
                let g_RR=(&epsilon-eps_t).inv().unwrap();
                let g_B=(&epsilon-epi).inv().unwrap();
                let ((N_R,N_L),N_B)=if self.spin{
                    let ((NR,NL),NB)=match spin{
                        0=>{
                            let NR:f64=-1.0/(PI)*g_RR.into_diag().sum().im;
                            let NL:f64=-1.0/(PI)*g_LL.into_diag().sum().im;
                            let NB:f64=-1.0/(PI)*g_B.into_diag().sum().im;
                            ((NR,NL),NB)
                        },
                        1=>{
                            let s=array![[Complex::new(0.0,0.0),Complex::new(1.0,0.0)],[Complex::new(1.0,0.0),Complex::new(0.0,0.0)]];
                            let s=kron(&Array2::from_diag(&Array1::ones(self.norb).mapv(|x| Complex::new(x,0.0))),&s);
                            let NR:f64=-1.0/(PI)*(g_RR.dot(&s)).into_diag().sum().im;
                            let NL:f64=-1.0/(PI)*(g_LL.dot(&s)).into_diag().sum().im;
                            let NB:f64=-1.0/(PI)*(g_B.dot(&s)).into_diag().sum().im;
                            ((NR,NL),NB)
                        },
                        2=>{
                            let s=array![[Complex::new(0.0,0.0),Complex::new(0.0,-1.0)],[Complex::new(0.0,1.0),Complex::new(0.0,0.0)]];
                            let s=kron(&Array2::from_diag(&Array1::ones(self.norb).mapv(|x| Complex::new(x,0.0))),&s);
                            let NR:f64=-1.0/(PI)*(g_RR.dot(&s)).into_diag().sum().im;
                            let NL:f64=-1.0/(PI)*(g_LL.dot(&s)).into_diag().sum().im;
                            let NB:f64=-1.0/(PI)*(g_B.dot(&s)).into_diag().sum().im;
                            ((NR,NL),NB)
                        },
                        3=>{
                            let s=array![[Complex::new(1.0,0.0),Complex::new(0.0,0.0)],[Complex::new(0.0,0.0),Complex::new(-1.0,0.0)]];
                            let s=kron(&Array2::from_diag(&Array1::ones(self.norb).mapv(|x| Complex::new(x,0.0))),&s);
                            let NR:f64=-1.0/(PI)*(g_RR.dot(&s)).into_diag().sum().im;
                            let NL:f64=-1.0/(PI)*(g_LL.dot(&s)).into_diag().sum().im;
                            let NB:f64=-1.0/(PI)*(g_B.dot(&s)).into_diag().sum().im;
                            ((NR,NL),NB)
                        },
                        _=>todo!(),
                    };
                    ((NR,NL),NB)
                }else{
                    let NR:f64=-1.0/(PI)*g_RR.into_diag().sum().im;
                    let NL:f64=-1.0/(PI)*g_LL.into_diag().sum().im;
                    let NB:f64=-1.0/(PI)*g_B.into_diag().sum().im;
                    ((NR,NL),NB)
                };
                ((N_R,N_L),N_B)
             }).collect();
            let N_R=Array1::from_vec(N_R);
            let N_L=Array1::from_vec(N_L);
            let N_B=Array1::from_vec(N_B);
            (N_R,N_L,N_B)
        }
        pub fn surf_green_path(&self,kvec:&Array2<f64>,E_min:f64,E_max:f64,E_n:usize,spin:usize)->(Array2<f64>,Array2<f64>,Array2<f64>){
            let Energy=Array1::<f64>::linspace(E_min,E_max,E_n);
            let nk=kvec.nrows();
            let mut N_R=Array2::<f64>::zeros((nk,E_n));
            let mut N_L=Array2::<f64>::zeros((nk,E_n));
            let mut N_B=Array2::<f64>::zeros((nk,E_n));
            Zip::from(N_R.outer_iter_mut()).and(N_L.outer_iter_mut()).and(N_B.outer_iter_mut()).and(kvec.outer_iter()).par_apply(|mut nr,mut nl,mut nb, k| {
                let (NR,NL,NB)=self.surf_green_onek(&k.to_owned(),&Energy,spin);
                nr.assign(&NR);
                nl.assign(&NL);
                nb.assign(&NB);
            });
            (N_L,N_R,N_B)
        }
        pub fn show_surf_state(&self,name:&str,kpath:&Array2::<f64>,label:&Vec<&str>,nk:usize,E_min:f64,E_max:f64,E_n:usize,spin:usize){
            use std::io::{BufWriter, Write};
            use std::fs::create_dir_all;
            create_dir_all(name).expect("can't creat the file");
            let (kvec,kdist,knode)=self.k_path(kpath, nk);
            let Energy=Array1::<f64>::linspace(E_min,E_max,E_n);
            let (N_L,N_R,N_B)=self.surf_green_path(&kvec,E_min,E_max,E_n,spin);
            //let N_L=N_L.mapv(|x| if x > 0.0 {x.ln()} else if  x< 0.0 {-x.abs().ln()} else {0.0});
            //let N_R=N_R.mapv(|x| if x > 0.0 {x.ln()} else if  x< 0.0 {-x.abs().ln()} else {0.0});
            //let N_B=N_B.mapv(|x| if x > 0.0 {x.ln()} else if  x< 0.0 {-x.abs().ln()} else {0.0});
            let ((N_L,N_R),N_B)=if spin==0{
                let N_L=N_L.mapv(|x| x.ln());
                let N_R=N_R.mapv(|x| x.ln());
                let N_B=N_B.mapv(|x| x.ln());
                let max = N_L.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                let min = N_L.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                let N_L=(N_L-min)/(max-min)*20.0-10.0;
                let max = N_R.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                let min = N_R.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                let N_R=(N_R-min)/(max-min)*20.0-10.0;
                let max = N_B.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                let min = N_B.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                let N_B=(N_B-min)/(max-min)*20.0-10.0;
                ((N_L,N_R),N_B)
            }else{
                ((N_L,N_R),N_B)
            };

            //绘制 left_dos------------------------
            let mut left_name:String=String::new();
            left_name.push_str(&name.clone());
            left_name.push_str("/dos.surf_l");
            let mut file=File::create(left_name).expect("Unable to dos.surf_l.dat");
            let mut writer = BufWriter::new(file);
            let mut s = String::new();
            for i in 0..nk{
                for j in 0..E_n{
                    let aa= format!("{:.6}", kdist[[i]]);
                    s.push_str(&aa);
                    let bb:String=format!("{:.6}",Energy[[j]]);
                    if Energy[[j]]>=0.0{
                        s.push_str("    ");
                    }else{
                        s.push_str("   ");
                    }
                    s.push_str(&bb);
                    let cc:String=format!("{:.6}",N_L[[i,j]]);
                    if N_L[[i,j]]>=0.0{
                        s.push_str("    ");
                    }else{
                        s.push_str("   ");
                    }
                    s.push_str(&cc);
                    s.push_str("    ");
                    //writeln!(file,"{}",s);
                    s.push_str("\n");

                }
                s.push_str("\n");
                //writeln!(file,"\n");
            }
            writer.write_all(s.as_bytes()).unwrap();
            let _=file;

            //接下来我们绘制表面态 
            let mut fg = Figure::new();
            let width:usize=nk;
            let height:usize=E_n;
            let mut heatmap_data = vec![];
            for i in 0..height {
                for j in 0..width {
                    heatmap_data.push(N_L[[j, i]]);
                }
            }
            let axes = fg.axes2d();
            //axes.set_palette(RAINBOW);
            axes.set_palette(Custom(&[(-1.0,0.0,0.0,0.0),(-0.9,65.0/255.0,9.0/255.0,103.0/255.0),(0.0,147.0/255.0,37.0/255.0,103.0/255.0),(0.2,220.0/255.0,80.0/255.0,57.0/255.0),(1.0,252.0/255.0,254.0/255.0,164.0/255.0)]));
            axes.image(heatmap_data.iter(), width, height,Some((kdist[[0]],E_min,kdist[[nk-1]],E_max)), &[]);
            let axes=axes.set_y_range(Fix(E_min), Fix(E_max));
            let axes=axes.set_x_range(Fix(kdist[[0]]), Fix(kdist[[nk-1]]));
            let axes=axes.set_aspect_ratio(Fix(1.0));
            let mut show_ticks=Vec::new();
            for i in 0..knode.len(){
                let A=knode[[i]];
                let B=label[i];
                show_ticks.push(Major(A,Fix(B)));
            }
            axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",24.0)]);
            axes.set_y_ticks(Some((Auto,0)),&[],&[Font("Times New Roman",24.0)]);
            //axes.set_cb_ticks(Some((Fix(5.0),0)),&[],&[Font("Times New Roman",24.0)]);
            axes.set_cb_ticks_custom([Major(-10.0,Fix("low")),Major(0.0,Fix("0")),Major(10.0,Fix("high"))].into_iter(),&[],&[Font("Times New Roman",24.0)]);
            let mut pdfname=String::new();
            pdfname.push_str(&name.clone());
            pdfname.push_str("/surf_state_l.pdf");
            fg.set_terminal("pdfcairo",&pdfname);
            fg.show().expect("Unable to draw heatmap");
            let _=fg;

            //绘制右表面态----------------------
            let mut left_name:String=String::new();
            left_name.push_str(&name.clone());
            left_name.push_str("/dos.surf_r");
            let mut file=File::create(left_name).expect("Unable to dos.surf_l.dat");
            let mut writer = BufWriter::new(file);
            let mut s = String::new();
            for i in 0..nk{
                for j in 0..E_n{
                    let aa= format!("{:.6}", kdist[[i]]);
                    s.push_str(&aa);
                    let bb:String=format!("{:.6}",Energy[[j]]);
                    if Energy[[j]]>=0.0{
                        s.push_str("    ");
                    }else{
                        s.push_str("   ");
                    }
                    s.push_str(&bb);
                    let cc:String=format!("{:.6}",N_R[[i,j]]);
                    if N_L[[i,j]]>=0.0{
                        s.push_str("    ");
                    }else{
                        s.push_str("   ");
                    }
                    s.push_str(&cc);
                    s.push_str("    ");
                    //writeln!(file,"{}",s);
                    s.push_str("\n");

                }
                s.push_str("\n");
                //writeln!(file,"\n");
            }
            writer.write_all(s.as_bytes()).unwrap();
            let _=file;

            //接下来我们绘制表面态 
            let mut fg = Figure::new();
            let width:usize=nk;
            let height:usize=E_n;
            let mut heatmap_data = vec![];
            for i in 0..height {
                for j in 0..width {
                    heatmap_data.push(N_R[[j, i]]);
                }
            }
            let axes = fg.axes2d();
            //axes.set_palette(RAINBOW);
            axes.set_palette(Custom(&[(-1.0,0.0,0.0,0.0),(-0.9,65.0/255.0,9.0/255.0,103.0/255.0),(0.0,147.0/255.0,37.0/255.0,103.0/255.0),(0.2,220.0/255.0,80.0/255.0,57.0/255.0),(1.0,252.0/255.0,254.0/255.0,164.0/255.0)]));
            axes.image(heatmap_data.iter(), width, height,Some((kdist[[0]],E_min,kdist[[nk-1]],E_max)), &[]);
            let axes=axes.set_y_range(Fix(E_min), Fix(E_max));
            let axes=axes.set_x_range(Fix(kdist[[0]]), Fix(kdist[[nk-1]]));
            let axes=axes.set_aspect_ratio(Fix(1.0));
            let mut show_ticks=Vec::new();
            for i in 0..knode.len(){
                let A=knode[[i]];
                let B=label[i];
                show_ticks.push(Major(A,Fix(B)));
            }
            axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",24.0)]);
            axes.set_y_ticks(Some((Auto,0)),&[],&[Font("Times New Roman",24.0)]);
            //axes.set_cb_ticks(Some((Fix(5.0),0)),&[],&[Font("Times New Roman",24.0)]);
            axes.set_cb_ticks_custom([Major(-10.0,Fix("low")),Major(0.0,Fix("0")),Major(10.0,Fix("high"))].into_iter(),&[],&[Font("Times New Roman",24.0)]);
            let mut pdfname=String::new();
            pdfname.push_str(&name.clone());
            pdfname.push_str("/surf_state_r.pdf");
            fg.set_terminal("pdfcairo",&pdfname);
            fg.show().expect("Unable to draw heatmap");
            let _=fg;
            //绘制体态----------------------
            let mut left_name:String=String::new();
            left_name.push_str(&name.clone());
            left_name.push_str("/dos.surf_bulk");
            let mut file=File::create(left_name).expect("Unable to dos.surf_l.dat");
            let mut writer = BufWriter::new(file);
            let mut s = String::new();
            for i in 0..nk{
                for j in 0..E_n{
                    let aa= format!("{:.6}", kdist[[i]]);
                    s.push_str(&aa);
                    let bb:String=format!("{:.6}",Energy[[j]]);
                    if Energy[[j]]>=0.0{
                        s.push_str("    ");
                    }else{
                        s.push_str("   ");
                    }
                    s.push_str(&bb);
                    let cc:String=format!("{:.6}",N_B[[i,j]]);
                    if N_L[[i,j]]>=0.0{
                        s.push_str("    ");
                    }else{
                        s.push_str("   ");
                    }
                    s.push_str(&cc);
                    s.push_str("    ");
                    //writeln!(file,"{}",s);
                    s.push_str("\n");

                }
                s.push_str("\n");
                //writeln!(file,"\n");
            }
            writer.write_all(s.as_bytes()).unwrap();
            let _=file;

            //接下来我们绘制表面态 
            let mut fg = Figure::new();
            let width:usize=nk;
            let height:usize=E_n;
            let mut heatmap_data = vec![];
            for i in 0..height {
                for j in 0..width {
                    heatmap_data.push(N_B[[j, i]]);
                }
            }
            let axes = fg.axes2d();
            //axes.set_palette(RAINBOW);
            axes.set_palette(Custom(&[(-1.0,0.0,0.0,0.0),(-0.9,65.0/255.0,9.0/255.0,103.0/255.0),(0.0,147.0/255.0,37.0/255.0,103.0/255.0),(0.2,220.0/255.0,80.0/255.0,57.0/255.0),(1.0,252.0/255.0,254.0/255.0,164.0/255.0)]));
            axes.image(heatmap_data.iter(), width, height,Some((kdist[[0]],E_min,kdist[[nk-1]],E_max)), &[]);
            let axes=axes.set_y_range(Fix(E_min), Fix(E_max));
            let axes=axes.set_x_range(Fix(kdist[[0]]), Fix(kdist[[nk-1]]));
            let axes=axes.set_aspect_ratio(Fix(1.0));
            let mut show_ticks=Vec::new();
            for i in 0..knode.len(){
                let A=knode[[i]];
                let B=label[i];
                show_ticks.push(Major(A,Fix(B)));
            }
            axes.set_x_ticks_custom(show_ticks.into_iter(),&[],&[Font("Times New Roman",24.0)]);
            axes.set_y_ticks(Some((Auto,0)),&[],&[Font("Times New Roman",24.0)]);
            //axes.set_cb_ticks(Some((Fix(5.0),0)),&[],&[Font("Times New Roman",24.0)]);
            axes.set_cb_ticks_custom([Major(-10.0,Fix("low")),Major(0.0,Fix("0")),Major(10.0,Fix("high"))].into_iter(),&[],&[Font("Times New Roman",24.0)]);
            let mut pdfname=String::new();
            pdfname.push_str(&name.clone());
            pdfname.push_str("/surf_state_b.pdf");
            fg.set_terminal("pdfcairo",&pdfname);
            fg.show().expect("Unable to draw heatmap");
            let _=fg;
        }
    }

}
