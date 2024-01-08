pub mod conductivity{
    use crate::{Model,gen_kmesh,gen_krange,comm,anti_comm};
    use num_complex::Complex;
    use ndarray::linalg::kron;
    use ndarray::*;
    use ndarray::prelude::*;
    use ndarray_linalg::*;
    use std::f64::consts::PI;
    use ndarray_linalg::conjugate;
    use rayon::prelude::*;
    use std::ops::AddAssign;
    use std::ops::MulAssign;

    #[inline(always)]
    fn adapted_integrate_quick(f0:&dyn Fn(&Array1::<f64>)->f64,k_range:&Array2::<f64>,re_err:f64,ab_err:f64)->f64{
        //!对于任意维度的积分 n, 我们的将区域刨分成 n+1面体的小块, 然后用线性插值来近似这个n+1的积分结果
        //!设被积函数为 $f(x_1,x_2,...,x_n)$, 存在 $n+1$ 个点 $(y_{01},y_{02},\cdots y_{0n})\cdots(y_{n1},y_{n2}\cdots y_{nn})$, 对应的值为 $z_0,z_1,...,z_n$
        //!这样我们就能得到这一块积分的近似值为 $$ \f{1}{(n+1)!}\times\sum_{i=0}^n z_i *\dd V.$$ 其中$\dd V$ 是正 $n+1$ 面体的体积.

        let dim=k_range.len_of(Axis(0));
        if dim==1{
            //对于一维情况, 我们就是用梯形算法的 (a+b)*h/2, 这里假设的是函数的插值为线性插值.
            let mut use_range=vec![(k_range.clone(),re_err,ab_err)];
            let mut result=0.0;
            while let Some((k_range,re_err,ab_err))=use_range.pop() {
                let kvec_l:Array1::<f64>=arr1(&[k_range[[0,0]]]);
                let kvec_r:Array1::<f64>=arr1(&[k_range[[0,1]]]);
                let kvec_m:Array1::<f64>=arr1(&[(k_range[[0,1]]+k_range[[0,0]])/2.0]);
                let dk:f64=k_range[[0,1]]-k_range[[0,0]];
                let y_l:f64=f0(&kvec_l);
                let y_r:f64=f0(&kvec_r);
                let y_m:f64=f0(&kvec_m);
                let all:f64=(y_l+y_r)*dk/2.0;
                let all_1=(y_l+y_m)*dk/4.0;
                let all_2=(y_r+y_m)*dk/4.0;
                let err=all_1+all_2-all;
                let abs_err= if ab_err>all*re_err{ab_err} else {all*re_err};
                if err< abs_err{
                    result+=all_1+all_2;
                }else{
                    let k_range_l=arr2(&[[kvec_l[[0]],kvec_m[[0]]]]);
                    let k_range_r=arr2(&[[kvec_m[[0]],kvec_r[[0]]]]);
                    use_range.push((k_range_l,re_err,ab_err/2.0));
                    use_range.push((k_range_r,re_err,ab_err/2.0));
                }
            }
            return result;
        }else if dim==2{
        //对于二维, 我们依旧假设线性插值, 这样我们考虑的就是二维平面上的三角形上的任意一点的值是到其余三个点的距离的加权系数的平均值, 我们将四边形变成两个三角形来考虑.
            let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第一个三角形
            let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1]],[k_range.row(0)[1],k_range.row(1)[0]],[k_range.row(0)[0],k_range.row(1)[1]]]);//第二个三角形
            #[inline(always)]
            fn adapt_integrate_triangle(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64,s1:f64,s2:f64,s3:f64)->f64{
                //这个函数是用来进行自适应算法的
                let mut result=0.0;
                let mut use_kvec=vec![(kvec.clone(),re_err,ab_err,s1,s2,s3)];
                while let Some((kvec,re_err,ab_err,s1,s2,s3))=use_kvec.pop() {
                    let S:f64=((kvec[[1,0]]*kvec[[2,1]]-kvec[[2,0]]*kvec[[1,1]])-(kvec[[0,0]]*kvec[[2,1]]-kvec[[0,1]]*kvec[[2,0]])+(kvec[[0,0]]*kvec[[1,1]]-kvec[[1,0]]*kvec[[0,1]])).abs();
                    let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                    let sm:f64=f0(&kvec_m.to_owned());

                    let mut kvec_1=Array2::<f64>::zeros((0,2));
                    kvec_1.push_row(kvec.row(0));
                    kvec_1.push_row(kvec.row(1));
                    kvec_1.push_row(kvec_m.view());

                    let mut kvec_2=Array2::<f64>::zeros((0,2));
                    kvec_2.push_row(kvec.row(0));
                    kvec_2.push_row(kvec_m.view());
                    kvec_2.push_row(kvec.row(2));

                    let mut kvec_3=Array2::<f64>::zeros((0,2));
                    kvec_3.push_row(kvec_m.view());
                    kvec_3.push_row(kvec.row(1));
                    kvec_3.push_row(kvec.row(2));

                    let all:f64=(s1+s2+s3)*S/6.0;
                    let all_new:f64=all/3.0*2.0+sm*S/6.0;
                    let abs_err:f64= if ab_err>all*re_err{ab_err} else {all*re_err};
                    if (all_new-all).abs() > abs_err && S>1e-8{
                       use_kvec.push((kvec_1,re_err,ab_err/3.0,s1,s2,sm));
                       use_kvec.push((kvec_2,re_err,ab_err/3.0,s1,sm,s3));
                       use_kvec.push((kvec_3,re_err,ab_err/3.0,sm,s2,s3));
                    }else{
                        result+=all_new; 
                    }
                }
                result
            }
            let s1=f0(&arr1(&[k_range.row(0)[0],k_range.row(1)[0]]));
            let s2=f0(&arr1(&[k_range.row(0)[1],k_range.row(1)[0]]));
            let s3=f0(&arr1(&[k_range.row(0)[0],k_range.row(1)[1]]));
            let s4=f0(&arr1(&[k_range.row(0)[1],k_range.row(1)[1]]));
            let all_1=adapt_integrate_triangle(f0,&area_1,re_err,ab_err/2.0,s1,s2,s3);
            let all_2=adapt_integrate_triangle(f0,&area_2,re_err,ab_err/2.0,s4,s2,s3);
            return all_1+all_2;
        }else if dim==3{
        //对于三位情况, 需要用到四面体, 所以需要先将6面体变成6个四面体
            #[inline(always)]
            fn adapt_integrate_tetrahedron(f0:&dyn Fn(&Array1::<f64>)->f64,kvec:&Array2::<f64>,re_err:f64,ab_err:f64,s1:f64,s2:f64,s3:f64,s4:f64,S:f64)->f64{
                //这个函数是用来进行自适应算法的
                let mut result=0.0;
                let mut use_kvec=vec![(kvec.clone(),re_err,ab_err,s1,s2,s3,s4,S)];
                while let Some((kvec,re_err,ab_err,s1,s2,s3,s4,S))=use_kvec.pop() {
                    let kvec_m=kvec.mean_axis(Axis(0)).unwrap();
                    let sm=f0(&kvec_m.to_owned());
                    /////////////////////////
                    let mut kvec_1=Array2::<f64>::zeros((0,3));
                    kvec_1.push_row(kvec.row(0));
                    kvec_1.push_row(kvec.row(1));
                    kvec_1.push_row(kvec.row(2));
                    kvec_1.push_row(kvec_m.view());
                    
                    let mut kvec_2=Array2::<f64>::zeros((0,3));
                    kvec_2.push_row(kvec.row(0));
                    kvec_2.push_row(kvec.row(1));
                    kvec_2.push_row(kvec_m.view());
                    kvec_2.push_row(kvec.row(3));

                    let mut kvec_3=Array2::<f64>::zeros((0,3));
                    kvec_3.push_row(kvec.row(0));
                    kvec_3.push_row(kvec_m.view());
                    kvec_3.push_row(kvec.row(2));
                    kvec_3.push_row(kvec.row(3));

                    let mut kvec_4=Array2::<f64>::zeros((0,3));
                    kvec_4.push_row(kvec_m.view());
                    kvec_4.push_row(kvec.row(1));
                    kvec_4.push_row(kvec.row(2));
                    kvec_4.push_row(kvec.row(3));

                    let all=(s1+s2+s3+s4)*S/24.0;
                    let all_new=all/4.0*3.0+sm*S/24.0;
                    let S1=S/4.0;
                    let abs_err= if ab_err>all*re_err{ab_err} else {all*re_err};
                    if (all_new-all).abs()> abs_err && S > 1e-9{
                        use_kvec.push((kvec_1,re_err,ab_err*0.25,s1,s2,s3,sm,S1));
                        use_kvec.push((kvec_2,re_err,ab_err*0.25,s1,s2,sm,s4,S1));
                        use_kvec.push((kvec_3,re_err,ab_err*0.25,s1,sm,s3,s4,S1));
                        use_kvec.push((kvec_4,re_err,ab_err*0.25,sm,s2,s3,s4,S1));
                    }else{
                        result+=all_new;
                    }
                }
                result
            }
            let area_1:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[0]],
                                            [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                            [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                            [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]]]);//第一个四面体
            let area_2:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[0]],
                                            [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                            [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                            [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第二个四面体
            let area_3:Array2::<f64>=arr2(&[[k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[1]],
                                            [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]],
                                            [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                            [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第三个四面体
            let area_4:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[1]],
                                            [k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                            [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                            [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]]]);//第四个四面体
            let area_5:Array2::<f64>=arr2(&[[k_range.row(0)[0],k_range.row(1)[0],k_range.row(2)[1]],
                                            [k_range.row(0)[1],k_range.row(1)[1],k_range.row(2)[1]],
                                            [k_range.row(0)[0],k_range.row(1)[1],k_range.row(2)[0]],
                                            [k_range.row(0)[1],k_range.row(1)[0],k_range.row(2)[0]]]);//第五个四面体
            let s1=f0(&area_1.row(0).to_owned());
            let s2=f0(&area_1.row(1).to_owned());
            let s3=f0(&area_2.row(0).to_owned());
            let s4=f0(&area_1.row(2).to_owned());
            let s5=f0(&area_1.row(3).to_owned());
            let s6=f0(&area_3.row(0).to_owned());
            let s7=f0(&area_2.row(3).to_owned());
            let s8=f0(&area_4.row(0).to_owned());
            let V=(k_range[[0,1]]-k_range[[0,0]])*(k_range[[1,1]]-k_range[[1,0]])*(k_range[[2,1]]-k_range[[2,0]]);
            let all_1=adapt_integrate_tetrahedron(f0,&area_1,re_err,ab_err/6.0,s1,s2,s4,s5,V/6.0);
            let all_2=adapt_integrate_tetrahedron(f0,&area_2,re_err,ab_err/6.0,s3,s2,s4,s7,V/6.0);
            let all_3=adapt_integrate_tetrahedron(f0,&area_3,re_err,ab_err/6.0,s6,s2,s5,s7,V/6.0);
            let all_4=adapt_integrate_tetrahedron(f0,&area_4,re_err,ab_err/6.0,s8,s5,s4,s7,V/6.0);
            let all_5=adapt_integrate_tetrahedron(f0,&area_5,re_err,ab_err/3.0,s5,s7,s4,s2,V/3.0);
            return all_1+all_2+all_3+all_4+all_5
        }else{
            panic!("wrong, the row_dim if k_range must be 1,2 or 3, but you's give {}",dim);
        }
    }
    #[allow(non_snake_case)]
    impl Model{
        #[allow(non_snake_case)]
        #[inline(always)]
        pub fn berry_curvature_n_onek<S:Data<Elem=f64>>(&self,k_vec:&ArrayBase<S,Ix1>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array1::<f64>,Array1::<f64>){
            //!给定一个k点, 返回 $\Omega_n(\bm k)$
            //返回 $Omega_{n,\ap\bt}, \ve_{n\bm k}$
            let li:Complex<f64>=1.0*Complex::i();
            let (band,evec)=self.solve_onek(&k_vec);
            let mut v:Array3::<Complex<f64>>=self.gen_v(k_vec);
            let mut J:Array3::<Complex<f64>>=v.clone();
            let (J,v)=if self.spin {
                let mut X:Array2::<Complex<f64>>=Array2::eye(self.nsta);
                let pauli:Array2::<Complex<f64>>= match spin{
                    0=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,1.0+0.0*li]]),
                    1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                    2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                    3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                    _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
                };
                X=kron(&pauli,&Array2::eye(self.norb));
                let J=J.outer_iter().zip(dir_1.iter()).fold(Array2::zeros((self.nsta,self.nsta)),|acc,(x,d)| {acc+&anti_comm(&X,&x)*(*d*0.5+0.0*li)});
                let v=v.outer_iter().zip(dir_2.iter()).fold(Array2::zeros((self.nsta,self.nsta)),|acc,(x,d)| {acc+&x*(*d+0.0*li)});
                (J,v)
            }else{ 
                if spin !=0{
                    println!("Warning, the model haven't got spin, so the spin input will be ignord");
                }

                let J=J.outer_iter().zip(dir_1.iter()).fold(Array2::zeros((self.nsta,self.nsta)),|acc,(x,d)| {acc+&x*(*d+0.0*li)});
                let v=v.outer_iter().zip(dir_2.iter()).fold(Array2::zeros((self.nsta,self.nsta)),|acc,(x,d)| {acc+&x*(*d+0.0*li)});
                (J,v)
            };

            let evec_conj:Array2::<Complex<f64>>=evec.mapv(|x| x.conj());
            let evec=evec.t();
            let A1=J.dot(&evec);
            let A1=&evec_conj.dot(&A1);
            let A2=v.dot(&evec);
            let A2=&evec_conj.dot(&A2).reversed_axes();
            let mut U0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let a0=(og+li*eta).powi(2);
            for i in 0..self.nsta{
                for j in 0..self.nsta{
                    U0[[i,j]]=((band[[i]]-band[[j]]).powi(2)-a0).finv();
                }
                U0[[i,i]]=Complex::new(0.0,0.0);
            }
            //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
            let mut omega_n=Array1::<f64>::zeros(self.nsta);
            let A1=A1*U0;
            let A1=A1.outer_iter();
            let A2=A2.outer_iter();
            //Zip::from(omega_n.view_mut()).and(A1).and(A2).apply(|mut x,a,b|{*x=-2.0*(a.dot(&b)).im;});
            omega_n.iter_mut().zip(A1.zip(A2)).for_each(|(mut x,(a,b))| {*x=-2.0*(a.dot(&b)).im;});
            (omega_n,band)
        }

        #[allow(non_snake_case)]
        pub fn berry_curvature_onek<S:Data<Elem=f64>>(&self,k_vec:&ArrayBase<S,Ix1>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:f64,T:f64,og:f64,spin:usize,eta:f64)->f64{
            //!给定一个 k 点, 指定 dir_1=$\alpha$, dir_2=$\beta$, T 代表温度, og= $\og$, 
            //!mu=$\mu$ 为费米能级, spin=0,1,2,3 为$\sg_0,\sg_x,\sg_y,\sg_z$,
            //!当体系不存在自旋的时候无论如何输入spin都默认 spin=0
            //!eta=$\eta$ 是一个小量
            //! 这个函数返回的是 
            //! $$ \sum_n f_n\Omega_{n,\ap\bt}^\gm(\bm k)=\sum_n \f{1}{e^{(\ve_{n\bm k}-\mu)/T/k_B}+1} \sum_{m=\not n}\f{J_{\ap,nm}^\gm v_{\bt,mn}}{(\ve_{n\bm k}-\ve_{m\bm k})^2-(\og+i\eta)^2}$$
            //! 其中 $J_\ap^\gm=\\{s_\gm,v_\ap\\}$
            let (omega_n,band)=self.berry_curvature_n_onek(&k_vec,&dir_1,&dir_2,og,spin,eta);
            let mut omega:f64=0.0;
            if T==0.0{
                /*
                for (i,(e0,o0)) in band.iter().zip(omega_n.iter()).enumerate(){
                    omega+= if e0> &mu {0.0} else {*o0};
                }
                */
                omega=omega_n.iter().zip(band.iter()).fold(0.0,|acc,(x,y)| {if *y> mu {acc} else {acc+*x}});
            }else{
                let beta=(T*8.617e-5).recip();
                let fermi_dirac=band.mapv(|x| ((beta*(x-mu)).exp()+1.0).recip());
                omega=(omega_n*fermi_dirac).sum();
            }
            omega
        }
        #[allow(non_snake_case)]
        pub fn berry_curvature<S:Data<Elem=f64>>(&self,k_vec:&ArrayBase<S,Ix2>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:f64,T:f64,og:f64,spin:usize,eta:f64)->Array1::<f64>{
        //!这个是用来并行计算大量k点的贝利曲率
        //!这个可以用来画能带上的贝利曲率, 或者画一个贝利曲率的热图
            if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r{
                panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
            }
            let nk=k_vec.len_of(Axis(0));
            let omega:Vec<f64>=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                let omega_one=self.berry_curvature_onek(&x.to_owned(),&dir_1,&dir_2,mu,T,og,spin,eta); 
                omega_one
                }).collect();
            let omega=arr1(&omega);
            omega
        }
        #[allow(non_snake_case)]
        pub fn Hall_conductivity(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:f64,T:f64,og:f64,spin:usize,eta:f64)->f64{
        //!这个是用来计算霍尔电导的.
        //!这里采用的是均匀撒点的方法, 利用 berry_curvature, 我们有
        //!$$\sg_{\ap\bt}^\gm=\f{1}{N(2\pi)^r V}\sum_{\bm k} \Og_{\ap\bt}(\bm k),$$ 其中 $N$ 是 k 点数目, 
            let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
            let nk:usize=kvec.len_of(Axis(0));
            let omega=self.berry_curvature(&kvec,&dir_1,&dir_2,mu,T,og,spin,eta);
            let conductivity:f64=omega.sum()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
            conductivity
        }
        #[allow(non_snake_case)]
        ///这个是采用自适应积分算法来计算霍尔电导的, 一般来说, 我们建议 re_err 设置为 1, 而 ab_err 设置为 0.01
        pub fn Hall_conductivity_adapted(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:f64,T:f64,og:f64,spin:usize,eta:f64,re_err:f64,ab_err:f64)->f64{
            let mut k_range=gen_krange(k_mesh);//将要计算的区域分成小块
            let n_range=k_range.len_of(Axis(0));
            let ab_err=ab_err/(n_range as f64);
            let use_fn=|k0:&Array1::<f64>| self.berry_curvature_onek(k0,&dir_1,&dir_2,mu,T,og,spin,eta);
            let inte=|k_range| adapted_integrate_quick(&use_fn,&k_range,re_err,ab_err);
            let omega:Vec<f64>=k_range.axis_iter(Axis(0)).into_par_iter().map(|x| { inte(x.to_owned())}).collect();
            let omega:Array1::<f64>=arr1(&omega);
            let conductivity:f64=omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
            conductivity
        }
        ///用来计算多个 $\mu$ 值的, 这个函数是先求出 $\Omega_n$, 然后再分别用不同的费米能级来求和, 这样速度更快, 因为避免了重复求解 $\Omega_n$, 但是相对来说更耗内存, 而且不能做到自适应积分算法.
        pub fn Hall_conductivity_mu(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,mu:&Array1::<f64>,T:f64,og:f64,spin:usize,eta:f64)->Array1::<f64>{
            let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
            let nk:usize=kvec.len_of(Axis(0));
            let (omega_n,band):(Vec<_>,Vec<_>)=kvec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                let (omega_n,band)=self.berry_curvature_n_onek(&x.to_owned(),&dir_1,&dir_2,og,spin,eta); 
                (omega_n,band)
                }).collect();
            let omega_n=Array2::<f64>::from_shape_vec((nk, self.nsta),omega_n.into_iter().flatten().collect()).unwrap();
            let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
            let n_mu:usize=mu.len();
            let conductivity=if T==0.0{
                let conductivity_new:Vec<f64>=mu.into_par_iter().map(|x| {
                    let mut omega=Array1::<f64>::zeros(nk);
                    for k in 0..nk{
                        for i in 0..self.nsta{
                            omega[[k]]+= if band[[k,i]]> *x {0.0} else {omega_n[[k,i]]};
                        }
                    }
                    omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap()/(nk as f64)
                }).collect();
                Array1::<f64>::from_vec(conductivity_new)
            }else{
                let beta=1.0/(T*8.617e-5);
                let conductivity_new:Vec<f64>=mu.into_par_iter().map(|x| {
                    let fermi_dirac=band.mapv(|x0| 1.0/((beta*(x0-x)).exp()+1.0));
                    let omega:Vec<f64>=omega_n.axis_iter(Axis(0)).zip(fermi_dirac.axis_iter(Axis(0))).map(|(a,b)| {(&a*&b).sum()}).collect();
                    let omega:Array1::<f64>=arr1(&omega);
                    omega.sum()*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap()/(nk as f64)
                }).collect();
                Array1::<f64>::from_vec(conductivity_new)
            };
            conductivity
        }

        pub fn berry_curvature_dipole_n_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array1<f64>,Array1<f64>){
            //! 这个是用来计算 $$\pdv{\ve_{n\bm k}}{k_\gm}\Og_{n,\ap\bt}$$
            //!
            //!这里需要注意的一点是, 一般来说对于 $\p_\ap\ve_{\bm k}$, 需要用差分法来求解, 我这里提供了一个算法. 
            //!$$ \ve_{\bm k}=U^\dag H_{\bm k} U\Rightarrow \pdv{\ve_{\bm k}}{\bm k}=U^\dag\pdv{H_{\bm k}}{\bm k}U+\pdv{U^\dag}{\bm k} H_{\bm k}U+U^\dag H_{\bm k}\pdv{U}{\bm k}$$
            //!因为 $U^\dag U=1\Rightarrow \p_{\bm k}U^\dag U=-U^\dag\p_{\bm k}U$, $\p_{\bm k}H_{\bm k}=v_{\bm k}$我们有
            //!$$\pdv{\ve_{\bm k}}{\bm k}=v_{\bm k}+\lt[\ve_{\bm k},U^\dag\p_{\bm k}U\rt]$$
            //!而这里面唯一比较难求的项是 $D_{\bm k}=U^\dag\p_{\bm k}U$. 按照 vanderbilt 2008 年的论文中的公式, 用微扰论有 
            //!$$D_{mn,\bm k}=\left\\{\\begin{aligned}\f{v_{mn,\bm k}}{\ve_n-\ve_m} \quad &\text{if}\\ m\\ =\not n\\\ 0 \quad \quad &\text{if}\\ m\\ = n\\end{aligned}\right\.$$
            //!我们观察到第二项对对角部分没有贡献, 所以我们可以直接设置为
            //!$$\pdv{\ve_{\bm k}}{\bm k}=\text{diag}\lt(v_{\bm k}\rt)$$
            //我们首先求解 omega_n 和 U^\dag j

        let li:Complex<f64>=1.0*Complex::i();
        let (band,evec)=self.solve_onek(&k_vec);
        let mut v:Array3::<Complex<f64>>=self.gen_v(k_vec);
        let mut J:Array3::<Complex<f64>>=v.clone();
        let mut v0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//这个是速度项, 对应的dir_3 的速度
        for r in 0..self.dim_r{
            v0=v0+v.slice(s![r,..,..]).to_owned()*dir_3[[r]];
        }
        if self.spin {
            let mut X:Array2::<Complex<f64>>=Array2::eye(self.nsta);
            let pauli:Array2::<Complex<f64>>= match spin{
                0=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,1.0+0.0*li]]),
                1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
            };
            X=kron(&pauli,&Array2::eye(self.norb));
            for i in 0..self.dim_r{
                let j=J.slice(s![i,..,..]).to_owned();
                let j=anti_comm(&X,&j)/2.0; //这里做反对易
                J.slice_mut(s![i,..,..]).assign(&(j*dir_1[[i]]));
                v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
            }
        }else{ 
            if spin !=0{
                println!("Warning, the model haven't got spin, so the spin input will be ignord");
            }
            for i in 0..self.dim_r{
                J.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_1[[i]],0.0));
                v.slice_mut(s![i,..,..]).mul_assign(Complex::new(dir_2[[i]],0.0));
            }
        };

        let J:Array2::<Complex<f64>>=J.sum_axis(Axis(0));
        let v:Array2::<Complex<f64>>=v.sum_axis(Axis(0));
        let evec_conj:Array2::<Complex<f64>>=evec.clone().map(|x| x.conj()).to_owned();
        let v0=v0.dot(&evec.clone().reversed_axes());
        let v0=&evec_conj.dot(&v0);
        let partial_ve=v0.diag().map(|x| x.re);
        let A1=J.dot(&evec.clone().reversed_axes());
        let A1=&evec_conj.dot(&A1);
        let A2=v.dot(&evec.reversed_axes());
        let A2=&evec_conj.dot(&A2);
        let mut U0=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
        for i in 0..self.nsta{
            for j in 0..self.nsta{
                if i != j{
                    U0[[i,j]]=1.0/((band[[i]]-band[[j]]).powi(2)-(og+li*eta).powi(2));
                }else{
                    U0[[i,j]]=Complex::new(0.0,0.0);
                }
            }
        }
        //let omega_n:Array1::<f64>=(-Complex::new(2.0,0.0)*(A1*U0).dot(&A2)).diag().map(|x| x.im).to_owned();
        let mut omega_n=Array1::<f64>::zeros(self.nsta);
        let A1=A1*U0;
        for i in 0..self.nsta{
            omega_n[[i]]=-2.0*A1.slice(s![i,..]).dot(&A2.slice(s![..,i])).im;
        }
        
        //let (omega_n,band)=self.berry_curvature_n_onek(&k_vec,&dir_1,&dir_2,og,spin,eta);
        let omega_n:Array1::<f64>=omega_n*partial_ve;
        (omega_n,band) //最后得到的 D
        }
        pub fn berry_curvature_dipole_n(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,og:f64,spin:usize,eta:f64)->(Array2::<f64>,Array2::<f64>){
            //这个是在 onek的基础上进行并行计算得到一系列k点的berry curvature dipole
            //!This function performs parallel computation based on the onek function to obtain a series of Berry curvature dipoles at different k-points.
            //!这个方法用的是对费米分布的修正, 因为高阶的dipole 修正导致的非线性霍尔电导为 $$\sg_{\ap\bt\gm}=\tau\int\dd\bm k\sum_n\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}\lt\.\pdv{f_{\bm k}}{\ve}\rt\rvert_{E=\ve_{n\bm k}}.$$ 所以我们这里输出的是 
            //!$$\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}.$$ 
            if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
                panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
            }
            let nk=k_vec.len_of(Axis(0));
            let (omega,band):(Vec<_>,Vec<_>)=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                let (omega_one,band)=self.berry_curvature_dipole_n_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,og,spin,eta);
                (omega_one,band)
                }).collect();
            let omega=Array2::<f64>::from_shape_vec((nk, self.nsta),omega.into_iter().flatten().collect()).unwrap();
            let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
            (omega,band)
        }
        pub fn Nonlinear_Hall_conductivity_Extrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,mu:&Array1<f64>,T:f64,og:f64,spin:usize,eta:f64)->Array1<f64>{
            //这个是用 berry curvature dipole 对整个布里渊去做积分得到非线性霍尔电导, 是extrinsic 的 
            //!This function calculates the extrinsic nonlinear Hall conductivity by integrating the Berry curvature dipole over the entire Brillouin zone. The Berry curvature dipole is first computed at a series of k-points using parallel computation based on the onek function.

            //! 我们基于 berry_curvature_n_dipole 来并行得到所有 k 点的 $\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$,
            //! 但是我们最后的公式为
            //! $$\\mathcal D_{\ap\bt\gm}=\int \dd\bm k \sum_n\lt(-\pdv{f_{n}}{\ve}\rt)\p_\gm\ve_{n\bm k}\Og_{n,\ap\bt}$$
            //! 然而, 
            //! $$-\pdv{f_{n}}{\ve}=\beta\f{e^{beta(\ve_n-\mu)}}{(e^{beta(\ve_n-\mu)}+1)^2}=\beta f_n(1-f_n)$$
            //! 对于 T=0 的情况, 我们将采用四面体积分来替代, 这个需要很高的k点密度, 不建议使用
            //! 对于 T!=0 的情况, 我们会采用类似 Dos 的方法来计算


            if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
                panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
            }
            let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
            let nk:usize=kvec.len_of(Axis(0));
            //为了节省内存, 本来是可以直接算完求和, 但是为了方便, 我是先存下来再算, 让程序结构更合理
            let (omega,band)=self.berry_curvature_dipole_n(&kvec,&dir_1,&dir_2,&dir_3,og,spin,eta);
            let omega=omega.into_raw_vec();
            let band=band.into_raw_vec();
            let n_e=mu.len();
            let mut conductivity=Array1::<f64>::zeros(n_e);
            if T !=0.0{
                let beta=1.0/T/(8.617e-5);
                let use_iter=band.iter().zip(omega.iter()).par_bridge();
                conductivity=use_iter.fold(|| Array1::<f64>::zeros(n_e),|acc,(energy,omega0)|{
                    let f=1.0/(beta*(mu-*energy)).mapv(|x| x.exp()+1.0);
                    acc+&f*(1.0-&f)*beta**omega0
                }).reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
                conductivity=conductivity.clone()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
            }else{
                //采用四面体积分法, 或者对于二维体系, 采用三角形积分法
                //积分的思路是, 通过将一个六面体变成5个四面体, 然后用线性插值的方法, 得到费米面,
                //以及费米面上的数, 最后, 通过积分算出来结果
                panic!("When T=0, the algorithm have not been writed, please wait for next version");
            }
            return conductivity
        }

        pub fn berry_connection_dipole_onek(&self,k_vec:&Array1::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,spin:usize)->(Array1::<f64>,Array1::<f64>,Option<Array1<f64>>){
            //!这个是根据 Nonlinear_Hall_conductivity_intrinsic 的注释, 当不存在自旋的时候提供
            //!$$v_\ap G_{\bt\gm}-v_\bt G_{\ap\gm}$$
            //!其中 $$ G_{ij}=2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3} $$
            //!如果存在自旋, 即spin不等于0, 则还存在 $\p_{h_i} G_{jk}$ 项, 具体请看下面的非线性霍尔部分
            //!我们这里暂时不考虑磁场, 只考虑电场
            let mut v:Array3::<Complex<f64>>=self.gen_v(&k_vec);//这是速度算符
            let mut J=v.clone();
            let (band,evec)=self.solve_onek(&k_vec);//能带和本征值
            let evec_conj=evec.clone().mapv(|x| x.conj());//本征值的复共轭
            for i in 0..self.dim_r{
                let v_s=v.slice(s![i,..,..]).to_owned();
                let v_s=evec_conj.clone().dot(&(v_s.dot(&evec.clone().reversed_axes())));//变换到本征态基函数
                v.slice_mut(s![i,..,..]).assign(&v_s);//将 v 变换到以本征态为基底
            }
            //现在速度算符已经是以本征态为基函数
            let mut v_1=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//三个方向的速度算符
            let mut v_2=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            let mut v_3=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            for i in 0..self.dim_r{
                v_1=v_1.clone()+v.slice(s![i,..,..]).to_owned()*Complex::new(dir_1[[i]],0.0);
                v_2=v_2.clone()+v.slice(s![i,..,..]).to_owned()*Complex::new(dir_2[[i]],0.0);
                v_3=v_3.clone()+v.slice(s![i,..,..]).to_owned()*Complex::new(dir_3[[i]],0.0);
            }
            //三个方向的速度算符都得到了
            let mut U0=Array2::<f64>::zeros((self.nsta,self.nsta));
            for i in 0..self.nsta{
                for j in 0..self.nsta{
                    if (band[[i]]-band[[j]]).abs() < 1e-5{
                        U0[[i,j]]=0.0;
                    }else{
                        U0[[i,j]]=1.0/(band[[i]]-band[[j]]);
                    }
                }
            }
            //这样U0[[i,j]]=1/(E_i-E_j), 这样就可以省略判断, 减少计算量

            //开始计算能带的导数, 详细的公式请看 berry_curvature_dipole_onek 的公式
            //其实就是速度算符的对角项
            //开始计算速度的偏导项, 这里偏导来自实空间
            let partial_ve_1=v_1.diag().map(|x| x.re);
            let partial_ve_2=v_2.diag().map(|x| x.re);
            let partial_ve_3=v_3.diag().map(|x| x.re);

            //开始最后的计算
            if self.spin{//如果考虑自旋, 我们就计算 \partial_h G_{ij}
                let mut S:Array2::<Complex<f64>>=Array2::eye(self.nsta);
                let li=Complex::<f64>::new(0.0,1.0);
                let pauli:Array2::<Complex<f64>>= match spin{
                    0=> Array2::<Complex<f64>>::eye(2),
                    1=> arr2(&[[0.0+0.0*li,1.0+0.0*li],[1.0+0.0*li,0.0+0.0*li]])/2.0,
                    2=> arr2(&[[0.0+0.0*li,0.0-1.0*li],[0.0+1.0*li,0.0+0.0*li]])/2.0,
                    3=> arr2(&[[1.0+0.0*li,0.0+0.0*li],[0.0+0.0*li,-1.0+0.0*li]])/2.0,
                    _=>panic!("Wrong, spin should be 0, 1, 2, 3, but you input {}",spin),
                };
                let X=kron(&pauli,&Array2::eye(self.norb));
                let mut S=Array3::<Complex<f64>>::zeros((self.dim_r,self.nsta,self.nsta));
                for i in 0..self.dim_r{
                    let v0=J.slice(s![i,..,..]).to_owned();
                    let v0=anti_comm(&X,&v0)/2.0;
                    let v0=evec_conj.clone().dot(&(v0.dot(&evec.clone().reversed_axes())));//变换到本征态基函数
                    S.slice_mut(s![i,..,..]).assign(&v0);
                }
                let mut s_1=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));//三个方向的速度算符
                let mut s_2=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
                let mut s_3=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
                for i in 0..self.dim_r{
                    s_1=s_1.clone()+S.slice(s![i,..,..]).to_owned()*Complex::new(dir_1[[i]],0.0);
                    s_2=s_2.clone()+S.slice(s![i,..,..]).to_owned()*Complex::new(dir_2[[i]],0.0);
                    s_3=s_3.clone()+S.slice(s![i,..,..]).to_owned()*Complex::new(dir_3[[i]],0.0);
                }
                let G_23:Array1::<f64>={//用来计算  beta gamma 的 G 
                    let A=&v_2*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                    let mut G=Array1::<f64>::zeros(self.nsta);
                    for i in 0..self.nsta{
                        G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                    }
                    G
                };
                let G_13_h:Array1::<f64>={//用来计算 alpha gamma 的 G 
                    let A=&s_1*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                    let mut G=Array1::<f64>::zeros(self.nsta);
                    for i in 0..self.nsta{
                        G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                    }
                    G
                };
                //开始计算partial_s
                let partial_s_1=s_1.clone().diag().map(|x| x.re);
                let partial_s_2=s_2.clone().diag().map(|x| x.re);
                let partial_s_3=s_3.clone().diag().map(|x| x.re);
                let mut partial_s=Array2::<f64>::zeros((self.dim_r,self.nsta));
                for r in 0..self.dim_r{
                    let s0=S.slice(s![r,..,..]).to_owned();
                    partial_s.slice_mut(s![r,..]).assign(&s0.diag().map(|x| x.re));//\p_i s算符的对角项
                }
                //开始计算partial G
                let partial_G:Array1::<f64>={
                    let mut A=Array1::<Complex<f64>>::zeros(self.nsta);//第一项
                    for i in 0..self.nsta{
                        for j in 0..self.nsta{
                            A[[i]]+=3.0*(partial_s_1[[i]]-partial_s_1[[j]])*v_2[[i,j]]*v_3[[j,i]]*U0[[i,j]].powi(4);
                        }
                    }
                    let mut B=Array1::<Complex<f64>>::zeros(self.nsta);//第二项
                    for n in 0..self.nsta{
                        for n1 in 0..self.nsta{
                            for n2 in 0..self.nsta{
                                B[[n]]+=s_1[[n,n2]]*(v_2[[n2,n1]]*v_3[[n1,n]]+v_3[[n2,n1]]*v_2[[n1,n]])*U0[[n,n1]].powi(3)*U0[[n,n2]];
                            }
                        }
                    }
                    let mut C=Array1::<Complex<f64>>::zeros(self.nsta);//第三项
                    for n in 0..self.nsta{
                        for n1 in 0..self.nsta{
                            for n2 in 0..self.nsta{
                                C[[n]]+=s_1[[n1,n2]]*(v_2[[n2,n]]*v_3[[n,n1]]+v_3[[n2,n]]*v_2[[n,n1]])*U0[[n,n1]].powi(3)*U0[[n1,n2]];
                            }
                        }
                    }
                    2.0*(A-B-C).map(|x| x.re)
                };
                //计算结束
                //开始最后的输出
                return ((partial_s_1*G_23-partial_ve_2*G_13_h),band,Some(partial_G))
            }else{
                //开始计算 G_{ij}
                //G_{ij}=2Re\sum_{m\neq n} v_{i,nm}v_{j,mn}/(E_n-E_m)^3
                let G_23:Array1::<f64>={//用来计算  beta gamma 的 G 
                    let A=&v_2*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                    let mut G=Array1::<f64>::zeros(self.nsta);
                    for i in 0..self.nsta{
                        G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                    }
                    G
                };
                let G_13:Array1::<f64>={//用来计算 alpha gamma 的 G 
                    let A=&v_1*(U0.map(|x| Complex::<f64>::new(x.powi(3),0.0)));
                    let mut G=Array1::<f64>::zeros(self.nsta);
                    for i in 0..self.nsta{
                        G[[i]]=A.slice(s![i,..]).dot(&v_3.slice(s![..,i])).re*2.0
                    }
                    G
                };
                return (partial_ve_1*G_23-partial_ve_2*G_13,band,None)
            }
        }
        pub fn berry_connection_dipole(&self,k_vec:&Array2::<f64>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,spin:usize)->(Array2<f64>,Array2<f64>,Option<Array2<f64>>){
            //! 这个是基于 onek 的, 进行关于 k 点并行求解
            if dir_1.len() !=self.dim_r || dir_2.len() != self.dim_r || dir_3.len() != self.dim_r{
                panic!("Wrong, the dir_1 or dir_2 you input has wrong length, it must equal to dim_r={}, but you input {},{}",self.dim_r,dir_1.len(),dir_2.len())
            }
            let nk=k_vec.len_of(Axis(0));
            
            if self.spin{
                let ((omega,band),partial_G):((Vec<_>,Vec<_>),Vec<_>)=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                    let (omega_one,band,partial_G)=self.berry_connection_dipole_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,spin); 
                    let partial_G=partial_G.unwrap();
                    ((omega_one,band),partial_G)
                    }).collect();

                let omega=Array2::<f64>::from_shape_vec((nk, self.nsta),omega.into_iter().flatten().collect()).unwrap();
                let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
                let partial_G=Array2::<f64>::from_shape_vec((nk, self.nsta),partial_G.into_iter().flatten().collect()).unwrap();

                return (omega,band,Some(partial_G))
            }else{
                let (omega,band):(Vec<_>,Vec<_>)=k_vec.axis_iter(Axis(0)).into_par_iter().map(|x| {
                    let (omega_one,band,partial_G)=self.berry_connection_dipole_onek(&x.to_owned(),&dir_1,&dir_2,&dir_3,spin); 
                    (omega_one,band)
                    }).collect();
                let omega=Array2::<f64>::from_shape_vec((nk, self.nsta),omega.into_iter().flatten().collect()).unwrap();
                let band=Array2::<f64>::from_shape_vec((nk, self.nsta),band.into_iter().flatten().collect()).unwrap();
                return (omega,band,None)
            }
        }
        pub fn Nonlinear_Hall_conductivity_Intrinsic(&self,k_mesh:&Array1::<usize>,dir_1:&Array1::<f64>,dir_2:&Array1::<f64>,dir_3:&Array1::<f64>,mu:&Array1<f64>,T:f64,spin:usize)->Array1<f64>{
            //! The Intrinsic Nonlinear Hall Conductivity arises from the correction of the Berry connection by the electric and magnetic fields [PRL 112, 166601 (2014)]. The formula employed is:
            //!$$\tilde\bm\Og_{\bm k}=\nb_{\bm k}\times\lt(\bm A_{\bm k}+\bm A_{\bm k}^\prime\rt)$$
            //!and the $\bm A_{i,\bm k}^\prime=F_{ij}B_j+G_{ij}E_j$, where
            //!$$ 
            //!\\begin{aligned}
            //!F_{ij}&=\text{Im}\sum_{m=\not n}\f{v_{i,nm}\og_{j,mn}}{\lt(\ve_{n}-\ve_m\rt)^2}\\\\
            //!G_{ij}&=2\text{Re}\sum_{m=\not n}\f{v_{i,nm}v_{j,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
            //!\og_{\ap,mn}&=-i\ep_{\ap\bt\gm}\sum_{l=\not n}\f{\lt(v_{\bt,ml}+\p_\bt \ve_{\bm k}\dt_{ml}\rt)v_{\gm,ln}}{\ve_l-\ve_n}
            //!\\end{aligned}
            //!$$
            //!最后我们有
            //!$$
            //!\bm j^\prime=\bm E\times\int\f{\dd\bm k}{(2\pi)^3}\lt[\p_{\bm k}\ve_{\bm k}\times\bm A^\prime+\bm\Og\lt(\bm B\cdot\bm m\rt)\rt]\pdv{f_{\bm k}}{\ve}
            //!$$
            //!对其对电场和磁场进行偏导, 有
            //!$$
            //!\\begin{aligned}
            //!\f{\p^2 j_{\ap}^\prime}{\p E_\bt\p E_\gm}&=\int\f{\dd\bm k}{(2\pi)^3}\lt(\p_\ap\ve_{\bm k} G_{\bt\gm}-\p_\bt\ve_{\bm k} G_{\ap\gm}\rt)\pdv{f_{\bm k}}{\ve}\\\\
            //!\f{\p^2 j_{\ap}^\prime}{\p E_\bt\p B_\gm}&=\int\f{\dd\bm k}{(2\pi)^3}\lt(\p_\ap\ve_{\bm k} F_{\bt\gm}-\p_\bt\ve_{\bm k} F_{\ap\gm}+\ep_{\ap\bt\ell}\Og_{\ell} m_\gm\rt)\pdv{f_{\bm k}}{\ve}
            //!\\end{aligned}
            //!$$
            //!由于存在 $\pdv{f_{\bm k}}{\ve}$, 不建议将温度 T=0
            //!
            //!可以考虑当 T=0 时候, 利用高斯公式, 将费米面内的部分进行积分, 得到精确解. 但是我现在还没办法很好的求解费米面, 所以暂时不考虑这个算法.而且对于二维体系, 公式还不一样, 还得分步讨论, 后面有时间再考虑这个程序.
            //!
            //!对于自旋霍尔效应, 按照文章 [PRL 112, 166601 (2014)], 非线性自旋霍尔电导为
            //!$$\sg_{\ap\bt\gm}^i=-\int\dd\bm k \lt[\f{1}{2}f_{\bm k}\pdv{G_{\bt\gm}}{h_\ap}+\pdv{f_{\bm k}}{\ve}\lt(\p_{\ap}s_{\bm k}^i G_{\bt\gm}-\p_\bt\ve_{\bm k}G_{\ap\gm}^h\rt)\rt]$$
            //!其中
            //!$$\f{\p G_{\bt\gm,n}}{\p h_\ap}=2\text{Re}\sum_{n^\pr =\not n}\f{3\lt(s^i_{\ap,n}-s^i_{\ap,n_1}\rt)v_{\bt,nn_1} v_{\gm,n^\pr n}}{\lt(\ve_n-\ve_{n^\pr}\rt)^4}-2\text{Re}\sum_{n_1=\not n}\sum_{n_2=\not n}\lt[\f{s^i_{\ap,nn_2} v_{\bt,n_2n_1} v_{\gm,n_1 n}}{\lt(\ve_n-\ve_{n_1}\rt)^3(\ve_n-\ve_{n_2})}+(\bt \leftrightarrow \gm)\rt]-2\text{Re}\sum_{n_1=\not n}\sum_{n_2=\not n_1}\lt[\f{s^i_{\ap,n_1n_2} v_{\bt,n_2n} v_{\gm,n n_1}}{\lt(\ve_n-\ve_{n_1}\rt)^3(\ve_{n_1}-\ve_{n_2})}+(\bt \leftrightarrow \gm)\rt]$$
            //!以及
            //!$$
            //!\lt\\\{\\begin{aligned}
            //!G_{\ap\bt}&=2\text{Re}\sum_{m=\not n}\f{v_{\ap,nm}v_{\bt,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
            //!G_{\ap\bt}^h&=2\text{Re}\sum_{m=\not n}\f{s^i_{\ap,nm}v_{\bt,mn}}{\lt(\ve_n-\ve_m\rt)^3}\\\\
            //!\\end{aligned}\rt\.
            //!$$
            //!
            //!这里 $s^i_{\ap,mn}$ 的具体形式, 原文中没有明确给出, 但是我根据霍尔效应的类比, 我猜是
            //!$\\\{\hat s^i,v_\ap\\\}$


            let kvec:Array2::<f64>=gen_kmesh(&k_mesh);
            let nk:usize=kvec.len_of(Axis(0));
            let (omega,band,mut partial_G):(Array2<f64>,Array2<f64>,Option<Array2<f64>>)=self.berry_connection_dipole(&kvec,&dir_1,&dir_2,&dir_3,spin);
            let omega=omega.into_raw_vec();
            let omega=Array1::from(omega);
            let band0=band.clone();
            let band=band.into_raw_vec();
            let band=Array1::from(band);
            let n_e=mu.len();
            let mut conductivity=Array1::<f64>::zeros(n_e);
            if T !=0.0{
                let beta=1.0/T/8.617e-5;
                let use_iter=band.iter().zip(omega.iter()).par_bridge();
                conductivity=use_iter.fold(|| Array1::<f64>::zeros(n_e),|acc,(energy,omega0)|{
                    let f=1.0/((beta*(mu-*energy)).mapv(|x| x.exp()+1.0));
                    acc+&f*(1.0-&f)*beta**omega0
                }).reduce(|| Array1::<f64>::zeros(n_e), |acc, x| acc + x);
                if self.spin{
                    let partial_G=partial_G.unwrap();
                    let conductivity_new:Vec<f64>=mu.into_par_iter().map(|x| {
                        let f=band0.map(|x0| 1.0/((beta*(x-x0)).exp()+1.0));
                        let mut omega=Array1::<f64>::zeros(nk);
                        for i in 0..nk{
                            omega[[i]]=(partial_G.row(i).to_owned()*f.row(i).to_owned()).sum();
                        }
                        omega.sum()/2.0
                    }).collect();
                    let conductivity_new=Array1::<f64>::from_vec(conductivity_new);
                    conductivity=conductivity.clone()+conductivity_new;
                }
                conductivity=conductivity.clone()/(nk as f64)*(2.0*PI).powi(self.dim_r as i32)/self.lat.det().unwrap();
            }else{
                //采用四面体积分法, 或者对于二维体系, 采用三角形积分法
                //积分的思路是, 通过将一个六面体变成5个四面体, 然后用线性插值的方法, 得到费米面,
                //以及费米面上的数, 最后, 通过积分算出来结果
                panic!("the code can not support for T=0");
            }
            return conductivity
        }
    }
}
