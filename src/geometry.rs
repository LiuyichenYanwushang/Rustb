pub mod geometry{
    use crate::{Model,gen_kmesh,comm};
    use std::fs::File;
    use std::io::Write;
    use num_complex::Complex;
    use ndarray::linalg::kron;
    use ndarray::*;
    use ndarray::prelude::*;
    use ndarray::concatenate;
    use ndarray_linalg::*;
    use std::f64::consts::PI;
    use ndarray_linalg::{Eigh, UPLO};
    use ndarray_linalg::conjugate;
    use rayon::prelude::*;
    use std::ops::AddAssign;
    use std::ops::MulAssign;

    impl Model{
        //!这个模块是用wilson loop 的方法来计算各种几何量.
        pub fn berry_loop<S>(&self,kvec:&ArrayBase<S,Ix2>,occ:&Vec<usize>)->Array1::<f64>
        where
            S:Data<Elem=f64>,
        {
            let n_k=kvec.nrows();
            let diff=&kvec.row(n_k-1)-&kvec.row(0);
            for i in diff.iter(){
                if i.fract() >1e-3{
                    panic!("wrong, the end of this loop must differ from the beginning by an integer grid vector.")
                }
            }
            let add_phase=diff.dot(&self.orb.t());
            let add_phase=if self.spin{
                let mut a=add_phase.mapv(|x| Complex::new(0.0,-2.0*x).exp());
                let b=a.clone();
                a.append(Axis(0),b.view()).unwrap();
                a
            }else{
                add_phase.mapv(|x| Complex::new(0.0,-2.0*x).exp())
            };
            let (eval,mut evec)=self.solve_all(kvec);
            let first_evec=&evec.slice(s![0,..,..]);
            let mut end_evec=Array2::<Complex<f64>>::zeros((self.nsta,self.nsta));
            Zip::from(end_evec.outer_iter_mut()).and(first_evec.outer_iter()).apply(|mut A,B|{A.assign(&(&B*&add_phase))});
            evec.slice_mut(s![n_k-1,..,..]).assign(&end_evec);
            let evec=evec.select(Axis(1),occ);
            let n_occ=occ.len();
            let evec_conj=evec.map(|x| x.conj());
            let evec=evec.slice(s![1..n_k,..,..]).to_owned();
            let evec_conj=evec_conj.slice(s![0..n_k-1,..,..]).to_owned();
            let mut ovr=Array3::zeros((n_k-1,n_occ,n_occ));
            Zip::from(ovr.outer_iter_mut()).and(evec.outer_iter()).and(evec_conj.outer_iter()).apply(|mut O,e,e_j|{
                for (i,a) in e_j.outer_iter().enumerate(){
                    for (j,b) in e.outer_iter().enumerate(){
                        O[[i,j]]=a.dot(&b);
                    }
                }
            });
            Zip::from(ovr.outer_iter_mut()).apply(|mut O|{
                let (U,S,V)=O.svd(true,true).unwrap();
                let U=U.unwrap();
                let V=V.unwrap();
                O.assign(&U.dot(&V));
            });
            let result:Array2::<Complex<f64>>=ovr.outer_iter().fold(Array2::from_diag(&Array1::<Complex<f64>>::ones(n_occ)),|acc,x| {acc.dot(&x)});
            let result=result.eigvals().unwrap();
            let result=result.mapv(|x| -x.arg());
            result
        }
        pub fn berry_flux(&self,occ:&Vec<usize>,k_start:&Array1<f64>,dir_1:&Array1<f64>,dir_2:&Array1<f64>,nk1:usize,nk2:usize)->Array3<f64>{
            //!这个函数是用来计算berry curvature 的. 根据给定的平面, 其返回一个 Array2<f64>
            //!前两个指标表示横和纵, 数值表示大小
            assert_eq!(k_start.len(),self.dim_r,"Wrong!, the k_start's length is {} but dim_r is {}, it's not equal!",k_start.len(),self.dim_r);
            assert_eq!(dir_1.len(),self.dim_r,"Wrong!, the dir_1's length is {} but dim_r is {}, it's not equal!",dir_1.len(),self.dim_r);
            assert_eq!(dir_2.len(),self.dim_r,"Wrong!, the dir_2's length is {} but dim_r is {}, it's not equal!",dir_2.len(),self.dim_r);
            //开始构造loop
            let mut k_loop=Array3::<f64>::zeros((nk1*nk2,9,self.dim_r));
            for i in 0..nk1{
                for j in 0..nk2{
                    let i0=(i as f64)/(nk1 as f64);
                    let j0=(j as f64)/(nk2 as f64);
                    let dx=1.0/(nk1 as f64);
                    let dy=1.0/(nk2 as f64);
                    let mut s=k_loop.slice_mut(s![i*nk2+j,0,..]);
                    s.assign(&(k_start+(i0+dx)*dir_1+j0*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,1,..]);
                    s.assign(&(k_start+(i0+dx)*dir_1+(j0+dy)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,2,..]);
                    s.assign(&(k_start+(i0)*dir_1+(j0+dy)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,3,..]);
                    s.assign(&(k_start+(i0-dx)*dir_1+(j0+dy)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,4,..]);
                    s.assign(&(k_start+(i0-dx)*dir_1+(j0)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,5,..]);
                    s.assign(&(k_start+(i0-dx)*dir_1+(j0-dy)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,6,..]);
                    s.assign(&(k_start+(i0)*dir_1+(j0-dy)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,7,..]);
                    s.assign(&(k_start+(i0+dx)*dir_1+(j0-dy)*dir_2));
                    let mut s=k_loop.slice_mut(s![i*nk2+j,8,..]);
                    s.assign(&(k_start+(i0+dx)*dir_1+(j0)*dir_2));
                }
            }
            let berry_flux:Vec<_>=k_loop.outer_iter().into_par_iter().map(|x| self.berry_loop(&x,occ).to_vec()).collect();
            let berry_flux=Array3::from_shape_vec((nk1,nk2,occ.len()),berry_flux.into_iter().flatten().collect()).unwrap();
            berry_flux

        }

        pub fn wannier_centre(&self,occ:&Vec<usize>,k_start:&Array1<f64>,dir_1:&Array1<f64>,dir_2:&Array1<f64>,nk1:usize,nk2:usize)->Array2::<f64>{
            //!这里是计算wcc的, 沿着第一个方向走, 沿着第二个方向积分
            if k_start.len() != self.dim_r {
                panic!("Wrong!, the k_start's length is {} but dim_r is {}, it's not equal!",k_start.len(),self.dim_r);
            }else if dir_1.len() != self.dim_r {
                panic!("Wrong!, the dir_1's length is {} but dim_r is {}, it's not equal!",dir_1.len(),self.dim_r);
            }else if dir_1.len() != self.dim_r {
                panic!("Wrong!, the dir_2's length is {} but dim_r is {}, it's not equal!",dir_2.len(),self.dim_r);
            }
            let mut kvec=Array3::zeros((nk1,nk2,self.dim_r));
            for i in 0..nk1{
                for j in 0..nk2{
                    let mut s=kvec.slice_mut(s![i,j,..]);
                    let used_k=k_start+dir_1*(i as f64)/((nk1 -1) as f64)+dir_2*(j as f64)/((nk2-1) as f64);
                    s.assign(&used_k);
                }
            }
            let nocc=occ.len();
            let mut wcc=Array2::zeros((nk1,nocc));
            Zip::from(wcc.outer_iter_mut()).and(kvec.outer_iter()).par_apply(|mut w,k|{w.assign(&self.berry_loop(&k.to_owned(),occ));});
            for mut row in wcc.outer_iter_mut(){
                row.as_slice_mut().unwrap().sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            let wcc=wcc.reversed_axes();
            wcc
        }
    }
}
