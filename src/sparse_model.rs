pub mod sparse_model{
    use crate::{Model,Model_sparse};
    use gnuplot::Major;
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
    use std::io::Write;
    use std::fs::File;
    use std::ops::AddAssign;
    use std::ops::MulAssign;


    #[derive(Clone,Debug)]
    struct hop_element{
        R:[isize;3],
        orb_i:usize,
        orb_j:usize,
        hop:Complex<f64>,
        rmatrix:Option<[Complex<f64>;3]>
    }
    impl Model_sparse<hop_element>{
        pub fn from_Model(model:Model)->Model_sparse<hop_element>{
            let mut ham_sparse=Vec::new();
            if model.rmatrix.len_of(Axis(1))!=1{
                for (r,(hamR,(ham,rmatrix))) in model.hamR.outer_iter().zip(model.ham.outer_iter().zip(model.rmatrix.outer_iter())).enumerate(){
                    let mut R=[0;3];
                    for i in 0..model.dim_r{
                        R[i]=hamR[[i]];
                    }
                    //构造稀疏矩阵的哈密顿量
                    for i in 0..model.nsta{
                        for j in 0..model.nsta{
                            let mut hop_rmatrix=[Complex::new(0.0,0.0);3];
                            hop_rmatrix[0]=rmatrix[[0,i,j]];
                            hop_rmatrix[1]=rmatrix[[1,i,j]];
                            hop_rmatrix[2]=rmatrix[[2,i,j]];
                            if ham[[i,j]].norm() < 1e-6{
                                let mut R=[0;3];
                                for dim in 0..model.dim_r{
                                    R[dim]=hamR[[dim]];
                                }
                                let hopping=hop_element{
                                    R,
                                    orb_i:i,
                                    orb_j:j,
                                    hop:ham[[i,j]],
                                    rmatrix:Some(hop_rmatrix),
                                };
                                ham_sparse.push(hopping);
                            }else{
                                continue
                            }
                        }
                    }
                }
            }else{
                for (r,(hamR,ham)) in model.hamR.outer_iter().zip(model.ham.outer_iter()).enumerate(){
                    let mut R=[0;3];
                    for i in 0..model.dim_r{
                        R[i]=hamR[[i]];
                    }
                    //构造稀疏矩阵的哈密顿量
                    for i in 0..model.nsta{
                        for j in 0..model.nsta{
                            if ham[[i,j]].norm() < 1e-6{
                                let mut R=[0;3];
                                for dim in 0..model.dim_r{
                                    R[dim]=hamR[[dim]];
                                }
                                let hopping=hop_element{
                                    R,
                                    orb_i:i,
                                    orb_j:j,
                                    hop:ham[[i,j]],
                                    rmatrix:None,
                                };
                                ham_sparse.push(hopping);
                            }else{
                                continue
                            }
                        }
                    }
                }
            }
            Model_sparse{
                dim_r:model.dim_r,
                norb:model.norb,
                nsta:model.nsta,
                natom:model.natom,
                lat:model.lat,
                orb:model.orb,
                spin:model.spin,
                atom:model.atom,
                atom_list:model.atom_list,
                ham:ham_sparse,
            }
        }
    }
}
