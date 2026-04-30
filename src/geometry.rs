//! Geometric and topological quantities computed via the Wilson loop method.
//!
//! This module provides the [`Berry`] trait with methods for:
//!
//! - Berry phase along a closed k-space loop.
//! - Berry curvature (flux) on a 2D k-mesh.
//! - Wannier centres (hybrid Wannier functions) via Wilson loops.
//!
//! # Wilson loop algorithm
//!
//! For a closed loop of k-points \\(\{\mathbf k_i\}\\), the overlap matrix is
//!
//! $$
//! F_{mn,\mathbf k} = \langle \psi_{m,\mathbf k} | \psi_{n,\mathbf k+\Delta\mathbf k} \rangle.
//! $$
//!
//! The overlap matrices are orthonormalized via SVD: \\(F = U V^\dagger\\).  The
//! Wilson loop is the product
//!
//! $$
//! W = \prod_i F_{\mathbf k_i},
//! $$
//!
//! whose eigenvalues \\(e^{i\Theta}\\) give the Wannier centres.
//!
//! For a closed loop that wraps the Brillouin zone, the Bloch functions at the
//! endpoints are related by a phase factor:
//!
//! $$
//! |u_{n,\mathbf k_{\text{end}}}\rangle =
//! e^{-2\pi i \bm\tau} |u_{n,\mathbf k_{\text{first}}}\rangle.
//! $$
//!
//! # Examples
//!
//! ```ignore
//! use rustb::geometry::Berry;
//!
//! let occ = vec![0]; // occupied band index
//! let phase = model.berry_loop(&loop_kvec, &occ);
//! let wcc = model.wannier_centre(&occ, &k_start, &dir1, &dir2, nk1, nk2);
//! ```

use crate::math::comm;
use crate::solve_ham::solve;
use crate::{Model, gen_kmesh};
use ndarray::concatenate;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::ops::AddAssign;
use std::ops::MulAssign;

/// Trait for computing Berry-phase, Berry curvature, and Wannier centre
/// quantities via the Wilson loop method.
pub trait Berry {
    /// Compute the Berry phase (Wannier centres) along a closed k-space loop
    /// using Wilson loops.
    ///
    /// # Parameters
    ///
    /// - `kvec`: array of k-points defining the loop (shape `(N_k, dim_r)`).
    ///   The loop must close modulo a reciprocal lattice vector.
    /// - `occ`: indices of the occupied bands.
    ///
    /// # Returns
    ///
    /// An `Array1<f64>` of Wannier centre positions (Berry phases divided by \\(2\pi\\)).
    ///
    /// # Algorithm
    ///
    /// The overlap matrices \\(F_{mn,\mathbf k}\\) are computed between adjacent
    /// k-points, orthonormalized via SVD (\\(F = UV^\dagger\\)), and multiplied
    /// along the loop.  The phases of the eigenvalues of the product give the
    /// Wannier centres.
    ///
    /// # Panics
    ///
    /// Panics if the loop endpoints do not differ by an integer reciprocal
    /// lattice vector.
    fn berry_loop<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &Vec<usize>) -> Array1<f64>
    where
        S: Data<Elem = f64>;

    /// Compute the Berry phase along a closed loop without SVD orthonormalization,
    /// using the determinant instead.
    ///
    /// This captures the total Berry phase of all occupied bands.
    fn berry_loop_det<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &Vec<usize>) -> f64
    where
        S: Data<Elem = f64>;

    /// Compute the Berry curvature (flux) on a 2D k-mesh using Wilson loops.
    ///
    /// # Parameters
    ///
    /// - `occ`: occupied band indices.
    /// - `k_start`: starting k-point.
    /// - `dir_1`: first reciprocal lattice direction.
    /// - `dir_2`: second reciprocal lattice direction.
    /// - `nk1`, `nk2`: number of k-points in each direction.
    ///
    /// # Returns
    ///
    /// An `Array3<f64>` of shape `(nk1, nk2, n_occ)` containing the Berry flux
    /// through each plaquette.
    ///
    /// This method uses Wilson loops (high accuracy, fast convergence) but
    /// requires a band gap (insulator).
    fn berry_flux(
        &self,
        occ: &Vec<usize>,
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Array3<f64>;

    /// Compute Berry phases along closed k-space loops.
    ///
    /// Each slice `kvec[i, :, :]` defines one loop.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` of shape `(nk1, n_occ)`.
    fn berry_phase(&self, occ: &Vec<usize>, kvec: &Array3<f64>) -> Array2<f64>;

    /// Compute hybrid Wannier centres via Wilson loops.
    ///
    /// The Wilson loop is taken along `dir_2` (the integration direction),
    /// while `dir_1` is the transverse direction that is sampled.
    ///
    /// # Parameters
    ///
    /// - `occ`: occupied band indices.
    /// - `k_start`: origin of the 2D k-mesh.
    /// - `dir_1`: transverse direction (sampled at `nk1` points).
    /// - `dir_2`: integration direction (`nk2` points per loop).
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` of shape `(nk1, n_occ)` with the sorted Wannier centres.
    fn wannier_centre(
        &self,
        occ: &Vec<usize>,
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Array2<f64>;
}

impl<const SPIN: bool> Berry for Model<SPIN> {
    fn berry_loop<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &Vec<usize>) -> Array1<f64>
    where
        S: Data<Elem = f64>,
    {
        let n_k = kvec.nrows();
        let diff = &kvec.row(n_k - 1) - &kvec.row(0);
        for i in diff.iter() {
            if (i - i.round()).abs() > 1e-9 {
                panic!(
                    "wrong, the end of this loop must differ from the beginning by an integer grid vector. yours {}\n",
                    i.fract()
                )
            }
        }
        let use_orb = if SPIN {
            let mut orb0 = self.orb.to_owned();
            orb0.append(Axis(0), self.orb.view());
            orb0
        } else {
            self.orb.to_owned()
        };
        let add_phase = diff.dot(&use_orb.t());
        let add_phase = add_phase.mapv(|x| Complex::new(0.0, -2.0 * x * PI).exp());
        let (eval, mut evec) = self.solve_all(kvec);
        let first_evec: &ArrayRef<_, Dim<[_; 2]>> = &evec.slice(s![0, .., ..]);
        let add_phase = Array2::from_diag(&add_phase);
        let end_evec = first_evec.to_owned().dot(&add_phase);

        evec.slice_mut(s![n_k - 1, .., ..]).assign(&end_evec);
        let evec = evec.select(Axis(1), occ);
        let n_occ = occ.len();
        let evec_conj = evec.map(|x| x.conj());
        let evec = evec.slice(s![1..n_k, .., ..]).to_owned();
        let evec_conj = evec_conj.slice(s![0..n_k - 1, .., ..]).to_owned();
        let mut ovr = Array3::zeros((n_k - 1, n_occ, n_occ));
        Zip::from(ovr.outer_iter_mut())
            .and(evec.outer_iter())
            .and(evec_conj.outer_iter())
            .for_each(|mut O, e, e_j| {
                for (i, a) in e_j.outer_iter().enumerate() {
                    for (j, b) in e.outer_iter().enumerate() {
                        O[[i, j]] = a.dot(&b);
                    }
                }
            });
        Zip::from(ovr.outer_iter_mut()).for_each(|mut O| {
            let (U, S, V) = O.svd(true, true).unwrap();
            let U = U.unwrap();
            let V = V.unwrap();
            O.assign(&U.dot(&V));
        });
        let result: Array2<Complex<f64>> = ovr.outer_iter().fold(
            Array2::from_diag(&Array1::<Complex<f64>>::ones(n_occ)),
            |acc, x| acc.dot(&x),
        );
        let result = result.eigvals().unwrap();
        let result = result.mapv(|x| -x.arg());
        result
    }

    fn berry_loop_det<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &Vec<usize>) -> f64
    where
        S: Data<Elem = f64>,
    {
        let n_k = kvec.nrows();
        let diff = &kvec.row(n_k - 1) - &kvec.row(0);
        for i in diff.iter() {
            if (i - i.round()).abs() > 1e-9 {
                panic!(
                    "wrong, the end of this loop must differ from the beginning by an integer grid vector. yours {}\n",
                    i.fract()
                )
            }
        }
        let use_orb = if SPIN {
            let mut orb0 = self.orb.to_owned();
            orb0.append(Axis(0), self.orb.view());
            orb0
        } else {
            self.orb.to_owned()
        };
        let add_phase = diff.dot(&use_orb.t());
        let add_phase = add_phase.mapv(|x| Complex::new(0.0, -2.0 * x * PI).exp());
        let (eval, mut evec) = self.solve_all(kvec);
        let first_evec: &ArrayRef<_, Dim<[_; 2]>> = &evec.slice(s![0, .., ..]);
        let add_phase = Array2::from_diag(&add_phase);
        let end_evec = first_evec.to_owned().dot(&add_phase);

        evec.slice_mut(s![n_k - 1, .., ..]).assign(&end_evec);
        let evec = evec.select(Axis(1), occ);
        let n_occ = occ.len();
        let evec_conj = evec.map(|x| x.conj());
        let evec = evec.slice(s![1..n_k, .., ..]).to_owned();
        let evec_conj = evec_conj.slice(s![0..n_k - 1, .., ..]).to_owned();
        let mut ovr = Array3::zeros((n_k - 1, n_occ, n_occ));
        Zip::from(ovr.outer_iter_mut())
            .and(evec.outer_iter())
            .and(evec_conj.outer_iter())
            .for_each(|mut O, e, e_j| {
                for (i, a) in e_j.outer_iter().enumerate() {
                    for (j, b) in e.outer_iter().enumerate() {
                        O[[i, j]] = a.dot(&b);
                    }
                }
            });
        let result: Array2<Complex<f64>> = ovr.outer_iter().fold(
            Array2::from_diag(&Array1::<Complex<f64>>::ones(n_occ)),
            |acc, x| acc.dot(&x),
        );
        let result = result.det().unwrap();
        let result = -result.arg();
        result
    }

    fn berry_flux(
        &self,
        occ: &Vec<usize>,
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Array3<f64> {
        assert_eq!(
            k_start.len(),
            self.dim_r(),
            "Wrong!, the k_start's length is {} but dim_r is {}, it's not equal!",
            k_start.len(),
            self.dim_r()
        );
        assert_eq!(
            dir_1.len(),
            self.dim_r(),
            "Wrong!, the dir_1's length is {} but dim_r is {}, it's not equal!",
            dir_1.len(),
            self.dim_r()
        );
        assert_eq!(
            dir_2.len(),
            self.dim_r(),
            "Wrong!, the dir_2's length is {} but dim_r is {}, it's not equal!",
            dir_2.len(),
            self.dim_r()
        );
        // Construct plaquette loops
        let mut k_loop = Array3::<f64>::zeros((nk1 * nk2, 5, self.dim_r()));
        for i in 0..nk1 {
            for j in 0..nk2 {
                let i0 = (i as f64) / (nk1 as f64);
                let j0 = (j as f64) / (nk2 as f64);
                let dx = 1.0 / (nk1 as f64);
                let dy = 1.0 / (nk2 as f64);
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 0, ..]);
                s.assign(&(k_start + (i0) * dir_1 + (j0) * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 1, ..]);
                s.assign(&(k_start + (i0 + dx) * dir_1 + j0 * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 2, ..]);
                s.assign(&(k_start + (i0 + dx) * dir_1 + (j0 + dy) * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 3, ..]);
                s.assign(&(k_start + (i0) * dir_1 + (j0 + dy) * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 4, ..]);
                s.assign(&(k_start + (i0) * dir_1 + (j0) * dir_2));
            }
        }
        let berry_flux: Vec<_> = k_loop
            .outer_iter()
            .into_par_iter()
            .map(|x| self.berry_loop(&x, occ).to_vec())
            .collect();
        let berry_flux = Array3::from_shape_vec(
            (nk1, nk2, occ.len()),
            berry_flux.into_iter().flatten().collect(),
        )
        .unwrap();
        berry_flux
    }

    fn berry_phase(&self, occ: &Vec<usize>, kvec: &Array3<f64>) -> Array2<f64> {
        let nk1 = kvec.shape()[0];
        let nk2 = kvec.shape()[1];
        let nocc = occ.len();
        let mut wcc = Array2::zeros((nk1, nocc));
        Zip::from(wcc.outer_iter_mut())
            .and(kvec.outer_iter())
            .par_for_each(|mut w, k| {
                w.assign(&self.berry_loop(&k.to_owned(), occ));
            });
        for mut row in wcc.outer_iter_mut() {
            row.as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let wcc = wcc.reversed_axes();
        wcc
    }

    fn wannier_centre(
        &self,
        occ: &Vec<usize>,
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Array2<f64> {
        if k_start.len() != self.dim_r() {
            panic!(
                "Wrong!, the k_start's length is {} but dim_r is {}, it's not equal!",
                k_start.len(),
                self.dim_r()
            );
        } else if dir_1.len() != self.dim_r() {
            panic!(
                "Wrong!, the dir_1's length is {} but dim_r is {}, it's not equal!",
                dir_1.len(),
                self.dim_r()
            );
        } else if dir_1.len() != self.dim_r() {
            panic!(
                "Wrong!, the dir_2's length is {} but dim_r is {}, it's not equal!",
                dir_2.len(),
                self.dim_r()
            );
        }
        let mut kvec = Array3::zeros((nk1, nk2, self.dim_r()));
        for i in 0..nk1 {
            for j in 0..nk2 {
                let mut s = kvec.slice_mut(s![i, j, ..]);
                let used_k = k_start
                    + dir_1 * (i as f64) / ((nk1 - 1) as f64)
                    + dir_2 * (j as f64) / ((nk2 - 1) as f64);
                s.assign(&used_k);
            }
        }
        let nocc = occ.len();
        let mut wcc = Array2::zeros((nk1, nocc));
        Zip::from(wcc.outer_iter_mut())
            .and(kvec.outer_iter())
            .par_for_each(|mut w, k| {
                w.assign(&self.berry_loop(&k.to_owned(), occ));
            });
        for mut row in wcc.outer_iter_mut() {
            row.as_slice_mut()
                .unwrap()
                .sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let wcc = wcc.reversed_axes();
        wcc
    }
}
