//! Surface Green's function calculations for semi-infinite systems.
//!
//! This module implements algorithms for computing the surface Green's function
//! $G^s(\omega, \mathbf{k}_\parallel)$ of semi-infinite tight-binding systems.
//! The surface Green's function is defined as:
//! $$
//! G^s(\omega, \mathbf{k}_\parallel) = [\omega - H_{00}(\mathbf{k}_\parallel) - \Sigma(\omega, \mathbf{k}_\parallel)]^{-1}
//! $$
//! where $\Sigma$ is the self-energy due to the semi-infinite bulk.
use crate::Model;
use crate::error::{Result, TbError};
use crate::kpath::*;
use crate::kpoints::gen_kmesh;
use crate::model::Dimension;
pub use crate::model_utils::{remove_col, remove_row};
use gnuplot::Major;
use gnuplot::{Auto, AutoOption::Fix, AxesCommon, Custom, Figure, Font, HOT, RAINBOW};
use ndarray::concatenate;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::ops::AddAssign;
use std::ops::MulAssign;
use std::time::Instant;

/// Basic building block for surface Green's function calculations.
#[derive(Clone, Debug)]
pub struct surf_Green {
    /// - The real space dimension of the model.
    pub dim_r: usize,
    /// - The number of orbitals in the model.
    pub norb: usize,
    /// - The number of states in the model. If spin is enabled, nsta=norb$\times$2
    pub nsta: usize,
    /// - The number of atoms in the model. The atom and atom_list at the back are used to store the positions of the atoms, and the number of orbitals corresponding to each atom.
    pub natom: usize,
    /// - Whether this surface Green's function was built from a spinful model.
    ///   Derived from `Model<SPIN>` via [`from_Model`](surf_Green::from_Model).
    pub spin: bool,
    /// - The lattice vector of the model, a dim_r$\times$dim_r matrix, the axis0 direction stores a 1$\times$dim_r lattice vector.
    pub lat: Array2<f64>,
    /// - The position of the orbitals in the model. We use fractional coordinates uniformly.
    pub orb: Array2<f64>,
    /// - The position of the atoms in the model, also in fractional coordinates.
    pub atom: Array2<f64>,
    /// - The number of orbitals in the atoms, in the same order as the atom positions.
    pub atom_list: Vec<usize>,
    /// - The bulk Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub eta: f64,
    pub ham_bulk: Array3<Complex<f64>>,
    /// - The distance between the unit cell hoppings, i.e. R in $\bra{m0}\hat H\ket{nR}$.
    pub ham_bulkR: Array2<isize>,
    /// - The bulk Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub ham_hop: Array3<Complex<f64>>,
    pub ham_hopR: Array2<isize>,
}

impl Kpath for surf_Green {
    fn k_path(
        &self,
        path: &Array2<f64>,
        nk: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        //! Generate a k-path from high-symmetry points for band structure plotting.
        if self.dim_r == 0 {
            return Err(TbError::ZeroDimKPathError);
        }
        let n_node: usize = path.len_of(Axis(0));
        if self.dim_r != path.len_of(Axis(1)) {
            return Err(TbError::PathLengthMismatch {
                expected: self.dim_r,
                actual: path.len_of(Axis(1)),
            });
        }
        let k_metric = (self.lat.dot(&self.lat.t())).inv().unwrap();
        let mut k_node = Array1::<f64>::zeros(n_node);
        for n in 1..n_node {
            let dk = &path.row(n) - &path.row(n - 1);
            let a = k_metric.dot(&dk);
            let dklen = dk.dot(&a).sqrt();
            k_node[[n]] = k_node[[n - 1]] + dklen;
        }
        let mut node_index: Vec<usize> = vec![0];
        for n in 1..n_node - 1 {
            let frac = k_node[[n]] / k_node[[n_node - 1]];
            let a = (frac * ((nk - 1) as f64).round()) as usize;
            node_index.push(a)
        }
        node_index.push(nk - 1);
        let mut k_dist = Array1::<f64>::zeros(nk);
        let mut k_vec = Array2::<f64>::zeros((nk, self.dim_r));
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&path.row(0));
        for n in 1..n_node {
            let n_i = node_index[n - 1];
            let n_f = node_index[n];
            let kd_i = k_node[[n - 1]];
            let kd_f = k_node[[n]];
            let k_i = path.row(n - 1);
            let k_f = path.row(n);
            for j in n_i..n_f + 1 {
                let frac: f64 = ((j - n_i) as f64) / ((n_f - n_i) as f64);
                k_dist[[j]] = kd_i + frac * (kd_f - kd_i);
                k_vec
                    .row_mut(j)
                    .assign(&((1.0 - frac) * &k_i + frac * &k_f));
            }
        }
        Ok((k_vec, k_dist, k_node))
    }
}

impl surf_Green {
    /// Construct a `surf_Green` from a [`Model`].
    ///
    /// This method is generic over the const parameter `SPIN`, which
    /// determines whether spin is enabled. The `surf_Green::spin` field is
    /// set from this const generic (`spin = SPIN`), and the internal
    /// Hamiltonian construction (`orb_phase` doubling, etc.) uses it at
    /// runtime.
    ///
    /// `dir` specifies the surface normal direction.
    ///
    /// `eta` is the small imaginary part for the Green's function.
    ///
    /// For directions not aligned with a lattice vector, use [`Model::make_supercell`] first.
    pub fn from_Model<const SPIN: bool>(
        model: &Model<SPIN>,
        dir: usize,
        eta: f64,
        Np: Option<usize>,
    ) -> Result<surf_Green> {
        if dir >= model.dim_r as usize {
            return Err(TbError::InvalidDirection {
                index: dir,
                dim: model.dim_r as usize,
            });
        }
        let mut R_max: usize = 0;
        for R0 in model.hamR.rows() {
            if R_max < R0[[dir]].abs() as usize {
                R_max = R0[[dir]].abs() as usize;
            }
        }
        let R_max = match Np {
            Some(np) => {
                if R_max > np {
                    np
                } else {
                    R_max
                }
            }
            None => R_max,
        };

        let mut U = Array2::<f64>::eye(model.dim_r as usize);
        U[[dir, dir]] = R_max as f64;
        let model = model.make_supercell(&U)?;
        let mut ham0 = Array3::<Complex<f64>>::zeros((0, model.nsta(), model.nsta()));
        let mut hamR0 = Array2::<isize>::zeros((0, model.dim_r as usize));
        let mut hamR = Array3::<Complex<f64>>::zeros((0, model.nsta(), model.nsta()));
        let mut hamRR = Array2::<isize>::zeros((0, model.dim_r as usize));
        let use_hamR = model.hamR.rows();
        let use_ham = model.ham.outer_iter();
        // No clone needed: push/push_row copy from the view directly
        for (ham, R) in use_ham.zip(use_hamR) {
            if R[[dir]] == 0 {
                ham0.push(Axis(0), ham);
                hamR0.push_row(R);
            } else if R[[dir]] > 0 {
                hamR.push(Axis(0), ham);
                hamRR.push_row(R);
            }
        }
        let new_lat = remove_row(model.lat.clone(), dir);
        let new_lat = remove_col(new_lat.clone(), dir);
        let new_orb = remove_col(model.orb.clone(), dir);
        let new_atom = remove_col(model.atom_position(), dir);
        let new_hamR0 = remove_col(hamR0, dir);
        let new_hamRR = remove_col(hamRR, dir);
        let green: surf_Green = surf_Green {
            dim_r: model.dim_r as usize - 1,
            norb: model.norb(),
            nsta: model.nsta(),
            natom: model.natom(),
            spin: SPIN,
            lat: new_lat,
            orb: new_orb,
            atom: new_atom,
            atom_list: model.atom_list(),
            ham_bulk: ham0,
            ham_bulkR: new_hamR0,
            ham_hop: hamR,
            ham_hopR: new_hamRR,
            eta,
        };
        Ok(green)
    }

    #[inline(always)]
    pub fn gen_ham_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> (Array2<Complex<f64>>, Array2<Complex<f64>>) {
        if kvec.len() != self.dim_r {
            panic!("Wrong, the k-vector's length must equal to the dimension of model.")
        }
        // Orbital gauge phases: exp(i 2π τ·k)
        let orb_phase: Array1<Complex<f64>> = self
            .orb
            .dot(kvec)
            .mapv(|p| Complex::new(0.0, 2.0 * PI * p).exp());
        let orb_phase = if self.spin {
            concatenate![Axis(0), orb_phase, orb_phase]
        } else {
            orb_phase
        };
        // R-space phases for ham_bulk and ham_hop
        let U0 = (self.ham_bulkR.map(|x| *x as f64))
            .dot(kvec)
            .map(|x| Complex::<f64>::new(0.0, *x * 2.0 * PI).exp());
        let UR = (self.ham_hopR.map(|x| *x as f64))
            .dot(kvec)
            .map(|x| Complex::<f64>::new(0.0, *x * 2.0 * PI).exp());
        // Build H(k) with scaled_add: zero-allocation per R
        let mut ham0k = Array2::<Complex<f64>>::zeros((self.nsta, self.nsta));
        Zip::from(self.ham_bulk.outer_iter())
            .and(&U0)
            .for_each(|ham, &u| ham0k.scaled_add(u, &ham));
        let mut hamRk = Array2::<Complex<f64>>::zeros((self.nsta, self.nsta));
        Zip::from(self.ham_hop.outer_iter())
            .and(&UR)
            .for_each(|ham, &u| hamRk.scaled_add(u, &ham));
        // Gauge transform: H'[m,n] = conj(φ[m]) * H[m,n] * φ[n]
        // In-place O(nsta²) replaces two O(nsta³) matrix multiplies with diagonal U.
        for mut ham in [&mut ham0k, &mut hamRk] {
            for m in 0..self.nsta {
                let mut row = ham.slice_mut(s![m, ..]);
                let conj_pm = orb_phase[m].conj();
                Zip::from(&mut row)
                    .and(&orb_phase)
                    .for_each(|h, &pn| *h *= conj_pm * pn);
            }
        }
        (ham0k, hamRk)
    }
    #[inline(always)]
    pub fn surf_green_one<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        Energy: f64,
    ) -> (f64, f64, f64) {
        let (hamk, hamRk) = self.gen_ham_onek(kvec);
        let hamRk_conj: Array2<Complex<f64>> =
            conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>>(&hamRk);
        let I0 = Array2::<Complex<f64>>::eye(self.nsta);
        let accurate: f64 = 1e-8;
        let epsilon = Complex::new(Energy, self.eta) * &I0;
        let mut epi = hamk.clone();
        let mut eps = hamk.clone();
        let mut eps_t = hamk.clone();
        let mut ap = hamRk.clone();
        let mut bt = hamRk_conj.clone();

        for _ in 0..10 {
            let g0 = (&epsilon - &epi).inv().unwrap();
            let mat_1 = &ap.dot(&g0);
            let mat_2 = &bt.dot(&g0);
            let g0 = &mat_1.dot(&bt);
            epi = epi + g0;
            eps = eps + g0;
            let g0 = &mat_2.dot(&ap);
            epi = epi + g0;
            eps_t = eps_t + g0;
            ap = mat_1.dot(&ap);
            bt = mat_2.dot(&bt);
            if ap.iter().map(|x| x.norm()).sum::<f64>() < accurate {
                break;
            }
        }
        let g_LL = (&epsilon - eps).inv().unwrap();
        let g_RR = (&epsilon - eps_t).inv().unwrap();
        let g_B = (&epsilon - epi).inv().unwrap();
        let N_R: f64 = -1.0 / (PI) * g_RR.into_diag().sum().im;
        let N_L: f64 = -1.0 / (PI) * g_LL.into_diag().sum().im;
        let N_B: f64 = -1.0 / (PI) * g_B.into_diag().sum().im;
        (N_R, N_L, N_B)
    }

    #[inline(always)]
    pub fn surf_green_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        Energy: &Array1<f64>,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let (hamk, hamRk) = self.gen_ham_onek(kvec);
        let hamRk_conj: Array2<Complex<f64>> =
            conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>>(&hamRk);
        let I0 = Array2::<Complex<f64>>::eye(self.nsta);
        let accurate: f64 = 1e-6;
        let ((N_R, N_L), N_B): ((Vec<_>, Vec<_>), Vec<_>) = Energy
            .map(|e| {
                let epsilon = Complex::new(*e, self.eta) * &I0;
                let mut epi = hamk.clone();
                let mut eps = hamk.clone();
                let mut eps_t = hamk.clone();
                let mut ap = hamRk.clone();
                let mut bt = hamRk_conj.clone();
                for _ in 0..10 {
                    let g0 = (&epsilon - &epi).inv().unwrap();
                    let mat_1 = &ap.dot(&g0);
                    let mat_2 = &bt.dot(&g0);
                    let g0 = &mat_1.dot(&bt);
                    epi += g0;
                    eps += g0;
                    let g0 = &mat_2.dot(&ap);
                    epi += g0;
                    eps_t += g0;
                    ap = mat_1.dot(&ap);
                    bt = mat_2.dot(&bt);
                    if ap.iter().map(|x| x.norm()).sum::<f64>() < accurate {
                        break;
                    }
                }
                let g_LL = (&epsilon - eps).inv().unwrap();
                let g_RR = (&epsilon - eps_t).inv().unwrap();
                let g_B = (&epsilon - epi).inv().unwrap();
                let N_R: f64 = -1.0 / (PI) * g_RR.into_diag().sum().im;
                let N_L: f64 = -1.0 / (PI) * g_LL.into_diag().sum().im;
                let N_B: f64 = -1.0 / (PI) * g_B.into_diag().sum().im;
                ((N_R, N_L), N_B)
            })
            .into_iter()
            .unzip();
        let N_R = Array1::from_vec(N_R);
        let N_L = Array1::from_vec(N_L);
        let N_B = Array1::from_vec(N_B);
        (N_R, N_L, N_B)
    }

    pub fn surf_green_path(
        &self,
        kvec: &Array2<f64>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        spin: usize,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let Energy = Array1::<f64>::linspace(E_min, E_max, E_n);
        let nk = kvec.nrows();
        let mut N_R = Array2::<f64>::zeros((nk, E_n));
        let mut N_L = Array2::<f64>::zeros((nk, E_n));
        let mut N_B = Array2::<f64>::zeros((nk, E_n));
        Zip::from(N_R.outer_iter_mut())
            .and(N_L.outer_iter_mut())
            .and(N_B.outer_iter_mut())
            .and(kvec.outer_iter())
            .par_for_each(|mut nr, mut nl, mut nb, k| {
                let (NR, NL, NB) = self.surf_green_onek(&k, &Energy);
                nr.assign(&NR);
                nl.assign(&NL);
                nb.assign(&NB);
            });
        (N_L, N_R, N_B)
    }

    pub fn show_arc_state(&self, name: &str, kmesh: &Array1<usize>, energy: f64, spin: usize) {
        use std::fs::create_dir_all;
        use std::io::{BufWriter, Write};
        create_dir_all(name).expect("can't creat the file");
        assert_eq!(
            kmesh.len(),
            2,
            "show_arc_state can only calculated the three dimension system, so the kmesh need to be [m,n], but you give {}",
            kmesh
        );
        let kvec = gen_kmesh(kmesh).expect("Failed to generate k-mesh");
        let nk = kvec.nrows();
        let mut N_R = Array1::<f64>::zeros(nk);
        let mut N_L = Array1::<f64>::zeros(nk);
        let mut N_B = Array1::<f64>::zeros(nk);
        Zip::from(N_R.view_mut())
            .and(N_L.view_mut())
            .and(N_B.view_mut())
            .and(kvec.outer_iter())
            .par_for_each(|mut nr, mut nl, mut nb, k| {
                let (NR, NL, NB) = self.surf_green_one(&k, energy);
                *nr = NR;
                *nl = NL;
                *nb = NB;
            });
        let K = 2.0 * PI * self.lat.inv().unwrap().reversed_axes();
        let kvec_real = kvec.dot(&K);
        let mut file_name = String::new();
        file_name.push_str(&name);
        file_name.push_str("/arc.dat");
        let mut file = File::create(file_name).expect("Uable to create arc.dat");
        writeln!(file, r"# nk1, nk2, N_L, N_R, N_B");
        let mut writer = BufWriter::new(file);
        let mut s = String::new();
        for i in 0..nk {
            let aa = format!("{:.6}", kvec_real[[i, 0]]);
            s.push_str(&aa);
            let bb: String = format!("{:.6}", kvec_real[[i, 1]]);
            if kvec_real[[i, 1]] >= 0.0 {
                s.push_str("    ");
            } else {
                s.push_str("   ");
            }
            s.push_str(&bb);
            let cc: String = format!("{:.6}", N_L[[i]]);
            if N_L[[i]] >= 0.0 {
                s.push_str("    ");
            } else {
                s.push_str("   ");
            }
            s.push_str(&cc);
            let cc: String = format!("{:.6}", N_R[[i]]);
            if N_R[[i]] >= 0.0 {
                s.push_str("    ");
            } else {
                s.push_str("   ");
            }
            let cc: String = format!("{:.6}", N_B[[i]]);
            if N_B[[i]] >= 0.0 {
                s.push_str("    ");
            } else {
                s.push_str("   ");
            }
            s.push_str(&cc);
            s.push_str("\n");
        }
        writer.write_all(s.as_bytes()).unwrap();
        let _ = file;

        let width: usize = kmesh[[0]];
        let height: usize = kmesh[[1]];

        let N_L: Array2<f64> =
            Array2::from_shape_vec((height, width), N_L.to_vec()).expect("Shape error");
        let N_L = N_L.reversed_axes(); // 转置操作
        let N_L = N_L.iter().cloned().collect::<Vec<f64>>();
        let N_R: Array2<f64> =
            Array2::from_shape_vec((height, width), N_R.to_vec()).expect("Shape error");
        let N_R = N_R.reversed_axes(); // 转置操作
        let N_R = N_R.iter().cloned().collect::<Vec<f64>>();
        let N_B: Array2<f64> =
            Array2::from_shape_vec((height, width), N_B.to_vec()).expect("Shape error");
        let N_B = N_B.reversed_axes(); // 转置操作
        let N_B = N_B.iter().cloned().collect::<Vec<f64>>();

        //接下来我们绘制表面态
        let mut fg = Figure::new();
        let mut heatmap_data = N_L;
        let axes = fg.axes2d();
        //axes.set_palette(RAINBOW);
        axes.set_palette(Custom(&[
            (-1.0, 0.0, 0.0, 0.0),
            (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
            (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
            (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
            (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
        ]));
        axes.image(
            heatmap_data.iter(),
            width,
            height,
            Some((0.0, 0.0, 1.0, 1.0)),
            &[],
        );
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_aspect_ratio(Fix(1.0));
        axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        axes.set_cb_ticks_custom(
            [
                Major(-10.0, Fix("low")),
                Major(0.0, Fix("0")),
                Major(10.0, Fix("high")),
            ]
            .into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        let mut pdfname = String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_l.pdf");
        fg.set_terminal("pdfcairo", &pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _ = fg;

        let mut fg = Figure::new();
        let mut heatmap_data = N_R;
        let axes = fg.axes2d();
        //axes.set_palette(RAINBOW);
        axes.set_palette(Custom(&[
            (-1.0, 0.0, 0.0, 0.0),
            (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
            (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
            (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
            (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
        ]));
        axes.image(
            heatmap_data.iter(),
            width,
            height,
            Some((0.0, 0.0, 1.0, 1.0)),
            &[],
        );
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_aspect_ratio(Fix(1.0));
        axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        axes.set_cb_ticks_custom(
            [
                Major(-10.0, Fix("low")),
                Major(0.0, Fix("0")),
                Major(10.0, Fix("high")),
            ]
            .into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        let mut pdfname = String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_r.pdf");
        fg.set_terminal("pdfcairo", &pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _ = fg;

        let mut fg = Figure::new();
        let mut heatmap_data = N_B;
        let axes = fg.axes2d();
        //axes.set_palette(RAINBOW);
        axes.set_palette(Custom(&[
            (-1.0, 0.0, 0.0, 0.0),
            (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
            (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
            (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
            (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
        ]));
        axes.image(
            heatmap_data.iter(),
            width,
            height,
            Some((0.0, 0.0, 1.0, 1.0)),
            &[],
        );
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_aspect_ratio(Fix(1.0));
        axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        axes.set_cb_ticks_custom(
            [
                Major(-10.0, Fix("low")),
                Major(0.0, Fix("0")),
                Major(10.0, Fix("high")),
            ]
            .into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        let mut pdfname = String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_b.pdf");
        fg.set_terminal("pdfcairo", &pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _ = fg;
    }
    pub fn show_surf_state(
        &self,
        name: &str,
        kpath: &Array2<f64>,
        label: &Vec<&str>,
        nk: usize,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        spin: usize,
    ) {
        use std::fs::create_dir_all;
        use std::io::{BufWriter, Write};
        create_dir_all(name).expect("can't creat the file");
        let (kvec, kdist, knode) = self.k_path(kpath, nk).expect("Failed to generate k-path");
        let Energy = Array1::<f64>::linspace(E_min, E_max, E_n);
        let (N_L, N_R, N_B) = self.surf_green_path(&kvec, E_min, E_max, E_n, spin);
        //let N_L=N_L.mapv(|x| if x > 0.0 {x.ln()} else if  x< 0.0 {-x.abs().ln()} else {0.0});
        //let N_R=N_R.mapv(|x| if x > 0.0 {x.ln()} else if  x< 0.0 {-x.abs().ln()} else {0.0});
        //let N_B=N_B.mapv(|x| if x > 0.0 {x.ln()} else if  x< 0.0 {-x.abs().ln()} else {0.0});
        let ((N_L, N_R), N_B) = if spin == 0 {
            let N_L = N_L.mapv(|x| x.ln());
            let N_R = N_R.mapv(|x| x.ln());
            let N_B = N_B.mapv(|x| x.ln());
            let max = N_L.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let min = N_L.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
            let N_L = (N_L - min) / (max - min) * 20.0 - 10.0;
            let max = N_R.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let min = N_R.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
            let N_R = (N_R - min) / (max - min) * 20.0 - 10.0;
            let max = N_B.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
            let min = N_B.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
            let N_B = (N_B - min) / (max - min) * 20.0 - 10.0;
            ((N_L, N_R), N_B)
        } else {
            ((N_L, N_R), N_B)
        };

        //绘制 left_dos------------------------
        let mut left_name: String = String::new();
        left_name.push_str(&name.clone());
        left_name.push_str("/dos.surf_l");
        let mut file = File::create(left_name).expect("Unable to dos.surf_l.dat");
        let mut writer = BufWriter::new(file);
        let mut s = String::new();
        for i in 0..nk {
            for j in 0..E_n {
                let aa = format!("{:.6}", kdist[[i]]);
                s.push_str(&aa);
                let bb: String = format!("{:.6}", Energy[[j]]);
                if Energy[[j]] >= 0.0 {
                    s.push_str("    ");
                } else {
                    s.push_str("   ");
                }
                s.push_str(&bb);
                let cc: String = format!("{:.6}", N_L[[i, j]]);
                if N_L[[i, j]] >= 0.0 {
                    s.push_str("    ");
                } else {
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
        let _ = file;

        //绘制右表面态----------------------
        let mut left_name: String = String::new();
        left_name.push_str(&name.clone());
        left_name.push_str("/dos.surf_r");
        let mut file = File::create(left_name).expect("Unable to dos.surf_l.dat");
        let mut writer = BufWriter::new(file);
        let mut s = String::new();
        for i in 0..nk {
            for j in 0..E_n {
                let aa = format!("{:.6}", kdist[[i]]);
                s.push_str(&aa);
                let bb: String = format!("{:.6}", Energy[[j]]);
                if Energy[[j]] >= 0.0 {
                    s.push_str("    ");
                } else {
                    s.push_str("   ");
                }
                s.push_str(&bb);
                let cc: String = format!("{:.6}", N_R[[i, j]]);
                if N_L[[i, j]] >= 0.0 {
                    s.push_str("    ");
                } else {
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
        let _ = file;

        //绘制体态----------------------
        let mut left_name: String = String::new();
        left_name.push_str(&name.clone());
        left_name.push_str("/dos.surf_bulk");
        let mut file = File::create(left_name).expect("Unable to dos.surf_l.dat");
        let mut writer = BufWriter::new(file);
        let mut s = String::new();
        for i in 0..nk {
            for j in 0..E_n {
                let aa = format!("{:.6}", kdist[[i]]);
                s.push_str(&aa);
                let bb: String = format!("{:.6}", Energy[[j]]);
                if Energy[[j]] >= 0.0 {
                    s.push_str("    ");
                } else {
                    s.push_str("   ");
                }
                s.push_str(&bb);
                let cc: String = format!("{:.6}", N_B[[i, j]]);
                if N_L[[i, j]] >= 0.0 {
                    s.push_str("    ");
                } else {
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
        let _ = file;

        //接下来我们绘制表面态
        let mut fg = Figure::new();
        let width: usize = nk;
        let height: usize = E_n;
        let mut heatmap_data = vec![];
        for i in 0..height {
            for j in 0..width {
                heatmap_data.push(N_L[[j, i]]);
            }
        }
        let axes = fg.axes2d();
        //axes.set_palette(RAINBOW);
        axes.set_palette(Custom(&[
            (-1.0, 0.0, 0.0, 0.0),
            (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
            (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
            (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
            (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
        ]));
        axes.image(
            heatmap_data.iter(),
            width,
            height,
            Some((kdist[[0]], E_min, kdist[[nk - 1]], E_max)),
            &[],
        );
        let axes = axes.set_y_range(Fix(E_min), Fix(E_max));
        let axes = axes.set_x_range(Fix(kdist[[0]]), Fix(kdist[[nk - 1]]));
        let axes = axes.set_aspect_ratio(Fix(1.0));
        let mut show_ticks = Vec::new();
        for i in 0..knode.len() {
            let A = knode[[i]];
            let B = label[i];
            show_ticks.push(Major(A, Fix(B)));
        }
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        //axes.set_cb_ticks(Some((Fix(5.0),0)),&[],&[Font("Times New Roman",24.0)]);
        axes.set_cb_ticks_custom(
            [
                Major(-10.0, Fix("low")),
                Major(0.0, Fix("0")),
                Major(10.0, Fix("high")),
            ]
            .into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        let mut pdfname = String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_l.pdf");
        fg.set_terminal("pdfcairo", &pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _ = fg;

        //接下来我们绘制right表面态
        let mut fg = Figure::new();
        let width: usize = nk;
        let height: usize = E_n;
        let mut heatmap_data = vec![];
        for i in 0..height {
            for j in 0..width {
                heatmap_data.push(N_R[[j, i]]);
            }
        }
        let axes = fg.axes2d();
        //axes.set_palette(RAINBOW);
        axes.set_palette(Custom(&[
            (-1.0, 0.0, 0.0, 0.0),
            (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
            (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
            (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
            (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
        ]));
        axes.image(
            heatmap_data.iter(),
            width,
            height,
            Some((kdist[[0]], E_min, kdist[[nk - 1]], E_max)),
            &[],
        );
        let axes = axes.set_y_range(Fix(E_min), Fix(E_max));
        let axes = axes.set_x_range(Fix(kdist[[0]]), Fix(kdist[[nk - 1]]));
        let axes = axes.set_aspect_ratio(Fix(1.0));
        let mut show_ticks = Vec::new();
        for i in 0..knode.len() {
            let A = knode[[i]];
            let B = label[i];
            show_ticks.push(Major(A, Fix(B)));
        }
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        //axes.set_cb_ticks(Some((Fix(5.0),0)),&[],&[Font("Times New Roman",24.0)]);
        axes.set_cb_ticks_custom(
            [
                Major(-10.0, Fix("low")),
                Major(0.0, Fix("0")),
                Major(10.0, Fix("high")),
            ]
            .into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        let mut pdfname = String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_r.pdf");
        fg.set_terminal("pdfcairo", &pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _ = fg;
        //接下来我们绘制bulk表面态
        let mut fg = Figure::new();
        let width: usize = nk;
        let height: usize = E_n;
        let mut heatmap_data = vec![];
        for i in 0..height {
            for j in 0..width {
                heatmap_data.push(N_B[[j, i]]);
            }
        }
        let axes = fg.axes2d();
        //axes.set_palette(RAINBOW);
        axes.set_palette(Custom(&[
            (-1.0, 0.0, 0.0, 0.0),
            (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
            (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
            (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
            (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
        ]));
        axes.image(
            heatmap_data.iter(),
            width,
            height,
            Some((kdist[[0]], E_min, kdist[[nk - 1]], E_max)),
            &[],
        );
        let axes = axes.set_y_range(Fix(E_min), Fix(E_max));
        let axes = axes.set_x_range(Fix(kdist[[0]]), Fix(kdist[[nk - 1]]));
        let axes = axes.set_aspect_ratio(Fix(1.0));
        let mut show_ticks = Vec::new();
        for i in 0..knode.len() {
            let A = knode[[i]];
            let B = label[i];
            show_ticks.push(Major(A, Fix(B)));
        }
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
        //axes.set_cb_ticks(Some((Fix(5.0),0)),&[],&[Font("Times New Roman",24.0)]);
        axes.set_cb_ticks_custom(
            [
                Major(-10.0, Fix("low")),
                Major(0.0, Fix("0")),
                Major(10.0, Fix("high")),
            ]
            .into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
        let mut pdfname = String::new();
        pdfname.push_str(&name.clone());
        pdfname.push_str("/surf_state_b.pdf");
        fg.set_terminal("pdfcairo", &pdfname);
        fg.show().expect("Unable to draw heatmap");
        let _ = fg;
    }
}
