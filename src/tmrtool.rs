//! TMR (Tunneling Magnetoresistance) tool for NEGF transport calculations.
//!
//! Given a sandwich [`Model`] and two fractional cut coordinates, this module
//! cuts the Hamiltonian into 5 blocks for NEGF transport:
//!
//! 1. Left lead (semi-infinite)
//! 2. Left contact (V_LC)
//! 3. Central region (device)
//! 4. Right contact (V_CR)
//! 5. Right lead (semi-infinite)
//!
//! The central entry point is [`TMRBlocks::from_sandwich`].
//!
//! # Examples
//!
//! ```ignore
//! use rustb::tmrtool::TMRBlocks;
//!
//! // Create blocks for a sandwich model along direction 2,
//! // with cuts at fractional coordinates 0.3 and 0.7
//! let blocks = TMRBlocks::from_sandwich(&model, 2, [0.3, 0.7], 0.01).unwrap();
//!
//! // Get the central-region Hamiltonian at a given k_parallel
//! let H_center = blocks.center_ham(&k_par);
//! ```

use crate::Model;
use crate::error::{Result, TbError};
use crate::model_utils::{remove_col, remove_row};
use crate::surfgreen::surf_Green;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use std::f64::consts::PI;

type C64 = Complex<f64>;

/// The 5 blocks for NEGF transport: left lead, coupling to centre, centre
/// Hamiltonian, coupling to right lead, and right lead.
///
/// Constructed via [`TMRBlocks::from_sandwich`].
///
/// # Fields
///
/// - `lead_L`, `lead_R`: surface Green's function objects for the semi-infinite leads.
/// - `V_LC`, `V_CR`: coupling matrices between leads and the central region.
/// - `ham_C`, `hamR_C`: real-space Hamiltonian blocks for the central region.
/// - `orb_C`: orbital positions (fractional) in the central region (reduced dimension).
/// - `dir`: the transport direction.
/// - `dim_r`: reduced spatial dimension (excluding `dir`).
#[derive(Clone, Debug)]
pub struct TMRBlocks {
    /// Surface Green's function for the left lead.
    pub lead_L: surf_Green,
    /// Coupling from left lead to central region.
    pub V_LC: Array2<C64>,
    /// Central-region Hamiltonian blocks (indexed by R).
    pub ham_C: Array3<C64>,
    /// Lattice vectors for central-region Hamiltonian blocks.
    pub hamR_C: Array2<isize>,
    /// Orbital positions in the central region (reduced dimension, excluding `dir`).
    pub orb_C: Array2<f64>,
    /// Coupling from central region to right lead.
    pub V_CR: Array2<C64>,
    /// Surface Green's function for the right lead.
    pub lead_R: surf_Green,
    /// Transport direction (the lattice-vector index along which the sandwich is stacked).
    pub dir: usize,
    /// Reduced spatial dimension (total dim_r minus one, for the transport direction).
    pub dim_r: usize,
}

impl TMRBlocks {
    /// Split a sandwich [`Model`] into left lead, central region, and right lead
    /// blocks for NEGF transport calculation.
    ///
    /// # Parameters
    ///
    /// - `sandwich`: the full sandwich model.
    /// - `dir`: the transport direction (0-based lattice-vector index).
    /// - `cuts`: two fractional coordinates along `dir` defining the boundaries
    ///   between left lead / centre / right lead.  Orbitals with `x < cuts[0]`
    ///   go to the left lead, `cuts[0] <= x <= cuts[1]` to the centre,
    ///   `x > cuts[1]` to the right lead.
    /// - `eta`: small imaginary broadening for the lead surface Green's functions.
    ///
    /// # Errors
    ///
    /// - [`TbError::InvalidDirection`] if `dir` is out of range.
    /// - [`TbError::InvalidSupercellSize`] if any of the three regions is empty.
    pub fn from_sandwich(
        sandwich: &Model,
        dir: usize,
        cuts: [f64; 2],
        eta: f64,
    ) -> Result<TMRBlocks> {
        if dir >= sandwich.dim_r() {
            return Err(TbError::InvalidDirection {
                index: dir,
                dim: sandwich.dim_r(),
            });
        }
        let norb = sandwich.norb();
        let nsta_total = sandwich.nsta();
        let dim_r = sandwich.dim_r();
        let mut orb_L = vec![];
        let mut orb_C = vec![];
        let mut orb_R = vec![];
        for i in 0..norb {
            let x = sandwich.orb[[i, dir]];
            if x < cuts[0] {
                orb_L.push(i);
            } else if x <= cuts[1] {
                orb_C.push(i);
            } else {
                orb_R.push(i);
            }
        }
        if orb_L.is_empty() || orb_C.is_empty() || orb_R.is_empty() {
            return Err(TbError::InvalidSupercellSize(0));
        }
        let (norb_L, norb_C, norb_R) = (orb_L.len(), orb_C.len(), orb_R.len());
        let sf = if sandwich.spin { 2 } else { 1 };
        let (nsta_L, nsta_C, nsta_R) = (sf * norb_L, sf * norb_C, sf * norb_R);
        let wan_L = wan_list(&orb_L, norb, sandwich.spin);
        let wan_C = wan_list(&orb_C, norb, sandwich.spin);
        let wan_R = wan_list(&orb_R, norb, sandwich.spin);

        let mut h00_L: Vec<(Array2<C64>, Array1<isize>)> = vec![];
        let mut h00_R: Vec<(Array2<C64>, Array1<isize>)> = vec![];
        let mut H_LC = Array2::<C64>::zeros((nsta_L, nsta_C));
        let mut H_CR = Array2::<C64>::zeros((nsta_C, nsta_R));
        let nR = sandwich.hamR.len_of(Axis(0));
        let mut ham_C = Array3::<C64>::zeros((nR, nsta_C, nsta_C));
        let mut hamR_C = Array2::<isize>::zeros((nR, dim_r));
        for (i, (ham, R)) in sandwich
            .ham
            .outer_iter()
            .zip(sandwich.hamR.outer_iter())
            .enumerate()
        {
            if R[dir] == 0 {
                let r = remove_dir(&R, dir);
                h00_L.push((select_sub(ham.view(), &wan_L, &wan_L), r.clone()));
                h00_R.push((select_sub(ham.view(), &wan_R, &wan_R), r.clone()));
                H_LC += &select_sub(ham.view(), &wan_L, &wan_C);
                H_CR += &select_sub(ham.view(), &wan_C, &wan_R);
            }
            ham_C
                .slice_mut(s![i, .., ..])
                .assign(&select_sub(ham.view(), &wan_C, &wan_C));
            let mut Rc = R.to_owned();
            Rc[dir] = 0;
            hamR_C.slice_mut(s![i, ..]).assign(&Rc);
        }

        let ham_01_L = if nsta_C >= nsta_L {
            H_LC.slice(s![.., ..nsta_L]).to_owned()
        } else {
            let s = sum_blocks(&h00_L);
            let nsp = count_per_layer(sandwich, dir);
            embed_h01(&s, nsta_L / nsp, nsp)
        };
        let ham_01_R = if nsta_C >= nsta_R {
            H_CR.slice(s![(nsta_C - nsta_R).., ..]).to_owned()
        } else {
            let s = sum_blocks(&h00_R);
            let nsp = count_per_layer(sandwich, dir);
            embed_h01(&s, nsta_R / nsp, nsp)
        };

        let lat_no_dir = remove_col(remove_row(sandwich.lat.clone(), dir), dir);
        let orb_L_no = remove_col(select_orb_rows(&sandwich.orb, &orb_L), dir);
        let orb_C_full = remove_col(select_orb_rows(&sandwich.orb, &orb_C), dir);
        let orb_R_no = remove_col(select_orb_rows(&sandwich.orb, &orb_R), dir);

        let sgf_L = build_lead_surf(
            &h00_L,
            &ham_01_L,
            &orb_L_no,
            &lat_no_dir,
            nsta_L,
            norb_L,
            sandwich.spin,
            eta,
        );
        let sgf_R = build_lead_surf(
            &h00_R,
            &ham_01_R,
            &orb_R_no,
            &lat_no_dir,
            nsta_R,
            norb_R,
            sandwich.spin,
            eta,
        );

        Ok(TMRBlocks {
            lead_L: sgf_L,
            V_LC: H_LC,
            ham_C,
            hamR_C,
            orb_C: orb_C_full,
            V_CR: H_CR,
            lead_R: sgf_R,
            dir,
            dim_r,
        })
    }

    /// Build the central-region Bloch Hamiltonian at a given parallel k-point.
    ///
    /// `k_par` is the k-vector in the reduced (surface-parallel) Brillouin zone
    /// (fractional coordinates, length = dim_r).
    ///
    /// # Returns
    ///
    /// The \\(H_C(k_\parallel)\\) matrix of size `(nsta_C, nsta_C)`.
    pub fn center_ham(&self, k_par: &Array1<f64>) -> Array2<C64> {
        let nsta_C = self.V_CR.nrows();
        let mut H = Array2::<C64>::zeros((nsta_C, nsta_C));
        for ir in 0..self.hamR_C.len_of(Axis(0)) {
            let R = self.hamR_C.slice(s![ir, ..]);
            let ham = self.ham_C.slice(s![ir, .., ..]);
            let phase = kphase(R, self.dir, k_par);
            let f = C64::new(phase.cos(), phase.sin());
            for a in 0..nsta_C {
                for b in 0..nsta_C {
                    H[[a, b]] += f * ham[[a, b]];
                }
            }
        }
        H
    }
}

/// Compute the phase factor \\(e^{i k_\parallel \cdot R}\\) for a given R-vector.
fn kphase(R: ArrayView1<isize>, dir: usize, k_par: &Array1<f64>) -> f64 {
    let mut p = 0.0;
    let mut ki = 0;
    for d in 0..R.len() {
        if d != dir {
            p += k_par[ki] * (R[d] as f64);
            ki += 1;
        }
    }
    2.0 * PI * p
}

/// Build the Wannier index list for a set of orbital indices, accounting for spin.
fn wan_list(orbs: &[usize], norb: usize, spin: bool) -> Vec<usize> {
    if spin {
        let mut w = vec![];
        for &i in orbs {
            w.push(i);
        }
        for &i in orbs {
            w.push(i + norb);
        }
        w
    } else {
        orbs.to_vec()
    }
}

/// Select a submatrix from a matrix view using the given row and column index lists.
fn select_sub(mat: ArrayView2<C64>, rows: &[usize], cols: &[usize]) -> Array2<C64> {
    let mut out = Array2::zeros((rows.len(), cols.len()));
    for (i, &ri) in rows.iter().enumerate() {
        for (j, &cj) in cols.iter().enumerate() {
            out[[i, j]] = mat[[ri, cj]];
        }
    }
    out
}

/// Count the number of states per layer along a given direction.
fn count_per_layer(model: &Model, dir: usize) -> usize {
    let norb = model.norb();
    let x0 = model.orb[[0, dir]];
    let mut c = 0;
    for i in 0..norb {
        if (model.orb[[i, dir]] - x0).abs() < 1e-10 {
            c += 1;
        } else {
            break;
        }
    }
    if model.spin { c * 2 } else { c }
}

/// Embed the inter-layer hopping into a larger matrix for leads with fewer layers
/// than the central region.
fn embed_h01(H: &Array2<C64>, n_layers: usize, nsp: usize) -> Array2<C64> {
    let nsta = n_layers * nsp;
    let mut h = Array2::<C64>::zeros((nsta, nsta));
    if n_layers < 2 {
        return h;
    }
    let from = (n_layers - 2) * nsp;
    let to = (n_layers - 1) * nsp;
    let sub = H.slice(s![to..to + nsp, from..from + nsp]);
    let last = (n_layers - 1) * nsp;
    for i in 0..nsp {
        for j in 0..nsp {
            h[[last + i, j]] = sub[[i, j]];
        }
    }
    h
}

/// Select rows from an orbital position matrix.
fn select_orb_rows(orb: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((indices.len(), orb.ncols()));
    for (i, &idx) in indices.iter().enumerate() {
        out.row_mut(i).assign(&orb.row(idx));
    }
    out
}

/// Remove the component at index `dir` from an R-vector.
fn remove_dir(R: &ArrayView1<isize>, dir: usize) -> Array1<isize> {
    let mut out = Array1::<isize>::zeros(R.len() - 1);
    let mut j = 0;
    for d in 0..R.len() {
        if d != dir {
            out[j] = R[d];
            j += 1;
        }
    }
    out
}

/// Sum a list of (matrix, R-vector) tuples into a single matrix.
fn sum_blocks(blocks: &[(Array2<C64>, Array1<isize>)]) -> Array2<C64> {
    if blocks.is_empty() {
        return Array2::zeros((0, 0));
    }
    let (n, m) = (blocks[0].0.nrows(), blocks[0].0.ncols());
    let mut s = Array2::<C64>::zeros((n, m));
    for (mat, _) in blocks {
        s += mat;
    }
    s
}

/// Build a `surf_Green` for a semi-infinite lead from its Hamiltonian blocks.
fn build_lead_surf(
    h00_blocks: &[(Array2<C64>, Array1<isize>)],
    h01: &Array2<C64>,
    orb: &Array2<f64>,
    lat: &Array2<f64>,
    nsta: usize,
    norb: usize,
    spin: bool,
    eta: f64,
) -> surf_Green {
    let nR0 = h00_blocks.len();
    let mut ham_bulk = Array3::<C64>::zeros((nR0, nsta, nsta));
    let mut ham_bulkR = Array2::<isize>::zeros((nR0, lat.nrows()));
    for (i, (mat, r)) in h00_blocks.iter().enumerate() {
        ham_bulk.slice_mut(s![i, .., ..]).assign(mat);
        ham_bulkR.slice_mut(s![i, ..]).assign(r);
    }
    let ham_hop = h01.clone().insert_axis(Axis(0));
    let ham_hopR = Array2::<isize>::zeros((1, lat.nrows()));
    surf_Green {
        dim_r: lat.nrows(),
        norb,
        nsta,
        natom: 0,
        spin,
        lat: lat.clone(),
        orb: orb.clone(),
        atom: Array2::<f64>::zeros((0, 0)),
        atom_list: vec![],
        eta,
        ham_bulk,
        ham_bulkR,
        ham_hop,
        ham_hopR,
    }
}
