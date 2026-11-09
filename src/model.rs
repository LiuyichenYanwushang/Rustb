//! Core implementation of tight-binding model operations and Hamiltonian construction.
//!
//! # Examples
//!
//! Build a model, inspect the arrays that define it, validate the invariants
//! every solver relies on, and round-trip it through the serde wire format.
//!
//! ```rust
//! use ndarray::{arr1, arr2, array};
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
//! let orb = arr2(&[[0.0, 0.0], [0.5, 0.5]]);
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! model.add_hop(-1.0, 0, 1, &array![0, 0], None);
//! model.set_onsite(&arr1(&[0.2, -0.2]), None);
//!
//! // Structural invariants checked by the solvers and response entry points.
//! model.validate()?;
//! assert_eq!(model.dim_r(), 2);
//! assert_eq!((model.norb(), model.nsta()), (2, 2));
//! assert_eq!(model.hamR.ncols(), 2);
//!
//! // Reciprocal vectors are rows: B = 2*pi*(A^T)^-1.
//! let reciprocal = model.rec_lat()?;
//! assert_eq!(reciprocal.dim(), (2, 2));
//!
//! // The serde wire format is validated on load: shape mismatches,
//! // non-finite data and unknown fields are rejected instead of trusted.
//! let encoded = toml::to_string(&model).unwrap();
//! let restored: Model<false, 2> = toml::from_str(&encoded).unwrap();
//! assert_eq!(restored.ham, model.ham);
//!
//! // Gauge selects the Bloch-basis phase convention, not the physics.
//! let k = arr1(&[0.25, 0.125]);
//! let h_atom = model.gen_ham(&k, Gauge::Atom);
//! let h_lattice = model.gen_ham(&k, Gauge::Lattice);
//! assert_eq!(h_atom.dim(), h_lattice.dim());
//! # Ok(())
//! # }
//! ```

pub mod access;
mod build;
mod edit;
pub mod physics;
pub mod rmatrix;
pub(crate) mod supercell;
pub mod types;
pub mod utils;
pub mod validate;

// Serde impls live in a private module; trait impls need no re-export.
mod wire;

mod operators;

pub use self::access::{AtomView, OrbitalRef};
pub use self::rmatrix::{HasRMatrix, NoRMatrix, RMatrixData};
pub use self::types::{Dimension, Gauge, SpinDirection};
pub use self::utils::{find_R, remove_col, remove_row};

// Preserve existing module imports while using shorter implementation names.
#[doc(hidden)]
pub use self::physics as model_physics;
#[doc(hidden)]
pub use self::utils as model_utils;

use crate::atom_struct::{Atom, OrbProj};
use crate::error::{Result, TbError};
use ndarray::*;
use ndarray_linalg::Inverse;
use num_complex::Complex;

/// Soft warning threshold for the distance (fractional coordinates, modulo a
/// lattice vector) between an orbital and its parent atom's position.
///
/// [`Model::validate`] no longer rejects a model that exceeds this distance;
/// it emits a one-time warning instead. Atom-centred orbital data, the owner
/// mapping, and all Hamiltonian arrays remain well-defined for displaced
/// Wannier centres. Symmetry and irrep algorithms that explicitly require
/// atom-centred orbitals validate that requirement themselves.
///
/// The diagnostic is component-wise on the CURRENT lattice basis and is
/// therefore not invariant under general `GL(d, Z)` basis changes.
pub const ORBITAL_ATOM_POSITION_TOLERANCE: f64 = 0.1;

/// Tight-binding model structure.
///
/// Const generic `SPIN`: spinless (false, default) / spinful (true).
/// Const generic `DIM`: spatial dimension 1/2/3 (default 3).
/// Type parameter `R`: [`HasRMatrix`] or [`NoRMatrix`] (default).
#[derive(Clone, Debug)]
pub struct Model<const SPIN: bool = false, const DIM: usize = 3, R: RMatrixData = NoRMatrix> {
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    pub orb_projection: Vec<OrbProj>,
    pub atoms: Vec<Atom>,
    pub ham: Array3<Complex<f64>>,
    pub hamR: Array2<isize>,
    pub rmatrix: R,
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    #[inline(always)]
    pub fn has_rmatrix(&self) -> bool {
        R::HAS_RMATRIX
    }

    pub fn dim_r(&self) -> usize {
        DIM
    }
    /// Reciprocal lattice vectors satisfying `B Aᵀ = 2π·I`, where `A` is the
    /// real-space lattice (rows = lattice vectors, stored in [`Model::lat`]).
    ///
    /// Each row of the returned matrix is a reciprocal lattice vector
    /// `bᵢ`.  The inversion can fail for a degenerate real-space lattice.
    pub fn rec_lat(&self) -> Result<Array2<f64>> {
        let inv_t = self
            .lat
            .t()
            .to_owned()
            .inv()
            .map_err(|e| TbError::Other(format!("Failed to invert lattice: {e}")))?;
        Ok(std::f64::consts::TAU * inv_t)
    }
    #[inline(always)]
    pub fn atom_list(&self) -> Vec<usize> {
        let mut atom_list = Vec::new();
        for a in self.atoms.iter() {
            atom_list.push(a.norb());
        }
        atom_list
    }
    #[inline(always)]
    pub fn natom(&self) -> usize {
        self.atoms.len()
    }
    #[inline(always)]
    pub fn norb(&self) -> usize {
        self.orb.nrows()
    }
    #[inline(always)]
    pub fn nsta(&self) -> usize {
        if SPIN { 2 * self.norb() } else { self.norb() }
    }
}

#[cfg(test)]
mod ownership_tests;
