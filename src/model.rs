//! Core implementation of tight-binding model operations and Hamiltonian construction.
//!
//! This module provides the fundamental methods for working with tight-binding models,
//! including Hamiltonian construction, eigenvalue solving, and various physical
//! property calculations. The main `Model` struct implements methods for:
//! - Setting hopping parameters and on-site energies
//! - Solving the eigenvalue problem $H(\mathbf{k}) \psi_n = E_n \psi_n$
//! - Computing velocity operators $\mathbf{v} = \frac{1}{\hbar} \nabla_\mathbf{k} H(\mathbf{k})$
//! - Calculating Berry curvature and topological invariants
//! - Constructing surface Green's functions

// Re-export all model-related functionality from submodules
pub use crate::model_enums::{Dimension, Gauge, SpinDirection};
pub use crate::model_utils::{find_R, remove_col, remove_row};

// Import for Model struct definition
use crate::atom_struct::{Atom, OrbProj};
use ndarray::*;
use num_complex::Complex;
use serde::{Deserialize, Serialize};

/// Tight-binding model structure representing the Hamiltonian $H(\mathbf{k})$ and related properties.
///
/// The model is defined by its real-space hopping parameters $t_{ij}(\mathbf{R})$ where $\mathbf{R}$
/// is the lattice vector connecting unit cells. The Bloch Hamiltonian is given by:
/// $$
/// H(\mathbf{k}) = \sum_{\mathbf{R}} t(\mathbf{R}) e^{i\mathbf{k}\cdot\mathbf{R}}
/// $$
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Model {
    /// Real space dimension $d$ of the model (2D or 3D systems)
    pub dim_r: Dimension,
    /// Whether the model includes spin degrees of freedom
    pub spin: bool,
    /// Lattice vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$ as a $d \times d$ matrix
    /// where each row represents a lattice vector
    pub lat: Array2<f64>,
    /// Orbital positions in fractional coordinates within the unit cell
    pub orb: Array2<f64>,
    /// Orbital projection information (s, p, d orbitals etc.)
    pub orb_projection: Vec<OrbProj>,
    /// Atomic positions and information
    pub atoms: Vec<Atom>,
    /// Hamiltonian matrix elements $H_{mn}(\mathbf{R}) = \bra{m\mathbf{0}} H \ket{n\mathbf{R}}$
    /// stored as a 3D array: [orbital_m, orbital_n, R_index]
    pub ham: Array3<Complex<f64>>,
    /// Lattice vectors $\mathbf{R}$ corresponding to the hoppings in `ham`
    pub hamR: Array2<isize>,
    /// Position matrix elements $\mathbf{r}_{mn}(\mathbf{R}) = \bra{m\mathbf{0}} \mathbf{\hat{r}} \ket{n\mathbf{R}}$
    /// used for velocity operator calculations
    pub rmatrix: Array4<Complex<f64>>,
}

// Include Model implementation from submodules
pub use crate::model_build::*;
pub use crate::model_physics::*;
pub use crate::model_transform::*;
