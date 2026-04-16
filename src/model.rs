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
pub use crate::model_utils::{find_R, remove_col, remove_row};

// Import for Model struct definition
use crate::atom_struct::{Atom, AtomType, OrbProj};
use crate::error::{Result, TbError};
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
    /// If spin is true, then the orbital sequence is [orb_1_up, orb_2_up, ..., orb_1_dn, orb_2_dn, ...]
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

/// Gauge choice for Bloch wavefunctions
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Gauge {
    /// Lattice gauge: $\ket{\phi_{n\bm k}} = \sum_{\bm R} e^{i\bm k\cdot\bm R}\ket{n\bm R}$
    Lattice = 0,
    /// Atomic gauge: $\ket{u_{n\bm k}} = \sum_{\bm R} e^{i\bm k\cdot(\bm R+\bm\tau_n)}\ket{n\bm R}$
    Atom = 1,
}

/// Real-space dimensionality of the model
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Dimension {
    zero = 0,
    one = 1,
    two = 2,
    three = 3,
}

/// Spin direction for Pauli matrices
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum SpinDirection {
    /// Identity matrix $\sigma_0$
    None = 0,
    /// Pauli matrix $\sigma_x$
    x = 1,
    /// Pauli matrix $\sigma_y$
    y = 2,
    /// Pauli matrix $\sigma_z$
    z = 3,
}

// Include Model implementation from submodules
pub use crate::model_build::*;
pub use crate::model_physics::*;

impl Model {
    #[inline(always)]
    pub fn atom_position(&self) -> Array2<f64> {
        let mut atom_position = Array2::zeros((self.natom(), self.dim_r as usize));
        atom_position
            .outer_iter_mut()
            .zip(self.atoms.iter())
            .for_each(|(mut atom_p, atom)| {
                atom_p.assign(&atom.position());
            });
        atom_position
    }
    pub fn dim_r(&self) -> usize {
        self.lat.nrows()
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
    pub fn atom_type(&self) -> Vec<AtomType> {
        let mut atom_type = Vec::new();
        for a in self.atoms.iter() {
            atom_type.push(a.atom_type());
        }
        atom_type
    }
    #[inline(always)]
    pub fn nR(&self) -> usize {
        self.hamR.nrows()
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
        if self.spin {
            2 * self.norb()
        } else {
            self.norb()
        }
    }
    #[inline(always)]
    pub fn orb_angular(&self) -> Result<Array3<Complex<f64>>> {
        //!这个函数输出 $\bra{m,\bm k}L\ket{n,\bm k}$ 矩阵, 这里 $\ket{n,\bm k}$
        //!是根据轨道的projection 得到这个基函下的表示
        //!这个表示是依据每个原子来构造的, 所以是一个块对角的矩阵
        //!
        //!目前根据最新的轨道流公式, 这个已经废弃不使用了, 求轨道角动量见
        //!
        //!orbital_angular_momentom 函数
        let li = Complex::i() * 1.0;
        let mut i = 0;
        let mut L = Array3::<Complex<f64>>::zeros((self.dim_r(), self.norb(), self.norb()));
        let mut Lx = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        let mut Ly = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        let mut Lz = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        //开始构造在量子数为基的情况下的 Lz, L+ 和 L-.
        let mut Lz_orig = Array2::<Complex<f64>>::zeros((16, 16));
        Lz_orig
            .slice_mut(s![1..4, 1..4])
            .assign(&Array2::from_diag(&array![-1.0, 0.0, 1.0]).mapv(|x| Complex::new(x, 0.0)));
        Lz_orig.slice_mut(s![4..9, 4..9]).assign(
            &Array2::from_diag(&array![-2.0, -1.0, 0.0, 1.0, 2.0]).mapv(|x| Complex::new(x, 0.0)),
        );
        Lz_orig.slice_mut(s![9..16, 9..16]).assign(
            &Array2::from_diag(&array![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
                .mapv(|x| Complex::new(x, 0.0)),
        );
        let mut Lup_orig = Array2::<Complex<f64>>::zeros((16, 16));
        let mut Ldn_orig = Array2::<Complex<f64>>::zeros((16, 16));
        for l in 0..4 {
            for (m0, m) in (-(l as isize)..=l as isize).enumerate() {
                let i = l * l + m0;
                if m + 1 > l as isize && m - 1 < -(l as isize) {
                    continue;
                } else if m + 1 > l as isize {
                    let l = l as f64;
                    let m = m as f64;
                    Ldn_orig[[i - 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m - 1.0)).sqrt(), 0.0);
                } else if m - 1 < -(l as isize) {
                    let l = l as f64;
                    let m = m as f64;
                    Lup_orig[[i + 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m + 1.0)).sqrt(), 0.0);
                } else {
                    let l = l as f64;
                    let m = m as f64;
                    Ldn_orig[[i - 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m - 1.0)).sqrt(), 0.0);
                    Lup_orig[[i + 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m + 1.0)).sqrt(), 0.0);
                }
            }
        }
        //Lx=L+ + L-, Ly=-i( L+ - L-)
        let Lx_orig = &Lup_orig + &Ldn_orig;
        let Ly_orig = -li * (&Lup_orig - &Ldn_orig);
        //接下来我们要根据我们的轨道基函数写出我们的轨道角动量的矩阵
        //轨道角动量矩阵是按照原子进行分块对角化的
        //所以, 我们作循环的时候也要先按照原子进行分块对角化
        let mut a = 0;
        for atom0 in self.atoms.iter() {
            for i in a..a + atom0.norb() {
                let proj_i: Array1<Complex<f64>> = self.orb_projection[i]
                    .to_quantum_number()?
                    .mapv(|x: Complex<f64>| x.conj());
                for j in a..a + atom0.norb() {
                    let proj_j: Array1<Complex<f64>> =
                        self.orb_projection[j].to_quantum_number()?;
                    L[[0, i, j]] = proj_i.dot(&Lx_orig.dot(&proj_j));
                    L[[1, i, j]] = proj_i.dot(&Ly_orig.dot(&proj_j));
                    L[[2, i, j]] = proj_i.dot(&Lz_orig.dot(&proj_j));
                }
            }
            a += atom0.norb();
        }
        Ok(L)
    }
}
