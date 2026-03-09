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
use crate::atom_struct::{Atom, AtomType, OrbProj};
use crate::error::{Result, TbError};
use crate::generics::hop_use;
use crate::kpoints::gen_kmesh;
use crate::math::comm;
use crate::solve_ham::*;
use ndarray::concatenate;
use ndarray::linalg::kron;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use ndarray_linalg::{Eigh, UPLO};
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;
use std::ops::AddAssign;

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

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Gauge {
    Lattice = 0,
    Atom = 1,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Dimension {
    zero = 0,
    one = 1,
    two = 2,
    three = 3,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum SpinDirection {
    None = 0,
    x = 1,
    y = 2,
    z = 3,
}

#[allow(non_snake_case)]
#[inline(always)]
pub fn find_R<A: Data<Elem = T>, B: Data<Elem = T>, T: std::cmp::PartialEq>(
    hamR: &ArrayBase<A, Ix2>,
    R: &ArrayBase<B, Ix1>,
) -> Option<usize> {
    //! Find the index of lattice vector $\mathbf{R}$ in the `hamR` array.
    //!
    //! This utility function searches for a specific lattice vector in the array
    //! of all hopping vectors and returns its index if found.
    //!
    //! # Arguments
    //! * `hamR` - Array of all lattice vectors $\mathbf{R}$ for hoppings
    //! * `R` - Target lattice vector to search for
    //!
    //! # Returns
    //! `Option<usize>` containing the index if found, `None` otherwise
    let n_R: usize = hamR.len_of(Axis(0));
    let dim_R: usize = hamR.len_of(Axis(1));
    for i in 0..(n_R) {
        let mut a = true;
        for j in 0..(dim_R) {
            a = a && (hamR[[i, j]] == R[[j]]);
        }
        if a {
            return Some(i);
        }
    }
    None
}

#[inline(always)]
fn remove_row<T: Copy>(array: Array2<T>, row_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.nrows()).filter(|&r| r != row_to_remove).collect();
    array.select(Axis(0), &indices)
}
#[inline(always)]
fn remove_col<T: Copy>(array: Array2<T>, col_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.ncols()).filter(|&r| r != col_to_remove).collect();
    array.select(Axis(1), &indices)
}

macro_rules! update_hamiltonian {
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                SpinDirection::None => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = $tmp;
                }
                SpinDirection::x => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] = $tmp;
                }
                SpinDirection::y => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] = -$tmp * Complex::<f64>::i();
                }
                SpinDirection::z => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = -$tmp;
                }
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] = $tmp;
        }
        $new_ham
    }};
}

macro_rules! add_hamiltonian {
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                SpinDirection::None => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] += $tmp;
                }
                SpinDirection::x => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] += $tmp;
                }
                SpinDirection::y => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] -= $tmp * Complex::<f64>::i();
                }
                SpinDirection::z => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] -= $tmp;
                }
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] += $tmp;
        }
        $new_ham
    }};
}

impl Model {
    /// Create a new tight-binding model with the given crystal structure.
    ///
    /// This constructor initializes a `Model` with the specified lattice and orbital
    /// positions. The Hamiltonian and position matrices are initially empty and can
    /// be populated using `set_hop`, `set_onsite`, and related methods.
    ///
    /// # Arguments
    /// * `dim_r` - Real space dimensionality (1, 2, or 3)
    /// * `lat` - Lattice vectors as a $d \times d$ matrix
    /// * `orb` - Orbital positions in fractional coordinates
    /// * `spin` - Whether to include spin degrees of freedom
    /// * `atom` - Optional list of atoms with orbital information
    ///
    /// # Returns
    /// `Result<Model>` containing the initialized tight-binding model
    ///
    /// # Examples
    /// ```
    /// use ndarray::*;
    /// use num_complex::Complex;
    /// use Rustb::*;
    ///
    /// // Create graphene model
    /// let lat = array![[1.0, 0.0], [-0.5, 3_f64.sqrt() / 2.0]];
    /// let orb = array![[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]];
    /// let spin = false;
    /// let mut graphene_model = Model::tb_model(2, lat, orb, spin, None).unwrap();
    /// ```
    pub fn tb_model(
        dim_r: usize,
        lat: Array2<f64>,
        orb: Array2<f64>,
        spin: bool,
        atom: Option<Vec<Atom>>,
    ) -> Result<Model> {
        //! This function is used to initialize a Model. The variables that need to be input are as follows:
        //!
        //! - dim_r: the dimension of the model
        //!
        //! - lat: the lattice constant
        //!
        //! - orb: the orbital coordinates
        //!
        //! - spin: whether to consider spin
        //!
        //! - atom: the atomic coordinates, which can be None
        //!
        //! - atom_list: the number of orbitals for each atom, which can be None.
        //!
        //! Note that if any of the atomic variables are None, it is better to make them all None.
        //!
        //!
        let norb: usize = orb.len_of(Axis(0));
        let mut nsta: usize = norb;
        if spin {
            nsta *= 2;
        }
        let mut new_atom_list: Vec<usize> = vec![1];
        let mut new_atom = Array2::<f64>::zeros((0, dim_r));
        if lat.len_of(Axis(1)) != dim_r {
            return Err(TbError::LatticeDimensionError {
                expected: dim_r,
                actual: lat.len_of(Axis(1)),
            });
        }
        if lat.len_of(Axis(0)) != lat.len_of(Axis(1)) {
            return Err(TbError::LatticeDimensionError {
                expected: lat.len_of(Axis(1)),
                actual: lat.len_of(Axis(0)),
            });
        }
        let new_atom = match atom {
            Some(atom0) => atom0,
            None => {
                // Determine if orbitals belong to the same atom by checking if they are too close;
                // this method only works when wannier90 does not perform maximal localization.
                let mut new_atom = Vec::new();
                new_atom.push(Atom::new(orb.row(0).to_owned(), 1, AtomType::H));
                for i in 1..norb {
                    if (orb.row(i).to_owned() - new_atom[new_atom.len() - 1].position()).norm_l2()
                        > 1e-2
                    {
                        let use_atom = Atom::new(orb.row(i).to_owned(), 1, AtomType::H);
                        new_atom.push(use_atom);
                    } else {
                        let n = new_atom.len();
                        new_atom[n - 1].push_orb();
                    }
                }
                new_atom
            }
        };
        let natom = new_atom.len();
        let ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
        let hamR = Array2::<isize>::zeros((1, dim_r));
        let mut rmatrix = Array4::<Complex<f64>>::zeros((1, dim_r, nsta, nsta));
        for i in 0..norb {
            for r in 0..dim_r {
                rmatrix[[0, r, i, i]] = Complex::<f64>::from(orb[[i, r]]);
                if spin {
                    rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                }
            }
        }
        let orb_projection = vec![OrbProj::s; norb];
        let mut model = Model {
            dim_r: Dimension::try_from(dim_r)?,
            spin,
            lat,
            orb,
            orb_projection,
            atoms: new_atom,
            ham,
            hamR,
            rmatrix,
        };
        Ok(model)
    }
    pub fn set_projection(&mut self, proj: &Vec<OrbProj>) {
        //! This function sets the tight-binding model's orbital projections.
        self.orb_projection = proj.clone();
    }
    #[allow(non_snake_case)]
    pub fn set_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<SpinDirection>,
    ) {
        //! This function is used to add hopping to the model. The "set" indicates that it can be used to override previous hopping.
        //!
        //! - tmp: the parameters for hopping
        //!
        //! - ind_i and ind_j: the orbital indices in the Hamiltonian, representing hopping from i to j
        //!
        //! - R: the position of the target unit cell for hopping
        //!
        //! - pauli: can take the values of 0, 1, 2, or 3, representing $\sigma_0$, $\sigma_x$, $\sigma_y$, $\sigma_z$.
        //!
        //! In general, this function is used to set $\bra{i\bm 0}\hat H\ket{j\bm R}=$tmp.
        //!
        //!
        //! #Examples
        //! ```
        //!use ndarray::*;
        //!use ndarray::prelude::*;
        //!use num_complex::Complex;
        //!use Rustb::*;
        //! //set the graphene model
        //!let lat=array![[1.0,0.0],[-1.0/2.0,3_f64.sqrt()/2.0]];
        //!let orb=array![[1.0/3.0,2.0/3.0],[2.0/3.0,1.0/3.0]];
        //!let spin=false;
        //!let mut graphene_model=Model::tb_model(2,lat,orb,spin,None).unwrap();
        //! let t=1.0; //the nearst hopping
        //! graphene_model.set_hop(t,0,1,&array![0,0],SpinDirection::None);
        //! // t is the hopping, 0, 1 is the orbital ,array![0,0] is the unit cell
        //! graphene_model.set_hop(t,0,1,&arr1(&[1,0]),SpinDirection::None);
        //! graphene_model.set_hop(t,0,1,&arr1(&vec![0,-1]),SpinDirection::None);
        //!
        //! ```

        let pauli: SpinDirection = pauli.into();
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli != SpinDirection::None && self.spin == false {
            eprintln!("Wrong, if spin is True and pauli is not zero, the pauli is not use")
        }
        assert!(
            R.len() == self.dim_r(),
            "Wrong, the R length should equal to dim_r"
        );
        assert!(
            ind_i < self.norb() && ind_j < self.norb(),
            "Wrong, ind_i and ind_j must be less than norb, here norb is {}, but ind_i={} and ind_j={}",
            self.norb(),
            ind_i,
            ind_j
        );

        let norb = self.norb();
        let negative_R = &(-R);
        match find_R(&self.hamR, &R) {
            Some(index) => {
                // Get the index of negative R (must exist, otherwise panic)
                let index_inv =
                    find_R(&self.hamR, &negative_R).expect("Negative R not found in hamR");

                if self.ham[[index, ind_i, ind_j]] != Complex::new(0.0, 0.0) {
                    eprintln!(
                        "Warning, the data of ham you input is {}, not zero, I hope you know what you are doing. If you want to eliminate this warning, use del_add to remove hopping.",
                        self.ham[[index, ind_i, ind_j]]
                    );
                }

                // Update matrix elements at R position
                update_hamiltonian!(
                    self.spin,
                    pauli,
                    tmp,
                    self.ham.slice_mut(s![index, .., ..]),
                    ind_i,
                    ind_j,
                    norb
                );

                // Update matrix elements at negative R position (unless onsite and R=0)
                if index != 0 || ind_i != ind_j {
                    update_hamiltonian!(
                        self.spin,
                        pauli,
                        tmp.conj(),
                        self.ham.slice_mut(s![index_inv, .., ..]),
                        ind_j,
                        ind_i,
                        norb
                    );
                }

                // Check if onsite matrix element is real
                assert!(
                    !(ind_i == ind_j && tmp.im != 0.0 && index == 0),
                    "Wrong, the onsite hopping must be real, but here is {}",
                    tmp
                )
            }
            None => {
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(self.spin, pauli, tmp, new_ham, ind_i, ind_j, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), R.view()).unwrap();
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(self.spin, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), negative_R.view()).unwrap();
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn add_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<SpinDirection>,
    ) {
        //! Parameters are the same as set_hop, but $\bra{i\bm 0}\hat H\ket{j\bm R}$ += tmp
        let pauli: SpinDirection = pauli.into();
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli != SpinDirection::None && self.spin == false {
            eprintln!("Wrong, if spin is True and pauli is not zero, the pauli is not use")
        }
        assert!(
            R.len() == self.dim_r(),
            "Wrong, the R length should equal to dim_r"
        );
        assert!(
            ind_i < self.norb() && ind_j < self.norb(),
            "Wrong, ind_i and ind_j must be less than norb, here norb is {}, but ind_i={} and ind_j={}",
            self.norb(),
            ind_i,
            ind_j
        );
        let norb = self.norb();
        let negative_R = &(-R);
        match find_R(&self.hamR, &R) {
            Some(index) => {
                // Get the index of negative R (must exist, otherwise panic)
                let index_inv =
                    find_R(&self.hamR, &negative_R).expect("Negative R not found in hamR");

                // Update matrix elements at R position
                add_hamiltonian!(
                    self.spin,
                    pauli,
                    tmp,
                    self.ham.slice_mut(s![index, .., ..]),
                    ind_i,
                    ind_j,
                    norb
                );

                // Update matrix elements at negative R position (unless onsite and R=0)
                if index != 0 || ind_i != ind_j {
                    add_hamiltonian!(
                        self.spin,
                        pauli,
                        tmp.conj(),
                        self.ham.slice_mut(s![index_inv, .., ..]),
                        ind_j,
                        ind_i,
                        norb
                    );
                }

                // Check if onsite matrix element is real
                assert!(
                    !(ind_i == ind_j && tmp.im != 0.0 && index == 0),
                    "Wrong, the onsite hopping must be real, but here is {}",
                    tmp
                )
            }
            None => {
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(self.spin, pauli, tmp, new_ham, ind_i, ind_j, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), R.view()).unwrap();
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(self.spin, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), negative_R.view()).unwrap();
            }
        }
    }

    #[allow(non_snake_case)]
    pub fn add_element(
        &mut self,
        tmp: Complex<f64>,
        ind_i: usize,
        ind_j: usize,
        R: &Array1<isize>,
    ) -> Result<()> {
        //! Parameters are the same as set_hop, but $\bra{i\bm 0}\hat H\ket{j\bm R}$ += tmp, ignoring spin, directly adding parameters
        if R.len() != self.dim_r() {
            return Err(TbError::RVectorLengthError {
                expected: self.dim_r(),
                actual: R.len(),
            });
        }
        if ind_i >= self.nsta() || ind_j >= self.nsta() {
            return Err(TbError::DimensionMismatch {
                context: "orbital indices".to_string(),
                expected: self.nsta(),
                found: std::cmp::max(ind_i, ind_j),
            });
        }
        if let Some(index) = find_R(&self.hamR, &R) {
            let index_inv = find_R(&self.hamR, &(-R)).expect("Negative R not found in hamR");
            self.ham[[index, ind_i, ind_j]] = tmp;
            if index != 0 && ind_i != ind_j {
                self.ham[[index_inv, ind_j, ind_i]] = tmp.conj();
            }
            if ind_i == ind_j && tmp.im != 0.0 && index == 0 {
                return Err(TbError::OnsiteHoppingMustBeReal(tmp));
            }
        } else {
            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            new_ham[[ind_i, ind_j]] = tmp;
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), R.view()).unwrap();

            let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
            new_ham[[ind_j, ind_i]] = tmp.conj();
            self.ham.push(Axis(0), new_ham.view()).unwrap();
            self.hamR.push(Axis(0), (-R).view()).unwrap();
        }
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn set_onsite(&mut self, tmp: &Array1<f64>, pauli: impl Into<SpinDirection>) {
        //! Directly set diagonal elements
        let pauli = pauli.into();
        if tmp.len() != self.norb() {
            panic!(
                "Wrong, the norb is {}, however, the onsite input's length is {}",
                self.norb(),
                tmp.len()
            )
        }
        for (i, item) in tmp.iter().enumerate() {
            self.set_onsite_one(*item, i, pauli);
        }
    }

    #[allow(non_snake_case)]
    pub fn add_onsite(&mut self, tmp: &Array1<f64>, pauli: impl Into<SpinDirection>) {
        //! Directly set diagonal elements
        let pauli = pauli.into();
        if tmp.len() != self.norb() {
            panic!(
                "Wrong, the norb is {}, however, the onsite input's length is {}",
                self.norb(),
                tmp.len()
            )
        }
        let R = Array1::zeros(self.dim_r());
        for (i, item) in tmp.iter().enumerate() {
            //self.set_onsite_one(*item,i,pauli)
            self.add_hop(Complex::new(*item, 0.0), i, i, &R, pauli)
        }
    }
    #[allow(non_snake_case)]
    pub fn set_onsite_one(&mut self, tmp: f64, ind: usize, pauli: impl Into<SpinDirection>) {
        //! Set $\bra{i\bm 0}\hat H\ket{i\bm 0}$
        let pauli = pauli.into();
        let R = Array1::<isize>::zeros(self.dim_r());
        self.set_hop(Complex::new(tmp, 0.0), ind, ind, &R, pauli)
    }
    pub fn del_hop(
        &mut self,
        ind_i: usize,
        ind_j: usize,
        R: &Array1<isize>,
        pauli: impl Into<SpinDirection>,
    ) {
        //! Delete $\bra{i\bm 0}\hat H\ket{j\bm R}$
        let pauli = pauli.into();
        if R.len() != self.dim_r() {
            panic!("Wrong, the R length should equal to dim_r")
        }
        if ind_i >= self.norb() || ind_j >= self.norb() {
            panic!(
                "Wrong, ind_i and ind_j must less than norb, here norb is {}, but ind_i={} and ind_j={}",
                self.norb(),
                ind_i,
                ind_j
            )
        }
        self.set_hop(Complex::new(0.0, 0.0), ind_i, ind_j, &R, pauli);
    }

    #[allow(non_snake_case)]
    pub fn k_path(
        &self,
        path: &Array2<f64>,
        nk: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        //! Generate high symmetry path from high symmetry points, plot band structure
        if self.dim_r() == 0 {
            return Err(TbError::ZeroDimKPathError);
        }
        let n_node: usize = path.len_of(Axis(0));
        if self.dim_r() != path.len_of(Axis(1)) {
            return Err(TbError::PathLengthMismatch {
                expected: self.dim_r(),
                actual: path.len_of(Axis(1)),
            });
        }
        let k_metric = (self.lat.dot(&self.lat.t())).inv().unwrap();
        let mut k_node = Array1::<f64>::zeros(n_node);
        for n in 1..n_node {
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk = path.row(n).to_owned() - path.slice(s![n - 1, ..]).to_owned();
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
        let mut k_vec = Array2::<f64>::zeros((nk, self.dim_r()));
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
                    .assign(&((1.0 - frac) * k_i.to_owned() + frac * k_f.to_owned()));
            }
        }
        Ok((k_vec, k_dist, k_node))
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    #[cfg_attr(doc, katexit::katexit)]
    ///Performs Fourier transform, converting real-space Hamiltonian to reciprocal-space Hamiltonian.
    ///
    ///There are two gauge choices: lattice gauge and atomic gauge, corresponding to `Gauge::Lattice` and `Gauge::Atom`.
    ///
    ///For the atomic gauge, the transformation between real-space wavefunction $\ket{n\bm R}$ and reciprocal-space wavefunction $\ket{u_{\bm k,n}}$ is:
    ///
    ///$$\ket{u_{n\bm k}(\bm r)}=\sum_{\bm R} e^{i\bm k\cdot(\bm R+\bm\tau_n)}\ket{n\bm R}$$
    ///
    ///satisfying $\ket{u_{i\bm k}(\bm r+\bm R)}=\ket{u_{i\bm k}(\bm r)}$.
    ///
    ///For the Hamiltonian, we have:
    ///$$
    ///H_{mn,\bm k}=\bra{u_{m\bm k}}\hat H\ket{u_{n\bm k}}=\sum_{\bm R^\prime}\sum_{\bm R} \bra{m\bm R^\prime}\hat H\ket{n\bm R}e^{-i(\bm R'-\bm R+\bm\tau_m-\bm \tau_n)\cdot\bm k}.
    ///$$
    ///Due to translational symmetry, only $\bm R'-\bm R$ matters, thus:
    ///$$
    ///H_{mn,\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R-\bm\tau_m+\bm \tau_n)\cdot\bm k}
    ///$$
    ///
    ///For the lattice gauge, we have $$\ket{\phi_{n\bm k}}=\sum_{\bm R} e^{i\bm k\cdot\bm R}\ket{n\bm R},$$ so:
    ///$$
    ///H_{mn,\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R)\cdot\bm k}
    ///$$
    ///
    ///Here $\ket{\psi_{n\bm k}}$ is periodic in reciprocal space: $\ket{\phi_{n\bm k}(\bm r)}=\ket{\phi_{n\bm k+\bm G}(\bm r)}$.
    pub fn gen_ham<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> Array2<Complex<f64>> {
        assert!(
            kvec.len() == self.dim_r(),
            "Wrong, the k-vector's length must equal to the dimension of model."
        );

        let Us = (self.hamR.mapv(|x| x as f64))
            .dot(kvec)
            .mapv(|x| Complex::<f64>::new(x, 0.0));
        let Us = Us * Complex::new(0.0, 2.0 * PI);
        let Us = Us.mapv(Complex::exp);
        let hamk: Array2<Complex<f64>> = self
            .ham
            .outer_iter()
            .zip(Us.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (hm, u)| {
                acc + &hm * *u
            });

        let hamk = match gauge {
            Gauge::Lattice => hamk,
            Gauge::Atom => {
                let U0: Array1<f64> = self.orb.dot(kvec);
                let U0: Array1<Complex<f64>> = U0.mapv(|x| Complex::<f64>::new(x, 0.0));
                let U0 = U0 * Complex::new(0.0, 2.0 * PI);
                let mut U0: Array1<Complex<f64>> = U0.mapv(Complex::exp);
                let U0 = if self.spin {
                    let U0 = concatenate![Axis(0), U0, U0];
                    U0
                } else {
                    U0
                };
                let U = Array2::from_diag(&U0);
                let U_dag = Array2::from_diag(&U0.map(|x| x.conj()));
                // Next two steps fill in the phase due to orbital coordinates
                let hamk: Array2<Complex<f64>> = U_dag.dot(&hamk);
                let re_ham = hamk.dot(&U);
                re_ham
            }
        };
        hamk
    }
    /// This function generates the velocity operator, i.e., $\bra{m\bm k}\p_\ap H_{\bm k}\ket{n\bm k},$
    /// The basis functions are Bloch wavefunctions.
    ///
    /// The velocity operator formula uses the tight-binding model,
    /// where the Fourier transform includes atomic positions.
    ///
    /// Thus we have
    ///
    /// $$
    /// \\begin\{aligned\}
    /// \\bra{m\bm k}\p_\ap H_{\bm k}\ket{n\bm k}&=\p_\ap\left(\bra{m\bm k} H\ket{n\bm k}\rt)-\p_\ap\left(\bra{m\bm k}\rt) H\ket{n\bm k}-\bra{m\bm k} H\p_\ap\ket{n\bm k}\\\\
    /// &=\sum_{\bm R} i(\bm R-\bm\tau_m+\bm\tau_n)H_{mn}(\bm R) e^{i\bm k\cdot(\bm R-\bm\tau_m+\bm\tau_n)}-\lt[H_{\bm k},\\mathcal A_{\bm k,\ap}\rt]_{mn}
    /// \\end\{aligned\}
    /// $$
    ///
    /// Here $\\mathcal A_{\bm k}$ is defined as $$\\mathcal A_{\bm k,\ap,mn}=-i\sum_{\bm R}r_{mn,\ap}(\bm R)e^{i\bm k\cdot(\bm R-\bm\tau_m+\bm\tau_{n})}+i\tau_{n\ap}\dt_{mn}$$
    /// where $\bm r_{mn}$ can be provided by wannier90 by setting write_rmn=true
    /// Here, all $\bm R$, $\bm r$, and $\bm \tau$ are in real-space coordinates.
    #[allow(non_snake_case)]
    #[inline(always)]
    pub fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        // We use lattice gauge rather than atomic gauge
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length {} must equal to the dimension of model {}.",
            kvec.len(),
            self.dim_r()
        );

        let Us = (self.hamR.mapv(|x| x as f64))
            .dot(kvec)
            .mapv(|x| Complex::<f64>::new(x, 0.0));
        let Us = Us * Complex::new(0.0, 2.0 * PI);
        let Us = Us.mapv(Complex::exp); // Us is exp(i k R)
        // Define an initialized velocity matrix
        let mut v = Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
        let R0 = &self.hamR.mapv(|x| Complex::<f64>::new(x as f64, 0.0));
        // R0 is the real-space hamR
        let R0 = R0.dot(&self.lat.mapv(|x| Complex::new(x, 0.0)));
        let hamk: Array2<Complex<f64>> = self
            .ham
            .outer_iter()
            .zip(Us.iter())
            .fold(Array2::zeros((self.nsta(), self.nsta())), |acc, (hm, u)| {
                acc + &hm * *u
            });
        let (v, hamk) = match gauge {
            Gauge::Atom => {
                let orb_sta = if self.spin {
                    let orb0 = concatenate(Axis(0), &[self.orb.view(), self.orb.view()]).unwrap();
                    orb0
                } else {
                    self.orb.to_owned()
                };
                let U0 = orb_sta.dot(kvec);
                let U0 = U0.mapv(|x| Complex::<f64>::new(x, 0.0));
                let U0 = U0 * Complex::new(0.0, 2.0 * PI);
                let mut U0 = U0.mapv(Complex::exp);
                // U0 is the phase factor
                let U = Array2::from_diag(&U0);
                let U_conj = Array2::from_diag(&U0.mapv(|x| x.conj()));
                let orb_real = orb_sta.dot(&self.lat);
                // Start constructing -orb_real[[i,r]]+orb_real[[j,r]];-----------------
                let mut UU = Array3::<f64>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                let A = orb_real.view().insert_axis(Axis(2));
                let A = A
                    .broadcast((self.nsta(), self.dim_r(), self.nsta()))
                    .unwrap()
                    .permuted_axes([1, 0, 2]);
                let mut B = A.view().permuted_axes([0, 2, 1]);
                let UU = &B - &A;
                let UU = UU.mapv(|x| Complex::<f64>::new(0.0, x)); //UU[i,j]=i(-tau[i]+tau[j])
                // Define an initialized velocity matrix
                Zip::from(v.outer_iter_mut())
                    .and(R0.axis_iter(Axis(1)))
                    .and(UU.outer_iter())
                    .for_each(|mut v0, r, det_tau| {
                        let vv: Array2<Complex<f64>> =
                            self.ham.outer_iter().zip(Us.iter().zip(r.iter())).fold(
                                Array2::zeros((self.nsta(), self.nsta())),
                                |acc, (ham, (us, r0))| acc + &ham * *us * *r0 * Complex::i(),
                            );
                        let vv: Array2<Complex<f64>> = &vv + &hamk * &det_tau;
                        let vv = &U_conj.dot(&vv);
                        let vv = vv.dot(&U); // Next two steps fill in the phase due to orbital coordinates
                        v0.assign(&vv);
                    });
                // At this point, we have computed sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)}
                // Next, compute Berry connection A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                let hamk = U_conj.dot(&hamk.dot(&U)); // Don't forget to add the phase to hamk
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let mut rk =
                        Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                    let mut rk = self
                        .rmatrix
                        .axis_iter(Axis(0))
                        .zip(Us.iter())
                        .fold(rk, |acc, (ham, us)| acc + &ham * *us);
                    for i in 0..3 {
                        let mut r0: ArrayViewMut2<Complex<f64>> = rk.slice_mut(s![i, .., ..]);
                        let r_new = r0.dot(&U);
                        let r_new = U_conj.dot(&r_new);
                        r0.assign(&r_new);
                        let mut dig = r0.diag_mut();
                        //dig.assign(&(&dig - &orb_real.column(i)));
                        dig.assign(&Array1::zeros(self.nsta()));
                        let A = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&A);
                    }
                }
                (v, hamk)
            }
            Gauge::Lattice => {
                // Use lattice gauge
                Zip::from(v.outer_iter_mut())
                    .and(R0.axis_iter(Axis(1)))
                    .for_each(|mut v0, r| {
                        let vv: Array2<Complex<f64>> =
                            self.ham.outer_iter().zip(Us.iter().zip(r.iter())).fold(
                                Array2::zeros((self.nsta(), self.nsta())),
                                |acc, (ham, (us, r0))| acc + &ham * *us * *r0 * Complex::i(),
                            );
                        v0.assign(&vv);
                    });
                // At this point, we have computed sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)}
                // Next, compute Berry connection A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let mut rk =
                        Array3::<Complex<f64>>::zeros((self.dim_r(), self.nsta(), self.nsta()));
                    let mut rk = self
                        .rmatrix
                        .axis_iter(Axis(0))
                        .zip(Us.iter())
                        .fold(rk, |acc, (ham, us)| acc + &ham * *us);
                    for i in 0..3 {
                        let mut r0: ArrayViewMut2<Complex<f64>> = rk.slice_mut(s![i, .., ..]);
                        //let mut dig = r0.diag_mut();
                        //dig.assign(&Array1::zeros(self.nsta()));
                        let A = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&A);
                    }
                }
                (v, hamk)
            }
        };
        (v, hamk)
    }


    /// This function truncates a certain direction of the model.
    ///
    /// num: number of unit cells to truncate
    ///
    /// dir: direction
    ///
    /// Returns a model where the dir direction matches the input model, but the number of orbitals and atoms is multiplied by num, with no inter-cell hopping along the dir direction.
    pub fn cut_piece(&self, num: usize, dir: usize) -> Result<Model> {
        //! This function is used to truncate a certain direction of a model.
        //!
        //! Parameters:
        //! - num: number of unit cells to truncate.
        //! - dir: the direction to be truncated.
        //!
        //! Returns a new model with the same direction as the input model, but with the number of orbitals and atoms increased by a factor of "num". There is no inter-cell hopping along the "dir" direction.
        if num < 1 {
            return Err(TbError::InvalidSupercellSize(num));
        }
        if dir >= self.dim_r() {
            return Err(TbError::InvalidDirection {
                index: dir,
                dim: self.dim_r(),
            });
        }
        let mut new_orb = Array2::<f64>::zeros((self.norb() * num, self.dim_r())); // Define a new orbital
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new(); // Define a new atom
        let new_norb = self.norb() * num;
        let new_nsta = self.nsta() * num;
        let new_natom = self.natom() * num;
        let mut new_lat = self.lat.clone();
        new_lat
            .row_mut(dir)
            .assign(&(self.lat.row(dir).to_owned() * (num as f64)));
        for i in 0..num {
            for n in 0..self.norb() {
                let mut use_orb = self.orb.row(n).to_owned();
                use_orb[[dir]] += i as f64;
                use_orb[[dir]] = use_orb[[dir]] / (num as f64);
                new_orb.row_mut(i * self.norb() + n).assign(&use_orb);
                new_orb_proj.push(self.orb_projection[n]);
            }
            for n in 0..self.natom() {
                let mut use_atom_position = self.atoms[n].position();
                use_atom_position[[dir]] += i as f64;
                use_atom_position[[dir]] *= 1.0 / (num as f64);
                let use_atom = Atom::new(
                    use_atom_position,
                    self.atoms[n].norb(),
                    self.atoms[n].atom_type(),
                );
                new_atom.push(use_atom);
            }
        }
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, new_nsta, new_nsta));
        let mut new_rmatrix = Array4::<Complex<f64>>::zeros((1, self.dim_r(), new_nsta, new_nsta));
        let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
        // New orbitals and atoms constructed, start building Hamiltonian
        // First attempt to construct position function
        let exist_r = self.rmatrix.len_of(Axis(0)) != 1;
        if exist_r == false {
            for n in 0..num {
                for i in 0..new_norb {
                    for r in 0..self.dim_r() {
                        new_rmatrix[[0, r, i, i]] = Complex::new(new_orb[[i, r]], 0.0);
                        if self.spin {
                            new_rmatrix[[0, r, i + new_norb, i + new_norb]] =
                                Complex::new(new_orb[[i, r]], 0.0);
                        }
                    }
                }
            }

            let mut using_ham = self.ham.clone();
            let mut using_hamR = self.hamR.clone();
            for n in 0..num {
                for (i0, (ind_R, ham)) in using_hamR
                    .outer_iter()
                    .zip(using_ham.outer_iter())
                    .enumerate()
                {
                    let ind: usize = (ind_R[[dir]] + (n as isize)) as usize;
                    let mut ind_R = ind_R.to_owned();
                    let ham = ham.to_owned();
                    ind_R[[dir]] = 0;
                    if ind < num {
                        // Start building Hamiltonian
                        let mut use_ham = Array2::<Complex<f64>>::zeros((new_nsta, new_nsta));
                        if self.spin {
                            //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);

                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 =
                                ham.slice(s![self.norb()..self.nsta(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                        } else {
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);
                        }
                        if let Some(index) = find_R(&new_hamR, &ind_R) {
                            new_ham.slice_mut(s![index, .., ..]).add_assign(&use_ham);
                        } else {
                            new_ham.push(Axis(0), use_ham.view());
                            new_hamR.push(Axis(0), ind_R.view());
                        }
                    } else {
                        continue;
                    }
                }
            }
        } else {
            let mut using_ham = self.ham.clone();
            let mut using_hamR = self.hamR.clone();
            let mut using_rmatrix = self.rmatrix.clone();
            for n in 0..num {
                for (i0, (ind_R, (ham, rmatrix))) in using_hamR
                    .outer_iter()
                    .zip(using_ham.outer_iter().zip(using_rmatrix.outer_iter()))
                    .enumerate()
                {
                    let ind: usize = (ind_R[[dir]] + (n as isize)) as usize;
                    let mut ind_R = ind_R.to_owned();
                    let ham = ham.to_owned();
                    let rmatrix = rmatrix.to_owned();
                    ind_R[[dir]] = 0;
                    if ind < num {
                        // Start building Hamiltonian
                        let mut use_ham = Array2::<Complex<f64>>::zeros((new_nsta, new_nsta));
                        if self.spin {
                            //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);

                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 =
                                ham.slice(s![self.norb()..self.nsta(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                        } else {
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);
                        }
                        //开始对 r_matrix 进行操作
                        let mut use_rmatrix =
                            Array3::<Complex<f64>>::zeros((self.dim_r(), new_nsta, new_nsta));
                        if exist_r {
                            if self.spin {
                                //如果体系包含自旋, 那需要将其重新排序, 自旋上和下分开
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    n * self.norb()..(n + 1) * self.norb(),
                                    ind * self.norb()..(ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., 0..self.norb(), 0..self.norb()]);
                                s.assign(&rmatrix0);

                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                    ind * self.norb()..(ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., self.norb()..self.nsta(), 0..self.norb()]);
                                s.assign(&rmatrix0);
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    n * self.norb()..(n + 1) * self.norb(),
                                    new_norb + ind * self.norb()
                                        ..new_norb + (ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., 0..self.norb(), self.norb()..self.nsta()]);
                                s.assign(&rmatrix0);
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                    new_norb + ind * self.norb()
                                        ..new_norb + (ind + 1) * self.norb()
                                ]);
                                let rmatrix0 = rmatrix.slice(s![
                                    ..,
                                    self.norb()..self.nsta(),
                                    self.norb()..self.nsta()
                                ]);
                                s.assign(&rmatrix0);
                            } else {
                                for i in 0..self.norb() {
                                    for j in 0..self.norb() {
                                        for r in 0..self.dim_r() {
                                            use_rmatrix
                                                [[r, i + n * self.norb(), j + ind * self.norb()]] =
                                                rmatrix[[r, i, j]];
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(index) = find_R(&new_hamR, &ind_R) {
                            new_ham.slice_mut(s![index, .., ..]).add_assign(&use_ham);
                            new_rmatrix
                                .slice_mut(s![index, .., .., ..])
                                .add_assign(&use_rmatrix);
                        } else {
                            new_ham.push(Axis(0), use_ham.view());
                            new_hamR.push(Axis(0), ind_R.view());
                            new_rmatrix.push(Axis(0), use_rmatrix.view());
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
        let mut model = Model {
            dim_r: Dimension::try_from(self.dim_r())?,
            spin: self.spin,
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_proj,
            atoms: new_atom,
            ham: new_ham,
            hamR: new_hamR,
            rmatrix: new_rmatrix,
        };
        Ok(model)
    }
    pub fn shift_to_atom(&mut self) {
        //!这个是将轨道移动到原子位置上
        let mut a = 0;
        for (i, atom) in self.atoms.iter().enumerate() {
            for j in 0..atom.norb() {
                self.orb.row_mut(a).assign(&atom.position());
                a += 1;
            }
        }
    }
    pub fn cut_dot(&self, num: usize, shape: usize, dir: Option<Vec<usize>>) -> Result<Model> {
        //! 这个是用来且角态或者切棱态的

        match self.dim_r() {
            3 => {
                let dir = if dir == None {
                    eprintln!(
                        "Wrong!, the dir is None, but model's dimension is 3, here we use defult 0,1 direction"
                    );
                    let dir = vec![0, 1];
                    dir
                } else {
                    dir.unwrap()
                };
                let (old_model, use_orb_item, use_atom_item) = {
                    let model_1 = self.cut_piece(num + 1, dir[0])?;
                    let model_2 = model_1.cut_piece(num + 1, dir[1])?;
                    let mut use_atom_item = Vec::<usize>::new();
                    let mut use_orb_item = Vec::<usize>::new(); //这个是确定要保留哪些轨道
                    let mut a: usize = 0;
                    match shape {
                        3 => {
                            let num0 = num as f64;
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[dir[0]]] + atom_position[[dir[1]]]
                                    > num0 / (num0 + 1.0) + 1e-5
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        4 => {
                            let num0 = num as f64;
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[dir[0]]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                    || atom_position[[dir[1]]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        6 => {
                            let num0 = num as f64;
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[dir[0]]] - atom_position[[dir[1]]]
                                    > 0.5 * num0 / (num0 + 1.0) + 1e-5)
                                    || (atom_position[[dir[0]]] - atom_position[[1]]
                                        < -0.5 * num0 / (num0 + 1.0) - 1e-5)
                                    || (atom_position[[dir[0]]] * (num0 + 1.0) / num0 > 1.0 + 1e-5)
                                    || (atom_position[[dir[1]]] * (num0 + 1.0) / num0 > 1.0 + 1e-5)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        8 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[dir[1]]] - atom_position[[dir[0]]] + 0.5 < 0.0)
                                    || (atom_position[[dir[0]]] - atom_position[[dir[1]]] + 0.5
                                        < 0.0)
                                    || (atom_position[[dir[1]]] + atom_position[[dir[0]]] < 0.5)
                                    || (atom_position[[dir[1]]] - atom_position[[dir[0]]] > 0.5)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        _ => {
                            return Err(TbError::InvalidShape {
                                shape,
                                supported: vec![3, 4, 6, 8],
                            });
                        }
                    };
                    (model_2, use_orb_item, use_atom_item)
                };
                let natom = use_atom_item.len();
                let norb = use_orb_item.len();
                let mut new_atom = Vec::new();
                let mut new_orb = Array2::<f64>::zeros((norb, self.dim_r()));
                let mut new_orb_proj = Vec::new();
                for (i, use_i) in use_atom_item.iter().enumerate() {
                    new_atom.push(old_model.atoms[*use_i].clone());
                }
                for (i, use_i) in use_orb_item.iter().enumerate() {
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                    new_orb_proj.push(old_model.orb_projection[*use_i])
                }
                let mut new_model = Model::tb_model(
                    self.dim_r(),
                    old_model.lat.clone(),
                    new_orb,
                    self.spin,
                    Some(new_atom),
                )?;
                new_model.orb_projection = new_orb_proj;
                let n_R = old_model.hamR.len_of(Axis(0));
                let mut new_ham =
                    Array3::<Complex<f64>>::zeros((n_R, new_model.nsta(), new_model.nsta()));
                let mut new_hamR = Array2::<isize>::zeros((0, self.dim_r()));
                let norb = new_model.norb();

                if self.spin {
                    let norb2 = old_model.norb();
                    for (r, R) in old_model.hamR.axis_iter(Axis(0)).enumerate() {
                        new_hamR.push_row(R);
                        for (i, use_i) in use_orb_item.iter().enumerate() {
                            for (j, use_j) in use_orb_item.iter().enumerate() {
                                new_ham[[r, i, j]] = old_model.ham[[r, *use_i, *use_j]];
                                new_ham[[r, i + norb, j + norb]] =
                                    old_model.ham[[r, *use_i + norb2, *use_j + norb2]];
                                new_ham[[r, i + norb, j]] =
                                    old_model.ham[[r, *use_i + norb2, *use_j]];
                                new_ham[[r, i, j + norb]] =
                                    old_model.ham[[r, *use_i, *use_j + norb2]];
                            }
                        }
                    }
                } else {
                    for (r, R) in old_model.hamR.axis_iter(Axis(0)).enumerate() {
                        new_hamR.push_row(R);
                        for (i, use_i) in use_orb_item.iter().enumerate() {
                            for (j, use_j) in use_orb_item.iter().enumerate() {
                                new_ham[[r, i, j]] = old_model.ham[[r, *use_i, *use_j]];
                            }
                        }
                    }
                }
                new_model.ham = new_ham;
                new_model.hamR = new_hamR;
                let nsta = new_model.nsta();
                if self.rmatrix.len_of(Axis(0)) == 1 {
                    for r in 0..self.dim_r() {
                        let mut use_rmatrix = Array2::<Complex<f64>>::zeros((nsta, nsta));
                        for i in 0..norb {
                            use_rmatrix[[i, i]] = Complex::new(new_model.orb[[i, r]], 0.0);
                        }
                        if new_model.spin {
                            for i in 0..norb {
                                use_rmatrix[[i + norb, i + norb]] =
                                    Complex::new(new_model.orb[[i, r]], 0.0);
                            }
                        }
                        new_model
                            .rmatrix
                            .slice_mut(s![0, r, .., ..])
                            .assign(&use_rmatrix);
                    }
                } else {
                    let mut new_rmatrix = Array4::<Complex<f64>>::zeros((
                        n_R,
                        self.dim_r(),
                        new_model.nsta(),
                        new_model.nsta(),
                    ));
                    if old_model.spin {
                        let norb2 = old_model.norb();
                        for r in 0..n_R {
                            for dim in 0..self.dim_r() {
                                for (i, use_i) in use_orb_item.iter().enumerate() {
                                    for (j, use_j) in use_orb_item.iter().enumerate() {
                                        new_rmatrix[[r, dim, i, j]] =
                                            old_model.rmatrix[[r, dim, *use_i, *use_j]];
                                        new_rmatrix[[r, dim, i + norb, j + norb]] = old_model
                                            .rmatrix[[r, dim, *use_i + norb2, *use_j + norb2]];
                                        new_rmatrix[[r, dim, i + norb, j]] =
                                            old_model.rmatrix[[r, dim, *use_i + norb2, *use_j]];
                                        new_rmatrix[[r, dim, i, j + norb]] =
                                            old_model.rmatrix[[r, dim, *use_i, *use_j + norb2]];
                                    }
                                }
                            }
                        }
                    } else {
                        for r in 0..n_R {
                            for dim in 0..self.dim_r() {
                                for (i, use_i) in use_orb_item.iter().enumerate() {
                                    for (j, use_j) in use_orb_item.iter().enumerate() {
                                        new_rmatrix[[r, dim, i, j]] =
                                            old_model.rmatrix[[r, dim, *use_i, *use_j]];
                                    }
                                }
                            }
                        }
                    }
                    new_model.rmatrix = new_rmatrix;
                }
                return Ok(new_model);
            }
            2 => {
                if dir != None {
                    eprintln!(
                        "Wrong!, the dimension of model is 2, but the dir is not None, you should give None!, here we use 0,1 direction"
                    );
                }

                let (old_model, use_orb_item, use_atom_item) = {
                    let model_1 = self.cut_piece(num + 1, 0)?;
                    let model_2 = model_1.cut_piece(num + 1, 1)?;
                    let mut use_atom_item = Vec::<usize>::new();
                    let mut use_orb_item = Vec::<usize>::new(); //这个是确定要保留哪些轨道
                    let mut a: usize = 0;
                    match shape {
                        3 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[0]] + atom_position[[1]]
                                    > (num as f64) / (num as f64 + 1.0)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        4 => {
                            let num0 = num as f64;
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if atom_position[[0]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                    || atom_position[[1]] * (num0 + 1.0) / num0 > 1.0 + 1e-5
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        6 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[1]] - atom_position[[0]] + 0.5 < 0.0)
                                    || (atom_position[[0]] - atom_position[[1]] + 0.5 < 0.0)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        8 => {
                            for i in 0..model_2.natom() {
                                let atom_position = model_2.atoms[i].position();
                                if (atom_position[[1]] - atom_position[[0]] + 0.5 < 0.0)
                                    || (atom_position[[0]] - atom_position[[1]] + 0.5 < 0.0)
                                    || (atom_position[[1]] + atom_position[[0]] < 0.5)
                                    || (atom_position[[1]] - atom_position[[0]] > 0.5)
                                {
                                    a += model_2.atoms[i].norb();
                                } else {
                                    use_atom_item.push(i);
                                    for i in 0..model_2.atoms[i].norb() {
                                        use_orb_item.push(a);
                                        a += 1;
                                    }
                                }
                            }
                        }
                        _ => {
                            return Err(TbError::InvalidShape {
                                shape,
                                supported: vec![3, 4, 6, 8],
                            });
                        }
                    };
                    (model_2, use_orb_item, use_atom_item)
                };
                let natom = use_atom_item.len();
                let norb = use_orb_item.len();
                let mut new_atom = Vec::new();
                let mut new_orb = Array2::<f64>::zeros((norb, self.dim_r()));
                let mut new_orb_proj = Vec::new();
                for (i, use_i) in use_atom_item.iter().enumerate() {
                    new_atom.push(old_model.atoms[*use_i].clone());
                }
                for (i, use_i) in use_orb_item.iter().enumerate() {
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                    new_orb_proj.push(old_model.orb_projection[*use_i])
                }
                let mut new_model = Model::tb_model(
                    self.dim_r(),
                    old_model.lat.clone(),
                    new_orb,
                    self.spin,
                    Some(new_atom),
                )?;
                new_model.orb_projection = new_orb_proj;
                let n_R = new_model.hamR.len_of(Axis(0));
                let mut new_ham =
                    Array3::<Complex<f64>>::zeros((n_R, new_model.nsta(), new_model.nsta()));
                let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
                let norb = new_model.norb();
                let nsta = new_model.nsta();

                if self.spin {
                    let norb2 = old_model.norb();
                    for (i, use_i) in use_orb_item.iter().enumerate() {
                        for (j, use_j) in use_orb_item.iter().enumerate() {
                            new_ham[[0, i, j]] = old_model.ham[[0, *use_i, *use_j]];
                            new_ham[[0, i + norb, j + norb]] =
                                old_model.ham[[0, *use_i + norb2, *use_j + norb2]];
                            new_ham[[0, i + norb, j]] = old_model.ham[[0, *use_i + norb2, *use_j]];
                            new_ham[[0, i, j + norb]] = old_model.ham[[0, *use_i, *use_j + norb2]];
                        }
                    }
                } else {
                    for (i, use_i) in use_orb_item.iter().enumerate() {
                        for (j, use_j) in use_orb_item.iter().enumerate() {
                            new_ham[[0, i, j]] = old_model.ham[[0, *use_i, *use_j]];
                        }
                    }
                }
                new_model.ham = new_ham;
                new_model.hamR = new_hamR;
                if self.rmatrix.len_of(Axis(0)) == 1 {
                    for r in 0..self.dim_r() {
                        let mut use_rmatrix = Array2::<Complex<f64>>::zeros((nsta, nsta));
                        for i in 0..norb {
                            use_rmatrix[[i, i]] = Complex::new(new_model.orb[[i, r]], 0.0);
                        }
                        if new_model.spin {
                            for i in 0..norb {
                                use_rmatrix[[i + norb, i + norb]] =
                                    Complex::new(new_model.orb[[i, r]], 0.0);
                            }
                        }
                        new_model
                            .rmatrix
                            .slice_mut(s![0, r, .., ..])
                            .assign(&use_rmatrix);
                    }
                } else {
                    let mut new_rmatrix = Array4::<Complex<f64>>::zeros((
                        n_R,
                        self.dim_r(),
                        new_model.nsta(),
                        new_model.nsta(),
                    ));
                    if old_model.spin {
                        let norb2 = old_model.norb();
                        for dim in 0..self.dim_r() {
                            for (i, use_i) in use_orb_item.iter().enumerate() {
                                for (j, use_j) in use_orb_item.iter().enumerate() {
                                    new_rmatrix[[0, dim, i, j]] =
                                        old_model.rmatrix[[0, dim, *use_i, *use_j]];
                                    new_rmatrix[[0, dim, i + norb, j + norb]] =
                                        old_model.rmatrix[[0, dim, *use_i + norb2, *use_j + norb2]];
                                    new_rmatrix[[0, dim, i + norb, j]] =
                                        old_model.rmatrix[[0, dim, *use_i + norb2, *use_j]];
                                    new_rmatrix[[0, dim, i, j + norb]] =
                                        old_model.rmatrix[[0, dim, *use_i, *use_j + norb2]];
                                }
                            }
                        }
                    } else {
                        for dim in 0..self.dim_r() {
                            for (i, use_i) in use_orb_item.iter().enumerate() {
                                for (j, use_j) in use_orb_item.iter().enumerate() {
                                    new_rmatrix[[0, dim, i, j]] =
                                        old_model.rmatrix[[0, dim, *use_i, *use_j]];
                                }
                            }
                        }
                    }
                    new_model.rmatrix = new_rmatrix;
                }
                return Ok(new_model);
            }
            _ => {
                return Err(TbError::InvalidDimension {
                    dim: self.dim_r(),
                    supported: vec![2, 3],
                });
            }
        }
    }

    pub fn move_to_atom(&mut self) {
        ///This function moves the orbital position to the atomic position
        let mut a = 0;
        for i in 0..self.natom() {
            for j in 0..self.atoms[i].norb() {
                self.orb.row_mut(a).assign(&self.atoms[i].position());
                a += 1;
            }
        }
    }
    pub fn remove_orb(&mut self, orb_list: &Vec<usize>) {
        let mut use_orb_list = orb_list.clone();
        use_orb_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates = { use_orb_list.windows(2).any(|window| window[0] == window[1]) };
        if has_duplicates {
            panic!("Wrong, make sure no duplicates in orb_list");
        }
        let mut index: Vec<_> = (0..=self.norb() - 1)
            .filter(|&num| !use_orb_list.contains(&num))
            .collect(); //要保留下来的元素
        let delete_n = orb_list.len();
        self.orb = self.orb.select(Axis(0), &index);
        let mut new_orb_proj = Vec::new();
        for i in index.iter() {
            new_orb_proj.push(self.orb_projection[*i])
        }
        self.orb_projection = new_orb_proj;

        let mut b = 0;
        for (i, a) in self.atoms.clone().iter().enumerate() {
            b += a.norb();
            while b > use_orb_list[0] {
                self.atoms[i].remove_orb();
                let _ = use_orb_list.remove(0);
            }
        }
        self.atoms.retain(|x| x.norb() != 0);
        //开始计算nsta
        if self.spin {
            let index_add: Vec<_> = index.iter().map(|x| *x + self.norb()).collect();
            index.extend(index_add);
        }
        let mut b = 0;
        //开始操作哈密顿量
        let new_ham = self.ham.select(Axis(1), &index);
        let new_ham = new_ham.select(Axis(2), &index);
        self.ham = new_ham;
        //开始操作rmatrix
        let new_rmatrix = self.rmatrix.select(Axis(2), &index);
        let new_rmatrix = new_rmatrix.select(Axis(3), &index);
        self.rmatrix = new_rmatrix;
    }

    pub fn remove_atom(&mut self, atom_list: &Vec<usize>) {
        //----------判断是否存在重复, 并给出保留的index
        let mut use_atom_list = atom_list.clone();
        use_atom_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates = {
            use_atom_list
                .windows(2)
                .any(|window| window[0] == window[1])
        };
        if has_duplicates {
            panic!("Wrong, make sure no duplicates in orb_list");
        }

        let mut atom_index: Vec<_> = (0..=self.natom() - 1)
            .filter(|&num| !use_atom_list.contains(&num))
            .collect(); //要保留下来的元素

        let new_atoms = {
            let mut new_atoms = Vec::new();
            for i in atom_index.iter() {
                new_atoms.push(self.atoms[*i].clone());
            }
            new_atoms
        }; //选出需要的原子以及需要的轨道
        //接下来选择需要的轨道

        let mut b = 0;
        let mut orb_index = Vec::new(); //要保留下来的轨道
        let atom_list = self.atom_list();
        let mut int_atom_list = Array1::zeros(self.natom());
        int_atom_list[[0]] = 0;
        for i in 1..self.natom() {
            int_atom_list[[i]] = int_atom_list[[i - 1]] + atom_list[i - 1];
        }
        for i in atom_index.iter() {
            for j in 0..self.atoms[*i].norb() {
                orb_index.push(int_atom_list[[*i]] + j);
            }
        }
        let norb = self.norb(); //保留之前的norb
        self.orb = self.orb.select(Axis(0), &orb_index);
        self.atoms = new_atoms;

        let mut new_orb_proj = Vec::new();
        for i in orb_index.iter() {
            new_orb_proj.push(self.orb_projection[*i])
        }
        self.orb_projection = new_orb_proj;
        if self.spin {
            let index_add: Vec<_> = orb_index.iter().map(|x| *x + norb).collect();
            orb_index.extend(index_add);
        }
        //开始操作哈密顿量
        let new_ham = self.ham.select(Axis(1), &orb_index);
        let new_ham = new_ham.select(Axis(2), &orb_index);
        self.ham = new_ham;
        //开始操作rmatrix
        let new_rmatrix = self.rmatrix.select(Axis(2), &orb_index);
        let new_rmatrix = new_rmatrix.select(Axis(3), &orb_index);
        self.rmatrix = new_rmatrix;
    }

    pub fn reorder_atom(&mut self, order: &Vec<usize>) {
        ///这个函数是用来调整模型的原子顺序的, 主要用来检查某些模型
        if order.len() != self.natom() {
            panic!(
                "Wrong! when you using reorder_atom, the order's length {} must equal to the num of atoms {}.",
                order.len(),
                self.natom()
            );
        };
        //首先我们根据原子顺序得到轨道顺序
        let mut new_orb_order = Vec::new();
        //第n个原子的最开始的轨道数
        let mut orb_atom_map = Vec::new();
        let mut a = 0;
        for atom in self.atoms.iter() {
            orb_atom_map.push(a);
            a += atom.norb();
        }
        for i in order.iter() {
            let mut s = String::new();
            for j in 0..self.atoms[*i].norb() {
                new_orb_order.push(orb_atom_map[*i] + j);
            }
        }
        //重排轨道顺序
        self.orb = self.orb.select(Axis(0), &new_orb_order);
        let mut new_atom = Vec::new();
        //重排轨道projection顺序
        let mut new_orb_proj = Vec::new();
        for i in new_orb_order.iter() {
            new_orb_proj.push(self.orb_projection[*i]);
        }
        self.orb_projection = new_orb_proj;
        //重排原子顺序
        for i in 0..self.natom() {
            new_atom.push(self.atoms[order[i]].clone());
        }
        self.atoms = new_atom;
        //开始重排哈密顿量
        let new_state_order = if self.spin {
            //如果有自旋
            let mut new_state_order = new_orb_order.clone();
            for i in new_orb_order.iter() {
                new_state_order.push(*i + self.norb());
            }
            new_state_order
        } else {
            new_orb_order
        };
        self.ham = self.ham.select(Axis(1), &new_state_order);
        self.ham = self.ham.select(Axis(2), &new_state_order);
        self.rmatrix = self.rmatrix.select(Axis(2), &new_state_order);
        self.rmatrix = self.rmatrix.select(Axis(3), &new_state_order);
    }


    pub fn make_supercell(&self, U: &Array2<f64>) -> Result<Model> {
        //这个函数是用来对模型做变换的, 变换前后模型的基矢 $L'=UL$.
        //!This function is used to transform the model, where the new basis after transformation is given by $L' = UL$.
        if self.dim_r() != U.len_of(Axis(0)) {
            return Err(TbError::TransformationMatrixDimMismatch {
                expected: self.dim_r(),
                actual: U.len_of(Axis(0)),
            });
        }
        //新的lattice
        let new_lat = U.dot(&self.lat);
        //体积的扩大倍数
        let U_det = U.det().unwrap() as isize;
        if U_det < 0 {
            return Err(TbError::InvalidSupercellDet { det: U_det as f64 });
        } else if U_det == 0 {
            return Err(TbError::InvalidSupercellDet { det: 0.0 });
        }
        let U_inv = U.inv().unwrap();
        //开始判断是否存在小数
        for i in 0..U.len_of(Axis(0)) {
            for j in 0..U.len_of(Axis(1)) {
                if U[[i, j]].fract() > 1e-8 {
                    return Err(TbError::InvalidSupercellMatrix);
                }
            }
        }

        //开始构建新的轨道位置和原子位置
        //新的轨道
        let mut use_orb = self.orb.dot(&U_inv);
        //新的原子位置
        let use_atom_position = self.atom_position().dot(&U_inv);
        //新的atom_list
        let mut use_atom_list: Vec<usize> = Vec::new();
        let mut orb_list: Vec<usize> = Vec::new();
        let mut new_orb = Array2::<f64>::zeros((0, self.dim_r()));
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new();
        let mut a = 0;
        for i in 0..self.natom() {
            use_atom_list.push(a);
            a += self.atoms[i].norb();
        }

        match self.dim_r() {
            3 => {
                for i in -U_det - 1..U_det + 1 {
                    for j in -U_det - 1..U_det + 1 {
                        for k in -U_det - 1..U_det + 1 {
                            for n in 0..self.natom() {
                                let mut atoms = use_atom_position.row(n).to_owned()
                                    + (i as f64) * U_inv.row(0).to_owned()
                                    + (j as f64) * U_inv.row(1).to_owned()
                                    + (k as f64) * U_inv.row(2).to_owned(); //原子的位置在新的坐标系下的坐标
                                atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                                    0.0
                                } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                                    1.0
                                } else {
                                    atoms[[0]]
                                };
                                atoms[[1]] = if atoms[[1]].abs() < 1e-8 {
                                    0.0
                                } else if (atoms[[1]] - 1.0).abs() < 1e-8 {
                                    1.0
                                } else {
                                    atoms[[1]]
                                };
                                atoms[[2]] = if atoms[[2]].abs() < 1e-8 {
                                    0.0
                                } else if (atoms[[2]] - 1.0).abs() < 1e-8 {
                                    1.0
                                } else {
                                    atoms[[2]]
                                };
                                if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                                    //判断是否在原胞内
                                    new_atom.push(Atom::new(
                                        atoms,
                                        self.atoms[n].norb(),
                                        self.atoms[n].atom_type(),
                                    ));
                                    for n0 in
                                        use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb()
                                    {
                                        //开始根据原子位置开始生成轨道
                                        let mut orbs = use_orb.row(n0).to_owned()
                                            + (i as f64) * U_inv.row(0).to_owned()
                                            + (j as f64) * U_inv.row(1).to_owned()
                                            + (k as f64) * U_inv.row(2).to_owned(); //新的轨道的坐标
                                        new_orb.push_row(orbs.view());
                                        new_orb_proj.push(self.orb_projection[n0]);
                                        orb_list.push(n0);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            2 => {
                for i in -U_det - 1..U_det + 1 {
                    for j in -U_det - 1..U_det + 1 {
                        for n in 0..self.natom() {
                            let mut atoms = use_atom_position.row(n).to_owned()
                                + (i as f64) * U_inv.row(0).to_owned()
                                + (j as f64) * U_inv.row(1).to_owned(); //原子的位置在新的坐标系下的坐标
                            atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                                0.0
                            } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                                1.0
                            } else {
                                atoms[[0]]
                            };
                            atoms[[1]] = if atoms[[1]].abs() < 1e-8 {
                                0.0
                            } else if (atoms[[1]] - 1.0).abs() < 1e-8 {
                                1.0
                            } else {
                                atoms[[1]]
                            };
                            if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                                //判断是否在原胞内
                                new_atom.push(Atom::new(
                                    atoms,
                                    self.atoms[n].norb(),
                                    self.atoms[n].atom_type(),
                                ));
                                for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb()
                                {
                                    //开始根据原子位置开始生成轨道
                                    let mut orbs = use_orb.row(n0).to_owned()
                                        + (i as f64) * U_inv.row(0).to_owned()
                                        + (j as f64) * U_inv.row(1).to_owned(); //新的轨道的坐标
                                    new_orb.push_row(orbs.view());
                                    new_orb_proj.push(self.orb_projection[n0]);
                                    orb_list.push(n0);
                                    //orb_list_R.push_row(&arr1(&[i,j]));
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                for i in -U_det - 1..U_det + 1 {
                    for n in 0..self.natom() {
                        let mut atoms = use_atom_position.row(n).to_owned()
                            + (i as f64) * U_inv.row(0).to_owned(); //原子的位置在新的坐标系下的坐标
                        atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                            0.0
                        } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                            1.0
                        } else {
                            atoms[[0]]
                        };
                        if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                            //判断是否在原胞内
                            new_atom.push(Atom::new(
                                atoms,
                                self.atoms[n].norb(),
                                self.atoms[n].atom_type(),
                            ));
                            for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb() {
                                //开始根据原子位置开始生成轨道
                                let mut orbs = use_orb.row(n0).to_owned()
                                    + (i as f64) * U_inv.row(0).to_owned(); //新的轨道的坐标
                                new_orb.push_row(orbs.view());
                                new_orb_proj.push(self.orb_projection[n0]);
                                orb_list.push(n0);
                                //orb_list_R.push_row(&arr1(&[i]));
                            }
                        }
                    }
                }
            }
            _ => todo!(),
        }
        //轨道位置和原子位置构建完成, 接下来我们开始构建哈密顿量
        let norb = new_orb.len_of(Axis(0));
        let nsta = if self.spin { 2 * norb } else { norb };
        let natom = new_atom.len();
        let n_R = self.hamR.len_of(Axis(0));
        let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r())); //超胞准备用的hamR
        let mut use_hamR = Array2::<isize>::zeros((1, self.dim_r())); //超胞的hamR的可能, 如果这个hamR没有对应的hopping就会被删除
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta)); //超胞准备用的ham
        //超胞准备用的rmatrix
        let mut new_rmatrix = Array4::<Complex<f64>>::zeros((1, self.dim_r(), nsta, nsta));
        let max_use_hamR = self.hamR.mapv(|x| x as f64);
        let max_use_hamR = max_use_hamR.dot(&U.inv().unwrap());
        let mut max_hamR =
            max_use_hamR
                .outer_iter()
                .fold(Array1::zeros(self.dim_r()), |mut acc, x| {
                    for i in 0..self.dim_r() {
                        acc[[i]] = if acc[[i]] > x[[i]].abs() {
                            acc[[i]]
                        } else {
                            x[[i]].abs()
                        };
                    }
                    acc
                });
        let max_R = max_hamR.mapv(|x| (x.ceil() as isize) + 1);
        //let mut max_R=Array1::<isize>::zeros(self.dim_r());
        //let max_R:isize=U_det.abs()*(self.dim_r() as isize);
        //let max_R=Array1::<isize>::ones(self.dim_r())*max_R;
        //用来产生可能的hamR
        match self.dim_r() {
            1 => {
                for i in -max_R[[0]]..max_R[[0]] + 1 {
                    if i != 0 {
                        use_hamR.push_row(array![i].view());
                    }
                }
            }
            2 => {
                for j in -max_R[[1]]..max_R[[1]] + 1 {
                    for i in -max_R[[0]]..max_R[[0]] + 1 {
                        if i != 0 || j != 0 {
                            use_hamR.push_row(array![i, j].view());
                        }
                    }
                }
            }
            3 => {
                for k in -max_R[[2]]..max_R[[2]] + 1 {
                    for i in -max_R[[0]]..max_R[[0]] + 1 {
                        for j in -max_R[[1]]..max_R[[1]] + 1 {
                            if i != 0 || j != 0 || k != 0 {
                                use_hamR.push_row(array![i, j, k].view());
                            }
                        }
                    }
                }
            }
            _ => todo!(),
        }
        let use_n_R = use_hamR.len_of(Axis(0));
        let mut gen_rmatrix: bool = false;
        if self.rmatrix.len_of(Axis(0)) == 1 {
            for i in 0..self.dim_r() {
                for s in 0..norb {
                    new_rmatrix[[0, i, s, s]] = Complex::new(new_orb[[s, i]], 0.0);
                }
            }
            if self.spin {
                for i in 0..self.dim_r() {
                    for s in 0..norb {
                        new_rmatrix[[0, i, s + norb, s + norb]] =
                            Complex::new(new_orb[[s, i]], 0.0);
                    }
                }
            }
        } else {
            gen_rmatrix = true;
        }
        if self.spin && gen_rmatrix {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                let mut use_rmatrix = Array3::<Complex<f64>>::zeros((self.dim_r(), nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64); //超胞的 R 在原始原胞的 R
                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        if let Some(index) = find_R(&self.hamR, &R0) {
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                            useham[[int_i + norb, int_j]] =
                                self.ham[[index, *use_i + self.norb(), *use_j]];
                            useham[[int_i, int_j + norb]] =
                                self.ham[[index, *use_i, *use_j + self.norb()]];
                            useham[[int_i + norb, int_j + norb]] =
                                self.ham[[index, *use_i + self.norb(), *use_j + self.norb()]];
                            for r in 0..self.dim_r() {
                                use_rmatrix[[r, int_i, int_j]] =
                                    self.rmatrix[[index, r, *use_i, *use_j]];
                                use_rmatrix[[r, int_i + norb, int_j]] =
                                    self.rmatrix[[index, r, *use_i + self.norb(), *use_j]];
                                use_rmatrix[[r, int_i, int_j + norb]] =
                                    self.rmatrix[[index, r, *use_i, *use_j + self.norb()]];
                                use_rmatrix[[r, int_i + norb, int_j + norb]] = self.rmatrix
                                    [[index, r, *use_i + self.norb(), *use_j + self.norb()]];
                            }
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_hamR.push_row(use_R.view());
                    new_rmatrix.push(Axis(0), use_rmatrix.view());
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                    new_rmatrix
                        .slice_mut(s![0, .., .., ..])
                        .assign(&use_rmatrix);
                }
            }
        } else if gen_rmatrix && !self.spin {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((norb, norb));
                let mut use_rmatrix = Array3::<Complex<f64>>::zeros((self.dim_r(), norb, norb));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64); //超胞的 R 在原始原胞的 R
                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        if let Some(index) = find_R(&self.hamR, &R0) {
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                            for r in 0..self.dim_r() {
                                use_rmatrix[[r, int_i, int_j]] =
                                    self.rmatrix[[index, r, *use_i, *use_j]]
                            }
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_rmatrix.push(Axis(0), use_rmatrix.view());
                    new_hamR.push_row(use_R);
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                    new_rmatrix
                        .slice_mut(s![0, .., .., ..])
                        .assign(&use_rmatrix);
                }
            }
        } else if self.spin {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> =
                            &new_orb.row(int_j) - &new_orb.row(int_i) + &use_R.map(|x| *x as f64); //超胞的 R 在原始原胞的 R

                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });

                        if let Some(index) = find_R(&self.hamR, &R0) {
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                            useham[[int_i + norb, int_j]] =
                                self.ham[[index, *use_i + self.norb(), *use_j]];
                            useham[[int_i, int_j + norb]] =
                                self.ham[[index, *use_i, *use_j + self.norb()]];
                            useham[[int_i + norb, int_j + norb]] =
                                self.ham[[index, *use_i + self.norb(), *use_j + self.norb()]];
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_hamR.push_row(use_R.view());
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                }
            }
        } else {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        //接下来计算超胞中的R在原胞中对应的hamR
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64); //超胞的 R 在原始原胞的 R
                        let R0: Array1<isize> =
                            (R0.dot(U) - self.orb.row(*use_j) + self.orb.row(*use_i)).mapv(|x| {
                                if x.fract().abs() < 1e-8 || x.fract().abs() > 1.0 - 1e-8 {
                                    x.round() as isize
                                } else {
                                    x.floor() as isize
                                }
                            });
                        if let Some(index) = find_R(&self.hamR, &R0) {
                            add_R = true;
                            useham[[int_i, int_j]] = self.ham[[index, *use_i, *use_j]];
                        } else {
                            continue;
                        }
                    }
                }
                if add_R && R != 0 {
                    new_ham.push(Axis(0), useham.view());
                    new_hamR.push_row(use_R);
                } else if R == 0 {
                    new_ham.slice_mut(s![0, .., ..]).assign(&useham);
                }
            }
        }
        let mut model = Model {
            dim_r: Dimension::try_from(self.dim_r())?,
            spin: self.spin,
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_proj,
            atoms: new_atom,
            ham: new_ham,
            hamR: new_hamR,
            rmatrix: new_rmatrix,
        };
        Ok(model)
    }

    #[allow(non_snake_case)]
    pub fn dos(
        &self,
        k_mesh: &Array1<usize>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        sigma: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        //! 我这里用的算法是高斯算法, 其算法过程如下
        //!
        //! 首先, 根据 k_mesh 算出所有的能量 $\ve_n$, 然后, 按照定义
        //! $$\rho(\ve)=\sum_N\int\dd\bm k \delta(\ve_n-\ve)$$
        //! 我们将 $\delta(\ve_n-\ve)$ 做了替换, 换成了 $\f{1}{\sqrt{2\pi}\sigma}e^{-\f{(\ve_n-\ve)^2}{2\sigma^2}}$
        //!
        //! 然后, 计算方法是先算出所有的能量, 再将能量乘以高斯分布, 就能得到态密度.
        //!
        //! 态密度的光滑程度和k点密度以及高斯分布的展宽有关
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk = kvec.len_of(Axis(0));
        let band = self.solve_band_all_parallel(&kvec);
        let E = Array1::linspace(E_min, E_max, E_n);
        let mut dos = Array1::<f64>::zeros(E_n);
        let dim: usize = k_mesh.len();
        let centre = band.into_raw_vec().into_par_iter();
        let sigma0 = 1.0 / sigma;
        let pi0 = 1.0 / (2.0 * PI).sqrt();
        let dos = Array1::<f64>::zeros(E_n);
        let dos = centre
            .fold(
                || Array1::<f64>::zeros(E_n),
                |acc, x| {
                    let A = (&E - x) * sigma0;
                    let f = (-&A * &A / 2.0).mapv(|x| x.exp()) * sigma0 * pi0;
                    acc + &f
                },
            )
            .reduce(|| Array1::<f64>::zeros(E_n), |acc, x| acc + x);
        let dos = dos / (nk as f64);
        Ok((E, dos))
    }

    ///这个函数是用来给模型添加磁场的
    pub fn add_magnetic_field(&self) -> Result<Model> {
        Ok(self.clone())
    }
}
