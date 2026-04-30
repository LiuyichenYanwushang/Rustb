//! Model construction and Hamiltonian manipulation methods.
//!
//! This module provides the builder pattern for [`Model`] construction. It contains
//! methods for setting on-site energies, adding hopping terms, managing orbital and
//! atomic positions, and building supercells via transformation matrices.
//!
//! All hopping terms are stored in the convention
//!
//! ```math
//! \langle i,\mathbf{0} | \hat{H} | j,\mathbf{R} \rangle
//! ```
//!
//! where `i` and `j` are orbital indices, and `R` is a lattice vector in units of
//! primitive cell vectors. Hermitian conjugates are automatically generated: when a
//! hopping with lattice vector `R` is added, the term with `-R` and interchanged
//! orbital indices is also added with the complex conjugate of the hopping
//! amplitude.
//!
//! For spinful models (`spin = true`), the basis is doubled: the first `norb`
//! entries correspond to spin-up, and the second `norb` entries to spin-down. The
//! `pauli` parameter controls which Pauli matrix acts in spin space.
//!
//! # Conventions
//!
//! - **Lattice vectors** `R` are stored as integer vectors in units of primitive
//!   cell vectors (dimensionless).
//! - **Orbital positions** are stored in fractional coordinates with respect to the
//!   lattice vectors.
//! - **On-site energies** must be real. A panic (or error) will result if an
//!   on-site term with a non-zero imaginary part is set.

use crate::Model;
use crate::atom_struct::{Atom, AtomType, OrbProj};
use crate::error::{Result, TbError};
use crate::generics::hop_use;
use crate::model_utils::find_R;
use crate::{Dimension, SpinDirection};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::Norm;
use ndarray_linalg::{Determinant, Inverse};
use num_complex::Complex;

/// Overwrite Hamiltonian matrix elements with spin decoration.
///
/// This internal macro writes a hopping amplitude `tmp` into the Hamiltonian
/// matrix at the given orbital indices `(ind_i, ind_j)`, respecting the spin
/// degree of freedom. The behavior depends on the [`SpinDirection`]:
///
/// - [`SpinDirection::None`]: Writes `tmp` to both spin blocks (spin-up/up and
///   spin-down/down). Corresponds to `sigma_0` (identity) in spin space.
/// - [`SpinDirection::x`]: Writes `tmp` to the off-diagonal spin blocks
///   (up/down and down/up). Corresponds to `sigma_x`.
/// - [`SpinDirection::y`]: Writes `+i * tmp` to up/down and `-i * tmp` to
///   down/up. Corresponds to `sigma_y`.
/// - [`SpinDirection::z`]: Writes `+tmp` to up/up and `-tmp` to down/down.
///   Corresponds to `sigma_z`.
///
/// If the model is spinless (`spin = false`), the hopping is simply written
/// at `(ind_i, ind_j)` without any spin structure.
///
/// # Parameters
/// * `$spin` - `bool` indicating whether the model has spin
/// * `$pauli` - [`SpinDirection`] selecting the Pauli matrix in spin space
/// * `$tmp` - The hopping amplitude (type `Complex<f64>`)
/// * `$new_ham` - Mutable view of the Hamiltonian matrix
/// * `$ind_i` - Row orbital index (without spin doubling)
/// * `$ind_j` - Column orbital index (without spin doubling)
/// * `$norb` - Number of orbitals (without spin doubling)
macro_rules! update_hamiltonian {
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                crate::SpinDirection::None => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = $tmp;
                }
                crate::SpinDirection::x => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] = $tmp;
                }
                crate::SpinDirection::y => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] = -$tmp * Complex::<f64>::i();
                }
                crate::SpinDirection::z => {
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

/// Add to Hamiltonian matrix elements with spin decoration (accumulating version).
///
/// This internal macro is the accumulating counterpart of
/// [`update_hamiltonian!`]. Instead of overwriting the matrix element, it
/// **adds** the hopping amplitude `tmp` to the existing value. The spin
/// decoration follows the same Pauli matrix rules described in
/// [`update_hamiltonian!`].
///
/// # Parameters
/// * `$spin` - `bool` indicating whether the model has spin
/// * `$pauli` - [`SpinDirection`] selecting the Pauli matrix in spin space
/// * `$tmp` - The hopping amplitude (type `Complex<f64>`)
/// * `$new_ham` - Mutable view of the Hamiltonian matrix
/// * `$ind_i` - Row orbital index (without spin doubling)
/// * `$ind_j` - Column orbital index (without spin doubling)
/// * `$norb` - Number of orbitals (without spin doubling)
macro_rules! add_hamiltonian {
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                crate::SpinDirection::None => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] += $tmp;
                }
                crate::SpinDirection::x => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] += $tmp;
                }
                crate::SpinDirection::y => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] -= $tmp * Complex::<f64>::i();
                }
                crate::SpinDirection::z => {
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

impl<const SPIN: bool> Model<SPIN> {
    /// Create a new tight-binding model with the given crystal structure.
    ///
    /// This constructor initializes a [`Model`] with the specified lattice
    /// vectors and orbital positions. The Hamiltonian and position matrices
    /// start with a single on-site block (for `R = 0`) and are populated using
    /// [`set_hop`], [`set_onsite`], and related methods.
    ///
    /// If no `atom` list is provided, atoms are inferred from the orbital
    /// positions: orbitals closer than `1e-2` (in fractional coordinates) are
    /// assigned to the same atom. This heuristic works best when orbitals from
    /// different atoms are well separated.
    ///
    /// # Arguments
    /// * `dim_r` - Real-space dimensionality: 1, 2, or 3.
    /// * `lat` - Lattice vectors as a `dim_r x dim_r` matrix. Each row is a
    ///   lattice vector.
    /// * `orb` - Orbital positions in fractional coordinates, shape
    ///   `(norb, dim_r)`.
    /// * `atom` - Optional list of [`Atom`] objects. If `None`, atoms are
    ///   inferred from `orb`.
    ///
    /// # Returns
    /// `Result<Model<SPIN>>` containing the initialized tight-binding model.
    ///
    /// # Errors
    /// Returns [`TbError::LatticeDimensionError`] if `lat` is not a square
    /// `dim_r x dim_r` matrix.
    ///
    /// # Examples
    ///
    /// Create a 2D graphene model:
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0, 0.0], [-0.5, 3_f64.sqrt() / 2.0]];
    /// let orb = array![[1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0 / 3.0]];
    /// let mut model = Model::<false>::tb_model(2, lat, orb, None).unwrap();
    /// ```
    ///
    /// Create a spinful model with explicit atoms:
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    /// use Rustb::atom_struct::*;
    ///
    /// let lat = array![[1.0, 0.0, 0.0],
    ///                  [0.0, 1.0, 0.0],
    ///                  [0.0, 0.0, 1.0]];
    /// let orb = array![[0.0, 0.0, 0.0]];
    /// let atom = vec![Atom::new(arr1(&[0.0, 0.0, 0.0]), 1, AtomType::H)];
    /// let mut model = Model::<true>::tb_model(3, lat, orb, Some(atom)).unwrap();
    /// ```
    pub fn tb_model(
        dim_r: usize,
        lat: Array2<f64>,
        orb: Array2<f64>,
        atom: Option<Vec<Atom>>,
    ) -> Result<Model<SPIN>> {
        let norb: usize = orb.len_of(Axis(0));
        let nsta: usize = if SPIN { 2 * norb } else { norb };
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
                if SPIN {
                    rmatrix[[0, r, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, r]]);
                }
            }
        }
        let orb_projection = vec![OrbProj::s; norb];
        let mut model = Model {
            dim_r: Dimension::try_from(dim_r)?,
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

    /// Set the orbital projections for every orbital in the model.
    ///
    /// Orbital projections determine the angular-momentum character of each
    /// orbital (e.g., `s`, `px`, `dxy`). They are needed for Slater-Koster
    /// interpolation, Wannier90 import, and operations that depend on orbital
    /// symmetry.
    ///
    /// The length of `proj` should match `self.norb()`.
    ///
    /// # Arguments
    /// * `proj` - A vector of [`OrbProj`] values, one per orbital.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    /// use Rustb::atom_struct::*;
    ///
    /// let mut model = Model::<false>::tb_model(
    ///     3,
    ///     array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ///     array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
    ///     None,
    /// ).unwrap();
    /// model.set_projection(&vec![OrbProj::pz, OrbProj::pz]);
    /// ```
    pub fn set_projection(&mut self, proj: &Vec<OrbProj>) {
        self.orb_projection = proj.clone();
    }

    /// Set (overwrite) a hopping term in the tight-binding Hamiltonian.
    ///
    /// Sets the matrix element
    ///
    /// ```math
    /// \langle i,\mathbf{0} | \hat{H} | j,\mathbf{R} \rangle = \text{tmp}
    /// ```
    ///
    /// where `ind_i` and `ind_j` are orbital indices in the primitive-cell
    /// basis (without spin doubling), and `R` is the lattice vector to the
    /// target unit cell in units of primitive cell vectors.
    ///
    /// # Hermitian conjugate
    ///
    /// The Hermitian conjugate at `-R` is **automatically** set:
    ///
    /// ```math
    /// \langle j,-\mathbf{R} | \hat{H} | i,\mathbf{0} \rangle = \text{tmp}^*
    /// ```
    ///
    /// For on-site terms (`R = 0`, `i != j`), the conjugate is set within the
    /// same block. Diagonal on-site terms (`R = 0`, `i == j`) must be real.
    ///
    /// # Spin handling
    ///
    /// If the model is spinful, `pauli` determines the Pauli matrix
    /// decoration:
    ///
    /// - [`SpinDirection::None`] (0): `tmp * sigma_0` (identity)
    /// - [`SpinDirection::x`] (1): `tmp * sigma_x`
    /// - [`SpinDirection::y`] (2): `tmp * sigma_y`
    /// - [`SpinDirection::z`] (3): `tmp * sigma_z`
    ///
    /// For a spinless model, `pauli` is ignored.
    ///
    /// # Arguments
    /// * `tmp` - Hopping amplitude, `f64` (real) or `Complex<f64>`.
    /// * `ind_i` - Row orbital index (0-based, in the spinless basis).
    /// * `ind_j` - Column orbital index (0-based, in the spinless basis).
    /// * `R` - Lattice vector to the target cell. Must have length `dim_r`.
    /// * `pauli` - Pauli matrix decoration. Accepts `u8`, `usize`, or
    ///   [`SpinDirection`].
    ///
    /// # Panics
    /// Panics if `R.len() != dim_r`, if `ind_i` or `ind_j` is out of bounds,
    /// or if an on-site term (`R=0`, `i=j`) has a non-zero imaginary part.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0]];
    /// let orb = array![[0.0]];
    /// let mut model = Model::<false>::tb_model(1, lat, orb, None).unwrap();
    ///
    /// // Nearest-neighbor hopping to the right: <0,0|H|0,R=+1> = -1.0
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), 0);
    /// // Set on-site energy: <0,0|H|0,R=0> = 0.0
    /// model.set_hop(0.0_f64, 0, 0, &arr1(&[0isize]), 0);
    /// ```
    #[allow(non_snake_case)]
    pub fn set_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<SpinDirection>,
    ) {
        let pauli: SpinDirection = pauli.into();
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli != SpinDirection::None && !SPIN {
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
                    SPIN,
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
                        SPIN,
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
                    update_hamiltonian!(SPIN, pauli, tmp, new_ham, ind_i, ind_j, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), R.view()).unwrap();
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(SPIN, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), negative_R.view()).unwrap();
            }
        }
    }

    /// Add to a hopping term (accumulate without overwriting).
    ///
    /// Identical to [`set_hop`] except the hopping amplitude is **added** to
    /// any existing value:
    ///
    /// ```math
    /// \langle i,\mathbf{0} | \hat{H} | j,\mathbf{R} \rangle \mathrel{+}= \text{tmp}
    /// ```
    ///
    /// Useful when building a Hamiltonian from multiple contributions (e.g.,
    /// separate kinetic and spin-orbit coupling terms for the same orbital
    /// pair). The Hermitian conjugate at `-R` is also updated with `tmp*`.
    ///
    /// See [`set_hop`] for a full description of the parameters and panics.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use num_complex::Complex;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0]];
    /// let orb = array![[0.0]];
    /// let mut model = Model::<false>::tb_model(1, lat, orb, None).unwrap();
    ///
    /// // Set real part
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), 0);
    /// // Add an imaginary part on top
    /// model.add_hop(Complex::new(0.0, 0.1), 0, 0, &arr1(&[1isize]), 0);
    /// ```
    #[allow(non_snake_case)]
    pub fn add_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<SpinDirection>,
    ) {
        let pauli: SpinDirection = pauli.into();
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli != SpinDirection::None && !SPIN {
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
                    SPIN,
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
                        SPIN,
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
                    update_hamiltonian!(SPIN, pauli, tmp, new_ham, ind_i, ind_j, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), R.view()).unwrap();
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(SPIN, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), negative_R.view()).unwrap();
            }
        }
    }

    /// Add a matrix element directly, bypassing spin decoration.
    ///
    /// Sets the single matrix element
    ///
    /// ```math
    /// \langle i,\mathbf{0} | \hat{H} | j,\mathbf{R} \rangle = \text{tmp}
    /// ```
    ///
    /// using the **full** (spin-doubled) basis indices. Unlike [`set_hop`] and
    /// [`add_hop`], it does **not** apply Pauli matrix decoration. The indices
    /// `ind_i` and `ind_j` must be in `0..nsta()`.
    ///
    /// This is the low-level interface for Hamiltonian manipulation, useful
    /// when fine-grained control over individual spin components is needed.
    ///
    /// The Hermitian conjugate at `-R` is automatically set.
    ///
    /// # Arguments
    /// * `tmp` - Complex hopping amplitude in the full spin-doubled basis.
    /// * `ind_i` - Row orbital index (0-based, up to `nsta()`).
    /// * `ind_j` - Column orbital index (0-based, up to `nsta()`).
    /// * `R` - Lattice vector to the target unit cell.
    ///
    /// # Returns
    /// `Result<()>` with an error on invalid input.
    ///
    /// # Errors
    /// - [`TbError::RVectorLengthError`] if `R.len() != dim_r`.
    /// - [`TbError::DimensionMismatch`] if `ind_i` or `ind_j` >= `nsta()`.
    /// - [`TbError::OnsiteHoppingMustBeReal`] if an on-site term has a
    ///   non-zero imaginary part.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use num_complex::Complex;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0, 0.0], [0.0, 1.0]];
    /// let orb = array![[0.0, 0.0]];
    /// // Spinful model: norb=1, nsta=2
    /// let mut model = Model::<true>::tb_model(2, lat, orb, None).unwrap();
    ///
    /// // Spin-flip hopping: <up,0|H|down,R=(1,0)> = 0.5
    /// model.add_element(
    ///     Complex::new(0.5, 0.0),
    ///     0, 1, // up orbital -> down orbital
    ///     &arr1(&[1isize, 0isize]),
    /// ).unwrap();
    /// ```
    #[allow(non_snake_case)]
    pub fn add_element(
        &mut self,
        tmp: Complex<f64>,
        ind_i: usize,
        ind_j: usize,
        R: &Array1<isize>,
    ) -> Result<()> {
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

    /// Set (overwrite) all on-site energies at once.
    ///
    /// Convenience method that calls [`set_hop`] for every orbital `i` with
    /// `R = 0`:
    ///
    /// ```math
    /// \langle i,\mathbf{0} | \hat{H} | i,\mathbf{0} \rangle = \text{tmp}[i]
    /// ```
    ///
    /// # Arguments
    /// * `tmp` - Array of length `norb` with on-site energies (real).
    /// * `pauli` - Pauli matrix decoration. Use [`SpinDirection::None`] for
    ///   spin-independent on-site energies.
    ///
    /// # Panics
    /// Panics if `tmp.len() != norb`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0, 0.0], [0.0, 1.0]];
    /// let orb = array![[0.0, 0.0], [0.5, 0.5]];
    /// let mut model = Model::<false>::tb_model(2, lat, orb, None).unwrap();
    /// model.set_onsite(&arr1(&[1.0, -1.0]), 0);
    /// ```
    #[allow(non_snake_case)]
    pub fn set_onsite(&mut self, tmp: &Array1<f64>, pauli: impl Into<SpinDirection>) {
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

    /// Add to all on-site energies (accumulate without overwriting).
    ///
    /// Accumulating counterpart of [`set_onsite`]. Adds `tmp[i]` to the
    /// existing on-site energy of orbital `i`:
    ///
    /// ```math
    /// \langle i,\mathbf{0} | \hat{H} | i,\mathbf{0} \rangle \mathrel{+}= \text{tmp}[i]
    /// ```
    ///
    /// Useful when building up on-site energies from multiple contributions
    /// (e.g., crystal-field splitting plus a Zeeman term).
    ///
    /// # Arguments
    /// * `tmp` - Array of length `norb` with on-site energies to add.
    /// * `pauli` - Pauli matrix decoration for spinful models.
    ///
    /// # Panics
    /// Panics if `tmp.len() != norb`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0]];
    /// let orb = array![[0.0]];
    /// let mut model = Model::<false>::tb_model(1, lat, orb, None).unwrap();
    ///
    /// model.set_onsite(&arr1(&[1.0]), 0);
    /// model.add_onsite(&arr1(&[0.5]), 0);
    /// // Total on-site energy is now 1.5
    /// ```
    #[allow(non_snake_case)]
    pub fn add_onsite(&mut self, tmp: &Array1<f64>, pauli: impl Into<SpinDirection>) {
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

    /// Set a single on-site energy for one orbital.
    ///
    /// Sets the diagonal matrix element for orbital `ind` at `R = 0`:
    ///
    /// ```math
    /// \langle \text{ind},\mathbf{0} | \hat{H} | \text{ind},\mathbf{0} \rangle = \text{tmp}
    /// ```
    ///
    /// Convenience wrapper around [`set_hop`] with `R = 0`.
    ///
    /// # Arguments
    /// * `tmp` - The on-site energy (must be real).
    /// * `ind` - Orbital index (0-based, in the spinless basis).
    /// * `pauli` - Pauli matrix decoration for spinful models.
    ///
    /// # Panics
    /// Panics if `ind >= norb`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let lat = array![[1.0, 0.0], [0.0, 1.0]];
    /// let orb = array![[0.0, 0.0], [0.5, 0.5]];
    /// let mut model = Model::<false>::tb_model(2, lat, orb, None).unwrap();
    ///
    /// model.set_onsite_one(1.0, 0, 0); // E_0 = 1.0
    /// model.set_onsite_one(-1.0, 1, 0); // E_1 = -1.0
    /// ```
    #[allow(non_snake_case)]
    pub fn set_onsite_one(&mut self, tmp: f64, ind: usize, pauli: impl Into<SpinDirection>) {
        let pauli = pauli.into();
        let R = Array1::<isize>::zeros(self.dim_r());
        self.set_hop(Complex::new(tmp, 0.0), ind, ind, &R, pauli)
    }

    /// Delete (zero out) a hopping term.
    ///
    /// Sets the specified hopping to zero via [`set_hop`] with amplitude 0.
    /// Both `+R` and `-R` terms (and their spin components) are zeroed.
    ///
    /// # Arguments
    /// * `ind_i` - Row orbital index (spinless basis).
    /// * `ind_j` - Column orbital index (spinless basis).
    /// * `R` - Lattice vector of the hopping to delete.
    /// * `pauli` - Pauli matrix decoration (must match the one used when
    ///   the hopping was originally set).
    ///
    /// # Panics
    /// Panics if `R.len() != dim_r` or orbital indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let mut model = Model::<false>::tb_model(
    ///     1, array![[1.0]], array![[0.0]], None,
    /// ).unwrap();
    ///
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), 0);
    /// // Remove the hopping
    /// model.del_hop(0, 0, &arr1(&[1isize]), 0);
    /// ```
    pub fn del_hop(
        &mut self,
        ind_i: usize,
        ind_j: usize,
        R: &Array1<isize>,
        pauli: impl Into<SpinDirection>,
    ) {
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
}

impl<const SPIN: bool> Model<SPIN> {
    /// Move the orbital positions to the positions of their parent atoms.
    ///
    /// Sets each orbital's fractional-coordinate position to the
    /// fractional-coordinate position of the atom it belongs to. Useful
    /// when orbitals are initially at their Wannier centers but you want to
    /// align them with atomic positions for Slater-Koster parametrization
    /// or symmetry analysis.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    /// use Rustb::atom_struct::*;
    ///
    /// let lat = array![[1.0, 0.0, 0.0],
    ///                  [0.0, 1.0, 0.0],
    ///                  [0.0, 0.0, 1.0]];
    /// let orb = array![[0.1, 0.1, 0.0], [0.6, 0.6, 0.0]];
    /// let atoms = vec![
    ///     Atom::new(arr1(&[0.0, 0.0, 0.0]), 1, AtomType::H),
    ///     Atom::new(arr1(&[0.5, 0.5, 0.0]), 1, AtomType::H),
    /// ];
    /// let mut model = Model::<false>::tb_model(3, lat, orb, Some(atoms)).unwrap();
    /// model.shift_to_atom();
    /// ```
    pub fn shift_to_atom(&mut self) {
        let mut a = 0;
        for (i, atom) in self.atoms.iter().enumerate() {
            for j in 0..atom.norb() {
                self.orb.row_mut(a).assign(&atom.position());
                a += 1;
            }
        }
    }

    /// Move the orbital positions to the positions of their parent atoms
    /// (alternate implementation).
    ///
    /// Performs the same operation as [`shift_to_atom`] but uses a different
    /// indexing pattern (iterates by atom index rather than by atom reference).
    /// See [`shift_to_atom`] for details.
    pub fn move_to_atom(&mut self) {
        let mut a = 0;
        for i in 0..self.natom() {
            for j in 0..self.atoms[i].norb() {
                self.orb.row_mut(a).assign(&self.atoms[i].position());
                a += 1;
            }
        }
    }

    /// Remove orbitals from the model.
    ///
    /// Deletes the specified orbitals together with all Hamiltonian and
    /// position matrix elements involving them. The `orb_projection` list
    /// is updated, and atoms whose orbital count drops to zero are removed.
    ///
    /// For spinful models, the corresponding spin-doubled indices are also
    /// removed (index `i + norb` is removed alongside `i`).
    ///
    /// # Arguments
    /// * `orb_list` - Indices of orbitals to remove (0-based, spinless
    ///   basis). Duplicate entries are not allowed.
    ///
    /// # Panics
    /// Panics if `orb_list` contains duplicate entries.
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
        if SPIN {
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

    /// Remove entire atoms from the model.
    ///
    /// Removes the specified atoms and all orbitals belonging to them. The
    /// Hamiltonian, position matrix, and orbital projections are all updated
    /// to reflect the reduced basis.
    ///
    /// # Arguments
    /// * `atom_list` - Indices of atoms to remove (0-based). Duplicates are
    ///   not allowed.
    ///
    /// # Panics
    /// Panics if `atom_list` contains duplicates.
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
        if SPIN {
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

    /// Reorder atoms and their associated orbitals.
    ///
    /// Rearranges atoms according to the given permutation `order`. Orbitals
    /// are reordered to follow their parent atoms, and the Hamiltonian
    /// matrix, position matrix, and orbital projections are all permuted
    /// accordingly. Primarily useful for checking and debugging models
    /// (e.g., verifying invariance under atom permutations).
    ///
    /// # Arguments
    /// * `order` - A permutation of `0..natom()` giving the new atom order.
    ///   Must have length `natom()`.
    ///
    /// # Panics
    /// Panics if `order.len() != natom()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    /// use Rustb::atom_struct::*;
    ///
    /// let lat = array![[1.0, 0.0, 0.0],
    ///                  [0.0, 1.0, 0.0],
    ///                  [0.0, 0.0, 1.0]];
    /// let orb = array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]];
    /// let atoms = vec![
    ///     Atom::new(arr1(&[0.0, 0.0, 0.0]), 1, AtomType::H),
    ///     Atom::new(arr1(&[0.5, 0.5, 0.0]), 1, AtomType::H),
    /// ];
    /// let mut model = Model::<false>::tb_model(3, lat, orb, Some(atoms)).unwrap();
    ///
    /// // Swap atom 0 and atom 1
    /// model.reorder_atom(&vec![1, 0]);
    /// ```
    pub fn reorder_atom(&mut self, order: &Vec<usize>) {
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
        let new_state_order = if SPIN {
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

    /// Build a supercell by applying an integer transformation matrix `U`.
    ///
    /// The new lattice vectors are
    ///
    /// ```math
    /// L' = U \, L
    /// ```
    ///
    /// where `L` is the original lattice matrix (each row is a lattice
    /// vector) and `U` is an integer matrix with `det(U) > 0`. The supercell
    /// volume is multiplied by `det(U)`.
    ///
    /// # Algorithm
    ///
    /// 1. Compute the new lattice `L' = U * L`.
    /// 2. Map all orbitals into the enlarged cell, keeping those whose
    ///    fractional coordinates fall in `[0, 1)` on the new basis.
    /// 3. For each orbital pair and each possible supercell lattice vector
    ///    `R'`, compute the corresponding primitive-cell `R0` and copy the
    ///    hopping from the original Hamiltonian.
    /// 4. Position matrix elements are also mapped if present.
    ///
    /// # Arguments
    /// * `U` - A `dim_r x dim_r` integer matrix with `det(U) > 0`.
    ///
    /// # Returns
    /// `Result<Model>` containing the supercell model.
    ///
    /// # Errors
    /// - [`TbError::TransformationMatrixDimMismatch`] if `U` has wrong
    ///   dimensions.
    /// - [`TbError::InvalidSupercellDet`] if `det(U) <= 0`.
    /// - [`TbError::InvalidSupercellMatrix`] if `U` contains non-integer
    ///   entries.
    pub fn make_supercell(&self, U: &Array2<f64>) -> Result<Model<SPIN>> {
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
        if U.iter().any(|&x| x.fract() > 1e-8) {
            return Err(TbError::InvalidSupercellMatrix);
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

        // Pre-fetch U_inv rows and use scalar arithmetic: avoids per-iteration
        // .to_owned() heap allocations, replacing 3-5 allocs/iter with 1 arr1! call.
        match self.dim_r() {
            3 => {
                let u0 = U_inv.row(0).to_owned();
                let u1 = U_inv.row(1).to_owned();
                let u2 = U_inv.row(2).to_owned();
                for i in -U_det - 1..U_det + 1 {
                    let i_f = i as f64;
                    for j in -U_det - 1..U_det + 1 {
                        let j_f = j as f64;
                        for k in -U_det - 1..U_det + 1 {
                            let k_f = k as f64;
                            for n in 0..self.natom() {
                                let a = use_atom_position.row(n);
                                let mut atoms = arr1(&[
                                    a[0] + i_f * u0[0] + j_f * u1[0] + k_f * u2[0],
                                    a[1] + i_f * u0[1] + j_f * u1[1] + k_f * u2[1],
                                    a[2] + i_f * u0[2] + j_f * u1[2] + k_f * u2[2],
                                ]);
                                atoms[[0]] = if atoms[[0]].abs() < 1e-8 { 0.0 } else if (atoms[[0]] - 1.0).abs() < 1e-8 { 1.0 } else { atoms[[0]] };
                                atoms[[1]] = if atoms[[1]].abs() < 1e-8 { 0.0 } else if (atoms[[1]] - 1.0).abs() < 1e-8 { 1.0 } else { atoms[[1]] };
                                atoms[[2]] = if atoms[[2]].abs() < 1e-8 { 0.0 } else if (atoms[[2]] - 1.0).abs() < 1e-8 { 1.0 } else { atoms[[2]] };
                                if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                                    new_atom.push(Atom::new(atoms, self.atoms[n].norb(), self.atoms[n].atom_type()));
                                    for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb() {
                                        let o = use_orb.row(n0);
                                        let orbs = arr1(&[
                                            o[0] + i_f * u0[0] + j_f * u1[0] + k_f * u2[0],
                                            o[1] + i_f * u0[1] + j_f * u1[1] + k_f * u2[1],
                                            o[2] + i_f * u0[2] + j_f * u1[2] + k_f * u2[2],
                                        ]);
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
                let u0 = U_inv.row(0).to_owned();
                let u1 = U_inv.row(1).to_owned();
                for i in -U_det - 1..U_det + 1 {
                    let i_f = i as f64;
                    for j in -U_det - 1..U_det + 1 {
                        let j_f = j as f64;
                        for n in 0..self.natom() {
                            let a = use_atom_position.row(n);
                            let mut atoms = arr1(&[
                                a[0] + i_f * u0[0] + j_f * u1[0],
                                a[1] + i_f * u0[1] + j_f * u1[1],
                            ]);
                            atoms[[0]] = if atoms[[0]].abs() < 1e-8 { 0.0 } else if (atoms[[0]] - 1.0).abs() < 1e-8 { 1.0 } else { atoms[[0]] };
                            atoms[[1]] = if atoms[[1]].abs() < 1e-8 { 0.0 } else if (atoms[[1]] - 1.0).abs() < 1e-8 { 1.0 } else { atoms[[1]] };
                            if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                                new_atom.push(Atom::new(atoms, self.atoms[n].norb(), self.atoms[n].atom_type()));
                                for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb() {
                                    let o = use_orb.row(n0);
                                    let orbs = arr1(&[
                                        o[0] + i_f * u0[0] + j_f * u1[0],
                                        o[1] + i_f * u0[1] + j_f * u1[1],
                                    ]);
                                    new_orb.push_row(orbs.view());
                                    new_orb_proj.push(self.orb_projection[n0]);
                                    orb_list.push(n0);
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                let u0 = U_inv.row(0).to_owned();
                for i in -U_det - 1..U_det + 1 {
                    let i_f = i as f64;
                    for n in 0..self.natom() {
                        let a = use_atom_position.row(n);
                        let mut atoms = arr1(&[a[0] + i_f * u0[0]]);
                        atoms[[0]] = if atoms[[0]].abs() < 1e-8 { 0.0 } else if (atoms[[0]] - 1.0).abs() < 1e-8 { 1.0 } else { atoms[[0]] };
                        if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                            new_atom.push(Atom::new(atoms, self.atoms[n].norb(), self.atoms[n].atom_type()));
                            for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb() {
                                let o = use_orb.row(n0);
                                let orbs = arr1(&[o[0] + i_f * u0[0]]);
                                new_orb.push_row(orbs.view());
                                new_orb_proj.push(self.orb_projection[n0]);
                                orb_list.push(n0);
                            }
                        }
                    }
                }
            }
            _ => todo!(),
        }
        //轨道位置和原子位置构建完成, 接下来我们开始构建哈密顿量
        let norb = new_orb.len_of(Axis(0));
        let nsta = if SPIN { 2 * norb } else { norb };
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
            if SPIN {
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
        if SPIN && gen_rmatrix {
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
        } else if gen_rmatrix && !SPIN {
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
        } else if SPIN {
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
        // Keep new_rmatrix in sync with new_ham for magnetic field compatibility
        let n_r = new_ham.len_of(Axis(0));
        if new_rmatrix.len_of(Axis(0)) < n_r {
            let extra = n_r - new_rmatrix.len_of(Axis(0));
            let zero_rm = Array3::<Complex<f64>>::zeros((self.dim_r(), nsta, nsta));
            for _ in 0..extra {
                new_rmatrix.push(Axis(0), zero_rm.view());
            }
        }
        let mut model = Model {
            dim_r: Dimension::try_from(self.dim_r())?,
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
}
