//! Model construction and Hamiltonian setup methods.
//!
//! This module provides the [`Model`] constructor and in-place setup methods:
//! setting orbital projections and on-site energies, adding or replacing
//! hopping terms, and removing hopping blocks. Structural editing lives in
//! `model::edit`; supercell construction lives in `model::supercell`.
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
//! For spinful models (`Model::<true>`), the basis is doubled: the first `norb`
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

//!
//! # Examples
//!
//! Construct a chain, fill in hoppings and on-site terms, then manipulate the
//! result with the supercell and atom/orbital editing API.
//!
//! ```rust
//! use ndarray::{arr1, arr2, array};
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let mut chain = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
//! chain.set_onsite(&arr1(&[0.1]), None);
//!
//! // set_hop replaces a block, add_hop accumulates it. Both maintain the
//! // Hermitian-conjugate block at -R, so the second call adds two rows.
//! chain.set_hop(-1.0, 0, 0, &array![1], None);
//! chain.add_hop(-0.2, 0, 0, &array![2], None);
//! assert_eq!(chain.hamR.nrows(), 5); // 0, +-1, +-2
//!
//! // Spin decoration: None is the identity term, Some(axis) a Pauli term.
//! let mut spinful = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
//! spinful.add_hop(0.1, 0, 0, &array![1], SpinDirection::Z);
//!
//! // Integer transformation matrices define supercells.
//! let doubled = chain.make_supercell(&arr2(&[[2.0]]))?;
//! assert_eq!(doubled.norb(), 2 * chain.norb());
//!
//! // Orbital, atom and shape editing keep the dense arrays consistent.
//! let mut two = Model::<false, 1>::tb_model(
//!     array![[1.0]],
//!     array![[0.0], [0.25]],
//!     None,
//! )?;
//! two.remove_orb(&[1])?;
//! assert_eq!(two.norb(), 1);
//! # Ok(())
//! # }
//! ```

use crate::Model;
use crate::SpinDirection;
use crate::atom_struct::{Atom, OrbProj};
use crate::error::{Result, TbError};
use crate::model::RMatrixData;
use crate::model_utils::find_R;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

/// Overwrite Hamiltonian matrix elements with spin decoration.
///
/// This internal macro writes a hopping amplitude `tmp` into the Hamiltonian
/// matrix at the given orbital indices `(ind_i, ind_j)`, respecting the spin
/// degree of freedom. The behavior depends on [`Option<SpinDirection>`]:
///
/// - `None`: Writes `tmp` to both spin blocks (spin-up/up and
///   spin-down/down). Corresponds to `sigma_0` (identity) in spin space.
/// - `Some(SpinDirection::X)`: Writes `tmp` to the off-diagonal spin blocks
///   (up/down and down/up). Corresponds to `sigma_x`.
/// - `Some(SpinDirection::Y)`: Writes `-i * tmp` to up/down and `+i * tmp` to
///   down/up. Corresponds to `sigma_y`.
/// - `Some(SpinDirection::Z)`: Writes `+tmp` to up/up and `-tmp` to down/down.
///   Corresponds to `sigma_z`.
///
/// For spinless models (the default), the hopping is simply written
/// at `(ind_i, ind_j)` without any spin structure.
///
/// # Parameters
/// * `$spin` - compile-time constant (`SPIN` const generic) indicating
///   whether the model has spin
/// * `$pauli` - [`Option<SpinDirection>`] selecting the Pauli matrix in spin space
/// * `$tmp` - The hopping amplitude (type `Complex<f64>`)
/// * `$new_ham` - Mutable view of the Hamiltonian matrix
/// * `$ind_i` - Row orbital index (without spin doubling)
/// * `$ind_j` - Column orbital index (without spin doubling)
/// * `$norb` - Number of orbitals (without spin doubling)
macro_rules! update_hamiltonian {
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                None => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = $tmp;
                }
                Some(crate::SpinDirection::X) => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] = $tmp;
                }
                Some(crate::SpinDirection::Y) => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] = -$tmp * Complex::<f64>::i();
                }
                Some(crate::SpinDirection::Z) => {
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
/// * `$spin` - compile-time constant (`SPIN` const generic) indicating
///   whether the model has spin
/// * `$pauli` - [`Option<SpinDirection>`] selecting the Pauli matrix in spin space
/// * `$tmp` - The hopping amplitude (type `Complex<f64>`)
/// * `$new_ham` - Mutable view of the Hamiltonian matrix
/// * `$ind_i` - Row orbital index (without spin doubling)
/// * `$ind_j` - Column orbital index (without spin doubling)
/// * `$norb` - Number of orbitals (without spin doubling)
macro_rules! add_hamiltonian {
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                None => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] += $tmp;
                }
                Some(crate::SpinDirection::X) => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] += $tmp;
                }
                Some(crate::SpinDirection::Y) => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] -= $tmp * Complex::<f64>::i();
                }
                Some(crate::SpinDirection::Z) => {
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

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Create a new tight-binding model with the given crystal structure.
    ///
    /// This constructor initializes a [`Model`] with the specified lattice
    /// vectors and orbital positions. The Hamiltonian and position matrices
    /// start with a single on-site block (for `R = 0`) and are populated using
    /// [`Self::set_hop`], [`Self::set_onsite`], and related methods.
    ///
    /// If no `atom` list is provided, the result is an orbital-only model with
    /// an empty atomic structure. Rustb never invents atomic species or treats
    /// Wannier centers as crystallographic sites.
    ///
    /// # Arguments
    /// * `lat` - Lattice vectors as a `DIM x DIM` matrix. Each row is a
    ///   lattice vector. The dimensionality `DIM` is determined by the
    ///   const generic on [`Model<SPIN, DIM>`] (default: 3).
    /// * `orb` - Orbital positions in fractional coordinates, shape
    ///   `(norb, DIM)`.
    /// * `atom` - Optional explicit list of [`Atom`] objects. `None` creates an
    ///   orbital-only model.
    ///
    /// The `SPIN` and `DIM` const generics determine the basis and
    /// dimensionality. Use `Model::<true>::tb_model(...)` for spinful models,
    /// `Model::<false>::tb_model(...)` for spinless. For non-default
    /// dimensionality, specify `DIM`: e.g., `Model::<false, 2>::tb_model(...)`.
    ///
    /// # Returns
    /// `Result<Model<SPIN>>` containing the initialized tight-binding model.
    ///
    /// # Errors
    /// Returns [`TbError::LatticeDimensionError`] if `lat` is not a square
    /// `DIM x DIM` matrix.
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
    /// let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
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
    /// let atom = vec![Atom::with_orbitals(
    ///     arr1(&[0.0, 0.0, 0.0]),
    ///     AtomType::H,
    ///     [OrbitalId::new(0)],
    /// )];
    /// let mut model = Model::<true>::tb_model(lat, orb, Some(atom)).unwrap();
    /// ```
    pub fn tb_model(
        lat: Array2<f64>,
        orb: Array2<f64>,
        atom: Option<Vec<Atom>>,
    ) -> Result<Model<SPIN, DIM, R>> {
        let norb: usize = orb.len_of(Axis(0));
        let nsta: usize = if SPIN { 2 * norb } else { norb };
        if lat.len_of(Axis(1)) != DIM {
            return Err(TbError::LatticeDimensionError {
                expected: DIM,
                actual: lat.len_of(Axis(1)),
            });
        }
        if lat.len_of(Axis(0)) != lat.len_of(Axis(1)) {
            return Err(TbError::LatticeDimensionError {
                expected: lat.len_of(Axis(1)),
                actual: lat.len_of(Axis(0)),
            });
        }
        // Check the orbital shape before from_orb: the Cartesian conversion
        // orb.dot(lat) would otherwise panic on a column mismatch instead of
        // returning a structured error.
        if orb.len_of(Axis(1)) != DIM {
            return Err(TbError::InvalidModelInvariant {
                invariant: "orbital_position_shape",
                message: format!("expected {DIM} columns, found {}", orb.len_of(Axis(1))),
            });
        }
        let new_atom = match atom {
            Some(atom0) => atom0,
            None => Vec::new(),
        };
        let ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
        let hamR = Array2::<isize>::zeros((1, DIM));
        let rmatrix = R::from_orb(&orb, &lat, norb, SPIN, DIM);
        let orb_projection = vec![OrbProj::s; norb];
        let model = Model {
            lat,
            orb,
            orb_projection,
            atoms: new_atom,
            ham,
            hamR,
            rmatrix,
        };
        model.validate()?;
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
    /// \langle j,\mathbf{0} | \hat{H} | i,-\mathbf{R} \rangle = \text{tmp}^*
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
    /// - `None`: `tmp * sigma_0` (identity)
    /// - [`SpinDirection::X`]: `tmp * sigma_x`
    /// - [`SpinDirection::Y`]: `tmp * sigma_y`
    /// - [`SpinDirection::Z`]: `tmp * sigma_z`
    ///
    /// For a spinless model (`Model<false>`), a `Some` spin decoration is
    /// ignored with a warning on standard error.
    ///
    /// # Arguments
    /// * `tmp` - Hopping amplitude convertible into `Complex<f64>`, including
    ///   real `f64` and complex `Complex<f64>` values.
    /// * `ind_i` - Row orbital index (0-based, in the spinless basis).
    /// * `ind_j` - Column orbital index (0-based, in the spinless basis).
    /// * `R` - Lattice vector to the target cell. Must have length `DIM`.
    /// * `pauli` - Pauli matrix decoration. Pass a [`SpinDirection`] or an
    ///   `Option<SpinDirection>`; `None` selects the spin-independent term.
    ///
    /// # Panics
    /// Panics if `R.len() != DIM`, if `ind_i` or `ind_j` is out of bounds,
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
    /// let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    ///
    /// // Nearest-neighbor hopping to the right: <0,0|H|0,R=+1> = -1.0
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), None);
    /// // Set on-site energy: <0,0|H|0,R=0> = 0.0
    /// model.set_hop(0.0_f64, 0, 0, &arr1(&[0isize]), None);
    /// ```
    #[allow(non_snake_case)]
    pub fn set_hop<T: Data<Elem = isize>, U: Into<Complex<f64>>>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<Option<SpinDirection>>,
    ) {
        let pauli: Option<SpinDirection> = pauli.into();
        let tmp: Complex<f64> = tmp.into();
        if pauli.is_some() && !SPIN {
            eprintln!("Warning: pauli is ignored because this Model is spinless (SPIN=false)")
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

        // Transactional pre-check: on-site terms (R = 0, i == j) must be
        // real. This must happen before any Hamiltonian block is written, so a
        // rejected input leaves the model unchanged.
        let is_onsite = ind_i == ind_j && R.iter().all(|&x| x == 0);
        assert!(
            !(is_onsite && tmp.im != 0.0),
            "Wrong, the onsite hopping must be real, but here is {}",
            tmp
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
                if !is_onsite {
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
            }
            None => {
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham = update_hamiltonian!(SPIN, pauli, tmp, new_ham, ind_i, ind_j, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), R.view()).unwrap();
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(SPIN, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), negative_R.view()).unwrap();
                self.grow_rmatrix_rows(2);
            }
        }
    }

    /// Add to a hopping term (accumulate without overwriting).
    ///
    /// Identical to [`Self::set_hop`] except the hopping amplitude is **added** to
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
    /// See [`Self::set_hop`] for a full description of the parameters and panics.
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
    /// let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    ///
    /// // Set real part
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), None);
    /// // Add an imaginary part on top
    /// model.add_hop(Complex::new(0.0, 0.1), 0, 0, &arr1(&[1isize]), None);
    /// ```
    #[allow(non_snake_case)]
    pub fn add_hop<T: Data<Elem = isize>, U: Into<Complex<f64>>>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<Option<SpinDirection>>,
    ) {
        let pauli: Option<SpinDirection> = pauli.into();
        let tmp: Complex<f64> = tmp.into();
        if pauli.is_some() && !SPIN {
            eprintln!("Warning: pauli is ignored because this Model is spinless (SPIN=false)")
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

        // Transactional pre-check: on-site terms (R = 0, i == j) must be
        // real. This must happen before any Hamiltonian block is written, so a
        // rejected input leaves the model unchanged.
        let is_onsite = ind_i == ind_j && R.iter().all(|&x| x == 0);
        assert!(
            !(is_onsite && tmp.im != 0.0),
            "Wrong, the onsite hopping must be real, but here is {}",
            tmp
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
                if !is_onsite {
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
            }
            None => {
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham = update_hamiltonian!(SPIN, pauli, tmp, new_ham, ind_i, ind_j, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), R.view()).unwrap();
                let mut new_ham = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));

                let new_ham =
                    update_hamiltonian!(SPIN, pauli, tmp.conj(), new_ham, ind_j, ind_i, norb);
                self.ham.push(Axis(0), new_ham.view()).unwrap();
                self.hamR.push(Axis(0), negative_R.view()).unwrap();
                self.grow_rmatrix_rows(2);
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
    /// using the **full** (spin-doubled) basis indices. Unlike [`Self::set_hop`] and
    /// [`Self::add_hop`], it does **not** apply Pauli matrix decoration. The indices
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
    /// - [`TbError::RVectorLengthError`] if `R.len() != DIM`.
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
    /// let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
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
        // Transactional pre-check: an on-site term with a non-zero imaginary
        // part must error before any Hamiltonian block is written.
        let onsite = ind_i == ind_j && R.iter().all(|&x| x == 0);
        if onsite && tmp.im != 0.0 {
            return Err(TbError::OnsiteHoppingMustBeReal(tmp));
        }
        if let Some(index) = find_R(&self.hamR, &R) {
            let index_inv = find_R(&self.hamR, &(-R)).expect("Negative R not found in hamR");
            self.ham[[index, ind_i, ind_j]] = tmp;
            if !onsite {
                self.ham[[index_inv, ind_j, ind_i]] = tmp.conj();
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
            self.grow_rmatrix_rows(2);
        }
        Ok(())
    }

    /// Set (overwrite) all on-site energies at once.
    ///
    /// Convenience method that calls [`Self::set_hop`] for every orbital `i` with
    /// `R = 0`:
    ///
    /// ```math
    /// \langle i,\mathbf{0} | \hat{H} | i,\mathbf{0} \rangle = \text{tmp}[i]
    /// ```
    ///
    /// # Arguments
    /// * `tmp` - Array of length `norb` with on-site energies (real).
    /// * `pauli` - Pauli matrix decoration. Use `None` for
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
    /// let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    /// model.set_onsite(&arr1(&[1.0, -1.0]), None);
    /// ```
    #[allow(non_snake_case)]
    pub fn set_onsite(&mut self, tmp: &Array1<f64>, pauli: impl Into<Option<SpinDirection>>) {
        let pauli: Option<SpinDirection> = pauli.into();
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
    /// Accumulating counterpart of [`Self::set_onsite`]. Adds `tmp[i]` to the
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
    /// let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    ///
    /// model.set_onsite(&arr1(&[1.0]), None);
    /// model.add_onsite(&arr1(&[0.5]), None);
    /// // Total on-site energy is now 1.5
    /// ```
    #[allow(non_snake_case)]
    pub fn add_onsite(&mut self, tmp: &Array1<f64>, pauli: impl Into<Option<SpinDirection>>) {
        let pauli: Option<SpinDirection> = pauli.into();
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
    /// Convenience wrapper around [`Self::set_hop`] with `R = 0`.
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
    /// let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    ///
    /// model.set_onsite_one(1.0, 0, None); // E_0 = 1.0
    /// model.set_onsite_one(-1.0, 1, None); // E_1 = -1.0
    /// ```
    #[allow(non_snake_case)]
    pub fn set_onsite_one(
        &mut self,
        tmp: f64,
        ind: usize,
        pauli: impl Into<Option<SpinDirection>>,
    ) {
        let pauli: Option<SpinDirection> = pauli.into();
        let R = Array1::<isize>::zeros(self.dim_r());
        self.set_hop(Complex::new(tmp, 0.0), ind, ind, &R, pauli)
    }

    /// Delete (zero out) a hopping term.
    ///
    /// Sets the specified hopping to zero via [`Self::set_hop`] with amplitude 0.
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
    /// Panics if `R.len() != DIM` or orbital indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::*;
    /// use Rustb::*;
    ///
    /// let mut model = Model::<false, 1>::tb_model(
    ///     array![[1.0]], array![[0.0]], None,
    /// ).unwrap();
    ///
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), None);
    /// // Remove the hopping
    /// model.del_hop(0, 0, &arr1(&[1isize]), None);
    /// ```
    pub fn del_hop(
        &mut self,
        ind_i: usize,
        ind_j: usize,
        R: &Array1<isize>,
        pauli: impl Into<Option<SpinDirection>>,
    ) {
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

    /// Append `count` zero position-matrix blocks for newly added hopping vectors.
    ///
    /// Keeps `rmatrix` shape in sync with `hamR` when new hopping vectors are
    /// added by [`Self::set_hop`], [`Self::add_hop`], or [`add_element`]: a hopping
    /// introduced through these methods has no position-matrix elements yet,
    /// so the corresponding blocks are zero-filled.  Compile-time eliminated
    /// for `NoRMatrix` models.
    fn grow_rmatrix_rows(&mut self, count: usize) {
        if R::HAS_RMATRIX {
            let zero_block = Array3::<Complex<f64>>::zeros((DIM, self.nsta(), self.nsta()));
            for _ in 0..count {
                self.rmatrix
                    .as_array4_mut()
                    .push(Axis(0), zero_block.view())
                    .expect("rmatrix row push cannot fail");
            }
        }
    }
}
