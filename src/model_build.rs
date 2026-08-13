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

use crate::Model;
use crate::SpinDirection;
use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
use crate::error::{Result, TbError};
use crate::generics::hop_use;
use crate::model::RMatrixData;
use crate::model_utils::find_R;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::Norm;
use ndarray_linalg::{Determinant, Inverse};
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
/// - `Some(SpinDirection::Y)`: Writes `+i * tmp` to up/down and `-i * tmp` to
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
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
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
    /// [`set_hop`], [`set_onsite`], and related methods.
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
    /// - `None` (0): `tmp * sigma_0` (identity)
    /// - [`SpinDirection::X`] (1): `tmp * sigma_x`
    /// - [`SpinDirection::Y`] (2): `tmp * sigma_y`
    /// - [`SpinDirection::Z`] (3): `tmp * sigma_z`
    ///
    /// For a spinless model (`Model<false>`), `pauli` is silently ignored.
    ///
    /// # Arguments
    /// * `tmp` - Hopping amplitude, `f64` (real) or `Complex<f64>`.
    /// * `ind_i` - Row orbital index (0-based, in the spinless basis).
    /// * `ind_j` - Column orbital index (0-based, in the spinless basis).
    /// * `R` - Lattice vector to the target cell. Must have length `DIM`.
    /// * `pauli` - Pauli matrix decoration. Accepts `u8`, `usize`, or
    ///   [`SpinDirection`].
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
    pub fn set_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<Option<SpinDirection>>,
    ) {
        let pauli: Option<SpinDirection> = pauli.into();
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli.is_some() && !SPIN {
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
    /// let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    ///
    /// // Set real part
    /// model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), None);
    /// // Add an imaginary part on top
    /// model.add_hop(Complex::new(0.0, 0.1), 0, 0, &arr1(&[1isize]), None);
    /// ```
    #[allow(non_snake_case)]
    pub fn add_hop<T: Data<Elem = isize>, U: hop_use>(
        &mut self,
        tmp: U,
        ind_i: usize,
        ind_j: usize,
        R: &ArrayBase<T, Ix1>,
        pauli: impl Into<Option<SpinDirection>>,
    ) {
        let pauli: Option<SpinDirection> = pauli.into();
        let tmp: Complex<f64> = tmp.to_complex();
        if pauli.is_some() && !SPIN {
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
        if let Some(index) = find_R(&self.hamR, &R) {
            let index_inv = find_R(&self.hamR, &(-R)).expect("Negative R not found in hamR");
            self.ham[[index, ind_i, ind_j]] = tmp;
            if index != 0 || ind_i != ind_j {
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
            self.grow_rmatrix_rows(2);
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
    /// added by [`set_hop`], [`add_hop`], or [`add_element`]: a hopping
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

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
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
    ///     Atom::with_orbitals(arr1(&[0.0, 0.0, 0.0]), AtomType::H, [OrbitalId::new(0)]),
    ///     Atom::with_orbitals(arr1(&[0.5, 0.5, 0.0]), AtomType::H, [OrbitalId::new(1)]),
    /// ];
    /// let mut model = Model::<false>::tb_model(lat, orb, Some(atoms)).unwrap();
    /// model.shift_to_atom().unwrap();
    /// ```
    pub fn shift_to_atom(&mut self) -> Result<()> {
        self.validate()?;
        for atom in &self.atoms {
            for &orbital in atom.orbitals() {
                self.orb
                    .row_mut(orbital.index())
                    .assign(atom.position_ref());
            }
        }
        Ok(())
    }

    /// Move the orbital positions to the positions of their parent atoms
    /// (alternate implementation).
    ///
    /// Performs the same operation as [`shift_to_atom`] but uses a different
    /// indexing pattern (iterates by atom index rather than by atom reference).
    /// See [`shift_to_atom`] for details.
    pub fn move_to_atom(&mut self) -> Result<()> {
        self.shift_to_atom()
    }

    /// Remove orbitals from the model.
    ///
    /// Deletes the specified orbitals together with all Hamiltonian and
    /// position matrix elements involving them. The `orb_projection` list
    /// is updated. Atoms whose orbital list becomes empty are retained as
    /// valid structural sites; call [`Model::prune_empty_atoms`] explicitly if
    /// they should also be removed.
    ///
    /// For spinful models, the corresponding spin-doubled indices are also
    /// removed (index `i + norb` is removed alongside `i`).
    ///
    /// # Arguments
    /// * `orb_list` - Indices of orbitals to remove (0-based, spinless
    ///   basis). Duplicate entries are not allowed.
    ///
    /// Returns a structured error for duplicate or out-of-range IDs.
    pub fn remove_orb(&mut self, orb_list: &[usize]) -> Result<()> {
        self.validate()?;
        let mut use_orb_list = orb_list.to_vec();
        use_orb_list.sort_unstable();
        let has_duplicates = { use_orb_list.windows(2).any(|window| window[0] == window[1]) };
        if has_duplicates {
            return Err(TbError::DuplicateOrbitals);
        }
        if let Some(&index) = use_orb_list.iter().find(|&&index| index >= self.norb()) {
            return Err(TbError::InvalidOrbitalId {
                index,
                norb: self.norb(),
            });
        }
        if use_orb_list.is_empty() {
            return Ok(());
        }
        let old_norb = self.norb();
        let mut index: Vec<_> = (0..old_norb)
            .filter(|num| use_orb_list.binary_search(num).is_err())
            .collect(); //要保留下来的元素
        let mut old_to_new = vec![None; old_norb];
        for (new, &old) in index.iter().enumerate() {
            old_to_new[old] = Some(OrbitalId::new(new));
        }
        self.orb = self.orb.select(Axis(0), &index);
        self.orb_projection = index.iter().map(|&old| self.orb_projection[old]).collect();
        for atom in &mut self.atoms {
            let remapped = atom
                .orbitals()
                .iter()
                .filter_map(|id| old_to_new[id.index()])
                .collect();
            atom.set_orbitals(remapped);
        }
        //开始计算nsta
        if SPIN {
            let index_add: Vec<_> = index.iter().map(|x| *x + old_norb).collect();
            index.extend(index_add);
        }
        //开始操作哈密顿量
        let new_ham = self.ham.select(Axis(1), &index);
        let new_ham = new_ham.select(Axis(2), &index);
        self.ham = new_ham;
        //开始操作rmatrix
        self.rmatrix = self.rmatrix.select_axes(Axis(2), &index, Axis(3), &index);
        self.validate()?;
        Ok(())
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
    /// Returns a structured error for duplicate or out-of-range IDs.
    pub fn remove_atom(&mut self, atom_list: &[usize]) -> Result<()> {
        self.validate()?;
        //----------判断是否存在重复, 并给出保留的index
        let mut use_atom_list = atom_list.to_vec();
        use_atom_list.sort_unstable();
        let has_duplicates = {
            use_atom_list
                .windows(2)
                .any(|window| window[0] == window[1])
        };
        if has_duplicates {
            return Err(TbError::DuplicateAtoms);
        }

        if let Some(&index) = use_atom_list.iter().find(|&&index| index >= self.natom()) {
            return Err(TbError::InvalidAtomId {
                index,
                natom: self.natom(),
            });
        }
        if use_atom_list.is_empty() {
            return Ok(());
        }
        let mut removed_orbitals = use_atom_list
            .iter()
            .flat_map(|&atom| self.atoms[atom].orbitals().iter())
            .map(|id| id.index())
            .collect::<Vec<_>>();
        removed_orbitals.sort_unstable();
        self.atoms = self
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| use_atom_list.binary_search(index).is_err())
            .map(|(_, atom)| atom.clone())
            .collect();
        self.remove_orb(&removed_orbitals)
    }

    /// Remove atoms together with every orbital they own.
    ///
    /// This is the explicit name for the historical cascade behavior of
    /// [`Self::remove_atom`].
    pub fn remove_atoms_and_orbitals(&mut self, atom_list: &[usize]) -> Result<()> {
        self.remove_atom(atom_list)
    }

    /// Remove only atomic-site metadata and leave its orbitals unassigned.
    pub fn remove_atoms_only(&mut self, atom_list: &[usize]) -> Result<()> {
        self.validate()?;
        let mut removed = atom_list.to_vec();
        removed.sort_unstable();
        if removed.windows(2).any(|window| window[0] == window[1]) {
            return Err(TbError::DuplicateAtoms);
        }
        if let Some(&index) = removed.iter().find(|&&index| index >= self.natom()) {
            return Err(TbError::InvalidAtomId {
                index,
                natom: self.natom(),
            });
        }
        self.atoms = self
            .atoms
            .iter()
            .enumerate()
            .filter(|(index, _)| removed.binary_search(index).is_err())
            .map(|(_, atom)| atom.clone())
            .collect();
        self.validate()
    }

    /// Remove atomic sites that currently own no tight-binding orbitals.
    pub fn prune_empty_atoms(&mut self) -> Result<()> {
        self.validate()?;
        self.atoms.retain(|atom| !atom.orbitals().is_empty());
        self.validate()
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
    ///     Atom::with_orbitals(arr1(&[0.0, 0.0, 0.0]), AtomType::H, [OrbitalId::new(0)]),
    ///     Atom::with_orbitals(arr1(&[0.5, 0.5, 0.0]), AtomType::H, [OrbitalId::new(1)]),
    /// ];
    /// let mut model = Model::<false>::tb_model(lat, orb, Some(atoms)).unwrap();
    ///
    /// // Swap atom 0 and atom 1
    /// model.reorder_atom(&[1, 0]).unwrap();
    /// ```
    pub fn reorder_atom(&mut self, order: &[usize]) -> Result<()> {
        self.validate()?;
        if order.len() != self.natom() {
            return Err(TbError::InvalidAtomPermutation {
                natom: self.natom(),
                order: order.to_vec(),
            });
        };
        let mut sorted_order = order.to_vec();
        sorted_order.sort_unstable();
        if sorted_order != (0..self.natom()).collect::<Vec<_>>() {
            return Err(TbError::InvalidAtomPermutation {
                natom: self.natom(),
                order: order.to_vec(),
            });
        }
        let owners = self.orbital_owners()?;
        let mut new_orb_order = order
            .iter()
            .flat_map(|&atom| self.atoms[atom].orbitals().iter())
            .map(|id| id.index())
            .collect::<Vec<_>>();
        new_orb_order.extend(
            owners
                .iter()
                .enumerate()
                .filter_map(|(orbital, owner)| owner.is_none().then_some(orbital)),
        );
        let mut old_to_new = vec![0usize; self.norb()];
        for (new, &old) in new_orb_order.iter().enumerate() {
            old_to_new[old] = new;
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
        //重排原子顺序并重映射其轨道引用
        for &old_atom in order {
            let mut atom = self.atoms[old_atom].clone();
            atom.set_orbitals(
                atom.orbitals()
                    .iter()
                    .map(|id| OrbitalId::new(old_to_new[id.index()]))
                    .collect(),
            );
            new_atom.push(atom);
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
        self.rmatrix =
            self.rmatrix
                .select_axes(Axis(2), &new_state_order, Axis(3), &new_state_order);
        self.validate()?;
        Ok(())
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
    /// * `U` - A `DIM x DIM` integer matrix with `det(U) > 0`.
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
    pub fn make_supercell(&self, U: &Array2<f64>) -> Result<Model<SPIN, DIM, R>> {
        self.validate()?;
        let orbital_owners = self.orbital_owners()?;
        if !self.atoms.is_empty() && orbital_owners.iter().any(Option::is_none) {
            // In a model with atoms, orbitals follow their parent atom; an
            // orbital without an owner has no atom whose image selection it
            // can follow, so reject the mixed state instead of treating it
            // independently.
            return Err(TbError::InvalidModelInvariant {
                invariant: "supercell_orbital_ownership",
                message: "the model has atoms, but some orbitals do not belong to \
                          any atom; orbitals must follow their parent atom in a supercell"
                    .to_string(),
            });
        }
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
        // Fold each owned orbital to the representative nearest its parent
        // atom before the basis change: the image shift is chosen to bring
        // the ATOM into [0, 1), and only an atom-adjacent orbital
        // representative can follow it into the supercell frame.  The
        // nearest representative is unambiguous because validate() enforces
        // |orb - atom| <= ORBITAL_ATOM_POSITION_TOLERANCE < 1/2 (mod 1).
        let mut use_orb = self.orb.clone();
        // Track the per-component span of the fold vectors: the R-vector
        // candidate range below must cover (n_j - n_i) · U_inv, otherwise
        // hoppings between oppositely folded orbitals fall outside the
        // enumeration and are silently dropped.
        let mut fold_min = Array1::<isize>::zeros(DIM);
        let mut fold_max = Array1::<isize>::zeros(DIM);
        for atom in &self.atoms {
            for &orbital_id in atom.orbitals() {
                let orbital = orbital_id.index();
                for axis in 0..DIM {
                    let shift = (use_orb[[orbital, axis]] - atom.position_ref()[[axis]]).round();
                    use_orb[[orbital, axis]] -= shift;
                    let shift = shift as isize;
                    fold_min[axis] = fold_min[axis].min(shift);
                    fold_max[axis] = fold_max[axis].max(shift);
                }
            }
        }
        let mut use_orb = use_orb.dot(&U_inv);
        //新的原子位置
        let use_atom_position = self.atom_position().dot(&U_inv);
        let mut orb_list: Vec<usize> = Vec::new();
        let mut new_orb = Array2::<f64>::zeros((0, self.dim_r()));
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new();
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
                                    let first_new_orbital = new_orb.nrows();
                                    for &source_orbital in self.atoms[n].orbitals() {
                                        let n0 = source_orbital.index();
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
                                    let mut atom = Atom::with_orbitals(
                                        atoms,
                                        self.atoms[n].atom_type(),
                                        (first_new_orbital..new_orb.nrows()).map(OrbitalId::new),
                                    );
                                    if let Some(moment) = self.atoms[n].magnetic_moment() {
                                        atom.set_magnetic_moment(moment)?;
                                    }
                                    new_atom.push(atom);
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
                                let first_new_orbital = new_orb.nrows();
                                for &source_orbital in self.atoms[n].orbitals() {
                                    let n0 = source_orbital.index();
                                    let o = use_orb.row(n0);
                                    let orbs = arr1(&[
                                        o[0] + i_f * u0[0] + j_f * u1[0],
                                        o[1] + i_f * u0[1] + j_f * u1[1],
                                    ]);
                                    new_orb.push_row(orbs.view());
                                    new_orb_proj.push(self.orb_projection[n0]);
                                    orb_list.push(n0);
                                }
                                let mut atom = Atom::with_orbitals(
                                    atoms,
                                    self.atoms[n].atom_type(),
                                    (first_new_orbital..new_orb.nrows()).map(OrbitalId::new),
                                );
                                if let Some(moment) = self.atoms[n].magnetic_moment() {
                                    atom.set_magnetic_moment(moment)?;
                                }
                                new_atom.push(atom);
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
                        atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                            0.0
                        } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                            1.0
                        } else {
                            atoms[[0]]
                        };
                        if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                            let first_new_orbital = new_orb.nrows();
                            for &source_orbital in self.atoms[n].orbitals() {
                                let n0 = source_orbital.index();
                                let o = use_orb.row(n0);
                                let orbs = arr1(&[o[0] + i_f * u0[0]]);
                                new_orb.push_row(orbs.view());
                                new_orb_proj.push(self.orb_projection[n0]);
                                orb_list.push(n0);
                            }
                            let mut atom = Atom::with_orbitals(
                                atoms,
                                self.atoms[n].atom_type(),
                                (first_new_orbital..new_orb.nrows()).map(OrbitalId::new),
                            );
                            if let Some(moment) = self.atoms[n].magnetic_moment() {
                                atom.set_magnetic_moment(moment)?;
                            }
                            new_atom.push(atom);
                        }
                    }
                }
            }
            _ => todo!(),
        }

        // Orbitals without an atomic owner are independent first-class basis
        // states. Replicate them by their own positions so orbital-only models
        // survive supercell construction without fabricated atoms.
        let unassigned_orbitals = orbital_owners
            .iter()
            .enumerate()
            .filter_map(|(orbital, owner)| owner.is_none().then_some(orbital))
            .collect::<Vec<_>>();
        let shifts = match self.dim_r() {
            3 => {
                let mut shifts = Vec::new();
                for i in -U_det - 1..U_det + 1 {
                    for j in -U_det - 1..U_det + 1 {
                        for k in -U_det - 1..U_det + 1 {
                            shifts.push(
                                i as f64 * U_inv.row(0).to_owned()
                                    + j as f64 * U_inv.row(1).to_owned()
                                    + k as f64 * U_inv.row(2).to_owned(),
                            );
                        }
                    }
                }
                shifts
            }
            2 => {
                let mut shifts = Vec::new();
                for i in -U_det - 1..U_det + 1 {
                    for j in -U_det - 1..U_det + 1 {
                        shifts.push(
                            i as f64 * U_inv.row(0).to_owned() + j as f64 * U_inv.row(1).to_owned(),
                        );
                    }
                }
                shifts
            }
            1 => (-U_det - 1..U_det + 1)
                .map(|i| i as f64 * U_inv.row(0).to_owned())
                .collect(),
            _ => {
                return Err(TbError::InvalidDimension {
                    dim: self.dim_r(),
                    supported: vec![1, 2, 3],
                });
            }
        };
        for shift in shifts {
            for &source in &unassigned_orbitals {
                let mut position = use_orb.row(source).to_owned() + &shift;
                for component in &mut position {
                    if component.abs() < 1e-8 {
                        *component = 0.0;
                    } else if (*component - 1.0).abs() < 1e-8 {
                        *component = 1.0;
                    }
                }
                if position.iter().all(|&value| (0.0..1.0).contains(&value)) {
                    new_orb.push_row(position.view());
                    new_orb_proj.push(self.orb_projection[source]);
                    orb_list.push(source);
                }
            }
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
        // Extend the candidate range by the pre-fold span mapped through
        // U_inv: the R0 lookup includes the -n_j + n_i contribution, so the
        // enumerated supercell vectors must reach
        // (R0 + n_j - n_i) · U_inv for every pair of fold vectors.
        let fold_span = &fold_max - &fold_min;
        let fold_extra = fold_span
            .mapv(|x| x as f64)
            .dot(&U_inv.mapv(|x| x.abs()))
            .mapv(|x| x.ceil() as isize);
        let max_R = max_hamR.mapv(|x| (x.ceil() as isize) + 1) + fold_extra;
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
        if !<R as RMatrixData>::HAS_RMATRIX {
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
        }
        if SPIN && R::HAS_RMATRIX {
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
                                let rmat = self.rmatrix.as_array4();
                                use_rmatrix[[r, int_i, int_j]] = rmat[[index, r, *use_i, *use_j]];
                                use_rmatrix[[r, int_i + norb, int_j]] =
                                    rmat[[index, r, *use_i + self.norb(), *use_j]];
                                use_rmatrix[[r, int_i, int_j + norb]] =
                                    rmat[[index, r, *use_i, *use_j + self.norb()]];
                                use_rmatrix[[r, int_i + norb, int_j + norb]] =
                                    rmat[[index, r, *use_i + self.norb(), *use_j + self.norb()]];
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
        } else if R::HAS_RMATRIX && !SPIN {
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
                                let rmat = self.rmatrix.as_array4();
                                use_rmatrix[[r, int_i, int_j]] = rmat[[index, r, *use_i, *use_j]]
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
        fold_supercell_positions_covariantly::<DIM>(
            &mut new_orb,
            &mut new_ham,
            &mut new_hamR,
            &mut new_rmatrix,
            SPIN,
        );
        // Covariant translation of the position-matrix diagonal: preserve
        // the source offset r_old(ss, 0) and add the Cartesian cell
        // displacement tau_new·L_new - tau_old·L_old of each copy.
        let old_diagonal = rmatrix_diagonal_cartesian::<SPIN, DIM, R>(self);
        set_rmatrix_diagonal_with_displacement::<DIM>(
            &mut new_rmatrix,
            &new_hamR,
            &new_orb,
            &new_lat,
            &self.orb,
            &self.lat,
            &old_diagonal,
            &orb_list,
            SPIN,
        );
        let model = Model {
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_proj,
            atoms: new_atom,
            ham: new_ham,
            hamR: new_hamR,
            rmatrix: R::from_array(new_rmatrix),
        };
        model.validate()?;
        Ok(model)
    }
}

/// Extract the `R = 0` position-matrix diagonal (Cartesian, shape
/// `(norb, DIM)`); all zeros for `NoRMatrix` models.
pub(crate) fn rmatrix_diagonal_cartesian<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
) -> Array2<f64> {
    let norb = model.norb();
    let mut diagonal = Array2::<f64>::zeros((norb, DIM));
    if !R::HAS_RMATRIX {
        return diagonal;
    }
    let zero_r = Array1::<isize>::zeros(DIM);
    let Some(r0) = find_R(&model.hamR, &zero_r) else {
        return diagonal;
    };
    let rmatrix = model.rmatrix.as_array4();
    for s in 0..norb {
        for axis in 0..DIM {
            diagonal[[s, axis]] = rmatrix[[r0, axis, s, s]].re;
        }
    }
    diagonal
}

/// Translate the `R = 0` position-matrix diagonal covariantly with a change
/// of orbital representative and lattice.
///
/// Position matrix elements are Cartesian (matching Wannier90 `_r.dat`), and
/// the on-site diagonal `r_ii(0)` may carry an arbitrary offset relative to
/// the orbital center (e.g. custom `_r.dat` data).  Moving orbital `s` from
/// `τ_old` on lattice `L_old` to `τ_new` on lattice `L_new` must translate
/// the diagonal by the Cartesian displacement without discarding the offset:
///
/// ```math
/// r^{new}_{ii}(0) = r^{old}_{ss}(0) + \tau^{new}_i \cdot L_{new}
///                   - \tau^{old}_s \cdot L_{old}.
/// ```
pub(crate) fn set_rmatrix_diagonal_with_displacement<const DIM: usize>(
    rmatrix: &mut Array4<Complex<f64>>,
    ham_r: &Array2<isize>,
    new_orb: &Array2<f64>,
    new_lat: &Array2<f64>,
    old_orb: &Array2<f64>,
    old_lat: &Array2<f64>,
    old_diagonal: &Array2<f64>,
    source: &[usize],
    spin: bool,
) {
    let zero_r = Array1::<isize>::zeros(DIM);
    let Some(r0) = find_R(ham_r, &zero_r) else {
        return;
    };
    let nsta = rmatrix.dim().2;
    let norb = new_orb.nrows();
    let new_cart = new_orb.dot(new_lat);
    let old_cart = old_orb.dot(old_lat);
    for i in 0..nsta {
        let s = if spin { i % norb } else { i };
        let src = source[s];
        for axis in 0..DIM {
            rmatrix[[r0, axis, i, i]] = Complex::new(
                old_diagonal[[src, axis]] + (new_cart[[s, axis]] - old_cart[[src, axis]]),
                0.0,
            );
        }
    }
}

/// Fold supercell orbital positions into `[0, 1)` and compensate every hopping
/// block so the physical link `(R + τ_j − τ_i)·L` is unchanged.
///
/// A supercell image places atom and orbital at the same shifted position, but
/// only the atom is tested against `[0, 1)`; an orbital displaced from its
/// parent atom can land outside the cell.  Folding orbital `s` by an integer
/// vector `n_s` must therefore move the hopping block `H_ij(R)` to
/// `R + n_j − n_i` (and the position-matrix block identically), which keeps the
/// Peierls link displacement and the `[r, H]` commutator invariant.
///
/// [`Model::validate`] guarantees every orbital sits within
/// [`ORBITAL_ATOM_POSITION_TOLERANCE`] of its parent atom (modulo a lattice
/// vector), so after folding the orbital remains attached to its atom; pure
/// orbital-only models already store in-cell positions and this function is a
/// no-op for them.
fn fold_supercell_positions_covariantly<const DIM: usize>(
    orb: &mut Array2<f64>,
    ham: &mut Array3<Complex<f64>>,
    ham_r: &mut Array2<isize>,
    rmatrix: &mut Array4<Complex<f64>>,
    spin: bool,
) {
    let nsta = ham.dim().1;
    let norb = orb.nrows();
    // Component-wise floor brings each coordinate into [0, 1).
    let mut fold = Array2::<isize>::zeros((norb, DIM));
    for s in 0..norb {
        for axis in 0..DIM {
            let n = orb[[s, axis]].floor() as isize;
            fold[[s, axis]] = n;
            orb[[s, axis]] -= n as f64;
        }
    }
    relabel_hamiltonian_by_orbital_fold::<DIM>(ham, ham_r, rmatrix, &fold, spin);
}

/// Apply an orbital gauge fold `τ_s → τ_s − n_s` to the Hamiltonian blocks.
///
/// The caller has already subtracted `n_s` from the orbital positions; this
/// function moves every hopping block `H_ij(R)` (and position-matrix block)
/// to `R + n_j − n_i`, keeping the physical link displacement
/// `(R + τ_j − τ_i)·L` and the `[r, H]` commutator invariant.
fn relabel_hamiltonian_by_orbital_fold<const DIM: usize>(
    ham: &mut Array3<Complex<f64>>,
    ham_r: &mut Array2<isize>,
    rmatrix: &mut Array4<Complex<f64>>,
    fold: &Array2<isize>,
    spin: bool,
) {
    let nsta = ham.dim().1;
    let norb = fold.nrows();
    // Fold vector per state: spin copies share their orbital's position.
    let mut state_fold = Vec::with_capacity(nsta);
    for i in 0..nsta {
        let s = if spin { i % norb } else { i };
        state_fold.push(fold.row(s).to_owned());
    }
    if state_fold.iter().all(|n| n.iter().all(|&x| x == 0)) {
        return;
    }
    // Rebuild the blocks with compensated R vectors.  A compensated vector may
    // leave the original hamR set, so find-or-append a row for it.
    let old_ham = ham.clone();
    let old_rmatrix = rmatrix.clone();
    let old_ham_r = ham_r.clone();
    ham.fill(Complex::new(0.0, 0.0));
    rmatrix.fill(Complex::new(0.0, 0.0));
    for (i_r, r_vec) in old_ham_r.outer_iter().enumerate() {
        for i in 0..nsta {
            for j in 0..nsta {
                let element = old_ham[[i_r, i, j]];
                let mut has_rmatrix = false;
                for axis in 0..DIM {
                    has_rmatrix |= old_rmatrix[[i_r, axis, i, j]].norm_sqr() != 0.0;
                }
                if element.norm_sqr() == 0.0 && !has_rmatrix {
                    continue;
                }
                let shift = &state_fold[j] - &state_fold[i];
                let new_r = &r_vec + &shift;
                let target = match find_R(ham_r, &new_r) {
                    Some(target) => target,
                    None => {
                        ham_r.push_row(new_r.view());
                        ham.push(Axis(0), Array2::<Complex<f64>>::zeros((nsta, nsta)).view());
                        rmatrix.push(
                            Axis(0),
                            Array3::<Complex<f64>>::zeros((DIM, nsta, nsta)).view(),
                        );
                        ham_r.nrows() - 1
                    }
                };
                ham[[target, i, j]] += element;
                for axis in 0..DIM {
                    rmatrix[[target, axis, i, j]] += old_rmatrix[[i_r, axis, i, j]];
                }
            }
        }
    }
}

/// Normalize a model's orbital gauge before operations that interpret
/// positions geometrically (cutting).
///
/// 1. Bring every atom into `[0, 1)`.
/// 2. Fold each owned orbital to the periodic image **nearest its parent
///    atom** (which may lie just outside `[0, 1)`, e.g. `atom = 0.99`
///    keeps `orb = 1.01`); unowned orbitals are folded into `[0, 1)`.
/// 3. Covariantly relabel the Hamiltonian blocks and reset the position
///    matrix diagonal to `τ · L`.
///
/// The result is physically identical to the input; the fold is a pure
/// gauge transformation `H_ij(R) → H_ij(R + n_j − n_i)`.
pub(crate) fn normalized_to_atoms<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
) -> Result<Model<SPIN, DIM, R>> {
    let mut out = model.clone();
    let old_orb = model.orb.clone();
    let old_lat = model.lat.clone();
    let old_diagonal = rmatrix_diagonal_cartesian::<SPIN, DIM, R>(model);
    // 1. Atoms into the cell.
    for atom in &mut out.atoms {
        let mut position = atom.position();
        for axis in 0..DIM {
            position[axis] -= position[axis].floor();
        }
        atom.set_position(position);
    }
    // 2. Per-orbital fold vectors.
    let owners = out.orbital_owners()?;
    let mut fold = Array2::<isize>::zeros((out.norb(), DIM));
    for (s, owner) in owners.iter().enumerate() {
        for axis in 0..DIM {
            let n = match owner {
                Some(atom_id) => {
                    (out.orb[[s, axis]] - out.atoms[atom_id.index()].position_ref()[[axis]]).round()
                }
                None => out.orb[[s, axis]].floor(),
            } as isize;
            fold[[s, axis]] = n;
            out.orb[[s, axis]] -= n as f64;
        }
    }
    // 3. Covariant relabel + covariant Cartesian diagonal translation
    // (preserving any offset the source diagonal carried relative to the
    // orbital center).
    let mut ham = out.ham.clone();
    let mut ham_r = out.hamR.clone();
    let mut rmatrix = if R::HAS_RMATRIX {
        out.rmatrix.as_array4().clone()
    } else {
        Array4::<Complex<f64>>::zeros((ham_r.nrows(), DIM, out.nsta(), out.nsta()))
    };
    relabel_hamiltonian_by_orbital_fold::<DIM>(&mut ham, &mut ham_r, &mut rmatrix, &fold, SPIN);
    let identity_source: Vec<usize> = (0..out.norb()).collect();
    set_rmatrix_diagonal_with_displacement::<DIM>(
        &mut rmatrix,
        &ham_r,
        &out.orb,
        &out.lat,
        &old_orb,
        &old_lat,
        &old_diagonal,
        &identity_source,
        SPIN,
    );
    out.ham = ham;
    out.hamR = ham_r;
    out.rmatrix = R::from_array(rmatrix);
    out.validate()?;
    Ok(out)
}

#[cfg(test)]
mod fold_tests {
    use super::*;
    use crate::solve_ham::solve;
    use crate::{Atom, AtomType, HasRMatrix, OrbitalId};

    /// 1D model whose orbital sits just across the cell boundary from its
    /// atom: atom at 0.99, orbital at `orbital_x`.  Both representatives
    /// (`1.01` and `0.01`) describe the same physical site.
    fn boundary_model(orbital_x: f64) -> Model<false, 1> {
        let mut model = Model::<false, 1>::tb_model(
            array![[1.0]],
            array![[orbital_x]],
            Some(vec![Atom::with_orbitals(
                array![0.99],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.add_hop(-1.0, 0, 0, &array![1], None);
        model
    }

    #[test]
    fn supercell_fold_preserves_physics_across_gauge_choices() {
        // With U = [2] the second supercell image places the atom at 0.995
        // but pushes the orbital at 1.005 outside [0, 1); the covariant fold
        // plus R compensation must keep the physics identical.
        let model_a = boundary_model(1.01);
        let model_b = boundary_model(0.01); // same site, other representative

        let sc_a = model_a.make_supercell(&array![[2.0]]).unwrap();
        let sc_b = model_b.make_supercell(&array![[2.0]]).unwrap();

        // The fold must bring every orbital back into [0, 1).
        for s in 0..sc_a.norb() {
            assert!(
                (0.0..1.0).contains(&sc_a.orb[[s, 0]]),
                "supercell orbital {s} = {} outside [0, 1)",
                sc_a.orb[[s, 0]]
            );
        }
        // Atoms keep owning orbitals within tolerance after folding.
        sc_a.validate().unwrap();
        sc_b.validate().unwrap();

        // Gauge equivalence: the two representatives must produce identical
        // supercells (same folded positions, same compensated hoppings).
        assert_eq!(sc_a.orb, sc_b.orb);
        assert_eq!(sc_a.hamR, sc_b.hamR);
        for (block_a, block_b) in sc_a.ham.outer_iter().zip(sc_b.ham.outer_iter()) {
            assert!(
                block_a
                    .iter()
                    .zip(block_b.iter())
                    .all(|(a, b)| (*a - *b).norm() < 1e-14),
                "supercell hopping blocks differ between gauge choices"
            );
        }

        // Band-folding check: the supercell spectrum at fractional k_sc must
        // equal the primitive spectrum at k = k_sc / 2 and k = k_sc / 2 + 1/2.
        let k_sc = 0.3;
        let band_sc = sc_a.solve_band_onek(&array![k_sc]);
        let e_prim_1 = model_a.solve_band_onek(&array![k_sc / 2.0])[0];
        let e_prim_2 = model_a.solve_band_onek(&array![k_sc / 2.0 + 0.5])[0];
        let mut expected = vec![e_prim_1, e_prim_2];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut got: Vec<f64> = band_sc.to_vec();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "supercell band {b} does not match folded primitive band {a} at k_sc = {k_sc}"
            );
        }
    }

    #[test]
    fn identity_supercell_preserves_spectrum_with_opposite_folds() {
        // Regression: the R-vector candidate range did not cover the
        // pre-fold contribution (n_j - n_i)·U_inv.  With two orbitals
        // folded in opposite directions (n_0 = -1, n_1 = +1) and a hopping
        // at the largest R, the identity supercell (U = 1) silently dropped
        // the hopping and its spectrum differed from the original model.
        let atoms = vec![
            Atom::with_orbitals(array![0.01], AtomType::C, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.99], AtomType::O, [OrbitalId::new(1)]),
        ];
        // Orbital 0 sits just left of atom A (fold n_0 = -1), orbital 1
        // just right of atom B (fold n_1 = +1).
        let mut model =
            Model::<false, 1>::tb_model(array![[1.0]], array![[-1.01], [1.99]], Some(atoms))
                .unwrap();
        model.add_hop(-0.7, 0, 1, &array![1], None);
        model.validate().unwrap();

        let sc = model.make_supercell(&array![[1.0]]).unwrap();
        sc.validate().unwrap();
        assert_eq!(sc.norb(), model.norb(), "identity supercell must keep norb");

        // The identity supercell spectrum must equal the original spectrum.
        let k = array![0.3];
        let mut expected: Vec<f64> = model.solve_band_onek(&k).to_vec();
        let mut got: Vec<f64> = sc.solve_band_onek(&k).to_vec();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (a, b) in expected.iter().zip(got.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "identity supercell band {b} does not match original band {a}"
            );
        }
    }

    #[test]
    fn identity_supercell_preserves_custom_rmatrix_diagonal() {
        // Regression: the diagonal was unconditionally overwritten with
        // tau·L, clobbering a legitimate custom offset from _r.dat data.
        // The identity supercell must preserve r_old(ss, 0) exactly.
        let mut model = Model::<false, 1, HasRMatrix>::tb_model(
            array![[1.0]],
            array![[0.5]],
            None,
        )
        .unwrap();
        model.rmatrix.as_array4_mut()[[0, 0, 0, 0]] = Complex::new(0.7, 0.0);
        model.add_hop(-1.0, 0, 0, &array![1], None);

        let sc = model.make_supercell(&array![[1.0]]).unwrap();
        sc.validate().unwrap();
        let rmatrix = sc.rmatrix.as_array4();
        let zero_r = Array1::<isize>::zeros(1);
        let r0 = find_R(&sc.hamR, &zero_r).unwrap();
        assert!(
            (rmatrix[[r0, 0, 0, 0]] - Complex::new(0.7, 0.0)).norm() < 1e-12,
            "identity supercell must preserve the custom diagonal 0.7, found {}",
            rmatrix[[r0, 0, 0, 0]]
        );
    }

    #[test]
    fn supercell_rmatrix_diagonal_is_cartesian_position() {
        // Regression: supercell copies must carry the per-image Cartesian
        // cell displacement in the position-matrix diagonal, and the fold
        // must translate it consistently, so that
        // rmatrix[0, :, i, i] == orb[i, :].dot(lat) holds for the folded
        // supercell (HasRMatrix variant of the boundary model).
        let lat = array![[1.0]];
        let orb = array![[1.01]];
        let atoms = vec![Atom::with_orbitals(
            array![0.99],
            AtomType::C,
            [OrbitalId::new(0)],
        )];
        let mut model = Model::<false, 1, HasRMatrix>::tb_model(lat, orb, Some(atoms)).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1], None);

        let sc = model.make_supercell(&array![[2.0]]).unwrap();
        sc.validate().unwrap();

        let rmatrix = sc.rmatrix.as_array4();
        let cart = sc.orb.dot(&sc.lat);
        let zero_r = Array1::<isize>::zeros(1);
        let r0 = find_R(&sc.hamR, &zero_r).unwrap();
        for i in 0..sc.nsta() {
            assert!(
                (rmatrix[[r0, 0, i, i]] - Complex::new(cart[[i, 0]], 0.0)).norm() < 1e-12,
                "rmatrix diagonal ({i}) must equal frac·lat = {}, found {}",
                cart[[i, 0]],
                rmatrix[[r0, 0, i, i]]
            );
        }
    }
}
