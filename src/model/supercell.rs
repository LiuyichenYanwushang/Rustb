//! Supercell construction and integer-lattice helpers.
//!
//! `make_supercell` maps a model to an integer-transformed cell and
//! `normalized_to_atoms` performs the gauge normalization used by cutting.
//! The private helpers below implement checked integer linear algebra, quotient
//! enumeration and the covariant orbital/R-matrix relabeling.

use super::utils::find_R;
use super::{Model, RMatrixData};
use crate::atom_struct::{Atom, OrbitalId};
use crate::error::{Result, TbError};
use ndarray::*;
use num_complex::Complex;
use std::collections::{HashMap, VecDeque};

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
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
    /// 1. Round the tolerance-validated input to an exact integer matrix and
    ///    compute its determinant and adjugate with checked integer arithmetic.
    /// 2. Enumerate the row-vector quotient `Z^DIM / (Z^DIM U)` exactly using
    ///    the canonical key `r adj(U) mod det(U)`.
    /// 3. Normalize each orbital to its parent atom, replicate every quotient
    ///    image, and retain its exact primitive-cell label.
    /// 4. Map each existing Hamiltonian and position-matrix block directly to
    ///    its unique target image and new lattice vector. No candidate-`R`
    ///    search or floating-point block lookup is used.
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
    /// - [`TbError::InvalidSupercellMatrix`] if `U` contains non-finite or
    ///   non-integer entries, or if an integer label overflows the supported
    ///   range.
    pub fn make_supercell(&self, U: &Array2<f64>) -> Result<Model<SPIN, DIM, R>> {
        self.validate()?;
        let orbital_owners = self.orbital_owners()?;
        if !self.atoms.is_empty() && orbital_owners.iter().any(Option::is_none) {
            return Err(TbError::InvalidModelInvariant {
                invariant: "supercell_orbital_ownership",
                message: "the model has atoms, but some orbitals do not belong to \
                          any atom; orbitals must follow their parent atom in a supercell"
                    .to_string(),
            });
        }
        if U.dim() != (DIM, DIM) {
            return Err(TbError::TransformationMatrixDimMismatch {
                expected: DIM,
                actual: U.len_of(Axis(0)),
            });
        }
        if !U.iter().all(|value| value.is_finite())
            || U.iter().any(|&value| (value - value.round()).abs() > 1e-8)
        {
            return Err(TbError::InvalidSupercellMatrix);
        }

        // Canonicalize values accepted within the integer tolerance. Using the
        // rounded matrix consistently avoids building the lattice with U while
        // enumerating images with round(U).
        let rounded_u = U.mapv(f64::round);
        let mut integer_u = vec![vec![0_isize; DIM]; DIM];
        for row in 0..DIM {
            for column in 0..DIM {
                let value = rounded_u[[row, column]];
                if value < isize::MIN as f64 || value > isize::MAX as f64 {
                    return Err(TbError::InvalidSupercellMatrix);
                }
                integer_u[row][column] = value as isize;
            }
        }
        let determinant =
            checked_integer_determinant(&integer_u).ok_or(TbError::InvalidSupercellMatrix)?;
        if determinant <= 0 {
            return Err(TbError::InvalidSupercellDet {
                det: determinant as f64,
            });
        }
        let cell_count =
            usize::try_from(determinant).map_err(|_| TbError::InvalidSupercellMatrix)?;
        let expected_norb = self
            .norb()
            .checked_mul(cell_count)
            .ok_or(TbError::InvalidSupercellMatrix)?;
        let expected_natom = self
            .natom()
            .checked_mul(cell_count)
            .ok_or(TbError::InvalidSupercellMatrix)?;
        let nsta = if SPIN {
            expected_norb
                .checked_mul(2)
                .ok_or(TbError::InvalidSupercellMatrix)?
        } else {
            expected_norb
        };

        let adjugate =
            checked_integer_adjugate(&integer_u).ok_or(TbError::InvalidSupercellMatrix)?;
        // Defend the exact quotient construction against arithmetic mistakes:
        // U * adj(U) must equal det(U) I.
        for row in 0..DIM {
            let product = checked_integer_row_product(&integer_u[row], &adjugate)
                .ok_or(TbError::InvalidSupercellMatrix)?;
            for column in 0..DIM {
                let expected = if row == column { determinant } else { 0 };
                if product[column] != expected {
                    return Err(TbError::InvalidSupercellMatrix);
                }
            }
        }
        let u_inverse = Array2::from_shape_fn((DIM, DIM), |(row, column)| {
            adjugate[row][column] as f64 / determinant as f64
        });
        let coset_representatives = row_coset_representatives(&adjugate, determinant)
            .ok_or(TbError::InvalidSupercellMatrix)?;
        if coset_representatives.len() != cell_count {
            return Err(TbError::InvalidSupercellMatrix);
        }
        let new_lat = rounded_u.dot(&self.lat);

        // Canonicalize the source gauge before replicating it. Atoms are
        // brought into [0,1); each owned orbital is moved to the image nearest
        // its parent componentwise in fractional coordinates, not by Cartesian
        // distance in a skew lattice. The integer n_s is retained so every old
        // block is relabeled as R -> R + n_j - n_i.
        let mut normalized_atom_positions = self.atom_position();
        for mut position in normalized_atom_positions.outer_iter_mut() {
            for component in &mut position {
                *component -= component.floor();
            }
        }
        let mut normalized_orb = self.orb.clone();
        let mut orbital_gauge_shift = vec![vec![0_isize; DIM]; self.norb()];
        for (orbital, owner) in orbital_owners.iter().enumerate() {
            for axis in 0..DIM {
                let shift = match owner {
                    Some(atom) => (normalized_orb[[orbital, axis]]
                        - normalized_atom_positions[[atom.index(), axis]])
                    .round(),
                    None => normalized_orb[[orbital, axis]].floor(),
                };
                if !shift.is_finite() || shift < isize::MIN as f64 || shift > isize::MAX as f64 {
                    return Err(TbError::InvalidSupercellMatrix);
                }
                orbital_gauge_shift[orbital][axis] = shift as isize;
                normalized_orb[[orbital, axis]] -= shift;
            }
        }

        let mut new_orb = Array2::<f64>::zeros((0, DIM));
        let mut new_orb_projection = Vec::with_capacity(expected_norb);
        let mut new_atoms = Vec::with_capacity(expected_natom);
        let mut source_orbital = Vec::with_capacity(expected_norb);
        let mut primitive_label = Vec::<Vec<isize>>::with_capacity(expected_norb);
        let mut copies_by_source = vec![Vec::<usize>::with_capacity(cell_count); self.norb()];
        let mut copy_by_source_and_coset =
            vec![HashMap::<Vec<isize>, usize>::with_capacity(cell_count); self.norb()];

        if self.atoms.is_empty() {
            for source in 0..self.norb() {
                for representative in &coset_representatives {
                    let (position, label) = fold_supercell_copy(
                        normalized_orb.row(source),
                        representative,
                        &u_inverse,
                        &integer_u,
                    )?;
                    let copy = new_orb.nrows();
                    new_orb.push_row(position.view())?;
                    new_orb_projection.push(self.orb_projection[source]);
                    source_orbital.push(source);
                    primitive_label.push(label);
                    copies_by_source[source].push(copy);
                    let key = row_coset_key(representative, &adjugate, determinant)
                        .ok_or(TbError::InvalidSupercellMatrix)?;
                    if copy_by_source_and_coset[source].insert(key, copy).is_some() {
                        return Err(TbError::InvalidSupercellMatrix);
                    }
                }
            }
        } else {
            for (atom_index, atom) in self.atoms.iter().enumerate() {
                for representative in &coset_representatives {
                    let (atom_position, _) = fold_supercell_copy(
                        normalized_atom_positions.row(atom_index),
                        representative,
                        &u_inverse,
                        &integer_u,
                    )?;
                    let first_orbital = new_orb.nrows();
                    for &orbital_id in atom.orbitals() {
                        let source = orbital_id.index();
                        let (position, label) = fold_supercell_copy(
                            normalized_orb.row(source),
                            representative,
                            &u_inverse,
                            &integer_u,
                        )?;
                        let copy = new_orb.nrows();
                        new_orb.push_row(position.view())?;
                        new_orb_projection.push(self.orb_projection[source]);
                        source_orbital.push(source);
                        primitive_label.push(label);
                        copies_by_source[source].push(copy);
                        let key = row_coset_key(representative, &adjugate, determinant)
                            .ok_or(TbError::InvalidSupercellMatrix)?;
                        if copy_by_source_and_coset[source].insert(key, copy).is_some() {
                            return Err(TbError::InvalidSupercellMatrix);
                        }
                    }
                    let mut new_atom = Atom::with_orbitals(
                        atom_position,
                        atom.atom_type(),
                        (first_orbital..new_orb.nrows()).map(OrbitalId::new),
                    );
                    if let Some(moment) = atom.magnetic_moment() {
                        new_atom.set_magnetic_moment(moment)?;
                    }
                    new_atoms.push(new_atom);
                }
            }
        }

        if new_orb.nrows() != expected_norb
            || new_atoms.len() != expected_natom
            || copies_by_source
                .iter()
                .any(|copies| copies.len() != cell_count)
            || copy_by_source_and_coset
                .iter()
                .any(|copies| copies.len() != cell_count)
        {
            return Err(TbError::Other(
                "make_supercell: exact row-coset enumeration produced an incomplete image set"
                    .to_string(),
            ));
        }

        // Map each existing primitive block forward. Integer labels identify
        // the unique target image and give R_new exactly, so no shear-dependent
        // candidate-R box or floating-point reverse lookup is needed.
        let mut new_ham_r = Array2::<isize>::zeros((1, DIM));
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
        let mut new_rmatrix = if R::HAS_RMATRIX {
            Array4::<Complex<f64>>::zeros((1, DIM, nsta, nsta))
        } else {
            Array4::zeros((0, 0, 0, 0))
        };
        let mut block_by_vector = HashMap::<Vec<isize>, usize>::new();
        block_by_vector.insert(vec![0_isize; DIM], 0);
        let old_rmatrix = if R::HAS_RMATRIX {
            Some(self.rmatrix.as_array4())
        } else {
            None
        };
        let spin_components = if SPIN { 2 } else { 1 };

        for (old_block, old_r) in self.hamR.outer_iter().enumerate() {
            for old_source in 0..self.norb() {
                for old_target in 0..self.norb() {
                    let mut has_data = false;
                    for source_spin in 0..spin_components {
                        for target_spin in 0..spin_components {
                            let old_i = old_source + source_spin * self.norb();
                            let old_j = old_target + target_spin * self.norb();
                            has_data |= self.ham[[old_block, old_i, old_j]].norm_sqr() != 0.0;
                            if let Some(rmatrix) = old_rmatrix {
                                for axis in 0..DIM {
                                    has_data |=
                                        rmatrix[[old_block, axis, old_i, old_j]].norm_sqr() != 0.0;
                                }
                            }
                        }
                    }
                    if !has_data {
                        continue;
                    }

                    let normalized_r = (0..DIM)
                        .map(|axis| {
                            (old_r[axis] as isize)
                                .checked_add(orbital_gauge_shift[old_target][axis])
                                .and_then(|value| {
                                    value.checked_sub(orbital_gauge_shift[old_source][axis])
                                })
                                .ok_or(TbError::InvalidSupercellMatrix)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    for &new_source in &copies_by_source[old_source] {
                        let target_class = primitive_label[new_source]
                            .iter()
                            .zip(&normalized_r)
                            .map(|(&label, &r)| {
                                label.checked_add(r).ok_or(TbError::InvalidSupercellMatrix)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let target_key = row_coset_key(&target_class, &adjugate, determinant)
                            .ok_or(TbError::InvalidSupercellMatrix)?;
                        let &new_target = copy_by_source_and_coset[old_target]
                            .get(&target_key)
                            .ok_or_else(|| {
                                TbError::Other(
                                    "make_supercell: target image is absent from its row coset"
                                        .to_string(),
                                )
                            })?;
                        let primitive_delta = target_class
                            .iter()
                            .zip(&primitive_label[new_target])
                            .map(|(&target, &representative)| {
                                target
                                    .checked_sub(representative)
                                    .ok_or(TbError::InvalidSupercellMatrix)
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let new_r =
                            supercell_lattice_vector(&primitive_delta, &adjugate, determinant)?;
                        let new_block = match block_by_vector.get(&new_r) {
                            Some(&block) => block,
                            None => {
                                let block = new_ham_r.nrows();
                                let row = Array1::from_vec(new_r.clone());
                                new_ham_r.push_row(row.view())?;
                                new_ham.push(
                                    Axis(0),
                                    Array2::<Complex<f64>>::zeros((nsta, nsta)).view(),
                                )?;
                                if R::HAS_RMATRIX {
                                    new_rmatrix.push(
                                        Axis(0),
                                        Array3::<Complex<f64>>::zeros((DIM, nsta, nsta)).view(),
                                    )?;
                                }
                                block_by_vector.insert(new_r, block);
                                block
                            }
                        };

                        for source_spin in 0..spin_components {
                            for target_spin in 0..spin_components {
                                let old_i = old_source + source_spin * self.norb();
                                let old_j = old_target + target_spin * self.norb();
                                let new_i = new_source + source_spin * expected_norb;
                                let new_j = new_target + target_spin * expected_norb;
                                new_ham[[new_block, new_i, new_j]] +=
                                    self.ham[[old_block, old_i, old_j]];
                                if let Some(rmatrix) = old_rmatrix {
                                    for axis in 0..DIM {
                                        new_rmatrix[[new_block, axis, new_i, new_j]] +=
                                            rmatrix[[old_block, axis, old_i, old_j]];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if R::HAS_RMATRIX {
            let old_diagonal = rmatrix_diagonal_cartesian::<SPIN, DIM, R>(self);
            set_rmatrix_diagonal_with_displacement::<DIM>(
                &mut new_rmatrix,
                &new_ham_r,
                &new_orb,
                &new_lat,
                &self.orb,
                &self.lat,
                &old_diagonal,
                &source_orbital,
                SPIN,
            );
        }
        let model = Model {
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_projection,
            atoms: new_atoms,
            ham: new_ham,
            hamR: new_ham_r,
            rmatrix: R::from_array(new_rmatrix),
        };
        model.validate()?;
        Ok(model)
    }
}

/// Exact determinant of a square integer matrix using fraction-free Gaussian
/// elimination (Bareiss). Checked arithmetic turns an unrepresentable
/// transformation into an error instead of wrapping an integer cell label.
fn checked_integer_determinant(matrix: &[Vec<isize>]) -> Option<isize> {
    let n = matrix.len();
    if matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    if n == 0 {
        return Some(1);
    }
    if n == 1 {
        return Some(matrix[0][0]);
    }

    let mut work = matrix.to_vec();
    let mut sign = 1_isize;
    let mut previous_pivot = 1_isize;
    for pivot_index in 0..n - 1 {
        let Some(pivot_row) = (pivot_index..n).find(|&row| work[row][pivot_index] != 0) else {
            return Some(0);
        };
        if pivot_row != pivot_index {
            work.swap(pivot_row, pivot_index);
            sign = sign.checked_neg()?;
        }
        let pivot = work[pivot_index][pivot_index];
        for row in pivot_index + 1..n {
            for column in pivot_index + 1..n {
                let diagonal = work[row][column].checked_mul(pivot)?;
                let cross = work[row][pivot_index].checked_mul(work[pivot_index][column])?;
                let numerator = diagonal.checked_sub(cross)?;
                if pivot_index > 0 && numerator % previous_pivot != 0 {
                    return None;
                }
                work[row][column] = if pivot_index == 0 {
                    numerator
                } else {
                    numerator / previous_pivot
                };
            }
            work[row][pivot_index] = 0;
        }
        previous_pivot = pivot;
    }
    sign.checked_mul(work[n - 1][n - 1])
}

/// Exact adjugate satisfying `U * adj(U) = det(U) * I`.
fn checked_integer_adjugate(matrix: &[Vec<isize>]) -> Option<Vec<Vec<isize>>> {
    let n = matrix.len();
    if matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    if n == 0 {
        return Some(Vec::new());
    }
    if n == 1 {
        return Some(vec![vec![1]]);
    }

    let mut adjugate = vec![vec![0_isize; n]; n];
    for adjugate_row in 0..n {
        for adjugate_column in 0..n {
            // adj(U)[i,j] is the cofactor C[j,i].
            let removed_row = adjugate_column;
            let removed_column = adjugate_row;
            let minor = matrix
                .iter()
                .enumerate()
                .filter(|(row, _)| *row != removed_row)
                .map(|(_, row)| {
                    row.iter()
                        .enumerate()
                        .filter_map(|(column, &value)| (column != removed_column).then_some(value))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let cofactor = checked_integer_determinant(&minor)?;
            adjugate[adjugate_row][adjugate_column] = if (removed_row + removed_column) % 2 == 0 {
                cofactor
            } else {
                cofactor.checked_neg()?
            };
        }
    }
    Some(adjugate)
}

fn checked_integer_row_product(row: &[isize], matrix: &[Vec<isize>]) -> Option<Vec<isize>> {
    if row.len() != matrix.len()
        || matrix
            .iter()
            .any(|matrix_row| matrix_row.len() != row.len())
    {
        return None;
    }
    (0..row.len())
        .map(|column| {
            row.iter()
                .zip(matrix)
                .try_fold(0_isize, |sum, (&value, matrix_row)| {
                    sum.checked_add(value.checked_mul(matrix_row[column])?)
                })
        })
        .collect()
}

/// Canonical key for the row-vector quotient `Z^d / (Z^d U)`.
///
/// Two primitive-cell labels `a` and `b` represent the same supercell image
/// exactly when `(a-b) * adj(U)` vanishes modulo `det(U)`.
fn row_coset_key(row: &[isize], adjugate: &[Vec<isize>], determinant: isize) -> Option<Vec<isize>> {
    (determinant > 0).then_some(())?;
    Some(
        checked_integer_row_product(row, adjugate)?
            .into_iter()
            .map(|value| value.rem_euclid(determinant))
            .collect(),
    )
}

/// Enumerate exactly `det(U)` representatives of the row-vector quotient.
/// Adding the positive coordinate generators is sufficient because the
/// quotient is finite; the canonical adjugate key prevents duplicates.
fn row_coset_representatives(
    adjugate: &[Vec<isize>],
    determinant: isize,
) -> Option<Vec<Vec<isize>>> {
    let count = usize::try_from(determinant).ok()?;
    let dim = adjugate.len();
    let zero = vec![0_isize; dim];
    let zero_key = row_coset_key(&zero, adjugate, determinant)?;
    let mut representatives = vec![zero.clone()];
    let mut seen = HashMap::<Vec<isize>, usize>::from([(zero_key, 0)]);
    let mut frontier = VecDeque::from([zero]);

    while representatives.len() < count {
        let representative = frontier.pop_front()?;
        for axis in 0..dim {
            let mut candidate = representative.clone();
            candidate[axis] = candidate[axis].checked_add(1)?;
            let key = row_coset_key(&candidate, adjugate, determinant)?;
            if seen.contains_key(&key) {
                continue;
            }
            let index = representatives.len();
            seen.insert(key, index);
            representatives.push(candidate.clone());
            frontier.push_back(candidate);
            if representatives.len() == count {
                break;
            }
        }
    }
    Some(representatives)
}

/// Fold `(tau + representative) * U^-1` into the supercell and return both
/// its fractional representative and the exact primitive-cell label `ell`
/// satisfying `fractional * U = tau + ell`.
fn fold_supercell_copy(
    tau: ArrayView1<'_, f64>,
    representative: &[isize],
    u_inverse: &Array2<f64>,
    u_integer: &[Vec<isize>],
) -> Result<(Array1<f64>, Vec<isize>)> {
    const SNAP_TOLERANCE: f64 = 1e-10;
    let dim = tau.len();
    if representative.len() != dim
        || u_inverse.dim() != (dim, dim)
        || u_integer.len() != dim
        || u_integer.iter().any(|row| row.len() != dim)
    {
        return Err(TbError::InvalidSupercellMatrix);
    }

    let translated = Array1::from_iter(
        tau.iter()
            .zip(representative)
            .map(|(&position, &shift)| position + shift as f64),
    );
    let raw = translated.dot(u_inverse);
    let mut fractional = Array1::<f64>::zeros(dim);
    let mut supercell_shift = vec![0_isize; dim];
    for axis in 0..dim {
        let value = raw[axis];
        if !value.is_finite() {
            return Err(TbError::InvalidSupercellMatrix);
        }
        let nearest = value.round();
        let (cell, position) = if (value - nearest).abs() < SNAP_TOLERANCE {
            (nearest, 0.0)
        } else {
            let floor = value.floor();
            (floor, value - floor)
        };
        if cell < isize::MIN as f64 || cell > isize::MAX as f64 {
            return Err(TbError::InvalidSupercellMatrix);
        }
        fractional[axis] = position;
        supercell_shift[axis] = cell as isize;
    }

    let shift_in_primitive = checked_integer_row_product(&supercell_shift, u_integer)
        .ok_or(TbError::InvalidSupercellMatrix)?;
    let primitive_label = representative
        .iter()
        .zip(shift_in_primitive)
        .map(|(&representative, shift)| {
            representative
                .checked_sub(shift)
                .ok_or(TbError::InvalidSupercellMatrix)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((fractional, primitive_label))
}

/// Solve the exact row equation `new_r * U = primitive_delta`.
fn supercell_lattice_vector(
    primitive_delta: &[isize],
    adjugate: &[Vec<isize>],
    determinant: isize,
) -> Result<Vec<isize>> {
    let numerator = checked_integer_row_product(primitive_delta, adjugate)
        .ok_or(TbError::InvalidSupercellMatrix)?;
    numerator
        .into_iter()
        .map(|value| {
            if value % determinant != 0 {
                return Err(TbError::InvalidSupercellMatrix);
            }
            isize::try_from(value / determinant).map_err(|_| TbError::InvalidSupercellMatrix)
        })
        .collect()
}

/// Extract the `R = 0` position-matrix diagonal (Cartesian, shape
/// `(nsta, DIM)`); all zeros for `NoRMatrix` models.
///
/// The diagonal is stored per STATE so spin-dependent position diagonals
/// (e.g. distinct r↑↑ and r↓↓ offsets) survive identity transforms.
pub(crate) fn rmatrix_diagonal_cartesian<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
) -> Array2<f64> {
    let nsta = model.nsta();
    let mut diagonal = Array2::<f64>::zeros((nsta, DIM));
    if !R::HAS_RMATRIX {
        return diagonal;
    }
    let zero_r = Array1::<isize>::zeros(DIM);
    let Some(r0) = find_R(&model.hamR, &zero_r) else {
        return diagonal;
    };
    let rmatrix = model.rmatrix.as_array4();
    for s in 0..nsta {
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
    let norb_old = old_orb.nrows();
    let new_cart = new_orb.dot(new_lat);
    let old_cart = old_orb.dot(old_lat);
    for i in 0..nsta {
        let s = if spin { i % norb } else { i };
        let src_orb = source[s];
        // The source STATE preserves the spin component of state i
        // (up block before down block in both models).
        let src_state = if spin {
            if i < norb {
                src_orb
            } else {
                src_orb + norb_old
            }
        } else {
            src_orb
        };
        for axis in 0..DIM {
            rmatrix[[r0, axis, i, i]] = Complex::new(
                old_diagonal[[src_state, axis]] + (new_cart[[s, axis]] - old_cart[[src_orb, axis]]),
                0.0,
            );
        }
    }
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
                        ham_r
                            .push_row(new_r.view())
                            .expect("ham_r row must match DIM");
                        ham.push(Axis(0), Array2::<Complex<f64>>::zeros((nsta, nsta)).view())
                            .expect("ham block shape must match (nsta, nsta)");
                        rmatrix
                            .push(
                                Axis(0),
                                Array3::<Complex<f64>>::zeros((DIM, nsta, nsta)).view(),
                            )
                            .expect("rmatrix block shape must match (DIM, nsta, nsta)");
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
/// 2. Fold each owned orbital to the image nearest its parent atom
///    componentwise in fractional coordinates (not necessarily in Cartesian
///    distance). This image may lie outside `[0, 1)`, e.g. `atom = 0.99`
///    keeps `orb = 1.01`; unowned orbitals are folded into `[0, 1)`.
/// 3. Relabel Hamiltonian and position-matrix blocks by the orbital folds,
///    then translate the position-matrix diagonal by the Cartesian orbital
///    displacement, preserving its original offset from `τ · L`.
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
    use crate::solve_ham::Solve;
    use crate::{Atom, AtomType, Gauge, HasRMatrix, OrbitalId};
    use std::collections::HashSet;

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
    fn hopping_updates_are_independent_of_origin_row() {
        let mut model =
            Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
        model.add_hop(0.2, 0, 0, &array![1], None);
        let order = [1, 0, 2];
        model.ham = model.ham.select(Axis(0), &order);
        model.hamR = model.hamR.select(Axis(0), &order);
        model.set_onsite(&array![0.5, 0.25], None);
        model.add_onsite(&array![1.0, 1.0], None);
        assert_eq!(model.ham[[1, 0, 0]], Complex::new(1.5, 0.0));
        assert_eq!(model.ham[[1, 1, 1]], Complex::new(1.25, 0.0));

        // A nonzero translation in row zero still needs its conjugate.
        let value = Complex::new(0.3, 0.4);
        model.set_hop(value, 0, 0, &array![1], None);
        assert_eq!(model.ham[[2, 0, 0]], value.conj());
        model.add_hop(value, 0, 0, &array![1], None);
        assert_eq!(model.ham[[2, 0, 0]], 2.0 * value.conj());
        model.add_element(value, 0, 0, &array![1]).unwrap();
        assert_eq!(model.ham[[2, 0, 0]], value.conj());

        // Off-diagonal intracell terms must continue to update both entries.
        model.add_hop(value, 0, 1, &array![0], None);
        assert_eq!(model.ham[[1, 0, 1]], value);
        assert_eq!(model.ham[[1, 1, 0]], value.conj());
    }

    #[test]
    fn exact_row_cosets_follow_the_model_row_vector_convention() {
        // For row vectors, Z^2/(Z^2 U) with U=[[2,1],[0,1]] is classified by
        // the parity of the first coordinate. A column-lattice HNF computes a
        // different quotient and the former rounded Euclidean loop did not
        // terminate for this matrix.
        let u = vec![vec![2_isize, 1_isize], vec![0_isize, 1_isize]];
        let determinant = checked_integer_determinant(&u).unwrap();
        let adjugate = checked_integer_adjugate(&u).unwrap();
        assert_eq!(determinant, 2);
        assert_eq!(
            checked_integer_row_product(&u[0], &adjugate).unwrap(),
            vec![determinant, 0]
        );
        let representatives = row_coset_representatives(&adjugate, determinant).unwrap();
        assert_eq!(representatives.len(), 2);
        let keys = representatives
            .iter()
            .map(|representative| row_coset_key(representative, &adjugate, determinant).unwrap())
            .collect::<Vec<_>>();
        assert_ne!(keys[0], keys[1]);
        assert_eq!(
            row_coset_key(&[2, 1], &adjugate, determinant).unwrap(),
            keys[0],
            "a row of U must be equivalent to zero"
        );
    }

    #[test]
    fn exact_row_cosets_exhaust_small_two_dimensional_matrices() {
        for a in -3_isize..=3 {
            for b in -3_isize..=3 {
                for c in -3_isize..=3 {
                    for d in -3_isize..=3 {
                        let expected_determinant = a * d - b * c;
                        if !(1..=12).contains(&expected_determinant) {
                            continue;
                        }
                        let u = vec![vec![a, b], vec![c, d]];
                        let determinant = checked_integer_determinant(&u).unwrap();
                        let adjugate = checked_integer_adjugate(&u).unwrap();
                        assert_eq!(determinant, expected_determinant);
                        let representatives =
                            row_coset_representatives(&adjugate, determinant).unwrap();
                        let keys = representatives
                            .iter()
                            .map(|representative| {
                                row_coset_key(representative, &adjugate, determinant).unwrap()
                            })
                            .collect::<HashSet<_>>();
                        assert_eq!(keys.len(), determinant as usize, "U={u:?}");
                        let zero_key = row_coset_key(&[0, 0], &adjugate, determinant).unwrap();
                        assert_eq!(
                            row_coset_key(&u[0], &adjugate, determinant).unwrap(),
                            zero_key,
                            "first row of U must be the zero coset: U={u:?}"
                        );
                        assert_eq!(
                            row_coset_key(&u[1], &adjugate, determinant).unwrap(),
                            zero_key,
                            "second row of U must be the zero coset: U={u:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn supercell_off_diagonal_pivot_terminates_with_distinct_images() {
        // The former rounded HNF loop stalled forever because round(1/3)=0.
        let atom = Atom::with_orbitals(array![0.5, 0.5], AtomType::C, [OrbitalId::new(0)]);
        let model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.5, 0.5]], Some(vec![atom]))
                .unwrap();
        let supercell = model
            .make_supercell(&array![[3.0, 1.0], [0.0, 1.0]])
            .unwrap();
        let expected = [[1.0 / 6.0, 1.0 / 3.0], [0.5, 0.0], [5.0 / 6.0, 2.0 / 3.0]];
        assert_eq!(supercell.norb(), expected.len());
        for (position, expected) in supercell.orb.outer_iter().zip(expected) {
            for axis in 0..2 {
                assert!((position[axis] - expected[axis]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn supercell_uses_row_quotient_for_nonsymmetric_transform() {
        // The old column quotient returned six copies but only four distinct
        // positions for this matrix, silently omitting two physical images.
        let atom = Atom::with_orbitals(array![0.25, 0.25], AtomType::C, [OrbitalId::new(0)]);
        let model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.25, 0.25]], Some(vec![atom]))
                .unwrap();
        let supercell = model
            .make_supercell(&array![[2.0, 4.0], [-1.0, 1.0]])
            .unwrap();
        let expected = [
            [1.0 / 12.0, 11.0 / 12.0],
            [0.25, 0.25],
            [5.0 / 12.0, 7.0 / 12.0],
            [7.0 / 12.0, 11.0 / 12.0],
            [0.75, 0.25],
            [11.0 / 12.0, 7.0 / 12.0],
        ];
        assert_eq!(supercell.norb(), expected.len());
        for (position, expected) in supercell.orb.outer_iter().zip(expected) {
            for axis in 0..2 {
                assert!((position[axis] - expected[axis]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn skew_supercell_spectrum_matches_primitive_band_folding() {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.13, 0.27]], None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-0.4, 0, 0, &array![0, 1], None);
        let supercell = model
            .make_supercell(&array![[2.0, 1.0], [0.0, 1.0]])
            .unwrap();

        let k_supercell = array![0.23, 0.37];
        let u_inverse_transpose = array![[0.5, 0.0], [-0.5, 1.0]];
        let mut expected = [[0.0, 0.0], [1.0, 0.0]]
            .into_iter()
            .map(|reciprocal_image| {
                let reciprocal_image = Array1::from_vec(reciprocal_image.to_vec());
                let primitive_k = (&k_supercell + &reciprocal_image).dot(&u_inverse_transpose);
                model.solve_band_onek(&primitive_k).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let mut actual = supercell.solve_band_onek(&k_supercell).unwrap().to_vec();
        expected.sort_by(f64::total_cmp);
        actual.sort_by(f64::total_cmp);
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn skew_supercell_preserves_spin_dependent_cartesian_rmatrix_offsets() {
        let lattice = array![[2.0, 0.4], [0.0, 1.5]];
        let mut model =
            Model::<true, 2, HasRMatrix>::tb_model(lattice, array![[0.2, 0.3]], None).unwrap();
        let old_cartesian = model.orb.dot(&model.lat);
        let up = [old_cartesian[[0, 0]] + 0.1, old_cartesian[[0, 1]] - 0.2];
        let down = [old_cartesian[[0, 0]] - 0.3, old_cartesian[[0, 1]] + 0.4];
        for axis in 0..2 {
            model.rmatrix.as_array4_mut()[[0, axis, 0, 0]] = Complex::new(up[axis], 0.0);
            model.rmatrix.as_array4_mut()[[0, axis, 1, 1]] = Complex::new(down[axis], 0.0);
        }

        let supercell = model
            .make_supercell(&array![[2.0, 1.0], [0.0, 1.0]])
            .unwrap();
        let new_cartesian = supercell.orb.dot(&supercell.lat);
        for orbital in 0..supercell.norb() {
            for axis in 0..2 {
                let displacement = new_cartesian[[orbital, axis]] - old_cartesian[[0, axis]];
                assert!(
                    (supercell.rmatrix.as_array4()[[0, axis, orbital, orbital]].re
                        - (up[axis] + displacement))
                        .abs()
                        < 1e-12
                );
                let down_state = orbital + supercell.norb();
                assert!(
                    (supercell.rmatrix.as_array4()[[0, axis, down_state, down_state]].re
                        - (down[axis] + displacement))
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn nonsymmetric_supercell_preserves_empty_atoms_and_3d_images() {
        let atoms = vec![
            Atom::with_orbitals(array![0.2, 0.3, 0.4], AtomType::C, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.6, 0.7, 0.8], AtomType::H, []),
        ];
        let model =
            Model::<false, 3>::tb_model(Array2::eye(3), array![[0.2, 0.3, 0.4]], Some(atoms))
                .unwrap();
        let supercell = model
            .make_supercell(&array![[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .unwrap();
        assert_eq!(supercell.norb(), 2);
        assert_eq!(supercell.natom(), 4);
        assert_eq!(
            supercell
                .atoms
                .iter()
                .filter(|atom| atom.atom_type() == AtomType::H && atom.norb() == 0)
                .count(),
            2
        );
        assert_ne!(supercell.orb.row(0), supercell.orb.row(1));
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
        let band_sc = sc_a.solve_band_onek(&array![k_sc]).unwrap();
        let e_prim_1 = model_a.solve_band_onek(&array![k_sc / 2.0]).unwrap()[0];
        let e_prim_2 = model_a.solve_band_onek(&array![k_sc / 2.0 + 0.5]).unwrap()[0];
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
        let mut expected: Vec<f64> = model.solve_band_onek(&k).unwrap().to_vec();
        let mut got: Vec<f64> = sc.solve_band_onek(&k).unwrap().to_vec();
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
    fn supercell_rejects_negative_non_integer_entries() {
        // Regression: fract(-0.5) = -0.5 passed the old `fract() > 1e-8`
        // test, so [[1, -0.5], [0, 1]] was accepted as a supercell matrix.
        let model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.5, 0.5]], None).unwrap();
        let result = model.make_supercell(&array![[1.0, -0.5], [0.0, 1.0]]);
        assert!(matches!(result, Err(TbError::InvalidSupercellMatrix)));
    }

    #[test]
    fn skew_supercell_preserves_hoppings_beyond_small_r_range() {
        // Regression: the R-vector candidate range did not include the
        // actual image shifts, so a skew basis change silently dropped
        // hoppings (H_01(0) = 0.7 became 0 at k = 0 for
        // U = [[1, 100], [0, 1]]).
        let atoms = vec![
            Atom::with_orbitals(array![0.3, 0.3], AtomType::C, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.5, 0.5], AtomType::O, [OrbitalId::new(1)]),
        ];
        let mut model = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.3, 0.3], [0.5, 0.5]],
            Some(atoms),
        )
        .unwrap();
        model
            .add_element(Complex::new(0.7, 0.0), 0, 1, &array![0, 0])
            .unwrap();

        let sc = model
            .make_supercell(&array![[1.0, 100.0], [0.0, 1.0]])
            .unwrap();
        sc.validate().unwrap();
        // det = 1: one copy per orbital; the onsite hopping H_01(0) must
        // survive the transform.
        assert_eq!(sc.norb(), 2);
        let k = array![0.0, 0.0];
        let ham = sc.gen_ham(&k, Gauge::Lattice);
        let mut found = false;
        for i in 0..sc.nsta() {
            for j in 0..sc.nsta() {
                if i != j && (ham[[i, j]] - Complex::new(0.7, 0.0)).norm() < 1e-10 {
                    found = true;
                }
            }
        }
        assert!(
            found,
            "onsite hopping 0.7 must survive the skew basis change"
        );
    }

    #[test]
    fn nested_shear_3d_supercell_succeeds() {
        // Regression: U = [[1, 10, 100], [0, 1, 10], [0, 0, 1]] needs image
        // coefficients up to ~55, far beyond any fixed coefficient box.
        let atoms = vec![Atom::with_orbitals(
            array![0.5, 0.5, 0.5],
            AtomType::C,
            [OrbitalId::new(0)],
        )];
        let mut model =
            Model::<false, 3>::tb_model(Array2::eye(3), array![[0.5, 0.5, 0.5]], Some(atoms))
                .unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0, 0], None);

        let sc = model
            .make_supercell(&array![
                [1.0, 10.0, 100.0],
                [0.0, 1.0, 10.0],
                [0.0, 0.0, 1.0]
            ])
            .unwrap();
        sc.validate().unwrap();
        assert_eq!(sc.norb(), 1);
        assert_eq!(sc.natom(), 1);
    }

    #[test]
    fn supercell_preserves_empty_orbital_atoms() {
        // Regression: the copy-count check only counted orbitals, so an
        // atom owning no orbitals was silently dropped (natom 2 -> 1).
        let atoms = vec![
            Atom::with_orbitals(array![0.5, 0.5], AtomType::C, [OrbitalId::new(0)]),
            Atom::new(array![0.7, 0.7], AtomType::O), // no orbitals
        ];
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.5, 0.5]], Some(atoms)).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);

        let sc = model
            .make_supercell(&array![[2.0, 0.0], [0.0, 1.0]])
            .unwrap();
        sc.validate().unwrap();
        assert_eq!(sc.natom(), 4, "both atoms must keep det(U) images each");
        assert_eq!(sc.norb(), 2);
    }

    #[test]
    fn skew_integer_basis_change_supercell_succeeds() {
        // Regression: the image-shift enumeration was bounded by det(U)
        // per coefficient; U = [[1, 100], [0, 1]] with det = 1 needs the
        // shift (0, 50) to bring atom (0.5, 0.5) into the cell, so the old
        // range returned Err(NoOrbitals).
        let atoms = vec![Atom::with_orbitals(
            array![0.5, 0.5],
            AtomType::C,
            [OrbitalId::new(0)],
        )];
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.5, 0.5]], Some(atoms)).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-1.0, 0, 0, &array![0, 1], None);

        let sc = model
            .make_supercell(&array![[1.0, 100.0], [0.0, 1.0]])
            .unwrap();
        sc.validate().unwrap();
        // det = 1: exactly one copy of the source orbital.
        assert_eq!(sc.norb(), 1);
        assert_eq!(sc.natom(), 1);
        for s in 0..sc.norb() {
            for axis in 0..2 {
                assert!(
                    (0.0..1.0).contains(&sc.orb[[s, axis]]),
                    "supercell orbital {s} axis {axis} = {} outside [0, 1)",
                    sc.orb[[s, axis]]
                );
            }
        }
    }

    #[test]
    fn identity_supercell_preserves_spin_split_rmatrix_diagonal() {
        // Regression: the old diagonal was stored per orbital and written
        // to both spin states via i % norb, scrambling distinct r↑↑/r↓↓
        // offsets into (r↑↑, r↑↑) under the identity supercell.
        let mut model =
            Model::<true, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.5]], None).unwrap();
        model.rmatrix.as_array4_mut()[[0, 0, 0, 0]] = Complex::new(0.7, 0.0);
        model.rmatrix.as_array4_mut()[[0, 0, 1, 1]] = Complex::new(0.9, 0.0);
        model.add_hop(-1.0, 0, 0, &array![1], None);

        let sc = model.make_supercell(&array![[1.0]]).unwrap();
        sc.validate().unwrap();
        let rmatrix = sc.rmatrix.as_array4();
        let zero_r = Array1::<isize>::zeros(1);
        let r0 = find_R(&sc.hamR, &zero_r).unwrap();
        assert!(
            (rmatrix[[r0, 0, 0, 0]] - Complex::new(0.7, 0.0)).norm() < 1e-12,
            "r↑↑ must stay 0.7, found {}",
            rmatrix[[r0, 0, 0, 0]]
        );
        assert!(
            (rmatrix[[r0, 0, 1, 1]] - Complex::new(0.9, 0.0)).norm() < 1e-12,
            "r↓↓ must stay 0.9, found {}",
            rmatrix[[r0, 0, 1, 1]]
        );
    }

    #[test]
    fn identity_supercell_preserves_custom_rmatrix_diagonal() {
        // Regression: the diagonal was unconditionally overwritten with
        // tau·L, clobbering a legitimate custom offset from _r.dat data.
        // The identity supercell must preserve r_old(ss, 0) exactly.
        let mut model =
            Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.5]], None).unwrap();
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
