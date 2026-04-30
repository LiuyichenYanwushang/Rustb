//! Model transformation and manipulation methods.
//!
//! This module provides [`Model`] methods for structural transformations:
//!
//! - Supercell construction via [`Model::make_supercell`].
//! - Orbital removal and reordering.
//! - Shifting orbital positions to atomic positions.
//!
//! # Examples
//!
//! ```ignore
//! // Create a 2x2 supercell
//! let U = array![[2.0, 0.0], [0.0, 2.0]];
//! let super_model = model.make_supercell(&U).unwrap();
//! ```

use crate::Model;
use crate::atom_struct::Atom;
use crate::error::{Result, TbError};
use crate::model_enums::Dimension;
use crate::model_utils::find_R;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::{Determinant, Inverse};
use num_complex::Complex;

impl<const SPIN: bool> Model<SPIN> {
    /// Shift all orbital positions onto their corresponding atomic positions.
    pub fn shift_to_atom(&mut self) {
        let mut a = 0;
        for (i, atom) in self.atoms.iter().enumerate() {
            for j in 0..atom.norb() {
                self.orb.row_mut(a).assign(&atom.position());
                a += 1;
            }
        }
    }

    /// Move all orbital positions onto their corresponding atomic positions.
    ///
    /// This is an alias for `shift_to_atom` with the same behaviour.
    pub fn move_to_atom(&mut self) {
        let mut a = 0;
        for i in 0..self.natom() {
            for j in 0..self.atoms[i].norb() {
                self.orb.row_mut(a).assign(&self.atoms[i].position());
                a += 1;
            }
        }
    }

    /// Remove specified orbitals from the model.
    ///
    /// Orbital indices are 0-based.  The orbital list is sorted and deduplicated
    /// internally.  Atoms with zero remaining orbitals are also removed.
    /// Both the Hamiltonian and `rmatrix` are updated accordingly.
    ///
    /// # Panics
    ///
    /// Panics if `orb_list` contains duplicates.
    pub fn remove_orb(&mut self, orb_list: &Vec<usize>) {
        let mut use_orb_list = orb_list.clone();
        use_orb_list.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let has_duplicates = { use_orb_list.windows(2).any(|window| window[0] == window[1]) };
        if has_duplicates {
            panic!("Wrong, make sure no duplicates in orb_list");
        }
        let mut index: Vec<_> = (0..=self.norb() - 1)
            .filter(|&num| !use_orb_list.contains(&num))
            .collect(); // indices to keep
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
        // Extend indices for spin
        if SPIN {
            let index_add: Vec<_> = index.iter().map(|x| *x + self.norb()).collect();
            index.extend(index_add);
        }
        // Update Hamiltonian
        let new_ham = self.ham.select(Axis(1), &index);
        let new_ham = new_ham.select(Axis(2), &index);
        self.ham = new_ham;
        // Update rmatrix
        let new_rmatrix = self.rmatrix.select(Axis(2), &index);
        let new_rmatrix = new_rmatrix.select(Axis(3), &index);
        self.rmatrix = new_rmatrix;
    }

    /// Remove specified atoms (and their orbitals) from the model.
    ///
    /// Atom indices are 0-based.  All orbitals belonging to the removed atoms
    /// are removed, and the Hamiltonian and rmatrix are updated.
    ///
    /// # Panics
    ///
    /// Panics if `atom_list` contains duplicates.
    pub fn remove_atom(&mut self, atom_list: &Vec<usize>) {
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
            .collect(); // atoms to keep

        let new_atoms = {
            let mut new_atoms = Vec::new();
            for i in atom_index.iter() {
                new_atoms.push(self.atoms[*i].clone());
            }
            new_atoms
        };

        // Build orbital index list for kept atoms
        let mut b = 0;
        let mut orb_index = Vec::new(); // orbitals to keep
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
        let norb = self.norb();
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
        // Update Hamiltonian
        let new_ham = self.ham.select(Axis(1), &orb_index);
        let new_ham = new_ham.select(Axis(2), &orb_index);
        self.ham = new_ham;
        // Update rmatrix
        let new_rmatrix = self.rmatrix.select(Axis(2), &orb_index);
        let new_rmatrix = new_rmatrix.select(Axis(3), &orb_index);
        self.rmatrix = new_rmatrix;
    }

    /// Reorder atoms and their orbitals.
    ///
    /// `order` gives the new permutation of atoms (0-based indices).
    /// Orbitals are reordered accordingly.  The Hamiltonian and rmatrix are
    /// updated to reflect the new ordering.
    ///
    /// # Panics
    ///
    /// Panics if `order.len() != self.natom()`.
    pub fn reorder_atom(&mut self, order: &Vec<usize>) {
        if order.len() != self.natom() {
            panic!(
                "Wrong! when you using reorder_atom, the order's length {} must equal to the num of atoms {}.",
                order.len(),
                self.natom()
            );
        };
        // Compute orbital ordering from atom ordering
        let mut new_orb_order = Vec::new();
        let mut orb_atom_map = Vec::new();
        let mut a = 0;
        for atom in self.atoms.iter() {
            orb_atom_map.push(a);
            a += atom.norb();
        }
        for i in order.iter() {
            for j in 0..self.atoms[*i].norb() {
                new_orb_order.push(orb_atom_map[*i] + j);
            }
        }
        // Reorder orbitals
        self.orb = self.orb.select(Axis(0), &new_orb_order);
        // Reorder atoms
        let mut new_atom = Vec::new();
        for i in 0..self.natom() {
            new_atom.push(self.atoms[order[i]].clone());
        }
        self.atoms = new_atom;
        // Reorder orbital projections
        let mut new_orb_proj = Vec::new();
        for i in new_orb_order.iter() {
            new_orb_proj.push(self.orb_projection[*i]);
        }
        self.orb_projection = new_orb_proj;
        // Build state ordering (accounts for spin)
        let new_state_order = if SPIN {
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
    /// The new lattice vectors are \\(L' = U L\\), where `L` is the original
    /// lattice matrix (rows are lattice vectors).  `U` must be an integer matrix
    /// with \\(\det U > 0\\).  The supercell contains \\(\det U\\) primitive cells.
    ///
    /// # Parameters
    ///
    /// - `U`: the `dim_r x dim_r` integer transformation matrix.
    ///
    /// # Returns
    ///
    /// A new `Model` with the supercell lattice, orbital positions, and
    /// Hamiltonian.
    ///
    /// # Errors
    ///
    /// - [`TbError::TransformationMatrixDimMismatch`] if `U` has the wrong
    ///   dimensions.
    /// - [`TbError::InvalidSupercellDet`] if `det(U) <= 0`.
    /// - [`TbError::InvalidSupercellMatrix`] if `U` has non-integer entries.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let U = array![[2.0, 0.0], [0.0, 2.0]];
    /// let supercell = model.make_supercell(&U).unwrap();
    /// ```
    pub fn make_supercell(&self, U: &Array2<f64>) -> Result<Model<SPIN>> {
        if self.dim_r() != U.len_of(Axis(0)) {
            return Err(TbError::TransformationMatrixDimMismatch {
                expected: self.dim_r(),
                actual: U.len_of(Axis(0)),
            });
        }
        let new_lat = U.dot(&self.lat);
        let U_det = U.det().unwrap() as isize;
        if U_det < 0 {
            return Err(TbError::InvalidSupercellDet { det: U_det as f64 });
        } else if U_det == 0 {
            return Err(TbError::InvalidSupercellDet { det: 0.0 });
        }
        let U_inv = U.inv().unwrap();
        // Check that U is integer-valued
        for i in 0..U.len_of(Axis(0)) {
            for j in 0..U.len_of(Axis(1)) {
                if U[[i, j]].fract() > 1e-8 {
                    return Err(TbError::InvalidSupercellMatrix);
                }
            }
        }

        // Map orbitals and atoms to the supercell coordinates
        let mut use_orb = self.orb.dot(&U_inv);
        let use_atom_position = self.atom_position().dot(&U_inv);
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
                                    + (k as f64) * U_inv.row(2).to_owned();
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
                                    new_atom.push(Atom::new(
                                        atoms,
                                        self.atoms[n].norb(),
                                        self.atoms[n].atom_type(),
                                    ));
                                    for n0 in
                                        use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb()
                                    {
                                        let mut orbs = use_orb.row(n0).to_owned()
                                            + (i as f64) * U_inv.row(0).to_owned()
                                            + (j as f64) * U_inv.row(1).to_owned()
                                            + (k as f64) * U_inv.row(2).to_owned();
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
                                + (j as f64) * U_inv.row(1).to_owned();
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
                                new_atom.push(Atom::new(
                                    atoms,
                                    self.atoms[n].norb(),
                                    self.atoms[n].atom_type(),
                                ));
                                for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb()
                                {
                                    let mut orbs = use_orb.row(n0).to_owned()
                                        + (i as f64) * U_inv.row(0).to_owned()
                                        + (j as f64) * U_inv.row(1).to_owned();
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
                for i in -U_det - 1..U_det + 1 {
                    for n in 0..self.natom() {
                        let mut atoms = use_atom_position.row(n).to_owned()
                            + (i as f64) * U_inv.row(0).to_owned();
                        atoms[[0]] = if atoms[[0]].abs() < 1e-8 {
                            0.0
                        } else if (atoms[[0]] - 1.0).abs() < 1e-8 {
                            1.0
                        } else {
                            atoms[[0]]
                        };
                        if atoms.iter().all(|x| *x >= 0.0 && *x < 1.0) {
                            new_atom.push(Atom::new(
                                atoms,
                                self.atoms[n].norb(),
                                self.atoms[n].atom_type(),
                            ));
                            for n0 in use_atom_list[n]..use_atom_list[n] + self.atoms[n].norb() {
                                let mut orbs = use_orb.row(n0).to_owned()
                                    + (i as f64) * U_inv.row(0).to_owned();
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

        // Build the supercell Hamiltonian
        let norb = new_orb.len_of(Axis(0));
        let nsta = if SPIN { 2 * norb } else { norb };
        let natom = new_atom.len();
        let n_R = self.hamR.len_of(Axis(0));
        let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
        let mut use_hamR = Array2::<isize>::zeros((1, self.dim_r()));
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, nsta, nsta));
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

        // Generate all possible R-vectors within the search range
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

        // Populate the supercell Hamiltonian by mapping hopping terms
        if SPIN && gen_rmatrix {
            for (R, use_R) in use_hamR.outer_iter().enumerate() {
                let mut add_R: bool = false;
                let mut useham = Array2::<Complex<f64>>::zeros((nsta, nsta));
                let mut use_rmatrix = Array3::<Complex<f64>>::zeros((self.dim_r(), nsta, nsta));
                for (int_i, use_i) in orb_list.iter().enumerate() {
                    for (int_j, use_j) in orb_list.iter().enumerate() {
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64);
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
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64);
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
                        let R0: Array1<f64> =
                            &new_orb.row(int_j) - &new_orb.row(int_i) + &use_R.map(|x| *x as f64);

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
                        let R0: Array1<f64> = new_orb.row(int_j).to_owned()
                            - new_orb.row(int_i).to_owned()
                            + use_R.mapv(|x| x as f64);
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
