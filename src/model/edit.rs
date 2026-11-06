//! Structural editing of model arrays and atom/orbital ownership.
//!
//! Orbital removal and reordering also select or permute the Hamiltonian,
//! position-matrix state axes, projections and ownership references. Metadata-only
//! edits leave the dense basis unchanged. `shift_to_atom` and its alias
//! `move_to_atom` only overwrite owned orbital positions; they do not transform
//! Hamiltonian or position-matrix elements.

use super::{Model, RMatrixData};
use crate::atom_struct::OrbitalId;
use crate::error::{Result, TbError};
use ndarray::Axis;

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

    /// Alias for [`Self::shift_to_atom`].
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
        // Transactional pre-check: removing every orbital would leave the
        // model empty, and validate() would only report it after the arrays
        // were already mutated.
        if use_orb_list.len() == self.norb() {
            return Err(TbError::NoOrbitals);
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
        self.rmatrix = self.rmatrix.select(Axis(2), &index).select(Axis(3), &index);
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
        // Transactional pre-check: removing every orbital would leave the
        // model empty, and the error must fire before any mutation.
        if removed_orbitals.len() == self.norb() {
            return Err(TbError::NoOrbitals);
        }
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
        self.rmatrix = self
            .rmatrix
            .select(Axis(2), &new_state_order)
            .select(Axis(3), &new_state_order);
        self.validate()?;
        Ok(())
    }
}
