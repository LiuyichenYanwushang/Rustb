use super::{Model, RMatrixData};
use crate::atom_struct::{Atom, AtomId, OrbProj, OrbitalId};
use crate::error::{Result, TbError};
use ndarray::{Array2, ArrayView1};
/// Borrowed view of one physical orbital in a [`Model`].
#[derive(Debug)]
pub struct OrbitalRef<'a> {
    id: OrbitalId,
    position: ArrayView1<'a, f64>,
    projection: &'a OrbProj,
}

impl<'a> OrbitalRef<'a> {
    #[inline]
    pub const fn id(&self) -> OrbitalId {
        self.id
    }

    #[inline]
    pub fn position(&self) -> ArrayView1<'a, f64> {
        self.position
    }

    #[inline]
    pub const fn projection(&self) -> &'a OrbProj {
        self.projection
    }
}

/// Borrowed atom metadata together with borrowed views of its model orbitals.
///
/// The view is created on demand and contains no self-reference inside the
/// owning model. Holding it prevents mutable access to the model through the
/// usual Rust borrowing rules.
#[derive(Debug)]
pub struct AtomView<'a> {
    id: AtomId,
    atom: &'a Atom,
    orbitals: Vec<OrbitalRef<'a>>,
}

impl<'a> AtomView<'a> {
    #[inline]
    pub const fn id(&self) -> AtomId {
        self.id
    }

    #[inline]
    pub const fn atom(&self) -> &'a Atom {
        self.atom
    }

    #[inline]
    pub fn orbitals(&self) -> &[OrbitalRef<'a>] {
        &self.orbitals
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    #[inline(always)]
    pub fn atom_position(&self) -> Array2<f64> {
        let mut atom_position = Array2::zeros((self.natom(), DIM));
        atom_position
            .outer_iter_mut()
            .zip(self.atoms.iter())
            .for_each(|(mut atom_p, atom)| {
                atom_p.assign(atom.position_ref());
            });
        atom_position
    }

    /// Borrow one orbital by its typed model-local ID.
    pub fn orbital(&self, id: OrbitalId) -> Result<OrbitalRef<'_>> {
        let index = id.index();
        let projection = self
            .orb_projection
            .get(index)
            .ok_or(TbError::InvalidOrbitalId {
                index,
                norb: self.norb(),
            })?;
        if index >= self.orb.nrows() {
            return Err(TbError::InvalidOrbitalId {
                index,
                norb: self.norb(),
            });
        }
        Ok(OrbitalRef {
            id,
            position: self.orb.row(index),
            projection,
        })
    }

    /// Borrow an atom and all orbitals explicitly assigned to it.
    pub fn atom(&self, id: AtomId) -> Result<AtomView<'_>> {
        let index = id.index();
        let atom = self.atoms.get(index).ok_or(TbError::InvalidAtomId {
            index,
            natom: self.natom(),
        })?;
        let orbitals = atom
            .orbitals()
            .iter()
            .copied()
            .map(|orbital| self.orbital(orbital))
            .collect::<Result<Vec<_>>>()?;
        Ok(AtomView { id, atom, orbitals })
    }

    /// Derive the unique owner of each orbital.
    ///
    /// Unassigned orbitals are represented by `None`. Duplicate or out-of-range
    /// assignments are reported as structured errors.
    pub fn orbital_owners(&self) -> Result<Vec<Option<AtomId>>> {
        let mut owners: Vec<Option<AtomId>> = vec![None; self.norb()];
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            let atom_id = AtomId::new(atom_index);
            for &orbital_id in atom.orbitals() {
                let orbital = orbital_id.index();
                if orbital >= self.norb() {
                    return Err(TbError::InvalidOrbitalId {
                        index: orbital,
                        norb: self.norb(),
                    });
                }
                if let Some(previous) = owners[orbital] {
                    return Err(TbError::DuplicateOrbitalOwner {
                        orbital,
                        first_atom: previous.index(),
                        second_atom: atom_index,
                    });
                }
                owners[orbital] = Some(atom_id);
            }
        }
        Ok(owners)
    }
}
