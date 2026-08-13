//! Core implementation of tight-binding model operations and Hamiltonian construction.

pub use crate::model_utils::{find_R, remove_col, remove_row};

use crate::atom_struct::{Atom, AtomId, AtomType, AtomWire, OrbProj, OrbitalId, atoms_from_wire};
use crate::error::{Result, TbError};
use ndarray::*;
use ndarray_linalg::Inverse;
use num_complex::Complex;
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::{Deref, DerefMut};

// ── RMatrix type-level tag ──────────────────────────────────────────────────

/// Trait marking whether a model stores position matrix elements.
pub trait RMatrixData: Clone + std::fmt::Debug + Sync {
    const HAS_RMATRIX: bool;
    /// Create the default rmatrix with orbital positions on the diagonal.
    fn from_orb(orb: &Array2<f64>, norb: usize, spin: bool, dim: usize) -> Self;
    /// Wrap an Array4 into the RMatrixData type.
    fn from_array(arr: Array4<Complex<f64>>) -> Self;
    /// Get a reference to the underlying Array4. Panics for NoRMatrix.
    fn as_array4(&self) -> &Array4<Complex<f64>>;
    /// Get a mutable reference to the underlying Array4. Panics for NoRMatrix.
    fn as_array4_mut(&mut self) -> &mut Array4<Complex<f64>>;
    /// Select axes for the underlying Array4. No-op for NoRMatrix.
    fn select_axes(&self, axis1: Axis, indices1: &[usize], axis2: Axis, indices2: &[usize])
    -> Self;
}

/// Position matrix elements are stored. Wraps [`Array4<Complex<f64>>`] with
/// zero-overhead newtype.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HasRMatrix(pub Array4<Complex<f64>>);

impl RMatrixData for HasRMatrix {
    const HAS_RMATRIX: bool = true;
    fn from_orb(orb: &Array2<f64>, norb: usize, spin: bool, dim: usize) -> Self {
        let nsta = if spin { 2 * norb } else { norb };
        let mut r = Array4::<Complex<f64>>::zeros((1, dim, nsta, nsta));
        for i in 0..norb {
            for ri in 0..dim {
                r[[0, ri, i, i]] = Complex::<f64>::from(orb[[i, ri]]);
                if spin {
                    r[[0, ri, i + norb, i + norb]] = Complex::<f64>::from(orb[[i, ri]]);
                }
            }
        }
        HasRMatrix(r)
    }
    fn from_array(arr: Array4<Complex<f64>>) -> Self {
        HasRMatrix(arr)
    }
    fn as_array4(&self) -> &Array4<Complex<f64>> {
        &self.0
    }
    fn as_array4_mut(&mut self) -> &mut Array4<Complex<f64>> {
        &mut self.0
    }
    fn select_axes(
        &self,
        axis1: Axis,
        indices1: &[usize],
        axis2: Axis,
        indices2: &[usize],
    ) -> Self {
        HasRMatrix(self.0.select(axis1, indices1).select(axis2, indices2))
    }
}

impl Deref for HasRMatrix {
    type Target = Array4<Complex<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HasRMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// No position matrix elements stored. Zero-sized type.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NoRMatrix;

impl RMatrixData for NoRMatrix {
    const HAS_RMATRIX: bool = false;
    fn from_orb(_orb: &Array2<f64>, _norb: usize, _spin: bool, _dim: usize) -> Self {
        NoRMatrix
    }
    fn from_array(_arr: Array4<Complex<f64>>) -> Self {
        NoRMatrix
    }
    fn as_array4(&self) -> &Array4<Complex<f64>> {
        panic!("NoRMatrix has no underlying Array4; only HasRMatrix supports this operation")
    }
    fn as_array4_mut(&mut self) -> &mut Array4<Complex<f64>> {
        panic!("NoRMatrix has no underlying Array4; only HasRMatrix supports this operation")
    }
    fn select_axes(
        &self,
        _axis1: Axis,
        _indices1: &[usize],
        _axis2: Axis,
        _indices2: &[usize],
    ) -> Self {
        NoRMatrix
    }
}

// ── Model struct ────────────────────────────────────────────────────────────

/// Tight-binding model structure.
///
/// Const generic `SPIN`: spinless (false, default) / spinful (true).
/// Const generic `DIM`: spatial dimension 1/2/3 (default 3).
/// Type parameter `R`: [`HasRMatrix`] or [`NoRMatrix`] (default).
#[derive(Clone, Debug)]
pub struct Model<const SPIN: bool = false, const DIM: usize = 3, R: RMatrixData = NoRMatrix> {
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    pub orb_projection: Vec<OrbProj>,
    pub atoms: Vec<Atom>,
    pub ham: Array3<Complex<f64>>,
    pub hamR: Array2<isize>,
    pub rmatrix: R,
}

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

// Manual Serialize
impl<const SPIN: bool, const DIM: usize, R: RMatrixData + Serialize> Serialize
    for Model<SPIN, DIM, R>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let n_fields = if R::HAS_RMATRIX { 9 } else { 8 };
        let mut s = serializer.serialize_struct("Model", n_fields)?;
        s.serialize_field("dim_r", &DIM)?;
        s.serialize_field("spin", &SPIN)?;
        s.serialize_field("lat", &self.lat)?;
        s.serialize_field("orb", &self.orb)?;
        s.serialize_field("orb_projection", &self.orb_projection)?;
        s.serialize_field("atoms", &self.atoms)?;
        s.serialize_field("ham", &self.ham)?;
        s.serialize_field("hamR", &self.hamR)?;
        if R::HAS_RMATRIX {
            s.serialize_field("rmatrix", &self.rmatrix)?;
        }
        s.end()
    }
}

// Helper for deserialization
#[derive(Deserialize)]
#[serde(field_identifier)]
enum ModelField {
    #[serde(rename = "dim_r")]
    DimR,
    #[serde(rename = "spin")]
    Spin,
    #[serde(rename = "lat")]
    Lat,
    #[serde(rename = "orb")]
    Orb,
    #[serde(rename = "orb_projection")]
    OrbProjection,
    #[serde(rename = "atoms")]
    Atoms,
    #[serde(rename = "ham")]
    Ham,
    #[serde(rename = "hamR")]
    HamR,
    #[serde(rename = "rmatrix")]
    Rmatrix,
}

impl<'de, const SPIN: bool, const DIM: usize> Deserialize<'de> for Model<SPIN, DIM, NoRMatrix> {
    fn deserialize<De: Deserializer<'de>>(
        deserializer: De,
    ) -> std::result::Result<Self, De::Error> {
        struct ModelVisitor<const S: bool, const D: usize>;

        impl<'de, const S: bool, const D: usize> de::Visitor<'de> for ModelVisitor<S, D> {
            type Value = Model<S, D, NoRMatrix>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a Model struct without rmatrix")
            }

            fn visit_map<A: de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut dim_r: Option<usize> = None;
                let mut spin: Option<bool> = None;
                let mut lat: Option<Array2<f64>> = None;
                let mut orb: Option<Array2<f64>> = None;
                let mut orb_projection: Option<Vec<OrbProj>> = None;
                let mut atoms: Option<Vec<AtomWire>> = None;
                let mut ham: Option<Array3<Complex<f64>>> = None;
                let mut hamR: Option<Array2<isize>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        ModelField::DimR => dim_r = Some(map.next_value()?),
                        ModelField::Spin => spin = Some(map.next_value()?),
                        ModelField::Lat => lat = Some(map.next_value()?),
                        ModelField::Orb => orb = Some(map.next_value()?),
                        ModelField::OrbProjection => orb_projection = Some(map.next_value()?),
                        ModelField::Atoms => atoms = Some(map.next_value()?),
                        ModelField::Ham => ham = Some(map.next_value()?),
                        ModelField::HamR => hamR = Some(map.next_value()?),
                        ModelField::Rmatrix => {
                            let _: Array4<Complex<f64>> = map.next_value()?;
                        }
                    }
                }

                let spin = spin.ok_or_else(|| de::Error::missing_field("spin"))?;
                if spin != S {
                    return Err(de::Error::custom(format!(
                        "spin mismatch: file has spin={}, but Model<{}> was requested",
                        spin, S
                    )));
                }
                let dim_r = dim_r.ok_or_else(|| de::Error::missing_field("dim_r"))?;
                if dim_r != D {
                    return Err(de::Error::custom(format!(
                        "dimension mismatch: file has dim_r={}, but Model<DIM={}> was requested",
                        dim_r, D
                    )));
                }

                let atoms =
                    atoms_from_wire(atoms.ok_or_else(|| de::Error::missing_field("atoms"))?)
                        .map_err(de::Error::custom)?;
                let model = Model {
                    lat: lat.ok_or_else(|| de::Error::missing_field("lat"))?,
                    orb: orb.ok_or_else(|| de::Error::missing_field("orb"))?,
                    orb_projection: orb_projection
                        .ok_or_else(|| de::Error::missing_field("orb_projection"))?,
                    atoms,
                    ham: ham.ok_or_else(|| de::Error::missing_field("ham"))?,
                    hamR: hamR.ok_or_else(|| de::Error::missing_field("hamR"))?,
                    rmatrix: NoRMatrix,
                };
                model.validate().map_err(de::Error::custom)?;
                Ok(model)
            }
        }

        deserializer.deserialize_struct(
            "Model",
            &[
                "dim_r",
                "spin",
                "lat",
                "orb",
                "orb_projection",
                "atoms",
                "ham",
                "hamR",
                "rmatrix",
            ],
            ModelVisitor::<SPIN, DIM>,
        )
    }
}

impl<'de, const SPIN: bool, const DIM: usize> Deserialize<'de> for Model<SPIN, DIM, HasRMatrix> {
    fn deserialize<De: Deserializer<'de>>(
        deserializer: De,
    ) -> std::result::Result<Self, De::Error> {
        struct ModelVisitor<const S: bool, const D: usize>;

        impl<'de, const S: bool, const D: usize> de::Visitor<'de> for ModelVisitor<S, D> {
            type Value = Model<S, D, HasRMatrix>;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a Model struct with rmatrix")
            }

            fn visit_map<A: de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut dim_r: Option<usize> = None;
                let mut spin: Option<bool> = None;
                let mut lat: Option<Array2<f64>> = None;
                let mut orb: Option<Array2<f64>> = None;
                let mut orb_projection: Option<Vec<OrbProj>> = None;
                let mut atoms: Option<Vec<AtomWire>> = None;
                let mut ham: Option<Array3<Complex<f64>>> = None;
                let mut hamR: Option<Array2<isize>> = None;
                let mut rmatrix: Option<Array4<Complex<f64>>> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        ModelField::DimR => dim_r = Some(map.next_value()?),
                        ModelField::Spin => spin = Some(map.next_value()?),
                        ModelField::Lat => lat = Some(map.next_value()?),
                        ModelField::Orb => orb = Some(map.next_value()?),
                        ModelField::OrbProjection => orb_projection = Some(map.next_value()?),
                        ModelField::Atoms => atoms = Some(map.next_value()?),
                        ModelField::Ham => ham = Some(map.next_value()?),
                        ModelField::HamR => hamR = Some(map.next_value()?),
                        ModelField::Rmatrix => rmatrix = Some(map.next_value()?),
                    }
                }

                let spin = spin.ok_or_else(|| de::Error::missing_field("spin"))?;
                if spin != S {
                    return Err(de::Error::custom(format!(
                        "spin mismatch: file has spin={}, but Model<{}> was requested",
                        spin, S
                    )));
                }
                let dim_r = dim_r.ok_or_else(|| de::Error::missing_field("dim_r"))?;
                if dim_r != D {
                    return Err(de::Error::custom(format!(
                        "dimension mismatch: file has dim_r={}, but Model<DIM={}> was requested",
                        dim_r, D
                    )));
                }

                let atoms =
                    atoms_from_wire(atoms.ok_or_else(|| de::Error::missing_field("atoms"))?)
                        .map_err(de::Error::custom)?;
                let model = Model {
                    lat: lat.ok_or_else(|| de::Error::missing_field("lat"))?,
                    orb: orb.ok_or_else(|| de::Error::missing_field("orb"))?,
                    orb_projection: orb_projection
                        .ok_or_else(|| de::Error::missing_field("orb_projection"))?,
                    atoms,
                    ham: ham.ok_or_else(|| de::Error::missing_field("ham"))?,
                    hamR: hamR.ok_or_else(|| de::Error::missing_field("hamR"))?,
                    rmatrix: HasRMatrix(
                        rmatrix.ok_or_else(|| de::Error::missing_field("rmatrix"))?,
                    ),
                };
                model.validate().map_err(de::Error::custom)?;
                Ok(model)
            }
        }

        deserializer.deserialize_struct(
            "Model",
            &[
                "dim_r",
                "spin",
                "lat",
                "orb",
                "orb_projection",
                "atoms",
                "ham",
                "hamR",
                "rmatrix",
            ],
            ModelVisitor::<SPIN, DIM>,
        )
    }
}

/// Gauge choice for the Bloch basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Gauge {
    Lattice,
    Atom,
}

/// System dimensionality.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Dimension {
    one = 1,
    two = 2,
    three = 3,
}

/// Pauli matrix selector for spin-dependent operators.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum SpinDirection {
    X = 1,
    Y = 2,
    Z = 3,
}

// Include Model implementation from submodules
pub use crate::model_build::*;
pub use crate::model_physics::*;

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    #[inline(always)]
    pub fn has_rmatrix(&self) -> bool {
        R::HAS_RMATRIX
    }
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

    /// Validate synchronized model arrays and atom-to-orbital references.
    pub fn validate(&self) -> Result<()> {
        if self.norb() == 0 {
            return Err(TbError::NoOrbitals);
        }
        if self.lat.dim() != (DIM, DIM) {
            return Err(TbError::InvalidModelInvariant {
                invariant: "lattice_shape",
                message: format!("expected ({DIM}, {DIM}), found {:?}", self.lat.dim()),
            });
        }
        if self.orb.ncols() != DIM {
            return Err(TbError::InvalidModelInvariant {
                invariant: "orbital_position_shape",
                message: format!("expected {} columns, found {}", DIM, self.orb.ncols()),
            });
        }
        if !self
            .lat
            .iter()
            .chain(self.orb.iter())
            .all(|value| value.is_finite())
        {
            return Err(TbError::InvalidModelInvariant {
                invariant: "finite_geometry",
                message: "lattice and orbital positions must be finite".to_string(),
            });
        }
        if self.orb_projection.len() != self.norb() {
            return Err(TbError::InvalidModelInvariant {
                invariant: "orbital_projection_count",
                message: format!(
                    "expected {}, found {}",
                    self.norb(),
                    self.orb_projection.len()
                ),
            });
        }
        for (index, atom) in self.atoms.iter().enumerate() {
            if atom.position_ref().len() != DIM
                || !atom.position_ref().iter().all(|value| value.is_finite())
            {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "atomic_position",
                    message: format!("atom {index} must have {DIM} finite coordinates"),
                });
            }
            if atom
                .magnetic_moment()
                .is_some_and(|moment| moment.iter().any(|component| !component.is_finite()))
            {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "magnetic_moment",
                    message: format!("atom {index} has a non-finite magnetic moment"),
                });
            }
        }
        self.orbital_owners()?;

        let expected_ham = (self.hamR.nrows(), self.nsta(), self.nsta());
        if self.ham.dim() != expected_ham {
            return Err(TbError::InvalidModelInvariant {
                invariant: "hamiltonian_shape",
                message: format!("expected {expected_ham:?}, found {:?}", self.ham.dim()),
            });
        }
        if self.hamR.ncols() != DIM {
            return Err(TbError::InvalidModelInvariant {
                invariant: "hopping_translation_shape",
                message: format!("expected {DIM} columns, found {}", self.hamR.ncols()),
            });
        }
        if R::HAS_RMATRIX {
            let expected = (self.hamR.nrows(), DIM, self.nsta(), self.nsta());
            if self.rmatrix.as_array4().dim() != expected {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "position_matrix_shape",
                    message: format!(
                        "expected {expected:?}, found {:?}",
                        self.rmatrix.as_array4().dim()
                    ),
                });
            }
        }
        Ok(())
    }
    pub fn dim_r(&self) -> usize {
        DIM
    }
    /// Reciprocal lattice vectors satisfying `B Aᵀ = 2π·I`, where `A` is the
    /// real-space lattice (rows = lattice vectors, stored in [`Model::lat`]).
    ///
    /// Each row of the returned matrix is a reciprocal lattice vector
    /// `bᵢ`.  The inversion can fail for a degenerate real-space lattice.
    pub fn rec_lat(&self) -> Result<Array2<f64>> {
        let inv_t = self
            .lat
            .t()
            .to_owned()
            .inv()
            .map_err(|e| TbError::Other(format!("Failed to invert lattice: {e}")))?;
        Ok(std::f64::consts::TAU * inv_t)
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
    pub fn natom(&self) -> usize {
        self.atoms.len()
    }
    #[inline(always)]
    pub fn norb(&self) -> usize {
        self.orb.nrows()
    }
    #[inline(always)]
    pub fn nsta(&self) -> usize {
        if SPIN { 2 * self.norb() } else { self.norb() }
    }
    #[inline(always)]
    pub fn orb_angular(&self) -> Result<Array3<Complex<f64>>> {
        if self.atoms.is_empty() {
            return Err(TbError::MissingAtomicStructure);
        }
        self.validate()?;
        // Every orbital must be owned by an atom.  Unowned orbitals (reachable
        // through remove_atoms_only or partial-ownership Atom construction)
        // would silently keep zero angular-momentum matrix elements, which is
        // indistinguishable from a genuine s-orbital zero.
        let owners = self.orbital_owners()?;
        if let Some(unowned) = owners.iter().position(Option::is_none) {
            return Err(TbError::Other(format!(
                "orb_angular requires every orbital to be owned by an atom, \
                 but orbital {unowned} has no owner. \
                 Remove the orbital or attach it to an Atom first."
            )));
        }
        let li = Complex::i() * 1.0;
        let mut L = Array3::<Complex<f64>>::zeros((self.dim_r(), self.norb(), self.norb()));
        let mut Lx = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        let mut Ly = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
        let mut Lz = Array2::<Complex<f64>>::zeros((self.norb(), self.norb()));
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
        let Lx_orig = &Lup_orig + &Ldn_orig;
        let Ly_orig = -li * (&Lup_orig - &Ldn_orig);
        for atom0 in self.atoms.iter() {
            for &orbital_i in atom0.orbitals() {
                let i = orbital_i.index();
                let proj_i: Array1<Complex<f64>> = self.orb_projection[i]
                    .to_quantum_number()?
                    .mapv(|x: Complex<f64>| x.conj());
                for &orbital_j in atom0.orbitals() {
                    let j = orbital_j.index();
                    let proj_j = self.orb_projection[j].to_quantum_number()?;
                    L[[0, i, j]] = proj_i.dot(&Lx_orig.dot(&proj_j));
                    L[[1, i, j]] = proj_i.dot(&Ly_orig.dot(&proj_j));
                    L[[2, i, j]] = proj_i.dot(&Lz_orig.dot(&proj_j));
                }
            }
        }
        Ok(L)
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use ndarray::array;

    fn assert_model_round_trip<const SPIN: bool, R>()
    where
        R: RMatrixData + Serialize,
        for<'de> Model<SPIN, 3, R>: Deserialize<'de>,
    {
        let mut model = Model::<SPIN, 3, R>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0), OrbitalId::new(1)],
            )]),
        )
        .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.0]).unwrap();

        let encoded = toml::to_string(&model).unwrap();
        let decoded: Model<SPIN, 3, R> = toml::from_str(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.norb(), 2);
        assert_eq!(
            decoded.atoms[0].orbitals(),
            &[OrbitalId::new(0), OrbitalId::new(1)]
        );
        assert_eq!(decoded.atoms[0].magnetic_moment(), Some([0.0, 0.0, 2.0]));
        assert_eq!(decoded.has_rmatrix(), R::HAS_RMATRIX);
    }

    fn non_contiguous_model() -> Model<false, 3> {
        Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.4, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(
                    array![0.0, 0.0, 0.0],
                    AtomType::C,
                    [OrbitalId::new(0), OrbitalId::new(2)],
                ),
                Atom::with_orbitals(array![0.2, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn atom_view_borrows_explicit_non_contiguous_orbitals() {
        let model = non_contiguous_model();
        let atom = model.atom(AtomId::new(0)).unwrap();
        assert_eq!(atom.orbitals().len(), 2);
        assert_eq!(atom.orbitals()[0].id(), OrbitalId::new(0));
        assert_eq!(atom.orbitals()[1].id(), OrbitalId::new(2));
        assert_eq!(atom.orbitals()[1].position(), array![0.4, 0.0, 0.0].view());
    }

    #[test]
    fn orbital_only_model_has_no_fabricated_atoms() {
        let model =
            Model::<false, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None).unwrap();
        assert!(model.atoms.is_empty());
        assert_eq!(model.orbital_owners().unwrap(), vec![None]);
        assert!(matches!(
            model.orb_angular(),
            Err(TbError::MissingAtomicStructure)
        ));
    }

    #[test]
    fn tb_model_rejects_empty_orbital_set() {
        // Regression: tb_model with a zero-row orb matrix used to succeed and
        // produce a model that silently flowed into solve/response entry
        // points. It must now fail with NoOrbitals.
        let result =
            Model::<false, 3>::tb_model(Array2::eye(3), Array2::<f64>::zeros((0, 3)), None);
        assert!(matches!(result, Err(TbError::NoOrbitals)));
    }

    #[test]
    fn orb_angular_rejects_unowned_orbitals() {
        // Regression: an orbital not owned by any atom used to silently keep
        // zero angular-momentum matrix elements (indistinguishable from a
        // genuine s-orbital). It must now produce a clear error.
        let model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let err = model.orb_angular().unwrap_err();
        assert!(
            err.to_string().contains("has no owner"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_orbital_ownership_is_rejected() {
        let result = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::O, [OrbitalId::new(0)]),
            ]),
        );
        assert!(matches!(result, Err(TbError::DuplicateOrbitalOwner { .. })));
    }

    #[test]
    fn removing_an_orbital_remaps_ids_and_keeps_empty_sites() {
        let mut model = non_contiguous_model();
        model.remove_orb(&[1]).unwrap();
        model.validate().unwrap();
        assert_eq!(model.norb(), 2);
        assert_eq!(
            model.atoms[0].orbitals(),
            &[OrbitalId::new(0), OrbitalId::new(1)]
        );
        assert!(model.atoms[1].orbitals().is_empty());
    }

    #[test]
    fn explicit_atom_removal_modes_preserve_their_named_semantics() {
        let mut metadata_only = non_contiguous_model();
        metadata_only.remove_atoms_only(&[0]).unwrap();
        assert_eq!(metadata_only.norb(), 3);
        assert_eq!(metadata_only.natom(), 1);
        assert_eq!(
            metadata_only.orbital_owners().unwrap(),
            vec![None, Some(AtomId::new(0)), None]
        );

        metadata_only.remove_orb(&[1]).unwrap();
        assert_eq!(metadata_only.natom(), 1);
        assert!(metadata_only.atoms[0].orbitals().is_empty());
        metadata_only.prune_empty_atoms().unwrap();
        assert_eq!(metadata_only.natom(), 0);

        let mut cascading = non_contiguous_model();
        cascading.remove_atoms_and_orbitals(&[0]).unwrap();
        assert_eq!(cascading.natom(), 1);
        assert_eq!(cascading.norb(), 1);
        assert_eq!(cascading.atoms[0].orbitals(), &[OrbitalId::new(0)]);
    }

    #[test]
    fn orbital_only_supercell_remains_orbital_only() {
        let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.25]], None).unwrap();
        let supercell = model.make_supercell(&array![[2.0]]).unwrap();
        assert_eq!(supercell.norb(), 2);
        assert!(supercell.atoms.is_empty());
        assert_eq!(supercell.orbital_owners().unwrap(), vec![None, None]);
        supercell.validate().unwrap();
    }

    #[test]
    fn supercell_replicates_optional_atom_moments() {
        let mut model = Model::<false, 1>::tb_model(
            array![[1.0]],
            array![[0.25]],
            Some(vec![Atom::with_orbitals(
                array![0.25],
                AtomType::Fe,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.5]).unwrap();
        let supercell = model.make_supercell(&array![[3.0]]).unwrap();
        assert_eq!(supercell.natom(), 3);
        assert!(
            supercell
                .atoms
                .iter()
                .all(|atom| atom.magnetic_moment() == Some([0.0, 0.0, 2.5]))
        );
        supercell.validate().unwrap();
    }

    #[test]
    fn model_serde_round_trips_all_type_level_storage_combinations() {
        assert_model_round_trip::<false, NoRMatrix>();
        assert_model_round_trip::<true, NoRMatrix>();
        assert_model_round_trip::<false, HasRMatrix>();
        assert_model_round_trip::<true, HasRMatrix>();
    }

    #[test]
    fn legacy_atom_counts_deserialize_to_contiguous_typed_ids() {
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        model.atoms[1].set_magnetic_moment([0.0, 1.0, 0.0]).unwrap();
        let mut value = toml::Value::try_from(&model).unwrap();
        let atoms = value["atoms"].as_array_mut().unwrap();
        for atom in atoms {
            let table = atom.as_table_mut().unwrap();
            let count = table.remove("orbitals").unwrap().as_array().unwrap().len();
            table.insert("atom_list".to_string(), toml::Value::Integer(count as i64));
            if let Some(moment) = table.remove("magnetic_moment") {
                table.insert("magnetic".to_string(), moment);
            }
        }

        let decoded: Model<false, 3> = toml::from_str(&toml::to_string(&value).unwrap()).unwrap();
        assert_eq!(decoded.atoms[0].orbitals(), &[OrbitalId::new(0)]);
        assert_eq!(decoded.atoms[1].orbitals(), &[OrbitalId::new(1)]);
        assert_eq!(decoded.atoms[0].magnetic_moment(), None);
        assert_eq!(decoded.atoms[1].magnetic_moment(), Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn legacy_zero_magnetic_alias_is_distinct_from_a_missing_moment() {
        let model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Fe,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let mut value = toml::Value::try_from(&model).unwrap();
        let atom = value["atoms"].as_array_mut().unwrap()[0]
            .as_table_mut()
            .unwrap();
        atom.insert(
            "magnetic".to_string(),
            toml::Value::Array(vec![
                toml::Value::Float(0.0),
                toml::Value::Float(0.0),
                toml::Value::Float(0.0),
            ]),
        );
        let decoded: Model<false, 3> = toml::from_str(&toml::to_string(&value).unwrap()).unwrap();
        assert_eq!(decoded.atoms[0].magnetic_moment(), Some([0.0; 3]));
    }

    #[test]
    fn standalone_atom_deserialization_rejects_nonfinite_moments() {
        let atom = Atom::new(array![0.0, 0.0, 0.0], AtomType::Fe);
        let mut value = toml::Value::try_from(&atom).unwrap();
        value.as_table_mut().unwrap().insert(
            "magnetic_moment".to_string(),
            toml::Value::Array(vec![
                toml::Value::Float(f64::NAN),
                toml::Value::Float(0.0),
                toml::Value::Float(0.0),
            ]),
        );
        let encoded = toml::to_string(&value).unwrap();
        assert!(toml::from_str::<Atom>(&encoded).is_err());
    }
}
