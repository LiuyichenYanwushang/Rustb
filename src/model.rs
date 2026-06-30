//! Core implementation of tight-binding model operations and Hamiltonian construction.

pub use crate::model_utils::{find_R, remove_col, remove_row};

use crate::atom_struct::{Atom, AtomType, OrbProj};
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

// Manual Serialize
impl<const SPIN: bool, const DIM: usize, R: RMatrixData + Serialize> Serialize
    for Model<SPIN, DIM, R>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let n_fields = if R::HAS_RMATRIX { 8 } else { 7 };
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
#[serde(field_identifier, rename_all = "lowercase")]
enum ModelField {
    DimR,
    Spin,
    Lat,
    Orb,
    OrbProjection,
    Atoms,
    Ham,
    HamR,
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
                let mut atoms: Option<Vec<Atom>> = None;
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

                Ok(Model {
                    lat: lat.ok_or_else(|| de::Error::missing_field("lat"))?,
                    orb: orb.ok_or_else(|| de::Error::missing_field("orb"))?,
                    orb_projection: orb_projection
                        .ok_or_else(|| de::Error::missing_field("orb_projection"))?,
                    atoms: atoms.ok_or_else(|| de::Error::missing_field("atoms"))?,
                    ham: ham.ok_or_else(|| de::Error::missing_field("ham"))?,
                    hamR: hamR.ok_or_else(|| de::Error::missing_field("hamR"))?,
                    rmatrix: NoRMatrix,
                })
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
                let mut atoms: Option<Vec<Atom>> = None;
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

                Ok(Model {
                    lat: lat.ok_or_else(|| de::Error::missing_field("lat"))?,
                    orb: orb.ok_or_else(|| de::Error::missing_field("orb"))?,
                    orb_projection: orb_projection
                        .ok_or_else(|| de::Error::missing_field("orb_projection"))?,
                    atoms: atoms.ok_or_else(|| de::Error::missing_field("atoms"))?,
                    ham: ham.ok_or_else(|| de::Error::missing_field("ham"))?,
                    hamR: hamR.ok_or_else(|| de::Error::missing_field("hamR"))?,
                    rmatrix: HasRMatrix(
                        rmatrix.ok_or_else(|| de::Error::missing_field("rmatrix"))?,
                    ),
                })
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
                atom_p.assign(&atom.position());
            });
        atom_position
    }
    pub fn dim_r(&self) -> usize {
        DIM
    }
    /// Reciprocal lattice vectors satisfying `B Aᵀ = 2π·I`, where `A` is the
    /// real-space lattice (columns = lattice vectors, stored in [`Model::lat`]).
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
        let mut a = 0;
        for atom0 in self.atoms.iter() {
            for i in a..a + atom0.norb() {
                let proj_i: Array1<Complex<f64>> = self.orb_projection[i]
                    .to_quantum_number()?
                    .mapv(|x: Complex<f64>| x.conj());
                for j in a..a + atom0.norb() {
                    let proj_j = self.orb_projection[j].to_quantum_number()?;
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
