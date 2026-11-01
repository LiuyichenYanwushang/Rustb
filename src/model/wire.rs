//! Serde wire format for [`Model`].
//!
//! `Model` is parameterized by `SPIN`, `DIM`, and the rmatrix storage type
//! `R`. The on-disk field names are shared, but the field set is not fully
//! independent of `R`: `HasRMatrix` serializes 9 fields including `rmatrix`,
//! while `NoRMatrix` serializes 8 and omits it. On deserialization `rmatrix`
//! is always optional at the serde layer and is parsed as a concrete
//! `HasRMatrix` value; the two `Deserialize` implementations decide whether
//! it is required and whether it is kept.
//!
//! Deserialization stages:
//!
//! 1. serde parses the on-disk fields into `ModelWire`. The derive and
//!    `deny_unknown_fields` reject unknown fields; serde's derived struct
//!    decoder rejects duplicate fields. No `Model` is constructed yet.
//! 2. the `Deserialize` implementation for the requested storage type handles
//!    the already-parsed optional `rmatrix`: `NoRMatrix` tolerates and drops
//!    it, while `HasRMatrix` moves it out and reports a missing field.
//! 3. `ModelWire::into_model` checks that the file's `spin` and `dim_r` match
//!    the requested `SPIN` and `DIM`, converts `AtomWire` into typed atoms,
//!    stores the caller-supplied `R`, and runs `Model::validate`.
//!
//! `Model::validate` checks the structural invariants serde cannot know
//! about: array shapes, finite values, and atom/orbital references.
//!
//! Consequently, a `HasRMatrix` file can be read as `NoRMatrix` (the matrix
//! is parsed and then dropped), but a `NoRMatrix` file cannot satisfy a
//! `HasRMatrix` target because it has no `rmatrix` field. `SPIN` and `DIM`
//! must match in every case.
//!
//! Serialization is manual rather than routed through `ModelWire`: the field
//! count and the presence of `rmatrix` depend on `R::HAS_RMATRIX`, and writing
//! directly from `&self` avoids copying the large arrays into a temporary DTO.

use super::{HasRMatrix, Model, NoRMatrix, RMatrixData};
use crate::atom_struct::{AtomWire, OrbProj, atoms_from_wire};
use ndarray::{Array2, Array3};
use num_complex::Complex;
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// Manual serialization: `rmatrix` is present only when `R` stores it.
// `HasRMatrix` writes 9 fields; `NoRMatrix` writes 8.
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

// Private deserialization DTO. It deliberately has no const generics and no
// `R`, so one parser handles the on-disk fields for every `Model`
// instantiation. The DTO parses and carries all on-disk fields; the only
// `rmatrix`-specific division of labour is that the two `Deserialize`
// implementations below decide whether it is required and whether it is kept.
// `into_model` performs the `SPIN`/`DIM` checks and the common
// construction/validation.
//
// The derive plus `deny_unknown_fields` rejects unknown fields; serde's
// derived struct decoder also rejects duplicate fields. This happens before
// any `Model` is constructed or validated.
//
// `rename = "Model"` only affects formats or diagnostics that use a struct
// name; TOML matches field names and ignores it. The name is kept aligned
// with the manual `serialize_struct("Model", ...)` call in this module.
#[derive(Deserialize)]
#[serde(rename = "Model", deny_unknown_fields)]
struct ModelWire {
    // Serialized metadata, not `Model` fields: `dim_r` and `spin` are const
    // generics in memory, so they are read only to detect a mismatched target.
    dim_r: usize,
    spin: bool,
    lat: Array2<f64>,
    orb: Array2<f64>,
    orb_projection: Vec<OrbProj>,
    // `Atom` carries typed orbital IDs; `atoms_from_wire` rebuilds and
    // validates those handles from the simpler on-disk representation.
    atoms: Vec<AtomWire>,
    ham: Array3<Complex<f64>>,
    hamR: Array2<isize>,
    // Parsed as concrete `HasRMatrix`, never as `R`, so the wire layer is
    // independent of the target storage. The `Deserialize` implementation for
    // the target decides whether to require it (`HasRMatrix`) or drop it
    // (`NoRMatrix`); `into_model` never branches on `R::HAS_RMATRIX`.
    #[serde(default, deserialize_with = "deserialize_rmatrix")]
    rmatrix: Option<HasRMatrix>,
}

// `#[serde(default)]` turns a missing field into `None`. When the field is
// present, parse it directly as an actual position matrix; this keeps absent
// and present unambiguous and rejects a `null`/non-array value instead of
// silently treating it as missing.
fn deserialize_rmatrix<'de, De: Deserializer<'de>>(
    deserializer: De,
) -> std::result::Result<Option<HasRMatrix>, De::Error> {
    HasRMatrix::deserialize(deserializer).map(Some)
}

impl ModelWire {
    /// Finish deserialization: check the wire metadata, convert the atom
    /// records, store the caller-supplied rmatrix value, and run the
    /// model-level invariant checks.
    ///
    /// The rmatrix requirement and the decision to keep or drop a parsed value
    /// belong to the two `Deserialize` implementations; this function only
    /// stores the `R` value they pass in. The constructor moves the parsed
    /// arrays; it does not copy them.
    fn into_model<const SPIN: bool, const DIM: usize, R: RMatrixData, E: de::Error>(
        self,
        rmatrix: R,
    ) -> std::result::Result<Model<SPIN, DIM, R>, E> {
        if self.spin != SPIN {
            return Err(E::custom(format!(
                "spin mismatch: file has spin={}, but Model<{}> was requested",
                self.spin, SPIN
            )));
        }
        if self.dim_r != DIM {
            return Err(E::custom(format!(
                "dimension mismatch: file has dim_r={}, but Model<DIM={}> was requested",
                self.dim_r, DIM
            )));
        }
        let model = Model {
            lat: self.lat,
            orb: self.orb,
            orb_projection: self.orb_projection,
            atoms: atoms_from_wire(self.atoms).map_err(E::custom)?,
            ham: self.ham,
            hamR: self.hamR,
            rmatrix,
        };
        model.validate().map_err(E::custom)?;
        Ok(model)
    }
}

// A `NoRMatrix` target does not require `rmatrix`. If present and serde-valid,
// the value was already parsed into `ModelWire` and is then dropped because
// this target cannot represent position-matrix data; serde-level parse errors
// still reject the file, but model-level checks on the dropped array (for
// example its shape) are not performed.
impl<'de, const SPIN: bool, const DIM: usize> Deserialize<'de> for Model<SPIN, DIM, NoRMatrix> {
    fn deserialize<De: Deserializer<'de>>(
        deserializer: De,
    ) -> std::result::Result<Self, De::Error> {
        ModelWire::deserialize(deserializer)?.into_model(NoRMatrix)
    }
}

// A `HasRMatrix` target requires the position matrix. `take()` removes it from
// the wire value so the rest can be passed to `into_model`; a missing field is
// reported as a normal serde error. The parsed array is not validated here
// beyond serde's type/array parsing; `Model::validate`, called from
// `into_model`, checks its shape and finite values.
impl<'de, const SPIN: bool, const DIM: usize> Deserialize<'de> for Model<SPIN, DIM, HasRMatrix> {
    fn deserialize<De: Deserializer<'de>>(
        deserializer: De,
    ) -> std::result::Result<Self, De::Error> {
        let mut wire = ModelWire::deserialize(deserializer)?;
        let rmatrix = wire
            .rmatrix
            .take()
            .ok_or_else(|| de::Error::missing_field("rmatrix"))?;
        wire.into_model(rmatrix)
    }
}
