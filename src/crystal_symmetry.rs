//! Optional crystallographic-symmetry services backed by `cryspglib`.
//!
//! Enable the Cargo feature `cryspglib` to make this module available. The
//! adapter analyzes atomic sites, not Wannier/orbital centers. Structure
//! symmetry and tight-binding Hamiltonian symmetry are deliberately separate:
//! the results here do not assert that `H(k)` transforms under every detected
//! operation.

use crate::error::{Result, TbError};
use crate::model::{Model, RMatrixData};
use cryspglib::irrep::corep::symmetry_operations_of;
use cryspglib::irrep::magnetic_summary::{
    MagneticCharacterTableColumns, format_magnetic_character_table_with_columns,
    magnetic_irrep_summary_by_uni,
};
use cryspglib::irrep::query::{format_character_table, irreps_of, kpoints_of};
use cryspglib::irrep::types::generated_data::SG_DATA_HALL;
use ndarray::{Array1, Array2, Array3};
use ndarray_linalg::Determinant;
use std::collections::BTreeMap;

/// Numerical parameters for crystallographic symmetry analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryParameters {
    /// Cartesian position tolerance.
    pub symprec: f64,
    /// Angular tolerance in radians. A negative value requests automatic
    /// tolerance selection in cryspglib.
    pub angle_tolerance: f64,
    /// Optional uniform external fields used to determine the surviving
    /// symmetry-operation subset.
    pub external_fields: ExternalFields,
    /// Relative/absolute comparison tolerance for transformed field vectors.
    pub field_tolerance: f64,
}

impl Default for SymmetryParameters {
    fn default() -> Self {
        Self {
            symprec: 1e-5,
            angle_tolerance: -1.0,
            external_fields: ExternalFields::default(),
            field_tolerance: 1e-10,
        }
    }
}

impl SymmetryParameters {
    fn validate(self) -> Result<Self> {
        if !self.symprec.is_finite() || self.symprec <= 0.0 {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "symprec",
                message: "must be finite and positive".to_string(),
            });
        }
        if !self.angle_tolerance.is_finite() {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "angle_tolerance",
                message: "must be finite".to_string(),
            });
        }
        if !self.field_tolerance.is_finite() || self.field_tolerance <= 0.0 {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "field_tolerance",
                message: "must be finite and positive".to_string(),
            });
        }
        for (name, field) in [
            ("electric_field", self.external_fields.electric),
            ("magnetic_field", self.external_fields.magnetic),
        ] {
            if field.is_some_and(|vector| vector.iter().any(|value| !value.is_finite())) {
                return Err(TbError::InvalidCrystalSymmetryInput {
                    parameter: name,
                    message: "all Cartesian components must be finite".to_string(),
                });
            }
        }
        Ok(self)
    }
}

/// Optional uniform Cartesian fields present during symmetry analysis.
///
/// An electric field is a time-even polar vector. A magnetic field is a
/// time-odd axial vector. `None` means that field is absent.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ExternalFields {
    pub electric: Option<[f64; 3]>,
    pub magnetic: Option<[f64; 3]>,
}

/// One crystallographic Seitz operation `{R|t}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrystalSymmetryOperation {
    /// Integer fractional-coordinate rotation.
    pub rotation: [[i32; 3]; 3],
    /// Fractional translation in the original model's lattice basis.
    pub translation: [f64; 3],
    /// Whether the operation is anti-unitary.
    pub time_reversal: bool,
}

/// Rustb-owned result of non-magnetic crystallographic analysis.
#[derive(Debug, Clone)]
pub struct CrystalSymmetryDataset {
    pub spacegroup_number: usize,
    pub hall_number: usize,
    pub international_symbol: String,
    pub hall_symbol: String,
    pub choice: String,
    pub pointgroup_symbol: String,
    /// Coordinate transform `x_standard = P x_input + p`.
    pub transformation_matrix: Array2<f64>,
    pub origin_shift: [f64; 3],
    /// Structural operations expressed in the original model's fractional
    /// basis.
    pub operations: Vec<CrystalSymmetryOperation>,
    /// Operations from `operations` that also preserve the selected external
    /// electric and magnetic fields.
    pub field_preserving_operations: Vec<CrystalSymmetryOperation>,
    pub external_fields: ExternalFields,
    pub wyckoff_letters: Vec<char>,
    pub site_symmetry_symbols: Vec<String>,
    pub equivalent_atoms: Vec<usize>,
    pub crystallographic_orbits: Vec<usize>,
    pub mapping_to_primitive: Vec<usize>,
    /// Standard lattice in Rustb's row-vector convention.
    pub standard_lattice: Array2<f64>,
    pub standard_positions: Array2<f64>,
    pub standard_types: Vec<i32>,
    /// Primitive lattice in Rustb's row-vector convention.
    pub primitive_lattice: Array2<f64>,
}

/// A high-symmetry reciprocal-space record.
#[derive(Debug, Clone, PartialEq)]
pub struct HighSymmetryKPoint {
    pub label: String,
    /// Coordinate in cryspglib's irrep data-Hall reciprocal basis.
    pub canonical_coordinate: [f64; 3],
    /// Coordinate transformed to the original Rustb reciprocal basis.
    pub model_coordinate: [f64; 3],
    /// `false` identifies a line/plane representative rather than an isolated
    /// high-symmetry point.
    pub is_point: bool,
}

impl CrystalSymmetryDataset {
    fn ensure_database_symmetry_is_effective(&self) -> Result<()> {
        if self.field_preserving_operations.len() != self.operations.len() {
            return Err(TbError::FieldReducedSymmetryData {
                structural_operations: self.operations.len(),
                effective_operations: self.field_preserving_operations.len(),
            });
        }
        let sg = u8::try_from(self.spacegroup_number).map_err(|_| {
            TbError::InvalidCrystalSymmetryInput {
                parameter: "spacegroup_number",
                message: format!("{} is outside 1..=230", self.spacegroup_number),
            }
        })?;
        let data_hall = SG_DATA_HALL[sg as usize] as usize;
        if data_hall != self.hall_number {
            return Err(TbError::UnsupportedHighSymmetrySetting {
                detected_hall: self.hall_number,
                data_hall,
            });
        }
        Ok(())
    }

    /// Return high-symmetry records in both canonical and original model bases.
    ///
    /// cryspglib's irrep tables use one fixed data-Hall setting per space
    /// group. Until a unique general setting transform is available, a
    /// different detected Hall setting is rejected instead of returning
    /// silently incorrect coordinates.
    pub fn high_symmetry_kpoints(&self) -> Result<Vec<HighSymmetryKPoint>> {
        self.ensure_database_symmetry_is_effective()?;
        let sg = self.spacegroup_number as u8;
        let irreps = irreps_of(sg);
        kpoints_of(sg)
            .into_iter()
            .map(|point| {
                let (kx, ky, kz, denominator) = point.coords;
                if denominator == 0 {
                    return Err(TbError::InvalidCrystalSymmetryInput {
                        parameter: "high_symmetry_coordinate",
                        message: format!("k-point {} has a zero denominator", point.label),
                    });
                }
                let d = denominator as f64;
                let canonical = [kx as f64 / d, ky as f64 / d, kz as f64 / d];
                let mut model = [0.0; 3];
                for (j, component) in model.iter_mut().enumerate() {
                    *component = (0..3)
                        .map(|i| canonical[i] * self.transformation_matrix[[i, j]])
                        .sum();
                }
                let is_point = point
                    .irreps
                    .first()
                    .and_then(|&index| irreps.get(index))
                    .is_some_and(|irrep| irrep.is_point());
                Ok(HighSymmetryKPoint {
                    label: point.label,
                    canonical_coordinate: canonical,
                    model_coordinate: model,
                    is_point,
                })
            })
            .collect()
    }

    /// Complete operation-column character table at a selected label.
    ///
    /// Its operation columns use cryspglib's canonical database basis, not the
    /// original model basis. Use [`Self::character_table_operations`] to obtain
    /// the headers in exactly that order and coordinate frame.
    pub fn character_table_at(&self, label: &str) -> Result<String> {
        self.ensure_database_symmetry_is_effective()?;
        let sg = self.spacegroup_number as u8;
        let mut matches = kpoints_of(sg)
            .into_iter()
            .filter(|point| point.label == label);
        let point = matches
            .next()
            .ok_or_else(|| TbError::InvalidCrystalSymmetryInput {
                parameter: "kpoint_label",
                message: format!(
                    "space group {} has no label {label}",
                    self.spacegroup_number
                ),
            })?;
        if matches.next().is_some() {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "kpoint_label",
                message: format!("label {label} is not unique; select by coordinates"),
            });
        }
        let (kx, ky, kz, kd) = point.coords;
        Ok(format_character_table(sg, kx, ky, kz, kd))
    }

    /// Canonical operation columns used by [`Self::character_table_at`].
    ///
    /// These operations deliberately remain in the irrep database basis; they
    /// must not be positionally matched against [`Self::operations`] for an
    /// input cell related by a nontrivial basis transformation.
    pub fn character_table_operations(&self) -> Result<Vec<CrystalSymmetryOperation>> {
        self.ensure_database_symmetry_is_effective()?;
        let sg = u8::try_from(self.spacegroup_number).map_err(|_| {
            TbError::InvalidCrystalSymmetryInput {
                parameter: "spacegroup_number",
                message: format!("{} is outside 1..=230", self.spacegroup_number),
            }
        })?;
        Ok(symmetry_operations_of(sg)?
            .iter()
            .copied()
            .map(convert_operation)
            .collect())
    }
}

/// Rustb-owned magnetic-space-group analysis.
#[derive(Debug, Clone)]
pub struct MagneticCrystalSymmetry {
    /// Number of the nonmagnetic structural parent detected before applying
    /// site moments. This is not necessarily the family-space-group number of
    /// the identified MSG; for example a Type-IV magnetic subgroup can have a
    /// lower family space group.
    pub spacegroup_number: usize,
    pub international_symbol: String,
    /// Hall setting of the nonmagnetic structural parent, retained as
    /// coordinate-setting provenance.
    pub hall_number: usize,
    pub hall_symbol: String,
    pub uni_number: usize,
    pub magnetic_type: MagneticGroupType,
    pub bns_number: String,
    pub og_number: String,
    pub operations: Vec<CrystalSymmetryOperation>,
    pub field_preserving_operations: Vec<CrystalSymmetryOperation>,
    pub external_fields: ExternalFields,
}

/// Magnetic-space-group type without exposing cryspglib's pre-1.0 enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagneticGroupType {
    NonMagnetic,
    Ordinary,
    Grey,
    BlackWhite,
    AntiTranslation,
}

/// Column grouping for magnetic character tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MagneticTableColumns {
    Operations,
    ConjugacyClasses,
}

impl MagneticCrystalSymmetry {
    fn ensure_database_symmetry_is_effective(&self) -> Result<()> {
        if self.field_preserving_operations.len() != self.operations.len() {
            return Err(TbError::FieldReducedSymmetryData {
                structural_operations: self.operations.len(),
                effective_operations: self.field_preserving_operations.len(),
            });
        }
        Ok(())
    }

    /// Canonical fixed-k magnetic corep labels and coordinates.
    pub fn high_symmetry_kpoints(&self) -> Result<Vec<(String, [f64; 3])>> {
        self.ensure_database_symmetry_is_effective()?;
        let summary = self.corep_summary()?;
        Ok(summary
            .kpoints
            .iter()
            .map(|point| {
                let (kx, ky, kz, kd) = point.coords;
                let d = kd as f64;
                (
                    point.label.clone(),
                    [kx as f64 / d, ky as f64 / d, kz as f64 / d],
                )
            })
            .collect())
    }

    /// Formal magnetic character table at a selected fixed-k label.
    pub fn character_table_at(&self, label: &str, columns: MagneticTableColumns) -> Result<String> {
        self.ensure_database_symmetry_is_effective()?;
        let summary = self.corep_summary()?;
        let mut matches = summary.kpoints.iter().filter(|point| point.label == label);
        let point = matches
            .next()
            .ok_or_else(|| TbError::InvalidCrystalSymmetryInput {
                parameter: "magnetic_kpoint_label",
                message: format!("UNI {} has no label {label}", self.uni_number),
            })?;
        if matches.next().is_some() {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "magnetic_kpoint_label",
                message: format!("label {label} is not unique"),
            });
        }
        let columns = match columns {
            MagneticTableColumns::Operations => MagneticCharacterTableColumns::Operations,
            MagneticTableColumns::ConjugacyClasses => {
                MagneticCharacterTableColumns::ConjugacyClasses
            }
        };
        Ok(format_magnetic_character_table_with_columns(point, columns))
    }

    fn corep_summary(&self) -> Result<cryspglib::irrep::magnetic_summary::MagneticIrrepSummary> {
        if self.uni_number == 0 {
            return Err(TbError::MagneticIrrepAnalysis {
                uni: 0,
                message: "the detected structure is non-magnetic".to_string(),
            });
        }
        magnetic_irrep_summary_by_uni(self.uni_number).map_err(|error| {
            TbError::MagneticIrrepAnalysis {
                uni: self.uni_number,
                message: format!("{error:?}"),
            }
        })
    }
}

/// Symmetry-reduced reciprocal mesh with explicit multiplicities and weights.
#[derive(Debug, Clone)]
pub struct IrreducibleKMesh {
    /// Irreducible fractional reciprocal coordinates in `[0, 1)`.
    pub kpoints: Array2<f64>,
    /// Number of full-grid points represented by each row of `kpoints`.
    pub multiplicities: Array1<usize>,
    /// Multiplicities normalized by the full grid size.
    pub weights: Array1<f64>,
    /// cryspglib full-grid index to compact irreducible row.
    pub full_to_irreducible: Array1<usize>,
    /// Rustb [`crate::gen_kmesh`] row index to compact irreducible row.
    pub rustb_full_to_irreducible: Array1<usize>,
    /// cryspglib full-grid addresses. Their linear order is not Rustb's
    /// `gen_kmesh` order.
    pub full_grid_addresses: Array2<i32>,
}

/// Structure-level symmetry services for a three-dimensional Rustb model.
pub trait CrystalSymmetry {
    fn crystal_symmetry(&self, parameters: &SymmetryParameters) -> Result<CrystalSymmetryDataset>;

    /// Detect magnetic symmetry from the optional moments stored on [`crate::Atom`].
    ///
    /// An Atom with `magnetic_moment() == None` contributes an explicit zero
    /// vector to the magnetic-structure analysis. Thus a default structure is
    /// nonmagnetic and yields its grey magnetic group, while callers can add
    /// magnetic sites through [`crate::Atom::set_magnetic_moment`].
    fn magnetic_crystal_symmetry_from_atoms(
        &self,
        parameters: &SymmetryParameters,
    ) -> Result<MagneticCrystalSymmetry>;

    /// Detect magnetic symmetry from a temporary per-call moment override.
    fn magnetic_crystal_symmetry(
        &self,
        moments: &[[f64; 3]],
        parameters: &SymmetryParameters,
    ) -> Result<MagneticCrystalSymmetry>;

    /// Reduce a structural/non-magnetic mesh. `time_reversal=true` is an
    /// explicit assertion about the Hamiltonian; for magnetic order use
    /// [`Self::magnetic_irreducible_kmesh`] with site moments.
    fn irreducible_kmesh(
        &self,
        mesh: [i32; 3],
        shift: [i32; 3],
        time_reversal: bool,
        parameters: &SymmetryParameters,
    ) -> Result<IrreducibleKMesh>;

    /// Mesh reduced by the magnetic operations detected from explicit site
    /// moments, including anti-unitary operations and external-field context.
    fn magnetic_irreducible_kmesh(
        &self,
        moments: &[[f64; 3]],
        mesh: [i32; 3],
        shift: [i32; 3],
        parameters: &SymmetryParameters,
    ) -> Result<IrreducibleKMesh>;

    /// Reduce a mesh by magnetic moments stored directly on the Atoms.
    fn magnetic_irreducible_kmesh_from_atoms(
        &self,
        mesh: [i32; 3],
        shift: [i32; 3],
        parameters: &SymmetryParameters,
    ) -> Result<IrreducibleKMesh>;
}

impl<const SPIN: bool, R: RMatrixData> CrystalSymmetry for Model<SPIN, 3, R> {
    fn crystal_symmetry(&self, parameters: &SymmetryParameters) -> Result<CrystalSymmetryDataset> {
        let parameters = parameters.validate()?;
        let crystal = model_crystal(self, None)?;
        let analysis = crystal
            .analyze()
            .symprec(parameters.symprec)
            .angle_tolerance(parameters.angle_tolerance)
            .external_fields(cryspglib_fields(parameters.external_fields))
            .field_tolerance(parameters.field_tolerance);
        let dataset = analysis.dataset()?;
        let effective_operations = analysis.effective_symmetry()?;
        convert_dataset(dataset, effective_operations, parameters.external_fields)
    }

    fn magnetic_crystal_symmetry_from_atoms(
        &self,
        parameters: &SymmetryParameters,
    ) -> Result<MagneticCrystalSymmetry> {
        let moments = atom_magnetic_moments(self)?;
        self.magnetic_crystal_symmetry(&moments, parameters)
    }

    fn magnetic_crystal_symmetry(
        &self,
        moments: &[[f64; 3]],
        parameters: &SymmetryParameters,
    ) -> Result<MagneticCrystalSymmetry> {
        let parameters = parameters.validate()?;
        if moments.len() != self.natom() {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "moments",
                message: format!("expected {}, found {}", self.natom(), moments.len()),
            });
        }
        if !moments.iter().flatten().all(|value| value.is_finite()) {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "moments",
                message: "all moment components must be finite".to_string(),
            });
        }
        let crystal = model_crystal(self, Some(moments))?;
        let analysis = crystal
            .analyze()
            .symprec(parameters.symprec)
            .angle_tolerance(parameters.angle_tolerance);
        let result = analysis.magnetic_dataset()?;
        let operations = cryspglib::SymmetryOps::from_parallel(
            &result.rotations,
            &result.translations,
            &result.time_reversals,
        );
        let effective_operations = operations.preserving_fields(
            &crystal.lattice,
            cryspglib_fields(parameters.external_fields),
            parameters.field_tolerance,
        )?;
        convert_magnetic(result, effective_operations, parameters.external_fields)
    }

    fn irreducible_kmesh(
        &self,
        mesh: [i32; 3],
        shift: [i32; 3],
        time_reversal: bool,
        parameters: &SymmetryParameters,
    ) -> Result<IrreducibleKMesh> {
        let parameters = parameters.validate()?;
        let crystal = model_crystal(self, None)?;
        let analysis = crystal
            .analyze()
            .symprec(parameters.symprec)
            .angle_tolerance(parameters.angle_tolerance)
            .external_fields(cryspglib_fields(parameters.external_fields))
            .field_tolerance(parameters.field_tolerance);
        let structural_operations = analysis.symmetry()?;
        let candidates = if time_reversal {
            structural_operations.grey_extension()?
        } else {
            structural_operations
        };
        let effective_operations = candidates.preserving_fields(
            &crystal.lattice,
            cryspglib_fields(parameters.external_fields),
            parameters.field_tolerance,
        )?;
        let rotations = effective_operations
            .iter()
            .map(|operation| {
                if operation.time_reversal {
                    operation.rotation.map(|row| row.map(|value| -value))
                } else {
                    operation.rotation
                }
            })
            .collect::<Vec<_>>();
        let stabilized =
            cryspglib::stabilized_reciprocal_mesh(mesh, shift, false, &rotations, &[])?;
        let raw = cryspglib::IrMesh {
            grid_addresses: stabilized.grid_addresses,
            mapping_table: stabilized.mapping_table,
            num_ir: stabilized.num_ir,
        };
        convert_mesh(raw, mesh, shift)
    }

    fn magnetic_irreducible_kmesh(
        &self,
        moments: &[[f64; 3]],
        mesh: [i32; 3],
        shift: [i32; 3],
        parameters: &SymmetryParameters,
    ) -> Result<IrreducibleKMesh> {
        let parameters = parameters.validate()?;
        if moments.len() != self.natom() {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "moments",
                message: format!("expected {}, found {}", self.natom(), moments.len()),
            });
        }
        if !moments.iter().flatten().all(|value| value.is_finite()) {
            return Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "moments",
                message: "all moment components must be finite".to_string(),
            });
        }
        let crystal = model_crystal(self, Some(moments))?;
        let result = crystal
            .analyze()
            .symprec(parameters.symprec)
            .angle_tolerance(parameters.angle_tolerance)
            .magnetic_dataset()?;
        let operations = cryspglib::SymmetryOps::from_parallel(
            &result.rotations,
            &result.translations,
            &result.time_reversals,
        )
        .preserving_fields(
            &crystal.lattice,
            cryspglib_fields(parameters.external_fields),
            parameters.field_tolerance,
        )?;
        let reciprocal_actions = operations
            .iter()
            .map(|operation| {
                if operation.time_reversal {
                    operation.rotation.map(|row| row.map(|value| -value))
                } else {
                    operation.rotation
                }
            })
            .collect::<Vec<_>>();
        let stabilized =
            cryspglib::stabilized_reciprocal_mesh(mesh, shift, false, &reciprocal_actions, &[])?;
        convert_mesh(
            cryspglib::IrMesh {
                grid_addresses: stabilized.grid_addresses,
                mapping_table: stabilized.mapping_table,
                num_ir: stabilized.num_ir,
            },
            mesh,
            shift,
        )
    }

    fn magnetic_irreducible_kmesh_from_atoms(
        &self,
        mesh: [i32; 3],
        shift: [i32; 3],
        parameters: &SymmetryParameters,
    ) -> Result<IrreducibleKMesh> {
        let moments = atom_magnetic_moments(self)?;
        self.magnetic_irreducible_kmesh(&moments, mesh, shift, parameters)
    }
}

fn atom_magnetic_moments<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
) -> Result<Vec<[f64; 3]>> {
    model.validate()?;
    if model.atoms.is_empty() {
        return Err(TbError::MissingAtomicStructure);
    }
    Ok(model
        .atoms
        .iter()
        .map(|atom| atom.magnetic_moment().unwrap_or([0.0; 3]))
        .collect())
}

fn model_crystal<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    moments: Option<&[[f64; 3]]>,
) -> Result<cryspglib::Crystal> {
    model.validate()?;
    if model.atoms.is_empty() {
        return Err(TbError::MissingAtomicStructure);
    }
    let determinant = model.lat.det()?;
    if !determinant.is_finite() || determinant.abs() <= 1e-14 {
        return Err(TbError::InvalidCrystalSymmetryInput {
            parameter: "lattice",
            message: "lattice must be finite and non-singular".to_string(),
        });
    }

    // Rustb: lattice vectors are rows. cryspglib: lattice[cart][vec].
    let mut lattice = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            lattice[i][j] = model.lat[[j, i]];
        }
    }
    let positions = model
        .atoms
        .iter()
        .map(|atom| {
            let position = atom.position_ref();
            [position[0], position[1], position[2]]
        })
        .collect::<Vec<_>>();
    let types = model
        .atoms
        .iter()
        .map(|atom| atom.atom_type().atomic_number())
        .collect::<Vec<_>>();
    let mut crystal = cryspglib::Crystal::new(lattice, positions, types);
    if let Some(moments) = moments {
        crystal = crystal.with_magnetic(moments.to_vec());
    }
    Ok(crystal)
}

fn convert_dataset(
    dataset: cryspglib::SpaceGroup,
    effective_operations: cryspglib::SymmetryOps,
    external_fields: ExternalFields,
) -> Result<CrystalSymmetryDataset> {
    let operations: Vec<CrystalSymmetryOperation> = dataset
        .rotations
        .iter()
        .copied()
        .zip(dataset.translations.iter().copied())
        .map(|(rotation, translation)| CrystalSymmetryOperation {
            rotation,
            translation,
            time_reversal: false,
        })
        .collect();
    let field_preserving_operations = effective_operations
        .iter()
        .copied()
        .map(convert_operation)
        .collect();
    Ok(CrystalSymmetryDataset {
        spacegroup_number: dataset.spacegroup_number,
        hall_number: dataset.hall_number,
        international_symbol: dataset.international_symbol,
        hall_symbol: dataset.hall_symbol,
        choice: dataset.choice,
        pointgroup_symbol: dataset.pointgroup_symbol,
        transformation_matrix: matrix3(dataset.transformation_matrix),
        origin_shift: dataset.origin_shift,
        operations,
        field_preserving_operations,
        external_fields,
        wyckoff_letters: dataset
            .wyckoffs
            .iter()
            .map(|&value| char::from_u32('a' as u32 + value.max(0) as u32).unwrap_or('?'))
            .collect(),
        site_symmetry_symbols: dataset.site_symmetry_symbols,
        equivalent_atoms: dataset
            .equivalent_atoms
            .iter()
            .map(|&value| value.max(0) as usize)
            .collect(),
        crystallographic_orbits: dataset
            .crystallographic_orbits
            .iter()
            .map(|&value| value.max(0) as usize)
            .collect(),
        mapping_to_primitive: dataset
            .mapping_to_primitive
            .iter()
            .map(|&value| value.max(0) as usize)
            .collect(),
        standard_lattice: lattice_from_cry(dataset.std_lattice),
        standard_positions: positions_array(&dataset.std_positions),
        standard_types: dataset.std_types,
        primitive_lattice: lattice_from_cry(dataset.primitive_lattice),
    })
}

fn convert_magnetic(
    result: cryspglib::MagneticSymmetry,
    effective_operations: cryspglib::SymmetryOps,
    external_fields: ExternalFields,
) -> Result<MagneticCrystalSymmetry> {
    let magnetic_type = match result.magnetic_type {
        cryspglib::MagneticType::NonMagnetic => MagneticGroupType::NonMagnetic,
        cryspglib::MagneticType::Ordinary => MagneticGroupType::Ordinary,
        cryspglib::MagneticType::Grey => MagneticGroupType::Grey,
        cryspglib::MagneticType::BlackWhite => MagneticGroupType::BlackWhite,
        cryspglib::MagneticType::AntiTranslation => MagneticGroupType::AntiTranslation,
    };
    let operations: Vec<CrystalSymmetryOperation> = result
        .rotations
        .iter()
        .copied()
        .zip(result.translations.iter().copied())
        .zip(result.time_reversals.iter().copied())
        .map(
            |((rotation, translation), time_reversal)| CrystalSymmetryOperation {
                rotation,
                translation,
                time_reversal,
            },
        )
        .collect();
    let field_preserving_operations = effective_operations
        .iter()
        .copied()
        .map(convert_operation)
        .collect();
    Ok(MagneticCrystalSymmetry {
        spacegroup_number: result.spacegroup_number,
        international_symbol: result.international_short,
        hall_number: result.hall_number,
        hall_symbol: result.hall_symbol,
        uni_number: result.uni_number,
        magnetic_type,
        bns_number: result.bns_number,
        og_number: result.og_number,
        operations,
        field_preserving_operations,
        external_fields,
    })
}

fn convert_operation(operation: cryspglib::SymmetryOp) -> CrystalSymmetryOperation {
    CrystalSymmetryOperation {
        rotation: operation.rotation,
        translation: operation.translation,
        time_reversal: operation.time_reversal,
    }
}

fn cryspglib_fields(fields: ExternalFields) -> cryspglib::ExternalFields {
    cryspglib::ExternalFields {
        electric: fields.electric,
        magnetic: fields.magnetic,
    }
}

fn convert_mesh(
    raw: cryspglib::IrMesh,
    mesh: [i32; 3],
    shift: [i32; 3],
) -> Result<IrreducibleKMesh> {
    if mesh.iter().any(|&value| value <= 0) {
        return Err(TbError::InvalidCrystalSymmetryInput {
            parameter: "mesh",
            message: "all mesh dimensions must be positive".to_string(),
        });
    }
    let mut representative_to_compact = BTreeMap::new();
    for &representative in &raw.mapping_table {
        if representative >= raw.grid_addresses.len() {
            return Err(TbError::InvalidModelInvariant {
                invariant: "cryspglib_mesh_mapping",
                message: format!("representative {representative} is out of range"),
            });
        }
        let next = representative_to_compact.len();
        representative_to_compact
            .entry(representative)
            .or_insert(next);
    }
    if representative_to_compact.len() != raw.num_ir {
        return Err(TbError::InvalidModelInvariant {
            invariant: "cryspglib_irreducible_count",
            message: format!(
                "reported {}, derived {}",
                raw.num_ir,
                representative_to_compact.len()
            ),
        });
    }
    let mut kpoints = Array2::zeros((raw.num_ir, 3));
    for (&representative, &compact) in &representative_to_compact {
        let address = raw.grid_addresses[representative];
        for d in 0..3 {
            let coordinate = (2 * address[d] + shift[d]) as f64 / (2 * mesh[d]) as f64;
            kpoints[[compact, d]] = coordinate.rem_euclid(1.0);
        }
    }
    let mut multiplicities = Array1::zeros(raw.num_ir);
    let mut full_to_irreducible = Array1::zeros(raw.mapping_table.len());
    for (full, &representative) in raw.mapping_table.iter().enumerate() {
        let compact = representative_to_compact[&representative];
        full_to_irreducible[full] = compact;
        multiplicities[compact] += 1;
    }
    let full_size = raw.mapping_table.len() as f64;
    let weights = multiplicities.mapv(|count| count as f64 / full_size);
    let mut full_grid_addresses = Array2::zeros((raw.grid_addresses.len(), 3));
    let mut rustb_full_to_irreducible = Array1::zeros(raw.grid_addresses.len());
    for (row, address) in raw.grid_addresses.iter().enumerate() {
        for d in 0..3 {
            full_grid_addresses[[row, d]] = address[d];
        }
        let x = address[0].rem_euclid(mesh[0]) as usize;
        let y = address[1].rem_euclid(mesh[1]) as usize;
        let z = address[2].rem_euclid(mesh[2]) as usize;
        let rustb_row = x * mesh[1] as usize * mesh[2] as usize + y * mesh[2] as usize + z;
        rustb_full_to_irreducible[rustb_row] = full_to_irreducible[row];
    }
    Ok(IrreducibleKMesh {
        kpoints,
        multiplicities,
        weights,
        full_to_irreducible,
        rustb_full_to_irreducible,
        full_grid_addresses,
    })
}

fn matrix3(matrix: [[f64; 3]; 3]) -> Array2<f64> {
    Array2::from_shape_fn((3, 3), |(i, j)| matrix[i][j])
}

fn lattice_from_cry(lattice: [[f64; 3]; 3]) -> Array2<f64> {
    Array2::from_shape_fn((3, 3), |(i, j)| lattice[j][i])
}

fn positions_array(positions: &[[f64; 3]]) -> Array2<f64> {
    Array2::from_shape_fn((positions.len(), 3), |(i, j)| positions[i][j])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Atom, AtomType, OrbitalId};
    use ndarray::array;
    use ndarray_linalg::Inverse;

    fn simple_cubic() -> Model<false, 3> {
        Model::tb_model(
            Array2::eye(3),
            array![[0.31, 0.27, 0.19]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap()
    }

    #[test]
    fn adapter_transposes_skew_row_lattice_and_uses_atoms() {
        let mut model = simple_cubic();
        model.lat = array![[2.0, 0.1, 0.2], [0.3, 3.0, 0.4], [0.5, 0.6, 4.0]];
        let crystal = model_crystal(&model, None).unwrap();
        assert_eq!(crystal.lattice[0], [2.0, 0.3, 0.5]);
        assert_eq!(crystal.lattice[1], [0.1, 3.0, 0.6]);
        assert_eq!(crystal.positions, vec![[0.0, 0.0, 0.0]]);
        assert_eq!(crystal.types, vec![14]);
    }

    #[test]
    fn simple_cubic_dataset_and_character_table() {
        let dataset = simple_cubic()
            .crystal_symmetry(&SymmetryParameters::default())
            .unwrap();
        assert_eq!(dataset.spacegroup_number, 221);
        assert_eq!(dataset.hall_number, 517);
        assert_eq!(dataset.operations.len(), 48);
        let points = dataset.high_symmetry_kpoints().unwrap();
        assert!(points.iter().any(|point| point.label == "GM"));
        let table = dataset.character_table_at("GM").unwrap();
        assert!(table.contains("| ML | BC |"));
        assert!(table.contains("| 1 |"));
        assert_eq!(dataset.character_table_operations().unwrap().len(), 48);
    }

    #[test]
    fn character_table_columns_expose_their_canonical_basis_operations() {
        let mut model = simple_cubic();
        model.lat = array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]];
        let dataset = model
            .crystal_symmetry(&SymmetryParameters::default())
            .unwrap();
        let canonical = dataset.character_table_operations().unwrap();

        assert_eq!(dataset.hall_number, 517);
        assert_eq!(canonical.len(), dataset.operations.len());
        assert_ne!(canonical, dataset.operations);
        assert!(dataset.character_table_at("GM").unwrap().contains("| 1 |"));

        let point = dataset
            .high_symmetry_kpoints()
            .unwrap()
            .into_iter()
            .find(|point| {
                point.is_point
                    && point
                        .canonical_coordinate
                        .iter()
                        .any(|component| component.abs() > 1e-12)
            })
            .unwrap();
        let input_reciprocal = model.rec_lat().unwrap();
        let standard_reciprocal =
            std::f64::consts::TAU * dataset.standard_lattice.t().to_owned().inv().unwrap();
        let canonical_cartesian =
            Array1::from_vec(point.canonical_coordinate.to_vec()).dot(&standard_reciprocal);
        let model_cartesian =
            Array1::from_vec(point.model_coordinate.to_vec()).dot(&input_reciprocal);
        assert!(
            canonical_cartesian
                .iter()
                .zip(&model_cartesian)
                .all(|(canonical, model)| (canonical - model).abs() < 1e-10)
        );
    }

    #[test]
    fn irreducible_mesh_has_normalized_weights() {
        let reduced = simple_cubic()
            .irreducible_kmesh([2, 3, 4], [1, 0, 1], true, &SymmetryParameters::default())
            .unwrap();
        assert_eq!(reduced.multiplicities.sum(), 24);
        assert!((reduced.weights.sum() - 1.0).abs() < 1e-12);
        assert_eq!(reduced.full_to_irreducible.len(), 24);
        assert_eq!(reduced.rustb_full_to_irreducible.len(), 24);
    }

    #[test]
    fn optional_fields_change_effective_operations_not_structural_group() {
        let electric = simple_cubic()
            .crystal_symmetry(&SymmetryParameters {
                external_fields: ExternalFields {
                    electric: Some([0.0, 0.0, 1.0]),
                    magnetic: None,
                },
                ..SymmetryParameters::default()
            })
            .unwrap();
        let magnetic = simple_cubic()
            .crystal_symmetry(&SymmetryParameters {
                external_fields: ExternalFields {
                    electric: None,
                    magnetic: Some([0.0, 0.0, 1.0]),
                },
                ..SymmetryParameters::default()
            })
            .unwrap();
        let inversion = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]];
        assert_eq!(electric.spacegroup_number, 221);
        assert_eq!(electric.operations.len(), 48);
        assert!(electric.field_preserving_operations.len() < electric.operations.len());
        assert!(
            !electric
                .field_preserving_operations
                .iter()
                .any(|operation| operation.rotation == inversion)
        );
        assert!(matches!(
            electric.high_symmetry_kpoints(),
            Err(TbError::FieldReducedSymmetryData { .. })
        ));
        assert!(matches!(
            electric.character_table_at("GM"),
            Err(TbError::FieldReducedSymmetryData { .. })
        ));
        assert!(
            magnetic
                .field_preserving_operations
                .iter()
                .any(|operation| operation.rotation == inversion)
        );
    }

    #[test]
    fn optional_atom_moment_drives_magnetic_group_analysis_directly() {
        let mut model = simple_cubic();
        assert_eq!(model.atoms[0].magnetic_moment(), None);

        let nonmagnetic = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        assert_eq!(nonmagnetic.magnetic_type, MagneticGroupType::Grey);

        model.atoms[0].set_magnetic_moment([0.0, 0.0, 1.0]).unwrap();
        let magnetic = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        assert_eq!(magnetic.magnetic_type, MagneticGroupType::BlackWhite);
        assert!(magnetic.operations.len() < nonmagnetic.operations.len());
        let stored_mesh = model
            .magnetic_irreducible_kmesh_from_atoms(
                [4, 4, 4],
                [0, 0, 0],
                &SymmetryParameters::default(),
            )
            .unwrap();
        let explicit_mesh = model
            .magnetic_irreducible_kmesh(
                &[[0.0, 0.0, 1.0]],
                [4, 4, 4],
                [0, 0, 0],
                &SymmetryParameters::default(),
            )
            .unwrap();
        assert_eq!(
            stored_mesh.full_to_irreducible,
            explicit_mesh.full_to_irreducible
        );
        assert_eq!(stored_mesh.multiplicities, explicit_mesh.multiplicities);

        model.atoms[0].clear_magnetic_moment();
        assert_eq!(model.atoms[0].magnetic_moment(), None);
    }

    #[test]
    fn external_fields_reduce_the_irreducible_mesh_with_effective_operations() {
        let structural = simple_cubic()
            .irreducible_kmesh([4, 4, 4], [0, 0, 0], true, &SymmetryParameters::default())
            .unwrap();
        let electric = simple_cubic()
            .irreducible_kmesh(
                [4, 4, 4],
                [0, 0, 0],
                true,
                &SymmetryParameters {
                    external_fields: ExternalFields {
                        electric: Some([0.0, 0.0, 1.0]),
                        magnetic: None,
                    },
                    ..SymmetryParameters::default()
                },
            )
            .unwrap();
        let magnetic_without_time_reversal = simple_cubic()
            .irreducible_kmesh(
                [4, 4, 4],
                [0, 0, 0],
                true,
                &SymmetryParameters {
                    external_fields: ExternalFields {
                        electric: None,
                        magnetic: Some([0.0, 0.0, 1.0]),
                    },
                    ..SymmetryParameters::default()
                },
            )
            .unwrap();

        assert!(electric.kpoints.nrows() >= structural.kpoints.nrows());
        assert!(magnetic_without_time_reversal.kpoints.nrows() >= structural.kpoints.nrows());
        assert!((electric.weights.sum() - 1.0).abs() < 1e-12);
        assert!((magnetic_without_time_reversal.weights.sum() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn magnetic_analysis_keeps_uni_context_and_formats_coreps() {
        let magnetic = simple_cubic()
            .magnetic_crystal_symmetry(&[[0.0, 0.0, 1.0]], &SymmetryParameters::default())
            .unwrap();
        assert!(magnetic.uni_number > 0);
        assert!(!magnetic.bns_number.is_empty());
        assert_eq!(
            magnetic.operations.len(),
            magnetic.field_preserving_operations.len()
        );
        let points = magnetic.high_symmetry_kpoints().unwrap();
        let label = &points.first().unwrap().0;
        let table = magnetic
            .character_table_at(label, MagneticTableColumns::Operations)
            .unwrap();
        assert!(table.contains("operation"));
    }

    #[test]
    fn magnetic_mesh_uses_detected_antiunitary_group_not_pure_time_reversal() {
        let nonmagnetic = simple_cubic()
            .irreducible_kmesh([4, 4, 4], [0, 0, 0], true, &SymmetryParameters::default())
            .unwrap();
        let magnetic = simple_cubic()
            .magnetic_irreducible_kmesh(
                &[[0.0, 0.0, 1.0]],
                [4, 4, 4],
                [0, 0, 0],
                &SymmetryParameters::default(),
            )
            .unwrap();

        assert!(magnetic.kpoints.nrows() > nonmagnetic.kpoints.nrows());
        assert_eq!(magnetic.multiplicities.sum(), 64);
        assert!((magnetic.weights.sum() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn non_data_hall_setting_is_rejected_for_canonical_kpoints() {
        let mut dataset = simple_cubic()
            .crystal_symmetry(&SymmetryParameters::default())
            .unwrap();
        dataset.hall_number = 1;
        assert!(matches!(
            dataset.high_symmetry_kpoints(),
            Err(TbError::UnsupportedHighSymmetrySetting { .. })
        ));
    }

    #[test]
    fn missing_atoms_and_bad_moments_are_errors() {
        let orbital_only =
            Model::<false, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], Some(Vec::new()))
                .unwrap();
        assert!(matches!(
            orbital_only.crystal_symmetry(&SymmetryParameters::default()),
            Err(TbError::MissingAtomicStructure)
        ));
        assert!(matches!(
            simple_cubic().magnetic_crystal_symmetry(&[], &SymmetryParameters::default()),
            Err(TbError::InvalidCrystalSymmetryInput {
                parameter: "moments",
                ..
            })
        ));
    }
}
