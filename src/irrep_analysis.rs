//! Numerical magnetic little-group labels for tight-binding bands.
//!
//! This module plays the same role for a Rustb [`Model`] that programs such as
//! `irvsp` play for first-principles wavefunctions: it restricts localized-basis
//! symmetry actions to degenerate Bloch eigenspaces, computes their characters,
//! and compares those characters with cryspglib's magnetic corepresentation
//! tables.
//!
//! A missing localized-basis representation is a hard error.  In particular,
//! incomplete orbital shells, ambiguous local frames, and general Wannier
//! gauges are never guessed.  A Hamiltonian that breaks the requested magnetic
//! group is different: the numerical calculation still runs, but every label
//! is withheld when exact real-space covariance fails. A subspace with
//! excessive leakage, non-integer corep multiplicities, or a poor character
//! reconstruction is likewise labelled `???`. Raw complex characters remain
//! available in both cases. This makes the same API useful before and after
//! [`Model::symmetrize_hamiltonian`].
//!
//! High-symmetry k points are independent and are evaluated with Rayon.  The
//! indexed parallel collection preserves the canonical database order in the
//! returned report.  Users should normally set the selected BLAS backend to one
//! thread when using this outer parallelism.

use crate::crystal_symmetry::{
    CrystalSymmetry, CrystalSymmetryOperation, MagneticCrystalSymmetry, cry_lattice,
};
use crate::error::{Result, TbError};
use crate::hamiltonian_symmetry::{
    BasisActionContext, BasisSymmetryRepresentation, HamiltonianSymmetrizationParameters,
    LocalizedBasisAction, hamiltonian_residual, validate_action, validate_action_geometry,
    validate_hamiltonian, validate_projective_corepresentation, validate_tolerances,
};
use crate::{Gauge, Model, RMatrixData};
use cryspglib::irrep::magnetic_summary::{
    MagneticKPointSummary, UnresolvedMagneticCorep, magnetic_irrep_summary_by_uni_partial,
};
use cryspglib::irrep::types::{CharacterViewError, IrrepRecord, SeitzOperation};
use cryspglib::irrep::wigner::SettingTransform;
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eigh, LeastSquaresSvd, UPLO};
use num_complex::Complex64;
use rayon::prelude::*;
use std::fmt::{Display, Formatter, Write};

/// Numerical controls for [`Model::calculate_irrep`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrrepCalculationOptions {
    /// Structural, localized-representation, and operation tolerances.
    pub symmetry: HamiltonianSymmetrizationParameters,
    /// Absolute energy tolerance used to form degenerate band clusters.
    pub degeneracy_absolute: f64,
    /// Relative energy tolerance, scaled by the shift-invariant spectral
    /// width at each k point, used to form degenerate band clusters.
    pub degeneracy_relative: f64,
    /// Maximum normalized leakage of a transformed band subspace.
    pub subspace_tolerance: f64,
    /// Maximum relative residual of the fitted band-character vector.
    ///
    /// This controls only the numerical band-character fitting residual.
    pub character_tolerance: f64,
    /// Tolerance for a fitted corep multiplicity to be a non-negative integer.
    pub integer_tolerance: f64,
    /// Include database representatives of high-symmetry lines and planes.
    /// The default reports isolated high-symmetry points only.
    pub include_k_manifolds: bool,
}

impl Default for IrrepCalculationOptions {
    fn default() -> Self {
        Self {
            symmetry: HamiltonianSymmetrizationParameters::default(),
            degeneracy_absolute: 1e-7,
            degeneracy_relative: 1e-9,
            subspace_tolerance: 1e-6,
            character_tolerance: 1e-5,
            integer_tolerance: 1e-5,
            include_k_manifolds: false,
        }
    }
}

/// One numerical unitary character, or one anti-unitary sewing diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepCharacter {
    /// Character-table column in cryspglib's magnetic little-group order.
    pub column: usize,
    pub rotation: [[i32; 3]; 3],
    pub translation: [f64; 3],
    pub time_reversal: bool,
    /// `None` for anti-unitary operations, whose ordinary trace is not a
    /// basis-invariant character.
    pub value: Option<Complex64>,
    /// Frobenius leakage after projection back into the band cluster.
    pub subspace_leakage: f64,
    /// Residual of unitarity of the projected sewing matrix.
    pub projected_unitarity_residual: f64,
}

/// Raw least-squares multiplicity of one formal magnetic corepresentation.
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepMultiplicity {
    pub label: String,
    pub value: Complex64,
    /// Populated only when `value` is within the requested integer tolerance.
    pub rounded: Option<usize>,
}

/// Irrep/corep diagnosis for one consecutive degenerate band cluster.
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepBandReport {
    /// One-based first band index, inclusive.
    pub band_start: usize,
    /// One-based last band index, inclusive.
    pub band_end: usize,
    pub energies: Vec<f64>,
    /// A Miller-Love/corep sum, or exactly `"???"` when identification fails.
    pub label: String,
    pub multiplicities: Vec<IrrepMultiplicity>,
    pub characters: Vec<IrrepCharacter>,
    pub character_fit_residual: f64,
    pub decomposition_rank: usize,
    pub max_subspace_leakage: f64,
    pub max_projected_unitarity_residual: f64,
    /// Human-readable reasons for an unresolved `???` label.
    pub diagnostics: Vec<String>,
}

impl IrrepBandReport {
    pub fn is_identified(&self) -> bool {
        self.label != "???"
    }
}

/// All band clusters at one canonical high-symmetry k point.
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepKPointReport {
    pub label: String,
    pub canonical_coordinate: [f64; 3],
    pub model_coordinate: [f64; 3],
    pub is_point: bool,
    pub bands: Vec<IrrepBandReport>,
}

/// Exact real-space covariance diagnostic for one target magnetic operation.
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepHamiltonianOperationDiagnostic {
    pub operation_index: usize,
    pub rotation: [[i32; 3]; 3],
    pub translation: [f64; 3],
    pub time_reversal: bool,
    pub max_absolute_residual: f64,
    pub max_relative_residual: f64,
    pub relative_frobenius_residual: f64,
    pub acceptance_threshold: f64,
    pub preserved: bool,
}

/// Structured and printable result of [`Model::calculate_irrep`].
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepCalculationReport {
    pub uni_number: usize,
    pub bns_number: String,
    pub spinful: bool,
    /// Whether every requested magnetic operation preserves the complete
    /// real-space Hamiltonian, not merely the sampled high-symmetry subspaces.
    pub target_hamiltonian_compatible: bool,
    pub hamiltonian_operation_diagnostics: Vec<IrrepHamiltonianOperationDiagnostic>,
    pub high_symmetry_kpoints: Vec<IrrepKPointReport>,
}

impl IrrepCalculationReport {
    /// Produce a compact, irvsp-like plain-text table.
    pub fn format_irvsp(&self) -> String {
        self.to_string()
    }
}

impl Display for IrrepCalculationReport {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "Magnetic band irreps: UNI {}  BNS {}  {}",
            self.uni_number,
            self.bns_number,
            if self.spinful { "spinful" } else { "spinless" }
        )?;
        let max_hamiltonian_residual = self
            .hamiltonian_operation_diagnostics
            .iter()
            .map(|diagnostic| diagnostic.max_absolute_residual)
            .fold(0.0_f64, f64::max);
        let broken_operations = self
            .hamiltonian_operation_diagnostics
            .iter()
            .filter(|diagnostic| !diagnostic.preserved)
            .count();
        writeln!(
            formatter,
            "Target-H covariance: {}  broken operations: {}  max residual: {:.3e}",
            if self.target_hamiltonian_compatible {
                "compatible"
            } else {
                "BROKEN"
            },
            broken_operations,
            max_hamiltonian_residual,
        )?;
        writeln!(
            formatter,
            "band       energy range                 dim  irrep/corep  leakage      char.residual"
        )?;
        for point in &self.high_symmetry_kpoints {
            writeln!(
                formatter,
                "\nk-point {}  canonical=({:.8},{:.8},{:.8})  model=({:.8},{:.8},{:.8}){}",
                point.label,
                point.canonical_coordinate[0],
                point.canonical_coordinate[1],
                point.canonical_coordinate[2],
                point.model_coordinate[0],
                point.model_coordinate[1],
                point.model_coordinate[2],
                if point.is_point { "" } else { "  [manifold]" },
            )?;
            for band in &point.bands {
                let first = *band.energies.first().unwrap_or(&f64::NAN);
                let last = *band.energies.last().unwrap_or(&f64::NAN);
                writeln!(
                    formatter,
                    "{:>4}-{: <4} [{:>12.6},{:>12.6}] {:>4}  {:<12} {:>10.3e}  {:>10.3e}",
                    band.band_start,
                    band.band_end,
                    first,
                    last,
                    band.band_end - band.band_start + 1,
                    band.label,
                    band.max_subspace_leakage,
                    band.character_fit_residual,
                )?;
                for character in &band.characters {
                    write!(
                        formatter,
                        "             chi[{:>3}] {} = ",
                        character.column,
                        if character.time_reversal { "A" } else { "U" },
                    )?;
                    if let Some(value) = character.value {
                        write!(formatter, "{:+.8}{:+.8}i", value.re, value.im)?;
                    } else {
                        write!(formatter, "N/A (antiunitary)")?;
                    }
                    writeln!(
                        formatter,
                        "  leak={:.3e} unit={:.3e} R={:?} t=({:.8},{:.8},{:.8})",
                        character.subspace_leakage,
                        character.projected_unitarity_residual,
                        character.rotation,
                        character.translation[0],
                        character.translation[1],
                        character.translation[2],
                    )?;
                }
                if !band.is_identified() {
                    let mut detail = String::new();
                    for (index, multiplicity) in band.multiplicities.iter().enumerate() {
                        if index != 0 {
                            detail.push_str(", ");
                        }
                        write!(
                            detail,
                            "{}={:.6}{:+.6}i",
                            multiplicity.label, multiplicity.value.re, multiplicity.value.im
                        )?;
                    }
                    if !detail.is_empty() {
                        writeln!(formatter, "             raw multiplicities: {detail}")?;
                    }
                    for diagnostic in &band.diagnostics {
                        writeln!(formatter, "             ! {diagnostic}")?;
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
struct PreparedOperation {
    operation: CrystalSymmetryOperation,
    action: LocalizedBasisAction,
    data_rotation: [[i32; 3]; 3],
    data_translation_modulo: [f64; 3],
    data_translation_exact: [f64; 3],
}

#[derive(Clone)]
struct PointOperation {
    prepared_index: usize,
    character_column: usize,
    lattice_shift_in_data_frame: [f64; 3],
}

#[derive(Clone)]
struct PreparedKPoint {
    summary: MagneticKPointSummary,
    canonical_coordinate: [f64; 3],
    model_coordinate: [f64; 3],
    is_point: bool,
    operations: Vec<PointOperation>,
    formal_coreps: Vec<FormalCorep>,
    unresolved_coreps: Vec<UnresolvedMagneticCorep>,
}

#[derive(Clone)]
struct FormalCorep {
    label: String,
    dimension: usize,
    characters: Vec<Option<Complex64>>,
}

struct RecoveredFormalCorep {
    source_label: String,
    corep_type: cryspglib::irrep::corep::CorepType,
    formal: FormalCorep,
}

impl<const SPIN: bool, R: RMatrixData> Model<SPIN, 3, R> {
    /// Detect the Atom-defined magnetic group and calculate its band irreps.
    ///
    /// This is the usual entry point. The same `Model` may be passed before or
    /// after [`Model::symmetrize_hamiltonian`]: a broken Hamiltonian yields raw
    /// characters and `???`, while a valid symmetrized Hamiltonian yields
    /// integer multiplicities and database labels. Optional electric and
    /// magnetic fields are taken from [`IrrepCalculationOptions::symmetry`].
    pub fn calculate_irrep<P>(
        &self,
        representation: &P,
        options: Option<&IrrepCalculationOptions>,
    ) -> Result<IrrepCalculationReport>
    where
        P: BasisSymmetryRepresentation<SPIN, R>,
    {
        let owned_options = options.copied().unwrap_or_default();
        validate_options(&owned_options)?;
        let target_group = self
            .magnetic_crystal_symmetry_from_atoms(&owned_options.symmetry.structural_parameters)?;
        self.calculate_irrep_for_group(&target_group, representation, Some(&owned_options))
    }

    /// Calculate magnetic little-group characters and corep labels at all
    /// canonical high-symmetry k points for an explicitly supplied target.
    ///
    /// `target_group` supplies the magnetic group whose labels are requested.
    /// When `options` is `Some`, its electric/magnetic fields are reapplied to
    /// the full target operation set. With `None`, the field context stored in
    /// `target_group` is retained. The resulting operation set is validated
    /// and reidentified before any eigenproblem is solved.
    ///
    /// Every target operation is first resolved through `representation`.
    /// Failure here (for example, an orbital set not closed under a rotation)
    /// returns [`TbError::IrrepBasisRepresentation`] immediately.  In contrast,
    /// the Hamiltonian itself need not preserve the target: broken band
    /// subspaces remain in the report with raw characters and the label `???`.
    ///
    /// Independent k points are diagonalized in parallel with Rayon.  Result
    /// ordering remains deterministic.  To avoid nested oversubscription, use
    /// a single-threaded BLAS backend or set its thread count to one.
    pub fn calculate_irrep_for_group<P>(
        &self,
        target_group: &MagneticCrystalSymmetry,
        representation: &P,
        options: Option<&IrrepCalculationOptions>,
    ) -> Result<IrrepCalculationReport>
    where
        P: BasisSymmetryRepresentation<SPIN, R>,
    {
        let supplied_options = options.is_some();
        let options = options.copied().unwrap_or_default();
        validate_options(&options)?;
        self.validate()?;

        let target_operations = if supplied_options {
            let external_fields = options.symmetry.structural_parameters.external_fields;
            cryspglib::SymmetryOps {
                operations: target_group
                    .operations
                    .iter()
                    .map(|operation| cryspglib::SymmetryOp {
                        rotation: operation.rotation,
                        translation: operation.translation,
                        time_reversal: operation.time_reversal,
                    })
                    .collect(),
            }
            .preserving_fields(
                &cry_lattice(self),
                cryspglib::ExternalFields {
                    electric: external_fields.electric,
                    magnetic: external_fields.magnetic,
                },
                options.symmetry.structural_parameters.field_tolerance,
            )?
            .iter()
            .map(|operation| CrystalSymmetryOperation {
                rotation: operation.rotation,
                translation: operation.translation,
                time_reversal: operation.time_reversal,
            })
            .collect::<Vec<_>>()
        } else {
            target_group.field_preserving_operations.clone()
        };
        if target_operations.is_empty() {
            return Err(TbError::IrrepCalculation {
                message: "the target magnetic group has no field-preserving operations".to_string(),
            });
        }
        let cry_operations = cryspglib::SymmetryOps {
            operations: target_operations
                .iter()
                .map(|operation| cryspglib::SymmetryOp {
                    rotation: operation.rotation,
                    translation: operation.translation,
                    time_reversal: operation.time_reversal,
                })
                .collect(),
        };
        let validated = cryspglib::ValidatedMagneticOperationSet::try_from_symmetry_ops(
            &cry_operations,
            options.symmetry.tolerances.operation,
        )
        .map_err(|error| TbError::IrrepCalculation {
            message: format!("target operations are not a closed magnetic group: {error}"),
        })?;
        let identification = validated
            .identify(
                &cry_lattice(self),
                Some(target_group.hall_number),
                options.symmetry.structural_parameters.symprec,
            )
            .map_err(|error| TbError::IrrepCalculation {
                message: format!("target magnetic-group identification failed: {error}"),
            })?;
        let uni = identification.uni_number;
        let partial_summary = magnetic_irrep_summary_by_uni_partial(uni).map_err(|error| {
            TbError::MagneticIrrepAnalysis {
                uni,
                message: format!("{error:?}"),
            }
        })?;
        let summary = partial_summary.summary;
        let unresolved_coreps = partial_summary.unresolved_coreps;

        let input_to_standard = SettingTransform {
            basis: identification.transformation_matrix,
            origin: identification.origin_shift,
        };
        let subgroup = cryspglib::irrep::corep::identify_unitary_subgroup_with_hall(uni)
            .ok_or_else(|| TbError::MagneticIrrepAnalysis {
                uni,
                message: "the unitary subgroup could not be identified".to_string(),
            })?;
        let input_to_data = subgroup
            .msg_to_data
            .as_ref()
            .map_or(input_to_standard.clone(), |msg_to_data| {
                input_to_standard.then(msg_to_data)
            });

        // This is intentionally serial and precedes every eigenproblem.  A
        // partially representable orbital basis must fail as a whole rather
        // than producing a mixture of valid and guessed k-point labels.
        let position_tolerance = options
            .symmetry
            .tolerances
            .position
            .unwrap_or(options.symmetry.structural_parameters.symprec);
        let mut prepared_operations = Vec::with_capacity(validated.len());
        for (operation_index, operation) in validated.operations().iter().enumerate() {
            let operation = CrystalSymmetryOperation {
                rotation: operation.rotation,
                translation: operation.translation,
                time_reversal: operation.time_reversal,
            };
            let action = representation
                .resolve(BasisActionContext {
                    model: self,
                    operation: &operation,
                    position_tolerance,
                    representation_tolerance: options.symmetry.tolerances.representation,
                })
                .and_then(|action| {
                    let action = validate_action(
                        action,
                        self.nsta(),
                        options.symmetry.tolerances.representation,
                    )?;
                    validate_action_geometry(
                        self,
                        &operation,
                        &action,
                        position_tolerance,
                        options.symmetry.tolerances.representation,
                    )?;
                    Ok(action)
                })
                .map_err(|error| TbError::IrrepBasisRepresentation {
                    operation_index,
                    reason: error.to_string(),
                })?;
            let (data_rotation, data_translation_modulo) = input_to_data
                .transform_seitz(&operation.rotation, &operation.translation)
                .ok_or_else(|| TbError::IrrepCalculation {
                    message: format!(
                        "operation {operation_index} cannot be transformed to the irrep data-Hall frame"
                    ),
                })?;
            let data_translation_exact =
                exact_transformed_translation(&input_to_data, &operation, &data_rotation);
            prepared_operations.push(PreparedOperation {
                operation,
                action,
                data_rotation,
                data_translation_modulo,
                data_translation_exact,
            });
        }

        let operation_tolerance = options.symmetry.tolerances.operation.max(1e-8);
        for left in 0..prepared_operations.len() {
            for right in (left + 1)..prepared_operations.len() {
                let left_operation = &prepared_operations[left];
                let right_operation = &prepared_operations[right];
                if left_operation.data_rotation == right_operation.data_rotation
                    && left_operation.operation.time_reversal
                        == right_operation.operation.time_reversal
                    && translations_equivalent(
                        left_operation.data_translation_modulo,
                        right_operation.data_translation_modulo,
                        operation_tolerance,
                    )
                {
                    return Err(TbError::IrrepCalculation {
                        message: format!(
                            "target operations {left} and {right} collapse onto the same data-Hall Seitz representative; folded/nonprimitive cells must be unfolded before band-irrep analysis"
                        ),
                    });
                }
            }
        }

        let operations_and_actions = prepared_operations
            .iter()
            .map(|prepared| (prepared.operation, prepared.action.clone()))
            .collect::<Vec<_>>();
        validate_projective_corepresentation(
            self,
            &operations_and_actions,
            options.symmetry.tolerances.operation,
            options.symmetry.tolerances.representation,
        )
        .map_err(|error| TbError::IrrepBasisCorepresentation {
            reason: error.to_string(),
        })?;

        // A little-group eigenspace check alone cannot certify H: for a
        // one-dimensional basis the full band subspace is invariant under any
        // scalar sewing matrix, even when anisotropic hopping breaks the
        // requested crystal rotation. Certify every target operation on the
        // complete real-space Hamiltonian before starting any eigensolver.
        let support = validate_hamiltonian(self, options.symmetry.tolerances.hermiticity)?;
        let hamiltonian_operation_diagnostics = prepared_operations
            .par_iter()
            .enumerate()
            .map(|(operation_index, prepared)| {
                let residual = hamiltonian_residual(
                    &support,
                    &prepared.operation,
                    &prepared.action,
                    options.symmetry.tolerances,
                )
                .map_err(|error| TbError::IrrepCalculation {
                    message: format!(
                        "target operation {operation_index} Hamiltonian covariance failed: {error}"
                    ),
                })?;
                Ok(IrrepHamiltonianOperationDiagnostic {
                    operation_index,
                    rotation: prepared.operation.rotation,
                    translation: prepared.operation.translation,
                    time_reversal: prepared.operation.time_reversal,
                    max_absolute_residual: residual.max_absolute,
                    max_relative_residual: residual.max_relative,
                    relative_frobenius_residual: residual.relative_frobenius,
                    acceptance_threshold: residual.acceptance_threshold,
                    preserved: residual.max_absolute <= residual.acceptance_threshold,
                })
            })
            .collect::<Vec<Result<_>>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        let target_hamiltonian_compatible = hamiltonian_operation_diagnostics
            .iter()
            .all(|diagnostic| diagnostic.preserved);
        let target_hamiltonian_diagnostic = (!target_hamiltonian_compatible).then(|| {
            let broken_count = hamiltonian_operation_diagnostics
                .iter()
                .filter(|diagnostic| !diagnostic.preserved)
                .count();
            let max_residual = hamiltonian_operation_diagnostics
                .iter()
                .map(|diagnostic| diagnostic.max_absolute_residual)
                .fold(0.0_f64, f64::max);
            format!(
                "the real-space Hamiltonian breaks {broken_count} target magnetic operations (maximum covariance residual {max_residual:e})"
            )
        });

        let mut points = Vec::new();
        for point in summary.kpoints {
            let is_point = point_is_isolated(summary.unitary_sg, &point);
            if !is_point && !options.include_k_manifolds {
                continue;
            }
            let (kx, ky, kz, denominator) = point.coords;
            if denominator == 0 {
                return Err(TbError::MagneticIrrepAnalysis {
                    uni,
                    message: format!("k point {} has a zero denominator", point.label),
                });
            }
            let denominator = f64::from(denominator);
            let canonical_coordinate = [
                f64::from(kx) / denominator,
                f64::from(ky) / denominator,
                f64::from(kz) / denominator,
            ];
            let model_coordinate =
                row_vector_times_matrix(canonical_coordinate, &input_to_data.basis);
            let operations =
                map_point_operations(&point, &prepared_operations, operation_tolerance)?;
            let mut formal_coreps = prepare_formal_coreps::<SPIN>(summary.unitary_sg, &point)?;
            let point_failures = unresolved_coreps
                .iter()
                .filter(|failure| failure.spinor == SPIN && failure.k_label == point.label)
                .cloned()
                .collect::<Vec<_>>();
            let mut unresolved_coreps = Vec::new();
            let mut recovered_coreps = Vec::new();
            for failure in point_failures {
                match recover_complex_formal_corep::<SPIN>(summary.unitary_sg, &point, &failure)? {
                    Some(corep) => recovered_coreps.push(corep),
                    None => unresolved_coreps.push(failure),
                }
            }
            formal_coreps.extend(merge_recovered_formal_coreps(recovered_coreps));
            points.push(PreparedKPoint {
                summary: point,
                canonical_coordinate,
                model_coordinate,
                is_point,
                operations,
                formal_coreps,
                unresolved_coreps,
            });
        }

        let point_results = points
            .par_iter()
            .map(|point| {
                calculate_kpoint(
                    self,
                    point,
                    &prepared_operations,
                    target_hamiltonian_diagnostic.as_deref(),
                    &options,
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        Ok(IrrepCalculationReport {
            uni_number: uni,
            bns_number: identification.bns_number,
            spinful: SPIN,
            target_hamiltonian_compatible,
            hamiltonian_operation_diagnostics,
            high_symmetry_kpoints: point_results,
        })
    }
}

fn validate_options(options: &IrrepCalculationOptions) -> Result<()> {
    options.symmetry.structural_parameters.validate()?;
    validate_tolerances(options.symmetry.tolerances)?;
    for (name, value) in [
        ("degeneracy_absolute", options.degeneracy_absolute),
        ("degeneracy_relative", options.degeneracy_relative),
        ("subspace_tolerance", options.subspace_tolerance),
        ("character_tolerance", options.character_tolerance),
        ("integer_tolerance", options.integer_tolerance),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(TbError::IrrepCalculation {
                message: format!("{name} must be finite and positive"),
            });
        }
    }
    Ok(())
}

fn exact_transformed_translation(
    transform: &SettingTransform,
    operation: &CrystalSymmetryOperation,
    data_rotation: &[[i32; 3]; 3],
) -> [f64; 3] {
    std::array::from_fn(|row| {
        transform.origin[row]
            - (0..3)
                .map(|column| f64::from(data_rotation[row][column]) * transform.origin[column])
                .sum::<f64>()
            + (0..3)
                .map(|column| transform.basis[row][column] * operation.translation[column])
                .sum::<f64>()
    })
}

fn row_vector_times_matrix(vector: [f64; 3], matrix: &[[f64; 3]; 3]) -> [f64; 3] {
    std::array::from_fn(|column| (0..3).map(|row| vector[row] * matrix[row][column]).sum())
}

fn translations_equivalent(left: [f64; 3], right: [f64; 3], tolerance: f64) -> bool {
    left.into_iter().zip(right).all(|(left, right)| {
        let difference = left - right;
        (difference - difference.round()).abs() <= tolerance
    })
}

fn map_point_operations(
    point: &MagneticKPointSummary,
    prepared: &[PreparedOperation],
    tolerance: f64,
) -> Result<Vec<PointOperation>> {
    point
        .operations
        .iter()
        .map(|operation| {
            let matches = prepared
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    candidate.data_rotation == operation.rotation
                        && candidate.operation.time_reversal == operation.time_reversal
                        && translations_equivalent(
                            candidate.data_translation_modulo,
                            operation.translation,
                            tolerance,
                        )
                })
                .collect::<Vec<_>>();
            if matches.len() != 1 {
                return Err(TbError::IrrepCalculation {
                    message: format!(
                        "k point {} operation column {} maps to {} target operations instead of exactly one",
                        point.label,
                        operation.column,
                        matches.len()
                    ),
                });
            }
            let (prepared_index, candidate) = matches[0];
            let shift = std::array::from_fn(|axis| {
                candidate.data_translation_exact[axis] - operation.translation[axis]
            });
            if shift
                .iter()
                .any(|value| (*value - value.round()).abs() > tolerance)
            {
                return Err(TbError::IrrepCalculation {
                    message: format!(
                        "k point {} operation column {} has a non-integral setting translation shift {shift:?}",
                        point.label, operation.column
                    ),
                });
            }
            Ok(PointOperation {
                prepared_index,
                character_column: operation.column,
                lattice_shift_in_data_frame: shift.map(f64::round),
            })
        })
        .collect()
}

fn point_is_isolated(unitary_sg: u8, point: &MagneticKPointSummary) -> bool {
    let (kx, ky, kz, kd) = point.coords;
    cryspglib::irrep::query::irreps_of(unitary_sg)
        .iter()
        .any(|irrep| {
            irrep.kx == kx && irrep.ky == ky && irrep.kz == kz && irrep.kd == kd && irrep.is_point()
        })
}

fn source_record_matches(
    irrep: &IrrepRecord,
    source_sg: u8,
    source_ml: &str,
    source_spinor: bool,
    unitary_sg: u8,
    coords: (i8, i8, i8, i8),
) -> bool {
    irrep.sg == source_sg
        && irrep.sg == unitary_sg
        && irrep.ml == source_ml
        && irrep.spinor == source_spinor
        && (irrep.kx, irrep.ky, irrep.kz, irrep.kd) == coords
}

fn prepare_formal_coreps<const SPIN: bool>(
    unitary_sg: u8,
    point: &MagneticKPointSummary,
) -> Result<Vec<FormalCorep>> {
    let irreps = cryspglib::irrep::query::irreps_of(unitary_sg);
    point
        .coreps
        .iter()
        .filter(|corep| {
            !corep.source_irreps.is_empty()
                && corep
                    .source_irreps
                    .iter()
                    .all(|source| source.spinor == SPIN)
        })
        .map(|corep| {
            let expected_sources = corep_source_arity(corep.corep_type);
            if corep.source_irreps.len() != expected_sources {
                return Err(formal_corep_error(
                    unitary_sg,
                    point,
                    &corep.label,
                    None,
                    format!(
                        "corep type {:?} requires {expected_sources} source irreps, found {}",
                        corep.corep_type,
                        corep.source_irreps.len()
                    ),
                ));
            }
            let sources = corep
                .source_irreps
                .iter()
                .map(|source| {
                    if source.spinor != SPIN {
                        return Err(formal_corep_error(
                            unitary_sg,
                            point,
                            &corep.label,
                            Some(source.ml),
                            format!(
                                "source spinor={} disagrees with requested SPIN={SPIN}",
                                source.spinor
                            ),
                        ));
                    }
                    let matches = irreps
                        .iter()
                        .filter(|irrep| {
                            source_record_matches(
                                irrep,
                                source.sg,
                                source.ml,
                                source.spinor,
                                unitary_sg,
                                point.coords,
                            )
                        })
                        .collect::<Vec<_>>();
                    let [irrep] = matches.as_slice() else {
                        return Err(formal_corep_error(
                            unitary_sg,
                            point,
                            &corep.label,
                            Some(source.ml),
                            format!(
                                "source lookup matched {} records for SG {} at the exact k tuple {:?}",
                                matches.len(),
                                source.sg,
                                point.coords
                            ),
                        ));
                    };
                    let typed = typed_source_characters(*irrep, point, unitary_sg, &corep.label)?;
                    if source.dim as usize != typed.dimension {
                        return Err(formal_corep_error(
                            unitary_sg,
                            point,
                            &corep.label,
                            Some(source.ml),
                            format!(
                                "summary source dimension {} disagrees with typed dimension {}",
                                source.dim, typed.dimension
                            ),
                        ));
                    }
                    Ok(typed)
                })
                .collect::<Result<Vec<_>>>()?;
            let (dimension, factor) = corep_shape(
                corep.corep_type,
                &sources.iter().map(|source| source.dimension).collect::<Vec<_>>(),
            )
            .map_err(|detail| {
                formal_corep_error(unitary_sg, point, &corep.label, None, detail)
            })?;
            if dimension == 0 || corep.dim != dimension {
                return Err(formal_corep_error(
                    unitary_sg,
                    point,
                    &corep.label,
                    None,
                    format!(
                        "corep dimension {} disagrees with typed expected dimension {}",
                        corep.dim, dimension
                    ),
                ));
            }
            let characters = combine_formal_characters(
                corep.corep_type,
                &sources,
                &point.operations,
                factor,
            )
            .map_err(|detail| formal_corep_error(unitary_sg, point, &corep.label, None, detail))?;
            Ok(FormalCorep {
                label: corep.label.clone(),
                dimension,
                characters,
            })
        })
        .collect()
}

fn recover_complex_formal_corep<const SPIN: bool>(
    unitary_sg: u8,
    point: &MagneticKPointSummary,
    failure: &UnresolvedMagneticCorep,
) -> Result<Option<RecoveredFormalCorep>> {
    let Some(corep_type) = failure.classified_type else {
        return Ok(None);
    };
    let matches = cryspglib::irrep::query::irreps_of(unitary_sg)
        .iter()
        .filter(|irrep| {
            source_record_matches(
                irrep,
                failure.sg,
                &failure.source_irrep,
                failure.spinor,
                unitary_sg,
                point.coords,
            )
        })
        .collect::<Vec<_>>();
    let [irrep] = matches.as_slice() else {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            &failure.source_irrep,
            Some(&failure.source_irrep),
            format!(
                "classified complex source lookup matched {} records instead of one",
                matches.len()
            ),
        ));
    };
    if irrep.spinor != SPIN {
        return Ok(None);
    }
    let complex = match irrep.complex_corepresentation(failure.uni) {
        Ok(corep) => corep,
        // A completed legacy classification can still lack an exact complex
        // transport (for example a non-identity spin Type-C representative).
        // Preserve that source as unresolved instead of guessing a row.
        Err(_) => return Ok(None),
    };
    if complex.corep_type != corep_type {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            &failure.source_irrep,
            Some(&failure.source_irrep),
            format!(
                "complex API returned type {:?}, but the partial summary classified {:?}",
                complex.corep_type, corep_type
            ),
        ));
    }
    if failure
        .wigner_source
        .is_some_and(|source| source != complex.source)
    {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            &failure.source_irrep,
            Some(&failure.source_irrep),
            format!(
                "complex API returned Wigner source {:?}, but the partial summary classified {:?}",
                complex.source, failure.wigner_source
            ),
        ));
    }
    if failure.classified_dimension != Some(complex.dim) {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            &failure.source_irrep,
            Some(&failure.source_irrep),
            format!(
                "complex dimension {} disagrees with classified dimension {:?}",
                complex.dim, failure.classified_dimension
            ),
        ));
    }
    if let Some(source_dimension) = failure.minimum_dimension {
        let expected_dimension = match corep_type {
            cryspglib::irrep::corep::CorepType::A => Some(source_dimension),
            cryspglib::irrep::corep::CorepType::B | cryspglib::irrep::corep::CorepType::C => {
                source_dimension.checked_mul(2)
            }
        };
        if expected_dimension != Some(complex.dim) {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                &failure.source_irrep,
                Some(&failure.source_irrep),
                format!(
                    "complex dimension {} disagrees with selected-arm source dimension {} for {:?}",
                    complex.dim, source_dimension, corep_type
                ),
            ));
        }
    }
    let characters =
        match_complex_corep_to_point(&complex, point, unitary_sg, &failure.source_irrep)?;
    Ok(Some(RecoveredFormalCorep {
        source_label: failure.source_irrep.clone(),
        corep_type,
        formal: FormalCorep {
            label: failure.source_irrep.clone(),
            dimension: complex.dim,
            characters,
        },
    }))
}

fn match_complex_corep_to_point(
    corep: &cryspglib::irrep::corep::ComplexCorepresentation,
    point: &MagneticKPointSummary,
    unitary_sg: u8,
    source_label: &str,
) -> Result<Vec<Option<Complex64>>> {
    let column_count = corep.characters.len();
    if corep.timerev.len() != column_count
        || corep.magnetic_operation_indices.len() != column_count
        || corep.operations.len() != column_count
        || point.operations.len() != column_count
    {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            source_label,
            Some(source_label),
            format!(
                "complex API parallel column lengths ({column_count}, {}, {}, {}) disagree with the summary length {}",
                corep.timerev.len(),
                corep.magnetic_operation_indices.len(),
                corep.operations.len(),
                point.operations.len()
            ),
        ));
    }

    let mut used = vec![false; column_count];
    let mut characters = Vec::with_capacity(column_count);
    for (column, target) in point.operations.iter().enumerate() {
        if target.column != column {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                source_label,
                Some(source_label),
                format!(
                    "summary operation at position {column} declares column {}",
                    target.column
                ),
            ));
        }
        let matches = corep
            .magnetic_operation_indices
            .iter()
            .enumerate()
            .filter(|(_, index)| **index == target.magnetic_operation_index)
            .map(|(position, _)| position)
            .collect::<Vec<_>>();
        let [position] = matches.as_slice() else {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                source_label,
                Some(source_label),
                format!(
                    "summary column {column} magnetic operation {} matched {} complex API columns",
                    target.magnetic_operation_index,
                    matches.len()
                ),
            ));
        };
        if std::mem::replace(&mut used[*position], true) {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                source_label,
                Some(source_label),
                format!("complex API column {position} was matched more than once"),
            ));
        }
        let operation = &corep.operations[*position];
        if corep.timerev[*position] != target.time_reversal
            || operation.time_reversal != target.time_reversal
            || operation.rotation != target.rotation
            || operation.translation != target.translation
        {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                source_label,
                Some(source_label),
                format!(
                    "complex API column {position} does not describe summary operation column {column}"
                ),
            ));
        }
        let value = corep.characters[*position];
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                source_label,
                Some(source_label),
                format!("complex API column {position} is non-finite: {value}"),
            ));
        }
        characters.push((!target.time_reversal).then_some(value));
    }
    if used.iter().any(|used| !used) {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            source_label,
            Some(source_label),
            "complex API contains an unmatched magnetic little-group column".to_string(),
        ));
    }
    Ok(characters)
}

fn merge_recovered_formal_coreps(recovered: Vec<RecoveredFormalCorep>) -> Vec<FormalCorep> {
    let mut formal = Vec::<FormalCorep>::new();
    let mut type_c_groups = Vec::<(usize, Vec<String>)>::new();

    for recovered in recovered {
        if recovered.corep_type != cryspglib::irrep::corep::CorepType::C {
            formal.push(recovered.formal);
            continue;
        }
        if let Some((_, labels)) = type_c_groups.iter_mut().find(|(formal_index, _)| {
            formal_corep_rows_equivalent(&formal[*formal_index], &recovered.formal)
        }) {
            labels.push(recovered.source_label);
            continue;
        }
        let formal_index = formal.len();
        formal.push(recovered.formal);
        type_c_groups.push((formal_index, vec![recovered.source_label]));
    }

    for (formal_index, mut labels) in type_c_groups {
        labels.sort();
        labels.dedup();
        formal[formal_index].label = labels.join(" + ");
    }
    formal
}

fn formal_corep_rows_equivalent(left: &FormalCorep, right: &FormalCorep) -> bool {
    let dimension = left.dimension;
    left.dimension == right.dimension
        && left.characters.len() == right.characters.len()
        && left
            .characters
            .iter()
            .zip(&right.characters)
            .all(
                |(left_value, right_value)| match (left_value, right_value) {
                    (None, None) => true,
                    (Some(left), Some(right)) => {
                        representational_roundoff_equal(left.re, right.re, dimension)
                            && representational_roundoff_equal(left.im, right.im, dimension)
                    }
                    _ => false,
                },
            )
}

fn representational_roundoff_equal(left: f64, right: f64, dimension: usize) -> bool {
    let scale = (dimension as f64).max(1.0).max(left.abs()).max(right.abs());
    (left - right).abs() <= 8.0 * f64::EPSILON * scale
}

fn corep_source_arity(corep_type: cryspglib::irrep::corep::CorepType) -> usize {
    match corep_type {
        cryspglib::irrep::corep::CorepType::A | cryspglib::irrep::corep::CorepType::B => 1,
        cryspglib::irrep::corep::CorepType::C => 2,
    }
}

fn corep_shape(
    corep_type: cryspglib::irrep::corep::CorepType,
    source_dimensions: &[usize],
) -> std::result::Result<(usize, usize), String> {
    let expected_sources = corep_source_arity(corep_type);
    if source_dimensions.len() != expected_sources {
        return Err(format!(
            "corep type {corep_type:?} requires {expected_sources} source irreps, found {}",
            source_dimensions.len()
        ));
    }
    let sum_dimension = source_dimensions
        .iter()
        .try_fold(0usize, |sum, dimension| {
            sum.checked_add(*dimension)
                .ok_or_else(|| "typed source dimensions overflowed usize".to_string())
        })?;
    let (dimension, factor) = match corep_type {
        cryspglib::irrep::corep::CorepType::A | cryspglib::irrep::corep::CorepType::C => {
            (sum_dimension, 1)
        }
        cryspglib::irrep::corep::CorepType::B => sum_dimension
            .checked_mul(2)
            .map(|dimension| (dimension, 2))
            .ok_or_else(|| "Type-B corep dimension overflowed usize".to_string())?,
    };
    if dimension == 0 {
        return Err("corep has zero typed dimension".to_string());
    }
    Ok((dimension, factor))
}

fn combine_formal_characters(
    corep_type: cryspglib::irrep::corep::CorepType,
    sources: &[TypedSourceCharacters],
    operations: &[cryspglib::irrep::magnetic_summary::MagneticLittleGroupOperation],
    factor: usize,
) -> std::result::Result<Vec<Option<Complex64>>, String> {
    if sources.len() != corep_source_arity(corep_type) {
        return Err(format!(
            "corep type {corep_type:?} requires {} typed source rows, found {}",
            corep_source_arity(corep_type),
            sources.len()
        ));
    }
    if sources
        .iter()
        .any(|source| source.values.len() != operations.len())
    {
        return Err("typed source and magnetic operation lengths differ".to_string());
    }
    operations
        .iter()
        .enumerate()
        .map(|(column, operation)| {
            if operation.time_reversal {
                return Ok(None);
            }
            let mut value = Complex64::new(0.0, 0.0);
            for source in sources {
                value += source.values[column].ok_or_else(|| {
                    format!("unitary character column {column} has no typed source value")
                })?;
            }
            value *= factor as f64;
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(format!(
                    "unitary character column {column} is non-finite: {value}"
                ));
            }
            Ok(Some(value))
        })
        .collect()
}

struct TypedSourceCharacters {
    dimension: usize,
    values: Vec<Option<Complex64>>,
}

fn formal_corep_error(
    unitary_sg: u8,
    point: &MagneticKPointSummary,
    corep_label: &str,
    source: Option<&str>,
    detail: String,
) -> TbError {
    TbError::IrrepCalculation {
        message: format!(
            "SG {unitary_sg} k-point {} corep {}{}: {detail}",
            point.label,
            corep_label,
            source.map_or_else(String::new, |source| format!(" source {source}"))
        ),
    }
}

fn typed_source_characters(
    irrep: &'static IrrepRecord,
    point: &MagneticKPointSummary,
    unitary_sg: u8,
    corep_label: &str,
) -> Result<TypedSourceCharacters> {
    let (dimension, entries): (usize, Vec<(SeitzOperation, Complex64)>) = if irrep.spinor {
        let row = irrep.spinor_selected_arm_view().map_err(|error| {
            formal_corep_error(
                unitary_sg,
                point,
                corep_label,
                Some(irrep.ml),
                format!("typed spinor row failed: {error}"),
            )
        })?;
        if row.values().len() != row.operations().len() {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                corep_label,
                Some(irrep.ml),
                "typed spinor values and operations have different lengths".to_string(),
            ));
        }
        (
            row.dimension(),
            row.values()
                .iter()
                .copied()
                .zip(row.operations().iter().map(|operation| operation.seitz))
                .map(|(value, operation)| (operation, value))
                .collect(),
        )
    } else {
        let row = match irrep.ordinary_scalar_selected_arm_block_trace() {
            Ok(row) => row,
            Err(CharacterViewError::NotApplicable) => irrep
                .compound_selected_arm_view()
                .map_err(|error| {
                    formal_corep_error(
                        unitary_sg,
                        point,
                        corep_label,
                        Some(irrep.ml),
                        format!("typed compound row failed: {error}"),
                    )
                })?
                .block_trace()
                .clone(),
            Err(error) => {
                return Err(formal_corep_error(
                    unitary_sg,
                    point,
                    corep_label,
                    Some(irrep.ml),
                    format!("typed scalar row failed: {error}"),
                ));
            }
        };
        if row.values().len() != row.operations().len() {
            return Err(formal_corep_error(
                unitary_sg,
                point,
                corep_label,
                Some(irrep.ml),
                "typed scalar values and operations have different lengths".to_string(),
            ));
        }
        (
            row.dimension(),
            row.values()
                .iter()
                .copied()
                .zip(row.operations().iter().copied())
                .map(|(value, operation)| (operation, value))
                .collect(),
        )
    };
    if dimension == 0
        || entries.iter().any(|(operation, value)| {
            !value.re.is_finite()
                || !value.im.is_finite()
                || operation.translation.iter().any(|value| !value.is_finite())
        })
    {
        return Err(formal_corep_error(
            unitary_sg,
            point,
            corep_label,
            Some(irrep.ml),
            "typed source contains zero dimension or non-finite value".to_string(),
        ));
    }
    let values = match_typed_entries_to_point(&entries, point, unitary_sg, corep_label, irrep.ml)?;
    Ok(TypedSourceCharacters { dimension, values })
}

fn match_typed_entries_to_point(
    entries: &[(SeitzOperation, Complex64)],
    point: &MagneticKPointSummary,
    unitary_sg: u8,
    corep_label: &str,
    source_label: &str,
) -> Result<Vec<Option<Complex64>>> {
    point
        .operations
        .iter()
        .enumerate()
        .map(|(column, target)| {
            if target.time_reversal {
                return Ok(None);
            }
            if target.translation.iter().any(|value| !value.is_finite()) {
                return Err(formal_corep_error(
                    unitary_sg,
                    point,
                    corep_label,
                    Some(source_label),
                    format!("unitary column {column} has a non-finite translation"),
                ));
            }
            let matches = entries
                .iter()
                .filter(|(operation, _)| {
                    operation
                        .rotation
                        .into_iter()
                        .eq(target.rotation.into_iter().flatten())
                        && operation.translation == target.translation
                })
                .collect::<Vec<_>>();
            let [(_, value)] = matches.as_slice() else {
                return Err(formal_corep_error(
                    unitary_sg,
                    point,
                    corep_label,
                    Some(source_label),
                    format!(
                        "unitary column {column} matched {} typed complete Seitz operations",
                        matches.len()
                    ),
                ));
            };
            Ok(Some(*value))
        })
        .collect()
}

fn calculate_kpoint<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    point: &PreparedKPoint,
    prepared_operations: &[PreparedOperation],
    target_hamiltonian_diagnostic: Option<&str>,
    options: &IrrepCalculationOptions,
) -> Result<IrrepKPointReport> {
    let k = Array1::from_vec(point.model_coordinate.to_vec());
    let hamiltonian = model.gen_ham(&k, Gauge::Lattice);
    let (energies, eigenvectors) = hamiltonian.eigh(UPLO::Lower)?;
    let groups = degeneracy_groups(
        energies.as_slice().expect("eigenvalues are contiguous"),
        options.degeneracy_absolute,
        options.degeneracy_relative,
    );
    let mut bands = Vec::with_capacity(groups.len());
    for (start, end) in groups {
        let vectors = eigenvectors.slice(s![.., start..end]).to_owned();
        let dimension = end - start;
        let mut characters = Vec::with_capacity(point.operations.len());
        for (summary_operation, mapped) in point.summary.operations.iter().zip(&point.operations) {
            debug_assert_eq!(summary_operation.column, mapped.character_column);
            let prepared = &prepared_operations[mapped.prepared_index];
            let image_k = image_momentum(
                point.model_coordinate,
                &prepared.operation.rotation,
                prepared.operation.time_reversal,
            )?;
            let action = prepared
                .action
                .lattice_gauge_matrix(image_k)
                .map_err(|error| TbError::IrrepCalculation {
                    message: format!(
                        "k point {} operation column {} sewing matrix failed: {error}",
                        point.summary.label, summary_operation.column
                    ),
                })?;
            let source_vectors = if prepared.operation.time_reversal {
                vectors.mapv(|value| value.conj())
            } else {
                vectors.clone()
            };
            let transformed = action.dot(&source_vectors);
            let projected = vectors.t().mapv(|value| value.conj()).dot(&transformed);
            let reconstructed = vectors.dot(&projected);
            let leakage =
                frobenius_norm_difference(&transformed, &reconstructed) / (dimension as f64).sqrt();
            let gram = projected.t().mapv(|value| value.conj()).dot(&projected);
            let unitarity = identity_residual(&gram) / (dimension as f64).sqrt();
            let value = if summary_operation.time_reversal {
                None
            } else {
                let trace = (0..dimension)
                    .map(|index| projected[[index, index]])
                    .sum::<Complex64>();
                let phase_argument = point
                    .canonical_coordinate
                    .iter()
                    .zip(mapped.lattice_shift_in_data_frame)
                    .map(|(component, shift)| component * shift)
                    .sum::<f64>();
                // x_data=P*x_input+s can change the chosen Seitz representative
                // by a lattice translation N.  The database representative is
                // recovered from the exact transformed action by exp(+ik.N).
                let setting_phase =
                    Complex64::new(0.0, std::f64::consts::TAU * phase_argument).exp();
                Some(setting_phase * trace)
            };
            characters.push(IrrepCharacter {
                column: summary_operation.column,
                rotation: summary_operation.rotation,
                translation: summary_operation.translation,
                time_reversal: summary_operation.time_reversal,
                value,
                subspace_leakage: leakage,
                projected_unitarity_residual: unitarity,
            });
        }

        let max_subspace_leakage = characters
            .iter()
            .map(|character| character.subspace_leakage)
            .fold(0.0_f64, f64::max);
        let max_projected_unitarity_residual = characters
            .iter()
            .map(|character| character.projected_unitarity_residual)
            .fold(0.0_f64, f64::max);
        let fit = fit_corepresentations(&point.formal_coreps, &characters, dimension, options)?;
        let mut diagnostics = fit.diagnostics;
        diagnostics.extend(
            point
                .unresolved_coreps
                .iter()
                .filter(|failure| failure.minimum_dimension.is_none_or(|minimum| dimension >= minimum))
                .map(|failure| {
                    format!(
                        "database corep {} is unavailable at {} and may contribute to this {}-band cluster: {}",
                        failure.source_irrep, point.summary.label, dimension, failure.reason
                    )
                }),
        );
        if let Some(diagnostic) = target_hamiltonian_diagnostic {
            diagnostics.push(diagnostic.to_string());
        }
        if max_subspace_leakage > options.subspace_tolerance {
            diagnostics.push(format!(
                "symmetry-transformed eigenspace leakage {max_subspace_leakage:e} exceeds {:e}",
                options.subspace_tolerance
            ));
        }
        if max_projected_unitarity_residual > options.subspace_tolerance {
            diagnostics.push(format!(
                "projected sewing unitarity residual {max_projected_unitarity_residual:e} exceeds {:e}",
                options.subspace_tolerance
            ));
        }
        let label = if diagnostics.is_empty() {
            fit.label
        } else {
            "???".to_string()
        };
        bands.push(IrrepBandReport {
            band_start: start + 1,
            band_end: end,
            energies: energies.slice(s![start..end]).to_vec(),
            label,
            multiplicities: fit.multiplicities,
            characters,
            character_fit_residual: fit.residual,
            decomposition_rank: fit.rank,
            max_subspace_leakage,
            max_projected_unitarity_residual,
            diagnostics,
        });
    }
    Ok(IrrepKPointReport {
        label: point.summary.label.clone(),
        canonical_coordinate: point.canonical_coordinate,
        model_coordinate: point.model_coordinate,
        is_point: point.is_point,
        bands,
    })
}

fn degeneracy_groups(energies: &[f64], absolute: f64, relative: f64) -> Vec<(usize, usize)> {
    if energies.is_empty() {
        return Vec::new();
    }
    let spectral_width = (energies[energies.len() - 1] - energies[0]).abs();
    let mut groups = Vec::new();
    let mut start = 0;
    for boundary in 1..energies.len() {
        if (energies[boundary] - energies[boundary - 1]).abs()
            > absolute + relative * spectral_width
        {
            groups.push((start, boundary));
            start = boundary;
        }
    }
    groups.push((start, energies.len()));
    groups
}

fn inverse_three(matrix: &[[i32; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let m = matrix.map(|row| row.map(f64::from));
    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !determinant.is_finite() || determinant.abs() < 1e-14 {
        return Err(TbError::IrrepCalculation {
            message: "a target symmetry operation has a singular rotation".to_string(),
        });
    }
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / determinant,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / determinant,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / determinant,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / determinant,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / determinant,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / determinant,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / determinant,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / determinant,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / determinant,
        ],
    ])
}

fn image_momentum(
    momentum: [f64; 3],
    rotation: &[[i32; 3]; 3],
    time_reversal: bool,
) -> Result<[f64; 3]> {
    let inverse = inverse_three(rotation)?;
    let sign = if time_reversal { -1.0 } else { 1.0 };
    Ok(std::array::from_fn(|row| {
        sign * (0..3)
            .map(|column| inverse[column][row] * momentum[column])
            .sum::<f64>()
    }))
}

fn frobenius_norm_difference(left: &Array2<Complex64>, right: &Array2<Complex64>) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left - *right).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn identity_residual(matrix: &Array2<Complex64>) -> f64 {
    matrix
        .indexed_iter()
        .map(|((row, column), value)| {
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            (*value - expected).norm_sqr()
        })
        .sum::<f64>()
        .sqrt()
}

struct CorepFit {
    label: String,
    multiplicities: Vec<IrrepMultiplicity>,
    residual: f64,
    rank: usize,
    diagnostics: Vec<String>,
}

fn fit_corepresentations(
    coreps: &[FormalCorep],
    numerical_characters: &[IrrepCharacter],
    band_dimension: usize,
    options: &IrrepCalculationOptions,
) -> Result<CorepFit> {
    let unitary_columns = numerical_characters
        .iter()
        .filter(|character| !character.time_reversal)
        .collect::<Vec<_>>();
    let mut diagnostics = Vec::new();
    if coreps.is_empty() {
        diagnostics
            .push("the database has no matching corepresentations at this k point".to_string());
        return Ok(CorepFit {
            label: "???".to_string(),
            multiplicities: Vec::new(),
            residual: f64::INFINITY,
            rank: 0,
            diagnostics,
        });
    }
    if unitary_columns.is_empty() {
        diagnostics.push("the magnetic little group has no unitary character columns".to_string());
        return Ok(CorepFit {
            label: "???".to_string(),
            multiplicities: Vec::new(),
            residual: f64::INFINITY,
            rank: 0,
            diagnostics,
        });
    }

    let mut character_matrix = Array2::<Complex64>::zeros((unitary_columns.len(), coreps.len()));
    let mut numerical = Array1::<Complex64>::zeros(unitary_columns.len());
    for (row, numerical_character) in unitary_columns.iter().enumerate() {
        numerical[row] = numerical_character
            .value
            .expect("unitary character has a trace");
        for (column, corep) in coreps.iter().enumerate() {
            let formal = corep
                .characters
                .get(numerical_character.column)
                .copied()
                .flatten()
                .ok_or_else(|| TbError::IrrepCalculation {
                    message: format!(
                        "formal corep {} has no character column {}",
                        corep.label, numerical_character.column
                    ),
                })?;
            character_matrix[[row, column]] = formal;
        }
    }
    let least_squares = character_matrix.least_squares(&numerical)?;
    let rank = usize::try_from(least_squares.rank).unwrap_or(0);
    let multiplicities = coreps
        .iter()
        .zip(least_squares.solution.iter())
        .map(|(corep, value)| {
            let nearest = value.re.round();
            let rounded = (value.im.abs() <= options.integer_tolerance
                && value.re >= -options.integer_tolerance
                && (value.re - nearest).abs() <= options.integer_tolerance)
                .then(|| nearest.max(0.0) as usize);
            IrrepMultiplicity {
                label: corep.label.clone(),
                value: *value,
                rounded,
            }
        })
        .collect::<Vec<_>>();
    if rank != coreps.len() {
        diagnostics.push(format!(
            "formal unitary character matrix has rank {rank}, smaller than {} candidate coreps",
            coreps.len()
        ));
    }
    if multiplicities
        .iter()
        .any(|multiplicity| multiplicity.rounded.is_none())
    {
        diagnostics.push(
            "one or more fitted corep multiplicities are not non-negative integers".to_string(),
        );
    }
    let rounded = Array1::from_iter(
        multiplicities
            .iter()
            .map(|multiplicity| Complex64::new(multiplicity.rounded.unwrap_or(0) as f64, 0.0)),
    );
    let reconstructed = character_matrix.dot(&rounded);
    let residual = reconstructed
        .iter()
        .zip(&numerical)
        .map(|(reconstructed, numerical)| (*reconstructed - *numerical).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / frobenius_vector_norm(&numerical).max(1.0);
    if residual > options.character_tolerance {
        diagnostics.push(format!(
            "relative character reconstruction residual {residual:e} exceeds {:e}",
            options.character_tolerance
        ));
    }
    let reconstructed_dimension = coreps
        .iter()
        .zip(&multiplicities)
        .map(|(corep, multiplicity)| corep.dimension * multiplicity.rounded.unwrap_or(0))
        .sum::<usize>();
    if reconstructed_dimension != band_dimension {
        diagnostics.push(format!(
            "integer corep sum has dimension {reconstructed_dimension}, but the band cluster has dimension {band_dimension}"
        ));
    }
    let labels = coreps
        .iter()
        .zip(&multiplicities)
        .filter_map(|(corep, multiplicity)| {
            let count = multiplicity.rounded?;
            match count {
                0 => None,
                1 => Some(corep.label.clone()),
                _ => Some(format!("{count}*{}", corep.label)),
            }
        })
        .collect::<Vec<_>>();
    if labels.is_empty() {
        diagnostics.push("no nonzero corep multiplicity was identified".to_string());
    }
    Ok(CorepFit {
        label: labels.join(" + "),
        multiplicities,
        residual,
        rank,
        diagnostics,
    })
}

fn frobenius_vector_norm(vector: &Array1<Complex64>) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Atom, AtomType, AtomicOrbitalBasis, CellShiftAction, CrystalSymmetry, NoRMatrix, OrbProj,
        OrbitalId, ScalarSiteBasis,
    };
    use ndarray::{Array2, array};

    fn orbital_cubic<const SPIN: bool>(projections: &[OrbProj]) -> Model<SPIN, 3, NoRMatrix> {
        let orbital_ids = (0..projections.len())
            .map(OrbitalId::new)
            .collect::<Vec<_>>();
        let mut model = Model::tb_model(
            Array2::eye(3),
            Array2::zeros((projections.len(), 3)),
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                orbital_ids,
            )]),
        )
        .unwrap();
        model.orb_projection = projections.to_vec();
        model
    }

    fn permuted_orthorhombic_scalar_model() -> Model<false, 3, NoRMatrix> {
        Model::tb_model(
            array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::H,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap()
    }

    fn shifted_cubic_scalar_model() -> Model<false, 3, NoRMatrix> {
        let position = array![0.173, 0.271, 0.337];
        Model::tb_model(
            Array2::eye(3),
            position.clone().insert_axis(ndarray::Axis(0)),
            Some(vec![Atom::with_orbitals(
                position,
                AtomType::H,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap()
    }

    fn binary_hexagonal_scalar_model<const SPIN: bool>() -> Model<SPIN, 3, NoRMatrix> {
        let mut model = Model::tb_model(
            array![
                [1.0, 0.0, 0.0],
                [-0.5, 3.0_f64.sqrt() / 2.0, 0.0],
                [0.0, 0.0, 5.0]
            ],
            array![[0.0, 0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::B, [OrbitalId::new(0)]),
                Atom::with_orbitals(
                    array![1.0 / 3.0, 2.0 / 3.0, 0.0],
                    AtomType::N,
                    [OrbitalId::new(1)],
                ),
            ]),
        )
        .unwrap();
        model.set_onsite(&array![-1.0, 1.0], None);
        model
    }

    struct InvalidIdentityCorepresentation;

    impl BasisSymmetryRepresentation<true, NoRMatrix> for InvalidIdentityCorepresentation {
        fn resolve(
            &self,
            context: BasisActionContext<'_, true, NoRMatrix>,
        ) -> std::result::Result<LocalizedBasisAction, crate::BasisRepresentationError> {
            let mut matrix = Array2::eye(context.model.nsta());
            matrix[[1, 1]] = Complex64::new(-1.0, 0.0);
            Ok(LocalizedBasisAction {
                sectors: vec![CellShiftAction {
                    shift: [0, 0, 0],
                    matrix,
                }],
            })
        }
    }

    struct WrongSpinfulFactorSystem;

    impl BasisSymmetryRepresentation<true, NoRMatrix> for WrongSpinfulFactorSystem {
        fn resolve(
            &self,
            context: BasisActionContext<'_, true, NoRMatrix>,
        ) -> std::result::Result<LocalizedBasisAction, crate::BasisRepresentationError> {
            Ok(LocalizedBasisAction {
                sectors: vec![CellShiftAction {
                    shift: [0, 0, 0],
                    matrix: Array2::eye(context.model.nsta()),
                }],
            })
        }
    }

    #[test]
    fn incomplete_orbital_shell_is_rejected_before_band_labelling() {
        let model = orbital_cubic::<false>(&[OrbProj::px, OrbProj::py]);
        let error = model
            .calculate_irrep(&AtomicOrbitalBasis, None)
            .unwrap_err();
        assert!(matches!(error, TbError::IrrepBasisRepresentation { .. }));
    }

    #[test]
    fn inconsistent_local_actions_are_rejected_before_band_labelling() {
        let model = orbital_cubic::<true>(&[OrbProj::s]);
        let error = model
            .calculate_irrep(&InvalidIdentityCorepresentation, None)
            .unwrap_err();
        assert!(matches!(error, TbError::IrrepBasisCorepresentation { .. }));
    }

    #[test]
    fn spinful_time_reversal_with_positive_square_is_rejected() {
        let model: Model<true, 3, NoRMatrix> = Model::tb_model(
            array![[1.0, 0.1, 0.2], [0.2, 1.3, 0.1], [0.1, 0.3, 1.7]],
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::H,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let error = model
            .calculate_irrep(&WrongSpinfulFactorSystem, None)
            .unwrap_err();
        assert!(matches!(error, TbError::IrrepBasisCorepresentation { .. }));
        assert!(error.to_string().contains("factor-system phase"));
    }

    #[test]
    fn anisotropic_scalar_hopping_invalidates_all_target_labels() {
        let mut model = orbital_cubic::<false>(&[OrbProj::s]);
        model.set_hop(1.0, 0, 0, &array![1_isize, 0, 0], None);
        model.set_hop(2.0, 0, 0, &array![0_isize, 1, 0], None);
        model.set_hop(3.0, 0, 0, &array![0_isize, 0, 1], None);
        model.set_onsite(&array![1e12], None);

        let report = model.calculate_irrep(&AtomicOrbitalBasis, None).unwrap();
        assert!(!report.target_hamiltonian_compatible);
        assert!(
            report
                .hamiltonian_operation_diagnostics
                .iter()
                .any(|diagnostic| !diagnostic.preserved)
        );
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(|band| band.label == "???"),
            "{report}"
        );
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(|band| band.max_subspace_leakage < 1e-12),
            "this regression must exercise the failure of a local leakage-only check"
        );
    }

    #[test]
    fn degeneracy_grouping_is_invariant_under_energy_origin_shifts() {
        let original = degeneracy_groups(&[0.0, 1.0, 2.0], 1e-7, 1e-9);
        let shifted = degeneracy_groups(&[1e12, 1e12 + 1.0, 1e12 + 2.0], 1e-7, 1e-9);
        assert_eq!(original, vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(shifted, original);
    }

    #[test]
    fn nested_symmetry_tolerances_are_validated() {
        let model = orbital_cubic::<false>(&[OrbProj::s]);
        let mut options = IrrepCalculationOptions::default();
        options.symmetry.tolerances.hermiticity = f64::NAN;
        let error = model
            .calculate_irrep(&AtomicOrbitalBasis, Some(&options))
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::InvalidHamiltonianSymmetryInput { .. }
        ));
    }

    #[test]
    fn broken_cubic_hamiltonian_reports_unknown_then_projection_restores_labels() {
        let mut model = orbital_cubic::<false>(&[OrbProj::px, OrbProj::py, OrbProj::pz]);
        model.set_onsite(&array![0.0, 1.0, 2.0], None);
        let target = model
            .magnetic_crystal_symmetry_from_atoms(
                &crate::crystal_symmetry::SymmetryParameters::default(),
            )
            .unwrap();

        let broken = model.calculate_irrep(&AtomicOrbitalBasis, None).unwrap();
        let gamma = broken
            .high_symmetry_kpoints
            .iter()
            .find(|point| point.label == "GM")
            .unwrap();
        assert!(gamma.bands.iter().any(|band| band.label == "???"));
        let broken_text = broken.format_irvsp();
        assert!(broken_text.contains("???"));
        assert!(broken_text.contains("Target-H covariance: BROKEN"));
        assert!(broken_text.contains("chi["));
        assert!(broken_text.contains("N/A (antiunitary)"));

        let symmetrized = model
            .symmetrize_hamiltonian(
                &target,
                &AtomicOrbitalBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        let restored = symmetrized
            .calculate_irrep_for_group(&target, &AtomicOrbitalBasis, None)
            .unwrap();
        let gamma = restored
            .high_symmetry_kpoints
            .iter()
            .find(|point| point.label == "GM")
            .unwrap();
        assert!(gamma.bands.iter().all(IrrepBandReport::is_identified));
    }

    #[test]
    fn spinful_grey_group_uses_spinor_corepresentations() {
        let model = orbital_cubic::<true>(&[OrbProj::s]);
        let report = model.calculate_irrep(&AtomicOrbitalBasis, None).unwrap();
        let gamma = report
            .high_symmetry_kpoints
            .iter()
            .find(|point| point.label == "GM")
            .unwrap();
        assert_eq!(gamma.bands.len(), 1);
        assert_eq!(gamma.bands[0].band_end - gamma.bands[0].band_start + 1, 2);
        assert!(gamma.bands[0].is_identified(), "{report}");
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(IrrepBandReport::is_identified),
            "{report}"
        );
        assert!(
            gamma.bands[0]
                .characters
                .iter()
                .filter(|character| character.time_reversal)
                .all(|character| character.value.is_none())
        );
    }

    #[test]
    fn transformed_setting_and_parallel_collection_are_deterministic() {
        let model = permuted_orthorhombic_scalar_model();
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let first = serial_pool
            .install(|| model.calculate_irrep(&ScalarSiteBasis, None))
            .unwrap();
        let second = parallel_pool
            .install(|| model.calculate_irrep(&ScalarSiteBasis, None))
            .unwrap();
        assert_eq!(first, second);
        assert!(first.high_symmetry_kpoints.len() > 1);
        assert!(
            first
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(IrrepBandReport::is_identified),
            "{first}"
        );
    }

    #[test]
    fn magnetic_field_reidentifies_the_effective_target_group() {
        let model = orbital_cubic::<false>(&[OrbProj::s]);
        let full = model
            .magnetic_crystal_symmetry_from_atoms(
                &crate::crystal_symmetry::SymmetryParameters::default(),
            )
            .unwrap();
        let mut options = IrrepCalculationOptions::default();
        options
            .symmetry
            .structural_parameters
            .external_fields
            .magnetic = Some([0.0, 0.0, 1.0]);
        let report = model
            .calculate_irrep(&AtomicOrbitalBasis, Some(&options))
            .unwrap();
        assert_ne!(report.uni_number, full.uni_number);
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(IrrepBandReport::is_identified),
            "{report}"
        );

        let explicit = model
            .calculate_irrep_for_group(&full, &AtomicOrbitalBasis, Some(&options))
            .unwrap();
        assert_eq!(explicit.uni_number, report.uni_number);
        assert_eq!(explicit.bns_number, report.bns_number);
        assert_eq!(explicit.high_symmetry_kpoints, report.high_symmetry_kpoints);
    }

    #[test]
    fn explicit_target_none_preserves_its_precomputed_field_subset() {
        let model = orbital_cubic::<false>(&[OrbProj::s]);
        let mut parameters = crate::crystal_symmetry::SymmetryParameters::default();
        parameters.external_fields.magnetic = Some([1e-6, 0.0, 0.0]);
        parameters.field_tolerance = 1e-3;
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&parameters)
            .unwrap();
        assert_eq!(
            target.field_preserving_operations.len(),
            target.operations.len()
        );

        let report = model
            .calculate_irrep_for_group(&target, &AtomicOrbitalBasis, None)
            .unwrap();
        assert_eq!(report.uni_number, target.uni_number);
        assert_eq!(
            report.hamiltonian_operation_diagnostics.len(),
            target.field_preserving_operations.len()
        );
    }

    #[test]
    fn nonzero_origin_shift_preserves_database_character_phases() {
        let model = shifted_cubic_scalar_model();
        let report = model.calculate_irrep(&ScalarSiteBasis, None).unwrap();
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(IrrepBandReport::is_identified),
            "{report}"
        );
    }

    #[test]
    fn complex_little_group_characters_are_identified() {
        let model = binary_hexagonal_scalar_model::<false>();
        let report = model.calculate_irrep(&ScalarSiteBasis, None).unwrap();
        assert_eq!(report.uni_number, 1440);
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(IrrepBandReport::is_identified),
            "{report}"
        );
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .flat_map(|band| &band.characters)
                .filter_map(|character| character.value)
                .any(|value| value.im.abs() > 0.5),
            "the scalar regression must exercise a genuinely complex character"
        );
    }

    #[test]
    fn complex_spinor_little_group_characters_are_identified() {
        let model = binary_hexagonal_scalar_model::<true>();
        let report = model.calculate_irrep(&ScalarSiteBasis, None).unwrap();
        assert_eq!(report.uni_number, 1440);
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .all(IrrepBandReport::is_identified),
            "{report}"
        );
        assert!(
            report
                .high_symmetry_kpoints
                .iter()
                .flat_map(|point| &point.bands)
                .flat_map(|band| &band.characters)
                .filter_map(|character| character.value)
                .any(|value| value.im.abs() > 0.5),
            "the spinor regression must exercise a genuinely complex character"
        );
    }

    #[test]
    fn type_i_compound_summary_retains_r1r2_selected_arm_corep() {
        let partial = magnetic_irrep_summary_by_uni_partial(667)
            .expect("UNI 667 partial summary must remain available");
        let point = partial
            .summary
            .kpoints
            .iter()
            .find(|point| point.label == "R")
            .expect("SG 76 has an R point");
        let corep = point
            .coreps
            .iter()
            .find(|corep| corep.label == "R1R2")
            .expect("R1R2 selected-arm compound corep must not be dropped");
        assert_eq!(corep.dim, 2);
        assert!(
            partial
                .unresolved_coreps
                .iter()
                .all(|failure| failure.source_irrep != "R1R2")
        );
    }

    fn synthetic_magnetic_operation(
        column: usize,
        magnetic_operation_index: usize,
        rotation: [[i32; 3]; 3],
        translation: [f64; 3],
        time_reversal: bool,
    ) -> cryspglib::irrep::magnetic_summary::MagneticLittleGroupOperation {
        cryspglib::irrep::magnetic_summary::MagneticLittleGroupOperation {
            column,
            magnetic_operation_index,
            rotation,
            translation,
            time_reversal,
        }
    }

    fn synthetic_point(
        operations: Vec<cryspglib::irrep::magnetic_summary::MagneticLittleGroupOperation>,
    ) -> MagneticKPointSummary {
        let unitary_order = operations
            .iter()
            .filter(|operation| !operation.time_reversal)
            .count();
        MagneticKPointSummary {
            label: "synthetic".to_string(),
            coords: (0, 0, 0, 1),
            little_group_order: operations.len(),
            unitary_order,
            antiunitary_order: operations.len() - unitary_order,
            operations,
            conjugacy_classes: Vec::new(),
            coreps: Vec::new(),
        }
    }

    #[test]
    fn typed_mapping_matches_complete_seitz_operations_exactly() {
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let quarter_turn = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
        let point = synthetic_point(vec![
            synthetic_magnetic_operation(0, 4, identity, [0.5, 0.0, 0.0], false),
            synthetic_magnetic_operation(1, 5, identity, [0.0, 0.0, 0.0], true),
            synthetic_magnetic_operation(2, 6, quarter_turn, [0.0, 0.0, 0.0], false),
        ]);
        let entries = vec![
            (
                SeitzOperation {
                    rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1],
                    translation: [0.0, 0.0, 0.0],
                },
                Complex64::new(3.0, 0.0),
            ),
            (
                SeitzOperation {
                    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    translation: [0.5, 0.0, 0.0],
                },
                Complex64::new(2.0, 0.0),
            ),
        ];
        let values = match_typed_entries_to_point(&entries, &point, 1, "synthetic", "S1")
            .expect("complete Seitz operations should match");
        assert_eq!(
            values,
            vec![
                Some(Complex64::new(2.0, 0.0)),
                None,
                Some(Complex64::new(3.0, 0.0))
            ]
        );
    }

    #[test]
    fn typed_mapping_distinguishes_translations_and_rejects_lattice_shifts() {
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let point = synthetic_point(vec![synthetic_magnetic_operation(
            0,
            0,
            identity,
            [0.5, 0.0, 0.0],
            false,
        )]);
        let matching = vec![
            (
                SeitzOperation {
                    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    translation: [0.0, 0.0, 0.0],
                },
                Complex64::new(1.0, 0.0),
            ),
            (
                SeitzOperation {
                    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    translation: [0.5, 0.0, 0.0],
                },
                Complex64::new(2.0, 0.0),
            ),
        ];
        assert_eq!(
            match_typed_entries_to_point(&matching, &point, 1, "synthetic", "S1").unwrap(),
            vec![Some(Complex64::new(2.0, 0.0))]
        );
        let lattice_shift = vec![(
            SeitzOperation {
                rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                translation: [1.5, 0.0, 0.0],
            },
            Complex64::new(2.0, 0.0),
        )];
        assert!(
            match_typed_entries_to_point(&lattice_shift, &point, 1, "synthetic", "S1").is_err(),
            "integer lattice shifts must not be treated as exact matches"
        );
    }

    #[test]
    fn typed_mapping_rejects_duplicate_operations_and_is_order_independent() {
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let quarter_turn = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
        let duplicate = vec![
            (
                SeitzOperation {
                    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    translation: [0.0, 0.0, 0.0],
                },
                Complex64::new(1.0, 0.0),
            ),
            (
                SeitzOperation {
                    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    translation: [0.0, 0.0, 0.0],
                },
                Complex64::new(1.0, 0.0),
            ),
        ];
        let point = synthetic_point(vec![synthetic_magnetic_operation(
            0,
            0,
            identity,
            [0.0, 0.0, 0.0],
            false,
        )]);
        assert!(match_typed_entries_to_point(&duplicate, &point, 1, "synthetic", "S1").is_err());

        let reordered = synthetic_point(vec![
            synthetic_magnetic_operation(0, 1, quarter_turn, [0.0, 0.0, 0.0], false),
            synthetic_magnetic_operation(1, 0, identity, [0.0, 0.0, 0.0], false),
        ]);
        let entries = vec![
            (
                SeitzOperation {
                    rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    translation: [0.0, 0.0, 0.0],
                },
                Complex64::new(1.0, 0.0),
            ),
            (
                SeitzOperation {
                    rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1],
                    translation: [0.0, 0.0, 0.0],
                },
                Complex64::new(2.0, 0.0),
            ),
        ];
        assert_eq!(
            match_typed_entries_to_point(&entries, &reordered, 1, "synthetic", "S1").unwrap(),
            vec![
                Some(Complex64::new(2.0, 0.0)),
                Some(Complex64::new(1.0, 0.0))
            ]
        );
        assert_eq!(
            corep_shape(cryspglib::irrep::corep::CorepType::A, &[1]),
            Ok((1, 1))
        );
    }

    #[test]
    fn corep_type_factors_and_source_identity_are_explicit() {
        use cryspglib::irrep::corep::CorepType;

        assert_eq!(corep_shape(CorepType::A, &[2]), Ok((2, 1)));
        assert_eq!(corep_shape(CorepType::B, &[2]), Ok((4, 2)));
        assert_eq!(corep_shape(CorepType::C, &[1, 3]), Ok((4, 1)));
        assert!(corep_shape(CorepType::C, &[4]).is_err());

        let operations = synthetic_point(vec![
            synthetic_magnetic_operation(
                0,
                0,
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0.0, 0.0, 0.0],
                false,
            ),
            synthetic_magnetic_operation(
                1,
                1,
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [0.0, 0.0, 0.0],
                true,
            ),
        ])
        .operations;
        let scalar_source = |value: f64| TypedSourceCharacters {
            dimension: 1,
            values: vec![Some(Complex64::new(value, 0.0)), None],
        };
        assert_eq!(
            combine_formal_characters(CorepType::A, &[scalar_source(1.0)], &operations, 1).unwrap(),
            vec![Some(Complex64::new(1.0, 0.0)), None]
        );
        assert_eq!(
            combine_formal_characters(CorepType::B, &[scalar_source(2.0)], &operations, 2).unwrap(),
            vec![Some(Complex64::new(4.0, 0.0)), None]
        );
        assert_eq!(
            combine_formal_characters(
                CorepType::C,
                &[scalar_source(1.0), scalar_source(3.0)],
                &operations,
                1,
            )
            .unwrap(),
            vec![Some(Complex64::new(4.0, 0.0)), None]
        );

        let record = cryspglib::irrep::query::irreps_of(1)
            .iter()
            .find(|record| !record.spinor)
            .expect("SG1 must provide a scalar record");
        let coords = (record.kx, record.ky, record.kz, record.kd);
        assert!(source_record_matches(
            record, 1, record.ml, false, 1, coords
        ));
        assert!(!source_record_matches(
            record, 1, record.ml, true, 1, coords
        ));
        assert!(!source_record_matches(
            record,
            1,
            record.ml,
            false,
            1,
            (record.kx, record.ky, record.kz, record.kd + 1)
        ));
        assert!(!source_record_matches(
            record, 2, record.ml, false, 1, coords
        ));
        assert_eq!(
            cryspglib::irrep::query::irreps_of(1)
                .iter()
                .filter(|candidate| {
                    source_record_matches(candidate, 1, record.ml, false, 1, coords)
                })
                .count(),
            1
        );
        let no_match = cryspglib::irrep::query::irreps_of(1)
            .iter()
            .filter(|candidate| source_record_matches(candidate, 1, "missing", false, 1, coords))
            .count();
        assert_eq!(no_match, 0);
        let duplicate_candidates = [record, record]
            .iter()
            .filter(|candidate| source_record_matches(candidate, 1, record.ml, false, 1, coords))
            .count();
        assert_eq!(duplicate_candidates, 2);
    }

    #[test]
    fn complex_type_c_recovery_uses_operation_columns_and_merges_partners() {
        use cryspglib::irrep::corep::CorepType;

        let partial = magnetic_irrep_summary_by_uni_partial(54)
            .expect("UNI 54 partial magnetic summary must remain available");
        let point = partial
            .summary
            .kpoints
            .iter()
            .find(|point| point.label == "Y")
            .expect("UNI 54 has a Y point");
        let failures = partial
            .unresolved_coreps
            .iter()
            .filter(|failure| {
                failure.k_label == point.label
                    && failure.spinor
                    && failure.classified_type == Some(CorepType::C)
            })
            .collect::<Vec<_>>();
        assert!(
            failures.len() >= 2 && failures.len().is_multiple_of(2),
            "UNI 54 Y must expose complete Type-C partner pairs"
        );

        let recovered = failures
            .iter()
            .map(|failure| {
                recover_complex_formal_corep::<true>(partial.summary.unitary_sg, point, failure)
                    .expect("complex recovery must preserve structural invariants")
                    .expect("UNI 54 Type-C row is available on the complex API")
            })
            .collect::<Vec<_>>();
        for recovered in &recovered {
            assert_eq!(recovered.corep_type, CorepType::C);
            assert_eq!(recovered.formal.dimension, 2);
            assert_eq!(recovered.formal.characters.len(), point.operations.len());
            assert!(
                recovered
                    .formal
                    .characters
                    .iter()
                    .flatten()
                    .any(|character| character.im.abs() > 1.0),
                "the regression must exercise a genuinely complex Type-C character"
            );
            for (character, operation) in recovered.formal.characters.iter().zip(&point.operations)
            {
                assert_eq!(character.is_none(), operation.time_reversal);
            }
        }

        let merged = merge_recovered_formal_coreps(recovered);
        assert_eq!(
            merged.len() * 2,
            failures.len(),
            "each pair of partner seeds must describe one Type-C corep"
        );
        let mut expected_labels = failures
            .iter()
            .map(|failure| failure.source_irrep.as_str())
            .collect::<Vec<_>>();
        expected_labels.sort();
        expected_labels.dedup();
        let mut merged_labels = merged
            .iter()
            .flat_map(|corep| corep.label.split(" + "))
            .collect::<Vec<_>>();
        merged_labels.sort();
        merged_labels.dedup();
        assert_eq!(merged_labels, expected_labels);
        assert!(
            merged
                .iter()
                .all(|corep| corep.label.split(" + ").count() == 2)
        );
    }

    #[test]
    #[ignore = "exhaustive typed-row acceptance census"]
    fn exhaustive_typed_formal_corep_acceptance_census() {
        let mut summaries = 0usize;
        let mut points = 0usize;
        let mut formal_coreps = 0usize;
        for uni in 1..=1651 {
            let Ok(summary) =
                cryspglib::irrep::magnetic_summary::magnetic_irrep_summary_by_uni(uni)
            else {
                continue;
            };
            summaries += 1;
            for point in &summary.kpoints {
                points += 1;
                for spinor in [false, true] {
                    let result = if spinor {
                        prepare_formal_coreps::<true>(summary.unitary_sg, point)
                    } else {
                        prepare_formal_coreps::<false>(summary.unitary_sg, point)
                    };
                    let coreps = result.unwrap_or_else(|error| {
                        panic!(
                            "UNI {uni} SG {} k-point {} SPIN={spinor} typed mapping failed: {error:?}",
                            summary.unitary_sg, point.label
                        )
                    });
                    formal_coreps += coreps.len();
                }
            }
        }
        println!(
            "typed formal corep acceptance census: summaries={summaries}, points={points}, coreps={formal_coreps}"
        );
    }

    #[test]
    #[ignore = "exhaustive 1651-UNI partial complex-corep recovery census"]
    fn exhaustive_partial_complex_corep_recovery_census() {
        let mut points = 0usize;
        let mut recovered_sources = 0usize;
        let mut recovered_type_c_sources = 0usize;
        let mut merged_coreps = 0usize;
        let mut still_unresolved = 0usize;

        for uni in 1..=1651 {
            let partial = magnetic_irrep_summary_by_uni_partial(uni)
                .unwrap_or_else(|error| panic!("UNI {uni} partial summary failed: {error:?}"));
            for point in &partial.summary.kpoints {
                points += 1;
                for spinor in [false, true] {
                    let failures = partial
                        .unresolved_coreps
                        .iter()
                        .filter(|failure| {
                            failure.k_label == point.label && failure.spinor == spinor
                        })
                        .collect::<Vec<_>>();
                    let mut recovered = Vec::new();
                    for failure in failures {
                        let result = if spinor {
                            recover_complex_formal_corep::<true>(
                                partial.summary.unitary_sg,
                                point,
                                failure,
                            )
                        } else {
                            recover_complex_formal_corep::<false>(
                                partial.summary.unitary_sg,
                                point,
                                failure,
                            )
                        }
                        .unwrap_or_else(|error| {
                            panic!(
                                "UNI {uni} SG {} k-point {} source {} SPIN={spinor} recovery invariant failed: {error:?}",
                                partial.summary.unitary_sg, point.label, failure.source_irrep
                            )
                        });
                        match result {
                            Some(corep) => {
                                recovered_sources += 1;
                                recovered_type_c_sources += usize::from(
                                    corep.corep_type == cryspglib::irrep::corep::CorepType::C,
                                );
                                recovered.push(corep);
                            }
                            None => still_unresolved += 1,
                        }
                    }
                    let merged = merge_recovered_formal_coreps(recovered);
                    assert!(merged.iter().all(|corep| {
                        corep.dimension > 0 && corep.characters.len() == point.operations.len()
                    }));
                    merged_coreps += merged.len();
                }
            }
        }
        assert!(recovered_sources > 0);
        assert!(recovered_type_c_sources > 0);
        println!(
            "partial complex recovery census: points={points} recovered_sources={recovered_sources} recovered_type_c_sources={recovered_type_c_sources} merged_coreps={merged_coreps} still_unresolved={still_unresolved}"
        );
    }

    fn assert_summary_error(uni: usize, sg: u8, source: &str, reason_fragment: &str) {
        let error = cryspglib::irrep::magnetic_summary::magnetic_irrep_summary_by_uni(uni)
            .expect_err("compound summary must propagate its structured upstream error");
        match error {
            cryspglib::irrep::magnetic_summary::MagneticIrrepError::CorepComputationFailed {
                uni: error_uni,
                sg: error_sg,
                k_label,
                source_irrep,
                reason,
            } => {
                assert_eq!(error_uni, uni);
                assert_eq!(error_sg, sg);
                assert_eq!(k_label, "GM");
                assert_eq!(source_irrep, source);
                assert!(
                    reason.contains(reason_fragment),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("unexpected UNI {uni} compound summary error: {other:?}"),
        }
    }

    #[test]
    fn sg199_compound_summary_propagates_upstream_error() {
        assert_summary_error(1515, 199, "GM2GM3", "compound");
    }

    #[test]
    fn sg220_partial_summary_retains_safe_rows_alongside_unresolved_sources() {
        let partial = magnetic_irrep_summary_by_uni_partial(1592)
            .expect("UNI 1592 partial summary must retain independently safe rows");
        assert!(
            !partial.unresolved_coreps.is_empty(),
            "the remaining unsupported sources must stay explicit"
        );
        assert!(
            partial
                .unresolved_coreps
                .iter()
                .all(|failure| failure.uni == 1592 && !failure.reason.is_empty())
        );
        assert!(
            partial
                .summary
                .kpoints
                .iter()
                .flat_map(|point| &point.coreps)
                .next()
                .is_some(),
            "one unsupported source must not discard independently safe rows"
        );
    }
}
