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
//! group is different: the numerical calculation still runs, but a subspace
//! with excessive leakage, non-integer corep multiplicities, or a poor
//! character reconstruction is labelled `???`.  This makes the same API useful
//! both before and after [`Model::symmetrize_hamiltonian`].
//!
//! High-symmetry k points are independent and are evaluated with Rayon.  The
//! indexed parallel collection preserves the canonical database order in the
//! returned report.  Users should normally set the selected BLAS backend to one
//! thread when using this outer parallelism.

use crate::crystal_symmetry::{CrystalSymmetryOperation, MagneticCrystalSymmetry, cry_lattice};
use crate::error::{Result, TbError};
use crate::hamiltonian_symmetry::{
    BasisActionContext, BasisSymmetryRepresentation, HamiltonianSymmetrizationParameters,
    LocalizedBasisAction, validate_action, validate_action_geometry,
};
use crate::{Gauge, Model, RMatrixData};
use cryspglib::irrep::magnetic_summary::{
    MagneticCorepSummary, MagneticKPointSummary, magnetic_irrep_summary_by_uni,
};
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
    /// Relative energy tolerance used to form degenerate band clusters.
    pub degeneracy_relative: f64,
    /// Maximum normalized leakage of a transformed band subspace.
    pub subspace_tolerance: f64,
    /// Maximum relative residual of the fitted character vector.
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

/// Structured and printable result of [`Model::calculate_irrep`].
#[derive(Debug, Clone, PartialEq)]
pub struct IrrepCalculationReport {
    pub uni_number: usize,
    pub bns_number: String,
    pub spinful: bool,
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
}

impl<const SPIN: bool, R: RMatrixData> Model<SPIN, 3, R> {
    /// Calculate magnetic little-group characters and corep labels at all
    /// canonical high-symmetry k points.
    ///
    /// `target_group` supplies the magnetic group whose labels are requested.
    /// The effective `field_preserving_operations` are used, so a uniform
    /// electric or magnetic field can reduce the target group.  The operation
    /// set is validated and reidentified before any eigenproblem is solved.
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
    pub fn calculate_irrep<P>(
        &self,
        target_group: &MagneticCrystalSymmetry,
        representation: &P,
        options: Option<&IrrepCalculationOptions>,
    ) -> Result<IrrepCalculationReport>
    where
        P: BasisSymmetryRepresentation<SPIN, R>,
    {
        let options = options.copied().unwrap_or_default();
        validate_options(&options)?;
        self.validate()?;

        let target_operations = &target_group.field_preserving_operations;
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
        let summary =
            magnetic_irrep_summary_by_uni(uni).map_err(|error| TbError::MagneticIrrepAnalysis {
                uni,
                message: format!("{error:?}"),
            })?;

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
            points.push(PreparedKPoint {
                summary: point,
                canonical_coordinate,
                model_coordinate,
                is_point,
                operations,
            });
        }

        let point_results = points
            .par_iter()
            .map(|point| calculate_kpoint(self, point, &prepared_operations, &options))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        Ok(IrrepCalculationReport {
            uni_number: uni,
            bns_number: identification.bns_number,
            spinful: SPIN,
            high_symmetry_kpoints: point_results,
        })
    }
}

fn validate_options(options: &IrrepCalculationOptions) -> Result<()> {
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

fn calculate_kpoint<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    point: &PreparedKPoint,
    prepared_operations: &[PreparedOperation],
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
        let fit =
            fit_corepresentations::<SPIN>(&point.summary.coreps, &characters, dimension, options)?;
        let mut diagnostics = fit.diagnostics;
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
    let mut groups = Vec::new();
    let mut start = 0;
    for boundary in 1..energies.len() {
        let scale = energies[boundary - 1].abs().max(energies[boundary].abs());
        if (energies[boundary] - energies[boundary - 1]).abs() > absolute + relative * scale {
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

fn fit_corepresentations<const SPIN: bool>(
    all_coreps: &[MagneticCorepSummary],
    numerical_characters: &[IrrepCharacter],
    band_dimension: usize,
    options: &IrrepCalculationOptions,
) -> Result<CorepFit> {
    let coreps = all_coreps
        .iter()
        .filter(|corep| {
            !corep.source_irreps.is_empty()
                && corep
                    .source_irreps
                    .iter()
                    .all(|source| source.spinor == SPIN)
        })
        .collect::<Vec<_>>();
    let unitary_columns = numerical_characters
        .iter()
        .filter(|character| !character.time_reversal)
        .collect::<Vec<_>>();
    let mut diagnostics = Vec::new();
    if coreps.is_empty() {
        diagnostics.push(format!(
            "the database has no {} corepresentations at this k point",
            if SPIN { "spinor" } else { "single-valued" }
        ));
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
                .ok_or_else(|| TbError::IrrepCalculation {
                    message: format!(
                        "formal corep {} has no character column {}",
                        corep.label, numerical_character.column
                    ),
                })?;
            character_matrix[[row, column]] = Complex64::new(*formal, 0.0);
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
        .map(|(corep, multiplicity)| corep.dim * multiplicity.rounded.unwrap_or(0))
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
        Atom, AtomType, AtomicOrbitalBasis, CrystalSymmetry, NoRMatrix, OrbProj, OrbitalId,
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

    #[test]
    fn incomplete_orbital_shell_is_rejected_before_band_labelling() {
        let model = orbital_cubic::<false>(&[OrbProj::px, OrbProj::py]);
        let target = model
            .magnetic_crystal_symmetry_from_atoms(
                &crate::crystal_symmetry::SymmetryParameters::default(),
            )
            .unwrap();
        let error = model
            .calculate_irrep(&target, &AtomicOrbitalBasis, None)
            .unwrap_err();
        assert!(matches!(error, TbError::IrrepBasisRepresentation { .. }));
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

        let broken = model
            .calculate_irrep(&target, &AtomicOrbitalBasis, None)
            .unwrap();
        let gamma = broken
            .high_symmetry_kpoints
            .iter()
            .find(|point| point.label == "GM")
            .unwrap();
        assert!(gamma.bands.iter().any(|band| band.label == "???"));
        assert!(broken.format_irvsp().contains("???"));

        let symmetrized = model
            .symmetrize_hamiltonian(
                &target,
                &AtomicOrbitalBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        let restored = symmetrized
            .calculate_irrep(&target, &AtomicOrbitalBasis, None)
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
        let target = model
            .magnetic_crystal_symmetry_from_atoms(
                &crate::crystal_symmetry::SymmetryParameters::default(),
            )
            .unwrap();
        let report = model
            .calculate_irrep(&target, &AtomicOrbitalBasis, None)
            .unwrap();
        let gamma = report
            .high_symmetry_kpoints
            .iter()
            .find(|point| point.label == "GM")
            .unwrap();
        assert_eq!(gamma.bands.len(), 1);
        assert_eq!(gamma.bands[0].band_end - gamma.bands[0].band_start + 1, 2);
        assert!(gamma.bands[0].is_identified(), "{report}");
        assert!(
            gamma.bands[0]
                .characters
                .iter()
                .filter(|character| character.time_reversal)
                .all(|character| character.value.is_none())
        );
    }
}
