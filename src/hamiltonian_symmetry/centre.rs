use super::*;
use crate::crystal_symmetry::{CrystalSymmetryOperation, MagneticCrystalSymmetry};
use crate::model::Model;
use cryspglib::spin_group::{CollinearAction, CollinearSpinGroup};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::collections::BTreeMap;

/// One operation of a collinear two-channel corepresentation.
///
/// The spatial operation carries the anti-unitarity flag used by the
/// Hamiltonian transform. The channel action is independent: a unitary
/// Swap exchanges the two spin blocks without conjugating orbital matrices.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PairOperation {
    spatial: CrystalSymmetryOperation,
    channel: CollinearAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairSide {
    Up,
    Down,
}

/// Localized-basis provider for Wannier centres that do not coincide with
/// their parent atoms.
///
/// For each operation the provider maps parent atoms by species and periodic
/// position, uses the parent-atom image to fix the cell shift, and applies the
/// global Cartesian angular rotation to the atom-local projection coefficients.
/// The centres themselves are not required to be lattice images of the target
/// orbitals, so dense local rotations inside one atom are allowed.
///
/// The default uses the global cubic angular rotation and preserves the
/// standard Wannier labels. An opt-in onsite-Hamiltonian refinement
/// (`infer_local_rotations = true`) diagonalizes the onsite blocks and forms a
/// unitary similarity candidate when the standard frame is inconsistent and
/// the onsite spectra match. An onsite Hermitian block alone determines a local
/// frame only up to degenerate eigenspaces and eigenvector phases, so this
/// refinement is deliberately not the default; offsite covariance remains the
/// authority and the standard rotation is the fallback.
#[derive(Debug, Clone, Copy)]
pub struct CenterResolvedBasis {
    /// Enable the experimental onsite-Hamiltonian local-frame fit.
    pub infer_local_rotations: bool,
}

impl Default for CenterResolvedBasis {
    fn default() -> Self {
        Self {
            infer_local_rotations: false,
        }
    }
}

impl CenterResolvedBasis {
    /// Provider using the global cubic angular rotation.
    pub fn new() -> Self {
        Self::default()
    }
    /// Provider using only the global cubic angular rotation.
    pub fn without_hamiltonian_inference() -> Self {
        Self {
            infer_local_rotations: false,
        }
    }
    /// Provider with opt-in onsite-Hamiltonian local-frame refinement.
    pub fn with_hamiltonian_inference() -> Self {
        Self {
            infer_local_rotations: true,
        }
    }
}

/// One orbital-centre displacement reported by centre symmetrization.
#[derive(Debug, Clone, PartialEq)]
pub struct CentreDisplacement {
    /// Global orbital index.
    pub orbital: usize,
    /// Parent atom index.
    pub atom: usize,
    /// Centre before projection, in fractional coordinates.
    pub old_fractional: [f64; 3],
    /// Centre after projection, in fractional coordinates.
    pub new_fractional: [f64; 3],
    /// Torus displacement in Angstrom, using the model lattice.
    pub displacement_angstrom: f64,
}

/// Result of projecting orbital centres onto a target magnetic group.
///
/// A displacement larger than the caller threshold is reported through
/// `warnings` and `moved_over_tolerance`; it is not an error. This keeps the
/// projection usable when the input Wannier gauge is intentionally rough while
/// still making a large change visible.
#[derive(Debug, Clone, PartialEq)]
pub struct CentreSymmetrizationReport {
    /// One entry per orbital, in global orbital order.
    pub displacements: Vec<CentreDisplacement>,
    /// Largest torus displacement in Angstrom.
    pub max_displacement_angstrom: f64,
    /// Orbital indices whose displacement exceeds `tolerance_angstrom`.
    pub moved_over_tolerance: Vec<usize>,
    /// Caller-supplied warning threshold in Angstrom.
    pub tolerance_angstrom: f64,
    /// Human-readable warnings; empty when every displacement is below the
    /// threshold.
    pub warnings: Vec<String>,
}

/// Joint centre and Hamiltonian symmetrization report.
///
/// The centre field describes the first stage. The two Hamiltonian numbers are
/// computed after the final projection: the maximum element-wise change from
/// the input Hamiltonian and the maximum post-projection covariance residual
/// over every target operation.
#[derive(Debug, Clone, PartialEq)]
pub struct SymmetrizationReport {
    pub centres: CentreSymmetrizationReport,
    pub hamiltonian_change_max_absolute: f64,
    pub final_covariance_max_absolute: f64,
}

/// Two spin-channel symmetrization report.
///
/// The up and down centre reports describe the joint centre projection. The
/// pair covariance residual compares the up Hamiltonian with the spin-flip
/// image of the down Hamiltonian for anti-unitary target operations (and each
/// model with its own image for unitary operations).
#[derive(Debug, Clone, PartialEq)]
pub struct PairSymmetrizationReport {
    pub up_centres: CentreSymmetrizationReport,
    pub down_centres: CentreSymmetrizationReport,
    pub hamiltonian_change_max_absolute: f64,
    pub pair_covariance_max_absolute: f64,
}

fn projection_matrix_unchecked<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    atom_index: usize,
) -> std::result::Result<Array2<Complex64>, BasisRepresentationError> {
    let atom = &model.atoms[atom_index];
    let mut coefficients = Array2::zeros((16, atom.orbitals().len()));
    for (column, orbital_id) in atom.orbitals().iter().enumerate() {
        let state = projection_in_real_orbital_basis(model.orb_projection[orbital_id.index()])?;
        for row in 0..16 {
            coefficients[[row, column]] = state[row];
        }
    }
    Ok(coefficients)
}

fn atom_shift<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    operation: &CrystalSymmetryOperation,
    source_atom: usize,
    position_tolerance: f64,
) -> std::result::Result<[isize; 3], BasisRepresentationError> {
    let target_atom = uniquely_mapped_atom(
        model,
        operation,
        source_atom,
        &BasisActionContext {
            model,
            operation,
            position_tolerance,
            representation_tolerance: position_tolerance,
        },
    )?;
    let source = model.atoms[source_atom].position_ref();
    let target = model.atoms[target_atom].position_ref();
    let transformed: [f64; 3] = std::array::from_fn(|row| {
        operation.translation[row]
            + (0..3)
                .map(|column| f64::from(operation.rotation[row][column]) * source[column])
                .sum::<f64>()
    });
    let displacement: [f64; 3] = std::array::from_fn(|axis| transformed[axis] - target[axis]);
    integer_shift_if_close(displacement, &model.lat, position_tolerance).ok_or_else(|| {
        BasisRepresentationError::Invalid(format!(
            "operation does not map parent atom {source_atom} onto a lattice image of its target atom"
        ))
    })
}

fn refine_local_action(
    standard: &Array2<Complex64>,
    source_onsite: &Array2<Complex64>,
    target_onsite: &Array2<Complex64>,
    tolerance: f64,
) -> Array2<Complex64> {
    let n = standard.nrows();
    if n == 0 || source_onsite.dim() != (n, n) || target_onsite.dim() != (n, n) {
        return standard.clone();
    }
    let Ok((source_energies, source_vectors)) =
        crate::ndarray_lapack::eigh_full(source_onsite, UPLO::Lower)
    else {
        return standard.clone();
    };
    let Ok((target_energies, target_vectors)) =
        crate::ndarray_lapack::eigh_full(target_onsite, UPLO::Lower)
    else {
        return standard.clone();
    };
    if source_energies.len() != target_energies.len() {
        return standard.clone();
    }
    for energies in [&source_energies, &target_energies] {
        let mut previous = f64::NEG_INFINITY;
        for &energy in energies.iter() {
            if energy - previous <= tolerance.max(1e-10) {
                return standard.clone();
            }
            previous = energy;
        }
    }
    for (left, right) in source_energies.iter().zip(target_energies.iter()) {
        if (left - right).abs() > tolerance.max(1e-8) {
            return standard.clone();
        }
    }
    let source_t = source_vectors.t().to_owned();
    let target_conj = target_vectors.mapv(|value| value.conj());
    let mut candidate = source_t.dot(&target_conj);
    for column in 0..n {
        let mut overlap = Complex64::new(0.0, 0.0);
        for row in 0..n {
            overlap += standard[[row, column]].conj() * candidate[[row, column]];
        }
        if overlap.norm() > 1e-15 {
            let phase = overlap / overlap.norm();
            for row in 0..n {
                candidate[[row, column]] *= phase;
            }
        }
    }
    let residual = |matrix: &Array2<Complex64>| {
        let left = matrix
            .t()
            .mapv(|value| value.conj())
            .dot(source_onsite)
            .dot(matrix);
        left.iter()
            .zip(target_onsite.iter())
            .map(|(left, right)| (left - right).norm())
            .fold(0.0, f64::max)
    };
    let candidate_residual = residual(&candidate);
    let standard_residual = residual(standard);
    let gram = candidate.t().mapv(|value| value.conj()).dot(&candidate);
    let mut unitarity = 0.0f64;
    for row in 0..n {
        for column in 0..n {
            let expected = if row == column { 1.0 } else { 0.0 };
            unitarity = unitarity.max((gram[[row, column]] - expected).norm());
        }
    }
    // An onsite similarity is only informative when the standard frame is
    // actually inconsistent and the fitted candidate materially improves it.
    // Near-degenerate or already-covariant onsite blocks otherwise admit
    // arbitrary unitary candidates that are not determined by H.
    if standard_residual > tolerance.max(1e-6)
        && candidate_residual < 0.5 * standard_residual
        && unitarity < tolerance.max(1e-6)
    {
        candidate
    } else {
        standard.clone()
    }
}

impl<const SPIN: bool, R: RMatrixData> BasisSymmetryRepresentation<SPIN, R>
    for CenterResolvedBasis
{
    fn resolve(
        &self,
        context: BasisActionContext<'_, SPIN, R>,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
        let model = context.model;
        validate_automatic_basis_context(&context, "CenterResolvedBasis")?;
        let angular_rotation = real_orbital_rotation(
            cartesian_rotation(model, context.operation)?,
            context.representation_tolerance.max(1e-6),
        )?;
        let spin_action =
            scalar_spin_action(model, context.operation, context.representation_tolerance)?;
        let site_projections = model
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_index, _)| projection_matrix_unchecked(model, atom_index))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let origin = find_origin_index(model);
        let mut sectors: BTreeMap<[isize; 3], Array2<Complex64>> = BTreeMap::new();
        for (source_atom, atom) in model.atoms.iter().enumerate() {
            let target_atom =
                uniquely_mapped_atom(model, context.operation, source_atom, &context)?;
            let shift = atom_shift(
                model,
                context.operation,
                source_atom,
                context.position_tolerance,
            )?;
            let source_coefficients = &site_projections[source_atom];
            let target_coefficients = &site_projections[target_atom];
            if source_coefficients.ncols() != target_coefficients.ncols() {
                return Err(BasisRepresentationError::Unsupported(format!(
                    "operation maps atom {source_atom} with {} orbitals to atom {target_atom} with {} orbitals",
                    source_coefficients.ncols(),
                    target_coefficients.ncols()
                )));
            }
            if source_coefficients.ncols() == 0 {
                continue;
            }
            let rotated_source = angular_rotation.dot(source_coefficients);
            let target_dagger = target_coefficients.t().mapv(|value| value.conj());
            let mut local_action = target_dagger.dot(&rotated_source);
            if self.infer_local_rotations
                && !SPIN
                && let Some(origin) = origin
                && let Some(source_onsite) = onsite_block(model, origin, atom.orbitals())
                && let Some(target_onsite) =
                    onsite_block(model, origin, model.atoms[target_atom].orbitals())
            {
                local_action = refine_local_action(
                    &local_action,
                    &source_onsite,
                    &target_onsite,
                    context.representation_tolerance,
                );
            }
            let matrix = sectors
                .entry(shift)
                .or_insert_with(|| Array2::zeros((model.nsta(), model.nsta())));
            for (source_local, source_id) in atom.orbitals().iter().enumerate() {
                let source_orbital = source_id.index();
                for (target_local, target_id) in
                    model.atoms[target_atom].orbitals().iter().enumerate()
                {
                    let coefficient = local_action[[target_local, source_local]];
                    if coefficient.norm() <= context.representation_tolerance {
                        continue;
                    }
                    let target_orbital = target_id.index();
                    if SPIN {
                        for target_spin in 0..2 {
                            for source_spin in 0..2 {
                                matrix[[
                                    target_spin * model.norb() + target_orbital,
                                    source_spin * model.norb() + source_orbital,
                                ]] += coefficient * spin_action[target_spin][source_spin];
                            }
                        }
                    } else {
                        matrix[[target_orbital, source_orbital]] += coefficient;
                    }
                }
            }
        }
        let sectors = sectors
            .into_iter()
            .map(|(shift, matrix)| CellShiftAction { shift, matrix })
            .collect();
        Ok(LocalizedBasisAction { sectors })
    }

    fn validates_orbital_centres(&self) -> bool {
        false
    }
}

fn find_origin_index<const SPIN: bool, R: RMatrixData>(model: &Model<SPIN, 3, R>) -> Option<usize> {
    (0..model.hamR.nrows()).find(|&index| {
        model.hamR[[index, 0]] == 0 && model.hamR[[index, 1]] == 0 && model.hamR[[index, 2]] == 0
    })
}

fn onsite_block<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    origin: usize,
    orbitals: &[crate::OrbitalId],
) -> Option<Array2<Complex64>> {
    let n = orbitals.len();
    if n == 0 {
        return None;
    }
    let mut block = Array2::zeros((n, n));
    for (row, target) in orbitals.iter().enumerate() {
        for (column, source) in orbitals.iter().enumerate() {
            block[[row, column]] = model.ham[[origin, target.index(), source.index()]];
        }
    }
    Some(block)
}

fn transformed_center(operation: &CrystalSymmetryOperation, center: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        operation.translation[row]
            + (0..3)
                .map(|column| f64::from(operation.rotation[row][column]) * center[column])
                .sum::<f64>()
    })
}

fn minimum_image_delta(left: &[f64; 3], right: &[f64; 3], lattice: &Array2<f64>) -> [f64; 3] {
    let base: [f64; 3] = std::array::from_fn(|axis| left[axis] - right[axis]);
    let mut best: [f64; 3] = std::array::from_fn(|axis| base[axis] - base[axis].round());
    let mut best_norm = cartesian_norm(&best, lattice);
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                let candidate = [
                    base[0] - (base[0].round() + x as f64),
                    base[1] - (base[1].round() + y as f64),
                    base[2] - (base[2].round() + z as f64),
                ];
                let norm = cartesian_norm(&candidate, lattice);
                if norm < best_norm {
                    best_norm = norm;
                    best = candidate;
                }
            }
        }
    }
    best
}

fn cartesian_norm(delta: &[f64; 3], lattice: &Array2<f64>) -> f64 {
    let cart = Array1::from_shape_fn(3, |axis| {
        (0..3)
            .map(|component| delta[component] * lattice[[component, axis]])
            .sum::<f64>()
    });
    cart.dot(&cart).sqrt()
}

fn map_atoms<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    operation: &CrystalSymmetryOperation,
    tolerance: f64,
) -> Result<Vec<usize>> {
    model
        .atoms
        .iter()
        .enumerate()
        .map(|(source, _)| {
            uniquely_mapped_atom(
                model,
                operation,
                source,
                &BasisActionContext {
                    model,
                    operation,
                    position_tolerance: tolerance,
                    representation_tolerance: tolerance,
                },
            )
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("operation cannot map atom {source}: {error}"),
            })
        })
        .collect()
}

fn centre_permutation<R: RMatrixData>(
    model: &Model<false, 3, R>,
    centres: &Array2<f64>,
    operation: &CrystalSymmetryOperation,
    atom_map: &[usize],
    action: Option<&LocalizedBasisAction>,
) -> Result<Vec<(usize, usize, [f64; 3])>> {
    let mut inverse = vec![usize::MAX; model.natom()];
    for (source, &target) in atom_map.iter().enumerate() {
        inverse[target] = source;
    }
    let mut entries = Vec::with_capacity(model.norb());
    for (target_atom, atom) in model.atoms.iter().enumerate() {
        let source_atom = inverse[target_atom];
        if source_atom == usize::MAX {
            return Err(crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("operation has no source atom for target atom {target_atom}"),
            });
        }
        for target_id in atom.orbitals() {
            let target_orbital = target_id.index();
            let target_center = [
                centres[[target_orbital, 0]],
                centres[[target_orbital, 1]],
                centres[[target_orbital, 2]],
            ];
            let target_projection = model.orb_projection[target_orbital];
            let mut action_source = usize::MAX;
            if let Some(action) = action {
                let mut largest = 0.0f64;
                for source_id in model.atoms[source_atom].orbitals() {
                    let source_orbital = source_id.index();
                    let mut coefficient = 0.0f64;
                    for sector in &action.sectors {
                        coefficient =
                            coefficient.max(sector.matrix[[target_orbital, source_orbital]].norm());
                    }
                    if coefficient > largest {
                        largest = coefficient;
                        action_source = source_orbital;
                    }
                }
            }
            let mut best = (action_source, f64::INFINITY);
            for source_id in model.atoms[source_atom].orbitals() {
                let source_orbital = source_id.index();
                if action_source != usize::MAX {
                    if source_orbital != action_source {
                        continue;
                    }
                } else if model.orb_projection[source_orbital] != target_projection {
                    continue;
                }
                let source_center = [
                    centres[[source_orbital, 0]],
                    centres[[source_orbital, 1]],
                    centres[[source_orbital, 2]],
                ];
                let shifted = transformed_center(operation, &source_center);
                let delta = minimum_image_delta(&shifted, &target_center, &model.lat);
                let distance = cartesian_norm(&delta, &model.lat);
                if distance < best.1 {
                    best = (source_orbital, distance);
                }
            }
            if best.0 == usize::MAX {
                for source_id in model.atoms[source_atom].orbitals() {
                    let source_orbital = source_id.index();
                    let source_center = [
                        centres[[source_orbital, 0]],
                        centres[[source_orbital, 1]],
                        centres[[source_orbital, 2]],
                    ];
                    let shifted = transformed_center(operation, &source_center);
                    let delta = minimum_image_delta(&shifted, &target_center, &model.lat);
                    let distance = cartesian_norm(&delta, &model.lat);
                    if distance < best.1 {
                        best = (source_orbital, distance);
                    }
                }
            }
            if best.0 == usize::MAX {
                return Err(crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("no source orbital for target orbital {target_orbital}"),
                });
            }
            let source_center = [
                centres[[best.0, 0]],
                centres[[best.0, 1]],
                centres[[best.0, 2]],
            ];
            let image = transformed_center(operation, &source_center);
            let target_parent = atom.position_ref();
            let integer_shift: [f64; 3] =
                std::array::from_fn(|axis| (image[axis] - target_parent[axis]).round());
            // Keep the exact operation image modulo an integer lattice shift.
            // The caller adds the target parent position back; this preserves
            // any small atom-rounding discrepancy instead of silently
            // subtracting it, so centre covariance is tied to the operation,
            // not to the rounded parent positions.
            let rotated: [f64; 3] =
                std::array::from_fn(|axis| image[axis] - integer_shift[axis] - target_parent[axis]);
            entries.push((target_orbital, best.0, rotated));
        }
    }
    Ok(entries)
}

fn centre_report(
    old_centres: &Array2<f64>,
    new_centres: &Array2<f64>,
    atoms: &[crate::Atom],
    lattice: &Array2<f64>,
    tolerance: f64,
) -> CentreSymmetrizationReport {
    let mut displacements = Vec::with_capacity(old_centres.nrows());
    let mut moved = Vec::new();
    let mut max = 0.0f64;
    for orbital in 0..old_centres.nrows() {
        let old = [
            old_centres[[orbital, 0]],
            old_centres[[orbital, 1]],
            old_centres[[orbital, 2]],
        ];
        let new = [
            new_centres[[orbital, 0]],
            new_centres[[orbital, 1]],
            new_centres[[orbital, 2]],
        ];
        let delta = minimum_image_delta(&new, &old, lattice);
        let displacement = cartesian_norm(&delta, lattice);
        if displacement > tolerance {
            moved.push(orbital);
        }
        max = max.max(displacement);
        let atom = atoms
            .iter()
            .position(|atom| atom.orbitals().iter().any(|id| id.index() == orbital))
            .unwrap_or(usize::MAX);
        displacements.push(CentreDisplacement {
            orbital,
            atom,
            old_fractional: old,
            new_fractional: new,
            displacement_angstrom: displacement,
        });
    }
    let mut warnings = Vec::new();
    if max > tolerance {
        warnings.push(format!(
            "maximum centre displacement {max:.6e} A exceeds threshold {tolerance:.6e} A"
        ));
    }
    CentreSymmetrizationReport {
        displacements,
        max_displacement_angstrom: max,
        moved_over_tolerance: moved,
        tolerance_angstrom: tolerance,
        warnings,
    }
}

impl<R: RMatrixData> Model<false, 3, R> {
    /// Project orbital centres onto the spatial parts of `target_group`.
    ///
    /// For each target orbital i with parent atom a and inverse-mapped source
    /// orbital j on parent atom b, one step computes
    ///
    /// ```text
    /// c_i -> (1/N) sum_g (W_g c_j + t_g - n_g),
    /// ```
    ///
    /// where n_g is the integer lattice shift that brings the image nearest the
    /// target parent atom. This preserves the exact operation image, including
    /// any small parent-position rounding in a Wannier90 structure file, instead
    /// of absorbing that residual into the target parent.
    ///
    /// Time reversal does not move a centre, so both unitary and anti-unitary
    /// operations contribute their spatial part. The source orbital is selected
    /// from the resolved localized action as the column with the largest matrix
    /// coefficient in the target orbital's row, restricted to the inverse-mapped
    /// parent atom. If the action has no usable coefficient, the code falls back
    /// to an OrbProj label match and then to the nearest transformed centre. For a genuine spin-flip pair use [`Self::symmetrize_pair`],
    /// which couples the two channels instead of applying an anti-unitary
    /// spatial part to one model alone.
    ///
    /// A target operation that cannot map the atom structure returns
    /// [`crate::TbError::TargetMagneticGroupIncompatible`]. A maximum
    /// displacement above `tolerance` produces warning entries in the returned
    /// report but does not fail the projection. The argument is measured in
    /// Angstrom. The model lattice and atoms are not changed; only `orb` is
    /// updated.
    pub fn symmetrize_centres(
        &mut self,
        target_group: &MagneticCrystalSymmetry,
        basis: &impl BasisSymmetryRepresentation<false, R>,
        tolerance: f64,
    ) -> Result<CentreSymmetrizationReport> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(crate::TbError::InvalidResponseParameter {
                parameter: "tolerance",
                message: "centre tolerance must be finite and non-negative".into(),
            });
        }
        let original = self.orb.clone();
        for _ in 0..64 {
            let report = self.symmetrize_centres_once(target_group, basis, tolerance)?;
            if report.max_displacement_angstrom < 1e-10 {
                break;
            }
        }
        Ok(centre_report(
            &original,
            &self.orb,
            &self.atoms,
            &self.lat,
            tolerance,
        ))
    }

    fn symmetrize_centres_once(
        &mut self,
        target_group: &MagneticCrystalSymmetry,
        basis: &impl BasisSymmetryRepresentation<false, R>,
        tolerance: f64,
    ) -> Result<CentreSymmetrizationReport> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(crate::TbError::InvalidResponseParameter {
                parameter: "tolerance",
                message: "centre tolerance must be finite and non-negative".into(),
            });
        }
        if target_group.operations.is_empty() {
            return Err(crate::TbError::TargetMagneticGroupIncompatible {
                reason: "target magnetic group has no operations".into(),
            });
        }
        let atom_tolerance = 1e-5;
        let operations: Vec<_> = target_group.operations.iter().collect();
        if operations.is_empty() {
            return Err(crate::TbError::TargetMagneticGroupIncompatible {
                reason: "target magnetic group has no operations".into(),
            });
        }
        let old = self.orb.clone();
        let mut accumulated: Vec<(f64, [f64; 3])> = Vec::new();
        accumulated.resize(self.norb(), (0.0, [0.0; 3]));
        for operation in &operations {
            let action = basis
                .resolve(BasisActionContext {
                    model: self,
                    operation,
                    position_tolerance: 1e-5,
                    representation_tolerance: 1e-6,
                })
                .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("centre basis action failed: {error}"),
                })?;
            let action = validate_action(action, self.nsta(), 1e-6).map_err(|error| {
                crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("centre basis action is invalid: {error}"),
                }
            })?;
            let atom_map = map_atoms(self, operation, atom_tolerance)?;
            let entries =
                centre_permutation(&*self, &self.orb, operation, &atom_map, Some(&action))?;
            for (target_orbital, source_orbital, delta) in entries {
                let target_atom = self
                    .atoms
                    .iter()
                    .position(|atom| {
                        atom.orbitals()
                            .iter()
                            .any(|id| id.index() == target_orbital)
                    })
                    .ok_or_else(|| crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("target orbital {target_orbital} has no parent atom"),
                    })?;
                let target_position = self.atoms[target_atom].position_ref();
                let source_atom = self
                    .atoms
                    .iter()
                    .position(|atom| {
                        atom.orbitals()
                            .iter()
                            .any(|id| id.index() == source_orbital)
                    })
                    .ok_or_else(|| crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("source orbital {source_orbital} has no parent atom"),
                    })?;
                let _ = source_atom;
                let _ = target_position;
                let (count, sum) = &mut accumulated[target_orbital];
                *count += 1.0;
                for axis in 0..3 {
                    sum[axis] += delta[axis];
                }
            }
        }
        let mut new = old.clone();
        for orbital in 0..self.norb() {
            let (count, sum) = accumulated[orbital];
            if count == 0.0 {
                return Err(crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!(
                        "orbital {orbital} was not reached by any spin-preserving operation"
                    ),
                });
            }
            let atom = self
                .atoms
                .iter()
                .position(|atom| atom.orbitals().iter().any(|id| id.index() == orbital))
                .ok_or_else(|| crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("orbital {orbital} has no parent atom"),
                })?;
            let parent = self.atoms[atom].position_ref();
            for axis in 0..3 {
                new[[orbital, axis]] = parent[axis] + sum[axis] / count;
            }
        }
        let report = centre_report(&old, &new, &self.atoms, &self.lat, tolerance);
        self.orb = new;
        Ok(report)
    }
}

fn model_ham_blocks<R: RMatrixData>(
    model: &Model<false, 3, R>,
) -> BTreeMap<[isize; 3], Array2<Complex64>> {
    let mut blocks = BTreeMap::new();
    for index in 0..model.hamR.nrows() {
        blocks.insert(
            [
                model.hamR[[index, 0]],
                model.hamR[[index, 1]],
                model.hamR[[index, 2]],
            ],
            model.ham.index_axis(Axis(0), index).to_owned(),
        );
    }
    blocks
}

fn max_map_change(
    old: &BTreeMap<[isize; 3], Array2<Complex64>>,
    new: &BTreeMap<[isize; 3], Array2<Complex64>>,
) -> f64 {
    let mut keys: BTreeSet<[isize; 3]> = old.keys().copied().collect();
    keys.extend(new.keys().copied());
    let mut max = 0.0f64;
    for key in keys {
        let left = old.get(&key);
        let right = new.get(&key);
        let rows = left
            .map_or(0, Array2::nrows)
            .max(right.map_or(0, Array2::nrows));
        let columns = left
            .map_or(0, Array2::ncols)
            .max(right.map_or(0, Array2::ncols));
        let zero = Complex64::new(0.0, 0.0);
        for row in 0..rows {
            for column in 0..columns {
                let a = left
                    .and_then(|matrix| matrix.get((row, column)))
                    .copied()
                    .unwrap_or(zero);
                let b = right
                    .and_then(|matrix| matrix.get((row, column)))
                    .copied()
                    .unwrap_or(zero);
                max = max.max((a - b).norm());
            }
        }
    }
    max
}

fn target_covariance_max<R: RMatrixData>(
    model: &Model<false, 3, R>,
    target: &MagneticCrystalSymmetry,
    basis: &impl BasisSymmetryRepresentation<false, R>,
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<f64> {
    let support = validate_hamiltonian(model, parameters.tolerances.hermiticity)?;
    let position_tolerance = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec);
    let mut max = 0.0f64;
    for operation in &target.operations {
        let action = basis
            .resolve(BasisActionContext {
                model,
                operation,
                position_tolerance,
                representation_tolerance: parameters.tolerances.representation,
            })
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("post-projection action failed: {error}"),
            })?;
        let action = validate_action(action, model.nsta(), parameters.tolerances.representation)
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("post-projection action is invalid: {error}"),
            })?;
        let residual = hamiltonian_residual(&support, operation, &action, parameters.tolerances)
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("post-projection residual failed: {error}"),
            })?;
        max = max.max(residual.max_absolute);
    }
    Ok(max)
}

fn validate_pair_group(
    group: &MagneticCrystalSymmetry,
    operation_tolerance: f64,
) -> Result<MagneticCrystalSymmetry> {
    let operations = to_cry_operations(&group.operations);
    let validated = cryspglib::ValidatedMagneticOperationSet::try_from_symmetry_ops(
        &operations,
        operation_tolerance,
    )
    .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
        reason: format!("the supplied pair operations are not a magnetic group: {error}"),
    })?;
    let normalized = validated
        .operations()
        .iter()
        .map(from_cry_operation)
        .collect::<Vec<_>>();
    let mut result = group.clone();
    result.operations = normalized.clone();
    result.field_preserving_operations = normalized;
    Ok(result)
}

fn pair_centre_average<R: RMatrixData>(
    target: &mut Model<false, 3, R>,
    side: PairSide,
    up_centres: &Array2<f64>,
    down_centres: &Array2<f64>,
    operations: &[PairOperation],
    basis: &impl BasisSymmetryRepresentation<false, R>,
    tolerance: f64,
) -> Result<CentreSymmetrizationReport> {
    if operations.is_empty() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "target group has no operations".into(),
        });
    }
    let old = target.orb.clone();
    let mut sums = vec![[0.0f64; 3]; target.norb()];
    let mut counts = vec![0usize; target.norb()];
    for operation in operations {
        let centres =
            match (side, operation.channel) {
                (PairSide::Up, CollinearAction::Preserve)
                | (PairSide::Down, CollinearAction::Swap) => up_centres,
                (PairSide::Up, CollinearAction::Swap)
                | (PairSide::Down, CollinearAction::Preserve) => down_centres,
            };
        let action = basis
            .resolve(BasisActionContext {
                model: &*target,
                operation: &operation.spatial,
                position_tolerance: 1e-5,
                representation_tolerance: 1e-6,
            })
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair centre action failed: {error}"),
            })?;
        let action = validate_action(action, target.nsta(), 1e-6).map_err(|error| {
            crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair centre action is invalid: {error}"),
            }
        })?;
        let atom_map = map_atoms(&*target, &operation.spatial, 1e-5)?;
        let entries = centre_permutation(
            &*target,
            centres,
            &operation.spatial,
            &atom_map,
            Some(&action),
        )?;
        for (target_orbital, _source_orbital, delta) in entries {
            sums[target_orbital][0] += delta[0];
            sums[target_orbital][1] += delta[1];
            sums[target_orbital][2] += delta[2];
            counts[target_orbital] += 1;
        }
    }
    let mut new = old.clone();
    for orbital in 0..target.norb() {
        if counts[orbital] == 0 {
            return Err(crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("orbital {orbital} was not mapped by any operation"),
            });
        }
        let atom = target
            .atoms
            .iter()
            .position(|atom| atom.orbitals().iter().any(|id| id.index() == orbital))
            .ok_or_else(|| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("orbital {orbital} has no parent atom"),
            })?;
        let parent = target.atoms[atom].position_ref();
        for axis in 0..3 {
            new[[orbital, axis]] = parent[axis] + sums[orbital][axis] / counts[orbital] as f64;
        }
    }
    let report = centre_report(&old, &new, &target.atoms, &target.lat, tolerance);
    target.orb = new;
    Ok(report)
}

fn resolve_pair_actions<R: RMatrixData>(
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
    operations: &[PairOperation],
    basis: &impl BasisSymmetryRepresentation<false, R>,
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<Vec<(PairOperation, LocalizedBasisAction, LocalizedBasisAction)>> {
    let position_tolerance = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec);
    let mut result = Vec::with_capacity(operations.len());
    for operation in operations {
        let action_a = basis
            .resolve(BasisActionContext {
                model: a,
                operation: &operation.spatial,
                position_tolerance,
                representation_tolerance: parameters.tolerances.representation,
            })
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair action failed on first model: {error}"),
            })?;
        let action_b = basis
            .resolve(BasisActionContext {
                model: b,
                operation: &operation.spatial,
                position_tolerance,
                representation_tolerance: parameters.tolerances.representation,
            })
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair action failed on second model: {error}"),
            })?;
        let action_tolerance = parameters.tolerances.representation.max(1e-6);
        let action_a = validate_action(action_a, a.nsta(), action_tolerance).map_err(|error| {
            crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair action invalid on first model: {error}"),
            }
        })?;
        let action_b = validate_action(action_b, b.nsta(), action_tolerance).map_err(|error| {
            crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair action invalid on second model: {error}"),
            }
        })?;
        result.push((*operation, action_a, action_b));
    }
    Ok(result)
}

fn hermitize_support_map(matrices: &mut BTreeMap<[isize; 3], Array2<Complex64>>) -> Result<()> {
    let keys = matrices.keys().copied().collect::<Vec<_>>();
    let mut processed = BTreeSet::new();
    for lattice_vector in keys {
        if !processed.insert(lattice_vector) {
            continue;
        }
        let negative = checked_negate(lattice_vector).ok_or_else(|| {
            crate::TbError::TargetMagneticGroupIncompatible {
                reason: "pair support vector cannot be negated".into(),
            }
        })?;
        processed.insert(negative);
        let matrix = matrices.get(&lattice_vector).cloned().unwrap_or_else(|| {
            Array2::zeros((
                matrices.values().next().map_or(0, Array2::nrows),
                matrices.values().next().map_or(0, Array2::ncols),
            ))
        });
        let partner = matrices.get(&negative).cloned().unwrap_or_else(|| {
            Array2::zeros((
                matrices.values().next().map_or(0, Array2::nrows),
                matrices.values().next().map_or(0, Array2::ncols),
            ))
        });
        let partner_dagger = partner.t().mapv(|value| value.conj());
        let hermitian = (matrix + partner_dagger).mapv(|value| value * 0.5);
        let hermitian_dagger = hermitian.t().mapv(|value| value.conj());
        matrices.insert(lattice_vector, hermitian);
        matrices.insert(negative, hermitian_dagger);
    }
    Ok(())
}

fn pair_project_hamiltonian<R: RMatrixData>(
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
    operations: &[PairOperation],
    basis: &impl BasisSymmetryRepresentation<false, R>,
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<(Model<false, 3, R>, Model<false, 3, R>, f64, f64)> {
    let support_a = validate_hamiltonian(a, parameters.tolerances.hermiticity)?;
    let support_b = validate_hamiltonian(b, parameters.tolerances.hermiticity)?;
    let actions = resolve_pair_actions(a, b, operations, basis, parameters)?;
    if actions.is_empty() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "target group has no operations".into(),
        });
    }
    let actions_a = actions
        .iter()
        .map(|(operation, action, _)| (operation.spatial, action.clone()))
        .collect::<Vec<_>>();
    let actions_b = actions
        .iter()
        .map(|(operation, _, action)| (operation.spatial, action.clone()))
        .collect::<Vec<_>>();
    validate_projective_corepresentation(
        a,
        &actions_a,
        parameters.tolerances.operation,
        parameters.tolerances.representation.max(1e-6),
    )
    .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
        reason: format!("first pair channel actions are not a magnetic corepresentation: {error}"),
    })?;
    validate_projective_corepresentation(
        b,
        &actions_b,
        parameters.tolerances.operation,
        parameters.tolerances.representation.max(1e-6),
    )
    .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
        reason: format!("second pair channel actions are not a magnetic corepresentation: {error}"),
    })?;
    let mut domain: BTreeSet<[isize; 3]> = support_a.matrices.keys().copied().collect();
    domain.extend(support_b.matrices.keys().copied());
    for (operation, action_a, action_b) in &actions {
        let (source_a, source_b) = match operation.channel {
            CollinearAction::Preserve => (&support_a, &support_b),
            CollinearAction::Swap => (&support_b, &support_a),
        };
        for (support, action) in [(source_a, action_a), (source_b, action_b)] {
            domain.extend(
                transformed_support_domain(support, &operation.spatial, action).map_err(
                    |error| crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("pair support failed: {error}"),
                    },
                )?,
            );
        }
    }
    let existing = domain.iter().copied().collect::<Vec<_>>();
    for lattice_vector in existing {
        let negative = checked_negate(lattice_vector).ok_or_else(|| {
            crate::TbError::TargetMagneticGroupIncompatible {
                reason: "pair support vector cannot be negated".into(),
            }
        })?;
        domain.insert(negative);
    }
    let normalizer = actions.len() as f64;
    let mut projected_a = BTreeMap::new();
    let mut projected_b = BTreeMap::new();
    for lattice_vector in domain {
        let mut matrix_a = Array2::zeros((a.nsta(), a.nsta()));
        let mut matrix_b = Array2::zeros((b.nsta(), b.nsta()));
        for (operation, action_a, action_b) in &actions {
            let (source_a, source_b) = match operation.channel {
                CollinearAction::Preserve => (&support_a, &support_b),
                CollinearAction::Swap => (&support_b, &support_a),
            };
            matrix_a += &transformed_hamiltonian_at(
                source_a,
                &operation.spatial,
                action_a,
                lattice_vector,
                false,
            )
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair projection failed: {error}"),
            })?;
            matrix_b += &transformed_hamiltonian_at(
                source_b,
                &operation.spatial,
                action_b,
                lattice_vector,
                false,
            )
            .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                reason: format!("pair projection failed: {error}"),
            })?;
        }
        projected_a.insert(lattice_vector, matrix_a.mapv(|value| value / normalizer));
        projected_b.insert(lattice_vector, matrix_b.mapv(|value| value / normalizer));
    }
    hermitize_support_map(&mut projected_a)?;
    hermitize_support_map(&mut projected_b)?;
    let symmetrized_a = model_with_hamiltonian_support(a, projected_a.clone())?;
    let symmetrized_b = model_with_hamiltonian_support(b, projected_b.clone())?;
    let change = max_map_change(&model_ham_blocks(a), &model_ham_blocks(&symmetrized_a)).max(
        max_map_change(&model_ham_blocks(b), &model_ham_blocks(&symmetrized_b)),
    );
    let support_a = validate_hamiltonian(&symmetrized_a, parameters.tolerances.hermiticity)?;
    let support_b = validate_hamiltonian(&symmetrized_b, parameters.tolerances.hermiticity)?;
    let mut pair_residual = 0.0f64;
    for (operation, action_a, action_b) in &actions {
        let (source_a, source_b) = match operation.channel {
            CollinearAction::Preserve => (&support_a, &support_b),
            CollinearAction::Swap => (&support_b, &support_a),
        };
        for (target_support, source_support, action) in [
            (&support_a, source_a, action_a),
            (&support_b, source_b, action_b),
        ] {
            let domain = transformed_support_domain(source_support, &operation.spatial, action)
                .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("pair residual failed: {error}"),
                })?;
            for lattice_vector in domain {
                let predicted = transformed_hamiltonian_at(
                    source_support,
                    &operation.spatial,
                    action,
                    lattice_vector,
                    false,
                )
                .map_err(|error| {
                    crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("pair residual failed: {error}"),
                    }
                })?;
                let actual = target_support
                    .matrices
                    .get(&lattice_vector)
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros(predicted.dim()));
                for (left, right) in actual.iter().zip(predicted.iter()) {
                    pair_residual = pair_residual.max((left - right).norm());
                }
            }
        }
    }
    let acceptance = parameters.tolerances.absolute
        + parameters.tolerances.relative * support_a.max_element.max(support_b.max_element);
    if pair_residual > acceptance {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: format!(
                "pair projection covariance residual {pair_residual:e} exceeds acceptance {acceptance:e}"
            ),
        });
    }
    Ok((symmetrized_a, symmetrized_b, change, pair_residual))
}

fn validate_pair_models<R: RMatrixData>(
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
) -> Result<()> {
    if a.nsta() != b.nsta() || a.norb() != b.norb() || a.natom() != b.natom() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "pair models must have identical basis sizes and atom counts".into(),
        });
    }
    if a.lat != b.lat || a.orb_projection != b.orb_projection {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "pair models must have identical lattices and orbital projections".into(),
        });
    }
    for (left, right) in a.atoms.iter().zip(b.atoms.iter()) {
        if left.atom_type() != right.atom_type() || left.position_ref() != right.position_ref() {
            return Err(crate::TbError::TargetMagneticGroupIncompatible {
                reason: "pair models must have identical atom positions and types".into(),
            });
        }
    }
    Ok(())
}

fn pair_operations_from_magnetic(group: &MagneticCrystalSymmetry) -> Vec<PairOperation> {
    group
        .operations
        .iter()
        .map(|operation| PairOperation {
            spatial: *operation,
            channel: if operation.time_reversal {
                CollinearAction::Swap
            } else {
                CollinearAction::Preserve
            },
        })
        .collect()
}

fn pair_operations_from_spin_group(group: &CollinearSpinGroup) -> Vec<PairOperation> {
    group
        .operations()
        .iter()
        .map(|operation| PairOperation {
            spatial: CrystalSymmetryOperation {
                rotation: operation.rotation,
                translation: operation.translation,
                time_reversal: operation.kind.is_antiunitary(),
            },
            channel: operation.action,
        })
        .collect()
}

fn pair_operations_from_unitary_spin_group(group: &CollinearSpinGroup) -> Vec<PairOperation> {
    group
        .operations()
        .iter()
        .filter(|operation| operation.kind.is_unitary())
        .map(|operation| PairOperation {
            spatial: CrystalSymmetryOperation {
                rotation: operation.rotation,
                translation: operation.translation,
                time_reversal: false,
            },
            channel: operation.action,
        })
        .collect()
}

fn symmetrize_pair_with_operations<R: RMatrixData>(
    a: &mut Model<false, 3, R>,
    b: &mut Model<false, 3, R>,
    operations: &[PairOperation],
    basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<PairSymmetrizationReport> {
    validate_tolerances(parameters.tolerances)?;
    validate_pair_models(a, b)?;
    if operations.is_empty() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "target spin group has no operations".into(),
        });
    }
    let margin = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec)
        * 3.0;
    let original_up = a.clone();
    let original_down = b.clone();
    let mut up = original_up.clone();
    let mut down = original_down.clone();
    let mut up_centres = up.orb.clone();
    let mut down_centres = down.orb.clone();
    for _ in 0..64 {
        let old_up = up_centres.clone();
        let old_down = down_centres.clone();
        pair_centre_average(
            &mut up,
            PairSide::Up,
            &old_up,
            &old_down,
            operations,
            basis,
            margin,
        )?;
        pair_centre_average(
            &mut down,
            PairSide::Down,
            &old_up,
            &old_down,
            operations,
            basis,
            margin,
        )?;
        let delta = centre_report(&old_up, &up.orb, &up.atoms, &up.lat, margin)
            .max_displacement_angstrom
            .max(
                centre_report(&old_down, &down.orb, &down.atoms, &down.lat, margin)
                    .max_displacement_angstrom,
            );
        up_centres = up.orb.clone();
        down_centres = down.orb.clone();
        if delta < 1e-9 {
            break;
        }
    }
    let up_centres = centre_report(&original_up.orb, &up.orb, &up.atoms, &up.lat, margin);
    let down_centres = centre_report(
        &original_down.orb,
        &down.orb,
        &down.atoms,
        &down.lat,
        margin,
    );
    let (projected_up, projected_down, change, pair_covariance) =
        pair_project_hamiltonian(&up, &down, operations, basis, parameters)?;
    *a = projected_up;
    *b = projected_down;
    Ok(PairSymmetrizationReport {
        up_centres,
        down_centres,
        hamiltonian_change_max_absolute: change,
        pair_covariance_max_absolute: pair_covariance,
    })
}

impl<R: RMatrixData> Model<false, 3, R> {
    /// Project centres and then the Hamiltonian in the required order.
    ///
    /// 1. Project `orb` over the spatial parts of every target operation.
    /// 2. Resolve the localized actions again on the projected centres.
    /// 3. Run the ordinary strict Hamiltonian Reynolds projection.
    /// 4. Re-resolve every action and report the final covariance residual.
    ///
    /// The centre warning threshold is three times
    /// `parameters.tolerances.position` (a Cartesian length in Angstrom,
    /// falling back to `symprec`). The target group is still checked against the model's
    /// atoms, moments, and external-field context by the underlying
    /// [`Self::symmetrize_hamiltonian`] call. The input model is left unchanged
    /// if any stage fails.
    pub fn symmetrize_model_and_centres(
        &mut self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<SymmetrizationReport> {
        let old = self.clone();
        let margin = parameters
            .tolerances
            .position
            .unwrap_or(parameters.structural_parameters.symprec)
            * 3.0;
        let mut working = self.clone();
        let centres = working.symmetrize_centres(target_group, basis, margin)?;
        let symmetrized = working.symmetrize_hamiltonian(target_group, basis, parameters)?;
        let change = max_map_change(&model_ham_blocks(&old), &model_ham_blocks(&symmetrized));
        let covariance = target_covariance_max(&symmetrized, target_group, basis, parameters)?;
        *self = symmetrized;
        Ok(SymmetrizationReport {
            centres,
            hamiltonian_change_max_absolute: change,
            final_covariance_max_absolute: covariance,
        })
    }

    /// Project two spinless models that represent the two spin channels of a
    /// magnetic group.
    ///
    /// For a unitary target operation g the contribution to each model is its
    /// own localized Reynolds image. For an anti-unitary (spin-flip) operation
    /// the contribution to model a is the image of model b and the contribution
    /// to model b is the image of model a:
    ///
    /// ```text
    /// H_a' = (1/N) [ sum_unitary P_g H_a + sum_anti P_g H_b ]
    /// H_b' = (1/N) [ sum_unitary P_g H_b + sum_anti P_g H_a ]
    /// ```
    ///
    /// This is the pair corepresentation induced by the magnetic group; it is
    /// intentionally not the difference combination that would symmetrize the
    /// two channels against each other for unitary operations. Centres are
    /// projected jointly by the same unitary/anti-unitary split, so up and down
    /// centres become mutual spin-flip images. The function iterates the joint
    /// centre projection until the centre update is below 1e-9 Angstrom (or a
    /// fixed safety limit is reached), then projects both Hamiltonians and
    /// checks the pair covariance.
    ///
    /// Both models must have the same basis size, atom count, atom ordering,
    /// lattice, and orbital projections. The input models are unchanged if any
    /// stage fails. Use the spin-flip flag [`crate::CrystalSymmetryOperation::time_reversal`]
    /// in the target group to identify anti-unitary pair operations.
    ///
    /// This legacy entry maps `time_reversal = true` to a simultaneous channel
    /// swap and orbital conjugation. For cryspglib-certified collinear spin
    /// groups, where a unitary Swap is distinct from an antiunitary operation,
    /// use [`Self::symmetrize_pair_with_collinear_spin_group`] or
    /// [`Self::symmetrize_pair_with_unitary_collinear_spin_group`].
    pub fn symmetrize_pair(
        a: &mut Self,
        b: &mut Self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<PairSymmetrizationReport> {
        validate_tolerances(parameters.tolerances)?;
        validate_pair_models(a, b)?;
        let target_group = validate_pair_group(target_group, parameters.tolerances.operation)?;
        let operations = pair_operations_from_magnetic(&target_group);
        symmetrize_pair_with_operations(a, b, &operations, basis, parameters)
    }

    /// Project two spinless channel models under a complete collinear spin
    /// corepresentation, including the spin-only C2T antiunitary extension.
    ///
    /// The supplied [cryspglib::spin_group::CollinearSpinGroup] is the complete
    /// validated group: its unitary half and the antiunitary g A0 partners.
    /// The projection therefore averages over both the channel action
    /// (Preserve/Swap) and the independent antiunitarity flag. In particular a
    /// unitary Swap exchanges the two Hamiltonians without conjugating them.
    ///
    /// Models must have identical basis sizes, atom ordering, lattice, and
    /// orbital projections. The operation group is already geometrically
    /// certified by cryspglib; Rustb still resolves and validates the localized
    /// orbital actions and checks the final pair covariance.
    pub fn symmetrize_pair_with_collinear_spin_group(
        a: &mut Self,
        b: &mut Self,
        group: &CollinearSpinGroup,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<PairSymmetrizationReport> {
        validate_tolerances(parameters.tolerances)?;
        validate_pair_models(a, b)?;
        let operations = pair_operations_from_spin_group(group);
        symmetrize_pair_with_operations(a, b, &operations, basis, parameters)
    }

    /// Project two spinless channel models under the unitary collinear spin
    /// subgroup only.
    ///
    /// This is the G-only counterpart of
    /// [Self::symmetrize_pair_with_collinear_spin_group]. The complete
    /// [cryspglib::spin_group::CollinearSpinGroup] is accepted, but only its
    /// unitary operations are projected; the A0 antiunitary partners are
    /// filtered out. Use it when orbital time-reversal breaking is known or
    /// when only the unitary subgroup should be imposed.
    pub fn symmetrize_pair_with_unitary_collinear_spin_group(
        a: &mut Self,
        b: &mut Self,
        group: &CollinearSpinGroup,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<PairSymmetrizationReport> {
        validate_tolerances(parameters.tolerances)?;
        validate_pair_models(a, b)?;
        let operations = pair_operations_from_unitary_spin_group(group);
        symmetrize_pair_with_operations(a, b, &operations, basis, parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Solve;
    use crate::crystal_symmetry::{CrystalSymmetry, SymmetryParameters};
    use crate::wannier90::Wannier90;
    use ndarray::array;

    #[test]
    #[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
    fn debug_real_pair_actions() {
        let base = "/mnt/e/vasp计算结果/altermagnetic/odd_parity_wave/material/MgFe6Ge6/data_new";
        let up = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/up/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let dn = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/dn/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let mut moments = vec![[0.0, 0.0, 0.0]; up.natom()];
        for (index, atom) in up.atoms.iter().enumerate() {
            if format!("{:?}", atom.atom_type()).contains("Fe") {
                moments[index] = if atom.position_ref()[2] < 0.5 {
                    [0.0, 0.0, 1.0]
                } else {
                    [0.0, 0.0, -1.0]
                };
            }
        }
        let group = up
            .magnetic_crystal_symmetry(&moments, &SymmetryParameters::default())
            .unwrap();
        let mut a = up.clone();
        let mut b = dn.clone();
        let source_up = a.clone();
        let source_down = b.clone();
        let operations = pair_operations_from_magnetic(&group);
        pair_centre_average(
            &mut a,
            PairSide::Up,
            &source_up.orb,
            &source_down.orb,
            &operations,
            &CenterResolvedBasis::new(),
            0.05,
        )
        .unwrap();
        pair_centre_average(
            &mut b,
            PairSide::Down,
            &source_down.orb,
            &source_up.orb,
            &operations,
            &CenterResolvedBasis::new(),
            0.05,
        )
        .unwrap();
        let basis = CenterResolvedBasis::new();
        let parameters = HamiltonianSymmetrizationParameters::default();
        let position_tolerance = parameters
            .tolerances
            .position
            .unwrap_or(parameters.structural_parameters.symprec);
        let support_a = validate_hamiltonian(&a, parameters.tolerances.hermiticity).unwrap();
        let support_b = validate_hamiltonian(&b, parameters.tolerances.hermiticity).unwrap();
        for (index, operation) in group.operations.iter().enumerate() {
            let action_a = basis
                .resolve(BasisActionContext {
                    model: &a,
                    operation,
                    position_tolerance,
                    representation_tolerance: 1e-6,
                })
                .unwrap();
            let action_a = validate_action(action_a, a.nsta(), 1e-6).unwrap();
            let action_b = basis
                .resolve(BasisActionContext {
                    model: &b,
                    operation,
                    position_tolerance,
                    representation_tolerance: 1e-6,
                })
                .unwrap();
            let action_b = validate_action(action_b, b.nsta(), 1e-6).unwrap();
            if operation.time_reversal {
                let domain = transformed_support_domain(&support_b, operation, &action_b).unwrap();
                let mut max = 0.0f64;
                for vector in domain {
                    let predicted =
                        transformed_hamiltonian_at(&support_b, operation, &action_b, vector, false)
                            .unwrap();
                    let actual = support_a
                        .matrices
                        .get(&vector)
                        .cloned()
                        .unwrap_or_else(|| Array2::zeros((a.nsta(), a.nsta())));
                    for (left, right) in actual.iter().zip(predicted.iter()) {
                        max = max.max((left - right).norm());
                    }
                }
                println!("op {index} anti pair residual {max:e}");
            } else {
                let residual =
                    hamiltonian_residual(&support_a, operation, &action_a, parameters.tolerances)
                        .unwrap();
                let residual_b =
                    hamiltonian_residual(&support_b, operation, &action_b, parameters.tolerances)
                        .unwrap();
                println!(
                    "op {index} unitary residuals {:.3e} {:.3e}",
                    residual.max_absolute, residual_b.max_absolute
                );
            }
        }
    }

    #[test]
    #[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
    fn debug_standard_actions_snapped() {
        let base = "/mnt/e/vasp计算结果/altermagnetic/odd_parity_wave/material/MgFe6Ge6/data_new";
        let model = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/up/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let mut snapped = model.clone();
        for atom in &snapped.atoms {
            for orbital in atom.orbitals() {
                snapped
                    .orb
                    .row_mut(orbital.index())
                    .assign(&atom.position_ref());
            }
        }
        let ds = snapped
            .crystal_symmetry(&SymmetryParameters::default())
            .unwrap();
        let basis = CenterResolvedBasis::new();
        let atomic = AtomicOrbitalBasis;
        let parameters = HamiltonianSymmetrizationParameters::default();
        let support = validate_hamiltonian(&snapped, parameters.tolerances.hermiticity).unwrap();
        for (index, operation) in ds.operations.iter().enumerate() {
            let context = BasisActionContext {
                model: &snapped,
                operation,
                position_tolerance: 1e-5,
                representation_tolerance: 1e-6,
            };
            let custom = basis.resolve(context).unwrap();
            let custom = validate_action(custom, snapped.nsta(), 1e-6).unwrap();
            let residual =
                hamiltonian_residual(&support, operation, &custom, parameters.tolerances).unwrap();
            let context = BasisActionContext {
                model: &snapped,
                operation,
                position_tolerance: 1e-5,
                representation_tolerance: 1e-6,
            };
            let standard = atomic.resolve(context).unwrap();
            let standard = validate_action(standard, snapped.nsta(), 1e-6).unwrap();
            let standard_residual =
                hamiltonian_residual(&support, operation, &standard, parameters.tolerances)
                    .unwrap();
            println!(
                "snapped op {index} custom {:.3e} atomic {:.3e}",
                residual.max_absolute, standard_residual.max_absolute
            );
        }
    }

    #[test]
    #[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
    fn debug_pair_flags() {
        let base = "/mnt/e/vasp计算结果/altermagnetic/odd_parity_wave/material/MgFe6Ge6/data_new";
        let up = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/up/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let dn = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/dn/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let snap = |model: &Model<false, 3>| {
            let mut snapped = model.clone();
            for atom in &snapped.atoms {
                for orbital in atom.orbitals() {
                    snapped
                        .orb
                        .row_mut(orbital.index())
                        .assign(&atom.position_ref());
                }
            }
            snapped
        };
        let up = snap(&up);
        let dn = snap(&dn);
        let ds = up.crystal_symmetry(&SymmetryParameters::default()).unwrap();
        let support_up = validate_hamiltonian(&up, 1e-8).unwrap();
        let support_dn = validate_hamiltonian(&dn, 1e-8).unwrap();
        let basis = AtomicOrbitalBasis;
        for (index, operation) in ds.operations.iter().enumerate() {
            let resolve = |model: &Model<false, 3>| {
                let action = basis
                    .resolve(BasisActionContext {
                        model,
                        operation,
                        position_tolerance: 1e-5,
                        representation_tolerance: 1e-6,
                    })
                    .unwrap();
                validate_action(action, model.nsta(), 1e-6).unwrap()
            };
            let action_up = resolve(&up);
            let action_dn = resolve(&dn);
            let unitary = hamiltonian_residual(
                &support_up,
                operation,
                &action_up,
                HamiltonianSymmetryTolerances {
                    absolute: 1e-8,
                    relative: 0.0,
                    ..Default::default()
                },
            )
            .unwrap()
            .max_absolute;
            let domain = transformed_support_domain(&support_dn, operation, &action_dn).unwrap();
            let mut anti = 0.0f64;
            for vector in domain {
                let predicted =
                    transformed_hamiltonian_at(&support_dn, operation, &action_dn, vector, false)
                        .unwrap();
                let actual = support_up
                    .matrices
                    .get(&vector)
                    .cloned()
                    .unwrap_or_else(|| Array2::zeros((up.nsta(), up.nsta())));
                for (left, right) in actual.iter().zip(predicted.iter()) {
                    anti = anti.max((left - right).norm());
                }
            }
            println!("flag op {index} unitary {unitary:.3e} anti {anti:.3e}");
        }
    }

    fn displaced_p_model() -> Model<false, 3> {
        let lattice = Array2::eye(3);
        let centres = array![
            [0.013, -0.007, 0.004],
            [-0.005, 0.011, -0.003],
            [0.002, 0.006, 0.009]
        ];
        let atom = crate::Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            crate::AtomType::Si,
            [
                crate::OrbitalId::new(0),
                crate::OrbitalId::new(1),
                crate::OrbitalId::new(2),
            ],
        );
        let mut model = Model::<false, 3>::tb_model(lattice, centres, Some(vec![atom])).unwrap();
        model.orb_projection = vec![crate::OrbProj::px, crate::OrbProj::py, crate::OrbProj::pz];
        model
    }

    #[test]
    fn centre_symmetrization_moves_displaced_shell_to_atom_and_is_idempotent() {
        let mut model = displaced_p_model();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&crate::SymmetryParameters::default())
            .unwrap();
        let first = model
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.05)
            .unwrap();
        assert!(first.max_displacement_angstrom > 1e-3);
        let second = model
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.05)
            .unwrap();
        assert!(second.max_displacement_angstrom < 1e-8);
    }

    #[test]
    fn center_resolved_basis_resolves_cubic_p_rotation() {
        let model = displaced_p_model();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&crate::SymmetryParameters::default())
            .unwrap();
        let operation = group
            .operations
            .iter()
            .find(|operation| {
                !operation.time_reversal && operation.rotation == [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
            })
            .expect("cubic p model has a C4z operation");
        let action = CenterResolvedBasis::new()
            .resolve(BasisActionContext {
                model: &model,
                operation,
                position_tolerance: 1e-6,
                representation_tolerance: 1e-6,
            })
            .unwrap();
        let action = validate_action(action, model.nsta(), 1e-6).unwrap();
        assert!(!action.sectors.is_empty());
        assert!(
            action
                .sectors
                .iter()
                .all(|sector| sector.matrix.iter().all(|value| value.norm().is_finite()))
        );
    }

    #[test]
    fn joint_symmetrization_accepts_center_resolved_zero_hamiltonian() {
        let mut model = displaced_p_model();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&crate::SymmetryParameters::default())
            .unwrap();
        let report = model
            .symmetrize_model_and_centres(
                &group,
                &CenterResolvedBasis::new(),
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        assert!(report.final_covariance_max_absolute < 1e-10);
        assert!(report.hamiltonian_change_max_absolute < 1e-12);
    }

    #[test]
    fn centre_symmetrization_rejects_incompatible_target_operations() {
        let mut model = displaced_p_model();
        let mut group = model
            .magnetic_crystal_symmetry_from_atoms(&crate::SymmetryParameters::default())
            .unwrap();
        group.operations[0].translation = [0.123, 0.0, 0.0];
        assert!(
            model
                .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.05)
                .is_err()
        );
    }

    fn real_pair_models() -> (Model<false, 3>, Model<false, 3>, MagneticCrystalSymmetry) {
        let base = "/mnt/e/vasp计算结果/altermagnetic/odd_parity_wave/material/MgFe6Ge6/data_new";
        let up = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/up/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let dn = <Model<false, 3> as Wannier90>::from_hr(
            &format!("{base}/dn/"),
            "wannier90",
            7.62075874,
        )
        .unwrap();
        let moments = vec![[0.0, 0.0, 1.0]; up.natom()];
        let structure = up.crystal_symmetry(&SymmetryParameters::default()).unwrap();
        let mut group = up
            .magnetic_crystal_symmetry(&moments, &SymmetryParameters::default())
            .unwrap();
        for operation in &mut group.operations {
            if let Some(source) = structure.operations.iter().find(|candidate| {
                candidate.rotation == operation.rotation
                    && candidate.translation == operation.translation
            }) {
                operation.time_reversal = source.rotation[2][2] == -1;
            }
        }
        group.field_preserving_operations = group.operations.clone();
        (up, dn, group)
    }

    fn partition_centres(
        up: &Model<false, 3>,
        dn: &Model<false, 3>,
        group: &MagneticCrystalSymmetry,
    ) -> f64 {
        let mut max = 0.0f64;
        for operation in &group.operations {
            let source = if operation.time_reversal { dn } else { up };
            for source_orbital in 0..source.norb() {
                let centre = [
                    source.orb[[source_orbital, 0]],
                    source.orb[[source_orbital, 1]],
                    source.orb[[source_orbital, 2]],
                ];
                let transformed: [f64; 3] = std::array::from_fn(|row| {
                    operation.translation[row]
                        + (0..3)
                            .map(|column| operation.rotation[row][column] as f64 * centre[column])
                            .sum::<f64>()
                });
                let mut best = f64::INFINITY;
                for target_orbital in 0..up.norb() {
                    if up.orb_projection[target_orbital] != source.orb_projection[source_orbital] {
                        continue;
                    }
                    let mut delta = [0.0; 3];
                    for axis in 0..3 {
                        let mut value = transformed[axis] - up.orb[[target_orbital, axis]];
                        value -= value.round();
                        delta[axis] = value;
                    }
                    let cart = Array1::from_vec(delta.to_vec()).dot(&up.lat);
                    best = best.min(cart.dot(&cart).sqrt());
                }
                max = max.max(best);
            }
        }
        max
    }

    fn spectrum_pair_error(
        up: &Model<false, 3>,
        dn: &Model<false, 3>,
        group: &MagneticCrystalSymmetry,
        kpoints: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let (e_up, _) = up.solve_all_parallel(kpoints).unwrap();
        let (e_dn, _) = dn.solve_all_parallel(kpoints).unwrap();
        let mut max = 0.0f64;
        let mut band_change = 0.0f64;
        for operation in &group.operations {
            if !operation.time_reversal {
                continue;
            }
            for ik in 0..kpoints.nrows() {
                let image = Array1::from_shape_fn(3, |row| {
                    -(0..3)
                        .map(|column| {
                            operation.rotation[column][row] as f64 * kpoints[[ik, column]]
                        })
                        .sum::<f64>()
                });
                let mut best = f64::INFINITY;
                let mut image_index = 0usize;
                for jk in 0..kpoints.nrows() {
                    let mut delta = [0.0; 3];
                    for axis in 0..3 {
                        let mut value = image[axis] - kpoints[[jk, axis]];
                        value -= value.round();
                        delta[axis] = value;
                    }
                    let norm = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
                    if norm < best {
                        best = norm;
                        image_index = jk;
                    }
                }
                if best < 1e-8 {
                    for band in 0..e_up.ncols() {
                        max = max.max((e_up[[ik, band]] - e_dn[[image_index, band]]).abs());
                    }
                }
            }
        }
        band_change = 0.0;
        (max, band_change, 0.0)
    }

    #[test]
    #[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
    fn mgfe6ge6_pair_acceptance() {
        let (mut up, mut dn, group) = real_pair_models();
        let before_up = up.clone();
        let before_dn = dn.clone();
        let params = HamiltonianSymmetrizationParameters::default();
        let report = Model::<false, 3>::symmetrize_pair(
            &mut up,
            &mut dn,
            &group,
            &CenterResolvedBasis::without_hamiltonian_inference(),
            &params,
        )
        .unwrap();
        let centre_error = partition_centres(&up, &dn, &group);
        let kpoints = array![
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.0, 0.25, 0.0],
            [0.0, 0.0, 0.25],
        ];
        let (spectrum_error, _spectrum_change, _) = spectrum_pair_error(&up, &dn, &group, &kpoints);
        let (before_up_e, _) = before_up.solve_all_parallel(&kpoints).unwrap();
        let (before_dn_e, _) = before_dn.solve_all_parallel(&kpoints).unwrap();
        let (after_up_e, _) = up.solve_all_parallel(&kpoints).unwrap();
        let (after_dn_e, _) = dn.solve_all_parallel(&kpoints).unwrap();
        let mut band_change = 0.0f64;
        for ik in 0..kpoints.nrows() {
            for band in 0..before_up_e.ncols() {
                band_change =
                    band_change.max((before_up_e[[ik, band]] - after_up_e[[ik, band]]).abs());
                band_change =
                    band_change.max((before_dn_e[[ik, band]] - after_dn_e[[ik, band]]).abs());
            }
        }
        let (gamma_up, _) = up.solve_all_parallel(&array![[0.0, 0.0, 0.0]]).unwrap();
        let (gamma_dn, _) = dn.solve_all_parallel(&array![[0.0, 0.0, 0.0]]).unwrap();
        let mut gamma = 0.0f64;
        for band in 0..gamma_up.ncols() {
            gamma = gamma.max((gamma_up[[0, band]] - gamma_dn[[0, band]]).abs());
        }
        let (before_up2, before_dn2) = (up.clone(), dn.clone());
        let _ = Model::<false, 3>::symmetrize_pair(
            &mut up,
            &mut dn,
            &group,
            &CenterResolvedBasis::without_hamiltonian_inference(),
            &params,
        )
        .unwrap();
        let mut orb_change = 0.0f64;
        let mut ham_change = 0.0f64;
        for orbital in 0..up.norb() {
            for axis in 0..3 {
                orb_change = orb_change
                    .max((before_up2.orb[[orbital, axis]] - up.orb[[orbital, axis]]).abs())
                    .max((before_dn2.orb[[orbital, axis]] - dn.orb[[orbital, axis]]).abs());
            }
        }
        for (left, right) in before_up2.ham.iter().zip(up.ham.iter()) {
            ham_change = ham_change.max((left - right).norm());
        }
        for (left, right) in before_dn2.ham.iter().zip(dn.ham.iter()) {
            ham_change = ham_change.max((left - right).norm());
        }
        assert_eq!(up.lat, before_up.lat);
        assert_eq!(dn.lat, before_dn.lat);
        assert_eq!(up.atoms.len(), before_up.atoms.len());
        println!(
            "REAL centre displacement up {:.6e} dn {:.6e}",
            report.up_centres.max_displacement_angstrom,
            report.down_centres.max_displacement_angstrom
        );
        println!("REAL centre covariance {centre_error:.6e} A");
        println!("REAL spectrum error {spectrum_error:.6e} eV, Gamma {gamma:.6e} eV");
        println!(
            "REAL band change {band_change:.6e} eV, H change {:.6e}",
            report.hamiltonian_change_max_absolute
        );
        println!(
            "REAL pair covariance {:.6e}, idempotent orb {orb_change:.3e} ham {ham_change:.3e}",
            report.pair_covariance_max_absolute
        );
        assert!(centre_error < 1e-8);
        assert!(spectrum_error < 1e-5);
        assert!(gamma < 1e-10);
        assert!(report.pair_covariance_max_absolute < 1e-6);
        assert!(orb_change < 1e-9);
        assert!(ham_change < 1e-8);
    }

    #[test]
    fn pair_symmetrization_accepts_two_identical_zero_models() {
        let mut up = displaced_p_model();
        let mut down = up.clone();
        let group = up
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let report = Model::<false, 3>::symmetrize_pair(
            &mut up,
            &mut down,
            &group,
            &CenterResolvedBasis::new(),
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap();
        assert!(report.pair_covariance_max_absolute < 1e-10);
        assert!(report.hamiltonian_change_max_absolute < 1e-12);
    }

    #[test]
    fn centre_projection_uses_resolved_action_mapping() {
        let mut model = displaced_p_model();
        model.orb.row_mut(0).assign(&array![0.013, -0.007, 0.020]);
        model.orb.row_mut(1).assign(&array![-0.005, 0.011, -0.020]);
        let full = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let c4 = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
        let c4_inv = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]];
        let c2 = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]];
        let identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        let mut group = full.clone();
        group.operations.retain(|operation| {
            !operation.time_reversal && [c4, c4_inv, c2, identity].contains(&operation.rotation)
        });
        group.field_preserving_operations = group.operations.clone();
        assert_eq!(group.operations.len(), 4);
        model
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.05)
            .unwrap();
        let operation = group
            .operations
            .iter()
            .find(|operation| operation.rotation == c4)
            .unwrap();
        let action = CenterResolvedBasis::new()
            .resolve(BasisActionContext {
                model: &model,
                operation,
                position_tolerance: 1e-6,
                representation_tolerance: 1e-8,
            })
            .unwrap();
        let mut best = (usize::MAX, 0.0f64);
        for source in 0..model.norb() {
            let mut coefficient = 0.0f64;
            for sector in &action.sectors {
                coefficient = coefficient.max(sector.matrix[[1, source]].norm());
            }
            if coefficient > best.1 {
                best = (source, coefficient);
            }
        }
        assert_eq!(best.0, 0);
        let source_center = [model.orb[[0, 0]], model.orb[[0, 1]], model.orb[[0, 2]]];
        let rotated: [f64; 3] = std::array::from_fn(|row| {
            (0..3)
                .map(|column| operation.rotation[row][column] as f64 * source_center[column])
                .sum::<f64>()
        });
        let target_center = [model.orb[[1, 0]], model.orb[[1, 1]], model.orb[[1, 2]]];
        let delta = minimum_image_delta(&rotated, &target_center, &model.lat);
        assert!(cartesian_norm(&delta, &model.lat) < 1e-8);
    }
}
