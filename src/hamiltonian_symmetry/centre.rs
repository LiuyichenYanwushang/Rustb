use super::*;
use crate::crystal_symmetry::{CrystalSymmetryOperation, MagneticCrystalSymmetry};
use crate::model::Model;
use cryspglib::spin_group::{CollinearAction, CollinearSpinGroup};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};

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

// Search the rounded fractional image and its 26 neighbours. This is not
// an exact closest-vector search for highly skew unreduced lattices.
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
) -> Result<Vec<(usize, [f64; 3])>> {
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
            let target_center = std::array::from_fn(|axis| model.orb[[target_orbital, axis]]);
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
            // Average the displacement in the target orbital's existing cell
            // gauge, not the parent's cell: H and r have not been reindexed.
            let delta = minimum_image_delta(&image, &target_center, &model.lat);
            entries.push((target_orbital, delta));
        }
    }
    Ok(entries)
}

fn resolve_centre_action<R: RMatrixData>(
    target: &Model<false, 3, R>,
    operation: &CrystalSymmetryOperation,
    basis: &impl BasisSymmetryRepresentation<false, R>,
) -> Result<LocalizedBasisAction> {
    basis
        .resolve(BasisActionContext {
            model: target,
            operation,
            position_tolerance: 1e-5,
            representation_tolerance: 1e-6,
        })
        .and_then(|action| validate_action(action, target.nsta(), 1e-6))
        .map_err(|error| TbError::TargetMagneticGroupIncompatible {
            reason: format!("centre basis action is invalid: {error}"),
        })
}

fn centre_image_deltas<R: RMatrixData>(
    target: &Model<false, 3, R>,
    source_centres: &Array2<f64>,
    operation: &CrystalSymmetryOperation,
    action: &LocalizedBasisAction,
) -> Result<Vec<(usize, [f64; 3])>> {
    let atom_map = map_atoms(target, operation, 1e-5)?;
    centre_permutation(target, source_centres, operation, &atom_map, Some(action))
}

fn check_centre_image<R: RMatrixData>(
    target: &Model<false, 3, R>,
    source_centres: &Array2<f64>,
    operation: &CrystalSymmetryOperation,
    action: &LocalizedBasisAction,
    tolerance: f64,
) -> Result<()> {
    for (orbital, delta) in centre_image_deltas(target, source_centres, operation, action)? {
        let residual = cartesian_norm(&delta, &target.lat);
        if !residual.is_finite() || residual > tolerance {
            return Err(TbError::TargetMagneticGroupIncompatible {
                reason: format!(
                    "orbital centre covariance residual {residual:e} Å at orbital {orbital} under {operation:?} exceeds {tolerance:e} Å"
                ),
            });
        }
    }
    Ok(())
}

fn validate_centre_tolerance(tolerance: f64) -> Result<()> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(TbError::InvalidResponseParameter {
            parameter: "tolerance",
            message: "centre tolerance must be finite and non-negative".into(),
        });
    }
    Ok(())
}

fn max_centre_step(old: &Array2<f64>, new: &Array2<f64>, lattice: &Array2<f64>) -> f64 {
    old.rows()
        .into_iter()
        .zip(new.rows())
        .fold(0.0_f64, |max, (a, b)| {
            let from = std::array::from_fn(|axis| a[axis]);
            let to = std::array::from_fn(|axis| b[axis]);
            max.max(cartesian_norm(
                &minimum_image_delta(&to, &from, lattice),
                lattice,
            ))
        })
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
    /// current target orbital centre, retaining its cell representative. The
    /// final check without a position matrix tests the individual mapped
    /// centres at 1e-5 Å, not just the change in their average.
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
    /// [`crate::TbError::TargetMagneticGroupIncompatible`]. If a position matrix
    /// is present, the projected centres must agree with its unchanged onsite
    /// diagonal within `tolerance` in Å. Only the returned model's orbital
    /// centres can differ; the input, Hamiltonian, and position matrix are unchanged.
    pub fn symmetrize_centres(
        &self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        tolerance: f64,
    ) -> Result<Self> {
        validate_centre_tolerance(tolerance)?;
        let projected = self.project_geometrical_centres(target_group, basis)?;
        projected.validate()?;
        if !R::HAS_RMATRIX {
            super::symmetry_ordered_map(
                (0..target_group.operations.len()).collect::<Vec<_>>(),
                |index| -> Result<()> {
                    let operation = &target_group.operations[index];
                    let action = resolve_centre_action(&projected, operation, basis)?;
                    check_centre_image(&projected, &projected.orb, operation, &action, 1e-5)
                },
            )?;
        }
        validate_position_centres(&projected, tolerance)?;
        Ok(projected)
    }

    // The joint projection uses these centres to resolve its input actions;
    // its final centres may come from the projected position matrix instead.
    fn project_geometrical_centres(
        &self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    ) -> Result<Self> {
        self.validate()?;
        if target_group.operations.is_empty() {
            return Err(crate::TbError::TargetMagneticGroupIncompatible {
                reason: "target magnetic group has no operations".into(),
            });
        }
        let mut projected = self.clone();
        for _ in 0..64 {
            if projected.symmetrize_centres_once(target_group, basis)? < 1e-10 {
                break;
            }
        }
        Ok(projected)
    }

    fn symmetrize_centres_once(
        &mut self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    ) -> Result<f64> {
        let norb = self.norb();
        let old = self.orb.clone();
        let model: &Self = self;
        let per_operation = super::symmetry_ordered_map(
            (0..target_group.operations.len()).collect::<Vec<_>>(),
            |index| -> Result<Vec<(usize, [f64; 3])>> {
                let operation = &target_group.operations[index];
                let action = resolve_centre_action(model, operation, basis)?;
                centre_image_deltas(model, &old, operation, &action)
            },
        )?;
        // One delta vector per operation is collected before the ordered
        // reduction; this is O(|group| * norb), small next to the projection.
        let mut accumulated: Vec<(f64, [f64; 3])> = vec![(0.0, [0.0; 3]); norb];
        for entries in per_operation {
            for (target_orbital, delta) in entries {
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
            for axis in 0..3 {
                new[[orbital, axis]] += sum[axis] / count;
            }
        }
        let step = max_centre_step(&old, &new, &self.lat);
        self.orb = new;
        Ok(step)
    }
}

fn validate_joint_projection<R: RMatrixData>(
    model: &Model<false, 3, R>,
    target: &MagneticCrystalSymmetry,
    basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<()> {
    let support = validate_hamiltonian(model, parameters.tolerances.hermiticity)?;
    let position_tolerance = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec);
    super::symmetry_ordered_map(
        (0..target.operations.len()).collect::<Vec<_>>(),
        |index| -> Result<()> {
            let operation = &target.operations[index];
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
            let action =
                validate_action(action, model.nsta(), parameters.tolerances.representation)
                    .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("post-projection action is invalid: {error}"),
                    })?;
            hamiltonian_residual(&support, operation, &action, parameters.tolerances).map_err(
                |error| crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("post-projection residual failed: {error}"),
                },
            )?;
            if !R::HAS_RMATRIX {
                check_centre_image(model, &model.orb, operation, &action, position_tolerance)?;
            }
            Ok(())
        },
    )?;
    Ok(())
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
    basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
) -> Result<()> {
    if operations.is_empty() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "target group has no operations".into(),
        });
    }
    let old = target.orb.clone();
    let norb = target.norb();
    let model: &Model<false, 3, R> = target;
    let per_operation = super::symmetry_ordered_map(
        (0..operations.len()).collect::<Vec<_>>(),
        |index| -> Result<Vec<(usize, [f64; 3])>> {
            let operation = &operations[index];
            let centres = match (side, operation.channel) {
                (PairSide::Up, CollinearAction::Preserve)
                | (PairSide::Down, CollinearAction::Swap) => up_centres,
                (PairSide::Up, CollinearAction::Swap)
                | (PairSide::Down, CollinearAction::Preserve) => down_centres,
            };
            let action = resolve_centre_action(model, &operation.spatial, basis)?;
            centre_image_deltas(model, centres, &operation.spatial, &action)
        },
    )?;
    let mut sums = vec![[0.0f64; 3]; norb];
    let mut counts = vec![0usize; norb];
    for entries in per_operation {
        for (target_orbital, delta) in entries {
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
        for axis in 0..3 {
            new[[orbital, axis]] += sums[orbital][axis] / counts[orbital] as f64;
        }
    }
    target.orb = new;
    Ok(())
}

fn resolve_pair_actions<R: RMatrixData>(
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
    operations: &[PairOperation],
    basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<Vec<(PairOperation, LocalizedBasisAction, LocalizedBasisAction)>> {
    let position_tolerance = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec);
    let action_tolerance = parameters.tolerances.representation.max(1e-6);
    super::symmetry_ordered_map(
        (0..operations.len()).collect::<Vec<_>>(),
        |index| -> Result<(PairOperation, LocalizedBasisAction, LocalizedBasisAction)> {
            let operation = operations[index];
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
            let action_a =
                validate_action(action_a, a.nsta(), action_tolerance).map_err(|error| {
                    crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("pair action invalid on first model: {error}"),
                    }
                })?;
            let action_b =
                validate_action(action_b, b.nsta(), action_tolerance).map_err(|error| {
                    crate::TbError::TargetMagneticGroupIncompatible {
                        reason: format!("pair action invalid on second model: {error}"),
                    }
                })?;
            Ok((operation, action_a, action_b))
        },
    )
}

fn validate_pair_corepresentations<R: RMatrixData>(
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
    actions: &[(PairOperation, LocalizedBasisAction, LocalizedBasisAction)],
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<()> {
    for (model, side) in [(a, PairSide::Up), (b, PairSide::Down)] {
        let channel_actions = actions
            .iter()
            .map(|(operation, action_a, action_b)| {
                let action = match side {
                    PairSide::Up => action_a,
                    PairSide::Down => action_b,
                };
                (operation.spatial, action.clone())
            })
            .collect::<Vec<_>>();
        validate_projective_corepresentation(
            model,
            &channel_actions,
            parameters.tolerances.operation,
            parameters.tolerances.representation.max(1e-6),
        )
        .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
            reason: format!(
                "{side:?} channel actions are not a magnetic corepresentation: {error}"
            ),
        })?;
    }
    Ok(())
}

fn hermitize_support_map(matrices: &mut BTreeMap<[isize; 3], Array2<Complex64>>) -> Result<()> {
    let dimension = matrices
        .values()
        .next()
        .map_or((0, 0), |matrix| matrix.dim());
    let keys = matrices.keys().copied().collect::<Vec<_>>();
    let mut processed = BTreeSet::new();
    let mut pairs = Vec::new();
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
        pairs.push((lattice_vector, negative));
    }
    // The pairs partition the key set, so each task reads only its own two
    // blocks from the frozen map. Ordered collection keeps the later insertions
    // deterministic; the R=0 self-pair repeats the serial double insert. All
    // pair results are materialized before insertion, so the peak holds one
    // extra support-sized copy: the same order as the returned model, bounded
    // by |hamR| * nsta^2.
    let current: &BTreeMap<[isize; 3], Array2<Complex64>> = matrices;
    let computed = pairs
        .into_par_iter()
        .map(|(lattice_vector, negative)| {
            let matrix = current
                .get(&lattice_vector)
                .cloned()
                .unwrap_or_else(|| Array2::zeros(dimension));
            let partner = current
                .get(&negative)
                .cloned()
                .unwrap_or_else(|| Array2::zeros(dimension));
            let partner_dagger = partner.t().mapv(|value| value.conj());
            let hermitian = (matrix + partner_dagger).mapv(|value| value * 0.5);
            let hermitian_dagger = hermitian.t().mapv(|value| value.conj());
            (lattice_vector, negative, hermitian, hermitian_dagger)
        })
        .collect::<Vec<_>>();
    for (lattice_vector, negative, hermitian, hermitian_dagger) in computed {
        matrices.insert(lattice_vector, hermitian);
        matrices.insert(negative, hermitian_dagger);
    }
    Ok(())
}

fn pair_project_hamiltonian<R: RMatrixData>(
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
    operations: &[PairOperation],
    basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<(Model<false, 3, R>, Model<false, 3, R>)> {
    let support_a = validate_hamiltonian(a, parameters.tolerances.hermiticity)?;
    let support_b = validate_hamiltonian(b, parameters.tolerances.hermiticity)?;
    if R::HAS_RMATRIX {
        position_origin(a)?;
        position_origin(b)?;
    }
    let actions = resolve_pair_actions(a, b, operations, basis, parameters)?;
    if actions.is_empty() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "target group has no operations".into(),
        });
    }
    validate_pair_corepresentations(a, b, &actions, parameters)?;
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
    let domain = domain.into_iter().collect::<Vec<_>>();
    let blocks = symmetry_ordered_map(domain, |lattice_vector| {
        // Keep the original operation order within each block.
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
        matrix_a.mapv_inplace(|value| value / normalizer);
        matrix_b.mapv_inplace(|value| value / normalizer);
        Ok((lattice_vector, matrix_a, matrix_b))
    })?;
    let mut projected_a = BTreeMap::new();
    let mut projected_b = BTreeMap::new();
    for (lattice_vector, matrix_a, matrix_b) in blocks {
        projected_a.insert(lattice_vector, matrix_a);
        projected_b.insert(lattice_vector, matrix_b);
    }
    hermitize_support_map(&mut projected_a)?;
    hermitize_support_map(&mut projected_b)?;
    let position_a = model_position_support(a);
    let position_b = model_position_support(b);
    let projected_positions =
        if let (Some(position_a), Some(position_b)) = (position_a.as_ref(), position_b.as_ref()) {
            let domain = projected_a
                .keys()
                .chain(projected_b.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            let up_sources = actions
                .iter()
                .map(|(operation, action_a, _)| PositionSource {
                    operation: &operation.spatial,
                    action: action_a,
                    support: match operation.channel {
                        CollinearAction::Preserve => position_a,
                        CollinearAction::Swap => position_b,
                    },
                })
                .collect::<Vec<_>>();
            let down_sources = actions
                .iter()
                .map(|(operation, _, action_b)| PositionSource {
                    operation: &operation.spatial,
                    action: action_b,
                    support: match operation.channel {
                        CollinearAction::Preserve => position_b,
                        CollinearAction::Swap => position_a,
                    },
                })
                .collect::<Vec<_>>();
            Some((
                project_position_support_with_sources(a, &domain, &up_sources)?,
                project_position_support_with_sources(b, &domain, &down_sources)?,
            ))
        } else {
            None
        };
    let mut symmetrized_a = model_with_hamiltonian_support(
        a,
        projected_a,
        projected_positions.as_ref().map(|(position, _)| position),
    )?;
    let mut symmetrized_b = model_with_hamiltonian_support(
        b,
        projected_b,
        projected_positions.as_ref().map(|(_, position)| position),
    )?;
    let position_tolerance = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec);
    synchronize_position_centres(&mut symmetrized_a, position_tolerance)?;
    synchronize_position_centres(&mut symmetrized_b, position_tolerance)?;
    let support_a = validate_hamiltonian(&symmetrized_a, parameters.tolerances.hermiticity)?;
    let support_b = validate_hamiltonian(&symmetrized_b, parameters.tolerances.hermiticity)?;
    // Providers can depend on the final centres and H even without r.
    let actions = resolve_pair_actions(
        &symmetrized_a,
        &symmetrized_b,
        operations,
        basis,
        parameters,
    )?;
    validate_pair_corepresentations(&symmetrized_a, &symmetrized_b, &actions, parameters)?;
    let position_a = model_position_support(&symmetrized_a);
    let position_b = model_position_support(&symmetrized_b);
    let target_a = (&symmetrized_a, &support_a, position_a.as_ref());
    let target_b = (&symmetrized_b, &support_b, position_b.as_ref());
    let checks = actions
        .iter()
        .flat_map(|(operation, action_a, action_b)| {
            let (source_a, source_b) = match operation.channel {
                CollinearAction::Preserve => (target_a, target_b),
                CollinearAction::Swap => (target_b, target_a),
            };
            [
                (operation, target_a, source_a, action_a),
                (operation, target_b, source_b, action_b),
            ]
        })
        .collect();
    // Each check visits the same sorted support and does H before r or centres.
    let pair_residuals = symmetry_ordered_map(
        checks,
        |(
            operation,
            (target_model, target_support, target_position),
            (source_model, source_support, source_position),
            action,
        )| {
            let mut pair_residual = 0.0f64;
            let mut domain = transformed_support_domain(source_support, &operation.spatial, action)
                .map_err(|error| crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!("pair residual failed: {error}"),
                })?;
            domain.extend(target_support.matrices.keys().copied());
            for &lattice_vector in &domain {
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
                    let residual = (left - right).norm();
                    if !residual.is_finite() {
                        return Err(crate::TbError::TargetMagneticGroupIncompatible {
                            reason: format!(
                                "non-finite pair Hamiltonian residual at R={lattice_vector:?}"
                            ),
                        });
                    }
                    pair_residual = pair_residual.max(residual);
                }
            }
            if let (Some(target), Some(source)) = (target_position, source_position) {
                validate_position_covariance(
                    target_model,
                    target,
                    PositionSource {
                        operation: &operation.spatial,
                        action,
                        support: source,
                    },
                    domain,
                    position_tolerance,
                )?;
            } else if !R::HAS_RMATRIX {
                check_centre_image(
                    target_model,
                    &source_model.orb,
                    &operation.spatial,
                    action,
                    position_tolerance,
                )?;
            }
            Ok(pair_residual)
        },
    )?;
    let pair_residual = pair_residuals.into_iter().fold(0.0_f64, f64::max);
    let acceptance = parameters.tolerances.absolute
        + parameters.tolerances.relative * support_a.max_element.max(support_b.max_element);
    if pair_residual > acceptance {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: format!(
                "pair projection covariance residual {pair_residual:e} exceeds acceptance {acceptance:e}"
            ),
        });
    }
    Ok((symmetrized_a, symmetrized_b))
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
    a: &Model<false, 3, R>,
    b: &Model<false, 3, R>,
    operations: &[PairOperation],
    basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
    parameters: &HamiltonianSymmetrizationParameters,
) -> Result<(Model<false, 3, R>, Model<false, 3, R>)> {
    validate_tolerances(parameters.tolerances)?;
    validate_pair_models(a, b)?;
    if operations.is_empty() {
        return Err(crate::TbError::TargetMagneticGroupIncompatible {
            reason: "target spin group has no operations".into(),
        });
    }
    let position_tolerance = parameters
        .tolerances
        .position
        .unwrap_or(parameters.structural_parameters.symprec);
    if !position_tolerance.is_finite() || position_tolerance <= 0.0 {
        return Err(crate::TbError::InvalidHamiltonianSymmetryInput {
            parameter: "position_tolerance",
            message: "must be a finite positive Cartesian length".into(),
        });
    }
    a.validate()?;
    b.validate()?;
    if operations
        .iter()
        .any(|op| op.channel == CollinearAction::Swap)
    {
        for (orbital, (left, right)) in a.orb.rows().into_iter().zip(b.orb.rows()).enumerate() {
            // Swap uses single-channel sewing matrices in a shared cell gauge.
            // Do not apply them to a different or ambiguous representative.
            if left
                .iter()
                .zip(right.iter())
                .any(|(&x, &y)| (x - y).abs() >= 0.5)
            {
                return Err(crate::TbError::TargetMagneticGroupIncompatible {
                    reason: format!(
                        "pair orbital {orbital} has incompatible cell representatives; align the Wannier gauges before swapping channels"
                    ),
                });
            }
        }
    }
    let mut up = a.clone();
    let mut down = b.clone();
    for _ in 0..64 {
        let old_up = up.orb.clone();
        let old_down = down.orb.clone();
        pair_centre_average(&mut up, PairSide::Up, &old_up, &old_down, operations, basis)?;
        pair_centre_average(
            &mut down,
            PairSide::Down,
            &old_up,
            &old_down,
            operations,
            basis,
        )?;
        let delta = max_centre_step(&old_up, &up.orb, &up.lat)
            .max(max_centre_step(&old_down, &down.orb, &down.lat));
        if delta < 1e-9 {
            break;
        }
    }
    pair_project_hamiltonian(&up, &down, operations, basis, parameters)
}

impl<R: RMatrixData> Model<false, 3, R> {
    /// Project centres and then the Hamiltonian in the required order.
    ///
    /// 1. Project `orb` over the spatial parts of every target operation.
    /// 2. Resolve the localized actions again on the projected centres.
    /// 3. Run the strict Reynolds projection, taking the final centres from the
    ///    projected onsite position matrix when one is present.
    /// 4. Check the final model and return it without modifying the input.
    ///
    /// The target group is checked against the model's atoms, moments, and
    /// external-field context by [`Self::symmetrize_hamiltonian`].
    pub fn symmetrize_model_and_centres(
        &self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<Self> {
        validate_centre_tolerance(
            parameters
                .tolerances
                .position
                .unwrap_or(parameters.structural_parameters.symprec),
        )?;
        let working = self.project_geometrical_centres(target_group, basis)?;
        let symmetrized = working.symmetrize_hamiltonian(target_group, basis, parameters)?;
        validate_joint_projection(&symmetrized, target_group, basis, parameters)?;
        Ok(symmetrized)
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
    /// fixed safety limit is reached), then projects both Hamiltonians and any
    /// position matrices. When position matrices are present, their onsite
    /// diagonals determine the final orbital centres. Pair covariance is checked
    /// on the final models.
    ///
    /// Both models must have the same basis size, atom count, atom ordering,
    /// lattice, and orbital projections. Channel swaps require a common
    /// Wannier gauge and corresponding centre differences below half a cell
    /// in each fractional component. Returns two new models in input order
    /// without modifying either input. Use [`crate::CrystalSymmetryOperation::time_reversal`]
    /// in the target group to identify anti-unitary pair operations.
    ///
    /// This legacy entry maps `time_reversal = true` to a simultaneous channel
    /// swap and orbital conjugation. For cryspglib-certified collinear spin
    /// groups, where a unitary Swap is distinct from an antiunitary operation,
    /// use [`Self::symmetrize_pair_with_collinear_spin_group`] or
    /// [`Self::symmetrize_pair_with_unitary_collinear_spin_group`].
    pub fn symmetrize_pair(
        a: &Self,
        b: &Self,
        target_group: &MagneticCrystalSymmetry,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<(Self, Self)> {
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
    /// orbital projections, with the same cell-gauge requirement as
    /// [`Self::symmetrize_pair`]. The operation group is already geometrically
    /// certified by cryspglib; Rustb still resolves and validates the localized
    /// orbital actions and checks the final pair covariance. Returns two new
    /// models in input order, leaving the inputs unchanged.
    pub fn symmetrize_pair_with_collinear_spin_group(
        a: &Self,
        b: &Self,
        group: &CollinearSpinGroup,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<(Self, Self)> {
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
    /// when only the unitary subgroup should be imposed. Returns two new models
    /// in input order, leaving the inputs unchanged.
    pub fn symmetrize_pair_with_unitary_collinear_spin_group(
        a: &Self,
        b: &Self,
        group: &CollinearSpinGroup,
        basis: &(impl BasisSymmetryRepresentation<false, R> + Sync),
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<(Self, Self)> {
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
    use std::sync::atomic::{AtomicU64, Ordering};

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
        )
        .unwrap();
        pair_centre_average(
            &mut b,
            PairSide::Down,
            &source_down.orb,
            &source_up.orb,
            &operations,
            &CenterResolvedBasis::new(),
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

    fn displaced_s_model_with_positions(x: f64) -> Model<false, 3, crate::HasRMatrix> {
        Model::tb_model(
            array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            array![[x, 0.0, 0.0]],
            Some(vec![crate::Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                crate::AtomType::Si,
                [crate::OrbitalId::new(0)],
            )]),
        )
        .unwrap()
    }

    #[test]
    fn cancelling_centre_updates_do_not_certify_symmetry() {
        use crate::{Atom, AtomType, OrbitalId};

        let sites = array![[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]];
        let offsets = [0.0, 1.0 / 3.0, -1.0 / 3.0];
        let centres = Array2::from_shape_fn((3, 3), |(i, j)| sites[[i, j]] + offsets[i]);
        let atoms = sites
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, site)| {
                Atom::with_orbitals(site.to_owned(), AtomType::Si, [OrbitalId::new(i)])
            })
            .collect();
        let model = Model::<false, 3>::tb_model(Array2::eye(3), centres, Some(atoms)).unwrap();
        let mut group = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let cyclic = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        ];
        group.operations.retain(|op| {
            !op.time_reversal
                && op.translation.iter().all(|x| x.abs() < 1e-12)
                && cyclic.contains(&op.rotation)
        });
        group.field_preserving_operations = group.operations.clone();
        assert_eq!(group.operations.len(), 3);
        let error = model
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.1)
            .unwrap_err();
        assert!(
            matches!(error, TbError::TargetMagneticGroupIncompatible { reason }
            if reason.contains("orbital centre covariance residual"))
        );
    }

    #[test]
    fn already_symmetric_hoppings_keep_their_orbital_cell_gauge() {
        use crate::{Atom, AtomType, HasRMatrix, OrbitalId};

        let mut model = Model::<false, 3, HasRMatrix>::tb_model(
            array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            array![[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::Si, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        model.set_hop(1.0, 0, 1, &array![0, 0, 0], None);
        model.set_hop(1.0, 0, 1, &array![1, 0, 0], None);
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let parameters = HamiltonianSymmetrizationParameters::default();
        let only_centres = model
            .symmetrize_centres(&group, &ScalarSiteBasis, 1e-10)
            .unwrap();
        let joint = model
            .symmetrize_model_and_centres(&group, &ScalarSiteBasis, &parameters)
            .unwrap();
        let (up, down) =
            Model::symmetrize_pair(&model, &model, &group, &ScalarSiteBasis, &parameters).unwrap();
        let k = array![0.25, 0.0, 0.0];
        for projected in [only_centres, joint, up, down] {
            assert_eq!(projected.orb, model.orb);
            // Check both supports; an absent block is zero regardless of row order.
            for (left, right) in [(&model, &projected), (&projected, &model)] {
                for (r, vector) in left.hamR.rows().into_iter().enumerate() {
                    let matching = crate::find_R(&right.hamR, &vector);
                    for ((i, j), &hop) in left.ham.index_axis(ndarray::Axis(0), r).indexed_iter() {
                        let other = matching.map_or(Complex64::default(), |s| right.ham[[s, i, j]]);
                        assert!((hop - other).norm() < 1e-12);
                    }
                }
            }
            let hab = projected.gen_ham(&k, crate::Gauge::Lattice)[[0, 1]];
            assert!((hab.norm() - 2.0_f64.sqrt()).abs() < 1e-12);
        }
    }

    #[test]
    fn channel_swap_rejects_different_cell_representatives() {
        let up = displaced_s_model_with_positions(1.0);
        let down = displaced_s_model_with_positions(0.0);
        let group = up
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        assert!(group.operations.iter().any(|g| g.time_reversal));
        let error = Model::symmetrize_pair(
            &up,
            &down,
            &group,
            &ScalarSiteBasis,
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap_err();
        assert!(
            matches!(error, crate::TbError::TargetMagneticGroupIncompatible { reason }
            if reason.contains("pair orbital 0 has incompatible cell representatives"))
        );
    }

    #[test]
    fn centre_only_position_mismatch_leaves_model_unchanged() {
        let model = displaced_s_model_with_positions(0.125);
        let original = model.clone();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        // The projected x centre is zero. Its separation from the onsite r
        // diagonal is 0.25 Å, even though the fractional separation is 0.125.
        let error = model
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.2)
            .unwrap_err();
        assert!(matches!(
            error,
            crate::TbError::TargetMagneticGroupIncompatible { reason }
                if reason.contains("onsite position diagonal")
        ));
        assert_eq!(model.orb, original.orb);
        assert_eq!(model.ham, original.ham);
        assert_eq!(model.hamR, original.hamR);
        assert_eq!(model.rmatrix.0, original.rmatrix.0);
    }

    #[test]
    fn pair_unitary_swap_uses_projected_positions_for_final_centres() {
        let mut up = displaced_s_model_with_positions(0.125);
        let mut down = displaced_s_model_with_positions(-0.125);
        let r_plus = array![1_isize, 0, 0];
        let dipole = Complex64::new(0.0, 0.125);
        for (model, sign) in [(&mut up, 1.0), (&mut down, -1.0)] {
            // The provisional geometrical centres stay at zero, while the
            // position projection keeps the two opposite onsite diagonals.
            model.orb.fill(0.0);
            model.set_hop(Complex64::new(0.0, sign), 0, 0, &r_plus, None);
            for (index, vector) in model.hamR.rows().into_iter().enumerate() {
                model.rmatrix[[index, 0, 0, 0]].im = dipole.im * vector[0] as f64;
            }
        }
        let operations = [
            PairOperation {
                spatial: CrystalSymmetryOperation {
                    rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    translation: [0.0; 3],
                    time_reversal: false,
                },
                channel: CollinearAction::Preserve,
            },
            PairOperation {
                spatial: CrystalSymmetryOperation {
                    rotation: [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                    translation: [0.0; 3],
                    time_reversal: false,
                },
                channel: CollinearAction::Swap,
            },
        ];
        let original_up = up.clone();
        let original_down = down.clone();
        let (projected_up, projected_down) = symmetrize_pair_with_operations(
            &up,
            &down,
            &operations,
            &CenterResolvedBasis::new(),
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap();
        for (model, sign) in [(&projected_up, 1.0), (&projected_down, -1.0)] {
            assert!((model.orb[[0, 0]] - sign * 0.125).abs() < 1e-12);
            assert!((model.rmatrix[[0, 0, 0, 0]].re - sign * 0.25).abs() < 1e-12);
            let plus = model
                .hamR
                .rows()
                .into_iter()
                .position(|r| r.iter().eq(r_plus.iter()))
                .unwrap();
            assert!((model.ham[[plus, 0, 0]] - Complex64::new(0.0, sign)).norm() < 1e-12);
            assert!((model.rmatrix[[plus, 0, 0, 0]] - dipole).norm() < 1e-12);
        }
        for (input, original) in [(&up, &original_up), (&down, &original_down)] {
            assert_eq!(input.orb, original.orb);
            assert_eq!(input.ham, original.ham);
            assert_eq!(input.hamR, original.hamR);
            assert_eq!(input.rmatrix.0, original.rmatrix.0);
        }
    }

    #[test]
    fn centre_symmetrization_moves_displaced_shell_to_atom_and_is_idempotent() {
        let model = displaced_p_model();
        let original_orb = model.orb.clone();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&crate::SymmetryParameters::default())
            .unwrap();
        let projected = model
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.05)
            .unwrap();
        assert!(
            (&projected.orb - &original_orb)
                .iter()
                .any(|d| d.abs() > 1e-3)
        );
        assert_eq!(model.orb, original_orb);
        assert!(projected.orb.iter().all(|value| value.abs() < 1e-8));
        assert_eq!(projected.ham, model.ham);
        assert_eq!(projected.hamR, model.hamR);
        let projected_again = projected
            .symmetrize_centres(&group, &CenterResolvedBasis::new(), 0.05)
            .unwrap();
        assert!(
            (&projected_again.orb - &projected.orb)
                .iter()
                .all(|d| d.abs() < 1e-8)
        );
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
    fn joint_symmetrization_rejects_nonfinite_tolerance_before_resolving_basis() {
        let model = displaced_p_model();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let mut parameters = HamiltonianSymmetrizationParameters::default();
        parameters.tolerances.position = Some(f64::NAN);
        let basis = |_: BasisActionContext<'_, false, crate::NoRMatrix>| -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
            panic!("invalid tolerance must be rejected before basis resolution")
        };
        assert!(matches!(
            model.symmetrize_model_and_centres(&group, &basis, &parameters),
            Err(TbError::InvalidResponseParameter {
                parameter: "tolerance",
                ..
            })
        ));
    }

    #[test]
    fn joint_symmetrization_accepts_center_resolved_zero_hamiltonian() {
        let model = displaced_p_model();
        let original = model.clone();
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&crate::SymmetryParameters::default())
            .unwrap();
        let projected = model
            .symmetrize_model_and_centres(
                &group,
                &CenterResolvedBasis::new(),
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        assert!(projected.ham.iter().all(|hop| hop.norm() < 1e-12));
        assert_eq!(model.orb, original.orb);
        assert_eq!(model.ham, original.ham);
        assert_eq!(model.hamR, original.hamR);
        assert!(projected.orb.iter().all(|value| value.abs() < 1e-8));
    }

    fn real_bits(array: &Array2<f64>) -> Vec<u64> {
        array.iter().map(|value| value.to_bits()).collect()
    }

    fn complex_bits(array: &ndarray::Array3<Complex64>) -> Vec<(u64, u64)> {
        array
            .iter()
            .map(|value| (value.re.to_bits(), value.im.to_bits()))
            .collect()
    }

    fn complex4_bits(array: &ndarray::Array4<Complex64>) -> Vec<(u64, u64)> {
        array
            .iter()
            .map(|value| (value.re.to_bits(), value.im.to_bits()))
            .collect()
    }

    /// Records which Rayon workers actually called resolve, so the parallel
    /// invariance test cannot pass on a silently serial fallback.
    struct ThreadRecordingBasis {
        inner: CenterResolvedBasis,
        seen: AtomicU64,
    }

    impl ThreadRecordingBasis {
        fn new() -> Self {
            Self {
                inner: CenterResolvedBasis::new(),
                seen: AtomicU64::new(0),
            }
        }

        fn worker_count(&self) -> u32 {
            self.seen.load(Ordering::Relaxed).count_ones()
        }

        fn reset(&self) {
            self.seen.store(0, Ordering::Relaxed);
        }
    }

    impl BasisSymmetryRepresentation<false, crate::NoRMatrix> for ThreadRecordingBasis {
        fn resolve(
            &self,
            context: BasisActionContext<'_, false, crate::NoRMatrix>,
        ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
            if let Some(index) = rayon::current_thread_index() {
                if index < 64 {
                    self.seen.fetch_or(1u64 << index, Ordering::Relaxed);
                }
            }
            self.inner.resolve(context)
        }

        fn validates_orbital_centres(&self) -> bool {
            <CenterResolvedBasis as BasisSymmetryRepresentation<false, crate::NoRMatrix>>::validates_orbital_centres(
                &self.inner,
            )
        }
    }

    #[test]
    fn symmetry_projections_are_thread_count_invariant() {
        if cfg!(any(feature = "netlib-static", feature = "netlib-system")) {
            // BLAS-touching symmetry kernels intentionally fall back to serial
            // for reference Netlib CBLAS; skip the worker-count proof here.
            return;
        }
        let model = displaced_p_model();
        let mut down = model.clone();
        down.ham.mapv_inplace(|value| value * 1.5);
        let group = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let basis = ThreadRecordingBasis::new();
        let parameters = HamiltonianSymmetrizationParameters::default();
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let centres_serial = serial
            .install(|| model.symmetrize_centres(&group, &basis, 0.05))
            .unwrap();
        let mut centres_proven = false;
        for _ in 0..3 {
            basis.reset();
            let centres_parallel = parallel
                .install(|| {
                    assert_eq!(rayon::current_num_threads(), 4);
                    model.symmetrize_centres(&group, &basis, 0.05)
                })
                .unwrap();
            centres_proven |= basis.worker_count() >= 2;
            assert_eq!(
                real_bits(&centres_serial.orb),
                real_bits(&centres_parallel.orb)
            );
        }
        // A serial fallback can only ever record one worker per call, so a
        // single call with two workers proves the parallel path executed.
        assert!(
            centres_proven,
            "symmetrize_centres must resolve actions on more than one worker in one call"
        );

        let joint_serial = serial
            .install(|| model.symmetrize_model_and_centres(&group, &basis, &parameters))
            .unwrap();
        let mut joint_proven = false;
        for _ in 0..3 {
            basis.reset();
            let joint_parallel = parallel
                .install(|| {
                    assert_eq!(rayon::current_num_threads(), 4);
                    model.symmetrize_model_and_centres(&group, &basis, &parameters)
                })
                .unwrap();
            joint_proven |= basis.worker_count() >= 2;
            assert_eq!(real_bits(&joint_serial.orb), real_bits(&joint_parallel.orb));
            assert_eq!(joint_serial.hamR, joint_parallel.hamR);
            assert_eq!(
                complex_bits(&joint_serial.ham),
                complex_bits(&joint_parallel.ham)
            );
        }
        assert!(
            joint_proven,
            "symmetrize_model_and_centres must resolve actions on more than one worker in one call"
        );

        let pair_serial = serial
            .install(|| {
                Model::<false, 3>::symmetrize_pair(&model, &down, &group, &basis, &parameters)
            })
            .unwrap();
        let mut pair_proven = false;
        for _ in 0..3 {
            basis.reset();
            let pair_parallel = parallel
                .install(|| {
                    assert_eq!(rayon::current_num_threads(), 4);
                    Model::<false, 3>::symmetrize_pair(&model, &down, &group, &basis, &parameters)
                })
                .unwrap();
            pair_proven |= basis.worker_count() >= 2;
            assert_eq!(
                real_bits(&pair_serial.0.orb),
                real_bits(&pair_parallel.0.orb)
            );
            assert_eq!(
                real_bits(&pair_serial.1.orb),
                real_bits(&pair_parallel.1.orb)
            );
            assert_eq!(pair_serial.0.hamR, pair_parallel.0.hamR);
            assert_eq!(pair_serial.1.hamR, pair_parallel.1.hamR);
            assert_eq!(
                complex_bits(&pair_serial.0.ham),
                complex_bits(&pair_parallel.0.ham)
            );
            assert_eq!(
                complex_bits(&pair_serial.1.ham),
                complex_bits(&pair_parallel.1.ham)
            );
        }
        assert!(
            pair_proven,
            "symmetrize_pair must resolve actions on more than one worker in one call"
        );

        // Position-matrix projection tail: HasRMatrix model with symmetric centres.
        let positioned = displaced_s_model_with_positions(0.0);
        let positioned_group = positioned
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let positioned_serial = serial
            .install(|| {
                positioned.symmetrize_hamiltonian(&positioned_group, &ScalarSiteBasis, &parameters)
            })
            .unwrap();
        for _ in 0..3 {
            let positioned_parallel = parallel
                .install(|| {
                    assert_eq!(rayon::current_num_threads(), 4);
                    positioned.symmetrize_hamiltonian(
                        &positioned_group,
                        &ScalarSiteBasis,
                        &parameters,
                    )
                })
                .unwrap();
            assert_eq!(positioned_serial.hamR, positioned_parallel.hamR);
            assert_eq!(
                real_bits(&positioned_serial.orb),
                real_bits(&positioned_parallel.orb)
            );
            assert_eq!(
                complex_bits(&positioned_serial.ham),
                complex_bits(&positioned_parallel.ham)
            );
            assert_eq!(
                complex4_bits(positioned_serial.rmatrix.as_array4()),
                complex4_bits(positioned_parallel.rmatrix.as_array4())
            );
        }
    }

    #[test]
    fn centre_symmetrization_rejects_incompatible_target_operations() {
        let model = displaced_p_model();
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
    ) -> f64 {
        let (e_up, _) = up.solve_all_parallel(kpoints).unwrap();
        let (e_dn, _) = dn.solve_all_parallel(kpoints).unwrap();
        let mut max = 0.0f64;
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
        max
    }

    #[test]
    #[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
    fn mgfe6ge6_pair_acceptance() {
        let (before_up, before_dn, group) = real_pair_models();
        let params = HamiltonianSymmetrizationParameters::default();
        let (up, dn) = Model::<false, 3>::symmetrize_pair(
            &before_up,
            &before_dn,
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
        let spectrum_error = spectrum_pair_error(&up, &dn, &group, &kpoints);
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
        let (up_again, dn_again) = Model::<false, 3>::symmetrize_pair(
            &up,
            &dn,
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
                    .max((up.orb[[orbital, axis]] - up_again.orb[[orbital, axis]]).abs())
                    .max((dn.orb[[orbital, axis]] - dn_again.orb[[orbital, axis]]).abs());
            }
        }
        assert_eq!(up.hamR, up_again.hamR);
        assert_eq!(dn.hamR, dn_again.hamR);
        assert_eq!(up.ham.dim(), up_again.ham.dim());
        assert_eq!(dn.ham.dim(), dn_again.ham.dim());
        for (left, right) in up.ham.iter().zip(up_again.ham.iter()) {
            ham_change = ham_change.max((left - right).norm());
        }
        for (left, right) in dn.ham.iter().zip(dn_again.ham.iter()) {
            ham_change = ham_change.max((left - right).norm());
        }
        assert_eq!(up_again.lat, before_up.lat);
        assert_eq!(dn_again.lat, before_dn.lat);
        assert_eq!(up_again.atoms.len(), before_up.atoms.len());
        println!("REAL centre covariance {centre_error:.6e} A");
        println!("REAL spectrum error {spectrum_error:.6e} eV, Gamma {gamma:.6e} eV");
        println!("REAL band change {band_change:.6e} eV");
        println!("REAL idempotent orb {orb_change:.3e} ham {ham_change:.3e}");
        assert!(centre_error < 1e-8);
        assert!(spectrum_error < 1e-5);
        assert!(gamma < 1e-10);
        assert!(orb_change < 1e-9);
        assert!(ham_change < 1e-8);
    }

    #[test]
    fn pair_symmetrization_accepts_two_identical_zero_models() {
        let up = displaced_p_model();
        let down = up.clone();
        let original_orb = up.orb.clone();
        let group = up
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let (projected_up, projected_down) = Model::<false, 3>::symmetrize_pair(
            &up,
            &down,
            &group,
            &CenterResolvedBasis::new(),
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap();
        assert_eq!(up.orb, original_orb);
        assert_eq!(down.orb, original_orb);
        for projected in [&projected_up, &projected_down] {
            assert!(projected.ham.iter().all(|hop| hop.norm() < 1e-12));
            assert!(projected.orb.iter().all(|value| value.abs() < 1e-8));
        }
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
        let model = model
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
