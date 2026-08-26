//! Hamiltonian symmetry certification and magnetic-subgroup identification.
//!
//! Structural symmetry and Hamiltonian symmetry are different statements.  A
//! structure supplies candidate Seitz operations
//!
//! $$
//! g = \{W\mid\mathbf w\}\mathcal T^\theta,
//! \qquad \theta\in\{0,1\},
//! $$
//!
//! while the tight-binding basis must additionally supply a representation of
//! every candidate operation.  In a localized basis, that representation is
//! stored as a finite set of cell-shift sectors,
//!
//! $$
//! g\lvert b,\mathbf R\rangle
//! = \sum_{\mathbf s,a}(D^g_{\mathbf s})_{ab}
//!   \lvert a,W\mathbf R+\mathbf s\rangle .
//! $$
//!
//! Rustb stores
//!
//! $$
//! H_{ab}(\mathbf R)=\langle a,\mathbf 0\rvert\hat H
//! \lvert b,\mathbf R\rangle .
//! $$
//!
//! Therefore a unitary candidate is preserved exactly when
//!
//! $$
//! H_{ab}(\mathbf R)=
//! \sum_{\mathbf s,\mathbf t}
//! \left[(D^g_{\mathbf s})^\dagger
//! H(W\mathbf R+\mathbf t-\mathbf s)
//! D^g_{\mathbf t}\right]_{ab},
//! $$
//!
//! and an anti-unitary candidate is preserved when the right-hand side equals
//! $H_{ab}(\mathbf R)^*$.  The implementation checks this identity on the
//! complete finite real-space support, including transformed support points
//! absent from `Model::hamR` (which represent zero matrices).  It is therefore
//! stronger than sampling a few $\mathbf k$ points.
//!
//! The surviving operation set is validated as a magnetic group by cryspglib
//! before UNI/BNS identification.  A numerically non-closed set or an
//! unresolved basis action produces an inconclusive report; the code never
//! guesses a subgroup by silently adding or deleting operations.

use crate::atom_struct::OrbProj;
use crate::crystal_symmetry::{
    CrystalSymmetry, CrystalSymmetryDataset, CrystalSymmetryOperation, MagneticCrystalSymmetry,
    MagneticGroupType, SymmetryParameters, convert_magnetic_type, cry_lattice, matrix3,
};
use crate::error::{Result, TbError};
use crate::model::{Model, RMatrixData};
use ndarray::{Array2, Array3, Array4, Axis};
use ndarray_linalg::Inverse;
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

/// One cell-shift sector of a localized-basis symmetry action.
///
/// `matrix` has rows in the transformed basis and columns in the source basis.
/// Its meaning is
///
/// $$
/// g\lvert b,\mathbf R\rangle \supset
/// \sum_a D_{ab}\lvert a,W\mathbf R+\mathbf s\rangle,
/// $$
///
/// where $\mathbf s$ is [`Self::shift`].
#[derive(Debug, Clone)]
pub struct CellShiftAction {
    pub shift: [isize; 3],
    pub matrix: Array2<Complex64>,
}

/// Complete localized-basis representation of one Seitz operation.
#[derive(Debug, Clone)]
pub struct LocalizedBasisAction {
    pub sectors: Vec<CellShiftAction>,
}

impl LocalizedBasisAction {
    /// Construct the lattice-gauge Bloch representation at the image momentum.
    ///
    /// If $\mathbf k_g=W^{-T}\mathbf k$ for a unitary operation and
    /// $\mathbf k_g=-W^{-T}\mathbf k$ for an anti-unitary operation, then
    ///
    /// $$
    /// U_g^L(\mathbf k)=\sum_{\mathbf s}D^g_{\mathbf s}
    /// e^{-2\pi i\mathbf k_g\cdot\mathbf s}.
    /// $$
    ///
    /// This sewing matrix is retained in the Hamiltonian report so that a
    /// later band-irrep implementation can restrict it to a little group and
    /// take traces on degenerate eigenspaces.
    pub fn lattice_gauge_matrix(
        &self,
        image_k: [f64; 3],
    ) -> std::result::Result<Array2<Complex64>, BasisRepresentationError> {
        if image_k.iter().any(|component| !component.is_finite()) {
            return Err(BasisRepresentationError::Invalid(
                "Bloch momentum must have finite components".to_string(),
            ));
        }
        let Some(first) = self.sectors.first() else {
            return Err(BasisRepresentationError::Invalid(
                "localized action has no cell-shift sectors".to_string(),
            ));
        };
        let dimension = first.matrix.nrows();
        if first.matrix.ncols() != dimension {
            return Err(BasisRepresentationError::Invalid(format!(
                "sector {:?} is not square",
                first.shift
            )));
        }
        let mut matrix = Array2::zeros((dimension, dimension));
        for sector in &self.sectors {
            if sector.matrix.dim() != (dimension, dimension) {
                return Err(BasisRepresentationError::Invalid(format!(
                    "sector {:?} has shape {:?}, expected ({dimension}, {dimension})",
                    sector.shift,
                    sector.matrix.dim()
                )));
            }
            let phase_argument = image_k
                .iter()
                .zip(sector.shift)
                .map(|(k, shift)| k * shift as f64)
                .sum::<f64>();
            if !phase_argument.is_finite() {
                return Err(BasisRepresentationError::Invalid(
                    "Bloch phase argument is not finite".to_string(),
                ));
            }
            let phase = Complex64::new(0.0, -std::f64::consts::TAU * phase_argument).exp();
            matrix.scaled_add(phase, &sector.matrix);
        }
        Ok(matrix)
    }
}

/// Context supplied to a localized-basis representation provider.
pub struct BasisActionContext<'a, const SPIN: bool, R: RMatrixData> {
    pub model: &'a Model<SPIN, 3, R>,
    pub operation: &'a CrystalSymmetryOperation,
    pub position_tolerance: f64,
    pub representation_tolerance: f64,
}

/// Why an operation could not be represented in the current localized basis.
///
/// `Unsupported` and `Ambiguous` mean “not enough basis metadata”; they are
/// deliberately distinct from a Hamiltonian symmetry violation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BasisRepresentationError {
    #[error("basis representation is unsupported: {0}")]
    Unsupported(String),
    #[error("basis representation is ambiguous: {0}")]
    Ambiguous(String),
    #[error("basis representation is invalid: {0}")]
    Invalid(String),
}

/// Provider for the action of crystallographic operations on the TB basis.
///
/// Implement this trait for Wannier gauges, local frames, orbital shells, or
/// other bases whose transformation law cannot be inferred from [`OrbProj`].
/// Returning an error marks only that operation as unresolved; it does not
/// falsely label the operation as broken.
pub trait BasisSymmetryRepresentation<const SPIN: bool, R: RMatrixData> {
    fn resolve(
        &self,
        context: BasisActionContext<'_, SPIN, R>,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError>;
}

impl<const SPIN: bool, R: RMatrixData, F> BasisSymmetryRepresentation<SPIN, R> for F
where
    F: for<'a> Fn(
        BasisActionContext<'a, SPIN, R>,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError>,
{
    fn resolve(
        &self,
        context: BasisActionContext<'_, SPIN, R>,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
        self(context)
    }
}

/// Strict convenience representation for one scalar orbital on every atom.
///
/// This provider accepts a model only when every atom owns exactly one
/// [`OrbProj::s`] orbital, every orbital has an owner, and each orbital center
/// coincides with its atom modulo a lattice vector.  These restrictions make
/// the automatic representation unambiguous.  More general Wannier models
/// should implement [`BasisSymmetryRepresentation`] explicitly.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScalarSiteBasis;

/// Automatic representation for atom-centred Wannier90 orbital projections.
///
/// The provider reads each atom's owned orbitals and the corresponding
/// [`OrbProj`] labels, rotates their global Cartesian angular functions, and
/// tensors that orbital action with the existing spin-1/2 action when
/// `SPIN=true`. It supports the pure `s`, `p`, `d`, and `f` projections and the
/// Wannier90 `sp`, `sp2`, `sp3`, `sp3d`, and `sp3d2` hybrids.
///
/// This inference is intentionally strict. Every orbital must be atom-centred
/// modulo a lattice vector, each atom-local projection list must be
/// orthonormal and closed under the requested operation, and same-species
/// sites must map uniquely. Repeated angular projections (for example two
/// radial `p_x` functions), incomplete shells, local orbital frames, and
/// general Wannier gauges require a custom [`BasisSymmetryRepresentation`]
/// instead of being guessed.
#[derive(Debug, Default, Clone, Copy)]
pub struct AtomicOrbitalBasis;

/// Candidate magnetic supergroup tested against the Hamiltonian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HamiltonianSymmetryCandidates {
    /// Test $G+G\mathcal T$.  This is required for a certified magnetic group.
    StructuralGrey,
    /// Test only the unitary structural group $G$.
    ///
    /// This is useful diagnostically but cannot certify that no anti-unitary
    /// operations survive.
    StructuralUnitary,
}

/// Numerical tolerances for exact real-space Hamiltonian checking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HamiltonianSymmetryTolerances {
    /// Absolute matrix-element residual in Hamiltonian energy units.
    pub absolute: f64,
    /// Residual relative to the largest Hamiltonian matrix element.
    pub relative: f64,
    /// Optional Cartesian tolerance for basis-center matching.
    ///
    /// `None` reuses [`SymmetryParameters::symprec`], so structural detection
    /// and localized-basis matching use the same physical length tolerance.
    pub position: Option<f64>,
    /// Tolerance for $H(-\mathbf R)=H(\mathbf R)^\dagger$ validation.
    pub hermiticity: f64,
    /// Tolerance for localized-action unitarity and spin lifts.
    pub representation: f64,
    /// Fractional-translation tolerance for magnetic-operation equality.
    ///
    /// This must lie in $(0,\frac12)$ so distinct lattice cosets remain
    /// separable.
    pub operation: f64,
    /// Tolerance for matching supplied target operations against the
    /// operation set detected from the Model structure.
    ///
    /// Structural detection only guarantees [`SymmetryParameters::symprec`]
    /// accuracy, so a target group assembled from database-rounded values can
    /// legitimately differ from the detected operations by more than
    /// [`Self::operation`].  `None` reuses [`SymmetryParameters::symprec`].
    pub membership: Option<f64>,
}

impl Default for HamiltonianSymmetryTolerances {
    fn default() -> Self {
        Self {
            absolute: 1e-10,
            relative: 1e-8,
            position: None,
            hermiticity: 1e-10,
            representation: 1e-8,
            operation: 1e-8,
            membership: None,
        }
    }
}

/// Request for Hamiltonian magnetic-symmetry analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HamiltonianSymmetryRequest {
    pub structural_parameters: SymmetryParameters,
    pub candidates: HamiltonianSymmetryCandidates,
    pub tolerances: HamiltonianSymmetryTolerances,
}

/// Parameters for projecting a Hamiltonian onto a supplied magnetic group.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct HamiltonianSymmetrizationParameters {
    /// Parameters used to re-detect and verify the current Atom structure.
    pub structural_parameters: SymmetryParameters,
    /// Numerical tolerances shared with Hamiltonian certification.
    pub tolerances: HamiltonianSymmetryTolerances,
}

impl Default for HamiltonianSymmetryRequest {
    fn default() -> Self {
        Self {
            structural_parameters: SymmetryParameters::default(),
            candidates: HamiltonianSymmetryCandidates::StructuralGrey,
            tolerances: HamiltonianSymmetryTolerances::default(),
        }
    }
}

/// Location of the largest covariance violation for one operation.
#[derive(Debug, Clone)]
pub struct HamiltonianResidualWitness {
    pub lattice_vector: [isize; 3],
    pub bra: usize,
    pub ket: usize,
    pub original: Complex64,
    pub transformed: Complex64,
}

/// Norms of the Hamiltonian covariance residual for one operation.
///
/// With $\Delta_g(\mathbf R)=H(\mathbf R)-H_g(\mathbf R)$, the report stores
///
/// $$
/// r_\infty=\max_{\mathbf R,a,b}|\Delta_{g,ab}(\mathbf R)|,
/// \qquad
/// r_F=\frac{\sqrt{\sum_{\mathbf R}\|\Delta_g(\mathbf R)\|_F^2}}
/// {\max(\sqrt{\sum_{\mathbf R}\|H(\mathbf R)\|_F^2},\epsilon)}.
/// $$
#[derive(Debug, Clone)]
pub struct HamiltonianResidual {
    pub max_absolute: f64,
    pub max_relative: f64,
    pub relative_frobenius: f64,
    pub acceptance_threshold: f64,
    pub witness: HamiltonianResidualWitness,
}

/// Result for one candidate operation.
#[derive(Debug, Clone)]
pub enum OperationHamiltonianStatus {
    Preserved(HamiltonianResidual),
    Broken(HamiltonianResidual),
    Unresolved(BasisRepresentationError),
}

/// Candidate operation, its localized representation, and its covariance test.
#[derive(Debug, Clone)]
pub struct OperationHamiltonianCheck {
    pub operation: CrystalSymmetryOperation,
    pub action: Option<LocalizedBasisAction>,
    pub status: OperationHamiltonianStatus,
}

/// Whether the report exhaustively decided every candidate operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HamiltonianSymmetryCompleteness {
    Complete,
    LowerBound,
}

/// Compatibility of the field-allowed structural candidates with $H$.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HamiltonianCompatibility {
    /// Every field-allowed operation preserves the Hamiltonian.
    Compatible,
    /// Every operation was decided, but at least one is broken.
    SymmetryReduced,
    /// At least one localized-basis action could not be established.
    Inconclusive,
}

/// cryspglib identification of the Hamiltonian's surviving magnetic group.
#[derive(Debug, Clone)]
pub struct IdentifiedMagneticSubgroup {
    pub uni_number: usize,
    pub litvin_number: usize,
    /// International number of the surviving group's family space group.
    pub family_spacegroup_number: usize,
    pub bns_number: String,
    pub og_number: String,
    pub magnetic_type: MagneticGroupType,
    /// Hall setting derived from the surviving group's family space group.
    pub family_hall_number: usize,
    /// Hall setting returned by magnetic-group identification.
    pub hall_number: usize,
    /// Hall setting of the atom-derived structural supergroup, retained only
    /// as provenance and never used as the effective family Hall hint.
    pub structural_supergroup_hall: usize,
    /// $\mathbf x_{\rm std}=T\mathbf x+\mathbf s$.
    pub transformation_matrix: Array2<f64>,
    pub origin_shift: [f64; 3],
    pub standard_rotation_matrix: Array2<f64>,
    pub subgroup_index_in_candidates: usize,
}

/// Final group-label status.
#[derive(Debug, Clone)]
pub enum FinalMagneticGroup {
    Identified(Box<IdentifiedMagneticSubgroup>),
    Inconclusive { reason: String },
}

/// Full distinction between structure candidates and actual Hamiltonian symmetry.
#[derive(Debug, Clone)]
pub struct HamiltonianSymmetryReport {
    pub structure: CrystalSymmetryDataset,
    /// Structural grey/unitary candidates before external-field filtering.
    pub structure_candidates: Vec<CrystalSymmetryOperation>,
    /// Candidate operations surviving the explicitly supplied $\mathbf E$ and
    /// $\mathbf B$ context.
    pub field_allowed_operations: Vec<CrystalSymmetryOperation>,
    pub operation_checks: Vec<OperationHamiltonianCheck>,
    pub surviving_operations: Vec<CrystalSymmetryOperation>,
    pub compatibility: HamiltonianCompatibility,
    pub completeness: HamiltonianSymmetryCompleteness,
    pub final_group: FinalMagneticGroup,
}

impl HamiltonianSymmetryReport {
    /// Return a safe three-valued answer to whether $H$ preserves every
    /// field-allowed structural candidate.
    ///
    /// `Some(true)` means full compatibility, `Some(false)` means a certified
    /// reduction, and `None` means that incomplete basis metadata prevented a
    /// decision. This intentionally avoids folding “unknown” into `false`.
    pub fn is_fully_compatible(&self) -> Option<bool> {
        match self.compatibility {
            HamiltonianCompatibility::Compatible => Some(true),
            HamiltonianCompatibility::SymmetryReduced => Some(false),
            HamiltonianCompatibility::Inconclusive => None,
        }
    }
}

impl<const SPIN: bool, R: RMatrixData> BasisSymmetryRepresentation<SPIN, R> for ScalarSiteBasis {
    fn resolve(
        &self,
        context: BasisActionContext<'_, SPIN, R>,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
        let model = context.model;
        if !context.position_tolerance.is_finite() || context.position_tolerance <= 0.0 {
            return Err(BasisRepresentationError::Invalid(
                "position tolerance must be finite and positive".to_string(),
            ));
        }
        if !context.representation_tolerance.is_finite() || context.representation_tolerance <= 0.0
        {
            return Err(BasisRepresentationError::Invalid(
                "representation tolerance must be finite and positive".to_string(),
            ));
        }
        model
            .validate()
            .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
        if model.atoms.is_empty() {
            return Err(BasisRepresentationError::Unsupported(
                "ScalarSiteBasis requires explicit atoms".to_string(),
            ));
        }

        let owners = model
            .orbital_owners()
            .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
        if let Some(unowned) = owners.iter().position(Option::is_none) {
            return Err(BasisRepresentationError::Unsupported(format!(
                "orbital {unowned} is not owned by an atom"
            )));
        }
        for (atom_index, atom) in model.atoms.iter().enumerate() {
            if atom.norb() != 1 {
                return Err(BasisRepresentationError::Unsupported(format!(
                    "atom {atom_index} owns {} orbitals; ScalarSiteBasis requires exactly one",
                    atom.norb()
                )));
            }
            let orbital = atom.orbitals()[0].index();
            if model.orb_projection[orbital] != OrbProj::s {
                return Err(BasisRepresentationError::Unsupported(format!(
                    "orbital {orbital} is {}, not an s orbital",
                    model.orb_projection[orbital]
                )));
            }
            let displacement = [
                model.orb[[orbital, 0]] - atom.position_ref()[0],
                model.orb[[orbital, 1]] - atom.position_ref()[1],
                model.orb[[orbital, 2]] - atom.position_ref()[2],
            ];
            if integer_shift_if_close(displacement, &model.lat, context.position_tolerance)
                .is_none()
            {
                return Err(BasisRepresentationError::Unsupported(format!(
                    "orbital {orbital} is not centered on atom {atom_index} modulo a lattice vector"
                )));
            }
        }

        let spin_action =
            scalar_spin_action(model, context.operation, context.representation_tolerance)?;
        let mut mapped_sites: BTreeMap<[isize; 3], Vec<(usize, usize)>> = BTreeMap::new();
        let mut used_target_atoms = BTreeSet::new();
        for (source_atom, atom) in model.atoms.iter().enumerate() {
            let source_position = atom.position_ref();
            let transformed: [f64; 3] = std::array::from_fn(|row| {
                context.operation.translation[row]
                    + (0..3)
                        .map(|column| {
                            f64::from(context.operation.rotation[row][column])
                                * source_position[column]
                        })
                        .sum::<f64>()
            });
            let mut matches = model
                .atoms
                .iter()
                .enumerate()
                .filter_map(|(target_atom, target)| {
                    if target.atom_type() != atom.atom_type() {
                        return None;
                    }
                    let displacement =
                        std::array::from_fn(|axis| transformed[axis] - target.position_ref()[axis]);
                    integer_shift_if_close(displacement, &model.lat, context.position_tolerance)
                        .map(|shift| (target_atom, shift))
                });
            let Some((target_atom, _atom_shift)) = matches.next() else {
                return Err(BasisRepresentationError::Invalid(format!(
                    "operation does not map atom {source_atom} to an atom of the same type"
                )));
            };
            if matches.next().is_some() {
                return Err(BasisRepresentationError::Ambiguous(format!(
                    "operation maps atom {source_atom} to more than one atom within tolerance"
                )));
            }
            if !used_target_atoms.insert(target_atom) {
                return Err(BasisRepresentationError::Ambiguous(format!(
                    "operation maps more than one source atom onto target atom {target_atom}"
                )));
            }
            let source_orbital = atom.orbitals()[0].index();
            let target_orbital = model.atoms[target_atom].orbitals()[0].index();
            let transformed_orbital: [f64; 3] = std::array::from_fn(|row| {
                context.operation.translation[row]
                    + (0..3)
                        .map(|column| {
                            f64::from(context.operation.rotation[row][column])
                                * model.orb[[source_orbital, column]]
                        })
                        .sum::<f64>()
            });
            let orbital_displacement = std::array::from_fn(|axis| {
                transformed_orbital[axis] - model.orb[[target_orbital, axis]]
            });
            let Some(orbital_shift) = integer_shift_if_close(
                orbital_displacement,
                &model.lat,
                context.position_tolerance,
            ) else {
                return Err(BasisRepresentationError::Invalid(format!(
                    "operation maps atom {source_atom} to atom {target_atom}, but their scalar orbital representatives are inconsistent"
                )));
            };
            mapped_sites
                .entry(orbital_shift)
                .or_default()
                .push((target_orbital, source_orbital));
        }

        let mut sectors = Vec::with_capacity(mapped_sites.len());
        for (shift, mappings) in mapped_sites {
            let mut matrix = Array2::zeros((model.nsta(), model.nsta()));
            for (target_orbital, source_orbital) in mappings {
                if SPIN {
                    for target_spin in 0..2 {
                        for source_spin in 0..2 {
                            matrix[[
                                target_spin * model.norb() + target_orbital,
                                source_spin * model.norb() + source_orbital,
                            ]] = spin_action[target_spin][source_spin];
                        }
                    }
                } else {
                    matrix[[target_orbital, source_orbital]] = Complex64::new(1.0, 0.0);
                }
            }
            sectors.push(CellShiftAction { shift, matrix });
        }
        Ok(LocalizedBasisAction { sectors })
    }
}

impl<const SPIN: bool, R: RMatrixData> BasisSymmetryRepresentation<SPIN, R> for AtomicOrbitalBasis {
    fn resolve(
        &self,
        context: BasisActionContext<'_, SPIN, R>,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
        let model = context.model;
        validate_automatic_basis_context(&context, "AtomicOrbitalBasis")?;

        let angular_rotation = real_orbital_rotation(
            cartesian_rotation(model, context.operation)?,
            context.representation_tolerance,
        )?;
        let spin_action =
            scalar_spin_action(model, context.operation, context.representation_tolerance)?;
        let site_projections = model
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_index, atom)| {
                atom_projection_matrix(model, atom_index, atom.orbitals(), &context)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut used_target_atoms = BTreeSet::new();
        let mut mapped_entries: BTreeMap<[isize; 3], Vec<(usize, usize, Complex64)>> =
            BTreeMap::new();
        for (source_atom, atom) in model.atoms.iter().enumerate() {
            let target_atom =
                uniquely_mapped_atom(model, context.operation, source_atom, &context)?;
            if !used_target_atoms.insert(target_atom) {
                return Err(BasisRepresentationError::Ambiguous(format!(
                    "operation maps more than one source atom onto target atom {target_atom}"
                )));
            }

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
            let local_action = target_dagger.dot(&rotated_source);
            let reconstructed = target_coefficients.dot(&local_action);
            let closure_residual = matrix_max_difference(&rotated_source, &reconstructed);
            if closure_residual > context.representation_tolerance {
                return Err(BasisRepresentationError::Unsupported(format!(
                    "atom {source_atom} projections are not closed on target atom {target_atom} under the operation (maximum residual {closure_residual:e}); include the complete shell or provide a custom representation"
                )));
            }
            let local_gram = local_action
                .t()
                .mapv(|value| value.conj())
                .dot(&local_action);
            let local_unitarity = identity_residual(&local_gram);
            if local_unitarity > context.representation_tolerance {
                return Err(BasisRepresentationError::Invalid(format!(
                    "atom-local orbital action from atom {source_atom} to {target_atom} is not unitary (maximum residual {local_unitarity:e})"
                )));
            }

            for (source_local, source_id) in atom.orbitals().iter().enumerate() {
                let source_orbital = source_id.index();
                let transformed_source: [f64; 3] = std::array::from_fn(|axis| {
                    context.operation.translation[axis]
                        + (0..3)
                            .map(|input| {
                                f64::from(context.operation.rotation[axis][input])
                                    * model.orb[[source_orbital, input]]
                            })
                            .sum::<f64>()
                });
                for (target_local, target_id) in
                    model.atoms[target_atom].orbitals().iter().enumerate()
                {
                    let coefficient = local_action[[target_local, source_local]];
                    if coefficient.norm() <= context.representation_tolerance {
                        continue;
                    }
                    let target_orbital = target_id.index();
                    let displacement = std::array::from_fn(|axis| {
                        transformed_source[axis] - model.orb[[target_orbital, axis]]
                    });
                    let Some(shift) = integer_shift_if_close(
                        displacement,
                        &model.lat,
                        context.position_tolerance,
                    ) else {
                        return Err(BasisRepresentationError::Invalid(format!(
                            "operation mixes source orbital {source_orbital} into target orbital {target_orbital}, but their centres differ by a non-lattice vector"
                        )));
                    };
                    mapped_entries.entry(shift).or_default().push((
                        target_orbital,
                        source_orbital,
                        coefficient,
                    ));
                }
            }
        }

        if mapped_entries.is_empty() {
            return Err(BasisRepresentationError::Invalid(
                "automatic orbital action contains no nonzero matrix elements".to_string(),
            ));
        }
        let mut sectors = Vec::with_capacity(mapped_entries.len());
        for (shift, entries) in mapped_entries {
            let mut matrix = Array2::zeros((model.nsta(), model.nsta()));
            for (target_orbital, source_orbital, orbital_coefficient) in entries {
                if SPIN {
                    for target_spin in 0..2 {
                        for source_spin in 0..2 {
                            matrix[[
                                target_spin * model.norb() + target_orbital,
                                source_spin * model.norb() + source_orbital,
                            ]] += orbital_coefficient * spin_action[target_spin][source_spin];
                        }
                    }
                } else {
                    matrix[[target_orbital, source_orbital]] += orbital_coefficient;
                }
            }
            sectors.push(CellShiftAction { shift, matrix });
        }
        Ok(LocalizedBasisAction { sectors })
    }
}

fn validate_automatic_basis_context<const SPIN: bool, R: RMatrixData>(
    context: &BasisActionContext<'_, SPIN, R>,
    provider: &str,
) -> std::result::Result<(), BasisRepresentationError> {
    if !context.position_tolerance.is_finite() || context.position_tolerance <= 0.0 {
        return Err(BasisRepresentationError::Invalid(
            "position tolerance must be finite and positive".to_string(),
        ));
    }
    if !context.representation_tolerance.is_finite() || context.representation_tolerance <= 0.0 {
        return Err(BasisRepresentationError::Invalid(
            "representation tolerance must be finite and positive".to_string(),
        ));
    }
    context
        .model
        .validate()
        .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
    if context.model.atoms.is_empty() {
        return Err(BasisRepresentationError::Unsupported(format!(
            "{provider} requires explicit atoms"
        )));
    }
    let owners = context
        .model
        .orbital_owners()
        .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
    if let Some(unowned) = owners.iter().position(Option::is_none) {
        return Err(BasisRepresentationError::Unsupported(format!(
            "orbital {unowned} is not owned by an atom"
        )));
    }
    Ok(())
}

fn atom_projection_matrix<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    atom_index: usize,
    orbitals: &[crate::OrbitalId],
    context: &BasisActionContext<'_, SPIN, R>,
) -> std::result::Result<Array2<Complex64>, BasisRepresentationError> {
    let mut coefficients = Array2::zeros((16, orbitals.len()));
    for (column, orbital_id) in orbitals.iter().enumerate() {
        let orbital = orbital_id.index();
        let displacement = std::array::from_fn(|axis| {
            model.orb[[orbital, axis]] - model.atoms[atom_index].position_ref()[axis]
        });
        if integer_shift_if_close(displacement, &model.lat, context.position_tolerance).is_none() {
            return Err(BasisRepresentationError::Unsupported(format!(
                "orbital {orbital} is not centred on atom {atom_index} modulo a lattice vector"
            )));
        }
        let state = projection_in_real_orbital_basis(model.orb_projection[orbital])?;
        for row in 0..16 {
            coefficients[[row, column]] = state[row];
        }
    }

    let gram = coefficients
        .t()
        .mapv(|value| value.conj())
        .dot(&coefficients);
    let residual = identity_residual(&gram);
    if residual > context.representation_tolerance {
        return Err(BasisRepresentationError::Ambiguous(format!(
            "atom {atom_index} orbital projections are not an orthonormal labelled basis (maximum Gram residual {residual:e}); repeated radial shells need an explicit representation"
        )));
    }
    Ok(coefficients)
}

fn uniquely_mapped_atom<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    operation: &CrystalSymmetryOperation,
    source_atom: usize,
    context: &BasisActionContext<'_, SPIN, R>,
) -> std::result::Result<usize, BasisRepresentationError> {
    let source = &model.atoms[source_atom];
    let transformed: [f64; 3] = std::array::from_fn(|row| {
        operation.translation[row]
            + (0..3)
                .map(|column| {
                    f64::from(operation.rotation[row][column]) * source.position_ref()[column]
                })
                .sum::<f64>()
    });
    let mut matches = model
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(target_atom, target)| {
            if target.atom_type() != source.atom_type() {
                return None;
            }
            let displacement =
                std::array::from_fn(|axis| transformed[axis] - target.position_ref()[axis]);
            integer_shift_if_close(displacement, &model.lat, context.position_tolerance)
                .map(|_| target_atom)
        });
    let Some(target_atom) = matches.next() else {
        return Err(BasisRepresentationError::Invalid(format!(
            "operation does not map atom {source_atom} to an atom of the same type"
        )));
    };
    if matches.next().is_some() {
        return Err(BasisRepresentationError::Ambiguous(format!(
            "operation maps atom {source_atom} to more than one atom within tolerance"
        )));
    }
    Ok(target_atom)
}

fn identity_residual(matrix: &Array2<Complex64>) -> f64 {
    let (rows, columns) = matrix.dim();
    if rows != columns {
        return f64::INFINITY;
    }
    let mut residual = 0.0_f64;
    for row in 0..rows {
        for column in 0..columns {
            let expected = if row == column {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            residual = residual.max((matrix[[row, column]] - expected).norm());
        }
    }
    residual
}

fn matrix_max_difference(left: &Array2<Complex64>, right: &Array2<Complex64>) -> f64 {
    if left.dim() != right.dim() {
        return f64::INFINITY;
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0_f64, f64::max)
}

const PURE_REAL_ORBITALS: [OrbProj; 16] = [
    OrbProj::s,
    OrbProj::px,
    OrbProj::py,
    OrbProj::pz,
    OrbProj::dxy,
    OrbProj::dyz,
    OrbProj::dxz,
    OrbProj::dz2,
    OrbProj::dx2y2,
    OrbProj::fz3,
    OrbProj::fxz2,
    OrbProj::fyz2,
    OrbProj::fzx2y2,
    OrbProj::fxyz,
    OrbProj::fxx23y2,
    OrbProj::fy3x2y2,
];

fn projection_in_real_orbital_basis(
    projection: OrbProj,
) -> std::result::Result<[Complex64; 16], BasisRepresentationError> {
    let state = projection
        .to_quantum_number()
        .map_err(|error| BasisRepresentationError::Unsupported(error.to_string()))?;
    let mut coefficients = [Complex64::new(0.0, 0.0); 16];
    for (row, pure_projection) in PURE_REAL_ORBITALS.iter().enumerate() {
        let pure = pure_projection
            .to_quantum_number()
            .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
        coefficients[row] = pure
            .iter()
            .zip(&state)
            .map(|(pure, state)| pure.conj() * state)
            .sum();
    }
    Ok(coefficients)
}

fn real_orbital_rotation(
    cartesian: [[f64; 3]; 3],
    tolerance: f64,
) -> std::result::Result<Array2<Complex64>, BasisRepresentationError> {
    if cartesian.iter().flatten().any(|value| !value.is_finite()) {
        return Err(BasisRepresentationError::Invalid(
            "Cartesian orbital rotation contains non-finite entries".to_string(),
        ));
    }
    let rotation = Array2::from_shape_fn((3, 3), |(row, column)| cartesian[row][column]);
    let orthogonality = rotation.t().dot(&rotation);
    let orthogonality_residual = (0..3)
        .flat_map(|row| (0..3).map(move |column| (row, column)))
        .map(|(row, column)| {
            let expected = if row == column { 1.0 } else { 0.0 };
            (orthogonality[[row, column]] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    if orthogonality_residual > tolerance {
        return Err(BasisRepresentationError::Invalid(format!(
            "Cartesian orbital rotation is not orthogonal (maximum residual {orthogonality_residual:e})"
        )));
    }

    let sample_count = 32;
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let sample_points = (0..sample_count)
        .map(|sample| {
            let z = 1.0 - 2.0 * (sample as f64 + 0.5) / sample_count as f64;
            let radius = (1.0 - z * z).sqrt();
            let phi = golden_angle * sample as f64;
            [radius * phi.cos(), radius * phi.sin(), z]
        })
        .collect::<Vec<_>>();
    let inverse_points = sample_points
        .iter()
        .map(|point| {
            std::array::from_fn(|row| {
                (0..3)
                    .map(|column| cartesian[column][row] * point[column])
                    .sum::<f64>()
            })
        })
        .collect::<Vec<[f64; 3]>>();

    let mut result = Array2::zeros((16, 16));
    for angular_momentum in 0..=3_isize {
        let dimension = (2 * angular_momentum + 1) as usize;
        let basis = Array2::from_shape_fn((sample_count, dimension), |(sample, column)| {
            let orbital = (angular_momentum * angular_momentum) as usize + column;
            Complex64::new(real_orbital_value(orbital, sample_points[sample]), 0.0)
        });
        let rotated = Array2::from_shape_fn((sample_count, dimension), |(sample, column)| {
            let orbital = (angular_momentum * angular_momentum) as usize + column;
            Complex64::new(real_orbital_value(orbital, inverse_points[sample]), 0.0)
        });
        let dagger = basis.t().mapv(|value| value.conj());
        let gram_inverse = dagger
            .dot(&basis)
            .inv()
            .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
        let block = gram_inverse.dot(&dagger).dot(&rotated);
        let fit_residual = matrix_max_difference(&basis.dot(&block), &rotated);
        if fit_residual > tolerance.max(1e-12) * 8.0 {
            return Err(BasisRepresentationError::Invalid(format!(
                "failed to construct l={angular_momentum} orbital rotation (fit residual {fit_residual:e})"
            )));
        }
        let start = (angular_momentum * angular_momentum) as usize;
        for row in 0..dimension {
            for column in 0..dimension {
                result[[start + row, start + column]] = block[[row, column]];
            }
        }
    }
    let unitarity = identity_residual(&result.t().mapv(|value| value.conj()).dot(&result));
    if unitarity > tolerance.max(1e-12) * 8.0 {
        return Err(BasisRepresentationError::Invalid(format!(
            "angular orbital rotation is not unitary (maximum residual {unitarity:e})"
        )));
    }
    Ok(result)
}

fn real_orbital_value(orbital: usize, point: [f64; 3]) -> f64 {
    debug_assert!(orbital < 16);
    let norm = point.iter().map(|value| value * value).sum::<f64>().sqrt();
    let [x, y, z] = point.map(|component| component / norm);
    let pi = std::f64::consts::PI;
    match orbital {
        0 => 1.0 / (4.0 * pi).sqrt(),
        1 => (3.0 / (4.0 * pi)).sqrt() * x,
        2 => (3.0 / (4.0 * pi)).sqrt() * y,
        3 => (3.0 / (4.0 * pi)).sqrt() * z,
        4 => (15.0 / (4.0 * pi)).sqrt() * x * y,
        5 => (15.0 / (4.0 * pi)).sqrt() * y * z,
        6 => (15.0 / (4.0 * pi)).sqrt() * x * z,
        7 => (5.0 / (16.0 * pi)).sqrt() * (3.0 * z * z - 1.0),
        8 => (15.0 / (16.0 * pi)).sqrt() * (x * x - y * y),
        9 => 7.0_f64.sqrt() / (4.0 * pi.sqrt()) * (5.0 * z * z * z - 3.0 * z),
        10 => 21.0_f64.sqrt() / (4.0 * (2.0 * pi).sqrt()) * (5.0 * z * z - 1.0) * x,
        11 => 21.0_f64.sqrt() / (4.0 * (2.0 * pi).sqrt()) * (5.0 * z * z - 1.0) * y,
        12 => 105.0_f64.sqrt() / (4.0 * pi.sqrt()) * z * (x * x - y * y),
        13 => 105.0_f64.sqrt() / (2.0 * pi.sqrt()) * x * y * z,
        14 => 35.0_f64.sqrt() / (4.0 * (2.0 * pi).sqrt()) * x * (x * x - 3.0 * y * y),
        15 => 35.0_f64.sqrt() / (4.0 * (2.0 * pi).sqrt()) * y * (3.0 * x * x - y * y),
        _ => unreachable!("real orbital index is bounded by the 16-state basis"),
    }
}

fn integer_shift_if_close(
    displacement: [f64; 3],
    lattice: &Array2<f64>,
    tolerance: f64,
) -> Option<[isize; 3]> {
    if displacement.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let rounded = displacement.map(f64::round);
    if rounded
        .iter()
        .any(|&value| value < isize::MIN as f64 || value > isize::MAX as f64)
    {
        return None;
    }

    // Search neighboring integer representatives and measure the residual in
    // Cartesian space. This keeps the tolerance in the same units as
    // cryspglib's `symprec` and remains correct for skew row-vector lattices.
    let base = rounded.map(|value| value as isize);
    let mut best: Option<([isize; 3], f64)> = None;
    for offset_x in -1_isize..=1 {
        for offset_y in -1_isize..=1 {
            for offset_z in -1_isize..=1 {
                let shift = [
                    base[0].checked_add(offset_x)?,
                    base[1].checked_add(offset_y)?,
                    base[2].checked_add(offset_z)?,
                ];
                let residual = [
                    displacement[0] - shift[0] as f64,
                    displacement[1] - shift[1] as f64,
                    displacement[2] - shift[2] as f64,
                ];
                let cartesian_norm = (0..3)
                    .map(|cartesian| {
                        (0..3)
                            .map(|vector| residual[vector] * lattice[[vector, cartesian]])
                            .sum::<f64>()
                    })
                    .map(|component| component * component)
                    .sum::<f64>()
                    .sqrt();
                if best.is_none_or(|(_, best_norm)| cartesian_norm < best_norm) {
                    best = Some((shift, cartesian_norm));
                }
            }
        }
    }
    best.and_then(|(shift, norm)| (norm <= tolerance).then_some(shift))
}

fn scalar_spin_action<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    operation: &CrystalSymmetryOperation,
    tolerance: f64,
) -> std::result::Result<[[Complex64; 2]; 2], BasisRepresentationError> {
    if !SPIN {
        return Ok([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);
    }
    let cartesian = cartesian_rotation(model, operation)?;
    let [q0, qx, qy, qz] = cryspglib::axial_spin_half_lift(&cartesian, tolerance)
        .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;

    // cryspglib returns U = q0 I - i(qx sigma_x + qy sigma_y + qz sigma_z).
    let spatial = [
        [Complex64::new(q0, -qz), Complex64::new(-qy, -qx)],
        [Complex64::new(qy, -qx), Complex64::new(q0, qz)],
    ];
    if !operation.time_reversal {
        return Ok(spatial);
    }

    // The anti-unitary operator is U(W) (i sigma_y) K. Only its linear
    // coefficient matrix U(W) i sigma_y is stored in a basis action.
    let time_reversal = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    Ok(multiply_2x2(spatial, time_reversal))
}

fn multiply_2x2(left: [[Complex64; 2]; 2], right: [[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (0..2)
                .map(|inner| left[row][inner] * right[inner][column])
                .sum()
        })
    })
}

fn cartesian_rotation<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    operation: &CrystalSymmetryOperation,
) -> std::result::Result<[[f64; 3]; 3], BasisRepresentationError> {
    let lattice =
        Array2::from_shape_fn((3, 3), |(cartesian, vector)| model.lat[[vector, cartesian]]);
    let inverse = lattice
        .clone()
        .inv()
        .map_err(|error| BasisRepresentationError::Invalid(error.to_string()))?;
    let fractional = Array2::from_shape_fn((3, 3), |(row, column)| {
        f64::from(operation.rotation[row][column])
    });
    let cartesian = lattice.dot(&fractional).dot(&inverse);
    Ok(std::array::from_fn(|row| {
        std::array::from_fn(|column| cartesian[[row, column]])
    }))
}

#[derive(Debug)]
struct HamiltonianSupport {
    matrices: BTreeMap<[isize; 3], Array2<Complex64>>,
    max_element: f64,
    frobenius_norm: f64,
}

fn validate_hamiltonian<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    tolerance: f64,
) -> Result<HamiltonianSupport> {
    model.validate()?;
    if model.nsta() == 0 {
        return Err(TbError::InvalidHamiltonianSymmetryInput {
            parameter: "basis",
            message: "the Hamiltonian basis must not be empty".to_string(),
        });
    }
    if model.atoms.is_empty() {
        return Err(TbError::MissingAtomicStructure);
    }
    if let Some(orbital) = model.orbital_owners()?.iter().position(Option::is_none) {
        return Err(TbError::InvalidHamiltonianSymmetryInput {
            parameter: "orbital_ownership",
            message: format!("orbital {orbital} is not assigned to an atom"),
        });
    }
    if model
        .ham
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(TbError::InvalidHamiltonianSymmetryInput {
            parameter: "hamiltonian",
            message: "all Hamiltonian matrix elements must be finite".to_string(),
        });
    }

    let mut matrices = BTreeMap::new();
    let mut max_element = 0.0_f64;
    let mut norm_squared = 0.0_f64;
    for index in 0..model.hamR.nrows() {
        let lattice_vector = [
            model.hamR[[index, 0]],
            model.hamR[[index, 1]],
            model.hamR[[index, 2]],
        ];
        if matrices.contains_key(&lattice_vector) {
            return Err(TbError::InvalidHamiltonianSymmetryInput {
                parameter: "hamR",
                message: format!("duplicate hopping lattice vector {lattice_vector:?}"),
            });
        }
        let matrix = model.ham.index_axis(Axis(0), index).to_owned();
        for value in &matrix {
            let norm = value.norm();
            max_element = max_element.max(norm);
            norm_squared += norm * norm;
        }
        matrices.insert(lattice_vector, matrix);
    }

    for (&lattice_vector, matrix) in &matrices {
        let Some(negative) = checked_negate(lattice_vector) else {
            return Err(TbError::InvalidHamiltonianSymmetryInput {
                parameter: "hamR",
                message: format!("cannot negate hopping lattice vector {lattice_vector:?}"),
            });
        };
        let Some(partner) = matrices.get(&negative) else {
            return Err(TbError::InvalidHamiltonianSymmetryInput {
                parameter: "hermiticity",
                message: format!(
                    "hopping lattice vector {lattice_vector:?} has no stored partner {negative:?}"
                ),
            });
        };
        let mut max_residual = 0.0_f64;
        for row in 0..model.nsta() {
            for column in 0..model.nsta() {
                max_residual = max_residual
                    .max((matrix[[row, column]] - partner[[column, row]].conj()).norm());
            }
        }
        if max_residual > tolerance {
            return Err(TbError::InvalidHamiltonianSymmetryInput {
                parameter: "hermiticity",
                message: format!(
                    "H({lattice_vector:?}) differs from H({negative:?})^dagger by {max_residual:e}"
                ),
            });
        }
    }

    Ok(HamiltonianSupport {
        matrices,
        max_element,
        frobenius_norm: norm_squared.sqrt(),
    })
}

fn checked_negate(vector: [isize; 3]) -> Option<[isize; 3]> {
    Some([
        vector[0].checked_neg()?,
        vector[1].checked_neg()?,
        vector[2].checked_neg()?,
    ])
}

fn validate_action(
    action: LocalizedBasisAction,
    dimension: usize,
    tolerance: f64,
) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
    if action.sectors.is_empty() {
        return Err(BasisRepresentationError::Invalid(
            "localized action has no cell-shift sectors".to_string(),
        ));
    }
    let mut combined: BTreeMap<[isize; 3], Array2<Complex64>> = BTreeMap::new();
    for sector in action.sectors {
        if sector.matrix.dim() != (dimension, dimension) {
            return Err(BasisRepresentationError::Invalid(format!(
                "sector {:?} has shape {:?}, expected ({dimension}, {dimension})",
                sector.shift,
                sector.matrix.dim()
            )));
        }
        if sector
            .matrix
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(BasisRepresentationError::Invalid(format!(
                "sector {:?} contains non-finite coefficients",
                sector.shift
            )));
        }
        combined
            .entry(sector.shift)
            .and_modify(|matrix| *matrix += &sector.matrix)
            .or_insert(sector.matrix);
    }
    let sectors = combined
        .into_iter()
        .filter(|(_, matrix)| matrix.iter().any(|value| value.norm() > tolerance))
        .map(|(shift, matrix)| CellShiftAction { shift, matrix })
        .collect::<Vec<_>>();
    if sectors.is_empty() {
        return Err(BasisRepresentationError::Invalid(
            "localized action is numerically zero".to_string(),
        ));
    }

    // Exact Laurent unitarity: sum_{t-s=delta} D_s^dagger D_t is I for
    // delta=0 and zero for every other shift.
    let mut correlations: BTreeMap<[isize; 3], Array2<Complex64>> = BTreeMap::new();
    for left in &sectors {
        let dagger = left.matrix.t().mapv(|value| value.conj());
        for right in &sectors {
            let delta = checked_shift_difference(right.shift, left.shift).ok_or_else(|| {
                BasisRepresentationError::Invalid(
                    "cell-shift difference overflows isize".to_string(),
                )
            })?;
            let contribution = dagger.dot(&right.matrix);
            correlations
                .entry(delta)
                .and_modify(|matrix| *matrix += &contribution)
                .or_insert(contribution);
        }
    }
    let mut max_residual = 0.0_f64;
    for (delta, matrix) in correlations {
        for row in 0..dimension {
            for column in 0..dimension {
                let expected = if delta == [0, 0, 0] && row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                max_residual = max_residual.max((matrix[[row, column]] - expected).norm());
            }
        }
    }
    if max_residual > tolerance {
        return Err(BasisRepresentationError::Invalid(format!(
            "localized action is not unitary as a Laurent operator (maximum residual {max_residual:e})"
        )));
    }
    Ok(LocalizedBasisAction { sectors })
}

fn validate_action_geometry<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    operation: &CrystalSymmetryOperation,
    action: &LocalizedBasisAction,
    position_tolerance: f64,
    representation_tolerance: f64,
) -> std::result::Result<(), BasisRepresentationError> {
    for sector in &action.sectors {
        for row in 0..model.nsta() {
            for column in 0..model.nsta() {
                if sector.matrix[[row, column]].norm() <= representation_tolerance {
                    continue;
                }
                let target_orbital = row % model.norb();
                let source_orbital = column % model.norb();
                let transformed_source: [f64; 3] = std::array::from_fn(|axis| {
                    operation.translation[axis]
                        + (0..3)
                            .map(|input| {
                                f64::from(operation.rotation[axis][input])
                                    * model.orb[[source_orbital, input]]
                            })
                            .sum::<f64>()
                });
                let displacement = std::array::from_fn(|axis| {
                    transformed_source[axis] - model.orb[[target_orbital, axis]]
                });
                let expected_shift =
                    integer_shift_if_close(displacement, &model.lat, position_tolerance);
                if expected_shift != Some(sector.shift) {
                    return Err(BasisRepresentationError::Invalid(format!(
                        "sector {:?} maps basis state {column} to {row}, but orbital centers require shift {expected_shift:?}",
                        sector.shift
                    )));
                }
            }
        }
    }
    Ok(())
}

fn checked_shift_difference(left: [isize; 3], right: [isize; 3]) -> Option<[isize; 3]> {
    Some([
        left[0].checked_sub(right[0])?,
        left[1].checked_sub(right[1])?,
        left[2].checked_sub(right[2])?,
    ])
}

fn determinant(rotation: &[[i32; 3]; 3]) -> i128 {
    let value = |row: usize, column: usize| i128::from(rotation[row][column]);
    value(0, 0) * (value(1, 1) * value(2, 2) - value(1, 2) * value(2, 1))
        + value(0, 1) * (value(1, 2) * value(2, 0) - value(1, 0) * value(2, 2))
        + value(0, 2) * (value(1, 0) * value(2, 1) - value(1, 1) * value(2, 0))
}

fn inverse_rotation(rotation: &[[i32; 3]; 3]) -> Option<[[i128; 3]; 3]> {
    let determinant = determinant(rotation);
    if determinant != 1 && determinant != -1 {
        return None;
    }
    let value = |row: usize, column: usize| i128::from(rotation[row][column]);
    let cofactor = |row: usize, column: usize| {
        let rows = (0..3)
            .filter(|&candidate| candidate != row)
            .collect::<Vec<_>>();
        let columns = (0..3)
            .filter(|&candidate| candidate != column)
            .collect::<Vec<_>>();
        let minor = value(rows[0], columns[0]) * value(rows[1], columns[1])
            - value(rows[0], columns[1]) * value(rows[1], columns[0]);
        if (row + column).is_multiple_of(2) {
            minor
        } else {
            -minor
        }
    };
    Some(std::array::from_fn(|row| {
        std::array::from_fn(|column| cofactor(column, row) / determinant)
    }))
}

fn checked_rotation_vector(rotation: &[[i32; 3]; 3], vector: [isize; 3]) -> Option<[isize; 3]> {
    let mut result = [0_isize; 3];
    for row in 0..3 {
        let value = (0..3).try_fold(0_i128, |sum, column| {
            sum.checked_add(i128::from(rotation[row][column]) * vector[column] as i128)
        })?;
        result[row] = isize::try_from(value).ok()?;
    }
    Some(result)
}

fn checked_inverse_rotation_vector(
    inverse: &[[i128; 3]; 3],
    vector: [isize; 3],
) -> Option<[isize; 3]> {
    let mut result = [0_isize; 3];
    for row in 0..3 {
        let value = (0..3).try_fold(0_i128, |sum, column| {
            sum.checked_add(inverse[row][column] * vector[column] as i128)
        })?;
        result[row] = isize::try_from(value).ok()?;
    }
    Some(result)
}

fn transformed_support_domain(
    support: &HamiltonianSupport,
    operation: &CrystalSymmetryOperation,
    action: &LocalizedBasisAction,
) -> std::result::Result<BTreeSet<[isize; 3]>, BasisRepresentationError> {
    let Some(inverse) = inverse_rotation(&operation.rotation) else {
        return Err(BasisRepresentationError::Invalid(
            "operation rotation is not unimodular".to_string(),
        ));
    };
    let mut domain: BTreeSet<[isize; 3]> = support.matrices.keys().copied().collect();
    for &image_support in support.matrices.keys() {
        for left in &action.sectors {
            for right in &action.sectors {
                let preimage_argument = checked_shift_difference(image_support, right.shift)
                    .and_then(|value| {
                        Some([
                            value[0].checked_add(left.shift[0])?,
                            value[1].checked_add(left.shift[1])?,
                            value[2].checked_add(left.shift[2])?,
                        ])
                    })
                    .ok_or_else(|| {
                        BasisRepresentationError::Invalid(
                            "transformed Hamiltonian support overflows isize".to_string(),
                        )
                    })?;
                let preimage = checked_inverse_rotation_vector(&inverse, preimage_argument)
                    .ok_or_else(|| {
                        BasisRepresentationError::Invalid(
                            "inverse-rotated Hamiltonian support overflows isize".to_string(),
                        )
                    })?;
                domain.insert(preimage);
            }
        }
    }
    Ok(domain)
}

fn transformed_hamiltonian_at(
    support: &HamiltonianSupport,
    operation: &CrystalSymmetryOperation,
    action: &LocalizedBasisAction,
    lattice_vector: [isize; 3],
) -> std::result::Result<Array2<Complex64>, BasisRepresentationError> {
    let rotated =
        checked_rotation_vector(&operation.rotation, lattice_vector).ok_or_else(|| {
            BasisRepresentationError::Invalid(
                "rotated Hamiltonian lattice vector overflows isize".to_string(),
            )
        })?;
    let dimension = action.sectors[0].matrix.nrows();
    let mut covariance = Array2::zeros((dimension, dimension));
    for left in &action.sectors {
        let dagger = left.matrix.t().mapv(|value| value.conj());
        for right in &action.sectors {
            let image = [
                rotated[0]
                    .checked_add(right.shift[0])
                    .and_then(|value| value.checked_sub(left.shift[0])),
                rotated[1]
                    .checked_add(right.shift[1])
                    .and_then(|value| value.checked_sub(left.shift[1])),
                rotated[2]
                    .checked_add(right.shift[2])
                    .and_then(|value| value.checked_sub(left.shift[2])),
            ];
            let Some(image) = collect_three_options(image) else {
                return Err(BasisRepresentationError::Invalid(
                    "translated Hamiltonian lattice vector overflows isize".to_string(),
                ));
            };
            if let Some(hamiltonian) = support.matrices.get(&image) {
                covariance += &dagger.dot(hamiltonian).dot(&right.matrix);
            }
        }
    }
    if operation.time_reversal {
        Ok(covariance.mapv(|value| value.conj()))
    } else {
        Ok(covariance)
    }
}

fn hamiltonian_residual(
    support: &HamiltonianSupport,
    operation: &CrystalSymmetryOperation,
    action: &LocalizedBasisAction,
    tolerances: HamiltonianSymmetryTolerances,
) -> std::result::Result<HamiltonianResidual, BasisRepresentationError> {
    let domain = transformed_support_domain(support, operation, action)?;
    let dimension = action.sectors[0].matrix.nrows();
    let threshold = tolerances.absolute + tolerances.relative * support.max_element;
    let first_vector = domain.first().copied().unwrap_or([0, 0, 0]);
    let mut witness = HamiltonianResidualWitness {
        lattice_vector: first_vector,
        bra: 0,
        ket: 0,
        original: Complex64::new(0.0, 0.0),
        transformed: Complex64::new(0.0, 0.0),
    };
    let mut max_absolute = 0.0_f64;
    let mut residual_norm_squared = 0.0_f64;

    for lattice_vector in domain {
        let predicted = transformed_hamiltonian_at(support, operation, action, lattice_vector)?;
        let original = support.matrices.get(&lattice_vector);
        for row in 0..dimension {
            for column in 0..dimension {
                let original_value = original
                    .map(|matrix| matrix[[row, column]])
                    .unwrap_or_else(|| Complex64::new(0.0, 0.0));
                let transformed_value = predicted[[row, column]];
                let residual = (original_value - transformed_value).norm();
                residual_norm_squared += residual * residual;
                if residual > max_absolute {
                    max_absolute = residual;
                    witness = HamiltonianResidualWitness {
                        lattice_vector,
                        bra: row,
                        ket: column,
                        original: original_value,
                        transformed: transformed_value,
                    };
                }
            }
        }
    }

    Ok(HamiltonianResidual {
        max_absolute,
        max_relative: max_absolute / support.max_element.max(f64::EPSILON),
        relative_frobenius: residual_norm_squared.sqrt() / support.frobenius_norm.max(f64::EPSILON),
        acceptance_threshold: threshold,
        witness,
    })
}

fn collect_three_options(values: [Option<isize>; 3]) -> Option<[isize; 3]> {
    Some([values[0]?, values[1]?, values[2]?])
}

fn validate_request(request: &HamiltonianSymmetryRequest) -> Result<()> {
    validate_tolerances(request.tolerances)
}

fn validate_tolerances(tolerances: HamiltonianSymmetryTolerances) -> Result<()> {
    for (parameter, value) in [
        ("absolute_tolerance", tolerances.absolute),
        ("relative_tolerance", tolerances.relative),
        ("hermiticity_tolerance", tolerances.hermiticity),
        ("representation_tolerance", tolerances.representation),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(TbError::InvalidHamiltonianSymmetryInput {
                parameter,
                message: "must be finite and positive".to_string(),
            });
        }
    }
    if tolerances
        .position
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(TbError::InvalidHamiltonianSymmetryInput {
            parameter: "position_tolerance",
            message: "must be None or a finite positive Cartesian length".to_string(),
        });
    }
    if !tolerances.operation.is_finite()
        || tolerances.operation <= 0.0
        || tolerances.operation >= 0.5
    {
        return Err(TbError::InvalidHamiltonianSymmetryInput {
            parameter: "operation_tolerance",
            message: "must be finite and lie in (0, 0.5)".to_string(),
        });
    }
    if tolerances
        .membership
        .is_some_and(|value| !value.is_finite() || value <= 0.0 || value >= 0.5)
    {
        return Err(TbError::InvalidHamiltonianSymmetryInput {
            parameter: "membership_tolerance",
            message: "must be None or finite and lie in (0, 0.5)".to_string(),
        });
    }
    Ok(())
}

fn to_cry_operations(operations: &[CrystalSymmetryOperation]) -> cryspglib::SymmetryOps {
    cryspglib::SymmetryOps {
        operations: operations
            .iter()
            .map(|operation| cryspglib::SymmetryOp {
                rotation: operation.rotation,
                translation: operation.translation,
                time_reversal: operation.time_reversal,
            })
            .collect(),
    }
}

fn from_cry_operation(operation: &cryspglib::SymmetryOp) -> CrystalSymmetryOperation {
    CrystalSymmetryOperation {
        rotation: operation.rotation,
        translation: operation.translation,
        time_reversal: operation.time_reversal,
    }
}

fn operations_equivalent(
    left: &CrystalSymmetryOperation,
    right: &CrystalSymmetryOperation,
    tolerance: f64,
) -> bool {
    left.rotation == right.rotation
        && left.time_reversal == right.time_reversal
        && left
            .translation
            .into_iter()
            .zip(right.translation)
            .all(|(left, right)| {
                let difference = left - right;
                (difference - difference.round()).abs() <= tolerance
            })
}

fn compose_rotation(
    left: &[[i32; 3]; 3],
    right: &[[i32; 3]; 3],
) -> std::result::Result<[[i32; 3]; 3], BasisRepresentationError> {
    let mut product = [[0_i32; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let value = (0..3).try_fold(0_i128, |sum, inner| {
                sum.checked_add(i128::from(left[row][inner]) * i128::from(right[inner][column]))
            });
            product[row][column] = value
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| {
                    BasisRepresentationError::Invalid("rotation product overflows i32".to_string())
                })?;
        }
    }
    Ok(product)
}

fn compose_localized_actions(
    left_operation: &CrystalSymmetryOperation,
    left_action: &LocalizedBasisAction,
    right_action: &LocalizedBasisAction,
) -> std::result::Result<BTreeMap<[isize; 3], Array2<Complex64>>, BasisRepresentationError> {
    let mut composed: BTreeMap<[isize; 3], Array2<Complex64>> = BTreeMap::new();
    for left in &left_action.sectors {
        for right in &right_action.sectors {
            let rotated_right = checked_rotation_vector(&left_operation.rotation, right.shift)
                .ok_or_else(|| {
                    BasisRepresentationError::Invalid(
                        "cell shift overflows while composing basis actions".to_string(),
                    )
                })?;
            let shift = [
                left.shift[0].checked_add(rotated_right[0]),
                left.shift[1].checked_add(rotated_right[1]),
                left.shift[2].checked_add(rotated_right[2]),
            ];
            let shift = collect_three_options(shift).ok_or_else(|| {
                BasisRepresentationError::Invalid(
                    "cell shift overflows while composing basis actions".to_string(),
                )
            })?;
            let right_matrix = if left_operation.time_reversal {
                right.matrix.mapv(|value| value.conj())
            } else {
                right.matrix.clone()
            };
            let contribution = left.matrix.dot(&right_matrix);
            composed
                .entry(shift)
                .and_modify(|matrix| *matrix += &contribution)
                .or_insert(contribution);
        }
    }
    Ok(composed)
}

fn validate_projective_corepresentation(
    operations_and_actions: &[(CrystalSymmetryOperation, LocalizedBasisAction)],
    operation_tolerance: f64,
    representation_tolerance: f64,
) -> std::result::Result<(), BasisRepresentationError> {
    let identity_operation = CrystalSymmetryOperation {
        rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        translation: [0.0; 3],
        time_reversal: false,
    };
    let mut identities = operations_and_actions
        .iter()
        .enumerate()
        .filter(|(_, (operation, _))| {
            operations_equivalent(operation, &identity_operation, operation_tolerance)
        });
    let Some((identity_index, (_, identity_action))) = identities.next() else {
        return Err(BasisRepresentationError::Invalid(
            "the localized corepresentation has no unprimed identity action".to_string(),
        ));
    };
    if identities.next().is_some() {
        return Err(BasisRepresentationError::Ambiguous(
            "the localized corepresentation has multiple unprimed identity actions".to_string(),
        ));
    }
    let dimension = identity_action.sectors[0].matrix.nrows();
    let identity_matrix = identity_action
        .sectors
        .iter()
        .find(|sector| sector.shift == [0, 0, 0])
        .map(|sector| &sector.matrix)
        .ok_or_else(|| {
            BasisRepresentationError::Invalid(format!(
                "identity action {identity_index} has no zero-cell sector"
            ))
        })?;
    let phase = (0..dimension)
        .map(|index| identity_matrix[[index, index]])
        .max_by(|left, right| left.norm().total_cmp(&right.norm()))
        .ok_or_else(|| {
            BasisRepresentationError::Invalid(
                "the localized corepresentation has zero dimension".to_string(),
            )
        })?;
    if (phase.norm() - 1.0).abs() > representation_tolerance {
        return Err(BasisRepresentationError::Invalid(format!(
            "identity action {identity_index} is not a unit-modulus phase times the identity"
        )));
    }
    let mut identity_residual = 0.0_f64;
    for sector in &identity_action.sectors {
        for row in 0..dimension {
            for column in 0..dimension {
                let expected = if sector.shift == [0, 0, 0] && row == column {
                    phase
                } else {
                    Complex64::new(0.0, 0.0)
                };
                identity_residual =
                    identity_residual.max((sector.matrix[[row, column]] - expected).norm());
            }
        }
    }
    if identity_residual > representation_tolerance {
        return Err(BasisRepresentationError::Invalid(format!(
            "identity action {identity_index} is not a global phase times the zero-shift identity (maximum residual {identity_residual:e})"
        )));
    }

    for (left_index, (left_operation, left_action)) in operations_and_actions.iter().enumerate() {
        for (right_index, (right_operation, right_action)) in
            operations_and_actions.iter().enumerate()
        {
            let product_rotation =
                compose_rotation(&left_operation.rotation, &right_operation.rotation)?;
            let product_translation: [f64; 3] = std::array::from_fn(|axis| {
                left_operation.translation[axis]
                    + (0..3)
                        .map(|input| {
                            f64::from(left_operation.rotation[axis][input])
                                * right_operation.translation[input]
                        })
                        .sum::<f64>()
            });
            let product_operation = CrystalSymmetryOperation {
                rotation: product_rotation,
                translation: product_translation,
                time_reversal: left_operation.time_reversal ^ right_operation.time_reversal,
            };
            let mut products =
                operations_and_actions
                    .iter()
                    .enumerate()
                    .filter(|(_, (candidate, _))| {
                        operations_equivalent(&product_operation, candidate, operation_tolerance)
                    });
            let Some((product_index, (canonical_product, product_action))) = products.next() else {
                return Err(BasisRepresentationError::Invalid(format!(
                    "basis-action product ({left_index}, {right_index}) has no target operation"
                )));
            };
            if products.next().is_some() {
                return Err(BasisRepresentationError::Ambiguous(format!(
                    "basis-action product ({left_index}, {right_index}) matches multiple target operations"
                )));
            }
            let mut representative_shift = [0_isize; 3];
            for axis in 0..3 {
                let difference = product_translation[axis] - canonical_product.translation[axis];
                let nearest = difference.round();
                if (difference - nearest).abs() > operation_tolerance
                    || nearest < isize::MIN as f64
                    || nearest > isize::MAX as f64
                {
                    return Err(BasisRepresentationError::Invalid(format!(
                        "operation product ({left_index}, {right_index}) has an inconsistent translation representative"
                    )));
                }
                representative_shift[axis] = nearest as isize;
            }

            let actual = compose_localized_actions(left_operation, left_action, right_action)?;
            let mut expected = BTreeMap::new();
            for sector in &product_action.sectors {
                let shifted = [
                    sector.shift[0].checked_add(representative_shift[0]),
                    sector.shift[1].checked_add(representative_shift[1]),
                    sector.shift[2].checked_add(representative_shift[2]),
                ];
                let shifted = collect_three_options(shifted).ok_or_else(|| {
                    BasisRepresentationError::Invalid(
                        "target action shift overflows during composition check".to_string(),
                    )
                })?;
                expected.insert(shifted, sector.matrix.clone());
            }

            let reference = expected
                .iter()
                .flat_map(|(shift, matrix)| {
                    matrix
                        .indexed_iter()
                        .map(move |((row, column), value)| (*shift, row, column, *value))
                })
                .max_by(|left, right| left.3.norm().total_cmp(&right.3.norm()))
                .ok_or_else(|| {
                    BasisRepresentationError::Invalid(format!(
                        "target action {product_index} is empty"
                    ))
                })?;
            let actual_reference = actual
                .get(&reference.0)
                .map(|matrix| matrix[[reference.1, reference.2]])
                .unwrap_or_else(|| Complex64::new(0.0, 0.0));
            if reference.3.norm() <= representation_tolerance
                || actual_reference.norm() <= representation_tolerance
            {
                return Err(BasisRepresentationError::Invalid(format!(
                    "basis actions {left_index} and {right_index} do not compose to target action {product_index}"
                )));
            }
            let phase_ratio = actual_reference / reference.3;
            let phase = phase_ratio / phase_ratio.norm();
            let keys = actual
                .keys()
                .chain(expected.keys())
                .copied()
                .collect::<BTreeSet<_>>();
            let dimension = left_action.sectors[0].matrix.nrows();
            let mut max_residual = 0.0_f64;
            for shift in keys {
                for row in 0..dimension {
                    for column in 0..dimension {
                        let actual_value = actual
                            .get(&shift)
                            .map(|matrix| matrix[[row, column]])
                            .unwrap_or_else(|| Complex64::new(0.0, 0.0));
                        let expected_value = expected
                            .get(&shift)
                            .map(|matrix| phase * matrix[[row, column]])
                            .unwrap_or_else(|| Complex64::new(0.0, 0.0));
                        max_residual = max_residual.max((actual_value - expected_value).norm());
                    }
                }
            }
            if max_residual > representation_tolerance {
                return Err(BasisRepresentationError::Invalid(format!(
                    "basis actions {left_index} and {right_index} violate projective group composition for target {product_index} (maximum residual {max_residual:e})"
                )));
            }
        }
    }
    Ok(())
}

fn symmetrized_support(
    support: &HamiltonianSupport,
    operations_and_actions: &[(CrystalSymmetryOperation, LocalizedBasisAction)],
) -> std::result::Result<BTreeMap<[isize; 3], Array2<Complex64>>, BasisRepresentationError> {
    let mut domain: BTreeSet<[isize; 3]> = support.matrices.keys().copied().collect();
    for (operation, action) in operations_and_actions {
        domain.extend(transformed_support_domain(support, operation, action)?);
    }
    let existing = domain.iter().copied().collect::<Vec<_>>();
    for lattice_vector in existing {
        let negative = checked_negate(lattice_vector).ok_or_else(|| {
            BasisRepresentationError::Invalid(
                "symmetrized support contains a lattice vector that cannot be negated".to_string(),
            )
        })?;
        domain.insert(negative);
    }

    let dimension = support.matrices.values().next().map_or(0, Array2::nrows);
    let normalizer = operations_and_actions.len() as f64;
    let mut averaged = BTreeMap::new();
    for lattice_vector in domain {
        let mut matrix = Array2::zeros((dimension, dimension));
        for (operation, action) in operations_and_actions {
            matrix += &transformed_hamiltonian_at(support, operation, action, lattice_vector)?;
        }
        matrix.mapv_inplace(|value| value / normalizer);
        averaged.insert(lattice_vector, matrix);
    }

    // A Reynolds average of a Hermitian Hamiltonian is Hermitian in exact
    // arithmetic. Enforce the paired relation once to remove roundoff without
    // changing the mathematical projection.
    let keys = averaged.keys().copied().collect::<Vec<_>>();
    let mut processed = BTreeSet::new();
    for lattice_vector in keys {
        if !processed.insert(lattice_vector) {
            continue;
        }
        let negative = checked_negate(lattice_vector).ok_or_else(|| {
            BasisRepresentationError::Invalid(
                "symmetrized support contains a lattice vector that cannot be negated".to_string(),
            )
        })?;
        processed.insert(negative);
        let matrix = averaged[&lattice_vector].clone();
        let partner_dagger = averaged[&negative].t().mapv(|value| value.conj());
        let hermitian = (matrix + partner_dagger).mapv(|value| value * 0.5);
        let hermitian_dagger = hermitian.t().mapv(|value| value.conj());
        averaged.insert(lattice_vector, hermitian);
        averaged.insert(negative, hermitian_dagger);
    }
    Ok(averaged)
}

fn model_with_hamiltonian_support<const SPIN: bool, R: RMatrixData>(
    model: &Model<SPIN, 3, R>,
    mut matrices: BTreeMap<[isize; 3], Array2<Complex64>>,
) -> Result<Model<SPIN, 3, R>> {
    // Rustb's hopping builders treat row zero as the onsite block. Preserve
    // that invariant, retain the input order for existing blocks, and append
    // symmetry-generated blocks deterministically.
    matrices
        .entry([0, 0, 0])
        .or_insert_with(|| Array2::zeros((model.nsta(), model.nsta())));
    let mut ordered_vectors = Vec::with_capacity(matrices.len());
    let mut seen = BTreeSet::new();
    ordered_vectors.push([0, 0, 0]);
    seen.insert([0, 0, 0]);
    for index in 0..model.hamR.nrows() {
        let lattice_vector = [
            model.hamR[[index, 0]],
            model.hamR[[index, 1]],
            model.hamR[[index, 2]],
        ];
        if matrices.contains_key(&lattice_vector) && seen.insert(lattice_vector) {
            ordered_vectors.push(lattice_vector);
        }
    }
    for lattice_vector in matrices.keys().copied() {
        if seen.insert(lattice_vector) {
            ordered_vectors.push(lattice_vector);
        }
    }
    let mut support = Vec::with_capacity(ordered_vectors.len());
    for lattice_vector in ordered_vectors {
        let matrix = matrices.remove(&lattice_vector).ok_or_else(|| {
            TbError::TargetMagneticGroupIncompatible {
                reason: format!(
                    "internal support reconstruction lost lattice vector {lattice_vector:?}"
                ),
            }
        })?;
        support.push((lattice_vector, matrix));
    }
    let mut ham = Array3::zeros((support.len(), model.nsta(), model.nsta()));
    let mut ham_r = Array2::zeros((support.len(), 3));
    for (index, (lattice_vector, matrix)) in support.iter().enumerate() {
        ham.index_axis_mut(Axis(0), index).assign(matrix);
        for axis in 0..3 {
            ham_r[[index, axis]] = lattice_vector[axis];
        }
    }

    // Position-matrix elements are independent operator data, not part of the
    // Hamiltonian Reynolds projection. Preserve every existing block and add
    // explicit zeros only for newly generated hopping support.
    let mut rmatrix = Array4::zeros((support.len(), 3, model.nsta(), model.nsta()));
    if R::HAS_RMATRIX {
        let old_indices = (0..model.hamR.nrows())
            .map(|index| {
                (
                    [
                        model.hamR[[index, 0]],
                        model.hamR[[index, 1]],
                        model.hamR[[index, 2]],
                    ],
                    index,
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (new_index, (lattice_vector, _)) in support.iter().enumerate() {
            if let Some(&old_index) = old_indices.get(lattice_vector) {
                rmatrix
                    .index_axis_mut(Axis(0), new_index)
                    .assign(&model.rmatrix.as_array4().index_axis(Axis(0), old_index));
            }
        }
    }

    let mut result = model.clone();
    result.ham = ham;
    result.hamR = ham_r;
    result.rmatrix = R::from_array(rmatrix);
    Ok(result)
}

impl<const SPIN: bool, R: RMatrixData> Model<SPIN, 3, R> {
    /// Certify which atomic-structure symmetries are symmetries of $H$.
    ///
    /// The structure is analyzed from [`crate::Atom`] positions first. A
    /// model without atoms is rejected even if it has orbital centers. For
    /// every field-allowed candidate
    /// $g=\{W|\mathbf w\}\mathcal T^\theta$, `representation` supplies the
    /// localized action
    ///
    /// $$
    /// g|b,\mathbf R\rangle=
    /// \sum_{\mathbf s,a}(D^g_{\mathbf s})_{ab}
    /// |a,W\mathbf R+\mathbf s\rangle.
    /// $$
    ///
    /// Rustb's stored convention is
    /// $H_{ab}(\mathbf R)=\langle a,0|H|b,\mathbf R\rangle$. Consequently the
    /// implementation evaluates
    ///
    /// $$
    /// K_g(\mathbf R)=\sum_{\mathbf s,\mathbf t}
    /// (D^g_{\mathbf s})^\dagger
    /// H(W\mathbf R+\mathbf t-\mathbf s)D^g_{\mathbf t}.
    /// $$
    ///
    /// It checks $H(\mathbf R)=K_g(\mathbf R)$ for unitary operations and
    /// $H(\mathbf R)=K_g(\mathbf R)^*$ for anti-unitary operations. The
    /// comparison domain is the union of the stored Laurent support and its
    /// exact inverse image; an absent hopping block is zero. This makes the
    /// result independent of a chosen $k$ mesh and includes nonsymmorphic cell
    /// shifts exactly.
    ///
    /// By default the candidate supergroup is the grey extension $G+G1'$ of
    /// the atom-derived space group. This is necessary to discover Type-II,
    /// Type-III, and Type-IV magnetic groups. Explicit electric and magnetic
    /// fields in [`SymmetryParameters`] filter that supergroup without
    /// modifying the model. The final survivor set is accepted as a named
    /// magnetic group only after cryspglib verifies identity, inverses,
    /// multiplication closure, and derives the survivor's own family Hall
    /// setting. A structural Hall number is retained only as provenance.
    ///
    /// # Basis metadata and inconclusive results
    ///
    /// [`OrbProj`] alone does not determine the symmetry gauge of a general
    /// Wannier basis. [`ScalarSiteBasis`] is therefore intentionally strict;
    /// use a custom [`BasisSymmetryRepresentation`] for orbital mixing, local
    /// frames, or an explicit Wannier gauge. If any action is unsupported or
    /// ambiguous, the result is [`FinalMagneticGroup::Inconclusive`] and
    /// [`HamiltonianSymmetryCompleteness::LowerBound`]. Such an operation is
    /// never misreported as physically broken.
    ///
    /// # Periodic-Hamiltonian scope
    ///
    /// The certification assumes an ordinary periodic tight-binding
    /// Hamiltonian. A uniform electric field in an infinite bulk or a
    /// Peierls magnetic field requiring a position-dependent gauge phase needs
    /// a richer representation provider; field metadata alone does not add
    /// those gauge compensations.
    pub fn check_hamiltonian_symmetry<P>(
        &self,
        representation: &P,
        request: &HamiltonianSymmetryRequest,
    ) -> Result<HamiltonianSymmetryReport>
    where
        P: BasisSymmetryRepresentation<SPIN, R>,
    {
        validate_request(request)?;
        let support = validate_hamiltonian(self, request.tolerances.hermiticity)?;
        let structure = self.crystal_symmetry(&request.structural_parameters)?;
        let position_tolerance = request
            .tolerances
            .position
            .unwrap_or(request.structural_parameters.symprec);
        let lattice = cry_lattice(self);
        let structural = to_cry_operations(&structure.operations);
        let candidates = match request.candidates {
            HamiltonianSymmetryCandidates::StructuralGrey => structural.grey_extension()?,
            HamiltonianSymmetryCandidates::StructuralUnitary => structural,
        };
        let structure_candidates = candidates
            .iter()
            .map(from_cry_operation)
            .collect::<Vec<_>>();
        let field_allowed = candidates.preserving_fields(
            &lattice,
            cryspglib::ExternalFields {
                electric: request.structural_parameters.external_fields.electric,
                magnetic: request.structural_parameters.external_fields.magnetic,
            },
            request.structural_parameters.field_tolerance,
        )?;
        let field_allowed_operations = field_allowed
            .iter()
            .map(from_cry_operation)
            .collect::<Vec<_>>();

        let mut operation_checks = Vec::with_capacity(field_allowed_operations.len());
        let mut surviving_operations = Vec::new();
        let mut unresolved = false;
        let mut broken = false;
        for operation in &field_allowed_operations {
            let resolved = representation
                .resolve(BasisActionContext {
                    model: self,
                    operation,
                    position_tolerance,
                    representation_tolerance: request.tolerances.representation,
                })
                .and_then(|action| {
                    let action =
                        validate_action(action, self.nsta(), request.tolerances.representation)?;
                    validate_action_geometry(
                        self,
                        operation,
                        &action,
                        position_tolerance,
                        request.tolerances.representation,
                    )?;
                    Ok(action)
                });
            match resolved {
                Ok(action) => {
                    let residual =
                        hamiltonian_residual(&support, operation, &action, request.tolerances);
                    match residual {
                        Ok(residual) if residual.max_absolute <= residual.acceptance_threshold => {
                            surviving_operations.push(*operation);
                            operation_checks.push(OperationHamiltonianCheck {
                                operation: *operation,
                                action: Some(action),
                                status: OperationHamiltonianStatus::Preserved(residual),
                            });
                        }
                        Ok(residual) => {
                            broken = true;
                            operation_checks.push(OperationHamiltonianCheck {
                                operation: *operation,
                                action: Some(action),
                                status: OperationHamiltonianStatus::Broken(residual),
                            });
                        }
                        Err(error) => {
                            unresolved = true;
                            operation_checks.push(OperationHamiltonianCheck {
                                operation: *operation,
                                action: Some(action),
                                status: OperationHamiltonianStatus::Unresolved(error),
                            });
                        }
                    }
                }
                Err(error) => {
                    unresolved = true;
                    operation_checks.push(OperationHamiltonianCheck {
                        operation: *operation,
                        action: None,
                        status: OperationHamiltonianStatus::Unresolved(error),
                    });
                }
            }
        }

        let completeness = if unresolved {
            HamiltonianSymmetryCompleteness::LowerBound
        } else {
            HamiltonianSymmetryCompleteness::Complete
        };
        let compatibility = if unresolved {
            HamiltonianCompatibility::Inconclusive
        } else if broken {
            HamiltonianCompatibility::SymmetryReduced
        } else {
            HamiltonianCompatibility::Compatible
        };

        let final_group = if request.candidates == HamiltonianSymmetryCandidates::StructuralUnitary
        {
            FinalMagneticGroup::Inconclusive {
                reason: "only unitary structural candidates were tested; anti-unitary symmetry is not exhausted"
                    .to_string(),
            }
        } else if unresolved {
            FinalMagneticGroup::Inconclusive {
                reason: "one or more localized-basis actions are unresolved; surviving operations are only a lower bound"
                    .to_string(),
            }
        } else {
            identify_surviving_group(
                &surviving_operations,
                structure_candidates.len(),
                &lattice,
                structure.hall_number,
                request,
            )
        };

        Ok(HamiltonianSymmetryReport {
            structure,
            structure_candidates,
            field_allowed_operations,
            operation_checks,
            surviving_operations,
            compatibility,
            completeness,
            final_group,
        })
    }

    /// Return a new Model whose Hamiltonian is forced to preserve `target_group`.
    ///
    /// This method never mutates `self`. The target must be a complete
    /// [`MagneticCrystalSymmetry`] expressed in the current Model's fractional
    /// basis, normally obtained from
    /// [`CrystalSymmetry::magnetic_crystal_symmetry_from_atoms`] or
    /// [`CrystalSymmetry::magnetic_crystal_symmetry`]. Before changing a matrix
    /// element, the implementation verifies that:
    ///
    /// 1. every target operation belongs to the current Atom-derived magnetic
    ///    structure. `None` moments are treated as zero and hence give the
    ///    structural grey group $G+G\mathcal T$; explicit site moments and
    ///    electric/magnetic fields further reduce the admissible operations;
    /// 2. the target operations contain the identity and are unique, invertible,
    ///    and closed;
    /// 3. identifying those operations on the current lattice reproduces the
    ///    supplied target UNI number; and
    /// 4. `representation` resolves a finite unitary localized action for every
    ///    target operation, and those actions form one projective magnetic
    ///    corepresentation.
    ///
    /// A failure of the target operations, target metadata, structural/magnetic
    /// compatibility, or localized corepresentation is returned as
    /// [`TbError::TargetMagneticGroupIncompatible`]. A database group in a
    /// different setting is therefore rejected rather than silently applied
    /// to the wrong coordinate frame. Malformed Model data and invalid numeric
    /// parameters retain their more specific validation errors. In particular,
    /// target compatibility finishes before `representation` is called and
    /// before any Hamiltonian matrix is averaged.
    ///
    /// # Errors
    ///
    /// Returns the Model's existing validation errors for malformed Hamiltonian
    /// data or invalid tolerances. Returns
    /// [`TbError::TargetMagneticGroupIncompatible`] when the supplied magnetic
    /// group is malformed, belongs to a different structural/magnetic setting,
    /// is broken by the Atom moments or field context, has inconsistent
    /// metadata, or cannot be represented by `representation`.
    ///
    /// # Structure and magnetic compatibility
    ///
    /// Let $\boldsymbol\tau_i$ be the fractional coordinate of Atom $i$ and
    /// let
    ///
    /// $$
    /// Q_g=L W_g L^{-1}
    /// $$
    ///
    /// be its Cartesian rotation, where the columns of $L$ are the lattice
    /// vectors. Every target operation must induce a same-`AtomType`
    /// bijection satisfying
    ///
    /// $$
    /// W_g\boldsymbol\tau_i+\mathbf w_g
    /// =\boldsymbol\tau_{g(i)}+\mathbf n_i,
    /// \qquad \mathbf n_i\in\mathbb Z^3.
    /// $$
    ///
    /// If optional Cartesian moments are attached to the Atoms, they are
    /// time-odd axial vectors and must additionally obey
    ///
    /// $$
    /// \mathbf m_{g(i)}=(-1)^{\theta_g}\det(Q_g)Q_g\mathbf m_i.
    /// $$
    ///
    /// `None` is analyzed as a zero moment, so a default nonmagnetic structure
    /// admits its grey extension. The electric and magnetic fields in
    /// [`SymmetryParameters`] are also checked as, respectively, a time-even
    /// polar vector and a time-odd axial vector. Thus a target broken by the
    /// Model's Atom moments or by the supplied Hamiltonian field context is a
    /// hard error, even when the bare lattice is unchanged.
    ///
    /// For operations $g$ and $h$, resolved localized actions must compose as
    ///
    /// $$
    /// \sum_{\mathbf s+W_g\mathbf t=\mathbf u}
    /// D^g_{\mathbf s}\left(D^h_{\mathbf t}\right)^{*\theta_g}
    /// =z_{g,h}D^{gh}_{\mathbf u-\mathbf n_{g,h}},
    /// \qquad |z_{g,h}|=1,
    /// $$
    ///
    /// where $\mathbf n_{g,h}$ accounts for the integer translation used to
    /// normalize the Seitz representative. Allowing the global phase
    /// $z_{g,h}$ admits spin-$\tfrac12$ double groups and
    /// $\mathcal T^2=-1$, while still rejecting unrelated per-operation
    /// matrices.
    ///
    /// For a localized action $D^g_{\mathbf s}$, define the real-linear group
    /// action on Hamiltonians by
    ///
    /// $$
    /// (\mathcal P_g H)(\mathbf R)=
    /// \begin{cases}
    /// \displaystyle\sum_{\mathbf s,\mathbf t}
    /// (D^g_{\mathbf s})^\dagger
    /// H(W_g\mathbf R+\mathbf t-\mathbf s)D^g_{\mathbf t},
    /// & g\text{ unitary},\\
    /// \displaystyle\left[\sum_{\mathbf s,\mathbf t}
    /// (D^g_{\mathbf s})^\dagger
    /// H(W_g\mathbf R+\mathbf t-\mathbf s)D^g_{\mathbf t}\right]^*,
    /// & g\text{ anti-unitary}.
    /// \end{cases}
    /// $$
    ///
    /// The returned Hamiltonian is the magnetic Reynolds projection
    ///
    /// $$
    /// H_{\mathrm{sym}}=\frac{1}{|M|}\sum_{g\in M}\mathcal P_g H.
    /// $$
    ///
    /// Its finite `hamR` support is expanded to the union of all transformed
    /// supports, then Hermiticity is enforced pairwise. Finally every target
    /// covariance equation is rechecked; inconsistent custom representation
    /// matrices cause an error instead of returning a plausibly symmetrized
    /// model.
    ///
    /// `rmatrix`, when present, is independent operator data: existing blocks
    /// are preserved and newly introduced `hamR` blocks receive zero position
    /// matrices. This method certifies the returned Hamiltonian, not arbitrary
    /// auxiliary operators.
    pub fn symmetrize_hamiltonian<P>(
        &self,
        target_group: &MagneticCrystalSymmetry,
        representation: &P,
        parameters: &HamiltonianSymmetrizationParameters,
    ) -> Result<Self>
    where
        P: BasisSymmetryRepresentation<SPIN, R>,
    {
        validate_tolerances(parameters.tolerances)?;
        let support = validate_hamiltonian(self, parameters.tolerances.hermiticity)?;
        let structure = self.crystal_symmetry(&parameters.structural_parameters)?;
        let lattice = cry_lattice(self);
        if target_group.operations.is_empty() {
            return Err(TbError::TargetMagneticGroupIncompatible {
                reason: "the supplied magnetic group has no operations".to_string(),
            });
        }
        let target_operations = to_cry_operations(&target_group.operations);
        let validated = cryspglib::ValidatedMagneticOperationSet::try_from_symmetry_ops(
            &target_operations,
            parameters.tolerances.operation,
        )
        .map_err(|error| TbError::TargetMagneticGroupIncompatible {
            reason: format!("the supplied operations are not a magnetic group: {error}"),
        })?;
        let normalized_target_operations = validated
            .operations()
            .iter()
            .map(from_cry_operation)
            .collect::<Vec<_>>();

        // Recompute the admissible operation set from this Model rather than
        // trusting the provenance of the supplied public data object. This
        // checks lattice, atom positions/types, optional site moments, and the
        // explicitly supplied field context before invoking the basis provider
        // or changing any Hamiltonian block.
        let atom_magnetic_structure = self
            .magnetic_crystal_symmetry_from_atoms(&parameters.structural_parameters)
            .map_err(|error| TbError::TargetMagneticGroupIncompatible {
                reason: format!(
                    "the current Atom structure and optional magnetic moments cannot be analyzed: {error}"
                ),
            })?;
        let structure_compatible_operations = &atom_magnetic_structure.field_preserving_operations;
        // Structural detection only guarantees symprec accuracy, so target
        // operations may legitimately differ from detected ones by more than
        // tolerances.operation (e.g. database values rounded to 6 decimals).
        // Use a dedicated membership tolerance: by default the same symprec
        // used to detect the structure in the first place.
        let membership_tolerance = parameters
            .tolerances
            .membership
            .unwrap_or(parameters.structural_parameters.symprec);
        for (index, operation) in normalized_target_operations.iter().enumerate() {
            if !structure_compatible_operations
                .iter()
                .any(|candidate| operations_equivalent(operation, candidate, membership_tolerance))
            {
                return Err(TbError::TargetMagneticGroupIncompatible {
                    reason: format!(
                        "operation {index} is not compatible with the current Model lattice, Atom positions/types, optional Atom magnetic moments, and external-field context"
                    ),
                });
            }
        }

        let identified = validated
            .identify(
                &lattice,
                Some(structure.hall_number),
                parameters.structural_parameters.symprec,
            )
            .map_err(|error| TbError::TargetMagneticGroupIncompatible {
                reason: format!("the supplied group cannot be identified on this lattice: {error}"),
            })?;
        if identified.uni_number != target_group.uni_number {
            return Err(TbError::TargetMagneticGroupIncompatible {
                reason: format!(
                    "the operations identify as UNI {}, but the supplied group metadata says UNI {}",
                    identified.uni_number, target_group.uni_number
                ),
            });
        }
        // MagneticCrystalSymmetry::spacegroup_number/hall_number describe the
        // nonmagnetic structural parent used during site-moment analysis. The
        // identifier's spacegroup_number instead describes the MSG family
        // space group, so those intentionally different quantities must not
        // be compared here.
        if convert_magnetic_type(identified.magnetic_type) != target_group.magnetic_type
            || identified.bns_number != target_group.bns_number
            || identified.og_number != target_group.og_number
        {
            return Err(TbError::TargetMagneticGroupIncompatible {
                reason: format!(
                    "the supplied magnetic-group metadata does not match the normalized operations (identified BNS {}, OG {}, type {:?}; supplied BNS {}, OG {}, type {:?})",
                    identified.bns_number,
                    identified.og_number,
                    convert_magnetic_type(identified.magnetic_type),
                    target_group.bns_number,
                    target_group.og_number,
                    target_group.magnetic_type,
                ),
            });
        }

        let position_tolerance = parameters
            .tolerances
            .position
            .unwrap_or(parameters.structural_parameters.symprec);
        let mut operations_and_actions = Vec::with_capacity(normalized_target_operations.len());
        for (index, operation) in normalized_target_operations.iter().enumerate() {
            let action = representation
                .resolve(BasisActionContext {
                    model: self,
                    operation,
                    position_tolerance,
                    representation_tolerance: parameters.tolerances.representation,
                })
                .and_then(|action| {
                    let action =
                        validate_action(action, self.nsta(), parameters.tolerances.representation)?;
                    validate_action_geometry(
                        self,
                        operation,
                        &action,
                        position_tolerance,
                        parameters.tolerances.representation,
                    )?;
                    Ok(action)
                })
                .map_err(|error| TbError::TargetMagneticGroupIncompatible {
                    reason: format!(
                        "operation {index} has no compatible localized-basis action: {error}"
                    ),
                })?;
            operations_and_actions.push((*operation, action));
        }

        validate_projective_corepresentation(
            &operations_and_actions,
            parameters.tolerances.operation,
            parameters.tolerances.representation,
        )
        .map_err(|error| TbError::TargetMagneticGroupIncompatible {
            reason: format!(
                "localized basis actions do not form the target corepresentation: {error}"
            ),
        })?;

        let averaged = symmetrized_support(&support, &operations_and_actions).map_err(|error| {
            TbError::TargetMagneticGroupIncompatible {
                reason: format!("failed to apply the target representation: {error}"),
            }
        })?;
        let symmetrized = model_with_hamiltonian_support(self, averaged)?;
        let projected_support =
            validate_hamiltonian(&symmetrized, parameters.tolerances.hermiticity)?;
        for (index, (operation, action)) in operations_and_actions.iter().enumerate() {
            let residual =
                hamiltonian_residual(&projected_support, operation, action, parameters.tolerances)
                    .map_err(|error| TbError::TargetMagneticGroupIncompatible {
                        reason: format!(
                            "post-projection check failed for operation {index}: {error}"
                        ),
                    })?;
            if residual.max_absolute > residual.acceptance_threshold {
                return Err(TbError::TargetMagneticGroupIncompatible {
                    reason: format!(
                        "the supplied basis actions are not mutually consistent: post-projection operation {index} has residual {:e} above {:e}",
                        residual.max_absolute, residual.acceptance_threshold
                    ),
                });
            }
        }
        Ok(symmetrized)
    }
}

fn identify_surviving_group(
    surviving_operations: &[CrystalSymmetryOperation],
    candidate_count: usize,
    lattice: &[[f64; 3]; 3],
    structural_hall_number: usize,
    request: &HamiltonianSymmetryRequest,
) -> FinalMagneticGroup {
    let surviving = to_cry_operations(surviving_operations);
    let validated = match cryspglib::ValidatedMagneticOperationSet::try_from_symmetry_ops(
        &surviving,
        request.tolerances.operation,
    ) {
        Ok(validated) => validated,
        Err(error) => {
            return FinalMagneticGroup::Inconclusive {
                reason: format!(
                    "Hamiltonian-preserving operations do not form a numerically certified magnetic group: {error}"
                ),
            };
        }
    };
    if !candidate_count.is_multiple_of(validated.len()) {
        return FinalMagneticGroup::Inconclusive {
            reason: format!(
                "a closed survivor set of order {} does not divide the candidate order {candidate_count}",
                validated.len()
            ),
        };
    }
    let identification = match validated.identify(
        lattice,
        Some(structural_hall_number),
        request.structural_parameters.symprec,
    ) {
        Ok(identification) => identification,
        Err(error) => {
            return FinalMagneticGroup::Inconclusive {
                reason: format!("magnetic-group identification failed: {error}"),
            };
        }
    };
    FinalMagneticGroup::Identified(Box::new(IdentifiedMagneticSubgroup {
        uni_number: identification.uni_number,
        litvin_number: identification.litvin_number,
        family_spacegroup_number: identification.spacegroup_number,
        bns_number: identification.bns_number,
        og_number: identification.og_number,
        magnetic_type: convert_magnetic_type(identification.magnetic_type),
        family_hall_number: identification.family_hall_number,
        hall_number: identification.hall_number,
        structural_supergroup_hall: structural_hall_number,
        transformation_matrix: matrix3(identification.transformation_matrix),
        origin_shift: identification.origin_shift,
        standard_rotation_matrix: matrix3(identification.std_rotation_matrix),
        subgroup_index_in_candidates: candidate_count / validated.len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Atom, AtomType, HasRMatrix, Model, NoRMatrix, OrbitalId, RMatrixData, SpinDirection,
    };
    use ndarray::{Array2, Array4, Axis, array};

    struct MustNotResolve;

    impl BasisSymmetryRepresentation<false, NoRMatrix> for MustNotResolve {
        fn resolve(
            &self,
            _context: BasisActionContext<'_, false, NoRMatrix>,
        ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
            panic!("the representation provider must not be called for an incompatible structure")
        }
    }

    struct InvalidIdentityCorepresentation;

    impl BasisSymmetryRepresentation<true, NoRMatrix> for InvalidIdentityCorepresentation {
        fn resolve(
            &self,
            context: BasisActionContext<'_, true, NoRMatrix>,
        ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
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

    fn hopping(model: &Model<false, 3, NoRMatrix>, lattice_vector: [isize; 3]) -> Complex64 {
        let index = (0..model.hamR.nrows())
            .find(|&index| (0..3).all(|axis| model.hamR[[index, axis]] == lattice_vector[axis]))
            .expect("requested hopping block must be stored");
        model.ham[[index, 0, 0]]
    }

    fn scalar_cubic() -> Model<false, 3, NoRMatrix> {
        Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap()
    }

    fn spinful_cubic() -> Model<true, 3, NoRMatrix> {
        Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap()
    }

    fn spinful_half_translation_model() -> Model<true, 3, NoRMatrix> {
        Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::H, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::H, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap()
    }

    fn atomic_orbital_cubic<const SPIN: bool>(
        projections: &[OrbProj],
    ) -> Model<SPIN, 3, NoRMatrix> {
        let orbitals = (0..projections.len())
            .map(OrbitalId::new)
            .collect::<Vec<_>>();
        let mut model = Model::tb_model(
            Array2::eye(3),
            Array2::zeros((projections.len(), 3)),
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                orbitals,
            )]),
        )
        .unwrap();
        model.orb_projection = projections.to_vec();
        model
    }

    fn c4z(time_reversal: bool) -> CrystalSymmetryOperation {
        CrystalSymmetryOperation {
            rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            translation: [0.0; 3],
            time_reversal,
        }
    }

    fn identity_operation(time_reversal: bool) -> CrystalSymmetryOperation {
        CrystalSymmetryOperation {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0; 3],
            time_reversal,
        }
    }

    fn resolve_atomic<const SPIN: bool>(
        model: &Model<SPIN, 3, NoRMatrix>,
        operation: &CrystalSymmetryOperation,
    ) -> std::result::Result<LocalizedBasisAction, BasisRepresentationError> {
        AtomicOrbitalBasis.resolve(BasisActionContext {
            model,
            operation,
            position_tolerance: 1e-8,
            representation_tolerance: 1e-8,
        })
    }

    fn half_translation_model(intracell: f64, intercell: f64) -> Model<false, 3, NoRMatrix> {
        let mut model: Model<false, 3, NoRMatrix> = Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::H, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::H, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        model.set_hop(intracell, 0, 1, &array![0_isize, 0, 0], None);
        model.set_hop(intercell, 1, 0, &array![1_isize, 0, 0], None);
        model
    }

    fn identity_half_translation(time_reversal: bool) -> CrystalSymmetryOperation {
        CrystalSymmetryOperation {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.5, 0.0, 0.0],
            time_reversal,
        }
    }

    #[test]
    fn scalar_cubic_zero_hamiltonian_has_full_grey_group() {
        let report = scalar_cubic()
            .check_hamiltonian_symmetry(
                &ScalarSiteBasis::default(),
                &HamiltonianSymmetryRequest::default(),
            )
            .unwrap();

        assert_eq!(report.structure.spacegroup_number, 221);
        assert_eq!(report.structure_candidates.len(), 96);
        assert_eq!(report.field_allowed_operations.len(), 96);
        assert_eq!(report.surviving_operations.len(), 96);
        assert_eq!(report.is_fully_compatible(), Some(true));
        assert_eq!(
            report.completeness,
            HamiltonianSymmetryCompleteness::Complete
        );
        let FinalMagneticGroup::Identified(group) = report.final_group else {
            panic!("full grey group should be identified");
        };
        assert_eq!(group.magnetic_type, MagneticGroupType::Grey);
        assert_eq!(group.subgroup_index_in_candidates, 1);
    }

    #[test]
    fn unitary_only_diagnostic_never_claims_a_final_magnetic_group() {
        let mut request = HamiltonianSymmetryRequest::default();
        request.candidates = HamiltonianSymmetryCandidates::StructuralUnitary;
        let report = scalar_cubic()
            .check_hamiltonian_symmetry(&ScalarSiteBasis, &request)
            .unwrap();
        assert_eq!(report.structure_candidates.len(), 48);
        assert_eq!(report.surviving_operations.len(), 48);
        assert_eq!(report.is_fully_compatible(), Some(true));
        let FinalMagneticGroup::Inconclusive { reason } = report.final_group else {
            panic!("unitary candidates cannot exhaust anti-unitary symmetry");
        };
        assert!(reason.contains("only unitary structural candidates"));
    }

    #[test]
    fn anisotropic_hopping_reduces_cubic_structure_to_orthorhombic_grey_group() {
        let mut model = scalar_cubic();
        model.set_hop(1.0_f64, 0, 0, &array![1_isize, 0, 0], None);
        model.set_hop(2.0_f64, 0, 0, &array![0_isize, 1, 0], None);
        model.set_hop(3.0_f64, 0, 0, &array![0_isize, 0, 1], None);

        let report = model
            .check_hamiltonian_symmetry(
                &ScalarSiteBasis::default(),
                &HamiltonianSymmetryRequest::default(),
            )
            .unwrap();

        assert_eq!(
            report.compatibility,
            HamiltonianCompatibility::SymmetryReduced
        );
        assert_eq!(report.surviving_operations.len(), 16);
        let FinalMagneticGroup::Identified(group) = report.final_group else {
            panic!("closed orthorhombic survivor should be identified");
        };
        assert_eq!(group.magnetic_type, MagneticGroupType::Grey);
        assert_eq!(group.subgroup_index_in_candidates, 6);
    }

    #[test]
    fn complex_directed_hopping_breaks_pure_time_reversal() {
        let mut model = scalar_cubic();
        model.set_hop(
            Complex64::new(1.0, 0.25),
            0,
            0,
            &array![1_isize, 0, 0],
            None,
        );
        let report = model
            .check_hamiltonian_symmetry(
                &ScalarSiteBasis::default(),
                &HamiltonianSymmetryRequest::default(),
            )
            .unwrap();
        let pure_time_reversal = report
            .operation_checks
            .iter()
            .find(|check| {
                check.operation.rotation == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    && check.operation.translation == [0.0; 3]
                    && check.operation.time_reversal
            })
            .expect("grey extension contains pure time reversal");
        let OperationHamiltonianStatus::Broken(residual) = &pure_time_reversal.status else {
            panic!("complex directed hopping must break pure time reversal");
        };
        assert!(residual.max_absolute > residual.acceptance_threshold);
        assert_eq!(residual.witness.lattice_vector[0].abs(), 1);
        assert_eq!(residual.witness.bra, 0);
        assert_eq!(residual.witness.ket, 0);
        let FinalMagneticGroup::Identified(group) = report.final_group else {
            panic!("the complex-hopping survivor should be a closed magnetic group");
        };
        assert_eq!(group.magnetic_type, MagneticGroupType::BlackWhite);
    }

    #[test]
    fn spinful_zeeman_term_is_classified_as_a_reduced_magnetic_group() {
        let mut model = spinful_cubic();
        model.set_hop(0.4_f64, 0, 0, &array![0_isize, 0, 0], SpinDirection::Z);
        let report = model
            .check_hamiltonian_symmetry(
                &ScalarSiteBasis::default(),
                &HamiltonianSymmetryRequest::default(),
            )
            .unwrap();
        assert_eq!(
            report.compatibility,
            HamiltonianCompatibility::SymmetryReduced
        );
        let FinalMagneticGroup::Identified(group) = report.final_group else {
            panic!("Zeeman survivor should be identified");
        };
        assert_eq!(group.magnetic_type, MagneticGroupType::BlackWhite);
    }

    #[test]
    fn unsupported_orbital_metadata_is_inconclusive_not_broken() {
        let mut model = scalar_cubic();
        model.orb_projection[0] = OrbProj::px;
        let report = model
            .check_hamiltonian_symmetry(
                &ScalarSiteBasis::default(),
                &HamiltonianSymmetryRequest::default(),
            )
            .unwrap();
        assert_eq!(report.compatibility, HamiltonianCompatibility::Inconclusive);
        assert_eq!(report.is_fully_compatible(), None);
        assert_eq!(
            report.completeness,
            HamiltonianSymmetryCompleteness::LowerBound
        );
        assert!(report.operation_checks.iter().all(|check| matches!(
            check.status,
            OperationHamiltonianStatus::Unresolved(BasisRepresentationError::Unsupported(_))
        )));
        assert!(matches!(
            report.final_group,
            FinalMagneticGroup::Inconclusive { .. }
        ));
    }

    #[test]
    fn nonhermitian_public_mutation_is_rejected_before_symmetry_testing() {
        let mut model = scalar_cubic();
        model.ham[[0, 0, 0]] = Complex64::new(0.0, 1.0);
        let error = model
            .check_hamiltonian_symmetry(
                &ScalarSiteBasis::default(),
                &HamiltonianSymmetryRequest::default(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::InvalidHamiltonianSymmetryInput {
                parameter: "hermiticity",
                ..
            }
        ));
    }

    #[test]
    fn half_translation_action_keeps_cross_cell_shift_sectors() {
        let model = half_translation_model(0.0, 0.0);
        let operation = identity_half_translation(false);
        let action = ScalarSiteBasis::default()
            .resolve(BasisActionContext {
                model: &model,
                operation: &operation,
                position_tolerance: 1e-8,
                representation_tolerance: 1e-8,
            })
            .unwrap();
        let action = validate_action(action, model.nsta(), 1e-8).unwrap();
        assert_eq!(
            action
                .sectors
                .iter()
                .map(|sector| sector.shift)
                .collect::<Vec<_>>(),
            vec![[0, 0, 0], [1, 0, 0]]
        );
        let sewing = action.lattice_gauge_matrix([0.25, 0.0, 0.0]).unwrap();
        assert!((sewing[[1, 0]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
        assert!((sewing[[0, 1]] - Complex64::new(0.0, -1.0)).norm() < 1e-12);
    }

    #[test]
    fn spinful_anti_half_translation_squares_to_minus_one_in_the_next_cell() {
        let model = spinful_half_translation_model();
        let operation = identity_half_translation(true);
        let action = ScalarSiteBasis
            .resolve(BasisActionContext {
                model: &model,
                operation: &operation,
                position_tolerance: 1e-8,
                representation_tolerance: 1e-8,
            })
            .and_then(|action| validate_action(action, model.nsta(), 1e-8))
            .unwrap();
        let square = compose_localized_actions(&operation, &action, &action).unwrap();
        assert!(
            square
                .iter()
                .filter(|(_, matrix)| matrix.iter().any(|value| value.norm() > 1e-12))
                .map(|(shift, _)| *shift)
                .eq([[1, 0, 0]])
        );
        let matrix = &square[&[1, 0, 0]];
        for row in 0..model.nsta() {
            for column in 0..model.nsta() {
                let expected = if row == column {
                    Complex64::new(-1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!((matrix[[row, column]] - expected).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn half_translation_is_checked_through_nonzero_cross_cell_hoppings() {
        for (intracell, intercell, expected_preserved) in [(1.0, 1.0, true), (1.0, 2.0, false)] {
            let report = half_translation_model(intracell, intercell)
                .check_hamiltonian_symmetry(
                    &ScalarSiteBasis,
                    &HamiltonianSymmetryRequest::default(),
                )
                .unwrap();
            let check = report
                .operation_checks
                .iter()
                .find(|check| {
                    operations_equivalent(&check.operation, &identity_half_translation(false), 1e-8)
                })
                .expect("the Atom structure contains its half translation");
            assert_eq!(
                matches!(check.status, OperationHamiltonianStatus::Preserved(_)),
                expected_preserved
            );
        }
    }

    #[test]
    fn orbital_cell_representatives_change_sectors_not_physical_symmetry() {
        let mut model = half_translation_model(0.0, 0.0);
        model.orb[[1, 0]] = 1.5;
        model.set_hop(1.0_f64, 0, 1, &array![-1_isize, 0, 0], None);
        model.set_hop(1.0_f64, 0, 1, &array![-2_isize, 0, 0], None);

        let operation = identity_half_translation(false);
        let action = ScalarSiteBasis
            .resolve(BasisActionContext {
                model: &model,
                operation: &operation,
                position_tolerance: 1e-8,
                representation_tolerance: 1e-8,
            })
            .unwrap();
        assert_eq!(
            action
                .sectors
                .iter()
                .map(|sector| sector.shift)
                .collect::<Vec<_>>(),
            vec![[-1, 0, 0], [2, 0, 0]]
        );
        let report = model
            .check_hamiltonian_symmetry(&ScalarSiteBasis, &HamiltonianSymmetryRequest::default())
            .unwrap();
        let check = report
            .operation_checks
            .iter()
            .find(|check| operations_equivalent(&check.operation, &operation, 1e-8))
            .unwrap();
        assert!(matches!(
            check.status,
            OperationHamiltonianStatus::Preserved(_)
        ));
    }

    #[test]
    fn magnetic_field_context_filters_the_grey_candidates_before_h_check() {
        let mut request = HamiltonianSymmetryRequest::default();
        request.structural_parameters.external_fields.magnetic = Some([0.0, 0.0, 1.0]);
        let report = scalar_cubic()
            .check_hamiltonian_symmetry(&ScalarSiteBasis::default(), &request)
            .unwrap();

        assert_eq!(report.structure_candidates.len(), 96);
        assert_eq!(report.field_allowed_operations.len(), 16);
        assert_eq!(report.surviving_operations.len(), 16);
        assert_eq!(report.compatibility, HamiltonianCompatibility::Compatible);
        let FinalMagneticGroup::Identified(group) = report.final_group else {
            panic!("field-stabilizer subgroup should be identified");
        };
        assert_eq!(group.magnetic_type, MagneticGroupType::BlackWhite);
        assert_eq!(group.subgroup_index_in_candidates, 6);
    }

    #[test]
    fn malformed_public_sewing_action_returns_error_instead_of_panicking() {
        let action = LocalizedBasisAction {
            sectors: vec![CellShiftAction {
                shift: [0, 0, 0],
                matrix: Array2::zeros((2, 3)),
            }],
        };
        assert!(action.lattice_gauge_matrix([0.0; 3]).is_err());
    }

    #[test]
    fn scalar_projection_is_the_l_zero_quantum_state() {
        let state = OrbProj::s.to_quantum_number().unwrap();
        assert_eq!(state[0], Complex64::new(1.0, 0.0));
        assert!(
            state
                .iter()
                .skip(1)
                .all(|coefficient| coefficient.norm() == 0.0)
        );
    }

    #[test]
    fn automatic_p_shell_representation_has_the_expected_c4z_action() {
        let model = atomic_orbital_cubic::<false>(&[OrbProj::px, OrbProj::py, OrbProj::pz]);
        let action =
            validate_action(resolve_atomic(&model, &c4z(false)).unwrap(), 3, 1e-8).unwrap();
        assert_eq!(action.sectors.len(), 1);
        assert_eq!(action.sectors[0].shift, [0, 0, 0]);
        let matrix = &action.sectors[0].matrix;
        let expected = array![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            .mapv(|value| Complex64::new(value, 0.0));
        assert!(matrix_max_difference(matrix, &expected) < 1e-12);
    }

    #[test]
    fn automatic_d_shell_representation_uses_real_wannier90_orbitals() {
        let model = atomic_orbital_cubic::<false>(&[
            OrbProj::dxy,
            OrbProj::dyz,
            OrbProj::dxz,
            OrbProj::dz2,
            OrbProj::dx2y2,
        ]);
        let action =
            validate_action(resolve_atomic(&model, &c4z(false)).unwrap(), 5, 1e-8).unwrap();
        let expected = array![
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -1.0],
        ]
        .mapv(|value| Complex64::new(value, 0.0));
        assert!(matrix_max_difference(&action.sectors[0].matrix, &expected) < 1e-12);
    }

    #[test]
    fn automatic_hybrid_sp3_basis_is_closed_and_unitary() {
        let model = atomic_orbital_cubic::<false>(&[
            OrbProj::sp3_1,
            OrbProj::sp3_2,
            OrbProj::sp3_3,
            OrbProj::sp3_4,
        ]);
        let action = resolve_atomic(&model, &c4z(false)).unwrap();
        validate_action(action, model.nsta(), 1e-8).unwrap();
    }

    #[test]
    fn automatic_angular_rotation_has_correct_inversion_parity_through_f() {
        let inversion = real_orbital_rotation(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            1e-10,
        )
        .unwrap();
        for angular_momentum in 0..=3 {
            let expected = if angular_momentum % 2 == 0 {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(-1.0, 0.0)
            };
            let start = angular_momentum * angular_momentum;
            let end = (angular_momentum + 1) * (angular_momentum + 1);
            for row in start..end {
                for column in start..end {
                    let target = if row == column {
                        expected
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    assert!((inversion[[row, column]] - target).norm() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn automatic_basis_rejects_incomplete_and_duplicate_angular_shells() {
        let incomplete = atomic_orbital_cubic::<false>(&[OrbProj::px]);
        assert!(matches!(
            resolve_atomic(&incomplete, &c4z(false)),
            Err(BasisRepresentationError::Unsupported(_))
        ));

        let duplicate = atomic_orbital_cubic::<false>(&[OrbProj::px, OrbProj::px]);
        assert!(matches!(
            resolve_atomic(&duplicate, &identity_operation(false)),
            Err(BasisRepresentationError::Ambiguous(_))
        ));
    }

    #[test]
    fn automatic_spinful_time_reversal_tensors_orbitals_with_i_sigma_y() {
        let model = atomic_orbital_cubic::<true>(&[OrbProj::px, OrbProj::py, OrbProj::pz]);
        let operation = identity_operation(true);
        let action = validate_action(resolve_atomic(&model, &operation).unwrap(), 6, 1e-8).unwrap();
        let matrix = &action.sectors[0].matrix;
        for orbital in 0..3 {
            assert!((matrix[[orbital, 3 + orbital]] - Complex64::new(1.0, 0.0)).norm() < 1e-12);
            assert!((matrix[[3 + orbital, orbital]] + Complex64::new(1.0, 0.0)).norm() < 1e-12);
        }
    }

    #[test]
    fn automatic_basis_tracks_orbital_cell_representatives_by_sector() {
        let mut model = atomic_orbital_cubic::<false>(&[OrbProj::px, OrbProj::py, OrbProj::pz]);
        model.orb.row_mut(0).assign(&array![1.0, 0.0, 0.0]);
        model.orb.row_mut(1).assign(&array![0.0, 1.0, 0.0]);
        let action = resolve_atomic(&model, &c4z(false)).unwrap();
        let action = validate_action(action, model.nsta(), 1e-8).unwrap();
        validate_action_geometry(&model, &c4z(false), &action, 1e-8, 1e-8).unwrap();
        assert!(
            action
                .sectors
                .iter()
                .any(|sector| sector.shift == [0, 0, 0])
        );
        assert!(
            action
                .sectors
                .iter()
                .any(|sector| sector.shift == [-2, 0, 0])
        );
    }

    #[test]
    fn automatic_p_shell_certifies_the_full_cubic_grey_group() {
        let model = atomic_orbital_cubic::<false>(&[OrbProj::px, OrbProj::py, OrbProj::pz]);
        let report = model
            .check_hamiltonian_symmetry(&AtomicOrbitalBasis, &HamiltonianSymmetryRequest::default())
            .unwrap();
        assert_eq!(report.structure_candidates.len(), 96);
        assert_eq!(report.surviving_operations.len(), 96);
        assert_eq!(report.compatibility, HamiltonianCompatibility::Compatible);
        assert_eq!(
            report.completeness,
            HamiltonianSymmetryCompleteness::Complete
        );
    }

    #[test]
    fn automatic_complete_s_p_d_f_basis_certifies_cubic_corepresentation() {
        let model = atomic_orbital_cubic::<false>(&PURE_REAL_ORBITALS);
        let report = model
            .check_hamiltonian_symmetry(&AtomicOrbitalBasis, &HamiltonianSymmetryRequest::default())
            .unwrap();
        assert_eq!(report.surviving_operations.len(), 96);
        assert_eq!(report.compatibility, HamiltonianCompatibility::Compatible);
        assert_eq!(
            report.completeness,
            HamiltonianSymmetryCompleteness::Complete
        );
    }

    #[test]
    fn orbital_only_model_is_rejected_at_the_hamiltonian_symmetry_boundary() {
        let model: Model<false, 3, NoRMatrix> =
            Model::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None).unwrap();
        let error = model
            .check_hamiltonian_symmetry(&ScalarSiteBasis, &HamiltonianSymmetryRequest::default())
            .unwrap_err();
        assert!(matches!(error, TbError::MissingAtomicStructure));
    }

    #[test]
    fn forced_cubic_symmetrization_averages_axes_and_is_idempotent() {
        let mut model = scalar_cubic();
        model.set_hop(1.0_f64, 0, 0, &array![1_isize, 0, 0], None);
        model.set_hop(2.0_f64, 0, 0, &array![0_isize, 1, 0], None);
        model.set_hop(3.0_f64, 0, 0, &array![0_isize, 0, 1], None);
        let original = model.clone();
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();

        let symmetrized = model
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();

        assert_eq!(hopping(&model, [1, 0, 0]), Complex64::new(1.0, 0.0));
        assert_eq!(
            model.ham, original.ham,
            "the input Model must not be mutated"
        );
        assert_eq!(symmetrized.hamR.row(0).to_vec(), vec![0, 0, 0]);
        for lattice_vector in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
            assert!((hopping(&symmetrized, lattice_vector).re - 2.0).abs() < 1e-10);
            assert!(hopping(&symmetrized, lattice_vector).im.abs() < 1e-12);
        }
        let report = symmetrized
            .check_hamiltonian_symmetry(&ScalarSiteBasis, &HamiltonianSymmetryRequest::default())
            .unwrap();
        assert_eq!(report.is_fully_compatible(), Some(true));

        let twice = symmetrized
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        assert_eq!(twice.hamR, symmetrized.hamR);
        assert!(
            twice
                .ham
                .iter()
                .zip(symmetrized.ham.iter())
                .all(|(left, right)| (*left - *right).norm() < 1e-12)
        );
    }

    #[test]
    fn forced_grey_symmetrization_removes_time_reversal_breaking_terms() {
        let mut spinless = scalar_cubic();
        spinless.set_hop(
            Complex64::new(1.0, 0.25),
            0,
            0,
            &array![1_isize, 0, 0],
            None,
        );
        let target = spinless
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let spinless = spinless
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        assert!(spinless.ham.iter().all(|value| value.im.abs() < 1e-12));

        let mut spinful = spinful_cubic();
        spinful.set_hop(0.4_f64, 0, 0, &array![0_isize, 0, 0], SpinDirection::Z);
        let target = spinful
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let spinful = spinful
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        assert!(spinful.ham.iter().all(|value| value.norm() < 1e-12));
    }

    #[test]
    fn staggered_zeeman_order_is_identified_as_type_iv() {
        let mut model = spinful_half_translation_model();
        model.set_hop(0.4_f64, 0, 0, &array![0_isize, 0, 0], SpinDirection::Z);
        model.set_hop(-0.4_f64, 1, 1, &array![0_isize, 0, 0], SpinDirection::Z);
        let report = model
            .check_hamiltonian_symmetry(&ScalarSiteBasis, &HamiltonianSymmetryRequest::default())
            .unwrap();

        let status = |translation: [f64; 3], time_reversal: bool| {
            &report
                .operation_checks
                .iter()
                .find(|check| {
                    operations_equivalent(
                        &check.operation,
                        &CrystalSymmetryOperation {
                            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            translation,
                            time_reversal,
                        },
                        1e-8,
                    )
                })
                .expect("expected translation operation")
                .status
        };
        assert!(matches!(
            status([0.0, 0.0, 0.0], true),
            OperationHamiltonianStatus::Broken(_)
        ));
        assert!(matches!(
            status([0.5, 0.0, 0.0], false),
            OperationHamiltonianStatus::Broken(_)
        ));
        assert!(matches!(
            status([0.5, 0.0, 0.0], true),
            OperationHamiltonianStatus::Preserved(_)
        ));
        let FinalMagneticGroup::Identified(group) = report.final_group else {
            panic!("the closed staggered survivor must be identified");
        };
        assert_eq!(group.magnetic_type, MagneticGroupType::AntiTranslation);
    }

    #[test]
    fn symmetrization_adds_missing_support_and_preserves_rmatrix_alignment() {
        let mut model: Model<false, 3, HasRMatrix> = Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.set_hop(3.0_f64, 0, 0, &array![1_isize, 0, 0], None);
        let mut rmatrix = Array4::zeros((model.hamR.nrows(), 3, 1, 1));
        for index in 0..model.hamR.nrows() {
            if (0..3).all(|axis| model.hamR[[index, axis]] == [1, 0, 0][axis]) {
                rmatrix[[index, 0, 0, 0]] = Complex64::new(7.0, 0.0);
            }
        }
        model.rmatrix = HasRMatrix(rmatrix);
        model.validate().unwrap();
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let symmetrized = model
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();

        assert_eq!(symmetrized.hamR.row(0).to_vec(), vec![0, 0, 0]);
        assert_eq!(
            symmetrized.rmatrix.as_array4().len_of(Axis(0)),
            symmetrized.hamR.nrows()
        );
        for lattice_vector in [[1, 0, 0], [0, 1, 0], [0, 0, 1]] {
            let index = (0..symmetrized.hamR.nrows())
                .find(|&index| {
                    (0..3).all(|axis| symmetrized.hamR[[index, axis]] == lattice_vector[axis])
                })
                .expect("cubic projection must add all axis-related blocks");
            assert!((symmetrized.ham[[index, 0, 0]].re - 1.0).abs() < 1e-10);
            let expected_r = if lattice_vector == [1, 0, 0] {
                7.0
            } else {
                0.0
            };
            assert_eq!(symmetrized.rmatrix[[index, 0, 0, 0]].re, expected_r);
        }
        symmetrized.validate().unwrap();
    }

    #[test]
    fn incompatible_lattice_is_rejected_before_basis_resolution() {
        let target = scalar_cubic()
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let model: Model<false, 3, NoRMatrix> = Model::tb_model(
            array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Si,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();

        let error = model
            .symmetrize_hamiltonian(
                &target,
                &MustNotResolve,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::TargetMagneticGroupIncompatible { .. }
        ));
        assert!(error.to_string().contains("not compatible"));
    }

    #[test]
    fn incompatible_atom_types_are_rejected_before_basis_resolution() {
        let target = half_translation_model(0.0, 0.0)
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let mut model = half_translation_model(0.0, 0.0);
        model.atoms[1].change_type(AtomType::He);
        let error = model
            .symmetrize_hamiltonian(
                &target,
                &MustNotResolve,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::TargetMagneticGroupIncompatible { .. }
        ));
        assert!(error.to_string().contains("not compatible"));
    }

    #[test]
    fn optional_atom_moments_are_part_of_target_compatibility() {
        let mut model = half_translation_model(1.0, 1.0);
        let nonmagnetic_target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 1.0]).unwrap();
        model.atoms[1]
            .set_magnetic_moment([0.0, 0.0, -1.0])
            .unwrap();

        let error = model
            .symmetrize_hamiltonian(
                &nonmagnetic_target,
                &MustNotResolve,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::TargetMagneticGroupIncompatible { .. }
        ));
        assert!(error.to_string().contains("optional Atom magnetic moments"));
    }

    #[test]
    fn magnetic_target_detected_from_the_same_atom_moments_is_accepted() {
        let mut model = spinful_half_translation_model();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 1.0]).unwrap();
        model.atoms[1]
            .set_magnetic_moment([0.0, 0.0, -1.0])
            .unwrap();
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        assert_eq!(target.magnetic_type, MagneticGroupType::AntiTranslation);
        let symmetrized = model
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap();
        symmetrized.validate().unwrap();
    }

    #[test]
    fn external_field_incompatibility_is_rejected_before_basis_resolution() {
        let model = scalar_cubic();
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let mut parameters = HamiltonianSymmetrizationParameters::default();
        parameters.structural_parameters.external_fields.magnetic = Some([0.0, 0.0, 1.0]);
        let error = model
            .symmetrize_hamiltonian(&target, &MustNotResolve, &parameters)
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::TargetMagneticGroupIncompatible { .. }
        ));
        assert!(error.to_string().contains("external-field context"));
    }

    #[test]
    fn unsupported_basis_is_a_hard_error_for_forced_symmetrization() {
        let mut model = scalar_cubic();
        model.orb_projection[0] = OrbProj::px;
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let error = model
            .symmetrize_hamiltonian(
                &target,
                &ScalarSiteBasis,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::TargetMagneticGroupIncompatible { .. }
        ));
        assert!(
            error
                .to_string()
                .contains("no compatible localized-basis action")
        );
    }

    #[test]
    fn individually_unitary_but_inconsistent_corepresentation_is_rejected() {
        let model = spinful_cubic();
        let target = model
            .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())
            .unwrap();
        let error = model
            .symmetrize_hamiltonian(
                &target,
                &InvalidIdentityCorepresentation,
                &HamiltonianSymmetrizationParameters::default(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            TbError::TargetMagneticGroupIncompatible { .. }
        ));
        assert!(error.to_string().contains("identity action"));
    }

    #[test]
    fn valid_identity_but_bad_pairwise_projective_composition_is_rejected() {
        let identity = CrystalSymmetryOperation {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0; 3],
            time_reversal: false,
        };
        let c2z = CrystalSymmetryOperation {
            rotation: [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            translation: [0.0; 3],
            time_reversal: false,
        };
        let identity_action = LocalizedBasisAction {
            sectors: vec![CellShiftAction {
                shift: [0, 0, 0],
                matrix: Array2::eye(2),
            }],
        };
        let bad_c2_action = LocalizedBasisAction {
            sectors: vec![CellShiftAction {
                shift: [0, 0, 0],
                matrix: array![
                    [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                    [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)]
                ],
            }],
        };
        let error = validate_projective_corepresentation(
            &[(identity, identity_action), (c2z, bad_c2_action)],
            1e-8,
            1e-8,
        )
        .unwrap_err();
        assert!(error.to_string().contains("projective group composition"));
    }

    #[test]
    fn spinful_time_reversal_sewing_is_i_sigma_y() {
        let model = spinful_cubic();
        let operation = CrystalSymmetryOperation {
            rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation: [0.0; 3],
            time_reversal: true,
        };
        let action = ScalarSiteBasis
            .resolve(BasisActionContext {
                model: &model,
                operation: &operation,
                position_tolerance: 1e-8,
                representation_tolerance: 1e-8,
            })
            .unwrap();
        let sewing = action.lattice_gauge_matrix([0.37, -0.11, 0.29]).unwrap();
        let expected = array![
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(-1.0, 0.0), Complex64::new(0.0, 0.0)]
        ];
        assert!(
            sewing
                .iter()
                .zip(expected.iter())
                .all(|(left, right)| (*left - *right).norm() < 1e-12)
        );
    }
}
