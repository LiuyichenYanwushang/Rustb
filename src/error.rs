//! src/error.rs
//! This module defines the custom error types for the entire tight-binding library.
//! By using a centralized error enum, we can replace all panics with recoverable
//! Results, making the library safer and more robust for consumers.

use crate::atom_struct::AtomType;
use crate::{OrbProj, SpinDirection};
use ndarray::{Array1, Array2, ShapeError};
use num_complex::Complex;
use thiserror::Error;

/// The primary error type for all fallible operations in this library.
#[derive(Error, Debug)]
pub enum TbError {
    #[cfg(feature = "cryspglib")]
    #[error("cryspglib symmetry analysis failed: {0}")]
    Cryspglib(#[from] cryspglib::SymError),

    #[cfg(feature = "cryspglib")]
    #[error("invalid crystal-symmetry input '{parameter}': {message}")]
    InvalidCrystalSymmetryInput {
        parameter: &'static str,
        message: String,
    },

    #[cfg(feature = "cryspglib")]
    #[error(
        "high-symmetry data uses Hall setting {data_hall}, but the detected structure uses Hall setting {detected_hall}"
    )]
    UnsupportedHighSymmetrySetting {
        detected_hall: usize,
        data_hall: usize,
    },

    #[cfg(feature = "cryspglib")]
    #[error(
        "external fields reduce the symmetry from {structural_operations} to {effective_operations} operations; canonical high-symmetry points and character tables for the structural group are not valid for this effective symmetry"
    )]
    FieldReducedSymmetryData {
        structural_operations: usize,
        effective_operations: usize,
    },

    #[cfg(feature = "cryspglib")]
    #[error("magnetic irrep/corep analysis failed for UNI {uni}: {message}")]
    MagneticIrrepAnalysis { uni: usize, message: String },

    #[cfg(feature = "cryspglib")]
    #[error("invalid Hamiltonian-symmetry input '{parameter}': {message}")]
    InvalidHamiltonianSymmetryInput {
        parameter: &'static str,
        message: String,
    },

    #[cfg(feature = "cryspglib")]
    #[error("target magnetic group is incompatible with this Model: {reason}")]
    TargetMagneticGroupIncompatible { reason: String },

    // --- I/O and Parsing Errors ---
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse data from file '{file}': {message}")]
    FileParse { file: String, message: String },

    #[error("Invalid orbital projection string: '{0}'")]
    InvalidOrbitalProjection(String),

    #[error("Invalid atom type string: '{0}'")]
    InvalidAtomType(String),

    #[error("atomic magnetic moment must have finite Cartesian components, got {moment:?}")]
    InvalidAtomicMagneticMoment { moment: [f64; 3] },

    #[error("Failed to create directory '{path}': {message}")]
    DirectoryCreation { path: String, message: String },

    #[error("Failed to create file '{path}': {message}")]
    FileCreation { path: String, message: String },

    // --- Linear Algebra and Numerical Errors ---
    #[error("Linear algebra operation failed: {0}")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    #[error("LAPACK routine '{routine}' failed with non-zero info code: {info}")]
    Lapack { routine: &'static str, info: i32 },

    #[error("Matrix inversion failed: matrix is singular or ill-conditioned")]
    MatrixInversionFailed,

    #[error("Eigenvalue computation failed")]
    EigenvalueComputationFailed,

    #[error("SVD computation failed")]
    SvdComputationFailed,

    // --- Invalid Input and Arguments ---
    #[error("Dimension mismatch for '{context}': expected {expected}, got {found}")]
    DimensionMismatch {
        context: String,
        expected: usize,
        found: usize,
    },

    #[error("Invalid array shape: expected {expected:?}, got {found:?}")]
    InvalidArrayShape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("The path for the Wilson Loop is not closed. Remainder vector: {0:?}")]
    UnclosedWilsonLoop(Array1<f64>),

    #[error("The provided direction index '{index}' is out of bounds for dimension '{dim}'")]
    InvalidDirectionIndex { index: usize, dim: usize },

    #[error("Invalid supercell size 'num' for cut_piece: {0}. Must be >= 1.")]
    InvalidSupercellSize(usize),

    #[error("Invalid shape identifier for cut_dot: {0}. Supported shapes are 3, 4, 6, 8.")]
    InvalidShapeIdentifier(usize),

    #[error(
        "The supercell transformation matrix U must have integer elements and a non-zero determinant."
    )]
    InvalidSupercellMatrix,

    #[error("Spin direction '{0:?}' is invalid for a model without spin.")]
    SpinNotAllowed(SpinDirection),

    #[error("Invalid k-point mesh dimensions: {0:?}")]
    InvalidKmeshDimensions(Array1<usize>),

    #[error("Invalid energy range: min={min}, max={max}")]
    InvalidEnergyRange { min: f64, max: f64 },

    // --- Model Consistency and Physics Errors ---
    #[error("On-site hopping energy must be a real number, but got {0}")]
    OnsiteHoppingMustBeReal(Complex<f64>),

    #[error(
        "Internal model inconsistency: Hopping for vector R={r:?} exists, but its Hermitian conjugate for -R does not."
    )]
    MissingHermitianConjugateHopping { r: Array1<isize> },

    #[error("Invalid operation for a zero-dimensional model.")]
    InvalidOperationForZeroDimension,

    #[error("Model has not been properly initialized")]
    ModelNotInitialized,

    #[error("No bands found in the specified energy range")]
    NoBandsInEnergyRange,

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceFailed { iterations: usize },

    #[error(
        "Hubbard mean-field self-consistency failed after {iterations} iterations (density residual {residual:e})"
    )]
    MeanFieldNotConverged { iterations: usize, residual: f64 },

    #[error("Invalid Hubbard mean-field parameter '{parameter}': {message}")]
    InvalidMeanFieldParameter {
        parameter: &'static str,
        message: String,
    },

    #[error("Invalid thermodynamic parameter '{parameter}': {message}")]
    InvalidThermodynamicParameter {
        parameter: &'static str,
        message: String,
    },

    #[error("Invalid response parameter '{parameter}': {message}")]
    InvalidResponseParameter {
        parameter: &'static str,
        message: String,
    },

    #[error("Invalid DOS parameter '{parameter}': {message}")]
    InvalidDosParameter {
        parameter: &'static str,
        message: String,
    },

    // --- Slater-Koster Specific Errors ---
    #[error(
        "Missing Slater-Koster parameter '{param}' for atom pair {atom1:?}-{atom2:?} at shell {shell}"
    )]
    SkParameterMissing {
        param: String,
        atom1: AtomType,
        atom2: AtomType,
        shell: usize,
    },

    #[error("Unsupported orbital combination: {0:?} - {1:?}")]
    UnsupportedOrbitalCombination(OrbProj, OrbProj),

    #[error("Invalid neighbor search range: {0}")]
    InvalidSearchRange(i32),

    #[error("No neighbor shells found")]
    NoShellsFound,

    // --- Feature Not Implemented ---
    #[error("Hybrid orbital projection '{0}' is not currently supported for this operation")]
    HybridOrbitalNotSupported(String),

    #[error("Feature '{0}' is not yet implemented")]
    NotImplemented(String),

    // --- New error variants to replace panics ---
    #[error(
        "Lattice matrix dimension error: second dimension length must equal dim_r, but got {actual} (expected {expected})"
    )]
    LatticeDimensionError { expected: usize, actual: usize },

    #[error("R vector length error: expected {expected}, got {actual}")]
    RVectorLengthError { expected: usize, actual: usize },

    #[error("Invalid k-path operation for zero-dimensional model")]
    ZeroDimKPathError,

    #[error("Path length mismatch: expected {expected}, got {actual}")]
    PathLengthMismatch { expected: usize, actual: usize },

    #[error("Invalid direction index: {index} for dimension {dim}")]
    InvalidDirection { index: usize, dim: usize },

    #[error("Invalid shape: {shape}. Supported shapes: {supported:?}")]
    InvalidShape { shape: usize, supported: Vec<usize> },

    #[error("Invalid dimension for operation: {dim}. Supported dimensions: {supported:?}")]
    InvalidDimension { dim: usize, supported: Vec<usize> },

    #[error("Duplicate orbitals found in orbital list")]
    DuplicateOrbitals,

    #[error("Duplicate atoms found in atom list")]
    DuplicateAtoms,

    #[error("atom reorder list must be a permutation of 0..{natom}, got {order:?}")]
    InvalidAtomPermutation { natom: usize, order: Vec<usize> },

    #[error("Invalid supercell transformation matrix determinant: {det}")]
    InvalidSupercellDet { det: f64 },

    #[error("Invalid atom positions or count in unit cell")]
    InvalidAtomConfiguration,

    #[error("atom index {index} is out of bounds for {natom} atoms")]
    InvalidAtomId { index: usize, natom: usize },

    #[error("orbital index {index} is out of bounds for {norb} orbitals")]
    InvalidOrbitalId { index: usize, norb: usize },

    #[error("orbital {orbital} is assigned to both atom {first_atom} and atom {second_atom}")]
    DuplicateOrbitalOwner {
        orbital: usize,
        first_atom: usize,
        second_atom: usize,
    },

    #[error("invalid model invariant '{invariant}': {message}")]
    InvalidModelInvariant {
        invariant: &'static str,
        message: String,
    },

    #[error("this operation requires explicit atomic-site metadata")]
    MissingAtomicStructure,

    #[error("a model must contain at least one orbital")]
    NoOrbitals,

    #[error("Transformation matrix dimension mismatch: expected {expected}, got {actual}")]
    TransformationMatrixDimMismatch { expected: usize, actual: usize },

    #[error("Missing Hermitian conjugate for R vector: {r:?}")]
    MissingHermitianConjugate { r: Array1<isize> },

    #[error("Invalid spin value: {spin}. Supported values: {supported:?}")]
    InvalidSpinValue { spin: usize, supported: Vec<usize> },

    #[error("Invalid k-vector length: expected {expected}, got {actual}")]
    KVectorLengthMismatch { expected: usize, actual: usize },

    #[error("LAPACK eigenvalue computation failed with info code: {info}")]
    LapackEigenFailed { info: i32 },

    // --- Generic Error ---
    #[error("Unexpected error: {0}")]
    Other(String),
}

/// A specialized `Result` type for this library's operations.
pub type Result<T> = std::result::Result<T, TbError>;
