//! src/error.rs
//! This module defines the custom error types for the entire tight-binding library.
//! By using a centralized error enum, we can replace all panics with recoverable
//! Results, making the library safer and more robust for consumers.

use ndarray::Array1;
use num_complex::Complex;
use thiserror::Error;
use crate::SpinDirection;

/// The primary error type for all fallible operations in this library.
#[derive(Error, Debug)]
pub enum TbError {
    // --- I/O and Parsing Errors ---
    #[error("I/O error")]
    Io(#[from] std::io::Error),

    #[error("Failed to parse data from file '{file}': {message}")]
    FileParse { file: String, message: String },

    #[error("Invalid orbital projection string: '{0}'")]
    InvalidOrbitalProjection(String),

    #[error("Invalid atom type string: '{0}'")]
    InvalidAtomType(String),

    // --- Linear Algebra and Numerical Errors ---
    #[error("Linear algebra operation failed")]
    Linalg(#[from] ndarray_linalg::error::LinalgError),

    #[error("LAPACK routine '{routine}' failed with non-zero info code: {info}")]
    Lapack {
        routine: &'static str,
        info: i32,
    },

    // --- Invalid Input and Arguments ---
    #[error("Dimension mismatch for '{context}': expected {expected}, got {found}")]
    DimensionMismatch {
        context: String,
        expected: usize,
        found: usize,
    },

    #[error("The path for the Wilson Loop is not closed. Remainder vector: {0:?}")]
    UnclosedWilsonLoop(Array1<f64>),

    #[error("The provided direction index '{index}' is out of bounds for dimension '{dim}'")]
    InvalidDirectionIndex { index: usize, dim: usize },

    #[error("Invalid supercell size 'num' for cut_piece: {0}. Must be >= 1.")]
    InvalidSupercellSize(usize),
    
    #[error("Invalid shape identifier for cut_dot: {0}. Supported shapes are 3, 4, 6, 8.")]
    InvalidShapeIdentifier(usize),

    #[error("The supercell transformation matrix U must have integer elements and a non-zero determinant.")]
    InvalidSupercellMatrix,
    
    #[error("Spin direction '{0:?}' is invalid for a model without spin.")]
    SpinNotAllowed(SpinDirection),

    // --- Model Consistency and Physics Errors ---
    #[error("On-site hopping energy must be a real number, but got {0}")]
    OnsiteHoppingMustBeReal(Complex<f64>),

    #[error("Internal model inconsistency: Hopping for vector R={r:?} exists, but its Hermitian conjugate for -R does not.")]
    MissingHermitianConjugateHopping { r: Array1<isize> },
    
    #[error("Invalid operation for a zero-dimensional model.")]
    InvalidOperationForZeroDimension,

    // --- Feature Not Implemented ---
    #[error("Hybrid orbital projection '{0}' is not currently supported for this operation")]
    HybridOrbitalNotSupported(String),
}

/// A specialized `Result` type for this library's operations.
pub type Result<T> = std::result::Result<T, TbError>;
