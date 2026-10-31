use serde::{Deserialize, Serialize};
/// Gauge choice for the Bloch basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum Gauge {
    Lattice,
    Atom,
}

/// System dimensionality.
///
/// Variant names (`one`, `two`, `three`) are kept for backward compatibility
/// instead of CamelCase renames.
#[repr(u8)]
#[allow(non_camel_case_types)]
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
