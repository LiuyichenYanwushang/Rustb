//! Enum definitions for tight-binding models

use serde::{Deserialize, Serialize};

/// Gauge choice for Bloch wavefunctions
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Gauge {
    /// Lattice gauge: $\ket{\phi_{n\bm k}} = \sum_{\bm R} e^{i\bm k\cdot\bm R}\ket{n\bm R}$
    Lattice = 0,
    /// Atomic gauge: $\ket{u_{n\bm k}} = \sum_{\bm R} e^{i\bm k\cdot(\bm R+\bm\tau_n)}\ket{n\bm R}$
    Atom = 1,
}

/// Real-space dimensionality of the model
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum Dimension {
    zero = 0,
    one = 1,
    two = 2,
    three = 3,
}

/// Spin direction for Pauli matrices
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum SpinDirection {
    /// Identity matrix $\sigma_0$
    None = 0,
    /// Pauli matrix $\sigma_x$
    x = 1,
    /// Pauli matrix $\sigma_y$
    y = 2,
    /// Pauli matrix $\sigma_z$
    z = 3,
}
