//! Data structures for simplex quadrature and energy‑cut integration.
//!
//! ## Core types
//!
//! `VertexKernel` stores gauge‑invariant per‑k‑point primitives suitable
//! for linear interpolation inside simplices:
//!
//! | Field | Formula | Invariant? |
//! |-------|---------|-----------|
//! | `band[n]` | $E_n$ | ✓ |
//! | `k_ab[n,m]` | $K^{ab}_{nm}=v^a_{nm}v^b_{mn}$ | ✓ |
//! | `k_bc[n,m]` | $K^{bc}_{nm}=v^b_{nm}v^c_{mn}$ | ✓ |
//! | `k_ac[n,m]` | $K^{ac}_{nm}=v^a_{nm}v^c_{mn}$ | ✓ |
//! | `vdiag[n]` | $v^c_n=\partial_c E_n$ | ✓ |
//! | `vdiag_a[n]` | $v^a_n=\partial_a E_n$ | ✓ |
//! | `vdiag_b[n]` | $v^b_n=\partial_b E_n$ | ✓ |
//!
//! All are invariant under independent U(1) rotations $|u_n\rangle\to
//! e^{i\phi_n}|u_n\rangle$ of individual band eigenstates.
//!
//! ## Safety threshold
//!
//! `SIMPLEX_GAP_TOL = 10^{-4}` eV — simplexes with a band gap smaller
//! than this are flagged as potentially unsafe for single‑band evaluation.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

/// Per‑k‑point gauge‑invariant primitives for energy‑cut integration.
///
/// Fields dependent on `dir_c` are `Option`: `None` when only two
/// directions were requested (e.g. Berry curvature).  Callers must
/// handle the absence explicitly rather than receiving silent zeros.
#[derive(Clone)]
pub struct VertexKernel {
    /// Band energies $E_n$, length `nsta`.
    pub band: Array1<f64>,
    /// $K^{ab}_{nm}=v^a_{nm}v^b_{mn}$, shape `(nsta, nsta)` (always computed).
    pub k_ab: Array2<Complex<f64>>,
    /// $K^{bc}_{nm}=v^b_{nm}v^c_{mn}$ — `None` if `dir_c` was not supplied.
    pub k_bc: Option<Array2<Complex<f64>>>,
    /// $K^{ac}_{nm}=v^a_{nm}v^c_{mn}$ — `None` if `dir_c` was not supplied.
    pub k_ac: Option<Array2<Complex<f64>>>,
    /// Diagonal velocity $v^c_n$ — `None` if `dir_c` was not supplied.
    pub vdiag: Option<Array1<f64>>,
    /// Diagonal velocity $v^a_n$ — `None` if `dir_c` was not supplied.
    pub vdiag_a: Option<Array1<f64>>,
    /// Diagonal velocity $v^b_n$ — `None` if `dir_c` was not supplied.
    pub vdiag_b: Option<Array1<f64>>,
    /// Eigenvectors $U[:, n]$, shape `(norb, nsta)` — for band tracking.
    pub evec: Array2<Complex<f64>>,
}

/// A simplex whose vertices have been band‑tracked (label‑aligned).
pub struct TrackedSimplex {
    /// `d + 1` vertices, already label‑aligned.
    pub vertices: Vec<VertexKernel>,
    /// Physical volume of this simplex (fractional coordinates).
    pub volume: f64,
    /// Fractional coordinates of each vertex, shape `(d+1, dim)`.
    pub coords: Array2<f64>,
    /// Diagnostic counters for this simplex.
    pub diag: SimplexDiagnostics,
}

/// Per‑simplex safety / quality diagnostics.
#[derive(Clone, Copy, Default)]
pub struct SimplexDiagnostics {
    /// Minimum band gap $\min_{n\neq m} |E_n - E_m|$ across all vertices.
    pub min_gap: f64,
    /// Minimum assignment overlap from band tracking (1.0 = perfect).
    pub min_assignment_overlap: f64,
    pub tracking_conflict: bool,
}

/// Zero‑clone simplex referencing `all_pts` vertex data by borrowed pointer.
///
/// `NV` = 3 for triangle, 4 for tetrahedron.
/// `coords` uses `[f64; 3]` for each vertex (pad z=0 for 2D).
/// No allocation on construction — just indices/coords/volume.
pub struct TrackedSimplexRef<'a, const NV: usize> {
    pub vertices: [&'a VertexKernel; NV],
    pub coords: [[f64; 3]; NV],
    pub volume: f64,
    pub diag: SimplexDiagnostics,
}

pub const SIMPLEX_GAP_TOL: f64 = 1e-4;
