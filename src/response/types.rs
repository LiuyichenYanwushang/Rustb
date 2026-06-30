//! Data structures for simplex quadrature.
//!
//! ## Core types
//!
//! `VertexKernel` stores gauge‑invariant per‑k‑point primitives:
//! band energies $E_n$, eigenvectors $U$, the velocity kernel
//! $K^{ab}_{nm}=v^a_{nm}v^b_{mn}$, and optionally $v^c_n = \partial_c E_n$.
//!
//! `TrackedSimplex` bundles band‑aligned vertices with their geometry
//! (barycentric coords, volume) and diagnostic information.
//!
//! ## Safety threshold
//!
//! `SIMPLEX_GAP_TOL = 10^{-4}` eV — simplexes with a band gap smaller
//! than this are flagged as potentially unsafe for single‑band evaluation
//! of Berry/QGT quantities.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

/// Per‑k‑point primitive data for one direction pair `(a,b)`.
///
/// `k_ab[n,m] = v^a_nm · v^b_mn` is gauge‑invariant under independent
/// U(1) rotations of bands n and m (when both bands are isolated).
#[derive(Clone)]
pub struct VertexKernel {
    /// Band energies `ε_n`, length `nsta`.
    pub band: Array1<f64>,
    /// `K^{ab}_{nm}`, shape `(nsta, nsta)`.
    pub k_ab: Array2<Complex<f64>>,
    /// Diagonal velocity `v^c_n = ⟨n|∂_c H|n⟩` (only when dipole needed).
    pub vdiag: Option<Array1<f64>>,
    /// Eigenvectors `U[:, n]`, shape `(norb, nsta)` — for band tracking.
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
#[derive(Clone, Default)]
pub struct SimplexDiagnostics {
    /// Minimum band gap `min_{n≠m} |E_n − E_m|` across all vertices.
    pub min_gap: f64,
    /// Minimum assignment overlap from band tracking (1.0 = perfect).
    pub min_assignment_overlap: f64,
    /// True if different neighbour paths imply inconsistent band permutations.
    pub tracking_conflict: bool,
}

/// Safety threshold: simplexes with `min_gap < GAP_TOL` are skipped for
/// single‑band Berry/QGT evaluation.
pub const SIMPLEX_GAP_TOL: f64 = 1e-4;
