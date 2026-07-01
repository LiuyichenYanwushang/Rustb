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

/// Rectangle‑corner vdiag values for bilinear interpolation on 2D cells.
///
/// Corners are indexed `[i00, i10, i01, i11]` where
///
/// ```text
/// i00=(0,0)  i10=(1,0)  i01=(0,1)  i11=(1,1)
/// ```
///
/// in the parent rectangle local coordinates.  At a quadrature point
/// $(x,y)\in[0,1]^2$ the bilinear interpolation is
///
/// $$v(x,y) = (1-x)(1-y)v_{00} + x(1-y)v_{10} + xy v_{11} + (1-x)y v_{01}$$
#[derive(Clone)]
pub struct VdiagRect {
    pub corners: [Vec<f64>; 4],
}

impl VdiagRect {
    /// Bilinear interpolation at rectangle‑local coords $(x,y)\in[0,1]^2$.
    pub fn interp(&self, x: f64, y: f64, nsta: usize) -> Vec<f64> {
        let (c00, c10, c01, c11) = (
            &self.corners[0],
            &self.corners[1],
            &self.corners[2],
            &self.corners[3],
        );
        let mut v = vec![0.0; nsta];
        for n in 0..nsta {
            v[n] = (1.0 - x) * (1.0 - y) * c00[n]
                + x * (1.0 - y) * c10[n]
                + x * y * c11[n]
                + (1.0 - x) * y * c01[n];
        }
        v
    }
}

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
    /// Rectangle‑corner vdiag for bilinear interpolation (2D only).
    /// Four corners indexed `[i00, i10, i01, i11]` in parent‑rectangle
    /// local coordinates $(0,0),(1,0),(0,1),(1,1)$.
    pub vdiag_rect: Option<VdiagRect>,
    /// Per‑vertex $(x,y)$ coordinates in the parent rectangle $[0,1]^2$.
    /// Shape `(nv, 2)`: row $i$ holds $(x_i, y_i)$ for `vertices[i]`.
    pub vertex_xy: Array2<f64>,
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
    /// Placeholder: currently always `false`.  Future global band‑tracking
    /// pass will detect path‑dependent permutation conflicts.
    pub tracking_conflict: bool,
}

/// Safety threshold: simplexes with `min_gap < GAP_TOL` are skipped for
/// single‑band Berry/QGT evaluation.
pub const SIMPLEX_GAP_TOL: f64 = 1e-4;
