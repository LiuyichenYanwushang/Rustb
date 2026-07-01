//! Symmetric simplex quadrature rules and barycentric interpolation.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

// ── Quadrature rules ────────────────────────────────────────────────────

/// Degree‑2 symmetric 3‑point rule for the reference triangle.
/// Barycentric coordinates; weights sum to 1.
pub const TRI_QUAD_PTS_3: [[f64; 3]; 3] = [
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];
pub const TRI_QUAD_WTS_3: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

/// Degree‑2 symmetric 4‑point rule for the reference tetrahedron.
/// Barycentric coordinates; weights sum to 1.
pub const TET_QUAD_PTS_4: [[f64; 4]; 4] = [
    [
        0.5854101966249685,
        0.1381966011250105,
        0.1381966011250105,
        0.1381966011250105,
    ],
    [
        0.1381966011250105,
        0.5854101966249685,
        0.1381966011250105,
        0.1381966011250105,
    ],
    [
        0.1381966011250105,
        0.1381966011250105,
        0.5854101966249685,
        0.1381966011250105,
    ],
    [
        0.1381966011250105,
        0.1381966011250105,
        0.1381966011250105,
        0.5854101966249685,
    ],
];
pub const TET_QUAD_WTS_4: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

/// Degree‑3 6‑point rule for the reference triangle (all positive weights).
/// Exact for polynomials up to degree 3.
pub const TRI_QUAD_PTS_6: [[f64; 3]; 6] = [
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];
pub const TRI_QUAD_WTS_6: [f64; 6] = [
    1.0 / 15.0,
    1.0 / 15.0,
    1.0 / 15.0,
    4.0 / 15.0,
    4.0 / 15.0,
    4.0 / 15.0,
];

// ── Barycentric interpolation ───────────────────────────────────────────

/// Interpolate a scalar field `vals[d+1]` at barycentric coords `lam[d+1]`.
#[inline]
pub fn bary_interp_scalar(vals: &[f64], lam: &[f64]) -> f64 {
    vals.iter().zip(lam.iter()).map(|(v, l)| v * l).sum()
}

/// Interpolate band energies at barycentric coords `lam`.
pub fn bary_interp_band(bands: &[Vec<f64>], lam: &[f64], nsta: usize) -> Vec<f64> {
    let mut out = vec![0.0; nsta];
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for n in 0..nsta {
            out[n] += bands[v][n] * lv;
        }
    }
    out
}

/// Interpolate a matrix field `mats[d+1]` each `(nsta, nsta)` at
/// barycentric coords `lam`.
pub fn bary_interp_matrix(mats: &[Array2<Complex<f64>>], lam: &[f64]) -> Array2<Complex<f64>> {
    let n = mats[0].nrows();
    let mut out = Array2::<Complex<f64>>::zeros((n, n));
    for (mat, &w) in mats.iter().zip(lam.iter()) {
        if w == 0.0 {
            continue;
        }
        for i in 0..n {
            for j in 0..n {
                out[[i, j]] += mat[[i, j]] * w;
            }
        }
    }
    out
}
