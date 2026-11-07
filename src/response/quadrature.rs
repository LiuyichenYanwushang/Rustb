//! Symmetric simplex quadrature rules and barycentric interpolation.

use ndarray::prelude::*;
use num_complex::Complex;

// ── Quadrature rules ────────────────────────────────────────────────────

/// Degree‑2 symmetric 3‑point rule for the reference triangle.
/// Barycentric coordinates; weights sum to 1.
pub(crate) const TRI_QUAD_PTS_3: [[f64; 3]; 3] = [
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];
pub(crate) const TRI_QUAD_WTS_3: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

/// Degree‑2 symmetric 4‑point rule for the reference tetrahedron.
/// Barycentric coordinates; weights sum to 1.
pub(crate) const TET_QUAD_PTS_4: [[f64; 4]; 4] = [
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
pub(crate) const TET_QUAD_WTS_4: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

// ── Barycentric interpolation ───────────────────────────────────────────

/// Interpolate band energies at barycentric coords `lam` from borrowed
/// vertex band slices (no per-vertex clone).
pub(crate) fn bary_interp_band_refs(bands: &[&[f64]], lam: &[f64], nsta: usize) -> Vec<f64> {
    let mut out = vec![0.0; nsta];
    bary_interp_band_refs_into(bands, lam, nsta, &mut out);
    out
}

/// Buffer-reusing form of [`bary_interp_band_refs`].
pub(crate) fn bary_interp_band_refs_into(
    bands: &[&[f64]],
    lam: &[f64],
    nsta: usize,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), nsta);
    for n in 0..nsta {
        let origin = bands[0][n];
        // Affine form preserves constant bands exactly, including occupations
        // at their chemical potential. Avoid overflowing opposite-sign shifts.
        if bands.iter().all(|band| (band[n] - origin).is_finite()) {
            out[n] = origin
                + bands
                    .iter()
                    .zip(lam)
                    .filter(|(_, w)| **w != 0.0)
                    .map(|(band, w)| (band[n] - origin) * w)
                    .sum::<f64>();
        } else {
            out[n] = bands
                .iter()
                .zip(lam)
                .filter(|(_, w)| **w != 0.0)
                .map(|(band, w)| band[n] * w)
                .sum();
        }
    }
}

/// Interpolate a matrix field `mats[d+1]` each `(nsta, nsta)` at
/// barycentric coords `lam` from borrowed vertex matrices.
pub(crate) fn bary_interp_matrix_refs_into(
    mats: &[&Array2<Complex<f64>>],
    lam: &[f64],
    out: &mut Array2<Complex<f64>>,
) {
    let n = mats[0].nrows();
    debug_assert_eq!(out.dim(), (n, n));
    out.fill(Complex::new(0.0, 0.0));
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
}
