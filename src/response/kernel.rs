//! Kernel evaluators at quadrature points.
//!
//! Each function takes interpolated band energies $E_n(q)$ and the
//! gauge‑invariant velocity kernel $K^{ab}_{nm}(q) = v^a_{nm}v^b_{mn}$
//! at a single quadrature point $q$, then evaluates the singular
//! denominator:
//!
//! | Function | Denominator | Returns |
//! |----------|-------------|---------|
//! | `eval_berry_kernel` | $\Delta_{nm}^2 + \eta^2$ | $(g_n, \Omega_n)$ per band (all bands) |
//! | `eval_berry_band_at_lam_buf` | $\Delta_{nm}^2 + \eta^2$ | $\Omega_n$ (single band at barycentrics) |
//! | `eval_berry_complex_at_lam_buf` | $\Delta_{nm}^2 + \eta^2$ | $(g_n, \Omega_n)$ (single band) |
//! | `eval_intrinsic_G_at_lam` | $\Delta_{nm}^3$ | $G^{ij}_n$ (single band, no $\eta$) |
//! | `eval_optical_kernel` | $\Delta_{nm}(\Delta_{nm}+\omega+i\eta)$ | $\sum_{nm} -i(f_n-f_m)K_{nm}/{\rm denom}$ |
//!
//! The single‑band functions interpolate only the $n$‑th row
//! of $K$ and avoid allocating the full $nsta\times nsta$ matrix.
//!
//! The single‑simplex quadrature helpers (`quadrature_berry_simplex`
//! etc.) loop over all quadrature points, interpolate, call the
//! evaluator, and accumulate with quadrature weights.

use ndarray::prelude::*;
use num_complex::Complex;

use crate::thermodynamics::{Occupation, fermi_derivative_from_width, fermi_from_width};

use super::quadrature::*;
use super::types::TrackedSimplex;

// ── Fermi functions ─────────────────────────────────────────────────────

// Shared hard cutoff for the isolated-band intrinsic response. This is an
// energy-gap tolerance in eV, not lifetime broadening. Both integration paths
// omit these terms; degenerate subspaces require a separate prescription.
pub(crate) const INTRINSIC_GAP_TOL: f64 = 1e-10;

#[inline]
pub(crate) fn intrinsic_inverse_gap(gap: f64) -> f64 {
    if gap.abs() <= INTRINSIC_GAP_TOL {
        0.0
    } else {
        1.0 / gap
    }
}

#[inline]
pub(crate) fn fermi(e: f64, mu: f64, thermal_width: f64) -> f64 {
    fermi_from_width(e, mu, thermal_width)
}

#[inline]
pub(crate) fn fermi_deriv(e: f64, mu: f64, thermal_width: f64) -> f64 {
    if thermal_width == 0.0 {
        0.0
    } else {
        fermi_derivative_from_width(e, mu, thermal_width)
    }
}

// ── Berry / QGT kernel ──────────────────────────────────────────────────

/// Evaluate `G_n = Σ_{m≠n} K_nm / ((E_n−E_m)² + η²)` at one quadrature point.
///
/// Returns per‑band `(metric_n, berry_n)` where
/// `g_n = Re G_n`, `Ω_n = −2 Im G_n`.
pub(crate) fn eval_berry_kernel(
    band_q: &[f64],
    k_ab_q: &Array2<Complex<f64>>,
    eta: f64,
    nsta: usize,
) -> (Array1<f64>, Array1<f64>) {
    let mut metric = Array1::<f64>::zeros(nsta);
    let mut berry = Array1::<f64>::zeros(nsta);
    let eta2 = eta * eta;
    for n in 0..nsta {
        let mut g_sum = Complex::new(0.0, 0.0);
        for m in 0..nsta {
            if m == n {
                continue;
            }
            let de = band_q[n] - band_q[m];
            let denom = de * de + eta2;
            if denom < 1e-30 {
                continue;
            }
            g_sum += k_ab_q[[n, m]] / denom;
        }
        metric[n] = g_sum.re;
        berry[n] = -2.0 * g_sum.im;
    }
    (metric, berry)
}

/// Evaluate $\Omega_n$ for a single band at one quadrature point.
///
/// Interpolates vertex gaps $E_n-E_m$ and $K_{nm}$ at barycentric coords `lam`,
/// then computes $\Omega_n = -2\\,\mathrm{Im}\sum_{m\ne n} K_{nm}/(\Delta_{nm}^2+\eta^2)$.
/// Reuses caller-provided buffers and computes only the requested band.
#[inline]
pub(crate) fn eval_berry_band_at_lam_buf(
    n: usize,
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    lam: &[f64],
    eta: f64,
    nsta: usize,
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
) -> f64 {
    e_buf[..nsta].fill(0.0);
    k_buf[..nsta].fill(Complex::new(0.0, 0.0));
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            // Interpolate the gap, not two large absolute energies whose
            // separately rounded sums can erase a representable splitting.
            e_buf[m] += (bands[v][n] - bands[v][m]) * lv;
        }
    }
    for v in 0..kmats.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            k_buf[m] += kmats[v][[n, m]] * lv;
        }
    }
    let eta2 = eta * eta;
    let mut g_sum = Complex::new(0.0, 0.0);
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = e_buf[m];
        let denom = de * de + eta2;
        if denom < 1e-30 {
            continue;
        }
        g_sum += k_buf[m] / denom;
    }
    -2.0 * g_sum.im
}

/// Evaluate $G_n = \Sigma_{m\ne n} K_{nm} / (\Delta_{nm}^2 + \eta^2)$
/// for a single band at barycentrics, returning `(metric_n, berry_n)`.
/// Interpolates only vertex gaps $E_n-E_m$ and the $n$-th row of $K$
/// into caller-provided buffers; the energy scratch buffer stores gaps.
#[inline]
pub(crate) fn eval_berry_complex_at_lam_buf(
    n: usize,
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    lam: &[f64],
    eta: f64,
    nsta: usize,
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
) -> (f64, f64) {
    e_buf[..nsta].fill(0.0);
    k_buf[..nsta].fill(Complex::new(0.0, 0.0));
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            // Interpolate the gap, not two large absolute energies whose
            // separately rounded sums can erase a representable splitting.
            e_buf[m] += (bands[v][n] - bands[v][m]) * lv;
        }
    }
    for v in 0..kmats.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            k_buf[m] += kmats[v][[n, m]] * lv;
        }
    }
    let eta2 = eta * eta;
    let mut g_sum = Complex::new(0.0, 0.0);
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = e_buf[m];
        let denom = de * de + eta2;
        if denom < 1e-30 {
            continue;
        }
        g_sum += k_buf[m] / denom;
    }
    (g_sum.re, -2.0 * g_sum.im)
}

/// Evaluate $G^{ij}_n = \operatorname{Re} \sum_{m\ne n} K_{nm} / (E_n-E_m)^3$
/// for a single band at barycentrics (no $\eta$ regularization — used for
/// intrinsic NLH).
#[inline]
#[allow(dead_code)]
pub(crate) fn eval_intrinsic_G_at_lam(
    n: usize,
    bands: &[Vec<f64>],
    kmats: &[Array2<Complex<f64>>],
    lam: &[f64],
    nsta: usize,
) -> f64 {
    let mut e_q = vec![0.0; nsta];
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            e_q[m] += bands[v][m] * lv;
        }
    }
    let mut k_row = vec![Complex::new(0.0, 0.0); nsta];
    for v in 0..kmats.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            k_row[m] += kmats[v][[n, m]] * lv;
        }
    }
    let mut g_sum = 0.0f64;
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = e_q[n] - e_q[m];
        let inv_de3 = intrinsic_inverse_gap(de).powi(3);
        g_sum += k_row[m].re * inv_de3;
    }
    g_sum
}

/// Evaluate $G^{ab}_n$, $G^{bc}_n$, $G^{ac}_n$ in one fused pass.
///
/// Interpolates $E_m$ once, then computes the three G components
/// reusing the same energy interpolation and caller-provided buffers.
#[inline]
pub(crate) fn eval_intrinsic_G3_at_lam_buf(
    n: usize,
    bands: &[&[f64]],
    kmat_ab: &[&Array2<Complex<f64>>],
    kmat_bc: &[&Array2<Complex<f64>>],
    kmat_ac: &[&Array2<Complex<f64>>],
    lam: &[f64],
    nsta: usize,
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
) -> (f64, f64, f64) {
    // Interpolate energies + all three K rows in one pass.
    e_buf[..nsta].fill(0.0);
    let (k_ab_row, rest) = k_buf.split_at_mut(nsta);
    let (k_bc_row, k_ac_row) = rest.split_at_mut(nsta);
    k_ab_row.fill(Complex::new(0.0, 0.0));
    k_bc_row.fill(Complex::new(0.0, 0.0));
    k_ac_row.fill(Complex::new(0.0, 0.0));
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            e_buf[m] += bands[v][m] * lv;
            k_ab_row[m] += kmat_ab[v][[n, m]] * lv;
            k_bc_row[m] += kmat_bc[v][[n, m]] * lv;
            k_ac_row[m] += kmat_ac[v][[n, m]] * lv;
        }
    }
    let en = e_buf[n];
    let mut g_ab = 0.0f64;
    let mut g_bc = 0.0f64;
    let mut g_ac = 0.0f64;
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = en - e_buf[m];
        let inv_de3 = intrinsic_inverse_gap(de).powi(3);
        g_ab += k_ab_row[m].re * inv_de3;
        g_bc += k_bc_row[m].re * inv_de3;
        g_ac += k_ac_row[m].re * inv_de3;
    }
    (g_ab, g_bc, g_ac)
}

// ── Optical kernel ──────────────────────────────────────────────────────

/// Evaluate the optical conductivity kernel at one quadrature point.
///
/// ```text
/// σ_nm = -i (f_n − f_m) · K_nm / [d (d + ω + iη)]
/// ```
/// This is the interband Kubo conductivity with `e²/hbar` omitted. Exactly
/// degenerate pairs and intraband Drude terms are excluded. Unbroadened poles
/// propagate nonfinite values, which the public entry point reports as an error.
pub(crate) fn eval_optical_kernel(
    band_q: &[f64],
    k_ab_q: &Array2<Complex<f64>>,
    omega: f64,
    eta: f64,
    mu: f64,
    thermal_width: f64,
    nsta: usize,
) -> Complex<f64> {
    let mut total = Complex::new(0.0, 0.0);
    let w_plus_ieta = Complex::new(omega, eta);
    for n in 0..nsta {
        let fn_val = fermi(band_q[n], mu, thermal_width);
        for m in 0..nsta {
            if m == n {
                continue;
            }
            let d = band_q[n] - band_q[m];
            if d == 0.0 || k_ab_q[[n, m]] == Complex::new(0.0, 0.0) {
                continue;
            }
            let fm_val = fermi(band_q[m], mu, thermal_width);
            // The midpoint derivative avoids cancellation in the divided
            // difference at finite T; its error is O((d/thermal_width)^2).
            let df_over_d = if thermal_width > 0.0 && d.abs() < 1e-5 * thermal_width {
                -fermi_deriv(band_q[n].midpoint(band_q[m]), mu, thermal_width)
            } else {
                (fn_val - fm_val) / d
            };
            if df_over_d == 0.0 {
                continue;
            }
            total += Complex::new(0.0, -df_over_d) * k_ab_q[[n, m]] / (d + w_plus_ieta);
        }
    }
    total
}

// ── Quadrature over single simplex ──────────────────────────────────────

/// Occupation-weighted Berry curvature and quantum metric on one simplex.
pub(crate) fn quadrature_occupied_geometry_simplex<const NV: usize>(
    sim: &TrackedSimplex<'_, NV>,
    eta: f64,
    chemical_potentials: &Array1<f64>,
    occupation: Occupation,
) -> (Array1<f64>, Array1<f64>) {
    let dimension = NV - 1;
    let nsta = sim.vertices[0].band.len();
    let bands: Vec<&[f64]> = (0..NV)
        .map(|vertex| sim.vertices[vertex].band.as_slice().unwrap())
        .collect();
    let kernels: Vec<&Array2<Complex<f64>>> =
        (0..NV).map(|vertex| &sim.vertices[vertex].k_ab).collect();
    let mut metric = Array1::<f64>::zeros(chemical_potentials.len());
    let mut berry = Array1::<f64>::zeros(chemical_potentials.len());

    let mut gaps = vec![0.0; nsta];
    let mut row_kernel = vec![Complex::new(0.0, 0.0); nsta];
    let mut accumulate = |lambda: &[f64], weight: f64| {
        let energies = bary_interp_band_refs(&bands, lambda, nsta);
        for band in 0..nsta {
            let (metric_n, berry_n) = eval_berry_complex_at_lam_buf(
                band,
                &bands,
                &kernels,
                lambda,
                eta,
                nsta,
                &mut gaps,
                &mut row_kernel,
            );
            for (index, &mu) in chemical_potentials.iter().enumerate() {
                let f = occupation.value_unchecked(energies[band], mu);
                metric[index] += weight * f * metric_n;
                berry[index] += weight * f * berry_n;
            }
        }
    };

    if dimension == 2 {
        for index in 0..TRI_QUAD_PTS_3.len() {
            accumulate(&TRI_QUAD_PTS_3[index], TRI_QUAD_WTS_3[index]);
        }
    } else {
        for index in 0..TET_QUAD_PTS_4.len() {
            accumulate(&TET_QUAD_PTS_4[index], TET_QUAD_WTS_4[index]);
        }
    }
    (metric * sim.volume, berry * sim.volume)
}

pub(crate) fn quadrature_optical_simplex<const NV: usize>(
    sim: &TrackedSimplex<'_, NV>,
    frequencies: &Array1<f64>,
    eta: f64,
    mu: f64,
    thermal_width: f64,
    total: &mut Array1<Complex<f64>>,
) {
    let d = NV - 1;
    let nsta = sim.vertices[0].band.len();
    let bands: Vec<&[f64]> = (0..NV)
        .map(|v| sim.vertices[v].band.as_slice().unwrap())
        .collect();
    let kmats: Vec<&Array2<Complex<f64>>> = (0..NV).map(|v| &sim.vertices[v].k_ab).collect();
    let mut accumulate = |lam: &[f64], weight: f64| {
        let band_q = bary_interp_band_refs(&bands, lam, nsta);
        let k_ab_q = bary_interp_matrix_refs(&kmats, lam);
        for (out, &omega) in total.iter_mut().zip(frequencies) {
            *out += weight
                * sim.volume
                * eval_optical_kernel(&band_q, &k_ab_q, omega, eta, mu, thermal_width, nsta);
        }
    };
    if d == 2 {
        for iq in 0..3 {
            accumulate(TRI_QUAD_PTS_3[iq].as_slice(), TRI_QUAD_WTS_3[iq]);
        }
    } else {
        for iq in 0..4 {
            accumulate(TET_QUAD_PTS_4[iq].as_slice(), TET_QUAD_WTS_4[iq]);
        }
    }
}
