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
//! | `eval_berry_band_at_lam` | $\Delta_{nm}^2 + \eta^2$ | $\Omega_n$ (single band at barycentrics) |
//! | `eval_berry_complex_at_lam` | $\Delta_{nm}^2 + \eta^2$ | $(g_n, \Omega_n)$ (single band) |
//! | `eval_intrinsic_G_at_lam` | $\Delta_{nm}^3$ | $G^{ij}_n$ (single band, no $\eta$) |
//! | `eval_optical_kernel` | $\Delta_{nm}^2 - (\omega+i\eta)^2$ | $\sum_{nm} (f_n-f_m)K_{nm}/{\rm denom}$ |
//!
//! The single‑band functions (`_at_lam`) interpolate only the $n$‑th row
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

#[inline]
pub(crate) fn fermi(e: f64, mu: f64, thermal_width: f64) -> f64 {
    fermi_from_width(e, mu, thermal_width)
}

#[inline]
#[allow(dead_code)]
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
/// Interpolates $E_m(q)$ and $K_{nm}(q)$ at barycentric coords `lam`,
/// then computes $\Omega_n = -2\,\mathrm{Im}\sum_{m\ne n} K_{nm}/(\Delta_{nm}^2+\eta^2)$.
/// Avoids allocating the full $K$ matrix and computing $\Omega$ for other bands.
#[allow(dead_code)]
pub(crate) fn eval_berry_band_at_lam(
    n: usize,
    bands: &[Vec<f64>],
    kmats: &[Array2<Complex<f64>>],
    lam: &[f64],
    eta: f64,
    nsta: usize,
) -> f64 {
    // Interpolate E_m for all bands
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

    // Interpolate K_{nm} for row n only
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

    let eta2 = eta * eta;
    let mut g_sum = Complex::new(0.0, 0.0);
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = e_q[n] - e_q[m];
        let denom = de * de + eta2;
        if denom < 1e-30 {
            continue;
        }
        g_sum += k_row[m] / denom;
    }
    -2.0 * g_sum.im
}

/// Pre‑allocated buffer version: avoids Vec allocation on every call.
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
            e_buf[m] += bands[v][m] * lv;
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
        let de = e_buf[n] - e_buf[m];
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
/// Interpolates only $E_m$ and the $n$-th row of $K$.
#[allow(dead_code)]
pub(crate) fn eval_berry_complex_at_lam(
    n: usize,
    bands: &[Vec<f64>],
    kmats: &[Array2<Complex<f64>>],
    lam: &[f64],
    eta: f64,
    nsta: usize,
) -> (f64, f64) {
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
    let eta2 = eta * eta;
    let mut g_sum = Complex::new(0.0, 0.0);
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = e_q[n] - e_q[m];
        let denom = de * de + eta2;
        if denom < 1e-30 {
            continue;
        }
        g_sum += k_row[m] / denom;
    }
    (g_sum.re, -2.0 * g_sum.im)
}

/// Pre‑allocated buffer version of [`eval_berry_complex_at_lam`].
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
            e_buf[m] += bands[v][m] * lv;
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
        let de = e_buf[n] - e_buf[m];
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
        let de3 = de * de * de;
        if de3.abs() < 1e-30 {
            continue;
        }
        g_sum += k_row[m].re / de3;
    }
    g_sum
}

/// Pre‑allocated buffer version of [`eval_intrinsic_G_at_lam`].
#[inline]
#[allow(dead_code)]
pub(crate) fn eval_intrinsic_G_at_lam_buf(
    n: usize,
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    lam: &[f64],
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
            e_buf[m] += bands[v][m] * lv;
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
    let mut g_sum = 0.0f64;
    for m in 0..nsta {
        if m == n {
            continue;
        }
        let de = e_buf[n] - e_buf[m];
        let de3 = de * de * de;
        if de3.abs() < 1e-30 {
            continue;
        }
        g_sum += k_buf[m].re / de3;
    }
    g_sum
}

/// Evaluate $G^{ab}_n$, $G^{bc}_n$, $G^{ac}_n$ in one fused pass.
///
/// Interpolates $E_m$ once, then computes the three G components
/// reusing the same energy interpolation.  Saves 2 redundant energy
/// interpolations vs three separate `eval_intrinsic_G_at_lam_buf` calls.
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
        let de3 = de * de * de;
        if de3.abs() < 1e-30 {
            continue;
        }
        g_ab += k_ab_row[m].re / de3;
        g_bc += k_bc_row[m].re / de3;
        g_ac += k_ac_row[m].re / de3;
    }
    (g_ab, g_bc, g_ac)
}

// ── Optical kernel ──────────────────────────────────────────────────────

/// Evaluate the optical conductivity kernel at one quadrature point.
///
/// ```text
/// σ_nm = (f_n − f_m) · K_nm / (d² − (ω+iη)²)
/// ```
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
    let denom_shift = w_plus_ieta * w_plus_ieta;
    for n in 0..nsta {
        let fn_val = fermi(band_q[n], mu, thermal_width);
        for m in 0..nsta {
            if m == n {
                continue;
            }
            let fm_val = fermi(band_q[m], mu, thermal_width);
            let df = fn_val - fm_val;
            if df.abs() < 1e-30 {
                continue;
            }
            let d = band_q[n] - band_q[m];
            let denom = d * d - denom_shift;
            if denom.norm_sqr() < 1e-30 {
                continue;
            }
            total += df * k_ab_q[[n, m]] / denom;
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

    let mut accumulate = |lambda: &[f64], weight: f64| {
        let energies = bary_interp_band_refs(&bands, lambda, nsta);
        let kernel = bary_interp_matrix_refs(&kernels, lambda);
        let (metric_n, berry_n) = eval_berry_kernel(&energies, &kernel, eta, nsta);
        for (index, &mu) in chemical_potentials.iter().enumerate() {
            for band in 0..nsta {
                let f = occupation.value_unchecked(energies[band], mu);
                metric[index] += weight * f * metric_n[band];
                berry[index] += weight * f * berry_n[band];
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
    omega: f64,
    eta: f64,
    mu: f64,
    thermal_width: f64,
) -> Complex<f64> {
    let d = NV - 1;
    let nsta = sim.vertices[0].band.len();
    let bands: Vec<&[f64]> = (0..NV)
        .map(|v| sim.vertices[v].band.as_slice().unwrap())
        .collect();
    let kmats: Vec<&Array2<Complex<f64>>> = (0..NV).map(|v| &sim.vertices[v].k_ab).collect();
    let mut total = Complex::new(0.0, 0.0);
    if d == 2 {
        for iq in 0..3 {
            let lam = TRI_QUAD_PTS_3[iq].as_slice();
            let w = TRI_QUAD_WTS_3[iq];
            let band_q = bary_interp_band_refs(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix_refs(&kmats, lam);
            total += w * eval_optical_kernel(&band_q, &k_ab_q, omega, eta, mu, thermal_width, nsta);
        }
    } else {
        for iq in 0..4 {
            let lam = TET_QUAD_PTS_4[iq].as_slice();
            let w = TET_QUAD_WTS_4[iq];
            let band_q = bary_interp_band_refs(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix_refs(&kmats, lam);
            total += w * eval_optical_kernel(&band_q, &k_ab_q, omega, eta, mu, thermal_width, nsta);
        }
    }
    total * sim.volume
}
