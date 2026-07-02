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
use ndarray::*;
use num_complex::Complex;

use super::quadrature::*;
use super::types::{SIMPLEX_GAP_TOL, TrackedSimplex};

// ── Fermi functions ─────────────────────────────────────────────────────

#[inline]
pub fn fermi(e: f64, mu: f64, beta: f64) -> f64 {
    if beta == 0.0 {
        0.5
    } else {
        let x = beta * (e - mu);
        if x > 50.0 {
            0.0
        } else if x < -50.0 {
            1.0
        } else {
            1.0 / (1.0 + x.exp())
        }
    }
}

#[inline]
pub fn fermi_deriv(e: f64, mu: f64, beta: f64) -> f64 {
    let x = beta * (e - mu);
    if x > 50.0 || x < -50.0 {
        0.0
    } else {
        let ex = x.exp();
        beta * ex / ((1.0 + ex) * (1.0 + ex))
    }
}

// ── Berry / QGT kernel ──────────────────────────────────────────────────

/// Evaluate `G_n = Σ_{m≠n} K_nm / ((E_n−E_m)² + η²)` at one quadrature point.
///
/// Returns per‑band `(metric_n, berry_n)` where
/// `g_n = Re G_n`, `Ω_n = −2 Im G_n`.
pub fn eval_berry_kernel(
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
pub fn eval_berry_band_at_lam(
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
pub fn eval_berry_band_at_lam_buf(
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
pub fn eval_berry_complex_at_lam(
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
pub fn eval_berry_complex_at_lam_buf(
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
pub fn eval_intrinsic_G_at_lam(
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
pub fn eval_intrinsic_G_at_lam_buf(
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
pub fn eval_intrinsic_G3_at_lam_buf(
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
    // Interpolate energies once
    e_buf[..nsta].fill(0.0);
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 {
            continue;
        }
        for m in 0..nsta {
            e_buf[m] += bands[v][m] * lv;
        }
    }
    let en = e_buf[n];

    macro_rules! g_one {
        ($kmats:expr) => {{
            k_buf[..nsta].fill(Complex::new(0.0, 0.0));
            for v in 0..$kmats.len() {
                let lv = lam[v];
                if lv == 0.0 {
                    continue;
                }
                for m in 0..nsta {
                    k_buf[m] += $kmats[v][[n, m]] * lv;
                }
            }
            let mut s = 0.0f64;
            for m in 0..nsta {
                if m == n {
                    continue;
                }
                let de = en - e_buf[m];
                let de3 = de * de * de;
                if de3.abs() < 1e-30 {
                    continue;
                }
                s += k_buf[m].re / de3;
            }
            s
        }};
    }

    let g_ab = g_one!(kmat_ab);
    let g_bc = g_one!(kmat_bc);
    let g_ac = g_one!(kmat_ac);
    (g_ab, g_bc, g_ac)
}

// ── Optical kernel ──────────────────────────────────────────────────────

/// Evaluate the optical conductivity kernel at one quadrature point.
///
/// ```text
/// σ_nm = (f_n − f_m) · K_nm / (d² − (ω+iη)²)
/// ```
pub fn eval_optical_kernel(
    band_q: &[f64],
    k_ab_q: &Array2<Complex<f64>>,
    omega: f64,
    eta: f64,
    mu: f64,
    beta: f64,
    nsta: usize,
) -> Complex<f64> {
    let mut total = Complex::new(0.0, 0.0);
    let w_plus_ieta = Complex::new(omega, eta);
    let denom_shift = w_plus_ieta * w_plus_ieta;
    for n in 0..nsta {
        let fn_val = fermi(band_q[n], mu, beta);
        for m in 0..nsta {
            if m == n {
                continue;
            }
            let fm_val = fermi(band_q[m], mu, beta);
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

/// Quadrature over one simplex for the Berry + metric kernel.
pub fn quadrature_berry_simplex(sim: &TrackedSimplex, eta: f64) -> (f64, f64) {
    let d = sim.vertices.len() - 1;
    let nsta = sim.vertices[0].band.len();
    let nv = d + 1;
    let bands: Vec<Vec<f64>> = (0..nv).map(|v| sim.vertices[v].band.to_vec()).collect();
    let kmats: Vec<Array2<Complex<f64>>> = (0..nv).map(|v| sim.vertices[v].k_ab.clone()).collect();

    let mut total_g = 0.0;
    let mut total_o = 0.0;
    if d == 2 {
        for iq in 0..3 {
            let lam = TRI_QUAD_PTS_3[iq].as_slice();
            let w = TRI_QUAD_WTS_3[iq];
            let band_q = bary_interp_band(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam);
            let (g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
            total_g += w * g_n.iter().copied().sum::<f64>();
            total_o += w * o_n.iter().copied().sum::<f64>();
        }
    } else {
        for iq in 0..4 {
            let lam = TET_QUAD_PTS_4[iq].as_slice();
            let w = TET_QUAD_WTS_4[iq];
            let band_q = bary_interp_band(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam);
            let (g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
            total_g += w * g_n.iter().copied().sum::<f64>();
            total_o += w * o_n.iter().copied().sum::<f64>();
        }
    }
    (total_g * sim.volume, total_o * sim.volume)
}

pub fn quadrature_optical_simplex(
    sim: &TrackedSimplex,
    omega: f64,
    eta: f64,
    mu: f64,
    beta: f64,
) -> Complex<f64> {
    let d = sim.vertices.len() - 1;
    let nsta = sim.vertices[0].band.len();
    let nv = d + 1;
    let bands: Vec<Vec<f64>> = (0..nv).map(|v| sim.vertices[v].band.to_vec()).collect();
    let kmats: Vec<Array2<Complex<f64>>> = (0..nv).map(|v| sim.vertices[v].k_ab.clone()).collect();
    let mut total = Complex::new(0.0, 0.0);
    if d == 2 {
        for iq in 0..3 {
            let lam = TRI_QUAD_PTS_3[iq].as_slice();
            let w = TRI_QUAD_WTS_3[iq];
            let band_q = bary_interp_band(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam);
            total += w * eval_optical_kernel(&band_q, &k_ab_q, omega, eta, mu, beta, nsta);
        }
    } else {
        for iq in 0..4 {
            let lam = TET_QUAD_PTS_4[iq].as_slice();
            let w = TET_QUAD_WTS_4[iq];
            let band_q = bary_interp_band(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam);
            total += w * eval_optical_kernel(&band_q, &k_ab_q, omega, eta, mu, beta, nsta);
        }
    }
    total * sim.volume
}
