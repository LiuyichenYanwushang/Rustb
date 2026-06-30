//! Kernel evaluators at quadrature points.
//!
//! These functions take interpolated `E_n(q)` and `K_nm(q)` at a
//! quadrature point and assemble the Berry / QGT / optical response.

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

// ── Intrinsic NLH Q‑tensor kernel ───────────────────────────────────────

/// Evaluate the intrinsic NLH kernel `Q^{ab;c}_n` at one quadrature point.
///
/// Requires three direction pairs: `k_ab`, `k_bc`, `k_ac` and diagonal
/// velocities `v_a`, `v_b`, `v_c`.
///
/// ```text
/// Q^{ab;c}_n = 2 v^c_n G^{ab}_n − ½(v^a_n G^{bc}_n + v^b_n G^{ac}_n)
/// G^{ij}_n = Re Σ_{m≠n} K^{ij}_nm / (d² + η²)
/// ```
pub fn eval_q_tensor(
    band_q: &[f64],
    k_ab_q: &Array2<Complex<f64>>,
    k_bc_q: &Array2<Complex<f64>>,
    k_ac_q: &Array2<Complex<f64>>,
    vdiag_a: &[f64],
    vdiag_b: &[f64],
    vdiag_c: &[f64],
    eta: f64,
    nsta: usize,
) -> Array1<f64> {
    let eta2 = eta * eta;
    let mut g_ab = Array1::<f64>::zeros(nsta);
    let mut g_bc = Array1::<f64>::zeros(nsta);
    let mut g_ac = Array1::<f64>::zeros(nsta);
    for n in 0..nsta {
        for m in 0..nsta {
            if m == n {
                continue;
            }
            let de = band_q[n] - band_q[m];
            let denom = de * de + eta2;
            if denom < 1e-30 {
                continue;
            }
            g_ab[n] += (k_ab_q[[n, m]] / denom).re;
            g_bc[n] += (k_bc_q[[n, m]] / denom).re;
            g_ac[n] += (k_ac_q[[n, m]] / denom).re;
        }
    }
    let mut q = Array1::<f64>::zeros(nsta);
    for n in 0..nsta {
        q[n] = 2.0 * vdiag_c[n] * g_ab[n] - 0.5 * (vdiag_a[n] * g_bc[n] + vdiag_b[n] * g_ac[n]);
    }
    q
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

/// Quadrature over one simplex for the Berry dipole.
pub fn quadrature_dipole_simplex(
    sim: &TrackedSimplex,
    eta: f64,
    mu: &Array1<f64>,
    beta: f64,
) -> Array1<f64> {
    let d = sim.vertices.len() - 1;
    let nsta = sim.vertices[0].band.len();
    let nv = d + 1;
    let n_mu = mu.len();
    let bands: Vec<Vec<f64>> = (0..nv).map(|v| sim.vertices[v].band.to_vec()).collect();
    let kmats: Vec<Array2<Complex<f64>>> = (0..nv).map(|v| sim.vertices[v].k_ab.clone()).collect();
    let vdiags: Vec<Vec<f64>> = (0..nv)
        .map(|v| {
            sim.vertices[v]
                .vdiag
                .as_ref()
                .map(|vd| vd.to_vec())
                .unwrap_or_else(|| vec![0.0; nsta])
        })
        .collect();
    let mut acc = Array1::<f64>::zeros(n_mu);
    if d == 2 {
        for iq in 0..3 {
            let lam = TRI_QUAD_PTS_3[iq].as_slice();
            let w = TRI_QUAD_WTS_3[iq];
            let band_q = bary_interp_band(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam);
            let vdiag_q = bary_interp_band(&vdiags, lam, nsta);
            let (_g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
            for im in 0..n_mu {
                for n in 0..nsta {
                    let df = fermi_deriv(band_q[n], mu[im], beta);
                    acc[[im]] += w * df * vdiag_q[n] * o_n[n];
                }
            }
        }
    } else {
        for iq in 0..4 {
            let lam = TET_QUAD_PTS_4[iq].as_slice();
            let w = TET_QUAD_WTS_4[iq];
            let band_q = bary_interp_band(&bands, lam, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam);
            let vdiag_q = bary_interp_band(&vdiags, lam, nsta);
            let (_g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
            for im in 0..n_mu {
                for n in 0..nsta {
                    let df = fermi_deriv(band_q[n], mu[im], beta);
                    acc[[im]] += w * df * vdiag_q[n] * o_n[n];
                }
            }
        }
    }
    acc * sim.volume
}

/// Quadrature over one simplex for the optical conductivity.
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
