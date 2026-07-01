//! Kernel evaluators at quadrature points.
//!
//! Each function takes interpolated band energies $E_n(q)$ and the
//! gauge‑invariant velocity kernel $K^{ab}_{nm}(q) = v^a_{nm}v^b_{mn}$
//! at a single quadrature point $q$, then evaluates the singular
//! denominator:
//!
//! | Function | Denominator | Returns |
//! |----------|-------------|---------|
//! | `eval_berry_kernel` | $d_{nm}^2 + \eta^2$ | $(g_n, \Omega_n)$ per band |
//! | `eval_optical_kernel` | $d_{nm}^2 - (\omega+i\eta)^2$ | $\sum_{nm} (f_n-f_m)K_{nm}/{\rm denom}$ |
//! | `eval_q_tensor` | $d_{nm}^2 + \eta^2$ | $Q^{ab;c}_n$ per band |
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
/// G^{ij}_n = Re Σ_{m≠n} K^{ij}_nm · d_{nm} / (d_{nm}² + η²)²
/// ```
///
/// The odd regularizer `R_η(d) = d / (d²+η²)²` preserves the correct
/// `1/d³` asymptotic for |d| ≫ η while staying finite at d = 0.
/// Note: this is an experimental kernel not yet validated against
/// the direct‑sum `berry_connection_dipole` reference.
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
            // Odd regularizer: R_η(d) = d / (d² + η²)²  →  ~1/d³ for |d| ≫ η
            let d2 = de * de + eta2;
            if d2 < 1e-30 {
                continue;
            }
            let weight = de / (d2 * d2);
            g_ab[n] += (k_ab_q[[n, m]] * weight).re;
            g_bc[n] += (k_bc_q[[n, m]] * weight).re;
            g_ac[n] += (k_ac_q[[n, m]] * weight).re;
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
///
/// When `sim.vdiag_4th` is `Some` (2D triangles), vdiag is interpolated
/// bilinearly over the parent rectangle using the 4th corner value.
/// Otherwise, barycentric interpolation is used.
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
        if let Some(ref vdiag_4th) = sim.vdiag_4th {
            // Bilinear interpolation on the parent rectangle.
            // Detect triangle type from vertex coordinates.
            let v0_x = sim.coords[[0, 0]];
            let v0_y = sim.coords[[0, 1]];
            let v1_x = sim.coords[[1, 0]];
            let v1_y = sim.coords[[1, 1]];
            let v2_x = sim.coords[[2, 0]];
            let v2_y = sim.coords[[2, 1]];
            let min_x = v0_x.min(v1_x).min(v2_x);
            let min_y = v0_y.min(v1_y).min(v2_y);
            // vertex0 is always diagonally opposite the 4th rectangle corner.
            // If vertex0 is lower-left → Triangle 1 (i00,i10,i01).
            // If vertex0 is upper-right → Triangle 2 (i11,i10,i01).
            let v0_is_ll = v0_x <= min_x + 1e-15 && v0_y <= min_y + 1e-15;

            for iq in 0..3 {
                let lam = TRI_QUAD_PTS_3[iq].as_slice();
                let w = TRI_QUAD_WTS_3[iq];
                let band_q = bary_interp_band(&bands, lam, nsta);
                let k_ab_q = bary_interp_matrix(&kmats, lam);

                // Map barycentric λ to rectangle (x, y) coords.
                let (x, y) = if v0_is_ll {
                    // Triangle 1: v0=(0,0), v1=(1,0), v2=(0,1)
                    (lam[1], lam[2])
                } else {
                    // Triangle 2: v0=(1,1), v1=(1,0), v2=(0,1)
                    (lam[0] + lam[1], lam[0] + lam[2])
                };
                // Bilinear corners: v00=(0,0), v10=(1,0), v01=(0,1), v11=(1,1)
                let v00: &[f64] = if v0_is_ll { &vdiags[0] } else { vdiag_4th };
                let v10: &[f64] = &vdiags[1];
                let v01: &[f64] = &vdiags[2];
                let v11: &[f64] = if v0_is_ll { vdiag_4th } else { &vdiags[0] };

                let mut vdiag_q = vec![0.0_f64; nsta];
                for n in 0..nsta {
                    vdiag_q[n] = (1.0 - x) * (1.0 - y) * v00[n]
                        + x * (1.0 - y) * v10[n]
                        + x * y * v11[n]
                        + (1.0 - x) * y * v01[n];
                }

                let (_g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
                for im in 0..n_mu {
                    for n in 0..nsta {
                        let df = fermi_deriv(band_q[n], mu[im], beta);
                        acc[[im]] += w * df * vdiag_q[n] * o_n[n];
                    }
                }
            }
        } else {
            // Fallback barycentric interpolation for vdiag (no 4th corner).
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
