//! Energy‑cut integration via K‑quadrature on simplex intersections.
//!
//! ## 2D (triangle)
//!
//! The $E=\mu$ line intersects a triangle in a line segment.  The integral
//!
//! $$\int_T A(k)\,\delta(E(k)-\mu)\,d^2k = \frac{|pq|}{|\nabla E|}\int_0^1 A(t)\,dt$$
//!
//! is evaluated with 2‑point Gauss‑Legendre K‑quadrature along the segment:
//! $E_m$, $K_{nm}^{ab}$, and diagonal velocities are barycentrically
//! interpolated, then $\Omega_n$ (or $G_n$) is recomputed from primitives.
//!
//! ## 3D (tetrahedron)
//!
//! The $E=\mu$ plane intersects a tetrahedron in a convex polygon (triangle
//! or quadrilateral).  The surface integral
//!
//! $$\int_T A(k)\,\delta(E(k)-\mu)\,d^3k = \frac{\mathrm{area}}{|\nabla E|}\,
//!   \frac{1}{\mathrm{area}}\int_{\mathrm{polygon}} A\,dS$$
//!
//! uses polygon triangulation + 3‑point K‑quadrature on each sub‑triangle.
//! `build_tetrahedra_3d_diagavg` provides diagonal‑averaged tetrahedralization
//! for restoring $k\to-k$ cancellation of P‑odd quantities.
//!
//! ## Occupancy‑cut (AHC)
//!
//! Hybrid strategy: fully occupied → vertex‑average $\Omega$; partially
//! occupied → K‑quadrature on clipped polygon; empty → skip.  3D uses a
//! sorted‑$\mu$ sweep with binary search for empty/partial/full $\mu$ ranges.
//!
//! ## Thermal convolution
//!
//! $T>0$ integrals use 1D convolution of the $T=0$ result with the thermal
//! window $w(x) = e^x/(1+e^x)^2$, avoiding per‑$T$ recomputation.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use rayon::prelude::*;

use super::kernel::{
    eval_berry_band_at_lam_buf, eval_berry_complex_at_lam_buf, eval_berry_kernel,
};
use super::quadrature::{TET_QUAD_PTS_4, TET_QUAD_WTS_4, TRI_QUAD_PTS_3, TRI_QUAD_WTS_3};
use super::tracking::{
    build_tetrahedra_3d, build_tetrahedra_3d_diagavg, build_triangles_2d_diagavg,
};
use super::types::{SIMPLEX_GAP_TOL, TrackedSimplex, VertexKernel};

const KB_EV_PER_K: f64 = 8.617333262e-5;
const ENERGY_CUT_EPS: f64 = 1e-12;
const FERMI_X_CUT: f64 = 18.0;
const FERMI_X_STEPS: usize = 72;

#[inline]
fn fermi_window_x(x: f64) -> f64 {
    if x.abs() > 50.0 {
        0.0
    } else {
        let ex = x.exp();
        ex / ((1.0 + ex) * (1.0 + ex))
    }
}

#[inline]
fn triangle_area(coords: &Array2<f64>) -> f64 {
    let x0 = coords[[0, 0]];
    let y0 = coords[[0, 1]];
    let x1 = coords[[1, 0]];
    let y1 = coords[[1, 1]];
    let x2 = coords[[2, 0]];
    let y2 = coords[[2, 1]];
    0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs()
}

fn unique_push(points: &mut Vec<([f64; 2], f64)>, point: [f64; 2], amp: f64) {
    for (p, _) in points.iter() {
        let dx = p[0] - point[0];
        let dy = p[1] - point[1];
        if dx * dx + dy * dy < 1e-24 {
            return;
        }
    }
    points.push((point, amp));
}

/// Analytic line-cut integral over one 2D triangle.
///
/// `energy_v[i]` and `amp_v[i]` are the vertex values of the linearly
/// interpolated energy and response amplitude.  The return value is
/// `int_T A(k) delta(E(k)-energy) d^2k` in the coordinate measure of `coords`.
pub fn triangle_line_cut(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    amp_v: [f64; 3],
    energy: f64,
) -> f64 {
    let emin = energy_v.iter().copied().fold(f64::INFINITY, f64::min);
    let emax = energy_v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if energy < emin - ENERGY_CUT_EPS || energy > emax + ENERGY_CUT_EPS {
        return 0.0;
    }

    let x0 = coords[[0, 0]];
    let y0 = coords[[0, 1]];
    let x1 = coords[[1, 0]];
    let y1 = coords[[1, 1]];
    let x2 = coords[[2, 0]];
    let y2 = coords[[2, 1]];
    let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if det.abs() < ENERGY_CUT_EPS {
        return 0.0;
    }

    let de1 = energy_v[1] - energy_v[0];
    let de2 = energy_v[2] - energy_v[0];
    let grad_x = (de1 * (y2 - y0) - de2 * (y1 - y0)) / det;
    let grad_y = ((x1 - x0) * de2 - (x2 - x0) * de1) / det;
    let grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt();
    if grad_norm < ENERGY_CUT_EPS {
        return 0.0;
    }

    let xy = [[x0, y0], [x1, y1], [x2, y2]];
    let mut points: Vec<([f64; 2], f64)> = Vec::with_capacity(3);
    for &(i, j) in &[(0usize, 1usize), (1, 2), (2, 0)] {
        let ei = energy_v[i];
        let ej = energy_v[j];
        let denom = ej - ei;
        if denom.abs() < ENERGY_CUT_EPS {
            continue;
        }
        let t = (energy - ei) / denom;
        if (-ENERGY_CUT_EPS..=1.0 + ENERGY_CUT_EPS).contains(&t) {
            let tc = t.clamp(0.0, 1.0);
            let point = [
                xy[i][0] + tc * (xy[j][0] - xy[i][0]),
                xy[i][1] + tc * (xy[j][1] - xy[i][1]),
            ];
            let amp = amp_v[i] + tc * (amp_v[j] - amp_v[i]);
            unique_push(&mut points, point, amp);
        }
    }
    if points.len() < 2 {
        return 0.0;
    }

    let mut best = (0usize, 1usize);
    let mut best_l2 = -1.0;
    for i in 0..points.len() {
        for j in i + 1..points.len() {
            let dx = points[i].0[0] - points[j].0[0];
            let dy = points[i].0[1] - points[j].0[1];
            let l2 = dx * dx + dy * dy;
            if l2 > best_l2 {
                best_l2 = l2;
                best = (i, j);
            }
        }
    }
    if best_l2 <= ENERGY_CUT_EPS * ENERGY_CUT_EPS {
        return 0.0;
    }

    let length = best_l2.sqrt();
    let amp_avg = 0.5 * (points[best.0].1 + points[best.1].1);
    length * amp_avg / grad_norm
}

// ── K‑quadrature line‑cut helpers ──────────────────────────────────────

/// Find the two intersection points of the iso‑energy line $E=\mu$ with
/// the triangle edges, returning both physical coordinates and barycentrics.
fn find_line_intersections(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    energy: f64,
) -> (Vec<[f64; 2]>, Vec<[f64; 3]>) {
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(3);
    let mut bcs: Vec<[f64; 3]> = Vec::with_capacity(3);
    let xy = [
        [coords[[0, 0]], coords[[0, 1]]],
        [coords[[1, 0]], coords[[1, 1]]],
        [coords[[2, 0]], coords[[2, 1]]],
    ];
    for &(i, j) in &[(0usize, 1usize), (1, 2), (2, 0)] {
        let ei = energy_v[i];
        let ej = energy_v[j];
        let denom = ej - ei;
        if denom.abs() < ENERGY_CUT_EPS {
            continue;
        }
        let t = (energy - ei) / denom;
        if (-ENERGY_CUT_EPS..=1.0 + ENERGY_CUT_EPS).contains(&t) {
            let tc = t.clamp(0.0, 1.0);
            let px = xy[i][0] + tc * (xy[j][0] - xy[i][0]);
            let py = xy[i][1] + tc * (xy[j][1] - xy[i][1]);
            // Deduplicate
            let mut dup = false;
            for k in 0..pts.len() {
                let dx = pts[k][0] - px;
                let dy = pts[k][1] - py;
                if dx * dx + dy * dy < 1e-24 {
                    dup = true;
                    break;
                }
            }
            if !dup {
                pts.push([px, py]);
                let mut lam = [0.0; 3];
                lam[i] = 1.0 - tc;
                lam[j] = tc;
                bcs.push(lam);
            }
        }
    }
    (pts, bcs)
}

/// K‑quadrature line‑cut for the Berry curvature dipole amplitude
/// $A_n = v^c_n \cdot \Omega_n$ along the $E_n=\mu$ line.
///
/// Uses 2‑point Gauss‑Legendre quadrature along the line segment, with
/// barycentric interpolation of $K^{ab}_{nm}$, $E_m$, and $v^c_n$.
fn kquad_line_cut_dipole(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    vdiag_v: [f64; 3],
    energy: f64,
    eta: f64,
    n: usize,
    nsta: usize,
) -> f64 {
    let (pts, bcs) = find_line_intersections(coords, energy_v, energy);
    if pts.len() < 2 {
        return 0.0;
    }

    // |∇E|
    let (x0, y0) = (coords[[0, 0]], coords[[0, 1]]);
    let (x1, y1) = (coords[[1, 0]], coords[[1, 1]]);
    let (x2, y2) = (coords[[2, 0]], coords[[2, 1]]);
    let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if det.abs() < ENERGY_CUT_EPS {
        return 0.0;
    }
    let de1 = energy_v[1] - energy_v[0];
    let de2 = energy_v[2] - energy_v[0];
    let grad_x = (de1 * (y2 - y0) - de2 * (y1 - y0)) / det;
    let grad_y = ((x1 - x0) * de2 - (x2 - x0) * de1) / det;
    let grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt();
    if grad_norm < ENERGY_CUT_EPS {
        return 0.0;
    }

    // Find the two most distant points
    let mut best = (0usize, 1usize);
    let mut best_l2 = -1.0;
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let dx = pts[i][0] - pts[j][0];
            let dy = pts[i][1] - pts[j][1];
            let l2 = dx * dx + dy * dy;
            if l2 > best_l2 {
                best_l2 = l2;
                best = (i, j);
            }
        }
    }
    if best_l2 <= ENERGY_CUT_EPS * ENERGY_CUT_EPS {
        return 0.0;
    }
    let length = best_l2.sqrt();
    let (lam0, lam1) = (bcs[best.0], bcs[best.1]);

    // 2-point Gauss-Legendre along [0,1]
    const SQ3: f64 = 0.5773502691896257; // 1/√3
    let t_vals = [0.5 * (1.0 - SQ3), 0.5 * (1.0 + SQ3)];

    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];
    let mut amp_sum = 0.0;
    for t in &t_vals {
        let lam = [
            (1.0 - t) * lam0[0] + t * lam1[0],
            (1.0 - t) * lam0[1] + t * lam1[1],
            (1.0 - t) * lam0[2] + t * lam1[2],
        ];
        let (_metric, berry) = eval_berry_complex_at_lam_buf(n, bands, kmats, &lam, eta, nsta, &mut e_buf, &mut k_buf);
        let vc = lam[0] * vdiag_v[0] + lam[1] * vdiag_v[1] + lam[2] * vdiag_v[2];
        amp_sum += vc * berry;
    }
    // Gauss weights are 0.5 each
    0.5 * length * amp_sum / grad_norm
}

/// K‑quadrature dipole accumulator (replaces the old linear‑Ω version).
fn accumulate_triangle_dipole_kquad(
    sim: &TrackedSimplex,
    eta: f64,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let area = triangle_area(&sim.coords);
    if area < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / area;
    let nsta = sim.vertices[0].band.len();

    // Borrow vertex data (no clone).
    let v0 = &sim.vertices[0];
    let v1 = &sim.vertices[1];
    let v2 = &sim.vertices[2];
    let bands: [&[f64]; 3] = [v0.band.as_slice().unwrap(), v1.band.as_slice().unwrap(), v2.band.as_slice().unwrap()];
    let kmats: [&Array2<Complex<f64>>; 3] = [&v0.k_ab, &v1.k_ab, &v2.k_ab];
    let vdiags: [&[f64]; 3] = [
        v0.vdiag.as_ref().expect("vdiag required").as_slice().unwrap(),
        v1.vdiag.as_ref().expect("vdiag required").as_slice().unwrap(),
        v2.vdiag.as_ref().expect("vdiag required").as_slice().unwrap(),
    ];
    for n in 0..nsta {
        let e_v = [
            sim.vertices[0].band[n],
            sim.vertices[1].band[n],
            sim.vertices[2].band[n],
        ];
        let vdiag_v = [vdiags[0][n], vdiags[1][n], vdiags[2][n]];

        if beta == 0.0 {
            for im in 0..mu.len() {
                acc[im] += volume_scale
                    * kquad_line_cut_dipole(
                        &sim.coords,
                        e_v,
                        &bands,
                        &kmats,
                        vdiag_v,
                        mu[im],
                        eta,
                        n,
                        nsta,
                    );
            }
        } else {
            let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
            for im in 0..mu.len() {
                let mut sum = 0.0;
                for iq in 0..FERMI_X_STEPS {
                    let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
                    let energy = mu[im] + x / beta;
                    let rho = kquad_line_cut_dipole(
                        &sim.coords,
                        e_v,
                        &bands,
                        &kmats,
                        vdiag_v,
                        energy,
                        eta,
                        n,
                        nsta,
                    );
                    sum += dx * fermi_window_x(x) * rho;
                }
                acc[im] += volume_scale * sum;
            }
        }
    }
}

/// Integrate the Berry-curvature dipole using K‑quadrature energy cuts.
///
/// The per-band amplitude $A_n = v^c_n \cdot \Omega^{ab}_n$ is evaluated
/// via 2‑point Gauss‑Legendre quadrature along the $E_n=\mu$ line inside
/// each triangle.  $K^{ab}_{nm}$, $E_m$, and $v^c_n$ are barycentrically
/// interpolated at each quadrature point so the $1/\Delta^2$ structure is
/// preserved.
pub fn integrate_dipole_energy_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
    eta: f64,
) -> (Array1<f64>, usize) {
    debug_assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(
        k_mesh.len(),
        2,
        "energy-cut dipole currently supports 2D only"
    );
    let beta = if T > 0.0 {
        1.0 / (T * KB_EV_PER_K)
    } else {
        0.0
    };
    let (nx, ny) = (k_mesh[0], k_mesh[1]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let mut acc = Array1::<f64>::zeros(mu.len());
    let mut unsafe_count = 0usize;

    for ix in 0..nx {
        for iy in 0..ny {
            let sims = build_triangles_2d_diagavg(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
            for sim in &sims {
                if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                    unsafe_count += 1;
                }
                accumulate_triangle_dipole_kquad(sim, eta, mu, beta, &mut acc);
            }
        }
    }

    (acc, unsafe_count)
}

// ── Intrinsic NLH energy‑cut ────────────────────────────────────────────

use super::kernel::eval_intrinsic_G_at_lam_buf;

/// K‑quadrature line‑cut for the intrinsic NLH kernel
/// $Q^{ab;c}_n = 2 v^c_n G^{ab}_n - \frac12(v^a_n G^{bc}_n + v^b_n G^{ac}_n)$
/// along the $E_n=\mu$ line.
fn kquad_line_cut_intrinsic(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    bands: &[&[f64]],
    kmat_ab: &[&Array2<Complex<f64>>],
    kmat_bc: &[&Array2<Complex<f64>>],
    kmat_ac: &[&Array2<Complex<f64>>],
    vdiag_c: [f64; 3],
    vdiag_a: [f64; 3],
    vdiag_b: [f64; 3],
    energy: f64,
    n: usize,
    nsta: usize,
) -> f64 {
    let (pts, bcs) = find_line_intersections(coords, energy_v, energy);
    if pts.len() < 2 {
        return 0.0;
    }

    // |∇E|
    let (x0, y0) = (coords[[0, 0]], coords[[0, 1]]);
    let (x1, y1) = (coords[[1, 0]], coords[[1, 1]]);
    let (x2, y2) = (coords[[2, 0]], coords[[2, 1]]);
    let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if det.abs() < ENERGY_CUT_EPS {
        return 0.0;
    }
    let de1 = energy_v[1] - energy_v[0];
    let de2 = energy_v[2] - energy_v[0];
    let grad_x = (de1 * (y2 - y0) - de2 * (y1 - y0)) / det;
    let grad_y = ((x1 - x0) * de2 - (x2 - x0) * de1) / det;
    let grad_norm = (grad_x * grad_x + grad_y * grad_y).sqrt();
    if grad_norm < ENERGY_CUT_EPS {
        return 0.0;
    }

    let mut best = (0usize, 1usize);
    let mut best_l2 = -1.0;
    for i in 0..pts.len() {
        for j in i + 1..pts.len() {
            let dx = pts[i][0] - pts[j][0];
            let dy = pts[i][1] - pts[j][1];
            let l2 = dx * dx + dy * dy;
            if l2 > best_l2 {
                best_l2 = l2;
                best = (i, j);
            }
        }
    }
    if best_l2 <= ENERGY_CUT_EPS * ENERGY_CUT_EPS {
        return 0.0;
    }
    let length = best_l2.sqrt();
    let (lam0, lam1) = (bcs[best.0], bcs[best.1]);

    const SQ3: f64 = 0.5773502691896257;
    let t_vals = [0.5 * (1.0 - SQ3), 0.5 * (1.0 + SQ3)];

    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];
    let mut amp_sum = 0.0;
    for t in &t_vals {
        let lam = [
            (1.0 - t) * lam0[0] + t * lam1[0],
            (1.0 - t) * lam0[1] + t * lam1[1],
            (1.0 - t) * lam0[2] + t * lam1[2],
        ];
        let g_ab = eval_intrinsic_G_at_lam_buf(n, bands, kmat_ab, &lam, nsta, &mut e_buf, &mut k_buf);
        let g_bc = eval_intrinsic_G_at_lam_buf(n, bands, kmat_bc, &lam, nsta, &mut e_buf, &mut k_buf);
        let g_ac = eval_intrinsic_G_at_lam_buf(n, bands, kmat_ac, &lam, nsta, &mut e_buf, &mut k_buf);
        let va = lam[0] * vdiag_a[0] + lam[1] * vdiag_a[1] + lam[2] * vdiag_a[2];
        let vb = lam[0] * vdiag_b[0] + lam[1] * vdiag_b[1] + lam[2] * vdiag_b[2];
        let vc = lam[0] * vdiag_c[0] + lam[1] * vdiag_c[1] + lam[2] * vdiag_c[2];
        let q = 2.0 * vc * g_ab - 0.5 * (va * g_bc + vb * g_ac);
        amp_sum -= q; // sign convention: direct sum returns −Q, so EC returns −∫ δ Q dk
    }
    0.5 * length * amp_sum / grad_norm
}

fn accumulate_triangle_intrinsic_kquad(
    sim: &TrackedSimplex,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let area = triangle_area(&sim.coords);
    if area < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / area;
    let nsta = sim.vertices[0].band.len();

    let v0 = &sim.vertices[0];
    let v1 = &sim.vertices[1];
    let v2 = &sim.vertices[2];
    let bands: [&[f64]; 3] = [v0.band.as_slice().unwrap(), v1.band.as_slice().unwrap(), v2.band.as_slice().unwrap()];
    let kmat_ab: [&Array2<Complex<f64>>; 3] = [&v0.k_ab, &v1.k_ab, &v2.k_ab];
    let kmat_bc: [&Array2<Complex<f64>>; 3] = [
        v0.k_bc.as_ref().expect("k_bc required"),
        v1.k_bc.as_ref().expect("k_bc required"),
        v2.k_bc.as_ref().expect("k_bc required"),
    ];
    let kmat_ac: [&Array2<Complex<f64>>; 3] = [
        v0.k_ac.as_ref().expect("k_ac required"),
        v1.k_ac.as_ref().expect("k_ac required"),
        v2.k_ac.as_ref().expect("k_ac required"),
    ];
    let vdiag_c: [&[f64]; 3] = [
        v0.vdiag.as_ref().expect("vdiag required").as_slice().unwrap(),
        v1.vdiag.as_ref().expect("vdiag required").as_slice().unwrap(),
        v2.vdiag.as_ref().expect("vdiag required").as_slice().unwrap(),
    ];
    let vdiag_a: [&[f64]; 3] = [
        v0.vdiag_a.as_ref().expect("vdiag_a required").as_slice().unwrap(),
        v1.vdiag_a.as_ref().expect("vdiag_a required").as_slice().unwrap(),
        v2.vdiag_a.as_ref().expect("vdiag_a required").as_slice().unwrap(),
    ];
    let vdiag_b: [&[f64]; 3] = [
        v0.vdiag_b.as_ref().expect("vdiag_b required").as_slice().unwrap(),
        v1.vdiag_b.as_ref().expect("vdiag_b required").as_slice().unwrap(),
        v2.vdiag_b.as_ref().expect("vdiag_b required").as_slice().unwrap(),
    ];

    for n in 0..nsta {
        let e_v = [
            sim.vertices[0].band[n],
            sim.vertices[1].band[n],
            sim.vertices[2].band[n],
        ];
        let vc_v = [vdiag_c[0][n], vdiag_c[1][n], vdiag_c[2][n]];
        let va_v = [vdiag_a[0][n], vdiag_a[1][n], vdiag_a[2][n]];
        let vb_v = [vdiag_b[0][n], vdiag_b[1][n], vdiag_b[2][n]];
        let e_min = e_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let e_max = e_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let mu_slice = mu.as_slice().unwrap();
        let (i_start, i_end) = if beta == 0.0 {
            let s = mu_slice.partition_point(|&x| x < e_min - ENERGY_CUT_EPS);
            let e = mu_slice.partition_point(|&x| x <= e_max + ENERGY_CUT_EPS);
            (s, e)
        } else {
            let window = FERMI_X_CUT / beta;
            let s = mu_slice.partition_point(|&x| x < e_min - window - ENERGY_CUT_EPS);
            let e = mu_slice.partition_point(|&x| x <= e_max + window + ENERGY_CUT_EPS);
            (s, e)
        };

        if beta == 0.0 {
            for im in i_start..i_end {
                acc[im] += volume_scale
                    * kquad_line_cut_intrinsic(
                        &sim.coords,
                        e_v,
                        &bands,
                        &kmat_ab,
                        &kmat_bc,
                        &kmat_ac,
                        vc_v,
                        va_v,
                        vb_v,
                        mu[im],
                        n,
                        nsta,
                    );
            }
        } else {
            let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
            for im in i_start..i_end {
                let mut sum = 0.0;
                for iq in 0..FERMI_X_STEPS {
                    let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
                    let energy = mu[im] + x / beta;
                    let rho = kquad_line_cut_intrinsic(
                        &sim.coords,
                        e_v,
                        &bands,
                        &kmat_ab,
                        &kmat_bc,
                        &kmat_ac,
                        vc_v,
                        va_v,
                        vb_v,
                        energy,
                        n,
                        nsta,
                    );
                    sum += dx * fermi_window_x(x) * rho;
                }
                acc[im] += volume_scale * sum;
            }
        }
    }
}

/// 2D intrinsic NLH via K‑quadrature energy‑cut.
pub fn integrate_intrinsic_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
) -> Array1<f64> {
    debug_assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 2);
    let (nx, ny) = (k_mesh[0], k_mesh[1]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let beta = if T > 0.0 {
        1.0 / (T * KB_EV_PER_K)
    } else {
        0.0
    };
    let n_mu = mu.len();
    let mut acc = Array1::<f64>::zeros(n_mu);

    for ix in 0..nx {
        for iy in 0..ny {
            let sims = build_triangles_2d_diagavg(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
            for sim in &sims {
                accumulate_triangle_intrinsic_kquad(sim, mu, beta, &mut acc);
            }
        }
    }

    acc
}

// ── 3D intrinsic NLH energy‑cut ─────────────────────────────────────────

/// Gradient $\nabla E$ (3D) computed from tetrahedron vertex coordinates
/// and energies.  Returns `(gx, gy, gz, |∇E|)`.
fn energy_gradient_3d(coords: &Array2<f64>, de: [f64; 3]) -> (f64, f64, f64, f64) {
    let x0 = coords[[0, 0]];
    let y0 = coords[[0, 1]];
    let z0 = coords[[0, 2]];
    let dx1 = coords[[1, 0]] - x0;
    let dy1 = coords[[1, 1]] - y0;
    let dz1 = coords[[1, 2]] - z0;
    let dx2 = coords[[2, 0]] - x0;
    let dy2 = coords[[2, 1]] - y0;
    let dz2 = coords[[2, 2]] - z0;
    let dx3 = coords[[3, 0]] - x0;
    let dy3 = coords[[3, 1]] - y0;
    let dz3 = coords[[3, 2]] - z0;
    // J = [dr1, dr2, dr3] as columns; solve J^T · grad = de
    // Cramer's rule for 3×3
    let det = dx1 * (dy2 * dz3 - dz2 * dy3) - dy1 * (dx2 * dz3 - dz2 * dx3)
        + dz1 * (dx2 * dy3 - dy2 * dx3);
    if det.abs() < ENERGY_CUT_EPS {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let (e1, e2, e3) = (de[0], de[1], de[2]);
    let gx = ((dy2 * dz3 - dz2 * dy3) * e1
        + (dz1 * dy3 - dy1 * dz3) * e2
        + (dy1 * dz2 - dz1 * dy2) * e3)
        / det;
    let gy = ((dz2 * dx3 - dx2 * dz3) * e1
        + (dx1 * dz3 - dz1 * dx3) * e2
        + (dz1 * dx2 - dx1 * dz2) * e3)
        / det;
    let gz = ((dx2 * dy3 - dy2 * dx3) * e1
        + (dy1 * dx3 - dx1 * dy3) * e2
        + (dx1 * dy2 - dy1 * dx2) * e3)
        / det;
    let norm = (gx * gx + gy * gy + gz * gz).sqrt();
    (gx, gy, gz, norm)
}

/// Find the intersection polygon of the $E=\mu$ plane with a tetrahedron.
/// Returns vertices as barycentric coordinates in the original tet.
fn tet_plane_intersection(energy_v: [f64; 4], mu: f64) -> Vec<[f64; 4]> {
    let eps = 1e-12;

    // Sort by energy
    let mut idx: [usize; 4] = [0, 1, 2, 3];
    idx.sort_by(|&i, &j| energy_v[i].partial_cmp(&energy_v[j]).unwrap());
    let (a, b, c, d) = (idx[0], idx[1], idx[2], idx[3]);

    if mu <= energy_v[a] + eps || mu >= energy_v[d] - eps {
        return vec![];
    }

    let unit = |i: usize| -> [f64; 4] {
        let mut l = [0.0; 4];
        l[i] = 1.0;
        l
    };

    let cut = |i: usize, j: usize| -> [f64; 4] {
        let de = energy_v[j] - energy_v[i];
        let t = if de.abs() < eps {
            0.5
        } else {
            ((mu - energy_v[i]) / de).clamp(0.0, 1.0)
        };
        let mut l = [0.0; 4];
        l[i] = 1.0 - t;
        l[j] = t;
        l
    };

    if mu <= energy_v[b] + eps {
        // 1 below (a), 3 above → triangle: cut(a,b), cut(a,c), cut(a,d)
        vec![cut(a, b), cut(a, c), cut(a, d)]
    } else if mu <= energy_v[c] + eps {
        // 2 below (a,b), 2 above → quadrilateral
        // Order: cut(a,c) → cut(a,d) → cut(b,d) → cut(b,c)
        vec![cut(a, c), cut(a, d), cut(b, d), cut(b, c)]
    } else {
        // 3 below (a,b,c), 1 above → triangle
        // cut(a,d), cut(b,d), cut(c,d) — but order for proper triangulation:
        // Same as case 1 with roles reversed: intersection is the triangle
        // connecting the 3 cut points on edges to the above vertex
        vec![cut(a, d), cut(b, d), cut(c, d)]
    }
}

/// Area of a polygon in 3D defined by barycentric vertices.
/// Triangulates by fan from vertex 0.
fn polygon_area_3d(coords: &Array2<f64>, verts: &[[f64; 4]]) -> f64 {
    if verts.len() < 3 {
        return 0.0;
    }
    let to_xyz = |lam: &[f64; 4]| -> [f64; 3] {
        [
            lam[0] * coords[[0, 0]]
                + lam[1] * coords[[1, 0]]
                + lam[2] * coords[[2, 0]]
                + lam[3] * coords[[3, 0]],
            lam[0] * coords[[0, 1]]
                + lam[1] * coords[[1, 1]]
                + lam[2] * coords[[2, 1]]
                + lam[3] * coords[[3, 1]],
            lam[0] * coords[[0, 2]]
                + lam[1] * coords[[1, 2]]
                + lam[2] * coords[[2, 2]]
                + lam[3] * coords[[3, 2]],
        ]
    };
    let v0 = to_xyz(&verts[0]);
    let mut area = 0.0;
    for i in 1..verts.len() - 1 {
        let v1 = to_xyz(&verts[i]);
        let v2 = to_xyz(&verts[i + 1]);
        let dx1 = v1[0] - v0[0];
        let dy1 = v1[1] - v0[1];
        let dz1 = v1[2] - v0[2];
        let dx2 = v2[0] - v0[0];
        let dy2 = v2[1] - v0[1];
        let dz2 = v2[2] - v0[2];
        let cx = dy1 * dz2 - dz1 * dy2;
        let cy = dz1 * dx2 - dx1 * dz2;
        let cz = dx1 * dy2 - dy1 * dx2;
        area += 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
    }
    area
}

/// Combine sub‑triangle barycentrics $\alpha$ with the polygon vertex
/// barycentrics to get barycentrics in the original tetrahedron.
fn combine_bary_3d(alpha: &[f64; 3], lam: &[[f64; 4]]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for k in 0..3 {
        for i in 0..4 {
            out[i] += alpha[k] * lam[k][i];
        }
    }
    out
}

/// K‑quadrature surface integral for intrinsic NLH over the $E=\mu$ plane
/// intersection with one tetrahedron.
fn kquad_surface_cut_intrinsic(
    coords: &Array2<f64>,
    energy_v: [f64; 4],
    bands: &[&[f64]],
    kmat_ab: &[&Array2<Complex<f64>>],
    kmat_bc: &[&Array2<Complex<f64>>],
    kmat_ac: &[&Array2<Complex<f64>>],
    vdiag_c: [f64; 4],
    vdiag_a: [f64; 4],
    vdiag_b: [f64; 4],
    mu: f64,
    n: usize,
    nsta: usize,
    grad_norm: f64,
) -> f64 {
    let verts = tet_plane_intersection(energy_v, mu);
    if verts.len() < 3 {
        return 0.0;
    }

    // Area of intersection polygon
    let area = polygon_area_3d(coords, &verts);

    // K‑quadrature over polygon (fan triangulation from vertex 0)
    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];
    let mut amp_sum = 0.0;
    for i in 1..verts.len() - 1 {
        let sub_tri = [&verts[0], &verts[i], &verts[i + 1]];
        let sub_area = {
            let lam_ref: [[f64; 4]; 3] = [*sub_tri[0], *sub_tri[1], *sub_tri[2]];
            polygon_area_3d(coords, &lam_ref)
        };
        if sub_area < 1e-30 {
            continue;
        }
        for iq in 0..3 {
            let alpha = &TRI_QUAD_PTS_3[iq];
            let w = TRI_QUAD_WTS_3[iq];
            let lam = combine_bary_3d(alpha, &[*sub_tri[0], *sub_tri[1], *sub_tri[2]]);
            let g_ab = eval_intrinsic_G_at_lam_buf(n, bands, kmat_ab, &lam, nsta, &mut e_buf, &mut k_buf);
            let g_bc = eval_intrinsic_G_at_lam_buf(n, bands, kmat_bc, &lam, nsta, &mut e_buf, &mut k_buf);
            let g_ac = eval_intrinsic_G_at_lam_buf(n, bands, kmat_ac, &lam, nsta, &mut e_buf, &mut k_buf);
            let va = lam[0] * vdiag_a[0]
                + lam[1] * vdiag_a[1]
                + lam[2] * vdiag_a[2]
                + lam[3] * vdiag_a[3];
            let vb = lam[0] * vdiag_b[0]
                + lam[1] * vdiag_b[1]
                + lam[2] * vdiag_b[2]
                + lam[3] * vdiag_b[3];
            let vc = lam[0] * vdiag_c[0]
                + lam[1] * vdiag_c[1]
                + lam[2] * vdiag_c[2]
                + lam[3] * vdiag_c[3];
            let q = 2.0 * vc * g_ab - 0.5 * (va * g_bc + vb * g_ac);
            amp_sum -= sub_area * w * q;
        }
    }

    amp_sum / grad_norm
}

fn accumulate_tetrahedron_intrinsic_kquad(
    sim: &TrackedSimplex,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let _vol = tet_vol_from_pts(
        [sim.coords[[0, 0]], sim.coords[[0, 1]], sim.coords[[0, 2]]],
        [sim.coords[[1, 0]], sim.coords[[1, 1]], sim.coords[[1, 2]]],
        [sim.coords[[2, 0]], sim.coords[[2, 1]], sim.coords[[2, 2]]],
        [sim.coords[[3, 0]], sim.coords[[3, 1]], sim.coords[[3, 2]]],
    );
    if _vol < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / _vol;
    let nsta = sim.vertices[0].band.len();

    let v0 = &sim.vertices[0];
    let v1 = &sim.vertices[1];
    let v2 = &sim.vertices[2];
    let v3 = &sim.vertices[3];
    let bands: [&[f64]; 4] = [v0.band.as_slice().unwrap(), v1.band.as_slice().unwrap(), v2.band.as_slice().unwrap(), v3.band.as_slice().unwrap()];
    let kmat_ab: [&Array2<Complex<f64>>; 4] = [&v0.k_ab, &v1.k_ab, &v2.k_ab, &v3.k_ab];
    let kmat_bc: [&Array2<Complex<f64>>; 4] = [
        v0.k_bc.as_ref().expect("k_bc"),
        v1.k_bc.as_ref().expect("k_bc"),
        v2.k_bc.as_ref().expect("k_bc"),
        v3.k_bc.as_ref().expect("k_bc"),
    ];
    let kmat_ac: [&Array2<Complex<f64>>; 4] = [
        v0.k_ac.as_ref().expect("k_ac"),
        v1.k_ac.as_ref().expect("k_ac"),
        v2.k_ac.as_ref().expect("k_ac"),
        v3.k_ac.as_ref().expect("k_ac"),
    ];
    let vdiag_c: [&[f64]; 4] = [
        v0.vdiag.as_ref().expect("vdiag").as_slice().unwrap(),
        v1.vdiag.as_ref().expect("vdiag").as_slice().unwrap(),
        v2.vdiag.as_ref().expect("vdiag").as_slice().unwrap(),
        v3.vdiag.as_ref().expect("vdiag").as_slice().unwrap(),
    ];
    let vdiag_a: [&[f64]; 4] = [
        v0.vdiag_a.as_ref().expect("vdiag_a").as_slice().unwrap(),
        v1.vdiag_a.as_ref().expect("vdiag_a").as_slice().unwrap(),
        v2.vdiag_a.as_ref().expect("vdiag_a").as_slice().unwrap(),
        v3.vdiag_a.as_ref().expect("vdiag_a").as_slice().unwrap(),
    ];
    let vdiag_b: [&[f64]; 4] = [
        v0.vdiag_b.as_ref().expect("vdiag_b").as_slice().unwrap(),
        v1.vdiag_b.as_ref().expect("vdiag_b").as_slice().unwrap(),
        v2.vdiag_b.as_ref().expect("vdiag_b").as_slice().unwrap(),
        v3.vdiag_b.as_ref().expect("vdiag_b").as_slice().unwrap(),
    ];

    let n_mu = mu.len();
    for n in 0..nsta {
        let e_v = [
            sim.vertices[0].band[n],
            sim.vertices[1].band[n],
            sim.vertices[2].band[n],
            sim.vertices[3].band[n],
        ];
        let vc_v = [vdiag_c[0][n], vdiag_c[1][n], vdiag_c[2][n], vdiag_c[3][n]];
        let va_v = [vdiag_a[0][n], vdiag_a[1][n], vdiag_a[2][n], vdiag_a[3][n]];
        let vb_v = [vdiag_b[0][n], vdiag_b[1][n], vdiag_b[2][n], vdiag_b[3][n]];
        let e_min = e_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let e_max = e_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Precompute |∇E| once per band (same for all μ).
        let de = [e_v[1] - e_v[0], e_v[2] - e_v[0], e_v[3] - e_v[0]];
        let (_, _, _, grad_norm) = energy_gradient_3d(&sim.coords, de);
        if grad_norm < ENERGY_CUT_EPS {
            continue;
        }

        let mu_slice = mu.as_slice().unwrap();
        let (i_start, i_end) = if beta == 0.0 {
            let s = mu_slice.partition_point(|&x| x < e_min - ENERGY_CUT_EPS);
            let e = mu_slice.partition_point(|&x| x <= e_max + ENERGY_CUT_EPS);
            (s, e)
        } else {
            let window = FERMI_X_CUT / beta;
            let s = mu_slice.partition_point(|&x| x < e_min - window - ENERGY_CUT_EPS);
            let e = mu_slice.partition_point(|&x| x <= e_max + window + ENERGY_CUT_EPS);
            (s, e)
        };

        if beta == 0.0 {
            for im in i_start..i_end {
                acc[im] += volume_scale
                    * kquad_surface_cut_intrinsic(
                        &sim.coords,
                        e_v,
                        &bands,
                        &kmat_ab,
                        &kmat_bc,
                        &kmat_ac,
                        vc_v,
                        va_v,
                        vb_v,
                        mu[im],
                        n,
                        nsta,
                        grad_norm,
                    );
            }
        } else {
            let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
            for im in i_start..i_end {
                let mut sum = 0.0;
                for iq in 0..FERMI_X_STEPS {
                    let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
                    let energy = mu[im] + x / beta;
                    let rho = kquad_surface_cut_intrinsic(
                        &sim.coords,
                        e_v,
                        &bands,
                        &kmat_ab,
                        &kmat_bc,
                        &kmat_ac,
                        vc_v,
                        va_v,
                        vb_v,
                        energy,
                        n,
                        nsta,
                        grad_norm,
                    );
                    sum += dx * fermi_window_x(x) * rho;
                }
                acc[im] += volume_scale * sum;
            }
        }
    }
}

/// 3D intrinsic NLH via K‑quadrature surface energy‑cut.
pub fn integrate_intrinsic_cut_3d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
) -> Array1<f64> {
    debug_assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 3);
    let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let inv_nz = 1.0 / nz as f64;
    let beta = if T > 0.0 {
        1.0 / (T * KB_EV_PER_K)
    } else {
        0.0
    };
    let mut acc = Array1::<f64>::zeros(mu.len());

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let sims = build_tetrahedra_3d_diagavg(
                    ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts,
                );
                for sim in &sims {
                    accumulate_tetrahedron_intrinsic_kquad(sim, mu, beta, &mut acc);
                }
            }
        }
    }

    acc
}

// ── Fermi‑window energy‑cut (AHC) ───────────────────────────────────────

/// Map barycentric coordinates in the original triangle to physical coords.
fn bary_to_phys_2d(coords: &Array2<f64>, lam: &[f64; 3]) -> [f64; 2] {
    [
        lam[0] * coords[[0, 0]] + lam[1] * coords[[1, 0]] + lam[2] * coords[[2, 0]],
        lam[0] * coords[[0, 1]] + lam[1] * coords[[1, 1]] + lam[2] * coords[[2, 1]],
    ]
}

/// Area of a sub‑triangle defined by 3 barycentric points.
fn sub_tri_area_2d(coords: &Array2<f64>, tri: &[[f64; 3]; 3]) -> f64 {
    let p0 = bary_to_phys_2d(coords, &tri[0]);
    let p1 = bary_to_phys_2d(coords, &tri[1]);
    let p2 = bary_to_phys_2d(coords, &tri[2]);
    0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])).abs()
}

/// Clip the original triangle by $E_n(k)\le\mu$ and return the occupied
/// polygon as 1–2 sub‑triangles, each vertex given by its barycentric
/// coordinates in the original triangle.
fn clip_triangle(energy_v: [f64; 3], mu: f64) -> Vec<[[f64; 3]; 3]> {
    let eps = 1e-12;
    let e_min = energy_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = energy_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if mu <= e_min + eps {
        return vec![];
    }
    if mu >= e_max - eps {
        return vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]];
    }

    let mut idx: [usize; 3] = [0, 1, 2];
    idx.sort_by(|&i, &j| energy_v[i].partial_cmp(&energy_v[j]).unwrap());
    let (a, b, c) = (idx[0], idx[1], idx[2]);

    // Barycentric unit vector at original vertex i
    let unit = |i: usize| -> [f64; 3] {
        let mut l = [0.0; 3];
        l[i] = 1.0;
        l
    };

    // Barycentrics of the intersection on edge i→j at energy μ
    let cut = |i: usize, j: usize| -> [f64; 3] {
        let de = energy_v[j] - energy_v[i];
        let t = if de.abs() < eps {
            0.5
        } else {
            ((mu - energy_v[i]) / de).clamp(0.0, 1.0)
        };
        let mut l = [0.0; 3];
        l[i] = 1.0 - t;
        l[j] = t;
        l
    };

    if mu <= energy_v[b] + eps {
        // 1 vertex below (a) → sub‑triangle {a, cut(a,b), cut(a,c)}
        vec![[unit(a), cut(a, b), cut(a, c)]]
    } else {
        // 2 vertices below (a,b) → quadrilateral → two sub‑triangles
        vec![
            [unit(a), unit(b), cut(a, c)],
            [unit(b), cut(a, c), cut(b, c)],
        ]
    }
}

/// Combine sub‑triangle barycentrics $\alpha$ with vertex barycentrics
/// $\lambda_i$ to get the overall barycentrics in the original triangle:
/// $\lambda = \sum_i \alpha_i \lambda_i$.
fn combine_bary(alpha: &[f64; 3], lam0: &[f64; 3], lam1: &[f64; 3], lam2: &[f64; 3]) -> [f64; 3] {
    [
        alpha[0] * lam0[0] + alpha[1] * lam1[0] + alpha[2] * lam2[0],
        alpha[0] * lam0[1] + alpha[1] * lam1[1] + alpha[2] * lam2[1],
        alpha[0] * lam0[2] + alpha[1] * lam1[2] + alpha[2] * lam2[2],
    ]
}

/// Hybrid integration: vertex‑average $\Omega$ for fully‑occupied / empty
/// triangles; K‑quadrature on clipped polygons for partially‑occupied ones.
///
/// Returns the integral in the coordinate measure of `coords`.
fn triangle_occupied_hybrid(
    coords: &Array2<f64>,
    e_v: [f64; 3],
    omega_vertex: [f64; 3],
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    mu: f64,
    eta: f64,
    n: usize,
    nsta: usize,
) -> f64 {
    let eps = ENERGY_CUT_EPS;
    let e_min = e_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = e_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let area = triangle_area(coords);

    if mu <= e_min + eps {
        return 0.0;
    }
    if mu >= e_max - eps {
        // Full occupancy: vertex‑average Ω preserves gap quantization
        return area * (omega_vertex[0] + omega_vertex[1] + omega_vertex[2]) / 3.0;
    }

    // Partial occupancy: clipped polygon + K‑quadrature
    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];
    let sub_tris = clip_triangle(e_v, mu);
    let mut total = 0.0;
    for sub_tri in &sub_tris {
        let sub_area = sub_tri_area_2d(coords, sub_tri);
        if sub_area < 1e-30 {
            continue;
        }
        for iq in 0..3 {
            let alpha = &TRI_QUAD_PTS_3[iq];
            let w = TRI_QUAD_WTS_3[iq];
            let lam = combine_bary(alpha, &sub_tri[0], &sub_tri[1], &sub_tri[2]);
            total += sub_area
                * w
                * eval_berry_band_at_lam_buf(
                    n, bands, kmats, &lam, eta, nsta, &mut e_buf, &mut k_buf,
                );
        }
    }
    total
}

/// T=0 occupancy‑cut with hybrid vertex‑Ω / K‑quadrature.
fn integrate_fermi_cut_2d_t0(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    eta: f64,
) -> Array1<f64> {
    let (nx, ny) = (k_mesh[0], k_mesh[1]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let n_mu = mu.len();
    let mut result = Array1::<f64>::zeros(n_mu);

    for ix in 0..nx {
        for iy in 0..ny {
            let sims = build_triangles_2d_diagavg(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
            for sim in &sims {
                let area = triangle_area(&sim.coords);
                if area < ENERGY_CUT_EPS {
                    continue;
                }
                let volume_scale = sim.volume / area;
                let nsta = sim.vertices[0].band.len();

                // Precompute vertex Ω (shared across μ).
                let (_g0, o0) = eval_berry_kernel(
                    sim.vertices[0].band.as_slice().unwrap(),
                    &sim.vertices[0].k_ab,
                    eta,
                    nsta,
                );
                let (_g1, o1) = eval_berry_kernel(
                    sim.vertices[1].band.as_slice().unwrap(),
                    &sim.vertices[1].k_ab,
                    eta,
                    nsta,
                );
                let (_g2, o2) = eval_berry_kernel(
                    sim.vertices[2].band.as_slice().unwrap(),
                    &sim.vertices[2].k_ab,
                    eta,
                    nsta,
                );

                // Borrow band energies and K matrices for K‑quadrature.
                let v0 = &sim.vertices[0];
                let v1 = &sim.vertices[1];
                let v2 = &sim.vertices[2];
                let bands: [&[f64]; 3] = [v0.band.as_slice().unwrap(), v1.band.as_slice().unwrap(), v2.band.as_slice().unwrap()];
                let kmats: [&Array2<Complex<f64>>; 3] = [&v0.k_ab, &v1.k_ab, &v2.k_ab];

                for n in 0..nsta {
                    let e_v = [
                        sim.vertices[0].band[n],
                        sim.vertices[1].band[n],
                        sim.vertices[2].band[n],
                    ];
                    let omega_v = [o0[n], o1[n], o2[n]];
                    let e_min = e_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    let e_max = e_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let full_val = area * (omega_v[0] + omega_v[1] + omega_v[2]) / 3.0;

                    for im in 0..n_mu {
                        let m = mu[im];
                        let contrib = if m <= e_min + ENERGY_CUT_EPS {
                            0.0
                        } else if m >= e_max - ENERGY_CUT_EPS {
                            full_val
                        } else {
                            triangle_occupied_hybrid(
                                &sim.coords,
                                e_v,
                                omega_v,
                                &bands,
                                &kmats,
                                m,
                                eta,
                                n,
                                nsta,
                            )
                        };
                        result[im] += volume_scale * contrib;
                    }
                }
            }
        }
    }

    result
}

/// 2D Fermi‑window integration via per‑μ triangle occupancy cut.
///
/// At $T=0$ the occupation is a step function; the integral is computed
/// exactly within each triangle (no energy binning).  At $T>0$ the
/// $T=0$ result is thermally convolved:
///
/// $$\sigma_T(\mu) = \int_{-\infty}^\infty w(x)\,\sigma_0(\mu + x/\beta)\,dx,
/// \qquad w(x)=\frac{e^x}{(1+e^x)^2}$$
///
/// Returns $\sigma(\mu_i)$ for each $\mu$ in `mu` (fractional BZ volume;
/// divide by $\det(L)$ for Cartesian).
pub fn integrate_fermi_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
    eta: f64,
) -> Array1<f64> {
    debug_assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 2);

    if T == 0.0 {
        return integrate_fermi_cut_2d_t0(all_pts, k_mesh, mu, eta);
    }

    // T>0: thermal convolution of the T=0 result.
    let beta = 1.0 / (T * KB_EV_PER_K);
    let x_max = 12.0; // w(x) < 6e-6 for |x| > 12
    let mu_min = mu.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mu_max = mu.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mu_lo = mu_min - x_max / beta;
    let mu_hi = mu_max + x_max / beta;
    let dmu_fine = 0.1 / beta; // fine grid for accurate linear interpolation
    let n_ext = ((mu_hi - mu_lo) / dmu_fine).ceil() as usize + 1;
    let mu_ext = Array1::linspace(mu_lo, mu_hi, n_ext);

    let sigma0 = integrate_fermi_cut_2d_t0(all_pts, k_mesh, &mu_ext, eta);

    let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
    let result: Vec<f64> = mu
        .into_par_iter()
        .map(|&m| {
            let mut sum = 0.0;
            for iq in 0..FERMI_X_STEPS {
                let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
                let w = fermi_window_x(x);
                let e_target = m + x / beta;
                let i_f = (e_target - mu_lo) / dmu_fine;
                let i_lo = (i_f.floor() as isize).max(0) as usize;
                let i_hi = (i_lo + 1).min(n_ext - 1);
                if i_hi > i_lo {
                    let t = i_f - i_lo as f64;
                    let val = sigma0[i_lo] + t * (sigma0[i_hi] - sigma0[i_lo]);
                    sum += dx * w * val;
                }
            }
            sum
        })
        .collect();
    Array1::from_vec(result)
}

/// Diagnostic counters for hybrid energy‑cut.
#[derive(Default, Clone)]
pub struct FermiCutCounts {
    pub empty: usize,
    pub full: usize,
    pub partial: usize,
}

#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(debug_assertions)]
static CNT_EMPTY: AtomicUsize = AtomicUsize::new(0);
#[cfg(debug_assertions)]
static CNT_FULL: AtomicUsize = AtomicUsize::new(0);
#[cfg(debug_assertions)]
static CNT_PARTIAL: AtomicUsize = AtomicUsize::new(0);

/// Read and reset the internal hybrid energy‑cut counters (debug builds only).
/// In release builds returns all zeros.
pub fn read_reset_fermi_cut_counts() -> FermiCutCounts {
    #[cfg(debug_assertions)]
    {
        FermiCutCounts {
            empty: CNT_EMPTY.swap(0, Ordering::Relaxed),
            full: CNT_FULL.swap(0, Ordering::Relaxed),
            partial: CNT_PARTIAL.swap(0, Ordering::Relaxed),
        }
    }
    #[cfg(not(debug_assertions))]
    {
        FermiCutCounts::default()
    }
}

// ── 3D tetrahedron occupancy cut ─────────────────────────────────────────

#[inline]
fn tet_vol_from_pts(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let (x0, y0, z0) = (p0[0], p0[1], p0[2]);
    let (x1, y1, z1) = (p1[0], p1[1], p1[2]);
    let (x2, y2, z2) = (p2[0], p2[1], p2[2]);
    let (x3, y3, z3) = (p3[0], p3[1], p3[2]);
    let det = (x1 - x0) * ((y2 - y0) * (z3 - z0) - (z2 - z0) * (y3 - y0))
        - (y1 - y0) * ((x2 - x0) * (z3 - z0) - (z2 - z0) * (x3 - x0))
        + (z1 - z0) * ((x2 - x0) * (y3 - y0) - (y2 - y0) * (x3 - x0));
    det.abs() / 6.0
}

/// Map 4 barycentrics to physical 3D coords.
fn bary_to_phys_3d(coords: &Array2<f64>, lam: &[f64; 4]) -> [f64; 3] {
    [
        lam[0] * coords[[0, 0]]
            + lam[1] * coords[[1, 0]]
            + lam[2] * coords[[2, 0]]
            + lam[3] * coords[[3, 0]],
        lam[0] * coords[[0, 1]]
            + lam[1] * coords[[1, 1]]
            + lam[2] * coords[[2, 1]]
            + lam[3] * coords[[3, 1]],
        lam[0] * coords[[0, 2]]
            + lam[1] * coords[[1, 2]]
            + lam[2] * coords[[2, 2]]
            + lam[3] * coords[[3, 2]],
    ]
}

/// Volume of a sub‑tet defined by 4 barycentric vertices.
fn sub_tet_vol_3d(coords: &Array2<f64>, sub_lam: &[[f64; 4]; 4]) -> f64 {
    tet_vol_from_pts(
        bary_to_phys_3d(coords, &sub_lam[0]),
        bary_to_phys_3d(coords, &sub_lam[1]),
        bary_to_phys_3d(coords, &sub_lam[2]),
        bary_to_phys_3d(coords, &sub_lam[3]),
    )
}

/// Combine sub‑tet barycentrics $\alpha$ (4‑vector) with vertex
/// barycentrics $\lambda_i$ to get barycentrics in the original tet.
fn combine_bary_4(alpha: &[f64; 4], lam: &[[f64; 4]; 4]) -> [f64; 4] {
    let mut out = [0.0; 4];
    for k in 0..4 {
        for i in 0..4 {
            out[i] += alpha[k] * lam[k][i];
        }
    }
    out
}

/// K‑quadrature over a single sub‑tetrahedron defined by barycentric vertices.
fn sub_tet_k_quad(
    sub_lam: &[[f64; 4]; 4],
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    n: usize,
    eta: f64,
    nsta: usize,
    coords: &Array2<f64>,
) -> f64 {
    let sub_vol = sub_tet_vol_3d(coords, sub_lam);
    if sub_vol < 1e-30 {
        return 0.0;
    }
    let mut total = 0.0;
    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];
    for iq in 0..4 {
        let alpha = &TET_QUAD_PTS_4[iq];
        let w = TET_QUAD_WTS_4[iq];
        let lam = combine_bary_4(alpha, sub_lam);
        total += sub_vol * w * eval_berry_band_at_lam_buf(n, bands, kmats, &lam, eta, nsta, &mut e_buf, &mut k_buf);
    }
    total
}

/// Hybrid 3D: vertex‑average $\Omega$ for full/empty; K‑quadrature for partial.
fn tetrahedron_occupied_hybrid(
    coords: &Array2<f64>,
    e_v: [f64; 4],
    omega_v: [f64; 4],
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    mu: f64,
    eta: f64,
    n: usize,
    nsta: usize,
) -> f64 {
    let eps = ENERGY_CUT_EPS;
    let full_vol = tet_vol_from_pts(
        [coords[[0, 0]], coords[[0, 1]], coords[[0, 2]]],
        [coords[[1, 0]], coords[[1, 1]], coords[[1, 2]]],
        [coords[[2, 0]], coords[[2, 1]], coords[[2, 2]]],
        [coords[[3, 0]], coords[[3, 1]], coords[[3, 2]]],
    );
    if full_vol < eps {
        return 0.0;
    }
    let e_min = e_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = e_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if mu <= e_min + eps {
        return 0.0;
    }
    if mu >= e_max - eps {
        return full_vol * (omega_v[0] + omega_v[1] + omega_v[2] + omega_v[3]) / 4.0;
    }

    // Sort by energy: e[a] ≤ e[b] ≤ e[c] ≤ e[d]
    let mut idx: [usize; 4] = [0, 1, 2, 3];
    idx.sort_by(|&i, &j| e_v[i].partial_cmp(&e_v[j]).unwrap());
    let (a, b, c, d) = (idx[0], idx[1], idx[2], idx[3]);

    let unit = |i: usize| -> [f64; 4] {
        let mut l = [0.0; 4];
        l[i] = 1.0;
        l
    };

    let cut_bary = |i: usize, j: usize| -> [f64; 4] {
        let de = e_v[j] - e_v[i];
        let t = if de.abs() < eps {
            0.5
        } else {
            ((mu - e_v[i]) / de).clamp(0.0, 1.0)
        };
        let mut l = [0.0; 4];
        l[i] = 1.0 - t;
        l[j] = t;
        l
    };

    if mu <= e_v[b] + eps {
        // 1 below (a) → sub‑tet {a, cut(a,b), cut(a,c), cut(a,d)}
        sub_tet_k_quad(
            &[unit(a), cut_bary(a, b), cut_bary(a, c), cut_bary(a, d)],
            bands,
            kmats,
            n,
            eta,
            nsta,
            coords,
        )
    } else if mu <= e_v[c] + eps {
        // 2 below (a,b) → 3 sub‑tets via diagonal p_ac → p_bd
        sub_tet_k_quad(
            &[unit(a), unit(b), cut_bary(a, c), cut_bary(b, d)],
            bands,
            kmats,
            n,
            eta,
            nsta,
            coords,
        ) + sub_tet_k_quad(
            &[unit(a), cut_bary(a, c), cut_bary(a, d), cut_bary(b, d)],
            bands,
            kmats,
            n,
            eta,
            nsta,
            coords,
        ) + sub_tet_k_quad(
            &[unit(b), cut_bary(b, c), cut_bary(b, d), cut_bary(a, c)],
            bands,
            kmats,
            n,
            eta,
            nsta,
            coords,
        )
    } else {
        // 3 below (a,b,c) → full − sub‑tet(d, cut(a,d), cut(b,d), cut(c,d))
        let full_val = full_vol * (omega_v[0] + omega_v[1] + omega_v[2] + omega_v[3]) / 4.0;
        full_val
            - sub_tet_k_quad(
                &[unit(d), cut_bary(a, d), cut_bary(b, d), cut_bary(c, d)],
                bands,
                kmats,
                n,
                eta,
                nsta,
                coords,
            )
    }
}

/// T=0 3D hybrid occupancy‑cut integration.
fn integrate_fermi_cut_3d_t0(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    eta: f64,
) -> Array1<f64> {
    let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let inv_nz = 1.0 / nz as f64;
    let n_mu = mu.len();
    let mut result = Array1::<f64>::zeros(n_mu);

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let sims =
                    build_tetrahedra_3d(ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts);
                for sim in &sims {
                    let vol = tet_vol_from_pts(
                        [sim.coords[[0, 0]], sim.coords[[0, 1]], sim.coords[[0, 2]]],
                        [sim.coords[[1, 0]], sim.coords[[1, 1]], sim.coords[[1, 2]]],
                        [sim.coords[[2, 0]], sim.coords[[2, 1]], sim.coords[[2, 2]]],
                        [sim.coords[[3, 0]], sim.coords[[3, 1]], sim.coords[[3, 2]]],
                    );
                    if vol < ENERGY_CUT_EPS {
                        continue;
                    }
                    let volume_scale = sim.volume / vol;
                    let nsta = sim.vertices[0].band.len();

                    // Precompute vertex Ω.
                    let (_g0, o0) = eval_berry_kernel(
                        sim.vertices[0].band.as_slice().unwrap(),
                        &sim.vertices[0].k_ab,
                        eta,
                        nsta,
                    );
                    let (_g1, o1) = eval_berry_kernel(
                        sim.vertices[1].band.as_slice().unwrap(),
                        &sim.vertices[1].k_ab,
                        eta,
                        nsta,
                    );
                    let (_g2, o2) = eval_berry_kernel(
                        sim.vertices[2].band.as_slice().unwrap(),
                        &sim.vertices[2].k_ab,
                        eta,
                        nsta,
                    );
                    let (_g3, o3) = eval_berry_kernel(
                        sim.vertices[3].band.as_slice().unwrap(),
                        &sim.vertices[3].k_ab,
                        eta,
                        nsta,
                    );

                    let v0 = &sim.vertices[0];
                    let v1 = &sim.vertices[1];
                    let v2 = &sim.vertices[2];
                    let v3 = &sim.vertices[3];
                    let bands: [&[f64]; 4] = [v0.band.as_slice().unwrap(), v1.band.as_slice().unwrap(), v2.band.as_slice().unwrap(), v3.band.as_slice().unwrap()];
                    let kmats: [&Array2<Complex<f64>>; 4] = [&v0.k_ab, &v1.k_ab, &v2.k_ab, &v3.k_ab];

                    for n in 0..nsta {
                        let e_v = [
                            sim.vertices[0].band[n],
                            sim.vertices[1].band[n],
                            sim.vertices[2].band[n],
                            sim.vertices[3].band[n],
                        ];
                        let omega_v = [o0[n], o1[n], o2[n], o3[n]];
                        let e_min = e_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                        let e_max = e_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let full_val =
                            vol * (omega_v[0] + omega_v[1] + omega_v[2] + omega_v[3]) / 4.0;

                        // Sorted-μ sweep: binary search for break points.
                        let mu_slice = mu.as_slice().unwrap();
                        let i_partial = mu_slice.partition_point(|&x| x <= e_min + ENERGY_CUT_EPS);
                        let i_full = mu_slice.partition_point(|&x| x < e_max - ENERGY_CUT_EPS);

                        #[cfg(debug_assertions)]
                        {
                            CNT_EMPTY.fetch_add(i_partial, Ordering::Relaxed);
                            CNT_FULL.fetch_add(n_mu.saturating_sub(i_full), Ordering::Relaxed);
                            CNT_PARTIAL
                                .fetch_add(i_full.saturating_sub(i_partial), Ordering::Relaxed);
                        }

                        // Empty region μ[0..i_partial]: skip (zero contribution).

                        // Partial region μ[i_partial..i_full]: K‑quadrature.
                        for im in i_partial..i_full {
                            result[im] += volume_scale
                                * tetrahedron_occupied_hybrid(
                                    &sim.coords,
                                    e_v,
                                    omega_v,
                                    &bands,
                                    &kmats,
                                    mu[im],
                                    eta,
                                    n,
                                    nsta,
                                );
                        }

                        // Full region μ[i_full..]: range add.
                        if i_full < n_mu {
                            let add = volume_scale * full_val;
                            for im in i_full..n_mu {
                                result[im] += add;
                            }
                        }
                    }
                }
            }
        }
    }

    result
}

/// 3D Fermi‑window integration via per‑μ tetrahedron occupancy cut.
///
/// Same algorithm as [`integrate_fermi_cut_2d`] but for tetrahedra.
/// At $T=0$ each tetrahedron's linearly‑interpolated $E(k)$ and $\Omega(k)$
/// are integrated exactly over the occupied sub‑region; $T>0$ uses
/// thermal convolution of the $T=0$ result.
pub fn integrate_fermi_cut_3d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
    eta: f64,
) -> Array1<f64> {
    debug_assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 3);

    if T == 0.0 {
        return integrate_fermi_cut_3d_t0(all_pts, k_mesh, mu, eta);
    }

    // T>0: thermal convolution
    let beta = 1.0 / (T * KB_EV_PER_K);
    let x_max = 12.0;
    let mu_min = mu.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mu_max = mu.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mu_lo = mu_min - x_max / beta;
    let mu_hi = mu_max + x_max / beta;
    let dmu_fine = 0.1 / beta;
    let n_ext = ((mu_hi - mu_lo) / dmu_fine).ceil() as usize + 1;
    let mu_ext = Array1::linspace(mu_lo, mu_hi, n_ext);

    let sigma0 = integrate_fermi_cut_3d_t0(all_pts, k_mesh, &mu_ext, eta);

    let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
    let result: Vec<f64> = mu
        .into_par_iter()
        .map(|&m| {
            let mut sum = 0.0;
            for iq in 0..FERMI_X_STEPS {
                let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
                let w = fermi_window_x(x);
                let e_target = m + x / beta;
                let i_f = (e_target - mu_lo) / dmu_fine;
                let i_lo = (i_f.floor() as isize).max(0) as usize;
                let i_hi = (i_lo + 1).min(n_ext - 1);
                if i_hi > i_lo {
                    let t = i_f - i_lo as f64;
                    let val = sigma0[i_lo] + t * (sigma0[i_hi] - sigma0[i_lo]);
                    sum += dx * w * val;
                }
            }
            sum
        })
        .collect();
    Array1::from_vec(result)
}
