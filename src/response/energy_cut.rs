//! Energy-cut integration for Fermi-surface weighted response functions.
//!
//! This module avoids sampling the narrow finite-temperature Fermi window by
//! quadrature points in k-space.  For a triangle with linearly interpolated
//! band energy `E(k)` and scalar response amplitude `A(k)`, it evaluates the
//! iso-energy line integral
//!
//! ```text
//! rho_A(E) = int_T A(k) delta(E(k) - E) d^2k
//!          = |p - q| / |grad E| * (A(p) + A(q)) / 2
//! ```
//!
//! where `p` and `q` are the two intersections of the level set with the
//! triangle edges.  Finite temperature is then a one-dimensional convolution
//! with `beta f(1-f)`.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use rayon::prelude::*;

use super::kernel::{eval_berry_band_at_lam, eval_berry_kernel};
use super::quadrature::{TET_QUAD_PTS_4, TET_QUAD_WTS_4, TRI_QUAD_PTS_3, TRI_QUAD_WTS_3};
use super::tracking::{build_tetrahedra_3d, build_triangles_2d_diagavg};
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

fn triangle_band_values(sim: &TrackedSimplex, eta: f64) -> Option<Vec<([f64; 3], [f64; 3])>> {
    if sim.vertices.len() != 3 {
        return None;
    }
    let nsta = sim.vertices[0].band.len();
    let mut out = vec![([0.0; 3], [0.0; 3]); nsta];
    for iv in 0..3 {
        let v = &sim.vertices[iv];
        let vdiag = v.vdiag.as_ref().expect(
            "VertexKernel.vdiag is None — call compute_velocity_kernel with dir_c to populate it",
        );
        let band_q = v.band.to_vec();
        let (_metric, omega) = eval_berry_kernel(&band_q, &v.k_ab, eta, nsta);
        for n in 0..nsta {
            out[n].0[iv] = v.band[n];
            out[n].1[iv] = vdiag[n] * omega[n];
        }
    }
    Some(out)
}

fn accumulate_triangle_energy_cut(
    sim: &TrackedSimplex,
    eta: f64,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let Some(values) = triangle_band_values(sim, eta) else {
        return;
    };
    let area = triangle_area(&sim.coords);
    if area < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / area;

    for (energy_v, amp_v) in values {
        if beta == 0.0 {
            for im in 0..mu.len() {
                acc[im] += volume_scale * triangle_line_cut(&sim.coords, energy_v, amp_v, mu[im]);
            }
        } else {
            let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
            for im in 0..mu.len() {
                let mut sum = 0.0;
                for iq in 0..FERMI_X_STEPS {
                    let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
                    let energy = mu[im] + x / beta;
                    let rho = triangle_line_cut(&sim.coords, energy_v, amp_v, energy);
                    sum += dx * fermi_window_x(x) * rho;
                }
                acc[im] += volume_scale * sum;
            }
        }
    }
}

/// Integrate the Berry-curvature dipole using analytic 2D energy cuts.
///
/// The per-band scalar amplitude is
///
/// ```text
/// A_n(k) = v^c_n(k) * Omega^{ab}_n(k)
/// ```
///
/// with `E_n(k)` and `A_n(k)` linearly interpolated on each triangle.  In 2D
/// the two possible square diagonals are averaged by `build_triangles_2d_diagavg`.
pub fn integrate_dipole_energy_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
    eta: f64,
) -> (Array1<f64>, usize) {
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
                accumulate_triangle_energy_cut(sim, eta, mu, beta, &mut acc);
            }
        }
    }

    (acc, unsafe_count)
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
    bands: &[Vec<f64>],
    kmats: &[Array2<Complex<f64>>],
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
            total += sub_area * w * eval_berry_band_at_lam(n, bands, kmats, &lam, eta, nsta);
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
                    &sim.vertices[0].band.to_vec(),
                    &sim.vertices[0].k_ab,
                    eta,
                    nsta,
                );
                let (_g1, o1) = eval_berry_kernel(
                    &sim.vertices[1].band.to_vec(),
                    &sim.vertices[1].k_ab,
                    eta,
                    nsta,
                );
                let (_g2, o2) = eval_berry_kernel(
                    &sim.vertices[2].band.to_vec(),
                    &sim.vertices[2].k_ab,
                    eta,
                    nsta,
                );

                // Pre-extract band energies and K matrices for K‑quadrature.
                let bands: [Vec<f64>; 3] = [
                    sim.vertices[0].band.to_vec(),
                    sim.vertices[1].band.to_vec(),
                    sim.vertices[2].band.to_vec(),
                ];
                let kmats: [Array2<Complex<f64>>; 3] = [
                    sim.vertices[0].k_ab.clone(),
                    sim.vertices[1].k_ab.clone(),
                    sim.vertices[2].k_ab.clone(),
                ];

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
    bands: &[Vec<f64>],
    kmats: &[Array2<Complex<f64>>],
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
    for iq in 0..4 {
        let alpha = &TET_QUAD_PTS_4[iq];
        let w = TET_QUAD_WTS_4[iq];
        let lam = combine_bary_4(alpha, sub_lam);
        total += sub_vol * w * eval_berry_band_at_lam(n, bands, kmats, &lam, eta, nsta);
    }
    total
}

/// Hybrid 3D: vertex‑average $\Omega$ for full/empty; K‑quadrature for partial.
fn tetrahedron_occupied_hybrid(
    coords: &Array2<f64>,
    e_v: [f64; 4],
    omega_v: [f64; 4],
    bands: &[Vec<f64>],
    kmats: &[Array2<Complex<f64>>],
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
                        &sim.vertices[0].band.to_vec(),
                        &sim.vertices[0].k_ab,
                        eta,
                        nsta,
                    );
                    let (_g1, o1) = eval_berry_kernel(
                        &sim.vertices[1].band.to_vec(),
                        &sim.vertices[1].k_ab,
                        eta,
                        nsta,
                    );
                    let (_g2, o2) = eval_berry_kernel(
                        &sim.vertices[2].band.to_vec(),
                        &sim.vertices[2].k_ab,
                        eta,
                        nsta,
                    );
                    let (_g3, o3) = eval_berry_kernel(
                        &sim.vertices[3].band.to_vec(),
                        &sim.vertices[3].k_ab,
                        eta,
                        nsta,
                    );

                    let bands: [Vec<f64>; 4] = [
                        sim.vertices[0].band.to_vec(),
                        sim.vertices[1].band.to_vec(),
                        sim.vertices[2].band.to_vec(),
                        sim.vertices[3].band.to_vec(),
                    ];
                    let kmats: [Array2<Complex<f64>>; 4] = [
                        sim.vertices[0].k_ab.clone(),
                        sim.vertices[1].k_ab.clone(),
                        sim.vertices[2].k_ab.clone(),
                        sim.vertices[3].k_ab.clone(),
                    ];

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
