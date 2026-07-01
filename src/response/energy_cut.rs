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
use rayon::prelude::*;

use super::kernel::eval_berry_kernel;
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

/// Exact integral of linearly interpolated $\Omega(k)$ over the occupied
/// region $\{k \in \text{triangle} : E(k) \le \mu\}$.
///
/// Three cases:
/// - $\mu$ below all vertices → 0
/// - $\mu$ above all vertices → $\text{area} \times (\Omega_0+\Omega_1+\Omega_2)/3$
/// - otherwise → triangle clipped by the iso‑energy line $E=\mu$,
///   integrated analytically (sub‑triangle or two triangles for a quadrilateral)
///
/// Returns the integral in the coordinate measure of `coords`.
fn triangle_occupied_integral(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    omega_v: [f64; 3],
    mu: f64,
) -> f64 {
    let eps = ENERGY_CUT_EPS;
    let area = triangle_area(coords);
    if area < eps {
        return 0.0;
    }

    let e_min = energy_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = energy_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if mu <= e_min + eps {
        return 0.0;
    }
    if mu >= e_max - eps {
        return area * (omega_v[0] + omega_v[1] + omega_v[2]) / 3.0;
    }

    // Sort vertices by energy: e[a] ≤ e[b] ≤ e[c]
    let mut idx: [usize; 3] = [0, 1, 2];
    idx.sort_by(|&i, &j| energy_v[i].partial_cmp(&energy_v[j]).unwrap());
    let (a, b, c) = (idx[0], idx[1], idx[2]);

    let tri = |p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]| -> f64 {
        0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])).abs()
    };

    // Intersection point + Ω on edge i→j at energy μ
    let cut = |i: usize, j: usize| -> ([f64; 2], f64) {
        let de = energy_v[j] - energy_v[i];
        let t = if de.abs() < eps {
            0.5
        } else {
            ((mu - energy_v[i]) / de).clamp(0.0, 1.0)
        };
        let x = coords[[i, 0]] + t * (coords[[j, 0]] - coords[[i, 0]]);
        let y = coords[[i, 1]] + t * (coords[[j, 1]] - coords[[i, 1]]);
        let omega = omega_v[i] + t * (omega_v[j] - omega_v[i]);
        ([x, y], omega)
    };

    let a_xy = [coords[[a, 0]], coords[[a, 1]]];
    let b_xy = [coords[[b, 0]], coords[[b, 1]]];

    if mu <= energy_v[b] + eps {
        // a below μ, b and c above → sub‑triangle a → p_ab → p_ac
        let (p_ab, o_ab) = cut(a, b);
        let (p_ac, o_ac) = cut(a, c);
        tri(a_xy, p_ab, p_ac) * (omega_v[a] + o_ab + o_ac) / 3.0
    } else {
        // a and b below μ, c above → quadrilateral → two triangles
        let (p_ac, o_ac) = cut(a, c);
        let (p_bc, o_bc) = cut(b, c);
        tri(a_xy, b_xy, p_ac) * (omega_v[a] + omega_v[b] + o_ac) / 3.0
            + tri(b_xy, p_ac, p_bc) * (omega_v[b] + o_ac + o_bc) / 3.0
    }
}

/// T=0 occupancy‑cut integration (no energy binning).
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

                for n in 0..nsta {
                    let e_v = [
                        sim.vertices[0].band[n],
                        sim.vertices[1].band[n],
                        sim.vertices[2].band[n],
                    ];
                    let omega_v = [o0[n], o1[n], o2[n]];

                    for im in 0..n_mu {
                        result[im] += volume_scale
                            * triangle_occupied_integral(&sim.coords, e_v, omega_v, mu[im]);
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

/// Exact integral of linearly interpolated $\Omega(k)$ over the occupied
/// region $\{k \in \text{tetrahedron} : E(k) \le \mu\}$.
///
/// Five cases depending on how many vertices lie below $\mu$ (after sorting
/// by energy).  The occupied polyhedron is decomposed into at most 3
/// sub‑tetrahedra whose volume‑weighted average $\Omega$ is summed.
///
/// Returns the integral in the coordinate measure of `coords`.
fn tetrahedron_occupied_integral(
    coords: &Array2<f64>, // 4×3
    energy_v: [f64; 4],
    omega_v: [f64; 4],
    mu: f64,
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

    let e_min = energy_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = energy_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if mu <= e_min + eps {
        return 0.0;
    }
    if mu >= e_max - eps {
        return full_vol * (omega_v[0] + omega_v[1] + omega_v[2] + omega_v[3]) / 4.0;
    }

    // Sort vertices by energy: e[a] ≤ e[b] ≤ e[c] ≤ e[d]
    let mut idx: [usize; 4] = [0, 1, 2, 3];
    idx.sort_by(|&i, &j| energy_v[i].partial_cmp(&energy_v[j]).unwrap());
    let (a, b, c, d) = (idx[0], idx[1], idx[2], idx[3]);

    let xyzi = |i: usize| -> [f64; 3] { [coords[[i, 0]], coords[[i, 1]], coords[[i, 2]]] };

    // Intersection point + Ω on edge i→j at energy μ
    let cut = |i: usize, j: usize| -> ([f64; 3], f64) {
        let de = energy_v[j] - energy_v[i];
        let t = if de.abs() < eps {
            0.5
        } else {
            ((mu - energy_v[i]) / de).clamp(0.0, 1.0)
        };
        let x = coords[[i, 0]] + t * (coords[[j, 0]] - coords[[i, 0]]);
        let y = coords[[i, 1]] + t * (coords[[j, 1]] - coords[[i, 1]]);
        let z = coords[[i, 2]] + t * (coords[[j, 2]] - coords[[i, 2]]);
        let omega = omega_v[i] + t * (omega_v[j] - omega_v[i]);
        ([x, y, z], omega)
    };

    let tet_int = |v0: [f64; 3],
                   v1: [f64; 3],
                   v2: [f64; 3],
                   v3: [f64; 3],
                   o0: f64,
                   o1: f64,
                   o2: f64,
                   o3: f64|
     -> f64 { tet_vol_from_pts(v0, v1, v2, v3) * (o0 + o1 + o2 + o3) / 4.0 };

    if mu <= energy_v[b] + eps {
        // Case 3: 1 vertex below (a), 3 above (b,c,d)
        // Occupied = sub‑tet (v_a, p_ab, p_ac, p_ad)
        let (p_ab, o_ab) = cut(a, b);
        let (p_ac, o_ac) = cut(a, c);
        let (p_ad, o_ad) = cut(a, d);
        tet_int(xyzi(a), p_ab, p_ac, p_ad, omega_v[a], o_ab, o_ac, o_ad)
    } else if mu <= energy_v[c] + eps {
        // Case 4: 2 below (a,b), 2 above (c,d)
        // Decompose occupied polyhedron into 3 tets via diagonal p_ac → p_bd
        let (p_ac, o_ac) = cut(a, c);
        let (p_ad, o_ad) = cut(a, d);
        let (p_bc, o_bc) = cut(b, c);
        let (p_bd, o_bd) = cut(b, d);

        let a_xyz = xyzi(a);
        let b_xyz = xyzi(b);

        tet_int(a_xyz, b_xyz, p_ac, p_bd, omega_v[a], omega_v[b], o_ac, o_bd)
            + tet_int(a_xyz, p_ac, p_ad, p_bd, omega_v[a], o_ac, o_ad, o_bd)
            + tet_int(b_xyz, p_bc, p_bd, p_ac, omega_v[b], o_bc, o_bd, o_ac)
    } else {
        // Case 5: 3 below (a,b,c), 1 above (d)
        // Occupied = full tet − sub‑tet (v_d, p_ad, p_bd, p_cd)
        let (p_ad, o_ad) = cut(a, d);
        let (p_bd, o_bd) = cut(b, d);
        let (p_cd, o_cd) = cut(c, d);

        let full = full_vol * (omega_v[0] + omega_v[1] + omega_v[2] + omega_v[3]) / 4.0;
        let sub = tet_int(xyzi(d), p_ad, p_bd, p_cd, omega_v[d], o_ad, o_bd, o_cd);
        full - sub
    }
}

/// T=0 3D occupancy‑cut integration.
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

                    for n in 0..nsta {
                        let e_v = [
                            sim.vertices[0].band[n],
                            sim.vertices[1].band[n],
                            sim.vertices[2].band[n],
                            sim.vertices[3].band[n],
                        ];
                        let omega_v = [o0[n], o1[n], o2[n], o3[n]];

                        for im in 0..n_mu {
                            result[im] += volume_scale
                                * tetrahedron_occupied_integral(&sim.coords, e_v, omega_v, mu[im]);
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
