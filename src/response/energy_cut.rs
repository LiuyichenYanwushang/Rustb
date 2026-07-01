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

use super::kernel::eval_berry_kernel;
use super::tracking::build_triangles_2d_diagavg;
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
        let vdiag = v.vdiag.as_ref()?;
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
