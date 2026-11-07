//! Fermi-surface energy-cut integration (dipole and intrinsic).

use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

use super::{
    ENERGY_CUT_EPS, FERMI_X_CUT, FERMI_X_STEPS, fermi_window_x, fixed_coords_to_array2,
    tet_vol_from_pts, triangle_area,
};
use crate::response::kernel::{eval_berry_complex_at_lam_buf, eval_intrinsic_G3_at_lam_buf};
use crate::response::quadrature::{TRI_QUAD_PTS_3, TRI_QUAD_WTS_3};
use crate::response::tracking::{build_tetrahedra_3d_diagavg, build_triangles_2d_diagavg};
use crate::response::types::{SIMPLEX_GAP_TOL, TrackedSimplex, VertexKernel};

// ── K‑quadrature line‑cut helpers ──────────────────────────────────────

/// Find the two intersection points of the iso‑energy line $E=\mu$ with
/// the triangle edges, returning both physical coordinates and barycentrics.
fn find_line_intersections(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    energy: f64,
) -> ([[f64; 2]; 2], [[f64; 3]; 2], usize) {
    let mut pts = [[0.0f64; 2]; 2];
    let mut bcs = [[0.0f64; 3]; 2];
    let mut count = 0usize;
    if !energy.is_finite() || energy_v.iter().any(|e| !e.is_finite()) {
        return ([[f64::NAN; 2]; 2], [[f64::NAN; 3]; 2], 2);
    }
    let xy = [
        [coords[[0, 0]], coords[[0, 1]]],
        [coords[[1, 0]], coords[[1, 1]]],
        [coords[[2, 0]], coords[[2, 1]]],
    ];
    for &(i, j) in &[(0usize, 1usize), (1, 2), (2, 0)] {
        let ei = energy_v[i];
        let ej = energy_v[j];
        let denom = ej - ei;
        let offset = energy - ei;
        if !denom.is_finite() || !offset.is_finite() {
            // An edge outside the cut needs no representable subtraction.
            if energy < ei.min(ej) || energy > ei.max(ej) {
                continue;
            }
            return ([[f64::NAN; 2]; 2], [[f64::NAN; 3]; 2], 2);
        }
        if denom.abs() < ENERGY_CUT_EPS {
            continue;
        }
        let t = offset / denom;
        if !t.is_finite() {
            if energy < ei.min(ej) || energy > ei.max(ej) {
                continue;
            }
            return ([[f64::NAN; 2]; 2], [[f64::NAN; 3]; 2], 2);
        }
        if (-ENERGY_CUT_EPS..=1.0 + ENERGY_CUT_EPS).contains(&t) {
            let tc = t.clamp(0.0, 1.0);
            let px = xy[i][0] + tc * (xy[j][0] - xy[i][0]);
            let py = xy[i][1] + tc * (xy[j][1] - xy[i][1]);
            if !px.is_finite() || !py.is_finite() {
                return ([[f64::NAN; 2]; 2], [[f64::NAN; 3]; 2], 2);
            }
            // Deduplicate
            let mut dup = false;
            for k in 0..count {
                let dx = pts[k][0] - px;
                let dy = pts[k][1] - py;
                let distance2 = dx * dx + dy * dy;
                if !distance2.is_finite() {
                    return ([[f64::NAN; 2]; 2], [[f64::NAN; 3]; 2], 2);
                }
                if distance2 < 1e-24 {
                    dup = true;
                    break;
                }
            }
            if !dup {
                pts[count] = [px, py];
                let mut lam = [0.0; 3];
                lam[i] = 1.0 - tc;
                lam[j] = tc;
                bcs[count] = lam;
                count += 1;
            }
        }
    }
    (pts, bcs, count)
}

/// K‑quadrature line‑cut for the Berry curvature dipole amplitude
/// $A_n = v^c_n \cdot \Omega_n$ along the $E_n=\mu$ line.
///
/// Uses 2‑point Gauss‑Legendre quadrature along the line segment, with
/// barycentric interpolation of $K^{ab}_{nm}$, $E_m$, and $v^c_n$.
pub(super) fn kquad_line_cut_dipole(
    coords: &Array2<f64>,
    energy_v: [f64; 3],
    bands: &[&[f64]],
    kmats: &[&Array2<Complex<f64>>],
    vdiag_v: [f64; 3],
    energy: f64,
    eta: f64,
    n: usize,
    nsta: usize,
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
) -> f64 {
    let (pts, bcs, n_pts) = find_line_intersections(coords, energy_v, energy);
    if pts.iter().flatten().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    if n_pts < 2 {
        return 0.0;
    }

    // |∇E|
    let (x0, y0) = (coords[[0, 0]], coords[[0, 1]]);
    let (x1, y1) = (coords[[1, 0]], coords[[1, 1]]);
    let (x2, y2) = (coords[[2, 0]], coords[[2, 1]]);
    let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if !det.is_finite() {
        return f64::NAN;
    }
    if det.abs() < ENERGY_CUT_EPS {
        return 0.0;
    }
    let de1 = energy_v[1] - energy_v[0];
    let de2 = energy_v[2] - energy_v[0];
    let grad_x = (de1 * (y2 - y0) - de2 * (y1 - y0)) / det;
    let grad_y = ((x1 - x0) * de2 - (x2 - x0) * de1) / det;
    let grad_norm = grad_x.hypot(grad_y);
    if [de1, de2, grad_x, grad_y, grad_norm]
        .iter()
        .any(|x| !x.is_finite())
    {
        return f64::NAN;
    }
    if grad_norm < ENERGY_CUT_EPS {
        return 0.0;
    }

    // Find the two most distant points
    let mut best = (0usize, 1usize);
    let mut best_l2 = -1.0;
    for i in 0..n_pts {
        for j in i + 1..n_pts {
            let dx = pts[i][0] - pts[j][0];
            let dy = pts[i][1] - pts[j][1];
            let l2 = dx * dx + dy * dy;
            if !l2.is_finite() {
                return f64::NAN;
            }
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

    let mut amp_sum = 0.0;
    for t in &t_vals {
        let lam = [
            (1.0 - t) * lam0[0] + t * lam1[0],
            (1.0 - t) * lam0[1] + t * lam1[1],
            (1.0 - t) * lam0[2] + t * lam1[2],
        ];
        let (_metric, berry) =
            eval_berry_complex_at_lam_buf(n, bands, kmats, &lam, eta, nsta, e_buf, k_buf);
        let vc = lam[0] * vdiag_v[0] + lam[1] * vdiag_v[1] + lam[2] * vdiag_v[2];
        if !berry.is_finite()
            || !vc.is_finite()
            || e_buf[..nsta].iter().any(|x| !x.is_finite())
            || k_buf[..nsta]
                .iter()
                .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return f64::NAN;
        }
        amp_sum += vc * berry;
    }
    // Gauss weights are 0.5 each
    let integral = 0.5 * length * amp_sum / grad_norm;
    if integral.is_finite() {
        integral
    } else {
        f64::NAN
    }
}

/// K‑quadrature dipole accumulator (zero-clone, borrows simplex vertices).
pub(super) fn accumulate_triangle_dipole_kquad(
    sim: &TrackedSimplex<'_, 3>,
    eta: f64,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let coords = fixed_coords_to_array2(&sim.coords);

    let area = triangle_area(&coords);
    if area < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / area;
    let nsta = sim.vertices[0].band.len();

    // Borrow vertex data (no clone).
    let nonlinear = sim
        .vertices
        .map(|v| v.nonlinear.as_ref().expect("nonlinear primitives required"));
    let bands = sim.vertices.map(|v| v.band.as_slice().unwrap());
    let kmats = sim.vertices.map(|v| &v.k_ab);
    let vdiags = nonlinear.map(|n| n.vdiag.as_slice().unwrap());
    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];
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
                        &coords, e_v, &bands, &kmats, vdiag_v, mu[im], eta, n, nsta, &mut e_buf,
                        &mut k_buf,
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
                        &coords, e_v, &bands, &kmats, vdiag_v, energy, eta, n, nsta, &mut e_buf,
                        &mut k_buf,
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
/// `all_pts` must have been band‑tracked via [`global_band_track`].
/// The public `Model`‑level methods handle this automatically.
pub(crate) fn integrate_dipole_energy_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    thermal_width: f64,
    eta: f64,
) -> (Array1<f64>, usize) {
    assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(
        k_mesh.len(),
        2,
        "energy-cut dipole currently supports 2D only"
    );
    let beta = if thermal_width > 0.0 {
        1.0 / thermal_width
    } else {
        0.0
    };
    let (nx, ny) = (k_mesh[0], k_mesh[1]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let n_mu = mu.len();
    let (acc, unsafe_count) = (0..nx * ny)
        .into_par_iter()
        .fold(
            || (Array1::<f64>::zeros(n_mu), 0usize),
            |(mut local_acc, mut local_us), idx| {
                let iy = idx % ny;
                let ix = idx / ny;
                let sims = build_triangles_2d_diagavg(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
                for sim in &sims {
                    if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                        local_us += 1;
                    }
                    accumulate_triangle_dipole_kquad(sim, eta, mu, beta, &mut local_acc);
                }
                (local_acc, local_us)
            },
        )
        .reduce(
            || (Array1::zeros(n_mu), 0),
            |(mut a1, u1), (a2, u2)| {
                a1 += &a2;
                (a1, u1 + u2)
            },
        );

    (acc, unsafe_count)
}

// ── Intrinsic NLH energy‑cut ────────────────────────────────────────────

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
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
    grad_norm: f64,
) -> f64 {
    let (pts, bcs, n_pts) = find_line_intersections(coords, energy_v, energy);
    if pts.iter().flatten().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    if n_pts < 2 {
        return 0.0;
    }
    if !grad_norm.is_finite() {
        return f64::NAN;
    }

    let mut best = (0usize, 1usize);
    let mut best_l2 = -1.0;
    for i in 0..n_pts {
        for j in i + 1..n_pts {
            let dx = pts[i][0] - pts[j][0];
            let dy = pts[i][1] - pts[j][1];
            let l2 = dx * dx + dy * dy;
            if !l2.is_finite() {
                return f64::NAN;
            }
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

    let mut amp_sum = 0.0;
    for t in &t_vals {
        let lam = [
            (1.0 - t) * lam0[0] + t * lam1[0],
            (1.0 - t) * lam0[1] + t * lam1[1],
            (1.0 - t) * lam0[2] + t * lam1[2],
        ];
        let (g_ab, g_bc, g_ac) = eval_intrinsic_G3_at_lam_buf(
            n, bands, kmat_ab, kmat_bc, kmat_ac, &lam, nsta, e_buf, k_buf,
        );
        let va = lam[0] * vdiag_a[0] + lam[1] * vdiag_a[1] + lam[2] * vdiag_a[2];
        let vb = lam[0] * vdiag_b[0] + lam[1] * vdiag_b[1] + lam[2] * vdiag_b[2];
        let vc = lam[0] * vdiag_c[0] + lam[1] * vdiag_c[1] + lam[2] * vdiag_c[2];
        let q = 2.0 * vc * g_ab - 0.5 * (va * g_bc + vb * g_ac);
        if [g_ab, g_bc, g_ac, va, vb, vc, q]
            .iter()
            .any(|x| !x.is_finite())
            || e_buf[..nsta]
                .iter()
                .any(|&e| !e.is_finite() || !(e_buf[n] - e).is_finite())
            || k_buf[..nsta * 3]
                .iter()
                .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return f64::NAN;
        }
        amp_sum -= q; // sign convention: direct sum returns −Q, so EC returns −∫ δ Q dk
    }
    let integral = 0.5 * length * amp_sum / grad_norm;
    if integral.is_finite() {
        integral
    } else {
        f64::NAN
    }
}

/// K‑quadrature intrinsic accumulator (zero-clone, borrows simplex vertices).
pub(super) fn accumulate_triangle_intrinsic_kquad(
    sim: &TrackedSimplex<'_, 3>,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let coords = fixed_coords_to_array2(&sim.coords);

    let area = triangle_area(&coords);
    if area < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / area;
    let nsta = sim.vertices[0].band.len();

    let nonlinear = sim
        .vertices
        .map(|v| v.nonlinear.as_ref().expect("nonlinear primitives required"));
    let bands = sim.vertices.map(|v| v.band.as_slice().unwrap());
    let kmat_ab = sim.vertices.map(|v| &v.k_ab);
    let kmat_bc = nonlinear.map(|n| &n.k_bc);
    let kmat_ac = nonlinear.map(|n| &n.k_ac);
    let vdiag_c = nonlinear.map(|n| n.vdiag.as_slice().unwrap());
    let vdiag_a = nonlinear.map(|n| n.vdiag_a.as_slice().unwrap());
    let vdiag_b = nonlinear.map(|n| n.vdiag_b.as_slice().unwrap());

    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta * 3];
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

        // Precompute |∇E| once per band.
        let c = &coords;
        let (x0, y0) = (c[[0, 0]], c[[0, 1]]);
        let (x1, y1) = (c[[1, 0]], c[[1, 1]]);
        let (x2, y2) = (c[[2, 0]], c[[2, 1]]);
        let det_grad = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        if det_grad.abs() < ENERGY_CUT_EPS {
            continue;
        }
        let de1 = e_v[1] - e_v[0];
        let de2 = e_v[2] - e_v[0];
        let gx = (de1 * (y2 - y0) - de2 * (y1 - y0)) / det_grad;
        let gy = ((x1 - x0) * de2 - (x2 - x0) * de1) / det_grad;
        let norm = gx.hypot(gy);
        let grad_norm = if [det_grad, de1, de2, gx, gy, norm]
            .iter()
            .all(|x| x.is_finite())
        {
            norm
        } else {
            f64::NAN
        };

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
                        &coords, e_v, &bands, &kmat_ab, &kmat_bc, &kmat_ac, vc_v, va_v, vb_v,
                        mu[im], n, nsta, &mut e_buf, &mut k_buf, grad_norm,
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
                        &coords, e_v, &bands, &kmat_ab, &kmat_bc, &kmat_ac, vc_v, va_v, vb_v,
                        energy, n, nsta, &mut e_buf, &mut k_buf, grad_norm,
                    );
                    sum += dx * fermi_window_x(x) * rho;
                }
                acc[im] += volume_scale * sum;
            }
        }
    }
}

/// 2D intrinsic NLH via K‑quadrature energy‑cut.
/// `all_pts` must have been band‑tracked via [`global_band_track`].
/// The public `Model`‑level methods handle this automatically.
pub(crate) fn integrate_intrinsic_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    thermal_width: f64,
) -> Array1<f64> {
    assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 2);
    let (nx, ny) = (k_mesh[0], k_mesh[1]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let beta = if thermal_width > 0.0 {
        1.0 / thermal_width
    } else {
        0.0
    };
    let n_mu = mu.len();
    let acc = (0..nx * ny)
        .into_par_iter()
        .fold(
            || Array1::<f64>::zeros(n_mu),
            |mut local_acc, idx| {
                let iy = idx % ny;
                let ix = idx / ny;
                let sims = build_triangles_2d_diagavg(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
                for sim in &sims {
                    accumulate_triangle_intrinsic_kquad(sim, mu, beta, &mut local_acc);
                }
                local_acc
            },
        )
        .reduce(
            || Array1::zeros(n_mu),
            |mut a, b| {
                a += &b;
                a
            },
        );

    acc
}

// ── 3D intrinsic NLH energy‑cut ─────────────────────────────────────────

/// Gradient $\nabla E$ (3D) computed from tetrahedron vertex coordinates
/// and energies.  Returns `(gx, gy, gz, |∇E|)`.
pub(super) fn energy_gradient_3d(coords: &Array2<f64>, de: [f64; 3]) -> (f64, f64, f64, f64) {
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
    if !det.is_finite() || de.iter().any(|x| !x.is_finite()) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
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
    let norm = gx.hypot(gy).hypot(gz);
    if [gx, gy, gz, norm].iter().any(|x| !x.is_finite()) {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    (gx, gy, gz, norm)
}

/// Find the intersection polygon of the $E=\mu$ plane with a tetrahedron.
/// Returns vertices as barycentric coordinates in the original tet.
pub(super) fn tet_plane_intersection(energy_v: [f64; 4], mu: f64) -> ([[f64; 4]; 4], usize) {
    let eps = 1e-12;
    if !mu.is_finite() || energy_v.iter().any(|e| !e.is_finite()) {
        return ([[f64::NAN; 4]; 4], 3);
    }

    // Sort by energy
    let mut idx: [usize; 4] = [0, 1, 2, 3];
    idx.sort_by(|&i, &j| energy_v[i].partial_cmp(&energy_v[j]).unwrap());
    let (a, b, c, d) = (idx[0], idx[1], idx[2], idx[3]);

    if mu <= energy_v[a] + eps || mu >= energy_v[d] - eps {
        return ([[0.0f64; 4]; 4], 0);
    }

    let cut = |i: usize, j: usize| -> [f64; 4] {
        let de = energy_v[j] - energy_v[i];
        let offset = mu - energy_v[i];
        if !de.is_finite() || !offset.is_finite() {
            return [f64::NAN; 4];
        }
        let t = if de.abs() < eps { 0.5 } else { offset / de };
        if !t.is_finite() {
            return [f64::NAN; 4];
        }
        let t = t.clamp(0.0, 1.0);
        let mut l = [0.0; 4];
        l[i] = 1.0 - t;
        l[j] = t;
        l
    };

    if mu <= energy_v[b] + eps {
        // 1 below (a), 3 above → triangle
        let verts = [cut(a, b), cut(a, c), cut(a, d), [0.0; 4]];
        (verts, 3)
    } else if mu <= energy_v[c] + eps {
        // 2 below (a,b), 2 above → quadrilateral
        let verts = [cut(a, c), cut(a, d), cut(b, d), cut(b, c)];
        (verts, 4)
    } else {
        // 3 below (a,b,c), 1 above → triangle
        let verts = [cut(a, d), cut(b, d), cut(c, d), [0.0; 4]];
        (verts, 3)
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
        let norm = cx.hypot(cy).hypot(cz);
        if [cx, cy, cz, norm].iter().any(|x| !x.is_finite()) {
            return f64::NAN;
        }
        area += 0.5 * norm;
    }
    if area.is_finite() { area } else { f64::NAN }
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
pub(super) fn kquad_surface_cut_intrinsic(
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
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
) -> f64 {
    let (verts, n_verts) = tet_plane_intersection(energy_v, mu);
    if verts.iter().flatten().any(|x| !x.is_finite()) {
        return f64::NAN;
    }
    if n_verts < 3 {
        return 0.0;
    }
    if !grad_norm.is_finite() {
        return f64::NAN;
    }

    // K‑quadrature over polygon (fan triangulation from vertex 0)
    let mut amp_sum = 0.0;
    for i in 1..n_verts - 1 {
        let sub_tri = [&verts[0], &verts[i], &verts[i + 1]];
        let sub_area = {
            let lam_ref: [[f64; 4]; 3] = [*sub_tri[0], *sub_tri[1], *sub_tri[2]];
            polygon_area_3d(coords, &lam_ref)
        };
        if !sub_area.is_finite() {
            return f64::NAN;
        }
        if sub_area < 1e-30 {
            continue;
        }
        for iq in 0..3 {
            let alpha = &TRI_QUAD_PTS_3[iq];
            let w = TRI_QUAD_WTS_3[iq];
            let lam = combine_bary_3d(alpha, &[*sub_tri[0], *sub_tri[1], *sub_tri[2]]);
            let (g_ab, g_bc, g_ac) = eval_intrinsic_G3_at_lam_buf(
                n, bands, kmat_ab, kmat_bc, kmat_ac, &lam, nsta, e_buf, k_buf,
            );
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
            if [g_ab, g_bc, g_ac, va, vb, vc, q]
                .iter()
                .any(|x| !x.is_finite())
                || e_buf[..nsta]
                    .iter()
                    .any(|&e| !e.is_finite() || !(e_buf[n] - e).is_finite())
                || k_buf[..nsta * 3]
                    .iter()
                    .any(|x| !x.re.is_finite() || !x.im.is_finite())
            {
                return f64::NAN;
            }
            amp_sum -= sub_area * w * q;
        }
    }

    let integral = amp_sum / grad_norm;
    if integral.is_finite() {
        integral
    } else {
        f64::NAN
    }
}

/// K‑quadrature intrinsic accumulator for tetrahedra (zero-clone).
pub(super) fn accumulate_tetrahedron_intrinsic_kquad(
    sim: &TrackedSimplex<'_, 4>,
    mu: &Array1<f64>,
    beta: f64,
    acc: &mut Array1<f64>,
) {
    let coords = fixed_coords_to_array2(&sim.coords);

    let _vol = tet_vol_from_pts(
        [sim.coords[0][0], sim.coords[0][1], sim.coords[0][2]],
        [sim.coords[1][0], sim.coords[1][1], sim.coords[1][2]],
        [sim.coords[2][0], sim.coords[2][1], sim.coords[2][2]],
        [sim.coords[3][0], sim.coords[3][1], sim.coords[3][2]],
    );
    if _vol < ENERGY_CUT_EPS {
        return;
    }
    let volume_scale = sim.volume / _vol;
    let nsta = sim.vertices[0].band.len();

    let nonlinear = sim
        .vertices
        .map(|v| v.nonlinear.as_ref().expect("nonlinear primitives required"));
    let bands = sim.vertices.map(|v| v.band.as_slice().unwrap());
    let kmat_ab = sim.vertices.map(|v| &v.k_ab);
    let kmat_bc = nonlinear.map(|n| &n.k_bc);
    let kmat_ac = nonlinear.map(|n| &n.k_ac);
    let vdiag_c = nonlinear.map(|n| n.vdiag.as_slice().unwrap());
    let vdiag_a = nonlinear.map(|n| n.vdiag_a.as_slice().unwrap());
    let vdiag_b = nonlinear.map(|n| n.vdiag_b.as_slice().unwrap());

    let _n_mu = mu.len();
    let mut e_buf = vec![0.0f64; nsta];
    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta * 3];
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
        let (_, _, _, grad_norm) = energy_gradient_3d(&coords, de);
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
                        &coords, e_v, &bands, &kmat_ab, &kmat_bc, &kmat_ac, vc_v, va_v, vb_v,
                        mu[im], n, nsta, grad_norm, &mut e_buf, &mut k_buf,
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
                        &coords, e_v, &bands, &kmat_ab, &kmat_bc, &kmat_ac, vc_v, va_v, vb_v,
                        energy, n, nsta, grad_norm, &mut e_buf, &mut k_buf,
                    );
                    sum += dx * fermi_window_x(x) * rho;
                }
                acc[im] += volume_scale * sum;
            }
        }
    }
}

/// 3D intrinsic NLH via K‑quadrature surface energy‑cut.
/// `all_pts` must have been band‑tracked via [`global_band_track`].
/// The public `Model`‑level methods handle this automatically.
pub(crate) fn integrate_intrinsic_cut_3d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    thermal_width: f64,
) -> Array1<f64> {
    assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 3);
    let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
    let inv_nx = 1.0 / nx as f64;
    let inv_ny = 1.0 / ny as f64;
    let inv_nz = 1.0 / nz as f64;
    let beta = if thermal_width > 0.0 {
        1.0 / thermal_width
    } else {
        0.0
    };
    let n_mu = mu.len();
    let acc = (0..nx * ny * nz)
        .into_par_iter()
        .fold(
            || Array1::<f64>::zeros(n_mu),
            |mut local_acc, idx| {
                let iz = idx % nz;
                let iy = (idx / nz) % ny;
                let ix = idx / (ny * nz);
                let sims = build_tetrahedra_3d_diagavg(
                    ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts,
                );
                for sim in &sims {
                    accumulate_tetrahedron_intrinsic_kquad(sim, mu, beta, &mut local_acc);
                }
                local_acc
            },
        )
        .reduce(
            || Array1::zeros(n_mu),
            |mut a, b| {
                a += &b;
                a
            },
        );

    acc
}
