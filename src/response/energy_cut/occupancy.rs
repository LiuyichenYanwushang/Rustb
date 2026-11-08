//! Occupancy-cut (AHC) integration and thermal convolution.

use ndarray::prelude::*;
use num_complex::Complex;
use rayon::prelude::*;

use super::{
    ENERGY_CUT_EPS, FERMI_X_CUT, FERMI_X_STEPS, fermi_window_x, fixed_coords_to_array2,
    tet_vol_from_pts, triangle_area,
};
use crate::response::kernel::{eval_berry_band_at_lam_buf, eval_berry_kernel};
use crate::response::quadrature::{TET_QUAD_PTS_4, TET_QUAD_WTS_4, TRI_QUAD_PTS_3, TRI_QUAD_WTS_3};
use crate::response::tracking::{build_tetrahedra_3d_diagavg, build_triangles_2d_diagavg};
use crate::response::types::VertexKernel;

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
///
/// The fixed buffer avoids one heap allocation per partial triangle, band and
/// chemical potential. The returned count is in `0..=2`; only the first
/// `count` entries of the buffer are initialized and may be read.
fn clip_triangle(energy_v: [f64; 3], mu: f64) -> ([[[f64; 3]; 3]; 2], usize) {
    let eps = 1e-12;
    let e_min = energy_v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let e_max = energy_v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut out = [[[0.0; 3]; 3]; 2];

    if mu <= e_min + eps {
        return (out, 0);
    }
    if mu >= e_max - eps {
        out[0] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        return (out, 1);
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
        out[0] = [unit(a), cut(a, b), cut(a, c)];
        (out, 1)
    } else {
        // 2 vertices below (a,b) → quadrilateral → two sub‑triangles
        out[0] = [unit(a), unit(b), cut(a, c)];
        out[1] = [unit(b), cut(a, c), cut(b, c)];
        (out, 2)
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
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
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
    let (sub_tris, n_sub_tris) = clip_triangle(e_v, mu);
    let mut total = 0.0;
    for sub_tri in &sub_tris[..n_sub_tris] {
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
                * eval_berry_band_at_lam_buf(n, bands, kmats, &lam, eta, nsta, e_buf, k_buf);
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
    let mu_slice = mu.as_slice().unwrap();

    let result = (0..nx * ny)
        .into_par_iter()
        .fold(
            || Array1::<f64>::zeros(n_mu),
            |mut local_acc, idx| {
                let iy = idx % ny;
                let ix = idx / ny;
                let sims = build_triangles_2d_diagavg(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
                for sim in &sims {
                    let coords = fixed_coords_to_array2(&sim.coords);
                    let area = triangle_area(&coords);
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
                    let bands: [&[f64]; 3] = [
                        v0.band.as_slice().unwrap(),
                        v1.band.as_slice().unwrap(),
                        v2.band.as_slice().unwrap(),
                    ];
                    let kmats: [&Array2<Complex<f64>>; 3] = [&v0.k_ab, &v1.k_ab, &v2.k_ab];
                    let mut e_buf = vec![0.0f64; nsta];
                    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];

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

                        // Sorted-μ sweep: empty / partial / full.
                        let i_partial = mu_slice.partition_point(|&x| x <= e_min + ENERGY_CUT_EPS);
                        let i_full = mu_slice.partition_point(|&x| x < e_max - ENERGY_CUT_EPS);

                        // Empty region μ[0..i_partial]: skip.

                        // Partial region μ[i_partial..i_full]: K‑quadrature.
                        for im in i_partial..i_full {
                            local_acc[im] += volume_scale
                                * triangle_occupied_hybrid(
                                    &coords, e_v, omega_v, &bands, &kmats, mu[im], eta, n, nsta,
                                    &mut e_buf, &mut k_buf,
                                );
                        }

                        // Full region μ[i_full..]: range add.
                        if i_full < n_mu {
                            let add = volume_scale * full_val;
                            for im in i_full..n_mu {
                                local_acc[im] += add;
                            }
                        }
                    }
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

    result
}

/// 2D Fermi‑window integration via per‑μ triangle occupancy cut.
///
/// At zero thermal width the occupation is a step function; the integral is
/// computed exactly within each triangle (no energy binning). At finite width
/// the zero-width result is thermally convolved:
///
/// ```math
/// \sigma_T(\mu) = \int_{-\infty}^\infty w(x)\,\sigma_0(\mu + x/\beta)\,dx,
/// \qquad w(x)=\frac{e^x}{(1+e^x)^2}
/// ```
///
/// Returns $\sigma(\mu_i)$ for each $\mu$ in `mu` (fractional BZ volume;
/// divide by $\det(L)$ for Cartesian).
/// `all_pts` must have been band‑tracked via [`global_band_track`].
/// The public `Model`‑level methods handle this automatically.
pub(crate) fn integrate_fermi_cut_2d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    thermal_width: f64,
    eta: f64,
) -> Array1<f64> {
    assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 2);

    if thermal_width == 0.0 {
        return integrate_fermi_cut_2d_t0(all_pts, k_mesh, mu, eta);
    }

    thermal_convolution(mu, thermal_width, |shifted_mu| {
        integrate_fermi_cut_2d_t0(all_pts, k_mesh, shifted_mu, eta)
    })
}

/// Diagnostic counters for hybrid energy‑cut.
#[derive(Default, Clone)]
#[cfg(test)]
pub(crate) struct FermiCutCounts {
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
#[cfg(test)]
pub(crate) fn read_reset_fermi_cut_counts() -> FermiCutCounts {
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
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
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
        total += sub_vol
            * w
            * eval_berry_band_at_lam_buf(n, bands, kmats, &lam, eta, nsta, e_buf, k_buf);
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
    e_buf: &mut [f64],
    k_buf: &mut [Complex<f64>],
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
            e_buf,
            k_buf,
        )
    } else if mu <= e_v[c] + eps {
        // Average the two diagonals of the cut prism. Either triangulation
        // is valid, but using only one makes quadrature depend on how nearly
        // equal vertex energies are ordered (including k / -k partners).
        let mut split = |c, d| {
            sub_tet_k_quad(
                &[unit(a), unit(b), cut_bary(a, c), cut_bary(b, d)],
                bands,
                kmats,
                n,
                eta,
                nsta,
                coords,
                e_buf,
                k_buf,
            ) + sub_tet_k_quad(
                &[unit(a), cut_bary(a, c), cut_bary(a, d), cut_bary(b, d)],
                bands,
                kmats,
                n,
                eta,
                nsta,
                coords,
                e_buf,
                k_buf,
            ) + sub_tet_k_quad(
                &[unit(b), cut_bary(b, c), cut_bary(b, d), cut_bary(a, c)],
                bands,
                kmats,
                n,
                eta,
                nsta,
                coords,
                e_buf,
                k_buf,
            )
        };
        0.5 * (split(c, d) + split(d, c))
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
                e_buf,
                k_buf,
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
    let _mu_slice = mu.as_slice().unwrap();

    let result = (0..nx * ny * nz)
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
                    let coords = fixed_coords_to_array2(&sim.coords);
                    let vol = tet_vol_from_pts(
                        [sim.coords[0][0], sim.coords[0][1], sim.coords[0][2]],
                        [sim.coords[1][0], sim.coords[1][1], sim.coords[1][2]],
                        [sim.coords[2][0], sim.coords[2][1], sim.coords[2][2]],
                        [sim.coords[3][0], sim.coords[3][1], sim.coords[3][2]],
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
                    let bands: [&[f64]; 4] = [
                        v0.band.as_slice().unwrap(),
                        v1.band.as_slice().unwrap(),
                        v2.band.as_slice().unwrap(),
                        v3.band.as_slice().unwrap(),
                    ];
                    let kmats: [&Array2<Complex<f64>>; 4] =
                        [&v0.k_ab, &v1.k_ab, &v2.k_ab, &v3.k_ab];
                    let mut e_buf = vec![0.0f64; nsta];
                    let mut k_buf = vec![Complex::new(0.0, 0.0); nsta];

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
                            local_acc[im] += volume_scale
                                * tetrahedron_occupied_hybrid(
                                    &coords, e_v, omega_v, &bands, &kmats, mu[im], eta, n, nsta,
                                    &mut e_buf, &mut k_buf,
                                );
                        }

                        // Full region μ[i_full..]: range add.
                        if i_full < n_mu {
                            let add = volume_scale * full_val;
                            for im in i_full..n_mu {
                                local_acc[im] += add;
                            }
                        }
                    }
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

    result
}

/// 3D Fermi‑window integration via per‑μ tetrahedron occupancy cut.
///
/// Same algorithm as [`integrate_fermi_cut_2d`] but for tetrahedra.
/// At zero thermal width each tetrahedron's linearly-interpolated $E(k)$ and
/// $\Omega(k)$ are integrated exactly over the occupied sub-region; finite
/// width uses thermal convolution of the zero-width result.
/// `all_pts` must have been band‑tracked via [`global_band_track`].
/// The public `Model`‑level methods handle this automatically.
pub(crate) fn integrate_fermi_cut_3d(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    thermal_width: f64,
    eta: f64,
) -> Array1<f64> {
    assert!(
        mu.as_slice().unwrap().windows(2).all(|w| w[0] <= w[1]),
        "mu must be sorted ascending"
    );
    assert_eq!(k_mesh.len(), 3);

    if thermal_width == 0.0 {
        return integrate_fermi_cut_3d_t0(all_pts, k_mesh, mu, eta);
    }

    thermal_convolution(mu, thermal_width, |shifted_mu| {
        integrate_fermi_cut_3d_t0(all_pts, k_mesh, shifted_mu, eta)
    })
}

// Evaluate only the chemical potentials used by the thermal quadrature. The
// work is bounded by mu.len() * FERMI_X_STEPS, independently of 1/T or the
// distance between requested chemical potentials. Sorting preserves the
// zero-temperature integrators' ascending-mu contract; no table extrapolation.
pub(super) fn thermal_convolution(
    mu: &Array1<f64>,
    thermal_width: f64,
    zero_temperature: impl FnOnce(&Array1<f64>) -> Array1<f64>,
) -> Array1<f64> {
    let dx = 2.0 * FERMI_X_CUT / FERMI_X_STEPS as f64;
    let mut samples = Vec::new();
    for (index, &m) in mu.iter().enumerate() {
        for iq in 0..FERMI_X_STEPS {
            let x = -FERMI_X_CUT + (iq as f64 + 0.5) * dx;
            samples.push((m + thermal_width * x, index, dx * fermi_window_x(x)));
        }
    }
    samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let shifted_mu = Array1::from_iter(samples.iter().map(|sample| sample.0));
    let values = zero_temperature(&shifted_mu);
    let mut result = Array1::zeros(mu.len());
    for ((_, index, weight), value) in samples.into_iter().zip(values) {
        result[index] += weight * value;
    }
    result
}
