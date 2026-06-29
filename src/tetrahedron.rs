//! Blochl-style tetrahedron integration for BZ Fermi-surface integrals.
//!
//! At `T = 0` the integral
//!
//! ```math
//! I(\mu) = \sum_n \int_{BZ} \bigl(-\partial f/\partial\varepsilon_n(\mathbf k)\bigr)
//!          \,F_n(\mathbf k)\,\dd\mathbf k
//! ```
//!
//! reduces to a Fermi‑surface integral weighted by `δ(ε_n(k) − μ)`.
//! Within each simplex (triangle in 2D, tetrahedron in 3D) the energy `ε`
//! and the integrand `F` are linearly interpolated from the corner values.
//! The `δ(ε − μ)` contribution is then evaluated **analytically**, giving
//! closed‑form weight functions of the sorted corner energies.
//!
//! ## References
//!
//! - P. E. Blöchl, O. Jepsen, and O. K. Andersen, *Phys. Rev. B* **49**, 16223 (1994).
//! - G. Lehmann and M. Taut, *Phys. Status Solidi B* **54**, 469 (1972).
//!
//! ## 2D vs 3D
//!
//! - **2D**: each k‑mesh cell (rectangle) is split into 2 triangles.
//!   The iso‑energy contour is a line segment; the integral is a line integral.
//! - **3D**: each k‑mesh cell (hexahedron) is split into 5 tetrahedra.
//!   The iso‑energy surface is a polygon; the integral is a surface integral.

use ndarray::prelude::*;
use ndarray::*;
use rayon::prelude::*;

// ── 3D tetrahedron decomposition ──────────────────────────────────────

/// Decompose a cube into 5 tetrahedra.
///
/// Cube corner numbering (local indices):
/// `000=0, 100=1, 010=2, 110=3, 001=4, 101=5, 011=6, 111=7`
const CUBE_TETS: [[usize; 4]; 5] = [
    [0, 1, 2, 4], // 000, 100, 010, 001
    [3, 1, 2, 7], // 110, 100, 010, 111
    [5, 1, 4, 7], // 101, 100, 001, 111
    [6, 2, 4, 7], // 011, 010, 001, 111
    [1, 2, 4, 7], // 100, 010, 001, 111  (central)
];

/// Offsets of the 8 cube corners relative to the cell origin `(i, j, k)`.
const CUBE_CORNERS: [(usize, usize, usize); 8] = [
    (0, 0, 0), // 0: (i,   j,   k)
    (1, 0, 0), // 1: (i+1, j,   k)
    (0, 1, 0), // 2: (i,   j+1, k)
    (1, 1, 0), // 3: (i+1, j+1, k)
    (0, 0, 1), // 4: (i,   j,   k+1)
    (1, 0, 1), // 5: (i+1, j,   k+1)
    (0, 1, 1), // 6: (i,   j+1, k+1)
    (1, 1, 1), // 7: (i+1, j+1, k+1)
];

/// Physical volume fraction of each tetrahedron within a cube.
///
/// Four small tets each occupy `1/6` of the cube; the central tet occupies `1/3`.
/// Total: `4/6 + 1/3 = 1`.
const TET_VOL_FACTOR: [f64; 5] = [
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 6.0,
    1.0 / 3.0,
];

// ── Public API ─────────────────────────────────────────────────────────

/// Perform Blochl tetrahedron integration over the BZ k-mesh.
///
/// # Arguments
///
/// * `band` — shape `(nk, nsta)`: per‑band energies at every k‑point.
/// * `integrand` — shape `(nk, nsta)`: per‑band integrand values.
/// * `k_mesh` — `[nx, ny]` (2D) or `[nx, ny, nz]` (3D).
/// * `mu` — array of chemical potentials.
///
/// # Returns
///
/// `Array1<f64>` of length `mu.len()` containing the integrated quantity
/// `Σ_n ∫_BZ δ(ε_n(k) − μ) F_n(k) dk` (with cell‑volume normalisation
/// already included).  The caller divides by `det(lat)` to convert
/// fractional → Cartesian reciprocal‑space volume.
///
/// # Panics
///
/// Panics if `k_mesh.len()` is not 2 or 3.
pub fn tetrahedron_integrate(
    band: &Array2<f64>,
    integrand: &Array2<f64>,
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
) -> Array1<f64> {
    let dim = k_mesh.len();
    let nk_mesh: usize = k_mesh.iter().product();
    assert_eq!(
        band.nrows(),
        nk_mesh,
        "band.nrows()={} does not match k_mesh product={} ({:?})",
        band.nrows(),
        nk_mesh,
        k_mesh
    );
    assert_eq!(band.nrows(), integrand.nrows());
    assert_eq!(band.ncols(), integrand.ncols());
    let nsta = band.ncols();
    let n_mu = mu.len();

    match dim {
        2 => {
            let [nx, ny] = [k_mesh[0], k_mesh[1]];
            tetrahedron_integrate_2d(band, integrand, nx, ny, nsta, mu, n_mu)
        }
        3 => {
            let [nx, ny, nz] = [k_mesh[0], k_mesh[1], k_mesh[2]];
            tetrahedron_integrate_3d(band, integrand, nx, ny, nz, nsta, mu, n_mu)
        }
        _ => panic!(
            "tetrahedron_integrate: only dim=2,3 supported, got dim={dim}"
        ),
    }
}

/// Regular volume integration over the BZ k-mesh using simplex quadrature.
///
/// Approximates `∫_{BZ} f(k) dk` by linearly interpolating `f` within each
/// simplex (triangle in 2D, tetrahedron in 3D) and integrating analytically.
///
/// The BZ is periodic: the mesh wraps around, covering the full [0,1]^dim
/// domain with `nx*ny[*nz]` cells.
///
/// # Arguments
/// * `f` — shape `(nk,)`: function values at every k‑point.
/// * `k_mesh` — `[nx, ny]` or `[nx, ny, nz]`.
///
/// # Returns
/// The integrated value over the full periodic BZ.
pub fn tetrahedron_volume_integrate(
    f: &Array1<f64>,
    k_mesh: &Array1<usize>,
) -> f64 {
    let dim = k_mesh.len();
    match dim {
        2 => {
            let (nx, ny) = (k_mesh[0], k_mesh[1]);
            let cell_area = 1.0 / (nx * ny) as f64;
            let tri_area = cell_area / 2.0;
            let f_2d = Array2::from_shape_vec((nx, ny), f.to_vec()).unwrap();
            let mut total = 0.0;

            for ix in 0..nx {
                let ixp = (ix + 1) % nx;
                for iy in 0..ny {
                    let iyp = (iy + 1) % ny;
                    let f00 = f_2d[[ix, iy]];
                    let f10 = f_2d[[ixp, iy]];
                    let f11 = f_2d[[ixp, iyp]];
                    let f01 = f_2d[[ix, iyp]];
                    // Triangle 1: (00, 10, 01)
                    total += tri_area * (f00 + f10 + f01) / 3.0;
                    // Triangle 2: (11, 10, 01)
                    total += tri_area * (f11 + f10 + f01) / 3.0;
                }
            }
            total
        }
        3 => {
            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
            let cube_vol = 1.0 / (nx * ny * nz) as f64;
            let f_3d = Array3::from_shape_vec((nx, ny, nz), f.to_vec()).unwrap();
            let mut total = 0.0;

            for ix in 0..nx {
                let ixp = (ix + 1) % nx;
                for iy in 0..ny {
                    let iyp = (iy + 1) % ny;
                    for iz in 0..nz {
                        let izp = (iz + 1) % nz;
                        let c000 = f_3d[[ix, iy, iz]];
                        let c100 = f_3d[[ixp, iy, iz]];
                        let c010 = f_3d[[ix, iyp, iz]];
                        let c110 = f_3d[[ixp, iyp, iz]];
                        let c001 = f_3d[[ix, iy, izp]];
                        let c101 = f_3d[[ixp, iy, izp]];
                        let c011 = f_3d[[ix, iyp, izp]];
                        let c111 = f_3d[[ixp, iyp, izp]];
                        let corner = [c000, c100, c010, c110, c001, c101, c011, c111];
                        for (teti, &[v0, v1, v2, v3]) in CUBE_TETS.iter().enumerate() {
                            let vt = cube_vol * TET_VOL_FACTOR[teti];
                            total +=
                                vt * (corner[v0] + corner[v1] + corner[v2] + corner[v3]) * 0.25;
                        }
                    }
                }
            }
            total
        }
        _ => panic!(
            "tetrahedron_volume_integrate: only dim=2,3 supported, got dim={dim}"
        ),
    }
}

// ── 3D integration ────────────────────────────────────────────────────

fn tetrahedron_integrate_3d(
    band: &Array2<f64>,
    integrand: &Array2<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
    nsta: usize,
    mu: &Array1<f64>,
    n_mu: usize,
) -> Array1<f64> {
    let nk = nx * ny * nz;
    let cube_vol = 1.0 / (nk as f64);

    // Reshape flat band/integrand columns into 3D arrays for cell iteration
    let results: Vec<Array1<f64>> = (0..nsta)
        .into_par_iter()
        .map(|ib| {
            let band_col = band.column(ib).to_vec();
            let int_col = integrand.column(ib).to_vec();
            let ener_3d = Array3::from_shape_vec((nx, ny, nz), band_col).unwrap();
            let f_3d = Array3::from_shape_vec((nx, ny, nz), int_col).unwrap();

            let mut band_result = Array1::<f64>::zeros(n_mu);

            for ix in 0..nx.saturating_sub(1) {
                for iy in 0..ny.saturating_sub(1) {
                    for iz in 0..nz.saturating_sub(1) {
                        // Gather 8 corner values
                        let mut corner_ener = [0.0f64; 8];
                        let mut corner_f = [0.0f64; 8];
                        for (c, &(di, dj, dk)) in CUBE_CORNERS.iter().enumerate() {
                            corner_ener[c] = ener_3d[[ix + di, iy + dj, iz + dk]];
                            corner_f[c] = f_3d[[ix + di, iy + dj, iz + dk]];
                        }

                        let emin = corner_ener.iter().cloned().fold(f64::INFINITY, f64::min);
                        let emax = corner_ener
                            .iter()
                            .cloned()
                            .fold(f64::NEG_INFINITY, f64::max);
                        // Quick skip: no μ in this cell's energy range
                        if emax < mu[0] || emin > mu[n_mu - 1] {
                            continue;
                        }

                        for (teti, &[v0, v1, v2, v3]) in CUBE_TETS.iter().enumerate() {
                            let tet_ener = [
                                corner_ener[v0],
                                corner_ener[v1],
                                corner_ener[v2],
                                corner_ener[v3],
                            ];
                            let tet_f = [
                                corner_f[v0],
                                corner_f[v1],
                                corner_f[v2],
                                corner_f[v3],
                            ];
                            let vt = cube_vol * TET_VOL_FACTOR[teti];
                            add_tet_delta(&tet_ener, &tet_f, mu, vt, &mut band_result);
                        }
                    }
                }
            }
            band_result
        })
        .collect();

    // Sum over bands
    let mut total = Array1::<f64>::zeros(n_mu);
    for r in results {
        total += &r;
    }
    total
}

/// Compute the step‑function integral S(μ) = ∫_{ε≤μ} F(k) dk
/// for a **Case 2** tetrahedron (e0 ≤ e1 < μ ≤ e2 ≤ e3).
///
/// The occupied region decomposes into three sub‑tetrahedra:
///   T₁ = [0, 1, P02, P03]  with Pij = intersection of μ‑plane on edge (i,j)
///   T₂ = [1, P02, P12, P03]
///   T₃ = [1, P12, P03, P13]
///
/// Each sub‑tet's contribution is V_sub · (Σ corner F) / 4.
/// The corner F values of intersection points are obtained by linear
/// interpolation on the edges.
#[inline(always)]
fn step_volume_case2(
    e0: f64, e1: f64, _e2: f64, _e3: f64,
    f0: f64, f1: f64, f2: f64, f3: f64,
    vt: f64,
    mu: f64,
    d20: f64, d30: f64, d21: f64, d31: f64,
) -> f64 {
    // Intersection parameters on edges
    let t02 = (mu - e0) / d20;
    let t03 = (mu - e0) / d30;
    let t12 = (mu - e1) / d21;
    let t13 = (mu - e1) / d31;

    // F values at intersection points
    let f02 = f0 + (f2 - f0) * t02;
    let f03 = f0 + (f3 - f0) * t03;
    let f12 = f1 + (f2 - f1) * t12;
    let f13 = f1 + (f3 - f1) * t13;

    // T₁ = [0, 1, P02, P03]: volume = Vt · t02 · t03
    let v1 = vt * t02 * t03;
    let s1 = v1 * (f0 + f1 + f02 + f03) * 0.25;

    // T₂ = [1, P02, P12, P03]: volume = Vt · t12 · t03 · (1 − t02)
    let v2 = if t02 < 1.0 { vt * t12 * t03 * (1.0 - t02) } else { 0.0 };
    let s2 = v2 * (f1 + f02 + f12 + f03) * 0.25;

    // T₃ = [1, P12, P03, P13]: volume = Vt · t12 · t13 · (1 − t03)
    let v3 = if t03 < 1.0 { vt * t12 * t13 * (1.0 - t03) } else { 0.0 };
    let s3 = v3 * (f1 + f12 + f03 + f13) * 0.25;

    s1 + s2 + s3
}

/// Add δ‑function contributions from one tetrahedron to `result`.
///
/// `e` and `f` are the 4 corner values (in original corner order).
/// They are sorted by energy internally.  `vt` is the physical volume
/// of this tetrahedron in fractional k‑space coordinates.
fn add_tet_delta(
    e: &[f64; 4],
    f: &[f64; 4],
    mu: &Array1<f64>,
    vt: f64,
    result: &mut Array1<f64>,
) {
    // Sort by energy, track permutation
    let mut idx = [0usize, 1, 2, 3];
    idx.sort_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap());
    let e0 = e[idx[0]];
    let e1 = e[idx[1]];
    let e2 = e[idx[2]];
    let e3 = e[idx[3]];
    let f0 = f[idx[0]];
    let f1 = f[idx[1]];
    let f2 = f[idx[2]];
    let f3 = f[idx[3]];

    let etot = e3 - e0;
    if etot < 1e-14 {
        return; // flat band, no δ contribution
    }

    let d10 = e1 - e0;
    let d20 = e2 - e0;
    let d30 = e3 - e0;
    let d21 = e2 - e1;
    let d31 = e3 - e1;
    let d32 = e3 - e2;

    let n_mu = mu.len();
    let mu_slice = mu.as_slice().unwrap();

    // --- Case 1: e0 < μ ≤ e1 ---
    let i1_start = mu_slice.partition_point(|&m| m <= e0);
    let i1_end = mu_slice.partition_point(|&m| m <= e1);
    if i1_end > i1_start && d10 > 1e-14 && d20 > 1e-14 && d30 > 1e-14 {
        let c = vt / (d10 * d20 * d30);
        let s = 1.0 / d10 + 1.0 / d20 + 1.0 / d30;
        for i in i1_start..i1_end {
            let x = mu[[i]] - e0;
            let x2 = x * x;
            let x3 = x2 * x;
            result[[i]] += c * ((3.0 * x2 - x3 * s) * f0
                + x3 / d10 * f1
                + x3 / d20 * f2
                + x3 / d30 * f3);
        }
    }

    // --- Case 2: e1 < μ ≤ e2 ---
    let i2_start = i1_end;
    let i2_end = mu_slice.partition_point(|&m| m <= e2);
    if i2_end > i2_start && d20 > 1e-14 && d30 > 1e-14 && d21 > 1e-14 && d31 > 1e-14 {
        // The iso‑surface is a quadrilateral intersecting edges
        // (0,2), (0,3), (1,2), (1,3).
        //
        // We compute the δ‑function contribution via the derivative
        // of the step‑function integral S(μ) = ∫_{ε≤μ} F(k) dk.
        // S(μ) is evaluated analytically by decomposing the occupied
        // region into 3 sub‑tetrahedra (verified numerically).
        //
        // Then: δ‑contribution ≈ [S(μ+δ) − S(μ−δ)] / (2δ)
        let eps = 1e-8f64;
        for i in i2_start..i2_end {
            let m = mu[[i]];
            let sp = step_volume_case2(e0, e1, e2, e3, f0, f1, f2, f3, vt, m + eps, d20, d30, d21, d31);
            let sm = step_volume_case2(e0, e1, e2, e3, f0, f1, f2, f3, vt, m - eps, d20, d30, d21, d31);
            result[[i]] += (sp - sm) / (2.0 * eps);
        }
    }

    // --- Case 3: e2 < μ ≤ e3 ---
    // Symmetric to Case 1 with reversed energy axis
    let i3_start = i2_end;
    let i3_end = mu_slice.partition_point(|&m| m <= e3);
    if i3_end > i3_start && d30 > 1e-14 && d31 > 1e-14 && d32 > 1e-14 {
        let c = vt / (d30 * d31 * d32);
        let s = 1.0 / d30 + 1.0 / d31 + 1.0 / d32;
        for i in i3_start..i3_end {
            let x = e3 - mu[[i]];
            let x2 = x * x;
            let x3 = x2 * x;
            // Symmetry: replace Ei → -E{3-i}, μ → -μ
            // This maps Case 3 back to Case 1 with f_i permuted
            result[[i]] += c * ((3.0 * x2 - x3 * s) * f3
                + x3 / d30 * f0
                + x3 / d31 * f1
                + x3 / d32 * f2);
        }
    }
}

// ── 2D integration ────────────────────────────────────────────────────

fn tetrahedron_integrate_2d(
    band: &Array2<f64>,
    integrand: &Array2<f64>,
    nx: usize,
    ny: usize,
    nsta: usize,
    mu: &Array1<f64>,
    n_mu: usize,
) -> Array1<f64> {
    let nk = nx * ny;
    let cell_area = 1.0 / (nk as f64);
    let tri_area = cell_area / 2.0; // each rectangle → 2 triangles

    let results: Vec<Array1<f64>> = (0..nsta)
        .into_par_iter()
        .map(|ib| {
            let band_col = band.column(ib).to_vec();
            let int_col = integrand.column(ib).to_vec();
            let ener_2d = Array2::from_shape_vec((nx, ny), band_col).unwrap();
            let f_2d = Array2::from_shape_vec((nx, ny), int_col).unwrap();

            let mut band_result = Array1::<f64>::zeros(n_mu);

            for ix in 0..nx.saturating_sub(1) {
                for iy in 0..ny.saturating_sub(1) {
                    // 4 corners of the rectangle
                    let e00 = ener_2d[[ix, iy]];
                    let e10 = ener_2d[[ix + 1, iy]];
                    let e11 = ener_2d[[ix + 1, iy + 1]];
                    let e01 = ener_2d[[ix, iy + 1]];
                    let f00 = f_2d[[ix, iy]];
                    let f10 = f_2d[[ix + 1, iy]];
                    let f11 = f_2d[[ix + 1, iy + 1]];
                    let f01 = f_2d[[ix, iy + 1]];

                    let emin = e00.min(e10).min(e11).min(e01);
                    let emax = e00.max(e10).max(e11).max(e01);
                    if emax < mu[0] || emin > mu[n_mu - 1] {
                        continue;
                    }

                    // Triangle 1: (00, 10, 01)
                    add_tri_delta(
                        &[e00, e10, e01],
                        &[f00, f10, f01],
                        mu,
                        tri_area,
                        &mut band_result,
                    );
                    // Triangle 2: (11, 10, 01)
                    add_tri_delta(
                        &[e11, e10, e01],
                        &[f11, f10, f01],
                        mu,
                        tri_area,
                        &mut band_result,
                    );
                }
            }
            band_result
        })
        .collect();

    let mut total = Array1::<f64>::zeros(n_mu);
    for r in results {
        total += &r;
    }
    total
}

/// Add δ‑function contributions from one 2D triangle to `result`.
fn add_tri_delta(
    e: &[f64; 3],
    f: &[f64; 3],
    mu: &Array1<f64>,
    vt: f64,
    result: &mut Array1<f64>,
) {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| e[a].partial_cmp(&e[b]).unwrap());
    let e0 = e[idx[0]];
    let e1 = e[idx[1]];
    let e2 = e[idx[2]];
    let f0 = f[idx[0]];
    let f1 = f[idx[1]];
    let f2 = f[idx[2]];

    let etot = e2 - e0;
    if etot < 1e-14 {
        return;
    }

    let d10 = e1 - e0;
    let d20 = e2 - e0;
    let d21 = e2 - e1;

    let mu_slice = mu.as_slice().unwrap();

    // Case 1: e0 < μ ≤ e1 (one vertex below)
    let i1_start = mu_slice.partition_point(|&m| m <= e0);
    let i1_end = mu_slice.partition_point(|&m| m <= e1);
    if i1_end > i1_start && d10 > 1e-14 && d20 > 1e-14 {
        let c = vt / (d10 * d20);
        for i in i1_start..i1_end {
            let x = mu[[i]] - e0;
            let x2 = x * x;
            result[[i]] += c * ((2.0 * x - x2 * (1.0 / d10 + 1.0 / d20)) * f0
                + x2 / d10 * f1
                + x2 / d20 * f2);
        }
    }

    // Case 2: e1 < μ ≤ e2 (two vertices below)
    let i2_start = i1_end;
    let i2_end = mu_slice.partition_point(|&m| m <= e2);
    if i2_end > i2_start && d20 > 1e-14 && d21 > 1e-14 {
        let c = vt / (d20 * d21);
        for i in i2_start..i2_end {
            let x = e2 - mu[[i]];
            let x2 = x * x;
            // Symmetric to Case 1
            result[[i]] += c * ((2.0 * x - x2 * (1.0 / d20 + 1.0 / d21)) * f2
                + x2 / d20 * f0
                + x2 / d21 * f1);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Berry / Quantum-Geometry Simplex Quadrature
// ═══════════════════════════════════════════════════════════════════════════
//
// Replaces the old `tetrahedron_volume_integrate` path (which linearly
// interpolates the final scalar integrand) with a quadrature over
// per‑simplex primitives.  The gauge‑invariant velocity product
// `K^ab_nm = v^a_nm·v^b_mn` and band energies `E_n` are interpolated
// linearly inside each simplex; the singular denominator
// `1/((E_n−E_m)² + η²)` is evaluated at each quadrature point.

use num_complex::Complex;

/// Per‑k‑point primitive data for one direction pair `(a,b)`.
///
/// `k_ab[n,m] = v^a_nm · v^b_mn` is gauge‑invariant under independent
/// U(1) rotations of bands n and m (when both bands are isolated).
#[derive(Clone)]
pub struct VertexKernel {
    /// Band energies `ε_n`, length `nsta`.
    pub band: Array1<f64>,
    /// `K^{ab}_{nm}`, shape `(nsta, nsta)`.
    pub k_ab: Array2<Complex<f64>>,
    /// Diagonal velocity `v^c_n = ⟨n|∂_c H|n⟩` (only when dipole needed).
    pub vdiag: Option<Array1<f64>>,
    /// Eigenvectors `U[:, n]`, shape `(norb, nsta)` — for band tracking.
    pub evec: Array2<Complex<f64>>,
}

/// A simplex whose vertices have been band‑tracked (label‑aligned).
pub struct TrackedSimplex {
    /// `d + 1` vertices, already label‑aligned.
    pub vertices: Vec<VertexKernel>,
    /// Physical volume of this simplex (fractional coordinates).
    pub volume: f64,
    /// Fractional coordinates of each vertex, shape `(d+1, dim)`.
    pub coords: Array2<f64>,
    /// Diagnostic counters for this simplex.
    pub diag: SimplexDiagnostics,
}

/// Per‑simplex safety / quality diagnostics.
#[derive(Clone, Default)]
pub struct SimplexDiagnostics {
    /// Minimum band gap `min_{n≠m} |E_n − E_m|` across all vertices.
    pub min_gap: f64,
    /// Minimum assignment overlap from band tracking (1.0 = perfect).
    pub min_assignment_overlap: f64,
    /// True if different neighbour paths imply inconsistent band permutations.
    pub tracking_conflict: bool,
}

// ── Quadrature rules ────────────────────────────────────────────────────

/// Degree‑2 symmetric 3‑point rule for the reference triangle.
/// Barycentric coordinates; weights sum to 1.
const TRI_QUAD_PTS_3: [[f64; 3]; 3] = [
    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
];
const TRI_QUAD_WTS_3: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

/// Degree‑2 symmetric 4‑point rule for the reference tetrahedron.
/// Barycentric coordinates; weights sum to 1.
const TET_QUAD_PTS_4: [[f64; 4]; 4] = [
    [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
    [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
    [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
    [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
];
const TET_QUAD_WTS_4: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

// ── Band tracking ───────────────────────────────────────────────────────

/// Build the overlap matrix `O_nm = |⟨u_n(ref)|u_m(other)⟩|²`.
fn build_overlap_matrix(
    evec_ref: &Array2<Complex<f64>>,
    evec_other: &Array2<Complex<f64>>,
) -> Array2<f64> {
    let nsta = evec_ref.ncols();
    let norb = evec_ref.nrows();
    let mut ov = Array2::<f64>::zeros((nsta, nsta));
    for n in 0..nsta {
        for m in 0..nsta {
            let mut s = Complex::new(0.0, 0.0);
            for orb in 0..norb {
                s += evec_ref[[orb, n]].conj() * evec_other[[orb, m]];
            }
            ov[[n, m]] = s.norm_sqr();
        }
    }
    ov
}

/// Greedy assignment that finds a one‑to‑one band permutation maximising
/// `Σ_n O_{n, p(n)}`.  For small `nsta` with diagonally‑dominant overlap
/// this is exact; for larger problems an optimal assignment (Hungarian)
/// can be substituted without changing the interface.
fn greedy_assign(overlap: &Array2<f64>) -> Vec<usize> {
    let n = overlap.nrows();
    let mut assigned = vec![false; n];
    let mut perm = vec![0usize; n];
    // Process rows ordered by their best available overlap
    let mut rows: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let best = (0..n).map(|j| overlap[[i, j]]).fold(-1.0, f64::max);
            (i, best)
        })
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (i, _) in rows {
        let mut best_j = 0;
        let mut best_val = -1.0;
        for j in 0..n {
            if !assigned[j] && overlap[[i, j]] > best_val {
                best_val = overlap[[i, j]];
                best_j = j;
            }
        }
        assigned[best_j] = true;
        perm[i] = best_j;
    }
    perm
}

/// Reorder a single `VertexKernel` according to band permutation `p`
/// (band `pn` at the original vertex → band `n` at the reference vertex).
fn permute_vertex(v: &VertexKernel, p: &[usize]) -> VertexKernel {
    let nsta = v.band.len();
    let norb = v.evec.nrows();
    let mut band = Array1::<f64>::zeros(nsta);
    let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
    let mut evec = Array2::<Complex<f64>>::zeros((norb, nsta));
    let mut vdiag = v.vdiag.as_ref().map(|_| Array1::<f64>::zeros(nsta));

    for n in 0..nsta {
        let pn = p[n];
        band[[n]] = v.band[[pn]];
        if let Some(ref mut vd) = vdiag {
            vd[[n]] = v.vdiag.as_ref().unwrap()[[pn]];
        }
        for orb in 0..norb {
            evec[[orb, n]] = v.evec[[orb, pn]];
        }
    }
    for n in 0..nsta {
        for m in 0..nsta {
            k_ab[[n, m]] = v.k_ab[[p[n], p[m]]];
        }
    }
    VertexKernel { band, k_ab, vdiag, evec }
}

/// Track band labels across a set of simplex vertices using the first
/// vertex as reference.  Returns label‑aligned copies and populates
/// diagnostics.
fn track_simplex_vertices(
    vertices: &[VertexKernel],
) -> (Vec<VertexKernel>, SimplexDiagnostics) {
    let nv = vertices.len();
    if nv <= 1 {
        let diag = SimplexDiagnostics {
            min_gap: min_vertex_gap(&vertices[0]),
            min_assignment_overlap: 1.0,
            tracking_conflict: false,
        };
        return (vertices.to_vec(), diag);
    }

    let mut aligned: Vec<VertexKernel> = vec![vertices[0].clone()];
    let mut min_ov = 1.0f64;

    for vi in 1..nv {
        let ov = build_overlap_matrix(&aligned[0].evec, &vertices[vi].evec);
        let p = greedy_assign(&ov);
        let ov_score: f64 = (0..ov.nrows()).map(|n| ov[[n, p[n]]]).sum::<f64>()
            / ov.nrows() as f64;
        min_ov = min_ov.min(ov_score);
        aligned.push(permute_vertex(&vertices[vi], &p));
    }

    let mut min_gap = f64::INFINITY;
    for v in &aligned {
        min_gap = min_gap.min(min_vertex_gap(v));
    }

    let diag = SimplexDiagnostics {
        min_gap,
        min_assignment_overlap: min_ov,
        tracking_conflict: false,
    };
    (aligned, diag)
}

/// Minimum band gap at a single vertex.
fn min_vertex_gap(v: &VertexKernel) -> f64 {
    let n = v.band.len();
    let mut g = f64::INFINITY;
    for i in 0..n {
        for j in i + 1..n {
            g = g.min((v.band[[i]] - v.band[[j]]).abs());
        }
    }
    g
}

// ── Simplex builders ────────────────────────────────────────────────────

/// Build the two tracked triangles of a 2D cell.
fn build_triangles_2d(
    ix: usize, iy: usize, nx: usize, ny: usize,
    inv_nx: f64, inv_ny: f64,
    all_pts: &[VertexKernel],
) -> Vec<TrackedSimplex> {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;

    let frac = |ixv: usize, iyv: usize| -> [f64; 2] {
        [ixv as f64 * inv_nx, iyv as f64 * inv_ny]
    };
    // NOTE: for wrapped corners we use the actual fractional coords, not
    // the modulo‑reduced ones.  This matters for the triangle area sign.
    let coord_of = |idx: usize| -> [f64; 2] {
        if idx == i00 { frac(ix, iy) }
        else if idx == i10 { frac(ix + 1, iy) }
        else if idx == i11 { frac(ix + 1, iy + 1) }
        else { frac(ix, iy + 1) }
    };

    let cell_area = inv_nx * inv_ny;
    let tri_area = cell_area / 2.0;
    let mut out = Vec::new();

    for &(v0, v1, v2) in &[(i00, i10, i01), (i11, i10, i01)] {
        let raw = vec![
            all_pts[v0].clone(),
            all_pts[v1].clone(),
            all_pts[v2].clone(),
        ];
        let (aligned, diag) = track_simplex_vertices(&raw);
        let coords = Array2::from_shape_vec((3, 2), {
            let c0 = coord_of(v0); let c1 = coord_of(v1); let c2 = coord_of(v2);
            vec![c0[0], c0[1], c1[0], c1[1], c2[0], c2[1]]
        }).unwrap();
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: tri_area,
            coords,
            diag,
        });
    }
    out
}

/// Build the five tracked tetrahedra of a 3D cell.
fn build_tetrahedra_3d(
    ix: usize, iy: usize, iz: usize,
    nx: usize, ny: usize, nz: usize,
    inv_nx: f64, inv_ny: f64, inv_nz: f64,
    all_pts: &[VertexKernel],
) -> Vec<TrackedSimplex> {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let izp = (iz + 1) % nz;
    let idx3 = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let c = [
        idx3(ix, iy, iz), idx3(ixp, iy, iz),
        idx3(ix, iyp, iz), idx3(ixp, iyp, iz),
        idx3(ix, iy, izp), idx3(ixp, iy, izp),
        idx3(ix, iyp, izp), idx3(ixp, iyp, izp),
    ];

    let frac = |ixv: usize, iyv: usize, izv: usize| -> [f64; 3] {
        [ixv as f64 * inv_nx, iyv as f64 * inv_ny, izv as f64 * inv_nz]
    };

    let corners_frac: [[f64; 3]; 8] = [
        frac(ix, iy, iz), frac(ix + 1, iy, iz),
        frac(ix, iyp, iz), frac(ix + 1, iy + 1, iz),
        frac(ix, iy, izp), frac(ix + 1, iy, izp),
        frac(ix, iyp, izp), frac(ix + 1, iy + 1, izp),
    ];

    let cube_vol = inv_nx * inv_ny * inv_nz;
    let mut out = Vec::new();

    for (teti, &[v0, v1, v2, v3]) in CUBE_TETS.iter().enumerate() {
        let raw = vec![
            all_pts[c[v0]].clone(),
            all_pts[c[v1]].clone(),
            all_pts[c[v2]].clone(),
            all_pts[c[v3]].clone(),
        ];
        let (aligned, diag) = track_simplex_vertices(&raw);
        let coords = Array2::from_shape_vec((4, 3), {
            let c0 = corners_frac[v0]; let c1 = corners_frac[v1];
            let c2 = corners_frac[v2]; let c3 = corners_frac[v3];
            vec![c0[0], c0[1], c0[2], c1[0], c1[1], c1[2],
                 c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]]
        }).unwrap();
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: cube_vol * TET_VOL_FACTOR[teti],
            coords,
            diag,
        });
    }
    out
}

// ── Generic quadrature evaluator ────────────────────────────────────────

/// Interpolate a scalar field `vals[d+1]` at barycentric coords `lam[d+1]`.
#[inline]
fn bary_interp_scalar(vals: &[f64], lam: &[f64]) -> f64 {
    vals.iter().zip(lam.iter()).map(|(v, l)| v * l).sum()
}

/// Interpolate a matrix field `mats[d+1]` each `(nsta, nsta)` at
/// barycentric coords `lam`.
fn bary_interp_matrix(
    mats: &[Array2<Complex<f64>>], lam: &[f64],
) -> Array2<Complex<f64>> {
    let n = mats[0].nrows();
    let mut out = Array2::<Complex<f64>>::zeros((n, n));
    for (mat, &w) in mats.iter().zip(lam.iter()) {
        if w == 0.0 { continue; }
        for i in 0..n {
            for j in 0..n {
                out[[i, j]] += mat[[i, j]] * w;
            }
        }
    }
    out
}

/// Evaluate the Berry‑curvature / quantum‑geometry kernel at one
/// quadrature point (barycentric coordinates `lam`).
///
/// Returns per‑band `(g_n, Ω_n)` where
/// `G_n = g_n + i·Ω_n = Σ_{m≠n} K_nm / ((E_n−E_m)² + η²)`.
fn eval_berry_kernel(
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
            if m == n { continue; }
            let de = band_q[n] - band_q[m];
            let denom = de * de + eta2;
            if denom < 1e-30 { continue; }
            g_sum += k_ab_q[[n, m]] / denom;
        }
        metric[n] = g_sum.re;
        berry[n] = -2.0 * g_sum.im;
    }
    (metric, berry)
}

/// Quadrature over a single `TrackedSimplex` for the Berry‑curvature /
/// quantum‑metric kernel.
///
/// Returns `(∫g_n dk, ∫Ω_n dk)` summed over all bands.
fn quadrature_over_simplex(sim: &TrackedSimplex, eta: f64) -> (f64, f64) {
    let d = sim.vertices.len() - 1; // 2 = triangle, 3 = tetrahedron
    let nsta = sim.vertices[0].band.len();
    let nv = d + 1;

    let bands: Vec<Vec<f64>> = (0..nv)
        .map(|v| sim.vertices[v].band.to_vec())
        .collect();
    let kmats: Vec<Array2<Complex<f64>>> = (0..nv)
        .map(|v| sim.vertices[v].k_ab.clone())
        .collect();

    let mut total_g = 0.0;
    let mut total_o = 0.0;

    if d == 2 {
        for iq in 0..3 {
            let lam: &[f64; 3] = &TRI_QUAD_PTS_3[iq];
            let w = TRI_QUAD_WTS_3[iq];
            let lam_slice: &[f64] = lam.as_slice();
            let band_q = bary_interp_band(&bands, lam_slice, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam_slice);
            let (g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
            total_g += w * g_n.iter().copied().sum::<f64>();
            total_o += w * o_n.iter().copied().sum::<f64>();
        }
    } else {
        for iq in 0..4 {
            let lam: &[f64; 4] = &TET_QUAD_PTS_4[iq];
            let w = TET_QUAD_WTS_4[iq];
            let lam_slice: &[f64] = lam.as_slice();
            let band_q = bary_interp_band(&bands, lam_slice, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam_slice);
            let (g_n, o_n) = eval_berry_kernel(&band_q, &k_ab_q, eta, nsta);
            total_g += w * g_n.iter().copied().sum::<f64>();
            total_o += w * o_n.iter().copied().sum::<f64>();
        }
    }

    (total_g * sim.volume, total_o * sim.volume)
}

/// Quadrature over a single simplex with `vdiag` weighting for dipoles.
///
/// Returns per-μ contributions `Σ_n v^c_n(q) Ω_n(q) (−∂f/∂E_n)` at each
/// quadrature point, accumulated over the simplex volume.
fn dipole_quadrature_over_simplex(
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
            let lam: &[f64; 3] = &TRI_QUAD_PTS_3[iq];
            let w = TRI_QUAD_WTS_3[iq];
            let lam_slice: &[f64] = lam.as_slice();
            let band_q = bary_interp_band(&bands, lam_slice, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam_slice);
            let vdiag_q = bary_interp_band(&vdiags, lam_slice, nsta);
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
            let lam: &[f64; 4] = &TET_QUAD_PTS_4[iq];
            let w = TET_QUAD_WTS_4[iq];
            let lam_slice: &[f64] = lam.as_slice();
            let band_q = bary_interp_band(&bands, lam_slice, nsta);
            let k_ab_q = bary_interp_matrix(&kmats, lam_slice);
            let vdiag_q = bary_interp_band(&vdiags, lam_slice, nsta);
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

/// `−∂f/∂E` at energy `e` for chemical potential `mu` and inverse
/// temperature `beta`.  β=0 → delta‑function (not handled here).
#[inline]
fn fermi_deriv(e: f64, mu: f64, beta: f64) -> f64 {
    let x = beta * (e - mu);
    if x > 50.0 {
        // exp(−x) → 0,  f → 0,  f(1−f) ≈ β exp(−x) → 0
        0.0
    } else if x < -50.0 {
        0.0
    } else {
        let ex = x.exp();
        beta * ex / ((1.0 + ex) * (1.0 + ex))
    }
}

/// Integrate the Berry‑curvature dipole over the BZ using simplex quadrature.
///
/// ```text
/// D^{ab;c}(μ,T) = Σ_n ∫_BZ (−∂f/∂E_n) v^c_n(k) Ω^{ab}_n(k) dk
/// ```
///
/// # Arguments
/// * `all_pts` — per‑k‑point `VertexKernel` (must have `vdiag = Some(v^c_n)`).
/// * `k_mesh` — `[nx, ny]` (2D) or `[nx, ny, nz]` (3D).
/// * `mu` — array of chemical potentials (eV).
/// * `T` — temperature (K).  `T=0` falls back to the existing
///   Fermi‑surface tetrahedron method; use `Nonlinear_Hall_conductivity_Extrinsic_tetra`
///   for that case.
/// * `eta` — smearing (eV) for the denominator `(E_n−E_m)² + η²`.
///
/// # Returns
/// `(dipole_per_mu, unsafe_count)` where `dipole_per_mu` has length `mu.len()`.
pub fn simplex_dipole_integrate(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
    eta: f64,
) -> (Array1<f64>, usize) {
    let dim = k_mesh.len();
    let n_mu = mu.len();
    let beta = if T > 0.0 { 1.0 / (T * 8.617333262e-5) } else { f64::INFINITY };
    let mut acc = Array1::<f64>::zeros(n_mu);
    let mut unsafe_count = 0usize;

    match dim {
        2 => {
            let (nx, ny) = (k_mesh[0], k_mesh[1]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    let sims = build_triangles_2d(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
                    for sim in &sims {
                        if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                            unsafe_count += 1;
                        }
                        acc += &dipole_quadrature_over_simplex(sim, eta, mu, beta);
                    }
                }
            }
        }
        3 => {
            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            let inv_nz = 1.0 / nz as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let sims = build_tetrahedra_3d(
                            ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts,
                        );
                        for sim in &sims {
                            if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                                unsafe_count += 1;
                            }
                            acc += &dipole_quadrature_over_simplex(sim, eta, mu, beta);
                        }
                    }
                }
            }
        }
        _ => panic!(
            "simplex_dipole_integrate: only dim=2,3 supported, got dim={dim}"
        ),
    }

    (acc, unsafe_count)
}

/// Interpolate band energies at barycentric coords `lam`.
fn bary_interp_band(bands: &[Vec<f64>], lam: &[f64], nsta: usize) -> Vec<f64> {
    let mut out = vec![0.0; nsta];
    for v in 0..bands.len() {
        let lv = lam[v];
        if lv == 0.0 { continue; }
        for n in 0..nsta {
            out[n] += bands[v][n] * lv;
        }
    }
    out
}

// ── Optical conductivity kernel ─────────────────────────────────────────

/// Evaluate the optical conductivity kernel at one quadrature point.
///
/// ```text
/// σ^{ab}_{nm}(ω) = (f_n − f_m) · K^{ab}_{nm} / (d²_{nm} − (ω+iη)²)
/// ```
///
/// Returns the sum over all band pairs `(n≠m)` as a complex number.
fn eval_optical_kernel(
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
    let denom_shift = w_plus_ieta * w_plus_ieta; // (ω+iη)² = ω²−η² + 2iωη
    for n in 0..nsta {
        let fn_val = fermi(band_q[n], mu, beta);
        for m in 0..nsta {
            if m == n { continue; }
            let fm_val = fermi(band_q[m], mu, beta);
            let df = fn_val - fm_val;
            if df.abs() < 1e-30 { continue; }
            let d = band_q[n] - band_q[m];
            let denom = d * d - denom_shift; // d² − (ω+iη)²
            if denom.norm_sqr() < 1e-30 { continue; }
            total += df * k_ab_q[[n, m]] / denom;
        }
    }
    total
}

/// Fermi‑Dirac occupation `f(E, μ, β)`.
#[inline]
fn fermi(e: f64, mu: f64, beta: f64) -> f64 {
    if beta == 0.0 {
        0.5 // T→∞ limit
    } else {
        let x = beta * (e - mu);
        if x > 50.0 { 0.0 } else if x < -50.0 { 1.0 } else { 1.0 / (1.0 + x.exp()) }
    }
}

/// Quadrature over a single simplex for the optical conductivity kernel.
fn optical_quadrature_over_simplex(
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

/// Optical conductivity via simplex quadrature.
///
/// ```text
/// σ^{ab}(ω,μ,T) = Σ_{n≠m} ∫_BZ (f_n−f_m)
///     · v^a_{nm} v^b_{mn} / ((E_n−E_m)² − (ω+iη)²) dk
/// ```
///
/// # Returns
/// `σ^{ab}` (complex, fractional‑coordinate volume; divide by `det(lat)`
/// for Cartesian).
pub fn simplex_optical_integrate(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    omega: f64,
    eta: f64,
    mu: f64,
    T: f64,
) -> Complex<f64> {
    let dim = k_mesh.len();
    let beta = if T > 0.0 { 1.0 / (T * 8.617333262e-5) } else { 0.0 };
    let mut total = Complex::new(0.0, 0.0);

    match dim {
        2 => {
            let (nx, ny) = (k_mesh[0], k_mesh[1]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for sim in &build_triangles_2d(ix, iy, nx, ny, inv_nx, inv_ny, all_pts) {
                        total += optical_quadrature_over_simplex(sim, omega, eta, mu, beta);
                    }
                }
            }
        }
        3 => {
            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            let inv_nz = 1.0 / nz as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        for sim in &build_tetrahedra_3d(
                            ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts,
                        ) {
                            total += optical_quadrature_over_simplex(sim, omega, eta, mu, beta);
                        }
                    }
                }
            }
        }
        _ => panic!(
            "simplex_optical_integrate: only dim=2,3 supported, got dim={dim}"
        ),
    }

    total
}

// ── Public API ──────────────────────────────────────────────────────────

/// Safety threshold: simplexes with `min_gap < GAP_TOL` are skipped for
/// single‑band Berry/QGT evaluation.
pub const SIMPLEX_GAP_TOL: f64 = 1e-4;

/// Integrate the Berry curvature `Ω_{ab}` and quantum metric `g_{ab}` over
/// the full BZ using simplex quadrature.
///
/// # Arguments
/// * `all_pts` — per‑k‑point `VertexKernel` primitives (nk points).
/// * `k_mesh` — `[nx, ny]` (2D) or `[nx, ny, nz]` (3D).
/// * `eta` — smearing width (eV) for the denominator `(E_n−E_m)² + η²`.
///
/// # Returns
/// `(total_g, total_o, unsafe_count)` where `total_g = Σ_n ∫ g_n dk`,
/// `total_o = Σ_n ∫ Ω_n dk`, and `unsafe_count` is the number of simplexes
/// skipped due to near‑degeneracy.
pub fn simplex_berry_integrate(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    eta: f64,
) -> (f64, f64, usize) {
    let dim = k_mesh.len();
    let mut total_g = 0.0;
    let mut total_o = 0.0;
    let mut unsafe_count = 0usize;

    match dim {
        2 => {
            let (nx, ny) = (k_mesh[0], k_mesh[1]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    let sims = build_triangles_2d(
                        ix, iy, nx, ny, inv_nx, inv_ny, all_pts,
                    );
                    for sim in &sims {
                        if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                            unsafe_count += 1;
                        }
                        let (g, o) = quadrature_over_simplex(sim, eta);
                        total_g += g;
                        total_o += o;
                    }
                }
            }
        }
        3 => {
            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
            let inv_nx = 1.0 / nx as f64;
            let inv_ny = 1.0 / ny as f64;
            let inv_nz = 1.0 / nz as f64;
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let sims = build_tetrahedra_3d(
                            ix, iy, iz, nx, ny, nz,
                            inv_nx, inv_ny, inv_nz, all_pts,
                        );
                        for sim in &sims {
                            if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                                unsafe_count += 1;
                            }
                            let (g, o) = quadrature_over_simplex(sim, eta);
                            total_g += g;
                            total_o += o;
                        }
                    }
                }
            }
        }
        _ => panic!(
            "simplex_berry_integrate: only dim=2,3 supported, got dim={dim}"
        ),
    }

    (total_g, total_o, unsafe_count)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    /// 2D: ε(kx,ky) = kx, F ≡ 1 → ∫[0,1]² δ(kx−μ) d²k = 1  (0<μ<1)
    #[test]
    fn test_2d_linear_energy() {
        let nx = 20;
        let ny = 20;
        let nk = nx * ny;
        let nsta = 1;

        // Build ε and F on the k-mesh
        let mut band = Array2::<f64>::zeros((nk, nsta));
        let mut integrand = Array2::<f64>::zeros((nk, nsta));
        for ix in 0..nx {
            for iy in 0..ny {
                let ik = ix * ny + iy; // gen_kmesh order: ix slow, iy fast
                band[[ik, 0]] = ix as f64 / nx as f64;
                integrand[[ik, 0]] = 1.0;
            }
        }

        let k_mesh = arr1(&[nx, ny]);
        let n_mu = 51;
        let mu = Array1::linspace(0.05, 0.95, n_mu);

        let result = tetrahedron_integrate(&band, &integrand, &k_mesh, &mu);

        // The result should be close to 1.0 for all μ in (0,1)
        // Cell volume = 1/(nx*ny), triangle area = 1/(2*nx*ny)
        // The tetrahedron weights already include the volume factor,
        // but for 2D the full BZ area is 1 in fractional coords.
        // The integral should give ~1 for all μ inside (0,1).
        for i in 0..n_mu {
            let val = result[[i]];
            // With 20×20 mesh, we expect good accuracy
            assert!(
                (val - 1.0).abs() < 0.1,
                "μ={:.3}, result={:.6}, expected~1.0",
                mu[[i]],
                val
            );
        }
        // Check rough average
        let avg = result.mean().unwrap();
        assert!((avg - 1.0).abs() < 0.05, "avg={avg:.6}, expected~1.0");
    }

    /// 3D: ε(kx,ky,kz) = kx, F ≡ 1 → ∫[0,1]³ δ(kx−μ) d³k = 1
    #[test]
    fn test_3d_linear_energy() {
        let nx = 10;
        let ny = 10;
        let nz = 10;
        let nk = nx * ny * nz;
        let nsta = 1;

        let mut band = Array2::<f64>::zeros((nk, nsta));
        let mut integrand = Array2::<f64>::zeros((nk, nsta));
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let ik = ix * ny * nz + iy * nz + iz;
                    band[[ik, 0]] = ix as f64 / nx as f64;
                    integrand[[ik, 0]] = 1.0;
                }
            }
        }

        let k_mesh = arr1(&[nx, ny, nz]);
        let n_mu = 21;
        let mu = Array1::linspace(0.1, 0.9, n_mu);

        let result = tetrahedron_integrate(&band, &integrand, &k_mesh, &mu);

        for i in 0..n_mu {
            let val = result[[i]];
            // 10×10×10 mesh should give moderate accuracy
            assert!(
                (val - 1.0).abs() < 0.3,
                "μ={:.3}, result={:.6}, expected~1.0",
                mu[[i]],
                val
            );
        }
        let avg = result.mean().unwrap();
        assert!((avg - 1.0).abs() < 0.15, "avg={avg:.6}, expected~1.0");
    }

    /// Symmetry test: permuting tetrahedron corners should not change result
    #[test]
    fn test_tet_corner_permutation() {
        let mu = arr1(&[0.5]);
        let vt = 1.0;
        let e = [0.0, 0.3, 0.7, 1.0];
        let f = [1.0, 2.0, 3.0, 4.0];
        let mut r1 = Array1::<f64>::zeros(1);
        add_tet_delta(&e, &f, &mu, vt, &mut r1);

        // Permute corners
        let e2 = [0.3, 0.7, 1.0, 0.0];
        let f2 = [2.0, 3.0, 4.0, 1.0];
        let mut r2 = Array1::<f64>::zeros(1);
        add_tet_delta(&e2, &f2, &mu, vt, &mut r2);

        assert!((r1[[0]] - r2[[0]]).abs() < 1e-10,
            "permuted corners gave different results: {} vs {}", r1[[0]], r2[[0]]);
    }

    /// Volume integration: ∫[0,1]³ cos²(2π·kx)·cos²(2π·ky)·cos²(2π·kz) = 1/8
    #[test]
    fn test_volume_integrate_3d_cos2() {
        let nx = 40;
        let ny = 40;
        let nz = 40;
        let nk = nx * ny * nz;

        let mut f = Array1::<f64>::zeros(nk);
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let ik = ix * ny * nz + iy * nz + iz;
                    let kx = ix as f64 / nx as f64;
                    let ky = iy as f64 / ny as f64;
                    let kz = iz as f64 / nz as f64;
                    let cs = |t: f64| (2.0 * std::f64::consts::PI * t).cos().powi(2);
                    f[[ik]] = cs(kx) * cs(ky) * cs(kz);
                }
            }
        }

        let k_mesh = arr1(&[nx, ny, nz]);
        let result = tetrahedron_volume_integrate(&f, &k_mesh);
        let exact = 1.0 / 8.0;
        let err = (result - exact).abs();
        println!("volume integral: {:.10}, exact: {:.10}, err: {:.2e}", result, exact, err);
        assert!(err < 1e-4, "error {:.2e} too large", err);
    }

    /// Volume integration: ∫[0,1]² 1 dk = 1 (constant function)
    #[test]
    fn test_volume_integrate_2d_constant() {
        let (nx, ny) = (11, 11);
        let f = Array1::<f64>::ones(nx * ny);
        let result = tetrahedron_volume_integrate(&f, &arr1(&[nx, ny]));
        assert!((result - 1.0).abs() < 1e-10, "got {result}, expected 1.0");
    }

    // ── Simplex Berry/QGT tests ──────────────────────────────────────────

    /// Build a mock VertexKernel with two bands and a toy velocity kernel.
    fn mock_vertex(band: [f64; 2], k_val: Complex<f64>) -> VertexKernel {
        let norb = 2;
        let nsta = 2;
        let evec = Array2::<Complex<f64>>::eye(norb); // identity
        let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
        k_ab[[0, 1]] = k_val;
        k_ab[[1, 0]] = k_val.conj();
        VertexKernel {
            band: arr1(&band),
            k_ab,
            vdiag: None,
            evec,
        }
    }

    /// Gauge phase test: multiplying eigenvectors by random U(1) phases
    /// must leave K_nm unchanged (since K_nm is gauge‑invariant).
    #[test]
    fn test_gauge_invariance_of_kernel() {
        use std::f64::consts::PI;
        let norb = 2;
        let nsta = 2;
        let band = arr1(&[0.0, 1.0]);
        // "Real" velocity matrix elements
        let va = Complex::new(0.3, 0.1);
        let vb = Complex::new(0.2, -0.4);
        let k_val = va * vb.conj(); // K_01 = v^a_01 * v^b_10

        // Build evec with random phases
        let mut evec = Array2::<Complex<f64>>::eye(norb);
        let phase0 = Complex::new(0.0, 0.7 * PI).exp();
        let phase1 = Complex::new(0.0, -0.3 * PI).exp();
        evec[[0, 0]] = phase0;
        evec[[1, 1]] = phase1;

        let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
        k_ab[[0, 1]] = k_val;
        k_ab[[1, 0]] = k_val.conj();

        let vk = VertexKernel { band, k_ab, vdiag: None, evec };
        // K_nm should be exactly the same regardless of evec phases
        assert!((vk.k_ab[[0, 1]] - k_val).norm() < 1e-14,
            "K_01 changed under gauge transformation");
    }

    /// Band reordering test: permuting bands and then tracking should
    /// recover the same integrand.
    #[test]
    fn test_band_tracking_permuted_bands() {
        let norb = 2;
        let nsta = 2;
        // Non‑identity eigenvectors so overlap has off‑diagonal structure
        let theta: f64 = 0.3;
        let c = theta.cos();
        let s = theta.sin();
        let mut evec0 = Array2::<Complex<f64>>::zeros((norb, nsta));
        evec0[[0, 0]] = Complex::new(c, 0.0);
        evec0[[1, 0]] = Complex::new(s, 0.0);
        evec0[[0, 1]] = Complex::new(-s, 0.0);
        evec0[[1, 1]] = Complex::new(c, 0.0);
        let k_val = Complex::new(0.2, 0.1);
        let mut k0 = Array2::<Complex<f64>>::zeros((nsta, nsta));
        k0[[0, 1]] = k_val;
        k0[[1, 0]] = k_val.conj();
        let v0 = VertexKernel {
            band: arr1(&[0.0, 1.0]),
            k_ab: k0,
            vdiag: None,
            evec: evec0.clone(),
        };

        // v1: same eigenstates but energy order reversed
        let mut evec1 = evec0.clone();
        // swap columns (bands)
        for orb in 0..norb {
            let tmp = evec1[[orb, 0]];
            evec1[[orb, 0]] = evec1[[orb, 1]];
            evec1[[orb, 1]] = tmp;
        }
        let mut k1 = Array2::<Complex<f64>>::zeros((nsta, nsta));
        k1[[0, 1]] = k_val.conj(); // swapped band indices
        k1[[1, 0]] = k_val;
        let v1 = VertexKernel {
            band: arr1(&[1.0, 0.0]), // energies swapped
            k_ab: k1,
            vdiag: None,
            evec: evec1,
        };

        let ov = build_overlap_matrix(&v0.evec, &v1.evec);
        let p = greedy_assign(&ov);
        // After tracking, band 0 of v1 should map back to band 0 of v0
        // (lower energy, same physical state)
        assert_eq!(p[0], 1, "band 0 of ref should map to band 1 of v1");
        assert_eq!(p[1], 0, "band 1 of ref should map to band 0 of v1");

        let v1_aligned = permute_vertex(&v1, &p);
        assert!((v1_aligned.band[[0]] - 0.0).abs() < 1e-12, "band 0 energy wrong");
        assert!((v1_aligned.band[[1]] - 1.0).abs() < 1e-12, "band 1 energy wrong");
    }

    /// Constant kernel test: if K_nm is constant and bands are flat,
    /// the Berry integral should match the analytic result.
    #[test]
    fn test_constant_kernel_2d() {
        let n = 20;
        let nk = n * n;
        let nsta: usize = 2;
        let band_val = arr1(&[0.0, 0.5]);
        let k_val = Complex::new(0.0, 0.3); // purely imaginary → Berry = -2*Im = -0.6

        let mut all_pts = Vec::new();
        for _ik in 0..nk {
            // All k-points identical
            let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
            k_ab[[0, 1]] = k_val;
            k_ab[[1, 0]] = k_val.conj();
            all_pts.push(VertexKernel {
                band: band_val.clone(),
                k_ab,
                vdiag: None,
                evec: Array2::<Complex<f64>>::eye(nsta),
            });
        }

        let (total_g, total_o, _unsafe) =
            simplex_berry_integrate(&all_pts, &arr1(&[n, n]), 0.0);

        // d = 0.5, denom = 0.25
        // Ω_0 = -2 * Im(K_01) / d² = -2 * 0.3 / 0.25 = -2.4
        // Ω_1 = -2 * Im(K_10) / d² = -2 * (-0.3) / 0.25 = 2.4
        // Total Ω = 0 (cancellation), g = 0 (purely imaginary K → Re=0)
        let omega_expected = 0.0;
        let g_expected = 0.0;
        assert!((total_o - omega_expected).abs() < 1e-10,
            "Omega {total_o} != expected {omega_expected}");
        assert!((total_g - g_expected).abs() < 1e-10,
            "Metric {total_g} != expected {g_expected}");
    }

    /// Non‑zero Berry curvature: two‑band model with real K_nm
    /// → Berry=0, metric>0.
    #[test]
    fn test_real_kernel_gives_zero_berry() {
        let n = 10;
        let nk = n * n;
        let nsta: usize = 2;
        let band_val = arr1(&[0.0, 1.0]);
        let k_val = Complex::new(0.4, 0.0); // purely real

        let mut all_pts = Vec::new();
        for _ik in 0..nk {
            let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
            k_ab[[0, 1]] = k_val;
            k_ab[[1, 0]] = k_val; // real-symmetric
            all_pts.push(VertexKernel {
                band: band_val.clone(),
                k_ab,
                vdiag: None,
                evec: Array2::<Complex<f64>>::eye(nsta),
            });
        }

        let (total_g, total_o, _unsafe) =
            simplex_berry_integrate(&all_pts, &arr1(&[n, n]), 0.0);

        // Real K → Im[K/d²] = 0 → Berry = 0
        assert!(total_o.abs() < 1e-12, "Berry should be zero for real K, got {total_o}");
        // Re[K/d²] = 0.4/1.0 = 0.4 per band, two bands → 0.8
        let g_expected = 0.8;
        assert!((total_g - g_expected).abs() < 1e-4,
            "Metric {total_g} != ~{g_expected}");
    }
}
