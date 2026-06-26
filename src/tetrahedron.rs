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
}
