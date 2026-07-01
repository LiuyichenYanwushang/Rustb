//! Band tracking and simplex construction.
//!
//! Uses eigenvector overlap maximisation to align band labels across
//! simplex vertices, then builds 2D triangles / 3D tetrahedra with
//! periodic boundary conditions.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

use super::types::{SimplexDiagnostics, TrackedSimplex, VdiagRect, VertexKernel};

// ── Cube / tetrahedron decomposition constants ──────────────────────────

const CUBE_TETS: [[usize; 4]; 5] = [
    [0, 1, 2, 4],
    [3, 1, 2, 7],
    [5, 1, 4, 7],
    [6, 2, 4, 7],
    [1, 2, 4, 7],
];
const TET_VOL_FACTOR: [f64; 5] = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0];

// ── Overlap and assignment ──────────────────────────────────────────────

/// Build the overlap matrix `O_nm = |⟨u_n(ref)|u_m(other)⟩|²`.
pub fn build_overlap_matrix(
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

/// Greedy band assignment (row‑wise maximum, not guaranteed globally optimal).
///
/// For small `nsta` with diagonally‑dominant overlap this is exact; for
/// larger or near‑degenerate problems, replace with Hungarian / exhaustive
/// search over small `n`.
pub fn greedy_assign(overlap: &Array2<f64>) -> Vec<usize> {
    let n = overlap.nrows();
    let mut assigned = vec![false; n];
    let mut perm = vec![0usize; n];
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

/// Reorder a `VertexKernel` according to band permutation `p`.
pub fn permute_vertex(v: &VertexKernel, p: &[usize]) -> VertexKernel {
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
    VertexKernel {
        band,
        k_ab,
        vdiag,
        evec,
    }
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

/// Track band labels across simplex vertices (first vertex = reference).
pub fn track_simplex_vertices(
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
        let ov_score: f64 = (0..ov.nrows()).map(|n| ov[[n, p[n]]]).sum::<f64>() / ov.nrows() as f64;
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

// ── Global band tracking ────────────────────────────────────────────────

/// Globally track band labels across the entire k‑mesh.
///
/// Starts from the Γ point (first k‑point, assumed to be `k = 0`) and
/// propagates outward using BFS.  At each step the new point is matched
/// against **all already‑tracked neighbours** to maximise robustness.
///
/// The `all_pts` slice is mutated in place — each `VertexKernel` gets
/// its band indices permuted for global consistency.  After this call,
/// local simplex tracking is no longer needed.
pub fn global_band_track(all_pts: &mut [VertexKernel], k_mesh: &[usize]) {
    let nk = all_pts.len();
    if nk <= 1 {
        return;
    }
    let dim = k_mesh.len();

    // ── BFS queue; visited[i] = true once the point is tracked ──────
    let mut visited = vec![false; nk];
    let mut queue = std::collections::VecDeque::new();

    // Γ point (index 0) is the reference.
    visited[0] = true;
    queue.push_back(0usize);

    while let Some(current) = queue.pop_front() {
        // Find all unvisited neighbours of `current`
        let neighbours = neighbour_indices(current, k_mesh);
        for &nb in &neighbours {
            if visited[nb] {
                continue;
            }
            // Collect overlap with all *visited* neighbours
            let mut ov_sum: Option<Array2<f64>> = None;
            let mut n_contrib = 0usize;
            for &vn in &neighbour_indices(nb, k_mesh) {
                if !visited[vn] {
                    continue;
                }
                let ov = build_overlap_matrix(&all_pts[vn].evec, &all_pts[nb].evec);
                if let Some(ref mut s) = ov_sum {
                    *s += &ov;
                } else {
                    ov_sum = Some(ov);
                }
                n_contrib += 1;
            }
            if let Some(ov) = ov_sum {
                // Average overlap over all visited neighbours
                let ov_avg = ov / n_contrib as f64;
                let p = greedy_assign(&ov_avg);
                let permuted = permute_vertex(&all_pts[nb], &p);
                all_pts[nb] = permuted;
            }
            visited[nb] = true;
            queue.push_back(nb);
        }
    }
}

/// Return the indices of all nearest‑neighbour k‑points for index `i`.
fn neighbour_indices(i: usize, k_mesh: &[usize]) -> Vec<usize> {
    let dim = k_mesh.len();
    let mut out = Vec::new();
    if dim == 2 {
        let (nx, ny) = (k_mesh[0], k_mesh[1]);
        let ix = i / ny;
        let iy = i % ny;
        for &(dx, dy) in &[(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
            let jx = ix as isize + dx;
            let jy = iy as isize + dy;
            if jx >= 0 && jx < nx as isize && jy >= 0 && jy < ny as isize {
                out.push((jx as usize) * ny + (jy as usize));
            }
        }
    } else {
        let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
        let ix = i / (ny * nz);
        let rem = i % (ny * nz);
        let iy = rem / nz;
        let iz = rem % nz;
        for &(dx, dy, dz) in &[
            (1isize, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let jx = ix as isize + dx;
            let jy = iy as isize + dy;
            let jz = iz as isize + dz;
            if jx >= 0
                && jx < nx as isize
                && jy >= 0
                && jy < ny as isize
                && jz >= 0
                && jz < nz as isize
            {
                out.push((jx as usize) * ny * nz + (jy as usize) * nz + (jz as usize));
            }
        }
    }
    out
}

// ── Simplex builders ────────────────────────────────────────────────────

/// Build the two tracked triangles of a 2D cell.
/// Caller should have run `global_band_track` first for consistent labels.
pub fn build_triangles_2d(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    inv_nx: f64,
    inv_ny: f64,
    all_pts: &[VertexKernel],
) -> Vec<TrackedSimplex> {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;

    let frac = |ixv: usize, iyv: usize| -> [f64; 2] { [ixv as f64 * inv_nx, iyv as f64 * inv_ny] };
    let coord_of = |idx: usize| -> [f64; 2] {
        if idx == i00 {
            frac(ix, iy)
        } else if idx == i10 {
            frac(ix + 1, iy)
        } else if idx == i11 {
            frac(ix + 1, iy + 1)
        } else {
            frac(ix, iy + 1)
        }
    };

    let cell_area = inv_nx * inv_ny;
    let tri_area = cell_area / 2.0;
    let mut out = Vec::new();

    // After global_band_track, all points share consistent band labels.
    // Local tracking is skipped; diagnostics use the vertex data directly.
    //
    // Build VdiagRect from all four rectangle corners (when vdiag is present).
    let vdiag_rect: Option<VdiagRect> = match (
        all_pts[i00].vdiag.as_ref(),
        all_pts[i10].vdiag.as_ref(),
        all_pts[i01].vdiag.as_ref(),
        all_pts[i11].vdiag.as_ref(),
    ) {
        (Some(v00), Some(v10), Some(v01), Some(v11)) => Some(VdiagRect {
            corners: [v00.to_vec(), v10.to_vec(), v01.to_vec(), v11.to_vec()],
        }),
        _ => None,
    };

    for &(v0, v1, v2, local_xy) in &[
        // Triangle 1: ↘ diagonal, vertices at (0,0), (1,0), (0,1)
        (i00, i10, i01, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        // Triangle 2: ↘ diagonal, vertices at (1,1), (1,0), (0,1)
        (i11, i10, i01, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
    ] {
        let aligned = vec![
            all_pts[v0].clone(),
            all_pts[v1].clone(),
            all_pts[v2].clone(),
        ];
        let mut min_gap = f64::INFINITY;
        for v in &aligned {
            let n = v.band.len();
            for i in 0..n {
                for j in i + 1..n {
                    min_gap = min_gap.min((v.band[[i]] - v.band[[j]]).abs());
                }
            }
        }
        let diag = SimplexDiagnostics {
            min_gap,
            min_assignment_overlap: 1.0,
            tracking_conflict: false,
        };
        let coords = Array2::from_shape_vec((3, 2), {
            let c0 = coord_of(v0);
            let c1 = coord_of(v1);
            let c2 = coord_of(v2);
            vec![c0[0], c0[1], c1[0], c1[1], c2[0], c2[1]]
        })
        .unwrap();
        let vertex_xy = Array2::from_shape_vec((3, 2), {
            let r0 = local_xy[0];
            let r1 = local_xy[1];
            let r2 = local_xy[2];
            vec![r0[0], r0[1], r1[0], r1[1], r2[0], r2[1]]
        })
        .unwrap();
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: tri_area,
            coords,
            diag,
            vdiag_rect: vdiag_rect.clone(),
            vertex_xy,
        });
    }
    out
}

/// Build four triangles of a 2D cell averaging both diagonal partitions.
///
/// $$\int_{\square} f \approx \frac{1}{2}\bigl(I_{\diagdown} + I_{\diagup}\bigr)$$
///
/// Returns 4 triangles, each with `volume = cell_area / 4` so that the
/// sum over all four equals the diagonal‑averaged cell integral.
///
/// | Diagonal | Vertices     | vertex_xy                |
/// |----------|-------------|--------------------------|
/// | ↘        | (i00,i10,i01) | (0,0),(1,0),(0,1)       |
/// | ↘        | (i11,i10,i01) | (1,1),(1,0),(0,1)       |
/// | ↗        | (i00,i10,i11) | (0,0),(1,0),(1,1)       |
/// | ↗        | (i00,i11,i01) | (0,0),(1,1),(0,1)       |
///
/// Caller must have run `global_band_track` first for consistent labels.
pub fn build_triangles_2d_diagavg(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    inv_nx: f64,
    inv_ny: f64,
    all_pts: &[VertexKernel],
) -> Vec<TrackedSimplex> {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;

    let frac = |ixv: usize, iyv: usize| -> [f64; 2] { [ixv as f64 * inv_nx, iyv as f64 * inv_ny] };
    let coord_of = |idx: usize| -> [f64; 2] {
        if idx == i00 {
            frac(ix, iy)
        } else if idx == i10 {
            frac(ix + 1, iy)
        } else if idx == i11 {
            frac(ix + 1, iy + 1)
        } else {
            frac(ix, iy + 1)
        }
    };

    let cell_area = inv_nx * inv_ny;
    let quad_area = cell_area / 4.0; // each of 4 triangles → 1/4 cell area

    // VdiagRect from all four rectangle corners.
    let vdiag_rect: Option<VdiagRect> = match (
        all_pts[i00].vdiag.as_ref(),
        all_pts[i10].vdiag.as_ref(),
        all_pts[i01].vdiag.as_ref(),
        all_pts[i11].vdiag.as_ref(),
    ) {
        (Some(v00), Some(v10), Some(v01), Some(v11)) => Some(VdiagRect {
            corners: [v00.to_vec(), v10.to_vec(), v01.to_vec(), v11.to_vec()],
        }),
        _ => None,
    };

    // Four triangles: (vertex_indices, vertex_xy)
    let triangles: [([usize; 3], [[f64; 2]; 3]); 4] = [
        ([i00, i10, i01], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ([i11, i10, i01], [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        ([i00, i10, i11], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        ([i00, i11, i01], [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
    ];

    let mut out = Vec::with_capacity(4);
    for &(tri_indices, local_xy) in &triangles {
        let (v0, v1, v2) = (tri_indices[0], tri_indices[1], tri_indices[2]);
        let aligned = vec![
            all_pts[v0].clone(),
            all_pts[v1].clone(),
            all_pts[v2].clone(),
        ];
        let mut min_gap = f64::INFINITY;
        for v in &aligned {
            let n = v.band.len();
            for i in 0..n {
                for j in i + 1..n {
                    min_gap = min_gap.min((v.band[[i]] - v.band[[j]]).abs());
                }
            }
        }
        let diag = SimplexDiagnostics {
            min_gap,
            min_assignment_overlap: 1.0,
            tracking_conflict: false,
        };
        let coords = Array2::from_shape_vec((3, 2), {
            let c0 = coord_of(v0);
            let c1 = coord_of(v1);
            let c2 = coord_of(v2);
            vec![c0[0], c0[1], c1[0], c1[1], c2[0], c2[1]]
        })
        .unwrap();
        let vertex_xy = Array2::from_shape_vec((3, 2), {
            let r0 = local_xy[0];
            let r1 = local_xy[1];
            let r2 = local_xy[2];
            vec![r0[0], r0[1], r1[0], r1[1], r2[0], r2[1]]
        })
        .unwrap();
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: quad_area,
            coords,
            diag,
            vdiag_rect: vdiag_rect.clone(),
            vertex_xy,
        });
    }
    out
}

/// Build the five tracked tetrahedra of a 3D cell.
pub fn build_tetrahedra_3d(
    ix: usize,
    iy: usize,
    iz: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    inv_nx: f64,
    inv_ny: f64,
    inv_nz: f64,
    all_pts: &[VertexKernel],
) -> Vec<TrackedSimplex> {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let izp = (iz + 1) % nz;
    let idx3 = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;
    let c = [
        idx3(ix, iy, iz),
        idx3(ixp, iy, iz),
        idx3(ix, iyp, iz),
        idx3(ixp, iyp, iz),
        idx3(ix, iy, izp),
        idx3(ixp, iy, izp),
        idx3(ix, iyp, izp),
        idx3(ixp, iyp, izp),
    ];

    let frac = |ixv: usize, iyv: usize, izv: usize| -> [f64; 3] {
        [
            ixv as f64 * inv_nx,
            iyv as f64 * inv_ny,
            izv as f64 * inv_nz,
        ]
    };
    // Use unwrapped ix+1, iy+1, iz+1 for fractional coords (match 2D convention).
    let corners_frac: [[f64; 3]; 8] = [
        frac(ix, iy, iz),
        frac(ix + 1, iy, iz),
        frac(ix, iy + 1, iz),
        frac(ix + 1, iy + 1, iz),
        frac(ix, iy, iz + 1),
        frac(ix + 1, iy, iz + 1),
        frac(ix, iy + 1, iz + 1),
        frac(ix + 1, iy + 1, iz + 1),
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
            let c0 = corners_frac[v0];
            let c1 = corners_frac[v1];
            let c2 = corners_frac[v2];
            let c3 = corners_frac[v3];
            vec![
                c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2],
            ]
        })
        .unwrap();
        let vertex_xy = Array2::<f64>::zeros((4, 2)); // unused in 3D
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: cube_vol * TET_VOL_FACTOR[teti],
            coords,
            diag,
            vdiag_rect: None,
            vertex_xy,
        });
    }
    out
}
