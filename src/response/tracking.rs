//! Band tracking and simplex construction.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

use super::types::{SimplexDiagnostics, TrackedSimplex, VertexKernel};

const CUBE_TETS: [[usize; 4]; 5] = [
    [0, 1, 2, 4],
    [3, 1, 2, 7],
    [5, 1, 4, 7],
    [6, 2, 4, 7],
    [1, 2, 4, 7],
];
/// Opposite body diagonal [0,3,5,6]; used for diagonal averaging.
const CUBE_TETS_ALT: [[usize; 4]; 5] = [
    [1, 0, 3, 5],
    [2, 0, 3, 6],
    [4, 0, 5, 6],
    [7, 3, 5, 6],
    [0, 3, 5, 6],
];
const TET_VOL_FACTOR: [f64; 5] = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0];

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

pub fn permute_vertex(v: &VertexKernel, p: &[usize]) -> VertexKernel {
    let nsta = v.band.len();
    let norb = v.evec.nrows();

    let perm_mat_opt = |m_opt: &Option<Array2<Complex<f64>>>| -> Option<Array2<Complex<f64>>> {
        m_opt.as_ref().map(|m| {
            let mut out = Array2::<Complex<f64>>::zeros((nsta, nsta));
            for ni in 0..nsta {
                for mi in 0..nsta {
                    out[[ni, mi]] = m[[p[ni], p[mi]]];
                }
            }
            out
        })
    };
    let perm_vec_opt = |v_opt: &Option<Array1<f64>>| -> Option<Array1<f64>> {
        v_opt.as_ref().map(|arr| {
            let mut out = Array1::<f64>::zeros(nsta);
            for ni in 0..nsta {
                out[[ni]] = arr[[p[ni]]];
            }
            out
        })
    };

    let mut band = Array1::<f64>::zeros(nsta);
    let mut evec = Array2::<Complex<f64>>::zeros((norb, nsta));
    for n in 0..nsta {
        band[[n]] = v.band[[p[n]]];
        for orb in 0..norb {
            evec[[orb, n]] = v.evec[[orb, p[n]]];
        }
    }

    let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
    for ni in 0..nsta {
        for mi in 0..nsta {
            k_ab[[ni, mi]] = v.k_ab[[p[ni], p[mi]]];
        }
    }

    VertexKernel {
        band,
        k_ab,
        k_bc: perm_mat_opt(&v.k_bc),
        k_ac: perm_mat_opt(&v.k_ac),
        vdiag: perm_vec_opt(&v.vdiag),
        vdiag_a: perm_vec_opt(&v.vdiag_a),
        vdiag_b: perm_vec_opt(&v.vdiag_b),
        evec,
    }
}

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

pub fn global_band_track(all_pts: &mut [VertexKernel], k_mesh: &[usize]) {
    let nk = all_pts.len();
    if nk <= 1 {
        return;
    }
    let dim = k_mesh.len();
    let mut visited = vec![false; nk];
    let mut queue = std::collections::VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);

    while let Some(current) = queue.pop_front() {
        let neighbours = neighbour_indices(current, k_mesh);
        for &nb in &neighbours {
            if visited[nb] {
                continue;
            }
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
                let ov_avg = ov / n_contrib as f64;
                let p = greedy_assign(&ov_avg);
                all_pts[nb] = permute_vertex(&all_pts[nb], &p);
            }
            visited[nb] = true;
            queue.push_back(nb);
        }
    }
}

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

    let tri_area = inv_nx * inv_ny / 2.0;
    let mut out = Vec::new();

    for &(v0, v1, v2) in &[(i00, i10, i01), (i11, i10, i01)] {
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
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: tri_area,
            coords,
            diag,
        });
    }
    out
}

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

    let quad_area = inv_nx * inv_ny / 4.0;

    let triangles: [[usize; 3]; 4] = [
        [i00, i10, i01],
        [i11, i10, i01], // ↘
        [i00, i10, i11],
        [i00, i11, i01], // ↗
    ];

    let mut out = Vec::with_capacity(4);
    for &[v0, v1, v2] in &triangles {
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
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: quad_area,
            coords,
            diag,
        });
    }
    out
}

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
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: cube_vol * TET_VOL_FACTOR[teti],
            coords,
            diag,
        });
    }
    out
}

/// Helper: build one set of 5 tetrahedra from the given decomposition table,
/// scaling each tet volume by `scale`.
fn build_one_tet_decomp(
    tet_table: &[[usize; 4]; 5],
    c: &[usize; 8],
    corners_frac: &[[f64; 3]; 8],
    cube_vol: f64,
    scale: f64,
    all_pts: &[VertexKernel],
) -> Vec<TrackedSimplex> {
    let mut out = Vec::with_capacity(5);
    for (teti, &[v0, v1, v2, v3]) in tet_table.iter().enumerate() {
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
        out.push(TrackedSimplex {
            vertices: aligned,
            volume: cube_vol * TET_VOL_FACTOR[teti] * scale,
            coords,
            diag,
        });
    }
    out
}

/// Diagonal‑averaged 3D tetrahedralization (10 tets per cell).
pub fn build_tetrahedra_3d_diagavg(
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
    let mut out = build_one_tet_decomp(&CUBE_TETS, &c, &corners_frac, cube_vol, 0.5, all_pts);
    out.extend(build_one_tet_decomp(
        &CUBE_TETS_ALT,
        &c,
        &corners_frac,
        cube_vol,
        0.5,
        all_pts,
    ));
    out
}

// ── Zero‑clone ref builders (for EC hot paths) ───────────────────────────

use super::types::TrackedSimplexRef;

/// 2D reference triangles (2 per cell, no clone, single diagonal).
pub fn build_triangles_2d_ref<'a>(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    inv_nx: f64,
    inv_ny: f64,
    all_pts: &'a [VertexKernel],
) -> [TrackedSimplexRef<'a, 3>; 2] {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;
    let frac =
        |ixv: usize, iyv: usize| -> [f64; 3] { [ixv as f64 * inv_nx, iyv as f64 * inv_ny, 0.0] };
    let tri_vol = inv_nx * inv_ny / 2.0;
    let coord_of = |idx: usize| -> [f64; 3] {
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
    let vertices_of = |v0: usize, v1: usize, v2: usize| -> [&'a VertexKernel; 3] {
        [&all_pts[v0], &all_pts[v1], &all_pts[v2]]
    };
    let min_gap_of = |vs: &[&VertexKernel; 3]| -> f64 {
        vs.iter().fold(f64::INFINITY, |mg, v| {
            let n = v.band.len();
            (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (v.band[[i]] - v.band[[j]]).abs()))
                .fold(mg, f64::min)
        })
    };

    let tri = |v0: usize, v1: usize, v2: usize| -> TrackedSimplexRef<'a, 3> {
        let vertices = vertices_of(v0, v1, v2);
        let mg = min_gap_of(&vertices);
        TrackedSimplexRef {
            vertices,
            coords: [coord_of(v0), coord_of(v1), coord_of(v2)],
            volume: tri_vol,
            diag: SimplexDiagnostics {
                min_gap: mg,
                min_assignment_overlap: 1.0,
                tracking_conflict: false,
            },
        }
    };
    [tri(i00, i10, i01), tri(i11, i10, i01)]
}

/// 2D diagonal‑averaged reference triangles (4 per cell, both diagonals).
pub fn build_triangles_2d_diagavg_ref<'a>(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    inv_nx: f64,
    inv_ny: f64,
    all_pts: &'a [VertexKernel],
) -> [TrackedSimplexRef<'a, 3>; 4] {
    let ixp = (ix + 1) % nx;
    let iyp = (iy + 1) % ny;
    let i00 = ix * ny + iy;
    let i10 = ixp * ny + iy;
    let i11 = ixp * ny + iyp;
    let i01 = ix * ny + iyp;
    let frac =
        |ixv: usize, iyv: usize| -> [f64; 3] { [ixv as f64 * inv_nx, iyv as f64 * inv_ny, 0.0] };
    let half_vol = inv_nx * inv_ny / 4.0;
    let coord_of = |idx: usize| -> [f64; 3] {
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
    let min_gap_of = |vs: &[&VertexKernel]| -> f64 {
        vs.iter().fold(f64::INFINITY, |mg, v| {
            let n = v.band.len();
            (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (v.band[[i]] - v.band[[j]]).abs()))
                .fold(mg, f64::min)
        })
    };
    let tri = |v0: usize, v1: usize, v2: usize| -> TrackedSimplexRef<'a, 3> {
        let vs = [&all_pts[v0], &all_pts[v1], &all_pts[v2]];
        let mg = min_gap_of(&vs);
        TrackedSimplexRef {
            vertices: vs,
            coords: [coord_of(v0), coord_of(v1), coord_of(v2)],
            volume: half_vol,
            diag: SimplexDiagnostics {
                min_gap: mg,
                min_assignment_overlap: 1.0,
                tracking_conflict: false,
            },
        }
    };
    // Both diagonals: ↘ uses (i10,i01), ↗ uses (i00,i11)
    [
        tri(i00, i10, i01),
        tri(i11, i10, i01), // ↘
        tri(i00, i10, i11),
        tri(i00, i11, i01), // ↗
    ]
}

/// 3D reference tetrahedra (5 per cell, no clone, single decomposition).
pub fn build_tetrahedra_3d_ref<'a>(
    ix: usize,
    iy: usize,
    iz: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    inv_nx: f64,
    inv_ny: f64,
    inv_nz: f64,
    all_pts: &'a [VertexKernel],
) -> [TrackedSimplexRef<'a, 4>; 5] {
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
    let corners: [[f64; 3]; 8] = [
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
    let tet_vol_factor = [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0];

    let min_gap_of = |vs: &[&VertexKernel]| -> f64 {
        vs.iter().fold(f64::INFINITY, |mg, v| {
            let n = v.band.len();
            (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (v.band[[i]] - v.band[[j]]).abs()))
                .fold(mg, f64::min)
        })
    };

    std::array::from_fn(|i| {
        let &[lv0, lv1, lv2, lv3] = &CUBE_TETS[i];
        let (g0, g1, g2, g3) = (c[lv0], c[lv1], c[lv2], c[lv3]);
        TrackedSimplexRef {
            vertices: [&all_pts[g0], &all_pts[g1], &all_pts[g2], &all_pts[g3]],
            coords: [corners[lv0], corners[lv1], corners[lv2], corners[lv3]],
            volume: cube_vol * tet_vol_factor[i],
            diag: SimplexDiagnostics {
                min_gap: min_gap_of(&[&all_pts[g0], &all_pts[g1], &all_pts[g2], &all_pts[g3]]),
                min_assignment_overlap: 1.0,
                tracking_conflict: false,
            },
        }
    })
}

/// 3D diagonal‑averaged reference tetrahedra (10 per cell, restores k→−k).
pub fn build_tetrahedra_3d_diagavg_ref<'a>(
    ix: usize,
    iy: usize,
    iz: usize,
    nx: usize,
    ny: usize,
    nz: usize,
    inv_nx: f64,
    inv_ny: f64,
    inv_nz: f64,
    all_pts: &'a [VertexKernel],
) -> [TrackedSimplexRef<'a, 4>; 10] {
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
    let corners: [[f64; 3]; 8] = [
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

    let min_gap_of = |vs: &[&VertexKernel]| -> f64 {
        vs.iter().fold(f64::INFINITY, |mg, v| {
            let n = v.band.len();
            (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (v.band[[i]] - v.band[[j]]).abs()))
                .fold(mg, f64::min)
        })
    };

    let make_tet = |local_v: &[usize; 4], vol: f64| -> TrackedSimplexRef<'a, 4> {
        let &[lv0, lv1, lv2, lv3] = local_v;
        let (g0, g1, g2, g3) = (c[lv0], c[lv1], c[lv2], c[lv3]);
        let vs = [&all_pts[g0], &all_pts[g1], &all_pts[g2], &all_pts[g3]];
        let mg = min_gap_of(&vs);
        TrackedSimplexRef {
            vertices: vs,
            coords: [corners[lv0], corners[lv1], corners[lv2], corners[lv3]],
            volume: vol,
            diag: SimplexDiagnostics {
                min_gap: mg,
                min_assignment_overlap: 1.0,
                tracking_conflict: false,
            },
        }
    };

    let mut out = std::array::from_fn(|_i| make_tet(&CUBE_TETS[0], 0.0));
    for (i, tet) in CUBE_TETS.iter().enumerate() {
        out[i] = make_tet(tet, cube_vol * TET_VOL_FACTOR[i] * 0.5);
    }
    for (i, tet) in CUBE_TETS_ALT.iter().enumerate() {
        out[5 + i] = make_tet(tet, cube_vol * TET_VOL_FACTOR[i] * 0.5);
    }
    out
}
