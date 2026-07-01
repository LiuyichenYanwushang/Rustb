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

    VertexKernel {
        band,
        k_ab: perm_mat_opt(&Some(v.k_ab.clone())).unwrap(),
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
