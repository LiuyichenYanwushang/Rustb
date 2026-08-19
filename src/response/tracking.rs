//! Band tracking and simplex construction.

use ndarray::prelude::*;
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

pub(crate) fn build_overlap_matrix(
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

pub(crate) fn greedy_assign(overlap: &Array2<f64>) -> Vec<usize> {
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

pub(crate) fn permute_vertex(v: &VertexKernel, p: &[usize]) -> VertexKernel {
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

pub(crate) fn global_band_track(all_pts: &mut [VertexKernel], k_mesh: &[usize]) {
    let nk = all_pts.len();
    if nk <= 1 {
        return;
    }
    let _dim = k_mesh.len();
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
    // Intentionally non-periodic: `global_band_track` builds one connected
    // spanning tree through the interior of the mesh. The simplex builders
    // later wrap around cell boundaries, but direct wrap-around eigenvector
    // overlap in the BFS was found to reduce tracking robustness in the
    // energy-cut Hall tests.
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

// ── Simplex builders (zero-clone, borrow from `all_pts`) ────────────────

pub(crate) fn build_triangles_2d<'a>(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    inv_nx: f64,
    inv_ny: f64,
    all_pts: &'a [VertexKernel],
) -> [TrackedSimplex<'a, 3>; 2] {
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

    let tri = |v0: usize, v1: usize, v2: usize| -> TrackedSimplex<'a, 3> {
        let vertices = vertices_of(v0, v1, v2);
        let mg = min_gap_of(&vertices);
        TrackedSimplex {
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

pub(crate) fn build_triangles_2d_diagavg<'a>(
    ix: usize,
    iy: usize,
    nx: usize,
    ny: usize,
    inv_nx: f64,
    inv_ny: f64,
    all_pts: &'a [VertexKernel],
) -> [TrackedSimplex<'a, 3>; 4] {
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
    let tri = |v0: usize, v1: usize, v2: usize| -> TrackedSimplex<'a, 3> {
        let vs = [&all_pts[v0], &all_pts[v1], &all_pts[v2]];
        let mg = min_gap_of(&vs);
        TrackedSimplex {
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

pub(crate) fn build_tetrahedra_3d<'a>(
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
) -> [TrackedSimplex<'a, 4>; 5] {
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
        TrackedSimplex {
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

pub(crate) fn build_tetrahedra_3d_diagavg<'a>(
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
) -> [TrackedSimplex<'a, 4>; 10] {
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

    let make_tet = |local_v: &[usize; 4], vol: f64| -> TrackedSimplex<'a, 4> {
        let &[lv0, lv1, lv2, lv3] = local_v;
        let (g0, g1, g2, g3) = (c[lv0], c[lv1], c[lv2], c[lv3]);
        let vs = [&all_pts[g0], &all_pts[g1], &all_pts[g2], &all_pts[g3]];
        let mg = min_gap_of(&vs);
        TrackedSimplex {
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
