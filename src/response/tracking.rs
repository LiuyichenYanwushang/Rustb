//! Band tracking and simplex construction.

use ndarray::prelude::*;
use num_complex::Complex;

use crate::error::Result;

use super::types::{NonlinearKernel, SimplexDiagnostics, TrackedSimplex, VertexKernel};

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
    let nsta = evec_ref.nrows();
    let norb = evec_ref.ncols();
    let mut ov = Array2::<f64>::zeros((nsta, nsta));
    for n in 0..nsta {
        for m in 0..nsta {
            let mut s = Complex::new(0.0, 0.0);
            for orb in 0..norb {
                s += evec_ref[[n, orb]].conj() * evec_other[[m, orb]];
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
    let matrix = |m: &Array2<Complex<f64>>| m.select(Axis(0), p).select(Axis(1), p);
    VertexKernel {
        band: v.band.select(Axis(0), p),
        k_ab: matrix(&v.k_ab),
        nonlinear: v.nonlinear.as_ref().map(|n| NonlinearKernel {
            k_bc: matrix(&n.k_bc),
            k_ac: matrix(&n.k_ac),
            vdiag: n.vdiag.select(Axis(0), p),
            vdiag_a: n.vdiag_a.select(Axis(0), p),
            vdiag_b: n.vdiag_b.select(Axis(0), p),
        }),
    }
}

/// Collect in input order, then track vertices without retaining eigenvectors.
pub(crate) fn collect_tracked_vertices(
    rows: Vec<Result<(VertexKernel, Array2<Complex<f64>>)>>,
    k_mesh: &[usize],
) -> Result<Vec<VertexKernel>> {
    let prepared = rows.into_iter().collect::<Result<Vec<_>>>()?;
    let (mut vertices, mut eigenvectors): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    global_band_track(&mut eigenvectors, k_mesh, |index, permutation| {
        vertices[index] = permute_vertex(&vertices[index], permutation);
    });
    drop(eigenvectors);
    Ok(vertices)
}

/// Track row-ket eigenvectors and apply each band permutation to caller data.
/// Each callback concerns a previously unvisited vertex, so `p` indexes its
/// original band order. The seed vertex keeps its original order.
pub(crate) fn global_band_track(
    eigenvectors: &mut [Array2<Complex<f64>>],
    k_mesh: &[usize],
    mut permute_data: impl FnMut(usize, &[usize]),
) {
    let nk = eigenvectors.len();
    if nk <= 1 {
        return;
    }
    #[cfg(test)]
    super::config::counters::count_band_tracking();
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
                let ov = build_overlap_matrix(&eigenvectors[vn], &eigenvectors[nb]);
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
                eigenvectors[nb] = eigenvectors[nb].select(Axis(0), &p);
                permute_data(nb, &p);
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
    if dim == 1 {
        let nx = k_mesh[0];
        let ix = i as isize;
        for dx in [-1isize, 1] {
            let jx = ix + dx;
            if jx >= 0 && jx < nx as isize {
                out.push(jx as usize);
            }
        }
    } else if dim == 2 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::TbError;
    use crate::response::config::counters;
    use ndarray::array;
    use rayon::prelude::*;

    fn vertex(base: f64) -> VertexKernel {
        let diagonal = |offset: f64| Array1::from_shape_fn(3, |n| base + offset + n as f64);
        let matrix = |offset: f64| {
            Array2::from_shape_fn((3, 3), |(n, m)| {
                let value = base + offset + (10 * n + m) as f64;
                Complex::new(value, -value - 0.25)
            })
        };
        VertexKernel {
            band: array![base + 7.0, base - 2.0, base + 3.0],
            k_ab: matrix(100.0),
            nonlinear: Some(NonlinearKernel {
                k_bc: matrix(200.0),
                k_ac: matrix(300.0),
                vdiag: diagonal(10.0),
                vdiag_a: diagonal(20.0),
                vdiag_b: diagonal(30.0),
            }),
        }
    }

    fn assert_vertex_permutation(
        actual: &VertexKernel,
        original: &VertexKernel,
        permutation: &[usize; 3],
    ) {
        // Independent index lookups, not the production permutation helper.
        let diagonal = |values: &Array1<f64>| Array1::from_shape_fn(3, |n| values[permutation[n]]);
        let matrix = |values: &Array2<Complex<f64>>| {
            Array2::from_shape_fn((3, 3), |(n, m)| values[[permutation[n], permutation[m]]])
        };
        assert_eq!(actual.band, diagonal(&original.band));
        assert_eq!(actual.k_ab, matrix(&original.k_ab));
        match (&actual.nonlinear, &original.nonlinear) {
            (Some(actual), Some(original)) => {
                assert_eq!(actual.k_bc, matrix(&original.k_bc));
                assert_eq!(actual.k_ac, matrix(&original.k_ac));
                assert_eq!(actual.vdiag, diagonal(&original.vdiag));
                assert_eq!(actual.vdiag_a, diagonal(&original.vdiag_a));
                assert_eq!(actual.vdiag_b, diagonal(&original.vdiag_b));
            }
            (None, None) => {}
            _ => panic!("nonlinear payload presence changed"),
        }
    }

    #[test]
    fn collection_tracks_all_payloads_with_three_band_cycles() {
        let mut original = [vertex(0.0), vertex(1000.0), vertex(2000.0)];
        original[2].nonlinear = None;
        // The noncanonical seed must stay unchanged; the last row also checks
        // that tracking propagates through the already-permuted middle row.
        let row_orders = [[1, 2, 0], [2, 0, 1], [0, 1, 2]];
        let rows = original
            .iter()
            .zip(row_orders)
            .map(|(vertex, order)| Ok((vertex.clone(), Array2::eye(3).select(Axis(0), &order))))
            .collect();
        let (eigen, tracking, result) = counters::measure(|| collect_tracked_vertices(rows, &[3]));
        assert_eq!((eigen, tracking), (0, 1));
        let tracked = result.unwrap();
        assert_eq!(tracked.len(), original.len());
        for (index, permutation) in [[0, 1, 2], [2, 0, 1], [1, 2, 0]].iter().enumerate() {
            assert_vertex_permutation(&tracked[index], &original[index], permutation);
        }
    }

    #[test]
    fn collection_selects_first_error_after_indexed_parallel_preparation() {
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for first_error in [0usize, 2, 4] {
                let result = pool.install(|| {
                    let rows: Vec<_> = (0..6usize)
                        .into_par_iter()
                        .map(|index| {
                            if index == first_error || index == 5 {
                                Err(TbError::Lapack {
                                    routine: "tracking_fixture",
                                    info: index as i32 + 1,
                                })
                            } else {
                                Ok((vertex(index as f64 * 1000.0), Array2::eye(3)))
                            }
                        })
                        .collect();
                    collect_tracked_vertices(rows, &[6])
                });
                assert!(
                    matches!(
                        result,
                        Err(TbError::Lapack { routine: "tracking_fixture", info })
                            if info == first_error as i32 + 1
                    ),
                    "expected row {first_error} to win in the {threads}-thread pool"
                );
            }
        }
    }

    #[test]
    fn collection_does_not_track_a_successful_prefix_before_error() {
        // Counters are thread-local: measure separately from the parallel test.
        let (eigen, tracking, result) = counters::measure(|| {
            let rows = vec![
                Ok((vertex(0.0), Array2::eye(3))),
                Ok((vertex(1000.0), Array2::eye(3))),
                Err(TbError::Lapack {
                    routine: "first_failure",
                    info: 7,
                }),
                Err(TbError::SvdComputationFailed),
            ];
            collect_tracked_vertices(rows, &[4])
        });
        assert!(matches!(
            result,
            Err(TbError::Lapack {
                routine: "first_failure",
                info: 7,
            })
        ));
        assert_eq!((eigen, tracking), (0, 0));
    }

    #[test]
    fn collection_preserves_singleton_and_accepts_empty() {
        let seed = vertex(0.0);
        let (eigen, tracking, (empty, singleton)) = counters::measure(|| {
            (
                collect_tracked_vertices(Vec::new(), &[0]).unwrap(),
                collect_tracked_vertices(
                    vec![Ok((
                        seed.clone(),
                        Array2::eye(3).select(Axis(0), &[1, 2, 0]),
                    ))],
                    &[1],
                )
                .unwrap(),
            )
        });
        assert_eq!((eigen, tracking), (0, 0));
        assert!(empty.is_empty());
        assert_eq!(singleton.len(), 1);
        assert_vertex_permutation(&singleton[0], &seed, &[0, 1, 2]);
    }
}
