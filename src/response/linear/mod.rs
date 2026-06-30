//! Linear‑response simplex quadrature: Berry curvature + quantum metric.
//!
//! ```text
//! Ω^{ab} = Σ_n ∫_BZ Ω^{ab}_n(k) dk
//! g^{ab} = Σ_n ∫_BZ  g^{ab}_n(k) dk
//! ```
//!
//! where `Ω_n = −2 Im G_n`, `g_n = Re G_n`, and
//! `G_n = Σ_{m≠n} v^a_nm v^b_mn / ((E_n−E_m)² + η²)`.

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::Result;

use super::kernel::quadrature_berry_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d};
use super::types::{VertexKernel, SIMPLEX_GAP_TOL};

/// Integrate Berry curvature and quantum metric over the BZ.
///
/// ```text
/// total_berry  = Σ_n ∫_BZ Ω^{ab}_n(k) dk
/// total_metric = Σ_n ∫_BZ  g^{ab}_n(k) dk
/// ```
///
/// where `Ω_n = −2 Im G_n`, `g_n = Re G_n`, and
/// `G_n = Σ_{m≠n} K^{ab}_nm / ((E_n−E_m)² + η²)`.
///
/// Inside each simplex, `K_nm` and `E_n` are linearly interpolated,
/// then the kernel is evaluated at degree‑2 symmetric quadrature points.
///
/// # Returns
/// `(total_metric, total_berry, unsafe_count)` in fractional‑coordinate
/// volume.  Divide by `det(lat)` for Cartesian volume.
pub fn integrate(
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
                    let sims = build_triangles_2d(ix, iy, nx, ny, inv_nx, inv_ny, all_pts);
                    for sim in &sims {
                        if sim.diag.min_gap < SIMPLEX_GAP_TOL { unsafe_count += 1; }
                        let (g, o) = quadrature_berry_simplex(sim, eta);
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
                            ix, iy, iz, nx, ny, nz, inv_nx, inv_ny, inv_nz, all_pts,
                        );
                        for sim in &sims {
                            if sim.diag.min_gap < SIMPLEX_GAP_TOL { unsafe_count += 1; }
                            let (g, o) = quadrature_berry_simplex(sim, eta);
                            total_g += g;
                            total_o += o;
                        }
                    }
                }
            }
        }
        _ => panic!("linear::integrate: only dim=2,3 supported, got dim={dim}"),
    }

    (total_g, total_o, unsafe_count)
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Berry curvature + quantum metric via simplex quadrature.
    ///
    /// Returns `(total_metric, total_berry, unsafe_count)` in Cartesian
    /// reciprocal‑space volume.
    pub fn berry_curvature_simplex(
        &self,
        k_mesh: &Array1<usize>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        eta: f64,
    ) -> Result<(f64, f64, usize)> {
        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let gauge = Gauge::Atom;

        let all_pts: Vec<VertexKernel> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_velocity_kernel(&kv, dir_a, dir_b, None, gauge, None)
            })
            .collect();

        let (total_g, total_o, unsafe_count) = integrate(&all_pts, k_mesh, eta);
        let det = self.lat.det().unwrap();
        Ok((total_g / det, total_o / det, unsafe_count))
    }
}
