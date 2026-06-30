//! Nonlinear‑response simplex quadrature: Berry curvature dipole.
//!
//! ```text
//! D^{ab;c}(μ,T) = Σ_n ∫_BZ (−∂f/∂E_n) v^c_n(k) Ω^{ab}_n(k) dk
//! ```
//!
//! Requires `T > 0`.  At `T = 0` the δ‑function Fermi‑surface integral
//! is not yet implemented in the simplex path.

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::Result;

use super::kernel::quadrature_dipole_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d};
use super::types::{VertexKernel, SIMPLEX_GAP_TOL};

/// Integrate the Berry‑curvature dipole over the BZ.
///
/// Returns `(dipole_per_mu, unsafe_count)` in fractional‑coordinate
/// volume.  Divide by `det(lat)` for Cartesian.
pub fn integrate_dipole(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    mu: &Array1<f64>,
    T: f64,
    eta: f64,
) -> (Array1<f64>, usize) {
    let dim = k_mesh.len();
    let n_mu = mu.len();
    let beta = 1.0 / (T * 8.617333262e-5);
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
                        if sim.diag.min_gap < SIMPLEX_GAP_TOL { unsafe_count += 1; }
                        acc += &quadrature_dipole_simplex(sim, eta, mu, beta);
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
                            acc += &quadrature_dipole_simplex(sim, eta, mu, beta);
                        }
                    }
                }
            }
        }
        _ => panic!("nonlinear::integrate_dipole: only dim=2,3 supported"),
    }

    (acc, unsafe_count)
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Berry‑curvature dipole via simplex quadrature (T > 0 only).
    ///
    /// ```text
    /// D^{ab;c}(μ,T) = Σ_n ∫_BZ (−∂f/∂E_n) v^c_n(k) Ω^{ab}_n(k) dk
    /// ```
    pub fn berry_curvature_dipole_simplex(
        &self,
        k_mesh: &Array1<usize>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        eta: f64,
    ) -> Result<(Array1<f64>, usize)> {
        assert!(T > 0.0, "dipole simplex requires T>0");
        let kvec = crate::kpoints::gen_kmesh(k_mesh)?;
        let nk = kvec.nrows();
        let gauge = Gauge::Atom;

        let all_pts: Vec<VertexKernel> = (0..nk)
            .into_par_iter()
            .map(|ik| {
                let kv = kvec.row(ik).to_owned();
                self.compute_velocity_kernel(&kv, dir_a, dir_b, Some(dir_c), gauge, None)
            })
            .collect();

        let (dipole, unsafe_count) = integrate_dipole(&all_pts, k_mesh, mu, T, eta);
        let det = self.lat.det().unwrap();
        Ok((dipole / det, unsafe_count))
    }
}
