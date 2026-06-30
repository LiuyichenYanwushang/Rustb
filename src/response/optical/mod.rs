//! Optical conductivity via simplex quadrature.
//!
//! ```text
//! σ^{ab}(ω,μ,T) = Σ_{n≠m} ∫_BZ (f_n−f_m)
//!     · v^a_{nm} v^b_{mn} / ((E_n−E_m)² − (ω+iη)²) dk
//! ```

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::Result;

use super::kernel::quadrature_optical_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d};
use super::types::VertexKernel;

/// Integrate the optical conductivity kernel over the BZ.
///
/// Returns the complex conductivity `σ^{ab}` in fractional‑coordinate
/// volume.  Divide by `det(lat)` for Cartesian.
pub fn integrate(
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
                        total += quadrature_optical_simplex(sim, omega, eta, mu, beta);
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
                            total += quadrature_optical_simplex(sim, omega, eta, mu, beta);
                        }
                    }
                }
            }
        }
        _ => panic!("optical::integrate: only dim=2,3 supported"),
    }

    total
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Optical conductivity via simplex quadrature.
    pub fn optical_conductivity_simplex(
        &self,
        k_mesh: &Array1<usize>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        omega: f64,
        eta: f64,
        mu: f64,
        T: f64,
    ) -> Result<Complex<f64>> {
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

        let sigma = integrate(&all_pts, k_mesh, omega, eta, mu, T);
        let det = self.lat.det().unwrap();
        Ok(sigma / det)
    }
}
