//! # Linear response: Berry curvature and quantum metric
//!
//! ## Theory
//!
//! The quantum geometric tensor is
//!
//! $$G^{ab}_n(\mathbf{k}) = \sum_{m\neq n}
//!   \frac{\langle\partial_a u_n|u_m\rangle\langle u_m|\partial_b u_n\rangle}
//!        {(E_n-E_m)^2} = g^{ab}_n - \frac{i}{2}\Omega^{ab}_n$$
//!
//! In terms of velocity matrix elements $v^\alpha_{nm} = \langle u_n|\partial_\alpha H|u_m\rangle$:
//!
//! $$G^{ab}_n(\mathbf{k}) = \sum_{m\neq n} \frac{v^a_{nm} v^b_{mn}}{(E_n-E_m)^2 + \eta^2}$$
//!
//! where $\eta$ is a small regularisation width.  The real and imaginary
//! parts give the quantum metric $g^{ab}_n$ and Berry curvature $\Omega^{ab}_n$:
//!
//! $$g^{ab}_n = \operatorname{Re} G^{ab}_n, \qquad
//!   \Omega^{ab}_n = -2\operatorname{Im} G^{ab}_n$$
//!
//! The **anomalous Hall conductivity** (AHC) at $T=0$ is
//!
//! $$\sigma^{xy}_{\text{AHC}}(\mu) = -\frac{e^2}{\hbar}\sum_n
//!   \int_{\text{BZ}} \Theta(\mu-E_n)\,\Omega^{xy}_n(\mathbf{k})\,d\mathbf{k}$$
//!
//! ## API
//!
//! | Method | Path | Formula |
//! |--------|------|---------|
//! | `berry_curvature_simplex` | simplex | $\sum_n \int \Omega^{ab}_n\,d\mathbf{k}$ (Cartesian) |
//! | `Hall_conductivity` | direct sum | $\sigma_{\text{AHC}}(\mu,T)$ |
//! | `Hall_conductivity_mu` | direct sum | $\sigma_{\text{AHC}}(\mu)$ per μ |

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::SpinDirection;
use crate::error::Result;

use super::kernel::quadrature_berry_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d};
use super::traits::BerryCurvature;
use super::types::{SIMPLEX_GAP_TOL, VertexKernel};

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
pub fn integrate(all_pts: &[VertexKernel], k_mesh: &Array1<usize>, eta: f64) -> (f64, f64, usize) {
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
                        if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                            unsafe_count += 1;
                        }
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
                            if sim.diag.min_gap < SIMPLEX_GAP_TOL {
                                unsafe_count += 1;
                            }
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

    /// Computes the anomalous Hall conductivity at a given chemical potential and temperature.
    ///
    /// Uses a uniform k-mesh and direct summation:
    /// $$ \sigma_{\alpha\beta}^\gamma = \f{1}{N (2\pi)^d V} \sum_{\mathbf k} \Omega_{\alpha\beta}^\gamma(\mathbf k), $$
    /// where $N$ is the number of k-points, $d$ is the dimension, and $V$ is the unit cell volume.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction, e.g. `arr1(&[nk, nk])` for 2D.
    /// * `current_dir` - Direction vector for the first index $\alpha$ of $\sigma_{\alpha\beta}$.
    /// * `dir_2` - Direction vector for the second index $\beta$.
    /// * `mu` - Chemical potential $\mu$ (in eV).
    /// * `T` - Temperature (in K). Use `T=0` for the zero-temperature step function.
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter $\eta$ (in eV).
    ///
    /// # Returns
    ///
    /// The Hall conductivity $\sigma_{\alpha\beta}$ in units of $e^2/\hbar/\AA$ (3D) or $e^2/\hbar$ (2D).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ndarray::arr1;
    /// # use rustb::Model;
    /// # fn example(model: &Model) -> Result<(), rustb::error::TbError> {
    /// let kmesh = arr1(&[31, 31]);
    /// let current_dir = arr1(&[1.0, 0.0]);
    /// let dir_2 = arr1(&[0.0, 1.0]);
    /// let sigma_xy = model.Hall_conductivity(&kmesh, &current_dir, &dir_2, 0.0, 0.0, None, 1e-3)?;
    /// println!("Hall conductivity = {}", sigma_xy);
    /// # Ok(())
    /// # }
    /// ```
    #[allow(non_snake_case)]
    pub fn Hall_conductivity(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: f64,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<f64> {
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let omega = self.berry_curvature(&kvec, &current_dir, &dir_2, mu, T, spin, eta);
        let conductivity: f64 = omega.sum() / (nk as f64) / self.lat.det().unwrap();
        Ok(conductivity)
    }

    /// Computes the Hall conductivity for multiple chemical potential values efficiently.
    ///
    /// This method first computes $\Omega_n$ (the Berry curvature per band) at each k-point,
    /// then evaluates the Fermi-Dirac-weighted sum for each $\mu$. This avoids repeatedly
    /// computing $\Omega_n$, making it much faster than calling [`Hall_conductivity`] for each $\mu$.
    /// However, it uses more memory and cannot use adaptive integration.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` - Number of k-points along each direction.
    /// * `current_dir`, `dir_2` - Direction vectors for the conductivity tensor indices.
    /// * `mu` - Array of chemical potential values (in eV).
    /// * `T` - Temperature (in K).
    /// * `spin` - Spin operator index (0, 1, 2, 3).
    /// * `eta` - Broadening parameter (in eV).
    ///
    /// # Returns
    ///
    /// An `Array1<f64>` of Hall conductivity values, one for each $\mu$, in units of $e^2/\hbar/\AA$ (3D) or $e^2/\hbar$ (2D).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use ndarray::Array1;
    /// # use rustb::Model;
    /// # fn example(model: &Model) -> Result<(), rustb::error::TbError> {
    /// let kmesh = ndarray::arr1(&[31, 31]);
    /// let current_dir = ndarray::arr1(&[1.0, 0.0]);
    /// let dir_2 = ndarray::arr1(&[0.0, 1.0]);
    /// let mu = Array1::linspace(-2.0, 2.0, 101);
    /// let sigma_vs_mu = model.Hall_conductivity_mu(&kmesh, &current_dir, &dir_2, &mu, 0.0, None, 1e-3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn Hall_conductivity_mu(
        &self,
        k_mesh: &Array1<usize>,
        current_dir: &Array1<f64>,
        dir_2: &Array1<f64>,
        mu: &Array1<f64>,
        T: f64,
        spin: Option<SpinDirection>,
        eta: f64,
    ) -> Result<Array1<f64>> {
        let kvec: Array2<f64> = crate::kpoints::gen_kmesh(&k_mesh)?;
        let nk: usize = kvec.len_of(Axis(0));
        let (omega_n, band): (Vec<_>, Vec<_>) = kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|x| {
                let (omega_n, band) =
                    self.berry_curvature_n_onek(&x.to_owned(), &current_dir, &dir_2, spin, eta);
                (omega_n, band)
            })
            .collect();
        let omega_n = Array2::<f64>::from_shape_vec(
            (nk, self.nsta()),
            omega_n.into_iter().flatten().collect(),
        )
        .unwrap();
        let band =
            Array2::<f64>::from_shape_vec((nk, self.nsta()), band.into_iter().flatten().collect())
                .unwrap();
        let n_mu: usize = mu.len();
        let conductivity = if T == 0.0 {
            let conductivity_new: Vec<f64> = mu
                .into_par_iter()
                .map(|x| {
                    let mut omega = Array1::<f64>::zeros(nk);
                    for k in 0..nk {
                        for i in 0..self.nsta() {
                            omega[[k]] += if band[[k, i]] > *x {
                                0.0
                            } else {
                                omega_n[[k, i]]
                            };
                        }
                    }
                    omega.sum() / self.lat.det().unwrap() / (nk as f64)
                })
                .collect();
            Array1::<f64>::from_vec(conductivity_new)
        } else {
            let beta = 1.0 / (T * 8.617e-5);
            let conductivity_new: Vec<f64> = mu
                .into_par_iter()
                .map(|x| {
                    let fermi_dirac = band.mapv(|x0| 1.0 / ((beta * (x0 - x)).exp() + 1.0));
                    let omega: Vec<f64> = omega_n
                        .axis_iter(Axis(0))
                        .zip(fermi_dirac.axis_iter(Axis(0)))
                        .map(|(a, b)| (&a * &b).sum())
                        .collect();
                    let omega: Array1<f64> = arr1(&omega);
                    omega.sum() / self.lat.det().unwrap() / (nk as f64)
                })
                .collect();
            Array1::<f64>::from_vec(conductivity_new)
        };
        Ok(conductivity)
    }
}
