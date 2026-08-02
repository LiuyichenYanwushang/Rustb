//! # Linear response: Berry curvature and quantum metric
//!
//! ## Theory
//!
//! The quantum geometric tensor is
//!
//! $$G^{ab}_n(\mathbf{k}) = \sum_{m=\not n}
//!   \frac{\langle\partial_a u_n|u_m\rangle\langle u_m|\partial_b u_n\rangle}
//!        {(E_n-E_m)^2} = g^{ab}_n - \frac{i}{2}\Omega^{ab}_n$$
//!
//! In terms of velocity matrix elements $v^\alpha_{nm} = \langle u_n|\partial_\alpha H|u_m\rangle$:
//!
//! $$G^{ab}_n(\mathbf{k}) = \sum_{m=\not n} \frac{v^a_{nm} v^b_{mn}}{(E_n-E_m)^2 + \eta^2}$$
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
//!   \int_{\text{BZ}} \Theta(\mu-E_n)\Omega^{xy}_n(\mathbf{k})d\mathbf{k}$$
//!
//! ## API
//!
//! | Method | Path | Formula |
//! |--------|------|---------|
//! | `BerryCurvature::berry_curvature_at` | one k-point | band-resolved $\Omega_n^{ab}$ |
//! | `Model::hall_conductivity` | direct sum or energy cut | $\sigma_{\text{AHC}}(\mu)$ |

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use rayon::prelude::*;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::Result;
use crate::thermodynamics::Occupation;

use super::config::{
    Integration, Parameters, mesh_array, parameters_occupation, validate_sorted,
};
use super::energy_cut::{integrate_fermi_cut_2d, integrate_fermi_cut_3d};
use super::kernel::quadrature_occupied_geometry_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d, global_band_track};
use super::traits::BerryCurvature;
use super::types::{SIMPLEX_GAP_TOL, VertexKernel};

/// Hall conductivity evaluated on the requested chemical-potential grid.
#[derive(Clone, Debug, PartialEq)]
pub struct HallConductivityResult {
    /// Chemical potentials copied from the input configuration.
    pub chemical_potentials: Array1<f64>,
    /// Hall response at every chemical potential.
    pub conductivity: Array1<f64>,
}

impl HallConductivityResult {
    /// Return the scalar value produced by [`Parameters::at_mu`].
    pub fn single(&self) -> Option<f64> {
        (self.conductivity.len() == 1).then(|| self.conductivity[0])
    }
}

/// Integrate occupation-weighted Berry curvature and quantum metric over the BZ.
///
/// ```text
/// total_berry(μ)  = Σ_n ∫_BZ f(E_n, μ) Ω^{ab}_n(k) dk
/// total_metric(μ) = Σ_n ∫_BZ f(E_n, μ) g^{ab}_n(k) dk
/// ```
///
/// where `Ω_n = −2 Im G_n`, `g_n = Re G_n`, and
/// `G_n = Σ_{m≠n} K^{ab}_nm / ((E_n−E_m)² + η²)`.
///
/// Inside each simplex, `K_nm` and `E_n` are linearly interpolated,
/// then the kernel is evaluated at degree‑2 symmetric quadrature points.
///
/// Values use fractional-coordinate volume. Divide by `det(lat)` for
/// Cartesian reciprocal-space volume.
pub(crate) fn integrate_occupied_geometry(
    all_pts: &[VertexKernel],
    k_mesh: &Array1<usize>,
    eta: f64,
    chemical_potentials: &Array1<f64>,
    occupation: Occupation,
) -> (Array1<f64>, Array1<f64>, usize) {
    let dim = k_mesh.len();
    let mut total_g = Array1::<f64>::zeros(chemical_potentials.len());
    let mut total_o = Array1::<f64>::zeros(chemical_potentials.len());
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
                        let (g, o) = quadrature_occupied_geometry_simplex(
                            sim,
                            eta,
                            chemical_potentials,
                            occupation,
                        );
                        total_g += &g;
                        total_o += &o;
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
                            let (g, o) = quadrature_occupied_geometry_simplex(
                                sim,
                                eta,
                                chemical_potentials,
                                occupation,
                            );
                            total_g += &g;
                            total_o += &o;
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
    /// Evaluate charge or spin Hall conductivity using the unified
    /// [`Parameters`] configuration.
    ///
    /// Reads `kmesh`, `direction` (rank 2), `mu`, `T`, `eta` and `spin` from
    /// the parameter set; `omega` and `field_symmetry` are ignored. The
    /// returned array has one value per chemical potential. Direct
    /// integration reuses the band-resolved Berry curvature for the entire
    /// chemical-potential grid; energy-cut integration tracks bands between
    /// simplex vertices before integrating the occupied region.
    pub fn hall_conductivity(
        &self,
        params: &Parameters<DIM>,
    ) -> Result<HallConductivityResult> {
        params.validate_rank2()?;
        let spin = params.spin;
        if !SPIN && let Some(direction) = spin {
            return Err(crate::TbError::SpinNotAllowed(direction));
        }
        match params.integration {
            Integration::Direct | Integration::EnergyCut => {}
            Integration::Simplex => {
                return Err(crate::TbError::InvalidResponseParameter {
                    parameter: "integration",
                    message: "hall_conductivity supports Integration::Direct or EnergyCut, not Simplex".into(),
                });
            }
        }
        if params.integration == Integration::EnergyCut {
            validate_sorted(&params.mu, "mu")?;
            if DIM != 2 && DIM != 3 {
                return Err(crate::TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2, 3],
                });
            }
        }

        let k_mesh = mesh_array(&params.kmesh);
        let determinant = self.lat.det()?;
        let dir_a = params.direction.row(0).to_owned();
        let dir_b = params.direction.row(1).to_owned();
        let occupation = parameters_occupation(params);
        let conductivity = match params.integration {
            Integration::Direct => {
                let kvec: Array2<f64> = crate::kpoints::gen_kmesh(&k_mesh)?;
                let nk = kvec.nrows();
                let band_data: Vec<Result<_>> = kvec
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|k| self.berry_curvature_at(&k, params))
                    .collect();
                let band_data = band_data.into_iter().collect::<Result<Vec<_>>>()?;
                let values: Vec<f64> = params
                    .mu
                    .par_iter()
                    .map(|&mu| {
                        let sum: f64 = band_data
                            .iter()
                            .map(|bands| {
                                bands
                                    .berry_curvature
                                    .iter()
                                    .zip(&bands.energies)
                                    .map(|(&omega, &energy)| {
                                        omega * occupation.value_unchecked(energy, mu)
                                    })
                                    .sum::<f64>()
                            })
                            .sum();
                        sum / nk as f64 / determinant
                    })
                    .collect();
                Array1::from_vec(values)
            }
            Integration::EnergyCut => {
                let chemical_potentials =
                    Array1::from_iter(params.mu.iter().copied());
                let kvec = crate::kpoints::gen_kmesh(&k_mesh)?;
                let mut all_pts: Vec<VertexKernel> = (0..kvec.nrows())
                    .into_par_iter()
                    .map(|ik| {
                        self.compute_velocity_kernel(
                            &kvec.row(ik).to_owned(),
                            &dir_a,
                            &dir_b,
                            None,
                            Gauge::Atom,
                            spin,
                        )
                    })
                    .collect();
                global_band_track(&mut all_pts, &params.kmesh);
                let width = occupation.energy_width()?;
                let sigma = match DIM {
                    2 => integrate_fermi_cut_2d(
                        &all_pts,
                        &k_mesh,
                        &chemical_potentials,
                        width,
                        params.eta,
                    ),
                    3 => integrate_fermi_cut_3d(
                        &all_pts,
                        &k_mesh,
                        &chemical_potentials,
                        width,
                        params.eta,
                    ),
                    _ => {
                        return Err(crate::TbError::InvalidDimension {
                            dim: DIM,
                            supported: vec![2, 3],
                        });
                    }
                };
                sigma / determinant
            }
            Integration::Simplex => unreachable!("rejected during validation"),
        };

        Ok(HallConductivityResult {
            chemical_potentials: params.mu.clone(),
            conductivity,
        })
    }
}
