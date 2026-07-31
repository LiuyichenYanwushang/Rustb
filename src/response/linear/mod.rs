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
    CurrentOperator, DirectionPair, mesh_array, validate_broadening, validate_chemical_potentials,
    validate_k_mesh, validate_sorted,
};
use super::energy_cut::{integrate_fermi_cut_2d, integrate_fermi_cut_3d};
use super::kernel::quadrature_occupied_geometry_simplex;
use super::tracking::{build_tetrahedra_3d, build_triangles_2d, global_band_track};
use super::traits::{BerryCurvature, BerryCurvatureParams};
use super::types::{SIMPLEX_GAP_TOL, VertexKernel};

/// Brillouin-zone integration used for Hall conductivity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HallIntegration {
    /// Uniform k-mesh summation of band-resolved Berry curvature.
    #[default]
    Direct,
    /// Simplex energy-cut integration, exact in the zero-temperature limit
    /// for linearly interpolated energies and velocity kernels.
    EnergyCut,
}

/// Complete configuration for anomalous or spin Hall conductivity.
#[derive(Clone, Debug, PartialEq)]
pub struct HallConductivityParams<const DIM: usize> {
    /// Number of uniform samples along each reciprocal-lattice direction.
    pub k_mesh: [usize; DIM],
    /// Ordered Hall tensor directions.
    pub directions: DirectionPair<DIM>,
    /// Chemical potentials in eV.
    pub chemical_potentials: Array1<f64>,
    /// Electronic occupation or smearing convention.
    pub occupation: Occupation,
    /// Charge current or a selected spin-current polarization.
    pub current: CurrentOperator,
    /// Non-negative energy-denominator regularization in eV.
    pub broadening: f64,
    /// Brillouin-zone integration algorithm.
    pub integration: HallIntegration,
}

impl<const DIM: usize> HallConductivityParams<DIM> {
    /// Construct a zero-temperature charge-Hall calculation using direct
    /// k-mesh summation and a `1e-3` eV denominator broadening.
    pub fn new(
        k_mesh: [usize; DIM],
        directions: DirectionPair<DIM>,
        chemical_potentials: Array1<f64>,
    ) -> Self {
        Self {
            k_mesh,
            directions,
            chemical_potentials,
            occupation: Occupation::ZeroTemperature,
            current: CurrentOperator::Charge,
            broadening: 1e-3,
            integration: HallIntegration::Direct,
        }
    }

    /// Convenience constructor for a single chemical potential.
    pub fn at_mu(
        k_mesh: [usize; DIM],
        directions: DirectionPair<DIM>,
        chemical_potential: f64,
    ) -> Self {
        Self::new(
            k_mesh,
            directions,
            Array1::from_vec(vec![chemical_potential]),
        )
    }

    fn validate(&self) -> Result<()> {
        validate_k_mesh(&self.k_mesh)?;
        self.directions.validate()?;
        validate_chemical_potentials(&self.chemical_potentials)?;
        self.occupation.validate()?;
        validate_broadening(self.broadening)?;
        if self.integration == HallIntegration::EnergyCut {
            validate_sorted(&self.chemical_potentials, "chemical_potentials")?;
            if DIM != 2 && DIM != 3 {
                return Err(crate::TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2, 3],
                });
            }
        }
        Ok(())
    }
}

/// Hall conductivity evaluated on the requested chemical-potential grid.
#[derive(Clone, Debug, PartialEq)]
pub struct HallConductivityResult {
    /// Chemical potentials copied from the input configuration.
    pub chemical_potentials: Array1<f64>,
    /// Hall response at every chemical potential.
    pub conductivity: Array1<f64>,
}

impl HallConductivityResult {
    /// Return the scalar value produced by [`HallConductivityParams::at_mu`].
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
    /// Evaluate charge or spin Hall conductivity using one coherent API.
    ///
    /// The returned array has one value per chemical potential. Direct
    /// integration reuses the band-resolved Berry curvature for the entire
    /// chemical-potential grid; energy-cut integration tracks bands between
    /// simplex vertices before integrating the occupied region.
    pub fn hall_conductivity(
        &self,
        params: &HallConductivityParams<DIM>,
    ) -> Result<HallConductivityResult> {
        params.validate()?;
        let spin = params.current.spin_direction();
        if !SPIN && let Some(direction) = spin {
            return Err(crate::TbError::SpinNotAllowed(direction));
        }

        let k_mesh = mesh_array(&params.k_mesh);
        let determinant = self.lat.det()?;
        let (dir_a, dir_b) = params.directions.as_arrays();
        let conductivity = match params.integration {
            HallIntegration::Direct => {
                let kvec: Array2<f64> = crate::kpoints::gen_kmesh(&k_mesh)?;
                let nk = kvec.nrows();
                let berry_params = BerryCurvatureParams {
                    directions: params.directions,
                    current: params.current,
                    broadening: params.broadening,
                };
                let band_data: Vec<Result<_>> = kvec
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|k| self.berry_curvature_at(&k, &berry_params))
                    .collect();
                let band_data = band_data.into_iter().collect::<Result<Vec<_>>>()?;
                let values: Vec<f64> = params
                    .chemical_potentials
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
                                        omega * params.occupation.value_unchecked(energy, mu)
                                    })
                                    .sum::<f64>()
                            })
                            .sum();
                        sum / nk as f64 / determinant
                    })
                    .collect();
                Array1::from_vec(values)
            }
            HallIntegration::EnergyCut => {
                let chemical_potentials =
                    Array1::from_iter(params.chemical_potentials.iter().copied());
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
                global_band_track(&mut all_pts, &params.k_mesh);
                let width = params.occupation.energy_width()?;
                let sigma = match DIM {
                    2 => integrate_fermi_cut_2d(
                        &all_pts,
                        &k_mesh,
                        &chemical_potentials,
                        width,
                        params.broadening,
                    ),
                    3 => integrate_fermi_cut_3d(
                        &all_pts,
                        &k_mesh,
                        &chemical_potentials,
                        width,
                        params.broadening,
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
        };

        Ok(HallConductivityResult {
            chemical_potentials: params.chemical_potentials.clone(),
            conductivity,
        })
    }
}
