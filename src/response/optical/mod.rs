//! Optical conductivity from gauge-invariant velocity kernels.
//!
//! The direct and simplex algorithms evaluate the same kernel,
//!
//! ```math
//! \sigma^{ab}(\omega) = \sum_{n\ne m}\int_{BZ}
//! \frac{(f_n-f_m)v^a_{nm}v^b_{mn}}
//! {(E_n-E_m)^2-(\omega+i\eta)^2}\,dk,
//! ```
//!
//! and therefore share one [`Parameters`] input and one named
//! [`OpticalConductivityResult`] output. A component calculation and a full
//! Cartesian tensor calculation differ only through the direction matrix.

use ndarray::prelude::*;
use ndarray_linalg::Determinant;
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::{Gauge, Model, RMatrixData};

use super::config::{
    Integration, IntegrationDiagnostics, Parameters, mesh_array, parameters_occupation,
    validate_broadening, validate_chemical_potentials, validate_direction_matrix,
    validate_temperature,
};
use super::kernel::{eval_optical_kernel, quadrature_optical_simplex};
use super::tracking::{build_tetrahedra_3d, build_triangles_2d, global_band_track};
use super::types::{SIMPLEX_GAP_TOL, VertexKernel};

/// Direction pairs requested from an optical calculation.
///
/// A `direction` matrix with two rows selects one projected component; an
/// empty direction matrix selects every ordered Cartesian component in
/// row-major order `(0,0), (0,1), ..., (DIM-1,DIM-1)`.
fn direction_pairs<const DIM: usize>(params: &Parameters<DIM>) -> Result<Vec<Array2<f64>>> {
    if params.direction.nrows() == 0 {
        let mut pairs = Vec::with_capacity(DIM * DIM);
        for first in 0..DIM {
            for second in 0..DIM {
                let mut matrix = Array2::<f64>::zeros((2, DIM));
                matrix[[0, first]] = 1.0;
                matrix[[1, second]] = 1.0;
                pairs.push(matrix);
            }
        }
        return Ok(pairs);
    }
    validate_direction_matrix(&params.direction, 2, DIM)?;
    Ok(vec![params.direction.clone()])
}

/// Optical conductivity for one or more tensor components.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalConductivityResult<const DIM: usize> {
    /// Frequencies corresponding to the columns of `conductivity`.
    pub frequencies: Array1<f64>,
    /// Direction matrix (shape `(2, DIM)`) corresponding to every row of
    /// `conductivity`.
    pub directions: Vec<Array2<f64>>,
    /// Complex conductivity with shape `(number_of_components, frequencies)`.
    pub conductivity: Array2<Complex<f64>>,
    /// Present only for simplex integration.
    pub diagnostics: Option<IntegrationDiagnostics>,
}

impl<const DIM: usize> OpticalConductivityResult<DIM> {
    /// View one component by row index.
    pub fn component(&self, index: usize) -> Option<ArrayView1<'_, Complex<f64>>> {
        (index < self.conductivity.nrows()).then(|| self.conductivity.row(index))
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Compute a projected component or the full Cartesian optical tensor.
    ///
    /// Reads `kmesh`, `direction` (rank 2, or empty for the full Cartesian
    /// tensor), `mu` (single value), `T`, `eta` and `omega` from the
    /// parameter set; `spin` and `field_symmetry` are ignored.
    pub fn optical_conductivity(
        &self,
        params: &Parameters<DIM>,
    ) -> Result<OpticalConductivityResult<DIM>> {
        validate_chemical_potentials(&params.mu)?;
        if params.mu.len() != 1 {
            return Err(TbError::InvalidResponseParameter {
                parameter: "mu",
                message: "optical_conductivity expects a single chemical potential".into(),
            });
        }
        if params.omega.is_empty() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "omega",
                message: "must contain at least one value".into(),
            });
        }
        if params.omega.iter().any(|frequency| !frequency.is_finite()) {
            return Err(TbError::InvalidResponseParameter {
                parameter: "omega",
                message: "all values must be finite".into(),
            });
        }
        validate_temperature(&params.T)?;
        validate_broadening(params.eta)?;
        crate::response::config::validate_k_mesh(&params.kmesh)?;
        match params.integration {
            Integration::Direct | Integration::Simplex => {}
            Integration::EnergyCut => {
                return Err(TbError::InvalidResponseParameter {
                    parameter: "integration",
                    message: "optical_conductivity supports Integration::Direct or Simplex, not EnergyCut".into(),
                });
            }
        }
        if params.integration == Integration::Simplex && DIM == 1 {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![2, 3],
            });
        }
        let direction_pairs = direction_pairs(params)?;
        let k_mesh = mesh_array(&params.kmesh);
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let thermal_width = parameters_occupation(params).energy_width()?;
        let chemical_potential = params.mu[0];
        let determinant = self.lat.det()?;
        let mut conductivity =
            Array2::<Complex<f64>>::zeros((direction_pairs.len(), params.omega.len()));
        let mut unsafe_simplex_count = 0usize;

        for (component, directions) in direction_pairs.iter().enumerate() {
            let direction_a = directions.row(0).to_owned();
            let direction_b = directions.row(1).to_owned();
            let vertices: Vec<Result<VertexKernel>> = (0..k_points.nrows())
                .into_par_iter()
                .map(|index| {
                    self.compute_velocity_kernel(
                        &k_points.row(index).to_owned(),
                        &direction_a,
                        &direction_b,
                        None,
                        Gauge::Atom,
                        None,
                    )
                })
                .collect();
            let mut vertices: Vec<VertexKernel> = vertices.into_iter().collect::<Result<_>>()?;

            match params.integration {
                Integration::Direct => {
                    let values: Vec<Complex<f64>> = params
                        .omega
                        .par_iter()
                        .map(|&frequency| {
                            vertices
                                .iter()
                                .map(|vertex| {
                                    eval_optical_kernel(
                                        vertex.band.as_slice().unwrap(),
                                        &vertex.k_ab,
                                        frequency,
                                        params.eta,
                                        chemical_potential,
                                        thermal_width,
                                        self.nsta(),
                                    )
                                })
                                .sum::<Complex<f64>>()
                                / k_points.nrows() as f64
                                / determinant
                        })
                        .collect();
                    conductivity
                        .row_mut(component)
                        .assign(&Array1::from_vec(values));
                }
                Integration::Simplex => {
                    global_band_track(&mut vertices, &params.kmesh);
                    let values: Vec<(Complex<f64>, usize)> = params
                        .omega
                        .par_iter()
                        .map(|&frequency| {
                            integrate_simplex(
                                &vertices,
                                &k_mesh,
                                frequency,
                                params.eta,
                                chemical_potential,
                                thermal_width,
                            )
                        })
                        .collect();
                    for (frequency, (value, unsafe_count)) in values.into_iter().enumerate() {
                        conductivity[[component, frequency]] = value / determinant;
                        unsafe_simplex_count = unsafe_simplex_count.max(unsafe_count);
                    }
                }
                Integration::EnergyCut => unreachable!("rejected during validation"),
            }
        }

        Ok(OpticalConductivityResult {
            frequencies: params.omega.clone(),
            directions: direction_pairs,
            conductivity,
            diagnostics: (params.integration == Integration::Simplex).then_some(
                IntegrationDiagnostics {
                    unsafe_simplex_count,
                },
            ),
        })
    }
}

fn integrate_simplex(
    vertices: &[VertexKernel],
    k_mesh: &Array1<usize>,
    frequency: f64,
    broadening: f64,
    chemical_potential: f64,
    thermal_width: f64,
) -> (Complex<f64>, usize) {
    let mut total = Complex::new(0.0, 0.0);
    let mut unsafe_simplex_count = 0usize;
    match k_mesh.len() {
        2 => {
            let (nx, ny) = (k_mesh[0], k_mesh[1]);
            for ix in 0..nx {
                for iy in 0..ny {
                    for simplex in &build_triangles_2d(
                        ix,
                        iy,
                        nx,
                        ny,
                        1.0 / nx as f64,
                        1.0 / ny as f64,
                        vertices,
                    ) {
                        unsafe_simplex_count += usize::from(simplex.diag.min_gap < SIMPLEX_GAP_TOL);
                        total += quadrature_optical_simplex(
                            simplex,
                            frequency,
                            broadening,
                            chemical_potential,
                            thermal_width,
                        );
                    }
                }
            }
        }
        3 => {
            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        for simplex in &build_tetrahedra_3d(
                            ix,
                            iy,
                            iz,
                            nx,
                            ny,
                            nz,
                            1.0 / nx as f64,
                            1.0 / ny as f64,
                            1.0 / nz as f64,
                            vertices,
                        ) {
                            unsafe_simplex_count +=
                                usize::from(simplex.diag.min_gap < SIMPLEX_GAP_TOL);
                            total += quadrature_optical_simplex(
                                simplex,
                                frequency,
                                broadening,
                                chemical_potential,
                                thermal_width,
                            );
                        }
                    }
                }
            }
        }
        _ => unreachable!("validated before simplex integration"),
    }
    (total, unsafe_simplex_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn qwz_model(mass: f64) -> Model<false, 2> {
        let mut model = Model::<false, 2>::tb_model(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 0.0], [0.0, 0.0]],
            None,
        )
        .unwrap();
        model.add_onsite(&array![mass, -mass], None);
        model.add_hop(Complex::new(0.0, -0.5), 0, 1, &array![1, 0], None);
        model.add_hop(Complex::new(0.0, 0.5), 0, 1, &array![-1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);
        model.add_hop(0.5, 0, 1, &array![0, -1], None);
        for displacement in [array![1, 0], array![-1, 0], array![0, 1], array![0, -1]] {
            model.add_hop(0.5, 0, 0, &displacement, None);
            model.add_hop(-0.5, 1, 1, &displacement, None);
        }
        model
    }

    #[test]
    fn cartesian_request_has_dim_squared_components() {
        let model =
            Model::<false, 2>::tb_model(array![[1.0, 0.0], [0.0, 1.0]], array![[0.0, 0.0]], None)
                .unwrap();
        // An empty direction matrix selects the full Cartesian tensor.
        let mut params = Parameters::at_mu([2, 2], Array2::zeros((0, 2)), 0.0);
        params.omega = array![0.1, 0.2];
        let result = model.optical_conductivity(&params).unwrap();
        assert_eq!(result.conductivity.dim(), (4, 2));
        assert_eq!(result.directions.len(), 4);
    }

    #[test]
    fn direct_and_simplex_evaluate_the_same_optical_kernel() {
        let model = qwz_model(-1.0);
        let mut params = Parameters::rank2([31, 31], [1.0, 0.0], [1.0, 0.0], array![0.0]);
        params.omega = array![0.2, 0.8];
        params.eta = 0.1;
        let direct = model.optical_conductivity(&params).unwrap();

        params.integration = Integration::Simplex;
        let simplex = model.optical_conductivity(&params).unwrap();
        assert!(simplex.diagnostics.is_some());
        for (&direct_value, &simplex_value) in
            direct.conductivity.iter().zip(simplex.conductivity.iter())
        {
            let scale = direct_value.norm().max(simplex_value.norm()).max(1.0);
            assert!((direct_value - simplex_value).norm() < 5e-2 * scale);
        }
    }
}
