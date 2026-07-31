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
//! and therefore share one [`OpticalConductivityParams`] input and one named
//! [`OpticalConductivityResult`] output. A component calculation and a full
//! Cartesian tensor calculation differ only through [`OpticalDirections`].

use ndarray::prelude::*;
use ndarray_linalg::Determinant;
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::thermodynamics::Occupation;
use crate::{Gauge, Model, RMatrixData};

use super::config::{
    DirectionPair, IntegrationDiagnostics, mesh_array, validate_broadening, validate_k_mesh,
};
use super::kernel::{eval_optical_kernel, quadrature_optical_simplex};
use super::tracking::{build_tetrahedra_3d, build_triangles_2d, global_band_track};
use super::types::{SIMPLEX_GAP_TOL, VertexKernel};

/// Brillouin-zone integration used for optical conductivity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OpticalIntegration {
    /// Evaluate the optical kernel directly on a uniform k-mesh.
    #[default]
    Direct,
    /// Interpolate band energies and velocity kernels inside simplices before
    /// evaluating the frequency denominator.
    Simplex,
}

/// Tensor components requested from an optical calculation.
#[derive(Clone, Debug, PartialEq)]
pub enum OpticalDirections<const DIM: usize> {
    /// One projected tensor component.
    Pair(DirectionPair<DIM>),
    /// Every ordered Cartesian component in row-major order:
    /// `(0,0), (0,1), ..., (DIM-1,DIM-1)`.
    Cartesian,
}

impl<const DIM: usize> OpticalDirections<DIM> {
    fn pairs(&self) -> Result<Vec<DirectionPair<DIM>>> {
        match self {
            Self::Pair(pair) => {
                pair.validate()?;
                Ok(vec![*pair])
            }
            Self::Cartesian => {
                let mut pairs = Vec::with_capacity(DIM * DIM);
                for first in 0..DIM {
                    for second in 0..DIM {
                        pairs.push(DirectionPair::cartesian(first, second)?);
                    }
                }
                Ok(pairs)
            }
        }
    }
}

/// Complete optical-conductivity configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalConductivityParams<const DIM: usize> {
    /// Number of uniform samples along each reciprocal-lattice direction.
    pub k_mesh: [usize; DIM],
    /// One projected component or the complete Cartesian tensor.
    pub directions: OpticalDirections<DIM>,
    /// Photon energies in eV.
    pub frequencies: Array1<f64>,
    /// Chemical potential in eV.
    pub chemical_potential: f64,
    /// Electronic occupation or smearing convention.
    pub occupation: Occupation,
    /// Non-negative response broadening in eV.
    pub broadening: f64,
    /// Brillouin-zone integration algorithm.
    pub integration: OpticalIntegration,
}

impl<const DIM: usize> OpticalConductivityParams<DIM> {
    /// Construct a zero-temperature component calculation using direct
    /// k-mesh integration and `1e-3` eV broadening.
    pub fn new(
        k_mesh: [usize; DIM],
        directions: DirectionPair<DIM>,
        frequencies: Array1<f64>,
        chemical_potential: f64,
    ) -> Self {
        Self {
            k_mesh,
            directions: OpticalDirections::Pair(directions),
            frequencies,
            chemical_potential,
            occupation: Occupation::ZeroTemperature,
            broadening: 1e-3,
            integration: OpticalIntegration::Direct,
        }
    }

    fn validate(&self) -> Result<Vec<DirectionPair<DIM>>> {
        validate_k_mesh(&self.k_mesh)?;
        let pairs = self.directions.pairs()?;
        if self.frequencies.is_empty() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "frequencies",
                message: "must contain at least one value".into(),
            });
        }
        if self
            .frequencies
            .iter()
            .any(|frequency| !frequency.is_finite())
        {
            return Err(TbError::InvalidResponseParameter {
                parameter: "frequencies",
                message: "all values must be finite".into(),
            });
        }
        if !self.chemical_potential.is_finite() {
            return Err(TbError::InvalidResponseParameter {
                parameter: "chemical_potential",
                message: "must be finite".into(),
            });
        }
        self.occupation.validate()?;
        validate_broadening(self.broadening)?;
        if self.integration == OpticalIntegration::Simplex && DIM == 1 {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![2, 3],
            });
        }
        Ok(pairs)
    }
}

/// Optical conductivity for one or more tensor components.
#[derive(Clone, Debug, PartialEq)]
pub struct OpticalConductivityResult<const DIM: usize> {
    /// Frequencies corresponding to the columns of `conductivity`.
    pub frequencies: Array1<f64>,
    /// Direction pair corresponding to every row of `conductivity`.
    pub directions: Vec<DirectionPair<DIM>>,
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
    pub fn optical_conductivity(
        &self,
        params: &OpticalConductivityParams<DIM>,
    ) -> Result<OpticalConductivityResult<DIM>> {
        let direction_pairs = params.validate()?;
        let k_mesh = mesh_array(&params.k_mesh);
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let thermal_width = params.occupation.energy_width()?;
        let determinant = self.lat.det()?;
        let mut conductivity =
            Array2::<Complex<f64>>::zeros((direction_pairs.len(), params.frequencies.len()));
        let mut unsafe_simplex_count = 0usize;

        for (component, directions) in direction_pairs.iter().enumerate() {
            let (direction_a, direction_b) = directions.as_arrays();
            let mut vertices: Vec<VertexKernel> = (0..k_points.nrows())
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

            match params.integration {
                OpticalIntegration::Direct => {
                    let values: Vec<Complex<f64>> = params
                        .frequencies
                        .par_iter()
                        .map(|&frequency| {
                            vertices
                                .iter()
                                .map(|vertex| {
                                    eval_optical_kernel(
                                        vertex.band.as_slice().unwrap(),
                                        &vertex.k_ab,
                                        frequency,
                                        params.broadening,
                                        params.chemical_potential,
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
                OpticalIntegration::Simplex => {
                    global_band_track(&mut vertices, &params.k_mesh);
                    let values: Vec<(Complex<f64>, usize)> = params
                        .frequencies
                        .par_iter()
                        .map(|&frequency| {
                            integrate_simplex(
                                &vertices,
                                &k_mesh,
                                frequency,
                                params.broadening,
                                params.chemical_potential,
                                thermal_width,
                            )
                        })
                        .collect();
                    for (frequency, (value, unsafe_count)) in values.into_iter().enumerate() {
                        conductivity[[component, frequency]] = value / determinant;
                        unsafe_simplex_count = unsafe_simplex_count.max(unsafe_count);
                    }
                }
            }
        }

        Ok(OpticalConductivityResult {
            frequencies: params.frequencies.clone(),
            directions: direction_pairs,
            conductivity,
            diagnostics: (params.integration == OpticalIntegration::Simplex).then_some(
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
        let params = OpticalConductivityParams {
            k_mesh: [2, 2],
            directions: OpticalDirections::Cartesian,
            frequencies: array![0.1, 0.2],
            chemical_potential: 0.0,
            occupation: Occupation::ZeroTemperature,
            broadening: 1e-3,
            integration: OpticalIntegration::Direct,
        };
        let result = model.optical_conductivity(&params).unwrap();
        assert_eq!(result.conductivity.dim(), (4, 2));
        assert_eq!(result.directions.len(), 4);
    }

    #[test]
    fn direct_and_simplex_evaluate_the_same_optical_kernel() {
        let model = qwz_model(-1.0);
        let mut params = OpticalConductivityParams::new(
            [31, 31],
            DirectionPair::new([1.0, 0.0], [1.0, 0.0]),
            array![0.2, 0.8],
            0.0,
        );
        params.broadening = 0.1;
        let direct = model.optical_conductivity(&params).unwrap();

        params.integration = OpticalIntegration::Simplex;
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
