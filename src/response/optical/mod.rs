//! Optical conductivity from gauge-invariant velocity kernels.
//!
//! The direct and simplex algorithms evaluate the same kernel,
//!
//! ```math
//! \widetilde\sigma^{ab}(\omega) = \frac1V\sum_{n\ne m}\int_{[0,1)^{DIM}}
//! \frac{-i(f_n-f_m)v^a_{nm}v^b_{mn}}
//! {(E_n-E_m)(E_n-E_m+\omega+i\eta)}\,d k_{\rm frac},
//! ```
//!
//! with `V = |det(lat)|` and `v = dH/dk_cart`. The physical factor `e²/hbar`
//! is omitted. This is the interband part of the
//! [Kubo conductivity](https://wannier90.readthedocs.io/en/latest/user_guide/postw90/berry/#berry-taskkubo-optical-conductivity-and-joint-density-of-states);
//! intraband Drude terms and exactly degenerate pairs are excluded. In the
//! insulating, zero-broadening DC limit, for `det(lat) > 0`, its antisymmetric
//! part is minus the occupied Berry-curvature integral returned by
//! `hall_conductivity` (which retains a signed-determinant convention).
//!
//! The algorithms share one [`Parameters`] input and one named
//! [`OpticalConductivityResult`] output. A component calculation and a full
//! Cartesian tensor calculation differ only through the direction matrix.

use ndarray::array;
use ndarray::prelude::*;
use ndarray_linalg::{Determinant, Eigh, UPLO};
use num_complex::Complex;
use rayon::prelude::*;

use crate::error::{Result, TbError};
use crate::{Gauge, Model, RMatrixData};

use super::config::{
    Integration, IntegrationDiagnostics, Parameters, ResponseAxis, mesh_array, occupation_for,
    validate_direction_matrix,
};
use super::kernel::{eval_optical_kernel, quadrature_optical_simplex};
use super::tracking::{
    build_tetrahedra_3d_diagavg, build_triangles_2d, global_band_track, global_band_track_with,
};
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
    /// The axis the columns of `conductivity` are indexed by;
    /// [`ResponseAxis::Fixed`] for a single evaluation at the fixed conditions.
    pub axis: ResponseAxis,
    /// Direction matrix (shape `(2, DIM)`) corresponding to every row of
    /// `conductivity`.
    pub directions: Vec<Array2<f64>>,
    /// Complex conductivity with shape `(number_of_components, samples)`.
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
    /// Reads `conditions` (at most one axis sampled; this is the only entry
    /// point that may sample `omega_ev`), `kmesh`, `direction` (rank 2, or
    /// empty for the full Cartesian tensor), `eta_ev` and `integration`;
    /// `spin` and `field_symmetry` are ignored. The columns of
    /// `conductivity` follow the sampled axis. Eigenstates,
    /// velocity kernels and band tracking are prepared once and reused by every
    /// sample.
    ///
    /// A full tensor shares one eigendecomposition and band-tracking pass
    /// across all components, retaining `DIM` band-velocity matrices per k.
    ///
    /// Returns the interband conductivity with `e²/hbar` omitted; no Drude
    /// term is included. Positive `eta_ev` resolves optical resonances. A pole
    /// at `eta_ev = 0`, or nonfinite numerical output, returns an error.
    pub fn optical_conductivity(
        &self,
        params: &Parameters<DIM>,
    ) -> Result<OpticalConductivityResult<DIM>> {
        self.validate()?;
        let resolved = params.validate_common()?;
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
        let eta = params.broadening()?;
        let samples = resolved.len();
        let widths: Vec<f64> = (0..samples)
            .map(|index| occupation_for(resolved.point(index).0).energy_width())
            .collect::<Result<Vec<_>>>()?;
        let direction_pairs = direction_pairs(params)?;
        let k_mesh = mesh_array(&params.kmesh);
        let k_points = crate::kpoints::gen_kmesh::<f64>(&k_mesh)?;
        let determinant = self.lat.det()?.abs();
        let mut conductivity = Array2::<Complex<f64>>::zeros((direction_pairs.len(), samples));
        let mut unsafe_simplex_count = 0usize;
        let full_tensor = direction_pairs.len() > 1;
        let mut velocities = Vec::new();
        let mut vertices: Vec<VertexKernel> = if full_tensor {
            let directions = Array2::<f64>::eye(DIM);
            let data: Vec<Result<_>> = k_points
                .outer_iter()
                .into_par_iter()
                .map(|k| {
                    let (projected, ham) = self.gen_v_projected(&k, Gauge::Atom, &directions);
                    #[cfg(test)]
                    super::config::counters::count_eigen_decomposition();
                    let (band, evec) = ham.eigh(UPLO::Lower)?;
                    // Match compute_velocity_kernel's ndarray-linalg convention.
                    let ket = evec.mapv(|z| z.conj());
                    let mut velocity = Array3::zeros((DIM, self.nsta(), self.nsta()));
                    for (mut out, v) in velocity.outer_iter_mut().zip(projected.outer_iter()) {
                        out.assign(&evec.t().dot(&v.dot(&ket)));
                    }
                    let vertex = VertexKernel {
                        band,
                        evec,
                        k_ab: &velocity.index_axis(Axis(0), 0)
                            * &velocity.index_axis(Axis(0), 0).t(),
                        k_bc: None,
                        k_ac: None,
                        vdiag: None,
                        vdiag_a: None,
                        vdiag_b: None,
                    };
                    Ok((vertex, velocity))
                })
                .collect();
            let (vertices, all_velocities): (Vec<_>, Vec<_>) = data
                .into_iter()
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();
            velocities = all_velocities;
            vertices
        } else {
            let direction_a = direction_pairs[0].row(0).to_owned();
            let direction_b = direction_pairs[0].row(1).to_owned();
            let vertices: Vec<Result<_>> = (0..k_points.nrows())
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
            vertices.into_iter().collect::<Result<_>>()?
        };
        if params.integration == Integration::Simplex {
            if full_tensor {
                global_band_track_with(&mut vertices, &params.kmesh, |index, permutation| {
                    velocities[index] = velocities[index]
                        .select(Axis(1), permutation)
                        .select(Axis(2), permutation);
                });
            } else {
                global_band_track(&mut vertices, &params.kmesh);
            }
        }
        // Simplex construction uses energies and kernels after tracking; the
        // eigenvectors can be released before integrating any frequencies.
        for vertex in &mut vertices {
            vertex.evec = Array2::zeros((0, 0));
        }

        for component in 0..direction_pairs.len() {
            if full_tensor {
                let (a, b) = (component / DIM, component % DIM);
                for (vertex, velocity) in vertices.iter_mut().zip(&velocities) {
                    vertex.k_ab.assign(
                        &(&velocity.index_axis(Axis(0), a) * &velocity.index_axis(Axis(0), b).t()),
                    );
                }
            }

            match params.integration {
                Integration::Direct => {
                    let values: Vec<Complex<f64>> = (0..samples)
                        .into_par_iter()
                        .map(|index| {
                            let (_, mu, frequency) = resolved.point(index);
                            vertices
                                .iter()
                                .map(|vertex| {
                                    eval_optical_kernel(
                                        vertex.band.as_slice().unwrap(),
                                        &vertex.k_ab,
                                        frequency,
                                        eta,
                                        mu,
                                        widths[index],
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
                    let values = match &resolved.axis {
                        ResponseAxis::Frequency(frequencies) => {
                            let (_, mu, _) = resolved.point(0);
                            let (values, unsafe_count) = integrate_simplex(
                                &vertices,
                                &k_mesh,
                                frequencies,
                                eta,
                                mu,
                                widths[0],
                            );
                            unsafe_simplex_count = unsafe_simplex_count.max(unsafe_count);
                            values
                        }
                        // A sampled temperature or chemical potential keeps one
                        // frequency and redoes the quadrature on the shared
                        // simplex decomposition.
                        _ => {
                            let mut values = Array1::<Complex<f64>>::zeros(samples);
                            for index in 0..samples {
                                let (_, mu, frequency) = resolved.point(index);
                                let (sample, unsafe_count) = integrate_simplex(
                                    &vertices,
                                    &k_mesh,
                                    &array![frequency],
                                    eta,
                                    mu,
                                    widths[index],
                                );
                                values[index] = sample[0];
                                unsafe_simplex_count = unsafe_simplex_count.max(unsafe_count);
                            }
                            values
                        }
                    };
                    conductivity
                        .row_mut(component)
                        .assign(&(values / determinant));
                }
                Integration::EnergyCut => unreachable!("rejected during validation"),
            }
        }

        if conductivity
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(TbError::Other(
                "optical conductivity is nonfinite; use positive eta at resonances and check the model's energy scales".into(),
            ));
        }
        Ok(OpticalConductivityResult {
            axis: resolved.axis,
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
    frequencies: &Array1<f64>,
    broadening: f64,
    chemical_potential: f64,
    thermal_width: f64,
) -> (Array1<Complex<f64>>, usize) {
    // Fixed spatial chunks bound spectrum buffers and preserve summation order
    // across Rayon thread counts. Interpolate each quadrature point once for
    // the whole frequency grid, without storing every quadrature matrix.
    let cell_count: usize = k_mesh.iter().product();
    let chunk_count = cell_count.div_ceil(256);
    let mut result = Array1::zeros(frequencies.len());
    let mut unsafe_count = 0;
    // Retain at most 32 partial spectra, independent of mesh size.
    for first_chunk in (0..chunk_count).step_by(32) {
        let chunks: Vec<_> = (first_chunk..chunk_count.min(first_chunk + 32))
            .into_par_iter()
            .map(|chunk| {
                let mut total = Array1::<Complex<f64>>::zeros(frequencies.len());
                let mut unsafe_simplex_count = 0usize;
                for cell in (chunk * 256)..cell_count.min((chunk + 1).saturating_mul(256)) {
                    match k_mesh.len() {
                        2 => {
                            let (nx, ny) = (k_mesh[0], k_mesh[1]);
                            let (ix, iy) = (cell / ny, cell % ny);
                            for simplex in &build_triangles_2d(
                                ix,
                                iy,
                                nx,
                                ny,
                                1.0 / nx as f64,
                                1.0 / ny as f64,
                                vertices,
                            ) {
                                unsafe_simplex_count +=
                                    usize::from(simplex.diag.min_gap < SIMPLEX_GAP_TOL);
                                quadrature_optical_simplex(
                                    simplex,
                                    frequencies,
                                    broadening,
                                    chemical_potential,
                                    thermal_width,
                                    &mut total,
                                );
                            }
                        }
                        3 => {
                            let (nx, ny, nz) = (k_mesh[0], k_mesh[1], k_mesh[2]);
                            let (ix, iy, iz) = (cell / (ny * nz), (cell / nz) % ny, cell % nz);
                            for simplex in &build_tetrahedra_3d_diagavg(
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
                                quadrature_optical_simplex(
                                    simplex,
                                    frequencies,
                                    broadening,
                                    chemical_potential,
                                    thermal_width,
                                    &mut total,
                                );
                            }
                        }
                        _ => unreachable!("validated before simplex integration"),
                    }
                }
                (total, unsafe_simplex_count)
            })
            .collect();
        for (values, extra) in chunks {
            result += &values;
            unsafe_count += extra;
        }
    }
    (result, unsafe_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::config::{Conditions, FieldSymmetry, ResponseOptions, Sampling};
    use ndarray::array;

    #[test]
    fn optical_dimer_matches_analytic_absorption() {
        // Independent dimers: E=+-1, |v_x,-+|^2=1/4 at every k, no Drude term.
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0], [0.5, 0.0]], None)
                .unwrap();
        model.set_hop(1.0, 0, 1, &array![0, 0], None);
        let omegas = array![0.0, 0.3, -0.3, 2.0];
        let eta = 0.2;
        let mut params = Parameters::<2> {
            conditions: Conditions {
                t_kelvin: Sampling::Fixed(0.0),
                mu_ev: Sampling::Fixed(0.0),
                omega_ev: Sampling::Values(omegas.clone()),
            },
            kmesh: [3, 4],
            direction: Array2::zeros((0, 2)),
            integration: Integration::Direct,
            spin: None,
            field_symmetry: FieldSymmetry::Symmetrized,
            eta_ev: Some(eta),
        };
        for handedness in [1.0, -1.0] {
            model.lat[[0, 0]] = handedness;
            for integration in [Integration::Direct, Integration::Simplex] {
                params.integration = integration;
                let result = model.optical_conductivity(&params).unwrap();
                for (index, &omega) in omegas.iter().enumerate() {
                    let z = Complex::new(omega, eta);
                    let expected = Complex::new(0.0, -0.25) * z / (4.0 - z * z);
                    assert!(expected.re > 0.0);
                    assert!((result.conductivity[[0, index]] - expected).norm() < 1e-12);
                    for component in 1..4 {
                        assert!(result.conductivity[[component, index]].norm() < 1e-12);
                    }
                }
                assert!(
                    (result.conductivity[[0, 1]].conj() - result.conductivity[[0, 2]]).norm()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn optical_gamma_tensor_keeps_longitudinal_and_hall_parts() {
        // H(q) = sin(q_x) sigma_x + sin(q_y) sigma_y + sigma_z.
        // At Gamma: gap=2, Kxx=Kyy=1, Kxy=-i for the occupied lower band.
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0], [0.0, 0.0]], None)
                .unwrap();
        model.set_onsite(&array![1.0, -1.0], None);
        model.set_hop(Complex::new(0.0, -0.5), 0, 1, &array![1, 0], None);
        model.set_hop(Complex::new(0.0, 0.5), 0, 1, &array![-1, 0], None);
        model.set_hop(-0.5, 0, 1, &array![0, 1], None);
        model.set_hop(0.5, 0, 1, &array![0, -1], None);
        let omegas = array![0.0, 0.3, -0.3, 2.0];
        let eta = 0.2;
        let params = Parameters::<2> {
            conditions: Conditions {
                t_kelvin: Sampling::Fixed(0.0),
                mu_ev: Sampling::Fixed(0.0),
                omega_ev: Sampling::Values(omegas.clone()),
            },
            kmesh: [1, 1],
            direction: Array2::zeros((0, 2)),
            integration: Integration::Direct,
            spin: None,
            field_symmetry: FieldSymmetry::Symmetrized,
            eta_ev: Some(eta),
        };
        let result = model.optical_conductivity(&params).unwrap();
        for (index, &omega) in omegas.iter().enumerate() {
            let z = Complex::new(omega, eta);
            let diagonal = Complex::new(0.0, -1.0) * z / (4.0 - z * z);
            let hall = -2.0 / (4.0 - z * z);
            for (component, expected) in [diagonal, hall, -hall, diagonal].into_iter().enumerate() {
                assert!((result.conductivity[[component, index]] - expected).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn optical_unbroadened_poles_return_errors() {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0], [0.5, 0.0]], None)
                .unwrap();
        model.set_hop(1.0, 0, 1, &array![0, 0], None);
        let mut params = Parameters::rank2(
            Conditions {
                t_kelvin: Sampling::Fixed(0.0),
                mu_ev: Sampling::Fixed(0.0),
                omega_ev: Sampling::Values(array![0.3]),
            },
            [1, 1],
            [1.0, 0.0],
            [1.0, 0.0],
            ResponseOptions {
                integration: Integration::Direct,
                spin: None,
                field_symmetry: FieldSymmetry::Symmetrized,
                eta_ev: Some(0.0),
            },
        );
        for integration in [Integration::Direct, Integration::Simplex] {
            params.integration = integration;
            params.conditions.omega_ev = Sampling::Values(array![0.3]);
            let regular = model.optical_conductivity(&params).unwrap();
            let expected = Complex::new(0.0, -0.25 * 0.3 / (4.0 - 0.3 * 0.3));
            assert!((regular.conductivity[[0, 0]] - expected).norm() < 1e-12);
            params.conditions.omega_ev = Sampling::Values(array![2.0]);
            assert!(model.optical_conductivity(&params).is_err());
        }
    }

    #[test]
    fn optical_kernel_resolves_thermal_near_degeneracy() {
        let k = array![
            [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]
        ];
        let gap = 1e-18;
        let z = Complex::new(0.3, 0.2);
        let actual = eval_optical_kernel(&[-gap / 2.0, gap / 2.0], &k, z.re, z.im, 0.0, 1.0, 2);
        // tanh(gap/4)/gap -> 1/4, so the pair tends to i/(2z).
        let expected = Complex::new(0.0, 0.5) / z;
        assert!((actual - expected).norm() < 1e-12);
        assert_eq!(
            eval_optical_kernel(&[0.0, 0.0], &k, z.re, z.im, 0.0, 1.0, 2),
            Complex::new(0.0, 0.0)
        );
    }

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
        let params = Parameters::<2> {
            conditions: Conditions {
                t_kelvin: Sampling::Fixed(0.0),
                mu_ev: Sampling::Fixed(0.0),
                omega_ev: Sampling::Values(array![0.1, 0.2]),
            },
            kmesh: [2, 2],
            direction: Array2::zeros((0, 2)),
            integration: Integration::Direct,
            spin: None,
            field_symmetry: FieldSymmetry::Symmetrized,
            eta_ev: Some(1e-3),
        };
        let result = model.optical_conductivity(&params).unwrap();
        assert_eq!(result.conductivity.dim(), (4, 2));
        assert_eq!(result.directions.len(), 4);
    }

    fn assert_tensor_components<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
    ) {
        let omegas = array![-0.7, 0.0, 0.2, 0.8];
        let mut params = Parameters::<DIM> {
            conditions: Conditions {
                t_kelvin: Sampling::Fixed(300.0),
                mu_ev: Sampling::Fixed(0.3),
                omega_ev: Sampling::Values(omegas.clone()),
            },
            kmesh: [5; DIM],
            direction: Array2::zeros((0, DIM)),
            integration: Integration::Direct,
            spin: None,
            field_symmetry: FieldSymmetry::Symmetrized,
            eta_ev: Some(0.13),
        };
        for integration in [Integration::Direct, Integration::Simplex] {
            params.integration = integration;
            params.direction = Array2::zeros((0, DIM));
            let tensor = model.optical_conductivity(&params).unwrap();
            assert!(tensor.conductivity.iter().any(|z| z.norm() > 1e-8));
            for (component, directions) in tensor.directions.iter().enumerate() {
                params.direction = directions.clone();
                let projected = model.optical_conductivity(&params).unwrap();
                assert_eq!(tensor.diagnostics, projected.diagnostics);
                for (&full, &single) in tensor
                    .conductivity
                    .row(component)
                    .iter()
                    .zip(&projected.conductivity)
                {
                    assert!((full - single).norm() < 2e-11 * single.norm().max(1.0));
                }
            }
            // Arbitrary direction projection is bilinear in the two vectors.
            params.direction = Array2::from_shape_fn((2, DIM), |(row, axis)| {
                (axis + 1) as f64 * if row == 0 { 0.3 } else { -0.2 }
            });
            let projected = model.optical_conductivity(&params).unwrap();
            for frequency in 0..omegas.len() {
                let mut expected = Complex::new(0.0, 0.0);
                for a in 0..DIM {
                    for b in 0..DIM {
                        expected += params.direction[[0, a]]
                            * params.direction[[1, b]]
                            * tensor.conductivity[[a * DIM + b, frequency]];
                    }
                }
                assert!(
                    (projected.conductivity[[0, frequency]] - expected).norm()
                        < 2e-11 * expected.norm().max(1.0)
                );
                let mut single_frequency = params.clone();
                single_frequency.conditions.omega_ev = Sampling::Values(array![omegas[frequency]]);
                let single = model.optical_conductivity(&single_frequency).unwrap();
                assert!(
                    (single.conductivity[[0, 0]] - projected.conductivity[[0, frequency]]).norm()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn tensor_reuses_bands_without_changing_components_or_frequency_scans() {
        assert_tensor_components(&qwz_model(-0.7));
        let mut spinful = Model::<true, 3, crate::HasRMatrix>::tb_model(
            array![[1.0, 0.2, 0.1], [0.0, 1.3, 0.2], [0.1, 0.0, 0.9]],
            array![[0.1, 0.2, 0.3]],
            None,
        )
        .unwrap();
        spinful.set_onsite(&array![0.8], Some(crate::SpinDirection::Z));
        for (axis, spin) in [
            crate::SpinDirection::X,
            crate::SpinDirection::Y,
            crate::SpinDirection::Z,
        ]
        .into_iter()
        .enumerate()
        {
            let mut displacement = Array1::zeros(3);
            displacement[axis] = 1;
            spinful.set_hop(
                Complex::new(0.2, 0.1 * (axis + 1) as f64),
                0,
                0,
                &displacement,
                Some(spin),
            );
        }
        assert_tensor_components(&spinful);
    }

    #[test]
    fn tracking_permutes_both_velocity_band_indices() {
        let canonical = array![
            [Complex::new(1.0, 0.0), Complex::new(0.3, 0.7)],
            [Complex::new(0.3, -0.7), Complex::new(2.0, 0.0)],
        ];
        let mut velocities = vec![
            canonical.clone(),
            canonical.select(Axis(0), &[1, 0]).select(Axis(1), &[1, 0]),
        ];
        let mut vertices: Vec<_> = velocities
            .iter()
            .enumerate()
            .map(|(index, v)| VertexKernel {
                band: array![1.0, 2.0],
                evec: if index == 0 {
                    Array2::eye(2)
                } else {
                    Array2::eye(2).select(Axis(1), &[1, 0])
                },
                k_ab: v * &v.t(),
                k_bc: None,
                k_ac: None,
                vdiag: None,
                vdiag_a: None,
                vdiag_b: None,
            })
            .collect();
        let mut calls = 0;
        global_band_track_with(&mut vertices, &[2, 1], |index, permutation| {
            calls += 1;
            velocities[index] = velocities[index]
                .select(Axis(0), permutation)
                .select(Axis(1), permutation);
        });
        assert_eq!(calls, 1);
        assert_eq!(vertices[1].band, array![2.0, 1.0]);
        assert_eq!(velocities[1], canonical);
        assert_eq!(vertices[1].k_ab, &canonical * &canonical.t());
    }

    #[test]
    fn simplex_spectrum_preserves_constant_integrals_across_chunk_batches() {
        let vertex = VertexKernel {
            band: array![-1.0, 1.0],
            k_ab: array![
                [Complex::new(0.0, 0.0), Complex::i()],
                [-Complex::i(), Complex::new(0.0, 0.0)]
            ],
            evec: Array2::eye(2),
            k_bc: None,
            k_ac: None,
            vdiag: None,
            vdiag_a: None,
            vdiag_b: None,
        };
        let frequencies = array![0.3, 1.1];
        // f_lower=1, f_upper=0: summing both off-diagonal terms gives
        // 2 / (4 - (omega+i eta)^2), independent of the spatial mesh.
        let exact: Array1<Complex<f64>> =
            frequencies.mapv(|omega| 2.0 / (4.0 - Complex::new(omega, 0.2).powi(2)));
        for cells in [1, 256, 8193] {
            let vertices = vec![vertex.clone(); cells];
            let mut previous: Option<Array1<Complex<f64>>> = None;
            for threads in [1, 3] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let (actual, unsafe_count) = pool.install(|| {
                    integrate_simplex(&vertices, &array![cells, 1], &frequencies, 0.2, 0.0, 0.0)
                });
                assert_eq!(unsafe_count, 0);
                for (&value, &expected) in actual.iter().zip(&exact) {
                    assert!((value - expected).norm() < 2e-12);
                }
                if let Some(previous) = &previous {
                    assert_eq!(&actual, previous);
                }
                previous = Some(actual);
            }
        }
    }

    #[test]
    fn direct_and_simplex_evaluate_the_same_optical_kernel() {
        let model = qwz_model(-1.0);
        let mut params = Parameters::rank2(
            Conditions {
                t_kelvin: Sampling::Fixed(0.0),
                mu_ev: Sampling::Fixed(0.0),
                omega_ev: Sampling::Values(array![0.2, 0.8]),
            },
            [31, 31],
            [1.0, 0.0],
            [1.0, 0.0],
            ResponseOptions {
                integration: Integration::Direct,
                spin: None,
                field_symmetry: FieldSymmetry::Symmetrized,
                eta_ev: Some(0.1),
            },
        );
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
