//! # Response functions via simplex quadrature
//!
//! This module provides the full set of linear, nonlinear, and optical
//! response functions for tight‑binding models.  Both **direct k‑mesh
//! summation** and **simplex quadrature** paths are available for most
//! quantities.
//!
//! ## Theoretical background
//!
//! ### Velocity operator
//!
//! The velocity operator in direction $\alpha$ is
//!
//! $$v_\alpha(\mathbf{k}) = \frac{\partial H(\mathbf{k})}{\partial k_\alpha}$$
//!
//! In the band eigenbasis $\lbrace|\psi_{n\mathbf{k}}\rangle\rbrace$, its matrix elements are
//!
//! $$v^\alpha_{nm}(\mathbf{k}) = \langle\psi_{n\mathbf{k}}|
//!     \partial_\alpha H(\mathbf{k}) |\psi_{m\mathbf{k}}\rangle$$
//!
//! Diagonal elements $v^\alpha_{nn} = \partial E_n/\partial k_\alpha$ give the
//! band velocity; off‑diagonal elements enter response formulas.
//!
//! ### Gauge‑invariant velocity kernel
//!
//! Individual matrix elements $v^\alpha_{nm}$ are not smooth across the
//! Brillouin zone because eigenstates carry arbitrary U(1) phases.  The
//! product
//!
//! ```math
//! K^{ab}_{nm}(\mathbf{k}) \equiv v^a_{nm}(\mathbf{k})v^b_{mn}(\mathbf{k})
//! ```
//!
//! is invariant under independent phase rotations of bands $n$ and $m$
//! (when the bands are isolated), making it safe to interpolate inside
//! each simplex.
//!
//! ### Simplex quadrature (vs. old Blochl method)
//!
//! The **simplex path** interpolates the gauge‑invariant primitives
//! $K_{nm}(\mathbf{k})$ and $E_n(\mathbf{k})$ linearly inside each
//! simplex (triangle in 2D, tetrahedron in 3D), then evaluates the
//! singular denominator at symmetric quadrature points:
//!
//! $$\int_{\text{simplex}} f(K(\mathbf{k}), E(\mathbf{k}))d\mathbf{k}
//!   \approx V_{\text{simplex}} \sum_q w_qf(K(\mathbf{k}_q), E(\mathbf{k}_q))$$
//!
//! This correctly preserves the $1/(E_n-E_m)^p$ singularity structure
//! near small gaps, unlike the old Blochl method which linearly
//! interpolated the final scalar integrand at simplex vertices.
//!
//! ### Band tracking
//!
//! Interpolating $K_{nm}$ requires consistent band labels across simplex
//! vertices.  Energy ordering fails near crossings.  We use eigenvector
//! overlap maximisation: for each vertex $r$, find the permutation $P_r$
//! maximising $\sum_n |\langle u_n(\text{ref})|u_{P_r(n)}(r)\rangle|^2$.
//!
//! ## Sub‑modules
//!
//! | Module | Quantity | Formula |
//! |--------|----------|---------|
//! | [`linear`]   | Berry curvature $\Omega^{ab}$, quantum metric $g^{ab}$ | $\sum_n \int \Omega_n^{ab}d\mathbf{k}$ |
//! | [`nonlinear`]| Berry dipole $D^{ab;c}$, intrinsic/extrinsic NLH | $\sum_n \int (-\partial f/\partial E_n) v^c_n \Omega_n^{ab}d\mathbf{k}$ |
//! | [`optical`]  | Optical conductivity $\sigma^{ab}(\omega)$ | $\sum_{n\ne m} \int \frac{(f_n-f_m)K^{ab}_{nm}}{(E_n-E_m)^2-(\omega+i\eta)^2}d\mathbf{k}$ |
//! | energy cut | 2D Berry dipole line cuts | $\int A_n(k)\delta(E_n-\mu)d^2k$ with finite-T convolution |
//! | [`traits`]   | `BerryCurvature` trait (per‑k‑point Berry curvature) | |
//!
//! ## Quick start
//!
//! ```no_run
//! use ndarray::Array1;
//! use Rustb::*;
//!
//! # fn calculate(model: &Model<false, 2>) -> Result<()> {
//! let chemical_potentials = Array1::linspace(-1.0, 1.0, 201);
//!
//! let mut hall = Parameters::rank2([101, 101], [1.0, 0.0], [0.0, 1.0], chemical_potentials)
//!     .with_temperature(20.0);
//! hall.integration = Integration::EnergyCut;
//! let hall_result = model.hall_conductivity(&hall)?;
//!
//! let mut optical = Parameters::rank2([101, 101], [1.0, 0.0], [0.0, 1.0], Array1::zeros(1));
//! optical.omega = Array1::linspace(0.0, 4.0, 401);
//! optical.integration = Integration::Simplex;
//! let optical_result = model.optical_conductivity(&optical)?;
//! # let _ = (hall_result, optical_result);
//! # Ok(())
//! # }
//! ```

pub mod config;
mod helpers;
pub mod linear;
pub mod nonlinear;
pub mod optical;
pub mod traits;

mod energy_cut;
mod kernel;
mod primitives;
mod quadrature;
mod tracking;
mod types;

// Stable high-level response API.
pub use config::{FieldSymmetry, Integration, IntegrationDiagnostics, Parameters};
pub use linear::HallConductivityResult;
pub use nonlinear::NonlinearHallResult;
pub use optical::OpticalConductivityResult;
pub use traits::{BandBerryCurvature, BerryCurvature};

// Internal numerical machinery shared with crate-level tests and
// `quantum_geometry`; it is deliberately not part of the public API.
pub(crate) use tracking::global_band_track;
pub(crate) use types::VertexKernel;

#[cfg(test)]
pub(crate) use energy_cut::read_reset_fermi_cut_counts;

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::{Model, QuantumGeometry};
    use ndarray::prelude::*;

    // Real hoppings enforce spinless time reversal. Mixed-axis harmonics
    // deliberately avoid reflection symmetries that could conceal a biased
    // tetrahedralization. Berry curvature is nonzero locally and odd in k.
    fn time_reversal_model() -> Model<false, 3> {
        let mut model = Model::tb_model(Array2::eye(3), Array2::zeros((2, 3)), None).unwrap();
        model.set_onsite(&array![2.5, -2.5], None);
        for (r, amplitude) in [(array![1, 0, 0], 1.0), (array![0, 1, 1], 0.3)] {
            model.add_hop(amplitude / 2.0, 0, 1, &r, None);
            model.add_hop(amplitude / 2.0, 0, 1, &(-&r), None);
        }
        for (r, amplitude) in [(array![0, 1, 0], 1.0), (array![1, 0, 1], 0.2)] {
            model.add_hop(-amplitude / 2.0, 0, 1, &r, None);
            model.add_hop(amplitude / 2.0, 0, 1, &(-&r), None);
        }
        for (r, amplitude) in [(array![0, 0, 1], 1.0), (array![1, 1, 0], 0.3)] {
            model.add_hop(amplitude / 2.0, 0, 0, &r, None);
            model.add_hop(-amplitude / 2.0, 1, 1, &r, None);
        }
        model
    }

    #[test]
    fn three_dimensional_integrals_preserve_time_reversal() {
        let model = time_reversal_model();
        let mut params = Parameters::rank2(
            [5, 6, 7],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            array![-3.0, -2.4, -1.8, -1.4],
        );
        params.eta = 0.05;
        let local = model
            .quantum_geometry_at(&array![0.13, 0.21, 0.17], &params)
            .unwrap();
        assert!(local.berry_curvature.iter().any(|x| x.abs() > 1e-4));
        let direct = model.hall_conductivity(&params).unwrap().conductivity;
        params.integration = Integration::EnergyCut;
        let ec = model.hall_conductivity(&params).unwrap().conductivity;
        params.integration = Integration::Simplex;
        params.T = 500.0;
        let geometry = model.quantum_geometry(&params).unwrap().berry_curvature;
        params.mu = array![-2.4];
        params.omega = array![0.0, 0.7, 2.0];
        let xy = model.optical_conductivity(&params).unwrap().conductivity;
        params.direction = array![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
        let yx = model.optical_conductivity(&params).unwrap().conductivity;
        let optical = (&xy - &yx).iter().map(|z| z.norm()).fold(0.0_f64, f64::max);
        let direct = direct.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let ec = ec.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let geometry = geometry.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(
            direct < 1e-12 && ec < 1e-12 && geometry < 1e-12 && optical < 1e-12,
            "TR residuals: direct={direct:e}, energy-cut={ec:e}, geometry={geometry:e}, optical={optical:e}"
        );
    }

    fn finite_temperature_hall<const DIM: usize>(mesh: [usize; DIM], reference_mesh: [usize; DIM]) {
        // Gapped QWZ model, with flat extra dimensions. Unlike a TR model its
        // Hall signal is nonzero, including partially occupied bands.
        let mut model =
            Model::<false, DIM>::tb_model(Array2::eye(DIM), Array2::zeros((2, DIM)), None).unwrap();
        model.set_onsite(&array![-1.0, 1.0], None);
        let mut x = Array1::zeros(DIM);
        x[0] = 1;
        let mut y = Array1::zeros(DIM);
        y[1] = 1;
        model.add_hop(num_complex::Complex::new(0.0, -0.5), 0, 1, &x, None);
        model.add_hop(num_complex::Complex::new(0.0, 0.5), 0, 1, &(-&x), None);
        model.add_hop(-0.5, 0, 1, &y, None);
        model.add_hop(0.5, 0, 1, &(-&y), None);
        for r in [x, y] {
            model.add_hop(0.5, 0, 0, &r, None);
            model.add_hop(-0.5, 1, 1, &r, None);
        }
        let mut a = [0.0; DIM];
        a[0] = 1.0;
        let mut b = [0.0; DIM];
        b[1] = 1.0;
        let mut params = Parameters::rank2(reference_mesh, a, b, array![-1.7, -1.3, 0.0, 1.3, 1.7]);
        params.eta = 0.05;
        let zero = model.hall_conductivity(&params).unwrap().conductivity;
        params.T = 600.0;
        let reference = model.hall_conductivity(&params).unwrap().conductivity;
        assert!((&zero - &reference).iter().any(|x| x.abs() > 1e-4));
        params.kmesh = mesh;
        params.integration = Integration::EnergyCut;
        let ec = model.hall_conductivity(&params).unwrap().conductivity;
        let error = (&ec - &reference)
            .iter()
            .map(|x| x.abs())
            .fold(0.0_f64, f64::max);
        assert!(
            error < 0.005,
            "{DIM}D finite-T EC error={error:e}, EC={ec:?}, direct={reference:?}"
        );
    }

    #[test]
    fn finite_temperature_energy_cut_matches_direct_in_2d_and_3d() {
        finite_temperature_hall([25, 25], [81, 81]);
        finite_temperature_hall([25, 25, 2], [81, 81, 1]);
    }

    #[test]
    fn intrinsic_near_gap_kernel_agrees_between_integration_paths() {
        let mut model = time_reversal_model();
        // Put all interband gaps in the interval the old direct path silently
        // discarded, while retaining a nonzero intrinsic kernel.
        model.ham *= num_complex::Complex::new(1e-7, 0.0);
        let (a, b, c) = (
            array![1.0, 0.0, 0.0],
            array![0.0, 1.0, 0.0],
            array![0.0, 0.0, 1.0],
        );
        let k = array![0.13, 0.21, 0.17];
        let (direct, energies, _) = model
            .berry_connection_dipole_onek(&k, &a, &b, &c, None)
            .unwrap();
        assert!((1e-10..1e-5).contains(&(energies[1] - energies[0])));
        assert!(direct.iter().any(|x| x.abs() > 1e-6));
        let vertex = model
            .compute_velocity_kernel(&k, &a, &b, Some(&c), crate::Gauge::Atom, None)
            .unwrap();
        for n in 0..2 {
            let bands = [vertex.band.to_vec()];
            let g = |matrix: &ndarray::Array2<num_complex::Complex<f64>>| {
                kernel::eval_intrinsic_G_at_lam(n, &bands, &[matrix.clone()], &[1.0], 2)
            };
            let ec = -(2.0 * vertex.vdiag.as_ref().unwrap()[n] * g(&vertex.k_ab)
                - 0.5
                    * (vertex.vdiag_a.as_ref().unwrap()[n] * g(vertex.k_bc.as_ref().unwrap())
                        + vertex.vdiag_b.as_ref().unwrap()[n] * g(vertex.k_ac.as_ref().unwrap())));
            assert!((direct[n] - ec).abs() < 1e-12 * direct[n].abs().max(1.0));
        }
        assert_eq!(kernel::intrinsic_inverse_gap(1e-10), 0.0);
        assert_eq!(kernel::intrinsic_inverse_gap(-1e-10), 0.0);
    }

    #[test]
    fn intrinsic_spin_request_returns_a_structured_error() {
        let model = Model::<true, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let mut params = Parameters::rank3([2, 2], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], array![0.0]);
        params.spin = Some(crate::SpinDirection::Z);
        for integration in [Integration::Direct, Integration::EnergyCut] {
            params.integration = integration;
            assert!(matches!(
                model.intrinsic_nonlinear_hall(&params),
                Err(crate::TbError::InvalidResponseParameter {
                    parameter: "spin",
                    ..
                })
            ));
        }
    }

    fn check_extrinsic_field_permutations<const SPIN: bool>() {
        use crate::thermodynamics::fermi_derivative_from_width;
        let mut model = Model::<SPIN, 2>::tb_model(
            Array2::eye(2),
            Array2::zeros((if SPIN { 1 } else { 2 }, 2)),
            None,
        )
        .unwrap();
        model.ham[[0, 0, 0]].re = 0.7;
        model.ham[[0, 1, 1]].re = -0.7;
        for (r, amplitude) in [
            (array![1, 0], num_complex::Complex::new(0.4, -0.3)),
            (array![0, 1], num_complex::Complex::new(0.2, 0.5)),
            (array![1, 1], num_complex::Complex::new(0.15, 0.2)),
        ] {
            model.add_element(amplitude, 0, 1, &r).unwrap();
            model.add_element(amplitude * 0.2, 0, 0, &r).unwrap();
        }
        let mut params = Parameters::rank3(
            [9, 10],
            [1.0, 0.3],
            [0.1, 1.0],
            [1.0, -0.2],
            array![-1.3, -0.5, 0.0, 0.6, 1.4],
        );
        params.eta = 0.07;
        let spins = if SPIN {
            vec![
                None,
                Some(crate::SpinDirection::X),
                Some(crate::SpinDirection::Y),
                Some(crate::SpinDirection::Z),
            ]
        } else {
            vec![None]
        };
        let original_directions = params.direction.clone();
        let mut nonzero = false;
        for spin in spins {
            params.spin = spin;
            for (integration, temperature) in [
                (Integration::Direct, 400.0),
                (Integration::EnergyCut, 0.0),
                (Integration::EnergyCut, 400.0),
            ] {
                params.integration = integration;
                params.T = temperature;
                params.direction = original_directions.clone();
                params.field_symmetry = FieldSymmetry::Ordered;
                let first = model.extrinsic_nonlinear_hall(&params).unwrap();
                nonzero |= first.conductivity.iter().any(|x| x.abs() > 1e-5);
                if integration == Integration::Direct {
                    let scale = 1e-16;
                    let mut scaled_model = model.clone();
                    scaled_model.ham *= num_complex::Complex::new(scale, 0.0);
                    let mut scaled_params = params.clone();
                    scaled_params.mu *= scale;
                    // The thermal energy k_B T must shrink with the
                    // Hamiltonian for the homogeneity check below to hold.
                    scaled_params.T *= scale;
                    scaled_params.eta *= scale;
                    let scaled = scaled_model
                        .extrinsic_nonlinear_hall(&scaled_params)
                        .unwrap();
                    assert!(
                        (&scaled.conductivity - &first.conductivity)
                            .iter()
                            .all(|x| x.abs() < 1e-11),
                        "rescaling all energy scales must not introduce an extrinsic gap cutoff"
                    );
                    // Independent pre-refactor per-k implementation, including
                    // spin-current dressing and the Fermi-window sign.
                    let mesh = crate::kpoints::gen_kmesh::<f64>(&array![9, 10]).unwrap();
                    let (kernel, energies) = model
                        .berry_curvature_dipole_n(
                            &mesh,
                            &params.direction.row(0).to_owned(),
                            &params.direction.row(1).to_owned(),
                            &params.direction.row(2).to_owned(),
                            spin,
                            params.eta,
                        )
                        .unwrap();
                    let width = super::config::parameters_occupation(&params)
                        .energy_width()
                        .unwrap();
                    for (index, &mu) in params.mu.iter().enumerate() {
                        let reference = kernel
                            .iter()
                            .zip(&energies)
                            .map(|(&value, &energy)| {
                                value * fermi_derivative_from_width(energy, mu, width)
                            })
                            .sum::<f64>()
                            / mesh.nrows() as f64;
                        assert!((reference - first.conductivity[index]).abs() < 1e-12);
                    }
                }
                params
                    .direction
                    .row_mut(1)
                    .assign(&original_directions.row(2));
                params
                    .direction
                    .row_mut(2)
                    .assign(&original_directions.row(1));
                let second = model.extrinsic_nonlinear_hall(&params).unwrap();
                params.field_symmetry = FieldSymmetry::Symmetrized;
                let swapped = model.extrinsic_nonlinear_hall(&params).unwrap();
                params.direction = original_directions.clone();
                let symmetrized = model.extrinsic_nonlinear_hall(&params).unwrap();
                assert_eq!(symmetrized.diagnostics, first.diagnostics);
                let expected = (&first.conductivity + &second.conductivity) * 0.5;
                for ((&actual, &swapped), &expected) in symmetrized
                    .conductivity
                    .iter()
                    .zip(&swapped.conductivity)
                    .zip(&expected)
                {
                    assert!((actual - expected).abs() < 1e-11);
                    assert!((actual - swapped).abs() < 1e-11);
                }
                params
                    .direction
                    .row_mut(2)
                    .assign(&original_directions.row(1));
                let equal_fields = model.extrinsic_nonlinear_hall(&params).unwrap();
                params.field_symmetry = FieldSymmetry::Ordered;
                let ordered = model.extrinsic_nonlinear_hall(&params).unwrap();
                let mut strided_params = params.clone();
                strided_params.mu = array![-1.3, 99.0, -0.5, 99.0, 0.0, 99.0, 0.6, 99.0, 1.4]
                    .slice_move(ndarray::s![..;2]);
                assert!(strided_params.mu.as_slice().is_none());
                let strided = model.extrinsic_nonlinear_hall(&strided_params).unwrap();
                assert!(
                    (&strided.conductivity - &ordered.conductivity)
                        .iter()
                        .all(|x| x.abs() < 1e-11)
                );
                assert!(
                    (&equal_fields.conductivity - &ordered.conductivity)
                        .iter()
                        .all(|x| x.abs() < 1e-11)
                );
            }
        }
        assert!(nonzero, "the regression must exercise a nonzero response");
    }

    #[test]
    fn extrinsic_symmetrization_reuses_the_ordered_charge_and_spin_kernels() {
        check_extrinsic_field_permutations::<false>();
        check_extrinsic_field_permutations::<true>();
    }

    #[test]
    fn response_entry_points_reject_invalid_model_data() {
        let base = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let rank2 = Parameters::rank2([2, 2], [1.0, 0.0], [0.0, 1.0], array![0.0]);
        let rank3 = Parameters::rank3([2, 2], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], array![0.0]);
        for invalid in 0..3 {
            let mut model = base.clone();
            match invalid {
                0 => model.ham[[0, 0, 0]].re = f64::NAN,
                1 => model.lat.fill(0.0),
                _ => model.ham = ndarray::Array3::zeros((1, 2, 2)),
            }
            assert!(matches!(
                model.hall_conductivity(&rank2),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.quantum_geometry(&rank2),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.berry_curvature_at(&array![0.0, 0.0], &rank2),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.occupied_berry_curvature_at(&array![0.0, 0.0], &rank2),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.occupied_berry_curvature_on(&Array2::zeros((0, 2)), &rank2),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.optical_conductivity(&rank2),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.extrinsic_nonlinear_hall(&rank3),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.intrinsic_nonlinear_hall(&rank3),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.dos(&array![2, 2], -1.0, 1.0, 3, 0.1),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
        }
        for (lower, upper) in [(f64::NAN, 1.0), (-1.0, f64::INFINITY)] {
            assert!(matches!(
                base.dos(&array![2, 2], lower, upper, 3, 0.1),
                Err(crate::TbError::InvalidEnergyRange { .. })
            ));
        }
    }

    // Every entry point that reads `T` validates it at its own boundary. A
    // negative or non-finite temperature must be rejected there instead of
    // silently collapsing to the exact zero-temperature occupation.
    #[test]
    fn response_entry_points_reject_invalid_temperature() {
        let model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let k = array![0.0, 0.0];
        let k_points = array![[0.0, 0.0], [0.5, 0.5]];
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let mut rank2 = Parameters::rank2([2, 2], [1.0, 0.0], [0.0, 1.0], array![0.0]);
            rank2.T = invalid;
            let mut rank3 =
                Parameters::rank3([2, 2], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], array![0.0]);
            rank3.T = invalid;
            for rejected in [
                model.hall_conductivity(&rank2).map(|_| ()),
                model.quantum_geometry(&rank2).map(|_| ()),
                model.optical_conductivity(&rank2).map(|_| ()),
                model.extrinsic_nonlinear_hall(&rank3).map(|_| ()),
                model.intrinsic_nonlinear_hall(&rank3).map(|_| ()),
                model.occupied_berry_curvature_at(&k, &rank2).map(|_| ()),
                model
                    .occupied_berry_curvature_on(&k_points, &rank2)
                    .map(|_| ()),
            ] {
                assert!(matches!(
                    rejected,
                    Err(crate::TbError::InvalidResponseParameter { parameter: "T", .. })
                ));
            }
        }
        // A legitimate zero-temperature request stays rejected on the direct
        // Fermi-surface path, which samples -df/dE at k-points.
        let cold = Parameters::rank3([2, 2], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], array![0.0]);
        for rejected in [
            model.extrinsic_nonlinear_hall(&cold),
            model.intrinsic_nonlinear_hall(&cold),
        ] {
            assert!(matches!(
                rejected,
                Err(crate::TbError::InvalidThermodynamicParameter { .. })
            ));
        }
    }
}
