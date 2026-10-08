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
//! let omegas = Array1::linspace(0.0, 4.0, 401);
//!
//! let hall = Parameters {
//!     conditions: Conditions {
//!         t_kelvin: Sampling::Fixed(20.0),
//!         mu_ev: Sampling::Values(chemical_potentials),
//!         omega_ev: Sampling::Fixed(0.0),
//!     },
//!     kmesh: [101, 101],
//!     integration: Integration::EnergyCut,
//! };
//! // Projection directions and the broadening are method arguments.
//! let hall_result = model.hall_conductivity(&hall, [[1.0, 0.0], [0.0, 1.0]], 1e-3, None)?;
//!
//! let optical = Parameters {
//!     conditions: Conditions {
//!         t_kelvin: Sampling::Fixed(0.0),
//!         mu_ev: Sampling::Fixed(0.0),
//!         omega_ev: Sampling::Values(omegas),
//!     },
//!     kmesh: [101, 101],
//!     integration: Integration::Simplex,
//! };
//! let optical_result = model.optical_conductivity(&optical, [[1.0, 0.0], [0.0, 1.0]], 1e-3)?;
//! # let _ = (hall_result, optical_result);
//! # Ok(())
//! # }
//! ```

pub mod config;
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
pub use config::{
    Conditions, FieldSymmetry, Integration, IntegrationDiagnostics, Parameters, ResponseAxis,
    Sampling,
};
pub use linear::HallConductivityResult;
pub use nonlinear::NonlinearHallResult;
pub use optical::OpticalConductivityResult;
pub use traits::{BandBerryCurvature, BerryCurvature};

// Internal numerical machinery shared with crate-level tests and
// `quantum_geometry`; it is deliberately not part of the public API.
pub(crate) use tracking::{global_band_track, permute_vertex};
#[cfg(test)]
pub(crate) use types::VertexKernel;

#[cfg(test)]
pub(crate) use energy_cut::read_reset_fermi_cut_counts;

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::{Model, QuantumGeometry};
    use ndarray::prelude::*;

    /// Broadening shared by the regression parameter sets.
    const ETA_EV: f64 = 0.07;

    /// The two rank-two projection directions of a `DIM`-dimensional model.
    fn dirs2<const DIM: usize>() -> [[f64; DIM]; 2] {
        let mut first = [0.0; DIM];
        first[0] = 1.0;
        let mut second = [0.0; DIM];
        second[if DIM > 1 { 1 } else { 0 }] = 1.0;
        [first, second]
    }

    /// The three rank-three projection directions `(current, field_1, field_2)`.
    ///
    /// The field directions are deliberately mixed so that the regression
    /// responses are nonzero and no two rows coincide: `current = e_0`,
    /// `field_1 = e_1`, `field_2 = e_0 + e_1` (in one dimension, where only a
    /// single axis exists, all three rows collapse onto it as they must).
    fn dirs3<const DIM: usize>() -> [[f64; DIM]; 3] {
        let [current, field_1] = dirs2::<DIM>();
        let mut field_2 = [0.0; DIM];
        for axis in 0..DIM {
            field_2[axis] = current[axis] + field_1[axis];
        }
        [current, field_1, field_2]
    }

    /// Conditions that sample the chemical-potential axis at fixed temperature.
    fn mu_conditions(chemical_potentials: &Array1<f64>, temperature_kelvin: f64) -> Conditions {
        Conditions {
            t_kelvin: Sampling::Fixed(temperature_kelvin),
            mu_ev: Sampling::Values(chemical_potentials.clone()),
            omega_ev: Sampling::Fixed(0.0),
        }
    }

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
        let mut params = Parameters {
            conditions: mu_conditions(&array![-3.0, -2.4, -1.8, -1.4], 0.0),
            kmesh: [5, 6, 7],
            integration: Integration::Direct,
        };
        // This test only compares time-reversal residuals; keep HEAD's 0.05.
        let time_reversal_eta = 0.05;
        // The per-k kernels evaluate one state, so they take all three axes
        // pinned instead of the sampled chemical potential.
        let local = model
            .quantum_geometry_at(&array![0.13, 0.21, 0.17], dirs2(), 0.05)
            .unwrap();
        assert!(local.berry_curvature.iter().any(|x| x.abs() > 1e-4));
        let direct = model
            .hall_conductivity(&params, dirs2(), time_reversal_eta, None)
            .unwrap()
            .conductivity;
        params.integration = Integration::EnergyCut;
        let ec = model
            .hall_conductivity(&params, dirs2(), time_reversal_eta, None)
            .unwrap()
            .conductivity;
        params.integration = Integration::Simplex;
        params.conditions.t_kelvin = Sampling::Fixed(500.0);
        let geometry = model
            .quantum_geometry(&params, dirs2(), time_reversal_eta)
            .unwrap()
            .berry_curvature;
        params.conditions.mu_ev = Sampling::Fixed(-2.4);
        params.conditions.omega_ev = Sampling::Values(array![0.0, 0.7, 2.0]);
        let xy = model
            .optical_conductivity(&params, dirs2(), time_reversal_eta)
            .unwrap()
            .conductivity;
        let yx = model
            .optical_conductivity(
                &params,
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
                time_reversal_eta,
            )
            .unwrap()
            .conductivity;
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
        let mut params = Parameters {
            conditions: mu_conditions(&array![-1.7, -1.3, 0.0, 1.3, 1.7], 0.0),
            kmesh: reference_mesh,
            integration: Integration::Direct,
        };
        let zero = model
            .hall_conductivity(&params, dirs2(), ETA_EV, None)
            .unwrap()
            .conductivity;
        params.conditions.t_kelvin = Sampling::Fixed(600.0);
        let reference = model
            .hall_conductivity(&params, dirs2(), ETA_EV, None)
            .unwrap()
            .conductivity;
        assert!((&zero - &reference).iter().any(|x| x.abs() > 1e-4));
        params.kmesh = mesh;
        params.integration = Integration::EnergyCut;
        let ec = model
            .hall_conductivity(&params, dirs2(), ETA_EV, None)
            .unwrap()
            .conductivity;
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
        let (direct, energies) = model.berry_connection_dipole_onek(&k, &a, &b, &c).unwrap();
        assert!((1e-10..1e-5).contains(&(energies[1] - energies[0])));
        assert!(direct.iter().any(|x| x.abs() > 1e-6));
        let (vertex, _) = model
            .compute_velocity_kernel(&k, &a, &b, Some(&c), crate::Gauge::Atom, None)
            .unwrap();
        let nonlinear = vertex.nonlinear.as_ref().unwrap();
        for n in 0..2 {
            let bands = [vertex.band.to_vec()];
            let g = |matrix: &ndarray::Array2<num_complex::Complex<f64>>| {
                kernel::eval_intrinsic_G_at_lam(n, &bands, &[matrix.clone()], &[1.0], 2)
            };
            let ec = -(2.0 * nonlinear.vdiag[n] * g(&vertex.k_ab)
                - 0.5
                    * (nonlinear.vdiag_a[n] * g(&nonlinear.k_bc)
                        + nonlinear.vdiag_b[n] * g(&nonlinear.k_ac)));
            assert!((direct[n] - ec).abs() < 1e-12 * direct[n].abs().max(1.0));
        }
        assert_eq!(kernel::intrinsic_inverse_gap(1e-10), 0.0);
        assert_eq!(kernel::intrinsic_inverse_gap(-1e-10), 0.0);
    }

    /// The scalar accessors must not panic on a constructible empty result and
    /// must distinguish a `Fixed` point from a one-element sampled series.
    #[test]
    fn single_returns_a_scalar_only_for_a_fixed_point() {
        let fixed = HallConductivityResult {
            axis: ResponseAxis::Fixed,
            conductivity: array![1.5],
        };
        assert_eq!(fixed.single(), Some(1.5));
        let scanned = HallConductivityResult {
            axis: ResponseAxis::ChemicalPotential(array![0.0]),
            conductivity: array![1.5],
        };
        assert_eq!(
            scanned.single(),
            None,
            "a one-element series is still a scan"
        );
        let empty = HallConductivityResult {
            axis: ResponseAxis::Fixed,
            conductivity: Array1::zeros(0),
        };
        assert_eq!(empty.single(), None, "an empty result has no scalar");

        let fixed = NonlinearHallResult {
            axis: ResponseAxis::Fixed,
            conductivity: array![2.5],
            diagnostics: None,
        };
        assert_eq!(fixed.single(), Some(2.5));
        let scanned = NonlinearHallResult {
            axis: ResponseAxis::ChemicalPotential(array![0.0]),
            conductivity: array![2.5],
            diagnostics: None,
        };
        assert_eq!(
            scanned.single(),
            None,
            "a one-element series is still a scan"
        );
        let empty = NonlinearHallResult {
            axis: ResponseAxis::Fixed,
            conductivity: Array1::zeros(0),
            diagnostics: None,
        };
        assert_eq!(empty.single(), None, "an empty result has no scalar");
    }

    /// The spinless extrinsic entry point keeps its runtime rejection: `spin`
    /// is a real argument there (the spinful model supports spin currents), so
    /// a spinless model must refuse it before any k-mesh work.
    #[test]
    fn spinless_extrinsic_rejects_a_spin_current_before_preparation() {
        use crate::response::config::counters;
        let model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        let params = Parameters {
            conditions: mu_conditions(&array![-1.0, 0.0, 1.0], 0.0),
            kmesh: [5, 6],
            integration: Integration::Direct,
        };
        let (eigen, tracking, result) = counters::measure(|| {
            model.extrinsic_nonlinear_hall(
                &params,
                dirs3(),
                ETA_EV,
                Some(crate::SpinDirection::Z),
                FieldSymmetry::Symmetrized,
            )
        });
        assert!(matches!(
            result,
            Err(crate::TbError::SpinNotAllowed(crate::SpinDirection::Z))
        ));
        assert_eq!((eigen, tracking), (0, 0));
    }

    type ExtrinsicActor<const SPIN: bool> = fn(
        &Model<SPIN, 2>,
        &Parameters<2>,
        [[f64; 2]; 3],
        f64,
        Option<crate::SpinDirection>,
        FieldSymmetry,
    ) -> crate::Result<NonlinearHallResult>;

    fn check_extrinsic_field_permutations<const SPIN: bool>(calculate: ExtrinsicActor<SPIN>) {
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
        let mut params = Parameters {
            conditions: mu_conditions(&array![-1.3, -0.5, 0.0, 0.6, 1.4], 0.0),
            kmesh: [9, 10],
            integration: Integration::Direct,
        };
        // `(current, field_1, field_2)`; swapping field_1 and field_2 must leave
        // the symmetrized kernel unchanged.
        let base_directions = [[1.0, 0.3], [0.1, 1.0], [1.0, -0.2]];
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
        let mut nonzero = false;
        for spin in spins {
            // Independent spin-1/2 matrices for the one-orbital spinful model.
            // Keep this oracle separate from Model::build_spin_matrix.
            let spin_matrix = spin.map(|axis| {
                use num_complex::Complex;
                let zero = Complex::new(0.0, 0.0);
                let half = Complex::new(0.5, 0.0);
                let i_half = Complex::new(0.0, 0.5);
                match axis {
                    crate::SpinDirection::X => array![[zero, half], [half, zero]],
                    crate::SpinDirection::Y => array![[zero, -i_half], [i_half, zero]],
                    crate::SpinDirection::Z => array![[half, zero], [zero, -half]],
                }
            });
            for (integration, temperature) in [
                (Integration::Direct, 400.0),
                (Integration::EnergyCut, 0.0),
                (Integration::EnergyCut, 400.0),
            ] {
                params.integration = integration;
                params.conditions.t_kelvin = Sampling::Fixed(temperature);
                let directions = base_directions;
                let ordered = FieldSymmetry::Ordered;
                let first = calculate(&model, &params, directions, ETA_EV, spin, ordered).unwrap();
                nonzero |= first.conductivity.iter().any(|x| x.abs() > 1e-5);
                if integration == Integration::Direct {
                    let scale = 1e-16;
                    let mut scaled_model = model.clone();
                    scaled_model.ham *= num_complex::Complex::new(scale, 0.0);
                    let mut scaled_params = params.clone();
                    if let Sampling::Values(mu_ev) = &mut scaled_params.conditions.mu_ev {
                        *mu_ev *= scale;
                    }
                    // The thermal energy k_B T must shrink with the
                    // Hamiltonian for the homogeneity check below to hold.
                    if let Sampling::Fixed(t_kelvin) = &mut scaled_params.conditions.t_kelvin {
                        *t_kelvin *= scale;
                    }
                    let scaled = calculate(
                        &scaled_model,
                        &scaled_params,
                        directions,
                        ETA_EV * scale,
                        spin,
                        ordered,
                    )
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
                            &Array1::from_vec(directions[0].to_vec()),
                            &Array1::from_vec(directions[1].to_vec()),
                            &Array1::from_vec(directions[2].to_vec()),
                            spin_matrix.as_ref(),
                            ETA_EV,
                        )
                        .unwrap();
                    let width =
                        super::config::occupation_for(params.conditions.t_kelvin.value_at(0))
                            .energy_width()
                            .unwrap();
                    let chemical_potentials = params
                        .conditions
                        .mu_ev
                        .values()
                        .expect("the chemical-potential axis is sampled")
                        .clone();
                    assert_eq!(
                        first.axis,
                        ResponseAxis::ChemicalPotential(chemical_potentials.clone())
                    );
                    for (index, &mu) in chemical_potentials.iter().enumerate() {
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
                let mut swapped_directions = base_directions;
                swapped_directions.swap(1, 2);
                let second =
                    calculate(&model, &params, swapped_directions, ETA_EV, spin, ordered).unwrap();
                let symmetrized_field = FieldSymmetry::Symmetrized;
                let swapped = calculate(
                    &model,
                    &params,
                    swapped_directions,
                    ETA_EV,
                    spin,
                    symmetrized_field,
                )
                .unwrap();
                let symmetrized = calculate(
                    &model,
                    &params,
                    base_directions,
                    ETA_EV,
                    spin,
                    symmetrized_field,
                )
                .unwrap();
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
                // Equal field directions make their exchange a no-op. Keep
                // the current distinct so the charge Berry curvature does not
                // vanish merely because its two directions coincide.
                let equal_directions = [base_directions[0], base_directions[1], base_directions[1]];
                let equal_ordered =
                    calculate(&model, &params, equal_directions, ETA_EV, spin, ordered).unwrap();
                if spin.is_none() {
                    assert!(
                        equal_ordered.conductivity.iter().any(|x| x.abs() > 1e-5),
                        "equal-field charge response must be nonzero: SPIN={SPIN}, {integration:?}, T={temperature}"
                    );
                }
                let equal_symmetrized = calculate(
                    &model,
                    &params,
                    equal_directions,
                    ETA_EV,
                    spin,
                    symmetrized_field,
                )
                .unwrap();
                assert!(
                    (&equal_symmetrized.conductivity - &equal_ordered.conductivity)
                        .iter()
                        .all(|x| x.abs() < 1e-11)
                );

                // Exercise packing through the public entry point for charge
                // and spin currents on both direct and energy-cut paths.
                let mut strided_params = params.clone();
                let strided_mu = array![-1.3, 99.0, -0.5, 99.0, 0.0, 99.0, 0.6, 99.0, 1.4]
                    .slice_move(ndarray::s![..;2]);
                assert!(strided_mu.as_slice().is_none());
                strided_params.conditions.mu_ev = Sampling::Values(strided_mu);
                let strided = calculate(
                    &model,
                    &strided_params,
                    equal_directions,
                    ETA_EV,
                    spin,
                    ordered,
                )
                .unwrap();
                assert_eq!(strided.axis, equal_ordered.axis);
                assert!(
                    (&strided.conductivity - &equal_ordered.conductivity)
                        .iter()
                        .all(|x| x.abs() < 1e-11)
                );
            }
        }
        assert!(nonzero, "the regression must exercise a nonzero response");
    }

    #[test]
    fn extrinsic_symmetrization_reuses_the_ordered_charge_and_spin_kernels() {
        check_extrinsic_field_permutations::<false>(
            |model, params, directions, eta_ev, spin, field_symmetry| {
                model.extrinsic_nonlinear_hall(params, directions, eta_ev, spin, field_symmetry)
            },
        );
        check_extrinsic_field_permutations::<true>(
            |model, params, directions, eta_ev, spin, field_symmetry| {
                model.extrinsic_nonlinear_hall(params, directions, eta_ev, spin, field_symmetry)
            },
        );
    }

    #[test]
    fn response_entry_points_reject_invalid_model_data() {
        let base = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let conditions = Conditions::fixed(0.0, 0.0, 0.0);
        let rank2 = Parameters {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [2, 2],
            integration: Integration::Direct,
        };
        let rank3 = Parameters {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [2, 2],
            integration: Integration::Direct,
        };
        for invalid in 0..3 {
            let mut model = base.clone();
            match invalid {
                0 => model.ham[[0, 0, 0]].re = f64::NAN,
                1 => model.lat.fill(0.0),
                _ => model.ham = ndarray::Array3::zeros((1, 2, 2)),
            }
            assert!(matches!(
                model.hall_conductivity(&rank2, dirs2(), ETA_EV, None),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.quantum_geometry(&rank2, dirs2(), ETA_EV),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.berry_curvature_at(&array![0.0, 0.0], dirs2(), ETA_EV, None,),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.occupied_berry_curvature_at(
                    &array![0.0, 0.0],
                    &conditions,
                    dirs2(),
                    ETA_EV,
                    None,
                ),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.occupied_berry_curvature_on(
                    &Array2::zeros((0, 2)),
                    &conditions,
                    dirs2(),
                    ETA_EV,
                    None,
                ),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.optical_conductivity(&rank2, dirs2(), ETA_EV),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.extrinsic_nonlinear_hall(
                    &rank3,
                    dirs3(),
                    ETA_EV,
                    None,
                    FieldSymmetry::Symmetrized
                ),
                Err(crate::TbError::InvalidModelInvariant { .. })
            ));
            assert!(matches!(
                model.intrinsic_nonlinear_hall(&rank3, dirs3()),
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

    // Every entry point that reads `t_kelvin` validates it at its own
    // boundary. A negative or non-finite temperature must be rejected there
    // instead of silently collapsing to the exact zero-temperature occupation.
    #[test]
    fn response_entry_points_reject_invalid_temperature() {
        let model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let k = array![0.0, 0.0];
        let k_points = array![[0.0, 0.0], [0.5, 0.5]];
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            let mut rank2 = Parameters {
                conditions: Conditions::fixed(0.0, 0.0, 0.0),
                kmesh: [2, 2],
                integration: Integration::Direct,
            };
            rank2.conditions.t_kelvin = Sampling::Fixed(invalid);
            let mut rank3 = Parameters {
                conditions: Conditions::fixed(0.0, 0.0, 0.0),
                kmesh: [2, 2],
                integration: Integration::Direct,
            };
            rank3.conditions.t_kelvin = Sampling::Fixed(invalid);
            for rejected in [
                model
                    .hall_conductivity(&rank2, dirs2(), ETA_EV, None)
                    .map(|_| ()),
                model.quantum_geometry(&rank2, dirs2(), ETA_EV).map(|_| ()),
                model
                    .optical_conductivity(&rank2, dirs2(), ETA_EV)
                    .map(|_| ()),
                model
                    .extrinsic_nonlinear_hall(
                        &rank3,
                        dirs3(),
                        ETA_EV,
                        None,
                        FieldSymmetry::Symmetrized,
                    )
                    .map(|_| ()),
                model.intrinsic_nonlinear_hall(&rank3, dirs3()).map(|_| ()),
                model
                    .occupied_berry_curvature_at(&k, &rank2.conditions, dirs2(), ETA_EV, None)
                    .map(|_| ()),
                model
                    .occupied_berry_curvature_on(
                        &k_points,
                        &rank2.conditions,
                        dirs2(),
                        ETA_EV,
                        None,
                    )
                    .map(|_| ()),
            ] {
                assert!(matches!(
                    rejected,
                    Err(crate::TbError::InvalidResponseParameter {
                        parameter: "t_kelvin",
                        ..
                    })
                ));
            }
        }
        // A legitimate zero-temperature request stays rejected on the direct
        // Fermi-surface path, which samples -df/dE at k-points.
        let cold = Parameters {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: [2, 2],
            integration: Integration::Direct,
        };
        for rejected in [
            model.extrinsic_nonlinear_hall(
                &cold,
                dirs3(),
                ETA_EV,
                None,
                FieldSymmetry::Symmetrized,
            ),
            model.intrinsic_nonlinear_hall(&cold, dirs3()),
        ] {
            assert!(matches!(
                rejected,
                Err(crate::TbError::InvalidThermodynamicParameter { .. })
            ));
        }
    }

    /// Two-band 2D model with non-zero Hall, quantum-geometry, nonlinear and
    /// optical responses, used by the sampled-axis regressions.
    fn sampled_axis_model() -> Model<false, 2> {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
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
        model
    }

    fn assert_close(left: f64, right: f64, context: &str) {
        assert!(
            (left - right).abs() <= 1e-12 * right.abs().max(1.0),
            "{context}: sampled {left} vs single point {right}"
        );
    }

    /// Guard against a vacuous equivalence check: if the sampled axis does not
    /// actually change the response, comparing sample `i` with the single point
    /// at value `i` could not fail.
    fn assert_axis_matters(values: impl Iterator<Item = f64>, context: &str) {
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for value in values {
            low = low.min(value);
            high = high.max(value);
        }
        assert!(
            low.is_finite() && high.is_finite() && high - low > 1e-6,
            "{context}: the sampled axis is empty or does not change the response, so the comparison is vacuous"
        );
    }

    /// Every sample of a scan must equal the single-point calculation at that
    /// value, for every entry point that accepts a sampled axis.
    #[test]
    fn a_sampled_axis_reproduces_single_point_calls() {
        let model = sampled_axis_model();
        let mesh = [7, 8];
        let chemical_potentials = array![-1.2, -0.4, 0.3, 1.1];

        for integration in [Integration::Direct, Integration::EnergyCut] {
            let scan = Parameters {
                conditions: mu_conditions(&chemical_potentials, 200.0),
                kmesh: mesh,
                integration,
            };
            let scanned = model
                .hall_conductivity(&scan, dirs2(), ETA_EV, None)
                .unwrap();
            assert!(matches!(scanned.axis, ResponseAxis::ChemicalPotential(_)));
            assert_eq!(scanned.conductivity.len(), chemical_potentials.len());
            assert_axis_matters(
                scanned.conductivity.iter().copied(),
                "hall chemical-potential scan",
            );
            for (index, &mu) in chemical_potentials.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(200.0, mu, 0.0),
                    kmesh: mesh,
                    integration,
                };
                let expected = model
                    .hall_conductivity(&single, dirs2(), ETA_EV, None)
                    .unwrap();
                assert!(expected.axis.is_fixed());
                assert_close(
                    scanned.conductivity[index],
                    expected.single().unwrap(),
                    "hall conductivity",
                );
            }
        }

        for integration in [Integration::Direct, Integration::Simplex] {
            let scan = Parameters {
                conditions: mu_conditions(&chemical_potentials, 0.0),
                kmesh: mesh,
                integration,
            };
            let scanned = model.quantum_geometry(&scan, dirs2(), ETA_EV).unwrap();
            assert_axis_matters(
                scanned.metric.iter().copied(),
                "quantum-geometry chemical-potential scan",
            );
            for (index, &mu) in chemical_potentials.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(0.0, mu, 0.0),
                    kmesh: mesh,
                    integration,
                };
                let expected = model.quantum_geometry(&single, dirs2(), ETA_EV).unwrap();
                assert_close(scanned.metric[index], expected.metric[0], "quantum metric");
                assert_close(
                    scanned.berry_curvature[index],
                    expected.berry_curvature[0],
                    "occupied Berry curvature",
                );
            }
        }

        for integration in [Integration::Direct, Integration::EnergyCut] {
            let scan = Parameters {
                conditions: mu_conditions(&chemical_potentials, 200.0),
                kmesh: mesh,
                integration,
            };
            let extrinsic = model
                .extrinsic_nonlinear_hall(&scan, dirs3(), ETA_EV, None, FieldSymmetry::Symmetrized)
                .unwrap()
                .conductivity;
            let intrinsic = model
                .intrinsic_nonlinear_hall(&scan, dirs3())
                .unwrap()
                .conductivity;
            assert_axis_matters(
                extrinsic.iter().copied(),
                "extrinsic nonlinear Hall chemical-potential scan",
            );
            assert_axis_matters(
                intrinsic.iter().copied(),
                "intrinsic nonlinear Hall chemical-potential scan",
            );
            for (index, &mu) in chemical_potentials.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(200.0, mu, 0.0),
                    kmesh: mesh,
                    integration,
                };
                let expected = model
                    .extrinsic_nonlinear_hall(
                        &single,
                        dirs3(),
                        ETA_EV,
                        None,
                        FieldSymmetry::Symmetrized,
                    )
                    .unwrap();
                assert!(expected.axis.is_fixed());
                assert_close(
                    extrinsic[index],
                    expected.conductivity[0],
                    "extrinsic nonlinear Hall",
                );
                let expected = model.intrinsic_nonlinear_hall(&single, dirs3()).unwrap();
                assert_close(
                    intrinsic[index],
                    expected.conductivity[0],
                    "intrinsic nonlinear Hall",
                );
            }
        }
    }

    /// A temperature or frequency scan must also equal the single-point calls.
    #[test]
    fn a_sampled_temperature_or_frequency_reproduces_single_point_calls() {
        let model = sampled_axis_model();
        let mesh = [7, 8];
        // A wide window is needed for the occupation, and therefore the Hall
        // response, to actually change: k_B * 6000 K is about 0.5 eV.
        let temperatures = array![500.0, 2000.0, 6000.0];

        for integration in [Integration::Direct, Integration::EnergyCut] {
            // A chemical potential inside the band makes the occupation, and
            // therefore the Hall response, actually depend on temperature.
            let scan = Parameters {
                conditions: Conditions {
                    t_kelvin: Sampling::Values(temperatures.clone()),
                    mu_ev: Sampling::Fixed(-0.5),
                    omega_ev: Sampling::Fixed(0.0),
                },
                kmesh: mesh,
                integration,
            };
            let scanned = model
                .hall_conductivity(&scan, dirs2(), ETA_EV, None)
                .unwrap();
            assert!(matches!(scanned.axis, ResponseAxis::Temperature(_)));
            assert_axis_matters(
                scanned.conductivity.iter().copied(),
                "hall temperature scan",
            );
            for (index, &t_kelvin) in temperatures.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(t_kelvin, -0.5, 0.0),
                    kmesh: mesh,
                    integration,
                };
                assert_close(
                    scanned.conductivity[index],
                    model
                        .hall_conductivity(&single, dirs2(), ETA_EV, None)
                        .unwrap()
                        .single()
                        .unwrap(),
                    "temperature scan",
                );
            }
        }

        let frequencies = array![0.0, 0.7, 1.9];
        for integration in [Integration::Direct, Integration::Simplex] {
            let scan = Parameters {
                conditions: Conditions {
                    t_kelvin: Sampling::Fixed(300.0),
                    mu_ev: Sampling::Fixed(0.0),
                    omega_ev: Sampling::Values(frequencies.clone()),
                },
                kmesh: mesh,
                integration,
            };
            let scanned = model.optical_conductivity(&scan, dirs2(), ETA_EV).unwrap();
            assert!(matches!(scanned.axis, ResponseAxis::Frequency(_)));
            assert_eq!(scanned.conductivity.ncols(), frequencies.len());
            assert_axis_matters(
                scanned.conductivity.iter().map(|value| value.norm()),
                "optical frequency scan",
            );
            for (index, &omega) in frequencies.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(300.0, 0.0, omega),
                    kmesh: mesh,
                    integration,
                };
                let expected = model
                    .optical_conductivity(&single, dirs2(), ETA_EV)
                    .unwrap();
                for component in 0..scanned.conductivity.nrows() {
                    let left = scanned.conductivity[[component, index]];
                    let right = expected.conductivity[[component, 0]];
                    assert!(
                        (left - right).norm() <= 1e-12 * right.norm().max(1.0),
                        "optical conductivity component {component} sample {index}: {left} vs {right}"
                    );
                }
            }
        }
    }

    /// The performance contract: one k-mesh preparation serves every sample, so
    /// diagonalizations and band-tracking passes do not grow with the sample
    /// count. `counters` is thread-local and the measurement runs on one
    /// dedicated thread, so a parallel test cannot contribute to the counts.
    #[test]
    fn a_sampled_axis_shares_the_k_mesh_preparation() {
        use crate::response::config::counters;
        let model = sampled_axis_model();
        let mesh = [6, 7];
        let prepared = mesh[0] * mesh[1];

        let single = Parameters {
            conditions: Conditions::fixed(0.0, 0.0, 0.0),
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let (single_eigen, single_tracking, _) = counters::measure(|| {
            model
                .hall_conductivity(&single, dirs2(), ETA_EV, None)
                .unwrap()
        });
        assert_eq!(
            single_eigen, prepared,
            "the measurement lost the k-mesh preparation"
        );
        assert_eq!(single_tracking, 0, "direct integration tracks no bands");

        let scan = Parameters {
            conditions: mu_conditions(&Array1::linspace(-1.5, 1.5, 16), 0.0),
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let (scan_eigen, _, _) = counters::measure(|| {
            model
                .hall_conductivity(&scan, dirs2(), ETA_EV, None)
                .unwrap()
        });
        assert_eq!(
            scan_eigen, single_eigen,
            "16 chemical potentials must not repeat the k-mesh preparation"
        );

        let temperature_scan = Parameters {
            conditions: Conditions {
                t_kelvin: Sampling::Values(Array1::linspace(50.0, 400.0, 16)),
                mu_ev: Sampling::Fixed(0.0),
                omega_ev: Sampling::Fixed(0.0),
            },
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let (temperature_eigen, _, _) = counters::measure(|| {
            model
                .hall_conductivity(&temperature_scan, dirs2(), ETA_EV, None)
                .unwrap()
        });
        assert_eq!(
            temperature_eigen, single_eigen,
            "16 temperatures must not repeat the k-mesh preparation"
        );

        let cut = Parameters {
            conditions: mu_conditions(&Array1::linspace(-1.5, 1.5, 16), 0.0),
            kmesh: mesh,
            integration: Integration::EnergyCut,
        };
        let (cut_eigen, cut_tracking, _) = counters::measure(|| {
            model
                .hall_conductivity(&cut, dirs2(), ETA_EV, None)
                .unwrap()
        });
        assert_eq!(
            cut_eigen, prepared,
            "energy-cut integration must prepare the vertices once"
        );
        assert_eq!(
            cut_tracking, 1,
            "band tracking must run once for the whole scan"
        );

        // The same contract must hold for the other four entry points and for
        // the full-Cartesian optical tensor.
        let sampled_mu = Array1::linspace(-1.5, 1.5, 16);

        let geometry = Parameters {
            conditions: mu_conditions(&sampled_mu, 0.0),
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let (geometry_eigen, geometry_tracking, _) =
            counters::measure(|| model.quantum_geometry(&geometry, dirs2(), ETA_EV).unwrap());
        assert_eq!(geometry_eigen, prepared, "quantum geometry prepares once");
        assert_eq!(geometry_tracking, 0);

        let geometry_simplex = Parameters {
            conditions: mu_conditions(&sampled_mu, 0.0),
            kmesh: mesh,
            integration: Integration::Simplex,
        };
        let (geometry_simplex_eigen, geometry_simplex_tracking, _) = counters::measure(|| {
            model
                .quantum_geometry(&geometry_simplex, dirs2(), ETA_EV)
                .unwrap()
        });
        assert_eq!(geometry_simplex_eigen, prepared);
        assert_eq!(geometry_simplex_tracking, 1);

        let extrinsic_cut = Parameters {
            conditions: mu_conditions(&sampled_mu, 0.0),
            kmesh: mesh,
            integration: Integration::EnergyCut,
        };
        let (extrinsic_eigen, extrinsic_tracking, _) = counters::measure(|| {
            model
                .extrinsic_nonlinear_hall(
                    &extrinsic_cut,
                    dirs3(),
                    ETA_EV,
                    None,
                    FieldSymmetry::Symmetrized,
                )
                .unwrap()
        });
        assert_eq!(extrinsic_eigen, prepared);
        assert_eq!(
            extrinsic_tracking, 1,
            "the symmetrized second pass must reuse the tracked vertices"
        );

        let intrinsic_direct = Parameters {
            conditions: mu_conditions(&sampled_mu, 200.0),
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let (intrinsic_eigen, intrinsic_tracking, _) = counters::measure(|| {
            model
                .intrinsic_nonlinear_hall(&intrinsic_direct, dirs3())
                .unwrap()
        });
        assert_eq!(intrinsic_eigen, prepared);
        assert_eq!(intrinsic_tracking, 0);

        let intrinsic_cut = Parameters {
            conditions: mu_conditions(&sampled_mu, 0.0),
            kmesh: mesh,
            integration: Integration::EnergyCut,
        };
        let (intrinsic_cut_eigen, intrinsic_cut_tracking, _) = counters::measure(|| {
            model
                .intrinsic_nonlinear_hall(&intrinsic_cut, dirs3())
                .unwrap()
        });
        assert_eq!(intrinsic_cut_eigen, prepared);
        assert_eq!(intrinsic_cut_tracking, 1);

        let optical_scan = Parameters {
            conditions: Conditions {
                t_kelvin: Sampling::Fixed(200.0),
                mu_ev: Sampling::Fixed(-0.5),
                omega_ev: Sampling::Values(Array1::linspace(0.0, 2.0, 16)),
            },
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let (optical_eigen, _, _) = counters::measure(|| {
            model
                .optical_conductivity(&optical_scan, dirs2(), ETA_EV)
                .unwrap()
        });
        assert_eq!(optical_eigen, prepared, "optical prepares once per k-point");

        // The full Cartesian tensor has its own entry point; one component loop
        // wraps the sample loop, and the preparation is still shared.
        let optical_tensor = Parameters::<2> {
            conditions: Conditions {
                t_kelvin: Sampling::Fixed(200.0),
                mu_ev: Sampling::Fixed(-0.5),
                omega_ev: Sampling::Values(Array1::linspace(0.0, 2.0, 16)),
            },
            kmesh: mesh,
            integration: Integration::Simplex,
        };
        let (tensor_eigen, tensor_tracking, _) = counters::measure(|| {
            model
                .optical_conductivity_tensor(&optical_tensor, ETA_EV)
                .unwrap()
        });
        assert_eq!(tensor_eigen, prepared);
        assert_eq!(tensor_tracking, 1);
    }

    /// Reject an unrepresentable Fermi-window peak before preparing any k-point,
    /// including when it occurs after a valid sample in a temperature scan.
    #[test]
    fn unrepresentable_fermi_window_rejects_the_whole_direct_call() {
        use crate::response::config::counters;
        // No hoppings: every response kernel is zero. At E=mu=0 an infinite
        // Fermi window would nevertheless produce 0 * infinity = NaN.
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.ham[[0, 1, 1]].re = 1.0;
        for temperature in [0.0, 1e-320, 3e-320, 1e-310] {
            for t_kelvin in [
                Sampling::Fixed(temperature),
                Sampling::Values(array![300.0, temperature]),
            ] {
                let params = Parameters {
                    conditions: Conditions {
                        t_kelvin,
                        mu_ev: Sampling::Fixed(0.0),
                        omega_ev: Sampling::Fixed(0.0),
                    },
                    kmesh: [3, 4],
                    integration: Integration::Direct,
                };
                for evaluate in [
                    |model: &Model<false, 2>, params: &Parameters<2>| {
                        model.extrinsic_nonlinear_hall(
                            params,
                            dirs3(),
                            ETA_EV,
                            None,
                            FieldSymmetry::Symmetrized,
                        )
                    },
                    |model: &Model<false, 2>, params: &Parameters<2>| {
                        model.intrinsic_nonlinear_hall(params, dirs3())
                    },
                ] {
                    let (eigen, tracking, result) = counters::measure(|| evaluate(&model, &params));
                    assert!(
                        matches!(
                            result,
                            Err(crate::TbError::InvalidThermodynamicParameter {
                                parameter: "t_kelvin",
                                ..
                            })
                        ),
                        "temperature {temperature}: {result:?}"
                    );
                    assert_eq!((eigen, tracking), (0, 0));
                }
            }
        }
    }

    #[test]
    fn finite_fermi_windows_allow_subnormal_thermal_widths() {
        let temperature = 3e-305;
        let width = crate::Occupation::FermiDirac {
            temperature_kelvin: temperature,
        }
        .energy_width()
        .unwrap();
        assert!(width.is_subnormal());
        assert!((1.0 / width).is_infinite());

        // A k-independent, gapped Hamiltonian has zero velocity, including at
        // a chemical potential equal to one band's energy.
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.ham[[0, 1, 1]].re = 1.0;
        for t_kelvin in [
            Sampling::Fixed(temperature),
            Sampling::Values(array![temperature, 300.0]),
        ] {
            let params = Parameters {
                conditions: Conditions {
                    t_kelvin,
                    mu_ev: Sampling::Fixed(0.0),
                    omega_ev: Sampling::Fixed(0.0),
                },
                kmesh: [2, 2],
                integration: Integration::Direct,
            };
            for evaluate in [
                |model: &Model<false, 2>, params: &Parameters<2>| {
                    model.extrinsic_nonlinear_hall(
                        params,
                        dirs3(),
                        ETA_EV,
                        None,
                        FieldSymmetry::Symmetrized,
                    )
                },
                |model: &Model<false, 2>, params: &Parameters<2>| {
                    model.intrinsic_nonlinear_hall(params, dirs3())
                },
            ] {
                let result = evaluate(&model, &params).unwrap();
                assert_eq!(result.conductivity.len(), params.conditions.t_kelvin.len());
                assert!(result.conductivity.iter().all(|&value| value == 0.0));
            }
        }
    }

    /// The temperature axis is newly sampleable, so every entry point that
    /// reads it must reproduce its single-point calls. Energy-cut paths also
    /// accept an exact 0 K sample; direct Fermi-surface paths must not see one.
    #[test]
    fn a_temperature_scan_reproduces_single_point_calls() {
        let model = sampled_axis_model();
        let mesh = [7, 8];
        let positive_temperatures = array![500.0, 2000.0, 6000.0];
        let temperatures_with_zero = array![0.0, 2000.0, 6000.0];
        // The chemical potential sits inside the lower band, so the occupation
        // and therefore the response really change with temperature.
        let sampled = |temperatures: &Array1<f64>| Conditions {
            t_kelvin: Sampling::Values(temperatures.clone()),
            mu_ev: Sampling::Fixed(-0.5),
            omega_ev: Sampling::Fixed(0.0),
        };

        for integration in [Integration::Direct, Integration::Simplex] {
            let scan = Parameters {
                conditions: sampled(&positive_temperatures),
                kmesh: mesh,
                integration,
            };
            let scanned = model.quantum_geometry(&scan, dirs2(), ETA_EV).unwrap();
            assert!(matches!(scanned.axis, ResponseAxis::Temperature(_)));
            assert_axis_matters(
                scanned.metric.iter().copied(),
                "quantum-geometry temperature scan",
            );
            for (index, &t_kelvin) in positive_temperatures.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(t_kelvin, -0.5, 0.0),
                    kmesh: mesh,
                    integration,
                };
                let expected = model.quantum_geometry(&single, dirs2(), ETA_EV).unwrap();
                assert_close(
                    scanned.metric[index],
                    expected.metric[0],
                    "temperature quantum metric",
                );
                assert_close(
                    scanned.berry_curvature[index],
                    expected.berry_curvature[0],
                    "temperature occupied Berry curvature",
                );
            }
        }

        for integration in [Integration::Direct, Integration::EnergyCut] {
            let temperatures = if integration == Integration::Direct {
                &positive_temperatures
            } else {
                &temperatures_with_zero
            };
            let scan = Parameters {
                conditions: sampled(temperatures),
                kmesh: mesh,
                integration,
            };
            let extrinsic = model
                .extrinsic_nonlinear_hall(&scan, dirs3(), ETA_EV, None, FieldSymmetry::Symmetrized)
                .unwrap()
                .conductivity;
            let intrinsic = model
                .intrinsic_nonlinear_hall(&scan, dirs3())
                .unwrap()
                .conductivity;
            assert_axis_matters(
                extrinsic.iter().copied(),
                "extrinsic nonlinear Hall temperature scan",
            );
            assert_axis_matters(
                intrinsic.iter().copied(),
                "intrinsic nonlinear Hall temperature scan",
            );
            for (index, &t_kelvin) in temperatures.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(t_kelvin, -0.5, 0.0),
                    kmesh: mesh,
                    integration,
                };
                assert_close(
                    extrinsic[index],
                    model
                        .extrinsic_nonlinear_hall(
                            &single,
                            dirs3(),
                            ETA_EV,
                            None,
                            FieldSymmetry::Symmetrized,
                        )
                        .unwrap()
                        .conductivity[0],
                    "temperature extrinsic nonlinear Hall",
                );
                assert_close(
                    intrinsic[index],
                    model
                        .intrinsic_nonlinear_hall(&single, dirs3())
                        .unwrap()
                        .conductivity[0],
                    "temperature intrinsic nonlinear Hall",
                );
            }
        }

        // Optical keeps one fixed frequency or chemical potential, because only
        // one axis may be sampled.
        let omega = 0.7;
        for integration in [Integration::Direct, Integration::Simplex] {
            let scan = Parameters {
                conditions: Conditions {
                    t_kelvin: Sampling::Values(positive_temperatures.clone()),
                    mu_ev: Sampling::Fixed(-0.3),
                    omega_ev: Sampling::Fixed(omega),
                },
                kmesh: mesh,
                integration,
            };
            let scanned = model.optical_conductivity(&scan, dirs2(), ETA_EV).unwrap();
            assert_axis_matters(
                scanned.conductivity.iter().map(|value| value.norm()),
                "optical temperature scan",
            );
            let sampled_chemical_potentials = array![-1.0, 0.0, 0.5];
            let mu_scan = Parameters {
                conditions: Conditions {
                    t_kelvin: Sampling::Fixed(200.0),
                    mu_ev: Sampling::Values(sampled_chemical_potentials.clone()),
                    omega_ev: Sampling::Fixed(omega),
                },
                kmesh: mesh,
                integration,
            };
            let scanned_mu = model
                .optical_conductivity(&mu_scan, dirs2(), ETA_EV)
                .unwrap();
            assert_axis_matters(
                scanned_mu.conductivity.iter().map(|value| value.norm()),
                "optical chemical-potential scan",
            );
            for (index, &t_kelvin) in positive_temperatures.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(t_kelvin, -0.3, omega),
                    kmesh: mesh,
                    integration,
                };
                let expected = model
                    .optical_conductivity(&single, dirs2(), ETA_EV)
                    .unwrap();
                for component in 0..scanned.conductivity.nrows() {
                    let left = scanned.conductivity[[component, index]];
                    let right = expected.conductivity[[component, 0]];
                    assert!(
                        (left - right).norm() <= 1e-12 * right.norm().max(1.0),
                        "temperature optical component {component} sample {index}: {left} vs {right}"
                    );
                }
            }
            for (index, &mu) in sampled_chemical_potentials.iter().enumerate() {
                let single = Parameters {
                    conditions: Conditions::fixed(200.0, mu, omega),
                    kmesh: mesh,
                    integration,
                };
                let expected = model
                    .optical_conductivity(&single, dirs2(), ETA_EV)
                    .unwrap();
                for component in 0..scanned_mu.conductivity.nrows() {
                    let left = scanned_mu.conductivity[[component, index]];
                    let right = expected.conductivity[[component, 0]];
                    assert!(
                        (left - right).norm() <= 1e-12 * right.norm().max(1.0),
                        "chemical-potential optical component {component} sample {index}: {left} vs {right}"
                    );
                }
            }
        }
    }

    /// A frequency-independent response must reject a sampled frequency instead
    /// of returning copies of one value labelled as a frequency dependence.
    /// `optical_conductivity` is the one entry point where the axis is physical.
    #[test]
    fn dc_responses_reject_a_sampled_frequency() {
        use crate::response::config::counters;
        let model = sampled_axis_model();
        let mesh = [5, 6];
        let conditions = Conditions {
            t_kelvin: Sampling::Fixed(200.0),
            mu_ev: Sampling::Fixed(-0.5),
            omega_ev: Sampling::Values(array![0.0, 0.7, 1.4]),
        };
        let rank2 = Parameters {
            conditions: conditions.clone(),
            kmesh: mesh,
            integration: Integration::Direct,
        };
        let rank3 = Parameters {
            conditions: conditions.clone(),
            kmesh: mesh,
            integration: Integration::EnergyCut,
        };
        for rejected in [
            model
                .hall_conductivity(&rank2, dirs2(), ETA_EV, None)
                .map(|_| ()),
            model.quantum_geometry(&rank2, dirs2(), ETA_EV).map(|_| ()),
            model
                .extrinsic_nonlinear_hall(&rank3, dirs3(), ETA_EV, None, FieldSymmetry::Symmetrized)
                .map(|_| ()),
            model.intrinsic_nonlinear_hall(&rank3, dirs3()).map(|_| ()),
        ] {
            assert!(matches!(
                rejected,
                Err(crate::TbError::InvalidResponseParameter {
                    parameter: "omega_ev",
                    ..
                })
            ));
        }
        for (label, measured) in [
            (
                "hall_conductivity",
                counters::measure(|| {
                    model
                        .hall_conductivity(&rank2, dirs2(), ETA_EV, None)
                        .map(|_| ())
                }),
            ),
            (
                "quantum_geometry",
                counters::measure(|| model.quantum_geometry(&rank2, dirs2(), ETA_EV).map(|_| ())),
            ),
            (
                "extrinsic_nonlinear_hall",
                counters::measure(|| {
                    model
                        .extrinsic_nonlinear_hall(
                            &rank3,
                            dirs3(),
                            ETA_EV,
                            None,
                            FieldSymmetry::Symmetrized,
                        )
                        .map(|_| ())
                }),
            ),
            (
                "intrinsic_nonlinear_hall",
                counters::measure(|| model.intrinsic_nonlinear_hall(&rank3, dirs3()).map(|_| ())),
            ),
        ] {
            let (eigen, tracking, _) = measured;
            assert_eq!(
                eigen, 0,
                "{label} must reject before the k-mesh preparation"
            );
            assert_eq!(tracking, 0, "{label} must reject before band tracking");
        }

        // The physical frequency axis must keep working.
        let optical = Parameters {
            conditions,
            kmesh: mesh,
            integration: Integration::Simplex,
        };
        let scanned = model
            .optical_conductivity(&optical, dirs2(), ETA_EV)
            .unwrap();
        assert!(matches!(scanned.axis, ResponseAxis::Frequency(_)));
        assert_eq!(scanned.conductivity.ncols(), 3);
    }

    fn assert_parameter_rejected_before_preparation(
        parameter: &'static str,
        evaluate: impl FnOnce() -> crate::Result<()> + Send,
    ) {
        let (eigen, tracking, result) = super::config::counters::measure(evaluate);
        assert!(
            matches!(
                result,
                Err(crate::TbError::InvalidResponseParameter { parameter: found, .. })
                    if found == parameter
            ),
            "expected rejection of {parameter}: {result:?}"
        );
        assert_eq!((eigen, tracking), (0, 0));
    }

    #[test]
    fn dc_responses_reject_fixed_nonzero_frequencies() {
        let model = sampled_axis_model();
        let k = array![0.2, 0.3];
        let k_points = array![[0.2, 0.3]];
        for omega in [-0.7, 0.7, 1e-320] {
            let rank2 = Parameters {
                conditions: Conditions::fixed(300.0, -0.5, omega),
                kmesh: [3, 4],
                integration: Integration::Direct,
            };
            let rank3 = Parameters {
                conditions: rank2.conditions.clone(),
                kmesh: rank2.kmesh,
                integration: Integration::Direct,
            };
            assert_parameter_rejected_before_preparation("omega_ev", || {
                model
                    .hall_conductivity(&rank2, dirs2(), ETA_EV, None)
                    .map(|_| ())
            });
            assert_parameter_rejected_before_preparation("omega_ev", || {
                model.quantum_geometry(&rank2, dirs2(), ETA_EV).map(|_| ())
            });
            assert_parameter_rejected_before_preparation("omega_ev", || {
                model
                    .extrinsic_nonlinear_hall(
                        &rank3,
                        dirs3(),
                        ETA_EV,
                        None,
                        FieldSymmetry::Symmetrized,
                    )
                    .map(|_| ())
            });
            assert_parameter_rejected_before_preparation("omega_ev", || {
                model.intrinsic_nonlinear_hall(&rank3, dirs3()).map(|_| ())
            });
            // The band-resolved per-k kernels take no thermodynamic state, so a
            // nonzero frequency cannot reach them at all; only the
            // occupation-weighted helpers still read the fixed DC point.
            assert_parameter_rejected_before_preparation("omega_ev", || {
                model
                    .occupied_berry_curvature_at(&k, &rank2.conditions, dirs2(), ETA_EV, None)
                    .map(|_| ())
            });
            assert_parameter_rejected_before_preparation("omega_ev", || {
                model
                    .occupied_berry_curvature_on(
                        &k_points,
                        &rank2.conditions,
                        dirs2(),
                        ETA_EV,
                        None,
                    )
                    .map(|_| ())
            });
            // The same fixed nonzero frequency is physical for optical response.
            assert!(model.optical_conductivity(&rank2, dirs2(), ETA_EV).is_ok());
        }
    }

    /// A spin current for a charge-only response is no longer expressible: the
    /// optical, quantum-geometry and intrinsic nonlinear Hall signatures take no
    /// `spin` argument, so these requests are compile errors rather than
    /// runtime rejections. The reachable runtime part of the old contract — the
    /// spinless model's rejection in `hall_conductivity` — is covered by
    /// `spin_hall_and_berry_match_independent_spinless_sectors`.
    #[test]
    fn charge_only_responses_have_no_spin_argument() {
        fn check<const SPIN: bool>() {
            let model =
                Model::<SPIN, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
            let params = Parameters {
                conditions: Conditions::fixed(300.0, -0.5, 0.0),
                kmesh: [3, 4],
                integration: Integration::Direct,
            };
            assert!(model.optical_conductivity(&params, dirs2(), ETA_EV).is_ok());
            assert!(model.quantum_geometry(&params, dirs2(), ETA_EV).is_ok());
            assert!(
                model
                    .quantum_geometry_at(&array![0.2, 0.3], dirs2(), ETA_EV)
                    .is_ok()
            );
            assert!(
                model
                    .quantum_geometry_on(&array![[0.2, 0.3]], dirs2(), ETA_EV)
                    .is_ok()
            );
        }
        check::<false>();
        check::<true>();
    }
}
