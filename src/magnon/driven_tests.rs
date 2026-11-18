use super::{EffectiveSpinModel, StrongCouplingOptions};
use crate::floquet::{FloquetDrive, LightMode};
use crate::magnon::{MagnonReference, MagnonSolve};
use crate::{HubbardModel, Model};
use ndarray::{Array1, Array2, array};
use num_complex::Complex;

fn static_drive(dim: usize, omega: f64) -> FloquetDrive {
    let mut drive = FloquetDrive::new(omega);
    drive.add_mode(LightMode::new(
        1,
        Array1::from_elem(dim, Complex::new(0.0, 0.0)),
    ));
    drive
}

fn two_frequency_drive() -> FloquetDrive {
    let mut drive = FloquetDrive::new(3.0);
    drive.add_mode(LightMode::new(1, array![Complex::new(0.8, 0.0)]));
    drive.add_mode(LightMode::new(2, array![Complex::new(0.0, 0.45)]));
    drive
}

fn one_site<const D: usize>(hopping: Complex<f64>, u: f64, onsite: f64) -> HubbardModel<D> {
    let lat = Array2::eye(D);
    let orb = Array2::zeros((1, D));
    let mut model = Model::<true, D>::tb_model(lat, orb, None).unwrap();
    model.set_hop(onsite, 0, 0, &Array1::zeros(D), None);
    let mut r = Array1::zeros(D);
    r[0] = 1;
    model.set_hop(hopping, 0, 0, &r, None);
    HubbardModel::with_uniform_u(model, u).unwrap()
}

fn options() -> StrongCouplingOptions {
    StrongCouplingOptions::default()
}

fn relative_close(actual: f64, expected: f64, rtol: f64) {
    let scale = actual.abs().max(expected.abs());
    assert!(
        scale > 0.0,
        "relative comparison requires a nonzero reference"
    );
    assert!(
        (actual - expected).abs() <= rtol * scale,
        "{actual:e} != {expected:e} (relative error {:e})",
        (actual - expected).abs() / scale
    );
}

fn only_pair<const DIM: usize>(model: &EffectiveSpinModel<DIM>) -> f64 {
    assert_eq!(model.pairs().len(), 1);
    model.pairs()[0].exchange_ev
}

fn dimer<const DIM: usize>(t: Complex<f64>, u: f64, swap_sites: bool) -> HubbardModel<DIM> {
    // In 1D the physical bond is one length unit, matching the DFT oracle's
    // phase A*cos(theta)+B*sin(2*theta), not half that phase.
    let lat = Array2::eye(DIM) * 2.0;
    let mut positions = Array2::zeros((2, DIM));
    for a in 0..DIM {
        positions[[if swap_sites { 0 } else { 1 }, a]] = 0.5;
    }
    let mut model = Model::<true, DIM>::tb_model(lat, positions, None).unwrap();
    let amplitude = if swap_sites { t.conj() } else { t };
    model.set_hop(amplitude, 0, 1, &Array1::zeros(DIM), None);
    HubbardModel::with_uniform_u(model, u).unwrap()
}

#[test]
fn static_exchange_is_dimension_generic_and_removes_common_scalar_onsite() {
    for dim in [1, 2, 3] {
        let t = Complex::new(0.37, -0.21);
        let u = 5.3;
        let onsite = 1.17;
        let exchange = match dim {
            1 => only_pair(
                &one_site::<1>(t, u, onsite)
                    .floquet_spin_model(&static_drive(1, 3.0), &options())
                    .unwrap(),
            ),
            2 => only_pair(
                &one_site::<2>(t, u, onsite)
                    .floquet_spin_model(&static_drive(2, 3.0), &options())
                    .unwrap(),
            ),
            3 => only_pair(
                &one_site::<3>(t, u, onsite)
                    .floquet_spin_model(&static_drive(3, 3.0), &options())
                    .unwrap(),
            ),
            _ => unreachable!(),
        };
        relative_close(exchange, 4.0 * t.norm_sqr() / u, 1e-12);
    }
}

#[test]
fn tiny_exchange_survives_extreme_energy_scale_and_uniform_rescaling() {
    let t = Complex::new(1.0e8, 0.0);
    let u = 1.0e308;
    let model = one_site::<1>(t, u, 0.0);
    let mut opts = options();
    opts.min_detuning_ev = 1.0;
    let tiny = only_pair(
        &model
            .floquet_spin_model(&static_drive(1, 1.0), &opts)
            .unwrap(),
    );
    relative_close(tiny, 4.0e-292, 1e-12);

    let t0 = 0.23;
    let u0 = 4.0;
    let scale = 1.0e-5;
    let base = one_site::<1>(Complex::new(t0, 0.0), u0, 0.0)
        .floquet_spin_model(&static_drive(1, 3.0), &options())
        .unwrap();
    let mut scaled_options = options();
    scaled_options.min_detuning_ev *= scale;
    let scaled = one_site::<1>(Complex::new(scale * t0, 0.0), scale * u0, 0.0)
        .floquet_spin_model(&static_drive(1, scale * 3.0), &scaled_options)
        .unwrap();
    relative_close(only_pair(&scaled), scale * only_pair(&base), 1e-12);
    relative_close(scaled.constant_ev(), scale * base.constant_ev(), 1e-12);
}

#[test]
fn double_frequency_dimer_matches_independent_oracle_and_site_reindexing() {
    // DFT reference in scripts/floquet_spin_reference/dimer_reference.py.
    // It includes both doublon directions: the one-direction result is 0.90528311,
    // while the correct coefficient is 1.1713956489238915 eV^-1.
    let expected_k = 1.1713956489238915;
    let u = 4.0;
    let t = 0.16;
    let mut opts = options();
    opts.hop_harmonic_max = 20;
    opts.intermediate_photon_max = 10;
    opts.max_hopping_ratio = 0.9;
    let drive = two_frequency_drive();
    let forward = dimer::<1>(Complex::new(t, 0.0), u, false)
        .floquet_spin_model(&drive, &opts)
        .unwrap();
    let reverse = dimer::<1>(Complex::new(t, 0.0), u, true)
        .floquet_spin_model(&drive, &opts)
        .unwrap();
    relative_close(only_pair(&forward), t * t * expected_k, 2e-9);
    relative_close(only_pair(&reverse), only_pair(&forward), 2e-9);
    relative_close(forward.constant_ev(), -only_pair(&forward) / 4.0, 1e-12);
    assert!((only_pair(&forward) / (t * t) - 0.90528311).abs() > 0.1);
}

#[test]
fn hopping_and_intermediate_cutoffs_are_independent_active_limits() {
    let model = dimer::<1>(Complex::new(0.16, 0.0), 4.0, false);
    let drive = two_frequency_drive();
    let run = |h, m| {
        let mut opts = options();
        opts.hop_harmonic_max = h;
        opts.intermediate_photon_max = m;
        opts.max_hopping_ratio = 0.9;
        only_pair(&model.floquet_spin_model(&drive, &opts).unwrap())
    };
    relative_close(run(0, 8), run(8, 0), 1e-12);
    relative_close(run(1, 8), run(8, 1), 1e-12);
    assert!((run(0, 8) - run(1, 8)).abs() > 1e-8);
    assert!((run(1, 8) - run(8, 8)).abs() > 1e-8);
}

#[test]
fn rejects_nonuniform_u_spin_dependent_hopping_onsite_and_resource_limits() {
    let mut two = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], None).unwrap();
    two.set_hop(0.2, 0, 1, &array![0isize], None);
    let nonuniform = HubbardModel::new(two.clone(), array![4.0, 5.0]).unwrap();
    assert!(
        nonuniform
            .floquet_spin_model(&static_drive(1, 3.0), &options())
            .is_err()
    );

    two.ham[[0, 0, 0]] = Complex::new(0.1, 0.0);
    two.ham[[0, 2, 2]] = Complex::new(-0.1, 0.0);
    assert!(
        HubbardModel::with_uniform_u(two, 4.0)
            .unwrap()
            .floquet_spin_model(&static_drive(1, 3.0), &options())
            .is_err()
    );

    let mut onsite_model =
        Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], None).unwrap();
    onsite_model.set_hop(0.1, 1, 1, &array![0isize], None);
    onsite_model.set_hop(0.2, 0, 1, &array![0isize], None);
    assert!(
        HubbardModel::with_uniform_u(onsite_model, 4.0)
            .unwrap()
            .floquet_spin_model(&static_drive(1, 3.0), &options())
            .is_err()
    );

    let hopping_model = one_site::<1>(Complex::new(0.2, 0.0), 4.0, 0.0);
    let mut opts = options();
    opts.max_work = 1;
    assert!(
        hopping_model
            .floquet_spin_model(&static_drive(1, 3.0), &opts)
            .is_err()
    );
    let mut opts = options();
    opts.max_cache_bytes = 1;
    assert!(
        hopping_model
            .floquet_spin_model(&static_drive(1, 3.0), &opts)
            .is_err()
    );
    let mut wrong_dim = FloquetDrive::new(3.0);
    wrong_dim.add_mode(LightMode::new(
        1,
        array![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ));
    assert!(
        hopping_model
            .floquet_spin_model(&wrong_dim, &options())
            .is_err()
    );
}

#[test]
fn zero_hopping_is_well_defined_and_resonance_checks_only_coupled_states() {
    let zero = one_site::<1>(Complex::new(0.0, 0.0), 4.0, 0.0)
        .floquet_spin_model(&static_drive(1, 3.0), &options())
        .unwrap();
    assert!(zero.pairs().is_empty());
    assert_eq!(zero.constant_ev(), 0.0);
    assert_eq!(zero.diagnostics().virtual_terms_visited, 0);

    let model = one_site::<1>(Complex::new(0.1, 0.0), 1.0, 0.0);
    let mut options_u2w = options();
    options_u2w.max_hopping_ratio = 0.9;
    // U=1, W=2 is a bare 2U±W resonance, but not a second-order charge state.
    assert!(
        model
            .floquet_spin_model(&static_drive(1, 2.0), &options_u2w)
            .is_ok()
    );

    let mut reachable = FloquetDrive::new(1.0);
    reachable.add_mode(LightMode::new(1, array![Complex::new(0.2, 0.0)]));
    assert!(model.floquet_spin_model(&reachable, &options_u2w).is_err());
}

#[test]
fn afm_chain_and_bcc_3d_magnons_are_stable_and_match_dispersion() {
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.5]];
    let mut electron = Model::<true, 1>::tb_model(lat, orb, None).unwrap();
    electron.set_hop(0.2, 0, 1, &array![0isize], None);
    electron.set_hop(0.2, 0, 1, &array![-1isize], None);
    let spin = HubbardModel::with_uniform_u(electron, 4.0)
        .unwrap()
        .floquet_spin_model(&static_drive(1, 3.0), &options())
        .unwrap();
    let axes = array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let magnon = spin.magnon(&axes, MagnonReference::collinear()).unwrap();
    assert!(magnon.spin.iter().all(|s| *s == 0.5));
    let onsite = magnon
        .hamR
        .rows()
        .into_iter()
        .position(|r| r.iter().all(|x| *x == 0))
        .unwrap();
    let onsite_block = magnon.ham.index_axis(ndarray::Axis(0), onsite);
    // R=0 also holds physical A-B intracell exchange; only same-site blocks
    // must vanish when no SIA is supplied.
    for site in 0..2 {
        for a in 0..3 {
            for b in 0..3 {
                assert_eq!(onsite_block[[3 * site + a, 3 * site + b]], 0.0);
            }
        }
    }
    let gamma = magnon.magnon_band_onek(&array![0.0]).unwrap();
    assert!(
        gamma.iter().all(|w| w.is_finite() && *w >= -1e-8),
        "{gamma:?}"
    );
    assert!(
        gamma.iter().any(|w| w.abs() < 1e-8),
        "Goldstone expected: {gamma:?}"
    );
    let chain_k = 4.0 * 0.2_f64.powi(2) / 4.0;
    for q in [0.125, 0.25, 0.5] {
        let actual = magnon.magnon_band_onek(&array![q]).unwrap();
        let expected = chain_k * (std::f64::consts::PI * q).sin().abs();
        for value in actual {
            relative_close(value, expected, 1e-12);
        }
    }

    // BCC bipartite graph: eight A->B bonds with R in {0,-1}^3.
    // Independent LSWT result at S=1/2 is 4K*sqrt(1-product cos^2(pi*q_a)).
    let lat3 = Array2::eye(3);
    let orb3 = array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
    let mut electron3 = Model::<true, 3>::tb_model(lat3, orb3, None).unwrap();
    for x in [0isize, -1] {
        for y in [0isize, -1] {
            for z in [0isize, -1] {
                electron3.set_hop(0.2, 0, 1, &array![x, y, z], None);
            }
        }
    }
    let spin3 = HubbardModel::with_uniform_u(electron3, 4.0)
        .unwrap()
        .floquet_spin_model(&static_drive(3, 3.0), &options())
        .unwrap();
    let k = 4.0 * 0.2_f64.powi(2) / 4.0;
    let magnon3 = spin3.magnon(&axes, MagnonReference::collinear()).unwrap();
    for q in [[0.0, 0.0, 0.0], [0.13, 0.21, 0.31], [0.5, 0.0, 0.25]] {
        let actual = magnon3.magnon_band_onek(&array![q[0], q[1], q[2]]).unwrap();
        let product = q
            .iter()
            .map(|v| (std::f64::consts::PI * v).cos().powi(2))
            .product::<f64>();
        let expected = 4.0 * k * (1.0 - product).sqrt();
        assert!(
            actual.iter().all(|w| w.is_finite() && *w >= -1e-8),
            "{actual:?}"
        );
        for value in actual.iter() {
            assert!(
                (value - expected).abs() < 1e-8 * (1.0 + expected),
                "q={q:?}: {value} != {expected}"
            );
        }
    }
}

#[test]
fn magnon_conversion_rejects_noncollinear_axes_and_nonzero_reference_q() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.5, 0.0]];
    let mut electron = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    electron.set_hop(0.2, 0, 1, &array![0isize, 0], None);
    let spin = HubbardModel::with_uniform_u(electron, 4.0)
        .unwrap()
        .floquet_spin_model(&static_drive(2, 3.0), &options())
        .unwrap();
    let noncollinear = array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
    assert!(
        spin.magnon(&noncollinear, MagnonReference::collinear())
            .is_err()
    );
    let mut reference = MagnonReference::<2>::collinear();
    reference.q[0] = 0.1;
    let collinear = array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    assert!(spin.magnon(&collinear, reference).is_err());
}
