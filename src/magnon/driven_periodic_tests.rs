//! Public high-order checks against static strong-coupling limits.
use super::{
    EffectiveSpinModel, MagnonReference, PeriodicSpinSite, SpinChirality, SpinQuartic,
    StrongCouplingOptions,
};
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::array;
use num_complex::Complex;

#[test]
fn third_order_uses_two_photon_jump_between_retained_replicas() {
    // Independent 4096-point Peierls DFT, site-major Fock, 60x60 dense Sambe,
    // and Pauli Kronecker projection (not the Rust cluster kernel).
    let flux: f64 = 0.41;
    let mut electron =
        Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [-0.79], [-1.53]], None).unwrap();
    for (i, j, r) in [(0, 1, 1), (1, 2, 1), (2, 0, -2)] {
        electron.set_hop(
            Complex::from_polar(0.15, flux / 3.0),
            i,
            j,
            &array![r],
            None,
        );
    }
    let input = HubbardModel::with_uniform_u(electron, 4.0).unwrap();
    let drive =
        FloquetDrive::with_modes(3.0, vec![LightMode::new(1, array![Complex::new(0.8, 0.0)])]);
    for (h, expected) in [(1, -0.002356557978679586), (2, -0.002356560020317937)] {
        let spin = input
            .floquet_spin_model(
                &drive,
                &StrongCouplingOptions {
                    hop_harmonic_max: h,
                    intermediate_photon_max: 1,
                    ..options(3)
                },
            )
            .unwrap();
        assert_eq!(spin.chiralities().len(), 1);
        let only = &spin.chiralities()[0];
        assert_eq!(
            only.sites.map(|s| (s.orbital, s.cell)),
            [(0, [-2]), (1, [-1]), (2, [0])]
        );
        assert_close(only.coefficient_ev, expected, 1e-13);
    }
}

#[test]
fn collinear_high_order_terms_feed_signed_landscape_blocks() {
    use crate::magnon::Magnon;
    let electron =
        Model::<true, 1>::tb_model(array![[4.0]], array![[0.0], [0.25], [0.5], [0.75]], None)
            .unwrap();
    let mut spin = HubbardModel::with_uniform_u(electron, 4.0)
        .unwrap()
        .floquet_spin_model(&FloquetDrive::new(3.0), &options(2))
        .unwrap();
    let site = |orbital, r| PeriodicSpinSite { orbital, cell: [r] };
    let axes = array![[0., 0., 1.], [0., 0., 1.], [0., 0., -1.], [0., 0., 1.]];
    let reference = MagnonReference::collinear();
    let assert_blocks = |m: &Magnon<1>, directed: &[(isize, usize, usize, [[f64; 3]; 3])]| {
        for (r, i, j, _) in directed {
            assert!(m.hamR.rows().into_iter().any(|row| row[0] == *r));
            assert_ne!(i, j);
        }
        for (r_index, row) in m.hamR.rows().into_iter().enumerate() {
            let r = row[0];
            for i in 0..m.nspin() {
                for j in 0..m.nspin() {
                    for a in 0..3 {
                        for b in 0..3 {
                            let mut expected = 0.;
                            for &(d, x, y, tensor) in directed {
                                if (r, i, j) == (d, x, y) {
                                    expected += tensor[a][b];
                                }
                                if (r, i, j) == (-d, y, x) {
                                    expected += tensor[b][a];
                                }
                            }
                            assert_eq!(m.ham[[r_index, 3 * i + a, 3 * j + b]], expected);
                        }
                    }
                }
            }
        }
    };
    let e = |s| [[0., s, 0.], [-s, 0., 0.], [0., 0., 0.]];
    let iso = |s| [[s, 0., 0.], [0., s, 0.], [0., 0., s]];
    spin.chiralities.push(SpinChirality {
        sites: [site(0, 0), site(1, 1), site(2, 1)],
        coefficient_ev: 0.016,
    });
    let m = spin.magnon(&axes, reference.clone()).unwrap();
    assert_blocks(
        &m,
        &[
            (1, 0, 1, e(0.001)),
            (0, 1, 2, e(-0.001)),
            (-1, 2, 0, e(-0.001)),
        ],
    );
    // At q=1/4 the two equal-polarization chiral sites have a normal term;
    // mixed polarizations instead have anomalous terms. Do not sort a spectrum.
    let bdg = m.gen_bdg(&array![0.25]).unwrap();
    for (i, j, z) in [
        (0, 1, Complex::new(-0.004, 0.)),
        (1, 6, Complex::new(0., -0.004)),
        (2, 4, Complex::new(0.004, 0.)),
    ] {
        assert!(
            (bdg[[i, j]] - z).norm() < 1e-16,
            "({i},{j}): {} vs {z}",
            bdg[[i, j]]
        );
    }
    spin.chiralities.clear();
    spin.quartics.push(SpinQuartic {
        pairs: [[site(0, 0), site(1, 0)], [site(3, 1), site(2, 2)]],
        coefficient_ev: 0.032,
    });
    let m = spin.magnon(&axes, reference).unwrap();
    assert_blocks(&m, &[(0, 0, 1, iso(0.001)), (1, 3, 2, iso(-0.001))]);
    let bdg = m.gen_bdg(&array![0.25]).unwrap();
    assert_close(bdg[[0, 0]].re, 0.004, 1e-16);
    assert_close(bdg[[0, 1]].re, -0.004, 1e-16);
    assert_close(bdg[[3, 6]].im, 0.004, 1e-16);
}

fn options(order: usize) -> StrongCouplingOptions {
    StrongCouplingOptions {
        max_hopping_order: order,
        ..StrongCouplingOptions::default()
    }
}

fn assert_close(actual: f64, expected: f64, absolute_tolerance: f64) {
    assert!(
        (actual - expected).abs() <= absolute_tolerance,
        "actual {actual:e}, expected {expected:e}, tolerance {absolute_tolerance:e}"
    );
}

fn assert_ferromagnetic_shift_zero<const D: usize>(spin: &EffectiveSpinModel<D>) {
    // With every electron up, the original hopping annihilates the half-filled
    // state. All >0-degree interactions plus the retained constant must give 0.
    let shift = spin.constant_ev()
        + spin
            .pairs()
            .iter()
            .map(|x| x.exchange_ev / 4.0)
            .sum::<f64>()
        + spin
            .quartics()
            .iter()
            .map(|x| x.coefficient_ev / 16.0)
            .sum::<f64>();
    assert_close(shift, 0.0, 2e-14);
}

#[test]
fn public_third_order_preserves_signed_triangle_flux_and_relabeling() {
    let t: f64 = 0.15;
    let u: f64 = 4.0;
    let flux: f64 = 0.41;
    let drive = FloquetDrive::new(3.0);
    // t_{ij} means an electron hop j -> i. Conjugating the product or
    // exchanging the two sites reverses the canonical triangle orientation.
    for (phase, label, sign) in [
        (flux, [0, 1, 2], -1.0),
        (-flux, [0, 1, 2], 1.0),
        (flux, [1, 0, 2], 1.0),
        (0.0, [0, 1, 2], 0.0),
    ] {
        let mut model =
            Model::<true, 1>::tb_model(array![[4.0]], array![[0.0], [0.25], [0.5]], None).unwrap();
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            model.set_hop(
                Complex::from_polar(t, phase / 3.0),
                label[i],
                label[j],
                &array![0],
                None,
            );
        }
        let spin = HubbardModel::with_uniform_u(model, u)
            .unwrap()
            .floquet_spin_model(&drive, &options(3))
            .unwrap();
        assert_eq!(spin.pairs().len(), 3);
        assert_eq!(spin.diagnostics().hopping_order, 3);
        for pair in spin.pairs() {
            assert_close(pair.exchange_ev, 4.0 * t * t / u, 1e-14);
        }
        assert!(spin.quartics().is_empty());
        let total = spin
            .chiralities()
            .iter()
            .map(|term| {
                assert_eq!(term.sites.map(|site| site.orbital), [0, 1, 2]);
                assert_eq!(term.sites.map(|site| site.cell), [[0], [0], [0]]);
                term.coefficient_ev
            })
            .sum::<f64>();
        assert_close(
            total,
            sign * 24.0 * t.powi(3) * flux.sin() / u.powi(2),
            1e-14,
        );
        assert_ferromagnetic_shift_zero(&spin);
    }
}

#[test]
fn public_third_order_retains_subnormal_chirality() {
    let mut electron =
        Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.25], [0.5]], None).unwrap();
    let t = 2.0_f64.powi(-358);
    electron.set_hop(t, 0, 1, &array![0], None);
    electron.set_hop(t, 1, 2, &array![0], None);
    electron.set_hop(Complex::new(0.0, t), 2, 0, &array![0], None);
    let result = HubbardModel::with_uniform_u(electron, 1.0)
        .unwrap()
        .floquet_spin_model(&FloquetDrive::new(3.0), &options(3))
        .unwrap();
    assert_eq!(result.chiralities().len(), 1);
    // t^3 = 2^-1074; Φ = π/2 and U = 1 give exactly -24 subnormal units.
    assert_eq!(result.chiralities()[0].coefficient_ev, -f64::from_bits(24));
}

#[test]
fn public_fourth_order_matches_the_periodic_square_limit() {
    let t: f64 = 0.2;
    let u: f64 = 4.0;
    let t4 = t.powi(4) / u.powi(3);
    let mut model =
        Model::<true, 2>::tb_model(array![[1.0, 0.0], [0.0, 1.0]], array![[0.0, 0.0]], None)
            .unwrap();
    model.set_hop(t, 0, 0, &array![1, 0], None);
    model.set_hop(t, 0, 0, &array![0, 1], None);
    let spin = HubbardModel::with_uniform_u(model, u)
        .unwrap()
        .floquet_spin_model(&FloquetDrive::new(3.0), &options(4))
        .unwrap();
    let mut counts = [0; 3];
    for term in spin.pairs() {
        assert_eq!((term.i, term.j), (0, 0));
        let [x, y] = term.cell;
        let (kind, expected) = match x * x + y * y {
            1 => (0, 4.0 * t * t / u - 24.0 * t4),
            2 => (1, 4.0 * t4),
            4 => (2, 4.0 * t4),
            n => panic!("unexpected pair length squared {n}"),
        };
        counts[kind] += 1;
        assert_close(term.exchange_ev, expected, 2e-14);
    }
    assert_eq!(counts, [2, 2, 2]);
    assert!(
        spin.chiralities()
            .iter()
            .all(|x| x.coefficient_ev.abs() < 1e-14)
    );
    assert_eq!(spin.quartics().len(), 3);
    for term in spin.quartics() {
        let [p, q] = term.pairs;
        let [x, y] = [p[1].cell[0] - p[0].cell[0], p[1].cell[1] - p[0].cell[1]];
        let [z, w] = [q[1].cell[0] - q[0].cell[0], q[1].cell[1] - q[0].cell[1]];
        assert_eq!(x * x + y * y, z * z + w * w);
        let sign = match x * x + y * y {
            1 => 1.0,
            2 => -1.0,
            n => panic!("unexpected plaquette pairing length squared {n}"),
        };
        assert_close(term.coefficient_ev, sign * 80.0 * t4, 2e-14);
    }
    assert_ferromagnetic_shift_zero(&spin);
}

#[test]
fn public_fourth_order_requires_a_complete_photon_window() {
    let mut model = Model::<true, 1>::tb_model(array![[2.0]], array![[0.0], [0.5]], None).unwrap();
    model.set_hop(0.08, 0, 1, &array![0], None);
    let input = HubbardModel::with_uniform_u(model, 4.0).unwrap();
    let drive =
        FloquetDrive::with_modes(3.0, vec![LightMode::new(1, array![Complex::new(0.2, 0.1)])]);
    let mut selected = options(4);
    selected.hop_harmonic_max = 1;
    selected.intermediate_photon_max = 1;
    assert!(
        input
            .floquet_spin_model(&drive, &selected)
            .unwrap_err()
            .to_string()
            .contains("photon")
    );
    selected.intermediate_photon_max = 2;
    let full = input.floquet_spin_model(&drive, &selected).unwrap();
    selected.intermediate_photon_max = 5;
    let larger = input.floquet_spin_model(&drive, &selected).unwrap();
    assert_eq!(full.pairs(), larger.pairs());
    assert_ferromagnetic_shift_zero(&full);
}
