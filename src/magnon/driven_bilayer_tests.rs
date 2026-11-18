//! AA-stacked honeycomb bilayer: electronic bonds, not preloaded spin couplings.
use super::{EffectiveSpinModel, PeriodicSpinSite, StrongCouplingOptions};
use crate::magnon::MagnonReference;
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::{arr1, array};
use num_complex::Complex64;
use std::collections::BTreeMap;

// Basis [A1, B1, A2, B2]. Unit in-plane NN distance and unit layer spacing.
const NN: [[isize; 3]; 3] = [[0, 0, 0], [0, 1, 0], [-1, 1, 0]];
const NNN: [[isize; 3]; 3] = [[1, 0, 0], [-1, 1, 0], [0, -1, 0]];
const O: [isize; 3] = [0; 3];
const T: f64 = 0.12;
const TP: f64 = 0.09;
const U: f64 = 4.0;
const W: f64 = 5.0;

type Site = PeriodicSpinSite<3>;

fn aa_bilayer() -> HubbardModel<3> {
    let root3 = 3.0_f64.sqrt();
    let mut electron = Model::<true, 3>::tb_model(
        array![[root3, 0., 0.], [root3 / 2., 1.5, 0.], [0., 0., 4.]],
        array![
            [0., 0., 0.],
            [1. / 3., -2. / 3., 0.],
            [0., 0., 0.25],
            [1. / 3., -2. / 3., 0.25],
        ],
        None,
    )
    .unwrap();
    for i in [0, 2] {
        for r in NN {
            electron.set_hop(T, i, i + 1, &arr1(&r), None);
        }
    }
    for i in [0, 1] {
        electron.set_hop(TP, i, i + 2, &arr1(&O), None);
    }
    HubbardModel::with_uniform_u(electron, U).unwrap()
}

fn site(orbital: usize, cell: [isize; 3]) -> Site {
    Site { orbital, cell }
}

fn pair_key(mut pair: [Site; 2]) -> [Site; 2] {
    pair.sort();
    for a in 0..3 {
        pair[1].cell[a] -= pair[0].cell[a];
    }
    pair[0].cell = O;
    pair
}

fn ring_key(mut pairs: [[Site; 2]; 2]) -> [[Site; 2]; 2] {
    let origin = pairs.iter().flatten().min().unwrap().cell;
    for p in &mut pairs {
        for s in &mut *p {
            for (r, r0) in s.cell.iter_mut().zip(origin) {
                *r -= r0;
            }
        }
        p.sort();
    }
    pairs.sort();
    pairs
}

// The three inequivalent products on the rectangle A1(0), B1(R), B2(R), A2(0).
fn rectangle(r: [isize; 3]) -> [[[Site; 2]; 2]; 3] {
    let [a, b, c, d] = [site(0, O), site(1, r), site(3, r), site(2, O)];
    [[[a, b], [c, d]], [[a, d], [b, c]], [[a, c], [b, d]]]
}

fn assert_close(got: f64, want: f64, label: impl std::fmt::Debug) {
    assert!(
        (got - want).abs() <= 2e-13,
        "{label:?}: {got:e} vs {want:e} eV"
    );
}

#[test]
fn periodic_aa_bilayer_static_matches_closed_exchange_on_every_orbit() {
    let spin = aa_bilayer()
        .floquet_spin_model(
            &FloquetDrive::new(W),
            &StrongCouplingOptions {
                max_hopping_order: 4,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(spin.diagnostics().active_photon_max, 0);

    // Supplemental static AA formulas. Each rectangle contains two wedges
    // for each diagonal; the connected cycle reduces their sum from 8 to 4.
    let t4 = T.powi(4) / U.powi(3);
    let tp4 = TP.powi(4) / U.powi(3);
    let mixed = (T * TP).powi(2) / U.powi(3);
    let j1 = 4. * T.powi(2) / U - 16. * t4 - 4. * mixed;
    let j2 = 4. * t4;
    let jperp = 4. * TP.powi(2) / U - 16. * tp4 - 12. * mixed;
    let jcross = 4. * mixed;
    let k = 80. * mixed;

    let mut pairs = BTreeMap::new();
    let mut quartics = BTreeMap::new();
    for r in NN {
        for (i, j, want) in [(0, 1, j1), (2, 3, j1), (0, 3, jcross), (2, 1, jcross)] {
            assert!(
                pairs
                    .insert(pair_key([site(i, O), site(j, r)]), want)
                    .is_none()
            );
        }
        for (q, want) in rectangle(r).into_iter().zip([k, k, -k]) {
            assert!(quartics.insert(ring_key(q), want).is_none());
        }
    }
    for i in 0..4 {
        for r in NNN {
            assert!(
                pairs
                    .insert(pair_key([site(i, O), site(i, r)]), j2)
                    .is_none()
            );
        }
    }
    for i in 0..2 {
        assert!(
            pairs
                .insert(pair_key([site(i, O), site(i + 2, O)]), jperp)
                .is_none()
        );
    }
    assert_eq!((pairs.len(), quartics.len()), (26, 9));
    // The fully polarized state is Pauli-blocked: each pair has 1/4,
    // each quartic 1/16, and every chirality has zero expectation.
    let constant = -pairs.values().sum::<f64>() / 4. - quartics.values().sum::<f64>() / 16.;
    assert_close(spin.constant_ev(), constant, "cell constant");
    for p in spin.pairs() {
        let key = pair_key([site(p.i, O), site(p.j, p.cell)]);
        assert_close(p.exchange_ev, pairs.remove(&key).unwrap_or(0.), key);
    }
    for q in spin.quartics() {
        let key = ring_key(q.pairs);
        assert_close(q.coefficient_ev, quartics.remove(&key).unwrap_or(0.), key);
    }
    for c in spin.chiralities() {
        assert_close(c.coefficient_ev, 0., c.sites);
    }
    assert!(
        pairs.is_empty() && quartics.is_empty(),
        "missing {pairs:?} / {quartics:?}"
    );
}

#[test]
fn periodic_aa_bilayer_mixed_drive_matches_powered_rectangles() {
    // Re[a exp(-iθ)] = (0.4 cos θ, 0.3 sin θ, 0.22 cos θ + 0.17 sin θ).
    // No bond family is dark. The relative phase in each vertical rectangle
    // differs, so counting all three rectangles with one weight would fail.
    let drive = FloquetDrive::with_modes(
        W,
        vec![LightMode::new(
            1,
            array![
                Complex64::new(0.4, 0.),
                Complex64::new(0., 0.3),
                Complex64::new(0.22, 0.17),
            ],
        )],
    );
    let input = aa_bilayer();
    let run = |order| {
        input
            .floquet_spin_model(
                &drive,
                &StrongCouplingOptions {
                    max_hopping_order: order,
                    hop_harmonic_max: 4,
                    intermediate_photon_max: 8,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let second = run(2);
    let full = run(4);
    assert_eq!(full.diagnostics().active_photon_max, 8);

    // Dense Fock/DFT oracle: scripts/floquet_spin_reference/aa_rectangle_reference.py.
    // All entries (eV) drift <4e-15 from (H,M,grid)=(4,8,128) to (6,12,256).
    // Columns: in-plane H2 J, diagonal H4 J, K1=-K3=K2, χ(A1,B1,B2),
    // and H4[flip_A1,flip_B2] about spin bit pattern 0101 (MSB at A1).
    let references = [
        [
            0.012_635_837_963_673,
            2.241_337_207_429e-5,
            8.985_098_853_137e-5,
            3.266_155_880_757e-7,
            -2.468_752_927_411e-8,
        ],
        [
            0.011_639_375_616_690,
            2.624_794_014_984e-5,
            7.273_930_067_058e-5,
            -2.521_359_763_843e-7,
            4.031_557_491_098e-6,
        ],
        [
            0.011_639_375_616_690,
            2.676_031_146_622e-5,
            7.366_823_310_726e-5,
            -4.461_735_640_361e-7,
            4.171_626_594_700e-6,
        ],
    ];
    let pair_coefficient = |spin: &EffectiveSpinModel<3>, key| {
        spin.pairs()
            .iter()
            .find(|p| pair_key([site(p.i, O), site(p.j, p.cell)]) == key)
            .unwrap_or_else(|| panic!("missing pair {key:?}"))
            .exchange_ev
    };
    for i in 0..2 {
        let key = pair_key([site(i, O), site(i + 2, O)]);
        assert_close(pair_coefficient(&second, key), 0.007_245_264_803_172, key);
    }
    let q = array![0.17, 0.29, 0.37];
    let mut a03 = Complex64::new(0., 0.);
    for (r, [j2_in_plane, jcross, ring, chi, flip]) in NN.into_iter().zip(references) {
        for (i, j) in [(0, 1), (2, 3)] {
            let key = pair_key([site(i, O), site(j, r)]);
            assert_close(pair_coefficient(&second, key), j2_in_plane, key);
        }
        // Both two-bond paths for a diagonal are inside its unique rectangle;
        // no outside site can enter its connected H4 exchange or mixed χ.
        for (i, j) in [(0, 3), (2, 1)] {
            let key = pair_key([site(i, O), site(j, r)]);
            assert_close(pair_coefficient(&full, key), jcross, key);
        }
        for (pairing, want) in rectangle(r).into_iter().zip([ring, ring, -ring]) {
            let key = ring_key(pairing);
            let got = full
                .quartics()
                .iter()
                .find(|x| ring_key(x.pairs) == key)
                .unwrap();
            assert_close(got.coefficient_ev, want, key);
        }
        // Keep the ascending orbital orientation, but shift the first site
        // to the cell origin independently of the production anchor choice.
        let mut matches = full.chiralities().iter().filter(|term| {
            let [a, b, c] = term.sites;
            [a.orbital, b.orbital, c.orbital] == [0, 1, 3]
                && (0..3).all(|d| b.cell[d] - a.cell[d] == r[d] && c.cell[d] - a.cell[d] == r[d])
        });
        let got = matches
            .next()
            .unwrap_or_else(|| panic!("missing mixed χ for {r:?}"));
        assert!(matches.next().is_none());
        assert_close(got.coefficient_ev, chi, r);
        let phase = std::f64::consts::TAU * (q[0] * r[0] as f64 + q[1] * r[1] as f64);
        a03 += Complex64::from_polar(1., phase) * flip;
    }

    // For S=1/2 the one-flip matrix element is the transverse normal block
    // between these parallel sites; no on-site matrix elements enter. A1/B2
    // and B1/A2 are opposite spin labels, with opposite bond displacements.
    // The Sambe flip values on both diagonals are equal and real here.
    let axes = array![[0., 0., 1.], [0., 0., -1.], [0., 0., -1.], [0., 0., 1.]];
    let magnon = full.magnon(&axes, MagnonReference::collinear()).unwrap();
    let bdg = magnon.gen_bdg(&q).unwrap();
    assert!(a03.im.abs() > 5e-6);
    for (ij, want) in [
        ([0, 3], a03),
        ([1, 2], a03.conj()),
        ([4, 7], a03),
        ([5, 6], a03.conj()),
    ] {
        assert_close((bdg[ij] - want).norm(), 0., ij);
    }
    // A1/A2 lie in the same cell. The imaginary pair-creation amplitude
    // from their three adjacent rectangles sees the mixed chirality. A03
    // alone does not: the chirality cancels there between its spectators.
    assert_close(bdg[[0, 6]].im, -1.858_469_761_723_3e-7, "Im B02");
}
