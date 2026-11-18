//! Coefficient-level covariance under doubling the electronic unit cell.

use super::{EffectiveSpinModel, PeriodicSpinSite, StrongCouplingOptions};
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::array;
use num_complex::Complex;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Term {
    Pair([isize; 2]),
    Chirality([isize; 3]),
    Quartic([[isize; 2]; 2]),
}

fn primitive_coefficients(spin: &EffectiveSpinModel<1>, copies: usize) -> Vec<BTreeMap<Term, f64>> {
    // There is one orbital at the primitive origin. In the doubled model
    // orbital j is at primitive cell j, hence its image is 2R+j, not R+j.
    let image = |site: PeriodicSpinSite<1>| copies as isize * site.cell[0] + site.orbital as isize;
    let mut coeffs = vec![BTreeMap::new(); copies];
    let mut record = |origin: isize, term, coefficient| {
        let parity = origin.rem_euclid(copies as isize) as usize;
        assert!(coeffs[parity].insert(term, coefficient).is_none());
    };
    for p in spin.pairs() {
        let mut sites = [
            p.i as isize,
            image(PeriodicSpinSite {
                orbital: p.j,
                cell: p.cell,
            }),
        ];
        sites.sort_unstable();
        record(
            sites[0],
            Term::Pair([0, sites[1] - sites[0]]),
            p.exchange_ev,
        );
    }
    for chi in spin.chiralities() {
        let mut sites = chi.sites.map(image);
        let [a, b, c] = sites;
        // An odd permutation changes the sign of S_a · (S_b × S_c).
        let sign = if (a > b) ^ (a > c) ^ (b > c) {
            -1.0
        } else {
            1.0
        };
        sites.sort_unstable();
        let origin = sites[0];
        record(
            origin,
            Term::Chirality(sites.map(|r| r - origin)),
            sign * chi.coefficient_ev,
        );
    }
    for q in spin.quartics() {
        let mut pairs = q.pairs.map(|pair| pair.map(image));
        // Translate the full four-site operator together, not each dot product.
        let origin = *pairs.iter().flatten().min().unwrap();
        for pair in &mut pairs {
            *pair = pair.map(|r| r - origin);
            pair.sort_unstable();
        }
        pairs.sort_unstable();
        record(origin, Term::Quartic(pairs), q.coefficient_ev);
    }
    coeffs
}

#[test]
fn driven_fourth_order_is_covariant_under_cell_doubling() {
    let mut electron = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
    // Next-nearest hopping closes flux triangles (0,1,2) and four-site
    // loops (0,1,3,2), including terms crossing the doubled-cell boundary.
    electron.set_hop(0.13, 0, 0, &array![1], None);
    electron.set_hop(Complex::from_polar(0.09, 0.51), 0, 0, &array![2], None);
    let doubled = electron.make_supercell(&array![[2.0]]).unwrap();
    assert_eq!(doubled.lat, array![[2.0]]);
    assert_eq!(doubled.orb, array![[0.0], [0.5]]);

    // U=4 and W=5 give nonzero detunings for both one and two doublons.
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![LightMode::new(1, array![Complex::new(0.23, -0.11)])],
    );
    let opts = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 2,
        intermediate_photon_max: 4,
        ..StrongCouplingOptions::default()
    };
    let run = |model| {
        let spin = HubbardModel::with_uniform_u(model, 4.0)
            .unwrap()
            .floquet_spin_model(&drive, &opts)
            .unwrap();
        assert_eq!(spin.diagnostics().active_photon_max, 4);
        assert_eq!(spin.diagnostics().nearest_detuning_ev, Some(1.0));
        spin
    };
    let primitive = run(electron);
    let supercell = run(doubled);
    let tol = 2e-13;
    let expected = &primitive_coefficients(&primitive, 1)[0];
    // A missing interaction family must not pass via an empty comparison.
    assert!(expected[&Term::Pair([0, 1])].abs() > 1e-7);
    assert!(expected[&Term::Chirality([0, 1, 2])].abs() > 1e-7);
    assert!(expected[&Term::Quartic([[0, 1], [2, 3]])].abs() > 1e-7);
    // Compare both primitive copies separately; averaging could hide a sign
    // error or a missing interaction in one of them. Absent terms are zero.
    for (copy, obtained) in primitive_coefficients(&supercell, 2).iter().enumerate() {
        for term in expected.keys().chain(obtained.keys()) {
            let a = obtained.get(term).copied().unwrap_or(0.0);
            let b = expected.get(term).copied().unwrap_or(0.0);
            assert!(
                (a - b).abs() <= tol,
                "primitive copy {copy}, {term:?}: {a:e} vs {b:e} eV"
            );
        }
    }
    let a = supercell.constant_ev() / 2.0;
    let b = primitive.constant_ev();
    assert!(
        (a - b).abs() <= tol,
        "constant per primitive cell: {a:e} vs {b:e} eV"
    );
}
