//! Independent regressions for per-state Wannier centres in spinful models.
//!
//! The decomposition oracle compares a spinful model against two separately
//! constructed spinless models. Reusing one spin block's centres for the other
//! changes those independent spectra, so the check cannot pass by accident.

use super::{Gauge, HasRMatrix, Model, NoRMatrix};
use crate::atom_struct::{Atom, AtomType, OrbitalId};
use crate::error::TbError;
use crate::output::OutPut;
use crate::solve_ham::Solve;
use ndarray::{Axis, arr1, array, concatenate, s};
use std::f64::consts::PI;
use std::fs;

fn spinor_chain(centres: ndarray::Array2<f64>) -> Model<true, 1, NoRMatrix> {
    let mut model =
        Model::<true, 1>::tb_model_with_state_centres(array![[1.0]], centres, None).unwrap();
    // Spin-independent terms give both channels identical hopping and onsite
    // data, so only their own state centres can separate the two spectra.
    model.set_hop(-0.8, 0, 1, &arr1(&[0]), None);
    model.set_onsite(&array![0.3, -0.2], None);
    model
}

fn spinless_chain(centres: ndarray::Array2<f64>) -> Model<false, 1, NoRMatrix> {
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], centres, None).unwrap();
    model.set_hop(-0.8, 0, 1, &arr1(&[0]), None);
    model.set_onsite(&array![0.3, -0.2], None);
    model
}

fn ascending(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    values
}

#[test]
fn spinor_centres_are_stored_per_state_and_keep_both_channels() {
    let centres = array![[0.0], [0.5], [0.10], [0.35]];
    let model = spinor_chain(centres.clone());

    assert_eq!(model.norb(), 2);
    assert_eq!(model.nsta(), 4);
    assert_eq!(model.orb.dim(), (4, 1));
    for (row, expected) in model.orb.rows().into_iter().zip([0.0, 0.5, 0.10, 0.35]) {
        assert_eq!(row[0], expected);
    }
    model.validate().unwrap();

    // The common-centre constructor still copies its norb rows into both blocks.
    let shared = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], None).unwrap();
    assert_eq!(shared.orb.dim(), (4, 1));
    assert_eq!(shared.orb.slice(s![..2, ..]), shared.orb.slice(s![2.., ..]));

    // An odd spinful state table cannot be paired and must be rejected.
    assert!(matches!(
        Model::<true, 1>::tb_model_with_state_centres(
            array![[1.0]],
            array![[0.0], [0.5], [0.25]],
            None,
        ),
        Err(TbError::InvalidModelInvariant {
            invariant: "spinor_centre_count",
            ..
        })
    ));
}

#[test]
fn spinor_hamiltonian_decomposes_into_independent_channel_blocks() {
    let pair = spinor_chain(array![[0.0], [0.5], [0.10], [0.35]]);
    let reference_up = spinless_chain(array![[0.0], [0.5]]);
    let reference_down = spinless_chain(array![[0.10], [0.35]]);

    // At k = 0 every centre phase is unity, so start where the channels differ.
    for k in [0.125_f64, 0.375, 0.5] {
        let point = arr1(&[k]);
        let pair_ham = pair.gen_ham(&point, Gauge::Atom);
        let up_ham = reference_up.gen_ham(&point, Gauge::Atom);
        let down_ham = reference_down.gen_ham(&point, Gauge::Atom);
        let norb = pair.norb();

        // Each spin block must equal the independently built spinless model,
        // including the phase set by that block's own centres.
        for row in 0..norb {
            for column in 0..norb {
                assert!(
                    (pair_ham[[row, column]] - up_ham[[row, column]]).norm() < 1e-12,
                    "k={k}: spin-up block differs from the independent spinless model"
                );
                assert!(
                    (pair_ham[[norb + row, norb + column]] - down_ham[[row, column]]).norm()
                        < 1e-12,
                    "k={k}: spin-down block differs from the independent spinless model"
                );
            }
        }
        // The two channels must actually be distinguishable at this k.
        assert!(
            (up_ham[[0, 1]] - down_ham[[0, 1]]).norm() > 1e-6,
            "fixture must distinguish the two channels at k={k}"
        );

        let (pair_bands, _) = pair.solve_onek(&point).unwrap();
        let (up_bands, _) = reference_up.solve_onek(&point).unwrap();
        let (down_bands, _) = reference_down.solve_onek(&point).unwrap();
        let oracle = ascending(up_bands.iter().chain(down_bands.iter()).copied().collect());
        for (value, expected) in pair_bands.iter().zip(&oracle) {
            assert!(
                (value - expected).abs() < 1e-10,
                "k={k}: spinful bands {pair_bands:?} differ from independent channels {oracle:?}"
            );
        }
    }
}

#[test]
fn integer_centre_shift_acts_as_a_diagonal_gauge_transformation() {
    let plain = spinor_chain(array![[0.0], [0.5], [0.10], [0.35]]);
    let moved = spinor_chain(array![[0.0], [1.5], [0.10], [0.35]]);

    for k in [0.25_f64, 0.5, 0.75] {
        let point = arr1(&[k]);
        let (plain_bands, _) = plain.solve_onek(&point).unwrap();
        let (moved_bands, _) = moved.solve_onek(&point).unwrap();
        for (value, expected) in moved_bands.iter().zip(&plain_bands) {
            assert!(
                (value - expected).abs() < 1e-10,
                "an integer centre shift must not change the spectrum at k={k}"
            );
        }

        // The transmitted phase must still change, proving the shift is used.
        let plain_ham = plain.gen_ham(&point, Gauge::Atom);
        let moved_ham = moved.gen_ham(&point, Gauge::Atom);
        let expected = plain_ham[[0, 1]] * num_complex::Complex::from_polar(1.0, 2.0 * PI * k);
        assert!((moved_ham[[0, 1]] - expected).norm() < 1e-10);
    }
}

#[test]
fn spinor_position_matrix_uses_each_state_centre() {
    let centres = array![[0.0, 0.25], [0.5, -0.25], [0.125, 0.375], [0.625, -0.125]];
    let model = Model::<true, 2, HasRMatrix>::tb_model_with_state_centres(
        array![[1.0, 0.0], [0.0, 1.0]],
        centres.clone(),
        None,
    )
    .unwrap();
    model.validate().unwrap();
    let cartesian = centres.dot(&model.lat);
    for state in 0..model.nsta() {
        for axis in 0..model.dim_r() {
            let diagonal = model.rmatrix[[0, axis, state, state]];
            assert!((diagonal.re - cartesian[[state, axis]]).abs() < 1e-12);
            assert_eq!(diagonal.im, 0.0);
        }
    }
}

#[test]
fn spinor_centres_survive_round_trip_and_export() {
    let atom = vec![Atom::with_orbitals(
        array![0.0],
        AtomType::H,
        [OrbitalId::new(0), OrbitalId::new(1)],
    )];
    let centres = array![[0.0], [0.5], [0.125], [0.375]];
    let model =
        Model::<true, 1>::tb_model_with_state_centres(array![[1.0]], centres.clone(), Some(atom))
            .unwrap();

    let encoded = toml::to_string(&model).unwrap();
    let restored: Model<true, 1> = toml::from_str(&encoded).unwrap();
    assert_eq!(restored.orb, centres);

    let directory = format!("target/test-output/spinor_centres_{}/", std::process::id());
    fs::create_dir_all(&directory).unwrap();
    model.output_xyz(&directory, "spinor").unwrap();
    let text = fs::read_to_string(format!("{directory}spinor_centres.xyz")).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    let rows: Vec<&str> = text.lines().filter(|line| line.starts_with("X")).collect();
    assert_eq!(rows.len(), model.nsta(), "one centre row per state: {text}");
    assert!(rows[2].contains("0.12500000"), "{}", rows[2]);
    assert!(rows[3].contains("0.37500000"), "{}", rows[3]);
}

#[test]
fn multi_orbital_spinful_supercell_keeps_each_state_centre() {
    let up = array![[0.0], [0.4]];
    let down = array![[0.1], [0.55]];
    let centres = concatenate(Axis(0), &[up.view(), down.view()]).unwrap();
    let mut model =
        Model::<true, 1>::tb_model_with_state_centres(array![[1.0]], centres.clone(), None)
            .unwrap();
    model.set_onsite(&array![0.3, -0.15], None);
    model.set_hop(-0.7, 0, 1, &arr1(&[1]), None);
    let supercell = model.make_supercell(&array![[2.0]]).unwrap();
    assert_eq!((supercell.norb(), supercell.nsta()), (4, 8));

    // One supercell row per (state, cell copy), up copies first within each block.
    for state in 0..4 {
        for copy in 0..2 {
            let row = 2 * state + copy;
            let expected = ((centres[[state, 0]] + copy as f64) / 2.0).rem_euclid(1.0);
            assert!(
                (supercell.orb[[row, 0]] - expected).abs() < 1e-12,
                "state {state} copy {copy} centre {} differs from {expected}",
                supercell.orb[[row, 0]]
            );
        }
    }

    // Folding relation: the supercell spectrum at k is the union of the
    // primitive spectrum at k/2 and k/2 + 1/2.
    for k in [0.0_f64, 0.25, 0.5] {
        let (super_bands, _) = supercell.solve_onek(&arr1(&[k])).unwrap();
        let (first, _) = model.solve_onek(&arr1(&[k / 2.0])).unwrap();
        let (second, _) = model.solve_onek(&arr1(&[(k + 1.0) / 2.0])).unwrap();
        let oracle = ascending(first.iter().chain(second.iter()).copied().collect());
        for (value, expected) in super_bands.iter().zip(&oracle) {
            assert!(
                (value - expected).abs() < 1e-10,
                "k={k}: supercell bands {super_bands:?} differ from folded {oracle:?}"
            );
        }
    }
}
