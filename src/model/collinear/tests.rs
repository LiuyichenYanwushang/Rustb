use super::CollinearModel;
use crate::{
    Atom, AtomType, FloquetDrive, FloquetEffectiveOptions, Gauge, HasRMatrix, Kpath, LightMode,
    Model, OrbProj, OrbitalId, Solve, TbError,
};
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, array};
use num_complex::Complex64 as C;

type Channel = Model<false, 2, HasRMatrix>;
// The first two requests share an FFT grid; the duplicate crosses batches.
const REQUESTS: [(usize, isize); 5] = [(2, 2), (2, 1), (1, 1), (2, 2), (0, 0)];

fn channels() -> [Channel; 2] {
    let id = OrbitalId::new;
    let mut atoms = vec![
        Atom::with_orbitals(array![0.0, 0.0], AtomType::Fe, [id(0), id(1)]),
        Atom::with_orbitals(array![0.25, 0.25], AtomType::O, [id(2)]),
    ];
    atoms[0].set_magnetic_moment([0.0, 0.0, 1.0]).unwrap();
    let make = |x| {
        let mut model = Channel::tb_model(
            array![[1.0, 0.0], [0.25, 1.0]],
            array![[0.0, 0.0], [0.0, 0.0], [x, 0.25]],
            Some(atoms.clone()),
        )
        .unwrap();
        model.set_hop(0.35_f64, 0, 2, &array![0, 0], None);
        model
    };
    let mut up = make(0.25);
    let mut down = make(0.28);
    up.set_onsite(&array![-0.4, 0.3, 0.7], None);
    down.set_onsite(&array![0.2, -0.6, 0.9], None);
    up.set_hop(C::new(-0.4, 0.11), 0, 2, &array![1, 0], None);
    down.set_hop(C::new(0.6, -0.2), 0, 2, &array![0, 1], None);
    down.set_hop(C::new(0.18, 0.08), 1, 2, &array![1, 0], None);
    for (model, dipole) in [
        (&mut up, C::new(0.12, 0.04)),
        (&mut down, C::new(-0.09, 0.02)),
    ] {
        model.rmatrix[[0, 0, 0, 2]] = dipole;
        model.rmatrix[[0, 0, 2, 0]] = dipole.conj();
    }
    [up, down]
}

fn assert_atoms(actual: &[Atom], expected: &[Atom]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.atom_type(), b.atom_type());
        assert_eq!(a.position_ref(), b.position_ref());
        assert_eq!(a.orbitals(), b.orbitals());
        assert_eq!(a.magnetic_moment(), b.magnetic_moment());
    }
}

fn assert_channel(actual: &Channel, expected: &Channel, tol: f64) {
    assert_eq!(actual.lat, expected.lat);
    assert_atoms(&actual.atoms, &expected.atoms);
    assert_eq!(actual.orb, expected.orb);
    assert_eq!(actual.orb_projection, expected.orb_projection);
    assert_eq!(actual.hamR, expected.hamR);
    assert_eq!(actual.rmatrix.0, expected.rmatrix.0);
    assert_eq!(actual.ham.dim(), expected.ham.dim());
    for (idx, &want) in expected.ham.indexed_iter() {
        let got = actual.ham[idx];
        assert!((got - want).norm() <= tol, "ham{idx:?}: {got} != {want}");
    }
}

#[test]
fn test_to_spinful_preserves_complex_blocks_centres_and_common_metadata() {
    let [mut up, mut down] = channels();
    up.set_hop(C::new(-0.23, 0.17), 1, 2, &array![1, -1], None);
    down.orb = array![[0.12, -0.07], [-0.04, 0.16], [0.39, 0.22]];
    for (spin, model) in [&mut up, &mut down].into_iter().enumerate() {
        model.orb_projection = vec![OrbProj::px, OrbProj::py, OrbProj::pz];
        let cart = model.orb.dot(&model.lat);
        // Full Hermitian position data, including every nonlocal matrix entry.
        for ((row, axis, i, j), value) in model.rmatrix.indexed_iter_mut() {
            let rx = model.hamR[[row, 0]] as f64;
            let ry = model.hamR[[row, 1]] as f64;
            *value = if rx == 0.0 && ry == 0.0 && i == j {
                C::new(cart[[i, axis]], 0.0)
            } else {
                C::new(
                    0.01 * ((1 + 11 * spin + 3 * axis + i + j) as f64 + rx * rx + ry * ry),
                    0.07 * (i as f64 - j as f64 + 2.0 * rx - ry),
                )
            };
        }
    }
    // Neither channel keeps R=0 first, and each has support absent from the other.
    for (model, order) in [(&mut up, [2, 0, 4, 1, 3]), (&mut down, [3, 1, 4, 0, 2])] {
        model.hamR = model.hamR.select(Axis(0), &order);
        model.ham = model.ham.select(Axis(0), &order);
        model.rmatrix.0 = model.rmatrix.0.select(Axis(0), &order);
    }
    let snapshots = [up.clone(), down.clone()];
    let pair = CollinearModel::new(up, down).unwrap();
    let spinful: Model<true, 2, HasRMatrix> = pair.to_spinful().unwrap();
    assert_eq!(spinful.norb(), pair.norb());
    assert_eq!(spinful.nsta(), 2 * pair.nsta());
    assert_eq!(spinful.lat, pair.up().lat);
    assert_eq!(spinful.natom(), 2);
    assert_atoms(&spinful.atoms, pair.atoms());
    assert_eq!(
        spinful.orb_projection,
        vec![OrbProj::px, OrbProj::py, OrbProj::pz]
    );
    assert_eq!(spinful.ham.dim(), (7, 6, 6));
    assert_eq!(spinful.rmatrix.dim(), (7, 2, 6, 6));
    assert_eq!(
        spinful.hamR,
        array![[0, 0], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0]]
    );
    for (spin, source) in [pair.up(), pair.down()].into_iter().enumerate() {
        for i in 0..3 {
            assert_eq!(spinful.orb.row(spin * 3 + i), source.orb.row(i));
        }
        for (row, vector) in spinful.hamR.outer_iter().enumerate() {
            // A direct scan is independent of the conversion's support lookup.
            let matching: Vec<_> = source
                .hamR
                .outer_iter()
                .enumerate()
                .filter_map(|(index, r)| (r == vector).then_some(index))
                .collect();
            for i in 0..3 {
                for j in 0..3 {
                    let want_h: C = matching.iter().map(|&r| source.ham[[r, i, j]]).sum();
                    assert_eq!(spinful.ham[[row, spin * 3 + i, spin * 3 + j]], want_h);
                    for axis in 0..2 {
                        let want_r: C = matching
                            .iter()
                            .map(|&r| source.rmatrix[[r, axis, i, j]])
                            .sum();
                        assert_eq!(
                            spinful.rmatrix[[row, axis, spin * 3 + i, spin * 3 + j]],
                            want_r
                        );
                        assert_eq!(spinful.rmatrix[[row, axis, i, 3 + j]], C::new(0.0, 0.0));
                        assert_eq!(spinful.rmatrix[[row, axis, 3 + i, j]], C::new(0.0, 0.0));
                    }
                    assert_eq!(spinful.ham[[row, i, 3 + j]], C::new(0.0, 0.0));
                    assert_eq!(spinful.ham[[row, 3 + i, j]], C::new(0.0, 0.0));
                }
            }
        }
    }
    for k in [array![0.0, 0.0], array![0.23, -0.17], array![-0.31, 0.29]] {
        for gauge in [Gauge::Atom, Gauge::Lattice] {
            let h = spinful.gen_ham(&k, gauge);
            for (spin, source) in [pair.up(), pair.down()].into_iter().enumerate() {
                let expected = source.gen_ham(&k, gauge);
                for ((i, j), &want) in expected.indexed_iter() {
                    assert!((h[[spin * 3 + i, spin * 3 + j]] - want).norm() < 1e-12);
                }
            }
        }
        let mut expected = pair.up().solve_band_onek(&k).unwrap().to_vec();
        expected.extend(pair.down().solve_band_onek(&k).unwrap());
        expected.sort_by(f64::total_cmp);
        let actual = spinful.solve_band_onek(&k).unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-11)
        );
    }
    assert_channel(pair.up(), &snapshots[0], 0.0);
    assert_channel(pair.down(), &snapshots[1], 0.0);
}

#[test]
fn test_to_spinful_without_atoms_or_position_storage() {
    let mut up = Model::<false, 1>::tb_model(array![[1.7]], array![[-0.1], [0.2]], None).unwrap();
    let mut down =
        Model::<false, 1>::tb_model(array![[1.7]], array![[0.35], [-0.25]], None).unwrap();
    up.set_hop(C::new(-0.3, 0.1), 0, 1, &array![1], None);
    down.set_hop(C::new(0.4, 0.2), 1, 0, &array![2], None);
    // A valid model need not store the origin at all.
    for model in [&mut up, &mut down] {
        model.hamR = model.hamR.select(Axis(0), &[2, 1]);
        model.ham = model.ham.select(Axis(0), &[2, 1]);
    }
    let pair = CollinearModel::new(up, down).unwrap();
    let snapshot = format!("{pair:?}");
    // NoRMatrix::as_array4 panics, so this also detects absent-storage access.
    let spinful: Model<true, 1> = pair.to_spinful().unwrap();
    assert!(!spinful.has_rmatrix());
    assert_eq!(std::mem::size_of_val(&spinful.rmatrix), 0);
    assert!(spinful.atoms.is_empty());
    assert_eq!(spinful.norb(), 2);
    assert_eq!(spinful.nsta(), 4);
    assert_eq!(spinful.orb_projection, pair.up().orb_projection);
    assert_eq!(spinful.orb, array![[-0.1], [0.2], [0.35], [-0.25]]);
    assert_eq!(spinful.hamR, array![[0], [-2], [-1], [1], [2]]);
    assert_eq!(spinful.ham[[1, 2, 3]], C::new(0.4, -0.2));
    assert_eq!(spinful.ham[[2, 1, 0]], C::new(-0.3, -0.1));
    assert_eq!(spinful.ham[[3, 0, 1]], C::new(-0.3, 0.1));
    assert_eq!(spinful.ham[[4, 3, 2]], C::new(0.4, 0.2));
    assert_eq!(spinful.ham.iter().filter(|z| z.norm() > 0.0).count(), 4);
    for gauge in [Gauge::Atom, Gauge::Lattice] {
        let k = array![0.27];
        let h = spinful.gen_ham(&k, gauge);
        for (spin, source) in [pair.up(), pair.down()].into_iter().enumerate() {
            for ((i, j), &want) in source.gen_ham(&k, gauge).indexed_iter() {
                assert!((h[[spin * 2 + i, spin * 2 + j]] - want).norm() < 1e-12);
            }
        }
    }
    assert_eq!(format!("{pair:?}"), snapshot);
}

#[test]
fn test_to_spinful_empty_support_and_coalescing_overflow() {
    let mut empty = Model::<false, 1>::tb_model(array![[1.0]], array![[0.2]], None).unwrap();
    empty.hamR = empty.hamR.select(Axis(0), &[]);
    empty.ham = empty.ham.select(Axis(0), &[]);
    let pair = CollinearModel::new(empty.clone(), empty).unwrap();
    let spinful = pair.to_spinful().unwrap();
    assert_eq!(spinful.hamR, array![[0]]);
    assert_eq!(spinful.ham.dim(), (1, 2, 2));
    assert!(spinful.ham.iter().all(|&z| z == C::new(0.0, 0.0)));
    for source in [pair.up(), pair.down()] {
        assert_eq!(source.hamR.nrows(), 0);
        assert_eq!(
            source.gen_ham(&array![0.31], Gauge::Atom)[[0, 0]],
            C::new(0.0, 0.0)
        );
    }
    let mut overflow = pair.up().clone();
    overflow.hamR = array![[0], [0]];
    overflow.ham = ndarray::Array3::from_elem((2, 1, 1), C::new(f64::MAX, 0.0));
    // Each source entry is finite; only coalescing the duplicate origin overflows.
    let pair = CollinearModel::new(overflow, pair.down().clone()).unwrap();
    assert!(matches!(
        pair.to_spinful(),
        Err(TbError::InvalidModelInvariant {
            invariant: "finite_hamiltonian",
            ..
        })
    ));
}

#[test]
fn test_to_spinful_sums_duplicate_hamiltonian_and_position_rows() {
    let mut originals = channels();
    for (spin, model) in originals.iter_mut().enumerate() {
        for ((row, axis, i, j), value) in model.rmatrix.indexed_iter_mut() {
            *value = C::new(
                (1 + spin + row + axis + i + j) as f64,
                0.1 * (i as f64 - j as f64),
            );
        }
    }
    let mut duplicated = originals.clone();
    for model in &mut duplicated {
        let nr = model.hamR.nrows();
        let order: Vec<_> = (0..nr).rev().chain(0..nr).collect();
        model.hamR = model.hamR.select(Axis(0), &order);
        model.ham = model.ham.select(Axis(0), &order);
        model.rmatrix.0 = model.rmatrix.0.select(Axis(0), &order);
        // Distinct contributions to every R, including the origin, not mere copies.
        for row in nr..2 * nr {
            model
                .ham
                .index_axis_mut(Axis(0), row)
                .mapv_inplace(|z| 2.0 * z);
            model
                .rmatrix
                .index_axis_mut(Axis(0), row)
                .mapv_inplace(|z| 3.0 * z);
        }
    }
    let [up, down] = duplicated;
    let pair = CollinearModel::new(up, down).unwrap();
    let snapshot = pair.clone();
    let spinful = pair.to_spinful().unwrap();
    assert_eq!(
        spinful.hamR,
        array![[0, 0], [-1, 0], [0, -1], [0, 1], [1, 0]]
    );
    for (spin, source) in originals.iter().enumerate() {
        for (row, vector) in source.hamR.outer_iter().enumerate() {
            let target = spinful.hamR.outer_iter().position(|r| r == vector).unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    assert!(
                        (spinful.ham[[target, spin * 3 + i, spin * 3 + j]]
                            - 3.0 * source.ham[[row, i, j]])
                        .norm()
                            < 1e-12
                    );
                    for axis in 0..2 {
                        assert!(
                            (spinful.rmatrix[[target, axis, spin * 3 + i, spin * 3 + j]]
                                - 4.0 * source.rmatrix[[row, axis, i, j]])
                            .norm()
                                < 1e-12
                        );
                    }
                }
            }
        }
    }
    assert_channel(pair.up(), snapshot.up(), 0.0);
    assert_channel(pair.down(), snapshot.down(), 0.0);
}

#[test]
fn test_new_keeps_both_channels_independent() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up.clone(), down.clone()).unwrap();
    assert_eq!(pair.lat(), &up.lat);
    assert_atoms(pair.atoms(), &up.atoms);
    assert_channel(pair.up(), &up, 0.0);
    assert_channel(pair.down(), &down, 0.0);
    let (actual_up, actual_down) = pair.into_channels();
    assert_channel(&actual_up, &up, 0.0);
    assert_channel(&actual_down, &down, 0.0);
}

#[test]
fn test_queries_use_common_metadata_and_single_channel_size() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    let pair: &CollinearModel<2, HasRMatrix> = &pair;
    assert_eq!(pair.dim_r(), 2);
    assert!(pair.has_rmatrix());
    assert_eq!(pair.natom(), 2);
    assert_eq!(pair.norb(), 3);
    assert_eq!(pair.nsta(), pair.norb());
    assert_eq!(pair.nsta(), pair.up().nsta());
    assert_eq!(pair.down().nsta(), 3);
    assert_eq!(pair.atom_list(), vec![2, 1]);
    assert_eq!(pair.atom_position(), array![[0.0, 0.0], [0.25, 0.25]]);
    assert_eq!(pair.atom_position(), pair.down().atom_position());
    assert_ne!(pair.up().orb, pair.down().orb);
    assert_atoms(pair.atoms(), &pair.down().atoms);

    let reciprocal = pair.rec_lat().unwrap();
    assert_eq!(reciprocal, pair.up().rec_lat().unwrap());
    // Reciprocal vectors are rows, not columns, even for a skew lattice.
    let dual = reciprocal.dot(&pair.lat().t());
    let expected = Array2::<f64>::eye(2) * std::f64::consts::TAU;
    assert!((&dual - &expected).iter().all(|x| x.abs() < 1e-12));
}

#[test]
fn test_queries_keep_no_rmatrix_single_channel_size() {
    let up = Model::<false, 1>::tb_model(array![[2.0]], array![[0.0]], None).unwrap();
    let down = Model::<false, 1>::tb_model(array![[2.0]], array![[0.25]], None).unwrap();
    let pair = CollinearModel::new(up, down).unwrap();
    let pair: &CollinearModel<1> = &pair;
    assert_eq!(pair.dim_r(), 1);
    assert!(!pair.has_rmatrix());
    assert_eq!(pair.natom(), 0);
    assert!(pair.atom_list().is_empty());
    assert_eq!(pair.atom_position().dim(), (0, 1));
    assert_eq!(pair.norb(), pair.up().norb());
    assert_eq!(pair.nsta(), pair.norb());
    assert_eq!(pair.nsta(), 1); // Never doubled or summed across channels.
    assert_eq!(pair.down().nsta(), pair.nsta());
    assert_eq!(pair.rec_lat().unwrap(), array![[std::f64::consts::PI]]);

    let path = array![[0.0], [0.5]];
    let sampled = pair.k_path(&path, 3).unwrap();
    assert_eq!(sampled, pair.up().k_path(&path, 3).unwrap());
    assert_eq!(sampled.0, array![[0.0], [0.25], [0.5]]);
    assert_eq!(sampled.1, array![0.0, 0.125, 0.25]);
    assert_eq!(sampled.2, array![0.0, 0.25]);
}

#[test]
fn test_new_rejects_orbital_count_and_projection_mismatches() {
    for atoms in [
        None,
        Some(vec![Atom::with_orbitals(
            array![0.0],
            AtomType::Fe,
            [OrbitalId::new(0), OrbitalId::new(1)],
        )]),
    ] {
        let make = |n| {
            Model::<false, 1>::tb_model(array![[2.0]], Array2::zeros((n, 1)), atoms.clone())
                .unwrap()
        };
        let mut up = make(2);
        up.orb_projection = vec![OrbProj::px, OrbProj::py];
        let err = CollinearModel::new(up.clone(), make(3)).unwrap_err();
        assert!(matches!(
            err,
            TbError::InvalidModelInvariant {
                invariant: "collinear_orbital_count",
                ..
            }
        ));
        for projections in [
            vec![OrbProj::px, OrbProj::pz], // Different type.
            vec![OrbProj::py, OrbProj::px], // Same types in a different order.
        ] {
            let mut down = up.clone();
            down.orb_projection = projections;
            down.validate().unwrap();
            assert!(matches!(
                CollinearModel::new(up.clone(), down),
                Err(TbError::InvalidModelInvariant {
                    invariant: "collinear_orbital_projections",
                    ..
                })
            ));
        }
        let mut down = up.clone();
        down.orb[[0, 0]] = 0.125;
        assert!(CollinearModel::new(up, down).is_ok());
    }
}

#[test]
fn test_k_path_preserves_short_nodes_and_skew_distances() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    let pair: &CollinearModel<2, HasRMatrix> = &pair;
    let path = array![[0.0, 0.0], [1e-9, 0.0], [1.0, 0.0], [1.0, 1.0]];
    // For this lattice, reciprocal steps along x and y have lengths
    // sqrt(1 + 0.25^2) and 1, respectively, without the 2*pi factor.
    let x_length = 1.0625_f64.sqrt();
    let expected_nodes = array![0.0, 1e-9 * x_length, x_length, x_length + 1.0];
    for nk in [4, 9] {
        let sampled = pair.k_path(&path, nk).unwrap();
        assert_eq!(sampled, pair.up().k_path(&path, nk).unwrap());
        assert_eq!(sampled, pair.down().k_path(&path, nk).unwrap());
        let (points, distances, nodes) = sampled;
        assert_eq!(points.dim(), (nk, 2));
        assert_eq!(distances.len(), nk);
        assert_eq!(nodes.len(), path.nrows());
        assert!((&nodes - &expected_nodes).iter().all(|x| x.abs() < 1e-14));
        assert_eq!(distances[0], 0.0);
        assert_eq!(distances[nk - 1], nodes[nodes.len() - 1]);
        assert!(distances.windows(2).into_iter().all(|w| w[1] > w[0]));
        for node in path.rows() {
            assert!(points.rows().into_iter().any(|point| point == node));
        }
    }
}

#[test]
fn test_k_path_rejects_invalid_nodes_and_sample_counts() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    assert!(matches!(
        pair.k_path(&array![[0.0], [1.0]], 3),
        Err(TbError::PathLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
    for path in [
        Array2::zeros((0, 2)),
        array![[0.0, 0.0]],
        array![[0.0, 0.0], [0.0, 0.0]],
        array![[0.0, 0.0], [f64::NAN, 0.0]],
        array![[0.0, 0.0], [0.0, f64::INFINITY]],
    ] {
        assert!(matches!(pair.k_path(&path, 3), Err(TbError::Other(_))));
    }
    let path = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]];
    for nk in [0, 2, usize::MAX] {
        assert!(matches!(pair.k_path(&path, nk), Err(TbError::Other(_))));
    }
}

#[test]
fn test_new_rejects_lattice_or_atom_metadata_mismatches() {
    let [up, down] = channels();
    let reject = |case, invariant, change: fn(&mut Channel)| {
        let mut changed = down.clone();
        change(&mut changed);
        changed.validate().unwrap();
        let err = CollinearModel::new(up.clone(), changed).unwrap_err();
        assert!(
            matches!(err, TbError::InvalidModelInvariant { invariant: got, .. } if got == invariant),
            "{case}: {err}"
        );
    };
    reject("one-ULP lattice change", "collinear_lattice", |m| {
        m.lat[[0, 0]] = m.lat[[0, 0]].next_up();
    });
    reject("atom count", "collinear_atoms", |m| m.atoms.truncate(1));
    reject("atom order", "collinear_atoms", |m| m.atoms.swap(0, 1));
    reject("species", "collinear_atoms", |m| {
        m.atoms[1].change_type(AtomType::Si)
    });
    reject("one-ULP position change", "collinear_atoms", |m| {
        m.atoms[1].set_position(array![0.25_f64.next_up(), 0.25]);
    });
    reject("orbital IDs", "collinear_atoms", |m| {
        m.atoms[0].set_orbitals(vec![OrbitalId::new(0), OrbitalId::new(2)]);
        m.atoms[1].set_orbitals(vec![OrbitalId::new(1)]);
    });
    reject("orbital ID order", "collinear_atoms", |m| {
        m.atoms[0].set_orbitals(vec![OrbitalId::new(1), OrbitalId::new(0)]);
    });
    reject("absent vs explicit zero moment", "collinear_atoms", |m| {
        m.atoms[1].set_magnetic_moment([0.0; 3]).unwrap();
    });
    reject("one-ULP moment change", "collinear_atoms", |m| {
        m.atoms[0]
            .set_magnetic_moment([0.0, 0.0, 1.0_f64.next_up()])
            .unwrap();
    });
}

#[test]
fn test_bands_match_independent_solves_in_spin_and_k_order() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up.clone(), down.clone()).unwrap();
    let points = array![[0.27, -0.13], [0.0, 0.0], [-0.31, 0.2], [0.27, -0.13]];
    let (batch_up, batch_down) = pair.solve_band_all_parallel(&points).unwrap();
    for (row, k) in points.outer_iter().enumerate() {
        let (single_up, single_down) = pair.solve_band_onek(&k).unwrap();
        for (spin, (model, single, batch)) in [
            (&up, &single_up, &batch_up),
            (&down, &single_down, &batch_down),
        ]
        .into_iter()
        .enumerate()
        {
            let expected = model.solve_band_onek(&k).unwrap();
            assert_eq!(single, &expected);
            assert_eq!(batch.dim(), (points.nrows(), model.norb()));
            for (&got, &want) in batch.row(row).iter().zip(&expected) {
                assert!(
                    (got - want).abs() < 1e-12,
                    "spin {spin}, k row {row}: {got} != {want}"
                );
            }
        }
    }
}

fn drive() -> FloquetDrive {
    FloquetDrive::with_modes(
        4.7,
        vec![LightMode::new(
            1,
            array![C::new(0.61, 0.13), C::new(0.19, 0.72)],
        )],
    )
}

#[test]
fn test_floquet_pair_and_sweep_match_independent_channels() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up.clone(), down.clone()).unwrap();
    let drive = drive();
    let refs = REQUESTS.map(|(order, h)| {
        let options = FloquetEffectiveOptions::new()
            .with_order(order)
            .with_harmonic_max(h);
        [&up, &down].map(|m| m.floquet_effective_model(&drive, Some(&options)).unwrap())
    });
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(2);
    let fixed = pair
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    assert_channel(fixed.up(), &refs[0][0], 2e-11);
    assert_channel(fixed.down(), &refs[0][1], 2e-11);

    let mut seen = Vec::new();
    pair.floquet_effective_sweep(&drive, &REQUESTS, |key, actual| {
        let i = seen.len();
        assert_eq!(key, REQUESTS[i]);
        assert_channel(actual.up(), &refs[i][0], 2e-11);
        assert_channel(actual.down(), &refs[i][1], 2e-11);
        assert_eq!(actual.norb(), 3);
        assert_eq!(actual.nsta(), actual.norb());
        assert_eq!(actual.nsta(), actual.up().nsta());
        assert_eq!(actual.nsta(), actual.down().nsta());
        seen.push(key);
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, REQUESTS);
}

#[test]
fn test_floquet_sweep_rejects_bad_late_requests_and_skips_empty() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    for invalid in [(4, 0), (1, -1), (3, isize::MAX)] {
        let result =
            pair.floquet_effective_sweep(&drive(), &[(0, 0), (1, 1), invalid], |key, _| {
                panic!("callback ran for {key:?} before rejecting {invalid:?}")
            });
        assert!(result.is_err(), "accepted {invalid:?}");
    }
    pair.floquet_effective_sweep(&FloquetDrive::new(0.0), &[], |_, _| {
        panic!("empty sweep called the callback")
    })
    .unwrap();
}

#[test]
fn test_floquet_sweep_returns_callback_error_and_stops() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    let mut seen = Vec::new();
    let result = pair.floquet_effective_sweep(&drive(), &REQUESTS, |key, _| {
        seen.push(key);
        if seen.len() == 3 {
            return Err(TbError::Other("stop from collinear visitor".into()));
        }
        Ok(())
    });
    assert!(matches!(result, Err(TbError::Other(s)) if s == "stop from collinear visitor"));
    assert_eq!(seen.as_slice(), &REQUESTS[..3]);
}

#[test]
fn test_floquet_sweep_delivers_prefix_before_later_up_error() {
    let down = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
    let mut up = down.clone();
    // A valid sparse input whose order-3 static sumset overflows isize.
    up.set_hop(0.5_f64, 0, 0, &array![isize::MAX / 2], None);
    let pair = CollinearModel::new(up, down).unwrap();
    let drive = FloquetDrive::new(1.0);
    let requests = [(0, 0), (3, 1)];

    let mut seen = Vec::new();
    let err = pair
        .floquet_effective_sweep(&drive, &requests, |key, _| {
            seen.push(key);
            Ok(())
        })
        .unwrap_err();
    assert_eq!(seen, [requests[0]]);
    assert!(matches!(err, TbError::Other(s) if s.contains("translation overflows")));

    let err = pair
        .floquet_effective_sweep(&drive, &requests, |key, _| {
            assert_eq!(key, requests[0]);
            Err(TbError::Other("stop at first pair".into()))
        })
        .unwrap_err();
    assert!(matches!(err, TbError::Other(s) if s == "stop at first pair"));
}

fn assert_row_kets(ham: &Array2<C>, energies: ArrayView1<'_, f64>, vectors: ArrayView2<'_, C>) {
    assert_eq!(vectors.dim(), (energies.len(), ham.nrows()));
    assert!(
        energies
            .iter()
            .zip(energies.iter().skip(1))
            .all(|(a, b)| a <= b)
    );
    for (band, ket) in vectors.outer_iter().enumerate() {
        let residual = ham.dot(&ket) - ket.mapv(|z| z * energies[band]);
        assert!(residual.iter().all(|z| z.norm() < 1e-11));
        for (other, other_ket) in vectors.outer_iter().enumerate() {
            let overlap = ket.mapv(|z| z.conj()).dot(&other_ket);
            let expected = if band == other { 1.0 } else { 0.0 };
            assert!((overlap - expected).norm() < 1e-11);
        }
    }
}

#[test]
fn test_paired_solvers_keep_complex_row_kets_and_input_k_order() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    let points = array![[0.27, -0.13], [0.0, 0.0], [-0.31, 0.2], [0.27, -0.13]];
    let bands = pair.solve_band_all(&points).unwrap();
    let batches = [
        pair.solve_all(&points).unwrap(),
        pair.solve_all_parallel(&points).unwrap(),
    ];
    let window = (-1.0, 0.8);
    for (row, k) in points.outer_iter().enumerate() {
        let single = pair.solve_onek(&k).unwrap();
        let ranged = pair.solve_range_onek(&k, window, 0.0).unwrap();
        let range_bands = pair.solve_band_range_onek(&k, window, 0.0).unwrap();
        for (spin, (model, bands, single, ranged, range_bands)) in [
            (pair.up(), &bands.0, &single.0, &ranged.0, &range_bands.0),
            (pair.down(), &bands.1, &single.1, &ranged.1, &range_bands.1),
        ]
        .into_iter()
        .enumerate()
        {
            let expected = model.solve_band_onek(&k).unwrap();
            let ham = model.gen_ham(&k, Gauge::Atom);
            assert!(ham.iter().any(|z| z.im.abs() > 1e-3));
            assert_eq!(bands.dim(), (points.nrows(), model.nsta()));
            for energies in [bands.row(row), single.0.view()] {
                assert_eq!(energies.len(), expected.len());
                assert!(
                    energies
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).abs() < 1e-11)
                );
            }
            assert_row_kets(&ham, single.0.view(), single.1.view());
            for batch in &batches {
                let (energies, vectors) = [&batch.0, &batch.1][spin];
                assert_eq!(energies.dim(), (points.nrows(), model.nsta()));
                assert_eq!(vectors.dim(), (points.nrows(), model.nsta(), model.nsta()));
                assert!(
                    energies
                        .row(row)
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| (a - b).abs() < 1e-11)
                );
                assert_row_kets(&ham, energies.row(row), vectors.index_axis(Axis(0), row));
            }
            let selected: Vec<_> = expected
                .iter()
                .copied()
                .filter(|&e| e > window.0 && e <= window.1)
                .collect();
            assert!(!selected.is_empty());
            for energies in [range_bands.view(), ranged.0.view()] {
                assert_eq!(energies.len(), selected.len());
                assert!(
                    energies
                        .iter()
                        .zip(&selected)
                        .all(|(a, b)| (a - b).abs() < 1e-11)
                );
            }
            assert_row_kets(&ham, ranged.0.view(), ranged.1.view());
        }
    }
}

#[test]
fn test_range_solvers_keep_unequal_selections_and_empty_channels() {
    let [mut up, mut down] = channels();
    up.ham.fill(C::new(0.0, 0.0));
    down.ham.fill(C::new(0.0, 0.0));
    up.set_onsite(&array![-1.0, 0.0, 2.0], None);
    down.set_onsite(&array![-2.0, 1.0, 3.0], None);
    let pair = CollinearModel::new(up, down).unwrap();
    let k = array![0.0, 0.0]; // Exact diagonal energies pin the open/closed endpoints.
    for (window, expected) in [
        ((-1.0, 2.0), [vec![0.0, 2.0], vec![1.0]]),
        ((0.0, 1.0), [vec![], vec![1.0]]),
        ((1.0, 2.0), [vec![2.0], vec![]]),
        ((4.0, 5.0), [vec![], vec![]]),
    ] {
        for tolerance in [-1.0, 0.0, 1e-12] {
            let bands = pair.solve_band_range_onek(&k, window, tolerance).unwrap();
            let states = pair.solve_range_onek(&k, window, tolerance).unwrap();
            for (spin, (model, bands, (energies, vectors))) in [
                (pair.up(), &bands.0, &states.0),
                (pair.down(), &bands.1, &states.1),
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(bands.as_slice().unwrap(), expected[spin].as_slice());
                assert_eq!(energies.as_slice().unwrap(), expected[spin].as_slice());
                assert_row_kets(
                    &model.gen_ham(&k, Gauge::Atom),
                    energies.view(),
                    vectors.view(),
                );
            }
        }
    }
}

#[test]
fn test_paired_solvers_reject_bad_inputs_and_down_overflow() {
    let [up, down] = channels();
    let pair = CollinearModel::new(up, down).unwrap();
    for k in [
        array![],
        array![0.0],
        array![0.0, 0.0, 0.0],
        array![f64::NAN, 0.0],
        array![0.0, f64::INFINITY],
        array![f64::NEG_INFINITY, 0.0],
    ] {
        let mut points = Array2::zeros((2, k.len()));
        points.row_mut(1).assign(&k); // A bad later row must reject the whole batch.
        for result in [
            pair.solve_band_onek(&k).map(|_| ()),
            pair.solve_band_range_onek(&k, (-1.0, 1.0), 0.0).map(|_| ()),
            pair.solve_onek(&k).map(|_| ()),
            pair.solve_range_onek(&k, (-1.0, 1.0), 0.0).map(|_| ()),
            pair.solve_band_all(&points).map(|_| ()),
            pair.solve_band_all_parallel(&points).map(|_| ()),
            pair.solve_all(&points).map(|_| ()),
            pair.solve_all_parallel(&points).map(|_| ()),
        ] {
            let err = result.unwrap_err();
            if k.len() != 2 {
                assert!(
                    matches!(err, TbError::DimensionMismatch { expected: 2, found, .. } if found == k.len())
                );
            } else {
                assert!(
                    matches!(err, TbError::Other(message) if message.contains("k-points must be finite"))
                );
            }
        }
    }
    for (window, tolerance) in [
        ((1.0, 1.0), 0.0),
        ((2.0, -1.0), 0.0),
        ((f64::NAN, 1.0), 0.0),
        ((f64::NEG_INFINITY, 1.0), 0.0),
        ((-1.0, f64::INFINITY), 0.0),
        ((-1.0, 1.0), f64::NAN),
        ((-1.0, 1.0), f64::INFINITY),
        ((-1.0, 1.0), f64::NEG_INFINITY),
    ] {
        let k = array![0.0, 0.0];
        for result in [
            pair.solve_band_range_onek(&k, window, tolerance)
                .map(|_| ()),
            pair.solve_range_onek(&k, window, tolerance).map(|_| ()),
        ] {
            assert!(matches!(result, Err(TbError::Other(_))));
        }
    }
    let [up, mut down] = channels();
    // Each real-space block is finite; only down's Fourier sum at Gamma overflows.
    down.set_hop(1e308, 0, 0, &array![1, 0], None);
    let pair = CollinearModel::new(up, down).unwrap();
    let k = array![0.0, 0.0];
    let points = k.view().insert_axis(Axis(0));
    assert!(pair.up().solve_onek(&k).is_ok());
    for result in [
        pair.solve_band_onek(&k).map(|_| ()),
        pair.solve_band_range_onek(&k, (-1.0, 1.0), 0.0).map(|_| ()),
        pair.solve_onek(&k).map(|_| ()),
        pair.solve_range_onek(&k, (-1.0, 1.0), 0.0).map(|_| ()),
        pair.solve_band_all(&points).map(|_| ()),
        pair.solve_band_all_parallel(&points).map(|_| ()),
        pair.solve_all(&points).map(|_| ()),
        pair.solve_all_parallel(&points).map(|_| ()),
    ] {
        assert!(
            matches!(result, Err(TbError::Other(message)) if message.contains("H(k) contains nonfinite"))
        );
    }
}

#[test]
fn test_orbital_only_independent_bands_and_empty_batches() {
    let mut up = Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
    let mut down =
        Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
    up.set_onsite(&array![0.5, 1.5], None);
    down.set_onsite(&array![2.0, -1.0], None);
    let pair = CollinearModel::new(up, down).unwrap();
    assert!(pair.atoms().is_empty());
    assert_eq!(pair.nsta(), 2);
    let expected = [array![0.5, 1.5], array![-1.0, 2.0]];
    let k = array![0.31, -0.11];
    // Equal basis sizes still allow unequal numbers of states in one window.
    let window = (-2.0, 1.75);
    let bands = pair.solve_band_range_onek(&k, window, 0.0).unwrap();
    assert_eq!(bands, (array![0.5, 1.5], array![-1.0]));
    let states = pair.solve_range_onek(&k, window, 0.0).unwrap();
    assert_eq!(states.0.0, bands.0);
    assert_eq!(states.1.0, bands.1);
    assert_eq!(states.0.1.dim(), (2, 2));
    assert_eq!(states.1.1.dim(), (1, 2));
    for (up_bands, down_bands) in [
        pair.solve_band_onek(&k).unwrap(),
        pair.solve_band_range_onek(&k, (-3.0, 3.0), 0.0).unwrap(),
    ] {
        assert_eq!(up_bands, expected[0]);
        assert_eq!(down_bands, expected[1]);
    }
    for states in [
        pair.solve_onek(&k).unwrap(),
        pair.solve_range_onek(&k, (-3.0, 3.0), 0.0).unwrap(),
    ] {
        for (spin, (model, (energies, vectors))) in
            [(pair.up(), &states.0), (pair.down(), &states.1)]
                .into_iter()
                .enumerate()
        {
            assert_eq!(energies, &expected[spin]);
            assert_row_kets(
                &model.gen_ham(&k, Gauge::Atom),
                energies.view(),
                vectors.view(),
            );
        }
    }
    for points in [array![[0.31, -0.11], [0.0, 0.0]], Array2::zeros((0, 2))] {
        for bands in [
            pair.solve_band_all(&points).unwrap(),
            pair.solve_band_all_parallel(&points).unwrap(),
        ] {
            for (bands, expected) in [(&bands.0, &expected[0]), (&bands.1, &expected[1])] {
                assert_eq!(bands.dim(), (points.nrows(), expected.len()));
                for row in bands.outer_iter() {
                    assert_eq!(&row, expected);
                }
            }
        }
        for states in [
            pair.solve_all(&points).unwrap(),
            pair.solve_all_parallel(&points).unwrap(),
        ] {
            for (spin, (model, (energies, vectors))) in
                [(pair.up(), &states.0), (pair.down(), &states.1)]
                    .into_iter()
                    .enumerate()
            {
                let n = expected[spin].len();
                assert_eq!(energies.dim(), (points.nrows(), n));
                assert_eq!(vectors.dim(), (points.nrows(), n, n));
                for (row, k) in points.outer_iter().enumerate() {
                    assert_eq!(energies.row(row), expected[spin]);
                    assert_row_kets(
                        &model.gen_ham(&k, Gauge::Atom),
                        energies.row(row),
                        vectors.index_axis(Axis(0), row),
                    );
                }
            }
        }
    }
    let wrong_shape = Array2::<f64>::zeros((0, 3));
    for result in [
        pair.solve_band_all(&wrong_shape).map(|_| ()),
        pair.solve_band_all_parallel(&wrong_shape).map(|_| ()),
        pair.solve_all(&wrong_shape).map(|_| ()),
        pair.solve_all_parallel(&wrong_shape).map(|_| ()),
    ] {
        assert!(matches!(
            result,
            Err(TbError::DimensionMismatch {
                expected: 2,
                found: 3,
                ..
            })
        ));
    }
}
