use super::*;
use crate::AtomType;
use crate::atom_struct::{AtomId, OrbitalId};
use ndarray::array;
use serde::{Deserialize, Serialize, de};

#[test]
fn validation_rejects_singular_lattices_and_nonfinite_operators() {
    let valid = Model::<false, 2, HasRMatrix>::tb_model(
        array![[1.0, 0.2], [0.3, 1.0]],
        array![[0.0, 0.0]],
        None,
    )
    .unwrap();
    for lattice in [
        array![[0.0, 0.0], [0.0, 0.0]],
        array![[1.0, 2.0], [2.0, 4.0]],
    ] {
        let mut model = valid.clone();
        model.lat = lattice.clone();
        assert!(matches!(
            model.validate(),
            Err(TbError::InvalidModelInvariant {
                invariant: "invertible_lattice",
                ..
            })
        ));
        assert!(Model::<false, 2>::tb_model(lattice, array![[0.0, 0.0]], None).is_err());
        let encoded = toml::to_string(&model).unwrap();
        assert!(toml::from_str::<Model<false, 2, HasRMatrix>>(&encoded).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut model = valid.clone();
        model.ham[[0, 0, 0]].re = value;
        assert!(matches!(
            model.validate(),
            Err(TbError::InvalidModelInvariant {
                invariant: "finite_hamiltonian",
                ..
            })
        ));
        let encoded = toml::to_string(&model).unwrap();
        assert!(toml::from_str::<Model<false, 2, HasRMatrix>>(&encoded).is_err());
        model.ham[[0, 0, 0]].re = 0.0;
        model.rmatrix[[0, 0, 0, 0]].im = value;
        assert!(matches!(
            model.validate(),
            Err(TbError::InvalidModelInvariant {
                invariant: "finite_position_matrix",
                ..
            })
        ));
    }
}

#[test]
fn validation_rejects_nonfinite_atomic_and_orbital_geometry() {
    let valid = Model::<false, 2>::tb_model(
        Array2::eye(2),
        array![[0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    )
    .unwrap();
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut model = valid.clone();
        model.orb[[0, 1]] = value;
        assert!(matches!(
            model.validate(),
            Err(TbError::InvalidModelInvariant {
                invariant: "finite_geometry",
                ..
            })
        ));
        model.orb[[0, 1]] = 0.0;
        model.atoms[0].set_position(array![value, 0.0]);
        assert!(matches!(
            model.validate(),
            Err(TbError::InvalidModelInvariant {
                invariant: "atomic_position",
                ..
            })
        ));
        // Nonfinite moments are rejected by the public setter before
        // they can enter a model; preserve the previous valid metadata.
        let mut atom = valid.atoms[0].clone();
        atom.set_magnetic_moment([1.0, 2.0, 3.0]).unwrap();
        assert!(atom.set_magnetic_moment([value, 0.0, 0.0]).is_err());
        assert_eq!(atom.magnetic_moment(), Some([1.0, 2.0, 3.0]));
    }
}

fn assert_model_round_trip<const SPIN: bool, R>()
where
    R: RMatrixData + Serialize,
    for<'de> Model<SPIN, 3, R>: Deserialize<'de>,
{
    let mut model = Model::<SPIN, 3, R>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0), OrbitalId::new(1)],
        )]),
    )
    .unwrap();
    model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.0]).unwrap();
    model.set_hop(Complex::new(0.3, -0.2), 0, 1, &array![1, 0, 0], None);
    model.add_hop(0.7, 0, 1, &array![1, 0, 0], None);

    let encoded = toml::to_string(&model).unwrap();
    let decoded: Model<SPIN, 3, R> = toml::from_str(&encoded).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded.norb(), 2);
    assert_eq!(decoded.lat, model.lat);
    assert_eq!(decoded.orb, model.orb);
    assert_eq!(decoded.orb_projection, model.orb_projection);
    assert_eq!(decoded.ham, model.ham);
    assert_eq!(decoded.hamR, model.hamR);
    assert_eq!(
        decoded.atoms[0].orbitals(),
        &[OrbitalId::new(0), OrbitalId::new(1)]
    );
    assert_eq!(decoded.atoms[0].magnetic_moment(), Some([0.0, 0.0, 2.0]));
    assert_eq!(decoded.has_rmatrix(), R::HAS_RMATRIX);
    if R::HAS_RMATRIX {
        assert_eq!(decoded.rmatrix.as_array4(), model.rmatrix.as_array4());
    }
}

fn non_contiguous_model() -> Model<false, 3> {
    // Orbitals must sit at their parent atom's position (validate()
    // enforces ORBITAL_ATOM_POSITION_TOLERANCE).
    Model::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        Some(vec![
            Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0), OrbitalId::new(2)],
            ),
            Atom::with_orbitals(array![0.2, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
        ]),
    )
    .unwrap()
}

#[test]
fn atom_view_borrows_explicit_non_contiguous_orbitals() {
    let model = non_contiguous_model();
    let atom = model.atom(AtomId::new(0)).unwrap();
    assert_eq!(atom.orbitals().len(), 2);
    assert_eq!(atom.orbitals()[0].id(), OrbitalId::new(0));
    assert_eq!(atom.orbitals()[1].id(), OrbitalId::new(2));
    assert_eq!(atom.orbitals()[1].position(), array![0.0, 0.0, 0.0].view());
}

#[test]
fn orbital_only_model_has_no_fabricated_atoms() {
    let model = Model::<false, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None).unwrap();
    assert!(model.atoms.is_empty());
    assert_eq!(model.orbital_owners().unwrap(), vec![None]);
    assert!(matches!(
        model.orb_angular(),
        Err(TbError::MissingAtomicStructure)
    ));
}

#[test]
fn rmatrix_diagonal_is_cartesian_orbital_position() {
    // Position matrix elements are Cartesian (matching Wannier90
    // _r.dat); the default diagonal must equal frac · lat, not the raw
    // fractional coordinates.
    let lat = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    let orb = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
    let model = Model::<true, 3, HasRMatrix>::tb_model(lat, orb, None).unwrap();
    let rmatrix = model.rmatrix.as_array4();
    let cart = model.orb.dot(&model.lat);
    for i in 0..model.nsta() {
        let s = i % model.norb();
        for axis in 0..3 {
            assert!(
                (rmatrix[[0, axis, i, i]] - Complex::new(cart[[s, axis]], 0.0)).norm() < 1e-12,
                "rmatrix diagonal ({i}, {axis}) must equal frac·lat = {}",
                cart[[s, axis]]
            );
        }
    }
}

#[test]
fn validate_rejects_orbitals_far_from_their_atom() {
    // An orbital displaced beyond ORBITAL_ATOM_POSITION_TOLERANCE from its
    // parent atom must be rejected at construction.
    let ok = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.05, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    );
    assert!(ok.is_ok(), "deviation within tolerance must pass");

    let bad = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.15, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    );
    assert!(
        matches!(
            bad,
            Err(TbError::InvalidModelInvariant {
                invariant: "orbital_atom_position",
                ..
            })
        ),
        "deviation beyond tolerance must fail with orbital_atom_position"
    );

    // Periodicity-aware: 0.97 vs 0.03 is 0.06 apart mod 1 (across the
    // boundary) — must pass even though the raw difference is 0.94.
    let across_boundary = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.97, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.03, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    );
    assert!(
        across_boundary.is_ok(),
        "mod-1 distance 0.06 must be allowed"
    );
}

#[test]
fn remove_last_orbital_is_transactional() {
    // Regression: removing every orbital mutated all arrays before the
    // final validate() reported NoOrbitals, leaving the caller's model
    // corrupted. The error must fire before any mutation.
    let mut model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    )
    .unwrap();
    let norb_before = model.norb();
    assert!(matches!(model.remove_orb(&[0]), Err(TbError::NoOrbitals)));
    assert_eq!(
        model.norb(),
        norb_before,
        "failed removal must not mutate the model"
    );
    assert!(matches!(model.remove_atom(&[0]), Err(TbError::NoOrbitals)));
    assert_eq!(model.norb(), norb_before);
    assert_eq!(
        model.natom(),
        1,
        "failed atom removal must not mutate atoms"
    );
}

#[test]
fn tb_model_rejects_empty_orbital_set() {
    // Regression: tb_model with a zero-row orb matrix used to succeed and
    // produce a model that silently flowed into solve/response entry
    // points. It must now fail with NoOrbitals.
    let result = Model::<false, 3>::tb_model(Array2::eye(3), Array2::<f64>::zeros((0, 3)), None);
    assert!(matches!(result, Err(TbError::NoOrbitals)));
}

#[test]
fn orb_angular_rejects_unowned_orbitals() {
    // Regression: an orbital not owned by any atom used to silently keep
    // zero angular-momentum matrix elements (indistinguishable from a
    // genuine s-orbital). It must now produce a clear error.
    let model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    )
    .unwrap();
    let err = model.orb_angular().unwrap_err();
    assert!(
        err.to_string().contains("has no owner"),
        "unexpected error: {err}"
    );
}

#[test]
fn duplicate_orbital_ownership_is_rejected() {
    let result = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0]],
        Some(vec![
            Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::O, [OrbitalId::new(0)]),
        ]),
    );
    assert!(matches!(result, Err(TbError::DuplicateOrbitalOwner { .. })));
}

#[test]
fn removing_an_orbital_remaps_ids_and_keeps_empty_sites() {
    let mut model = non_contiguous_model();
    model.remove_orb(&[1]).unwrap();
    model.validate().unwrap();
    assert_eq!(model.norb(), 2);
    assert_eq!(
        model.atoms[0].orbitals(),
        &[OrbitalId::new(0), OrbitalId::new(1)]
    );
    assert!(model.atoms[1].orbitals().is_empty());
}

#[test]
fn explicit_atom_removal_modes_preserve_their_named_semantics() {
    let mut metadata_only = non_contiguous_model();
    metadata_only.remove_atoms_only(&[0]).unwrap();
    assert_eq!(metadata_only.norb(), 3);
    assert_eq!(metadata_only.natom(), 1);
    assert_eq!(
        metadata_only.orbital_owners().unwrap(),
        vec![None, Some(AtomId::new(0)), None]
    );

    metadata_only.remove_orb(&[1]).unwrap();
    assert_eq!(metadata_only.natom(), 1);
    assert!(metadata_only.atoms[0].orbitals().is_empty());
    metadata_only.prune_empty_atoms().unwrap();
    assert_eq!(metadata_only.natom(), 0);

    let mut cascading = non_contiguous_model();
    cascading.remove_atoms_and_orbitals(&[0]).unwrap();
    assert_eq!(cascading.natom(), 1);
    assert_eq!(cascading.norb(), 1);
    assert_eq!(cascading.atoms[0].orbitals(), &[OrbitalId::new(0)]);
}

#[test]
fn orbital_only_supercell_remains_orbital_only() {
    let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.25]], None).unwrap();
    let supercell = model.make_supercell(&array![[2.0]]).unwrap();
    assert_eq!(supercell.norb(), 2);
    assert!(supercell.atoms.is_empty());
    assert_eq!(supercell.orbital_owners().unwrap(), vec![None, None]);
    supercell.validate().unwrap();
}

#[test]
fn supercell_replicates_optional_atom_moments() {
    let mut model = Model::<false, 1>::tb_model(
        array![[1.0]],
        array![[0.25]],
        Some(vec![Atom::with_orbitals(
            array![0.25],
            AtomType::Fe,
            [OrbitalId::new(0)],
        )]),
    )
    .unwrap();
    model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.5]).unwrap();
    let supercell = model.make_supercell(&array![[3.0]]).unwrap();
    assert_eq!(supercell.natom(), 3);
    assert!(
        supercell
            .atoms
            .iter()
            .all(|atom| atom.magnetic_moment() == Some([0.0, 0.0, 2.5]))
    );
    supercell.validate().unwrap();
}

#[test]
fn model_serde_round_trips_all_type_level_storage_combinations() {
    assert_model_round_trip::<false, NoRMatrix>();
    assert_model_round_trip::<true, NoRMatrix>();
    assert_model_round_trip::<false, HasRMatrix>();
    assert_model_round_trip::<true, HasRMatrix>();
}

#[test]
fn model_serde_checks_required_fields_types_and_storage() {
    let model =
        Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None)
            .unwrap();
    let value = toml::Value::try_from(&model).unwrap();
    let encoded = toml::to_string(&value).unwrap();
    let stripped: Model<false, 3> = toml::from_str(&encoded).unwrap();
    assert_eq!(stripped.ham, model.ham);
    assert_eq!(stripped.orb, model.orb);
    assert!(
        toml::from_str::<Model<true, 3>>(&encoded)
            .unwrap_err()
            .to_string()
            .contains("spin mismatch")
    );
    assert!(
        toml::from_str::<Model<false, 2, HasRMatrix>>(&encoded)
            .unwrap_err()
            .to_string()
            .contains("dimension mismatch")
    );
    for field in [
        "dim_r",
        "spin",
        "lat",
        "orb",
        "orb_projection",
        "atoms",
        "ham",
        "hamR",
        "rmatrix",
    ] {
        let mut missing = value.clone();
        missing.as_table_mut().unwrap().remove(field);
        let encoded = toml::to_string(&missing).unwrap();
        let error = toml::from_str::<Model<false, 3, HasRMatrix>>(&encoded).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("missing field `{field}`"))
        );
        if field == "rmatrix" {
            assert!(toml::from_str::<Model<false, 3>>(&encoded).is_ok());
        } else {
            assert!(toml::from_str::<Model<false, 3>>(&encoded).is_err());
        }
    }
    let mut malformed = value;
    malformed["rmatrix"] = toml::Value::Integer(0);
    let encoded = toml::to_string(&malformed).unwrap();
    assert!(toml::from_str::<Model<false, 3>>(&encoded).is_err());
    assert!(toml::from_str::<Model<false, 3, HasRMatrix>>(&encoded).is_err());
}

#[test]
fn model_serde_rejects_unknown_and_duplicate_fields() {
    let model =
        Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None)
            .unwrap();
    let value = toml::Value::try_from(&model).unwrap();
    for field in ["spin", "rmatrix", "unexpected"] {
        let mut fields: Vec<_> = value.as_table().unwrap().clone().into_iter().collect();
        fields.push((
            field.to_string(),
            value.get(field).cloned().unwrap_or(toml::Value::Integer(0)),
        ));
        // Feed entries directly: a TOML parser would reject duplicate keys
        // itself and never exercise the derived Model deserializer.
        let error = Model::<false, 3>::deserialize(
            de::value::MapDeserializer::<_, toml::de::Error>::new(fields.clone().into_iter()),
        )
        .unwrap_err();
        let expected = if field == "unexpected" {
            "unknown field"
        } else {
            "duplicate field"
        };
        assert!(error.to_string().contains(expected), "{error}");
        let error = Model::<false, 3, HasRMatrix>::deserialize(de::value::MapDeserializer::<
            _,
            toml::de::Error,
        >::new(fields.into_iter()))
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn legacy_atom_counts_deserialize_to_contiguous_typed_ids() {
    let mut model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        Some(vec![
            Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
        ]),
    )
    .unwrap();
    model.atoms[1].set_magnetic_moment([0.0, 1.0, 0.0]).unwrap();
    let mut value = toml::Value::try_from(&model).unwrap();
    let atoms = value["atoms"].as_array_mut().unwrap();
    for atom in atoms {
        let table = atom.as_table_mut().unwrap();
        let count = table.remove("orbitals").unwrap().as_array().unwrap().len();
        table.insert("atom_list".to_string(), toml::Value::Integer(count as i64));
        if let Some(moment) = table.remove("magnetic_moment") {
            table.insert("magnetic".to_string(), moment);
        }
    }

    let decoded: Model<false, 3> = toml::from_str(&toml::to_string(&value).unwrap()).unwrap();
    assert_eq!(decoded.atoms[0].orbitals(), &[OrbitalId::new(0)]);
    assert_eq!(decoded.atoms[1].orbitals(), &[OrbitalId::new(1)]);
    assert_eq!(decoded.atoms[0].magnetic_moment(), None);
    assert_eq!(decoded.atoms[1].magnetic_moment(), Some([0.0, 1.0, 0.0]));
}

#[test]
fn standalone_legacy_atom_deserialization_is_rejected() {
    // Regression: deserializing a standalone Atom in the legacy
    // per-atom-count format assigned IDs 0..N to each atom independently,
    // producing overlapping OrbitalIds that surfaced later as a cryptic
    // DuplicateOrbitalOwner at Model attach. It must now be rejected with
    // a migration hint.
    let model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::C,
            [OrbitalId::new(0)],
        )]),
    )
    .unwrap();
    let mut value = toml::Value::try_from(&model).unwrap();
    let atom = value["atoms"].as_array_mut().unwrap()[0]
        .as_table_mut()
        .unwrap();
    atom.remove("orbitals").unwrap();
    atom.insert("atom_list".to_string(), toml::Value::Integer(1));
    let wire = toml::to_string(&atom).unwrap();
    let err = toml::from_str::<Atom>(&wire).unwrap_err();
    assert!(
        err.to_string().contains("standalone Atom"),
        "unexpected error: {err}"
    );
}

#[test]
fn legacy_zero_magnetic_alias_is_distinct_from_a_missing_moment() {
    let model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0]],
        Some(vec![Atom::with_orbitals(
            array![0.0, 0.0, 0.0],
            AtomType::Fe,
            [OrbitalId::new(0)],
        )]),
    )
    .unwrap();
    let mut value = toml::Value::try_from(&model).unwrap();
    let atom = value["atoms"].as_array_mut().unwrap()[0]
        .as_table_mut()
        .unwrap();
    atom.insert(
        "magnetic".to_string(),
        toml::Value::Array(vec![
            toml::Value::Float(0.0),
            toml::Value::Float(0.0),
            toml::Value::Float(0.0),
        ]),
    );
    let decoded: Model<false, 3> = toml::from_str(&toml::to_string(&value).unwrap()).unwrap();
    assert_eq!(decoded.atoms[0].magnetic_moment(), Some([0.0; 3]));
}

#[test]
fn standalone_atom_deserialization_rejects_nonfinite_moments() {
    let atom = Atom::new(array![0.0, 0.0, 0.0], AtomType::Fe);
    let mut value = toml::Value::try_from(&atom).unwrap();
    value.as_table_mut().unwrap().insert(
        "magnetic_moment".to_string(),
        toml::Value::Array(vec![
            toml::Value::Float(f64::NAN),
            toml::Value::Float(0.0),
            toml::Value::Float(0.0),
        ]),
    );
    let encoded = toml::to_string(&value).unwrap();
    assert!(toml::from_str::<Atom>(&encoded).is_err());
}
