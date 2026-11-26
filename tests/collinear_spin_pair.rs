#![cfg(feature = "cryspglib")]

use Rustb::{
    Atom, AtomType, CenterResolvedBasis, CollinearModel, CollinearMoments,
    HamiltonianSymmetrizationParameters, Model, OrbitalId, SpinTolerances,
    detect_collinear_spin_group,
};
use cryspglib::Crystal;
use ndarray::{Array2, array};
use num_complex::Complex64;

fn onsite(model: &Model<false, 3>, orbital: usize) -> Complex64 {
    let index = (0..model.hamR.nrows())
        .find(|&index| (0..3).all(|axis| model.hamR[[index, axis]] == 0))
        .expect("onsite block");
    model.ham[[index, orbital, orbital]]
}

fn two_site_model(onsite_energy: f64) -> Model<false, 3> {
    let mut model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        Some(vec![
            Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::Fe, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::Fe, [OrbitalId::new(1)]),
        ]),
    )
    .unwrap();
    for orbital in 0..2 {
        model.set_hop(
            Complex64::new(onsite_energy, 0.0),
            orbital,
            orbital,
            &array![0, 0, 0],
            None,
        );
    }
    model.set_hop(Complex64::new(0.5, 0.25), 0, 1, &array![0, 0, 0], None);
    model
}

fn crystal() -> Crystal {
    Crystal::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        vec![26, 26],
    )
    .unwrap()
}

#[test]
fn full_collinear_spin_group_projects_the_channel_average() {
    let mut input_up = two_site_model(1.0);
    let mut input_down = two_site_model(2.0);
    let moments = [1.0, -1.0];
    for channel in [&mut input_up, &mut input_down] {
        for (atom, moment) in channel.atoms.iter_mut().zip(moments) {
            atom.set_magnetic_moment([0.0, 0.0, moment]).unwrap();
        }
    }
    let original_atoms = input_up.atoms.clone();
    let input = CollinearModel::new(input_up, input_down).unwrap();
    let group = detect_collinear_spin_group(
        &crystal(),
        CollinearMoments::Signed(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    let projected = input
        .symmetrize(
            &group,
            &CenterResolvedBasis::new(),
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap();
    let (up, down) = (projected.up(), projected.down());
    assert!((onsite(up, 0) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert!((onsite(up, 1) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert!((onsite(down, 0) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert!((onsite(down, 1) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert_eq!(input.up().ham, two_site_model(1.0).ham);
    assert_eq!(input.down().ham, two_site_model(2.0).ham);
    for channel in [input.up(), input.down(), up, down] {
        assert_eq!(channel.lat, Array2::<f64>::eye(3));
        assert_eq!(channel.atoms.len(), original_atoms.len());
        for (atom, original) in channel.atoms.iter().zip(&original_atoms) {
            assert_eq!(atom.atom_type(), original.atom_type());
            assert_eq!(atom.position_ref(), original.position_ref());
            assert_eq!(atom.orbitals(), original.orbitals());
            assert_eq!(atom.magnetic_moment(), original.magnetic_moment());
        }
    }
    let (up_bands, down_bands) = projected
        .solve_band_onek(&array![0.17, 0.23, -0.11])
        .unwrap();
    for bands in [up_bands, down_bands] {
        assert_eq!(bands.len(), 2);
        assert!(bands.iter().all(|energy| energy.is_finite()));
        assert!((bands.sum() - 3.0).abs() < 1e-10);
    }
}

#[test]
fn collinear_pair_rejects_projection_mismatch_before_symmetrization() {
    use Rustb::{OrbProj, TbError};

    let up = two_site_model(1.0);
    let mut down = two_site_model(2.0);
    down.set_projection(&vec![OrbProj::s, OrbProj::px]);
    up.validate().unwrap();
    down.validate().unwrap();
    // Individually valid channels must share their ordered orbital types
    // before the pair can enter any solver or channel-swapping projection.
    assert!(matches!(
        CollinearModel::new(up, down),
        Err(TbError::InvalidModelInvariant {
            invariant: "collinear_orbital_projections",
            ..
        })
    ));
}

#[test]
fn unitary_collinear_subgroup_is_accepted_separately() {
    let input_up = two_site_model(1.0);
    let input_down = two_site_model(3.0);
    let moments = [1.0, -1.0];
    let group = detect_collinear_spin_group(
        &crystal(),
        CollinearMoments::Signed(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    let input = CollinearModel::new(input_up, input_down).unwrap();
    let projected = input
        .symmetrize_unitary(
            &group,
            &CenterResolvedBasis::new(),
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap();
    assert!((onsite(projected.up(), 0) - Complex64::new(2.0, 0.0)).norm() < 1e-10);
    assert!((onsite(projected.down(), 0) - Complex64::new(2.0, 0.0)).norm() < 1e-10);
    assert_eq!(input.up().ham, two_site_model(1.0).ham);
    assert_eq!(input.down().ham, two_site_model(3.0).ham);
}

#[test]
fn full_projection_imposes_antiunitary_partners_but_unitary_projection_does_not() {
    // Three generic sites remove inversion as well as every other nonidentity
    // spatial symmetry, isolating the spin-only antiunitary conjugation.
    let lattice = array![[1.0, 0.0, 0.0], [0.17, 1.37, 0.0], [0.23, 0.31, 1.91]];
    let positions = vec![[0.0; 3], [0.173, 0.281, 0.397], [0.313, 0.109, 0.227]];
    let crystal = Crystal::new(
        std::array::from_fn(|cart| std::array::from_fn(|vector| lattice[[vector, cart]])),
        positions.clone(),
        vec![26; 3],
    )
    .unwrap();
    let group = detect_collinear_spin_group(
        &crystal,
        CollinearMoments::Signed(&[1.0; 3]),
        SpinTolerances::default(),
    )
    .unwrap();
    assert_eq!(group.operations().len(), 2);
    assert_eq!(
        group
            .operations()
            .iter()
            .filter(|op| op.kind.is_unitary())
            .count(),
        1
    );
    let make_channel = |hopping| {
        let mut model = Model::<false, 3>::tb_model(
            lattice.clone(),
            ndarray::arr2(&positions),
            Some(
                positions
                    .iter()
                    .enumerate()
                    .map(|(orbital, position)| {
                        Atom::with_orbitals(
                            ndarray::arr1(position),
                            AtomType::Fe,
                            [OrbitalId::new(orbital)],
                        )
                    })
                    .collect(),
            ),
        )
        .unwrap();
        model.set_hop(hopping, 0, 1, &array![0, 0, 0], None);
        model
    };
    let original_up = make_channel(Complex64::new(0.5, 0.25));
    let original_down = make_channel(Complex64::new(-0.7, 0.4));
    let input = CollinearModel::new(original_up.clone(), original_down.clone()).unwrap();
    let basis = CenterResolvedBasis::new();
    let params = HamiltonianSymmetrizationParameters::default();
    let unitary = input.symmetrize_unitary(&group, &basis, &params).unwrap();
    let full = input.symmetrize(&group, &basis, &params).unwrap();
    for (input, original, unitary, full) in [
        (input.up(), &original_up, unitary.up(), full.up()),
        (input.down(), &original_down, unitary.down(), full.down()),
    ] {
        assert_eq!(unitary.hamR, original.hamR);
        assert_eq!(unitary.ham.dim(), original.ham.dim());
        // Projection may reassociate floating-point arithmetic even for identity.
        assert!(
            unitary
                .ham
                .iter()
                .zip(&original.ham)
                .all(|(actual, expected)| {
                    (actual - expected).norm() <= 1e-12 * expected.norm().max(1.0)
                })
        );
        assert!(unitary.ham[[0, 0, 1]].im.abs() > 0.1);
        assert!(
            (full.ham[[0, 0, 1]] - Complex64::new(original.ham[[0, 0, 1]].re, 0.0)).norm() < 1e-10
        );
        assert_eq!(input.ham, original.ham);
        assert_eq!(input.hamR, original.hamR);
        assert_eq!(input.orb, original.orb);
    }
}

#[test]
fn pair_projection_reports_the_same_first_final_covariance_error() {
    use Rustb::{
        BasisActionContext, CellShiftAction, CrystalSymmetryOperation, HasRMatrix,
        LocalizedBasisAction, TbError,
    };

    // Distinct orthorhombic axes: every point operation acts on x by a sign.
    let lattice = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]];
    let crystal = Crystal::new(lattice, vec![[0.0; 3]], vec![26]).unwrap();
    let group = detect_collinear_spin_group(
        &crystal,
        CollinearMoments::Signed(&[1.0]),
        SpinTolerances::default(),
    )
    .unwrap();
    let make_channel = |dipole| {
        let mut model = Model::<false, 3, HasRMatrix>::tb_model(
            ndarray::arr2(&lattice),
            Array2::zeros((2, 3)),
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Fe,
                (0..2).map(OrbitalId::new),
            )]),
        )
        .unwrap();
        model.set_hop(1.0, 0, 1, &array![0, 0, 0], None);
        model.rmatrix[[0, 0, 0, 1]] = Complex64::new(dipole, 0.0);
        model.rmatrix[[0, 0, 1, 0]] = Complex64::new(dipole, 0.0);
        model
    };
    let input = CollinearModel::new(make_channel(0.125), make_channel(0.25)).unwrap();

    // The initial action has opposite x parities when x is reversed. Its
    // average removes the onsite H mixing but preserves the x dipole. On the
    // projected H, switch to the also-valid trivial action, which fails r covariance.
    let basis = |context: BasisActionContext<'_, false, HasRMatrix>| {
        let mut matrix = Array2::eye(2);
        if context.model.ham[[0, 0, 1]].norm() > 0.5 {
            matrix[[1, 1]] = Complex64::new(f64::from(context.operation.rotation[0][0]), 0.0);
        }
        Ok(LocalizedBasisAction {
            sectors: vec![CellShiftAction {
                shift: [0; 3],
                matrix,
            }],
        })
    };
    let first = group
        .operations()
        .iter()
        .find(|op| op.kind.is_unitary() && op.rotation[0][0] == -1)
        .unwrap();
    let expected_operation = CrystalSymmetryOperation {
        rotation: first.rotation,
        translation: first.translation,
        time_reversal: false,
    };
    let reject = |workers| {
        let error = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                input.symmetrize_unitary(
                    &group,
                    &basis,
                    &HamiltonianSymmetrizationParameters::default(),
                )
            })
            .unwrap_err();
        match error {
            TbError::TargetMagneticGroupIncompatible { reason } => reason,
            other => panic!("unexpected pair projection error: {other}"),
        }
    };
    let serial = reject(1);
    // Both channels fail for every x reversal. The up dipole gives the first
    // residual, 2 * 0.125, distinct from the down channel's 2 * 0.25.
    assert!(
        serial.starts_with("position matrix covariance residual 2.5e-1 Å at R=[0, 0, 0]"),
        "{serial}"
    );
    assert!(
        serial.ends_with(&format!("for operation {expected_operation:?}")),
        "{serial}"
    );
    assert_eq!(serial, reject(4));
}

#[test]
#[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
fn mgfe6ge6_collinear_spin_pair_acceptance() {
    use Rustb::Wannier90;

    let base = "/mnt/e/vasp计算结果/altermagnetic/odd_parity_wave/material/MgFe6Ge6/data_new";
    let up =
        <Model<false, 3> as Wannier90>::from_hr(&format!("{base}/up/"), "wannier90", 7.62075874)
            .unwrap();
    let down =
        <Model<false, 3> as Wannier90>::from_hr(&format!("{base}/dn/"), "wannier90", 7.62075874)
            .unwrap();
    let lattice: [[f64; 3]; 3] =
        std::array::from_fn(|cart| std::array::from_fn(|vector| up.lat[[vector, cart]]));
    let positions = up
        .atoms
        .iter()
        .map(|atom| {
            let position = atom.position_ref();
            [position[0], position[1], position[2]]
        })
        .collect::<Vec<_>>();
    let types = up
        .atoms
        .iter()
        .map(|atom| atom.atom_type().atomic_number())
        .collect::<Vec<_>>();
    let crystal = Crystal::new(lattice, positions, types).unwrap();
    let moments = up
        .atoms
        .iter()
        .map(|atom| {
            if format!("{:?}", atom.atom_type()).contains("Fe") {
                if atom.position_ref()[2] < 0.5 {
                    [0.0, 0.0, 1.0]
                } else {
                    [0.0, 0.0, -1.0]
                }
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect::<Vec<_>>();
    let group = detect_collinear_spin_group(
        &crystal,
        CollinearMoments::Vectors(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    CollinearModel::new(up, down)
        .unwrap()
        .symmetrize(
            &group,
            &CenterResolvedBasis::new(),
            &HamiltonianSymmetrizationParameters::default(),
        )
        .unwrap();
}

#[test]
fn pair_projection_updates_position_matrices() {
    use Rustb::{HasRMatrix, OrbProj, find_R};

    // Two complete s+p shells exercise dense BLAS as well as Rayon scheduling.
    let mut input_up: Model<false, 3, HasRMatrix> = Model::tb_model(
        Array2::eye(3),
        Array2::from_shape_fn((8, 3), |(i, j)| if i >= 4 && j == 0 { 0.5 } else { 0.0 }),
        Some(vec![
            Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Fe,
                (0..4).map(OrbitalId::new),
            ),
            Atom::with_orbitals(
                array![0.5, 0.0, 0.0],
                AtomType::Fe,
                (4..8).map(OrbitalId::new),
            ),
        ]),
    )
    .unwrap();
    input_up.set_projection(&[OrbProj::s, OrbProj::px, OrbProj::py, OrbProj::pz].repeat(2));
    let mut input_down = input_up.clone();
    input_up.set_hop(1.0, 0, 0, &array![0, 0, 0], None);
    input_down.set_hop(2.0, 0, 0, &array![0, 0, 0], None);
    // A y-directed dipole is odd under the mirror fixing both sites.
    input_up.rmatrix[[0, 1, 0, 4]] = Complex64::new(0.125, 0.0);
    input_up.rmatrix[[0, 1, 4, 0]] = Complex64::new(0.125, 0.0);
    // Seed two distinct offsite shells in both channels, with Hermitian R/-R partners.
    for (shell, hop, dipole) in [(1, 0.23, 0.13), (2, -0.31, 0.17)] {
        let r = array![0, shell, 0];
        for (model, site, hop, dipole) in [
            (&mut input_up, 0, hop, Complex64::new(dipole, 0.07)),
            (
                &mut input_down,
                4,
                hop * 0.7,
                Complex64::new(dipole * 0.6, -0.03),
            ),
        ] {
            model.set_hop(hop, site, site, &r, None);
            let forward = find_R(&model.hamR, &r).unwrap();
            let reverse = find_R(&model.hamR, &(-&r)).unwrap();
            model.rmatrix[[forward, 1, site, 4 - site]] = dipole;
            model.rmatrix[[reverse, 1, 4 - site, site]] = dipole.conj();
        }
    }
    input_up.validate().unwrap();
    input_down.validate().unwrap();
    let original_up = input_up.clone();
    let original_down = input_down.clone();
    let input = CollinearModel::new(input_up, input_down).unwrap();
    let moments = [1.0, -1.0];
    let group = detect_collinear_spin_group(
        &crystal(),
        CollinearMoments::Signed(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    let project = |workers| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| {
                input
                    .symmetrize(
                        &group,
                        &CenterResolvedBasis::new(),
                        &HamiltonianSymmetrizationParameters::default(),
                    )
                    .unwrap()
            })
    };
    let serial = project(1);
    let parallel = project(4);
    let bits = |z: Complex64| (z.re.to_bits(), z.im.to_bits());
    for (channel, serial, parallel) in [
        ("up", serial.up(), parallel.up()),
        ("down", serial.down(), parallel.down()),
    ] {
        assert_eq!(serial.hamR, parallel.hamR, "{channel} hamR");
        assert_eq!(
            serial.ham.mapv(bits),
            parallel.ham.mapv(bits),
            "{channel} H"
        );
        assert_eq!(
            serial.rmatrix.mapv(bits),
            parallel.rmatrix.mapv(bits),
            "{channel} r"
        );
        assert_eq!(
            serial.orb.mapv(f64::to_bits),
            parallel.orb.mapv(f64::to_bits),
            "{channel} orb"
        );
    }
    for (projected, site) in [(serial.up(), 0), (serial.down(), 4)] {
        assert!(projected.rmatrix[[0, 1, 0, 4]].norm() < 1e-10);
        for shell in [1, 2] {
            let index = find_R(&projected.hamR, &array![0, shell, 0]).unwrap();
            assert!(projected.ham[[index, site, site]].norm() > 1e-10);
            assert!(projected.rmatrix[[index, 1, site, 4 - site]].norm() > 1e-10);
        }
    }
    for (input, original) in [(input.up(), &original_up), (input.down(), &original_down)] {
        assert_eq!(input.orb, original.orb);
        assert_eq!(input.ham, original.ham);
        assert_eq!(input.hamR, original.hamR);
        assert_eq!(input.rmatrix.0, original.rmatrix.0);
    }
}
