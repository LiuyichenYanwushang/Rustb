#![cfg(feature = "cryspglib")]

use Rustb::{
    Atom, AtomType, CenterResolvedBasis, CollinearMoments, HamiltonianSymmetrizationParameters,
    Model, OrbitalId, SpinTolerances, detect_collinear_spin_group,
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
    let mut up = two_site_model(1.0);
    let mut down = two_site_model(2.0);
    let moments = [1.0, -1.0];
    let group = detect_collinear_spin_group(
        &crystal(),
        CollinearMoments::Signed(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    println!("complete group len {}", group.operations().len());
    let report = Model::<false, 3>::symmetrize_pair_with_collinear_spin_group(
        &mut up,
        &mut down,
        &group,
        &CenterResolvedBasis::new(),
        &HamiltonianSymmetrizationParameters::default(),
    )
    .unwrap();
    println!(
        "full covariance {:?} up {:?}/{:?} down {:?}/{:?}",
        report.pair_covariance_max_absolute,
        onsite(&up, 0),
        onsite(&up, 1),
        onsite(&down, 0),
        onsite(&down, 1)
    );
    assert!(report.pair_covariance_max_absolute < 1e-8);
    assert!((onsite(&up, 0) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert!((onsite(&up, 1) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert!((onsite(&down, 0) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
    assert!((onsite(&down, 1) - Complex64::new(1.5, 0.0)).norm() < 1e-10);
}

#[test]
fn unitary_collinear_subgroup_is_accepted_separately() {
    let mut up = two_site_model(1.0);
    let mut down = two_site_model(3.0);
    let moments = [1.0, -1.0];
    let group = detect_collinear_spin_group(
        &crystal(),
        CollinearMoments::Signed(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    let report = Model::<false, 3>::symmetrize_pair_with_unitary_collinear_spin_group(
        &mut up,
        &mut down,
        &group,
        &CenterResolvedBasis::new(),
        &HamiltonianSymmetrizationParameters::default(),
    )
    .unwrap();
    println!(
        "unitary covariance {:?} up {:?}/{:?} down {:?}/{:?}",
        report.pair_covariance_max_absolute,
        onsite(&up, 0),
        onsite(&up, 1),
        onsite(&down, 0),
        onsite(&down, 1)
    );
    assert!(report.pair_covariance_max_absolute < 1e-8);
    assert!((onsite(&up, 0) - Complex64::new(2.0, 0.0)).norm() < 1e-10);
    assert!((onsite(&down, 0) - Complex64::new(2.0, 0.0)).norm() < 1e-10);
}

#[test]
#[ignore = "loads the external MgFe6Ge6 Wannier90 dataset"]
fn mgfe6ge6_collinear_spin_pair_acceptance() {
    use Rustb::Wannier90;
    use std::time::Instant;

    let base = "/mnt/e/vasp计算结果/altermagnetic/odd_parity_wave/material/MgFe6Ge6/data_new";
    let mut up =
        <Model<false, 3> as Wannier90>::from_hr(&format!("{base}/up/"), "wannier90", 7.62075874)
            .unwrap();
    let mut down =
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
    println!("real complete group {}", group.operations().len());
    let started = Instant::now();
    let report = Model::<false, 3>::symmetrize_pair_with_collinear_spin_group(
        &mut up,
        &mut down,
        &group,
        &CenterResolvedBasis::new(),
        &HamiltonianSymmetrizationParameters::default(),
    )
    .unwrap();
    println!(
        "real full report: change {:e} covariance {:e} centre up {:e} down {:e} elapsed {:?}",
        report.hamiltonian_change_max_absolute,
        report.pair_covariance_max_absolute,
        report.up_centres.max_displacement_angstrom,
        report.down_centres.max_displacement_angstrom,
        started.elapsed()
    );
    assert!(report.pair_covariance_max_absolute < 1e-3);
}

#[test]
fn pair_projection_updates_position_matrices() {
    use Rustb::HasRMatrix;

    let mut up: Model<false, 3, HasRMatrix> = Model::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        Some(vec![
            Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::Fe, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::Fe, [OrbitalId::new(1)]),
        ]),
    )
    .unwrap();
    let mut down = up.clone();
    up.set_hop(1.0, 0, 0, &array![0, 0, 0], None);
    down.set_hop(2.0, 0, 0, &array![0, 0, 0], None);
    up.validate().unwrap();
    down.validate().unwrap();
    let moments = [1.0, -1.0];
    let group = detect_collinear_spin_group(
        &crystal(),
        CollinearMoments::Signed(&moments),
        SpinTolerances::default(),
    )
    .unwrap();
    let report = Model::<false, 3, HasRMatrix>::symmetrize_pair_with_collinear_spin_group(
        &mut up,
        &mut down,
        &group,
        &CenterResolvedBasis::new(),
        &HamiltonianSymmetrizationParameters::default(),
    )
    .unwrap();
    assert!(report.pair_covariance_max_absolute < 1e-8);
}
