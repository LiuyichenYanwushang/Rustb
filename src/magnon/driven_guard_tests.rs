use super::StrongCouplingOptions;
use crate::magnon::{MagnonReference, MagnonSolve};
use crate::{FloquetDrive, HubbardModel, LightMode, Model, TbError};
use ndarray::array;
use num_complex::Complex;

#[test]
fn weak_antiferromagnetic_hopping_still_rejects_parallel_saddle() {
    let mut electron = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
    electron.set_hop(1e-7, 0, 0, &array![1], None);
    let spin = HubbardModel::with_uniform_u(electron, 4.0)
        .unwrap()
        .floquet_spin_model(&FloquetDrive::new(3.0), &StrongCouplingOptions::default())
        .unwrap();
    let magnon = spin
        .magnon(&array![[0., 0., 1.]], MagnonReference::collinear())
        .unwrap();
    // SU(2) gives an exact zero at Γ, but at the zone boundary both
    // curvatures are -2 K = -2e-14 eV, independent of an eigensolver.
    assert_eq!(magnon.magnon_band_onek(&array![0.]).unwrap()[0], 0.0);
    let q = array![0.5];
    let h = magnon.gen_bdg(&q).unwrap();
    assert!((h[[0, 0]].re + 2e-14).abs() < 1e-28);
    assert!(matches!(
        magnon.magnon_band_onek(&q),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "positive_semidefinite_bdg",
            ..
        })
    ));
}

#[test]
fn fourth_order_is_deterministic_and_charges_periodic_expansion_budgets() {
    let bonds = [(0.11, 1), (0.07, 2)];
    let chain = |reverse: bool| {
        let mut model = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        let n = bonds.len();
        for i in 0..n {
            let (t, r) = bonds[if reverse { n - i - 1 } else { i }];
            model.set_hop(Complex::from_polar(t, 0.2), 0, 0, &array![r], None);
        }
        HubbardModel::with_uniform_u(model, 4.0).unwrap()
    };
    let drive =
        FloquetDrive::with_modes(5.0, vec![LightMode::new(1, array![Complex::new(0.5, 0.2)])]);
    let options = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 2,
        intermediate_photon_max: 4,
        ..Default::default()
    };
    let expand = |threads, reverse| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| chain(reverse).floquet_spin_model(&drive, &options).unwrap())
    };
    let one = expand(1, false);
    let four = expand(4, true);
    assert_eq!(one.pairs(), four.pairs());
    assert_eq!(one.chiralities(), four.chiralities());
    assert_eq!(one.quartics(), four.quartics());
    assert_eq!(one.constant_ev().to_bits(), four.constant_ev().to_bits());
    assert!(
        one.chiralities()
            .iter()
            .any(|c| c.coefficient_ev.abs() > 1e-5)
    );
    assert!(one.quartics().iter().any(|c| c.coefficient_ev.abs() > 1e-6));
    let diagnostics = one.diagnostics();
    assert_eq!(diagnostics.virtual_terms_visited, 20); // two directions × two bonds × five bins
    assert!(diagnostics.expanded_clusters > 4);
    assert_eq!(
        diagnostics.expanded_clusters,
        four.diagnostics().expanded_clusters
    );

    let model = chain(false);
    // One root, four neighbors per vertex and depths 1..=4: exactly 340
    // visits fit if the 20 already charged second-order terms are forgotten.
    // That bug would get past enumeration and fail later in the Fock kernel.
    let work_limited = StrongCouplingOptions {
        max_work: 4 + 16 + 64 + 256,
        ..options.clone()
    };
    assert!(matches!(
        model.floquet_spin_model(&drive, &work_limited),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "expansion_work",
            ..
        })
    ));
    let cluster_limited = StrongCouplingOptions {
        max_cluster_bytes: 1,
        ..options.clone()
    };
    assert!(matches!(
        model.floquet_spin_model(&drive, &cluster_limited),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "cluster_workspace_budget",
            ..
        })
    ));
    let second = StrongCouplingOptions {
        max_hopping_order: 2,
        max_work: diagnostics.virtual_terms_visited,
        ..cluster_limited
    };
    model.floquet_spin_model(&drive, &second).unwrap();
}
