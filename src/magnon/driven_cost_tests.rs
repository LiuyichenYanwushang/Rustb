use super::StrongCouplingOptions;
use crate::{FloquetDrive, HubbardModel, LightMode, Model, TbError};
use ndarray::array;
use num_complex::Complex;

#[test]
fn fourth_order_work_scales_with_coordination_and_harmonic_cutoff() {
    let drive =
        FloquetDrive::with_modes(5.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.0)])]);
    let mib = 1 << 20;
    // For ±1 the live orbits are one bond and one triple. Adding ±2 gives
    // two bonds, four triples (gap sequences 11, 12, 21, 22) and one quartet
    // (0123, five bonds). Disconnected bondless subsets are not expanded.
    // With z neighbors and c live orbits, the work counts below are
    // z(2H+1) + (z+z²+z³+z⁴) + 2[cB + (B-H(H+1))F], B=(2H+1)(2M+1).
    // F counts sparse Fock candidates per harmonic: 8+2*24=56 for ±1,
    // 2*8+9*24+5*80=632 for ±1,±2 (8/24/80 per bond for 2/3/4 sites).
    for (range, clusters, work) in [(1, 2, [1552, 4588]), (2, 7, [16994, 50286])] {
        let mut electron = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        for r in 1..=range {
            electron.set_hop(0.1, 0, 0, &array![r], None);
        }
        let input = HubbardModel::with_uniform_u(electron, 4.0).unwrap();
        for (h, max_work) in (1..=2).zip(work) {
            let large = range == 2 && h == 2;
            let options = StrongCouplingOptions {
                max_hopping_order: 4,
                hop_harmonic_max: h,
                intermediate_photon_max: 2 * h,
                max_work,
                max_cluster_bytes: if large { 2 * mib } else { mib },
                ..StrongCouplingOptions::default()
            };
            let result = input.floquet_spin_model(&drive, &options).unwrap();
            let diagnostics = result.diagnostics();
            assert_eq!(diagnostics.expanded_clusters, clusters);
            assert_eq!(diagnostics.active_photon_max, 2 * h);
            let directed_bins = (2 * range * (2 * h + 1)) as usize;
            assert_eq!(diagnostics.virtual_terms_visited, directed_bins);
            assert_eq!(diagnostics.nonzero_virtual_terms, directed_bins);
            let work_limited = StrongCouplingOptions {
                max_work: max_work - 1,
                ..options.clone()
            };
            assert!(matches!(
                input.floquet_spin_model(&drive, &work_limited),
                Err(TbError::InvalidMagnonInvariant {
                    invariant: "cluster_work_budget",
                    ..
                })
            ));
            // Logical cluster payload cap only: excludes the harmonic cache,
            // allocator overhead and BLAS scratch. It is not process RSS.
            if large {
                let memory_limited = StrongCouplingOptions {
                    max_cluster_bytes: mib,
                    ..options.clone()
                };
                assert!(matches!(
                    input.floquet_spin_model(&drive, &memory_limited),
                    Err(TbError::InvalidMagnonInvariant {
                        invariant: "cluster_workspace_budget",
                        ..
                    })
                ));
            }
            eprintln!(
                "z={} H={h} M={} clusters={clusters} min_work_count={max_work} \
                 passing_cluster_payload_cap_bytes={} (not wall time or RSS)",
                2 * range,
                2 * h,
                options.max_cluster_bytes
            );
        }
    }
}

#[test]
fn fourth_order_checks_a_reachable_singly_occupied_photon_replica() {
    let mut electron =
        Model::<true, 1>::tb_model(array![[2.0]], array![[0.0], [0.5]], None).unwrap();
    electron.set_hop(0.1, 0, 1, &array![0], None);
    let input = HubbardModel::with_uniform_u(electron, 4.0).unwrap();
    let drive = |carrier| {
        FloquetDrive::with_modes(
            0.25,
            vec![LightMode::new(carrier, array![Complex::new(0.4, 0.0)])],
        )
    };
    let fundamental = drive(1);
    let fourth = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 1,
        intermediate_photon_max: 2,
        min_detuning_ev: 0.6,
        ..StrongCouplingOptions::default()
    };
    // m=0 -> -1 -> -2, with one doublon only at -1. For antiparallel
    // spins the two out-and-back paths add on the diagonal of V R V:
    // -2 t² J_1(0.4)²/(U-W) != 0. The final Nd=0 replica has |mW|=0.5 eV
    // below the guard; all charge detunings in the window are >= 3.5 eV.
    for order in [2, 3] {
        let lower = StrongCouplingOptions {
            max_hopping_order: order,
            ..fourth.clone()
        };
        let spin = input.floquet_spin_model(&fundamental, &lower).unwrap();
        assert_eq!(spin.diagnostics().expanded_clusters, order - 2);
        assert_eq!(spin.diagnostics().nonzero_virtual_terms, 6);
    }
    let error = input.floquet_spin_model(&fundamental, &fourth).unwrap_err();
    assert!(
        matches!(
            &error,
            TbError::InvalidMagnonInvariant {
                invariant: "nonresonant_cluster_state",
                ..
            }
        ),
        "{error}"
    );
    // Same window and threshold; with only the second carrier and H=1,
    // only n=0 survives per hop so the guarded replicas are unreachable.
    let inactive = input.floquet_spin_model(&drive(2), &fourth).unwrap();
    assert_eq!(inactive.diagnostics().active_photon_max, 2);
    assert_eq!(inactive.diagnostics().nonzero_virtual_terms, 2);
}

#[test]
fn fourth_order_rejects_large_two_hop_resolvent_amplitude() {
    let mut electron =
        Model::<true, 1>::tb_model(array![[3.0]], array![[0.0], [1.0 / 3.0], [2.0 / 3.0]], None)
            .unwrap();
    electron.set_hop(0.10, 0, 1, &array![0], None);
    electron.set_hop(0.08, 1, 2, &array![0], None);
    let input = HubbardModel::with_uniform_u(electron, 4.0).unwrap();
    let drive =
        |w| FloquetDrive::with_modes(w, vec![LightMode::new(1, array![Complex::new(0.8, 0.0)])]);
    let near = drive(1.9999995);
    let fourth = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 1,
        intermediate_photon_max: 2,
        min_detuning_ev: 1e-8,
        max_hopping_ratio: 0.25,
        ..StrongCouplingOptions::default()
    };
    // Two successive +1 harmonics can move a particle along both links,
    // leaving one doublon in m=-2. Its 1e-6 eV detuning clears the absolute
    // guard, but a dense Fock / time-DFT check gives |(V R V)_ij|≈1.09e-3 eV,
    // so the second resolvent is not perturbative. One-hop ratios stay <0.022.
    for order in [2, 3] {
        let lower = StrongCouplingOptions {
            max_hopping_order: order,
            ..fourth.clone()
        };
        let spin = input.floquet_spin_model(&near, &lower).unwrap();
        assert!(spin.diagnostics().largest_hopping_ratio < 0.022);
    }
    input.floquet_spin_model(&drive(1.9), &fourth).unwrap();
    // At W=1.9975 eV the largest individual entry is ≈0.2174, but
    // the largest row norm is ≈0.2663. An entrywise guard would miss it.
    for w in [near.omega0_ev, 1.9975] {
        let result = input.floquet_spin_model(&drive(w), &fourth);
        assert!(
            matches!(
                &result,
                Err(TbError::InvalidMagnonInvariant {
                    invariant: "cluster_two_hop_ratio",
                    ..
                })
            ),
            "expected two-hop ratio rejection at W={w}, got {result:?}"
        );
    }
}
