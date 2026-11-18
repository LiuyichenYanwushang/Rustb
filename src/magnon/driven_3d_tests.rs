//! A driven noncoplanar loop with two layers, not a material bilayer benchmark.
use super::{PeriodicSpinSite, StrongCouplingOptions};
use crate::magnon::{MagnonReference, MagnonSolve};
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::{arr1, array};
use num_complex::Complex64;

const T: f64 = 0.1;
const CELLS: [[isize; 3]; 4] = [[0; 3], [0; 3], [0, 0, 1], [0, 0, 1]];
const TRIPLES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
const QUARTICS: [[[usize; 2]; 2]; 3] = [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]];

fn on_ring<const N: usize>(sites: [PeriodicSpinSite<3>; N]) -> bool {
    let origin = sites[0];
    sites.iter().all(|s| {
        (0..3).all(|a| s.cell[a] - origin.cell[a] == CELLS[s.orbital][a] - CELLS[origin.orbital][a])
    })
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 2e-12,
        "{what}: {actual:e} vs {expected:e} eV"
    );
}

#[test]
fn driven_nonplanar_ring_uses_z_polarization_at_fourth_order() {
    // One unwrapped ring has Cartesian vertices (1,0,3.5), (1,1,3.5),
    // (2,1,4.5), (0,0,4.5): scalar triple product of the three relative
    // vectors is -2. Both layers have an in-plane bond (different directions),
    // and the two interlayer bonds actually cross the z-cell boundary.
    let mut model = Model::<true, 3>::tb_model(
        array![[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
        array![
            [0.25, 0., 0.875],
            [0.25, 0.25, 0.875],
            [0.5, 0.25, 0.125],
            [0., 0., 0.125]
        ],
        None,
    )
    .unwrap();
    for (i, weight) in [1.0, 0.8, 0.6, 0.9].into_iter().enumerate() {
        let j = (i + 1) % 4;
        let r: [isize; 3] = std::array::from_fn(|a| CELLS[j][a] - CELLS[i][a]);
        model.set_hop(T * weight, i, j, &arr1(&r), None);
    }
    // One y-only bridge (orthogonal to the drive) per ring produces real
    // q_y dispersion; q_z probes the phase of translated intra-ring terms.
    model.set_hop(T * 0.5, 1, 0, &array![0, 1, 0], None);
    let input = HubbardModel::with_uniform_u(model, 4.0).unwrap();
    let opts = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 4,
        intermediate_photon_max: 8,
        ..Default::default()
    };

    // Independent 128-point Peierls DFT and dense Fock/Sambe calculation
    // from scripts/floquet_spin_reference/nonplanar_reference.py
    // (H=4, M=8, min |E_Q|=1 eV), which also checks the H=6 sweep.
    // Values are the signed per-ring H4 coefficients in units of t^4;
    // all four chi slots follow increasing orbital order, not spatial order.
    // At order 4 the cut-edge bridge enters only dimers or two-bond wedges.
    // These have at most one driven leg; a shift of time makes its linear
    // drive time-reversal symmetric, so they cannot add either chi or quartics.
    let sambe_chi = [
        -0.152_363_035_479_024,
        -0.152_363_035_479_024,
        0.088_061_277_023_041,
        0.068_251_330_817_347,
    ];
    for (z_sign, ring, a02) in [
        (
            1.0,
            0.162_351_302_870_054,
            Complex64::new(9.460_309_807_891_381e-7, -9.330_077_974_342_236e-6),
        ),
        (
            0.0,
            0.282_014_334_032_602,
            Complex64::new(7.631_633_406_619_847e-6, -7.557_586_847_121_914e-6),
        ),
        (
            -1.0,
            0.162_351_302_870_054,
            Complex64::new(9.709_120_034_957_458e-6, -1.100_989_608_433_093e-6),
        ),
    ] {
        // Re[a exp(-iθ)] = (0.4 cos θ, 0, 0.35 z_sign sin θ). In
        // particular z_sign=0 drops only the third drive component.
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(
                1,
                array![
                    Complex64::new(0.4, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.35 * z_sign)
                ],
            )],
        );
        let spin = input.floquet_spin_model(&drive, &opts).unwrap();
        assert_eq!(spin.diagnostics().active_photon_max, 8);
        assert_eq!(spin.diagnostics().nearest_detuning_ev, Some(1.0));

        let mut chi = [None; 4];
        for c in spin.chiralities() {
            let key = c.sites.map(|s| s.orbital);
            if let Some(i) = TRIPLES
                .iter()
                .position(|&ids| ids == key && on_ring(c.sites))
            {
                assert!(chi[i].replace(c.coefficient_ev).is_none());
            } else {
                assert_close(c.coefficient_ev, 0.0, "off-ring chi");
            }
        }
        for (got, want) in chi.into_iter().zip(sambe_chi) {
            assert_close(got.unwrap_or(0.0), z_sign * want * T.powi(4), "ring chi");
        }

        let mut quartic = [None; 3];
        for c in spin.quartics() {
            let [[a, b], [d, e]] = c.pairs;
            let key = c.pairs.map(|pair| pair.map(|s| s.orbital));
            if let Some(i) = QUARTICS
                .iter()
                .position(|&ids| ids == key && on_ring([a, b, d, e]))
            {
                assert!(quartic[i].replace(c.coefficient_ev).is_none());
            } else {
                assert_close(c.coefficient_ev, 0.0, "off-ring quartic");
            }
        }
        for (got, want) in quartic.into_iter().zip([ring, -ring, ring]) {
            assert_close(got.unwrap(), want * T.powi(4), "ring quartic");
        }

        let axes = array![[0., 0., 1.], [0., 0., -1.], [0., 0., 1.], [0., 0., -1.]];
        let magnon = spin.magnon(&axes, MagnonReference::collinear()).unwrap();
        let q = array![0.19, 0.23, 0.37];
        // Independent BdG baseline: linearize the spin-coherent-state
        // expectation of each Fock/Sambe spin matrix using direct flip matrix
        // elements, then Fourier transform with exp(+2πiq·(R_j-R_i))
        // and local transverse axes (x, ±y). The matrix elements keep the
        // sign and translated phase that a sorted band comparison can lose.
        let bdg = magnon.gen_bdg(&q).unwrap();
        assert_close((bdg[[0, 2]] - a02).norm(), 0.0, "signed finite-q BdG A02");

        if z_sign > 0.0 {
            let b12 = Complex64::new(-0.001_364_904_177_041, 0.001_461_532_267_089);
            assert_close((bdg[[1, 6]] - b12).norm(), 0.0, "signed finite-q BdG B12");
            // The same independent Hessian (with bridge dimer and its four
            // connected wedges) has minimum eigenvalues >4e-4 eV here.
            for (qy, expected) in [
                (
                    0.23,
                    [
                        0.001_350_735_559_627,
                        0.001_506_471_209_088,
                        0.006_168_625_739_720,
                        0.006_326_905_107_307,
                    ],
                ),
                (
                    0.41,
                    [
                        0.001_709_665_346_860,
                        0.001_971_876_119_456,
                        0.006_919_000_979_183,
                        0.007_183_732_729_073,
                    ],
                ),
            ] {
                let bands = magnon.magnon_band_onek(&array![q[0], qy, q[2]]).unwrap();
                assert_eq!(bands.len(), 4);
                for (got, want) in bands.into_iter().zip(expected) {
                    assert_close(got, want, "stable finite-q magnon frequency");
                }
            }
        }
    }
}
