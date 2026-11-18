//! Signed, coefficient-level check of the driven intracell square against finite Sambe.
use super::{EffectiveSpinModel, StrongCouplingOptions};
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::array;
use num_complex::Complex;

// Coefficient order: I; pairs and oriented triples in lexicographic order;
// then (01)(23), (02)(13), (03)(12). All sites are in the same cell.
const PAIRS: [[usize; 2]; 6] = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
const TRIPLES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
const QUARTICS: [[[usize; 2]; 2]; 3] = [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]];

// scripts/floquet_spin_reference/square_reference.py: sambe_orders() on the
// weighted fixture. The NumPy Fock/DFT oracle's 16x16 blocks were projected
// onto Pauli/2 operators with a Gram solve (the three quartics overlap).
// The signed chi012, reversed chi012 and cutoff norms are also anchored in
// final-square-reference.json. These are coefficients in t, not finite-t
// propagation results or a comparison of sorted quasienergies.
const SAMBE: [[[f64; 14]; 2]; 2] = [
    // (hop cutoff H, photon cutoff M, DFT grid) = (4, 8, 128)
    [
        [
            -0.551_876_169_629_014,
            0.785_588_853_564_433,
            0.0,
            0.636_326_971_387_191,
            0.502_776_866_281_237,
            0.0,
            0.282_811_987_283_196,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.234_106_403_900_843,
            -0.505_465_093_857_800,
            0.128_710_071_626_902,
            -0.356_974_597_959_100,
            -0.252_840_192_800_842,
            0.131_924_496_617_182,
            -0.123_902_320_437_256,
            0.046_731_200_165_466_544,
            0.084_924_969_531_473,
            -0.031_543_560_111_690,
            -0.045_293_317_083_452,
            0.168_488_084_830_166,
            -0.168_488_084_830_167,
            0.168_488_084_830_167,
        ],
    ],
    // (H, M, DFT grid) = (6, 12, 256), still fourth order in hopping
    [
        [
            -0.551_876_169_628_755,
            0.785_588_853_564_064,
            0.0,
            0.636_326_971_386_892,
            0.502_776_866_281_001,
            0.0,
            0.282_811_987_283_063,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.234_106_403_912_604,
            -0.505_465_093_879_292,
            0.128_710_071_627_127,
            -0.356_974_597_972_979,
            -0.252_840_192_809_334,
            0.131_924_496_616_250,
            -0.123_902_320_439_310,
            0.046_731_200_167_537_69,
            0.084_924_969_535_237,
            -0.031_543_560_113_088,
            -0.045_293_317_085_460,
            0.168_488_084_828_480,
            -0.168_488_084_828_480,
            0.168_488_084_828_480,
        ],
    ],
];

fn coefficients(spin: &EffectiveSpinModel<2>) -> [f64; 14] {
    let mut values = [None; 14];
    values[0] = Some(spin.constant_ev());
    let mut record = |slot: usize, value: f64| assert!(values[slot].replace(value).is_none());
    for p in spin.pairs() {
        assert_eq!(p.cell, [0; 2]);
        let i = PAIRS.iter().position(|&key| key == [p.i, p.j]).unwrap();
        record(1 + i, p.exchange_ev);
    }
    for c in spin.chiralities() {
        assert!(c.sites.iter().all(|site| site.cell == [0; 2]));
        let key = c.sites.map(|site| site.orbital);
        let i = TRIPLES.iter().position(|&sites| sites == key).unwrap();
        record(7 + i, c.coefficient_ev);
    }
    for q in spin.quartics() {
        assert!(q.pairs.iter().flatten().all(|site| site.cell == [0; 2]));
        let key = q.pairs.map(|pair| pair.map(|site| site.orbital));
        let i = QUARTICS.iter().position(|&pairs| pairs == key).unwrap();
        record(11 + i, q.coefficient_ev);
    }
    values.map(|x| x.unwrap_or(0.0))
}

fn frobenius_distance(a: [f64; 14], b: [f64; 14]) -> f64 {
    let d: [f64; 14] = std::array::from_fn(|i| a[i] - b[i]);
    let sum_sq = |x: &[f64]| x.iter().map(|v| v * v).sum::<f64>();
    // Unnormalized Pauli/2 trace Gram: norms squared 16, 3, 3/2;
    // quartic diagonal 9/16 and off-diagonal 3/16. All other overlaps vanish.
    (16.0 * d[0].powi(2)
        + 3.0 * sum_sq(&d[1..7])
        + 1.5 * sum_sq(&d[7..11])
        + (3.0 / 8.0) * sum_sq(&d[11..])
        + (3.0 / 16.0) * d[11..].iter().sum::<f64>().powi(2))
    .sqrt()
}

#[test]
fn driven_weighted_square_matches_signed_finite_sambe() {
    let t = 0.1_f64;
    // Fractional coordinates in a 2x2 cell give unit Cartesian edges, CCW.
    let mut electron = Model::<true, 2>::tb_model(
        array![[2.0, 0.0], [0.0, 2.0]],
        array![[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
        None,
    )
    .unwrap();
    for (i, j, weight) in [(0, 1, 1.0), (1, 2, 0.8), (2, 3, 0.6), (3, 0, 0.9)] {
        electron.set_hop(t * weight, i, j, &array![0, 0], None);
    }
    let input = HubbardModel::with_uniform_u(electron, 4.0).unwrap();
    let run = |h, helicity| {
        // Re[(A, i*helicity*A) exp(-i theta)] = A(cos theta, helicity*sin theta).
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.4, 0.0), Complex::new(0.0, 0.4 * helicity)],
            )],
        );
        let options = StrongCouplingOptions {
            max_hopping_order: 4,
            hop_harmonic_max: h,
            intermediate_photon_max: 2 * h,
            ..StrongCouplingOptions::default()
        };
        let full = input.floquet_spin_model(&drive, &options).unwrap();
        assert_eq!(full.diagnostics().active_photon_max, 2 * h);
        assert_eq!(full.diagnostics().nearest_detuning_ev, Some(1.0));
        let second = input
            .floquet_spin_model(
                &drive,
                &StrongCouplingOptions {
                    max_hopping_order: 2,
                    ..options
                },
            )
            .unwrap();
        let h2 = coefficients(&second);
        let full = coefficients(&full);
        // The square is bipartite, so the cubic term vanishes.
        [h2, std::array::from_fn(|i| full[i] - h2[i])]
    };
    let coarse = run(4, 1.0);
    let fine = run(6, 1.0);
    let reverse = run(4, -1.0);
    let mut reverse_ref = SAMBE[0];
    // Reversal conjugates both blocks; only the imaginary chirality basis is odd.
    for chi in &mut reverse_ref[1][7..11] {
        *chi = -*chi;
    }
    let scales = [t.powi(2), t.powi(4)];
    for (h, helicity, actual, reference) in [
        (4, 1, coarse, SAMBE[0]),
        (6, 1, fine, SAMBE[1]),
        (4, -1, reverse, reverse_ref),
    ] {
        for (o, tolerance) in [2e-14, 5e-13].into_iter().enumerate() {
            for (i, expected) in reference[o].map(|x| x * scales[o]).into_iter().enumerate() {
                let value = actual[o][i];
                assert!(
                    (value - expected).abs() <= tolerance * scales[o],
                    "H={h}, helicity={helicity}, hopping order {}, coefficient {i}: {value:e} vs {expected:e} eV",
                    2 * (o + 1),
                );
            }
        }
    }
    let [cutoff2, cutoff4] = [0, 1].map(|o| frobenius_distance(coarse[o], fine[o]) / scales[o]);
    eprintln!(
        "signed chi012 at t={t} eV: H=4 (+) {:e}, H=6 (+) {:e}, H=4 (-) {:e} eV; \
         H=4 to H=6 coefficient Frobenius changes: order 2 {cutoff2:e}, order 4 {cutoff4:e}",
        coarse[1][7], fine[1][7], reverse[1][7],
    );
    let cutoff_ref = [1.400_789_343_635_553_8e-12, 6.670_038_469_874_075e-11];
    let cutoff_tol = [1e-14, 1e-12];
    for (o, cutoff) in [cutoff2, cutoff4].into_iter().enumerate() {
        assert!(
            (cutoff - cutoff_ref[o]).abs() <= cutoff_tol[o],
            "hopping order {}: H=4 to H=6 cutoff change {cutoff:e} vs {:e}",
            2 * (o + 1),
            cutoff_ref[o],
        );
    }
}
