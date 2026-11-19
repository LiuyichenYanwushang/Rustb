use crate::TbError;
use crate::floquet::{FloquetDrive, LightMode};
use crate::magnon::{Magnon, MagnonReference, MagnonSolve};
use ndarray::{Array3, array};
use num_complex::Complex;
use std::f64::consts::TAU;

fn fourier_weight(m: isize, phase: impl Fn(f64) -> f64) -> f64 {
    // Direct time quadrature, independent of the Peierls coefficient backend.
    const N: usize = 512;
    let c = (0..N)
        .map(|k| {
            let theta = TAU * k as f64 / N as f64;
            Complex::from_polar(1.0, m as f64 * theta - phase(theta))
        })
        .sum::<Complex<f64>>()
        / N as f64;
    c.norm_sqr()
}

fn oracle_factor(u: f64, omega: f64, cut: isize, phase: impl Fn(f64) -> f64) -> f64 {
    let mut factor = fourier_weight(0, &phase);
    // For the fixture amplitudes the weight above |m|=16 is negligible.
    // Outside the requested window the virtual denominator is the bare U.
    for m in 1..=16 {
        let weight = fourier_weight(m, &phase) + fourier_weight(-m, &phase);
        let photon = m as f64 * omega;
        let ratio = if m <= cut {
            // Average the U +/- mW denominators for the two doublon directions.
            (u / (u + photon) + u / (u - photon)) / 2.0
        } else {
            1.0
        };
        factor += weight * ratio;
    }
    factor
}

#[test]
fn superexchange_projects_each_bond_and_is_even_under_time_reversal() {
    let mut ham = Array3::zeros((3, 6, 6));
    ham[[0, 2, 2]] = 0.03;
    ham[[0, 3, 3]] = 0.02;
    let mixed = [
        [0.07, 0.005, 0.004],
        [-0.005, 0.04, 0.0],
        [0.004, 0.0, 0.01],
    ];
    for (a, row) in mixed.into_iter().enumerate() {
        for (b, value) in row.into_iter().enumerate() {
            // (0 -> 1, R=0) is along x; (1 -> 0, R=(0,1)) is along y.
            // Both Cartesian bond lengths are one despite the skew lattice.
            ham[[0, a, 3 + b]] = value;
            ham[[0, 3 + b, a]] = value;
            ham[[1, 3 + a, b]] = value;
            ham[[2, b, 3 + a]] = value;
        }
    }
    let magnon = Magnon::<2>::new(
        array![[2.0, 0.0], [1.0, 1.0]],
        array![[0.0, 0.0], [0.5, 0.0]],
        array![1.0, 1.5],
        array![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ham,
        array![[0, 0], [0, 1], [0, -1]],
        MagnonReference::collinear(),
    )
    .unwrap();
    let (u, omega, cut) = (4.0, 3.0, 3);
    let check = |modes, x_factor: f64, y_factor: f64| {
        let drive = FloquetDrive::with_modes(omega, modes);
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| magnon.floquet_superexchange(&drive, u, cut).unwrap())
        };
        let dressed = run(1);
        assert_eq!(dressed.ham, run(4).ham);
        assert_eq!(dressed.hamR, magnon.hamR);
        let mut expected = magnon.ham.clone();
        for a in 0..3 {
            // trace(mixed)/3 = 0.04. Shift only the isotropic component,
            // without rescaling the symmetric anisotropy or antisymmetric DM.
            let dx = 0.04 * (x_factor - 1.0);
            let dy = 0.04 * (y_factor - 1.0);
            expected[[0, a, 3 + a]] += dx;
            expected[[0, 3 + a, a]] += dx;
            expected[[1, 3 + a, a]] += dy;
            expected[[2, a, 3 + a]] += dy;
        }
        for (idx, &want) in expected.indexed_iter() {
            let got = dressed.ham[idx];
            assert!((got - want).abs() < 2e-12, "{idx:?}: {got} != {want}");
        }
    };
    let (a, b) = (0.8, 0.45);
    let x_phase = |theta: f64| a * theta.cos();
    let two_colour = |theta: f64| x_phase(theta) + b * (2.0 * theta).sin();
    let linear_factor = oracle_factor(u, omega, cut, x_phase);
    let coherent_factor = oracle_factor(u, omega, cut, two_colour);
    assert!((linear_factor - 1.0).abs() > 0.05);
    assert!((fourier_weight(1, two_colour) - fourier_weight(-1, two_colour)).abs() > 0.02);
    let ax = Complex::new(a, 0.0);
    check(
        vec![LightMode::new(1, array![ax, 0.0.into()])],
        linear_factor,
        1.0,
    );
    for helicity in [-1.0, 1.0] {
        let ay = Complex::new(0.0, a * helicity);
        let circular = LightMode::new(1, array![ax, ay]);
        check(vec![circular.clone()], linear_factor, linear_factor);
        let second = LightMode::new(2, array![Complex::new(0.0, b * helicity), 0.0.into()]);
        check(vec![circular, second], coherent_factor, linear_factor);
    }
}

#[test]
fn superexchange_chain_spectrum_and_active_resonance() {
    let (j, spin) = (0.025, 1.5);
    let mut ham = Array3::zeros((3, 3, 3));
    for a in 0..3 {
        ham[[1, a, a]] = j;
        ham[[2, a, a]] = j;
    }
    let mut magnon = Magnon::<1>::new(
        array![[1.0]],
        array![[0.0]],
        array![spin],
        array![[0.0, 0.0, 1.0]],
        ham,
        array![[0], [1], [-1]],
        MagnonReference::collinear(),
    )
    .unwrap();
    let (omega, amp) = (3.0, 0.7);
    let drive = FloquetDrive::with_modes(omega, vec![LightMode::new(1, array![amp.into()])]);
    // U=W is allowed when the resonant sideband is outside the retained window.
    for (u, cut) in [(4.0, 3), (omega, 0)] {
        let dressed = magnon.floquet_superexchange(&drive, u, cut).unwrap();
        let factor = oracle_factor(u, omega, cut, |theta| amp * theta.cos());
        for q in [0.0, 0.125, 0.37, 0.5] {
            let want = 4.0 * j * factor * (1.0 - (TAU * q).cos()) / spin;
            let got = dressed.magnon_band_onek(&array![q]).unwrap()[0];
            assert!((got - want).abs() < 2e-12, "q={q}: {got} != {want}");
        }
    }
    assert!(matches!(
        magnon.floquet_superexchange(&drive, omega, 1),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "nonresonant_charge_state",
            ..
        })
    ));

    let static_phase = LightMode::new(0, array![Complex::new(1.25, -0.2)]);
    for modes in [
        vec![],
        vec![LightMode::new(1, array![0.0.into()])],
        vec![static_phase],
    ] {
        let drive = FloquetDrive::with_modes(omega, modes);
        let dressed = magnon.floquet_superexchange(&drive, omega, 1).unwrap();
        assert_eq!(dressed.ham, magnon.ham);
    }

    // A cancelled fundamental cannot spuriously hit the m=1 pole. A DC
    // amplitude may be huge but has no effect on the exchange correction.
    let second = LightMode::new(2, array![0.4.into()]);
    let only_second = FloquetDrive::with_modes(omega, vec![second.clone()]);
    for amplitude in [0.7, 200.0] {
        let grouped = FloquetDrive::with_modes(
            omega,
            vec![
                LightMode::new(1, array![amplitude.into()]),
                LightMode::new(-1, array![(-amplitude).into()]),
                LightMode::new(0, array![f64::MAX.into()]),
                second.clone(),
            ],
        );
        assert_eq!(
            magnon
                .floquet_superexchange(&grouped, omega, 4)
                .unwrap()
                .ham,
            magnon
                .floquet_superexchange(&only_second, omega, 4)
                .unwrap()
                .ham
        );
    }

    let huge = 2.0_f64.powi(54);
    let residual_drive = FloquetDrive::with_modes(
        omega,
        [huge, 1.0, -huge]
            .into_iter()
            .map(|a| LightMode::new(1, array![a.into()]))
            .collect(),
    );
    assert!(matches!(
        magnon.floquet_superexchange(&residual_drive, omega, 1),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "nonresonant_charge_state",
            ..
        })
    ));

    let small = 2.0_f64.powi(-54);
    for r in [1, 2] {
        for (a, j) in [1.0, small, -1.0].into_iter().enumerate() {
            magnon.ham[[r, a, a]] = j;
        }
    }
    let dressed_trace = magnon.floquet_superexchange(&drive, 4.0, 3).unwrap();
    let factor = oracle_factor(4.0, omega, 3, |theta| amp * theta.cos());
    let want = small * (1.0 + (factor - 1.0) / 3.0);
    assert!((dressed_trace.ham[[1, 1, 1]] / want - 1.0).abs() < 2e-14);

    for (r, j) in [(1, 100.0), (2, 100.00009)] {
        for a in 0..3 {
            magnon.ham[[r, a, a]] = j;
        }
    }
    let suppress = FloquetDrive::with_modes(
        10.0,
        vec![LightMode::new(1, array![2.404825557695773.into()])],
    );
    let suppressed = magnon.floquet_superexchange(&suppress, 1.0, 12).unwrap();
    assert!((suppressed.ham[[1, 0, 0]] + 0.6472694781).abs() < 1e-8);
    assert_eq!(suppressed.ham[[1, 0, 0]], suppressed.ham[[2, 0, 0]]);

    for a in 0..3 {
        magnon.ham[[1, a, a]] = j;
        magnon.ham[[2, a, a]] = j + if a < 2 { 0.9e-6 } else { -0.9e-6 };
    }
    let mixed = magnon.floquet_superexchange(&drive, 4.0, 3).unwrap();
    for a in 0..3 {
        assert_eq!(mixed.ham[[1, a, a]], mixed.ham[[2, a, a]]);
    }
    // Averaging removes a reciprocity residual, not the physical mean anisotropy.
    assert!((mixed.ham[[1, 0, 0]] - mixed.ham[[1, 2, 2]] - 0.9e-6).abs() < 1e-15);

    let dressed = magnon.floquet_superexchange(&drive, 4.0, 3).unwrap();
    // Translating the only site by an integer cannot change its intercell bonds.
    magnon.orb[[0, 0]] = (1_u64 << 53) as f64;
    assert_eq!(
        magnon.floquet_superexchange(&drive, 4.0, 3).unwrap().ham,
        dressed.ham
    );
}
