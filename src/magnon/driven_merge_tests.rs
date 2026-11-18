use crate::floquet::{FloquetDrive, LightMode};
use crate::magnon::driven::{EffectiveSpinModel, SpinLengthMapping, StrongCouplingOptions};
use crate::magnon::{Magnon, MagnonReference};
use crate::{HubbardModel, Model};
use ndarray::{Array2, Array3, array};
use num_complex::Complex;
use std::f64::consts::TAU;

const U: f64 = 4.0;
const W: f64 = 3.0;
const T: f64 = 0.16;
const H: isize = 6;
const TOL: f64 = 2e-12;
const A_AMPLITUDE: f64 = 0.24;
const B_AMPLITUDE: f64 = 0.10;
const MODEL_TO_BASELINE: [usize; 2] = [1, 0];

fn dimer() -> (HubbardModel<1>, Magnon<1>) {
    let lat = array![[2.0]];
    // Electronic order is (A, B), with representatives shifted by (+1, +2)
    // relative to the TB2J ones. Hence A(0) -> B(+1) becomes A(0) -> B(+2).
    let mut model = Model::<true, 1>::tb_model(lat.clone(), array![[1.125], [2.5]], None).unwrap();
    model.set_hop(T, 0, 1, &array![1isize], None);

    // Baseline order is (B, A). The delta should add to a pre-existing
    // anisotropic bond, leaving SIA and an unrelated translation untouched.
    let mut ham = Array3::zeros((5, 6, 6));
    ham[[0, 2, 2]] = 0.07;
    ham[[0, 5, 5]] = 0.09;
    ham[[1, 2, 2]] = 0.011;
    ham[[2, 2, 2]] = 0.011;
    for (a, b, j) in [
        (0, 0, -0.031),
        (1, 1, -0.027),
        (2, 2, -0.046),
        (0, 1, 0.003),
        (1, 0, -0.003),
    ] {
        ham[[3, 3 + a, b]] = j;
        ham[[4, b, 3 + a]] = j;
    }
    let baseline = Magnon::new(
        lat,
        array![[0.5], [0.125]],
        array![2.0, 1.25],
        array![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        ham,
        array![[0], [3], [-3], [2], [-2]],
        MagnonReference::collinear()
            .with_rotation_axis(&array![0.0, 1.0, 0.0])
            .unwrap(),
    )
    .unwrap();
    (HubbardModel::with_uniform_u(model, U).unwrap(), baseline)
}

fn drive() -> FloquetDrive {
    FloquetDrive::with_modes(
        W,
        vec![
            LightMode::new(1, array![Complex::new(A_AMPLITUDE, 0.0)]),
            LightMode::new(2, array![Complex::new(0.0, B_AMPLITUDE)]),
        ],
    )
}

fn options() -> StrongCouplingOptions {
    StrongCouplingOptions {
        max_hopping_order: 2,
        hop_harmonic_max: H,
        intermediate_photon_max: H,
        ..Default::default()
    }
}

fn dimer_delta_k() -> f64 {
    // Independent fixed DFT of the two directed electron hops. The physical
    // bond length is (1 + 2.5 - 1.125)*2 = 4.75 (also (2 + .5 - .125)*2).
    let coefficient = |m: isize, direction: f64| {
        (0..256)
            .map(|j| {
                let theta = TAU * f64::from(j) / 256.0;
                let phase = 4.75 * (A_AMPLITUDE * theta.cos() + B_AMPLITUDE * (2.0 * theta).sin());
                Complex::from_polar(T, m as f64 * theta - direction * phase)
            })
            .sum::<Complex<f64>>()
            / 256.0
    };
    let driven = (-H..=H)
        .map(|m| {
            let weights = coefficient(m, 1.0).norm_sqr() + coefficient(m, -1.0).norm_sqr();
            2.0 * weights / (U + m as f64 * W)
        })
        .sum::<f64>();
    driven - 4.0 * T * T / U
}

fn assert_same_basis(actual: &Magnon<1>, baseline: &Magnon<1>) {
    assert_eq!(actual.lat, baseline.lat);
    assert_eq!(actual.orb, baseline.orb);
    assert_eq!(actual.spin, baseline.spin);
    assert_eq!(actual.axis, baseline.axis);
    assert_eq!(actual.reference.q, baseline.reference.q);
    assert_eq!(
        actual.reference.rotation_axis,
        baseline.reference.rotation_axis
    );
    assert_eq!(
        actual.reference.quantization_axis,
        baseline.reference.quantization_axis
    );
}

#[test]
fn dimer_delta_remaps_sites_and_cell_representatives_at_unequal_spin() {
    let (model, baseline) = dimer();
    let merged = model
        .floquet_magnon_delta(
            &drive(),
            &options(),
            &baseline,
            &MODEL_TO_BASELINE,
            SpinLengthMapping::KeepOperatorCoefficients,
        )
        .unwrap();
    merged.validate().unwrap();
    assert_same_basis(&merged, &baseline);

    let delta_k = dimer_delta_k();
    let (s_b, s_a) = (2.0_f64, 1.25_f64);
    // TB2J stores two ordered blocks with a minus sign, on unit vectors.
    let delta_j = -0.5 * delta_k * s_a * s_b;
    assert_eq!(merged.nR(), baseline.nR());
    for (row, r) in merged.hamR.column(0).iter().copied().enumerate() {
        let old = baseline
            .hamR
            .column(0)
            .iter()
            .position(|&v| v == r)
            .unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let static_j = baseline.ham[[old, i, j]];
                let actual = merged.ham[[row, i, j]];
                if (r == 2 && i == j + 3) || (r == -2 && j == i + 3) {
                    assert!(
                        (actual - static_j - delta_j).abs() < TOL,
                        "incorrect J[{i},{j}] at R={r}: {actual} vs {}",
                        static_j + delta_j
                    );
                } else {
                    assert_eq!(actual, static_j, "changed J[{i},{j}] at R={r}");
                }
            }
        }
    }

    // Direct HP expansion of the antiparallel bond: diagonal weight is the
    // other site's spin; pairing weight is sqrt(S_A*S_B). Check the labelled
    // BdG matrix, not just the spectrum, to retain the translated bond phase.
    let q = array![0.137];
    let ab = Complex::from_polar(delta_k * (s_a * s_b).sqrt(), TAU * 2.0 * q[0]);
    let ba = ab.conj();
    let db = Complex::new(delta_k * s_a, 0.0);
    let da = Complex::new(delta_k * s_b, 0.0);
    let z = Complex::new(0.0, 0.0);
    let expected = array![
        [db, z, z, ba],
        [z, da, ab, z],
        [z, ba, db, z],
        [ab, z, z, da]
    ];
    let actual = merged.gen_bdg(&q).unwrap() - baseline.gen_bdg(&q).unwrap();
    for (index, &want) in expected.indexed_iter() {
        assert!(
            (actual[index] - want).norm() < TOL,
            "incorrect delta BdG{index:?}: {} vs {want}",
            actual[index]
        );
    }
}

#[test]
fn zero_drive_is_exactly_baseline_and_merge_requires_explicit_valid_mapping() {
    let (model, baseline) = dimer();
    let merged = model
        .floquet_magnon_delta(
            &FloquetDrive::new(W),
            // One undirected static bond has two directed terms, regardless
            // of the requested nonzero harmonic window.
            &StrongCouplingOptions {
                max_work: 2,
                ..options()
            },
            &baseline,
            &MODEL_TO_BASELINE,
            SpinLengthMapping::KeepOperatorCoefficients,
        )
        .unwrap();
    assert_same_basis(&merged, &baseline);
    assert_eq!(merged.hamR, baseline.hamR);
    assert_eq!(merged.ham, baseline.ham);

    let drive = drive();
    let opts = options();
    for bad in [[0, 1], [1, 1], [1, 2]] {
        assert!(
            model
                .floquet_magnon_delta(
                    &drive,
                    &opts,
                    &baseline,
                    &bad,
                    SpinLengthMapping::KeepOperatorCoefficients,
                )
                .is_err(),
            "accepted incorrect model-to-baseline map {bad:?}"
        );
    }
    assert!(
        model
            .floquet_magnon_delta(
                &drive,
                &opts,
                &baseline,
                &MODEL_TO_BASELINE,
                SpinLengthMapping::SpinHalf,
            )
            .is_err()
    );
}

#[test]
fn cell_map_checks_fractional_positions_before_subtracting_large_integers() {
    let (model, baseline) = dimer();
    let mut electron = model.bare_model().clone();
    // In f64, 2^52 - 0.125 rounds to 2^52; the fractional sites are not equal.
    electron.orb[[0, 0]] = 2.0_f64.powi(52);
    let error = HubbardModel::with_uniform_u(electron, U)
        .unwrap()
        .floquet_magnon_delta(
            &FloquetDrive::new(W),
            &options(),
            &baseline,
            &MODEL_TO_BASELINE,
            SpinLengthMapping::KeepOperatorCoefficients,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        crate::TbError::InvalidMagnonInvariant {
            invariant: "magnon_delta_site_map",
            ..
        }
    ));
}

#[test]
fn cell_map_cancels_common_shift_before_checking_translation_range() {
    let r = isize::MAX - 2047; // exactly representable as f64 on supported targets
    let mut electron =
        Model::<true, 1>::tb_model(array![[1.0]], array![[2048.0], [2048.0]], None).unwrap();
    electron.set_hop(T, 0, 1, &array![r], None);
    let baseline = Magnon::new(
        array![[1.0]],
        array![[0.0], [0.0]],
        array![0.5, 0.5],
        array![[0., 0., 1.], [0., 0., -1.]],
        Array3::zeros((1, 6, 6)),
        array![[0]],
        MagnonReference::collinear(),
    )
    .unwrap();
    let light = FloquetDrive::with_modes(
        W,
        vec![LightMode::new(1, array![Complex::new(0.8 / r as f64, 0.0)])],
    );
    let result = HubbardModel::with_uniform_u(electron, U)
        .unwrap()
        .floquet_magnon_delta(
            &light,
            &options(),
            &baseline,
            &[0, 1],
            SpinLengthMapping::SpinHalf,
        )
        .unwrap();
    assert_eq!(result.hamR, array![[0], [-r], [r]]);
    let forward = result.jtensor(0, 1, 2).unwrap();
    assert!(forward[0][0].abs() > 1e-6);
    assert_eq!(forward, result.jtensor(1, 0, 1).unwrap());
}

#[test]
fn peierls_phase_survives_cancellation_of_large_cell_representatives() {
    let baseline = Magnon::new(
        array![[1.0]],
        array![[0.0], [0.125]],
        array![0.5, 0.5],
        array![[0., 0., 1.], [0., 0., -1.]],
        Array3::zeros((1, 6, 6)),
        array![[0]],
        MagnonReference::collinear(),
    )
    .unwrap();
    let expand = |shift: isize| {
        let mut electron =
            Model::<true, 1>::tb_model(array![[1.0]], array![[shift as f64], [0.125]], None)
                .unwrap();
        electron.set_hop(T, 0, 1, &array![shift], None);
        HubbardModel::with_uniform_u(electron, U)
            .unwrap()
            .floquet_magnon_delta(
                &drive(),
                &options(),
                &baseline,
                &[0, 1],
                SpinLengthMapping::SpinHalf,
            )
            .unwrap()
    };
    let ordinary = expand(0);
    let shifted = expand(1_isize << (isize::BITS - 12).min(52));
    assert!(ordinary.ham[[0, 0, 3]].abs() > 1e-8);
    assert_eq!(shifted.hamR, ordinary.hamR);
    assert_eq!(shifted.ham, ordinary.ham);
}

#[test]
fn lifting_rejects_underflow_in_the_weighted_interaction() {
    let (template, mut baseline) = dimer();
    let mut electron = template.bare_model().clone();
    electron.set_hop(1e-12, 0, 1, &array![1], None);
    let model = HubbardModel::with_uniform_u(electron, U).unwrap();
    // S_i*S_j and half that product are normal, but ΔK*S_i*S_j/2 is zero
    // in f64. Do not silently return the unchanged TB2J model.
    for spins in [
        array![2.0_f64.powi(-510), 2.0_f64.powi(-510)],
        array![1e200, 1e-200],
    ] {
        baseline.spin = spins;
        let error = model
            .floquet_magnon_delta(
                &drive(),
                &options(),
                &baseline,
                &MODEL_TO_BASELINE,
                SpinLengthMapping::KeepOperatorCoefficients,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            crate::TbError::InvalidMagnonInvariant {
                invariant: "spin_lifting_scale",
                ..
            }
        ));
    }
}

// One isolated four-site motif per cell. The electronic cell labels and the
// TB2J row order are intentionally different from the motif's vertex order.
const LOOP_CELLS: [isize; 4] = [0, 1, 1, 0];
const LOOP_MAP: [usize; 4] = [2, 0, 3, 1];
const LOOP_SPINS: [f64; 4] = [1.25, 1.75, 0.75, 2.0];
const LOOP_AXES: [[f64; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
];

fn loop_energy(raw: &EffectiveSpinModel<1>, directions: &[[f64; 3]; 4]) -> [f64; 3] {
    // Electronic strong coupling supplies only the raw coefficients. Evaluate
    // the ordinary scalar spin polynomial here; no LSWT reduction is shared.
    // Each translated copy of the isolated motif contains each orbital once.
    let spin = |i: usize| directions[LOOP_MAP[i]].map(|x| LOOP_SPINS[LOOP_MAP[i]] * x);
    let dot = |a: [f64; 3], b: [f64; 3]| a.into_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let pair = raw
        .pairs()
        .iter()
        .map(|p| p.exchange_ev * dot(spin(p.i), spin(p.j)))
        .sum();
    let chirality = raw
        .chiralities()
        .iter()
        .map(|term| {
            let [a, b, c] = term.sites.map(|site| spin(site.orbital));
            let cross = [
                b[1] * c[2] - b[2] * c[1],
                b[2] * c[0] - b[0] * c[2],
                b[0] * c[1] - b[1] * c[0],
            ];
            term.coefficient_ev * dot(a, cross)
        })
        .sum();
    let quartic = raw
        .quartics()
        .iter()
        .map(|term| {
            let [[a, b], [c, d]] = term.pairs.map(|p| p.map(|site| spin(site.orbital)));
            term.coefficient_ev * dot(a, b) * dot(c, d)
        })
        .sum();
    [pair, chirality, quartic]
}

fn loop_mixed_hessian(raw: &EffectiveSpinModel<1>) -> [Array2<f64>; 3] {
    let mut out = std::array::from_fn(|_| Array2::zeros((12, 12)));
    for i in 0..12 {
        for j in i + 1..12 {
            if i / 3 == j / 3 {
                continue;
            }
            // Unconstrained directions: multilinearity on distinct sites makes
            // the signed four-point difference exact even at unit step.
            for s in [-1.0, 1.0] {
                for t in [-1.0, 1.0] {
                    let mut directions = LOOP_AXES;
                    directions[i / 3][i % 3] += s;
                    directions[j / 3][j % 3] += t;
                    for (h, energy) in out.iter_mut().zip(loop_energy(raw, &directions)) {
                        let contribution = 0.25 * s * t * energy;
                        h[[i, j]] += contribution;
                        h[[j, i]] += contribution;
                    }
                }
            }
        }
    }
    // A collinear dot product has no first-order transverse variation, so
    // only one dot at a time fluctuates to quadratic order. Extend its xx
    // curvature isotropically to zz for the TB2J representation. The raw zz
    // derivative also contains a product of two radial variations, which is
    // fourth order for fixed-length spins and does not belong in that tensor.
    for i in 0..4 {
        for j in 0..4 {
            let transverse = out[2][[3 * i, 3 * j]];
            out[2][[3 * i + 2, 3 * j + 2]] = transverse;
        }
    }
    out
}

#[test]
fn driven_triangle_and_ring_replace_static_curvature_at_unequal_spin() {
    let lat = array![[2.0]];
    let electronic_orb = [0.125, 0.375, 0.625, 0.875];
    let mut electron = Model::<true, 1>::tb_model(
        lat.clone(),
        Array2::from_shape_fn((4, 1), |(i, _)| electronic_orb[i]),
        None,
    )
    .unwrap();
    for (i, j, magnitude, phase) in [
        (0, 1, 0.16, 0.0),
        (1, 2, 0.13, 0.0),
        (2, 3, 0.15, 0.0),
        (3, 0, 0.12, 0.0),
        // Magnetic flux through the two triangles, with a real ring product.
        (2, 0, 0.14, 0.53),
    ] {
        electron.set_hop(
            Complex::from_polar(magnitude, phase),
            i,
            j,
            &array![LOOP_CELLS[j] - LOOP_CELLS[i]],
            None,
        );
    }
    let model = HubbardModel::with_uniform_u(electron, U).unwrap();
    let opts = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 1,
        intermediate_photon_max: 2,
        ..Default::default()
    };
    let drive =
        FloquetDrive::with_modes(W, vec![LightMode::new(1, array![Complex::new(0.32, 0.0)])]);
    let dark = model
        .floquet_spin_model(&FloquetDrive::new(W), &opts)
        .unwrap();
    let light = model.floquet_spin_model(&drive, &opts).unwrap();
    assert_eq!(light.diagnostics().active_photon_max, 2);

    let dark_components = loop_mixed_hessian(&dark);
    let light_components = loop_mixed_hessian(&light);
    for (term, label) in [(1, "chirality"), (2, "quartic")] {
        let norm = |h: &Array2<f64>| h.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            norm(&dark_components[term]) > 1e-6,
            "missing static {label}"
        );
        assert!(
            norm(&(&light_components[term] - &dark_components[term])) > 1e-6,
            "drive does not change {label} curvature"
        );
    }
    let total = |[pair, chi, ring]: [Array2<f64>; 3]| pair + chi + ring;
    let dark_hessian = total(dark_components);
    let delta_hessian = total(light_components) - &dark_hessian;
    let mut orb = Array2::zeros((4, 1));
    let mut cell = [0; 4];
    for (i, j) in LOOP_MAP.into_iter().enumerate() {
        orb[[j, 0]] = electronic_orb[i];
        cell[j] = LOOP_CELLS[i];
    }

    // The baseline already includes the dark polynomial's LP curvature,
    // plus unrelated SIA and inter-cell couplings. For an ordered TB2J sum
    // the mixed energy derivative is -2J for each distinct site pair.
    let tensor = |h: &Array2<f64>, r, i: usize, j: usize| {
        if i / 3 != j / 3 && r == cell[j / 3] - cell[i / 3] {
            -0.5 * h[[i, j]]
        } else {
            0.0
        }
    };
    let translations = [0, 3, -3, 1, -1];
    let mut ham = Array3::from_shape_fn((5, 12, 12), |(r, i, j)| {
        tensor(&dark_hessian, translations[r], i, j)
    });
    ham[[0, 2, 2]] = 0.07;
    ham[[1, 5, 5]] = 0.011;
    ham[[2, 5, 5]] = 0.011;
    let baseline = Magnon::new(
        lat,
        orb,
        ndarray::Array1::from_vec(LOOP_SPINS.to_vec()),
        Array2::from_shape_fn((4, 3), |(i, a)| LOOP_AXES[i][a]),
        ham,
        Array2::from_shape_fn((5, 1), |(i, _)| translations[i]),
        MagnonReference::collinear(),
    )
    .unwrap();
    let merged = model
        .floquet_magnon_delta(
            &drive,
            &opts,
            &baseline,
            &LOOP_MAP,
            SpinLengthMapping::KeepOperatorCoefficients,
        )
        .unwrap();
    assert_same_basis(&merged, &baseline);
    assert_eq!(merged.nR(), translations.len());
    for ((row, i, j), &actual) in merged.ham.indexed_iter() {
        let r = merged.hamR[[row, 0]];
        let old = translations.iter().position(|&v| v == r).unwrap();
        let expected = baseline.ham[[old, i, j]] + tensor(&delta_hessian, r, i, j);
        assert!(
            (actual - expected).abs() < TOL,
            "signed J[{i},{j}] at R={r}: {actual:e} vs {expected:e}"
        );
    }

    // Independently normalize the canonical fluctuation coordinates (x,p):
    // n = (x/sqrt(S), sign*p/sqrt(S), sign*(1-(x²+p²)/(2S))).
    // Fourier-transform the energy Hessian in these coordinates, then change basis
    // from (x,p) to (a_q, a^†_{-q}); compare the full labelled matrix.
    let q = 0.137;
    let mut canonical = Array2::<Complex<f64>>::zeros((8, 8));
    let mut oscillator = Array2::<Complex<f64>>::zeros((8, 8));
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    for (i, sign) in LOOP_AXES.iter().map(|axis| axis[2]).enumerate() {
        let shifted_energy = |step| {
            let mut directions = LOOP_AXES;
            directions[i][2] += step;
            let light = loop_energy(&light, &directions).into_iter().sum::<f64>();
            let dark = loop_energy(&dark, &directions).into_iter().sum::<f64>();
            light - dark
        };
        let gradient = 0.5 * (shifted_energy(1.0) - shifted_energy(-1.0));
        oscillator[[2 * i, i]] = Complex::new(s2, 0.0);
        oscillator[[2 * i, i + 4]] = Complex::new(s2, 0.0);
        oscillator[[2 * i + 1, i]] = Complex::new(0.0, -s2);
        oscillator[[2 * i + 1, i + 4]] = Complex::new(0.0, s2);
        for (a, x) in [1.0, sign].into_iter().enumerate() {
            canonical[[2 * i + a, 2 * i + a]] = Complex::new(-sign * gradient / LOOP_SPINS[i], 0.0);
            for j in (0..4).filter(|&j| j != i) {
                let weight = (LOOP_SPINS[i] * LOOP_SPINS[j]).sqrt();
                let phase = TAU * q * (cell[j] - cell[i]) as f64;
                for (b, y) in [1.0, LOOP_AXES[j][2]].into_iter().enumerate() {
                    let curvature = x * delta_hessian[[3 * i + a, 3 * j + b]] * y / weight;
                    canonical[[2 * i + a, 2 * j + b]] =
                        Complex::new(curvature, 0.0) * Complex::from_polar(1.0, phase);
                }
            }
        }
    }
    let expected = oscillator
        .t()
        .mapv(|z| z.conj())
        .dot(&canonical)
        .dot(&oscillator);
    let actual = merged.gen_bdg(&array![q]).unwrap() - baseline.gen_bdg(&array![q]).unwrap();
    for (index, &want) in expected.indexed_iter() {
        assert!(
            (actual[index] - want).norm() < TOL,
            "delta BdG{index:?}: {} vs {want}",
            actual[index]
        );
    }
}
