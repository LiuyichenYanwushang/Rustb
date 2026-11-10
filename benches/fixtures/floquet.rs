//! Shared synthetic fixtures for Floquet timing and allocation measurements.
//!
//! All orbitals are coupled by complex hopping blocks; increasing the state
//! count does not pad independent copies of a smaller model. These are not a
//! substitute for the user's as-yet unspecified slow model.

use Rustb::{FloquetDrive, FloquetEffectiveOptions, LightMode, Model};
use ndarray::{Array1, Array2, array};
use num_complex::Complex;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub struct Case {
    pub name: &'static str,
    pub nsta: usize,
    pub radius: isize,
    pub order: usize,
    pub harmonic_max: isize,
    pub drive: &'static str,
}

pub const CASES: &[Case] = &[
    Case {
        name: "dense16_r9/mixed_o0_n2",
        nsta: 16,
        radius: 1,
        order: 0,
        harmonic_max: 2,
        drive: "mixed",
    },
    Case {
        name: "dense16_r9/mixed_o1_n2",
        nsta: 16,
        radius: 1,
        order: 1,
        harmonic_max: 2,
        drive: "mixed",
    },
    Case {
        name: "dense16_r9/mixed_o2_n1",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 1,
        drive: "mixed",
    },
    Case {
        name: "dense16_r9/mixed_o2_n2",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 2,
        drive: "mixed",
    },
    Case {
        name: "dense16_r9/mixed_o2_n4",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 4,
        drive: "mixed",
    },
    Case {
        name: "dense16_r9/mixed_o2_n8",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 8,
        drive: "mixed",
    },
    Case {
        name: "dense16_r9/fundamental_o2_n2",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 2,
        drive: "fundamental",
    },
    Case {
        name: "dense16_r9/second_o2_n2",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 2,
        drive: "second",
    },
    Case {
        name: "dense16_r9/zero_o2_n8",
        nsta: 16,
        radius: 1,
        order: 2,
        harmonic_max: 8,
        drive: "zero",
    },
    Case {
        name: "dense4_r9/mixed_o2_n2",
        nsta: 4,
        radius: 1,
        order: 2,
        harmonic_max: 2,
        drive: "mixed",
    },
    Case {
        name: "dense32_r9/mixed_o2_n2",
        nsta: 32,
        radius: 1,
        order: 2,
        harmonic_max: 2,
        drive: "mixed",
    },
    Case {
        name: "dense16_r25/mixed_o2_n2",
        nsta: 16,
        radius: 2,
        order: 2,
        harmonic_max: 2,
        drive: "mixed",
    },
];

impl Case {
    pub fn model(self) -> Model<false, 2> {
        assert!(self.nsta >= 2 && self.radius > 0);
        let nsta = self.nsta;
        let lat = array![[1.1, 0.15], [0.0, 0.95]];
        let orb = Array2::from_shape_fn((nsta, 2), |(i, axis)| {
            let numerator = if axis == 0 {
                i + 1
            } else {
                (7 * i + 3) % (nsta + 1)
            };
            numerator as f64 / (nsta + 1) as f64
        });
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        let onsite = Array1::from_shape_fn(nsta, |i| 0.12 * (i as f64 - 0.5 * (nsta - 1) as f64));
        model.set_onsite(&onsite, None);
        let normalization = (nsta as f64).sqrt();
        for i in 0..nsta {
            for j in (i + 1)..nsta {
                let phase = 0.37 * (i + 1) as f64 + 0.19 * (j + 1) as f64;
                model.set_hop(
                    Complex::new(phase.sin(), 0.7 * phase.cos()) * (0.2 / normalization),
                    i,
                    j,
                    &array![0, 0],
                    None,
                );
            }
        }
        for x in -self.radius..=self.radius {
            for y in -self.radius..=self.radius {
                // Set one representative of each R/-R pair. set_hop writes
                // the reversed, conjugated matrix element automatically.
                if x < 0 || (x == 0 && y <= 0) {
                    continue;
                }
                let scale = 0.3 / (normalization * (1 + x.abs() + y.abs()) as f64);
                for i in 0..nsta {
                    for j in 0..nsta {
                        let re_phase =
                            0.37 * (i + 1) as f64 + 0.19 * (j + 1) as f64 + 0.23 * x as f64
                                - 0.31 * y as f64;
                        let im_phase =
                            0.29 * ((i + 1) * (j + 1)) as f64 - 0.17 * x as f64 + 0.13 * y as f64;
                        model.set_hop(
                            Complex::new(re_phase.sin(), 0.7 * im_phase.cos()) * scale,
                            i,
                            j,
                            &array![x, y],
                            None,
                        );
                    }
                }
            }
        }
        model
    }

    pub fn light(self) -> FloquetDrive {
        let fundamental = array![Complex::new(0.31, 0.07), Complex::new(-0.04, 0.25)];
        let modes = match self.drive {
            "fundamental" => vec![LightMode::new(1, fundamental)],
            "second" => vec![LightMode::new(2, fundamental)],
            "mixed" => vec![
                LightMode::new(1, fundamental),
                LightMode::new(
                    2,
                    array![Complex::new(0.10, -0.06), Complex::new(0.08, 0.09)],
                ),
            ],
            "zero" => Vec::new(),
            _ => panic!("unknown benchmark drive: {}", self.drive),
        };
        FloquetDrive::with_modes(10.0, modes)
    }

    pub fn options(self) -> FloquetEffectiveOptions {
        FloquetEffectiveOptions::new()
            .with_order(self.order)
            .with_harmonic_max(self.harmonic_max)
    }
}

/// Check fixture correctness before measurement and return the sampled E2 max-entry norm.
/// Order-0/1 and zero-drive cases return zero.
pub fn validate_case(case: Case) -> f64 {
    use Rustb::Gauge;

    let model = case.model();
    model.validate().expect("invalid Floquet benchmark input");
    assert_eq!(model.nsta(), case.nsta, "{}: input state count", case.name);
    for ((r, i, j), hopping) in model.ham.indexed_iter() {
        if i != j {
            assert!(
                hopping.norm() > 0.0,
                "{}: missing interorbital input link at ({r}, {i}, {j})",
                case.name
            );
        }
    }

    // Builder insertion order is not part of the fixture contract: compare
    // the input support as a set, then check the row count so duplicates
    // cannot pass as the advertised full square.
    let square: BTreeSet<[isize; 2]> = (-case.radius..=case.radius)
        .flat_map(|x| (-case.radius..=case.radius).map(move |y| [x, y]))
        .collect();
    let rows: BTreeSet<[isize; 2]> = model
        .hamR
        .outer_iter()
        .map(|row| [row[0], row[1]])
        .collect();
    assert_eq!(
        model.hamR.nrows(),
        square.len(),
        "{}: input hamR row count",
        case.name
    );
    assert_eq!(
        rows.len(),
        model.hamR.nrows(),
        "{}: duplicate input hamR rows",
        case.name
    );
    assert_eq!(
        rows, square,
        "{}: input hamR is not the advertised full square",
        case.name
    );

    let drive = case.light();
    let options = case.options();
    let effective = model
        .floquet_effective_model(&drive, Some(&options))
        .expect("Floquet benchmark construction failed");
    effective
        .validate()
        .expect("invalid Floquet benchmark output");
    assert_eq!(
        effective.nsta(),
        case.nsta,
        "{}: output state count",
        case.name
    );
    assert_eq!(effective.lat, model.lat, "{}: lattice changed", case.name);
    assert_eq!(effective.orb, model.orb, "{}: orbitals changed", case.name);
    for (label, checked) in [("input", &model), ("output", &effective)] {
        assert!(
            checked
                .ham
                .iter()
                .all(|z| z.re.is_finite() && z.im.is_finite()),
            "{}: non-finite {label} hopping",
            case.name
        );
        for (r, row) in checked.hamR.outer_iter().enumerate() {
            let opposite = checked
                .hamR
                .outer_iter()
                .position(|other| other[0] == -row[0] && other[1] == -row[1])
                .expect("Floquet fixture support must contain -R");
            for i in 0..case.nsta {
                for j in 0..case.nsta {
                    assert_eq!(
                        checked.ham[[r, i, j]],
                        checked.ham[[opposite, j, i]].conj(),
                        "{}: {label} is not Hermitian at ({r}, {i}, {j})",
                        case.name
                    );
                }
            }
        }
    }

    // A full square's double/triple Minkowski sum is a full square, even
    // when the drive vanishes or its odd harmonics are exactly zero.
    let radius = case.radius
        * if case.harmonic_max > 0 {
            case.order as isize + 1
        } else {
            1
        };
    let expected: Vec<_> = std::iter::once([0, 0])
        .chain(
            (-radius..=radius)
                .flat_map(|x| (-radius..=radius).map(move |y| [x, y]))
                .filter(|row| *row != [0, 0]),
        )
        .collect();
    let actual: Vec<_> = effective
        .hamR
        .outer_iter()
        .map(|row| [row[0], row[1]])
        .collect();
    assert_eq!(
        effective.hamR.nrows(),
        ((2 * radius + 1) as usize).pow(2),
        "{}: support count",
        case.name
    );
    assert_eq!(
        actual, expected,
        "{}: exact origin-first lexicographic support",
        case.name
    );

    let points = array![[0.137, 0.283], [0.419, 0.067], [0.731, 0.593]];
    let max_bloch_difference = |left: &Model<false, 2>, right: &Model<false, 2>| {
        left.validate().expect("invalid Floquet comparison model");
        right.validate().expect("invalid Floquet comparison model");
        let lhs = left.gen_ham_batch(&points, Gauge::Lattice);
        let rhs = right.gen_ham_batch(&points, Gauge::Lattice);
        assert_eq!(lhs.dim(), rhs.dim(), "{}: Bloch shape mismatch", case.name);
        lhs.iter()
            .zip(rhs.iter())
            .map(|(a, b)| {
                let difference = (a - b).norm();
                assert!(
                    difference.is_finite(),
                    "{}: non-finite Bloch difference",
                    case.name
                );
                difference
            })
            .fold(0.0_f64, f64::max)
    };

    if case.drive == "zero" {
        // Compare all real-space blocks, including exact-zero support padding.
        for (r, row) in effective.hamR.outer_iter().enumerate() {
            let original = model.hamR.outer_iter().position(|other| other == row);
            for i in 0..case.nsta {
                for j in 0..case.nsta {
                    let expected =
                        original.map_or(Complex::new(0.0, 0.0), |s| model.ham[[s, i, j]]);
                    assert_eq!(
                        effective.ham[[r, i, j]],
                        expected,
                        "{}: zero drive changed static hopping",
                        case.name
                    );
                }
            }
        }
        assert!(
            max_bloch_difference(&effective, &model) < 1e-12,
            "{}: zero drive changed H(k)",
            case.name
        );
        return 0.0;
    }

    let e2_max_entry = if case.order == 2 {
        let first_options = case.options().with_order(1);
        let first = model
            .floquet_effective_model(&drive, Some(&first_options))
            .expect("Floquet order-1 comparison failed");
        let correction = max_bloch_difference(&effective, &first);
        assert!(
            correction > 1e-10,
            "{}: E2 correction vanished ({correction:e})",
            case.name
        );
        correction
    } else {
        0.0
    };

    if case.drive == "second" {
        // Pure l=2 has no odd harmonics: N=1 must give only the time average,
        // while the N=2 case above must retain a genuine E2 correction.
        let odd_options = case.options().with_order(2).with_harmonic_max(1);
        let average_options = case.options().with_order(0);
        let odd = model
            .floquet_effective_model(&drive, Some(&odd_options))
            .expect("Floquet odd-harmonic control failed");
        let average = model
            .floquet_effective_model(&drive, Some(&average_options))
            .expect("Floquet time-average control failed");
        let correction = max_bloch_difference(&odd, &average);
        assert!(
            correction < 1e-12,
            "{}: odd-harmonic control contributed ({correction:e})",
            case.name
        );
    }
    e2_max_entry
}
