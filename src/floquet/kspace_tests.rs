//! Fourier-space cross-checks of the streamed effective model.

use super::coefficients::BESSEL_CUTOFF_MARGIN;
use super::{
    FloquetDrive, FloquetEffectiveOptions, LightMode, PeierlsFourierMethod,
    drive_has_ac_components, effective_cache_max,
};
use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
use crate::model::{HasRMatrix, RMatrixData};
use crate::{Gauge, Model};
use ndarray::{Array1, Array2, Array3, Axis, arr1, array};
use num_complex::Complex;
use std::collections::BTreeSet;

/// Photon energy shared by every drive in this module.
const OMEGA: f64 = 6.0;

/// Coefficient backend the effective-model driver uses internally.
const BESSEL: PeierlsFourierMethod = PeierlsFourierMethod::Bessel {
    cutoff_margin: BESSEL_CUTOFF_MARGIN,
};

// ---------------------------------------------------------------------------
// Model fixtures: two or three orbitals and a handful of hopping vectors.
// ---------------------------------------------------------------------------

fn one_dimensional_model() -> Model<false, 1> {
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.0], [0.37]];
    let atoms = vec![
        Atom::with_orbitals(
            arr1(&[0.0]),
            AtomType::C,
            [OrbitalId::new(0), OrbitalId::new(1)],
        ),
        Atom::with_orbitals(arr1(&[0.37]), AtomType::O, [OrbitalId::new(2)]),
    ];
    let mut model = Model::<false, 1>::tb_model(lat, orb, Some(atoms)).unwrap();
    model.orb_projection = vec![OrbProj::s, OrbProj::px, OrbProj::py];
    model.set_onsite(&array![0.21, -0.33, 0.17], None);
    for (row, column, cell, re, im) in [
        (0_usize, 1_usize, 0_isize, 0.41_f64, 0.19_f64),
        (1, 2, 0, -0.23, 0.31),
        (0, 1, 1, -0.83, 0.27),
        (1, 2, 1, 0.47, -0.36),
        (2, 0, -1, 0.29, 0.14),
    ] {
        model.set_hop(Complex::new(re, im), row, column, &arr1(&[cell]), None);
    }
    model
}

fn two_dimensional_model() -> Model<false, 2> {
    let lat = array![[1.0, 0.18], [-0.11, 1.2]];
    let orb = array![[0.02, 0.07], [0.02, 0.07], [0.61, 0.43]];
    let atoms = vec![
        Atom::with_orbitals(
            arr1(&[0.02, 0.07]),
            AtomType::C,
            [OrbitalId::new(0), OrbitalId::new(1)],
        ),
        Atom::with_orbitals(arr1(&[0.61, 0.43]), AtomType::N, [OrbitalId::new(2)]),
    ];
    let mut model = Model::<false, 2>::tb_model(lat, orb, Some(atoms)).unwrap();
    model.orb_projection = vec![OrbProj::s, OrbProj::px, OrbProj::py];
    model.set_onsite(&array![0.18, -0.29, 0.24], None);
    for (row, column, cell, re, im) in [
        (0_usize, 1_usize, [0_isize, 0_isize], 0.44_f64, 0.21_f64),
        (1, 2, [0, 0], -0.23, 0.31),
        (0, 1, [1, 0], -0.81, 0.27),
        (1, 2, [0, 1], 0.47, -0.36),
        (2, 0, [1, 1], 0.29, 0.14),
    ] {
        model.set_hop(Complex::new(re, im), row, column, &arr1(&cell), None);
    }
    model
}

fn three_dimensional_model(z_hop: isize) -> Model<false, 3> {
    let lat = array![[1.0, 0.0, 0.0], [0.13, 0.94, 0.0], [-0.07, 0.11, 1.03]];
    let orb = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.31, 0.22, 0.13]];
    let atoms = vec![
        Atom::with_orbitals(
            arr1(&[0.0, 0.0, 0.0]),
            AtomType::C,
            [OrbitalId::new(0), OrbitalId::new(1)],
        ),
        Atom::with_orbitals(arr1(&[0.31, 0.22, 0.13]), AtomType::O, [OrbitalId::new(2)]),
    ];
    let mut model = Model::<false, 3>::tb_model(lat, orb, Some(atoms)).unwrap();
    model.orb_projection = vec![OrbProj::s, OrbProj::px, OrbProj::py];
    model.set_onsite(&array![0.16, -0.24, 0.31], None);
    for (row, column, cell, re, im) in [
        (
            0_usize,
            1_usize,
            [0_isize, 0_isize, 0_isize],
            0.39_f64,
            0.18_f64,
        ),
        (1, 2, [0, 0, 0], -0.26, 0.29),
        (0, 1, [1, 0, 0], -0.78, 0.24),
        (1, 2, [0, 1, 0], 0.45, -0.33),
        (2, 0, [0, 0, z_hop], 0.31, 0.12),
        (0, 2, [1, -1, 0], 0.18, -0.21),
    ] {
        model.set_hop(Complex::new(re, im), row, column, &arr1(&cell), None);
    }
    model
}

/// Two-mode drive: a fundamental and its second harmonic.
fn two_mode_drive<const DIM: usize>() -> FloquetDrive {
    let fundamental = Array1::from_shape_fn(DIM, |axis| {
        Complex::new(0.19 + 0.04 * axis as f64, 0.06 - 0.02 * axis as f64)
    });
    let second = Array1::from_shape_fn(DIM, |axis| {
        Complex::new(-0.08 + 0.01 * axis as f64, 0.05 + 0.03 * axis as f64)
    });
    FloquetDrive::with_modes(
        OMEGA,
        vec![LightMode::new(1, fundamental), LightMode::new(2, second)],
    )
}

/// A 1D chain with hopping vectors up to |R| = 32: 65 real-space blocks and
/// 4225 block pairs: a support the automatic selector must route to k space.
fn long_range_chain() -> Model<false, 1> {
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.21]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_onsite(&array![-0.4, 0.3], None);
    for cell in 1..=32_isize {
        let re = 0.7 / (cell as f64).powi(2);
        let im = 0.15 / (cell as f64);
        model.set_hop(Complex::new(re, im), 0, 1, &arr1(&[cell]), None);
    }
    model
}

/// The cheapest possible support: origin block plus one hermitian R = +-1 pair.

fn support_vectors<const DIM: usize>(ham_r: &Array2<isize>) -> Vec<[isize; DIM]> {
    ham_r
        .outer_iter()
        .map(|row| {
            let mut vector = [0_isize; DIM];
            for axis in 0..DIM {
                vector[axis] = row[axis];
            }
            vector
        })
        .collect()
}

fn add_vectors<const DIM: usize>(left: [isize; DIM], right: [isize; DIM]) -> [isize; DIM] {
    let mut sum = [0_isize; DIM];
    for axis in 0..DIM {
        sum[axis] = left[axis] + right[axis];
    }
    sum
}

fn row_of<const DIM: usize>(ham_r: &Array2<isize>, vector: &[isize; DIM]) -> Option<usize> {
    (0..ham_r.nrows()).find(|&row| (0..DIM).all(|axis| ham_r[[row, axis]] == vector[axis]))
}

/// The documented effective-model support: the input blocks, plus the pairwise
/// Minkowski layer for order >= 1 and the triple layer for order >= 2, all
/// present whenever harmonic_max > 0. The origin row comes first; the rest keep
/// lexicographic order.
fn documented_support<const DIM: usize>(
    model: &Model<false, DIM>,
    harmonic_max: isize,
    order: usize,
) -> Vec<[isize; DIM]> {
    let primitive = support_vectors::<DIM>(&model.hamR);
    let mut vectors = BTreeSet::<[isize; DIM]>::new();
    for vector in &primitive {
        vectors.insert(*vector);
    }
    if harmonic_max > 0 && order >= 1 {
        let mut pairs = BTreeSet::<[isize; DIM]>::new();
        for left in &primitive {
            for right in &primitive {
                pairs.insert(add_vectors(*left, *right));
            }
        }
        vectors.extend(pairs.iter().copied());
        if order >= 2 {
            for pair in &pairs {
                for right in &primitive {
                    vectors.insert(add_vectors(*pair, *right));
                }
            }
        }
    }
    let zero = [0_isize; DIM];
    let mut ordered = Vec::with_capacity(vectors.len());
    ordered.push(zero);
    ordered.extend(vectors.into_iter().filter(|vector| *vector != zero));
    ordered
}

#[track_caller]
fn assert_metadata_preserved<const DIM: usize>(
    context: &str,
    input: &Model<false, DIM>,
    output: &Model<false, DIM>,
) {
    assert_eq!(
        output.nsta(),
        input.nsta(),
        "{context}: state count changed"
    );
    assert_eq!(output.hamR.ncols(), DIM, "{context}: support width changed");
    assert_eq!(output.lat, input.lat, "{context}: lattice changed");
    assert_eq!(
        output.orb, input.orb,
        "{context}: orbital positions changed"
    );
    assert_eq!(
        output.orb_projection, input.orb_projection,
        "{context}: orbital projections changed"
    );
    assert_eq!(
        output.atoms.len(),
        input.atoms.len(),
        "{context}: atom count changed"
    );
    for (index, (expected, actual)) in input.atoms.iter().zip(output.atoms.iter()).enumerate() {
        assert_eq!(
            actual.position(),
            expected.position(),
            "{context}: atom {index} position changed"
        );
        assert_eq!(
            actual.atom_type(),
            expected.atom_type(),
            "{context}: atom {index} species changed"
        );
        assert_eq!(
            actual.orbitals(),
            expected.orbitals(),
            "{context}: atom {index} orbital ownership changed"
        );
        assert_eq!(
            actual.magnetic_moment(),
            expected.magnetic_moment(),
            "{context}: atom {index} magnetic moment changed"
        );
    }
}

#[track_caller]
fn assert_support<const DIM: usize>(
    context: &str,
    model: &Model<false, DIM>,
    expected: &[[isize; DIM]],
) {
    assert_eq!(
        model.hamR.nrows(),
        expected.len(),
        "{context}: support size differs"
    );
    for (row, expected_row) in expected.iter().enumerate() {
        for axis in 0..DIM {
            assert_eq!(
                model.hamR[[row, axis]],
                expected_row[axis],
                "{context}: support row {row} axis {axis} differs"
            );
        }
    }
}

/// Every block must be the exact adjoint of its `-R` partner, and the origin
/// block must be Hermitian: the engine writes the conjugate half from the
/// accumulated representative, so these equalities are exact value comparisons,
/// not a tolerance.
#[track_caller]
fn assert_exactly_hermitian<const DIM: usize>(context: &str, model: &Model<false, DIM>) {
    let nsta = model.nsta();
    for index in 0..model.hamR.nrows() {
        let mut negative = Array1::zeros(DIM);
        for axis in 0..DIM {
            negative[axis] = -model.hamR[[index, axis]];
        }
        let partner = super::find_R(&model.hamR, &negative)
            .unwrap_or_else(|| panic!("{context}: support is not closed under negation"));
        let block = model.ham.index_axis(Axis(0), index);
        let partner_block = model.ham.index_axis(Axis(0), partner);
        for i in 0..nsta {
            for j in 0..nsta {
                assert_eq!(
                    block[[i, j]],
                    partner_block[[j, i]].conj(),
                    "{context}: block {index} entry ({i},{j}) is not the exact adjoint"
                );
            }
        }
    }
}

fn effective<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    drive: &FloquetDrive,
    order: usize,
    harmonic_max: isize,
) -> Model<SPIN, DIM, R> {
    model
        .floquet_effective_model(
            drive,
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(order)
                    .with_harmonic_max(harmonic_max),
            ),
        )
        .unwrap()
}

/// Literal signed sums evaluated directly at fractional k, independently of
/// the FFT grid and all adjoint and swapped-inner reuse in the slab engine.
pub(super) fn assert_matches_oracle<const SPIN: bool, const DIM: usize>(
    model: &Model<SPIN, DIM>,
    evaluated: &Model<SPIN, DIM>,
    drive: &FloquetDrive,
    order: usize,
    harmonic_max: isize,
) {
    let cache_max = effective_cache_max(order, harmonic_max).unwrap();
    let cache = model.floquet_harmonic_cache(drive, -cache_max, cache_max, &BESSEL);
    for seed in [0.0, 0.17, 0.29, 0.41, 0.73, 0.91] {
        let k = Array1::from_shape_fn(DIM, |axis| (seed * (axis + 1) as f64) % 1.0);
        let expected = model
            .floquet_effective_ham_onek_lattice(
                &k,
                drive,
                order,
                harmonic_max,
                drive_has_ac_components(drive),
                &cache,
            )
            .unwrap();
        let actual = evaluated.gen_ham(&k, Gauge::Lattice);
        let scale = expected
            .iter()
            .map(|z| z.norm())
            .fold(f64::MIN_POSITIVE, f64::max);
        for (index, &entry) in actual.indexed_iter() {
            let reference = expected[index];
            assert!(
                (entry - reference).norm() <= 1e-10 * scale,
                "order {order}, cutoff {harmonic_max}, k={k:?}, at {index:?}: {entry} vs {reference}"
            );
        }
    }
}

type C = Complex<f64>;

fn pauli([x, y, z]: [C; 3]) -> Array2<C> {
    array![[z, x - C::I * y], [x + C::I * y, -z]]
}

fn matrix_error(a: &Array2<C>, b: &Array2<C>) -> f64 {
    (a - b).iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

#[test]
fn cubic_circular_dipoles_have_signed_mass_and_nonzero_fourth_neighbor() {
    use ndarray::s;
    use std::f64::consts::TAU;

    let (delta, g, w, scalar): (f64, f64, f64, f64) = (0.39, 0.56, 6.0, 0.1);
    let mut model =
        Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), Array2::zeros((2, 3)), None)
            .unwrap();
    model.set_onsite(&array![delta, -delta], None);
    for orbital in 0..2 {
        model.set_hop(scalar, orbital, orbital, &array![0, 0, 1], None);
    }
    for rz in [-1, 1] {
        let ir = row_of(&model.hamR, &[0, 0, rz]).unwrap();
        model
            .rmatrix
            .slice_mut(s![ir, 0, .., ..])
            .assign(&pauli([C::ONE, C::ZERO, C::ZERO]));
        model
            .rmatrix
            .slice_mut(s![ir, 1, .., ..])
            .assign(&pauli([C::ZERO, C::ONE, C::ZERO]));
    }

    for helicity in [1.0, -1.0] {
        // The transverse drive has zero Peierls phase along z. It gives only
        // H_{±1}, with rotating coupling of magnitude |G(k)| = |2g cos(2πk_z)|.
        let drive = FloquetDrive::with_modes(
            w,
            vec![LightMode::new(
                1,
                array![C::from(g / w), C::I * (helicity * g / w), C::ZERO],
            )],
        );
        for order in 0..=3 {
            let out = effective(&model, &drive, order, 1);
            for kz in [0.0, 0.19] {
                let cos = (TAU * kz).cos();
                let g2 = (2.0 * g * cos).powi(2);
                // Expand the signed central eigenvalue of the exact rotating
                // two-state block: s W/2 - s sqrt((W/2-s δ)^2+G^2), s=helicity.
                let terms = [
                    delta,
                    -helicity * g2 / w,
                    -2.0 * delta * g2 / w.powi(2),
                    helicity * (g2 * g2 - 4.0 * delta * delta * g2) / w.powi(3),
                ];
                let mass = terms[..=order].iter().sum::<f64>();
                let shift = 2.0 * scalar * cos;
                let want = array![
                    [C::from(shift + mass), C::ZERO],
                    [C::ZERO, C::from(shift - mass)]
                ];
                let actual = out.gen_ham(&array![0.0, 0.0, kz], Gauge::Lattice);
                let error = matrix_error(&actual, &want);
                assert!(
                    error < 2e-13,
                    "order={order}, helicity={helicity}, k_z={kz}: {error:e}"
                );
            }
            if order == 3 {
                assert_eq!(out.hamR.nrows(), 9);
                for rz in [-4, 4] {
                    let ir = row_of(&out.hamR, &[0, 0, rz]).unwrap();
                    // Only G^4 can reach ±4. The coefficient of exp(±4ik_z) is g^4.
                    let want = pauli([C::ZERO, C::ZERO, C::from(helicity * g.powi(4) / w.powi(3))]);
                    assert!(
                        matrix_error(&out.ham.index_axis(Axis(0), ir).to_owned(), &want) < 2e-13
                    );
                    assert!(
                        out.rmatrix
                            .index_axis(Axis(0), ir)
                            .iter()
                            .all(|&z| z == C::ZERO)
                    );
                }
            }
        }
    }
}

#[test]
fn cubic_multiharmonic_matches_sambe_and_retains_three_cutoff_edge() {
    use crate::ndarray_lapack::eigvalsh_v;
    use ndarray::s;
    use ndarray_linalg::UPLO;

    let harmonic = [
        [C::from(0.32), C::from(-0.21), C::from(0.68)],
        [C::from(0.41), C::new(0.0, 0.29), C::new(0.12, 0.08)],
        [C::new(0.17, -0.13), C::from(-0.27), C::new(0.0, 0.19)],
        [C::new(0.0, 0.12), C::from(0.15), C::from(-0.07)],
    ];
    let hm = harmonic.map(pauli);
    let mut model =
        Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), Array2::zeros((2, 3)), None)
            .unwrap();
    model.set_onsite(&array![0.68, -0.68], None);
    model.set_hop(hm[0][[0, 1]], 0, 1, &array![0, 0, 0], None);
    for axis in 0..3 {
        let mut coefficient = [C::ZERO; 3];
        coefficient[axis] = C::ONE;
        model
            .rmatrix
            .slice_mut(s![0, axis, .., ..])
            .assign(&pauli(coefficient));
    }
    let drive = |w, carriers: usize| {
        let modes = (1..=carriers).map(|m| {
            // With d=0 and X-X_d=σ, H_m=(-imW/2) a_m·σ. Scale a_m
            // with 1/W to hold actual H_m, not the field amplitude, fixed.
            let a = harmonic[m].map(|z| z * C::I * (2.0 / (m as f64 * w)));
            LightMode::new(m as isize, arr1(&a))
        });
        FloquetDrive::with_modes(w, modes.collect())
    };
    let h_at = |w, carriers, order, cutoff| {
        effective(&model, &drive(w, carriers), order, cutoff)
            .gen_ham(&array![0.0, 0.0, 0.0], Gauge::Lattice)
    };

    // Matrix oracle from directly diagonalizing the literal N=16 Sambe
    // operator, reconstructing P(t)=Σ_n C_n exp(+inθ), and choosing constant
    // R by iterating R <- R exp(-mean(log(P R))) on 64 time samples. Listed
    // below are the Pauli coordinates of R† diag(ε) R, not the FM log at t=0
    // or the polar gauge of C_0 (the latter differs already at cubic order).
    // N=12 changed the oracle by less than 5e-13. No vV coefficient is used.
    let exact = [
        [0.342344843008385, -0.188186476773819, 0.633025636374451],
        [0.331680545971553, -0.199409503913026, 0.658834464708965],
        [0.325948108496347, -0.204769447106267, 0.669953997949794],
        [0.322998924938020, -0.207399173543170, 0.675106280292674],
    ];
    let mut errors = [[0.0; 2]; 4];
    for (iw, (w, xyz)) in [10.0, 20.0, 40.0, 80.0].into_iter().zip(exact).enumerate() {
        let want = pauli(xyz.map(C::from));
        // Verify the static matrix oracle's eigenvalues against a live,
        // independent Sambe assembly using the literal harmonics above.
        let ncut = 12;
        let dim = 2 * (2 * ncut + 1);
        let sambe = Array2::from_shape_fn((dim, dim), |(i, j)| {
            let m = (j / 2) as isize - (i / 2) as isize;
            let (a, b) = (i % 2, j % 2);
            match m {
                0 => {
                    hm[0][[a, b]]
                        + if i == j {
                            C::from(((i / 2) as f64 - ncut as f64) * w)
                        } else {
                            C::ZERO
                        }
                }
                1..=3 => hm[m as usize][[a, b]],
                -3..=-1 => hm[(-m) as usize][[b, a]].conj(),
                _ => C::ZERO,
            }
        });
        let levels = eigvalsh_v(&sambe, UPLO::Lower).unwrap();
        let mass = xyz.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((levels[2 * ncut] + mass).abs() < 2e-12);
        assert!((levels[2 * ncut + 1] - mass).abs() < 2e-12);
        for (index, order) in [2, 3].into_iter().enumerate() {
            errors[iw][index] = matrix_error(&h_at(w, 3, order, 3), &want);
        }
        assert!(
            errors[iw][1] < 0.14 * errors[iw][0],
            "W={w}: errors={:?}",
            errors[iw]
        );
    }
    for pair in errors.windows(2) {
        let second = pair[0][0] / pair[1][0];
        let cubic = pair[0][1] / pair[1][1];
        assert!((7.0..10.0).contains(&second), "second order ratio={second}");
        assert!((15.5..16.5).contains(&cubic), "cubic ratio={cubic}");
    }

    // At h=1 the ±3 bare harmonics cannot enter orders 0..2, but they
    // contribute to K3. At h=0 the public cubic request is just H0.
    let w = 10.0_f64;
    for order in 0..=2 {
        assert!(matrix_error(&h_at(w, 3, order, 1), &h_at(w, 2, order, 1)) < 2e-14);
    }
    let edge = matrix_error(&h_at(w, 3, 3, 1), &h_at(w, 2, 3, 1)) * w.powi(3);
    assert!(
        (edge - 0.016633725904877).abs() < 2e-10,
        "3h edge coefficient norm={edge:e}"
    );
    assert_eq!(h_at(w, 3, 3, 0), h_at(w, 3, 0, 0));
}

#[test]
fn cubic_peierls_only_support_and_time_origin() {
    let model = one_dimensional_model();
    let drive = two_mode_drive::<1>();
    let second = effective(&model, &drive, 2, 2);
    let third = effective(&model, &drive, 3, 2);
    assert_support(
        "cubic Peierls",
        &third,
        &[[0], [-4], [-3], [-2], [-1], [1], [2], [3], [4]],
    );
    assert_exactly_hermitian("cubic Peierls", &third);
    let shifted = FloquetDrive::with_modes(
        OMEGA,
        drive
            .modes
            .into_iter()
            .map(|mode| {
                LightMode::new(
                    mode.harmonic,
                    mode.a_complex * C::from_polar(1.0, 0.73 * mode.harmonic as f64),
                )
            })
            .collect(),
    );
    let shifted = effective(&model, &shifted, 3, 2);
    let k = array![0.217];
    let h = third.gen_ham(&k, Gauge::Lattice);
    assert!(matrix_error(&h, &second.gen_ham(&k, Gauge::Lattice)) > 1e-9);
    assert!(matrix_error(&h, &shifted.gen_ham(&k, Gauge::Lattice)) < 2e-13);
}

#[test]
fn streaming_fft_preserves_support_metadata_and_matches_literal_sum() {
    fn check<const DIM: usize>(model: &Model<false, DIM>) {
        let drive = two_mode_drive::<DIM>();
        for order in 0..=2 {
            for cutoff in 0..=2 {
                let actual = effective(model, &drive, order, cutoff);
                assert_metadata_preserved("FFT", model, &actual);
                assert_support("FFT", &actual, &documented_support(model, cutoff, order));
                assert_exactly_hermitian("FFT", &actual);
                assert_matches_oracle(model, &actual, &drive, order, cutoff);
            }
        }
    }
    check(&one_dimensional_model());
    check(&two_dimensional_model());
    check(&three_dimensional_model(1));
    check(&long_range_chain());
}

#[test]
fn static_or_cancelling_drives_keep_exact_h0_and_zero_minkowski_layers() {
    let model = two_dimensional_model();
    let dc = LightMode::new(0, array![Complex::new(0.23, 0.0), Complex::new(-0.11, 0.0)]);
    let ac = array![Complex::new(0.09, 0.04), Complex::new(0.11, -0.03)];
    for drive in [
        FloquetDrive::new(OMEGA),
        FloquetDrive::with_modes(OMEGA, vec![dc.clone()]),
        FloquetDrive::with_modes(
            1e-200,
            vec![dc, LightMode::new(1, ac.clone()), LightMode::new(1, -ac)],
        ),
    ] {
        let static_model = effective(&model, &drive, 0, 2);
        let order_two = effective(&model, &drive, 2, 2);
        assert_support("static FFT", &order_two, &documented_support(&model, 2, 2));
        assert!(order_two.hamR.nrows() > static_model.hamR.nrows());
        for (ir, row) in support_vectors::<2>(&order_two.hamR).iter().enumerate() {
            let block = order_two.ham.index_axis(Axis(0), ir);
            if let Some(input) = row_of(&static_model.hamR, row) {
                assert_eq!(block, static_model.ham.index_axis(Axis(0), input));
            } else {
                assert!(block.iter().all(|z| *z == Complex::new(0.0, 0.0)));
            }
        }
    }
}

#[test]
fn sparse_static_delta_has_no_fft_radius_limit() {
    let mut model =
        Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((1, 2)), None).unwrap();
    model.set_hop(0.7, 0, 0, &array![1_000_000, 0], None);
    model.set_hop(-0.4, 0, 0, &array![0, 1_000_000], None);
    let drive = FloquetDrive::with_modes(
        1e-200,
        vec![LightMode::new(
            0,
            array![Complex::new(0.4e-6, 0.0), Complex::new(0.3e-6, 0.0)],
        )],
    );
    let reference = effective(&model, &drive, 0, 2);
    let cache = model.floquet_harmonic_cache(&drive, 0, 0, &BESSEL);
    let (delta, rows) = model
        .floquet_effective_delta_kspace(&drive, &cache, 2, 2, false)
        .unwrap();
    assert_eq!(rows.nrows(), 25);
    assert_eq!(rows.row(0), array![0, 0]);
    for (ir, row) in support_vectors::<2>(&rows).iter().enumerate() {
        let expected = match (row_of(&reference.hamR, row), row_of(&model.hamR, row)) {
            (Some(new), Some(old)) => reference.ham[[new, 0, 0]] - model.ham[[old, 0, 0]],
            (None, None) => Complex::new(0.0, 0.0),
            _ => panic!("reference lost or gained a primitive row"),
        };
        assert_eq!(delta[[ir, 0, 0]], expected);
    }
    assert!(delta.iter().any(|z| z.norm() > 0.1));
}

#[test]
fn driven_scalar_commutes_even_at_unrepresentable_inverse_square() {
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
    model.set_hop(0.7, 0, 0, &array![1], None);
    let drive = FloquetDrive::with_modes(
        1e-200,
        vec![LightMode::new(1, array![Complex::new(0.8, 0.0)])],
    );
    let zero = effective(&model, &drive, 0, 2);
    let second = effective(&model, &drive, 2, 2);
    assert_eq!(second.hamR.nrows(), 7);
    for k in [array![0.0], array![0.217]] {
        assert_eq!(
            zero.gen_ham(&k, Gauge::Lattice),
            second.gen_ham(&k, Gauge::Lattice)
        );
    }
}

#[test]
fn slab_fft_matches_with_one_and_four_rayon_threads() {
    #[track_caller]
    fn assert_same<const DIM: usize>(serial: &Model<false, DIM>, parallel: &Model<false, DIM>) {
        assert_eq!(serial.hamR, parallel.hamR);
        assert_eq!(serial.ham.dim(), parallel.ham.dim());
        for (index, &a) in serial.ham.indexed_iter() {
            let b = parallel.ham[index];
            assert!((a - b).norm() < 2e-12, "DIM={DIM}, {index:?}: {a} vs {b}");
        }
    }

    let pools = [1, 4].map(|threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    });
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(1);
    let mut model = two_dimensional_model();
    let hopping = 0.26_f64;
    model.set_hop(hopping, 0, 0, &array![1, -1], None);
    let drive = two_mode_drive::<2>();
    let q = array![0.013, -0.017];
    let [serial, parallel] = pools.each_ref().map(|pool| {
        pool.install(|| {
            let uniform = model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap();
            let finite_q = model
                .floquet_effective_q_model(&drive, Some(&options), &q)
                .unwrap();
            (uniform, finite_q)
        })
    });
    assert_same(&serial.0, &parallel.0);
    assert_same(&serial.1, &parallel.1);
    assert_matches_oracle(&model, &parallel.0, &drive, 2, 1);

    // Only the diagonal hop at R=(1,-1) can pair with itself to reach 2R.
    // At that row the two gradient products give -2 |a·d|² t², then the
    // final derivative contributes i q·(2d) and the denominator is 8W.
    assert_eq!(parallel.0.hamR, parallel.1.hamR);
    let row = row_of(&parallel.1.hamR, &[2, -2]).unwrap();
    let d = array![1.0, -1.0].dot(&model.lat);
    let a_dot_d: C = drive.modes[0]
        .a_complex
        .iter()
        .zip(&d)
        .map(|(&a, &d)| a * d)
        .sum();
    let expected = -C::I * (q.dot(&d) * a_dot_d.norm_sqr() * hopping.powi(2) / (2.0 * OMEGA));
    let actual = parallel.1.ham[[row, 0, 0]] - parallel.0.ham[[row, 0, 0]];
    assert!(
        (actual - expected).norm() < 2e-12,
        "finite q: {actual} vs {expected}"
    );

    let model = three_dimensional_model(-1);
    let drive = two_mode_drive::<3>();
    let [serial, parallel] = pools
        .each_ref()
        .map(|pool| pool.install(|| effective(&model, &drive, 3, 1)));
    assert_same(&serial, &parallel);
    // Every primitive row has x+y-z <= 1. This mixed-sign row has
    // x+y-z = 4, so only the quartic spatial convolution reaches it.
    let row = row_of(&parallel.hamR, &[2, 1, -1]).unwrap();
    // Literal cubic sums on only the x+y-z=1 links, with the Peierls
    // harmonics evaluated by direct Fourier sums on 128 time samples.
    let expected = C::new(8.511_063_955_274e-6, 5.263_975_578_966e-6);
    let actual = parallel.ham[[row, 0, 1]];
    assert!(
        (actual - expected).norm() < 2e-12,
        "quartic edge: {actual} vs {expected}"
    );
}

#[test]
fn streaming_output_is_bitwise_repeatable() {
    let model = three_dimensional_model(1);
    let drive = two_mode_drive::<3>();
    let first = effective(&model, &drive, 2, 2);
    let second = effective(&model, &drive, 2, 2);
    assert_eq!(first.hamR, second.hamR);
    for (a, b) in first.ham.iter().zip(&second.ham) {
        assert_eq!(a.re.to_bits(), b.re.to_bits());
        assert_eq!(a.im.to_bits(), b.im.to_bits());
    }
}

#[test]
fn streaming_spinful_model_matches_literal_sum() {
    use crate::SpinDirection;
    let mut model = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.3]], None).unwrap();
    model.set_hop(0.6, 0, 1, &arr1(&[0]), None);
    model.set_hop(-0.4, 0, 0, &arr1(&[1]), None);
    model.set_hop(0.15, 0, 1, &arr1(&[1]), Some(SpinDirection::Z));
    model.set_onsite(&arr1(&[0.2, -0.2]), None);
    let drive = two_mode_drive::<1>();
    let actual = effective(&model, &drive, 2, 2);
    assert_matches_oracle(&model, &actual, &drive, 2, 2);
}

#[test]
fn quadratic_fft_matches_literal_anticommutator_coefficients() {
    // No origin input row, so neither primitive row belongs to the doubled
    // support. The two spatial axes check both in-slab and streaming phases.
    let translations = array![[1, -2], [-1, 2]];
    let a = Array3::from_shape_fn((2, 2, 2), |(r, i, j)| {
        Complex::new((r + 2 * i + 1) as f64 * 0.3, (j + r + 1) as f64 * 0.2)
    });
    let b = Array3::from_shape_fn(a.raw_dim(), |(r, i, j)| a[[1 - r, j, i]].conj());
    let (got, rows, restore) =
        super::kspace::kspace_anticommutator(a.view(), b.view(), &translations).unwrap();
    assert_eq!(restore, 1.0);
    assert_eq!(rows, array![[0, 0], [-2, 4], [2, -4]]);
    let mut expected = Array3::<Complex<f64>>::zeros(got.raw_dim());
    for left in 0..2 {
        for right in 0..2 {
            let row = match (left, right) {
                (0, 0) => 2,
                (1, 1) => 1,
                _ => 0,
            };
            let term = a
                .index_axis(Axis(0), left)
                .dot(&b.index_axis(Axis(0), right))
                + b.index_axis(Axis(0), left)
                    .dot(&a.index_axis(Axis(0), right));
            let mut block = expected.index_axis_mut(Axis(0), row);
            block += &term;
        }
    }
    for (actual, expected) in got.iter().zip(expected.iter()) {
        assert!((actual - expected).norm() < 1e-13);
    }
    let scale = 2.0_f64.powi(600);
    let (large, large_rows, restore) = super::kspace::kspace_anticommutator(
        (&a * scale).view(),
        (&b * scale).view(),
        &translations,
    )
    .unwrap();
    assert_eq!(large_rows, rows);
    assert!(restore > 1.0);
    for (actual, expected) in large.iter().zip(expected.iter()) {
        assert!(((actual * (restore / scale)) * (restore / scale) - expected).norm() < 1e-13);
    }
}

#[test]
fn quadratic_fft_rejects_lossy_operand_rescaling() {
    let rows = array![[0, 0]];
    let large = 2.0_f64.powi(489);
    // The first small component vanishes under the common shift. The second
    // rounds to a nonzero subnormal, but is still not preserved exactly.
    for small in [2.0_f64.powi(-1006), 1.5 * 2.0_f64.powi(-985)] {
        let mut a = Array3::<Complex<f64>>::zeros((1, 3, 3));
        a[[0, 0, 1]] = Complex::new(large, 0.0);
        a[[0, 1, 0]] = a[[0, 0, 1]];
        a[[0, 1, 2]] = Complex::new(0.0, small);
        a[[0, 2, 1]] = a[[0, 1, 2]].conj();
        // A=A†, and (AB+BA)_{02}=2i*large*small is normal in original units.
        assert!((2.0 * large * small).is_normal());
        let error = super::kspace::kspace_anticommutator(a.view(), a.view(), &rows).unwrap_err();
        assert!(
            error.to_string().contains("representable product range"),
            "{error}"
        );
    }
}

#[test]
fn quadratic_fft_rejects_product_underflow_after_lossless_shift() {
    let rows = array![[0, 0], [1, 0], [-1, 0]];
    let large = 2.0_f64.powi(900);
    let small = 2.0_f64.powi(-45);
    let mut a = Array3::<Complex<f64>>::zeros((3, 2, 2));
    for (ir, sign) in [(1, 1.0), (2, -1.0)] {
        a[[ir, 0, 0]] = Complex::new(0.0, sign * large);
        a[[ir, 1, 1]] = Complex::new(0.0, sign * small);
    }
    // Both gradients survive the shift, but the weak (R=2) coefficient
    // -2*small^2 would vanish in scaled units before restore.
    let scaled_small = small * 2.0_f64.powi(-500);
    assert_eq!(scaled_small * 2.0_f64.powi(500), small);
    assert_eq!(scaled_small * scaled_small, 0.0);
    assert!((small * small).is_normal());
    let error = super::kspace::kspace_anticommutator(a.view(), a.view(), &rows).unwrap_err();
    assert!(
        error.to_string().contains("representable product range"),
        "{error}"
    );
}
