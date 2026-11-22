//! Integration tests for the streaming k-space effective-model engine.
//!
//! Every test pins the streaming k-space path against the pairwise real-space
//! path of the same van Vleck construction through the crate-internal
//! EffectivePath selector. The properties covered here are:
//!
//! - small 1D, 2D and 3D models for orders 0, 1, 2 and harmonic_max 1, 2:
//!   exact hamR equality, Hamiltonian agreement to 1e-10 relative, unchanged
//!   lattice/orbital/atom metadata, and the documented Minkowski support;
//! - static drives: order 2 must reproduce the Peierls-dressed static blocks
//!   while still retaining the documented zero-valued pair and triple
//!   Minkowski layers;
//! - the size-based Auto selector: a 65-block 1D chain must route to k space,
//!   a 3-block model must stay in real space, and both routes must reproduce
//!   the explicitly forced paths;
//! - exactly zero harmonic blocks produced by selection rules and by coherent
//!   mode cancellation;
//! - bitwise determinism of two independent k-space runs;
//! - exact block-level Hermiticity `H(R) = H(-R)^dagger` of the k-space output,
//!   because the conjugate half is written as the stored representative's
//!   adjoint rather than summed independently.
//!
//! The relative tolerance always uses the largest block element as scale, as
//! documented on the real-space constructor.

use super::coefficients::BESSEL_CUTOFF_MARGIN;
use super::kspace::EffectivePath;
use super::{
    FloquetDrive, FloquetEffectiveOptions, LightMode, PeierlsFourierMethod, drive_has_ac_components,
};
use crate::Model;
use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, arr1, array};
use num_complex::Complex;
use std::collections::BTreeSet;

/// Largest accepted relative difference between the two evaluation paths.
const PATH_RELATIVE_TOLERANCE: f64 = 1e-10;

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

fn three_dimensional_model() -> Model<false, 3> {
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
        (2, 0, [0, 0, 1], 0.31, 0.12),
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
fn three_block_model() -> Model<false, 1> {
    let lat = array![[1.0]];
    let orb = array![[0.0]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_onsite(&array![-0.5], None);
    model.set_hop(-0.9, 0, 0, &arr1(&[1_isize]), None);
    model
}

// ---------------------------------------------------------------------------
// Comparison helpers.
// ---------------------------------------------------------------------------

/// Run the effective-model driver with production selector settings: the
/// order-two regroup and the adjoint H_0 family are both enabled, matching
/// Model::floquet_effective_model.
fn effective<const DIM: usize>(
    model: &Model<false, DIM>,
    drive: &FloquetDrive,
    order: usize,
    harmonic_max: isize,
    path: EffectivePath,
) -> Model<false, DIM> {
    effective_with_selector(model, drive, order, harmonic_max, path, true, true)
}

/// Run the effective-model driver with explicit order-two selector flags.
fn effective_with_selector<const DIM: usize>(
    model: &Model<false, DIM>,
    drive: &FloquetDrive,
    order: usize,
    harmonic_max: isize,
    path: EffectivePath,
    regroup_order_two: bool,
    adjoint_h0_family: bool,
) -> Model<false, DIM> {
    let options = FloquetEffectiveOptions::new()
        .with_order(order)
        .with_harmonic_max(harmonic_max);
    model
        .floquet_effective_model_impl(
            drive,
            Some(&options),
            regroup_order_two,
            adjoint_h0_family,
            path,
        )
        .unwrap_or_else(|error| {
            panic!(
                "{path:?} order {order} harmonic_max {harmonic_max} regroup {regroup_order_two} adjoint {adjoint_h0_family} failed: {error}"
            )
        })
}

/// Extract the translation rows of a support as fixed-size vectors.
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

#[test]
fn kspace_output_is_exactly_hermitian() {
    let two = two_dimensional_model();
    let drive = two_mode_drive::<2>();
    let kspace = effective(&two, &drive, 2, 2, EffectivePath::Kspace);
    assert_exactly_hermitian("2D order 2", &kspace);

    let three = three_dimensional_model();
    let drive = two_mode_drive::<3>();
    let kspace = effective(&three, &drive, 2, 2, EffectivePath::Kspace);
    assert_exactly_hermitian("3D order 2", &kspace);
}

#[track_caller]
fn assert_ham_matches<const DIM: usize>(
    context: &str,
    actual: &Model<false, DIM>,
    expected: &Model<false, DIM>,
    tolerance: f64,
) {
    assert_eq!(actual.hamR, expected.hamR, "{context}: hamR differs");
    assert_eq!(
        actual.ham.dim(),
        expected.ham.dim(),
        "{context}: block count differs"
    );
    for (index, value) in actual.ham.iter().enumerate() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "{context}: block entry {index} is not finite ({value})"
        );
    }
    let scale = actual
        .ham
        .iter()
        .chain(expected.ham.iter())
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    assert!(
        scale > 0.0 && scale.is_finite(),
        "{context}: vanishing or non-finite Hamiltonian scale {scale:e}"
    );
    for (index, (actual_value, expected_value)) in
        actual.ham.iter().zip(expected.ham.iter()).enumerate()
    {
        let difference = (actual_value - expected_value).norm();
        assert!(
            difference <= tolerance * scale,
            "{context}: entry {index} differs by {difference:e} with scale {scale:e} \
             (actual {actual_value}, expected {expected_value})"
        );
    }
}

#[track_caller]
fn assert_block_matches(
    context: &str,
    actual: ArrayView2<'_, Complex<f64>>,
    expected: ArrayView2<'_, Complex<f64>>,
    tolerance: f64,
) {
    assert_eq!(
        actual.dim(),
        expected.dim(),
        "{context}: block shape differs"
    );
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let difference = (actual_value - expected_value).norm();
        assert!(
            difference <= tolerance,
            "{context}: entry {index} differs by {difference:e} (tolerance {tolerance:e}): \
             actual {actual_value}, expected {expected_value}"
        );
    }
}

#[track_caller]
fn assert_block_is_bitwise_equal(
    context: &str,
    actual: ArrayView2<'_, Complex<f64>>,
    expected: ArrayView2<'_, Complex<f64>>,
) {
    assert_eq!(
        actual.dim(),
        expected.dim(),
        "{context}: block shape differs"
    );
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_value.re.to_bits(),
            expected_value.re.to_bits(),
            "{context}: entry {index} real part is not bitwise equal: \
             actual {actual_value}, expected {expected_value}"
        );
        assert_eq!(
            actual_value.im.to_bits(),
            expected_value.im.to_bits(),
            "{context}: entry {index} imaginary part is not bitwise equal: \
             actual {actual_value}, expected {expected_value}"
        );
    }
}

#[track_caller]
fn assert_block_is_zero(context: &str, block: ArrayView2<'_, Complex<f64>>) {
    for (index, value) in block.iter().enumerate() {
        assert!(
            value.re == 0.0 && value.im == 0.0,
            "{context}: entry {index} is {value}, not exactly zero"
        );
    }
}

/// Compare both paths for one (order, harmonic_max) pair and check the
/// documented support and the preserved metadata.
fn assert_paths_agree<const DIM: usize>(
    context: &str,
    model: &Model<false, DIM>,
    drive: &FloquetDrive,
    order: usize,
    harmonic_max: isize,
) -> (Model<false, DIM>, Model<false, DIM>) {
    let real_space = effective(model, drive, order, harmonic_max, EffectivePath::RealSpace);
    let kspace = effective(model, drive, order, harmonic_max, EffectivePath::Kspace);
    assert_metadata_preserved(context, model, &real_space);
    assert_metadata_preserved(context, model, &kspace);
    let support = documented_support(model, harmonic_max, order);
    assert_support(&format!("{context}: real-space"), &real_space, &support);
    assert_support(&format!("{context}: k-space"), &kspace, &support);
    assert_ham_matches(
        &format!("{context}: k-space versus real-space"),
        &kspace,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );
    (real_space, kspace)
}

/// Check that an effective model without AC corrections equals the static
/// Peierls reference on the primitive support while the documented zero-valued
/// Minkowski layers are still present.
///
/// The real-space path keeps those layers as exact zeros and, when
/// exact_real_space is true, reproduces the order-0 static blocks bitwise. The
/// k-space path carries the same layers only up to inverse-transform round-off,
/// so its primitive blocks are checked against the same relative tolerance used
/// between the paths.
fn assert_static_limit<const DIM: usize>(
    context: &str,
    model: &Model<false, DIM>,
    drive: &FloquetDrive,
    reference_drive: &FloquetDrive,
    order: usize,
    harmonic_max: isize,
    exact_real_space: bool,
) {
    let (real_space, kspace) = assert_paths_agree(context, model, drive, order, harmonic_max);
    let reference = effective(
        model,
        reference_drive,
        0,
        harmonic_max,
        EffectivePath::RealSpace,
    );
    let primitive: BTreeSet<[isize; DIM]> =
        support_vectors::<DIM>(&model.hamR).into_iter().collect();
    let scale = reference
        .ham
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    assert!(
        scale > 0.0 && scale.is_finite(),
        "{context}: the static reference vanishes"
    );
    assert!(
        real_space.hamR.nrows() > primitive.len(),
        "{context}: the documented pair/triple support layers are missing"
    );
    for (label, evaluated, exact) in [
        ("real-space", &real_space, exact_real_space),
        ("k-space", &kspace, false),
    ] {
        for row in 0..evaluated.hamR.nrows() {
            let mut vector = [0_isize; DIM];
            for axis in 0..DIM {
                vector[axis] = evaluated.hamR[[row, axis]];
            }
            let block = evaluated.ham.index_axis(Axis(0), row);
            let block_context = format!("{context} ({label}, block {vector:?})");
            if primitive.contains(&vector) {
                let reference_row = row_of(&reference.hamR, &vector).unwrap_or_else(|| {
                    panic!("{block_context}: block is missing from the static reference")
                });
                let expected = reference.ham.index_axis(Axis(0), reference_row);
                if exact {
                    assert_block_is_bitwise_equal(&block_context, block, expected);
                } else {
                    assert_block_matches(
                        &block_context,
                        block,
                        expected,
                        PATH_RELATIVE_TOLERANCE * scale,
                    );
                }
            } else if exact || label == "real-space" {
                assert_block_is_zero(&block_context, block);
            } else {
                let largest = block
                    .iter()
                    .map(|value| value.norm())
                    .fold(0.0_f64, f64::max);
                assert!(
                    largest <= PATH_RELATIVE_TOLERANCE * scale,
                    "{block_context}: zero-valued support layer holds {largest:e} \
                     with scale {scale:e}"
                );
            }
        }
    }
}

#[track_caller]
fn assert_bitwise_models<const DIM: usize>(
    context: &str,
    actual: &Model<false, DIM>,
    expected: &Model<false, DIM>,
) {
    assert_eq!(
        actual.hamR, expected.hamR,
        "{context}: hamR differs bitwise"
    );
    assert_eq!(
        actual.ham.dim(),
        expected.ham.dim(),
        "{context}: block count differs"
    );
    for (index, (actual_value, expected_value)) in
        actual.ham.iter().zip(expected.ham.iter()).enumerate()
    {
        assert_eq!(
            actual_value.re.to_bits(),
            expected_value.re.to_bits(),
            "{context}: entry {index} real part is not bitwise deterministic"
        );
        assert_eq!(
            actual_value.im.to_bits(),
            expected_value.im.to_bits(),
            "{context}: entry {index} imaginary part is not bitwise deterministic"
        );
    }
}

// ---------------------------------------------------------------------------
// 1. Small 1D, 2D and 3D models: orders 0, 1, 2 and harmonic_max 1, 2.
// ---------------------------------------------------------------------------

fn check_small_model<const DIM: usize>(context: &str, model: &Model<false, DIM>) {
    assert!(
        (2..=3).contains(&model.norb()),
        "{context}: fixture must have 2 or 3 orbitals"
    );
    assert!(
        !model.atoms.is_empty(),
        "{context}: fixture must carry atom metadata"
    );
    assert_eq!(model.hamR.ncols(), DIM);
    let drive = two_mode_drive::<DIM>();
    for order in 0..=2 {
        for harmonic_max in [1_isize, 2] {
            let label = format!("{context} order {order} harmonic_max {harmonic_max}");
            assert_paths_agree(&label, model, &drive, order, harmonic_max);
        }
    }
}

#[test]
fn kspace_matches_real_space_for_small_one_dimensional_models() {
    check_small_model("1D", &one_dimensional_model());
}

#[test]
fn kspace_matches_real_space_for_small_two_dimensional_models() {
    check_small_model("2D", &two_dimensional_model());
}

#[test]
fn kspace_matches_real_space_for_small_three_dimensional_models() {
    check_small_model("3D", &three_dimensional_model());
}

/// The streaming engine must also reproduce the termwise real-space reference,
/// which keeps both order-two selectors disabled. This pins the regroup and
/// adjoint-H_0-family optimizations against the k-space arithmetic separately
/// from the production real-space path used above.
#[test]
fn kspace_matches_the_termwise_real_space_reference() {
    fn check<const DIM: usize>(context: &str, model: &Model<false, DIM>) {
        let drive = two_mode_drive::<DIM>();
        for order in 1..=2 {
            let termwise = effective_with_selector(
                model,
                &drive,
                order,
                2,
                EffectivePath::RealSpace,
                false,
                false,
            );
            let regrouped = effective(model, &drive, order, 2, EffectivePath::RealSpace);
            let kspace = effective(model, &drive, order, 2, EffectivePath::Kspace);
            assert_metadata_preserved(context, model, &termwise);
            assert_metadata_preserved(context, model, &kspace);
            assert_ham_matches(
                &format!("{context} order {order}: termwise versus regrouped real space"),
                &termwise,
                &regrouped,
                PATH_RELATIVE_TOLERANCE,
            );
            assert_ham_matches(
                &format!("{context} order {order}: k-space versus termwise real space"),
                &kspace,
                &termwise,
                PATH_RELATIVE_TOLERANCE,
            );
        }
    }
    check("1D", &one_dimensional_model());
    check("2D", &two_dimensional_model());
    check("3D", &three_dimensional_model());
}

// ---------------------------------------------------------------------------
// 2. Static drives: order 2 reduces to the Peierls blocks but keeps the
// documented zero-valued Minkowski layers.
// ---------------------------------------------------------------------------

#[test]
fn static_drives_reproduce_the_peierls_blocks_and_keep_zero_layers() {
    let model = two_dimensional_model();
    let empty = FloquetDrive::new(OMEGA);
    let harmonic_zero = FloquetDrive::with_modes(
        OMEGA,
        vec![LightMode::new(
            0,
            arr1(&[Complex::new(0.23, 0.0), Complex::new(-0.11, 0.0)]),
        )],
    );

    // An empty drive leaves the bare hopping blocks; the harmonic-0 mode adds
    // only the static Peierls phase exp(-i Re(a_0) . d). Both order-2 results
    // must equal the corresponding order-0 static reference. The real-space
    // path keeps the layer zeros exact and the primitive blocks bitwise; the
    // k-space path is compared relatively because its inverse transform adds
    // round-off.
    assert_static_limit("empty static drive", &model, &empty, &empty, 2, 2, true);
    assert_static_limit(
        "harmonic-0 static drive",
        &model,
        &harmonic_zero,
        &harmonic_zero,
        2,
        2,
        true,
    );

    // The static Peierls phase must actually change the blocks, so the
    // harmonic-0 comparison above is not vacuous.
    let dressed = effective(&model, &harmonic_zero, 0, 2, EffectivePath::RealSpace);
    let bare = effective(&model, &empty, 0, 2, EffectivePath::RealSpace);
    assert_eq!(dressed.hamR, bare.hamR);
    let largest_change = dressed
        .ham
        .iter()
        .zip(bare.ham.iter())
        .map(|(dressed_value, bare_value)| (dressed_value - bare_value).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        largest_change > 1e-3,
        "the harmonic-0 Peierls phase is not exercised: largest change {largest_change:e}"
    );
}

// ---------------------------------------------------------------------------
// 3. Size-based Auto selection.
// ---------------------------------------------------------------------------

#[test]
fn auto_routes_the_sixty_five_block_chain_to_k_space() {
    let model = long_range_chain();
    assert_eq!(
        model.hamR.nrows(),
        65,
        "chain must have 65 real-space blocks"
    );
    let pairs = (model.hamR.nrows() as u128) * (model.hamR.nrows() as u128);
    assert!(pairs >= 4096, "chain has only {pairs} block pairs");
    assert!(model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 2, true, OMEGA));
    assert!(model.prefer_kspace_effective_model(EffectivePath::Kspace, 2, 2, true, OMEGA));
    assert!(!model.prefer_kspace_effective_model(EffectivePath::RealSpace, 2, 2, true, OMEGA));

    let drive = two_mode_drive::<1>();
    let auto = effective(&model, &drive, 2, 2, EffectivePath::Auto);
    let kspace = effective(&model, &drive, 2, 2, EffectivePath::Kspace);
    let real_space = effective(&model, &drive, 2, 2, EffectivePath::RealSpace);
    assert_metadata_preserved("65-block chain: auto", &model, &auto);
    assert_support(
        "65-block chain: auto",
        &auto,
        &documented_support(&model, 2, 2),
    );
    assert_ham_matches(
        "65-block chain: auto versus k-space",
        &auto,
        &kspace,
        PATH_RELATIVE_TOLERANCE,
    );
    assert_ham_matches(
        "65-block chain: auto versus real-space",
        &auto,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );
    assert_ham_matches(
        "65-block chain: k-space versus real-space",
        &kspace,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );
}

#[test]
fn auto_keeps_the_three_block_model_in_real_space() {
    let model = three_block_model();
    assert_eq!(model.hamR.nrows(), 3, "fixture must have exactly 3 blocks");
    assert!(!model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 2, true, OMEGA));

    let drive = two_mode_drive::<1>();
    let auto = effective(&model, &drive, 2, 2, EffectivePath::Auto);
    let real_space = effective(&model, &drive, 2, 2, EffectivePath::RealSpace);
    assert_metadata_preserved("3-block model: auto", &model, &auto);
    assert_support(
        "3-block model: auto",
        &auto,
        &documented_support(&model, 2, 2),
    );
    assert_ham_matches(
        "3-block model: auto versus real-space",
        &auto,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );

    // Forcing k space must reproduce the same cheap model as well.
    let kspace = effective(&model, &drive, 2, 2, EffectivePath::Kspace);
    assert_ham_matches(
        "3-block model: k-space versus real-space",
        &kspace,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );
}

// ---------------------------------------------------------------------------
// 4. Exactly zero harmonic blocks and exact cancellation.
// ---------------------------------------------------------------------------

#[test]
fn exactly_zero_harmonic_blocks_keep_the_paths_in_agreement() {
    let model = two_dimensional_model();

    // A drive with a single second-harmonic mode has a closed-form coefficient
    // series that vanishes identically for every odd harmonic: H_+-1 is exactly
    // zero while H_+-2 is not.
    let even_only = FloquetDrive::with_modes(
        OMEGA,
        vec![LightMode::new(
            2,
            arr1(&[Complex::new(0.17, 0.09), Complex::new(-0.06, 0.13)]),
        )],
    );
    let cache = model.floquet_harmonic_cache(&even_only, -4, 4, &BESSEL);
    for harmonic in [-1_isize, 1] {
        assert!(
            cache
                .harmonic_blocks(harmonic)
                .iter()
                .all(|value| value.re == 0.0 && value.im == 0.0),
            "harmonic {harmonic} of the single second-harmonic drive is not exactly zero"
        );
    }
    for harmonic in [-2_isize, 2] {
        assert!(
            cache
                .harmonic_blocks(harmonic)
                .iter()
                .any(|value| value.re != 0.0 || value.im != 0.0),
            "harmonic {harmonic} of the single second-harmonic drive must be active"
        );
    }
    for order in 1..=2 {
        assert_paths_agree("even-only drive", &model, &even_only, order, 2);
    }

    // Two equal harmonic-1 modes with opposite amplitudes cancel coherently
    // after link projection, so every AC harmonic block is exactly zero. The
    // drive still counts as time dependent, so the engine keeps the documented
    // pair and triple layers instead of taking the static shortcut.
    let amplitude = arr1(&[Complex::new(0.27, 0.11), Complex::new(-0.14, 0.23)]);
    let cancelled = FloquetDrive::with_modes(
        OMEGA,
        vec![
            LightMode::new(1, amplitude.clone()),
            LightMode::new(1, amplitude.mapv(|value| -value)),
        ],
    );
    assert!(
        drive_has_ac_components(&cancelled),
        "the cancelling drive must still count as time dependent"
    );
    let cache = model.floquet_harmonic_cache(&cancelled, -4, 4, &BESSEL);
    for harmonic in -4_isize..=4 {
        let blocks = cache.harmonic_blocks(harmonic);
        if harmonic == 0 {
            assert!(
                blocks
                    .iter()
                    .any(|value| value.re != 0.0 || value.im != 0.0),
                "the static block of the cancelling drive vanished"
            );
        } else {
            assert!(
                blocks
                    .iter()
                    .all(|value| value.re == 0.0 && value.im == 0.0),
                "harmonic {harmonic} of the cancelling drive is not exactly zero"
            );
        }
    }
    let empty = FloquetDrive::new(OMEGA);
    for order in 1..=2 {
        assert_static_limit(
            &format!("cancelling drive order {order}"),
            &model,
            &cancelled,
            &empty,
            order,
            2,
            false,
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Determinism: two independent k-space runs must agree bitwise.
// ---------------------------------------------------------------------------

#[test]
fn kspace_engine_is_bitwise_deterministic() {
    let model = two_dimensional_model();
    let drive = two_mode_drive::<2>();
    let first = effective(&model, &drive, 2, 2, EffectivePath::Kspace);
    let second = effective(&model, &drive, 2, 2, EffectivePath::Kspace);
    assert_bitwise_models("2D repeated k-space run", &second, &first);

    let chain = long_range_chain();
    let drive = two_mode_drive::<1>();
    let first = effective(&chain, &drive, 2, 2, EffectivePath::Kspace);
    let second = effective(&chain, &drive, 2, 2, EffectivePath::Kspace);
    assert_bitwise_models("65-block chain repeated k-space run", &second, &first);
}

// ---------------------------------------------------------------------------
// 6. Degenerate harmonic windows.
// ---------------------------------------------------------------------------

#[test]
fn zero_harmonic_window_reduces_to_the_static_block_on_every_path() {
    // harmonic_max = 0 keeps no AC harmonic at all: every van Vleck correction
    // sum is empty, so orders 0, 1 and 2 must all return the static Peierls
    // block while both evaluation paths still agree.
    let model = two_dimensional_model();
    let drive = two_mode_drive::<2>();
    let reference = effective(&model, &drive, 0, 0, EffectivePath::RealSpace);
    for order in [0_usize, 1, 2] {
        let (real_space, kspace) = assert_paths_agree("harmonic_max 0", &model, &drive, order, 0);
        assert_ham_matches(
            "harmonic_max 0 versus the static reference: real-space",
            &real_space,
            &reference,
            PATH_RELATIVE_TOLERANCE,
        );
        assert_ham_matches(
            "harmonic_max 0 versus the static reference: k-space",
            &kspace,
            &reference,
            PATH_RELATIVE_TOLERANCE,
        );
    }
}

#[test]
fn auto_selection_keeps_a_zero_harmonic_window_consistent() {
    // The automatic selector must not change the answer for a degenerate
    // window: a large support still routes through k space and must equal the
    // real-space result.
    let chain = long_range_chain();
    let drive = two_mode_drive::<1>();
    for order in [1_usize, 2] {
        let auto = effective(&chain, &drive, order, 0, EffectivePath::Auto);
        let real_space = effective(&chain, &drive, order, 0, EffectivePath::RealSpace);
        assert_ham_matches(
            "harmonic_max 0 auto versus real-space",
            &auto,
            &real_space,
            PATH_RELATIVE_TOLERANCE,
        );
    }
}

// ---------------------------------------------------------------------------
// 7. Selection and validation behaviour added after the adversarial review.
// ---------------------------------------------------------------------------

#[test]
fn auto_stays_in_real_space_when_the_grid_would_dwarf_the_pair_set() {
    // A sparse 3D support with a large radius: seven blocks give 49 pairs, but
    // the zero-aliasing grid needs (2*3*20+1)^3 points, which is orders of
    // magnitude more work than one convolution per pair. The automatic
    // selector must keep this in real space.
    let mut model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        Array2::from_shape_fn((2, 3), |(i, j)| 0.1 * (i + j) as f64),
        None,
    )
    .unwrap();
    let mut vectors = Vec::new();
    let mut blocks = Vec::new();
    for axis in 0..3 {
        for sign in [-1_isize, 1] {
            let mut vector = [0_isize; 3];
            vector[axis] = 20 * sign;
            vectors.push(vector);
            blocks.push(Array2::from_elem((2, 2), Complex::new(0.05, 0.0)));
        }
    }
    // Add a real onsite block so the model is a genuine Hamiltonian.
    let mut ham_r = Array2::<isize>::zeros((vectors.len() + 1, 3));
    let mut ham = Array3::<Complex<f64>>::zeros((vectors.len() + 1, 2, 2));
    for (index, block) in blocks.iter().enumerate() {
        for axis in 0..3 {
            ham_r[[index, axis]] = vectors[index][axis];
        }
        ham.index_axis_mut(Axis(0), index).assign(block);
    }
    ham.index_axis_mut(Axis(0), vectors.len())
        .assign(&Array2::eye(2));
    model.ham = ham;
    model.hamR = ham_r;
    model.validate().unwrap();

    let drive = two_mode_drive::<3>();
    let selector = model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 2, true, OMEGA);
    assert!(
        !selector,
        "a huge grid on a tiny pair set must stay in real space"
    );
    let auto = effective(&model, &drive, 2, 2, EffectivePath::Auto);
    let real_space = effective(&model, &drive, 2, 2, EffectivePath::RealSpace);
    assert_ham_matches(
        "auto versus real-space",
        &auto,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );
}

#[test]
fn static_and_zero_order_calls_stay_in_real_space() {
    let model = long_range_chain();
    let drive = two_mode_drive::<1>();
    assert!(!model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 0, true, OMEGA));
    assert!(!model.prefer_kspace_effective_model(EffectivePath::Auto, 0, 2, true, OMEGA));
    let static_drive = FloquetDrive::new(OMEGA);
    assert!(!model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 2, false, OMEGA));
    let auto = effective(&model, &static_drive, 2, 2, EffectivePath::Auto);
    let real_space = effective(&model, &static_drive, 2, 2, EffectivePath::RealSpace);
    assert_ham_matches("static auto", &auto, &real_space, PATH_RELATIVE_TOLERANCE);
}

#[test]
fn duplicate_and_unpaired_translations_are_rejected_by_both_paths() {
    let drive = two_mode_drive::<2>();
    let options = FloquetEffectiveOptions::new()
        .with_order(1)
        .with_harmonic_max(1);

    // A duplicated translation row: the real-space block map would merge the
    // two rows while the k-space transform would sum them, so both paths
    // must refuse the model instead of disagreeing silently.
    let mut model = two_dimensional_model();
    let row = model.hamR.row(0).to_owned();
    model.hamR.push_row(row.view()).unwrap();
    let block = model.ham.index_axis(Axis(0), 0).to_owned();
    model.ham.push(Axis(0), block.view()).unwrap();
    for path in [EffectivePath::RealSpace, EffectivePath::Kspace] {
        let outcome = model.floquet_effective_model_impl(&drive, Some(&options), true, true, path);
        assert!(
            outcome.is_err(),
            "duplicate translations must be rejected by {path:?}"
        );
    }

    // A support that keeps a hopping vector but drops its -R partner cannot be
    // Hermitised.
    let mut model = two_dimensional_model();
    let mut removed = None;
    for index in 0..model.hamR.nrows() {
        let row = model.hamR.row(index);
        let negated = row.mapv(|value| -value);
        let has_partner = (0..model.hamR.nrows()).any(|other| {
            other != index
                && model
                    .hamR
                    .row(other)
                    .iter()
                    .zip(negated.iter())
                    .all(|(a, b)| a == b)
        });
        if has_partner {
            removed = Some(index);
            break;
        }
    }
    let removed = removed.expect("the fixture must contain a paired hopping vector");
    let kept = (0..model.hamR.nrows())
        .filter(|index| *index != removed)
        .collect::<Vec<_>>();
    let ham_r = Array2::from_shape_fn((kept.len(), 2), |(row, column)| {
        model.hamR[[kept[row], column]]
    });
    let ham = Array3::from_shape_fn((kept.len(), model.nsta(), model.nsta()), |(row, i, j)| {
        model.ham[[kept[row], i, j]]
    });
    model.ham = ham;
    model.hamR = ham_r;
    for path in [EffectivePath::RealSpace, EffectivePath::Kspace] {
        let outcome = model.floquet_effective_model_impl(&drive, Some(&options), true, true, path);
        assert!(
            outcome.is_err(),
            "an unpaired translation must be rejected by {path:?}"
        );
    }
}

#[test]
fn a_window_of_exact_zero_harmonics_never_asks_for_a_frequency_scale() {
    // Two opposite harmonic-1 modes cancel coherently, so every AC harmonic is
    // exactly zero inside the window while the drive still has AC components.
    // A tiny photon energy must not turn the k-space assembly into an error:
    // there is nothing to scale, and the real-space path returns H_0.
    let model = long_range_chain();
    let cancelling = FloquetDrive::with_modes(
        OMEGA,
        vec![
            LightMode::new(1, array![Complex::new(0.09, 0.04)]),
            LightMode::new(1, array![Complex::new(-0.09, -0.04)]),
        ],
    );
    assert!(drive_has_ac_components(&cancelling));
    let tiny = FloquetDrive::with_modes(
        1e-200,
        vec![
            LightMode::new(1, array![Complex::new(0.09, 0.04)]),
            LightMode::new(1, array![Complex::new(-0.09, -0.04)]),
        ],
    );
    assert!(drive_has_ac_components(&tiny));
    let reference = effective(&model, &cancelling, 2, 2, EffectivePath::RealSpace);
    let real_space = effective(&model, &tiny, 2, 2, EffectivePath::RealSpace);
    let kspace = effective(&model, &tiny, 2, 2, EffectivePath::Kspace);
    // A non-representable frequency scale must keep the automatic choice on the
    // exact real-space path, where a zero correction never touches the scale.
    assert!(!model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 2, true, 1e-200));
    // Order one only needs a representable 1/omega.
    assert!(!model.prefer_kspace_effective_model(EffectivePath::Auto, 2, 1, true, 1e-320));
    let auto = effective(&model, &tiny, 2, 2, EffectivePath::Auto);
    assert_ham_matches(
        "tiny omega auto",
        &auto,
        &real_space,
        PATH_RELATIVE_TOLERANCE,
    );
    assert_ham_matches(
        "tiny omega real-space",
        &real_space,
        &reference,
        PATH_RELATIVE_TOLERANCE,
    );
    assert_ham_matches(
        "tiny omega k-space",
        &kspace,
        &reference,
        PATH_RELATIVE_TOLERANCE,
    );
}

#[test]
fn spinful_models_take_the_same_path() {
    use crate::SpinDirection;

    let mut model = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.3]], None).unwrap();
    model.set_hop(0.6, 0, 1, &arr1(&[0]), None);
    model.set_hop(-0.4, 0, 0, &arr1(&[1]), None);
    model.set_hop(0.15, 0, 1, &arr1(&[1]), Some(SpinDirection::Z));
    model.set_onsite(&arr1(&[0.2, -0.2]), None);
    model.validate().unwrap();

    let drive = FloquetDrive::with_modes(
        OMEGA,
        vec![
            LightMode::new(1, array![Complex::new(0.11, 0.03)]),
            LightMode::new(2, array![Complex::new(-0.04, 0.02)]),
        ],
    );
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(2);
    let real_space = model
        .floquet_effective_model_impl(&drive, Some(&options), true, true, EffectivePath::RealSpace)
        .unwrap();
    let kspace = model
        .floquet_effective_model_impl(&drive, Some(&options), true, true, EffectivePath::Kspace)
        .unwrap();
    assert_eq!(real_space.hamR, kspace.hamR, "spinful support differs");
    let scale = real_space
        .ham
        .iter()
        .chain(kspace.ham.iter())
        .map(|value| value.norm())
        .fold(f64::MIN_POSITIVE, f64::max);
    let mut worst = 0.0_f64;
    for (left, right) in real_space.ham.iter().zip(kspace.ham.iter()) {
        worst = worst.max((left - right).norm());
    }
    assert!(
        worst <= PATH_RELATIVE_TOLERANCE * scale,
        "spinful paths differ by {worst:e} (scale {scale:e})"
    );
}
