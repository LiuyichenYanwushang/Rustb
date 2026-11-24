//! Validation and cost probes for the streaming k-space engine.
//!
//! The small differential test runs with the suite. The larger cost probes are
//! ignored by default: the 55-orbital one builds a few-thousand-block model,
//! which takes a minute and about ten gigabytes. Pin OPENBLAS_NUM_THREADS=1
//! when RAYON_NUM_THREADS is above one, or BLAS oversubscription dominates.

use super::*;
use ndarray::{Array2, Array3};
use num_complex::Complex;
use std::time::Instant;

type C = Complex<f64>;

/// Deterministic pseudo-random Hermitian model on a symmetric translation box.
fn synthetic_model(nsta: usize, box_radii: [isize; 3]) -> Model<false, 3> {
    let mut model = Model::<false, 3>::tb_model(
        Array2::eye(3),
        Array2::from_shape_fn((nsta, 3), |(i, j)| 0.01 * ((i * 7 + j) % 13) as f64),
        None,
    )
    .unwrap();
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64) - 0.5
    };
    let mut vectors = Vec::new();
    let mut blocks: Vec<Array2<C>> = Vec::new();
    let mut assigned = std::collections::BTreeSet::new();
    for x in -box_radii[0]..=box_radii[0] {
        for y in -box_radii[1]..=box_radii[1] {
            for z in -box_radii[2]..=box_radii[2] {
                if !assigned.insert([x, y, z]) {
                    continue;
                }
                let mut block = Array2::from_shape_fn((nsta, nsta), |_| C::new(next(), next()));
                if [x, y, z] == [0, 0, 0] {
                    // The onsite block must be Hermitian for the model to be
                    // physically consistent; the adjoint fast path assumes it.
                    block = (&block + &block.t().mapv(|value| value.conj())) * 0.5;
                }
                assigned.insert([-x, -y, -z]);
                vectors.push([x, y, z]);
                blocks.push(block.clone());
                if [x, y, z] != [0, 0, 0] {
                    vectors.push([-x, -y, -z]);
                    blocks.push(block.t().mapv(|value| value.conj()));
                }
            }
        }
    }
    let mut ham_r = Array2::<isize>::zeros((vectors.len(), 3));
    let mut ham = Array3::<C>::zeros((vectors.len(), nsta, nsta));
    for (index, (vector, block)) in vectors.iter().zip(&blocks).enumerate() {
        for axis in 0..3 {
            ham_r[[index, axis]] = vector[axis];
        }
        ham.index_axis_mut(Axis(0), index).assign(block);
    }
    model.ham = ham;
    model.hamR = ham_r;
    model.validate().unwrap();
    model
}

fn drive() -> FloquetDrive {
    FloquetDrive::with_modes(
        4.0,
        vec![
            LightMode::new(
                1,
                array![C::new(0.12, 0.02), C::new(-0.05, 0.09), C::new(0.03, -0.02)],
            ),
            LightMode::new(
                2,
                array![C::new(0.04, -0.01), C::new(0.02, 0.03), C::new(-0.01, 0.02)],
            ),
        ],
    )
}

fn options() -> FloquetEffectiveOptions {
    FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(2)
}

#[test]
fn kspace_engine_matches_literal_sum_on_a_synthetic_model() {
    let model = synthetic_model(12, [2, 2, 1]);
    let drive = drive();
    let started = Instant::now();
    let kspace = model
        .floquet_effective_model(&drive, Some(&options()))
        .unwrap();
    println!(
        "synthetic: blocks={} FFT={:.3} s",
        kspace.hamR.nrows(),
        started.elapsed().as_secs_f64()
    );
    super::kspace_tests::assert_matches_oracle(&model, &kspace, &drive, 2, 2);
}

#[test]
#[ignore = "builds a 55-orbital model with a few thousand translations"]
fn kspace_engine_cost_probe() {
    let model = synthetic_model(55, [9, 9, 4]);
    let drive = drive();
    let options = options();
    let started = Instant::now();
    let kspace = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    let blocks = kspace.hamR.nrows();
    let bytes = blocks * 55 * 55 * std::mem::size_of::<C>();
    println!(
        "55-orbital probe: input_blocks={} output_blocks={blocks} kspace={elapsed:.1} s output={:.2} GiB",
        model.hamR.nrows(),
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    assert!(elapsed > 0.0);
}
