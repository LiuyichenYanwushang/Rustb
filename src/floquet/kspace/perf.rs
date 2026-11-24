//! Stage timing probe for the streaming engine (55-orbital slab).

use super::grid::KspaceGrid;
use super::{C, transform, vanvleck};
use crate::Model;
use crate::floquet::coefficients::BESSEL_CUTOFF_MARGIN;
use crate::floquet::{FloquetDrive, LightMode, PeierlsFourierMethod, effective_cache_max};
use ndarray::{Array2, Array3, Array4};
use std::time::Instant;

type Kc = C;

/// Deterministic Hermitian model on a symmetric translation box.
fn synthetic_model(nsta: usize, radii: [isize; 3]) -> Model<false, 3> {
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
    let mut blocks: Vec<Array2<Kc>> = Vec::new();
    let mut assigned = std::collections::BTreeSet::new();
    for x in -radii[0]..=radii[0] {
        for y in -radii[1]..=radii[1] {
            for z in -radii[2]..=radii[2] {
                if !assigned.insert([x, y, z]) {
                    continue;
                }
                let mut block = Array2::from_shape_fn((nsta, nsta), |_| Kc::new(next(), next()));
                if [x, y, z] == [0, 0, 0] {
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
    let mut ham = Array3::<Kc>::zeros((vectors.len(), nsta, nsta));
    for (index, (vector, block)) in vectors.iter().zip(&blocks).enumerate() {
        for axis in 0..3 {
            ham_r[[index, axis]] = vector[axis];
        }
        ham.index_axis_mut(ndarray::Axis(0), index).assign(block);
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
                ndarray::array![
                    Kc::new(0.12, 0.02),
                    Kc::new(-0.05, 0.09),
                    Kc::new(0.03, -0.02)
                ],
            ),
            LightMode::new(
                2,
                ndarray::array![
                    Kc::new(0.04, -0.01),
                    Kc::new(0.02, 0.03),
                    Kc::new(-0.01, 0.02)
                ],
            ),
        ],
    )
}

#[test]
#[ignore = "times the streaming stages on a 55-orbital slab"]
fn kspace_stage_timing_probe() {
    let model = synthetic_model(55, [9, 9, 4]);
    let drive = drive();
    let harmonic_max = 2_isize;
    let order = 2_usize;
    let used_max = effective_cache_max(order, harmonic_max).unwrap();
    let prepared = Instant::now();
    let cache = model.floquet_harmonic_cache(
        &drive,
        -used_max,
        used_max,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    let grid = KspaceGrid::for_polynomial(&model.hamR, 3, order + 1).unwrap();
    let prepare = prepared.elapsed().as_secs_f64();
    let lanes = grid.points_per_slice();
    let slices = grid.slices();
    let nsta = model.nsta();
    println!(
        "grid={:?} lanes={lanes} slices={slices} total={} nsta={nsta}",
        grid.shape(),
        grid.total_points()
    );

    // Support enumeration plus the full output allocation.
    let started = Instant::now();
    let (support_box, support) =
        super::blocks::effective_support(&model.hamR, 3, harmonic_max, order).unwrap();
    let mut blocks = super::blocks::DenseBlocks::new(support_box, support, nsta, 3).unwrap();
    let support_time = started.elapsed().as_secs_f64();
    assert!(
        blocks.stored_blocks() > 0,
        "the probe model must store blocks"
    );

    let mut harmonics = Array4::<Kc>::zeros(((2 * used_max + 1) as usize, lanes, nsta, nsta));
    let mut assembled = Array3::<Kc>::zeros((lanes, nsta, nsta));
    let mut real = Array3::<Kc>::zeros((lanes, nsta, nsta));

    // Only a few slices are needed to extrapolate: the per-slice work is
    // identical, and the full loop would dominate the probe itself.
    let sampled = 3_usize.min(slices);
    let started = Instant::now();
    for kz in 0..sampled {
        transform::slice_harmonics_into(&cache, used_max, nsta, &grid, kz, harmonics.view_mut())
            .unwrap();
    }
    let forward = started.elapsed().as_secs_f64() / sampled as f64 * slices as f64;

    let started = Instant::now();
    for _ in 0..sampled {
        vanvleck::assemble_pointwise(
            harmonics.view(),
            harmonic_max,
            order,
            true,
            true,
            drive.omega0_ev,
            assembled.view_mut(),
        )
        .unwrap();
    }
    let assembly = started.elapsed().as_secs_f64() / sampled as f64 * slices as f64;

    let started = Instant::now();
    for _ in 0..sampled {
        transform::inverse_slice_into(assembled.view(), &grid, real.view_mut()).unwrap();
    }
    let inverse = started.elapsed().as_secs_f64() / sampled as f64 * slices as f64;

    let slots = super::phase_slots(&grid, &blocks);
    assert_eq!(slots.len(), blocks.stored_blocks());
    let started = Instant::now();
    for kz in 0..sampled {
        super::accumulate_slice(&grid, kz, &slots, real.view(), &mut blocks);
    }
    let accumulate = started.elapsed().as_secs_f64() / sampled as f64 * slices as f64;

    // One-time output assembly: the accumulated rows are already in place, so
    // finish only writes the conjugate half and symmetrises the origin.
    let started = Instant::now();
    let (mut ham, mut ham_r) = blocks.finish().unwrap();
    let finish = started.elapsed().as_secs_f64();
    let started = Instant::now();
    super::place_origin_first(&mut ham, &mut ham_r).unwrap();
    let origin = started.elapsed().as_secs_f64();
    println!(
        "estimated totals: prepare={prepare:.1}s support={support_time:.1}s forward={forward:.1}s assembly={assembly:.1}s inverse={inverse:.1}s accumulate={accumulate:.1}s finish={finish:.1}s origin={origin:.1}s blocks={}",
        ham_r.nrows()
    );
}
