//! Per-slice transforms between real space and k space for the streaming
//! effective-model engine.

use super::super::FloquetHarmonicCache;
use super::C;
use super::grid::KspaceGrid;
use crate::error::{Result, TbError};
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3, ArrayViewMut4, Axis, s};
use ndrustfft::{FftHandler, ndfft_inplace, ndfft_inplace_par};
use std::f64::consts::TAU;

/// Harmonic blocks on one slice of the streaming axis.
///
/// Layout (n_harm, points_per_slice, nsta, nsta), with the harmonic slot equal
/// to n + harmonic_max and the point slot ordered as i_0 * (N_1) + i_1.
/// Values are the Bloch sums
///
///   H_n(k) = sum_R T_n(R) exp(+2*pi*i* k . R)
///
/// evaluated at k = (i_0 / N_0, i_1 / N_1, kz / N_last). The forward transform
/// is unnormalised; use the conjugation identity
/// H(k) = conj(fft(conj(T)))(k) so only forward transforms are needed.
///
/// Missing axes (dim < 3) keep shape 1 and are not transformed.
///
/// This allocates the output; the sibling slice_harmonics_into is the
/// reusable-buffer form used by the streaming driver.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn slice_harmonics(
    cache: &FloquetHarmonicCache,
    harmonic_max: isize,
    nsta: usize,
    grid: &KspaceGrid,
    kz: usize,
) -> Result<Array4<C>> {
    let mut out = Array4::<C>::zeros((
        harmonic_count(harmonic_max)?,
        grid.points_per_slice(),
        nsta,
        nsta,
    ));
    slice_harmonics_into(cache, harmonic_max, nsta, grid, kz, out.view_mut())?;
    Ok(out)
}

/// Write one harmonic slice into a caller-provided buffer.
///
/// The output has the same shape and content as slice_harmonics. The buffer
/// must be a standard-layout (n_harm, points_per_slice, nsta, nsta) array;
/// every element of it is overwritten, so stale data from an earlier slice
/// never leaks into the result.
///
/// Harmonics outside the cache's stored range are written as exact zeros: the
/// streaming driver always requests the full van Vleck window, while the cache
/// only stores the range that window can read (an order-0 run caches H_0
/// alone).
pub(super) fn slice_harmonics_into(
    cache: &FloquetHarmonicCache,
    harmonic_max: isize,
    nsta: usize,
    grid: &KspaceGrid,
    kz: usize,
    mut out: ArrayViewMut4<'_, C>,
) -> Result<()> {
    let slots = harmonic_count(harmonic_max)?;
    let lanes = grid.points_per_slice();
    if nsta == 0 {
        return Err(TbError::Other(
            "the k-space slice transform needs at least one state".into(),
        ));
    }
    if kz >= grid.slices() {
        return Err(TbError::Other(format!(
            "the k-space slice index {kz} is outside 0..{}",
            grid.slices()
        )));
    }
    if out.shape() != [slots, lanes, nsta, nsta] {
        return Err(TbError::Other(format!(
            "the k-space harmonic slice needs shape ({slots}, {lanes}, {nsta}, {nsta}), got {:?}",
            out.shape()
        )));
    }
    let n_r = grid.support().nrows();
    let cache_shape = cache.blocks.shape();
    if cache_shape[1] != n_r || cache_shape[2] != nsta || cache_shape[3] != nsta {
        return Err(TbError::Other(format!(
            "the Floquet harmonic cache stores (_, {}, {}, {}), expected (_, {n_r}, {nsta}, {nsta})",
            cache_shape[1], cache_shape[2], cache_shape[3]
        )));
    }

    let dim = grid.dim();
    let last_axis = dim - 1;
    let last_len = grid.axis_len(last_axis);
    let transform_axes = grid.slice_axes();
    let plane = lanes * nsta * nsta;
    let buffer = out.as_slice_mut().ok_or_else(|| {
        TbError::Other("the k-space harmonic slice needs a standard-layout buffer".into())
    })?;
    let (first, second) = padded_transform_axes(grid);
    let first_handler = FftHandler::<f64>::new(first);
    let second_handler = FftHandler::<f64>::new(second);
    let support = grid.support();
    for (slot, plane_data) in buffer.chunks_exact_mut(plane).enumerate() {
        let harmonic = slot as isize - harmonic_max;
        let mut workspace = padded_workspace(plane_data, nsta, grid)?;
        workspace.fill(C::new(0.0, 0.0));
        if harmonic < cache.harmonic_min || harmonic > cache.harmonic_max {
            continue;
        }
        let blocks = cache.harmonic_blocks(harmonic);
        for (i_r, row) in support.outer_iter().enumerate() {
            let first = if transform_axes >= 1 {
                grid.folded(row[0], 0)
            } else {
                0
            };
            let second = if transform_axes >= 2 {
                grid.folded(row[1], 1)
            } else {
                0
            };
            let phase = conjugate_phase(kz, grid.folded(row[last_axis], last_axis), last_len);
            let block = blocks.index_axis(Axis(0), i_r);
            workspace
                .slice_mut(s![first, second, .., ..])
                .zip_mut_with(&block, |target, value| *target += value.conj() * phase);
        }
        forward_transform(&mut workspace, grid, &first_handler, &second_handler);
        workspace.mapv_inplace(|value| value.conj());
    }
    Ok(())
}

/// Inverse transform of one slice of assembled effective blocks.
///
/// Input and output both have shape (points_per_slice, nsta, nsta). The result
/// holds
///
///   X(R_0, R_1) = (1 / N_0 N_1) sum_{k_0, k_1}
///                  X(k_0, k_1) exp(-2*pi*i* k . R)
///
/// with R folded onto the grid. The normalisation is applied here.
///
/// This allocates the output; the sibling inverse_slice_into is the
/// reusable-buffer form used by the streaming driver.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn inverse_slice(blocks: ArrayView3<'_, C>, grid: &KspaceGrid) -> Result<Array3<C>> {
    let mut out = Array3::<C>::zeros(blocks.raw_dim());
    inverse_slice_into(blocks, grid, out.view_mut())?;
    Ok(out)
}

/// Write the inverse slice transform into a caller-provided buffer.
///
/// The buffer must be a standard-layout (points_per_slice, nsta, nsta) array;
/// every element is overwritten. The transform is a plain unnormalised forward
/// FFT over the slice axes followed by the explicit normalisation factor, so a
/// forward slice followed by this inverse recovers the real-space blocks.
pub(super) fn inverse_slice_into(
    blocks: ArrayView3<'_, C>,
    grid: &KspaceGrid,
    mut out: ArrayViewMut3<'_, C>,
) -> Result<()> {
    let lanes = grid.points_per_slice();
    let shape = blocks.shape();
    if shape[0] != lanes || shape[1] != shape[2] {
        return Err(TbError::Other(format!(
            "the k-space inverse slice needs shape ({lanes}, nsta, nsta), got {shape:?}"
        )));
    }
    if out.shape() != shape {
        return Err(TbError::Other(format!(
            "the k-space inverse slice output has shape {:?}, expected {shape:?}",
            out.shape()
        )));
    }
    let nsta = shape[1];
    if nsta == 0 {
        return Err(TbError::Other(
            "the k-space slice transform needs at least one state".into(),
        ));
    }
    out.assign(&blocks);
    let buffer = out.as_slice_mut().ok_or_else(|| {
        TbError::Other("the k-space inverse slice needs a standard-layout buffer".into())
    })?;
    let (first, second) = padded_transform_axes(grid);
    let first_handler = FftHandler::<f64>::new(first);
    let second_handler = FftHandler::<f64>::new(second);
    let mut workspace = padded_workspace(buffer, nsta, grid)?;
    forward_transform(&mut workspace, grid, &first_handler, &second_handler);
    // The padded transform covers exactly points_per_slice points because the
    // missing axes contribute one; the forward FFT is unnormalised.
    let norm = 1.0 / lanes as f64;
    workspace.mapv_inplace(|value| value * norm);
    Ok(())
}

/// Number of slots in the harmonic window [-harmonic_max, harmonic_max].
fn harmonic_count(harmonic_max: isize) -> Result<usize> {
    if harmonic_max < 0 {
        return Err(TbError::Other(format!(
            "the k-space harmonic window must be non-negative, got {harmonic_max}"
        )));
    }
    let slots = harmonic_max
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| TbError::Other("the k-space harmonic window overflows isize".into()))?;
    usize::try_from(slots)
        .map_err(|_| TbError::Other("the k-space harmonic window overflows usize".into()))
}

/// Padded lengths of the two transform axes of one slice workspace.
///
/// Axis 0 and axis 1 hold the grid's first dim - 1 axes; a missing axis keeps
/// length one, so the workspace shape does not depend on the model dimension.
fn padded_transform_axes(grid: &KspaceGrid) -> (usize, usize) {
    let axes = grid.slice_axes();
    let first = if axes >= 1 { grid.axis_len(0) } else { 1 };
    let second = if axes >= 2 { grid.axis_len(1) } else { 1 };
    (first, second)
}

/// exp(-2*pi*i* k*r / n) with the integer product reduced modulo n first.
///
/// This is the conjugate of one Bloch phase, accumulated before a forward FFT
/// to obtain H through the conjugation identity.
fn conjugate_phase(k: usize, folded_r: usize, n: usize) -> C {
    let turn = (k as i128 * folded_r as i128) % n as i128;
    C::from_polar(1.0, -TAU * turn as f64 / n as f64)
}

/// View one padded slice workspace over a contiguous slice of the output.
fn padded_workspace<'a>(
    buffer: &'a mut [C],
    nsta: usize,
    grid: &KspaceGrid,
) -> Result<ArrayViewMut4<'a, C>> {
    let (first, second) = padded_transform_axes(grid);
    let len = buffer.len();
    ArrayViewMut4::<C>::from_shape((first, second, nsta, nsta), buffer).map_err(|error| {
        TbError::Other(format!(
            "a k-space slice workspace of shape ({first}, {second}, {nsta}, {nsta}) does not fit {len} elements: {error}"
        ))
    })
}

/// Apply the unnormalised forward FFT over the slice's transform axes.
///
/// Only axes that exist in the model are transformed; length-one axes are
/// skipped because their transform is the identity. The handlers are built
/// once by the caller so the harmonics of one slice share their FFT plans.
///
/// Above one thread the parallel entry point fans out over the lanes of the
/// untransformed axes on the pool of the calling thread (the global pool in
/// production, because rustfft's Fft is Send + Sync and one plan feeds every
/// lane); a single-thread pool keeps the serial entry point, which reuses one
/// temporary instead of allocating per lane. Both entry points run the same
/// plan, so their results are bitwise identical. FFT never touches BLAS, so
/// the fan-out is safe for every backend feature, including the serial Netlib
/// policy used by the matrix products.
///
/// ndrustfft's parallel in-place branch for NON-standard layouts never copies
/// the lane into its temporary (upstream bug), so the workspace must stay
/// standard layout; every caller builds it that way.
fn forward_transform(
    workspace: &mut ArrayViewMut4<'_, C>,
    grid: &KspaceGrid,
    first_handler: &FftHandler<f64>,
    second_handler: &FftHandler<f64>,
) {
    debug_assert!(workspace.is_standard_layout());
    let axes = grid.slice_axes();
    let (first, second) = padded_transform_axes(grid);
    let parallel = rayon::current_num_threads() > 1;
    if axes >= 1 && first > 1 {
        if parallel {
            ndfft_inplace_par(workspace, first_handler, 0);
        } else {
            ndfft_inplace(workspace, first_handler, 0);
        }
    }
    if axes >= 2 && second > 1 {
        if parallel {
            ndfft_inplace_par(workspace, second_handler, 1);
        } else {
            ndfft_inplace(workspace, second_handler, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model;
    use crate::floquet::coefficients::{BESSEL_CUTOFF_MARGIN, PeierlsFourierMethod};
    use crate::floquet::{FloquetDrive, LightMode};
    use ndarray::{Array2, Array3, ArrayView2, arr1, array};

    fn bessel() -> PeierlsFourierMethod {
        PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        }
    }

    fn one_dimensional_model() -> Model<false, 1> {
        let mut model =
            Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.35]], None).unwrap();
        model.set_onsite(&array![0.2, -0.1], None);
        model.set_hop(C::new(0.5, 0.2), 0, 1, &array![0_isize], None);
        model.set_hop(C::new(-0.3, 0.4), 1, 0, &array![1_isize], None);
        model
    }

    fn two_dimensional_model() -> Model<false, 2> {
        let mut model = Model::<false, 2>::tb_model(
            array![[1.0, 0.18], [-0.11, 1.2]],
            array![[0.02, 0.07], [0.31, 0.14], [0.67, 0.43]],
            None,
        )
        .unwrap();
        model.set_onsite(&array![0.21, -0.33, 0.17], None);
        for (i, j, r, re, im) in [
            (0_usize, 1_usize, [0_isize, 0_isize], 0.4, 0.19),
            (1, 2, [0, 0], -0.23, 0.31),
            (0, 1, [1, 0], -0.8, 0.27),
            (1, 2, [0, 1], 0.47, -0.36),
            (2, 0, [1, 1], 0.29, 0.14),
        ] {
            model.set_hop(C::new(re, im), i, j, &arr1(&r), None);
        }
        model
    }

    fn three_dimensional_model() -> Model<false, 3> {
        let mut model = Model::<false, 3>::tb_model(
            array![[1.0, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.8]],
            array![[0.0, 0.0, 0.0], [0.41, 0.13, 0.07]],
            None,
        )
        .unwrap();
        model.set_onsite(&array![0.24, -0.31], None);
        model.set_hop(C::new(0.42, 0.11), 0, 1, &array![0_isize, 0, 0], None);
        model.set_hop(C::new(-0.27, 0.05), 0, 1, &array![1_isize, 0, 0], None);
        model.set_hop(C::new(0.19, -0.21), 1, 0, &array![0_isize, 1, 0], None);
        model.set_hop(C::new(0.13, 0.07), 0, 1, &array![0_isize, 0, 1], None);
        model
    }

    fn one_dimensional_drive() -> FloquetDrive {
        FloquetDrive::with_modes(4.0, vec![LightMode::new(1, array![C::new(0.31, 0.08)])])
    }

    fn two_dimensional_drive() -> FloquetDrive {
        FloquetDrive::with_modes(
            4.0,
            vec![
                LightMode::new(1, array![C::new(0.21, 0.05), C::new(-0.12, 0.17)]),
                LightMode::new(2, array![C::new(0.03, -0.02), C::new(0.04, 0.01)]),
            ],
        )
    }

    fn three_dimensional_drive() -> FloquetDrive {
        FloquetDrive::with_modes(
            4.0,
            vec![LightMode::new(
                1,
                array![C::new(0.22, 0.05), C::new(-0.13, 0.17), C::new(0.09, -0.06)],
            )],
        )
    }

    /// Independent oracle: the defining Bloch sum over the support rows.
    fn direct_bloch_sum(
        cache: &FloquetHarmonicCache,
        harmonic: isize,
        grid: &KspaceGrid,
        k: &[usize],
    ) -> Array2<C> {
        let dim = grid.dim();
        let nsta = cache.blocks.shape()[2];
        let harmonics = cache.harmonic_blocks(harmonic);
        let mut sum = Array2::<C>::zeros((nsta, nsta));
        for (i_r, row) in grid.support().outer_iter().enumerate() {
            let mut turns = 0.0_f64;
            for axis in 0..dim {
                turns += row[axis] as f64 * k[axis] as f64 / grid.axis_len(axis) as f64;
            }
            let phase = C::from_polar(1.0, TAU * turns);
            let block = harmonics.index_axis(Axis(0), i_r);
            sum.scaled_add(phase, &block);
        }
        sum
    }

    fn assert_blocks_close(actual: ArrayView2<'_, C>, expected: &Array2<C>) {
        assert_eq!(actual.dim(), expected.dim());
        let scale = actual
            .iter()
            .chain(expected.iter())
            .map(|value| value.norm())
            .fold(f64::MIN_POSITIVE, f64::max);
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!(
                (a - b).norm() <= 1e-12 * scale,
                "{a} vs {b} (scale {scale:e})"
            );
        }
    }

    #[test]
    fn one_dimensional_forward_matches_the_direct_bloch_sum() {
        let model = one_dimensional_model();
        let drive = one_dimensional_drive();
        let nsta = model.nsta();
        let harmonic_max = 1_isize;
        let cache = model.floquet_harmonic_cache(&drive, -harmonic_max, harmonic_max, &bessel());
        let grid = KspaceGrid::for_polynomial(&model.hamR, 1, 2).unwrap();
        assert_eq!(grid.slice_axes(), 0);
        assert_eq!(grid.points_per_slice(), 1);
        assert_eq!(grid.slices(), grid.axis_len(0));
        let slots = (2 * harmonic_max + 1) as usize;
        for kz in 0..grid.slices() {
            let slice = slice_harmonics(&cache, harmonic_max, nsta, &grid, kz).unwrap();
            assert_eq!(slice.shape(), &[slots, 1, nsta, nsta]);
            for (slot, harmonic) in (-harmonic_max..=harmonic_max).enumerate() {
                let expected = direct_bloch_sum(&cache, harmonic, &grid, &[kz]);
                assert_blocks_close(
                    slice.index_axis(Axis(0), slot).index_axis(Axis(0), 0),
                    &expected,
                );
            }
        }
    }

    #[test]
    fn two_dimensional_forward_matches_the_direct_bloch_sum() {
        let model = two_dimensional_model();
        let drive = two_dimensional_drive();
        let nsta = model.nsta();
        let harmonic_max = 2_isize;
        let cache = model.floquet_harmonic_cache(&drive, -harmonic_max, harmonic_max, &bessel());
        let grid = KspaceGrid::for_polynomial(&model.hamR, 2, 3).unwrap();
        assert_eq!(grid.slice_axes(), 1);
        assert_eq!(grid.points_per_slice(), grid.axis_len(0));
        assert_eq!(grid.slices(), grid.axis_len(1));
        let slots = (2 * harmonic_max + 1) as usize;
        for kz in 0..grid.slices() {
            let slice = slice_harmonics(&cache, harmonic_max, nsta, &grid, kz).unwrap();
            assert_eq!(slice.shape(), &[slots, grid.points_per_slice(), nsta, nsta]);
            for (slot, harmonic) in (-harmonic_max..=harmonic_max).enumerate() {
                for lane in 0..grid.points_per_slice() {
                    let expected = direct_bloch_sum(&cache, harmonic, &grid, &[lane, kz]);
                    assert_blocks_close(
                        slice.index_axis(Axis(0), slot).index_axis(Axis(0), lane),
                        &expected,
                    );
                }
            }
        }
    }

    #[test]
    fn three_dimensional_forward_matches_the_direct_bloch_sum() {
        let model = three_dimensional_model();
        let drive = three_dimensional_drive();
        let nsta = model.nsta();
        let harmonic_max = 1_isize;
        let cache = model.floquet_harmonic_cache(&drive, -harmonic_max, harmonic_max, &bessel());
        let grid = KspaceGrid::for_polynomial(&model.hamR, 3, 2).unwrap();
        assert_eq!(grid.slice_axes(), 2);
        assert_eq!(grid.points_per_slice(), grid.axis_len(0) * grid.axis_len(1));
        assert_eq!(grid.slices(), grid.axis_len(2));
        let width = grid.axis_len(1);
        let slots = (2 * harmonic_max + 1) as usize;
        for kz in 0..grid.slices() {
            let slice = slice_harmonics(&cache, harmonic_max, nsta, &grid, kz).unwrap();
            assert_eq!(slice.shape(), &[slots, grid.points_per_slice(), nsta, nsta]);
            for (slot, harmonic) in (-harmonic_max..=harmonic_max).enumerate() {
                for lane in 0..grid.points_per_slice() {
                    let k = [lane / width, lane % width, kz];
                    let expected = direct_bloch_sum(&cache, harmonic, &grid, &k);
                    assert_blocks_close(
                        slice.index_axis(Axis(0), slot).index_axis(Axis(0), lane),
                        &expected,
                    );
                }
            }
        }
    }

    fn arbitrary_blocks(support: &Array2<isize>, nsta: usize) -> Array4<C> {
        let n_r = support.nrows();
        Array4::from_shape_fn((1, n_r, nsta, nsta), |(_, i_r, i, j)| {
            let re = ((7 * i_r + 3 * i + 5 * j) % 11) as f64 * 0.19 - 0.9;
            let im = ((2 * i_r + 9 * i + 4 * j) % 13) as f64 * 0.13 - 0.7;
            C::new(re, im)
        })
    }

    /// Arbitrary blocks must survive one slice forward + inverse exactly.
    ///
    /// The support is chosen with one row per slice position, so a single kz
    /// slice carries every block and the round trip is invertible.
    #[test]
    fn round_trip_recovers_arbitrary_blocks() {
        let nsta = 2_usize;
        let cases = [
            (1_usize, array![[0_isize]]),
            (2, array![[0_isize, 0], [1, -1], [-1, 2]]),
            (
                3,
                array![[0_isize, 0, 0], [1, 0, -1], [-1, 2, 0], [1, -2, 1]],
            ),
        ];
        for (dim, support) in cases {
            let cache = FloquetHarmonicCache {
                harmonic_min: 0,
                harmonic_max: 0,
                blocks: arbitrary_blocks(&support, nsta),
            };
            let grid = KspaceGrid::for_polynomial(&support, dim, 1).unwrap();
            let lanes = grid.points_per_slice();
            let forward = slice_harmonics(&cache, 0, nsta, &grid, 0).unwrap();
            assert_eq!(forward.shape(), &[1, lanes, nsta, nsta]);
            let recovered = inverse_slice(forward.index_axis(Axis(0), 0), &grid).unwrap();
            assert_eq!(recovered.shape(), &[lanes, nsta, nsta]);
            // The in-place form must discard stale data from an earlier slice.
            let mut reused = Array3::from_shape_fn((lanes, nsta, nsta), |(lane, i, j)| {
                C::new(7.0 + lane as f64, -3.0 + (i + j) as f64)
            });
            inverse_slice_into(forward.index_axis(Axis(0), 0), &grid, reused.view_mut()).unwrap();
            let source = cache.harmonic_blocks(0);
            let scale = source
                .iter()
                .map(|value| value.norm())
                .fold(f64::MIN_POSITIVE, f64::max);
            let mut seen = vec![false; lanes];
            for (i_r, row) in support.outer_iter().enumerate() {
                let mut lane = 0;
                for axis in 0..dim - 1 {
                    lane = lane * grid.axis_len(axis) + grid.folded(row[axis], axis);
                }
                assert!(
                    !seen[lane],
                    "dim {dim}: support rows share a slice position"
                );
                seen[lane] = true;
                for i in 0..nsta {
                    for j in 0..nsta {
                        let expected = source[[i_r, i, j]];
                        let error = (recovered[[lane, i, j]] - expected).norm();
                        assert!(
                            error <= 1e-12 * scale,
                            "dim {dim}: block {i_r} entry ({i}, {j}) differs by {error:e}"
                        );
                        let error = (reused[[lane, i, j]] - expected).norm();
                        assert!(
                            error <= 1e-12 * scale,
                            "dim {dim}: reused block {i_r} entry ({i}, {j}) differs by {error:e}"
                        );
                    }
                }
            }
            for (lane, covered) in seen.iter().enumerate() {
                if *covered {
                    continue;
                }
                for (wrapped, in_place) in recovered
                    .index_axis(Axis(0), lane)
                    .iter()
                    .zip(reused.index_axis(Axis(0), lane).iter())
                {
                    assert!(
                        wrapped.norm() <= 1e-12 * scale,
                        "dim {dim}: lane {lane} is not empty"
                    );
                    assert!(
                        in_place.norm() <= 1e-12 * scale,
                        "dim {dim}: reused lane {lane} is not empty"
                    );
                }
            }
        }
    }

    #[test]
    fn harmonics_outside_the_cache_window_are_exact_zeros() {
        let model = two_dimensional_model();
        let drive = two_dimensional_drive();
        let nsta = model.nsta();
        let harmonic_max = 2_isize;
        // An order-0 run caches H_0 alone, while the driver still requests the
        // full van Vleck window.
        let cache = model.floquet_harmonic_cache(&drive, 0, 0, &bessel());
        let grid = KspaceGrid::for_polynomial(&model.hamR, 2, 3).unwrap();
        let lanes = grid.points_per_slice();
        let slots = (2 * harmonic_max + 1) as usize;
        let mut out = Array4::from_shape_fn((slots, lanes, nsta, nsta), |(slot, lane, i, j)| {
            C::new(3.5 + slot as f64, -2.0 + (lane + i + j) as f64)
        });
        slice_harmonics_into(&cache, harmonic_max, nsta, &grid, 1, out.view_mut()).unwrap();
        for (slot, harmonic) in (-harmonic_max..=harmonic_max).enumerate() {
            if harmonic != 0 {
                assert!(
                    out.index_axis(Axis(0), slot)
                        .iter()
                        .all(|value| value.re == 0.0 && value.im == 0.0),
                    "harmonic {harmonic} outside the cache window must be exactly zero"
                );
                continue;
            }
            for lane in 0..lanes {
                let expected = direct_bloch_sum(&cache, 0, &grid, &[lane, 1]);
                assert_blocks_close(
                    out.index_axis(Axis(0), slot).index_axis(Axis(0), lane),
                    &expected,
                );
            }
        }
    }

    #[test]
    fn parallel_and_serial_transforms_are_bitwise_equal() {
        // A three-dimensional slice exercises both transform axes (axis 0 and
        // axis 1 are each non-outermost), and a one-thread pool takes the
        // serial entry point while four threads take the parallel one. Each
        // lane runs the same plan, so the results must agree bit for bit. The
        // inverse input varies by lane and by matrix element, so a permuted or
        // dropped lane cannot hide behind identical data.
        let model = three_dimensional_model();
        let drive = three_dimensional_drive();
        let harmonic_max = 2_isize;
        let cache = model.floquet_harmonic_cache(&drive, -harmonic_max, harmonic_max, &bessel());
        let grid = KspaceGrid::for_polynomial(&model.hamR, 3, 3).unwrap();
        let nsta = model.nsta();
        let slots = (2 * harmonic_max + 1) as usize;
        let lanes = grid.points_per_slice();
        let kz = if grid.slices() > 1 { 1 } else { 0 };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a rayon pool");
            pool.install(|| {
                let mut harmonics = Array4::<C>::zeros((slots, lanes, nsta, nsta));
                slice_harmonics_into(&cache, harmonic_max, nsta, &grid, kz, harmonics.view_mut())
                    .unwrap();
                let k_blocks = Array3::from_shape_fn((lanes, nsta, nsta), |(lane, i, j)| {
                    C::new(
                        0.25 + (lane % 5) as f64 + i as f64,
                        -0.5 + (lane % 3) as f64 - j as f64,
                    )
                });
                let mut real = Array3::<C>::zeros((lanes, nsta, nsta));
                inverse_slice_into(k_blocks.view(), &grid, real.view_mut()).unwrap();
                (harmonics, real)
            })
        };
        let (serial, serial_inverse) = run(1);
        let (parallel, parallel_inverse) = run(4);
        assert_eq!(serial.shape(), parallel.shape());
        for (left, right) in serial.iter().zip(parallel.iter()) {
            assert_eq!(
                left.re.to_bits(),
                right.re.to_bits(),
                "parallel forward transform changed a real part"
            );
            assert_eq!(
                left.im.to_bits(),
                right.im.to_bits(),
                "parallel forward transform changed an imaginary part"
            );
        }
        for (left, right) in serial_inverse.iter().zip(parallel_inverse.iter()) {
            assert_eq!(
                left.re.to_bits(),
                right.re.to_bits(),
                "parallel inverse transform changed a real part"
            );
            assert_eq!(
                left.im.to_bits(),
                right.im.to_bits(),
                "parallel inverse transform changed an imaginary part"
            );
        }
    }
}
