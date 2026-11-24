//! Ordered, memory-bounded batches of effective-model corrections.

use super::*;
use ndarray::Array5;

pub(super) fn visit_effective<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    drive: &FloquetDrive,
    cache: &FloquetHarmonicCache,
    has_ac: bool,
    mut requests: &[(usize, isize)],
    mut visit: impl FnMut((usize, isize), &Model<SPIN, DIM, R>) -> Result<()>,
) -> Result<()> {
    if requests.is_empty() {
        return Ok(());
    }
    let nsta = model.nsta();
    let single = SweepWorkspace {
        requests: 1,
        cache_bytes: cache.blocks.len() as u128 * size_of::<C>() as u128,
        has_position: R::HAS_RMATRIX,
    };
    let first_active = if has_ac && nsta > 1 {
        let mut used_max = 0;
        let mut outer_max = 0;
        for &(order, h) in requests {
            used_max = used_max.max(effective_cache_max(order, h)?);
            if order > 0 {
                outer_max = outer_max.max(h);
            }
        }
        model.validate_kspace_cache(cache, used_max)?;
        first_active_harmonic(cache, outer_max)
    } else {
        None
    };
    let active = |(order, h)| order > 0 && first_active.is_some_and(|first| first <= h);
    let mut rows = None;
    while let Some(&first) = requests.first() {
        let mut batch_len = 1;
        if active(first) && requests.get(1).is_some_and(|&key| active(key)) {
            let (order, h) = first;
            let grid = KspaceGrid::for_polynomial(&model.hamR, DIM, order + 1)?;
            let mut shapes = [None; 4];
            shapes[order] = Some(grid.shape());
            let mut used_max = effective_cache_max(order, h)?;
            let mut cubic_max = if order == 3 { h } else { 0 };
            let mut concurrent = 0;
            // Preserve input order. Different degrees can only share the transform
            // if the same smooth grid already resolves both without aliasing.
            for &(next_order, next_h) in &requests[1..] {
                if !active((next_order, next_h)) {
                    break;
                }
                if shapes[next_order].is_none() {
                    // Defer any grid failure until this request reaches the front.
                    let Ok(next_grid) =
                        KspaceGrid::for_polynomial(&model.hamR, DIM, next_order + 1)
                    else {
                        break;
                    };
                    shapes[next_order] = Some(next_grid.shape());
                }
                if shapes[next_order] != shapes[order] {
                    break;
                }
                let next_used = used_max.max(effective_cache_max(next_order, next_h)?);
                let next_cubic = cubic_max.max(if next_order == 3 { next_h } else { 0 });
                let slabs = sweep_slab_limit(
                    &grid,
                    nsta,
                    next_used,
                    next_cubic,
                    SweepWorkspace {
                        requests: batch_len + 1,
                        ..single
                    },
                );
                if slabs == 0 {
                    break;
                }
                used_max = next_used;
                cubic_max = next_cubic;
                concurrent = slabs;
                batch_len += 1;
            }
            if batch_len > 1 {
                let batch = &requests[..batch_len];
                // On extreme arithmetic the pointwise optimization may decline.
                // Use the scalar solver's established retry and error policy, in
                // order, before exposing any results from this batch.
                if let Ok(mut volumes) = inverse_sweep(
                    &grid,
                    cache,
                    nsta,
                    used_max,
                    batch,
                    drive.omega0_ev,
                    concurrent,
                ) {
                    let rows = rows.get_or_insert_with(|| model.sorted_effective_rows());
                    let h0 = cache.harmonic_blocks(0);
                    for (&key, volume) in batch.iter().zip(volumes.outer_iter_mut()) {
                        let (order, h) = key;
                        let mut accumulator = model.effective_accumulator(order, h)?;
                        gather_correction(&grid, volume, &mut accumulator)?;
                        let (ham, ham_r) = model.finish_effective_correction(
                            accumulator,
                            rows,
                            h0.view(),
                            false,
                        )?;
                        let output = model.floquet_effective_output(ham, ham_r)?;
                        visit(key, &output)?;
                    }
                    requests = &requests[batch_len..];
                    continue;
                }
            }
        }
        for &key in &requests[..batch_len] {
            let (ham, ham_r) = model.floquet_effective_blocks_kspace(
                drive,
                cache,
                key,
                has_ac,
                false,
                Some(single),
            )?;
            let output = model.floquet_effective_output(ham, ham_r)?;
            visit(key, &output)?;
        }
        requests = &requests[batch_len..];
    }
    Ok(())
}

fn sweep_slab_limit(
    grid: &KspaceGrid,
    nsta: usize,
    used_max: isize,
    cubic_max: isize,
    batch: SweepWorkspace,
) -> usize {
    // Four running/commutator matrices plus the largest cubic generator
    // workspace (including its binary-normalization buffers), if needed.
    let scratch_blocks = if cubic_max == 0 {
        4
    } else {
        8 * cubic_max as u128 + 10
    };
    workspace_slab_limit(
        grid,
        (2 * used_max + 1) as usize,
        nsta,
        scratch_blocks,
        0,
        Some(batch),
    )
    .0
}

fn inverse_sweep(
    grid: &KspaceGrid,
    cache: &FloquetHarmonicCache,
    nsta: usize,
    used_max: isize,
    requests: &[(usize, isize)],
    omega: f64,
    concurrent: usize,
) -> Result<Array5<C>> {
    let slices = grid.slices();
    let lanes = grid.points_per_slice();
    // Request-major volumes keep the final-axis inverse standard layout.
    // Split the slab axis to give each task disjoint writes into every volume.
    let mut volumes = Array5::zeros((requests.len(), slices, lanes, nsta, nsta));
    let mut rest = volumes.view_mut();
    let mut tasks = Vec::with_capacity(concurrent);
    let mut first = 0;
    for task in 0..concurrent {
        let len = slices / concurrent + usize::from(task < slices % concurrent);
        let (part, remain) = rest.split_at(Axis(1), len);
        tasks.push((first, part));
        rest = remain;
        first += len;
    }
    tasks
        .into_par_iter()
        .try_for_each(|(first, mut part)| -> Result<()> {
            let mut harmonics = Array4::zeros(((2 * used_max + 1) as usize, lanes, nsta, nsta));
            for (i, mut dst) in part.axis_iter_mut(Axis(1)).enumerate() {
                transform::slice_harmonics_into(
                    cache,
                    used_max,
                    nsta,
                    grid,
                    first + i,
                    harmonics.view_mut(),
                )?;
                vanvleck::assemble_sweep_pointwise(
                    harmonics.view(),
                    requests,
                    omega,
                    dst.view_mut(),
                )?;
                // Request planes remain contiguous even though the assembled
                // slab has a stride between requests.
                for slab in dst.outer_iter_mut() {
                    transform::inverse_slice_inplace(slab, grid)?;
                }
            }
            Ok(())
        })?;
    Ok(volumes)
}

#[cfg(test)]
#[test]
fn sweep_workspace_counts_correction_grids_and_cubic_scratch() {
    let input = ndarray::array![[0, 0], [-2, 0], [2, 0]];
    let grid = KspaceGrid::for_polynomial(&input, 2, 2).unwrap();
    let nsta = 1_000;
    let block = (nsta as u128).pow(2) * size_of::<C>() as u128;
    let points = grid.total_points() as u128;
    let batch = SweepWorkspace {
        requests: 2,
        cache_bytes: 0,
        has_position: false,
    };
    let (_, bytes_two) = workspace_slab_limit(&grid, 3, nsta, 4, 0, Some(batch));
    let (_, bytes_three) = workspace_slab_limit(
        &grid,
        3,
        nsta,
        4,
        0,
        Some(SweepWorkspace {
            requests: 3,
            ..batch
        }),
    );
    assert_eq!(
        bytes_three - bytes_two,
        points * block + size_of::<(isize, usize, usize)>() as u128
    );
    assert!(sweep_slab_limit(&grid, nsta, 3, 0, batch) > 0);
    assert_eq!(
        sweep_slab_limit(
            &grid,
            nsta,
            3,
            0,
            SweepWorkspace {
                requests: 1_000,
                ..batch
            }
        ),
        0
    );
    let onsite = KspaceGrid::for_polynomial(&ndarray::array![[0]], 1, 4).unwrap();
    let limit = KSPACE_MEMORY_HARD_LIMIT.min(isize::MAX as u128);
    let h = (limit / (12 * block)) as isize;
    assert!(sweep_slab_limit(&onsite, nsta, 3 * h, 0, batch) > 0);
    assert_eq!(sweep_slab_limit(&onsite, nsta, 3 * h, h, batch), 0);
}

#[cfg(test)]
#[test]
fn sweep_workspace_counts_shared_cache_and_one_emitted_position_array() {
    let input = ndarray::array![[0, 0, 0], [-3, -3, -3], [3, 3, 3]];
    let grid = KspaceGrid::for_polynomial(&input, 3, 2).unwrap();
    let (nsta, requests) = (64, 2);
    let limit = KSPACE_MEMORY_HARD_LIMIT.min(isize::MAX as u128);
    let batch = SweepWorkspace {
        requests,
        cache_bytes: 0,
        has_position: false,
    };
    let (_, without_position) = workspace_slab_limit(&grid, 3, nsta, 4, 0, Some(batch));
    let batch = SweepWorkspace {
        has_position: true,
        ..batch
    };
    let (_, with_position) = workspace_slab_limit(&grid, 3, nsta, 4, 0, Some(batch));
    let position_bytes =
        grid.total_points() as u128 * 3 * (nsta as u128).pow(2) * size_of::<C>() as u128;
    assert!(with_position > without_position);
    // Assembly scratch is not live during position materialization. Adding
    // the two phases would make this difference exactly position_bytes.
    assert!(with_position - without_position < position_bytes);

    // Supplement the live FFT buffers with a synthetic cache at the cap.
    // No actual large allocation is needed to test the admission decision.
    let batch = SweepWorkspace {
        cache_bytes: limit - with_position,
        ..batch
    };
    assert!(sweep_slab_limit(&grid, nsta, 1, 0, batch) > 0);
    let batch = SweepWorkspace {
        cache_bytes: batch.cache_bytes + 1,
        ..batch
    };
    assert_eq!(sweep_slab_limit(&grid, nsta, 1, 0, batch), 0);
    assert!(
        sweep_slab_limit(
            &grid,
            nsta,
            1,
            0,
            SweepWorkspace {
                has_position: false,
                ..batch
            }
        ) > 0
    );
}

#[cfg(test)]
#[test]
fn sweep_workspace_caps_active_and_static_singletons() {
    let limit = KSPACE_MEMORY_HARD_LIMIT.min(isize::MAX as u128);
    let grid = KspaceGrid::for_polynomial(&ndarray::array![[0, 0]], 2, 2).unwrap();
    let single = SweepWorkspace {
        requests: 1,
        cache_bytes: limit,
        has_position: true,
    };
    assert!(workspace_concurrency(&grid, 3, 2, Some(0), None).is_ok());
    assert!(workspace_concurrency(&grid, 3, 2, Some(0), Some(single)).is_err());

    let (rows, nsta) = (8, 5);
    let output_row_bytes =
        (1 + 2) * 5 * 5 * size_of::<C>() + 2 * size_of::<isize>() + size_of::<usize>();
    let single = SweepWorkspace {
        cache_bytes: limit - (rows * output_row_bytes) as u128,
        ..single
    };
    assert!(single.check_static_output(rows, nsta, 2).is_ok());
    assert!(single.check_static_output(rows + 1, nsta, 2).is_err());

    // Input cache, output, and five co-live matrices for an offsite pair.
    let wide_no_position = SweepWorkspace {
        cache_bytes: 3 * 20_000_u128.pow(2) * size_of::<C>() as u128,
        has_position: false,
        ..single
    };
    assert!(wide_no_position.check_static_output(3, 20_000, 1).is_err());
}
