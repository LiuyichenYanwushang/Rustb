//! Streaming k-space evaluation of the Floquet effective Hamiltonian.
//!
//! The real-space path convolves hopping blocks pairwise, which costs
//! O(N_R^2) matrix products. Every order-two van Vleck term is instead a
//! polynomial of bounded degree in the harmonic blocks, so the same operator
//! can be evaluated on a zero-aliasing k-point grid in O(N_k n^3) products and
//! transformed back once. This module implements that engine.
//!
//! The engine assumes a Hermitian input model: H(R) = H(-R)^dagger, so every
//! Peierls harmonic satisfies H_-n = H_n^dagger and H_0 = H_0^dagger (up to
//! coefficient round-off). Both order-two families are evaluated on one
//! representative of each conjugate pair and completed with its adjoint, which
//! halves their matrix products. Hand-mutating the public ham blocks to break
//! that relation is unsupported, exactly as for the real-space path.
//!
//! The last resolved axis is sampled in concurrent slices within the memory
//! budget. Each task owns its harmonic scratch and a disjoint slice of the
//! correction grid. After pointwise assembly and the in-slice inverse, a
//! batched FFT along the last axis completes the inverse before gathering R.

use super::*;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3};
use num_complex::Complex;

mod blocks;
mod grid;
mod sweep;
mod transform;
mod vanvleck;

use grid::KspaceGrid;

/// Complex scalar used throughout the Floquet kernels.
pub(super) type C = Complex<f64>;

// Bound the FFT working set. For sweep batches, also include the shared
// harmonic cache and the position array for the current callback model.
pub(super) const KSPACE_MEMORY_HARD_LIMIT: u128 = 64 * 1024 * 1024 * 1024;

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Assemble on the same translation rows (including r-only rows) and in
    /// the same order used when building the cache. At order two the cache
    /// includes +/-2N, although only m and m' in +/-N are summed.
    pub(super) fn floquet_effective_model_kspace(
        &self,
        drive: &FloquetDrive,
        harmonic_cache: &FloquetHarmonicCache,
        order: usize,
        harmonic_max: isize,
        has_time_dependence: bool,
    ) -> Result<Model<SPIN, DIM, R>> {
        let (ham, ham_r) = self.floquet_effective_blocks_kspace(
            drive,
            harmonic_cache,
            (order, harmonic_max),
            has_time_dependence,
            false,
            None,
        )?;
        self.floquet_effective_output(ham, ham_r)
    }

    /// Mode-resolved correction; subtract bare H from H_0 before adding the
    /// dynamic terms, so they are not rounded away against the bare model.
    pub(super) fn floquet_effective_delta_kspace(
        &self,
        drive: &FloquetDrive,
        harmonic_cache: &FloquetHarmonicCache,
        order: usize,
        harmonic_max: isize,
        has_time_dependence: bool,
    ) -> Result<(Array3<C>, Array2<isize>)> {
        self.floquet_effective_blocks_kspace(
            drive,
            harmonic_cache,
            (order, harmonic_max),
            has_time_dependence,
            true,
            None,
        )
    }

    fn floquet_effective_blocks_kspace(
        &self,
        drive: &FloquetDrive,
        harmonic_cache: &FloquetHarmonicCache,
        (order, harmonic_max): (usize, isize),
        has_time_dependence: bool,
        subtract_bare: bool,
        sweep: Option<SweepWorkspace>,
    ) -> Result<(Array3<C>, Array2<isize>)> {
        let used_max = if has_time_dependence {
            effective_cache_max(order, harmonic_max)?
        } else {
            0
        };
        self.validate_kspace_cache(harmonic_cache, used_max)?;
        let slots = (2 * used_max + 1) as usize;
        let nsta = self.nsta();
        let h0 = harmonic_cache.harmonic_blocks(0);
        let degree = if harmonic_max > 0 { order + 1 } else { 1 };
        let rows = self.sorted_effective_rows();
        // A window whose AC harmonics are zero carries no correction, even
        // when the inverse frequency is not representable. Still keep every
        // documented Minkowski layer, exactly zero outside the input rows.
        // Scalar harmonics also commute identically at every k.
        let has_corrections =
            nsta > 1 && first_active_harmonic(harmonic_cache, harmonic_max.min(used_max)).is_some();
        if !has_corrections {
            let mut ham_r = if degree == 1 {
                self.hamR.select(Axis(0), &rows)
            } else {
                blocks::zero_padded_support::<DIM>(
                    &self.hamR,
                    degree,
                    sweep.map(|sweep| {
                        move |rows| sweep.check_static_support_growth(rows, nsta, DIM)
                    }),
                )?
            };
            if let Some(sweep) = sweep {
                let missing_origin = !ham_r.outer_iter().any(|row| row.iter().all(|&r| r == 0));
                let output_rows = ham_r.nrows() + usize::from(missing_origin);
                sweep.check_static_output(output_rows, nsta, DIM)?;
            }
            let mut ham = Array3::<C>::zeros((ham_r.nrows(), nsta, nsta));
            let mut out = ham_r.outer_iter().zip(ham.outer_iter_mut()).peekable();
            for &ir in &rows {
                let row = self.hamR.row(ir);
                while out
                    .peek()
                    .is_some_and(|(r, _)| r.iter().cmp(row.iter()).is_lt())
                {
                    out.next();
                }
                let (_, mut target) = out.next().ok_or_else(|| {
                    TbError::Other("the static Floquet support lost an input row".into())
                })?;
                target.assign(&h0.index_axis(Axis(0), ir));
                if subtract_bare {
                    target -= &self.ham.index_axis(Axis(0), ir);
                }
            }
            drop(out);
            place_origin_first(&mut ham, &mut ham_r)?;
            enforce_real_space_hermiticity(&mut ham, &ham_r)?;
            return Ok((ham, ham_r));
        }
        let grid = KspaceGrid::for_polynomial(&self.hamR, DIM, degree)?;
        let concurrent_slabs = workspace_concurrency(
            &grid,
            slots,
            nsta,
            Some(if order == 3 { harmonic_max as usize } else { 0 }),
            sweep,
        )?;
        let mut accumulator = self.effective_accumulator(order, harmonic_max)?;
        inverse_polynomial(
            &grid,
            nsta,
            slots,
            concurrent_slabs,
            &mut accumulator,
            |kz, mut harmonics, assembled| {
                transform::slice_harmonics_into(
                    harmonic_cache,
                    used_max,
                    nsta,
                    &grid,
                    kz,
                    harmonics.view_mut(),
                )?;
                vanvleck::assemble_pointwise(
                    harmonics.view(),
                    harmonic_max,
                    order,
                    true,
                    true,
                    drive.omega0_ev,
                    assembled,
                )
            },
        )?;
        self.finish_effective_correction(accumulator, &rows, h0, subtract_bare)
    }

    /// Visit prevalidated requests in input order using one driven harmonic cache.
    pub(super) fn floquet_effective_sweep_kspace(
        &self,
        drive: &FloquetDrive,
        cache: &FloquetHarmonicCache,
        has_ac: bool,
        requests: &[(usize, isize)],
        visit: impl FnMut((usize, isize), Model<SPIN, DIM, R>) -> Result<()>,
    ) -> Result<()> {
        sweep::visit_effective(self, drive, cache, has_ac, requests, visit)
    }

    fn validate_kspace_cache(&self, cache: &FloquetHarmonicCache, used_max: isize) -> Result<()> {
        let nsta = self.nsta();
        let n_r = self.hamR.nrows();
        validate_effective_cache_layout(used_max, n_r, nsta)?;
        if cache.harmonic_min > -used_max
            || cache.harmonic_max < used_max
            || cache.blocks.shape()[1..] != [n_r, nsta, nsta]
        {
            return Err(TbError::Other(
                "the Floquet harmonic cache does not cover the required harmonics and model rows"
                    .into(),
            ));
        }
        Ok(())
    }

    fn sorted_effective_rows(&self) -> Vec<usize> {
        let mut rows: Vec<_> = (0..self.hamR.nrows()).collect();
        rows.sort_unstable_by(|&a, &b| self.hamR.row(a).iter().cmp(self.hamR.row(b).iter()));
        rows
    }

    fn effective_accumulator(&self, order: usize, h: isize) -> Result<blocks::DenseBlocks> {
        let (bounds, support) = blocks::effective_support(&self.hamR, DIM, h, order)?;
        blocks::DenseBlocks::new(bounds, support, self.nsta(), DIM)
    }

    fn finish_effective_correction(
        &self,
        mut accumulator: blocks::DenseBlocks,
        rows: &[usize],
        h0: ArrayView3<'_, C>,
        subtract_bare: bool,
    ) -> Result<(Array3<C>, Array2<isize>)> {
        let nsta = self.nsta();
        // H_0 is already in real space, so never take an FFT round trip for
        // the baseline. Add each aligned row to the canonical half.
        // Within a negation-closed sorted support the negative partner is at
        // the reversed index. Average both cached blocks just like order zero.
        for (&i_r, &j_r) in rows.iter().zip(rows.iter().rev()) {
            let row = self.hamR.row(i_r);
            let mut vector = [0; 3];
            for axis in 0..DIM {
                vector[axis] = row[axis];
            }
            let slot = accumulator.slot(&vector);
            if slot != blocks::NO_SLOT {
                let block = Array2::from_shape_fn((nsta, nsta), |(i, j)| {
                    let value = |r, i, j| {
                        let h = h0[[r, i, j]];
                        if subtract_bare {
                            h - self.ham[[r, i, j]]
                        } else {
                            h
                        }
                    };
                    let a = value(i_r, i, j);
                    let b = value(j_r, j, i).conj();
                    C::new(a.re.midpoint(b.re), a.im.midpoint(b.im))
                });
                accumulator.add_slot(slot, C::new(1.0, 0.0), block.view());
            }
        }
        finish_checked(accumulator)
    }
}

fn first_active_harmonic(cache: &FloquetHarmonicCache, limit: isize) -> Option<isize> {
    (1..=limit).find(|&n| {
        [n, -n].into_iter().any(|n| {
            cache
                .harmonic_blocks(n)
                .iter()
                .any(|z| *z != C::new(0.0, 0.0))
        })
    })
}

/// Evaluate AB + BA on the zero-aliasing quadratic grid. Both operands
/// use ham_r, with B = A† (the real-space adjoint reverses R). Returns the
/// exact doubled support, origin first, and blocks paired by adjunction.
/// Restore the original units by multiplying each block by the supplied
/// power of two **twice**, along with the derivative and frequency factors.
pub(super) fn kspace_anticommutator(
    a: ArrayView3<'_, C>,
    b: ArrayView3<'_, C>,
    ham_r: &Array2<isize>,
) -> Result<(Array3<C>, Array2<isize>, f64)> {
    let nsta = a.shape()[1];
    let shape = [ham_r.nrows(), nsta, nsta];
    if a.shape() != shape || b.shape() != shape {
        return Err(TbError::Other(
            "k-space anticommutator block shape mismatch".into(),
        ));
    }
    let largest = a
        .iter()
        .chain(b.iter())
        .map(|z| z.re.abs().max(z.im.abs()))
        .fold(0.0, f64::max);
    // Reserve headroom for the unnormalized sums, matrix contraction and
    // inverse transform before multiplying any two gradient blocks.
    let shift = (largest.log2().floor() as i32).saturating_sub(400).max(0);
    let restore = 2.0_f64.powi(shift);
    let input_factor = restore.recip();
    // Preserve the inputs and keep headroom for the quadratic products.
    // Even a lossless shift can erase a small-times-small term before its
    // units are restored. Conservatively require normal squares for every
    // nonzero component when the operands actually need a downshift.
    if shift != 0
        && a.iter().chain(b.iter()).any(|z| {
            [z.re, z.im].into_iter().any(|part| {
                let scaled = part * input_factor;
                part != 0.0 && (scaled * restore != part || !(scaled * scaled).is_normal())
            })
        })
    {
        return Err(TbError::Other(
            "the k-space anticommutator operands exceed the representable product range".into(),
        ));
    }
    let dim = ham_r.ncols();
    let grid = KspaceGrid::for_polynomial(ham_r, dim, 2)?;
    let concurrent_slabs = workspace_concurrency(&grid, 2, nsta, None, None)?;
    let (support_box, support) = blocks::quadratic_support(ham_r, dim)?;
    let mut accumulator = blocks::DenseBlocks::new(support_box, support, nsta, dim)?;
    inverse_polynomial(
        &grid,
        nsta,
        2,
        concurrent_slabs,
        &mut accumulator,
        |kz, scratch, mut assembled| {
            let (mut left, mut right) = scratch.split_at(Axis(0), 1);
            let mut left = left.index_axis_mut(Axis(0), 0);
            let mut right = right.index_axis_mut(Axis(0), 0);
            transform::forward_slice_into(a, &grid, kz, left.view_mut(), input_factor)?;
            transform::forward_slice_into(b, &grid, kz, right.view_mut(), input_factor)?;
            for ((a_k, b_k), mut target) in left
                .outer_iter()
                .zip(right.outer_iter())
                .zip(assembled.outer_iter_mut())
            {
                let one = C::new(1.0, 0.0);
                general_mat_mul(one, &a_k, &b_k, C::new(0.0, 0.0), &mut target);
                general_mat_mul(one, &b_k, &a_k, one, &mut target);
            }
            Ok(())
        },
    )?;
    let (blocks, rows) = finish_checked(accumulator)?;
    Ok((blocks, rows, restore))
}

/// Return the maximum number of simultaneously live slab buffers.
fn workspace_concurrency(
    grid: &KspaceGrid,
    slots: usize,
    nsta: usize,
    van_vleck: Option<usize>,
    sweep: Option<SweepWorkspace>,
) -> Result<usize> {
    // Include the two products and the two scaled inputs of the extreme-range
    // second-order retry, even when cubic normalization scratch is live.
    let scratch_blocks = match van_vleck {
        None => 0,
        Some(0) => 7,
        Some(h) => 8 * h as u128 + 13,
    };
    let views = if van_vleck.is_some() {
        size_of::<ArrayViewMut2<'_, C>>() as u128
    } else {
        0
    };
    let (slabs, one_slab_bytes) =
        workspace_slab_limit(grid, slots, nsta, scratch_blocks, views, sweep);
    if slabs > 0 {
        return Ok(slabs);
    }
    Err(TbError::Other(format!(
        "the k-space convolution needs about {:.1} GiB with one slab; reduce the support, harmonic cutoff, or order",
        one_slab_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    )))
}

#[derive(Clone, Copy)]
struct SweepWorkspace {
    requests: usize,
    cache_bytes: u128,
    has_position: bool,
}

impl SweepWorkspace {
    fn check_static_support_growth(self, rows: usize, nsta: usize, dim: usize) -> Result<()> {
        self.check_static_output(rows, nsta, dim)?;
        // The input, accumulated support, current layer, and next layer can
        // coexist. Reserve room for the tree nodes as well as their keys.
        let tree_bytes = 4 * (dim * size_of::<isize>() + 8 * size_of::<usize>()) as u128;
        if self.cache_bytes.saturating_add(rows as u128 * tree_bytes)
            > KSPACE_MEMORY_HARD_LIMIT.min(isize::MAX as u128)
        {
            return Err(TbError::Other(
                "the Floquet sweep support enumeration exceeds its memory budget".into(),
            ));
        }
        Ok(())
    }

    fn check_static_output(self, rows: usize, nsta: usize, dim: usize) -> Result<()> {
        let rows = rows as u128;
        let block = (nsta as u128).pow(2).saturating_mul(size_of::<C>() as u128);
        let ham = rows.saturating_mul(block);
        let metadata =
            rows * (dim as u128 * size_of::<isize>() as u128 + size_of::<usize>() as u128);
        // Origin-first reordering can temporarily copy the full Hamiltonian.
        // A nonzero conjugate pair can use five temporary matrices.
        let reorder = ham.saturating_add(metadata);
        let hermiticity = block
            .saturating_mul(if rows > 1 { 5 } else { 3 })
            .saturating_add(rows * (size_of::<usize>() + size_of::<bool>()) as u128);
        let position = if self.has_position {
            ham.saturating_mul(dim as u128)
        } else {
            0
        };
        let bytes = self
            .cache_bytes
            .saturating_add(ham)
            .saturating_add(metadata)
            .saturating_add(reorder.max(hermiticity).max(position));
        if bytes > KSPACE_MEMORY_HARD_LIMIT.min(isize::MAX as u128) {
            return Err(TbError::Other(format!(
                "the Floquet sweep needs about {:.1} GiB for its harmonic cache and static output; reduce the support or harmonic cutoff",
                bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )));
        }
        Ok(())
    }
}

/// Return the permitted slab concurrency (zero if none) and one-slab estimate.
fn workspace_slab_limit(
    grid: &KspaceGrid,
    slots: usize,
    nsta: usize,
    scratch_blocks: u128,
    views_per_point: u128,
    sweep: Option<SweepWorkspace>,
) -> (usize, u128) {
    let requests = sweep.map_or(1, |batch| batch.requests as u128);
    let cache_bytes = sweep.map_or(0, |batch| batch.cache_bytes);
    let lanes = grid.points_per_slice() as u128;
    let points = grid.total_points() as u128;
    let threads = rayon::current_num_threads() as u128;
    let netlib = cfg!(any(feature = "netlib-static", feature = "netlib-system"));
    let matrix_elements = (nsta as u128).pow(2);
    let block_bytes = matrix_elements.saturating_mul(size_of::<C>() as u128);
    // One correction grid per request plus one full real-space Hamiltonian.
    // Account for the output's row labels, dense slot map, support vectors,
    // and populated FFT coordinates separately.
    let grid_metadata = (size_of::<usize>() * grid.dim()
        + size_of::<u32>()
        + 2 * size_of::<[isize; 3]>()
        + size_of::<(usize, usize, u32)>()) as u128;
    let base = cache_bytes.saturating_add(
        points.saturating_mul(
            block_bytes
                .saturating_mul(requests + 1)
                .saturating_add(grid_metadata),
        ),
    );
    let position = if sweep.is_some_and(|batch| batch.has_position) {
        // At most one emitted position matrix is live, after its inverse FFT.
        points
            .saturating_mul(block_bytes)
            .saturating_mul(grid.dim() as u128)
    } else {
        0
    };
    // Reference Netlib uses process globals. Do not overlap its matrix products
    // across slabs, although the transforms themselves can still use Rayon.
    let max_slabs = (grid.slices() as u128).min(if netlib { 1 } else { threads });
    // Both solvers write straight to their correction grids. Each sweep slab
    // additionally holds its pointwise kernel's sorted request descriptors.
    let descriptors = if requests > 1 {
        requests * size_of::<(isize, usize, usize)>() as u128
    } else {
        0
    };
    let per_slab = lanes
        .saturating_mul(
            (slots as u128)
                .saturating_mul(block_bytes)
                .saturating_add(views_per_point),
        )
        .saturating_add(descriptors);
    let in_slice_axis = (0..grid.slice_axes())
        .map(|d| grid.axis_len(d))
        .max()
        .unwrap_or(1) as u128;
    // Allow for FFT plans and their private lane scratch. The last-axis
    // transform starts only after the input slab/pointwise buffers are gone.
    let fft_bytes = 8 * size_of::<C>() as u128;
    let inverse = if grid.slices() > 1 {
        let final_lanes = lanes.saturating_mul(matrix_elements);
        (threads.min(final_lanes) + 1)
            .saturating_mul(grid.slices() as u128)
            .saturating_mul(fft_bytes)
    } else {
        0
    };
    let estimate = |slabs: u128| {
        let points_in_flight = lanes * slabs;
        let workers = if netlib {
            1
        } else {
            threads.min(points_in_flight)
        };
        let front_fft = (threads + 2 * slabs)
            .saturating_mul(in_slice_axis)
            .saturating_mul(fft_bytes);
        let assemble = per_slab
            .saturating_mul(slabs)
            .saturating_add(
                workers
                    .saturating_mul(scratch_blocks)
                    .saturating_mul(block_bytes),
            )
            .saturating_add(front_fft);
        base.saturating_add(assemble.max(inverse).max(position))
    };
    let limit = KSPACE_MEMORY_HARD_LIMIT.min(isize::MAX as u128);
    for slabs in (1..=max_slabs).rev() {
        if estimate(slabs) <= limit {
            return (slabs as usize, estimate(1));
        }
    }
    (0, estimate(1))
}

/// Give each task a disjoint part of the output grid and one reusable buffer
/// for the input harmonic slab. The full harmonic volume is never stored.
fn inverse_polynomial(
    grid: &KspaceGrid,
    nsta: usize,
    scratch_slots: usize,
    concurrent_slabs: usize,
    accumulator: &mut blocks::DenseBlocks,
    assemble: impl Fn(usize, ArrayViewMut4<'_, C>, ArrayViewMut3<'_, C>) -> Result<()> + Sync,
) -> Result<()> {
    let slices = grid.slices();
    let lanes = grid.points_per_slice();
    let mut volume = Array4::<C>::zeros((slices, lanes, nsta, nsta));
    let mut rest = volume.view_mut();
    let mut tasks = Vec::with_capacity(concurrent_slabs);
    let mut first = 0;
    for task in 0..concurrent_slabs {
        let len = slices / concurrent_slabs + usize::from(task < slices % concurrent_slabs);
        let (part, remain) = rest.split_at(Axis(0), len);
        tasks.push((first, part));
        rest = remain;
        first += len;
    }
    tasks
        .into_par_iter()
        .try_for_each(|(first, mut part)| -> Result<()> {
            let mut scratch = Array4::zeros((scratch_slots, lanes, nsta, nsta));
            for (i, mut slab) in part.outer_iter_mut().enumerate() {
                assemble(first + i, scratch.view_mut(), slab.view_mut())?;
                transform::inverse_slice_inplace(slab, grid)?;
            }
            Ok(())
        })?;
    gather_correction(grid, volume.view_mut(), accumulator)
}

fn gather_correction(
    grid: &KspaceGrid,
    mut volume: ArrayViewMut4<'_, C>,
    accumulator: &mut blocks::DenseBlocks,
) -> Result<()> {
    transform::inverse_streaming_axis(volume.view_mut())?;
    for (residue, lane, slot) in output_slots(grid, accumulator) {
        accumulator.set_slot(
            slot,
            volume
                .index_axis(Axis(0), residue)
                .index_axis(Axis(0), lane),
        );
    }
    Ok(())
}

fn finish_checked(accumulator: blocks::DenseBlocks) -> Result<(Array3<C>, Array2<isize>)> {
    let out = accumulator.finish()?;
    if out.0.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(TbError::Other(
            "the k-space convolution produced non-finite blocks".into(),
        ));
    }
    Ok(out)
}

/// Map each stored output row to its folded FFT coordinate; read in grid order.
fn output_slots(grid: &KspaceGrid, accumulator: &blocks::DenseBlocks) -> Vec<(usize, usize, u32)> {
    let last = grid.dim() - 1;
    let mut slots = Vec::with_capacity(accumulator.stored_blocks());
    for (r, slot) in accumulator.populated_slots() {
        let lane = r[..last].iter().enumerate().fold(0, |lane, (axis, &r)| {
            lane * grid.axis_len(axis) + grid.folded(r, axis)
        });
        slots.push((grid.folded(r[last], last), lane, slot));
    }
    slots.sort_unstable();
    slots
}

#[cfg(test)]
#[test]
fn independent_slabs_overlap_on_the_rayon_pool() {
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    if cfg!(any(feature = "netlib-static", feature = "netlib-system")) {
        return;
    }
    let input = ndarray::array![[0], [-1], [1]];
    let grid = KspaceGrid::for_polynomial(&input, 1, 2).unwrap();
    let (bounds, support) = blocks::effective_support(&input, 1, 1, 1).unwrap();
    let mut accumulator = blocks::DenseBlocks::new(bounds, support, 1, 1).unwrap();
    let gate = (Mutex::new(0), Condvar::new());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    pool.install(|| {
        let slabs = workspace_concurrency(&grid, 1, 1, None, None).unwrap();
        assert_eq!(slabs, 2);
        inverse_polynomial(&grid, 1, 1, slabs, &mut accumulator, |k, _, mut out| {
            if k == 0 || k == grid.slices().div_ceil(2) {
                let mut count = gate.0.lock().unwrap();
                *count += 1;
                gate.1.notify_all();
                let (count, _) = gate
                    .1
                    .wait_timeout_while(count, Duration::from_secs(5), |n| *n < 2)
                    .unwrap();
                assert_eq!(*count, 2, "two slices should run concurrently");
            }
            out.fill(C::new(2.0, 0.0));
            Ok(())
        })
        .unwrap();
    });
    let (out, _) = accumulator.finish().unwrap();
    assert!((out[[0, 0, 0]] - C::new(2.0, 0.0)).norm() < 1e-14);
}

#[cfg(test)]
#[test]
fn cubic_workspace_counts_parallel_generator_buffers() {
    if cfg!(any(feature = "netlib-static", feature = "netlib-system")) {
        return;
    }
    let input = ndarray::array![[0, 0], [-2, 0], [2, 0]];
    let grid = KspaceGrid::for_polynomial(&input, 2, 4).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(16)
        .build()
        .unwrap();
    pool.install(|| {
        assert_eq!(
            workspace_concurrency(&grid, 193, 1_000, Some(0), None).unwrap(),
            1
        );
        assert!(workspace_concurrency(&grid, 193, 1_000, Some(32), None).is_err());
    });
}

#[cfg(test)]
#[test]
fn workspace_counts_output_metadata_for_small_matrices() {
    // Test the bounding grid without constructing hundreds of millions of rows.
    let input = ndarray::array![[0, 0], [5_625, 5_625], [-5_625, -5_625]];
    let grid = KspaceGrid::for_polynomial(&input, 2, 2).unwrap();
    assert!(workspace_concurrency(&grid, 3, 2, None, None).is_err());
}

#[cfg(test)]
#[test]
fn workspace_counts_simultaneous_retry_matrices() {
    let input = ndarray::array![[0]];
    let grid = KspaceGrid::for_polynomial(&input, 1, 3).unwrap();
    assert!(workspace_concurrency(&grid, 5, 18_000, Some(0), None).is_err());
}

#[cfg(test)]
#[test]
fn workspace_counts_only_live_fft_and_pointwise_workers() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    pool.install(|| {
        let grid = KspaceGrid::for_polynomial(&ndarray::array![[0], [12], [-12]], 1, 2).unwrap();
        assert!(workspace_concurrency(&grid, 2, 4_000, None, None).is_ok());
        let grid =
            KspaceGrid::for_polynomial(&ndarray::array![[0], [2_000_000], [-2_000_000]], 1, 2)
                .unwrap();
        // One matrix element permits just one final-axis FFT lane.
        assert!(workspace_concurrency(&grid, 2, 1, None, None).is_ok());
    });
}

#[cfg(test)]
#[test]
fn output_slots_only_visit_populated_representatives() {
    let input = ndarray::array![
        [0, 0, 0],
        [1, -10, 20],
        [-1, 10, -20],
        [0, 3, 20],
        [0, -3, -20]
    ];
    let grid = KspaceGrid::for_polynomial(&input, 3, 1).unwrap();
    let (bounds, support) = blocks::effective_support(&input, 3, 0, 0).unwrap();
    let acc = blocks::DenseBlocks::new(bounds, support, 1, 3).unwrap();
    let ny = grid.axis_len(1);
    assert_eq!(
        output_slots(&grid, &acc),
        [(0, 0, 0), (20, 3, 3), (20, 2 * ny - 10, 4)]
    );
}
