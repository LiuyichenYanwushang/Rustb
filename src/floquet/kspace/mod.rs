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
//! Streaming layout: the last resolved axis is sampled slice by slice. For each
//! k-slice the harmonics are reduced along that axis, transformed over the
//! remaining axes, assembled pointwise, transformed back, and accumulated into
//! the output blocks with the slice phase. Peak memory is one slice of harmonic
//! blocks plus one slice of assembled blocks, not the full k-point volume.

use super::*;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array3, Array4, ArrayView3, ArrayViewMut3};
use num_complex::Complex;
use std::f64::consts::TAU;

mod blocks;
mod grid;
#[cfg(test)]
mod perf;
mod transform;
mod vanvleck;

use grid::KspaceGrid;

/// Complex scalar used throughout the Floquet kernels.
pub(super) type C = Complex<f64>;

// Bound the FFT working set (not the already-built harmonic cache or the
// position array owned by the returned model). A sparse large-radius support
// may exceed the limit even if its input blocks fit in memory.
const KSPACE_MEMORY_HARD_LIMIT: u128 = 64 * 1024 * 1024 * 1024;

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
            order,
            harmonic_max,
            has_time_dependence,
            false,
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
            order,
            harmonic_max,
            has_time_dependence,
            true,
        )
    }

    fn floquet_effective_blocks_kspace(
        &self,
        drive: &FloquetDrive,
        harmonic_cache: &FloquetHarmonicCache,
        order: usize,
        harmonic_max: isize,
        has_time_dependence: bool,
        subtract_bare: bool,
    ) -> Result<(Array3<C>, Array2<isize>)> {
        let used_max = if has_time_dependence {
            effective_cache_max(order, harmonic_max)?
        } else {
            0
        };
        validate_effective_cache_layout(used_max, self.hamR.nrows(), self.nsta())?;
        let slots = (2 * used_max + 1) as usize;
        let nsta = self.nsta();
        let shape = harmonic_cache.blocks.shape();
        if harmonic_cache.harmonic_min > -used_max
            || harmonic_cache.harmonic_max < used_max
            || shape[1..] != [self.hamR.nrows(), nsta, nsta]
        {
            return Err(TbError::Other(
                "the Floquet harmonic cache does not cover the required harmonics and model rows"
                    .into(),
            ));
        }
        let h0 = harmonic_cache.harmonic_blocks(0);
        let degree = if harmonic_max > 0 { order + 1 } else { 1 };
        let mut rows: Vec<_> = (0..self.hamR.nrows()).collect();
        rows.sort_unstable_by(|&a, &b| self.hamR.row(a).iter().cmp(self.hamR.row(b).iter()));
        // A window whose AC harmonics are zero carries no correction, even
        // when the inverse frequency is not representable. Still keep every
        // documented Minkowski layer, exactly zero outside the input rows.
        // Scalar harmonics also commute identically at every k.
        let has_corrections = nsta > 1
            && (1..=harmonic_max.min(used_max)).any(|n| {
                [n, -n].into_iter().any(|n| {
                    harmonic_cache
                        .harmonic_blocks(n)
                        .iter()
                        .any(|z| *z != C::new(0.0, 0.0))
                })
            });
        if !has_corrections {
            let mut ham_r = if degree == 1 {
                self.hamR.select(Axis(0), &rows)
            } else {
                blocks::zero_padded_support::<DIM>(&self.hamR, degree)?
            };
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
        validate_workspace(&grid, slots, nsta)?;
        let (support_box, support) =
            blocks::effective_support(&self.hamR, DIM, harmonic_max, order)?;
        let mut accumulator = blocks::DenseBlocks::new(support_box, support, nsta, DIM)?;
        {
            let mut harmonics = Array4::<C>::zeros((slots, grid.points_per_slice(), nsta, nsta));
            inverse_polynomial(&grid, nsta, &mut accumulator, |kz, assembled| {
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
            })?
        }
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
    validate_workspace(&grid, 2, nsta)?;
    let (support_box, support) = blocks::quadratic_support(ham_r, dim)?;
    let mut accumulator = blocks::DenseBlocks::new(support_box, support, nsta, dim)?;
    let mut left = Array3::zeros((grid.points_per_slice(), nsta, nsta));
    let mut right = Array3::zeros(left.raw_dim());
    inverse_polynomial(&grid, nsta, &mut accumulator, |kz, mut assembled| {
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
    })?;
    let (blocks, rows) = finish_checked(accumulator)?;
    Ok((blocks, rows, restore))
}

fn validate_workspace(grid: &KspaceGrid, slots: usize, nsta: usize) -> Result<()> {
    let lanes = grid.points_per_slice() as u128;
    let points = grid.total_points() as u128;
    let block_bytes = (nsta as u128)
        .pow(2)
        .saturating_mul(std::mem::size_of::<C>() as u128);
    // Two full grid volumes of matrices conservatively cover the output,
    // support and slot arrays, including sparse supports and the padded grid.
    let bytes = (2 * points + (slots as u128 + 2) * lanes).saturating_mul(block_bytes.max(48));
    if bytes > KSPACE_MEMORY_HARD_LIMIT || bytes > isize::MAX as u128 {
        return Err(TbError::Other(format!(
            "the k-space convolution needs about {:.1} GiB; reduce the support, harmonic cutoff, or order",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )));
    }
    Ok(())
}

/// Apply the pointwise polynomial one slab at a time and inverse-transform
/// onto the exact output support. The callback overwrites every output point.
fn inverse_polynomial(
    grid: &KspaceGrid,
    nsta: usize,
    accumulator: &mut blocks::DenseBlocks,
    mut assemble: impl FnMut(usize, ArrayViewMut3<'_, C>) -> Result<()>,
) -> Result<()> {
    let slices = grid.slices();
    let lanes = grid.points_per_slice();
    let slots = phase_slots(grid, accumulator);
    let mut assembled = Array3::<C>::zeros((lanes, nsta, nsta));
    let mut real = Array3::<C>::zeros(assembled.raw_dim());
    for kz in 0..slices {
        assemble(kz, assembled.view_mut())?;
        transform::inverse_slice_into(assembled.view(), grid, real.view_mut())?;
        accumulate_slice(grid, kz, &slots, real.view(), accumulator);
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

/// Only occupied (streaming residue, lane, model row) entries, sorted so
/// each streaming phase is evaluated once and then shared by all its lanes.
fn phase_slots(grid: &KspaceGrid, accumulator: &blocks::DenseBlocks) -> Vec<(usize, usize, u32)> {
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

fn accumulate_slice(
    grid: &KspaceGrid,
    kz: usize,
    slots: &[(usize, usize, u32)],
    real: ArrayView3<'_, C>,
    accumulator: &mut blocks::DenseBlocks,
) {
    let slices = grid.slices();
    // inverse_slice normalized the in-slice FFT. Reduce integral turns
    // before applying the streaming-axis phase and normalization.
    // ponytail: this direct sum keeps just two slice buffers; consider a
    // batched 1D FFT if a densely populated streaming axis dominates timings.
    for group in slots.chunk_by(|a, b| a.0 == b.0) {
        let turn = (kz as i128 * group[0].0 as i128) % slices as i128;
        let phase = C::from_polar(1.0 / slices as f64, -TAU * turn as f64 / slices as f64);
        for &(_, lane, slot) in group {
            accumulator.add_slot(slot, phase, real.index_axis(Axis(0), lane));
        }
    }
}

#[cfg(test)]
#[test]
fn phase_slots_only_visit_populated_representatives() {
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
        phase_slots(&grid, &acc),
        [(0, 0, 0), (20, 3, 3), (20, 2 * ny - 10, 4)]
    );
}
