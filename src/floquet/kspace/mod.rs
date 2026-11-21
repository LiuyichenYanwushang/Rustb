//! Streaming k-space evaluation of the Floquet effective Hamiltonian.
//!
//! The real-space path convolves hopping blocks pairwise, which costs
//! O(N_R^2) matrix products. Every order-two van Vleck term is instead a
//! polynomial of bounded degree in the harmonic blocks, so the same operator
//! can be evaluated on a zero-aliasing k-point grid in O(N_k n^3) products and
//! transformed back once. This module implements that engine.
//!
//! Streaming layout: the last resolved axis is sampled slice by slice. For each
//! k-slice the harmonics are reduced along that axis, transformed over the
//! remaining axes, assembled pointwise, transformed back, and accumulated into
//! the output blocks with the slice phase. Peak memory is one slice of harmonic
//! blocks plus one slice of assembled blocks, not the full k-point volume.

use super::*;
use ndarray::{Array2, Array3};
use num_complex::Complex;
use std::collections::BTreeMap;
use std::f64::consts::TAU;

mod grid;
#[cfg(test)]
mod perf;
mod transform;
mod vanvleck;

use grid::KspaceGrid;

/// Complex scalar used throughout the Floquet kernels.
pub(super) type C = Complex<f64>;

/// Which evaluation path the effective-model constructor uses.
///
/// Public callers get Auto; the other variants exist so regression tests can
/// pin one implementation against the other.
/// Constructed by tests to pin one implementation; production uses `Auto`.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EffectivePath {
    /// Choose by problem size: k space for large supports, real space otherwise.
    Auto,
    /// Force the pairwise real-space convolution.
    RealSpace,
    /// Force the streaming k-space engine.
    Kspace,
}

/// Automatic selection refuses an estimated engine working set above this
/// bound; an explicit request is only refused above the hard limit.
///
/// The estimate covers the allocations the engine itself controls: the
/// harmonic slab plus two dense copies of the output model (the accumulation
/// map and the final Hamiltonian). It does not cover the harmonic cache, which
/// the caller builds before the path is chosen and shares with the real-space
/// path. The bound is therefore an engine working-set bound, not a process
/// peak: the measured 55-orbital benchmark peaks at 9.7 GiB against a 7.5 GiB
/// estimate because of that cache plus allocator overhead.
const KSPACE_MEMORY_LIMIT: u128 = 32 * 1024 * 1024 * 1024;
const KSPACE_MEMORY_HARD_LIMIT: u128 = 64 * 1024 * 1024 * 1024;

/// Estimated cost and footprint of one k-space evaluation.
#[derive(Clone, Copy, Debug)]
struct KspaceEstimate {
    /// Total number of sampled k points.
    total_points: u128,
    /// One slab of harmonic blocks, the largest transient allocation.
    slab_bytes: u128,
    /// Two dense copies of the output model (accumulation map plus final ham).
    output_bytes: u128,
}

impl KspaceEstimate {
    fn working_set_bytes(&self) -> u128 {
        self.slab_bytes.saturating_add(self.output_bytes)
    }
}

impl<const SPIN: bool, const DIM: usize, RMatrix: RMatrixData> Model<SPIN, DIM, RMatrix> {
    /// Decide whether the streaming engine should handle this model.
    ///
    /// The automatic choice requires that there are corrections to compute
    /// (order 1 or 2 with an AC drive), that the k-space grid needs strictly
    /// less matrix work than one convolution per input pair, and that the whole
    /// working set fits. An explicit request wins over the work estimate but
    /// still respects the hard memory limit inside the engine.
    pub(super) fn prefer_kspace_effective_model(
        &self,
        path: EffectivePath,
        harmonic_max: isize,
        order: usize,
        has_time_dependence: bool,
        omega0_ev: f64,
    ) -> bool {
        match path {
            EffectivePath::Kspace => true,
            EffectivePath::RealSpace => false,
            EffectivePath::Auto => {
                if order == 0 || harmonic_max <= 0 || !has_time_dependence {
                    // The k-space answer would just be H_0 at the price of a
                    // full FFT round trip; the static path is exact and cheap.
                    return false;
                }
                // The assembly validates the frequency scale whenever it is
                // asked to compute corrections, while the real-space path only
                // touches the scale for a nonzero term. Keep calls whose scale
                // is not representable on the exact real-space path.
                let scale_is_finite = if order >= 2 {
                    let inverse = omega0_ev.recip();
                    inverse.is_finite() && (inverse * inverse).is_finite()
                } else {
                    omega0_ev.recip().is_finite()
                };
                if !scale_is_finite {
                    return false;
                }
                let estimate = self.kspace_estimate(harmonic_max, order, has_time_dependence);
                let pairs = (self.hamR.nrows() as u128).saturating_mul(self.hamR.nrows() as u128);
                // The assembly performs a handful of matrix products per k
                // point while one convolution performs a handful per pair, so
                // demand a factor two fewer sampled points than pairs before
                // switching. A support radius large enough to blow up the grid
                // therefore stays in real space instead of taking a grid that
                // is orders of magnitude larger than the pair set.
                let work_wins = estimate.total_points.saturating_mul(2) <= pairs;
                work_wins && estimate.working_set_bytes() <= KSPACE_MEMORY_LIMIT
            }
        }
    }

    /// Grid, sampled points and footprint for the streaming engine, computed
    /// without allocating any of it.
    fn kspace_estimate(
        &self,
        harmonic_max: isize,
        order: usize,
        has_time_dependence: bool,
    ) -> KspaceEstimate {
        // Order zero keeps no harmonic at all, so it must neither index nor
        // size anything by harmonic_max (which may be huge on an order-0 call).
        let used_max = if has_time_dependence && order >= 1 {
            harmonic_max.max(0)
        } else {
            0
        };
        let degree = if harmonic_max > 0 { order + 1 } else { 1 };
        let nsta = self.nsta() as u128;
        let element = std::mem::size_of::<C>() as u128;
        let mut shape = [1_usize; 3];
        for axis in 0..DIM {
            let radius = self
                .hamR
                .column(axis)
                .iter()
                .map(|r| r.unsigned_abs())
                .max()
                .unwrap_or(0);
            let span = radius
                .checked_mul(2 * degree)
                .and_then(|value| value.checked_add(1))
                .unwrap_or(usize::MAX);
            shape[axis] = if span == usize::MAX {
                usize::MAX
            } else {
                grid::next_smooth(span.max(1))
            };
        }
        let mut lanes = 1_u128;
        let mut total_points = 1_u128;
        for (axis, length) in shape.iter().enumerate() {
            total_points = total_points.saturating_mul(*length as u128);
            if axis + 1 < DIM {
                lanes = lanes.saturating_mul(*length as u128);
            }
        }
        let slab_bytes = (2 * used_max + 1) as u128 * lanes * nsta * nsta * element;
        // The output support is contained in the degree times radius box, and
        // the engine holds two dense copies of it (map plus final ham).
        let mut support = 1_u128;
        for axis in 0..DIM {
            let radius = self
                .hamR
                .column(axis)
                .iter()
                .map(|r| r.unsigned_abs())
                .max()
                .unwrap_or(0);
            let extent = radius.saturating_mul(2 * degree).saturating_add(1) as u128;
            support = support.saturating_mul(extent);
        }
        let output_bytes = 2 * support * nsta * nsta * element;
        KspaceEstimate {
            total_points,
            slab_bytes,
            output_bytes,
        }
    }

    /// Build the effective model with the streaming k-space engine.
    ///
    /// Returns the same real-space model as the pairwise path: identical
    /// support (including zero-valued Minkowski layers), identical blocks up to
    /// floating-point round-off.
    pub(super) fn floquet_effective_model_kspace(
        &self,
        drive: &FloquetDrive,
        options: &FloquetEffectiveOptions,
        harmonic_cache: &FloquetHarmonicCache,
        harmonic_max: isize,
        has_time_dependence: bool,
    ) -> Result<Model<SPIN, DIM, RMatrix>> {
        let nsta = self.nsta();
        let order = options.order;
        // Only harmonics up to harmonic_max enter after the self-consistent
        // window, so the transform needs at most 2*harmonic_max+1 blocks. Order
        // zero keeps none, so it must not size an array from harmonic_max.
        let used_max = if has_time_dependence && order >= 1 {
            harmonic_max
        } else {
            0
        };
        let degree = if harmonic_max > 0 { order + 1 } else { 1 };
        // The working-set estimate covers the slab and both dense copies of the
        // output model, which dominate memory for a large support.
        let estimate = self.kspace_estimate(harmonic_max, order, has_time_dependence);
        if estimate.working_set_bytes() > KSPACE_MEMORY_HARD_LIMIT {
            return Err(TbError::Other(format!(
                "the k-space effective model needs about {:.1} GiB (slab {:.1} GiB plus two output copies); reduce the support or use the real-space path",
                estimate.working_set_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                estimate.slab_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            )));
        }
        let grid = KspaceGrid::for_polynomial(&self.hamR, DIM, degree)?;
        // A window whose every harmonic is exactly zero carries no correction,
        // so the assembly must not demand a frequency scaling for it. This
        // matches the real-space path, which skips exact-zero operands rather
        // than rejecting a vanishing frequency scale.
        let has_corrections = has_time_dependence
            && order >= 1
            && (-used_max..=used_max)
                .filter(|n| {
                    *n != 0
                        && *n >= harmonic_cache.harmonic_min
                        && *n <= harmonic_cache.harmonic_max
                })
                .any(|n| {
                    harmonic_cache
                        .harmonic_blocks(n)
                        .iter()
                        .any(|value| value.re != 0.0 || value.im != 0.0)
                });
        let support = effective_support(&self.hamR, DIM, harmonic_max, order)?;
        let mut blocks: BTreeMap<[isize; 3], Array2<C>> = support
            .into_iter()
            .map(|vector| (vector, Array2::zeros((nsta, nsta))))
            .collect();

        let slices = grid.slices();
        let lanes = grid.points_per_slice();
        // inverse_slice applies 1 / (N_0 * ... * N_{dim-2}); the streaming axis
        // needs its own 1 / N_last factor here.
        let slice_norm = 1.0 / slices as f64;
        // Lane -> translation on the non-streaming axes, decoded once.
        let lane_vectors: Vec<[isize; 3]> = (0..lanes)
            .map(|lane| {
                let mut vector = [0_isize; 3];
                let mut rest = lane;
                for axis in (0..DIM - 1).rev() {
                    let length = grid.axis_len(axis);
                    let index = rest % length;
                    rest /= length;
                    vector[axis] = grid.signed(index, axis);
                }
                vector
            })
            .collect();
        // The harmonic slab and the two slice buffers are reused across kz.
        let mut harmonics = Array4::<C>::zeros(((2 * used_max + 1) as usize, lanes, nsta, nsta));
        let mut assembled = Array3::<C>::zeros((lanes, nsta, nsta));
        let mut real = Array3::<C>::zeros((lanes, nsta, nsta));
        for kz in 0..slices {
            transform::slice_harmonics_into(
                harmonic_cache,
                used_max,
                nsta,
                &grid,
                kz,
                harmonics.view_mut(),
            )?;
            assembled.fill(C::new(0.0, 0.0));
            vanvleck::assemble_pointwise(
                harmonics.view(),
                used_max,
                order,
                has_corrections,
                drive.omega0_ev,
                assembled.view_mut(),
            )?;
            transform::inverse_slice_into(assembled.view(), &grid, real.view_mut())?;
            for r_last in 0..slices {
                // Reduce the integer turn before converting to a float so the
                // phase stays exact for long streaming axes.
                let turn = (kz as i128 * r_last as i128) % slices as i128;
                let phase = C::from_polar(slice_norm, -TAU * turn as f64 / slices as f64);
                let last = grid.signed(r_last, DIM - 1);
                for (lane, block) in real.outer_iter().enumerate() {
                    let mut vector = lane_vectors[lane];
                    vector[DIM - 1] = last;
                    if let Some(target) = blocks.get_mut(&vector) {
                        target.scaled_add(phase, &block);
                    }
                }
            }
        }

        let mut ham = Array3::<Complex<f64>>::zeros((blocks.len(), nsta, nsta));
        let mut ham_r = Array2::<isize>::zeros((blocks.len(), DIM));
        for (index, (vector, block)) in blocks.into_iter().enumerate() {
            for axis in 0..DIM {
                ham_r[[index, axis]] = vector[axis];
            }
            ham.index_axis_mut(Axis(0), index).assign(&block);
        }
        place_origin_first(&mut ham, &mut ham_r)?;
        enforce_real_space_hermiticity(&mut ham, &ham_r)?;
        self.floquet_effective_output(ham, ham_r)
    }
}

/// Exact real-space support of the effective model, in lexicographic order.
///
/// Mirrors the pairwise path: the input support is always present, the pairwise
/// Minkowski layer appears when harmonic_max > 0 and order >= 1, and the triple
/// layer when harmonic_max > 0 and order >= 2.
fn effective_support(
    support: &Array2<isize>,
    dim: usize,
    harmonic_max: isize,
    order: usize,
) -> Result<Vec<[isize; 3]>> {
    let mut vectors = std::collections::BTreeSet::new();
    for row in support.outer_iter() {
        vectors.insert(fold_row(row, dim)?);
    }
    if harmonic_max > 0 && order >= 1 {
        let pairs = minkowski(&vectors, support, dim)?;
        vectors.extend(pairs.iter().copied());
        if order >= 2 {
            vectors.extend(minkowski(&pairs, support, dim)?);
        }
    }
    Ok(vectors.into_iter().collect())
}

fn fold_row(row: ndarray::ArrayView1<'_, isize>, dim: usize) -> Result<[isize; 3]> {
    let mut vector = [0_isize; 3];
    for axis in 0..dim {
        vector[axis] = row[axis];
    }
    Ok(vector)
}

fn minkowski(
    left: &std::collections::BTreeSet<[isize; 3]>,
    right: &Array2<isize>,
    dim: usize,
) -> Result<std::collections::BTreeSet<[isize; 3]>> {
    let mut out = std::collections::BTreeSet::new();
    for lhs in left {
        for row in right.outer_iter() {
            let mut vector = [0_isize; 3];
            for axis in 0..dim {
                vector[axis] = lhs[axis].checked_add(row[axis]).ok_or_else(|| {
                    TbError::Other(format!(
                        "k-space support sum overflows isize on axis {axis}: {} + {}",
                        lhs[axis], row[axis]
                    ))
                })?;
            }
            out.insert(vector);
        }
    }
    Ok(out)
}
