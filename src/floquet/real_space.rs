//! Private real-space hopping-block machinery for the Floquet effective-model
//! path: origin-first ordering, Minkowski-support convolutions, and the exact
//! `R <-> -R` Hermiticity pairing.  Nothing here is re-exported: the items are
//! `pub(super)` only for the `floquet` parent and its sibling tests.

use crate::error::{Result, TbError};
use crate::model::utils::find_R;
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::{Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, Data, Ix3};
use num_complex::Complex;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

type RealSpaceBlocks = (Vec<Array2<Complex<f64>>>, Array2<isize>);
pub(super) type RealSpaceBlockMap = std::collections::BTreeMap<Vec<isize>, Array2<Complex<f64>>>;

/// Keep public real-space models compatible with origin-first consumers.
/// Other support rows retain their relative order.
pub(super) fn place_origin_first(
    ham: &mut Array3<Complex<f64>>,
    ham_r: &mut Array2<isize>,
) -> Result<()> {
    let zero = Array1::zeros(ham_r.ncols());
    let origin = match find_R(ham_r, &zero) {
        Some(index) => index,
        None => {
            ham.push(
                Axis(0),
                Array2::zeros((ham.len_of(Axis(1)), ham.len_of(Axis(2)))).view(),
            )?;
            ham_r.push_row(zero.view())?;
            ham_r.nrows() - 1
        }
    };
    if origin != 0 {
        let order: Vec<_> = std::iter::once(origin)
            .chain((0..ham_r.nrows()).filter(|&index| index != origin))
            .collect();
        *ham = ham.select(Axis(0), &order);
        *ham_r = ham_r.select(Axis(0), &order);
    }
    Ok(())
}

pub(super) trait RealSpaceBlockSource: Sync {
    fn nblocks(&self) -> usize;
    fn block(&self, index: usize) -> ArrayView2<'_, Complex<f64>>;
}

impl<S> RealSpaceBlockSource for ArrayBase<S, Ix3>
where
    S: Data<Elem = Complex<f64>> + Sync,
{
    fn nblocks(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn block(&self, index: usize) -> ArrayView2<'_, Complex<f64>> {
        self.index_axis(Axis(0), index)
    }
}

impl RealSpaceBlockSource for Vec<Array2<Complex<f64>>> {
    fn nblocks(&self) -> usize {
        self.len()
    }

    fn block(&self, index: usize) -> ArrayView2<'_, Complex<f64>> {
        self[index].view()
    }
}

pub(super) fn accumulate_scaled_real_space_blocks(
    target: &mut RealSpaceBlockMap,
    source_blocks: &[Array2<Complex<f64>>],
    source_r: &Array2<isize>,
    scale: f64,
) -> Result<()> {
    debug_assert_eq!(source_blocks.len(), source_r.nrows());
    if source_blocks.is_empty() {
        return Ok(());
    }
    if source_blocks
        .iter()
        .all(|block| block.iter().all(|value| value.re == 0.0 && value.im == 0.0))
    {
        for row in source_r.outer_iter() {
            target
                .entry(row.to_vec())
                .or_insert_with(|| Array2::<Complex<f64>>::zeros(source_blocks[0].raw_dim()));
        }
        return Ok(());
    }
    if !scale.is_finite() {
        return Err(TbError::Other(
            "a van Vleck frequency scaling factor is non-finite; increase omega0_ev or lower the requested order"
                .to_string(),
        ));
    }
    let scale = Complex::new(scale, 0.0);
    for (i_r, row) in source_r.outer_iter().enumerate() {
        target
            .entry(row.to_vec())
            .and_modify(|block| block.scaled_add(scale, &source_blocks[i_r]))
            .or_insert_with(|| source_blocks[i_r].mapv(|value| scale * value));
    }
    Ok(())
}

/// Merge two real-space block accumulators, preferring to insert the smaller
/// map into the larger one to limit tree lookups and reallocations.
pub(super) fn merge_real_space_block_maps(
    mut left: RealSpaceBlockMap,
    mut right: RealSpaceBlockMap,
) -> RealSpaceBlockMap {
    if left.len() < right.len() {
        std::mem::swap(&mut left, &mut right);
    }
    for (r, block) in right {
        left.entry(r)
            .and_modify(|target| *target += &block)
            .or_insert(block);
    }
    left
}

/// Real-space commutator blocks `comm_n(R) = (AB)(R) − (BA)(R)` for the
/// harmonic pair `A_R = T_n(R)` and `B_R = T_{−n}(R)`:
///
/// ```math
/// (AB)(R) = \sum_{R'} A_{R-R'}\, B_{R'}, \qquad
/// (BA)(R) = \sum_{R'} B_{R-R'}\, A_{R'}.
/// ```
///
/// Both convolutions are accumulated with [`ndarray::linalg::general_mat_mul`],
/// which handles the input storage layouts; the returned support is the Minkowski sum
/// `{R1 + R2 : R1, R2 ∈ hamR}` in lexicographic order.  The result
/// satisfies the Hermiticity pairing `comm(R) = comm(−R)†` exactly — a
/// final pass ([`enforce_real_space_hermiticity`]) averages each ±R pair
/// with its conjugate-transposed partner, removing the summation-order
/// noise of the two independent convolutions.
///
/// # Errors
/// Returns [`TbError::MissingHermitianConjugateHopping`] if the support
/// is not closed under `R -> −R`; that can only happen for hand-built
/// models whose `hamR` itself violates the closure (a `Model` invariant
/// for all constructed models).  Returns [`TbError::Other`] if adding two
/// real-space hopping vectors would overflow `isize`.
///
/// # Panics
/// Debug-asserts that `a_blocks` and `b_blocks` each contain
/// `ham_r.nrows()` square blocks of one common size.
pub(super) fn real_space_commutator<A, B>(
    a_blocks: &A,
    b_blocks: &B,
    ham_r: &Array2<isize>,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    let (blocks, support_rows) =
        real_space_commutator_with_supports(a_blocks, ham_r, b_blocks, ham_r)?;
    let nsta = if a_blocks.nblocks() == 0 {
        0
    } else {
        a_blocks.block(0).nrows()
    };

    // Enforce comm(R) = comm(−R)† exactly (fp symmetrization).
    let mut stacked = Array3::<Complex<f64>>::zeros((blocks.len(), nsta, nsta));
    for (i, block) in blocks.iter().enumerate() {
        stacked.index_axis_mut(Axis(0), i).assign(block);
    }
    enforce_real_space_hermiticity(&mut stacked, &support_rows)?;
    let blocks: Vec<Array2<Complex<f64>>> = (0..blocks.len())
        .map(|i| stacked.index_axis(Axis(0), i).to_owned())
        .collect();

    Ok((blocks, support_rows))
}

/// Real-space commutator for operands with independent hopping supports.
///
/// Unlike [`real_space_commutator`], this low-level helper does not impose a
/// Hermiticity relation on the result: an intermediate nested commutator such
/// as `[H_0,H_m]` is not Hermitian by itself.  The caller must only
/// symmetrize the final effective Hamiltonian after the complete signed
/// harmonic sum has been accumulated.
pub(super) fn real_space_commutator_with_supports<A, B>(
    a_blocks: &A,
    a_r: &Array2<isize>,
    b_blocks: &B,
    b_r: &Array2<isize>,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    real_space_two_product_sum_with_supports(a_blocks, a_r, b_blocks, b_r, -1.0, "commutator")
}

/// Real-space anticommutator `{A,B} = AB + BA` on one common support.
///
/// When `B = A^dagger`, as in the finite-`q` weak-field kernel, the result is
/// Hermitian.  The final pairing pass removes only floating-point accumulation
/// noise and supplies exact `R <-> -R` conjugation to downstream derivatives.
pub(super) fn real_space_anticommutator<A, B>(
    a_blocks: &A,
    b_blocks: &B,
    ham_r: &Array2<isize>,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    let (blocks, support_rows) = real_space_two_product_sum_with_supports(
        a_blocks,
        ham_r,
        b_blocks,
        ham_r,
        1.0,
        "anticommutator",
    )?;
    let nsta = if a_blocks.nblocks() == 0 {
        0
    } else {
        a_blocks.block(0).nrows()
    };
    let mut stacked = Array3::<Complex<f64>>::zeros((blocks.len(), nsta, nsta));
    for (i, block) in blocks.iter().enumerate() {
        stacked.index_axis_mut(Axis(0), i).assign(block);
    }
    enforce_real_space_hermiticity(&mut stacked, &support_rows)?;
    let blocks = (0..blocks.len())
        .map(|i| stacked.index_axis(Axis(0), i).to_owned())
        .collect();
    Ok((blocks, support_rows))
}

#[cfg(test)]
thread_local! {
    // Per-test-thread count: assert bounded work without flaky timing thresholds.
    pub(super) static REAL_SPACE_SUPPORT_CONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Accumulate `AB + reverse_scale * BA` for operands with independent
/// real-space supports.
pub(super) fn real_space_two_product_sum_with_supports<A, B>(
    a_blocks: &A,
    a_r: &Array2<isize>,
    b_blocks: &B,
    b_r: &Array2<isize>,
    reverse_scale: f64,
    operation: &str,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    #[cfg(test)]
    REAL_SPACE_SUPPORT_CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
    debug_assert_eq!(a_blocks.nblocks(), a_r.nrows(), "a_blocks must match a_r");
    debug_assert_eq!(b_blocks.nblocks(), b_r.nrows(), "b_blocks must match b_r");
    debug_assert_eq!(a_r.ncols(), b_r.ncols(), "support dimensions must match");
    let nsta = if a_blocks.nblocks() > 0 {
        a_blocks.block(0).nrows()
    } else if b_blocks.nblocks() > 0 {
        b_blocks.block(0).nrows()
    } else {
        0
    };
    for index in 0..a_blocks.nblocks() {
        let block = a_blocks.block(index);
        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (nsta, nsta),
            "all blocks must be square nsta x nsta"
        );
    }
    for index in 0..b_blocks.nblocks() {
        let block = b_blocks.block(index);
        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (nsta, nsta),
            "all blocks must be square nsta x nsta"
        );
    }

    let a_is_zero = (0..a_blocks.nblocks()).all(|index| {
        a_blocks
            .block(index)
            .iter()
            .all(|value| value.re == 0.0 && value.im == 0.0)
    });
    let b_is_zero = (0..b_blocks.nblocks()).all(|index| {
        b_blocks
            .block(index)
            .iter()
            .all(|value| value.re == 0.0 && value.im == 0.0)
    });
    let skip_products = a_is_zero || b_is_zero;
    let one = Complex::new(1.0, 0.0);
    let reverse_scale = Complex::new(reverse_scale, 0.0);
    let n_a = a_r.nrows();
    let n_b = b_r.nrows();
    let pair_count = n_a
        .checked_mul(n_b)
        .ok_or_else(|| TbError::Other(format!("real-space {operation} pair count overflow")))?;

    let accumulate_pair = |accumulated: &mut RealSpaceBlockMap, pair_index: usize| -> Result<()> {
        let i_a = pair_index / n_b;
        let i_b = pair_index % n_b;
        let r_a = a_r.row(i_a);
        let r_b = b_r.row(i_b);
        let mut total = Vec::with_capacity(a_r.ncols());
        for (axis, (&left, &right)) in r_a.iter().zip(r_b.iter()).enumerate() {
            total.push(left.checked_add(right).ok_or_else(|| {
                TbError::Other(format!(
                    "real-space Minkowski sum overflow on axis {axis}: {left} + {right}"
                ))
            })?);
        }
        let comm = accumulated
            .entry(total)
            .or_insert_with(|| Array2::<Complex<f64>>::zeros((nsta, nsta)));
        if !skip_products {
            let a = a_blocks.block(i_a);
            let b = b_blocks.block(i_b);
            ndarray::linalg::general_mat_mul(one, &a, &b, one, comm);
            ndarray::linalg::general_mat_mul(reverse_scale, &b, &a, one, comm);
        }
        Ok(())
    };

    // Each Rayon worker accumulates into a private ordered map, so the hot
    // GEMM loop needs no locks.  The final reduction merges those maps before
    // the deterministic lexicographic output pass below.  Small and exact-zero
    // products stay serial to avoid scheduling and per-worker allocation cost.
    const PARALLEL_PAIR_THRESHOLD: usize = 128;
    let accumulated = if !skip_products
        && pair_count >= PARALLEL_PAIR_THRESHOLD
        && rayon::current_num_threads() > 1
    {
        // Rounding up can prevent Rayon's last binary split (e.g. 575²
        // pairs on eight threads), leaving half the workers without a job.
        let min_pairs_per_job = (pair_count / rayon::current_num_threads()).max(1);
        (0..pair_count)
            .into_par_iter()
            .with_min_len(min_pairs_per_job)
            .try_fold(
                RealSpaceBlockMap::new,
                |mut partial, pair_index| -> Result<RealSpaceBlockMap> {
                    accumulate_pair(&mut partial, pair_index)?;
                    Ok(partial)
                },
            )
            .try_reduce(RealSpaceBlockMap::new, |left, right| {
                Ok(merge_real_space_block_maps(left, right))
            })?
    } else {
        let mut accumulated = RealSpaceBlockMap::new();
        for pair_index in 0..pair_count {
            accumulate_pair(&mut accumulated, pair_index)?;
        }
        accumulated
    };

    let mut support_rows = Array2::<isize>::zeros((accumulated.len(), a_r.ncols()));
    let mut blocks = Vec::with_capacity(accumulated.len());
    for (i, (r, block)) in accumulated.into_iter().enumerate() {
        for (axis, value) in r.iter().enumerate() {
            support_rows[[i, axis]] = *value;
        }
        blocks.push(block);
    }
    Ok((blocks, support_rows))
}

pub(super) fn enforce_real_space_hermiticity(
    ham: &mut Array3<Complex<f64>>,
    ham_r: &Array2<isize>,
) -> Result<()> {
    let n_r = ham_r.nrows();
    let mut visited = vec![false; n_r];

    for i_r in 0..n_r {
        if visited[i_r] {
            continue;
        }
        let neg_r = ham_r.row(i_r).mapv(|x| -x);
        let Some(j_r) = find_R(ham_r, &neg_r) else {
            return Err(TbError::MissingHermitianConjugateHopping {
                r: ham_r.row(i_r).to_owned(),
            });
        };

        if i_r == j_r {
            let block = ham.index_axis(Axis(0), i_r).to_owned();
            let block_dag = hermitian_conjugate(&block);
            let herm = Array2::from_shape_fn(block.raw_dim(), |index| {
                Complex::new(
                    block[index].re.midpoint(block_dag[index].re),
                    block[index].im.midpoint(block_dag[index].im),
                )
            });
            ham.index_axis_mut(Axis(0), i_r).assign(&herm);
            visited[i_r] = true;
        } else {
            let block_i = ham.index_axis(Axis(0), i_r).to_owned();
            let block_j = ham.index_axis(Axis(0), j_r).to_owned();
            let block_j_dag = hermitian_conjugate(&block_j);
            let avg = Array2::from_shape_fn(block_i.raw_dim(), |index| {
                Complex::new(
                    block_i[index].re.midpoint(block_j_dag[index].re),
                    block_i[index].im.midpoint(block_j_dag[index].im),
                )
            });
            let avg_dag = hermitian_conjugate(&avg);
            ham.index_axis_mut(Axis(0), i_r).assign(&avg);
            ham.index_axis_mut(Axis(0), j_r).assign(&avg_dag);
            visited[i_r] = true;
            visited[j_r] = true;
        }
    }
    Ok(())
}

pub(super) fn hermitian_conjugate(a: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    a.t().mapv(|x| x.conj())
}
