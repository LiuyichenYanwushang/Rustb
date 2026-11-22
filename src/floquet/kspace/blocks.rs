//! Dense slot storage for the exact output support of the effective model.
//!
//! The effective Hamiltonian is a polynomial of degree order + 1 in the
//! harmonic hopping blocks, so every retained term reaches at most
//! degree * R_max on each axis. That bounding box is known before any
//! transform runs: the support is contained in it, and the real-space wrapper
//! consumes exactly the union of the input support, its pairwise Minkowski sum
//! (order >= 1) and its triple sum (order >= 2). Building the support from a
//! dense bitmap over that box avoids the O(|pairs| * |support|) tree
//! insertions of a Minkowski sum over a BTreeSet.
//!
//! The accumulator keeps one representative of every conjugate pair (-R is the
//! Hermitian conjugate of R) and writes the partner as its adjoint when the
//! model rows are produced, so only half the blocks are ever accumulated.

use super::C;
use crate::error::{Result, TbError};
use crate::floquet::real_space::hermitian_conjugate;
use ndarray::{Array2, Array3, ArrayView2, Axis, s};

/// Slot value for a translation that is not stored (outside the support or the
/// non-canonical half of a conjugate pair).
pub(super) const NO_SLOT: u32 = u32::MAX;

/// Bounding box of the exact output support, one extent per resolved axis.
#[derive(Clone, Copy, Debug)]
pub(super) struct SupportBox {
    min: [isize; 3],
    extent: [usize; 3],
}

impl SupportBox {
    /// Number of box positions, or None when the box does not fit a usize.
    pub(super) fn checked_volume(&self) -> Option<usize> {
        self.extent[0]
            .checked_mul(self.extent[1])?
            .checked_mul(self.extent[2])
    }

    /// Row-major index of a translation inside the box.
    pub(super) fn index(&self, vector: &[isize; 3]) -> Option<usize> {
        let mut index = 0;
        for axis in 0..3 {
            let shifted = vector[axis] - self.min[axis];
            if shifted < 0 || shifted as usize >= self.extent[axis] {
                return None;
            }
            index = index * self.extent[axis] + shifted as usize;
        }
        Some(index)
    }

    /// Inverse of index for an in-range position.
    fn vector(&self, mut index: usize) -> [isize; 3] {
        let mut vector = [0_isize; 3];
        for axis in (0..3).rev() {
            let length = self.extent[axis];
            vector[axis] = self.min[axis] + (index % length) as isize;
            index /= length;
        }
        vector
    }
}

/// One representative of a conjugate pair: the vector whose first nonzero
/// component is positive (the origin is its own representative).
fn canonical(vector: &[isize; 3]) -> bool {
    vector
        .iter()
        .find(|component| **component != 0)
        .map(|component| *component > 0)
        .unwrap_or(true)
}

/// Exact output support, in lexicographic order.
///
/// The result is the union of the input support, its pairwise Minkowski sum
/// when harmonic_max > 0 and order >= 1, and its triple sum when order >= 2,
/// matching the zero-valued layers the real-space wrapper inserts.
pub(super) fn effective_support(
    support: &Array2<isize>,
    dim: usize,
    harmonic_max: isize,
    order: usize,
) -> Result<(SupportBox, Vec<[isize; 3]>)> {
    if !(1..=3).contains(&dim) || support.ncols() != dim {
        return Err(TbError::Other(format!(
            "the effective support needs 1..=3 translation components, got {} for dim {dim}",
            support.ncols()
        )));
    }
    let degree = if harmonic_max > 0 { order + 1 } else { 1 };
    let mut extent = [1_usize; 3];
    let mut min = [0_isize; 3];
    for axis in 0..dim {
        let radius = support
            .column(axis)
            .iter()
            .map(|r| r.unsigned_abs())
            .max()
            .unwrap_or(0);
        let span = radius
            .checked_mul(2 * degree)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| TbError::Other("the effective support extent overflows usize".into()))?;
        let half = isize::try_from(radius)
            .ok()
            .and_then(|radius| radius.checked_mul(degree as isize))
            .ok_or_else(|| TbError::Other("the effective support radius overflows isize".into()))?;
        extent[axis] = span;
        min[axis] = -half;
    }
    let support_box = SupportBox { min, extent };
    let volume = support_box
        .checked_volume()
        .filter(|volume| i64::try_from(*volume).is_ok())
        .ok_or_else(|| {
            TbError::Other("the effective support box does not fit the index type".into())
        })?;
    let stride = [
        extent[1]
            .checked_mul(extent[2])
            .ok_or_else(|| TbError::Other("the effective support stride overflows usize".into()))?,
        extent[2],
        1,
    ];
    // linear(a + b) = linear(a) + linear(b) + shift, with
    // linear(r) = sum_axis (r - min) . stride.
    let shift: i64 = (0..3)
        .map(|axis| min[axis] as i64 * stride[axis] as i64)
        .sum();
    let linear = |vector: &[isize; 3]| -> i64 {
        (0..3)
            .map(|axis| vector[axis] as i64 * stride[axis] as i64)
            .sum::<i64>()
            - shift
    };
    let mut marked = vec![0_u8; volume];
    let mut input = Vec::with_capacity(support.nrows());
    for row in support.outer_iter() {
        let mut vector = [0_isize; 3];
        for axis in 0..dim {
            vector[axis] = row[axis];
        }
        let index = linear(&vector);
        debug_assert!((0..volume as i64).contains(&index));
        marked[index as usize] = 1;
        input.push(index);
    }
    if harmonic_max > 0 && order >= 1 {
        let mut pairs = Vec::new();
        for &left in &input {
            for &right in &input {
                let index = left + right + shift;
                debug_assert!((0..volume as i64).contains(&index));
                if marked[index as usize] == 0 {
                    marked[index as usize] = 1;
                    pairs.push(index);
                }
            }
        }
        if order >= 2 {
            for &pair in &pairs {
                for &right in &input {
                    let index = pair + right + shift;
                    debug_assert!((0..volume as i64).contains(&index));
                    marked[index as usize] = 1;
                }
            }
        }
    }
    let mut vectors = Vec::new();
    for index in 0..volume {
        if marked[index] != 0 {
            vectors.push(support_box.vector(index));
        }
    }
    Ok((support_box, vectors))
}

/// Output accumulator with a dense slot table over the support box.
///
/// The rows are the model rows themselves: origin first, then the support in
/// lexicographic order. Only the canonical representative of each conjugate
/// pair is accumulated; `DenseBlocks::finish` writes the partner row as its
/// exact adjoint, so the accumulator never holds more than one full copy of the
/// output and never allocates a second one.
pub(super) struct DenseBlocks {
    support_box: SupportBox,
    slots: Vec<u32>,
    ham: Array3<C>,
    ham_r: Array2<isize>,
    support: Vec<[isize; 3]>,
    has_origin: bool,
    nsta: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    stored: usize,
}

impl DenseBlocks {
    pub(super) fn new(
        support_box: SupportBox,
        support: Vec<[isize; 3]>,
        nsta: usize,
        dim: usize,
    ) -> Result<Self> {
        debug_assert!(support.windows(2).all(|pair| pair[0] < pair[1]));
        let origin = [0_isize; 3];
        let has_origin = support.binary_search(&origin).is_ok();
        let n_rows = support.len() + usize::from(!has_origin);
        let volume = support_box.checked_volume().ok_or_else(|| {
            TbError::Other("the effective support box does not fit a usize".into())
        })?;
        let mut slots = vec![NO_SLOT; volume];
        let mut ham_r = Array2::<isize>::zeros((n_rows, dim));
        let mut stored = 0_usize;
        // Row zero is the origin: the real one when the support contains it,
        // otherwise the zero block the model needs there.
        let mut row = 1_usize;
        for vector in &support {
            let assigned = if *vector == origin {
                0
            } else {
                let assigned = row;
                row += 1;
                assigned
            };
            for axis in 0..dim {
                ham_r[[assigned, axis]] = vector[axis];
            }
            if canonical(vector) {
                let index = support_box.index(vector).ok_or_else(|| {
                    TbError::Other("the effective support vector left its own box".into())
                })?;
                if slots[index] != NO_SLOT {
                    return Err(TbError::Other(
                        "the effective support contains a duplicate translation".into(),
                    ));
                }
                if assigned >= NO_SLOT as usize {
                    return Err(TbError::Other(
                        "the effective support has too many blocks".into(),
                    ));
                }
                slots[index] = assigned as u32;
                stored += 1;
            }
        }
        debug_assert_eq!(row, n_rows);
        // The input support is validated as negation-closed at the entry point
        // and every Minkowski layer preserves that, so this cannot fire through
        // the public API; keep the symmetric check so a hand-built support still
        // fails here instead of producing a non-Hermitian model.
        for vector in &support {
            let partner = [-vector[0], -vector[1], -vector[2]];
            if support.binary_search(&partner).is_err() {
                return Err(TbError::Other(
                    "the effective support is not closed under negation".into(),
                ));
            }
        }
        Ok(Self {
            support_box,
            slots,
            ham: Array3::zeros((n_rows, nsta, nsta)),
            ham_r,
            support,
            has_origin,
            nsta,
            stored,
        })
    }

    /// Number of accumulated blocks (one per conjugate pair).
    #[cfg(test)]
    pub(super) fn stored_blocks(&self) -> usize {
        self.stored
    }

    /// Model row of a translation, or NO_SLOT when it is not accumulated here.
    pub(super) fn slot(&self, vector: &[isize; 3]) -> u32 {
        match self.support_box.index(vector) {
            Some(index) => self.slots[index],
            None => NO_SLOT,
        }
    }

    /// Accumulate one k-slice contribution into a model row.
    pub(super) fn add_slot(&mut self, slot: u32, phase: C, block: ArrayView2<'_, C>) {
        debug_assert_ne!(slot, NO_SLOT);
        let mut target = self.ham.index_axis_mut(Axis(0), slot as usize);
        target.scaled_add(phase, &block);
    }

    /// Write the conjugate half and return the model rows.
    ///
    /// The accumulated rows are already exactly Hermitian as a pair: the stored
    /// representative is written as-is and its partner as its adjoint. The only
    /// self-paired row is the origin, which is symmetrised here, so the caller
    /// does not need the full real-space Hermiticity pass.
    pub(super) fn finish(mut self) -> Result<(Array3<C>, Array2<isize>)> {
        let origin = [0_isize; 3];
        let mut row = 1_usize;
        for vector in &self.support {
            let assigned = if *vector == origin {
                0
            } else {
                let assigned = row;
                row += 1;
                assigned
            };
            if canonical(vector) {
                continue;
            }
            let partner = [-vector[0], -vector[1], -vector[2]];
            let index = self.support_box.index(&partner).ok_or_else(|| {
                TbError::Other("the effective support lost a conjugate block".into())
            })?;
            let partner_row = self.slots[index];
            if partner_row == NO_SLOT {
                return Err(TbError::Other(
                    "the effective support lost a conjugate block slot".into(),
                ));
            }
            let partner_row = partner_row as usize;
            // Both rows belong to the same array, so the disjoint borrows come
            // from multi_slice_mut instead of two index_axis calls.
            let (source, mut target) = self
                .ham
                .multi_slice_mut((s![partner_row, .., ..], s![assigned, .., ..]));
            for i in 0..self.nsta {
                for j in 0..self.nsta {
                    target[[i, j]] = source[[j, i]].conj();
                }
            }
        }
        debug_assert_eq!(row, self.ham.shape()[0]);
        if self.has_origin {
            let block = self.ham.index_axis(Axis(0), 0).to_owned();
            let block_dag = hermitian_conjugate(&block);
            let mut target = self.ham.index_axis_mut(Axis(0), 0);
            for i in 0..self.nsta {
                for j in 0..self.nsta {
                    target[[i, j]] = C::new(
                        block[[i, j]].re.midpoint(block_dag[[i, j]].re),
                        block[[i, j]].im.midpoint(block_dag[[i, j]].im),
                    );
                }
            }
        }
        let Self { ham, ham_r, .. } = self;
        Ok((ham, ham_r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::BTreeSet;

    fn brute_force_support(
        support: &Array2<isize>,
        dim: usize,
        harmonic_max: isize,
        order: usize,
    ) -> Vec<[isize; 3]> {
        let mut vectors = BTreeSet::new();
        for row in support.outer_iter() {
            let mut vector = [0_isize; 3];
            for axis in 0..dim {
                vector[axis] = row[axis];
            }
            vectors.insert(vector);
        }
        if harmonic_max > 0 && order >= 1 {
            let input = vectors.clone();
            let mut pairs = BTreeSet::new();
            for left in &input {
                for right in &input {
                    pairs.insert([left[0] + right[0], left[1] + right[1], left[2] + right[2]]);
                }
            }
            vectors.extend(pairs.iter().copied());
            if order >= 2 {
                for pair in &pairs {
                    for right in &input {
                        vectors.insert([
                            pair[0] + right[0],
                            pair[1] + right[1],
                            pair[2] + right[2],
                        ]);
                    }
                }
            }
        }
        vectors.into_iter().collect()
    }

    #[test]
    fn support_matches_the_brute_force_minkowski_sum() {
        let support = array![
            [0_isize, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [2, 1],
            [-2, -1],
        ];
        for harmonic_max in [0_isize, 1, 2] {
            for order in [0_usize, 1, 2] {
                let (support_box, vectors) =
                    effective_support(&support, 2, harmonic_max, order).unwrap();
                assert_eq!(
                    vectors,
                    brute_force_support(&support, 2, harmonic_max, order),
                    "harmonic_max {harmonic_max} order {order}"
                );
                for vector in &vectors {
                    assert!(support_box.index(vector).is_some());
                }
                let degree = if harmonic_max > 0 { order + 1 } else { 1 };
                assert_eq!(
                    support_box.checked_volume().unwrap(),
                    (2 * degree * 2 + 1) * (2 * degree + 1)
                );
            }
        }
    }

    #[test]
    fn box_index_round_trips_and_rejects_outside_vectors() {
        let support = array![[0_isize, 0, 0], [2, -1, 1], [-2, 1, -1]];
        let (support_box, _) = effective_support(&support, 3, 1, 1).unwrap();
        for index in 0..support_box.checked_volume().unwrap() {
            let vector = support_box.vector(index);
            assert_eq!(support_box.index(&vector), Some(index));
        }
        assert_eq!(support_box.index(&[100, 0, 0]), None);
        assert_eq!(support_box.index(&[0, 0, -100]), None);
    }

    #[test]
    fn dense_blocks_reconstruct_the_conjugate_half() {
        let support = array![[0_isize, 0], [1, 0], [-1, 0], [0, 1], [0, -1]];
        let (support_box, vectors) = effective_support(&support, 2, 1, 1).unwrap();
        let nsta = 2;
        let mut blocks = DenseBlocks::new(support_box, vectors.clone(), nsta, 2).unwrap();
        // Give every conjugate pair an arbitrary block through the stored half.
        let mut state = 0x1234_5678_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) - 0.5
        };
        for vector in &vectors {
            if !canonical(vector) {
                continue;
            }
            let slot = blocks.slot(vector);
            assert_ne!(slot, NO_SLOT);
            let phase = C::new(next(), next());
            let value = Array2::from_shape_fn((nsta, nsta), |_| C::new(next(), next()) * phase);
            // The origin is its own conjugate partner, so the engine keeps the
            // stored value as-is and it must already be Hermitian.
            let value = if vector.iter().all(|component| *component == 0) {
                Array2::from_shape_fn((nsta, nsta), |(i, j)| {
                    (value[[i, j]] + value[[j, i]].conj()) * 0.5
                })
            } else {
                value
            };
            blocks.add_slot(slot, C::new(1.0, 0.0), value.view());
        }
        let (ham, ham_r) = blocks.finish().unwrap();
        let find = |vector: &[isize; 3]| {
            (0..ham_r.nrows())
                .find(|row| (0..2).all(|axis| ham_r[[*row, axis]] == vector[axis]))
                .expect("every support vector must have a row")
        };
        assert_eq!(&ham_r.row(0).to_vec(), &vec![0, 0]);
        for vector in &vectors {
            let negative = [-vector[0], -vector[1], -vector[2]];
            let left = ham.index_axis(Axis(0), find(vector));
            let right = ham.index_axis(Axis(0), find(&negative));
            for i in 0..nsta {
                for j in 0..nsta {
                    assert_eq!(
                        left[[i, j]],
                        right[[j, i]].conj(),
                        "vector {vector:?} negative {negative:?} i {i} j {j}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_support_leaves_empty_slots() {
        // empty and the slot table must keep NO_SLOT inside the box.
        // empty and the slot table must keep NO_SLOT inside the box.
        let support = array![[0_isize, 0], [1, 1], [-1, -1]];
        let (support_box, vectors) = effective_support(&support, 2, 2, 2).unwrap();
        let blocks = DenseBlocks::new(support_box, vectors.clone(), 2, 2).unwrap();
        let volume = support_box.checked_volume().unwrap();
        assert!(vectors.len() < volume);
        let mut filled = 0;
        for x in -3..=3 {
            for y in -3..=3 {
                if blocks.slot(&[x, y, 0]) != NO_SLOT {
                    filled += 1;
                }
            }
        }
        assert_eq!(filled, vectors.iter().filter(|v| canonical(v)).count());
        assert_eq!(
            blocks.slot(&[0, 1, 0]),
            NO_SLOT,
            "an interior hole must stay empty"
        );
        assert_ne!(blocks.slot(&[1, 1, 0]), NO_SLOT);
    }

    #[test]
    fn one_and_three_dimensional_supports_match_brute_force() {
        let one = array![[0_isize], [1], [-1], [2], [-2]];
        for harmonic_max in [0_isize, 1, 2] {
            for order in [0_usize, 1, 2] {
                let (_, vectors) = effective_support(&one, 1, harmonic_max, order).unwrap();
                assert_eq!(vectors, brute_force_support(&one, 1, harmonic_max, order));
            }
        }
        let three = array![
            [0_isize, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ];
        for harmonic_max in [0_isize, 1] {
            for order in [0_usize, 1, 2] {
                let (support_box, vectors) =
                    effective_support(&three, 3, harmonic_max, order).unwrap();
                assert_eq!(vectors, brute_force_support(&three, 3, harmonic_max, order));
                for vector in &vectors {
                    assert!(support_box.index(vector).is_some());
                }
            }
        }
    }

    #[test]
    fn finish_orders_rows_and_handles_degenerate_supports() {
        // An empty support still produces the zero origin row the real-space
        // wrapper appends.
        let empty = Array2::<isize>::zeros((0, 2));
        let (support_box, vectors) = effective_support(&empty, 2, 0, 0).unwrap();
        assert!(vectors.is_empty());
        let accumulator = DenseBlocks::new(support_box, vectors, 2, 2).unwrap();
        let (ham, ham_r) = accumulator.finish().unwrap();
        assert_eq!(ham_r.dim(), (1, 2));
        assert_eq!(&ham_r.row(0).to_vec(), &vec![0, 0]);
        assert!(ham.iter().all(|value| value.re == 0.0 && value.im == 0.0));

        // A support without the origin keeps the zero origin row first and
        // leaves the remaining rows lexicographic.
        let unpaired = array![[1_isize, 0], [-1, 0]];
        let (support_box, vectors) = effective_support(&unpaired, 2, 0, 0).unwrap();
        assert_eq!(vectors, vec![[-1, 0, 0], [1, 0, 0]]);
        let mut accumulator = DenseBlocks::new(support_box, vectors.clone(), 2, 2).unwrap();
        let slot = accumulator.slot(&[1, 0, 0]);
        accumulator.add_slot(slot, C::new(1.0, 0.0), Array2::<C>::zeros((2, 2)).view());
        let (ham, ham_r) = accumulator.finish().unwrap();
        assert_eq!(ham_r.nrows(), 3);
        assert_eq!(&ham_r.row(0).to_vec(), &vec![0, 0]);
        assert_eq!(&ham_r.row(1).to_vec(), &vec![-1, 0]);
        assert_eq!(&ham_r.row(2).to_vec(), &vec![1, 0]);
        // The conjugate row is the adjoint of the accumulated zero block, so
        // both stay zero and the appended origin row stays zero.
        assert!(ham.iter().all(|value| value.re == 0.0 && value.im == 0.0));
        // Also pin the adjoint relation with a nonzero block.
        let mut accumulator = DenseBlocks::new(
            effective_support(&unpaired, 2, 0, 0).unwrap().0,
            vectors,
            2,
            2,
        )
        .unwrap();
        let block = Array2::from_shape_fn((2, 2), |(i, j)| C::new(1.0 + i as f64, -0.5 - j as f64));
        let slot = accumulator.slot(&[1, 0, 0]);
        accumulator.add_slot(slot, C::new(1.0, 0.0), block.view());
        let (ham, _) = accumulator.finish().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(ham[[1, i, j]], ham[[2, j, i]].conj());
            }
        }

        // An origin-only support is a single row.
        let origin_only = array![[0_isize, 0]];
        let (support_box, vectors) = effective_support(&origin_only, 2, 1, 1).unwrap();
        let accumulator = DenseBlocks::new(support_box, vectors, 2, 2).unwrap();
        let (_, ham_r) = accumulator.finish().unwrap();
        assert_eq!(ham_r.dim(), (1, 2));
        assert_eq!(&ham_r.row(0).to_vec(), &vec![0, 0]);
    }

    #[test]
    fn finish_symmetrises_an_asymmetric_origin_block() {
        let support = array![[0_isize, 0]];
        let (support_box, vectors) = effective_support(&support, 2, 0, 0).unwrap();
        let mut accumulator = DenseBlocks::new(support_box, vectors, 2, 2).unwrap();
        let slot = accumulator.slot(&[0, 0, 0]);
        let block = Array2::from_shape_fn((2, 2), |(i, j)| C::new(1.0 + i as f64, -0.5 + j as f64));
        accumulator.add_slot(slot, C::new(1.0, 0.0), block.view());
        let (ham, _) = accumulator.finish().unwrap();
        let origin = ham.index_axis(Axis(0), 0);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    origin[[i, j]].re,
                    block[[i, j]].re.midpoint(block[[j, i]].re),
                    "origin real part ({i},{j}) was not symmetrised"
                );
                assert_eq!(
                    origin[[i, j]].im,
                    block[[i, j]].im.midpoint(-block[[j, i]].im),
                    "origin imaginary part ({i},{j}) was not symmetrised"
                );
                assert_eq!(origin[[i, j]], origin[[j, i]].conj());
            }
        }
    }

    #[test]
    fn finished_rows_need_no_real_space_repair() {
        // The store writes the conjugate half as the exact adjoint and
        // symmetrises the origin, so the real-space repair pass must be a
        // bitwise no-op on the finished rows.
        let support = array![[0_isize, 0], [1, 0], [-1, 0], [0, 1], [0, -1]];
        let (support_box, vectors) = effective_support(&support, 2, 1, 1).unwrap();
        let mut accumulator = DenseBlocks::new(support_box, vectors.clone(), 2, 2).unwrap();
        let mut state = 0x9e37_79b9_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) - 0.5
        };
        for vector in &vectors {
            let slot = accumulator.slot(vector);
            if slot == NO_SLOT {
                continue;
            }
            let block = Array2::from_shape_fn((2, 2), |_| C::new(next(), next()));
            accumulator.add_slot(slot, C::new(next(), next()), block.view());
        }
        let (ham, ham_r) = accumulator.finish().unwrap();
        let mut repaired = ham.clone();
        crate::floquet::real_space::enforce_real_space_hermiticity(&mut repaired, &ham_r).unwrap();
        for (index, value) in ham.iter().enumerate() {
            let repaired_value = repaired[[index / 4, (index / 2) % 2, index % 2]];
            assert_eq!(
                value.re.to_bits(),
                repaired_value.re.to_bits(),
                "repair changed real part {index}"
            );
            assert_eq!(
                value.im.to_bits(),
                repaired_value.im.to_bits(),
                "repair changed imaginary part {index}"
            );
        }
    }

    #[test]
    fn non_closed_supports_are_rejected() {
        let closed = array![[0_isize, 0], [1, 0], [-1, 0]];
        let (support_box, _) = effective_support(&closed, 2, 0, 0).unwrap();
        let missing_partner = vec![[0, 0, 0], [1, 0, 0]];
        assert!(DenseBlocks::new(support_box, missing_partner, 2, 2).is_err());
    }
}
