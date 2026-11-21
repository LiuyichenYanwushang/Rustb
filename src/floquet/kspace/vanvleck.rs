//! Guarded second-order van Vleck assembly at individual k points.

use super::C;
use crate::error::{Result, TbError};
use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut2, ArrayViewMut3, Axis};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

const NONFINITE_SCALE: &str = "a van Vleck frequency scaling factor is non-finite; \
increase omega0_ev or lower the requested order";

/// Assemble the effective Hamiltonian at every k point of one slice.
///
/// Layout: harmonics is (n_harm, points, nsta, nsta) with slot n + harmonic_max;
/// out is (points, nsta, nsta) and receives
///
///   H_eff = H_0
///         + sum_{n=1..h} [H_-n, H_n] / (n W)                       (order >= 1)
///         + sum_{m != 0, |m| <= h}
///             [H_-m, [H_0, H_m]] / (2 m^2 W^2)                      (order >= 2)
///         + sum_{m, m' with m' != 0, m' != m, m' != 2m,
///                         |m|, |m'|, |m'-m| <= h}
///             [H_-m', [H_(m'-m), H_m]] / (3 m m' W^2)               (order >= 2)
///
/// The correction sums are skipped when order is 0 or the drive has no time
/// dependence. The window and the m' == 2m skip must match the real-space path
/// exactly: the double-commutator form equals the truncated Loewdin triple sum
/// only under that window, and m' == 2m makes the inner commutator vanish.
///
/// With reuse_adjoint = true the two order-two families are evaluated on one
/// representative of each conjugate pair and their adjoints are added, which
/// halves their matrix products. Real Peierls harmonics satisfy
/// H_0 = H_0^dagger and H_-n = H_n^dagger up to coefficient round-off, so the
/// missing half is the algebraic adjoint of the computed half; the same holds
/// for the mixed family because T(m, m')^dagger = T(-m, -m') and the weight
/// 1/(3 m m') is invariant under (m, m') -> (-m, -m'). Pass false for
/// unrestricted (not harmonic consistent) blocks, which only the test oracle
/// does.
///
/// Every point is assembled with one fixed accumulation order, so a point value
/// never depends on the worker count. Points are independent and are the only
/// parallel axis, so each point keeps its full serial sum. Per-point scratch
/// buffers are allocated once per worker chunk, so the hot loop performs no
/// heap allocation. The reference Netlib CBLAS is not reentrant, which is why
/// those builds stay serial.
pub(super) fn assemble_pointwise(
    harmonics: ArrayView4<'_, C>,
    harmonic_max: isize,
    order: usize,
    has_corrections: bool,
    reuse_adjoint: bool,
    omega: f64,
    mut out: ArrayViewMut3<'_, C>,
) -> Result<()> {
    if order > 2 {
        return Err(TbError::Other(format!(
            "the k-space van Vleck assembly supports order 0, 1 or 2, got {order}"
        )));
    }
    if harmonic_max < 0 {
        return Err(TbError::Other(format!(
            "the k-space van Vleck assembly needs a non-negative harmonic cutoff, got {harmonic_max}"
        )));
    }
    let out_shape = out.shape();
    let points = out_shape[0];
    let nsta = out_shape[1];
    if out_shape[2] != nsta {
        return Err(TbError::Other(format!(
            "k-space van Vleck output blocks must be square nsta x nsta, got {nsta} x {}",
            out_shape[2]
        )));
    }
    let harmonic_shape = harmonics.shape();
    if harmonic_shape[1] != points || harmonic_shape[2] != nsta || harmonic_shape[3] != nsta {
        return Err(TbError::Other(format!(
            "k-space harmonic slice is ({}, {}, {}) but the output is ({points}, {nsta}, {nsta})",
            harmonic_shape[1], harmonic_shape[2], harmonic_shape[3]
        )));
    }
    let effective_order = if has_corrections { order } else { 0 };
    let slots_needed = required_slots(harmonic_max, effective_order)?;
    if harmonic_shape[0] < slots_needed {
        return Err(TbError::Other(format!(
            "k-space harmonic slice has {} slots but {slots_needed} are required for \
harmonic_max = {harmonic_max} and order {effective_order}",
            harmonic_shape[0]
        )));
    }
    let mut inverse_omega = 0.0;
    let mut inverse_omega_squared = 0.0;
    if effective_order >= 1 {
        if !omega.is_finite() || omega == 0.0 {
            return Err(TbError::Other(format!(
                "the k-space van Vleck assembly needs a finite nonzero omega, got {omega}"
            )));
        }
        inverse_omega = omega.recip();
        if !inverse_omega.is_finite() {
            return Err(TbError::Other(NONFINITE_SCALE.to_string()));
        }
        if effective_order >= 2 {
            inverse_omega_squared = inverse_omega * inverse_omega;
            if !inverse_omega_squared.is_finite() {
                return Err(TbError::Other(NONFINITE_SCALE.to_string()));
            }
        }
    }

    let mut rows: Vec<ArrayViewMut2<'_, C>> = out.outer_iter_mut().collect();
    // Reference Netlib CBLAS may use process globals; keep the fan-out serial
    // there, matching the real-space convolution policy. Chunking gives every
    // worker one scratch set for many points instead of one allocation per
    // matrix product.
    if !cfg!(any(feature = "netlib-static", feature = "netlib-system"))
        && rayon::current_num_threads() > 1
    {
        let workers = rayon::current_num_threads();
        let chunk_len = rows.len().div_ceil(workers * 4).max(1);
        rows.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(chunk_index, chunk)| {
                let mut scratch = Scratch::new(nsta);
                for (offset, target) in chunk.iter_mut().enumerate() {
                    let point = chunk_index * chunk_len + offset;
                    assemble_point_into(
                        harmonics.index_axis(Axis(1), point),
                        harmonic_max,
                        effective_order,
                        inverse_omega,
                        inverse_omega_squared,
                        reuse_adjoint,
                        target,
                        &mut scratch,
                    );
                }
            });
    } else {
        let mut scratch = Scratch::new(nsta);
        for (point, target) in rows.iter_mut().enumerate() {
            assemble_point_into(
                harmonics.index_axis(Axis(1), point),
                harmonic_max,
                effective_order,
                inverse_omega,
                inverse_omega_squared,
                reuse_adjoint,
                target,
                &mut scratch,
            );
        }
    }
    Ok(())
}

/// Reusable per-worker buffers for one k point.
///
/// total holds the running effective block; inner and outer hold the nested
/// commutators. Keeping them out of the loop removes three nsta^2 allocations
/// per commutator.
struct Scratch {
    total: Array2<C>,
    inner: Array2<C>,
    outer: Array2<C>,
}

impl Scratch {
    fn new(nsta: usize) -> Self {
        Self {
            total: Array2::zeros((nsta, nsta)),
            inner: Array2::zeros((nsta, nsta)),
            outer: Array2::zeros((nsta, nsta)),
        }
    }
}

/// Number of harmonic slots the requested effective order reads.
fn required_slots(harmonic_max: isize, effective_order: usize) -> Result<usize> {
    let span = if effective_order >= 1 {
        harmonic_max
            .checked_mul(2)
            .and_then(|span| span.checked_add(1))
    } else {
        harmonic_max.checked_add(1)
    };
    let span = span.ok_or_else(|| {
        TbError::Other(format!(
            "the k-space van Vleck harmonic cutoff {harmonic_max} is too large to index"
        ))
    })?;
    usize::try_from(span).map_err(|_| {
        TbError::Other(format!(
            "the k-space van Vleck harmonic cutoff {harmonic_max} does not fit a slice index"
        ))
    })
}

/// Harmonic slot of order n: the slice stores n + harmonic_max.
#[inline]
fn harmonic_slot(n: isize, harmonic_max: isize) -> usize {
    (n + harmonic_max) as usize
}

/// out = left * right - right * left, into a caller-owned buffer.
///
/// Two BLAS products with the second accumulating into the first: no heap
/// allocation and no intermediate subtraction buffer.
#[inline]
fn commutator_into(left: ArrayView2<'_, C>, right: ArrayView2<'_, C>, out: &mut Array2<C>) {
    general_mat_mul(C::new(1.0, 0.0), &left, &right, C::new(0.0, 0.0), out);
    general_mat_mul(C::new(-1.0, 0.0), &right, &left, C::new(1.0, 0.0), out);
}

/// total += scale * (matrix + matrix^dagger), in one pass.
///
/// The adjoint is formed by reading the transposed element, so this adds the
/// conjugate partner of a double-commutator term without materializing it.
fn add_hermitian_scaled(total: &mut Array2<C>, matrix: &Array2<C>, scale: f64) {
    let scale = C::new(scale, 0.0);
    let nsta = matrix.nrows();
    for row in 0..nsta {
        for column in 0..nsta {
            total[[row, column]] += scale * (matrix[[row, column]] + matrix[[column, row]].conj());
        }
    }
}

/// A-family orders to evaluate: one representative per conjugate pair when
/// adjoints are reused, otherwise every signed order.
fn a_family_orders(harmonic_max: isize, reuse_adjoint: bool) -> impl Iterator<Item = isize> {
    let first = if reuse_adjoint { 1 } else { -harmonic_max };
    (first..=harmonic_max).filter(move |m| *m != 0)
}

/// Mixed-family (m, m') pairs to evaluate, in ascending order.
///
/// With reuse_adjoint the first index runs over m > 0 only; the partner of
/// (m, m') is (-m, -m'), whose first index is negative, so every conjugate
/// pair is covered exactly once. Without it both signs are summed explicitly.
fn mixed_family_pairs(
    harmonic_max: isize,
    reuse_adjoint: bool,
) -> impl Iterator<Item = (isize, isize)> {
    let first = if reuse_adjoint { 1 } else { -harmonic_max };
    (first..=harmonic_max)
        .filter(move |m| *m != 0)
        .flat_map(move |m| {
            (-harmonic_max..=harmonic_max)
                .filter(move |m_prime| {
                    *m_prime != 0
                        && *m_prime != m
                        && *m_prime != 2 * m
                        && (*m_prime - m).abs() <= harmonic_max
                })
                .map(move |m_prime| (m, m_prime))
        })
}

/// Effective Hamiltonian of one k point, written into its output block.
///
/// The terms are added exactly in the order documented on assemble_pointwise.
/// With reuse_adjoint only the m > 0 representatives of both order-two families
/// are formed, and each is added together with its adjoint.
fn assemble_point_into(
    point_harmonics: ArrayView3<'_, C>,
    harmonic_max: isize,
    order: usize,
    inverse_omega: f64,
    inverse_omega_squared: f64,
    reuse_adjoint: bool,
    target: &mut ArrayViewMut2<'_, C>,
    scratch: &mut Scratch,
) {
    let harmonic =
        |n: isize| point_harmonics.index_axis_move(Axis(0), harmonic_slot(n, harmonic_max));
    scratch.total.assign(&harmonic(0));

    if order >= 1 {
        // [H_-n, H_n] is already Hermitian, so no pairing is needed here.
        for n in 1..=harmonic_max {
            commutator_into(harmonic(-n), harmonic(n), &mut scratch.inner);
            scratch
                .total
                .scaled_add(C::new(inverse_omega / (n as f64), 0.0), &scratch.inner);
        }
    }

    if order >= 2 {
        // A family: T_m = [H_-m, [H_0, H_m]]. With reuse_adjoint only the
        // m > 0 representative of each conjugate pair is formed and added
        // together with its adjoint; T_m^dagger = T_-m carries the same
        // weight 1/(2 m^2 W^2).
        for m in a_family_orders(harmonic_max, reuse_adjoint) {
            commutator_into(harmonic(0), harmonic(m), &mut scratch.inner);
            commutator_into(harmonic(-m), scratch.inner.view(), &mut scratch.outer);
            let scale = inverse_omega_squared / (2.0 * (m as f64).powi(2));
            if reuse_adjoint {
                add_hermitian_scaled(&mut scratch.total, &scratch.outer, scale);
            } else {
                scratch.total.scaled_add(C::new(scale, 0.0), &scratch.outer);
            }
        }
        // B family: T(m, m') = [H_-m', [H_(m'-m), H_m]]. The double sum only
        // equals the truncated Loewdin triple sum under the shared harmonic
        // window; m' == 2m is skipped because its inner commutator is exactly
        // zero. With reuse_adjoint only the m > 0 representative is formed and
        // added with its adjoint; T(m,m')^dagger = T(-m,-m') carries the same
        // weight 1/(3 m m').
        for (m, m_prime) in mixed_family_pairs(harmonic_max, reuse_adjoint) {
            commutator_into(harmonic(m_prime - m), harmonic(m), &mut scratch.inner);
            commutator_into(harmonic(-m_prime), scratch.inner.view(), &mut scratch.outer);
            let scale = inverse_omega_squared / (3.0 * (m as f64) * (m_prime as f64));
            if reuse_adjoint {
                add_hermitian_scaled(&mut scratch.total, &scratch.outer, scale);
            } else {
                scratch.total.scaled_add(C::new(scale, 0.0), &scratch.outer);
            }
        }
    }

    target.assign(&scratch.total);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4, s};

    /// Deterministic xorshift generator, so the tests need no extra dependency.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed ^ 0x9e37_79b9_7f4a_7c15)
        }

        fn next_u64(&mut self) -> u64 {
            let mut state = self.0;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            self.0 = state;
            state
        }

        /// Uniform on [-0.5, 0.5).
        fn next_signed(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64) - 0.5
        }

        fn next_complex(&mut self) -> C {
            C::new(self.next_signed(), self.next_signed())
        }
    }

    fn random_matrix(rng: &mut Lcg, nsta: usize) -> Array2<C> {
        Array2::from_shape_fn((nsta, nsta), |_| rng.next_complex())
    }

    fn dagger(matrix: &Array2<C>) -> Array2<C> {
        matrix.t().mapv(|value| value.conj())
    }

    /// Unconstrained random harmonic blocks; nothing guarantees H_-n = H_n^dagger.
    fn random_harmonics(points: usize, nsta: usize, harmonic_max: isize, seed: u64) -> Array4<C> {
        let slots = (2 * harmonic_max + 1) as usize;
        let mut harmonics = Array4::zeros((slots, points, nsta, nsta));
        let mut rng = Lcg::new(seed);
        for point in 0..points {
            for slot in 0..slots {
                let block = random_matrix(&mut rng, nsta);
                harmonics.slice_mut(s![slot, point, .., ..]).assign(&block);
            }
        }
        harmonics
    }

    /// Random harmonics with H_0 = H_0^dagger and H_-n = H_n^dagger, the
    /// relations implied by real Peierls coefficients and Hermitian hoppings.
    fn hermitian_consistent_harmonics(
        points: usize,
        nsta: usize,
        harmonic_max: isize,
        seed: u64,
    ) -> Array4<C> {
        let slots = (2 * harmonic_max + 1) as usize;
        let mut harmonics = Array4::zeros((slots, points, nsta, nsta));
        let mut rng = Lcg::new(seed);
        let zero_slot = harmonic_slot(0, harmonic_max);
        for point in 0..points {
            let raw = random_matrix(&mut rng, nsta);
            let h_zero = &raw + &dagger(&raw);
            harmonics
                .slice_mut(s![zero_slot, point, .., ..])
                .assign(&h_zero);
            for n in 1..=harmonic_max {
                let positive = random_matrix(&mut rng, nsta);
                let negative = dagger(&positive);
                harmonics
                    .slice_mut(s![harmonic_slot(n, harmonic_max), point, .., ..])
                    .assign(&positive);
                harmonics
                    .slice_mut(s![harmonic_slot(-n, harmonic_max), point, .., ..])
                    .assign(&negative);
            }
        }
        harmonics
    }

    /// Plain triple-loop matrix product: a different arithmetic path from the
    /// BLAS-backed ndarray products used by the assembly.
    fn hand_mat_mul(left: &Array2<C>, right: &Array2<C>) -> Array2<C> {
        let rows = left.nrows();
        let inner = left.ncols();
        let columns = right.ncols();
        let mut product = Array2::zeros((rows, columns));
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = C::new(0.0, 0.0);
                for middle in 0..inner {
                    sum += left[[row, middle]] * right[[middle, column]];
                }
                product[[row, column]] = sum;
            }
        }
        product
    }

    /// [a, [b, c]] expanded into four matrix products: a b c - a c b - b c a + c b a.
    fn expanded_double_commutator(a: &Array2<C>, b: &Array2<C>, c: &Array2<C>) -> Array2<C> {
        let a_b = hand_mat_mul(a, b);
        let a_c = hand_mat_mul(a, c);
        let b_c = hand_mat_mul(b, c);
        let c_b = hand_mat_mul(c, b);
        let a_b_c = hand_mat_mul(&a_b, c);
        let a_c_b = hand_mat_mul(&a_c, b);
        let b_c_a = hand_mat_mul(&b_c, a);
        let c_b_a = hand_mat_mul(&c_b, a);
        &a_b_c - &a_c_b - &b_c_a + &c_b_a
    }

    /// Independent assembly of the same truncated van Vleck sum at one point.
    fn reference_effective(
        harmonics: &Array4<C>,
        point: usize,
        harmonic_max: isize,
        order: usize,
        has_time_dependence: bool,
        omega: f64,
    ) -> Array2<C> {
        let block = |n: isize| -> Array2<C> {
            harmonics
                .index_axis(Axis(1), point)
                .index_axis_move(Axis(0), harmonic_slot(n, harmonic_max))
                .to_owned()
        };
        let mut total = block(0);
        if !has_time_dependence {
            return total;
        }
        let inverse_omega = omega.recip();
        if order >= 1 {
            for n in 1..=harmonic_max {
                let negative = block(-n);
                let positive = block(n);
                let comm = hand_mat_mul(&negative, &positive) - hand_mat_mul(&positive, &negative);
                let scale = inverse_omega / (n as f64);
                total = total + comm.mapv(|value| value * scale);
            }
        }
        if order >= 2 {
            let inverse_omega_squared = inverse_omega * inverse_omega;
            for m in -harmonic_max..=harmonic_max {
                if m == 0 {
                    continue;
                }
                let term = expanded_double_commutator(&block(-m), &block(0), &block(m));
                let scale = inverse_omega_squared / (2.0 * (m as f64).powi(2));
                total = total + term.mapv(|value| value * scale);
            }
            for m in -harmonic_max..=harmonic_max {
                if m == 0 {
                    continue;
                }
                for m_prime in -harmonic_max..=harmonic_max {
                    if m_prime == 0 || m_prime == m || m_prime == 2 * m {
                        continue;
                    }
                    if (m_prime - m).abs() > harmonic_max {
                        continue;
                    }
                    let term = expanded_double_commutator(
                        &block(-m_prime),
                        &block(m_prime - m),
                        &block(m),
                    );
                    let scale = inverse_omega_squared / (3.0 * (m as f64) * (m_prime as f64));
                    total = total + term.mapv(|value| value * scale);
                }
            }
        }
        total
    }

    fn assemble(
        harmonics: &Array4<C>,
        harmonic_max: isize,
        order: usize,
        has_time_dependence: bool,
        omega: f64,
    ) -> Array3<C> {
        assemble_with(
            harmonics,
            harmonic_max,
            order,
            has_time_dependence,
            true,
            omega,
        )
    }

    /// Termwise reference assembly: both signs of every double-commutator
    /// family, no harmonic-consistency relation assumed.
    fn assemble_termwise(
        harmonics: &Array4<C>,
        harmonic_max: isize,
        order: usize,
        has_time_dependence: bool,
        omega: f64,
    ) -> Array3<C> {
        assemble_with(
            harmonics,
            harmonic_max,
            order,
            has_time_dependence,
            false,
            omega,
        )
    }

    fn assemble_with(
        harmonics: &Array4<C>,
        harmonic_max: isize,
        order: usize,
        has_time_dependence: bool,
        reuse_adjoint: bool,
        omega: f64,
    ) -> Array3<C> {
        let shape = harmonics.shape();
        let mut out = Array3::<C>::zeros((shape[1], shape[2], shape[3]));
        assemble_pointwise(
            harmonics.view(),
            harmonic_max,
            order,
            has_time_dependence,
            reuse_adjoint,
            omega,
            out.view_mut(),
        )
        .expect("a well-formed slice should assemble");
        out
    }

    fn assert_close(actual: ArrayView2<'_, C>, expected: &Array2<C>, tolerance: f64) {
        assert_eq!(actual.shape(), expected.shape());
        for ((row, column), value) in actual.indexed_iter() {
            let error = (*value - expected[[row, column]]).norm();
            assert!(
                error <= tolerance,
                "assembly differs at ({row}, {column}): {value} vs {} with absolute error {error:e}",
                expected[[row, column]]
            );
        }
    }

    fn assert_hermitian(matrix: ArrayView2<'_, C>, tolerance: f64) {
        let adjoint = matrix.t().mapv(|value| value.conj());
        assert_close(matrix, &adjoint, tolerance);
    }

    #[test]
    fn second_order_matches_expanded_triple_commutators() {
        // h = 2 keeps the B-family window active: the paired formula keeps one
        // representative per conjugate pair, and the termwise reference sums
        // both signs of m.
        let harmonic_max = 2_isize;
        let points = 3;
        let omega = 2.5;
        let harmonics = hermitian_consistent_harmonics(points, 3, harmonic_max, 0x5eed_1234);
        let out = assemble(&harmonics, harmonic_max, 2, true, omega);
        for point in 0..points {
            let expected = reference_effective(&harmonics, point, harmonic_max, 2, true, omega);
            let actual = out.index_axis(Axis(0), point);
            assert_close(actual, &expected, 1e-12);
            // The consistency relations make the complete sum Hermitian.
            assert_hermitian(actual, 1e-12);
        }
    }

    #[test]
    fn second_order_does_not_assume_harmonic_consistency() {
        // nsta = 8 puts the matrix products on the BLAS path, so this also
        // covers the geometric dot implementation rather than the small-size
        // fallback the other tests use.
        let harmonic_max = 2_isize;
        let points = 2;
        let omega = 1.75;
        let harmonics = random_harmonics(points, 8, harmonic_max, 0x00c0_ffee);
        let out = assemble_termwise(&harmonics, harmonic_max, 2, true, omega);
        for point in 0..points {
            let expected = reference_effective(&harmonics, point, harmonic_max, 2, true, omega);
            assert_close(out.index_axis(Axis(0), point), &expected, 1e-12);
        }
    }

    #[test]
    fn first_order_matches_the_single_commutator_sum() {
        let harmonic_max = 2_isize;
        let points = 3;
        let omega = 1.75;
        let harmonics = hermitian_consistent_harmonics(points, 3, harmonic_max, 0x1357_9bdf);
        let out = assemble(&harmonics, harmonic_max, 1, true, omega);
        let zero_slot = harmonic_slot(0, harmonic_max);
        let mut largest_change = 0.0_f64;
        for point in 0..points {
            let expected = reference_effective(&harmonics, point, harmonic_max, 1, true, omega);
            let actual = out.index_axis(Axis(0), point);
            assert_close(actual, &expected, 1e-12);
            let point_harmonics = harmonics.index_axis(Axis(1), point);
            let static_block = point_harmonics.index_axis_move(Axis(0), zero_slot);
            for (value, reference) in actual.iter().zip(static_block.iter()) {
                largest_change = largest_change.max((*value - *reference).norm());
            }
        }
        assert!(
            largest_change > 1e-6,
            "the order-one commutators must change the static block"
        );
    }

    #[test]
    fn zero_order_and_static_drives_copy_the_static_block() {
        let harmonic_max = 2_isize;
        let points = 3;
        let harmonics = random_harmonics(points, 2, harmonic_max, 0x2468_ace0);
        let zero_slot = harmonic_slot(0, harmonic_max);
        for &(order, has_time_dependence) in &[(0_usize, true), (0, false), (1, false), (2, false)]
        {
            // A zero frequency is accepted whenever no correction is requested.
            let out = assemble(&harmonics, harmonic_max, order, has_time_dependence, 0.0);
            for point in 0..points {
                assert_eq!(
                    out.index_axis(Axis(0), point),
                    harmonics
                        .index_axis(Axis(1), point)
                        .index_axis(Axis(0), zero_slot),
                    "order {order} with has_time_dependence = {has_time_dependence} \
must return H_0 exactly"
                );
            }
        }
    }

    #[test]
    fn thread_count_does_not_change_the_result() {
        let harmonic_max = 2_isize;
        let points = 6;
        let harmonics = hermitian_consistent_harmonics(points, 3, harmonic_max, 0x0bad_f00d);
        let serial = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("one rayon worker");
            pool.install(|| assemble(&harmonics, harmonic_max, 2, true, 2.5))
        };
        let parallel = {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("four rayon workers");
            pool.install(|| assemble(&harmonics, harmonic_max, 2, true, 2.5))
        };
        // Per-point sums keep their fixed order, so the two runs must be
        // bit-identical rather than merely close.
        assert_eq!(serial, parallel);
    }

    #[test]
    fn rejects_malformed_slices() {
        let harmonic_max = 2_isize;
        let harmonics = random_harmonics(3, 2, harmonic_max, 0x1111_2222);
        let mut out = Array3::<C>::zeros((3, 2, 2));
        assemble_pointwise(
            harmonics.view(),
            harmonic_max,
            2,
            true,
            true,
            1.0,
            out.view_mut(),
        )
        .expect("a well-formed slice should assemble");

        let mut wrong_points = Array3::<C>::zeros((2, 2, 2));
        assert!(
            assemble_pointwise(
                harmonics.view(),
                harmonic_max,
                2,
                true,
                true,
                1.0,
                wrong_points.view_mut()
            )
            .is_err()
        );
        let mut non_square = Array3::<C>::zeros((3, 2, 3));
        assert!(
            assemble_pointwise(
                harmonics.view(),
                harmonic_max,
                2,
                true,
                true,
                1.0,
                non_square.view_mut()
            )
            .is_err()
        );
        let mut wrong_states = Array3::<C>::zeros((3, 3, 3));
        assert!(
            assemble_pointwise(
                harmonics.view(),
                harmonic_max,
                2,
                true,
                true,
                1.0,
                wrong_states.view_mut()
            )
            .is_err()
        );
        let truncated = random_harmonics(3, 2, 1, 0x3333_4444);
        assert!(
            assemble_pointwise(
                truncated.view(),
                harmonic_max,
                2,
                true,
                true,
                1.0,
                out.view_mut(),
            )
            .is_err()
        );
        assert!(
            assemble_pointwise(harmonics.view(), -1, 2, true, true, 1.0, out.view_mut()).is_err()
        );
        assert!(
            assemble_pointwise(
                harmonics.view(),
                harmonic_max,
                3,
                true,
                true,
                1.0,
                out.view_mut(),
            )
            .is_err()
        );
        assert!(
            assemble_pointwise(
                harmonics.view(),
                harmonic_max,
                1,
                true,
                true,
                0.0,
                out.view_mut(),
            )
            .is_err()
        );
        assert!(
            assemble_pointwise(
                harmonics.view(),
                harmonic_max,
                1,
                true,
                true,
                f64::NAN,
                out.view_mut()
            )
            .is_err()
        );
    }

    #[test]
    fn adjoint_reuse_matches_the_termwise_sum() {
        // The paired form drops the explicitly summed conjugate half, so it
        // must reproduce the termwise sum on harmonic-consistent blocks for
        // every supported order and window.
        for harmonic_max in [1_isize, 2, 3] {
            let points = 2;
            let omega = 2.25;
            let harmonics = hermitian_consistent_harmonics(points, 3, harmonic_max, 0x5151_2024);
            for order in [1_usize, 2] {
                let paired = assemble(&harmonics, harmonic_max, order, true, omega);
                let termwise = assemble_termwise(&harmonics, harmonic_max, order, true, omega);
                for point in 0..points {
                    assert_close(
                        paired.index_axis(Axis(0), point),
                        &termwise.index_axis(Axis(0), point).to_owned(),
                        1e-12,
                    );
                }
            }
        }
    }

    #[test]
    fn adjoint_reuse_halves_the_order_two_terms() {
        // Count the terms each branch enumerates. Every term costs two matrix
        // products, so a halved term count is a halved product count.
        for harmonic_max in [1_isize, 2, 3, 4] {
            let a_paired = a_family_orders(harmonic_max, true).count();
            let a_termwise = a_family_orders(harmonic_max, false).count();
            assert_eq!(a_paired * 2, a_termwise, "A family at h={harmonic_max}");
            let b_paired = mixed_family_pairs(harmonic_max, true).count();
            let b_termwise = mixed_family_pairs(harmonic_max, false).count();
            assert_eq!(b_paired * 2, b_termwise, "B family at h={harmonic_max}");
        }
        // The production window h = 2 has two A terms (m = 1, 2) and two B
        // pairs ((1, -1) and (2, 1)) against four termwise B pairs.
        assert_eq!(a_family_orders(2, true).count(), 2);
        assert_eq!(
            mixed_family_pairs(2, true).collect::<Vec<_>>(),
            vec![(1, -1), (2, 1)]
        );
        assert_eq!(mixed_family_pairs(2, false).count(), 4);
    }

    #[test]
    fn adjoint_reuse_flag_selects_the_paired_branch() {
        // On harmonic-inconsistent blocks the two branches are different
        // formulas, so this pins that the flag actually switches the branch.
        let harmonic_max = 2_isize;
        let inconsistent = random_harmonics(2, 3, harmonic_max, 0x9999_aaaa);
        let paired = assemble(&inconsistent, harmonic_max, 2, true, 2.0);
        let termwise = assemble_termwise(&inconsistent, harmonic_max, 2, true, 2.0);
        let mut largest = 0.0_f64;
        for (left, right) in paired.iter().zip(termwise.iter()) {
            largest = largest.max((left - right).norm());
        }
        assert!(
            largest > 1e-6,
            "the paired branch must differ from the termwise branch on \
harmonic-inconsistent blocks, largest difference {largest:e}"
        );
    }
}
