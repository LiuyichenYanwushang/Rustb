//! Van Vleck assembly through cubic order in inverse frequency.

use super::C;
use crate::error::{Result, TbError};
use ndarray::linalg::general_mat_mul;
use ndarray::{
    Array2, Array3, ArrayView2, ArrayView3, ArrayView4, ArrayViewMut2, ArrayViewMut3, Axis, Zip,
};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

const NONFINITE_SCALE: &str = "a van Vleck frequency scaling factor is non-finite; \
increase omega0_ev or lower the requested order";

/// Assemble the effective Hamiltonian at every k point of one slice.
///
/// Layout: harmonics is (2 * cache_max + 1, points, nsta, nsta) with slot n + cache_max;
/// out is (points, nsta, nsta) and receives
///
///   H_eff - H_0 = sum_{n=1..h} [H_-n, H_n] / (n W)                       (order >= 1)
///         + sum_{m != 0, |m| <= h}
///             [H_-m, [H_0, H_m]] / (2 m^2 W^2)                      (order >= 2)
///         + sum_{m, m' with m' != 0, m' != m, m' != 2m,
///                         |m|, |m'| <= h}
///             [H_-m', [H_(m'-m), H_m]] / (3 m m' W^2)               (order >= 2)
///
/// The correction sums are skipped when order is 0 or the drive has no time
/// dependence. The difference m'-m can reach 2h; m' == 2m makes the inner
/// commutator vanish and is skipped. Order 3 additionally evaluates the
/// zero-average generator expression in `ThirdOrderScratch` with bare
/// harmonics through 3h.
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
    if order > 3 {
        return Err(TbError::Other(format!(
            "the k-space van Vleck assembly supports orders 0 through 3, got {order}"
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
    if harmonic_shape[0] < slots_needed || harmonic_shape[0] % 2 != 1 {
        return Err(TbError::Other(format!(
            "k-space harmonic slice has {} slots but {slots_needed} are required for \
harmonic_max = {harmonic_max} and order {effective_order}",
            harmonic_shape[0]
        )));
    }
    let cache_max = (harmonic_shape[0] / 2) as isize;
    if effective_order >= 1 && (!omega.is_finite() || omega == 0.0) {
        return Err(TbError::Other(format!(
            "the k-space van Vleck assembly needs a finite nonzero omega, got {omega}"
        )));
    }
    let inverse_omega = omega.recip();
    let extreme_products = harmonics.iter().any(|z| z.re.abs().max(z.im.abs()) > 1e95);
    let third_order_cutoff = (effective_order == 3 && harmonic_max > 0).then_some(harmonic_max);

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
        rows.par_chunks_mut(chunk_len).enumerate().try_for_each(
            |(chunk_index, chunk)| -> Result<()> {
                let mut scratch = Scratch::new(nsta, third_order_cutoff);
                for (offset, target) in chunk.iter_mut().enumerate() {
                    let point = chunk_index * chunk_len + offset;
                    assemble_point_into(
                        harmonics.index_axis(Axis(1), point),
                        harmonic_max,
                        cache_max,
                        effective_order,
                        inverse_omega,
                        omega,
                        reuse_adjoint,
                        extreme_products,
                        target,
                        &mut scratch,
                    )?;
                }
                Ok(())
            },
        )?;
    } else {
        let mut scratch = Scratch::new(nsta, third_order_cutoff);
        for (point, target) in rows.iter_mut().enumerate() {
            assemble_point_into(
                harmonics.index_axis(Axis(1), point),
                harmonic_max,
                cache_max,
                effective_order,
                inverse_omega,
                omega,
                reuse_adjoint,
                extreme_products,
                target,
                &mut scratch,
            )?;
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
    third: Option<ThirdOrderScratch>,
}

impl Scratch {
    fn new(nsta: usize, third_order_cutoff: Option<isize>) -> Self {
        Self {
            total: Array2::zeros((nsta, nsta)),
            inner: Array2::zeros((nsta, nsta)),
            outer: Array2::zeros((nsta, nsta)),
            third: third_order_cutoff.map(|h| ThirdOrderScratch::new(nsta, h)),
        }
    }
}

/// Third-order Fourier contractions. All storage is reused across the chunk's
/// k points. Bare operands are normalized only at extreme amplitudes.
struct ThirdOrderScratch {
    h: isize,
    va: Array3<C>,
    b: Array3<C>,
    inner: Array2<C>,
    coefficient: Array2<C>,
    scaled_operands: Option<[Array2<C>; 2]>,
}

impl ThirdOrderScratch {
    fn new(nsta: usize, h: isize) -> Self {
        let slots = harmonic_slot(2 * h, 2 * h) + 1;
        Self {
            h,
            va: Array3::zeros((slots, nsta, nsta)),
            b: Array3::zeros((slots, nsta, nsta)),
            inner: Array2::zeros((nsta, nsta)),
            coefficient: Array2::zeros((nsta, nsta)),
            scaled_operands: None,
        }
    }

    fn add_into<'a>(
        &mut self,
        harmonic: impl Fn(isize) -> ArrayView2<'a, C>,
        omega: f64,
        total: &mut Array2<C>,
        forward: &mut Array2<C>,
        backward: &mut Array2<C>,
    ) -> Result<()> {
        let h = self.h;
        let slot = |m| harmonic_slot(m, 2 * h);
        let a_orders = || (-h..=h).filter(|&m| m != 0);
        let b_orders = || (-2 * h..=2 * h).filter(|&m| m != 0);
        let zero = C::new(0.0, 0.0);
        let one = C::new(1.0, 0.0);
        let mut largest = 0.0_f64;
        let mut smallest = f64::INFINITY;
        for m in -3 * h..=3 * h {
            for z in &harmonic(m) {
                for value in [z.re.abs(), z.im.abs()] {
                    if !value.is_finite() {
                        return Err(TbError::Other(
                            "non-finite third-order van Vleck operand".into(),
                        ));
                    }
                    largest = largest.max(value);
                    if value != 0.0 {
                        smallest = smallest.min(value);
                    }
                }
            }
        }
        let input_exponent = if largest != 0.0 && !(1e-64..=1e64).contains(&largest) {
            binary_parts(largest).1
        } else {
            0
        };
        // A normal-sized component can hide a much smaller quartic product.
        // Fail closed when dividing by |W| < 1 could restore such products,
        // even if the largest component needs no scaling. Leave headroom
        // above the fourth root of the smallest normal f64 for harmonic weights.
        if input_exponent == 0 && omega.abs() < 1.0 && smallest < 1e-72 {
            return Err(TbError::Other(
                "third-order van Vleck operands exceed the representable product range".into(),
            ));
        }
        let mut scaled_operands = if input_exponent != 0 {
            // Reserve exponent headroom for four nonzero components and their
            // harmonic weights. A single normalization cannot safely span
            // arbitrary dynamic range.
            if binary_parts(smallest).1 - input_exponent < -200 {
                return Err(TbError::Other(
                    "third-order van Vleck operands exceed the representable product range".into(),
                ));
            }
            Some(
                self.scaled_operands.get_or_insert_with(|| {
                    [Array2::zeros(total.dim()), Array2::zeros(total.dim())]
                }),
            )
        } else {
            None
        };
        let exponent_step = (-input_exponent).clamp(-1022, 1023);
        let factor = normal_power_of_two(exponent_step);
        let tail = normal_power_of_two(-input_exponent - exponent_step);
        let copy_scaled = |dst: &mut Array2<C>, src: ArrayView2<'_, C>| {
            Zip::from(dst)
                .and(src)
                .for_each(|dst, &z| *dst = (z * factor) * tail);
        };
        let bare = |m| (harmonic(m), true);
        let mut add_commutator = |(left, left_bare): (ArrayView2<'_, C>, bool),
                                  (right, right_bare): (ArrayView2<'_, C>, bool),
                                  mut out: ArrayViewMut2<'_, C>,
                                  weight: f64| {
            if left.iter().all(|&z| z == zero)
                || right.iter().all(|&z| z == zero)
                || ((input_exponent == 0 || left_bare == right_bare)
                    && (left == right || left.iter().zip(right.iter()).all(|(a, b)| *a == -*b)))
                || (diagonal(left) && diagonal(right))
            {
                return;
            }
            // Separate the two products so exact cancellation is not lost
            // to BLAS's fused accumulation. No temporary allocation.
            if let Some([scaled_left, scaled_right]) = scaled_operands.as_deref_mut() {
                let left = if left_bare {
                    copy_scaled(scaled_left, left);
                    scaled_left.view()
                } else {
                    left
                };
                let right = if right_bare {
                    copy_scaled(scaled_right, right);
                    scaled_right.view()
                } else {
                    right
                };
                general_mat_mul(one, &left, &right, zero, forward);
                general_mat_mul(one, &right, &left, zero, backward);
            } else {
                general_mat_mul(one, &left, &right, zero, forward);
                general_mat_mul(one, &right, &left, zero, backward);
            }
            Zip::from(&mut out)
                .and(forward.view())
                .and(backward.view())
                .for_each(|out, a, b| *out += (*a - *b) * weight);
        };

        // A_m = H_m/m for 0 < |m| <= h. Compute C = [V,A] through 2h
        // (requiring V through 3h), then B_m = [H0+V/2,A]_m/m there.
        // Apply A's harmonic denominators outside each commutator.
        for q in -2 * h..=2 * h {
            let mut va = self.va.index_axis_mut(Axis(0), slot(q));
            va.fill(zero);
            for m in a_orders().filter(|&m| q != m) {
                add_commutator(bare(q - m), bare(m), va.view_mut(), 1.0 / m as f64);
            }
            if q != 0 {
                let mut b = self.b.index_axis_mut(Axis(0), slot(q));
                let qi = 1.0 / q as f64;
                b.assign(&va);
                b.mapv_inplace(|z| z * (0.5 * qi));
                if q.abs() <= h {
                    add_commutator(bare(0), bare(q), b, qi * qi);
                }
            }
        }
        let va = |m| (self.va.index_axis(Axis(0), slot(m)), false);
        let b = |m| (self.b.index_axis(Axis(0), slot(m)), false);

        // K3 = < [C,B]/12 + [[H0/2+V/3,B]+[C,A]/24+[C_0,A]/12,A] >,
        // since K1 = C_0/2. Compute only the inner harmonics needed by A.
        self.coefficient.fill(zero);
        for q in b_orders() {
            add_commutator(va(-q), b(q), self.coefficient.view_mut(), 1.0 / 12.0);
        }
        for q in a_orders() {
            self.inner.fill(zero);
            add_commutator(bare(0), b(q), self.inner.view_mut(), 0.5);
            for m in b_orders().filter(|&m| q != m) {
                add_commutator(bare(q - m), b(m), self.inner.view_mut(), 1.0 / 3.0);
            }
            for m in a_orders() {
                add_commutator(
                    va(q - m),
                    bare(m),
                    self.inner.view_mut(),
                    1.0 / (24.0 * m as f64),
                );
            }
            add_commutator(
                va(0),
                bare(q),
                self.inner.view_mut(),
                1.0 / (12.0 * q as f64),
            );
            add_commutator(
                (self.inner.view(), false),
                bare(-q),
                self.coefficient.view_mut(),
                -1.0 / q as f64,
            );
        }

        let (omega_mantissa, omega_exponent) = binary_parts(omega);
        let mantissa_scale = omega_mantissa * omega_mantissa * omega_mantissa;
        let exponent_scale = 4 * input_exponent - 3 * omega_exponent;
        let restore = |value: f64| {
            if value == 0.0 || !value.is_finite() {
                value
            } else {
                let (mantissa, exponent) = binary_parts(value);
                restore_binary_scale(mantissa / mantissa_scale, exponent + exponent_scale)
            }
        };
        total.zip_mut_with(&self.coefficient, |sum, &z| {
            *sum += C::new(restore(z.re), restore(z.im));
        });
        if !matrix_finite(total) {
            return Err(TbError::Other(
                "the third-order van Vleck correction exceeds the representable range".into(),
            ));
        }
        Ok(())
    }
}

// IEEE-754 normal powers of two. Callers keep the exponent in -1022..=1023.
fn normal_power_of_two(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

/// Finite nonzero x = m * 2^e, with 1 <= |m| < 2, including subnormal x.
fn binary_parts(x: f64) -> (f64, i32) {
    let (normal, shift) = if x.is_subnormal() {
        (x * normal_power_of_two(52), -52)
    } else {
        (x, 0)
    };
    let exponent = (normal.abs().to_bits() >> 52) as i32 - 1023;
    (normal / normal_power_of_two(exponent), exponent + shift)
}

/// Restore the scale of a bounded mantissa (1/8 <= |m| < 2). Splitting the
/// boundary exponents keeps the first product normal, so subnormal results
/// round only once and no intermediate overflows before a finite result.
fn restore_binary_scale(mantissa: f64, exponent: i32) -> f64 {
    match exponent {
        ..=-1076 => 0.0_f64.copysign(mantissa),
        -1075..=-1023 => {
            (mantissa * normal_power_of_two(-512)) * normal_power_of_two(exponent + 512)
        }
        1024..=1026 => (mantissa * normal_power_of_two(512)) * normal_power_of_two(exponent - 512),
        1027.. => f64::INFINITY.copysign(mantissa),
        _ => mantissa * normal_power_of_two(exponent),
    }
}

/// Number of harmonic slots the requested effective order reads.
fn required_slots(harmonic_max: isize, effective_order: usize) -> Result<usize> {
    let span = harmonic_max
        .checked_mul(2 * effective_order as isize)
        .and_then(|span| span.checked_add(1));
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
fn commutator_into(
    left: ArrayView2<'_, C>,
    right: ArrayView2<'_, C>,
    out: &mut Array2<C>,
    separate: bool,
) {
    if separate {
        if left == right
            || left.iter().zip(right.iter()).all(|(a, b)| *a == -*b)
            || left.iter().all(|z| *z == C::new(0.0, 0.0))
            || right.iter().all(|z| *z == C::new(0.0, 0.0))
            || (diagonal(left) && diagonal(right))
        {
            out.fill(C::new(0.0, 0.0));
            return;
        }
        // Do not turn an exactly commuting pair into an out-of-range
        // correction merely because a fused multiply-add rounded differently.
        out.assign(&(left.dot(&right) - right.dot(&left)));
        return;
    }
    general_mat_mul(C::new(1.0, 0.0), &left, &right, C::new(0.0, 0.0), out);
    general_mat_mul(C::new(-1.0, 0.0), &right, &left, C::new(1.0, 0.0), out);
}

fn diagonal(matrix: ArrayView2<'_, C>) -> bool {
    matrix
        .indexed_iter()
        .all(|((i, j), z)| i == j || *z == C::new(0.0, 0.0))
}

/// Add a second-order term and, when requested, its adjoint. If the
/// combined frequency factor is tiny, apply it in two steps rather than
/// throwing away a representable final correction before scaling the block.
fn add_second_order(
    total: &mut Array2<C>,
    matrix: &Array2<C>,
    inverse: f64,
    inverse_squared: f64,
    weight: f64,
    reuse_adjoint: bool,
) -> Result<()> {
    if !inverse_squared.is_finite() {
        return if matrix.iter().all(|z| *z == C::new(0.0, 0.0)) {
            Ok(())
        } else {
            Err(TbError::Other(NONFINITE_SCALE.into()))
        };
    }
    let scale = inverse_squared * weight;
    let scaled = |value: C| {
        if scale.is_normal() {
            value * scale
        } else {
            (value * inverse) * (inverse * weight)
        }
    };
    for ((i, j), target) in total.indexed_iter_mut() {
        *target += scaled(matrix[[i, j]]);
        if reuse_adjoint {
            *target += scaled(matrix[[j, i]].conj());
        }
    }
    Ok(())
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
                .filter(move |m_prime| *m_prime != 0 && *m_prime != m && *m_prime != 2 * m)
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
    cache_max: isize,
    order: usize,
    inverse_omega: f64,
    omega: f64,
    reuse_adjoint: bool,
    extreme_products: bool,
    target: &mut ArrayViewMut2<'_, C>,
    scratch: &mut Scratch,
) -> Result<()> {
    let inverse_omega_squared = inverse_omega * inverse_omega;
    let harmonic = |n: isize| point_harmonics.index_axis_move(Axis(0), harmonic_slot(n, cache_max));
    scratch.total.fill(C::new(0.0, 0.0));

    if order >= 1 {
        // [H_-n, H_n] is already Hermitian, so no pairing is needed here.
        for n in 1..=harmonic_max {
            let left = harmonic(-n);
            let right = harmonic(n);
            let mut scale = inverse_omega / n as f64;
            commutator_into(
                left,
                right,
                &mut scratch.inner,
                !scale.is_finite() || extreme_products,
            );
            if !scale.is_finite() {
                if scratch.inner.iter().all(|z| *z == C::new(0.0, 0.0)) {
                    continue;
                }
                return Err(TbError::Other(NONFINITE_SCALE.into()));
            }
            if extreme_products && !matrix_finite(&scratch.inner) && scale.abs() < 1.0 {
                let scaled = scale_for_retry(right, scale)?;
                commutator_into(left, scaled.view(), &mut scratch.inner, true);
                scale = 1.0;
            }
            scratch.total.scaled_add(C::new(scale, 0.0), &scratch.inner);
        }
    }

    if order >= 2 {
        // A family: T_m = [H_-m, [H_0, H_m]]. With reuse_adjoint only the
        // m > 0 representative of each conjugate pair is formed and added
        // together with its adjoint; T_m^dagger = T_-m carries the same
        // weight 1/(2 m^2 W^2).
        for m in a_family_orders(harmonic_max, reuse_adjoint) {
            let inverse = nested_into(
                harmonic(-m),
                harmonic(0),
                harmonic(m),
                scratch,
                inverse_omega,
                inverse_omega_squared,
                extreme_products,
            )?;
            let weight = 1.0 / (2.0 * (m as f64).powi(2));
            add_second_order(
                &mut scratch.total,
                &scratch.outer,
                inverse.0,
                inverse.1,
                weight,
                reuse_adjoint,
            )?;
        }
        // B family: sign reversal gives the adjoint with unchanged weight.
        // Separately, at fixed m', swapping m and m'-m negates the inner
        // commutator. Fold the reciprocal weights only when the swapped m
        // is also in the window; otherwise the original term stands alone.
        // Its positive-m representative may be the sign-reversed (adjoint)
        // partner, so canonicalize by the magnitudes of both inner indices.
        for (m, m_prime) in mixed_family_pairs(harmonic_max, reuse_adjoint) {
            let difference = m_prime - m;
            let paired = reuse_adjoint && difference.abs() <= harmonic_max;
            if paired && m > difference.abs() {
                continue;
            }
            let inverse = nested_into(
                harmonic(-m_prime),
                harmonic(difference),
                harmonic(m),
                scratch,
                inverse_omega,
                inverse_omega_squared,
                extreme_products,
            )?;
            let weight = if paired {
                (difference - m) as f64 / (3.0 * m_prime as f64 * m as f64 * difference as f64)
            } else {
                1.0 / (3.0 * m as f64 * m_prime as f64)
            };
            add_second_order(
                &mut scratch.total,
                &scratch.outer,
                inverse.0,
                inverse.1,
                weight,
                reuse_adjoint,
            )?;
        }
    }

    if let Some(third) = &mut scratch.third {
        third.add_into(
            harmonic,
            omega,
            &mut scratch.total,
            &mut scratch.inner,
            &mut scratch.outer,
        )?;
    }

    target.assign(&scratch.total);
    Ok(())
}

fn matrix_finite(matrix: &Array2<C>) -> bool {
    matrix.iter().all(|z| z.re.is_finite() && z.im.is_finite())
}

fn scale_for_retry(matrix: ArrayView2<'_, C>, factor: f64) -> Result<Array2<C>> {
    let scaled = matrix.mapv(|z| z * factor);
    if matrix.iter().zip(&scaled).any(|(before, after)| {
        [(before.re, after.re), (before.im, after.im)]
            .into_iter()
            .any(|(a, b)| a != 0.0 && !b.is_normal())
    }) {
        return Err(TbError::Other(
            "the k-space van Vleck operands exceed the representable product range".into(),
        ));
    }
    Ok(scaled)
}

fn nested_into(
    outer: ArrayView2<'_, C>,
    left: ArrayView2<'_, C>,
    right: ArrayView2<'_, C>,
    scratch: &mut Scratch,
    inverse: f64,
    inverse_squared: f64,
    extreme_products: bool,
) -> Result<(f64, f64)> {
    let separate = !inverse_squared.is_finite() || extreme_products;
    commutator_into(left, right, &mut scratch.inner, separate);
    commutator_into(outer, scratch.inner.view(), &mut scratch.outer, separate);
    if extreme_products && !matrix_finite(&scratch.outer) && inverse.abs() < 1.0 {
        // Each commutator uses one factor of 1/W. Retry with the factors
        // before the products only when the usual arithmetic overflows.
        let right = scale_for_retry(right, inverse)?;
        let outer = scale_for_retry(outer, inverse)?;
        commutator_into(left, right.view(), &mut scratch.inner, true);
        commutator_into(outer.view(), scratch.inner.view(), &mut scratch.outer, true);
        Ok((1.0, 1.0))
    } else {
        Ok((inverse, inverse_squared))
    }
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
        let cache_max = (harmonics.shape()[0] / 2) as isize;
        let block = |n: isize| -> Array2<C> {
            harmonics
                .index_axis(Axis(1), point)
                .index_axis_move(Axis(0), harmonic_slot(n, cache_max))
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
        out += &harmonics.index_axis(Axis(0), shape[0] / 2);
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

    fn third_harmonic_toy(amplitude: f64) -> Array4<C> {
        let h = 1;
        let mut harmonics = Array4::zeros((7, 1, 2, 2));
        for m in [-1, 1] {
            let s = harmonic_slot(m, 3 * h);
            harmonics[[s, 0, 0, 1]] = C::new(amplitude, 0.0);
            harmonics[[s, 0, 1, 0]] = C::new(amplitude, 0.0);
        }
        for m in [-3, 3] {
            let s = harmonic_slot(m, 3 * h);
            let value = C::new(0.0, (m.signum() as f64) * amplitude);
            harmonics[[s, 0, 0, 0]] = value;
            harmonics[[s, 0, 1, 1]] = -value;
        }
        harmonics
    }

    #[test]
    fn cubic_reads_third_harmonic_at_extreme_energy_scales() {
        // With H_{±1}=X, H_{±3}=±iZ, K3=2Y and both lower orders vanish.
        // The first case needs split frequency scaling even with normal K3;
        // the next two require normalized operands, and the last pins support
        // for subnormal W without ever forming its reciprocal in the cubic.
        for (amplitude, omega, correction) in [
            (1e60, 1e120, 2e-120),
            (1e90, 1e90, 2e90),
            (1e-90, 1e-90, 2e-90),
            (f64::from_bits(1), f64::from_bits(1), f64::from_bits(2)),
        ] {
            let out = assemble(&third_harmonic_toy(amplitude), 1, 3, true, omega);
            assert_eq!(out[[0, 0, 0]], C::new(0.0, 0.0));
            assert!((out[[0, 0, 1]] / correction - C::new(0.0, -1.0)).norm() < 1e-14);
            assert!((out[[0, 1, 0]] / correction - C::new(0.0, 1.0)).norm() < 1e-14);
            assert_eq!(out[[0, 1, 1]], C::new(0.0, 0.0));
        }
    }

    #[test]
    fn cubic_rejects_unsafe_dynamic_range_before_normalizing() {
        let mut harmonics = third_harmonic_toy(1e90);
        harmonics[[0, 0, 0, 0]].re = 1e-250;
        let mut out = Array3::zeros((1, 2, 2));
        let error = assemble_pointwise(harmonics.view(), 1, 3, true, true, 1e90, out.view_mut())
            .unwrap_err();
        assert!(error.to_string().contains("representable product range"));
    }

    #[test]
    fn cubic_rejects_mixed_scales_that_can_hide_representable_products() {
        let mut harmonics = third_harmonic_toy(1e-90);
        harmonics[[3, 0, 0, 0]].re = 1.0;
        harmonics[[3, 0, 1, 1]].re = 1.0;
        let mut out = Array3::zeros((1, 2, 2));
        // The scalar H0 does not change K3. Its largest entry must not mask
        // underflow of a raw quartic that would give a 2e-120 correction.
        for omega in [1e-80, 1e-39, 0.5] {
            let err = assemble_pointwise(harmonics.view(), 1, 3, true, true, omega, out.view_mut())
                .unwrap_err();
            assert!(err.to_string().contains("representable product range"));
        }
    }

    #[test]
    fn second_order_matches_expanded_triple_commutators() {
        // h = 2 keeps the B-family window active: the paired formula keeps one
        // representative per conjugate pair, and the termwise reference sums
        // both signs of m.
        let harmonic_max = 2_isize;
        let points = 3;
        let omega = 2.5;
        let harmonics = hermitian_consistent_harmonics(points, 3, 2 * harmonic_max, 0x5eed_1234);
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
    fn large_frequency_preserves_a_representable_second_order_delta() {
        // The raw double commutator is representable but W^-2 is not.
        let mut harmonics = Array4::<C>::zeros((5, 1, 2, 2));
        harmonics[[2, 0, 0, 0]] = C::new(1e100, 0.0);
        harmonics[[2, 0, 1, 1]] = C::new(-1e100, 0.0);
        for m in [1, 3] {
            harmonics[[m, 0, 0, 1]] = C::new(1e100, 0.0);
            harmonics[[m, 0, 1, 0]] = C::new(1e100, 0.0);
        }
        let mut delta = Array3::<C>::zeros((1, 2, 2));
        assemble_pointwise(harmonics.view(), 1, 2, true, true, 1e200, delta.view_mut()).unwrap();
        assert!((delta[[0, 0, 0]].re / -4e-100 - 1.0).abs() < 1e-14);
        assert_eq!(delta[[0, 0, 0]], -delta[[0, 1, 1]]);
    }

    #[test]
    fn identical_harmonics_commute_without_forming_unrepresentable_products() {
        let mut harmonics = Array4::<C>::zeros((3, 1, 2, 2));
        for m in [0, 2] {
            harmonics[[m, 0, 0, 1]] = C::new(2.0_f64.powi(600), 0.0);
            harmonics[[m, 0, 1, 0]] = C::new(2.0_f64.powi(600), 0.0);
        }
        let mut delta = Array3::<C>::zeros((1, 2, 2));
        assemble_pointwise(
            harmonics.view(),
            1,
            1,
            true,
            true,
            f64::from_bits(1),
            delta.view_mut(),
        )
        .unwrap();
        assert!(delta.iter().all(|z| *z == C::new(0.0, 0.0)));
    }

    #[test]
    fn retry_rejects_erasing_a_small_operand_before_a_large_product() {
        let mut harmonics = Array4::<C>::zeros((5, 1, 3, 3));
        for i in [0, 1] {
            harmonics[[2, 0, i, i]] = C::new(2.0_f64.powi(1000), 0.0);
        }
        for m in [1, 3] {
            for (i, j, exponent) in [(0, 1, 500), (1, 2, -100)] {
                harmonics[[m, 0, i, j]] = C::new(2.0_f64.powi(exponent), 0.0);
                harmonics[[m, 0, j, i]] = harmonics[[m, 0, i, j]];
            }
        }
        let mut delta = Array3::<C>::zeros((1, 3, 3));
        let err = assemble_pointwise(
            harmonics.view(),
            1,
            2,
            true,
            true,
            2.0_f64.powi(1000),
            delta.view_mut(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("representable product range"));
    }

    #[test]
    fn second_order_does_not_assume_harmonic_consistency() {
        // nsta = 8 puts the matrix products on the BLAS path, so this also
        // covers the geometric dot implementation rather than the small-size
        // fallback the other tests use.
        let harmonic_max = 2_isize;
        let points = 2;
        let omega = 1.75;
        let harmonics = random_harmonics(points, 8, 2 * harmonic_max, 0x00c0_ffee);
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
        let harmonics = hermitian_consistent_harmonics(points, 3, 2 * harmonic_max, 0x0bad_f00d);
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
        let harmonics = random_harmonics(3, 2, 2 * harmonic_max, 0x1111_2222);
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
            let harmonics =
                hermitian_consistent_harmonics(points, 3, 2 * harmonic_max, 0x5151_2024);
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
        // At h = 2 there are two A terms and five B sign representatives
        // before swapped-inner folding. The last difference can be -2h.
        assert_eq!(a_family_orders(2, true).count(), 2);
        assert_eq!(
            mixed_family_pairs(2, true).collect::<Vec<_>>(),
            vec![(1, -2), (1, -1), (2, -2), (2, -1), (2, 1)]
        );
        assert_eq!(mixed_family_pairs(2, false).count(), 10);
    }

    #[test]
    fn adjoint_reuse_flag_selects_the_paired_branch() {
        // On harmonic-inconsistent blocks the two branches are different
        // formulas, so this pins that the flag actually switches the branch.
        let harmonic_max = 2_isize;
        let inconsistent = random_harmonics(2, 3, 2 * harmonic_max, 0x9999_aaaa);
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
