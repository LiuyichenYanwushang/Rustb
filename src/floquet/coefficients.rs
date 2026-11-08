//! Scalar Peierls-Floquet coefficient backend, link validation, and harmonic cache.

use super::FloquetDrive;
use crate::error::{Result, TbError};
use crate::{Model, RMatrixData};
use ndarray::prelude::*;
use ndarray::s;
use num_complex::Complex;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};

/// Backend selection for the Peierls Fourier coefficients `C_n(d)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeierlsFourierMethod {
    /// Numerical DFT on a uniform time grid.  Compiled for test builds only:
    /// production has exactly one coefficient backend, and the tests use this
    /// variant to cross-check it against an independent oracle
    /// ([`crate::floquet_test`]).
    #[cfg(test)]
    TimeGrid { n_time: usize },
    /// Generalized Bessel expansion via sequential one-mode convolutions.
    /// Exact and independent of `n_time`, but restricted to per-mode
    /// projections `R_α = |a_α·d| ≤` [`MAX_BESSEL_ARG`], and the same bound on
    /// the coherent sum for the single-harmonic closed form
    /// ([`MAX_BESSEL_ARG_CLOSED_FORM`]); the cache falls back to a self-sized
    /// per-link time-grid DFT (`fallback_time_grid_coeffs`) beyond that.
    Bessel {
        /// Minimum number of Bessel orders beyond `⌈R_α⌉` (the adaptive tail
        /// check may push the cutoff higher).
        cutoff_margin: isize,
    },
}

/// Precomputed `t_ij(R) * C_n(d)` for all harmonics `n ∈ [harmonic_min, harmonic_max]`.
///
/// `blocks` has shape `(harmonic_count, n_r, nsta, nsta)` where `harmonic_count = harmonic_max - harmonic_min + 1`.
/// Index `[i_n, i_r, i, j]` stores the `n = harmonic_min + i_n` Fourier component of hopping
/// from orbital `j` in cell `R = hamR[i_r]` to orbital `i` at the origin.
/// This is independent of `k` and reusable across the entire k-mesh.
pub(super) struct FloquetHarmonicCache {
    pub(super) harmonic_min: isize,
    pub(super) harmonic_max: isize,
    pub(super) blocks: Array4<Complex<f64>>,
}

impl FloquetHarmonicCache {
    #[inline]
    pub(super) fn harmonic_index(&self, n: isize) -> usize {
        debug_assert!(
            n >= self.harmonic_min && n <= self.harmonic_max,
            "Floquet harmonic n={n} is outside cached range [{}, {}]",
            self.harmonic_min,
            self.harmonic_max
        );
        (n - self.harmonic_min) as usize
    }

    #[inline]
    pub(super) fn harmonic_blocks(&self, n: isize) -> ArrayView3<'_, Complex<f64>> {
        self.blocks.index_axis(Axis(0), self.harmonic_index(n))
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Build the harmonic cache: `t_ij(R) * C_n(d)` for all `n ∈ [harmonic_min, harmonic_max]`.
    ///
    /// Returns a [`FloquetHarmonicCache`] whose `blocks[harmonic_index(n), i_r, i, j]`
    /// stores the `n`-th Fourier coefficient of the Peierls-dressed hopping from
    /// orbital `j` at cell `R = hamR[i_r]` to orbital `i` at the origin.  The
    /// cache is consumed by the real-space constructors. For a k-mesh, build
    /// `floquet_model` once and reuse its ordinary Hamiltonian and band solvers.
    ///
    /// When `drive.modes` is empty (static limit), the zero-frequency block is
    /// set to the original `ham` directly.
    pub(super) fn floquet_harmonic_cache(
        &self,
        drive: &FloquetDrive,
        harmonic_min: isize,
        harmonic_max: isize,
        method: &PeierlsFourierMethod,
    ) -> FloquetHarmonicCache {
        let nsta = self.nsta();
        let norb = self.norb();
        let n_r = self.hamR.nrows();
        let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
        let mut blocks = Array4::<Complex<f64>>::zeros((harmonic_count, n_r, nsta, nsta));

        if drive.modes.is_empty() {
            if harmonic_min <= 0 && 0 <= harmonic_max {
                blocks
                    .slice_mut(s![(0 - harmonic_min) as usize, .., .., ..])
                    .assign(&self.ham);
            }
            return FloquetHarmonicCache {
                harmonic_min,
                harmonic_max,
                blocks,
            };
        }

        // Phase 1: collect the DISTINCT link displacements among non-zero
        // hoppings.  Spin copies of the same orbital pair share the same
        // d, so this deduplicates the coefficient computation (a 4x saving
        // for spinful models).
        // Distinct-link map keyed by the bit pattern of the Cartesian
        // displacement (a fixed-size array avoids a heap allocation per
        // hopping entry).
        let mut d_index = std::collections::HashMap::<[u64; DIM], usize>::new();
        let mut unique_d = Vec::<Array1<f64>>::new();
        let mut entries = Vec::<(usize, usize, usize, usize)>::new(); // (i_r, i, j, d_idx)
        for i_r in 0..n_r {
            let r_vec = self.hamR.row(i_r);
            for i in 0..nsta {
                for j in 0..nsta {
                    if self.ham[[i_r, i, j]].re == 0.0 && self.ham[[i_r, i, j]].im == 0.0 {
                        continue;
                    }
                    let d_cart = self.link_displacement_cartesian(i % norb, j % norb, &r_vec);
                    let mut key = [0_u64; DIM];
                    for a in 0..DIM {
                        key[a] = d_cart[a].to_bits();
                    }
                    let index = *d_index.entry(key).or_insert_with(|| {
                        unique_d.push(d_cart);
                        unique_d.len() - 1
                    });
                    entries.push((i_r, i, j, index));
                }
            }
        }

        // Phase 2: coefficients per distinct d (parallel).  The Bessel backend
        // sizes its own per-link fallback grid from the link's bandwidth, so
        // production allocates no eager grid at all; the reference time grid
        // exists only in test builds.
        #[cfg(test)]
        let time_grid = match method {
            PeierlsFourierMethod::TimeGrid { n_time } => {
                Some(crate::floquet_test::FloquetTimeGrid::new(
                    drive,
                    *n_time,
                    harmonic_min,
                    harmonic_max,
                ))
            }
            PeierlsFourierMethod::Bessel { .. } => None,
        };
        // Per-call warn-once flag: the parallel loop below may hit the
        // fallback branch for many links, but the user only needs one
        // message per cache build.
        let fallback_warned = AtomicBool::new(false);
        let fallback_clamped = AtomicBool::new(false);
        let fallback_saturated = AtomicBool::new(false);
        let coeffs_per_d: Vec<Array1<Complex<f64>>> = unique_d
            .par_iter()
            .map(|d| match method {
                PeierlsFourierMethod::Bessel { cutoff_margin } => {
                    match bessel_peierls_coeffs(
                        d,
                        drive,
                        harmonic_min,
                        harmonic_max,
                        *cutoff_margin,
                    ) {
                        Ok(coeffs) => coeffs,
                        Err(error) => {
                            if !fallback_warned.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "Bessel backend unavailable for some links \
                                     ({error}); falling back to a per-link \
                                     time-grid DFT"
                                );
                            }
                            fallback_time_grid_coeffs(
                                d,
                                drive,
                                harmonic_min,
                                harmonic_max,
                                &fallback_clamped,
                                &fallback_saturated,
                            )
                        }
                    }
                }
                #[cfg(test)]
                PeierlsFourierMethod::TimeGrid { .. } => {
                    let time_grid = time_grid
                        .as_ref()
                        .expect("shared time grid is built for the TimeGrid backend");
                    Array1::from(crate::floquet_test::peierls_fourier_coeffs(
                        d,
                        harmonic_min,
                        harmonic_max,
                        drive,
                        time_grid,
                    ))
                }
            })
            .collect();

        // Phase 3: fill the blocks.
        for (i_r, i, j, d_index) in entries {
            let t = self.ham[[i_r, i, j]];
            for (i_n, coeff) in coeffs_per_d[d_index].iter().enumerate() {
                if coeff.re != 0.0 || coeff.im != 0.0 {
                    blocks[[i_n, i_r, i, j]] = t * coeff;
                }
            }
        }

        FloquetHarmonicCache {
            harmonic_min,
            harmonic_max,
            blocks,
        }
    }
}

// Every link must be either exactly representable by the Bessel backend or
// resolvable by the per-link time grid it falls back to.  Both public paths
// (Sambe and the effective model) share this check, so a drive that cannot be
// represented exactly anywhere is refused instead of being answered from a
// clamped, aliasing grid.
#[cfg(test)]
thread_local! {
    pub(super) static LINK_RESOLVABILITY_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn validate_link_resolvability<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
) -> Result<()> {
    if drive.modes.is_empty() {
        return Ok(());
    }
    let span = harmonic_max.saturating_sub(harmonic_min).saturating_add(1) as usize;
    let mut checked = std::collections::HashSet::<[u64; DIM]>::new();
    // Drive-level count, which chooses the backend's amplitude cap.
    let drive_carriers = drive.modes.iter().filter(|mode| mode.harmonic != 0).count();
    for ((r, i, j), hopping) in model.ham.indexed_iter() {
        if hopping.re == 0.0 && hopping.im == 0.0 {
            continue;
        }
        let d = model.link_displacement_cartesian(
            i % model.norb(),
            j % model.norb(),
            &model.hamR.row(r),
        );
        if d.iter().any(|x| !x.is_finite()) {
            return Err(TbError::Other(
                "Floquet link displacement is not finite".into(),
            ));
        }
        // Skip repeated links before projecting any modes; exact bit keys keep
        // the same numerical equivalence as the harmonic cache (no tolerance).
        if !checked.insert(std::array::from_fn(|a| d[a].to_bits())) {
            continue;
        }
        #[cfg(test)]
        LINK_RESOLVABILITY_SCANS.with(|count| count.set(count.get() + 1));
        // Mirror the backend's own applicability test.  It groups modes
        // differently depending on how many distinct harmonics survive:
        //
        // * a drive whose nonzero projections collapse to one coherent harmonic
        //   takes the closed form.  Its only precondition is that the *summed*
        //   projection fits the cap, and it allocates no convolution window at
        //   all, so no window bound applies;
        // * exactly two carriers may enumerate resonances without a window;
        //   the remaining drives convolve individual AC modes and bound their
        //   work window. DC never consumes an AC cap, ladder or window.
        //
        // Checking per mode in both cases would let a split amplitude (a + a)
        // slip past the cap; checking the sum in both would reject drives the
        // convolution handles.  Modes whose projection is exactly zero are
        // dropped first, as the backend drops them.
        // Walk the modes exactly like `single_harmonic_closed_form` does: it
        // keeps one running coherent sum and bails out to the convolution at the
        // *first* harmonic change — including when a later mode would have
        // cancelled that sum back to zero.  Accumulating per harmonic instead
        // would call such a drive exact while the backend convolves it.
        let mut single: Option<(isize, Complex<f64>)> = None;
        let mut per_mode = Vec::<(isize, f64)>::new();
        let mut several_harmonics = false;
        let mut carriers = 0_usize;
        let mut dc_re = 0.0_f64;
        for mode in &drive.modes {
            if mode.harmonic == 0 {
                dc_re += mode
                    .a_complex
                    .iter()
                    .zip(&d)
                    .map(|(a, x)| a.re * x)
                    .sum::<f64>();
                continue;
            }
            let phase: Complex<f64> = mode.a_complex.iter().zip(&d).map(|(a, x)| a * x).sum();
            if !phase.re.is_finite() || !phase.im.is_finite() {
                return Err(TbError::Other(
                    "Floquet link phase amplitude is not finite".into(),
                ));
            }
            if phase.re == 0.0 && phase.im == 0.0 {
                continue;
            }
            // A single coherent carrier only uses the summed projection. An
            // individual norm may overflow even when that sum cancels exactly.
            per_mode.push((mode.harmonic, phase.norm()));
            carriers += 1;
            match &mut single {
                Some((harmonic, total)) if *harmonic == mode.harmonic => *total += phase,
                Some(_) => several_harmonics = true,
                None => single = Some((mode.harmonic, phase)),
            }
        }
        if !dc_re.is_finite() {
            return Err(TbError::Other(
                "Floquet link static phase is not finite".into(),
            ));
        }
        let exact = if !several_harmonics {
            // One coherent harmonic (or none at all): the closed form is exact
            // for any amplitude whose coherent sum fits its cap, and an exactly
            // cancelling sum never reaches the ladder.
            match single {
                Some((_, z)) => {
                    z.re == 0.0 && z.im == 0.0 || z.norm() <= MAX_BESSEL_ARG_CLOSED_FORM
                }
                None => true,
            }
        } else {
            // Mirror the backend: the enumeration runs on exactly two carriers,
            // counted exactly as the backend counts them (nonzero harmonic AND
            // nonzero link projection), and only with the drive-level count in
            // agreement.  Its scan is the *smaller carrier's* cutoff, so static
            // modes must not enter the minimum.
            let carrier_cap = if carriers == drive_carriers && carriers == 2 {
                per_mode
                    .iter()
                    .filter(|(harmonic, _)| *harmonic != 0)
                    .all(|(_, r)| *r <= MAX_BESSEL_ARG_CLOSED_FORM)
            } else {
                false
            };
            if carrier_cap {
                // An upper bound on that cutoff: the ladder seeds at
                // `⌈r⌉ + margin + 8 + 16·r^{1/3}` and at most doubles a bounded
                // number of times, so `4·(⌈r⌉ + 64 + 16·r^{1/3})` covers every
                // case the sweep can reach.  Overestimating only makes the
                // validator stricter than the backend, never looser.
                let scan = per_mode
                    .iter()
                    .filter(|(harmonic, _)| *harmonic != 0)
                    .map(|(_, r)| {
                        4_usize.saturating_mul(
                            (r.ceil() as usize)
                                .saturating_add(64)
                                .saturating_add((16.0 * r.cbrt()) as usize),
                        )
                    })
                    .min()
                    .unwrap_or(0);
                span.saturating_mul(2_usize.saturating_mul(scan) + 1) <= MAX_BESSEL_ENUM_WORK
            } else {
                // Same conservative cutoff estimate as the enumeration branch:
                // ceil(r) + 64 alone understates M at large r (measured +230 at
                // r = 16000), so the window would look smaller than the backend's.
                let cutoff = |r: f64| {
                    4_usize.saturating_mul(
                        (r.ceil() as usize)
                            .saturating_add(64)
                            .saturating_add((16.0 * r.cbrt()) as usize),
                    )
                };
                per_mode.iter().all(|(_, r)| *r <= MAX_BESSEL_ARG)
                    && per_mode.iter().fold(span, |acc, (harmonic, r)| {
                        acc.saturating_add(
                            2_usize
                                .saturating_mul(harmonic.unsigned_abs())
                                .saturating_mul(cutoff(*r)),
                        )
                    }) <= MAX_BESSEL_WINDOW
            }
        };
        if exact {
            continue;
        }
        // Otherwise the per-link grid must resolve the link's spectrum; refuse
        // the whole call rather than return coefficients that alias.  The
        // requested range counts too: an n-point DFT only returns true `C_n`
        // for `|n| < n/2`.
        let grid = fallback_grid_size(drive, &d, harmonic_min, harmonic_max);
        let needed = grid.required.max(grid.request_range);
        if needed > FALLBACK_GRID_MAX {
            // The cheap cutoff upper bound is only a sufficient test. Before
            // rejecting, let the actual backend certify its adaptive cutoffs
            // and work budget; this expensive probe runs only on this boundary.
            if bessel_peierls_coeffs(&d, drive, harmonic_min, harmonic_max, BESSEL_CUTOFF_MARGIN)
                .is_ok()
            {
                continue;
            }
            return Err(TbError::Other(format!(
                "Floquet drive needs {} samples on a link the Bessel backend cannot \
                 represent exactly (coherent sum ≤ {MAX_BESSEL_ARG_CLOSED_FORM}, convolution \
                 operand ≤ {MAX_BESSEL_ARG} and window ≤ {MAX_BESSEL_WINDOW}); the per-link \
                 grid cap is {FALLBACK_GRID_MAX}",
                needed,
            )));
        }
    }
    Ok(())
}

/// Integer-order Bessel function of the first kind, `J_m(r)`, for real
/// non-negative arguments.
///
/// Thin wrapper over [`puruspe::Jn`] (pure Rust special-functions crate,
/// MIT/Apache-2.0).  Negative orders use the symmetry
///
/// ```math
/// J_{-m}(r) = (-1)^m J_m(r).
/// ```
///
/// It is accurate to a few ulp around `|m| ~ r` — which is where
/// [`bessel_backward_sweep`] anchors its normalization — and over the ranges
/// the tests probe, cross-checked against an independent Miller
/// downward-recurrence reference, tabulated NIST values and `mpmath` at 50
/// digits.  It is *not* a general oracle for `|m| >> r`: measured
/// `Jn(4096, 0.5) = 1.7e34` where the true value underflows to zero, and
/// `Jn(n ≥ 2, r) = 0` for `r ≤ 4.2e-154`.  The ladder therefore never asks it
/// for such orders, and tests that need them build their own reference.
///
/// # Arguments
/// * `m` - integer order (may be negative).
/// * `r` - non-negative finite argument (call sites pass `|a·d|`).
///
/// # Panics
/// Panics on a negative or non-finite argument, both outside the Floquet
/// backend's domain.
pub(crate) fn bessel_j(m: isize, r: f64) -> f64 {
    assert!(
        r.is_finite() && r >= 0.0,
        "bessel_j expects a non-negative finite argument, got {r}"
    );
    if m < 0 {
        // J_{-m}(r) = (-1)^m J_m(r)
        return if m.rem_euclid(2) == 1 {
            -bessel_j(-m, r)
        } else {
            bessel_j(-m, r)
        };
    }
    if r == 0.0 {
        return if m == 0 { 1.0 } else { 0.0 };
    }
    puruspe::Jn(m as u32, r)
}

/// Production tail margin shared by coefficient generation and its validation probe.
pub(super) const BESSEL_CUTOFF_MARGIN: isize = 6;

/// Largest per-mode link amplitude `R_α = |a_α·d|` handled by the one-mode
/// convolution path.  Above it [`bessel_peierls_coeffs`] reports an error and
/// the harmonic cache falls back to a per-link time-grid DFT.
///
/// The cap is a **cost** contract, not a mathematical or accuracy limit: the
/// generalized Bessel expansion converges for every finite `R`, and the
/// backward-recurrence ladder below keeps machine precision far beyond this
/// value (measured `max|ΔJ| ≤ 5e-14` at `R = 4000` against independent
/// `scipy`/`mpmath` references).  It exists because the one-mode convolution
/// costs `O(M·W)` with `M ≈ R` and window `W ≈ 2·Σ_α |l_α|M_α`, while the DFT
/// it replaces costs `O(N·K)` with `N ≈ 2R`: at `R = 128` and a five-bin
/// requested range the former is already ~50x the work, and the ratio grows
/// linearly with `R`.
pub(super) const MAX_BESSEL_ARG: f64 = 128.0;

/// Largest link amplitude `R = |Σ_α a_α·d|` handled by the single-harmonic
/// closed form, i.e. by a drive whose nonzero projections collapse to one
/// coherent harmonic.
///
/// That path allocates no convolution window — one `O(R)` ladder plus the
/// `O(K)` requested bins — so its cost is `O(R + K)` against the DFT's
/// `O(R·K)`, and it reaches two orders of magnitude further than the
/// convolution path at ~25 µs per distinct link.  Above this cap the link
/// falls back to a per-link time grid.  The ladder stays at machine precision
/// across the whole range (checked against 60-digit references).
pub(super) const MAX_BESSEL_ARG_CLOSED_FORM: f64 = 16384.0;

/// Hard cap on the order range evaluated for one link.  A tail that still
/// exceeds its error share here means the amplitude is beyond practical use;
/// the fallback sizing then switches to its conservative analytic bound.
///
/// This covers the whole closed-form range: the sweep for an argument `R` seeds
/// at `⌈R⌉ + 8 + 16·R^{1/3}` and runs to
/// `max(seed, ⌈R⌉) + 20 + √(160·max(seed, ⌈R⌉))` orders, i.e. 18459 at
/// `R = MAX_BESSEL_ARG_CLOSED_FORM` — about half the cap.  It also bounds the
/// ladder the fallback sizing builds to estimate a link's bandwidth.
pub(super) const MAX_BESSEL_ORDER: usize = 1 << 15;

/// Cap on the two-carrier resonance enumeration's inner iterations,
/// `K·(2·min(M₁,M₂)+1)`.  The enumeration scans the *smaller* carrier, so an
/// asymmetric drive stays cheap; this bound keeps a symmetric one from turning a
/// large requested range into quadratic work.  Exceeding it sends the link to
/// the per-link time grid, exactly like an over-large amplitude.
pub(super) const MAX_BESSEL_ENUM_WORK: usize = 1 << 22;

/// Cap on the one-mode convolution window, `harmonic span + 2·Σ_α|l_α|M_α`.
/// The window is allocated as two `Complex<f64>` vectors, so an unguarded
/// high-harmonic drive would abort on allocation instead of returning an error;
/// at this size the pair costs ~64 MB, well above any useful drive (the window
/// is `O(4·n_max + Σ|l_α|·r_α)`).
pub(super) const MAX_BESSEL_WINDOW: usize = 1 << 22;

/// Magnitudes below this are treated as decayed; the ladder extends until its
/// top order is below the floor, which is what makes every suffix tail a
/// complete truncation budget.
pub(super) const BESSEL_DECAY_FLOOR: f64 = 1e-20;

/// Magnitude band the sweep keeps its running values inside.  Unscaled sweep
/// values grow from the seed toward the turning point like
/// `J_⌈r⌉(r)/J_start(r)`, which overflows `f64` for a small argument and a high
/// seed; only the ratios are physical, so an exact power-of-two rescale costs
/// no accuracy.  The upper end must satisfy
/// `SWEEP_BAND_HIGH · 2·start/SWEEP_MIN_ARG < f64::MAX` (`10^30 · 10^254 <<
/// 10^308`), so no value that passes the check can overflow on the next
/// multiplication — checking "did this value overflow?" alone is not enough,
/// because the growth factor itself is unbounded.  The lower end is what stops
/// the rescale from driving *representable* orders to zero: a one-sided upper
/// guard did exactly that for `r ~ 1e-150`, where every rescale multiplied the
/// whole suffix by `2^-1000` while the recurrence grew only `~10^152` per step.
pub(super) const SWEEP_BAND_LOW: f64 = 1e-30;
pub(super) const SWEEP_BAND_HIGH: f64 = 1e30;
/// Arguments below this are evaluated analytically instead of by recurrence:
/// with `k ≤ ~5·10^3` the growth factor `2k/r` then approaches `10^254`, so the
/// band rescale can no longer guarantee the next product stays finite.  At this
/// size `J_0 = 1`, `J_1 = r/2` and every higher order underflows to zero anyway.
pub(super) const SWEEP_MIN_ARG: f64 = 1e-250;
/// Divide every value from `newest` on by one power of two chosen so the
/// **largest** entry of that suffix lands at `~1`.  The factor is exact, so
/// every ratio `u[m]/u[anchor]` — the only thing the normalization reads — is
/// preserved, and no entry the sweep still needs can overflow.  Normalizing on
/// the suffix maximum rather than on its newest entry also covers the decay
/// region, where an older entry can be the largest one.
///
/// Deep-tail entries may still underflow to zero — their `J_m(r)` is
/// negligible there — but the normalization's `anchor` is the turning-point
/// order, which carries the largest magnitude of the finished sweep, so it is
/// never one of them.
pub(super) fn rescale_sweep_tail(u: &mut [f64], newest: usize) {
    let magnitude = u[newest..]
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    debug_assert!(
        magnitude.is_finite() && magnitude > 0.0,
        "rescale_sweep_tail expects a finite non-zero entry"
    );
    // The caller only rescales when the newest magnitude left the band, and
    // `SWEEP_BAND_HIGH` bounds it, so `1023 - exponent` stays a valid biased
    // exponent and the scale is a finite positive power of two.
    let exponent = (magnitude.log2().floor() as i32).clamp(-1000, 1000);
    let scale = f64::from_bits(((1023 - exponent) as u64) << 52);
    for value in u[newest..].iter_mut() {
        *value *= scale;
    }
}

/// Every `J_m(r)` for `m ∈ [0, n]` at one argument, plus the two-sided tail
/// sums `tail[m] = 2·Σ_{k>m} |J_k(r)|`.
///
/// Both consumers of the Bessel backend need a *contiguous* order range at one
/// fixed argument: the Peierls fold sums `B_α(m)` over `m ∈ [-M, M]`, and the
/// adaptive cutoff needs `M+1, M+2, …` until the tail decays.  One backward
/// (Miller) sweep therefore yields the entire object in `O(n)`; requesting
/// each order separately reruns the special-function library's internal
/// recurrence (`O(n)` each) and costs `O(n²)`, and re-summing each tail
/// restarts an overlapping suffix every time.
///
/// # Cost
///
/// `n` grows by doubling until the top order decays, so the sequence of sweeps
/// is geometric and construction is `O(n + √n)` recurrence steps with
/// `n ≤ MAX_BESSEL_ORDER`.  The band rescale adds `O(n)` work per trigger and
/// fires roughly once per `30 / log10(2k/r)` orders.  Because every caller
/// passes `n_min ≈ ⌈r⌉ + margin`, large `r` means a large `n` but a small growth
/// factor and small `r` decays within a handful of orders, so the measured
/// rescale work stays below ~7 multiplies per sweep step (~5 ns/step) across
/// the whole reachable range.  A caller that paired a tiny argument with a huge
/// requested order would make the rescale quadratic (`≤ MAX_BESSEL_ORDER²`
/// multiplies, a few ms); `bessel_ladder_with_cutoff` never does.
pub(super) struct BesselLadder {
    /// `J_0..J_n` at this argument.
    pub(super) j: Vec<f64>,
    /// `tail[m] = 2·Σ_{k>m} |j[k]|`, with `tail[max_order()] = 0`.
    pub(super) tail: Vec<f64>,
    /// Whether the top order fell below [`BESSEL_DECAY_FLOOR`], i.e. whether
    /// the tail sums are complete truncation budgets.  False only when the
    /// sweep hit [`MAX_BESSEL_ORDER`] first, so `decayed == false` always
    /// implies `max_order() == MAX_BESSEL_ORDER`; the caller must then treat
    /// the bandwidth estimate as truncated rather than converged.
    pub(super) decayed: bool,
}

impl BesselLadder {
    /// Sweep a range of at least `n_min` orders, doubling at most a constant
    /// number of times when the initial estimate leaves the top order above the
    /// decay floor (a range that still sits below the turning point cannot show
    /// decay).
    pub(super) fn new(r: f64, n_min: usize) -> Self {
        debug_assert!(r.is_finite() && r > 0.0, "ladder argument must be positive");
        // The top order can only decay past the turning point: `|J_m(r)|` falls
        // below `BESSEL_DECAY_FLOOR` about `13·r^(1/3)` orders beyond it
        // (measured 17, 28, 66 and 199 for r = 1, 8, 128 and 4000).  Starting the
        // estimate there rather than doubling up from the requested order alone
        // roughly halves the number of sweeps; an estimate that is still too
        // short is caught by the doubling below.
        let decay_estimate = (r.ceil() as usize).saturating_add(8 + (16.0 * r.cbrt()) as usize);
        let mut n = n_min.max(decay_estimate).clamp(1, MAX_BESSEL_ORDER);
        loop {
            let j = bessel_backward_sweep(r, n);
            if j[n].abs() < BESSEL_DECAY_FLOOR {
                return Self::from_orders(j, true);
            }
            if n >= MAX_BESSEL_ORDER {
                return Self::from_orders(j, false);
            }
            n = (n * 2).min(MAX_BESSEL_ORDER);
        }
    }

    fn from_orders(j: Vec<f64>, decayed: bool) -> Self {
        let n = j.len() - 1;
        let mut tail = vec![0.0_f64; n + 1];
        let mut suffix = 0.0;
        for k in (0..=n).rev() {
            tail[k] = 2.0 * suffix;
            suffix += j[k].abs();
        }
        Self { j, tail, decayed }
    }

    /// Largest order stored in the ladder.
    #[inline]
    pub(super) fn max_order(&self) -> usize {
        self.j.len() - 1
    }

    /// Two-sided tail `2·Σ_{k>m} |J_k(r)|`, complete when [`Self::decayed`].
    #[inline]
    pub(super) fn tail_after(&self, m: usize) -> f64 {
        self.tail[m]
    }
}

/// Backward (Miller) recurrence returning `J_0..J_n` for one argument.
///
/// The recurrence is stable only in the backward direction, so it is seeded at
/// `N = max(n, ⌈r⌉) + 20 + √(160·max(n, ⌈r⌉))` — above the turning point and
/// high enough for the seed's contamination to decay.  A margin of only a few
/// orders above `r` is not enough: measured `|ΔJ| = 4e-9` at `r = 4000` against
/// `5e-14` with this rule.
///
/// The sweep is normalized on the order with the largest `|J|` (the turning
/// point), not on `J_0`: `J_0` has zeros, and one ulp of absolute error there
/// becomes a relative error for *every* order.  Measured at `r = 128`, where
/// `|J_0|` is 2% of its envelope, normalizing by `J_0` costs 1.3e-13 relative
/// while the turning point stays at 7.8e-16.
pub(super) fn bessel_backward_sweep(r: f64, n: usize) -> Vec<f64> {
    if r < SWEEP_MIN_ARG {
        // The growth factor `2k/r` would overflow before any rescale can act;
        // at this size `J_0 = 1`, `J_1 = r/2` and every higher order underflows.
        let mut j = vec![0.0_f64; n + 1];
        j[0] = 1.0;
        if n >= 1 {
            j[1] = r / 2.0;
        }
        return j;
    }
    let base = n.max(r.ceil() as usize).min(MAX_BESSEL_ORDER);
    let start = base + 20 + (160.0 * base as f64).sqrt() as usize;
    let mut u = vec![0.0_f64; start + 2];
    u[start] = 1.0;
    for k in (1..=start).rev() {
        u[k - 1] = (2.0 * k as f64 / r) * u[k] - u[k + 1];
        let magnitude = u[k - 1].abs();
        // Every written value shares one scale (the sweep only ever reads what
        // it has already written), so rescaling the written suffix keeps all
        // ratios exact.
        if magnitude > 0.0 && !(SWEEP_BAND_LOW..=SWEEP_BAND_HIGH).contains(&magnitude) {
            rescale_sweep_tail(&mut u, k - 1);
        }
    }
    // Anchor on the largest stored magnitude, which for the callers here is the
    // turning point.  Picking an order by index instead would break whenever
    // that order sits on a zero of `J_m` (`J_0` has one at 2.4048…, `J_1` at
    // 7.0156…): the whole ladder would then be scaled by a value whose relative
    // error is enormous.
    let anchor = (0..=n)
        .max_by(|&a, &b| u[a].abs().total_cmp(&u[b].abs()))
        .expect("the sweep always stores at least one order");
    let scale = bessel_j(anchor as isize, r) / u[anchor];
    u.truncate(n + 1);
    for value in u.iter_mut() {
        *value *= scale;
    }
    u
}

/// Adaptive Bessel order cutoff for one link — the smallest
/// `M ≥ ⌈r⌉ + margin` whose two-sided tail `2·Σ_{m>M} |J_m(r)|` fits
/// `error_share` — together with the ladder that backs it.
///
/// `margin` doubles as a higher starting estimate for the time-grid fallback
/// sizing, which passes `0`.  At [`MAX_BESSEL_ORDER`] the cutoff saturates and
/// the returned ladder reports `decayed == false`, so the caller can tell a
/// converged bandwidth estimate from a truncated one.
pub(super) fn bessel_ladder_with_cutoff(
    r: f64,
    error_share: f64,
    margin: isize,
) -> (BesselLadder, isize) {
    debug_assert!(
        r.is_finite() && r > 0.0,
        "bessel_ladder_with_cutoff: r must be finite and positive"
    );
    // The float-to-int cast saturates for huge r; saturating_add keeps the
    // +margin step overflow-free before the clamp.
    let m_start = (r.ceil() as isize)
        .saturating_add(margin)
        .clamp(0, MAX_BESSEL_ORDER as isize) as usize;
    let mut ladder = BesselLadder::new(r, m_start);
    loop {
        // The scan stops *before* `max_order`, whose suffix sum is zero by
        // construction: accepting it would report a truncated (non-decayed)
        // ladder as a converged cutoff.  A decayed ladder always fits some
        // `m <= max_order - 1`, because `tail[max_order - 1] = 2|J_top| <
        // 2·BESSEL_DECAY_FLOOR` is far below any per-mode error share.
        let found = (m_start..ladder.max_order()).find(|&m| ladder.tail_after(m) <= error_share);
        if let Some(m) = found {
            return (ladder, m as isize);
        }
        if ladder.max_order() >= MAX_BESSEL_ORDER {
            return (ladder, MAX_BESSEL_ORDER as isize);
        }
        ladder = BesselLadder::new(r, ladder.max_order() + 1);
    }
}

/// Ladder and adaptive cutoff for one link amplitude, with the validation both
/// coefficient paths need.  `max_arg` is the cap of the calling path:
/// [`MAX_BESSEL_ARG_CLOSED_FORM`] for the closed form, whose cost is linear in
/// `R`, and [`MAX_BESSEL_ARG`] for a convolution operand.
pub(super) fn bessel_ladder_for_amplitude(
    r: f64,
    error_share: f64,
    cutoff_margin: isize,
    max_arg: f64,
) -> Result<(BesselLadder, isize)> {
    if !r.is_finite() || r > max_arg {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: mode amplitude R = {r:.3} is outside the \
             Bessel backend's range (R ≤ {max_arg}); use the time-grid backend"
        )));
    }
    // Adaptive cutoff plus the ladder that backs it: the smallest M whose
    // two-sided tail 2 * Σ_{m>M} |J_m(r)| fits the per-mode error share,
    // delivered together with J_0..J_M and every candidate tail from one
    // backward sweep.
    let (ladder, m_cap) = bessel_ladder_with_cutoff(r, error_share, cutoff_margin);
    if !ladder.decayed {
        // A truncated ladder's suffix sums stop at MAX_BESSEL_ORDER, so they
        // cannot certify a truncation budget.  The closed-form cap plus the largest
        // allowed margin keeps this out of reach, but a future cap change must
        // fail loudly instead of folding a truncated expansion.
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: the order range for R = {r:.3} saturated at \
             {MAX_BESSEL_ORDER} orders without decaying; lower MAX_BESSEL_ARG or \
             raise MAX_BESSEL_ORDER"
        )));
    }
    Ok((ladder, m_cap))
}

/// Closed form for a drive whose nonzero temporal harmonics collapse to one
/// value `l`.
///
/// The resonance condition `l·m = -n` then fixes the multi-index sum to a single
/// term,
///
/// ```math
/// C_n = (-i)^m J_m(R)\, e^{-im\delta},\qquad l \mid n,
/// ```
///
/// and every other bin vanishes: one lookup per requested order instead of the
/// `O(M·W)` one-mode convolution.  Orders past the adaptive cutoff `M` stay
/// zero, which is exactly what the convolution returns once its Bessel orders
/// are truncated there.  DC modes only multiply the result by
/// `exp(-i Re(a_0·d))`, the static Peierls phase.
///
/// Returns `Ok(None)` when the drive has two or more distinct nonzero harmonics
/// — coherent sums of equal harmonics count once — because that sum genuinely
/// needs the convolution.
fn single_harmonic_closed_form(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    cutoff_margin: isize,
) -> Result<Option<Array1<Complex<f64>>>> {
    let mut single: Option<(isize, Complex<f64>)> = None;
    let mut dc = 0.0_f64;
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            // Never form the unused imaginary projection: it may overflow even
            // when the static real field is identically zero.
            dc += mode
                .a_complex
                .iter()
                .zip(d)
                .map(|(a, x)| a.re * x)
                .sum::<f64>();
            continue;
        }
        // Mode projection onto the link: z = a·d = R e^{iδ}.
        let z: Complex<f64> = mode
            .a_complex
            .iter()
            .zip(d.iter())
            .map(|(a, x)| *a * *x)
            .sum();
        if z.re == 0.0 && z.im == 0.0 {
            continue;
        }
        match &mut single {
            Some((harmonic, total)) if *harmonic == mode.harmonic => *total += z,
            Some(_) => return Ok(None),
            None => single = Some((mode.harmonic, z)),
        }
    }
    if !dc.is_finite() {
        return Err(TbError::Other(
            "bessel_peierls_coeffs: the static phase is not finite".into(),
        ));
    }
    let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
    let dc_phase = Complex::new(0.0, -dc).exp();
    // Equal harmonics are summed coherently, so the total can cancel exactly;
    // such a drive dresses nothing, exactly like a single zero projection, and
    // must not reach the ladder (whose argument is asserted positive).
    let single = single.filter(|(_, z)| z.re != 0.0 || z.im != 0.0);
    let Some((harmonic, z)) = single else {
        // Purely static drive: the Peierls phase shifts the on-site term only.
        let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = dc_phase;
        }
        return Ok(Some(coeffs));
    };

    let r = z.norm();
    // Same per-mode share as the convolution path, so the adaptive cutoff — and
    // therefore every truncated bin — agrees exactly.
    let error_share = 1e-12 / (drive.modes.len() as f64);
    let (ladder, m_cap) =
        bessel_ladder_for_amplitude(r, error_share, cutoff_margin, MAX_BESSEL_ARG_CLOSED_FORM)?;
    let delta = z.arg();
    let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
    for (index, n) in (harmonic_min..=harmonic_max).enumerate() {
        let n = n as i128;
        let harmonic = harmonic as i128;
        if n % harmonic != 0 {
            continue;
        }
        // Widen before negation/division: MIN / -1 must not panic, and only
        // an index inside the bounded ladder is converted back to isize.
        let m = -n / harmonic;
        let order = m.unsigned_abs();
        if order > m_cap as u128 {
            continue;
        }
        let order = order as usize;
        let m = m as isize;
        let j_m = if m < 0 && order % 2 == 1 {
            -ladder.j[order]
        } else {
            ladder.j[order]
        };
        coeffs[index] = dc_phase
            * Complex::new(0.0, -1.0).powi(m as i32)
            * j_m
            * Complex::from_polar(1.0, -(m as f64) * delta);
    }
    Ok(Some(coeffs))
}

/// Peierls Fourier coefficients `C_n(d)` via the generalized Bessel
/// expansion, for `n ∈ [harmonic_min, harmonic_max]`.
///
/// For a drive `a(t) = Re Σ_α a_α e^{−i l_α Ω₀ t}` each mode contributes a
/// scalar pair `z_α = a_α·d = R_α e^{iδ_α}` per link displacement `d`, and
/// the Jacobi–Anger expansion of the factorized Peierls exponential gives
///
/// ```math
/// C_n(d) = \sum_{\{m_α\} : Σ_α l_α m_α = -n}
///          \prod_α (-i)^{m_α} J_{m_α}(R_α)\, e^{-i m_α δ_α}.
/// ```
///
/// (Resonance `n + Σ l m = 0`; the equivalent form `Σ l m = +n` with phase
/// `e^{+imδ}` must not be mixed in.)  A drive whose nonzero harmonics collapse
/// to one value needs no sum at all: [`single_harmonic_closed_form`] returns one
/// term per bin.  Everything else evaluates the multi-index sum as a sequence
/// of one-mode discrete convolutions
///
/// ```math
/// S^{(0)}_n = δ_{n,0},\qquad
/// S^{(α)}_n = \sum_{m=-M_α}^{M_α} S^{(α-1)}_{n + l_α m}\, B_α(m),
/// \qquad
/// B_α(m) = (-i)^m J_m(R_α)\, e^{-imδ_α},
/// ```
///
/// which costs `O(N_mode · N_n · M_avg)` — independent of the time-grid
/// size, `N_n` being the requested harmonic range.
///
/// Two carriers are the common multi-colour case and are evaluated differently:
/// the constraint `l₁m₁ + l₂m₂ = −n` leaves one free index, so the same sum is
/// enumerated directly at `O(N_n · (2·min(M₁,M₂) + 1))` — linear in the
/// amplitude, with no working window and therefore no BLAS call per requested
/// bin.  That path admits amplitudes up to [`MAX_BESSEL_ARG_CLOSED_FORM`] and is
/// bounded by [`MAX_BESSEL_ENUM_WORK`]; three or more carriers fall back to the
/// convolution above, which keeps the lower [`MAX_BESSEL_ARG`] cap and the
/// [`MAX_BESSEL_WINDOW`] bound.
///
/// Each mode's cutoff `M_α` is chosen adaptively so the truncated
/// tail `Σ_{|m|>M_α} |J_m(R_α)|` stays below a per-mode error share
/// (`1e-12 / N_mode`), with `cutoff_margin` as an additional minimum, and
/// it must stay at or below [`MAX_BESSEL_ORDER`].
///
/// Each mode needs `J_0..J_{M_α}` **and** every tail `Σ_{m>M}|J_m(R_α)|`;
/// both come from one [`BesselLadder`] per mode, i.e. a single backward
/// recurrence sweep plus one suffix sum.  In particular the adaptive search
/// never evaluates an order on its own, so its cost is linear in the order
/// range instead of quadratic.
///
/// Verified against the independent time-grid DFT
/// (`crate::floquet_test::peierls_fourier_coeffs`) for linear, circular,
/// elliptical, and multi-harmonic drives to ~1e-15.
///
/// # Arguments
/// * `d` - real link displacement (Cartesian, length `DIM`).
/// * `drive` - the light drive (modes `(l_α, a_α)`, base frequency `Ω₀`).
/// * `harmonic_min`, `harmonic_max` - inclusive harmonic range to return.
/// * `cutoff_margin` - minimum number of Bessel orders beyond `⌈R_α⌉`,
///   in `0..=48` (the adaptive tail check may push `M_α` higher; only
///   lower bounded by this).
///
/// # Returns
/// `C_n(d)` for `n = harmonic_min..=harmonic_max` as an [`Array1<Complex<f64>>`] of
/// length `harmonic_max - harmonic_min + 1`.
///
/// # Errors
/// Returns [`TbError::Other`] when `harmonic_min > harmonic_max`, when `cutoff_margin`
/// is outside `0..=48`, when a carrier amplitude exceeds the branch's cap
/// ([MAX_BESSEL_ARG_CLOSED_FORM] for one carrier (the closed form) and for two
/// (the enumeration), [MAX_BESSEL_ARG] for the convolution), when the
/// enumeration's work budget or the convolution's window bound is exceeded (the
/// caller must fall back to the time-grid backend), or when the harmonic range
/// or working window would overflow `isize`.
pub(crate) fn bessel_peierls_coeffs(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    cutoff_margin: isize,
) -> Result<Array1<Complex<f64>>> {
    if harmonic_min > harmonic_max {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: empty harmonic range [{harmonic_min}, {harmonic_max}]"
        )));
    }
    if !(0..=48).contains(&cutoff_margin) {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: cutoff_margin = {cutoff_margin} outside [0, 48]"
        )));
    }
    // harmonic_max >= harmonic_min here, so the span is non-negative; checked_sub guards
    // the isize::MIN..=isize::MAX range against overflow.
    let harmonic_count = harmonic_max.checked_sub(harmonic_min).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: harmonic range too wide".to_string())
    })? as usize
        + 1;
    // Empty drive: the Peierls exponential is 1, so only C_0 survives.
    if drive.modes.is_empty() {
        let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = Complex::new(1.0, 0.0);
        }
        return Ok(coeffs);
    }

    // A drive whose nonzero harmonics collapse to one value needs no
    // convolution: the resonance condition leaves exactly one term per bin.
    if let Some(coeffs) =
        single_harmonic_closed_form(d, drive, harmonic_min, harmonic_max, cutoff_margin)?
    {
        return Ok(coeffs);
    }

    // Certify exact physical cancellation after link projection, not by summing
    // Cartesian amplitudes. Separate truncated Bessel products for opposite
    // carriers otherwise leave spurious nonzero edge harmonics for a zero field.
    let (dc, carriers, exact_sum) = time_grid_projections(drive, d);
    if carriers.is_empty() && exact_sum {
        if !dc.is_finite() {
            return Err(TbError::Other(
                "bessel_peierls_coeffs: the static phase is not finite".into(),
            ));
        }
        let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = Complex::new(0.0, -dc).exp();
        }
        return Ok(coeffs);
    }

    // Two-pass construction.  First pass: per-mode projections and adaptive
    // cutoffs.  The convolution only supports R_α ≤ MAX_BESSEL_ARG; the caller falls back
    // to the time-grid backend beyond that.
    struct ModeData {
        /// `J_0..J_{M_α}` and the tail sums behind the adaptive cutoff.
        ladder: BesselLadder,
        /// Link projection `z_α = a_α·d = R_α e^{iδ_α}`.
        projection: Complex<f64>,
        harmonic: isize,
        m_cap: isize,
    }
    let mut modes = Vec::<ModeData>::with_capacity(drive.modes.len());
    let error_share = 1e-12 / (drive.modes.len() as f64);
    let mut dc_re = 0.0_f64;
    // Two carriers are enumerated, and that path is linear in the amplitude
    // instead of quadratic, so it can reach as far as the closed form.  The
    // count is taken over the drive's harmonics up front; a mode whose link
    // projection vanishes is dropped later and only makes this more permissive,
    // which the budget check below then catches.
    let two_carriers = drive.modes.iter().filter(|mode| mode.harmonic != 0).count() == 2;
    let mode_cap = if two_carriers {
        MAX_BESSEL_ARG_CLOSED_FORM
    } else {
        MAX_BESSEL_ARG
    };
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            dc_re += mode
                .a_complex
                .iter()
                .zip(d)
                .map(|(a, x)| a.re * x)
                .sum::<f64>();
            continue;
        }
        // Mode projection onto the link: z = a·d = R e^{iδ}.
        let z: Complex<f64> = mode
            .a_complex
            .iter()
            .zip(d.iter())
            .map(|(a, d)| *a * *d)
            .sum();
        let r = z.norm();
        if r == 0.0 {
            // Degenerate mode: only m = 0 contributes (B = 1), a no-op fold.
            continue;
        }
        let (ladder, m_cap) = bessel_ladder_for_amplitude(r, error_share, cutoff_margin, mode_cap)?;
        modes.push(ModeData {
            ladder,
            projection: z,
            harmonic: mode.harmonic,
            m_cap,
        });
    }
    if !dc_re.is_finite() {
        // Two finite static projections can still sum past f64::MAX; the phase
        // would then be NaN and every coefficient with it.
        return Err(TbError::Other(
            "bessel_peierls_coeffs: the static phase is not finite".to_string(),
        ));
    }
    // The working window only concerns the convolution, and its drift is
    // computed after the enumeration has had its chance: an enumeration never
    // allocates a window, so an overflowing |l|·M there is not an error.
    let drift_for_fold = |modes: &[ModeData]| -> Result<isize> {
        let mut total = 0_isize;
        for mode in modes {
            let harmonic_abs = mode.harmonic.checked_abs().ok_or_else(|| {
                TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
            })?;
            total = total
                .checked_add(harmonic_abs.checked_mul(mode.m_cap).ok_or_else(|| {
                    TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
                })?)
                .ok_or_else(|| {
                    TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
                })?;
        }
        Ok(total)
    };

    // Two nonzero harmonics are the common multi-colour case (a fundamental plus
    // a harmonic, or two commensurate colours).  The resonance condition
    // l₁m₁ + l₂m₂ = −n leaves one free index, so those coefficients can be
    // enumerated directly in O(K·(2M+1)) instead of folding a window of width
    // span + 2·Σ|l|M — one BLAS call per requested bin dominated that fold's
    // measured cost.  A static mode contributes the pure phase e^{-i Re z}.
    let carriers: Vec<&ModeData> = modes.iter().filter(|mode| mode.harmonic != 0).collect();
    // Both counts must agree: `mode_cap` was chosen from the drive-level count,
    // so a link that drops one of those carriers (a zero projection) must not
    // silently take this path with the higher cap.
    if two_carriers && carriers.len() == 2 {
        let requested = (harmonic_max - harmonic_min + 1) as usize;
        let scan_len = 2 * carriers[0].m_cap.min(carriers[1].m_cap) as usize + 1;
        let work = requested.saturating_mul(scan_len);
        if work > MAX_BESSEL_ENUM_WORK {
            return Err(TbError::Other(format!(
                "bessel_peierls_coeffs: the two-carrier enumeration would need \
                 {work} iterations for {requested} requested orders, above the \
                 {MAX_BESSEL_ENUM_WORK} cap; lower the amplitude or the harmonic range"
            )));
        }
        let dc_phase = Complex::new(0.0, -dc_re).exp();
        // Iterate over the mode with the smaller cutoff and divmod by the other's
        // harmonic: only m pairs whose partner index is an integer inside its own
        // cutoff contribute.
        let (scan, other) = if carriers[0].m_cap <= carriers[1].m_cap {
            (carriers[0], carriers[1])
        } else {
            (carriers[1], carriers[0])
        };
        let sequence = |mode: &ModeData| -> Vec<Complex<f64>> {
            let delta = mode.projection.arg();
            let mut out = Vec::with_capacity((2 * mode.m_cap + 1) as usize);
            let mut minus_i_power = Complex::new(1.0, 0.0);
            for m in 0..=mode.m_cap {
                let j_m = mode.ladder.j[m as usize];
                let value = minus_i_power * j_m * Complex::from_polar(1.0, -(m as f64) * delta);
                if m == 0 {
                    out.push(value);
                } else {
                    let neg = minus_i_power * j_m * Complex::from_polar(1.0, (m as f64) * delta);
                    out.push(neg);
                    out.push(value);
                }
                minus_i_power *= Complex::new(0.0, -1.0);
            }
            out
        };
        let scan_seq = sequence(scan);
        let other_seq = sequence(other);
        // Both vectors follow the fold's layout: index 0 is m = 0, then the pairs
        // (m = -1, m = +1), (m = -2, m = +2), ... so m > 0 sits at 2m and m < 0
        // at 2|m| - 1.
        let slot = |m: isize| -> usize {
            if m >= 0 {
                (2 * m) as usize
            } else {
                (2 * m.unsigned_abs() - 1) as usize
            }
        };
        let mut coeffs = Array1::<Complex<f64>>::zeros((harmonic_max - harmonic_min + 1) as usize);
        for (index, n) in (harmonic_min..=harmonic_max).enumerate() {
            let target = -(n as i128);
            let mut acc = Complex::new(0.0, 0.0);
            for m in -scan.m_cap..=scan.m_cap {
                // Huge products can cancel to a small requested order. isize
                // harmonics times the bounded Bessel cutoff fit in i128, including
                // the subtraction, division and absolute value at either limit.
                let numerator = target - (scan.harmonic as i128) * (m as i128);
                let other_harmonic = other.harmonic as i128;
                if numerator % other_harmonic != 0 {
                    continue;
                }
                let partner = numerator / other_harmonic;
                if partner.abs() > other.m_cap as i128 {
                    continue;
                }
                acc += scan_seq[slot(m)] * other_seq[slot(partner as isize)];
            }
            coeffs[index] = acc * dc_phase;
        }
        return Ok(coeffs);
    }

    // Second pass: the working window must cover the actual reachable
    // support [−drift, +drift] around [harmonic_min, harmonic_max], because intermediates
    // outside the requested range can fold back into it.
    let total_drift = drift_for_fold(&modes)?;
    let work_min = harmonic_min.checked_sub(total_drift).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: working window underflow".to_string())
    })?;
    let work_max = harmonic_max.checked_add(total_drift).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: working window overflow".to_string())
    })?;
    // work_max >= work_min by construction (harmonic_max >= harmonic_min, drift >= 0), so
    // the span is non-negative; checked_sub and usize::try_from are kept
    // for hygiene.
    let work_span = work_max.checked_sub(work_min).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: working window span overflow".to_string())
    })?;
    let work_len = usize::try_from(work_span).map_err(|_| {
        TbError::Other("bessel_peierls_coeffs: working window too large".to_string())
    })? + 1;
    if work_len > MAX_BESSEL_WINDOW {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: the one-mode convolution window needs {work_len} entries, \
             above the {MAX_BESSEL_WINDOW} cap; lower the drive harmonics or the amplitudes"
        )));
    }

    let requested = (harmonic_min - work_min) as usize..(harmonic_max - work_min + 1) as usize;
    let mut sequence = vec![Complex::new(0.0, 0.0); work_len];
    let mut live = 0..0;
    if let Some(zero_index) = work_min.checked_neg()
        && (0_isize..work_len as isize).contains(&zero_index)
    {
        sequence[zero_index as usize] = Complex::new(1.0, 0.0);
        live = zero_index as usize..zero_index as usize + 1;
    }
    let mut next = vec![Complex::new(0.0, 0.0); work_len];
    let mut remaining_drift = total_drift as usize;

    for mode in &modes {
        if live.is_empty() {
            break;
        }
        // Keep only indices reachable from this prefix that can still reach a
        // requested bin through the remaining modes. This is exact structural
        // support pruning, not a small-value cutoff. The original full-window
        // overflow/resource checks above still apply before any pruning.
        let mode_drift = mode.harmonic.unsigned_abs() * mode.m_cap as usize;
        remaining_drift -= mode_drift;
        let target = live
            .start
            .saturating_sub(mode_drift)
            .max(requested.start.saturating_sub(remaining_drift))
            ..(live.end + mode_drift)
                .min(work_len)
                .min(requested.end + remaining_drift);
        next.fill(Complex::new(0.0, 0.0));
        if target.is_empty() {
            std::mem::swap(&mut sequence, &mut next);
            break;
        }

        // One-mode sequence B(m) = (-i)^m J_m(r) e^{-imδ}, m ∈ [-M, M].
        // Accumulate (-i)^m iteratively.
        let mut minus_i_power = Complex::new(1.0, 0.0); // (-i)^0
        // The cutoff is at most the ladder length (the scan stops before the
        // ladder's top entry), so every lookup below is in range; the ladder
        // itself is capped at MAX_BESSEL_ORDER.
        debug_assert!(mode.m_cap >= 0 && mode.m_cap as usize <= mode.ladder.max_order());
        let delta = mode.projection.arg();
        let mut b = Vec::<(isize, Complex<f64>)>::with_capacity((2 * mode.m_cap + 1) as usize);
        for m in 0..=mode.m_cap {
            let j_m = mode.ladder.j[m as usize];
            let value = minus_i_power * j_m * Complex::from_polar(1.0, -(m as f64) * delta);
            if m == 0 {
                b.push((0, value));
            } else {
                // B(-m) = (-i)^{-m} J_{-m}(r) e^{+imδ}
                //       = i^m · (-1)^m J_m(r) e^{+imδ}
                //       = (-i)^m J_m(r) e^{+imδ} (since i^m (-1)^m = (-i)^m)
                let neg = minus_i_power * j_m * Complex::from_polar(1.0, (m as f64) * delta);
                b.push((-m, neg));
                b.push((m, value));
            }
            minus_i_power *= Complex::new(0.0, -1.0); // times (-i)
        }

        // Fold: S'_n = Σ_m S_{n + l·m} B(m). Intersect shifted live
        // intervals once per m, then accumulate contiguous slices without
        // per-element index arithmetic. Preserve the original m/sum order.
        for &(m, weight) in &b {
            // drift_for_fold checked |harmonic| * m_cap and their sum;
            // |m| <= m_cap, so this signed product is representable. The
            // accepted window also bounds every index addition below.
            let shift = mode.harmonic * m;
            let offset = shift.unsigned_abs();
            let (source, destination) = if shift >= 0 {
                let start = target.start.max(live.start.saturating_sub(offset));
                let end = target.end.min(live.end.saturating_sub(offset));
                if start >= end {
                    continue;
                }
                (
                    &sequence[start + offset..end + offset],
                    &mut next[start..end],
                )
            } else {
                let start = target.start.max(live.start + offset);
                let end = target.end.min(live.end + offset);
                if start >= end {
                    continue;
                }
                (
                    &sequence[start - offset..end - offset],
                    &mut next[start..end],
                )
            };
            for (output, &value) in destination.iter_mut().zip(source) {
                *output += value * weight;
            }
        }
        std::mem::swap(&mut sequence, &mut next);
        live = target;
    }

    // Static modes were folded into one phase above, exactly as the closed form
    // and the enumeration do.
    let dc_phase = Complex::new(0.0, -dc_re).exp();
    Ok(Array1::from(
        sequence[requested]
            .iter()
            .map(|coeff| coeff * dc_phase)
            .collect::<Vec<_>>(),
    ))
}

/// Maximum number of time points for a per-link fallback DFT.
pub(super) const FALLBACK_GRID_MAX: usize = 1 << 20;

/// Grid-size decision for a fallback link (see [`fallback_time_grid_coeffs`]).
pub(super) struct FallbackGridSize {
    /// Alias-free grid size to evaluate the DFT at.
    pub(super) n_req: usize,
    /// Signal-bandwidth Nyquist term `2·Σ_α |l_α|·M_α + 4`.
    pub(super) required: usize,
    /// Requested-range Nyquist term `2·max(|harmonic_min|, |harmonic_max|) + 1`.
    pub(super) request_range: usize,
    /// The unclamped size exceeded [`FALLBACK_GRID_MAX`].
    pub(super) clamped: bool,
    /// A mode's adaptive cutoff saturated at [`MAX_BESSEL_ORDER`] orders,
    /// requiring a conservative analytic tail bound instead.
    pub(super) saturated: bool,
}

// Project before time evolution, sum coherent carriers (including l/-l),
// and keep DC outside the oscillating phase. Sizing and sampling must use
// the same scalar signal, not separately rounded Cartesian fields.
pub(super) fn time_grid_projections(
    drive: &FloquetDrive,
    d: &Array1<f64>,
) -> (f64, std::collections::BTreeMap<usize, Complex<f64>>, bool) {
    let mut dc = 0.0;
    let mut carriers = std::collections::BTreeMap::new();
    let mut exact_sum = true;
    // TwoSum detects discarded rounding without changing the computed signal.
    // A rounded zero alone must not certify the new exact-static shortcut.
    let addition_is_exact = |a: f64, b: f64, sum: f64| {
        let recovered_b = sum - a;
        (a - (sum - recovered_b)) + (b - recovered_b) == 0.0
    };
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            dc += mode
                .a_complex
                .iter()
                .zip(d)
                .map(|(a, x)| a.re * x)
                .sum::<f64>();
            continue;
        }
        let z: Complex<f64> = mode.a_complex.iter().zip(d).map(|(a, x)| a * x).sum();
        let z = if mode.harmonic > 0 { z } else { z.conj() };
        let total = carriers
            .entry(mode.harmonic.unsigned_abs())
            .or_insert(Complex::new(0.0, 0.0));
        let next = *total + z;
        exact_sum &= addition_is_exact(total.re, z.re, next.re)
            && addition_is_exact(total.im, z.im, next.im);
        *total = next;
    }
    carriers.retain(|_, z| z.re != 0.0 || z.im != 0.0);
    (dc, carriers, exact_sum)
}

/// Size the alias-free grid from the coherent physical carriers after projection:
/// `max(2·Σ |l|·M + 4, 2·max(|harmonic_min|, |harmonic_max|) + 1)`.
/// Keep the unclamped requirement so public validation can reject aliasing.
pub(super) fn fallback_grid_size(
    drive: &FloquetDrive,
    d: &Array1<f64>,
    harmonic_min: isize,
    harmonic_max: isize,
) -> FallbackGridSize {
    // Nyquist bandwidth of the Peierls exponential on this link.
    let mut bandwidth = 0_usize;
    let mut saturated = false;
    let mode_count = drive.modes.len().max(1);
    let error_share = 1e-12 / mode_count as f64;
    let (_, carriers, _) = time_grid_projections(drive, d);
    for (harmonic, z) in carriers {
        let r = z.norm();
        if !r.is_finite() {
            // An unrepresentable physical carrier has unbounded required work,
            // not zero bandwidth. Public validation must refuse this fallback.
            bandwidth = usize::MAX;
            break;
        }
        // The Bessel backend's precision margin is not a sampling floor.
        // A ladder that never decayed inside MAX_BESSEL_ORDER cannot certify a
        // bandwidth: its suffix sums stop at the cap, so a cutoff read from them
        // omits the orders above it — measured at r = 32483 a suffix budget of
        // 9.5e-13 against a complete two-sided tail of 1.9e-12 for a 1e-12
        // share.  Such an argument therefore always takes the conservative
        // analytic bound below, exactly like r > MAX_BESSEL_ORDER.
        let (m_cap, truncated) = if r > MAX_BESSEL_ORDER as f64 {
            (MAX_BESSEL_ORDER as isize, true)
        } else {
            let (ladder, m_cap) = bessel_ladder_with_cutoff(r, error_share, 0);
            (m_cap, !ladder.decayed)
        };
        let cutoff = if truncated {
            saturated = true;
            // For m >= 3r, |J_m(r)| <= (r/2)^m/m! <= (e/6)^m < 2^-m
            // (DLMF 10.14.4). Thus the two-sided tail is <= 2^(1-M).
            // Split its budget across modes. Float casts and all grid-size
            // arithmetic saturate; the explicit requirement is never clamped.
            let tail_floor = 44 + (usize::BITS - mode_count.leading_zeros()) as usize;
            ((3.0 * r).ceil() as usize).max(tail_floor)
        } else {
            m_cap as usize
        };
        let drift = harmonic.saturating_mul(cutoff);
        bandwidth = bandwidth.saturating_add(drift);
    }
    let required = bandwidth.saturating_mul(2).saturating_add(4);
    let harmonic_max_abs = harmonic_min.unsigned_abs().max(harmonic_max.unsigned_abs());
    let request_range = (2_usize).saturating_mul(harmonic_max_abs).saturating_add(1);
    let mut n_req = required.max(request_range);
    let mut clamped = false;
    if n_req > FALLBACK_GRID_MAX {
        n_req = FALLBACK_GRID_MAX;
        clamped = true;
    }
    if saturated {
        // Automatic fallback retains its maximum-grid policy; explicit Sambe
        // grids are checked against the unclamped analytic requirement above.
        n_req = FALLBACK_GRID_MAX;
    }
    FallbackGridSize {
        n_req,
        required,
        request_range,
        clamped,
        saturated,
    }
}

/// Time-grid fallback for links outside the Bessel backend's range.
///
/// The fallback sizes its own grid from the link's spectral bandwidth
/// instead of relying on the caller's `n_time`: a link's Peierls
/// exponential has spectral content up to `Σ_α |l_α|·M_α(R_α)` (with `M_α`
/// the adaptive tail cutoff), and a coarse grid aliases it silently — e.g.
/// for a single `l = 100` mode at `R = 50`, a 512-point grid puts
/// `C_−8 ≈ −J_46(50) ≈ −0.17` where the true value is `0`
/// (`C_0 = J_0(50) = 0.0558` survives there only by a divisibility
/// coincidence).  The DFT is evaluated directly at the size chosen by
/// [`fallback_grid_size`], clamped to [`FALLBACK_GRID_MAX`] = 2^20 points
/// (beyond that the drive or truncation is pathological; accuracy degrades
/// and a warn-once message is printed).  When the adaptive bandwidth
/// estimate saturates at its [`MAX_BESSEL_ORDER`]-order cap — the ladder
/// reaches the cap without decaying, which for a single mode starts around
/// `R ≈ 32483` at the `1e-12` budget — sizing switches to a conservative
/// analytic tail bound.
/// The automatic grid uses the maximum size and prints a warn-once message;
/// it can still be too small when the analytic requirement exceeds this cap.
pub(super) fn fallback_time_grid_coeffs(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    clamped: &AtomicBool,
    saturated: &AtomicBool,
) -> Array1<Complex<f64>> {
    let size = fallback_grid_size(drive, d, harmonic_min, harmonic_max);
    if size.clamped && !clamped.swap(true, Ordering::Relaxed) {
        eprintln!(
            "Floquet fallback grid clamped to {FALLBACK_GRID_MAX} time points \
             (link requires {required}, requested harmonic range {request_range}); \
             coefficients on this link may be inaccurate",
            required = size.required,
            request_range = size.request_range,
        );
    }
    if size.saturated && !saturated.swap(true, Ordering::Relaxed) {
        // Warn once that alias-freedom is no longer guaranteed.
        eprintln!(
            "Floquet fallback bandwidth estimate saturated at {MAX_BESSEL_ORDER} Bessel \
             orders; using the maximum {FALLBACK_GRID_MAX}-point grid — \
             coefficients on this link may still be inaccurate for extreme \
             amplitudes"
        );
    }
    let n_req = size.n_req;
    // Direct DFT at the fine resolution (no shared harmonic_count × n_time
    // Fourier matrix to reuse).
    let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
    let inv_n = 1.0 / (n_req as f64);
    let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
    let (dc, carriers, _) = time_grid_projections(drive, d);
    let dc_phase = Complex::new(0.0, -dc).exp();
    for it in 0..n_req {
        let theta = TAU * (it as f64) * inv_n;
        let mut link_phase = 0.0;
        for (&harmonic, &z) in &carriers {
            let harmonic_phase = Complex::new(0.0, -(harmonic as f64) * theta).exp();
            link_phase += (z * harmonic_phase).re;
        }
        let peierls = dc_phase * Complex::new(0.0, -link_phase).exp();
        for (i_n, n) in (harmonic_min..=harmonic_max).enumerate() {
            coeffs[i_n] += Complex::new(0.0, (n as f64) * theta).exp() * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= inv_n;
    }
    Array1::from(coeffs)
}
