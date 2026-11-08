use super::*;
use crate::SpinDirection;
use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
use crate::floquet_test::{FloquetTimeGrid, peierls_fourier_coeffs};
use crate::model::NoRMatrix;

use crate::solve_ham::Solve;
use ndarray::{arr1, array};

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    // Keep the range/phase fixtures concise while exercising the public
    // real-space constructor, not the independent k-space reference.
    fn sambe_ham_from_model<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array2<Complex<f64>>> {
        validate_floquet_input(self, kvec, drive, trunc)?;
        Ok(self.floquet_model(drive, trunc)?.gen_ham(kvec, gauge))
    }
}

#[test]
fn bessel_j_matches_tabulated_values() {
    // Reference values (NIST DLMF, mpmath 50-digit evaluation).
    let table: [(isize, f64, f64); 12] = [
        (0, 0.0, 1.0),
        (1, 0.0, 0.0),
        (5, 0.0, 0.0),
        (0, 1.0, 0.76519768655796655145),
        (1, 1.0, 0.44005058574493351596),
        (5, 1.0, 0.00024975773021123443176),
        (0, 2.0, 0.22389077914123566805),
        (1, 2.0, 0.57672480775687338720),
        (2, 2.0, 0.35283402861563771915),
        (3, 2.0, 0.12894324947440205110),
        (0, 5.0, -0.17759677131433830435),
        (10, 5.0, 0.00146780264731047436),
    ];
    for (m, r, expected) in table {
        let got = bessel_j(m, r);
        assert!(
            (got - expected).abs() < 1e-14,
            "J_{m}({r}) = {got}, expected {expected}"
        );
    }
}

#[test]
fn bessel_j_matches_independent_miller_reference() {
    // The planned Bessel backend operates up to r = 8 with orders up to
    // ~r + 16.  Cross-check the ascending series against an INDEPENDENT
    // algorithm: Miller's downward recurrence, normalized by the
    // identity J_0(r) + 2*sum_k J_{2k}(r) = 1 (stable in the direction
    // the series is not).
    let miller = |r: f64, mmax: usize| -> Vec<f64> {
        let start = mmax + 20;
        let mut next = 0.0_f64; // J_{start+1} ~ 0
        let mut current = 1.0_f64; // J_start (unscaled)
        let mut values = vec![0.0_f64; start + 1];
        for k in (0..start).rev() {
            values[k] = current;
            let k_prev = k as f64;
            // J_{k-1} = (2k/r) J_k - J_{k+1}
            let prev = if k == 0 {
                0.0
            } else {
                (2.0 * k_prev / r) * current - next
            };
            next = current;
            current = prev;
        }
        // Normalize: J_0(r) + 2 sum_{k>=1} J_{2k}(r) = 1.
        let j0_unscaled = values[0];
        let even_sum: f64 = values.iter().step_by(2).skip(1).sum::<f64>();
        let scale = 1.0 / (j0_unscaled + 2.0 * even_sum);
        values.iter().map(|v| v * scale).collect()
    };

    for r in [0.3, 1.0, 3.0, 5.0, 8.0] {
        let reference = miller(r, 20);
        for m in 0..=16 {
            let got = bessel_j(m as isize, r);
            assert!(
                (got - reference[m]).abs() < 1e-11,
                "J_{m}({r}) = {got}, Miller reference {}",
                reference[m]
            );
        }
    }

    // Near a Bessel zero only absolute error is meaningful
    // (J_1 has a zero at 7.01559...).
    let near_zero = bessel_j(1, 7.015586669815619);
    assert!(
        near_zero.abs() < 1e-12,
        "J_1 near its zero must be tiny in absolute value, got {near_zero}"
    );
    // Tiny argument: J_0(1e-12) = 1 - 2.5e-25, J_1(1e-12) = 5e-13.
    // puruspe is accurate to ~1 ulp of 1.0 (2.2e-16), so assert the
    // 1-ulp bound rather than the exact analytic deviation.
    assert!((bessel_j(0, 1e-12) - 1.0).abs() < 3e-16);
    assert!((bessel_j(1, 1e-12) - 5e-13).abs() < 1e-25);
}

#[test]
fn bessel_ladder_matches_high_precision_reference() {
    // Reference values evaluated with mpmath at 60 significant digits.
    // The ladder is a backward recurrence normalized on the turning-point
    // order, so this pins its shape and its normalization independently of
    // both `puruspe` and the time-grid DFT used elsewhere.
    let reference: [(f64, usize, f64); 16] = [
        (8.0, 0, 0.17165080713755390609),
        (8.0, 8, 0.22345498635110295428),
        (8.0, 20, 2.0805829639717027777e-7),
        (8.0, 40, 1.0010983703741214214e-24),
        (64.0, 0, 0.092590012216048114331),
        (64.0, 32, -0.069532193048999732619),
        (64.0, 64, 0.1118209766528825465),
        (64.0, 96, 4.1885539768510200271e-11),
        (128.0, 0, 0.0014722223281851497517),
        (128.0, 64, 0.044765812254841955288),
        (128.0, 128, 0.08875518402355613961),
        (128.0, 160, 1.36538822309338787e-8),
        (200.0, 0, -0.015437439930565091592),
        (200.0, 100, 0.0093332141865575864571),
        (200.0, 200, 0.076487608930953319678),
        (200.0, 240, 1.9238421623930608555e-9),
    ];
    for (r, m, expected) in reference {
        let ladder = BesselLadder::new(r, m);
        let got = ladder.j[m];
        // Pure relative tolerance: an absolute floor would make the
        // deep-tail references (1e-24 at (8, 40), 4e-11 at (64, 96))
        // meaningless, and the measured ladder error there is ~1e-15
        // relative, so 1e-12 is both safe and much stronger.
        assert!(
            (got - expected).abs() <= 1e-12 * expected.abs(),
            "J_{m}({r}) = {got}, mpmath reference {expected}"
        );
    }
}
#[test]
fn bessel_ladder_holds_machine_precision_across_the_closed_form_range() {
    // The closed-form cap is two orders of magnitude above the convolution
    // cap, so the ladder has to stay accurate there.  References come from a
    // 60-digit Miller recurrence normalized by Σ_m J_m(R)² = 1, whose
    // generator reproduces mpmath's own `besselj` to 1e-59 relative at
    // R = 2048 and 4096 (where mpmath's series still converges).
    let reference: [(f64, usize, f64); 21] = [
        (4096.0, 0, 1.84512862999888826e-3),
        (4096.0, 2048, 1.07157451300922933e-2),
        (4096.0, 4096, 2.79567017949838674e-2),
        (4096.0, 4097, 2.63531832981593467e-2),
        (8192.0, 0, -4.13522621428857352e-3),
        (8192.0, 1, -7.78564119591970000e-3),
        (8192.0, 4096, -4.55870884789159186e-3),
        (8192.0, 7372, 1.33484833425830847e-2),
        (8192.0, 8191, 2.31997231698089813e-2),
        (8192.0, 8192, 2.21892516034517713e-2),
        (8192.0, 8193, 2.11787800370945647e-2),
        (12000.0, 0, -7.16635491761340063e-4),
        (12000.0, 1, -7.24834561689710610e-3),
        (12000.0, 6000, 5.05249117141911858e-3),
        (12000.0, 10800, 8.70171159659731581e-3),
        (12000.0, 11999, 2.03214636355080677e-2),
        (12000.0, 12000, 1.95379458204867872e-2),
        (16384.0, 0, -6.12000513004155172e-3),
        (16384.0, 8192, 1.45380086823188362e-3),
        (16384.0, 14745, 8.84966808545540294e-3),
        (16384.0, 16384, 1.76116215137440066e-2),
    ];
    for (r, m, expected) in reference {
        let ladder = BesselLadder::new(r, m);
        let got = ladder.j[m];
        assert!(
            (got - expected).abs() <= 1e-12 * expected.abs(),
            "J_{m}({r}) = {got}, 60-digit reference {expected}"
        );
    }

    // The same orders where mpmath's own `besselj` still converges, so the
    // reference generator above is anchored to an independent implementation
    // rather than only to itself.
    let mpmath: [(f64, usize, f64); 8] = [
        (2048.0, 0, 7.93646094171144602e-3),
        (2048.0, 1024, -1.88980147990797168e-2),
        (2048.0, 2048, 3.52232263308058999e-2),
        (2048.0, 2049, 3.26790722294217559e-2),
        (4096.0, 0, 1.84512862999888826e-3),
        (4096.0, 2048, 1.07157451300922933e-2),
        (4096.0, 4096, 2.79567017949838674e-2),
        (4096.0, 4097, 2.63531832981593467e-2),
    ];
    for (r, m, expected) in mpmath {
        let ladder = BesselLadder::new(r, m);
        let got = ladder.j[m];
        assert!(
            (got - expected).abs() <= 1e-12 * expected.abs(),
            "J_{m}({r}) = {got}, mpmath reference {expected}"
        );
    }

    // The cap must be reachable: at the largest supported amplitude the
    // adaptive tail still converges inside MAX_BESSEL_ORDER, otherwise every
    // call at the cap would take the saturation error path instead.
    for margin in [0_isize, 48] {
        let (ladder, m_cap) = bessel_ladder_for_amplitude(
            MAX_BESSEL_ARG_CLOSED_FORM,
            1e-12,
            margin,
            MAX_BESSEL_ARG_CLOSED_FORM,
        )
        .unwrap();
        assert!(ladder.decayed && m_cap as usize <= ladder.max_order());
        assert!(ladder.max_order() < MAX_BESSEL_ORDER);
    }
}

#[test]
fn bessel_ladder_survives_rescale_heavy_sweeps() {
    // Small arguments with a high requested order make the growth factor
    // 2k/r swing by hundreds of decades per step, so the band rescale fires
    // on nearly every step; a rescale that lost the running scale would
    // show up here as a wrong low order or an underflowed tail.
    //
    // The reference is an ascending series evaluated here, not
    // [`bessel_j`]: the special-function wrapper is only verified around
    // `x ~ n`, and returns garbage for `x << n` (measured
    // `Jn(4096, 0.5) = 1.7e34`, where the true value underflows to zero).
    let series = |m: usize, r: f64| -> f64 {
        let half = r / 2.0;
        let mut term = 1.0_f64;
        for i in 1..=m {
            term *= half / i as f64;
        }
        let mut sum = term;
        for k in 1..200 {
            term *= -(half * half) / ((k * (m + k)) as f64);
            sum += term;
            if term.abs() <= f64::MIN_POSITIVE {
                break;
            }
        }
        sum
    };
    for (r, n) in [
        (0.5_f64, 4096_usize),
        (1.0, 4096),
        (2.0, 4096),
        (1e-6, 4096),
    ] {
        let ladder = BesselLadder::new(r, n);
        assert!(
            ladder.j.iter().all(|value| value.is_finite()),
            "r = {r} produced a non-finite ladder"
        );
        assert!(ladder.max_order() >= n);
        for m in [0_usize, 1, 2, 8, 64] {
            let expected = series(m, r);
            if expected == 0.0 {
                // The true order underflows in f64 (r = 1e-6, m = 64), so
                // the ladder must report it as negligible rather than
                // invent a value; a relative check here would be vacuous.
                assert!(
                    ladder.j[m].abs() <= BESSEL_DECAY_FLOOR,
                    "r = {r}, m = {m}: expected an underflowed order, got {}",
                    ladder.j[m]
                );
                continue;
            }
            assert!(
                (ladder.j[m] - expected).abs() <= 1e-12 * expected.abs(),
                "r = {r}, m = {m}: ladder {} vs series {expected}",
                ladder.j[m]
            );
        }
        for m in [2048_usize, 4096] {
            assert!(
                ladder.j[m].abs() <= BESSEL_DECAY_FLOOR,
                "r = {r}, m = {m}: expected a decayed tail entry, got {}",
                ladder.j[m]
            );
        }
    }
}

#[test]
fn bessel_arg_cap_and_adaptive_cutoff_are_pinned() {
    // The single-harmonic cap is inclusive and rejects the next float
    // above it; it sits two orders of magnitude above the convolution cap.
    let d = array![1.0];
    let at_cap = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(MAX_BESSEL_ARG_CLOSED_FORM, 0.0)],
        )],
    );
    assert!(bessel_peierls_coeffs(&d, &at_cap, -2, 2, 0).is_ok());
    let above_cap = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(
                f64::from_bits(MAX_BESSEL_ARG_CLOSED_FORM.to_bits() + 1),
                0.0,
            )],
        )],
    );
    assert!(bessel_peierls_coeffs(&d, &above_cap, -2, 2, 0).is_err());

    // Three or more carriers take the convolution, which keeps the low cap
    // even for an amplitude the other two paths would handle.
    let folded = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(MAX_BESSEL_ARG * 1.5, 0.0)]),
            LightMode::new(2, array![Complex::new(1.0, 0.0)]),
            LightMode::new(3, array![Complex::new(1.0, 0.0)]),
        ],
    );
    assert!(bessel_peierls_coeffs(&d, &folded, -2, 2, 0).is_err());

    // Cutoffs across the raised range, with and without the margin floor:
    // the returned order is the first one whose two-sided tail fits the
    // share, the ladder backs it, and the invariant the caller relies on
    // (`m_cap` strictly below the ladder's top) holds.
    for r in [9.0_f64, 64.0, MAX_BESSEL_ARG, MAX_BESSEL_ARG_CLOSED_FORM] {
        for margin in [0_isize, 48] {
            let (ladder, m_cap) = bessel_ladder_with_cutoff(r, 1e-12, margin);
            let m = m_cap as usize;
            assert!(ladder.decayed, "r = {r} ladder must decay");
            assert!(m > 0 && m < ladder.max_order());
            assert!(ladder.tail_after(m) <= 1e-12);
            let m_start = (r.ceil() as usize) + margin as usize;
            assert!(m >= m_start, "r = {r}, margin = {margin}");
            assert!(
                m == m_start || ladder.tail_after(m - 1) > 1e-12,
                "r = {r}, margin = {margin}: {m} is not the first fitting order"
            );
        }
    }
}

#[test]
fn fallback_saturation_covers_the_undecayed_window() {
    // Just below MAX_BESSEL_ORDER the ladder's top order has not decayed, so
    // its suffix sums stop at the cap and understate the truncation budget:
    // the stored suffix can look converged while the true two-sided tail,
    // which needs orders past the cap, does not.  Sizing must therefore fall
    // back to the conservative analytic bound and the maximum grid, exactly
    // as for r > MAX_BESSEL_ORDER, instead of certifying a cutoff from the
    // truncated sum.
    // r = 32483 sits on that knife edge for the current order cap: the scan
    // finds a cutoff whose *stored* suffix looks converged (9.5e-13 <= 1e-12)
    // while the complete two-sided tail, which needs orders past
    // MAX_BESSEL_ORDER, is 1.9e-12 and breaks the budget.  Sizing must read
    // `decayed == false` and take the conservative analytic bound with the
    // maximum grid instead of certifying that cutoff.
    let r_sat = 32_483.0;
    let (ladder, m_cap) = bessel_ladder_with_cutoff(r_sat, 1e-12, 0);
    assert!(!ladder.decayed, "r = {r_sat} must exhaust MAX_BESSEL_ORDER");
    let m = m_cap as usize;
    assert!(m < ladder.max_order(), "the cutoff must be a stored order");
    let stored = ladder.tail_after(m);
    assert!(
        stored <= 1e-12,
        "the stored suffix must look converged: {stored:e}"
    );
    let complete: f64 = 2.0
        * ((m + 1)..=(m + 200))
            .map(|k| bessel_j(k as isize, r_sat).abs())
            .sum::<f64>();
    assert!(
        complete > 1e-12,
        "the omitted orders must break the budget, got {complete:e}"
    );
    let near_cap = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(1, array![Complex::new(r_sat, 0.0)])],
    );
    let size = fallback_grid_size(&near_cap, &array![1.0, 0.0], -1, 1);
    assert!(size.saturated, "an undecayed ladder must saturate");
    assert_eq!(size.n_req, FALLBACK_GRID_MAX);
}

#[test]
fn bessel_ladder_anchors_away_from_bessel_zeros() {
    // J_0 has a zero at 2.404825557695773 and J_1 at 7.015586669815619.
    // Anchoring the sweep on such an order would scale every order by a
    // value whose relative error is enormous, so the anchor is chosen at the
    // largest magnitude instead and the orders next to the zero stay
    // accurate.  References from mpmath at 50 digits.
    let reference: [(f64, usize, f64); 8] = [
        (2.404_825_557_695_773, 0, -1.201_195_007_367_686_1e-16),
        (2.404_825_557_695_773, 1, 0.519_147_497_289_466_7),
        (2.404_825_557_695_773, 2, 0.431_754_807_019_680_4),
        (2.404_825_557_695_773, 3, 0.198_999_905_357_690_85),
        (7.015_586_669_815_619, 0, 0.300_115_752_526_132_56),
        (7.015_586_669_815_619, 1, 7.396_741_371_461_977e-17),
        (7.015_586_669_815_619, 2, -0.300_115_752_526_132_54),
        (7.015_586_669_815_619, 3, -0.171_113_702_474_732_7),
    ];
    for (r, m, expected) in reference {
        let ladder = BesselLadder::new(r, m);
        assert!(
            (ladder.j[m] - expected).abs() <= 1e-12 * expected.abs() + 1e-15,
            "J_{m}({r}) = {}, mpmath reference {expected}",
            ladder.j[m]
        );
    }
}

#[test]
fn bessel_ladder_tails_match_per_order_sum() {
    // `tail[m] = 2·Σ_{k>m}|J_k(r)|` must equal the direct sum built from
    // the single-order wrapper [`bessel_j`]; the adaptive cutoff reads
    // nothing else.  That wrapper also supplies the ladder's scale at the
    // anchor, so this is a shape/summation check rather than a fully
    // independent one; the mpmath and ascending-series tests above supply
    // the independent anchors.
    for r in [0.5, 3.0, 8.0, 64.0, 128.0] {
        let ladder = BesselLadder::new(r, (r.ceil() as usize) + 6);
        assert!(ladder.decayed, "r = {r} ladder must decay");
        for m in [
            (r.ceil() as usize).max(1),
            ladder.max_order() / 2,
            ladder.max_order() - 1,
        ] {
            let direct = 2.0
                * ((m + 1)..=ladder.max_order())
                    .map(|k| bessel_j(k as isize, r).abs())
                    .sum::<f64>();
            let got = ladder.tail_after(m);
            assert!(
                (got - direct).abs() <= 1e-11 * direct + 1e-18,
                "r = {r}, m = {m}: ladder tail {got} vs direct sum {direct}"
            );
        }
    }
}

#[test]
fn bessel_ladder_handles_extreme_arguments() {
    // The unscaled recurrence grows like (2/r)^k, which overflows f64 for a
    // small argument and a high seed.  The ratio-preserving rescale covers
    // the range above SWEEP_MIN_ARG, and the analytic branch below it; both
    // must reproduce the leading orders with every entry finite.
    // The band around 1e-150 is the one a one-sided rescale guard used to
    // drive to zero (J_1 returned as 0 instead of r/2).
    for r in [1e-3, 1e-8, 1e-40, 1e-100, 1e-150, 1e-200, 1e-250, 1e-320] {
        let ladder = BesselLadder::new(r, 48);
        assert!(ladder.decayed, "r = {r} ladder must decay");
        assert!(
            ladder.j.iter().all(|value| value.is_finite()),
            "r = {r} produced a non-finite ladder"
        );
        // J_0 = 1 - r²/4 + O(r⁴), J_1 = (r/2)(1 - r²/8) + O(r⁵).
        let j0 = 1.0 - r * r / 4.0 + r.powi(4) / 64.0;
        let j1 = r / 2.0 * (1.0 - r * r / 8.0);
        assert!(
            (ladder.j[0] - j0).abs() <= 1e-15 + 1e-14 * r * r,
            "J_0({r}) = {} vs {j0}",
            ladder.j[0]
        );
        assert!(
            (ladder.j[1] - j1).abs() <= 1e-14 * r + f64::MIN_POSITIVE,
            "J_1({r}) = {} vs {j1}",
            ladder.j[1]
        );
    }
    // Large argument with a requested order past the turning point: the
    // range extends until the top decays and the tails stay monotone.
    let ladder = BesselLadder::new(400.0, 512);
    assert!(ladder.max_order() >= 512);
    assert!(ladder.decayed);
    assert!(
        ladder
            .tail
            .iter()
            .all(|tail| tail.is_finite() && *tail >= 0.0)
    );
    assert!(ladder.tail_after(0) > ladder.tail_after(400));
}

#[test]
fn bessel_j_satisfies_recurrence_and_negative_order_symmetry() {
    // Recurrence: J_{m-1}(r) + J_{m+1}(r) = (2m/r) J_m(r).
    for r in [0.3, 0.7, 1.3, 2.5, 4.0, 7.0] {
        for m in 1..12 {
            let left = bessel_j(m - 1, r) + bessel_j(m + 1, r);
            let right = (2.0 * m as f64 / r) * bessel_j(m, r);
            assert!(
                (left - right).abs() < 1e-12,
                "recurrence failed at m={m}, r={r}: {left} vs {right}"
            );
        }
    }
    // Negative-order symmetry: J_{-m}(r) = (-1)^m J_m(r).
    for m in 1..8 {
        let expected = if m % 2 == 0 {
            bessel_j(m, 1.7)
        } else {
            -bessel_j(m, 1.7)
        };
        assert!((bessel_j(-m, 1.7) - expected).abs() < 1e-15);
    }
}

#[test]
fn bessel_coeffs_match_single_mode_closed_form() {
    // Single mode l=1: C_n = (-i)^n J_n(R) e^{+inδ} (verified reduction
    // of the generalized Bessel sum; the +iδ phase is the discriminating
    // one for complex amplitudes).
    let d = array![1.0];
    for (amplitude, name) in [
        (array![Complex::new(0.4, 0.0)], "linear"),
        (array![Complex::new(0.0, 0.4)], "circular"),
        (array![Complex::new(0.3, 0.2)], "elliptical"),
    ] {
        let drive = FloquetDrive::with_modes(0.8, vec![LightMode::new(1, amplitude.clone())]);
        let r = amplitude[0].norm();
        let delta = amplitude[0].arg();
        let coeffs = bessel_peierls_coeffs(&d, &drive, -6, 6, 6).unwrap();
        for n in -6..=6 {
            let expected = Complex::from_polar(1.0, -(n as f64) * std::f64::consts::FRAC_PI_2)
                * bessel_j(n, r)
                * Complex::from_polar(1.0, (n as f64) * delta);
            let got = coeffs[(n + 6) as usize];
            assert!(
                (got - expected).norm() < 1e-13,
                "{name}: C_{n} = {got}, closed form {expected}"
            );
        }
    }
}
#[test]
fn validator_scan_bound_dominates_the_true_cutoff() {
    // validate_sambe_allocation estimates an enumeration scan as
    // 4·(⌈r⌉ + 64 + 16·r^(1/3)) per carrier.  That estimate must never be
    // below the cutoff the backend actually computes, otherwise the validator
    // could certify a link the backend refuses.  Sweep the reachable range at
    // the error shares real drives produce.
    let shares = [1e-12_f64, 5e-13, 1e-12 / 3.0, 1e-12 / 8.0];
    for r in [
        1.0_f64,
        8.0,
        128.0,
        264.0,
        400.0,
        1000.0,
        4000.0,
        8000.0,
        12_000.0,
        16_000.0,
        MAX_BESSEL_ARG_CLOSED_FORM,
    ] {
        let estimate = 4 * ((r.ceil() as usize) + 64 + (16.0 * r.cbrt()) as usize);
        for share in shares {
            let (_, m_cap) = bessel_ladder_with_cutoff(r, share, 6);
            assert!(
                m_cap as usize <= estimate,
                "r = {r}, share = {share:e}: cutoff {m_cap} exceeds the validator's bound {estimate}"
            );
        }
    }
}

#[test]
fn bessel_enumeration_reaches_past_the_convolution_cap() {
    // Two carriers are linear in the amplitude, so they are exact well past
    // the convolution's 128 cap.  R = 400 is 3x beyond it; the time-grid DFT
    // is the independent oracle (its 4096-point grid resolves the band).
    let d = array![1.0, 0.5];
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0), Complex::new(0.0, 0.0)]),
            LightMode::new(3, array![Complex::new(0.0, 0.0), Complex::new(160.0, 40.0)]),
        ],
    );
    let grid = FloquetTimeGrid::new(&drive, 4096, -8, 8);
    let reference = peierls_fourier_coeffs(&d, -8, 8, &drive, &grid);
    let got = bessel_peierls_coeffs(&d, &drive, -8, 8, 6).unwrap();
    for (n, (a, b)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-12,
            "n = {}: enumeration {a} vs time-grid DFT {b}",
            n as isize - 8
        );
    }

    // The budget bounds the scan: a symmetric pair at the closed-form cap
    // with a wide requested range must be refused, not ground through.
    let heavy = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(
                1,
                array![Complex::new(16000.0, 0.0), Complex::new(0.0, 0.0)],
            ),
            LightMode::new(
                2,
                array![Complex::new(16000.0, 0.0), Complex::new(0.0, 0.0)],
            ),
        ],
    );
    assert!(bessel_peierls_coeffs(&d, &heavy, -2000, 2000, 6).is_err());
    assert!(bessel_peierls_coeffs(&d, &heavy, -2, 2, 6).is_ok());
}

#[test]
fn bessel_enumeration_matches_the_fold() {
    // Two carriers take the resonance enumeration; adding a third carrier
    // with a negligible amplitude forces the one-mode convolution instead.
    // The two paths must agree bin for bin, which pins the enumeration
    // against the code it replaced.
    let d = array![0.8, -1.1];
    let cases: Vec<Vec<LightMode>> = vec![
        vec![
            LightMode::new(1, array![Complex::new(0.9, 0.2), Complex::new(-0.3, 0.4)]),
            LightMode::new(
                2,
                array![Complex::new(0.15, -0.25), Complex::new(0.35, 0.05)],
            ),
        ],
        vec![
            LightMode::new(3, array![Complex::new(1.4, 0.0), Complex::new(0.0, -0.6)]),
            LightMode::new(-2, array![Complex::new(-0.2, 0.7), Complex::new(0.1, 0.3)]),
        ],
        vec![
            LightMode::new(7, array![Complex::new(2.0, 0.5), Complex::new(0.25, -0.75)]),
            LightMode::new(11, array![Complex::new(0.6, -0.4), Complex::new(-0.9, 0.2)]),
        ],
        // Large amplitudes: the adaptive cutoffs are ~60 orders per carrier,
        // so the enumeration's partner search spans the whole ladder.
        vec![
            LightMode::new(1, array![Complex::new(40.0, 5.0), Complex::new(-3.0, 8.0)]),
            LightMode::new(2, array![Complex::new(6.0, -9.0), Complex::new(11.0, 2.0)]),
        ],
    ];
    for (case, modes) in cases.iter().enumerate() {
        let enumerated = FloquetDrive::with_modes(1.0, modes.clone());
        let mut folded_modes = modes.clone();
        folded_modes.push(LightMode::new(
            5,
            Array1::from_elem(2, Complex::new(1e-300, 0.0)),
        ));
        let folded = FloquetDrive::with_modes(1.0, folded_modes);
        let a = bessel_peierls_coeffs(&d, &enumerated, -6, 6, 6).unwrap();
        let b = bessel_peierls_coeffs(&d, &folded, -6, 6, 6).unwrap();
        for (n, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).norm() < 1e-14,
                "case {case}, n = {}: enumeration {x} vs fold {y}",
                n as isize - 6
            );
        }
    }
}

#[test]
fn bessel_coeffs_match_time_grid_dft() {
    // The strongest test: the generalized Bessel convolution must
    // reproduce the independent time-grid DFT for multi-mode and
    // multi-harmonic drives.
    let d = array![1.3, -0.7];
    let cases: Vec<FloquetDrive> = vec![
        FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(0.25, 0.0), Complex::new(0.0, 0.25)]),
                LightMode::new(2, array![Complex::new(0.1, 0.0), Complex::new(0.05, -0.05)]),
            ],
        ),
        FloquetDrive::with_modes(
            0.7,
            vec![
                LightMode::new(1, array![Complex::new(0.3, 0.1), Complex::new(-0.1, 0.2)]),
                LightMode::new(
                    -3,
                    array![Complex::new(0.08, -0.04), Complex::new(0.02, 0.06)],
                ),
            ],
        ),
        FloquetDrive::with_modes(
            0.5,
            vec![
                LightMode::new(1, array![Complex::new(0.2, 0.0), Complex::new(0.0, 0.2)]),
                LightMode::new(2, array![Complex::new(0.05, 0.0), Complex::new(0.0, -0.05)]),
                LightMode::new(
                    3,
                    array![Complex::new(0.02, 0.01), Complex::new(0.01, -0.02)],
                ),
            ],
        ),
    ];
    for (case, drive) in cases.iter().enumerate() {
        let harmonic_min = -5_isize;
        let harmonic_max = 5_isize;
        let bessel = bessel_peierls_coeffs(&d, drive, harmonic_min, harmonic_max, 6).unwrap();
        let time_grid = FloquetTimeGrid::new(drive, 512, harmonic_min, harmonic_max);
        let dft = peierls_fourier_coeffs(&d, harmonic_min, harmonic_max, drive, &time_grid);
        for (n, (got, expected)) in bessel.iter().zip(dft.iter()).enumerate() {
            assert!(
                (got - expected).norm() < 1e-10,
                "case {case}, n={}: Bessel {got} vs DFT {expected}",
                harmonic_min + n as isize
            );
        }
    }
}

#[test]
fn bessel_single_harmonic_closed_form_matches_time_grid() {
    // One nonzero temporal harmonic collapses the resonance sum to a single
    // term, so the coefficient path takes the closed form instead of the
    // one-mode convolution.  Pin it against the independent time-grid DFT,
    // covering a negative harmonic, a coherently summed pair at one
    // harmonic (still a single harmonic), a DC mode riding along, and a
    // purely static drive (which is only an on-site Peierls phase).
    let d = array![1.3, -0.7];
    let harmonic_min = -6_isize;
    let harmonic_max = 6_isize;
    let cases: Vec<FloquetDrive> = vec![
        FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.25, 0.1), Complex::new(-0.05, 0.2)],
            )],
        ),
        FloquetDrive::with_modes(
            0.7,
            vec![LightMode::new(
                -2,
                array![Complex::new(0.3, 0.0), Complex::new(0.1, -0.15)],
            )],
        ),
        FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(0.2, 0.0), Complex::new(0.0, 0.2)]),
                LightMode::new(1, array![Complex::new(0.05, -0.05), Complex::new(0.0, 0.0)]),
            ],
        ),
        FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(0, array![Complex::new(0.3, 0.2), Complex::new(0.0, 0.0)]),
                LightMode::new(3, array![Complex::new(0.1, 0.0), Complex::new(0.0, -0.1)]),
            ],
        ),
        FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                0,
                array![Complex::new(0.4, -0.25), Complex::new(0.0, 0.0)],
            )],
        ),
        // Two DC modes: the static phase accumulates over both.
        FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(0, array![Complex::new(0.2, 0.0), Complex::new(0.0, 0.0)]),
                LightMode::new(0, array![Complex::new(0.15, -0.4), Complex::new(0.0, 0.0)]),
            ],
        ),
        // Equal harmonics that cancel exactly: the drive is static, and the
        // ladder must never be asked for R = 0.
        FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(0.3, 0.1), Complex::new(0.0, 0.0)]),
                LightMode::new(1, array![Complex::new(-0.3, -0.1), Complex::new(0.0, 0.0)]),
            ],
        ),
    ];
    for (case, drive) in cases.iter().enumerate() {
        let closed = bessel_peierls_coeffs(&d, drive, harmonic_min, harmonic_max, 6).unwrap();
        let grid = FloquetTimeGrid::new(drive, 512, harmonic_min, harmonic_max);
        let reference = peierls_fourier_coeffs(&d, harmonic_min, harmonic_max, drive, &grid);
        for (n, (got, want)) in closed.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).norm() < 1e-12,
                "case {case}, n = {}: closed form {got} vs DFT {want}",
                harmonic_min + n as isize
            );
        }
    }
}

#[test]
fn bessel_single_harmonic_fast_path_is_actually_taken() {
    // Two modes at the same harmonic whose individual amplitudes exceed
    // MAX_BESSEL_ARG while their coherent sum does not: the convolution path
    // validates each mode and would error, so a successful call proves the
    // closed form handled it.  The result must equal the single coherent
    // drive it is physically identical to.
    let d = array![1.0];
    let split = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(1, array![Complex::new(-150.0, 0.0)]),
        ],
    );
    let coherent = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(1, array![Complex::new(50.0, 0.0)])],
    );
    let split_coeffs = bessel_peierls_coeffs(&d, &split, -3, 3, 6).unwrap();
    let coherent_coeffs = bessel_peierls_coeffs(&d, &coherent, -3, 3, 6).unwrap();
    for (n, (a, b)) in split_coeffs.iter().zip(coherent_coeffs.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-14,
            "n = {}: split {a} vs coherent {b}",
            n as isize - 3
        );
    }

    // A single 200-amplitude mode is past the convolution cap but well
    // inside the closed form’s, so it must be evaluated, not refused:
    // that is the whole point of the split cap.
    let over = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(1, array![Complex::new(200.0, 0.0)])],
    );
    assert!(bessel_peierls_coeffs(&d, &over, -2, 2, 6).is_ok());
    let single = bessel_peierls_coeffs(&d, &over, -2, 2, 6).unwrap();
    assert!((single[2].re - bessel_j(0, 200.0)).abs() < 1e-14);

    // Three carriers force the convolution, which keeps the low cap and must
    // refuse the same amplitude; two carriers are enumerated and accept it.
    let folded = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(2, array![Complex::new(1.0, 0.0)]),
            LightMode::new(3, array![Complex::new(1.0, 0.0)]),
        ],
    );
    assert!(bessel_peierls_coeffs(&d, &folded, -2, 2, 6).is_err());
    let two = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(2, array![Complex::new(1.0, 0.0)]),
        ],
    );
    assert!(bessel_peierls_coeffs(&d, &two, -2, 2, 6).is_ok());
}

#[test]
fn bessel_coeffs_handle_empty_drive() {
    let d = array![0.5];
    let drive = FloquetDrive::new(1.0);
    let coeffs = bessel_peierls_coeffs(&d, &drive, -3, 3, 6).unwrap();
    for n in -3..=3 {
        let expected = if n == 0 {
            Complex::new(1.0, 0.0)
        } else {
            Complex::new(0.0, 0.0)
        };
        assert!((coeffs[(n + 3) as usize] - expected).norm() < 1e-15);
    }
    // The reference oracle has its own empty-drive branch; the harmonic cache
    // returns before reaching it, so exercise it directly here.
    let grid = FloquetTimeGrid::new(&drive, 8, -3, 3);
    let reference = peierls_fourier_coeffs(&d, -3, 3, &drive, &grid);
    for (n, (a, b)) in coeffs.iter().zip(reference.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-15,
            "n = {}: Bessel {a} vs reference {b}",
            n as isize - 3
        );
    }
}

#[test]
fn harmonic_cache_bessel_matches_time_grid_and_dedupes_links() {
    // Spinful 2-orbital model: the four spin blocks of every hopping
    // share the same link displacement, so the dedup path must produce
    // identical blocks to the non-dedup reference, and the Bessel
    // backend must agree with the time grid.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.3, 0.0]];
    let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 0, &array![1, 0], None);
    model.add_hop(-0.5, 0, 1, &array![0, 1], None);
    model.add_hop(
        Complex::new(0.1, 0.2),
        0,
        1,
        &array![1, 1],
        SpinDirection::X,
    );

    // Observable dedup premise: the model's non-zero hopping entries
    // share only 6 distinct link displacements (3 bonds and their
    // Hermitian partners at -R; the spin blocks of a bond share one d).
    let mut distinct_d = Vec::<Vec<u64>>::new();
    for i_r in 0..model.hamR.nrows() {
        let r_vec = model.hamR.row(i_r);
        for i in 0..model.nsta() {
            for j in 0..model.nsta() {
                if model.ham[[i_r, i, j]].re == 0.0 && model.ham[[i_r, i, j]].im == 0.0 {
                    continue;
                }
                let d_cart =
                    model.link_displacement_cartesian(i % model.norb(), j % model.norb(), &r_vec);
                let key: Vec<u64> = d_cart.iter().map(|value| value.to_bits()).collect();
                if !distinct_d.contains(&key) {
                    distinct_d.push(key);
                }
            }
        }
    }
    assert_eq!(
        distinct_d.len(),
        6,
        "dedup premise: expected 6 distinct link displacements, found {}",
        distinct_d.len()
    );

    let drive = FloquetDrive::with_modes(
        0.8,
        vec![
            LightMode::new(1, array![Complex::new(0.2, 0.0), Complex::new(0.0, 0.2)]),
            LightMode::new(2, array![Complex::new(0.05, -0.05), Complex::new(0.0, 0.0)]),
        ],
    );
    let n_time = 512;
    let time_grid_cache =
        model.floquet_harmonic_cache(&drive, -4, 4, &PeierlsFourierMethod::TimeGrid { n_time });
    let bessel_cache = model.floquet_harmonic_cache(
        &drive,
        -4,
        4,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    assert_eq!(time_grid_cache.blocks.dim(), bessel_cache.blocks.dim());
    for (a, b) in time_grid_cache
        .blocks
        .iter()
        .zip(bessel_cache.blocks.iter())
    {
        assert!(
            (a - b).norm() < 1e-10,
            "Bessel cache {b} vs time-grid cache {a}"
        );
    }
}

#[test]
fn sambe_validation_mirrors_the_backend_limits() {
    let model = chain_model();
    let k = array![0.21];
    let trunc = FloquetTruncation::new(0);

    // The backend sums equal harmonics coherently before it checks the cap,
    // so two modes at the closed-form cap sum to twice it: the call must be
    // refused rather than silently aliased through a clamped fallback grid.
    let split = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(4096, array![Complex::new(MAX_BESSEL_ARG_CLOSED_FORM, 0.0)]),
            LightMode::new(4096, array![Complex::new(MAX_BESSEL_ARG_CLOSED_FORM, 0.0)]),
        ],
    );
    assert!(model.floquet_model(&split, &trunc).is_err());

    // Exactly at the cap is still the ladder's range, and one float above it
    // is resolvable by the per-link fallback, so both must build and agree.
    // The fallback grid is the accuracy-limiting side: its measured
    // relative deviation here is 4.1e-12, so 1e-10 is a real bound rather
    // than a formality.
    let at_cap = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(MAX_BESSEL_ARG_CLOSED_FORM, 0.0)],
        )],
    );
    let above_cap = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(
                f64::from_bits(MAX_BESSEL_ARG_CLOSED_FORM.to_bits() + 1),
                0.0,
            )],
        )],
    );
    let from_ladder = model
        .sambe_ham_from_model(&k, &at_cap, &trunc, Gauge::Lattice)
        .unwrap();
    let from_grid = model
        .sambe_ham_from_model(&k, &above_cap, &trunc, Gauge::Lattice)
        .unwrap();
    assert!(
        from_ladder
            .iter()
            .zip(&from_grid)
            .all(|(a, b)| (a - b).norm() < 1e-10),
        "the two backends must agree across the cap boundary"
    );

    // A mode at another harmonic sends the backend to the convolution even
    // when the equal-harmonic pair cancels exactly.  Summing per harmonic
    // instead would call this drive exact and let it through; the
    // convolution operand is past its cap and the huge harmonic makes the
    // per-link grid unresolvable, so the call must be refused.  The photon
    // cutoff stays small on purpose: a wrong verdict here would otherwise
    // build a multi-terabyte Sambe matrix instead of failing.
    let cancelling = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(100_000, array![Complex::new(1.0, 0.0)]),
            LightMode::new(1, array![Complex::new(-200.0, 0.0)]),
        ],
    );
    assert!(model.floquet_model(&cancelling, &trunc).is_err());

    // A drive-level carrier count that disagrees with the link's must keep both
    // sides in step: three raw carriers (one with a zero projection) hold the
    // convolution cap on both sides, so the two spellings must agree.
    let zero_third = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(2, array![Complex::new(1.0, 0.0)]),
            LightMode::new(3, array![Complex::new(0.0, 0.0)]),
        ],
    );
    let plain = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(2, array![Complex::new(1.0, 0.0)]),
        ],
    );
    let from_plain = model
        .sambe_ham_from_model(&k, &plain, &trunc, Gauge::Lattice)
        .unwrap();
    let from_zero = model
        .sambe_ham_from_model(&k, &zero_third, &trunc, Gauge::Lattice)
        .unwrap();
    assert!(
        from_plain
            .iter()
            .zip(&from_zero)
            .all(|(a, b)| (a - b).norm() < 1e-11)
    );

    // Two large carriers plus a static mode: the enumeration scans the
    // *carriers* only, so the static mode must not make its budget look small,
    // and a request the per-link grid cannot resolve must be refused.
    let static_and_big = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(900, array![Complex::new(16000.0, 0.0)]),
            LightMode::new(901, array![Complex::new(16000.0, 0.0)]),
            LightMode::new(0, array![Complex::new(1.0, 0.0)]),
        ],
    );
    assert!(
        model
            .floquet_model(&static_and_big, &FloquetTruncation::new(500))
            .is_err()
    );

    // The requested-bin range alone can exceed the fallback grid cap: with a
    // huge photon cutoff, a link beyond the closed-form range cannot be
    // resolved, and the call must be refused before the Sambe matrix is even
    // allocated.
    let above = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(2.0 * MAX_BESSEL_ARG_CLOSED_FORM, 0.0)],
        )],
    );
    assert!(
        model
            .floquet_model(&above, &FloquetTruncation::new(300_000))
            .is_err()
    );

    // A single coherent harmonic needs no convolution window at all, so even
    // an enormous harmonic is fine: only the closed form runs.
    let high_harmonic = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            20_000,
            array![Complex::new(MAX_BESSEL_ARG, 0.0)],
        )],
    );
    assert!(model.floquet_model(&high_harmonic, &trunc).is_ok());
    assert!(bessel_peierls_coeffs(&array![1.0], &high_harmonic, 0, 0, 6).is_ok());

    // Tiny amplitudes at enormous distinct harmonics stay inside the cap on
    // every mode.  Two carriers are enumerated without any window, so this
    // drive is exact and every requested bin is reachable only through
    // m = 0: C_0 = 1 and the rest vanish.
    let wide = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1_000_000_000, array![Complex::new(1e-6, 0.0)]),
            LightMode::new(1_000_000_001, array![Complex::new(1e-6, 0.0)]),
        ],
    );
    let wide_coeffs = bessel_peierls_coeffs(&array![1.0], &wide, -2, 2, 6).unwrap();
    // The harmonics differ by 1, so besides the trivial m = 0 term the pair
    // (m, m') = (-1, +1) resonates exactly at n = -1 and (+1, -1) at n = +1:
    // B(-1)B'(+1) = B(+1)B'(-1) = -J_1(r)^2 with r = 1e-6.  Orders +-2 have
    // no integer solution at all.
    let j0_sq = bessel_j(0, 1e-6).powi(2);
    let j1_sq = bessel_j(1, 1e-6).powi(2);
    for (n, value) in (-2_isize..=2).zip(wide_coeffs.iter()) {
        let (expected, tolerance) = match n {
            0 => (j0_sq, 1e-15),
            -1 | 1 => (-j1_sq, 1e-25),
            _ => (0.0, 1e-25),
        };
        assert!(
            (value.re - expected).abs() < tolerance && value.im.abs() < tolerance,
            "n = {n}: {value} vs {expected}"
        );
    }
    assert!(model.floquet_model(&wide, &trunc).is_ok());

    // A third carrier falls back to the fold, whose window is `Σ|l|·M` wide:
    // that one must be refused, not allocated.
    let wider = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1_000_000_000, array![Complex::new(1e-6, 0.0)]),
            LightMode::new(1_000_000_001, array![Complex::new(1e-6, 0.0)]),
            LightMode::new(1_000_000_002, array![Complex::new(1e-6, 0.0)]),
        ],
    );
    assert!(model.floquet_model(&wider, &trunc).is_err());
    assert!(bessel_peierls_coeffs(&array![1.0], &wider, -2, 2, 6).is_err());
}

#[test]
fn sambe_review_regression_null_imaginary_dc_preserves_the_drive() {
    let model = chain_model();
    let k = array![0.0];
    let trunc = FloquetTruncation::new(1);
    let harmonic = FALLBACK_GRID_MAX as isize;
    let mut drive = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(harmonic, array![Complex::new(1.0, 0.0)]),
            LightMode::new(harmonic + 1, array![Complex::new(1.0, 0.0)]),
        ],
    );
    let expected = model
        .sambe_ham_from_model(&k, &drive, &trunc, Gauge::Lattice)
        .unwrap();
    // Independent J0(1), J1(1), J2(1) references. For |n| <= 2 the
    // resonance (m1, m2) = (n, -n) gives C_n = (-1)^n J_|n|(1)^2.
    let bessel = [
        0.765_197_686_557_966_6_f64,
        0.440_050_585_744_933_5,
        0.114_903_484_931_900_5,
    ];
    for ((row, col), &value) in expected.indexed_iter() {
        let order = row.abs_diff(col);
        let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
        let mut reference = -2.0 * sign * bessel[order].powi(2);
        if row == col {
            reference += (row as f64 - 1.0) * drive.omega0_ev;
        }
        assert!(
            (value - reference).norm() < 1e-10,
            "the baseline disagrees with the independent resonance sum: {value} vs {reference}"
        );
    }
    // Re(i A exp(0)) = 0: this mode changes neither the field nor its
    // bandwidth, even when its imaginary amplitude exceeds the AC cap.
    drive.add_mode(LightMode::new(0, array![Complex::new(0.0, 20_000.0)]));
    let from_model = model
        .floquet_model(&drive, &trunc)
        .unwrap()
        .gen_ham(&k, Gauge::Lattice);
    for actual in [from_model] {
        for (index, &value) in expected.indexed_iter() {
            assert!(
                (actual[index] - value).norm() < 1e-10,
                "a null DC mode changed H_F{index:?}: {} vs {value}",
                actual[index]
            );
        }
    }
}

#[test]
fn sambe_review_regression_overflowing_drift_never_returns_aliased_coefficients() {
    let model = chain_model();
    let k = array![0.0];
    let trunc = FloquetTruncation::new(1);
    // Independent J0(1), J1(1), J2(1) references. Adjacent huge carriers
    // have C_n = (-1)^n J_|n|(1)^2 for this whole requested range. Products
    // outside isize must still cancel, and construction must succeed, not merely
    // agree on an error. n_max=0 would miss the lost second harmonic.
    let bessel = [
        0.765_197_686_557_966_6_f64,
        0.440_050_585_744_933_5,
        0.114_903_484_931_900_5,
    ];
    for harmonics in [[isize::MAX - 1, isize::MAX], [isize::MIN, isize::MIN + 1]] {
        let drive = FloquetDrive::with_modes(
            5.0,
            harmonics
                .into_iter()
                .map(|h| LightMode::new(h, array![Complex::new(1.0, 0.0)]))
                .collect(),
        );
        let matrices = [model
            .floquet_model(&drive, &trunc)
            .unwrap()
            .gen_ham(&k, Gauge::Lattice)];
        for ham in matrices {
            for ((row, col), &value) in ham.indexed_iter() {
                let order = row.abs_diff(col);
                let sign = if order % 2 == 0 { 1.0 } else { -1.0 };
                let expected = -2.0 * sign * bessel[order].powi(2)
                    + if row == col {
                        (row as f64 - 1.0) * drive.omega0_ev
                    } else {
                        0.0
                    };
                assert!(
                    (value - expected).norm() < 1e-10,
                    "{harmonics:?}: H[{row},{col}]={value}, expected {expected}"
                );
            }
        }
    }
}

#[test]
fn static_shortcut_preserves_mode_local_link_projections() {
    let mut model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, -1], None);
    let big = (1_u64 << 53) as f64;
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(0, array![Complex::new(big, 0.0), Complex::new(big, 0.0)]),
            LightMode::new(0, array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]),
        ],
    );
    let k = array![0.0, 0.0];
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    let mut ac = drive.clone();
    for mode in &mut ac.modes {
        mode.harmonic = 1;
    }
    ac.add_mode(LightMode::new(
        1,
        array![Complex::new(-big, 0.0), Complex::new(-big, 0.0)],
    ));
    for (drive, expected) in [
        (drive, -2.0 * 1.0_f64.cos()),
        (ac, -2.0 * 0.765_197_686_557_966_6),
    ] {
        let expected = Complex::new(expected, 0.0);
        let sambe = model
            .sambe_ham_from_model(&k, &drive, &FloquetTruncation::new(0), Gauge::Lattice)
            .unwrap();
        assert!((sambe[[0, 0]] - expected).norm() < 1e-12);
        for effective in [
            model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap(),
            model
                .floquet_effective_q_model(&drive, Some(&options), &array![0.0, 0.0])
                .unwrap(),
            model
                .floquet_effective_mode_resolved_model(&drive, Some(&options))
                .unwrap(),
            model
                .floquet_effective_model_legacy(&drive, 32, [3, 3], Some(&options))
                .unwrap(),
        ] {
            assert!((effective.gen_ham(&k, Gauge::Lattice)[[0, 0]] - expected).norm() < 1e-12);
        }
    }
}

#[test]
fn true_fallback_nonzero_bins_preserve_negative_carrier_and_coherent_residual_phase() {
    // Independent 60-digit mpmath J0/J1/J2(24576), not the production ladder.
    let j = [
        -0.000_538_813_002_458_071_2,
        0.005_060_997_376_662_55,
        0.000_539_224_867_478_958_5,
    ];
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [1.0]], None).unwrap();
    model.set_hop(1.0, 0, 1, &array![0], None);
    let z = Complex::new(24576.0 * 0.6, 24576.0 * 0.8);
    let drives = [
        (
            FloquetDrive::with_modes(5.0, vec![LightMode::new(-1, array![z])]),
            -1,
            z.arg(),
        ),
        (
            FloquetDrive::with_modes(
                5.0,
                vec![
                    LightMode::new(1, array![Complex::new(1e19, 1e19)]),
                    LightMode::new(-1, array![Complex::new(-1e19 + 24576.0, 1e19)]),
                ],
            ),
            1,
            0.0,
        ),
    ];
    for (drive, carrier, phase) in drives {
        assert!(bessel_peierls_coeffs(&array![1.0], &drive, -2, 2, 6).is_err());
        let size = fallback_grid_size(&drive, &array![1.0], -2, 2);
        assert!(!size.clamped && !size.saturated);
        let ham = model
            .sambe_ham_from_model(
                &array![0.0],
                &drive,
                &FloquetTruncation::new(1),
                Gauge::Lattice,
            )
            .unwrap();
        for left in 0..3 {
            for right in 0..3 {
                let n = left as isize - right as isize;
                let m = -n / carrier;
                let bessel = j[m.unsigned_abs()] * if m < 0 && m % 2 != 0 { -1.0 } else { 1.0 };
                let expected = Complex::new(0.0, -1.0).powi(m as i32)
                    * bessel
                    * Complex::from_polar(1.0, -(m as f64) * phase);
                assert!(
                    (ham[[2 * left, 2 * right + 1]] - expected).norm() < 1e-11,
                    "n={n}: {:?} vs {expected}",
                    ham[[2 * left, 2 * right + 1]]
                );
            }
        }
    }
}

#[test]
fn bessel_integer_limits_preserve_resonances_without_panics() {
    let model = chain_model();
    let k = array![0.13];
    let trunc = FloquetTruncation::new(1);
    let bessel = [
        0.765_197_686_557_966_6_f64,
        0.440_050_585_744_933_5,
        0.114_903_484_931_900_5,
    ];
    let bloch_phase = Complex::from_polar(1.0, TAU * k[0]);
    for large in [isize::MIN, isize::MAX] {
        for small in [-1, 1] {
            let drive = FloquetDrive::with_modes(
                5.0,
                vec![
                    LightMode::new(large, array![Complex::new(1.0, 0.0)]),
                    LightMode::new(small, array![Complex::new(1.0, 0.0)]),
                ],
            );
            let coefficients = bessel_peierls_coeffs(&array![1.0], &drive, -2, 2, 6).unwrap();
            for (index, n) in (-2_isize..=2).enumerate() {
                let expected = bessel[0]
                    * bessel[n.unsigned_abs()]
                    * Complex::new(0.0, -1.0).powi(n.unsigned_abs() as i32);
                assert!((coefficients[index] - expected).norm() < 1e-12);
            }
            for ham in [model
                .floquet_model(&drive, &trunc)
                .unwrap()
                .gen_ham(&k, Gauge::Lattice)]
            {
                for ((row, col), &value) in ham.indexed_iter() {
                    let order = row.abs_diff(col);
                    let coefficient =
                        bessel[0] * bessel[order] * Complex::new(0.0, -1.0).powi(order as i32);
                    let expected = -2.0 * (coefficient * bloch_phase).re
                        + if row == col {
                            (row as f64 - 1.0) * drive.omega0_ev
                        } else {
                            0.0
                        };
                    assert!((value - expected).norm() < 1e-11);
                }
            }
        }
    }
    // The single-carrier lookup must widen MIN negation and MIN / -1 too.
    for harmonic in [isize::MIN, -1, 1] {
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(harmonic, array![Complex::new(1.0, 0.0)])],
        );
        let value =
            bessel_peierls_coeffs(&array![1.0], &drive, isize::MIN, isize::MIN, 6).unwrap()[0];
        let expected = if harmonic == isize::MIN {
            Complex::new(0.0, -bessel[1])
        } else {
            Complex::new(0.0, 0.0)
        };
        assert!((value - expected).norm() < 1e-12);
    }
}

// Definition-only oracle: evolve signed modes in Cartesian components before
// projecting. No production grouping, DC split, Bessel ladder or grid sizing.
fn independent_cartesian_dft(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    n_time: usize,
    min: isize,
    max: isize,
) -> Array1<Complex<f64>> {
    let mut coeffs = Array1::<Complex<f64>>::zeros((max - min + 1) as usize);
    for it in 0..n_time {
        let theta = TAU * it as f64 / n_time as f64;
        let mut field = 0.0;
        for mode in &drive.modes {
            let carrier = Complex::from_polar(1.0, -(mode.harmonic as f64) * theta);
            for a in 0..d.len() {
                field += (mode.a_complex[a] * carrier).re * d[a];
            }
        }
        let peierls = Complex::from_polar(1.0, -field);
        for (index, n) in (min..=max).enumerate() {
            coeffs[index] += Complex::from_polar(1.0, n as f64 * theta) * peierls;
        }
    }
    coeffs.mapv(|value| value / n_time as f64)
}

#[test]
fn projected_fallback_matches_an_independent_cartesian_dft() {
    // The production fallback and the test-only reference now share the same
    // carrier grouping, so neither can falsify that grouping.  This oracle
    // goes back to the definition: build the Cartesian field from every mode
    // and direction with a *signed* harmonic, and Fourier transform that
    // field.  No projection helper, no grouping by |l|, no separate DC phase.
    let d = array![1.0, -0.5];
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(
                1,
                array![Complex::new(24_576.0, 0.0), Complex::new(0.0, 0.0)],
            ),
            LightMode::new(-1, array![Complex::new(0.0, 300.0), Complex::new(0.0, 0.0)]),
            LightMode::new(2, array![Complex::new(0.0, 0.0), Complex::new(120.0, 0.0)]),
            LightMode::new(0, array![Complex::new(0.5, 9.0), Complex::new(0.25, 0.0)]),
        ],
    );
    let (harmonic_min, harmonic_max) = (-1_isize, 1_isize);
    let clamped = std::sync::atomic::AtomicBool::new(false);
    let saturated = std::sync::atomic::AtomicBool::new(false);
    let got =
        fallback_time_grid_coeffs(&d, &drive, harmonic_min, harmonic_max, &clamped, &saturated);
    let size = fallback_grid_size(&drive, &d, harmonic_min, harmonic_max);
    assert!(!size.clamped, "the oracle needs an unclamped grid");
    // Fixed independent grid, comfortably above twice the signal bandwidth
    // (~24700): a sizing regression must not alias the reference identically.
    let want = independent_cartesian_dft(&d, &drive, 1 << 18, harmonic_min, harmonic_max);
    for (i_n, (actual, reference)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (actual - reference).norm() < 1e-9,
            "n = {}: {actual} vs independent Cartesian DFT {reference}",
            i_n as isize + harmonic_min
        );
    }
}

#[test]
fn enumeration_matches_the_cartesian_oracle_for_a_negative_carrier() {
    // Complex amplitude is essential: with purely real amplitudes flipping
    // the carrier's sign leaves cos(l theta), and thus all coefficients, unchanged.
    let d = array![1.0];
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(-1, array![Complex::new(1.0, 0.4)]),
            LightMode::new(3, array![Complex::new(0.7, -0.2)]),
        ],
    );
    let reference = independent_cartesian_dft(&d, &drive, 1024, -2, 2);
    let refined = independent_cartesian_dft(&d, &drive, 2048, -2, 2);
    assert!(
        reference
            .iter()
            .zip(&refined)
            .all(|(a, b)| (*a - *b).norm() < 1e-13)
    );
    let mut wrong_sign = drive.clone();
    wrong_sign.modes[0].harmonic = 1;
    let sign_flipped = independent_cartesian_dft(&d, &wrong_sign, 1024, -2, 2);
    assert!(
        reference
            .iter()
            .zip(&sign_flipped)
            .any(|(a, b)| (*a - *b).norm() > 1e-3)
    );
    let got = bessel_peierls_coeffs(&d, &drive, -2, 2, BESSEL_CUTOFF_MARGIN).unwrap();
    for (i_n, (actual, expected)) in got.iter().zip(reference.iter()).enumerate() {
        assert!(
            (actual - expected).norm() < 1e-11,
            "n = {}: {actual} vs Cartesian oracle {expected}",
            i_n as isize - 2
        );
    }
}

#[test]
fn fallback_projection_matches_an_independent_bessel_reference() {
    let mut model =
        Model::<false, 2>::tb_model(array![[1.0, 0.0], [0.0, 1.0]], array![[0.0, 0.0]], None)
            .unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, -1], None);
    let k = array![0.0, 0.0];
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    // 60-digit mpmath reference: -2 J0(24576). Both representations have
    // exactly the same projection. This fallback is neither clamped nor saturated.
    let expected = 0.001_077_626_004_916_142_4;
    assert!(24_576.0 > MAX_BESSEL_ARG_CLOSED_FORM);
    for amplitude in [array![24_576.0, 0.0], array![1e19, 1e19 - 24_576.0]] {
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(1, amplitude.mapv(|a| Complex::new(a, 0.0)))],
        );
        assert!(bessel_peierls_coeffs(&array![1.0, -1.0], &drive, 0, 0, 6).is_err());
        let size = fallback_grid_size(&drive, &array![1.0, -1.0], 0, 0);
        assert!(!size.clamped && !size.saturated);
        let reference_grid = FloquetTimeGrid::new(&drive, 131_072, 0, 0);
        let reference =
            peierls_fourier_coeffs(&array![1.0, -1.0], 0, 0, &drive, &reference_grid)[0];
        assert!((reference - (-0.5 * expected)).norm() < 1e-10);
        for ham in [
            model
                .floquet_model(&drive, &FloquetTruncation::new(0))
                .unwrap()
                .gen_ham(&k, Gauge::Lattice),
            model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap()
                .gen_ham(&k, Gauge::Lattice),
            model
                .floquet_effective_q_model(&drive, Some(&options), &k)
                .unwrap()
                .gen_ham(&k, Gauge::Lattice),
            model
                .floquet_effective_mode_resolved_model(&drive, Some(&options))
                .unwrap()
                .gen_ham(&k, Gauge::Lattice),
        ] {
            assert!((ham[[0, 0]] - expected).norm() < 1e-10);
        }
    }
}

#[test]
fn fallback_factors_real_dc_and_coherent_cancellations_before_sampling() {
    let model = chain_model();
    let k = array![0.0];
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    let j0_200 = -0.015_437_439_930_565_085_f64;
    assert!(200.0 > MAX_BESSEL_ARG);
    let mut drive = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(1, array![Complex::new(200.0, 0.0)]),
            LightMode::new(2, array![Complex::new(1e-15, 0.0)]),
            LightMode::new(3, array![Complex::new(1e-15, 0.0)]),
        ],
    );
    drive.add_mode(LightMode::new(0, array![Complex::new(1e20, 0.0)]));
    assert!(bessel_peierls_coeffs(&array![1.0], &drive, 0, 0, 6).is_err());
    let expected = 0.023_587_494_454_597_398_f64; // -2 cos(1e20) J0(200), independent 60-digit reference
    let grid = FloquetTimeGrid::new(&drive, 4096, -2, 2);
    let oracle = peierls_fourier_coeffs(&array![1.0], -2, 2, &drive, &grid);
    let ac_drive = FloquetDrive::with_modes(5.0, drive.modes[..3].to_vec());
    let ac_grid = FloquetTimeGrid::new(&ac_drive, 4096, -2, 2);
    let ac = peierls_fourier_coeffs(&array![1.0], -2, 2, &ac_drive, &ac_grid);
    for (actual, reference) in oracle.iter().zip(&ac) {
        assert!((*actual - Complex::new(0.0, -1e20).exp() * reference).norm() < 1e-11);
    }
    for ham in [
        model
            .floquet_model(&drive, &FloquetTruncation::new(0))
            .unwrap()
            .gen_ham(&k, Gauge::Lattice),
        model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap()
            .gen_ham(&k, Gauge::Lattice),
        model
            .floquet_effective_q_model(&drive, Some(&options), &k)
            .unwrap()
            .gen_ham(&k, Gauge::Lattice),
    ] {
        assert!((ham[[0, 0]] - expected).norm() < 1e-10);
    }
    // The per-mode norm overflows, but the coherent pair cancels. A later
    // distinct carrier prevents the single-harmonic fast path from hiding it.
    let mixed = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(1, array![Complex::new(f64::MAX, f64::MAX)]),
            LightMode::new(2, array![Complex::new(200.0, 0.0)]),
            LightMode::new(1, array![Complex::new(-f64::MAX, -f64::MAX)]),
        ],
    );
    let size = fallback_grid_size(&mixed, &array![1.0], 0, 0);
    assert!(!size.clamped);
    let ham = model
        .sambe_ham_from_model(&k, &mixed, &FloquetTruncation::new(0), Gauge::Lattice)
        .unwrap();
    assert!((ham[[0, 0]] + 2.0 * j0_200).norm() < 1e-11);
    // A truly unrepresentable carrier must not be mistaken for zero bandwidth.
    let overflowing = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(1, array![Complex::new(f64::MAX, f64::MAX)]),
            LightMode::new(2, array![Complex::new(1.0, 0.0)]),
        ],
    );
    assert!(fallback_grid_size(&overflowing, &array![1.0], 0, 0).clamped);
    assert!(
        model
            .floquet_model(&overflowing, &FloquetTruncation::new(0))
            .is_err()
    );
    assert!(
        model
            .floquet_effective_model(&overflowing, Some(&options))
            .is_err()
    );
}

#[test]
fn validation_ignores_inert_dc_and_defers_coherent_norms() {
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    for (length, modes) in [
        (
            2.0,
            vec![LightMode::new(0, array![Complex::new(0.0, 1e308)])],
        ),
        (
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(f64::MAX, f64::MAX)]),
                LightMode::new(1, array![Complex::new(-f64::MAX, -f64::MAX)]),
            ],
        ),
        (
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(f64::MAX, f64::MAX)]),
                LightMode::new(-1, array![Complex::new(-f64::MAX, f64::MAX)]),
            ],
        ),
    ] {
        let mut model = chain_model();
        model.lat[[0, 0]] = length;
        let drive = FloquetDrive::with_modes(5.0, modes);
        let k = array![0.0];
        let trunc = FloquetTruncation::new(1);
        for ham in [model
            .floquet_model(&drive, &trunc)
            .unwrap()
            .gen_ham(&k, Gauge::Lattice)]
        {
            for ((row, col), &value) in ham.indexed_iter() {
                let expected = if row == col {
                    -2.0 + (row as f64 - 1.0) * 5.0
                } else {
                    0.0
                };
                assert!((value - expected).norm() < 1e-11);
            }
        }
        for effective in [
            model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap(),
            model
                .floquet_effective_q_model(&drive, Some(&options), &k)
                .unwrap(),
        ] {
            assert!((effective.gen_ham(&k, Gauge::Lattice)[[0, 0]] + 2.0).norm() < 1e-12);
        }
    }
}

#[test]
fn validation_accepts_the_true_enumeration_budget() {
    let model = chain_model();
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![
            LightMode::new(1_000_000, array![Complex::new(1000.0, 0.0)]),
            LightMode::new(1_000_001, array![Complex::new(1000.0, 0.0)]),
        ],
    );
    let (_, actual_cutoff) =
        bessel_ladder_for_amplitude(1000.0, 5e-13, 6, MAX_BESSEL_ARG_CLOSED_FORM).unwrap();
    assert!(501 * (2 * actual_cutoff as usize + 1) <= MAX_BESSEL_ENUM_WORK);
    let conservative = 4 * (1000 + 64 + 160);
    assert!(501 * (2 * conservative + 1) > MAX_BESSEL_ENUM_WORK);
    let reference = 0.000_614_379_810_418_578_f64; // J0(1000)^2, independent high-precision reference
    let coefficients = bessel_peierls_coeffs(&array![1.0], &drive, -250, 250, 6).unwrap();
    assert!((coefficients[250] - reference).norm() < 1e-12);
    validate_link_resolvability(&model, &drive, -250, 250).unwrap();
    let options = FloquetEffectiveOptions::new()
        .with_order(1)
        .with_harmonic_max(250);
    for effective in [
        model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap(),
        model
            .floquet_effective_q_model(&drive, Some(&options), &array![0.0])
            .unwrap(),
    ] {
        assert!(
            (effective.gen_ham(&array![0.0], Gauge::Lattice)[[0, 0]] + 2.0 * reference).norm()
                < 1e-10
        );
    }
}

#[test]
fn mode_resolved_accumulates_corrections_without_scaling_the_common_baseline() {
    let mut model = chain_model();
    model.set_onsite(&array![6e307], None);
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    let j0 = 0.765_197_686_557_966_6_f64;
    for modes in [
        (1..=4)
            .map(|h| LightMode::new(0, array![Complex::new(0.0, h as f64)]))
            .collect::<Vec<_>>(),
        (1..=4)
            .map(|h| LightMode::new(h, array![Complex::new(1.0, 0.0)]))
            .collect::<Vec<_>>(),
    ] {
        let inert = modes[0].harmonic == 0;
        let drive = FloquetDrive::with_modes(5.0, modes);
        let effective = model
            .floquet_effective_mode_resolved_model(&drive, Some(&options))
            .unwrap();
        let origin = find_R(&effective.hamR, &array![0]).unwrap();
        let link = find_R(&effective.hamR, &array![1]).unwrap();
        assert_eq!(effective.ham[[origin, 0, 0]], Complex::new(6e307, 0.0));
        let expected = if inert { -1.0 } else { 3.0 - 4.0 * j0 };
        assert!((effective.ham[[link, 0, 0]] - expected).norm() < 1e-12);
    }
}

#[test]
fn harmonic_cache_bessel_matches_time_grid_at_sambe_range() {
    // The Sambe path now shares the Bessel backend with the effective-model
    // path.  Compare the two backends over a full Sambe harmonic range for a
    // multi-harmonic drive, i.e. the case that takes the convolution.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.3, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 0, &array![1, 0], None);
    model.add_hop(-0.5, 0, 1, &array![0, 1], None);
    let drive = FloquetDrive::with_modes(
        0.8,
        vec![
            LightMode::new(1, array![Complex::new(0.2, 0.1), Complex::new(0.0, 0.15)]),
            LightMode::new(
                2,
                array![Complex::new(0.05, -0.05), Complex::new(0.02, 0.0)],
            ),
        ],
    );
    let n_max = 2;
    let time_grid = model.floquet_harmonic_cache(
        &drive,
        -2 * n_max,
        2 * n_max,
        &PeierlsFourierMethod::TimeGrid { n_time: 4096 },
    );
    let bessel = model.floquet_harmonic_cache(
        &drive,
        -2 * n_max,
        2 * n_max,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    for (a, b) in time_grid.blocks.iter().zip(bessel.blocks.iter()) {
        assert!((a - b).norm() < 1e-10, "Sambe-range Bessel {b} vs grid {a}");
    }
    // And the Sambe entry point over the same drive is Hermitian.
    let ham = model
        .sambe_ham_from_model(
            &array![0.21, 0.0],
            &drive,
            &FloquetTruncation::new(n_max as isize),
            Gauge::Lattice,
        )
        .unwrap();
    for ((i, j), value) in ham.indexed_iter() {
        assert!((*value - ham[[j, i]].conj()).norm() < 1e-12);
    }
}

#[test]
fn harmonic_cache_bessel_falls_back_for_large_amplitudes() {
    // |a·d| > MAX_BESSEL_ARG must silently fall back to the time grid per
    // link, so the Bessel-method cache still matches the time-grid cache.
    // The (0,1) hopping at R=(0,1) has d = (10, 1), so |a·d| = 130 > 128
    // and the fallback branch must actually execute, while the (1,0)
    // hopping at d = (1, 0) has R = 13 and stays on the ladder.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [10.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 0, &array![1, 0], None);
    model.add_hop(-0.5, 0, 1, &array![0, 1], None);

    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(13.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let n_time = 1024;
    let time_grid_cache =
        model.floquet_harmonic_cache(&drive, -3, 3, &PeierlsFourierMethod::TimeGrid { n_time });
    let bessel_cache = model.floquet_harmonic_cache(
        &drive,
        -3,
        3,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    for (a, b) in time_grid_cache
        .blocks
        .iter()
        .zip(bessel_cache.blocks.iter())
    {
        assert!(
            (a - b).norm() < 1e-10,
            "fallback Bessel cache {b} vs time-grid cache {a}"
        );
    }
}

#[test]
fn harmonic_cache_bessel_fallback_uses_alias_free_grid() {
    // R > MAX_BESSEL_ARG forces the per-link time-grid fallback.  A fixed
    // small n_time aliases the high-order Bessel tails into the wrong bins
    // for l = 100, R = 150 (C_−8 ≈ −J_46(150)-like tails instead of 0;
    // C_0 = J_0(150) survives there only by a divisibility coincidence).
    // The fallback must size its grid to the link's bandwidth
    // (n ≳ 2·|l|·M(R)) and match a 65536-point oracle.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [55.555_555_555_555_56, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 1, &array![0, 1], None);

    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            100,
            array![Complex::new(2.7, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let oracle = model.floquet_harmonic_cache(
        &drive,
        -10,
        10,
        &PeierlsFourierMethod::TimeGrid { n_time: 65536 },
    );
    let bessel_cache = model.floquet_harmonic_cache(
        &drive,
        -10,
        10,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    for (a, b) in oracle.blocks.iter().zip(bessel_cache.blocks.iter()) {
        assert!(
            (a - b).norm() < 1e-10,
            "65536-point oracle {a} vs adaptive fallback {b}"
        );
    }

    // Physical anchor: the block stores t·C_0 with t = -1, so its
    // n = 0 entry on the r = 150 link is -J_0(150).
    let i_r_link = find_R(&model.hamR, &array![0, 1]).unwrap();
    let c0 = bessel_cache.blocks[[bessel_cache.harmonic_index(0), i_r_link, 0, 1]];
    assert!(
        (c0 - Complex::new(-bessel_j(0, 150.0), 0.0)).norm() < 1e-10,
        "C_0 on the r = 150 link should be -J_0(150) (t = -1), got {c0}"
    );
}

#[test]
fn harmonic_cache_bessel_wide_range_matches_time_grid() {
    // The Bessel backend takes no time-sampling parameter. Its adaptive
    // fallback must still match an independent dense DFT when the requested
    // harmonic range exceeds the fallback link's spectral bandwidth.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [10.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 0, &array![1, 0], None);
    model.add_hop(-0.5, 0, 1, &array![0, 1], None); // d = (10, 1), R = 130 > MAX_BESSEL_ARG

    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(13.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let coarse = model.floquet_harmonic_cache(
        &drive,
        -110,
        110,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    let fine = model.floquet_harmonic_cache(
        &drive,
        -110,
        110,
        &PeierlsFourierMethod::TimeGrid { n_time: 65536 },
    );
    assert_eq!(
        coarse.blocks.dim(),
        fine.blocks.dim(),
        "harmonic ranges must match"
    );
    for (a, b) in coarse.blocks.iter().zip(fine.blocks.iter()) {
        assert!(
            (a - b).norm() < 1e-10,
            "adaptive Bessel cache {a} differs from dense DFT {b}"
        );
    }
}

#[test]
fn harmonic_cache_bessel_fallback_resolves_requested_range() {
    // The per-link fallback DFT must also resolve the requested
    // harmonic range, not just the signal bandwidth: an n-point DFT
    // returns Σ_m C_{n+mn}, so bins with |n| >= n_req/2 fold genuine
    // low-order coefficients in.  For the l = 1, R = 130 link below the
    // bandwidth-only grid has n_req = 2·M(130) + 4 ≈ 294 and would alias
    // bins |n| >= 147 — the requested range [-300, 300] therefore spans
    // bins the bandwidth term does not cover.  Sizing the grid to the
    // requested range keeps every requested bin alias-free, and the true
    // C_n for |n| > 147 is exponentially small.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [10.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 0, &array![1, 0], None);
    model.add_hop(-0.5, 0, 1, &array![0, 1], None); // d = (10, 1), R = 130 > MAX_BESSEL_ARG

    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(13.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let cache = model.floquet_harmonic_cache(
        &drive,
        -300,
        300,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    let i_r_link = find_R(&model.hamR, &array![0, 1]).unwrap();
    // n beyond the signal bandwidth (~147 for R = 130, l = 1) must
    // vanish: |C_200| = |J_200(130)| ~ 1e-30, while a bandwidth-only grid
    // would fold J_{200-294}(130) ≈ J_{-94}(130) into that bin.
    for n in 200..=300 {
        let coeff = cache.blocks[[cache.harmonic_index(n), i_r_link, 0, 1]];
        assert!(
            coeff.norm() < 1e-10,
            "C_{n} on the fallback link must be ~0 (beyond the signal \
                 bandwidth), got {coeff}"
        );
    }
}

#[test]
fn floquet_effective_apis_use_explicit_default_options() {
    // All three public APIs use the same defaults, including when the
    // Bessel backend needs its automatically sized time-grid fallback
    // (the (0,1) link has d = (10, 1), so R = 130 > MAX_BESSEL_ARG).
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [10.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-1.0, 0, 0, &array![1, 0], None);
    model.add_hop(-0.5, 0, 1, &array![0, 1], None);
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(13.0, 0.0), Complex::new(0.0, 0.0)]),
            LightMode::new(2, array![Complex::new(0.03, 0.01), Complex::new(0.0, 0.02)]),
        ],
    );
    let defaults = FloquetEffectiveOptions::default();
    assert_eq!(defaults.order, 1);
    assert_eq!(defaults.harmonic_max, 2);
    let options = FloquetEffectiveOptions::new()
        .with_order(1)
        .with_harmonic_max(2);
    let q = array![0.001, 0.002];
    for (implicit, explicit) in [
        (
            model.floquet_effective_model(&drive, None).unwrap(),
            model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap(),
        ),
        (
            model
                .floquet_effective_mode_resolved_model(&drive, None)
                .unwrap(),
            model
                .floquet_effective_mode_resolved_model(&drive, Some(&options))
                .unwrap(),
        ),
        (
            model.floquet_effective_q_model(&drive, None, &q).unwrap(),
            model
                .floquet_effective_q_model(&drive, Some(&options), &q)
                .unwrap(),
        ),
    ] {
        assert_eq!(implicit.hamR, explicit.hamR);
        assert_eq!(implicit.ham, explicit.ham);
    }
}

#[test]
fn floquet_drive_and_method_controls_are_validated_separately() {
    let model = chain_model();
    let q = array![0.0];
    let drive = FloquetDrive::new(1.0);
    for trunc in [
        FloquetTruncation::new(-1),
        FloquetTruncation::new(isize::MIN),
    ] {
        assert!(model.floquet_model(&drive, &trunc).is_err());
    }
    assert!(
        model
            .floquet_effective_model_legacy(&drive, 0, [8], None)
            .is_err()
    );

    for (drive, options) in [
        (FloquetDrive::new(0.0), FloquetEffectiveOptions::new()),
        (FloquetDrive::new(f64::NAN), FloquetEffectiveOptions::new()),
        (
            FloquetDrive::with_modes(
                1.0,
                vec![LightMode::new(
                    1,
                    Array1::from_elem(2, Complex::new(0.1, 0.0)),
                )],
            ),
            FloquetEffectiveOptions::new(),
        ),
        (
            FloquetDrive::with_modes(
                1.0,
                vec![LightMode::new(1, array![Complex::new(f64::NAN, 0.0)])],
            ),
            FloquetEffectiveOptions::new(),
        ),
        (drive.clone(), FloquetEffectiveOptions::new().with_order(3)),
        (
            drive.clone(),
            FloquetEffectiveOptions::new().with_harmonic_max(-1),
        ),
    ] {
        for result in [
            model.floquet_effective_model(&drive, Some(&options)),
            model.floquet_effective_mode_resolved_model(&drive, Some(&options)),
            model.floquet_effective_q_model(&drive, Some(&options), &q),
        ] {
            assert!(result.is_err());
        }
    }
}

#[test]
fn sambe_rejects_overflowing_cutoffs() {
    let model = chain_model();
    let k = array![0.2];
    let static_drive = FloquetDrive::new(1.0);
    for n in [isize::MIN, -1, isize::MAX / 2, isize::MAX] {
        let trunc = FloquetTruncation::new(n);
        assert!(model.floquet_model(&static_drive, &trunc).is_err());
    }
    for n in [isize::MIN, -1, isize::MAX] {
        assert!(FloquetTruncation::new(n).n_sector().is_err());
        assert!(FloquetTruncation::new(n).sectors().is_err());
    }
    // The Sambe backend is grid-free: there is no sampling count to alias
    // it, and the only truncation left is the photon cutoff.
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(100, array![Complex::new(50.0, 0.0)])],
    );
    let sambe = model
        .sambe_ham_from_model(&k, &drive, &FloquetTruncation::new(4), Gauge::Atom)
        .unwrap();
    // Nonzero drive harmonics are multiples of 100, outside this Sambe
    // cutoff. The diagonal is J0(50)*H(k), plus the photon shift.
    let base = model.gen_ham(&k, Gauge::Atom)[[0, 0]] * bessel_j(0, 50.0);
    for ((i, j), value) in sambe.indexed_iter() {
        let expected = if i == j {
            base + (i as f64 - 4.0)
        } else {
            Complex::new(0.0, 0.0)
        };
        assert!((*value - expected).norm() < 1e-10);
    }
}

#[test]
fn sambe_places_added_or_displaced_origin_first() {
    let original = chain_model();
    let drive = FloquetDrive::new(1.0);
    let trunc = FloquetTruncation::new(1);
    for order in [vec![1, 2], vec![1, 2, 0]] {
        let mut model = original.clone();
        model.hamR = model.hamR.select(Axis(0), &order);
        model.ham = model.ham.select(Axis(0), &order);
        let sambe = model.floquet_model(&drive, &trunc).unwrap();
        assert!(sambe.hamR.row(0).iter().all(|r| *r == 0));
        let k = array![0.23];
        let expected = model
            .reference_sambe_ham_onek(&k, &drive, &trunc, Gauge::Atom)
            .unwrap();
        let actual = sambe.gen_ham(&k, Gauge::Atom);
        assert!(
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).norm() < 1e-12)
        );
    }
}

#[test]
fn sambe_rejects_nonfinite_arithmetic_from_finite_inputs() {
    let mut model = chain_model();
    let drive = FloquetDrive::new(f64::MAX);
    for cutoff in [1, 2] {
        model.ham[[0, 0, 0]] = Complex::new(f64::MAX, 0.0);
        let trunc = FloquetTruncation::new(cutoff);
        assert!(model.floquet_model(&drive, &trunc).is_err());
    }
    let mut diagonal =
        Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
    diagonal.add_hop(1.0, 0, 0, &array![1, 1], None);
    let k = array![0.2, 0.1];
    let trunc = FloquetTruncation::new(0);
    // Two modes whose Cartesian components overflow when summed, but whose
    // projection onto the link cancels exactly.  The Bessel backend projects
    // each mode before exponentiating, so this is the undressed static model
    // rather than the overflow the time grid used to report.
    let mode = LightMode::new(
        1,
        array![Complex::new(1e308, 0.0), Complex::new(-1e308, 0.0)],
    );
    let cancelling = FloquetDrive::with_modes(1.0, vec![mode.clone(), mode]);
    let sambe = diagonal.floquet_model(&cancelling, &trunc).unwrap();
    let base = diagonal.gen_ham(&k, Gauge::Atom);
    assert!(
        sambe
            .gen_ham(&k, Gauge::Atom)
            .iter()
            .zip(base.iter())
            .all(|(a, b)| (a - b).norm() < 1e-12),
        "a drive whose projection vanishes must leave the hoppings undressed"
    );
    // A projection that really is huge is refused up front instead: the
    // ladder cannot represent it and the per-link grid cannot resolve it.
    let huge = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(1e308, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    assert!(diagonal.floquet_model(&huge, &trunc).is_err());
}

#[test]
fn sambe_allocation_validation_accepts_an_exactly_representable_drive() {
    let model = chain_model();
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(10000, array![Complex::new(50.0, 0.0)])],
    );
    let trunc = FloquetTruncation::new(0);
    // Validation allocates nothing and no longer reasons about a sampling
    // count: this drive is resolved exactly by the Bessel ladder, and the
    // `2^20` cap applies only to the per-link fallback grid.
    validate_floquet_truncation(&trunc).unwrap();
    validate_sambe_allocation(&model, &drive, &trunc).unwrap();
    assert!(
        model
            .sambe_ham_from_model(&array![0.21], &drive, &trunc, Gauge::Atom)
            .is_ok()
    );
}

#[test]
fn sambe_dc_mode_is_a_pure_phase_shift() {
    let model = chain_model();
    let trunc = FloquetTruncation::new(1);
    let k = array![0.21];
    for amplitude in [0.2, 5000.0] {
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(0, array![Complex::new(amplitude, 17.0)])],
        );
        let dc = model
            .sambe_ham_from_model(&k, &drive, &trunc, Gauge::Atom)
            .unwrap();
        // Only Re(a_0) enters: t(R) -> t(R) exp(-i a_0 R).
        let energy = -2.0 * (TAU * k[0] - amplitude).cos();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { energy + i as f64 - 1.0 } else { 0.0 };
                assert!((dc[[i, j]] - expected).norm() < 2e-12);
            }
        }
    }
}

#[test]
fn weak_sambe_drive_matches_the_analytic_bessel_series() {
    let model = chain_model();
    for amplitude in [0.05, 0.1, 0.5] {
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(1, array![Complex::new(amplitude, 0.0)])],
        );
        let sambe = model
            .sambe_ham_from_model(
                &array![0.21],
                &drive,
                &FloquetTruncation::new(1),
                Gauge::Atom,
            )
            .unwrap();
        let bloch = Complex::new(0.0, TAU * 0.21).exp();
        for i in 0..3 {
            for j in 0..3 {
                let harmonic = i as i32 - j as i32;
                let mut expected = -bessel_j(harmonic as isize, amplitude)
                    * ((-Complex::<f64>::i()).powi(harmonic) * bloch
                        + Complex::<f64>::i().powi(harmonic) * bloch.conj());
                if i == j {
                    expected += (i as f64 - 1.0) * drive.omega0_ev;
                }
                assert!((sambe[[i, j]] - expected).norm() < 1e-12);
            }
        }
    }
}

#[test]
fn saturated_sambe_estimate_uses_amplitude_and_harmonic_bounds() {
    let model = chain_model();
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(1, array![Complex::new(4000.0, 0.0)])],
    );
    let h = model
        .sambe_ham_from_model(
            &array![0.0],
            &drive,
            &FloquetTruncation::new(0),
            Gauge::Atom,
        )
        .unwrap();
    assert!((h[[0, 0]].re + 2.0 * bessel_j(0, 4000.0)).abs() < 1e-10);
    // Both cases need more orders than the per-link grid can resolve.  The
    // second needs three harmonics: one at R = 4000 takes the closed form and
    // two take the enumeration, and both of those are exact.
    for (harmonics, amplitude) in [
        (vec![1_isize], FALLBACK_GRID_MAX as f64),
        (vec![10000, 9999, 9998], 4000.0),
    ] {
        let unresolved = FloquetDrive::with_modes(
            1.0,
            harmonics
                .into_iter()
                .map(|harmonic| LightMode::new(harmonic, array![Complex::new(amplitude, 0.0)]))
                .collect(),
        );
        assert!(
            validate_sambe_allocation(&model, &unresolved, &FloquetTruncation::new(0),).is_err()
        );
    }
}

#[test]
fn effective_models_put_origin_first_and_add_onsite_once() {
    let mut original = chain_model();
    original.set_onsite(&array![0.5], None);
    for support in [vec![0, 1, 2], vec![1, 0, 2], vec![1, 2]] {
        let mut model = original.clone();
        model.ham = model.ham.select(Axis(0), &support);
        model.hamR = model.hamR.select(Axis(0), &support);
        for order in 0..=2 {
            let options = FloquetEffectiveOptions::new()
                .with_order(order)
                .with_harmonic_max(1);
            for modes in [
                vec![],
                vec![LightMode::new(1, array![Complex::new(0.1, 0.0)])],
                vec![
                    LightMode::new(1, array![Complex::new(0.1, 0.0)]),
                    LightMode::new(2, array![Complex::new(0.05, 0.0)]),
                ],
            ] {
                let drive = FloquetDrive::with_modes(5.0, modes);
                for mut effective in [
                    model
                        .floquet_effective_model(&drive, Some(&options))
                        .unwrap(),
                    model
                        .floquet_effective_q_model(&drive, Some(&options), &array![0.01])
                        .unwrap(),
                    model
                        .floquet_effective_mode_resolved_model(&drive, Some(&options))
                        .unwrap(),
                ] {
                    let origin = find_R(&effective.hamR, &array![0]).unwrap();
                    let before = effective.ham[[origin, 0, 0]];
                    effective.add_onsite(&array![1.0], None);
                    assert!((effective.ham[[origin, 0, 0]] - before - 1.0).norm() < 1e-12);
                    assert_eq!(origin, 0);
                    effective.validate().unwrap();
                }
            }
        }
    }
}

#[test]
fn effective_review_regression_rejects_unresolved_fallback() {
    let model = chain_model();
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![LightMode::new(
            FALLBACK_GRID_MAX as isize,
            array![Complex::new(2.0 * MAX_BESSEL_ARG_CLOSED_FORM, 0.0)],
        )],
    );
    let options = FloquetEffectiveOptions::new().with_order(0);
    let grid = fallback_grid_size(&drive, &array![1.0], 0, 0);
    assert!(grid.required > FALLBACK_GRID_MAX);
    // The amplitude exceeds the closed-form cap and the signal bandwidth
    // exceeds the fallback grid. A finite result is not evidence of accuracy.
    assert!(
        model
            .floquet_model(&drive, &FloquetTruncation::new(0))
            .is_err()
    );
    for (entry_point, result) in [
        (
            "ordinary",
            model.floquet_effective_model(&drive, Some(&options)),
        ),
        (
            "zero-q",
            model.floquet_effective_q_model(&drive, Some(&options), &array![0.0]),
        ),
        (
            "mode-resolved",
            model.floquet_effective_mode_resolved_model(&drive, Some(&options)),
        ),
    ] {
        assert!(
            result.is_err(),
            "{entry_point} effective model accepted an unresolved fallback grid"
        );
    }
}

#[test]
fn effective_model_rejects_nonfinite_output_from_finite_input() {
    let mut model = chain_model();
    model.hamR *= 2;
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![LightMode::new(0, array![Complex::new(f64::MAX, 0.0)])],
    );
    assert!(
        model
            .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(0)),)
            .is_err()
    );
    let mut cancellation =
        Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
    cancellation.add_hop(1.0, 0, 0, &array![2, 2], None);
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![LightMode::new(
            0,
            array![Complex::new(f64::MAX, 0.0), Complex::new(-f64::MAX, 0.0)],
        )],
    );
    // Finite operands produce inf-inf in a·d. This must return Err
    // instead of entering Bessel recursion with a NaN amplitude.
    let options = FloquetEffectiveOptions::new().with_order(0);
    assert!(
        cancellation
            .floquet_effective_model(&drive, Some(&options))
            .is_err()
    );
    assert!(
        cancellation
            .floquet_effective_q_model(&drive, Some(&options), &array![0.0, 0.0])
            .is_err()
    );
    assert!(
        cancellation
            .floquet_effective_mode_resolved_model(&drive, Some(&options))
            .is_err()
    );
}

#[test]
fn quasienergy_folding_does_not_overflow_finite_energies() {
    let unit = f64::from_bits(1);
    assert_eq!(fold_quasienergy(0.0, unit), 0.0);
    for (energy, expected) in [(2.0, 2.0), (3.0, -2.0), (-2.0, -2.0), (-3.0, 2.0)] {
        assert_eq!(fold_quasienergy(energy * unit, 5.0 * unit), expected * unit);
    }
    for energy in [unit, -unit, 1.0, -1.0] {
        assert_eq!(fold_quasienergy(energy, f64::MAX), energy);
    }
    for omega in [2.0, f64::MAX] {
        for (energy, expected) in [
            (0.75 * omega, -0.25 * omega),
            (-0.75 * omega, 0.25 * omega),
            (0.5 * omega, -0.5 * omega),
            (-0.5 * omega, -0.5 * omega),
        ] {
            let folded = fold_quasienergy(energy, omega);
            assert!(folded.is_finite());
            assert!((folded / omega - expected / omega).abs() < 1e-15);
        }
    }
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_onsite(&array![0.75 * f64::MAX, 0.1 * f64::MAX], None);
    let sambe = model
        .floquet_model(&FloquetDrive::new(f64::MAX), &FloquetTruncation::new(0))
        .unwrap();
    let mut folded = sambe.solve_band_onek(&array![0.0]).unwrap();
    folded.mapv_inplace(|energy| fold_quasienergy(energy, f64::MAX));
    folded.as_slice_mut().unwrap().sort_by(f64::total_cmp);
    assert!(folded.iter().all(|x| x.is_finite()));
    assert!((folded[0] / f64::MAX + 0.25).abs() < 1e-15);
    assert!((folded[1] / f64::MAX - 0.1).abs() < 1e-15);
}

#[test]
fn floquet_rejects_nonfinite_eigenvalues_before_folding() {
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    // Every matrix entry is finite, but its upper eigenvalue is 2e308.
    model.ham.fill(Complex::new(1e308, 0.0));
    let drive = FloquetDrive::new(1.0);
    let trunc = FloquetTruncation::new(0);
    let sambe = model.floquet_model(&drive, &trunc).unwrap();
    assert!(sambe.solve_band_onek(&array![0.0]).is_err());
    assert!(sambe.solve_band_all_parallel(&array![[0.0]]).is_err());
}

#[test]
fn fallback_grid_size_clamps_and_saturates() {
    // Pin the exact sizing decision (n_req, clamp flag, saturation
    // flag) for the normal, clamp, saturation, and request-dominant
    // regimes.  A broken clamp or saturation detector fails here with
    // exact values instead of a finiteness smoke test.
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let d = array![10.0, 1.0]; // R = 9: a link the harmonic cache now feeds to the ladder, but this helper sizes any r

    // Normal: sized from the signal bandwidth (M(9) = 29).
    let s = fallback_grid_size(&drive, &d, -3, 3);
    assert_eq!(s.n_req, 62); // 2·29 + 4
    assert!(!s.clamped && !s.saturated);

    // Request-dominant: 2·max(|harmonic_min|,|harmonic_max|) + 1 exceeds the
    // bandwidth term.
    let s = fallback_grid_size(&drive, &d, -100, 100);
    assert_eq!(s.n_req, 201);
    assert!(!s.clamped && !s.saturated);

    // Clamp: |l|·M beyond FALLBACK_GRID_MAX.
    let big = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            20000,
            array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let s = fallback_grid_size(&big, &d, -1, 1);
    assert_eq!(s.n_req, 1 << 20);
    assert!(s.clamped);
    assert!(!s.saturated);

    // Saturation: R past MAX_BESSEL_ORDER cannot be certified by the ladder,
    // so sizing switches to the conservative analytic cutoff.
    let sat = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let d_sat = array![4.0 * MAX_BESSEL_ORDER as f64, 1.0];
    let s = fallback_grid_size(&sat, &d_sat, -1, 1);
    assert_eq!(s.n_req, 1 << 20);
    assert!(!s.clamped);
    assert!(s.saturated);
}

#[test]
fn harmonic_cache_bessel_fallback_saturation_matches_oracle() {
    // R = 40000 is past MAX_BESSEL_ORDER, so the adaptive cutoff cannot be
    // certified and the fallback must switch to the conservative analytic
    // bound with the maximum grid.  A broken saturation detector would size
    // the grid from a truncated bandwidth and alias the true band, which the
    // 131072-point oracle resolves.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [40_000.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.add_hop(-0.5, 0, 1, &array![0, 1], None); // d = (40000, 1), R = 40000
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let oracle = model.floquet_harmonic_cache(
        &drive,
        -3,
        3,
        &PeierlsFourierMethod::TimeGrid { n_time: 131_072 },
    );
    let cache = model.floquet_harmonic_cache(
        &drive,
        -3,
        3,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    for (a, b) in oracle.blocks.iter().zip(cache.blocks.iter()) {
        assert!(
            (a - b).norm() < 1e-9,
            "saturated fallback {b} vs 65536-point oracle {a}"
        );
    }
}

/// Build the `n = ±1` harmonic blocks of a spinless model and return
/// them aligned with `model.hamR`, together with the cache itself.
fn commutator_test_blocks<const DIM: usize>(
    model: &Model<false, DIM, NoRMatrix>,
    drive: &FloquetDrive,
) -> (
    FloquetHarmonicCache,
    Vec<Array2<Complex<f64>>>,
    Vec<Array2<Complex<f64>>>,
) {
    let cache = model.floquet_harmonic_cache(
        drive,
        -1,
        1,
        &PeierlsFourierMethod::Bessel {
            cutoff_margin: BESSEL_CUTOFF_MARGIN,
        },
    );
    let n_r = model.hamR.nrows();
    let n1 = cache.harmonic_index(1);
    let nm1 = cache.harmonic_index(-1);
    let a_blocks = (0..n_r)
        .map(|i_r| cache.blocks.slice(s![n1, i_r, .., ..]).to_owned())
        .collect();
    let b_blocks = (0..n_r)
        .map(|i_r| cache.blocks.slice(s![nm1, i_r, .., ..]).to_owned())
        .collect();
    (cache, a_blocks, b_blocks)
}

#[test]
fn real_space_commutator_matches_k_space_commutator() {
    // The real-space convolution must equal the Fourier transform of
    // the k-space commutator [H^(1)(k), H^(-1)(k)] at every k: this
    // validates the two-convolution structure (the naive "P − P†"
    // single-convolution simplification is wrong) against the
    // existing, independently validated harmonic evaluator.
    //
    // The drive must be circularly polarized in 2D: for any single
    // linear mode H^(1)(k) is a scalar multiple of an anti-Hermitian
    // matrix (T_1(R) = t(R)·(−i)·J_1(r)·e^{iθ}·sgn(d)), whose
    // commutator [X, X†] vanishes identically — the first-order van
    // Vleck correction is exactly zero there, and the oracle would
    // be vacuous (it could not catch swapped operand order or
    // transposed-product bugs).
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.35, 0.2]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(-0.3, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);

    // Circular polarization: a = 0.3·(e_x + i·e_y).
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
        )],
    );
    let (cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
    let (comm_blocks, comm_r) = real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();

    let nsta = model.nsta();
    let mut oracle_scale = 0.0_f64;
    for k in [[0.0, 0.0], [0.123, 0.321], [0.5, 0.5], [0.877, 0.111]] {
        let kvec = array![k[0], k[1]];
        let a_k = model.floquet_cached_harmonic_onek(&kvec, 1, Gauge::Lattice, &cache);
        let b_k = model.floquet_cached_harmonic_onek(&kvec, -1, Gauge::Lattice, &cache);
        let oracle = a_k.dot(&b_k) - b_k.dot(&a_k);
        oracle_scale = oracle_scale.max(oracle.iter().fold(0.0, |m, c| m.max(c.norm())));
        let mut from_rs = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for (i_r, row) in comm_r.outer_iter().enumerate() {
            let mut phase_arg = 0.0;
            for a in 0..2 {
                phase_arg += row[a] as f64 * kvec[a];
            }
            let phase = Complex::new(0.0, TAU * phase_arg).exp();
            from_rs.scaled_add(phase, &comm_blocks[i_r]);
        }
        for (a, b) in from_rs.iter().zip(oracle.iter()) {
            assert!(
                (a - b).norm() < 1e-12,
                "k = {k:?}: real-space {a} vs k-space {b}"
            );
        }
    }
    assert!(
        oracle_scale > 1e-6,
        "oracle must be non-trivial (circular 2D drive); otherwise \
             the comparison is vacuous, got {oracle_scale}"
    );
}

#[test]
fn real_space_products_match_explicit_oracle_across_storage_layouts() {
    let a_r = array![[-1_isize], [1]];
    let b_r = array![[-2_isize], [0], [2]];
    let expected_r = array![[-3_isize], [-1], [1], [3]];
    // Size 9 also exercises ndarray's BLAS path for contiguous blocks;
    // size 3 covers its small-matrix path.
    for n in [3, 9] {
        let a = Array3::from_shape_fn((2, n, n), |(r, i, j)| {
            Complex::new(
                ((3 * r + 2 * i + j) % 7) as f64 / 8.0,
                ((r + i + 3 * j) % 5) as f64 / 8.0 - 0.25,
            )
        });
        let b = Array3::from_shape_fn((3, n, n), |(r, i, j)| {
            Complex::new(
                ((r + i + 2 * j) % 5) as f64 / 8.0 - 0.5,
                ((2 * r + 3 * i + j) % 7) as f64 / 8.0,
            )
        });
        let mut a_f = Array3::zeros(a.raw_dim());
        let mut b_f = Array3::zeros(b.raw_dim());
        a_f.swap_axes(1, 2);
        b_f.swap_axes(1, 2);
        a_f.assign(&a);
        b_f.assign(&b);
        let mut a_storage = Array3::zeros((2, 2 * n, 2 * n));
        let mut b_storage = Array3::zeros((3, 2 * n, 2 * n));
        a_storage.slice_mut(s![.., ..;-2, ..;2]).assign(&a);
        b_storage.slice_mut(s![.., ..;-2, ..;2]).assign(&b);
        let a_layouts = [a.view(), a_f.view(), a_storage.slice(s![.., ..;-2, ..;2])];
        let b_layouts = [b.view(), b_f.view(), b_storage.slice(s![.., ..;-2, ..;2])];

        for (sign, operation) in [(-1.0, "commutator"), (1.0, "anticommutator")] {
            // Compute each convolution coefficient by scalar multiplication,
            // independently of ndarray dot/GEMM and the production map.
            let expected = Array3::from_shape_fn((4, n, n), |(r, i, j)| {
                let mut value = Complex::new(0.0, 0.0);
                for ia in 0..a_r.nrows() {
                    for ib in 0..b_r.nrows() {
                        if a_r[[ia, 0]] + b_r[[ib, 0]] == expected_r[[r, 0]] {
                            for k in 0..n {
                                value += a[[ia, i, k]] * b[[ib, k, j]]
                                    + sign * b[[ib, i, k]] * a[[ia, k, j]];
                            }
                        }
                    }
                }
                value
            });
            assert!(expected.iter().any(|value| value.norm() > 1e-4));
            for (a_layout, a_blocks) in a_layouts.iter().enumerate() {
                for (b_layout, b_blocks) in b_layouts.iter().enumerate() {
                    let (actual, support) = real_space_two_product_sum_with_supports(
                        a_blocks, &a_r, b_blocks, &b_r, sign, operation,
                    )
                    .unwrap();
                    assert_eq!(support, expected_r);
                    for (r, block) in actual.iter().enumerate() {
                        for ((i, j), value) in block.indexed_iter() {
                            assert!(
                                (*value - expected[[r, i, j]]).norm() < 1e-12,
                                "{operation}, n={n}, layouts=({a_layout},{b_layout}), \
                                     coefficient=({r},{i},{j}): {value} vs {}",
                                expected[[r, i, j]]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn real_space_commutator_with_different_supports_matches_k_space() {
    // Nested van Vleck terms feed a pair-support inner commutator into a
    // primitive-support outer commutator.  This oracle uses deliberately
    // different, non-Hermitian supports so an implementation that reuses
    // one lookup table for both operands or prematurely symmetrizes the
    // inner result fails visibly.
    let a_r = array![[-1_isize], [2]];
    let b_r = array![[-2_isize], [0], [1]];
    let a_blocks = vec![
        array![
            [Complex::new(0.2, 0.1), Complex::new(-0.3, 0.4)],
            [Complex::new(0.7, -0.2), Complex::new(-0.1, 0.5)]
        ],
        array![
            [Complex::new(-0.4, 0.3), Complex::new(0.6, 0.2)],
            [Complex::new(-0.2, -0.8), Complex::new(0.9, -0.1)]
        ],
    ];
    let b_blocks = vec![
        array![
            [Complex::new(0.5, -0.2), Complex::new(0.1, 0.7)],
            [Complex::new(-0.6, 0.3), Complex::new(0.2, 0.4)]
        ],
        array![
            [Complex::new(-0.1, 0.6), Complex::new(0.8, -0.5)],
            [Complex::new(0.3, 0.2), Complex::new(-0.7, 0.1)]
        ],
        array![
            [Complex::new(0.4, 0.4), Complex::new(-0.2, 0.3)],
            [Complex::new(0.5, -0.6), Complex::new(0.1, -0.2)]
        ],
    ];
    let (comm_blocks, comm_r) =
        real_space_commutator_with_supports(&a_blocks, &a_r, &b_blocks, &b_r).unwrap();

    for k in [0.0, 0.137, 0.5, 0.819] {
        let mut a_k = Array2::<Complex<f64>>::zeros((2, 2));
        let mut b_k = Array2::<Complex<f64>>::zeros((2, 2));
        let mut comm_k = Array2::<Complex<f64>>::zeros((2, 2));
        for (index, r) in a_r.outer_iter().enumerate() {
            let phase = Complex::new(0.0, TAU * k * r[0] as f64).exp();
            a_k.scaled_add(phase, &a_blocks[index]);
        }
        for (index, r) in b_r.outer_iter().enumerate() {
            let phase = Complex::new(0.0, TAU * k * r[0] as f64).exp();
            b_k.scaled_add(phase, &b_blocks[index]);
        }
        for (index, r) in comm_r.outer_iter().enumerate() {
            let phase = Complex::new(0.0, TAU * k * r[0] as f64).exp();
            comm_k.scaled_add(phase, &comm_blocks[index]);
        }
        let oracle = a_k.dot(&b_k) - b_k.dot(&a_k);
        for (actual, expected) in comm_k.iter().zip(oracle.iter()) {
            assert!(
                (actual - expected).norm() < 2e-13,
                "k={k}: real-space {actual} vs k-space {expected}"
            );
        }
    }
}

#[test]
fn real_space_commutator_parallel_matches_serial_and_k_space() {
    // 12 * 13 pairs deliberately crosses PARALLEL_PAIR_THRESHOLD.  Run
    // the same convolution in isolated one- and four-thread pools so the
    // test covers both dispatch paths independently of the test runner's
    // global Rayon configuration.
    let a_r = Array2::from_shape_fn((12, 2), |(i, axis)| match axis {
        0 => i as isize - 6,
        _ => (i * i % 7) as isize - 3,
    });
    let b_r = Array2::from_shape_fn((13, 2), |(i, axis)| match axis {
        0 => 2 * i as isize - 12,
        _ => ((3 * i + 1) % 11) as isize - 5,
    });
    let a_blocks = (0..a_r.nrows())
        .map(|i| {
            let x = i as f64 + 1.0;
            array![
                [
                    Complex::new(0.03 * x, -0.02 * x),
                    Complex::new(-0.01 * x, 0.04 * (x + 1.0))
                ],
                [
                    Complex::new(0.02 * (x + 2.0), 0.01 * x),
                    Complex::new(-0.025 * x, 0.015 * (x - 1.0))
                ]
            ]
        })
        .collect::<Vec<_>>();
    let b_blocks = (0..b_r.nrows())
        .map(|i| {
            let x = i as f64 + 0.5;
            array![
                [
                    Complex::new(-0.02 * x, 0.01 * (x + 1.0)),
                    Complex::new(0.035 * x, -0.015 * x)
                ],
                [
                    Complex::new(-0.04 * (x + 1.0), 0.02 * x),
                    Complex::new(0.01 * x, 0.03 * (x - 2.0))
                ]
            ]
        })
        .collect::<Vec<_>>();

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let serial = serial_pool
        .install(|| real_space_commutator_with_supports(&a_blocks, &a_r, &b_blocks, &b_r).unwrap());
    let parallel = parallel_pool
        .install(|| real_space_commutator_with_supports(&a_blocks, &a_r, &b_blocks, &b_r).unwrap());

    assert_eq!(parallel.1, serial.1);
    for (actual, expected) in parallel.0.iter().zip(serial.0.iter()) {
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).norm() < 2e-14);
        }
    }

    for k in [[0.137, 0.291], [0.419, 0.073], [0.811, 0.557]] {
        let kvec = array![k[0], k[1]];
        let mut a_k = Array2::<Complex<f64>>::zeros((2, 2));
        let mut b_k = Array2::<Complex<f64>>::zeros((2, 2));
        let mut comm_k = Array2::<Complex<f64>>::zeros((2, 2));
        for (index, r) in a_r.outer_iter().enumerate() {
            a_k.scaled_add(bloch_phase::<2, _>(&r, &kvec), &a_blocks[index]);
        }
        for (index, r) in b_r.outer_iter().enumerate() {
            b_k.scaled_add(bloch_phase::<2, _>(&r, &kvec), &b_blocks[index]);
        }
        for (index, r) in parallel.1.outer_iter().enumerate() {
            comm_k.scaled_add(bloch_phase::<2, _>(&r, &kvec), &parallel.0[index]);
        }
        let oracle = a_k.dot(&b_k) - b_k.dot(&a_k);
        for (actual, expected) in comm_k.iter().zip(oracle.iter()) {
            assert!(
                (actual - expected).norm() < 2e-12,
                "k={k:?}: real-space {actual} vs k-space {expected}"
            );
        }
    }
}

#[test]
fn real_space_commutator_rejects_triple_support_overflow() {
    // Two copies of half_max still fit, so this only overflows when that
    // pair support is combined with the third primitive support.
    let half_max = isize::MAX / 2;
    let primitive_r = array![[half_max]];
    let primitive_blocks = vec![array![[Complex::new(0.0, 0.0)]]];
    let (pair_blocks, pair_r) = real_space_commutator_with_supports(
        &primitive_blocks,
        &primitive_r,
        &primitive_blocks,
        &primitive_r,
    )
    .unwrap();
    assert_eq!(pair_r[[0, 0]], isize::MAX - 1);

    let error =
        real_space_commutator_with_supports(&primitive_blocks, &primitive_r, &pair_blocks, &pair_r)
            .unwrap_err();
    assert!(
        error.to_string().contains("Minkowski sum overflow"),
        "unexpected error: {error}"
    );
}

#[test]
fn exact_zero_fast_paths_preserve_underflow_sized_values() {
    // norm_sqr(1e-200) underflows to zero, but the value itself is not
    // zero and can yield a finite result when multiplied by 1e200.
    let support = array![[0_isize]];
    let tiny = 1e-200;
    let huge = 1e200;
    let a_blocks = vec![array![
        [Complex::new(0.0, 0.0), Complex::new(tiny, 0.0)],
        [Complex::new(tiny, 0.0), Complex::new(0.0, 0.0)]
    ]];
    let b_blocks = vec![array![
        [Complex::new(huge, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(-huge, 0.0)]
    ]];
    let (comm_blocks, _) =
        real_space_commutator_with_supports(&a_blocks, &support, &b_blocks, &support).unwrap();
    let expected = matrix_commutator(&a_blocks[0], &b_blocks[0]);
    for (actual, expected) in comm_blocks[0].iter().zip(expected.iter()) {
        assert!((actual - expected).norm() < 1e-14);
    }
    assert!(comm_blocks[0][[0, 1]].norm() > 1.0);

    let mut target = std::collections::BTreeMap::new();
    let scalar_source = vec![array![[Complex::new(tiny, 0.0)]]];
    accumulate_scaled_real_space_blocks(&mut target, &scalar_source, &support, huge).unwrap();
    assert!((target[&vec![0]][[0, 0]].re - 1.0).abs() < 1e-14);
}

#[test]
fn real_space_commutator_vanishes_for_linear_polarization() {
    // For a single linear mode in 1D the first-order van Vleck
    // commutator vanishes exactly (H^(1)(k) is a scalar multiple of
    // an anti-Hermitian matrix).  The wrong "P − P†"
    // single-convolution implementation produces a nonzero answer
    // here, so this pins the double-convolution structure from the
    // other side.
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.35]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1], None);
    model.set_hop(Complex::new(-0.3, 0.1), 0, 1, &array![1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![2], None);

    let drive =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
    let (_cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
    let (comm_blocks, _comm_r) = real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();
    for block in &comm_blocks {
        let max = block.iter().fold(0.0_f64, |m, c| m.max(c.norm()));
        assert!(
            max < 1e-15,
            "linear-polarization commutator must vanish exactly, got {max}"
        );
    }
}

#[test]
fn real_space_commutator_support_and_hermiticity() {
    // hamR = {−2, −1, 1, 2} ⇒ the Minkowski sum is {−4..=4}; and the
    // symmetrized blocks must satisfy comm(R) = comm(−R)† exactly.
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.35]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1], None);
    model.set_hop(-0.3, 0, 1, &array![1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![2], None);

    let drive =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
    let (_cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
    let (comm_blocks, comm_r) = real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();

    // Support: the Minkowski sum of {−2, −1, 1, 2} with itself.
    let expected: Vec<Vec<isize>> = (-4..=4).map(|r| vec![r]).collect();
    let got: Vec<Vec<isize>> = comm_r.outer_iter().map(|row| row.to_vec()).collect();
    assert_eq!(
        got, expected,
        "commutator support must be the Minkowski sum"
    );

    // Hermiticity pairing, exact after symmetrization.
    for i in 0..comm_r.nrows() {
        let j = find_R(&comm_r, &comm_r.row(i).mapv(|v| -v)).unwrap();
        let conj = hermitian_conjugate(&comm_blocks[j]);
        for (a, b) in comm_blocks[i].iter().zip(conj.iter()) {
            assert!(
                (a - b).norm() < 1e-15,
                "comm(R) != comm(−R)† at R = {:?}",
                comm_r.row(i).to_vec()
            );
        }
    }
}

#[test]
fn real_space_commutator_scalar_blocks_vanish() {
    // nsta = 1: matrix products commute, so comm(R) = 0 for every R —
    // a structural check on the (AB) − (BA) accumulation.
    let lat = array![[1.0]];
    let orb = array![[0.0]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1], None);

    let drive =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
    let (_cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
    let (comm_blocks, _comm_r) = real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();
    for block in &comm_blocks {
        assert!(
            block[[0, 0]].norm() < 1e-15,
            "scalar commutator must vanish, got {}",
            block[[0, 0]]
        );
    }
}

#[test]
fn floquet_effective_model_bessel_matches_legacy_bands() {
    // Cross-validate the real-space Bessel path against the legacy
    // k-space path on the same support: the legacy path is given the
    // Bessel output's support as target_hamR (its default — the
    // original hamR — would truncate the longer-range commutator
    // terms) and a fine k-mesh / time grid, so both compute the same
    // H_eff.  The two-mode drive exercises the multi-mode Bessel
    // convolution end to end.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.35, 0.2]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(-0.3, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);

    // Circular l = 1 plus a second harmonic l = 2.
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)]),
            LightMode::new(2, array![Complex::new(0.05, -0.05), Complex::new(0.0, 0.0)]),
        ],
    );
    let n_time = 4096;
    let options = FloquetEffectiveOptions::new().with_harmonic_max(4);
    let bessel = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();

    // Legacy path on the same (automatically determined) support.
    let legacy = model
        .floquet_effective_model_legacy(
            &drive,
            n_time,
            [64, 64],
            Some(&options.with_target_hamR(bessel.hamR.clone())),
        )
        .unwrap();

    for k in [[0.1, 0.2], [0.5, 0.5], [0.9, 0.7]] {
        let kvec = array![k[0], k[1]];
        let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
        let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
        for (a, b) in e_b.iter().zip(e_l.iter()) {
            assert!((a - b).abs() < 1e-8, "k = {k:?}: Bessel {a} vs legacy {b}");
        }
    }
}

/// Independent weak-field one-photon vertex from the exact straight-bond
/// plane-wave integral.  This keeps the finite-q sinc form factor instead
/// of starting from the derivative formula used by the implementation.
fn independent_finite_q_peierls_vertex(
    model: &Model<false, 2, NoRMatrix>,
    k_reduced: &Array1<f64>,
    center_shift_cartesian: &Array1<f64>,
    transfer_cartesian: &Array1<f64>,
    amplitude: &Array1<Complex<f64>>,
) -> Array2<Complex<f64>> {
    let nsta = model.nsta();
    let norb = model.norb();
    let mut out = Array2::<Complex<f64>>::zeros((nsta, nsta));
    for (i_r, r_vec) in model.hamR.outer_iter().enumerate() {
        for i in 0..nsta {
            for j in 0..nsta {
                let hopping = model.ham[[i_r, i, j]];
                if hopping.re == 0.0 && hopping.im == 0.0 {
                    continue;
                }
                let mut d_fractional = Array1::<f64>::zeros(2);
                for axis in 0..2 {
                    d_fractional[axis] = r_vec[axis] as f64 + model.orb[[j % norb, axis]]
                        - model.orb[[i % norb, axis]];
                }
                let d_cartesian = d_fractional.dot(&model.lat);
                let phase_angle =
                    TAU * k_reduced.dot(&d_fractional) + center_shift_cartesian.dot(&d_cartesian);
                let phase = Complex::new(0.0, phase_angle).exp();
                let projection = amplitude
                    .iter()
                    .zip(d_cartesian.iter())
                    .fold(Complex::new(0.0, 0.0), |sum, (a, d)| sum + *a * *d);
                let half_q_dot_d = 0.5 * transfer_cartesian.dot(&d_cartesian);
                let sinc = if half_q_dot_d.abs() < 1.0e-8 {
                    let x2 = half_q_dot_d * half_q_dot_d;
                    1.0 - x2 / 6.0 + x2 * x2 / 120.0
                } else {
                    half_q_dot_d.sin() / half_q_dot_d
                };
                out[[i, j]] += Complex::new(0.0, -0.5) * projection * sinc * hopping * phase;
            }
        }
    }
    out
}

fn independent_q_linear_commutator_odd_part(
    model: &Model<false, 2, NoRMatrix>,
    k_reduced: &Array1<f64>,
    wavevector_cartesian: &Array1<f64>,
    amplitude: &Array1<Complex<f64>>,
    harmonic: isize,
    photon_energy_ev: f64,
) -> Array2<Complex<f64>> {
    let amplitude_conjugate = amplitude.mapv(|value| value.conj());
    let commutator_at = |q_sign: f64| {
        let q = wavevector_cartesian.mapv(|value| q_sign * value);
        let minus_q = q.mapv(|value| -value);
        let half_q = q.mapv(|value| 0.5 * value);
        let minus_half_q = half_q.mapv(|value| -value);
        let positive_minus =
            independent_finite_q_peierls_vertex(model, k_reduced, &minus_half_q, &q, amplitude);
        let negative_minus = independent_finite_q_peierls_vertex(
            model,
            k_reduced,
            &minus_half_q,
            &minus_q,
            &amplitude_conjugate,
        );
        let positive_plus =
            independent_finite_q_peierls_vertex(model, k_reduced, &half_q, &q, amplitude);
        let negative_plus = independent_finite_q_peierls_vertex(
            model,
            k_reduced,
            &half_q,
            &minus_q,
            &amplitude_conjugate,
        );
        (positive_minus.dot(&negative_minus) - negative_plus.dot(&positive_plus))
            * Complex::new(1.0 / ((harmonic as f64) * photon_energy_ev), 0.0)
    };
    (commutator_at(1.0) - commutator_at(-1.0)) * Complex::new(0.5, 0.0)
}

#[test]
fn floquet_effective_q_model_matches_center_momentum_oracle() {
    let lat = array![[1.3, 0.2], [0.1, 0.9]];
    let orb = array![[0.07, 0.11], [0.31, 0.23]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(0.4, 0, 0, &array![0, 0], None);
    model.set_hop(-0.8, 0, 0, &array![1, 0], None);
    model.set_hop(Complex::new(0.35, -0.17), 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(-0.21, 0.09), 1, 1, &array![1, 1], None);

    let amplitude = array![Complex::new(0.24, 0.05), Complex::new(-0.08, 0.19)];
    let drive = FloquetDrive::with_modes(3.7, vec![LightMode::new(1, amplitude.clone())]);
    let options = FloquetEffectiveOptions::new().with_harmonic_max(1);
    let q = array![8.0e-4, -6.0e-4];
    let k = array![0.17, 0.29];

    let baseline = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let finite_q = model
        .floquet_effective_q_model(&drive, Some(&options), &q)
        .unwrap();
    let implemented = finite_q.gen_ham(&k, Gauge::Atom) - baseline.gen_ham(&k, Gauge::Atom);
    let expected =
        independent_q_linear_commutator_odd_part(&model, &k, &q, &amplitude, 1, drive.omega0_ev);
    let error = (&implemented - &expected)
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(
        error < 2.0e-11,
        "finite-q weak-field correction mismatch: error={error:e}\nimplemented={implemented:?}\nexpected={expected:?}"
    );
}

#[test]
fn floquet_effective_q_model_is_coherent_and_preserves_size() {
    let lat = array![[1.0, 0.0], [0.0, 1.4]];
    let orb = array![[0.0, 0.0], [0.2, 0.3]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-0.9, 0, 0, &array![1, 0], None);
    model.set_hop(Complex::new(0.3, 0.2), 0, 1, &array![0, 1], None);

    let amplitude = array![Complex::new(0.2, 0.04), Complex::new(-0.03, 0.16)];
    let split_drive = FloquetDrive::with_modes(
        2.4,
        vec![
            LightMode::new(1, amplitude.mapv(|value| value * 0.3)),
            LightMode::new(1, amplitude.mapv(|value| value * 0.7)),
        ],
    );
    let joined_drive = FloquetDrive::with_modes(2.4, vec![LightMode::new(1, amplitude.clone())]);
    let q = array![0.013, -0.007];
    let split = model
        .floquet_effective_q_model(&split_drive, None, &q)
        .unwrap();
    let joined = model
        .floquet_effective_q_model(&joined_drive, None, &q)
        .unwrap();

    assert_eq!(split.nsta(), model.nsta());
    assert_eq!(split.hamR, joined.hamR);
    for (left, right) in split.ham.iter().zip(joined.ham.iter()) {
        assert!((*left - *right).norm() < 1.0e-12);
    }
}

#[test]
fn floquet_effective_q_model_zero_q_is_exact_uniform_path() {
    let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.0]]);
    let drive = circular_drive(0.35, 1.0, 4.0);
    let options = FloquetEffectiveOptions::new().with_order(2);
    let uniform = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let q_zero = model
        .floquet_effective_q_model(&drive, Some(&options), &array![0.0, 0.0])
        .unwrap();
    assert_eq!(q_zero.hamR, uniform.hamR);
    assert_eq!(q_zero.ham, uniform.ham);
}

#[test]
fn floquet_effective_q_model_rejects_invalid_q_and_nonpositive_harmonic() {
    let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.0]]);
    let drive = circular_drive(0.2, 1.0, 3.0);
    assert!(
        model
            .floquet_effective_q_model(&drive, None, &array![0.01])
            .is_err()
    );
    assert!(
        model
            .floquet_effective_q_model(&drive, None, &array![f64::NAN, 0.0])
            .is_err()
    );

    let negative = FloquetDrive::with_modes(
        3.0,
        vec![LightMode::new(
            -1,
            array![Complex::new(0.2, 0.0), Complex::new(0.0, -0.2)],
        )],
    );
    let error = model
        .floquet_effective_q_model(&negative, None, &array![0.01, 0.0])
        .unwrap_err();
    assert!(error.to_string().contains("positive temporal harmonics"));
}

#[test]
fn floquet_effective_q_model_rejects_finite_arithmetic_overflow() {
    let overflow_drive = FloquetDrive::with_modes(
        2.0,
        vec![
            LightMode::new(1, array![Complex::new(0.75 * f64::MAX, 0.0)]),
            LightMode::new(1, array![Complex::new(0.75 * f64::MAX, 0.0)]),
        ],
    );
    let error = coherent_positive_harmonic_amplitudes::<1>(&overflow_drive, 1)
        .expect_err("coherent amplitude overflow must be rejected");
    assert!(error.to_string().contains("amplitude sum overflowed"));

    let lat = array![[2.0]];
    let orb = array![[0.0]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1], None);
    let drive =
        FloquetDrive::with_modes(2.0, vec![LightMode::new(1, array![Complex::new(0.1, 0.0)])]);
    let error = model
        .floquet_effective_q_model(&drive, None, &array![f64::MAX])
        .expect_err("q·d overflow must be rejected");
    assert!(error.to_string().contains("q·d overflowed"));

    let subnormal_scale = q_linear_frequency_scale(2, 1.0e308);
    assert!(
        subnormal_scale.is_finite() && subnormal_scale < 0.0,
        "representable inverse-frequency scale was lost: {subnormal_scale}"
    );
}

#[test]
fn floquet_effective_q_model_distinct_harmonics_add_at_linear_q_order() {
    let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.2]]);
    let mode_one = LightMode::new(
        1,
        array![Complex::new(0.18, 0.03), Complex::new(-0.02, 0.12)],
    );
    let mode_two = LightMode::new(
        2,
        array![Complex::new(0.07, -0.01), Complex::new(0.04, 0.05)],
    );
    let combined_drive = FloquetDrive::with_modes(3.0, vec![mode_one.clone(), mode_two.clone()]);
    let drive_one = FloquetDrive::with_modes(3.0, vec![mode_one]);
    let drive_two = FloquetDrive::with_modes(3.0, vec![mode_two]);
    let options = FloquetEffectiveOptions::new().with_harmonic_max(2);
    let q = array![0.009, -0.004];
    let k = array![0.21, 0.37];

    let q_delta = |drive: &FloquetDrive| {
        let base = model
            .floquet_effective_model(drive, Some(&options))
            .unwrap();
        let finite = model
            .floquet_effective_q_model(drive, Some(&options), &q)
            .unwrap();
        finite.gen_ham(&k, Gauge::Atom) - base.gen_ham(&k, Gauge::Atom)
    };
    let combined = q_delta(&combined_drive);
    let expected = q_delta(&drive_one) + q_delta(&drive_two);
    let error = (&combined - &expected)
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    assert!(error < 1.0e-12, "distinct-harmonic q correction: {error:e}");

    let cutoff_one = FloquetEffectiveOptions::new().with_harmonic_max(1);
    let base_two = model
        .floquet_effective_model(&drive_two, Some(&cutoff_one))
        .unwrap();
    let finite_two = model
        .floquet_effective_q_model(&drive_two, Some(&cutoff_one), &q)
        .unwrap();
    assert_eq!(finite_two.hamR, base_two.hamR);
    assert_eq!(finite_two.ham, base_two.ham);
}

#[test]
fn floquet_effective_q_model_spinful_matches_two_spinless_copies() {
    let lat = array![[1.1, 0.2], [0.0, 0.9]];
    let orb = array![[0.03, 0.07], [0.28, 0.19]];
    let mut spinless = Model::<false, 2>::tb_model(lat.clone(), orb.clone(), None).unwrap();
    let mut spinful = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    for model_hop in [
        (Complex::new(-0.7, 0.0), 0, 0, array![1, 0]),
        (Complex::new(0.24, -0.13), 0, 1, array![0, 1]),
        (Complex::new(-0.18, 0.04), 1, 1, array![1, 1]),
    ] {
        let (value, i, j, r) = model_hop;
        spinless.set_hop(value, i, j, &r, None);
        spinful.set_hop(value, i, j, &r, None);
    }
    let drive = FloquetDrive::with_modes(
        2.8,
        vec![LightMode::new(
            1,
            array![Complex::new(0.16, 0.02), Complex::new(-0.03, 0.14)],
        )],
    );
    let q = array![0.011, -0.006];
    let k = array![0.23, 0.31];
    let h0 = spinless
        .floquet_effective_q_model(&drive, None, &q)
        .unwrap()
        .gen_ham(&k, Gauge::Atom);
    let hs = spinful
        .floquet_effective_q_model(&drive, None, &q)
        .unwrap()
        .gen_ham(&k, Gauge::Atom);
    let norb = spinless.norb();
    for spin in 0..2 {
        for i in 0..norb {
            for j in 0..norb {
                assert!((hs[[spin * norb + i, spin * norb + j]] - h0[[i, j]]).norm() < 1e-12);
                assert!(hs[[spin * norb + i, (1 - spin) * norb + j]].norm() < 1e-12);
            }
        }
    }
}

#[test]
fn floquet_mode_resolved_matches_sum_of_isolated_corrections() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.22, 0.14]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(0.9, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.4, -0.2), 1, 1, &array![1, 1], None);

    let mode1 = LightMode::new(1, array![Complex::new(0.28, 0.0), Complex::new(0.0, 0.21)]);
    let mode2 = LightMode::new(
        2,
        array![Complex::new(0.04, 0.02), Complex::new(0.03, -0.01)],
    );
    let drive = FloquetDrive::with_modes(1.0, vec![mode1.clone(), mode2.clone()]);
    let options = FloquetEffectiveOptions::new().with_harmonic_max(3);

    let mode1_only = model
        .floquet_effective_model(&FloquetDrive::with_modes(1.0, vec![mode1]), Some(&options))
        .unwrap();
    let mode2_only = model
        .floquet_effective_model(&FloquetDrive::with_modes(1.0, vec![mode2]), Some(&options))
        .unwrap();
    let recombined = model
        .floquet_effective_mode_resolved_model(&drive, Some(&options))
        .unwrap();

    for k in [array![0.13, 0.27], array![0.31, 0.08]] {
        let h0 = model.gen_ham(&k, Gauge::Lattice);
        let expected =
            mode1_only.gen_ham(&k, Gauge::Lattice) + mode2_only.gen_ham(&k, Gauge::Lattice) - h0;
        let actual = recombined.gen_ham(&k, Gauge::Lattice);
        let diff = actual - expected;
        assert!(
            diff.iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                .sqrt()
                < 1e-11,
            "mode-diagonal correction sum mismatch: {diff:?}"
        );
    }
}

#[test]
fn floquet_mode_resolved_single_mode_matches_ordinary_model() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.22, 0.14]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(0.9, 0, 1, &array![0, 1], None);

    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.28, 0.0), Complex::new(0.0, 0.21)],
        )],
    );
    let options = FloquetEffectiveOptions::new().with_harmonic_max(3);

    let ordinary = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let resolved = model
        .floquet_effective_mode_resolved_model(&drive, Some(&options))
        .unwrap();
    assert_eq!(resolved.hamR, ordinary.hamR);
    assert_eq!(resolved.ham, ordinary.ham);
}

#[test]
fn hermiticity_midpoint_preserves_finite_extremes() {
    for value in [f64::from_bits(1), 1.0e308] {
        let mut ham = Array3::from_elem((1, 1, 1), Complex::new(value, 0.0));
        let ham_r = Array2::<isize>::zeros((1, 1));
        enforce_real_space_hermiticity(&mut ham, &ham_r).unwrap();
        assert_eq!(ham[[0, 0, 0]].re.to_bits(), value.to_bits());
        assert_eq!(ham[[0, 0, 0]].im.to_bits(), 0.0_f64.to_bits());
    }
}

#[test]
fn floquet_mode_resolved_empty_and_zero_modes_return_static_model() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.22, 0.14]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(0.9, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.4, -0.2), 1, 1, &array![1, 1], None);

    let empty = FloquetDrive::new(1.0);
    let zero = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    for drive in [&empty, &zero] {
        let resolved = model
            .floquet_effective_mode_resolved_model(drive, None)
            .unwrap();
        assert_eq!(resolved.hamR, model.hamR);
        assert_eq!(resolved.ham, model.ham);
    }
}

#[test]
fn floquet_effective_model_order_two_matches_legacy_and_has_triple_support() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.35, 0.2]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(-0.3, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);
    let drive = FloquetDrive::with_modes(
        1.7,
        vec![
            LightMode::new(1, array![Complex::new(0.28, 0.03), Complex::new(0.0, 0.24)]),
            LightMode::new(
                2,
                array![Complex::new(0.06, -0.04), Complex::new(0.02, 0.01)],
            ),
        ],
    );
    let n_time = 4096;
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(2);
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let parallel_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let serial = serial_pool.install(|| {
        model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap()
    });
    let real_space = parallel_pool.install(|| {
        model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap()
    });

    assert_eq!(real_space.hamR, serial.hamR);
    for (actual, expected) in real_space.ham.iter().zip(serial.ham.iter()) {
        assert!(
            (actual - expected).norm() < 5e-13,
            "parallel {actual} vs serial {expected}"
        );
    }

    // Through 1/ω² the deterministic support is the union of the
    // primitive, double-, and triple-Minkowski supports.
    let mut expected_support = std::collections::BTreeSet::<Vec<isize>>::new();
    for r1 in model.hamR.outer_iter() {
        expected_support.insert(r1.to_vec());
        for r2 in model.hamR.outer_iter() {
            expected_support.insert(r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect());
            for r3 in model.hamR.outer_iter() {
                expected_support.insert(
                    r1.iter()
                        .zip(r2.iter())
                        .zip(r3.iter())
                        .map(|((a, b), c)| a + b + c)
                        .collect(),
                );
            }
        }
    }
    let actual_support: Vec<Vec<isize>> = real_space
        .hamR
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let mut expected_support = Vec::from_iter(expected_support);
    expected_support.sort_by_key(|row| row.iter().any(|&r| r != 0));
    assert_eq!(actual_support, expected_support);

    let legacy = model
        .floquet_effective_model_legacy(
            &drive,
            n_time,
            [32, 32],
            Some(&options.clone().with_target_hamR(real_space.hamR.clone())),
        )
        .unwrap();
    for k in [[0.07, 0.19], [0.31, 0.43], [0.73, 0.61]] {
        let kvec = array![k[0], k[1]];
        let from_real_space = real_space.gen_ham(&kvec, Gauge::Lattice);
        let from_legacy = legacy.gen_ham(&kvec, Gauge::Lattice);
        for (actual, expected) in from_real_space.iter().zip(from_legacy.iter()) {
            assert!(
                (actual - expected).norm() < 2e-9,
                "k={k:?}: real-space {actual} vs legacy {expected}"
            );
        }
    }

    // The final signed sum, unlike its individual nested terms, is
    // Hermitian in real space.
    for i_r in 0..real_space.hamR.nrows() {
        let opposite = find_R(
            &real_space.hamR,
            &real_space.hamR.row(i_r).mapv(|value| -value),
        )
        .unwrap();
        let expected =
            hermitian_conjugate(&real_space.ham.index_axis(Axis(0), opposite).to_owned());
        for (actual, expected) in real_space
            .ham
            .index_axis(Axis(0), i_r)
            .iter()
            .zip(expected.iter())
        {
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn floquet_effective_order_two_improves_high_frequency_scaling() {
    // Independent oracle: diagonalize the enlarged Sambe Hamiltonian and
    // compare its central-sector quasienergies with the same-size van
    // Vleck models.  With fixed Fourier blocks, an order-1 truncation has
    // O(Ω^-2) spectral error while order 2 has O(Ω^-3) error.
    let model = two_band_qwz(0.9, 0.7, 0.35, 0.4, [[1.0, 0.0], [0.0, 1.0]]);
    let kvec = array![0.173, 0.287];
    let mode = LightMode::new(
        1,
        array![Complex::new(0.23, 0.04), Complex::new(-0.02, 0.19)],
    );
    let trunc = FloquetTruncation::new(8);
    let base_options = FloquetEffectiveOptions::new().with_harmonic_max(8);
    let mut first_order_errors = Vec::new();
    let mut second_order_errors = Vec::new();

    for omega in [20.0, 40.0, 80.0] {
        let drive = FloquetDrive::with_modes(omega, vec![mode.clone()]);
        let sambe = model
            .sambe_ham_from_model(&kvec, &drive, &trunc, Gauge::Lattice)
            .unwrap();
        let exact_all = eigvalsh_v(&sambe, UPLO::Lower).unwrap();
        let exact: Vec<f64> = exact_all
            .iter()
            .copied()
            .filter(|energy| energy.abs() < 0.5 * omega)
            .collect();
        assert_eq!(
            exact.len(),
            model.nsta(),
            "expected one central quasienergy per static state at Ω={omega}"
        );

        let first = model
            .floquet_effective_model(&drive, Some(&base_options.clone().with_order(1)))
            .unwrap()
            .solve_band_onek(&kvec)
            .unwrap();
        let second = model
            .floquet_effective_model(&drive, Some(&base_options.clone().with_order(2)))
            .unwrap()
            .solve_band_onek(&kvec)
            .unwrap();
        let first_error = first
            .iter()
            .zip(exact.iter())
            .map(|(approx, exact)| (approx - exact).abs())
            .fold(0.0_f64, f64::max);
        let second_error = second
            .iter()
            .zip(exact.iter())
            .map(|(approx, exact)| (approx - exact).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            second_error < first_error,
            "order 2 must improve the central quasienergies at Ω={omega}: \
                 order-1 error={first_error:e}, order-2 error={second_error:e}"
        );
        first_order_errors.push(first_error);
        second_order_errors.push(second_error);
    }

    for pair in first_order_errors.windows(2) {
        assert!(
            pair[0] / pair[1] > 3.5,
            "order-1 error should scale as Ω^-2, got {pair:?}"
        );
    }
    for pair in second_order_errors.windows(2) {
        assert!(
            pair[0] / pair[1] > 6.5,
            "order-2 error should scale as Ω^-3, got {pair:?}"
        );
    }
}

#[test]
fn floquet_effective_model_bessel_order_zero() {
    // order = 0 keeps only the Peierls-dressed static model T_0(R);
    // the support must stay the original hamR and the bands must
    // match the legacy order-0 path.
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.35]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1], None);
    model.set_hop(Complex::new(-0.3, 0.1), 0, 1, &array![1], None);

    let drive =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
    let n_time = 4096;
    let bessel = model
        .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(0)))
        .unwrap();
    // order-0 support = input hamR.
    assert_eq!(bessel.hamR.nrows(), model.hamR.nrows());
    let legacy = model
        .floquet_effective_model_legacy(
            &drive,
            n_time,
            [64],
            Some(&FloquetEffectiveOptions::new().with_order(0)),
        )
        .unwrap();
    for k in [0.0, 0.23, 0.5, 0.71] {
        let kvec = array![k];
        let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
        let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
        for (a, b) in e_b.iter().zip(e_l.iter()) {
            assert!((a - b).abs() < 1e-8, "k = {k}: Bessel {a} vs legacy {b}");
        }
    }
}

#[test]
fn floquet_effective_model_bessel_support_and_hermiticity() {
    // First-order support = Minkowski sum of hamR with itself, origin
    // first and then lexicographic order, and the output blocks satisfy
    // T(R) = T(−R)† exactly.
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.35, 0.2]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(-0.3, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
        )],
    );
    let bessel = model.floquet_effective_model(&drive, None).unwrap();

    // Expected support: the Minkowski sum of the input hamR with
    // itself, with the origin first and other rows lexicographically ordered.
    let mut expected = std::collections::BTreeSet::<Vec<isize>>::new();
    for r1 in model.hamR.outer_iter() {
        for r2 in model.hamR.outer_iter() {
            expected.insert(r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect());
        }
    }
    let got: Vec<Vec<isize>> = bessel.hamR.outer_iter().map(|row| row.to_vec()).collect();
    let expected: Vec<_> = std::iter::once(vec![0, 0])
        .chain(expected.into_iter().filter(|row| row != &[0, 0]))
        .collect();
    assert_eq!(got, expected, "support mismatch");

    // Exact Hermiticity pairing.
    for i in 0..bessel.hamR.nrows() {
        let j = find_R(&bessel.hamR, &bessel.hamR.row(i).mapv(|v| -v)).unwrap();
        let conj = hermitian_conjugate(&bessel.ham.index_axis(Axis(0), j).to_owned());
        for (a, b) in bessel.ham.index_axis(Axis(0), i).iter().zip(conj.iter()) {
            assert!((a - b).norm() < 1e-15, "T(R) != T(−R)† at row {i}");
        }
    }
}

#[test]
fn floquet_effective_model_bessel_rejects_invalid_options() {
    let lat = array![[1.0]];
    let orb = array![[0.0]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1], None);
    let drive =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
    let n_time = 512;

    // order 2 is supported; higher orders are rejected.
    assert!(
        model
            .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(2)))
            .is_ok()
    );
    assert!(
        model
            .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(3)))
            .is_err()
    );
    // target_hamR is rejected on the real-space path.
    let target = array![[-1], [0], [1]];
    assert!(
        model
            .floquet_effective_model(
                &drive,
                Some(&FloquetEffectiveOptions::new().with_target_hamR(target))
            )
            .is_err()
    );
    // negative harmonic_max.
    assert!(
        model
            .floquet_effective_model(
                &drive,
                Some(&FloquetEffectiveOptions::new().with_harmonic_max(-1))
            )
            .is_err()
    );

    // An order-2 symmetric cache needs 4*harmonic_max+1 harmonics.  Reject an
    // unrepresentable inclusive range before ndarray arithmetic/allocation.
    let range_error = model
        .floquet_effective_model(
            &drive,
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(isize::MAX / 2),
            ),
        )
        .unwrap_err();
    assert!(
        range_error
            .to_string()
            .contains("too large to index safely"),
        "unexpected error: {range_error}"
    );

    // A finite positive frequency can still have a non-representable
    // inverse-square factor.  Exact-zero scalar commutators remain valid.
    let tiny_frequency_drive = FloquetDrive::with_modes(
        1e-200,
        vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])],
    );
    let scalar_result = model
        .floquet_effective_model(
            &tiny_frequency_drive,
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(1),
            ),
        )
        .unwrap();
    assert!(
        scalar_result
            .ham
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    // A genuinely nonzero nested commutator cannot be represented with
    // the same scale and must return an error instead of NaN/Inf blocks.
    let matrix_model = two_band_qwz(0.9, 0.7, 0.35, 0.4, [[1.0, 0.0], [0.0, 1.0]]);
    let matrix_drive = FloquetDrive::with_modes(
        1e-200,
        vec![LightMode::new(
            1,
            array![Complex::new(0.23, 0.04), Complex::new(-0.02, 0.19)],
        )],
    );
    let scale_error = matrix_model
        .floquet_effective_model(
            &matrix_drive,
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(1),
            ),
        )
        .unwrap_err();
    assert!(
        scale_error.to_string().contains("frequency scaling factor"),
        "unexpected error: {scale_error}"
    );
    model
        .floquet_effective_model_legacy(
            &tiny_frequency_drive,
            n_time,
            [8],
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(1),
            ),
        )
        .unwrap();
    let legacy_scale_error = matrix_model
        .floquet_effective_model_legacy(
            &matrix_drive,
            n_time,
            [4, 4],
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(1),
            ),
        )
        .unwrap_err();
    assert!(
        legacy_scale_error
            .to_string()
            .contains("frequency scaling factor"),
        "unexpected legacy error: {legacy_scale_error}"
    );

    // harmonic_max=0 never evaluates inverse-frequency corrections and remains
    // well-defined even at the same tiny frequency.
    let zero_cutoff = model
        .floquet_effective_model(
            &tiny_frequency_drive,
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(0),
            ),
        )
        .unwrap();
    assert!(
        zero_cutoff
            .ham
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

#[test]
fn floquet_effective_model_spinful_matches_legacy() {
    // Spinful 2D model: the real-space path must be agnostic to the
    // spin structure (blocks are nsta x nsta with nsta = 2·norb).
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.35, 0.2]];
    let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(-0.3, 0, 1, &array![0, 1], SpinDirection::X);
    model.set_hop(
        Complex::new(0.1, -0.2),
        1,
        1,
        &array![1, 1],
        SpinDirection::Z,
    );

    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
        )],
    );
    let n_time = 4096;
    let bessel = model.floquet_effective_model(&drive, None).unwrap();
    let legacy = model
        .floquet_effective_model_legacy(
            &drive,
            n_time,
            [32, 32],
            Some(&FloquetEffectiveOptions::new().with_target_hamR(bessel.hamR.clone())),
        )
        .unwrap();
    for k in [[0.1, 0.2], [0.5, 0.5]] {
        let kvec = array![k[0], k[1]];
        let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
        let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
        for (a, b) in e_b.iter().zip(e_l.iter()) {
            assert!((a - b).abs() < 1e-8, "k = {k:?}: Bessel {a} vs legacy {b}");
        }
    }
}

#[test]
fn floquet_effective_model_no_drive_matches_static_bands() {
    // Empty drive: the real-space path returns T_0 = static blocks
    // plus exact-zero commutator blocks on the Minkowski support —
    // the bands must equal the static model's.
    let model = chain_model();
    let drive = FloquetDrive::new(0.9);
    let effective = model.floquet_effective_model(&drive, None).unwrap();

    let k = arr1(&[0.271]);
    for gauge in [Gauge::Lattice, Gauge::Atom] {
        let from_effective = effective.gen_ham(&k, gauge);
        let from_static = model.gen_ham(&k, gauge);
        let mut max_diff = 0.0f64;
        for i in 0..from_effective.nrows() {
            for j in 0..from_effective.ncols() {
                max_diff = max_diff.max((from_effective[[i, j]] - from_static[[i, j]]).norm());
            }
        }
        assert!(
            max_diff < 1e-12,
            "no-drive effective mismatch in {gauge:?}: {max_diff:e}"
        );
    }

    // Documented support contract: even for an empty drive the
    // Minkowski-blown support with exact-zero commutator blocks is
    // retained — chain_model's hamR = {-1, 0, 1} gives {-2..=2}.
    let got: Vec<Vec<isize>> = effective
        .hamR
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    let expected: Vec<Vec<isize>> = [0, -2, -1, 1, 2].into_iter().map(|r| vec![r]).collect();
    assert_eq!(
        got, expected,
        "empty-drive support must be the Minkowski union"
    );

    // Empty harmonics take the exact-zero fast path: even when 1/W^2
    // overflows, order 2 must remain the static model rather than forming
    // inf*0 = NaN.  Its documented support is still the triple sum.
    let tiny_drive = FloquetDrive::new(1e-200);
    let order_two = model
        .floquet_effective_model(
            &tiny_drive,
            Some(
                &FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(1),
            ),
        )
        .unwrap();
    assert!(
        order_two
            .ham
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let order_two_support: Vec<Vec<isize>> = order_two
        .hamR
        .outer_iter()
        .map(|row| row.to_vec())
        .collect();
    assert_eq!(
        order_two_support,
        [0, -3, -2, -1, 1, 2, 3]
            .into_iter()
            .map(|r| vec![r])
            .collect::<Vec<_>>()
    );

    // Equivalent non-dynamic representations stay finite at a large cutoff.
    // Exact-zero/static entries use the shortcut; coherent ±frequency
    // cancellation is certified after projection by the scalar backend.
    let non_dynamic_drives = [
        FloquetDrive::with_modes(
            1e-200,
            vec![LightMode::new(1, array![Complex::new(0.0, 0.0)])],
        ),
        FloquetDrive::with_modes(
            1e-200,
            vec![
                LightMode::new(1, array![Complex::new(0.3, -0.2)]),
                LightMode::new(-1, array![Complex::new(-0.3, -0.2)]),
            ],
        ),
        FloquetDrive::with_modes(
            1e-200,
            vec![LightMode::new(0, array![Complex::new(0.2, 7.0)])],
        ),
    ];
    let large_cutoff = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(1000);
    for equivalent_drive in non_dynamic_drives {
        let effective = model
            .floquet_effective_model(&equivalent_drive, Some(&large_cutoff))
            .unwrap();
        assert!(
            effective
                .ham
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        assert_eq!(
            effective
                .hamR
                .outer_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>(),
            order_two_support
        );
    }
}

#[test]
fn floquet_effective_model_bessel_matches_legacy() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.35, 0.2]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0, 0, 0, &array![1, 0], None);
    model.set_hop(-0.3, 0, 1, &array![0, 1], None);
    model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
        )],
    );
    let n_time = 512;

    let bessel = model.floquet_effective_model(&drive, None).unwrap();
    let legacy_options = FloquetEffectiveOptions::new().with_target_hamR(bessel.hamR.clone());
    let legacy = model
        .floquet_effective_model_legacy(&drive, n_time, [128, 128], Some(&legacy_options))
        .unwrap();

    // Compare the real-space Bessel result with the independent k-space DFT.
    let kvec = array![0.37, 0.19];
    let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
    let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower).unwrap();
    for (a, b) in e_b.iter().zip(e_l.iter()) {
        assert!((a - b).abs() < 1e-8, "Bessel {a} vs legacy {b}");
    }
}

/// Honeycomb graphene: two sublattices at the origin, nearest-neighbour
/// hopping `j` along `e_0 = (0,1)a`, `e_1 = (−√3/2, −1/2)a`,
/// `e_2 = (√3/2, −1/2)a` (with `a = 1`), using the triangular Bravais
/// basis `a1 = (√3/2, 1/2)`, `a2 = (√3/2, −1/2)`.
fn graphene_model(j: f64) -> Model<false, 2, NoRMatrix> {
    let lat = array![[3f64.sqrt() / 2.0, 0.5], [3f64.sqrt() / 2.0, -0.5],];
    let orb = array![[0.0, 0.0], [0.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    // e_0 = a1 − a2, e_1 = −a1, e_2 = a2.
    for r in [[1, -1], [-1, 0], [0, 1]] {
        model.set_hop(j, 0, 1, &array![r[0], r[1]], None);
    }
    model
}

/// Literature Fourier component for right-handed circular light
/// (arXiv:1511.00755 conventions):
/// `q_n(k) = J·J_n(α)·Σ_l e^{−ik·e_l} e^{i2πnl/3}` with the matrix
/// `H_n = [[0, q_n], [n*_{−n}, 0]]`.  The literature `k` is the
/// Cartesian wavevector; with fractional `k` the bond phase is
/// `k_cart·e_l = 2π·k_frac·R_int` where `R_int` is the integer
/// lattice vector of the bond (the non-orthonormal `lat` matrix must
/// not be dropped).
fn graphene_q_n_lit(n: isize, k: &[f64; 2], j: f64, alpha: f64) -> Complex<f64> {
    // Integer bond vectors: e_0 = a1 − a2, e_1 = −a1, e_2 = a2.
    let e_int = [[1, -1], [-1, 0], [0, 1]];
    let mut sum = Complex::new(0.0, 0.0);
    for (l, r) in e_int.iter().enumerate() {
        let phase =
            -TAU * (k[0] * r[0] as f64 + k[1] * r[1] as f64) + TAU * (n as f64) * (l as f64) / 3.0;
        sum += Complex::new(0.0, phase).exp();
    }
    j * bessel_j(n, alpha) * sum
}

/// Right-handed circular drive `a = α·(1, i)`, which reproduces the
/// literature `a(t)·e_l = α·sin(ωt − 2πl/3)`.
fn graphene_circular_drive(alpha: f64) -> FloquetDrive {
    FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            1,
            array![Complex::new(alpha, 0.0), Complex::new(0.0, alpha)],
        )],
    )
}

#[test]
fn graphene_harmonics_match_literature_fourier_components() {
    // Benchmark A: H_n(k) elementwise against the literature Fourier
    // components.  Convention mapping: this library's gen_ham uses
    // e^{+i2πk·R}, the literature uses e^{−ik·R}, so our H_n(k)
    // equals the literature H_n(−k) — asserted elementwise to 1e-12.
    // This pins the Peierls phase sign, the Bessel phase δ_l
    // (including the (−1)^n structure from e^{±in(θ_l+π)}), and the
    // H_{−n} = H_n† pairing.
    let j = -1.0;
    let model = graphene_model(j);
    for alpha in [0.3, 0.8] {
        let drive = graphene_circular_drive(alpha);
        let cache = model.floquet_harmonic_cache(
            &drive,
            -4,
            4,
            &PeierlsFourierMethod::Bessel {
                cutoff_margin: BESSEL_CUTOFF_MARGIN,
            },
        );
        for k in [[0.13, 0.21], [0.5, 0.5], [0.87, 0.11]] {
            let kvec = array![k[0], k[1]];
            let k_neg = [-k[0], -k[1]];
            for n in -4..=4 {
                let h_n = model.floquet_cached_harmonic_onek(&kvec, n, Gauge::Lattice, &cache);
                let lit = array![
                    [
                        Complex::new(0.0, 0.0),
                        graphene_q_n_lit(n, &k_neg, j, alpha),
                    ],
                    [
                        graphene_q_n_lit(-n, &k_neg, j, alpha).conj(),
                        Complex::new(0.0, 0.0),
                    ],
                ];
                for (a, b) in h_n.iter().zip(lit.iter()) {
                    assert!(
                        (a - b).norm() < 1e-12,
                        "alpha = {alpha}, k = {k:?}, n = {n}: {a} vs {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn graphene_dirac_gap_matches_exact_rotating_frame() {
    // Benchmark C: the Dirac-point quasienergy gap of the full Sambe
    // matrix converges to the exact rotating-frame value
    //
    //   Δ_exact = √((ħω)² + 4g²) − ħω,  g = ev_F A₀ = (3/2)|J|α,
    //
    // measured as the outermost folded branch separation (twice the
    // largest |folded eigenvalue|).  Small α = 0.2 keeps the lattice
    // corrections to the Dirac model are O(α⁴) in absolute units
    // (~1% relative at α = 0.2); the folded
    // spectrum also contains near-zero states from higher photon
    // sectors (the "minimal spacing" would pick those instead of the
    // physical branch gap).  The outer branches must simultaneously
    // agree with the first-order van Vleck mass d_z(K) = 3√3·|K_eff|
    // (Benchmark D's series), tying the Sambe construction to the
    // real-space effective model.
    let j = -1.0;
    let model = graphene_model(j);
    let alpha = 0.2;
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![LightMode::new(
            1,
            array![Complex::new(alpha, 0.0), Complex::new(0.0, alpha)],
        )],
    );
    let w = drive.omega0_ev;
    let g = 1.5 * j.abs() * alpha;
    let delta_exact = (w * w + 4.0 * g * g).sqrt() - w;
    let k = array![1.0 / 3.0, 1.0 / 3.0];

    // First-order van Vleck mass at K (Benchmark D series).
    let n_cut = 8;
    let k_eff = -(2.0 * j * j / w)
        * (1..=n_cut)
            .map(|n| bessel_j(n, alpha).powi(2) / (n as f64) * (TAU * (n as f64) / 3.0).sin())
            .sum::<f64>();
    let d_z_k = -3.0 * 3f64.sqrt() * k_eff;

    let mut outer_previous = f64::INFINITY;
    for n_max in [4, 8, 12] {
        let trunc = FloquetTruncation::new(n_max);
        let hf = model
            .sambe_ham_from_model(&k, &drive, &trunc, Gauge::Lattice)
            .unwrap();
        let e = eigvalsh_v(&hf, UPLO::Lower).unwrap();
        let outer = e
            .iter()
            .map(|x| ((*x + w / 2.0).rem_euclid(w) - w / 2.0).abs())
            .fold(0.0_f64, f64::max);
        // Outermost branch ≈ Δ_exact/2; the residual deviation from
        // Δ_exact/2 is physical (higher van Vleck orders + lattice
        // corrections O(α⁴)), not truncation — the branch is
        // converged already at n_max = 4.
        let tol = if n_max <= 4 { 2e-3 } else { 1e-3 };
        assert!(
            (outer - delta_exact / 2.0).abs() < tol,
            "n_max = {n_max}: outer branch {outer} vs exact {:.6}",
            delta_exact / 2.0
        );
        // The van Vleck mass (all-photon Bessel series) must agree
        // with the outer branch to O(1/W²) corrections.
        assert!(
            (outer - d_z_k.abs()).abs() < 5e-3,
            "n_max = {n_max}: outer branch {outer} vs van Vleck mass {d_z_k}"
        );
        assert!(
            outer <= outer_previous + 1e-12,
            "outer branch must converge downward: {outer} vs previous {outer_previous}"
        );
        outer_previous = outer;
    }
}

#[test]
fn graphene_haldane_mass_matches_full_bessel_series() {
    // Benchmark D: the order-1 effective model's mass term must equal
    //
    //   d_z(k) = 2·K_eff·Σ_j sin(2πk·b_j),
    //   K_eff  = −(2J²/ħω)·Σ_{n=1..N} J_n²(α)/n · sin(2πn/3),
    //
    // with N the harmonic_max truncation (n = 3m terms vanish exactly).
    //
    // Convention note: the van Vleck commutator order here is
    // [H_n, H_{−n}]/(nħω), opposite to the literature's
    // [H_{−n}, H_n]/(nħω); combined with our k-convention being the
    // literature's mirror (H_n(k) = H_n^lit(−k)) the two sign flips
    // cancel for the TR-odd Haldane term — our d_z(k) equals the
    // literature's pointwise, and the Dirac-point mass reproduces
    // the exact rotating-frame leading order +g²/(ħω).
    let j = -1.0;
    let model = graphene_model(j);
    let alpha = 0.5;
    let drive = FloquetDrive::with_modes(
        5.0,
        vec![LightMode::new(
            1,
            array![Complex::new(alpha, 0.0), Complex::new(0.0, alpha)],
        )],
    );
    let w = drive.omega0_ev;

    // Integer NNN vectors: b_1 = e_2 − e_1, b_2 = e_0 − e_2,
    // b_3 = e_1 − e_0.
    let b_int = [[1, 1], [1, -2], [-2, 1]];
    let k_eff_series = |n_cut: isize| -> f64 {
        -(2.0 * j * j / w)
            * (1..=n_cut)
                .map(|n| bessel_j(n, alpha).powi(2) / (n as f64) * (TAU * (n as f64) / 3.0).sin())
                .sum::<f64>()
    };
    let d_z_lit = |k: &[f64; 2], n_cut: isize| -> f64 {
        2.0 * k_eff_series(n_cut)
            * b_int
                .iter()
                .map(|b| (TAU * (k[0] * b[0] as f64 + k[1] * b[1] as f64)).sin())
                .sum::<f64>()
    };

    let eff = model
        .floquet_effective_model(
            &drive,
            Some(&FloquetEffectiveOptions::new().with_harmonic_max(8)),
        )
        .unwrap();
    let eff2 = model
        .floquet_effective_model(
            &drive,
            Some(&FloquetEffectiveOptions::new().with_harmonic_max(2)),
        )
        .unwrap();
    for k in [[0.13, 0.07], [0.23, 0.11], [-0.07, 0.31]] {
        let kvec = array![k[0], k[1]];
        let h = eff.gen_ham(&kvec, Gauge::Lattice);
        let h2 = eff2.gen_ham(&kvec, Gauge::Lattice);
        // The mass term is the diagonal difference (T_0 is purely
        // off-diagonal for graphene).
        let d_z = ((h[[0, 0]] - h[[1, 1]]).re) / 2.0;
        let d_z_2 = ((h2[[0, 0]] - h2[[1, 1]]).re) / 2.0;
        assert!(
            (d_z - d_z_lit(&k, 8)).abs() < 1e-8,
            "k = {k:?}: d_z {d_z} vs series {expect}",
            expect = d_z_lit(&k, 8)
        );
        assert!(
            (d_z_2 - d_z_lit(&k, 2)).abs() < 1e-8,
            "k = {k:?}: harmonic_max = 2: d_z {d_z_2} vs series {}",
            d_z_lit(&k, 2)
        );
        // The neglected tail (n = 3m vanishes; n = 4, 5 are the
        // next contributors at ~1e-8 for α = 0.5).
        assert!(
            (d_z - d_z_2).abs() < 1e-5,
            "k = {k:?}: truncation tail {d_z} vs {d_z_2}"
        );
    }
}

#[test]
fn graphene_order_zero_matches_renormalized_nn_hopping() {
    // Benchmark B: the order-0 effective model renormalizes the NN
    // hopping to J·J_0(α) (non-perturbative in α); its Hamiltonian
    // equals the literature H_0(−k) elementwise.
    let j = -1.0;
    let model = graphene_model(j);
    let alpha = 0.6;
    let drive = graphene_circular_drive(alpha);
    let eff = model
        .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(0)))
        .unwrap();
    for k in [[0.13, 0.21], [0.5, 0.5], [0.87, 0.11]] {
        let kvec = array![k[0], k[1]];
        let h0 = eff.gen_ham(&kvec, Gauge::Lattice);
        let lit = array![
            [
                Complex::new(0.0, 0.0),
                graphene_q_n_lit(0, &[-k[0], -k[1]], j, alpha),
            ],
            [
                graphene_q_n_lit(0, &[-k[0], -k[1]], j, alpha).conj(),
                Complex::new(0.0, 0.0),
            ],
        ];
        for (a, b) in h0.iter().zip(lit.iter()) {
            assert!(
                (a - b).norm() < 1e-12,
                "alpha = {alpha}, k = {k:?}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn bessel_coeffs_reject_large_amplitudes_and_bad_ranges() {
    // R above the single-harmonic cap must error (the caller falls back to
    // the time grid) instead of silently reporting a truncated order range.
    let drive =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(1.0, 0.0)])]);
    let d = array![2.0 * MAX_BESSEL_ARG_CLOSED_FORM];
    assert!(bessel_peierls_coeffs(&d, &drive, -4, 4, 6).is_err());

    // Inside the cap the ladder must still resolve the coefficients against
    // an alias-free grid: the oracle needs at least 2·R samples, so it is
    // sized for the largest amplitude in the loop.
    let grid = FloquetTimeGrid::new(&drive, 65536, -4, 4);
    for r in [60.0, MAX_BESSEL_ARG, MAX_BESSEL_ARG_CLOSED_FORM] {
        let d_allowed = array![r];
        let allowed = bessel_peierls_coeffs(&d_allowed, &drive, -4, 4, 6).unwrap();
        let reference = peierls_fourier_coeffs(&d_allowed, -4, 4, &drive, &grid);
        for (n, (got, want)) in allowed.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).norm() < 1e-12,
                "C_{} at R = {r}: ladder {got} vs time-grid DFT {want}",
                n as isize - 4
            );
        }
    }

    // Harmonic ranges that exclude 0 must not panic; harmonic_min > harmonic_max errors.
    let zero_l =
        FloquetDrive::with_modes(1.0, vec![LightMode::new(0, array![Complex::new(0.1, 0.0)])]);
    let out_of_range = bessel_peierls_coeffs(&d, &zero_l, 5, 7, 6).unwrap();
    for n in 5..=7 {
        assert!((out_of_range[(n - 5) as usize]).norm() == 0.0);
    }
    assert!(bessel_peierls_coeffs(&d, &drive, 3, 2, 6).is_err());

    // cutoff_margin is a documented minimum-order floor bounded to
    // [0, 48]; a larger value would only force a needlessly long sweep.
    assert!(bessel_peierls_coeffs(&d, &drive, -4, 4, 49).is_err());
    assert!(bessel_peierls_coeffs(&d, &drive, -4, 4, -1).is_err());
}

#[test]
fn bessel_coeffs_large_harmonics_stay_within_error_budget() {
    // Regression for the two-pass window sizing: two modes l = ±400 at
    // r = 8 make e^{-i a(t)·d} = e^{-i·16·cos(400·Ω₀·t)}, whose closed
    // form is C_n = (-i)^n J_n(16) on bins divisible by 400 and zero
    // for |n| < 400.  A fixed window capped at 4096 drift units
    // clipped the fold support and got C_0 wrong by ~1e-3 for this
    // drive.
    let d = array![1.0, 0.0];
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(400, array![Complex::new(8.0, 0.0), Complex::new(0.0, 0.0)]),
            LightMode::new(-400, array![Complex::new(8.0, 0.0), Complex::new(0.0, 0.0)]),
        ],
    );
    let coeffs = bessel_peierls_coeffs(&d, &drive, -10, 10, 0).unwrap();
    for n in -10..=10 {
        // Graf's addition theorem: C_0 = Σ_m (-1)^m J_m(8)² = J_0(16).
        let expected = if n == 0 { bessel_j(0, 16.0) } else { 0.0 };
        let got = coeffs[(n + 10) as usize];
        assert!(
            (got - Complex::new(expected, 0.0)).norm() < 1e-12,
            "n = {n}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn bessel_sliced_fold_matches_independent_dense_fold() {
    // Preserve the old full-window scalar convolution as an assembly
    // oracle. Shared ladders fix the truncation; the independent DFT and
    // tabulated tests elsewhere still check the coefficients themselves.
    fn dense(drive: &FloquetDrive, low: isize, high: isize) -> Array1<Complex<f64>> {
        let mut dc = 0.0;
        let mut modes = Vec::new();
        let mut drift = 0;
        for mode in &drive.modes {
            let z = mode.a_complex[0];
            if mode.harmonic == 0 {
                dc += z.re;
                continue;
            }
            if z.re == 0.0 && z.im == 0.0 {
                continue;
            }
            let (ladder, cap) = bessel_ladder_for_amplitude(
                z.norm(),
                1e-12 / drive.modes.len() as f64,
                BESSEL_CUTOFF_MARGIN,
                MAX_BESSEL_ARG,
            )
            .unwrap();
            drift += mode.harmonic.abs() * cap;
            let mut weights = Vec::new();
            let mut power = Complex::new(1.0, 0.0);
            for m in 0..=cap {
                let value =
                    power * ladder.j[m as usize] * Complex::from_polar(1.0, -(m as f64) * z.arg());
                if m == 0 {
                    weights.push((0, value));
                } else {
                    weights.push((
                        -m,
                        power
                            * ladder.j[m as usize]
                            * Complex::from_polar(1.0, (m as f64) * z.arg()),
                    ));
                    weights.push((m, value));
                }
                power *= Complex::new(0.0, -1.0);
            }
            modes.push((mode.harmonic, weights));
        }
        let work_min = low - drift;
        let len = (high + drift - work_min + 1) as usize;
        let mut sequence = vec![Complex::new(0.0, 0.0); len];
        if let Some(index) = work_min.checked_neg()
            && (0..len as isize).contains(&index)
        {
            sequence[index as usize] = Complex::new(1.0, 0.0);
        }
        for (harmonic, weights) in modes {
            let mut next = vec![Complex::new(0.0, 0.0); len];
            for (m, weight) in weights {
                for (index, output) in next.iter_mut().enumerate() {
                    let n = work_min + index as isize;
                    if let Some(source) = n
                        .checked_add(harmonic * m)
                        .and_then(|n| n.checked_sub(work_min))
                        && (0..len as isize).contains(&source)
                    {
                        *output += sequence[source as usize] * weight;
                    }
                }
            }
            sequence = next;
        }
        let phase = Complex::new(0.0, -dc).exp();
        Array1::from(
            sequence[(low - work_min) as usize..(high - work_min + 1) as usize]
                .iter()
                .map(|z| *z * phase)
                .collect::<Vec<_>>(),
        )
    }

    let tiny = f64::from_bits(2); // J1 is the minimum subnormal coefficient.
    let min_field = f64::from_bits(1);
    for carriers in [
        [(1, 0.8, 0.17), (-3, 0.4, -0.13), (2, 1.2, 0.3)],
        [(17, 4.1, 0.7), (-19, 3.8, -0.4), (3, 0.2, 0.06)],
        [(1, 1e-200, 0.0), (2, 2e-200, 0.0), (3, 3e-200, 0.0)],
        [(1, tiny, 0.0), (2, tiny, 0.0), (3, tiny, 0.0)],
        [
            (1, min_field, 0.0),
            (2, min_field, 0.0),
            (3, min_field, 0.0),
        ],
    ] {
        let mut drive = FloquetDrive::with_modes(
            1.0,
            carriers
                .into_iter()
                .map(|(n, re, im)| LightMode::new(n, array![Complex::new(re, im)]))
                .collect(),
        );
        drive.add_mode(LightMode::new(0, array![Complex::new(0.35, 1e308)]));
        drive.add_mode(LightMode::new(-7, array![Complex::new(0.0, 0.0)]));
        drive.add_mode(LightMode::new(11, array![Complex::new(-0.0, -0.0)]));
        if carriers[0].0 == 17 {
            // Four live folds also exercise reuse beyond the first swaps.
            drive.add_mode(LightMode::new(5, array![Complex::new(0.12, -0.04)]));
        }
        let drift: isize = drive
            .modes
            .iter()
            .filter(|mode| {
                mode.harmonic != 0 && (mode.a_complex[0].re != 0.0 || mode.a_complex[0].im != 0.0)
            })
            .map(|mode| {
                let (_, cap) = bessel_ladder_for_amplitude(
                    mode.a_complex[0].norm(),
                    1e-12 / drive.modes.len() as f64,
                    BESSEL_CUTOFF_MARGIN,
                    MAX_BESSEL_ARG,
                )
                .unwrap();
                mode.harmonic.abs() * cap
            })
            .sum();
        for (low, high) in [
            (-17, -3),
            (-4, 4),
            (3, 17),
            (0, 0),
            (-1000, 0),
            (0, 1000),
            (drift - 1, drift + 1),
            (-drift - 1, -drift + 1),
            (-5000, -4993),
            (4993, 5000),
        ] {
            let actual =
                bessel_peierls_coeffs(&array![1.0], &drive, low, high, BESSEL_CUTOFF_MARGIN)
                    .unwrap();
            assert_eq!(
                actual,
                dense(&drive, low, high),
                "clipped fold changed [{low}, {high}]"
            );
        }
    }
}

#[test]
fn bessel_sliced_fold_keeps_full_window_resource_checks() {
    let drive = FloquetDrive::with_modes(
        1.0,
        (1..=3)
            .map(|n| LightMode::new(n * 1_000_000, array![Complex::new(1.0, 0.0)]))
            .collect(),
    );
    // The seed lies outside this window, but that must not bypass the
    // existing overflow/window policy and turn a refused call into Ok(0).
    for n in [1_000_000_000, isize::MIN, isize::MAX] {
        assert!(bessel_peierls_coeffs(&array![1.0], &drive, n, n, BESSEL_CUTOFF_MARGIN,).is_err());
    }
}

#[test]
fn link_validation_projects_each_exact_displacement_once() {
    let mut model = Model::<true, 1>::tb_model(array![[1.0]], Array2::zeros((2, 1)), None).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            model.add_hop(0.2, i, j, &array![1], None);
            model.add_hop(0.1, i, j, &array![1], Some(SpinDirection::X));
        }
    }
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(0.3, 0.1)]),
            LightMode::new(-2, array![Complex::new(0.2, -0.1)]),
            LightMode::new(3, array![Complex::new(0.1, 0.05)]),
        ],
    );
    LINK_RESOLVABILITY_SCANS.with(|count| count.set(0));
    validate_link_resolvability(&model, &drive, -4, 4).unwrap();
    assert_eq!(LINK_RESOLVABILITY_SCANS.with(|count| count.get()), 2);
    let invalid = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(1, array![Complex::new(f64::MAX, 0.0)])],
    );
    assert!(validate_link_resolvability(&model, &invalid, -4, 4).is_err());
    let mut overflow = model.clone();
    overflow.lat[[0, 0]] = 2.0;
    assert!(
        validate_link_resolvability(&overflow, &invalid, -4, 4)
            .unwrap_err()
            .to_string()
            .contains("phase amplitude is not finite")
    );

    // Unequal displacement bits must not be merged through a tolerance.
    let mut nearby = model.clone();
    nearby.orb[[1, 0]] = f64::EPSILON;
    LINK_RESOLVABILITY_SCANS.with(|count| count.set(0));
    validate_link_resolvability(&nearby, &drive, -4, 4).unwrap();
    assert_eq!(LINK_RESOLVABILITY_SCANS.with(|count| count.get()), 6);

    // A later hopping with an already-seen d still has to pass the public
    // finite-Hamiltonian check before either constructor scans any link.
    model.ham[[1, 3, 3]].re = f64::NAN;
    LINK_RESOLVABILITY_SCANS.with(|count| count.set(0));
    assert!(
        model
            .floquet_model(&drive, &FloquetTruncation::new(0))
            .is_err()
    );
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    assert!(
        model
            .floquet_effective_model(&drive, Some(&options))
            .is_err()
    );
    assert_eq!(LINK_RESOLVABILITY_SCANS.with(|count| count.get()), 0);
}

#[test]
fn bessel_fold_handles_windows_at_both_integer_limits() {
    let drive = FloquetDrive::with_modes(
        1.0,
        (1..=3)
            .map(|harmonic| LightMode::new(harmonic, array![Complex::new(1.0, 0.0)]))
            .collect(),
    );
    let (_, cutoff) = bessel_ladder_for_amplitude(1.0, 1e-12 / 3.0, 0, MAX_BESSEL_ARG).unwrap();
    let drift = 6 * cutoff;
    for n in [isize::MIN + drift, isize::MAX - drift] {
        // Three distinct carriers force the fold; the entire finite window
        // lies away from zero, including work_min = MIN on the first case.
        let got = bessel_peierls_coeffs(&array![1.0], &drive, n, n, 0).unwrap();
        assert_eq!(got, array![Complex::new(0.0, 0.0)]);
    }
}

#[test]
fn bessel_coeffs_window_edge_overflows_are_skipped() {
    // The fold evaluates n + l·m for every (n, m) pair in the working
    // window; pairs whose source would leave the isize range must be
    // skipped, not panic (they contribute nothing — the source is
    // outside the window).  With l = 400, r = 8 and margin 48 the
    // drift is 400·56 = 22400, so a request near isize::MAX pushes
    // the window's top edge past isize::MAX during the fold.
    // Regression for the previously unchecked n + shift addition.
    let d = array![1.0, 0.0];
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(
            400,
            array![Complex::new(8.0, 0.0), Complex::new(0.0, 0.0)],
        )],
    );
    let n = isize::MAX - 23_999;
    let coeffs = bessel_peierls_coeffs(&d, &drive, n, n, 48).unwrap();
    // The requested bin is far outside the mode's spectral support
    // (±22400), so C_n = 0.
    assert!(coeffs[0].norm() == 0.0);
}

#[test]
fn rounded_carrier_cancellation_does_not_certify_static_coefficients() {
    let epsilon = 2.0_f64.powi(-54);
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![
            LightMode::new(1, array![Complex::new(1.0, 0.0)]),
            LightMode::new(1, array![Complex::new(epsilon, 0.0)]),
            LightMode::new(-1, array![Complex::new(-1.0, 0.0)]),
        ],
    );
    let d = array![1.0];
    let (_, carriers, exact_sum) = time_grid_projections(&drive, &d);
    assert!(carriers.is_empty()); // Rounded (1 + epsilon) - 1, not an exact cancellation.
    assert!(!exact_sum);
    let got = bessel_peierls_coeffs(&d, &drive, -40, 40, BESSEL_CUTOFF_MARGIN).unwrap();
    assert!(
        got.iter()
            .enumerate()
            .any(|(index, z)| index != 40 && (z.re != 0.0 || z.im != 0.0)),
        "uncertified cancellation must retain the previous non-static backend path"
    );
    for amplitude in [1e-200, f64::from_bits(1)] {
        let tiny = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(1, array![Complex::new(amplitude, 0.0)])],
        );
        let (_, carriers, exact_sum) = time_grid_projections(&tiny, &d);
        assert_eq!(carriers[&1].re, amplitude);
        assert!(exact_sum);
    }
}

#[test]
fn cancelled_ac_harmonics_construct_support_once_at_large_cutoff() {
    // Count Rayon work as well as caller work, without cross-test interference.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let model = chain_model();
            let drive = FloquetDrive::with_modes(
                1e-200,
                vec![
                    LightMode::new(1, array![Complex::new(0.3, -0.2)]),
                    LightMode::new(-1, array![Complex::new(-0.3, -0.2)]),
                ],
            );
            assert!(drive_has_ac_components(&drive)); // Must not restore Cartesian compaction.
            let mut small_support = None;
            for harmonic_max in [1, 1000] {
                REAL_SPACE_SUPPORT_CONSTRUCTIONS.with(|count| count.set(0));
                let options = FloquetEffectiveOptions::new()
                    .with_order(2)
                    .with_harmonic_max(harmonic_max);
                let out = model
                    .floquet_effective_model(&drive, Some(&options))
                    .unwrap();
                assert_eq!(
                    REAL_SPACE_SUPPORT_CONSTRUCTIONS.with(|count| count.get()),
                    2,
                    "construct pair/triple support once, never harmonic_max squared times"
                );
                for (row, r) in out.hamR.outer_iter().enumerate() {
                    let expected = model
                        .hamR
                        .outer_iter()
                        .position(|source| source == r)
                        .map_or(Complex::new(0.0, 0.0), |source| model.ham[[source, 0, 0]]);
                    assert_eq!(out.ham[[row, 0, 0]], expected); // No new nonzero hopping.
                }
                if let Some(support) = &small_support {
                    assert_eq!(&out.hamR, support);
                }
                small_support = Some(out.hamR);
            }
        });
}

fn chain_model() -> Model<false, 1, NoRMatrix> {
    let lat = array![[1.0]];
    let orb = array![[0.0]];
    let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), None);
    model
}

fn metadata_model() -> Model<false, 1, NoRMatrix> {
    let lat = array![[1.0]];
    let orb = array![[0.0], [0.0], [0.35]];
    let atoms = vec![
        Atom::with_orbitals(
            arr1(&[0.0]),
            AtomType::C,
            [OrbitalId::new(0), OrbitalId::new(1)],
        ),
        Atom::with_orbitals(arr1(&[0.35]), AtomType::O, [OrbitalId::new(2)]),
    ];
    let mut model = Model::<false, 1>::tb_model(lat, orb, Some(atoms)).unwrap();
    model.orb_projection = vec![OrbProj::s, OrbProj::px, OrbProj::py];
    model.set_hop(-0.8_f64, 0, 2, &arr1(&[1isize]), None);
    model
}

fn assert_same_atom_metadata(expected: &Atom, actual: &Atom) {
    assert_eq!(actual.position(), expected.position());
    assert_eq!(actual.norb(), expected.norb());
    assert_eq!(actual.atom_type(), expected.atom_type());
}

#[test]
fn floquet_review_regression_preserves_underflow_sized_hoppings() {
    let mut model = chain_model();
    let scale = 1e-200;
    model.ham.mapv_inplace(|value| value * scale);
    let k = array![0.0];
    let trunc = FloquetTruncation::new(0);
    // Include a genuinely active field: static fast paths otherwise bypass
    // the harmonic cache's nonzero-hopping filter. J0(1) is independent.
    for (drive, renormalization) in [
        (FloquetDrive::new(5.0), 1.0),
        (
            FloquetDrive::with_modes(5.0, vec![LightMode::new(1, array![Complex::new(0.0, 0.0)])]),
            1.0,
        ),
        (
            FloquetDrive::with_modes(5.0, vec![LightMode::new(1, array![Complex::new(1.0, 0.0)])]),
            0.765_197_686_557_966_6,
        ),
    ] {
        let from_model = model
            .floquet_model(&drive, &trunc)
            .unwrap()
            .gen_ham(&k, Gauge::Lattice);
        let from_effective = model
            .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(0)))
            .unwrap()
            .gen_ham(&k, Gauge::Lattice);
        for actual in [from_model, from_effective] {
            // Divide before comparing: an absolute 1e-12 tolerance, or a
            // squared norm, would also accept a silently deleted hopping.
            let rescaled = actual[[0, 0]] / scale;
            assert!(
                (rescaled + 2.0 * renormalization).norm() < 1e-12,
                "a nonzero hopping was lost with {} modes: H(0)/scale = {rescaled}",
                drive.modes.len()
            );
        }
    }
}

#[test]
fn floquet_no_drive_static_replicas() {
    let model = chain_model();
    let k = arr1(&[0.17]);
    let drive = FloquetDrive::new(0.7);
    let trunc = FloquetTruncation::new(1);

    let sambe = model.floquet_model(&drive, &trunc).unwrap();
    let bands = sambe.solve_band_onek(&k).unwrap();
    let e0 = model.solve_band_onek(&k).unwrap()[0];
    let mut expected = vec![e0 - 0.7, e0, e0 + 0.7];
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

    for (a, b) in bands.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-12, "got {a}, expected {b}");
    }
}

#[test]
fn floquet_hamiltonian_is_hermitian() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0]), None);
    model.set_hop(-0.7_f64, 0, 0, &arr1(&[0isize, 1]), None);

    let drive = FloquetDrive::with_modes(
        0.9,
        vec![LightMode::new(
            1,
            arr1(&[Complex::new(0.11, 0.0), Complex::new(0.0, 0.07)]),
        )],
    );
    let trunc = FloquetTruncation::new(2);
    let k = arr1(&[0.13, 0.29]);
    let hf = model
        .sambe_ham_from_model(&k, &drive, &trunc, Gauge::Atom)
        .unwrap();

    let mut max_diff = 0.0f64;
    for i in 0..hf.nrows() {
        for j in 0..hf.ncols() {
            max_diff = max_diff.max((hf[[i, j]] - hf[[j, i]].conj()).norm());
        }
    }
    assert!(max_diff < 1e-11, "max hermiticity error = {max_diff:e}");
}

#[test]
fn floquet_model_matches_onek_construction() {
    let lat = array![[1.0, 0.0], [0.2, 1.1]];
    let orb = array![[0.0, 0.0], [0.31, 0.17]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(0.2_f64, 0, 1, &arr1(&[0isize, 0]), None);
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0]), None);
    model.set_hop(-0.6_f64, 1, 1, &arr1(&[0isize, 1]), None);

    let drive = FloquetDrive::with_modes(
        0.8,
        vec![LightMode::new(
            1,
            arr1(&[Complex::new(0.13, 0.0), Complex::new(0.0, 0.09)]),
        )],
    );
    let trunc = FloquetTruncation::new(1);
    let k = arr1(&[0.23, 0.31]);
    let floquet_model = model.floquet_model(&drive, &trunc).unwrap();

    assert_eq!(
        floquet_model.nsta(),
        model.nsta() * trunc.n_sector().unwrap()
    );
    assert_eq!(floquet_model.hamR, model.hamR);

    for gauge in [Gauge::Lattice, Gauge::Atom] {
        let from_model = floquet_model.gen_ham(&k, gauge);
        let from_onek = model
            .reference_sambe_ham_onek(&k, &drive, &trunc, gauge)
            .unwrap();
        let mut max_diff = 0.0f64;
        for i in 0..from_model.nrows() {
            for j in 0..from_model.ncols() {
                max_diff = max_diff.max((from_model[[i, j]] - from_onek[[i, j]]).norm());
            }
        }
        assert!(
            max_diff < 1e-12,
            "floquet_model mismatch in {gauge:?}: {max_diff:e}"
        );
    }
}

#[test]
fn floquet_model_reuses_real_space_hamiltonian_across_batch_solvers() {
    let mut model =
        Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.23]], None).unwrap();
    model.set_onsite(&array![0.2, -0.3], None);
    model.set_hop(Complex::new(0.3, 0.2), 0, 1, &array![0], None);
    model.set_hop(Complex::new(-0.7, 0.1), 0, 1, &array![1], None);
    let drive = FloquetDrive::with_modes(
        2.5,
        vec![
            LightMode::new(-1, array![Complex::new(0.17, 0.06)]),
            LightMode::new(3, array![Complex::new(-0.02, 0.04)]),
        ],
    );
    let trunc = FloquetTruncation::new(2);
    let sambe = model.floquet_model(&drive, &trunc).unwrap();
    let stored = sambe.ham.clone();
    let support = sambe.hamR.clone();
    let points = array![[0.0], [0.21], [0.49]];
    let serial = sambe.solve_band_all(&points).unwrap();
    let parallel = sambe.solve_band_all_parallel(&points).unwrap();
    assert_eq!(serial.dim(), (3, model.nsta() * trunc.n_sector().unwrap()));
    for (row, k) in points.outer_iter().enumerate() {
        let reference_ham = model
            .reference_sambe_ham_onek(&k, &drive, &trunc, Gauge::Atom)
            .unwrap();
        let reference = eigvalsh_v(&reference_ham, UPLO::Upper).unwrap();
        for band in 0..reference.len() {
            assert!((serial[[row, band]] - reference[band]).abs() < 1e-12);
            assert!((parallel[[row, band]] - reference[band]).abs() < 1e-12);
        }
    }
    assert_eq!(sambe.ham, stored);
    assert_eq!(sambe.hamR, support);
    assert!(sambe.solve_band_onek(&array![0.0, 0.0]).is_err());
    assert!(sambe.solve_band_onek(&array![f64::NAN]).is_err());
}

#[test]
fn floquet_model_preserves_spinful_layout() {
    let lat = array![[1.0, 0.0], [0.0, 1.0]];
    let orb = array![[0.0, 0.0], [0.27, 0.19]];
    let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(0.3_f64, 0, 0, &arr1(&[0isize, 0]), crate::SpinDirection::Z);
    model.add_hop(0.2_f64, 0, 0, &arr1(&[0isize, 0]), crate::SpinDirection::X);
    model.set_hop(-0.9_f64, 0, 1, &arr1(&[1isize, 0]), None);
    model.set_hop(-0.4_f64, 1, 1, &arr1(&[0isize, 1]), None);

    let drive = FloquetDrive::with_modes(
        0.6,
        vec![LightMode::new(
            1,
            arr1(&[Complex::new(0.07, 0.0), Complex::new(0.0, 0.05)]),
        )],
    );
    let trunc = FloquetTruncation::new(1);
    let k = arr1(&[0.17, 0.29]);
    let floquet_model = model.floquet_model(&drive, &trunc).unwrap();

    assert_eq!(
        floquet_model.norb(),
        model.norb() * trunc.n_sector().unwrap()
    );
    assert_eq!(
        floquet_model.nsta(),
        model.nsta() * trunc.n_sector().unwrap()
    );

    for gauge in [Gauge::Lattice, Gauge::Atom] {
        let from_model = floquet_model.gen_ham(&k, gauge);
        let from_onek = model
            .reference_sambe_ham_onek(&k, &drive, &trunc, gauge)
            .unwrap();
        let mut max_diff = 0.0f64;
        for i in 0..from_model.nrows() {
            for j in 0..from_model.ncols() {
                max_diff = max_diff.max((from_model[[i, j]] - from_onek[[i, j]]).norm());
            }
        }
        assert!(
            max_diff < 1e-12,
            "spinful floquet_model mismatch in {gauge:?}: {max_diff:e}"
        );
    }
}

#[test]
fn floquet_models_preserve_atom_metadata() {
    let model = metadata_model();
    let drive = FloquetDrive::new(1.2);
    let trunc = FloquetTruncation::new(1);

    let sambe = model.floquet_model(&drive, &trunc).unwrap();
    assert_eq!(sambe.natom(), model.natom() * trunc.n_sector().unwrap());
    for sector in 0..trunc.n_sector().unwrap() {
        for i_atom in 0..model.natom() {
            assert_same_atom_metadata(
                &model.atoms[i_atom],
                &sambe.atoms[sector * model.natom() + i_atom],
            );
        }
    }
    let expected_projection: Vec<OrbProj> = (0..trunc.n_sector().unwrap())
        .flat_map(|_| model.orb_projection.iter().copied())
        .collect();
    assert_eq!(sambe.orb_projection, expected_projection);

    let effective = model.floquet_effective_model(&drive, None).unwrap();
    assert_eq!(effective.natom(), model.natom());
    for i_atom in 0..model.natom() {
        assert_same_atom_metadata(&model.atoms[i_atom], &effective.atoms[i_atom]);
    }
    assert_eq!(effective.orb_projection, model.orb_projection);
}

#[test]
fn floquet_effective_rejects_non_hermitian_target_range() {
    let model = chain_model();
    let drive = FloquetDrive::new(1.0);
    let n_time = 32;
    let options = FloquetEffectiveOptions::new().with_target_hamR(array![[0isize], [1isize]]);

    let err = model
        .floquet_effective_model_legacy(&drive, n_time, [8], Some(&options))
        .unwrap_err();
    match err {
        TbError::MissingHermitianConjugateHopping { r } => {
            assert_eq!(r, arr1(&[1isize]));
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn floquet_effective_rejects_duplicate_target_vectors() {
    let model = chain_model();
    let drive = FloquetDrive::new(1.0);
    let n_time = 32;
    let options = FloquetEffectiveOptions::new().with_target_hamR(array![[0isize], [0isize]]);

    let err = model
        .floquet_effective_model_legacy(&drive, n_time, [8], Some(&options))
        .unwrap_err();
    match err {
        TbError::Other(message) => {
            assert!(message.contains("duplicate vector"), "{message}");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn floquet_effective_order0_matches_h0() {
    let model = chain_model();
    let drive = FloquetDrive::with_modes(
        1.1,
        vec![LightMode::new(1, arr1(&[Complex::new(0.23, 0.0)]))],
    );
    let n_time = 512;
    let options = FloquetEffectiveOptions::new().with_order(0);
    let effective = model
        .floquet_effective_model_legacy(&drive, n_time, [32], Some(&options))
        .unwrap();

    assert_eq!(effective.nsta(), model.nsta());
    assert_eq!(effective.hamR, model.hamR);

    let k = arr1(&[0.173]);
    let from_model = effective.gen_ham(&k, Gauge::Lattice);
    let harmonic_cache =
        model.floquet_harmonic_cache(&drive, 0, 0, &PeierlsFourierMethod::TimeGrid { n_time });
    let h0 = model.floquet_cached_harmonic_onek(&k, 0, Gauge::Lattice, &harmonic_cache);
    let mut max_diff = 0.0f64;
    for i in 0..from_model.nrows() {
        for j in 0..from_model.ncols() {
            max_diff = max_diff.max((from_model[[i, j]] - h0[[i, j]]).norm());
        }
    }
    assert!(max_diff < 1e-12, "order-0 effective mismatch: {max_diff:e}");
}

#[test]
fn floquet_effective_no_drive_matches_static_model() {
    let model = chain_model();
    let drive = FloquetDrive::new(0.9);
    let n_time = 64;
    let effective = model
        .floquet_effective_model_legacy(&drive, n_time, [32], None)
        .unwrap();

    assert_eq!(effective.nsta(), model.nsta());
    assert_eq!(effective.hamR, model.hamR);

    let k = arr1(&[0.271]);
    for gauge in [Gauge::Lattice, Gauge::Atom] {
        let from_effective = effective.gen_ham(&k, gauge);
        let from_static = model.gen_ham(&k, gauge);
        let mut max_diff = 0.0f64;
        for i in 0..from_effective.nrows() {
            for j in 0..from_effective.ncols() {
                max_diff = max_diff.max((from_effective[[i, j]] - from_static[[i, j]]).norm());
            }
        }
        assert!(
            max_diff < 1e-12,
            "no-drive effective mismatch in {gauge:?}: {max_diff:e}"
        );
    }
}

#[test]
fn floquet_weak_drive_matches_first_order_peierls() {
    let model = chain_model();
    let amp = 1e-5;
    let drive = FloquetDrive::with_modes(
        1.0,
        vec![LightMode::new(1, arr1(&[Complex::new(amp, 0.0)]))],
    );
    let trunc = FloquetTruncation::new(1);
    let k = arr1(&[0.25]);
    let hf = model
        .sambe_ham_from_model(&k, &drive, &trunc, Gauge::Lattice)
        .unwrap();

    let nsta = model.nsta();
    let sector = |n: isize| -> usize { (n + trunc.n_max) as usize };
    let h_q1 = hf[[sector(0) * nsta, sector(-1) * nsta]];
    let expected = -amp * (TAU * k[0]).sin();
    assert!(
        (h_q1.re - expected).abs() < 1e-9,
        "got {}, expected {}",
        h_q1.re,
        expected
    );
    assert!(h_q1.im.abs() < 1e-9, "imag part = {}", h_q1.im);
}

#[test]
fn normalize3_is_scale_free_and_rejects_non_finite_inputs() {
    // A direction is a ray: the magnitude it is expressed in must not decide
    // whether it can be normalized.
    let expected = array![1.0, 2.0, 3.0] / 14.0_f64.sqrt();
    for scale in [1e-300_f64, 1e-200, 1e-14, 1e-8, 1.0, 1e8, 1e200, 1e300] {
        let unit = normalize3(&array![scale, 2.0 * scale, 3.0 * scale]).unwrap();
        for (value, reference) in unit.iter().zip(expected.iter()) {
            assert!(
                (value - reference).abs() < 1e-12,
                "scale {scale:e}: {value} vs {reference}"
            );
        }
    }
    assert!(normalize3(&array![0.0, 0.0, 0.0]).is_err());
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(normalize3(&array![bad, 1.0, 1.0]).is_err());
    }
}

#[test]
fn incident_basis_review_regression_validates_and_normalizes_extreme_directions() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for axis in 0..3 {
            let mut direction = array![1.0, 1.0, 1.0];
            direction[axis] = value;
            assert!(
                IncidentBasis::from_direction(&direction).is_err(),
                "a non-finite incident direction was accepted: {direction:?}"
            );
        }
    }
    // Normalize an oblique, large vector against independent geometry,
    // without prescribing the arbitrary orientation of the transverse axes.
    let basis = IncidentBasis::from_direction(&array![1e200, 2e200, 3e200]).unwrap();
    let expected_direction = array![1.0, 2.0, 3.0] / 14.0_f64.sqrt();
    for (&value, &reference) in basis.k_hat.iter().zip(expected_direction.iter()) {
        assert!(
            (value - reference).abs() < 1e-12,
            "the normalized direction is wrong: {value} vs {reference}"
        );
    }
    for vector in [&basis.k_hat, &basis.e1, &basis.e2] {
        assert!(vector.iter().all(|value| value.is_finite()));
        assert!((vector.dot(vector) - 1.0).abs() < 1e-12);
    }
    for (left, right) in [
        (&basis.k_hat, &basis.e1),
        (&basis.k_hat, &basis.e2),
        (&basis.e1, &basis.e2),
    ] {
        assert!(left.dot(right).abs() < 1e-12);
    }
    let cross = array![
        basis.e1[1] * basis.e2[2] - basis.e1[2] * basis.e2[1],
        basis.e1[2] * basis.e2[0] - basis.e1[0] * basis.e2[2],
        basis.e1[0] * basis.e2[1] - basis.e1[1] * basis.e2[0],
    ];
    for (&value, &reference) in cross.iter().zip(basis.k_hat.iter()) {
        assert!(
            (value - reference).abs() < 1e-12,
            "the transverse basis is not right-handed: {value} vs {reference}"
        );
    }
}

#[test]
fn floquet_incident_basis_public_api_example() {
    let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let orb = array![[0.0, 0.0, 0.0]];
    let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0, 0]), None);
    model.set_hop(-0.8_f64, 0, 0, &arr1(&[0isize, 1, 0]), None);
    model.set_hop(-0.6_f64, 0, 0, &arr1(&[0isize, 0, 1]), None);

    let incident = IncidentBasis::from_direction(&arr1(&[0.0, 0.0, 1.0])).unwrap();
    let circular = incident.polarization([
        Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex::new(0.0, 1.0 / 2.0_f64.sqrt()),
    ]);
    let drive = FloquetDrive::with_modes(0.8, vec![LightMode::new(1, circular.mapv(|z| 0.12 * z))]);
    let trunc = FloquetTruncation::new(1);
    let k = arr1(&[0.2, 0.1, 0.0]);

    let sambe = model.floquet_model(&drive, &trunc).unwrap();
    let hf = sambe.gen_ham(&k, Gauge::Lattice);
    assert_eq!(hf.dim(), (3, 3));

    let mut max_diff = 0.0f64;
    for i in 0..hf.nrows() {
        for j in 0..hf.ncols() {
            max_diff = max_diff.max((hf[[i, j]] - hf[[j, i]].conj()).norm());
        }
    }
    assert!(max_diff < 1e-11, "max hermiticity error = {max_diff:e}");

    let mut qe = sambe.solve_band_onek(&k).unwrap();
    qe.mapv_inplace(|energy| fold_quasienergy(energy, drive.omega0_ev));
    qe.as_slice_mut().unwrap().sort_by(f64::total_cmp);
    assert_eq!(qe.len(), 3);
    for &x in qe.iter() {
        assert!(
            x >= -0.5 * drive.omega0_ev - 1e-12 && x < 0.5 * drive.omega0_ev + 1e-12,
            "quasienergy {x} is outside the first Floquet zone"
        );
    }
}

// ════════════════════════════════════════════════════════════════════════
// General two-band analytical benchmarks (square & rectangular lattices).
//
// These verify the first-order van Vleck effective model against analytic
// results for H_0(k) = ε(k) σ_0 + d(k)·σ under three drives:
//   * circular   a(t) = κ (cos Ωt, η sin Ωt)          (η = ±1 helicity),
//   * elliptical a(t) = (A_x cos Ωt, A_y sin Ωt),
//   * an exotic two-harmonic drive (cos Ωt, sin(2Ωt+α)) whose harmonics are
//     each linearly polarized, so its first-order commutator is O(κ³).
//
// Two independent analytic predictions are checked (fractional-k units,
// Ω = ħω in eV, κ = |e|A₀/ħ):
//   Level I  (exact in field): d_eff = d_0 + (2i/Ω) Σ_{n>0} (d_n × d_{−n})/n
//   Level II (weak field):     δd_CPL = (A_x A_y / 4π²Ω)(∂_x d × ∂_y d),
//                              δd_A²  = (κ² / 16π²)(∂_x² + ∂_y²) d.

/// Two-band QWZ-type model on a rectangular lattice `lat` (rows = lattice
/// vectors, both orbitals at the origin):
///   d(k) = ( t_x cos 2πk_x, t_y sin 2πk_y, m − 2t_z (cos 2πk_x + cos 2πk_y) ),
///   ε(k) = 0.
fn two_band_qwz(
    tx: f64,
    ty: f64,
    tz: f64,
    m: f64,
    lat: [[f64; 2]; 2],
) -> Model<false, 2, NoRMatrix> {
    let lat = array![lat[0], lat[1]];
    let orb = array![[0.0, 0.0], [0.0, 0.0]];
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_hop(m, 0, 0, &array![0, 0], None);
    model.set_hop(-m, 1, 1, &array![0, 0], None);
    for r in [[1, 0], [0, 1]] {
        model.set_hop(-tz, 0, 0, &array![r[0], r[1]], None);
        model.set_hop(tz, 1, 1, &array![r[0], r[1]], None);
    }
    model.set_hop(tx / 2.0, 0, 1, &array![1, 0], None);
    model.set_hop(tx / 2.0, 0, 1, &array![-1, 0], None);
    model.set_hop(-ty / 2.0, 0, 1, &array![0, 1], None);
    model.set_hop(ty / 2.0, 0, 1, &array![0, -1], None);
    model
}

/// Analytic d(k) of `two_band_qwz` (fractional k, independent of `lat`).
fn qwz_d(k: &[f64; 2], tx: f64, ty: f64, tz: f64, m: f64) -> [f64; 3] {
    let (kx, ky) = (TAU * k[0], TAU * k[1]);
    [
        tx * kx.cos(),
        ty * ky.sin(),
        m - 2.0 * tz * (kx.cos() + ky.cos()),
    ]
}

/// Decompose a Hermitian 2×2 matrix H = ε σ_0 + d·σ into (ε, [d_x,d_y,d_z]).
fn decompose_two_band(h: &Array2<Complex<f64>>) -> (f64, [f64; 3]) {
    let eps = (h[[0, 0]] + h[[1, 1]]).re / 2.0;
    let dz = (h[[0, 0]] - h[[1, 1]]).re / 2.0;
    let dx = h[[0, 1]].re;
    let dy = -h[[0, 1]].im;
    (eps, [dx, dy, dz])
}

/// Independent (non-Bessel, non-convolution) reference for H^(n)(k), the
/// n-th Fourier block of the Peierls-dressed Hamiltonian, by direct
/// trapezoidal integration of H(k,t) = Σ_R t(R) e^{i2πk·R} e^{−i a(t)·d_R}
/// over one period, with a(t) reconstructed from the drive modes.  Shares
/// no code with the Bessel / convolution / commutator machinery under test.
fn independent_dressed_harmonic(
    model: &Model<false, 2, NoRMatrix>,
    k: &[f64; 2],
    drive: &FloquetDrive,
    n: isize,
    n_time: usize,
) -> Array2<Complex<f64>> {
    let nsta = model.nsta();
    let norb = model.norb();
    let mut h_n = Array2::<Complex<f64>>::zeros((nsta, nsta));
    for it in 0..n_time {
        let theta = TAU * (it as f64) / (n_time as f64);
        let mut a = [0.0f64; 2];
        for mode in &drive.modes {
            let phase = Complex::new(0.0, -(mode.harmonic as f64) * theta).exp();
            for (comp, ai) in mode.a_complex.iter().enumerate() {
                a[comp] += (ai * phase).re;
            }
        }
        let mut h = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for (i_r, r_row) in model.hamR.outer_iter().enumerate() {
            let r = [r_row[0], r_row[1]];
            let bloch = Complex::new(0.0, TAU * (r[0] as f64 * k[0] + r[1] as f64 * k[1])).exp();
            for i in 0..nsta {
                for j in 0..nsta {
                    let t = model.ham[[i_r, i, j]];
                    if t.re == 0.0 && t.im == 0.0 {
                        continue;
                    }
                    let mut d = [0.0f64; 2];
                    for c in 0..2 {
                        let mut acc = 0.0;
                        for b in 0..2 {
                            let frac =
                                r[b] as f64 + model.orb[[j % norb, b]] - model.orb[[i % norb, b]];
                            acc += frac * model.lat[[b, c]];
                        }
                        d[c] = acc;
                    }
                    let peierls = Complex::new(0.0, -(a[0] * d[0] + a[1] * d[1])).exp();
                    h[[i, j]] += t * bloch * peierls;
                }
            }
        }
        // C_n = (1/T)∫ e^{+inΩt} (⋯) dt — matches the code's convention.
        let fourier = Complex::new(0.0, (n as f64) * theta).exp();
        h_n.scaled_add(fourier, &h);
    }
    h_n.mapv(|x| x / (n_time as f64))
}

/// Independent first-order van Vleck H_eff from the integrated harmonics.
fn independent_heff(
    model: &Model<false, 2, NoRMatrix>,
    k: &[f64; 2],
    drive: &FloquetDrive,
    harmonic_max: isize,
    n_time: usize,
) -> Array2<Complex<f64>> {
    let mut h_eff = independent_dressed_harmonic(model, k, drive, 0, n_time);
    for n in 1..=harmonic_max {
        let hp = independent_dressed_harmonic(model, k, drive, n, n_time);
        let hm = independent_dressed_harmonic(model, k, drive, -n, n_time);
        let comm = hp.dot(&hm) - hm.dot(&hp);
        let scale = Complex::new(1.0 / ((n as f64) * drive.omega0_ev), 0.0);
        h_eff.scaled_add(scale, &comm);
    }
    h_eff
}

/// Code's first-order H_eff as a 2×2 matrix at fractional k.
fn code_heff(
    model: &Model<false, 2, NoRMatrix>,
    k: &[f64; 2],
    drive: &FloquetDrive,
    harmonic_max: isize,
) -> Array2<Complex<f64>> {
    let options = FloquetEffectiveOptions::new().with_harmonic_max(harmonic_max);
    let eff = model
        .floquet_effective_model(drive, Some(&options))
        .unwrap();
    eff.gen_ham(&array![k[0], k[1]], Gauge::Lattice)
}

/// Code's order-0 (Peierls-dressed static) H_eff at fractional k.
fn code_heff_order0(
    model: &Model<false, 2, NoRMatrix>,
    k: &[f64; 2],
    drive: &FloquetDrive,
) -> Array2<Complex<f64>> {
    let options = FloquetEffectiveOptions::new().with_order(0);
    let eff = model
        .floquet_effective_model(drive, Some(&options))
        .unwrap();
    eff.gen_ham(&array![k[0], k[1]], Gauge::Lattice)
}

fn circular_drive(kappa: f64, eta: f64, omega: f64) -> FloquetDrive {
    FloquetDrive::with_modes(
        omega,
        vec![LightMode::new(
            1,
            array![Complex::new(kappa, 0.0), Complex::new(0.0, eta * kappa)],
        )],
    )
}

fn elliptical_drive(ax: f64, ay: f64, omega: f64) -> FloquetDrive {
    FloquetDrive::with_modes(
        omega,
        vec![LightMode::new(
            1,
            array![Complex::new(ax, 0.0), Complex::new(0.0, ay)],
        )],
    )
}

/// a(t) = κ (cos Ωt, sin(2Ωt+α)): mode l=1 along x, l=2 along y with
/// a_y = κ(sin α + i cos α) (⇒ Re[…e^{−i2Ωt}] = κ sin(2Ωt+α)).
fn exotic_drive(kappa: f64, alpha: f64, omega: f64) -> FloquetDrive {
    FloquetDrive::with_modes(
        omega,
        vec![
            LightMode::new(1, array![Complex::new(kappa, 0.0), Complex::new(0.0, 0.0)]),
            LightMode::new(
                2,
                array![
                    Complex::new(0.0, 0.0),
                    Complex::new(kappa * alpha.sin(), kappa * alpha.cos()),
                ],
            ),
        ],
    )
}

#[test]
fn two_band_level1_matches_independent_integration() {
    // Level I: for a general two-band model the first-order van Vleck
    // effective model must equal H^(0) + Σ_{n≥1} [H^(n),H^(−n)]/(nΩ) with
    // H^(n) the dressed harmonics — checked against an independent
    // time-integration of the dressed Hamiltonian, for circular, elliptical
    // and exotic drives on both square and rectangular lattices.
    let (tx, ty, tz, m) = (1.0, 0.7, 0.5, 0.8);
    let omega = 8.0;
    let n_time = 4096;
    let harmonic_max = 4;
    let ks = [[0.13, 0.27], [0.44, 0.61]];
    let lattices = [
        ("square", [[1.0, 0.0], [0.0, 1.0]]),
        ("rectangular", [[1.0, 0.0], [0.0, 1.6]]),
    ];
    for (lname, lat) in lattices {
        let model = two_band_qwz(tx, ty, tz, m, lat);
        let drives: Vec<(&str, FloquetDrive)> = vec![
            ("circular", circular_drive(0.5, 1.0, omega)),
            ("elliptical", elliptical_drive(0.5, 0.3, omega)),
            ("exotic", exotic_drive(0.4, 0.6, omega)),
        ];
        for (dname, drive) in drives {
            for k in ks {
                let code = code_heff(&model, &k, &drive, harmonic_max);
                let indep = independent_heff(&model, &k, &drive, harmonic_max, n_time);
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            (code[[i, j]] - indep[[i, j]]).norm() < 1e-8,
                            "[{lname}/{dname}] k={k:?}: code {} vs independent {}",
                            code[[i, j]],
                            indep[[i, j]]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn two_band_weak_field_matches_cross_product() {
    // Level II: in the weak-field limit the helicity-odd and -even parts of
    // δd obey (fractional k, lat = I)
    //   δd_CPL = η (κ²/4π²Ω) (∂_x d × ∂_y d),
    //   δd_A²  =   (κ²/16π²) (∂_x² + ∂_y²) d.
    // They are isolated from the exact (all-κ) code result by a two-point
    // Richardson extrapolation in κ, removing the O(κ⁴) truncation error.
    let (tx, ty, tz, m) = (1.0, 0.7, 0.5, 0.8);
    let omega = 8.0;
    let k = [0.17, 0.31];
    let model = two_band_qwz(tx, ty, tz, m, [[1.0, 0.0], [0.0, 1.0]]);

    let (kx, ky) = (TAU * k[0], TAU * k[1]);
    let dx_d = [-TAU * tx * kx.sin(), 0.0, 2.0 * TAU * tz * kx.sin()];
    let dy_d = [0.0, TAU * ty * ky.cos(), 2.0 * TAU * tz * ky.sin()];
    let cross = [
        dx_d[1] * dy_d[2] - dx_d[2] * dy_d[1],
        dx_d[2] * dy_d[0] - dx_d[0] * dy_d[2],
        dx_d[0] * dy_d[1] - dx_d[1] * dy_d[0],
    ];
    let lap = [
        -TAU * TAU * tx * kx.cos(),
        -TAU * TAU * ty * ky.sin(),
        2.0 * tz * TAU * TAU * (kx.cos() + ky.cos()),
    ];
    let d_static = qwz_d(&k, tx, ty, tz, m);

    // d_eff(η) at order 1 (harmonic_max = 1 isolates the O(κ²) cross product).
    let d_eff = |kappa: f64, eta: f64| -> [f64; 3] {
        let drive = circular_drive(kappa, eta, omega);
        let h = code_heff(&model, &k, &drive, 1);
        decompose_two_band(&h).1
    };

    let cpl = |kappa: f64| -> [f64; 3] {
        let dp = d_eff(kappa, 1.0);
        let dm = d_eff(kappa, -1.0);
        [
            (dp[0] - dm[0]) / 2.0,
            (dp[1] - dm[1]) / 2.0,
            (dp[2] - dm[2]) / 2.0,
        ]
    };
    let a2 = |kappa: f64| -> [f64; 3] {
        let dp = d_eff(kappa, 1.0);
        let dm = d_eff(kappa, -1.0);
        [
            (dp[0] + dm[0]) / 2.0 - d_static[0],
            (dp[1] + dm[1]) / 2.0 - d_static[1],
            (dp[2] + dm[2]) / 2.0 - d_static[2],
        ]
    };

    // f(κ) = C κ² + O(κ⁴) ⇒ C = (16 f(κ/2) − f(κ)) / (3 κ²).
    let richardson = |f1: [f64; 3], f2: [f64; 3], k1: f64| -> [f64; 3] {
        [
            (16.0 * f2[0] - f1[0]) / (3.0 * k1 * k1),
            (16.0 * f2[1] - f1[1]) / (3.0 * k1 * k1),
            (16.0 * f2[2] - f1[2]) / (3.0 * k1 * k1),
        ]
    };

    let kappa1 = 0.1;
    let cpl_coeff = richardson(cpl(kappa1), cpl(kappa1 / 2.0), kappa1);
    let a2_coeff = richardson(a2(kappa1), a2(kappa1 / 2.0), kappa1);

    // Predicted coefficients (η = +1): cross/(4π²Ω) = cross/(TAU²Ω),
    // lap/(16π²) = lap/(4·TAU²).
    let cpl_pred = [
        cross[0] / (TAU * TAU * omega),
        cross[1] / (TAU * TAU * omega),
        cross[2] / (TAU * TAU * omega),
    ];
    let a2_pred = [
        lap[0] / (4.0 * TAU * TAU),
        lap[1] / (4.0 * TAU * TAU),
        lap[2] / (4.0 * TAU * TAU),
    ];

    for comp in 0..3 {
        assert!(
            (cpl_coeff[comp] - cpl_pred[comp]).abs() < 1e-4,
            "CPL coefficient[{comp}]: {:.6} vs analytic {:.6}",
            cpl_coeff[comp],
            cpl_pred[comp]
        );
        assert!(
            (a2_coeff[comp] - a2_pred[comp]).abs() < 1e-4,
            "A² coefficient[{comp}]: {:.6} vs analytic {:.6}",
            a2_coeff[comp],
            a2_pred[comp]
        );
    }

    // Elliptical generalization: δd_CPL = (A_x A_y / 4π²Ω) (∂_x d × ∂_y d).
    // Isolate the helicity-odd part by A_y → −A_y; the O(κ⁴) truncation
    // error is ~ (A δ)² relative ≈ 3e-3, i.e. ~1e-6 absolute — below the
    // 1e-5 tolerance.
    let d_eff_ell = |ax: f64, ay: f64| -> [f64; 3] {
        let drive = elliptical_drive(ax, ay, omega);
        let h = code_heff(&model, &k, &drive, 1);
        decompose_two_band(&h).1
    };
    let (ax, ay) = (0.08, 0.05);
    let dp = d_eff_ell(ax, ay);
    let dm = d_eff_ell(ax, -ay);
    let cpl_ell = [
        (dp[0] - dm[0]) / 2.0,
        (dp[1] - dm[1]) / 2.0,
        (dp[2] - dm[2]) / 2.0,
    ];
    for comp in 0..3 {
        let pred = ax * ay * cross[comp] / (TAU * TAU * omega);
        assert!(
            (cpl_ell[comp] - pred).abs() < 1e-5,
            "elliptical CPL[{comp}]: {:.6} vs analytic {:.6}",
            cpl_ell[comp],
            pred
        );
    }
}

#[test]
fn two_band_exotic_drive_first_order_commutator_is_cubic() {
    // For a(t) = κ(cos Ωt, sin(2Ωt+α)) each harmonic is linearly polarized,
    // so the O(κ²) first-order van Vleck commutator vanishes identically;
    // the leading correction is O(κ³) (mode 1's cos² feeds n = 2 and mixes
    // with mode 2's linear n = 2).  Verify the commutator part of the code's
    // d scales as κ³ (ratio 8 for κ→κ/2), in contrast to circular (κ², ratio 4).
    let (tx, ty, tz, m) = (1.0, 0.7, 0.5, 0.8);
    let omega = 8.0;
    let k = [0.21, 0.37];
    let model = two_band_qwz(tx, ty, tz, m, [[1.0, 0.0], [0.0, 1.0]]);

    let commutator_norm = |_kappa: f64, drive: FloquetDrive| -> f64 {
        let h1 = code_heff(&model, &k, &drive, 4);
        let h0 = code_heff_order0(&model, &k, &drive);
        let d1 = decompose_two_band(&h1).1;
        let d0 = decompose_two_band(&h0).1;
        ((d1[0] - d0[0]).powi(2) + (d1[1] - d0[1]).powi(2) + (d1[2] - d0[2]).powi(2)).sqrt()
    };

    let exo1 = commutator_norm(0.3, exotic_drive(0.3, 0.6, omega));
    let exo2 = commutator_norm(0.15, exotic_drive(0.15, 0.6, omega));
    let ratio_exo = exo1 / exo2;
    assert!(
        exo1 > 1e-8,
        "exotic-drive commutator must be non-vanishing at O(κ³), got {exo1:e}"
    );
    assert!(
        (ratio_exo - 8.0).abs() < 1.0,
        "exotic-drive commutator should scale as κ³ (ratio ≈ 8), got {ratio_exo:.2}"
    );

    let circ1 = commutator_norm(0.3, circular_drive(0.3, 1.0, omega));
    let circ2 = commutator_norm(0.15, circular_drive(0.15, 1.0, omega));
    let ratio_circ = circ1 / circ2;
    assert!(
        (ratio_circ - 4.0).abs() < 0.5,
        "circular-drive commutator should scale as κ² (ratio ≈ 4), got {ratio_circ:.2}"
    );
}
