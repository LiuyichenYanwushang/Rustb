//! Test-only reference implementation of the Peierls Fourier coefficients.
//!
//! The Floquet entry points evaluate `C_n(d)` with the grid-free Bessel
//! backend.  The uniform-time-grid DFT here is retained **only** as an
//! independent oracle for the tests — it is compiled for test builds alone, so
//! no public entry point depends on it and no sampling count can influence a
//! Floquet result.
//!
//! Keeping it out of `floquet.rs` is deliberate: the production module then
//! contains exactly one coefficient backend, and the cross-validation that
//! guards it lives beside the oracle it uses.

use ndarray::*;
use num_complex::Complex;
use std::f64::consts::TAU;

use crate::floquet::FloquetDrive;

/// Precomputed time-grid data for the discrete Fourier integration of
/// Peierls coefficients `C_n(d)`.
///
/// `mode_phases[it, mode]` stores the positive-carrier time phase. The
/// amplitudes are projected onto each link before applying these phases.
/// `fourier[i_n, it]` stores `exp(i * n * theta)` for each harmonic and time step.
/// Building these once avoids recomputing the same exponentials for every
/// hopping link.
pub(crate) struct FloquetTimeGrid {
    mode_phases: Array2<Complex<f64>>,
    fourier: Array2<Complex<f64>>,
    inv_n_time: f64,
}

impl FloquetTimeGrid {
    pub(crate) fn new(
        drive: &FloquetDrive,
        n_time: usize,
        harmonic_min: isize,
        harmonic_max: isize,
        _dim: usize,
    ) -> Self {
        let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
        let inv_n_time = 1.0 / (n_time as f64);
        let mut mode_phases = Array2::<Complex<f64>>::zeros((n_time, drive.modes.len()));
        let mut fourier = Array2::<Complex<f64>>::zeros((harmonic_count, n_time));

        for it in 0..n_time {
            let theta = TAU * (it as f64) * inv_n_time;
            for (index, mode) in drive.modes.iter().enumerate() {
                mode_phases[[it, index]] =
                    Complex::from_polar(1.0, -(mode.harmonic.unsigned_abs() as f64) * theta);
            }
            for (i_n, n) in (harmonic_min..=harmonic_max).enumerate() {
                fourier[[i_n, it]] = Complex::new(0.0, (n as f64) * theta).exp();
            }
        }

        Self {
            mode_phases,
            fourier,
            inv_n_time,
        }
    }
}

/// Reference Peierls coefficients by direct DFT on a uniform time grid:
/// `C_n = (1/N) Σ_it e^{i n θ_it} exp[-i a(t_it)·d]`.
///
/// Independent of the Bessel expansion (no Jacobi–Anger, no ladder, no
/// adaptive cutoff), which is exactly why the tests compare against it.
pub(crate) fn peierls_fourier_coeffs(
    d_cart: &Array1<f64>,
    harmonic_min: isize,
    harmonic_max: isize,
    drive: &FloquetDrive,
    time_grid: &FloquetTimeGrid,
) -> Vec<Complex<f64>> {
    let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
    if drive.modes.is_empty() {
        let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = Complex::new(1.0, 0.0);
        }
        return coeffs;
    }

    // Independent of the production projection helper: form scalar real/imag
    // projections explicitly, then sum equal physical carriers before sampling.
    let mut dc = 0.0;
    let mut projections = std::collections::BTreeMap::<usize, (usize, Complex<f64>)>::new();
    for (index, mode) in drive.modes.iter().enumerate() {
        let re = mode
            .a_complex
            .iter()
            .zip(d_cart)
            .map(|(a, x)| a.re * x)
            .sum::<f64>();
        if mode.harmonic == 0 {
            dc += re;
            continue;
        }
        let im = mode
            .a_complex
            .iter()
            .zip(d_cart)
            .map(|(a, x)| a.im * x)
            .sum::<f64>();
        let (_, total) = projections
            .entry(mode.harmonic.unsigned_abs())
            .or_insert((index, Complex::new(0.0, 0.0)));
        total.re += re;
        total.im += if mode.harmonic > 0 { im } else { -im };
    }
    let dc_phase = Complex::from_polar(1.0, -dc);
    let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
    for it in 0..time_grid.mode_phases.nrows() {
        let link_phase = projections
            .values()
            .map(|&(index, z)| {
                let phase = time_grid.mode_phases[[it, index]];
                z.re * phase.re - z.im * phase.im
            })
            .sum::<f64>();
        let peierls = dc_phase * Complex::from_polar(1.0, -link_phase);
        for (i_n, coeff) in coeffs.iter_mut().enumerate() {
            *coeff += time_grid.fourier[[i_n, it]] * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= time_grid.inv_n_time;
    }
    coeffs
}
