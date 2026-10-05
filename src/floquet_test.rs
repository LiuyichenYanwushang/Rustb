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
/// `link_field[it, a]` stores the real part of the total dimensionless
/// vector potential `a_a(t_it)` for each time step and spatial direction.
/// `fourier[i_n, it]` stores `exp(i * n * theta)` for each harmonic and time step.
/// Building these once avoids recomputing the same exponentials for every
/// hopping link.
pub(crate) struct FloquetTimeGrid {
    link_field: Array2<f64>,
    fourier: Array2<Complex<f64>>,
    inv_n_time: f64,
}

impl FloquetTimeGrid {
    pub(crate) fn new(
        drive: &FloquetDrive,
        n_time: usize,
        harmonic_min: isize,
        harmonic_max: isize,
        dim: usize,
    ) -> Self {
        let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
        let inv_n_time = 1.0 / (n_time as f64);
        let mut link_field = Array2::<f64>::zeros((n_time, dim));
        let mut fourier = Array2::<Complex<f64>>::zeros((harmonic_count, n_time));

        for it in 0..n_time {
            let theta = TAU * (it as f64) * inv_n_time;
            for mode in &drive.modes {
                let harmonic_phase = Complex::new(0.0, -(mode.harmonic as f64) * theta).exp();
                for a in 0..dim {
                    link_field[[it, a]] += (mode.a_complex[a] * harmonic_phase).re;
                }
            }
            for (i_n, n) in (harmonic_min..=harmonic_max).enumerate() {
                fourier[[i_n, it]] = Complex::new(0.0, (n as f64) * theta).exp();
            }
        }

        Self {
            link_field,
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

    let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
    for it in 0..time_grid.link_field.nrows() {
        let mut link_phase = 0.0;
        for a in 0..d_cart.len() {
            link_phase += time_grid.link_field[[it, a]] * d_cart[a];
        }
        let peierls = Complex::new(0.0, -link_phase).exp();
        for (i_n, coeff) in coeffs.iter_mut().enumerate() {
            *coeff += time_grid.fourier[[i_n, it]] * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= time_grid.inv_n_time;
    }
    coeffs
}
