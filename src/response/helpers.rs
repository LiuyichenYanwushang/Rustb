//! Shared helper functions used across response modules.

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

use crate::SpinDirection;

/// Directly construct spin Pauli matrix σ⊗I_{norb}/2 without kron.
/// Only sets 2*norb non-zero elements (O(norb)) instead of O(nsta²).
#[inline]
pub(crate) fn build_spin_matrix(norb: usize, spin: Option<SpinDirection>) -> Array2<Complex<f64>> {
    let nsta = 2 * norb;
    let mut m = Array2::<Complex<f64>>::zeros((nsta, nsta));
    let half = Complex::new(0.5, 0.0);
    let i_half = Complex::new(0.0, 0.5);
    match spin {
        None => {
            m = Array2::<Complex<f64>>::eye(2 * norb);
        }
        Some(SpinDirection::X) => {
            for i in 0..norb {
                m[[i, i + norb]] = half;
                m[[i + norb, i]] = half;
            }
        }
        Some(SpinDirection::Y) => {
            for i in 0..norb {
                m[[i, i + norb]] = -i_half;
                m[[i + norb, i]] = i_half;
            }
        }
        Some(SpinDirection::Z) => {
            for i in 0..norb {
                m[[i, i]] = half;
                m[[i + norb, i + norb]] = -half;
            }
        }
    }
    m
}
