//! Utility functions and macros

use ndarray::prelude::*;
use ndarray::*;
use num_complex::{Complex, Complex64};

use crate::generics::hop_use;
use crate::SpinDirection;

/// Find the index of a lattice vector R in the hamR array
pub fn find_R<A: Data<Elem = T>, B: Data<Elem = T>, T: std::cmp::PartialEq>(
    hamR: &ArrayBase<A, Ix2>,
    R: &ArrayBase<B, Ix1>,
) -> Option<usize> {
    for (i, row) in hamR.axis_iter(Axis(0)).enumerate() {
        if row == R {
            return Some(i);
        }
    }
    None
}

/// Remove a row from a 2D array
#[inline(always)]
pub fn remove_row<T: Copy>(array: Array2<T>, row_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.nrows()).filter(|&r| r != row_to_remove).collect();
    array.select(Axis(0), &indices)
}

/// Remove a column from a 2D array
#[inline(always)]
pub fn remove_col<T: Copy>(array: Array2<T>, col_to_remove: usize) -> Array2<T> {
    let indices: Vec<_> = (0..array.ncols()).filter(|&r| r != col_to_remove).collect();
    array.select(Axis(1), &indices)
}

/// Macro for updating Hamiltonian matrix elements
#[macro_export]
macro_rules! update_hamiltonian {
    ($spin:expr, $pauli:expr, $tmp:expr, $ham:expr, $ind_i:expr, $ind_j:expr, $norb:expr) => {
        {
            let mut ham = $ham;
            if $spin {
                match $pauli {
                    SpinDirection::None => {
                        ham[[$ind_i, $ind_j]] = $tmp;
                        ham[[$ind_i + $norb, $ind_j + $norb]] = $tmp;
                    }
                    SpinDirection::x => {
                        ham[[$ind_i, $ind_j + $norb]] = $tmp;
                        ham[[$ind_i + $norb, $ind_j]] = $tmp;
                    }
                    SpinDirection::y => {
                        ham[[$ind_i, $ind_j + $norb]] = Complex::new(0.0, -1.0) * $tmp;
                        ham[[$ind_i + $norb, $ind_j]] = Complex::new(0.0, 1.0) * $tmp;
                    }
                    SpinDirection::z => {
                        ham[[$ind_i, $ind_j]] = $tmp;
                        ham[[$ind_i + $norb, $ind_j + $norb]] = -$tmp;
                    }
                }
            } else {
                ham[[$ind_i, $ind_j]] = $tmp;
            }
            ham
        }
    };
}

/// Macro for adding to Hamiltonian matrix elements
#[macro_export]
macro_rules! add_hamiltonian {
    ($spin:expr, $pauli:expr, $tmp:expr, $ham:expr, $ind_i:expr, $ind_j:expr, $norb:expr) => {
        {
            let mut ham = $ham;
            if $spin {
                match $pauli {
                    SpinDirection::None => {
                        ham[[$ind_i, $ind_j]] += $tmp;
                        ham[[$ind_i + $norb, $ind_j + $norb]] += $tmp;
                    }
                    SpinDirection::x => {
                        ham[[$ind_i, $ind_j + $norb]] += $tmp;
                        ham[[$ind_i + $norb, $ind_j]] += $tmp;
                    }
                    SpinDirection::y => {
                        ham[[$ind_i, $ind_j + $norb]] += Complex::new(0.0, -1.0) * $tmp;
                        ham[[$ind_i + $norb, $ind_j]] += Complex::new(0.0, 1.0) * $tmp;
                    }
                    SpinDirection::z => {
                        ham[[$ind_i, $ind_j]] += $tmp;
                        ham[[$ind_i + $norb, $ind_j + $norb]] += -$tmp;
                    }
                }
            } else {
                ham[[$ind_i, $ind_j]] += $tmp;
            }
            ham
        }
    };
}