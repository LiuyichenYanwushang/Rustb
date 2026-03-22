//! Utility functions and macros for tight-binding model operations

use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

/// Find the index of lattice vector $\mathbf{R}$ in the `hamR` array.
///
/// This utility function searches for a specific lattice vector in the array
/// of all hopping vectors and returns its index if found.
///
/// # Arguments
/// * `hamR` - Array of all lattice vectors $\mathbf{R}$ for hoppings
/// * `R` - Target lattice vector to search for
///
/// # Returns
/// `Option<usize>` containing the index if found, `None` otherwise
#[allow(non_snake_case)]
#[inline(always)]
pub fn find_R<A: Data<Elem = T>, B: Data<Elem = T>, T: std::cmp::PartialEq>(
    hamR: &ArrayBase<A, Ix2>,
    R: &ArrayBase<B, Ix1>,
) -> Option<usize> {
    let n_R: usize = hamR.len_of(Axis(0));
    let dim_R: usize = hamR.len_of(Axis(1));
    for i in 0..(n_R) {
        let mut a = true;
        for j in 0..(dim_R) {
            a = a && (hamR[[i, j]] == R[[j]]);
        }
        if a {
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

/// Macro to update Hamiltonian matrix elements with spin consideration
///
/// This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
/// It takes a Hamiltonian and returns a new Hamiltonian.
macro_rules! update_hamiltonian {
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                crate::model_enums::SpinDirection::None => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = $tmp;
                }
                crate::model_enums::SpinDirection::x => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] = $tmp;
                }
                crate::model_enums::SpinDirection::y => {
                    $new_ham[[$ind_i + $norb, $ind_j]] = $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] = -$tmp * Complex::<f64>::i();
                }
                crate::model_enums::SpinDirection::z => {
                    $new_ham[[$ind_i, $ind_j]] = $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] = -$tmp;
                }
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] = $tmp;
        }
        $new_ham
    }};
}

/// Macro to add to Hamiltonian matrix elements with spin consideration
///
/// This macro adds to the Hamiltonian, checking for spin and the indices ind_i, ind_j.
/// It takes a Hamiltonian and returns a new Hamiltonian.
macro_rules! add_hamiltonian {
    // This macro updates the Hamiltonian, checking for spin and the indices ind_i, ind_j.
    // It takes a Hamiltonian and returns a new Hamiltonian.
    ($spin:expr, $pauli:expr, $tmp:expr, $new_ham:expr, $ind_i:expr, $ind_j:expr,$norb:expr) => {{
        if $spin {
            match $pauli {
                crate::model_enums::SpinDirection::None => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] += $tmp;
                }
                crate::model_enums::SpinDirection::x => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp;
                    $new_ham[[$ind_i, $ind_j + $norb]] += $tmp;
                }
                crate::model_enums::SpinDirection::y => {
                    $new_ham[[$ind_i + $norb, $ind_j]] += $tmp * Complex::<f64>::i();
                    $new_ham[[$ind_i, $ind_j + $norb]] -= $tmp * Complex::<f64>::i();
                }
                crate::model_enums::SpinDirection::z => {
                    $new_ham[[$ind_i, $ind_j]] += $tmp;
                    $new_ham[[$ind_i + $norb, $ind_j + $norb]] -= $tmp;
                }
            }
        } else {
            $new_ham[[$ind_i, $ind_j]] += $tmp;
        }
        $new_ham
    }};
}

// Export the macros
pub(crate) use add_hamiltonian;
pub(crate) use update_hamiltonian;
