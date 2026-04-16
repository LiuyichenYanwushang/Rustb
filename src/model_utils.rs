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

//读取VASP代码的CHGCAR
