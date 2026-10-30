//! Utility functions and macros for tight-binding model operations
//!
//! # Examples
//!
//! ```rust
//! use ndarray::{arr2, array};
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
//! let orb = arr2(&[[0.0, 0.0]]);
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! model.add_hop(-1.0, 0, 0, &array![1, 0], None);
//!
//! // Index of the R = (0, 0) block; unknown vectors return None.
//! let origin = find_R(&model.hamR, &array![0, 0]);
//! assert_eq!(origin, Some(0));
//! assert!(find_R(&model.hamR, &array![1, 0]).is_some());
//! assert!(find_R(&model.hamR, &array![7, 7]).is_none());
//!
//! let smaller = remove_row(model.hamR.clone(), 0);
//! assert_eq!(smaller.nrows() + 1, model.hamR.nrows());
//! let fewer_cols = remove_col(arr2(&[[1.0, 2.0], [3.0, 4.0]]), 1);
//! assert_eq!(fewer_cols.dim(), (2, 1));
//! # Ok(())
//! # }
//! ```

use ndarray::prelude::*;

use ndarray::*;

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
