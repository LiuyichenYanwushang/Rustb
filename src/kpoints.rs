//! k-point generation and Brillouin zone sampling utilities.
//!
//! This module provides functions for generating k-point meshes in the Brillouin zone
//! for numerical integration and band structure calculations. The k-points are
//! uniformly distributed in reciprocal space.
//!
//! # Examples
//! ```
//! use ndarray::{array,Array2};
//! use Rustb::kpoints::gen_kmesh;
//!
//! // Generate a 10×10 k-mesh for a 2D system
//! let kmesh:Array2<f64> = gen_kmesh(&array![10, 10]).unwrap();
//! ```

use crate::error::{Result, TbError};
use crate::generics::UseFloat;
use ndarray::{Array1, Array2, Array3};

fn mesh_len<T>(k_mesh: &Array1<usize>, values_per_point: usize) -> Result<usize> {
    let invalid = || TbError::InvalidKmeshDimensions(k_mesh.to_owned());
    if k_mesh.is_empty() || k_mesh.iter().any(|&n| n == 0) {
        return Err(invalid());
    }
    let points = k_mesh
        .iter()
        .try_fold(1usize, |n, &m| n.checked_mul(m))
        .ok_or_else(invalid)?;
    points
        .checked_mul(values_per_point)
        .and_then(|n| n.checked_mul(size_of::<T>().max(1)))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(invalid)?;
    Ok(points)
}

/// Generate a uniform k-point mesh in the Brillouin zone.
///
/// The k-points are distributed uniformly with coordinates in the range [0, 1)
/// in fractional reciprocal space coordinates. For a 2D system with mesh [Nx, Ny],
/// the k-points are arranged as:
/// $$
/// \mathbf{k} = \left(\frac{i}{N_x}, \frac{j}{N_y}\right) \quad \text{for } i=0,\ldots,N_x-1, j=0,\ldots,N_y-1
/// $$
///
/// # Arguments
/// * `k_mesh` - Array specifying the number of points along each reciprocal lattice direction
///
/// # Returns
/// `Result<Array2<T>>` where each row is a k-point in fractional coordinates
///
/// # Errors
/// Returns `TbError` if the mesh dimensions are invalid
pub fn gen_kmesh<T>(k_mesh: &Array1<usize>) -> Result<Array2<T>>
where
    T: UseFloat + std::ops::Div<Output = T>,
{
    let dim = k_mesh.len();
    let count = mesh_len::<T>(k_mesh, dim)?;
    let mut points = Array2::zeros((count, dim));
    for (index, mut point) in points.outer_iter_mut().enumerate() {
        let mut remainder = index;
        for axis in (0..dim).rev() {
            point[axis] = T::from(remainder % k_mesh[axis]) / T::from(k_mesh[axis]);
            remainder /= k_mesh[axis];
        }
    }
    Ok(points)
}

/// Generate fractional-coordinate cell bounds corresponding to [`gen_kmesh`].
/// The last coordinate varies fastest; each cell includes its upper endpoint.
/// Supports one, two and three dimensions, rejecting zero or overflowing sizes.
pub fn gen_krange<T>(k_mesh: &Array1<usize>) -> Result<Array3<T>>
where
    T: UseFloat + std::ops::Div<Output = T>,
{
    let dim = k_mesh.len();
    if !(1..=3).contains(&dim) {
        return Err(TbError::InvalidDimension {
            dim,
            supported: vec![1, 2, 3],
        });
    }
    let count = mesh_len::<T>(k_mesh, 2 * dim)?;
    let mut ranges = Array3::zeros((count, dim, 2));
    for (index, mut cell) in ranges.outer_iter_mut().enumerate() {
        let mut remainder = index;
        for axis in (0..dim).rev() {
            let coordinate = remainder % k_mesh[axis];
            for endpoint in 0..2 {
                cell[[axis, endpoint]] = T::from(coordinate + endpoint) / T::from(k_mesh[axis]);
            }
            remainder /= k_mesh[axis];
        }
    }
    Ok(ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn test_gen_kmesh() {
        let kmesh: Array2<f64> = gen_kmesh(&array![2, 2]).unwrap();
        assert_eq!(
            kmesh,
            array![[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
        );
        assert_eq!(
            gen_kmesh::<f64>(&array![4]).unwrap(),
            array![[0.0], [0.25], [0.5], [0.75]]
        );
        assert_eq!(gen_kmesh::<f32>(&array![1]).unwrap(), array![[0.0f32]]);
    }

    #[test]
    fn mesh_and_cell_bounds_have_the_same_order() {
        for shape in [array![3], array![2, 3], array![2, 1, 3]] {
            let mesh = gen_kmesh::<f64>(&shape).unwrap();
            let ranges = gen_krange::<f64>(&shape).unwrap();
            for point in 0..mesh.nrows() {
                for axis in 0..shape.len() {
                    assert_eq!(mesh[[point, axis]], ranges[[point, axis, 0]]);
                    assert!(
                        (ranges[[point, axis, 1]] - mesh[[point, axis]] - 1.0 / shape[axis] as f64)
                            .abs()
                            < 1e-15
                    );
                }
            }
        }
        let mesh = gen_kmesh::<f64>(&array![2, 1, 3]).unwrap();
        assert_eq!(mesh.row(3), array![0.5, 0.0, 0.0]);
        assert_eq!(mesh.row(5), array![0.5, 0.0, 2.0 / 3.0]);
    }

    #[test]
    fn meshes_reject_zero_and_overflowing_sizes() {
        for shape in [
            array![],
            array![0],
            array![2, 0],
            array![usize::MAX, 2],
            array![isize::MAX as usize],
        ] {
            assert!(gen_kmesh::<f64>(&shape).is_err());
            assert!(gen_krange::<f64>(&shape).is_err());
        }
    }
}
