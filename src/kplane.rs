//! k-plane generation for Fermi surface calculations.
//!
//! Provides [`gen_kplane`] for generating a uniform 2D grid of k-points spanning
//! a plane in reciprocal space, defined by an origin and two spanning vectors.

use crate::error::{Result, TbError};
use crate::generics::usefloat;
use ndarray::{Array1, Array2};

/// Generate k-points on a plane in reciprocal space.
///
/// Points are generated as:
///
/// ```math
/// \mathbf{k}(i,j) = \text{origin} + \frac{i}{n_1}\mathbf{v}_1 + \frac{j}{n_2}\mathbf{v}_2
/// ```
///
/// for `i = 0..n1`, `j = 0..n2`. The first direction (`vec1`) varies fastest
/// (row-major order: `index = i + j * n1`).
///
/// # Arguments
/// * `origin` - Origin of the plane (fractional reciprocal coordinates)
/// * `vec1` - First spanning vector (fractional reciprocal coordinates)
/// * `vec2` - Second spanning vector (fractional reciprocal coordinates)
/// * `n1` - Number of sample points along `vec1`
/// * `n2` - Number of sample points along `vec2`
///
/// # Returns
/// `Result<Array2<f64>>` of shape `(n1 * n2, dim)` where `dim = origin.len()`.
///
/// # Errors
/// Returns `TbError::KVectorLengthMismatch` if `origin`, `vec1`, `vec2` have
/// different lengths.
#[inline(always)]
pub fn gen_kplane(
    origin: &Array1<f64>,
    vec1: &Array1<f64>,
    vec2: &Array1<f64>,
    n1: usize,
    n2: usize,
) -> Result<Array2<f64>> {
    let dim = origin.len();
    if vec1.len() != dim || vec2.len() != dim {
        return Err(TbError::KVectorLengthMismatch {
            expected: dim,
            actual: vec1.len().max(vec2.len()),
        });
    }

    let nk = n1 * n2;
    let mut kvec = Array2::<f64>::zeros((nk, dim));

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;

    for j in 0..n2 {
        for i in 0..n1 {
            let idx = i + j * n1;
            let ti = i as f64 / n1_f;
            let tj = j as f64 / n2_f;
            for d in 0..dim {
                kvec[[idx, d]] = origin[d] + ti * vec1[d] + tj * vec2[d];
            }
        }
    }

    Ok(kvec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_gen_kplane_2d() {
        let origin = arr1(&[0.0, 0.0]);
        let vec1 = arr1(&[1.0, 0.0]);
        let vec2 = arr1(&[0.0, 1.0]);
        let k = gen_kplane(&origin, &vec1, &vec2, 2, 2).unwrap();
        assert_eq!(k.shape(), &[4, 2]);
        // i=0,j=0 → (0,0)
        assert!((k[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((k[[0, 1]] - 0.0).abs() < 1e-10);
        // i=1,j=0 → (0.5, 0)
        assert!((k[[1, 0]] - 0.5).abs() < 1e-10);
        assert!((k[[1, 1]] - 0.0).abs() < 1e-10);
        // i=0,j=1 → (0, 0.5)
        assert!((k[[2, 0]] - 0.0).abs() < 1e-10);
        assert!((k[[2, 1]] - 0.5).abs() < 1e-10);
        // i=1,j=1 → (0.5, 0.5)
        assert!((k[[3, 0]] - 0.5).abs() < 1e-10);
        assert!((k[[3, 1]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gen_kplane_3d() {
        let origin = arr1(&[0.0, 0.0, 0.5]);
        let vec1 = arr1(&[1.0, 0.0, 0.0]);
        let vec2 = arr1(&[0.0, 1.0, 0.0]);
        let k = gen_kplane(&origin, &vec1, &vec2, 2, 2).unwrap();
        assert_eq!(k.shape(), &[4, 3]);
        assert!((k[[0, 2]] - 0.5).abs() < 1e-10);
        assert!((k[[3, 2]] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gen_kplane_length_mismatch() {
        let origin = arr1(&[0.0, 0.0]);
        let vec1 = arr1(&[1.0, 0.0]);
        let vec2 = arr1(&[0.0, 1.0, 0.0]); // wrong length
        let result = gen_kplane(&origin, &vec1, &vec2, 2, 2);
        assert!(result.is_err());
    }
}
