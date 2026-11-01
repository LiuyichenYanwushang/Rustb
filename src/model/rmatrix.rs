use ndarray::{Array2, Array4, Axis};
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

/// Trait marking whether a model stores position matrix elements.
pub trait RMatrixData: Clone + std::fmt::Debug + Sync {
    const HAS_RMATRIX: bool;
    /// Create the default rmatrix with orbital positions on the diagonal.
    ///
    /// Position matrix elements are stored in **Cartesian** coordinates
    /// (matching Wannier90 `_r.dat`), so the fractional orbital positions are
    /// converted via `cart = frac · lat`.
    fn from_orb(orb: &Array2<f64>, lat: &Array2<f64>, norb: usize, spin: bool, dim: usize) -> Self;
    /// Wrap an Array4 into the RMatrixData type.
    fn from_array(arr: Array4<Complex<f64>>) -> Self;
    /// Get a reference to the underlying Array4. Panics for NoRMatrix.
    fn as_array4(&self) -> &Array4<Complex<f64>>;
    /// Get a mutable reference to the underlying Array4. Panics for NoRMatrix.
    fn as_array4_mut(&mut self) -> &mut Array4<Complex<f64>>;
    fn select(&self, axis: Axis, indices: &[usize]) -> Self {
        if Self::HAS_RMATRIX {
            Self::from_array(self.as_array4().select(axis, indices))
        } else {
            self.clone()
        }
    }
}

/// Position matrix elements are stored. Wraps [`Array4<Complex<f64>>`] with
/// zero-overhead newtype.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HasRMatrix(pub Array4<Complex<f64>>);

impl RMatrixData for HasRMatrix {
    const HAS_RMATRIX: bool = true;
    fn from_orb(orb: &Array2<f64>, lat: &Array2<f64>, norb: usize, spin: bool, dim: usize) -> Self {
        let nsta = if spin { 2 * norb } else { norb };
        let cart = orb.dot(lat);
        let mut r = Array4::<Complex<f64>>::zeros((1, dim, nsta, nsta));
        for i in 0..norb {
            for ri in 0..dim {
                r[[0, ri, i, i]] = Complex::<f64>::from(cart[[i, ri]]);
                if spin {
                    r[[0, ri, i + norb, i + norb]] = Complex::<f64>::from(cart[[i, ri]]);
                }
            }
        }
        HasRMatrix(r)
    }
    fn from_array(arr: Array4<Complex<f64>>) -> Self {
        HasRMatrix(arr)
    }
    fn as_array4(&self) -> &Array4<Complex<f64>> {
        &self.0
    }
    fn as_array4_mut(&mut self) -> &mut Array4<Complex<f64>> {
        &mut self.0
    }
}

impl Deref for HasRMatrix {
    type Target = Array4<Complex<f64>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HasRMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// No position matrix elements stored. Zero-sized type.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NoRMatrix;

impl RMatrixData for NoRMatrix {
    const HAS_RMATRIX: bool = false;
    fn from_orb(
        _orb: &Array2<f64>,
        _lat: &Array2<f64>,
        _norb: usize,
        _spin: bool,
        _dim: usize,
    ) -> Self {
        NoRMatrix
    }
    fn from_array(_arr: Array4<Complex<f64>>) -> Self {
        NoRMatrix
    }
    fn as_array4(&self) -> &Array4<Complex<f64>> {
        panic!("NoRMatrix has no underlying Array4; only HasRMatrix supports this operation")
    }
    fn as_array4_mut(&mut self) -> &mut Array4<Complex<f64>> {
        panic!("NoRMatrix has no underlying Array4; only HasRMatrix supports this operation")
    }
}
