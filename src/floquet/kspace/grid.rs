//! Zero-aliasing uniform grids for polynomial evaluation in k space.

use crate::error::{Result, TbError};
use ndarray::Array2;

/// A uniform k-point grid used to evaluate a real-space polynomial and map the
/// result back exactly.
///
/// The effective Hamiltonian at order p is a polynomial of degree p + 1 in the
/// hopping blocks, so its real-space support is contained in (p + 1) times the
/// input support. Sampling N_a >= 2 * degree * R_a,max + 1 points per axis makes
/// every aliased image vanish (Poisson summation), so the inverse transform
/// reproduces the real-space coefficients without approximation.
#[derive(Clone, Debug)]
pub(super) struct KspaceGrid {
    shape: [usize; 3],
    dim: usize,
    support: Array2<isize>,
}

impl KspaceGrid {
    /// Build the smallest smooth zero-aliasing grid for a polynomial degree.
    ///
    /// The support holds one translation per row; degree is the polynomial
    /// degree whose support must be resolved (2 for O(1/W), 3 for O(1/W^2)).
    pub(super) fn for_polynomial(
        support: &Array2<isize>,
        dim: usize,
        degree: usize,
    ) -> Result<Self> {
        if !(1..=3).contains(&dim) || support.ncols() != dim {
            return Err(TbError::Other(format!(
                "k-space grid needs 1..=3 translation components, got {} for dim {dim}",
                support.ncols()
            )));
        }
        if degree == 0 {
            return Err(TbError::Other(
                "k-space grid requires a positive polynomial degree".into(),
            ));
        }
        let mut shape = [1_usize; 3];
        for axis in 0..dim {
            let column = support.column(axis);
            let radius = column.iter().map(|r| r.unsigned_abs()).max().unwrap_or(0);
            let span = radius
                .checked_mul(2 * degree)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| TbError::Other("k-space grid extent overflows usize".into()))?;
            shape[axis] = next_smooth(span.max(1));
        }
        Ok(Self {
            shape,
            dim,
            support: support.clone(),
        })
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn shape(&self) -> [usize; 3] {
        self.shape
    }

    pub(super) fn dim(&self) -> usize {
        self.dim
    }

    pub(super) fn support(&self) -> &Array2<isize> {
        &self.support
    }

    /// Number of k points in one slice of the streaming axis (the last axis).
    ///
    /// This is the product over the first dim - 1 axes, so a 1D model has one
    /// point per slice and a 2D model has N_0 points per slice.
    pub(super) fn points_per_slice(&self) -> usize {
        self.shape[..self.dim - 1].iter().product()
    }

    /// Number of slices along the last resolved axis.
    pub(super) fn slices(&self) -> usize {
        self.shape[self.dim - 1]
    }

    /// Total number of k points on the grid.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fn total_points(&self) -> usize {
        self.points_per_slice() * self.slices()
    }

    /// Number of transform axes inside one slice (dim - 1).
    pub(super) fn slice_axes(&self) -> usize {
        self.dim - 1
    }

    /// Fold a translation component into 0..N.
    pub(super) fn folded(&self, value: isize, axis: usize) -> usize {
        value.rem_euclid(self.shape[axis] as isize) as usize
    }

    /// Signed representative of a folded index on an axis.
    pub(super) fn signed(&self, index: usize, axis: usize) -> isize {
        let n = self.shape[axis] as isize;
        if index as isize > n / 2 {
            index as isize - n
        } else {
            index as isize
        }
    }

    /// Length of the transform along an axis.
    pub(super) fn axis_len(&self, axis: usize) -> usize {
        self.shape[axis]
    }
}

/// Smallest 2/3/5/7-smooth integer not below n, so FFT sizes stay efficient.
pub(super) fn next_smooth(n: usize) -> usize {
    let mut candidate = n.max(1);
    loop {
        let mut rest = candidate;
        for factor in [2_usize, 3, 5, 7] {
            while rest % factor == 0 {
                rest /= factor;
            }
        }
        if rest == 1 {
            return candidate;
        }
        candidate += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn grid_bounds_every_aliased_image() {
        let support = array![[-2_isize, 0, 3], [1, -1, 0]];
        let grid = KspaceGrid::for_polynomial(&support, 3, 3).unwrap();
        let shape = grid.shape();
        for axis in 0..3 {
            let radius = support
                .column(axis)
                .iter()
                .map(|r| r.unsigned_abs())
                .max()
                .unwrap();
            assert!(shape[axis] >= 2 * 3 * radius + 1, "{shape:?}");
            assert!(shape[axis] >= 1);
        }
        assert_eq!(
            (grid.folded(-1, 0) + 1) % grid.axis_len(0),
            grid.folded(0, 0)
        );
        assert_eq!(grid.signed(grid.folded(-1, 0), 0), -1);
        assert_eq!(grid.signed(grid.folded(0, 0), 0), 0);
        assert_eq!(grid.total_points(), shape[0] * shape[1] * shape[2]);
    }

    #[test]
    fn smooth_sizes_stay_close_to_the_bound() {
        for n in [1_usize, 2, 3, 7, 8, 11, 55, 56, 100] {
            let smooth = next_smooth(n);
            assert!(smooth >= n);
            assert!(smooth <= n + 6, "{n} -> {smooth}");
        }
    }

    #[test]
    fn flat_supports_collapse_the_axis() {
        let support = array![[0_isize, 0], [2, 0]];
        let grid = KspaceGrid::for_polynomial(&support, 2, 3).unwrap();
        let shape = grid.shape();
        assert_eq!(shape[2], 1);
        assert_eq!(grid.slices(), shape[1]);
        assert_eq!(grid.points_per_slice(), shape[0]);
        assert_eq!(grid.slice_axes(), 1);
    }
}
