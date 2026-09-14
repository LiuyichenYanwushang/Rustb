//! Geometric and topological quantities computed via the Wilson loop method.
//!
//! This module provides the [`Berry`] trait with methods for:
//!
//! - Berry phase along a closed k-space loop.
//! - Berry curvature (flux) on a 2D k-mesh.
//! - Wannier centres (hybrid Wannier functions) via Wilson loops.
//!
//! # Wilson loop algorithm
//!
//! For a closed loop of k-points \\(\lbrace\mathbf k_i\rbrace\\), the overlap matrix is
//!
//! $$
//! F_{mn,\mathbf k} = \langle \psi_{m,\mathbf k} | \psi_{n,\mathbf k+\Delta\mathbf k} \rangle.
//! $$
//!
//! The overlap matrices are orthonormalized via SVD: \\(F = U V^\dagger\\).  The
//! Wilson loop is the product
//!
//! $$
//! W = \prod_i F_{\mathbf k_i},
//! $$
//!
//! whose eigenvalues `lambda` give returned phases `-arg(lambda)` in radians.
//! Divide these phases by `2*pi` for centres in lattice units.
//!
//! For a closed loop that wraps the Brillouin zone, the Bloch functions at the
//! endpoints are related by a phase factor:
//!
//! $$
//! |u_{n,\mathbf k_{\text{end}}}\rangle =
//! e^{-2\pi i \mathbf{G}\cdot\bm\tau} |u_{n,\mathbf k_{\text{first}}}\rangle.
//! $$
//!
//! Here `G = k_end - k_start` is the integer reciprocal displacement.
//!
//! # Examples
//!
//! ```
//! use Rustb::{Berry, Model};
//! use ndarray::array;
//!
//! let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.2]], None)?;
//! let loop_k = array![[0.0], [0.25], [0.5], [0.75], [1.0]];
//! let phases = model.berry_loop(&loop_k, &[0])?;
//! assert!((phases[0] - 0.4 * std::f64::consts::PI).abs() < 1e-12);
//! # Ok::<(), Rustb::error::TbError>(())
//! ```

use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::solve_ham::Solve;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

#[cfg(test)]
mod tests;

/// Trait for computing Berry-phase, Berry curvature, and Wannier centre
/// quantities via the Wilson loop method.
///
/// All methods return `Result`. Selected bands must be nonempty, distinct and
/// in range. Loops need at least two finite k-points of the model's dimension,
/// closed modulo an integer reciprocal vector within `1e-9` per coordinate.
/// The selected subspace must be isolated and the sampling fine enough for
/// nonsingular overlaps; these methods do not certify a spectral gap.
///
/// Phases are in radians, with convention `-arg(W)` (not divided by `2*pi`).
/// Batch and Wannier results keep the historical axis order `(n_occ, n_loop)`.
pub trait Berry {
    /// Compute the Berry phase (Wannier centres) along a closed k-space loop
    /// using Wilson loops.
    ///
    /// # Parameters
    ///
    /// - `kvec`: array of k-points defining the loop (shape `(N_k, dim_r)`).
    ///   The loop must close modulo a reciprocal lattice vector.
    /// - `occ`: indices of the occupied bands.
    ///
    /// # Returns
    ///
    /// An `Array1<f64>` of Wilson eigenphases in radians, without sorting.
    ///
    /// # Algorithm
    ///
    /// The overlap matrices \\(F_{mn,\mathbf k}\\) are computed between adjacent
    /// k-points, orthonormalized via SVD (\\(F = UV^\dagger\\)), and multiplied
    /// along the loop.  The phases of the eigenvalues of the product give the
    /// Wannier centres.
    ///
    /// # Errors
    ///
    /// Returns errors for invalid models, bands or loops, singular/nonfinite
    /// overlaps, and Hamiltonian, SVD or Wilson eigensolver failures.
    fn berry_loop<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &[usize]) -> Result<Array1<f64>>
    where
        S: Data<Elem = f64>;

    /// Compute the Berry phase along a closed loop without SVD orthonormalization,
    /// using the determinant instead.
    ///
    /// This captures the total Berry phase of all occupied bands.
    ///
    /// Returns the total phase in radians.
    ///
    /// # Errors
    ///
    /// Returns errors for invalid models, bands or loops, a zero/nonfinite
    /// determinant (including underflow), and linear algebra failures.
    fn berry_loop_det<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &[usize]) -> Result<f64>
    where
        S: Data<Elem = f64>;

    /// Compute the Berry curvature (flux) on a 2D k-mesh using Wilson loops.
    ///
    /// # Parameters
    ///
    /// - `occ`: occupied band indices.
    /// - `k_start`: starting k-point.
    /// - `dir_1`: first reciprocal lattice direction.
    /// - `dir_2`: second reciprocal lattice direction.
    /// - `nk1`, `nk2`: number of k-points in each direction.
    ///
    /// # Returns
    ///
    /// An `Array3<f64>` of shape `(nk1, nk2, n_occ)` containing the Berry flux
    /// through each plaquette, in radians. Plaquette order is start, +dir_1,
    /// +dir_1+dir_2, +dir_2, start; directions span the full sampled region.
    ///
    /// This method uses Wilson loops (high accuracy, fast convergence) but
    /// requires an isolated selected band subspace.
    ///
    /// # Errors
    /// Requires finite, correctly sized vectors and `nk1, nk2 >= 1`.
    /// Rejects overflowing grid sizes and propagates [`Self::berry_loop`] errors.
    fn berry_flux(
        &self,
        occ: &[usize],
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Result<Array3<f64>>;

    /// Compute Berry phases along closed k-space loops.
    ///
    /// Each slice `kvec[i, :, :]` defines one loop.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` of shape `(n_occ, n_loop)`. Each column contains
    /// the independently sorted phases in radians; no branch tracking is done.
    /// Zero loops return `(n_occ, 0)` after validating model, bands and loop shape.
    ///
    /// # Errors
    /// Propagates [`Self::berry_loop`] errors; rejects overflowing output sizes.
    fn berry_phase(&self, occ: &[usize], kvec: &Array3<f64>) -> Result<Array2<f64>>;

    /// Compute hybrid Wannier centres via Wilson loops.
    ///
    /// The Wilson loop is taken along `dir_2` (the integration direction),
    /// while `dir_1` is the transverse direction that is sampled.
    ///
    /// # Parameters
    ///
    /// - `occ`: occupied band indices.
    /// - `k_start`: origin of the 2D k-mesh.
    /// - `dir_1`: transverse direction (sampled at `nk1` points).
    /// - `dir_2`: integration direction (`nk2` points per loop).
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` of shape `(n_occ, nk1)` with independently sorted
    /// phases in radians. Divide by `2*pi` for centres in lattice units.
    /// Both directions include their endpoints. No branch tracking is done.
    ///
    /// # Errors
    /// Requires finite, correctly sized vectors, `nk1, nk2 >= 2` and an
    /// integer reciprocal `dir_2` within the loop-closure tolerance. Rejects
    /// overflowing grids and propagates [`Self::berry_phase`] errors.
    fn wannier_centre(
        &self,
        occ: &[usize],
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Result<Array2<f64>>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Berry for Model<SPIN, DIM, R> {
    fn berry_loop<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &[usize]) -> Result<Array1<f64>>
    where
        S: Data<Elem = f64>,
    {
        let mut ovr = self.wilson_overlaps(kvec, occ)?;
        let n_occ = occ.len();
        for mut O in ovr.outer_iter_mut() {
            let (U, singular_values, V) = O.svd(true, true)?;
            if singular_values.iter().any(|&s| !s.is_finite() || s <= 0.0) {
                return Err(TbError::Other(
                    "Wilson overlap is singular or nonfinite".into(),
                ));
            }
            let U = U.ok_or(TbError::SvdComputationFailed)?;
            let V = V.ok_or(TbError::SvdComputationFailed)?;
            O.assign(&U.dot(&V));
        }
        let result: Array2<Complex<f64>> = ovr.outer_iter().fold(
            Array2::from_diag(&Array1::<Complex<f64>>::ones(n_occ)),
            |acc, x| acc.dot(&x),
        );
        let result = result.eigvals()?;
        if result
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite() || *z == Complex::new(0.0, 0.0))
        {
            return Err(TbError::EigenvalueComputationFailed);
        }
        Ok(result.mapv(|x| -x.arg()))
    }

    fn berry_loop_det<S>(&self, kvec: &ArrayBase<S, Ix2>, occ: &[usize]) -> Result<f64>
    where
        S: Data<Elem = f64>,
    {
        let ovr = self.wilson_overlaps(kvec, occ)?;
        let n_occ = occ.len();
        let result: Array2<Complex<f64>> = ovr.outer_iter().fold(
            Array2::from_diag(&Array1::<Complex<f64>>::ones(n_occ)),
            |acc, x| acc.dot(&x),
        );
        let determinant = result.det()?;
        if !determinant.re.is_finite()
            || !determinant.im.is_finite()
            || determinant == Complex::new(0.0, 0.0)
        {
            return Err(TbError::Other(
                "Wilson-loop determinant is zero or nonfinite".into(),
            ));
        }
        Ok(-determinant.arg())
    }

    fn berry_flux(
        &self,
        occ: &[usize],
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Result<Array3<f64>> {
        self.validate_berry_plane(occ, k_start, dir_1, dir_2)?;
        if nk1 == 0 || nk2 == 0 {
            return Err(TbError::Other(
                "berry_flux requires nk1 and nk2 >= 1".into(),
            ));
        }
        check_array_size(&[nk1, nk2, 5, DIM], size_of::<f64>())?;
        check_array_size(&[nk1, nk2, occ.len()], size_of::<f64>())?;
        // Construct plaquette loops
        let mut k_loop = Array3::<f64>::zeros((nk1 * nk2, 5, self.dim_r()));
        for i in 0..nk1 {
            for j in 0..nk2 {
                let i0 = (i as f64) / (nk1 as f64);
                let j0 = (j as f64) / (nk2 as f64);
                let dx = 1.0 / (nk1 as f64);
                let dy = 1.0 / (nk2 as f64);
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 0, ..]);
                s.assign(&(k_start + (i0) * dir_1 + (j0) * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 1, ..]);
                s.assign(&(k_start + (i0 + dx) * dir_1 + j0 * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 2, ..]);
                s.assign(&(k_start + (i0 + dx) * dir_1 + (j0 + dy) * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 3, ..]);
                s.assign(&(k_start + (i0) * dir_1 + (j0 + dy) * dir_2));
                let mut s = k_loop.slice_mut(s![i * nk2 + j, 4, ..]);
                s.assign(&(k_start + (i0) * dir_1 + (j0) * dir_2));
            }
        }
        let berry_flux: Result<Vec<_>> = k_loop
            .outer_iter()
            .into_par_iter()
            .map(|x| self.berry_loop(&x, occ).map(|phases| phases.to_vec()))
            .collect();
        Ok(Array3::from_shape_vec(
            (nk1, nk2, occ.len()),
            berry_flux?.into_iter().flatten().collect(),
        )?)
    }

    fn berry_phase(&self, occ: &[usize], kvec: &Array3<f64>) -> Result<Array2<f64>> {
        let nk1 = kvec.shape()[0];
        let nocc = occ.len();
        self.validate()?;
        self.validate_berry_bands(occ)?;
        if kvec.shape()[2] != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: kvec.shape()[2],
            });
        }
        if kvec.shape()[1] < 2 {
            return Err(TbError::Other(
                "Berry loops require at least two k-points".into(),
            ));
        }
        check_array_size(&[nk1, nocc], size_of::<f64>())?;
        let phases: Result<Vec<_>> = kvec
            .outer_iter()
            .into_par_iter()
            .map(|k| {
                let mut phases = self.berry_loop(&k, occ)?.to_vec();
                phases.sort_by(f64::total_cmp);
                Ok(phases)
            })
            .collect();
        Ok(
            Array2::from_shape_vec((nk1, nocc), phases?.into_iter().flatten().collect())?
                .reversed_axes(),
        )
    }

    fn wannier_centre(
        &self,
        occ: &[usize],
        k_start: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
        nk1: usize,
        nk2: usize,
    ) -> Result<Array2<f64>> {
        self.validate_berry_plane(occ, k_start, dir_1, dir_2)?;
        if nk1 < 2 || nk2 < 2 {
            return Err(TbError::Other(
                "wannier_centre requires nk1 and nk2 >= 2".into(),
            ));
        }
        check_loop_closure(dir_2)?;
        check_array_size(&[nk1, nk2, DIM], size_of::<f64>())?;
        check_array_size(&[nk1, occ.len()], size_of::<f64>())?;
        let mut kvec = Array3::zeros((nk1, nk2, self.dim_r()));
        for i in 0..nk1 {
            for j in 0..nk2 {
                let mut s = kvec.slice_mut(s![i, j, ..]);
                let used_k = k_start
                    + dir_1 * (i as f64) / ((nk1 - 1) as f64)
                    + dir_2 * (j as f64) / ((nk2 - 1) as f64);
                s.assign(&used_k);
            }
        }
        self.berry_phase(occ, &kvec)
    }
}

fn check_array_size(shape: &[usize], element_size: usize) -> Result<()> {
    shape
        .iter()
        .try_fold(element_size, |bytes, &length| bytes.checked_mul(length))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| TbError::Other("Berry array size exceeds the addressable range".into()))?;
    Ok(())
}

fn check_loop_closure(displacement: &Array1<f64>) -> Result<()> {
    if displacement.iter().any(|x| !x.is_finite()) {
        return Err(TbError::Other(
            "Berry loop displacement must be finite".into(),
        ));
    }
    let remainder = displacement.mapv(|x| x - x.round());
    if remainder.iter().any(|x| x.abs() > 1e-9) {
        return Err(TbError::UnclosedWilsonLoop(remainder));
    }
    Ok(())
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    fn validate_berry_bands(&self, occ: &[usize]) -> Result<()> {
        if occ.is_empty() {
            return Err(TbError::Other(
                "Berry calculations require at least one selected band".into(),
            ));
        }
        for (i, &band) in occ.iter().enumerate() {
            if band >= self.nsta() {
                return Err(TbError::Other(format!(
                    "selected Berry band {band} is outside 0..{}",
                    self.nsta()
                )));
            }
            if occ[..i].contains(&band) {
                return Err(TbError::Other(format!(
                    "selected Berry band {band} appears more than once"
                )));
            }
        }
        Ok(())
    }

    fn validate_berry_plane(
        &self,
        occ: &[usize],
        origin: &Array1<f64>,
        dir_1: &Array1<f64>,
        dir_2: &Array1<f64>,
    ) -> Result<()> {
        self.validate()?;
        self.validate_berry_bands(occ)?;
        for vector in [origin, dir_1, dir_2] {
            if vector.len() != DIM {
                return Err(TbError::KVectorLengthMismatch {
                    expected: DIM,
                    actual: vector.len(),
                });
            }
            if vector.iter().any(|x| !x.is_finite()) {
                return Err(TbError::Other(
                    "Berry plane coordinates must be finite".into(),
                ));
            }
        }
        Ok(())
    }

    // Both Wilson variants use the same row-ket sewing and overlap matrices.
    fn wilson_overlaps<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
        occ: &[usize],
    ) -> Result<Array3<Complex<f64>>> {
        if kvec.ncols() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: kvec.ncols(),
            });
        }
        let n_k = kvec.nrows();
        if n_k < 2 {
            return Err(TbError::Other(
                "Berry loops require at least two k-points".into(),
            ));
        }
        if kvec.iter().any(|x| !x.is_finite()) {
            return Err(TbError::Other("Berry k-points must be finite".into()));
        }
        let diff = &kvec.row(n_k - 1) - &kvec.row(0);
        check_loop_closure(&diff)?;
        self.validate_berry_bands(occ)?;
        let n_occ = occ.len();
        check_array_size(&[n_k, self.nsta(), self.nsta()], size_of::<Complex<f64>>())?;
        check_array_size(&[n_k - 1, n_occ, n_occ], size_of::<Complex<f64>>())?;
        let (_, mut evec) = self.solve_all(kvec)?;
        // The spin-major basis repeats each orbital position for spin down.
        let phases = Array1::from_shape_fn(self.nsta(), |basis| {
            let angle = -2.0 * PI * diff.dot(&self.orb.row(basis % self.norb()));
            Complex::new(0.0, angle).exp()
        });
        if phases
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(TbError::Other(
                "Berry endpoint sewing phase is nonfinite".into(),
            ));
        }
        let end_evec = evec.index_axis(Axis(0), 0).dot(&Array2::from_diag(&phases));
        evec.index_axis_mut(Axis(0), n_k - 1).assign(&end_evec);
        let evec = evec.select(Axis(1), occ);
        let evec_conj = evec.map(|x| x.conj());
        let mut overlaps = Array3::zeros((n_k - 1, n_occ, n_occ));
        for (ik, mut overlap) in overlaps.outer_iter_mut().enumerate() {
            for (i, bra) in evec_conj.index_axis(Axis(0), ik).outer_iter().enumerate() {
                for (j, ket) in evec.index_axis(Axis(0), ik + 1).outer_iter().enumerate() {
                    overlap[[i, j]] = bra.dot(&ket);
                }
            }
        }
        if overlaps
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(TbError::Other("Wilson overlap is nonfinite".into()));
        }
        Ok(overlaps)
    }
}
