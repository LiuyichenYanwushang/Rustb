//! Per‑k‑point velocity kernel computation.
//!
//! Extracts band‑basis velocity matrix elements `v^a_nm` and builds the
//! gauge‑invariant product `K^{ab}_nm = v^a_nm · v^b_mn`.

use crate::ndarray_lapack::eigh_full;
use ndarray::prelude::*;
use ndarray_linalg::*;
use num_complex::Complex;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::math::anti_comm;

use super::types::{NonlinearKernel, VertexKernel};

// ponytail: interpolators store raw velocity products; reject overflow
// and off-diagonal underflow instead of inventing a zero response.
// Raw subnormals also cannot preserve relative accuracy when weighted.
// A scaled primitive representation is needed to extend this domain.
pub(super) fn checked_velocity_product(
    a: ArrayView2<'_, Complex<f64>>,
    b: ArrayView2<'_, Complex<f64>>,
) -> Result<Array2<Complex<f64>>> {
    let nsta = a.nrows();
    let mut product = Array2::<Complex<f64>>::zeros((nsta, nsta));
    for n in 0..nsta {
        for m in 0..nsta {
            let left = a[[n, m]];
            let right = b[[m, n]];
            let value = left * right;
            if !value.re.is_finite()
                || !value.im.is_finite()
                || (n != m
                    && [
                        (left.re, right.re),
                        (left.im, right.im),
                        (left.re, right.im),
                        (left.im, right.re),
                    ]
                    .into_iter()
                    .any(|(a, b)| a != 0.0 && b != 0.0 && (a * b == 0.0 || (a * b).is_subnormal())))
                || (n != m && (value.re.is_subnormal() || value.im.is_subnormal()))
            {
                return Err(TbError::InvalidResponseParameter {
                    parameter: "velocity_kernel",
                    message: format!(
                        "velocity product ({n}, {m}) is outside the f64 numerical range"
                    ),
                });
            }
            product[[n, m]] = value;
        }
    }
    Ok(product)
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Compute band‑basis velocity primitives at one k‑point.
    ///
    /// Returns row-ket eigenvectors separately from the kernel containing energies,
    /// the gauge‑invariant `K^{ab}_nm = v^a_nm · v^b_mn`, and optionally
    /// the diagonal velocity `v^c_n`.
    ///
    /// # Arguments
    /// * `k_vec` — single k‑point in fractional reciprocal coords.
    /// * `dir_a`, `dir_b` — direction pair for `K^{ab}`.
    /// * `dir_c` — optional diagonal velocity direction (for dipoles).
    /// * `gauge` — `Atom` or `Lattice`.
    /// * `spin_matrix` — optional model-basis `S / ℏ`, constructed once by
    ///   the spinful caller and shared across its k-points.
    pub(crate) fn compute_velocity_kernel(
        &self,
        k_vec: &Array1<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: Option<&Array1<f64>>,
        gauge: Gauge,
        spin_matrix: Option<&Array2<Complex<f64>>>,
    ) -> Result<(VertexKernel, Array2<Complex<f64>>)> {
        if k_vec.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k_vec.len(),
            });
        }
        for (name, dir) in [("dir_a", dir_a), ("dir_b", dir_b)] {
            if dir.len() != DIM {
                return Err(TbError::DimensionMismatch {
                    context: name.into(),
                    expected: DIM,
                    found: dir.len(),
                });
            }
        }
        if let Some(dir_c) = dir_c {
            if dir_c.len() != DIM {
                return Err(TbError::DimensionMismatch {
                    context: "dir_c".into(),
                    expected: DIM,
                    found: dir_c.len(),
                });
            }
        }

        // Build direction matrix: [dir_a, dir_b, (opt) dir_c]
        let n_dir = if dir_c.is_some() { 3 } else { 2 };
        let mut directions = Array2::<f64>::zeros((n_dir, DIM));
        directions.row_mut(0).assign(dir_a);
        directions.row_mut(1).assign(dir_b);
        if let Some(dc) = dir_c {
            directions.row_mut(2).assign(dc);
        }

        let (v_proj, hamk) = self.gen_v_projected(k_vec, gauge, &directions);
        #[cfg(test)]
        super::config::counters::count_eigen_decomposition();
        let (band, evec) = eigh_full(&hamk, UPLO::Lower)?;
        // Row-ket convention: C* · v · C^T
        let ut = evec.mapv(|x| x.conj());
        let uc = evec.t();

        let to_band = |d: usize, spin_dress: bool| -> Array2<Complex<f64>> {
            let v_raw = v_proj.slice(s![d, .., ..]).to_owned();
            if spin_dress && let Some(spin_matrix) = spin_matrix {
                let current = anti_comm(spin_matrix, &v_raw) * 0.5;
                ut.dot(&current.dot(&uc))
            } else {
                ut.dot(&v_raw.dot(&uc))
            }
        };

        // dir_a gets spin‑dressed for Berry curvature; dir_b does not
        let va = to_band(0, true);
        let vb = to_band(1, false);

        let k_ab = checked_velocity_product(va.view(), vb.view())?;

        let nonlinear = if dir_c.is_some() {
            let vc = to_band(2, false);
            Some(NonlinearKernel {
                k_bc: checked_velocity_product(vb.view(), vc.view())?,
                k_ac: checked_velocity_product(va.view(), vc.view())?,
                vdiag: vc.diag().mapv(|z| z.re),
                vdiag_a: va.diag().mapv(|z| z.re),
                vdiag_b: vb.diag().mapv(|z| z.re),
            })
        } else {
            None
        };
        Ok((
            VertexKernel {
                band,
                k_ab,
                nonlinear,
            },
            evec,
        ))
    }
}
