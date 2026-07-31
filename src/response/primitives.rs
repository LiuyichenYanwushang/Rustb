//! Per‑k‑point velocity kernel computation.
//!
//! Extracts band‑basis velocity matrix elements `v^a_nm` and builds the
//! gauge‑invariant product `K^{ab}_nm = v^a_nm · v^b_mn`.

use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;

use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::SpinDirection;
use crate::math::anti_comm;
use crate::velocity::Velocity;

use super::helpers::build_spin_matrix;
use super::types::VertexKernel;

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Compute band‑basis velocity primitives at one k‑point.
    ///
    /// Returns a `VertexKernel` containing band energies, eigenvectors,
    /// the gauge‑invariant `K^{ab}_nm = v^a_nm · v^b_mn`, and optionally
    /// the diagonal velocity `v^c_n`.
    ///
    /// # Arguments
    /// * `k_vec` — single k‑point in fractional reciprocal coords.
    /// * `dir_a`, `dir_b` — direction pair for `K^{ab}`.
    /// * `dir_c` — optional diagonal velocity direction (for dipoles).
    /// * `gauge` — `Atom` or `Lattice`.
    /// * `spin` — optional spin direction for spin‑current evaluation.
    pub(crate) fn compute_velocity_kernel(
        &self,
        k_vec: &Array1<f64>,
        dir_a: &Array1<f64>,
        dir_b: &Array1<f64>,
        dir_c: Option<&Array1<f64>>,
        gauge: Gauge,
        spin: Option<SpinDirection>,
    ) -> VertexKernel {
        let nsta = self.nsta();

        // Build direction matrix: [dir_a, dir_b, (opt) dir_c]
        let n_dir = if dir_c.is_some() { 3 } else { 2 };
        let mut directions = Array2::<f64>::zeros((n_dir, DIM));
        directions.row_mut(0).assign(dir_a);
        directions.row_mut(1).assign(dir_b);
        if let Some(dc) = dir_c {
            directions.row_mut(2).assign(dc);
        }

        let (v_proj, hamk) = self.gen_v_projected(k_vec, gauge, &directions);
        let (band, evec) = hamk.eigh(UPLO::Lower).unwrap();
        // Convention: U^T · v · U^*
        let ut = evec.t();
        let uc = evec.map(|x| x.conj());

        let to_band = |d: usize, spin_dress: bool| -> Array2<Complex<f64>> {
            let v_raw = v_proj.slice(s![d, .., ..]).to_owned();
            if spin_dress && SPIN && spin.is_some() {
                let x = build_spin_matrix(self.norb(), spin);
                let s = anti_comm(&x, &v_raw) * 0.5;
                ut.dot(&s.dot(&uc))
            } else {
                ut.dot(&v_raw.dot(&uc))
            }
        };

        // dir_a gets spin‑dressed for Berry curvature; dir_b does not
        let va = to_band(0, true);
        let vb = to_band(1, false);

        let mut k_ab = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for n in 0..nsta {
            for m in 0..nsta {
                k_ab[[n, m]] = va[[n, m]] * vb[[m, n]];
            }
        }

        let (vdiag, k_bc, k_ac, vdiag_a, vdiag_b) = if let Some(_dc) = dir_c {
            let vc = to_band(2, false);
            let mut bc = Array2::<Complex<f64>>::zeros((nsta, nsta));
            let mut ac = Array2::<Complex<f64>>::zeros((nsta, nsta));
            for n in 0..nsta {
                for m in 0..nsta {
                    bc[[n, m]] = vb[[n, m]] * vc[[m, n]];
                    ac[[n, m]] = va[[n, m]] * vc[[m, n]];
                }
            }
            (
                Some(vc.diag().map(|x| x.re).to_owned()),
                Some(bc),
                Some(ac),
                Some(va.diag().map(|x| x.re).to_owned()),
                Some(vb.diag().map(|x| x.re).to_owned()),
            )
        } else {
            (None, None, None, None, None)
        };

        VertexKernel {
            band,
            evec,
            k_ab,
            k_bc,
            k_ac,
            vdiag,
            vdiag_a,
            vdiag_b,
        }
    }
}
