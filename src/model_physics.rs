//! Physics calculation methods for tight-binding models
use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::kpoints::gen_kmesh;
use crate::solve_ham::Solve;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;

pub(crate) fn apply_atom_gauge(
    mut matrix: ArrayViewMut2<'_, Complex<f64>>,
    phases: &Array1<Complex<f64>>,
) {
    for (i, mut row) in matrix.outer_iter_mut().enumerate() {
        let left = phases[i].conj();
        Zip::from(&mut row)
            .and(phases)
            .for_each(|value, &right| *value *= left * right);
    }
}

/// Sum R blocks for several coefficient rows without copying the block array.
/// Used for H, its Cartesian derivatives, and the position matrix Fourier sum.
pub(crate) fn fourier_sum<S, D>(
    coefficients: &ArrayView2<'_, Complex<f64>>,
    blocks: &ArrayBase<S, D>,
) -> Array2<Complex<f64>>
where
    S: Data<Elem = Complex<f64>>,
    D: Dimension + RemoveAxis,
{
    let nr = blocks.len_of(Axis(0));
    assert_eq!(
        coefficients.ncols(),
        nr,
        "Fourier coefficient/support mismatch"
    );
    let width = blocks.shape()[1..]
        .iter()
        .try_fold(1usize, |size, &axis| size.checked_mul(axis))
        .expect("Fourier block size overflow");
    let mut result = Array2::zeros((coefficients.nrows(), width));
    if result.is_empty() || nr == 0 {
        return result;
    }
    if coefficients.nrows() > 1
        && let Some(data) = blocks.as_slice()
        && let (Ok(m), Ok(n), Ok(k)) = (
            i32::try_from(width),
            i32::try_from(coefficients.nrows()),
            i32::try_from(nr),
        )
    {
        let coefficients = coefficients.as_standard_layout();
        // SAFETY: C-row-major result = coefficients * blocks is the same
        // memory as column-major result^T = blocks^T * coefficients^T.
        // The slices have lengths m*k, k*n, m*n; all dimensions are positive
        // and fit BLAS integers. Output owns disjoint writable storage.
        unsafe {
            blas::zgemm(
                b'N',
                b'N',
                m,
                n,
                k,
                Complex::new(1.0, 0.0),
                data,
                m,
                coefficients.as_slice().unwrap(),
                k,
                Complex::new(0.0, 0.0),
                result.as_slice_mut().unwrap(),
                m,
            );
        }
    } else {
        // Preserve the cheap single-row AXPY path and support strided blocks
        // without materializing a potentially very large contiguous copy.
        for (weights, mut out) in coefficients.outer_iter().zip(result.outer_iter_mut()) {
            for (&weight, block) in weights.iter().zip(blocks.axis_iter(Axis(0))) {
                if let Some(data) = block.as_slice()
                    && i32::try_from(width).is_ok()
                {
                    crate::ndarray_lapack::zaxpy(weight, data, out.as_slice_mut().unwrap());
                } else {
                    for (value, &element) in out.iter_mut().zip(block.iter()) {
                        *value += weight * element;
                    }
                }
            }
        }
    }
    result
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    #[allow(non_snake_case)]
    #[inline(always)]
    ///Performs Fourier transform, converting real-space Hamiltonian to reciprocal-space Hamiltonian.
    ///
    ///There are two gauge choices: lattice gauge and atomic gauge, corresponding to `Gauge::Lattice` and `Gauge::Atom`.
    ///
    ///For the atomic gauge, the transformation between real-space wavefunction $\ket{n\bm R}$ and reciprocal-space wavefunction $\ket{u_{\bm k,n}}$ is:
    ///
    ///$$\ket{u_{n\bm k}(\bm r)}=\sum_{\bm R} e^{i\bm k\cdot(\bm R+\bm\tau_n)}\ket{n\bm R}$$
    ///
    ///satisfying $\ket{u_{i\bm k}(\bm r+\bm R)}=\ket{u_{i\bm k}(\bm r)}$.
    ///
    ///For the Hamiltonian, we have:
    ///$$
    ///H_{mn,\bm k}=\bra{u_{m\bm k}}\hat H\ket{u_{n\bm k}}=\sum_{\bm R^\prime}\sum_{\bm R} \bra{m\bm R^\prime}\hat H\ket{n\bm R}e^{-i(\bm R'-\bm R+\bm\tau_m-\bm \tau_n)\cdot\bm k}.
    ///$$
    ///Due to translational symmetry, only $\bm R'-\bm R$ matters, thus:
    ///$$
    ///H_{mn,\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R-\bm\tau_m+\bm \tau_n)\cdot\bm k}
    ///$$
    ///
    ///For the lattice gauge, we have $$\ket{\phi_{n\bm k}}=\sum_{\bm R} e^{i\bm k\cdot\bm R}\ket{n\bm R},$$ so:
    ///$$
    ///H_{mn,\bm k}=\sum_{\bm R} \bra{m\bm 0}\hat H\ket{n\bm R}e^{i(\bm R)\cdot\bm k}
    ///$$
    ///
    ///Here $\ket{\psi_{n\bm k}}$ is periodic in reciprocal space: $\ket{\phi_{n\bm k}(\bm r)}=\ket{\phi_{n\bm k+\bm G}(\bm r)}$.
    pub fn gen_ham<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> Array2<Complex<f64>> {
        self.gen_ham_batch(&kvec.view().insert_axis(Axis(0)), gauge)
            .index_axis_move(Axis(0), 0)
    }

    /// Construct H(k) for a batch of fractional k-points, shape `(nk, DIM)`.
    ///
    /// Returns `(nk, nsta, nsta)` in input order. A single GEMM reuses the
    /// hopping array across the supplied points when its storage is contiguous.
    /// This method does not create Rayon jobs. Callers choose the batch size
    /// and BLAS thread policy; the returned matrices occupy `nk * nsta²`
    /// complex numbers. Both lattice and atom gauges match [`Self::gen_ham`].
    pub fn gen_ham_batch<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        gauge: Gauge,
    ) -> Array3<Complex<f64>> {
        let phases = self.bloch_phases(points);
        let nsta = self.nsta();
        let mut hams = fourier_sum(&phases.view(), &self.ham)
            .into_shape_with_order((points.nrows(), nsta, nsta))
            .unwrap();
        if matches!(gauge, Gauge::Atom) {
            for (k, ham) in points.outer_iter().zip(hams.outer_iter_mut()) {
                apply_atom_gauge(ham, &self.orbital_phases(&k));
            }
        }
        hams
    }

    pub(crate) fn bloch_phases<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
    ) -> Array2<Complex<f64>> {
        assert_eq!(
            points.ncols(),
            DIM,
            "k-point dimension must match the model"
        );
        assert_eq!(
            self.hamR.ncols(),
            DIM,
            "hopping translation dimension must match the model"
        );
        assert_eq!(
            self.ham.dim(),
            (self.hamR.nrows(), self.nsta(), self.nsta()),
            "hopping shape must match the model"
        );
        Array2::from_shape_fn((points.nrows(), self.hamR.nrows()), |(ik, ir)| {
            let dot = (0..DIM)
                .map(|axis| self.hamR[[ir, axis]] as f64 * points[[ik, axis]])
                .sum::<f64>();
            Complex::new(0.0, 2.0 * PI * dot).exp()
        })
    }

    pub(crate) fn orbital_phases<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
    ) -> Array1<Complex<f64>> {
        let mut phases = Array1::zeros(self.nsta());
        for i in 0..self.norb() {
            let dot = (0..DIM)
                .map(|axis| self.orb[[i, axis]] * k[axis])
                .sum::<f64>();
            phases[i] = Complex::new(0.0, 2.0 * PI * dot).exp();
            if SPIN {
                phases[i + self.norb()] = phases[i];
            }
        }
        phases
    }

    /// Computes the density of states $\rho(E)$ using Gaussian smearing.
    ///
    /// The DOS is defined as:
    ///
    /// $$\rho(E) = \frac{1}{N_k} \sum_{n,\mathbf{k}} \delta(E - E_{n\mathbf{k}})$$
    ///
    /// The delta function is approximated by a Gaussian of width $\sigma$:
    ///
    /// ```math
    /// \delta(x) \approx \frac{1}{\sqrt{2\pi}\,\sigma}\, e^{-x^2 / (2\sigma^2)}
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Generate a uniform k-mesh from `k_mesh`
    /// 2. Diagonalize $H(\mathbf{k})$ at every k-point in parallel
    /// 3. Convolve eigenvalues with the Gaussian kernel and sum
    ///
    /// The smoothness depends on both the k-point density and $\sigma$.
    ///
    /// # Arguments
    ///
    /// * `k_mesh` — k-points along each direction, e.g. `[51, 51]`
    /// * `E_min`, `E_max` — Energy range
    /// * `E_n` — Number of energy bins
    /// * `sigma` — Gaussian smearing width (same units as energy)
    ///
    /// # Returns
    ///
    /// `(energies, dos)` — energy grid and corresponding DOS.
    #[allow(non_snake_case)]
    pub fn dos(
        &self,
        k_mesh: &Array1<usize>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        sigma: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        self.validate()?;
        if !E_min.is_finite() || !E_max.is_finite() || E_min >= E_max {
            return Err(TbError::InvalidEnergyRange {
                min: E_min,
                max: E_max,
            });
        }
        if E_n == 0 {
            return Err(TbError::InvalidDosParameter {
                parameter: "E_n",
                message: "number of energy bins must be at least 1".to_string(),
            });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(TbError::InvalidDosParameter {
                parameter: "sigma",
                message: "Gaussian smearing width must be finite and positive".to_string(),
            });
        }
        let kvec: Array2<f64> = gen_kmesh(&k_mesh)?;
        let nk = kvec.len_of(Axis(0));
        let eigenvalues = self.solve_band_all_parallel(&kvec);
        let E = Array1::linspace(E_min, E_max, E_n);
        let _dim: usize = k_mesh.len();
        let centre = eigenvalues.into_raw_vec_and_offset().0.into_par_iter();
        let sigma0 = 1.0 / sigma;
        let pi0 = 1.0 / (2.0 * PI).sqrt();
        let _dos = Array1::<f64>::zeros(E_n);
        let dos = centre
            .fold(
                || Array1::<f64>::zeros(E_n),
                |acc, x| {
                    let A: Array1<f64> = (&E - x) * sigma0;
                    let f: Array1<f64> = (-&A * &A / 2.0).mapv(|x: f64| x.exp()) * sigma0 * pi0;
                    acc + &f
                },
            )
            .reduce(|| Array1::<f64>::zeros(E_n), |acc, x| acc + x);
        let dos = dos / (nk as f64);
        Ok((E, dos))
    }
}
