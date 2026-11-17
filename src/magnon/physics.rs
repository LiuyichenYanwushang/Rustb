//! Spin-wave dispersion from the real-space exchange tensors.
//!
//! The construction follows TB2J's `Magnon.Hq`:
//!
//! 1. build local spin frames from [`Magnon::axis`] and the reference
//!    quantization axis (TB2J's `uz`);
//! 2. Fourier transform the ordered-pair exchange tensors,
//!    `J(q) = sum_R J(R) exp(-2 pi i q . R)`, after rotating each block by the
//!    reference propagation vector `q_ref` (TB2J's `Q`);
//! 3. assemble the `2n x 2n` Bogoliubov matrix
//!    `H = [[A - C, B], [B^*, A^*(-q) - C]]` with
//!    `A_ij = -sqrt(S_i S_j) U_i^T J^op_ij(q) U_j^*`,
//!    `B_ij = -sqrt(S_i S_j) U_i^T J^op_ij(q) U_j`,
//!    `C_i = sum_j 2 S_j V_i^T J^op_ij(0) V_j`,
//!    where `J^op = J/(S_i S_j)` converts the stored unit-vector landscape
//!    coefficient into the operator exchange (the `1/(S_i S_j)` factor is
//!    TB2J's `Magnon.Jq`, the `sqrt(S_i S_j)` and `S_j` factors are its `Hq`);
//! 4. solve the paraunitary problem with Colpa's method: factorize
//!    `H = K K^dagger`, diagonalize the Hermitian `K^dagger g K`
//!    (`g = diag(1, -1)`) and keep the `n` largest eigenvalues.
//!
//! Unlike TB2J, the factorization here uses the Hermitian eigendecomposition
//! `H = U L U^dagger` with `K = U L^(1/2)` rather than a Cholesky factor. The
//! two are equivalent for positive-definite `H`, but the eigendecomposition
//! keeps gapless (Goldstone) modes exactly zero instead of perturbing them by
//! `sqrt(shift * omega)`: a Cholesky of a singular `H` needs a diagonal shift,
//! which mixes the zero mode with the gapped one and produces a spurious
//! `~1e-4 eV` gap at `q` points that should be gapless.
//!
//! A stable magnetic state requires `H` to be positive semidefinite. A
//! genuinely negative eigenvalue means the reference structure is not a local
//! energy minimum; that is reported as an error instead of returning an
//! imaginary frequency.
//!
//! # Scope
//!
//! Collinear references (ferro-, ferri- and two-sublattice antiferromagnets,
//! with or without single-ion anisotropy) are verified against analytic
//! dispersions. A frustrated triangular reference described by a single-Q
//! propagation vector is verified to be gapless at the magnetic Gamma point and
//! stable everywhere else.
//!
//! An indefinite `H` means the reference structure is a saddle rather than a
//! minimum, and the solver reports it instead of returning a complex frequency.
//! A 3-site 120-degree ring in a 1D cell is such a case: it is a stationary
//! point of the bipartite chain whose true minimum is Neel order, and an
//! independent frozen-magnon Hessian of the classical energy has the same
//! negative eigenvalue at the same wavevector.

use super::{Magnon, MagnonSolve, SPIN_COMPONENTS};
use crate::error::{Result, TbError};
use crate::kpoints::gen_kmesh;
use crate::ndarray_lapack::eigh_full;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_linalg::UPLO;
use num_complex::Complex;
use rayon::prelude::*;

/// Relative tolerance (scaled by the largest `|eigenvalue|`) below which a
/// negative BdG eigenvalue is treated as round-off rather than an unstable
/// reference structure. This keeps the check unit-independent.
const BDG_POSITIVITY_RTOL: f64 = 1e-10;

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// Rotation matrix `R = I + sin(phi) K + (1 - cos(phi)) K^2` about `axis`.
fn rotation_about(axis: &[f64; 3], phi: f64) -> Option<[[f64; 3]; 3]> {
    let length = norm(axis);
    if !length.is_finite() || length <= 1e-15 || !phi.is_finite() {
        return None;
    }
    let n = [axis[0] / length, axis[1] / length, axis[2] / length];
    let (sin, cos) = phi.sin_cos();
    let k = [[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]];
    let mut out = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let k2 = (0..3).map(|m| k[i][m] * k[m][j]).sum::<f64>();
            out[i][j] = if i == j { 1.0 } else { 0.0 } + sin * k[i][j] + (1.0 - cos) * k2;
        }
    }
    Some(out)
}

/// Local spin frames (TB2J's `get_rotation_arrays`).
///
/// Site `i` gets a right-handed triad adapted to its reference moment
/// `axis[i]`: the complex transverse vector `U_i = e_x + i e_y` and the
/// quantization vector `V_i = e_z = axis[i]`. `quantization_axis` is TB2J's
/// `uz`. For a moment parallel to `uz` the general construction degenerates;
/// this implementation falls back to an orthonormal Gram-Schmidt triad, which
/// agrees with TB2J's literal fallback for moments along a coordinate axis and
/// stays orthonormal for any `uz`. Spectra do not depend on that gauge, the
/// Bogoliubov vectors differ by a phase.
pub(crate) fn local_frames<const DIM: usize>(
    magnon: &Magnon<DIM>,
) -> (Array2<Complex<f64>>, Array2<f64>) {
    let n = magnon.nspin();
    let mut u = Array2::<Complex<f64>>::zeros((n, SPIN_COMPONENTS));
    let mut v = Array2::<f64>::zeros((n, SPIN_COMPONENTS));
    let z = [
        magnon.reference.quantization_axis[0],
        magnon.reference.quantization_axis[1],
        magnon.reference.quantization_axis[2],
    ];
    for i in 0..n {
        let m = [
            magnon.axis[[i, 0]],
            magnon.axis[[i, 1]],
            magnon.axis[[i, 2]],
        ];
        let mut axis = cross(&z, &m);
        let length = norm(&axis);
        // `rotation[a][k]` is the `a`-th Cartesian component of local axis `k`.
        let rotation: [[f64; 3]; 3] = if length > 1e-12 {
            for component in axis.iter_mut() {
                *component /= length;
            }
            let a = [z, cross(&axis, &z), axis];
            let b = [m, cross(&axis, &m), axis];
            // R = A^T B with the rows of A and B holding the triads.
            let mut r = [[0.0f64; 3]; 3];
            for row in 0..3 {
                for column in 0..3 {
                    r[row][column] = (0..3).map(|k| a[k][row] * b[k][column]).sum();
                }
            }
            r
        } else {
            // Moment parallel to the quantization axis: the deterministic triad
            // TB2J falls back to, with the local axes stored as rows.
            let mut seed = [1.0, 0.0, 0.0];
            if norm(&cross(&seed, &m)) < 1e-12 {
                seed = [0.0, 1.0, 0.0];
            }
            let projection = dot(&seed, &m);
            let mut e1 = [
                seed[0] - projection * m[0],
                seed[1] - projection * m[1],
                seed[2] - projection * m[2],
            ];
            let e1_length = norm(&e1);
            for component in e1.iter_mut() {
                *component /= e1_length;
            }
            let e2 = cross(&m, &e1);
            [e1, e2, m]
        };
        // The triad rows are (e_x, e_y, e_z): U_i = e_x + i e_y and V_i = e_z.
        for a in 0..SPIN_COMPONENTS {
            u[[i, a]] = Complex::new(rotation[0][a], rotation[1][a]);
            v[[i, a]] = rotation[2][a];
        }
    }
    (u, v)
}

impl<const DIM: usize> Magnon<DIM> {
    fn check_q<S: Data<Elem = f64>>(&self, q: &ArrayBase<S, Ix1>) -> Result<[f64; 3]> {
        if q.len() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "magnon q-point".into(),
                expected: DIM,
                found: q.len(),
            });
        }
        let mut point = [0.0f64; 3];
        for (slot, value) in point.iter_mut().zip(q.iter()) {
            if !value.is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "finite_q_point",
                    message: "q-points must be finite".into(),
                });
            }
            *slot = *value;
        }
        Ok(point)
    }

    fn has_ordering(&self) -> bool {
        self.reference.q.iter().any(|&value| value != 0.0)
    }

    /// Fourier transform of the exchange tensors,
    /// `J(q) = sum_R J(R) exp(-2 pi i q . R)`, shape `(3n, 3n)`.
    ///
    /// Each `3 x 3` block is rotated by the reference propagation vector first
    /// (TB2J's single-Q convention), and the stored unit-vector coefficients are
    /// returned unchanged: this is the landscape `J(q)`, not `J(q)/(S_i S_j)`.
    pub fn gen_jq(&self, q: &Array1<f64>) -> Result<Array2<Complex<f64>>> {
        self.validate()?;
        self.jq_of(&self.check_q(q)?)
    }

    fn jq_of(&self, q: &[f64; 3]) -> Result<Array2<Complex<f64>>> {
        let n = self.nspin();
        let nsta = self.nsta();
        let ordering = if self.has_ordering() {
            let mut q_ref = [0.0f64; 3];
            for (slot, value) in q_ref.iter_mut().zip(self.reference.q.iter()) {
                *slot = *value;
            }
            Some(q_ref)
        } else {
            None
        };
        let rotation_axis = [
            self.reference.rotation_axis[0],
            self.reference.rotation_axis[1],
            self.reference.rotation_axis[2],
        ];
        let mut out = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for i_r in 0..self.nR() {
            let mut r = [0.0f64; 3];
            for (axis, slot) in r.iter_mut().enumerate().take(DIM) {
                *slot = self.hamR[[i_r, axis]] as f64;
            }
            let phase = -std::f64::consts::TAU * dot(&r, q);
            let factor = Complex::from_polar(1.0, phase);
            let rotation = match ordering {
                Some(q_ref) => {
                    rotation_about(&rotation_axis, std::f64::consts::TAU * dot(&r, &q_ref))
                }
                None => None,
            };
            for i in 0..n {
                for j in 0..n {
                    for a in 0..SPIN_COMPONENTS {
                        for b in 0..SPIN_COMPONENTS {
                            let value = match &rotation {
                                None => {
                                    self.ham
                                        [[i_r, SPIN_COMPONENTS * i + a, SPIN_COMPONENTS * j + b]]
                                }
                                Some(matrix) => (0..SPIN_COMPONENTS)
                                    .map(|c| {
                                        self.ham[[
                                            i_r,
                                            SPIN_COMPONENTS * i + a,
                                            SPIN_COMPONENTS * j + c,
                                        ]] * matrix[c][b]
                                    })
                                    .sum(),
                            };
                            out[[SPIN_COMPONENTS * i + a, SPIN_COMPONENTS * j + b]] +=
                                factor * Complex::new(value, 0.0);
                        }
                    }
                }
            }
        }
        // The single-Q rotation multiplies every stored block on the same side,
        // so a tensor that does not commute with the rotation makes the raw
        // Fourier sum non-Hermitian. Only its Hermitian part enters the
        // classical energy (a real quadratic form is blind to the antisymmetric
        // part), so symmetrize before the spin-wave construction.
        let adjoint = out.t().mapv(|value| value.conj());
        Ok((&out + &adjoint) * Complex::new(0.5, 0.0))
    }

    /// Bogoliubov-de Gennes matrix of the spin-wave problem at `q`, shape
    /// `(2n, 2n)`.
    ///
    /// The block structure is `[[A - C, B], [B^*, A^*(-q) - C]]` with the
    /// TB2J `A`, `B`, `C` matrices built from the local frames and the
    /// `J^op = J/(S_i S_j)` operator exchange. The matrix is Hermitian; a
    /// positive semidefinite result is the stability condition checked by
    /// [`MagnonSolve`](super::MagnonSolve).
    pub fn gen_bdg(&self, q: &Array1<f64>) -> Result<Array2<Complex<f64>>> {
        self.validate()?;
        let point = self.check_q(q)?;
        self.bdg_of(&point)
    }

    fn bdg_of(&self, q: &[f64; 3]) -> Result<Array2<Complex<f64>>> {
        let n = self.nspin();
        let jq = self.jq_of(q)?;
        let j0 = self.jq_of(&[0.0; 3])?;
        let jmq = jq.t().to_owned();
        let (u, v) = local_frames(self);
        let spin = &self.spin;

        let mut a1 = Array2::<Complex<f64>>::zeros((n, n));
        let mut a2 = Array2::<Complex<f64>>::zeros((n, n));
        let mut b = Array2::<Complex<f64>>::zeros((n, n));
        let mut c = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let (mut sum_a1, mut sum_a2, mut sum_b) = (
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                    Complex::new(0.0, 0.0),
                );
                for a in 0..SPIN_COMPONENTS {
                    for bb in 0..SPIN_COMPONENTS {
                        let m = jmq[[SPIN_COMPONENTS * i + a, SPIN_COMPONENTS * j + bb]];
                        sum_a1 += u[[i, a]] * m * u[[j, bb]].conj();
                        sum_a2 += u[[i, a]].conj() * m * u[[j, bb]];
                        sum_b += u[[i, a]] * m * u[[j, bb]];
                    }
                }
                // TB2J divides J by S_i S_j and then multiplies A and B by
                // sqrt(S_i S_j); the net factor is 1/sqrt(S_i S_j) with a minus
                // sign from the energy convention.
                let factor = -1.0 / (spin[i] * spin[j]).sqrt();
                a1[[i, j]] = sum_a1 * factor;
                a2[[i, j]] = sum_a2 * factor;
                b[[i, j]] = sum_b * factor;
            }
        }
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                let mut t = 0.0;
                for a in 0..SPIN_COMPONENTS {
                    for bb in 0..SPIN_COMPONENTS {
                        t += v[[i, a]]
                            * j0[[SPIN_COMPONENTS * i + a, SPIN_COMPONENTS * j + bb]].re
                            * v[[j, bb]];
                    }
                }
                sum += 2.0 * t;
            }
            // The 2 S_j/(S_i S_j) factor of TB2J reduces to 2/S_i.
            c[i] = -sum / spin[i];
        }

        let size = 2 * n;
        let mut h = Array2::<Complex<f64>>::zeros((size, size));
        for i in 0..n {
            for j in 0..n {
                h[[i, j]] = a1[[i, j]];
                h[[size - n + i, size - n + j]] = a2[[i, j]];
                h[[i, size - n + j]] = b[[i, j]];
                h[[size - n + i, j]] = b[[j, i]].conj();
            }
            h[[i, i]] -= Complex::new(c[i], 0.0);
            h[[size - n + i, size - n + i]] -= Complex::new(c[i], 0.0);
        }
        Ok(h)
    }

    /// Density of states `g(omega) = (1/N_q) sum_{q,n} N(omega; omega_n(q), sigma)`
    /// on a uniform frequency grid.
    ///
    /// Returns `(omega, dos)` with `omega = linspace(w_min, w_max, w_n)`; the
    /// Gaussians are normalized, so integrating the result over a window that
    /// contains all bands gives `nspin()`.
    pub fn magnon_dos(
        &self,
        k_mesh: &Array1<usize>,
        w_min: f64,
        w_max: f64,
        w_n: usize,
        sigma: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        self.validate()?;
        if !w_min.is_finite() || !w_max.is_finite() || w_min >= w_max {
            return Err(TbError::InvalidEnergyRange {
                min: w_min,
                max: w_max,
            });
        }
        if w_n == 0 {
            return Err(TbError::InvalidDosParameter {
                parameter: "w_n",
                message: "number of frequency bins must be at least 1".into(),
            });
        }
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(TbError::InvalidDosParameter {
                parameter: "sigma",
                message: "Gaussian smearing width must be finite and positive".into(),
            });
        }
        let points: Array2<f64> = gen_kmesh::<f64>(k_mesh)?;
        let bands = self.magnon_band_all(&points)?;
        let nq = bands.nrows() as f64;
        let omega = Array1::linspace(w_min, w_max, w_n);
        let mut dos = Array1::<f64>::zeros(w_n);
        let norm = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        for value in bands.iter() {
            for (index, centre) in omega.iter().enumerate() {
                let delta = (centre - value) / sigma;
                dos[index] += (-0.5 * delta * delta).exp() * norm;
            }
        }
        dos.mapv_inplace(|value| value / nq);
        Ok((omega, dos))
    }
}

impl<const DIM: usize> MagnonSolve for Magnon<DIM> {
    fn magnon_band_onek<S: Data<Elem = f64>>(&self, q: &ArrayBase<S, Ix1>) -> Result<Array1<f64>> {
        self.validate()?;
        let point = self.check_q(q)?;
        let h = self.bdg_of(&point)?;
        paraunitary_frequencies(&h, self.nspin())
    }

    fn magnon_band_all<S: Data<Elem = f64>>(&self, qs: &ArrayBase<S, Ix2>) -> Result<Array2<f64>> {
        self.validate()?;
        if qs.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "magnon q-point block".into(),
                expected: DIM,
                found: qs.ncols(),
            });
        }
        let n = self.nspin();
        let mut out = Array2::<f64>::zeros((qs.nrows(), n));
        for (row, q) in qs.axis_iter(Axis(0)).enumerate() {
            let point = self.check_q(&q)?;
            let h = self.bdg_of(&point)?;
            out.row_mut(row).assign(&paraunitary_frequencies(&h, n)?);
        }
        Ok(out)
    }

    fn magnon_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        qs: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<f64>> {
        self.validate()?;
        if qs.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "magnon q-point block".into(),
                expected: DIM,
                found: qs.ncols(),
            });
        }
        let n = self.nspin();
        let points: Vec<Array1<f64>> = qs.axis_iter(Axis(0)).map(|q| q.to_owned()).collect();
        let rows: Vec<Result<Array1<f64>>> = points
            .par_iter()
            .map(|q| {
                let point = self.check_q(q)?;
                let h = self.bdg_of(&point)?;
                paraunitary_frequencies(&h, n)
            })
            .collect();
        let mut out = Array2::<f64>::zeros((qs.nrows(), n));
        for (row, values) in rows.into_iter().enumerate() {
            out.row_mut(row).assign(&values?);
        }
        Ok(out)
    }
}

/// Positive paraunitary eigenvalues of `g H` via Colpa's factorization.
///
/// `H` must be positive semidefinite (a stable reference structure). The
/// Hermitian eigenfactorization `H = U L U^dagger` gives `K = U L^(1/2)` and
/// the spectrum of the Hermitian `K^dagger g K`; zero eigenvalues of `H`
/// produce exact zero frequencies.
pub(crate) fn paraunitary_frequencies(h: &Array2<Complex<f64>>, n: usize) -> Result<Array1<f64>> {
    let size = h.nrows();
    if size != 2 * n || h.ncols() != size {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "bdg_shape",
            message: format!(
                "expected a {} x {} BdG matrix, found {:?}",
                2 * n,
                2 * n,
                h.dim()
            ),
        });
    }
    let (values, vectors) = eigh_full(h, UPLO::Lower)?;
    let lowest = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let largest = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0, |acc: f64, value| acc.max(value));
    // Relative tolerance plus a tiny absolute floor: at a pure Goldstone point
    // the whole matrix is zero and a purely relative bound would vanish.
    let tolerance = BDG_POSITIVITY_RTOL * largest + 1e-13;
    if !lowest.is_finite() || lowest < -tolerance {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "positive_semidefinite_bdg",
            message: format!(
                "the reference structure is not a local energy minimum: \
                 lowest BdG eigenvalue {lowest:e} eV"
            ),
        });
    }
    // Vectors are stored as rows (ket coefficients): u[n, alpha] = <alpha|n>.
    let mut metric = Array2::<Complex<f64>>::zeros((size, size));
    for row in 0..size {
        for column in 0..size {
            let mut sum = Complex::new(0.0, 0.0);
            for alpha in 0..size {
                let sign = if alpha < n { 1.0 } else { -1.0 };
                sum += vectors[[row, alpha]].conj() * sign * vectors[[column, alpha]];
            }
            metric[[row, column]] = sum;
        }
    }
    // M[n, m] = sqrt(lambda_n lambda_m) metric[n, m]; zero eigenvalues give an
    // exactly zero row and column, so gapless modes are not perturbed.
    let mut m = Array2::<Complex<f64>>::zeros((size, size));
    for row in 0..size {
        let root_row = values[row].max(0.0).sqrt();
        for column in 0..size {
            let root_column = values[column].max(0.0).sqrt();
            m[[row, column]] = metric[[row, column]] * root_row * root_column;
        }
    }
    let (spectrum, _) = eigh_full(&m, UPLO::Lower)?;
    let mut out = Array1::<f64>::zeros(n);
    for (index, value) in spectrum.iter().skip(size - n).enumerate() {
        out[index] = *value;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magnon::MagnonReference;
    use ndarray::{Array3, array};

    fn isotropic(j: f64) -> ndarray::Array2<f64> {
        let mut block = ndarray::Array2::<f64>::zeros((3, 3));
        for d in 0..3 {
            block[[d, d]] = j;
        }
        block
    }

    /// One magnetic site, nearest-neighbour exchange `j` along x.
    fn chain(j: f64, spin: f64) -> Magnon<1> {
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        ham.slice_mut(ndarray::s![1, .., ..]).assign(&isotropic(j));
        ham.slice_mut(ndarray::s![2, .., ..]).assign(&isotropic(j));
        Magnon::<1>::new(
            array![[1.0]],
            array![[0.0]],
            array![spin],
            array![[0.0, 0.0, 1.0]],
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    /// Two-site chain: `parallel` moments (ferromagnet) or +z / -z (antiferromagnet).
    fn chain_two_site(j: f64, s0: f64, s1: f64, parallel: bool) -> Magnon<1> {
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 6, 6));
        let mut add = |i_r: usize, i: usize, site_j: usize| {
            for d in 0..3 {
                ham[[i_r, 3 * i + d, 3 * site_j + d]] = j;
            }
        };
        add(0, 0, 1);
        add(0, 1, 0);
        add(1, 1, 0); // J_10(R = +1)
        add(2, 0, 1); // J_01(R = -1)
        let second = if parallel { 1.0 } else { -1.0 };
        let axis = array![[0.0, 0.0, 1.0], [0.0, 0.0, second]];
        Magnon::<1>::new(
            array![[1.0]],
            array![[0.0], [0.5]],
            array![s0, s1],
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    /// Three-site ring with 120-degree moments in the xy plane.
    fn ring_120(j: f64, spin: f64) -> Magnon<1> {
        let angle = std::f64::consts::TAU / 3.0;
        let mut axis = ndarray::Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            axis[[i, 0]] = (angle * i as f64).cos();
            axis[[i, 1]] = (angle * i as f64).sin();
        }
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 9, 9));
        let mut add = |i_r: usize, i: usize, site_j: usize| {
            for d in 0..3 {
                ham[[i_r, 3 * i + d, 3 * site_j + d]] = j;
            }
        };
        add(0, 0, 1);
        add(0, 1, 0);
        add(0, 1, 2);
        add(0, 2, 1);
        add(1, 2, 0); // J_20(R = +1)
        add(2, 0, 2); // J_02(R = -1)
        Magnon::<1>::new(
            array![[1.0]],
            array![[0.0], [1.0 / 3.0], [2.0 / 3.0]],
            array![spin, spin, spin],
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    fn single_site_sia(k: f64, spin: f64) -> Magnon<1> {
        let mut ham = Array3::<f64>::zeros((1, 3, 3));
        ham[[0, 2, 2]] = k;
        Magnon::<1>::new(
            array![[1.0]],
            array![[0.0]],
            array![spin],
            array![[0.0, 0.0, 1.0]],
            ham,
            array![[0isize]],
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    fn afm_chain(j: f64, spin: f64) -> Magnon<1> {
        chain_two_site(j, spin, spin, false)
    }

    #[test]
    fn test_ferromagnetic_chain_dispersion() {
        let (j, spin) = (0.01, 1.5);
        let magnon = chain(j, spin);
        for q in [0.0, 0.125, 0.25, 0.5] {
            let point = array![q];
            let omega = magnon.magnon_band_onek(&point).unwrap()[0];
            let expected = 4.0 * j * (1.0 - (std::f64::consts::TAU * q).cos()) / spin;
            assert!(
                (omega - expected).abs() < 1e-12,
                "q = {q}: {omega} vs {expected}"
            );
        }
    }

    #[test]
    fn test_dispersion_scales_as_inverse_spin() {
        let point = array![0.25];
        let light = chain(0.01, 1.5).magnon_band_onek(&point).unwrap()[0];
        let heavy = chain(0.01, 0.75).magnon_band_onek(&point).unwrap()[0];
        assert!((heavy / light - 2.0).abs() < 1e-12, "{heavy} vs {light}");
    }

    #[test]
    fn test_two_sublattice_antiferromagnet_dispersion() {
        // TB2J reference values for the same model (scaled to S = 1.5):
        // omega(q) = (4 |j| / S) |sin(pi q)|, both branches degenerate.
        let (j, spin) = (-0.02, 1.5);
        let magnon = afm_chain(j, spin);
        for q in [0.0, 0.125, 0.25, 0.375, 0.5] {
            let point = array![q];
            let omega = magnon.magnon_band_onek(&point).unwrap();
            let expected = 4.0 * j.abs() / spin * (std::f64::consts::PI * q).sin().abs();
            for value in omega.iter() {
                assert!(
                    (value - expected).abs() < 1e-8,
                    "q = {q}: {value} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn test_ring_has_three_exact_goldstone_modes() {
        let magnon = ring_120(-0.02, 1.5);
        let omega = magnon.magnon_band_onek(&array![0.0]).unwrap();
        assert_eq!(omega.len(), 3);
        for value in omega.iter() {
            assert!(value.abs() < 1e-8, "Goldstone mode {value} is not zero");
        }
        // Away from the ordering vector this reference is a saddle of the
        // bipartite chain: an independent frozen-magnon Hessian has the same
        // negative eigenvalue, and the solver reports the indefinite BdG.
        let error = magnon.magnon_band_onek(&array![1.0 / 3.0]).unwrap_err();
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => {
                assert_eq!(invariant, "positive_semidefinite_bdg");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    /// Triangular lattice, one site per cell, 120-degree order carried by a
    /// single-Q propagation vector and the reference rotation.
    fn triangular_120(j: f64, spin: f64) -> Magnon<2> {
        let ham_r = array![
            [0isize, 0],
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [1, 1],
            [-1, -1]
        ];
        let mut ham = Array3::<f64>::zeros((7, 3, 3));
        for i_r in 1..7 {
            for d in 0..3 {
                ham[[i_r, d, d]] = j;
            }
        }
        let mut reference = MagnonReference::<2>::collinear();
        reference.q = array![1.0 / 3.0, 1.0 / 3.0];
        Magnon::<2>::new(
            array![[1.0, 0.0], [0.5, 3.0f64.sqrt() / 2.0]],
            array![[0.0, 0.0]],
            array![spin],
            array![[0.0, 0.0, 1.0]],
            ham,
            ham_r,
            reference,
        )
        .unwrap()
    }

    #[test]
    fn test_triangular_120_degree_order_is_stable() {
        let magnon = triangular_120(-0.02, 1.5);
        // Gapless at the magnetic Gamma point, which the chemical-zone folding
        // places at q = 0 and q = Q.
        for q in [array![0.0, 0.0], array![1.0 / 3.0, 1.0 / 3.0]] {
            let omega = magnon.magnon_band_onek(&q).unwrap()[0];
            assert!(omega.abs() < 1e-8, "q = {q:?}: {omega}");
        }
        // TB2J reference value for the same model (S = 1.5, j = -0.02).
        let omega = magnon.magnon_band_onek(&array![0.25, 0.0]).unwrap()[0];
        assert!((omega - 8.432740427116e-2).abs() < 1e-10, "{omega}");
        for q in [array![0.5, 0.5], array![0.4, 0.1], array![0.125, 0.375]] {
            let omega = magnon.magnon_band_onek(&q).unwrap()[0];
            assert!(omega.is_finite() && omega >= -1e-10, "q = {q:?}: {omega}");
        }
    }

    #[test]
    fn test_single_ion_anisotropy_gap() {
        let (k, spin) = (0.01, 1.5);
        let omega = single_site_sia(k, spin)
            .magnon_band_onek(&array![0.0])
            .unwrap();
        assert_eq!(omega.len(), 1);
        assert!((omega[0] - 2.0 * k / spin).abs() < 1e-12, "{omega:?}");
    }

    #[test]
    fn test_bdg_is_hermitian_with_dmi() {
        // Two sites, isotropic exchange plus an antisymmetric DMI tensor.
        let j = 0.01;
        let d = 0.002;
        let mut forward = ndarray::Array2::<f64>::zeros((3, 3));
        for a in 0..3 {
            forward[[a, a]] = j;
        }
        forward[[0, 1]] += d;
        forward[[1, 0]] -= d;
        let mut backward = ndarray::Array2::<f64>::zeros((3, 3));
        for a in 0..3 {
            backward[[a, a]] = j;
        }
        backward[[0, 1]] -= d;
        backward[[1, 0]] += d;
        let mut ham = Array3::<f64>::zeros((1, 6, 6));
        for a in 0..3 {
            for b in 0..3 {
                ham[[0, a, 3 + b]] = forward[[a, b]];
                ham[[0, 3 + a, b]] = backward[[a, b]];
            }
        }
        let magnon = Magnon::<1>::new(
            array![[1.0]],
            array![[0.0], [0.5]],
            array![1.5, 1.5],
            array![[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]],
            ham,
            array![[0isize]],
            MagnonReference::collinear(),
        )
        .unwrap();
        let h = magnon.gen_bdg(&array![0.0]).unwrap();
        let mut worst: f64 = 0.0;
        for i in 0..h.nrows() {
            for j in 0..h.ncols() {
                worst = worst.max((h[[i, j]] - h[[j, i]].conj()).norm());
            }
        }
        assert!(worst < 1e-15, "BdG matrix is not Hermitian: {worst:e}");
    }

    #[test]
    fn test_unstable_reference_is_reported() {
        // Antiferromagnetic coupling with a ferromagnetic reference: the BdG
        // matrix is indefinite.
        let magnon = chain(-0.02, 1.5);
        let error = magnon.magnon_band_onek(&array![0.25]).unwrap_err();
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => {
                assert_eq!(invariant, "positive_semidefinite_bdg");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_single_q_ordering_is_gapless() {
        // One-site chain with antiferromagnetic coupling and Q = (1/2, 0, 0):
        // the rotated reference structure is the classical ground state, so the
        // q = 0 mode is the Goldstone mode.
        let mut magnon = chain(-0.02, 1.5);
        magnon.reference.q = array![0.5];
        let omega = magnon.magnon_band_onek(&array![0.0]).unwrap();
        assert!(omega[0].abs() < 1e-8, "{omega:?}");
        let gapped = magnon.magnon_band_onek(&array![0.25]).unwrap();
        assert!(gapped[0] > 1e-3, "{gapped:?}");
    }

    #[test]
    fn test_band_all_matches_serial_and_parallel() {
        let magnon = chain(0.01, 1.5);
        let points = ndarray::Array2::from_shape_fn((7, 1), |(row, _)| row as f64 / 8.0);
        let serial = magnon.magnon_band_all(&points).unwrap();
        let parallel = magnon.magnon_band_all_parallel(&points).unwrap();
        assert_eq!(serial.dim(), (7, 1));
        for row in 0..7 {
            assert!((serial[[row, 0]] - parallel[[row, 0]]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_kpath_and_bands() {
        use crate::kpath::Kpath;
        let magnon = chain(0.01, 1.5);
        let path = array![[0.0], [0.5], [0.0]];
        let (points, distances, nodes) = magnon.k_path(&path, 11).unwrap();
        assert_eq!(points.dim(), (11, 1));
        assert_eq!(distances.len(), 11);
        assert_eq!(nodes.len(), 3);
        let bands = magnon.magnon_band_all(&points).unwrap();
        assert_eq!(bands.dim(), (11, 1));
        assert!(
            bands
                .iter()
                .all(|value| value.is_finite() && *value >= -1e-12)
        );
    }

    #[test]
    fn test_dos_normalization() {
        let magnon = chain(0.01, 1.5);
        let (omega, dos) = magnon
            .magnon_dos(&array![16usize], -0.01, 0.1, 401, 0.002)
            .unwrap();
        assert_eq!(omega.len(), 401);
        let d_omega = omega[1] - omega[0];
        let total: f64 = dos.sum() * d_omega;
        assert!((total - 1.0).abs() < 1e-3, "integrated DOS = {total}");
    }

    #[test]
    fn test_q_point_validation() {
        let magnon = chain(0.01, 1.5);
        let error = magnon.magnon_band_onek(&array![0.0, 0.0]).unwrap_err();
        assert!(matches!(error, TbError::DimensionMismatch { .. }));
        let error = magnon.magnon_band_onek(&array![f64::NAN]).unwrap_err();
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => {
                assert_eq!(invariant, "finite_q_point")
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_ferrimagnetic_chain_dispersion() {
        // TB2J reference values: two-site ferromagnet with S = (1, 2), j = 0.01.
        let magnon = chain_two_site(0.01, 1.0, 2.0, true);
        for (q, low, high) in [
            (0.0, 0.0, 6.0e-2),
            (0.125, 2.020673480682e-3, 5.797932651932e-2),
            (0.25, 7.639320225002e-3, 5.236067977500e-2),
            (0.5, 2.0e-2, 4.0e-2),
        ] {
            let omega = magnon.magnon_band_onek(&array![q]).unwrap();
            assert!((omega[0] - low).abs() < 1e-10, "q = {q}: {omega:?}");
            assert!((omega[1] - high).abs() < 1e-10, "q = {q}: {omega:?}");
        }
    }

    #[test]
    fn test_gen_jq_matches_hand_computed_fourier_sum() {
        let (j, _spin) = (0.01, 1.5);
        let magnon = chain(j, 1.5);
        for q in [0.0, 0.125, 0.25] {
            let jq = magnon.gen_jq(&array![q]).unwrap();
            // J(q) = 2 j cos(2 pi q) on the diagonal, zero elsewhere.
            let expected = 2.0 * j * (std::f64::consts::TAU * q).cos();
            assert!((jq[[0, 0]].re - expected).abs() < 1e-14, "q = {q}");
            assert!(jq[[0, 1]].norm() < 1e-15);
            assert!(jq[[1, 2]].norm() < 1e-15);
        }
    }

    #[test]
    fn test_single_q_rotation_keeps_anisotropic_exchange_hermitian() {
        // A tensor that does not commute with the reference rotation: the raw
        // TB2J-style right multiplication alone would leave J(q) non-Hermitian.
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        for (i_r, scale) in [(1usize, 1.0), (2, 1.0)] {
            ham[[i_r, 0, 0]] = -0.02 * scale;
            ham[[i_r, 1, 1]] = -0.02 * scale;
            ham[[i_r, 2, 2]] = -0.025 * scale;
        }
        let mut reference = MagnonReference::<1>::collinear();
        reference.q = array![0.25];
        let magnon = Magnon::<1>::new(
            array![[1.0]],
            array![[0.0]],
            array![1.5],
            array![[0.0, 0.0, 1.0]],
            ham,
            array![[0isize], [1], [-1]],
            reference,
        )
        .unwrap();
        for q in [0.0, 0.1, 0.25, 0.5] {
            let jq = magnon.gen_jq(&array![q]).unwrap();
            let adjoint = jq.t().mapv(|value| value.conj());
            let deviation = (&jq - &adjoint)
                .mapv(|value| value.norm())
                .fold(0.0, |acc: f64, value| acc.max(*value));
            assert!(
                deviation < 1e-15,
                "q = {q}: J(q) is not Hermitian ({deviation:e})"
            );
            let h = magnon.gen_bdg(&array![q]).unwrap();
            let hadj = h.t().mapv(|value| value.conj());
            let deviation = (&h - &hadj)
                .mapv(|value| value.norm())
                .fold(0.0, |acc: f64, value| acc.max(*value));
            assert!(
                deviation < 1e-15,
                "q = {q}: BdG is not Hermitian ({deviation:e})"
            );
        }
    }

    #[test]
    fn test_frames_fallback_for_non_coordinate_quantization_axis() {
        // Moment parallel to a non-coordinate quantization axis: the fallback
        // triad must still be orthonormal, and the spectrum is unchanged
        // because the exchange is isotropic.
        let mut magnon = chain(0.01, 1.5);
        let axis = 1.0 / 2.0f64.sqrt();
        magnon.axis = array![[axis, axis, 0.0]];
        magnon.reference.quantization_axis = array![axis, axis, 0.0];
        let (u, v) = super::local_frames(&magnon);
        let u_real: Vec<f64> = (0..3).map(|a| u[[0, a]].re).collect();
        let u_imag: Vec<f64> = (0..3).map(|a| u[[0, a]].im).collect();
        let v0 = [v[[0, 0]], v[[0, 1]], v[[0, 2]]];
        assert!(
            (dot(
                &[u_real[0], u_real[1], u_real[2]],
                &[u_real[0], u_real[1], u_real[2]]
            ) - 1.0)
                .abs()
                < 1e-12
        );
        assert!(
            dot(
                &[u_real[0], u_real[1], u_real[2]],
                &[u_imag[0], u_imag[1], u_imag[2]]
            )
            .abs()
                < 1e-12
        );
        assert!(dot(&[u_real[0], u_real[1], u_real[2]], &v0).abs() < 1e-12);
        assert!(dot(&[u_imag[0], u_imag[1], u_imag[2]], &v0).abs() < 1e-12);
        assert!((dot(&v0, &v0) - 1.0).abs() < 1e-12);
        for q in [0.0, 0.25, 0.5] {
            let omega = magnon.magnon_band_onek(&array![q]).unwrap()[0];
            let expected = 4.0 * 0.01 * (1.0 - (std::f64::consts::TAU * q).cos()) / 1.5;
            assert!((omega - expected).abs() < 1e-12, "q = {q}: {omega}");
        }
    }

    #[test]
    fn test_dos_rejects_invalid_parameters() {
        let magnon = chain(0.01, 1.5);
        let mesh = array![4usize];
        let error = magnon.magnon_dos(&mesh, 0.1, 0.0, 11, 0.002).unwrap_err();
        assert!(matches!(error, TbError::InvalidEnergyRange { .. }));
        let error = magnon.magnon_dos(&mesh, 0.0, 0.1, 0, 0.002).unwrap_err();
        assert!(matches!(error, TbError::InvalidDosParameter { .. }));
        let error = magnon.magnon_dos(&mesh, 0.0, 0.1, 11, 0.0).unwrap_err();
        assert!(matches!(error, TbError::InvalidDosParameter { .. }));
    }
}
