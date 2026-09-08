//! Velocity operator $\mathbf{v}(\mathbf{k}) = \nabla\_{\mathbf{k}} H(\mathbf{k})$ for tight-binding models.
//!
//! Provides the [`Velocity`] trait and its implementation for [`Model`]<SPIN, DIM, R>,
//! computing matrix elements $\bra{m\mathbf{k}} \partial\_\alpha H\_{\mathbf{k}} \ket{n\mathbf{k}}$
//! at a given k-point. Essential for Berry curvature, optical conductivity,
//! and other transport calculations.
//!
//! # Physical background
//!
//! ## Bloch basis and velocity operator
//!
//! The Bloch eigenstates satisfy
//! $H \ket{\psi\_{n\mathbf{k}}} = \varepsilon\_{n\mathbf{k}} \ket{\psi\_{n\mathbf{k}}}$.
//! Define the periodic part $\ket{u\_{n\mathbf{k}}} = e^{-i\mathbf{k}\cdot\mathbf{r}} \ket{\psi\_{n\mathbf{k}}}$,
//! which obeys $H\_{\mathbf{k}} \ket{u\_{n\mathbf{k}}} = \varepsilon\_{n\mathbf{k}} \ket{u\_{n\mathbf{k}}}$
//! with $H\_{\mathbf{k}} = e^{-i\mathbf{k}\cdot\mathbf{r}} H e^{+i\mathbf{k}\cdot\mathbf{r}}$.
//! The periodic functions are orthonormal within the unit cell $\Omega$:
//! $\braket{u\_{m\mathbf{k}}}{u\_{n\mathbf{k}}} = \int\_\Omega d\mathbf{r}\\, u\_{n\mathbf{k}}^*(\mathbf{r}) u\_{m\mathbf{k}}(\mathbf{r}) = \delta\_{mn}$.
//!
//! From the Heisenberg equation, $\mathbf{v} = \dot{\mathbf{r}} = \frac{i}{\hbar}[H, \mathbf{r}]$.
//! Transforming to the Bloch basis:
//!
//! ```math
//! \mathbf{v}_{\mathbf{k}} \equiv e^{-i\mathbf{k}\cdot\mathbf{r}} \mathbf{v} e^{+i\mathbf{k}\cdot\mathbf{r}}
//! = \frac{1}{\hbar} \nabla_{\mathbf{k}} H_{\mathbf{k}} .
//! ```
//!
//! The velocity operator matrix in the band eigenbasis is therefore
//!
//! ```math
//! \bra{\psi_{m\mathbf{k}}} \mathbf{v} \ket{\psi_{n\mathbf{k}}}
//! = \frac{1}{\hbar} \bra{u_{m\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{u_{n\mathbf{k}}} .
//! ```
//!
//! **Note**: In the code, $\hbar$ is absorbed into the unit system (effectively $\hbar = 1$),
//! so `gen_v` returns $\nabla\_{\mathbf{k}} H\_{\mathbf{k}}$ rather than $\frac{1}{\hbar}\nabla\_{\mathbf{k}} H\_{\mathbf{k}}$.
//! Physical constants are applied at the level of transport coefficients (e.g. the $e^2/\hbar$ prefactor
//! in the Hall conductivity).
//!
//! ## Wannier basis and gauges
//!
//! The tight-binding model is built from Wannier functions $\ket{\alpha\mathbf{R}}$,
//! where $\alpha$ labels orbitals and $\mathbf{R}$ labels unit cells.
//! Two common Fourier conventions (gauges) are used:
//!
//! **Lattice gauge** (Wannier90 default):
//! ```math
//! \ket{\psi^W_{\alpha\mathbf{k}}} = \frac{1}{\sqrt{N}} \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} \ket{\alpha\mathbf{R}},
//! \qquad
//! \ket{e^W_{\alpha\mathbf{k}}} = e^{-i\mathbf{k}\cdot\hat{\mathbf{r}}} \ket{\psi^W_{\alpha\mathbf{k}}} .
//! ```
//!
//! **Atom gauge** (includes orbital positions $\boldsymbol{\tau}\_\alpha$):
//! ```math
//! \ket{\alpha\mathbf{k}} = \frac{1}{\sqrt{N}} \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot(\mathbf{R} + \boldsymbol{\tau}_\alpha)} \ket{\alpha\mathbf{R}},
//! \qquad
//! \ket{e_{\alpha\mathbf{k}}} = e^{-i\mathbf{k}\cdot\hat{\mathbf{r}}} \ket{\alpha\mathbf{k}} .
//! ```
//!
//! Both bases satisfy $\ket{\alpha\mathbf{k} + \mathbf{G}} = \ket{\alpha\mathbf{k}}$ for reciprocal lattice
//! vectors $\mathbf{G}$, and are orthonormal: $\braket{e\_{\alpha\mathbf{k}}}{e\_{\beta\mathbf{k}}} = \delta\_{\alpha\beta}$.
//! The two are related by $\ket{\alpha\mathbf{k}} = e^{i\mathbf{k}\cdot\boldsymbol{\tau}\_\alpha} \ket{\psi^W\_{\alpha\mathbf{k}}}$.
//!
//! ### Lattice gauge velocity formula
//!
//! In the Lattice gauge, the Hamiltonian and its derivative are:
//!
//! ```math
//! \begin{aligned}
//! H^W_{\alpha\beta}(\mathbf{k}) &\equiv \bra{\psi^W_{\alpha\mathbf{k}}} H \ket{\psi^W_{\beta\mathbf{k}}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot\mathbf{R}}, \\[6pt]
//! \nabla_{\mathbf{k}} H^W_{\alpha\beta}(\mathbf{k}) &= i \sum_{\mathbf{R}} \mathbf{R} \, \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot\mathbf{R}} .
//! \end{aligned}
//! ```
//!
//! The Berry connection (position matrix) in the Lattice gauge is directly available from Wannier90
//! (`seedname_r.dat`):
//!
//! ```math
//! A^W_{\alpha\beta}(\mathbf{k}) \equiv i \bra{e^W_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} \ket{e^W_{\beta\mathbf{k}}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} \hat{\mathbf{r}} \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot\mathbf{R}} .
//! ```
//!
//! Using
//!
//! ```math
//! \nabla_{\mathbf{k}} H^W
//! = \bra{e^W} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{e^W}
//! + \bra{\nabla_{\mathbf{k}} e^W} H_{\mathbf{k}} \ket{e^W}
//! + \bra{e^W} H_{\mathbf{k}} \ket{\nabla_{\mathbf{k}} e^W}
//! ```
//!
//! and inserting $\mathbb{1} = \sum\_\gamma \ketbra{e^W\_{\gamma\mathbf{k}}}$, one obtains:
//!
//! ```math
//! \bra{e^W_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{e^W_{\beta\mathbf{k}}}
//! = \nabla_{\mathbf{k}} H^W_{\alpha\beta}(\mathbf{k}) + i\bigl[ H^W(\mathbf{k}), A^W(\mathbf{k}) \bigr]_{\alpha\beta} .
//! ```
//!
//! Transforming to the band eigenbasis $\ket{u\_{n\mathbf{k}}} = \sum\_\alpha \ket{e^W\_{\alpha\mathbf{k}}} C\_{\alpha n}(\mathbf{k})$
//! yields the lattice-gauge velocity:
//!
//! ```math
//! {\color{red}\boxed{\color{black}
//! \bra{u_{m\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{u_{n\mathbf{k}}}
//! = \bra{u_{m\mathbf{k}}} \Bigl( \nabla_{\mathbf{k}} H^W(\mathbf{k}) + i\bigl[ H^W(\mathbf{k}), A^W(\mathbf{k}) \bigr] \Bigr) \ket{u_{n\mathbf{k}}} }} .
//! ```
//!
//! ### Atom gauge velocity formula
//!
//! In the Atom gauge, the Hamiltonian includes $\boldsymbol{\tau}$ phases:
//!
//! ```math
//! \bar{H}_{\alpha\beta}(\mathbf{k}) \equiv \bra{\alpha\mathbf{k}} H \ket{\beta\mathbf{k}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta)} .
//! ```
//!
//! Its derivative splits into an $\mathbf{R}$-term and a $\boldsymbol{\tau}$-term:
//!
//! ```math
//! \nabla_{\mathbf{k}} \bar{H}_{\alpha\beta}(\mathbf{k})
//! = i \sum_{\mathbf{R}} \bigl(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta\bigr)
//!   \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta)} .
//! ```
//!
//! The Berry connection in the Atom gauge is
//!
//! ```math
//! \bar{\mathbf{r}}_{\alpha\beta}(\mathbf{k}) \equiv i \bra{e_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} \ket{e_{\beta\mathbf{k}}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} \hat{\mathbf{r}} \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta)}
//!   - \boldsymbol{\tau}_\alpha \delta_{\alpha\beta} .
//! ```
//!
//! Following the same derivation as for the Lattice gauge,
//! $\bra{e\_{\alpha\mathbf{k}}} \nabla\_{\mathbf{k}} H\_{\mathbf{k}} \ket{e\_{\beta\mathbf{k}}}
//! = \nabla\_{\mathbf{k}} \bar{H}\_{\alpha\beta}(\mathbf{k}) + i\bigl[ \bar{H}(\mathbf{k}), \bar{\mathbf{r}} \bigr]\_{\alpha\beta}$.
//!
//! A useful identity: the $\boldsymbol{\tau}$-dependent diagonal part of $\bar{\mathbf{r}}$ generates,
//! through the commutator $i[\bar{H}, \bar{\mathbf{r}}]$, a term $-i(\tau\_n - \tau\_m) \bar{H}\_{mn}$
//! that **exactly cancels** the $\boldsymbol{\tau}$-difference piece in $\nabla\_{\mathbf{k}} \bar{H}$.
//! Therefore one may drop the $-\boldsymbol{\tau}\_\alpha \delta\_{\alpha\beta}$ from $\bar{\mathbf{r}}$
//! (i.e. set its diagonal to zero) *provided* the $i(\boldsymbol{\tau}\_n - \boldsymbol{\tau}\_m) \bar{H}\_{mn}$ term
//! is also omitted from $\nabla\_{\mathbf{k}} \bar{H}$. This is how the code is implemented.
//!
//! The net Atom-gauge velocity, in component form (direction $\alpha$, Cartesian coordinates), is:
//!
//! ```math
//! {\color{red}\boxed{\color{black}
//! \begin{aligned}
//! \bra{m\mathbf{k}} \partial_\alpha H_{\mathbf{k}} \ket{n\mathbf{k}}
//! &= \sum_{\mathbf{R}} i R_\alpha^{\rm (cart)} H_{mn}(\mathbf{R})\, e^{2\pi i\,\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_m + \boldsymbol{\tau}_n)} \\[4pt]
//! &+ i\bigl(\tau_{n\alpha}^{\rm (cart)} - \tau_{m\alpha}^{\rm (cart)}\bigr)\, H_{mn}(\mathbf{k}) \\[4pt]
//! &- \bigl[ H(\mathbf{k}), \mathcal{A}_{\mathbf{k},\alpha} \bigr]_{mn}
//! \end{aligned}}}
//! ```
//!
//! where the Berry connection matrix is:
//!
//! ```math
//! \mathcal{A}_{\mathbf{k},\alpha,mn} = -i \sum_{\mathbf{R}} r_{mn,\alpha}(\mathbf{R})\, e^{2\pi i\,\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_m + \boldsymbol{\tau}_n)}
//! + i \tau_{n\alpha} \delta_{mn} .
//! ```
//!
//! The position matrix elements $\mathbf{r}\_{mn}(\mathbf{R})$ are provided by Wannier90
//! (setting `write_rmn = true`). Their availability is checked at compile time via
//! `<R as RMatrixData>::HAS_RMATRIX`. For [`crate::NoRMatrix`] (the default), the commutator
//! term $[H(\mathbf{k}), \mathcal{A}\_{\mathbf{k},\alpha}]$ is omitted.
//!
//! # Implementation notes
//!
//! The code first constructs $H(\mathbf{k}) = \sum\_{\mathbf{R}} H(\mathbf{R}) e^{2\pi i \mathbf{k}\cdot\mathbf{R}}$
//! (Lattice gauge). For [`Gauge::Atom`], it then:
//!
//! 1. Computes the $\mathbf{R}$-term: $i R\_\alpha H(\mathbf{R}) e^{2\pi i\mathbf{k}\cdot\mathbf{R}}$,
//!    then applies the gauge transform $e^{2\pi i\mathbf{k}\cdot(\boldsymbol{\tau}\_n - \boldsymbol{\tau}\_m)}$
//!    to convert Lattice-gauge phases to Atom-gauge phases ($\mathbf{R} \to \mathbf{R} - \boldsymbol{\tau}\_m + \boldsymbol{\tau}\_n$).
//! 2. Adds the $\boldsymbol{\tau}$-difference term $i(\tau\_{n\alpha} - \tau\_{m\alpha}) H\_{mn}(\mathbf{k})$.
//! 3. Computes the Berry connection $A^W$ in the Lattice gauge, applies the same gauge transform,
//!    sets the diagonal to zero (exploiting the $\boldsymbol{\tau}$-cancellation identity),
//!    and adds $-i[H(\mathbf{k}), A]$.
//!
//! For [`Gauge::Lattice`], the $\boldsymbol{\tau}$-dependent steps are skipped.
//!
//! # Conventions
//!
//! - **k**: fractional reciprocal coordinates; the phase factor is $e^{2\pi i\\,\mathbf{k}\cdot\mathbf{R}}$
//! - **R**: integer lattice vectors from `hamR`
//! - $R\_\alpha^{\rm (cart)}$, $\tau\_{n\alpha}^{\rm (cart)}$: Cartesian coordinates (in Å),
//!   obtained by multiplying fractional vectors with the lattice matrix `lat`
//! - For a Hermitian Hamiltonian and position operator, the returned velocity
//!   is **Hermitian**: $v\_\alpha^\dagger = v\_\alpha$.
use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::comm;
use crate::model_physics::{apply_atom_gauge, fourier_sum};
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;

/// Trait for computing the velocity operator $\mathbf{v}(\mathbf{k})$.
///
/// The velocity operator is defined as the k-derivative of the Bloch Hamiltonian:
///
/// ```math
/// \mathbf{v}(\mathbf{k}) = \nabla_{\mathbf{k}} H(\mathbf{k})
/// ```
///
/// Note this is the **full velocity operator matrix** in the Bloch basis,
/// not just the band-diagonal group velocity $\partial E\_n/\partial\mathbf{k}$.
///
/// In the [`Model`]<SPIN, DIM, R> implementation, the position-matrix
/// commutator term $i[H, \mathcal{A}]$ is included only when
/// `<R as RMatrixData>::HAS_RMATRIX` is `true`, checked at compile time.
///
/// # Returns
///
/// `(v, hamk)` where:
/// - `v` is a $d \times N\_{\rm sta} \times N\_{\rm sta}$ array giving
///   $v\_{\alpha,mn}$ for each direction $\alpha$
/// - `hamk` is the $N\_{\rm sta} \times N\_{\rm sta}$ Bloch Hamiltonian $H(\mathbf{k})$
pub trait Velocity {
    /// Compute the velocity operator at a single k-point.
    ///
    /// # Arguments
    ///
    /// * `kvec` — k-point in fractional reciprocal coordinates (length = `dim_r()`).
    /// * `gauge` — [`Gauge::Lattice`] or [`Gauge::Atom`]. Physical observables are
    ///   gauge-invariant.
    ///
    /// # Panics
    ///
    /// Panics if `kvec.len() != self.dim_r()`.
    fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>);
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Velocity for Model<SPIN, DIM, R> {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        let (velocity, hams) = self.gen_v_batch(&kvec.view().insert_axis(Axis(0)), gauge);
        (
            velocity.index_axis_move(Axis(0), 0),
            hams.index_axis_move(Axis(0), 0),
        )
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Projected velocity operator: computes direction-weighted velocity matrices
    /// directly without materializing the per-direction `(DIM, nsta, nsta)` array.
    ///
    /// For each row `w = directions.row(p)`, computes:
    ///
    /// ```text
    /// v_proj[p] = Σ_d w[d] · v_raw[d]
    /// ```
    ///
    /// where `v_raw[d]` is the velocity operator for Cartesian direction d.
    ///
    /// This fuses the direction summation into the R-term and τ-term of `gen_v`,
    /// avoiding the intermediate `(DIM, nsta, nsta)` allocation and the secondary
    /// projection pass. The R-term scalar becomes `i·(w·R)` instead of `i·R_d`,
    /// and the τ-term uses `i·(w·τ_n − w·τ_m)` instead of per-direction differences.
    ///
    /// # Arguments
    /// * `kvec` — k-point in fractional coordinates (length = `dim_r()`).
    /// * `gauge` — [`Gauge::Atom`] or [`Gauge::Lattice`].
    /// * `directions` — shape `(n_proj, dim_r)`. Each row is a direction weight vector.
    ///
    /// # Returns
    /// `(v_proj, hamk)` where `v_proj` has shape `(n_proj, nsta, nsta)` and
    /// `hamk` is the Bloch Hamiltonian `H(k)`.
    #[allow(non_snake_case)]
    pub fn gen_v_projected<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
        directions: &Array2<f64>,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        let (velocity, hams) =
            self.gen_v_projected_batch(&kvec.view().insert_axis(Axis(0)), gauge, directions);
        (
            velocity.index_axis_move(Axis(0), 0),
            hams.index_axis_move(Axis(0), 0),
        )
    }

    /// Construct Cartesian velocities and Hamiltonians for `(nk, DIM)` k-points.
    ///
    /// Returns arrays shaped `(nk, DIM, nsta, nsta)` and `(nk, nsta, nsta)`.
    /// H and all derivative sums share one GEMM. No Rayon tasks are created;
    /// callers control batch size and the selected BLAS backend's threads.
    /// The gauge and position-matrix corrections agree with [`Velocity::gen_v`].
    pub fn gen_v_batch<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        gauge: Gauge,
    ) -> (Array4<Complex<f64>>, Array3<Complex<f64>>) {
        self.gen_v_projected_batch(points, gauge, &Array2::eye(DIM))
    }

    /// Batch counterpart of [`Self::gen_v_projected`].
    ///
    /// `points` has shape `(nk, DIM)` and `directions` is `(n_proj, DIM)`.
    /// Returns `(nk, n_proj, nsta, nsta)` velocities and `(nk, nsta, nsta)`
    /// Hamiltonians in input order. Workspace grows with the supplied batch;
    /// pass bounded subsets when processing large k-meshes.
    ///
    /// # Panics
    /// Panics for incompatible dimensions. A stored position matrix must have
    /// one block per `hamR` row, as required by [`Model::validate`]; pad absent
    /// position blocks with zeros.
    pub fn gen_v_projected_batch<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        gauge: Gauge,
        directions: &Array2<f64>,
    ) -> (Array4<Complex<f64>>, Array3<Complex<f64>>) {
        assert_eq!(
            directions.ncols(),
            DIM,
            "direction dimension must match the model"
        );
        let phases = self.bloch_phases(points);
        let nk = points.nrows();
        let nr = self.hamR.nrows();
        let nsta = self.nsta();
        let nproj = directions.nrows();
        if R::HAS_RMATRIX {
            assert_eq!(self.rmatrix.as_array4().dim(), (nr, DIM, nsta, nsta));
        }
        let lattice_directions = self.lat.dot(&directions.t());
        let r_projected = Array2::from_shape_fn((nr, nproj), |(ir, p)| {
            (0..DIM)
                .map(|d| self.hamR[[ir, d]] as f64 * lattice_directions[[d, p]])
                .sum::<f64>()
        });
        // Each k-point contributes one H row and nproj derivative rows.
        // Even the single-k velocity API can reuse hopping across directions.
        let rows = nk.checked_mul(nproj.checked_add(1).unwrap()).unwrap();
        let coefficients = Array2::from_shape_fn((rows, nr), |(row, ir)| {
            let ik = row / (nproj + 1);
            let channel = row % (nproj + 1);
            if channel == 0 {
                phases[[ik, ir]]
            } else {
                phases[[ik, ir]] * Complex::new(0.0, r_projected[[ir, channel - 1]])
            }
        });
        let summed = fourier_sum(&coefficients.view(), &self.ham)
            .into_shape_with_order((nk, nproj + 1, nsta, nsta))
            .unwrap();
        let mut hams = summed.index_axis(Axis(1), 0).to_owned();
        // Keep velocity blocks in the GEMM allocation. Only H needs copying;
        // each k-point's velocity block remains contiguous.
        let mut velocities = summed.slice_move(s![.., 1.., .., ..]);
        drop(coefficients);

        // Position and hopping blocks share the same translation support.
        let connections = if R::HAS_RMATRIX && nproj > 0 {
            let position = self.rmatrix.as_array4();
            Some(
                fourier_sum(&phases.view(), position)
                    .into_shape_with_order((nk, DIM, nsta, nsta))
                    .unwrap(),
            )
        } else {
            None
        };
        let tau_projected = if matches!(gauge, Gauge::Atom) {
            Some(self.orb.dot(&lattice_directions))
        } else {
            None
        };
        for ik in 0..nk {
            let mut ham = hams.index_axis_mut(Axis(0), ik);
            let mut velocity = velocities.index_axis_mut(Axis(0), ik);
            let orbital_phases = tau_projected
                .as_ref()
                .map(|_| self.orbital_phases(&points.row(ik)));
            if let (Some(tau), Some(orbital_phases)) = (&tau_projected, &orbital_phases) {
                for (p, mut component) in velocity.outer_iter_mut().enumerate() {
                    for ((i, j), value) in component.indexed_iter_mut() {
                        let displacement = tau[[j % self.norb(), p]] - tau[[i % self.norb(), p]];
                        *value += Complex::new(0.0, displacement) * ham[[i, j]];
                    }
                    apply_atom_gauge(component, orbital_phases);
                }
                apply_atom_gauge(ham.view_mut(), orbital_phases);
            }
            if let Some(connections) = &connections {
                let connection = connections.index_axis(Axis(0), ik);
                for (p, mut component) in velocity.outer_iter_mut().enumerate() {
                    let mut projected = Array2::<Complex<f64>>::zeros((nsta, nsta));
                    for (d, &weight) in directions.row(p).iter().enumerate() {
                        if weight != 0.0 {
                            projected.scaled_add(
                                Complex::new(weight, 0.0),
                                &connection.index_axis(Axis(0), d),
                            );
                        }
                    }
                    if let Some(orbital_phases) = &orbital_phases {
                        apply_atom_gauge(projected.view_mut(), orbital_phases);
                        // Preserve the existing atom-gauge position convention.
                        projected.diag_mut().fill(Complex::new(0.0, 0.0));
                    }
                    component.scaled_add(Complex::i(), &comm(&ham, &projected));
                }
            }
        }
        (velocities, hams)
    }
}
