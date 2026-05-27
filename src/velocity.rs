//! Velocity operator $\mathbf{v}(\mathbf{k}) = \nabla_{\mathbf{k}} H(\mathbf{k})$ for tight-binding models.
//!
//! Provides the [`Velocity`] trait and its implementation for [`Model`],
//! computing matrix elements $\bra{m\mathbf{k}} \partial_\alpha H_{\mathbf{k}} \ket{n\mathbf{k}}$
//! at a given k-point. Essential for Berry curvature, optical conductivity,
//! and other transport calculations.
//!
//! # Physical background
//!
//! ## Bloch basis and velocity operator
//!
//! The Bloch eigenstates satisfy
//! $H \ket{\psi_{n\mathbf{k}}} = \varepsilon_{n\mathbf{k}} \ket{\psi_{n\mathbf{k}}}$.
//! Define the periodic part $\ket{u_{n\mathbf{k}}} = e^{-i\mathbf{k}\cdot\mathbf{r}} \ket{\psi_{n\mathbf{k}}}$,
//! which obeys $H_{\mathbf{k}} \ket{u_{n\mathbf{k}}} = \varepsilon_{n\mathbf{k}} \ket{u_{n\mathbf{k}}}$
//! with $H_{\mathbf{k}} = e^{-i\mathbf{k}\cdot\mathbf{r}} H e^{+i\mathbf{k}\cdot\mathbf{r}}$.
//! The periodic functions are orthonormal within the unit cell $\Omega$:
//! $\braket{u_{m\mathbf{k}}}{u_{n\mathbf{k}}} = \int_\Omega d\mathbf{r}\, u_{n\mathbf{k}}^*(\mathbf{r}) u_{m\mathbf{k}}(\mathbf{r}) = \delta_{mn}$.
//!
//! From the Heisenberg equation, $\mathbf{v} = \dot{\mathbf{r}} = \frac{i}{\hbar}[H, \mathbf{r}]$.
//! Transforming to the Bloch basis:
//!
//! $$
//! \mathbf{v}_{\mathbf{k}} \equiv e^{-i\mathbf{k}\cdot\mathbf{r}} \mathbf{v} e^{+i\mathbf{k}\cdot\mathbf{r}}
//! = \frac{1}{\hbar} \nabla_{\mathbf{k}} H_{\mathbf{k}} .
//! $$
//!
//! The velocity operator matrix in the band eigenbasis is therefore
//!
//! $$
//! \bra{\psi_{m\mathbf{k}}} \mathbf{v} \ket{\psi_{n\mathbf{k}}}
//! = \frac{1}{\hbar} \bra{u_{m\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{u_{n\mathbf{k}}} .
//! $$
//!
//! **Note**: In the code, $\hbar$ is absorbed into the unit system (effectively $\hbar = 1$),
//! so `gen_v` returns $\nabla_{\mathbf{k}} H_{\mathbf{k}}$ rather than $\frac{1}{\hbar}\nabla_{\mathbf{k}} H_{\mathbf{k}}$.
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
//! $$
//! \ket{\psi^W_{\alpha\mathbf{k}}} = \frac{1}{\sqrt{N}} \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} \ket{\alpha\mathbf{R}},
//! \qquad
//! \ket{e^W_{\alpha\mathbf{k}}} = e^{-i\mathbf{k}\cdot\hat{\mathbf{r}}} \ket{\psi^W_{\alpha\mathbf{k}}} .
//! $$
//!
//! **Atom gauge** (includes orbital positions $\boldsymbol{\tau}_\alpha$):
//! $$
//! \ket{\alpha\mathbf{k}} = \frac{1}{\sqrt{N}} \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot(\mathbf{R} + \boldsymbol{\tau}_\alpha)} \ket{\alpha\mathbf{R}},
//! \qquad
//! \ket{e_{\alpha\mathbf{k}}} = e^{-i\mathbf{k}\cdot\hat{\mathbf{r}}} \ket{\alpha\mathbf{k}} .
//! $$
//!
//! Both bases satisfy $\ket{\alpha\mathbf{k} + \mathbf{G}} = \ket{\alpha\mathbf{k}}$ for reciprocal lattice
//! vectors $\mathbf{G}$, and are orthonormal: $\braket{e_{\alpha\mathbf{k}}}{e_{\beta\mathbf{k}}} = \delta_{\alpha\beta}$.
//! The two are related by $\ket{\alpha\mathbf{k}} = e^{i\mathbf{k}\cdot\boldsymbol{\tau}_\alpha} \ket{\psi^W_{\alpha\mathbf{k}}}$.
//!
//! ### Lattice gauge velocity formula
//!
//! In the Lattice gauge, the Hamiltonian and its derivative are:
//!
//! $$
//! \begin{aligned}
//! H^W_{\alpha\beta}(\mathbf{k}) &\equiv \bra{\psi^W_{\alpha\mathbf{k}}} H \ket{\psi^W_{\beta\mathbf{k}}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot\mathbf{R}}, \\[6pt]
//! \nabla_{\mathbf{k}} H^W_{\alpha\beta}(\mathbf{k}) &= i \sum_{\mathbf{R}} \mathbf{R} \, \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot\mathbf{R}} .
//! \end{aligned}
//! $$
//!
//! The Berry connection (position matrix) in the Lattice gauge is directly available from Wannier90
//! (`seedname_r.dat`):
//!
//! $$
//! A^W_{\alpha\beta}(\mathbf{k}) \equiv i \bra{e^W_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} \ket{e^W_{\beta\mathbf{k}}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} \hat{\mathbf{r}} \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot\mathbf{R}} .
//! $$
//!
//! Using $\nabla_{\mathbf{k}} H^W = \bra{e^W} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{e^W}
//! + \bra{\nabla_{\mathbf{k}} e^W} H_{\mathbf{k}} \ket{e^W} + \bra{e^W} H_{\mathbf{k}} \ket{\nabla_{\mathbf{k}} e^W}$
//! and inserting $\mathbb{1} = \sum_\gamma \ketbra{e^W_{\gamma\mathbf{k}}}$, one obtains:
//!
//! $$
//! \bra{e^W_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{e^W_{\beta\mathbf{k}}}
//! = \nabla_{\mathbf{k}} H^W_{\alpha\beta}(\mathbf{k}) + i\bigl[ H^W(\mathbf{k}), A^W(\mathbf{k}) \bigr]_{\alpha\beta} .
//! $$
//!
//! Transforming to the band eigenbasis $\ket{u_{n\mathbf{k}}} = \sum_\alpha \ket{e^W_{\alpha\mathbf{k}}} C_{\alpha n}(\mathbf{k})$
//! yields the lattice-gauge velocity:
//!
//! $$
//! {\color{red}\boxed{\color{black}
//! \bra{u_{m\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{u_{n\mathbf{k}}}
//! = \bra{u_{m\mathbf{k}}} \Bigl( \nabla_{\mathbf{k}} H^W(\mathbf{k}) + i\bigl[ H^W(\mathbf{k}), A^W(\mathbf{k}) \bigr] \Bigr) \ket{u_{n\mathbf{k}}} }} .
//! $$
//!
//! ### Atom gauge velocity formula
//!
//! In the Atom gauge, the Hamiltonian includes $\boldsymbol{\tau}$ phases:
//!
//! $$
//! \bar{H}_{\alpha\beta}(\mathbf{k}) \equiv \bra{\alpha\mathbf{k}} H \ket{\beta\mathbf{k}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta)} .
//! $$
//!
//! Its derivative splits into an $\mathbf{R}$-term and a $\boldsymbol{\tau}$-term:
//!
//! $$
//! \nabla_{\mathbf{k}} \bar{H}_{\alpha\beta}(\mathbf{k})
//! = i \sum_{\mathbf{R}} \bigl(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta\bigr)
//!   \bra{\alpha\mathbf{0}} H \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta)} .
//! $$
//!
//! The Berry connection in the Atom gauge is
//!
//! $$
//! \bar{\mathbf{r}}_{\alpha\beta}(\mathbf{k}) \equiv i \bra{e_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} \ket{e_{\beta\mathbf{k}}}
//! = \sum_{\mathbf{R}} \bra{\alpha\mathbf{0}} \hat{\mathbf{r}} \ket{\beta\mathbf{R}} \, e^{i\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_\alpha + \boldsymbol{\tau}_\beta)}
//!   - \boldsymbol{\tau}_\alpha \delta_{\alpha\beta} .
//! $$
//!
//! Following the same derivation as for the Lattice gauge,
//! $\bra{e_{\alpha\mathbf{k}}} \nabla_{\mathbf{k}} H_{\mathbf{k}} \ket{e_{\beta\mathbf{k}}}
//! = \nabla_{\mathbf{k}} \bar{H}_{\alpha\beta}(\mathbf{k}) + i\bigl[ \bar{H}(\mathbf{k}), \bar{\mathbf{r}} \bigr]_{\alpha\beta}$.
//!
//! A useful identity: the $\boldsymbol{\tau}$-dependent diagonal part of $\bar{\mathbf{r}}$ generates,
//! through the commutator $i[\bar{H}, \bar{\mathbf{r}}]$, a term $-i(\tau_n - \tau_m) \bar{H}_{mn}$
//! that **exactly cancels** the $\boldsymbol{\tau}$-difference piece in $\nabla_{\mathbf{k}} \bar{H}$.
//! Therefore one may drop the $-\boldsymbol{\tau}_\alpha \delta_{\alpha\beta}$ from $\bar{\mathbf{r}}$
//! (i.e. set its diagonal to zero) *provided* the $i(\boldsymbol{\tau}_n - \boldsymbol{\tau}_m) \bar{H}_{mn}$ term
//! is also omitted from $\nabla_{\mathbf{k}} \bar{H}$. This is how the code is implemented.
//!
//! The net Atom-gauge velocity, in component form (direction $\alpha$, Cartesian coordinates), is:
//!
//! $$
//! {\color{red}\boxed{\color{black}
//! \begin{aligned}
//! \bra{m\mathbf{k}} \partial_\alpha H_{\mathbf{k}} \ket{n\mathbf{k}}
//! &= \sum_{\mathbf{R}} i R_\alpha^{\rm (cart)} H_{mn}(\mathbf{R})\, e^{2\pi i\,\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_m + \boldsymbol{\tau}_n)} \\[4pt]
//! &+ i\bigl(\tau_{n\alpha}^{\rm (cart)} - \tau_{m\alpha}^{\rm (cart)}\bigr)\, H_{mn}(\mathbf{k}) \\[4pt]
//! &- \bigl[ H(\mathbf{k}), \mathcal{A}_{\mathbf{k},\alpha} \bigr]_{mn}
//! \end{aligned}}}
//! $$
//!
//! where the Berry connection matrix is:
//!
//! $$
//! \mathcal{A}_{\mathbf{k},\alpha,mn} = -i \sum_{\mathbf{R}} r_{mn,\alpha}(\mathbf{R})\, e^{2\pi i\,\mathbf{k}\cdot(\mathbf{R} - \boldsymbol{\tau}_m + \boldsymbol{\tau}_n)}
//! + i \tau_{n\alpha} \delta_{mn} .
//! $$
//!
//! The position matrix elements $\mathbf{r}_{mn}(\mathbf{R})$ are provided by Wannier90
//! (setting `write_rmn = true`). If unavailable, the commutator term $[H(\mathbf{k}), \mathcal{A}_{\mathbf{k},\alpha}]$
//! is omitted.
//!
//! # Implementation notes
//!
//! The code first constructs $H(\mathbf{k}) = \sum_{\mathbf{R}} H(\mathbf{R}) e^{2\pi i \mathbf{k}\cdot\mathbf{R}}$
//! (Lattice gauge). For [`Gauge::Atom`], it then:
//!
//! 1. Computes the $\mathbf{R}$-term: $i R_\alpha H(\mathbf{R}) e^{2\pi i\mathbf{k}\cdot\mathbf{R}}$,
//!    then applies the gauge transform $e^{2\pi i\mathbf{k}\cdot(\boldsymbol{\tau}_n - \boldsymbol{\tau}_m)}$
//!    to convert Lattice-gauge phases to Atom-gauge phases ($\mathbf{R} \to \mathbf{R} - \boldsymbol{\tau}_m + \boldsymbol{\tau}_n$).
//! 2. Adds the $\boldsymbol{\tau}$-difference term $i(\tau_{n\alpha} - \tau_{m\alpha}) H_{mn}(\mathbf{k})$.
//! 3. Computes the Berry connection $A^W$ in the Lattice gauge, applies the same gauge transform,
//!    sets the diagonal to zero (exploiting the $\boldsymbol{\tau}$-cancellation identity),
//!    and adds $-i[H(\mathbf{k}), A]$.
//!
//! For [`Gauge::Lattice`], the $\boldsymbol{\tau}$-dependent steps are skipped.
//!
//! # Conventions
//!
//! - **k**: fractional reciprocal coordinates; the phase factor is $e^{2\pi i\,\mathbf{k}\cdot\mathbf{R}}$
//! - **R**: integer lattice vectors from `hamR`
//! - $R_\alpha^{\rm (cart)}$, $\tau_{n\alpha}^{\rm (cart)}$: Cartesian coordinates (in Å),
//!   obtained by multiplying fractional vectors with the lattice matrix `lat`
//! - The returned velocity matrix is **anti-Hermitian**: $v_\alpha^\dagger = -v_\alpha$
use crate::Gauge;
use crate::Model;
use crate::comm;
use crate::solve_ham::*;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::ops::AddAssign;

#[cfg_attr(doc, katexit::katexit)]
/// Trait for computing the velocity operator $\mathbf{v}(\mathbf{k})$.
///
/// The velocity operator is defined as the k-derivative of the Bloch Hamiltonian:
///
/// $$\mathbf{v}(\mathbf{k}) = \nabla_{\mathbf{k}} H(\mathbf{k})$$
///
/// Note this is the **full velocity operator matrix** in the Bloch basis,
/// not just the band-diagonal group velocity $\partial E_n/\partial\mathbf{k}$.
///
/// # Returns
///
/// `(v, hamk)` where:
/// - `v` is a $d \times N_{\rm sta} \times N_{\rm sta}$ array giving
///   $v_{\alpha,mn}$ for each direction $\alpha$
/// - `hamk` is the $N_{\rm sta} \times N_{\rm sta}$ Bloch Hamiltonian $H(\mathbf{k})$
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

impl<const SPIN: bool, const DIM: usize> Velocity for Model<SPIN, DIM> {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn gen_v<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length {} must equal to the dimension of model {}.",
            kvec.len(),
            self.dim_r()
        );

        let dim = self.dim_r();
        let nsta = self.nsta();

        // Phase factors exp(i 2π k·R): dimension-dispatched for loop unrolling.
        // Cached since each R's phase is reused in hamk, velocity (dim dirs), and rmatrix.
        let Us: Vec<Complex<f64>> = match DIM {
            1 => self
                .hamR
                .outer_iter()
                .map(|r| Complex::new(0.0, 2.0 * PI * r[0] as f64 * kvec[0]).exp())
                .collect(),
            2 => self
                .hamR
                .outer_iter()
                .map(|r| {
                    Complex::new(
                        0.0,
                        2.0 * PI * (r[0] as f64 * kvec[0] + r[1] as f64 * kvec[1]),
                    )
                    .exp()
                })
                .collect(),
            3 => self
                .hamR
                .outer_iter()
                .map(|r| {
                    Complex::new(
                        0.0,
                        2.0 * PI
                            * (r[0] as f64 * kvec[0]
                                + r[1] as f64 * kvec[1]
                                + r[2] as f64 * kvec[2]),
                    )
                    .exp()
                })
                .collect(),
            _ => unreachable!(),
        };

        // R in Cartesian: f64 matmul avoids Complex conversion + uses faster DGEMM
        let hamR_f64 = self.hamR.mapv(|x| x as f64);
        let R0: Array2<f64> = hamR_f64.dot(&self.lat);

        let mut v = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));

        // Build H(k) = Σ_R H(R) exp(i 2π k·R)
        let mut hamk = Array2::<Complex<f64>>::zeros((nsta, nsta));
        let hamk_slice = hamk.as_slice_mut().unwrap();
        for (iR, &u) in Us.iter().enumerate() {
            let hm = self.ham.index_axis(Axis(0), iR);
            crate::ndarray_lapack::zaxpy(u, hm.as_slice().unwrap(), hamk_slice);
        }
        let (v, hamk) = match gauge {
            Gauge::Atom => {
                let orb_sta = if SPIN {
                    let orb0 = concatenate(Axis(0), &[self.orb.view(), self.orb.view()]).unwrap();
                    orb0
                } else {
                    self.orb.to_owned()
                };
                // Dimension-dispatched τ·k phase factors
                let orb_phase: Vec<Complex<f64>> = match DIM {
                    1 => orb_sta
                        .outer_iter()
                        .map(|tau| Complex::new(0.0, 2.0 * PI * tau[0] * kvec[0]).exp())
                        .collect(),
                    2 => orb_sta
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(0.0, 2.0 * PI * (tau[0] * kvec[0] + tau[1] * kvec[1]))
                                .exp()
                        })
                        .collect(),
                    3 => orb_sta
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(
                                0.0,
                                2.0 * PI * (tau[0] * kvec[0] + tau[1] * kvec[1] + tau[2] * kvec[2]),
                            )
                            .exp()
                        })
                        .collect(),
                    _ => unreachable!(),
                };
                let orb_real = orb_sta.dot(&self.lat);
                // UU[d,m,n] = i*(tau[n,d] - tau[m,d])
                let A = orb_real.view().insert_axis(Axis(2));
                let A = A
                    .broadcast((self.nsta(), self.dim_r(), self.nsta()))
                    .unwrap()
                    .permuted_axes([1, 0, 2]);
                let B = A.view().permuted_axes([0, 2, 1]);
                let UU = (&B - &A).mapv(|x| Complex::<f64>::new(0.0, x));
                // Velocity per direction: Zip::for_each auto-vectorized accumulation
                // azip! merges the hamk*UU[d] term in-place
                for d in 0..dim {
                    let mut vv = Array2::<Complex<f64>>::zeros((nsta, nsta));
                    let R0_d = R0.column(d);
                    for (iR, &u) in Us.iter().enumerate() {
                        let hm = self.ham.index_axis(Axis(0), iR);
                        let alpha = u * R0_d[iR] * Complex::i();
                        crate::ndarray_lapack::zaxpy(alpha, hm.as_slice().unwrap(), vv.as_slice_mut().unwrap());
                    }
                    azip!((v in &mut vv, &h in &hamk, &u in &UU.slice(s![d, .., ..])) *v += h * u);
                    // Gauge transform: for m + Zip, no allocation
                    for m in 0..nsta {
                        let mut row = vv.slice_mut(s![m, ..]);
                        let conj_pm = orb_phase[m].conj();
                        Zip::from(&mut row)
                            .and(orb_phase.as_slice())
                            .for_each(|h, &pn| *h *= conj_pm * pn);
                    }
                    v.slice_mut(s![d, .., ..]).assign(&vv);
                }
                // At this point, we have computed sum_{R} iR H_{mn}(R) e^{ik(R+tau_n-tau_m)}
                // Next, compute Berry connection A_\alpha=\sum_R r(R)e^{ik(R+tau_n-tau_m)}-tau
                for m in 0..nsta {
                    let mut row = hamk.slice_mut(s![m, ..]);
                    let conj_pm = orb_phase[m].conj();
                    Zip::from(&mut row)
                        .and(orb_phase.as_slice())
                        .for_each(|h, &pn| *h *= conj_pm * pn);
                }
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let n_rmat = self.rmatrix.len_of(Axis(0));
                    let mut rk = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));
                    for (iR, &u) in Us[..n_rmat].iter().enumerate() {
                        let rm = self.rmatrix.index_axis(Axis(0), iR);
                        crate::ndarray_lapack::zaxpy(u, rm.as_slice().unwrap(), rk.as_slice_mut().unwrap());
                    }
                    for i in 0..dim {
                        let mut r0 = rk.slice_mut(s![i, .., ..]);
                        // Gauge transform: for m + Zip, no allocation
                        for m in 0..nsta {
                            let mut row = r0.slice_mut(s![m, ..]);
                            let conj_pm = orb_phase[m].conj();
                            Zip::from(&mut row)
                                .and(orb_phase.as_slice())
                                .for_each(|h, &pn| *h *= conj_pm * pn);
                        }
                        r0.diag_mut().assign(&Array1::zeros(nsta));
                        let a_comm = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&a_comm);
                    }
                }
                (v, hamk)
            }
            Gauge::Lattice => {
                for d in 0..dim {
                    let mut vv = Array2::<Complex<f64>>::zeros((nsta, nsta));
                    let R0_d = R0.column(d);
                    for (iR, &u) in Us.iter().enumerate() {
                        let hm = self.ham.index_axis(Axis(0), iR);
                        let alpha = u * R0_d[iR] * Complex::i();
                        crate::ndarray_lapack::zaxpy(alpha, hm.as_slice().unwrap(), vv.as_slice_mut().unwrap());
                    }
                    v.slice_mut(s![d, .., ..]).assign(&vv);
                }
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let n_rmat = self.rmatrix.len_of(Axis(0));
                    let mut rk = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));
                    for (iR, &u) in Us[..n_rmat].iter().enumerate() {
                        let rm = self.rmatrix.index_axis(Axis(0), iR);
                        crate::ndarray_lapack::zaxpy(u, rm.as_slice().unwrap(), rk.as_slice_mut().unwrap());
                    }
                    for i in 0..dim {
                        let r0 = rk.slice(s![i, .., ..]);
                        let a_comm = comm(&hamk, &r0) * Complex::i();
                        v.slice_mut(s![i, .., ..]).add_assign(&a_comm);
                    }
                }
                (v, hamk)
            }
        };
        (v, hamk)
    }
}
