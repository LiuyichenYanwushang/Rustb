//! Velocity operator $\mathbf{v}(\mathbf{k}) = \nabla_{\mathbf{k}} H(\mathbf{k})$ for tight-binding models.
//!
//! Provides the [`Velocity`] trait and its implementation for [`Model`],
//! computing matrix elements $\bra{m\mathbf{k}} \partial_\alpha H_{\mathbf{k}} \ket{n\mathbf{k}}$
//! at a given k-point. Essential for Berry curvature, optical conductivity,
//! and other transport calculations.
//!
//! # Formula (Atomic gauge)
//!
//! $$\begin{aligned}
//! \bra{m\mathbf{k}} \partial_\alpha H_{\mathbf{k}} \ket{n\mathbf{k}}
//! &= \sum_{\mathbf{R}} i R_\alpha^{\rm (cart)} H_{mn}(\mathbf{R})\, e^{2\pi i\,\mathbf{k}\cdot(\mathbf{R} - \bm{\tau}_m + \bm{\tau}_n)} \\
//! &+ i(\tau_{n\alpha}^{\rm (cart)} - \tau_{m\alpha}^{\rm (cart)})\, H_{mn}(\mathbf{k}) \\
//! &- [H(\mathbf{k}), \mathcal{A}_{\mathbf{k},\alpha}]_{mn}
//! \end{aligned}$$
//!
//! where $\mathcal{A}_{\mathbf{k}}$ is the Berry connection matrix:
//!
//! $$\mathcal{A}_{\mathbf{k},\alpha,mn} = -i\sum_{\mathbf{R}} r_{mn,\alpha}(\mathbf{R})\, e^{2\pi i\,\mathbf{k}\cdot(\mathbf{R} - \bm{\tau}_m + \bm{\tau}_n)} + i\tau_{n\alpha}\delta_{mn}$$
//!
//! The position matrix elements $\mathbf{r}_{mn}(\mathbf{R})$ can be provided by
//! Wannier90 (setting `write_rmn = true`). If unavailable, the rmatrix commutator
//! term is omitted.
//!
//! # Conventions
//!
//! - **k**: fractional reciprocal coordinates, phase factor uses $2\pi$
//! - **R**: integer lattice vectors from `hamR`
//! - $R_\alpha^{\rm (cart)}$, $\tau_{n\alpha}^{\rm (cart)}$: Cartesian coordinates
//!   obtained by multiplying fractional vectors with the lattice matrix `lat`
//! - The returned velocity matrix is **anti-Hermitian**: $v_\alpha^\dagger = -v_\alpha$
use crate::Dimension;
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

impl<const SPIN: bool> Velocity for Model<SPIN> {
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
        let Us: Vec<Complex<f64>> = match self.dim_r {
            Dimension::one => self
                .hamR
                .outer_iter()
                .map(|r| Complex::new(0.0, 2.0 * PI * r[0] as f64 * kvec[0]).exp())
                .collect(),
            Dimension::two => self
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
            Dimension::three => self
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
        };

        // R in Cartesian: f64 matmul avoids Complex conversion + uses faster DGEMM
        let hamR_f64 = self.hamR.mapv(|x| x as f64);
        let R0: Array2<f64> = hamR_f64.dot(&self.lat);

        let mut v = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));

        // Build H(k) = Σ_R H(R) exp(i 2π k·R)
        let mut hamk = Array2::<Complex<f64>>::zeros((nsta, nsta));
        Zip::from(self.ham.outer_iter())
            .and(&Us)
            .for_each(|hm, &u| hamk.scaled_add(u, &hm));
        let (v, hamk) = match gauge {
            Gauge::Atom => {
                let orb_sta = if SPIN {
                    let orb0 = concatenate(Axis(0), &[self.orb.view(), self.orb.view()]).unwrap();
                    orb0
                } else {
                    self.orb.to_owned()
                };
                // Dimension-dispatched τ·k phase factors
                let orb_phase: Vec<Complex<f64>> = match self.dim_r {
                    Dimension::one => orb_sta
                        .outer_iter()
                        .map(|tau| Complex::new(0.0, 2.0 * PI * tau[0] * kvec[0]).exp())
                        .collect(),
                    Dimension::two => orb_sta
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(
                                0.0,
                                2.0 * PI * (tau[0] * kvec[0] + tau[1] * kvec[1]),
                            )
                            .exp()
                        })
                        .collect(),
                    Dimension::three => orb_sta
                        .outer_iter()
                        .map(|tau| {
                            Complex::new(
                                0.0,
                                2.0 * PI
                                    * (tau[0] * kvec[0]
                                        + tau[1] * kvec[1]
                                        + tau[2] * kvec[2]),
                            )
                            .exp()
                        })
                        .collect(),
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
                // Velocity per direction: scaled_add avoids inner Zip allocation,
                // azip! merges the hamk*UU[d] term in-place
                for d in 0..dim {
                    let mut vv = Array2::<Complex<f64>>::zeros((nsta, nsta));
                    let R0_d = R0.column(d);
                    Zip::from(self.ham.outer_iter())
                        .and(&Us)
                        .and(&R0_d)
                        .for_each(|hm, &u, &r0_d| {
                            vv.scaled_add(u * r0_d * Complex::i(), &hm);
                        });
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
                    Zip::from(self.rmatrix.outer_iter())
                        .and(&Us[..n_rmat])
                        .for_each(|rm, &u| {
                            Zip::from(&mut rk).and(&rm).for_each(|a, &b| *a += b * u);
                        });
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
                    Zip::from(self.ham.outer_iter())
                        .and(&Us)
                        .and(&R0_d)
                        .for_each(|hm, &u, &r0_d| {
                            vv.scaled_add(u * r0_d * Complex::i(), &hm);
                        });
                    v.slice_mut(s![d, .., ..]).assign(&vv);
                }
                if self.rmatrix.len_of(Axis(0)) != 1 {
                    let n_rmat = self.rmatrix.len_of(Axis(0));
                    let mut rk = Array3::<Complex<f64>>::zeros((dim, nsta, nsta));
                    Zip::from(self.rmatrix.outer_iter())
                        .and(&Us[..n_rmat])
                        .for_each(|rm, &u| {
                            Zip::from(&mut rk).and(&rm).for_each(|a, &b| *a += b * u);
                        });
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
