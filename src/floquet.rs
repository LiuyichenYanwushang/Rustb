//! Real-space Peierls-Floquet utilities.
//!
//! The implementation works directly with the real-space hopping blocks stored
//! in [`Model::ham`] and [`Model::hamR`].  A spatially uniform light field is
//! introduced through a Peierls phase on every hopping link, then Fourier
//! transformed into a commensurate Sambe Hamiltonian.
//!
//! # Physical scope
//!
//! This module implements the **long-wavelength Peierls coupling** of a
//! periodic tight-binding model to a classical, spatially uniform light field.
//! It is appropriate when the optical wavelength is much larger than the unit
//! cell and the dominant coupling is through hopping phases.  In this first
//! implementation the light field must be commensurate with one base frequency
//! `omega0_ev`; arbitrary mixtures of integer harmonics of that base frequency
//! are supported.
//!
//! The current implementation does **not** add the length-gauge dipole term
//! `-e E(t) r` from Wannier90 `rmatrix` data.  That should be added later as a
//! separate coupling option rather than mixed silently with Peierls phases.
//!
//! # Real-space hopping convention
//!
//! Rustb stores hopping blocks as
//!
//! ```math
//! t_{ij}(\mathbf R) = \langle i,\mathbf 0|\hat H|j,\mathbf R\rangle .
//! ```
//!
//! where `hamR[a]` is the integer lattice vector `R` and `ham[a,i,j]` is the
//! corresponding matrix element.  The real-space link vector used by the Peierls
//! phase is
//!
//! ```math
//! \mathbf d_{ij\mathbf R} =
//! \bigl(\mathbf R+\boldsymbol\tau_j-\boldsymbol\tau_i\bigr)L .
//! ```
//!
//! Here `orb` stores fractional orbital coordinates `tau`, and `lat` is the
//! real-space lattice matrix used with row-vector fractional coordinates:
//! `cart = frac.dot(lat)`.  For spinful models the spin label is ignored in the
//! link geometry; state indices are mapped to orbital indices by `state % norb`.
//!
//! # Light-field convention
//!
//! The drive is represented by
//!
//! ```math
//! \mathbf a(t) = \frac{e}{\hbar}\mathbf A(t),
//! ```
//!
//! so `LightMode::a_complex` has units of inverse length, matching the length
//! unit of `lat`.  For one mode with harmonic `l`, the stored complex amplitude
//! means
//!
//! ```math
//! \mathbf a_l(t) =
//! \operatorname{Re}\left[
//! \mathbf a_l e^{-i l\Omega_0 t}
//! \right].
//! ```
//!
//! Multiple [`LightMode`] values are added before exponentiating:
//!
//! ```math
//! \mathbf a(t) =
//! \operatorname{Re}\sum_\alpha
//! \mathbf a_\alpha e^{-i l_\alpha\Omega_0 t}.
//! ```
//!
//! This representation covers linear, circular, elliptical, and mixed-harmonic
//! polarization without hard-coded special cases.
//!
//! # Peierls phase and Fourier blocks
//!
//! Every hopping is dressed as
//!
//! ```math
//! t_{ij}(\mathbf R,t) =
//! t_{ij}(\mathbf R)
//! \exp\left[-i\,\mathbf a(t)\cdot\mathbf d_{ij\mathbf R}\right].
//! ```
//!
//! The Fourier coefficient of the Peierls phase is
//!
//! ```math
//! C_n(\mathbf d) =
//! \frac{1}{T}\int_0^T dt\,
//! e^{in\Omega_0 t}
//! \exp\left[-i\,\mathbf a(t)\cdot\mathbf d\right].
//! ```
//!
//! Both Floquet paths evaluate `C_n` with the same backend: the generalized
//! Bessel expansion per link, which is exact, needs no sampling count, and
//! handles arbitrary complex polarization and arbitrary commensurate harmonic
//! mixing.  Three shapes cover a general drive: one nonzero temporal harmonic
//! collapses to a single Bessel term per requested order; two carriers enumerate
//! the resonance sum directly, because the constraint `l₁m₁ + l₂m₂ = −n` leaves
//! one free index; and three or more fold one-mode convolutions.  Links beyond
//! the backend's exact range fall back to a per-link time-grid DFT.  The uniform-time-grid
//! reference the tests cross-validate against lives in the test-only
//! `crate::floquet_test` module, so no entry point depends on it.
//!
//! # Cost of one link
//!
//! With `R_α = |a_α·d|`, `N_mode` the number of drive modes, `K` the requested
//! harmonic range, `L_α` the temporal harmonic of mode `α`, and `M_α` the
//! adaptive order cutoff of mode `α` — at least `⌈R_α⌉ + cutoff_margin`, raised
//! until the two-sided Bessel tail `2·Σ_{m>M_α}|J_m|` fits a `1e-12/N_mode`
//! share, so `M_α ≈ R_α`:
//!
//! | Drive shape | Cost | Amplitude cap |
//! |---|---|---|
//! | one nonzero harmonic | `O(R + K)`: one ladder, then table lookups | `MAX_BESSEL_ARG_CLOSED_FORM` |
//! | two carriers | `O(K·(2·min(M₁,M₂)+1))`, no working window | `MAX_BESSEL_ARG_CLOSED_FORM`, and `MAX_BESSEL_ENUM_WORK` iterations |
//! | three or more carriers | `O(Σ_α(2M_α+1)·W)` with window `W = K + 2·Σ_α|L_α|M_α` | `MAX_BESSEL_ARG` per operand, `MAX_BESSEL_WINDOW` window |
//! | beyond a cap | per-link time-grid DFT, `O(N·(N_carrier + K))` — it projects the AC field onto the link, folds DC into one phase, and evaluates one complex exponential per coherent carrier (a distinct `|L|` with nonzero projected amplitude) and sample | `N ≤ FALLBACK_GRID_MAX`, else the call is refused |
//!
//! Every `M_α` sweep comes from one backward recurrence whose length is
//! `max(⌈R⌉, requested) + O(√R)`; the ladder evaluates no order in isolation,
//! so the whole `J_0..J_M` range and every truncation tail cost one sweep and
//! one suffix sum.
//!
//! Because the closed form is linear in `R` instead of in `R·K`, it reaches two
//! orders of magnitude further than the convolution.  Measured per link with the
//! photon cutoff `n_max = 2`, the removed single-k Sambe path took R = 128 → 15 µs,
//! R = 4000 → 72 µs and R = 16000 → 286 µs, against 2121 µs and 8357 µs for the
//! same links through the old Cartesian fallback grid. Those fallback timings
//! are historical, not a benchmark of the current projected-carrier sampler. Two
//! carriers cost 9-15 µs per link at R ≤ 400, where the fold they replace costs
//! ~0.9 ms; three or more carriers still fold, which is the remaining slow path.
//!
//! The reciprocal-space Fourier block is
//!
//! ```math
//! H^{(n)}_{ij}(\mathbf k) =
//! \sum_{\mathbf R}
//! t_{ij}(\mathbf R)\,
//! C_n(\mathbf d_{ij\mathbf R})\,
//! e^{i2\pi\mathbf k\cdot\mathbf R}.
//! ```
//!
//! `Gauge::Lattice` returns this block directly.  `Gauge::Atom` applies the
//! same orbital-position phase convention as [`Model::gen_ham`].
//!
//! # Sambe Hamiltonian
//!
//! With photon sectors `n,m` in `[-n_max, n_max]`, the Floquet-Sambe
//! Hamiltonian is
//!
//! ```math
//! \left[H_F(\mathbf k)\right]_{i n,j m} =
//! H^{(n-m)}_{ij}(\mathbf k)
//! +
//! n\Omega_0\,\delta_{nm}\delta_{ij}.
//! ```
//!
//! The photon energy `Omega_0` is stored as `FloquetDrive::omega0_ev` in eV, so
//! the returned Floquet eigenvalues are also in eV.
//!
//! Build the real-space Sambe model with [`Floquet::floquet_model`], then use
//! [`crate::Solve::solve_band_onek`] or the ordinary batch solvers for unfolded
//! eigenvalues. Apply [`fold_quasienergy`] to fold them into the first Floquet zone
//! by
//!
//! ```math
//! \varepsilon_F =
//! \left(\varepsilon+\frac{\Omega_0}{2}\right)\bmod \Omega_0
//! -
//! \frac{\Omega_0}{2}.
//! ```
//! The implementation compares distances after taking a signed remainder,
//! avoiding overflow in the equivalent expression `epsilon + Omega_0/2`.
//!
//! # Van Vleck effective model (high-frequency expansion)
//!
//! When `Omega_0` is large compared to the bandwidth, the photon-dressed bands
//! are well separated and the physics can be captured by a **same-size** static
//! model obtained through the van Vleck expansion:
//!
//! Writing $W=\hbar\Omega_0=$ `FloquetDrive::omega0_ev`, the expansion is
//!
//! ```math
//! H_{\mathrm{eff}}(\mathbf k) =
//! H^{(0)}(\mathbf k)
//! +
//! \sum_{n=1}^{n_{\max}}
//! \frac{[H^{(n)}(\mathbf k), H^{(-n)}(\mathbf k)]}{nW}
//! +
//! H_{\mathrm{eff}}^{(2)}(\mathbf k)
//! +O(W^{-3}).
//! ```
//!
//! where `order = 2` includes both nested-commutator families through
//! $O(W^{-2})$ documented on [`Model::floquet_effective_model`].
//!
//! The Fourier blocks `H^{(n)}(k)` are defined in the [Peierls phase
//! section](#peierls-phase-and-fourier-blocks) above.  Each commutator term
//! `[H^(n), H^(-n)]` captures a virtual photon-exchange process where the
//! system absorbs a photon of energy `n W` and immediately re-emits it,
//! staying in the same photon sector but acquiring an effective hopping
//! correction of order `1/W`.
//!
//! Use [`Model::floquet_effective_model`] for this path.  It builds the
//! effective hopping blocks entirely in real space (generalized Bessel
//! backend — no k-mesh), returning a [`Model`]`<SPIN, DIM, NoRMatrix>` with
//! the same number of bands as the input model.  The k-space reference
//! implementation (uniform k-mesh + inverse Fourier transform) is kept as
//! a crate-internal `floquet_effective_model_legacy` for cross-validation.
//!
//! # API overview
//!
//! Use [`Floquet::floquet_model`] when you want a reusable static tight-binding
//! model for band plotting, cuts, or other existing `Model` workflows. Construct
//! it once in real space, then use ordinary `Model` Hamiltonian and band solvers.
//!
//! | Type / method | Meaning |
//! |---------------|---------|
//! | [`LightMode`] | One harmonic component `(harmonic, a_complex)` |
//! | [`FloquetDrive`] | Base photon energy plus all light modes |
//! | [`FloquetTruncation`] | Photon cutoff `n_max` |
//! | [`IncidentBasis`] | 3D transverse basis from an incident direction |
//! | [`FloquetEffectiveOptions`] | Effective-model expansion order and harmonic cutoff |
//! | [`Floquet::floquet_model`] | Build an enlarged static Sambe tight-binding model |
//! | [`Model::floquet_effective_model`] | Build a same-size high-frequency effective model |
//! | [`Model::gen_ham`] / [`crate::Solve`] | Use the constructed model's ordinary Hamiltonian and band solvers |
//! | [`fold_quasienergy`] | Fold solved energies into the first Floquet zone |
//!
//! # Example
//!
//! The example below builds a simple cubic one-orbital model, constructs a
//! circularly polarized drive incident along `+z`, and computes quasienergies at
//! one `k` point.
//!
//! ```no_run
//! use Rustb::*;
//! use ndarray::{arr1, array};
//! use num_complex::Complex;
//!
//! fn main() -> Result<()> {
//!     let lat = array![
//!         [1.0, 0.0, 0.0],
//!         [0.0, 1.0, 0.0],
//!         [0.0, 0.0, 1.0],
//!     ];
//!     let orb = array![[0.0, 0.0, 0.0]];
//!     let mut model = Model::<false, 3>::tb_model(lat, orb, None)?;
//!     model.set_hop(-1.0, 0, 0, &arr1(&[1isize, 0, 0]), None);
//!     model.set_hop(-1.0, 0, 0, &arr1(&[0isize, 1, 0]), None);
//!     model.set_hop(-1.0, 0, 0, &arr1(&[0isize, 0, 1]), None);
//!
//!     let incident = IncidentBasis::from_direction(&arr1(&[0.0, 0.0, 1.0]))?;
//!     let circular = incident.polarization([
//!         Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
//!         Complex::new(0.0, 1.0 / 2.0_f64.sqrt()),
//!     ]);
//!
//!     let drive = FloquetDrive::with_modes(
//!         0.8,
//!         vec![LightMode::new(1, circular.mapv(|z| 0.15 * z))],
//!     );
//!     let trunc = FloquetTruncation::new(1);
//!     let k = arr1(&[0.25, 0.0, 0.0]);
//!
//!     let floquet_model = model.floquet_model(&drive, &trunc)?;
//!     let unfolded = floquet_model.solve_band_onek(&k)?;
//!     let mut quasienergies = unfolded.mapv(|energy| fold_quasienergy(energy, drive.omega0_ev));
//!     quasienergies.as_slice_mut().unwrap().sort_by(f64::total_cmp);
//!     println!("unfolded Sambe bands = {unfolded:?}");
//!     println!("folded quasienergies = {quasienergies:?}");
//!     Ok(())
//! }
//! ```
//!
//! See also `examples/floquet_chain/main.rs`.

//!
//! # API guide
//!
//! | Entry point | Returned basis | Intended regime |
//! |---|---|---|
//! | [`Floquet::floquet_model`] | `nsta * (2*n_max + 1)` | Full truncated Sambe problem, exact in `n_max` |
//! | [`Model::floquet_effective_model`] | `nsta` | Coherent, off-resonant van Vleck expansion |
//! | [`Model::floquet_effective_mode_resolved_model`] | `nsta` | Mutually incoherent mode-diagonal correction sum |
//! | [`Model::floquet_effective_q_model`] | `nsta` | One coherent long-wavelength correction through `O(A^2 q/W)` |
//!
//! `LightMode::a_complex` is the complex amplitude of `eA/hbar` in inverse
//! lattice-length units; its integer harmonic is measured relative to
//! `FloquetDrive::omega0_ev`. The three effective-model methods take only
//! [`FloquetEffectiveOptions`] (`order` in 0..=2, `harmonic_max`) and no photon
//! truncation; [`FloquetTruncation`] controls the Sambe sector count only.
//! Every entry point returns an ordinary real-space model, so the usual
//! Hamiltonian and band solvers apply. Eigenvalues are unfolded and
//! gauge-independent; apply [`fold_quasienergy`] and re-sort when folded
//! quasienergies are required.
//!
//! # Examples
//!
//! ```no_run
//! use ndarray::{arr1, arr2, array};
//! use num_complex::Complex;
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
//! let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! for r in [[0, 0], [-1, 0], [0, -1]] {
//!     model.add_hop(-2.7, 0, 1, &array![r[0], r[1]], None);
//! }
//!
//! // A two-component (circularly polarized) single-harmonic drive.
//! let drive = FloquetDrive::with_modes(
//!     0.8,
//!     vec![LightMode::new(
//!         1,
//!         arr1(&[Complex::new(0.1, 0.0), Complex::new(0.0, 0.1)]),
//!     )],
//! );
//!
//! // Full Sambe model, then ordinary solvers plus folding.
//! let sambe = model.floquet_model(&drive, &FloquetTruncation::new(1))?;
//! assert_eq!(sambe.nsta(), model.nsta() * 3);
//! let k = arr1(&[0.2, 0.1]);
//! let unfolded = sambe.solve_band_onek(&k)?;
//! let mut folded: Vec<f64> = unfolded
//!     .iter()
//!     .map(|&e| fold_quasienergy(e, drive.omega0_ev))
//!     .collect();
//! folded.sort_by(f64::total_cmp);
//!
//! // Same-size effective models: order 1 by default, order 2 on request.
//! let first_order = model.floquet_effective_model(&drive, None)?;
//! let second_order = model.floquet_effective_model(
//!     &drive,
//!     Some(&FloquetEffectiveOptions::new().with_order(2)),
//! )?;
//! assert_eq!(first_order.nsta(), model.nsta());
//! assert_eq!(second_order.nsta(), model.nsta());
//!
//! // Incoherent modes: one correction per mode around the static H0.
//! let incoherent = model.floquet_effective_mode_resolved_model(&drive, None)?;
//! assert_eq!(incoherent.nsta(), model.nsta());
//!
//! // Coherent long-wavelength correction for one Cartesian q.
//! let q = arr1(&[2.0e-3, 0.0]);
//! let linear_q = model.floquet_effective_q_model(&drive, None, &q)?;
//! assert_eq!(linear_q.nsta(), model.nsta());
//! # Ok(())
//! # }
//! ```

mod coefficients;
mod real_space;

use coefficients::*;
use real_space::*;

#[cfg(test)]
use crate::Gauge;
use crate::error::{Result, TbError};
use crate::model::NoRMatrix;
use crate::model::utils::find_R;
#[cfg(test)]
use crate::ndarray_lapack::eigvalsh_v;
use crate::{Model, OrbitalId, RMatrixData};
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::prelude::*;
use ndarray::*;
#[cfg(test)]
use ndarray_linalg::UPLO;
use num_complex::Complex;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
#[cfg(test)]
use std::f64::consts::TAU;

/// One commensurate Fourier component of the vector potential.
///
/// `a_complex` stores the complex amplitude of
/// `a(t) = Re[a_complex * exp(-i * harmonic * omega0 * t)]`, where
/// `a = e A / hbar` has units of inverse length matching `Model::lat`.
///
/// In formulas,
///
/// $$ \mathbf a_l(t) =
/// \operatorname{Re}\left[
/// \mathbf a_l e^{-il\Omega_0 t}
/// \right]. $$
///
/// `harmonic = l` may be any integer.  Use `l = 1` for the fundamental,
/// `l = 2` for the second harmonic, etc.
#[derive(Clone, Debug)]
pub struct LightMode {
    /// Integer harmonic `l` measured in units of `FloquetDrive::omega0_ev`.
    pub harmonic: isize,
    /// Complex amplitude `a_l = e A_l / hbar` in inverse-length units.
    pub a_complex: Array1<Complex<f64>>,
}

impl LightMode {
    pub fn new(harmonic: isize, a_complex: Array1<Complex<f64>>) -> Self {
        Self {
            harmonic,
            a_complex,
        }
    }
}

/// Commensurate light drive with base photon energy `omega0_ev`.
///
/// The full field is the sum of all modes:
///
/// $$ \mathbf a(t) =
/// \operatorname{Re}\sum_\alpha
/// \mathbf a_\alpha e^{-il_\alpha\Omega_0 t}. $$
///
/// `omega0_ev` is the photon energy `Omega_0` in eV.  All `LightMode::harmonic`
/// values are integer multiples of this base frequency.
#[derive(Clone, Debug)]
pub struct FloquetDrive {
    /// Base photon energy `Omega_0` in eV.
    pub omega0_ev: f64,
    /// Harmonic components of the drive.
    pub modes: Vec<LightMode>,
}

impl FloquetDrive {
    /// Construct a drive with no light modes.
    ///
    /// This is useful for checking static photon replicas:
    /// `E_n(k) + m omega0_ev`.
    pub fn new(omega0_ev: f64) -> Self {
        Self {
            omega0_ev,
            modes: Vec::new(),
        }
    }

    /// Construct a drive from an explicit mode list.
    pub fn with_modes(omega0_ev: f64, modes: Vec<LightMode>) -> Self {
        Self { omega0_ev, modes }
    }

    /// Append one harmonic component to the drive.
    pub fn add_mode(&mut self, mode: LightMode) {
        self.modes.push(mode);
    }
}

/// Photon-sector and time-grid truncation for a commensurate drive.
///
/// The Sambe sector index is truncated to
///
/// $$
/// n \in [-N,N],
/// $$
///
/// where `N = n_max`, so the Hamiltonian dimension is
///
/// $$
/// N_{\mathrm{Sambe}} = N_{\mathrm{state}}(2N+1).
/// $$
///
/// Both Floquet paths evaluate the Peierls coefficients `C_n(d)` with the
/// exact, grid-free Bessel backend, so no entry point takes a sampling count:
/// this truncation carries the photon cutoff and nothing else.
///
/// Links beyond the Bessel backend's exact range fall back to a per-link
/// time-grid DFT whose resolution is sized from the link's own spectral
/// bandwidth and the requested harmonic range, clamped to `2^20` points with a
/// warn-once message; a drive whose link needs more than that is rejected up
/// front instead of being silently aliased.  The exact range is the closed
/// form's `MAX_BESSEL_ARG_CLOSED_FORM` when the drive's nonzero projections
/// collapse to one coherent harmonic or to two carriers, and the convolution's
/// smaller `MAX_BESSEL_ARG` otherwise.
///
/// The van Vleck effective-model path uses [`FloquetEffectiveOptions`]
/// instead of this truncation.
#[derive(Clone, Copy, Debug)]
pub struct FloquetTruncation {
    /// Photon cutoff `N`.
    pub n_max: isize,
}

impl FloquetTruncation {
    pub fn new(n_max: isize) -> Self {
        Self { n_max }
    }

    #[inline]
    /// Number of photon sectors; rejects negative or overflowing cutoffs.
    pub fn n_sector(&self) -> Result<usize> {
        self.n_max
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .filter(|&n| self.n_max >= 0 && n > 0)
            .map(|n| n as usize)
            .ok_or_else(|| {
                TbError::Other("Floquet photon-sector count is negative or overflows".into())
            })
    }

    #[inline]
    /// Inclusive photon sectors after validating the cutoff.
    pub fn sectors(&self) -> Result<std::ops::RangeInclusive<isize>> {
        self.n_sector()?;
        Ok(-self.n_max..=self.n_max)
    }
}

/// Transverse polarization basis for a 3D incident direction.
///
/// Given a propagation direction `k_hat`, this type constructs two orthonormal
/// transverse vectors `e1` and `e2`.  A Jones vector `(c1,c2)` then defines
///
/// $$
/// \boldsymbol\epsilon = c_1\mathbf e_1+c_2\mathbf e_2.
/// $$
///
/// Examples:
///
/// - linear polarization along `e1`: `(1,0)`; - circular polarization: `(1,i)/sqrt(2)`;
/// - elliptical polarization: arbitrary complex `(c1,c2)`.
#[derive(Clone, Debug)]
pub struct IncidentBasis {
    /// Normalized incident-light direction.
    pub k_hat: Array1<f64>,
    /// First transverse unit vector.
    pub e1: Array1<f64>,
    /// Second transverse unit vector.
    pub e2: Array1<f64>,
}

impl IncidentBasis {
    /// Build a right-handed transverse basis from an incident wave-vector
    /// direction in Cartesian coordinates.
    pub fn from_direction(k_hat_cart: &Array1<f64>) -> Result<Self> {
        if k_hat_cart.len() != 3 {
            return Err(TbError::DimensionMismatch {
                context: "IncidentBasis::from_direction".to_string(),
                expected: 3,
                found: k_hat_cart.len(),
            });
        }
        let k_hat = normalize3(k_hat_cart)?;
        let reference = if k_hat[2].abs() < 0.9 {
            arr1(&[0.0, 0.0, 1.0])
        } else {
            arr1(&[1.0, 0.0, 0.0])
        };
        let e1 = normalize3(&cross3(&reference, &k_hat))?;
        let e2 = normalize3(&cross3(&k_hat, &e1))?;
        Ok(Self { k_hat, e1, e2 })
    }

    /// Return `jones[0] * e1 + jones[1] * e2`.
    pub fn polarization(&self, jones: [Complex<f64>; 2]) -> Array1<Complex<f64>> {
        let mut out = Array1::<Complex<f64>>::zeros(3);
        for i in 0..3 {
            out[i] = jones[0] * self.e1[i] + jones[1] * self.e2[i];
        }
        out
    }
}

/// Optional controls for building a same-size high-frequency Floquet
/// effective model, shared by [`Model::floquet_effective_model`]
/// (real-space Bessel backend) and the crate-internal legacy k-space
/// reference path.
///
/// `order` and `harmonic_max` control the high-frequency expansion on both paths.
/// `target_hamR` (crate-internal) applies only to the legacy path, whose
/// inverse Fourier transform
///
/// $$ t_{\mathrm{eff}}(\mathbf R) =
/// \frac{1}{N_k}\sum_{\mathbf k}
/// H_{\mathrm{eff}}(\mathbf k)
/// e^{-i2\pi\mathbf k\cdot\mathbf R} $$
///
/// projects `H_eff(k)` onto the given hopping vectors.  If it is `None`,
/// the original model's `hamR` is used, keeping the returned model on the
/// same real-space hopping range as the input model; provide a larger
/// `target_hamR` when the commutator terms are expected to generate
/// longer-range effective hoppings.  Every vector must occur exactly once,
/// and the set must be closed under `R -> -R`, so the inverse-transformed
/// model can satisfy `H(-R) = H(R)^\dagger`.  The real-space path
/// determines its own support automatically and rejects a supplied
/// `target_hamR`.
#[derive(Clone, Debug)]
pub struct FloquetEffectiveOptions {
    /// van Vleck order.  Supported: `0`, `1`, and `2`, retaining terms
    /// through `O(omega^0)`, `O(omega^-1)`, and `O(omega^-2)`, respectively.
    pub order: usize,
    /// Signed-harmonic cutoff for commutator sums. Defaults to `2`,
    /// independently of any Sambe photon cutoff. At order 2 the mixed component
    /// `H_(m'-m)` is
    /// evaluated up to `|m'-m| = 2 * harmonic_max` automatically.
    pub harmonic_max: isize,
    /// Optional target real-space hopping vectors for the legacy path's
    /// inverse Fourier transform.  Rejected by the real-space path.
    pub(crate) target_hamR: Option<Array2<isize>>,
}

impl Default for FloquetEffectiveOptions {
    fn default() -> Self {
        Self {
            order: 1,
            harmonic_max: 2,
            target_hamR: None,
        }
    }
}

impl FloquetEffectiveOptions {
    /// Construct first-order options using `harmonic_max = 2`.
    /// The real-space effective model determines its hopping support automatically.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the van Vleck order.  `0`, `1`, and `2` are supported.
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set the signed-harmonic cutoff used in the van Vleck sums.
    pub fn with_harmonic_max(mut self, harmonic_max: isize) -> Self {
        self.harmonic_max = harmonic_max;
        self
    }

    /// Set the real-space hopping vectors used by the legacy path's
    /// inverse Fourier transform.  Rejected by the real-space path.
    #[cfg(test)]
    pub(crate) fn with_target_hamR(mut self, target_hamR: Array2<isize>) -> Self {
        self.target_hamR = Some(target_hamR);
        self
    }
}

fn coherent_positive_harmonic_amplitudes<const DIM: usize>(
    drive: &FloquetDrive,
    harmonic_max: isize,
) -> Result<std::collections::BTreeMap<isize, Array1<Complex<f64>>>> {
    let mut amplitudes = std::collections::BTreeMap::<isize, Array1<Complex<f64>>>::new();
    for (mode_index, mode) in drive.modes.iter().enumerate() {
        let active = mode
            .a_complex
            .iter()
            .any(|value| value.re != 0.0 || value.im != 0.0);
        if !active {
            continue;
        }
        if mode.harmonic <= 0 {
            return Err(TbError::Other(format!(
                "floquet_effective_q_model requires positive temporal harmonics for nonzero q; \
                 drive.modes[{mode_index}].harmonic = {}",
                mode.harmonic
            )));
        }
        if mode.harmonic > harmonic_max {
            continue;
        }
        let total = amplitudes
            .entry(mode.harmonic)
            .or_insert_with(|| Array1::<Complex<f64>>::zeros(DIM));
        *total += &mode.a_complex;
        if total
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(TbError::Other(format!(
                "floquet_effective_q_model coherent amplitude sum overflowed at harmonic {}",
                mode.harmonic
            )));
        }
    }
    amplitudes.retain(|_, amplitude| {
        debug_assert_eq!(amplitude.len(), DIM);
        amplitude
            .iter()
            .any(|value| value.re != 0.0 || value.im != 0.0)
    });
    Ok(amplitudes)
}

#[inline]
fn q_linear_frequency_scale(harmonic: isize, photon_energy_ev: f64) -> f64 {
    -photon_energy_ev.recip() / (8.0 * (harmonic as f64))
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Build the coherent finite-wavevector effective model through first
    /// order in one common Cartesian light wavevector.
    ///
    /// All entries of `drive.modes` describe one coherent field.  Complex
    /// amplitudes with the same positive temporal harmonic are summed before
    /// the finite-`q` correction is evaluated.  `wavevector_cartesian` is the
    /// physical wavevector shared by those modes, in inverse-length units
    /// reciprocal to [`Model::lat`].
    /// The spatial convention is
    ///
    /// ```math
    /// a(r,t)=\operatorname{Re}\sum_n a_n
    /// e^{+i q\cdot r-in\Omega_0t}.
    /// ```
    ///
    /// The returned model starts from the ordinary `q = 0` result of
    /// [`Model::floquet_effective_model`] (including its requested Bessel and
    /// van Vleck orders) and adds the weak-field correction
    ///
    /// ```math
    /// \delta_q H_{\rm eff}
    /// =-\sum_{n>0}\frac{q_a a_{ni}a^*_{nj}}{8nW}
    /// \partial_{k_a}\{H_i,H_j\},
    /// \qquad H_i=\partial_{k_i}H_0,
    /// ```
    ///
    /// with `W = drive.omega0_ev`.  The minus sign and factor `1/8` follow
    /// this module's conventions `exp[-i a(t)·d]`,
    /// `a(t) = Re[a exp(-in Omega t)]`, and
    /// `[H^(n), H^(-n)]/(n W)`.  Cartesian derivatives are evaluated
    /// analytically in the fixed Wannier/atomic basis from the full bond
    /// displacement; no finite difference of band eigenvectors is used.
    ///
    /// This is a controlled `O(A^2 q / W)` spatial-dispersion correction, not
    /// an exact finite-wavevector Sambe solver.  The `q = 0` baseline may be
    /// retained through `O(W^-2)`, but the added `q` term itself is only
    /// `O(W^-1)`.  The added term requires both `|a_n·d| << 1` and
    /// `|q·d| << 1` on relevant bonds; the all-orders `q = 0` baseline does not
    /// remove that weak-field requirement from the derivative itself.
    /// It keeps the primitive-cell state count and introduces no photon-sector
    /// or optical-supercell bands.
    ///
    /// A single ordinary periodic model cannot retain interference between
    /// coherent beams with distinct wavevectors.  Such beams therefore need
    /// separate calls (or a future graded/supercell API); this method models
    /// one common propagation wavevector exactly to linear order.
    pub fn floquet_effective_q_model(
        &self,
        drive: &FloquetDrive,
        options: Option<&FloquetEffectiveOptions>,
        wavevector_cartesian: &Array1<f64>,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>> {
        self.floquet_effective_q_linear_model(drive, wavevector_cartesian, options)
    }

    fn floquet_effective_q_linear_model(
        &self,
        drive: &FloquetDrive,
        wavevector_cartesian: &Array1<f64>,
        options: Option<&FloquetEffectiveOptions>,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>> {
        if wavevector_cartesian.len() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "floquet_effective_q_model wavevector_cartesian".to_string(),
                expected: DIM,
                found: wavevector_cartesian.len(),
            });
        }
        if wavevector_cartesian.iter().any(|value| !value.is_finite()) {
            return Err(TbError::Other(
                "floquet_effective_q_model wavevector_cartesian contains non-finite values"
                    .to_string(),
            ));
        }

        let mut effective = self.floquet_effective_model(drive, options)?;
        if wavevector_cartesian.iter().all(|value| *value == 0.0) {
            return Ok(effective);
        }

        let default_options;
        let options = match options {
            Some(options) => options,
            None => {
                default_options = FloquetEffectiveOptions::default();
                &default_options
            }
        };
        if options.order == 0 {
            return Ok(effective);
        }
        let harmonic_max = effective_harmonic_max(options)?;
        let amplitudes = coherent_positive_harmonic_amplitudes::<DIM>(drive, harmonic_max)?;
        if amplitudes.is_empty() {
            return Ok(effective);
        }

        let nsta = self.nsta();
        let norb = self.norb();
        let mut correction = RealSpaceBlockMap::new();
        for (harmonic, amplitude) in amplitudes {
            let conjugate = amplitude.mapv(|value| value.conj());
            let gradient = self.cartesian_gradient_contraction_blocks(&amplitude)?;
            let gradient_conjugate = self.cartesian_gradient_contraction_blocks(&conjugate)?;
            let (mut jordan_blocks, jordan_r) =
                real_space_anticommutator(&gradient, &gradient_conjugate, &self.hamR)?;

            for (i_r, r_vec) in jordan_r.outer_iter().enumerate() {
                let mut block = jordan_blocks[i_r].view_mut();
                for i in 0..nsta {
                    for j in 0..nsta {
                        let d_cart = self.link_displacement_cartesian(i % norb, j % norb, &r_vec);
                        let q_dot_d = wavevector_cartesian.dot(&d_cart);
                        if !q_dot_d.is_finite() {
                            return Err(TbError::Other(format!(
                                "floquet_effective_q_model q·d overflowed for harmonic {harmonic}, \
                                 support row {i_r}, state pair ({i}, {j})"
                            )));
                        }
                        block[[i, j]] *= Complex::new(0.0, q_dot_d);
                        if !block[[i, j]].re.is_finite() || !block[[i, j]].im.is_finite() {
                            return Err(TbError::Other(format!(
                                "floquet_effective_q_model finite-q correction overflowed for \
                                 harmonic {harmonic}, support row {i_r}, state pair ({i}, {j})"
                            )));
                        }
                    }
                }
            }

            let scale = q_linear_frequency_scale(harmonic, drive.omega0_ev);
            accumulate_scaled_real_space_blocks(&mut correction, &jordan_blocks, &jordan_r, scale)?;
        }

        let mut combined = RealSpaceBlockMap::new();
        for (i_r, row) in effective.hamR.outer_iter().enumerate() {
            combined.insert(
                row.to_vec(),
                effective.ham.index_axis(Axis(0), i_r).to_owned(),
            );
        }
        for (row, block) in correction {
            combined
                .entry(row)
                .and_modify(|target| *target += &block)
                .or_insert(block);
        }

        let mut ham_r = Array2::<isize>::zeros((combined.len(), DIM));
        let mut ham = Array3::<Complex<f64>>::zeros((combined.len(), nsta, nsta));
        for (i_r, (row, block)) in combined.into_iter().enumerate() {
            for axis in 0..DIM {
                ham_r[[i_r, axis]] = row[axis];
            }
            ham.index_axis_mut(Axis(0), i_r).assign(&block);
        }
        place_origin_first(&mut ham, &mut ham_r)?;
        enforce_real_space_hermiticity(&mut ham, &ham_r)?;
        if ham
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(TbError::Other(
                "floquet_effective_q_model produced non-finite hopping values".to_string(),
            ));
        }
        effective.ham = ham;
        effective.hamR = ham_r;
        effective.validate()?;
        Ok(effective)
    }

    /// Build the mode-diagonal effective model of mutually incoherent modes.
    ///
    /// Every nonzero [`LightMode`] is evaluated in its own one-mode
    /// [`FloquetDrive`], and the induced corrections are added around one
    /// common static Hamiltonian:
    ///
    /// ```math
    /// H_{\rm inc}=H_0+\sum_\alpha
    /// \left(H_{\rm eff}[a_\alpha]-H_0\right).
    /// ```
    ///
    /// The amplitude of each mode already specifies its physical intensity,
    /// so this API has no additional statistical weights.  Cross-mode
    /// coherent interference and mixed-mode virtual paths are intentionally
    /// absent.  This agrees with random-relative-phase averaging at leading
    /// `O(A^2)`, but is not an exact all-orders phase average: cross-intensity
    /// terms such as `A_alpha^2 A_beta^2` are omitted.
    ///
    /// Exact-zero modes are skipped.  Real-space supports from all one-mode
    /// results are unioned, and the returned model has the same state count
    /// and metadata as the input model. The sum uses ordinary `f64` arithmetic;
    /// changing mode order can change floating-point rounding. An unrepresentable
    /// intermediate correction is rejected even when an exact-arithmetic final
    /// sum could be finite; no arbitrary-precision accumulator is used.
    pub fn floquet_effective_mode_resolved_model(
        &self,
        drive: &FloquetDrive,
        options: Option<&FloquetEffectiveOptions>,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>> {
        let default_options;
        let options = match options {
            Some(options) => options,
            None => {
                default_options = FloquetEffectiveOptions::default();
                &default_options
            }
        };
        validate_floquet_drive::<DIM>(drive)?;
        if options.order > 2 {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.order must be 0, 1, or 2, got {}",
                options.order
            )));
        }
        if let Some(target) = &options.target_hamR {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.target_hamR is not supported by the \
                 real-space path: the effective support is determined \
                 automatically (got {} target vectors)",
                target.nrows()
            )));
        }
        effective_harmonic_max(options)?;
        let nsta = self.nsta();
        let active_modes = drive
            .modes
            .iter()
            .filter(|mode| {
                mode.a_complex
                    .iter()
                    .any(|value| value.re != 0.0 || value.im != 0.0)
            })
            .collect::<Vec<_>>();
        if active_modes.is_empty() {
            let mut static_model = Model::<SPIN, DIM, NoRMatrix>::tb_model(
                self.lat.clone(),
                self.orb.clone(),
                Some(self.atoms.clone()),
            )?;
            static_model.ham = self.ham.clone();
            static_model.hamR = self.hamR.clone();
            static_model.orb_projection = self.orb_projection.clone();
            place_origin_first(&mut static_model.ham, &mut static_model.hamR)?;
            static_model.validate()?;
            return Ok(static_model);
        }
        if active_modes.len() == 1 {
            let single_drive =
                FloquetDrive::with_modes(drive.omega0_ev, vec![active_modes[0].clone()]);
            return self.floquet_effective_model(&single_drive, Some(options));
        }

        // Accumulate the induced differences around H0. Multiplying H0 by
        // (1 - N) first can overflow even when every correction is exactly zero.
        let mut blocks = RealSpaceBlockMap::new();
        for (i_r, row) in self.hamR.outer_iter().enumerate() {
            blocks.insert(row.to_vec(), self.ham.index_axis(Axis(0), i_r).to_owned());
        }

        for mode_def in active_modes {
            let single_drive = FloquetDrive::with_modes(drive.omega0_ev, vec![mode_def.clone()]);
            let single = self.floquet_effective_model(&single_drive, Some(options))?;

            for (i_r, row) in single.hamR.outer_iter().enumerate() {
                let dst = blocks
                    .entry(row.to_vec())
                    .or_insert_with(|| Array2::zeros((nsta, nsta)));
                let dressed = single.ham.index_axis(Axis(0), i_r);
                if let Some(original) = find_R(&self.hamR, &row) {
                    let baseline = self.ham.index_axis(Axis(0), original);
                    Zip::from(&mut *dst)
                        .and(dressed)
                        .and(baseline)
                        .for_each(|out, &value, &static_value| *out += value - static_value);
                } else {
                    *dst += &dressed;
                }
            }
        }

        let mut effective = Model::<SPIN, DIM, NoRMatrix>::tb_model(
            self.lat.clone(),
            self.orb.clone(),
            Some(self.atoms.clone()),
        )?;
        effective.orb_projection = self.orb_projection.clone();
        effective.hamR = Array2::zeros((blocks.len(), DIM));
        effective.ham = Array3::zeros((blocks.len(), nsta, nsta));
        for (i_r, (row, block)) in blocks.into_iter().enumerate() {
            for axis in 0..DIM {
                effective.hamR[[i_r, axis]] = row[axis];
            }
            if block
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(TbError::Other(format!(
                    "floquet_effective_mode_resolved_model produced non-finite hopping values at support row {i_r}"
                )));
            }
            effective.ham.index_axis_mut(Axis(0), i_r).assign(&block);
        }
        place_origin_first(&mut effective.ham, &mut effective.hamR)?;
        enforce_real_space_hermiticity(&mut effective.ham, &effective.hamR)?;
        effective.validate()?;
        Ok(effective)
    }
}

/// Peierls-Floquet Sambe construction for tight-binding models.
///
/// The only construction entry point returns a real-space model. The removed
/// single-k shortcuts are deliberately not compatibility APIs; use the returned
/// model's ordinary Hamiltonian and band solvers instead.
///
/// ```compile_fail,E0599
/// use Rustb::{Floquet, FloquetDrive, FloquetTruncation, Gauge, Model};
/// fn removed(model: &Model<false, 1>, drive: &FloquetDrive, trunc: &FloquetTruncation) {
///     model.floquet_ham_onek(&ndarray::array![0.0], drive, trunc, Gauge::Lattice);
/// }
/// ```
///
/// ```compile_fail,E0599
/// use Rustb::{Floquet, FloquetDrive, FloquetTruncation, Gauge, Model};
/// fn removed(model: &Model<false, 1>, drive: &FloquetDrive, trunc: &FloquetTruncation) {
///     model.floquet_band_onek(&ndarray::array![0.0], drive, trunc, Gauge::Lattice);
/// }
/// ```
///
/// ```compile_fail,E0599
/// use Rustb::{Floquet, FloquetDrive, FloquetTruncation, Gauge, Model};
/// fn removed(model: &Model<false, 1>, drive: &FloquetDrive, trunc: &FloquetTruncation) {
///     model.floquet_quasienergy_onek(&ndarray::array![0.0], drive, trunc, Gauge::Lattice);
/// }
/// ```
pub trait Floquet {
    /// Static model type produced by [`Floquet::floquet_model`].
    type FloquetModel;

    /// Build an enlarged static tight-binding model in Sambe space.
    ///
    /// The returned model has the same spatial lattice and hopping range as
    /// the original model, but its internal basis is enlarged from
    /// `N_state` to
    ///
    /// ```math
    /// N_{\mathrm{state}}(2N+1),
    /// ```
    ///
    /// where `N = trunc.n_max`.  Photon sectors run from `-N` to `N`.
    /// Spinless models are ordered as `(photon sector, orbital)`.  Spinful
    /// models preserve the usual Rustb spin layout and are ordered as
    /// `(spin, photon sector, orbital)`.
    ///
    /// The real-space matrix elements are
    ///
    /// ```math
    /// \langle i,n;\mathbf 0|H_F|j,m;\mathbf R\rangle =
    /// t_{ij}(\mathbf R) C_{n-m}(\mathbf d_{ij\mathbf R})
    /// +
    /// n\Omega_0\delta_{nm}\delta_{ij}\delta_{\mathbf R,0}.
    /// ```
    ///
    /// The result preserves the input model's `SPIN` const generic.  Photon
    /// sectors are encoded as additional orbitals; if the input model is
    /// spinful, physical spin remains the `Model<true, DIM, _>` spin degree of
    /// freedom rather than being flattened away.
    ///
    /// This model is stored in real space as `ham` and `hamR`. Construct it
    /// once, then use ordinary [`Model::gen_ham`], [`crate::Solve`] and other
    /// model workflows. `Gauge::Atom` applies the atomic gauge phase to the
    /// enlarged orbital positions. Single-k and eigenvector solvers use that
    /// gauge; energy-only batch solvers use the lattice gauge. Their unfolded
    /// eigenvalues are gauge-independent and can be folded with
    /// [`fold_quasienergy`]; they must
    /// be sorted again if ascending quasienergies are desired.
    fn floquet_model(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
    ) -> Result<Self::FloquetModel>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Floquet for Model<SPIN, DIM, R> {
    type FloquetModel = Model<SPIN, DIM, NoRMatrix>;

    fn floquet_model(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
    ) -> Result<Self::FloquetModel> {
        validate_floquet_drive::<DIM>(drive)?;
        validate_floquet_truncation(trunc)?;
        validate_sambe_allocation(self, drive, trunc)?;

        let nsta = self.nsta();
        let norb = self.norb();
        let sectors: Vec<isize> = trunc.sectors()?.collect();
        let n_sector = sectors.len();
        let new_norb = norb * n_sector;
        let total = nsta * n_sector;
        let basis_indices = floquet_basis_indices::<SPIN>(nsta, norb, n_sector);
        let harmonic_min = -2 * trunc.n_max;
        let harmonic_max = 2 * trunc.n_max;
        // The Sambe path shares the exact, grid-free Bessel backend with the
        // effective-model path; links beyond the backend's range fall back to a
        // self-sized per-link grid inside the cache.
        let harmonic_cache = self.floquet_harmonic_cache(
            drive,
            harmonic_min,
            harmonic_max,
            &PeierlsFourierMethod::Bessel {
                cutoff_margin: BESSEL_CUTOFF_MARGIN,
            },
        );

        let mut orb = Array2::<f64>::zeros((new_norb, DIM));
        for isec in 0..n_sector {
            for iorb in 0..norb {
                let out_i = isec * norb + iorb;
                orb.row_mut(out_i).assign(&self.orb.row(iorb));
            }
        }

        let mut ham_r = self.hamR.clone();
        let mut ham = Array3::<Complex<f64>>::zeros((ham_r.nrows(), total, total));
        ham.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i_r, mut out)| {
                for i in 0..nsta {
                    for j in 0..nsta {
                        // Exact zero, not `norm_sqr`: a hopping of 1e-200 squares
                        // to zero and must still enter the Sambe matrix.
                        if harmonic_cache
                            .blocks
                            .slice(s![.., i_r, i, j])
                            .iter()
                            .all(|x| x.re == 0.0 && x.im == 0.0)
                        {
                            continue;
                        }

                        for (in_sec, &n) in sectors.iter().enumerate() {
                            let row = basis_indices[in_sec][i];
                            for (im_sec, &m) in sectors.iter().enumerate() {
                                let hopping = harmonic_cache.blocks
                                    [[harmonic_cache.harmonic_index(n - m), i_r, i, j]];
                                if hopping.re == 0.0 && hopping.im == 0.0 {
                                    continue;
                                }
                                let col = basis_indices[im_sec][j];
                                out[[row, col]] += hopping;
                            }
                        }
                    }
                }
            });

        place_origin_first(&mut ham, &mut ham_r)?;
        let onsite_index = 0;

        for (in_sec, &n) in sectors.iter().enumerate() {
            let photon_shift = n as f64 * drive.omega0_ev;
            for i in 0..nsta {
                let idx = basis_indices[in_sec][i];
                ham[[onsite_index, idx, idx]] += Complex::new(photon_shift, 0.0);
            }
        }

        let atoms = (0..n_sector)
            .flat_map(|sector| {
                self.atoms.iter().cloned().map(move |mut atom| {
                    atom.set_orbitals(
                        atom.orbitals()
                            .iter()
                            .map(|id| OrbitalId::new(sector * norb + id.index()))
                            .collect(),
                    );
                    atom
                })
            })
            .collect();
        let mut model =
            Model::<SPIN, DIM, NoRMatrix>::tb_model(self.lat.clone(), orb, Some(atoms))?;
        model.ham = ham;
        model.hamR = ham_r;
        model.orb_projection = (0..n_sector)
            .flat_map(|_| (0..norb).map(|i| self.orb_projection[i]))
            .collect();

        if model
            .ham
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(TbError::Other(
                "Sambe matrix arithmetic produced non-finite entries".into(),
            ));
        }
        Ok(model)
    }
}

#[cfg(test)]
impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    // Independent test oracle: assemble Fourier blocks in k-space instead of
    // Fourier-transforming the enlarged real-space model. Not a public API.
    fn reference_sambe_ham_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array2<Complex<f64>>> {
        validate_floquet_input(self, kvec, drive, trunc)?;
        validate_sambe_allocation(self, drive, trunc)?;

        let nsta = self.nsta();
        let norb = self.norb();
        let n_sector = trunc.n_sector()?;
        let total = nsta * n_sector;
        let mut hamf = Array2::<Complex<f64>>::zeros((total, total));
        let basis_indices = floquet_basis_indices::<SPIN>(nsta, norb, n_sector);

        let harmonic_min = -2 * trunc.n_max;
        let harmonic_max = 2 * trunc.n_max;
        let harmonic_cache = self.floquet_harmonic_cache(
            drive,
            harmonic_min,
            harmonic_max,
            &PeierlsFourierMethod::Bessel {
                cutoff_margin: BESSEL_CUTOFF_MARGIN,
            },
        );
        let harmonics: Vec<Array2<Complex<f64>>> = (harmonic_min..=harmonic_max)
            .map(|n| self.floquet_cached_harmonic_onek(kvec, n, gauge, &harmonic_cache))
            .collect();

        for (in_sec, n) in trunc.sectors()?.enumerate() {
            for (im_sec, m) in trunc.sectors()?.enumerate() {
                let harmonic = n - m;
                let block = &harmonics[(harmonic - harmonic_min) as usize];
                for i in 0..nsta {
                    for j in 0..nsta {
                        let row = basis_indices[in_sec][i];
                        let col = basis_indices[im_sec][j];
                        hamf[[row, col]] = block[[i, j]];
                    }
                }
            }
            let photon_shift = n as f64 * drive.omega0_ev;
            for i in 0..nsta {
                let idx = basis_indices[in_sec][i];
                hamf[[idx, idx]] += photon_shift;
            }
        }

        if hamf.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(TbError::Other(
                "Sambe matrix arithmetic produced non-finite entries".into(),
            ));
        }
        Ok(hamf)
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Legacy k-space reference path for the high-frequency Floquet
    /// effective model — crate-internal, retained for cross-validation
    /// tests and for custom `target_hamR`.  The public entry point is
    /// [`Model::floquet_effective_model`] (real-space Bessel backend),
    /// which needs neither `k_mesh` nor `target_hamR`.
    ///
    /// With the Fourier convention used in this module,
    ///
    /// $$
    /// H(t)=\sum_n H^{(n)}e^{-in\Omega t},
    /// $$
    ///
    /// the implemented van Vleck expansion through second order is
    ///
    /// $$ H_{\mathrm{eff}}(\mathbf k) =
    /// H^{(0)}(\mathbf k)
    /// +
    /// \sum_{n=1}^{n_{\max}}
    /// \frac{[H^{(n)}(\mathbf k),H^{(-n)}(\mathbf k)]}{nW}
    /// +H_{\mathrm{eff}}^{(2)}(\mathbf k)
    /// +O(W^{-3}), $$
    ///
    /// where, writing `W = omega0_ev`,
    ///
    /// ```math
    /// H_{\rm eff}^{(2)}=
    /// \sum_{m\ne0}\frac{[H_{-m},[H_0,H_m]]}{2m^2W^2}
    /// +\sum_{\substack{m\ne0,m'\ne0\\m'\ne m}}
    /// \frac{[H_{-m'},[H_{m'-m},H_m]]}{3mm'W^2}.
    /// ```
    ///
    /// `order = 0` keeps only `H^(0)`, `order = 1` adds the `1/W`
    /// commutator, and `order = 2` also adds both nested-commutator families
    /// above.  Pass `None` for `options` to use first order,
    /// `harmonic_max = 2`, and the input model's original `hamR`.
    ///
    /// The inverse Fourier transform is controlled by `k_mesh`:
    ///
    /// $$ t_{\mathrm{eff}}(\mathbf R) =
    /// \frac{1}{N_k}\sum_{\mathbf k}
    /// H_{\mathrm{eff}}(\mathbf k)
    /// e^{-i2\pi\mathbf k\cdot\mathbf R}. $$
    ///
    /// The returned model has the same number of states as the input model.
    /// It is an approximation to the off-resonant Floquet problem, not the full
    /// enlarged Sambe model returned by [`Floquet::floquet_model`].
    #[cfg(test)]
    pub(crate) fn floquet_effective_model_legacy(
        &self,
        drive: &FloquetDrive,
        n_time: usize,
        k_mesh: [usize; DIM],
        options: Option<&FloquetEffectiveOptions>,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>> {
        let default_options;
        let options = match options {
            Some(options) => options,
            None => {
                default_options = FloquetEffectiveOptions::default();
                &default_options
            }
        };

        validate_floquet_drive::<DIM>(drive)?;
        validate_floquet_time_samples(n_time)?;
        validate_effective_options::<DIM>(&k_mesh, options)?;

        let nsta = self.nsta();
        let target_ham_r = options
            .target_hamR
            .clone()
            .unwrap_or_else(|| self.hamR.clone());
        validate_target_hamr::<DIM>(&target_ham_r)?;

        let harmonic_max = effective_harmonic_max(options)?;
        let has_time_dependence = drive_has_ac_components(drive);
        let static_drive;
        let cache_drive = if has_time_dependence {
            drive
        } else {
            static_drive = static_component_drive(drive);
            &static_drive
        };
        let cache_max = if has_time_dependence {
            effective_cache_max(options.order, harmonic_max)?
        } else {
            0
        };
        validate_effective_cache_layout(cache_max, self.hamR.nrows(), nsta)?;
        let harmonic_cache = self.floquet_harmonic_cache(
            cache_drive,
            -cache_max,
            cache_max,
            &PeierlsFourierMethod::TimeGrid { n_time },
        );
        let kpoints = floquet_uniform_kmesh(&k_mesh);
        let norm = 1.0 / (kpoints.len() as f64);
        let ham = kpoints
            .par_iter()
            .try_fold(
                || Array3::<Complex<f64>>::zeros((target_ham_r.nrows(), nsta, nsta)),
                |mut partial, kvec| -> Result<Array3<Complex<f64>>> {
                    let h_eff = self.floquet_effective_ham_onek_lattice(
                        kvec,
                        drive,
                        options.order,
                        harmonic_max,
                        has_time_dependence,
                        &harmonic_cache,
                    )?;
                    for (i_r, r_vec) in target_ham_r.outer_iter().enumerate() {
                        let phase = inverse_bloch_phase::<DIM, _>(&r_vec, kvec) * norm;
                        let mut block = partial.index_axis_mut(Axis(0), i_r);
                        crate::ndarray_lapack::zaxpy(
                            phase,
                            h_eff.as_slice().unwrap(),
                            block.as_slice_mut().unwrap(),
                        );
                    }
                    Ok(partial)
                },
            )
            .try_reduce(
                || Array3::<Complex<f64>>::zeros((target_ham_r.nrows(), nsta, nsta)),
                |mut left, right| -> Result<Array3<Complex<f64>>> {
                    left.zip_mut_with(&right, |a, b| *a += *b);
                    Ok(left)
                },
            )?;

        let mut ham = ham;

        enforce_real_space_hermiticity(&mut ham, &target_ham_r)?;

        let mut model = Model::<SPIN, DIM, NoRMatrix>::tb_model(
            self.lat.clone(),
            self.orb.clone(),
            Some(self.atoms.clone()),
        )?;
        model.ham = ham;
        model.hamR = target_ham_r;
        model.orb_projection = self.orb_projection.clone();

        Ok(model)
    }

    /// Real-space van Vleck effective model through `O(omega^-2)` via the
    /// generalized Bessel backend. The numerical controls are
    /// [`FloquetEffectiveOptions`]; no photon cutoff, time-sampling count,
    /// or `k_mesh` is required. Links whose amplitude exceeds the Bessel range
    /// (beyond the backend's range) fall back to a per-link time-grid DFT whose resolution is
    /// sized from the link's own spectral bandwidth and the requested
    /// harmonic range.
    ///
    /// This is the main entry point.  The crate-internal
    /// `floquet_effective_model_legacy` is the k-space reference
    /// implementation, kept for cross-validation tests.
    ///
    /// The effective hopping blocks are built entirely in real space:
    ///
    /// ```math
    /// T_{\mathrm{eff}}(R)
    /// =
    /// T_0(R)
    /// +
    /// \sum_{n=1}^{n_{\max}} \frac{[T_n,T_{-n}](R)}{n\,W}
    /// +T_{\mathrm{eff}}^{(2)}(R),\qquad W=\hbar\Omega_0,
    /// ```
    ///
    /// where `T_0(R) = t(R)·C_0(d)` are the Peierls-dressed static blocks
    /// and `comm_n(R)` are the two-convolution commutator blocks of the
    /// internal real-space commutator for the harmonic pair `(T_n, T_{−n})`.
    /// The second-order term is
    ///
    /// ```math
    /// T_{\rm eff}^{(2)}=
    /// \sum_{m\ne0}\frac{[T_{-m},[T_0,T_m]]}{2m^2W^2}
    /// +\sum_{\substack{m\ne0,m'\ne0\\m'\ne m}}
    /// \frac{[T_{-m'},[T_{m'-m},T_m]]}{3mm'W^2}.
    /// ```
    ///
    /// Inner commutators use an internal generalized-support convolution
    /// because their support is already a double Minkowski sum and they are
    /// not individually Hermitian.  Hermiticity is enforced only after the
    /// full signed sum is accumulated.
    ///
    /// `options.order = 0` keeps only `T_0`; `order = 1` adds the `1/W`
    /// commutator terms; `order = 2` adds both `1/W^2` families.  The signed
    /// indices `m,m'` are truncated at `harmonic_max` (default `2`),
    /// while `T_(m'-m)` is evaluated automatically through `2*harmonic_max`.
    /// The real-space support is determined automatically: primitive at
    /// order 0, the union through the double Minkowski sum at order 1, and
    /// through the triple Minkowski sum at order 2.  No `target_hamR`
    /// parameter is needed, and the output is guaranteed Hermitian
    /// (`T(R) = T(−R)†` enforced exactly).
    /// The crate-internal `target_hamR` option is rejected: the real-space
    /// path determines its own support.  Blocks with
    /// vanishing coefficients (e.g. harmonics outside the drive's
    /// selection-rule reach) are retained as exact zeros — the support
    /// depends only on the input `hamR`, not on the drive content.
    ///
    /// The returned model has the same lattice, orbitals, atoms, and
    /// state count as the input model, and differs only in `ham`/`hamR`.
    /// It is an approximation to the off-resonant Floquet problem, not
    /// the full enlarged Sambe model returned by [`Floquet::floquet_model`].
    ///
    /// # Parallelism
    ///
    /// The order-2 signed-harmonic sum and sufficiently large real-space
    /// hopping convolutions automatically use Rayon's global thread pool.
    /// `RAYON_NUM_THREADS` controls the outer worker count.  When the linked
    /// BLAS also starts worker threads, workloads with many small hopping
    /// blocks usually perform best with `OPENBLAS_NUM_THREADS=1` or
    /// `MKL_NUM_THREADS=1`, avoiding nested oversubscription.
    ///
    /// # Errors
    /// Returns an error for an invalid drive, `order > 2`, a
    /// negative or unrepresentable harmonic range, a non-finite frequency
    /// scaling factor, a supplied `target_hamR`, a real-space support sum that
    /// overflows `isize`, or a support that is not closed under `R -> −R`.
    pub fn floquet_effective_model(
        &self,
        drive: &FloquetDrive,
        options: Option<&FloquetEffectiveOptions>,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>> {
        self.validate()?;
        let default_options;
        let options = match options {
            Some(options) => options,
            None => {
                default_options = FloquetEffectiveOptions::default();
                &default_options
            }
        };

        validate_floquet_drive::<DIM>(drive)?;
        if options.order > 2 {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.order must be 0, 1, or 2, got {}",
                options.order
            )));
        }
        if let Some(target) = &options.target_hamR {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.target_hamR is not supported by the \
                 real-space path: the effective support is determined \
                 automatically (got {} target vectors)",
                target.nrows()
            )));
        }
        let harmonic_max = effective_harmonic_max(options)?;

        let nsta = self.nsta();
        let has_time_dependence = drive_has_ac_components(drive);
        let static_drive;
        let cache_drive = if has_time_dependence {
            drive
        } else {
            static_drive = static_component_drive(drive);
            &static_drive
        };
        let cache_max = if has_time_dependence {
            effective_cache_max(options.order, harmonic_max)?
        } else {
            0
        };
        validate_effective_cache_layout(cache_max, self.hamR.nrows(), nsta)?;
        // The effective path has no photon truncation, but it has the same
        // fallback: a link the per-link grid cannot resolve must be refused
        // here too, not silently answered from a clamped grid.
        validate_link_resolvability(self, cache_drive, -cache_max, cache_max)?;
        let harmonic_cache = self.floquet_harmonic_cache(
            cache_drive,
            -cache_max,
            cache_max,
            &PeierlsFourierMethod::Bessel {
                cutoff_margin: BESSEL_CUTOFF_MARGIN,
            },
        );
        // Zeroth order: the Peierls-dressed static blocks on the input
        // support.  The BTreeMap merges the per-n contributions onto the
        // final support in lexicographic order (matching
        // real_space_commutator's deterministic output).
        let mut blocks = std::collections::BTreeMap::<Vec<isize>, Array2<Complex<f64>>>::new();
        let i_n0 = harmonic_cache.harmonic_index(0);
        for (i_r, row) in self.hamR.outer_iter().enumerate() {
            blocks.insert(
                row.to_vec(),
                harmonic_cache
                    .blocks
                    .slice(s![i_n0, i_r, .., ..])
                    .to_owned(),
            );
        }

        // Retain the documented support even when harmonic operands vanish,
        // but construct its zero-valued Minkowski layers only once.
        if harmonic_max > 0 && options.order >= 1 {
            let zero_primitive = (0..self.hamR.nrows())
                .map(|_| Array2::<Complex<f64>>::zeros((nsta, nsta)))
                .collect::<Vec<_>>();
            let (pair_blocks, pair_r) =
                real_space_commutator(&zero_primitive, &zero_primitive, &self.hamR)?;
            accumulate_scaled_real_space_blocks(&mut blocks, &pair_blocks, &pair_r, 0.0)?;
            if options.order >= 2 {
                let (triple_blocks, triple_r) = real_space_commutator_with_supports(
                    &zero_primitive,
                    &self.hamR,
                    &pair_blocks,
                    &pair_r,
                )?;
                accumulate_scaled_real_space_blocks(&mut blocks, &triple_blocks, &triple_r, 0.0)?;
            }
        }

        // Inspect the computed blocks, not a Cartesian sum of drive amplitudes:
        // exact-zero harmonics cannot contribute, but every tiny nonzero survives.
        let nonzero_harmonics = harmonic_cache
            .blocks
            .outer_iter()
            .map(|blocks| {
                blocks
                    .iter()
                    .any(|value| value.re != 0.0 || value.im != 0.0)
            })
            .collect::<Vec<_>>();
        let nonzero = |n| nonzero_harmonics[harmonic_cache.harmonic_index(n)];

        // First order: sum over n of comm_n/(n·ħΩ₀); omega0_ev carries
        // the ħΩ₀ energy (same convention as the legacy k-space path).
        let inverse_omega = drive.omega0_ev.recip();
        if options.order >= 1 && has_time_dependence {
            for n in 1..=harmonic_max {
                if !nonzero(n) || !nonzero(-n) {
                    continue;
                }
                let positive = harmonic_cache.harmonic_blocks(n);
                let negative = harmonic_cache.harmonic_blocks(-n);
                let (comm_blocks, comm_r) =
                    real_space_commutator(&positive, &negative, &self.hamR)?;
                let scale = inverse_omega / (n as f64);
                accumulate_scaled_real_space_blocks(&mut blocks, &comm_blocks, &comm_r, scale)?;
            }
        }

        // Second order in 1/(ħΩ₀): the two van Vleck nested-commutator
        // families.  Both signed harmonic sums are required; individual
        // summands are not Hermitian and therefore must not be symmetrized
        // before the complete sum is assembled.
        if options.order >= 2 && has_time_dependence {
            let inverse_omega_squared = inverse_omega * inverse_omega;
            let signed_harmonics = (-harmonic_max..=harmonic_max)
                .filter(|&n| n != 0 && (nonzero(n) || nonzero(-n)))
                .collect::<Vec<_>>();

            // Each fixed-m contribution is independent.  Accumulate one
            // private real-space map per Rayon job and merge maps only after
            // all nested commutators for that job are complete.  This exposes
            // the abundant harmonic-level parallelism of the O(omega^-2)
            // double sum without locking the output map in the hot loop.
            let accumulate_fixed_m = |partial: &mut RealSpaceBlockMap, m: isize| -> Result<()> {
                if !nonzero(m) {
                    return Ok(());
                }
                let h_m = harmonic_cache.harmonic_blocks(m);
                if nonzero(0) && nonzero(-m) {
                    let h_zero = harmonic_cache.harmonic_blocks(0);
                    let (inner, inner_r) =
                        real_space_commutator_with_supports(&h_zero, &self.hamR, &h_m, &self.hamR)?;
                    let h_minus_m = harmonic_cache.harmonic_blocks(-m);
                    let (outer, outer_r) = real_space_commutator_with_supports(
                        &h_minus_m, &self.hamR, &inner, &inner_r,
                    )?;
                    let scale = inverse_omega_squared / (2.0 * (m as f64).powi(2));
                    accumulate_scaled_real_space_blocks(partial, &outer, &outer_r, scale)?;
                }

                for &m_prime in &signed_harmonics {
                    if m_prime == m || !nonzero(-m_prime) || !nonzero(m_prime - m) {
                        continue;
                    }
                    let h_difference = harmonic_cache.harmonic_blocks(m_prime - m);
                    let (inner, inner_r) = real_space_commutator_with_supports(
                        &h_difference,
                        &self.hamR,
                        &h_m,
                        &self.hamR,
                    )?;
                    let h_minus_m_prime = harmonic_cache.harmonic_blocks(-m_prime);
                    let (outer, outer_r) = real_space_commutator_with_supports(
                        &h_minus_m_prime,
                        &self.hamR,
                        &inner,
                        &inner_r,
                    )?;
                    let scale = inverse_omega_squared / (3.0 * (m as f64) * (m_prime as f64));
                    accumulate_scaled_real_space_blocks(partial, &outer, &outer_r, scale)?;
                }
                Ok(())
            };

            let order_two_blocks = if signed_harmonics.len() > 1 && rayon::current_num_threads() > 1
            {
                let min_harmonics_per_job =
                    (signed_harmonics.len() / rayon::current_num_threads()).max(1);
                signed_harmonics
                    .par_iter()
                    .with_min_len(min_harmonics_per_job)
                    .try_fold(
                        RealSpaceBlockMap::new,
                        |mut partial, &m| -> Result<RealSpaceBlockMap> {
                            accumulate_fixed_m(&mut partial, m)?;
                            Ok(partial)
                        },
                    )
                    .try_reduce(RealSpaceBlockMap::new, |left, right| {
                        Ok(merge_real_space_block_maps(left, right))
                    })?
            } else {
                let mut partial = RealSpaceBlockMap::new();
                for &m in &signed_harmonics {
                    accumulate_fixed_m(&mut partial, m)?;
                }
                partial
            };
            blocks = merge_real_space_block_maps(blocks, order_two_blocks);
        }

        // Assemble the model on the merged support and enforce exact
        // real-space Hermiticity.
        let n_r_out = blocks.len();
        let mut ham = Array3::<Complex<f64>>::zeros((n_r_out, nsta, nsta));
        let mut ham_r = Array2::<isize>::zeros((n_r_out, DIM));
        for (i, (key, block)) in blocks.into_iter().enumerate() {
            for (a, v) in key.iter().enumerate() {
                ham_r[[i, a]] = *v;
            }
            ham.index_axis_mut(Axis(0), i).assign(&block);
        }
        place_origin_first(&mut ham, &mut ham_r)?;
        enforce_real_space_hermiticity(&mut ham, &ham_r)?;

        let mut model = Model::<SPIN, DIM, NoRMatrix>::tb_model(
            self.lat.clone(),
            self.orb.clone(),
            Some(self.atoms.clone()),
        )?;
        model.ham = ham;
        model.hamR = ham_r;
        model.orb_projection = self.orb_projection.clone();
        model.validate()?;
        Ok(model)
    }

    #[cfg(test)]
    fn floquet_effective_ham_onek_lattice<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        order: usize,
        harmonic_max: isize,
        has_time_dependence: bool,
        harmonic_cache: &FloquetHarmonicCache,
    ) -> Result<Array2<Complex<f64>>> {
        let cache_max = if has_time_dependence {
            effective_cache_max(order, harmonic_max)
                .expect("effective harmonic range was validated by the caller")
        } else {
            0
        };
        let harmonics: Vec<Array2<Complex<f64>>> = (-cache_max..=cache_max)
            .map(|harmonic| {
                self.floquet_cached_harmonic_onek(kvec, harmonic, Gauge::Lattice, harmonic_cache)
            })
            .collect();
        let harmonic =
            |index: isize| -> &Array2<Complex<f64>> { &harmonics[(index + cache_max) as usize] };
        let mut h_eff = harmonic(0).clone();

        let inverse_omega = drive.omega0_ev.recip();
        if order >= 1 && has_time_dependence {
            for n in 1..=harmonic_max {
                let comm = matrix_commutator(harmonic(n), harmonic(-n));
                accumulate_scaled_matrix(&mut h_eff, &comm, inverse_omega / (n as f64))?;
            }
        }

        if order >= 2 && has_time_dependence {
            let inverse_omega_squared = inverse_omega * inverse_omega;
            let signed_harmonics = (-harmonic_max..=harmonic_max)
                .filter(|index| *index != 0)
                .collect::<Vec<_>>();
            for &m in &signed_harmonics {
                let inner = matrix_commutator(harmonic(0), harmonic(m));
                let outer = matrix_commutator(harmonic(-m), &inner);
                accumulate_scaled_matrix(
                    &mut h_eff,
                    &outer,
                    inverse_omega_squared / (2.0 * (m as f64).powi(2)),
                )?;
            }
            for &m in &signed_harmonics {
                for &m_prime in &signed_harmonics {
                    if m_prime == m {
                        continue;
                    }
                    let inner = matrix_commutator(harmonic(m_prime - m), harmonic(m));
                    let outer = matrix_commutator(harmonic(-m_prime), &inner);
                    accumulate_scaled_matrix(
                        &mut h_eff,
                        &outer,
                        inverse_omega_squared / (3.0 * (m as f64) * (m_prime as f64)),
                    )?;
                }
            }
        }

        Ok(h_eff)
    }

    /// Build the `n`-th Fourier block `H^(n)(k)` from the precomputed cache.
    ///
    /// For each R-vector, multiplies the cached block `t * C_n(d)` by the Bloch
    /// phase `exp(2πi k·R)` via `zaxpy`.  The [`Gauge`] selects between the
    /// lattice gauge (raw Fourier sum) and the atom gauge (with orbital-position
    /// phases applied).
    #[cfg(test)]
    fn floquet_cached_harmonic_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        n: isize,
        gauge: Gauge,
        harmonic_cache: &FloquetHarmonicCache,
    ) -> Array2<Complex<f64>> {
        let nsta = self.nsta();
        let mut ham_n = Array2::<Complex<f64>>::zeros((nsta, nsta));
        let i_n = harmonic_cache.harmonic_index(n);
        let ham_n_slice = ham_n.as_slice_mut().unwrap();

        for i_r in 0..self.hamR.nrows() {
            let r_vec = self.hamR.row(i_r);
            let bloch = bloch_phase::<DIM, S>(&r_vec, kvec);
            let block = harmonic_cache.blocks.slice(s![i_n, i_r, .., ..]);
            crate::ndarray_lapack::zaxpy(bloch, block.as_slice().unwrap(), ham_n_slice);
        }

        match gauge {
            Gauge::Lattice => ham_n,
            Gauge::Atom => self.apply_atom_gauge(kvec, ham_n),
        }
    }

    fn link_displacement_cartesian(
        &self,
        i_orb: usize,
        j_orb: usize,
        r_vec: &ArrayView1<'_, isize>,
    ) -> Array1<f64> {
        let mut frac = Array1::<f64>::zeros(DIM);
        for a in 0..DIM {
            frac[a] = r_vec[a] as f64 + self.orb[[j_orb, a]] - self.orb[[i_orb, a]];
        }
        frac.dot(&self.lat)
    }

    /// Real-space blocks of `a_i partial_{k_i} H_0` in the fixed Wannier
    /// (atomic) gauge, with Cartesian momentum derivatives.
    fn cartesian_gradient_contraction_blocks(
        &self,
        amplitude: &Array1<Complex<f64>>,
    ) -> Result<Array3<Complex<f64>>> {
        debug_assert_eq!(amplitude.len(), DIM);
        let nsta = self.nsta();
        let norb = self.norb();
        let mut blocks = Array3::<Complex<f64>>::zeros(self.ham.raw_dim());
        for i_r in 0..self.hamR.nrows() {
            let r_vec = self.hamR.row(i_r);
            for i in 0..nsta {
                for j in 0..nsta {
                    let hopping = self.ham[[i_r, i, j]];
                    if hopping.re == 0.0 && hopping.im == 0.0 {
                        continue;
                    }
                    let d_cart = self.link_displacement_cartesian(i % norb, j % norb, &r_vec);
                    let projection = amplitude
                        .iter()
                        .zip(d_cart.iter())
                        .fold(Complex::new(0.0, 0.0), |sum, (a, d)| sum + *a * *d);
                    let value = Complex::new(0.0, 1.0) * projection * hopping;
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(TbError::Other(format!(
                            "floquet_effective_q_model Cartesian gradient overflowed at hopping \
                             row {i_r}, state pair ({i}, {j})"
                        )));
                    }
                    blocks[[i_r, i, j]] = value;
                }
            }
        }
        Ok(blocks)
    }

    #[cfg(test)]
    fn apply_atom_gauge<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        mut ham: Array2<Complex<f64>>,
    ) -> Array2<Complex<f64>> {
        let nsta = self.nsta();
        let norb = self.norb();
        let mut phase_orb = Array1::<Complex<f64>>::zeros(norb);
        for i in 0..norb {
            let mut tau_dot_k = 0.0;
            for a in 0..DIM {
                tau_dot_k += self.orb[[i, a]] * kvec[a];
            }
            phase_orb[i] = Complex::new(0.0, TAU * tau_dot_k).exp();
        }

        let mut phase = Array1::<Complex<f64>>::zeros(nsta);
        phase.slice_mut(s![..norb]).assign(&phase_orb);
        if SPIN {
            phase.slice_mut(s![norb..]).assign(&phase_orb);
        }

        for i in 0..nsta {
            let left = phase[i].conj();
            for j in 0..nsta {
                ham[[i, j]] *= left * phase[j];
            }
        }
        ham
    }
}

#[inline]
fn floquet_basis_indices<const SPIN: bool>(
    nsta: usize,
    norb: usize,
    n_sector: usize,
) -> Vec<Vec<usize>> {
    (0..n_sector)
        .map(|sector_index| {
            (0..nsta)
                .map(|state_index| {
                    floquet_basis_index::<SPIN>(sector_index, state_index, nsta, norb, n_sector)
                })
                .collect()
        })
        .collect()
}

#[inline]
fn floquet_basis_index<const SPIN: bool>(
    sector_index: usize,
    state_index: usize,
    nsta: usize,
    norb: usize,
    n_sector: usize,
) -> usize {
    if SPIN {
        let spin = state_index / norb;
        let orbital = state_index % norb;
        spin * n_sector * norb + sector_index * norb + orbital
    } else {
        sector_index * nsta + state_index
    }
}

#[inline]
/// Fold a finite energy into `[-omega0_ev/2, omega0_ev/2)`.
/// The frequency must be finite and positive, as validated by Floquet APIs.
pub fn fold_quasienergy(energy: f64, omega0_ev: f64) -> f64 {
    // A signed remainder preserves tiny negative energies even when omega is
    // huge. Compare the two distances instead of halving omega: the latter
    // can underflow or round the half-zone incorrectly for subnormal inputs.
    let reduced = energy % omega0_ev;
    if reduced >= 0.0 && reduced >= omega0_ev - reduced {
        reduced - omega0_ev
    } else if reduced < 0.0 && -reduced > omega0_ev + reduced {
        reduced + omega0_ev
    } else {
        reduced
    }
}

#[cfg(test)]
fn validate_floquet_input<
    const DIM: usize,
    S: Data<Elem = f64>,
    R: RMatrixData,
    const SPIN: bool,
>(
    model: &Model<SPIN, DIM, R>,
    kvec: &ArrayBase<S, Ix1>,
    drive: &FloquetDrive,
    trunc: &FloquetTruncation,
) -> Result<()> {
    if kvec.len() != DIM {
        return Err(TbError::KVectorLengthMismatch {
            expected: DIM,
            actual: kvec.len(),
        });
    }
    if kvec.iter().any(|x| !x.is_finite()) {
        return Err(TbError::Other(
            "Floquet k coordinates must be finite".into(),
        ));
    }
    if model.lat.nrows() != DIM || model.lat.ncols() != DIM {
        return Err(TbError::InvalidArrayShape {
            expected: vec![DIM, DIM],
            found: vec![model.lat.nrows(), model.lat.ncols()],
        });
    }
    validate_floquet_drive::<DIM>(drive)?;
    validate_floquet_truncation(trunc)
}

fn validate_floquet_drive<const DIM: usize>(drive: &FloquetDrive) -> Result<()> {
    if !drive.omega0_ev.is_finite() || drive.omega0_ev <= 0.0 {
        return Err(TbError::InvalidEnergyRange {
            min: 0.0,
            max: drive.omega0_ev,
        });
    }
    for (im, mode) in drive.modes.iter().enumerate() {
        if mode.a_complex.len() != DIM {
            return Err(TbError::DimensionMismatch {
                context: format!("FloquetDrive.modes[{im}].a_complex"),
                expected: DIM,
                found: mode.a_complex.len(),
            });
        }
        if mode
            .a_complex
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(TbError::Other(format!(
                "FloquetDrive.modes[{im}].a_complex contains non-finite values"
            )));
        }
    }
    Ok(())
}

fn validate_floquet_truncation(trunc: &FloquetTruncation) -> Result<()> {
    if trunc.n_max < 0 {
        return Err(TbError::Other(format!(
            "FloquetTruncation.n_max must be non-negative, got {}",
            trunc.n_max
        )));
    }
    trunc.n_sector()?;
    trunc
        .n_max
        .checked_mul(4)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| TbError::Other("Floquet harmonic range overflows".into()))?;
    Ok(())
}

// Check every Sambe array extent before allocating, and check that each link
// is either exactly representable by the Bessel ladder or resolvable by the
// per-link time grid it falls back to.  The Sambe path takes no time-grid
// parameter any more, so there is nothing to compare a sampling count against;
// what remains is the "never return silently aliased coefficients" guarantee.
fn validate_sambe_allocation<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    drive: &FloquetDrive,
    trunc: &FloquetTruncation,
) -> Result<()> {
    model.validate()?;
    let n_sector = trunc.n_sector()?;
    let total = model
        .nsta()
        .checked_mul(n_sector)
        .ok_or_else(|| TbError::Other("Floquet matrix dimension overflows".into()))?;
    let harmonic_count = (4 * trunc.n_max + 1) as usize; // checked by truncation validation
    let n_r = model
        .hamR
        .nrows()
        .checked_add(1)
        .ok_or_else(|| TbError::Other("Floquet translation count overflows".into()))?;
    for shape in [
        vec![n_r, total, total],
        vec![harmonic_count, n_r, model.nsta(), model.nsta()],
        vec![total, DIM],
    ] {
        shape
            .into_iter()
            .try_fold(size_of::<Complex<f64>>(), |bytes, n| bytes.checked_mul(n))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                TbError::Other("Floquet array size exceeds the addressable range".into())
            })?;
    }
    if model
        .ham
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(TbError::Other("Floquet Hamiltonian must be finite".into()));
    }
    validate_link_resolvability(model, drive, -2 * trunc.n_max, 2 * trunc.n_max)
}

fn validate_floquet_time_samples(n_time: usize) -> Result<()> {
    if n_time == 0 {
        return Err(TbError::Other(
            "Floquet n_time must be positive".to_string(),
        ));
    }
    Ok(())
}

fn effective_harmonic_max(options: &FloquetEffectiveOptions) -> Result<isize> {
    let harmonic_max = options.harmonic_max;
    if harmonic_max < 0 {
        return Err(TbError::Other(format!(
            "FloquetEffectiveOptions.harmonic_max must be non-negative, got {harmonic_max}"
        )));
    }
    Ok(harmonic_max)
}

fn effective_cache_max(order: usize, harmonic_max: isize) -> Result<isize> {
    match order {
        0 => Ok(0),
        1 => Ok(harmonic_max),
        2 => harmonic_max.checked_mul(2).ok_or_else(|| {
            TbError::Other(
                "FloquetEffectiveOptions.harmonic_max is too large for the order-2 harmonic range"
                    .to_string(),
            )
        }),
        _ => unreachable!("the effective order must be validated first"),
    }
}

fn validate_effective_cache_layout(cache_max: isize, n_r: usize, nsta: usize) -> Result<()> {
    let harmonic_count = cache_max
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            TbError::Other("the effective harmonic range is too large to index safely".to_string())
        })?;
    let elements = harmonic_count
        .checked_mul(n_r)
        .and_then(|value| value.checked_mul(nsta))
        .and_then(|value| value.checked_mul(nsta))
        .ok_or_else(|| {
            TbError::Other(
                "the effective harmonic cache shape exceeds addressable memory".to_string(),
            )
        })?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<Complex<f64>>())
        .ok_or_else(|| {
            TbError::Other(
                "the effective harmonic cache byte size exceeds addressable memory".to_string(),
            )
        })?;
    if bytes > isize::MAX as usize {
        return Err(TbError::Other(
            "the effective harmonic cache byte size exceeds isize::MAX".to_string(),
        ));
    }
    Ok(())
}

fn drive_has_ac_components(drive: &FloquetDrive) -> bool {
    // Cartesian cancellation cannot certify link-local cancellation: a rounded
    // zero could erase an ordinary physical phase. Let the scalar backend sum
    // all nonzero AC modes after projection instead.
    drive
        .modes
        .iter()
        .any(|mode| mode.harmonic != 0 && mode.a_complex.iter().any(|a| a.re != 0.0 || a.im != 0.0))
}

fn static_component_drive(drive: &FloquetDrive) -> FloquetDrive {
    // Keep DC projections mode-local: Cartesian summation can discard a
    // small physical phase before a large cancelling vector is projected.
    let modes = drive
        .modes
        .iter()
        .filter(|mode| mode.harmonic == 0)
        .map(|mode| LightMode::new(0, mode.a_complex.mapv(|a| Complex::new(a.re, 0.0))))
        .collect();
    FloquetDrive::with_modes(drive.omega0_ev, modes)
}

#[cfg(test)]
fn validate_effective_options<const DIM: usize>(
    k_mesh: &[usize; DIM],
    options: &FloquetEffectiveOptions,
) -> Result<()> {
    if options.order > 2 {
        return Err(TbError::Other(format!(
            "Floquet effective order {} is not implemented; supported orders are 0, 1, and 2",
            options.order
        )));
    }
    for (axis, &n) in k_mesh.iter().enumerate() {
        if n == 0 {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.k_mesh[{axis}] must be positive"
            )));
        }
    }
    effective_harmonic_max(options)?;
    if let Some(target_ham_r) = &options.target_hamR {
        validate_target_hamr::<DIM>(target_ham_r)?;
    }
    Ok(())
}

#[cfg(test)]
fn validate_target_hamr<const DIM: usize>(target_ham_r: &Array2<isize>) -> Result<()> {
    if target_ham_r.ncols() != DIM {
        return Err(TbError::InvalidArrayShape {
            expected: vec![target_ham_r.nrows(), DIM],
            found: vec![target_ham_r.nrows(), target_ham_r.ncols()],
        });
    }
    if target_ham_r.nrows() == 0 {
        return Err(TbError::Other(
            "target_hamR must contain at least one R vector".to_string(),
        ));
    }
    for i_r in 0..target_ham_r.nrows() {
        let r = target_ham_r.row(i_r).to_owned();
        if (0..i_r).any(|j_r| {
            target_ham_r
                .row(j_r)
                .iter()
                .zip(r.iter())
                .all(|(left, right)| left == right)
        }) {
            return Err(TbError::Other(format!(
                "target_hamR contains the duplicate vector R={:?}",
                r.to_vec()
            )));
        }
        let neg_r = r.mapv(|x| -x);
        if find_R(target_ham_r, &neg_r).is_none() {
            return Err(TbError::MissingHermitianConjugateHopping { r });
        }
    }
    Ok(())
}

#[cfg(test)]
#[inline]
fn matrix_commutator(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    a.dot(b) - b.dot(a)
}

#[cfg(test)]
fn accumulate_scaled_matrix(
    target: &mut Array2<Complex<f64>>,
    source: &Array2<Complex<f64>>,
    scale: f64,
) -> Result<()> {
    if source
        .iter()
        .all(|value| value.re == 0.0 && value.im == 0.0)
    {
        return Ok(());
    }
    if !scale.is_finite() {
        return Err(TbError::Other(
            "a van Vleck frequency scaling factor is non-finite; increase omega0_ev or lower the requested order"
                .to_string(),
        ));
    }
    target.scaled_add(Complex::new(scale, 0.0), source);
    Ok(())
}

#[cfg(test)]
fn bloch_phase<const DIM: usize, S: Data<Elem = f64>>(
    r_vec: &ArrayView1<'_, isize>,
    kvec: &ArrayBase<S, Ix1>,
) -> Complex<f64> {
    let mut r_dot_k = 0.0;
    for a in 0..DIM {
        r_dot_k += r_vec[a] as f64 * kvec[a];
    }
    Complex::new(0.0, TAU * r_dot_k).exp()
}

#[cfg(test)]
fn inverse_bloch_phase<const DIM: usize, S: Data<Elem = f64>>(
    r_vec: &ArrayView1<'_, isize>,
    kvec: &ArrayBase<S, Ix1>,
) -> Complex<f64> {
    bloch_phase::<DIM, S>(r_vec, kvec).conj()
}

#[cfg(test)]
fn floquet_uniform_kmesh<const DIM: usize>(mesh: &[usize; DIM]) -> Vec<Array1<f64>> {
    let n_total = mesh.iter().product();
    let mut points = Vec::with_capacity(n_total);
    for mut linear in 0..n_total {
        let mut k = Array1::<f64>::zeros(DIM);
        for a in (0..DIM).rev() {
            let n = mesh[a];
            let i = linear % n;
            linear /= n;
            k[a] = (i as f64) / (n as f64);
        }
        points.push(k);
    }
    points
}

/// Normalize a 3D vector to unit length.
///
/// Scale-free by construction: the largest component is divided out before the
/// squares are summed, so a direction given in units that overflow (`1e200`) or
/// underflow (`1e-200`) squares normalizes like any other ray.  Only an exactly
/// zero vector and a non-finite component are errors.
fn normalize3(v: &Array1<f64>) -> Result<Array1<f64>> {
    if v.iter().any(|x| !x.is_finite()) {
        return Err(TbError::Other(
            "Cannot normalize a non-finite 3D vector".to_string(),
        ));
    }
    // Scale by the largest component before squaring: a finite component can
    // still overflow (1e200^2 = inf) or underflow (1e-200^2 = 0) on its own,
    // and either would silently yield a zero or NaN direction.
    let scale = v.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
    if scale == 0.0 {
        return Err(TbError::Other(
            "Cannot normalize a zero-length 3D vector".to_string(),
        ));
    }
    let scaled = v.mapv(|x| x / scale);
    let norm = (scaled[0] * scaled[0] + scaled[1] * scaled[1] + scaled[2] * scaled[2]).sqrt();
    Ok(scaled.mapv(|x| x / norm))
}

fn cross3(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

#[cfg(test)]
mod tests;
