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
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};

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

type RealSpaceBlocks = (Vec<Array2<Complex<f64>>>, Array2<isize>);
type RealSpaceBlockMap = std::collections::BTreeMap<Vec<isize>, Array2<Complex<f64>>>;

/// Keep public real-space models compatible with origin-first consumers.
/// Other support rows retain their relative order.
fn place_origin_first(ham: &mut Array3<Complex<f64>>, ham_r: &mut Array2<isize>) -> Result<()> {
    let zero = Array1::zeros(ham_r.ncols());
    let origin = match find_R(ham_r, &zero) {
        Some(index) => index,
        None => {
            ham.push(
                Axis(0),
                Array2::zeros((ham.len_of(Axis(1)), ham.len_of(Axis(2)))).view(),
            )?;
            ham_r.push_row(zero.view())?;
            ham_r.nrows() - 1
        }
    };
    if origin != 0 {
        let order: Vec<_> = std::iter::once(origin)
            .chain((0..ham_r.nrows()).filter(|&index| index != origin))
            .collect();
        *ham = ham.select(Axis(0), &order);
        *ham_r = ham_r.select(Axis(0), &order);
    }
    Ok(())
}

trait RealSpaceBlockSource: Sync {
    fn nblocks(&self) -> usize;
    fn block(&self, index: usize) -> ArrayView2<'_, Complex<f64>>;
}

impl<S> RealSpaceBlockSource for ArrayBase<S, Ix3>
where
    S: Data<Elem = Complex<f64>> + Sync,
{
    fn nblocks(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn block(&self, index: usize) -> ArrayView2<'_, Complex<f64>> {
        self.index_axis(Axis(0), index)
    }
}

impl RealSpaceBlockSource for Vec<Array2<Complex<f64>>> {
    fn nblocks(&self) -> usize {
        self.len()
    }

    fn block(&self, index: usize) -> ArrayView2<'_, Complex<f64>> {
        self[index].view()
    }
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

/// Backend selection for the Peierls Fourier coefficients `C_n(d)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeierlsFourierMethod {
    /// Numerical DFT on a uniform time grid.  Compiled for test builds only:
    /// production has exactly one coefficient backend, and the tests use this
    /// variant to cross-check it against an independent oracle
    /// ([`crate::floquet_test`]).
    #[cfg(test)]
    TimeGrid { n_time: usize },
    /// Generalized Bessel expansion via sequential one-mode convolutions.
    /// Exact and independent of `n_time`, but restricted to per-mode
    /// projections `R_α = |a_α·d| ≤` [`MAX_BESSEL_ARG`], and the same bound on
    /// the coherent sum for the single-harmonic closed form
    /// ([`MAX_BESSEL_ARG_CLOSED_FORM`]); the cache falls back to a self-sized
    /// per-link time-grid DFT (`fallback_time_grid_coeffs`) beyond that.
    Bessel {
        /// Minimum number of Bessel orders beyond `⌈R_α⌉` (the adaptive tail
        /// check may push the cutoff higher).
        cutoff_margin: isize,
    },
}

/// Precomputed `t_ij(R) * C_n(d)` for all harmonics `n ∈ [harmonic_min, harmonic_max]`.
///
/// `blocks` has shape `(harmonic_count, n_r, nsta, nsta)` where `harmonic_count = harmonic_max - harmonic_min + 1`.
/// Index `[i_n, i_r, i, j]` stores the `n = harmonic_min + i_n` Fourier component of hopping
/// from orbital `j` in cell `R = hamR[i_r]` to orbital `i` at the origin.
/// This is independent of `k` and reusable across the entire k-mesh.
struct FloquetHarmonicCache {
    harmonic_min: isize,
    harmonic_max: isize,
    blocks: Array4<Complex<f64>>,
}

impl FloquetHarmonicCache {
    #[inline]
    fn harmonic_index(&self, n: isize) -> usize {
        debug_assert!(
            n >= self.harmonic_min && n <= self.harmonic_max,
            "Floquet harmonic n={n} is outside cached range [{}, {}]",
            self.harmonic_min,
            self.harmonic_max
        );
        (n - self.harmonic_min) as usize
    }

    #[inline]
    fn harmonic_blocks(&self, n: isize) -> ArrayView3<'_, Complex<f64>> {
        self.blocks.index_axis(Axis(0), self.harmonic_index(n))
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

    /// Build the harmonic cache: `t_ij(R) * C_n(d)` for all `n ∈ [harmonic_min, harmonic_max]`.
    ///
    /// Returns a [`FloquetHarmonicCache`] whose `blocks[harmonic_index(n), i_r, i, j]`
    /// stores the `n`-th Fourier coefficient of the Peierls-dressed hopping from
    /// orbital `j` at cell `R = hamR[i_r]` to orbital `i` at the origin.  The
    /// cache is consumed by the real-space constructors. For a k-mesh, build
    /// `floquet_model` once and reuse its ordinary Hamiltonian and band solvers.
    ///
    /// When `drive.modes` is empty (static limit), the zero-frequency block is
    /// set to the original `ham` directly.
    fn floquet_harmonic_cache(
        &self,
        drive: &FloquetDrive,
        harmonic_min: isize,
        harmonic_max: isize,
        method: &PeierlsFourierMethod,
    ) -> FloquetHarmonicCache {
        let nsta = self.nsta();
        let norb = self.norb();
        let n_r = self.hamR.nrows();
        let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
        let mut blocks = Array4::<Complex<f64>>::zeros((harmonic_count, n_r, nsta, nsta));

        if drive.modes.is_empty() {
            if harmonic_min <= 0 && 0 <= harmonic_max {
                blocks
                    .slice_mut(s![(0 - harmonic_min) as usize, .., .., ..])
                    .assign(&self.ham);
            }
            return FloquetHarmonicCache {
                harmonic_min,
                harmonic_max,
                blocks,
            };
        }

        // Phase 1: collect the DISTINCT link displacements among non-zero
        // hoppings.  Spin copies of the same orbital pair share the same
        // d, so this deduplicates the coefficient computation (a 4x saving
        // for spinful models).
        // Distinct-link map keyed by the bit pattern of the Cartesian
        // displacement (a fixed-size array avoids a heap allocation per
        // hopping entry).
        let mut d_index = std::collections::HashMap::<[u64; DIM], usize>::new();
        let mut unique_d = Vec::<Array1<f64>>::new();
        let mut entries = Vec::<(usize, usize, usize, usize)>::new(); // (i_r, i, j, d_idx)
        for i_r in 0..n_r {
            let r_vec = self.hamR.row(i_r);
            for i in 0..nsta {
                for j in 0..nsta {
                    if self.ham[[i_r, i, j]].re == 0.0 && self.ham[[i_r, i, j]].im == 0.0 {
                        continue;
                    }
                    let d_cart = self.link_displacement_cartesian(i % norb, j % norb, &r_vec);
                    let mut key = [0_u64; DIM];
                    for a in 0..DIM {
                        key[a] = d_cart[a].to_bits();
                    }
                    let index = *d_index.entry(key).or_insert_with(|| {
                        unique_d.push(d_cart);
                        unique_d.len() - 1
                    });
                    entries.push((i_r, i, j, index));
                }
            }
        }

        // Phase 2: coefficients per distinct d (parallel).  The Bessel backend
        // sizes its own per-link fallback grid from the link's bandwidth, so
        // production allocates no eager grid at all; the reference time grid
        // exists only in test builds.
        #[cfg(test)]
        let time_grid = match method {
            PeierlsFourierMethod::TimeGrid { n_time } => {
                Some(crate::floquet_test::FloquetTimeGrid::new(
                    drive,
                    *n_time,
                    harmonic_min,
                    harmonic_max,
                ))
            }
            PeierlsFourierMethod::Bessel { .. } => None,
        };
        // Per-call warn-once flag: the parallel loop below may hit the
        // fallback branch for many links, but the user only needs one
        // message per cache build.
        let fallback_warned = AtomicBool::new(false);
        let fallback_clamped = AtomicBool::new(false);
        let fallback_saturated = AtomicBool::new(false);
        let coeffs_per_d: Vec<Array1<Complex<f64>>> = unique_d
            .par_iter()
            .map(|d| match method {
                PeierlsFourierMethod::Bessel { cutoff_margin } => {
                    match bessel_peierls_coeffs(
                        d,
                        drive,
                        harmonic_min,
                        harmonic_max,
                        *cutoff_margin,
                    ) {
                        Ok(coeffs) => coeffs,
                        Err(error) => {
                            if !fallback_warned.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "Bessel backend unavailable for some links \
                                     ({error}); falling back to a per-link \
                                     time-grid DFT"
                                );
                            }
                            fallback_time_grid_coeffs(
                                d,
                                drive,
                                harmonic_min,
                                harmonic_max,
                                &fallback_clamped,
                                &fallback_saturated,
                            )
                        }
                    }
                }
                #[cfg(test)]
                PeierlsFourierMethod::TimeGrid { .. } => {
                    let time_grid = time_grid
                        .as_ref()
                        .expect("shared time grid is built for the TimeGrid backend");
                    Array1::from(crate::floquet_test::peierls_fourier_coeffs(
                        d,
                        harmonic_min,
                        harmonic_max,
                        drive,
                        time_grid,
                    ))
                }
            })
            .collect();

        // Phase 3: fill the blocks.
        for (i_r, i, j, d_index) in entries {
            let t = self.ham[[i_r, i, j]];
            for (i_n, coeff) in coeffs_per_d[d_index].iter().enumerate() {
                if coeff.re != 0.0 || coeff.im != 0.0 {
                    blocks[[i_n, i_r, i, j]] = t * coeff;
                }
            }
        }

        FloquetHarmonicCache {
            harmonic_min,
            harmonic_max,
            blocks,
        }
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

// Every link must be either exactly representable by the Bessel backend or
// resolvable by the per-link time grid it falls back to.  Both public paths
// (Sambe and the effective model) share this check, so a drive that cannot be
// represented exactly anywhere is refused instead of being answered from a
// clamped, aliasing grid.
#[cfg(test)]
thread_local! {
    static LINK_RESOLVABILITY_SCANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn validate_link_resolvability<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
) -> Result<()> {
    if drive.modes.is_empty() {
        return Ok(());
    }
    let span = harmonic_max.saturating_sub(harmonic_min).saturating_add(1) as usize;
    let mut checked = std::collections::HashSet::<[u64; DIM]>::new();
    // Drive-level count, which chooses the backend's amplitude cap.
    let drive_carriers = drive.modes.iter().filter(|mode| mode.harmonic != 0).count();
    for ((r, i, j), hopping) in model.ham.indexed_iter() {
        if hopping.re == 0.0 && hopping.im == 0.0 {
            continue;
        }
        let d = model.link_displacement_cartesian(
            i % model.norb(),
            j % model.norb(),
            &model.hamR.row(r),
        );
        if d.iter().any(|x| !x.is_finite()) {
            return Err(TbError::Other(
                "Floquet link displacement is not finite".into(),
            ));
        }
        // Skip repeated links before projecting any modes; exact bit keys keep
        // the same numerical equivalence as the harmonic cache (no tolerance).
        if !checked.insert(std::array::from_fn(|a| d[a].to_bits())) {
            continue;
        }
        #[cfg(test)]
        LINK_RESOLVABILITY_SCANS.with(|count| count.set(count.get() + 1));
        // Mirror the backend's own applicability test.  It groups modes
        // differently depending on how many distinct harmonics survive:
        //
        // * a drive whose nonzero projections collapse to one coherent harmonic
        //   takes the closed form.  Its only precondition is that the *summed*
        //   projection fits the cap, and it allocates no convolution window at
        //   all, so no window bound applies;
        // * exactly two carriers may enumerate resonances without a window;
        //   the remaining drives convolve individual AC modes and bound their
        //   work window. DC never consumes an AC cap, ladder or window.
        //
        // Checking per mode in both cases would let a split amplitude (a + a)
        // slip past the cap; checking the sum in both would reject drives the
        // convolution handles.  Modes whose projection is exactly zero are
        // dropped first, as the backend drops them.
        // Walk the modes exactly like `single_harmonic_closed_form` does: it
        // keeps one running coherent sum and bails out to the convolution at the
        // *first* harmonic change — including when a later mode would have
        // cancelled that sum back to zero.  Accumulating per harmonic instead
        // would call such a drive exact while the backend convolves it.
        let mut single: Option<(isize, Complex<f64>)> = None;
        let mut per_mode = Vec::<(isize, f64)>::new();
        let mut several_harmonics = false;
        let mut carriers = 0_usize;
        let mut dc_re = 0.0_f64;
        for mode in &drive.modes {
            if mode.harmonic == 0 {
                dc_re += mode
                    .a_complex
                    .iter()
                    .zip(&d)
                    .map(|(a, x)| a.re * x)
                    .sum::<f64>();
                continue;
            }
            let phase: Complex<f64> = mode.a_complex.iter().zip(&d).map(|(a, x)| a * x).sum();
            if !phase.re.is_finite() || !phase.im.is_finite() {
                return Err(TbError::Other(
                    "Floquet link phase amplitude is not finite".into(),
                ));
            }
            if phase.re == 0.0 && phase.im == 0.0 {
                continue;
            }
            // A single coherent carrier only uses the summed projection. An
            // individual norm may overflow even when that sum cancels exactly.
            per_mode.push((mode.harmonic, phase.norm()));
            carriers += 1;
            match &mut single {
                Some((harmonic, total)) if *harmonic == mode.harmonic => *total += phase,
                Some(_) => several_harmonics = true,
                None => single = Some((mode.harmonic, phase)),
            }
        }
        if !dc_re.is_finite() {
            return Err(TbError::Other(
                "Floquet link static phase is not finite".into(),
            ));
        }
        let exact = if !several_harmonics {
            // One coherent harmonic (or none at all): the closed form is exact
            // for any amplitude whose coherent sum fits its cap, and an exactly
            // cancelling sum never reaches the ladder.
            match single {
                Some((_, z)) => {
                    z.re == 0.0 && z.im == 0.0 || z.norm() <= MAX_BESSEL_ARG_CLOSED_FORM
                }
                None => true,
            }
        } else {
            // Mirror the backend: the enumeration runs on exactly two carriers,
            // counted exactly as the backend counts them (nonzero harmonic AND
            // nonzero link projection), and only with the drive-level count in
            // agreement.  Its scan is the *smaller carrier's* cutoff, so static
            // modes must not enter the minimum.
            let carrier_cap = if carriers == drive_carriers && carriers == 2 {
                per_mode
                    .iter()
                    .filter(|(harmonic, _)| *harmonic != 0)
                    .all(|(_, r)| *r <= MAX_BESSEL_ARG_CLOSED_FORM)
            } else {
                false
            };
            if carrier_cap {
                // An upper bound on that cutoff: the ladder seeds at
                // `⌈r⌉ + margin + 8 + 16·r^{1/3}` and at most doubles a bounded
                // number of times, so `4·(⌈r⌉ + 64 + 16·r^{1/3})` covers every
                // case the sweep can reach.  Overestimating only makes the
                // validator stricter than the backend, never looser.
                let scan = per_mode
                    .iter()
                    .filter(|(harmonic, _)| *harmonic != 0)
                    .map(|(_, r)| {
                        4_usize.saturating_mul(
                            (r.ceil() as usize)
                                .saturating_add(64)
                                .saturating_add((16.0 * r.cbrt()) as usize),
                        )
                    })
                    .min()
                    .unwrap_or(0);
                span.saturating_mul(2_usize.saturating_mul(scan) + 1) <= MAX_BESSEL_ENUM_WORK
            } else {
                // Same conservative cutoff estimate as the enumeration branch:
                // ceil(r) + 64 alone understates M at large r (measured +230 at
                // r = 16000), so the window would look smaller than the backend's.
                let cutoff = |r: f64| {
                    4_usize.saturating_mul(
                        (r.ceil() as usize)
                            .saturating_add(64)
                            .saturating_add((16.0 * r.cbrt()) as usize),
                    )
                };
                per_mode.iter().all(|(_, r)| *r <= MAX_BESSEL_ARG)
                    && per_mode.iter().fold(span, |acc, (harmonic, r)| {
                        acc.saturating_add(
                            2_usize
                                .saturating_mul(harmonic.unsigned_abs())
                                .saturating_mul(cutoff(*r)),
                        )
                    }) <= MAX_BESSEL_WINDOW
            }
        };
        if exact {
            continue;
        }
        // Otherwise the per-link grid must resolve the link's spectrum; refuse
        // the whole call rather than return coefficients that alias.  The
        // requested range counts too: an n-point DFT only returns true `C_n`
        // for `|n| < n/2`.
        let grid = fallback_grid_size(drive, &d, harmonic_min, harmonic_max);
        let needed = grid.required.max(grid.request_range);
        if needed > FALLBACK_GRID_MAX {
            // The cheap cutoff upper bound is only a sufficient test. Before
            // rejecting, let the actual backend certify its adaptive cutoffs
            // and work budget; this expensive probe runs only on this boundary.
            if bessel_peierls_coeffs(&d, drive, harmonic_min, harmonic_max, BESSEL_CUTOFF_MARGIN)
                .is_ok()
            {
                continue;
            }
            return Err(TbError::Other(format!(
                "Floquet drive needs {} samples on a link the Bessel backend cannot \
                 represent exactly (coherent sum ≤ {MAX_BESSEL_ARG_CLOSED_FORM}, convolution \
                 operand ≤ {MAX_BESSEL_ARG} and window ≤ {MAX_BESSEL_WINDOW}); the per-link \
                 grid cap is {FALLBACK_GRID_MAX}",
                needed,
            )));
        }
    }
    Ok(())
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

/// Integer-order Bessel function of the first kind, `J_m(r)`, for real
/// non-negative arguments.
///
/// Thin wrapper over [`puruspe::Jn`] (pure Rust special-functions crate,
/// MIT/Apache-2.0).  Negative orders use the symmetry
///
/// ```math
/// J_{-m}(r) = (-1)^m J_m(r).
/// ```
///
/// It is accurate to a few ulp around `|m| ~ r` — which is where
/// [`bessel_backward_sweep`] anchors its normalization — and over the ranges
/// the tests probe, cross-checked against an independent Miller
/// downward-recurrence reference, tabulated NIST values and `mpmath` at 50
/// digits.  It is *not* a general oracle for `|m| >> r`: measured
/// `Jn(4096, 0.5) = 1.7e34` where the true value underflows to zero, and
/// `Jn(n ≥ 2, r) = 0` for `r ≤ 4.2e-154`.  The ladder therefore never asks it
/// for such orders, and tests that need them build their own reference.
///
/// # Arguments
/// * `m` - integer order (may be negative).
/// * `r` - non-negative finite argument (call sites pass `|a·d|`).
///
/// # Panics
/// Panics on a negative or non-finite argument, both outside the Floquet
/// backend's domain.
pub(crate) fn bessel_j(m: isize, r: f64) -> f64 {
    assert!(
        r.is_finite() && r >= 0.0,
        "bessel_j expects a non-negative finite argument, got {r}"
    );
    if m < 0 {
        // J_{-m}(r) = (-1)^m J_m(r)
        return if m.rem_euclid(2) == 1 {
            -bessel_j(-m, r)
        } else {
            bessel_j(-m, r)
        };
    }
    if r == 0.0 {
        return if m == 0 { 1.0 } else { 0.0 };
    }
    puruspe::Jn(m as u32, r)
}

/// Production tail margin shared by coefficient generation and its validation probe.
const BESSEL_CUTOFF_MARGIN: isize = 6;

/// Largest per-mode link amplitude `R_α = |a_α·d|` handled by the one-mode
/// convolution path.  Above it [`bessel_peierls_coeffs`] reports an error and
/// the harmonic cache falls back to a per-link time-grid DFT.
///
/// The cap is a **cost** contract, not a mathematical or accuracy limit: the
/// generalized Bessel expansion converges for every finite `R`, and the
/// backward-recurrence ladder below keeps machine precision far beyond this
/// value (measured `max|ΔJ| ≤ 5e-14` at `R = 4000` against independent
/// `scipy`/`mpmath` references).  It exists because the one-mode convolution
/// costs `O(M·W)` with `M ≈ R` and window `W ≈ 2·Σ_α |l_α|M_α`, while the DFT
/// it replaces costs `O(N·K)` with `N ≈ 2R`: at `R = 128` and a five-bin
/// requested range the former is already ~50x the work, and the ratio grows
/// linearly with `R`.
const MAX_BESSEL_ARG: f64 = 128.0;

/// Largest link amplitude `R = |Σ_α a_α·d|` handled by the single-harmonic
/// closed form, i.e. by a drive whose nonzero projections collapse to one
/// coherent harmonic.
///
/// That path allocates no convolution window — one `O(R)` ladder plus the
/// `O(K)` requested bins — so its cost is `O(R + K)` against the DFT's
/// `O(R·K)`, and it reaches two orders of magnitude further than the
/// convolution path at ~25 µs per distinct link.  Above this cap the link
/// falls back to a per-link time grid.  The ladder stays at machine precision
/// across the whole range (checked against 60-digit references).
const MAX_BESSEL_ARG_CLOSED_FORM: f64 = 16384.0;

/// Hard cap on the order range evaluated for one link.  A tail that still
/// exceeds its error share here means the amplitude is beyond practical use;
/// the fallback sizing then switches to its conservative analytic bound.
///
/// This covers the whole closed-form range: the sweep for an argument `R` seeds
/// at `⌈R⌉ + 8 + 16·R^{1/3}` and runs to
/// `max(seed, ⌈R⌉) + 20 + √(160·max(seed, ⌈R⌉))` orders, i.e. 18459 at
/// `R = MAX_BESSEL_ARG_CLOSED_FORM` — about half the cap.  It also bounds the
/// ladder the fallback sizing builds to estimate a link's bandwidth.
const MAX_BESSEL_ORDER: usize = 1 << 15;

/// Cap on the two-carrier resonance enumeration's inner iterations,
/// `K·(2·min(M₁,M₂)+1)`.  The enumeration scans the *smaller* carrier, so an
/// asymmetric drive stays cheap; this bound keeps a symmetric one from turning a
/// large requested range into quadratic work.  Exceeding it sends the link to
/// the per-link time grid, exactly like an over-large amplitude.
const MAX_BESSEL_ENUM_WORK: usize = 1 << 22;

/// Cap on the one-mode convolution window, `harmonic span + 2·Σ_α|l_α|M_α`.
/// The window is allocated as two `Complex<f64>` vectors, so an unguarded
/// high-harmonic drive would abort on allocation instead of returning an error;
/// at this size the pair costs ~64 MB, well above any useful drive (the window
/// is `O(4·n_max + Σ|l_α|·r_α)`).
const MAX_BESSEL_WINDOW: usize = 1 << 22;

/// Magnitudes below this are treated as decayed; the ladder extends until its
/// top order is below the floor, which is what makes every suffix tail a
/// complete truncation budget.
const BESSEL_DECAY_FLOOR: f64 = 1e-20;

/// Magnitude band the sweep keeps its running values inside.  Unscaled sweep
/// values grow from the seed toward the turning point like
/// `J_⌈r⌉(r)/J_start(r)`, which overflows `f64` for a small argument and a high
/// seed; only the ratios are physical, so an exact power-of-two rescale costs
/// no accuracy.  The upper end must satisfy
/// `SWEEP_BAND_HIGH · 2·start/SWEEP_MIN_ARG < f64::MAX` (`10^30 · 10^254 <<
/// 10^308`), so no value that passes the check can overflow on the next
/// multiplication — checking "did this value overflow?" alone is not enough,
/// because the growth factor itself is unbounded.  The lower end is what stops
/// the rescale from driving *representable* orders to zero: a one-sided upper
/// guard did exactly that for `r ~ 1e-150`, where every rescale multiplied the
/// whole suffix by `2^-1000` while the recurrence grew only `~10^152` per step.
const SWEEP_BAND_LOW: f64 = 1e-30;
const SWEEP_BAND_HIGH: f64 = 1e30;
/// Arguments below this are evaluated analytically instead of by recurrence:
/// with `k ≤ ~5·10^3` the growth factor `2k/r` then approaches `10^254`, so the
/// band rescale can no longer guarantee the next product stays finite.  At this
/// size `J_0 = 1`, `J_1 = r/2` and every higher order underflows to zero anyway.
const SWEEP_MIN_ARG: f64 = 1e-250;
/// Divide every value from `newest` on by one power of two chosen so the
/// **largest** entry of that suffix lands at `~1`.  The factor is exact, so
/// every ratio `u[m]/u[anchor]` — the only thing the normalization reads — is
/// preserved, and no entry the sweep still needs can overflow.  Normalizing on
/// the suffix maximum rather than on its newest entry also covers the decay
/// region, where an older entry can be the largest one.
///
/// Deep-tail entries may still underflow to zero — their `J_m(r)` is
/// negligible there — but the normalization's `anchor` is the turning-point
/// order, which carries the largest magnitude of the finished sweep, so it is
/// never one of them.
fn rescale_sweep_tail(u: &mut [f64], newest: usize) {
    let magnitude = u[newest..]
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    debug_assert!(
        magnitude.is_finite() && magnitude > 0.0,
        "rescale_sweep_tail expects a finite non-zero entry"
    );
    // The caller only rescales when the newest magnitude left the band, and
    // `SWEEP_BAND_HIGH` bounds it, so `1023 - exponent` stays a valid biased
    // exponent and the scale is a finite positive power of two.
    let exponent = (magnitude.log2().floor() as i32).clamp(-1000, 1000);
    let scale = f64::from_bits(((1023 - exponent) as u64) << 52);
    for value in u[newest..].iter_mut() {
        *value *= scale;
    }
}

/// Every `J_m(r)` for `m ∈ [0, n]` at one argument, plus the two-sided tail
/// sums `tail[m] = 2·Σ_{k>m} |J_k(r)|`.
///
/// Both consumers of the Bessel backend need a *contiguous* order range at one
/// fixed argument: the Peierls fold sums `B_α(m)` over `m ∈ [-M, M]`, and the
/// adaptive cutoff needs `M+1, M+2, …` until the tail decays.  One backward
/// (Miller) sweep therefore yields the entire object in `O(n)`; requesting
/// each order separately reruns the special-function library's internal
/// recurrence (`O(n)` each) and costs `O(n²)`, and re-summing each tail
/// restarts an overlapping suffix every time.
///
/// # Cost
///
/// `n` grows by doubling until the top order decays, so the sequence of sweeps
/// is geometric and construction is `O(n + √n)` recurrence steps with
/// `n ≤ MAX_BESSEL_ORDER`.  The band rescale adds `O(n)` work per trigger and
/// fires roughly once per `30 / log10(2k/r)` orders.  Because every caller
/// passes `n_min ≈ ⌈r⌉ + margin`, large `r` means a large `n` but a small growth
/// factor and small `r` decays within a handful of orders, so the measured
/// rescale work stays below ~7 multiplies per sweep step (~5 ns/step) across
/// the whole reachable range.  A caller that paired a tiny argument with a huge
/// requested order would make the rescale quadratic (`≤ MAX_BESSEL_ORDER²`
/// multiplies, a few ms); `bessel_ladder_with_cutoff` never does.
struct BesselLadder {
    /// `J_0..J_n` at this argument.
    j: Vec<f64>,
    /// `tail[m] = 2·Σ_{k>m} |j[k]|`, with `tail[max_order()] = 0`.
    tail: Vec<f64>,
    /// Whether the top order fell below [`BESSEL_DECAY_FLOOR`], i.e. whether
    /// the tail sums are complete truncation budgets.  False only when the
    /// sweep hit [`MAX_BESSEL_ORDER`] first, so `decayed == false` always
    /// implies `max_order() == MAX_BESSEL_ORDER`; the caller must then treat
    /// the bandwidth estimate as truncated rather than converged.
    decayed: bool,
}

impl BesselLadder {
    /// Sweep a range of at least `n_min` orders, doubling at most a constant
    /// number of times when the initial estimate leaves the top order above the
    /// decay floor (a range that still sits below the turning point cannot show
    /// decay).
    fn new(r: f64, n_min: usize) -> Self {
        debug_assert!(r.is_finite() && r > 0.0, "ladder argument must be positive");
        // The top order can only decay past the turning point: `|J_m(r)|` falls
        // below `BESSEL_DECAY_FLOOR` about `13·r^(1/3)` orders beyond it
        // (measured 17, 28, 66 and 199 for r = 1, 8, 128 and 4000).  Starting the
        // estimate there rather than doubling up from the requested order alone
        // roughly halves the number of sweeps; an estimate that is still too
        // short is caught by the doubling below.
        let decay_estimate = (r.ceil() as usize).saturating_add(8 + (16.0 * r.cbrt()) as usize);
        let mut n = n_min.max(decay_estimate).clamp(1, MAX_BESSEL_ORDER);
        loop {
            let j = bessel_backward_sweep(r, n);
            if j[n].abs() < BESSEL_DECAY_FLOOR {
                return Self::from_orders(j, true);
            }
            if n >= MAX_BESSEL_ORDER {
                return Self::from_orders(j, false);
            }
            n = (n * 2).min(MAX_BESSEL_ORDER);
        }
    }

    fn from_orders(j: Vec<f64>, decayed: bool) -> Self {
        let n = j.len() - 1;
        let mut tail = vec![0.0_f64; n + 1];
        let mut suffix = 0.0;
        for k in (0..=n).rev() {
            tail[k] = 2.0 * suffix;
            suffix += j[k].abs();
        }
        Self { j, tail, decayed }
    }

    /// Largest order stored in the ladder.
    #[inline]
    fn max_order(&self) -> usize {
        self.j.len() - 1
    }

    /// Two-sided tail `2·Σ_{k>m} |J_k(r)|`, complete when [`Self::decayed`].
    #[inline]
    fn tail_after(&self, m: usize) -> f64 {
        self.tail[m]
    }
}

/// Backward (Miller) recurrence returning `J_0..J_n` for one argument.
///
/// The recurrence is stable only in the backward direction, so it is seeded at
/// `N = max(n, ⌈r⌉) + 20 + √(160·max(n, ⌈r⌉))` — above the turning point and
/// high enough for the seed's contamination to decay.  A margin of only a few
/// orders above `r` is not enough: measured `|ΔJ| = 4e-9` at `r = 4000` against
/// `5e-14` with this rule.
///
/// The sweep is normalized on the order with the largest `|J|` (the turning
/// point), not on `J_0`: `J_0` has zeros, and one ulp of absolute error there
/// becomes a relative error for *every* order.  Measured at `r = 128`, where
/// `|J_0|` is 2% of its envelope, normalizing by `J_0` costs 1.3e-13 relative
/// while the turning point stays at 7.8e-16.
fn bessel_backward_sweep(r: f64, n: usize) -> Vec<f64> {
    if r < SWEEP_MIN_ARG {
        // The growth factor `2k/r` would overflow before any rescale can act;
        // at this size `J_0 = 1`, `J_1 = r/2` and every higher order underflows.
        let mut j = vec![0.0_f64; n + 1];
        j[0] = 1.0;
        if n >= 1 {
            j[1] = r / 2.0;
        }
        return j;
    }
    let base = n.max(r.ceil() as usize).min(MAX_BESSEL_ORDER);
    let start = base + 20 + (160.0 * base as f64).sqrt() as usize;
    let mut u = vec![0.0_f64; start + 2];
    u[start] = 1.0;
    for k in (1..=start).rev() {
        u[k - 1] = (2.0 * k as f64 / r) * u[k] - u[k + 1];
        let magnitude = u[k - 1].abs();
        // Every written value shares one scale (the sweep only ever reads what
        // it has already written), so rescaling the written suffix keeps all
        // ratios exact.
        if magnitude > 0.0 && !(SWEEP_BAND_LOW..=SWEEP_BAND_HIGH).contains(&magnitude) {
            rescale_sweep_tail(&mut u, k - 1);
        }
    }
    // Anchor on the largest stored magnitude, which for the callers here is the
    // turning point.  Picking an order by index instead would break whenever
    // that order sits on a zero of `J_m` (`J_0` has one at 2.4048…, `J_1` at
    // 7.0156…): the whole ladder would then be scaled by a value whose relative
    // error is enormous.
    let anchor = (0..=n)
        .max_by(|&a, &b| u[a].abs().total_cmp(&u[b].abs()))
        .expect("the sweep always stores at least one order");
    let scale = bessel_j(anchor as isize, r) / u[anchor];
    u.truncate(n + 1);
    for value in u.iter_mut() {
        *value *= scale;
    }
    u
}

/// Adaptive Bessel order cutoff for one link — the smallest
/// `M ≥ ⌈r⌉ + margin` whose two-sided tail `2·Σ_{m>M} |J_m(r)|` fits
/// `error_share` — together with the ladder that backs it.
///
/// `margin` doubles as a higher starting estimate for the time-grid fallback
/// sizing, which passes `0`.  At [`MAX_BESSEL_ORDER`] the cutoff saturates and
/// the returned ladder reports `decayed == false`, so the caller can tell a
/// converged bandwidth estimate from a truncated one.
fn bessel_ladder_with_cutoff(r: f64, error_share: f64, margin: isize) -> (BesselLadder, isize) {
    debug_assert!(
        r.is_finite() && r > 0.0,
        "bessel_ladder_with_cutoff: r must be finite and positive"
    );
    // The float-to-int cast saturates for huge r; saturating_add keeps the
    // +margin step overflow-free before the clamp.
    let m_start = (r.ceil() as isize)
        .saturating_add(margin)
        .clamp(0, MAX_BESSEL_ORDER as isize) as usize;
    let mut ladder = BesselLadder::new(r, m_start);
    loop {
        // The scan stops *before* `max_order`, whose suffix sum is zero by
        // construction: accepting it would report a truncated (non-decayed)
        // ladder as a converged cutoff.  A decayed ladder always fits some
        // `m <= max_order - 1`, because `tail[max_order - 1] = 2|J_top| <
        // 2·BESSEL_DECAY_FLOOR` is far below any per-mode error share.
        let found = (m_start..ladder.max_order()).find(|&m| ladder.tail_after(m) <= error_share);
        if let Some(m) = found {
            return (ladder, m as isize);
        }
        if ladder.max_order() >= MAX_BESSEL_ORDER {
            return (ladder, MAX_BESSEL_ORDER as isize);
        }
        ladder = BesselLadder::new(r, ladder.max_order() + 1);
    }
}

/// Ladder and adaptive cutoff for one link amplitude, with the validation both
/// coefficient paths need.  `max_arg` is the cap of the calling path:
/// [`MAX_BESSEL_ARG_CLOSED_FORM`] for the closed form, whose cost is linear in
/// `R`, and [`MAX_BESSEL_ARG`] for a convolution operand.
fn bessel_ladder_for_amplitude(
    r: f64,
    error_share: f64,
    cutoff_margin: isize,
    max_arg: f64,
) -> Result<(BesselLadder, isize)> {
    if !r.is_finite() || r > max_arg {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: mode amplitude R = {r:.3} is outside the \
             Bessel backend's range (R ≤ {max_arg}); use the time-grid backend"
        )));
    }
    // Adaptive cutoff plus the ladder that backs it: the smallest M whose
    // two-sided tail 2 * Σ_{m>M} |J_m(r)| fits the per-mode error share,
    // delivered together with J_0..J_M and every candidate tail from one
    // backward sweep.
    let (ladder, m_cap) = bessel_ladder_with_cutoff(r, error_share, cutoff_margin);
    if !ladder.decayed {
        // A truncated ladder's suffix sums stop at MAX_BESSEL_ORDER, so they
        // cannot certify a truncation budget.  The closed-form cap plus the largest
        // allowed margin keeps this out of reach, but a future cap change must
        // fail loudly instead of folding a truncated expansion.
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: the order range for R = {r:.3} saturated at \
             {MAX_BESSEL_ORDER} orders without decaying; lower MAX_BESSEL_ARG or \
             raise MAX_BESSEL_ORDER"
        )));
    }
    Ok((ladder, m_cap))
}

/// Closed form for a drive whose nonzero temporal harmonics collapse to one
/// value `l`.
///
/// The resonance condition `l·m = -n` then fixes the multi-index sum to a single
/// term,
///
/// ```math
/// C_n = (-i)^m J_m(R)\, e^{-im\delta},\qquad l \mid n,
/// ```
///
/// and every other bin vanishes: one lookup per requested order instead of the
/// `O(M·W)` one-mode convolution.  Orders past the adaptive cutoff `M` stay
/// zero, which is exactly what the convolution returns once its Bessel orders
/// are truncated there.  DC modes only multiply the result by
/// `exp(-i Re(a_0·d))`, the static Peierls phase.
///
/// Returns `Ok(None)` when the drive has two or more distinct nonzero harmonics
/// — coherent sums of equal harmonics count once — because that sum genuinely
/// needs the convolution.
fn single_harmonic_closed_form(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    cutoff_margin: isize,
) -> Result<Option<Array1<Complex<f64>>>> {
    let mut single: Option<(isize, Complex<f64>)> = None;
    let mut dc = 0.0_f64;
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            // Never form the unused imaginary projection: it may overflow even
            // when the static real field is identically zero.
            dc += mode
                .a_complex
                .iter()
                .zip(d)
                .map(|(a, x)| a.re * x)
                .sum::<f64>();
            continue;
        }
        // Mode projection onto the link: z = a·d = R e^{iδ}.
        let z: Complex<f64> = mode
            .a_complex
            .iter()
            .zip(d.iter())
            .map(|(a, x)| *a * *x)
            .sum();
        if z.re == 0.0 && z.im == 0.0 {
            continue;
        }
        match &mut single {
            Some((harmonic, total)) if *harmonic == mode.harmonic => *total += z,
            Some(_) => return Ok(None),
            None => single = Some((mode.harmonic, z)),
        }
    }
    if !dc.is_finite() {
        return Err(TbError::Other(
            "bessel_peierls_coeffs: the static phase is not finite".into(),
        ));
    }
    let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
    let dc_phase = Complex::new(0.0, -dc).exp();
    // Equal harmonics are summed coherently, so the total can cancel exactly;
    // such a drive dresses nothing, exactly like a single zero projection, and
    // must not reach the ladder (whose argument is asserted positive).
    let single = single.filter(|(_, z)| z.re != 0.0 || z.im != 0.0);
    let Some((harmonic, z)) = single else {
        // Purely static drive: the Peierls phase shifts the on-site term only.
        let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = dc_phase;
        }
        return Ok(Some(coeffs));
    };

    let r = z.norm();
    // Same per-mode share as the convolution path, so the adaptive cutoff — and
    // therefore every truncated bin — agrees exactly.
    let error_share = 1e-12 / (drive.modes.len() as f64);
    let (ladder, m_cap) =
        bessel_ladder_for_amplitude(r, error_share, cutoff_margin, MAX_BESSEL_ARG_CLOSED_FORM)?;
    let delta = z.arg();
    let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
    for (index, n) in (harmonic_min..=harmonic_max).enumerate() {
        let n = n as i128;
        let harmonic = harmonic as i128;
        if n % harmonic != 0 {
            continue;
        }
        // Widen before negation/division: MIN / -1 must not panic, and only
        // an index inside the bounded ladder is converted back to isize.
        let m = -n / harmonic;
        let order = m.unsigned_abs();
        if order > m_cap as u128 {
            continue;
        }
        let order = order as usize;
        let m = m as isize;
        let j_m = if m < 0 && order % 2 == 1 {
            -ladder.j[order]
        } else {
            ladder.j[order]
        };
        coeffs[index] = dc_phase
            * Complex::new(0.0, -1.0).powi(m as i32)
            * j_m
            * Complex::from_polar(1.0, -(m as f64) * delta);
    }
    Ok(Some(coeffs))
}

/// Peierls Fourier coefficients `C_n(d)` via the generalized Bessel
/// expansion, for `n ∈ [harmonic_min, harmonic_max]`.
///
/// For a drive `a(t) = Re Σ_α a_α e^{−i l_α Ω₀ t}` each mode contributes a
/// scalar pair `z_α = a_α·d = R_α e^{iδ_α}` per link displacement `d`, and
/// the Jacobi–Anger expansion of the factorized Peierls exponential gives
///
/// ```math
/// C_n(d) = \sum_{\{m_α\} : Σ_α l_α m_α = -n}
///          \prod_α (-i)^{m_α} J_{m_α}(R_α)\, e^{-i m_α δ_α}.
/// ```
///
/// (Resonance `n + Σ l m = 0`; the equivalent form `Σ l m = +n` with phase
/// `e^{+imδ}` must not be mixed in.)  A drive whose nonzero harmonics collapse
/// to one value needs no sum at all: [`single_harmonic_closed_form`] returns one
/// term per bin.  Everything else evaluates the multi-index sum as a sequence
/// of one-mode discrete convolutions
///
/// ```math
/// S^{(0)}_n = δ_{n,0},\qquad
/// S^{(α)}_n = \sum_{m=-M_α}^{M_α} S^{(α-1)}_{n + l_α m}\, B_α(m),
/// \qquad
/// B_α(m) = (-i)^m J_m(R_α)\, e^{-imδ_α},
/// ```
///
/// which costs `O(N_mode · N_n · M_avg)` — independent of the time-grid
/// size, `N_n` being the requested harmonic range.
///
/// Two carriers are the common multi-colour case and are evaluated differently:
/// the constraint `l₁m₁ + l₂m₂ = −n` leaves one free index, so the same sum is
/// enumerated directly at `O(N_n · (2·min(M₁,M₂) + 1))` — linear in the
/// amplitude, with no working window and therefore no BLAS call per requested
/// bin.  That path admits amplitudes up to [`MAX_BESSEL_ARG_CLOSED_FORM`] and is
/// bounded by [`MAX_BESSEL_ENUM_WORK`]; three or more carriers fall back to the
/// convolution above, which keeps the lower [`MAX_BESSEL_ARG`] cap and the
/// [`MAX_BESSEL_WINDOW`] bound.
///
/// Each mode's cutoff `M_α` is chosen adaptively so the truncated
/// tail `Σ_{|m|>M_α} |J_m(R_α)|` stays below a per-mode error share
/// (`1e-12 / N_mode`), with `cutoff_margin` as an additional minimum, and
/// it must stay at or below [`MAX_BESSEL_ORDER`].
///
/// Each mode needs `J_0..J_{M_α}` **and** every tail `Σ_{m>M}|J_m(R_α)|`;
/// both come from one [`BesselLadder`] per mode, i.e. a single backward
/// recurrence sweep plus one suffix sum.  In particular the adaptive search
/// never evaluates an order on its own, so its cost is linear in the order
/// range instead of quadratic.
///
/// Verified against the independent time-grid DFT
/// (`crate::floquet_test::peierls_fourier_coeffs`) for linear, circular,
/// elliptical, and multi-harmonic drives to ~1e-15.
///
/// # Arguments
/// * `d` - real link displacement (Cartesian, length `DIM`).
/// * `drive` - the light drive (modes `(l_α, a_α)`, base frequency `Ω₀`).
/// * `harmonic_min`, `harmonic_max` - inclusive harmonic range to return.
/// * `cutoff_margin` - minimum number of Bessel orders beyond `⌈R_α⌉`,
///   in `0..=48` (the adaptive tail check may push `M_α` higher; only
///   lower bounded by this).
///
/// # Returns
/// `C_n(d)` for `n = harmonic_min..=harmonic_max` as an [`Array1<Complex<f64>>`] of
/// length `harmonic_max - harmonic_min + 1`.
///
/// # Errors
/// Returns [`TbError::Other`] when `harmonic_min > harmonic_max`, when `cutoff_margin`
/// is outside `0..=48`, when a carrier amplitude exceeds the branch's cap
/// ([MAX_BESSEL_ARG_CLOSED_FORM] for one carrier (the closed form) and for two
/// (the enumeration), [MAX_BESSEL_ARG] for the convolution), when the
/// enumeration's work budget or the convolution's window bound is exceeded (the
/// caller must fall back to the time-grid backend), or when the harmonic range
/// or working window would overflow `isize`.
pub(crate) fn bessel_peierls_coeffs(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    cutoff_margin: isize,
) -> Result<Array1<Complex<f64>>> {
    if harmonic_min > harmonic_max {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: empty harmonic range [{harmonic_min}, {harmonic_max}]"
        )));
    }
    if !(0..=48).contains(&cutoff_margin) {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: cutoff_margin = {cutoff_margin} outside [0, 48]"
        )));
    }
    // harmonic_max >= harmonic_min here, so the span is non-negative; checked_sub guards
    // the isize::MIN..=isize::MAX range against overflow.
    let harmonic_count = harmonic_max.checked_sub(harmonic_min).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: harmonic range too wide".to_string())
    })? as usize
        + 1;
    // Empty drive: the Peierls exponential is 1, so only C_0 survives.
    if drive.modes.is_empty() {
        let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = Complex::new(1.0, 0.0);
        }
        return Ok(coeffs);
    }

    // A drive whose nonzero harmonics collapse to one value needs no
    // convolution: the resonance condition leaves exactly one term per bin.
    if let Some(coeffs) =
        single_harmonic_closed_form(d, drive, harmonic_min, harmonic_max, cutoff_margin)?
    {
        return Ok(coeffs);
    }

    // Certify exact physical cancellation after link projection, not by summing
    // Cartesian amplitudes. Separate truncated Bessel products for opposite
    // carriers otherwise leave spurious nonzero edge harmonics for a zero field.
    let (dc, carriers, exact_sum) = time_grid_projections(drive, d);
    if carriers.is_empty() && exact_sum {
        if !dc.is_finite() {
            return Err(TbError::Other(
                "bessel_peierls_coeffs: the static phase is not finite".into(),
            ));
        }
        let mut coeffs = Array1::<Complex<f64>>::zeros(harmonic_count);
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = Complex::new(0.0, -dc).exp();
        }
        return Ok(coeffs);
    }

    // Two-pass construction.  First pass: per-mode projections and adaptive
    // cutoffs.  The convolution only supports R_α ≤ MAX_BESSEL_ARG; the caller falls back
    // to the time-grid backend beyond that.
    struct ModeData {
        /// `J_0..J_{M_α}` and the tail sums behind the adaptive cutoff.
        ladder: BesselLadder,
        /// Link projection `z_α = a_α·d = R_α e^{iδ_α}`.
        projection: Complex<f64>,
        harmonic: isize,
        m_cap: isize,
    }
    let mut modes = Vec::<ModeData>::with_capacity(drive.modes.len());
    let error_share = 1e-12 / (drive.modes.len() as f64);
    let mut dc_re = 0.0_f64;
    // Two carriers are enumerated, and that path is linear in the amplitude
    // instead of quadratic, so it can reach as far as the closed form.  The
    // count is taken over the drive's harmonics up front; a mode whose link
    // projection vanishes is dropped later and only makes this more permissive,
    // which the budget check below then catches.
    let two_carriers = drive.modes.iter().filter(|mode| mode.harmonic != 0).count() == 2;
    let mode_cap = if two_carriers {
        MAX_BESSEL_ARG_CLOSED_FORM
    } else {
        MAX_BESSEL_ARG
    };
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            dc_re += mode
                .a_complex
                .iter()
                .zip(d)
                .map(|(a, x)| a.re * x)
                .sum::<f64>();
            continue;
        }
        // Mode projection onto the link: z = a·d = R e^{iδ}.
        let z: Complex<f64> = mode
            .a_complex
            .iter()
            .zip(d.iter())
            .map(|(a, d)| *a * *d)
            .sum();
        let r = z.norm();
        if r == 0.0 {
            // Degenerate mode: only m = 0 contributes (B = 1), a no-op fold.
            continue;
        }
        let (ladder, m_cap) = bessel_ladder_for_amplitude(r, error_share, cutoff_margin, mode_cap)?;
        modes.push(ModeData {
            ladder,
            projection: z,
            harmonic: mode.harmonic,
            m_cap,
        });
    }
    if !dc_re.is_finite() {
        // Two finite static projections can still sum past f64::MAX; the phase
        // would then be NaN and every coefficient with it.
        return Err(TbError::Other(
            "bessel_peierls_coeffs: the static phase is not finite".to_string(),
        ));
    }
    // The working window only concerns the convolution, and its drift is
    // computed after the enumeration has had its chance: an enumeration never
    // allocates a window, so an overflowing |l|·M there is not an error.
    let drift_for_fold = |modes: &[ModeData]| -> Result<isize> {
        let mut total = 0_isize;
        for mode in modes {
            let harmonic_abs = mode.harmonic.checked_abs().ok_or_else(|| {
                TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
            })?;
            total = total
                .checked_add(harmonic_abs.checked_mul(mode.m_cap).ok_or_else(|| {
                    TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
                })?)
                .ok_or_else(|| {
                    TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
                })?;
        }
        Ok(total)
    };

    // Two nonzero harmonics are the common multi-colour case (a fundamental plus
    // a harmonic, or two commensurate colours).  The resonance condition
    // l₁m₁ + l₂m₂ = −n leaves one free index, so those coefficients can be
    // enumerated directly in O(K·(2M+1)) instead of folding a window of width
    // span + 2·Σ|l|M — one BLAS call per requested bin dominated that fold's
    // measured cost.  A static mode contributes the pure phase e^{-i Re z}.
    let carriers: Vec<&ModeData> = modes.iter().filter(|mode| mode.harmonic != 0).collect();
    // Both counts must agree: `mode_cap` was chosen from the drive-level count,
    // so a link that drops one of those carriers (a zero projection) must not
    // silently take this path with the higher cap.
    if two_carriers && carriers.len() == 2 {
        let requested = (harmonic_max - harmonic_min + 1) as usize;
        let scan_len = 2 * carriers[0].m_cap.min(carriers[1].m_cap) as usize + 1;
        let work = requested.saturating_mul(scan_len);
        if work > MAX_BESSEL_ENUM_WORK {
            return Err(TbError::Other(format!(
                "bessel_peierls_coeffs: the two-carrier enumeration would need \
                 {work} iterations for {requested} requested orders, above the \
                 {MAX_BESSEL_ENUM_WORK} cap; lower the amplitude or the harmonic range"
            )));
        }
        let dc_phase = Complex::new(0.0, -dc_re).exp();
        // Iterate over the mode with the smaller cutoff and divmod by the other's
        // harmonic: only m pairs whose partner index is an integer inside its own
        // cutoff contribute.
        let (scan, other) = if carriers[0].m_cap <= carriers[1].m_cap {
            (carriers[0], carriers[1])
        } else {
            (carriers[1], carriers[0])
        };
        let sequence = |mode: &ModeData| -> Vec<Complex<f64>> {
            let delta = mode.projection.arg();
            let mut out = Vec::with_capacity((2 * mode.m_cap + 1) as usize);
            let mut minus_i_power = Complex::new(1.0, 0.0);
            for m in 0..=mode.m_cap {
                let j_m = mode.ladder.j[m as usize];
                let value = minus_i_power * j_m * Complex::from_polar(1.0, -(m as f64) * delta);
                if m == 0 {
                    out.push(value);
                } else {
                    let neg = minus_i_power * j_m * Complex::from_polar(1.0, (m as f64) * delta);
                    out.push(neg);
                    out.push(value);
                }
                minus_i_power *= Complex::new(0.0, -1.0);
            }
            out
        };
        let scan_seq = sequence(scan);
        let other_seq = sequence(other);
        // Both vectors follow the fold's layout: index 0 is m = 0, then the pairs
        // (m = -1, m = +1), (m = -2, m = +2), ... so m > 0 sits at 2m and m < 0
        // at 2|m| - 1.
        let slot = |m: isize| -> usize {
            if m >= 0 {
                (2 * m) as usize
            } else {
                (2 * m.unsigned_abs() - 1) as usize
            }
        };
        let mut coeffs = Array1::<Complex<f64>>::zeros((harmonic_max - harmonic_min + 1) as usize);
        for (index, n) in (harmonic_min..=harmonic_max).enumerate() {
            let target = -(n as i128);
            let mut acc = Complex::new(0.0, 0.0);
            for m in -scan.m_cap..=scan.m_cap {
                // Huge products can cancel to a small requested order. isize
                // harmonics times the bounded Bessel cutoff fit in i128, including
                // the subtraction, division and absolute value at either limit.
                let numerator = target - (scan.harmonic as i128) * (m as i128);
                let other_harmonic = other.harmonic as i128;
                if numerator % other_harmonic != 0 {
                    continue;
                }
                let partner = numerator / other_harmonic;
                if partner.abs() > other.m_cap as i128 {
                    continue;
                }
                acc += scan_seq[slot(m)] * other_seq[slot(partner as isize)];
            }
            coeffs[index] = acc * dc_phase;
        }
        return Ok(coeffs);
    }

    // Second pass: the working window must cover the actual reachable
    // support [−drift, +drift] around [harmonic_min, harmonic_max], because intermediates
    // outside the requested range can fold back into it.
    let total_drift = drift_for_fold(&modes)?;
    let work_min = harmonic_min.checked_sub(total_drift).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: working window underflow".to_string())
    })?;
    let work_max = harmonic_max.checked_add(total_drift).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: working window overflow".to_string())
    })?;
    // work_max >= work_min by construction (harmonic_max >= harmonic_min, drift >= 0), so
    // the span is non-negative; checked_sub and usize::try_from are kept
    // for hygiene.
    let work_span = work_max.checked_sub(work_min).ok_or_else(|| {
        TbError::Other("bessel_peierls_coeffs: working window span overflow".to_string())
    })?;
    let work_len = usize::try_from(work_span).map_err(|_| {
        TbError::Other("bessel_peierls_coeffs: working window too large".to_string())
    })? + 1;
    if work_len > MAX_BESSEL_WINDOW {
        return Err(TbError::Other(format!(
            "bessel_peierls_coeffs: the one-mode convolution window needs {work_len} entries, \
             above the {MAX_BESSEL_WINDOW} cap; lower the drive harmonics or the amplitudes"
        )));
    }

    let requested = (harmonic_min - work_min) as usize..(harmonic_max - work_min + 1) as usize;
    let mut sequence = vec![Complex::new(0.0, 0.0); work_len];
    let mut live = 0..0;
    if let Some(zero_index) = work_min.checked_neg()
        && (0_isize..work_len as isize).contains(&zero_index)
    {
        sequence[zero_index as usize] = Complex::new(1.0, 0.0);
        live = zero_index as usize..zero_index as usize + 1;
    }
    let mut next = vec![Complex::new(0.0, 0.0); work_len];
    let mut remaining_drift = total_drift as usize;

    for mode in &modes {
        if live.is_empty() {
            break;
        }
        // Keep only indices reachable from this prefix that can still reach a
        // requested bin through the remaining modes. This is exact structural
        // support pruning, not a small-value cutoff. The original full-window
        // overflow/resource checks above still apply before any pruning.
        let mode_drift = mode.harmonic.unsigned_abs() * mode.m_cap as usize;
        remaining_drift -= mode_drift;
        let target = live
            .start
            .saturating_sub(mode_drift)
            .max(requested.start.saturating_sub(remaining_drift))
            ..(live.end + mode_drift)
                .min(work_len)
                .min(requested.end + remaining_drift);
        next.fill(Complex::new(0.0, 0.0));
        if target.is_empty() {
            std::mem::swap(&mut sequence, &mut next);
            break;
        }

        // One-mode sequence B(m) = (-i)^m J_m(r) e^{-imδ}, m ∈ [-M, M].
        // Accumulate (-i)^m iteratively.
        let mut minus_i_power = Complex::new(1.0, 0.0); // (-i)^0
        // The cutoff is at most the ladder length (the scan stops before the
        // ladder's top entry), so every lookup below is in range; the ladder
        // itself is capped at MAX_BESSEL_ORDER.
        debug_assert!(mode.m_cap >= 0 && mode.m_cap as usize <= mode.ladder.max_order());
        let delta = mode.projection.arg();
        let mut b = Vec::<(isize, Complex<f64>)>::with_capacity((2 * mode.m_cap + 1) as usize);
        for m in 0..=mode.m_cap {
            let j_m = mode.ladder.j[m as usize];
            let value = minus_i_power * j_m * Complex::from_polar(1.0, -(m as f64) * delta);
            if m == 0 {
                b.push((0, value));
            } else {
                // B(-m) = (-i)^{-m} J_{-m}(r) e^{+imδ}
                //       = i^m · (-1)^m J_m(r) e^{+imδ}
                //       = (-i)^m J_m(r) e^{+imδ} (since i^m (-1)^m = (-i)^m)
                let neg = minus_i_power * j_m * Complex::from_polar(1.0, (m as f64) * delta);
                b.push((-m, neg));
                b.push((m, value));
            }
            minus_i_power *= Complex::new(0.0, -1.0); // times (-i)
        }

        // Fold: S'_n = Σ_m S_{n + l·m} B(m). Intersect shifted live
        // intervals once per m, then accumulate contiguous slices without
        // per-element index arithmetic. Preserve the original m/sum order.
        for &(m, weight) in &b {
            // drift_for_fold checked |harmonic| * m_cap and their sum;
            // |m| <= m_cap, so this signed product is representable. The
            // accepted window also bounds every index addition below.
            let shift = mode.harmonic * m;
            let offset = shift.unsigned_abs();
            let (source, destination) = if shift >= 0 {
                let start = target.start.max(live.start.saturating_sub(offset));
                let end = target.end.min(live.end.saturating_sub(offset));
                if start >= end {
                    continue;
                }
                (
                    &sequence[start + offset..end + offset],
                    &mut next[start..end],
                )
            } else {
                let start = target.start.max(live.start + offset);
                let end = target.end.min(live.end + offset);
                if start >= end {
                    continue;
                }
                (
                    &sequence[start - offset..end - offset],
                    &mut next[start..end],
                )
            };
            for (output, &value) in destination.iter_mut().zip(source) {
                *output += value * weight;
            }
        }
        std::mem::swap(&mut sequence, &mut next);
        live = target;
    }

    // Static modes were folded into one phase above, exactly as the closed form
    // and the enumeration do.
    let dc_phase = Complex::new(0.0, -dc_re).exp();
    Ok(Array1::from(
        sequence[requested]
            .iter()
            .map(|coeff| coeff * dc_phase)
            .collect::<Vec<_>>(),
    ))
}

/// Maximum number of time points for a per-link fallback DFT.
const FALLBACK_GRID_MAX: usize = 1 << 20;

/// Grid-size decision for a fallback link (see [`fallback_time_grid_coeffs`]).
struct FallbackGridSize {
    /// Alias-free grid size to evaluate the DFT at.
    n_req: usize,
    /// Signal-bandwidth Nyquist term `2·Σ_α |l_α|·M_α + 4`.
    required: usize,
    /// Requested-range Nyquist term `2·max(|harmonic_min|, |harmonic_max|) + 1`.
    request_range: usize,
    /// The unclamped size exceeded [`FALLBACK_GRID_MAX`].
    clamped: bool,
    /// A mode's adaptive cutoff saturated at [`MAX_BESSEL_ORDER`] orders,
    /// requiring a conservative analytic tail bound instead.
    saturated: bool,
}

// Project before time evolution, sum coherent carriers (including l/-l),
// and keep DC outside the oscillating phase. Sizing and sampling must use
// the same scalar signal, not separately rounded Cartesian fields.
fn time_grid_projections(
    drive: &FloquetDrive,
    d: &Array1<f64>,
) -> (f64, std::collections::BTreeMap<usize, Complex<f64>>, bool) {
    let mut dc = 0.0;
    let mut carriers = std::collections::BTreeMap::new();
    let mut exact_sum = true;
    // TwoSum detects discarded rounding without changing the computed signal.
    // A rounded zero alone must not certify the new exact-static shortcut.
    let addition_is_exact = |a: f64, b: f64, sum: f64| {
        let recovered_b = sum - a;
        (a - (sum - recovered_b)) + (b - recovered_b) == 0.0
    };
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            dc += mode
                .a_complex
                .iter()
                .zip(d)
                .map(|(a, x)| a.re * x)
                .sum::<f64>();
            continue;
        }
        let z: Complex<f64> = mode.a_complex.iter().zip(d).map(|(a, x)| a * x).sum();
        let z = if mode.harmonic > 0 { z } else { z.conj() };
        let total = carriers
            .entry(mode.harmonic.unsigned_abs())
            .or_insert(Complex::new(0.0, 0.0));
        let next = *total + z;
        exact_sum &= addition_is_exact(total.re, z.re, next.re)
            && addition_is_exact(total.im, z.im, next.im);
        *total = next;
    }
    carriers.retain(|_, z| z.re != 0.0 || z.im != 0.0);
    (dc, carriers, exact_sum)
}

/// Size the alias-free grid from the coherent physical carriers after projection:
/// `max(2·Σ |l|·M + 4, 2·max(|harmonic_min|, |harmonic_max|) + 1)`.
/// Keep the unclamped requirement so public validation can reject aliasing.
fn fallback_grid_size(
    drive: &FloquetDrive,
    d: &Array1<f64>,
    harmonic_min: isize,
    harmonic_max: isize,
) -> FallbackGridSize {
    // Nyquist bandwidth of the Peierls exponential on this link.
    let mut bandwidth = 0_usize;
    let mut saturated = false;
    let mode_count = drive.modes.len().max(1);
    let error_share = 1e-12 / mode_count as f64;
    let (_, carriers, _) = time_grid_projections(drive, d);
    for (harmonic, z) in carriers {
        let r = z.norm();
        if !r.is_finite() {
            // An unrepresentable physical carrier has unbounded required work,
            // not zero bandwidth. Public validation must refuse this fallback.
            bandwidth = usize::MAX;
            break;
        }
        // The Bessel backend's precision margin is not a sampling floor.
        // A ladder that never decayed inside MAX_BESSEL_ORDER cannot certify a
        // bandwidth: its suffix sums stop at the cap, so a cutoff read from them
        // omits the orders above it — measured at r = 32483 a suffix budget of
        // 9.5e-13 against a complete two-sided tail of 1.9e-12 for a 1e-12
        // share.  Such an argument therefore always takes the conservative
        // analytic bound below, exactly like r > MAX_BESSEL_ORDER.
        let (m_cap, truncated) = if r > MAX_BESSEL_ORDER as f64 {
            (MAX_BESSEL_ORDER as isize, true)
        } else {
            let (ladder, m_cap) = bessel_ladder_with_cutoff(r, error_share, 0);
            (m_cap, !ladder.decayed)
        };
        let cutoff = if truncated {
            saturated = true;
            // For m >= 3r, |J_m(r)| <= (r/2)^m/m! <= (e/6)^m < 2^-m
            // (DLMF 10.14.4). Thus the two-sided tail is <= 2^(1-M).
            // Split its budget across modes. Float casts and all grid-size
            // arithmetic saturate; the explicit requirement is never clamped.
            let tail_floor = 44 + (usize::BITS - mode_count.leading_zeros()) as usize;
            ((3.0 * r).ceil() as usize).max(tail_floor)
        } else {
            m_cap as usize
        };
        let drift = harmonic.saturating_mul(cutoff);
        bandwidth = bandwidth.saturating_add(drift);
    }
    let required = bandwidth.saturating_mul(2).saturating_add(4);
    let harmonic_max_abs = harmonic_min.unsigned_abs().max(harmonic_max.unsigned_abs());
    let request_range = (2_usize).saturating_mul(harmonic_max_abs).saturating_add(1);
    let mut n_req = required.max(request_range);
    let mut clamped = false;
    if n_req > FALLBACK_GRID_MAX {
        n_req = FALLBACK_GRID_MAX;
        clamped = true;
    }
    if saturated {
        // Automatic fallback retains its maximum-grid policy; explicit Sambe
        // grids are checked against the unclamped analytic requirement above.
        n_req = FALLBACK_GRID_MAX;
    }
    FallbackGridSize {
        n_req,
        required,
        request_range,
        clamped,
        saturated,
    }
}

/// Time-grid fallback for links outside the Bessel backend's range.
///
/// The fallback sizes its own grid from the link's spectral bandwidth
/// instead of relying on the caller's `n_time`: a link's Peierls
/// exponential has spectral content up to `Σ_α |l_α|·M_α(R_α)` (with `M_α`
/// the adaptive tail cutoff), and a coarse grid aliases it silently — e.g.
/// for a single `l = 100` mode at `R = 50`, a 512-point grid puts
/// `C_−8 ≈ −J_46(50) ≈ −0.17` where the true value is `0`
/// (`C_0 = J_0(50) = 0.0558` survives there only by a divisibility
/// coincidence).  The DFT is evaluated directly at the size chosen by
/// [`fallback_grid_size`], clamped to [`FALLBACK_GRID_MAX`] = 2^20 points
/// (beyond that the drive or truncation is pathological; accuracy degrades
/// and a warn-once message is printed).  When the adaptive bandwidth
/// estimate saturates at its [`MAX_BESSEL_ORDER`]-order cap — the ladder
/// reaches the cap without decaying, which for a single mode starts around
/// `R ≈ 32483` at the `1e-12` budget — sizing switches to a conservative
/// analytic tail bound.
/// The automatic grid uses the maximum size and prints a warn-once message;
/// it can still be too small when the analytic requirement exceeds this cap.
fn fallback_time_grid_coeffs(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    clamped: &AtomicBool,
    saturated: &AtomicBool,
) -> Array1<Complex<f64>> {
    let size = fallback_grid_size(drive, d, harmonic_min, harmonic_max);
    if size.clamped && !clamped.swap(true, Ordering::Relaxed) {
        eprintln!(
            "Floquet fallback grid clamped to {FALLBACK_GRID_MAX} time points \
             (link requires {required}, requested harmonic range {request_range}); \
             coefficients on this link may be inaccurate",
            required = size.required,
            request_range = size.request_range,
        );
    }
    if size.saturated && !saturated.swap(true, Ordering::Relaxed) {
        // Warn once that alias-freedom is no longer guaranteed.
        eprintln!(
            "Floquet fallback bandwidth estimate saturated at {MAX_BESSEL_ORDER} Bessel \
             orders; using the maximum {FALLBACK_GRID_MAX}-point grid — \
             coefficients on this link may still be inaccurate for extreme \
             amplitudes"
        );
    }
    let n_req = size.n_req;
    // Direct DFT at the fine resolution (no shared harmonic_count × n_time
    // Fourier matrix to reuse).
    let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
    let inv_n = 1.0 / (n_req as f64);
    let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
    let (dc, carriers, _) = time_grid_projections(drive, d);
    let dc_phase = Complex::new(0.0, -dc).exp();
    for it in 0..n_req {
        let theta = TAU * (it as f64) * inv_n;
        let mut link_phase = 0.0;
        for (&harmonic, &z) in &carriers {
            let harmonic_phase = Complex::new(0.0, -(harmonic as f64) * theta).exp();
            link_phase += (z * harmonic_phase).re;
        }
        let peierls = dc_phase * Complex::new(0.0, -link_phase).exp();
        for (i_n, n) in (harmonic_min..=harmonic_max).enumerate() {
            coeffs[i_n] += Complex::new(0.0, (n as f64) * theta).exp() * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= inv_n;
    }
    Array1::from(coeffs)
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

fn accumulate_scaled_real_space_blocks(
    target: &mut RealSpaceBlockMap,
    source_blocks: &[Array2<Complex<f64>>],
    source_r: &Array2<isize>,
    scale: f64,
) -> Result<()> {
    debug_assert_eq!(source_blocks.len(), source_r.nrows());
    if source_blocks.is_empty() {
        return Ok(());
    }
    if source_blocks
        .iter()
        .all(|block| block.iter().all(|value| value.re == 0.0 && value.im == 0.0))
    {
        for row in source_r.outer_iter() {
            target
                .entry(row.to_vec())
                .or_insert_with(|| Array2::<Complex<f64>>::zeros(source_blocks[0].raw_dim()));
        }
        return Ok(());
    }
    if !scale.is_finite() {
        return Err(TbError::Other(
            "a van Vleck frequency scaling factor is non-finite; increase omega0_ev or lower the requested order"
                .to_string(),
        ));
    }
    let scale = Complex::new(scale, 0.0);
    for (i_r, row) in source_r.outer_iter().enumerate() {
        target
            .entry(row.to_vec())
            .and_modify(|block| block.scaled_add(scale, &source_blocks[i_r]))
            .or_insert_with(|| source_blocks[i_r].mapv(|value| scale * value));
    }
    Ok(())
}

/// Merge two real-space block accumulators, preferring to insert the smaller
/// map into the larger one to limit tree lookups and reallocations.
fn merge_real_space_block_maps(
    mut left: RealSpaceBlockMap,
    mut right: RealSpaceBlockMap,
) -> RealSpaceBlockMap {
    if left.len() < right.len() {
        std::mem::swap(&mut left, &mut right);
    }
    for (r, block) in right {
        left.entry(r)
            .and_modify(|target| *target += &block)
            .or_insert(block);
    }
    left
}

/// Real-space commutator blocks `comm_n(R) = (AB)(R) − (BA)(R)` for the
/// harmonic pair `A_R = T_n(R)` and `B_R = T_{−n}(R)`:
///
/// ```math
/// (AB)(R) = \sum_{R'} A_{R-R'}\, B_{R'}, \qquad
/// (BA)(R) = \sum_{R'} B_{R-R'}\, A_{R'}.
/// ```
///
/// Both convolutions are accumulated with [`ndarray::linalg::general_mat_mul`],
/// which handles the input storage layouts; the returned support is the Minkowski sum
/// `{R1 + R2 : R1, R2 ∈ hamR}` in lexicographic order.  The result
/// satisfies the Hermiticity pairing `comm(R) = comm(−R)†` exactly — a
/// final pass ([`enforce_real_space_hermiticity`]) averages each ±R pair
/// with its conjugate-transposed partner, removing the summation-order
/// noise of the two independent convolutions.
///
/// # Errors
/// Returns [`TbError::MissingHermitianConjugateHopping`] if the support
/// is not closed under `R -> −R`; that can only happen for hand-built
/// models whose `hamR` itself violates the closure (a `Model` invariant
/// for all constructed models).  Returns [`TbError::Other`] if adding two
/// real-space hopping vectors would overflow `isize`.
///
/// # Panics
/// Debug-asserts that `a_blocks` and `b_blocks` each contain
/// `ham_r.nrows()` square blocks of one common size.
fn real_space_commutator<A, B>(
    a_blocks: &A,
    b_blocks: &B,
    ham_r: &Array2<isize>,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    let (blocks, support_rows) =
        real_space_commutator_with_supports(a_blocks, ham_r, b_blocks, ham_r)?;
    let nsta = if a_blocks.nblocks() == 0 {
        0
    } else {
        a_blocks.block(0).nrows()
    };

    // Enforce comm(R) = comm(−R)† exactly (fp symmetrization).
    let mut stacked = Array3::<Complex<f64>>::zeros((blocks.len(), nsta, nsta));
    for (i, block) in blocks.iter().enumerate() {
        stacked.index_axis_mut(Axis(0), i).assign(block);
    }
    enforce_real_space_hermiticity(&mut stacked, &support_rows)?;
    let blocks: Vec<Array2<Complex<f64>>> = (0..blocks.len())
        .map(|i| stacked.index_axis(Axis(0), i).to_owned())
        .collect();

    Ok((blocks, support_rows))
}

/// Real-space commutator for operands with independent hopping supports.
///
/// Unlike [`real_space_commutator`], this low-level helper does not impose a
/// Hermiticity relation on the result: an intermediate nested commutator such
/// as `[H_0,H_m]` is not Hermitian by itself.  The caller must only
/// symmetrize the final effective Hamiltonian after the complete signed
/// harmonic sum has been accumulated.
fn real_space_commutator_with_supports<A, B>(
    a_blocks: &A,
    a_r: &Array2<isize>,
    b_blocks: &B,
    b_r: &Array2<isize>,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    real_space_two_product_sum_with_supports(a_blocks, a_r, b_blocks, b_r, -1.0, "commutator")
}

/// Real-space anticommutator `{A,B} = AB + BA` on one common support.
///
/// When `B = A^dagger`, as in the finite-`q` weak-field kernel, the result is
/// Hermitian.  The final pairing pass removes only floating-point accumulation
/// noise and supplies exact `R <-> -R` conjugation to downstream derivatives.
fn real_space_anticommutator<A, B>(
    a_blocks: &A,
    b_blocks: &B,
    ham_r: &Array2<isize>,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    let (blocks, support_rows) = real_space_two_product_sum_with_supports(
        a_blocks,
        ham_r,
        b_blocks,
        ham_r,
        1.0,
        "anticommutator",
    )?;
    let nsta = if a_blocks.nblocks() == 0 {
        0
    } else {
        a_blocks.block(0).nrows()
    };
    let mut stacked = Array3::<Complex<f64>>::zeros((blocks.len(), nsta, nsta));
    for (i, block) in blocks.iter().enumerate() {
        stacked.index_axis_mut(Axis(0), i).assign(block);
    }
    enforce_real_space_hermiticity(&mut stacked, &support_rows)?;
    let blocks = (0..blocks.len())
        .map(|i| stacked.index_axis(Axis(0), i).to_owned())
        .collect();
    Ok((blocks, support_rows))
}

#[cfg(test)]
thread_local! {
    // Per-test-thread count: assert bounded work without flaky timing thresholds.
    static REAL_SPACE_SUPPORT_CONSTRUCTIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Accumulate `AB + reverse_scale * BA` for operands with independent
/// real-space supports.
fn real_space_two_product_sum_with_supports<A, B>(
    a_blocks: &A,
    a_r: &Array2<isize>,
    b_blocks: &B,
    b_r: &Array2<isize>,
    reverse_scale: f64,
    operation: &str,
) -> Result<RealSpaceBlocks>
where
    A: RealSpaceBlockSource + ?Sized,
    B: RealSpaceBlockSource + ?Sized,
{
    #[cfg(test)]
    REAL_SPACE_SUPPORT_CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
    debug_assert_eq!(a_blocks.nblocks(), a_r.nrows(), "a_blocks must match a_r");
    debug_assert_eq!(b_blocks.nblocks(), b_r.nrows(), "b_blocks must match b_r");
    debug_assert_eq!(a_r.ncols(), b_r.ncols(), "support dimensions must match");
    let nsta = if a_blocks.nblocks() > 0 {
        a_blocks.block(0).nrows()
    } else if b_blocks.nblocks() > 0 {
        b_blocks.block(0).nrows()
    } else {
        0
    };
    for index in 0..a_blocks.nblocks() {
        let block = a_blocks.block(index);
        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (nsta, nsta),
            "all blocks must be square nsta x nsta"
        );
    }
    for index in 0..b_blocks.nblocks() {
        let block = b_blocks.block(index);
        debug_assert_eq!(
            (block.nrows(), block.ncols()),
            (nsta, nsta),
            "all blocks must be square nsta x nsta"
        );
    }

    let a_is_zero = (0..a_blocks.nblocks()).all(|index| {
        a_blocks
            .block(index)
            .iter()
            .all(|value| value.re == 0.0 && value.im == 0.0)
    });
    let b_is_zero = (0..b_blocks.nblocks()).all(|index| {
        b_blocks
            .block(index)
            .iter()
            .all(|value| value.re == 0.0 && value.im == 0.0)
    });
    let skip_products = a_is_zero || b_is_zero;
    let one = Complex::new(1.0, 0.0);
    let reverse_scale = Complex::new(reverse_scale, 0.0);
    let n_a = a_r.nrows();
    let n_b = b_r.nrows();
    let pair_count = n_a
        .checked_mul(n_b)
        .ok_or_else(|| TbError::Other(format!("real-space {operation} pair count overflow")))?;

    let accumulate_pair = |accumulated: &mut RealSpaceBlockMap, pair_index: usize| -> Result<()> {
        let i_a = pair_index / n_b;
        let i_b = pair_index % n_b;
        let r_a = a_r.row(i_a);
        let r_b = b_r.row(i_b);
        let mut total = Vec::with_capacity(a_r.ncols());
        for (axis, (&left, &right)) in r_a.iter().zip(r_b.iter()).enumerate() {
            total.push(left.checked_add(right).ok_or_else(|| {
                TbError::Other(format!(
                    "real-space Minkowski sum overflow on axis {axis}: {left} + {right}"
                ))
            })?);
        }
        let comm = accumulated
            .entry(total)
            .or_insert_with(|| Array2::<Complex<f64>>::zeros((nsta, nsta)));
        if !skip_products {
            let a = a_blocks.block(i_a);
            let b = b_blocks.block(i_b);
            ndarray::linalg::general_mat_mul(one, &a, &b, one, comm);
            ndarray::linalg::general_mat_mul(reverse_scale, &b, &a, one, comm);
        }
        Ok(())
    };

    // Each Rayon worker accumulates into a private ordered map, so the hot
    // GEMM loop needs no locks.  The final reduction merges those maps before
    // the deterministic lexicographic output pass below.  Small and exact-zero
    // products stay serial to avoid scheduling and per-worker allocation cost.
    const PARALLEL_PAIR_THRESHOLD: usize = 128;
    let accumulated = if !skip_products
        && pair_count >= PARALLEL_PAIR_THRESHOLD
        && rayon::current_num_threads() > 1
    {
        // Rounding up can prevent Rayon's last binary split (e.g. 575²
        // pairs on eight threads), leaving half the workers without a job.
        let min_pairs_per_job = (pair_count / rayon::current_num_threads()).max(1);
        (0..pair_count)
            .into_par_iter()
            .with_min_len(min_pairs_per_job)
            .try_fold(
                RealSpaceBlockMap::new,
                |mut partial, pair_index| -> Result<RealSpaceBlockMap> {
                    accumulate_pair(&mut partial, pair_index)?;
                    Ok(partial)
                },
            )
            .try_reduce(RealSpaceBlockMap::new, |left, right| {
                Ok(merge_real_space_block_maps(left, right))
            })?
    } else {
        let mut accumulated = RealSpaceBlockMap::new();
        for pair_index in 0..pair_count {
            accumulate_pair(&mut accumulated, pair_index)?;
        }
        accumulated
    };

    let mut support_rows = Array2::<isize>::zeros((accumulated.len(), a_r.ncols()));
    let mut blocks = Vec::with_capacity(accumulated.len());
    for (i, (r, block)) in accumulated.into_iter().enumerate() {
        for (axis, value) in r.iter().enumerate() {
            support_rows[[i, axis]] = *value;
        }
        blocks.push(block);
    }
    Ok((blocks, support_rows))
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

fn enforce_real_space_hermiticity(
    ham: &mut Array3<Complex<f64>>,
    ham_r: &Array2<isize>,
) -> Result<()> {
    let n_r = ham_r.nrows();
    let mut visited = vec![false; n_r];

    for i_r in 0..n_r {
        if visited[i_r] {
            continue;
        }
        let neg_r = ham_r.row(i_r).mapv(|x| -x);
        let Some(j_r) = find_R(ham_r, &neg_r) else {
            return Err(TbError::MissingHermitianConjugateHopping {
                r: ham_r.row(i_r).to_owned(),
            });
        };

        if i_r == j_r {
            let block = ham.index_axis(Axis(0), i_r).to_owned();
            let block_dag = hermitian_conjugate(&block);
            let herm = Array2::from_shape_fn(block.raw_dim(), |index| {
                Complex::new(
                    block[index].re.midpoint(block_dag[index].re),
                    block[index].im.midpoint(block_dag[index].im),
                )
            });
            ham.index_axis_mut(Axis(0), i_r).assign(&herm);
            visited[i_r] = true;
        } else {
            let block_i = ham.index_axis(Axis(0), i_r).to_owned();
            let block_j = ham.index_axis(Axis(0), j_r).to_owned();
            let block_j_dag = hermitian_conjugate(&block_j);
            let avg = Array2::from_shape_fn(block_i.raw_dim(), |index| {
                Complex::new(
                    block_i[index].re.midpoint(block_j_dag[index].re),
                    block_i[index].im.midpoint(block_j_dag[index].im),
                )
            });
            let avg_dag = hermitian_conjugate(&avg);
            ham.index_axis_mut(Axis(0), i_r).assign(&avg);
            ham.index_axis_mut(Axis(0), j_r).assign(&avg_dag);
            visited[i_r] = true;
            visited[j_r] = true;
        }
    }
    Ok(())
}

fn hermitian_conjugate(a: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    a.t().mapv(|x| x.conj())
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
