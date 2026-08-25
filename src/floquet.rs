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
//! $$ t_{ij}(\mathbf R) = \langle i,\mathbf 0|\hat H|j,\mathbf R\rangle , $$
//!
//! where `hamR[a]` is the integer lattice vector `R` and `ham[a,i,j]` is the
//! corresponding matrix element.  The real-space link vector used by the Peierls
//! phase is
//!
//! $$ \mathbf d_{ij\mathbf R} =
//! \bigl(\mathbf R+\boldsymbol\tau_j-\boldsymbol\tau_i\bigr)L . $$
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
//! $$
//! \mathbf a(t) = \frac{e}{\hbar}\mathbf A(t),
//! $$
//!
//! so `LightMode::a_complex` has units of inverse length, matching the length
//! unit of `lat`.  For one mode with harmonic `l`, the stored complex amplitude
//! means
//!
//! $$ \mathbf a_l(t) =
//! \operatorname{Re}\left[
//! \mathbf a_l e^{-i l\Omega_0 t}
//! \right]. $$
//!
//! Multiple [`LightMode`] values are added before exponentiating:
//!
//! $$ \mathbf a(t) =
//! \operatorname{Re}\sum_\alpha
//! \mathbf a_\alpha e^{-i l_\alpha\Omega_0 t}. $$
//!
//! This representation covers linear, circular, elliptical, and mixed-harmonic
//! polarization without hard-coded special cases.
//!
//! # Peierls phase and Fourier blocks
//!
//! Every hopping is dressed as
//!
//! $$ t_{ij}(\mathbf R,t) =
//! t_{ij}(\mathbf R)
//! \exp\left[-i\,\mathbf a(t)\cdot\mathbf d_{ij\mathbf R}\right]. $$
//!
//! The Fourier coefficient of the Peierls phase is
//!
//! $$ C_n(\mathbf d) =
//! \frac{1}{T}\int_0^T dt\,
//! e^{in\Omega_0 t}
//! \exp\left[-i\,\mathbf a(t)\cdot\mathbf d\right]. $$
//!
//! Two backends evaluate `C_n`.  The Sambe and time-grid paths
//! ([`Floquet::floquet_model`], `PeierlsFourierMethod::TimeGrid`) integrate
//! the uniform time grid over one period — deliberately more general than a
//! Bessel-function formula, handling arbitrary complex polarization and
//! arbitrary commensurate harmonic mixing.  The van Vleck effective-model
//! path (`Model::floquet_effective_model`, Bessel backend) uses the
//! generalized Bessel expansion per link, falling back to a per-link
//! time-grid DFT when a mode amplitude exceeds the Bessel range.
//!
//! The reciprocal-space Fourier block is
//!
//! $$ H^{(n)}_{ij}(\mathbf k) =
//! \sum_{\mathbf R}
//! t_{ij}(\mathbf R)\,
//! C_n(\mathbf d_{ij\mathbf R})\,
//! e^{i2\pi\mathbf k\cdot\mathbf R}. $$
//!
//! `Gauge::Lattice` returns this block directly.  `Gauge::Atom` applies the
//! same orbital-position phase convention as [`Model::gen_ham`].
//!
//! # Sambe Hamiltonian
//!
//! With photon sectors `n,m` in `[-n_max, n_max]`, the Floquet-Sambe
//! Hamiltonian is
//!
//! $$ \left[H_F(\mathbf k)\right]_{i n,j m} =
//! H^{(n-m)}_{ij}(\mathbf k)
//! +
//! n\Omega_0\,\delta_{nm}\delta_{ij}. $$
//!
//! The photon energy `Omega_0` is stored as `FloquetDrive::omega0_ev` in eV, so
//! the returned Floquet eigenvalues are also in eV.
//!
//! [`Floquet::floquet_band_onek`] returns the unfolded Sambe eigenvalues.
//! [`Floquet::floquet_quasienergy_onek`] folds them into the first Floquet zone
//! by
//!
//! $$ \varepsilon_F =
//! \left(\varepsilon+\frac{\Omega_0}{2}\right)\bmod \Omega_0
//! -
//! \frac{\Omega_0}{2}. $$
//!
//! # Van Vleck effective model (high-frequency expansion)
//!
//! When `Omega_0` is large compared to the bandwidth, the photon-dressed bands
//! are well separated and the physics can be captured by a **same-size** static
//! model obtained through the van Vleck expansion:
//!
//! Writing $W=\hbar\Omega_0=$ `FloquetDrive::omega0_ev`, the expansion is
//!
//! $$ H_{\mathrm{eff}}(\mathbf k) =
//! H^{(0)}(\mathbf k)
//! +
//! \sum_{n=1}^{n_{\max}}
//! \frac{[H^{(n)}(\mathbf k), H^{(-n)}(\mathbf k)]}{nW}
//! +
//! H_{\mathrm{eff}}^{(2)}(\mathbf k)
//! +O(W^{-3}), $$
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
//! model for band plotting, cuts, or other existing `Model` workflows.  The
//! one-`k` methods are convenience wrappers for direct Sambe diagonalization.
//!
//! | Type / method | Meaning |
//! |---------------|---------|
//! | [`LightMode`] | One harmonic component `(harmonic, a_complex)` |
//! | [`FloquetDrive`] | Base photon energy plus all light modes |
//! | [`FloquetTruncation`] | Photon cutoff and time-Fourier grid |
//! | [`IncidentBasis`] | 3D transverse basis from an incident direction |
//! | [`FloquetEffectiveOptions`] | Optional order, harmonic cutoff, and real-space truncation |
//! | [`Floquet::floquet_model`] | Build an enlarged static Sambe tight-binding model |
//! | [`Model::floquet_effective_model`] | Build a same-size high-frequency effective model |
//! | [`Floquet::floquet_ham_onek`] | Build the Sambe Hamiltonian at one `k` |
//! | [`Floquet::floquet_band_onek`] | Diagonalize the Sambe Hamiltonian |
//! | [`Floquet::floquet_quasienergy_onek`] | Diagonalize and fold quasienergies |
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
//!     let trunc = FloquetTruncation::new(1, 128);
//!     let k = arr1(&[0.25, 0.0, 0.0]);
//!
//!     let floquet_model = model.floquet_model(&drive, &trunc)?;
//!     let unfolded = floquet_model.solve_band_onek(&k);
//!     let quasienergies = model.floquet_quasienergy_onek(&k, &drive, &trunc, Gauge::Lattice)?;
//!     println!("unfolded Sambe bands = {unfolded:?}");
//!     println!("folded quasienergies = {quasienergies:?}");
//!     Ok(())
//! }
//! ```
//!
//! See also `examples/floquet_chain/main.rs`.

use crate::error::{Result, TbError};
use crate::model::NoRMatrix;
use crate::model_utils::find_R;
use crate::ndarray_lapack::eigvalsh_v;
use crate::{Gauge, Model, OrbitalId, RMatrixData};
use ndarray::parallel::prelude::IntoParallelIterator;
use ndarray::prelude::*;
use ndarray::*;
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
/// `n_time` is the number of samples per drive period used by the Sambe and
/// time-grid paths for the discrete Fourier transform of the Peierls
/// coefficients `C_n(d)`.  Increase it when the drive amplitude or the
/// maximum harmonic is large.
///
/// The van Vleck effective-model path (`Model::floquet_effective_model`,
/// Bessel backend) does **not** use the value of `n_time` for the
/// computation (it is only validated to be positive as a shared-type
/// invariant): links outside the Bessel range fall back to a per-link
/// time-grid DFT whose resolution is sized from the link's own spectral
/// bandwidth and the requested harmonic range, clamped to `2^20` points
/// with a warn-once message.
#[derive(Clone, Copy, Debug)]
pub struct FloquetTruncation {
    /// Photon cutoff `N`.
    pub n_max: isize,
    /// Number of time samples in one drive period.
    ///
    /// Used by the Sambe and time-grid paths; the van Vleck
    /// effective-model path sizes its own per-link fallback grid and does
    /// not use this field's value for the computation (it must still be
    /// positive as a type invariant).
    pub n_time: usize,
}

impl FloquetTruncation {
    pub fn new(n_max: isize, n_time: usize) -> Self {
        Self { n_max, n_time }
    }

    #[inline]
    pub fn n_sector(&self) -> usize {
        (2 * self.n_max + 1) as usize
    }

    #[inline]
    pub fn sectors(&self) -> impl Iterator<Item = isize> {
        -self.n_max..=self.n_max
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
    /// Signed-harmonic cutoff for commutator sums.  Defaults to
    /// `2 * trunc.n_max`.  At order 2 the mixed component `H_(m'-m)` is
    /// evaluated up to `|m'-m| = 2 * harmonic_max` automatically.
    pub harmonic_max: Option<isize>,
    /// Optional target real-space hopping vectors for the legacy path's
    /// inverse Fourier transform.  Rejected by the real-space path.
    pub(crate) target_hamR: Option<Array2<isize>>,
}

/// Strategy for combining mode contributions when a drive contains multiple
/// [`LightMode`]s.
#[derive(Clone, Debug)]
pub enum FloquetModeCombination {
    /// Treat all modes as a single coherent drive, identical to
    /// [`Model::floquet_effective_model`].
    Coherent,

    /// Build one-mode effective models and combine them with real weights,
    /// including residual static weight `1 - sum(weights)` in front of the
    /// zero-drive effective model.
    ///
    /// For weights `w_i`, the effective model is
    ///
    /// ```text
    /// H_eff = (1 - Σ_i w_i) H_eff^{(0)} + Σ_i w_i H_eff^{(i)}
    /// ```
    ///
    /// where `H_eff^{(0)}` is the static (`drive = 0`) effective model and
    /// `H_eff^{(i)}` is the effective model of mode `i` alone.
    ///
    /// This is suitable for constructing an incoherent superposition of light
    /// pulses/modes in the same base cell.
    /// `Σ_i w_i` must be finite and in `[0, 1]` by default.
    ModeResolvedIncoherent(Vec<f64>),

    /// Build one-mode effective models and combine them with real weights without
    /// adding a residual static contribution.
    ///
    /// For weights `w_i`, the effective model is
    ///
    /// ```text
    /// H_eff = Σ_i w_i H_eff^{(i)} .
    /// ```
    ///
    /// This is useful when the user intends a direct weighted sum of mode
    /// responses (for example, for intensity-weighted linear interpolation of
    /// independently computed mode responses).
    ModeResolvedIncoherentNoStatic(Vec<f64>),
}

/// Compatibility enum for the former, misleading mode-combination name.
#[deprecated(
    since = "0.7.2",
    note = "use FloquetModeCombination; finite-q is handled by Model::floquet_effective_q_model"
)]
#[derive(Clone, Debug)]
pub enum FloquetQModelMode {
    Coherent,
    ModeResolvedIncoherent(Vec<f64>),
    ModeResolvedIncoherentNoStatic(Vec<f64>),
}

#[allow(deprecated)]
impl From<FloquetQModelMode> for FloquetModeCombination {
    fn from(mode: FloquetQModelMode) -> Self {
        match mode {
            FloquetQModelMode::Coherent => Self::Coherent,
            FloquetQModelMode::ModeResolvedIncoherent(weights) => {
                Self::ModeResolvedIncoherent(weights)
            }
            FloquetQModelMode::ModeResolvedIncoherentNoStatic(weights) => {
                Self::ModeResolvedIncoherentNoStatic(weights)
            }
        }
    }
}

/// Input accepted by [`Model::floquet_effective_q_model`].
///
/// Passing a Cartesian wavevector selects the coherent `O(A^2 q / W)`
/// calculation.  The mode-combination variant preserves source compatibility
/// with the earlier mode-resolved helper; new code should call
/// [`Model::floquet_effective_mode_resolved_model`] directly for that task.
#[doc(hidden)]
pub enum FloquetQModelInput<'a> {
    WavevectorCartesian(&'a Array1<f64>),
    LegacyModeCombination(FloquetModeCombination),
}

impl<'a> From<&'a Array1<f64>> for FloquetQModelInput<'a> {
    fn from(wavevector: &'a Array1<f64>) -> Self {
        Self::WavevectorCartesian(wavevector)
    }
}

impl<'a> From<FloquetModeCombination> for FloquetQModelInput<'a> {
    fn from(mode: FloquetModeCombination) -> Self {
        Self::LegacyModeCombination(mode)
    }
}

#[allow(deprecated)]
impl<'a> From<FloquetQModelMode> for FloquetQModelInput<'a> {
    fn from(mode: FloquetQModelMode) -> Self {
        Self::LegacyModeCombination(mode.into())
    }
}

type RealSpaceBlocks = (Vec<Array2<Complex<f64>>>, Array2<isize>);
type RealSpaceBlockMap = std::collections::BTreeMap<Vec<isize>, Array2<Complex<f64>>>;

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
            harmonic_max: None,
            target_hamR: None,
        }
    }
}

impl FloquetEffectiveOptions {
    /// Construct first-order options using `harmonic_max = 2 * trunc.n_max` and the
    /// original model's `hamR`.
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
        self.harmonic_max = Some(harmonic_max);
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

fn validate_incoherent_weights(mode_weights: &[f64], include_static_residual: bool) -> Result<f64> {
    if mode_weights.is_empty() {
        return Ok(0.0);
    }

    let mut total = 0.0;
    for &w in mode_weights {
        if !w.is_finite() {
            return Err(TbError::Other(format!(
                "floquet_effective_mode_resolved_model: non-finite mode weight {w}"
            )));
        }
        if w < 0.0 {
            return Err(TbError::Other(format!(
                "floquet_effective_mode_resolved_model: negative mode weight {w}"
            )));
        }
        total += w;
    }

    if !total.is_finite() {
        return Err(TbError::Other(
            "floquet_effective_mode_resolved_model: total mode weight overflowed".to_string(),
        ));
    }

    if include_static_residual && total > 1.000_000_000_000_001 {
        return Err(TbError::Other(format!(
            "floquet_effective_mode_resolved_model: mode weights exceed unit total ({total}); expected Σw <= 1"
        )));
    }

    Ok(total)
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
    pub fn floquet_effective_q_model<'a, Q>(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        options: Option<&FloquetEffectiveOptions>,
        q_model_input: Q,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>>
    where
        Q: Into<FloquetQModelInput<'a>>,
    {
        match q_model_input.into() {
            FloquetQModelInput::WavevectorCartesian(wavevector_cartesian) => {
                self.floquet_effective_q_linear_model(drive, wavevector_cartesian, trunc, options)
            }
            FloquetQModelInput::LegacyModeCombination(mode) => {
                self.floquet_effective_mode_resolved_model(drive, trunc, options, mode)
            }
        }
    }

    fn floquet_effective_q_linear_model(
        &self,
        drive: &FloquetDrive,
        wavevector_cartesian: &Array1<f64>,
        trunc: &FloquetTruncation,
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

        let mut effective = self.floquet_effective_model(drive, trunc, options)?;
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
        let harmonic_max = effective_harmonic_max(options, trunc)?;
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

    /// Build a Floquet high-frequency effective model with explicit mode
    /// combination strategy.
    ///
    /// This entry is a name that describes the actual behavior:
    /// combining one-mode channels computed by [`Model::floquet_effective_model`].
    ///
    /// The method returns a same-size model and uses the existing real-space
    /// Bessel backend underneath.
    pub fn floquet_effective_mode_resolved_model<M>(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        options: Option<&FloquetEffectiveOptions>,
        mode: M,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>>
    where
        M: Into<FloquetModeCombination>,
    {
        match mode.into() {
            FloquetModeCombination::Coherent => self.floquet_effective_model(drive, trunc, options),
            FloquetModeCombination::ModeResolvedIncoherent(weights) => self
                .floquet_effective_mode_resolved_incoherent_impl(
                    drive, trunc, options, weights, true,
                ),
            FloquetModeCombination::ModeResolvedIncoherentNoStatic(weights) => self
                .floquet_effective_mode_resolved_incoherent_impl(
                    drive, trunc, options, weights, false,
                ),
        }
    }

    fn floquet_effective_mode_resolved_incoherent_impl(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        options: Option<&FloquetEffectiveOptions>,
        weights: Vec<f64>,
        include_static: bool,
    ) -> Result<Model<SPIN, DIM, NoRMatrix>> {
        if weights.len() != drive.modes.len() {
            return Err(TbError::Other(format!(
                "floquet_effective_mode_resolved_model incoherent weights must match mode count: {} != {}",
                weights.len(),
                drive.modes.len()
            )));
        }

        let total_weight = validate_incoherent_weights(&weights, include_static)?;
        let default_options;
        let options = match options {
            Some(options) => options,
            None => {
                default_options = FloquetEffectiveOptions::default();
                &default_options
            }
        };
        validate_floquet_drive::<DIM>(drive, trunc)?;
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
        effective_harmonic_max(options, trunc)?;
        let nsta = self.nsta();

        let (mut effective, mut union_rows, mut union_blocks) = if include_static {
            // With no drive, every van Vleck correction vanishes exactly, so
            // the static effective model is just the input Hamiltonian with
            // its position-matrix tag removed.
            let mut static_model = Model::<SPIN, DIM, NoRMatrix>::tb_model(
                self.lat.clone(),
                self.orb.clone(),
                Some(self.atoms.clone()),
            )?;
            static_model.orb_projection = self.orb_projection.clone();
            let union_rows = self
                .hamR
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let mut union_blocks = self
                .ham
                .axis_iter(Axis(0))
                .map(|block| block.to_owned())
                .collect::<Vec<_>>();

            // The validator permits a tiny roundoff excess above one.  Clamp
            // that tolerance window instead of introducing a negative static
            // contribution.
            let static_weight = (1.0 - total_weight).max(0.0);
            for block in union_blocks.iter_mut() {
                block.mapv_inplace(|x| x * static_weight);
            }
            static_model.ham = Array3::zeros((0, nsta, nsta));
            static_model.hamR = Array2::zeros((0, DIM));
            (static_model, union_rows, union_blocks)
        } else {
            // Preserve a conventional zero R=0 block without evaluating a
            // zero-drive Floquet model that will immediately be discarded.
            let mut zero_model = Model::<SPIN, DIM, NoRMatrix>::tb_model(
                self.lat.clone(),
                self.orb.clone(),
                Some(self.atoms.clone()),
            )?;
            zero_model.orb_projection = self.orb_projection.clone();
            let union_rows = zero_model
                .hamR
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let union_blocks = zero_model
                .ham
                .axis_iter(Axis(0))
                .map(|block| block.to_owned())
                .collect::<Vec<_>>();
            zero_model.ham = Array3::zeros((0, nsta, nsta));
            zero_model.hamR = Array2::zeros((0, DIM));
            (zero_model, union_rows, union_blocks)
        };

        let mut row_index = std::collections::BTreeMap::<Vec<isize>, usize>::new();
        for (idx, row) in union_rows.iter().enumerate() {
            row_index.insert(row.clone(), idx);
        }

        for (mode_weight, mode_def) in weights.iter().zip(drive.modes.iter()) {
            if *mode_weight == 0.0 {
                continue;
            }

            let single_drive = FloquetDrive::with_modes(drive.omega0_ev, vec![mode_def.clone()]);
            let single = self.floquet_effective_model(&single_drive, trunc, Some(options))?;

            for i_r in 0..single.ham.len_of(Axis(0)) {
                let key = single.hamR.row(i_r).to_vec();
                let union_idx = if let Some(idx) = row_index.get(&key).copied() {
                    idx
                } else {
                    let idx = union_rows.len();
                    union_rows.push(key.clone());
                    union_blocks.push(Array2::zeros((nsta, nsta)));
                    row_index.insert(key, idx);
                    idx
                };

                let src = single.ham.index_axis(Axis(0), i_r);
                let dst = &mut union_blocks[union_idx];
                for (output, input) in dst.iter_mut().zip(src.iter()) {
                    *output += *input * *mode_weight;
                }
            }
        }

        let mut ordered_support = union_rows.into_iter().zip(union_blocks).collect::<Vec<_>>();
        ordered_support.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let output_ham_views = ordered_support
            .iter()
            .map(|(_, block)| block.view())
            .collect::<Vec<_>>();
        effective.ham = ndarray::stack(Axis(0), &output_ham_views)?;
        effective.hamR = Array2::from_shape_fn((ordered_support.len(), DIM), |(i, d)| {
            ordered_support[i].0[d]
        });
        enforce_real_space_hermiticity(&mut effective.ham, &effective.hamR)?;
        effective.validate()?;
        Ok(effective)
    }
}

/// Backend selection for the Peierls Fourier coefficients `C_n(d)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PeierlsFourierMethod {
    /// Numerical DFT on a uniform time grid (`FloquetTruncation::n_time`
    /// samples).  The reference implementation: handles arbitrary drives,
    /// including non-commensurate content and large amplitudes.
    TimeGrid,
    /// Generalized Bessel expansion via sequential one-mode convolutions.
    /// Exact and independent of `n_time`, but restricted to per-mode
    /// projections `R_α = |a_α·d| ≤ 8`; the cache falls back to
    /// [`PeierlsFourierMethod::TimeGrid`] per link beyond that.
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

/// Precomputed time-grid data for the discrete Fourier integration of
/// Peierls coefficients `C_n(d)`.
///
/// `link_field[it, a]` stores the real part of the total dimensionless
/// vector potential `a_a(t_it)` for each time step and spatial direction.
/// `fourier[i_n, it]` stores `exp(i * n * theta)` for each harmonic and time step.
/// Building these once avoids recomputing the same exponentials for every
/// hopping link.
struct FloquetTimeGrid {
    link_field: Array2<f64>,
    fourier: Array2<Complex<f64>>,
    inv_n_time: f64,
}

impl FloquetTimeGrid {
    fn new(
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        harmonic_min: isize,
        harmonic_max: isize,
        dim: usize,
    ) -> Self {
        let n_time = trunc.n_time;
        let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
        let inv_n_time = 1.0 / (n_time as f64);
        let mut link_field = Array2::<f64>::zeros((n_time, dim));
        let mut fourier = Array2::<Complex<f64>>::zeros((harmonic_count, n_time));

        for it in 0..n_time {
            let theta = TAU * (it as f64) * inv_n_time;
            for mode in &drive.modes {
                let harmonic_phase = Complex::new(0.0, -(mode.harmonic as f64) * theta).exp();
                for a in 0..dim {
                    link_field[[it, a]] += (mode.a_complex[a] * harmonic_phase).re;
                }
            }
            for (i_n, n) in (harmonic_min..=harmonic_max).enumerate() {
                fourier[[i_n, it]] = Complex::new(0.0, (n as f64) * theta).exp();
            }
        }

        Self {
            link_field,
            fourier,
            inv_n_time,
        }
    }
}

/// Peierls-Floquet Sambe construction for tight-binding models.
pub trait Floquet {
    /// Static model type produced by [`Floquet::floquet_model`].
    type FloquetModel;

    /// Build an enlarged static tight-binding model in Sambe space.
    ///
    /// The returned model has the same spatial lattice and hopping range as
    /// the original model, but its internal basis is enlarged from
    /// `N_state` to
    ///
    /// $$
    /// N_{\mathrm{state}}(2N+1),
    /// $$
    ///
    /// where `N = trunc.n_max`.  Photon sectors run from `-N` to `N`.
    /// Spinless models are ordered as `(photon sector, orbital)`.  Spinful
    /// models preserve the usual Rustb spin layout and are ordered as
    /// `(spin, photon sector, orbital)`.
    ///
    /// The real-space matrix elements are
    ///
    /// $$ \langle i,n;\mathbf 0|H_F|j,m;\mathbf R\rangle =
    /// t_{ij}(\mathbf R) C_{n-m}(\mathbf d_{ij\mathbf R})
    /// +
    /// n\Omega_0\delta_{nm}\delta_{ij}\delta_{\mathbf R,0}. $$
    ///
    /// The result preserves the input model's `SPIN` const generic.  Photon
    /// sectors are encoded as additional orbitals; if the input model is
    /// spinful, physical spin remains the `Model<true, DIM, _>` spin degree of
    /// freedom rather than being flattened away.
    ///
    /// This model is stored in real space.  Calling
    /// `floquet_model.gen_ham(k, Gauge::Lattice)` is equivalent to
    /// [`Floquet::floquet_ham_onek`] with `Gauge::Lattice`; using
    /// `Gauge::Atom` applies the same atomic gauge phase to the enlarged
    /// orbital positions.
    fn floquet_model(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
    ) -> Result<Self::FloquetModel>;

    /// Build the full Sambe Hamiltonian at one fractional k point.
    ///
    /// The returned matrix has shape
    ///
    /// $$
    /// \bigl(N_{\mathrm{state}}(2N+1),\,N_{\mathrm{state}}(2N+1)\bigr),
    /// $$
    ///
    /// where `N = trunc.n_max`.
    ///
    /// The block convention is
    ///
    /// $$ \left[H_F\right]_{i n,j m} =
    /// H^{(n-m)}_{ij}(\mathbf k)
    /// +
    /// n\Omega_0\delta_{nm}\delta_{ij}. $$
    fn floquet_ham_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array2<Complex<f64>>>;

    /// Diagonalize [`Floquet::floquet_ham_onek`] and return unfolded Sambe
    /// eigenvalues.
    ///
    /// These values are not unique modulo `omega0_ev`; use
    /// [`Floquet::floquet_quasienergy_onek`] when the first Floquet zone is
    /// desired.
    fn floquet_band_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array1<f64>>;

    /// Return quasienergies folded into the first Floquet zone.
    ///
    /// The folding convention is
    ///
    /// $$
    /// \varepsilon_F \in [-\Omega_0/2,\Omega_0/2).
    /// $$
    fn floquet_quasienergy_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array1<f64>>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Floquet for Model<SPIN, DIM, R> {
    type FloquetModel = Model<SPIN, DIM, NoRMatrix>;

    fn floquet_model(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
    ) -> Result<Self::FloquetModel> {
        validate_floquet_drive::<DIM>(drive, trunc)?;

        let nsta = self.nsta();
        let norb = self.norb();
        let sectors: Vec<isize> = trunc.sectors().collect();
        let n_sector = sectors.len();
        let new_norb = norb * n_sector;
        let total = nsta * n_sector;
        let basis_indices = floquet_basis_indices::<SPIN>(nsta, norb, n_sector);
        let harmonic_min = -2 * trunc.n_max;
        let harmonic_max = 2 * trunc.n_max;
        let harmonic_cache = self.floquet_harmonic_cache(
            drive,
            trunc,
            harmonic_min,
            harmonic_max,
            &PeierlsFourierMethod::TimeGrid,
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
                        if harmonic_cache
                            .blocks
                            .slice(s![.., i_r, i, j])
                            .iter()
                            .all(|x| x.norm_sqr() == 0.0)
                        {
                            continue;
                        }

                        for (in_sec, &n) in sectors.iter().enumerate() {
                            let row = basis_indices[in_sec][i];
                            for (im_sec, &m) in sectors.iter().enumerate() {
                                let hopping = harmonic_cache.blocks
                                    [[harmonic_cache.harmonic_index(n - m), i_r, i, j]];
                                if hopping.norm_sqr() == 0.0 {
                                    continue;
                                }
                                let col = basis_indices[im_sec][j];
                                out[[row, col]] += hopping;
                            }
                        }
                    }
                }
            });

        let zero_r = Array1::<isize>::zeros(DIM);
        let onsite_index = match find_R(&ham_r, &zero_r) {
            Some(index) => index,
            None => {
                ham.push(
                    Axis(0),
                    Array2::<Complex<f64>>::zeros((total, total)).view(),
                )
                .unwrap();
                ham_r.push_row(zero_r.view()).unwrap();
                ham.len_of(Axis(0)) - 1
            }
        };

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

        Ok(model)
    }

    fn floquet_ham_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array2<Complex<f64>>> {
        validate_floquet_input(self, kvec, drive, trunc)?;

        let nsta = self.nsta();
        let norb = self.norb();
        let n_sector = trunc.n_sector();
        let total = nsta * n_sector;
        let mut hamf = Array2::<Complex<f64>>::zeros((total, total));
        let basis_indices = floquet_basis_indices::<SPIN>(nsta, norb, n_sector);

        let harmonic_min = -2 * trunc.n_max;
        let harmonic_max = 2 * trunc.n_max;
        let harmonic_cache = self.floquet_harmonic_cache(
            drive,
            trunc,
            harmonic_min,
            harmonic_max,
            &PeierlsFourierMethod::TimeGrid,
        );
        let harmonics: Vec<Array2<Complex<f64>>> = (harmonic_min..=harmonic_max)
            .map(|n| self.floquet_cached_harmonic_onek(kvec, n, gauge, &harmonic_cache))
            .collect();

        for (in_sec, n) in trunc.sectors().enumerate() {
            for (im_sec, m) in trunc.sectors().enumerate() {
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

        Ok(hamf)
    }

    fn floquet_band_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array1<f64>> {
        let hamf = self.floquet_ham_onek(kvec, drive, trunc, gauge)?;
        Ok(eigvalsh_v(&hamf, UPLO::Upper))
    }

    fn floquet_quasienergy_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        gauge: Gauge,
    ) -> Result<Array1<f64>> {
        let mut values = self.floquet_band_onek(kvec, drive, trunc, gauge)?;
        values.mapv_inplace(|x| fold_quasienergy(x, drive.omega0_ev));
        values
            .as_slice_mut()
            .unwrap()
            .sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(values)
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
    /// `harmonic_max = 2 * trunc.n_max`, and the input model's original `hamR`.
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
        trunc: &FloquetTruncation,
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

        validate_floquet_drive::<DIM>(drive, trunc)?;
        validate_effective_options::<DIM>(&k_mesh, options)?;

        let nsta = self.nsta();
        let target_ham_r = options
            .target_hamR
            .clone()
            .unwrap_or_else(|| self.hamR.clone());
        validate_target_hamr::<DIM>(&target_ham_r)?;

        let harmonic_max = effective_harmonic_max(options, trunc)?;
        let has_time_dependence = drive_has_time_dependent_field::<DIM>(drive);
        let static_drive;
        let cache_drive = if has_time_dependence {
            drive
        } else {
            static_drive = static_component_drive::<DIM>(drive);
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
            trunc,
            -cache_max,
            cache_max,
            &PeierlsFourierMethod::TimeGrid,
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
    /// generalized Bessel backend — no `k_mesh` parameter, and the value
    /// of `n_time` does not affect the computation (it is only validated
    /// to be positive): links whose amplitude exceeds the Bessel range
    /// (`R > 8`) fall back to a per-link time-grid DFT whose resolution is
    /// sized from the link's own spectral bandwidth and the requested
    /// harmonic range.
    ///
    /// This is the main entry point.  The crate-internal
    /// `floquet_effective_model_legacy` is the k-space reference
    /// implementation, kept for cross-validation tests.
    ///
    /// The effective hopping blocks are built entirely in real space
    /// (`FLOQUET_REAL_SPACE_PLAN.md` §3):
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
    /// indices `m,m'` are truncated at `harmonic_max` (default `2 * trunc.n_max`),
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
    /// Returns an error for an invalid drive or truncation, `order > 2`, a
    /// negative or unrepresentable harmonic range, a non-finite frequency
    /// scaling factor, a supplied `target_hamR`, a real-space support sum that
    /// overflows `isize`, or a support that is not closed under `R -> −R`.
    pub fn floquet_effective_model(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
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

        validate_floquet_drive::<DIM>(drive, trunc)?;
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
        let harmonic_max = effective_harmonic_max(options, trunc)?;

        let nsta = self.nsta();
        let has_time_dependence = drive_has_time_dependent_field::<DIM>(drive);
        let static_drive;
        let cache_drive = if has_time_dependence {
            drive
        } else {
            static_drive = static_component_drive::<DIM>(drive);
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
            trunc,
            -cache_max,
            cache_max,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
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

        // With no time-dependent field every nonzero harmonic vanishes.  The
        // documented support still depends on the requested order, so build
        // each zero-valued Minkowski layer once, independent of harmonic_max.
        if !has_time_dependence && harmonic_max > 0 && options.order >= 1 {
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

        // First order: sum over n of comm_n/(n·ħΩ₀); omega0_ev carries
        // the ħΩ₀ energy (same convention as the legacy k-space path).
        let inverse_omega = drive.omega0_ev.recip();
        if options.order >= 1 && has_time_dependence {
            for n in 1..=harmonic_max {
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
                .filter(|harmonic| *harmonic != 0)
                .collect::<Vec<_>>();

            // Each fixed-m contribution is independent.  Accumulate one
            // private real-space map per Rayon job and merge maps only after
            // all nested commutators for that job are complete.  This exposes
            // the abundant harmonic-level parallelism of the O(omega^-2)
            // double sum without locking the output map in the hot loop.
            let accumulate_fixed_m = |partial: &mut RealSpaceBlockMap, m: isize| -> Result<()> {
                let h_zero = harmonic_cache.harmonic_blocks(0);
                let h_m = harmonic_cache.harmonic_blocks(m);
                let (inner, inner_r) =
                    real_space_commutator_with_supports(&h_zero, &self.hamR, &h_m, &self.hamR)?;
                let h_minus_m = harmonic_cache.harmonic_blocks(-m);
                let (outer, outer_r) =
                    real_space_commutator_with_supports(&h_minus_m, &self.hamR, &inner, &inner_r)?;
                let scale = inverse_omega_squared / (2.0 * (m as f64).powi(2));
                accumulate_scaled_real_space_blocks(partial, &outer, &outer_r, scale)?;

                for &m_prime in &signed_harmonics {
                    if m_prime == m {
                        continue;
                    }
                    let h_difference = harmonic_cache.harmonic_blocks(m_prime - m);
                    let h_m = harmonic_cache.harmonic_blocks(m);
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
                let min_harmonics_per_job = signed_harmonics
                    .len()
                    .div_ceil(rayon::current_num_threads());
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
        enforce_real_space_hermiticity(&mut ham, &ham_r)?;

        let mut model = Model::<SPIN, DIM, NoRMatrix>::tb_model(
            self.lat.clone(),
            self.orb.clone(),
            Some(self.atoms.clone()),
        )?;
        model.ham = ham;
        model.hamR = ham_r;
        model.orb_projection = self.orb_projection.clone();

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
    /// cache is constructed once and reused across the whole k-mesh, avoiding the
    /// recomputation of `C_n(d)` for every k-point.
    ///
    /// When `drive.modes` is empty (static limit), the zero-frequency block is
    /// set to the original `ham` directly.
    fn floquet_harmonic_cache(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
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
                    if self.ham[[i_r, i, j]].norm_sqr() == 0.0 {
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

        // Phase 2: coefficients per distinct d (parallel).  The shared time
        // grid is built only for the TimeGrid backend; the Bessel backend
        // sizes its own per-link fallback grid from the link's bandwidth, so
        // no eager grid is allocated on that path.
        let time_grid = match method {
            PeierlsFourierMethod::TimeGrid => Some(FloquetTimeGrid::new(
                drive,
                trunc,
                harmonic_min,
                harmonic_max,
                DIM,
            )),
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
                                DIM,
                                &fallback_clamped,
                                &fallback_saturated,
                            )
                        }
                    }
                }
                PeierlsFourierMethod::TimeGrid => {
                    let time_grid = time_grid
                        .as_ref()
                        .expect("shared time grid is built for the TimeGrid backend");
                    Array1::from(peierls_fourier_coeffs(
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
                if coeff.norm_sqr() != 0.0 {
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
pub fn fold_quasienergy(energy: f64, omega0_ev: f64) -> f64 {
    (energy + 0.5 * omega0_ev).rem_euclid(omega0_ev) - 0.5 * omega0_ev
}

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
    if model.lat.nrows() != DIM || model.lat.ncols() != DIM {
        return Err(TbError::InvalidArrayShape {
            expected: vec![DIM, DIM],
            found: vec![model.lat.nrows(), model.lat.ncols()],
        });
    }
    validate_floquet_drive::<DIM>(drive, trunc)
}

fn validate_floquet_drive<const DIM: usize>(
    drive: &FloquetDrive,
    trunc: &FloquetTruncation,
) -> Result<()> {
    if !drive.omega0_ev.is_finite() || drive.omega0_ev <= 0.0 {
        return Err(TbError::InvalidEnergyRange {
            min: 0.0,
            max: drive.omega0_ev,
        });
    }
    if trunc.n_max < 0 {
        return Err(TbError::Other(format!(
            "FloquetTruncation.n_max must be non-negative, got {}",
            trunc.n_max
        )));
    }
    if trunc.n_time == 0 {
        return Err(TbError::Other(
            "FloquetTruncation.n_time must be positive".to_string(),
        ));
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

fn effective_harmonic_max(
    options: &FloquetEffectiveOptions,
    trunc: &FloquetTruncation,
) -> Result<isize> {
    let harmonic_max = match options.harmonic_max {
        Some(harmonic_max) => harmonic_max,
        None => trunc.n_max.checked_mul(2).ok_or_else(|| {
            TbError::Other(
                "FloquetTruncation.n_max is too large for the default effective harmonic cutoff"
                    .to_string(),
            )
        })?,
    };
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

fn drive_has_time_dependent_field<const DIM: usize>(drive: &FloquetDrive) -> bool {
    let mut amplitudes = std::collections::BTreeMap::<usize, [Complex<f64>; DIM]>::new();
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            continue;
        }
        let harmonic = mode.harmonic.unsigned_abs();
        let entry = amplitudes
            .entry(harmonic)
            .or_insert([Complex::new(0.0, 0.0); DIM]);
        for (component, amplitude) in entry.iter_mut().zip(mode.a_complex.iter()) {
            *component += if mode.harmonic > 0 {
                *amplitude
            } else {
                amplitude.conj()
            };
        }
    }
    amplitudes.values().any(|amplitude| {
        amplitude
            .iter()
            .any(|value| value.re != 0.0 || value.im != 0.0)
    })
}

fn static_component_drive<const DIM: usize>(drive: &FloquetDrive) -> FloquetDrive {
    let mut amplitude = Array1::<Complex<f64>>::zeros(DIM);
    for mode in &drive.modes {
        if mode.harmonic == 0 {
            for (total, value) in amplitude.iter_mut().zip(mode.a_complex.iter()) {
                total.re += value.re;
            }
        }
    }
    let modes = if amplitude
        .iter()
        .any(|value| value.re != 0.0 || value.im != 0.0)
    {
        vec![LightMode::new(0, amplitude)]
    } else {
        Vec::new()
    };
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
    if let Some(harmonic_max) = options.harmonic_max {
        if harmonic_max < 0 {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.harmonic_max must be non-negative, got {harmonic_max}"
            )));
        }
    }
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
/// MIT/Apache-2.0; measured worst relative error ~2e-15 over the Floquet
/// backend's range `r ≤ 8`, `|m| ≤ ~24`).  Negative orders use the
/// symmetry
///
/// ```math
/// J_{-m}(r) = (-1)^m J_m(r).
/// ```
///
/// Cross-checked in tests against an independent Miller downward-recurrence
/// reference and tabulated NIST values.
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

/// Two-sided Bessel tail `2·Σ_{m>M} |J_m(r)|`, accumulated until the terms
/// decay below the noise floor (bounded at 200 orders).  Shared by the
/// adaptive cutoff and the fallback saturation re-check so the two cannot
/// desync.
fn bessel_two_sided_tail(m: isize, r: f64) -> f64 {
    let mut tail = 0.0;
    let mut current = bessel_j(m + 1, r).abs();
    for order in (m + 2)..(m + 201) {
        tail += current;
        if current < 1e-20 {
            break;
        }
        current = bessel_j(order, r).abs();
    }
    2.0 * tail
}

/// Adaptive Bessel order cutoff for amplitude `r`: the smallest
/// `M ≥ ⌈r⌉ + margin` such that the two-sided tail `2·Σ_{m>M} |J_m(r)|`
/// stays at or below `error_share`.  Used by [`bessel_peierls_coeffs`]
/// (with `r ≤ 8`) and by the time-grid fallback sizing for arbitrary `r`;
/// there, `margin` doubles as a higher starting estimate.
///
/// The growth loop is bounded at `m = 4096`: a tail that still exceeds the
/// share there means the input amplitude is beyond any practical use (the
/// fallback sizing clamps its grid and warns).  `r` must be finite and
/// non-negative; huge `r` saturates the start estimate at 4096.
fn bessel_adaptive_m_cap(r: f64, error_share: f64, margin: isize) -> isize {
    debug_assert!(
        r.is_finite() && r >= 0.0,
        "bessel_adaptive_m_cap: r must be finite and non-negative"
    );
    // The float-to-int cast saturates for huge r; saturating_add keeps the
    // +margin step overflow-free before the growth loop clamps at 4096.
    let mut m_cap = (r.ceil() as isize).saturating_add(margin).min(4096);
    while m_cap <= 4096 {
        if bessel_two_sided_tail(m_cap, r) <= error_share {
            return m_cap;
        }
        m_cap += 1;
    }
    m_cap.min(4096)
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
/// `e^{+imδ}` must not be mixed in.)  The multi-index sum is evaluated as a
/// sequence of one-mode discrete convolutions
///
/// ```math
/// S^{(0)}_n = δ_{n,0},\qquad
/// S^{(α)}_n = \sum_{m=-M_α}^{M_α} S^{(α-1)}_{n + l_α m}\, B_α(m),
/// \qquad
/// B_α(m) = (-i)^m J_m(R_α)\, e^{-imδ_α},
/// ```
///
/// which costs `O(N_mode · N_n · M_avg)` — independent of the time-grid
/// size.  Each mode's cutoff `M_α` is chosen adaptively so the truncated
/// tail `Σ_{|m|>M_α} |J_m(R_α)|` stays below a per-mode error share
/// (`1e-12 / N_mode`), with `cutoff_margin` as an additional minimum.
/// With `R_α ≤ 8` and `cutoff_margin ≤ 48` every cutoff stays below the
/// 64-order safety bound (`⌈8⌉ + 48 = 56`, and the tail there is already
/// far below the share, so the growth loop never runs).
///
/// Verified against the independent time-grid DFT
/// ([`peierls_fourier_coeffs`]) for linear, circular, elliptical, and
/// multi-harmonic drives to ~1e-15.
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
/// is outside `0..=48`, when any mode amplitude `R_α` exceeds 8 (the
/// caller must fall back to the time-grid backend), or when the harmonic
/// range / working window would overflow `isize`.
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

    // Two-pass construction.  First pass: per-mode projections and adaptive
    // cutoffs.  The Bessel path only supports R_α ≤ 8; the caller falls back
    // to the time-grid backend beyond that (plan §7).
    struct ModeData {
        r: f64,
        delta: f64,
        harmonic: isize,
        m_cap: isize,
    }
    let mut modes = Vec::<ModeData>::with_capacity(drive.modes.len());
    let error_share = 1e-12 / (drive.modes.len() as f64);
    let mut total_drift = 0_isize;
    for mode in &drive.modes {
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
        if r > 8.0 {
            return Err(TbError::Other(format!(
                "bessel_peierls_coeffs: mode amplitude R = {r:.3} exceeds the \
                 Bessel backend's range (R ≤ 8); use the time-grid backend"
            )));
        }
        // Adaptive cutoff: grow M until the two-sided Bessel tail
        // 2 * Σ_{m>M} |J_m(r)| falls below the per-mode error share.  With
        // R ≤ 8 and cutoff_margin ≤ 48 the result is provably ≤ 64
        // (⌈8⌉ + 48 = 56 and the tail there is already ~1e-64, far below
        // the share, so the growth loop never runs) — asserted explicitly.
        let m_cap = bessel_adaptive_m_cap(r, error_share, cutoff_margin);
        assert!(
            m_cap <= 64,
            "Bessel cutoff exceeded the safety cap for R = {r}"
        );
        let harmonic_abs = mode.harmonic.checked_abs().ok_or_else(|| {
            TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
        })?;
        total_drift = total_drift
            .checked_add(harmonic_abs.checked_mul(m_cap).ok_or_else(|| {
                TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
            })?)
            .ok_or_else(|| {
                TbError::Other("bessel_peierls_coeffs: harmonic drift overflow".to_string())
            })?;
        modes.push(ModeData {
            r,
            delta: z.arg(),
            harmonic: mode.harmonic,
            m_cap,
        });
    }

    // Second pass: the working window must cover the actual reachable
    // support [−drift, +drift] around [harmonic_min, harmonic_max], because intermediates
    // outside the requested range can fold back into it.
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

    let mut sequence = vec![Complex::new(0.0, 0.0); work_len];
    if (0_isize..work_len as isize).contains(&(0 - work_min)) {
        sequence[(0 - work_min) as usize] = Complex::new(1.0, 0.0);
    }

    for mode in &modes {
        // One-mode sequence B(m) = (-i)^m J_m(r) e^{-imδ}, m ∈ [-M, M].
        // Accumulate (-i)^m iteratively.
        let mut minus_i_power = Complex::new(1.0, 0.0); // (-i)^0
        // m_cap <= 64 by the assert in the first pass, so this cannot overflow.
        let mut b = Vec::<(isize, Complex<f64>)>::with_capacity((2 * mode.m_cap + 1) as usize);
        for m in 0..=mode.m_cap {
            let value = minus_i_power
                * bessel_j(m, mode.r)
                * Complex::from_polar(1.0, -(m as f64) * mode.delta);
            if m == 0 {
                b.push((0, value));
            } else {
                // B(-m) = (-i)^{-m} J_{-m}(r) e^{+imδ}
                //       = i^m · (-1)^m J_m(r) e^{+imδ}
                //       = (-i)^m J_m(r) e^{+imδ} (since i^m (-1)^m = (-i)^m)
                let neg = minus_i_power
                    * bessel_j(m, mode.r)
                    * Complex::from_polar(1.0, (m as f64) * mode.delta);
                b.push((-m, neg));
                b.push((m, value));
            }
            minus_i_power *= Complex::new(0.0, -1.0); // times (-i)
        }

        // Fold: S'_n = Σ_m S_{n + l·m} B(m).
        let mut next = vec![Complex::new(0.0, 0.0); work_len];
        for &(m, weight) in &b {
            let shift = mode.harmonic * m;
            for (index, _) in sequence.iter().enumerate() {
                let n = work_min + index as isize;
                // Sources outside the working window contribute nothing;
                // checked arithmetic also skips the (n, m) pairs whose
                // source would leave the isize range entirely.
                let Some(source) = n.checked_add(shift) else {
                    continue;
                };
                let Some(source_index) = source.checked_sub(work_min) else {
                    continue;
                };
                if source_index >= 0 && (source_index as usize) < work_len {
                    next[index] += sequence[source_index as usize] * weight;
                }
            }
        }
        sequence = next;
    }

    Ok(Array1::from(
        sequence[(harmonic_min - work_min) as usize..(harmonic_max - work_min + 1) as usize]
            .to_vec(),
    ))
}

fn peierls_fourier_coeffs(
    d_cart: &Array1<f64>,
    harmonic_min: isize,
    harmonic_max: isize,
    drive: &FloquetDrive,
    time_grid: &FloquetTimeGrid,
) -> Vec<Complex<f64>> {
    let harmonic_count = (harmonic_max - harmonic_min + 1) as usize;
    if drive.modes.is_empty() {
        let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
        if harmonic_min <= 0 && 0 <= harmonic_max {
            coeffs[(0 - harmonic_min) as usize] = Complex::new(1.0, 0.0);
        }
        return coeffs;
    }

    let mut coeffs = vec![Complex::new(0.0, 0.0); harmonic_count];
    for it in 0..time_grid.link_field.nrows() {
        let mut link_phase = 0.0;
        for a in 0..d_cart.len() {
            link_phase += time_grid.link_field[[it, a]] * d_cart[a];
        }
        let peierls = Complex::new(0.0, -link_phase).exp();
        for (i_n, coeff) in coeffs.iter_mut().enumerate() {
            *coeff += time_grid.fourier[[i_n, it]] * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= time_grid.inv_n_time;
    }
    coeffs
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
    /// A mode's adaptive cutoff saturated at 4096 orders, so the
    /// bandwidth estimate is only a lower bound.
    saturated: bool,
}

/// Size the alias-free fallback grid: `max(2·Σ_α |l_α|·M_α + 4,
/// 2·max(|harmonic_min|, |harmonic_max|) + 1)`, clamped to [`FALLBACK_GRID_MAX`].  The
/// first term resolves the link's signal bandwidth, the second the
/// requested harmonic range: an n-point DFT returns the aliased sum
/// `Σ_m C_{n+mn}`, which equals the true `C_n` only for `|n| < n/2`, so
/// every requested bin needs `n ≥ 2·max(|harmonic_min|, |harmonic_max|) + 1`.  Kept
/// separate from the DFT so tests can pin the exact sizing, the clamp,
/// and the saturation flag.
fn fallback_grid_size(
    drive: &FloquetDrive,
    d: &Array1<f64>,
    harmonic_min: isize,
    harmonic_max: isize,
) -> FallbackGridSize {
    // Nyquist bandwidth of the Peierls exponential on this link.
    let mut bandwidth = 0_usize;
    // Set when a mode's adaptive cutoff saturates at 4096 orders, so the
    // bandwidth estimate below is only a lower bound.
    let mut saturated = false;
    for mode in &drive.modes {
        let z: Complex<f64> = mode
            .a_complex
            .iter()
            .zip(d.iter())
            .map(|(a, d)| *a * *d)
            .sum();
        let r = z.norm();
        if r == 0.0 {
            continue;
        }
        if !r.is_finite() {
            // Degenerate amplitude (NaN/inf): skip the sizing and let the
            // fallback DFT propagate the NaN visibly instead of panicking
            // inside the Bessel order search.
            continue;
        }
        // Adaptive tail cutoff from the same family the Bessel path uses
        // (there the 1e-12 budget is split over the modes; here the full
        // budget is a conservative sizing estimate).  The margin of 48 is
        // a starting estimate that already meets the 1e-12 budget for
        // R ≲ 65.
        let m_cap = bessel_adaptive_m_cap(r, 1e-12, 48);
        // bessel_adaptive_m_cap saturates at 4096 orders.  When the
        // returned order does not actually meet the 1e-12 tail budget
        // (extreme amplitudes), the bandwidth estimate is a lower bound
        // and the sized grid would alias the missing high orders.  The
        // tail re-check shares bessel_two_sided_tail with the cutoff
        // itself; amplitudes beyond 2^19 cannot be resolved by any
        // fallback grid and are treated as saturated unconditionally.
        if m_cap >= 4096 && (r > 5.242_88e5 || bessel_two_sided_tail(m_cap, r) > 1e-12) {
            saturated = true;
        }
        let drift = (mode.harmonic.unsigned_abs() as usize).saturating_mul(m_cap as usize);
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
        // The bandwidth estimate is a lower bound: use the maximum grid
        // so the fold only involves exponentially small tails.
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
/// estimate saturates at its 4096-order cap (mode amplitude ≈ 4000, the
/// point where the two-sided tail beyond order 4096 still exceeds the
/// 1e-12 budget) the estimate is only a lower bound: the grid then uses
/// the maximum size and a warn-once message is printed, and alias-freedom
/// is limited to resolvable bandwidths (≲ 2^19).
fn fallback_time_grid_coeffs(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    harmonic_min: isize,
    harmonic_max: isize,
    dim: usize,
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
            "Floquet fallback bandwidth estimate saturated at 4096 Bessel \
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
    for it in 0..n_req {
        let theta = TAU * (it as f64) * inv_n;
        let mut link_phase = 0.0;
        for mode in &drive.modes {
            let harmonic_phase = Complex::new(0.0, -(mode.harmonic as f64) * theta).exp();
            for a in 0..dim {
                link_phase += (mode.a_complex[a] * harmonic_phase).re * d[a];
            }
        }
        let peierls = Complex::new(0.0, -link_phase).exp();
        for (i_n, n) in (harmonic_min..=harmonic_max).enumerate() {
            coeffs[i_n] += Complex::new(0.0, (n as f64) * theta).exp() * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= inv_n;
    }
    Array1::from(coeffs)
}

/// Row-major in-place accumulation `C += α·A·B` via BLAS `zgemm`.
///
/// ndarray stores row-major (C order) while BLAS is column-major; the
/// identity `(A·B)^T = B^T·A^T` turns the row-major product into a
/// column-major `zgemm('N', 'N')` over the same memory with the operands
/// swapped, so no transposition copies are needed.
///
/// # Panics
/// Debug-asserts that all three matrices are square `n x n` of one
/// common size.
fn zgemm_row_accumulate<SA, SB>(
    alpha: Complex<f64>,
    a: &ArrayBase<SA, Ix2>,
    b: &ArrayBase<SB, Ix2>,
    c: &mut Array2<Complex<f64>>,
) where
    SA: Data<Elem = Complex<f64>>,
    SB: Data<Elem = Complex<f64>>,
{
    let n = a.nrows();
    debug_assert_eq!(
        (a.ncols(), b.nrows(), b.ncols(), c.nrows(), c.ncols()),
        (n, n, n, n, n),
        "zgemm_row_accumulate: square n x n blocks required"
    );
    let n_i = n as i32;
    let beta = Complex::new(1.0, 0.0);
    // Safety: owned ndarray matrices are contiguous standard-layout
    // buffers of length n·n; the transpose trick above makes every
    // leading dimension equal to n.
    unsafe {
        blas::zgemm(
            b'N',
            b'N',
            n_i,
            n_i,
            n_i,
            alpha,
            b.as_slice().unwrap(),
            n_i,
            a.as_slice().unwrap(),
            n_i,
            beta,
            c.as_slice_mut().unwrap(),
            n_i,
        );
    }
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
/// harmonic pair `A_R = T_n(R)` and `B_R = T_{−n}(R)` (see
/// `FLOQUET_REAL_SPACE_PLAN.md` §3):
///
/// ```math
/// (AB)(R) = \sum_{R'} A_{R-R'}\, B_{R'}, \qquad
/// (BA)(R) = \sum_{R'} B_{R-R'}\, A_{R'}.
/// ```
///
/// Both convolutions are accumulated with BLAS `zgemm`
/// ([`zgemm_row_accumulate`]); the returned support is the Minkowski sum
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
            zgemm_row_accumulate(one, &a, &b, comm);
            zgemm_row_accumulate(reverse_scale, &b, &a, comm);
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
        let min_pairs_per_job = pair_count.div_ceil(rayon::current_num_threads());
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
            let herm = (&block + &hermitian_conjugate(&block)) * Complex::new(0.5, 0.0);
            ham.index_axis_mut(Axis(0), i_r).assign(&herm);
            visited[i_r] = true;
        } else {
            let block_i = ham.index_axis(Axis(0), i_r).to_owned();
            let block_j = ham.index_axis(Axis(0), j_r).to_owned();
            let avg = (&block_i + &hermitian_conjugate(&block_j)) * Complex::new(0.5, 0.0);
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

fn normalize3(v: &Array1<f64>) -> Result<Array1<f64>> {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm < 1e-14 {
        return Err(TbError::Other(
            "Cannot normalize a zero-length 3D vector".to_string(),
        ));
    }
    Ok(v.mapv(|x| x / norm))
}

fn cross3(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpinDirection;
    use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
    use crate::model::NoRMatrix;

    use crate::solve_ham::Solve;
    use ndarray::{arr1, array};

    #[test]
    fn bessel_j_matches_tabulated_values() {
        // Reference values (NIST DLMF, mpmath 50-digit evaluation).
        let table: [(isize, f64, f64); 12] = [
            (0, 0.0, 1.0),
            (1, 0.0, 0.0),
            (5, 0.0, 0.0),
            (0, 1.0, 0.76519768655796655145),
            (1, 1.0, 0.44005058574493351596),
            (5, 1.0, 0.00024975773021123443176),
            (0, 2.0, 0.22389077914123566805),
            (1, 2.0, 0.57672480775687338720),
            (2, 2.0, 0.35283402861563771915),
            (3, 2.0, 0.12894324947440205110),
            (0, 5.0, -0.17759677131433830435),
            (10, 5.0, 0.00146780264731047436),
        ];
        for (m, r, expected) in table {
            let got = bessel_j(m, r);
            assert!(
                (got - expected).abs() < 1e-14,
                "J_{m}({r}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn bessel_j_matches_independent_miller_reference() {
        // The planned Bessel backend operates up to r = 8 with orders up to
        // ~r + 16.  Cross-check the ascending series against an INDEPENDENT
        // algorithm: Miller's downward recurrence, normalized by the
        // identity J_0(r) + 2*sum_k J_{2k}(r) = 1 (stable in the direction
        // the series is not).
        let miller = |r: f64, mmax: usize| -> Vec<f64> {
            let start = mmax + 20;
            let mut next = 0.0_f64; // J_{start+1} ~ 0
            let mut current = 1.0_f64; // J_start (unscaled)
            let mut values = vec![0.0_f64; start + 1];
            for k in (0..start).rev() {
                values[k] = current;
                let k_prev = k as f64;
                // J_{k-1} = (2k/r) J_k - J_{k+1}
                let prev = if k == 0 {
                    0.0
                } else {
                    (2.0 * k_prev / r) * current - next
                };
                next = current;
                current = prev;
            }
            // Normalize: J_0(r) + 2 sum_{k>=1} J_{2k}(r) = 1.
            let j0_unscaled = values[0];
            let even_sum: f64 = values.iter().step_by(2).skip(1).sum::<f64>();
            let scale = 1.0 / (j0_unscaled + 2.0 * even_sum);
            values.iter().map(|v| v * scale).collect()
        };

        for r in [0.3, 1.0, 3.0, 5.0, 8.0] {
            let reference = miller(r, 20);
            for m in 0..=16 {
                let got = bessel_j(m as isize, r);
                assert!(
                    (got - reference[m]).abs() < 1e-11,
                    "J_{m}({r}) = {got}, Miller reference {}",
                    reference[m]
                );
            }
        }

        // Near a Bessel zero only absolute error is meaningful
        // (J_1 has a zero at 7.01559...).
        let near_zero = bessel_j(1, 7.015586669815619);
        assert!(
            near_zero.abs() < 1e-12,
            "J_1 near its zero must be tiny in absolute value, got {near_zero}"
        );
        // Tiny argument: J_0(1e-12) = 1 - 2.5e-25, J_1(1e-12) = 5e-13.
        // puruspe is accurate to ~1 ulp of 1.0 (2.2e-16), so assert the
        // 1-ulp bound rather than the exact analytic deviation.
        assert!((bessel_j(0, 1e-12) - 1.0).abs() < 3e-16);
        assert!((bessel_j(1, 1e-12) - 5e-13).abs() < 1e-25);
    }

    #[test]
    fn bessel_j_satisfies_recurrence_and_negative_order_symmetry() {
        // Recurrence: J_{m-1}(r) + J_{m+1}(r) = (2m/r) J_m(r).
        for r in [0.3, 0.7, 1.3, 2.5, 4.0, 7.0] {
            for m in 1..12 {
                let left = bessel_j(m - 1, r) + bessel_j(m + 1, r);
                let right = (2.0 * m as f64 / r) * bessel_j(m, r);
                assert!(
                    (left - right).abs() < 1e-12,
                    "recurrence failed at m={m}, r={r}: {left} vs {right}"
                );
            }
        }
        // Negative-order symmetry: J_{-m}(r) = (-1)^m J_m(r).
        for m in 1..8 {
            let expected = if m % 2 == 0 {
                bessel_j(m, 1.7)
            } else {
                -bessel_j(m, 1.7)
            };
            assert!((bessel_j(-m, 1.7) - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn bessel_coeffs_match_single_mode_closed_form() {
        // Single mode l=1: C_n = (-i)^n J_n(R) e^{+inδ} (verified reduction
        // of the generalized Bessel sum; the +iδ phase is the discriminating
        // one for complex amplitudes).
        let d = array![1.0];
        for (amplitude, name) in [
            (array![Complex::new(0.4, 0.0)], "linear"),
            (array![Complex::new(0.0, 0.4)], "circular"),
            (array![Complex::new(0.3, 0.2)], "elliptical"),
        ] {
            let drive = FloquetDrive::with_modes(0.8, vec![LightMode::new(1, amplitude.clone())]);
            let r = amplitude[0].norm();
            let delta = amplitude[0].arg();
            let coeffs = bessel_peierls_coeffs(&d, &drive, -6, 6, 6).unwrap();
            for n in -6..=6 {
                let expected = Complex::from_polar(1.0, -(n as f64) * std::f64::consts::FRAC_PI_2)
                    * bessel_j(n, r)
                    * Complex::from_polar(1.0, (n as f64) * delta);
                let got = coeffs[(n + 6) as usize];
                assert!(
                    (got - expected).norm() < 1e-13,
                    "{name}: C_{n} = {got}, closed form {expected}"
                );
            }
        }
    }

    #[test]
    fn bessel_coeffs_match_time_grid_dft() {
        // The strongest test: the generalized Bessel convolution must
        // reproduce the independent time-grid DFT for multi-mode and
        // multi-harmonic drives.
        let d = array![1.3, -0.7];
        let cases: Vec<FloquetDrive> = vec![
            FloquetDrive::with_modes(
                1.0,
                vec![
                    LightMode::new(1, array![Complex::new(0.25, 0.0), Complex::new(0.0, 0.25)]),
                    LightMode::new(2, array![Complex::new(0.1, 0.0), Complex::new(0.05, -0.05)]),
                ],
            ),
            FloquetDrive::with_modes(
                0.7,
                vec![
                    LightMode::new(1, array![Complex::new(0.3, 0.1), Complex::new(-0.1, 0.2)]),
                    LightMode::new(
                        -3,
                        array![Complex::new(0.08, -0.04), Complex::new(0.02, 0.06)],
                    ),
                ],
            ),
            FloquetDrive::with_modes(
                0.5,
                vec![
                    LightMode::new(1, array![Complex::new(0.2, 0.0), Complex::new(0.0, 0.2)]),
                    LightMode::new(2, array![Complex::new(0.05, 0.0), Complex::new(0.0, -0.05)]),
                    LightMode::new(
                        3,
                        array![Complex::new(0.02, 0.01), Complex::new(0.01, -0.02)],
                    ),
                ],
            ),
        ];
        for (case, drive) in cases.iter().enumerate() {
            let harmonic_min = -5_isize;
            let harmonic_max = 5_isize;
            let bessel = bessel_peierls_coeffs(&d, drive, harmonic_min, harmonic_max, 6).unwrap();
            let time_grid = FloquetTimeGrid::new(
                drive,
                &FloquetTruncation::new(3, 512),
                harmonic_min,
                harmonic_max,
                2,
            );
            let dft = peierls_fourier_coeffs(&d, harmonic_min, harmonic_max, drive, &time_grid);
            for (n, (got, expected)) in bessel.iter().zip(dft.iter()).enumerate() {
                assert!(
                    (got - expected).norm() < 1e-10,
                    "case {case}, n={}: Bessel {got} vs DFT {expected}",
                    harmonic_min + n as isize
                );
            }
        }
    }

    #[test]
    fn bessel_coeffs_handle_empty_drive() {
        let d = array![0.5];
        let drive = FloquetDrive::new(1.0);
        let coeffs = bessel_peierls_coeffs(&d, &drive, -3, 3, 6).unwrap();
        for n in -3..=3 {
            let expected = if n == 0 {
                Complex::new(1.0, 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
            assert!((coeffs[(n + 3) as usize] - expected).norm() < 1e-15);
        }
    }

    #[test]
    fn harmonic_cache_bessel_matches_time_grid_and_dedupes_links() {
        // Spinful 2-orbital model: the four spin blocks of every hopping
        // share the same link displacement, so the dedup path must produce
        // identical blocks to the non-dedup reference, and the Bessel
        // backend must agree with the time grid.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.3, 0.0]];
        let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);
        model.add_hop(
            Complex::new(0.1, 0.2),
            0,
            1,
            &array![1, 1],
            SpinDirection::X,
        );

        // Observable dedup premise: the model's non-zero hopping entries
        // share only 6 distinct link displacements (3 bonds and their
        // Hermitian partners at -R; the spin blocks of a bond share one d).
        let mut distinct_d = Vec::<Vec<u64>>::new();
        for i_r in 0..model.hamR.nrows() {
            let r_vec = model.hamR.row(i_r);
            for i in 0..model.nsta() {
                for j in 0..model.nsta() {
                    if model.ham[[i_r, i, j]].norm_sqr() == 0.0 {
                        continue;
                    }
                    let d_cart = model.link_displacement_cartesian(
                        i % model.norb(),
                        j % model.norb(),
                        &r_vec,
                    );
                    let key: Vec<u64> = d_cart.iter().map(|value| value.to_bits()).collect();
                    if !distinct_d.contains(&key) {
                        distinct_d.push(key);
                    }
                }
            }
        }
        assert_eq!(
            distinct_d.len(),
            6,
            "dedup premise: expected 6 distinct link displacements, found {}",
            distinct_d.len()
        );

        let drive = FloquetDrive::with_modes(
            0.8,
            vec![
                LightMode::new(1, array![Complex::new(0.2, 0.0), Complex::new(0.0, 0.2)]),
                LightMode::new(2, array![Complex::new(0.05, -0.05), Complex::new(0.0, 0.0)]),
            ],
        );
        let trunc = FloquetTruncation::new(2, 512);
        let time_grid_cache =
            model.floquet_harmonic_cache(&drive, &trunc, -4, 4, &PeierlsFourierMethod::TimeGrid);
        let bessel_cache = model.floquet_harmonic_cache(
            &drive,
            &trunc,
            -4,
            4,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        assert_eq!(time_grid_cache.blocks.dim(), bessel_cache.blocks.dim());
        for (a, b) in time_grid_cache
            .blocks
            .iter()
            .zip(bessel_cache.blocks.iter())
        {
            assert!(
                (a - b).norm() < 1e-10,
                "Bessel cache {b} vs time-grid cache {a}"
            );
        }
    }

    #[test]
    fn harmonic_cache_bessel_falls_back_for_large_amplitudes() {
        // |a·d| > 8 must silently fall back to the time grid per link, so
        // the Bessel-method cache still matches the time-grid cache.
        // The (0,1) hopping at R=(0,1) has d = (10, 1), so |a·d| = 9.0 > 8
        // and the fallback branch must actually execute.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [10.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);

        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let trunc = FloquetTruncation::new(1, 512);
        let time_grid_cache =
            model.floquet_harmonic_cache(&drive, &trunc, -3, 3, &PeierlsFourierMethod::TimeGrid);
        let bessel_cache = model.floquet_harmonic_cache(
            &drive,
            &trunc,
            -3,
            3,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        for (a, b) in time_grid_cache
            .blocks
            .iter()
            .zip(bessel_cache.blocks.iter())
        {
            assert!(
                (a - b).norm() < 1e-10,
                "fallback Bessel cache {b} vs time-grid cache {a}"
            );
        }
    }

    #[test]
    fn harmonic_cache_bessel_fallback_uses_alias_free_grid() {
        // R > 8 forces the per-link time-grid fallback.  A fixed
        // n_time = 512 aliases the high-order Bessel tails into the
        // wrong bins for l = 100, R = 50 (C_−8 ≈ −J_46(50) ≈ −0.17
        // instead of 0; C_0 = J_0(50) = 0.0558 survives there only by a
        // divisibility coincidence).  The fallback must size its grid to
        // the link's bandwidth (n ≳ 2·|l|·M(R)) and match a 65536-point
        // oracle.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [55.555_555_555_555_56, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-1.0, 0, 1, &array![0, 1], None);

        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                100,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let oracle = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(100, 65536),
            -10,
            10,
            &PeierlsFourierMethod::TimeGrid,
        );
        let bessel_cache = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(100, 512),
            -10,
            10,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        for (a, b) in oracle.blocks.iter().zip(bessel_cache.blocks.iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "65536-point oracle {a} vs adaptive fallback {b}"
            );
        }

        // Physical anchor: the block stores t·C_0 with t = -1, so its
        // n = 0 entry on the r = 50 link is -J_0(50).
        let i_r_link = find_R(&model.hamR, &array![0, 1]).unwrap();
        let c0 = bessel_cache.blocks[[bessel_cache.harmonic_index(0), i_r_link, 0, 1]];
        assert!(
            (c0 - Complex::new(-bessel_j(0, 50.0), 0.0)).norm() < 1e-10,
            "C_0 on the r = 50 link should be -J_0(50) (t = -1), got {c0}"
        );
    }

    #[test]
    fn harmonic_cache_bessel_ignores_n_time() {
        // The Bessel backend must produce bitwise-identical coefficients for
        // wildly different n_time, even when a fallback link (R > 8) forces
        // the per-link time-grid path.  Reintroducing any n_time dependence
        // on this path (an eager shared grid, or a trunc.n_time short-circuit
        // in the fallback) changes the fp results and fails this test.
        //
        // The requested harmonic range [-110, 110] (n_max = 55) exceeds the
        // fallback link's bandwidth half-grid (required/2 = 59), so a
        // bandwidth-only grid would alias bins near n = 110 — exactly the
        // regime where the removed shared-grid branch used to hide the
        // difference behind a large user-chosen n_time.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [10.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None); // d = (10, 1), R = 9 > 8

        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let coarse = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(55, 8),
            -110,
            110,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        let fine = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(55, 65536),
            -110,
            110,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        assert_eq!(
            coarse.blocks.dim(),
            fine.blocks.dim(),
            "harmonic ranges must match"
        );
        for (a, b) in coarse.blocks.iter().zip(fine.blocks.iter()) {
            assert_eq!(
                a, b,
                "Bessel cache must not depend on n_time (got {a} vs {b})"
            );
        }
    }

    #[test]
    fn harmonic_cache_bessel_fallback_resolves_requested_range() {
        // The per-link fallback DFT must also resolve the requested
        // harmonic range, not just the signal bandwidth: an n-point DFT
        // returns Σ_m C_{n+mn}, so bins with |n| >= n_req/2 fold genuine
        // low-order coefficients in.  For the l = 1, R = 9 link below the
        // bandwidth-only grid has n_req = 118 and corrupts bins
        // n ∈ [103, 110] at the J_{118-n}(9) level (up to ~0.3); sizing the
        // grid to the requested range [-110, 110] keeps every requested bin
        // alias-free, and the true C_n for n > 57 is exponentially small.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [10.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None); // d = (10, 1), R = 9 > 8

        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let cache = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(55, 512),
            -110,
            110,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        let i_r_link = find_R(&model.hamR, &array![0, 1]).unwrap();
        // n beyond the signal bandwidth (57 for R = 9, l = 1) must vanish:
        // |C_60| ~ J_60(9) ~ 1e-45, and the grid (n_req = 221) folds only
        // coefficients with |n ± n_req| >= 111 (min |110 − 221|), also
        // exponentially small.
        for n in 60..=110 {
            let coeff = cache.blocks[[cache.harmonic_index(n), i_r_link, 0, 1]];
            assert!(
                coeff.norm() < 1e-10,
                "C_{n} on the fallback link must be ~0 (beyond the signal \
                 bandwidth), got {coeff}"
            );
        }
    }

    #[test]
    fn floquet_effective_model_ignores_n_time() {
        // End-to-end: the public effective-model entry point must return
        // bitwise-identical models for wildly different n_time, with a
        // fallback link (R = 9 > 8) forcing the per-link time-grid path.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [10.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let coarse = model
            .floquet_effective_model(&drive, &FloquetTruncation::new(55, 8), None)
            .unwrap();
        let fine = model
            .floquet_effective_model(&drive, &FloquetTruncation::new(55, 65536), None)
            .unwrap();
        assert_eq!(
            coarse.hamR, fine.hamR,
            "effective-model support must not depend on n_time"
        );
        for (a, b) in coarse.ham.iter().zip(fine.ham.iter()) {
            assert_eq!(
                a, b,
                "floquet_effective_model must ignore n_time (got {a} vs {b})"
            );
        }
    }

    #[test]
    fn fallback_grid_size_clamps_and_saturates() {
        // Pin the exact sizing decision (n_req, clamp flag, saturation
        // flag) for the normal, clamp, saturation, and request-dominant
        // regimes.  A broken clamp or saturation detector fails here with
        // exact values instead of a finiteness smoke test.
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let d = array![10.0, 1.0]; // R = 9 > 8: fallback link

        // Normal: sized from the signal bandwidth (M(9) = 57).
        let s = fallback_grid_size(&drive, &d, -3, 3);
        assert_eq!(s.n_req, 118); // 2·57 + 4
        assert!(!s.clamped && !s.saturated);

        // Request-dominant: 2·max(|harmonic_min|,|harmonic_max|) + 1 exceeds the
        // bandwidth term.
        let s = fallback_grid_size(&drive, &d, -100, 100);
        assert_eq!(s.n_req, 201);
        assert!(!s.clamped && !s.saturated);

        // Clamp: |l|·M beyond FALLBACK_GRID_MAX.
        let big = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                10000,
                array![Complex::new(0.9, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let s = fallback_grid_size(&big, &d, -1, 1);
        assert_eq!(s.n_req, 1 << 20);
        assert!(s.clamped);
        assert!(!s.saturated);

        // Saturation: R = 4000 exceeds the 4096-order adaptive cutoff,
        // so the bandwidth estimate is a lower bound.
        let sat = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let d_sat = array![4000.0, 1.0]; // R = 4000 > 8
        let s = fallback_grid_size(&sat, &d_sat, -1, 1);
        assert_eq!(s.n_req, 1 << 20);
        assert!(!s.clamped);
        assert!(s.saturated);
    }

    #[test]
    fn harmonic_cache_bessel_fallback_saturation_matches_oracle() {
        // R = 8900 saturates the 4096-order adaptive cutoff; the fallback
        // must detect the truncated bandwidth estimate and use the maximum
        // grid so its coefficients match a 65536-point oracle (which
        // resolves the true band ≈ 9100).  A broken saturation detector
        // leaves the bandwidth-only 8196-point grid, whose Nyquist
        // (4098) aliases the true band and fails this test.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [8900.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(-0.5, 0, 1, &array![0, 1], None); // d = (8900, 1), R = 8900 > 8
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let oracle = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(2, 65536),
            -3,
            3,
            &PeierlsFourierMethod::TimeGrid,
        );
        let cache = model.floquet_harmonic_cache(
            &drive,
            &FloquetTruncation::new(2, 512),
            -3,
            3,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        for (a, b) in oracle.blocks.iter().zip(cache.blocks.iter()) {
            assert!(
                (a - b).norm() < 1e-9,
                "saturated fallback {b} vs 65536-point oracle {a}"
            );
        }
    }

    /// Build the `n = ±1` harmonic blocks of a spinless model and return
    /// them aligned with `model.hamR`, together with the cache itself.
    fn commutator_test_blocks<const DIM: usize>(
        model: &Model<false, DIM, NoRMatrix>,
        drive: &FloquetDrive,
    ) -> (
        FloquetHarmonicCache,
        Vec<Array2<Complex<f64>>>,
        Vec<Array2<Complex<f64>>>,
    ) {
        let trunc = FloquetTruncation::new(1, 512);
        let cache = model.floquet_harmonic_cache(
            drive,
            &trunc,
            -1,
            1,
            &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
        );
        let n_r = model.hamR.nrows();
        let n1 = cache.harmonic_index(1);
        let nm1 = cache.harmonic_index(-1);
        let a_blocks = (0..n_r)
            .map(|i_r| cache.blocks.slice(s![n1, i_r, .., ..]).to_owned())
            .collect();
        let b_blocks = (0..n_r)
            .map(|i_r| cache.blocks.slice(s![nm1, i_r, .., ..]).to_owned())
            .collect();
        (cache, a_blocks, b_blocks)
    }

    #[test]
    fn real_space_commutator_matches_k_space_commutator() {
        // The real-space convolution must equal the Fourier transform of
        // the k-space commutator [H^(1)(k), H^(-1)(k)] at every k: this
        // validates the two-convolution structure (the naive "P − P†"
        // single-convolution simplification is wrong) against the
        // existing, independently validated harmonic evaluator.
        //
        // The drive must be circularly polarized in 2D: for any single
        // linear mode H^(1)(k) is a scalar multiple of an anti-Hermitian
        // matrix (T_1(R) = t(R)·(−i)·J_1(r)·e^{iθ}·sgn(d)), whose
        // commutator [X, X†] vanishes identically — the first-order van
        // Vleck correction is exactly zero there, and the oracle would
        // be vacuous (it could not catch swapped operand order or
        // transposed-product bugs).
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.35, 0.2]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(-0.3, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);

        // Circular polarization: a = 0.3·(e_x + i·e_y).
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
            )],
        );
        let (cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
        let (comm_blocks, comm_r) =
            real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();

        let nsta = model.nsta();
        let mut oracle_scale = 0.0_f64;
        for k in [[0.0, 0.0], [0.123, 0.321], [0.5, 0.5], [0.877, 0.111]] {
            let kvec = array![k[0], k[1]];
            let a_k = model.floquet_cached_harmonic_onek(&kvec, 1, Gauge::Lattice, &cache);
            let b_k = model.floquet_cached_harmonic_onek(&kvec, -1, Gauge::Lattice, &cache);
            let oracle = a_k.dot(&b_k) - b_k.dot(&a_k);
            oracle_scale = oracle_scale.max(oracle.iter().fold(0.0, |m, c| m.max(c.norm())));
            let mut from_rs = Array2::<Complex<f64>>::zeros((nsta, nsta));
            for (i_r, row) in comm_r.outer_iter().enumerate() {
                let mut phase_arg = 0.0;
                for a in 0..2 {
                    phase_arg += row[a] as f64 * kvec[a];
                }
                let phase = Complex::new(0.0, TAU * phase_arg).exp();
                from_rs.scaled_add(phase, &comm_blocks[i_r]);
            }
            for (a, b) in from_rs.iter().zip(oracle.iter()) {
                assert!(
                    (a - b).norm() < 1e-12,
                    "k = {k:?}: real-space {a} vs k-space {b}"
                );
            }
        }
        assert!(
            oracle_scale > 1e-6,
            "oracle must be non-trivial (circular 2D drive); otherwise \
             the comparison is vacuous, got {oracle_scale}"
        );
    }

    #[test]
    fn real_space_commutator_with_different_supports_matches_k_space() {
        // Nested van Vleck terms feed a pair-support inner commutator into a
        // primitive-support outer commutator.  This oracle uses deliberately
        // different, non-Hermitian supports so an implementation that reuses
        // one lookup table for both operands or prematurely symmetrizes the
        // inner result fails visibly.
        let a_r = array![[-1_isize], [2]];
        let b_r = array![[-2_isize], [0], [1]];
        let a_blocks = vec![
            array![
                [Complex::new(0.2, 0.1), Complex::new(-0.3, 0.4)],
                [Complex::new(0.7, -0.2), Complex::new(-0.1, 0.5)]
            ],
            array![
                [Complex::new(-0.4, 0.3), Complex::new(0.6, 0.2)],
                [Complex::new(-0.2, -0.8), Complex::new(0.9, -0.1)]
            ],
        ];
        let b_blocks = vec![
            array![
                [Complex::new(0.5, -0.2), Complex::new(0.1, 0.7)],
                [Complex::new(-0.6, 0.3), Complex::new(0.2, 0.4)]
            ],
            array![
                [Complex::new(-0.1, 0.6), Complex::new(0.8, -0.5)],
                [Complex::new(0.3, 0.2), Complex::new(-0.7, 0.1)]
            ],
            array![
                [Complex::new(0.4, 0.4), Complex::new(-0.2, 0.3)],
                [Complex::new(0.5, -0.6), Complex::new(0.1, -0.2)]
            ],
        ];
        let (comm_blocks, comm_r) =
            real_space_commutator_with_supports(&a_blocks, &a_r, &b_blocks, &b_r).unwrap();

        for k in [0.0, 0.137, 0.5, 0.819] {
            let mut a_k = Array2::<Complex<f64>>::zeros((2, 2));
            let mut b_k = Array2::<Complex<f64>>::zeros((2, 2));
            let mut comm_k = Array2::<Complex<f64>>::zeros((2, 2));
            for (index, r) in a_r.outer_iter().enumerate() {
                let phase = Complex::new(0.0, TAU * k * r[0] as f64).exp();
                a_k.scaled_add(phase, &a_blocks[index]);
            }
            for (index, r) in b_r.outer_iter().enumerate() {
                let phase = Complex::new(0.0, TAU * k * r[0] as f64).exp();
                b_k.scaled_add(phase, &b_blocks[index]);
            }
            for (index, r) in comm_r.outer_iter().enumerate() {
                let phase = Complex::new(0.0, TAU * k * r[0] as f64).exp();
                comm_k.scaled_add(phase, &comm_blocks[index]);
            }
            let oracle = a_k.dot(&b_k) - b_k.dot(&a_k);
            for (actual, expected) in comm_k.iter().zip(oracle.iter()) {
                assert!(
                    (actual - expected).norm() < 2e-13,
                    "k={k}: real-space {actual} vs k-space {expected}"
                );
            }
        }
    }

    #[test]
    fn real_space_commutator_parallel_matches_serial_and_k_space() {
        // 12 * 13 pairs deliberately crosses PARALLEL_PAIR_THRESHOLD.  Run
        // the same convolution in isolated one- and four-thread pools so the
        // test covers both dispatch paths independently of the test runner's
        // global Rayon configuration.
        let a_r = Array2::from_shape_fn((12, 2), |(i, axis)| match axis {
            0 => i as isize - 6,
            _ => (i * i % 7) as isize - 3,
        });
        let b_r = Array2::from_shape_fn((13, 2), |(i, axis)| match axis {
            0 => 2 * i as isize - 12,
            _ => ((3 * i + 1) % 11) as isize - 5,
        });
        let a_blocks = (0..a_r.nrows())
            .map(|i| {
                let x = i as f64 + 1.0;
                array![
                    [
                        Complex::new(0.03 * x, -0.02 * x),
                        Complex::new(-0.01 * x, 0.04 * (x + 1.0))
                    ],
                    [
                        Complex::new(0.02 * (x + 2.0), 0.01 * x),
                        Complex::new(-0.025 * x, 0.015 * (x - 1.0))
                    ]
                ]
            })
            .collect::<Vec<_>>();
        let b_blocks = (0..b_r.nrows())
            .map(|i| {
                let x = i as f64 + 0.5;
                array![
                    [
                        Complex::new(-0.02 * x, 0.01 * (x + 1.0)),
                        Complex::new(0.035 * x, -0.015 * x)
                    ],
                    [
                        Complex::new(-0.04 * (x + 1.0), 0.02 * x),
                        Complex::new(0.01 * x, 0.03 * (x - 2.0))
                    ]
                ]
            })
            .collect::<Vec<_>>();

        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let serial = serial_pool.install(|| {
            real_space_commutator_with_supports(&a_blocks, &a_r, &b_blocks, &b_r).unwrap()
        });
        let parallel = parallel_pool.install(|| {
            real_space_commutator_with_supports(&a_blocks, &a_r, &b_blocks, &b_r).unwrap()
        });

        assert_eq!(parallel.1, serial.1);
        for (actual, expected) in parallel.0.iter().zip(serial.0.iter()) {
            for (actual, expected) in actual.iter().zip(expected.iter()) {
                assert!((actual - expected).norm() < 2e-14);
            }
        }

        for k in [[0.137, 0.291], [0.419, 0.073], [0.811, 0.557]] {
            let kvec = array![k[0], k[1]];
            let mut a_k = Array2::<Complex<f64>>::zeros((2, 2));
            let mut b_k = Array2::<Complex<f64>>::zeros((2, 2));
            let mut comm_k = Array2::<Complex<f64>>::zeros((2, 2));
            for (index, r) in a_r.outer_iter().enumerate() {
                a_k.scaled_add(bloch_phase::<2, _>(&r, &kvec), &a_blocks[index]);
            }
            for (index, r) in b_r.outer_iter().enumerate() {
                b_k.scaled_add(bloch_phase::<2, _>(&r, &kvec), &b_blocks[index]);
            }
            for (index, r) in parallel.1.outer_iter().enumerate() {
                comm_k.scaled_add(bloch_phase::<2, _>(&r, &kvec), &parallel.0[index]);
            }
            let oracle = a_k.dot(&b_k) - b_k.dot(&a_k);
            for (actual, expected) in comm_k.iter().zip(oracle.iter()) {
                assert!(
                    (actual - expected).norm() < 2e-12,
                    "k={k:?}: real-space {actual} vs k-space {expected}"
                );
            }
        }
    }

    #[test]
    fn real_space_commutator_rejects_triple_support_overflow() {
        // Two copies of half_max still fit, so this only overflows when that
        // pair support is combined with the third primitive support.
        let half_max = isize::MAX / 2;
        let primitive_r = array![[half_max]];
        let primitive_blocks = vec![array![[Complex::new(0.0, 0.0)]]];
        let (pair_blocks, pair_r) = real_space_commutator_with_supports(
            &primitive_blocks,
            &primitive_r,
            &primitive_blocks,
            &primitive_r,
        )
        .unwrap();
        assert_eq!(pair_r[[0, 0]], isize::MAX - 1);

        let error = real_space_commutator_with_supports(
            &primitive_blocks,
            &primitive_r,
            &pair_blocks,
            &pair_r,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("Minkowski sum overflow"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn exact_zero_fast_paths_preserve_underflow_sized_values() {
        // norm_sqr(1e-200) underflows to zero, but the value itself is not
        // zero and can yield a finite result when multiplied by 1e200.
        let support = array![[0_isize]];
        let tiny = 1e-200;
        let huge = 1e200;
        let a_blocks = vec![array![
            [Complex::new(0.0, 0.0), Complex::new(tiny, 0.0)],
            [Complex::new(tiny, 0.0), Complex::new(0.0, 0.0)]
        ]];
        let b_blocks = vec![array![
            [Complex::new(huge, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(-huge, 0.0)]
        ]];
        let (comm_blocks, _) =
            real_space_commutator_with_supports(&a_blocks, &support, &b_blocks, &support).unwrap();
        let expected = matrix_commutator(&a_blocks[0], &b_blocks[0]);
        for (actual, expected) in comm_blocks[0].iter().zip(expected.iter()) {
            assert!((actual - expected).norm() < 1e-14);
        }
        assert!(comm_blocks[0][[0, 1]].norm() > 1.0);

        let mut target = std::collections::BTreeMap::new();
        let scalar_source = vec![array![[Complex::new(tiny, 0.0)]]];
        accumulate_scaled_real_space_blocks(&mut target, &scalar_source, &support, huge).unwrap();
        assert!((target[&vec![0]][[0, 0]].re - 1.0).abs() < 1e-14);
    }

    #[test]
    fn real_space_commutator_vanishes_for_linear_polarization() {
        // For a single linear mode in 1D the first-order van Vleck
        // commutator vanishes exactly (H^(1)(k) is a scalar multiple of
        // an anti-Hermitian matrix).  The wrong "P − P†"
        // single-convolution implementation produces a nonzero answer
        // here, so this pins the double-convolution structure from the
        // other side.
        let lat = array![[1.0]];
        let orb = array![[0.0], [0.35]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1], None);
        model.set_hop(Complex::new(-0.3, 0.1), 0, 1, &array![1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![2], None);

        let drive =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
        let (_cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
        let (comm_blocks, _comm_r) =
            real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();
        for block in &comm_blocks {
            let max = block.iter().fold(0.0_f64, |m, c| m.max(c.norm()));
            assert!(
                max < 1e-15,
                "linear-polarization commutator must vanish exactly, got {max}"
            );
        }
    }

    #[test]
    fn real_space_commutator_support_and_hermiticity() {
        // hamR = {−2, −1, 1, 2} ⇒ the Minkowski sum is {−4..=4}; and the
        // symmetrized blocks must satisfy comm(R) = comm(−R)† exactly.
        let lat = array![[1.0]];
        let orb = array![[0.0], [0.35]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1], None);
        model.set_hop(-0.3, 0, 1, &array![1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![2], None);

        let drive =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
        let (_cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
        let (comm_blocks, comm_r) =
            real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();

        // Support: the Minkowski sum of {−2, −1, 1, 2} with itself.
        let expected: Vec<Vec<isize>> = (-4..=4).map(|r| vec![r]).collect();
        let got: Vec<Vec<isize>> = comm_r.outer_iter().map(|row| row.to_vec()).collect();
        assert_eq!(
            got, expected,
            "commutator support must be the Minkowski sum"
        );

        // Hermiticity pairing, exact after symmetrization.
        for i in 0..comm_r.nrows() {
            let j = find_R(&comm_r, &comm_r.row(i).mapv(|v| -v)).unwrap();
            let conj = hermitian_conjugate(&comm_blocks[j]);
            for (a, b) in comm_blocks[i].iter().zip(conj.iter()) {
                assert!(
                    (a - b).norm() < 1e-15,
                    "comm(R) != comm(−R)† at R = {:?}",
                    comm_r.row(i).to_vec()
                );
            }
        }
    }

    #[test]
    fn real_space_commutator_scalar_blocks_vanish() {
        // nsta = 1: matrix products commute, so comm(R) = 0 for every R —
        // a structural check on the (AB) − (BA) accumulation.
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1], None);

        let drive =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
        let (_cache, a_blocks, b_blocks) = commutator_test_blocks(&model, &drive);
        let (comm_blocks, _comm_r) =
            real_space_commutator(&a_blocks, &b_blocks, &model.hamR).unwrap();
        for block in &comm_blocks {
            assert!(
                block[[0, 0]].norm() < 1e-15,
                "scalar commutator must vanish, got {}",
                block[[0, 0]]
            );
        }
    }

    #[test]
    fn floquet_effective_model_bessel_matches_legacy_bands() {
        // Cross-validate the real-space Bessel path against the legacy
        // k-space path on the same support: the legacy path is given the
        // Bessel output's support as target_hamR (its default — the
        // original hamR — would truncate the longer-range commutator
        // terms) and a fine k-mesh / time grid, so both compute the same
        // H_eff.  The two-mode drive exercises the multi-mode Bessel
        // convolution end to end.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.35, 0.2]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(-0.3, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);

        // Circular l = 1 plus a second harmonic l = 2.
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)]),
                LightMode::new(2, array![Complex::new(0.05, -0.05), Complex::new(0.0, 0.0)]),
            ],
        );
        let trunc = FloquetTruncation::new(2, 4096);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(4);
        let bessel = model
            .floquet_effective_model(&drive, &trunc, Some(&options))
            .unwrap();

        // Legacy path on the same (automatically determined) support.
        let legacy = model
            .floquet_effective_model_legacy(
                &drive,
                &trunc,
                [64, 64],
                Some(&options.with_target_hamR(bessel.hamR.clone())),
            )
            .unwrap();

        for k in [[0.1, 0.2], [0.5, 0.5], [0.9, 0.7]] {
            let kvec = array![k[0], k[1]];
            let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
            let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
            for (a, b) in e_b.iter().zip(e_l.iter()) {
                assert!((a - b).abs() < 1e-8, "k = {k:?}: Bessel {a} vs legacy {b}");
            }
        }
    }

    /// Independent weak-field one-photon vertex from the exact straight-bond
    /// plane-wave integral.  This keeps the finite-q sinc form factor instead
    /// of starting from the derivative formula used by the implementation.
    fn independent_finite_q_peierls_vertex(
        model: &Model<false, 2, NoRMatrix>,
        k_reduced: &Array1<f64>,
        center_shift_cartesian: &Array1<f64>,
        transfer_cartesian: &Array1<f64>,
        amplitude: &Array1<Complex<f64>>,
    ) -> Array2<Complex<f64>> {
        let nsta = model.nsta();
        let norb = model.norb();
        let mut out = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for (i_r, r_vec) in model.hamR.outer_iter().enumerate() {
            for i in 0..nsta {
                for j in 0..nsta {
                    let hopping = model.ham[[i_r, i, j]];
                    if hopping.re == 0.0 && hopping.im == 0.0 {
                        continue;
                    }
                    let mut d_fractional = Array1::<f64>::zeros(2);
                    for axis in 0..2 {
                        d_fractional[axis] = r_vec[axis] as f64 + model.orb[[j % norb, axis]]
                            - model.orb[[i % norb, axis]];
                    }
                    let d_cartesian = d_fractional.dot(&model.lat);
                    let phase_angle = TAU * k_reduced.dot(&d_fractional)
                        + center_shift_cartesian.dot(&d_cartesian);
                    let phase = Complex::new(0.0, phase_angle).exp();
                    let projection = amplitude
                        .iter()
                        .zip(d_cartesian.iter())
                        .fold(Complex::new(0.0, 0.0), |sum, (a, d)| sum + *a * *d);
                    let half_q_dot_d = 0.5 * transfer_cartesian.dot(&d_cartesian);
                    let sinc = if half_q_dot_d.abs() < 1.0e-8 {
                        let x2 = half_q_dot_d * half_q_dot_d;
                        1.0 - x2 / 6.0 + x2 * x2 / 120.0
                    } else {
                        half_q_dot_d.sin() / half_q_dot_d
                    };
                    out[[i, j]] += Complex::new(0.0, -0.5) * projection * sinc * hopping * phase;
                }
            }
        }
        out
    }

    fn independent_q_linear_commutator_odd_part(
        model: &Model<false, 2, NoRMatrix>,
        k_reduced: &Array1<f64>,
        wavevector_cartesian: &Array1<f64>,
        amplitude: &Array1<Complex<f64>>,
        harmonic: isize,
        photon_energy_ev: f64,
    ) -> Array2<Complex<f64>> {
        let amplitude_conjugate = amplitude.mapv(|value| value.conj());
        let commutator_at = |q_sign: f64| {
            let q = wavevector_cartesian.mapv(|value| q_sign * value);
            let minus_q = q.mapv(|value| -value);
            let half_q = q.mapv(|value| 0.5 * value);
            let minus_half_q = half_q.mapv(|value| -value);
            let positive_minus =
                independent_finite_q_peierls_vertex(model, k_reduced, &minus_half_q, &q, amplitude);
            let negative_minus = independent_finite_q_peierls_vertex(
                model,
                k_reduced,
                &minus_half_q,
                &minus_q,
                &amplitude_conjugate,
            );
            let positive_plus =
                independent_finite_q_peierls_vertex(model, k_reduced, &half_q, &q, amplitude);
            let negative_plus = independent_finite_q_peierls_vertex(
                model,
                k_reduced,
                &half_q,
                &minus_q,
                &amplitude_conjugate,
            );
            (positive_minus.dot(&negative_minus) - negative_plus.dot(&positive_plus))
                * Complex::new(1.0 / ((harmonic as f64) * photon_energy_ev), 0.0)
        };
        (commutator_at(1.0) - commutator_at(-1.0)) * Complex::new(0.5, 0.0)
    }

    #[test]
    fn floquet_effective_q_model_matches_center_momentum_oracle() {
        let lat = array![[1.3, 0.2], [0.1, 0.9]];
        let orb = array![[0.07, 0.11], [0.31, 0.23]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(0.4, 0, 0, &array![0, 0], None);
        model.set_hop(-0.8, 0, 0, &array![1, 0], None);
        model.set_hop(Complex::new(0.35, -0.17), 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(-0.21, 0.09), 1, 1, &array![1, 1], None);

        let amplitude = array![Complex::new(0.24, 0.05), Complex::new(-0.08, 0.19)];
        let drive = FloquetDrive::with_modes(3.7, vec![LightMode::new(1, amplitude.clone())]);
        let trunc = FloquetTruncation::new(1, 128);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(1);
        let q = array![8.0e-4, -6.0e-4];
        let k = array![0.17, 0.29];

        let baseline = model
            .floquet_effective_model(&drive, &trunc, Some(&options))
            .unwrap();
        let finite_q = model
            .floquet_effective_q_model(&drive, &trunc, Some(&options), &q)
            .unwrap();
        let implemented = finite_q.gen_ham(&k, Gauge::Atom) - baseline.gen_ham(&k, Gauge::Atom);
        let expected = independent_q_linear_commutator_odd_part(
            &model,
            &k,
            &q,
            &amplitude,
            1,
            drive.omega0_ev,
        );
        let error = (&implemented - &expected)
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            error < 2.0e-11,
            "finite-q weak-field correction mismatch: error={error:e}\nimplemented={implemented:?}\nexpected={expected:?}"
        );
    }

    #[test]
    fn floquet_effective_q_model_is_coherent_and_preserves_size() {
        let lat = array![[1.0, 0.0], [0.0, 1.4]];
        let orb = array![[0.0, 0.0], [0.2, 0.3]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-0.9, 0, 0, &array![1, 0], None);
        model.set_hop(Complex::new(0.3, 0.2), 0, 1, &array![0, 1], None);

        let amplitude = array![Complex::new(0.2, 0.04), Complex::new(-0.03, 0.16)];
        let split_drive = FloquetDrive::with_modes(
            2.4,
            vec![
                LightMode::new(1, amplitude.mapv(|value| value * 0.3)),
                LightMode::new(1, amplitude.mapv(|value| value * 0.7)),
            ],
        );
        let joined_drive =
            FloquetDrive::with_modes(2.4, vec![LightMode::new(1, amplitude.clone())]);
        let trunc = FloquetTruncation::new(1, 64);
        let q = array![0.013, -0.007];
        let split = model
            .floquet_effective_q_model(&split_drive, &trunc, None, &q)
            .unwrap();
        let joined = model
            .floquet_effective_q_model(&joined_drive, &trunc, None, &q)
            .unwrap();

        assert_eq!(split.nsta(), model.nsta());
        assert_eq!(split.hamR, joined.hamR);
        for (left, right) in split.ham.iter().zip(joined.ham.iter()) {
            assert!((*left - *right).norm() < 1.0e-12);
        }
    }

    #[test]
    fn floquet_effective_q_model_zero_q_is_exact_uniform_path() {
        let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.0]]);
        let drive = circular_drive(0.35, 1.0, 4.0);
        let trunc = FloquetTruncation::new(1, 128);
        let options = FloquetEffectiveOptions::new().with_order(2);
        let uniform = model
            .floquet_effective_model(&drive, &trunc, Some(&options))
            .unwrap();
        let q_zero = model
            .floquet_effective_q_model(&drive, &trunc, Some(&options), &array![0.0, 0.0])
            .unwrap();
        assert_eq!(q_zero.hamR, uniform.hamR);
        assert_eq!(q_zero.ham, uniform.ham);
    }

    #[test]
    fn floquet_effective_q_model_rejects_invalid_q_and_nonpositive_harmonic() {
        let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.0]]);
        let trunc = FloquetTruncation::new(1, 64);
        let drive = circular_drive(0.2, 1.0, 3.0);
        assert!(
            model
                .floquet_effective_q_model(&drive, &trunc, None, &array![0.01])
                .is_err()
        );
        assert!(
            model
                .floquet_effective_q_model(&drive, &trunc, None, &array![f64::NAN, 0.0])
                .is_err()
        );

        let negative = FloquetDrive::with_modes(
            3.0,
            vec![LightMode::new(
                -1,
                array![Complex::new(0.2, 0.0), Complex::new(0.0, -0.2)],
            )],
        );
        let error = model
            .floquet_effective_q_model(&negative, &trunc, None, &array![0.01, 0.0])
            .unwrap_err();
        assert!(error.to_string().contains("positive temporal harmonics"));
    }

    #[test]
    fn floquet_effective_q_model_rejects_finite_arithmetic_overflow() {
        let overflow_drive = FloquetDrive::with_modes(
            2.0,
            vec![
                LightMode::new(1, array![Complex::new(0.75 * f64::MAX, 0.0)]),
                LightMode::new(1, array![Complex::new(0.75 * f64::MAX, 0.0)]),
            ],
        );
        let error = coherent_positive_harmonic_amplitudes::<1>(&overflow_drive, 1)
            .expect_err("coherent amplitude overflow must be rejected");
        assert!(error.to_string().contains("amplitude sum overflowed"));

        let lat = array![[2.0]];
        let orb = array![[0.0]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1], None);
        let drive =
            FloquetDrive::with_modes(2.0, vec![LightMode::new(1, array![Complex::new(0.1, 0.0)])]);
        let error = model
            .floquet_effective_q_model(
                &drive,
                &FloquetTruncation::new(1, 32),
                None,
                &array![f64::MAX],
            )
            .expect_err("q·d overflow must be rejected");
        assert!(error.to_string().contains("q·d overflowed"));

        let subnormal_scale = q_linear_frequency_scale(2, 1.0e308);
        assert!(
            subnormal_scale.is_finite() && subnormal_scale < 0.0,
            "representable inverse-frequency scale was lost: {subnormal_scale}"
        );
    }

    #[test]
    fn floquet_effective_q_model_distinct_harmonics_add_at_linear_q_order() {
        let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.2]]);
        let mode_one = LightMode::new(
            1,
            array![Complex::new(0.18, 0.03), Complex::new(-0.02, 0.12)],
        );
        let mode_two = LightMode::new(
            2,
            array![Complex::new(0.07, -0.01), Complex::new(0.04, 0.05)],
        );
        let combined_drive =
            FloquetDrive::with_modes(3.0, vec![mode_one.clone(), mode_two.clone()]);
        let drive_one = FloquetDrive::with_modes(3.0, vec![mode_one]);
        let drive_two = FloquetDrive::with_modes(3.0, vec![mode_two]);
        let trunc = FloquetTruncation::new(2, 64);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(2);
        let q = array![0.009, -0.004];
        let k = array![0.21, 0.37];

        let q_delta = |drive: &FloquetDrive| {
            let base = model
                .floquet_effective_model(drive, &trunc, Some(&options))
                .unwrap();
            let finite = model
                .floquet_effective_q_model(drive, &trunc, Some(&options), &q)
                .unwrap();
            finite.gen_ham(&k, Gauge::Atom) - base.gen_ham(&k, Gauge::Atom)
        };
        let combined = q_delta(&combined_drive);
        let expected = q_delta(&drive_one) + q_delta(&drive_two);
        let error = (&combined - &expected)
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(error < 1.0e-12, "distinct-harmonic q correction: {error:e}");

        let cutoff_one = FloquetEffectiveOptions::new().with_harmonic_max(1);
        let base_two = model
            .floquet_effective_model(&drive_two, &trunc, Some(&cutoff_one))
            .unwrap();
        let finite_two = model
            .floquet_effective_q_model(&drive_two, &trunc, Some(&cutoff_one), &q)
            .unwrap();
        assert_eq!(finite_two.hamR, base_two.hamR);
        assert_eq!(finite_two.ham, base_two.ham);
    }

    #[test]
    fn floquet_effective_q_model_spinful_matches_two_spinless_copies() {
        let lat = array![[1.1, 0.2], [0.0, 0.9]];
        let orb = array![[0.03, 0.07], [0.28, 0.19]];
        let mut spinless = Model::<false, 2>::tb_model(lat.clone(), orb.clone(), None).unwrap();
        let mut spinful = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
        for model_hop in [
            (Complex::new(-0.7, 0.0), 0, 0, array![1, 0]),
            (Complex::new(0.24, -0.13), 0, 1, array![0, 1]),
            (Complex::new(-0.18, 0.04), 1, 1, array![1, 1]),
        ] {
            let (value, i, j, r) = model_hop;
            spinless.set_hop(value, i, j, &r, None);
            spinful.set_hop(value, i, j, &r, None);
        }
        let drive = FloquetDrive::with_modes(
            2.8,
            vec![LightMode::new(
                1,
                array![Complex::new(0.16, 0.02), Complex::new(-0.03, 0.14)],
            )],
        );
        let trunc = FloquetTruncation::new(1, 64);
        let q = array![0.011, -0.006];
        let k = array![0.23, 0.31];
        let h0 = spinless
            .floquet_effective_q_model(&drive, &trunc, None, &q)
            .unwrap()
            .gen_ham(&k, Gauge::Atom);
        let hs = spinful
            .floquet_effective_q_model(&drive, &trunc, None, &q)
            .unwrap()
            .gen_ham(&k, Gauge::Atom);
        let norb = spinless.norb();
        for spin in 0..2 {
            for i in 0..norb {
                for j in 0..norb {
                    assert!((hs[[spin * norb + i, spin * norb + j]] - h0[[i, j]]).norm() < 1e-12);
                    assert!(hs[[spin * norb + i, (1 - spin) * norb + j]].norm() < 1e-12);
                }
            }
        }
    }

    #[test]
    #[allow(deprecated)]
    fn floquet_effective_q_model_preserves_legacy_enum_glob_call() {
        use FloquetQModelMode::*;

        let model = two_band_qwz(1.0, 0.7, 0.4, 0.8, [[1.0, 0.0], [0.0, 1.0]]);
        let drive = circular_drive(0.2, 1.0, 3.0);
        let trunc = FloquetTruncation::new(1, 32);
        let legacy = model
            .floquet_effective_q_model(&drive, &trunc, None, Coherent)
            .unwrap();
        let direct = model.floquet_effective_model(&drive, &trunc, None).unwrap();
        assert_eq!(legacy.hamR, direct.hamR);
        assert_eq!(legacy.ham, direct.ham);

        let named = model
            .floquet_effective_mode_resolved_model(
                &drive,
                &trunc,
                None,
                FloquetQModelMode::Coherent,
            )
            .unwrap();
        assert_eq!(named.hamR, direct.hamR);
        assert_eq!(named.ham, direct.ham);
    }

    #[test]
    fn floquet_mode_resolved_incoherent_weight_matches_single_mode() {
        // Mode-resolved incoherent recombination must return the exact weighted
        // mixture of one-mode results: [w1, w2 = 0] reproduces mode1 exactly.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.22, 0.14]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(0.9, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.4, -0.2), 1, 1, &array![1, 1], None);

        let mode1 = LightMode::new(1, array![Complex::new(0.28, 0.0), Complex::new(0.0, 0.21)]);
        let mode2 = LightMode::new(
            2,
            array![Complex::new(0.04, 0.02), Complex::new(0.03, -0.01)],
        );
        let drive = FloquetDrive::with_modes(1.0, vec![mode1.clone(), mode2]);
        let trunc = FloquetTruncation::new(2, 512);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(3);

        let mode1_only = model
            .floquet_effective_mode_resolved_model(
                &FloquetDrive::with_modes(1.0, vec![mode1.clone()]),
                &trunc,
                Some(&options),
                FloquetModeCombination::Coherent,
            )
            .unwrap();
        let recombined = model
            .floquet_effective_q_model(
                &drive,
                &trunc,
                Some(&options),
                FloquetModeCombination::ModeResolvedIncoherent(vec![1.0, 0.0]),
            )
            .unwrap();

        assert_eq!(mode1_only.hamR, recombined.hamR);
        for (a, b) in mode1_only.ham.iter().zip(recombined.ham.iter()) {
            let diff = a - b;
            assert!(
                diff.norm() < 1e-12,
                "incoherent mode recombination mismatch: {diff}"
            );
        }
    }

    #[test]
    fn floquet_mode_resolved_incoherent_weights_enforce_unit_sum() {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.22, 0.14]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(0.9, 0, 1, &array![0, 1], None);

        let mode1 = LightMode::new(1, array![Complex::new(0.28, 0.0), Complex::new(0.0, 0.21)]);
        let mode2 = LightMode::new(
            2,
            array![Complex::new(0.04, 0.02), Complex::new(0.03, -0.01)],
        );
        let drive = FloquetDrive::with_modes(1.0, vec![mode1, mode2]);
        let trunc = FloquetTruncation::new(2, 512);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(3);

        let err = model
            .floquet_effective_mode_resolved_model(
                &drive,
                &trunc,
                Some(&options),
                FloquetModeCombination::ModeResolvedIncoherent(vec![0.6, 0.6]),
            )
            .expect_err("mode weights should fail to exceed unit sum");

        assert!(
            err.to_string().contains("Σw <= 1")
                || err.to_string().contains("mode weights exceed unit total"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn floquet_mode_resolved_incoherent_no_static_matches_weighted_single_modes() {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.22, 0.14]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(0.9, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.4, -0.2), 1, 1, &array![1, 1], None);

        let mode1 = LightMode::new(1, array![Complex::new(0.28, 0.0), Complex::new(0.0, 0.21)]);
        let mode2 = LightMode::new(
            2,
            array![Complex::new(0.04, 0.02), Complex::new(0.03, -0.01)],
        );
        let drive = FloquetDrive::with_modes(1.0, vec![mode1.clone(), mode2.clone()]);
        let trunc = FloquetTruncation::new(2, 512);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(3);

        let mode1_only = model
            .floquet_effective_mode_resolved_model(
                &FloquetDrive::with_modes(1.0, vec![mode1]),
                &trunc,
                Some(&options),
                FloquetModeCombination::Coherent,
            )
            .unwrap();
        let mode2_only = model
            .floquet_effective_mode_resolved_model(
                &FloquetDrive::with_modes(1.0, vec![mode2]),
                &trunc,
                Some(&options),
                FloquetModeCombination::Coherent,
            )
            .unwrap();
        let mixed = model
            .floquet_effective_mode_resolved_model(
                &drive,
                &trunc,
                Some(&options),
                FloquetModeCombination::ModeResolvedIncoherentNoStatic(vec![0.3, 1.7]),
            )
            .unwrap();

        let k = array![0.13, 0.27];
        let h_mix = mixed.gen_ham(&k, Gauge::Lattice);
        let h_expected = mode1_only.gen_ham(&k, Gauge::Lattice) * Complex::new(0.3, 0.0)
            + mode2_only.gen_ham(&k, Gauge::Lattice) * Complex::new(1.7, 0.0);
        let diff = &h_mix - &h_expected;
        assert!(
            diff.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt() < 1e-10,
            "no-static weighted mode recombination mismatch: {diff:?}"
        );
    }

    #[test]
    fn floquet_mode_resolved_incoherent_no_static_zero_weights_is_zero_model() {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.22, 0.14]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(0.9, 0, 1, &array![0, 1], None);

        let drive = FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(1, array![Complex::new(0.28, 0.0), Complex::new(0.0, 0.21)]),
                LightMode::new(
                    2,
                    array![Complex::new(0.04, 0.02), Complex::new(0.03, -0.01)],
                ),
            ],
        );
        let trunc = FloquetTruncation::new(2, 512);

        let zero = model
            .floquet_effective_mode_resolved_model(
                &drive,
                &trunc,
                None,
                FloquetModeCombination::ModeResolvedIncoherentNoStatic(vec![0.0, 0.0]),
            )
            .unwrap();

        assert_eq!(zero.hamR, Array2::<isize>::zeros((1, 2)));
        assert_eq!(zero.ham.dim(), (1, model.nsta(), model.nsta()));
        assert!(
            zero.ham
                .iter()
                .all(|value| *value == Complex::new(0.0, 0.0))
        );
        assert!(
            zero.gen_ham(&array![0.13, 0.27], Gauge::Lattice)
                .iter()
                .all(|value| *value == Complex::new(0.0, 0.0))
        );
    }

    #[test]
    fn floquet_effective_model_order_two_matches_legacy_and_has_triple_support() {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.35, 0.2]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(-0.3, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);
        let drive = FloquetDrive::with_modes(
            1.7,
            vec![
                LightMode::new(1, array![Complex::new(0.28, 0.03), Complex::new(0.0, 0.24)]),
                LightMode::new(
                    2,
                    array![Complex::new(0.06, -0.04), Complex::new(0.02, 0.01)],
                ),
            ],
        );
        let trunc = FloquetTruncation::new(2, 4096);
        let options = FloquetEffectiveOptions::new()
            .with_order(2)
            .with_harmonic_max(2);
        let serial_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let serial = serial_pool.install(|| {
            model
                .floquet_effective_model(&drive, &trunc, Some(&options))
                .unwrap()
        });
        let real_space = parallel_pool.install(|| {
            model
                .floquet_effective_model(&drive, &trunc, Some(&options))
                .unwrap()
        });

        assert_eq!(real_space.hamR, serial.hamR);
        for (actual, expected) in real_space.ham.iter().zip(serial.ham.iter()) {
            assert!(
                (actual - expected).norm() < 5e-13,
                "parallel {actual} vs serial {expected}"
            );
        }

        // Through 1/ω² the deterministic support is the union of the
        // primitive, double-, and triple-Minkowski supports.
        let mut expected_support = std::collections::BTreeSet::<Vec<isize>>::new();
        for r1 in model.hamR.outer_iter() {
            expected_support.insert(r1.to_vec());
            for r2 in model.hamR.outer_iter() {
                expected_support.insert(r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect());
                for r3 in model.hamR.outer_iter() {
                    expected_support.insert(
                        r1.iter()
                            .zip(r2.iter())
                            .zip(r3.iter())
                            .map(|((a, b), c)| a + b + c)
                            .collect(),
                    );
                }
            }
        }
        let actual_support: Vec<Vec<isize>> = real_space
            .hamR
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        assert_eq!(actual_support, Vec::from_iter(expected_support));

        let legacy = model
            .floquet_effective_model_legacy(
                &drive,
                &trunc,
                [32, 32],
                Some(&options.clone().with_target_hamR(real_space.hamR.clone())),
            )
            .unwrap();
        for k in [[0.07, 0.19], [0.31, 0.43], [0.73, 0.61]] {
            let kvec = array![k[0], k[1]];
            let from_real_space = real_space.gen_ham(&kvec, Gauge::Lattice);
            let from_legacy = legacy.gen_ham(&kvec, Gauge::Lattice);
            for (actual, expected) in from_real_space.iter().zip(from_legacy.iter()) {
                assert!(
                    (actual - expected).norm() < 2e-9,
                    "k={k:?}: real-space {actual} vs legacy {expected}"
                );
            }
        }

        // The final signed sum, unlike its individual nested terms, is
        // Hermitian in real space.
        for i_r in 0..real_space.hamR.nrows() {
            let opposite = find_R(
                &real_space.hamR,
                &real_space.hamR.row(i_r).mapv(|value| -value),
            )
            .unwrap();
            let expected =
                hermitian_conjugate(&real_space.ham.index_axis(Axis(0), opposite).to_owned());
            for (actual, expected) in real_space
                .ham
                .index_axis(Axis(0), i_r)
                .iter()
                .zip(expected.iter())
            {
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn floquet_effective_order_two_improves_high_frequency_scaling() {
        // Independent oracle: diagonalize the enlarged Sambe Hamiltonian and
        // compare its central-sector quasienergies with the same-size van
        // Vleck models.  With fixed Fourier blocks, an order-1 truncation has
        // O(Ω^-2) spectral error while order 2 has O(Ω^-3) error.
        let model = two_band_qwz(0.9, 0.7, 0.35, 0.4, [[1.0, 0.0], [0.0, 1.0]]);
        let kvec = array![0.173, 0.287];
        let mode = LightMode::new(
            1,
            array![Complex::new(0.23, 0.04), Complex::new(-0.02, 0.19)],
        );
        let trunc = FloquetTruncation::new(8, 4096);
        let base_options = FloquetEffectiveOptions::new().with_harmonic_max(8);
        let mut first_order_errors = Vec::new();
        let mut second_order_errors = Vec::new();

        for omega in [20.0, 40.0, 80.0] {
            let drive = FloquetDrive::with_modes(omega, vec![mode.clone()]);
            let sambe = model
                .floquet_ham_onek(&kvec, &drive, &trunc, Gauge::Lattice)
                .unwrap();
            let exact_all = eigvalsh_v(&sambe, UPLO::Lower);
            let exact: Vec<f64> = exact_all
                .iter()
                .copied()
                .filter(|energy| energy.abs() < 0.5 * omega)
                .collect();
            assert_eq!(
                exact.len(),
                model.nsta(),
                "expected one central quasienergy per static state at Ω={omega}"
            );

            let first = model
                .floquet_effective_model(&drive, &trunc, Some(&base_options.clone().with_order(1)))
                .unwrap()
                .solve_band_onek(&kvec);
            let second = model
                .floquet_effective_model(&drive, &trunc, Some(&base_options.clone().with_order(2)))
                .unwrap()
                .solve_band_onek(&kvec);
            let first_error = first
                .iter()
                .zip(exact.iter())
                .map(|(approx, exact)| (approx - exact).abs())
                .fold(0.0_f64, f64::max);
            let second_error = second
                .iter()
                .zip(exact.iter())
                .map(|(approx, exact)| (approx - exact).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                second_error < first_error,
                "order 2 must improve the central quasienergies at Ω={omega}: \
                 order-1 error={first_error:e}, order-2 error={second_error:e}"
            );
            first_order_errors.push(first_error);
            second_order_errors.push(second_error);
        }

        for pair in first_order_errors.windows(2) {
            assert!(
                pair[0] / pair[1] > 3.5,
                "order-1 error should scale as Ω^-2, got {pair:?}"
            );
        }
        for pair in second_order_errors.windows(2) {
            assert!(
                pair[0] / pair[1] > 6.5,
                "order-2 error should scale as Ω^-3, got {pair:?}"
            );
        }
    }

    #[test]
    fn floquet_effective_model_bessel_order_zero() {
        // order = 0 keeps only the Peierls-dressed static model T_0(R);
        // the support must stay the original hamR and the bands must
        // match the legacy order-0 path.
        let lat = array![[1.0]];
        let orb = array![[0.0], [0.35]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1], None);
        model.set_hop(Complex::new(-0.3, 0.1), 0, 1, &array![1], None);

        let drive =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
        let trunc = FloquetTruncation::new(1, 4096);
        let bessel = model
            .floquet_effective_model(
                &drive,
                &trunc,
                Some(&FloquetEffectiveOptions::new().with_order(0)),
            )
            .unwrap();
        // order-0 support = input hamR.
        assert_eq!(bessel.hamR.nrows(), model.hamR.nrows());
        let legacy = model
            .floquet_effective_model_legacy(
                &drive,
                &trunc,
                [64],
                Some(&FloquetEffectiveOptions::new().with_order(0)),
            )
            .unwrap();
        for k in [0.0, 0.23, 0.5, 0.71] {
            let kvec = array![k];
            let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
            let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
            for (a, b) in e_b.iter().zip(e_l.iter()) {
                assert!((a - b).abs() < 1e-8, "k = {k}: Bessel {a} vs legacy {b}");
            }
        }
    }

    #[test]
    fn floquet_effective_model_bessel_support_and_hermiticity() {
        // First-order support = Minkowski sum of hamR with itself, in
        // lexicographic order, and the output blocks satisfy
        // T(R) = T(−R)† exactly.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.35, 0.2]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(-0.3, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
            )],
        );
        let bessel = model
            .floquet_effective_model(&drive, &FloquetTruncation::new(1, 512), None)
            .unwrap();

        // Expected support: the Minkowski sum of the input hamR with
        // itself, lexicographically ordered.
        let mut expected = std::collections::BTreeSet::<Vec<isize>>::new();
        for r1 in model.hamR.outer_iter() {
            for r2 in model.hamR.outer_iter() {
                expected.insert(r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect());
            }
        }
        let got: Vec<Vec<isize>> = bessel.hamR.outer_iter().map(|row| row.to_vec()).collect();
        assert_eq!(got, Vec::from_iter(expected), "support mismatch");

        // Exact Hermiticity pairing.
        for i in 0..bessel.hamR.nrows() {
            let j = find_R(&bessel.hamR, &bessel.hamR.row(i).mapv(|v| -v)).unwrap();
            let conj = hermitian_conjugate(&bessel.ham.index_axis(Axis(0), j).to_owned());
            for (a, b) in bessel.ham.index_axis(Axis(0), i).iter().zip(conj.iter()) {
                assert!((a - b).norm() < 1e-15, "T(R) != T(−R)† at row {i}");
            }
        }
    }

    #[test]
    fn floquet_effective_model_bessel_rejects_invalid_options() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1], None);
        let drive =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])]);
        let trunc = FloquetTruncation::new(1, 512);

        // order 2 is supported; higher orders are rejected.
        assert!(
            model
                .floquet_effective_model(
                    &drive,
                    &trunc,
                    Some(&FloquetEffectiveOptions::new().with_order(2))
                )
                .is_ok()
        );
        assert!(
            model
                .floquet_effective_model(
                    &drive,
                    &trunc,
                    Some(&FloquetEffectiveOptions::new().with_order(3))
                )
                .is_err()
        );
        // target_hamR is rejected on the real-space path.
        let target = array![[-1], [0], [1]];
        assert!(
            model
                .floquet_effective_model(
                    &drive,
                    &trunc,
                    Some(&FloquetEffectiveOptions::new().with_target_hamR(target))
                )
                .is_err()
        );
        // negative harmonic_max.
        assert!(
            model
                .floquet_effective_model(
                    &drive,
                    &trunc,
                    Some(&FloquetEffectiveOptions::new().with_harmonic_max(-1))
                )
                .is_err()
        );

        // An order-2 symmetric cache needs 4*harmonic_max+1 harmonics.  Reject an
        // unrepresentable inclusive range before ndarray arithmetic/allocation.
        let range_error = model
            .floquet_effective_model(
                &drive,
                &trunc,
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(isize::MAX / 2),
                ),
            )
            .unwrap_err();
        assert!(
            range_error
                .to_string()
                .contains("too large to index safely"),
            "unexpected error: {range_error}"
        );

        // A finite positive frequency can still have a non-representable
        // inverse-square factor.  Exact-zero scalar commutators remain valid.
        let tiny_frequency_drive = FloquetDrive::with_modes(
            1e-200,
            vec![LightMode::new(1, array![Complex::new(0.4, 0.2)])],
        );
        let scalar_result = model
            .floquet_effective_model(
                &tiny_frequency_drive,
                &trunc,
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(1),
                ),
            )
            .unwrap();
        assert!(
            scalar_result
                .ham
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );

        // A genuinely nonzero nested commutator cannot be represented with
        // the same scale and must return an error instead of NaN/Inf blocks.
        let matrix_model = two_band_qwz(0.9, 0.7, 0.35, 0.4, [[1.0, 0.0], [0.0, 1.0]]);
        let matrix_drive = FloquetDrive::with_modes(
            1e-200,
            vec![LightMode::new(
                1,
                array![Complex::new(0.23, 0.04), Complex::new(-0.02, 0.19)],
            )],
        );
        let scale_error = matrix_model
            .floquet_effective_model(
                &matrix_drive,
                &trunc,
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(1),
                ),
            )
            .unwrap_err();
        assert!(
            scale_error.to_string().contains("frequency scaling factor"),
            "unexpected error: {scale_error}"
        );
        model
            .floquet_effective_model_legacy(
                &tiny_frequency_drive,
                &trunc,
                [8],
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(1),
                ),
            )
            .unwrap();
        let legacy_scale_error = matrix_model
            .floquet_effective_model_legacy(
                &matrix_drive,
                &trunc,
                [4, 4],
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(1),
                ),
            )
            .unwrap_err();
        assert!(
            legacy_scale_error
                .to_string()
                .contains("frequency scaling factor"),
            "unexpected legacy error: {legacy_scale_error}"
        );

        // harmonic_max=0 never evaluates inverse-frequency corrections and remains
        // well-defined even at the same tiny frequency.
        let zero_cutoff = model
            .floquet_effective_model(
                &tiny_frequency_drive,
                &trunc,
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(0),
                ),
            )
            .unwrap();
        assert!(
            zero_cutoff
                .ham
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }

    #[test]
    fn floquet_effective_model_spinful_matches_legacy() {
        // Spinful 2D model: the real-space path must be agnostic to the
        // spin structure (blocks are nsta x nsta with nsta = 2·norb).
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.35, 0.2]];
        let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(-0.3, 0, 1, &array![0, 1], SpinDirection::X);
        model.set_hop(
            Complex::new(0.1, -0.2),
            1,
            1,
            &array![1, 1],
            SpinDirection::Z,
        );

        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
            )],
        );
        let trunc = FloquetTruncation::new(1, 4096);
        let bessel = model.floquet_effective_model(&drive, &trunc, None).unwrap();
        let legacy = model
            .floquet_effective_model_legacy(
                &drive,
                &trunc,
                [32, 32],
                Some(&FloquetEffectiveOptions::new().with_target_hamR(bessel.hamR.clone())),
            )
            .unwrap();
        for k in [[0.1, 0.2], [0.5, 0.5]] {
            let kvec = array![k[0], k[1]];
            let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
            let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
            for (a, b) in e_b.iter().zip(e_l.iter()) {
                assert!((a - b).abs() < 1e-8, "k = {k:?}: Bessel {a} vs legacy {b}");
            }
        }
    }

    #[test]
    fn floquet_effective_model_no_drive_matches_static_bands() {
        // Empty drive: the real-space path returns T_0 = static blocks
        // plus exact-zero commutator blocks on the Minkowski support —
        // the bands must equal the static model's.
        let model = chain_model();
        let drive = FloquetDrive::new(0.9);
        let trunc = FloquetTruncation::new(2, 64);
        let effective = model.floquet_effective_model(&drive, &trunc, None).unwrap();

        let k = arr1(&[0.271]);
        for gauge in [Gauge::Lattice, Gauge::Atom] {
            let from_effective = effective.gen_ham(&k, gauge);
            let from_static = model.gen_ham(&k, gauge);
            let mut max_diff = 0.0f64;
            for i in 0..from_effective.nrows() {
                for j in 0..from_effective.ncols() {
                    max_diff = max_diff.max((from_effective[[i, j]] - from_static[[i, j]]).norm());
                }
            }
            assert!(
                max_diff < 1e-12,
                "no-drive effective mismatch in {gauge:?}: {max_diff:e}"
            );
        }

        // Documented support contract: even for an empty drive the
        // Minkowski-blown support with exact-zero commutator blocks is
        // retained — chain_model's hamR = {-1, 0, 1} gives {-2..=2}.
        let got: Vec<Vec<isize>> = effective
            .hamR
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        let expected: Vec<Vec<isize>> = (-2..=2).map(|r| vec![r]).collect();
        assert_eq!(
            got, expected,
            "empty-drive support must be the Minkowski union"
        );

        // Empty harmonics take the exact-zero fast path: even when 1/W^2
        // overflows, order 2 must remain the static model rather than forming
        // inf*0 = NaN.  Its documented support is still the triple sum.
        let tiny_drive = FloquetDrive::new(1e-200);
        let order_two = model
            .floquet_effective_model(
                &tiny_drive,
                &trunc,
                Some(
                    &FloquetEffectiveOptions::new()
                        .with_order(2)
                        .with_harmonic_max(1),
                ),
            )
            .unwrap();
        assert!(
            order_two
                .ham
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
        let order_two_support: Vec<Vec<isize>> = order_two
            .hamR
            .outer_iter()
            .map(|row| row.to_vec())
            .collect();
        assert_eq!(
            order_two_support,
            (-3..=3).map(|r| vec![r]).collect::<Vec<_>>()
        );

        // Equivalent non-dynamic representations use the same n-independent
        // path: a zero mode, exactly cancelling ±frequency amplitudes, and a
        // purely static harmonic.  A huge harmonic_max must not create a huge cache
        // or repeat the same support construction harmonic_max^2 times.
        let non_dynamic_drives = [
            FloquetDrive::with_modes(
                1e-200,
                vec![LightMode::new(1, array![Complex::new(0.0, 0.0)])],
            ),
            FloquetDrive::with_modes(
                1e-200,
                vec![
                    LightMode::new(1, array![Complex::new(0.3, -0.2)]),
                    LightMode::new(-1, array![Complex::new(-0.3, -0.2)]),
                ],
            ),
            FloquetDrive::with_modes(
                1e-200,
                vec![LightMode::new(0, array![Complex::new(0.2, 7.0)])],
            ),
        ];
        let large_cutoff = FloquetEffectiveOptions::new()
            .with_order(2)
            .with_harmonic_max(1000);
        for equivalent_drive in non_dynamic_drives {
            let effective = model
                .floquet_effective_model(&equivalent_drive, &trunc, Some(&large_cutoff))
                .unwrap();
            assert!(
                effective
                    .ham
                    .iter()
                    .all(|value| value.re.is_finite() && value.im.is_finite())
            );
            assert_eq!(
                effective
                    .hamR
                    .outer_iter()
                    .map(|row| row.to_vec())
                    .collect::<Vec<_>>(),
                (-3..=3).map(|r| vec![r]).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn floquet_effective_model_bessel_perf_smoke() {
        // The real-space path must be far faster than the k-space
        // reference at comparable accuracy: the legacy cost scales with
        // the k-mesh (and its n_time DFT), while the Bessel path is
        // k-mesh-independent.  The plan's target is >100x; the assertion
        // uses a generous 10x lower bound with a 100 µs floor so the
        // test is robust on loaded machines.  The comparison doubles as
        // a cross-check.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.35, 0.2]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0, 0, 0, &array![1, 0], None);
        model.set_hop(-0.3, 0, 1, &array![0, 1], None);
        model.set_hop(Complex::new(0.1, -0.2), 1, 1, &array![1, 1], None);
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(0.3, 0.0), Complex::new(0.0, 0.3)],
            )],
        );
        let trunc = FloquetTruncation::new(1, 512);

        // Warm up once: the first call pays rayon thread-pool
        // initialization and cold caches, which would distort the ratio.
        let warmup = model.floquet_effective_model(&drive, &trunc, None).unwrap();
        let legacy_options = FloquetEffectiveOptions::new().with_target_hamR(warmup.hamR.clone());
        let _ = model
            .floquet_effective_model_legacy(&drive, &trunc, [128, 128], Some(&legacy_options))
            .unwrap();

        // Min-of-3 sampling on both sides: transient load spikes (e.g.
        // from the parallel test suite) would otherwise inflate the
        // short Bessel call and fail the assertion (observed at 4x vs
        // the 10x threshold under suite-parallel load).
        let start = std::time::Instant::now();
        let bessel = model.floquet_effective_model(&drive, &trunc, None).unwrap();
        let mut t_bessel = start.elapsed();
        for _ in 0..2 {
            let start = std::time::Instant::now();
            let _ = model.floquet_effective_model(&drive, &trunc, None).unwrap();
            t_bessel = t_bessel.min(start.elapsed());
        }

        let start = std::time::Instant::now();
        let legacy = model
            .floquet_effective_model_legacy(&drive, &trunc, [128, 128], Some(&legacy_options))
            .unwrap();
        let mut t_legacy = start.elapsed();
        for _ in 0..2 {
            let start = std::time::Instant::now();
            let _ = model
                .floquet_effective_model_legacy(&drive, &trunc, [128, 128], Some(&legacy_options))
                .unwrap();
            t_legacy = t_legacy.min(start.elapsed());
        }

        // Sanity: both paths agree (the smoke doubles as a cross-check).
        let kvec = array![0.37, 0.19];
        let e_b = eigvalsh_v(&bessel.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
        let e_l = eigvalsh_v(&legacy.gen_ham(&kvec, Gauge::Lattice), UPLO::Lower);
        for (a, b) in e_b.iter().zip(e_l.iter()) {
            assert!((a - b).abs() < 1e-8, "Bessel {a} vs legacy {b}");
        }

        let ratio = t_legacy.as_secs_f64() / t_bessel.as_secs_f64().max(1e-9);
        eprintln!("perf smoke: bessel {t_bessel:?} vs legacy {t_legacy:?} ({ratio:.0}x)");
        // 这是一个防回归冒烟测试，不是严格的 benchmark。不同 BLAS 后端和
        // release/debug 组合下常数差异很大；这里只要求仍然“显著快于”
        // k-space 路径，避免把 timing-sensitive 的阈值卡得太死。
        let speedup = t_legacy.as_secs_f64()
            / t_bessel
                .max(std::time::Duration::from_micros(100))
                .as_secs_f64();
        assert!(
            speedup > 5.0,
            "real-space path should be far faster than the k-space path ({speedup:.1}x)"
        );
    }

    /// Honeycomb graphene: two sublattices at the origin, nearest-neighbour
    /// hopping `j` along `e_0 = (0,1)a`, `e_1 = (−√3/2, −1/2)a`,
    /// `e_2 = (√3/2, −1/2)a` (with `a = 1`), using the triangular Bravais
    /// basis `a1 = (√3/2, 1/2)`, `a2 = (√3/2, −1/2)`.
    fn graphene_model(j: f64) -> Model<false, 2, NoRMatrix> {
        let lat = array![[3f64.sqrt() / 2.0, 0.5], [3f64.sqrt() / 2.0, -0.5],];
        let orb = array![[0.0, 0.0], [0.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        // e_0 = a1 − a2, e_1 = −a1, e_2 = a2.
        for r in [[1, -1], [-1, 0], [0, 1]] {
            model.set_hop(j, 0, 1, &array![r[0], r[1]], None);
        }
        model
    }

    /// Literature Fourier component for right-handed circular light
    /// (arXiv:1511.00755 conventions):
    /// `q_n(k) = J·J_n(α)·Σ_l e^{−ik·e_l} e^{i2πnl/3}` with the matrix
    /// `H_n = [[0, q_n], [n*_{−n}, 0]]`.  The literature `k` is the
    /// Cartesian wavevector; with fractional `k` the bond phase is
    /// `k_cart·e_l = 2π·k_frac·R_int` where `R_int` is the integer
    /// lattice vector of the bond (the non-orthonormal `lat` matrix must
    /// not be dropped).
    fn graphene_q_n_lit(n: isize, k: &[f64; 2], j: f64, alpha: f64) -> Complex<f64> {
        // Integer bond vectors: e_0 = a1 − a2, e_1 = −a1, e_2 = a2.
        let e_int = [[1, -1], [-1, 0], [0, 1]];
        let mut sum = Complex::new(0.0, 0.0);
        for (l, r) in e_int.iter().enumerate() {
            let phase = -TAU * (k[0] * r[0] as f64 + k[1] * r[1] as f64)
                + TAU * (n as f64) * (l as f64) / 3.0;
            sum += Complex::new(0.0, phase).exp();
        }
        j * bessel_j(n, alpha) * sum
    }

    /// Right-handed circular drive `a = α·(1, i)`, which reproduces the
    /// literature `a(t)·e_l = α·sin(ωt − 2πl/3)`.
    fn graphene_circular_drive(alpha: f64) -> FloquetDrive {
        FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                1,
                array![Complex::new(alpha, 0.0), Complex::new(0.0, alpha)],
            )],
        )
    }

    #[test]
    fn graphene_harmonics_match_literature_fourier_components() {
        // Benchmark A: H_n(k) elementwise against the literature Fourier
        // components.  Convention mapping: this library's gen_ham uses
        // e^{+i2πk·R}, the literature uses e^{−ik·R}, so our H_n(k)
        // equals the literature H_n(−k) — asserted elementwise to 1e-12.
        // This pins the Peierls phase sign, the Bessel phase δ_l
        // (including the (−1)^n structure from e^{±in(θ_l+π)}), and the
        // H_{−n} = H_n† pairing.
        let j = -1.0;
        let model = graphene_model(j);
        let trunc = FloquetTruncation::new(4, 512);
        for alpha in [0.3, 0.8] {
            let drive = graphene_circular_drive(alpha);
            let cache = model.floquet_harmonic_cache(
                &drive,
                &trunc,
                -4,
                4,
                &PeierlsFourierMethod::Bessel { cutoff_margin: 6 },
            );
            for k in [[0.13, 0.21], [0.5, 0.5], [0.87, 0.11]] {
                let kvec = array![k[0], k[1]];
                let k_neg = [-k[0], -k[1]];
                for n in -4..=4 {
                    let h_n = model.floquet_cached_harmonic_onek(&kvec, n, Gauge::Lattice, &cache);
                    let lit = array![
                        [
                            Complex::new(0.0, 0.0),
                            graphene_q_n_lit(n, &k_neg, j, alpha),
                        ],
                        [
                            graphene_q_n_lit(-n, &k_neg, j, alpha).conj(),
                            Complex::new(0.0, 0.0),
                        ],
                    ];
                    for (a, b) in h_n.iter().zip(lit.iter()) {
                        assert!(
                            (a - b).norm() < 1e-12,
                            "alpha = {alpha}, k = {k:?}, n = {n}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn graphene_dirac_gap_matches_exact_rotating_frame() {
        // Benchmark C: the Dirac-point quasienergy gap of the full Sambe
        // matrix converges to the exact rotating-frame value
        //
        //   Δ_exact = √((ħω)² + 4g²) − ħω,  g = ev_F A₀ = (3/2)|J|α,
        //
        // measured as the outermost folded branch separation (twice the
        // largest |folded eigenvalue|).  Small α = 0.2 keeps the lattice
        // corrections to the Dirac model are O(α⁴) in absolute units
        // (~1% relative at α = 0.2); the folded
        // spectrum also contains near-zero states from higher photon
        // sectors (the "minimal spacing" would pick those instead of the
        // physical branch gap).  The outer branches must simultaneously
        // agree with the first-order van Vleck mass d_z(K) = 3√3·|K_eff|
        // (Benchmark D's series), tying the Sambe construction to the
        // real-space effective model.
        let j = -1.0;
        let model = graphene_model(j);
        let alpha = 0.2;
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(
                1,
                array![Complex::new(alpha, 0.0), Complex::new(0.0, alpha)],
            )],
        );
        let w = drive.omega0_ev;
        let g = 1.5 * j.abs() * alpha;
        let delta_exact = (w * w + 4.0 * g * g).sqrt() - w;
        let k = array![1.0 / 3.0, 1.0 / 3.0];

        // First-order van Vleck mass at K (Benchmark D series).
        let n_cut = 8;
        let k_eff = -(2.0 * j * j / w)
            * (1..=n_cut)
                .map(|n| bessel_j(n, alpha).powi(2) / (n as f64) * (TAU * (n as f64) / 3.0).sin())
                .sum::<f64>();
        let d_z_k = -3.0 * 3f64.sqrt() * k_eff;

        let mut outer_previous = f64::INFINITY;
        for n_max in [4, 8, 12] {
            let trunc = FloquetTruncation::new(n_max, 512);
            let hf = model
                .floquet_ham_onek(&k, &drive, &trunc, Gauge::Lattice)
                .unwrap();
            let e = eigvalsh_v(&hf, UPLO::Lower);
            let outer = e
                .iter()
                .map(|x| ((*x + w / 2.0).rem_euclid(w) - w / 2.0).abs())
                .fold(0.0_f64, f64::max);
            // Outermost branch ≈ Δ_exact/2; the residual deviation from
            // Δ_exact/2 is physical (higher van Vleck orders + lattice
            // corrections O(α⁴)), not truncation — the branch is
            // converged already at n_max = 4.
            let tol = if n_max <= 4 { 2e-3 } else { 1e-3 };
            assert!(
                (outer - delta_exact / 2.0).abs() < tol,
                "n_max = {n_max}: outer branch {outer} vs exact {:.6}",
                delta_exact / 2.0
            );
            // The van Vleck mass (all-photon Bessel series) must agree
            // with the outer branch to O(1/W²) corrections.
            assert!(
                (outer - d_z_k.abs()).abs() < 5e-3,
                "n_max = {n_max}: outer branch {outer} vs van Vleck mass {d_z_k}"
            );
            assert!(
                outer <= outer_previous + 1e-12,
                "outer branch must converge downward: {outer} vs previous {outer_previous}"
            );
            outer_previous = outer;
        }
    }

    #[test]
    fn graphene_haldane_mass_matches_full_bessel_series() {
        // Benchmark D: the order-1 effective model's mass term must equal
        //
        //   d_z(k) = 2·K_eff·Σ_j sin(2πk·b_j),
        //   K_eff  = −(2J²/ħω)·Σ_{n=1..N} J_n²(α)/n · sin(2πn/3),
        //
        // with N the harmonic_max truncation (n = 3m terms vanish exactly).
        //
        // Convention note: the van Vleck commutator order here is
        // [H_n, H_{−n}]/(nħω), opposite to the literature's
        // [H_{−n}, H_n]/(nħω); combined with our k-convention being the
        // literature's mirror (H_n(k) = H_n^lit(−k)) the two sign flips
        // cancel for the TR-odd Haldane term — our d_z(k) equals the
        // literature's pointwise, and the Dirac-point mass reproduces
        // the exact rotating-frame leading order +g²/(ħω).
        let j = -1.0;
        let model = graphene_model(j);
        let alpha = 0.5;
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(
                1,
                array![Complex::new(alpha, 0.0), Complex::new(0.0, alpha)],
            )],
        );
        let w = drive.omega0_ev;
        let trunc = FloquetTruncation::new(4, 512);

        // Integer NNN vectors: b_1 = e_2 − e_1, b_2 = e_0 − e_2,
        // b_3 = e_1 − e_0.
        let b_int = [[1, 1], [1, -2], [-2, 1]];
        let k_eff_series = |n_cut: isize| -> f64 {
            -(2.0 * j * j / w)
                * (1..=n_cut)
                    .map(|n| {
                        bessel_j(n, alpha).powi(2) / (n as f64) * (TAU * (n as f64) / 3.0).sin()
                    })
                    .sum::<f64>()
        };
        let d_z_lit = |k: &[f64; 2], n_cut: isize| -> f64 {
            2.0 * k_eff_series(n_cut)
                * b_int
                    .iter()
                    .map(|b| (TAU * (k[0] * b[0] as f64 + k[1] * b[1] as f64)).sin())
                    .sum::<f64>()
        };

        let eff = model
            .floquet_effective_model(
                &drive,
                &trunc,
                Some(&FloquetEffectiveOptions::new().with_harmonic_max(8)),
            )
            .unwrap();
        let eff2 = model
            .floquet_effective_model(
                &drive,
                &trunc,
                Some(&FloquetEffectiveOptions::new().with_harmonic_max(2)),
            )
            .unwrap();
        for k in [[0.13, 0.07], [0.23, 0.11], [-0.07, 0.31]] {
            let kvec = array![k[0], k[1]];
            let h = eff.gen_ham(&kvec, Gauge::Lattice);
            let h2 = eff2.gen_ham(&kvec, Gauge::Lattice);
            // The mass term is the diagonal difference (T_0 is purely
            // off-diagonal for graphene).
            let d_z = ((h[[0, 0]] - h[[1, 1]]).re) / 2.0;
            let d_z_2 = ((h2[[0, 0]] - h2[[1, 1]]).re) / 2.0;
            assert!(
                (d_z - d_z_lit(&k, 8)).abs() < 1e-8,
                "k = {k:?}: d_z {d_z} vs series {expect}",
                expect = d_z_lit(&k, 8)
            );
            assert!(
                (d_z_2 - d_z_lit(&k, 2)).abs() < 1e-8,
                "k = {k:?}: harmonic_max = 2: d_z {d_z_2} vs series {}",
                d_z_lit(&k, 2)
            );
            // The neglected tail (n = 3m vanishes; n = 4, 5 are the
            // next contributors at ~1e-8 for α = 0.5).
            assert!(
                (d_z - d_z_2).abs() < 1e-5,
                "k = {k:?}: truncation tail {d_z} vs {d_z_2}"
            );
        }
    }

    #[test]
    fn graphene_order_zero_matches_renormalized_nn_hopping() {
        // Benchmark B: the order-0 effective model renormalizes the NN
        // hopping to J·J_0(α) (non-perturbative in α); its Hamiltonian
        // equals the literature H_0(−k) elementwise.
        let j = -1.0;
        let model = graphene_model(j);
        let trunc = FloquetTruncation::new(1, 512);
        let alpha = 0.6;
        let drive = graphene_circular_drive(alpha);
        let eff = model
            .floquet_effective_model(
                &drive,
                &trunc,
                Some(&FloquetEffectiveOptions::new().with_order(0)),
            )
            .unwrap();
        for k in [[0.13, 0.21], [0.5, 0.5], [0.87, 0.11]] {
            let kvec = array![k[0], k[1]];
            let h0 = eff.gen_ham(&kvec, Gauge::Lattice);
            let lit = array![
                [
                    Complex::new(0.0, 0.0),
                    graphene_q_n_lit(0, &[-k[0], -k[1]], j, alpha),
                ],
                [
                    graphene_q_n_lit(0, &[-k[0], -k[1]], j, alpha).conj(),
                    Complex::new(0.0, 0.0),
                ],
            ];
            for (a, b) in h0.iter().zip(lit.iter()) {
                assert!(
                    (a - b).norm() < 1e-12,
                    "alpha = {alpha}, k = {k:?}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn bessel_coeffs_reject_large_amplitudes_and_bad_ranges() {
        // R > 8 must error (the caller falls back to the time grid) instead
        // of silently violating the 1e-12 error budget via the 64 cap.
        let d = array![60.0];
        let drive =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(1.0, 0.0)])]);
        assert!(bessel_peierls_coeffs(&d, &drive, -4, 4, 6).is_err());

        // Harmonic ranges that exclude 0 must not panic; harmonic_min > harmonic_max errors.
        let zero_l =
            FloquetDrive::with_modes(1.0, vec![LightMode::new(0, array![Complex::new(0.1, 0.0)])]);
        let out_of_range = bessel_peierls_coeffs(&d, &zero_l, 5, 7, 6).unwrap();
        for n in 5..=7 {
            assert!((out_of_range[(n - 5) as usize]).norm() == 0.0);
        }
        assert!(bessel_peierls_coeffs(&d, &drive, 3, 2, 6).is_err());

        // cutoff_margin is bounded to [0, 48]: beyond that the starting
        // cutoff can reach the Bessel library's inf region and the
        // 64-order invariant breaks.
        assert!(bessel_peierls_coeffs(&d, &drive, -4, 4, 49).is_err());
        assert!(bessel_peierls_coeffs(&d, &drive, -4, 4, -1).is_err());
    }

    #[test]
    fn bessel_coeffs_large_harmonics_stay_within_error_budget() {
        // Regression for the two-pass window sizing: two modes l = ±400 at
        // r = 8 make e^{-i a(t)·d} = e^{-i·16·cos(400·Ω₀·t)}, whose closed
        // form is C_n = (-i)^n J_n(16) on bins divisible by 400 and zero
        // for |n| < 400.  A fixed window capped at 4096 drift units
        // clipped the fold support and got C_0 wrong by ~1e-3 for this
        // drive.
        let d = array![1.0, 0.0];
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![
                LightMode::new(400, array![Complex::new(8.0, 0.0), Complex::new(0.0, 0.0)]),
                LightMode::new(-400, array![Complex::new(8.0, 0.0), Complex::new(0.0, 0.0)]),
            ],
        );
        let coeffs = bessel_peierls_coeffs(&d, &drive, -10, 10, 0).unwrap();
        for n in -10..=10 {
            // Graf's addition theorem: C_0 = Σ_m (-1)^m J_m(8)² = J_0(16).
            let expected = if n == 0 { bessel_j(0, 16.0) } else { 0.0 };
            let got = coeffs[(n + 10) as usize];
            assert!(
                (got - Complex::new(expected, 0.0)).norm() < 1e-12,
                "n = {n}: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn bessel_coeffs_window_edge_overflows_are_skipped() {
        // The fold evaluates n + l·m for every (n, m) pair in the working
        // window; pairs whose source would leave the isize range must be
        // skipped, not panic (they contribute nothing — the source is
        // outside the window).  With l = 400, r = 8 and margin 48 the
        // drift is 400·56 = 22400, so a request near isize::MAX pushes
        // the window's top edge past isize::MAX during the fold.
        // Regression for the previously unchecked n + shift addition.
        let d = array![1.0, 0.0];
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(
                400,
                array![Complex::new(8.0, 0.0), Complex::new(0.0, 0.0)],
            )],
        );
        let n = isize::MAX - 23_999;
        let coeffs = bessel_peierls_coeffs(&d, &drive, n, n, 48).unwrap();
        // The requested bin is far outside the mode's spectral support
        // (±22400), so C_n = 0.
        assert!(coeffs[0].norm() == 0.0);
    }

    fn chain_model() -> Model<false, 1, NoRMatrix> {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize]), None);
        model
    }

    fn metadata_model() -> Model<false, 1, NoRMatrix> {
        let lat = array![[1.0]];
        let orb = array![[0.0], [0.0], [0.35]];
        let atoms = vec![
            Atom::with_orbitals(
                arr1(&[0.0]),
                AtomType::C,
                [OrbitalId::new(0), OrbitalId::new(1)],
            ),
            Atom::with_orbitals(arr1(&[0.35]), AtomType::O, [OrbitalId::new(2)]),
        ];
        let mut model = Model::<false, 1>::tb_model(lat, orb, Some(atoms)).unwrap();
        model.orb_projection = vec![OrbProj::s, OrbProj::px, OrbProj::py];
        model.set_hop(-0.8_f64, 0, 2, &arr1(&[1isize]), None);
        model
    }

    fn assert_same_atom_metadata(expected: &Atom, actual: &Atom) {
        assert_eq!(actual.position(), expected.position());
        assert_eq!(actual.norb(), expected.norb());
        assert_eq!(actual.atom_type(), expected.atom_type());
    }

    #[test]
    fn floquet_no_drive_static_replicas() {
        let model = chain_model();
        let k = arr1(&[0.17]);
        let drive = FloquetDrive::new(0.7);
        let trunc = FloquetTruncation::new(1, 64);

        let bands = model
            .floquet_band_onek(&k, &drive, &trunc, Gauge::Atom)
            .unwrap();
        let e0 = model.solve_band_onek(&k)[0];
        let mut expected = vec![e0 - 0.7, e0, e0 + 0.7];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (a, b) in bands.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-12, "got {a}, expected {b}");
        }
    }

    #[test]
    fn floquet_hamiltonian_is_hermitian() {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0]), None);
        model.set_hop(-0.7_f64, 0, 0, &arr1(&[0isize, 1]), None);

        let drive = FloquetDrive::with_modes(
            0.9,
            vec![LightMode::new(
                1,
                arr1(&[Complex::new(0.11, 0.0), Complex::new(0.0, 0.07)]),
            )],
        );
        let trunc = FloquetTruncation::new(2, 512);
        let k = arr1(&[0.13, 0.29]);
        let hf = model
            .floquet_ham_onek(&k, &drive, &trunc, Gauge::Atom)
            .unwrap();

        let mut max_diff = 0.0f64;
        for i in 0..hf.nrows() {
            for j in 0..hf.ncols() {
                max_diff = max_diff.max((hf[[i, j]] - hf[[j, i]].conj()).norm());
            }
        }
        assert!(max_diff < 1e-11, "max hermiticity error = {max_diff:e}");
    }

    #[test]
    fn floquet_model_matches_onek_construction() {
        let lat = array![[1.0, 0.0], [0.2, 1.1]];
        let orb = array![[0.0, 0.0], [0.31, 0.17]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(0.2_f64, 0, 1, &arr1(&[0isize, 0]), None);
        model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0]), None);
        model.set_hop(-0.6_f64, 1, 1, &arr1(&[0isize, 1]), None);

        let drive = FloquetDrive::with_modes(
            0.8,
            vec![LightMode::new(
                1,
                arr1(&[Complex::new(0.13, 0.0), Complex::new(0.0, 0.09)]),
            )],
        );
        let trunc = FloquetTruncation::new(1, 256);
        let k = arr1(&[0.23, 0.31]);
        let floquet_model = model.floquet_model(&drive, &trunc).unwrap();

        assert_eq!(floquet_model.nsta(), model.nsta() * trunc.n_sector());
        assert_eq!(floquet_model.hamR, model.hamR);

        for gauge in [Gauge::Lattice, Gauge::Atom] {
            let from_model = floquet_model.gen_ham(&k, gauge);
            let from_onek = model.floquet_ham_onek(&k, &drive, &trunc, gauge).unwrap();
            let mut max_diff = 0.0f64;
            for i in 0..from_model.nrows() {
                for j in 0..from_model.ncols() {
                    max_diff = max_diff.max((from_model[[i, j]] - from_onek[[i, j]]).norm());
                }
            }
            assert!(
                max_diff < 1e-12,
                "floquet_model mismatch in {gauge:?}: {max_diff:e}"
            );
        }
    }

    #[test]
    fn floquet_model_preserves_spinful_layout() {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.27, 0.19]];
        let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(0.3_f64, 0, 0, &arr1(&[0isize, 0]), crate::SpinDirection::Z);
        model.add_hop(0.2_f64, 0, 0, &arr1(&[0isize, 0]), crate::SpinDirection::X);
        model.set_hop(-0.9_f64, 0, 1, &arr1(&[1isize, 0]), None);
        model.set_hop(-0.4_f64, 1, 1, &arr1(&[0isize, 1]), None);

        let drive = FloquetDrive::with_modes(
            0.6,
            vec![LightMode::new(
                1,
                arr1(&[Complex::new(0.07, 0.0), Complex::new(0.0, 0.05)]),
            )],
        );
        let trunc = FloquetTruncation::new(1, 256);
        let k = arr1(&[0.17, 0.29]);
        let floquet_model = model.floquet_model(&drive, &trunc).unwrap();

        assert_eq!(floquet_model.norb(), model.norb() * trunc.n_sector());
        assert_eq!(floquet_model.nsta(), model.nsta() * trunc.n_sector());

        for gauge in [Gauge::Lattice, Gauge::Atom] {
            let from_model = floquet_model.gen_ham(&k, gauge);
            let from_onek = model.floquet_ham_onek(&k, &drive, &trunc, gauge).unwrap();
            let mut max_diff = 0.0f64;
            for i in 0..from_model.nrows() {
                for j in 0..from_model.ncols() {
                    max_diff = max_diff.max((from_model[[i, j]] - from_onek[[i, j]]).norm());
                }
            }
            assert!(
                max_diff < 1e-12,
                "spinful floquet_model mismatch in {gauge:?}: {max_diff:e}"
            );
        }
    }

    #[test]
    fn floquet_models_preserve_atom_metadata() {
        let model = metadata_model();
        let drive = FloquetDrive::new(1.2);
        let trunc = FloquetTruncation::new(1, 32);

        let sambe = model.floquet_model(&drive, &trunc).unwrap();
        assert_eq!(sambe.natom(), model.natom() * trunc.n_sector());
        for sector in 0..trunc.n_sector() {
            for i_atom in 0..model.natom() {
                assert_same_atom_metadata(
                    &model.atoms[i_atom],
                    &sambe.atoms[sector * model.natom() + i_atom],
                );
            }
        }
        let expected_projection: Vec<OrbProj> = (0..trunc.n_sector())
            .flat_map(|_| model.orb_projection.iter().copied())
            .collect();
        assert_eq!(sambe.orb_projection, expected_projection);

        let effective = model.floquet_effective_model(&drive, &trunc, None).unwrap();
        assert_eq!(effective.natom(), model.natom());
        for i_atom in 0..model.natom() {
            assert_same_atom_metadata(&model.atoms[i_atom], &effective.atoms[i_atom]);
        }
        assert_eq!(effective.orb_projection, model.orb_projection);
    }

    #[test]
    fn floquet_effective_rejects_non_hermitian_target_range() {
        let model = chain_model();
        let drive = FloquetDrive::new(1.0);
        let trunc = FloquetTruncation::new(1, 32);
        let options = FloquetEffectiveOptions::new().with_target_hamR(array![[0isize], [1isize]]);

        let err = model
            .floquet_effective_model_legacy(&drive, &trunc, [8], Some(&options))
            .unwrap_err();
        match err {
            TbError::MissingHermitianConjugateHopping { r } => {
                assert_eq!(r, arr1(&[1isize]));
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn floquet_effective_rejects_duplicate_target_vectors() {
        let model = chain_model();
        let drive = FloquetDrive::new(1.0);
        let trunc = FloquetTruncation::new(1, 32);
        let options = FloquetEffectiveOptions::new().with_target_hamR(array![[0isize], [0isize]]);

        let err = model
            .floquet_effective_model_legacy(&drive, &trunc, [8], Some(&options))
            .unwrap_err();
        match err {
            TbError::Other(message) => {
                assert!(message.contains("duplicate vector"), "{message}");
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn floquet_effective_order0_matches_h0() {
        let model = chain_model();
        let drive = FloquetDrive::with_modes(
            1.1,
            vec![LightMode::new(1, arr1(&[Complex::new(0.23, 0.0)]))],
        );
        let trunc = FloquetTruncation::new(2, 512);
        let options = FloquetEffectiveOptions::new().with_order(0);
        let effective = model
            .floquet_effective_model_legacy(&drive, &trunc, [32], Some(&options))
            .unwrap();

        assert_eq!(effective.nsta(), model.nsta());
        assert_eq!(effective.hamR, model.hamR);

        let k = arr1(&[0.173]);
        let from_model = effective.gen_ham(&k, Gauge::Lattice);
        let harmonic_cache =
            model.floquet_harmonic_cache(&drive, &trunc, 0, 0, &PeierlsFourierMethod::TimeGrid);
        let h0 = model.floquet_cached_harmonic_onek(&k, 0, Gauge::Lattice, &harmonic_cache);
        let mut max_diff = 0.0f64;
        for i in 0..from_model.nrows() {
            for j in 0..from_model.ncols() {
                max_diff = max_diff.max((from_model[[i, j]] - h0[[i, j]]).norm());
            }
        }
        assert!(max_diff < 1e-12, "order-0 effective mismatch: {max_diff:e}");
    }

    #[test]
    fn floquet_effective_no_drive_matches_static_model() {
        let model = chain_model();
        let drive = FloquetDrive::new(0.9);
        let trunc = FloquetTruncation::new(2, 64);
        let effective = model
            .floquet_effective_model_legacy(&drive, &trunc, [32], None)
            .unwrap();

        assert_eq!(effective.nsta(), model.nsta());
        assert_eq!(effective.hamR, model.hamR);

        let k = arr1(&[0.271]);
        for gauge in [Gauge::Lattice, Gauge::Atom] {
            let from_effective = effective.gen_ham(&k, gauge);
            let from_static = model.gen_ham(&k, gauge);
            let mut max_diff = 0.0f64;
            for i in 0..from_effective.nrows() {
                for j in 0..from_effective.ncols() {
                    max_diff = max_diff.max((from_effective[[i, j]] - from_static[[i, j]]).norm());
                }
            }
            assert!(
                max_diff < 1e-12,
                "no-drive effective mismatch in {gauge:?}: {max_diff:e}"
            );
        }
    }

    #[test]
    fn floquet_weak_drive_matches_first_order_peierls() {
        let model = chain_model();
        let amp = 1e-5;
        let drive = FloquetDrive::with_modes(
            1.0,
            vec![LightMode::new(1, arr1(&[Complex::new(amp, 0.0)]))],
        );
        let trunc = FloquetTruncation::new(1, 512);
        let k = arr1(&[0.25]);
        let hf = model
            .floquet_ham_onek(&k, &drive, &trunc, Gauge::Lattice)
            .unwrap();

        let nsta = model.nsta();
        let sector = |n: isize| -> usize { (n + trunc.n_max) as usize };
        let h_q1 = hf[[sector(0) * nsta, sector(-1) * nsta]];
        let expected = -amp * (TAU * k[0]).sin();
        assert!(
            (h_q1.re - expected).abs() < 1e-9,
            "got {}, expected {}",
            h_q1.re,
            expected
        );
        assert!(h_q1.im.abs() < 1e-9, "imag part = {}", h_q1.im);
    }

    #[test]
    fn floquet_incident_basis_public_api_example() {
        let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let orb = array![[0.0, 0.0, 0.0]];
        let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();
        model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0, 0]), None);
        model.set_hop(-0.8_f64, 0, 0, &arr1(&[0isize, 1, 0]), None);
        model.set_hop(-0.6_f64, 0, 0, &arr1(&[0isize, 0, 1]), None);

        let incident = IncidentBasis::from_direction(&arr1(&[0.0, 0.0, 1.0])).unwrap();
        let circular = incident.polarization([
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(0.0, 1.0 / 2.0_f64.sqrt()),
        ]);
        let drive =
            FloquetDrive::with_modes(0.8, vec![LightMode::new(1, circular.mapv(|z| 0.12 * z))]);
        let trunc = FloquetTruncation::new(1, 128);
        let k = arr1(&[0.2, 0.1, 0.0]);

        let hf = model
            .floquet_ham_onek(&k, &drive, &trunc, Gauge::Lattice)
            .unwrap();
        assert_eq!(hf.dim(), (3, 3));

        let mut max_diff = 0.0f64;
        for i in 0..hf.nrows() {
            for j in 0..hf.ncols() {
                max_diff = max_diff.max((hf[[i, j]] - hf[[j, i]].conj()).norm());
            }
        }
        assert!(max_diff < 1e-11, "max hermiticity error = {max_diff:e}");

        let qe = model
            .floquet_quasienergy_onek(&k, &drive, &trunc, Gauge::Lattice)
            .unwrap();
        assert_eq!(qe.len(), 3);
        for &x in qe.iter() {
            assert!(
                x >= -0.5 * drive.omega0_ev - 1e-12 && x < 0.5 * drive.omega0_ev + 1e-12,
                "quasienergy {x} is outside the first Floquet zone"
            );
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // General two-band analytical benchmarks (square & rectangular lattices).
    //
    // These verify the first-order van Vleck effective model against analytic
    // results for H_0(k) = ε(k) σ_0 + d(k)·σ under three drives:
    //   * circular   a(t) = κ (cos Ωt, η sin Ωt)          (η = ±1 helicity),
    //   * elliptical a(t) = (A_x cos Ωt, A_y sin Ωt),
    //   * an exotic two-harmonic drive (cos Ωt, sin(2Ωt+α)) whose harmonics are
    //     each linearly polarized, so its first-order commutator is O(κ³).
    //
    // Two independent analytic predictions are checked (fractional-k units,
    // Ω = ħω in eV, κ = |e|A₀/ħ):
    //   Level I  (exact in field): d_eff = d_0 + (2i/Ω) Σ_{n>0} (d_n × d_{−n})/n
    //   Level II (weak field):     δd_CPL = (A_x A_y / 4π²Ω)(∂_x d × ∂_y d),
    //                              δd_A²  = (κ² / 16π²)(∂_x² + ∂_y²) d.

    /// Two-band QWZ-type model on a rectangular lattice `lat` (rows = lattice
    /// vectors, both orbitals at the origin):
    ///   d(k) = ( t_x cos 2πk_x, t_y sin 2πk_y, m − 2t_z (cos 2πk_x + cos 2πk_y) ),
    ///   ε(k) = 0.
    fn two_band_qwz(
        tx: f64,
        ty: f64,
        tz: f64,
        m: f64,
        lat: [[f64; 2]; 2],
    ) -> Model<false, 2, NoRMatrix> {
        let lat = array![lat[0], lat[1]];
        let orb = array![[0.0, 0.0], [0.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_hop(m, 0, 0, &array![0, 0], None);
        model.set_hop(-m, 1, 1, &array![0, 0], None);
        for r in [[1, 0], [0, 1]] {
            model.set_hop(-tz, 0, 0, &array![r[0], r[1]], None);
            model.set_hop(tz, 1, 1, &array![r[0], r[1]], None);
        }
        model.set_hop(tx / 2.0, 0, 1, &array![1, 0], None);
        model.set_hop(tx / 2.0, 0, 1, &array![-1, 0], None);
        model.set_hop(-ty / 2.0, 0, 1, &array![0, 1], None);
        model.set_hop(ty / 2.0, 0, 1, &array![0, -1], None);
        model
    }

    /// Analytic d(k) of `two_band_qwz` (fractional k, independent of `lat`).
    fn qwz_d(k: &[f64; 2], tx: f64, ty: f64, tz: f64, m: f64) -> [f64; 3] {
        let (kx, ky) = (TAU * k[0], TAU * k[1]);
        [
            tx * kx.cos(),
            ty * ky.sin(),
            m - 2.0 * tz * (kx.cos() + ky.cos()),
        ]
    }

    /// Decompose a Hermitian 2×2 matrix H = ε σ_0 + d·σ into (ε, [d_x,d_y,d_z]).
    fn decompose_two_band(h: &Array2<Complex<f64>>) -> (f64, [f64; 3]) {
        let eps = (h[[0, 0]] + h[[1, 1]]).re / 2.0;
        let dz = (h[[0, 0]] - h[[1, 1]]).re / 2.0;
        let dx = h[[0, 1]].re;
        let dy = -h[[0, 1]].im;
        (eps, [dx, dy, dz])
    }

    /// Independent (non-Bessel, non-convolution) reference for H^(n)(k), the
    /// n-th Fourier block of the Peierls-dressed Hamiltonian, by direct
    /// trapezoidal integration of H(k,t) = Σ_R t(R) e^{i2πk·R} e^{−i a(t)·d_R}
    /// over one period, with a(t) reconstructed from the drive modes.  Shares
    /// no code with the Bessel / convolution / commutator machinery under test.
    fn independent_dressed_harmonic(
        model: &Model<false, 2, NoRMatrix>,
        k: &[f64; 2],
        drive: &FloquetDrive,
        n: isize,
        n_time: usize,
    ) -> Array2<Complex<f64>> {
        let nsta = model.nsta();
        let norb = model.norb();
        let mut h_n = Array2::<Complex<f64>>::zeros((nsta, nsta));
        for it in 0..n_time {
            let theta = TAU * (it as f64) / (n_time as f64);
            let mut a = [0.0f64; 2];
            for mode in &drive.modes {
                let phase = Complex::new(0.0, -(mode.harmonic as f64) * theta).exp();
                for (comp, ai) in mode.a_complex.iter().enumerate() {
                    a[comp] += (ai * phase).re;
                }
            }
            let mut h = Array2::<Complex<f64>>::zeros((nsta, nsta));
            for (i_r, r_row) in model.hamR.outer_iter().enumerate() {
                let r = [r_row[0], r_row[1]];
                let bloch =
                    Complex::new(0.0, TAU * (r[0] as f64 * k[0] + r[1] as f64 * k[1])).exp();
                for i in 0..nsta {
                    for j in 0..nsta {
                        let t = model.ham[[i_r, i, j]];
                        if t.norm_sqr() == 0.0 {
                            continue;
                        }
                        let mut d = [0.0f64; 2];
                        for c in 0..2 {
                            let mut acc = 0.0;
                            for b in 0..2 {
                                let frac = r[b] as f64 + model.orb[[j % norb, b]]
                                    - model.orb[[i % norb, b]];
                                acc += frac * model.lat[[b, c]];
                            }
                            d[c] = acc;
                        }
                        let peierls = Complex::new(0.0, -(a[0] * d[0] + a[1] * d[1])).exp();
                        h[[i, j]] += t * bloch * peierls;
                    }
                }
            }
            // C_n = (1/T)∫ e^{+inΩt} (⋯) dt — matches the code's convention.
            let fourier = Complex::new(0.0, (n as f64) * theta).exp();
            h_n.scaled_add(fourier, &h);
        }
        h_n.mapv(|x| x / (n_time as f64))
    }

    /// Independent first-order van Vleck H_eff from the integrated harmonics.
    fn independent_heff(
        model: &Model<false, 2, NoRMatrix>,
        k: &[f64; 2],
        drive: &FloquetDrive,
        harmonic_max: isize,
        n_time: usize,
    ) -> Array2<Complex<f64>> {
        let mut h_eff = independent_dressed_harmonic(model, k, drive, 0, n_time);
        for n in 1..=harmonic_max {
            let hp = independent_dressed_harmonic(model, k, drive, n, n_time);
            let hm = independent_dressed_harmonic(model, k, drive, -n, n_time);
            let comm = hp.dot(&hm) - hm.dot(&hp);
            let scale = Complex::new(1.0 / ((n as f64) * drive.omega0_ev), 0.0);
            h_eff.scaled_add(scale, &comm);
        }
        h_eff
    }

    /// Code's first-order H_eff as a 2×2 matrix at fractional k.
    fn code_heff(
        model: &Model<false, 2, NoRMatrix>,
        k: &[f64; 2],
        drive: &FloquetDrive,
        n_max: isize,
        harmonic_max: isize,
    ) -> Array2<Complex<f64>> {
        let trunc = FloquetTruncation::new(n_max, 512);
        let options = FloquetEffectiveOptions::new().with_harmonic_max(harmonic_max);
        let eff = model
            .floquet_effective_model(drive, &trunc, Some(&options))
            .unwrap();
        eff.gen_ham(&array![k[0], k[1]], Gauge::Lattice)
    }

    /// Code's order-0 (Peierls-dressed static) H_eff at fractional k.
    fn code_heff_order0(
        model: &Model<false, 2, NoRMatrix>,
        k: &[f64; 2],
        drive: &FloquetDrive,
        n_max: isize,
    ) -> Array2<Complex<f64>> {
        let trunc = FloquetTruncation::new(n_max, 512);
        let options = FloquetEffectiveOptions::new().with_order(0);
        let eff = model
            .floquet_effective_model(drive, &trunc, Some(&options))
            .unwrap();
        eff.gen_ham(&array![k[0], k[1]], Gauge::Lattice)
    }

    fn circular_drive(kappa: f64, eta: f64, omega: f64) -> FloquetDrive {
        FloquetDrive::with_modes(
            omega,
            vec![LightMode::new(
                1,
                array![Complex::new(kappa, 0.0), Complex::new(0.0, eta * kappa)],
            )],
        )
    }

    fn elliptical_drive(ax: f64, ay: f64, omega: f64) -> FloquetDrive {
        FloquetDrive::with_modes(
            omega,
            vec![LightMode::new(
                1,
                array![Complex::new(ax, 0.0), Complex::new(0.0, ay)],
            )],
        )
    }

    /// a(t) = κ (cos Ωt, sin(2Ωt+α)): mode l=1 along x, l=2 along y with
    /// a_y = κ(sin α + i cos α) (⇒ Re[…e^{−i2Ωt}] = κ sin(2Ωt+α)).
    fn exotic_drive(kappa: f64, alpha: f64, omega: f64) -> FloquetDrive {
        FloquetDrive::with_modes(
            omega,
            vec![
                LightMode::new(1, array![Complex::new(kappa, 0.0), Complex::new(0.0, 0.0)]),
                LightMode::new(
                    2,
                    array![
                        Complex::new(0.0, 0.0),
                        Complex::new(kappa * alpha.sin(), kappa * alpha.cos()),
                    ],
                ),
            ],
        )
    }

    #[test]
    fn two_band_level1_matches_independent_integration() {
        // Level I: for a general two-band model the first-order van Vleck
        // effective model must equal H^(0) + Σ_{n≥1} [H^(n),H^(−n)]/(nΩ) with
        // H^(n) the dressed harmonics — checked against an independent
        // time-integration of the dressed Hamiltonian, for circular, elliptical
        // and exotic drives on both square and rectangular lattices.
        let (tx, ty, tz, m) = (1.0, 0.7, 0.5, 0.8);
        let omega = 8.0;
        let n_time = 4096;
        let harmonic_max = 4;
        let ks = [[0.13, 0.27], [0.44, 0.61]];
        let lattices = [
            ("square", [[1.0, 0.0], [0.0, 1.0]]),
            ("rectangular", [[1.0, 0.0], [0.0, 1.6]]),
        ];
        for (lname, lat) in lattices {
            let model = two_band_qwz(tx, ty, tz, m, lat);
            let drives: Vec<(&str, FloquetDrive)> = vec![
                ("circular", circular_drive(0.5, 1.0, omega)),
                ("elliptical", elliptical_drive(0.5, 0.3, omega)),
                ("exotic", exotic_drive(0.4, 0.6, omega)),
            ];
            for (dname, drive) in drives {
                for k in ks {
                    let code = code_heff(&model, &k, &drive, 2, harmonic_max);
                    let indep = independent_heff(&model, &k, &drive, harmonic_max, n_time);
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!(
                                (code[[i, j]] - indep[[i, j]]).norm() < 1e-8,
                                "[{lname}/{dname}] k={k:?}: code {} vs independent {}",
                                code[[i, j]],
                                indep[[i, j]]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn two_band_weak_field_matches_cross_product() {
        // Level II: in the weak-field limit the helicity-odd and -even parts of
        // δd obey (fractional k, lat = I)
        //   δd_CPL = η (κ²/4π²Ω) (∂_x d × ∂_y d),
        //   δd_A²  =   (κ²/16π²) (∂_x² + ∂_y²) d.
        // They are isolated from the exact (all-κ) code result by a two-point
        // Richardson extrapolation in κ, removing the O(κ⁴) truncation error.
        let (tx, ty, tz, m) = (1.0, 0.7, 0.5, 0.8);
        let omega = 8.0;
        let k = [0.17, 0.31];
        let model = two_band_qwz(tx, ty, tz, m, [[1.0, 0.0], [0.0, 1.0]]);

        let (kx, ky) = (TAU * k[0], TAU * k[1]);
        let dx_d = [-TAU * tx * kx.sin(), 0.0, 2.0 * TAU * tz * kx.sin()];
        let dy_d = [0.0, TAU * ty * ky.cos(), 2.0 * TAU * tz * ky.sin()];
        let cross = [
            dx_d[1] * dy_d[2] - dx_d[2] * dy_d[1],
            dx_d[2] * dy_d[0] - dx_d[0] * dy_d[2],
            dx_d[0] * dy_d[1] - dx_d[1] * dy_d[0],
        ];
        let lap = [
            -TAU * TAU * tx * kx.cos(),
            -TAU * TAU * ty * ky.sin(),
            2.0 * tz * TAU * TAU * (kx.cos() + ky.cos()),
        ];
        let d_static = qwz_d(&k, tx, ty, tz, m);

        // d_eff(η) at order 1 (harmonic_max = 1 isolates the O(κ²) cross product).
        let d_eff = |kappa: f64, eta: f64| -> [f64; 3] {
            let drive = circular_drive(kappa, eta, omega);
            let h = code_heff(&model, &k, &drive, 1, 1);
            decompose_two_band(&h).1
        };

        let cpl = |kappa: f64| -> [f64; 3] {
            let dp = d_eff(kappa, 1.0);
            let dm = d_eff(kappa, -1.0);
            [
                (dp[0] - dm[0]) / 2.0,
                (dp[1] - dm[1]) / 2.0,
                (dp[2] - dm[2]) / 2.0,
            ]
        };
        let a2 = |kappa: f64| -> [f64; 3] {
            let dp = d_eff(kappa, 1.0);
            let dm = d_eff(kappa, -1.0);
            [
                (dp[0] + dm[0]) / 2.0 - d_static[0],
                (dp[1] + dm[1]) / 2.0 - d_static[1],
                (dp[2] + dm[2]) / 2.0 - d_static[2],
            ]
        };

        // f(κ) = C κ² + O(κ⁴) ⇒ C = (16 f(κ/2) − f(κ)) / (3 κ²).
        let richardson = |f1: [f64; 3], f2: [f64; 3], k1: f64| -> [f64; 3] {
            [
                (16.0 * f2[0] - f1[0]) / (3.0 * k1 * k1),
                (16.0 * f2[1] - f1[1]) / (3.0 * k1 * k1),
                (16.0 * f2[2] - f1[2]) / (3.0 * k1 * k1),
            ]
        };

        let kappa1 = 0.1;
        let cpl_coeff = richardson(cpl(kappa1), cpl(kappa1 / 2.0), kappa1);
        let a2_coeff = richardson(a2(kappa1), a2(kappa1 / 2.0), kappa1);

        // Predicted coefficients (η = +1): cross/(4π²Ω) = cross/(TAU²Ω),
        // lap/(16π²) = lap/(4·TAU²).
        let cpl_pred = [
            cross[0] / (TAU * TAU * omega),
            cross[1] / (TAU * TAU * omega),
            cross[2] / (TAU * TAU * omega),
        ];
        let a2_pred = [
            lap[0] / (4.0 * TAU * TAU),
            lap[1] / (4.0 * TAU * TAU),
            lap[2] / (4.0 * TAU * TAU),
        ];

        for comp in 0..3 {
            assert!(
                (cpl_coeff[comp] - cpl_pred[comp]).abs() < 1e-4,
                "CPL coefficient[{comp}]: {:.6} vs analytic {:.6}",
                cpl_coeff[comp],
                cpl_pred[comp]
            );
            assert!(
                (a2_coeff[comp] - a2_pred[comp]).abs() < 1e-4,
                "A² coefficient[{comp}]: {:.6} vs analytic {:.6}",
                a2_coeff[comp],
                a2_pred[comp]
            );
        }

        // Elliptical generalization: δd_CPL = (A_x A_y / 4π²Ω) (∂_x d × ∂_y d).
        // Isolate the helicity-odd part by A_y → −A_y; the O(κ⁴) truncation
        // error is ~ (A δ)² relative ≈ 3e-3, i.e. ~1e-6 absolute — below the
        // 1e-5 tolerance.
        let d_eff_ell = |ax: f64, ay: f64| -> [f64; 3] {
            let drive = elliptical_drive(ax, ay, omega);
            let h = code_heff(&model, &k, &drive, 1, 1);
            decompose_two_band(&h).1
        };
        let (ax, ay) = (0.08, 0.05);
        let dp = d_eff_ell(ax, ay);
        let dm = d_eff_ell(ax, -ay);
        let cpl_ell = [
            (dp[0] - dm[0]) / 2.0,
            (dp[1] - dm[1]) / 2.0,
            (dp[2] - dm[2]) / 2.0,
        ];
        for comp in 0..3 {
            let pred = ax * ay * cross[comp] / (TAU * TAU * omega);
            assert!(
                (cpl_ell[comp] - pred).abs() < 1e-5,
                "elliptical CPL[{comp}]: {:.6} vs analytic {:.6}",
                cpl_ell[comp],
                pred
            );
        }
    }

    #[test]
    fn two_band_exotic_drive_first_order_commutator_is_cubic() {
        // For a(t) = κ(cos Ωt, sin(2Ωt+α)) each harmonic is linearly polarized,
        // so the O(κ²) first-order van Vleck commutator vanishes identically;
        // the leading correction is O(κ³) (mode 1's cos² feeds n = 2 and mixes
        // with mode 2's linear n = 2).  Verify the commutator part of the code's
        // d scales as κ³ (ratio 8 for κ→κ/2), in contrast to circular (κ², ratio 4).
        let (tx, ty, tz, m) = (1.0, 0.7, 0.5, 0.8);
        let omega = 8.0;
        let k = [0.21, 0.37];
        let model = two_band_qwz(tx, ty, tz, m, [[1.0, 0.0], [0.0, 1.0]]);

        let commutator_norm = |_kappa: f64, drive: FloquetDrive| -> f64 {
            let h1 = code_heff(&model, &k, &drive, 2, 4);
            let h0 = code_heff_order0(&model, &k, &drive, 2);
            let d1 = decompose_two_band(&h1).1;
            let d0 = decompose_two_band(&h0).1;
            ((d1[0] - d0[0]).powi(2) + (d1[1] - d0[1]).powi(2) + (d1[2] - d0[2]).powi(2)).sqrt()
        };

        let exo1 = commutator_norm(0.3, exotic_drive(0.3, 0.6, omega));
        let exo2 = commutator_norm(0.15, exotic_drive(0.15, 0.6, omega));
        let ratio_exo = exo1 / exo2;
        assert!(
            exo1 > 1e-8,
            "exotic-drive commutator must be non-vanishing at O(κ³), got {exo1:e}"
        );
        assert!(
            (ratio_exo - 8.0).abs() < 1.0,
            "exotic-drive commutator should scale as κ³ (ratio ≈ 8), got {ratio_exo:.2}"
        );

        let circ1 = commutator_norm(0.3, circular_drive(0.3, 1.0, omega));
        let circ2 = commutator_norm(0.15, circular_drive(0.15, 1.0, omega));
        let ratio_circ = circ1 / circ2;
        assert!(
            (ratio_circ - 4.0).abs() < 0.5,
            "circular-drive commutator should scale as κ² (ratio ≈ 4), got {ratio_circ:.2}"
        );
    }
}
