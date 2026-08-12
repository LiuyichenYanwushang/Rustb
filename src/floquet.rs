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
//! $$ C_q(\mathbf d) =
//! \frac{1}{T}\int_0^T dt\,
//! e^{iq\Omega_0 t}
//! \exp\left[-i\,\mathbf a(t)\cdot\mathbf d\right]. $$
//!
//! The implementation evaluates `C_q` by uniform time sampling over one period.
//! This is deliberately more general than a Bessel-function formula: it handles
//! arbitrary complex polarization and arbitrary commensurate harmonic mixing.
//!
//! The reciprocal-space Fourier block is
//!
//! $$ H^{(q)}_{ij}(\mathbf k) =
//! \sum_{\mathbf R}
//! t_{ij}(\mathbf R)\,
//! C_q(\mathbf d_{ij\mathbf R})\,
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
//! $$ H_{\mathrm{eff}}(\mathbf k) =
//! H^{(0)}(\mathbf k)
//! +
//! \sum_{q=1}^{q_{\max}}
//! \frac{[H^{(q)}(\mathbf k), H^{(-q)}(\mathbf k)]}{q\Omega_0}
//! +
//! O(\Omega_0^{-2}). $$
//!
//! The Fourier blocks `H^{(q)}(k)` are defined in the [Peierls phase
//! section](#peierls-phase-and-fourier-blocks) above.  Each commutator term
//! `[H^(q), H^(-q)]` captures a virtual photon-exchange process where the
//! system absorbs a photon of energy `q Omega_0` and immediately re-emits it,
//! staying in the same photon sector but acquiring an effective hopping
//! correction of order `1/Omega_0`.
//!
//! Use [`Model::floquet_effective_model`] for this path.  It computes
//! `H_eff(k)` on a uniform k-mesh and inverse Fourier transforms back to
//! real space, returning a [`Model`]`<SPIN, DIM, NoRMatrix>` with the same
//! number of bands as the input model.
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
//! | [`FloquetEffectiveOptions`] | Optional order, q cutoff, and real-space truncation |
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
/// `n_time` controls the discrete Fourier transform used to evaluate Peierls
/// coefficients `C_q(d)`.  Increase it when the drive amplitude or the maximum
/// harmonic is large.
#[derive(Clone, Copy, Debug)]
pub struct FloquetTruncation {
    /// Photon cutoff `N`.
    pub n_max: isize,
    /// Number of time samples in one drive period.
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

/// Optional controls for building a same-size high-frequency Floquet effective model.
///
/// The `k_mesh` itself is passed directly to
/// [`Model::floquet_effective_model`].  These options only control the
/// high-frequency expansion order, harmonic cutoff, and target real-space
/// hopping range.  The inverse Fourier transform uses
///
/// $$ t_{\mathrm{eff}}(\mathbf R) =
/// \frac{1}{N_k}\sum_{\mathbf k}
/// H_{\mathrm{eff}}(\mathbf k)
/// e^{-i2\pi\mathbf k\cdot\mathbf R}. $$
///
/// If `target_hamR` is `None`, the original model's `hamR` is used.  This keeps
/// the returned model on the same real-space hopping range as the input model.
/// Provide a larger `target_hamR` when the commutator terms are expected to
/// generate longer-range effective hoppings.  Every vector must occur exactly
/// once, and the set must be closed under `R -> -R`, so the inverse-transformed
/// model can satisfy `H(-R) = H(R)^\dagger`.
#[derive(Clone, Debug)]
pub struct FloquetEffectiveOptions {
    /// van Vleck order.  Currently supported: `0` and `1`.
    pub order: usize,
    /// Harmonic cutoff for commutator terms.  Defaults to `2 * trunc.n_max`.
    pub q_max: Option<isize>,
    /// Optional target real-space hopping vectors for inverse Fourier transform.
    pub target_hamR: Option<Array2<isize>>,
}

impl Default for FloquetEffectiveOptions {
    fn default() -> Self {
        Self {
            order: 1,
            q_max: None,
            target_hamR: None,
        }
    }
}

impl FloquetEffectiveOptions {
    /// Construct first-order options using `q_max = 2 * trunc.n_max` and the
    /// original model's `hamR`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the van Vleck order.  Currently `0` and `1` are supported.
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set the harmonic cutoff used in first-order commutator terms.
    pub fn with_q_max(mut self, q_max: isize) -> Self {
        self.q_max = Some(q_max);
        self
    }

    /// Set the real-space hopping vectors used by the inverse Fourier transform.
    pub fn with_target_hamR(mut self, target_hamR: Array2<isize>) -> Self {
        self.target_hamR = Some(target_hamR);
        self
    }
}

/// Precomputed `t_ij(R) * C_q(d)` for all harmonics `q ∈ [q_min, q_max]`.
///
/// `blocks` has shape `(q_count, n_r, nsta, nsta)` where `q_count = q_max - q_min + 1`.
/// Index `[iq, i_r, i, j]` stores the `q = q_min + iq` Fourier component of hopping
/// from orbital `j` in cell `R = hamR[i_r]` to orbital `i` at the origin.
/// This is independent of `k` and reusable across the entire k-mesh.
struct FloquetHarmonicCache {
    q_min: isize,
    q_max: isize,
    blocks: Array4<Complex<f64>>,
}

impl FloquetHarmonicCache {
    #[inline]
    fn q_index(&self, q: isize) -> usize {
        debug_assert!(
            q >= self.q_min && q <= self.q_max,
            "Floquet harmonic q={q} is outside cached range [{}, {}]",
            self.q_min,
            self.q_max
        );
        (q - self.q_min) as usize
    }
}

/// Precomputed time-grid data for the discrete Fourier integration of
/// Peierls coefficients `C_q(d)`.
///
/// `link_field[it, a]` stores the real part of the total dimensionless
/// vector potential `a_a(t_it)` for each time step and spatial direction.
/// `fourier[iq, it]` stores `exp(i * q * theta)` for each harmonic and time step.
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
        q_min: isize,
        q_max: isize,
        dim: usize,
    ) -> Self {
        let n_time = trunc.n_time;
        let q_count = (q_max - q_min + 1) as usize;
        let inv_n_time = 1.0 / (n_time as f64);
        let mut link_field = Array2::<f64>::zeros((n_time, dim));
        let mut fourier = Array2::<Complex<f64>>::zeros((q_count, n_time));

        for it in 0..n_time {
            let theta = TAU * (it as f64) * inv_n_time;
            for mode in &drive.modes {
                let harmonic_phase = Complex::new(0.0, -(mode.harmonic as f64) * theta).exp();
                for a in 0..dim {
                    link_field[[it, a]] += (mode.a_complex[a] * harmonic_phase).re;
                }
            }
            for (iq, q) in (q_min..=q_max).enumerate() {
                fourier[[iq, it]] = Complex::new(0.0, (q as f64) * theta).exp();
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
        let q_min = -2 * trunc.n_max;
        let q_max = 2 * trunc.n_max;
        let harmonic_cache = self.floquet_harmonic_cache(drive, trunc, q_min, q_max);

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
                                    [[harmonic_cache.q_index(n - m), i_r, i, j]];
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

        let q_min = -2 * trunc.n_max;
        let q_max = 2 * trunc.n_max;
        let harmonic_cache = self.floquet_harmonic_cache(drive, trunc, q_min, q_max);
        let hq: Vec<Array2<Complex<f64>>> = (q_min..=q_max)
            .map(|q| self.floquet_cached_harmonic_onek(kvec, q, gauge, &harmonic_cache))
            .collect();

        for (in_sec, n) in trunc.sectors().enumerate() {
            for (im_sec, m) in trunc.sectors().enumerate() {
                let q = n - m;
                let block = &hq[(q - q_min) as usize];
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
    /// Build a same-size high-frequency Floquet effective model.
    ///
    /// With the Fourier convention used in this module,
    ///
    /// $$
    /// H(t)=\sum_q H^{(q)}e^{-iq\Omega t},
    /// $$
    ///
    /// the implemented van Vleck expansion is
    ///
    /// $$ H_{\mathrm{eff}}(\mathbf k) =
    /// H^{(0)}(\mathbf k)
    /// +
    /// \sum_{q=1}^{q_{\max}}
    /// \frac{[H^{(q)}(\mathbf k),H^{(-q)}(\mathbf k)]}{q\Omega}
    /// +
    /// O(\Omega^{-2}). $$
    ///
    /// `order = 0` keeps only `H^(0)`.  `order = 1` adds the commutator term.
    /// Higher orders are not implemented yet.  Pass `None` for `options` to use
    /// first order, `q_max = 2 * trunc.n_max`, and the input model's original
    /// `hamR`.
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
    pub fn floquet_effective_model(
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

        let q_max = options.q_max.unwrap_or(2 * trunc.n_max);
        if q_max < 0 {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.q_max must be non-negative, got {q_max}"
            )));
        }

        let harmonic_cache = self.floquet_harmonic_cache(drive, trunc, -q_max, q_max);
        let kpoints = floquet_uniform_kmesh(&k_mesh);
        let norm = 1.0 / (kpoints.len() as f64);
        let ham = kpoints
            .par_iter()
            .fold(
                || Array3::<Complex<f64>>::zeros((target_ham_r.nrows(), nsta, nsta)),
                |mut partial, kvec| {
                    let h_eff = self.floquet_effective_ham_onek_lattice(
                        kvec,
                        drive,
                        options.order,
                        q_max,
                        &harmonic_cache,
                    );
                    for (i_r, r_vec) in target_ham_r.outer_iter().enumerate() {
                        let phase = inverse_bloch_phase::<DIM, _>(&r_vec, kvec) * norm;
                        let mut block = partial.index_axis_mut(Axis(0), i_r);
                        crate::ndarray_lapack::zaxpy(
                            phase,
                            h_eff.as_slice().unwrap(),
                            block.as_slice_mut().unwrap(),
                        );
                    }
                    partial
                },
            )
            .reduce(
                || Array3::<Complex<f64>>::zeros((target_ham_r.nrows(), nsta, nsta)),
                |mut left, right| {
                    left.zip_mut_with(&right, |a, b| *a += *b);
                    left
                },
            );

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

    fn floquet_effective_ham_onek_lattice<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        drive: &FloquetDrive,
        order: usize,
        q_max: isize,
        harmonic_cache: &FloquetHarmonicCache,
    ) -> Array2<Complex<f64>> {
        let mut h_eff = self.floquet_cached_harmonic_onek(kvec, 0, Gauge::Lattice, harmonic_cache);

        match order {
            0 => {}
            1 => {
                for q in 1..=q_max {
                    let h_pos =
                        self.floquet_cached_harmonic_onek(kvec, q, Gauge::Lattice, harmonic_cache);
                    let h_neg =
                        self.floquet_cached_harmonic_onek(kvec, -q, Gauge::Lattice, harmonic_cache);
                    let comm = h_pos.dot(&h_neg) - h_neg.dot(&h_pos);
                    h_eff = h_eff + comm.mapv(|x| x / ((q as f64) * drive.omega0_ev));
                }
            }
            _ => unreachable!("effective order is validated before evaluation"),
        }

        h_eff
    }

    /// Build the harmonic cache: `t_ij(R) * C_q(d)` for all `q ∈ [q_min, q_max]`.
    ///
    /// Returns a [`FloquetHarmonicCache`] whose `blocks[q_index(q), i_r, i, j]`
    /// stores the `q`-th Fourier coefficient of the Peierls-dressed hopping from
    /// orbital `j` at cell `R = hamR[i_r]` to orbital `i` at the origin.  The
    /// cache is constructed once and reused across the whole k-mesh, avoiding the
    /// recomputation of `C_q(d)` for every k-point.
    ///
    /// When `drive.modes` is empty (static limit), the zero-frequency block is
    /// set to the original `ham` directly.
    fn floquet_harmonic_cache(
        &self,
        drive: &FloquetDrive,
        trunc: &FloquetTruncation,
        q_min: isize,
        q_max: isize,
    ) -> FloquetHarmonicCache {
        let nsta = self.nsta();
        let norb = self.norb();
        let n_r = self.hamR.nrows();
        let q_count = (q_max - q_min + 1) as usize;
        let mut blocks = Array4::<Complex<f64>>::zeros((q_count, n_r, nsta, nsta));

        if drive.modes.is_empty() {
            if q_min <= 0 && 0 <= q_max {
                blocks
                    .slice_mut(s![(0 - q_min) as usize, .., .., ..])
                    .assign(&self.ham);
            }
            return FloquetHarmonicCache {
                q_min,
                q_max,
                blocks,
            };
        }

        let time_grid = FloquetTimeGrid::new(drive, trunc, q_min, q_max, DIM);
        blocks
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .enumerate()
            .for_each(|(i_r, mut out_r)| {
                let r_vec = self.hamR.row(i_r);
                let block = self.ham.index_axis(Axis(0), i_r);
                for i in 0..nsta {
                    for j in 0..nsta {
                        let t = block[[i, j]];
                        if t.norm_sqr() == 0.0 {
                            continue;
                        }

                        let d_cart = self.link_displacement_cartesian(i % norb, j % norb, &r_vec);
                        let coeffs =
                            peierls_fourier_coeffs(&d_cart, q_min, q_max, drive, &time_grid);
                        for (iq, coeff) in coeffs.into_iter().enumerate() {
                            if coeff.norm_sqr() != 0.0 {
                                out_r[[iq, i, j]] = t * coeff;
                            }
                        }
                    }
                }
            });

        FloquetHarmonicCache {
            q_min,
            q_max,
            blocks,
        }
    }

    /// Build the `q`-th Fourier block `H^(q)(k)` from the precomputed cache.
    ///
    /// For each R-vector, multiplies the cached block `t * C_q(d)` by the Bloch
    /// phase `exp(2πi k·R)` via `zaxpy`.  The [`Gauge`] selects between the
    /// lattice gauge (raw Fourier sum) and the atom gauge (with orbital-position
    /// phases applied).
    fn floquet_cached_harmonic_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        q: isize,
        gauge: Gauge,
        harmonic_cache: &FloquetHarmonicCache,
    ) -> Array2<Complex<f64>> {
        let nsta = self.nsta();
        let mut hamq = Array2::<Complex<f64>>::zeros((nsta, nsta));
        let iq = harmonic_cache.q_index(q);
        let hamq_slice = hamq.as_slice_mut().unwrap();

        for i_r in 0..self.hamR.nrows() {
            let r_vec = self.hamR.row(i_r);
            let bloch = bloch_phase::<DIM, S>(&r_vec, kvec);
            let block = harmonic_cache.blocks.slice(s![iq, i_r, .., ..]);
            crate::ndarray_lapack::zaxpy(bloch, block.as_slice().unwrap(), hamq_slice);
        }

        match gauge {
            Gauge::Lattice => hamq,
            Gauge::Atom => self.apply_atom_gauge(kvec, hamq),
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

fn validate_effective_options<const DIM: usize>(
    k_mesh: &[usize; DIM],
    options: &FloquetEffectiveOptions,
) -> Result<()> {
    if options.order > 1 {
        return Err(TbError::Other(format!(
            "Floquet effective order {} is not implemented; supported orders are 0 and 1",
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
    if let Some(q_max) = options.q_max {
        if q_max < 0 {
            return Err(TbError::Other(format!(
                "FloquetEffectiveOptions.q_max must be non-negative, got {q_max}"
            )));
        }
    }
    if let Some(target_ham_r) = &options.target_hamR {
        validate_target_hamr::<DIM>(target_ham_r)?;
    }
    Ok(())
}

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

fn peierls_fourier_coeffs(
    d_cart: &Array1<f64>,
    q_min: isize,
    q_max: isize,
    drive: &FloquetDrive,
    time_grid: &FloquetTimeGrid,
) -> Vec<Complex<f64>> {
    let q_count = (q_max - q_min + 1) as usize;
    if drive.modes.is_empty() {
        let mut coeffs = vec![Complex::new(0.0, 0.0); q_count];
        if q_min <= 0 && 0 <= q_max {
            coeffs[(0 - q_min) as usize] = Complex::new(1.0, 0.0);
        }
        return coeffs;
    }

    let mut coeffs = vec![Complex::new(0.0, 0.0); q_count];
    for it in 0..time_grid.link_field.nrows() {
        let mut link_phase = 0.0;
        for a in 0..d_cart.len() {
            link_phase += time_grid.link_field[[it, a]] * d_cart[a];
        }
        let peierls = Complex::new(0.0, -link_phase).exp();
        for (iq, coeff) in coeffs.iter_mut().enumerate() {
            *coeff += time_grid.fourier[[iq, it]] * peierls;
        }
    }
    for coeff in &mut coeffs {
        *coeff *= time_grid.inv_n_time;
    }
    coeffs
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

fn inverse_bloch_phase<const DIM: usize, S: Data<Elem = f64>>(
    r_vec: &ArrayView1<'_, isize>,
    kvec: &ArrayBase<S, Ix1>,
) -> Complex<f64> {
    bloch_phase::<DIM, S>(r_vec, kvec).conj()
}

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
    use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
    use crate::model::NoRMatrix;
    use crate::model_build::*;
    use crate::solve_ham::solve;
    use ndarray::{arr1, array};

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

        let effective = model
            .floquet_effective_model(&drive, &trunc, [16], None)
            .unwrap();
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
            .floquet_effective_model(&drive, &trunc, [8], Some(&options))
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
            .floquet_effective_model(&drive, &trunc, [8], Some(&options))
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
            .floquet_effective_model(&drive, &trunc, [32], Some(&options))
            .unwrap();

        assert_eq!(effective.nsta(), model.nsta());
        assert_eq!(effective.hamR, model.hamR);

        let k = arr1(&[0.173]);
        let from_model = effective.gen_ham(&k, Gauge::Lattice);
        let harmonic_cache = model.floquet_harmonic_cache(&drive, &trunc, 0, 0);
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
            .floquet_effective_model(&drive, &trunc, [32], None)
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
}
