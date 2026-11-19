//! TB2J-style exchange extraction from a tight-binding [`Model`].
//!
//! Two entry points build a [`Magnon`] from a collinear band structure:
//!
//! * [`Magnon::from_model`] takes one spinful `Model<true, DIM, R>` and splits
//!   its spin-major blocks into the up and down channels;
//! * [`Magnon::from_collinear`] takes two spinless models, one per channel.
//!
//! Both require the input Hamiltonians to be referenced to the Fermi level:
//! subtract the original Fermi energy from both spin channels before extraction.
//! Both run the same extraction, which follows the magnetic force theorem with
//! a local rigid spin rotation (LKAG / TB2J):
//!
//! ```text
//! J_ij(R) = (1/4 pi) Im int dE Tr_L [ Delta_i G^up_ij(R, E) Delta_j G^dn_ji(-R, E) ]
//! ```
//!
//! with `Delta_i = H^up_ii - H^dn_ii` the on-site exchange splitting of magnetic
//! site `i`, and
//!
//! ```text
//! G^s_ij(R, z) = (1/N_k) sum_k [ z I - H^s(k) ]^{-1}_{ij} exp(-2 pi i k.R).
//! ```
//!
//! The default energy integral uses the continued-fraction representation of
//! the Fermi function. The optional upper-half-plane semicircle connects
//! `emax` and `emin`, relative to the zero Fermi energy, and uses the composite
//! Simpson rule.
//!
//! The result is the TB2J **unit-vector landscape coefficient** in eV: `J > 0`
//! is ferromagnetic, and the sign is fixed by `sign(axis_i . axis_j)` so that an
//! antiferromagnetic reference still reports a positive coupling as
//! antiferromagnetic. This construction yields the isotropic part only; the
//! anisotropic and Dzyaloshinskii-Moriya parts need the spinor (four-component)
//! treatment and are not produced here.
//!
//! # Magnetic sites and moments
//!
//! Sites come from the model's [`Atom`] metadata: each atom owns a list of
//! orbital indices, and an atom is magnetic when it carries a magnetic moment
//! (or is listed in [`ExchangeParams::magnetic_atoms`]). An orbital-only model
//! without atoms falls back to one site per orbital.
//!
//! Moments are resolved in this order:
//!
//! 1. [`ExchangeParams::spin`] together with [`ExchangeParams::axis`];
//! 2. the atom magnetic moments;
//! 3. otherwise the **site magnetization of the two channels**,
//!    `m_i = Tr(rho_up - rho_dn)` over the site's orbitals at zero temperature.
//!    This is the usual case for Wannier90 input, which stores no moments, and
//!    it is what TB2J calls `spinat`: `S_i = |m_i| / 2` in units of `hbar`
//!    (a `g = 2`, orbital-quenched moment) and `axis_i = +z` or `-z` along the
//!    Hamiltonian's own quantization axis.
//!
//! An explicitly supplied direction is cross-checked against that
//! magnetization. A *global* spin flip is a symmetry of the problem and is
//! accepted, but a per-site contradiction is rejected with
//! `moment_direction_consistent`, because it would silently flip the sign of
//! `J` (the couplings are normalized by `sign(axis_i . axis_j)`).

//! # Examples
//!
//! ```rust
//! use ndarray::array;
//! use Rustb::magnon::{ExchangeParams, Magnon};
//! use Rustb::{Atom, AtomType, Model, OrbitalId, SpinDirection};
//!
//! # fn main() -> Rustb::Result<()> {
//! let atoms = vec![
//!     Atom::with_orbitals(array![0.0], AtomType::Fe, [OrbitalId::new(0)]),
//!     Atom::with_orbitals(array![0.5], AtomType::Fe, [OrbitalId::new(1)]),
//! ];
//! let mut model =
//!     Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], Some(atoms))?;
//! model.atoms[0].set_magnetic_moment([0.0, 0.0, 3.0])?;
//! model.atoms[1].set_magnetic_moment([0.0, 0.0, 3.0])?;
//! model.set_hop(0.3, 0, 1, &array![0isize], None);
//! model.set_onsite_one(0.5, 0, SpinDirection::Z);
//! model.set_onsite_one(0.5, 1, SpinDirection::Z);
//!
//! let params = ExchangeParams::new(&array![3usize]).with_nz(33);
//! let magnon = Magnon::from_model(&model, &params)?;
//! assert_eq!(magnon.nspin(), 2);
//! // S = 3 mu_B / 2 = 1.5 hbar, moments along +z.
//! assert!((magnon.spin[0] - 1.5).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

use super::{Magnon, MagnonReference, SPIN_COMPONENTS};
use crate::atom_struct::Atom;
use crate::error::{Result, TbError};
use crate::kpoints::gen_kmesh;
use crate::model::{Model, RMatrixData};
use crate::ndarray_lapack::eigh_full;
use ndarray::{Array1, Array2, Array3, ArrayBase, ArrayView1, ArrayView3, Axis, Data, Ix1, s};
use ndarray_linalg::UPLO;
use num_complex::Complex;
use rayon::prelude::*;

/// Energy-integration scheme for the exchange kernel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MagnonContour {
    /// Upper-half-plane semicircle with a composite Simpson rule.
    ///
    /// The integrand is analytic in the upper half plane, so the arc integral
    /// equals the real-axis integral. Accurate for gapped systems, but the arc
    /// ends on the real axis at `emax`. For a metal, ending at zero can put
    /// nodes near band energies and slow convergence.
    Semicircle,
    /// TB2J's continued-fraction representation of the Fermi function.
    ///
    /// The tridiagonal continued-fraction matrix gives poles and residues.
    /// Integration nodes lie off the real energy axis at finite temperature,
    /// avoiding real-axis poles in metals. The `-pi/2` conversion factor is
    /// applied to the accumulated sum as in TB2J. This is the default.
    Cfr {
        /// Electronic temperature of the representation, in kelvin.
        temperature_kelvin: f64,
    },
}

impl Default for MagnonContour {
    fn default() -> Self {
        // TB2J's default smearing is `600 K * k_B`.
        Self::Cfr {
            temperature_kelvin: 600.0,
        }
    }
}

/// Parameters of a TB2J-style exchange extraction.
///
/// Defaults follow TB2J: a `-10.0` to `0.0` eV window relative to the Fermi
/// level and 65 contour nodes. The k-mesh is forced to odd components, because
/// an odd mesh makes the commensurate real-space box
/// `R_i in [-(n_i-1)/2, (n_i-1)/2]` closed under `R -> -R`.
#[derive(Clone, Debug)]
pub struct ExchangeParams {
    /// k-point mesh; every component is rounded up to the next odd integer.
    pub kmesh: Array1<usize>,
    /// Lower end of the semicircle contour in eV, relative to the zero Fermi level.
    pub emin: f64,
    /// Upper end of the semicircle contour in eV, relative to the zero Fermi level.
    pub emax: f64,
    /// Number of contour nodes; interpretation depends on [`Self::contour`].
    ///
    /// [`MagnonContour::Cfr`] builds an `nz x nz` continued-fraction matrix (the
    /// TB2J convention), while [`MagnonContour::Semicircle`] needs an odd node
    /// count of at least three. The default, 101, is valid for both.
    pub nz: usize,
    /// Energy-integration scheme.
    pub contour: MagnonContour,
    /// Optional pair-distance cutoff in angstrom.
    pub rcut: Option<f64>,
    /// Optional spin quantum numbers `S_i` in units of `hbar`.
    ///
    /// When both this and [`Self::axis`] are set they take precedence over the
    /// atom moments and over the magnetization derived from the two channels.
    pub spin: Option<Array1<f64>>,
    /// Optional reference axes, shape `(n, 3)`, unit rows.
    ///
    /// A globally flipped set of axes is accepted, a per-site contradiction
    /// with the on-site magnetization is rejected.
    pub axis: Option<Array2<f64>>,
    /// Optional explicit magnetic sites, as atom indices.
    pub magnetic_atoms: Option<Vec<usize>>,
}

impl ExchangeParams {
    /// Default parameters (`emin = -10`, `emax = 0`, `nz = 101`).
    /// The electronic Hamiltonian must already be referenced to its Fermi level.
    pub fn new<S: Data<Elem = usize>>(kmesh: &ArrayBase<S, Ix1>) -> Self {
        Self {
            kmesh: kmesh.to_owned(),
            emin: -10.0,
            emax: 0.0,
            nz: 101,
            contour: MagnonContour::default(),
            rcut: None,
            spin: None,
            axis: None,
            magnetic_atoms: None,
        }
    }

    /// Set the semicircle window relative to the zero Fermi level.
    /// The default continued-fraction contour does not use this window.
    pub fn with_energy_window(mut self, emin: f64, emax: f64) -> Self {
        self.emin = emin;
        self.emax = emax;
        self
    }

    /// Set the number of contour nodes: odd and at least 3 for
    /// [`MagnonContour::Semicircle`], at least 2 for [`MagnonContour::Cfr`].
    pub fn with_nz(mut self, nz: usize) -> Self {
        self.nz = nz;
        self
    }

    /// Use the upper-half-plane semicircle (composite Simpson).
    pub fn with_semicircle_contour(mut self) -> Self {
        self.contour = MagnonContour::Semicircle;
        self
    }

    /// Use TB2J's continued-fraction representation at `temperature_kelvin`.
    pub fn with_cfr_contour(mut self, temperature_kelvin: f64) -> Self {
        self.contour = MagnonContour::Cfr { temperature_kelvin };
        self
    }

    /// Keep only pairs closer than `rcut` angstrom.
    pub fn with_rcut(mut self, rcut: f64) -> Self {
        self.rcut = Some(rcut);
        self
    }

    /// Override the spin quantum numbers.
    pub fn with_spin(mut self, spin: Array1<f64>) -> Self {
        self.spin = Some(spin);
        self
    }

    /// Override the reference axes.
    pub fn with_axis(mut self, axis: Array2<f64>) -> Self {
        self.axis = Some(axis);
        self
    }

    /// Restrict the magnetic sites to these atom indices.
    pub fn with_magnetic_atoms(mut self, atoms: Vec<usize>) -> Self {
        self.magnetic_atoms = Some(atoms);
        self
    }
}

impl<const DIM: usize> Magnon<DIM> {
    /// Extract exchange parameters from one spinful collinear model.
    ///
    /// Shift the electronic Hamiltonian so its Fermi level is at zero first.
    /// The spinful basis is spin-major (`[orb_0_up, ..., orb_{n-1}_up,
    /// orb_0_down, ...]`), so the two channels are the diagonal `norb x norb`
    /// blocks of every hopping matrix. A model with non-negligible off-diagonal
    /// spin blocks describes a non-collinear state and is rejected; use the
    /// spinor construction for that case.
    pub fn from_model<R: RMatrixData>(
        model: &Model<true, DIM, R>,
        params: &ExchangeParams,
    ) -> Result<Self> {
        let norb = model.norb();
        let up = model.ham.slice(s![.., 0..norb, 0..norb]);
        let down = model.ham.slice(s![.., norb.., norb..]);
        let mut off_diagonal: f64 = 0.0;
        let mut largest: f64 = 0.0;
        for i_r in 0..model.ham.shape()[0] {
            for i in 0..norb {
                for j in 0..norb {
                    off_diagonal = off_diagonal.max(model.ham[[i_r, i, norb + j]].norm());
                    off_diagonal = off_diagonal.max(model.ham[[i_r, norb + i, j]].norm());
                    largest = largest
                        .max(model.ham[[i_r, i, j]].norm())
                        .max(model.ham[[i_r, norb + i, norb + j]].norm());
                }
            }
        }
        if off_diagonal > 1e-8 * largest.max(1e-12) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "collinear_model",
                message: format!(
                    "the spin-up/down off-diagonal blocks are non-zero ({off_diagonal:e}); \
                     the collinear exchange extraction needs a collinear model"
                ),
            });
        }
        Self::build_from_channels(
            &model.lat,
            &model.orb,
            &model.hamR,
            up,
            down,
            &model.atoms,
            params,
        )
    }

    /// Extract exchange parameters from an up/down pair of spinless models.
    ///
    /// Shift both channels by the same original Fermi energy so their common
    /// Fermi level is zero. The two models must describe the same lattice and
    /// translation list; only their hopping matrices may differ. Independent
    /// Wannier calculations can have slightly different centres; these are
    /// accepted with a warning (the up channel supplies the geometry). A
    /// permuted orbital order cannot be detected and would change the result.
    pub fn from_collinear<R1: RMatrixData, R2: RMatrixData>(
        up: &Model<false, DIM, R1>,
        down: &Model<false, DIM, R2>,
        params: &ExchangeParams,
    ) -> Result<Self> {
        if up.hamR != down.hamR {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "matching_translations",
                message: "the up and down models must share the same translation list".into(),
            });
        }
        if up.ham.dim() != down.ham.dim() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "matching_basis",
                message: format!(
                    "the up and down models have different hopping shapes: {:?} and {:?}",
                    up.ham.dim(),
                    down.ham.dim()
                ),
            });
        }
        let lattice_gap = (&up.lat - &down.lat)
            .mapv(f64::abs)
            .fold(0.0, |acc: f64, value| acc.max(*value));
        if lattice_gap > 1e-8 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "matching_lattice",
                message: format!(
                    "the up and down models must share the same lattice (max deviation {lattice_gap:e} A)"
                ),
            });
        }
        // The two spin channels are independent Wannier calculations, so their
        // centres may differ slightly even though the orbital indices align.
        // TB2J uses them index-by-index; the up channel provides the geometry.
        let mut position_gap = 0.0f64;
        for row in 0..up.orb.nrows() {
            let mut delta = Array1::<f64>::zeros(DIM);
            for axis in 0..DIM {
                delta[axis] = up.orb[[row, axis]] - down.orb[[row, axis]];
                delta[axis] -= delta[axis].round();
            }
            let cartesian = delta.dot(&up.lat);
            let distance = cartesian
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            position_gap = position_gap.max(distance);
        }
        if position_gap > 1e-3 {
            eprintln!(
                "warning: the up and down Wannier centres differ by up to {position_gap:.3e} A; \
                 the up channel positions are used"
            );
        }
        Self::build_from_channels(
            &up.lat,
            &up.orb,
            &up.hamR,
            up.ham.view(),
            down.ham.view(),
            &up.atoms,
            params,
        )
    }

    fn build_from_channels(
        lat: &Array2<f64>,
        orb: &Array2<f64>,
        ham_r: &Array2<isize>,
        up: ArrayView3<Complex<f64>>,
        down: ArrayView3<Complex<f64>>,
        atoms: &[Atom],
        params: &ExchangeParams,
    ) -> Result<Self> {
        if up.dim() != down.dim() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "matching_basis",
                message: "the up and down channels must have the same shape".into(),
            });
        }
        if ham_r.ncols() != DIM {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "translation_shape",
                message: format!("expected {DIM} columns, found {}", ham_r.ncols()),
            });
        }
        if params.kmesh.len() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "exchange k-mesh".into(),
                expected: DIM,
                found: params.kmesh.len(),
            });
        }
        let n_orb = up.shape()[1];
        let (sites, site_positions) = resolve_sites(atoms, orb, n_orb, params)?;
        let n_site = sites.len();

        let mesh = forced_odd_mesh(&params.kmesh)?;
        let kpoints: Array2<f64> = gen_kmesh::<f64>(&mesh)?;
        let r_box = reciprocal_box::<DIM>(&mesh);
        let n_box = r_box.nrows();
        let mut negative = Vec::with_capacity(n_box);
        for i_r in 0..n_box {
            let mut target = [0isize; 3];
            for axis in 0..DIM {
                target[axis] = -r_box[[i_r, axis]];
            }
            negative.push(index_of_translation(&r_box, &target).ok_or_else(|| {
                TbError::InvalidMagnonInvariant {
                    invariant: "closed_real_space_box",
                    message: format!("the box is not closed under R -> -R at {target:?}"),
                }
            })?);
        }

        let zero_row = index_of_translation(ham_r, &[0isize]).ok_or_else(|| {
            TbError::InvalidMagnonInvariant {
                invariant: "zero_translation_present",
                message: "the hopping list must contain R = 0".into(),
            }
        })?;
        let delta = up.index_axis(Axis(0), zero_row).to_owned()
            - down.index_axis(Axis(0), zero_row).to_owned();

        let (contour, contour_factor) = match params.contour {
            MagnonContour::Semicircle => (
                semicircle_contour(params.emin, params.emax, params.nz)?,
                1.0,
            ),
            MagnonContour::Cfr { temperature_kelvin } => (
                cfr_contour(params.nz, temperature_kelvin)?,
                -0.5 * std::f64::consts::PI,
            ),
        };
        // One eigendecomposition per k-point and channel, reused by every
        // contour node: the k -> R transform is then one BLAS product per node.
        let band_up = prepare_channel(ham_r, up, &kpoints)?;
        let band_down = prepare_channel(ham_r, down, &kpoints)?;
        // The two channels carry no magnetic moments; derive them from the
        // density matrices and check any caller-supplied direction against them.
        let magnetization = site_magnetization(&band_up, &band_down, &sites)?;
        let (spin, axis) = resolve_moments(atoms, &sites, &magnetization, params)?;
        let mut phases = Array2::<Complex<f64>>::zeros((n_box, kpoints.nrows()));
        for i_r in 0..n_box {
            for (index, k) in kpoints.axis_iter(Axis(0)).enumerate() {
                let phase = -std::f64::consts::TAU
                    * (0..r_box.ncols())
                        .map(|axis| r_box[[i_r, axis]] as f64 * k[axis])
                        .sum::<f64>();
                phases[[i_r, index]] =
                    Complex::from_polar(1.0, phase) / Complex::new(kpoints.nrows() as f64, 0.0);
            }
        }
        let mut accumulator = Array3::<Complex<f64>>::zeros((n_box, n_site, n_site));
        for (energy, weight) in contour.iter() {
            let g_up = greens_on_box(&band_up, &phases, *energy)?;
            let g_down = greens_on_box(&band_down, &phases, *energy)?;
            for i_r in 0..n_box {
                let j_r = negative[i_r];
                for i in 0..n_site {
                    for j in 0..n_site {
                        let value =
                            trace_product(&delta, &sites[i], &sites[j], &g_up, i_r, &g_down, j_r);
                        accumulator[[i_r, i, j]] += *weight * value;
                    }
                }
            }
        }

        let mut ham = Array3::<f64>::zeros((n_box, 3 * n_site, 3 * n_site));
        for i_r in 0..n_box {
            for i in 0..n_site {
                for j in 0..n_site {
                    let is_self = i == j && r_box.row(i_r).iter().all(|&value| value == 0);
                    if is_self {
                        continue;
                    }
                    if let Some(cutoff) = params.rcut
                        && pair_distance(lat, &site_positions, i, j, &r_box, i_r) > cutoff
                    {
                        continue;
                    }
                    let alignment: f64 = (0..SPIN_COMPONENTS)
                        .map(|k| axis[[i, k]] * axis[[j, k]])
                        .sum();
                    let sign = if alignment < 0.0 { -1.0 } else { 1.0 };
                    let coupling = (accumulator[[i_r, i, j]] * contour_factor).im
                        / (4.0 * std::f64::consts::PI)
                        / sign;
                    for component in 0..SPIN_COMPONENTS {
                        ham[[i_r, 3 * i + component, 3 * j + component]] = coupling;
                    }
                }
            }
        }
        Self::new(
            lat.clone(),
            site_positions,
            spin,
            axis,
            ham,
            r_box,
            MagnonReference::collinear(),
        )
    }
}

/// Round every k-mesh component up to an odd integer.
pub(crate) fn forced_odd_mesh(kmesh: &Array1<usize>) -> Result<Array1<usize>> {
    if kmesh.is_empty() || kmesh.iter().any(|&n| n == 0) {
        return Err(TbError::InvalidKmeshDimensions(kmesh.clone()));
    }
    Ok(kmesh.mapv(|n| if n % 2 == 1 { n } else { n + 1 }))
}

/// Commensurate real-space box of an odd mesh, with `R = 0` first.
pub(crate) fn reciprocal_box<const DIM: usize>(mesh: &Array1<usize>) -> Array2<isize> {
    let mut points: Vec<[isize; 3]> = vec![[0; 3]];
    for axis in 0..DIM {
        let half = (mesh[axis] as isize - 1) / 2;
        let mut extended = Vec::with_capacity(points.len() * (2 * half as usize + 1));
        for point in &points {
            for value in -half..=half {
                let mut next = *point;
                next[axis] = value;
                extended.push(next);
            }
        }
        points = extended;
    }
    points.sort_by_key(|point| {
        let zero = point[..DIM].iter().all(|&value| value == 0);
        (!zero, point[0], point[1], point[2])
    });
    let mut out = Array2::<isize>::zeros((points.len(), DIM));
    for (row, point) in points.iter().enumerate() {
        for axis in 0..DIM {
            out[[row, axis]] = point[axis];
        }
    }
    out
}

/// Row of `translations` equal to `target` (missing components count as zero).
pub(crate) fn index_of_translation(
    translations: &Array2<isize>,
    target: &[isize],
) -> Option<usize> {
    'rows: for row in 0..translations.nrows() {
        for axis in 0..translations.ncols() {
            if translations[[row, axis]] != target.get(axis).copied().unwrap_or(0) {
                continue 'rows;
            }
        }
        return Some(row);
    }
    None
}

/// Distance between site `i` and site `j` in cell `R`, in angstrom.
fn pair_distance(
    lat: &Array2<f64>,
    site_positions: &Array2<f64>,
    i: usize,
    j: usize,
    r_box: &Array2<isize>,
    i_r: usize,
) -> f64 {
    let dim = lat.nrows();
    let mut fractional = [0.0f64; 3];
    for axis in 0..dim {
        fractional[axis] =
            r_box[[i_r, axis]] as f64 + site_positions[[j, axis]] - site_positions[[i, axis]];
    }
    let mut distance_sq = 0.0;
    for component in 0..dim {
        let mut cartesian = 0.0;
        for axis in 0..dim {
            cartesian += fractional[axis] * lat[[axis, component]];
        }
        distance_sq += cartesian * cartesian;
    }
    distance_sq.sqrt()
}

/// Magnetization magnitude below which a site is treated as non-magnetic.
const MAGNETIZATION_TOLERANCE: f64 = 1e-6;

/// Boltzmann constant in eV per kelvin.
const BOLTZMANN_EV_PER_KELVIN: f64 = 8.617_333_262e-5;

/// TB2J's continued-fraction representation of the Fermi function.
///
/// The poles are the eigenvalues of the tridiagonal matrix
/// `b_j = 1 / (2 sqrt((2j+1)(2j+3)))`, pushed onto the imaginary axis at
/// `k_B T / p`, with residues `|v_0|^2 / (4 p^2)`. Every pole appears as a
/// `+i`/`-i` pair with the same weight `2i/beta * |v_0|^2 / (4 p^2)`. The
/// `-pi/2` conversion factor is applied by the caller, exactly as TB2J applies
/// it after accumulating over the path.
pub(crate) fn cfr_contour(
    nz: usize,
    temperature_kelvin: f64,
) -> Result<Vec<(Complex<f64>, Complex<f64>)>> {
    if nz < 2 {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "contour_nodes",
            message: format!("the continued-fraction contour needs nz >= 2, found {nz}"),
        });
    }
    if !temperature_kelvin.is_finite() || temperature_kelvin <= 0.0 {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "cfr_temperature",
            message: format!(
                "the continued-fraction temperature must be positive, found {temperature_kelvin}"
            ),
        });
    }
    // TB2J builds an `nz x nz` Jacobi matrix with `nz - 1` off-diagonals.
    let size = nz;
    let mut tridiagonal = Array2::<Complex<f64>>::zeros((size, size));
    for j in 0..size.saturating_sub(1) {
        let denominator = 2.0 * (((2 * (j + 1) - 1) as f64) * ((2 * (j + 1) + 1) as f64)).sqrt();
        let value = Complex::new(1.0 / denominator, 0.0);
        tridiagonal[[j, j + 1]] = value;
        tridiagonal[[j + 1, j]] = value;
    }
    let (poles, vectors) = eigh_full(&tridiagonal, UPLO::Lower)?;
    // An odd-sized zero-diagonal Jacobi matrix has one exact zero eigenvalue;
    // round-off may make it a tiny positive number whose `thermal / pole` and
    // `1 / pole^2` blow up. Keep only the genuinely positive poles.
    let threshold = 1e-10 * poles.iter().map(|pole| pole.abs()).fold(0.0, f64::max);
    let thermal = BOLTZMANN_EV_PER_KELVIN * temperature_kelvin;
    let beta = 1.0 / thermal;
    let mut nodes = Vec::with_capacity(2 * size + 1);
    for k in 0..size {
        let pole = poles[k];
        if !pole.is_finite() || pole <= threshold {
            continue;
        }
        let residue = 0.25 * vectors[[k, 0]].norm_sqr() / (pole * pole);
        let weight = Complex::new(0.0, 2.0 / beta) * residue;
        nodes.push((Complex::new(0.0, thermal / pole), weight));
        nodes.push((Complex::new(0.0, -thermal / pole), weight));
    }
    if nodes.is_empty() {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "cfr_poles",
            message: "the continued-fraction contour produced no positive pole".into(),
        });
    }
    // TB2J closes the path with a far imaginary point of zero weight.
    nodes.push((Complex::new(0.0, 1e10), Complex::new(0.0, 0.0)));
    Ok(nodes)
}

/// Contour nodes and integration weights of the upper-half-plane semicircle.
///
/// The weights already include `dz` and the orientation sign, so
/// `sum_k w_k f(z_k)` approximates `int_{emin}^{emax} f(E) dE` for `f` analytic
/// in the upper half plane.
pub(crate) fn semicircle_contour(
    emin: f64,
    emax: f64,
    nz: usize,
) -> Result<Vec<(Complex<f64>, Complex<f64>)>> {
    if !emin.is_finite() || !emax.is_finite() || emin >= emax {
        return Err(TbError::InvalidEnergyRange {
            min: emin,
            max: emax,
        });
    }
    if nz < 3 || nz.is_multiple_of(2) {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "contour_nodes",
            message: format!("nz must be odd and at least 3, found {nz}"),
        });
    }
    let intervals = (nz - 1) as f64;
    let step = std::f64::consts::PI / intervals;
    let centre = 0.5 * (emin + emax);
    let radius = 0.5 * (emax - emin);
    let mut nodes = Vec::with_capacity(nz);
    for index in 0..nz {
        let phi = index as f64 * step;
        let node = Complex::from_polar(radius, phi) + Complex::new(centre, 0.0);
        let simpson = if index == 0 || index == nz - 1 {
            1.0
        } else if index % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let weight =
            -(Complex::i() * radius * Complex::from_polar(1.0, phi)) * (step / 3.0 * simpson);
        nodes.push((node, weight));
    }
    Ok(nodes)
}

/// Eigenvalues and eigenvectors of one spin channel on the whole k-mesh.
struct ChannelBands {
    /// Band energies, shape `(nk, n)`.
    values: Array2<f64>,
    /// Row `k` holds the eigenvectors with the band index in rows: `c[band, orbital]`.
    vectors: Vec<Array2<Complex<f64>>>,
}

fn prepare_channel(
    ham_r: &Array2<isize>,
    ham: ArrayView3<Complex<f64>>,
    kpoints: &Array2<f64>,
) -> Result<ChannelBands> {
    let n_k = kpoints.nrows();
    let n = ham.shape()[1];
    type EighOutput = Result<(Array1<f64>, Array2<Complex<f64>>)>;
    let results: Vec<EighOutput> = kpoints
        .axis_iter(Axis(0))
        .collect::<Vec<_>>()
        .par_iter()
        .map(|k| {
            let hk = hamiltonian_at(ham_r, ham, k);
            eigh_full(&hk, UPLO::Lower)
        })
        .collect();
    let mut values = Array2::<f64>::zeros((n_k, n));
    let mut vectors = Vec::with_capacity(n_k);
    for (index, result) in results.into_iter().enumerate() {
        let (energies, eigenvectors) = result?;
        values.row_mut(index).assign(&energies);
        vectors.push(eigenvectors);
    }
    Ok(ChannelBands { values, vectors })
}

/// Real-space Green's function on the requested box for one contour node.
///
/// `phases[i_r, k] = exp(-2 pi i k.R) / N_k`, so the whole box transform is a
/// single matrix product `phases.dot(gmat)`.
fn greens_on_box(
    bands: &ChannelBands,
    phases: &Array2<Complex<f64>>,
    energy: Complex<f64>,
) -> Result<Array3<Complex<f64>>> {
    let n_k = bands.values.nrows();
    let n = bands.values.ncols();
    let n_box = phases.nrows();
    let mut gmat = Array2::<Complex<f64>>::zeros((n_k, n * n));
    for k in 0..n_k {
        let vectors = &bands.vectors[k];
        for band in 0..n {
            let denominator = energy - bands.values[[k, band]];
            if denominator.norm() < 1e-14 {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "contour_hits_eigenvalue",
                    message: "a contour node coincides with a band energy".into(),
                });
            }
            let factor = Complex::new(1.0, 0.0) / denominator;
            for alpha in 0..n {
                let left = vectors[[band, alpha]] * factor;
                for beta in 0..n {
                    gmat[[k, alpha * n + beta]] += left * vectors[[band, beta]].conj();
                }
            }
        }
    }
    let transformed = phases.dot(&gmat);
    let mut out = Array3::<Complex<f64>>::zeros((n_box, n, n));
    for i_r in 0..n_box {
        for alpha in 0..n {
            for beta in 0..n {
                out[[i_r, alpha, beta]] = transformed[[i_r, alpha * n + beta]];
            }
        }
    }
    Ok(out)
}

fn hamiltonian_at(
    ham_r: &Array2<isize>,
    ham: ArrayView3<Complex<f64>>,
    k: &ArrayView1<f64>,
) -> Array2<Complex<f64>> {
    let n = ham.shape()[1];
    let mut hk = Array2::<Complex<f64>>::zeros((n, n));
    for i_r in 0..ham_r.nrows() {
        let phase = std::f64::consts::TAU
            * (0..ham_r.ncols())
                .map(|axis| ham_r[[i_r, axis]] as f64 * k[axis])
                .sum::<f64>();
        let factor = Complex::from_polar(1.0, phase);
        for i in 0..n {
            for j in 0..n {
                hk[[i, j]] += ham[[i_r, i, j]] * factor;
            }
        }
    }
    hk
}

/// `Tr[Delta_i G_up_ij(R) Delta_j G_down_ji(-R)]` over the two site blocks.
fn trace_product(
    delta: &Array2<Complex<f64>>,
    site_i: &[usize],
    site_j: &[usize],
    g_up: &Array3<Complex<f64>>,
    i_r: usize,
    g_down: &Array3<Complex<f64>>,
    j_r: usize,
) -> Complex<f64> {
    let mut total = Complex::new(0.0, 0.0);
    for &alpha in site_i {
        for &beta in site_j {
            let mut left = Complex::new(0.0, 0.0);
            for &alpha2 in site_i {
                left += delta[[alpha, alpha2]] * g_up[[i_r, alpha2, beta]];
            }
            let mut right = Complex::new(0.0, 0.0);
            for &beta2 in site_j {
                right += delta[[beta, beta2]] * g_down[[j_r, beta2, alpha]];
            }
            total += left * right;
        }
    }
    total
}

/// Magnetic sites, each as the list of its orbital indices.
fn resolve_sites(
    atoms: &[Atom],
    orbital_positions: &Array2<f64>,
    n_orb: usize,
    params: &ExchangeParams,
) -> Result<(Vec<Vec<usize>>, Array2<f64>)> {
    if let Some(indices) = &params.magnetic_atoms {
        if atoms.is_empty() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "missing_atom_structure",
                message: "magnetic_atoms requires atom metadata".into(),
            });
        }
        let mut sites = Vec::with_capacity(indices.len());
        for &index in indices {
            let atom = atoms.get(index).ok_or(TbError::InvalidMagnonInvariant {
                invariant: "magnetic_atom_index",
                message: format!("atom index {index} is out of range"),
            })?;
            if atom.orbitals().is_empty() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "magnetic_atom_orbitals",
                    message: format!("atom {index} owns no orbitals"),
                });
            }
            sites.push(atom.orbitals().iter().map(|id| id.index()).collect());
        }
        let positions = site_positions_from_atoms(atoms, indices, orbital_positions, sites.len());
        return Ok((sites, positions));
    }
    if !atoms.is_empty() {
        let any_moment = atoms.iter().any(|atom| atom.has_magnetic_moment());
        let mut sites = Vec::new();
        let mut selected = Vec::new();
        for (index, atom) in atoms.iter().enumerate() {
            // With moments attached, the magnetic atoms are those carrying one.
            // Without any moment, every atom is a candidate and the caller must
            // supply `ExchangeParams::spin` and `::axis`.
            if any_moment && !atom.has_magnetic_moment() {
                continue;
            }
            if atom.orbitals().is_empty() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "magnetic_atom_orbitals",
                    message: "a magnetic atom owns no orbitals".into(),
                });
            }
            sites.push(atom.orbitals().iter().map(|id| id.index()).collect());
            selected.push(index);
        }
        let positions = site_positions_from_atoms(atoms, &selected, orbital_positions, sites.len());
        return Ok((sites, positions));
    }
    if n_orb == 0 {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "missing_magnetic_sites",
            message: "the model has no orbitals".into(),
        });
    }
    let sites: Vec<Vec<usize>> = (0..n_orb).map(|index| vec![index]).collect();
    let mut positions = Array2::<f64>::zeros((sites.len(), orbital_positions.ncols()));
    for (row, site) in sites.iter().enumerate() {
        positions
            .row_mut(row)
            .assign(&orbital_positions.row(site[0]));
    }
    Ok((sites, positions))
}

/// Fractional positions of the magnetic sites, taken from the atoms when the
/// model carries them and from the first orbital otherwise.
fn site_positions_from_atoms(
    atoms: &[Atom],
    atom_indices: &[usize],
    orbital_positions: &Array2<f64>,
    site_count: usize,
) -> Array2<f64> {
    let dim = orbital_positions.ncols();
    let mut positions = Array2::<f64>::zeros((site_count, dim));
    for (row, &atom_index) in atom_indices.iter().enumerate() {
        if let Some(atom) = atoms.get(atom_index) {
            let position = atom.position();
            for axis in 0..dim {
                positions[[row, axis]] = position[axis];
            }
        }
    }
    positions
}

/// Site magnetization `Tr(rho_up - rho_dn)` in units of `mu_B`, evaluated at
/// zero temperature from the two channel band structures.
fn site_magnetization(
    band_up: &ChannelBands,
    band_down: &ChannelBands,
    sites: &[Vec<usize>],
) -> Result<Array1<f64>> {
    let n_k = band_up.values.nrows();
    let n = band_up.values.ncols();
    if band_down.values.dim() != (n_k, n) {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "matching_basis",
            message: "the two channels have different band structures".into(),
        });
    }
    let mut magnetization = Array1::<f64>::zeros(sites.len());
    for k in 0..n_k {
        for band in 0..n {
            let occupied_up = band_up.values[[k, band]] < 0.0;
            let occupied_down = band_down.values[[k, band]] < 0.0;
            if !occupied_up && !occupied_down {
                continue;
            }
            for (site, orbitals) in sites.iter().enumerate() {
                for &orbital in orbitals {
                    if occupied_up {
                        magnetization[site] += band_up.vectors[k][[band, orbital]].norm_sqr();
                    }
                    if occupied_down {
                        magnetization[site] -= band_down.vectors[k][[band, orbital]].norm_sqr();
                    }
                }
            }
        }
    }
    Ok(magnetization / n_k as f64)
}

/// Resolve the reference spins and axes.
///
/// Order of precedence: `ExchangeParams::spin`/`::axis`, then the atom magnetic
/// moments, then — the usual case for Wannier90 input, which stores no moments
/// — the site magnetization computed from the two channels (`S_i = |m_i| / 2`,
/// axis `+z` or `-z` along the Hamiltonian's own quantization axis). Whenever an
/// explicit direction is supplied it is cross-checked against that
/// magnetization: a per-site disagreement would silently flip `J`, while a
/// global spin flip is a symmetry and is accepted.
fn resolve_moments(
    atoms: &[Atom],
    sites: &[Vec<usize>],
    magnetization: &Array1<f64>,
    params: &ExchangeParams,
) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = sites.len();
    if let (Some(spin), Some(axis)) = (&params.spin, &params.axis) {
        if spin.len() != n || axis.dim() != (n, SPIN_COMPONENTS) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "moment_override_shape",
                message: format!(
                    "expected {n} spins and an ({n}, 3) axis array, found {} and {:?}",
                    spin.len(),
                    axis.dim()
                ),
            });
        }
        check_axis_against_hamiltonian(axis, magnetization, "ExchangeParams::axis")?;
        return Ok((spin.clone(), axis.clone()));
    }
    if !atoms.is_empty() && atoms.iter().any(|atom| atom.has_magnetic_moment()) {
        let mut spin = Array1::<f64>::zeros(n);
        let mut axis = Array2::<f64>::zeros((n, SPIN_COMPONENTS));
        for (index, site) in sites.iter().enumerate() {
            let atom = atoms
                .iter()
                .find(|atom| atom.orbitals().iter().any(|id| id.index() == site[0]));
            let moment = atom
                .and_then(|atom| atom.magnetic_moment())
                .ok_or_else(|| TbError::InvalidMagnonInvariant {
                    invariant: "missing_moments",
                    message: format!(
                        "magnetic site {index} has no moment; attach one to its atom or pass \
                         ExchangeParams::spin and ::axis"
                    ),
                })?;
            let magnitude =
                (moment[0] * moment[0] + moment[1] * moment[1] + moment[2] * moment[2]).sqrt();
            if magnitude <= 0.0 || !magnitude.is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "non_zero_moment",
                    message: format!("magnetic site {index} has a zero or non-finite moment"),
                });
            }
            spin[index] = magnitude / 2.0;
            for component in 0..SPIN_COMPONENTS {
                axis[[index, component]] = moment[component] / magnitude;
            }
        }
        check_axis_against_hamiltonian(&axis, magnetization, "the atom magnetic moments")?;
        return Ok((spin, axis));
    }
    // Derive the moments from the band structure, as TB2J does.
    let mut spin = Array1::<f64>::zeros(n);
    let mut axis = Array2::<f64>::zeros((n, SPIN_COMPONENTS));
    for index in 0..n {
        let moment = magnetization[index];
        if !moment.is_finite() || moment.abs() < MAGNETIZATION_TOLERANCE {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "non_zero_magnetization",
                message: format!(
                    "site {index} carries no magnetization in the two channels; pass \
                     ExchangeParams::spin and ::axis, or check the Fermi level"
                ),
            });
        }
        spin[index] = moment.abs() / 2.0;
        axis[[index, 2]] = moment.signum();
    }
    Ok((spin, axis))
}

/// Reject a per-site contradiction between a supplied direction and the
/// on-site magnetization of the two channels. A uniform global flip is allowed
/// because it leaves every `sign(axis_i . axis_j)` unchanged.
fn check_axis_against_hamiltonian(
    axis: &Array2<f64>,
    magnetization: &Array1<f64>,
    source: &str,
) -> Result<()> {
    let mut global: Option<f64> = None;
    for index in 0..magnetization.len() {
        let moment = magnetization[index];
        let direction = axis[[index, 2]];
        if !moment.is_finite() || moment.abs() < MAGNETIZATION_TOLERANCE {
            continue;
        }
        if direction.abs() < 0.5 {
            continue;
        }
        let relative = moment.signum() * direction.signum();
        match global {
            None => global = Some(relative),
            Some(expected) if (expected - relative).abs() > 0.5 => {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "moment_direction_consistent",
                    message: format!(
                        "site {index}: the direction from {source} contradicts the on-site \
                         magnetization of the two channels (they agree for the other sites, \
                         so this is not a global spin flip)"
                    ),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom_struct::{Atom, AtomType, OrbitalId};
    use crate::model::SpinDirection;
    use ndarray::array;

    fn atoms() -> Vec<Atom> {
        vec![
            Atom::with_orbitals(array![0.0], AtomType::Fe, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.5], AtomType::Fe, [OrbitalId::new(1)]),
        ]
    }

    /// Two-site chain: hopping `t` within the cell and across the boundary,
    /// on-site exchange splitting `delta[i]` along z.
    fn spinful_model(t: f64, delta: [f64; 2], moments: [[f64; 3]; 2]) -> Model<true, 1> {
        let mut atoms = atoms();
        atoms[0].set_magnetic_moment(moments[0]).unwrap();
        atoms[1].set_magnetic_moment(moments[1]).unwrap();
        let mut model =
            Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], Some(atoms)).unwrap();
        model.set_hop(t, 0, 1, &array![0isize], None);
        model.set_hop(t, 1, 0, &array![1isize], None);
        for (site, splitting) in delta.iter().enumerate() {
            model.set_onsite_one(splitting / 2.0, site, SpinDirection::Z);
        }
        model
    }

    /// The same physics as two spinless channels.
    fn collinear_models(
        t: f64,
        delta: [f64; 2],
        moments: [[f64; 3]; 2],
    ) -> (Model<false, 1>, Model<false, 1>) {
        let build = |shift: f64| {
            let mut atoms = atoms();
            atoms[0].set_magnetic_moment(moments[0]).unwrap();
            atoms[1].set_magnetic_moment(moments[1]).unwrap();
            let mut model =
                Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.5]], Some(atoms))
                    .unwrap();
            model.set_hop(t, 0, 1, &array![0isize], None);
            model.set_hop(t, 1, 0, &array![1isize], None);
            model.set_onsite(
                &array![shift * delta[0] / 2.0, shift * delta[1] / 2.0],
                None,
            );
            model
        };
        (build(1.0), build(-1.0))
    }

    fn exchange_value(magnon: &Magnon<1>, r: isize, i: usize, j: usize) -> f64 {
        let row = index_of_translation(&magnon.hamR, &[r]).unwrap();
        magnon.jtensor(i, j, row).unwrap()[0][0]
    }

    #[test]
    fn test_antiferromagnetic_extraction_matches_reference() {
        // Reference: an independent implementation of the same LKAG formula
        // (numpy), t = 0.3, delta = (1, -1), E_F = 0, 5x5x5 box in 1D (kmesh 5),
        // contour nodes nz = 257.
        let model = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        let params = ExchangeParams::new(&array![5usize])
            .with_nz(257)
            .with_semicircle_contour();
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        assert!((exchange_value(&magnon, 0, 0, 1) + 1.574591087982834e-2).abs() < 1e-12);
        assert!((exchange_value(&magnon, 1, 0, 1) + 4.803340889916693e-4).abs() < 1e-12);
        assert!((exchange_value(&magnon, -1, 0, 1) + 1.574591087982834e-2).abs() < 1e-12);
        // The reverse blocks are the transposes of the forward ones.
        assert!((exchange_value(&magnon, 0, 1, 0) + 1.574591087982834e-2).abs() < 1e-12);
        // Spin quantum numbers come from the moments (3 mu_B -> S = 1.5).
        assert!((magnon.spin[0] - 1.5).abs() < 1e-12);
        assert!((magnon.axis[[1, 2]] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_spinful_and_collinear_extractions_agree() {
        let delta = [1.0, -1.0];
        let moments = [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]];
        let params = ExchangeParams::new(&array![5usize]).with_nz(65);
        let from_model =
            Magnon::<1>::from_model(&spinful_model(0.3, delta, moments), &params).unwrap();
        let (up, down) = collinear_models(0.3, delta, moments);
        let from_pair = Magnon::<1>::from_collinear(&up, &down, &params).unwrap();
        assert_eq!(from_model.hamR, from_pair.hamR);
        let difference = (&from_model.ham - &from_pair.ham)
            .mapv(f64::abs)
            .fold(0.0, |acc: f64, value| acc.max(*value));
        assert!(difference < 1e-14, "max deviation {difference:e}");
        assert_eq!(from_model.spin, from_pair.spin);
        assert_eq!(from_model.axis, from_pair.axis);
    }

    #[test]
    fn test_extracted_magnon_has_a_gapped_spectrum() {
        use crate::magnon::MagnonSolve;
        // Two-sublattice antiferromagnet; the contour choice only shifts the
        // couplings, and the semicircle keeps this reference a minimum.
        let model = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        let params = ExchangeParams::new(&array![5usize])
            .with_nz(65)
            .with_semicircle_contour();
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        // The ferromagnetic Goldstone mode is zero at the Gamma point.
        let omega = magnon.magnon_band_onek(&array![0.0]).unwrap();
        for value in omega.iter() {
            assert!(
                value.abs() < 1e-8,
                "the Goldstone mode is not zero: {value}"
            );
        }
        // Away from Gamma the modes are real and positive.
        let gapped = magnon.magnon_band_onek(&array![0.5]).unwrap();
        assert!(gapped.iter().all(|value| value.is_finite() && *value > 0.0));
    }

    #[test]
    fn test_cfr_extraction_matches_tb2j_reference() {
        // TB2J 0.9.12.26 on the same Wannier90 dataset (t = 0.1 eV, exchange
        // splitting 2 eV, E_F = 0, kmesh 5 1 1, CFR with the 600 K default):
        // Fe1-Fe2 at R = 0 -> -2.5774 meV, Fe1-Fe1 at R = +1 -> +0.0197 meV.
        let model = spinful_model(0.1, [2.0, 2.0], [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]);
        let params = ExchangeParams::new(&array![5usize]);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        let cross = exchange_value(&magnon, 0, 0, 1);
        let self_term = exchange_value(&magnon, 1, 0, 0);
        assert!(
            (cross + 2.5774e-3).abs() < 2e-5,
            "J(Fe1-Fe2, R=0) = {cross:e} eV, expected -2.5774e-3"
        );
        assert!(
            (self_term - 1.97e-5).abs() < 1e-6,
            "J(Fe1-Fe1, R=1) = {self_term:e} eV, expected +1.97e-5"
        );
    }

    #[test]
    fn test_site_selection_skips_atoms_without_moment() {
        // Fe (magnetic), Ge (not magnetic), Fe (magnetic): two sites, and the
        // non-magnetic atom must not shift the site positions.
        let atoms = vec![
            Atom::with_orbitals(array![0.0], AtomType::Fe, [OrbitalId::new(0)]),
            Atom::with_orbitals(array![0.25], AtomType::Ge, [OrbitalId::new(1)]),
            Atom::with_orbitals(array![0.5], AtomType::Fe, [OrbitalId::new(2)]),
        ];
        let mut model =
            Model::<true, 1>::tb_model(array![[1.0]], array![[0.0], [0.25], [0.5]], Some(atoms))
                .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.0]).unwrap();
        model.atoms[2]
            .set_magnetic_moment([0.0, 0.0, -2.0])
            .unwrap();
        model.set_hop(0.2, 0, 1, &array![0isize], None);
        model.set_hop(0.2, 1, 2, &array![0isize], None);
        model.set_onsite_one(1.0, 0, SpinDirection::Z);
        model.set_onsite_one(-1.0, 2, SpinDirection::Z);
        let params = ExchangeParams::new(&array![3usize]);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        assert_eq!(magnon.nspin(), 2);
        assert!((magnon.orb[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((magnon.orb[[1, 0]] - 0.5).abs() < 1e-12);
        assert_eq!(magnon.spin.len(), 2);
    }

    #[test]
    fn test_multi_orbital_site_uses_the_atom_block() {
        // Two magnetic atoms with two orbitals each.
        let atoms = vec![
            Atom::with_orbitals(
                array![0.0],
                AtomType::Fe,
                [OrbitalId::new(0), OrbitalId::new(1)],
            ),
            Atom::with_orbitals(
                array![0.5],
                AtomType::Fe,
                [OrbitalId::new(2), OrbitalId::new(3)],
            ),
        ];
        let mut model = Model::<true, 1>::tb_model(
            array![[1.0]],
            array![[0.0], [0.0], [0.5], [0.5]],
            Some(atoms),
        )
        .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.0]).unwrap();
        model.atoms[1].set_magnetic_moment([0.0, 0.0, 2.0]).unwrap();
        for (i, j) in [
            (0, 1),
            (1, 0),
            (2, 3),
            (3, 2),
            (0, 2),
            (2, 0),
            (1, 3),
            (3, 1),
        ] {
            model.set_hop(0.2, i, j, &array![0isize], None);
        }
        for orbital in 0..4 {
            model.set_onsite_one(1.0, orbital, SpinDirection::Z);
        }
        let params = ExchangeParams::new(&array![3usize]);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        assert_eq!(magnon.nspin(), 2);
        assert_eq!(magnon.orb.dim(), (2, 1));
        let tensor = magnon.jtensor(0, 1, 0).unwrap();
        assert!(tensor[0][0].is_finite());
    }

    #[test]
    fn test_cfr_contour_matches_tb2j() {
        // TB2J `mycfr.CFR(nz=100)` at 200 K: the first node is the largest pole
        // pushed onto the imaginary axis, with residue one.
        let contour = cfr_contour(100, 200.0).unwrap();
        let thermal = BOLTZMANN_EV_PER_KELVIN * 200.0;
        // The largest pole sits closest to the real axis and carries residue one.
        let expected_node = 0.05414430173886841;
        let matched = contour.iter().any(|(node, weight)| {
            (node.im.abs() - expected_node).abs() < 1e-9
                && (weight.im - 2.0 * thermal).abs() < 1e-12
                && weight.re.abs() < 1e-18
        });
        assert!(
            matched,
            "no CFR node at +/-{expected_node}i with weight 2 k_B T"
        );
        // The path is closed by a far imaginary point with zero weight.
        let (tail, tail_weight) = *contour.last().unwrap();
        assert_eq!(tail, Complex::new(0.0, 1e10));
        assert_eq!(tail_weight, Complex::new(0.0, 0.0));
        // Every node lies on the imaginary axis.
        assert!(contour.iter().all(|(node, _)| node.re == 0.0));
    }

    #[test]
    fn test_cfr_contour_rejects_degenerate_input() {
        assert!(cfr_contour(1, 600.0).is_err());
        assert!(cfr_contour(100, 0.0).is_err());
        assert!(cfr_contour(100, -5.0).is_err());
        assert!(cfr_contour(100, f64::NAN).is_err());
        // nz = 2 is a valid one-pole contour, not a silent zero.
        let contour = cfr_contour(2, 600.0).unwrap();
        assert!(contour.len() >= 3);
        assert!(contour.iter().any(|(_, weight)| weight.norm() > 0.0));
    }

    #[test]
    fn test_moments_are_derived_from_the_hamiltonian() {
        // Wannier90 files carry no moments: clear the atom metadata and let the
        // two channels provide them (TB2J's `spinat`).
        let mut model = spinful_model(0.1, [2.0, 2.0], [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]);
        model.atoms[0].clear_magnetic_moment();
        model.atoms[1].clear_magnetic_moment();
        let params = ExchangeParams::new(&array![5usize]);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        // The up level sits above E_F, so each site carries m = -1 mu_B.
        for site in 0..2 {
            assert!((magnon.spin[site] - 0.5).abs() < 1e-12, "{:?}", magnon.spin);
            assert!((magnon.axis[[site, 2]] + 1.0).abs() < 1e-12);
        }
        // Same coupling as the explicit-moment path: the global sign cancels.
        assert!((exchange_value(&magnon, 0, 0, 1) + 2.5774e-3).abs() < 2e-5);

        // A common onsite shift leaves the spin splitting unchanged, but
        // now both channels lie above zero and neither site has a moment.
        model.add_onsite(&array![4.0, 4.0], None);
        assert!(matches!(
            Magnon::<1>::from_model(&model, &params),
            Err(TbError::InvalidMagnonInvariant {
                invariant: "non_zero_magnetization",
                ..
            })
        ));
    }

    #[test]
    fn test_mixed_axis_directions_are_rejected() {
        let model = spinful_model(0.1, [2.0, 2.0], [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]);
        let base = ExchangeParams::new(&array![5usize]);
        // A global spin flip keeps every sign(axis_i . axis_j) and is accepted.
        let flipped = base
            .clone()
            .with_spin(array![0.5, 0.5])
            .with_axis(array![[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]);
        let magnon = Magnon::<1>::from_model(&model, &flipped).unwrap();
        assert!((exchange_value(&magnon, 0, 0, 1) + 2.5774e-3).abs() < 2e-5);
        // Flipping only one site contradicts the Hamiltonian and must fail.
        let mixed = base
            .with_spin(array![0.5, 0.5])
            .with_axis(array![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]);
        let error = Magnon::<1>::from_model(&model, &mixed).unwrap_err();
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => {
                assert_eq!(invariant, "moment_direction_consistent")
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_extraction_rejects_inconsistent_inputs() {
        let model = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        // Even meshes are rounded up to odd, a wrong dimension is rejected.
        let params = ExchangeParams::new(&array![4usize]).with_nz(65);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        assert_eq!(magnon.hamR.nrows(), 5);
        let params = ExchangeParams::new(&array![4usize, 4]).with_nz(65);
        let error = Magnon::<1>::from_model(&model, &params).unwrap_err();
        assert!(matches!(error, TbError::DimensionMismatch { .. }));
        // A non-collinear model is rejected.
        let mut spinor = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        spinor.set_hop(0.05, 0, 0, &array![0isize], SpinDirection::X);
        let params = ExchangeParams::new(&array![3usize]).with_nz(65);
        let error = Magnon::<1>::from_model(&spinor, &params).unwrap_err();
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => {
                assert_eq!(invariant, "collinear_model")
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
