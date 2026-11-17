//! TB2J-style exchange extraction from a tight-binding [`Model`].
//!
//! Two entry points build a [`Magnon`] from a collinear band structure:
//!
//! * [`Magnon::from_model`] takes one spinful `Model<true, DIM, R>` and splits
//!   its spin-major blocks into the up and down channels;
//! * [`Magnon::from_collinear`] takes two spinless models, one per channel.
//!
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
//! G^s_ij(R, z) = (1/N_k) sum_k [ (z + E_F) I - H^s(k) ]^{-1}_{ij} exp(-2 pi i k.R).
//! ```
//!
//! The energy integral runs on the upper-half-plane semicircle that connects
//! `E_F + emax` and `E_F + emin`; the integrand is analytic there, so the arc
//! integral equals the real-axis integral. A composite Simpson rule provides the
//! nodes and weights.
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
//! without atoms falls back to one site per orbital. The spin quantum numbers
//! are `S_i = |m_i| / 2` in units of `hbar` (a `g = 2`, orbital-quenched
//! moment) and the reference axes are the normalized moments; both can be
//! overridden through [`ExchangeParams::spin`] and [`ExchangeParams::axis`].

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
//! let params = ExchangeParams::new(0.0, &array![3usize]).with_nz(33);
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

/// Parameters of a TB2J-style exchange extraction.
///
/// Defaults follow TB2J: a `-10.0` to `0.0` eV window relative to the Fermi
/// level and 65 contour nodes. The k-mesh is forced to odd components, because
/// an odd mesh makes the commensurate real-space box
/// `R_i in [-(n_i-1)/2, (n_i-1)/2]` closed under `R -> -R`.
#[derive(Clone, Debug)]
pub struct ExchangeParams {
    /// Fermi energy in eV (absolute, not relative).
    pub efermi: f64,
    /// k-point mesh; every component is rounded up to the next odd integer.
    pub kmesh: Array1<usize>,
    /// Lower end of the energy contour, relative to `efermi`, in eV.
    pub emin: f64,
    /// Upper end of the energy contour, relative to `efermi`, in eV.
    pub emax: f64,
    /// Number of contour nodes: odd and at least 3.
    pub nz: usize,
    /// Optional pair-distance cutoff in angstrom.
    pub rcut: Option<f64>,
    /// Optional spin quantum numbers `S_i` in units of `hbar`.
    pub spin: Option<Array1<f64>>,
    /// Optional reference axes, shape `(n, 3)`, unit rows.
    pub axis: Option<Array2<f64>>,
    /// Optional explicit magnetic sites, as atom indices.
    pub magnetic_atoms: Option<Vec<usize>>,
}

impl ExchangeParams {
    /// Default parameters (`emin = -10`, `emax = 0`, `nz = 65`).
    pub fn new<S: Data<Elem = usize>>(efermi: f64, kmesh: &ArrayBase<S, Ix1>) -> Self {
        Self {
            efermi,
            kmesh: kmesh.to_owned(),
            emin: -10.0,
            emax: 0.0,
            nz: 65,
            rcut: None,
            spin: None,
            axis: None,
            magnetic_atoms: None,
        }
    }

    /// Set the contour window relative to the Fermi energy.
    pub fn with_energy_window(mut self, emin: f64, emax: f64) -> Self {
        self.emin = emin;
        self.emax = emax;
        self
    }

    /// Set the number of contour nodes (odd, at least 3).
    pub fn with_nz(mut self, nz: usize) -> Self {
        self.nz = nz;
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
    /// The two models must describe the same lattice, orbital positions and
    /// translation list; only their hopping matrices may differ.
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
        let position_gap = (&up.orb - &down.orb)
            .mapv(f64::abs)
            .fold(0.0, |acc: f64, value| acc.max(*value));
        if lattice_gap > 1e-10 || position_gap > 1e-10 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "matching_geometry",
                message: "the up and down models must share the same lattice and orbital positions"
                    .into(),
            });
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
        if !params.efermi.is_finite() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_efermi",
                message: "the Fermi energy must be finite".into(),
            });
        }

        let n_orb = up.shape()[1];
        let sites = resolve_sites(atoms, n_orb, params)?;
        let (spin, axis) = resolve_moments(atoms, &sites, params)?;
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

        let contour = semicircle_contour(params.emin, params.emax, params.nz)?;
        let mut accumulator = Array3::<Complex<f64>>::zeros((n_box, n_site, n_site));
        for (energy, weight) in contour.iter() {
            let g_up = green_real_space(ham_r, up, &kpoints, *energy, params.efermi, &r_box)?;
            let g_down = green_real_space(ham_r, down, &kpoints, *energy, params.efermi, &r_box)?;
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
                        && pair_distance(lat, orb, &sites[i], &sites[j], &r_box, i_r) > cutoff
                    {
                        continue;
                    }
                    let alignment: f64 = (0..SPIN_COMPONENTS)
                        .map(|k| axis[[i, k]] * axis[[j, k]])
                        .sum();
                    let sign = if alignment < 0.0 { -1.0 } else { 1.0 };
                    let coupling =
                        accumulator[[i_r, i, j]].im / (4.0 * std::f64::consts::PI) / sign;
                    for component in 0..SPIN_COMPONENTS {
                        ham[[i_r, 3 * i + component, 3 * j + component]] = coupling;
                    }
                }
            }
        }
        Self::new(
            lat.clone(),
            orb.clone(),
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
    orb: &Array2<f64>,
    site_i: &[usize],
    site_j: &[usize],
    r_box: &Array2<isize>,
    i_r: usize,
) -> f64 {
    let dim = lat.nrows();
    let mut fractional = [0.0f64; 3];
    for axis in 0..dim {
        fractional[axis] =
            r_box[[i_r, axis]] as f64 + orb[[site_j[0], axis]] - orb[[site_i[0], axis]];
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

/// Real-space Green's function of one spin channel on the requested box.
fn green_real_space(
    ham_r: &Array2<isize>,
    ham: ArrayView3<Complex<f64>>,
    kpoints: &Array2<f64>,
    energy: Complex<f64>,
    efermi: f64,
    r_box: &Array2<isize>,
) -> Result<Array3<Complex<f64>>> {
    let n = ham.shape()[1];
    let n_box = r_box.nrows();
    let n_k = kpoints.nrows();
    let mut out = Array3::<Complex<f64>>::zeros((n_box, n, n));
    let weight = 1.0 / n_k as f64;
    let mut green = Array2::<Complex<f64>>::zeros((n, n));
    for k in kpoints.axis_iter(Axis(0)) {
        let hk = hamiltonian_at(ham_r, ham, &k);
        let (values, vectors) = eigh_full(&hk, UPLO::Lower)?;
        green.fill(Complex::new(0.0, 0.0));
        for band in 0..n {
            let denominator = energy + efermi - values[band];
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
                    green[[alpha, beta]] += left * vectors[[band, beta]].conj();
                }
            }
        }
        for i_r in 0..n_box {
            let phase = -std::f64::consts::TAU
                * (0..r_box.ncols())
                    .map(|axis| r_box[[i_r, axis]] as f64 * k[axis])
                    .sum::<f64>();
            let factor = Complex::from_polar(1.0, phase) * weight;
            for alpha in 0..n {
                for beta in 0..n {
                    out[[i_r, alpha, beta]] += green[[alpha, beta]] * factor;
                }
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
fn resolve_sites(atoms: &[Atom], n_orb: usize, params: &ExchangeParams) -> Result<Vec<Vec<usize>>> {
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
        return Ok(sites);
    }
    if !atoms.is_empty() {
        let any_moment = atoms.iter().any(|atom| atom.has_magnetic_moment());
        let mut sites = Vec::new();
        for atom in atoms {
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
        }
        return Ok(sites);
    }
    if n_orb == 0 {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "missing_magnetic_sites",
            message: "the model has no orbitals".into(),
        });
    }
    Ok((0..n_orb).map(|index| vec![index]).collect())
}

fn resolve_moments(
    atoms: &[Atom],
    sites: &[Vec<usize>],
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
        return Ok((spin.clone(), axis.clone()));
    }
    if atoms.is_empty() {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "missing_moments",
            message: "an orbital-only model needs explicit ExchangeParams::spin and ::axis".into(),
        });
    }
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
    Ok((spin, axis))
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
        for i in 0..2 {
            model.set_onsite_one(delta[i] / 2.0, i, SpinDirection::Z);
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
        let params = ExchangeParams::new(0.0, &array![5usize]).with_nz(257);
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
        let params = ExchangeParams::new(0.0, &array![5usize]).with_nz(65);
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
    fn test_extracted_magnon_has_a_gapped_antiferromagnetic_spectrum() {
        use crate::magnon::MagnonSolve;
        let model = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        let params = ExchangeParams::new(0.0, &array![5usize]).with_nz(65);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        // Two antiferromagnetic branches, gapless at the magnetic Gamma point.
        let omega = magnon.magnon_band_onek(&array![0.0]).unwrap();
        assert_eq!(omega.len(), 2);
        for value in omega.iter() {
            assert!(
                value.abs() < 1e-6,
                "the antiferromagnetic Goldstone mode is not zero: {value}"
            );
        }
        // Away from Gamma the modes are gapped and real.
        let gapped = magnon.magnon_band_onek(&array![0.5]).unwrap();
        assert!(gapped.iter().all(|value| value.is_finite() && *value > 0.0));
    }

    #[test]
    fn test_extraction_rejects_inconsistent_inputs() {
        let model = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        // Even meshes are rounded up to odd, a wrong dimension is rejected.
        let params = ExchangeParams::new(0.0, &array![4usize]).with_nz(65);
        let magnon = Magnon::<1>::from_model(&model, &params).unwrap();
        assert_eq!(magnon.hamR.nrows(), 5);
        let params = ExchangeParams::new(0.0, &array![4usize, 4]).with_nz(65);
        let error = Magnon::<1>::from_model(&model, &params).unwrap_err();
        assert!(matches!(error, TbError::DimensionMismatch { .. }));
        // A non-collinear model is rejected.
        let mut spinor = spinful_model(0.3, [1.0, -1.0], [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]]);
        spinor.set_hop(0.05, 0, 0, &array![0isize], SpinDirection::X);
        let params = ExchangeParams::new(0.0, &array![3usize]).with_nz(65);
        let error = Magnon::<1>::from_model(&spinor, &params).unwrap_err();
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => {
                assert_eq!(invariant, "collinear_model")
            }
            other => panic!("unexpected error: {other}"),
        }
    }
}
