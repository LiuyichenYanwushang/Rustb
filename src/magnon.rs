//! Magnon (spin-wave) module.
//!
//! This module mirrors the electronic-structure side of the crate: a
//! [`Magnon<DIM>`] owns the real-space exchange tensors of a classical spin
//! Hamiltonian together with the geometry and spin quantum numbers needed to
//! diagonalize it, and the [`io`] submodule writes and reads the "hr" form of
//! that Hamiltonian.
//!
//! # Model and conventions
//!
//! The classical spin Hamiltonian follows TB2J:
//!
//! $$
//! E = -\sum\_{i,j,\mathbf R}
//!     \mathbf s\_i\cdot\mathbf J\_{ij}(\mathbf R)\cdot\mathbf s\_j(\mathbf R),
//! $$
//!
//! where the spin directions are unit vectors and the sum runs over **ordered**
//! pairs. Both `(i, j, R)` and its
//! partner `(j, i, -R)` are stored in [`Magnon::ham`], so one physical bond
//! contributes twice and the effective bond coupling is twice the listed
//! value. $J>0$ favours ferromagnetic alignment and energies are in eV.
//!
//! The single-ion anisotropy is included in the onsite `(R = 0, i = j)`
//! block of `ham`; it is not added again as a separate term.
//!
//! The real-space to reciprocal-space transform is
//!
//! $$
//! \mathbf J(\mathbf q) = \sum\_{\mathbf R}
//! \mathbf J(\mathbf R)\\,e^{-2\pi i\\,\mathbf q\cdot\mathbf R}
//! $$
//!
//! (TB2J's sign convention, the opposite
//! of [`Model::gen_ham`](crate::model::Model::gen_ham)). Reversing the Fourier
//! sign relabels $\mathbf q$ as $-\mathbf q$; a nonreciprocal spectrum need not
//! have the same frequencies at the same numerical momentum.
//!
//! The stored $J$ is TB2J's **unit-vector** coefficient: it is the curvature of
//! the energy landscape with respect to the moment *directions*, so it does not
//! carry a factor $S\_iS\_j$. The spin-wave solver converts the landscape
//! coefficient into the operator exchange
//!
//! $$
//! J^{\mathrm{op}}\_{ij}(\mathbf R) = \frac{J\_{ij}(\mathbf R)}{S\_iS\_j}
//! $$
//!
//! before applying linear spin-wave theory, exactly as TB2J's `Magnon.Jq` does.
//! For a ferromagnetic chain this reproduces
//!
//! $$
//! \omega(\mathbf q) = \frac{2}{S}\lbrack J(\mathbf 0) - J(\mathbf q)\rbrack,
//! $$
//!
//! which for one bond of landscape coupling $j$ per direction is
//! $\omega(q) = 4j\\,\lbrack1-\cos(2\pi q)\rbrack/S$ for fractional momentum $q$:
//! larger moments precess more
//! slowly for a fixed energy landscape.
//!
//! [`Magnon::spin`] holds the spin quantum number $S\_i$ in units of $\hbar$
//! ($S\_i=|m\_i|/2$ for a magnetic moment $m\_i$ in Bohr magnetons, i.e. the
//! $g=2$ orbital-quenched moment), and [`Magnon::axis`] the reference
//! direction of each moment. Those two arrays are what the spin-wave expansion
//! needs: the local frame of site $i$ is an orthonormal triad whose third axis
//! is `axis[i]`.
//!
//! # Storage
//!
//! [`Magnon::ham`] is an `(nR, 3n, 3n)` array laid out exactly like
//! [`Model::ham`](crate::model::Model::ham): row `iR` is the matrix of the cell
//! translation `hamR[iR]`, and the Cartesian components of magnetic site `i`
//! occupy columns `3*i .. 3*i + 3` (the TB2J `i3j3` ordering). The
//! $3\times3$ block of a site pair is
//!
//! $$
//! \mathbf J\_{ij} =
//! J\_{\mathrm{iso}}\\,\mathbb{1}
//! \+ J\_{\mathrm{ani}}^{\mathrm{sym}}
//! \+ D^{\mathrm{anti}},
//! $$
//!
//! and the `(R = 0, i = i)` block additionally carries the single-ion
//! anisotropy.
//!
//! [`Magnon::atoms`] and [`Magnon::atom_of_site`] optionally record the crystal
//! the sites belong to. They are filled by the exchange extraction
//! ([`Model::to_magnon`](crate::model::Model::to_magnon),
//! [`CollinearModel::to_magnon`](crate::model::CollinearModel::to_magnon)) and the
//! TB2J reader, and are empty for metadata-free inputs ([`Magnon::new`] and
//! [`Magnon::from_hr`]). Atom metadata is what symmetry analysis, supercells and
//! atom-resolved output need; the spin-wave solver itself only uses positions,
//! spins and reference axes.
//!
//! # Examples
//!
//! ```rust
//! use ndarray::{array, Array3};
//! use Rustb::magnon::{Magnon, MagnonReference};
//!
//! # fn main() -> Rustb::Result<()> {
//! // One magnetic site, nearest-neighbour exchange along a 1D chain.
//! let lat = array![[1.0]];
//! let orb = array![[0.0]];
//! let spin = array![1.5];
//! let axis = array![[0.0, 0.0, 1.0]];
//! let ham_r = array![[0isize], [1], [-1]];
//!
//! let mut ham = Array3::<f64>::zeros((3, 3, 3));
//! let j0 = 0.01; // eV, ferromagnetic (J > 0)
//! for d in 0..3 {
//!     ham[[1, d, d]] = j0;
//!     ham[[2, d, d]] = j0;
//! }
//!
//! let magnon = Magnon::<1>::new(
//!     lat,
//!     orb,
//!     spin,
//!     axis,
//!     ham,
//!     ham_r,
//!     MagnonReference::collinear(),
//! )?;
//! magnon.validate()?;
//! assert_eq!((magnon.nspin(), magnon.nsta(), magnon.nR()), (1, 3, 3));
//!
//! // The 3x3 block of a site pair is recovered from the packed matrix.
//! let j = magnon.jtensor(0, 0, 1).unwrap();
//! assert!((j[0][0] - j0).abs() < 1e-12);
//! # Ok(())
//! # }
//! ```

pub mod driven;
pub mod exchange;
pub mod floquet;
pub mod io;
pub mod physics;
mod tb2j;

pub use self::exchange::{ExchangeParams, MagnonContour};
pub use self::floquet::{MagnonFloquet, MagnonSambe};

use crate::atom_struct::Atom;
use crate::error::{Result, TbError};
use crate::kpath::Kpath;
use ndarray::{Array1, Array2, Array3, ArrayBase, Data, Ix1, Ix2};
use ndarray_linalg::Inverse;
use std::collections::HashMap;

/// Number of Cartesian spin components stored per magnetic site.
pub const SPIN_COMPONENTS: usize = 3;

/// Largest fractional-coordinate deviation allowed between a magnetic site and
/// its atom, modulo a lattice vector.
const MAGNETIC_SITE_POSITION_TOLERANCE: f64 = 1e-6;

/// Reference magnetic structure of a [`Magnon`] model.
///
/// TB2J keeps three pieces of information beyond the exchange tensors: the
/// single-Q propagation vector $\mathbf q$, the global axis $\mathbf n$ the
/// reference structure rotates about, and the global quantization axis
/// $\mathbf u\_z$ used to build local frames. $\mathbf q=\mathbf 0$ (the
/// default) describes a structure that is already fully contained in the unit
/// cell.
#[derive(Clone, Debug)]
pub struct MagnonReference<const DIM: usize = 3> {
    /// Single-Q propagation vector in reduced reciprocal coordinates.
    pub q: Array1<f64>,
    /// Global rotation axis of the reference structure, a unit 3-vector.
    pub rotation_axis: Array1<f64>,
    /// Global quantization axis used to build the local frames, a unit 3-vector.
    pub quantization_axis: Array1<f64>,
}

impl<const DIM: usize> MagnonReference<DIM> {
    /// Collinear/ferromagnetic reference:
    /// $\mathbf q=\mathbf 0$, $\mathbf n=\hat{\mathbf x}$,
    /// $\mathbf u\_z=\hat{\mathbf z}$.
    ///
    /// TB2J's `Magnon._Q`/`_uz` defaults are matched, but its rotation axis
    /// (`_n`) defaults to $\hat{\mathbf z}$; this crate uses
    /// $\hat{\mathbf x}$, which is the transverse axis of the default
    /// $\hat{\mathbf z}$-polarized moments and therefore the physically
    /// meaningful spiral axis. With $\mathbf q=\mathbf 0$ the choice is inert.
    pub fn collinear() -> Self {
        Self {
            q: Array1::zeros(DIM),
            rotation_axis: Array1::from(vec![1.0, 0.0, 0.0]),
            quantization_axis: Array1::from(vec![0.0, 0.0, 1.0]),
        }
    }

    /// Single-Q spiral reference: propagation vector $\mathbf q$, default axes.
    ///
    /// The exchange tensors are rotated by $2\pi\\,\mathbf q\cdot\mathbf R$
    /// about `rotation_axis` when the BdG matrix is built.
    /// $\mathbf n=\hat{\mathbf x}$ keeps the rotation transverse to the default
    /// $\hat{\mathbf z}$-polarized moments; use
    /// [`Self::with_rotation_axis`] for another spiral axis.
    pub fn single_q<S: Data<Elem = f64>>(q: &ArrayBase<S, Ix1>) -> Result<Self> {
        let reference = Self {
            q: q.to_owned(),
            rotation_axis: Array1::from(vec![1.0, 0.0, 0.0]),
            quantization_axis: Array1::from(vec![0.0, 0.0, 1.0]),
        };
        reference.validate()?;
        Ok(reference)
    }

    /// Replace the global rotation axis of the spiral, checking that it is a
    /// unit vector.
    pub fn with_rotation_axis<S: Data<Elem = f64>>(
        mut self,
        axis: &ArrayBase<S, Ix1>,
    ) -> Result<Self> {
        self.rotation_axis = axis.to_owned();
        self.validate()?;
        Ok(self)
    }

    /// Replace the global quantization axis used to build the local frames.
    pub fn with_quantization_axis<S: Data<Elem = f64>>(
        mut self,
        axis: &ArrayBase<S, Ix1>,
    ) -> Result<Self> {
        self.quantization_axis = axis.to_owned();
        self.validate()?;
        Ok(self)
    }

    /// Validate the propagation vector and the two unit axes.
    pub fn validate(&self) -> Result<()> {
        if self.q.len() != DIM {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "propagation_vector_shape",
                message: format!("expected {DIM} components, found {}", self.q.len()),
            });
        }
        if !self.q.iter().all(|value| value.is_finite()) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_reference",
                message: "the propagation vector must be finite".into(),
            });
        }
        for (name, axis) in [
            ("rotation_axis", &self.rotation_axis),
            ("quantization_axis", &self.quantization_axis),
        ] {
            if axis.len() != SPIN_COMPONENTS {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "reference_axis_shape",
                    message: format!(
                        "{name} needs {SPIN_COMPONENTS} components, found {}",
                        axis.len()
                    ),
                });
            }
            let norm_sq: f64 = axis.iter().map(|x| x * x).sum();
            if !norm_sq.is_finite() || (norm_sq - 1.0).abs() > 1e-6 {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "reference_axis_norm",
                    message: format!("{name} has |n|^2 = {norm_sq}, expected 1"),
                });
            }
        }
        Ok(())
    }
}

/// Real-space spin Hamiltonian of a magnon problem.
///
/// See the [module documentation](self) for the model, the storage layout, the
/// energy conventions and the $J$ versus $J/(S\_iS\_j)$ bookkeeping. Every field
/// is public so that builders and analysis code can work on the arrays
/// directly; [`Magnon::validate`] checks the invariants those consumers rely
/// on, and code that mutates a field must call it again before solving or
/// writing.
#[derive(Clone, Debug)]
pub struct Magnon<const DIM: usize = 3> {
    /// Real-space lattice vectors as rows, shape `(DIM, DIM)`, in angstrom.
    pub lat: Array2<f64>,
    /// Fractional positions of the magnetic sites, shape `(n, DIM)`.
    pub orb: Array2<f64>,
    /// Spin quantum numbers $S\_i$ in units of $\hbar$, shape `(n,)`.
    pub spin: Array1<f64>,
    /// Reference moment direction of each site, shape `(n, 3)`, unit rows.
    pub axis: Array2<f64>,
    /// Exchange tensors $\mathbf J\_{ij}(\mathbf R)$, shape `(nR, 3n, 3n)`,
    /// in eV.
    pub ham: Array3<f64>,
    /// Cell translations $\mathbf R$ as integer multiples of the lattice
    /// rows, shape `(nR, DIM)`. Row `0` must be the zero vector.
    pub hamR: Array2<isize>,
    /// Reference magnetic structure (TB2J's $\mathbf Q$, $\mathbf n$ and
    /// $\mathbf u\_z$).
    pub reference: MagnonReference<DIM>,
    /// Atomic sites of the crystal, empty when the model was built without
    /// atom metadata.
    ///
    /// The exchange tensors only need site positions, but symmetry analysis,
    /// supercells, atom-resolved spectra and couplings to an electronic model
    /// need to know which atom each magnetic site belongs to. Atom metadata is
    /// optional so that standalone hr files and purely magnetic inputs keep
    /// working.
    pub atoms: Vec<Atom>,
    /// Atom index of each magnetic site, row `i` gives the atom of site `i`.
    ///
    /// Empty exactly when [`Magnon::atoms`] is empty; otherwise the mapping is
    /// injective and every `atoms[atom_of_site[i]]` is centred on `orb[i]`
    /// modulo a lattice vector.
    pub atom_of_site: Vec<usize>,
}

impl<const DIM: usize> Magnon<DIM> {
    /// Assemble a magnon model and validate it.
    ///
    /// The result carries no atom metadata; chain [`Magnon::with_atoms`] when
    /// the crystal is known. Exchange extraction and the TB2J reader already do
    /// that.
    pub fn new(
        lat: Array2<f64>,
        orb: Array2<f64>,
        spin: Array1<f64>,
        axis: Array2<f64>,
        ham: Array3<f64>,
        hamR: Array2<isize>,
        reference: MagnonReference<DIM>,
    ) -> Result<Self> {
        let magnon = Self {
            lat,
            orb,
            spin,
            axis,
            ham,
            hamR,
            reference,
            atoms: Vec::new(),
            atom_of_site: Vec::new(),
        };
        magnon.validate()?;
        Ok(magnon)
    }

    /// Attach the crystal's atoms and the site-to-atom mapping.
    ///
    /// `atom_of_site[i]` is the atom of magnetic site `i`. The mapping must be
    /// injective, in range, and every site centre must coincide with its atom
    /// position modulo a lattice vector. Passing empty vectors removes atom
    /// metadata again, which keeps metadata-free hr inputs valid.
    ///
    /// This is the validated way to add atoms: [`Magnon::new`] stays metadata-free so
    /// that hr round trips and hand-built test fixtures keep their old shape.
    pub fn with_atoms(mut self, atoms: Vec<Atom>, atom_of_site: Vec<usize>) -> Result<Self> {
        self.atoms = atoms;
        self.atom_of_site = atom_of_site;
        self.validate()?;
        Ok(self)
    }

    /// Number of stored crystal atoms; 0 when the model carries no atom metadata.
    ///
    /// This counts every atom, including non-magnetic ones, so it can exceed
    /// [`Magnon::nspin`].
    #[inline(always)]
    pub fn natom(&self) -> usize {
        self.atoms.len()
    }

    /// Whether this model carries atom metadata and a site mapping.
    #[inline(always)]
    pub fn has_atoms(&self) -> bool {
        !self.atoms.is_empty()
    }

    /// Atom of one magnetic site.
    ///
    /// Returns a structured error when the model has no atom metadata or the
    /// site index is out of range.
    pub fn atom_at_site(&self, site: usize) -> Result<&Atom> {
        if self.atoms.is_empty() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "missing_atoms",
                message: "this magnon model carries no atom metadata".into(),
            });
        }
        let atom_index = self.atom_of_site.get(site).copied().ok_or_else(|| {
            TbError::InvalidMagnonInvariant {
                invariant: "site_index",
                message: format!("site index {site} must be below nspin = {}", self.nspin()),
            }
        })?;
        self.atoms
            .get(atom_index)
            .ok_or_else(|| TbError::InvalidMagnonInvariant {
                invariant: "atom_index",
                message: format!("atom index {atom_index} is out of range"),
            })
    }

    /// Number of magnetic sites `n` (TB2J calls this `nspin`).
    #[inline(always)]
    pub fn nspin(&self) -> usize {
        self.orb.nrows()
    }

    /// Dimension of the packed exchange matrix, `3 * nspin()`.
    #[inline(always)]
    pub fn nsta(&self) -> usize {
        SPIN_COMPONENTS * self.nspin()
    }

    /// Number of stored cell translations.
    #[inline(always)]
    pub fn nR(&self) -> usize {
        self.hamR.nrows()
    }

    /// Spatial dimension of the model.
    #[inline(always)]
    pub fn dim_r(&self) -> usize {
        DIM
    }

    /// Reciprocal lattice vectors as rows, satisfying
    /// $\mathbf B\mathbf A^{\mathrm T} = 2\pi\mathbb{1}$.
    ///
    /// The formula matches [`Model::rec_lat`](crate::model::Model::rec_lat):
    ///
    /// $$
    /// \mathbf B = 2\pi\\,(\mathbf A^{\mathrm T})^{-1}
    /// $$
    ///
    /// with the real-space lattice $\mathbf A$ stored as rows.
    pub fn rec_lat(&self) -> Result<Array2<f64>> {
        let inv_t = self
            .lat
            .t()
            .to_owned()
            .inv()
            .map_err(|e| TbError::Other(format!("Failed to invert lattice: {e}")))?;
        Ok(std::f64::consts::TAU * inv_t)
    }

    /// The `3 x 3` exchange tensor of sites `(i, j)` in cell translation `iR`.
    ///
    /// Returns the block by value so that callers do not have to reproduce the
    /// `3*i .. 3*i + 3` packing, and so that an out-of-range index is reported
    /// instead of panicking. The guarantee assumes a validated model: fields
    /// edited directly must be re-checked with [`Magnon::validate`] first.
    pub fn jtensor(&self, i: usize, j: usize, iR: usize) -> Result<[[f64; 3]; 3]> {
        let n = self.nspin();
        if i >= n || j >= n {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "site_index",
                message: format!("site indices ({i}, {j}) must be below nspin = {n}"),
            });
        }
        if iR >= self.nR() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "cell_index",
                message: format!("cell index {iR} must be below nR = {}", self.nR()),
            });
        }
        let mut out = [[0.0f64; 3]; 3];
        for (a, row) in out.iter_mut().enumerate() {
            for (b, value) in row.iter_mut().enumerate() {
                *value = self.ham[[iR, SPIN_COMPONENTS * i + a, SPIN_COMPONENTS * j + b]];
            }
        }
        Ok(out)
    }

    /// Validate shapes, finiteness, the atom table and site mapping, the
    /// `R -> -R` pairing and Hermiticity.
    ///
    /// The checks mirror what the spin-wave solver and the hr writer need: a
    /// square lattice with a finite inverse, one positive spin per site, unit
    /// reference axes, at least one translation with the zero vector in row
    /// `0`, no duplicated translations, every $-\mathbf R$ present, and
    ///
    /// $$
    /// J^{ab}\_{ij}(\mathbf R) = J^{ba}\_{ji}(-\mathbf R).
    /// $$
    pub fn validate(&self) -> Result<()> {
        if !(1..=3).contains(&DIM) {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![1, 2, 3],
            });
        }
        let n = self.nspin();
        if n == 0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "magnetic_site_count",
                message: "a magnon model needs at least one magnetic site".into(),
            });
        }
        if self.lat.dim() != (DIM, DIM) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "lattice_shape",
                message: format!("expected ({DIM}, {DIM}), found {:?}", self.lat.dim()),
            });
        }
        if self.orb.dim() != (n, DIM) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "site_position_shape",
                message: format!("expected ({n}, {DIM}), found {:?}", self.orb.dim()),
            });
        }
        if self.spin.len() != n {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "spin_count",
                message: format!("expected {n} spin values, found {}", self.spin.len()),
            });
        }
        if self.axis.dim() != (n, SPIN_COMPONENTS) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "axis_shape",
                message: format!(
                    "expected ({n}, {SPIN_COMPONENTS}), found {:?}",
                    self.axis.dim()
                ),
            });
        }
        if self.atom_of_site.is_empty() {
            if !self.atoms.is_empty() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "atom_site_mapping",
                    message: "atom metadata is present but no site-to-atom mapping was given"
                        .into(),
                });
            }
        } else {
            if self.atoms.is_empty() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "atom_site_mapping",
                    message: "a site-to-atom mapping requires atom metadata".into(),
                });
            }
            if self.atom_of_site.len() != n {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "atom_site_mapping_length",
                    message: format!(
                        "expected {n} site-to-atom entries, found {}",
                        self.atom_of_site.len()
                    ),
                });
            }
            let mut seen = vec![false; self.atoms.len()];
            for (site, &atom_index) in self.atom_of_site.iter().enumerate() {
                let atom = self.atoms.get(atom_index).ok_or_else(|| {
                    TbError::InvalidMagnonInvariant {
                        invariant: "atom_index",
                        message: format!(
                            "site {site} maps to atom {atom_index}, but only {} atoms are stored",
                            self.atoms.len()
                        ),
                    }
                })?;
                if seen[atom_index] {
                    return Err(TbError::InvalidMagnonInvariant {
                        invariant: "atom_site_mapping_injective",
                        message: format!("atom {atom_index} is the parent of more than one site"),
                    });
                }
                seen[atom_index] = true;
                if atom.position_ref().len() != DIM {
                    return Err(TbError::InvalidMagnonInvariant {
                        invariant: "atom_position",
                        message: format!(
                            "atom {atom_index} needs {DIM} coordinates, found {}",
                            atom.position_ref().len()
                        ),
                    });
                }
                for axis in 0..DIM {
                    let delta =
                        (self.orb[[site, axis]] - atom.position_ref()[[axis]]).rem_euclid(1.0);
                    let distance = delta.min(1.0 - delta);
                    if !distance.is_finite() || distance > MAGNETIC_SITE_POSITION_TOLERANCE {
                        return Err(TbError::InvalidMagnonInvariant {
                            invariant: "site_atom_position",
                            message: format!(
                                "site {site} at fractional {:?} is {distance:e} from atom {atom_index} modulo a lattice vector",
                                self.orb.row(site).to_vec()
                            ),
                        });
                    }
                }
            }
        }

        // Every stored atom is part of the crystal, including ones that are not
        // magnetic sites; validate their shape and finiteness here.
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            if atom.position_ref().len() != DIM
                || !atom.position_ref().iter().all(|v| v.is_finite())
            {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "atom_position",
                    message: format!(
                        "atom {atom_index} needs {DIM} finite coordinates, found {:?}",
                        atom.position_ref().to_vec()
                    ),
                });
            }
        }

        if self.hamR.ncols() != DIM {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "translation_shape",
                message: format!("expected {DIM} columns, found {}", self.hamR.ncols()),
            });
        }
        if self.nR() == 0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "translation_count",
                message: "a magnon model needs at least the R = 0 translation".into(),
            });
        }
        if self.ham.dim() != (self.nR(), self.nsta(), self.nsta()) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "exchange_shape",
                message: format!(
                    "expected ({}, {}, {}), found {:?}",
                    self.nR(),
                    self.nsta(),
                    self.nsta(),
                    self.ham.dim()
                ),
            });
        }
        if !self
            .lat
            .iter()
            .chain(self.orb.iter())
            .chain(self.spin.iter())
            .chain(self.axis.iter())
            .chain(self.ham.iter())
            .all(|value| value.is_finite())
        {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_data",
                message: "lattice, positions, spins, axes and exchange tensors must be finite"
                    .into(),
            });
        }
        if self.spin.iter().any(|&s| s <= 0.0) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_spin",
                message: "every spin quantum number must be positive".into(),
            });
        }
        if self.lat.clone().inv().map_or(true, |inverse| {
            inverse.iter().any(|value| !value.is_finite())
        }) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "invertible_lattice",
                message: "the lattice must have a finite inverse".into(),
            });
        }
        for (index, axis) in self.axis.rows().into_iter().enumerate() {
            let norm_sq: f64 = axis.iter().map(|x| x * x).sum();
            if (norm_sq - 1.0).abs() > 1e-6 {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "unit_reference_axis",
                    message: format!("axis {index} has |n|^2 = {norm_sq}, expected 1"),
                });
            }
        }
        self.reference.validate()?;

        let mut index_of: HashMap<Vec<isize>, usize> = HashMap::with_capacity(self.nR());
        for iR in 0..self.nR() {
            let r: Vec<isize> = self.hamR.row(iR).to_vec();
            if index_of.insert(r.clone(), iR).is_some() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "unique_translations",
                    message: format!("translation {r:?} appears more than once"),
                });
            }
        }
        if self.hamR.row(0).iter().any(|&x| x != 0) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "zero_translation_first",
                message: format!(
                    "row 0 of the translation list must be R = 0, found {:?}",
                    self.hamR.row(0).to_vec()
                ),
            });
        }

        let mut neg_index = Vec::with_capacity(self.nR());
        for iR in 0..self.nR() {
            let mut neg: Vec<isize> = Vec::with_capacity(DIM);
            for &value in self.hamR.row(iR) {
                let flipped =
                    value
                        .checked_neg()
                        .ok_or_else(|| TbError::InvalidMagnonInvariant {
                            invariant: "translation_overflow",
                            message: format!(
                                "translation component {value} has no representable negation"
                            ),
                        })?;
                neg.push(flipped);
            }
            match index_of.get(&neg) {
                Some(&i) => neg_index.push(i),
                None => {
                    return Err(TbError::MissingHermitianConjugateExchange {
                        r: Array1::from(neg),
                    });
                }
            }
        }

        let nsta = self.nsta();
        for (iR, &iRm) in neg_index.iter().enumerate() {
            for a in 0..nsta {
                for b in 0..nsta {
                    let left = self.ham[[iR, a, b]];
                    let right = self.ham[[iRm, b, a]];
                    let scale = left.abs().max(right.abs()).max(1.0);
                    if (left - right).abs() > 1e-6 * scale {
                        return Err(TbError::InvalidMagnonInvariant {
                            invariant: "exchange_hermiticity",
                            message: format!(
                                "J[{a}, {b}](R = {:?}) = {left} differs from \
                                 J[{b}, {a}](R = {:?}) = {right}",
                                self.hamR.row(iR).to_vec(),
                                self.hamR.row(iRm).to_vec()
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Solve magnon (spin-wave) frequencies from the real-space exchange tensors.
///
/// The contract mirrors [`Solve`](crate::solve_ham::Solve) on the electronic
/// side: every method validates the model and the input points, returns
/// `nspin()` frequencies per $\mathbf q$-point in non-decreasing order, and
/// preserves the
/// input order. The frequencies are the positive paraunitary (Colpa)
/// eigenvalues of [`Magnon::gen_bdg`], in eV.
pub trait MagnonSolve {
    /// Frequencies at one q-point, shape `(nspin,)`, ascending.
    fn magnon_band_onek<S: Data<Elem = f64>>(&self, q: &ArrayBase<S, Ix1>) -> Result<Array1<f64>>;
    /// Frequencies at every row of `qs`, shape `(nq, nspin)`.
    fn magnon_band_all<S: Data<Elem = f64>>(&self, qs: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>;
    /// Same as [`MagnonSolve::magnon_band_all`], solved in parallel over
    /// $\mathbf q$-points.
    fn magnon_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        qs: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<f64>>;
}

/// Band paths use the same node-preserving sampler as [`Model`](crate::model::Model).
///
/// Distances are reciprocal-lattice units without a $2\pi$ factor, and the
/// path is sampled with `nk` points including every node.
impl<const DIM: usize> Kpath for Magnon<DIM> {
    fn k_path(
        &self,
        path: &Array2<f64>,
        nk: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        crate::kpath::interpolate_k_path(&self.lat, DIM, path, nk)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// One-site ferromagnetic chain with nearest-neighbour exchange `j`.
    fn chain_magnon(j: f64) -> Magnon<1> {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        for d in 0..SPIN_COMPONENTS {
            ham[[1, d, d]] = j;
            ham[[2, d, d]] = j;
        }
        Magnon::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    #[test]
    fn atom_metadata_is_optional_and_validated() {
        use crate::atom_struct::AtomType;

        let base = chain_magnon(0.01);
        assert!(!base.has_atoms());
        assert_eq!((base.natom(), base.atom_of_site.len()), (0, 0));
        assert!(base.atom_at_site(0).is_err());

        let attached = base
            .clone()
            .with_atoms(vec![Atom::new(array![0.0], AtomType::Fe)], vec![0])
            .unwrap();
        assert!(attached.has_atoms());
        assert_eq!(attached.natom(), 1);
        assert_eq!(attached.atom_at_site(0).unwrap().atom_type(), AtomType::Fe);

        // One lattice vector away is the same periodic site.
        base.clone()
            .with_atoms(vec![Atom::new(array![1.0], AtomType::Fe)], vec![0])
            .unwrap();
        // A different position is a different site and must be rejected.
        assert!(
            base.clone()
                .with_atoms(vec![Atom::new(array![0.5], AtomType::Fe)], vec![0])
                .is_err()
        );
        // Mapping without atoms, a wrong length, an out-of-range atom and a
        // duplicated parent all violate the mapping invariant.
        assert!(base.clone().with_atoms(Vec::new(), vec![0]).is_err());
        assert!(
            base.clone()
                .with_atoms(vec![Atom::new(array![0.0], AtomType::Fe)], vec![0, 0])
                .is_err()
        );
        assert!(
            base.clone()
                .with_atoms(vec![Atom::new(array![0.0], AtomType::Fe)], vec![1])
                .is_err()
        );

        let two_site = Magnon::<1>::new(
            array![[1.0]],
            array![[0.0], [0.5]],
            array![1.5, 1.5],
            array![[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            Array3::<f64>::zeros((1, 6, 6)),
            array![[0isize]],
            MagnonReference::collinear(),
        )
        .unwrap();
        let atoms = vec![
            Atom::new(array![0.0], AtomType::Fe),
            Atom::new(array![0.5], AtomType::Fe),
        ];
        two_site
            .clone()
            .with_atoms(atoms.clone(), vec![0, 1])
            .unwrap();
        assert!(two_site.clone().with_atoms(atoms, vec![0, 0]).is_err());
        // In-range but permuted indices are caught by the position check.
        let flipped = vec![
            Atom::new(array![0.0], AtomType::Fe),
            Atom::new(array![0.5], AtomType::Fe),
        ];
        assert!(two_site.clone().with_atoms(flipped, vec![1, 0]).is_err());

        // Every stored atom is checked, including ones that are not sites.
        assert!(
            base.clone()
                .with_atoms(
                    vec![
                        Atom::new(array![0.0], AtomType::Fe),
                        Atom::new(array![f64::NAN], AtomType::Fe),
                    ],
                    vec![0],
                )
                .is_err()
        );
    }

    fn invariant_of(error: &TbError) -> Option<&'static str> {
        match error {
            TbError::InvalidMagnonInvariant { invariant, .. } => Some(invariant),
            _ => None,
        }
    }

    #[test]
    fn test_magnon_accessors_and_jtensor() {
        let magnon = chain_magnon(0.01);
        assert_eq!(magnon.nspin(), 1);
        assert_eq!(magnon.nsta(), 3);
        assert_eq!(magnon.nR(), 3);
        assert_eq!(magnon.dim_r(), 1);
        let j = magnon.jtensor(0, 0, 1).unwrap();
        for (d, row) in j.iter().enumerate() {
            assert!((row[d] - 0.01).abs() < 1e-15);
        }
        assert!(
            magnon
                .jtensor(0, 0, 0)
                .unwrap()
                .iter()
                .flatten()
                .all(|x| *x == 0.0)
        );
    }

    fn check_two_site<const DIM: usize>(magnon: &Magnon<DIM>) {
        assert_eq!(magnon.nsta(), 6);
        assert!((magnon.jtensor(0, 1, 1).unwrap()[0][0] + 0.02).abs() < 1e-15);
        assert!((magnon.jtensor(1, 0, 2).unwrap()[0][0] + 0.02).abs() < 1e-15);
    }

    #[test]
    fn test_magnon_two_site_blocks() {
        let mut ham = Array3::<f64>::zeros((3, 6, 6));
        for i_r in 1..3 {
            for d in 0..3 {
                ham[[i_r, d, 3 + d]] = -0.02;
                ham[[i_r, 3 + d, d]] = -0.02;
            }
        }
        let spin = array![1.5, 1.5];
        let axis = array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];

        let one_d = Magnon::<1>::new(
            array![[1.0]],
            array![[0.0], [0.5]],
            spin.clone(),
            axis.clone(),
            ham.clone(),
            array![[0isize], [1], [-1]],
            MagnonReference::collinear(),
        )
        .unwrap();
        check_two_site(&one_d);

        let three_d = Magnon::<3>::new(
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            spin,
            axis,
            ham,
            array![[0isize, 0, 0], [1, 0, 0], [-1, 0, 0]],
            MagnonReference::collinear(),
        )
        .unwrap();
        check_two_site(&three_d);
    }

    #[test]
    fn test_magnon_rejects_non_unit_axis() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 2.0]];
        let ham_r = array![[0isize]];
        let ham = Array3::<f64>::zeros((1, 3, 3));
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("unit_reference_axis"));
    }

    #[test]
    fn test_magnon_rejects_non_positive_spin() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![0.0];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize]];
        let ham = Array3::<f64>::zeros((1, 3, 3));
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("positive_spin"));
    }

    #[test]
    fn test_magnon_rejects_empty_translation_list() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = Array2::<isize>::zeros((0, 1));
        let ham = Array3::<f64>::zeros((0, 3, 3));
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("translation_count"));
    }

    #[test]
    fn test_magnon_rejects_min_isize_translation() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize], [isize::MIN]];
        let ham = Array3::<f64>::zeros((2, 3, 3));
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("translation_overflow"));
    }

    #[test]
    fn test_magnon_rejects_missing_minus_r() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize], [1]];
        let mut ham = Array3::<f64>::zeros((2, 3, 3));
        for d in 0..3 {
            ham[[1, d, d]] = 0.01;
        }
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            TbError::MissingHermitianConjugateExchange { .. }
        ));
    }

    #[test]
    fn test_magnon_rejects_non_hermitian_exchange() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        ham[[1, 0, 0]] = 0.01;
        ham[[2, 0, 0]] = 0.02;
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("exchange_hermiticity"));
    }

    #[test]
    fn test_magnon_rejects_duplicate_translation() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize], [1], [1]];
        let ham = Array3::<f64>::zeros((3, 3, 3));
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("unique_translations"));
    }

    #[test]
    fn test_magnon_rejects_zero_translation_not_first() {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[1isize], [0], [-1]];
        let ham = Array3::<f64>::zeros((3, 3, 3));
        let error = Magnon::<1>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("zero_translation_first"));
    }

    #[test]
    fn test_magnon_rejects_singular_lattice() {
        let lat = array![[1.0, 0.0], [2.0, 0.0]];
        let error = Magnon::<2>::new(
            lat,
            array![[0.0, 0.0]],
            array![1.0],
            array![[0.0, 0.0, 1.0]],
            Array3::<f64>::zeros((1, 3, 3)),
            Array2::<isize>::zeros((1, 2)),
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert_eq!(invariant_of(&error), Some("invertible_lattice"));
    }

    #[test]
    fn test_reference_rejects_bad_propagation_vector() {
        let mut reference = MagnonReference::<2>::collinear();
        reference.q = array![0.0];
        let error = reference.validate().unwrap_err();
        assert_eq!(invariant_of(&error), Some("propagation_vector_shape"));
    }

    #[test]
    fn test_reference_rejects_bad_axis_norm() {
        let mut reference = MagnonReference::<1>::collinear();
        reference.quantization_axis = array![0.0, 0.0, 2.0];
        let error = reference.validate().unwrap_err();
        assert_eq!(invariant_of(&error), Some("reference_axis_norm"));
    }

    #[test]
    fn test_jtensor_rejects_out_of_range_indices() {
        let magnon = chain_magnon(0.01);
        assert_eq!(
            invariant_of(&magnon.jtensor(1, 0, 0).unwrap_err()),
            Some("site_index")
        );
        assert_eq!(
            invariant_of(&magnon.jtensor(0, 0, 99).unwrap_err()),
            Some("cell_index")
        );
    }

    #[test]
    fn test_rec_lat_matches_two_pi_inverse_transpose() {
        let magnon = Magnon::<2>::new(
            array![[1.0, 0.0], [0.5, 1.0]],
            array![[0.0, 0.0]],
            array![1.0],
            array![[0.0, 0.0, 1.0]],
            Array3::<f64>::zeros((1, 3, 3)),
            Array2::<isize>::zeros((1, 2)),
            MagnonReference::collinear(),
        )
        .unwrap();
        let rec = magnon.rec_lat().unwrap();
        let identity = rec.dot(&magnon.lat.t()) / std::f64::consts::TAU;
        assert!((identity - Array2::<f64>::eye(2)).mapv(f64::abs).sum() < 1e-12);
    }
}
