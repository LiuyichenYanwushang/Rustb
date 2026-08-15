//! Non-collinear unrestricted Hartree-Fock solver for on-site Hubbard interactions.
//!
//! [`HubbardModel`] combines a spinful tight-binding [`Model`] with an
//! orbital-resolved interaction
//!
//! ```math
//! H_U = \sum_i U_i n_{i\uparrow}n_{i\downarrow}.
//! ```
//!
//! For each orbital, define the local spin-density matrix
//!
//! ```math
//! \rho_{i,\alpha\beta}
//! =
//! \langle c^\dagger_{i\beta}c_{i\alpha}\rangle.
//! ```
//!
//! The unrestricted Hartree-Fock potential is then
//!
//! ```math
//! V_i^{\rm HF}
//! =
//! U_i
//! \begin{pmatrix}
//! \rho_{i,\downarrow\downarrow} & -\rho_{i,\uparrow\downarrow}\\
//! -\rho_{i,\downarrow\uparrow} & \rho_{i,\uparrow\uparrow}
//! \end{pmatrix}
//! =
//! U_i\left(\frac{n_i}{2}\sigma_0-\mathbf m_i\cdot\boldsymbol\sigma\right).
//! ```
//!
//! Thus both the Hartree density terms and the Fock spin-flip terms are updated
//! self-consistently. The constant double-counting contribution is omitted
//! because it does not enter the single-particle Hamiltonian. The solver
//! returns an ordinary
//! `Model<true, DIM, R>`, so all existing band, geometry, and response APIs can
//! be used without a Hubbard-specific wrapper. Before returning, the converged
//! chemical potential is subtracted from every on-site state, placing the Fermi
//! level at zero energy.
//!
//! # Example
//!
//! ```no_run
//! use ndarray::array;
//! use Rustb::{
//!     HubbardModel, InitialMagnetization, MeanFieldConstraint, MeanFieldParams, Model,
//!     Occupation, Result,
//! };
//!
//! # fn main() -> Result<()> {
//! let mut bare = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
//! bare.add_hop(-1.0, 0, 0, &array![1], None);
//!
//! let hubbard = HubbardModel::with_uniform_u(bare, 2.0)?;
//! let mut params = MeanFieldParams::new(
//!     [200],
//!     MeanFieldConstraint::FixedInitialFilling { reference_mu: 0.0 },
//!     Occupation::FermiSmearing { width: 0.01 },
//! );
//! params.initial_magnetization = InitialMagnetization::UniformVector {
//!     moment_per_orbital: [1e-3, 0.0, 0.0],
//! };
//!
//! // The returned value is an ordinary spinful Model with mu shifted to zero.
//! let model = hubbard.solve_hartree_fock(&params)?;
//! let moment = model.spin_moment(&[200], 0.0, params.occupation)?;
//! # let _ = moment;
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TbError};
use crate::model::{Gauge, Model, NoRMatrix, RMatrixData};
use crate::model_utils::find_R;
use crate::thermodynamics::{Occupation, fermi_from_width};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3, ArrayBase, Axis, Data, Ix1};
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;

const ENERGY_EQUAL_TOLERANCE: f64 = 1e-12;
const FILLING_TOLERANCE: f64 = 1e-12;
const MAX_CHEMICAL_POTENTIAL_ITERATIONS: usize = 256;

/// Thermodynamic constraint used during the mean-field iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MeanFieldConstraint {
    /// Keep the supplied chemical potential fixed.
    ///
    /// The particle number is allowed to change as the mean-field potential
    /// evolves.
    FixedChemicalPotential {
        /// Chemical potential in the same energy unit as the tight-binding
        /// Hamiltonian.
        mu: f64,
    },

    /// Preserve the filling of the bare model at `reference_mu`.
    ///
    /// The target filling is evaluated once from the bare model using the same
    /// k-mesh and occupation method as the self-consistency calculation. A new
    /// chemical potential is then found at every iteration.
    FixedInitialFilling {
        /// Reference chemical potential of the bare model.
        reference_mu: f64,
    },
}

impl MeanFieldConstraint {
    fn reference_mu(self) -> f64 {
        match self {
            Self::FixedChemicalPotential { mu } => mu,
            Self::FixedInitialFilling { reference_mu } => reference_mu,
        }
    }
}

/// Initial magnetic configuration.
///
/// Magnetic vectors use the dimensionless polarization
/// `p_i = 2⟨S_i⟩/ℏ`. Consequently, the scalar collinear variants specify
/// `p_z = n_up - n_down`.
#[derive(Clone, Debug, PartialEq)]
pub enum InitialMagnetization {
    /// Use the complete local spin-density matrix of the bare model.
    FromBareModel,

    /// Remove the initial spin polarization while preserving each orbital's
    /// total charge.
    Paramagnetic,

    /// Set the same occupation difference on every orbital.
    Ferromagnetic { moment_per_orbital: f64 },

    /// Alternate the sign of the occupation difference by orbital index.
    ///
    /// For multi-orbital atoms, [`InitialMagnetization::Custom`] is generally
    /// preferable because it can assign the same sign to all orbitals on an
    /// atomic site.
    Antiferromagnetic { moment_per_orbital: f64 },

    /// Set an explicit occupation difference for every orbital.
    Custom(Array1<f64>),

    /// Set the same non-collinear polarization vector on every orbital.
    UniformVector {
        /// `[p_x, p_y, p_z] = 2⟨S⟩/ℏ` for each orbital.
        moment_per_orbital: [f64; 3],
    },

    /// Set one non-collinear polarization vector per orbital.
    ///
    /// The array must have shape `(norb, 3)`, with columns ordered as
    /// `[p_x, p_y, p_z]`.
    CustomVectors(Array2<f64>),
}

/// Parameters controlling a Hubbard mean-field calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct MeanFieldParams<const DIM: usize = 3> {
    /// Uniform reciprocal-space mesh.
    pub k_mesh: [usize; DIM],
    /// Fixed chemical potential or fixed initial filling.
    pub constraint: MeanFieldConstraint,
    /// Occupation function used both for the initial filling and every
    /// self-consistency iteration.
    pub occupation: Occupation,
    /// Maximum number of density iterations.
    pub max_iterations: usize,
    /// Maximum absolute change in any local density-matrix element.
    pub density_tolerance: f64,
    /// Linear mixing coefficient in `(0, 1]`.
    pub mixing: f64,
    /// Initial magnetic seed.
    pub initial_magnetization: InitialMagnetization,
}

impl<const DIM: usize> MeanFieldParams<DIM> {
    /// Construct mean-field parameters with conservative iteration defaults.
    ///
    /// The defaults are 200 iterations, density tolerance `1e-9`, mixing
    /// `0.2`, and a density initialized from the bare model.
    pub fn new(
        k_mesh: [usize; DIM],
        constraint: MeanFieldConstraint,
        occupation: Occupation,
    ) -> Self {
        Self {
            k_mesh,
            constraint,
            occupation,
            max_iterations: 200,
            density_tolerance: 1e-9,
            mixing: 0.2,
            initial_magnetization: InitialMagnetization::FromBareModel,
        }
    }

    fn validate(&self) -> Result<()> {
        validate_k_mesh(&self.k_mesh)?;
        self.occupation.energy_width()?;

        if !self.constraint.reference_mu().is_finite() {
            return Err(invalid_mean_field_parameter(
                "chemical potential",
                "must be finite",
            ));
        }
        if self.max_iterations == 0 {
            return Err(invalid_mean_field_parameter(
                "max_iterations",
                "must be greater than zero",
            ));
        }
        if !self.density_tolerance.is_finite() || self.density_tolerance <= 0.0 {
            return Err(invalid_mean_field_parameter(
                "density_tolerance",
                "must be finite and positive",
            ));
        }
        if !self.mixing.is_finite() || self.mixing <= 0.0 || self.mixing > 1.0 {
            return Err(invalid_mean_field_parameter(
                "mixing",
                "must be finite and lie in (0, 1]",
            ));
        }
        Ok(())
    }
}

/// A spinful tight-binding model augmented by orbital-resolved on-site Hubbard
/// interactions.
#[derive(Clone, Debug)]
pub struct HubbardModel<const DIM: usize = 3, R: RMatrixData = NoRMatrix> {
    bare_model: Model<true, DIM, R>,
    onsite_u: Array1<f64>,
}

impl<const DIM: usize, R: RMatrixData> HubbardModel<DIM, R> {
    /// Construct a Hubbard model with one `U_i` for every physical orbital.
    pub fn new(bare_model: Model<true, DIM, R>, onsite_u: Array1<f64>) -> Result<Self> {
        if onsite_u.len() != bare_model.norb() {
            return Err(TbError::DimensionMismatch {
                context: "orbital-resolved Hubbard U".to_string(),
                expected: bare_model.norb(),
                found: onsite_u.len(),
            });
        }
        if onsite_u.iter().any(|u| !u.is_finite()) {
            return Err(invalid_mean_field_parameter(
                "onsite_u",
                "all Hubbard interactions must be finite",
            ));
        }
        Ok(Self {
            bare_model,
            onsite_u,
        })
    }

    /// Construct a Hubbard model with the same interaction on every orbital.
    pub fn with_uniform_u(bare_model: Model<true, DIM, R>, onsite_u: f64) -> Result<Self> {
        if !onsite_u.is_finite() {
            return Err(invalid_mean_field_parameter(
                "onsite_u",
                "the Hubbard interaction must be finite",
            ));
        }
        let norb = bare_model.norb();
        Self::new(bare_model, Array1::from_elem(norb, onsite_u))
    }

    /// Borrow the non-interacting tight-binding model.
    pub fn bare_model(&self) -> &Model<true, DIM, R> {
        &self.bare_model
    }

    /// Consume the Hubbard wrapper and return the non-interacting model.
    pub fn into_bare_model(self) -> Model<true, DIM, R> {
        self.bare_model
    }

    /// Return the orbital-resolved Hubbard interactions.
    pub fn onsite_u(&self) -> &Array1<f64> {
        &self.onsite_u
    }

    /// Solve the non-collinear unrestricted Hartree-Fock equations.
    ///
    /// The returned model is the converged effective single-particle
    /// Hamiltonian with its final chemical potential subtracted:
    ///
    /// ```math
    /// H_{\rm out}(k) = H_{\rm MF}(k) - \mu_{\rm final} I.
    /// ```
    ///
    /// Consequently, all occupation-dependent calculations performed on the
    /// returned model should use chemical potential zero.
    ///
    /// # Errors
    ///
    /// Returns [`TbError::MeanFieldNotConverged`] when the density residual
    /// remains above the requested tolerance after `max_iterations`.
    pub fn solve_hartree_fock(&self, params: &MeanFieldParams<DIM>) -> Result<Model<true, DIM, R>> {
        params.validate()?;
        let k_points = uniform_k_mesh(&params.k_mesh)?;
        let width = params.occupation.energy_width()?;

        // The initial filling and density are defined from exactly the same
        // mesh and occupation function that will be used by the SCF loop.
        let bare_spectra = diagonalize_model(&self.bare_model, &k_points)?;
        let reference_mu = params.constraint.reference_mu();
        let initial_occupations = occupations_at_mu(&bare_spectra, reference_mu, width);
        let target_filling = match params.constraint {
            MeanFieldConstraint::FixedChemicalPotential { .. } => None,
            MeanFieldConstraint::FixedInitialFilling { .. } => {
                Some(filling_from_occupations(&initial_occupations))
            }
        };
        let mut density =
            local_density_matrix(&bare_spectra, &initial_occupations, self.bare_model.norb());
        apply_initial_magnetization(&mut density, &params.initial_magnetization)?;

        let mut last_residual = f64::INFINITY;
        for _iteration in 0..params.max_iterations {
            let effective_model = self.build_effective_model(&density)?;
            let spectra = diagonalize_model(&effective_model, &k_points)?;
            let (_, occupations) =
                occupations_for_constraint(&spectra, params.constraint, target_filling, width)?;
            let new_density = local_density_matrix(&spectra, &occupations, self.bare_model.norb());

            last_residual = density_residual(&density, &new_density);
            if last_residual <= params.density_tolerance {
                return self.final_model(
                    &new_density,
                    &k_points,
                    params.constraint,
                    target_filling,
                    width,
                );
            }

            mix_density(&mut density, &new_density, params.mixing);
        }

        Err(TbError::MeanFieldNotConverged {
            iterations: params.max_iterations,
            residual: last_residual,
        })
    }

    fn build_effective_model(&self, density: &Array3<Complex<f64>>) -> Result<Model<true, DIM, R>> {
        let norb = self.bare_model.norb();
        if density.shape() != [norb, 2, 2] {
            return Err(TbError::InvalidArrayShape {
                expected: vec![norb, 2, 2],
                found: density.shape().to_vec(),
            });
        }

        let mut model = self.bare_model.clone();
        let zero_index = zero_hopping_index(&model)?;
        for orbital in 0..norb {
            let up = orbital;
            let down = orbital + norb;
            let onsite_u = self.onsite_u[orbital];

            // V_HF = U [[rho_dd, -rho_ud], [-rho_du, rho_uu]].
            model.ham[[zero_index, up, up]] +=
                Complex::new(onsite_u * density[[orbital, 1, 1]].re, 0.0);
            model.ham[[zero_index, down, down]] +=
                Complex::new(onsite_u * density[[orbital, 0, 0]].re, 0.0);
            model.ham[[zero_index, up, down]] -= onsite_u * density[[orbital, 0, 1]];
            model.ham[[zero_index, down, up]] -= onsite_u * density[[orbital, 1, 0]];
        }
        Ok(model)
    }

    fn final_model(
        &self,
        density: &Array3<Complex<f64>>,
        k_points: &Array2<f64>,
        constraint: MeanFieldConstraint,
        target_filling: Option<f64>,
        width: f64,
    ) -> Result<Model<true, DIM, R>> {
        let mut model = self.build_effective_model(density)?;
        let spectra = diagonalize_model(&model, k_points)?;
        let (chemical_potential, _) =
            occupations_for_constraint(&spectra, constraint, target_filling, width)?;
        shift_energy_origin(&mut model, chemical_potential)?;
        Ok(model)
    }
}

/// Additional filling utilities available on every tight-binding model.
impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Calculate the number of electrons per unit cell at a chemical potential.
    ///
    /// This performs a direct weighted sum over band occupations, rather than
    /// integrating a broadened DOS. Spinful models already contain both spin
    /// sectors, so no additional factor of two is applied.
    pub fn electron_filling(
        &self,
        k_mesh: &[usize; DIM],
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<f64> {
        if !chemical_potential.is_finite() {
            return Err(invalid_mean_field_parameter(
                "chemical_potential",
                "must be finite",
            ));
        }
        let width = occupation.energy_width()?;
        let k_points = uniform_k_mesh(k_mesh)?;
        let spectra = diagonalize_model(self, &k_points)?;
        let occupations = occupations_at_mu(&spectra, chemical_potential, width);
        Ok(filling_from_occupations(&occupations))
    }
}

/// Spin observables available on spinful tight-binding models.
impl<const DIM: usize, R: RMatrixData> Model<true, DIM, R> {
    /// Return `⟨Sx⟩`, `⟨Sy⟩`, and `⟨Sz⟩` for every band at one k-point.
    ///
    /// The result has shape `(nsta, 3)` and is expressed in units of `ℏ`.
    /// Individual values inside an exactly degenerate band subspace are
    /// gauge-dependent; the trace over the degenerate subspace is invariant.
    pub fn spin_expectation_onek<S: Data<Elem = f64>>(
        &self,
        k: &ArrayBase<S, Ix1>,
    ) -> Result<Array2<f64>> {
        if k.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: k.len(),
            });
        }
        let (_, mut eigenvectors) = self.gen_ham(k, Gauge::Atom).eigh(UPLO::Lower)?;
        // ndarray-linalg returns the LAPACK vectors conjugated relative to the
        // basis convention used by Model; this matches solve_ham::solve_onek.
        eigenvectors.mapv_inplace(|value| value.conj());
        Ok(band_spin_expectations(&eigenvectors, self.norb()))
    }

    /// Calculate the orbital-resolved spin moment over a uniform k-mesh.
    ///
    /// The returned array has shape `(norb, 3)` and contains
    /// `⟨Sx⟩`, `⟨Sy⟩`, and `⟨Sz⟩` in units of `ℏ`. For a model returned by
    /// [`HubbardModel::solve_hartree_fock`], use `chemical_potential = 0`.
    pub fn local_spin_moment(
        &self,
        k_mesh: &[usize; DIM],
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<Array2<f64>> {
        if !chemical_potential.is_finite() {
            return Err(invalid_mean_field_parameter(
                "chemical_potential",
                "must be finite",
            ));
        }
        let width = occupation.energy_width()?;
        let k_points = uniform_k_mesh(k_mesh)?;
        let spectra = diagonalize_model(self, &k_points)?;
        let occupations = occupations_at_mu(&spectra, chemical_potential, width);
        Ok(local_spin_moment_from_spectra(
            &spectra,
            &occupations,
            self.norb(),
        ))
    }

    /// Calculate the total spin moment per unit cell.
    ///
    /// The three components are returned in units of `ℏ`.
    pub fn spin_moment(
        &self,
        k_mesh: &[usize; DIM],
        chemical_potential: f64,
        occupation: Occupation,
    ) -> Result<Array1<f64>> {
        Ok(self
            .local_spin_moment(k_mesh, chemical_potential, occupation)?
            .sum_axis(Axis(0)))
    }
}

#[derive(Debug)]
struct KPointSpectrum {
    energies: Array1<f64>,
    eigenvectors: Array2<Complex<f64>>,
}

fn invalid_mean_field_parameter(parameter: &'static str, message: impl Into<String>) -> TbError {
    TbError::InvalidMeanFieldParameter {
        parameter,
        message: message.into(),
    }
}

fn validate_k_mesh<const DIM: usize>(k_mesh: &[usize; DIM]) -> Result<()> {
    if !(1..=3).contains(&DIM) {
        return Err(TbError::InvalidDimension {
            dim: DIM,
            supported: vec![1, 2, 3],
        });
    }
    if k_mesh.contains(&0) {
        return Err(TbError::InvalidKmeshDimensions(Array1::from_vec(
            k_mesh.to_vec(),
        )));
    }
    Ok(())
}

fn uniform_k_mesh<const DIM: usize>(k_mesh: &[usize; DIM]) -> Result<Array2<f64>> {
    validate_k_mesh(k_mesh)?;
    let number_of_k_points = k_mesh.iter().try_fold(1usize, |product, &count| {
        product
            .checked_mul(count)
            .ok_or_else(|| TbError::InvalidKmeshDimensions(Array1::from_vec(k_mesh.to_vec())))
    })?;
    let mut points = Array2::<f64>::zeros((number_of_k_points, DIM));
    for linear_index in 0..number_of_k_points {
        let mut remainder = linear_index;
        for direction in (0..DIM).rev() {
            let count = k_mesh[direction];
            points[[linear_index, direction]] = (remainder % count) as f64 / count as f64;
            remainder /= count;
        }
    }
    Ok(points)
}

fn diagonalize_model<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    k_points: &Array2<f64>,
) -> Result<Vec<KPointSpectrum>> {
    k_points
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|k| {
            let (energies, mut eigenvectors) = model.gen_ham(&k, Gauge::Atom).eigh(UPLO::Lower)?;
            // Keep the same eigenvector convention as Model::solve_onek.
            eigenvectors.mapv_inplace(|value| value.conj());
            Ok(KPointSpectrum {
                energies,
                eigenvectors,
            })
        })
        .collect()
}

#[inline]
fn fermi_occupation(energy: f64, chemical_potential: f64, width: f64) -> f64 {
    if width == 0.0 {
        let delta = energy - chemical_potential;
        let tolerance = ENERGY_EQUAL_TOLERANCE * (1.0 + energy.abs() + chemical_potential.abs());
        if delta < -tolerance {
            1.0
        } else if delta > tolerance {
            0.0
        } else {
            0.5
        }
    } else {
        fermi_from_width(energy, chemical_potential, width)
    }
}

fn occupations_at_mu(
    spectra: &[KPointSpectrum],
    chemical_potential: f64,
    width: f64,
) -> Vec<Array1<f64>> {
    spectra
        .iter()
        .map(|spectrum| {
            spectrum
                .energies
                .mapv(|energy| fermi_occupation(energy, chemical_potential, width))
        })
        .collect()
}

fn filling_from_occupations(occupations: &[Array1<f64>]) -> f64 {
    if occupations.is_empty() {
        return 0.0;
    }
    occupations
        .iter()
        .map(|occupation| occupation.sum())
        .sum::<f64>()
        / occupations.len() as f64
}

fn filling_at_mu(spectra: &[KPointSpectrum], chemical_potential: f64, width: f64) -> f64 {
    if spectra.is_empty() {
        return 0.0;
    }
    spectra
        .iter()
        .flat_map(|spectrum| spectrum.energies.iter())
        .map(|&energy| fermi_occupation(energy, chemical_potential, width))
        .sum::<f64>()
        / spectra.len() as f64
}

fn occupations_for_constraint(
    spectra: &[KPointSpectrum],
    constraint: MeanFieldConstraint,
    target_filling: Option<f64>,
    width: f64,
) -> Result<(f64, Vec<Array1<f64>>)> {
    match constraint {
        MeanFieldConstraint::FixedChemicalPotential { mu } => {
            Ok((mu, occupations_at_mu(spectra, mu, width)))
        }
        MeanFieldConstraint::FixedInitialFilling { .. } => {
            let target_filling = target_filling.ok_or_else(|| {
                invalid_mean_field_parameter(
                    "target_filling",
                    "fixed-filling calculation is missing its initial filling",
                )
            })?;
            if width == 0.0 {
                zero_temperature_fixed_filling(spectra, target_filling)
            } else {
                let mu = chemical_potential_for_filling(spectra, target_filling, width)?;
                Ok((mu, occupations_at_mu(spectra, mu, width)))
            }
        }
    }
}

fn chemical_potential_for_filling(
    spectra: &[KPointSpectrum],
    target_filling: f64,
    width: f64,
) -> Result<f64> {
    let Some(first) = spectra.first() else {
        return Err(invalid_mean_field_parameter(
            "k_mesh",
            "must contain at least one k-point",
        ));
    };
    let nsta = first.energies.len() as f64;
    if !target_filling.is_finite()
        || target_filling < -FILLING_TOLERANCE
        || target_filling > nsta + FILLING_TOLERANCE
    {
        return Err(invalid_mean_field_parameter(
            "target_filling",
            format!("must lie between 0 and {nsta}"),
        ));
    }

    let (minimum, maximum) = energy_bounds(spectra)?;
    let span = (maximum - minimum).abs();
    let margin = (64.0 * width).max(span).max(1.0);
    if target_filling <= FILLING_TOLERANCE {
        return Ok(minimum - margin);
    }
    if target_filling >= nsta - FILLING_TOLERANCE {
        return Ok(maximum + margin);
    }

    let mut lower = minimum - margin;
    let mut upper = maximum + margin;
    for _ in 0..MAX_CHEMICAL_POTENTIAL_ITERATIONS {
        let midpoint = 0.5 * (lower + upper);
        let filling = filling_at_mu(spectra, midpoint, width);
        if filling < target_filling {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
        let scale = 1.0 + midpoint.abs();
        if upper - lower <= ENERGY_EQUAL_TOLERANCE * scale {
            break;
        }
    }
    Ok(0.5 * (lower + upper))
}

fn zero_temperature_fixed_filling(
    spectra: &[KPointSpectrum],
    target_filling: f64,
) -> Result<(f64, Vec<Array1<f64>>)> {
    let Some(first) = spectra.first() else {
        return Err(invalid_mean_field_parameter(
            "k_mesh",
            "must contain at least one k-point",
        ));
    };
    let nsta = first.energies.len() as f64;
    if !target_filling.is_finite()
        || target_filling < -FILLING_TOLERANCE
        || target_filling > nsta + FILLING_TOLERANCE
    {
        return Err(invalid_mean_field_parameter(
            "target_filling",
            format!("must lie between 0 and {nsta}"),
        ));
    }

    let nk = spectra.len();
    let mut states = Vec::with_capacity(nk * first.energies.len());
    for (ik, spectrum) in spectra.iter().enumerate() {
        for (iband, &energy) in spectrum.energies.iter().enumerate() {
            states.push((energy, ik, iband));
        }
    }
    states.sort_by(|left, right| left.0.total_cmp(&right.0));

    let mut occupations: Vec<Array1<f64>> = spectra
        .iter()
        .map(|spectrum| Array1::zeros(spectrum.energies.len()))
        .collect();
    let state_weight = 1.0 / nk as f64;
    let mut filled = 0.0;
    let mut previous_energy: Option<f64> = None;
    let mut index = 0;

    while index < states.len() {
        let group_energy = states[index].0;
        let tolerance = ENERGY_EQUAL_TOLERANCE * (1.0 + group_energy.abs());
        let mut group_end = index + 1;
        while group_end < states.len() && (states[group_end].0 - group_energy).abs() <= tolerance {
            group_end += 1;
        }
        let group_capacity = (group_end - index) as f64 * state_weight;
        let remaining = target_filling - filled;

        if remaining <= FILLING_TOLERANCE {
            let mu = previous_energy
                .map(|energy| 0.5 * (energy + group_energy))
                .unwrap_or(group_energy - 1.0);
            return Ok((mu, occupations));
        }

        if remaining >= group_capacity - FILLING_TOLERANCE {
            for &(_, ik, iband) in &states[index..group_end] {
                occupations[ik][iband] = 1.0;
            }
            filled += group_capacity;
            previous_energy = Some(group_energy);
            index = group_end;
            continue;
        }

        // A finite k-mesh can require fractional occupation of the Fermi-level
        // shell. Equal occupation of all degenerate states preserves symmetry.
        let fraction = (remaining / group_capacity).clamp(0.0, 1.0);
        for &(_, ik, iband) in &states[index..group_end] {
            occupations[ik][iband] = fraction;
        }
        return Ok((group_energy, occupations));
    }

    let maximum = states.last().map(|state| state.0).unwrap_or(0.0);
    Ok((maximum + 1.0, occupations))
}

fn energy_bounds(spectra: &[KPointSpectrum]) -> Result<(f64, f64)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for &energy in spectra.iter().flat_map(|spectrum| spectrum.energies.iter()) {
        minimum = minimum.min(energy);
        maximum = maximum.max(energy);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(TbError::EigenvalueComputationFailed);
    }
    Ok((minimum, maximum))
}

fn local_density_matrix(
    spectra: &[KPointSpectrum],
    occupations: &[Array1<f64>],
    norb: usize,
) -> Array3<Complex<f64>> {
    let mut density = Array3::<Complex<f64>>::zeros((norb, 2, 2));
    let weight = 1.0 / spectra.len() as f64;

    for (spectrum, occupation) in spectra.iter().zip(occupations) {
        for (band, &f) in occupation.iter().enumerate() {
            if f == 0.0 {
                continue;
            }
            for orbital in 0..norb {
                let up = spectrum.eigenvectors[[orbital, band]];
                let down = spectrum.eigenvectors[[orbital + norb, band]];
                let prefactor = weight * f;
                density[[orbital, 0, 0]] += Complex::new(prefactor * up.norm_sqr(), 0.0);
                density[[orbital, 1, 1]] += Complex::new(prefactor * down.norm_sqr(), 0.0);
                density[[orbital, 0, 1]] += prefactor * up * down.conj();
            }
        }
    }
    for orbital in 0..norb {
        density[[orbital, 1, 0]] = density[[orbital, 0, 1]].conj();
    }
    density
}

fn apply_initial_magnetization(
    density: &mut Array3<Complex<f64>>,
    initial: &InitialMagnetization,
) -> Result<()> {
    let norb = density.len_of(Axis(0));
    let moments = match initial {
        InitialMagnetization::FromBareModel => return Ok(()),
        InitialMagnetization::Paramagnetic => Array2::zeros((norb, 3)),
        InitialMagnetization::Ferromagnetic { moment_per_orbital } => {
            let mut moments = Array2::zeros((norb, 3));
            moments.column_mut(2).fill(*moment_per_orbital);
            moments
        }
        InitialMagnetization::Antiferromagnetic { moment_per_orbital } => {
            let mut moments = Array2::zeros((norb, 3));
            for orbital in 0..norb {
                moments[[orbital, 2]] = if orbital % 2 == 0 {
                    *moment_per_orbital
                } else {
                    -*moment_per_orbital
                };
            }
            moments
        }
        InitialMagnetization::Custom(moments) => {
            if moments.len() != norb {
                return Err(TbError::DimensionMismatch {
                    context: "initial orbital magnetization".to_string(),
                    expected: norb,
                    found: moments.len(),
                });
            }
            let mut vectors = Array2::zeros((norb, 3));
            vectors.column_mut(2).assign(moments);
            vectors
        }
        InitialMagnetization::UniformVector { moment_per_orbital } => {
            Array2::from_shape_fn((norb, 3), |(_, direction)| moment_per_orbital[direction])
        }
        InitialMagnetization::CustomVectors(moments) => {
            if moments.shape() != [norb, 3] {
                return Err(TbError::InvalidArrayShape {
                    expected: vec![norb, 3],
                    found: moments.shape().to_vec(),
                });
            }
            moments.clone()
        }
    };

    for orbital in 0..norb {
        let charge = density[[orbital, 0, 0]].re + density[[orbital, 1, 1]].re;
        let moment = moments.row(orbital);
        if moment.iter().any(|value| !value.is_finite()) {
            return Err(invalid_mean_field_parameter(
                "initial_magnetization",
                "all components of every orbital moment must be finite",
            ));
        }
        let magnitude = moment.dot(&moment).sqrt();
        let maximum_moment = charge.min(2.0 - charge).max(0.0);
        if magnitude > maximum_moment + FILLING_TOLERANCE {
            return Err(invalid_mean_field_parameter(
                "initial_magnetization",
                format!(
                    "orbital {orbital} has charge {charge}, so |moment| cannot exceed {maximum_moment}"
                ),
            ));
        }
        let px = moment[0];
        let py = moment[1];
        let pz = moment[2];
        density[[orbital, 0, 0]] = Complex::new(0.5 * (charge + pz), 0.0);
        density[[orbital, 1, 1]] = Complex::new(0.5 * (charge - pz), 0.0);
        density[[orbital, 0, 1]] = Complex::new(0.5 * px, -0.5 * py);
        density[[orbital, 1, 0]] = density[[orbital, 0, 1]].conj();
    }
    Ok(())
}

fn density_residual(old_density: &Array3<Complex<f64>>, new_density: &Array3<Complex<f64>>) -> f64 {
    old_density
        .iter()
        .zip(new_density)
        .map(|(&old, &new)| (new - old).norm())
        .fold(0.0, f64::max)
}

fn mix_density(
    density: &mut Array3<Complex<f64>>,
    new_density: &Array3<Complex<f64>>,
    mixing: f64,
) {
    density
        .iter_mut()
        .zip(new_density)
        .for_each(|(old, &new)| *old = (1.0 - mixing) * *old + mixing * new);
}

fn zero_hopping_index<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
) -> Result<usize> {
    let zero = Array1::<isize>::zeros(DIM);
    find_R(&model.hamR, &zero).ok_or_else(|| {
        TbError::Other("model does not contain the required R=0 Hamiltonian block".to_string())
    })
}

fn shift_energy_origin<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &mut Model<SPIN, DIM, R>,
    chemical_potential: f64,
) -> Result<()> {
    let zero_index = zero_hopping_index(model)?;
    for state in 0..model.nsta() {
        model.ham[[zero_index, state, state]] -= Complex::new(chemical_potential, 0.0);
    }
    Ok(())
}

fn band_spin_expectations(eigenvectors: &Array2<Complex<f64>>, norb: usize) -> Array2<f64> {
    let nsta = eigenvectors.ncols();
    let mut spin = Array2::<f64>::zeros((nsta, 3));
    for band in 0..nsta {
        for orbital in 0..norb {
            let up = eigenvectors[[orbital, band]];
            let down = eigenvectors[[orbital + norb, band]];
            let coherence = up.conj() * down;
            spin[[band, 0]] += coherence.re;
            spin[[band, 1]] += coherence.im;
            spin[[band, 2]] += 0.5 * (up.norm_sqr() - down.norm_sqr());
        }
    }
    spin
}

fn local_spin_moment_from_spectra(
    spectra: &[KPointSpectrum],
    occupations: &[Array1<f64>],
    norb: usize,
) -> Array2<f64> {
    let mut spin = Array2::<f64>::zeros((norb, 3));
    let weight = 1.0 / spectra.len() as f64;
    for (spectrum, occupation) in spectra.iter().zip(occupations) {
        for (band, &f) in occupation.iter().enumerate() {
            if f == 0.0 {
                continue;
            }
            for orbital in 0..norb {
                let up = spectrum.eigenvectors[[orbital, band]];
                let down = spectrum.eigenvectors[[orbital + norb, band]];
                let coherence = up.conj() * down;
                let prefactor = weight * f;
                spin[[orbital, 0]] += prefactor * coherence.re;
                spin[[orbital, 1]] += prefactor * coherence.im;
                spin[[orbital, 2]] += 0.5 * prefactor * (up.norm_sqr() - down.norm_sqr());
            }
        }
    }
    spin
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpinDirection;
    use crate::solve_ham::Solve;
    use ndarray::{Axis, array};

    fn atomic_model() -> Model<true, 1> {
        Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap()
    }

    #[test]
    fn fixed_filling_atomic_hubbard_returns_fermi_centered_model() {
        let hubbard = HubbardModel::with_uniform_u(atomic_model(), 2.0).unwrap();
        let mut params = MeanFieldParams::new(
            [1],
            MeanFieldConstraint::FixedInitialFilling { reference_mu: 0.0 },
            Occupation::ZeroTemperature,
        );
        params.mixing = 0.5;
        params.density_tolerance = 1e-10;
        params.initial_magnetization = InitialMagnetization::Ferromagnetic {
            moment_per_orbital: 0.5,
        };

        let model = hubbard.solve_hartree_fock(&params).unwrap();
        let bands = model.solve_band_onek(&array![0.0]);
        assert!((bands[0] + 1.0).abs() < 1e-9);
        assert!((bands[1] - 1.0).abs() < 1e-9);

        let local_spin = model
            .local_spin_moment(&[1], 0.0, Occupation::ZeroTemperature)
            .unwrap();
        assert!(local_spin[[0, 0]].abs() < 1e-12);
        assert!(local_spin[[0, 1]].abs() < 1e-12);
        assert!((local_spin[[0, 2]] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn noncollinear_hartree_fock_preserves_spin_direction() {
        let hubbard = HubbardModel::with_uniform_u(atomic_model(), 2.0).unwrap();
        let mut params = MeanFieldParams::new(
            [1],
            MeanFieldConstraint::FixedInitialFilling { reference_mu: 0.0 },
            Occupation::ZeroTemperature,
        );
        params.mixing = 0.5;
        params.density_tolerance = 1e-10;
        params.initial_magnetization = InitialMagnetization::UniformVector {
            moment_per_orbital: [0.3, 0.4, 0.0],
        };

        let model = hubbard.solve_hartree_fock(&params).unwrap();
        let bands = model.solve_band_onek(&array![0.0]);
        assert!((bands[0] + 1.0).abs() < 1e-9);
        assert!((bands[1] - 1.0).abs() < 1e-9);

        let hamiltonian = model.gen_ham(&array![0.0], Gauge::Lattice);
        assert!((hamiltonian[[0, 1]].re + 0.6).abs() < 1e-9);
        assert!((hamiltonian[[0, 1]].im - 0.8).abs() < 1e-9);
        assert!((hamiltonian[[1, 0]] - hamiltonian[[0, 1]].conj()).norm() < 1e-12);

        let spin = model
            .spin_moment(&[1], 0.0, Occupation::ZeroTemperature)
            .unwrap();
        assert!((spin[0] - 0.3).abs() < 1e-9);
        assert!((spin[1] - 0.4).abs() < 1e-9);
        assert!(spin[2].abs() < 1e-9);
    }

    #[test]
    fn fixed_initial_filling_handles_a_metal_with_smearing() {
        let mut bare = atomic_model();
        bare.add_hop(-1.0, 0, 0, &array![1], None);
        let reference_mu = 0.37;
        let occupation = Occupation::FermiSmearing { width: 0.05 };
        let hubbard = HubbardModel::with_uniform_u(bare.clone(), 0.0).unwrap();
        let params = MeanFieldParams::new(
            [64],
            MeanFieldConstraint::FixedInitialFilling { reference_mu },
            occupation,
        );

        let model = hubbard.solve_hartree_fock(&params).unwrap();
        let k_points = uniform_k_mesh(&[64]).unwrap();
        let bare_bands = bare.solve_band_all_parallel(&k_points);
        let shifted_bands = model.solve_band_all_parallel(&k_points);
        let difference = &bare_bands - &shifted_bands;
        assert!(
            difference
                .iter()
                .all(|&value| (value - reference_mu).abs() < 1e-9)
        );

        let reference_filling = bare
            .electron_filling(&[64], reference_mu, occupation)
            .unwrap();
        let output_filling = model.electron_filling(&[64], 0.0, occupation).unwrap();
        assert!((reference_filling - output_filling).abs() < 1e-10);
    }

    #[test]
    fn fixed_chemical_potential_is_shifted_to_zero() {
        let bare = atomic_model();
        let hubbard = HubbardModel::with_uniform_u(bare, 0.0).unwrap();
        let params = MeanFieldParams::new(
            [1],
            MeanFieldConstraint::FixedChemicalPotential { mu: 0.25 },
            Occupation::FermiDirac {
                temperature_kelvin: 100.0,
            },
        );

        let model = hubbard.solve_hartree_fock(&params).unwrap();
        let bands = model.solve_band_onek(&array![0.0]);
        assert!(bands.iter().all(|&energy| (energy + 0.25).abs() < 1e-12));
    }

    #[test]
    fn spin_expectation_respects_spinor_basis() {
        let mut model = atomic_model();
        model.add_onsite(&array![1.0], SpinDirection::X);
        let spin = model.spin_expectation_onek(&array![0.0]).unwrap();

        assert!((spin[[0, 0]] + 0.5).abs() < 1e-12);
        assert!((spin[[1, 0]] - 0.5).abs() < 1e-12);
        assert!(
            spin.index_axis(Axis(1), 1)
                .iter()
                .all(|value| value.abs() < 1e-12)
        );
        assert!(
            spin.index_axis(Axis(1), 2)
                .iter()
                .all(|value| value.abs() < 1e-12)
        );

        let mut model = atomic_model();
        model.add_onsite(&array![1.0], SpinDirection::Y);
        let spin = model.spin_expectation_onek(&array![0.0]).unwrap();
        assert!((spin[[0, 1]] + 0.5).abs() < 1e-12);
        assert!((spin[[1, 1]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_wrong_number_of_hubbard_interactions() {
        let error = HubbardModel::new(atomic_model(), array![1.0, 2.0]).unwrap_err();
        assert!(matches!(
            error,
            TbError::DimensionMismatch {
                expected: 1,
                found: 2,
                ..
            }
        ));
    }
}
