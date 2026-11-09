use super::{Model, ORBITAL_ATOM_POSITION_TOLERANCE, RMatrixData};
use crate::error::{Result, TbError};
use ndarray_linalg::Inverse;
use std::sync::atomic::{AtomicBool, Ordering};

static ORBITAL_CENTER_WARNING: AtomicBool = AtomicBool::new(false);

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Validate synchronized model arrays, finite data, an invertible lattice,
    /// and atom-to-orbital references.
    ///
    /// This checks storage and geometry. Hermiticity and translation-support
    /// conventions must additionally be checked by the consuming algorithm.
    /// Orbital centres displaced from their parent atoms are accepted and
    /// produce a one-time process-global warning; symmetry/irrep algorithms
    /// that require atom-centred orbitals validate that stricter condition
    /// themselves.
    pub fn validate(&self) -> Result<()> {
        if !(1..=3).contains(&DIM) {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![1, 2, 3],
            });
        }
        if self.norb() == 0 {
            return Err(TbError::NoOrbitals);
        }
        if self.lat.dim() != (DIM, DIM) {
            return Err(TbError::InvalidModelInvariant {
                invariant: "lattice_shape",
                message: format!("expected ({DIM}, {DIM}), found {:?}", self.lat.dim()),
            });
        }
        if self.orb.ncols() != DIM {
            return Err(TbError::InvalidModelInvariant {
                invariant: "orbital_position_shape",
                message: format!("expected {} columns, found {}", DIM, self.orb.ncols()),
            });
        }
        if !self
            .lat
            .iter()
            .chain(self.orb.iter())
            .all(|value| value.is_finite())
        {
            return Err(TbError::InvalidModelInvariant {
                invariant: "finite_geometry",
                message: "lattice and orbital positions must be finite".to_string(),
            });
        }
        if self.orb_projection.len() != self.norb() {
            return Err(TbError::InvalidModelInvariant {
                invariant: "orbital_projection_count",
                message: format!(
                    "expected {}, found {}",
                    self.norb(),
                    self.orb_projection.len()
                ),
            });
        }
        if self.lat.inv().map_or(true, |inverse| {
            inverse.iter().any(|value| !value.is_finite())
        }) {
            return Err(TbError::InvalidModelInvariant {
                invariant: "invertible_lattice",
                message: "the lattice must have a finite inverse".into(),
            });
        }
        for (index, atom) in self.atoms.iter().enumerate() {
            if atom.position_ref().len() != DIM
                || !atom.position_ref().iter().all(|value| value.is_finite())
            {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "atomic_position",
                    message: format!("atom {index} must have {DIM} finite coordinates"),
                });
            }
            if atom
                .magnetic_moment()
                .is_some_and(|moment| moment.iter().any(|component| !component.is_finite()))
            {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "magnetic_moment",
                    message: format!("atom {index} has a non-finite magnetic moment"),
                });
            }
        }
        self.orbital_owners()?;
        // Orbital/Wannier centres are often displaced from their parent atom
        // (for example, bond- or plaquette-centred Wannier functions). This is
        // a soft diagnostic, not a structural error: the owner mapping and all
        // Hamiltonian data remain well-defined. Algorithms that require
        // atom-centred orbitals (symmetry/irrep representations) perform their
        // own strict checks.
        let mut largest_deviation: Option<(usize, usize, f64)> = None;
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            for &orbital_id in atom.orbitals() {
                let orbital = orbital_id.index();
                for axis in 0..DIM {
                    let delta = (self.orb[[orbital, axis]] - atom.position_ref()[[axis]])
                        .abs()
                        .rem_euclid(1.0);
                    let distance = delta.min(1.0 - delta);
                    if distance > ORBITAL_ATOM_POSITION_TOLERANCE
                        && largest_deviation
                            .as_ref()
                            .is_none_or(|(_, _, previous)| distance > *previous)
                    {
                        largest_deviation = Some((orbital, atom_index, distance));
                    }
                }
            }
        }
        // The warning is emitted only after every hard invariant below passes,
        // so an invalid model cannot consume the one-time warning slot.

        let expected_ham = (self.hamR.nrows(), self.nsta(), self.nsta());
        if self.ham.dim() != expected_ham {
            return Err(TbError::InvalidModelInvariant {
                invariant: "hamiltonian_shape",
                message: format!("expected {expected_ham:?}, found {:?}", self.ham.dim()),
            });
        }
        if self.hamR.ncols() != DIM {
            return Err(TbError::InvalidModelInvariant {
                invariant: "hopping_translation_shape",
                message: format!("expected {DIM} columns, found {}", self.hamR.ncols()),
            });
        }
        if self
            .ham
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(TbError::InvalidModelInvariant {
                invariant: "finite_hamiltonian",
                message: "all Hamiltonian matrix elements must be finite".into(),
            });
        }
        if R::HAS_RMATRIX {
            let expected = (self.hamR.nrows(), DIM, self.nsta(), self.nsta());
            if self.rmatrix.as_array4().dim() != expected {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "position_matrix_shape",
                    message: format!(
                        "expected {expected:?}, found {:?}",
                        self.rmatrix.as_array4().dim()
                    ),
                });
            }
            if self
                .rmatrix
                .as_array4()
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "finite_position_matrix",
                    message: "all position matrix elements must be finite".into(),
                });
            }
        }
        if let Some((orbital, atom_index, distance)) = largest_deviation {
            // Keep this diagnostic to one line per process: validate() is on
            // every hot response/solver path, but the condition is structural
            // and does not need to be repeated for every call.
            if !ORBITAL_CENTER_WARNING.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "Warning: orbital {orbital} has a component-wise fractional deviation \
                     {distance:.3} (modulo a lattice vector) from its parent atom {atom_index}, \
                     exceeding the soft warning threshold {ORBITAL_ATOM_POSITION_TOLERANCE}; \
                     continuing, but symmetry/irrep algorithms that require atom-centred \
                     orbitals may reject this model"
                );
            }
        }
        Ok(())
    }
}
