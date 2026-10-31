use super::{Model, ORBITAL_ATOM_POSITION_TOLERANCE, RMatrixData};
use crate::error::{Result, TbError};
use ndarray_linalg::Inverse;

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// Validate synchronized model arrays, finite data, an invertible lattice,
    /// and atom-to-orbital references.
    ///
    /// This checks storage and geometry. Hermiticity and translation-support
    /// conventions must additionally be checked by the consuming algorithm.
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
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            for &orbital_id in atom.orbitals() {
                let orbital = orbital_id.index();
                for axis in 0..DIM {
                    let delta = (self.orb[[orbital, axis]] - atom.position_ref()[[axis]])
                        .abs()
                        .rem_euclid(1.0);
                    let distance = delta.min(1.0 - delta);
                    if distance > ORBITAL_ATOM_POSITION_TOLERANCE {
                        return Err(TbError::InvalidModelInvariant {
                            invariant: "orbital_atom_position",
                            message: format!(
                                "orbital {orbital} is {distance:.3} (fractional, modulo a \
                                 lattice vector) away from its parent atom {atom_index}, \
                                 exceeding the tolerance {ORBITAL_ATOM_POSITION_TOLERANCE}; \
                                 orbitals must sit at their parent atom's position"
                            ),
                        });
                    }
                }
            }
        }

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
        Ok(())
    }
}
