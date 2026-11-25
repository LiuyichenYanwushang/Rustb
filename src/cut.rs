//! Tools for cutting slabs, ribbons, and shaped regions from a bulk model.
//!
//! This module provides the [`CutModel`] trait with two methods:
//!
//! - [`CutModel::cut_piece`]: truncate one direction into `num` layers,
//!   suitable for slab/ribbon construction.
//! - [`CutModel::cut_dot`]: cut a shaped dot/edge structure from a slab,
//!   supporting triangular (3), square (4), hexagonal (6), and octagonal (8)
//!   shapes.
//!
//! # Examples
//!
//! ```rust
//! use ndarray::{arr2, array};
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
//! let orb = arr2(&[[0.0, 0.0], [0.5, 0.5]]);
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! model.add_hop(-1.0, 0, 1, &array![0, 0], None);
//!
//! // A slab keeps the in-plane periodicity and removes hopping across the
//! // cut direction; the lattice vector along the cut is scaled by num.
//! let slab = model.cut_piece(10, 1)?;
//! assert_eq!(slab.norb(), 10 * model.norb());
//!
//! // A shaped finite region; codes are 3 (triangle), 4 (square),
//! // 6 (hexagon) and 8 (octagon). 2D models use directions 0 and 1.
//! let dot = slab.cut_dot(4, 6, None)?;
//! assert!(dot.norb() > 0);
//!
//! // 3D models pass the two in-plane directions explicitly.
//! # Ok(())
//! # }
//! ```

use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::find_R;
use crate::model::supercell::normalized_to_atoms;
use crate::{Atom, OrbitalId};
use ndarray::prelude::*;
use num_complex::Complex;
use std::ops::AddAssign;

/// Trait for cutting slabs, ribbons, and shaped dots from a bulk model.
pub trait CutModel {
    /// Cut `num` layers along direction `dir`, forming a slab with no periodic
    /// hopping along that direction.
    ///
    /// The resulting model has the same in-plane lattice vectors but the lattice
    /// vector along `dir` is scaled by `num`.  Orbitals and atoms are replicated
    /// `num` times, and inter-layer hopping from the original model is mapped
    /// to intra-slab hopping.
    ///
    /// # Parameters
    ///
    /// - `num`: number of unit-cell layers along the cut direction.
    /// - `dir`: the lattice-vector direction to cut (0-based).
    ///
    /// # Returns
    ///
    /// A new `Model` with `num * norb` orbitals and `num * natom` atoms.
    ///
    /// # Errors
    ///
    /// - [`TbError::InvalidSupercellSize`] if `num < 1`.
    /// - [`TbError::InvalidDirection`] if `dir` is out of range.
    fn cut_piece(&self, num: usize, dir: usize) -> Result<Self>
    where
        Self: Sized;

    /// Cut a shaped dot or edge from the model.
    ///
    /// Atom-owned slots follow the atom selection. In an orbital-only model,
    /// the first/up centre selects a paired slot; both spin states are retained
    /// or removed together without replacing their independent centres.
    ///
    /// # Parameters
    ///
    /// - `num`: size parameter controlling the number of unit cells.
    /// - `shape`: shape type:
    ///   - `3`: triangular.
    ///   - `4`: square.
    ///   - `6`: hexagonal.
    ///   - `8`: octagonal.
    /// - `dir`: for 3D models, the two in-plane directions; `None` selects
    ///   `[0, 1]`. A 2D model always uses both in-plane directions and rejects
    ///   an explicit `dir` instead of ignoring it.
    ///
    /// # Errors
    ///
    /// - [`TbError::InvalidDimension`] if dimension is not 2 or 3.
    /// - [`TbError::InvalidShape`] if shape is not 3, 4, 6, or 8.
    /// - [`TbError::Other`] for repeated directions or a 2D `dir` argument.
    fn cut_dot(&self, num: usize, shape: usize, dir: Option<Vec<usize>>) -> Result<Self>
    where
        Self: Sized;
}

fn position_is_inside_shape(
    position: ArrayView1<'_, f64>,
    num: usize,
    shape: usize,
    directions: [usize; 2],
    trim_outer_square: bool,
) -> Result<bool> {
    if num == 0 {
        return Err(TbError::InvalidSupercellSize(num));
    }
    let x = position[directions[0]];
    let y = position[directions[1]];
    let size = num as f64;
    let outer = size / (size + 1.0);
    let tolerance = 1e-5;
    let inside = match shape {
        3 => x + y <= outer + tolerance,
        4 => x <= outer + tolerance && y <= outer + tolerance,
        6 if trim_outer_square => {
            (x - y).abs() <= 0.5 * outer + tolerance
                && x <= outer + tolerance
                && y <= outer + tolerance
        }
        6 => (x - y).abs() <= 0.5 + tolerance,
        8 => {
            y - x + 0.5 >= -tolerance
                && x - y + 0.5 >= -tolerance
                && x + y >= 0.5 - tolerance
                && y - x <= 0.5 + tolerance
        }
        _ => {
            return Err(TbError::InvalidShape {
                shape,
                supported: vec![3, 4, 6, 8],
            });
        }
    };
    Ok(inside)
}

fn select_atoms_and_orbitals<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    model: &Model<SPIN, DIM, R>,
    num: usize,
    shape: usize,
    directions: [usize; 2],
    trim_outer_square: bool,
) -> Result<(Vec<usize>, Vec<usize>)> {
    model.validate()?;
    let owners = model.orbital_owners()?;
    let mut selected_orbitals = vec![false; model.norb()];
    let mut selected_atoms = Vec::new();

    for (atom_index, atom) in model.atoms.iter().enumerate() {
        if position_is_inside_shape(
            atom.position_ref().view(),
            num,
            shape,
            directions,
            trim_outer_square,
        )? {
            selected_atoms.push(atom_index);
            for orbital in atom.orbitals() {
                selected_orbitals[orbital.index()] = true;
            }
        }
    }

    if model.atoms.is_empty() {
        // Orbital-only models select paired slots using their first/up
        // centre. The corresponding down state follows that same decision,
        // retaining its own centre even if it lies across the shape boundary.
        for (orbital, owner) in owners.iter().enumerate() {
            debug_assert!(owner.is_none());
            if position_is_inside_shape(
                model.orb.row(orbital),
                num,
                shape,
                directions,
                trim_outer_square,
            )? {
                selected_orbitals[orbital] = true;
            }
        }
    } else if owners.iter().any(Option::is_none) {
        // In a model with atoms, an orbital without an owner has no atom
        // whose deletion decision it can follow; reject the mixed state
        // instead of deciding independently.
        return Err(TbError::InvalidModelInvariant {
            invariant: "cut_orbital_ownership",
            message: "the model has atoms, but some orbitals do not belong to \
                      any atom; orbitals must follow their parent atom when cutting"
                .to_string(),
        });
    }

    let selected_orbitals = selected_orbitals
        .into_iter()
        .enumerate()
        .filter_map(|(orbital, selected)| selected.then_some(orbital))
        .collect();
    Ok((selected_orbitals, selected_atoms))
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> CutModel for Model<SPIN, DIM, R> {
    fn cut_piece(&self, num: usize, dir: usize) -> Result<Model<SPIN, DIM, R>> {
        // Normalize the orbital gauge first: atoms into [0, 1), orbitals to
        // their atom-adjacent periodic image, with covariant relabeling of
        // the Hamiltonian and position matrix.
        let model = normalized_to_atoms(self)?;
        model.cut_piece_impl(num, dir)
    }

    fn cut_dot(
        &self,
        num: usize,
        shape: usize,
        dir: Option<Vec<usize>>,
    ) -> Result<Model<SPIN, DIM, R>> {
        let model = normalized_to_atoms(self)?;
        model.cut_dot_impl(num, shape, dir)
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    fn cut_piece_impl(&self, num: usize, dir: usize) -> Result<Model<SPIN, DIM, R>> {
        self.validate()?;
        if num < 1 {
            return Err(TbError::InvalidSupercellSize(num));
        }
        if dir >= self.dim_r() {
            return Err(TbError::InvalidDirection {
                index: dir,
                dim: self.dim_r(),
            });
        }
        let mut new_orb = Array2::<f64>::zeros((self.nsta() * num, self.dim_r()));
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new();
        let new_norb = self.norb() * num;
        let new_nsta = self.nsta() * num;
        let _new_natom = self.natom() * num;
        let mut new_lat = self.lat.clone();
        new_lat
            .row_mut(dir)
            .assign(&(self.lat.row(dir).to_owned() * (num as f64)));
        for i in 0..num {
            for n in 0..self.norb() {
                for spin in 0..if SPIN { 2 } else { 1 } {
                    let mut use_orb = self.orb.row(n + spin * self.norb()).to_owned();
                    use_orb[[dir]] = (use_orb[[dir]] + i as f64) / num as f64;
                    new_orb
                        .row_mut(spin * new_norb + i * self.norb() + n)
                        .assign(&use_orb);
                }
                new_orb_proj.push(self.orb_projection[n]);
            }
            for n in 0..self.natom() {
                let mut use_atom_position = self.atoms[n].position();
                use_atom_position[[dir]] += i as f64;
                use_atom_position[[dir]] *= 1.0 / (num as f64);
                let mut use_atom = Atom::with_orbitals(
                    use_atom_position,
                    self.atoms[n].atom_type(),
                    self.atoms[n]
                        .orbitals()
                        .iter()
                        .map(|id| OrbitalId::new(i * self.norb() + id.index())),
                );
                if let Some(moment) = self.atoms[n].magnetic_moment() {
                    use_atom.set_magnetic_moment(moment)?;
                }
                new_atom.push(use_atom);
            }
        }
        let mut new_ham = Array3::<Complex<f64>>::zeros((1, new_nsta, new_nsta));
        let mut new_rmatrix = Array4::<Complex<f64>>::zeros((1, self.dim_r(), new_nsta, new_nsta));
        let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
        {
            let using_ham = self.ham.clone();
            let using_hamR = self.hamR.clone();
            let using_rmatrix = if <R as RMatrixData>::HAS_RMATRIX {
                self.rmatrix.as_array4().clone()
            } else {
                Array4::zeros((using_ham.shape()[0], self.dim_r(), self.nsta(), self.nsta()))
            };
            for n in 0..num {
                for (_i0, (ind_R, (ham, rmatrix))) in using_hamR
                    .outer_iter()
                    .zip(using_ham.outer_iter().zip(using_rmatrix.outer_iter()))
                    .enumerate()
                {
                    let ind: usize = (ind_R[[dir]] + (n as isize)) as usize;
                    let mut ind_R = ind_R.to_owned();
                    let ham = ham.to_owned();
                    let rmatrix = rmatrix.to_owned();
                    ind_R[[dir]] = 0;
                    if ind < num {
                        let mut use_ham = Array2::<Complex<f64>>::zeros((new_nsta, new_nsta));
                        if SPIN {
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);

                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![self.norb()..self.nsta(), 0..self.norb()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                            let mut s = use_ham.slice_mut(s![
                                new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                new_norb + ind * self.norb()..new_norb + (ind + 1) * self.norb()
                            ]);
                            let ham0 =
                                ham.slice(s![self.norb()..self.nsta(), self.norb()..self.nsta()]);
                            s.assign(&ham0);
                        } else {
                            let mut s = use_ham.slice_mut(s![
                                n * self.norb()..(n + 1) * self.norb(),
                                ind * self.norb()..(ind + 1) * self.norb()
                            ]);
                            let ham0 = ham.slice(s![0..self.norb(), 0..self.norb()]);
                            s.assign(&ham0);
                        }
                        // Handle rmatrix
                        let mut use_rmatrix =
                            Array3::<Complex<f64>>::zeros((self.dim_r(), new_nsta, new_nsta));
                        if <R as RMatrixData>::HAS_RMATRIX {
                            if SPIN {
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    n * self.norb()..(n + 1) * self.norb(),
                                    ind * self.norb()..(ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., 0..self.norb(), 0..self.norb()]);
                                s.assign(&rmatrix0);

                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                    ind * self.norb()..(ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., self.norb()..self.nsta(), 0..self.norb()]);
                                s.assign(&rmatrix0);
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    n * self.norb()..(n + 1) * self.norb(),
                                    new_norb + ind * self.norb()
                                        ..new_norb + (ind + 1) * self.norb()
                                ]);
                                let rmatrix0 =
                                    rmatrix.slice(s![.., 0..self.norb(), self.norb()..self.nsta()]);
                                s.assign(&rmatrix0);
                                let mut s = use_rmatrix.slice_mut(s![
                                    ..,
                                    new_norb + n * self.norb()..new_norb + (n + 1) * self.norb(),
                                    new_norb + ind * self.norb()
                                        ..new_norb + (ind + 1) * self.norb()
                                ]);
                                let rmatrix0 = rmatrix.slice(s![
                                    ..,
                                    self.norb()..self.nsta(),
                                    self.norb()..self.nsta()
                                ]);
                                s.assign(&rmatrix0);
                            } else {
                                for i in 0..self.norb() {
                                    for j in 0..self.norb() {
                                        for r in 0..self.dim_r() {
                                            use_rmatrix
                                                [[r, i + n * self.norb(), j + ind * self.norb()]] =
                                                rmatrix[[r, i, j]];
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(index) = find_R(&new_hamR, &ind_R) {
                            new_ham.slice_mut(s![index, .., ..]).add_assign(&use_ham);
                            if <R as RMatrixData>::HAS_RMATRIX {
                                new_rmatrix
                                    .slice_mut(s![index, .., .., ..])
                                    .add_assign(&use_rmatrix);
                            }
                        } else {
                            new_ham.push(Axis(0), use_ham.view())?;
                            new_hamR.push_row(ind_R.view())?;
                            if <R as RMatrixData>::HAS_RMATRIX {
                                new_rmatrix.push(Axis(0), use_rmatrix.view())?;
                            }
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
        // Each layer shifts the orbital's Cartesian position by the layer
        // displacement along dir; the position-matrix diagonal must follow.
        // State order is (spin, layer, source orbital).
        let source: Vec<usize> = (0..if SPIN { 2 } else { 1 })
            .flat_map(|spin| {
                (0..num).flat_map(move |_| {
                    (0..self.norb()).map(move |orbital| orbital + spin * self.norb())
                })
            })
            .collect();
        crate::model::supercell::set_rmatrix_diagonal_with_displacement::<DIM>(
            &mut new_rmatrix,
            &new_hamR,
            &new_orb,
            &new_lat,
            &self.orb,
            &self.lat,
            &crate::model::supercell::rmatrix_diagonal_cartesian(self),
            &source,
        );
        let model = Self {
            lat: new_lat,
            orb: new_orb,
            orb_projection: new_orb_proj,
            atoms: new_atom,
            ham: new_ham,
            hamR: new_hamR,
            rmatrix: R::from_array(new_rmatrix),
        };
        model.validate()?;
        Ok(model)
    }

    fn cut_dot_impl(
        &self,
        num: usize,
        shape: usize,
        dir: Option<Vec<usize>>,
    ) -> Result<Model<SPIN, DIM, R>> {
        self.validate()?;
        let directions = match DIM {
            2 => {
                if dir.is_some() {
                    return Err(TbError::Other(
                        "cut_dot uses both in-plane directions for 2D models; remove dir".into(),
                    ));
                }
                [0, 1]
            }
            3 => {
                let dir = dir.unwrap_or_else(|| vec![0, 1]);
                if dir.len() != 2 || dir[0] == dir[1] {
                    return Err(TbError::Other(
                        "cut_dot requires two distinct directions".into(),
                    ));
                }
                [dir[0], dir[1]]
            }
            _ => {
                return Err(TbError::InvalidDimension {
                    dim: DIM,
                    supported: vec![2, 3],
                });
            }
        };
        let layers = num
            .checked_add(1)
            .ok_or(TbError::InvalidSupercellSize(num))?;
        let mut model = self
            .cut_piece(layers, directions[0])?
            .cut_piece(layers, directions[1])?;
        let (orbitals, atoms) =
            select_atoms_and_orbitals(&model, num, shape, directions, DIM == 3)?;
        if orbitals.is_empty() {
            return Err(TbError::NoOrbitals);
        }
        // Reuse the paired-state selection used by every structural removal;
        // keep selected empty atoms as structural sites as well.
        let removed_atoms = (0..model.natom())
            .filter(|index| atoms.binary_search(index).is_err())
            .collect::<Vec<_>>();
        let removed_orbitals = (0..model.norb())
            .filter(|index| orbitals.binary_search(index).is_err())
            .collect::<Vec<_>>();
        model.remove_atoms_only(&removed_atoms)?;
        model.remove_orb(&removed_orbitals)?;
        Ok(model)
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::AtomType;
    use crate::HasRMatrix;

    use ndarray::array;

    fn check_asymmetric_cut<R: RMatrixData>() {
        for with_atoms in [false, true] {
            let mut model = Model::<true, 1, R>::tb_model_with_state_centres(
                array![[2.0]],
                array![[1.1], [-0.2]],
                with_atoms.then(|| {
                    vec![Atom::with_orbitals(
                        array![0.15],
                        AtomType::C,
                        [OrbitalId::new(0)],
                    )]
                }),
            )
            .unwrap();
            let hopping = Complex::new(0.4, 0.2);
            model.add_element(hopping, 0, 1, &array![0]).unwrap();
            if R::HAS_RMATRIX {
                model.rmatrix.as_array4_mut()[[0, 0, 0, 1]] = hopping;
                model.rmatrix.as_array4_mut()[[0, 0, 1, 0]] = hopping.conj();
            }
            let cut = model.cut_piece(3, 0).unwrap();
            assert_eq!((cut.norb(), cut.nsta(), cut.orb.nrows()), (3, 6, 6));
            assert_eq!(cut.natom(), if with_atoms { 3 } else { 0 });
            let down = if with_atoms { -0.2 } else { 0.8 };
            for layer in 0..3 {
                assert!((cut.orb[[layer, 0]] - (0.1 + layer as f64) / 3.0).abs() < 1e-12);
                assert!((cut.orb[[3 + layer, 0]] - (down + layer as f64) / 3.0).abs() < 1e-12);
            }
            let shift = if with_atoms { 1 } else { 2 };
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j + shift {
                        hopping
                    } else {
                        Complex::new(0.0, 0.0)
                    };
                    assert_eq!(cut.ham[[0, i, j + 3]], expected);
                    assert_eq!(cut.ham[[0, j + 3, i]], expected.conj());
                    if R::HAS_RMATRIX {
                        assert_eq!(cut.rmatrix.as_array4()[[0, 0, i, j + 3]], expected);
                    }
                }
            }
            if R::HAS_RMATRIX {
                for state in 0..6 {
                    assert!(
                        (cut.rmatrix.as_array4()[[0, 0, state, state]].re
                            - 6.0 * cut.orb[[state, 0]])
                        .abs()
                            < 1e-12
                    );
                }
            }
            cut.validate().unwrap();
        }
    }

    #[test]
    fn cuts_preserve_asymmetric_centres_and_state_specific_cell_shifts() {
        check_asymmetric_cut::<crate::NoRMatrix>();
        check_asymmetric_cut::<HasRMatrix>();
    }

    #[test]
    fn dot_selects_paired_slots_without_discarding_down_centres() {
        let model = Model::<true, 2, HasRMatrix>::tb_model_with_state_centres(
            Array2::eye(2),
            array![[0.4, 0.4], [0.8, 0.8]],
            None,
        )
        .unwrap();
        let dot = model.cut_dot(1, 3, None).unwrap();
        assert_eq!(dot.orb, array![[0.2, 0.2], [0.4, 0.4]]);
        assert_eq!((dot.norb(), dot.nsta()), (1, 2));
        assert_eq!(
            dot.rmatrix.as_array4()[[0, 0, 1, 1]],
            Complex::new(0.8, 0.0)
        );
        dot.validate().unwrap();

        let model = Model::<true, 3, HasRMatrix>::tb_model_with_state_centres(
            Array2::eye(3),
            array![[0.1, 0.1, 0.2], [0.9, 0.9, 0.3]],
            Some(vec![Atom::with_orbitals(
                array![0.2, 0.2, 0.2],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let dot = model.cut_dot(1, 4, Some(vec![0, 1])).unwrap();
        let expected = array![[0.05, 0.05, 0.2], [-0.05, -0.05, 0.3]];
        assert!((&dot.orb - &expected).iter().all(|x| x.abs() < 1e-12));
        assert_eq!((dot.norb(), dot.nsta(), dot.natom()), (1, 2, 1));
        dot.validate().unwrap();
    }

    #[test]
    fn three_dimensional_cut_dot_selects_orbitals() {
        // Regression: new_orb.len() counted matrix elements (norb * 3)
        // instead of orbitals, so every non-empty 3D cut_dot built a
        // wrong-shaped Hamiltonian and failed validate().
        let orbital_model = |spin: bool| -> Result<Model<false, 3>> {
            let mut model = Model::<false, 3>::tb_model(
                Array2::eye(3),
                array![[0.2, 0.2, 0.0]],
                Some(vec![Atom::with_orbitals(
                    array![0.2, 0.2, 0.0],
                    AtomType::C,
                    [OrbitalId::new(0)],
                )]),
            )?;
            model.add_hop(-1.0, 0, 0, &array![1, 0, 0], None);
            model.add_hop(-1.0, 0, 0, &array![0, 1, 0], None);
            let _ = spin;
            Ok(model)
        };

        let model = orbital_model(false).unwrap();
        let dot = model.cut_dot(1, 4, Some(vec![0, 1])).unwrap();
        dot.validate().unwrap();
        assert_eq!(dot.norb(), 1);
        assert_eq!(dot.nsta(), 1);

        // Spinful variant exercises the nsta doubling path.
        let mut spinful = Model::<true, 3>::tb_model(
            Array2::eye(3),
            array![[0.2, 0.2, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.2, 0.2, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        spinful.add_hop(-1.0, 0, 0, &array![1, 0, 0], None);
        spinful.add_hop(-1.0, 0, 0, &array![0, 1, 0], None);
        let dot = spinful.cut_dot(1, 4, Some(vec![0, 1])).unwrap();
        dot.validate().unwrap();
        assert_eq!(dot.norb(), 1);
        assert_eq!(dot.nsta(), 2);
    }

    #[test]
    fn cut_dot_rejects_empty_selection() {
        // Regression: the 3D cut_dot path returned an invalid zero-orbital
        // model without validate() when the shape selected nothing.
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.9, 0.9, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.9, 0.9, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.add_hop(-1.0, 0, 0, &array![1, 0, 0], None);
        let result = model.cut_dot(1, 3, Some(vec![0, 1]));
        assert!(matches!(result, Err(TbError::NoOrbitals)));
    }

    #[test]
    fn cut_dot_rejects_an_explicit_direction_for_2d_models() {
        let model =
            Model::<false, 2>::tb_model(array![[1.0, 0.0], [0.0, 1.0]], array![[0.0, 0.0]], None)
                .unwrap();
        let error = model.cut_dot(1, 4, Some(vec![0, 1])).unwrap_err();
        assert!(error.to_string().contains("2D"), "{error}");
    }

    #[test]
    fn cut_piece_shifts_rmatrix_diagonal_by_layer_displacement() {
        // Regression: each cut layer shifts the orbital's Cartesian position
        // by the layer displacement along dir, but the position-matrix
        // diagonal was copied unchanged. 1D, lat=2, orb=0.25, two layers:
        // correct diagonals are 0.5 and 2.5 A.
        let mut model =
            Model::<false, 1, HasRMatrix>::tb_model(array![[2.0]], array![[0.25]], None).unwrap();
        model.add_hop(-1.0, 0, 0, &array![1], None);

        let cut = model.cut_piece(2, 0).unwrap();
        cut.validate().unwrap();
        let rmatrix = cut.rmatrix.as_array4();
        let zero_r = Array1::<isize>::zeros(1);
        let r0 = find_R(&cut.hamR, &zero_r).unwrap();
        let expected = [0.5, 2.5];
        for layer in 0..2 {
            assert!(
                (rmatrix[[r0, 0, layer, layer]] - Complex::new(expected[layer], 0.0)).norm()
                    < 1e-12,
                "layer {layer} diagonal must be {}, found {}",
                expected[layer],
                rmatrix[[r0, 0, layer, layer]]
            );
        }
    }

    #[test]
    fn boundary_cut_keeps_atom_adjacent_representative() {
        // Regression: cutting a model whose orbital sits just across the
        // cell boundary from its atom (atom = 0.99, orb = 0.01) must
        // normalize every gauge-equivalent representative (0.01, 1.01,
        // -0.99) to the same atom-adjacent canonical form with covariant
        // Hamiltonian relabeling; a bare floor into [0, 1) would leave the
        // Hamiltonian uncompensated and silently change the physics.
        let boundary_model = |orb_x: f64| {
            let mut model = Model::<false, 1>::tb_model(
                array![[1.0]],
                array![[orb_x]],
                Some(vec![Atom::with_orbitals(
                    array![0.99],
                    AtomType::C,
                    [OrbitalId::new(0)],
                )]),
            )
            .unwrap();
            model.add_hop(-1.0, 0, 0, &array![1], None);
            model
        };

        let cut_a = boundary_model(0.01).cut_piece(2, 0).unwrap();
        let cut_b = boundary_model(1.01).cut_piece(2, 0).unwrap();
        let cut_c = boundary_model(-0.99).cut_piece(2, 0).unwrap();

        cut_a.validate().unwrap();
        // Gauge equivalence: all three representatives must produce the
        // identical normalized cut.
        for (other, name) in [(&cut_b, "1.01"), (&cut_c, "-0.99")] {
            assert_eq!(
                cut_a.orb, other.orb,
                "orb differs for representative {name}"
            );
            assert_eq!(
                cut_a.hamR, other.hamR,
                "hamR differs for representative {name}"
            );
            for (block_a, block_b) in cut_a.ham.outer_iter().zip(other.ham.outer_iter()) {
                assert!(
                    block_a
                        .iter()
                        .zip(block_b.iter())
                        .all(|(a, b)| (*a - *b).norm() < 1e-14),
                    "hopping blocks differ for representative {name}"
                );
            }
        }
        // The cut keeps the atom-adjacent periodic images (0.505 for the
        // first layer, 1.005 for the second — the latter is the canonical
        // representative nearest its atom at 0.995).
        assert!((cut_a.orb[[0, 0]] - 0.505).abs() < 1e-12);
        assert!((cut_a.orb[[1, 0]] - 1.005).abs() < 1e-12);
    }

    #[test]
    fn orbital_only_dot_uses_orbital_positions() {
        let model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.2, 0.2], [0.8, 0.8]], None)
                .unwrap();
        let dot = model.cut_dot(1, 3, None).unwrap();

        assert!(dot.norb() > 0);
        assert!(dot.atoms.is_empty());
        dot.validate().unwrap();
    }

    #[test]
    fn dot_selection_follows_non_contiguous_orbital_ids() {
        let model = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.8, 0.8], [0.2, 0.2]],
            Some(vec![
                Atom::with_orbitals(array![0.2, 0.2], AtomType::C, [OrbitalId::new(1)]),
                Atom::with_orbitals(array![0.8, 0.8], AtomType::O, [OrbitalId::new(0)]),
            ]),
        )
        .unwrap();
        let (orbitals, atoms) = select_atoms_and_orbitals(&model, 1, 3, [0, 1], false).unwrap();

        assert_eq!(atoms, vec![0]);
        assert_eq!(orbitals, vec![1]);
    }

    #[test]
    fn cuts_reject_invalid_atom_orbital_references_at_entry() {
        let mut model = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.2, 0.2]],
            Some(vec![Atom::with_orbitals(
                array![0.2, 0.2],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.atoms[0].set_orbitals(vec![OrbitalId::new(3)]);

        assert!(matches!(
            model.cut_piece(2, 0),
            Err(TbError::InvalidOrbitalId { .. })
        ));
        assert!(matches!(
            model.cut_dot(2, 4, None),
            Err(TbError::InvalidOrbitalId { .. })
        ));
    }

    #[test]
    fn cut_piece_replicates_optional_atom_moments() {
        let mut model = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.2, 0.2]],
            Some(vec![Atom::with_orbitals(
                array![0.2, 0.2],
                AtomType::Fe,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.atoms[0].set_magnetic_moment([1.0, 0.0, 0.0]).unwrap();
        let ribbon = model.cut_piece(3, 0).unwrap();
        assert_eq!(ribbon.natom(), 3);
        assert!(
            ribbon
                .atoms
                .iter()
                .all(|atom| atom.magnetic_moment() == Some([1.0, 0.0, 0.0]))
        );
        ribbon.validate().unwrap();
    }
}
