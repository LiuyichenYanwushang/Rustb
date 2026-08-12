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
//! ```ignore
//! use rustb::cut::CutModel;
//!
//! // Create a 10-layer slab along direction 2
//! let slab = model.cut_piece(10, 2).unwrap();
//!
//! // Cut a hexagonal dot from the slab
//! let dot = slab.cut_dot(10, 6, None).unwrap();
//! ```

use crate::Model;
use crate::NoRMatrix;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::find_R;
use crate::{Atom, OrbitalId};
use ndarray::prelude::*;
use ndarray::*;
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
    /// # Parameters
    ///
    /// - `num`: size parameter controlling the number of unit cells.
    /// - `shape`: shape type:
    ///   - `3`: triangular.
    ///   - `4`: square.
    ///   - `6`: hexagonal.
    ///   - `8`: octagonal.
    /// - `dir`: for 3D models, the two in-plane directions.  For 2D models,
    ///   `None` uses directions 0 and 1.
    ///
    /// # Errors
    ///
    /// - [`TbError::InvalidDimension`] if dimension is not 2 or 3.
    /// - [`TbError::InvalidShape`] if shape is not 3, 4, 6, or 8.
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

    // Orbital-only models, and intentionally unassigned orbitals in mixed
    // models, are cut according to their own Wannier-center positions.
    for (orbital, owner) in owners.iter().enumerate() {
        if owner.is_none()
            && position_is_inside_shape(
                model.orb.row(orbital),
                num,
                shape,
                directions,
                trim_outer_square,
            )?
        {
            selected_orbitals[orbital] = true;
        }
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
        let mut new_orb = Array2::<f64>::zeros((self.norb() * num, self.dim_r()));
        let mut new_orb_proj = Vec::new();
        let mut new_atom = Vec::new();
        let new_norb = self.norb() * num;
        let new_nsta = self.nsta() * num;
        let new_natom = self.natom() * num;
        let mut new_lat = self.lat.clone();
        new_lat
            .row_mut(dir)
            .assign(&(self.lat.row(dir).to_owned() * (num as f64)));
        for i in 0..num {
            for n in 0..self.norb() {
                let mut use_orb = self.orb.row(n).to_owned();
                use_orb[[dir]] += i as f64;
                use_orb[[dir]] = use_orb[[dir]] / (num as f64);
                new_orb.row_mut(i * self.norb() + n).assign(&use_orb);
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
            let mut using_ham = self.ham.clone();
            let mut using_hamR = self.hamR.clone();
            let using_rmatrix = if <R as RMatrixData>::HAS_RMATRIX {
                self.rmatrix.as_array4().clone()
            } else {
                Array4::zeros((using_ham.shape()[0], self.dim_r(), self.nsta(), self.nsta()))
            };
            for n in 0..num {
                for (i0, (ind_R, (ham, rmatrix))) in using_hamR
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
                            new_ham.push(Axis(0), use_ham.view());
                            new_hamR.push(Axis(0), ind_R.view());
                            if <R as RMatrixData>::HAS_RMATRIX {
                                new_rmatrix.push(Axis(0), use_rmatrix.view());
                            }
                        }
                    } else {
                        continue;
                    }
                }
            }
        }
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

    fn cut_dot(
        &self,
        num: usize,
        shape: usize,
        dir: Option<Vec<usize>>,
    ) -> Result<Model<SPIN, DIM, R>> {
        self.validate()?;
        match self.dim_r() {
            3 => {
                let dir = if dir == None {
                    eprintln!(
                        "Wrong!, the dir is None, but model's dimension is 3, here we use default 0,1 direction"
                    );
                    let dir = vec![0, 1];
                    dir
                } else {
                    dir.unwrap()
                };
                let (old_model, use_orb_item, use_atom_item) = {
                    let model_1 = self.cut_piece(num + 1, dir[0])?;
                    let model_2 = model_1.cut_piece(num + 1, dir[1])?;
                    let (use_orb_item, use_atom_item) =
                        select_atoms_and_orbitals(&model_2, num, shape, [dir[0], dir[1]], true)?;
                    (model_2, use_orb_item, use_atom_item)
                };
                let norb = use_orb_item.len();
                let mut new_atom = Vec::new();
                let mut new_orb = Array2::<f64>::zeros((norb, self.dim_r()));
                let mut new_orb_proj = Vec::new();
                let mut old_to_new = vec![None; old_model.norb()];
                for (new, &old) in use_orb_item.iter().enumerate() {
                    old_to_new[old] = Some(OrbitalId::new(new));
                }
                for use_i in &use_atom_item {
                    let mut atom = old_model.atoms[*use_i].clone();
                    atom.set_orbitals(
                        atom.orbitals()
                            .iter()
                            .filter_map(|id| old_to_new[id.index()])
                            .collect(),
                    );
                    new_atom.push(atom);
                }
                for (i, use_i) in use_orb_item.iter().enumerate() {
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                    new_orb_proj.push(old_model.orb_projection[*use_i])
                }
                let new_nsta = if SPIN {
                    new_orb.len() * 2
                } else {
                    new_orb.len()
                };
                let n_R = old_model.hamR.len_of(Axis(0));
                let mut new_ham = Array3::<Complex<f64>>::zeros((n_R, new_nsta, new_nsta));
                let mut new_hamR = Array2::<isize>::zeros((0, self.dim_r()));

                let mut new_model = Self {
                    lat: old_model.lat.clone(),
                    orb: new_orb,
                    orb_projection: new_orb_proj,
                    atoms: new_atom,
                    ham: new_ham,
                    hamR: new_hamR,
                    rmatrix: R::from_array(Array4::zeros((n_R, self.dim_r(), new_nsta, new_nsta))),
                };

                let norb = new_model.norb();

                if SPIN {
                    let norb2 = old_model.norb();
                    for (r, R) in old_model.hamR.axis_iter(Axis(0)).enumerate() {
                        new_model.hamR.push_row(R);
                        for (i, use_i) in use_orb_item.iter().enumerate() {
                            for (j, use_j) in use_orb_item.iter().enumerate() {
                                new_model.ham[[r, i, j]] = old_model.ham[[r, *use_i, *use_j]];
                                new_model.ham[[r, i + norb, j + norb]] =
                                    old_model.ham[[r, *use_i + norb2, *use_j + norb2]];
                                new_model.ham[[r, i + norb, j]] =
                                    old_model.ham[[r, *use_i + norb2, *use_j]];
                                new_model.ham[[r, i, j + norb]] =
                                    old_model.ham[[r, *use_i, *use_j + norb2]];
                            }
                        }
                    }
                } else {
                    for (r, R) in old_model.hamR.axis_iter(Axis(0)).enumerate() {
                        new_model.hamR.push_row(R);
                        for (i, use_i) in use_orb_item.iter().enumerate() {
                            for (j, use_j) in use_orb_item.iter().enumerate() {
                                new_model.ham[[r, i, j]] = old_model.ham[[r, *use_i, *use_j]];
                            }
                        }
                    }
                }
                if <R as RMatrixData>::HAS_RMATRIX {
                    let nsta = new_model.nsta();
                    let mut new_rmatrix = Array4::<Complex<f64>>::zeros((
                        n_R,
                        self.dim_r(),
                        new_model.nsta(),
                        new_model.nsta(),
                    ));
                    if SPIN {
                        let norb2 = old_model.norb();
                        for r in 0..n_R {
                            for dim in 0..self.dim_r() {
                                for (i, use_i) in use_orb_item.iter().enumerate() {
                                    for (j, use_j) in use_orb_item.iter().enumerate() {
                                        new_rmatrix[[r, dim, i, j]] =
                                            old_model.rmatrix.as_array4()[[r, dim, *use_i, *use_j]];
                                        new_rmatrix[[r, dim, i + norb, j + norb]] = old_model
                                            .rmatrix
                                            .as_array4()[[r, dim, *use_i + norb2, *use_j + norb2]];
                                        new_rmatrix[[r, dim, i + norb, j]] = old_model
                                            .rmatrix
                                            .as_array4()[[r, dim, *use_i + norb2, *use_j]];
                                        new_rmatrix[[r, dim, i, j + norb]] = old_model
                                            .rmatrix
                                            .as_array4()[[r, dim, *use_i, *use_j + norb2]];
                                    }
                                }
                            }
                        }
                    } else {
                        for r in 0..n_R {
                            for dim in 0..self.dim_r() {
                                for (i, use_i) in use_orb_item.iter().enumerate() {
                                    for (j, use_j) in use_orb_item.iter().enumerate() {
                                        new_rmatrix[[r, dim, i, j]] =
                                            old_model.rmatrix.as_array4()[[r, dim, *use_i, *use_j]];
                                    }
                                }
                            }
                        }
                    }
                    new_model.rmatrix = R::from_array(new_rmatrix);
                }
                return Ok(new_model);
            }
            2 => {
                if dir != None {
                    eprintln!(
                        "Wrong!, the dimension of model is 2, but the dir is not None, you should give None!, here we use 0,1 direction"
                    );
                }

                let (old_model, use_orb_item, use_atom_item) = {
                    let model_1 = self.cut_piece(num + 1, 0)?;
                    let model_2 = model_1.cut_piece(num + 1, 1)?;
                    let (use_orb_item, use_atom_item) =
                        select_atoms_and_orbitals(&model_2, num, shape, [0, 1], false)?;
                    (model_2, use_orb_item, use_atom_item)
                };
                let norb = use_orb_item.len();
                let mut new_atom = Vec::new();
                let mut new_orb = Array2::<f64>::zeros((norb, self.dim_r()));
                let mut new_orb_proj = Vec::new();
                let mut old_to_new = vec![None; old_model.norb()];
                for (new, &old) in use_orb_item.iter().enumerate() {
                    old_to_new[old] = Some(OrbitalId::new(new));
                }
                for use_i in &use_atom_item {
                    let mut atom = old_model.atoms[*use_i].clone();
                    atom.set_orbitals(
                        atom.orbitals()
                            .iter()
                            .filter_map(|id| old_to_new[id.index()])
                            .collect(),
                    );
                    new_atom.push(atom);
                }
                for (i, use_i) in use_orb_item.iter().enumerate() {
                    new_orb.row_mut(i).assign(&old_model.orb.row(*use_i));
                    new_orb_proj.push(old_model.orb_projection[*use_i])
                }
                let mut new_model = Self::tb_model(old_model.lat.clone(), new_orb, Some(new_atom))?;
                new_model.orb_projection = new_orb_proj;
                let n_R = new_model.hamR.len_of(Axis(0));
                let mut new_ham =
                    Array3::<Complex<f64>>::zeros((n_R, new_model.nsta(), new_model.nsta()));
                let mut new_hamR = Array2::<isize>::zeros((1, self.dim_r()));
                let norb = new_model.norb();
                let nsta = new_model.nsta();

                if SPIN {
                    let norb2 = old_model.norb();
                    for (i, use_i) in use_orb_item.iter().enumerate() {
                        for (j, use_j) in use_orb_item.iter().enumerate() {
                            new_ham[[0, i, j]] = old_model.ham[[0, *use_i, *use_j]];
                            new_ham[[0, i + norb, j + norb]] =
                                old_model.ham[[0, *use_i + norb2, *use_j + norb2]];
                            new_ham[[0, i + norb, j]] = old_model.ham[[0, *use_i + norb2, *use_j]];
                            new_ham[[0, i, j + norb]] = old_model.ham[[0, *use_i, *use_j + norb2]];
                        }
                    }
                } else {
                    for (i, use_i) in use_orb_item.iter().enumerate() {
                        for (j, use_j) in use_orb_item.iter().enumerate() {
                            new_ham[[0, i, j]] = old_model.ham[[0, *use_i, *use_j]];
                        }
                    }
                }
                new_model.ham = new_ham;
                new_model.hamR = new_hamR;
                if <R as RMatrixData>::HAS_RMATRIX {
                    let mut new_rmatrix = Array4::<Complex<f64>>::zeros((
                        n_R,
                        self.dim_r(),
                        new_model.nsta(),
                        new_model.nsta(),
                    ));
                    if SPIN {
                        let norb2 = old_model.norb();
                        for dim in 0..self.dim_r() {
                            for (i, use_i) in use_orb_item.iter().enumerate() {
                                for (j, use_j) in use_orb_item.iter().enumerate() {
                                    new_rmatrix[[0, dim, i, j]] =
                                        old_model.rmatrix.as_array4()[[0, dim, *use_i, *use_j]];
                                    new_rmatrix[[0, dim, i + norb, j + norb]] = old_model
                                        .rmatrix
                                        .as_array4()[[0, dim, *use_i + norb2, *use_j + norb2]];
                                    new_rmatrix[[0, dim, i + norb, j]] = old_model
                                        .rmatrix
                                        .as_array4()[[0, dim, *use_i + norb2, *use_j]];
                                    new_rmatrix[[0, dim, i, j + norb]] = old_model
                                        .rmatrix
                                        .as_array4()[[0, dim, *use_i, *use_j + norb2]];
                                }
                            }
                        }
                    } else {
                        for dim in 0..self.dim_r() {
                            for (i, use_i) in use_orb_item.iter().enumerate() {
                                for (j, use_j) in use_orb_item.iter().enumerate() {
                                    new_rmatrix[[0, dim, i, j]] =
                                        old_model.rmatrix.as_array4()[[0, dim, *use_i, *use_j]];
                                }
                            }
                        }
                    }
                    new_model.rmatrix = R::from_array(new_rmatrix);
                }
                return Ok(new_model);
            }
            _ => {
                return Err(TbError::InvalidDimension {
                    dim: self.dim_r(),
                    supported: vec![2, 3],
                });
            }
        }
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::AtomType;
    use ndarray::array;

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
