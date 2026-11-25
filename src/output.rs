//!This module handles output in various standard formats, including:
//!
//!wannier90_hr.dat format
//!
//!wannier90_centres.xyz format
//!
//!wannier90.win format
//!
//!All-in-one Wannier90 format
//!
//!POSCAR format
//!
//! # Examples
//!
//! ```no_run
//! use ndarray::{arr1, arr2, array, Array2};
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
//! let orb = arr2(&[[0.0, 0.0], [0.5, 0.5]]);
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! model.add_hop(-1.0, 0, 1, &array![0, 0], None);
//!
//! // Band structure: data files plus a PDF named after the last argument.
//! let path = arr2(&[[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]);
//! let labels = vec!["Γ", "X", "Γ"];
//! model.show_band(&path, &labels, 101, "band_output")?;
//!
//! // Wannier90 interchange files.
//! model.output_hr("target/example-output/", "seedname")?;
//! model.output_win("target/example-output/", "seedname")?;
//!
//! // POSCAR and centres.xyz need atom metadata.
//! let atoms = vec![Atom::with_orbitals(
//!     array![0.0, 0.0, 0.0],
//!     AtomType::Si,
//!     [OrbitalId::new(0)],
//! )];
//! let silicon = Model::<false, 3>::tb_model(
//!     Array2::eye(3),
//!     array![[0.0, 0.0, 0.0]],
//!     Some(atoms),
//! )?;
//! silicon.output_POSCAR("target/example-output/")?;
//! silicon.output_xyz("target/example-output/", "seedname")?;
//!
//! // Heat maps render a 2D array through gnuplot.
//! draw_heatmap(&arr2(&[[1.0, 2.0], [3.0, 4.0]]), "heatmap");
//! # Ok(())
//! # }
//! ```

use crate::atom_struct::{AtomType, OrbProj};
use crate::{HasRMatrix, Model};

use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::kpath::*;
use crate::solve_ham::Solve;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex64;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::time::{SystemTime, UNIX_EPOCH};

fn projection_token(projection: OrbProj) -> &'static str {
    match projection {
        OrbProj::s => "s",
        OrbProj::px => "px",
        OrbProj::py => "py",
        OrbProj::pz => "pz",
        OrbProj::dxy => "dxy",
        OrbProj::dxz => "dxz",
        OrbProj::dyz => "dyz",
        OrbProj::dz2 => "dz2",
        OrbProj::dx2y2 => "dx2-y2",
        OrbProj::fz3 => "fz3",
        OrbProj::fxz2 => "fxz2",
        OrbProj::fyz2 => "fyz2",
        OrbProj::fzx2y2 => "fz(x2-y2)",
        OrbProj::fxyz => "fxyz",
        OrbProj::fxx23y2 => "fx(x2-3y2)",
        OrbProj::fy3x2y2 => "fy(3x2-y2)",
        OrbProj::sp_1 => "sp-1",
        OrbProj::sp_2 => "sp-2",
        OrbProj::sp2_1 => "sp2-1",
        OrbProj::sp2_2 => "sp2-2",
        OrbProj::sp2_3 => "sp2-3",
        OrbProj::sp3_1 => "sp3-1",
        OrbProj::sp3_2 => "sp3-2",
        OrbProj::sp3_3 => "sp3-3",
        OrbProj::sp3_4 => "sp3-4",
        OrbProj::sp3d_1 => "sp3d-1",
        OrbProj::sp3d_2 => "sp3d-2",
        OrbProj::sp3d_3 => "sp3d-3",
        OrbProj::sp3d_4 => "sp3d-4",
        OrbProj::sp3d_5 => "sp3d-5",
        OrbProj::sp3d2_1 => "sp3d2-1",
        OrbProj::sp3d2_2 => "sp3d2-2",
        OrbProj::sp3d2_3 => "sp3d2-3",
        OrbProj::sp3d2_4 => "sp3d2-4",
        OrbProj::sp3d2_5 => "sp3d2-5",
        OrbProj::sp3d2_6 => "sp3d2-6",
    }
}

fn utc_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = day_seconds / 3_600;
    let minute = (day_seconds % 3_600) / 60;
    let second = day_seconds % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

// Howard Hinnant civil-from-days, with days counted from 1970-01-01.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    (
        if month <= 2 { year + 1 } else { year },
        month as u32,
        day as u32,
    )
}

fn write_wannier_matrix<const DIM: usize>(
    name: &str,
    ham_r: &Array2<isize>,
    nsta: usize,
    matrix: ArrayView4<'_, Complex64>,
    position: bool,
) -> Result<()> {
    if !(1..=3).contains(&DIM) {
        return Err(TbError::InvalidDimension {
            dim: DIM,
            supported: vec![1, 2, 3],
        });
    }
    let invalid =
        |message: String| io::Error::new(io::ErrorKind::InvalidInput, format!("{name}: {message}"));
    let nr = ham_r.nrows();
    let stored_components = if position { DIM } else { 1 };
    if nr == 0 || ham_r.ncols() != DIM || matrix.dim() != (nr, stored_components, nsta, nsta) {
        return Err(invalid("matrix shape does not match hamR and the model basis".into()).into());
    }
    if !(1..=99_999).contains(&nsta) {
        return Err(invalid(format!(
            "state count {nsta} is outside the I5 index range 1..=99999"
        ))
        .into());
    }

    // Validate before opening the file so unrepresentable data cannot truncate it.
    let mut seen = HashSet::with_capacity(nr);
    let mut blocks = Vec::with_capacity(nr);
    for (index, r) in ham_r.rows().into_iter().enumerate() {
        let r = [
            r[0],
            r.get(1).copied().unwrap_or(0),
            r.get(2).copied().unwrap_or(0),
        ];
        if r.iter().any(|x| !(-9_999..=99_999).contains(x)) {
            return Err(invalid(format!("translation {r:?} exceeds I5")).into());
        }
        if !seen.insert(r) {
            return Err(invalid(format!("duplicate translation {r:?}")).into());
        }
        blocks.push((r, index));
    }
    blocks.sort_unstable();
    for (index, z) in matrix.indexed_iter() {
        for value in [z.re, z.im] {
            // Only values near or outside the limits need a formatted width check.
            if !value.is_finite()
                || (!(-9_999.0..=99_999.0).contains(&value) && format!("{value:.6}").len() > 12)
            {
                return Err(
                    invalid(format!("value {value} at {index:?} does not fit F12.6")).into(),
                );
            }
        }
    }

    let mut file = BufWriter::new(File::create(name)?);
    writeln!(file, "#generate by Rustb at time {}", utc_timestamp())?;
    writeln!(file, "{nsta}\n{nr}")?;
    if !position {
        for start in (0..nr).step_by(15) {
            for _ in 0..(nr - start).min(15) {
                write!(file, "    1")?;
            }
            writeln!(file)?;
        }
    }
    let components = if position { 3 } else { 1 };
    for ([rx, ry, rz], index) in blocks {
        for n in 0..nsta {
            for m in 0..nsta {
                write!(file, "{rx:>5}{ry:>5}{rz:>5}{:>5}{:>5}", m + 1, n + 1)?;
                for axis in 0..components {
                    let z = if axis < stored_components {
                        matrix[[index, axis, m, n]]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    write!(file, "{:>12.6}{:>12.6}", z.re, z.im)?;
                }
                writeln!(file)?;
            }
        }
    }
    file.flush()?;
    Ok(())
}

impl<const SPIN: bool, const DIM: usize> Model<SPIN, DIM, HasRMatrix> {
    /// Writes Cartesian position matrices (Å) to `{seedname}_r.dat` as `5I5,6F12.6`.
    ///
    /// Uses the same lexicographic translation order `(Rx, Ry, Rz)` and 1-based
    /// indices as [`OutPut::output_hr`], without a degeneracy section. Absent
    /// Cartesian axes are padded with zero.
    /// Non-finite values and fields that do not fit are rejected.
    pub fn output_r(&self, path: &str, seedname: &str) -> Result<()> {
        write_wannier_matrix::<DIM>(
            &format!("{path}{seedname}_r.dat"),
            &self.hamR,
            self.nsta(),
            self.rmatrix.view(),
            true,
        )
    }
}

pub trait OutPut {
    /// Writes the Hamiltonian to `{seedname}_hr.dat` as `5I5,2F12.6`.
    ///
    /// Translation blocks are sorted by `(Rx, Ry, Rz)` in ascending order
    /// (`Rz` varies fastest). The model is not reordered. Records use unit
    /// degeneracies and 1-based indices. No conjugate blocks are added.
    /// Non-finite values and fields that do not fit are rejected.
    fn output_hr(&self, path: &str, seedname: &str) -> Result<()>;
    fn output_POSCAR(&self, path: &str) -> Result<()>;
    fn output_win(&self, path: &str, seedname: &str) -> Result<()>;
    /// Writes a Wannier90 `_centres.xyz` file.
    ///
    /// Returns [`TbError::MissingAtomicStructure`] when the model has no
    /// atom metadata (e.g. built via `tb_model(..., None)`).
    fn output_xyz(&self, path: &str, seedname: &str) -> Result<()>;
    fn show_band(&self, path: &Array2<f64>, label: &[&str], nk: usize, name: &str) -> Result<()>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> OutPut for Model<SPIN, DIM, R> {
    fn output_hr(&self, path: &str, seedname: &str) -> Result<()> {
        write_wannier_matrix::<DIM>(
            &format!("{path}{seedname}_hr.dat"),
            &self.hamR,
            self.nsta(),
            self.ham.view().insert_axis(Axis(1)),
            false,
        )
    }

    fn output_POSCAR(&self, path: &str) -> Result<()> {
        let name = format!("{path}POSCAR");
        let mut file = File::create(&name)?;
        writeln!(file, "Generate by Rustb")?;
        writeln!(file, "1.0")?;
        let s = match self.dim_r() {
            3 => {
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],self.lat[[0,1]],self.lat[[0,2]],self.lat[[1,0]],self.lat[[1,1]],self.lat[[1,2]],self.lat[[2,0]],self.lat[[2,1]],self.lat[[2,2]]));
                s
            }
            2 => {
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],self.lat[[0,1]],0.0,self.lat[[1,0]],self.lat[[1,1]],0.0,0.0,0.0,10.0));
                s
            }
            1 => {
                let mut s = String::new();
                s.push_str(&format!("    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}\n    {:>15.8}    {:>15.8}    {:>15.8}",self.lat[[0,0]],0.0,0.0,0.0,10.0,0.0,0.0,0.0,10.0));
                s
            }
            _ => unreachable!(),
        };
        writeln!(file, "{}", s)?;
        //开始弄atom
        let mut atom_type = vec![];
        let mut atom_num = vec![];
        let mut new_atom_position: Vec<Vec<Array1<f64>>> = Vec::new();
        for i in 0..self.natom() {
            let mut have_atom = false;
            for j in 0..atom_type.len() {
                if self.atoms[i].atom_type() == atom_type[j] {
                    have_atom = true;
                    atom_num[j] += 1;
                    new_atom_position[j].push(self.atom_position().row(i).to_owned());
                }
            }
            if have_atom == false {
                atom_num.push(1);
                atom_type.push(self.atoms[i].atom_type());
                new_atom_position.push(vec![self.atom_position().row(i).to_owned()]);
            }
        }
        let mut s = String::new();
        for i in 0..atom_type.len() {
            s.push_str(&format!("   {}", atom_type[i]));
        }
        writeln!(file, "{}", s)?;
        let mut s = String::new();
        for i in 0..atom_type.len() {
            s.push_str(&format!("{:>4}", atom_num[i]));
        }
        writeln!(file, "{}", s)?;
        writeln!(file, "Direct")?;
        let _s = String::new();
        for i in 0..atom_type.len() {
            for j in 0..new_atom_position[i].len() {
                let s = match self.dim_r() {
                    3 => {
                        let mut s = String::new();
                        s.push_str(&format!(
                            "{:>15.8}   {:>15.8}   {:>15.8}",
                            new_atom_position[i][j][[0]],
                            new_atom_position[i][j][[1]],
                            new_atom_position[i][j][[2]]
                        ));
                        s
                    }
                    2 => {
                        let mut s = String::new();
                        s.push_str(&format!(
                            "{:>15.8}   {:>15.8}   {:>15.8}",
                            new_atom_position[i][j][[0]],
                            new_atom_position[i][j][[1]],
                            0.0
                        ));
                        s
                    }
                    1 => {
                        let mut s = String::new();
                        s.push_str(&format!(
                            "{:>15.8}   {:>15.8}   {:>15.8}",
                            new_atom_position[i][j][[0]],
                            0.0,
                            0.0
                        ));
                        s
                    }
                    _ => unreachable!(),
                };
                writeln!(file, "{}", s)?;
            }
        }
        Ok(())
    }

    fn output_win(&self, path: &str, seedname: &str) -> Result<()> {
        //!Output a Wannier90 `.win` file. The projection block is derived from
        //!the model Atom ownership, `orb_projection`, and the model orbital order.
        //!Pure and individual hybrid projections are written in that order; no
        //!empty placeholder or out-of-order compact hybrid expansion is emitted.
        let name = format!("{path}{seedname}.win");
        let mut file = File::create(&name)?;
        writeln!(file, "num_wann = {}", self.nsta())?;
        writeln!(
            file,
            "spinors = {}",
            if SPIN { ".true." } else { ".false." }
        )?;
        writeln!(file)?;

        // Atom positions are stored in fractional lattice coordinates, so the
        // Wannier90 block must be `atoms_frac` (previously `atoms_cart` was
        // written with fractional values).
        writeln!(file, "begin atoms_frac")?;
        for at in self.atoms.iter() {
            let atom_position = at.position();
            match self.dim_r() {
                3 => {
                    writeln!(
                        file,
                        "{}  {:>10.6}  {:>10.6}  {:>10.6}",
                        at.atom_type(),
                        atom_position[0],
                        atom_position[1],
                        atom_position[2]
                    )?;
                }
                2 => {
                    writeln!(
                        file,
                        "{}  {:>10.6}  {:>10.6}  {:>10.6}",
                        at.atom_type(),
                        atom_position[0],
                        atom_position[1],
                        0.0
                    )?;
                }
                1 => {
                    writeln!(
                        file,
                        "{}  {:>10.6}  {:>10.6}  {:>10.6}",
                        at.atom_type(),
                        atom_position[0],
                        0.0,
                        0.0
                    )?;
                }
                _ => {
                    return Err(TbError::InvalidDimension {
                        dim: self.dim_r(),
                        supported: vec![1, 2, 3],
                    });
                }
            }
        }
        writeln!(file, "end atoms_frac")?;
        writeln!(file)?;
        writeln!(file, "begin unit_cell_cart")?;
        match self.dim_r() {
            3 => {
                for i in 0..3 {
                    let mut s = String::new();
                    for j in 0..3 {
                        s.push_str(&format!("{:>10.6}  ", self.lat[[i, j]]));
                    }
                    writeln!(file, "{}", s)?;
                }
            }
            2 => {
                for i in 0..2 {
                    let mut s = String::new();
                    for j in 0..2 {
                        s.push_str(&format!("{:>10.6}  ", self.lat[[i, j]]));
                    }
                    s.push_str("   0.000000");
                    writeln!(file, "{}", s)?;
                }
                writeln!(file, "   0.000000     0.000000     1.000000")?;
            }
            1 => {
                let mut s = String::new();
                s.push_str(&format!("{:>10.6}  ", self.lat[[0, 0]]));
                s.push_str("   0.000000     0.000000");
                writeln!(file, "{}", s)?;
                writeln!(file, "   0.000000     0.000000     1.000000")?;
                writeln!(file, "   0.000000     0.000000     1.000000")?;
            }
            _ => {
                return Err(TbError::InvalidDimension {
                    dim: self.dim_r(),
                    supported: vec![1, 2, 3],
                });
            }
        }
        writeln!(file, "end unit_cell_cart")?;
        writeln!(file)?;
        writeln!(file, "begin projections")?;
        let mut entries = Vec::<(AtomType, Vec<&'static str>, [f64; 3])>::new();
        for atom in &self.atoms {
            let mut projections = Vec::with_capacity(atom.norb());
            for orbital in atom.orbitals() {
                let projection = *self.orb_projection.get(orbital.index()).ok_or_else(|| {
                    TbError::InvalidOrbitalProjection(format!(
                        "orbital {} has no entry in orb_projection",
                        orbital.index()
                    ))
                })?;
                projections.push(projection);
            }
            if projections.is_empty() {
                continue;
            }
            let tokens = projections
                .iter()
                .map(|projection| projection_token(*projection))
                .collect();
            let source_position = atom.position_ref();
            let mut position = [0.0f64; 3];
            for (axis, value) in source_position.iter().take(3).enumerate() {
                position[axis] = *value;
            }
            entries.push((atom.atom_type(), tokens, position));
        }
        let mut species = std::collections::BTreeMap::<AtomType, Vec<usize>>::new();
        for (index, (atom_type, _, _)) in entries.iter().enumerate() {
            species.entry(*atom_type).or_default().push(index);
        }
        let use_species_lines = species.values().all(|indices| {
            indices.windows(2).all(|pair| pair[1] == pair[0] + 1)
                && indices
                    .iter()
                    .all(|&index| entries[index].1 == entries[indices[0]].1)
        });
        if use_species_lines {
            let mut written = std::collections::BTreeSet::<AtomType>::new();
            for (atom_type, tokens, _) in &entries {
                if written.insert(*atom_type) {
                    writeln!(file, "{}: {}", atom_type.to_str(), tokens.join(","))?;
                }
            }
        } else {
            for (_atom_type, tokens, position) in &entries {
                writeln!(
                    file,
                    "f={:.8},{:.8},{:.8}: {}",
                    position[0],
                    position[1],
                    position[2],
                    tokens.join(",")
                )?;
            }
        }
        writeln!(file, "end projections")?;
        Ok(())
    }
    fn output_xyz(&self, path: &str, seedname: &str) -> Result<()> {
        if self.natom() == 0 {
            return Err(TbError::MissingAtomicStructure);
        }
        if !(1..=3).contains(&DIM) {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![1, 2, 3],
            });
        }
        let name = format!("{path}{seedname}_centres.xyz");
        let mut file = BufWriter::new(File::create(&name)?);
        let orb_real = self.orb.dot(&self.lat);
        let atom_position_real = self.atom_position().dot(&self.lat);
        writeln!(file, "{}", self.nsta() + self.natom())?;
        writeln!(file, "Wannier centres, written by Rustb")?;

        let mut write_row = |symbol: &str, position: ArrayView1<'_, f64>| {
            let x = position[0];
            let y = position.get(1).copied().unwrap_or(0.0);
            let z = position.get(2).copied().unwrap_or(0.0);
            writeln!(file, "{symbol:<2} {x:>18.8} {y:>16.8} {z:>16.8}")
        };
        for position in orb_real.rows() {
            write_row("X", position)?;
        }
        for (atom, position) in self.atoms.iter().zip(atom_position_real.rows()) {
            write_row(atom.atom_type().to_str(), position)?;
        }
        file.flush()?;
        Ok(())
    }

    ///Quickly plot a band structure using Python (gnuplot), since Rust-side
    ///plotting is less convenient.
    #[allow(non_snake_case)]
    fn show_band(&self, path: &Array2<f64>, label: &[&str], nk: usize, name: &str) -> Result<()> {
        use gnuplot::AutoOption::*;
        use gnuplot::AxesCommon;
        use gnuplot::Tick::*;
        use gnuplot::{Color, Figure, Font, LineStyle, Solid};
        use std::fs::create_dir_all;
        use std::path::Path;
        if path.len_of(Axis(0)) != label.len() {
            return Err(TbError::PathLengthMismatch {
                expected: path.len_of(Axis(0)),
                actual: label.len(),
            });
        }
        let (k_vec, k_dist, k_node) = self.k_path(&path, nk)?;
        let eval = self.solve_band_all_parallel(&k_vec)?;
        create_dir_all(name).map_err(|e| TbError::DirectoryCreation {
            path: name.to_string(),
            message: e.to_string(),
        })?;
        let mut name0 = String::new();
        name0.push_str("./");
        name0.push_str(&name);
        let name = name0;
        let mut band_name = name.clone();
        band_name.push_str("/BAND.dat");
        let band_name = Path::new(&band_name);
        let mut file = File::create(band_name)?;
        for i in 0..nk {
            let mut s = String::new();
            let aa = format!("{:.6}", k_dist[[i]]);
            s.push_str(&aa);
            for j in 0..self.nsta() {
                if eval[[i, j]] >= 0.0 {
                    s.push_str("     ");
                } else {
                    s.push_str("    ");
                }
                let aa = format!("{:.6}", eval[[i, j]]);
                s.push_str(&aa);
            }
            writeln!(file, "{}", s)?;
        }
        let mut k_name = name.clone();
        k_name.push_str("/KLABELS");
        let k_name = Path::new(&k_name);
        let mut file = File::create(k_name)?; //写下高对称点的位置
        for i in 0..path.len_of(Axis(0)) {
            let mut s = String::new();
            let aa = format!("{:.6}", k_node[[i]]);
            s.push_str(&aa);
            s.push_str("      ");
            s.push_str(&label[i]);
            writeln!(file, "{}", s)?;
        }
        let mut py_name = name.clone();
        py_name.push_str("/print.py");
        let py_name = Path::new(&py_name);
        let mut file = File::create(py_name)?;
        writeln!(
            file,
            "import numpy as np\nimport matplotlib.pyplot as plt\ndata=np.loadtxt('BAND.dat')\nk_nodes=[]\nlabel=[]\nf=open('KLABELS')\nfor i in f.readlines():\n    k_nodes.append(float(i.split()[0]))\n    label.append(i.split()[1])\nfig,ax=plt.subplots()\nax.plot(data[:,0],data[:,1:],c='b')\nfor x in k_nodes:\n    ax.axvline(x,c='k')\nax.set_xticks(k_nodes)\nax.set_xticklabels(label)\nax.set_xlim([0,k_nodes[-1]])\nfig.savefig('band.pdf')"
        )?;
        //开始绘制pdf图片
        let mut fg = Figure::new();
        let x: Vec<f64> = k_dist.to_vec();
        let axes = fg.axes2d();
        for i in 0..self.nsta() {
            let y: Vec<f64> = eval.slice(s![.., i]).to_owned().to_vec();
            axes.lines(&x, &y, &[Color("black"), LineStyle(Solid)]);
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(k_node[[k_node.len() - 1]]));
        let label = label;
        let mut show_ticks = Vec::new();
        for i in 0..k_node.len() {
            let A = k_node[[i]];
            let B = label[i];
            show_ticks.push(Major(A, Fix(B)));
        }
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 24.0)],
        );

        let _k_node = k_node.to_vec();
        let mut pdf_name = name.clone();
        pdf_name.push_str("/plot.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show()
            .map_err(|e| TbError::Other(format!("gnuplot: {e}")))?;
        Ok(())
    }
}

pub fn draw_heatmap<A: Data<Elem = f64>>(data: &ArrayBase<A, Ix2>, name: &str) {
    //!Draw a heatmap from a 2D matrix, producing a pixel-based image.
    use gnuplot::{AutoOption::Fix, AxesCommon, Figure, RAINBOW};
    let mut fg = Figure::new();
    let (height, width): (usize, usize) = (data.shape()[0], data.shape()[1]);
    let mut heatmap_data = vec![];

    for j in 0..width {
        for i in 0..height {
            heatmap_data.push(data[(i, j)]);
        }
    }
    let axes = fg.axes2d();
    axes.set_title("Heatmap", &[]);
    axes.set_cb_label("Values", &[]);
    axes.set_palette(RAINBOW);
    axes.image(heatmap_data.iter(), width, height, None, &[]);
    let size = data.shape();
    let axes = axes.set_x_range(Fix(0.0), Fix((size[0] - 1) as f64));
    let axes = axes.set_y_range(Fix(0.0), Fix((size[1] - 1) as f64));
    let _axes = axes.set_aspect_ratio(Fix(1.0));
    fg.set_terminal("pdfcairo", name);
    fg.show().expect("Unable to draw heatmap");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn utc_timestamp_has_iso_shape() {
        let value = utc_timestamp();
        assert_eq!(value.len(), 20, "{value}");
        assert_eq!(&value[4..5], "-");
        assert_eq!(&value[7..8], "-");
        assert_eq!(&value[10..11], "T");
        assert_eq!(&value[13..14], ":");
        assert_eq!(&value[16..17], ":");
        assert!(value.ends_with('Z'));
    }

    #[test]
    fn output_xyz_aligns_centres_and_atoms() {
        use crate::{Atom, OrbitalId};

        let dir = format!("target/test-output/output_xyz_{}/", std::process::id());
        std::fs::create_dir_all(&dir).unwrap();
        let model = Model::<true, 2>::tb_model(
            array![[2.0, 0.5], [0.0, 1.0]],
            array![[0.0, 0.625], [-0.25, 0.375], [0.5, -1_000_000.25]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.5], AtomType::Sr, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![-0.25, 0.25], AtomType::Mn, [OrbitalId::new(1)]),
                Atom::with_orbitals(
                    array![0.5, -1_000_000.375],
                    AtomType::O,
                    [OrbitalId::new(2)],
                ),
            ]),
        )
        .unwrap();
        model.output_xyz(&dir, "wannier90").unwrap();
        let text = std::fs::read_to_string(format!("{dir}wannier90_centres.xyz")).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(
            text.lines().collect::<Vec<_>>(),
            [
                "9",
                "Wannier centres, written by Rustb",
                "X          0.00000000       0.62500000       0.00000000",
                "X         -0.50000000       0.25000000       0.00000000",
                "X          1.00000000 -1000000.00000000       0.00000000",
                "X          0.00000000       0.62500000       0.00000000",
                "X         -0.50000000       0.25000000       0.00000000",
                "X          1.00000000 -1000000.00000000       0.00000000",
                "Sr         0.00000000       0.50000000       0.00000000",
                "Mn        -0.50000000       0.12500000       0.00000000",
                "O          1.00000000 -1000000.12500000       0.00000000",
            ],
        );
    }

    #[test]
    fn output_win_writes_model_projections() {
        use crate::{Atom, AtomType, OrbitalId};
        use ndarray::array;

        let dir = format!("target/test-output/output_win_{}/", std::process::id());
        std::fs::create_dir_all(&dir).unwrap();
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25]
            ],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::Si, [OrbitalId::new(0)]),
                Atom::with_orbitals(
                    array![0.25, 0.25, 0.25],
                    AtomType::Fe,
                    [
                        OrbitalId::new(1),
                        OrbitalId::new(2),
                        OrbitalId::new(3),
                        OrbitalId::new(4),
                    ],
                ),
            ]),
        )
        .unwrap();
        model.orb_projection = vec![
            OrbProj::s,
            OrbProj::s,
            OrbProj::px,
            OrbProj::py,
            OrbProj::pz,
        ];
        model.output_win(&dir, "wannier90").unwrap();
        let text = std::fs::read_to_string(format!("{dir}wannier90.win")).unwrap();
        assert!(text.contains("num_wann = 5"), "{text}");
        assert!(text.contains("spinors = .false."), "{text}");
        assert!(text.contains("Si: s"), "{text}");
        assert!(text.contains("Fe: s,px,py,pz"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn output_win_writes_complete_hybrid_sets() {
        use crate::{Atom, AtomType, OrbitalId};
        use ndarray::array;

        let dir = format!(
            "target/test-output/output_win_hybrid_{}/",
            std::process::id()
        );
        std::fs::create_dir_all(&dir).unwrap();
        let orbitals = (0..4).map(OrbitalId::new).collect::<Vec<_>>();
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            Array2::zeros((4, 3)),
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                orbitals,
            )]),
        )
        .unwrap();
        model.orb_projection = vec![
            OrbProj::sp3_1,
            OrbProj::sp3_2,
            OrbProj::sp3_3,
            OrbProj::sp3_4,
        ];
        model.output_win(&dir, "wannier90").unwrap();
        let text = std::fs::read_to_string(format!("{dir}wannier90.win")).unwrap();
        assert!(text.contains("C: sp3-1,sp3-2,sp3-3,sp3-4"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn output_win_does_not_duplicate_equivalent_species_lines() {
        use crate::{Atom, AtomType, OrbitalId};
        use ndarray::array;

        let dir = format!(
            "target/test-output/output_win_duplicate_{}/",
            std::process::id()
        );
        std::fs::create_dir_all(&dir).unwrap();
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::C, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        model.orb_projection = vec![OrbProj::s, OrbProj::s];
        model.output_win(&dir, "wannier90").unwrap();
        let text = std::fs::read_to_string(format!("{dir}wannier90.win")).unwrap();
        assert_eq!(text.matches("C: s").count(), 1, "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn output_win_marks_spinor_models() {
        use crate::{Atom, AtomType, OrbitalId};
        use ndarray::array;

        let dir = format!(
            "target/test-output/output_win_spinor_{}/",
            std::process::id()
        );
        std::fs::create_dir_all(&dir).unwrap();
        let mut model = Model::<true, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Fe,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.orb_projection = vec![OrbProj::s];
        model.output_win(&dir, "wannier90").unwrap();
        let text = std::fs::read_to_string(format!("{dir}wannier90.win")).unwrap();
        assert!(text.contains("num_wann = 2"), "{text}");
        assert!(text.contains("spinors = .true."), "{text}");
        assert!(text.contains("Fe: s"), "{text}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn output_win_handles_low_dimensional_atom_positions() {
        use crate::{Atom, AtomType, OrbitalId};
        use ndarray::array;

        let dir = format!(
            "target/test-output/output_win_lowdim_{}/",
            std::process::id()
        );
        std::fs::create_dir_all(&dir).unwrap();
        let mut two_d = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.0, 0.0], [0.5, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0], AtomType::C, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        two_d.orb_projection = vec![OrbProj::s, OrbProj::px];
        two_d.output_win(&dir, "two_d").unwrap();

        let mut one_d = Model::<false, 1>::tb_model(
            array![[1.0]],
            array![[0.0], [0.5]],
            Some(vec![
                Atom::with_orbitals(array![0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5], AtomType::C, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        one_d.orb_projection = vec![OrbProj::s, OrbProj::px];
        one_d.output_win(&dir, "one_d").unwrap();

        for seed in ["two_d", "one_d"] {
            let text = std::fs::read_to_string(format!("{dir}{seed}.win")).unwrap();
            assert!(text.contains("num_wann = 2"), "{text}");
            assert!(text.contains("f="), "{text}");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn wannier_matrix_writers_use_fixed_width_and_the_stored_support() {
        let dir = format!(
            "target/test-output/wannier-fixed-width-{}/",
            std::process::id()
        );
        std::fs::create_dir_all(&dir).unwrap();
        let mut model =
            Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), Array2::zeros((2, 3)), None)
                .unwrap();
        let translations = [
            [0, 0, 0],
            [-9_999, 99_999, 0],
            [3, 2, -4],
            [2, -1, 3],
            [8, -1, 3],
            [-2, 4, 1],
            [3, -1, 3],
            [12, 0, -2],
            [3, -1, -2],
            [2, 1, -3],
            [1, 0, 3],
            [-2, -3, 0],
            [5, 0, 0],
            [10, -1, 3],
            [4, -1, 3],
            [7, 1, 0],
        ];
        let ham_r = Array2::from_shape_fn((16, 3), |(b, d)| translations[b][d]);
        model.hamR = ham_r.clone();
        // Original row indices in ascending (Rx, Ry, Rz) order.
        let expected_rows = [1, 11, 5, 0, 10, 3, 9, 8, 6, 2, 14, 12, 15, 4, 13, 7];
        model.ham = Array3::from_shape_fn((16, 2, 2), |(b, m, n)| {
            let key = (100 * b + 10 * m + n) as f64;
            if (b, m, n) == (1, 1, 0) {
                Complex64::new(-9_999.999_999, 99_999.999_999)
            } else {
                Complex64::new(key + 0.123_456_7, -key - 0.765_432_1)
            }
        });
        model.rmatrix = HasRMatrix(Array4::from_shape_fn((16, 3, 2, 2), |(b, d, m, n)| {
            let key = (100 * b + 10 * m + n + 2 * d) as f64;
            Complex64::new(key + 0.234_567_8, -key - 0.876_543_2)
        }));
        model.output_hr(&dir, "wannier90").unwrap();
        model.output_r(&dir, "wannier90").unwrap();
        assert_eq!(model.hamR, ham_r);
        for (suffix, fields) in [("hr", 2), ("r", 6)] {
            let text = std::fs::read_to_string(format!("{dir}wannier90_{suffix}.dat")).unwrap();
            let mut lines = text.lines();
            assert!(
                lines
                    .next()
                    .unwrap()
                    .starts_with("#generate by Rustb at time ")
            );
            assert_eq!(lines.next().unwrap().trim(), "2");
            assert_eq!(lines.next().unwrap().trim(), "16");
            if suffix == "hr" {
                assert_eq!(lines.next().unwrap(), "    1".repeat(15));
                assert_eq!(lines.next().unwrap(), "    1");
            }
            for b in expected_rows {
                for n in 0..2 {
                    for m in 0..2 {
                        let line = lines.next().unwrap();
                        assert_eq!(line.len(), 25 + 12 * fields, "{line}");
                        let i5 =
                            |k: usize| line[5 * k..5 * (k + 1)].trim().parse::<isize>().unwrap();
                        let f12 = |k: usize| {
                            line[25 + 12 * k..25 + 12 * (k + 1)]
                                .trim()
                                .parse::<f64>()
                                .unwrap()
                        };
                        assert_eq!([i5(0), i5(1), i5(2)], translations[b], "{line}");
                        assert_eq!([i5(3), i5(4)], [m as isize + 1, n as isize + 1]);
                        for d in 0..fields / 2 {
                            let key = (100 * b + 10 * m + n + 2 * d) as f64;
                            let (re, im) = if suffix == "r" {
                                (key + 0.234_568, -key - 0.876_543)
                            } else if (b, m, n) == (1, 1, 0) {
                                (-9_999.999_999, 99_999.999_999)
                            } else {
                                (key + 0.123_457, -key - 0.765_432)
                            };
                            assert!((f12(2 * d) - re).abs() < 1e-10, "{line}");
                            assert!((f12(2 * d + 1) - im).abs() < 1e-10, "{line}");
                        }
                    }
                }
            }
            assert!(lines.next().is_none());
        }
        println!("Fixed-width fixtures: {dir}");
    }

    #[test]
    fn wannier_writers_reject_overflow_without_touching_existing_files() {
        let dir = format!(
            "target/test-output/wannier-overflow-{}/",
            std::process::id()
        );
        std::fs::create_dir_all(&dir).unwrap();
        let mut model =
            Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.5]], None).unwrap();
        model.output_hr(&dir, "seed").unwrap();
        model.output_r(&dir, "seed").unwrap();
        let hr_path = format!("{dir}seed_hr.dat");
        let r_path = format!("{dir}seed_r.dat");
        let hr = std::fs::read_to_string(&hr_path).unwrap();
        let r = std::fs::read_to_string(&r_path).unwrap();
        assert_eq!(
            r.lines().last().unwrap(),
            format!(
                "{:>5}{:>5}{:>5}{:>5}{:>5}{:>12.6}{}",
                0,
                0,
                0,
                1,
                1,
                0.5,
                "    0.000000".repeat(5)
            )
        );

        // Rounding to six fractional places itself can overflow the field.
        model.ham[[0, 0, 0]].re = 99_999.999_999_9;
        let error = model.output_hr(&dir, "seed").unwrap_err().to_string();
        assert!(error.contains("F12.6"), "{error}");
        for value in [-9_999.999_999_9, f64::NAN] {
            model.rmatrix[[0, 0, 0, 0]].im = value;
            let error = model.output_r(&dir, "seed").unwrap_err().to_string();
            assert!(error.contains("F12.6"), "{error}");
        }
        model.rmatrix = HasRMatrix(Array4::zeros((2, 1, 1, 1)));
        assert!(
            matches!(model.output_r(&dir, "seed"), Err(TbError::Io(err)) if err.kind() == io::ErrorKind::InvalidInput)
        );
        assert_eq!(std::fs::read_to_string(&r_path).unwrap(), r);
        model.hamR[[0, 0]] = -10_000;
        let error = model.output_hr(&dir, "seed").unwrap_err().to_string();
        assert!(error.contains("I5"), "{error}");
        assert_eq!(std::fs::read_to_string(hr_path).unwrap(), hr);
        assert_eq!(std::fs::read_to_string(r_path).unwrap(), r);
    }
}
