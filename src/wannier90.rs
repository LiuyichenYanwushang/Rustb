//! Wannier90 file import.
//!
//! [`Wannier90::from_hr`] reads a `seedname_hr.dat` hopping file and, when
//! present, `seedname_centres.xyz` centres and `seedname_wsvec.dat` weights.
//! Position matrices are read from `seedname_r.dat` only for a
//! `Model<SPIN, 3, HasRMatrix>` target, and are then required;
//! `Model<SPIN, 3, NoRMatrix>` never opens that file. The two files must declare
//! exactly the same set of translations, but block order can differ. The importer
//! validates declared record counts, indices, weights, sizes and energy shifts before
//! assembling the model, and rejects truncated or inconsistent files instead of
//! guessing.
//!
//! Element labels in the `.win` projection lines, `atoms_cart`/`atoms_frac`
//! blocks and the extra-atom labels of `_centres.xyz` are matched
//! case-insensitively; the orbital token after the colon remains case-sensitive.
//!
//! Orbital labels follow the Wannier90 built-in angular-function order. The
//! canonical base is the order of the delivered projection block: each projection
//! line applies to its matching atoms in atom order (this is Wannier90's state
//! order when the `.win` file matches the run). When `_centres.xyz` is present
//! and every atom's Wannier functions form one atom-centred contiguous block,
//! whole blocks are matched to the nearest atom instead, so a `.win` whose block
//! order differs from the centre file is handled. Bond-centred hybrids and
//! ambiguous blocks fall back to the delivered projection order with a warning;
//! when that happens the labels follow the delivered `.win` order and can be
//! wrong if that file was reordered relative to the run. Two atoms of one species
//! with equal projected counts could in principle exchange a whole block, but
//! that requires every centre of the block to be closer to the other atom; the
//! stored centres and Hamiltonian are unaffected, only atom ownership follows
//! the centres. For spinors, all `2 * N` centres are retained in spin-major
//! order, including distinct down-state centres. Projection labels and atom
//! ownership describe the `N` paired slots and are matched using only the first
//! (up) half; the corresponding down states share those labels and owners.
//!
//! # Examples
//!
//! ```no_run
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! // Reads wannier90_hr.dat from the given directory.
//! let model: Model<false, 3> = Model::from_hr("path/to/seedname/", "wannier90", 0.0)?;
//!
//! // The same files with position matrices, for velocity commutators.
//! let with_positions: Model<false, 3, HasRMatrix> =
//!     Model::from_hr("path/to/seedname/", "wannier90", 0.0)?;
//!
//! assert_eq!(model.nsta(), with_positions.nsta());
//! # Ok(())
//! # }
//! ```

use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
use crate::error::{Result, TbError};
use crate::{Model, RMatrixData, find_R};
use ndarray::prelude::*;
use ndarray_linalg::Inverse;
use num_complex::Complex64;
use std::collections::HashSet;
use std::str::FromStr;

const BOHR_TO_ANGSTROM: f64 = 0.529_177_210_67;

fn parse_error(file: &str, message: impl Into<String>) -> TbError {
    TbError::FileParse {
        file: file.into(),
        message: message.into(),
    }
}

// Only missing optional files are ignored; permission/read errors remain errors.
fn read_optional(file: &str) -> Result<Option<String>> {
    match std::fs::read_to_string(file) {
        Ok(text) => Ok(Some(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn read_required(file: &str) -> Result<String> {
    read_optional(file)?.ok_or_else(|| TbError::FileCreation {
        path: file.into(),
        message: "Required Wannier90 file not found".into(),
    })
}

fn field<'a, T: FromStr>(
    fields: &mut impl Iterator<Item = &'a str>,
    file: &str,
    name: &str,
) -> Result<T> {
    let token = fields
        .next()
        .ok_or_else(|| parse_error(file, format!("Missing {name}")))?;
    token
        .parse()
        .map_err(|_| parse_error(file, format!("Invalid {name}: '{token}'")))
}

fn data_line<'a>(lines: &mut impl Iterator<Item = &'a str>, file: &str) -> Result<&'a str> {
    lines
        .next()
        .ok_or_else(|| parse_error(file, "Truncated file"))
}

fn finite_field<'a>(fields: &mut impl Iterator<Item = &'a str>, file: &str) -> Result<f64> {
    let value: f64 = field(fields, file, "numeric component")?;
    if !value.is_finite() {
        return Err(parse_error(file, "Non-finite numeric component"));
    }
    Ok(value)
}

fn record_key<'a>(
    fields: &mut impl Iterator<Item = &'a str>,
    file: &str,
    nsta: usize,
) -> Result<([isize; 3], usize, usize)> {
    let translation = [
        field(fields, file, "R vector")?,
        field(fields, file, "R vector")?,
        field(fields, file, "R vector")?,
    ];
    let i: usize = field(fields, file, "orbital index")?;
    let j: usize = field(fields, file, "orbital index")?;
    if !(1..=nsta).contains(&i) || !(1..=nsta).contains(&j) {
        return Err(parse_error(
            file,
            format!("Orbital indices ({i}, {j}) must be in 1..={nsta}"),
        ));
    }
    Ok((translation, i - 1, j - 1))
}

fn matrix_record<const N: usize>(
    row: &str,
    file: &str,
    nsta: usize,
) -> Result<(([isize; 3], usize, usize), [Complex64; N])> {
    fn parse<'a, const N: usize>(
        mut fields: impl Iterator<Item = &'a str>,
        file: &str,
        nsta: usize,
    ) -> Result<(([isize; 3], usize, usize), [Complex64; N])> {
        let key = record_key(&mut fields, file, nsta)?;
        let mut values = [Complex64::new(0.0, 0.0); N];
        for value in &mut values {
            *value = Complex64::new(
                finite_field(&mut fields, file)?,
                finite_field(&mut fields, file)?,
            );
        }
        Ok((key, values))
    }

    // Keep the existing whitespace interpretation, including scientific notation.
    let error = match parse(row.split_whitespace(), file, nsta) {
        Ok(record) => return Ok(record),
        Err(error) => error,
    };
    // A full-width field can abut its predecessor in 5I5,2F12.6 or 5I5,6F12.6.
    // Only slice ASCII records of the expected length, retaining leading padding.
    let row = row.trim_end();
    if row.len() != 25 + 24 * N || !row.is_ascii() {
        return Err(error);
    }
    let integers = (0..5).map(|i| row[i * 5..(i + 1) * 5].trim_start_matches(' '));
    let floats = (0..2 * N).map(|i| &row[25 + i * 12..25 + (i + 1) * 12]);
    if !floats.clone().all(|field| {
        let bytes = field.as_bytes();
        bytes[5] == b'.' && bytes[6..].iter().all(u8::is_ascii_digit)
    }) {
        return Err(error);
    }
    parse(
        integers.chain(floats.map(|field| field.trim_start_matches(' '))),
        file,
        nsta,
    )
}

fn matrix_size(file: &str, nr: usize, nsta: usize, components: usize) -> Result<usize> {
    nr.checked_mul(nsta)
        .and_then(|n| n.checked_mul(nsta))
        .and_then(|n| n.checked_mul(components))
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<Complex64>())
        .ok_or_else(|| parse_error(file, "Matrix size overflow"))
}

struct HrData {
    ham: Array3<Complex64>,
    translations: Array2<isize>,
    // Aligned with translations; zero denotes the synthetic origin, if absent in HR.
    weights: Vec<usize>,
}

fn parse_hr(file: &str) -> Result<HrData> {
    let text = read_required(file)?;
    let mut lines = text.lines();
    data_line(&mut lines, file)?;
    let nsta: usize = field(
        &mut data_line(&mut lines, file)?.split_whitespace(),
        file,
        "state count",
    )?;
    let nr: usize = field(
        &mut data_line(&mut lines, file)?.split_whitespace(),
        file,
        "R count",
    )?;
    if nsta == 0 || nr == 0 {
        return Err(parse_error(file, "State and R counts must be positive"));
    }
    let records = matrix_size(file, nr, nsta, 1)?;
    matrix_size(
        file,
        nr.checked_add(1)
            .ok_or_else(|| parse_error(file, "R count overflow"))?,
        nsta,
        1,
    )?;
    let mut weights = Vec::new();
    while weights.len() < nr {
        let mut fields = data_line(&mut lines, file)?.split_whitespace();
        while fields.clone().next().is_some() {
            let weight: usize = field(&mut fields, file, "degeneracy weight")?;
            if weight == 0 || weights.len() == nr {
                return Err(parse_error(
                    file,
                    "Invalid degeneracy weight count or zero weight",
                ));
            }
            weights.push(weight);
        }
    }
    let rows = lines.collect::<Vec<_>>();
    if rows.len() < records {
        return Err(parse_error(file, "Truncated Hamiltonian records"));
    }
    if rows[records..].iter().any(|row| !row.trim().is_empty()) {
        return Err(parse_error(file, "Extra Hamiltonian records"));
    }
    let mut data = HrData {
        ham: Array3::zeros((1, nsta, nsta)),
        translations: Array2::zeros((1, 3)),
        weights: vec![0],
    };
    let mut seen_translations = HashSet::new();
    for (block, weight) in weights.into_iter().enumerate() {
        let start = block * nsta * nsta;
        let block = &rows[start..start + nsta * nsta];
        let first = matrix_record::<1>(block[0], file, nsta)?;
        let ((translation, _, _), _) = first;
        if !seen_translations.insert(translation) {
            return Err(parse_error(file, "Duplicate R block"));
        }
        let index = if translation == [0; 3] {
            0
        } else {
            let index = data.translations.nrows();
            data.translations.push_row(ArrayView1::from(&translation))?;
            data.ham.push(Axis(0), Array2::zeros((nsta, nsta)).view())?;
            data.weights.push(0);
            index
        };
        data.weights[index] = weight;
        let mut seen = HashSet::new();
        let rest = block[1..]
            .iter()
            .map(|row| matrix_record::<1>(row, file, nsta));
        for record in std::iter::once(Ok(first)).chain(rest) {
            let ((r, i, j), [value]) = record?;
            if r != translation || !seen.insert((i, j)) {
                return Err(parse_error(
                    file,
                    "Inconsistent R block or duplicate orbital record",
                ));
            }
            // The first Wannier90 index is the matrix row, regardless of record order.
            data.ham[[index, i, j]] = value / weight as f64;
        }
    }
    Ok(data)
}

fn parse_rmatrix(file: &str, hr: &HrData) -> Result<Array4<Complex64>> {
    let text = read_required(file)?;
    let mut lines = text.lines();
    data_line(&mut lines, file)?;
    let nsta: usize = field(
        &mut data_line(&mut lines, file)?.split_whitespace(),
        file,
        "state count",
    )?;
    let nr: usize = field(
        &mut data_line(&mut lines, file)?.split_whitespace(),
        file,
        "R count",
    )?;
    if nsta != hr.ham.shape()[1] {
        return Err(parse_error(file, "Position-matrix state count mismatch"));
    }
    // A zero weight marks an origin row synthesized by Rustb, not by _hr.dat.
    let expected_nr = hr.weights.iter().filter(|&&weight| weight > 0).count();
    if nr != expected_nr {
        return Err(parse_error(
            file,
            format!(
                "Position-matrix R count mismatch: _r.dat declares {nr}, _hr.dat declares {expected_nr}"
            ),
        ));
    }
    let records = matrix_size(file, nr, nsta, 1)?;
    matrix_size(file, hr.translations.nrows(), nsta, 3)?;
    let rows = lines.collect::<Vec<_>>();
    if rows.len() < records {
        return Err(parse_error(file, "Truncated position-matrix records"));
    }
    if rows[records..].iter().any(|row| !row.trim().is_empty()) {
        return Err(parse_error(file, "Extra position-matrix records"));
    }
    let mut result = Array4::zeros((hr.translations.nrows(), 3, nsta, nsta));
    // Equal counts plus unique blocks found in _hr.dat enforce set equality.
    let mut seen_translations = HashSet::new();
    let mut seen_orbitals = HashSet::new();
    for block in rows[..records].chunks(nsta * nsta) {
        let first = matrix_record::<3>(block[0], file, nsta)?;
        let ((translation, _, _), _) = first;
        if !seen_translations.insert(translation) {
            return Err(parse_error(
                file,
                format!("Duplicate position-matrix R block {translation:?}"),
            ));
        }
        let index = find_R(&hr.translations, &Array1::from_vec(translation.to_vec()))
            .filter(|&i| hr.weights[i] > 0)
            .ok_or_else(|| {
                parse_error(
                    file,
                    format!("Position-matrix R vector {translation:?} not found in _hr.dat"),
                )
            })?;
        seen_orbitals.clear();
        let rest = block[1..]
            .iter()
            .map(|row| matrix_record::<3>(row, file, nsta));
        for record in std::iter::once(Ok(first)).chain(rest) {
            let ((r, i, j), values) = record?;
            if r != translation || !seen_orbitals.insert((i, j)) {
                return Err(parse_error(
                    file,
                    "Inconsistent R block or duplicate orbital record",
                ));
            }
            for (axis, value) in values.into_iter().enumerate() {
                result[[index, axis, i, j]] = value / hr.weights[index] as f64;
            }
        }
    }
    Ok(result)
}

fn win_data_line(line: &str) -> &str {
    line.split(['!', '#']).next().unwrap_or_default().trim()
}

fn length_unit_scale(token: &str) -> Option<f64> {
    match token.to_ascii_lowercase().as_str() {
        "ang" | "angstrom" | "angstroms" => Some(1.0),
        "bohr" => Some(BOHR_TO_ANGSTROM),
        _ => None,
    }
}

struct WinData {
    lat: Array2<f64>,
    spin: bool,
    projections: Vec<(AtomType, Vec<OrbProj>)>,
    atoms: Vec<(AtomType, Array1<f64>)>,
}

fn parse_win(file: &str) -> Result<WinData> {
    let text = read_required(file)?;
    let mut lines = text
        .lines()
        .map(win_data_line)
        .filter(|line| !line.is_empty());
    let mut data = WinData {
        lat: Array2::zeros((3, 3)),
        spin: false,
        projections: Vec::new(),
        atoms: Vec::new(),
    };
    let mut fractional_atoms = Vec::new();
    while let Some(line) = lines.next() {
        let keyword = line.to_ascii_lowercase();
        if keyword == "begin unit_cell_cart" {
            let mut line = data_line(&mut lines, file)?;
            let scale = if let Some(scale) = length_unit_scale(line) {
                line = data_line(&mut lines, file)?;
                scale
            } else {
                1.0
            };
            for row in 0..3 {
                let mut fields = line.split_whitespace();
                for axis in 0..3 {
                    data.lat[[row, axis]] = finite_field(&mut fields, file)? * scale;
                }
                if fields.next().is_some() {
                    return Err(parse_error(
                        file,
                        "Lattice row must contain three components",
                    ));
                }
                line = data_line(&mut lines, file)?;
            }
            if !line.eq_ignore_ascii_case("end unit_cell_cart") {
                return Err(parse_error(file, "Missing end unit_cell_cart"));
            }
        } else if keyword.starts_with("spinors") {
            let value = keyword
                .trim_start_matches("spinors")
                .trim()
                .trim_start_matches(['=', ':'])
                .trim();
            data.spin = match value {
                "true" | ".true." | "t" => true,
                "false" | ".false." | "f" => false,
                _ => return Err(parse_error(file, "Invalid spinors value")),
            };
        } else if keyword == "begin projections" {
            loop {
                let line = data_line(&mut lines, file)?;
                if line.eq_ignore_ascii_case("end projections") {
                    break;
                }
                let mut fields = line.split([',', ';', ':']).map(str::trim);
                let species = fields.next().unwrap_or_default();
                let atom_type = species.parse::<AtomType>().map_err(|_| {
                    parse_error(
                        file,
                        format!("Unknown atomic species '{species}' in begin projections"),
                    )
                })?;
                let mut projections = Vec::new();
                for item in fields {
                    let orbitals: &[OrbProj] = match item {
                        "s" => &[OrbProj::s],
                        "p" => &[OrbProj::pz, OrbProj::px, OrbProj::py],
                        "d" => &[
                            OrbProj::dz2,
                            OrbProj::dxz,
                            OrbProj::dyz,
                            OrbProj::dx2y2,
                            OrbProj::dxy,
                        ],
                        "f" => &[
                            OrbProj::fz3,
                            OrbProj::fxz2,
                            OrbProj::fyz2,
                            OrbProj::fzx2y2,
                            OrbProj::fxyz,
                            OrbProj::fxx23y2,
                            OrbProj::fy3x2y2,
                        ],
                        "sp3" => &[
                            OrbProj::sp3_1,
                            OrbProj::sp3_2,
                            OrbProj::sp3_3,
                            OrbProj::sp3_4,
                        ],
                        "sp2" => &[OrbProj::sp2_1, OrbProj::sp2_2, OrbProj::sp2_3],
                        "sp" => &[OrbProj::sp_1, OrbProj::sp_2],
                        "sp3d" => &[
                            OrbProj::sp3d_1,
                            OrbProj::sp3d_2,
                            OrbProj::sp3d_3,
                            OrbProj::sp3d_4,
                            OrbProj::sp3d_5,
                        ],
                        "sp3d2" => &[
                            OrbProj::sp3d2_1,
                            OrbProj::sp3d2_2,
                            OrbProj::sp3d2_3,
                            OrbProj::sp3d2_4,
                            OrbProj::sp3d2_5,
                            OrbProj::sp3d2_6,
                        ],
                        "px" => &[OrbProj::px],
                        "py" => &[OrbProj::py],
                        "pz" => &[OrbProj::pz],
                        "dxy" => &[OrbProj::dxy],
                        "dxz" => &[OrbProj::dxz],
                        "dyz" => &[OrbProj::dyz],
                        "dz2" => &[OrbProj::dz2],
                        "dx2-y2" => &[OrbProj::dx2y2],
                        "fz3" => &[OrbProj::fz3],
                        "fxz2" => &[OrbProj::fxz2],
                        "fyz2" => &[OrbProj::fyz2],
                        "fxyz" => &[OrbProj::fxyz],
                        "fz(x2-y2)" | "fzx2y2" => &[OrbProj::fzx2y2],
                        "fx(x2-3y2)" | "fxx2-3y2" => &[OrbProj::fxx23y2],
                        "fy(3x2-y2)" | "fy3x2-y2" => &[OrbProj::fy3x2y2],
                        "sp-1" => &[OrbProj::sp_1],
                        "sp-2" => &[OrbProj::sp_2],
                        "sp2-1" => &[OrbProj::sp2_1],
                        "sp2-2" => &[OrbProj::sp2_2],
                        "sp2-3" => &[OrbProj::sp2_3],
                        "sp3-1" => &[OrbProj::sp3_1],
                        "sp3-2" => &[OrbProj::sp3_2],
                        "sp3-3" => &[OrbProj::sp3_3],
                        "sp3-4" => &[OrbProj::sp3_4],
                        "sp3d-1" => &[OrbProj::sp3d_1],
                        "sp3d-2" => &[OrbProj::sp3d_2],
                        "sp3d-3" => &[OrbProj::sp3d_3],
                        "sp3d-4" => &[OrbProj::sp3d_4],
                        "sp3d-5" => &[OrbProj::sp3d_5],
                        "sp3d2-1" => &[OrbProj::sp3d2_1],
                        "sp3d2-2" => &[OrbProj::sp3d2_2],
                        "sp3d2-3" => &[OrbProj::sp3d2_3],
                        "sp3d2-4" => &[OrbProj::sp3d2_4],
                        "sp3d2-5" => &[OrbProj::sp3d2_5],
                        "sp3d2-6" => &[OrbProj::sp3d2_6],
                        _ => {
                            return Err(TbError::InvalidOrbitalProjection(format!(
                                "Unrecognized projection '{item}' in seedname.win"
                            )));
                        }
                    };
                    projections.extend_from_slice(orbitals);
                }
                if projections.is_empty() {
                    return Err(parse_error(
                        file,
                        "Projection line must contain species:orbital",
                    ));
                }
                data.projections.push((atom_type, projections));
            }
        } else if keyword == "begin atoms_cart" || keyword == "begin atoms_frac" {
            let fractional = keyword == "begin atoms_frac";
            let end = if fractional {
                "end atoms_frac"
            } else {
                "end atoms_cart"
            };
            let mut first = true;
            let mut scale = 1.0;
            loop {
                let line = data_line(&mut lines, file)?;
                if line.eq_ignore_ascii_case(end) {
                    break;
                }
                if first
                    && !fractional
                    && let Some(unit) = length_unit_scale(line)
                {
                    scale = unit;
                    first = false;
                    continue;
                }
                first = false;
                let mut fields = line.split_whitespace();
                let species: AtomType = field(&mut fields, file, "atomic species")?;
                let position = array![
                    finite_field(&mut fields, file)? * scale,
                    finite_field(&mut fields, file)? * scale,
                    finite_field(&mut fields, file)? * scale
                ];
                if fields.next().is_some() {
                    return Err(parse_error(
                        file,
                        "Atom row must contain a species and three coordinates",
                    ));
                }
                fractional_atoms.push(fractional);
                data.atoms.push((species, position));
            }
        }
    }
    // Defer coordinate conversion until the lattice has been read, allowing either block order.
    let inverse = data.lat.inv()?;
    for ((_, position), fractional) in data.atoms.iter_mut().zip(fractional_atoms) {
        if !fractional {
            *position = position.dot(&inverse);
        }
    }
    Ok(data)
}

fn match_orbitals(
    file: &str,
    win: &WinData,
    nsta: usize,
) -> Result<(Array2<f64>, Vec<OrbProj>, Vec<Atom>)> {
    let norb = if win.spin { nsta / 2 } else { nsta };
    let text = read_optional(file)?;
    let mut centres = None;
    let mut xyz_atoms = Vec::new();
    if let Some(text) = &text {
        let mut lines = text.lines();
        let entries: usize = field(
            &mut data_line(&mut lines, file)?.split_whitespace(),
            file,
            "XYZ entry count",
        )?;
        data_line(&mut lines, file)?;
        if entries < nsta || lines.clone().count() < entries {
            return Err(parse_error(file, "Truncated _centres.xyz entries"));
        }
        let inverse = win.lat.inv()?;
        let mut orb = Array2::zeros((nsta, 3));
        for index in 0..entries {
            let mut fields = data_line(&mut lines, file)?.split_whitespace();
            let species: String = field(&mut fields, file, "XYZ label")?;
            let position = array![
                finite_field(&mut fields, file)?,
                finite_field(&mut fields, file)?,
                finite_field(&mut fields, file)?
            ]
            .dot(&inverse);
            if index < nsta {
                orb.row_mut(index).assign(&position);
            }
            if index >= nsta {
                xyz_atoms.push((
                    species.parse::<AtomType>().map_err(|_| {
                        parse_error(
                            file,
                            format!("Unknown atomic species '{species}' in _centres.xyz"),
                        )
                    })?,
                    position,
                ));
            }
        }
        centres = Some(orb);
    }
    let atoms = if text.is_some() {
        &xyz_atoms
    } else {
        &win.atoms
    };
    if win.projections.is_empty() {
        // Wannier90 input without a projections block (for example a dataset
        // written by a code that only records the centres): assign each paired
        // slot using its first/up centre. The orbital characters are
        // unknown, so every projection is reported as `s`; callers that need
        // the characters must supply a real projections block.
        let centres = centres.ok_or_else(|| {
            parse_error(
                file,
                "an empty projections block needs _centres.xyz to assign Wannier functions",
            )
        })?;
        let mut per_atom: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];
        for (index, centre) in centres.rows().into_iter().take(norb).enumerate() {
            let mut best = (usize::MAX, f64::INFINITY);
            for (atom_index, (_, position)) in atoms.iter().enumerate() {
                let mut delta = Array1::<f64>::zeros(3);
                for axis in 0..3 {
                    delta[axis] = centre[axis] - position[axis];
                    delta[axis] -= delta[axis].round();
                }
                let cartesian = delta.dot(&win.lat);
                let distance = cartesian
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if distance < best.1 {
                    best = (atom_index, distance);
                }
            }
            if best.0 == usize::MAX || best.1 > 1.0 {
                return Err(parse_error(
                    file,
                    format!(
                        "Wannier function {index} is {:.3} A from the nearest atom",
                        best.1
                    ),
                ));
            }
            per_atom[best.0].push(index);
        }
        let mut assigned = Vec::new();
        for (atom_index, (species, position)) in atoms.iter().enumerate() {
            if per_atom[atom_index].is_empty() {
                continue;
            }
            assigned.push(Atom::with_orbitals(
                position.clone(),
                *species,
                per_atom[atom_index]
                    .iter()
                    .map(|&index| OrbitalId::new(index)),
            ));
        }
        return Ok((centres, vec![OrbProj::s; norb], assigned));
    }
    // Build the projected orbital list of every atom in Wannier90 order. Each
    // projection token already carries the built-in angular-function order
    // (p: pz, px, py; d: dz2, dxz, dyz, dx2-y2, dxy; ...), so these labels match
    // the state order that Wannier90 writes.
    let mut atom_projections: Vec<Vec<OrbProj>> = vec![Vec::new(); atoms.len()];
    for (projected_species, orbitals) in &win.projections {
        for (atom_index, (species, _)) in atoms.iter().enumerate() {
            if species == projected_species {
                atom_projections[atom_index].extend_from_slice(orbitals);
            }
        }
    }
    // Wannier90 writes Wannier functions in projection-block order: each
    // projection line applies to its matching atoms in atom order. This
    // expansion is the authoritative fallback when the centres cannot be
    // grouped into one atom-centred block per atom.
    let mut ordered_atoms = Vec::with_capacity(norb);
    for (projected_species, orbitals) in &win.projections {
        for (atom_index, (species, _)) in atoms.iter().enumerate() {
            if species == projected_species {
                for orbital in orbitals {
                    ordered_atoms.push((atom_index, *orbital));
                }
            }
        }
    }
    if ordered_atoms.len() != norb {
        return Err(parse_error(
            file,
            format!(
                "species mismatch between atom data and projections: HR declares {norb} orbitals, but {} could be assigned to atoms",
                ordered_atoms.len()
            ),
        ));
    }
    let mut projections = ordered_atoms
        .iter()
        .map(|(_, orbital)| *orbital)
        .collect::<Vec<_>>();
    let mut per_atom: Vec<Vec<usize>> = vec![Vec::new(); atoms.len()];
    for (index, (atom_index, _)) in ordered_atoms.iter().enumerate() {
        per_atom[*atom_index].push(index);
    }
    let mut fallback = Array2::zeros((0, 3));
    if centres.is_none() {
        // Only absent source centres use common atom centres in both halves.
        for _ in 0..(1 + usize::from(win.spin)) {
            for (atom_index, _) in &ordered_atoms {
                fallback.push_row(atoms[*atom_index].1.view())?;
            }
        }
    } else {
        // Prefer whole-centre-block assignment when every atom's projected
        // orbitals form one contiguous block: this handles delivered .win/HR
        // pairs whose centre blocks are not in atom order. Bond-centred or
        // otherwise non-atom-centred Wannier functions fall back to the
        // projection-block order above.
        let centres = centres.as_ref().expect("checked above");
        let mut nearest_atom = Vec::with_capacity(norb);
        let mut atom_centred = true;
        for index in 0..norb {
            let centre = centres.row(index);
            let mut best = (usize::MAX, f64::INFINITY);
            for (atom_index, (_, position)) in atoms.iter().enumerate() {
                let mut delta = [0.0_f64; 3];
                for axis in 0..3 {
                    delta[axis] = centre[axis] - position[axis];
                    delta[axis] -= delta[axis].round();
                }
                let mut cartesian = [0.0_f64; 3];
                for axis in 0..3 {
                    for column in 0..3 {
                        cartesian[column] += delta[axis] * win.lat[[axis, column]];
                    }
                }
                let distance = cartesian
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    .sqrt();
                if distance < best.1 {
                    best = (atom_index, distance);
                }
            }
            if best.0 == usize::MAX {
                return Err(parse_error(
                    file,
                    "Wannier centres require at least one atom",
                ));
            }
            atom_centred &= best.1 <= 1.0;
            nearest_atom.push(best.0);
        }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for (index, &atom_index) in nearest_atom.iter().enumerate() {
            match blocks.last_mut() {
                Some((last_atom, _, len)) if *last_atom == atom_index => *len += 1,
                _ => blocks.push((atom_index, index, 1)),
            }
        }
        let mut block_valid = atom_centred;
        if block_valid {
            let mut seen = vec![false; atoms.len()];
            for &(atom_index, _, len) in &blocks {
                if atom_projections[atom_index].len() != len || seen[atom_index] {
                    block_valid = false;
                    break;
                }
                seen[atom_index] = true;
            }
            if block_valid {
                block_valid = atom_projections
                    .iter()
                    .enumerate()
                    .all(|(atom_index, orbitals)| orbitals.is_empty() || seen[atom_index]);
            }
        }
        if block_valid {
            projections = vec![OrbProj::s; norb];
            per_atom = vec![Vec::new(); atoms.len()];
            for &(atom_index, start, len) in &blocks {
                projections[start..start + len].copy_from_slice(&atom_projections[atom_index]);
                per_atom[atom_index] = (start..start + len).collect();
            }
        } else {
            let reason = if atom_centred {
                "the Wannier centre blocks do not form one contiguous block per projected atom"
            } else {
                "some Wannier centres are more than 1 A from their nearest atom"
            };
            eprintln!(
                "warning: {reason}; using the order of the delivered projection block to assign atoms and orbitals"
            );
        }
    }
    let mut assigned = Vec::new();
    let mut dropped = HashSet::new();
    for (atom_index, (species, position)) in atoms.iter().enumerate() {
        if per_atom[atom_index].is_empty() {
            if atom_projections[atom_index].is_empty() && dropped.insert(species.to_str()) {
                eprintln!(
                    "warning: dropping '{species}' atoms (no orbitals in the projections block)"
                );
            }
            continue;
        }
        assigned.push(Atom::with_orbitals(
            position.clone(),
            *species,
            per_atom[atom_index]
                .iter()
                .map(|&index| OrbitalId::new(index)),
        ));
    }
    Ok((centres.unwrap_or(fallback), projections, assigned))
}

fn adjust_support(
    file: &str,
    hr: &mut HrData,
    rmatrix: &mut Option<Array4<Complex64>>,
) -> Result<()> {
    let Some(text) = read_optional(file)? else {
        return Ok(());
    };
    let nsta = hr.ham.shape()[1];
    let records = matrix_size(
        file,
        hr.weights.iter().filter(|&&weight| weight > 0).count(),
        nsta,
        1,
    )?;
    let mut lines = text.lines();
    data_line(&mut lines, file)?;
    let mut new_r = Array2::zeros((1, 3));
    let mut new_ham = Array3::<Complex64>::zeros((1, nsta, nsta));
    let mut new_rmatrix = rmatrix
        .as_ref()
        .map(|_| Array4::<Complex64>::zeros((1, 3, nsta, nsta)));
    let mut seen = HashSet::new();
    for _ in 0..records {
        let (translation, i, j) = record_key(
            &mut data_line(&mut lines, file)?.split_whitespace(),
            file,
            nsta,
        )?;
        if !seen.insert((translation, i, j)) {
            return Err(parse_error(file, "Duplicate wsvec record"));
        }
        let source = find_R(&hr.translations, &Array1::from_vec(translation.to_vec()))
            .filter(|&index| hr.weights[index] > 0)
            .ok_or_else(|| parse_error(file, "wsvec R vector not found in Hamiltonian"))?;
        let weight: usize = field(
            &mut data_line(&mut lines, file)?.split_whitespace(),
            file,
            "wsvec multiplicity",
        )?;
        if weight == 0 {
            return Err(parse_error(file, "Zero wsvec multiplicity"));
        }
        for _ in 0..weight {
            let mut fields = data_line(&mut lines, file)?.split_whitespace();
            let mut shifted = Array1::zeros(3);
            for axis in 0..3 {
                shifted[axis] = translation[axis]
                    .checked_add(field(&mut fields, file, "wsvec shift")?)
                    .ok_or_else(|| parse_error(file, "wsvec translation overflow"))?;
            }
            let target = if let Some(index) = find_R(&new_r, &shifted) {
                index
            } else {
                let index = new_r.nrows();
                matrix_size(
                    file,
                    index
                        .checked_add(1)
                        .ok_or_else(|| parse_error(file, "R count overflow"))?,
                    nsta,
                    if rmatrix.is_some() { 3 } else { 1 },
                )?;
                new_r.push_row(shifted.view())?;
                new_ham.push(Axis(0), Array2::zeros((nsta, nsta)).view())?;
                if let Some(matrix) = &mut new_rmatrix {
                    matrix.push(Axis(0), Array3::zeros((3, nsta, nsta)).view())?;
                }
                index
            };
            new_ham[[target, i, j]] += hr.ham[[source, i, j]] / weight as f64;
            if let (Some(old), Some(new)) = (rmatrix.as_ref(), &mut new_rmatrix) {
                for axis in 0..3 {
                    new[[target, axis, i, j]] += old[[source, axis, i, j]] / weight as f64;
                }
            }
        }
    }
    if lines.any(|line| !line.trim().is_empty()) {
        return Err(parse_error(file, "Extra wsvec records"));
    }
    hr.ham = new_ham;
    hr.translations = new_r;
    *rmatrix = new_rmatrix;
    Ok(())
}

fn assemble_model<const SPIN: bool, const DIM: usize, R: RMatrixData>(
    win: WinData,
    orbitals: (Array2<f64>, Vec<OrbProj>, Vec<Atom>),
    mut hr: HrData,
    mut rmatrix: Option<Array4<Complex64>>,
    zero_energy: f64,
) -> Result<Model<SPIN, DIM, R>> {
    let nsta = hr.ham.shape()[1];
    for i in 0..nsta {
        hr.ham[[0, i, i]] -= zero_energy;
    }
    if let Some(matrix) = &mut rmatrix {
        for r in 0..hr.translations.nrows() {
            let translation = hr.translations.row(r);
            let opposite = translation
                .iter()
                .map(|x| x.checked_neg())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| parse_error("_r.dat", "Conjugate translation overflow"))?;
            let partner =
                find_R(&hr.translations, &Array1::from_vec(opposite)).ok_or_else(|| {
                    TbError::MissingHermitianConjugate {
                        r: translation.to_owned(),
                    }
                })?;
            if partner < r {
                continue;
            }
            for axis in 0..3 {
                for i in 0..nsta {
                    for j in 0..nsta {
                        if partner == r && j < i {
                            continue;
                        }
                        let value =
                            (matrix[[r, axis, i, j]] + matrix[[partner, axis, j, i]].conj()) / 2.0;
                        matrix[[r, axis, i, j]] = value;
                        matrix[[partner, axis, j, i]] = value.conj();
                    }
                }
            }
        }
    }
    let (orb, orb_projection, atoms) = orbitals;
    let model = Model {
        lat: win.lat,
        orb,
        orb_projection,
        atoms,
        ham: hr.ham,
        hamR: hr.translations,
        rmatrix: R::from_array(rmatrix.unwrap_or_else(|| Array4::zeros((0, 0, 0, 0)))),
    };
    model.validate()?;
    Ok(model)
}

/// Trait for loading a tight-binding model from Wannier90 output files.
///
/// `DIM` must be 3. `HasRMatrix` requires `_r.dat` (`write_rmn=true`) with
/// exactly the same translations as `_hr.dat`, in any block order. `NoRMatrix`
/// skips `_r.dat`. `_centres.xyz` and `_wsvec.dat` are optional; without
/// centres, the atomic positions from `.win` supply the orbital positions
/// (duplicated in spin-major order for spinors). Present centre files supply
/// every state centre exactly; the down half is never truncated or averaged.
pub trait Wannier90 {
    fn from_hr(path: &str, file_name: &str, zero_energy: f64) -> Result<Self>
    where
        Self: Sized;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Wannier90 for Model<SPIN, DIM, R> {
    fn from_hr(path: &str, file_name: &str, zero_energy: f64) -> Result<Self> {
        if DIM != 3 {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![3],
            });
        }
        if !zero_energy.is_finite() {
            return Err(TbError::Other(
                "Wannier90 zero_energy must be finite".into(),
            ));
        }
        let prefix = format!("{path}{file_name}");
        let mut hr = parse_hr(&format!("{prefix}_hr.dat"))?;
        let win = parse_win(&format!("{prefix}.win"))?;
        if win.spin != SPIN {
            return Err(TbError::Other(format!(
                "Spin mismatch: Wannier90 .win file has spin={} but Model was constructed with SPIN={SPIN}",
                win.spin
            )));
        }
        let nsta = hr.ham.shape()[1];
        if SPIN && nsta % 2 != 0 {
            return Err(parse_error(
                &format!("{prefix}_hr.dat"),
                "Spinful state count must be even",
            ));
        }
        let orbitals = match_orbitals(&format!("{prefix}_centres.xyz"), &win, nsta)?;
        let mut rmatrix = if R::HAS_RMATRIX {
            Some(parse_rmatrix(&format!("{prefix}_r.dat"), &hr)?)
        } else {
            None
        };
        adjust_support(&format!("{prefix}_wsvec.dat"), &mut hr, &mut rmatrix)?;
        assemble_model(win, orbitals, hr, rmatrix, zero_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HasRMatrix;
    use std::fs;
    use std::io::Write;

    /// Write a minimal valid Wannier90 dataset (one C atom, one s orbital)
    /// to `dir/seedname.*`, returning the directory name.
    fn write_minimal_dataset(dir: &str, atom_species: &str) {
        fs::create_dir_all(dir).unwrap();
        let win = format!(
            "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin projections\nC:s\nend projections\n"
        );
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        let xyz = format!("2\nWannier centres\nC 0.0 0.0 0.0\n{atom_species} 0.0 0.0 0.0\n");
        for (suffix, content) in [("_hr.dat", hr), ("_centres.xyz", xyz.as_str())] {
            let mut f = fs::File::create(format!("{dir}/seedname{suffix}")).unwrap();
            f.write_all(content.as_bytes()).unwrap();
        }
        let mut f = fs::File::create(format!("{dir}/seedname.win")).unwrap();
        f.write_all(win.as_bytes()).unwrap();
    }

    #[test]
    fn from_hr_loads_minimal_dataset() {
        let dir = "tests/tmp_w90_ok/";
        write_minimal_dataset(dir, "C");
        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 1);
        assert_eq!(model.natom(), 1);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_loads_adjacent_fixed_width_matrix_fields() {
        let dir = "tests/tmp_w90_adjacent_fields/";
        write_minimal_dataset(dir, "C");
        let minus = "-1234-2345    0    1    1";
        let plus = " 1234 2345    0    1    1";
        let hr = format!(
            "generated\n1\n3\n2 1 2\n{minus}12345.000000-1234.000000\n0 0 0 1 1 2.5e0 0E0\n{plus}12345.000000 1234.000000\n"
        );
        let hr_file = format!("{dir}seedname_hr.dat");
        fs::write(&hr_file, &hr).unwrap();
        fs::write(
            format!("{dir}seedname_r.dat"),
            format!(
                "generated\n1\n3\n{plus}12345.000000 1234.00000023456.000000 2345.00000034567.000000 3456.000000\n0 0 0 1 1 1e0 0 2E0 0 3e0 0\n{minus}12345.000000-1234.00000023456.000000-2345.00000034567.000000-3456.000000\n"
            ),
        )
        .unwrap();
        let model = Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(
            model.hamR,
            array![[0, 0, 0], [-1234, -2345, 0], [1234, 2345, 0]]
        );
        assert_eq!(model.ham[[0, 0, 0]], Complex64::new(2.5, 0.0));
        assert_eq!(model.ham[[1, 0, 0]], Complex64::new(6172.5, -617.0));
        for (axis, (re, im)) in [(6172.5, -617.0), (11728.0, -1172.5), (17283.5, -1728.0)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(model.rmatrix[[1, axis, 0, 0]], Complex64::new(re, im));
        }

        // Keep the byte width but place a multibyte character across the first field boundary.
        fs::write(&hr_file, hr.replacen("4-", "é", 1)).unwrap();
        assert!(matches!(
            Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0),
            Err(TbError::FileParse { .. })
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_assigns_orbitals_by_centres_without_projections() {
        // A dataset whose .win has an empty projections block: every Wannier
        // function is attached to its nearest atom and reported as `s`.
        let dir = "tests/tmp_w90_no_projections/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin atoms_frac\nC 0.0 0.0 0.0\nC 0.5 0.0 0.0\nend atoms_frac\n\nbegin projections\nend projections\n";
        let hr = "generated\n2\n1\n1\n0 0 0 1 1 0.0 0.0\n0 0 0 1 2 0.0 0.0\n0 0 0 2 1 0.0 0.0\n0 0 0 2 2 0.0 0.0\n";
        let xyz =
            "4\nWannier centres\nX 0.01 0.0 0.0\nX 0.49 0.0 0.0\nC 0.0 0.0 0.0\nC 0.5 0.0 0.0\n";
        for (suffix, content) in [("_hr.dat", hr), ("_centres.xyz", xyz)] {
            fs::write(format!("{dir}seedname{suffix}"), content).unwrap();
        }
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 2);
        assert_eq!(model.natom(), 2);
        let first: Vec<usize> = model.atoms[0]
            .orbitals()
            .iter()
            .map(|id| id.index())
            .collect();
        let second: Vec<usize> = model.atoms[1]
            .orbitals()
            .iter()
            .map(|id| id.index())
            .collect();
        assert_eq!(first, vec![0]);
        assert_eq!(second, vec![1]);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_assigns_centres_across_the_cell_boundary() {
        // A centre at 0.95 belongs to the atom at 0.05, not to the one at 0.55.
        let dir = "tests/tmp_w90_boundary/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin projections\nend projections\n";
        let hr = "generated\n2\n1\n1\n0 0 0 1 1 0.0 0.0\n0 0 0 1 2 0.0 0.0\n0 0 0 2 1 0.0 0.0\n0 0 0 2 2 0.0 0.0\n";
        let xyz =
            "4\nWannier centres\nX 0.95 0.0 0.0\nX 0.45 0.0 0.0\nC 0.05 0.0 0.0\nC 0.55 0.0 0.0\n";
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.atoms[0].orbitals().len(), 1);
        assert_eq!(model.atoms[0].orbitals()[0].index(), 0);
        assert_eq!(model.atoms[1].orbitals()[0].index(), 1);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_rejects_a_centre_far_from_every_atom() {
        let dir = "tests/tmp_w90_far_centre/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nend unit_cell_cart\n\nbegin projections\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        let xyz = "2\nWannier centres\nX 2.5 0.0 0.0\nC 0.0 0.0 0.0\n";
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        let error = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap_err();
        assert!(error.to_string().contains("nearest atom"), "{error}");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_accepts_lowercase_atom_labels() {
        // Regression: Wannier90 output may spell element labels in lower case
        // ("v", "cl"); labels, projections and the HR state count must agree.
        let dir = "tests/tmp_w90_lowercase_labels/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin atoms_cart\nv 0.0 0.0 0.0\ncl 0.5 0.5 0.5\nend atoms_cart\n\nbegin projections\nV:s\nCl:s\nend projections\n";
        let hr = "generated\n2\n1\n1\n0 0 0 1 1 0.0 0.0\n0 0 0 1 2 0.0 0.0\n0 0 0 2 1 0.0 0.0\n0 0 0 2 2 0.0 0.0\n";
        let xyz =
            "4\nWannier centres\nv 0.0 0.0 0.0\ncl 0.5 0.5 0.5\nv 0.0 0.0 0.0\ncl 0.5 0.5 0.5\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 2);
        assert_eq!(model.natom(), 2);
        assert_eq!(model.atoms[0].atom_type(), AtomType::V);
        assert_eq!(model.atoms[1].atom_type(), AtomType::Cl);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_assigns_orbitals_by_nearest_centres() {
        // The centre file writes the two Te blocks first and in reverse atom
        // order, while the atom list starts with Fe. Ownership must follow the
        // centres, and labels must use the Wannier90 built-in orbital order.
        let dir = "tests/tmp_w90_nearest_centres/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0 0.0\nTe 0.0 0.0 2.3\nTe 0.0 0.0 4.6\nend atoms_cart\n\nbegin projections\nFe:d\nTe:p\nend projections\n";
        let mut hr = String::from("generated\n11\n1\n1\n");
        for i in 1..=11 {
            for j in 1..=11 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        let xyz = "14\nWannier centres\nTe 0.0 0.0 4.6\nTe 0.0 0.0 4.6\nTe 0.0 0.0 4.6\nFe 0.0 0.0 0.0\nFe 0.0 0.0 0.0\nFe 0.0 0.0 0.0\nFe 0.0 0.0 0.0\nFe 0.0 0.0 0.0\nTe 0.0 0.0 2.3\nTe 0.0 0.0 2.3\nTe 0.0 0.0 2.3\nFe 0.0 0.0 0.0\nTe 0.0 0.0 2.3\nTe 0.0 0.0 4.6\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 11);
        assert_eq!(model.natom(), 3);
        assert_eq!(model.atoms[0].atom_type(), AtomType::Fe);
        assert_eq!(model.atoms[1].atom_type(), AtomType::Te);
        assert_eq!(model.atoms[2].atom_type(), AtomType::Te);
        let ids = |atom: usize| {
            model.atoms[atom]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(0), vec![3, 4, 5, 6, 7]);
        assert_eq!(ids(1), vec![8, 9, 10]);
        assert_eq!(ids(2), vec![0, 1, 2]);
        assert_eq!(
            model.orb_projection,
            vec![
                OrbProj::pz,
                OrbProj::px,
                OrbProj::py,
                OrbProj::dz2,
                OrbProj::dxz,
                OrbProj::dyz,
                OrbProj::dx2y2,
                OrbProj::dxy,
                OrbProj::pz,
                OrbProj::px,
                OrbProj::py,
            ]
        );
        assert!((model.orb[[0, 2]] - 0.46).abs() < 1e-12);
        assert!(model.orb[[3, 2]].abs() < 1e-12);
        assert!((model.orb[[8, 2]] - 0.23).abs() < 1e-12);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_handles_the_reported_62_centre_dataset() {
        // Geometry taken from a real Fe4Te2O12 data set: 62 Wannier centres
        // grouped Te(3), Te(3), Fe(5)x4, O(3)x12, while the atom list starts
        // with Fe. That is the observed block order the nearest-centre rule must
        // handle; the projections block is Fe:d, Te:p, O:p.
        let dir = "tests/tmp_w90_real_62/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = format!(
            "begin unit_cell_cart\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nend unit_cell_cart\n\nbegin atoms_cart\nfe 2.30049990 2.30049990 7.57401470\nfe 0.00000000 0.00000000 6.05648520\nfe 0.00000000 0.00000000 3.03051440\nfe 2.30049990 2.30049990 1.51298550\nte 0.00000000 0.00000000 0.00000000\nte 2.30049990 2.30049990 4.54349990\no 1.37569890 1.37569890 0.00000000\no 0.92480100 3.67619900 4.54349990\no 3.67619900 0.92480100 4.54349990\no 3.22530080 3.22530080 0.00000000\no 1.41250700 1.41250700 3.04868850\no 0.88799290 3.71300710 7.59218840\no 3.71300710 0.88799290 7.59218840\no 3.71300710 0.88799290 1.49481150\no 0.88799290 3.71300710 1.49481150\no 3.18849270 3.18849270 3.04868850\no 1.41250700 1.41250700 6.03831140\no 3.18849270 3.18849270 6.03831140\nend atoms_cart\n\nbegin projections\nFe:d\nTe:p\nO:p\nend projections\n"
        );
        let mut hr = String::from("generated\n62\n1\n1\n");
        for i in 1..=62 {
            for j in 1..=62 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        let xyz = "80\n Wannier centres, written by Wannier90 on23Nov2026 at 14:26:08\nX         -0.00000009      -0.00000010       0.01508405\nX         -0.00000014      -0.00000005       0.00412937\nX         -0.00000005      -0.00000014       0.00412972\nX          2.30049984       2.30049985       4.55858375\nX          2.30049987       2.30049997       4.54762983\nX          2.30049997       2.30049988       4.54763019\nX          2.30049990       2.30049990       7.57297993\nX          2.30049989       2.30049990       7.57396502\nX          2.30049990       2.30049989       7.57396492\nX          2.30049990       2.30049990       7.57464396\nX          2.30049990       2.30049990       7.57362089\nX         -0.00000000      -0.00000000       6.06213708\nX         -0.00000000      -0.00000000       6.05536535\nX         -0.00000000      -0.00000000       6.05536540\nX         -0.00000001      -0.00000001       6.05593748\nX         -0.00000000      -0.00000000       6.05894187\nX         -0.00000000      -0.00000000       3.02947958\nX          0.00000000      -0.00000000       3.03046478\nX         -0.00000000       0.00000000       3.03046467\nX          0.00000000       0.00000000       3.03114379\nX         -0.00000000      -0.00000000       3.03012075\nX          2.30049990       2.30049990       1.51863744\nX          2.30049989       2.30049990       1.51186569\nX          2.30049990       2.30049989       1.51186574\nX          2.30049990       2.30049990       1.51243772\nX          2.30049990       2.30049990       1.51544210\nX          1.38246512       1.38246521       0.00040771\nX          1.35287083       1.30293535      -0.00146505\nX          1.30293524       1.35287087      -0.00146511\nX          0.91803478       3.68296523       4.54390770\nX          0.94762906       3.60343541       4.54203503\nX          0.99756462       3.65337097       4.54203498\nX          3.68296514       0.91803469       4.54390770\nX          3.65337093       0.99756451       4.54203503\nX          3.60343531       0.94762901       4.54203498\nX          3.21853454       3.21853445       0.00040771\nX          3.24812877       3.29806428      -0.00146505\nX          3.29806439       3.24812873      -0.00146511\nX          1.46224622       1.46224632       3.06258484\nX          1.38802142       1.42940468       3.07897980\nX          1.42940458       1.38802066       3.07898002\nX          0.83825342       3.76274659       7.60608528\nX          0.91247828       3.72990483       7.62247978\nX          0.87109519       3.68852094       7.62248000\nX          3.76274650       0.83825332       7.60608528\nX          3.68852171       0.87109509       7.62247977\nX          3.72990473       0.91247905       7.62248000\nX          3.76238583       0.83861411       1.48575434\nX          3.69235685       0.86739084       1.46290000\nX          3.73360926       0.90864258       1.46289981\nX          0.83861409       3.76238581       1.48575434\nX          0.90864313       3.73360908       1.46289999\nX          0.86739066       3.69235740       1.46289982\nX          3.13875342       3.13875332       3.06258490\nX          3.21297820       3.17159497       3.07897982\nX          3.17159507       3.21297897       3.07898005\nX          1.46188601       1.46188600       6.02925386\nX          1.39185695       1.43310917       6.00639990\nX          1.43310935       1.39185750       6.00639972\nX          3.13911363       3.13911364       6.02925380\nX          3.20914267       3.16789049       6.00639988\nX          3.16789031       3.20914212       6.00639970\nfe         2.30049990       2.30049990       7.57401470\nfe         0.00000000       0.00000000       6.05648520\nfe         0.00000000       0.00000000       3.03051440\nfe         2.30049990       2.30049990       1.51298550\nte         0.00000000       0.00000000       0.00000000\nte         2.30049990       2.30049990       4.54349990\no          1.37569890       1.37569890       0.00000000\no          0.92480100       3.67619900       4.54349990\no          3.67619900       0.92480100       4.54349990\no          3.22530080       3.22530080       0.00000000\no          1.41250700       1.41250700       3.04868850\no          0.88799290       3.71300710       7.59218840\no          3.71300710       0.88799290       7.59218840\no          3.71300710       0.88799290       1.49481150\no          0.88799290       3.71300710       1.49481150\no          3.18849270       3.18849270       3.04868850\no          1.41250700       1.41250700       6.03831140\no          3.18849270       3.18849270       6.03831140\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 62);
        assert_eq!(model.natom(), 18);
        let ids = |atom: usize| {
            model.atoms[atom]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        };
        // The two Te blocks come first, then the four Fe d shells, then O.
        assert_eq!(ids(4), vec![0, 1, 2]);
        assert_eq!(ids(5), vec![3, 4, 5]);
        assert_eq!(ids(0), vec![6, 7, 8, 9, 10]);
        assert_eq!(ids(1), vec![11, 12, 13, 14, 15]);
        assert_eq!(ids(2), vec![16, 17, 18, 19, 20]);
        assert_eq!(ids(3), vec![21, 22, 23, 24, 25]);
        assert_eq!(
            model.atoms.iter().map(|atom| atom.norb()).sum::<usize>(),
            62
        );
        assert_eq!(model.orb_projection[0], OrbProj::pz);
        assert_eq!(model.orb_projection[2], OrbProj::py);
        assert_eq!(model.orb_projection[6], OrbProj::dz2);
        assert_eq!(model.orb_projection[10], OrbProj::dxy);
        assert_eq!(model.orb_projection[26], OrbProj::pz);
        assert!((model.orb[[0, 2]] - 0.001508405).abs() < 1e-12);
        assert!((model.orb[[6, 2]] - 0.757297993).abs() < 1e-12);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_falls_back_to_projection_order_for_ambiguous_blocks() {
        // Two centre blocks point at Fe and Te, but the projections block gives
        // Fe one s orbital and Te three p orbitals, so the blocks are not a
        // valid atom partition. The Wannier90 projection order is used instead.
        let dir = "tests/tmp_w90_block_mismatch/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0 0.0\nTe 0.0 0.0 2.3\nend atoms_cart\n\nbegin projections\nFe:s\nTe:p\nend projections\n";
        let mut hr = String::from("generated\n4\n1\n1\n");
        for i in 1..=4 {
            for j in 1..=4 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        let xyz = "6\nWannier centres\nFe 0.0 0.0 0.0\nFe 0.0 0.0 0.0\nTe 0.0 0.0 2.3\nTe 0.0 0.0 2.3\nFe 0.0 0.0 0.0\nTe 0.0 0.0 2.3\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        let ids = |atom: usize| {
            model.atoms[atom]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(0), vec![0]);
        assert_eq!(ids(1), vec![1, 2, 3]);
        assert_eq!(
            model.orb_projection,
            vec![OrbProj::s, OrbProj::pz, OrbProj::px, OrbProj::py]
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_falls_back_to_projection_order_for_bond_centred_wanniers() {
        // GaAs-like sp3 hybrids live at the bond centre, more than 1 A from
        // both atoms, so the nearest-atom block assignment cannot be used; the
        // projection order must still load the model.
        let dir = "tests/tmp_w90_bond_centred/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nend unit_cell_cart\n\nbegin atoms_cart\nGa 0.0 0.0 0.0\nAs 5.0 5.0 5.0\nend atoms_cart\n\nbegin projections\nGa:sp3\nAs:sp3\nend projections\n";
        let mut hr = String::from("generated\n8\n1\n1\n");
        for i in 1..=8 {
            for j in 1..=8 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        let xyz = "10\nWannier centres\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nX 2.5 2.5 2.5\nGa 0.0 0.0 0.0\nAs 5.0 5.0 5.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        let ids = |atom: usize| {
            model.atoms[atom]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(0), vec![0, 1, 2, 3]);
        assert_eq!(ids(1), vec![4, 5, 6, 7]);
        assert!((model.orb[[0, 0]] - 0.25).abs() < 1e-12);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_uses_projection_order_for_interleaved_lines() {
        // Two Fe atoms with Fe:s and Fe:p give line-major Wannier order:
        // both s functions first, then both p shells. The centres therefore do
        // not form one block per atom, and the projection order must be used.
        let dir = "tests/tmp_w90_interleaved_lines/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0 0.0\nFe 5.0 0.0 0.0\nend atoms_cart\n\nbegin projections\nFe:s\nFe:p\nend projections\n";
        let mut hr = String::from("generated\n8\n1\n1\n");
        for i in 1..=8 {
            for j in 1..=8 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        let xyz = "10\nWannier centres\nX 0.0 0.0 0.0\nX 5.0 0.0 0.0\nX 0.0 0.0 0.0\nX 0.0 0.0 0.0\nX 0.0 0.0 0.0\nX 5.0 0.0 0.0\nX 5.0 0.0 0.0\nX 5.0 0.0 0.0\nFe 0.0 0.0 0.0\nFe 5.0 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        let ids = |atom: usize| {
            model.atoms[atom]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(0), vec![0, 2, 3, 4]);
        assert_eq!(ids(1), vec![1, 5, 6, 7]);
        assert_eq!(
            model.orb_projection,
            vec![
                OrbProj::s,
                OrbProj::s,
                OrbProj::pz,
                OrbProj::px,
                OrbProj::py,
                OrbProj::pz,
                OrbProj::px,
                OrbProj::py,
            ]
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_matches_displaced_centres_across_the_cell_boundary() {
        // The centres are displaced from the atoms and one lies across the cell
        // boundary; the minimum-image distance must still select the right atom.
        let dir = "tests/tmp_w90_displaced_centres/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.6 0.0 0.0\nTe 1.6 0.0 0.0\nend atoms_cart\n\nbegin projections\nFe:d\nTe:p\nend projections\n";
        let mut hr = String::from("generated\n8\n1\n1\n");
        for i in 1..=8 {
            for j in 1..=8 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        let xyz = "10\nWannier centres\nX 0.04 0.0 0.0\nX 0.04 0.0 0.0\nX 0.04 0.0 0.0\nX 1.0 0.0 0.0\nX 1.0 0.0 0.0\nX 1.0 0.0 0.0\nX 1.0 0.0 0.0\nX 1.0 0.0 0.0\nFe 0.6 0.0 0.0\nTe 1.6 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        let ids = |atom: usize| {
            model.atoms[atom]
                .orbitals()
                .iter()
                .map(|id| id.index())
                .collect::<Vec<_>>()
        };
        assert_eq!(ids(0), vec![3, 4, 5, 6, 7]);
        assert_eq!(ids(1), vec![0, 1, 2]);
        assert!((model.orb[[0, 0]] - 0.02).abs() < 1e-12);
        assert!((model.orb[[3, 0]] - 0.5).abs() < 1e-12);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_drops_xyz_atom_without_orbitals() {
        // Regression: an xyz atom whose species has no projection (e.g. a Cs
        // site that was never Wannierized) has no fitted orbitals, so it must
        // be dropped while atoms with orbitals are kept.
        let dir = "tests/tmp_w90_drop/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin projections\nC:s\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        // One Wannier centre, one C atom with a fitted orbital, one Cs atom
        // with no orbitals: the Cs atom must be dropped.
        let xyz = "3\nWannier centres\nX 0.0 0.0 0.0\nC 0.1 0.0 0.0\nCs 0.2 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 1);
        assert_eq!(
            model.natom(),
            1,
            "Cs atom has no orbitals and must be dropped"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_drops_multiple_atoms_of_same_unprojected_species() {
        // Two Cs atoms, both without orbitals: both must be dropped (the
        // warning is emitted once per species, but that is stderr-only and
        // not asserted here).
        let dir = "tests/tmp_w90_drop_multi/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin projections\nC:s\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        let xyz =
            "4\nWannier centres\nX 0.0 0.0 0.0\nC 0.1 0.0 0.0\nCs 0.2 0.0 0.0\nCs 0.3 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 1);
        assert_eq!(model.natom(), 1, "both Cs atoms must be dropped");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_without_xyz_drops_atom_without_orbitals() {
        // No _centres.xyz: the atoms_cart fallback must also drop a species
        // with no projection instead of erroring.
        let dir = "tests/tmp_w90_no_xyz_drop/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin atoms_cart\nC 0.0 0.0 0.0\nCs 0.1 0.0 0.0\nend atoms_cart\n\nbegin projections\nC:s\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();

        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 1);
        assert_eq!(model.natom(), 1, "Cs atom must be dropped");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_rejects_projection_species_missing_from_xyz() {
        // Reverse direction: the projection block declares C:s, but no C atom
        // appears in _centres.xyz (only Fe, which is dropped). The C orbital
        // can never be assigned, so this must remain a hard error.
        let dir = "tests/tmp_w90_proj_missing/";
        write_minimal_dataset(dir, "Fe");
        let err = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap_err();
        assert!(matches!(err, TbError::FileParse { .. }));
        assert!(
            err.to_string().contains("species mismatch"),
            "unexpected error: {err}"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_rejects_unknown_projection_species() {
        // Regression: `parse::<AtomType>()` must reject an unknown species in
        // the projections block with the same FileParse error as before.
        let dir = "tests/tmp_w90_bad_proj_species/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin projections\nXx:s\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        let xyz = "1\nWannier centres\nC 0.0 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let err = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap_err();
        assert!(matches!(err, TbError::FileParse { .. }));
        assert!(
            err.to_string()
                .contains("Unknown atomic species 'Xx' in begin projections"),
            "unexpected error: {err}"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn win_data_line_strips_hash_and_bang_comments() {
        assert_eq!(win_data_line("spinors = .true."), "spinors = .true.");
        assert_eq!(win_data_line("# spinors = .true."), "");
        assert_eq!(win_data_line("#spinors=true"), "");
        assert_eq!(
            win_data_line("spinors = .true. ! enable spin"),
            "spinors = .true."
        );
        assert_eq!(win_data_line("Fe 1.0 0.0 0.0 # position"), "Fe 1.0 0.0 0.0");
        assert_eq!(win_data_line("  \t"), "");
    }

    #[test]
    fn from_hr_without_centres_xyz_uses_atoms_cart_fallback() {
        // Regression: without _centres.xyz the outer norb stayed 0 and the
        // atoms_cart fallback underflowed (0 - 1) in debug builds.
        let dir = "tests/tmp_w90_no_xyz/";
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0 0.0\nend atoms_cart\n\nbegin projections\nFe:s\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        let mut f = fs::File::create(format!("{dir}seedname.win")).unwrap();
        f.write_all(win.as_bytes()).unwrap();
        let mut f = fs::File::create(format!("{dir}seedname_hr.dat")).unwrap();
        f.write_all(hr.as_bytes()).unwrap();
        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 1);
        assert_eq!(model.natom(), 1);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_without_xyz_loads_multiple_atoms_of_same_species() {
        // Regression: the no-xyz fallback compared the constructed orbital
        // count against the once-only win projection declaration (4 for
        // Fe:s,p), but two Fe atoms each get their own copy (8 orbitals),
        // which the HR file confirms via nsta.
        let dir = "tests/tmp_w90_multi_atom/";
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0 0.0\nFe 1.0 0.0 0.0\nend atoms_cart\n\nbegin projections\nFe:s\nFe:p\nend projections\n";
        // 8 orbitals: 8x8 identity Hamiltonian at R=0.
        let mut hr = String::from("generated\n8\n1\n1\n0 0 0 1 1 0.0 0.0\n");
        for i in 1..=8 {
            for j in 1..=8 {
                if i != 1 || j != 1 {
                    let value = if i == j { "0.0 0.0" } else { "0.0 0.0" };
                    hr.push_str(&format!("0 0 0 {i} {j} {value}\n"));
                }
            }
        }
        let mut f = fs::File::create(format!("{dir}seedname.win")).unwrap();
        f.write_all(win.as_bytes()).unwrap();
        let mut f = fs::File::create(format!("{dir}seedname_hr.dat")).unwrap();
        f.write_all(hr.as_bytes()).unwrap();
        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 8);
        assert_eq!(model.natom(), 2);
        assert_eq!(model.atoms[0].norb(), 4);
        assert_eq!(model.atoms[1].norb(), 4);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_handles_coordinate_blocks_and_units_consistently() {
        // Regression: unit_cell_cart treated its optional unit as the first
        // lattice row, atoms_cart only accepted lower-case units, and
        // atoms_frac was not parsed at all. Each case below describes the same
        // atom at fractional x=0.5 in a two-unit cubic cell.
        for (lattice_unit, block, expected_lattice) in [
            (
                "BoHr",
                "ATOMS_CART\nBOHR\nFe 1.0 0.0 0.0 ! inline comment\nEND ATOMS_CART",
                2.0 * BOHR_TO_ANGSTROM,
            ),
            (
                "AnG",
                "atoms_cart\nAngstrom\nFe 1.0 0.0 0.0\nend atoms_cart",
                2.0,
            ),
            ("Ang", "atoms_frac\nFe 0.5 0.0 0.0\nend atoms_frac", 2.0),
        ] {
            let dir = "tests/tmp_w90_units/";
            let _ = fs::remove_dir_all(dir);
            fs::create_dir_all(dir).unwrap();
            let win = format!(
                "BEGIN UNIT_CELL_CART\n{lattice_unit}\n2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\nEND UNIT_CELL_CART\n\nBEGIN {block}\n\nbegin projections\nFe:s\nend projections\n"
            );
            let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
            let mut f = fs::File::create(format!("{dir}seedname.win")).unwrap();
            f.write_all(win.as_bytes()).unwrap();
            let mut f = fs::File::create(format!("{dir}seedname_hr.dat")).unwrap();
            f.write_all(hr.as_bytes()).unwrap();
            let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
            assert_eq!(model.norb(), 1);
            assert_eq!(model.natom(), 1);
            assert!((model.lat[[0, 0]] - expected_lattice).abs() < 1e-12);
            assert!((model.atoms[0].position()[0] - 0.5).abs() < 1e-12);
            assert!((model.orb[[0, 0]] - 0.5).abs() < 1e-12);
        }
        let _ = fs::remove_dir_all("tests/tmp_w90_units");
    }

    #[test]
    fn from_hr_rejects_malformed_atom_rows_without_panicking() {
        let dir = "tests/tmp_w90_malformed_atom/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1 0 0\n0 1 0\n0 0 1\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0\nend atoms_cart\n\nbegin projections\nFe:s\nend projections\n";
        let hr = "generated\n1\n1\n1\n0 0 0 1 1 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();

        let result = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0);
        assert!(matches!(result, Err(TbError::FileParse { .. })));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_rejects_truncated_centres_xyz_without_panicking() {
        let dir = "tests/tmp_w90_truncated_xyz/";
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1 0 0\n0 1 0\n0 0 1\nend unit_cell_cart\n\nbegin projections\nFe:s,p\nend projections\n";
        let mut hr = String::from("generated\n4\n1\n1\n");
        for i in 1..=4 {
            for j in 1..=4 {
                hr.push_str(&format!("0 0 0 {i} {j} 0.0 0.0\n"));
            }
        }
        // The header declares four centres plus one atom, but only one centre
        // and one atom line are present.
        let xyz = "5\nWannier centres\nX 0.0 0.0 0.0\nFe 0.0 0.0 0.0\n";
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();

        let error = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap_err();
        assert!(matches!(&error, TbError::FileParse { .. }));
        assert!(error.to_string().contains("Truncated"));
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_merges_multi_line_projections_into_one_atom() {
        // Regression: multiple projection lines of the same species (e.g.
        // Fe:s and Fe:p) created one Atom per line; Wannier90 supports
        // multiple lines per site and they must merge into a single Atom.
        let dir = "tests/tmp_w90_multiline/";
        fs::create_dir_all(dir).unwrap();
        let win = "begin unit_cell_cart\n1.0 0.0 0.0\n0.0 1.0 0.0\n0.0 0.0 1.0\nend unit_cell_cart\n\nbegin atoms_cart\nFe 0.0 0.0 0.0\nend atoms_cart\n\nbegin projections\nFe:s\nFe:p\nend projections\n";
        let hr = "generated\n4\n1\n1\n0 0 0 1 1 0.0 0.0\n0 0 0 1 2 0.0 0.0\n0 0 0 1 3 0.0 0.0\n0 0 0 1 4 0.0 0.0\n0 0 0 2 1 0.0 0.0\n0 0 0 2 2 0.0 0.0\n0 0 0 2 3 0.0 0.0\n0 0 0 2 4 0.0 0.0\n0 0 0 3 1 0.0 0.0\n0 0 0 3 2 0.0 0.0\n0 0 0 3 3 0.0 0.0\n0 0 0 3 4 0.0 0.0\n0 0 0 4 1 0.0 0.0\n0 0 0 4 2 0.0 0.0\n0 0 0 4 3 0.0 0.0\n0 0 0 4 4 0.0 0.0\n";
        let mut f = fs::File::create(format!("{dir}seedname.win")).unwrap();
        f.write_all(win.as_bytes()).unwrap();
        let mut f = fs::File::create(format!("{dir}seedname_hr.dat")).unwrap();
        f.write_all(hr.as_bytes()).unwrap();
        let model = <Model<false, 3> as Wannier90>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 4);
        assert_eq!(
            model.natom(),
            1,
            "multi-line projections must merge into one Atom"
        );
        assert_eq!(model.atoms[0].norb(), 4);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn from_hr_rejects_invalid_counts_records_and_parameters() {
        // Parameter validation precedes all file access.
        assert!(matches!(
            Model::<false, 2>::from_hr("missing/", "seed", 0.0),
            Err(TbError::InvalidDimension { .. })
        ));
        for energy in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                Model::<false, 3>::from_hr("missing/", "seed", energy)
                    .unwrap_err()
                    .to_string()
                    .contains("finite")
            );
        }
        let dir = "tests/tmp_w90_bad_records/";
        write_minimal_dataset(dir, "C");
        for hr in [
            String::new(),
            "generated\n".into(),
            "generated\n1\n".into(),
            "generated\n1\n1\n".into(),
            "generated\n1\n1\n0\n".into(),
            "generated\n1\n1\n1\n".into(),
            "generated\n0\n1\n1\n".into(),
            format!("generated\n{}\n2\n1 1\n", usize::MAX),
            "generated\n1\n1\n1\n0 0 0 0 1 2 0\n".into(),
            "generated\n1\n1\n1\n0 0 0 1 2 2 0\n".into(),
            "generated\n1\n1\n1\n0 0 0 1 1 NaN 0\n".into(),
            "generated\n2\n1\n1\n0 0 0 1 1 0 0\n0 0 0 1 1 0 0\n0 0 0 2 1 0 0\n0 0 0 2 2 0 0\n"
                .into(),
        ] {
            fs::write(format!("{dir}seedname_hr.dat"), &hr).unwrap();
            assert!(
                matches!(
                    Model::<false, 3>::from_hr(dir, "seedname", 0.0),
                    Err(TbError::FileParse { .. })
                ),
                "accepted {hr:?}"
            );
        }
        // No decimal point is needed to identify the beginning of the matrix records.
        fs::write(
            format!("{dir}seedname_hr.dat"),
            "generated\n1\n1\n2\n0 0 0 1 1 6 0\n",
        )
        .unwrap();
        assert_eq!(
            Model::<false, 3>::from_hr(dir, "seedname", 0.5)
                .unwrap()
                .ham[[0, 0, 0]],
            Complex64::new(2.5, 0.0)
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_retains_distinct_spin_centres_and_paired_metadata() {
        let dir = "tests/tmp_w90_distinct_spin_centres/";
        fs::create_dir_all(dir).unwrap();
        let win = "spinors = .true.\nbegin unit_cell_cart\n2 0 0\n0 4 0\n0 0 8\nend unit_cell_cart\nbegin atoms_frac\nC 0.125 0.125 0.125\nO 0.625 0.125 0.125\nend atoms_frac\n";
        // Up order is O, C, opposite to the atom/projection order. Each down
        // centre is nearer the other atom; neither ownership nor labels may
        // be reassigned from that half, and cell representatives stay intact.
        let xyz = "6\ncentres\nX 1.25 0.5 1\nX 0.25 0.5 1\nX 2.25 0.5 3\nX -0.75 -0.5 1\nC 0.25 0.5 1\nO 1.25 0.5 1\n";
        let expected_centres = array![
            [0.625, 0.125, 0.125],
            [0.125, 0.125, 0.125],
            [1.125, 0.125, 0.375],
            [-0.375, -0.125, 0.125]
        ];
        let expected_ham = Array3::from_shape_fn((1, 4, 4), |(_, i, j)| {
            Complex64::new((i + j + 1) as f64, i as f64 - j as f64)
        });
        // Deliberately independent of xyz: importing centres must not replace
        // the full position matrix or touch its spin-flip elements.
        let expected_r = Array4::from_shape_fn((1, 3, 4, 4), |(_, axis, i, j)| {
            Complex64::new(
                (10 * axis + i + j + 1) as f64,
                (axis + 1) as f64 * (i as f64 - j as f64),
            )
        });
        let mut hr = String::from("generated\n4\n1\n1\n");
        let mut r = String::from("generated\n4\n1\n");
        for i in 0..4 {
            for j in 0..4 {
                let h = expected_ham[[0, i, j]];
                hr.push_str(&format!("0 0 0 {} {} {} {}\n", i + 1, j + 1, h.re, h.im));
                r.push_str(&format!("0 0 0 {} {}", i + 1, j + 1));
                for axis in 0..3 {
                    let value = expected_r[[0, axis, i, j]];
                    r.push_str(&format!(" {} {}", value.re, value.im));
                }
                r.push('\n');
            }
        }
        fs::write(format!("{dir}seedname_hr.dat"), hr).unwrap();
        fs::write(format!("{dir}seedname_r.dat"), r).unwrap();
        fs::write(format!("{dir}seedname_centres.xyz"), xyz).unwrap();
        for with_projections in [false, true] {
            let projections = if with_projections { "C:s\nO:pz\n" } else { "" };
            fs::write(
                format!("{dir}seedname.win"),
                format!("{win}begin projections\n{projections}end projections\n"),
            )
            .unwrap();
            let plain = Model::<true, 3>::from_hr(dir, "seedname", 0.0).unwrap();
            let model = Model::<true, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0).unwrap();
            assert_eq!((model.norb(), model.nsta()), (2, 4));
            assert_eq!(model.orb, expected_centres);
            assert_eq!(plain.orb, expected_centres);
            assert_eq!(model.ham, expected_ham);
            assert_eq!(plain.ham, expected_ham);
            assert_eq!(model.hamR, array![[0, 0, 0]]);
            assert_eq!(model.rmatrix.0, expected_r);
            assert_eq!(model.atoms.len(), 2);
            assert_eq!(model.atoms[0].orbitals(), &[OrbitalId::new(1)]);
            assert_eq!(model.atoms[1].orbitals(), &[OrbitalId::new(0)]);
            assert_eq!(
                model.orb_projection,
                if with_projections {
                    vec![OrbProj::pz, OrbProj::s]
                } else {
                    vec![OrbProj::s; 2]
                }
            );
            model.validate().unwrap();
        }
        // With no xyz at all, and only then, the .win atom centres are copied
        // into both halves in projection order; matrices remain unchanged.
        fs::remove_file(format!("{dir}seedname_centres.xyz")).unwrap();
        let fallback = Model::<true, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(
            fallback.orb,
            array![
                [0.125, 0.125, 0.125],
                [0.625, 0.125, 0.125],
                [0.125, 0.125, 0.125],
                [0.625, 0.125, 0.125]
            ]
        );
        assert_eq!((fallback.norb(), fallback.nsta()), (2, 4));
        assert_eq!(fallback.atoms[0].orbitals(), &[OrbitalId::new(0)]);
        assert_eq!(fallback.atoms[1].orbitals(), &[OrbitalId::new(1)]);
        assert_eq!(fallback.orb_projection, vec![OrbProj::s, OrbProj::pz]);
        assert_eq!(fallback.ham, expected_ham);
        assert_eq!(fallback.rmatrix.0, expected_r);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_spin_and_position_storage_modes() {
        let dir = "tests/tmp_w90_spin_storage/";
        write_minimal_dataset(dir, "C");
        let mut win = fs::read_to_string(format!("{dir}seedname.win")).unwrap();
        win.push_str("\nspinors = .true.\n");
        fs::write(format!("{dir}seedname.win"), win).unwrap();
        fs::write(
            format!("{dir}seedname_hr.dat"),
            "generated\n2\n1\n1\n0 0 0 1 1 1 0\n0 0 0 2 1 0 2\n0 0 0 1 2 0 -2\n0 0 0 2 2 3 0\n",
        )
        .unwrap();
        fs::write(
            format!("{dir}seedname_centres.xyz"),
            "3\ncentres\nX 0 0 0\nX 0 0 0\nC 0 0 0\n",
        )
        .unwrap();
        assert!(
            Model::<false, 3>::from_hr(dir, "seedname", 0.0)
                .unwrap_err()
                .to_string()
                .contains("Spin mismatch")
        );
        let model = Model::<true, 3>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.norb(), 1);
        assert_eq!(model.ham[[0, 1, 0]], Complex64::new(0.0, 2.0));
        assert!(matches!(
            Model::<true, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0),
            Err(TbError::FileCreation { .. })
        ));
        fs::write(format!("{dir}seedname_r.dat"), "generated\n2\n1\n0 0 0 1 1 1 9 2 0 3 0\n0 0 0 2 1 2 4 0 0 0 0\n0 0 0 1 2 6 -2 0 0 0 0\n0 0 0 2 2 3 -9 2 0 1 0\n").unwrap();
        let model = Model::<true, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0).unwrap();
        // The sole R=0 block also undergoes Hermitian projection.
        assert_eq!(model.rmatrix[[0, 0, 0, 0]], Complex64::new(1.0, 0.0));
        assert_eq!(model.rmatrix[[0, 0, 1, 0]], Complex64::new(4.0, 3.0));
        assert_eq!(model.rmatrix[[0, 0, 0, 1]], Complex64::new(4.0, -3.0));
        // Every declared spin centre is parsed, including the second half.
        fs::write(
            format!("{dir}seedname_centres.xyz"),
            "3\ncentres\nX 0 0 0\nX NaN 0 0\nC 0 0 0\n",
        )
        .unwrap();
        assert!(matches!(
            Model::<true, 3>::from_hr(dir, "seedname", 0.0),
            Err(TbError::FileParse { .. })
        ));
        fs::write(
            format!("{dir}seedname_hr.dat"),
            "generated\n1\n1\n1\n0 0 0 1 1 0 0\n",
        )
        .unwrap();
        assert!(
            Model::<true, 3>::from_hr(dir, "seedname", 0.0)
                .unwrap_err()
                .to_string()
                .contains("even")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_remaps_support_and_matches_rmatrix_weights_by_translation() {
        let dir = "tests/tmp_w90_support/";
        write_minimal_dataset(dir, "C");
        // The origin is not the first declared block, and _r.dat has another block order.
        fs::write(
            format!("{dir}seedname_hr.dat"),
            "generated\n1\n3\n2 3 5\n-1 0 0 1 1 8 -4\n0 0 0 1 1 18 0\n1 0 0 1 1 20 10\n",
        )
        .unwrap();
        fs::write(format!("{dir}seedname_r.dat"), "generated\n1\n3\n1 0 0 1 1 50 20 0 0 0 0\n-1 0 0 1 1 20 -8 0 0 0 0\n0 0 0 1 1 9 3 0 0 0 0\n").unwrap();
        fs::write(format!("{dir}seedname_wsvec.dat"), "generated\n-1 0 0 1 1\n2\n0 0 0\n-1 0 0\n0 0 0 1 1\n1\n0 0 0\n1 0 0 1 1\n2\n0 0 0\n1 0 0\n").unwrap();
        let plain = Model::<false, 3>::from_hr(dir, "seedname", 1.0).unwrap();
        let model = Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 1.0).unwrap();
        assert_eq!(model.ham, plain.ham);
        assert_eq!(model.hamR, plain.hamR);
        assert_eq!(model.hamR.row(0), array![0, 0, 0]);
        assert_eq!(model.ham[[0, 0, 0]], Complex64::new(5.0, 0.0));
        assert_eq!(model.rmatrix[[0, 0, 0, 0]], Complex64::new(3.0, 0.0));
        for translation in [-2, -1, 1, 2] {
            let index = find_R(&model.hamR, &array![translation, 0, 0]).unwrap();
            let sign = translation.signum() as f64;
            assert_eq!(model.ham[[index, 0, 0]], Complex64::new(2.0, sign));
            assert_eq!(
                model.rmatrix[[index, 0, 0, 0]],
                Complex64::new(5.0, 2.0 * sign)
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_requires_matching_declared_position_translations() {
        let dir = "tests/tmp_w90_position_translations/";
        write_minimal_dataset(dir, "C");
        // Neither file declares R=0. Rustb's synthetic origin must not count.
        fs::write(
            format!("{dir}seedname_hr.dat"),
            "generated\n1\n2\n1 1\n-1 0 0 1 1 2 0\n1 0 0 1 1 2 0\n",
        )
        .unwrap();
        let rfile = format!("{dir}seedname_r.dat");
        let minus = "-1 0 0 1 1 3 0 0 0 0 0\n";
        let plus = "1 0 0 1 1 3 0 0 0 0 0\n";
        let origin = "0 0 0 1 1 0 0 0 0 0 0\n";
        for (count, blocks, expected) in [
            (1, vec![minus], "_r.dat declares 1, _hr.dat declares 2"),
            (
                3,
                vec![minus, plus, origin],
                "_r.dat declares 3, _hr.dat declares 2",
            ),
            (2, vec![minus, origin], "[0, 0, 0] not found in _hr.dat"),
            (
                2,
                vec![minus, minus],
                "Duplicate position-matrix R block [-1, 0, 0]",
            ),
        ] {
            fs::write(
                &rfile,
                format!("generated\n1\n{count}\n{}", blocks.concat()),
            )
            .unwrap();
            let error = Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0).unwrap_err();
            assert!(matches!(error, TbError::FileParse { file, message }
                if file == rfile && message.contains(expected)));
            Model::<false, 3>::from_hr(dir, "seedname", 0.0).unwrap();
        }
        fs::write(&rfile, format!("generated\n1\n2\n{plus}{minus}")).unwrap();
        let model = Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.hamR, array![[0, 0, 0], [-1, 0, 0], [1, 0, 0]]);
        assert_eq!(model.rmatrix[[0, 0, 0, 0]], Complex64::new(0.0, 0.0));
        assert_eq!(model.rmatrix[[1, 0, 0, 0]], Complex64::new(3.0, 0.0));
        assert_eq!(model.rmatrix[[2, 0, 0, 0]], Complex64::new(3.0, 0.0));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_rejects_malformed_rmatrix_and_wsvec() {
        let dir = "tests/tmp_w90_bad_optional/";
        write_minimal_dataset(dir, "C");
        let rfile = format!("{dir}seedname_r.dat");
        for input in [
            "",
            "generated\n",
            "generated\n1\n1\n",
            "generated\n2\n1\n",
            "generated\n1\n1\n0 0 0 0 1 1 0 0 0 0 0\n",
            "generated\n1\n1\n0 0 0 1 1 1 0 0\n",
        ] {
            fs::write(&rfile, input).unwrap();
            assert!(
                matches!(
                    Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0),
                    Err(TbError::FileParse { .. })
                ),
                "accepted {input:?}"
            );
            // NoRMatrix does not inspect an unused _r.dat file.
            Model::<false, 3>::from_hr(dir, "seedname", 0.0).unwrap();
        }
        fs::write(&rfile, "generated\n1\n1\n0 0 0 1 1 1 0 0 0 0 0\n").unwrap();
        let wsfile = format!("{dir}seedname_wsvec.dat");
        for input in [
            "",
            "generated\n",
            "generated\n0 0 0 1 1\n",
            "generated\n0 0 0 1 1\n0\n",
            "generated\n0 0 0 1 1\n1\n",
            "generated\n0 0 0 0 1\n1\n0 0 0\n",
            "generated\n0 0 0 1 1\n1\n0 bad 0\n",
            "generated\n0 0 0 1 1\n1\n0 0\n",
        ] {
            fs::write(&wsfile, input).unwrap();
            assert!(
                matches!(
                    Model::<false, 3>::from_hr(dir, "seedname", 0.0),
                    Err(TbError::FileParse { .. })
                ),
                "accepted {input:?}"
            );
            assert!(
                matches!(
                    Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0),
                    Err(TbError::FileParse { .. })
                ),
                "accepted {input:?}"
            );
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_checks_support_overflow_and_last_hermitian_partner() {
        let dir = "tests/tmp_w90_support_errors/";
        write_minimal_dataset(dir, "C");
        fs::write(
            format!("{dir}seedname_hr.dat"),
            format!("generated\n1\n1\n1\n{} 0 0 1 1 2 0\n", isize::MAX),
        )
        .unwrap();
        fs::write(
            format!("{dir}seedname_wsvec.dat"),
            format!("generated\n{} 0 0 1 1\n1\n1 0 0\n", isize::MAX),
        )
        .unwrap();
        assert!(
            Model::<false, 3>::from_hr(dir, "seedname", 0.0)
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        fs::write(
            format!("{dir}seedname_hr.dat"),
            "generated\n1\n1\n1\n1 0 0 1 1 2 0\n",
        )
        .unwrap();
        fs::write(
            format!("{dir}seedname_wsvec.dat"),
            "generated\n1 0 0 1 1\n1\n0 0 0\n",
        )
        .unwrap();
        let model = Model::<false, 3>::from_hr(dir, "seedname", 0.25).unwrap();
        // A synthetic R=0 survives support redistribution and carries the energy origin.
        assert_eq!(model.ham[[0, 0, 0]], Complex64::new(-0.25, 0.0));
        fs::write(
            format!("{dir}seedname_r.dat"),
            "generated\n1\n1\n1 0 0 1 1 1 0 0 0 0 0\n",
        )
        .unwrap();
        assert!(matches!(
            Model::<false, 3, HasRMatrix>::from_hr(dir, "seedname", 0.0),
            Err(TbError::MissingHermitianConjugate { .. })
        ));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn from_hr_fractional_atoms_may_precede_lattice() {
        let dir = "tests/tmp_w90_block_order/";
        write_minimal_dataset(dir, "C");
        fs::remove_file(format!("{dir}seedname_centres.xyz")).unwrap();
        fs::write(format!("{dir}seedname.win"), "spinors\t = .false.\nbegin atoms_frac\nC 0.25 0 0\nend atoms_frac\nbegin projections\nC:s\nend projections\nbegin unit_cell_cart\n2 0 0\n0 2 0\n0 0 2\nend unit_cell_cart\n").unwrap();
        let model = Model::<false, 3>::from_hr(dir, "seedname", 0.0).unwrap();
        assert_eq!(model.orb.row(0), array![0.25, 0.0, 0.0]);
        assert_eq!(model.atoms[0].position(), array![0.25, 0.0, 0.0]);
        fs::remove_dir_all(dir).unwrap();
    }
}
