use crate::atom_struct::{Atom, AtomType, OrbProj, OrbitalId};
use crate::error::{Result, TbError};
use crate::{HasRMatrix, Model, RMatrixData, find_R};
use ndarray::prelude::*;
use ndarray_linalg::Inverse;
use num_complex::Complex64;
use std::collections::HashSet;
use std::str::{FromStr, SplitWhitespace};

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

fn field<T: FromStr>(fields: &mut SplitWhitespace<'_>, file: &str, name: &str) -> Result<T> {
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

fn finite_field(fields: &mut SplitWhitespace<'_>, file: &str) -> Result<f64> {
    let value: f64 = field(fields, file, "numeric component")?;
    if !value.is_finite() {
        return Err(parse_error(file, "Non-finite numeric component"));
    }
    Ok(value)
}

fn record_key(
    fields: &mut SplitWhitespace<'_>,
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
        let (translation, _, _) = record_key(&mut rows[start].split_whitespace(), file, nsta)?;
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
        for row in &rows[start..start + nsta * nsta] {
            let mut fields = row.split_whitespace();
            let (r, i, j) = record_key(&mut fields, file, nsta)?;
            if r != translation || !seen.insert((i, j)) {
                return Err(parse_error(
                    file,
                    "Inconsistent R block or duplicate orbital record",
                ));
            }
            // The first Wannier90 index is the matrix row, regardless of record order.
            data.ham[[index, i, j]] = Complex64::new(
                finite_field(&mut fields, file)?,
                finite_field(&mut fields, file)?,
            ) / weight as f64;
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
    if nsta != hr.ham.shape()[1] || nr == 0 {
        return Err(parse_error(file, "Position-matrix state/R count mismatch"));
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
    // Preserve sparse position support: only its own declared blocks are
    // required; HR translations absent from this file start with zero entries.
    let mut result = Array4::zeros((hr.translations.nrows(), 3, nsta, nsta));
    let mut seen = HashSet::new();
    for block in rows[..records].chunks(nsta * nsta) {
        let (translation, _, _) = record_key(&mut block[0].split_whitespace(), file, nsta)?;
        let index = find_R(&hr.translations, &Array1::from_vec(translation.to_vec()))
            .filter(|&i| hr.weights[i] > 0)
            .ok_or_else(|| parse_error(file, "R vector not found in Hamiltonian"))?;
        for row in block {
            let mut fields = row.split_whitespace();
            let (r, i, j) = record_key(&mut fields, file, nsta)?;
            if r != translation || !seen.insert((r, i, j)) {
                return Err(parse_error(
                    file,
                    "Inconsistent R block or duplicate orbital record",
                ));
            }
            for axis in 0..3 {
                result[[index, axis, i, j]] = Complex64::new(
                    finite_field(&mut fields, file)?,
                    finite_field(&mut fields, file)?,
                ) / hr.weights[index] as f64;
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
        let mut orb = Array2::zeros((norb, 3));
        for index in 0..entries {
            let mut fields = data_line(&mut lines, file)?.split_whitespace();
            let species: String = field(&mut fields, file, "XYZ label")?;
            let position = array![
                finite_field(&mut fields, file)?,
                finite_field(&mut fields, file)?,
                finite_field(&mut fields, file)?
            ]
            .dot(&inverse);
            if index < norb {
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
    let mut projections = Vec::new();
    let mut assigned = Vec::new();
    let mut fallback = Array2::zeros((0, 3));
    let mut dropped = HashSet::new();
    for (species, position) in atoms {
        let first = projections.len();
        for (projected_species, orbitals) in &win.projections {
            if species == projected_species {
                projections.extend(orbitals.iter().copied());
            }
        }
        if first == projections.len() {
            if dropped.insert(species.to_str()) {
                eprintln!(
                    "warning: dropping '{species}' atoms (no orbitals in the projections block)"
                );
            }
            continue;
        }
        if centres.is_none() {
            for _ in first..projections.len() {
                fallback.push_row(position.view())?;
            }
        }
        assigned.push(Atom::with_orbitals(
            position.clone(),
            *species,
            (first..projections.len()).map(OrbitalId::new),
        ));
    }
    if projections.len() != norb {
        return Err(parse_error(
            file,
            format!(
                "species mismatch between atom data and projections: HR declares {norb} orbitals, but {} could be assigned to atoms",
                projections.len()
            ),
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
/// `DIM` must be 3. `HasRMatrix` requires `_r.dat` (`write_rmn=true`), while
/// `NoRMatrix` skips it. `_centres.xyz` and `_wsvec.dat` are optional; without
/// centres, the atomic positions from `.win` supply the orbital positions.
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

impl<const SPIN: bool, const DIM: usize> Model<SPIN, DIM, HasRMatrix> {
    /// Load a tight-binding model from Wannier90 files including position matrix elements.
    ///
    /// This is a convenience wrapper around [`Wannier90::from_hr`] that requires
    /// the Wannier90 `_r.dat` file to be present. Returns `Model<SPIN, DIM, HasRMatrix>`
    /// with position matrix elements (`rmatrix`) populated.
    ///
    /// # Errors
    ///
    /// Returns `TbError::FileCreation` if `_r.dat` is missing.
    pub fn from_hr_with_rmatrix(path: &str, file_name: &str, zero_energy: f64) -> Result<Self> {
        <Self as Wannier90>::from_hr(path, file_name, zero_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
