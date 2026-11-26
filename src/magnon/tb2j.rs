//! Reader for TB2J's `exchange.out` text output.
//!
//! TB2J's `wann2J.py` writes the extracted exchange parameters to
//! `exchange.out`: a plain-text file with the cell, an atom table, and one
//! block per ordered site pair `(i, j, R)`.
//!
//! Supports collinear scalar moments and isotropic exchange. Noncollinear
//! inputs are rejected rather than silently dropping tensor terms.

use super::{Magnon, MagnonReference};
use crate::atom_struct::{Atom, AtomType};
use crate::error::{Result, TbError};
use ndarray::{Array1, Array2, Array3};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

// Half a unit in the last printed digit of the four-decimal scalar plus
// three-decimal tensor, with a small floating-point margin (meV).
const PRINTED_VALUE_TOLERANCE_MEV: f64 = 5.5e-4 + 1e-12;
/// Tolerance (angstrom) for the redundant pair vector and distance columns.
const VECTOR_TOLERANCE_ANGSTROM: f64 = 5e-3;

impl Magnon<3> {
    /// Read a collinear TB2J `exchange.out` file as a [`Magnon`] model.
    ///
    /// The magnetic sites come from the optional orbital-decomposition table,
    /// or otherwise the atoms in `Exchange:`, ordered as in `Atoms:`. Positions
    /// are converted from the
    /// file's Cartesian angstrom coordinates to fractional coordinates of the
    /// cell, `spin[i] = |w_magmom| / 2`, and the axes are `+z` for a positive
    /// moment and `-z` for a negative one. Every `J_iso` is converted from meV
    /// to eV and stored as a diagonal `3 x 3` block; the translation list is the
    /// unique set of integer `R` vectors with `R = 0` first. The reference
    /// structure is the collinear one ([`MagnonReference::collinear`]). The
    /// exchange file does not store TB2J's magnon propagation vector, so this
    /// reader assumes `Q = 0`; replace [`Magnon::reference`] after reading when
    /// the magnon run used a finite `Q`.
    ///
    /// The `Atoms:` table is kept as [`Magnon::atoms`] (species from the leading
    /// symbol of names such as `Fe1`, fractional positions, and the collinear
    /// moment) together with the site-to-atom mapping, so symmetry and
    /// atom-resolved analysis keep the crystal identity. Labels whose leading
    /// alphabetic run is not a known element (ghost or pseudo atoms) keep the
    /// exchange model readable but drop the atom metadata.
    ///
    /// # Errors
    /// Returns [`TbError::FileParse`] when the file cannot be read or a section,
    /// number, or site name is malformed. A structured
    /// [`TbError::InvalidMagnonInvariant`] is returned for a noncollinear file
    /// (`tb2j_noncollinear_unsupported`), a single-ion anisotropy table, an
    /// unknown structure that violates the [`Magnon`] invariants, or a missing
    /// reciprocal entry (`tb2j_reciprocal_entry`).
    pub fn from_tb2j_exchange_out<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|error| TbError::FileParse {
            file: path.display().to_string(),
            message: error.to_string(),
        })?;
        parse_exchange_out(&text, path)
    }
}

struct AtomRow {
    name: String,
    position: [f64; 3],
    magmom: f64,
}

#[derive(Clone, Copy)]
struct Entry {
    /// Atom index (row of the `Atoms:` table) of the first spin.
    atom_i: usize,
    /// Atom index of the second spin.
    atom_j: usize,
    /// Cell translation of the second spin.
    r: [isize; 3],
    /// Isotropic exchange in meV.
    jiso_mev: f64,
}

fn parse_exchange_out(text: &str, path: &Path) -> Result<Magnon<3>> {
    let lines: Vec<&str> = text.lines().collect();
    if lines
        .iter()
        .any(|line| line.contains("Single Ion Anisotropy"))
    {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "tb2j_single_ion_anisotropy_unsupported",
            message: format!(
                "{}: single-ion anisotropy is not yet mapped into the onsite Magnon block",
                path.display()
            ),
        });
    }
    let cell_header = find_header(&lines, "Cell (Angstrom):", path)?;
    let lat = parse_cell(&lines, cell_header, path)?;
    let atoms_header = find_header(&lines, "Atoms:", path)?;
    let atoms = parse_atoms(&lines, atoms_header, path)?;
    let listed_sites = parse_listed_sites(&lines, &atoms, path)?;
    let exchange_header = find_header(&lines, "Exchange:", path)?;
    let entries = parse_exchange(&lines, exchange_header, &atoms, &lat, path)?;
    build_model(&atoms, &entries, listed_sites.as_deref(), lat, path)
}

fn fail(path: &Path, message: impl Into<String>) -> TbError {
    TbError::FileParse {
        file: path.display().to_string(),
        message: message.into(),
    }
}

fn noncollinear(path: &Path, detail: impl std::fmt::Display) -> TbError {
    TbError::InvalidMagnonInvariant {
        invariant: "tb2j_noncollinear_unsupported",
        message: format!(
            "{}: {detail}; noncollinear/spinor tensor conversion is not implemented by this reader",
            path.display()
        ),
    }
}

fn find_header(lines: &[&str], header: &str, path: &Path) -> Result<usize> {
    lines
        .iter()
        .position(|line| line.trim() == header)
        .ok_or_else(|| fail(path, format!("missing section header {header:?}")))
}

fn parse_number(token: &str, path: &Path, context: &str) -> Result<f64> {
    let value = token.parse::<f64>().map_err(|_| {
        fail(
            path,
            format!("cannot parse {token:?} as a number in {context}"),
        )
    })?;
    if !value.is_finite() {
        return Err(fail(
            path,
            format!("{context} value {token:?} must be finite"),
        ));
    }
    Ok(value)
}

fn parse_cell(lines: &[&str], header: usize, path: &Path) -> Result<Array2<f64>> {
    let mut rows = Vec::new();
    let mut index = header + 1;
    while rows.len() < 3 {
        let line = lines
            .get(index)
            .ok_or_else(|| fail(path, "truncated Cell section"))?;
        index += 1;
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != 3 {
            return Err(fail(
                path,
                format!("a cell row needs 3 numbers, found {line:?}"),
            ));
        }
        let row = tokens
            .iter()
            .map(|token| parse_number(token, path, "the cell rows"))
            .collect::<Result<Vec<f64>>>()?;
        rows.push([row[0], row[1], row[2]]);
    }
    let mut lat = Array2::<f64>::zeros((3, 3));
    for (row, values) in rows.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            lat[[row, column]] = value;
        }
    }
    Ok(lat)
}

fn parse_atoms(lines: &[&str], header: usize, path: &Path) -> Result<Vec<AtomRow>> {
    let mut index = header + 1;
    let collinear = loop {
        let line = lines
            .get(index)
            .ok_or_else(|| fail(path, "truncated Atoms section"))?;
        index += 1;
        if line.contains("w_magmom") {
            break true;
        }
        if line.contains("M(x)") {
            break false;
        }
    };
    if !collinear {
        return Err(noncollinear(path, "the atom table has vector moments"));
    }
    let mut atoms = Vec::new();
    let mut names = HashSet::new();
    while index < lines.len() {
        let trimmed = lines[index].trim();
        if trimmed.is_empty() || trimmed.starts_with('=') || trimmed.starts_with('-') {
            break;
        }
        if trimmed.starts_with("Single Ion") {
            break;
        }
        index += 1;
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.is_empty() || tokens[0] == "Total" {
            continue;
        }
        if tokens.len() != 6 {
            return Err(fail(path, format!("malformed atom row {trimmed:?}")));
        }
        let numbers = tokens[1..]
            .iter()
            .map(|token| parse_number(token, path, "an atom row"))
            .collect::<Result<Vec<f64>>>()?;
        if !names.insert(tokens[0]) {
            return Err(fail(path, format!("duplicate atom name {:?}", tokens[0])));
        }
        atoms.push(AtomRow {
            name: tokens[0].to_string(),
            position: [numbers[0], numbers[1], numbers[2]],
            magmom: numbers[4],
        });
    }
    if atoms.is_empty() {
        return Err(fail(path, "the Atoms section contains no atom rows"));
    }
    Ok(atoms)
}

/// Magnetic sites listed by the optional orbital-decomposition table.
///
/// When TB2J writes this table it names every magnetic atom, so a truncated
/// exchange table cannot silently drop a site. `Ok(None)` means the table is
/// absent and the sites are inferred from the exchange entries.
fn parse_listed_sites(
    lines: &[&str],
    atoms: &[AtomRow],
    path: &Path,
) -> Result<Option<Vec<usize>>> {
    let Some(header) = lines
        .iter()
        .position(|line| line.trim() == "Orbitals used in decomposition:")
    else {
        return Ok(None);
    };
    let name_to_atom: HashMap<&str, usize> = atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| (atom.name.as_str(), index))
        .collect();
    let mut sites = Vec::new();
    let mut index = header + 1;
    while index < lines.len() {
        let line = lines[index].trim();
        index += 1;
        if line.is_empty() || line.starts_with('=') {
            break;
        }
        if line.starts_with("The name of the orbitals") {
            continue;
        }
        let name = line.split(':').next().unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        let Some(&atom) = name_to_atom.get(name) else {
            return Err(fail(
                path,
                format!("the orbital-decomposition table names an unknown atom {name:?}"),
            ));
        };
        if sites.contains(&atom) {
            return Err(fail(
                path,
                format!("the orbital-decomposition table lists atom {name:?} twice"),
            ));
        }
        sites.push(atom);
    }
    if sites.is_empty() {
        return Ok(None);
    }
    Ok(Some(sites))
}

fn parse_exchange(
    lines: &[&str],
    header: usize,
    atoms: &[AtomRow],
    lat: &Array2<f64>,
    path: &Path,
) -> Result<Vec<Entry>> {
    let name_to_atom: HashMap<&str, usize> = atoms
        .iter()
        .enumerate()
        .map(|(index, atom)| (atom.name.as_str(), index))
        .collect();
    let mut entries: Vec<Entry> = Vec::new();
    let mut seen: HashSet<(usize, usize, [isize; 3])> = HashSet::new();
    let mut current: Option<usize> = None;
    let mut summaries = HashSet::new();
    let mut index = header + 1;
    while index < lines.len() {
        let line = lines[index].trim();
        index += 1;
        if line.is_empty() || line.starts_with('-') || line.starts_with('=') {
            continue;
        }
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() >= 2
            && name_to_atom.contains_key(tokens[0])
            && name_to_atom.contains_key(tokens[1])
        {
            let entry = parse_entry(line, tokens[0], tokens[1], &name_to_atom, atoms, lat, path)?;
            let key = (entry.atom_i, entry.atom_j, entry.r);
            if !seen.insert(key) {
                return Err(fail(
                    path,
                    format!(
                        "duplicate exchange entry for sites {:?} and {:?}, R = {:?}",
                        atoms[entry.atom_i].name, atoms[entry.atom_j].name, entry.r
                    ),
                ));
            }
            entries.push(entry);
            current = Some(entries.len() - 1);
            continue;
        }
        if let Some(rest) = line.strip_prefix("J_iso:") {
            let summary = parse_number(rest.trim(), path, "a J_iso summary line")?;
            let entry = current.ok_or_else(|| {
                fail(
                    path,
                    "a J_iso summary line appears before any exchange entry",
                )
            })?;
            if (summary - entries[entry].jiso_mev).abs() > PRINTED_VALUE_TOLERANCE_MEV {
                return Err(fail(
                    path,
                    format!(
                        "the J_iso summary ({summary} meV) disagrees with the table value ({} meV)",
                        entries[entry].jiso_mev
                    ),
                ));
            }
            if !summaries.insert(entry) {
                return Err(fail(path, "duplicate J_iso summary for one exchange entry"));
            }
            continue;
        }
        if line.contains("NJt") || line.contains("NJT") {
            return Err(noncollinear(
                path,
                "the exchange table contains non-JT exchange fields",
            ));
        }
        if line.contains("Combined J tensor") {
            let entry = current.ok_or_else(|| {
                fail(
                    path,
                    "a Combined J tensor block appears before any exchange entry",
                )
            })?;
            let tensor = consume_matrix(lines, &mut index, path)?;
            check_diagonal_tensor(&tensor, entries[entry].jiso_mev, path)?;
            continue;
        }
        if line.contains("DMI:") {
            let (inside, _) = parenthesized(line)
                .ok_or_else(|| fail(path, format!("malformed DMI line {line:?}")))?;
            let dmi = parse_whitespace_triplet(inside, path, "a DMI vector")?;
            if dmi.iter().any(|value| *value != 0.0) {
                return Err(noncollinear(
                    path,
                    "the exchange table contains a nonzero DMI",
                ));
            }
            continue;
        }
        if line.contains("J_ani") {
            let tensor = consume_matrix(lines, &mut index, path)?;
            if tensor.iter().flatten().any(|value| *value != 0.0) {
                return Err(noncollinear(
                    path,
                    "the exchange table contains an anisotropic tensor",
                ));
            }
            continue;
        }
        if tokens.get(2).is_some_and(|token| token.starts_with('(')) {
            return Err(fail(
                path,
                format!(
                    "an exchange row names an unknown atom: {:?} -> {:?}",
                    tokens[0], tokens[1]
                ),
            ));
        }
        // Strain derivatives and temporary debug fields do not enter the
        // exchange model; every other line is an unknown section and is
        // ignored, because TB2J versions append fields such as dJ/dx.
    }
    if entries.is_empty() {
        return Err(fail(path, "the Exchange section contains no entries"));
    }
    if summaries.len() != entries.len() {
        return Err(fail(
            path,
            format!(
                "the Exchange section has {} entries but {} J_iso summary lines; the file is \
                 truncated or malformed",
                entries.len(),
                summaries.len()
            ),
        ));
    }
    for entry in &entries {
        let mut reverse = [0; 3];
        for (out, &r) in reverse.iter_mut().zip(&entry.r) {
            *out = r
                .checked_neg()
                .ok_or_else(|| fail(path, "exchange R cannot be negated"))?;
        }
        if !seen.contains(&(entry.atom_j, entry.atom_i, reverse)) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "tb2j_reciprocal_entry",
                message: format!(
                    "{}: missing reverse entry for {} -> {}, R = {:?}",
                    path.display(),
                    atoms[entry.atom_i].name,
                    atoms[entry.atom_j].name,
                    entry.r
                ),
            });
        }
    }
    Ok(entries)
}

fn parse_entry(
    line: &str,
    first: &str,
    second: &str,
    name_to_atom: &HashMap<&str, usize>,
    atoms: &[AtomRow],
    lat: &Array2<f64>,
    path: &Path,
) -> Result<Entry> {
    let after_first = line
        .find(first)
        .map(|start| start + first.len())
        .ok_or_else(|| fail(path, format!("malformed exchange line {line:?}")))?;
    let after_second = line[after_first..]
        .find(second)
        .map(|start| after_first + start + second.len())
        .ok_or_else(|| fail(path, format!("malformed exchange line {line:?}")))?;
    let rest = &line[after_second..];
    let (r_text, after_r) =
        parenthesized(rest).ok_or_else(|| fail(path, format!("missing R in {line:?}")))?;
    let r = parse_integer_triplet(r_text, path, "an R vector")?;
    let after_r = after_r.trim_start();
    let jiso_text = after_r
        .split('(')
        .next()
        .map(str::trim)
        .ok_or_else(|| fail(path, format!("missing J_iso in {line:?}")))?;
    let jiso_mev = parse_number(jiso_text, path, "an exchange entry")?;
    let (vector_text, after_vector) = parenthesized(after_r)
        .ok_or_else(|| fail(path, format!("missing pair vector in {line:?}")))?;
    let vector = parse_comma_triplet(vector_text, path, "a pair vector")?;
    let distance_text = after_vector
        .split_whitespace()
        .next()
        .ok_or_else(|| fail(path, "an exchange entry is missing its pair distance"))?;
    let distance = parse_number(distance_text, path, "the pair distance")?;
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || (norm - distance).abs() > VECTOR_TOLERANCE_ANGSTROM {
        return Err(fail(
            path,
            format!("the pair distance {distance} disagrees with the vector in {line:?}"),
        ));
    }
    let atom_i = name_to_atom[first];
    let atom_j = name_to_atom[second];
    for component in 0..3 {
        let expected = atoms[atom_j].position[component] - atoms[atom_i].position[component]
            + (0..3)
                .map(|axis| r[axis] as f64 * lat[[axis, component]])
                .sum::<f64>();
        if !expected.is_finite() || (expected - vector[component]).abs() > VECTOR_TOLERANCE_ANGSTROM
        {
            return Err(fail(
                path,
                format!(
                    "the pair vector of {} -> {} (R = {:?}) does not match the atom positions \
                     and the cell",
                    atoms[atom_i].name, atoms[atom_j].name, r
                ),
            ));
        }
    }
    Ok(Entry {
        atom_i,
        atom_j,
        r,
        jiso_mev,
    })
}

/// Text between the first `(` and its matching `)`, and the remainder after it.
fn parenthesized(text: &str) -> Option<(&str, &str)> {
    let open = text.find('(')?;
    let rest = &text[open + 1..];
    let close = rest.find(')')?;
    Some((&rest[..close], &rest[close + 1..]))
}

fn parse_integer_triplet(text: &str, path: &Path, context: &str) -> Result<[isize; 3]> {
    let values = text
        .split(',')
        .map(|part| {
            part.trim().parse::<isize>().map_err(|_| {
                fail(
                    path,
                    format!("cannot parse {:?} as an integer in {context}", part.trim()),
                )
            })
        })
        .collect::<Result<Vec<isize>>>()?;
    if values.len() != 3 {
        return Err(fail(
            path,
            format!("{context} needs 3 components, found {}", values.len()),
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn parse_comma_triplet(text: &str, path: &Path, context: &str) -> Result<[f64; 3]> {
    let values = text
        .split(',')
        .map(|part| parse_number(part.trim(), path, context))
        .collect::<Result<Vec<f64>>>()?;
    finish_triplet(values, path, context)
}

fn parse_whitespace_triplet(text: &str, path: &Path, context: &str) -> Result<[f64; 3]> {
    let values = text
        .split_whitespace()
        .map(|token| parse_number(token, path, context))
        .collect::<Result<Vec<f64>>>()?;
    finish_triplet(values, path, context)
}

fn finish_triplet(values: Vec<f64>, path: &Path, context: &str) -> Result<[f64; 3]> {
    if values.len() != 3 {
        return Err(fail(
            path,
            format!("{context} needs 3 components, found {}", values.len()),
        ));
    }
    Ok([values[0], values[1], values[2]])
}

fn consume_matrix(lines: &[&str], index: &mut usize, path: &Path) -> Result<[[f64; 3]; 3]> {
    let mut numbers = Vec::new();
    let mut depth = 0usize;
    let mut opened = false;
    while *index < lines.len() {
        let line = lines[*index];
        *index += 1;
        for character in line.chars() {
            match character {
                '[' => {
                    opened = true;
                    depth += 1;
                }
                ']' => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        fail(path, "unmatched closing bracket in an exchange tensor")
                    })?;
                }
                _ => {}
            }
        }
        let cleaned: String = line
            .chars()
            .map(|character| match character {
                '[' | ']' | ',' => ' ',
                other => other,
            })
            .collect();
        for token in cleaned.split_whitespace() {
            numbers.push(parse_number(token, path, "an exchange tensor")?);
            if numbers.len() > 9 {
                return Err(fail(path, "an exchange tensor has more than 9 entries"));
            }
        }
        if opened && depth == 0 {
            break;
        }
    }
    if !opened || depth != 0 || numbers.len() != 9 {
        return Err(fail(
            path,
            format!(
                "an exchange tensor needs balanced brackets and 9 numbers, found {}",
                numbers.len()
            ),
        ));
    }
    Ok([
        [numbers[0], numbers[1], numbers[2]],
        [numbers[3], numbers[4], numbers[5]],
        [numbers[6], numbers[7], numbers[8]],
    ])
}

fn check_diagonal_tensor(tensor: &[[f64; 3]; 3], jiso_mev: f64, path: &Path) -> Result<()> {
    if tensor[1][1] != tensor[0][0] || tensor[2][2] != tensor[0][0] {
        return Err(noncollinear(
            path,
            "the combined tensor has unequal diagonal entries",
        ));
    }
    for (row, values) in tensor.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            let expected = if row == column { jiso_mev } else { 0.0 };
            if (row != column && value != 0.0)
                || (row == column && (value - expected).abs() > PRINTED_VALUE_TOLERANCE_MEV)
            {
                return Err(noncollinear(
                    path,
                    format!(
                        "the combined tensor entry [{row}][{column}] = {value} meV is not the \
                         diagonal isotropic value {jiso_mev} meV"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn build_model(
    atoms: &[AtomRow],
    entries: &[Entry],
    listed_sites: Option<&[usize]>,
    lat: Array2<f64>,
    path: &Path,
) -> Result<Magnon<3>> {
    let mut used = vec![false; atoms.len()];
    for entry in entries {
        used[entry.atom_i] = true;
        used[entry.atom_j] = true;
    }
    let mut magnetic: Vec<usize> = match listed_sites {
        Some(sites) => sites.to_vec(),
        None => (0..atoms.len()).filter(|&index| used[index]).collect(),
    };
    magnetic.sort_unstable();
    if magnetic.is_empty() {
        return Err(fail(
            path,
            "no magnetic sites occur in the Exchange section",
        ));
    }
    let inverse = invert3(&lat).ok_or_else(|| TbError::InvalidMagnonInvariant {
        invariant: "tb2j_cell",
        message: format!("{}: the TB2J cell has no finite inverse", path.display()),
    })?;
    let n = magnetic.len();
    let mut atom_to_site = vec![usize::MAX; atoms.len()];
    let mut orb = Array2::<f64>::zeros((n, 3));
    let mut spin = Array1::<f64>::zeros(n);
    let mut axis = Array2::<f64>::zeros((n, 3));
    for (site, &atom_index) in magnetic.iter().enumerate() {
        atom_to_site[atom_index] = site;
        let atom = &atoms[atom_index];
        if !atom.magmom.is_finite() || atom.magmom == 0.0 {
            return Err(fail(
                path,
                format!(
                    "magnetic site {} has a zero or non-finite w_magmom",
                    atom.name
                ),
            ));
        }
        spin[site] = 0.5 * atom.magmom.abs();
        axis[[site, 2]] = atom.magmom.signum();
        let fractional = Array1::from(atom.position.to_vec()).dot(&inverse);
        for component in 0..3 {
            orb[[site, component]] = fractional[component];
        }
    }

    let mut translations: Vec<[isize; 3]> = vec![[0, 0, 0]];
    let mut translation_index: HashMap<[isize; 3], usize> = HashMap::new();
    translation_index.insert([0, 0, 0], 0);
    for entry in entries {
        if let std::collections::hash_map::Entry::Vacant(slot) = translation_index.entry(entry.r) {
            slot.insert(translations.len());
            translations.push(entry.r);
        }
    }
    let mut ham = Array3::<f64>::zeros((translations.len(), 3 * n, 3 * n));
    for entry in entries {
        let cell = translation_index[&entry.r];
        let site_i = atom_to_site[entry.atom_i];
        let site_j = atom_to_site[entry.atom_j];
        if site_i == usize::MAX || site_j == usize::MAX {
            return Err(fail(
                path,
                format!(
                    "exchange entry {} -> {} is not covered by the magnetic-site table",
                    atoms[entry.atom_i].name, atoms[entry.atom_j].name
                ),
            ));
        }
        let coupling = entry.jiso_mev * 1e-3;
        for component in 0..3 {
            ham[[cell, 3 * site_i + component, 3 * site_j + component]] = coupling;
        }
    }
    let mut ham_r = Array2::<isize>::zeros((translations.len(), 3));
    for (row, translation) in translations.iter().enumerate() {
        for (column, &value) in translation.iter().enumerate() {
            ham_r[[row, column]] = value;
        }
    }
    // Keep the crystal: TB2J's atom table gives the species, positions and
    // moments that the magnetic sites were selected from. `AtomType` is a
    // closed element enum, so a pseudo/ghost label would make the whole file
    // unreadable; the matrix data is the contract and atom metadata stays
    // optional, so an unknown species drops the metadata instead of failing.
    let site_atoms = parse_tb2j_atoms(atoms, &inverse);
    let magnon = Magnon::new(
        lat,
        orb,
        spin,
        axis,
        ham,
        ham_r,
        MagnonReference::collinear(),
    )?;
    if site_atoms.is_empty() {
        // Unknown species: keep the matrix data and drop the optional
        // metadata rather than failing a previously readable file.
        Ok(magnon)
    } else {
        magnon
            .with_atoms(site_atoms, magnetic)
            .map_err(|error| fail(path, error.to_string()))
    }
}

/// Build the crystal atom table from TB2J's `Atoms:` rows.
///
/// Returns an empty vector when any label has no known element symbol, so the
/// exchange model stays readable without atom metadata. TB2J labels atoms by
/// element plus a counter (`Fe1`), so the element is the leading alphabetic
/// run. A nonzero `w_magmom` becomes the collinear moment along `+z`; a zero
/// moment stays `None` because the table does not distinguish a measured zero
/// from an unknown moment.
fn parse_tb2j_atoms(atoms: &[AtomRow], inverse: &Array2<f64>) -> Vec<Atom> {
    let mut parsed = Vec::with_capacity(atoms.len());
    for row in atoms {
        let symbol: String = row
            .name
            .chars()
            .take_while(|character| character.is_ascii_alphabetic())
            .collect();
        let Ok(species) = symbol.parse::<AtomType>() else {
            return Vec::new();
        };
        let fractional = Array1::from(row.position.to_vec()).dot(inverse);
        let mut atom = Atom::new(fractional, species);
        if row.magmom != 0.0 && atom.set_magnetic_moment([0.0, 0.0, row.magmom]).is_err() {
            return Vec::new();
        }
        parsed.push(atom);
    }
    parsed
}
/// Inverse of a 3 x 3 matrix, or `None` when it is singular or non-finite.
fn invert3(matrix: &Array2<f64>) -> Option<Array2<f64>> {
    let a = matrix;
    let determinant = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
        - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
        + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
    if !determinant.is_finite() || determinant.abs() < 1e-12 {
        return None;
    }
    let scale = 1.0 / determinant;
    let mut inverse = Array2::<f64>::zeros((3, 3));
    inverse[[0, 0]] = (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]]) * scale;
    inverse[[0, 1]] = (a[[0, 2]] * a[[2, 1]] - a[[0, 1]] * a[[2, 2]]) * scale;
    inverse[[0, 2]] = (a[[0, 1]] * a[[1, 2]] - a[[0, 2]] * a[[1, 1]]) * scale;
    inverse[[1, 0]] = (a[[1, 2]] * a[[2, 0]] - a[[1, 0]] * a[[2, 2]]) * scale;
    inverse[[1, 1]] = (a[[0, 0]] * a[[2, 2]] - a[[0, 2]] * a[[2, 0]]) * scale;
    inverse[[1, 2]] = (a[[0, 2]] * a[[1, 0]] - a[[0, 0]] * a[[1, 2]]) * scale;
    inverse[[2, 0]] = (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]) * scale;
    inverse[[2, 1]] = (a[[0, 1]] * a[[2, 0]] - a[[0, 0]] * a[[2, 1]]) * scale;
    inverse[[2, 2]] = (a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]) * scale;
    Some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magnon::MagnonSolve;
    use ndarray::array;
    use std::path::PathBuf;

    fn fixture(atoms: &str, exchange: &str) -> String {
        format!(
            "============================================================\n\
             Information:\n\
             Executed:  wann2J.py\n\
             ============================================================\n\
             Cell (Angstrom):\n2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0\n\n\
             ============================================================\n\
             Atoms:\n\
             (Note: charge and magmoms only count the wannier functions.)\n\
             Atom number        x             y             z       w_charge  w_magmom \n\
             {atoms}\
             Total 16.0000 0.0000\n\n\
             ============================================================\n\
             Exchange:\n\
                 i      j          R        J_iso(meV)          vector          distance(A) \n\
             --------------------------------------------------------\n\
             {exchange}"
        )
    }

    fn dimer_exchange() -> String {
        "   Fe1   Fe2   (  0,   0,   0) -10.0000   ( 1.000,  0.000,  0.000)  1.000 \n\
         J_iso: -10.0000 \n\
         --------------------------------------------------------\n\
         \x20  Fe2   Fe1   (  0,   0,   0) -10.0000   (-1.000,  0.000,  0.000)  1.000 \n\
         J_iso: -10.0000 \n"
            .to_string()
    }

    fn dimer_atoms() -> &'static str {
        "Fe1 0.00000000 0.00000000 0.00000000 8.0000 2.0000\n\
         Fe2 1.00000000 0.00000000 0.00000000 8.0000 -2.0000\n"
    }

    fn temp_file(name: &str, body: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("rustb-tb2j-{}-{name}.out", std::process::id()));
        std::fs::write(&path, body).expect("write fixture");
        path
    }

    #[test]
    fn rejects_nonfinite_and_malformed_redundant_values() {
        let exchange = dimer_exchange();
        let path = Path::new("invalid-values.out");
        for token in ["NaN", "inf", "-inf", "garbage"] {
            for invalid in [
                exchange.replacen("J_iso: -10.0000", &format!("J_iso: {token}"), 1),
                exchange.replacen("( 1.000,", &format!("( {token},"), 1),
                exchange.replacen(")  1.000", &format!(")  {token}"), 1),
                format!("{exchange}\nDMI: ({token} 0 0)\n"),
                format!("{exchange}\nCombined J tensor:\n[[-10 {token} 0] [0 -10 0] [0 0 -10]]\n"),
            ] {
                assert!(
                    matches!(
                        parse_exchange_out(&fixture(dimer_atoms(), &invalid), path),
                        Err(TbError::FileParse { .. })
                    ),
                    "accepted invalid redundant value {token}: {invalid}"
                );
            }
        }
        let missing_distance = exchange.replace(")  1.000 ", ") ");
        assert!(parse_exchange_out(&fixture(dimer_atoms(), &missing_distance), path).is_err());
    }

    #[test]
    fn rejects_duplicate_summaries_and_weak_missing_reverse() {
        let summary = "J_iso: -10.0000 \n";
        let duplicate = dimer_exchange().replace(summary, "").replacen(
            "--------------------------------------------------------",
            &format!("{summary}{summary}--------------------------------------------------------"),
            1,
        );
        let path = Path::new("incomplete-entries.out");
        assert!(matches!(
            parse_exchange_out(&fixture(dimer_atoms(), &duplicate), path),
            Err(TbError::FileParse { .. })
        ));
        for j in ["0.0000", "0.0005", "-10.0000"] {
            let exchange = format!("Fe1 Fe2 (0, 0, 0) {j} (1, 0, 0) 1\nJ_iso: {j}\n");
            assert!(matches!(
                parse_exchange_out(&fixture(dimer_atoms(), &exchange), path),
                Err(TbError::InvalidMagnonInvariant {
                    invariant: "tb2j_reciprocal_entry",
                    ..
                })
            ));
        }
    }

    #[test]
    fn rejects_incomplete_tensors_and_resolved_anisotropy() {
        let path = Path::new("invalid-tensor.out");
        for suffix in [
            "Combined J tensor:\n[[-10 0 0] [0 -10 0] [0 0 -10]\n",
            "Combined J tensor:\n[[-10 BAD 0 0] [0 -10 0] [0 0 -10]]\n",
            "Combined J tensor:\n[[-10 0 0] [0 -10 0] [0 0 -10 0]]\n",
        ] {
            let exchange = format!("{}{suffix}", dimer_exchange());
            assert!(matches!(
                parse_exchange_out(&fixture(dimer_atoms(), &exchange), path),
                Err(TbError::FileParse { .. })
            ));
        }
        for suffix in [
            "DMI: (0.0009 0 0)\n",
            "J_ani:\n[[0.001 0 0] [0 0 0] [0 0 0]]\n",
            "Combined J tensor:\n[[-10 0.004 0] [0 -10 0] [0 0 -10]]\n",
            "Combined J tensor:\n[[-10.004 0 0] [0 -10 0] [0 0 -10]]\n",
        ] {
            let exchange = format!("{}{suffix}", dimer_exchange());
            assert!(matches!(
                parse_exchange_out(&fixture(dimer_atoms(), &exchange), path),
                Err(TbError::InvalidMagnonInvariant {
                    invariant: "tb2j_noncollinear_unsupported",
                    ..
                })
            ));
        }
    }

    #[test]
    fn rejects_duplicate_names_and_keeps_atom_table_order() {
        let path = Path::new("site-order.out");
        let duplicates = format!("{}Fe1 0 0 0 8 2\n", dimer_atoms());
        assert!(matches!(
            parse_exchange_out(&fixture(&duplicates, &dimer_exchange()), path),
            Err(TbError::FileParse { .. })
        ));
        let body = fixture(dimer_atoms(), &dimer_exchange()).replace(
            "Exchange:\n",
            "Orbitals used in decomposition:\nFe2: ['d']\nFe1: ['d']\n\nExchange:\n",
        );
        let model = parse_exchange_out(&body, path).unwrap();
        assert_eq!(model.orb.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(model.axis.column(2).to_vec(), vec![1.0, -1.0]);
    }

    #[test]
    fn unknown_species_keeps_the_exchange_metadata_free() {
        // A ghost or pseudo label cannot name an AtomType; the matrix data is
        // still the contract, so the model is returned without atom metadata.
        let exchange = dimer_exchange().replace("Fe1", "Xx1").replace("Fe2", "Xx2");
        let atoms = dimer_atoms().replace("Fe1", "Xx1").replace("Fe2", "Xx2");
        let body = fixture(&atoms, &exchange);
        let magnon = parse_exchange_out(&body, Path::new("pseudo.out")).unwrap();
        assert_eq!(magnon.nspin(), 2);
        assert!(!magnon.has_atoms());
        assert_eq!(magnon.atom_of_site.len(), 0);
    }

    #[test]
    fn zero_moment_atoms_stay_in_the_crystal_without_a_moment() {
        let atoms = format!("{}Ge3 0.5 0.5 0.5 6.0 0.0\n", dimer_atoms());
        let body = fixture(&atoms, &dimer_exchange());
        let magnon = parse_exchange_out(&body, Path::new("zero-moment.out")).unwrap();
        // Non-magnetic atoms are kept in the crystal but are not sites, and a
        // zero moment stays unknown rather than becoming a measured zero.
        assert_eq!(magnon.natom(), 3);
        assert_eq!(magnon.atom_of_site, vec![0, 1]);
        assert_eq!(magnon.atoms[2].atom_type(), AtomType::Ge);
        assert_eq!(magnon.atoms[2].magnetic_moment(), None);
    }
    #[test]
    fn reads_skew_cell_chain_with_nonunit_spin() {
        let exchange = "Fe1 Fe1 (1, 0, 0) 10 (2, 1, 0) 2.236\nJ_iso: 10\n\
                        Fe1 Fe1 (-1, 0, 0) 10 (-2, -1, 0) 2.236\nJ_iso: 10\n";
        let body = fixture("Fe1 0.5 1.0 0.0 8.0 4.0\n", exchange).replace(
            "2.0 0.0 0.0\n0.0 2.0 0.0\n0.0 0.0 2.0",
            "2.0 1.0 0.0\n0.0 3.0 0.0\n0.0 0.0 2.0",
        );
        let model = parse_exchange_out(&body, Path::new("chain.out")).unwrap();
        assert_eq!(model.spin.to_vec(), vec![2.0]);
        assert_eq!(model.orb.row(0).to_vec(), vec![0.25, 0.25, 0.0]);
        // The atom table goes through the same row-vector cell inverse.
        assert_eq!(model.atoms[0].position().to_vec(), vec![0.25, 0.25, 0.0]);
        assert_eq!(model.hamR.row(0).to_vec(), vec![0, 0, 0]);
        for q in [0.125, 0.25, 0.37] {
            let actual = model.magnon_band_onek(&array![q, 0.0, 0.0]).unwrap();
            let expected = 4.0 * 0.01 / 2.0 * (1.0 - (std::f64::consts::TAU * q).cos());
            assert!((actual[0] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn combined_tensor_requires_one_rounded_isotropic_diagonal() {
        let exchange = dimer_exchange().replace("-10.0000", "-10.0005");
        let path = Path::new("rounded-tensor.out");
        for diagonal in ["-10.000", "-10.001"] {
            let input = format!(
                "{exchange}\nCombined J tensor:\n[[{diagonal} 0 0] [0 {diagonal} 0] [0 0 {diagonal}]]\n[Testing!] DMI: (0 0 0)\n"
            );
            parse_exchange_out(&fixture(dimer_atoms(), &input), path).unwrap();
        }
        let anisotropic = format!(
            "{exchange}\nCombined J tensor:\n[[-10.000 0 0] [0 -10.001 0] [0 0 -10.000]]\n"
        );
        assert!(matches!(
            parse_exchange_out(&fixture(dimer_atoms(), &anisotropic), path),
            Err(TbError::InvalidMagnonInvariant {
                invariant: "tb2j_noncollinear_unsupported",
                ..
            })
        ));
    }

    #[test]
    fn rejects_unknown_sites_before_their_summaries_can_credit_another_entry() {
        for (first, second) in [("Missing", "Fe2"), ("Fe1", "Missing"), ("Ghost", "Other")] {
            let exchange = dimer_exchange().replacen(
                "J_iso: -10.0000",
                &format!("{first} {second} (0, 0, 0) -10 (1, 0, 0) 1\nJ_iso: -10.0000"),
                1,
            );
            assert!(matches!(
                parse_exchange_out(
                    &fixture(dimer_atoms(), &exchange),
                    Path::new("unknown-site.out")
                ),
                Err(TbError::FileParse { .. })
            ));
        }
    }

    #[test]
    fn reads_collinear_dimer() {
        let body = fixture(dimer_atoms(), &dimer_exchange());
        let path = temp_file("dimer", &body);
        let magnon = Magnon::<3>::from_tb2j_exchange_out(&path).expect("parse dimer");
        let _ = std::fs::remove_file(&path);
        assert_eq!(magnon.nspin(), 2);
        assert_eq!(magnon.nR(), 1);
        // The atom table is kept, with the leading symbol from names like Fe1.
        assert!(magnon.has_atoms());
        assert_eq!(
            (magnon.natom(), magnon.atom_of_site.clone()),
            (2, vec![0, 1])
        );
        assert_eq!(
            magnon.atoms[0].atom_type(),
            crate::atom_struct::AtomType::Fe
        );
        assert_eq!(magnon.atoms[0].position().to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(magnon.atoms[1].position().to_vec(), vec![0.5, 0.0, 0.0]);
        assert_eq!(magnon.atoms[0].magnetic_moment(), Some([0.0, 0.0, 2.0]));
        assert_eq!(magnon.atoms[1].magnetic_moment(), Some([0.0, 0.0, -2.0]));
        assert_eq!(magnon.hamR.row(0).to_vec(), vec![0, 0, 0]);
        assert_eq!(magnon.spin.to_vec(), vec![1.0, 1.0]);
        assert_eq!(magnon.axis.row(0).to_vec(), vec![0.0, 0.0, 1.0]);
        assert_eq!(magnon.axis.row(1).to_vec(), vec![0.0, 0.0, -1.0]);
        assert_eq!(magnon.orb.row(0).to_vec(), vec![0.0, 0.0, 0.0]);
        assert_eq!(magnon.orb.row(1).to_vec(), vec![0.5, 0.0, 0.0]);
        for component in 0..3 {
            assert!((magnon.ham[[0, component, 3 + component]] + 0.01).abs() < 1e-12);
            assert!((magnon.ham[[0, 3 + component, component]] + 0.01).abs() < 1e-12);
        }
        let bands = magnon
            .magnon_band_onek(&array![0.0, 0.0, 0.0])
            .expect("dimer should be a stable antiferromagnetic reference");
        println!("dimer bands (eV) = {bands:?}");
        assert_eq!(bands.len(), 2);
        assert!(bands.iter().all(|energy| energy.is_finite()));
        assert!(bands[0].abs() < 1e-6);
    }

    #[test]
    fn accepts_diagonal_combined_tensor() {
        let combined = "Combined J tensor (meV) [J = Jiso*I + DMI_antisymmetric + Jani_symmetric]:\n\
                        [[-10.000  0.000  0.000]\n\
                        \x20[ 0.000 -10.000  0.000]\n\
                        \x20[ 0.000  0.000 -10.000]]\n";
        let exchange = format!(
            "{}{}{}",
            dimer_exchange().replace(
                "J_iso: -10.0000 \n",
                &format!("J_iso: -10.0000 \n{combined}")
            ),
            "",
            ""
        );
        let body = fixture(dimer_atoms(), &exchange);
        let path = temp_file("combined", &body);
        let magnon = Magnon::<3>::from_tb2j_exchange_out(&path).expect("diagonal combined tensor");
        let _ = std::fs::remove_file(&path);
        assert_eq!(magnon.nspin(), 2);
    }

    #[test]
    fn rejects_off_diagonal_combined_tensor() {
        let combined = "Combined J tensor (meV) [J = Jiso*I + DMI_antisymmetric + Jani_symmetric]:\n\
                        [[-10.000  0.500  0.000]\n\
                        \x20[-0.500 -10.000  0.000]\n\
                        \x20[ 0.000  0.000 -10.000]]\n";
        let exchange = dimer_exchange().replace(
            "J_iso: -10.0000 \n",
            &format!("J_iso: -10.0000 \n{combined}"),
        );
        let body = fixture(dimer_atoms(), &exchange);
        let path = temp_file("offdiag", &body);
        let error = Magnon::<3>::from_tb2j_exchange_out(&path).expect_err("off-diagonal tensor");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            TbError::InvalidMagnonInvariant {
                invariant: "tb2j_noncollinear_unsupported",
                ..
            }
        ));
    }

    #[test]
    fn rejects_vector_moments() {
        let atoms = "Fe1 0.00000000 0.00000000 0.00000000 8.0000 0.0000 0.0000 2.0000\n\
                     Fe2 1.00000000 0.00000000 0.00000000 8.0000 0.0000 0.0000 -2.0000\n";
        let body = format!(
            "{}\n",
            fixture(atoms, &dimer_exchange()).replace(
                "Atom number        x             y             z       w_charge  w_magmom ",
                "Atom_number        x             y             z       w_charge  M(x)      M(y)      M(z)    "
            )
        );
        let path = temp_file("vector", &body);
        let error = Magnon::<3>::from_tb2j_exchange_out(&path).expect_err("vector moments");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            TbError::InvalidMagnonInvariant {
                invariant: "tb2j_noncollinear_unsupported",
                ..
            }
        ));
    }

    #[test]
    fn rejects_single_ion_anisotropy() {
        let body = fixture(dimer_atoms(), &dimer_exchange())
            + "\nSingle Ion Anisotropy (meV):\n(k1 is the anisotropy constant)\n";
        let path = temp_file("sia", &body);
        let error = Magnon::<3>::from_tb2j_exchange_out(&path).expect_err("SIA");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            TbError::InvalidMagnonInvariant {
                invariant: "tb2j_single_ion_anisotropy_unsupported",
                ..
            }
        ));
    }

    #[test]
    fn rejects_missing_reciprocal_partner() {
        let exchange = "   Fe1   Fe2   (  0,   0,   0) -10.0000   ( 1.000,  0.000,  0.000)  1.000 \n\
                        J_iso: -10.0000 \n";
        let body = fixture(dimer_atoms(), exchange);
        let path = temp_file("missing-reverse", &body);
        let error = Magnon::<3>::from_tb2j_exchange_out(&path).expect_err("missing reverse");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            TbError::InvalidMagnonInvariant {
                invariant: "tb2j_reciprocal_entry",
                ..
            } | TbError::MissingHermitianConjugateExchange { .. }
        ));
    }

    #[test]
    fn rejects_missing_section() {
        let body = fixture(dimer_atoms(), &dimer_exchange()).replace("Exchange:", "NoExchange:");
        let path = temp_file("missing-section", &body);
        let error = Magnon::<3>::from_tb2j_exchange_out(&path).expect_err("missing exchange");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(error, TbError::FileParse { .. }));
    }

    #[test]
    #[ignore = "requires TB2J_EXCHANGE_OUT pointing at an MgFe6Ge6 exchange.out"]
    fn reads_external_mgfe6ge6_exchange_out() {
        let path = std::env::var("TB2J_EXCHANGE_OUT").expect("TB2J_EXCHANGE_OUT");
        let magnon = Magnon::<3>::from_tb2j_exchange_out(&path).expect("parse external file");
        assert_eq!(magnon.nspin(), 6);
        for spin in magnon.spin.iter() {
            assert!((spin - 1.3914).abs() < 2e-3);
        }
        let max = magnon
            .ham
            .iter()
            .fold(0.0f64, |acc, &value| acc.max(value.abs()));
        assert!((max - 0.0247825).abs() < 1e-6, "max |J| = {max}");
        let mut interlayer = 0.0;
        for row in 0..magnon.nR() {
            for i in 0..magnon.nspin() {
                for j in 0..magnon.nspin() {
                    if magnon.axis[[i, 2]] > 0.0 && magnon.axis[[j, 2]] < 0.0 {
                        interlayer += magnon.ham[[row, 3 * i, 3 * j]];
                    }
                }
            }
        }
        let per_iron = 1e3 * interlayer / 3.0;
        assert!(
            (per_iron - 3.4151).abs() < 1e-2,
            "interlayer sum = {per_iron} meV/Fe"
        );
    }
}
