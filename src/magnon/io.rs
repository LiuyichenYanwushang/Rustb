//! hr-style real-space exchange input/output for [`Magnon`].
//!
//! [`Magnon::write_hr`] writes a pair of files:
//!
//! * `<prefix>_magnon_hr.dat` — a wannier90-`_hr.dat`-style matrix element
//!   list. The first content line holds `nsta nR`; every following line is
//!   `Rx Ry Rz a b J` with 1-based matrix indices and a real `J` in eV.
//!   Lines starting with `#` and blank lines are ignored by the reader; CRLF
//!   line endings and tab separators are accepted. The file must list every
//!   `(translation, a, b)` entry exactly once, and the first listed
//!   translation must be `R = 0` (a [`Magnon::validate`] invariant).
//! * `<prefix>_magnon_meta.toml` — the geometry, spin data and reference
//!   structure needed to rebuild a [`Magnon`]: lattice, site positions, spin
//!   quantum numbers, reference axes, propagation vector, and the unit /
//!   exchange conventions used by the matrix file.
//!
//! Splitting the matrix from its metadata follows the Wannier90 convention
//! (`_hr.dat` plus `.win`/`_centres.xyz`): the matrix file stays a plain
//! `R`-resolved block list, while [`Magnon::read`] reconstructs the model from
//! both files. [`Magnon::from_hr`] is the metadata-free entry point for callers
//! that already know the geometry and the reference structure.
//!
//! # Conventions recorded by the metadata
//!
//! | field | value | meaning |
//! |---|---|---|
//! | `energy_unit` | `eV` | exchange tensors and frequencies |
//! | `spin_unit` | `hbar` | `spin[i]` is $S\_i=\lvert m\_i\rvert/2$ |
//! | `exchange_convention` | `unit-vector` | `J` is TB2J's landscape coefficient |
//!
//! # Examples
//!
//! ```rust
//! use ndarray::{array, Array3};
//! use Rustb::magnon::{Magnon, MagnonReference};
//!
//! # fn main() -> Rustb::Result<()> {
//! let lat = array![[1.0]];
//! let orb = array![[0.0]];
//! let spin = array![1.5];
//! let axis = array![[0.0, 0.0, 1.0]];
//! let ham_r = array![[0isize], [1], [-1]];
//! let mut ham = Array3::<f64>::zeros((3, 3, 3));
//! for d in 0..3 {
//!     ham[[1, d, d]] = 0.01;
//!     ham[[2, d, d]] = 0.01;
//! }
//! let magnon = Magnon::<1>::new(
//!     lat,
//!     orb,
//!     spin,
//!     axis,
//!     ham,
//!     ham_r,
//!     MagnonReference::collinear(),
//! )?;
//!
//! let dir = std::env::temp_dir().join(format!("rustb-magnon-doc-{}", std::process::id()));
//! let dir = dir.to_string_lossy().into_owned();
//! magnon.write_hr(&dir, "model")?;
//! let restored = Magnon::<1>::read(&dir, "model")?;
//! std::fs::remove_dir_all(&dir).ok();
//! assert_eq!(restored.ham, magnon.ham);
//! # Ok(())
//! # }
//! ```

use super::{Magnon, MagnonReference, SPIN_COMPONENTS};
use crate::error::{Result, TbError};
use ndarray::{Array1, Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Version of the Rustb magnon hr format written by [`Magnon::write_hr`].
pub const MAGNON_FORMAT_VERSION: u32 = 1;

/// Files produced by [`Magnon::write_hr`].
#[derive(Clone, Debug)]
pub struct MagnonFiles {
    /// Path of the `_magnon_hr.dat` matrix element list.
    pub hr: PathBuf,
    /// Path of the `_magnon_meta.toml` metadata file.
    pub meta: PathBuf,
}

/// Metadata half of an hr pair.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MagnonMeta {
    format_version: u32,
    dim: usize,
    nspin: usize,
    energy_unit: String,
    spin_unit: String,
    exchange_convention: String,
    lat: Vec<Vec<f64>>,
    orb: Vec<Vec<f64>>,
    spin: Vec<f64>,
    axis: Vec<Vec<f64>>,
    q: Vec<f64>,
    rotation_axis: Vec<f64>,
    quantization_axis: Vec<f64>,
}

impl<const DIM: usize> Magnon<DIM> {
    /// Write the hr pair for this model into `dir` with the given file prefix.
    ///
    /// The prefix is checked first, then the model is validated; nothing is
    /// written when either fails. Both files are written, and existing files
    /// with the same names are replaced. `prefix` is a plain file stem: empty
    /// names, `.`/`..` and names containing path separators are rejected so
    /// that the pair cannot escape `dir`.
    pub fn write_hr(&self, dir: &str, prefix: &str) -> Result<MagnonFiles> {
        check_prefix(prefix)?;
        self.validate()?;
        fs::create_dir_all(dir).map_err(|e| TbError::DirectoryCreation {
            path: dir.to_string(),
            message: e.to_string(),
        })?;
        let hr = hr_path(dir, prefix);
        let meta = meta_path(dir, prefix);
        self.write_hr_body(&hr)?;
        self.write_meta(&meta)?;
        Ok(MagnonFiles { hr, meta })
    }

    /// Read an hr pair written by [`Magnon::write_hr`].
    pub fn read(dir: &str, prefix: &str) -> Result<Self> {
        check_prefix(prefix)?;
        let meta_path = meta_path(dir, prefix);
        let text = fs::read_to_string(&meta_path).map_err(|e| TbError::FileParse {
            file: meta_path.display().to_string(),
            message: e.to_string(),
        })?;
        let meta: MagnonMeta = toml::from_str(&text).map_err(|e| TbError::FileParse {
            file: meta_path.display().to_string(),
            message: e.to_string(),
        })?;
        let file = meta_path.display().to_string();
        if meta.format_version != MAGNON_FORMAT_VERSION {
            return Err(TbError::FileParse {
                file,
                message: format!(
                    "unsupported format_version {}, expected {MAGNON_FORMAT_VERSION}",
                    meta.format_version
                ),
            });
        }
        if meta.dim != DIM {
            return Err(TbError::FileParse {
                file: file.clone(),
                message: format!("expected dim = {DIM}, found {}", meta.dim),
            });
        }
        if meta.energy_unit != "eV" {
            return Err(TbError::FileParse {
                file: file.clone(),
                message: format!(
                    "expected energy_unit = \"eV\", found \"{}\"",
                    meta.energy_unit
                ),
            });
        }
        if meta.spin_unit != "hbar" {
            return Err(TbError::FileParse {
                file: file.clone(),
                message: format!(
                    "expected spin_unit = \"hbar\", found \"{}\"",
                    meta.spin_unit
                ),
            });
        }
        if meta.exchange_convention != "unit-vector" {
            return Err(TbError::FileParse {
                file: file.clone(),
                message: format!(
                    "expected exchange_convention = \"unit-vector\", found \"{}\"",
                    meta.exchange_convention
                ),
            });
        }
        let lat = array2_from_rows(&meta.lat, DIM, &meta_path, "lat")?;
        let orb = array2_from_rows(&meta.orb, DIM, &meta_path, "orb")?;
        let axis = array2_from_rows(&meta.axis, SPIN_COMPONENTS, &meta_path, "axis")?;
        if orb.nrows() != meta.nspin {
            return Err(TbError::FileParse {
                file: file.clone(),
                message: format!(
                    "nspin = {} does not match the {} site positions",
                    meta.nspin,
                    orb.nrows()
                ),
            });
        }
        if axis.nrows() != meta.nspin {
            return Err(TbError::FileParse {
                file: file.clone(),
                message: format!(
                    "nspin = {} does not match the {} reference axes",
                    meta.nspin,
                    axis.nrows()
                ),
            });
        }
        if meta.spin.len() != meta.nspin {
            return Err(TbError::FileParse {
                file,
                message: format!(
                    "nspin = {} does not match the {} spin values",
                    meta.nspin,
                    meta.spin.len()
                ),
            });
        }
        let spin = Array1::from(meta.spin.clone());
        let reference = MagnonReference {
            q: Array1::from(meta.q.clone()),
            rotation_axis: Array1::from(meta.rotation_axis.clone()),
            quantization_axis: Array1::from(meta.quantization_axis.clone()),
        };
        Self::from_hr(dir, prefix, lat, orb, spin, axis, reference)
    }

    /// Read `<prefix>_magnon_hr.dat` with caller-supplied geometry.
    ///
    /// This is the metadata-free entry point: the lattice, site positions,
    /// spin quantum numbers, reference axes and reference structure are not
    /// stored in the matrix file, exactly as Wannier90 keeps its geometry
    /// outside `_hr.dat`. A mismatched geometry cannot be detected here beyond
    /// the matrix dimension `nsta = 3 * orb.nrows()`.
    pub fn from_hr(
        dir: &str,
        prefix: &str,
        lat: Array2<f64>,
        orb: Array2<f64>,
        spin: Array1<f64>,
        axis: Array2<f64>,
        reference: MagnonReference<DIM>,
    ) -> Result<Self> {
        check_prefix(prefix)?;
        let path = hr_path(dir, prefix);
        let text = fs::read_to_string(&path).map_err(|e| TbError::FileParse {
            file: path.display().to_string(),
            message: e.to_string(),
        })?;
        let (hamR, ham) = parse_hr_body::<DIM>(&text, &path, 3 * orb.nrows())?;
        Self::new(lat, orb, spin, axis, ham, hamR, reference)
    }

    fn write_hr_body(&self, path: &Path) -> Result<()> {
        let nsta = self.nsta();
        let nR = self.nR();
        let mut text = String::with_capacity(96 + nR * nsta * nsta * 40);
        text.push_str(&format!(
            "# Rustb magnon HR v{MAGNON_FORMAT_VERSION}: J_ij(R) in eV, \
             exchange_convention=unit-vector, 3*nspin x 3*nspin blocks, 1-based indices\n"
        ));
        text.push_str(&format!("{nsta} {nR}\n"));
        for iR in 0..nR {
            let [r0, r1, r2] = translation3(self.hamR.row(iR));
            for a in 0..nsta {
                for b in 0..nsta {
                    let value = self.ham[[iR, a, b]];
                    text.push_str(&format!(
                        "{r0:4} {r1:4} {r2:4} {:4} {:4} {value:.16e}\n",
                        a + 1,
                        b + 1
                    ));
                }
            }
        }
        fs::write(path, text).map_err(|e| TbError::FileCreation {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }

    fn write_meta(&self, path: &Path) -> Result<()> {
        let meta = MagnonMeta {
            format_version: MAGNON_FORMAT_VERSION,
            dim: DIM,
            nspin: self.nspin(),
            energy_unit: "eV".to_string(),
            spin_unit: "hbar".to_string(),
            exchange_convention: "unit-vector".to_string(),
            lat: rows_of(&self.lat),
            orb: rows_of(&self.orb),
            spin: self.spin.to_vec(),
            axis: rows_of(&self.axis),
            q: self.reference.q.to_vec(),
            rotation_axis: self.reference.rotation_axis.to_vec(),
            quantization_axis: self.reference.quantization_axis.to_vec(),
        };
        let text = toml::to_string(&meta).map_err(|e| TbError::FileCreation {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let header = "# Rustb magnon metadata (geometry, spins, reference structure)\n";
        fs::write(path, format!("{header}{text}")).map_err(|e| TbError::FileCreation {
            path: path.display().to_string(),
            message: e.to_string(),
        })
    }
}

fn check_prefix(prefix: &str) -> Result<()> {
    let invalid = prefix.is_empty()
        || prefix == "."
        || prefix == ".."
        || prefix.contains('/')
        || prefix.contains('\\');
    if invalid {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "file_prefix",
            message: format!("\"{prefix}\" must be a plain file stem without path separators"),
        });
    }
    Ok(())
}

fn hr_path(dir: &str, prefix: &str) -> PathBuf {
    PathBuf::from(dir).join(format!("{prefix}_magnon_hr.dat"))
}

fn meta_path(dir: &str, prefix: &str) -> PathBuf {
    PathBuf::from(dir).join(format!("{prefix}_magnon_meta.toml"))
}

fn rows_of(array: &Array2<f64>) -> Vec<Vec<f64>> {
    array.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn translation3(row: ndarray::ArrayView1<isize>) -> [isize; 3] {
    let mut out = [0isize; 3];
    for (slot, value) in out.iter_mut().zip(row.iter()) {
        *slot = *value;
    }
    out
}

fn array2_from_rows(
    rows: &[Vec<f64>],
    ncols: usize,
    path: &Path,
    what: &str,
) -> Result<Array2<f64>> {
    let file = path.display().to_string();
    if let Some(bad) = rows.iter().position(|row| row.len() != ncols) {
        return Err(TbError::FileParse {
            file,
            message: format!(
                "{what} row {bad} has {} entries, expected {ncols}",
                rows[bad].len()
            ),
        });
    }
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    Array2::from_shape_vec((rows.len(), ncols), flat).map_err(|e| TbError::FileParse {
        file,
        message: format!("cannot reshape {what}: {e}"),
    })
}

/// Parse the `_hr` body, returning `(hamR, ham)`.
///
/// The matrix dimension is cross-checked against `expected_nsta`; `DIM` sizes
/// the translation table. Every `(translation, a, b)` entry must appear
/// exactly once, and translations with nonzero out-of-plane components are
/// rejected for `DIM < 3`.
fn parse_hr_body<const DIM: usize>(
    text: &str,
    path: &Path,
    expected_nsta: usize,
) -> Result<(Array2<isize>, Array3<f64>)> {
    let file = path.display().to_string();
    let parse_error = |line: usize, message: String| TbError::FileParse {
        file: file.clone(),
        message: format!("line {line}: {message}"),
    };
    if !(1..=3).contains(&DIM) {
        return Err(TbError::InvalidDimension {
            dim: DIM,
            supported: vec![1, 2, 3],
        });
    }

    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                Some((index + 1, trimmed))
            }
        })
        .collect();

    let (header_line, header) = lines.first().ok_or_else(|| TbError::FileParse {
        file: file.clone(),
        message: "empty hr file".into(),
    })?;
    let tokens: Vec<&str> = header.split_whitespace().collect();
    if tokens.len() != 2 {
        return Err(parse_error(
            *header_line,
            format!(
                "the first content line must be 'nsta nR', found {} tokens",
                tokens.len()
            ),
        ));
    }
    let nsta: usize = tokens[0]
        .parse()
        .map_err(|_| parse_error(*header_line, format!("invalid nsta '{}'", tokens[0])))?;
    let nR: usize = tokens[1]
        .parse()
        .map_err(|_| parse_error(*header_line, format!("invalid nR '{}'", tokens[1])))?;
    if nsta != expected_nsta {
        return Err(parse_error(
            *header_line,
            format!("matrix dimension {nsta} does not match 3*nspin = {expected_nsta}"),
        ));
    }
    if nsta == 0 || nR == 0 {
        return Err(parse_error(
            *header_line,
            "nsta and nR must both be positive".into(),
        ));
    }
    let expected = nsta
        .checked_mul(nsta)
        .and_then(|blocks| blocks.checked_mul(nR))
        .filter(|&entries| entries <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| {
            parse_error(
                *header_line,
                format!("nsta = {nsta} nR = {nR} is too large"),
            )
        })?;

    let data = &lines[1..];
    if data.len() != expected {
        return Err(parse_error(
            *header_line,
            format!(
                "expected {expected} matrix element lines, found {}",
                data.len()
            ),
        ));
    }

    let mut hamR = Array2::<isize>::zeros((nR, DIM));
    let mut ham = Array3::<f64>::zeros((nR, nsta, nsta));
    let mut filled = vec![false; expected];
    let mut index_of: HashMap<[isize; 3], usize> = HashMap::new();
    for (line_number, line) in data {
        let columns: Vec<&str> = line.split_whitespace().collect();
        if columns.len() != 6 {
            return Err(parse_error(
                *line_number,
                format!("expected 6 columns, found {}", columns.len()),
            ));
        }
        let mut r = [0isize; 3];
        for (slot, token) in r.iter_mut().zip(columns.iter().take(3)) {
            *slot = token
                .parse()
                .map_err(|_| parse_error(*line_number, format!("invalid translation '{token}'")))?;
        }
        for &component in r.iter().skip(DIM) {
            if component != 0 {
                return Err(parse_error(
                    *line_number,
                    format!(
                        "translation component {component} must be zero for a {DIM}-dimensional model"
                    ),
                ));
            }
        }
        let a: usize = columns[3].parse().map_err(|_| {
            parse_error(*line_number, format!("invalid row index '{}'", columns[3]))
        })?;
        let b: usize = columns[4].parse().map_err(|_| {
            parse_error(
                *line_number,
                format!("invalid column index '{}'", columns[4]),
            )
        })?;
        let value: f64 = columns[5].parse().map_err(|_| {
            parse_error(
                *line_number,
                format!("invalid matrix element '{}'", columns[5]),
            )
        })?;
        if !value.is_finite() {
            return Err(parse_error(
                *line_number,
                format!("matrix element {value} is not finite"),
            ));
        }
        if a == 0 || a > nsta || b == 0 || b > nsta {
            return Err(parse_error(
                *line_number,
                format!("matrix indices ({a}, {b}) must be within 1..={nsta}"),
            ));
        }
        let iR = match index_of.get(&r) {
            Some(&index) => index,
            None => {
                let index = index_of.len();
                if index >= nR {
                    return Err(parse_error(
                        *line_number,
                        format!("more than nR = {nR} distinct translations"),
                    ));
                }
                for axis in 0..DIM {
                    hamR[[index, axis]] = r[axis];
                }
                index_of.insert(r, index);
                index
            }
        };
        let entry = (iR * nsta + (a - 1)) * nsta + (b - 1);
        if filled[entry] {
            return Err(parse_error(
                *line_number,
                format!("duplicate entry for R = {r:?}, row {a}, column {b}"),
            ));
        }
        filled[entry] = true;
        ham[[iR, a - 1, b - 1]] = value;
    }
    // Both checks below are defensive: with exactly nsta^2*nR unique entries and
    // at most nR distinct translations, a short translation list necessarily
    // duplicates an entry (rejected above), and a complete list fills every slot.
    if index_of.len() != nR {
        return Err(parse_error(
            *header_line,
            format!(
                "found only {} distinct translations, expected nR = {nR}",
                index_of.len()
            ),
        ));
    }
    if let Some(entry) = filled.iter().position(|&value| !value) {
        let b = entry % nsta + 1;
        let a = entry / nsta % nsta + 1;
        let iR = entry / (nsta * nsta);
        return Err(parse_error(
            *header_line,
            format!("missing entry for translation slot {iR}, row {a}, column {b}"),
        ));
    }
    if hamR.row(0).iter().any(|&component| component != 0) {
        return Err(parse_error(
            *header_line,
            format!(
                "the first listed translation must be R = 0, found {:?}",
                hamR.row(0).to_vec()
            ),
        ));
    }
    Ok((hamR, ham))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::magnon::Magnon;
    use ndarray::{Array2, Array3, array};

    fn temp_dir(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!("rustb-magnon-{}-{tag}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    /// One-site ferromagnetic chain with nearest-neighbour exchange `j`.
    fn chain(j: f64) -> Magnon<1> {
        let lat = array![[1.0]];
        let orb = array![[0.0]];
        let spin = array![1.5];
        let axis = array![[0.0, 0.0, 1.0]];
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        for d in 0..3 {
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

    /// Nine matrix elements of the single translation `R = 0`.
    fn single_cell_body() -> String {
        let mut body = String::from("3 1\n");
        for a in 1..=3 {
            for b in 1..=3 {
                body.push_str(&format!("0 0 0 {a} {b} 0.0\n"));
            }
        }
        body
    }

    fn parse(body: &str, tag: &str) -> Result<(Array2<isize>, Array3<f64>)> {
        let dir = temp_dir(tag);
        std::fs::create_dir_all(&dir).unwrap();
        let path = hr_path(&dir, "probe");
        std::fs::write(&path, body).unwrap();
        let result = parse_hr_body::<1>(body, &path, 3);
        std::fs::remove_dir_all(&dir).ok();
        result
    }

    fn message_of(error: &TbError) -> String {
        match error {
            TbError::FileParse { message, .. } => message.clone(),
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn test_hr_round_trip() {
        let dir = temp_dir("round-trip");
        let magnon = chain(0.01);
        let files = magnon.write_hr(&dir, "chain").unwrap();
        assert!(files.hr.exists());
        assert!(files.meta.exists());

        let restored = Magnon::<1>::read(&dir, "chain").unwrap();
        assert_eq!(restored.hamR, magnon.hamR);
        assert_eq!(restored.ham, magnon.ham);
        assert_eq!(restored.spin, magnon.spin);
        assert_eq!(restored.axis, magnon.axis);
        assert_eq!(restored.lat, magnon.lat);
        assert_eq!(restored.orb, magnon.orb);
        assert_eq!(restored.reference.q, magnon.reference.q);
        assert_eq!(
            restored.reference.quantization_axis,
            magnon.reference.quantization_axis
        );
        assert_eq!(
            restored.reference.rotation_axis,
            magnon.reference.rotation_axis
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hr_round_trip_2d_noncollinear() {
        let dir = temp_dir("round-trip-2d");
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.5, 0.5]];
        let spin = array![1.5, 1.5];
        let axis = array![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
        let ham_r = array![[0isize, 0], [1, 0], [-1, 0], [0, 1], [0, -1]];
        // General 3x3 tensors with DMI-like antisymmetric parts.
        let j = [
            [-0.02, 0.003, -0.001],
            [-0.003, -0.02, 0.002],
            [0.001, -0.002, -0.02],
        ];
        let mut ham = Array3::<f64>::zeros((5, 6, 6));
        for i_r in 1..5 {
            for a in 0..3 {
                for b in 0..3 {
                    ham[[i_r, a, 3 + b]] = j[a][b];
                    ham[[i_r, 3 + b, a]] = j[a][b];
                }
            }
        }
        let mut reference = MagnonReference::<2>::collinear();
        reference.q = array![0.25, 0.0];
        reference.quantization_axis = array![0.0, 1.0, 0.0];
        let magnon = Magnon::<2>::new(lat, orb, spin, axis, ham, ham_r, reference).unwrap();
        magnon.write_hr(&dir, "square").unwrap();
        let restored = Magnon::<2>::read(&dir, "square").unwrap();
        assert_eq!(restored.hamR, magnon.hamR);
        assert_eq!(restored.ham, magnon.ham);
        assert_eq!(restored.reference.q, magnon.reference.q);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_from_hr_with_explicit_metadata() {
        let dir = temp_dir("explicit-meta");
        let magnon = chain(-0.02);
        magnon.write_hr(&dir, "afm").unwrap();

        let restored = Magnon::<1>::from_hr(
            &dir,
            "afm",
            magnon.lat.clone(),
            magnon.orb.clone(),
            magnon.spin.clone(),
            magnon.axis.clone(),
            MagnonReference::collinear(),
        )
        .unwrap();
        assert_eq!(restored.ham, magnon.ham);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_rejects_wrong_dimension() {
        let dir = temp_dir("wrong-dim");
        chain(0.01).write_hr(&dir, "model").unwrap();
        let error = Magnon::<2>::read(&dir, "model").unwrap_err();
        assert!(matches!(error, TbError::FileParse { .. }));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_rejects_missing_meta() {
        let dir = temp_dir("missing-meta");
        let error = Magnon::<1>::read(&dir, "absent").unwrap_err();
        assert!(matches!(error, TbError::FileParse { .. }));
    }

    #[test]
    fn test_read_rejects_broken_toml() {
        let dir = temp_dir("broken-toml");
        chain(0.01).write_hr(&dir, "model").unwrap();
        std::fs::write(meta_path(&dir, "model"), "format_version = [").unwrap();
        let error = Magnon::<1>::read(&dir, "model").unwrap_err();
        assert!(matches!(error, TbError::FileParse { .. }));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_meta_rejects_wrong_conventions() {
        for (old, new) in [("eV", "meV"), ("hbar", "muB"), ("unit-vector", "operator")] {
            let dir = temp_dir(&format!("convention-{old}"));
            chain(0.01).write_hr(&dir, "model").unwrap();
            let path = meta_path(&dir, "model");
            let text = std::fs::read_to_string(&path).unwrap();
            std::fs::write(&path, text.replace(old, new)).unwrap();
            let error = Magnon::<1>::read(&dir, "model").unwrap_err();
            assert!(matches!(error, TbError::FileParse { .. }), "{old} -> {new}");
            assert!(
                message_of(&error).contains("expected"),
                "{old} -> {new}: {error}"
            );
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    #[test]
    fn test_meta_rejects_wrong_format_version() {
        let dir = temp_dir("format-version");
        chain(0.01).write_hr(&dir, "model").unwrap();
        let path = meta_path(&dir, "model");
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::write(
            &path,
            text.replace("format_version = 1", "format_version = 2"),
        )
        .unwrap();
        let error = Magnon::<1>::read(&dir, "model").unwrap_err();
        assert!(message_of(&error).contains("format_version"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_write_hr_rejects_escaping_prefix() {
        let magnon = chain(0.01);
        let dir = temp_dir("prefix");
        for prefix in ["", ".", "..", "a/b", "a\\b", "../x"] {
            assert!(matches!(
                magnon.write_hr(&dir, prefix),
                Err(TbError::InvalidMagnonInvariant { .. })
            ));
        }
    }

    #[test]
    fn test_hr_rejects_wrong_matrix_dimension() {
        let dir = temp_dir("wrong-nsta");
        chain(0.01).write_hr(&dir, "model").unwrap();
        let error = Magnon::<1>::from_hr(
            &dir,
            "model",
            array![[1.0]],
            array![[0.0], [0.5]],
            array![1.5, 1.5],
            array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            MagnonReference::collinear(),
        )
        .unwrap_err();
        assert!(
            message_of(&error).contains("does not match 3*nspin"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hr_rejects_out_of_range_index() {
        let mut body = String::from("3 1\n");
        for a in 1..=3 {
            for b in 1..=3 {
                let column = if a == 1 && b == 2 { 4 } else { b };
                body.push_str(&format!("0 0 0 {a} {column} 0.0\n"));
            }
        }
        let error = parse(&body, "bad-index").unwrap_err();
        assert!(
            message_of(&error).contains("must be within 1..=3"),
            "{error}"
        );
    }

    #[test]
    fn test_hr_rejects_duplicate_entry() {
        let mut body = String::from("3 1\n");
        for a in 1..=3 {
            for b in 1..=3 {
                let (row, column) = if a == 1 && b == 2 { (1, 1) } else { (a, b) };
                body.push_str(&format!("0 0 0 {row} {column} 0.0\n"));
            }
        }
        let error = parse(&body, "duplicate").unwrap_err();
        assert!(message_of(&error).contains("duplicate entry"), "{error}");
    }

    #[test]
    fn test_hr_rejects_non_finite_element() {
        let mut body = String::from("3 1\n");
        for a in 1..=3 {
            for b in 1..=3 {
                let value = if a == 2 && b == 2 { "NaN" } else { "0.0" };
                body.push_str(&format!("0 0 0 {a} {b} {value}\n"));
            }
        }
        let error = parse(&body, "non-finite").unwrap_err();
        assert!(message_of(&error).contains("not finite"), "{error}");
    }

    #[test]
    fn test_hr_rejects_header_extra_tokens() {
        let body = format!("3 1 1\n{}", single_cell_body().split_once('\n').unwrap().1);
        let error = parse(&body, "header").unwrap_err();
        assert!(message_of(&error).contains("nsta nR"), "{error}");
    }

    #[test]
    fn test_hr_rejects_zero_dimension() {
        let dir = temp_dir("zero-nsta");
        std::fs::create_dir_all(&dir).unwrap();
        let path = hr_path(&dir, "probe");
        let body = "0 5\n";
        std::fs::write(&path, body).unwrap();
        let error = parse_hr_body::<1>(body, &path, 0).unwrap_err();
        assert!(
            message_of(&error).contains("must both be positive"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hr_rejects_out_of_plane_translation_for_2d() {
        let dir = temp_dir("out-of-plane");
        std::fs::create_dir_all(&dir).unwrap();
        let path = hr_path(&dir, "probe");
        let body = "3 2\n0 0 0 1 1 0\n0 0 0 1 2 0\n0 0 0 1 3 0\n\
                    0 0 0 2 1 0\n0 0 0 2 2 0\n0 0 0 2 3 0\n\
                    0 0 0 3 1 0\n0 0 0 3 2 0\n0 0 0 3 3 0\n\
                    0 0 1 1 1 0\n0 0 1 1 2 0\n0 0 1 1 3 0\n\
                    0 0 1 2 1 0\n0 0 1 2 2 0\n0 0 1 2 3 0\n\
                    0 0 1 3 1 0\n0 0 1 3 2 0\n0 0 1 3 3 0\n";
        std::fs::write(&path, body).unwrap();
        let error = parse_hr_body::<2>(body, &path, 3).unwrap_err();
        assert!(
            message_of(&error).contains("must be zero for a 2-dimensional model"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hr_rejects_dim_greater_than_three() {
        let dir = temp_dir("dim4");
        std::fs::create_dir_all(&dir).unwrap();
        let path = hr_path(&dir, "probe");
        let body = single_cell_body();
        std::fs::write(&path, &body).unwrap();
        let error = parse_hr_body::<4>(&body, &path, 3).unwrap_err();
        assert!(matches!(error, TbError::InvalidDimension { dim: 4, .. }));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_hr_body_accepts_crlf_tabs_and_comments() {
        let mut body = String::from("# header\r\n\r\n  # indented comment\r\n3\t1\r\n");
        for a in 1..=3 {
            for b in 1..=3 {
                let value = match (a, b) {
                    (1, 3) => 0.25,
                    (a, b) if a == b => 0.5,
                    _ => 0.0,
                };
                body.push_str(&format!("0 0 0\t{a}\t{b}\t{value}\r\n"));
            }
        }
        let (ham_r, ham) = parse(&body, "crlf").unwrap();
        assert_eq!(ham_r, Array2::<isize>::zeros((1, 1)));
        assert_eq!(ham[[0, 0, 0]], 0.5);
        assert_eq!(ham[[0, 0, 2]], 0.25);
    }

    #[test]
    fn test_write_hr_rejects_invalid_model_without_touching_disk() {
        let dir = temp_dir("invalid-model");
        let magnon = Magnon::<1> {
            lat: array![[1.0]],
            orb: array![[0.0]],
            spin: array![1.5],
            axis: array![[0.0, 0.0, 1.0]],
            ham: Array3::<f64>::zeros((1, 3, 3)),
            hamR: array![[0isize], [1]],
            reference: MagnonReference::collinear(),
        };
        assert!(matches!(
            magnon.write_hr(&dir, "model"),
            Err(TbError::InvalidMagnonInvariant { .. })
        ));
        assert!(!std::path::Path::new(&dir).exists());
    }

    #[test]
    fn test_hr_rejects_more_translations_than_nr() {
        let mut body = String::from("3 1\n");
        for a in 1..=3 {
            for b in 1..=3 {
                let r = if a == 3 && b == 3 { 1 } else { 0 };
                body.push_str(&format!("{r} 0 0 {a} {b} 0.0\n"));
            }
        }
        let error = parse(&body, "more-r").unwrap_err();
        assert!(message_of(&error).contains("more than nR"), "{error}");
    }

    #[test]
    fn test_hr_rejects_oversized_header() {
        let error = parse("3 18446744073709551615\n", "huge-nr").unwrap_err();
        assert!(message_of(&error).contains("is too large"), "{error}");
    }

    #[test]
    fn test_hr_rejects_r_zero_not_first() {
        let mut body = String::from("3 3\n");
        for i_r in 0..3 {
            for a in 1..=3 {
                for b in 1..=3 {
                    let r = [1isize, 0, -1][i_r];
                    body.push_str(&format!("{r} 0 0 {a} {b} 0.0\n"));
                }
            }
        }
        let error = parse(&body, "r0-not-first").unwrap_err();
        assert!(message_of(&error).contains("must be R = 0"), "{error}");
    }

    #[test]
    fn test_hr_round_trip_3d() {
        let dir = temp_dir("round-trip-3d");
        let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let orb = array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
        let spin = array![1.0, 2.0];
        let axis = array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let ham_r = array![[0isize, 0, 0], [1, 0, 0], [-1, 0, 0]];
        let mut ham = Array3::<f64>::zeros((3, 6, 6));
        for i_r in 1..3 {
            for a in 0..3 {
                for b in 0..3 {
                    let forward = 0.01 * ((a + 1) as f64 - (b as f64) * 0.5);
                    let backward = 0.01 * ((b + 1) as f64 - (a as f64) * 0.5);
                    ham[[i_r, a, 3 + b]] = forward;
                    ham[[i_r, 3 + a, b]] = backward;
                }
            }
        }
        let magnon = Magnon::<3>::new(
            lat,
            orb,
            spin,
            axis,
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap();
        magnon.write_hr(&dir, "bulk").unwrap();
        let restored = Magnon::<3>::read(&dir, "bulk").unwrap();
        assert_eq!(restored.ham, magnon.ham);
        assert_eq!(restored.hamR, magnon.hamR);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_read_rejects_bad_reference_in_meta() {
        let dir = temp_dir("bad-reference");
        chain(0.01).write_hr(&dir, "model").unwrap();
        let path = meta_path(&dir, "model");
        let text = std::fs::read_to_string(&path).unwrap();
        for (old, new, needle) in [
            ("q = [0.0]", "q = []", "propagation_vector_shape"),
            (
                "quantization_axis = [0.0, 0.0, 1.0]",
                "quantization_axis = [0.0, 0.0, 4.0]",
                "reference_axis_norm",
            ),
        ] {
            std::fs::write(&path, text.replace(old, new)).unwrap();
            let message = Magnon::<1>::read(&dir, "model").unwrap_err().to_string();
            assert!(message.contains(needle), "{message}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
