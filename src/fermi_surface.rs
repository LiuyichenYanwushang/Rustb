//! Fermi surface visualization using marching squares (2D) and marching
//! tetrahedra (3D).
//!
//! Provides two traits:
//! - [`FermiSurface`]: 2D contour / 3D isosurface of E(k) = E_F
//! - [`FermiSurfacePlane`]: 2D Fermi surface slice on a specified k‑plane
//!   (3D models only)
//!
//! The underlying algorithms are classical isosurface extraction methods:
//!
//! - **2D**: [Marching squares] divides the k‑mesh into square cells. For each
//!   cell, the four corner energies are compared against E_F (above/below)
//!   to form a 4‑bit case index, which is looked up in a precomputed edge
//!   table. Linear interpolation along each active edge gives the exact
//!   contour crossing point.
//! - **3D**: [Marching tetrahedra] decomposes each cubic voxel of the k‑mesh
//!   into 5 tetrahedra, avoiding the topological ambiguities of the full
//!   marching‑cubes algorithm. Each tetrahedron has 4 vertices → 16 cases.
//!   Active cases produce 1–2 triangles, again with linear edge interpolation.
//!
//! Both methods return a set of line segments (2D) or triangles (3D) that
//! are then rendered to PDF via gnuplot.
//!
//! [Marching squares]: https://en.wikipedia.org/wiki/Marching_squares
//! [Marching tetrahedra]: https://en.wikipedia.org/wiki/Marching_tetrahedra

use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::kplane::gen_kplane;
use crate::kpoints::gen_kmesh;
use crate::solve_ham::solve;
use ndarray::prelude::*;
use ndarray::*;
use rayon::prelude::*;
use std::fs;
use std::io::Write;
use ndarray_linalg::Inverse;
use std::f64::consts::PI as PI_64;
use std::process::Command;

// ── Marching squares (2D) ────────────────────────────────────────────

/// Edge connection table for marching squares on a 2D grid.
///
/// Cell corners: c0=(i,j), c1=(i+1,j), c2=(i+1,j+1), c3=(i,j+1)
/// Edges: 0=c0-c1(bottom), 1=c1-c2(right), 2=c2-c3(top), 3=c3-c0(left)
/// Case = Σ bit_n << n, bit_n = energy[corner_n] >= e_fermi
///
/// Each row gives up to 4 edge-index pairs (line segments), -1 terminated.
const MS_EDGES: [[i8; 5]; 16] = [
    [-1, -1, -1, -1, -1], // case  0
    [0, 3, -1, -1, -1],   // case  1
    [0, 1, -1, -1, -1],   // case  2
    [1, 3, -1, -1, -1],   // case  3
    [1, 2, -1, -1, -1],   // case  4
    [0, 1, 2, 3, -1],     // case  5  saddle: 0-1, 2-3
    [0, 2, -1, -1, -1],   // case  6
    [2, 3, -1, -1, -1],   // case  7
    [2, 3, -1, -1, -1],   // case  8
    [0, 2, -1, -1, -1],   // case  9
    [0, 3, 1, 2, -1],     // case 10  saddle: 0-3, 1-2
    [1, 2, -1, -1, -1],   // case 11
    [1, 3, -1, -1, -1],   // case 12
    [0, 1, -1, -1, -1],   // case 13
    [0, 3, -1, -1, -1],   // case 14
    [-1, -1, -1, -1, -1], // case 15
];

/// Find E = E_F contour line segments on a 2D energy grid using marching
/// squares.
///
/// Returns a list of segments.  Each segment is a pair of k‑points (in the
/// same coordinate system as the input `kvec`).
fn marching_squares_2d(
    energy: &Array2<f64>, // shape (n2, n1) — row-major
    kvec: &Array2<f64>,   // shape (n1*n2, dim)
    n1: usize,
    n2: usize,
    e_fermi: f64,
) -> Vec<(Array1<f64>, Array1<f64>)> {
    let mut segments = Vec::new();

    for j in 0..n2.saturating_sub(1) {
        for i in 0..n1.saturating_sub(1) {
            let idx00 = i + j * n1;
            let idx10 = (i + 1) + j * n1;
            let idx11 = (i + 1) + (j + 1) * n1;
            let idx01 = i + (j + 1) * n1;

            let e00 = energy[[j, i]];
            let e10 = energy[[j, i + 1]];
            let e11 = energy[[j + 1, i + 1]];
            let e01 = energy[[j + 1, i]];

            let case = ((e00 >= e_fermi) as usize)
                | (((e10 >= e_fermi) as usize) << 1)
                | (((e11 >= e_fermi) as usize) << 2)
                | (((e01 >= e_fermi) as usize) << 3);

            let edges = &MS_EDGES[case];

            // Edges: 0=bottom(c0-c1), 1=right(c1-c2), 2=top(c3-c2), 3=left(c0-c3)
            let corners = [&idx00, &idx10, &idx11, &idx01];
            let edge_pairs: [(usize, usize); 4] = [(0, 1), (1, 2), (3, 2), (0, 3)];
            let edge_vals: [(f64, f64); 4] = [(e00, e10), (e10, e11), (e01, e11), (e00, e01)];

            let mut ei = 0;
            while ei < 5 && edges[ei] != -1 {
                let e_a = edges[ei] as usize;
                let e_b = edges[ei + 1] as usize;

                let (ca, cb) = edge_pairs[e_a];
                let (va, vb) = edge_vals[e_a];
                let p_a = interpolate_edge(
                    kvec.row(*corners[ca]),
                    kvec.row(*corners[cb]),
                    va,
                    vb,
                    e_fermi,
                );
                let (ca, cb) = edge_pairs[e_b];
                let (va, vb) = edge_vals[e_b];
                let p_b = interpolate_edge(
                    kvec.row(*corners[ca]),
                    kvec.row(*corners[cb]),
                    va,
                    vb,
                    e_fermi,
                );

                segments.push((p_a, p_b));
                ei += 2;
            }
        }
    }

    segments
}

/// Linear interpolation along an edge between two k‑points.
fn interpolate_edge(
    ka: ArrayView1<f64>,
    kb: ArrayView1<f64>,
    va: f64,
    vb: f64,
    e_fermi: f64,
) -> Array1<f64> {
    let denom = vb - va;
    let t = if denom.abs() < 1e-14 {
        0.5
    } else {
        (e_fermi - va) / denom
    };
    &ka + &((&kb - &ka) * t)
}

// ── Marching tetrahedra (3D) ─────────────────────────────────────────

/// Tetrahedron edges for a 4‑vertex cell.
///
/// Vertices: v0, v1, v2, v3
/// Edges: 0=v0-v1, 1=v1-v2, 2=v2-v0, 3=v0-v3, 4=v1-v3, 5=v2-v3
const TET_EDGE_PAIRS: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];

/// Marching-tetrahedra case table.
///
/// Each row gives up to 6 edge indices (2 triangles × 3 edges), -1 terminated.
const TET_CASES: [[i8; 7]; 16] = [
    [-1, -1, -1, -1, -1, -1, -1], // case  0
    [0, 2, 3, -1, -1, -1, -1],    // case  1
    [0, 1, 4, -1, -1, -1, -1],    // case  2
    [2, 1, 4, 2, 4, 3, -1],       // case  3
    [1, 2, 5, -1, -1, -1, -1],    // case  4
    [0, 1, 5, 0, 5, 3, -1],       // case  5
    [0, 2, 5, 0, 5, 4, -1],       // case  6
    [3, 4, 5, -1, -1, -1, -1],    // case  7
    [3, 5, 4, -1, -1, -1, -1],    // case  8
    [0, 2, 5, 0, 5, 4, -1],       // case  9
    [0, 1, 5, 0, 5, 3, -1],       // case 10
    [2, 1, 5, -1, -1, -1, -1],    // case 11
    [2, 1, 4, 2, 4, 3, -1],       // case 12
    [0, 1, 4, -1, -1, -1, -1],    // case 13
    [0, 2, 3, -1, -1, -1, -1],    // case 14
    [-1, -1, -1, -1, -1, -1, -1], // case 15
];

/// Decompose a cube into 5 tetrahedra.
///
/// Cube corner numbering (local indices):
/// 000=0, 100=1, 010=2, 110=3, 001=4, 101=5, 011=6, 111=7
const CUBE_TETS: [[usize; 4]; 5] = [
    [0, 1, 2, 4], // 000, 100, 010, 001
    [3, 1, 2, 7], // 110, 100, 010, 111
    [5, 1, 4, 7], // 101, 100, 001, 111
    [6, 2, 4, 7], // 011, 010, 001, 111
    [1, 2, 4, 7], // 100, 010, 001, 111  (central)
];

/// Indices of the 8 corners of a cube cell relative to (i, j, k):
const CUBE_CORNERS: [(usize, usize, usize); 8] = [
    (0, 0, 0), // 0: (i,   j,   k)
    (1, 0, 0), // 1: (i+1, j,   k)
    (0, 1, 0), // 2: (i,   j+1, k)
    (1, 1, 0), // 3: (i+1, j+1, k)
    (0, 0, 1), // 4: (i,   j,   k+1)
    (1, 0, 1), // 5: (i+1, j,   k+1)
    (0, 1, 1), // 6: (i,   j+1, k+1)
    (1, 1, 1), // 7: (i+1, j+1, k+1)
];

/// Find E = E_F isosurface triangles on a 3D energy grid using marching
/// tetrahedra.
///
/// Returns a list of triangles, each as a triple of k‑points.
fn marching_tetrahedra_3d(
    energy: &Array3<f64>, // shape (n3, n2, n1)
    kvec: &Array2<f64>,   // shape (n1*n2*n3, dim)
    n1: usize,
    n2: usize,
    n3: usize,
    e_fermi: f64,
) -> Vec<[Array1<f64>; 3]> {
    let mut triangles = Vec::new();

    for k in 0..n3.saturating_sub(1) {
        for j in 0..n2.saturating_sub(1) {
            for i in 0..n1.saturating_sub(1) {
                // Gather 8 corner values and flat indices
                let mut corner_val = [0.0f64; 8];
                let mut corner_idx = [0usize; 8];
                for (c, &(di, dj, dk)) in CUBE_CORNERS.iter().enumerate() {
                    let ci = i + di;
                    let cj = j + dj;
                    let ck = k + dk;
                    corner_val[c] = energy[[ck, cj, ci]];
                    corner_idx[c] = ci + cj * n1 + ck * (n1 * n2);
                }

                // Process each of the 5 tetrahedra
                for &[v0, v1, v2, v3] in &CUBE_TETS {
                    let ev = [
                        corner_val[v0],
                        corner_val[v1],
                        corner_val[v2],
                        corner_val[v3],
                    ];
                    let case = ((ev[0] >= e_fermi) as usize)
                        | (((ev[1] >= e_fermi) as usize) << 1)
                        | (((ev[2] >= e_fermi) as usize) << 2)
                        | (((ev[3] >= e_fermi) as usize) << 3);

                    let tet_verts = [v0, v1, v2, v3];
                    let edges = &TET_CASES[case];
                    let mut ei = 0;
                    while ei < 6 && edges[ei] != -1 {
                        let e_a = edges[ei] as usize;
                        let e_b = edges[ei + 1] as usize;
                        let e_c = edges[ei + 2] as usize;

                        let tri = [
                            tet_interp(kvec, &corner_idx, &corner_val, &tet_verts, e_a, e_fermi),
                            tet_interp(kvec, &corner_idx, &corner_val, &tet_verts, e_b, e_fermi),
                            tet_interp(kvec, &corner_idx, &corner_val, &tet_verts, e_c, e_fermi),
                        ];
                        triangles.push(tri);
                        ei += 3;
                    }
                }
            }
        }
    }

    triangles
}

/// Interpolate along a tetrahedron edge to find the E = E_F crossing point.
fn tet_interp(
    kvec: &Array2<f64>,
    corner_idx: &[usize; 8],
    corner_val: &[f64; 8],
    tet_verts: &[usize; 4],
    edge: usize,
    e_fermi: f64,
) -> Array1<f64> {
    let (a, b) = TET_EDGE_PAIRS[edge];
    let ca = corner_idx[tet_verts[a]];
    let cb = corner_idx[tet_verts[b]];
    let va = corner_val[tet_verts[a]];
    let vb = corner_val[tet_verts[b]];
    interpolate_edge(kvec.row(ca), kvec.row(cb), va, vb, e_fermi)
}

// ── Gnuplot rendering ────────────────────────────────────────────────

/// Plot 2D Fermi surface segments to a PDF file using gnuplot.
///
/// All segments from all bands are drawn on the same figure.  The first two
/// components of each k‑point are used as (x, y) coordinates.
fn render_fermi_2d(
    all_segments: &[Vec<(Array1<f64>, Array1<f64>)>],
    name: &str,
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    fs::create_dir_all(name)?;

    // Check if there are any segments to plot
    let has_segments = all_segments.iter().any(|s| !s.is_empty());
    if !has_segments {
        return Err(TbError::NoBandsInEnergyRange);
    }

    use gnuplot::{AxesCommon, Color, Figure, Font, LineStyle, Solid};

    let mut fg = Figure::new();
    let axes = fg.axes2d();
    axes.set_x_label(x_label, &[Font("Times New Roman", 18.0)]);
    axes.set_y_label(y_label, &[Font("Times New Roman", 18.0)]);
    axes.set_x_range(gnuplot::AutoOption::Auto, gnuplot::AutoOption::Auto);
    axes.set_y_range(gnuplot::AutoOption::Auto, gnuplot::AutoOption::Auto);

    let colors = [
        "red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta",
    ];

    for (band_idx, segments) in all_segments.iter().enumerate() {
        if segments.is_empty() {
            continue;
        }
        let color = colors[band_idx % colors.len()];
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (p1, p2) in segments {
            xs.push(p1[0]);
            ys.push(p1[1]);
            xs.push(p2[0]);
            ys.push(p2[1]);
            xs.push(f64::NAN);
            ys.push(f64::NAN);
        }
        axes.lines(&xs, &ys, &[Color(color), LineStyle(Solid)]);
    }

    let pdf_name = format!("{}/fermi_surface.pdf", name);
    fg.set_terminal("pdfcairo", &pdf_name);
    fg.show()
        .map_err(|e| TbError::Other(format!("gnuplot error: {}", e)))?;
    Ok(())
}

/// Plot 3D Fermi surface triangles to a PDF file using gnuplot.
fn render_fermi_3d(triangles: &[[Array1<f64>; 3]], name: &str) -> Result<()> {
    fs::create_dir_all(name)?;

    let data_path = format!("{}/fermi_triangles.dat", name);
    let pdf_path = format!("{}/fermi_surface.pdf", name);

    {
        let mut f = fs::File::create(&data_path)?;
        for tri in triangles {
            for v in tri {
                writeln!(f, "{:.8} {:.8} {:.8}", v[0], v[1], v[2])?;
            }
            writeln!(f)?;
        }
    }

    let mut gnuplot = Command::new("gnuplot")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| TbError::Other(format!("Failed to launch gnuplot: {}", e)))?;

    if let Some(stdin) = gnuplot.stdin.as_mut() {
        writeln!(stdin, "set terminal pdfcairo").ok();
        writeln!(stdin, "set output '{}'", pdf_path).ok();
        writeln!(stdin, "set pm3d depthorder").ok();
        writeln!(stdin, "set style fill transparent solid 0.5").ok();
        writeln!(stdin, "set view 60, 30").ok();
        writeln!(stdin, "set xlabel 'k_x' font 'Times New Roman,18'").ok();
        writeln!(stdin, "set ylabel 'k_y' font 'Times New Roman,18'").ok();
        writeln!(stdin, "set zlabel 'k_z' font 'Times New Roman,18'").ok();
        writeln!(stdin, "splot '{}' with pm3d notitle", data_path).ok();
    }

    let status = gnuplot
        .wait()
        .map_err(|e| TbError::Other(format!("gnuplot failed: {}", e)))?;
    if !status.success() {
        return Err(TbError::Other("gnuplot exited with error".into()));
    }
    Ok(())
}

// ── BXSF export (FermiSurfer / XCrySDen) ─────────────────────────────

/// Compute the reciprocal lattice vectors whose columns satisfy
/// BᵀA = 2π·I, where A is the real-space lattice (columns = lattice vectors).
fn rec_lat(lat: &Array2<f64>) -> Result<Array2<f64>> {
    let lat_t = lat.t().to_owned();
    let lat_t_inv = lat_t
        .inv()
        .map_err(|e| TbError::Other(format!("Failed to invert lattice for reciprocal vectors: {e}")))?;
    Ok(PI_64 * 2.0 * lat_t_inv)
}

/// Trait for exporting band energies in BXSF format (XCrySDen / FermiSurfer).
///
/// # Overview
///
/// The [BXSF format](https://web.mit.edu/xcrysden_v1.5.60/www/XCRYSDEN/doc/XSF.html#bxsf)
/// is an ASCII format that stores band energies `E_n(k)` on a uniform
/// k‑mesh.  FermiSurfer and XCrySDen read this file and perform
/// isosurface extraction (marching tetrahedra / ray‑casting) internally,
/// supporting interactive rotation, zoom, slice planes, and high‑quality
/// vector export — all without an external renderer on the Rust side.
///
/// Only implemented for 3D models (`DIM = 3`).  For 2D contour plots and
/// the existing gnuplot‑based 3D triangle‑mesh output, use
/// [`FermiSurface::show_fermi_surface`].
///
/// # Workflow
///
/// ```text
///                  write_bxsf()          $ fermisurfer model.bxsf
/// Model<SPIN,3,R> ────────────► .bxsf ─────────────────────────► interactive 3D view
/// ```
///
/// # Examples
///
/// Write a BXSF file for a BCC model and open it in FermiSurfer:
///
/// ```no_run
/// use ndarray::prelude::*;
/// use num_complex::Complex;
/// use Rustb::*;
///
/// let t = Complex::new(-1.0, 0.0);
/// let lat = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
/// let orb = arr2(&[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]);
/// let mut model = Model::<false, 3>::tb_model(lat, orb, None)?;
/// for &(i, j, k) in &[
///     (0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0),
///     (0, 0, -1), (-1, 0, -1), (0, -1, -1), (-1, -1, -1),
/// ] {
///     model.add_hop(t, 0, 1, &arr1(&[i, j, k]), None);
/// }
///
/// model.write_bxsf(&[50, 50, 50], 0.0, "bcc")?;
/// // Produces `bcc.bxsf` — open with:  $ fermisurfer bcc.bxsf
/// # Ok::<(), Rustb::error::TbError>(())
/// ```
pub trait BxsfExport {
    /// Write band energies on a k‑mesh to a BXSF file.
    ///
    /// # Arguments
    /// * `k_mesh` — Number of k‑points along each reciprocal direction
    ///   `[nx, ny, nz]`.  A mesh of `[50, 50, 50]` gives 125 000 k‑points.
    /// * `e_fermi` — Fermi energy in eV.
    /// * `filename` — Output path; `.bxsf` is appended automatically if not
    ///   already present.
    fn write_bxsf(&self, k_mesh: &[usize; 3], e_fermi: f64, filename: &str) -> Result<()>;
}

impl<const SPIN: bool, R: RMatrixData> BxsfExport for Model<SPIN, 3, R> {
    fn write_bxsf(&self, k_mesh: &[usize; 3], e_fermi: f64, filename: &str) -> Result<()> {
        let [nx, ny, nz] = *k_mesh;
        let nk = nx * ny * nz;

        // 1. Reciprocal lattice (spanning vectors for the grid)
        let b = rec_lat(&self.lat)?;

        // 2. Generate k‑mesh and solve band energies
        let kvec: Array2<f64> = gen_kmesh(&arr1(&[nx, ny, nz]))?;
        let eval = self.solve_band_all_parallel(&kvec);
        let nsta = self.nsta();

        // 3. Build output path (auto-append .bxsf)
        let path = if filename.ends_with(".bxsf") {
            filename.to_owned()
        } else {
            format!("{filename}.bxsf")
        };

        // 4. Write BXSF (plain ASCII)
        let mut f = fs::File::create(&path)?;

        writeln!(f, "BEGIN_INFO")?;
        writeln!(f, "  # BXSF exported by Rustb")?;
        writeln!(f, "  # Number of k-points: {nx}×{ny}×{nz} = {nk}")?;
        writeln!(f, "  # Number of bands: {nsta}")?;
        writeln!(f, "  Fermi Energy: {e_fermi:.12}")?;
        writeln!(f, "END_INFO")?;
        writeln!(f)?;

        writeln!(f, "BEGIN_BLOCK_BANDGRID_3D")?;
        writeln!(f, "  band_energies")?;
        writeln!(f, "  BEGIN_BANDGRID_3D_band_energies")?;
        writeln!(f, "    {nsta}")?;
        writeln!(f, "    {nx} {ny} {nz}")?;
        writeln!(f, "    0.0 0.0 0.0")?; // origin = Γ point
        writeln!(f, "    {:.10} {:.10} {:.10}", b[[0, 0]], b[[1, 0]], b[[2, 0]])?;
        writeln!(f, "    {:.10} {:.10} {:.10}", b[[0, 1]], b[[1, 1]], b[[2, 1]])?;
        writeln!(f, "    {:.10} {:.10} {:.10}", b[[0, 2]], b[[1, 2]], b[[2, 2]])?;

        for ib in 0..nsta {
            writeln!(f, "    BAND: {b0}", b0 = ib + 1)?;
            // Row‑major: ix slowest, iz fastest (XCrySDen convention)
            let mut line_buf = String::with_capacity(256);
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let idx = ix + iy * nx + iz * (nx * ny);
                        let e = eval[[idx, ib]];
                        // Use fixed-format for clean alignment
                        if iz % 10 == 0 && iz > 0 {
                            line_buf.push('\n');
                        }
                        use std::fmt::Write;
                        write!(line_buf, " {e:.10E}").ok();
                    }
                    line_buf.push('\n');
                }
            }
            writeln!(f, "{line_buf}")?;
        }

        writeln!(f, "  END_BANDGRID_3D")?;
        writeln!(f, "END_BLOCK_BANDGRID_3D")?;

        Ok(())
    }
}

// ── FRMSF spin-split export (FermiSurfer) ───────────────────────────

/// Flatten band energies into FRMSF loop order.
///
/// FRMSF nest order (Fortran):
/// `do ibnd; do ik1; do ik2; do ik3; write`
/// i.e. `ibnd` outermost, `ik3` (z) fastest.
fn frmsf_order(eval: &Array2<f64>, nk: &[usize; 3]) -> Vec<f64> {
    let [nx, ny, nz] = *nk;
    let nbnd = eval.ncols();
    let nk_total = nx * ny * nz;
    let mut out = Vec::with_capacity(nk_total * nbnd);
    for ib in 0..nbnd {
        for ik1 in 0..nx {
            for ik2 in 0..ny {
                for ik3 in 0..nz {
                    let idx = ik1 + ik2 * nx + ik3 * (nx * ny);
                    out.push(eval[[idx, ib]]);
                }
            }
        }
    }
    out
}

/// Build a text block in FRMSF format, 10 values per line, returned as a
/// single String to avoid repeated system calls.
fn format_frmsf_block(values: &[f64]) -> String {
    let mut buf = String::with_capacity(values.len() * 15);
    const COLS: usize = 10;
    for (i, &v) in values.iter().enumerate() {
        use std::fmt::Write;
        write!(buf, " {v:.10E}").ok();
        if (i + 1) % COLS == 0 {
            buf.push('\n');
        }
    }
    if values.len() % COLS != 0 {
        buf.push('\n');
    }
    buf
}

/// Write a spin-split Fermi surface in FRMSF format (FermiSurfer native input).
///
/// # Motivation
///
/// In altermagnetic or spin‑split systems without SOC, spin is a good
/// quantum number and one often builds separate spin‑up and spin‑down
/// tight‑binding models.  This function merges both models into a single
/// FRMSF file and writes a **matrix‑element block** that labels each band
/// by its spin character (`+1` for up, `−1` for down).  FermiSurfer can
/// then render the Fermi surface coloured by spin label rather than by
/// energy.
///
/// # FermiSurfer settings
///
/// Open the generated `.frmsf` file, then set:
///
/// ```text
/// Color scale mode = Input (1D)
/// Min of Scale      = -1
/// Max of Scale      = +1
/// ```
///
/// This gives red = spin‑up, blue = spin‑down (or vice‑versa depending on
/// the colour map).
///
/// # Examples
///
/// ```no_run
/// use ndarray::prelude::*;
/// use num_complex::Complex;
/// use Rustb::*;
///
/// let mut up = Model::<false, 3>::tb_model(
///     arr2(&[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]),
///     arr2(&[[0.0,0.0,0.0]]),
///     None,
/// )?;
/// up.add_hop(Complex::new(-1.0, 0.0), 0, 0, &arr1(&[1,0,0]), None);
///
/// let mut dn = Model::<false, 3>::tb_model(
///     arr2(&[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]),
///     arr2(&[[0.0,0.0,0.0]]),
///     None,
/// )?;
/// dn.add_hop(Complex::new(-1.0, 0.0), 0, 0, &arr1(&[1,0,0]), None);
///
/// write_spin_frmsf(&up, &dn, &[50, 50, 50], 0.0, "altermagnet")?;
/// // Open: $ fermisurfer altermagnet.frmsf
/// # Ok::<(), Rustb::error::TbError>(())
/// ```
pub fn write_spin_frmsf<const SPIN: bool, R: RMatrixData>(
    up_model: &Model<SPIN, 3, R>,
    dn_model: &Model<SPIN, 3, R>,
    k_mesh: &[usize; 3],
    e_fermi: f64,
    filename: &str,
) -> Result<()> {
    let [nx, ny, nz] = *k_mesh;
    let nsta = up_model.nsta();

    if dn_model.nsta() != nsta {
        return Err(TbError::Other(format!(
            "Spin models must have same band count: up={nsta}, dn={}",
            dn_model.nsta()
        )));
    }

    let lat_diff = (&up_model.lat - &dn_model.lat).mapv(|x| x.abs()).sum();
    if lat_diff > 1e-10 {
        return Err(TbError::Other(format!(
            "up_model and dn_model must have the same lattice, diff = {lat_diff}"
        )));
    }

    let b = rec_lat(&up_model.lat)?;
    let kvec: Array2<f64> = gen_kmesh(&arr1(&[nx, ny, nz]))?;
    let eval_up = up_model.solve_band_all_parallel(&kvec)-e_fermi;
    let eval_dn = dn_model.solve_band_all_parallel(&kvec)-e_fermi;

    // Merge: up bands first, then down
    let nk = nx * ny * nz;
    let nbnd_total = nsta * 2;
    let mut eval_merged = Array2::<f64>::zeros((nk, nbnd_total));
    for ib in 0..nsta {
        eval_merged.column_mut(ib).assign(&eval_up.column(ib));
        eval_merged
            .column_mut(ib + nsta)
            .assign(&eval_dn.column(ib));
    }

    let path = if filename.ends_with(".frmsf") {
        filename.to_owned()
    } else {
        format!("{filename}.frmsf")
    };
    let mut f = fs::File::create(&path)?;

    // Header
    writeln!(f, "{nx} {ny} {nz}")?;
    writeln!(f, "1")?; // gen_kmesh uses i/N grid (i = 0..N-1), matches ishift=1
    writeln!(f, "{nbnd_total}")?;
    writeln!(f, "{:.10} {:.10} {:.10}", b[[0, 0]], b[[1, 0]], b[[2, 0]])?;
    writeln!(f, "{:.10} {:.10} {:.10}", b[[0, 1]], b[[1, 1]], b[[2, 1]])?;
    writeln!(f, "{:.10} {:.10} {:.10}", b[[0, 2]], b[[1, 2]], b[[2, 2]])?;

    // Energy block
    let energy_flat = frmsf_order(&eval_merged, k_mesh);
    f.write_all(format_frmsf_block(&energy_flat).as_bytes())?;

    // Matrix‑element block: +1 for up, −1 for down
    let n_vals = nk * nbnd_total;
    let color_flat: Vec<f64> = (0..n_vals)
        .map(|i| {
            let ib = i / nk;
            if ib < nsta { 1.0 } else { -1.0 }
        })
        .collect();
    f.write_all(format_frmsf_block(&color_flat).as_bytes())?;

    Ok(())
}

// ── Traits ────────────────────────────────────────────────────────────

/// Trait for computing and visualizing Fermi surfaces.
///
/// * `dim = 2`: uses marching squares on a k‑mesh to extract the E(k) = E_F
///   contour, then renders it as a 2D PDF.
/// * `dim = 3`: uses marching tetrahedra on a 3D k‑mesh to extract the
///   E(k) = E_F isosurface, then renders it as a 3D PDF.
pub trait FermiSurface: solve {
    /// Show the Fermi surface at energy `e_fermi`.
    ///
    /// # Arguments
    /// * `k_mesh` - Number of k‑points along each reciprocal direction
    ///   (e.g. `[100, 100]` for 2D, `[50, 50, 50]` for 3D).
    /// * `e_fermi` - Fermi energy in eV.
    /// * `name` - Output directory name (receives `fermi_surface.pdf`).
    ///
    /// # Examples
    ///
    /// ## 2D: Graphene Fermi surface away from the Dirac point
    ///
    /// When the chemical potential is tuned away from the Dirac point
    /// (E = 0), the Fermi surface evolves from isolated points to small
    /// pockets around K and K':
    ///
    /// ```no_run
    /// use ndarray::prelude::*;
    /// use num_complex::Complex;
    /// use Rustb::*;
    ///
    /// let t = Complex::new(1.0, 0.0);
    /// let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
    /// let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
    /// let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
    /// model.add_hop(t, 0, 1, &arr1(&[0, 0]), None);
    /// model.add_hop(t, 0, 1, &arr1(&[-1, 0]), None);
    /// model.add_hop(t, 0, 1, &arr1(&[0, -1]), None);
    ///
    /// // Fermi surface at E_F = 0.5 (pockets around K, K')
    /// model.show_fermi_surface(&arr1(&[100, 100]), 0.5, "graphene_fs")?;
    /// # Ok::<(), Rustb::error::TbError>(())
    /// ```
    ///
    /// ## 3D: BCC isoenergy surface
    ///
    /// Body-centered cubic with one s‑orbital per atom in the conventional
    /// cubic cell. Nearest‑neighbor hopping connects the corner and body‑center
    /// sites. The isoenergy surface at E = 0 reveals the characteristic
    /// BCC Fermi‑surface topology:
    ///
    /// ```no_run
    /// use ndarray::prelude::*;
    /// use num_complex::Complex;
    /// use Rustb::*;
    ///
    /// let t = Complex::new(-1.0, 0.0);
    /// // Conventional cubic cell, two atoms: corner and body‑center
    /// let lat = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// let orb = arr2(&[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]);
    /// let mut model = Model::<false, 3>::tb_model(lat, orb, None)?;
    ///
    /// // 8 nearest‑neighbour hoppings from corner to body‑center
    /// for &(i, j, k) in &[
    ///     (0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0),
    ///     (0, 0, -1), (-1, 0, -1), (0, -1, -1), (-1, -1, -1),
    /// ] {
    ///     model.add_hop(t, 0, 1, &arr1(&[i, j, k]), None);
    /// }
    ///
    /// // Isoenergy surface at E = 0 (half‑filling)
    /// model.show_fermi_surface(&arr1(&[50, 50, 50]), 0.0, "bcc_fs")?;
    /// # Ok::<(), Rustb::error::TbError>(())
    /// ```
    fn show_fermi_surface(&self, k_mesh: &Array1<usize>, e_fermi: f64, name: &str) -> Result<()>;
}

/// Trait for Fermi surface slices on arbitrary k‑planes (3D models).
pub trait FermiSurfacePlane: solve {
    /// Show the Fermi surface on a user‑specified k‑plane.
    ///
    /// The plane is defined by an origin and two spanning vectors in
    /// fractional reciprocal coordinates:
    ///
    /// ```math
    /// \mathbf{k}(i,j) = \text{origin}
    ///    + \frac{i}{n_1}\mathbf{v}_1
    ///    + \frac{j}{n_2}\mathbf{v}_2
    /// ```
    ///
    /// # Errors
    /// Returns an error if the model dimension is not 3.
    ///
    /// # Examples
    ///
    /// A 2D slice through the BCC isoenergy surface at k_z = 0:
    ///
    /// ```no_run
    /// use ndarray::prelude::*;
    /// use num_complex::Complex;
    /// use Rustb::*;
    ///
    /// // Build the BCC model (same setup as the 3D example above)
    /// # let t = Complex::new(-1.0, 0.0);
    /// # let lat = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// # let orb = arr2(&[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]);
    /// # let mut model = Model::<false, 3>::tb_model(lat, orb, None)?;
    /// # for &(i, j, k) in &[
    /// #     (0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0),
    /// #     (0, 0, -1), (-1, 0, -1), (0, -1, -1), (-1, -1, -1),
    /// # ] {
    /// #     model.add_hop(t, 0, 1, &arr1(&[i, j, k]), None);
    /// # }
    ///
    /// // k_z = 0 plane spanning the k_x–k_y quadrant
    /// let origin = arr1(&[0.0, 0.0, 0.0]);
    /// let v1     = arr1(&[1.0, 0.0, 0.0]);
    /// let v2     = arr1(&[0.0, 1.0, 0.0]);
    ///
    /// model.show_fermi_surface_plane(
    ///     &origin, &v1, &v2, 100, 100, 0.0, "bcc_kz0_slice",
    /// )?;
    /// # Ok::<(), Rustb::error::TbError>(())
    /// ```
    fn show_fermi_surface_plane(
        &self,
        origin: &Array1<f64>,
        vec1: &Array1<f64>,
        vec2: &Array1<f64>,
        n1: usize,
        n2: usize,
        e_fermi: f64,
        name: &str,
    ) -> Result<()>;
}

// ── Trait implementations for Model ───────────────────────────────────

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> FermiSurface for Model<SPIN, DIM, R> {
    fn show_fermi_surface(&self, k_mesh: &Array1<usize>, e_fermi: f64, name: &str) -> Result<()> {
        match self.dim_r() {
            1 => Err(TbError::NotImplemented(
                "Fermi surface not meaningful for 1D systems".into(),
            )),
            2 => {
                let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
                let n1 = k_mesh[0];
                let n2 = k_mesh[1];
                let eval = self.solve_band_all_parallel(&kvec);

                let nsta = self.nsta();
                let mut all_segments: Vec<Vec<(Array1<f64>, Array1<f64>)>> =
                    Vec::with_capacity(nsta);

                for b in 0..nsta {
                    let mut energy = Array2::<f64>::zeros((n2, n1));
                    for j in 0..n2 {
                        for i in 0..n1 {
                            energy[[j, i]] = eval[[i + j * n1, b]];
                        }
                    }
                    let segs = marching_squares_2d(&energy, &kvec, n1, n2, e_fermi);
                    all_segments.push(segs);
                }

                render_fermi_2d(&all_segments, name, "k_x", "k_y")
            }
            3 => {
                let kvec: Array2<f64> = gen_kmesh(k_mesh)?;
                let n1 = k_mesh[0];
                let n2 = k_mesh[1];
                let n3 = k_mesh[2];
                let nk = n1 * n2 * n3;
                let eval = self.solve_band_all_parallel(&kvec);

                let nsta = self.nsta();
                let mut all_triangles: Vec<[Array1<f64>; 3]> = Vec::new();

                for b in 0..nsta {
                    // Check if this band crosses E_F anywhere
                    let mut emin = f64::INFINITY;
                    let mut emax = f64::NEG_INFINITY;
                    for ik in 0..nk {
                        let e = eval[[ik, b]];
                        if e < emin {
                            emin = e;
                        }
                        if e > emax {
                            emax = e;
                        }
                    }
                    if emin > e_fermi || emax < e_fermi {
                        continue; // band never crosses E_F
                    }

                    let mut energy = Array3::<f64>::zeros((n3, n2, n1));
                    for k in 0..n3 {
                        for j in 0..n2 {
                            for i in 0..n1 {
                                energy[[k, j, i]] = eval[[i + j * n1 + k * (n1 * n2), b]];
                            }
                        }
                    }
                    let tris = marching_tetrahedra_3d(&energy, &kvec, n1, n2, n3, e_fermi);
                    all_triangles.extend(tris);
                }

                if all_triangles.is_empty() {
                    return Err(TbError::NoBandsInEnergyRange);
                }
                render_fermi_3d(&all_triangles, name)
            }
            _ => unreachable!(),
        }
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> FermiSurfacePlane for Model<SPIN, DIM, R> {
    fn show_fermi_surface_plane(
        &self,
        origin: &Array1<f64>,
        vec1: &Array1<f64>,
        vec2: &Array1<f64>,
        n1: usize,
        n2: usize,
        e_fermi: f64,
        name: &str,
    ) -> Result<()> {
        if self.dim_r() != 3 {
            return Err(TbError::InvalidDimension {
                dim: self.dim_r(),
                supported: vec![3],
            });
        }

        let kvec: Array2<f64> = gen_kplane(origin, vec1, vec2, n1, n2)?;
        let eval = self.solve_band_all_parallel(&kvec);

        let nsta = self.nsta();
        let mut all_segments: Vec<Vec<(Array1<f64>, Array1<f64>)>> = Vec::with_capacity(nsta);

        for b in 0..nsta {
            let mut energy = Array2::<f64>::zeros((n2, n1));
            for j in 0..n2 {
                for i in 0..n1 {
                    energy[[j, i]] = eval[[i + j * n1, b]];
                }
            }
            let segs = marching_squares_2d(&energy, &kvec, n1, n2, e_fermi);
            all_segments.push(segs);
        }

        let x_label = format!("k · ({:.1},{:.1},{:.1})", vec1[0], vec1[1], vec1[2]);
        let y_label = format!("k · ({:.1},{:.1},{:.1})", vec2[0], vec2[1], vec2[2]);
        render_fermi_2d(&all_segments, name, &x_label, &y_label)
    }
}
