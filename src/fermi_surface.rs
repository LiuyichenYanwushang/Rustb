//! Fermi surface visualization using marching squares (2D) and marching
//! tetrahedra (3D).
//!
//! Provides two traits:
//! - [`FermiSurface`]: 2D contour / 3D isosurface of E(k) = E_F
//! - [`FermiSurfacePlane`]: 2D Fermi surface slice on a specified k‑plane
//!   (3D models only)

use crate::error::{Result, TbError};
use crate::kplane::gen_kplane;
use crate::kpoints::gen_kmesh;
use crate::model::Dimension;
use crate::solve_ham::solve;
use crate::Model;
use ndarray::prelude::*;
use ndarray::*;
use rayon::prelude::*;
use std::fs;
use std::io::Write;
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
            let edge_vals: [(f64, f64); 4] =
                [(e00, e10), (e10, e11), (e01, e11), (e00, e01)];

            let mut ei = 0;
            while ei < 5 && edges[ei] != -1 {
                let e_a = edges[ei] as usize;
                let e_b = edges[ei + 1] as usize;

                let (ca, cb) = edge_pairs[e_a];
                let (va, vb) = edge_vals[e_a];
                let p_a = interpolate_edge(
                    kvec.row(*corners[ca]),
                    kvec.row(*corners[cb]),
                    va, vb, e_fermi,
                );
                let (ca, cb) = edge_pairs[e_b];
                let (va, vb) = edge_vals[e_b];
                let p_b = interpolate_edge(
                    kvec.row(*corners[ca]),
                    kvec.row(*corners[cb]),
                    va, vb, e_fermi,
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
const TET_EDGE_PAIRS: [(usize, usize); 6] = [
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
];

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
                    let ev = [corner_val[v0], corner_val[v1], corner_val[v2], corner_val[v3]];
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
                            tet_interp(
                                kvec,
                                &corner_idx,
                                &corner_val,
                                &tet_verts,
                                e_a,
                                e_fermi,
                            ),
                            tet_interp(
                                kvec,
                                &corner_idx,
                                &corner_val,
                                &tet_verts,
                                e_b,
                                e_fermi,
                            ),
                            tet_interp(
                                kvec,
                                &corner_idx,
                                &corner_val,
                                &tet_verts,
                                e_c,
                                e_fermi,
                            ),
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
        writeln!(
            stdin,
            "set xlabel 'k_x' font 'Times New Roman,18'"
        ).ok();
        writeln!(
            stdin,
            "set ylabel 'k_y' font 'Times New Roman,18'"
        ).ok();
        writeln!(
            stdin,
            "set zlabel 'k_z' font 'Times New Roman,18'"
        ).ok();
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
    fn show_fermi_surface(
        &self,
        k_mesh: &Array1<usize>,
        e_fermi: f64,
        name: &str,
    ) -> Result<()>;
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

impl<const SPIN: bool> FermiSurface for Model<SPIN> {
    fn show_fermi_surface(
        &self,
        k_mesh: &Array1<usize>,
        e_fermi: f64,
        name: &str,
    ) -> Result<()> {
        match self.dim_r {
            Dimension::one => Err(TbError::NotImplemented(
                "Fermi surface not meaningful for 1D systems".into(),
            )),
            Dimension::two => {
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
            Dimension::three => {
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
                                energy[[k, j, i]] =
                                    eval[[i + j * n1 + k * (n1 * n2), b]];
                            }
                        }
                    }
                    let tris =
                        marching_tetrahedra_3d(&energy, &kvec, n1, n2, n3, e_fermi);
                    all_triangles.extend(tris);
                }

                if all_triangles.is_empty() {
                    return Err(TbError::NoBandsInEnergyRange);
                }
                render_fermi_3d(&all_triangles, name)
            }
        }
    }
}

impl<const SPIN: bool> FermiSurfacePlane for Model<SPIN> {
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
        if self.dim_r != Dimension::three {
            return Err(TbError::InvalidDimension {
                dim: self.dim_r as usize,
                supported: vec![3],
            });
        }

        let kvec: Array2<f64> = gen_kplane(origin, vec1, vec2, n1, n2)?;
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

        let x_label = format!("k · ({:.1},{:.1},{:.1})", vec1[0], vec1[1], vec1[2]);
        let y_label = format!("k · ({:.1},{:.1},{:.1})", vec2[0], vec2[1], vec2[2]);
        render_fermi_2d(&all_segments, name, &x_label, &y_label)
    }
}
