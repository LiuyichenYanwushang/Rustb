//! Energy‑cut integration via K‑quadrature on simplex intersections.
//!
//! ## 2D (triangle)
//!
//! The $E=\mu$ line intersects a triangle in a line segment.  The integral
//!
//! ```math
//! \int_T A(k)\,\delta(E(k)-\mu)\,d^2k = \frac{|pq|}{|\nabla E|}\int_0^1 A(t)\,dt
//! ```
//!
//! is evaluated with 2‑point Gauss‑Legendre K‑quadrature along the segment:
//! $E_m$, $K_{nm}^{ab}$, and diagonal velocities are barycentrically
//! interpolated, then $\Omega_n$ (or $G_n$) is recomputed from primitives.
//!
//! ## 3D (tetrahedron)
//!
//! The $E=\mu$ plane intersects a tetrahedron in a convex polygon (triangle
//! or quadrilateral).  The surface integral
//!
//! ```math
//! \int_T A(k)\,\delta(E(k)-\mu)\,d^3k = \frac{\mathrm{area}}{|\nabla E|}\,
//! \frac{1}{\mathrm{area}}\int_{\mathrm{polygon}} A\,dS
//! ```
//!
//! uses polygon triangulation + 3‑point K‑quadrature on each sub‑triangle.
//! `build_tetrahedra_3d_diagavg` provides diagonal‑averaged tetrahedralization
//! for restoring $k\to-k$ cancellation of P‑odd quantities.
//!
//! ## Occupancy‑cut (AHC)
//!
//! Hybrid strategy: fully occupied → vertex‑average $\Omega$; partially
//! occupied → K‑quadrature on clipped polygon; empty → skip.  3D uses a
//! sorted‑$\mu$ sweep with binary search for empty/partial/full $\mu$ ranges.
//!
//! ## Thermal convolution
//!
//! $T>0$ integrals use 1D convolution of the $T=0$ result with the thermal
//! window $w(x) = e^x/(1+e^x)^2$. Vertex preparation is shared; cuts and
//! thermal integration are evaluated separately for each temperature.

use ndarray::prelude::*;

const ENERGY_CUT_EPS: f64 = 1e-12;
const FERMI_X_CUT: f64 = 18.0;
const FERMI_X_STEPS: usize = 72;

/// Convert fixed `[[f64; 3]; NV]` coordinates to `Array2<f64>`.
/// Allocates a small Vec (9–12 elements); called once per simplex.
fn fixed_coords_to_array2<const NV: usize>(coords: &[[f64; 3]; NV]) -> Array2<f64> {
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();
    Array2::from_shape_vec((NV, 3), flat).unwrap()
}

#[inline]
fn fermi_window_x(x: f64) -> f64 {
    if x.abs() > 50.0 {
        0.0
    } else {
        let ex = x.exp();
        ex / ((1.0 + ex) * (1.0 + ex))
    }
}

#[inline]
fn triangle_area(coords: &Array2<f64>) -> f64 {
    let x0 = coords[[0, 0]];
    let y0 = coords[[0, 1]];
    let x1 = coords[[1, 0]];
    let y1 = coords[[1, 1]];
    let x2 = coords[[2, 0]];
    let y2 = coords[[2, 1]];
    0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs()
}

#[inline]
fn tet_vol_from_pts(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> f64 {
    let (x0, y0, z0) = (p0[0], p0[1], p0[2]);
    let (x1, y1, z1) = (p1[0], p1[1], p1[2]);
    let (x2, y2, z2) = (p2[0], p2[1], p2[2]);
    let (x3, y3, z3) = (p3[0], p3[1], p3[2]);
    let det = (x1 - x0) * ((y2 - y0) * (z3 - z0) - (z2 - z0) * (y3 - y0))
        - (y1 - y0) * ((x2 - x0) * (z3 - z0) - (z2 - z0) * (x3 - x0))
        + (z1 - z0) * ((x2 - x0) * (y3 - y0) - (y2 - y0) * (x3 - x0));
    det.abs() / 6.0
}

mod fermi_surface;
mod occupancy;

pub(crate) use fermi_surface::{
    integrate_dipole_energy_cut_2d, integrate_intrinsic_cut_2d, integrate_intrinsic_cut_3d,
};
#[cfg(test)]
pub(crate) use occupancy::read_reset_fermi_cut_counts;
pub(crate) use occupancy::{integrate_fermi_cut_2d, integrate_fermi_cut_3d};

#[cfg(test)]
mod thermal_tests {
    use super::fermi_surface::{
        accumulate_tetrahedron_intrinsic_kquad, accumulate_triangle_dipole_kquad,
        accumulate_triangle_intrinsic_kquad, energy_gradient_3d, kquad_line_cut_dipole,
        kquad_surface_cut_intrinsic, tet_plane_intersection,
    };
    use super::occupancy::thermal_convolution;
    use super::*;
    use crate::response::types::{
        NonlinearKernel, SimplexDiagnostics, TrackedSimplex, VertexKernel,
    };
    use ndarray::{Array1, array};
    use num_complex::Complex;

    fn distinct_vertices() -> [VertexKernel; 4] {
        let bands = [[-0.9, 2.0], [0.3, 2.7], [1.4, 3.5], [2.2, 4.8]];
        std::array::from_fn(|i| {
            let x = (i + 1) as f64;
            let a = Complex::new(0.3 + 0.2 * x, -0.4 + 0.1 * x);
            let b = Complex::new(-0.6 + 0.3 * x, 0.2 + 0.15 * x);
            let c = Complex::new(0.8 - 0.07 * x, -0.3 - 0.2 * x);
            let kernel = |off: Complex<f64>| {
                array![
                    [Complex::new(0.0, 0.0), off],
                    [off.conj(), Complex::new(0.0, 0.0)]
                ]
            };
            VertexKernel {
                band: Array1::from_vec(bands[i].to_vec()),
                k_ab: kernel(a * b.conj()),
                nonlinear: Some(NonlinearKernel {
                    k_bc: kernel(b * c.conj()),
                    k_ac: kernel(a * c.conj()),
                    vdiag: array![0.4 + 0.3 * x, -0.2 - 0.1 * x],
                    vdiag_a: array![-0.5 + 0.13 * x, 0.7 + 0.2 * x],
                    vdiag_b: array![0.6 - 0.11 * x, -0.3 + 0.17 * x],
                }),
            }
        })
    }

    fn assert_cut(actual: &Array1<f64>, expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "actual={actual}, expected={expected}"
            );
        }
    }

    // Frozen outputs from the explicit vertex bindings pin field/vertex order,
    // including the accumulated offset, at zero and finite temperature.
    #[test]
    fn triangle_vertex_projections_preserve_distinct_data() {
        let vertices = distinct_vertices();
        let sim = TrackedSimplex {
            vertices: [&vertices[2], &vertices[0], &vertices[1]],
            coords: [[0.0, 0.0, 0.0], [2.0, 0.25, 0.0], [0.5, 1.5, 0.0]],
            volume: 0.37,
            diag: SimplexDiagnostics::default(),
        };
        let mu = array![-5.0, -0.2, 0.8, 2.2, 3.4, 4.4, 8.0];
        for (beta, expected_dipole, expected_intrinsic) in [
            (
                0.0,
                [
                    0.125,
                    0.13651662533283176,
                    0.16105641999309017,
                    0.12689010237097934,
                    0.13253279096842613,
                    0.125,
                    0.125,
                ],
                [
                    0.125,
                    0.12067616298790167,
                    0.12509569263067044,
                    0.122755609456586,
                    0.12479274292048291,
                    0.125,
                    0.125,
                ],
            ),
            (
                4.0,
                [
                    0.1250000000636322,
                    0.13984387449196914,
                    0.15314942873529344,
                    0.13126537232515864,
                    0.1338042334570239,
                    0.12531582263687308,
                    0.125,
                ],
                [
                    0.12499999994329929,
                    0.12144878001957717,
                    0.12395988928967344,
                    0.1223730184592046,
                    0.1234739457002749,
                    0.12495878248514916,
                    0.125,
                ],
            ),
        ] {
            let mut dipole = Array1::from_elem(mu.len(), 0.125);
            let mut intrinsic = dipole.clone();
            accumulate_triangle_dipole_kquad(&sim, 0.17, &mu, beta, &mut dipole);
            accumulate_triangle_intrinsic_kquad(&sim, &mu, beta, &mut intrinsic);
            assert_cut(&dipole, &expected_dipole);
            assert_cut(&intrinsic, &expected_intrinsic);
        }
    }

    #[test]
    fn tetrahedron_vertex_projections_preserve_distinct_data() {
        let vertices = distinct_vertices();
        let sim = TrackedSimplex {
            vertices: [&vertices[3], &vertices[1], &vertices[0], &vertices[2]],
            coords: [
                [0.0, 0.0, 0.0],
                [2.0, 0.25, 0.0],
                [0.5, 1.5, 0.0],
                [0.25, 0.5, 1.25],
            ],
            volume: 0.19,
            diag: SimplexDiagnostics::default(),
        };
        let mu = array![-5.0, -0.2, 0.8, 2.2, 3.4, 4.4, 8.0];
        for (beta, expected) in [
            (
                0.0,
                [
                    0.125,
                    0.12438649829008028,
                    0.1263835970127515,
                    0.12488124083024488,
                    0.12443433766048481,
                    0.1252168841665595,
                    0.125,
                ],
            ),
            (
                4.0,
                [
                    0.12499999999699546,
                    0.12454954896888346,
                    0.12639715222131417,
                    0.12526197683632456,
                    0.12443242235305534,
                    0.1251678845491299,
                    0.12500000029520406,
                ],
            ),
        ] {
            let mut intrinsic = Array1::from_elem(mu.len(), 0.125);
            accumulate_tetrahedron_intrinsic_kquad(&sim, &mu, beta, &mut intrinsic);
            assert_cut(&intrinsic, &expected);
        }
    }

    #[test]
    fn nonlinear_cuts_preserve_large_finite_gradients_and_poison_overflow() {
        let triangle = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tetrahedron = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        // Synthetic constant amplitude: unit Berry curvature and -Q = 1.
        // The cut energies independently set a large, finite linear gradient.
        let band = [0.0, 1.0];
        let bands = [&band[..]; 4];
        let mut kernel = Array2::zeros((2, 2));
        kernel[[0, 1]] = Complex::new(0.5, -0.5);
        let kmats = [&kernel; 4];
        let mut e_buf = [0.0; 2];
        let mut k_buf = [Complex::new(0.0, 0.0); 6];
        let mut dipole = |energies, mu, velocity| {
            kquad_line_cut_dipole(
                &triangle,
                energies,
                &bands[..3],
                &kmats[..3],
                [velocity; 3],
                mu,
                0.0,
                0,
                2,
                &mut e_buf,
                &mut k_buf,
            )
        };
        let line = dipole([0.0, 1e200, 1e200], 5e199, 1.0);
        assert!((line / 5e-201 - 1.0).abs() < 1e-14);
        assert!(dipole([-f64::MAX, f64::MAX, f64::MAX], 0.0, 1.0).is_nan());
        assert!(dipole([0.0, 1e200, 1e200], 5e199, 1e308).is_nan());
        assert!(dipole([0.0, 1.0, 1.0], f64::INFINITY, 1.0).is_nan());
        assert_eq!(dipole([-f64::MAX; 3], f64::MAX, 1.0), 0.0);
        assert_eq!(dipole([0.0; 3], 0.0, 1.0), 0.0);

        let (_, _, _, norm) = energy_gradient_3d(&tetrahedron, [1e200; 3]);
        assert!(norm.is_finite());
        assert_eq!(
            energy_gradient_3d(&tetrahedron, [0.0; 3]),
            (0.0, 0.0, 0.0, 0.0)
        );
        assert!(energy_gradient_3d(&tetrahedron, [f64::MAX; 3]).3.is_nan());
        let surface = kquad_surface_cut_intrinsic(
            &tetrahedron,
            [0.0, 1e200, 1e200, 1e200],
            &bands,
            &kmats,
            &kmats,
            &kmats,
            [1.0; 4],
            [0.0; 4],
            [0.0; 4],
            5e199,
            0,
            2,
            norm,
            &mut e_buf,
            &mut k_buf,
        );
        assert!((surface / 1.25e-201 - 1.0).abs() < 1e-14);
        let (verts, count) = tet_plane_intersection([-f64::MAX, f64::MAX, f64::MAX, f64::MAX], 0.0);
        assert!(verts[..count].iter().flatten().any(|x| x.is_nan()));
        assert_eq!(tet_plane_intersection([0.0, 1.0, 1.0, 1.0], 1.0).1, 0);
    }

    #[test]
    fn finite_temperature_convolution_is_bounded_and_preserves_moments() {
        let mu = array![-10.0, 0.0, 0.2, 20.0];
        for width in [1e-12, 1e-4, 0.3] {
            let values = thermal_convolution(&mu, width, |points| {
                assert_eq!(points.len(), mu.len() * FERMI_X_STEPS);
                assert!(points.windows(2).into_iter().all(|w| w[0] <= w[1]));
                // Convolution of E^2: mu^2 + pi^2 * width^2 / 3.
                points.mapv(|energy| energy * energy)
            });
            for (&actual, &m) in values.iter().zip(&mu) {
                let expected = m * m + std::f64::consts::PI.powi(2) * width * width / 3.0;
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "width={width}, mu={m}, actual={actual}, expected={expected}"
                );
            }
        }
    }
}
