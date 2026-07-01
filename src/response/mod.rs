//! # Response functions via simplex quadrature
//!
//! This module provides the full set of linear, nonlinear, and optical
//! response functions for tight‑binding models.  Both **direct k‑mesh
//! summation** and **simplex quadrature** paths are available for most
//! quantities.
//!
//! ## Theoretical background
//!
//! ### Velocity operator
//!
//! The velocity operator in direction $\alpha$ is
//!
//! $$v_\alpha(\mathbf{k}) = \frac{\partial H(\mathbf{k})}{\partial k_\alpha}$$
//!
//! In the band eigenbasis $\{|\psi_{n\mathbf{k}}\rangle\}$, its matrix elements are
//!
//! $$v^\alpha_{nm}(\mathbf{k}) = \langle\psi_{n\mathbf{k}}|
//!     \partial_\alpha H(\mathbf{k}) |\psi_{m\mathbf{k}}\rangle$$
//!
//! Diagonal elements $v^\alpha_{nn} = \partial E_n/\partial k_\alpha$ give the
//! band velocity; off‑diagonal elements enter response formulas.
//!
//! ### Gauge‑invariant velocity kernel
//!
//! Individual matrix elements $v^\alpha_{nm}$ are not smooth across the
//! Brillouin zone because eigenstates carry arbitrary U(1) phases.  The
//! product
//!
//! $$K^{ab}_{nm}(\mathbf{k}) \equiv v^a_{nm}(\mathbf{k})v^b_{mn}(\mathbf{k})$$
//!
//! is invariant under independent phase rotations of bands $n$ and $m$
//! (when the bands are isolated), making it safe to interpolate inside
//! each simplex.
//!
//! ### Simplex quadrature (vs. old Blochl method)
//!
//! The **simplex path** interpolates the gauge‑invariant primitives
//! $K_{nm}(\mathbf{k})$ and $E_n(\mathbf{k})$ linearly inside each
//! simplex (triangle in 2D, tetrahedron in 3D), then evaluates the
//! singular denominator at symmetric quadrature points:
//!
//! $$\int_{\text{simplex}} f(K(\mathbf{k}), E(\mathbf{k}))d\mathbf{k}
//!   \approx V_{\text{simplex}} \sum_q w_qf(K(\mathbf{k}_q), E(\mathbf{k}_q))$$
//!
//! This correctly preserves the $1/(E_n-E_m)^p$ singularity structure
//! near small gaps, unlike the old Blochl method which linearly
//! interpolated the final scalar integrand at simplex vertices.
//!
//! ### Band tracking
//!
//! Interpolating $K_{nm}$ requires consistent band labels across simplex
//! vertices.  Energy ordering fails near crossings.  We use eigenvector
//! overlap maximisation: for each vertex $r$, find the permutation $P_r$
//! maximising $\sum_n |\langle u_n(\text{ref})|u_{P_r(n)}(r)\rangle|^2$.
//!
//! ## Sub‑modules
//!
//! | Module | Quantity | Formula |
//! |--------|----------|---------|
//! | [`linear`]   | Berry curvature $\Omega^{ab}$, quantum metric $g^{ab}$ | $\sum_n \int \Omega_n^{ab}d\mathbf{k}$ |
//! | [`nonlinear`]| Berry dipole $D^{ab;c}$, intrinsic/extrinsic NLH | $\sum_n \int (-\partial f/\partial E_n) v^c_n \Omega_n^{ab}d\mathbf{k}$ |
//! | [`optical`]  | Optical conductivity $\sigma^{ab}(\omega)$ | $\sum_{n=\not m} \int \frac{(f_n-f_m)K^{ab}_{nm}}{(E_n-E_m)^2-(\omega+i\eta)^2}d\mathbf{k}$ |
//! | energy cut | 2D Berry dipole line cuts | $\int A_n(k)\delta(E_n-\mu)d^2k$ with finite-T convolution |
//! | [`traits`]   | `BerryCurvature` trait (per‑k‑point Berry curvature) | |
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use rustb::response::{self, VertexKernel};
//!
//! // Direct k‑mesh sum (reference path)
//! let sigma = model.Hall_conductivity(&kmesh, &dx, &dy, &mu, T, None, eta)?;
//!
//! // Simplex quadrature (higher accuracy near small gaps)
//! let (metric, berry, _) = model.berry_curvature_simplex(&kmesh, &dx, &dy, eta)?;
//! let (dipole, _) = model.berry_curvature_dipole_simplex(&kmesh, &dx, &dy, &dx, &mu, T, eta)?;
//! let (dipole_cut, _) = model.berry_curvature_dipole_energy_cut(&kmesh, &dx, &dy, &dx, &mu, T, eta)?;
//! let sigma_opt = model.optical_conductivity_simplex(&kmesh, &dx, &dy, omega, eta, mu, T)?;
//! ```

pub mod helpers;
pub mod linear;
pub mod nonlinear;
pub mod optical;
pub mod traits;

mod energy_cut;
mod kernel;
mod primitives;
mod quadrature;
mod tracking;
mod types;

// Re‑export public types
pub use energy_cut::{
    FermiCutCounts, integrate_dipole_energy_cut_2d, integrate_fermi_cut_2d, integrate_fermi_cut_3d,
    integrate_intrinsic_cut_2d, integrate_intrinsic_cut_3d, read_reset_fermi_cut_counts,
    triangle_line_cut,
};
pub use kernel::{
    eval_berry_band_at_lam, eval_berry_complex_at_lam, eval_berry_kernel, eval_intrinsic_G_at_lam,
    eval_optical_kernel, fermi, fermi_deriv, quadrature_berry_simplex, quadrature_optical_simplex,
};
pub use optical::OpticalGeometry;
pub use quadrature::{TET_QUAD_PTS_4, TET_QUAD_WTS_4, TRI_QUAD_PTS_3, TRI_QUAD_WTS_3};
pub use tracking::{
    build_tetrahedra_3d, build_tetrahedra_3d_diagavg, build_triangles_2d,
    build_triangles_2d_diagavg, global_band_track,
};
pub use traits::BerryCurvature;
pub use types::{SIMPLEX_GAP_TOL, SimplexDiagnostics, TrackedSimplex, VertexKernel};
