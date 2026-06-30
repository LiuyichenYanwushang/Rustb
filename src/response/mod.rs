//! Simplex‑quadrature response functions.
//!
//! Instead of linearly interpolating the final scalar integrand
//! (old Blochl method), we interpolate gauge‑invariant primitives
//! `K_nm` and band energies `E_n` inside each simplex, then evaluate
//! the singular denominator at quadrature points.
//!
//! ## Sub‑modules
//!
//! | Module | Quantity | Formula |
//! |--------|----------|---------|
//! | `linear`   | Berry curvature Ω^{ab}, quantum metric g^{ab} | `Σ_n ∫ Ω_n dk` |
//! | `nonlinear`| Berry dipole D^{ab;c}                        | `Σ_n ∫ (−∂f/∂E) v^c Ω_n dk` |
//! | `optical`  | Optical conductivity σ^{ab}(ω)               | `Σ_{nm} ∫ (f_n−f_m) K_nm/(d²−(ω+iη)²) dk` |

pub mod helpers;
pub mod linear;
pub mod nonlinear;
pub mod optical;
pub mod traits;

mod kernel;
mod primitives;
mod quadrature;
mod tracking;
mod types;

// Re‑export public types
pub use kernel::{
    eval_berry_kernel, eval_optical_kernel, fermi, fermi_deriv, quadrature_berry_simplex,
    quadrature_dipole_simplex, quadrature_optical_simplex,
};
pub use optical::OpticalGeometry;
pub use quadrature::{TET_QUAD_PTS_4, TET_QUAD_WTS_4, TRI_QUAD_PTS_3, TRI_QUAD_WTS_3};
pub use tracking::{build_tetrahedra_3d, build_triangles_2d};
pub use traits::BerryCurvature;
pub use types::{SIMPLEX_GAP_TOL, SimplexDiagnostics, TrackedSimplex, VertexKernel};
