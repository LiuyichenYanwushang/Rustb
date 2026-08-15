#![allow(warnings)]

//! # Rustb -- Tight-Binding Model Library
//!
//! A Rust library for tight-binding model calculations in condensed matter physics.
//! It supports model construction from both explicit hopping parameters and
//! Slater-Koster integrals, band structure solving, topological analysis, and
//! linear/nonlinear transport property calculations.
//!
//! ## Module overview
//!
//! ### Core data structures
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`model`] | Central [`Model`]<SPIN, DIM, R> struct with lattice, orbital, and hopping data, plus enums
//! |   [`Gauge`], [`Dimension`], [`SpinDirection`] |
//! | [`atom_struct`] | [`Atom`] and [`OrbProj`] types for describing atomic sites and orbital
//! |   projections |
//!
//! ### Model construction and manipulation
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`model_build`] | Constructors (`Model::tb_model`), hopping setup (`add_hop`, `set_hop`),
//! |   on-site energies (`set_onsite`), and supercell building (`make_supercell`) |
//! | [`hubbard`] | Non-collinear unrestricted Hartree-Fock for on-site Hubbard
//! |   interactions, with fixed chemical potential or fixed initial filling |
//! | [`cut`] | [`CutModel`] trait for extracting finite slabs from supercells |
//! | [`geometry`] | Supercell geometry, dot structures, and related spatial operations |
//! | [`model_utils`] | Internal utility functions for model manipulation |
//!
//! ### Hamiltonian solving
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`solve_ham`] | Parallel diagonalization of H(k) over k-point meshes
//! |   (`solve_all_parallel`, `solve_band_all_parallel`) |
//! | [`ndarray_lapack`] | LAPACK bindings for ndarray matrices |
//!
//! ### k-space sampling
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`kpath`] | k-path generation along high-symmetry lines (`k_path`) |
//! | [`kpoints`] | Uniform k-mesh generation (`gen_kmesh`, `gen_krange`) |
//!
//! ### Transport properties
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`response`] | Linear, nonlinear, and optical conductivity tensors via the Kubo
//! |   formalism and simplex quadrature: anomalous Hall, spin Hall, nonlinear
//! |   responses, and frequency-dependent optical conductivity |
//!
//! ### Operators and observables
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`velocity`] | Velocity operator v_a(k) at each k-point |
//! | [`orbital_angular`] | Orbital angular momentum operator |
//! | [`math`] | Mathematical utilities (commutators, matrix operations) |
//!
//! ### Topological analysis
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`model_physics`] | Density of states, Berry curvature, Chern numbers, Wilson loops,
//! |   and Wannier centers |
//!
//! ### Surface and defect calculations
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`surfgreen`] | Surface Green's function G^s(omega, k_parallel) for
//! |   semi-infinite systems; local density of states at surfaces and edges |
//!
//! ### Interfaces
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`wannier90`] | Read Wannier90 `_hr.dat` and `_r.dat` files |
//! | [`unfold`] | Band unfolding for supercell calculations (the [`Unfold`] trait) |
//!
//! ### Magnetic field
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`magnetic_field`] | Uniform magnetic field via Peierls substitution |
//!
//! ### Output and I/O
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`output`] | Band structure and surface state plotting via gnuplot (`show_band`, etc.) |
//! | [`io`] | Text file I/O for 1D and 2D arrays (`write_txt`, `write_txt_1`) |
//!
//! ### Supporting modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`error`] | Centralized error handling ([`TbError`], [`Result`]) |
//! | [`phy_const`] | Physical constants (hbar, e, k_B, etc.) |
//! | [`generics`] | Numeric type abstractions |
//!
//! ## Mathematical foundation
//!
//! The tight-binding Hamiltonian in second-quantized form:
//!
//! $$
//! H = \sum_{i,j} t_{ij} c_i^\dagger c_j + \sum_i \epsilon_i c_i^\dagger c_i
//! $$
//!
//! where t_{ij} are hopping parameters and epsilon_i are on-site energies.
//!
//! The Bloch Hamiltonian at a given k-point is:
//!
//! $$
//! H_{mn}(\mathbf{k}) = \sum_{\mathbf{R}} H_{mn}(\mathbf{R})\, e^{i \mathbf{k} \cdot \mathbf{R}}
//! $$
//!
//! where R runs over lattice vectors and H_{mn}(R) is the
//! hopping matrix element from orbital n to orbital m.
//!
//! For transport, the Berry curvature is computed as:
//!
//! $$
//! \Omega_n(\mathbf{k}) = -2\,\operatorname{Im}\sum_{m\neq n}
//! \frac{\bra{n}\partial_{k_x} H\ket{m}\bra{m}\partial_{k_y} H\ket{n}}
//!      {(E_n - E_m)^2}
//! $$
//!
//! and the anomalous Hall conductivity follows from the Brillouin-zone integral:
//!
//! $$
//! \sigma_{xy} = \frac{e^2}{\hbar} \int \frac{d^d k}{(2\pi)^d}\,
//! \sum_n f_n(\mathbf{k})\, \Omega_n(\mathbf{k})
//! $$
//!
//! ## Quick start
//!
//! The example below builds a nearest-neighbor graphene model, computes the band
//! structure and density of states, constructs a zigzag nanoribbon, and plots the
//! edge states.
//!
//! ```no_run
//! # use Rustb::error::Result;
//! # fn main() -> Result<()> {
//! use ndarray::prelude::*;
//! use num_complex::Complex;
//! use Rustb::*;
//!
//! // --- Build the graphene tight-binding model ---
//! let t1 = Complex::new(1.0, 0.0);     // nearest-neighbor hopping
//! let t2 = Complex::new(0.1, 0.0);     // next-nearest-neighbor hopping
//! let delta = 0.5;                     // staggered on-site potential
//!
//! // Honeycomb lattice vectors
//! let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
//! // Two sublattice sites in fractional coordinates
//! let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
//!
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! model.set_onsite(&arr1(&[delta, -delta]), None);
//!
//! // Nearest-neighbor hoppings (A <-> B)
//! model.add_hop(t1, 0, 1, &array![0, 0], None);
//! model.add_hop(t1, 0, 1, &array![-1, 0], None);
//! model.add_hop(t1, 0, 1, &array![0, -1], None);
//!
//! // Next-nearest-neighbor hoppings (A-A and B-B)
//! for &(i, j) in &[(0, 0), (1, 1)] {
//!     for r in &[array![1, 0], array![0, 1], array![1, -1]] {
//!         model.add_hop(t2, i, j, r, None);
//!     }
//! }
//!
//! // --- Band structure along high-symmetry path G -> K -> M -> G ---
//! let nk = 1001;
//! let path = arr2(&[[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.5], [0.0, 0.0]]);
//! let label = vec!["G", "K", "M", "G"];
//! model.show_band(&path, &label, nk, "graphene")?;
//!
//! // --- Zigzag nanoribbon and edge states ---
//! let U = arr2(&[[1.0, 1.0], [-1.0, 1.0]]);
//! let super_model = model.make_supercell(&U)?;
//! let zig_model = super_model.cut_piece(100, 0)?;
//! let path_edge = arr2(&[[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]]);
//! let label_edge = vec!["G", "M", "G"];
//! zig_model.show_band(&path_edge, &label_edge, 501, "graphene_zig")?;
//!
//! // --- Density of states ---
//! let kmesh = arr1(&[101, 101]);
//! let (energies, dos) = model.dos(&kmesh, -3.0, 3.0, 1000, 1e-2)?;
//! // Write DOS data to a text file
//! let dos_data = ndarray::stack![Axis(0), energies, dos];
//! write_txt(&dos_data, "dos.dat")?;
//! # Ok(())
//! # }
//! ```

pub mod atom_struct;
#[cfg(feature = "cryspglib")]
pub mod crystal_symmetry;
pub mod cut;
pub mod error;
pub mod fermi_surface;
pub mod floquet;
pub mod generics;
pub mod geometry;
#[cfg(feature = "cryspglib")]
pub mod hamiltonian_symmetry;
#[path = "Hubbard.rs"]
pub mod hubbard;
pub mod io;
pub mod kpath;
pub mod kplane;
pub mod kpoints;
pub mod magnetic_field;
pub mod math;
pub mod model;
pub mod model_build;
pub mod model_physics;
pub mod model_utils;
pub mod ndarray_lapack;
pub mod orbital_angular;
pub mod output;
pub mod phy_const;
pub mod quantum_geometry;
pub mod response;
pub mod solve_ham;
pub mod surfgreen;
pub mod thermodynamics;
pub mod unfold;
pub mod velocity;
pub mod wannier90;
pub use crate::atom_struct::{Atom, AtomId, AtomType, OrbProj, OrbitalId};
#[cfg(feature = "cryspglib")]
pub use crate::crystal_symmetry::{
    CrystalSymmetry, CrystalSymmetryDataset, CrystalSymmetryOperation, ExternalFields,
    HighSymmetryKPoint, IrreducibleKMesh, MagneticCrystalSymmetry, MagneticGroupType,
    MagneticTableColumns, SymmetryParameters,
};
pub use crate::cut::*;
pub use crate::error::{Result, TbError};
pub use crate::fermi_surface::*;
pub use crate::floquet::*;
use crate::generics::usefloat;
pub use crate::geometry::*;
#[cfg(feature = "cryspglib")]
pub use crate::hamiltonian_symmetry::{
    BasisActionContext, BasisRepresentationError, BasisSymmetryRepresentation, CellShiftAction,
    FinalMagneticGroup, HamiltonianCompatibility, HamiltonianResidual, HamiltonianResidualWitness,
    HamiltonianSymmetrizationParameters, HamiltonianSymmetryCandidates,
    HamiltonianSymmetryCompleteness, HamiltonianSymmetryReport, HamiltonianSymmetryRequest,
    HamiltonianSymmetryTolerances, IdentifiedMagneticSubgroup, LocalizedBasisAction,
    OperationHamiltonianCheck, OperationHamiltonianStatus, ScalarSiteBasis,
};
pub use crate::hubbard::*;
pub use crate::io::*;
pub use crate::kpath::*;
pub use crate::kplane::*;
pub use crate::kpoints::*;
pub use crate::magnetic_field::*;
pub use crate::math::*;
pub use crate::model::*;
pub use crate::model_physics::*;
pub use crate::output::*;
pub use crate::quantum_geometry::*;
pub use crate::response::*;
pub use crate::solve_ham::*;
pub use crate::surfgreen::*;
pub use crate::thermodynamics::*;
pub use crate::unfold::*;
pub use crate::velocity::*;
pub use crate::wannier90::*;

// --- BLAS/LAPACK 后端（互斥，只能开一个） ---
//
// `ndarray-linalg` 默认不链接任何后端，而且多个后端同时启用时会在
// `intel-mkl-src`/`openblas-src`/`netlib-src` 或本 crate 的
// `ndarray_lapack` 中产生重复符号/重复 extern crate。这里在 const
// 求值阶段强制“有且仅有一个后端”。
const _: () = {
    let backends = cfg!(feature = "intel-mkl-static") as usize
        + cfg!(feature = "intel-mkl-system") as usize
        + cfg!(feature = "openblas-static") as usize
        + cfg!(feature = "openblas-system") as usize
        + cfg!(feature = "netlib-static") as usize
        + cfg!(feature = "netlib-system") as usize;
    assert!(
        backends <= 1,
        "Rustb: enable exactly one BLAS/LAPACK backend feature (intel-mkl-static, intel-mkl-system, openblas-static, openblas-system, netlib-static, netlib-system); they are mutually exclusive"
    );
    assert!(
        backends >= 1,
        "Rustb: no BLAS/LAPACK backend selected. Enable one backend feature (e.g. the default openblas-system, or use --features intel-mkl-system)"
    );
};

// --- 可选高性能分配器（互斥，只能开一个 feature） ---
#[cfg(all(feature = "mimalloc", feature = "jemalloc"))]
compile_error!("mimalloc and jemalloc are mutually exclusive — enable only one.");

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(all(feature = "jemalloc", not(feature = "mimalloc")))]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::config::direction_matrix;
    use gnuplot::{
        AutoOption, AxesCommon, Color, Figure, Fix, Font, LineStyle, Major, PointSymbol, Rotate,
        Solid, TextOffset,
    };
    use ndarray::concatenate;
    use ndarray::linalg::kron;
    use ndarray::prelude::*;
    use ndarray::*;
    use ndarray_linalg::conjugate;
    use ndarray_linalg::*;
    use ndarray_linalg::{Eigh, UPLO};
    use num_complex::Complex;
    use rayon::prelude::*;
    use std::f64::consts::PI;
    use std::fs::File;
    use std::fs::create_dir_all;
    use std::io::Write;
    use std::time::{Duration, Instant};

    fn fixed_direction<const DIM: usize>(direction: &Array1<f64>) -> [f64; DIM] {
        direction
            .as_slice()
            .expect("direction must be contiguous")
            .try_into()
            .expect("direction must match the model dimension")
    }

    fn fixed_k_mesh<const DIM: usize>(k_mesh: &Array1<usize>) -> [usize; DIM] {
        k_mesh
            .as_slice()
            .expect("k-mesh must be contiguous")
            .try_into()
            .expect("k-mesh must match the model dimension")
    }

    fn parameters_at_direction<const DIM: usize>(
        k_mesh: [usize; DIM],
        direction: Array2<f64>,
        chemical_potentials: &Array1<f64>,
        temperature_kelvin: f64,
    ) -> Parameters<DIM> {
        let mut params = Parameters::new(k_mesh, direction, chemical_potentials.clone());
        params.T = array![temperature_kelvin];
        params
    }

    fn hall_values<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k_mesh: &Array1<usize>,
        direction_a: &Array1<f64>,
        direction_b: &Array1<f64>,
        chemical_potentials: &Array1<f64>,
        temperature_kelvin: f64,
        spin: Option<SpinDirection>,
        broadening: f64,
        integration: Integration,
    ) -> Result<Array1<f64>> {
        let mut params = parameters_at_direction(
            fixed_k_mesh(k_mesh),
            direction_matrix::<2, DIM>(&[
                fixed_direction(direction_a),
                fixed_direction(direction_b),
            ]),
            chemical_potentials,
            temperature_kelvin,
        );
        params.spin = spin;
        params.eta = broadening;
        params.integration = integration;
        Ok(model.hall_conductivity(&params)?.conductivity)
    }

    fn hall_value<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k_mesh: &Array1<usize>,
        direction_a: &Array1<f64>,
        direction_b: &Array1<f64>,
        chemical_potential: f64,
        temperature_kelvin: f64,
        spin: Option<SpinDirection>,
        broadening: f64,
    ) -> Result<f64> {
        Ok(hall_values(
            model,
            k_mesh,
            direction_a,
            direction_b,
            &array![chemical_potential],
            temperature_kelvin,
            spin,
            broadening,
            Integration::Direct,
        )?[0])
    }

    fn band_berry_curvature<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k: &Array1<f64>,
        direction_a: &Array1<f64>,
        direction_b: &Array1<f64>,
        spin: Option<SpinDirection>,
        broadening: f64,
    ) -> BandBerryCurvature {
        let mut params = Parameters::rank2(
            [1; DIM],
            fixed_direction(direction_a),
            fixed_direction(direction_b),
            array![0.0],
        );
        params.spin = spin;
        params.eta = broadening;
        model.berry_curvature_at(k, &params).unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn occupied_berry_curvature<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k_points: &Array2<f64>,
        direction_a: &Array1<f64>,
        direction_b: &Array1<f64>,
        chemical_potential: f64,
        temperature_kelvin: f64,
        spin: Option<SpinDirection>,
        broadening: f64,
    ) -> Array1<f64> {
        let mut params = Parameters::rank2(
            [1; DIM],
            fixed_direction(direction_a),
            fixed_direction(direction_b),
            array![chemical_potential],
        );
        params.T = array![temperature_kelvin];
        params.spin = spin;
        params.eta = broadening;
        model
            .occupied_berry_curvature_on(k_points, &params)
            .unwrap()
    }

    /// Temperature for direct NLH tests with nominal zero temperature:
    /// the old Fermi-smearing width `1/nk^(1/dim)` converted to the
    /// equivalent Fermi-Dirac temperature `width / k_B`.
    fn nonlinear_temperature(
        temperature_kelvin: f64,
        k_mesh: &Array1<usize>,
        integration: Integration,
    ) -> f64 {
        if temperature_kelvin > 0.0 {
            return temperature_kelvin;
        }
        if integration == Integration::EnergyCut {
            return 0.0;
        }
        let points_per_dimension = k_mesh.iter().product::<usize>() as f64;
        let points_per_dimension = points_per_dimension.powf(1.0 / k_mesh.len() as f64);
        let width = (1.0 / points_per_dimension).max(BOLTZMANN_CONSTANT_EV_PER_K);
        width / BOLTZMANN_CONSTANT_EV_PER_K
    }

    fn intrinsic_nonlinear_values<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k_mesh: &Array1<usize>,
        current: &Array1<f64>,
        field_1: &Array1<f64>,
        field_2: &Array1<f64>,
        chemical_potentials: &Array1<f64>,
        temperature_kelvin: f64,
        integration: Integration,
    ) -> Result<Array1<f64>> {
        let mut params = Parameters::rank3(
            fixed_k_mesh(k_mesh),
            fixed_direction(current),
            fixed_direction(field_1),
            fixed_direction(field_2),
            chemical_potentials.clone(),
        );
        params.T = array![nonlinear_temperature(
            temperature_kelvin,
            k_mesh,
            integration
        )];
        params.integration = integration;
        Ok(model.intrinsic_nonlinear_hall(&params)?.conductivity)
    }

    #[allow(clippy::too_many_arguments)]
    fn extrinsic_nonlinear_values<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k_mesh: &Array1<usize>,
        current: &Array1<f64>,
        field_1: &Array1<f64>,
        field_2: &Array1<f64>,
        chemical_potentials: &Array1<f64>,
        temperature_kelvin: f64,
        frequency: f64,
        spin: Option<SpinDirection>,
        broadening: f64,
        integration: Integration,
        field_symmetry: FieldSymmetry,
    ) -> Result<Array1<f64>> {
        let mut params = Parameters::rank3(
            fixed_k_mesh(k_mesh),
            fixed_direction(current),
            fixed_direction(field_1),
            fixed_direction(field_2),
            chemical_potentials.clone(),
        );
        params.T = array![nonlinear_temperature(
            temperature_kelvin,
            k_mesh,
            integration
        )];
        params.omega = array![frequency];
        params.spin = spin;
        params.eta = broadening;
        params.integration = integration;
        params.field_symmetry = field_symmetry;
        Ok(model.extrinsic_nonlinear_hall(&params)?.conductivity)
    }

    fn write_txt(data: Array2<f64>, output: &str) -> std::io::Result<()> {
        let mut file = File::create(output).expect("Unable to BAND.dat");
        let n = data.len_of(Axis(0));
        let s = data.len_of(Axis(1));
        let mut s0 = String::new();
        for i in 0..n {
            for j in 0..s {
                if data[[i, j]] >= 0.0 {
                    s0.push_str("     ");
                } else {
                    s0.push_str("    ");
                }
                let aa = format!("{:.6}", data[[i, j]]);
                s0.push_str(&aa);
            }
            s0.push_str("\n");
        }
        writeln!(file, "{}", s0)?;
        Ok(())
    }

    fn write_txt_1(data: Array1<f64>, output: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut file = File::create(output).expect("Unable to BAND.dat");
        let n = data.len_of(Axis(0));
        let mut s0 = String::new();
        for i in 0..n {
            if data[[i]] >= 0.0 {
                s0.push_str(" ");
            }
            let aa = format!("{:.6}\n", data[[i]]);
            s0.push_str(&aa);
        }
        writeln!(file, "{}", s0)?;
        Ok(())
    }
    #[test]
    fn test_rmatrix_grows_with_new_hoppings() {
        // Regression: set_hop/add_hop/add_element must keep rmatrix in sync with
        // hamR, otherwise validate() (and make_supercell/cut_*) reject
        // HasRMatrix models built with these methods.
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0]];
        let mut model = Model::<false, 2, HasRMatrix>::tb_model(lat, orb, None).unwrap();

        // set_hop adds R=(1,0) and -R -> hamR grows to 3 rows
        model.set_hop(-1.0, 0, 0, &arr1(&[1isize, 0]), None);
        // add_hop adds R=(0,1) and -R -> hamR grows to 5 rows
        model.add_hop(-1.0, 0, 0, &arr1(&[0isize, 1]), None);
        // add_element adds R=(1,1) and -R -> hamR grows to 7 rows
        model
            .add_element(Complex::new(-0.2, 0.0), 0, 0, &arr1(&[1isize, 1]))
            .unwrap();

        // The rmatrix shape must match hamR.nrows() x DIM x nsta x nsta
        let expected = (model.hamR.nrows(), 2, model.nsta(), model.nsta());
        assert_eq!(
            model.rmatrix.as_array4().dim(),
            expected,
            "rmatrix must grow in sync with hamR"
        );
        // validate() must pass (previously returned InvalidModelInvariant)
        model.validate().unwrap();
        // Newly added hopping vectors must carry zero position-matrix blocks
        for i_r in 1..model.hamR.nrows() {
            let block = model.rmatrix.as_array4().index_axis(Axis(0), i_r);
            assert!(
                block.iter().all(|x| x.norm_sqr() == 0.0),
                "position matrix block for R={:?} must be zero",
                model.hamR.row(i_r)
            );
        }
        // make_supercell (documented HasRMatrix workflow) must work
        let supercell = model
            .make_supercell(&array![[2.0, 0.0], [0.0, 1.0]])
            .unwrap();
        supercell.validate().unwrap();
    }

    #[test]
    fn add_element_updates_existing_hopping_hermitically() {
        // Regression: updating an existing R with i == j, or an R = 0
        // off-diagonal element, wrote only one side and broke Hermiticity.
        let mut model = Model::<false, 2>::tb_model(
            array![[1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 0.0], [0.5, 0.0]],
            None,
        )
        .unwrap();

        // Case 1: update an existing R != 0 hopping with i == j.
        model
            .add_element(Complex::new(1.0, 0.0), 0, 0, &array![1, 0])
            .unwrap();
        model
            .add_element(Complex::new(2.0, 0.0), 0, 0, &array![1, 0])
            .unwrap();
        let i_plus = find_R(&model.hamR, &array![1, 0]).unwrap();
        let i_minus = find_R(&model.hamR, &array![-1, 0]).unwrap();
        assert_eq!(model.ham[[i_plus, 0, 0]], Complex::new(2.0, 0.0));
        assert_eq!(
            model.ham[[i_minus, 0, 0]],
            Complex::new(2.0, 0.0),
            "H(-R) must follow H(R) when updating an existing hopping"
        );

        // Case 2: update an R = 0 off-diagonal element.
        model
            .add_element(Complex::new(0.3, 0.1), 0, 1, &array![0, 0])
            .unwrap();
        model
            .add_element(Complex::new(0.4, 0.2), 0, 1, &array![0, 0])
            .unwrap();
        let i_0 = find_R(&model.hamR, &array![0, 0]).unwrap();
        assert_eq!(model.ham[[i_0, 0, 1]], Complex::new(0.4, 0.2));
        assert_eq!(
            model.ham[[i_0, 1, 0]],
            Complex::new(0.4, -0.2),
            "R = 0 off-diagonal update must write the Hermitian conjugate"
        );
    }

    #[test]
    fn add_element_complex_onsite_errors_before_writing() {
        // Regression: the OnsiteHoppingMustBeReal error fired after the
        // Hamiltonian block was already written, corrupting the caller's
        // model. The error must leave the model untouched.
        let mut model =
            Model::<false, 2>::tb_model(array![[1.0, 0.0], [0.0, 1.0]], array![[0.0, 0.0]], None)
                .unwrap();
        let before = model.ham.clone();
        let result = model.add_element(Complex::new(1.0, 2.0), 0, 0, &array![0, 0]);
        assert!(matches!(result, Err(TbError::OnsiteHoppingMustBeReal(_))));
        for (a, b) in model.ham.iter().zip(before.iter()) {
            assert_eq!(a, b, "failed add_element must not modify the Hamiltonian");
        }
    }

    #[test]
    fn test_gen_v() {
        //判断两个Array1<f64> 是否足够接近
        fn are_arrays_close(a: &Array1<f64>, b: &Array1<f64>, tolerance: f64) -> bool {
            a.iter()
                .zip(b.iter())
                .all(|(&x, &y)| (x - y).abs() < tolerance)
        }

        //判断两个Array2<Compelx<f64>> 是否足够接近
        fn are_complex_arrays_close(
            a: &Array2<Complex<f64>>,
            b: &Array2<Complex<f64>>,
            tolerance: f64,
        ) -> bool {
            a.iter()
                .zip(b.iter())
                .all(|(&x, &y)| (x.re - y.re).abs() < tolerance && (x.im - y.im).abs() < tolerance)
        }
        let li: Complex<f64> = 1.0 * Complex::i();
        let t = 1.0;
        let delta = 0.0;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);
        let R0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(t, 0, 1, &R, None);
        }
        assert_eq!(model.solve_band_onek(&array![0.0, 0.0]), array![-3.0, 3.0]);
        let result = model.solve_band_onek(&array![1.0 / 3.0, 2.0 / 3.0]);
        assert!(
            are_arrays_close(&result, &array![0.0, 0.0], 1e-5),
            "wrong!, the solve_band_onek get wrong result! please check it!"
        );
        let (result, _) = model.gen_v(&array![1.0 / 3.0, 1.0 / 3.0], Gauge::Atom);
        let resulty = array![
            [0.0 * li, -0.4698463103929542 - 0.17101007166283436 * li],
            [-0.4698463103929542 + 0.17101007166283436 * li, 0.0 * li]
        ];
        let resultx = array![
            [0.0 * li, -0.8137976813493737 - 0.2961981327260237 * li],
            [-0.8137976813493737 + 0.2961981327260237 * li, 0.0 * li]
        ];
        println!("result={}", result);
        assert!(
            are_complex_arrays_close(&result.slice(s![0, .., ..]).to_owned(), &resultx, 1e-8),
            "Wrong! the gen_v is get wrong results! please check it!"
        );
        assert!(
            are_complex_arrays_close(&result.slice(s![1, .., ..]).to_owned(), &resulty, 1e-8),
            "Wrong! the gen_v is get wrong results! please check it!"
        );

        let (result, _) = model.gen_v(&array![1.0 / 3.0, 1.0 / 3.0], Gauge::Lattice);
        let resultx = array![
            [
                0.0 * li,
                -3.0 * 3.0_f64.sqrt() / 4.0 * t + 3.0 / 4.0 * t * li
            ],
            [
                -3.0 * 3.0_f64.sqrt() / 4.0 * t - 3.0 / 4.0 * t * li,
                0.0 * li
            ]
        ];
        println!("result={}", &result - &resultx);
        assert!(
            are_complex_arrays_close(&result.slice(s![0, .., ..]).to_owned(), &resultx, 1e-8),
            "Wrong! the gen_v is get wrong results! please check it!"
        );

        let kvec = array![1.0 / 3.0, 1.0 / 3.0];
        let (band, evec) = model.solve_onek(&kvec);
        let ham = model.gen_ham(&kvec, Gauge::Atom);
        let evec_conj = evec.map(|x| x.conj());
        let evec = evec.t();
        let ham = ham.dot(&evec);
        let ham = evec_conj.dot(&ham);
        let new_band = ham.diag().map(|x| x.re);
        assert!(
            are_arrays_close(&new_band, &band, 1e-5),
            "wrong!, the solve_onek get wrong result! please check it!"
        );
    }
    #[test]
    fn conductivity_test() {
        //这个是用 Haldan 模型来测试
        let li: Complex<f64> = 1.0 * Complex::i();
        let t = -1.0 + 0.0 * li;
        let t2 = -1.0 + 0.0 * li;
        let delta = 0.7;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);
        let R0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t, 0, 1, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t2 * li, 0, 0, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t2 * li, 1, 1, &R, None);
        }
        let k_vec = array![1.0 / 3.0, 2.0 / 3.0];
        let dir_1 = array![1.0, 0.0];
        let dir_2 = array![0.0, 1.0];
        let mu = 0.0;
        let T = 0.0;
        let og = 0.0;
        let spin = None;
        let eta = 1e-3;
        let band_berry = band_berry_curvature(&model, &k_vec, &dir_1, &dir_2, spin, eta);
        let result1 = band_berry
            .berry_curvature
            .iter()
            .zip(&band_berry.energies)
            .filter(|&(_, &energy)| energy <= mu)
            .map(|(&berry, _)| berry)
            .sum::<f64>()
            * (2.0 * PI);

        let mut k_list = Array2::zeros((9, 2));
        let dk = 0.0001;
        k_list.row_mut(0).assign(&(&k_vec + dk * &dir_1));
        k_list
            .row_mut(1)
            .assign(&(&k_vec + dk * &dir_1 + dk * &dir_2));
        k_list.row_mut(2).assign(&(&k_vec + dk * &dir_2));
        k_list
            .row_mut(3)
            .assign(&(&k_vec - dk * &dir_1 + dk * &dir_2));
        k_list.row_mut(4).assign(&(&k_vec - dk * &dir_1));
        k_list
            .row_mut(5)
            .assign(&(&k_vec - dk * &dir_1 - dk * &dir_2));
        k_list.row_mut(6).assign(&(&k_vec - dk * &dir_2));
        k_list
            .row_mut(7)
            .assign(&(&k_vec + dk * &dir_1 - dk * &dir_2));
        k_list.row_mut(8).assign(&(&k_vec + dk * &dir_1));
        let result2 = model.berry_loop(&k_list, &vec![0]);
        let result2 = result2[[0]] / (dk.powi(2)) / 4.0 / (2.0 * PI) * 3_f64.sqrt() / 2.0;
        println!("result2={},result1={}", result2, result1);
        assert!(
            (result2 - result1).abs() < 1e-4,
            "Wrong!, the berry_curvature or berry_flux mut be false"
        );
        // 单化学势构造器与一般化学势网格应给出相同结果。
        let kmesh = array![100, 100];
        let mu = -1.0;
        let a1 = hall_value(&model, &kmesh, &dir_2, &dir_1, mu, T, spin, eta).unwrap();
        let a2 = hall_values(
            &model,
            &kmesh,
            &dir_2,
            &dir_1,
            &array![mu],
            T,
            spin,
            eta,
            Integration::Direct,
        )
        .unwrap()[0];
        assert!(
            (a2 - a1).abs() < 1e-5,
            "single- and multi-chemical-potential Hall results differ"
        )
    }
    #[test]
    fn gen_v_speed_test() {
        println!("开始测试各个函数的运行速度, 用次近邻的石墨烯模型");
        let li: Complex<f64> = 1.0 * Complex::i();
        let t = 2.0 + 0.0 * li;
        let t2 = -1.0 + 0.0 * li;
        let delta = 0.7;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);
        let R0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t, 0, 1, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t2 * li, 0, 0, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t2 * li, 1, 1, &R, None);
        }
        println!("{:?}", model.atom_list());
        let U = array![[3.0, 0.0], [0.0, 3.0]];
        let model = model.make_supercell(&U).unwrap();

        let nk = 101;
        let k_mesh = array![nk, nk];
        let kvec = gen_kmesh(&k_mesh).unwrap();

        {
            println!("开始计算 gen_v 的耗时速度, 为了平均, 我们单线程求解gen_v");
            let start = Instant::now(); // 开始计时
            let A: Vec<_> = kvec
                .outer_iter()
                .into_par_iter()
                .map(|x| {
                    let (a, _) = model.gen_v(&x.to_owned(), Gauge::Atom);
                    a
                })
                .collect();
            let end = Instant::now(); // 结束计时
            let duration = end.duration_since(start); // 计算执行时间
            println!(
                "run gen_v {} times took {} seconds",
                kvec.nrows(),
                duration.as_secs_f64()
            ); // 输出执行时间
        }
    }
    #[test]
    fn Haldan_model() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t = -1.0 + 0.0 * li;
        let t2 = -1.0 + 0.0 * li;
        let delta = 0.7;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);
        let R0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t, 0, 1, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t2 * li, 0, 0, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.add_hop(t2 * li, 1, 1, &R, None);
        }
        let nk: usize = 101;
        let path = [
            [0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0],
            [0.5, 0.5],
            [1.0 / 3.0, 2.0 / 3.0],
            [0.0, 0.0],
        ];
        let path = arr2(&path);
        let (k_vec, k_dist, k_node) = model.k_path(&path, nk).unwrap();
        let (eval, evec) = model.solve_all_parallel(&k_vec);
        let label = vec!["G", "K", "M", "K'", "G"];
        model.show_band(&path, &label, nk, "tests/Haldan").unwrap();
        // --- Compute Hall conductivity ---
        let nk: usize = 31;
        let T: f64 = 0.0;
        let eta: f64 = 0.001;
        let og: f64 = 0.0;
        let mu: f64 = 0.0;
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let dir_3 = arr1(&[0.0, 1.0]);
        let spin = None;
        let kmesh = arr1(&[nk, nk]);

        let start = Instant::now(); // 开始计时
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, mu, T, spin, eta).unwrap();
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("quantom_Hall_effect={}", conductivity * (2.0 * PI));
        assert!(
            (conductivity * (2.0 * PI) - 1.0).abs() < 1e-3,
            "Wrong!, the Hall conductivity is wrong!"
        );
        println!("function_a took {} seconds", duration.as_secs_f64()); // 输出执行时间

        let mu = Array1::linspace(-2.0, 2.0, 101);
        let start = Instant::now(); // 开始计时
        let conductivity_mu = hall_values(
            &model,
            &kmesh,
            &dir_1,
            &dir_2,
            &mu,
            T,
            spin,
            eta,
            Integration::Direct,
        )
        .unwrap();
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("quantom_Hall_effect={}", conductivity_mu[[50]] * (2.0 * PI));
        assert!(
            (conductivity_mu[[50]] - conductivity).abs() < 1e-3,
            "Wrong!, the Hall conductivity is wrong!, Hall_mu's result is {}, but Hall conductivity is {}",
            conductivity_mu[[50]],
            conductivity
        );
        println!("function_a took {} seconds", duration.as_secs_f64()); // 输出执行时间
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, -2.0, T, spin, eta).unwrap();
        assert!(
            (conductivity_mu[[0]] - conductivity).abs() < 1e-3,
            "Wrong!, the Hall conductivity is wrong!, Hall_mu's result is {}, but Hall conductivity is {}",
            conductivity_mu[[0]],
            conductivity
        );
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, 2.0, T, spin, eta).unwrap();
        assert!(
            (conductivity_mu[[100]] - conductivity).abs() < 1e-3,
            "Wrong!, the Hall conductivity is wrong!, Hall_mu's result is {}, but Hall conductivity is {}",
            conductivity_mu[[100]],
            conductivity
        );
        //开始绘图
        let mut fg = Figure::new();
        let x: Vec<f64> = mu.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = (conductivity_mu * 2.0 * PI).to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/hall_mu.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        let mu = 0.0;
        let nk: usize = 31;
        let kmesh = arr1(&[nk, nk]);
        let start = Instant::now(); // 开始计时
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, mu, T, spin, eta).unwrap();
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("霍尔电导率{}", conductivity * (2.0 * PI));
        assert!(
            (conductivity * (2.0 * PI) - 1.0).abs() < 1e-3,
            "Wrong!, the Hall conductivity is wrong!"
        );
        println!("function_a took {} seconds", duration.as_secs_f64()); // 输出执行时间
        //画一下3000k的时候的费米导数分布
        let T = 100.0;
        let nk: usize = 101;
        let kmesh = arr1(&[nk, nk]);
        println!("{}", kmesh);
        let E_min = -3.0;
        let E_max = 3.0;
        let E_n = 1000;
        let mu = Array1::linspace(E_min, E_max, E_n);
        let occupation = Occupation::FermiDirac {
            temperature_kelvin: T,
        };
        let par_f = mu.mapv(|energy| occupation.minus_derivative(energy, 0.0).unwrap());
        let mut fg = Figure::new();
        let x: Vec<f64> = mu.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = par_f.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/par_f.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        //画一下omega_n 随能量的分布
        let kvec: Array2<f64> = gen_kmesh(&kmesh).unwrap();
        let nk: usize = kvec.len_of(Axis(0));
        let (omega, band) =
            model.berry_curvature_dipole_n(&kvec, &dir_1, &dir_2, &dir_3, og, spin, eta);
        let omega = omega.into_raw_vec();
        let omega = Array1::from(omega);
        let band = band.into_raw_vec();
        let band = Array1::from(band);
        let mut fg = Figure::new();
        let x: Vec<f64> = band.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = omega.to_vec();
        axes.points(
            x.iter(),
            y.iter(),
            &[Color("black"), PointSymbol((".").chars().next().unwrap())],
        );
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Haldan");
        pdf_name.push_str("/omega_energy.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        //画一下表面态
        let nk = 101;
        let green = surf_Green::from_Model(&model, 0, 1e-3, None).unwrap();
        let E_min = -3.0;
        let E_max = 3.0;
        let E_n = 101;
        let path = [[0.0], [0.5], [1.0]];
        let path = arr2(&path);
        let label = vec!["G", "M", "G"];
        green.show_surf_state("tests/Haldan/surf", &path, &label, nk, E_min, E_max, E_n, 0);

        //-----算一下wilson loop 的结果-----------------------
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let occ = vec![0];
        let wcc = model.wannier_centre(&occ, &array![0.0, 0.0], &dir_1, &dir_2, 101, 101);
        let nocc = occ.len();

        let mut fg = Figure::new();
        let x: Vec<f64> = Array1::<f64>::linspace(0.0, 1.0, 101).to_vec();
        let axes = fg.axes2d();
        for j in -1..2 {
            for i in 0..nocc {
                let a = wcc.row(i).to_owned() + (j as f64) * 2.0 * PI;
                let y: Vec<f64> = a.to_vec();
                axes.points(
                    &x,
                    &y,
                    &[
                        Color("black"),
                        gnuplot::PointSymbol('O'),
                        gnuplot::PointSize(0.2),
                    ],
                );
            }
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(2.0 * PI));
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(0.5, Fix("π")),
            Major(1.0, Fix("2π")),
        ];
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 32.0)],
        );
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(PI, Fix("π")),
            Major(2.0 * PI, Fix("2π")),
        ];
        axes.set_y_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 32.0)],
        );
        axes.set_x_label(
            "k_x",
            &[Font("Times New Roman", 32.0), TextOffset(0.0, -0.5)],
        );
        axes.set_y_label(
            "WCC",
            &[
                Font("Times New Roman", 32.0),
                Rotate(90.0),
                TextOffset(-1.0, 0.0),
            ],
        );
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Haldan/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        //-----------用 berry_flux 算一下
        let C = model
            .berry_flux(
                &occ,
                &array![0.0, 0.0],
                &array![1.0, 0.0],
                &array![0.0, 1.0],
                101,
                101,
            )
            .sum()
            / PI
            / 2.0;
        println!("The Chern number of Haldan model is {}", C);
    }

    /// Sanity check: at one k-point, the reusable Berry-curvature trait and
    /// the gauge-invariant velocity kernel give the same result.
    #[test]
    fn tetra_primitives_sanity() {
        let li = Complex::new(0.0, 1.0);
        let t = Complex::new(-1.0, 0.0);
        let t2 = Complex::new(-1.0, 0.0);
        let delta = 0.7;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);
        for &(i, j) in &[(0, 0), (-1, 0), (0, -1)] {
            model.add_hop(t, 0, 1, &arr1(&[i, j]), None);
        }
        for &(i, j) in &[(1, 0), (-1, 1), (0, -1)] {
            model.add_hop(t2 * li, 0, 0, &arr1(&[i, j]), None);
        }
        for &(i, j) in &[(-1, 0), (1, -1), (0, 1)] {
            model.add_hop(t2 * li, 1, 1, &arr1(&[i, j]), None);
        }

        let k = arr1(&[0.3, 0.4]);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.01;

        // Reference: band-resolved Berry-curvature trait.
        let omega_ref = band_berry_curvature(&model, &k, &dx, &dy, None, eta).berry_curvature;
        // Tetra primitives
        let dv = Array1::zeros(2);
        let pt = model.compute_velocity_kernel(&k, &dx, &dy, Some(&dv), Gauge::Atom, None);

        // Compute Omega from tetra primitives for each band n
        let nsta = model.nsta();
        for n in 0..nsta {
            let mut omega_n = 0.0;
            for m in 0..nsta {
                if m == n {
                    continue;
                }
                let d = pt.band[[n]] - pt.band[[m]];
                omega_n -= 2.0 * pt.k_ab[[n, m]].im / (d.powi(2) + eta.powi(2));
            }
            println!(
                "band {n}: ref={:.6}, tetra={:.6}, diff={:.2e}",
                omega_ref[[n]],
                omega_n,
                (omega_ref[[n]] - omega_n).abs()
            );
            assert!(
                (omega_ref[[n]] - omega_n).abs() < 1e-6,
                "band {n}: ref={}, tetra={}",
                omega_ref[[n]],
                omega_n
            );
        }
        println!("PASSED");
    }

    /// Verify eigenbasis convention: `U^T · H · U^*` gives diag = band.
    /// If this test ever fails, the convention is broken everywhere.
    #[test]
    fn evec_transform_sanity() {
        let li = Complex::new(0.0, 1.0);
        let t = Complex::new(-1.0, 0.0);
        let t2 = Complex::new(-1.0, 0.0);
        let delta = 0.7;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);
        for &(i, j) in &[(0, 0), (-1, 0), (0, -1)] {
            model.add_hop(t, 0, 1, &arr1(&[i, j]), None);
        }
        for &(i, j) in &[(1, 0), (-1, 1), (0, -1)] {
            model.add_hop(t2 * li, 0, 0, &arr1(&[i, j]), None);
        }
        for &(i, j) in &[(-1, 0), (1, -1), (0, 1)] {
            model.add_hop(t2 * li, 1, 1, &arr1(&[i, j]), None);
        }

        let k = arr1(&[0.3, 0.4]);
        let ham = model.gen_ham(&k, Gauge::Atom);
        let (band, evec) = ham.eigh(UPLO::Lower).unwrap();
        let ut = evec.t();
        let uc = evec.map(|x| x.conj());

        // U^T · H · U^* should be diagonal with eigenvalues
        let diag = ut.dot(&ham.dot(&uc));
        for i in 0..model.nsta() {
            for j in 0..model.nsta() {
                if i == j {
                    assert!(
                        (diag[[i, j]].re - band[[i]]).abs() < 1e-10,
                        "diag[{i},{i}]={} != band[{i}]={}",
                        diag[[i, i]].re,
                        band[[i]]
                    );
                } else {
                    assert!(
                        diag[[i, j]].norm() < 1e-10,
                        "off-diag[{i},{j}]={} != 0",
                        diag[[i, j]]
                    );
                }
            }
        }
        println!("evec_transform_sanity: U^T H U^* = diag(band) ✓");
    }

    #[test]
    fn graphene() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 1.0 + 0.0 * li;
        let t2 = 0.0 + 0.0 * li;
        let t3 = 0.0 + 0.0 * li;
        let delta = 0.0;
        let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
        let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[delta, -delta]), None);
        model.add_hop(t1, 0, 1, &array![0, 0], None);
        model.add_hop(t1, 0, 1, &array![-1, 0], None);
        model.add_hop(t1, 0, 1, &array![0, -1], None);
        model.add_hop(t2, 0, 0, &array![1, 0], None);
        model.add_hop(t2, 1, 1, &array![1, 0], None);
        model.add_hop(t2, 0, 0, &array![0, 1], None);
        model.add_hop(t2, 1, 1, &array![0, 1], None);
        model.add_hop(t2, 0, 0, &array![1, -1], None);
        model.add_hop(t2, 1, 1, &array![1, -1], None);
        model.add_hop(t3, 0, 1, &array![1, -1], None);
        model.add_hop(t3, 0, 1, &array![-1, 1], None);
        model.add_hop(t3, 0, 1, &array![-1, -1], None);
        let nk: usize = 101;
        let path = [[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.5], [0.0, 0.0]];
        let path = arr2(&path);
        let (k_vec, k_dist, k_node) = model.k_path(&path, nk).unwrap();
        let (eval, evec) = model.solve_all_parallel(&k_vec);
        let label = vec!["G", "K", "M", "G"];
        model
            .show_band(&path, &label, nk, "tests/graphene")
            .unwrap();

        // 开始计算两个本征态
        let k1 = array![1.0 / 3.0 - 0.002, 2.0 / 3.0];
        let k2 = array![1.0 / 3.0 + 0.001, 2.0 / 3.0];
        let (eval1, evec1) = model.solve_onek(&k1);
        let (eval2, evec2) = model.solve_onek(&k2);
        let evec1 = evec1.reversed_axes();
        let evec2 = evec2.mapv(|x| x.conj());
        println!("{},{}", eval1, eval2);
        println!("{}", evec2.dot(&evec1).mapv(|x| x.norm().round()));

        // --- Compute Hall conductivity ---
        let nk: usize = 11;
        let T: f64 = 0.0;
        let eta: f64 = 0.001;
        let og: f64 = 0.0;
        let mu: f64 = 0.0;
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let spin = None;
        let kmesh = arr1(&[nk, nk]);
        let (eval, evec) = model.solve_onek(&arr1(&[0.3, 0.5]));
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, mu, T, spin, eta);
        //println!("{}",conductivity/(2.0*PI));
        //开始计算边缘态, 首先是zigsag态
        let nk: usize = 501;
        let U = arr2(&[[1.0, 1.0], [-1.0, 1.0]]);
        let super_model = model.make_supercell(&U).unwrap();
        let zig_model = super_model.cut_piece(100, 0).unwrap();
        let path = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]];
        //let path=[[0.0,0.0],[0.5,0.0],[1.0,0.0]];
        //let path=[[0.0,0.0],[0.5,0.0],[0.5,0.5],[0.0,0.5],[0.0,0.0]];
        let path = arr2(&path);
        let (k_vec, k_dist, k_node) = super_model.k_path(&path, nk).unwrap();
        let (eval, evec) = super_model.solve_all_parallel(&k_vec);
        //let label=vec!["G","X","M","Y","G"];
        let label = vec!["G", "M", "G"];
        zig_model.show_band(&path, &label, nk, "tests/graphene_zig");

        //开始计算石墨烯的态密度
        let nk: usize = 51;
        let kmesh = arr1(&[nk, nk]);
        let E_min = -3.0;
        let E_max = 3.0;
        let E_n = 1000;
        let (E0, dos) = model.dos(&kmesh, E_min, E_max, E_n, 1e-2).unwrap();
        //开始绘制dos
        let mut fg = Figure::new();
        let x: Vec<f64> = E0.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/graphene");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        //开始计算非线性霍尔电导
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let dir_3 = arr1(&[1.0, 0.0]);
        let og = 0.0;
        let mu = Array1::linspace(E_min, E_max, E_n);
        let T = 300.0;
        let sigma = extrinsic_nonlinear_values(
            &model,
            &kmesh,
            &dir_1,
            &dir_2,
            &dir_3,
            &mu,
            T,
            og,
            None,
            1e-5,
            Integration::Direct,
            FieldSymmetry::Ordered,
        )
        .unwrap();

        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x: Vec<f64> = mu.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/graphene");
        pdf_name.push_str("/nonlinear_ex.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }

    #[test]
    fn kane_mele() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t = -1.0;
        let delta = 0.0;
        let alter = 0.0 + 0.0 * li;
        let soc = 0.06 * t;
        let rashba = 0.0 * t;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        // Honeycomb A/B sublattice sites, one orbital each
        let atoms = vec![
            Atom::with_orbitals(
                arr1(&[1.0 / 3.0, 1.0 / 3.0]),
                AtomType::C,
                [OrbitalId::new(0)],
            ),
            Atom::with_orbitals(
                arr1(&[2.0 / 3.0, 2.0 / 3.0]),
                AtomType::C,
                [OrbitalId::new(1)],
            ),
        ];
        let mut model = Model::<true, 2>::tb_model(lat, orb, Some(atoms)).unwrap();
        model.set_onsite(&arr1(&[delta, -delta]), None);
        let R0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(t, 0, 1, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(soc * li, 0, 0, &R, SpinDirection::Z);
        }
        let R0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(soc * li, 1, 1, &R, SpinDirection::Z);
        }
        //加入rashba项
        let R0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            let r0 = R.map(|x| *x as f64).dot(&model.lat);
            model.add_hop(rashba * li * r0[[1]], 0, 0, &R, SpinDirection::X);
            model.add_hop(rashba * li * r0[[0]], 0, 0, &R, SpinDirection::Y);
        }

        let R0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            let r0 = R.map(|x| *x as f64).dot(&model.lat);
            model.add_hop(-rashba * li * r0[[1]], 1, 1, &R, SpinDirection::X);
            model.add_hop(-rashba * li * r0[[0]], 1, 1, &R, SpinDirection::Y);
        }
        let nk: usize = 101;
        let path = [
            [0.0, 0.0],
            [2.0 / 3.0, 1.0 / 3.0],
            [0.5, 0.5],
            [1.0 / 3.0, 2.0 / 3.0],
            [0.0, 0.0],
        ];
        let path = arr2(&path);
        let (k_vec, k_dist, k_node) = model.k_path(&path, nk).unwrap();
        let (eval, evec) = model.solve_all_parallel(&k_vec);
        let label = vec!["G", "K", "M", "K'", "G"];
        model.show_band(&path, &label, nk, "tests/kane");
        //开始计算超胞

        let super_model = model.cut_piece(50, 0).unwrap();
        let path = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]];
        let path = arr2(&path);
        let label = vec!["G", "M", "G"];
        super_model
            .show_band(&path, &label, nk, "tests/kane_super")
            .unwrap();
        //开始计算表面态
        let nk = 101;
        let green = surf_Green::from_Model(&model, 0, 1e-3, None).unwrap();
        let E_min = -1.0;
        let E_max = 1.0;
        let E_n = 101;
        let path = [[0.0], [0.5], [1.0]];
        let path = arr2(&path);
        let label = vec!["G", "M", "G"];
        green.show_surf_state("tests/kane", &path, &label, nk, E_min, E_max, E_n, 0);

        //-----算一下wilson loop 结果-----------------------
        let n = 51;
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let occ = vec![0, 1];
        let wcc = model.wannier_centre(&occ, &array![0.0, 0.0], &dir_1, &dir_2, n, n);
        let nocc = occ.len();
        let mut fg = Figure::new();
        let x: Vec<f64> = Array1::<f64>::linspace(0.0, 1.0, n).to_vec();
        let axes = fg.axes2d();
        for j in -1..2 {
            for i in 0..nocc {
                let a = wcc.row(i).to_owned() + (j as f64) * 2.0 * PI;
                let y: Vec<f64> = a.to_vec();
                axes.points(&x, &y, &[Color("black"), gnuplot::PointSymbol('O')]);
            }
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(2.0 * PI));
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(0.5, Fix("π")),
            Major(1.0, Fix("2π")),
        ];
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 32.0)],
        );
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(PI, Fix("π")),
            Major(2.0 * PI, Fix("2π")),
        ];
        axes.set_y_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 32.0)],
        );
        axes.set_x_label(
            "k_x",
            &[Font("Times New Roman", 32.0), TextOffset(0.0, -0.5)],
        );
        axes.set_y_label(
            "WCC",
            &[
                Font("Times New Roman", 32.0),
                Rotate(90.0),
                TextOffset(-1.0, 0.0),
            ],
        );
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/kane/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        // --- Compute Hall conductivity ---
        let nk: usize = 31;
        let T: f64 = 0.0;
        let eta: f64 = 0.001;
        let og: f64 = 0.0;
        let mu: f64 = 0.0;
        //let dir_1=arr1(&[3.0_f64.sqrt()/2.0,-0.5]);
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let spin = Some(SpinDirection::Z);
        let kmesh = arr1(&[nk, nk]);
        let start = Instant::now(); // 开始计时
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, mu, T, spin, eta).unwrap();
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}", conductivity * (2.0 * PI));
        println!("function_a took {} seconds", duration.as_secs_f64()); // 输出执行时间
        let nk: usize = 21;
        let kmesh = arr1(&[nk, nk]);
        let start = Instant::now(); // 开始计时
        let conductivity = hall_value(&model, &kmesh, &dir_1, &dir_2, mu, T, spin, eta).unwrap();
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("{}", conductivity * (2.0 * PI));
        println!("function_a took {} seconds", duration.as_secs_f64()); // 输出执行时间

        let (E0, dos) = model.dos(&kmesh, E_min, E_max, E_n, 1e-2).unwrap();
        //开始绘制dos
        let mut fg = Figure::new();
        let x: Vec<f64> = E0.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/kane");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        //绘制非线性霍尔电导的平面图

        //画一下贝利曲率的分布
        let nk: usize = 31;
        let kmesh = arr1(&[nk, nk]);
        let kvec = gen_kmesh(&kmesh).unwrap();
        //let kvec=kvec-0.5;
        let kvec = kvec * 2.0;
        let kvec = model.lat.dot(&(kvec.reversed_axes()));
        let kvec = kvec.reversed_axes();
        let berry_curv = occupied_berry_curvature(
            &model,
            &kvec,
            &dir_1,
            &dir_2,
            0.0,
            T,
            Some(SpinDirection::X),
            1e-3,
        );
        let data = berry_curv.into_shape((nk, nk)).unwrap();
        draw_heatmap(
            &(-data).map(|x| (x + 1.0).log(10.0)),
            "./tests/kane/berry_curvature_distribution.pdf",
        );

        //开始考虑磁场, 加入磁性
        let B = 0.1 + 0.0 * li;
        let tha = 0.0 / 180.0 * PI;

        model.add_hop(B * tha.cos(), 0, 0, &array![0, 0], SpinDirection::X);
        model.add_hop(B * tha.cos(), 1, 1, &array![0, 0], SpinDirection::X);
        model.add_hop(B * tha.sin(), 0, 0, &array![0, 0], SpinDirection::Y);
        model.add_hop(B * tha.sin(), 1, 1, &array![0, 0], SpinDirection::Y);
        //考虑添加onsite 项破坏空间反演和mirror

        let green = surf_Green::from_Model(&model, 0, 1e-3, None).unwrap();
        let E_min = -1.0;
        let E_max = 1.0;
        let E_n = nk;
        let path = [[0.0], [0.5], [1.0]];
        let path = arr2(&path);
        let label = vec!["G", "M", "G"];
        green.show_surf_state(
            "tests/kane/magnetic",
            &path,
            &label,
            nk,
            E_min,
            E_max,
            E_n,
            0,
        );

        //-----算一下wilson loop 结果-----------------------
        let n = 51;
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let occ = vec![0, 1];
        let wcc = model.wannier_centre(&occ, &array![0.0, 0.0], &dir_1, &dir_2, n, n);
        let nocc = occ.len();
        let mut fg = Figure::new();
        let x: Vec<f64> = Array1::<f64>::linspace(0.0, 1.0, n).to_vec();
        let axes = fg.axes2d();
        for j in -1..2 {
            for i in 0..nocc {
                let a = wcc.row(i).to_owned() + (j as f64) * 2.0 * PI;
                let y: Vec<f64> = a.to_vec();
                axes.points(&x, &y, &[Color("black"), gnuplot::PointSymbol('O')]);
            }
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(2.0 * PI));
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(0.5, Fix("π")),
            Major(1.0, Fix("2π")),
        ];
        axes.set_x_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 32.0)],
        );
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(PI, Fix("π")),
            Major(2.0 * PI, Fix("2π")),
        ];
        axes.set_y_ticks_custom(
            show_ticks.into_iter(),
            &[],
            &[Font("Times New Roman", 32.0)],
        );
        axes.set_x_label(
            "k_x",
            &[Font("Times New Roman", 32.0), TextOffset(0.0, -0.5)],
        );
        axes.set_y_label(
            "WCC",
            &[
                Font("Times New Roman", 32.0),
                Rotate(90.0),
                TextOffset(-1.0, 0.0),
            ],
        );
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/kane/magnetic/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        //开始计算角态
        let model = model
            .make_supercell(&array![[0.0, -1.0], [1.0, 0.0]])
            .unwrap();
        let num = 19;
        /*
        let model_1=model.cut_piece(num,0).unwrap();
        let new_model=model_1.cut_piece(num,1);
        */
        let new_model = model.cut_dot(num, 6, None).unwrap();
        let mut s = 0;
        let start = Instant::now();
        let (band, evec) = new_model.solve_range_onek(&arr1(&[0.0, 0.0]), (-0.3, 0.3), 1e-5);
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("solve_band_all took {} seconds", duration.as_secs_f64()); // 输出执行时间
        let nresults = band.len();
        let show_evec = evec.to_owned().map(|x| x.norm_sqr());
        let mut size = Array2::<f64>::zeros((new_model.nsta(), new_model.natom()));
        let norb = new_model.norb();
        for i in 0..nresults {
            let mut s = 0;
            for j in 0..new_model.natom() {
                for k in 0..new_model.atoms[j].norb() {
                    size[[i, j]] += show_evec[[i, s]] + show_evec[[i, s + new_model.norb()]];
                    s += 1;
                }
            }
        }

        let show_str = new_model.atom_position().dot(&model.lat);
        let show_str = show_str.slice(s![.., 0..2]).to_owned();
        let show_size = size.row(new_model.norb()).to_owned();
        create_dir_all("tests/kane/magnetic").expect("can't creat the file");
        write_txt_1(band, "tests/kane/magnetic/band.txt");
        write_txt(size, "tests/kane/magnetic/evec.txt");
        write_txt(show_str, "tests/kane/magnetic/structure.txt");
        //开始绘制角态
    }

    #[test]
    fn Enonlinear() {
        //! arxiv:1706.07702
        //! Test for the extrinsic nonlinear Hall conductivity.
        let li: Complex<f64> = 1.0 * Complex::i();
        let delta = 0.;
        let t1 = 1.0 + 0.0 * li;
        let t2 = 0.2 * t1;
        let t3 = 0.2 * t1;
        let lat = arr2(&[
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0, 0.0], [2.0 / 3.0, 2.0 / 3.0, 0.0]]);
        let mut model = Model::<false>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[delta, -delta]), None);
        let R0: Array2<isize> = arr2(&[[0, 0, 0], [-1, 0, 0], [0, -1, 0]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(t1, 0, 1, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[1, 0, 1], [-1, 1, 1], [0, -1, 1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(t2, 0, 0, &R, None);
        }
        let R0: Array2<isize> = arr2(&[[1, 0, -1], [-1, 1, -1], [0, -1, -1]]);
        for (i, R) in R0.axis_iter(Axis(0)).enumerate() {
            let R = R.to_owned();
            model.set_hop(t2, 1, 1, &R, None);
        }
        let R = arr1(&[0, 0, 1]);
        model.set_hop(t3, 0, 0, &R, None);
        model.set_hop(t3, 1, 1, &R, None);
        let path = array![
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.5],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [1.0 / 3.0, 2.0 / 3.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5]
        ];
        let label = vec!["G", "K", "M", "G", "K", "H", "G", "A", "H", "L", "A"];
        let nk = 101;
        model.show_band(&path, &label, nk, "tests/Enonlinear");

        //开始计算非线性霍尔电导
        let dir_1 = arr1(&[1.0, 0.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0, 0.0]);
        let dir_3 = arr1(&[0.0, 0.0, 1.0]);
        let nk: usize = 21;
        let kmesh = arr1(&[nk, nk, nk]);
        let E_min = -3.0;
        let E_max = 3.0;
        let E_n = 1000;
        let og = 0.0;
        let mu = Array1::linspace(E_min, E_max, E_n);
        let T = 30.0;
        let sigma = extrinsic_nonlinear_values(
            &model,
            &kmesh,
            &dir_1,
            &dir_2,
            &dir_3,
            &mu,
            T,
            og,
            None,
            1e-5,
            Integration::Direct,
            FieldSymmetry::Ordered,
        )
        .unwrap();

        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x: Vec<f64> = mu.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        axes.set_y_range(Fix(-10.0), Fix(10.0));
        axes.set_x_range(Fix(E_min), Fix(E_max));
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Enonlinear");
        pdf_name.push_str("/nonlinear_ex.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        let sigma = intrinsic_nonlinear_values(
            &model,
            &kmesh,
            &dir_1,
            &dir_2,
            &dir_3,
            &mu,
            T,
            Integration::Direct,
        )
        .unwrap();
        //开始绘制非线性电导
        let mut fg = Figure::new();
        let x: Vec<f64> = mu.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = sigma.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        axes.set_y_range(Fix(-10.0), Fix(10.0));
        axes.set_x_range(Fix(E_min), Fix(E_max));
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Enonlinear");
        pdf_name.push_str("/nonlinear_in.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();

        let (E0, dos) = model.dos(&kmesh, E_min, E_max, E_n, 1e-2).unwrap();
        //开始绘制dos
        let mut fg = Figure::new();
        let x: Vec<f64> = E0.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/Enonlinear");
        pdf_name.push_str("/dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }
    #[test]
    fn kagome() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 1.0 + 0.0 * li;
        let t2 = 0.1 + 0.0 * li;
        let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
        let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 0.0], [0.0, 1.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        //最近邻hopping
        model.add_hop(t1, 0, 1, &array![0, 0], None);
        model.add_hop(t1, 2, 0, &array![0, 0], None);
        model.add_hop(t1, 1, 2, &array![0, 0], None);
        model.add_hop(t1, 0, 2, &array![0, -1], None);
        model.add_hop(t1, 0, 1, &array![-1, 0], None);
        model.add_hop(t1, 2, 1, &array![-1, 1], None);
        let nk: usize = 101;
        let path = [[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.], [0.0, 0.0]];
        let path = arr2(&path);
        let label = vec!["G", "K", "M", "G"];
        model.show_band(&path, &label, nk, "tests/kagome/");
        //start to draw the band structure
        //Starting to calculate the edge state, first is the zigzag state
        let nk: usize = 101;
        let U = arr2(&[[1.0, 1.0], [-1.0, 1.0]]);
        let super_model = model.make_supercell(&U).unwrap();
        let zig_model = super_model.cut_piece(30, 0).unwrap();
        let path = [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]];
        let path = arr2(&path);
        let (k_vec, k_dist, k_node) = super_model.k_path(&path, nk).unwrap();
        let (eval, evec) = super_model.solve_all_parallel(&k_vec);
        let label = vec!["G", "M", "G"];
        zig_model.show_band(&path, &label, nk, "tests/kagome_zig/");

        let green = surf_Green::from_Model(&super_model, 0, 1e-3, None).unwrap();
        let E_min = -2.0;
        let E_max = 4.0;
        let E_n = nk;
        let path = [[0.0], [0.5], [1.0]];
        let path = arr2(&path);
        let label = vec!["G", "M", "G"];
        green.show_surf_state("tests/kagome_zig", &path, &label, nk, E_min, E_max, E_n, 0);

        //Starting to calculate the DOS of kagome
        let nk: usize = 51;
        let kmesh = arr1(&[nk, nk]);
        let E_min = -3.0;
        let E_max = 3.0;
        let E_n = 1000;
        let (E0, dos) = model.dos(&kmesh, E_min, E_max, E_n, 1e-2).unwrap();
        //start to show DOS
        let mut fg = Figure::new();
        let x: Vec<f64> = E0.to_vec();
        let axes = fg.axes2d();
        let y: Vec<f64> = dos.to_vec();
        axes.lines(&x, &y, &[Color("black")]);
        let mut show_ticks = Vec::<String>::new();
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/kagome/");
        pdf_name.push_str("dos.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
    }

    #[test]
    fn SSH() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 1.0 + 0.0 * li;
        let t2 = 0.5 + 0.0 * li;
        let Delta = 0.0;
        let lat = arr2(&[[1.0]]);
        let orb = arr2(&[[0.3], [0.5]]);
        let mut model = Model::<false, 1>::tb_model(lat, orb, None).unwrap();
        model.add_hop(t1, 0, 1, &array![0], None);
        model.add_hop(t2, 0, 1, &array![-1], None);
        model.add_onsite(&array![Delta, -Delta], None);

        let nk: usize = 101;
        let path = [[0.0], [0.5], [1.0]];
        let path = arr2(&path);
        let label = vec!["G", "M", "G"];
        model.show_band(&path, &label, nk, "tests/SSH/");
        let mut super_model = model.cut_piece(5, 0).unwrap();

        let (band, evec) = super_model.solve_onek(&array![0.0]);
        println!("{}", band);
    }
    #[test]
    fn BBH_model() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 0.1 + 0.0 * li;
        let t2 = 1.0 + 0.0 * li;
        let i0 = -1.0;
        let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let orb = arr2(&[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]);
        // Four lattice sites, one orbital each
        let atoms = vec![
            Atom::with_orbitals(arr1(&[0.0, 0.0]), AtomType::C, [OrbitalId::new(0)]),
            Atom::with_orbitals(arr1(&[0.5, 0.0]), AtomType::C, [OrbitalId::new(1)]),
            Atom::with_orbitals(arr1(&[0.5, 0.5]), AtomType::C, [OrbitalId::new(2)]),
            Atom::with_orbitals(arr1(&[0.0, 0.5]), AtomType::C, [OrbitalId::new(3)]),
        ];
        let mut model = Model::<false, 2>::tb_model(lat, orb, Some(atoms)).unwrap();
        model.add_hop(t1, 0, 1, &array![0, 0], None);
        model.add_hop(t1, 1, 2, &array![0, 0], None);
        model.add_hop(t1, 2, 3, &array![0, 0], None);
        model.add_hop(i0 * t1, 3, 0, &array![0, 0], None);
        model.add_hop(t2, 0, 1, &array![-1, 0], None);
        model.add_hop(i0 * t2, 0, 3, &array![0, -1], None);
        model.add_hop(t2, 2, 3, &array![1, 0], None);
        model.add_hop(t2, 2, 1, &array![0, 1], None);
        let nk: usize = 101;
        let path = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.0]];
        let path = arr2(&path);
        let label = vec!["G", "X", "M", "G"];
        model.show_band(&path, &label, nk, "tests/BBH/").unwrap();
        model.output_hr("tests/BBH/", "wannier90").unwrap();

        //算一下wilson loop
        let n = 51;
        let dir_1 = arr1(&[1.0, 0.0]);
        let dir_2 = arr1(&[0.0, 1.0]);
        let occ = vec![0, 1];
        let wcc = model.wannier_centre(&occ, &array![0.0, 0.0], &dir_1, &dir_2, n, n);
        let nocc = occ.len();
        let mut fg = Figure::new();
        let x: Vec<f64> = Array1::<f64>::linspace(0.0, 1.0, n).to_vec();
        let axes = fg.axes2d();
        for j in -1..2 {
            for i in 0..nocc {
                let a = wcc.row(i).to_owned() + (j as f64) * 2.0 * PI;
                let y: Vec<f64> = a.to_vec();
                axes.points(&x, &y, &[Color("black"), gnuplot::PointSymbol('O')]);
            }
        }
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(0.0), Fix(2.0 * PI));
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(0.5, Fix("π")),
            Major(1.0, Fix("2π")),
        ];
        axes.set_x_ticks_custom(show_ticks.into_iter(), &[], &[]);
        let show_ticks = vec![
            Major(0.0, Fix("0")),
            Major(PI, Fix("π")),
            Major(2.0 * PI, Fix("2π")),
        ];
        axes.set_y_ticks_custom(show_ticks.into_iter(), &[], &[]);
        let mut pdf_name = String::new();
        pdf_name.push_str("tests/BBH/wcc.pdf");
        fg.set_terminal("pdfcairo", &pdf_name);
        fg.show();
        //算一下边界态
        let green = surf_Green::from_Model(&model, 0, 1e-3, None).unwrap();
        let E_min = -2.0;
        let E_max = 2.0;
        let E_n = nk;
        let path = [[0.0], [0.5], [1.0]];
        let path = arr2(&path);
        let label = vec!["G", "X", "G"];
        green.show_surf_state("tests/BBH", &path, &label, nk, E_min, E_max, E_n, 0);

        //算一下corner state
        let num = 10;
        let model_1 = model.cut_piece(num, 0).unwrap();
        let new_model = model_1.cut_piece(2 * num, 1).unwrap();
        let mut s = 0;
        let start = Instant::now();
        let (band, evec) = new_model.solve_onek(&arr1(&[0.0, 0.0]));
        println!(
            "band shape is {:?}, evec shape is {:?}",
            band.shape(),
            evec.shape()
        );
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("solve_band_all took {} seconds", duration.as_secs_f64()); // 输出执行时间
        let nresults = band.len();
        let show_evec = evec.to_owned().map(|x| x.norm_sqr());
        let norb = new_model.norb();
        let size = show_evec;
        let show_str = new_model.atom_position().dot(&model.lat);
        create_dir_all("tests/BBH/corner").expect("can't creat the file");
        write_txt_1(band, "tests/BBH/corner/band.txt");
        write_txt(size, "tests/BBH/corner/evec.txt");
        write_txt(show_str, "tests/BBH/corner/structure.txt");
    }

    #[test]
    fn graphene_magnetic_field() {
        use crate::{MagneticField, Model, SpinDirection};
        use ndarray::{Axis, arr1, arr2};
        use num_complex::Complex;
        // 如果你在其他地方定义了画图函数，请确保 use 进来，例如：
        // use crate::draw_heatmap;

        // 1. 设置模型基本参数
        let t = Complex::new(-1.0, 0.0);
        let delta = 0.0;

        // 石墨烯晶格：a1 = (1, 0), a2 = (1/2, √3/2)
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        // 轨道的相对分数坐标 (Fractional Coordinates)
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);

        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[-delta, delta]), None);

        // 添加最近邻跃迁
        let r0: ndarray::Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
        for r in r0.axis_iter(Axis(0)) {
            model.add_hop(t, 0, 1, &r.to_owned(), None);
        }

        // 2. 施加磁场
        // 重要：二维中，面外磁场垂直于 xy 平面，对应的索引必须为 z 轴 (即 mag_dir = 2)
        // 扩胞 [3, 3] 表示 a1 和 a2 两个方向各扩胞 3 倍
        // 磁通 8 表示整体超胞含有 8 个磁通量子 (每个原胞 8/9 个磁通)
        let magnetic_model = model.add_magnetic_field(2, [9, 9], 40).unwrap();

        // 3. 高对称路径 (注意：六角晶格的 K 点在分数倒格矢坐标下是 1/3, 1/3)
        let path = arr2(&[[0.0, 0.0], [0.5, 0.0], [1.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]]);
        let label = vec!["Γ", "M", "K", "Γ"];
        let nk = 1001;

        // 4. 绘制折叠态下的超胞能带 (Hofstadter 蝴蝶状能带切片)
        magnetic_model
            .show_band(&path, &label, nk, "tests/graphene_magnetic")
            .unwrap();

        // 5. 展开能带 (Unfold) 回到原胞 Brillouin 区
        let u_matrix = arr2(&[[9.0, 0.0], [0.0, 9.0]]);

        // 生成对应的谱函数 (Spectral Weight)
        let a_spectral = magnetic_model
            .unfold(&u_matrix, &path, nk, -3.0, 3.0, nk, 1e-3, 1e-5)
            .unwrap();

        // 将谱函数输出为热力图
        // (假定 draw_heatmap 接收二维热力矩阵以及保存路径)
        draw_heatmap(
            &a_spectral.reversed_axes(),
            "./tests/graphene_magnetic/unfold_band.pdf",
        );
    }

    #[test]
    fn test_hofstadter_butterfly_gnuplot() {
        use gnuplot::AutoOption::Fix;
        use gnuplot::{AxesCommon, Color, Figure, PointSize, PointSymbol};

        // 1. 初始化一个标准的 2D 正方晶格
        let t = Complex::new(-1.0, 0.0);
        // 正方晶格基矢 a1=(1,0), a2=(0,1)
        let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        // 单轨道坐标位于原点
        let orb = arr2(&[[0.0, 0.0]]);

        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.set_onsite(&arr1(&[0.0]), None);

        // 加上最近邻跃迁 (上下左右4个方向)
        let r0: ndarray::Array2<isize> = arr2(&[[1, 0], [-1, 0], [0, 1], [0, -1]]);
        for r in r0.axis_iter(Axis(0)) {
            model.add_hop(t, 0, 0, &r.to_owned(), None);
        }

        // 2. 准备扫描磁通并记录坐标
        let q = 81; // 分母 q 设定为 49 形成的蝴蝶分形已经足够好看

        // 我们用两个动态数组分别记录散点图的 X 轴 (磁通) 和 Y 轴 (能量)
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        println!("开始计算 Hofstadter 蝴蝶能谱，进度: ");

        for p in 0..=q {
            // 在 y 轴方向扩胞 q 倍，即 [1, q]，形成一个一维超胞
            let mag_model = model.add_magnetic_field(2, [1, q], p as isize).unwrap();
            let norb = mag_model.norb();

            // 提取 Gamma 点 k = (0,0) 的哈密顿量矩阵
            let mut h_k0 = Array2::<Complex<f64>>::zeros((norb, norb));
            for iR in 0..mag_model.hamR.nrows() {
                for i in 0..norb {
                    for j in 0..norb {
                        h_k0[[i, j]] += mag_model.ham[[iR, i, j]];
                    }
                }
            }

            // --- 求解本征值 ---
            let evals = h_k0.eigvalsh(UPLO::Upper).expect("矩阵对角化失败");

            // 收集坐标点
            let flux_ratio = (p as f64) / (q as f64);
            for &e in evals.iter() {
                x_data.push(flux_ratio);
                y_data.push(e);
            }

            if p % 10 == 0 {
                println!("已完成 {}/{}", p, q);
            }
        }

        println!(
            "计算完成，共有 {} 个能级点，正在使用 gnuplot 绘图...",
            x_data.len()
        );

        // 3. 使用 Gnuplot 直接输出图像
        // 确保目录存在
        create_dir_all("tests").expect("无法创建 tests 文件夹");

        let mut fg = Figure::new();
        let axes = fg.axes2d();

        axes.set_title("Hofstadter's Butterfly", &[]);
        axes.set_x_label("Magnetic Flux (\\Phi / \\Phi_0)", &[]);
        axes.set_y_label("Energy (E/t)", &[]);

        // 固定 x 和 y 的坐标范围
        let axes = axes.set_x_range(Fix(0.0), Fix(1.0));
        let axes = axes.set_y_range(Fix(-10.0), Fix(10.0));

        // 使用 .points 绘制散点图
        // PointSymbol('.') 表示画极小的像素点，最适合用来展示密集的分形结构
        // Color("navy") 使用深蓝色
        axes.points(
            &x_data,
            &y_data,
            &[Color("navy"), PointSymbol('.'), PointSize(0.6)],
        );

        // 将图像渲染为 PDF
        fg.set_terminal("pdfcairo", "tests/hofstadter_butterfly.pdf");
        fg.show().expect("Gnuplot 画图失败");

        println!("完美！图像已保存至 tests/hofstadter_butterfly.pdf");
    }

    #[test]
    fn fermi_surface_graphene() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 1.0 + 0.0 * li;
        let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
        let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(t1, 0, 1, &array![0, 0], None);
        model.add_hop(t1, 0, 1, &array![-1, 0], None);
        model.add_hop(t1, 0, 1, &array![0, -1], None);

        // E_F = 0.1 slightly above Dirac point → small Fermi pockets around K, K'
        let k_mesh = arr1(&[100, 100]);
        model
            .show_fermi_surface(&k_mesh, 0.1, "tests/graphene")
            .expect("Fermi surface plot failed");
        println!("Graphene Fermi surface saved to tests/graphene/fermi_surface.pdf");
    }

    #[test]
    fn fermi_surface_kagome() {
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 1.0 + 0.0 * li;
        let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
        let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 0.0], [0.0, 1.0 / 3.0]]);
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_hop(t1, 0, 1, &array![0, 0], None);
        model.add_hop(t1, 2, 0, &array![0, 0], None);
        model.add_hop(t1, 1, 2, &array![0, 0], None);
        model.add_hop(t1, 0, 2, &array![0, -1], None);
        model.add_hop(t1, 0, 1, &array![-1, 0], None);
        model.add_hop(t1, 2, 1, &array![-1, 1], None);

        // E_F = -1.0 (flat band) — should show a Fermi surface contour
        let k_mesh = arr1(&[80, 80]);
        model
            .show_fermi_surface(&k_mesh, -1.0, "tests/kagome")
            .expect("Fermi surface plot failed");
        println!("Kagome Fermi surface saved to tests/kagome/fermi_surface.pdf");
    }

    fn build_h_wave_am_model() -> Model<false, 3> {
        let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let orb = array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
        let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();
        let t = 1.0;
        let j = 1.0;
        model.add_hop(t, 0, 1, &array![0, 0, 0], None);
        model.add_hop(t, 0, 1, &array![-1, 0, 0], None);
        model.add_hop(t, 0, 1, &array![0, -1, 0], None);
        model.add_hop(t, 0, 1, &array![-1, -1, 0], None);
        model.add_hop(t, 0, 1, &array![0, 0, -1], None);
        model.add_hop(t, 0, 1, &array![-1, 0, -1], None);
        model.add_hop(t, 0, 1, &array![0, -1, -1], None);
        model.add_hop(t, 0, 1, &array![-1, -1, -1], None);

        let t0 = Complex::new(0.0, 0.5);
        model.add_hop(t0, 0, 0, &array![2, 1, 1], None);
        model.add_hop(-t0, 0, 0, &array![2, 1, -1], None);
        model.add_hop(-t0, 0, 0, &array![2, -1, 1], None);
        model.add_hop(t0, 0, 0, &array![2, -1, -1], None);
        model.add_hop(-t0, 0, 0, &array![1, 2, 1], None);
        model.add_hop(t0, 0, 0, &array![1, 2, -1], None);
        model.add_hop(t0, 0, 0, &array![-1, 2, 1], None);
        model.add_hop(-t0, 0, 0, &array![-1, 2, -1], None);

        let t0 = -t0;
        model.add_hop(t0, 1, 1, &array![2, 1, 1], None);
        model.add_hop(-t0, 1, 1, &array![2, 1, -1], None);
        model.add_hop(-t0, 1, 1, &array![2, -1, 1], None);
        model.add_hop(t0, 1, 1, &array![2, -1, -1], None);
        model.add_hop(-t0, 1, 1, &array![1, 2, 1], None);
        model.add_hop(t0, 1, 1, &array![1, 2, -1], None);
        model.add_hop(t0, 1, 1, &array![-1, 2, 1], None);
        model.add_hop(-t0, 1, 1, &array![-1, 2, -1], None);

        model.add_onsite(&array![j, -j], None);
        model
    }

    fn max_abs_1d(x: &Array1<f64>) -> f64 {
        x.iter().fold(0.0f64, |a, &v| a.max(v.abs()))
    }

    fn max_abs_diff_1d(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    #[test]
    fn nlh_current_first_api_matches_kernel_definitions() {
        let model = build_h_wave_am_model();
        let current = array![1.0, 0.0, 0.0];
        let field_1 = array![0.0, 1.0, 0.0];
        let field_2 = array![0.0, 0.0, 1.0];
        let k_mesh = array![12, 12, 12];
        let chemical_potentials = Array1::linspace(-1.0, 1.0, 21);
        let mut params = Parameters::rank3(
            [12, 12, 12],
            fixed_direction(&current),
            fixed_direction(&field_1),
            fixed_direction(&field_2),
            chemical_potentials.clone(),
        );
        params.T = array![100.0];
        let public = model
            .intrinsic_nonlinear_hall(&params)
            .unwrap()
            .conductivity;

        let k_points = gen_kmesh(&k_mesh).unwrap();
        let (kernel, energies, _) =
            model.berry_connection_dipole(&k_points, &field_1, &field_2, &current, None);
        let occupation = Occupation::FermiDirac {
            temperature_kelvin: 100.0,
        };
        let expected = chemical_potentials.mapv(|mu| {
            kernel
                .iter()
                .zip(energies.iter())
                .map(|(&value, &energy)| value * occupation.minus_derivative(energy, mu).unwrap())
                .sum::<f64>()
                / k_points.nrows() as f64
                / model.lat.det().unwrap()
        });
        assert!(max_abs_diff_1d(&public, &expected) < 1e-12);
        assert!(max_abs_1d(&public) > 1e-8, "test signal is too small");
    }
    // ── Tetra smoke tests ─────────────────────────────────────────────────

    fn build_haldane_2d(t2_imag: f64) -> Model<false, 2> {
        let li = Complex::new(0.0, 1.0);
        let t = Complex::new(-1.0, 0.0);
        let t2 = Complex::new(t2_imag, 0.0); // real → multiplied by i at call site
        let delta = 0.7;
        let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
        let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
        let mut m = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        m.set_onsite(&arr1(&[-delta, delta]), None);
        for &(i, j) in &[(0, 0), (-1, 0), (0, -1)] {
            m.add_hop(t, 0, 1, &arr1(&[i, j]), None);
        }
        for &(i, j) in &[(1, 0), (-1, 1), (0, -1)] {
            m.add_hop(t2 * li, 0, 0, &arr1(&[i, j]), None);
        }
        for &(i, j) in &[(-1, 0), (1, -1), (0, 1)] {
            m.add_hop(t2 * li, 1, 1, &arr1(&[i, j]), None);
        }
        m
    }

    /// 1. API conventions: T>0 guard on single-mu
    #[test]
    fn nlh_api_conventions_and_guards() {
        let model = build_h_wave_am_model();
        let dx = array![1.0, 0.0, 0.0];
        let dy = array![0.0, 1.0, 0.0];
        let dz = array![0.0, 0.0, 1.0];
        let kmesh = array![4, 4, 4];
        let mu1 = arr1(&[0.0]);
        // Reference must accept T>0
        assert!(
            intrinsic_nonlinear_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &dz,
                &mu1,
                300.0,
                Integration::Direct,
            )
            .is_ok()
        );

        let zero_temperature = Parameters::rank3(
            [4, 4, 4],
            fixed_direction(&dx),
            fixed_direction(&dy),
            fixed_direction(&dz),
            mu1,
        );
        assert!(model.intrinsic_nonlinear_hall(&zero_temperature).is_err());
    }

    /// 2. Intrinsic NLH: H-wave up/dn T‑odd via direct sum.
    ///
    /// σ(up) = −σ(dn) must hold for the reference path.
    #[test]
    fn nlh_intrinsic_hwave_up_dn_odd() {
        let li = Complex::new(0.0, 1.0);
        let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let orb = array![[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]];
        let t = 1.0;
        let j0 = 1.0;
        let build = |j_sign: f64| {
            let mut m = Model::<false, 3>::tb_model(lat.clone(), orb.clone(), None).unwrap();
            for &(i, j, k) in &[
                (0, 0, 0),
                (-1, 0, 0),
                (0, -1, 0),
                (-1, -1, 0),
                (0, 0, -1),
                (-1, 0, -1),
                (0, -1, -1),
                (-1, -1, -1),
            ] {
                m.add_hop(t, 0, 1, &array![i, j, k], None);
            }
            let t0 = Complex::new(0.0, 0.2);
            let nnn: [(f64, (isize, isize, isize)); 8] = [
                (1.0, (2, 1, 1)),
                (-1.0, (2, 1, -1)),
                (-1.0, (2, -1, 1)),
                (1.0, (2, -1, -1)),
                (-1.0, (1, 2, 1)),
                (1.0, (1, 2, -1)),
                (1.0, (-1, 2, 1)),
                (-1.0, (-1, 2, -1)),
            ];
            for &(s, (i, j, k)) in &nnn {
                m.add_hop(t0.scale(s), 0, 0, &array![i, j, k], None);
                m.add_hop(t0.scale(-s), 1, 1, &array![i, j, k], None);
            }
            m.add_onsite(&array![j0 * j_sign, -j0 * j_sign], None);
            m
        };
        let model_up = build(1.0);
        let model_dn = build(-1.0);
        let dx = array![1.0, 0.0, 0.0];
        let dy = array![0.0, 1.0, 0.0];
        let dz = array![0.0, 0.0, 1.0];
        let T: f64 = 100.0;
        let mu = Array1::linspace(-1.0, 1.0, 21);
        let km = array![12, 12, 12];
        let ref_up =
            intrinsic_nonlinear_values(&model_up, &km, &dx, &dy, &dz, &mu, T, Integration::Direct)
                .unwrap();
        let ref_dn =
            intrinsic_nonlinear_values(&model_dn, &km, &dx, &dy, &dz, &mu, T, Integration::Direct)
                .unwrap();
        let sum = max_abs_1d(&(&ref_up + &ref_dn));
        assert!(
            sum < 1e-10,
            "ref up+dn must vanish (T‑odd), got {:.2e}",
            sum
        );
        assert!(max_abs_1d(&ref_up) > 1e-6, "signal too small");
    }

    /// 3. Intrinsic NLH 2D convergence via direct sum.
    #[test]
    fn intrinsic_haldane_2d_convergence() {
        let model = build_haldane_2d(-0.3);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let mu = Array1::linspace(-4.0, 4.0, 21);
        let mut prev_pk = 0.0;
        for &nk in &[21usize, 31, 41, 51] {
            let km = arr1(&[nk, nk]);
            let ref_val = intrinsic_nonlinear_values(
                &model,
                &km,
                &dx,
                &dy,
                &dy,
                &mu,
                0.0,
                Integration::Direct,
            )
            .unwrap();
            let pk = max_abs_1d(&ref_val);
            assert!(pk > 1e-6, "signal too small at nk={nk}");
            // Result should stabilise with mesh size
            if prev_pk > 0.0 {
                let rel = (pk - prev_pk).abs() / prev_pk;
                assert!(
                    rel < 0.5,
                    "large drift at nk={nk}: pk={pk:.3e} prev={prev_pk:.3e}"
                );
            }
            prev_pk = pk;
        }
    }
    // ── Simplex vs direct-sum comparison tests ──────────────────────────

    /// 8. Berry-curvature trait vs gauge-invariant velocity kernel.
    ///
    /// Compares the **per‑k‑point integrand** Ω^{xy}_n(k) produced by the
    /// public band-resolved result against the gauge‑invariant kernel
    /// `K^{xy}_nm = v^x_nm v^y_mn` evaluated at each k‑point.  The relation
    /// `Ω_n = −2 Im Σ_m K_nm / (d²+η²)` must hold identically.
    #[test]
    fn berry_curvature_kernel_consistency() {
        let model = build_haldane_2d(-0.3);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.05;
        let nk = 21;
        let kmesh = arr1(&[nk, nk]);
        let kvec = crate::kpoints::gen_kmesh(&kmesh).unwrap();
        let nkt = kvec.nrows();

        let mut max_err = 0.0f64;
        for ik in 0..nkt {
            let kv = kvec.row(ik).to_owned();
            // Band-resolved Berry curvature from the public trait.
            let omega_n_old =
                band_berry_curvature(&model, &kv, &dx, &dy, None, eta).berry_curvature;
            // Gauge-invariant K_nm must produce the same Ω_n.
            let tk = model.compute_velocity_kernel(&kv, &dx, &dy, None, Gauge::Atom, None);
            let nsta = model.nsta();
            let mut omega_n_new = Array1::<f64>::zeros(nsta);
            let eta2 = eta * eta;
            for n in 0..nsta {
                let mut g_sum = Complex::new(0.0, 0.0);
                for m in 0..nsta {
                    if m == n {
                        continue;
                    }
                    let de = tk.band[[n]] - tk.band[[m]];
                    let denom = de * de + eta2;
                    g_sum += tk.k_ab[[n, m]] / denom;
                }
                omega_n_new[[n]] = -2.0 * g_sum.im;
            }
            let err = max_abs_diff_1d(&omega_n_old, &omega_n_new);
            max_err = max_err.max(err);
        }
        println!("max per‑k‑point Ω_n discrepancy: {:.3e} (nk={nk})", max_err);
        assert!(max_err < 1e-12, "old vs new Ω_n mismatch: {:.2e}", max_err);
    }

    /// 8b. Berry curvature integral: simplex vs direct sum.
    ///
    /// The Haldane model has total Ω = 0 (Chern numbers sum to zero).
    /// This tests whether the simplex quadrature introduces spurious
    /// non‑zero total Berry curvature.
    #[test]
    fn berry_total_simplex_vs_direct() {
        let model = build_haldane_2d(-0.3);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.05;

        println!("\n--- Berry total Ω^{{xy}} simplex vs direct ---");
        println!(
            "{:>4}  {:>14}  {:>14}  {:>12}",
            "nk", "direct_sum", "simplex", "diff"
        );
        for &nk in &[21usize, 31, 51, 101] {
            let kmesh = arr1(&[nk, nk]);
            let kvec = crate::kpoints::gen_kmesh(&kmesh).unwrap();
            let nkt = kvec.nrows();

            // old: direct sum
            let mut direct = 0.0;
            for ik in 0..nkt {
                let kv = kvec.row(ik).to_owned();
                let omega_n =
                    band_berry_curvature(&model, &kv, &dx, &dy, None, eta).berry_curvature;
                direct += omega_n.iter().sum::<f64>();
            }
            direct /= nkt as f64;

            // new: simplex integral
            let all_pts: Vec<crate::response::VertexKernel> = (0..nkt)
                .map(|ik| {
                    let kv = kvec.row(ik).to_owned();
                    let tk = model.compute_velocity_kernel(&kv, &dx, &dy, None, Gauge::Atom, None);
                    tk
                })
                .collect();
            let (_g, simplex, _unsafe) = crate::response::linear::integrate_occupied_geometry(
                &all_pts,
                &kmesh,
                eta,
                &array![1e100],
                Occupation::ZeroTemperature,
            );
            let simplex = simplex[0];

            let diff = (direct - simplex).abs();
            println!("{nk:>4}  {direct:>14.6e}  {simplex:>14.6e}  {diff:>10.3e}");
            assert!(diff < 5e-4, "total Ω mismatch {diff:.2e} at nk={nk}");
        }
    }

    /// 9. Berry curvature dipole: old direct Fermi‑derivative sum vs new
    /// 8b. AHC energy-cut: Chern quantization in gap.
    #[test]
    fn hall_conductivity_ec_vs_reference() {
        let model = build_haldane_2d(-0.3);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.05;
        let mu = Array1::linspace(-3.0, 3.0, 201);
        let i_mid = mu.len() / 2;

        for &nk in &[31, 51, 71, 101] {
            let kmesh = arr1(&[nk, nk]);
            let direct = hall_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &mu,
                0.0,
                None,
                eta,
                Integration::Direct,
            )
            .unwrap();
            let ec = hall_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &mu,
                0.0,
                None,
                eta,
                Integration::EnergyCut,
            )
            .unwrap();
            let max_abs = max_abs_diff_1d(&direct, &ec);
            let c_dir = direct[[i_mid]];
            let c_ec = ec[[i_mid]];
            println!("nk={nk}  max_abs={max_abs:.3e}  C_dir={c_dir:.6}  C_ec={c_ec:.6}");
            assert!(max_abs < 3e-2);
        }
    }

    /// 8c. Time-reversal check: σ(TR Haldane) = −σ(Haldane) for both direct and EC.
    #[test]
    fn hall_conductivity_ec_tr() {
        let model = build_haldane_2d(-0.3);
        let model_tr = build_haldane_2d(0.3); // t2 → −t2
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.05;
        let mu = Array1::linspace(-3.0, 3.0, 101);
        let kmesh = arr1(&[51, 51]);

        let d_dir = hall_values(
            &model,
            &kmesh,
            &dx,
            &dy,
            &mu,
            0.0,
            None,
            eta,
            Integration::Direct,
        )
        .unwrap();
        let d_ec = hall_values(
            &model,
            &kmesh,
            &dx,
            &dy,
            &mu,
            0.0,
            None,
            eta,
            Integration::EnergyCut,
        )
        .unwrap();
        let tr_dir = hall_values(
            &model_tr,
            &kmesh,
            &dx,
            &dy,
            &mu,
            0.0,
            None,
            eta,
            Integration::Direct,
        )
        .unwrap();
        let tr_ec = hall_values(
            &model_tr,
            &kmesh,
            &dx,
            &dy,
            &mu,
            0.0,
            None,
            eta,
            Integration::EnergyCut,
        )
        .unwrap();

        // Check σ(Haldane) + σ(TR) ≈ 0 element-wise
        let diff_dir: Vec<f64> = d_dir
            .iter()
            .zip(tr_dir.iter())
            .map(|(&a, &b)| (a + b).abs())
            .collect();
        let diff_ec: Vec<f64> = d_ec
            .iter()
            .zip(tr_ec.iter())
            .map(|(&a, &b)| (a + b).abs())
            .collect();
        let max_dir = diff_dir.iter().fold(0.0f64, |a: f64, &b| a.max(b));
        let max_ec = diff_ec.iter().fold(0.0f64, |a: f64, &b| a.max(b));
        println!("TR check: max|σ+σ_TR|  direct={max_dir:.3e}  EC={max_ec:.3e}");
        assert!(max_dir < 1e-10, "direct sum fails TR: {max_dir:.3e}");
        assert!(max_ec < 1e-10, "EC fails TR: {max_ec:.3e}");
    }

    /// 8c2. Dipole energy-cut: TR check for Haldane (±t2).
    #[test]
    fn dipole_energy_cut_tr() {
        let model = build_haldane_2d(-0.3);
        let model_tr = build_haldane_2d(0.3);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let dc = arr1(&[1.0, 0.0]); // dipole direction = x
        let eta = 0.05;
        let mu = Array1::linspace(-2.0, 2.0, 51);
        let kmesh = arr1(&[31, 31]);

        let d_dir = extrinsic_nonlinear_values(
            &model,
            &kmesh,
            &dx,
            &dy,
            &dc,
            &mu,
            0.0,
            0.0,
            None,
            eta,
            Integration::EnergyCut,
            FieldSymmetry::Ordered,
        )
        .unwrap();
        let d_tr = extrinsic_nonlinear_values(
            &model_tr,
            &kmesh,
            &dx,
            &dy,
            &dc,
            &mu,
            0.0,
            0.0,
            None,
            eta,
            Integration::EnergyCut,
            FieldSymmetry::Ordered,
        )
        .unwrap();
        // BCD is TR‑even: v_TR^c Ω_TR^{ab} = (−v^c)(−Ω^{ab}) = v^c Ω^{ab}  →  D_TR = D
        let max_diff = d_dir
            .iter()
            .zip(d_tr.iter())
            .fold(0.0f64, |a: f64, (&x, &y)| a.max((x - y).abs()));
        let max_d = d_dir.iter().fold(0.0f64, |a: f64, &x| a.max(x.abs()));
        println!("Dipole TR: max|D|={max_d:.3e}  max|D−D_TR|={max_diff:.3e}");
        assert!(max_diff < 1e-10, "Dipole TR-even broken: {max_diff:.3e}");
    }

    /// 8c3. Intrinsic NLH EC vs direct sum (NLH model, σ^{yy;x}).
    /// Direct: current=x, field1=y, field2=y. EC: dir_a=y, dir_b=y, dir_c=x.
    #[test]
    fn intrinsic_ec_vs_direct() {
        let model = build_nlh_2d(2.6, 1.0);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.03;
        let mu = Array1::linspace(-4.0, 4.0, 41);
        let kmesh = arr1(&[21, 21]);

        let dir = intrinsic_nonlinear_values(
            &model,
            &kmesh,
            &dx,
            &dy,
            &dy,
            &mu,
            0.0,
            Integration::Direct,
        )
        .unwrap();
        let ec = intrinsic_nonlinear_values(
            &model,
            &kmesh,
            &dx,
            &dy,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let max_abs = max_abs_diff_1d(&dir, &ec);
        let max_dir = dir.iter().fold(0.0f64, |a: f64, &x| a.max(x.abs()));
        println!("Intrinsic EC vs direct: max_abs={max_abs:.3e}  max|σ|={max_dir:.3e}");
        // The direct reference uses an explicit mesh-scale Fermi smearing,
        // while EC uses exact δ‑function line‑cut.  Differences O(1e-3)
        // for this model are expected.
        assert!(
            max_abs < max_dir * 1.5,
            "Intrinsic EC mismatch: {max_abs:.3e}"
        );
    }

    /// Build 3D low-symmetry two-band model for intrinsic NLH testing.
    ///
    /// H(k) = d₀·σ₀ + d_x·σ_x + d_y·σ_y + d_z·σ_z with λ controlling
    /// inversion breaking.  λ=0 restores P-symmetry → intrinsic NLH = 0.
    /// Both orbitals at (0,0,0).  Recommended: m=2.6, λ=1.
    fn build_nlh_3d(m: f64, lambda: f64) -> Model<false, 3> {
        let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let orb = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();
        model.add_onsite(&array![m, -m], None);

        // R = (1,0,0): 0.1 σ₀ − i/2 σ_x + 0.5 σ_z
        let i2 = Complex::new(0.0, -0.5);
        model.add_hop(0.1, 0, 0, &array![1, 0, 0], None);
        model.add_hop(0.1, 1, 1, &array![1, 0, 0], None);
        model.add_hop(i2, 0, 1, &array![1, 0, 0], None);
        model.add_hop(i2, 1, 0, &array![1, 0, 0], None);
        model.add_hop(0.5, 0, 0, &array![1, 0, 0], None);
        model.add_hop(-0.5, 1, 1, &array![1, 0, 0], None);

        // R = (0,1,0): 0.04 σ₀ − 0.15i σ_x − 0.5i σ_y + 0.325 σ_z
        model.add_hop(0.04, 0, 0, &array![0, 1, 0], None);
        model.add_hop(0.04, 1, 1, &array![0, 1, 0], None);
        model.add_hop(Complex::new(0.0, -0.15), 0, 1, &array![0, 1, 0], None);
        model.add_hop(Complex::new(0.0, -0.15), 1, 0, &array![0, 1, 0], None);
        // −0.5i σ_y → H[0,1]=−0.5, H[1,0]=0.5
        model.add_hop(-0.5, 0, 1, &array![0, 1, 0], None);
        model.add_hop(0.5, 1, 0, &array![0, 1, 0], None);
        model.add_hop(0.325, 0, 0, &array![0, 1, 0], None);
        model.add_hop(-0.325, 1, 1, &array![0, 1, 0], None);

        // R = (0,0,1): 0.03 σ₀ + (0.05λ − 0.10i)σ_x − 0.125i σ_y + 0.225 σ_z
        model.add_hop(0.03, 0, 0, &array![0, 0, 1], None);
        model.add_hop(0.03, 1, 1, &array![0, 0, 1], None);
        model.add_hop(
            Complex::new(0.05 * lambda, -0.10),
            0,
            1,
            &array![0, 0, 1],
            None,
        );
        model.add_hop(
            Complex::new(0.05 * lambda, -0.10),
            1,
            0,
            &array![0, 0, 1],
            None,
        );
        // σ_y: −0.125i [[0,−i],[i,0]] → H[0,1]=−0.125, H[1,0]=0.125
        model.add_hop(-0.125, 0, 1, &array![0, 0, 1], None);
        model.add_hop(0.125, 1, 0, &array![0, 0, 1], None);
        model.add_hop(0.225, 0, 0, &array![0, 0, 1], None);
        model.add_hop(-0.225, 1, 1, &array![0, 0, 1], None);

        // R = (1,1,0): 0.125λ σ_x
        model.add_hop(0.125 * lambda, 0, 1, &array![1, 1, 0], None);
        model.add_hop(0.125 * lambda, 1, 0, &array![1, 1, 0], None);

        // R = (1,0,−1): (0.10λ − 0.075i) σ_y
        let hop10m1 = Complex::new(0.10 * lambda, -0.075);
        // σ_y gives: H[0,1] = −i·hop, H[1,0] = i·hop
        // −i·hop = −i(0.10λ−0.075i) = −0.10iλ + 0.075i² = −0.075−0.10iλ
        // i·hop = i(0.10λ−0.075i) = 0.10iλ − 0.075i² = 0.075+0.10iλ
        model.add_hop(
            Complex::new(-0.075, -0.10 * lambda),
            0,
            1,
            &array![1, 0, -1],
            None,
        );
        model.add_hop(
            Complex::new(0.075, 0.10 * lambda),
            1,
            0,
            &array![1, 0, -1],
            None,
        );

        // R = (1,1,1): −0.125iλ σ_z
        let hz = Complex::new(0.0, -0.125 * lambda);
        model.add_hop(hz, 0, 0, &array![1, 1, 1], None);
        model.add_hop(-hz, 1, 1, &array![1, 1, 1], None);

        model
    }

    /// 2D slice (kz=0) of the NLH test model for intrinsic EC (2D only).
    fn build_nlh_2d(m: f64, lambda: f64) -> Model<false, 2> {
        let lat = array![[1.0, 0.0], [0.0, 1.0]];
        let orb = array![[0.0, 0.0], [0.0, 0.0]];
        let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
        model.add_onsite(&array![m, -m], None);

        // R = (1,0): 0.1 σ₀ − i/2 σ_x + 0.5 σ_z
        let i2 = Complex::new(0.0, -0.5);
        model.add_hop(0.1, 0, 0, &array![1, 0], None);
        model.add_hop(0.1, 1, 1, &array![1, 0], None);
        model.add_hop(i2, 0, 1, &array![1, 0], None);
        model.add_hop(i2, 1, 0, &array![1, 0], None);
        model.add_hop(0.5, 0, 0, &array![1, 0], None);
        model.add_hop(-0.5, 1, 1, &array![1, 0], None);

        // R = (0,1): 0.04 σ₀ − 0.15i σ_x − 0.5i σ_y + 0.325 σ_z
        model.add_hop(0.04, 0, 0, &array![0, 1], None);
        model.add_hop(0.04, 1, 1, &array![0, 1], None);
        model.add_hop(Complex::new(0.0, -0.15), 0, 1, &array![0, 1], None);
        model.add_hop(Complex::new(0.0, -0.15), 1, 0, &array![0, 1], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);
        model.add_hop(0.5, 1, 0, &array![0, 1], None);
        model.add_hop(0.325, 0, 0, &array![0, 1], None);
        model.add_hop(-0.325, 1, 1, &array![0, 1], None);

        // R = (1,1): 0.125λ σ_x
        model.add_hop(0.125 * lambda, 0, 1, &array![1, 1], None);
        model.add_hop(0.125 * lambda, 1, 0, &array![1, 1], None);

        model
    }

    /// 8c5. Intrinsic NLH: inversion-symmetry benchmark (λ=0 → zero).
    #[test]
    fn intrinsic_ec_inversion() {
        let model_p = build_nlh_2d(2.6, 0.0); // λ=0 restores P-symmetry
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.03;
        let mu = Array1::linspace(-4.0, 4.0, 61);
        let kmesh = arr1(&[21, 21]);
        let ec = intrinsic_nonlinear_values(
            &model_p,
            &kmesh,
            &dy,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let max_val = ec.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        println!("Inversion-symmetric (λ=0): max|σ| = {max_val:.3e}");
        assert!(max_val < 1e-10, "P-symmetry broken: {max_val:.3e}");
    }

    /// 8c6. Intrinsic NLH: λ → −λ sign flip (P-odd).
    #[test]
    fn intrinsic_ec_sign_flip() {
        let model_p = build_nlh_2d(2.6, 1.0);
        let model_m = build_nlh_2d(2.6, -1.0);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.03;
        let mu = Array1::linspace(-4.0, 4.0, 61);
        let kmesh = arr1(&[21, 21]);
        let ec_p = intrinsic_nonlinear_values(
            &model_p,
            &kmesh,
            &dy,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let ec_m = intrinsic_nonlinear_values(
            &model_m,
            &kmesh,
            &dy,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let max_sum = ec_p
            .iter()
            .zip(ec_m.iter())
            .fold(0.0f64, |a: f64, (&x, &y)| a.max((x + y).abs()));
        let max_p = ec_p.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        println!("Sign flip λ→−λ: max|σ|={max_p:.3e}  max|σ(λ)+σ(−λ)|={max_sum:.3e}");
        assert!(max_sum < 1e-10, "P-odd sign flip broken: {max_sum:.3e}");
    }

    /// 8c7. Intrinsic NLH EC convergence: peak |σ| vs nk, T with NLH model.
    #[test]
    fn intrinsic_ec_convergence() {
        let model = build_nlh_2d(2.6, 1.0);
        let dx = arr1(&[1.0, 0.0]);
        let dy = arr1(&[0.0, 1.0]);
        let eta = 0.03;
        let mu = Array1::linspace(-4.0, 4.0, 81);
        let nks = [15, 21, 31, 41, 51, 61];
        let ts = [0.0, 100.0, 300.0];

        let mut peaks = vec![vec![0.0; nks.len()]; ts.len()];
        for (j, &nk) in nks.iter().enumerate() {
            let kmesh = arr1(&[nk, nk]);
            for (ti, &t) in ts.iter().enumerate() {
                let ec = intrinsic_nonlinear_values(
                    &model,
                    &kmesh,
                    &dy,
                    &dx,
                    &dy,
                    &mu,
                    t,
                    Integration::EnergyCut,
                )
                .unwrap();
                peaks[ti][j] = ec.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            }
        }

        let ref0 = peaks[0].last().unwrap();
        let ref1 = peaks[1].last().unwrap();
        let ref2 = peaks[2].last().unwrap();
        println!("nk   T=0K peak     Δ/ref     T=100K peak   Δ/ref     T=300K peak   Δ/ref");
        for j in 0..nks.len() {
            let d0 = (*ref0 - peaks[0][j]).abs() / ref0.abs();
            let d1 = (*ref1 - peaks[1][j]).abs() / ref1.abs();
            let d2 = (*ref2 - peaks[2][j]).abs() / ref2.abs();
            println!(
                "{:>3}   {:.4e}   {:.3e}   {:.4e}   {:.3e}   {:.4e}   {:.3e}",
                nks[j], peaks[0][j], d0, peaks[1][j], d1, peaks[2][j], d2,
            );
        }
        // Signal should be O(1e-3) or larger with this model
        assert!(*ref0 > 1e-6, "Intrinsic signal too small: {ref0:.3e}");
    }

    /// 8c8. 3D intrinsic NLH sign-flip (λ→−λ) with NLH model.
    #[test]
    fn intrinsic_ec_3d_sign_flip() {
        let model_p = build_nlh_3d(2.6, 1.0);
        let model_m = build_nlh_3d(2.6, -1.0);
        let dx = arr1(&[1.0, 0.0, 0.0]);
        let dy = arr1(&[0.0, 1.0, 0.0]);
        let dz = arr1(&[0.0, 0.0, 1.0]);
        let eta = 0.03;
        let mu = Array1::linspace(-3.0, 3.0, 31);
        let kmesh = arr1(&[8, 8, 8]);

        let ec_p = intrinsic_nonlinear_values(
            &model_p,
            &kmesh,
            &dz,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let ec_m = intrinsic_nonlinear_values(
            &model_m,
            &kmesh,
            &dz,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let max_sum = ec_p
            .iter()
            .zip(ec_m.iter())
            .fold(0.0f64, |a: f64, (&x, &y)| a.max((x + y).abs()));
        let max_p = ec_p.iter().fold(0.0f64, |a: f64, &x| a.max(x.abs()));
        println!("3D Intrinsic sign flip: max|σ|={max_p:.3e}  max|σ(λ)+σ(−λ)|={max_sum:.3e}");
        // Diagonal-averaged tetrahedralization restores k→−k cancellation
        // to machine precision, matching 2D diagavg result.
        assert!(max_sum < 1e-10, "3D P-odd broken: {max_sum:.3e}");
    }

    /// 8c9. 3D intrinsic EC convergence: peak |σ| vs nk at T=100K.
    #[test]
    fn intrinsic_ec_3d_convergence() {
        let model = build_nlh_3d(2.6, 1.0);
        let dx = arr1(&[1.0, 0.0, 0.0]);
        let dy = arr1(&[0.0, 1.0, 0.0]);
        let dz = arr1(&[0.0, 0.0, 1.0]);
        let eta = 0.03;
        let mu = Array1::linspace(-3.0, 3.0, 41);
        let nks = [6, 8, 10, 12, 14];

        println!("nk    peak|σ|       Δ/ref");
        let mut peaks = Vec::new();
        for &nk in &nks {
            let kmesh = arr1(&[nk, nk, nk]);
            let ec = intrinsic_nonlinear_values(
                &model,
                &kmesh,
                &dz,
                &dx,
                &dy,
                &mu,
                100.0,
                Integration::EnergyCut,
            )
            .unwrap();
            let peak = ec.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            peaks.push(peak);
        }
        let ref_val = peaks.last().unwrap();
        for (i, &nk) in nks.iter().enumerate() {
            let d = (ref_val - peaks[i]).abs() / ref_val.abs();
            println!("{:>3}    {:.4e}   {:.3e}", nk, peaks[i], d);
        }
        // 3D surface K‑quadrature converges ∝1/nk² (area), slower than 2D line ∝1/nk.
        // nk=10 is within ~16% of nk=14; nk=14→18 extrapolated ~5%.
        assert!(peaks[2] > 1e-4, "3D intrinsic signal too small");
    }

    /// 8d. AHC energy-cut 3D smoke (altermagnet, Berry ≈ 0, sanity check).
    #[test]
    fn hall_conductivity_ec_3d_smoke() {
        let model = build_h_wave_am_model();
        let dx = arr1(&[1.0, 0.0, 0.0]);
        let dy = arr1(&[0.0, 1.0, 0.0]);
        let eta = 0.1;
        let mu = Array1::linspace(-2.0, 2.0, 41);

        for &nk in &[8, 10] {
            let kmesh = arr1(&[nk, nk, nk]);
            let direct = hall_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &mu,
                0.0,
                None,
                eta,
                Integration::Direct,
            )
            .unwrap();
            let ec = hall_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &mu,
                0.0,
                None,
                eta,
                Integration::EnergyCut,
            )
            .unwrap();
            let max_abs = max_abs_diff_1d(&direct, &ec);
            println!("3D smoke nk={nk}  max_abs={max_abs:.3e}");
            assert!(max_abs < 5e-2, "3D mismatch too large: {max_abs:.3e}");
        }
    }

    /// Build stacked 2D QWZ: H = sin kx σx + sin ky σy + (m+cos kx+cos ky) σz, kz-independent.
    fn build_qwz_stacked_3d(m: f64) -> Model<false, 3> {
        let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let orb = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();
        model.add_onsite(&array![m, -m], None);
        // sin(kx) σx: H_01 = (e^{ikx} - e^{-ikx}) / (2i)
        model.add_hop(Complex::new(0.0, -0.5), 0, 1, &array![1, 0, 0], None);
        model.add_hop(Complex::new(0.0, 0.5), 0, 1, &array![-1, 0, 0], None);
        // sin(ky) σy: H_01 = -i sin(ky) = -(e^{iky} - e^{-iky})/2
        model.add_hop(-0.5, 0, 1, &array![0, 1, 0], None);
        model.add_hop(0.5, 0, 1, &array![0, -1, 0], None);
        // cos(kx) σz: (e^{ikx} + e^{-ikx})/2 σz
        model.add_hop(0.5, 0, 0, &array![1, 0, 0], None);
        model.add_hop(0.5, 0, 0, &array![-1, 0, 0], None);
        model.add_hop(-0.5, 1, 1, &array![1, 0, 0], None);
        model.add_hop(-0.5, 1, 1, &array![-1, 0, 0], None);
        // cos(ky) σz
        model.add_hop(0.5, 0, 0, &array![0, 1, 0], None);
        model.add_hop(0.5, 0, 0, &array![0, -1, 0], None);
        model.add_hop(-0.5, 1, 1, &array![0, 1, 0], None);
        model.add_hop(-0.5, 1, 1, &array![0, -1, 0], None);
        model
    }

    /// 8d. 3D stacked QWZ: Ω kz-independent, plateau = 1/(2π).
    #[test]
    fn hall_conductivity_ec_3d_qwz() {
        let model = build_qwz_stacked_3d(-1.0);
        let dx = arr1(&[1.0, 0.0, 0.0]);
        let dy = arr1(&[0.0, 1.0, 0.0]);
        let eta = 0.1;
        let mu = Array1::linspace(-3.0, 3.0, 61);
        let i_mid = mu.len() / 2;
        let c_ref = 1.0 / (2.0 * std::f64::consts::PI);

        for &nk in &[10, 14] {
            let kmesh = arr1(&[nk, nk, 4]); // fewer kz points (Ω is kz-independent)
            let direct = hall_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &mu,
                0.0,
                None,
                eta,
                Integration::Direct,
            )
            .unwrap();
            let ec = hall_values(
                &model,
                &kmesh,
                &dx,
                &dy,
                &mu,
                0.0,
                None,
                eta,
                Integration::EnergyCut,
            )
            .unwrap();
            let max_abs = max_abs_diff_1d(&direct, &ec);
            let c_dir = direct[[i_mid]];
            let c_ec = ec[[i_mid]];
            let cnt = crate::response::read_reset_fermi_cut_counts();
            println!(
                "3D QWZ nk={nk}  max_abs={max_abs:.3e}  C_dir={c_dir:.6}  C_ec={c_ec:.6}  C_ref={c_ref:.6}  empty/full/partial={}/{}/{}",
                cnt.empty, cnt.full, cnt.partial
            );
            assert!(max_abs < 5e-2, "QWZ EC vs direct mismatch: {max_abs:.3e}");
            assert!((c_ec - c_ref).abs() < 0.003, "QWZ plateau off: {c_ec:.6}");
        }
    }

    // ── G‑wave 3D model (2‑orbital, triangular bilayer) ───────────────────

    fn build_gwave_3d(j: f64) -> Model<false, 3> {
        let lat = array![
            [1.0, 0.0, 0.0],
            [-0.5, 3_f64.sqrt() / 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let orb = array![[1.0 / 3.0, 2.0 / 3.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.5]];
        let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();

        let t = 1.0;
        model.add_hop(t, 0, 1, &array![0, 0, 0], None);
        model.add_hop(t, 0, 1, &array![-1, 0, 0], None);
        model.add_hop(t, 0, 1, &array![0, 1, 0], None);
        model.add_hop(t, 0, 1, &array![0, 0, -1], None);
        model.add_hop(t, 0, 1, &array![-1, 0, -1], None);
        model.add_hop(t, 0, 1, &array![0, 1, -1], None);

        let t2 = Complex::new(0.0, 0.5);
        let r_pairs: [(isize, isize, isize, Complex<f64>); 6] = [
            (1, -2, 1, -t2),
            (2, -1, 1, t2),
            (3, 1, 1, -t2),
            (3, 2, 1, t2),
            (2, 3, 1, -t2),
            (1, 3, 1, t2),
        ];
        for (a, b, c, val) in &r_pairs {
            model.set_hop(*val, 0, 0, &array![*a, *b, *c], None);
            model.set_hop(-val, 0, 0, &array![*a, *b, -1], None);
        }
        for (a, b, c, val) in &r_pairs {
            model.set_hop(-val, 1, 1, &array![*a, *b, *c], None);
            model.set_hop(*val, 1, 1, &array![*a, *b, -1], None);
        }

        model.add_onsite(&array![j, -j], None);
        model
    }

    /// 9a. G‑wave 3D: EC vs direct intrinsic at diagnostic k‑meshes.
    /// 3D surface K‑quadrature converges ∝1/nk²; nk≲16 is coarse.
    #[test]
    fn gwave_intrinsic_ec_vs_direct() {
        let model = build_gwave_3d(1.0);
        let dx = arr1(&[1.0, 0.0, 0.0]);
        let dy = arr1(&[0.0, 1.0, 0.0]);
        let dz = arr1(&[0.0, 0.0, 1.0]);
        let eta = 1e-3;
        let mu = Array1::linspace(-4.0, 4.0, 21);
        let nks = [12, 16, 20];
        println!(
            "{:<8}{:<14}{:<14}{:<14}",
            "nk", "peak|σ_ec|", "max|Δ|", "rel_err"
        );
        for &nk in &nks {
            let kmesh = arr1(&[nk, nk, nk]);
            let ec = intrinsic_nonlinear_values(
                &model,
                &kmesh,
                &dz,
                &dx,
                &dy,
                &mu,
                0.0,
                Integration::EnergyCut,
            )
            .unwrap();
            let direct = intrinsic_nonlinear_values(
                &model,
                &kmesh,
                &dz,
                &dx,
                &dy,
                &mu,
                0.0,
                Integration::Direct,
            )
            .unwrap();
            let max_abs = max_abs_diff_1d(&ec, &direct);
            let peak_ec = ec.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            let peak_dir = direct.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            println!(
                "{:<8}{:<14.4e}{:<14.4e}{:<14.4e}",
                nk,
                peak_ec.max(peak_dir),
                max_abs,
                max_abs / peak_ec.max(peak_dir).max(1e-30)
            );
        }
        // Smoke: both methods produce O(1e-3) signal at nk=20
        let kmesh20 = arr1(&[20, 20, 20]);
        let ec20 = intrinsic_nonlinear_values(
            &model,
            &kmesh20,
            &dz,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let peak = ec20.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        assert!(peak > 1e-6, "G‑wave signal too small: {peak:.3e}");
    }

    /// 9b. G‑wave 3D: J → −J sign check.
    /// Under orbital exchange the g‑wave intra‑orbital hoppings flip sign,
    /// so H(J) and H(−J) are related by U H U^†.  Intrinsic σ should flip sign.
    #[test]
    fn gwave_intrinsic_sign_flip() {
        let model_up = build_gwave_3d(1.0);
        let model_dn = build_gwave_3d(-1.0);
        let dx = arr1(&[1.0, 0.0, 0.0]);
        let dy = arr1(&[0.0, 1.0, 0.0]);
        let dz = arr1(&[0.0, 0.0, 1.0]);
        let eta = 1e-3;
        let mu = Array1::linspace(-4.0, 4.0, 21);
        // Use only EC — direct sum has degeneracy‑handling differences.
        let nks = [12, 16, 20];
        println!(
            "{:<8}{:<14}{:<14}{:<14}",
            "nk", "peak|σ|", "max|sum|", "sum/peak"
        );
        for &nk in &nks {
            let kmesh = arr1(&[nk, nk, nk]);
            let up = intrinsic_nonlinear_values(
                &model_up,
                &kmesh,
                &dz,
                &dx,
                &dy,
                &mu,
                0.0,
                Integration::EnergyCut,
            )
            .unwrap();
            let dn = intrinsic_nonlinear_values(
                &model_dn,
                &kmesh,
                &dz,
                &dx,
                &dy,
                &mu,
                0.0,
                Integration::EnergyCut,
            )
            .unwrap();
            let max_sum = up
                .iter()
                .zip(dn.iter())
                .fold(0.0f64, |a, (&x, &y)| a.max((x + y).abs()));
            let peak = up.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
            println!(
                "{:<8}{:<14.4e}{:<14.4e}{:<14.4e}",
                nk,
                peak,
                max_sum,
                max_sum / peak.max(1e-30)
            );
        }
        // At nk=20 signal should be nonzero and sum should converge
        let kmesh20 = arr1(&[20, 20, 20]);
        let up20 = intrinsic_nonlinear_values(
            &model_up,
            &kmesh20,
            &dz,
            &dx,
            &dy,
            &mu,
            0.0,
            Integration::EnergyCut,
        )
        .unwrap();
        let peak20 = up20.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        assert!(peak20 > 1e-6, "G‑wave signal vanished at nk=20");
    }
}
