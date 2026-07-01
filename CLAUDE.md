# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# v0.8.0: Const-Generic DIM + RMatrixData (DONE)

> **Status**: DONE. Three generics replace runtime fields: `SPIN`, `DIM`, `R`.

## How Model<SPIN, DIM, R> works

```rust
pub struct Model<
    const SPIN: bool = false,
    const DIM: usize = 3,
    R: RMatrixData = NoRMatrix,
> {
    // NO dim_r field,  NO spin field
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    pub rmatrix: R,  // NoRMatrix = ZST, HasRMatrix = Array4 newtype
    // ...
}
```

- `DIM` defaults to 3. `dim_r()` returns `DIM`.
- `R` defaults to `NoRMatrix` (zero-sized, no storage overhead). `HasRMatrix(Array4)` stores position matrix elements.
- `tb_model(lat, orb, atoms)` — NO `dim_r` parameter.
- Hot path `match DIM { 1 => ..., 2 => ..., 3 => ... }` compile-time eliminated.
- Hot path `if <R as RMatrixData>::HAS_RMATRIX { ... }` compile-time eliminated.

### Constructing models

```rust
// Spinless 2D (most common)
let m = Model::<false, 2>::tb_model(lat, orb, None)?;

// Spinful 2D
let m = Model::<true, 2>::tb_model(lat, orb, None)?;

// 3D with position matrix (Wannier90)
let m = Model::<false, 3, HasRMatrix>::from_hr(path, seed, 0.0)?;
```

## Key design decisions

1. **No runtime fields** — SPIN, DIM, RMATRIX all compile-time. No dimension/spin/rmatrix flags in Model.
2. **`surf_Green` stores `spin: bool` and `dim_r: usize`** — const generics converted to runtime fields.
3. **Removed**: `SlaterKosterModel` — deleted as unused.
4. **`RMatrixData` trait** — `HasRMatrix` (Deref to Array4) and `NoRMatrix` (ZST). `NoRMatrix` model has literally no rmatrix field in memory.
5. **Safe `zaxpy`** — in `ndarray_lapack.rs`, wraps `blas::zaxpy` FFI; all call sites are safe Rust.
6. **`update_hamiltonian!` / `add_hamiltonian!` macros** — take `$spin` const generic; branches eliminated at compile time.

## Doc conventions

- `Model::<false, 2>` / `Model::<true, 3>` with turbofish for SPIN and DIM.
- `Model<SPIN, DIM, HasRMatrix>` when rmatrix is present.
- `surf_Green` retains `spin: bool` and `dim_r: usize` fields (derived from const generics via `from_Model`).

---

## Project Overview

Rustb is a Rust library for tight-binding model calculations in condensed matter physics. It computes band structures, density of states, transport properties (Hall conductivity, nonlinear responses), topological invariants (Chern numbers, Wilson loops), and surface states using Green's functions.

- **MSRV**: 1.90.0 (edition 2024)
- **Repo**: https://github.com/LiuyichenYanwushang/Rustb
- **Error handling**: Uses `thiserror` for `TbError` enum.
- **Docs**: `katexit` renders LaTeX in rustdoc; `docs-header.html` for custom CSS.
- **Version**: `Cargo.toml` still reads `0.7.0`; the actual API is v0.8.0. When bumping the crate, update `Cargo.toml` to `0.8.0`.

> **Note**: README.md still uses v0.6 API. Do NOT copy from README — use const-generic API as documented here.

## Common Commands

### Building
```bash
cargo build
cargo build --features intel-mkl-static   # Intel MKL
cargo build --features openblas-static    # OpenBLAS
cargo build --release
```

### Testing
```bash
cargo test
cargo test graphene                  # single test
cargo test -- --nocapture 2>&1 | head -100
```

Tests generate PDF plots via gnuplot (`pdfcairo` terminal).

### Development
```bash
cargo fmt
cargo clippy
cargo bench                          # criterion benchmarks
cargo mydoc                          # cargo doc --open --no-deps
cargo testall                        # cargo test --features intel-mkl-system
cargo runexample <name>              # cargo run --features intel-mkl-system --example <name>
```

## High-Level Architecture

### Core Data Structures

- **`Model<SPIN, DIM, R>`**: Central TB model. No runtime dimension, spin, or rmatrix flag.
- **`Gauge`**: `Atom` (orbital positions in phase) or `Lattice` (only R vectors).
- **`RMatrixData`**: Trait for rmatrix storage — `HasRMatrix` (newtype over Array4, Deref) or `NoRMatrix` (ZST).
- **`Dimension`**: Enum `one=1/two=2/three=3` for serde compat, NOT stored in Model.
- **`SpinDirection`**: `X/Y/Z` Pauli matrices, `None` = identity.
- **`surf_Green`**: Non-generic struct storing `spin: bool` and `dim_r: usize`.

### Trait Hierarchy

```
Velocity  (src/velocity.rs)          → v_α(k) operator
  ├─ BerryCurvature (src/response/traits.rs) → AHC, spin Hall, nonlinear Hall
  │    ├─ intrinsic NLH (src/response/nonlinear)
  │    └─ extrinsic NLH (src/response/nonlinear)
  └─ QuantumGeometry (src/quantum_geometry.rs) → QGT, quantum metric

FermiSurface / FermiSurfacePlane (src/fermi_surface.rs)

Berry (src/geometry.rs)              → Wilson loops, Berry phase, Wannier centres
CutModel (src/cut.rs)                → slab/ribbon (cut_piece), dot (cut_dot)

MagneticField (src/magnetic_field.rs)
Unfold (src/unfold.rs)
```

All trait impls: `impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Trait for Model<SPIN, DIM, R>`.

### Key Source Files

| File | Purpose |
|------|---------|
| `model.rs` | `Model<SPIN, DIM, R>` struct, `RMatrixData` trait, `HasRMatrix`/`NoRMatrix`, `Gauge`, serde |
| `model_build.rs` | Builder: `tb_model()`, `set_hop()`, `make_supercell()`, macros |
| `model_physics.rs` | `gen_ham()` (Bloch Hamiltonian), `dos()` |
| `velocity.rs` | `Velocity` trait — `gen_v()` with safe rmatrix commutator; `gen_v_projected()` fuses direction-weight projection |
| `response/` | All response functions: traits, direct‑sum APIs, simplex quadrature |
| `response/traits.rs` | `BerryCurvature` trait — per‑k‑point Berry curvature |
| `response/primitives.rs` | `compute_velocity_kernel` — band‑basis velocity matrix elements |
| `response/types.rs` | `VertexKernel`, `TrackedSimplex`, `SimplexDiagnostics` |
| `response/quadrature.rs` | 2D/3D symmetric quadrature rules, barycentric interpolation |
| `response/tracking.rs` | Band tracking (overlap → greedy assign → permute) + simplex builders |
| `response/kernel.rs` | Berry/QGT, optical, dipole kernel evaluators at quadrature points |
| `response/linear/` | Berry curvature Ω^{ab} + quantum metric g^{ab} simplex integration |
| `response/nonlinear/` | Berry dipole D^{ab;c} (energy‑cut) + intrinsic/extrinsic NLH |
| `response/optical/` | Optical conductivity σ^{ab}(ω) simplex integration |
| `quantum_geometry.rs` | `QuantumGeometry` trait — quantum geometric tensor, quantum metric |
| `geometry.rs` | `Berry` trait — Berry phase, Wilson loops, Wannier centres, hybrid Wannier functions |
| `surfgreen.rs` | Surface Green's function (Sancho-Rubio iterative method) |
| `wannier90.rs` | Wannier90 I/O, returns `Model<SPIN, 3, HasRMatrix>` when `_r.dat` present |
| `cut.rs` | `CutModel` trait — slab (`cut_piece`), dot/edge (`cut_dot`) from bulk models |
| `ndarray_lapack.rs` | LAPACK bindings + safe `zaxpy()` BLAS wrapper |
| `lib.rs` | Crate root, re-exports, integration tests |
| `fermi_surface.rs` | `FermiSurface`/`FermiSurfacePlane` traits (marching squares/tetrahedra → gnuplot); BXSF export (`BxsfExport` trait → XCrySDen/FermiSurfer); `write_spin_frmsf` free fn (spin‑split FRMSF for altermagnets) |
| `response/optical/` | Optical conductivity (direct sum + simplex quadrature) |
| `orbital_angular.rs` | Orbital angular momentum operator |
| `magnetic_field.rs` | Uniform magnetic field via Peierls substitution (`MagneticField` trait) |
| `unfold.rs` | Band unfolding for supercell→primitive projection (`Unfold` trait) |
| `solve_ham.rs` | Parallel diagonalization (`solve_all_parallel`, `solve_range_onek`) |
| `kpoints.rs`/`kpath.rs`/`kplane.rs` | k-mesh/k-path/k-plane generation |
| `output.rs` | gnuplot plotting (`show_band`, `show_surf_state`, `draw_heatmap`, `show_fermi_surface`) |
| `model_transform.rs` | Supercell construction, orbital removal/reordering, `shift_to_atom` |
| `model_utils.rs` | Internal utilities: `find_R()` for lattice vector lookup in `hamR` |
| `math.rs` | Matrix algebra: `comm()`, `anti_comm()`, `gauss()` smearing function |
| `atom_struct.rs` | `Atom` and `OrbProj` types |
| `error.rs` | `TbError` enum, `Result` alias |
| `generics.rs` | Numeric abstractions (`hop_use`, `usefloat`), `SpinDirection::from_index` |
| `phy_const.rs` | Physical constants: `e`, `ħ`, `μ_B`, `Φ₀`, quantum of conductance |
| `io.rs` | Text-file output helpers (`write_txt`, `write_txt_1`) |

### Conventions

- **k-points**: Fractional reciprocal coordinates; phase = `exp(2πi k·R)`.
- **Orbital positions**: Fractional coords (columns of `orb`).
- **Lattice vectors**: Stored in `Model::lat` (columns = real-space vectors).
  Reciprocal vectors via `Model::rec_lat()` (rows = reciprocal vectors,
  formula `B = 2π·(Aᵀ)⁻¹`).
- **Eigenbasis transformation** (`band` ↔ `evec` from `eigh`): the codebase
  convention is `U^T · O · U^*` (transpose, NOT conjugate‑transpose, on the
  left; conjugate on the right).  See `berry_curvature_n_onek` and
  `response::primitives::compute_velocity_kernel`.  Verify with the test
  `evec_transform_sanity` which checks `diag(U^T H U^*) == band` numerically.
- **Hermitian conjugates**: Auto-generated by `set_hop`/`add_hop`.
- **Public API**: `pub use X::*` re-exports from `lib.rs`.

## BLAS/LAPACK Backend

- **Intel MKL** (best x86_64 perf): `--features intel-mkl-static`
- **OpenBLAS**: `--features openblas-static`
- **netlib**: `--features netlib-static`

## Fermi Surface Export (3 output paths)

`fermi_surface.rs` provides three ways to export Fermi surfaces:

### 1. `FermiSurface::show_fermi_surface` — gnuplot (2D/3D, legacy)
Existing path. 2D: marching squares → PDF. 3D: marching tetrahedra → gnuplot pm3d → PDF.
Slow for 3D (fork gnuplot + triangle rendering).

### 2. `BxsfExport::write_bxsf` — BXSF ASCII (3D, XCrySDen/FermiSurfer)
Writes band energies on a k‑mesh in the XCrySDen BXSF format.
FermiSurfer/XCrySDen performs isosurface extraction internally.
Only implemented for `Model<SPIN, 3, R>`.

```rust
model.write_bxsf(&[50, 50, 50], 0.0, "bcc")?;  // → bcc.bxsf
```

### 3. `write_spin_frmsf` — FRMSF free function (3D, spin‑split)
Merges two spinless models (up/down) into one FRMSF file with a
matrix‑element block: `+1` for up bands, `−1` for down bands.
FermiSurfer renders this as red/blue spin‑split Fermi surfaces
(`Color scale mode = Input (1D)`, range `[-1, 1]`).

```rust
write_spin_frmsf(&up_model, &dn_model, &[50, 50, 50], 0.0, "altermagnet")?;
// → altermagnet.frmsf
```

### Critical conventions (BXSF / FRMSF)

These are hard-won; changing any of them breaks symmetry in the output.

**`Model::rec_lat()`** — returns reciprocal lattice vectors as **rows**:
`B = 2π · (Aᵀ)⁻¹` where `A = self.lat`. Row `i` is `bᵢ`. The BXSF/FRMSF
header writes three lines: `b[[i,0]], b[[i,1]], b[[i,2]]` for `i = 0,1,2`.

**`gen_kmesh` row ordering** — generated by recursion: ik1 (dim 0) outermost,
ik3 (dim 2) innermost (fastest). Row index: `ik1*ny*nz + ik2*nz + ik3`.
Fractional coords `k = (ik1/nx, ik2/ny, ik3/nz)` — matches `ishift = 1` in
FermiSurfer's convention (uniform Γ‑grid `i/N`).

**FRMSF/BXSF write order matches gen_kmesh** — both use ik1‑outer, ik3‑inner.
Therefore `eval` rows are already in the correct order for serialization:
`frmsf_order` simply iterates `row in 0..nk_total` sequentially. Do NOT
introduce an index formula like `ik1 + ik2*nx + ik3*nx*ny` — it will scramble
E(k) across k-points and destroy crystal symmetries (C3z, etc.).

**`e_fermi` must be subtracted** from band energies before writing.
FermiSurfer assumes `E_F = 0` by default; the export code computes
`E_n(k) - e_fermi`. Forgetting this draws the wrong isoenergy surface.

**`write_spin_frmsf` checks** — validates that up/dn models have the same
band count and lattice (element‑wise diff < 1e‑10). Both models use the
same k‑mesh and energy shift.

## Simplex Quadrature (replaces old Blochl tetrahedron)

> **Status**: IMPLEMENTED.  The old `tetrahedron.rs` (Blochl δ‑function on
> linearly‑interpolated final scalars) has been deleted.  All tetra methods
> (`*_tetra`, `compute_tetra_primitives`) are removed.  Use `response/` instead.

### Key idea

Instead of interpolating the final integrand `Ω_n(k)` at simplex vertices and
applying analytic δ‑function weights, the simplex path:

1. **Interpolates gauge‑invariant primitives** `K_nm = v^a_nm·v^b_mn` and band
   energies `E_n` linearly inside each simplex.
2. **Evaluates the kernel at quadrature points**: `Ω_n(q) = −2 Im Σ_m K_nm(q)/(d²(q)+η²)`.
3. **Weights by quadrature**: `∫_simplex f(k)dk ≈ V_simplex · Σ_q w_q f(q)`.

This preserves the singular `1/(E_n−E_m)²` structure near small gaps.

### Module structure

```
src/response/
├── types.rs        — VertexKernel, TrackedSimplex, SimplexDiagnostics
├── quadrature.rs   — 2D/3D symmetric quadrature rules, barycentric interp
├── tracking.rs     — band tracking (overlap → greedy → permute) + simplex builders
├── kernel.rs       — eval_berry_kernel, eval_optical_kernel, quadrature helpers
├── energy_cut.rs   — 2D/3D analytic energy‑cut integration for Berry dipole
├── primitives.rs   — Model::compute_velocity_kernel (band‑basis velocities)
├── linear/         — Berry curvature + quantum metric → berry_curvature_simplex
├── nonlinear/      — Berry dipole + intrinsic/extrinsic NLH
└── optical/        — Optical conductivity → optical_conductivity_simplex
```

### Usage

```rust
// Berry curvature + quantum metric
let (metric, berry, unsafe_count) = model.berry_curvature_simplex(
    &arr1(&[50, 50]), &dx, &dy, 0.05,
)?;

// Berry curvature dipole (2D/3D, analytic energy-cut)
let (dipole, _) = model.berry_curvature_dipole_energy_cut(
    &arr1(&[30, 30]), &dx, &dy, &dx, &mu, 10.0, 0.05,
)?;

// Optical conductivity
let sigma = model.optical_conductivity_simplex(
    &arr1(&[30, 30]), &dx, &dy, 0.5, 0.1, 0.0, 300.0,
)?;
```

### Public API reference

**Model‑level (recommended for most users)**

| Method | Returns | Description |
|--------|---------|-------------|
| `model.berry_curvature_simplex(k_mesh, dir_a, dir_b, eta)` | `(g, Ω, unsafe)` | Berry curvature + quantum metric in Cartesian volume |
| `model.berry_curvature_dipole_energy_cut(k_mesh, dir_a, dir_b, dir_c, mu, T, eta)` | `(D(μ), unsafe)` | Berry dipole D^{ab;c}(μ,T), 2D/3D analytic energy‑cut |
| `model.optical_conductivity_simplex(k_mesh, dir_a, dir_b, ω, η, μ, T)` | `σ(ω)` | Complex optical conductivity |
| `model.compute_velocity_kernel(k_vec, dir_a, dir_b, dir_c?, gauge, spin)` | `VertexKernel` | Per‑k‑point band‑basis velocity primitives |

**Low‑level (for custom integration loops)**

| Function | Module | Description |
|----------|--------|-------------|
| `linear::integrate(all_pts, k_mesh, eta)` | `response::linear` | BZ integral of Berry + metric (fractional coords) |
| `integrate_dipole_energy_cut_2d(all_pts, k_mesh, mu, T, eta)` | `response` | 2D dipole via analytic line cuts |
| `integrate_dipole_energy_cut_3d(all_pts, k_mesh, mu, T, eta)` | `response` | 3D dipole via analytic plane cuts |
| `optical::integrate(all_pts, k_mesh, ω, η, μ, T)` | `response::optical` | Optical conductivity (fractional coords) |
| `build_triangles_2d(ix, iy, nx, ny, inv_nx, inv_ny, all_pts)` | `response` | Build tracked 2D triangles for one cell |
| `build_tetrahedra_3d(ix, iy, iz, nx, ny, nz, ...)` | `response` | Build tracked 3D tetrahedra for one cell |
| `eval_berry_kernel(band_q, k_ab_q, eta, nsta)` | `response` | Evaluate `(g_n, Ω_n)` at one quadrature point |
| `eval_optical_kernel(band_q, k_ab_q, ω, η, μ, β, nsta)` | `response` | Evaluate `σ_nm` at one quadrature point |

**Data types**

| Type | Fields |
|------|--------|
| `VertexKernel` | `band`, `k_ab`, `k_bc: Option<_>`, `k_ac: Option<_>`, `vdiag: Option<_>`, `vdiag_a: Option<_>`, `vdiag_b: Option<_>`, `evec` |
| `TrackedSimplex` | `vertices: Vec<VertexKernel>`, `volume: f64`, `coords: Array2<f64>`, `diag: SimplexDiagnostics` |
| `SimplexDiagnostics` | `min_gap: f64`, `min_assignment_overlap: f64`, `tracking_conflict: bool` |

### API changes (v0.8 → post‑tetra)

**Deleted** — no longer exist:

| Old API | Reason |
|---------|--------|
| `tetrahedron_integrate()` | Blochl δ‑function on final scalar — replaced by simplex |
| `tetrahedron_volume_integrate()` | Linear vertex average — replaced by simplex |
| `Hall_conductivity_tetra()` | AHC via Blochl — removed |
| `Nonlinear_Hall_conductivity_Extrinsic_tetra()` | Extrinsic NLH via Fermi‑surface cuts — removed |
| `Nonlinear_Hall_conductivity_Extrinsic_tetra_sym()` | Symmetrised wrapper — removed |
| `Nonlinear_Hall_conductivity_Intrinsic_tetra()` | Intrinsic NLH via segment/triangle integrals — removed |
| `compute_tetra_primitives()` | Renamed → `compute_velocity_kernel` |
| `TetraKPoint`, `IntrinsicTetraPoint` | Replaced by `VertexKernel` |
| `src/tetrahedron.rs` | Entire file deleted |

**Retained** — unchanged:

| API | Description |
|-----|-------------|
| `Hall_conductivity()` / `Hall_conductivity_mu()` / `Hall_conductivity_adapted()` | AHC via direct k‑mesh sum |
| `Nonlinear_Hall_conductivity_Extrinsic()` / `_sym()` | Extrinsic NLH via direct k‑mesh sum |
| `Nonlinear_Hall_conductivity_Intrinsic()` | Intrinsic NLH via direct k‑mesh sum |
| `berry_curvature_n_onek()` / `berry_curvature_onek()` / `berry_curvature()` | Per‑k‑point / k‑path Berry curvature |
| `berry_curvature_dipole_n_onek()` / `berry_curvature_dipole_n()` | Per‑k‑point Berry dipole |
| `berry_connection_dipole_onek()` / `berry_connection_dipole()` | Per‑k‑point Berry connection dipole |
| `optical_geometry_n_onek()` | Per‑k‑point optical geometry |
| `quantum_geometry_n_onek()` / `quantum_geometry_n()` / `quantum_geometry()` | Per‑k‑point QGT |

**New** — added in this refactoring:

| API | Description |
|-----|-------------|
| `berry_curvature_simplex()` | Berry + metric via simplex quadrature (Cartesian) |
| `berry_curvature_dipole_energy_cut()` | Berry dipole via analytic energy‑cut (2D line, 3D plane) |
| `optical_conductivity_simplex()` | Optical σ(ω) via simplex quadrature (Cartesian) |
| `compute_velocity_kernel()` | Per‑k‑point band‑basis velocity primitives |
| `response::*` module | Public low‑level simplex integration primitives |

### Known limitations

- **Energy-cut dipole is 2D only** — 3D tetrahedral cut integration is future work
- **Intrinsic NLH tetra path removed** — `Nonlinear_Hall_conductivity_Intrinsic_tetra` deleted
- **Volume-quadrature dipole is noisy at low T** — use `berry_curvature_dipole_energy_cut` for 2D low-temperature Fermi-window integrals
- **No band tracking in intrinsic NLH** path (not yet ported to simplex)

## Performance Notes

**`zaxpy`** (`src/ndarray_lapack.rs`): safe BLAS `y += alpha * x` for `Complex<f64>` slices.
Preferred over `Zip`/elementwise for direction-weight accumulation of dense matrices.
Use `Complex::new(weight, 0.0)` as scalar; only call when source and destination are
standard contiguous slices (`.as_slice().unwrap()` / `.as_slice_mut().unwrap()`).

**Allocation reduction**: avoid `Array1::from_vec(...)` and `.to_owned()` inside hot loops;
prefer preallocated buffers. For rayon folds over `mu` values, mutate a local
`Array1<f64>` accumulator directly instead of allocating per-iteration.

**Autovec/SIMD**: simple contiguous slice loops autovectorize better than `ndarray`
indexed/transposed views. Use `RUSTFLAGS="-C target-cpu=native"` for AVX2/AVX512.
BLAS backends (MKL/OpenBLAS) dispatch optimized kernels independently.

## Refactoring Guidelines

- **No batch sed/Python**: modify files one at a time with proper tooling.
- **Parallel agents for multi-file changes**: core file first manually, then launch
  3-4 agents in parallel for distinct groups of files.
- **Commit after each successful `cargo check`**: prevents data loss from
  `git checkout` discarding uncommitted work.

---

# Deleted: Old Blochl Tetrahedron Code

> **Removed 2026-06-30.**  All `*_tetra` methods, `TetraKPoint`, `tetrahedron.rs`,
> and the ~2500 lines of Blochl/Fermi‑surface helper code have been deleted.
> The replacement is `src/response/` (simplex quadrature, see above).

## Nonlinear Hall index conventions

- `Nonlinear_Hall_conductivity_Extrinsic` returns the unsymmetrized
  kernel `S_{ab;c} = ∫(-df/dE) v_c Omega_ab dk`.  For current-first
  `chi_ext[a,b,c]`, use `Nonlinear_Hall_conductivity_Extrinsic_sym` which
  computes `0.5 * (S_{ab;c} + S_{ac;b})`.
- `Nonlinear_Hall_conductivity_Intrinsic` is current‑first:
  arguments `(current, field_1, field_2)` map to `sigma_int^{field_1 field_2; current}`.
- The charge intrinsic implementation uses
  `G^{ij}=Re sum_m v^i_nm v^j_mn/(E_n-E_m)^3`.  Literature formulas
  that define `G=2 Re sum ...` differ by an overall factor of 2.
