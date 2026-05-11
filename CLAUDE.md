# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# v0.7.0: Const-Generic Spin

> **Status**: DONE. `Model<const SPIN: bool = false>` replaces the `spin: bool` runtime field.
> All runtime `if self.spin` branches on hot paths are now compile-time dead-code-eliminated.

## How Model<SPIN> works

```rust
pub struct Model<const SPIN: bool = false> {  // no `spin` field
    pub dim_r: Dimension,
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    // ... other fields unchanged
}
```

- `Model<false>` → spinless, `nsta() == norb()`
- `Model<true>` → spinful, `nsta() == 2 * norb()`, basis is [orb_1↑, ..., orb_N↑, orb_1↓, ..., orb_N↓]
- `Model` without turbofish defaults to `Model<false>`
- `Model<false>` and `Model<true>` are **distinct types** — they cannot be mixed in collections (arrays, Vec, etc.) without type erasure

### Constructing models

```rust
// Spinless (default)
let m = Model::<false>::tb_model(2, lat, orb, None)?;
// or equivalently: Model::tb_model(2, lat, orb, None)?

// Spinful
let m = Model::<true>::tb_model(2, lat, orb, None)?;
```

The old `spin: bool` parameter on `tb_model()` is **removed**. SPIN is determined entirely by the const generic.

### Serde

Manual `Serialize`/`Deserialize` implementations. Serialize writes `"spin": SPIN` as a field. Deserialize reads the `spin` field from the data and verifies it matches `SPIN`, returning an error if it doesn't match.

## Key design decisions

1. **No `AnyModel` enum** — `Model<SPIN>` is the only type. Users specify SPIN at compile time via turbofish.
2. **All `impl Model` blocks are now `impl<const SPIN: bool> Model<SPIN>`** — methods are monomorphized for each SPIN value separately.
3. **Traits are NOT generic over SPIN** — they use `impl<const SPIN: bool> Trait for Model<SPIN>`. Trait methods are static dispatch, not trait objects.
4. **`surf_Green` keeps `spin: bool`** — its hot path is O(n³) matrix inversion; the single branch is negligible.
5. **`update_hamiltonian!` / `add_hamiltonian!` macros** — take `SPIN` (const generic) instead of `self.spin`. The `if $spin { match pauli ... } else { ... }` branches are eliminated at compile time.

## Doc comment guidelines

When writing or updating rustdoc for v0.7.0+:

1. **Never reference `self.spin` or `model.spin`** in documentation — the field no longer exists.
2. **Never reference `spin: bool` as a parameter** — `tb_model()` no longer takes it.
3. **Use const-generic terminology**: "spinful model (`SPIN = true`)" or "`Model<true>`" instead of "`spin = true`".
4. **Macro `$spin` parameter**: keep current docs (it's an internal implementation detail, still takes a bool value at the macro level — the caller now passes `SPIN` the const generic).
5. **`surf_Green`**: its `spin: bool` field still exists and is an implementation detail; documentation can reference it but should note it's derived from `Model<SPIN>` via `from_Model`.
6. **Doc examples**: use `Model::<false>::tb_model(...)` or `Model::<true>::tb_model(...)` (always with turbofish for clarity).

### Doc anti-patterns to fix

```rust
// WRONG — references deleted spin field/parameter
/// * `spin` - Whether to double the basis for spin (spinful model).
/// If the model is spinless (`spin = false`), ...
/// Create a spinful model with explicit atoms:

// CORRECT
/// * `SPIN` is a const generic on [`Model`]; `Model::<true>` doubles the basis for spin.
/// For spinless models (default), the hopping is simply written...
/// Create a spinful model:
/// ```
/// Model::<true>::tb_model(3, lat, orb, Some(atoms)).unwrap();
/// ```
```

### Key files with doc comments needing attention

- `src/model_build.rs` — `tb_model` docs, `update_hamiltonian!`/`add_hamiltonian!` docs, `set_hop`/`add_hop` docs, top-level module docs
- `src/model.rs` — struct-level rustdoc
- `src/velocity.rs` — `gen_v` comment about spin doubling
- `src/model_physics.rs` — `gen_ham` comment about spin doubling
- `src/conductivity.rs` — any spin-related docs
- `src/surfgreen.rs` — `surf_Green` spin field docs
- `src/SKmodel.rs` — `SlaterKosterModel` docs
- `src/lib.rs` — quick start example (line 164 uses `Model::<false>::`)

---

## Project Overview

Rustb is a Rust library for tight-binding model calculations in condensed matter physics. It computes band structures, density of states, transport properties (Hall conductivity, nonlinear responses), topological invariants (Chern numbers, Wilson loops), and surface states using Green's functions. The library interfaces with Wannier90 and supports Slater-Koster parameterized models.

## Common Commands

### Building
```bash
# Standard build (uses default BLAS backend)
cargo build

# Build with Intel MKL (static linking)
cargo build --features intel-mkl-static

# Build with OpenBLAS (static linking)
cargo build --features openblas-static

# Build with netlib (reference LAPACK)
cargo build --features netlib-static

# Build in release mode with optimizations
cargo build --release
```

### Testing
```bash
# Run all tests (generates PDF plots in tests/ directories)
cargo test

# Run a specific test (e.g., graphene model)
cargo test graphene

# Run tests without generating plots (if you want to avoid PDF creation)
cargo test -- --nocapture 2>&1 | head -100

# Run a test from a specific module
cargo test --test basis
```

**Note:** Tests generate PDF plots using gnuplot. Ensure `gnuplot` is installed and `pdfcairo` terminal is available (install gnuplot with PDF support). On Ubuntu: `sudo apt install gnuplot`. On macOS: `brew install gnuplot`.

### Running Examples
```bash
# List available examples (see Cargo.toml [[example]] sections)
cargo run --example graphene
cargo run --example WTe2_kp
cargo run --example Intrinsic_nonlinear
cargo run --example z2_monopole
cargo run --example BiF_square
cargo run --example Bi2F2
cargo run --example Bi2F2_new
cargo run --example yuxuan_try
cargo run --example RuO2
cargo run --example alterhexagonal
cargo run --example chern_alter
cargo run --example alter_twist
```

### Documentation
```bash
# Build library documentation (includes custom header docs-header.html)
cargo doc --open

# Build docs with all features
cargo doc --all-features --open

# Use the alias for documentation (opens browser automatically)
cargo mydoc
```

### Development
```bash
# Format code
cargo fmt

# Check for clippy lints
cargo clippy

# Run tests with verbose output
cargo test -- --nocapture

# Run a single integration test file
cargo test --test integration_test_name
```

### Cargo Aliases (from .cargo/config.toml)
```bash
# Build documentation and open in browser
cargo mydoc

# Run all tests with Intel MKL system library
cargo testall

# Run an example with Intel MKL system library
cargo runexample graphene  # or any other example name
```

## High-Level Architecture

### Core Data Structures

- **`Model<const SPIN: bool>`** (`src/model.rs`): Central tight-binding model struct. SPIN is a const generic (`Model<false>` = spinless, `Model<true>` = spinful with doubled basis [orb₁↑,…,orb_N↑, orb₁↓,…,orb_N↓]). Key fields:
  - `dim_r`: `Dimension` enum (Dim1/Dim2/Dim3)
  - `lat`: Lattice vectors (d×d matrix, in Å)
  - `orb`: Orbital positions in fractional coordinates
  - `ham`: Hopping amplitudes Hₘₙ(R) as 3D array `[norb, norb, nR]`
  - `hamR`: Lattice vectors R (integer, in units of primitive cell vectors) for each hopping
  - `rmatrix`: Position matrix elements rₘₙ(R) from Wannier90 (for velocity operator)
  - `atoms`: `Vec<Atom>` with orbital/position information
- **`Gauge`** (`src/model.rs`): Enum controlling velocity operator convention — `Gauge::Atom` (orbital positions included in phase) or `Gauge::Lattice` (only R vectors in phase)
- **`Dimension`** (`src/model.rs`): Enum for system dimensionality — `Dim1`, `Dim2`, `Dim3`
- **`SpinDirection`** (`src/model.rs`): Enum for spin projection in Berry curvature/quantum geometry — `X`, `Y`, `Z` (Pauli matrices σ_x, σ_y, σ_z). Pass `None` for identity (no spin projection)
- **`Atom`** (`src/atom_struct.rs`): Atomic site with orbital projections (s, p, d, f, and hybrid orbitals)
- **`OrbProj`**: Orbital projection descriptor for Slater-Koster integrals

### Trait Hierarchy (Trait-Based API)

Responses (Berry curvature, Hall conductivity, quantum geometry) use a layered trait design:

```
Velocity  (src/velocity.rs)
  └─ BerryCurvature  (src/conductivity.rs)
       └─ QuantumGeometry  (src/quantum_geometry.rs)
```

- **`Velocity`**: Computes the velocity operator matrix v_α(k) = (1/ħ) ∂H/∂k_α at a given k-point. Uses atomic-gauge formula with position matrix elements if available.
- **`BerryCurvature`**: Band-resolved and summed Berry curvature, Hall conductivity (AHC), nonlinear Hall (intrinsic + extrinsic), Berry curvature dipole, Berry connection dipole.
- **`QuantumGeometry`**: Quantum geometric tensor G_{n,αβ}(k), quantum metric g_{n,αβ}, Berry curvature Ω_{n,αβ}. Band-resolved and Fermi-sea summed.

A separate trait pair handles Fermi surface visualization:

```
FermiSurface   (src/fermi_surface.rs)   → 2D marching squares / 3D marching tetrahedra
FermiSurfacePlane (src/fermi_surface.rs) → 3D k-plane slice Fermi surface
```

All methods on these traits are provided methods with default implementations. Models get them via blanket impls (`impl<const SPIN: bool> Velocity for Model<SPIN>`, etc.).

### Key Source Files

| File | Purpose |
|------|---------|
| `model.rs` | `Model<SPIN>` struct, `Gauge`, `Dimension`, `SpinDirection` enums |
| `model_build.rs` | Builder methods: `tb_model()`, `set_hop()`, `add_hop()`, `set_onsite()`, `update_hamiltonian!`, `add_hamiltonian!` macros, supercell construction |
| `model_physics.rs` | Physics methods: `gen_ham()` (Bloch Hamiltonian), `gen_v()` (velocity), `solve_band()`, `solve_all()`, `solve_all_parallel()`, density of states |
| `model_transform.rs` | Model transformations: supercell, cut, slice, apply symmetry operations |
| `model_utils.rs` | Internal utilities: `find_R()`, `remove_col()`, `remove_row()` |
| `velocity.rs` | `Velocity` trait — velocity operator matrices |
| `conductivity.rs` | `BerryCurvature` trait — Berry curvature, AHC, nonlinear Hall |
| `quantum_geometry.rs` | `QuantumGeometry` trait — quantum metric, Berry curvature with broadening |
| `optical_conductivity.rs` | Optical conductivity (Kubo formula) |
| `magnetic_field.rs` | Peierls substitution, magnetic supercells |
| `surfgreen.rs` | Surface Green's function (iterative method) for edge/surface states |
| `wannier90.rs` | Wannier90 `_hr.dat`, `_r.dat`, `_wsvec.dat` file I/O |
| `SKmodel.rs` | Slater-Koster tight-binding with two-center integrals |
| `atom_struct.rs` | `Atom`, `OrbProj` types |
| `geometry.rs` | Supercell geometry, cutting pieces, dot/hole structures |
| `cut.rs` | Model slicing/cutting operations |
| `kpoints.rs` | k‑point mesh generation (`gen_kmesh`) |
| `kplane.rs` | k‑plane mesh generation (`gen_kplane`) |
| `kpath.rs` | High-symmetry k‑path construction |
| `fermi_surface.rs` | `FermiSurface` / `FermiSurfacePlane` traits, marching squares/tetrahedra, gnuplot rendering |
| `solve_ham.rs` | Eigensolver helpers (diagonalize H(k), compute eigenvectors) |
| `unfold.rs` | Band unfolding for supercell calculations |
| `orbital_angular.rs` | Orbital angular momentum operators |
| `output.rs` | Plotting: `show_band()`, `show_surf_state()`, `draw_heatmap()` (via gnuplot) |
| `io.rs` | Text file I/O: `write_txt()`, `write_txt_1()` |
| `error.rs` | `TbError` enum, `Result<T>` type alias |
| `math.rs` | `comm()` (commutator), matrix utilities |
| `generics.rs` | Numeric trait bounds |
| `ndarray_lapack.rs` | LAPACK FFI bindings |
| `phy_const.rs` | Physical constants (ħ, e, k_B, etc.) |
| `lib.rs` | Crate root: module declarations, re-exports (`pub use X::*`), integration tests |

### External Dependencies
- **`ndarray` / `ndarray-linalg`**: N‑dimensional arrays and linear algebra (eigendecomposition, SVD)
- **`rayon`**: Parallel iteration over k‑points
- **`gnuplot`**: Plotting band structures, DOS, surface states
- **`num-complex`**: Complex number support (Complex<f64>)
- **`serde`**: Model serialization/deserialization

### Calculation Workflow
1. **Model Construction**: `Model::<SPIN>::tb_model(dim, lat, orb, atoms)` creates an empty model
2. **Set Hopping**: `set_hop()`, `add_hop()`, `set_onsite()` populate Hₘₙ(R); Hermitian conjugates are auto-generated
3. **Band Structure**: `solve_all_parallel()` diagonalizes H(k) on a k‑path or k‑mesh
4. **Transport**: `berry_curvature()` / `Hall_conductivity()` via the `BerryCurvature` trait
5. **Surface States**: `surf_Green::from_Model()` constructs the surface Green's function; `surf_state()` computes local DOS
6. **Topology**: Wilson loops (`wannier_center()`), Chern numbers, Z₂ invariants
7. **Quantum Geometry**: `quantum_metric_n()` / `quantum_metric_sum()` via the `QuantumGeometry` trait
8. **Fermi Surface**: `show_fermi_surface()` via `FermiSurface` trait; 2D contour or 3D isosurface

### k‑Point and Coordinate Conventions
- **k‑points**: Fractional reciprocal coordinates; phase factor in `gen_ham()` uses `exp(2πi k·R)` where R is in integer lattice-vector units
- **Orbital positions**: Fractional coordinates (columns of `orb` are the fractional coordinates of orbital positions)
- **Cartesian conversion**: Multiply fractional vectors by the lattice matrix `lat` (each column of `lat` is a lattice vector in Å)

## BLAS/LAPACK Backend Selection
The library supports multiple BLAS/LAPACK implementations via Cargo features:
- **Intel MKL** (recommended for performance): `--features intel-mkl-static` or `intel-mkl-system`
- **OpenBLAS**: `--features openblas-static` or `openblas-system`
- **netlib** (reference): `--features netlib-static` or `netlib-system`

Without features, `ndarray-linalg` uses the default system BLAS. For optimal performance on x86_64, enable Intel MKL.

## Plotting Output
Tests and examples generate PDF plots via gnuplot in `tests/` subdirectories:
- Band structures: `band.pdf`
- Density of states: `dos.pdf`
- Surface states: `surf_state_{l,r,b}.pdf`
- Wilson loop spectra: `wcc.pdf`
- Unfolded band structures: `unfold_band.pdf`

The plotting functions (`show_band`, `show_surf_state`, `draw_heatmap`) automatically create these files.

## Examples Structure
Each example in `examples/` is a standalone binary (directory with `main.rs` and often data files like `KLABELS`, `BAND.dat`):
- `graphene`: Basic honeycomb lattice with edge states and band structure
- `BHZ`: Bernevig-Hughes-Zhang model for quantum spin Hall effect (not yet registered in Cargo.toml [[example]])
- `WTe2_kp`: k·p model for WTe₂
- `Intrinsic_nonlinear`: Nonlinear Hall conductivity
- `z2_monopole`: Z₂ monopole charge calculation
- `Bi2F2`, `BiF_square`, `RuO2`: Material‑specific models
- `alterhexagonal`, `chern_alter`, `alter_twist`: Altermagnetic systems
- `Bi2F2_new`: Updated Bi₂F₂ model with additional analyses
- `yuxuan_try`: Experimental/test configurations

Examples serve both as usage templates and validation tests.

## Testing Philosophy
The `tests` module in `src/lib.rs` contains comprehensive integration tests that:
1. Verify physical correctness (e.g., Chern numbers equal 1 for Haldane model)
2. Compare different calculation methods (Berry curvature vs. Wilson loop)
3. Generate visual output for manual inspection
4. Benchmark performance of key routines

Run `cargo test` to execute all tests; check generated PDFs in `tests/` for visual verification.

## File I/O
- **Text files**: Use `write_txt` and `write_txt_1` from `io` module for 1D/2D arrays
- **Wannier90 compatibility**: `from_hr(path, seedname, zero_energy)` loads `_hr.dat` files; `from_r()` also loads position matrix elements from `_r.dat`
- **Model serialization**: `Model` implements `Serialize`/`Deserialize` via `serde`
- **Plotting**: `show_band()`, `show_surf_state()`, `draw_heatmap()` generate PDFs via gnuplot

## Notes for Development
- When adding new hopping terms, Hermitian conjugates are auto-generated (the `-R` vector with conjugated amplitude is added automatically by `set_hop`/`add_hop`)
- `Model<SPIN>` is const-generic: `Model<true>` doubles the orbital basis with Pauli matrices in spin space. The two types are distinct — they cannot be mixed in collections without type erasure
- Position matrix `rmatrix` from Wannier90 (requires `write_rmn = true`) is used for accurate velocity operators. Without it, the rmatrix commutator term in `gen_v()` is omitted
- `Gauge::Atom` includes orbital positions in the phase factor; `Gauge::Lattice` uses only R vectors. This affects velocity matrices and all derived quantities
- k‑points are in fractional reciprocal coordinates; the phase factor uses `exp(2πi k·(R - τ_m + τ_n))`
- The trait hierarchy (`Velocity` → `BerryCurvature` → `QuantumGeometry`) means new physics quantities can be added as new traits that extend existing ones
- All public APIs use `pub use X::*` re-exports from `lib.rs`; users import from `rustb::` directly

## Performance Tips
- Release builds use `lto = "fat"` and `codegen-units = 1` for maximum optimization
- Use Intel MKL (`--features intel-mkl-static`) for best numerical performance on Intel/AMD CPUs
- k‑point loops are parallelized with `rayon`; the trait methods (e.g., `berry_curvature`) accept `&[Array1<f64>]` and parallelize internally
- For large supercells, consider iterative eigensolvers for partial spectra