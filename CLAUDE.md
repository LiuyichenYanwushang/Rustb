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
- **`Model`** (`src/basis.rs`): Central tight-binding model struct containing:
  - `lat`: Lattice vectors (d×d matrix)
  - `orb`: Orbital positions in fractional coordinates
  - `ham`: Hamiltonian matrix elements Hₘₙ(R) as 3D array [orb_m, orb_n, R_index]
  - `hamR`: Lattice vectors R for each hopping
  - `rmatrix`: Position matrix elements for velocity operator calculations
  - `atoms`: List of `Atom` objects with orbital information
  - `spin`: Boolean indicating spinful/spinless model
- **`Atom`** (`src/atom_struct.rs`): Atomic site with orbital projections (s, p, d, f, hybrid)
- **`OrbProj`**: Orbital projection descriptor for Slater-Koster integrals

### Key Modules
- **`basis`**: Core Model implementation, Hamiltonian construction, eigenvalue solving
- **`conductivity`**: Transport properties (Hall, spin Hall, nonlinear conductivities)
- **`surfgreen`**: Surface Green's functions for edge/surface state calculations
- **`wannier90`**: Interface with Wannier90 tight-binding models
- **`SKmodel`**: Slater-Koster parameterized models with two-center integrals
- **`geometry`**: Supercell construction, cutting pieces, dot structures
- **`kpoints`**: k‑point mesh generation and path construction
- **`math`**: Mathematical utilities (commutators, matrix operations)
- **`output`**: Band structure plotting and file output
- **`io`**: Text file I/O for arrays
- **`error`**: Centralized error handling with `TbError` enum
- **`generics`**: Trait definitions for numeric type flexibility
- **`ndarray_lapack`**: LAPACK bindings for ndarray matrices
- **`phy_const`**: Physical constants (ħ, e, k_B, etc.)
- **`unfold`**: Band unfolding for supercell calculations (implements `unfold` trait on `Model`)

### External Dependencies
- **`ndarray` / `ndarray-linalg`**: N‑dimensional arrays and linear algebra
- **`rayon`**: Parallel iteration over k‑points
- **`gnuplot`**: Plotting band structures and DOS
- **`num-complex`**: Complex number support
- **`serde`**: Serialization for model saving/loading

### Calculation Workflow
1. **Model Construction**: Create `Model` with lattice and orbital positions
2. **Set Hopping**: Add hopping terms with `add_hop`/`set_hop` and on‑site energies
3. **Band Structure**: Solve eigenvalue problem on k‑path with `solve_all_parallel`
4. **Transport**: Compute Berry curvature, Hall conductivity via `Hall_conductivity`
5. **Surface States**: Construct `surf_Green` object and compute local DOS
6. **Topology**: Calculate Wilson loops, Wannier centers, Chern numbers

### Parallelism
- k‑point loops are parallelized with `rayon` (e.g., `solve_all_parallel`)
- Large matrix operations use BLAS/LAPACK backends (Intel MKL, OpenBLAS, netlib)

### Error Handling
- All fallible operations return `Result<T, TbError>`
- `TbError` enum covers I/O, linear algebra, invalid input, and physics errors
- Use `?` operator for ergonomic error propagation

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
Each example in `examples/` is a standalone program demonstrating specific capabilities:
- `graphene`: Basic honeycomb lattice with edge states
- `WTe2_kp`: k·p model for WTe₂
- `Intrinsic_nonlinear`: Nonlinear Hall conductivity
- `z2_monopole`: Z₂ monopole charge calculation
- `Bi2F2`, `BiF_square`, `RuO2`: Material‑specific models
- `alterhexagonal`, `chern_alter`, `alter_twist`: Altermagnetic systems
- `Bi2F2_new`: Updated Bi₂F₂ model
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
- **Wannier90 compatibility**: `output_hr` writes `wannier90_hr.dat` format
- **Model serialization**: `Model` implements `Serialize`/`Deserialize` via `serde`

## Notes for Development
- When adding new hopping terms, ensure Hermitian conjugate exists (`-R` vector)
- Spin is a const generic (`Model<SPIN>`): `Model<true>` doubles the orbital basis with Pauli matrices in spin space
- Position matrix `rmatrix` is essential for velocity operator calculations
- Use `Gauge::Atom` or `Gauge::Lattice` for consistent phase conventions
- k‑points are in reciprocal lattice coordinates (fractions of reciprocal vectors)

## Performance Tips
- Enable link‑time optimization in release builds (`lto = "fat"` in Cargo.toml)
- Use Intel MKL for best numerical performance on Intel/AMD CPUs
- Parallelize over k‑points with `rayon` for embarrassingly parallel calculations
- For large supercells, consider iterative eigensolvers for partial spectra