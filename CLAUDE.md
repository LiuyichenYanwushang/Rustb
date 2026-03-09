# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- Spin‑ful models double the orbital basis (Pauli matrices in spin space)
- Position matrix `rmatrix` is essential for velocity operator calculations
- Use `Gauge::Atom` or `Gauge::Lattice` for consistent phase conventions
- k‑points are in reciprocal lattice coordinates (fractions of reciprocal vectors)

## Performance Tips
- Enable link‑time optimization in release builds (`lto = "fat"` in Cargo.toml)
- Use Intel MKL for best numerical performance on Intel/AMD CPUs
- Parallelize over k‑points with `rayon` for embarrassingly parallel calculations
- For large supercells, consider iterative eigensolvers for partial spectra