# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rustb is a Rust library for calculating band structures, edge states, linear and nonlinear conductivities based on tight-binding models. It integrates with Wannier90's wannier model or tight-binding models to compute various physical properties.

## Key Dependencies

- **ndarray**: Multi-dimensional arrays and numerical computing
- **ndarray-linalg**: Linear algebra operations for ndarray (requires BLAS/LAPACK backend)
- **num-complex**: Complex number support
- **rayon**: Parallel computation
- **gnuplot**: Plotting and visualization
- **serde**: Serialization/deserialization

## BLAS/LAPACK Backend Configuration

The project requires a BLAS/LAPACK backend. Configure using Cargo features:

```bash
# Intel MKL (static linking)
cargo build --features intel-mkl-static

# OpenBLAS (static linking)  
cargo build --features openblas-static

# Netlib (static linking)
cargo build --features netlib-static
```

## Build Commands

```bash
# Standard build
cargo build

# Release build with optimizations
cargo build --release

# Build with specific BLAS backend
cargo build --features openblas-static

# Build documentation
cargo doc --open
```

## Test Commands

```bash
# Run all tests (requires BLAS/LAPACK backend configured)
cargo test

# Run tests with specific BLAS backend
cargo test --features intel-mkl-system

# Run specific test by name pattern
cargo test test_function_name

# Run tests with verbose output
cargo test -- --nocapture

# Run tests and show output for passing tests
cargo test -- --show-output

# Run all examples as tests (requires BLAS backend)
cargo testall --quiet

# Run specific example as test
cargo runexample graphene
```

## Cargo Aliases Configuration

Check `.cargo/config.toml` for custom aliases:

```bash
# View cargo configuration
cat .cargo/config.toml

# Available aliases:
# testall: Run tests with Intel MKL system backend
# runexample: Run examples with Intel MKL system backend
# mydoc: Build documentation without dependencies
```

## Cargo Configuration Details

The `.cargo/config.toml` contains essential configuration:

```toml
[build]
rustdocflags = [ "--html-in-header", "./docs-header.html" ]

[alias]
mydoc=["doc","--open","--no-deps"]
testall=["test","--features", "intel-mkl-system"]
runexample=["run","--features", "intel-mkl-system","--example"]
```

Key aliases usage:
- `cargo testall`: Run tests with Intel MKL system backend
- `cargo runexample`: Run examples with Intel MKL system backend  
- `cargo mydoc`: Build documentation without dependencies

## Example Usage

Examples are located in `examples/` directory and serve as both usage examples and integration tests:

```bash
# Run specific example
cargo run --example graphene

# Run example with release optimizations
cargo run --release --example WTe2_kp

# Run example with specific BLAS backend
cargo run --features intel-mkl-system --example graphene

# List all available examples
cargo run --example --list

# Run examples as tests using alias
cargo runexample graphene
```

Available examples include:
- `graphene`: Basic graphene tight-binding model
- `WTe2_kp`: WTe2 k·p model calculations
- `RuO2`: Rutile structure calculations
- `Intrinsic_nonlinear`: Nonlinear conductivity examples
- Various topological insulator models (BHZ, Haldane, etc.)

## Code Architecture

### Core Modules
- **lib.rs**: Main library entry point, defines `Model` struct and core functionality
- **atom_struct.rs**: Atom and orbital projection definitions
- **basis.rs**: Basis set operations
- **conductivity.rs**: Electrical conductivity calculations
- **geometry.rs**: Geometric operations and lattice utilities
- **surfgreen.rs**: Surface Green's function calculations
- **kpoints.rs**: k-point generation and Brillouin zone sampling
- **SKmodel.rs**: Slater-Koster parameterized tight-binding models

### Key Data Structures
- **Model**: Main tight-binding model container
- **Atom**: Atomic position and orbital information
- **OrbProj**: Orbital projection data
- **SlaterKosterModel**: Slater-Koster parameterized model builder
- **SkParams**: Slater-Koster two-center integral parameters

### Physics Capabilities
- Band structure calculation
- Surface state computation
- Anomalous Hall conductivity
- Spin Hall conductivity
- Nonlinear conductivity
- Wilson loop calculations
- Berry curvature
- Density of states
- Slater-Koster parameterized models

## Development Notes

1. **BLAS/LAPACK Dependency**: The project requires a working BLAS/LAPACK installation. Common options:
   - Intel MKL
   - OpenBLAS
   - Netlib

2. **Parallel Computation**: Uses Rayon for parallel k-point calculations

3. **Visualization**: Generates plots using gnuplot backend

4. **Performance**: Enable release optimizations for large calculations

5. **Testing**: Extensive test suite covering various physical models

6. **Error Handling**: Many functions return `Result<T, TbError>` - always handle Result types properly with `.unwrap()`, `.expect()`, or proper error handling

7. **Testing Strategy**: 
   - Unit tests are located within each module using `#[cfg(test)]`
   - Integration tests use examples that serve as functional tests
   - Test data files are stored in `tests/` directory
   - Physical models are tested against known results (graphene, topological insulators, etc.)

## Common Development Tasks

- Adding new tight-binding models
- Implementing new physical properties
- Optimizing performance for large systems
- Adding visualization capabilities
- Extending k-point sampling methods
- Adding new Slater-Koster parameter sets

## Development Workflow

1. **Testing New Features**:
   ```bash
   # Run tests for specific module during development
   cargo test --lib -- SKmodel
   
   # Run with specific backend for testing
   cargo test --features openblas-static
   
   # Test individual example
   cargo run --example graphene
   ```

2. **Benchmarking**: Use release builds for performance testing
   ```bash
   cargo build --release --features intel-mkl-static
   cargo run --release --example large_model
   ```

3. **Documentation**: Always update documentation when adding new features
   ```bash
   cargo doc --open --no-deps
   ```

## Slater-Koster Model Usage

The `SKmodel.rs` module provides Slater-Koster parameterized model building:

```rust
use Rustb::skmodel::*;

// Create graphene model with pz orbitals
let model = example_graphene().unwrap();

// Custom Slater-Koster model building
let sk_model = SlaterKosterModel::new(dim, lattice, atoms, spin);
let params = HashMap::from([
    ((AtomType::C, AtomType::C, 0), SkParams { v_pp_pi: Some(-2.7), ..Default::default() })
]);
let model = sk_model.build_model(1, &params).unwrap();
```

Supported orbital interactions include:
- s-s, s-p, p-p interactions
- s-d, p-d, d-d interactions (partial implementation)
- All standard Slater-Koster two-center integrals