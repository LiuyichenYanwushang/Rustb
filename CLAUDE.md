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
- **thiserror**: Custom error types

## BLAS/LAPACK Backend Configuration

The project requires a BLAS/LAPACK backend. Configure using Cargo features:

```bash
# Intel MKL (static linking)
cargo build --features intel-mkl-static

# OpenBLAS (static linking)  
cargo build --features openblas-static

# Netlib (static linking)
cargo build --features netlib-static

# Intel MKL (system linking)
cargo build --features intel-mkl-system
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

# Build documentation without dependencies
cargo mydoc
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

# Run tests for specific module
cargo test --lib -- SKmodel
```

## Cargo Aliases Configuration

Check `.cargo/config.toml` for custom aliases:

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
- `BHZ`: Bernevig-Hughes-Zhang model
- `Haldane`: Haldane model for quantum Hall effect
- Various topological insulator models

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
- **error.rs**: Custom error types and error handling
- **wannier90.rs**: Wannier90 file format support
- **io.rs**: File I/O utilities for data export

### Key Data Structures
- **Model**: Main tight-binding model container (model_struct.rs:8)
- **Atom**: Atomic position and orbital information (atom_struct.rs)
- **OrbProj**: Orbital projection data
- **SlaterKosterModel**: Slater-Koster parameterized model builder (SKmodel.rs)
- **SkParams**: Slater-Koster two-center integral parameters
- **TbError**: Centralized error type for all fallible operations (error.rs:14)

### Physics Capabilities
- Band structure calculation
- Surface state computation using Green's functions
- Anomalous Hall conductivity
- Spin Hall conductivity
- Nonlinear conductivity
- Wilson loop calculations
- Berry curvature and Berry phase
- Density of states
- Slater-Koster parameterized models
- Wannier90 model import/export

## Mathematical Foundation

The library implements the tight-binding Hamiltonian:
$$
H = \sum_{i,j} t_{ij} c_i^\dagger c_j + \sum_i \epsilon_i c_i^\dagger c_i
$$
where $t_{ij}$ are hopping parameters and $\epsilon_i$ are on-site energies.

For transport calculations, we compute the Berry curvature:
$$
\Omega_n(\mathbf{k}) = -2\,\text{Im}\sum_{m\neq n} \frac{\bra{n}\partial_{k_x} H\ket{m}\bra{m}\partial_{k_y} H\ket{n}}{(E_n - E_m)^2}
$$

## Development Notes

1. **BLAS/LAPACK Dependency**: The project requires a working BLAS/LAPACK installation. Common options:
   - Intel MKL
   - OpenBLAS
   - Netlib

2. **Parallel Computation**: Uses Rayon for parallel k-point calculations (lib.rs:40)

3. **Visualization**: Generates plots using gnuplot backend

4. **Performance**: Enable release optimizations for large calculations

5. **Error Handling**: Many functions return `Result<T, TbError>` - always handle Result types properly with `.unwrap()`, `.expect()`, or proper error handling

6. **Testing Strategy**: 
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
- Improving error handling and documentation

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

4. **Error Handling**: Follow the pattern in error.rs for consistent error reporting

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

## File I/O Utilities

The `io.rs` module provides file output functions:
- `write_txt`: Export 2D arrays to formatted text files
- `write_txt_1`: Export 1D arrays to formatted text files

These utilities handle proper number formatting and spacing for scientific data analysis.

## Error Handling Patterns

All fallible functions return `Result<T, TbError>`. Common error handling patterns:

```rust
// Using unwrap for development (panic on error)
let model = Model::tb_model(dim, lat, orb, false, None).unwrap();

// Using expect with custom message
let model = Model::tb_model(dim, lat, orb, false, None)
    .expect("Failed to create TB model");

// Proper error handling with match
match Model::tb_model(dim, lat, orb, false, None) {
    Ok(model) => { /* success */ },
    Err(TbError::Io(e)) => { /* handle IO error */ },
    Err(e) => { /* handle other errors */ },
}
```

## Performance Optimization Tips

1. Use release builds for production calculations
2. Enable appropriate BLAS backend for your system
3. Leverage Rayon parallelism for k-point calculations
4. Precompute values where possible to avoid redundant calculations
5. Use appropriate k-point mesh sizes for the required precision

## Testing Patterns

The codebase uses extensive testing with both unit tests and integration tests:

```rust
#[test]
fn test_function_name() {
    // Test setup
    let model = create_test_model();
    
    // Test execution
    let result = model.some_function();
    
    // Assertions
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), expected_value);
}
```

Integration tests are implemented as examples that serve dual purpose as both documentation and functional tests.