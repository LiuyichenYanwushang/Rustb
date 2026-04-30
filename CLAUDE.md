# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# REFACTORING PLAN: Const-Generic Spin (v0.7.0)

> **Status**: PLANNING. Target: replace `spin: bool` field in `Model` with const generic `const SPIN: bool`, eliminating all runtime spin branches from hot paths.

## 1. Core Design

### 1.1 New Model definition (`src/model.rs`)

```rust
#[derive(Clone, Debug)]
pub struct Model<const SPIN: bool = false> {
    pub dim_r: Dimension,
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    pub orb_projection: Vec<OrbProj>,
    pub atoms: Vec<Atom>,
    pub ham: Array3<Complex<f64>>,
    pub hamR: Array2<isize>,
    pub rmatrix: Array4<Complex<f64>>,
}
// field `spin: bool` DELETED
```

Key method changes on `impl<const SPIN: bool> Model<SPIN>`:
- `nsta()` → `if SPIN { 2 * self.norb() } else { self.norb() }` — **compile-time constant, no branch in release**
- `norb()`, `natom()`, `nR()`, `dim_r()`, `atom_*()` — unchanged
- `orb_angular()` — uses `self.nsta()` internally, works generically

### 1.2 NO runtime dispatch enum

There is NO `AnyModel` enum. `Model<SPIN>` is the only Model type. Users specify spin at compile time:

```rust
let m = Model::<false>::tb_model(2, lat, orb, None)?;  // spinless
let m = Model::<true>::tb_model(2, lat, orb, None)?;   // spinful
```

### 1.3 Serde strategy

`Model<const SPIN: bool>` cannot derive `Deserialize` (const generics + serde derive issue).

- **Serialize**: manual impl, write `"spin": SPIN` field
- **Deserialize**: custom `Deserialize` impl reads `spin` field first, then dispatches internally. Both `Model<false>` and `Model<true>` get their own deserialize impls (duplicate code, ~30 lines each).
- Alternative: a private `ModelData` intermediate struct with `spin: bool`, deserialized first, then converted to `Model<SPIN>` with a compile-time check.

## 2. File-by-File Changes (ordered by dependency)

### Phase 1: Core (no downstream deps)

#### `src/model.rs`
- [ ] Define `Model<const SPIN: bool>` (remove `spin: bool` field)
- [ ] Update `nsta()`: `if SPIN { 2 * self.norb() } else { self.norb() }`
- [ ] Update `orb_angular()` — uses `self.nsta()`, no change needed
- [ ] Implement `Serialize` for `Model<SPIN>` manually (write `SPIN` as field)
- [ ] Implement `Deserialize` for `Model<SPIN>` — read spin in deserializer, verify against SPIN
- [ ] Remove `Serialize`/`Deserialize` derive from `Model`
- [ ] `Dimension` enum, `Gauge`, `SpinDirection` — unchanged

#### `src/generics.rs`
- [ ] `TryFrom<usize> for Dimension` — unchanged
- [ ] Any spin-related traits? Check. Probably none.

### Phase 2: Hot-path computation

#### `src/model_physics.rs`
- [ ] `gen_ham` → `impl<const SPIN: bool> Model<SPIN> { fn gen_ham(...) }`
  - `if SPIN` replaces `if self.spin` at lines 132, 134
  - Phase vector U0 dimension: compile-time
  - Gauge transform: already uses `for m + Zip` pattern, `nsta` from method
- [ ] `dos` → generic over SPIN, uses `self.nsta()` and `solve_band_all_parallel`
- [ ] No dispatch needed — `Model<SPIN>` is the only type

#### `src/velocity.rs`
- [ ] Make `Velocity` trait generic: `pub trait Velocity<const SPIN: bool>`
- [ ] `impl<const SPIN: bool> Velocity<SPIN> for Model<SPIN>`
- [ ] `gen_v` — `orb_sta` construction: `if SPIN { concatenate } else { to_owned }` → compile-time
- [ ] `UU` construction — uses `self.nsta()`, ok
- [ ] No dispatch needed

#### `src/solve_ham.rs`
- [ ] All solver methods → `impl<const SPIN: bool> Model<SPIN>`
- [ ] `solve_onek`, `solve_band_onek`, `solve_band_all`, `solve_all`, etc.
- [ ] Use `self.nsta()` everywhere (already done)
- [ ] All methods → generic over SPIN, no dispatch needed

#### `src/conductivity.rs`
- [ ] All berry curvature / Hall methods → generic over SPIN
- [ ] Change `if self.spin` → `if SPIN` at lines 547, 1003, 1300, 1458, 1633
- [ ] `build_spin_matrix` — this uses `SpinDirection` (Pauli x/y/z), **NOT** `self.spin`, unchanged

### Phase 3: Builders and constructors

#### `src/model_build.rs`
- [ ] `tb_model<const SPIN: bool>(...) -> Result<Model<SPIN>>`
  - Parameter `spin: bool` DELETED, replaced by const generic
  - Internal: `let nsta = if SPIN { 2 * norb } else { norb };`
  - rmatrix diagonal init (line 257): `if SPIN { ... duplicate ... }` → compile-time
- [ ] `set_hop`, `add_hop`, `add_element`, `set_onsite`, `add_onsite`, `set_onsite_one`
  - `update_hamiltonian!` macro (line 70): takes `$spin: bool` as first argument. Line 74: `if $spin { match $pauli { ... } } else { ... }`.
    With `SPIN` as const generic, `$spin` becomes `SPIN` (compile-time constant):
    - `SPIN == false` → compiler eliminates entire `if { match pauli }` block, leaves only `ham[[i,j]] = tmp`
    - `SPIN == true` → compiler eliminates `else` branch, keeps only `match pauli` (which IS necessary — x/y/z/None are independent of spinless/spinful)
  - Gates at lines 382, 502: `self.spin == false && pauli != SpinDirection::None` → `!SPIN && pauli != SpinDirection::None`
  - Matrix dimensions use `self.nsta()` — auto-correct
- [ ] `del_hop`, `shift_to_atom`, `move_to_atom`, `remove_orb`, `remove_atom` — generic
- [ ] `make_supercell<const SPIN: bool>` — internal loops use `SPIN` compile-time
- [ ] `reorder_atom` — uses `self.spin` at line 1021, 1108
- [ ] Output `spin: SPIN` at line 1553 for final Model construction
#### `src/wannier90.rs`
- [ ] `from_hr` → generic: `Model::<SPIN>::from_hr(...)` — verifies file spin matches SPIN, errors if not
- [ ] User must specify SPIN at call site. If spin is unknown, user manually checks file header first.
  ```rust
  // User code: explicit
  let m = Model::<true>::from_hr("wannier90_hr.dat")?;  // expects spinful
  let m = Model::<false>::from_hr("wannier90_hr.dat")?; // expects spinless, errors if spinors=T
  ```

#### `src/SKmodel.rs`
- [ ] Line 273: `Model::tb_model(self.dim_r, ..., self.spin, None)` → `SKmodel` itself needs SPIN

### Phase 4: Supporting modules

#### `src/surfgreen.rs`
- [ ] `surf_Green` struct: **KEEP** `spin: bool` field (not worth making generic)
  - Reason: constructed once, hot path is O(n³) matrix inversion, `if self.spin` is negligible
  - BUT: `from_Model` now takes `Model<SPIN>`, derive `spin = SPIN`
  - `from_Model<const SPIN: bool>(model: &Model<SPIN>, ...) -> Result<surf_Green>`
- [ ] `gen_ham_onek` — `self.spin` used for orb_phase doubling (line 213). Keep as-is.
- [ ] All other surf_Green methods — unchanged
- [ ] Add `from_model` generic: `from_Model<const SPIN: bool>(model: &Model<SPIN>, ...)`

#### `src/cut.rs`
- [ ] All methods → generic over SPIN
- [ ] `self.spin` → `SPIN` at lines 132, 154, 217, 256, 325, 451, 464, 476, 667, 678

#### `src/model_transform.rs`
- [ ] All methods → generic over SPIN
- [ ] `self.spin` → `SPIN` at lines 89, 158, 217, 427, 490, 503, 556, 597, 671

#### `src/geometry.rs`
- [ ] Methods → generic over SPIN
- [ ] `self.spin` → `SPIN` at lines 178, 238

#### `src/unfold.rs`
- [ ] `unfold` trait → `trait Unfold<const SPIN: bool>`
- [ ] `impl<const SPIN: bool> Unfold<SPIN> for Model<SPIN>`
- [ ] `self.spin` → `SPIN` at line 223

#### `src/output.rs`
- [ ] Output functions → generic over SPIN
- [ ] `self.spin` → `SPIN` at lines 422, 460, 494

#### `src/magnetic_field.rs`
- [ ] Check if `self.spin` is referenced (from grep, it's not directly). Uses `nsta()`.

### Phase 5: Tests, benches, examples

#### `src/lib.rs` (tests module only — NO Python bindings in this codebase)
- [ ] `build_small()` → returns `Model<false>`
- [ ] All test functions using Model: add type annotation or let inference work

#### `benches/bench_main.rs`
- [ ] `build_small()` → returns `Model<false>`
- [ ] `build_small_spinful()` → returns `Model<true>`
- [ ] `build_medium()`, `build_large()` → returns `Model<false>` (all supercells are spinless)
- [ ] Benchmark functions: accept `Model<SPIN>` directly (spinless builds for most, spinful for spin-orbit benchmarks)

#### `examples/*.rs`
- [ ] Each example: `tb_model::<true>(...)` or `tb_model::<false>(...)` as appropriate

## 3. Migration Order (for parallel agents)

```
Agent 1: src/model.rs          (core types, serde)
Agent 2: src/model_physics.rs  (gen_ham, dos)
Agent 3: src/velocity.rs       (Velocity trait, gen_v)
Agent 4: src/solve_ham.rs      (all solvers)
Agent 5: src/conductivity.rs   (berry, Hall, spin Hall)
Agent 6: src/model_build.rs    (tb_model, set_hop, add_hop, make_supercell, update_hamiltonian!)
Agent 7: src/wannier90.rs      (from_hr generic, verifies SPIN)
Agent 8: src/surfgreen.rs      (surf_Green keeps bool, from_Model generic)
Agent 9: src/cut.rs + src/model_transform.rs + src/geometry.rs
Agent 10: src/unfold.rs + src/output.rs + src/SKmodel.rs
Agent 11: benches/ + examples/ + tests in src/lib.rs
```

Agent 1 must complete first (core types). Agents 2-11 can then run in parallel.

## 4. Risk Points

- **Serde**: `Model<SPIN>` cannot derive Deserialize. Must use custom implementation. Internal `ModelData` helper struct may simplify this.
- **Trait objects**: `Velocity<SPIN>` trait cannot be made into trait object. All velocity usage is static dispatch already, so this is fine.
- **`update_hamiltonian!` macro**: Takes `$spin: bool` parameter. Replace with `SPIN` const generic in calling context. Macro itself unchanged.
- **surf_Green `spin` field**: Keep as bool. The `from_Model` function bridges `Model<SPIN>` to surf_Green.
- **Backward compatibility**: All existing code that constructs `Model` with `spin: bool` breaks. Provide `tb_model_any(spin, ...)` for migration.

## 5. Key Design Decisions

1. **`surf_Green` stays with `spin: bool`** — its hot path is O(n³) matrix inversion; the spin branch overhead is negligible.
2. **`Model<SPIN>` is the only type** — no runtime dispatch enum. Users specify SPIN at compile time.
3. **Default `SPIN = false`** allows `Model` without turbofish for spinless cases.
4. **Serde**: Custom serialize/deserialize on `Model<SPIN>`. Deserialization verifies file spin matches SPIN.
5. **`from_hr`**: Generic over SPIN, verifies file matches. Users must know spin at compile time.

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
- Spin‑ful models double the orbital basis (Pauli matrices in spin space)
- Position matrix `rmatrix` is essential for velocity operator calculations
- Use `Gauge::Atom` or `Gauge::Lattice` for consistent phase conventions
- k‑points are in reciprocal lattice coordinates (fractions of reciprocal vectors)

## Performance Tips
- Enable link‑time optimization in release builds (`lto = "fat"` in Cargo.toml)
- Use Intel MKL for best numerical performance on Intel/AMD CPUs
- Parallelize over k‑points with `rayon` for embarrassingly parallel calculations
- For large supercells, consider iterative eigensolvers for partial spectra