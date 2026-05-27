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
3. **`SlaterKosterModel<const SPIN, const DIM>`** — parallel const generics.
4. **`RMatrixData` trait** — `HasRMatrix` (Deref to Array4) and `NoRMatrix` (ZST). `NoRMatrix` model has literally no rmatrix field in memory.
5. **Safe `zaxpy`** — in `ndarray_lapack.rs`, wraps `blas::zaxpy` FFI; all call sites are safe Rust.
6. **`update_hamiltonian!` / `add_hamiltonian!` macros** — take `$spin` const generic; branches eliminated at compile time.

## Doc conventions

- `Model::<false, 2>` / `Model::<true, 3>` with turbofish for SPIN and DIM.
- `Model<SPIN, DIM, HasRMatrix>` when rmatrix is present.
- `surf_Green` retains `spin: bool` and `dim_r: usize` fields (derived from const generics via `from_Model`).

---

# v0.7.0: Const-Generic Spin

`Model<const SPIN: bool = false>` replaced the `spin: bool` runtime field. Superseded by v0.8.0.

---

## Project Overview

Rustb is a Rust library for tight-binding model calculations in condensed matter physics. It computes band structures, density of states, transport properties (Hall conductivity, nonlinear responses), topological invariants (Chern numbers, Wilson loops), and surface states using Green's functions.

- **MSRV**: 1.90.0 (edition 2024)
- **Repo**: https://github.com/LiuyichenYanwushang/Rustb
- **Error handling**: Uses `thiserror` for `TbError` enum.
- **Docs**: `katexit` renders LaTeX in rustdoc; `docs-header.html` for custom CSS.

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
Velocity  (src/velocity.rs)     → v_α(k) operator
  └─ BerryCurvature (src/conductivity.rs) → AHC, nonlinear Hall
       └─ QuantumGeometry (src/quantum_geometry.rs) → QGT, quantum metric
FermiSurface / FermiSurfacePlane (src/fermi_surface.rs)
```

All trait impls: `impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Trait for Model<SPIN, DIM, R>`.

### Key Source Files

| File | Purpose |
|------|---------|
| `model.rs` | `Model<SPIN, DIM, R>` struct, `RMatrixData` trait, `HasRMatrix`/`NoRMatrix`, `Gauge`, serde |
| `model_build.rs` | Builder: `tb_model()`, `set_hop()`, `make_supercell()`, macros |
| `model_physics.rs` | `gen_ham()` (Bloch Hamiltonian), `dos()` |
| `velocity.rs` | `Velocity` trait — `gen_v()` with safe rmatrix commutator |
| `conductivity.rs` | `BerryCurvature` trait — AHC, nonlinear Hall |
| `quantum_geometry.rs` | `QuantumGeometry` trait — QGT, quantum metric |
| `surfgreen.rs` | Surface Green's function (Sancho-Rubio iterative method) |
| `wannier90.rs` | Wannier90 I/O, returns `Model<SPIN, 3, HasRMatrix>` when `_r.dat` present |
| `ndarray_lapack.rs` | LAPACK bindings + safe `zaxpy()` BLAS wrapper |
| `lib.rs` | Crate root, re-exports, integration tests |

### Conventions

- **k-points**: Fractional reciprocal coordinates; phase = `exp(2πi k·R)`.
- **Orbital positions**: Fractional coords (columns of `orb`).
- **Hermitian conjugates**: Auto-generated by `set_hop`/`add_hop`.
- **Public API**: `pub use X::*` re-exports from `lib.rs`.

## BLAS/LAPACK Backend

- **Intel MKL** (best x86_64 perf): `--features intel-mkl-static`
- **OpenBLAS**: `--features openblas-static`
- **netlib**: `--features netlib-static`

---

## Large-Scale Refactoring Guidelines

### Use multiple parallel agents, NOT batch sed

When making a change that touches many files:

1. **Core file first**: Manually edit the central struct/trait definition.
2. **Parallel agents by file group**: Launch 3-4 agents simultaneously for distinct file sets.
3. **Each agent gets explicit instructions**: Patterns, replacements, file paths.
4. **Never use sed/Python for batch refactoring**: Creates scope issues and silent breakage.

### Don't wait — review in parallel

While agents run, review code already modified. Check for non-exhaustive matches, missed impl blocks, type mismatches.

### Commit after each successful build

`git add -A && git commit -m "intermediate: <description>"` prevents data loss.
