# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

# v0.8.0: Const-Generic Dimension (in progress)

> **Status**: IN PROGRESS. `Model<const SPIN: bool = false, const DIM: usize = 3>` replacing the `dim_r: Dimension` runtime field.

## How Model<SPIN, DIM> works

```rust
pub struct Model<const SPIN: bool = false, const DIM: usize = 3> {
    // NO dim_r field
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    // ...
}
```

- `DIM` defaults to 3 (most general case).
- `dim_r()` returns `DIM` directly (no field access).
- `tb_model(lat, orb, atoms)` — NO `dim_r` parameter; DIM comes from the const generic.
- Construction: `Model::<false, 2>::tb_model(lat, orb, None)` for 2D, `Model::<true>::tb_model(...)` for 3D spinful.
- `Dimension` enum kept for backward compat (serde), but NOT stored in Model.
- Hot path `match self.dim_r { Dimension::one => ... }` → `match DIM { 1 => ..., 2 => ..., 3 => ... }` — compiler eliminates dead branches.

### Key design decisions

1. **No runtime dimension field** — `DIM` is a const generic; array shapes that use it benefit from compile-time constants.
2. **`surf_Green` stores `dim_r: usize`** — same pattern as SPIN: const generic converted to runtime field.
3. **`SlaterKosterModel<const SPIN, const DIM>`** — parallel generics, `new()` no longer takes `dim_r`.

---

# v0.7.0: Const-Generic Spin (DONE)

`Model<const SPIN: bool = false>` replaced the `spin: bool` runtime field. All runtime `if self.spin` branches on hot paths are compile-time eliminated.

### Doc conventions (v0.7.0+)

- Use `Model::<false, 2>` / `Model::<true, 3>` with turbofish for both SPIN and DIM.
- `surf_Green` retains `spin: bool` and `dim_r: usize` fields (derived from const generics).

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
cargo test                           # all tests (generates PDFs)
cargo test graphene                  # single test
cargo test -- --nocapture 2>&1 | head -100
```

Tests generate PDF plots via gnuplot (`pdfcairo` terminal). `sudo apt install gnuplot` on Ubuntu.

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

- **`Model<const SPIN: bool = false, const DIM: usize = 3>`**: Central TB model. No runtime dimension or spin fields.
- **`Gauge`**: Velocity operator convention — `Atom` (orbital positions in phase) or `Lattice` (only R vectors).
- **`Dimension`**: Enum `one=1/two=2/three=3` for serde compat, NOT stored in Model.
- **`SpinDirection`**: `X/Y/Z` Pauli matrices, `None` = identity (no spin projection).
- **`surf_Green`**: Non-generic struct storing `spin: bool` and `dim_r: usize` (converted from const generics).

### Trait Hierarchy

```
Velocity  (src/velocity.rs)     → v_α(k) operator
  └─ BerryCurvature (src/conductivity.rs) → AHC, nonlinear Hall
       └─ QuantumGeometry (src/quantum_geometry.rs) → QGT, quantum metric
FermiSurface / FermiSurfacePlane (src/fermi_surface.rs)
```

All trait impls use `impl<const SPIN: bool, const DIM: usize> Trait for Model<SPIN, DIM>`.

### Key Source Files

| File | Purpose |
|------|---------|
| `model.rs` | `Model<SPIN, DIM>` struct, `Gauge`, `Dimension`, `SpinDirection`, serde |
| `model_build.rs` | Builder: `tb_model()`, `set_hop()`, `add_hop()`, `update_hamiltonian!`, `add_hamiltonian!` |
| `model_physics.rs` | `gen_ham()` (Bloch Hamiltonian), `gen_v()` (velocity), `solve_all_parallel()`, DOS |
| `velocity.rs` | `Velocity` trait |
| `conductivity.rs` | `BerryCurvature` trait |
| `quantum_geometry.rs` | `QuantumGeometry` trait |
| `surfgreen.rs` | Surface Green's function (Sancho-Rubio iterative method) |
| `wannier90.rs` | Wannier90 I/O |
| `SKmodel.rs` | Slater-Koster tight-binding |
| `fermi_surface.rs` | Fermi surface traits, marching squares/tetrahedra |
| `lib.rs` | Crate root, re-exports, integration tests |

### Conventions

- **k-points**: Fractional reciprocal coordinates; phase = `exp(2πi k·R)` with R in integer units.
- **Orbital positions**: Fractional coords (columns of `orb`).
- **Hermitian conjugates**: Auto-generated (`-R` vector + conjugated amplitude) by `set_hop`/`add_hop`.
- **Public API**: `pub use X::*` re-exports from `lib.rs`; users import from `rustb::` directly.

## BLAS/LAPACK Backend

- **Intel MKL** (best x86_64 perf): `--features intel-mkl-static`
- **OpenBLAS**: `--features openblas-static`
- **netlib**: `--features netlib-static`

Without features, uses default system BLAS.

---

## Large-Scale Refactoring Guidelines (from v0.8.0 experience)

### Use multiple parallel agents, NOT batch sed

When making a change that touches many files (like adding a const generic parameter):

1. **Core file first**: Manually edit the central struct/trait definition yourself.
2. **Parallel agents by file group**: Launch 3-4 agents simultaneously, each responsible for a distinct set of files:
   - Agent 1: The builder/constructor file (most complex)
   - Agent 2: Physics/trait implementation files (medium complexity)
   - Agent 3: Test files (mechanical changes)
   - Agent 4: Example files (mechanical changes)
3. **Each agent gets explicit instructions**: What patterns to find, what to replace with, file paths.
4. **Never use sed/Python for batch refactoring**: It creates scope issues, name conflicts, and silent breakage. Agents understand context and can handle edge cases.

### Don't wait — review in parallel

While agents are running, review the code that other agents have already modified. Don't sit idle. Check for:
- Non-exhaustive match patterns (enum → usize conversion needs `_ => unreachable!()` arms)
- Impl blocks that were missed
- Type mismatches from default generic parameters

### Commit after each successful build

After `cargo check` passes, commit immediately with `git add -A && git commit -m "intermediate: <description>"`. This prevents data loss from `git checkout`.
