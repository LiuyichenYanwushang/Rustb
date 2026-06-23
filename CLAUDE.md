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
| `velocity.rs` | `Velocity` trait — `gen_v()` with safe rmatrix commutator; `gen_v_projected()` fuses direction-weight projection |
| `conductivity.rs` | `BerryCurvature` trait — AHC, nonlinear Hall |
| `quantum_geometry.rs` | `QuantumGeometry` trait — QGT, quantum metric |
| `surfgreen.rs` | Surface Green's function (Sancho-Rubio iterative method) |
| `wannier90.rs` | Wannier90 I/O, returns `Model<SPIN, 3, HasRMatrix>` when `_r.dat` present |
| `ndarray_lapack.rs` | LAPACK bindings + safe `zaxpy()` BLAS wrapper |
| `lib.rs` | Crate root, re-exports, integration tests |
| `SKmodel.rs` | Slater-Koster parameterized models (`SlaterKosterModel`, `SkAtom`, `SkParams`) |
| `fermi_surface.rs` | `FermiSurface`/`FermiSurfacePlane` traits (marching squares/tetrahedra → gnuplot); BXSF export (`BxsfExport` trait → XCrySDen/FermiSurfer); `write_spin_frmsf` free fn (spin‑split FRMSF for altermagnets) |
| `magnetic_field.rs` | Uniform magnetic field via Peierls substitution (`MagneticField` trait) |
| `unfold.rs` | Band unfolding for supercell→primitive projection (`Unfold` trait) |
| `optical_conductivity.rs` | Frequency-dependent optical conductivity & optical Hall |
| `orbital_angular.rs` | Orbital angular momentum operator |
| `solve_ham.rs` | Parallel diagonalization (`solve_all_parallel`, `solve_range_onek`) |
| `kpoints.rs`/`kpath.rs`/`kplane.rs` | k-mesh/k-path/k-plane generation |
| `output.rs` | gnuplot plotting (`show_band`, `show_surf_state`, `draw_heatmap`, `show_fermi_surface`) |
| `model_transform.rs` | Supercell construction, orbital removal/reordering, `shift_to_atom` |
| `atom_struct.rs` | `Atom` and `OrbProj` types |
| `error.rs` | `TbError` enum, `Result` alias |
| `generics.rs` | Numeric abstractions (`hop_use`, `usefloat`), `SpinDirection::from_index` |

### Conventions

- **k-points**: Fractional reciprocal coordinates; phase = `exp(2πi k·R)`.
- **Orbital positions**: Fractional coords (columns of `orb`).
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

FRMSF loop order (Fortran): `do ibnd; do ik1; do ik2; do ik3; write` —
`ik3` (z) fastest, `ibnd` outermost. The helper `format_frmsf_block`
buffers all values into one `String` before a single `write_all`.

## Hot-Path Optimization Notes

### `zaxpy` candidates in conductivity code

`src/ndarray_lapack.rs::zaxpy(alpha, x, y)` is a safe wrapper for BLAS `y += alpha * x` on contiguous `Complex<f64>` slices. It is appropriate for hot loops that accumulate full dense matrices with one scalar weight per block.

Good candidates:

- `src/conductivity.rs::berry_curvature_n_onek`: the direction projections currently written with `Zip`:
  - `jmat += current_dir[d] * J[d, :, :]`
  - `vmat += dir_2[d] * v[d, :, :]`
  - spin-current branches after `anti_comm(...)`: `jmat += 0.5 * current_dir[d] * ac`
- `src/conductivity.rs::berry_curvature_dipole_n_onek`: replace allocation-heavy patterns:
  - `v0 = v0 + v[d, :, :].to_owned() * dir_3[d]`
  - after weighting `J[d, :, :]` and `v[d, :, :]`, prefer accumulating directly into final 2D `J`/`v` matrices instead of scaling every 3D slice and then `sum_axis(Axis(0))`.
- `src/conductivity.rs::berry_connection_dipole_onek`: direction projections:
  - `v_1 += current_dir[d] * v[d, :, :]`
  - `v_2 += dir_2[d] * v[d, :, :]`
  - `v_3 += dir_3[d] * v[d, :, :]`
  - spinful `s_1/s_2/s_3 += weight[d] * S[d, :, :]`

Implementation guidance:

- Use `Complex::new(weight, 0.0)` as the `zaxpy` scalar.
- Only call `zaxpy` when both source and destination are standard contiguous slices; use `.as_slice().unwrap()` / `.as_slice_mut().unwrap()` consistently with `gen_ham` and `gen_v`.
- For temporary outputs from `anti_comm`, bind the owned `Array2` first, then call `zaxpy` on its slice.
- Consider a small local helper such as `accumulate_dir(dst, blocks, weights)` to reduce duplicated projection code, but do not introduce a broad abstraction.

### Optical conductivity / Optical Hall notes

`src/optical_conductivity.rs` has related projection loops:

- `optical_geometry_n_onek`: `J = sum_d dir_1[d] * v[d, :, :]` and `v = sum_d dir_2[d] * v[d, :, :]` are direct `zaxpy` candidates.
- `optical_conductivity_all_direction`: most expensive inner work computes frequency-dependent row dot products like `re_xx.row(i).dot(U1.row(i))`, `im_xy.row(i).dot(UU.row(i))`, then dots with the Fermi vector. These are NOT direct `zaxpy` candidates because each element has a frequency-dependent denominator.

For Optical Hall, prefer optimizing by reducing allocations and using BLAS-backed matrix/vector operations where possible:

- Avoid repeated `Array1::from_vec(...)` inside each frequency loop when a reusable buffer or direct accumulation will do.
- Consider expressing per-frequency row-dot results as matrix/vector style operations if it preserves the elementwise denominator formula.
- Do not replace the row-dot denominator loops with `zaxpy`; that would only be valid for uniform scalar weights over an entire contiguous block.

### SIMD/autovec opportunities

If the goal is AVX2/AVX512-style codegen, first make loops simple, contiguous, and branch-light. Rust/LLVM is much more likely to vectorize loops over `&[f64]` / `&[Complex<f64>]` slices than loops over `ndarray` indexed views, transposed views, or closures with allocation.

Conductivity candidates:

- `berry_curvature_n_onek`: the final `omega_n` loop can be made more autovec-friendly:
  - avoid `a.powi(2)` in the inner loop; use `let de = band_i - band_j; de * de + eta2`.
  - hoist `eta2 = eta * eta` and `band_i = band[i]`.
  - iterate over contiguous row slices when possible instead of repeated `im_row[[j]]` indexing.
  - avoid allocating `AA` and `im` if practical; compute `-2.0 * (A1[i,j] * A2[j,i]).im` directly in the row loop, or keep an owned standard-layout numerator matrix.
- `berry_curvature_dipole_n_onek`: build `U0` with simple contiguous row loops and replace `powi(2)` in denominators with explicit multiplication. The final per-band dot `A1.row(i).dot(A2.col(i))` is a reduction; use contiguous owned/transposed storage if this becomes a hotspot.
- `berry_connection_dipole_onek`: the triple loops for `B` and `C` are not good autovec targets because of indirect 2D indexing and loop-carried scalar reductions. If they dominate runtime, prefer algebraic refactoring into matrix products or precomputed denominator-weighted matrices rather than expecting LLVM to vectorize them.
- Fermi-surface accumulation in `Nonlinear_Hall_conductivity_Extrinsic` and `Nonlinear_Hall_conductivity_Intrinsic` allocates a fresh `Array1` per `(energy, omega)` item in the Rayon fold. For many `mu` values, rewrite the fold to mutate a local accumulator with a simple indexed loop over `mu`; this reduces allocation and gives LLVM a straightforward `f64` slice loop.

Optical conductivity / Optical Hall candidates:

- `optical_geometry_n_onek`: after forming `UU` and `U1`, the per-frequency calculations are row-wise dot products. Avoid `map(...).collect()` plus `Array1::from_vec(...)` inside the frequency loop; fill preallocated output rows with explicit loops or use BLAS-backed `dot`/`gemv` if the expression can be arranged as matrix-vector work.
- `optical_conductivity_all_direction`: the repeated blocks for `xx/yy/zz/xy/yz/xz` duplicate the same row-dot pattern. A small helper that fills one output component from `(kernel, weight_matrix, fermi_dirac)` can remove allocation and expose a simple contiguous inner loop. Keep separate helpers for symmetric (`re_*` with `U1`) and antisymmetric/Hall (`im_*` with `UU`) pieces.
- Avoid transposed or reversed-axis views in inner loops when the next operation scans rows. If a transposed operand is reused many times, materialize one standard-layout owned copy once; this can be faster than repeated strided access and gives BLAS/LLVM a better chance.

Compiler/codegen notes:

- To actually get AVX512 from compiler-generated loops, benchmark with native CPU flags, for example `RUSTFLAGS="-C target-cpu=native"` or a specific `-C target-feature=+avx512f` when appropriate. BLAS backends such as MKL/OpenBLAS usually dispatch optimized kernels independently of Rust autovec.
- Do not rely on autovec for floating-point reductions unless benchmarks confirm it. Strict FP semantics can limit reduction vectorization; BLAS dot/gemv/gemm or explicit portable SIMD may be better for large reductions.

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
