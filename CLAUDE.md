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
- **Lattice vectors**: Stored in `Model::lat` (columns = real-space vectors).
  Reciprocal vectors via `Model::rec_lat()` (rows = reciprocal vectors,
  formula `B = 2π·(Aᵀ)⁻¹`).
- **Eigenbasis transformation** (`band` ↔ `evec` from `eigh`): the codebase
  convention is `U^T · O · U^*` (transpose, NOT conjugate‑transpose, on the
  left; conjugate on the right).  See `berry_curvature_n_onek` and
  `compute_tetra_primitives`.  Verify with the test `evec_transform_sanity`
  which checks `diag(U^T H U^*) == band` numerically.
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

## Planned: Tetrahedron Integration for Berry / Quantum-Geometry Kernels

> **Status**: design plan only. Do not treat the current
> `tetrahedron_volume_integrate()` as a high-accuracy Berry-curvature
> integrator; it linearly interpolates the final scalar integrand, which is
> unsafe near small gaps because Berry/QGT formulas contain energy denominators.

### Goal

Implement a simplex integration path for Berry curvature, quantum metric,
Berry-curvature dipole, Berry-connection dipole, and related response kernels
under this narrower assumption:

1. Band energies `E_n(k)` are linearly interpolated inside each simplex.
2. Gauge-invariant velocity kernels are linearly interpolated inside each
   simplex.
3. The nonlinear denominators are evaluated from the interpolated energies at
   quadrature points; the final Berry/QGT scalar is **not** linearly
   interpolated from vertex values.

This keeps the singular / near-singular structure from
`1 / (E_n - E_m)^p` in the integrand instead of hiding it inside a vertex
average.

### Basic formulas

At each k-point, use `gen_v_projected(k, Gauge::Atom, directions)` and
diagonalize `H(k)`.

For directions `a`, `b`, define band-basis velocity matrices

```text
A^a_nm(k) = <u_n(k)|v_a(k)|u_m(k)>
A^b_mn(k) = <u_m(k)|v_b(k)|u_n(k)>
```

Do **not** interpolate `A^a_nm` directly: eigenvectors have arbitrary U(1)
phases, so individual matrix elements are not smooth or gauge invariant.
Instead store and interpolate the product

```text
K^ab_nm(k) = A^a_nm(k) A^b_mn(k)
```

which is invariant under independent phase rotations of bands `n` and `m`
when the bands are isolated.

For quantum geometry:

```text
G^ab_n(k) = sum_{m != n} K^ab_nm(k) / ((E_n(k) - E_m(k))^2 + eta^2)
g^ab_n(k) = Re G^ab_n(k)
Omega^ab_n(k) = -2 Im G^ab_n(k)
```

For DC Berry curvature this is the same kernel with `eta -> 0` when safe.
For optical / finite-frequency variants, keep the denominator form used by the
calling routine, for example

```text
D_nm(k, omega, eta) =
    1 / ((E_n(k) - E_m(k))^2 - (omega + i eta)^2)
```

and evaluate `K_nm(k) * D_nm(k, omega, eta)` at quadrature points.

For velocity-weighted quantities such as Berry-curvature dipole, also store
the diagonal projected velocity

```text
v^c_n(k) = <u_n(k)|v_c(k)|u_n(k)> = dE_n / dk_c
```

and linearly interpolate `v^c_n(k)` as another vertex quantity. The quadrature
point integrand is then built as

```text
v^c_n(q) * Omega^ab_n(q)
```

or with the exact formula required by the target response.

### Simplex interpolation

For a simplex with vertices `r = 0..d` and barycentric coordinates `lambda_r`,
interpolate only primitive vertex data:

```text
E_n(q)       = sum_r lambda_r E_{r,n}
K^ab_nm(q)  = sum_r lambda_r K^ab_{r,nm}
v^c_n(q)    = sum_r lambda_r v^c_{r,n}
```

Then evaluate the response formula at `q`:

```text
Delta_nm(q) = E_n(q) - E_m(q)
G^ab_n(q)   = sum_{m != n} K^ab_nm(q) / (Delta_nm(q)^2 + eta^2)
```

The simplex contribution is a weighted quadrature sum:

```text
I_simplex = V_simplex * sum_q w_q F(q)
```

Start with fixed symmetric simplex quadrature rather than closed-form analytic
integration. Suggested rules:

- 2D triangles: degree-2 or degree-3 symmetric Gaussian rule.
- 3D tetrahedra: 4-point degree-2 rule as the first implementation; add
  higher-order rules later if convergence needs it.

This is easier to verify and works for all denominator variants. Analytic
integration of `linear numerator / (linear energy difference)^p` can be added
later only if benchmarks show the quadrature is a bottleneck.

### Band tracking inside the simplex

The interpolation requires the same band label at every vertex of the simplex.
Energy sorting alone can fail near crossings. Use eigenvector overlap to align
vertex bands.

For simplex vertex `0` as the local reference and another vertex `r`, compute

```text
O_nm = | <u_n(k_0) | u_m(k_r)> |^2
```

Find a one-to-one permutation `p_r` maximizing

```text
sum_n O_{n,p_r(n)}
```

Then reorder all vertex data from vertex `r` into the reference labeling:

```text
E_{r,n}          <- E_raw_{r,p_r(n)}
K_{r,nm}         <- K_raw_{r,p_r(n),p_r(m)}
vdiag_{r,n}      <- vdiag_raw_{r,p_r(n)}
eigenvector_{r,n} <- eigenvector_raw_{r,p_r(n)}
```

For small `nsta`, an exact assignment search or Hungarian algorithm is fine.
A row-wise greedy maximum is not sufficient because two reference bands can
choose the same target band.

Optional global tracking can be added as a preprocessing and diagnostic pass:
start from Gamma, propagate labels through nearest-neighbor k-mesh edges using
the same overlap assignment, and record loop/path conflicts. Do not rely on
global tracking alone. Each simplex should still perform a local consistency
check because Berry/QGT interpolation only needs local smoothness.

### Degeneracy and reliability checks

Single-band Berry curvature / QGT is not reliable when the target band is not
isolated. Mark a simplex as unsafe for single-band interpolation if any of the
following holds:

```text
min_{q estimate, m != n} |E_n - E_m| < gap_tol
max_assignment_overlap < overlap_tol
different neighbor paths imply different band permutations
```

Initial practical checks can use only vertex data:

```text
min_vertex_gap = min_{r,n != m} |E_{r,n} - E_{r,m}|
```

If `min_vertex_gap < gap_tol`, choose one of these policies:

1. Use finite `eta` and emit a warning / diagnostic count.
2. Refine the k-mesh around the simplex.
3. Merge near-degenerate bands into a band group and use a non-Abelian /
   occupied-subspace expression instead of a single-band formula.

Do not silently force a single-band label through a near-degenerate subspace.
The eigenvectors inside that subspace can rotate arbitrarily, making
single-band `K_nm` non-smooth even when the subspace is smooth.

### Data structures to add

Add internal structs in `src/tetrahedron.rs` or a new module such as
`src/simplex_response.rs`:

```rust
struct VertexKernel {
    band: Array1<f64>,                 // nsta
    kernel_ab: Array2<Complex<f64>>,   // nsta x nsta, K_nm
    vdiag_c: Option<Array1<f64>>,      // nsta, for dipoles
    evec: Array2<Complex<f64>>,        // norb x nsta, for overlap tracking
}

struct TrackedSimplex {
    vertices: Vec<VertexKernel>,       // already label-aligned
    volume: f64,
    diagnostics: SimplexDiagnostics,
}

struct SimplexDiagnostics {
    min_gap: f64,
    min_assignment_overlap: f64,
    tracking_conflict: bool,
}
```

Keep these internal at first. Public API should expose only higher-level
methods once the numerical behavior is tested.

### Implementation sequence

1. Add a helper that computes one-k-point primitive data:
   `band`, `evec`, projected band-basis velocity matrices, `K_nm`, and optional
   diagonal velocity.
2. Add assignment/permutation utilities based on overlap matrices.
3. Add simplex builders for the existing 2D triangle and 3D tetrahedron
   decompositions, reusing the current cell ordering and volume factors.
4. Add fixed simplex quadrature rules and interpolation helpers.
5. Implement a generic quadrature evaluator for kernels of the form
   `sum_m K_nm(q) * denom(E_n(q)-E_m(q))`.
6. Wire the evaluator first into quantum geometry / Berry curvature at fixed
   `mu` or full occupied-band sums, where the formula is simplest.
7. Extend to Berry-curvature dipole and optical kernels after the base path has
   convergence tests.
8. Add diagnostics returned or logged: number of unsafe simplexes, minimum gap,
   minimum assignment overlap, and optional global-tracking conflicts.

### Tests and validation

Minimum tests before using this path as a default:

- Constant / linear toy kernels where simplex quadrature has an exact answer.
- Two-band massive Dirac model with known Berry curvature trend; compare mesh
  convergence against dense direct summation.
- Gauge phase test: multiply eigenvectors at vertices by random phases and
  verify `K_nm`-based integration is unchanged.
- Band reordering test: artificially permute vertex bands and verify overlap
  tracking restores the same integral.
- Near-degeneracy diagnostic test: force a small gap and verify the unsafe
  simplex count is nonzero.

Default behavior should remain conservative until these tests pass. The current
direct k-mesh summation with finite `eta` is still the reference fallback for
Berry/QGT quantities.

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

---

# Tetrahedron Integration: Current Status & Known Issues

> **Last updated**: 2026-06-26.

## What's implemented

### `Hall_conductivity_tetra` (intrinsic AHC)
- 2D: 3-point triangle quadrature (approximate for rational integrand)
- 3D: Blochl sub-tet decomposition with analytic `compute_occ_omega`
- T>0: thermal convolution of T=0 result
- **Parallelized**: cell loops use `rayon::par_iter().fold().reduce()`

### `Nonlinear_Hall_conductivity_Extrinsic_tetra` (extrinsic NLH)
- 2D: triangle → line-segment Fermi-surface cut with analytic 1D integral (`segment_integral_1d`, elementary antiderivatives for η>0, ln/1/d for η=0)
- 3D: tetrahedron → triangle Fermi-surface cut with divided-difference K_αβ weights
- T>0: thermal convolution of T=0 result
- **Parallelized**: cell loops use `rayon::par_iter().fold().reduce()`

## Known issues

### 2D convergence oscillates
Comparing `Nonlinear_Hall_conductivity_Extrinsic_tetra` against the reference (`Nonlinear_Hall_conductivity_Extrinsic` at T=0, which uses `tetrahedron_integrate_2d`), the max-diff does not decrease monotonically with mesh size:
- nk=30: 3.99e-3, nk=40: 2.28e-2, nk=50: 3.25e-3, nk=100: 6.82e-3

Root causes (not yet fixed):
1. **Different triangulations**: The reference `tetrahedron_integrate_2d` uses `nx.saturating_sub(1)` (no periodic wrap, missing boundary cells). The tetra method uses periodic wrapping (`(ix+1)%nx`), covering the full BZ.
2. **Different interpolation strategies**: Reference interpolates the scalar `omega[n] = v^c_n * Ω_n` linearly, then applies Blochl δ-function weights. Tetra method does band-pair decomposition inside each simplex: interpolates u (=v^c), P (=Im K_nm), and d (=E_n-E_m) separately, then evaluates u·P/(d²+η²) analytically.
3. Both methods have different discretization error patterns at moderate mesh sizes. In 2D, there are only 2 triangles per cell (vs 5 tets in 3D), making the error more oscillatory.

3D converges well: nk=6: 1.84e-3, nk=8: 2.21e-3, nk=10: 5.96e-4 (monotonically improving overall).

### No band tracking
`TetraKPoint` stores `evec` but integration uses raw band indices. If band ordering swaps inside a simplex, E_n and K_nm interpolation is across bands. Haldane/graphene tests may avoid this because bands are sufficiently isolated.

## Parallelization pattern
Both tetra methods follow the same pattern:
```rust
let result = (0..ncell).into_par_iter()
    .fold(|| Array1::zeros(n_mu), |mut acc, cell_id| {
        accum_xxx_cell_xx(..., &mut acc);  // writes directly into thread-local acc
        acc
    })
    .reduce(|| Array1::zeros(n_mu), |mut a, b| { a += &b; a });
```
- k-point computation (`compute_tetra_primitives`) is parallelized via `into_par_iter()` on k-vector iteration
- Cell accumulation helpers take `&mut Array1<f64>` and write directly (no per-cell allocation)
- Each thread builds its own `Array1<f64>` accumulator; final reduce by `a += &b`
- Floating-point addition order varies between serial/parallel runs → last few bits may differ (not physically significant)
