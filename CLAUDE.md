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
Velocity  (src/velocity.rs)        → v_α(k) operator
  ├─ BerryCurvature (src/conductivity.rs) → AHC, spin Hall, nonlinear Hall
  │    ├─ intrinsic NLH (src/conductivity.rs)
  │    └─ extrinsic NLH (src/conductivity.rs)
  └─ QuantumGeometry (src/quantum_geometry.rs) → QGT, quantum metric

FermiSurface / FermiSurfacePlane (src/fermi_surface.rs)

Berry (src/geometry.rs)            → Wilson loops, Berry phase, Wannier centres
CutModel (src/cut.rs)              → slab/ribbon (cut_piece), dot (cut_dot)

MagneticField (src/magnetic_field.rs)
Unfold (src/unfold.rs)
```

All trait impls: `impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Trait for Model<SPIN, DIM, R>`.

### Key Source Files

| File | Purpose |
|------|---------|
| `model.rs` | `Model<SPIN, DIM, R>` struct, `RMatrixData` trait, `HasRMatrix`/`NoRMatrix`, `Gauge`, serde |
| `model_build.rs` | Builder: `tb_model()`, `set_hop()`, `make_supercell()`, macros |
| `model_physics.rs` | `gen_ham()` (Bloch Hamiltonian), `dos()` |
| `velocity.rs` | `Velocity` trait — `gen_v()` with safe rmatrix commutator; `gen_v_projected()` fuses direction-weight projection |
| `conductivity.rs` | `BerryCurvature` trait — AHC, spin Hall, nonlinear Hall (intrinsic + extrinsic) |
| `quantum_geometry.rs` | `QuantumGeometry` trait — quantum geometric tensor, quantum metric |
| `geometry.rs` | `Berry` trait — Berry phase, Wilson loops, Wannier centres, hybrid Wannier functions |
| `surfgreen.rs` | Surface Green's function (Sancho-Rubio iterative method) |
| `wannier90.rs` | Wannier90 I/O, returns `Model<SPIN, 3, HasRMatrix>` when `_r.dat` present |
| `tetrahedron.rs` | Blochl tetrahedron integration (AHC, NLH extrinsic/intrinsic); 2D triangle + 3D tetrahedron |
| `cut.rs` | `CutModel` trait — slab (`cut_piece`), dot/edge (`cut_dot`) from bulk models |
| `ndarray_lapack.rs` | LAPACK bindings + safe `zaxpy()` BLAS wrapper |
| `lib.rs` | Crate root, re-exports, integration tests |
| `SKmodel.rs` | Slater-Koster parameterized models (`SlaterKosterModel`, `SkAtom`, `SkParams`) |
| `fermi_surface.rs` | `FermiSurface`/`FermiSurfacePlane` traits (marching squares/tetrahedra → gnuplot); BXSF export (`BxsfExport` trait → XCrySDen/FermiSurfer); `write_spin_frmsf` free fn (spin‑split FRMSF for altermagnets) |
| `optical_conductivity.rs` | Frequency-dependent optical conductivity & optical Hall |
| `orbital_angular.rs` | Orbital angular momentum operator |
| `magnetic_field.rs` | Uniform magnetic field via Peierls substitution (`MagneticField` trait) |
| `unfold.rs` | Band unfolding for supercell→primitive projection (`Unfold` trait) |
| `solve_ham.rs` | Parallel diagonalization (`solve_all_parallel`, `solve_range_onek`) |
| `kpoints.rs`/`kpath.rs`/`kplane.rs` | k-mesh/k-path/k-plane generation |
| `output.rs` | gnuplot plotting (`show_band`, `show_surf_state`, `draw_heatmap`, `show_fermi_surface`) |
| `model_transform.rs` | Supercell construction, orbital removal/reordering, `shift_to_atom` |
| `model_utils.rs` | Internal utilities: `find_R()` for lattice vector lookup in `hamR` |
| `math.rs` | Matrix algebra: `comm()`, `anti_comm()`, `gauss()` smearing function |
| `atom_struct.rs` | `Atom` and `OrbProj` types |
| `error.rs` | `TbError` enum, `Result` alias |
| `generics.rs` | Numeric abstractions (`hop_use`, `usefloat`), `SpinDirection::from_index` |
| `phy_const.rs` | Physical constants: `e`, `ħ`, `μ_B`, `Φ₀`, quantum of conductance |
| `io.rs` | Text-file output helpers (`write_txt`, `write_txt_1`) |

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

## Future: Berry/QGT Simplex Integration (design plan)

> **Status**: not implemented. `tetrahedron_volume_integrate()` linearly interpolates
> final scalars and is unsafe near small gaps. The reference path remains direct
> k-mesh summation with finite `η`.

**Core idea**: interpolate gauge-invariant velocity product `K^ab_nm = A^a_nm * A^b_mn`
(not `A^a_nm` directly, which carries U(1) phase ambiguity), plus band energies
`E_n(k)`, then evaluate `K_nm / ((E_n-E_m)^2 + η^2)` at symmetric quadrature
points. For dipoles, also interpolate diagonal velocity `v^c_n(k)`.

**Band tracking**: energy sorting alone fails near crossings — use eigenvector
overlap maximization (per-simplex permutation matching) to align band labels
across simplex vertices. Mark simplexes as unsafe when `min_gap < gap_tol`.

**Quadrature**: 2D degree-2/3 triangles, 3D 4-point degree-2 tets first;
closed-form analytic integration only if benchmarks show quadrature is a bottleneck.

**Data structures**: internal `VertexKernel`, `TrackedSimplex`, `SimplexDiagnostics`
in `src/tetrahedron.rs` or `src/simplex_response.rs`.

See the full design doc in the commit history (`fb82fca` and neighbors) for formulas,
implementation sequence, and validation test plan.

## Performance Notes

**`zaxpy`** (`src/ndarray_lapack.rs`): safe BLAS `y += alpha * x` for `Complex<f64>` slices.
Preferred over `Zip`/elementwise for direction-weight accumulation of dense matrices.
Use `Complex::new(weight, 0.0)` as scalar; only call when source and destination are
standard contiguous slices (`.as_slice().unwrap()` / `.as_slice_mut().unwrap()`).

**Allocation reduction**: avoid `Array1::from_vec(...)` and `.to_owned()` inside hot loops;
prefer preallocated buffers. For rayon folds over `mu` values, mutate a local
`Array1<f64>` accumulator directly instead of allocating per-iteration.

**Autovec/SIMD**: simple contiguous slice loops autovectorize better than `ndarray`
indexed/transposed views. Use `RUSTFLAGS="-C target-cpu=native"` for AVX2/AVX512.
BLAS backends (MKL/OpenBLAS) dispatch optimized kernels independently.

## Refactoring Guidelines

- **No batch sed/Python**: modify files one at a time with proper tooling.
- **Parallel agents for multi-file changes**: core file first manually, then launch
  3-4 agents in parallel for distinct groups of files.
- **Commit after each successful `cargo check`**: prevents data loss from
  `git checkout` discarding uncommitted work.

---

# Tetrahedron Integration: Current Status

> **Last updated**: 2026-06-29.

## What's implemented

### `Hall_conductivity_tetra` (intrinsic AHC)
- 2D: 3-point triangle quadrature; 3D: Blochl sub-tet with analytic `compute_occ_omega`
- T>0: thermal convolution (requires n_mu>1, errors otherwise)
- **Parallelized**: `rayon::par_iter().fold().reduce()`

### `Nonlinear_Hall_conductivity_Extrinsic_tetra` (extrinsic NLH)
- 2D: triangle → line-segment Fermi-surface cut with analytic 1D integral
- 3D: tetrahedron → triangle Fermi-surface cut with divided-difference K_αβ weights
- T>0: thermal convolution
- **Parallelized**

### `Nonlinear_Hall_conductivity_Intrinsic_tetra` (intrinsic NLH)
- 2D+3D, charge branch only (no spin-current branch yet)
- T=0 Fermi-surface integration; T>0 thermal convolution of the T=0 result
- **Parallelized**

## Nonlinear Hall index conventions

- `Nonlinear_Hall_conductivity_Extrinsic` and
  `Nonlinear_Hall_conductivity_Extrinsic_tetra` return the unsymmetrized
  kernel `S_{ab;c} = ∫(-df/dE) v_c Omega_ab dk`.  For current-first
  `chi_ext[a,b,c]`, use `Nonlinear_Hall_conductivity_Extrinsic_sym` or
  `Nonlinear_Hall_conductivity_Extrinsic_tetra_sym`, which compute
  `0.5 * (S_{ab;c} + S_{ac;b})`.
- `Nonlinear_Hall_conductivity_Intrinsic` and
  `Nonlinear_Hall_conductivity_Intrinsic_tetra` are current-first APIs:
  arguments mean `(current, field_1, field_2)` and internally map to
  `sigma_int^{field_1 field_2; current}`.
- The charge intrinsic implementation uses
  `G_code^{ij}=Re sum_m v^i_nm v^j_mn/(E_n-E_m)^3`.  Literature formulas
  that define `G_lit=2 Re sum ...` differ by an overall factor of 2.

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

### Intrinsic tetra is not mathematically identical to `Nonlinear_Hall_conductivity_Intrinsic`

This is the main diagnostic for the observed failure of the expected time-reversal odd relation
between spin-up and spin-down H-wave / altermagnetic test models.

The current-first intrinsic API returns

```text
chi_int[c,a,b](mu,T) = sigma_int^{ab;c}(mu,T).
```

The direct implementation first constructs the full scalar kernel at each k point:

```text
G_n^{ij}(k) = Re sum_{m != n}
    v^i_nm(k) v^j_mn(k) / (E_n(k) - E_m(k))^3

Q_n^{ab;c}(k) =
    2 v^c_n(k) G_n^{ab}(k)
  - 1/2 [v^a_n(k) G_n^{bc}(k) + v^b_n(k) G_n^{ac}(k)]

omega_n(k) = -Q_n^{ab;c}(k)
```

Then it integrates the scalar `omega_n(k)`:

```text
sigma(mu,T) = 1/det(lat) * sum_n int_BZ
    (-df/dE_n) omega_n(k) dk.
```

At T=0 the direct path calls `tetrahedron_integrate(&band, &omega, ...)`, which means
the approximation is

```text
E_n(k)      -> linear interpolation inside each simplex
omega_n(k) -> linear interpolation inside each simplex

I_direct_tetra ~= int delta(Interp[E_n] - mu) Interp[omega_n] dk.
```

`Nonlinear_Hall_conductivity_Intrinsic_tetra` does a different approximation.  It does
not interpolate the final scalar `omega_n`.  It stores per-vertex primitives

```text
v^a_n, v^b_n, v^c_n,
P_nm^{ab} = Re[v^a_nm v^b_mn],
P_nm^{ac} = Re[v^a_nm v^c_mn],
P_nm^{bc} = Re[v^b_nm v^c_mn],
d_nm = E_n - E_m,
```

linearly interpolates those primitives, and evaluates the pair-decomposed singular
kernel on the Fermi cut:

```text
omega_n(k) = sum_{m != n}
  [-2 v^c_n P_nm^{ab}
   + 1/2 v^a_n P_nm^{bc}
   + 1/2 v^b_n P_nm^{ac}]
  / d_nm^3.
```

In 2D, for each triangle cut segment `L = {k in tri | E_n(k)=mu}`, the implemented
integral is

```text
sum_m int_L
  [-2 v^c P^{ab} + 1/2 v^a P^{bc} + 1/2 v^b P^{ac}]
  / d_nm^3
  * dl / |grad E_n|.
```

In 3D, for each tetrahedron cut triangle `S = {k in tet | E_n(k)=mu}`, it is

```text
sum_m int_S
  [-2 v^c P^{ab} + 1/2 v^a P^{bc} + 1/2 v^b P^{ac}]
  / d_nm^3
  * dS / |grad E_n|.
```

This is not algebraically equivalent to the direct scalar interpolation at finite
mesh size:

```text
Interp[sum_m N_m / d_m^3] != sum_m Interp[N_m] / Interp[d_m]^3.
```

The pair-decomposed version is defensible because it preserves the singular
denominator structure better than interpolating the final scalar, but it is more
sensitive to small gaps, band-index swaps, and asymmetric cutoff decisions.  It should
only be expected to approach the direct result when the mesh is fine, band labels are
continuous across every simplex, and all relevant pair gaps stay safely away from zero.

### Current intrinsic `1/d^3` cutoff

`Intrinsic_tetra` currently uses

```rust
const INTRINSIC_D3_GAP_TOL: f64 = 1e-5;
```

For each cut segment or cut triangle and each pair `(n,m)`, define

```text
d_min = min d_nm on the cut object,
d_max = max d_nm on the cut object.
```

The whole pair contribution on that cut object is skipped if

```text
d_min <= gap_tol && d_max >= -gap_tol,
```

equivalently if the interpolated gap interval intersects `[-gap_tol, gap_tol]`.
This differs from the direct intrinsic path, which sets the pointwise pair denominator
to zero only when `|E_n(k)-E_m(k)| < 1e-5` at an existing k point.  The tetra cutoff
therefore removes extended cut pieces and can break time-reversal cancellation at
finite mesh size.

### Regularizing the intrinsic `1/d^3` denominator

Hard cutoff is a crude diagnostic tool, not a good long-term regularization.  Any
replacement must preserve the odd parity of the kernel:

```text
R_eta(-d) = -R_eta(d),
R_eta(d) -> 1/d^3  for |d| >> eta.
```

Good candidates:

1. Retarded denominator broadening:

```text
R_eta(d) = Re[1/(d + i eta)^3]
         = d (d^2 - 3 eta^2) / (d^2 + eta^2)^3.
```

This has the cleanest Kubo/Green-function interpretation, but it changes sign near
`|d| = sqrt(3) eta`.

2. Sign-preserving smooth cutoff:

```text
R_eta(d) = d / (d^2 + eta^2)^2.
```

This is phenomenological, but it is odd, finite at `d=0`, and approaches `1/d^3`
away from the avoided crossing.  It is the recommended first implementation for
stabilizing `Intrinsic_tetra`.

3. Principal-value style exclusion:

```text
R_eta(d) = 1/d^3 for |d| > eta, excluded otherwise.
```

If this path is used, do not skip the whole cut segment/triangle.  Split the cut
object again by `|d|=eta` and integrate only the safe sub-regions.  Otherwise the
method keeps the present asymmetric cancellation error.

With a smooth `R_eta`, the analytic `segment_integral_d3` and `triangle_integral_d3`
formulas no longer apply directly.  The pragmatic implementation route is to replace
them by quadrature on the Fermi cut:

```text
2D segment: 8- or 16-point Gauss-Legendre quadrature over xi in [0,1].
3D triangle: symmetric barycentric triangle quadrature, degree >= 4 initially.
```

Because `eta > 0` makes the integrand smooth, this is easier to validate than deriving
new divided-difference formulas for each regularizer.

### Adaptive subdivision idea

Adaptive tetra refinement can help, but only if new vertices are evaluated by
diagonalizing the Hamiltonian and recomputing primitives.  Subdividing a linear
tetrahedron using only interpolation of the old four vertices adds no new information.

Reasonable refinement triggers:

```text
mu in [min E_n, max E_n] and max E_n - min E_n is small,
or min_{Fermi cut} |d_nm| < d_tol,
or |I_parent - sum_i I_child_i| > abs_tol + rel_tol * |sum_i I_child_i|.
```

Algorithm plan:

1. Compute the parent contribution `I_parent`.
2. Split the tetrahedron, preferably into 8 children to preserve shape quality.
3. Evaluate Hamiltonian, velocities, eigenvectors, and pair primitives at all new
   vertices; do not reuse only interpolated old values.
4. Compute `I_child = sum_i I_child_i`.
5. Accept `I_child` when the error estimate converges; otherwise recurse until
   `max_depth`.
6. If a true degeneracy remains on the Fermi cut at `max_depth`, report diagnostics
   or rely on explicit `eta` regularization.  Do not silently drop the contribution.

For time-reversal-sensitive tests, refinement must be paired: if a simplex is refined,
its `k -> -k` partner should be refined with the same depth and the same rule.  Otherwise
adaptive refinement itself introduces symmetry leakage.

### Recommended fix plan for intrinsic tetra

1. Add an explicit intrinsic denominator regularization option, starting with
   `R_eta(d)=d/(d^2+eta^2)^2`, and thread an `eta_intrinsic` parameter through
   `Nonlinear_Hall_conductivity_Intrinsic_tetra`.
2. Replace hard skip logic in `segment_integral_d3` / `triangle_integral_d3` with
   quadrature over the regularized integrand.
3. Add diagnostics that count: skipped near-gap pairs, minimum encountered `|d_nm|`,
   number of cut objects near a pair degeneracy, and contribution magnitude by pair.
4. Add a direct comparison test for a model with isolated bands:

```text
max_mu |Intrinsic(mu,T=0) - Intrinsic_tetra(mu,T=0)| / max_mu |Intrinsic(mu,T=0)|
```

and track convergence over k-mesh and `eta_intrinsic`.
5. Add the H-wave up/down odd test:

```text
max_mu |chi_up(mu) + chi_dn(mu)| / max_mu |chi_up(mu)|
```

for both direct and tetra methods.
6. Only after the regularized quadrature path is stable, add band tracking using
   eigenvector-overlap matching inside each simplex.  This is required for robust
   interpolation near band crossings.
7. Consider adaptive refinement last.  It is useful for accuracy but not a substitute
   for denominator regularization, band tracking, or symmetry-paired refinement.

## Parallelization pattern
Tetra response methods follow the same pattern:
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
