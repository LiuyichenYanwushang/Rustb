# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code review before commit (mandatory)

Before committing any code change or new code, run an adversarial review loop and do not commit until it converges:

1. Spawn a review agent (subagent) to review the diff. The reviewer must raise concrete, detailed questions and challenges — physics/sign/convention correctness, thread-safety, determinism, edge cases, test adequacy — not merely approve.
2. The main agent examines each challenge and either ACCEPTS it (fix the issue) or REBUTS it (with explicit reasoning why it is not a problem).
3. After applying accepted fixes, re-run the review loop (a fresh review agent on the updated changes).
4. Repeat until every challenge is resolved and both sides are convinced. Only then commit or otherwise finalize the change.

---

## Project Overview

Rustb is a Rust library for tight-binding model calculations in condensed matter physics. It computes band structures, density of states, transport properties (Hall conductivity, nonlinear responses), topological invariants (Chern numbers, Wilson loops), and surface states using Green's functions.

- **MSRV**: 1.90.0 (edition 2024)
- **Repo**: https://github.com/LiuyichenYanwushang/Rustb
- **Error handling**: Uses `thiserror` for `TbError` enum.
- **Docs**: `katexit` renders LaTeX in rustdoc; `docs-header.html` for custom CSS.
- **Version**: 0.7.2.
- **SKILLS.md**: Practical usage guide with code examples for the entire crate. When adding or changing any public API, update that file as well.

> **Note**: README.md and SKILLS.md both use the current const-generic API.

## 0.7.x Refactor Summary

Version 0.7 is the next release after the crates.io version 0.6.7 and
deliberately breaks the previous response API. The central design rule is that
physics workflows end in an ordinary
`Model<SPIN, DIM, R>` whenever that is physically meaningful. Solver-specific
wrappers hold input data and iteration policy; they do not create parallel
model hierarchies with duplicated band, geometry, or response methods.

### Hubbard unrestricted Hartree-Fock

- `HubbardModel<DIM, R>` wraps a spinful `Model<true, DIM, R>` plus one on-site
  `U_i` per physical orbital. It is an input/solver type, not a replacement for
  `Model`.
- `solve_hartree_fock(&MeanFieldParams<DIM>)` is the only mean-field solve
  entry point. It performs non-collinear unrestricted Hartree-Fock using the
  complete local `2 x 2` spin-density matrix. Both diagonal Hartree terms and
  off-diagonal Fock spin-flip terms are updated self-consistently.
- `MeanFieldConstraint::FixedChemicalPotential` keeps the supplied chemical
  potential fixed during iteration. `FixedInitialFilling` first computes the
  bare-model filling at `reference_mu`, on the requested k-mesh and with the
  requested occupation, then solves for a chemical potential that preserves
  that filling at every iteration. This is the preferred mode when the
  mean-field bands move, especially for metals.
- `MeanFieldParams` owns the numerical policy: `[usize; DIM]` k-mesh,
  thermodynamic constraint, `Occupation`, iteration limit, density tolerance,
  linear mixing, and the initial collinear or non-collinear magnetic seed.
- The converged chemical potential is subtracted from the final Hamiltonian,
  so `solve_hartree_fock` returns a normal `Model<true, DIM, R>` whose Fermi
  level is at zero. Magnetization is already encoded in its spin-dependent
  one-particle Hamiltonian and can be measured with ordinary `Model` spin
  expectation APIs such as `spin_moment`.
- Failure to converge is reported as `TbError::MeanFieldNotConverged`; invalid
  thermodynamic, mesh, interaction, or iteration parameters return structured
  errors rather than panicking.

### Shared thermodynamics and metallic calculations

- All occupation-dependent workflows use the same `Occupation` enum:
  `ZeroTemperature`, `FermiDirac { temperature_kelvin }`, or
  `FermiSmearing { width }`. Temperatures are kelvin; Hamiltonian energies and
  smearing widths are electronvolts.
- Exact zero-temperature occupations are step functions. A direct k-sum of a
  Fermi-surface derivative cannot represent the resulting Dirac delta and
  therefore requires finite `FermiDirac`/`FermiSmearing`; an energy-cut method
  should be used when the exact zero-temperature Fermi surface is required.
- For metallic Hubbard calculations, use a converged k-mesh and small finite
  smearing (or physical temperature), normally together with
  `FixedInitialFilling`. The target electron count is computed once from the
  bare model instead of being guessed independently.

### Unified response APIs

- Every high-level response calculation takes one shared `Parameters<DIM>`
  structure and returns one named `*Result` structure. Fields: `T` (kelvin,
  `0.0` = zero temperature), `mu` (eV), `eta` (broadening), `kmesh`,
  `omega` (eV), `spin` (`None` = charge current), `direction`
  (`Array2<f64>`, shape `(rank, DIM)`), `integration`, `field_symmetry`.
  Methods read only the fields they need and ignore the rest.
- The supported entry points are `hall_conductivity`, `quantum_geometry`,
  `optical_conductivity`, `extrinsic_nonlinear_hall`, and
  `intrinsic_nonlinear_hall`. Algorithm choice belongs in the shared
  `Integration` enum (`Direct`/`Simplex`/`EnergyCut`) instead of being encoded
  in alternate method names.
- `direction` replaces the old `DirectionPair`/`NonlinearHallDirections`/
  `OpticalDirections`: rank-2 responses use 2 rows, rank-3 responses use
  `(current, field_1, field_2)`. `spin` replaces `CurrentOperator`,
  `T` replaces `Occupation` at the response boundary, and `FieldSymmetry`
  (kept as a field) selects ordered or symmetrized nonlinear field indices.
- Direct, simplex, and energy-cut paths share the same gauge-invariant response
  kernels and Cartesian reciprocal-space normalization. Optical conductivity
  returns the full ordered `DIM x DIM` Cartesian tensor when `direction` is an
  empty matrix, or one projected component for a 2-row direction.
- Raw `VertexKernel`, band tracking, simplex construction, quadrature, and
  energy-cut helpers are crate-private numerical machinery. Do not expose them
  as compatibility APIs; extend the parameter/result layer instead.

### Breaking migration rules

- Legacy method families such as `Hall_conductivity*`, the old optical and
  nonlinear conductivity variants, tuple return values, and
  `solve_mean_field` were removed rather than deprecated. Update callers to
  the parameterized lowercase entry points and `solve_hartree_fock`.
- New public physics configuration should normally be a parameter structure
  with validation near the API boundary. Avoid long positional argument lists,
  bool switches, and duplicated direct/simplex method names.
- README.md, SKILLS.md, rustdoc examples, executable examples, tests, and
  benchmarks are part of the API migration. Any future public signature or
  convention change must update all of them in the same commit.

### Other 0.7 work in this development batch

- Floquet support now covers Peierls-Sambe models, quasienergy folding, and a
  validated van Vleck effective-model path that also returns an ordinary
  `Model`.
- Optional allocator features, stricter validation/error variants, response
  integration diagnostics, and benchmark baselines were updated alongside the
  public API consolidation.

## Common Commands

### Building
```bash
cargo build --features openblas-system
cargo build --features intel-mkl-static   # Intel MKL
cargo build --features openblas-static    # OpenBLAS
cargo build --release --features openblas-system
```

### Testing
```bash
cargo test --release --features openblas-system                # always use --release for numerics
cargo test --release --features intel-mkl-system              # with MKL
cargo test --release --features openblas-system graphene       # single test
cargo test --release --features openblas-system -- --nocapture 2>&1 | head -100
```

Simplex-quadrature tests involve heavy floating-point loops (band tracking,
K-quadrature, energy-cut).  Debug builds are 10-50x slower and can cause
timeouts.  Always test with `--release`.

Tests generate PDF plots via gnuplot (`pdfcairo` terminal).

### Development
```bash
cargo fmt
cargo clippy --features openblas-system
cargo bench --features intel-mkl-system             # criterion benchmarks
cargo mydoc                                          # cargo doc --open --no-deps --features openblas-system
cargo testall                                        # cargo test --features intel-mkl-system
cargo runexample <name>                              # cargo run --features intel-mkl-system --example <name>
```

Custom aliases (`mydoc`, `testall`, `runexample`) are defined in `.cargo/config.toml`.
`mydoc` hardcodes `openblas-system`; `testall` and `runexample` hardcode
`intel-mkl-system`. Cargo merges extra `--features` instead of replacing
alias-injected ones, so do not combine an alias with another backend feature.
To use a different backend, invoke `cargo doc`/`cargo test`/`cargo run`
directly with `--features <backend>` (or edit the alias).

## BLAS/LAPACK Backend

Rustb has no default BLAS/LAPACK backend; every build, test, and doc command
must select exactly one backend feature. Version 0.7.2 removed the previous
`openblas-system` default, so code upgrading from 0.7.1 must add a backend
feature explicitly.

| Feature | Backend |
|---------|---------|
| `intel-mkl-static` | Intel MKL (best x86_64 perf) |
| `openblas-static` | OpenBLAS |
| `netlib-static` | Reference netlib |
| `intel-mkl-system` | System-installed MKL |
| `openblas-system` | System-installed OpenBLAS |
| `netlib-system` | System Netlib |

---

## Core Architecture

### `Model<SPIN, DIM, R>` — central data structure

Three compile-time generics replace what were historically runtime fields:

```rust
pub struct Model<
    const SPIN: bool = false,
    const DIM: usize = 3,
    R: RMatrixData = NoRMatrix,
> {
    pub lat: Array2<f64>,     // real-space lattice vectors (rows)
    pub orb: Array2<f64>,     // fractional orbital positions (rows)
    pub rmatrix: R,           // NoRMatrix = ZST, HasRMatrix = Array4 newtype
    // ...
}
```

- `DIM` defaults to 3. `dim_r()` returns `DIM`.
- `R` defaults to `NoRMatrix` (zero-sized). `HasRMatrix(Array4)` stores position matrix elements.
- Hot-path `match DIM { 1 => ..., 2 => ..., 3 => ... }` is compile-time eliminated.
- Hot-path `if <R as RMatrixData>::HAS_RMATRIX { ... }` is compile-time eliminated.
- `surf_Green` is non-generic and stores `spin: bool` and `dim_r: usize` (converted from const generics).

### Other key types

- **`Gauge`**: `Atom` (orbital positions in phase) or `Lattice` (only R vectors).
- **`RMatrixData`**: Trait — `HasRMatrix` (Deref to Array4) or `NoRMatrix` (ZST, literally no field).
- **`SpinDirection`**: `X/Y/Z` Pauli matrices; `Option<SpinDirection>::None`
  selects the spin-independent identity term.
- **`Dimension`**: Enum `one=1/two=2/three=3` for serde compat, NOT stored in Model.

### Trait hierarchy

```
Velocity  (src/velocity.rs)          → v_α(k) operator
  ├─ BerryCurvature (src/response/traits.rs) → AHC, spin Hall, nonlinear Hall
  │    ├─ intrinsic NLH (src/response/nonlinear)
  │    └─ extrinsic NLH (src/response/nonlinear)
  └─ QuantumGeometry (src/quantum_geometry.rs) → QGT, quantum metric

Floquet (src/floquet.rs)             → Sambe Hamiltonian, quasienergies, van Vleck effective model

FermiSurface / FermiSurfacePlane (src/fermi_surface.rs)
Berry (src/geometry.rs)              → Wilson loops, Berry phase, Wannier centres
CutModel (src/cut.rs)                → slab/ribbon (cut_piece), dot (cut_dot)
MagneticField (src/magnetic_field.rs)
Unfold (src/unfold.rs)
CrystalSymmetry (src/crystal_symmetry.rs, feature `cryspglib`)
  → Atom-based SG/MSG, effective symmetry under optional E/B fields,
    character tables, high-symmetry k points, irreducible meshes
Model::check_hamiltonian_symmetry (src/hamiltonian_symmetry.rs, feature `cryspglib`)
  → exact real-space operation residuals, validated localized sewing actions,
    and setting-aware residual UNI/BNS magnetic-group identification
Model::symmetrize_hamiltonian (src/hamiltonian_symmetry.rs, feature `cryspglib`)
  → validated magnetic Reynolds projection returning a new Model
```

All trait impls: `impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Trait for Model<SPIN, DIM, R>`.

### Key source files

| File | Purpose |
|------|---------|
| `model.rs` | `Model<SPIN, DIM, R>` struct, `RMatrixData` trait, `HasRMatrix`/`NoRMatrix`, `Gauge`, serde |
| `model_build.rs` | Builder: `tb_model()`, `set_hop()`, `make_supercell()`, macros |
| `model_physics.rs` | `gen_ham()` (Bloch Hamiltonian), `dos()` |
| `velocity.rs` | `Velocity` trait — `gen_v()` with rmatrix commutator; `gen_v_projected()` fuses direction-weight projection |
| `response/` | Public parameter/result APIs plus private direct-sum and simplex machinery |
| `response/primitives.rs` | Internal `compute_velocity_kernel` band-basis velocity elements |
| `response/types.rs` | Internal `VertexKernel`, `TrackedSimplex`, `SimplexDiagnostics` |
| `response/quadrature.rs` | 2D/3D symmetric quadrature rules, barycentric interpolation |
| `response/tracking.rs` | Band tracking (overlap → greedy assign → permute) + simplex builders |
| `response/kernel.rs` | Berry/QGT, optical, dipole kernel evaluators at quadrature points |
| `response/energy_cut.rs` | 2D/3D energy-cut (AHC, dipole, intrinsic), hybrid + K-quadrature |
| `response/linear/` | Berry curvature + quantum metric simplex integration |
| `response/nonlinear/` | Berry dipole + intrinsic/extrinsic NLH |
| `response/optical/` | Optical conductivity simplex integration |
| `quantum_geometry.rs` | `QuantumGeometry` trait — QGT, quantum metric |
| `geometry.rs` | `Berry` trait — Berry phase, Wilson loops, Wannier centres |
| `surfgreen.rs` | Surface Green's function (Sancho-Rubio iterative method) |
| `wannier90.rs` | Wannier90 I/O; the requested `R` type controls whether `_r.dat` is loaded |
| `cut.rs` | `CutModel` trait — slab (`cut_piece`), dot/edge (`cut_dot`) |
| `fermi_surface.rs` | `FermiSurface`/`FermiSurfacePlane` traits; BXSF export; `write_spin_frmsf` for altermagnets |
| `floquet.rs` | `Floquet` trait + `Model::floquet_effective_model`; `LightMode`, `FloquetDrive`, `FloquetTruncation`, `IncidentBasis`, `FloquetEffectiveOptions`, `fold_quasienergy` |
| `ndarray_lapack.rs` | LAPACK bindings + safe `zaxpy()` BLAS wrapper |
| `lib.rs` | Crate root, re-exports, integration tests |
| `solve_ham.rs` | Parallel diagonalization (`solve_all_parallel`, `solve_range_onek`) |
| `kpoints.rs`/`kpath.rs`/`kplane.rs` | k-mesh/k-path/k-plane generation |
| `output.rs` | gnuplot plotting |
| `model_build.rs` | Model construction, supercells, typed orbital/atom removal and reordering |
| `model_utils.rs` | Internal: `find_R()` for lattice vector lookup in `hamR` |
| `math.rs` | `comm()`, `anti_comm()`, `gauss()` smearing |
| `atom_struct.rs` | `Atom` and `OrbProj` types |
| `crystal_symmetry.rs` | Optional `cryspglib` adapter; structure and field-effective symmetry |
| `hamiltonian_symmetry.rs` | Exact Hamiltonian covariance, residual MSG identification, and forced Hamiltonian symmetrization |
| `irrep_analysis.rs` | Parallel high-symmetry-k band characters and magnetic corep identification |
| `error.rs` | `TbError` enum, `Result` alias |
| `generics.rs` | Numeric abstractions, `SpinDirection::from_index` |
| `phy_const.rs` | Physical constants: `e`, `ħ`, `μ_B`, `Φ₀`, quantum of conductance |
| `io.rs` | Text-file output helpers |

### Conventions

- **k-points**: Fractional reciprocal coordinates; phase = `exp(2πi k·R)`.
- **Orbital positions**: Fractional coords (rows of `orb`).
- **Lattice vectors**: Stored as rows of `Model::lat`; Cartesian row
  coordinates are `fractional.dot(lat)`.
  Reciprocal vectors via `Model::rec_lat()` (rows = reciprocal vectors,
  formula `B = 2π·(Aᵀ)⁻¹`).
- **Eigenbasis transformation** (`band` ↔ `evec` from `eigh`): the codebase
  convention is `U^T · O · U^*` (transpose, NOT conjugate-transpose, on the
  left; conjugate on the right).  Verified by test `evec_transform_sanity`
  which checks `diag(U^T H U^*) == band` numerically.
- **Hermitian conjugates**: Auto-generated by `set_hop`/`add_hop`.
- **Public API**: `lib.rs` re-exports the high-level model, trait, parameter,
  and result types. Raw response kernels, band tracking, and simplex types are
  crate-private implementation details.

### Nonlinear Hall index conventions

- `extrinsic_nonlinear_hall` and `intrinsic_nonlinear_hall` both read
  `direction` rows as `(current, field_1, field_2)` — current-first.
- Extrinsic calculations default to `FieldSymmetry::Symmetrized`, which
  computes `0.5 * (S_ab;c + S_ac;b)`. Select `FieldSymmetry::Ordered` for
  the unsymmetrized kernel.
- The intrinsic response is current-first: `(current, field_1, field_2)` maps
  to `sigma_int^{field_1 field_2; current}`.
- The charge intrinsic implementation uses
  `G^{ij}=Re sum_m v^i_nm v^j_mn/(E_n-E_m)^3`.  Literature formulas
  that define `G=2 Re sum ...` differ by an overall factor of 2.

### Response API conventions

- High-level response methods take one shared `Parameters<DIM>` structure and
  return one named `*Result` structure. Directions are rows of an
  `Array2<f64>` matrix, so dimension mismatches are rejected at runtime with
  structured errors.
- `T` (kelvin; `T[0] == 0.0` = zero temperature) replaces `Occupation` at the
  response boundary. The `Occupation` enum itself remains for the Hubbard
  mean-field solver and spin-moment observables.
- Direct and simplex/energy-cut algorithms are selected by the shared
  `Integration` enum rather than separate method names.
- `compute_velocity_kernel`, `VertexKernel`, raw energy-cut integrators, and
  band-tracking helpers are intentionally not public compatibility surfaces.

---

## Simplex Quadrature (`src/response/`)

Replaces the old Blochl tetrahedron method. Instead of interpolating the final
integrand `Ω_n(k)` at simplex vertices and applying analytic δ-function weights:

1. **Interpolates gauge-invariant primitives** `K_nm = v^a_nm·v^b_mn` and band
   energies `E_n` linearly inside each simplex.
2. **Evaluates the kernel at quadrature points**: `Ω_n(q) = −2 Im Σ_m K_nm(q)/(d²(q)+η²)`.
3. **Weights by quadrature**: `∫_simplex f(k)dk ≈ V_simplex · Σ_q w_q f(q)`.

This preserves the singular `1/(E_n−E_m)²` structure near small gaps.

### Module structure

```
src/response/
├── types.rs        — internal VertexKernel, TrackedSimplex, diagnostics
├── quadrature.rs   — 2D/3D symmetric quadrature rules, barycentric interp
├── tracking.rs     — band tracking (overlap → greedy → permute) + simplex builders
├── kernel.rs       — Berry/QGT, optical, dipole kernel evaluators
├── energy_cut.rs   — 2D/3D energy-cut (AHC, dipole, intrinsic), hybrid + K-quadrature
├── primitives.rs   — internal velocity-kernel construction
├── linear/         — Berry curvature + quantum metric + AHC
├── nonlinear/      — Berry dipole + intrinsic/extrinsic NLH
└── optical/        — Optical conductivity
```

### Energy-cut design

**AHC** (volume integral `∫ Θ(μ−E) Ω dk`):

| Occupancy | Method | Cost |
|-----------|--------|------|
| Fully occupied | Vertex-average Ω (preserves gap quantization) | O(1) |
| Fully empty | Skip | O(1) |
| Partially occupied | K-quadrature on clipped polygon | O(N_quad) |

3D uses a sorted-μ sweep: binary search finds empty/partial/full μ ranges.

**Nonlinear** (Fermi-surface integral `∫ δ(μ−E) A dk`):

| Dim | Intersection | Quadrature |
|-----|-------------|------------|
| 2D | E=μ line ∩ triangle → segment | 2-pt Gauss K-quad along line |
| 3D | E=μ plane ∩ tetrahedron → polygon | polygon triangulation + 3-pt K-quad |

Nonlinear energy-cut has no full/empty fast path (integrand ∝ δ(E−μ) is zero
away from the Fermi surface).

| Method | Amplitude A_n |
|--------|----------------|
| Dipole (extrinsic) | `v^c_n · Ω^{ab}_n` |
| Intrinsic | `2v^c_n G^{ab}_n − ½(v^a_n G^{bc}_n + v^b_n G^{ac}_n)` |

**3D diagonal averaging** (`build_tetrahedra_3d_diagavg`): generates 10 tets per
cell by averaging the two possible 5-tet cube decompositions with opposite body
diagonals. Restores k→−k cancellation for P-odd quantities.

### Known limitations

- **Intrinsic NLH 3D convergence** — nk=12 gives 4.2% error vs nk=14 reference
  (surface K-quad ∝ h², slower than 2D line integral ∝ h)
- **Dipole energy-cut is 2D only** — 3D tetrahedron surface cut not yet implemented
- **No band tracking in intrinsic NLH** direct-sum path (not ported to simplex)

---

## Fermi Surface Export

Three output paths in `fermi_surface.rs`:

1. **`show_fermi_surface`** — gnuplot (2D/3D). Slow for 3D.
2. **`write_bxsf`** — BXSF ASCII for XCrySDen/FermiSurfer. `Model<SPIN, 3, R>` only.
3. **`write_spin_frmsf`** — FRMSF for spin-split Fermi surfaces (altermagnets).

### Critical conventions (hard-won — changing any breaks symmetry)

**`Model::rec_lat()`** returns reciprocal lattice vectors as **rows**:
`B = 2π · (Aᵀ)⁻¹`. BXSF/FRMSF header writes three lines: `b[i,0], b[i,1], b[i,2]`.

**`gen_kmesh` row ordering**: ik1 (dim 0) outermost, ik3 (dim 2) innermost.
Row index: `ik1*ny*nz + ik2*nz + ik3`. Fractional coords `k = (ik1/nx, ik2/ny, ik3/nz)`.

**FRMSF/BXSF write order matches gen_kmesh** — `eval` rows are already in the
correct order. Do NOT introduce an index formula like `ik1 + ik2*nx + ik3*nx*ny`
— it will scramble E(k) and destroy crystal symmetries (C3z, etc.).

**`e_fermi` must be subtracted** from band energies. FermiSurfer assumes `E_F = 0`.

**`write_spin_frmsf` checks** — validates up/dn models have same band count and
lattice (element-wise diff < 1e-10).

---

## Floquet: Real-Space Peierls-Sambe Solver

### Real-space convention

Hopping blocks: `t_{ij}(R) = ⟨i,0|H|j,R⟩`, with integer lattice vector `hamR[R]`
and matrix block `ham[R,i,j]`.  Real-space link for Peierls phase:

```math
d_{ijR} = (R + τ_j − τ_i) · L
```

where `orb` stores fractional orbital positions and `lat` is the real-space
lattice matrix used with row-vector fractional coordinates (`cart = frac · lat`).
For spinful models: `orbital_index = state_index % norb`.

### Light-field convention

Dimensionless vector-potential amplitude `a(t) = (e/ħ)A(t)` in inverse length
units matching `lat` (typically 1/Å). Complex Fourier components stored as
`a_complex`, real drive:

```math
a(t) = Re Σ_α a_α e^{−i l_α Ω₀ t}
```

`l_α` is an integer harmonic. Supports arbitrary linear, circular, elliptical,
and mixed-frequency commensurate polarization.

### Coherent and mode-resolved incoherent drives

`floquet_effective_model` always treats every entry in `FloquetDrive::modes`
as one **coherent** classical field: the complex mode amplitudes are summed
before the Peierls exponential and Fourier decomposition.  Modes at the same
temporal harmonic therefore interfere, and the generalized-Bessel expansion
also retains nonlinear mixed-frequency paths.  In general,

```math
H_{\rm eff}[a_1+a_2]\ne H_{\rm eff}[a_1]+H_{\rm eff}[a_2].
```

`floquet_effective_mode_resolved_model` is the unique mutually-incoherent
mode-combination helper.  It computes one complete effective model
`H_eff^(alpha)` for each nonzero mode in isolation, unions those supports, and
adds every induced correction around one common static Hamiltonian:

```math
H_{\rm inc}=H_0+\sum_\alpha
\left(H_{\rm eff}[a_\alpha]-H_0\right).
```

Each mode's `a_complex` already specifies its amplitude and intensity, so the
API has no additional weights or combination enum.  Exact-zero modes are
skipped and an empty/all-zero drive returns `H_0` exactly.  Modes are never put
into one Peierls exponential, so coherent cross-mode interference and
mixed-mode virtual paths are absent.  This mode-diagonal construction agrees
with random-relative-phase averaging at leading weak-field `O(A^2)`.  It is
not an exact all-orders phase average: at higher field order, phase averaging
can retain cross-intensity terms such as `A_alpha^2 A_beta^2`, which this
helper intentionally omits.  Probabilistic experimental mixtures or duty
cycles should be combined by the caller as observables/configurations, rather
than interpreted as extra optical amplitudes here.

### Coherent linear-`q` effective model

`Model::floquet_effective_q_model` implements the same-size, primitive-cell
long-wavelength correction for **one common coherent Cartesian wavevector**
`q`.  Its field convention is

```math
a(r,t)=\operatorname{Re}\sum_{n>0}a_n e^{+iq\cdot r-in\Omega_0t}.
```

Modes with the same temporal harmonic are summed coherently before the
correction is evaluated.  The method first computes the existing `q = 0`
baseline—Bessel all-orders in field amplitude and van Vleck through the
requested inverse-frequency order 0, 1, or 2—and, when `order >= 1`, adds

```math
\delta_qH_{\rm eff}
=-\sum_{1\le n\le n_{\rm max}}\frac{q_a a_{ni}a^*_{nj}}{8nW}
\,D_a\{D_iH_0,D_jH_0\},
\qquad W=\hbar\Omega_0.
```

Here every `D_a` is an ordinary derivative with respect to the physical
Cartesian electron wavevector in the fixed Wannier/atomic gauge.  It is
computed analytically from real-space hopping, not with a finite-difference
mesh and not by differentiating band eigenvectors:

```math
H_{0,ij}(k)=\sum_Rt_{ij}(R)e^{ik\cdot d_{ijR}},\qquad
D_aH_{0,ij}(k)=i\sum_Rd_{ijR,a}t_{ij}(R)e^{ik\cdot d_{ijR}}.
```

The implementation therefore builds
`G_n(R) = i (a_n · d_{ijR}) t(R)`, evaluates the real-space anticommutator
`J_n = {G_n,G_n†}`, and applies the final directional derivative by replacing
each output block with `i (q · d_out) J_n(R)`.  The full Cartesian bond
`d = (R + tau_j - tau_i) · lat` includes the orbital embedding; for spinful
models the same orbital bond is used in each spin block.

The added term is controlled only through `O(A^2 q/W)` and `O(W^-1)` and
requires both `|a_n · d| << 1` and `|q · d| << 1` on relevant bonds.  An
`order = 2` option still retains the `q = 0` baseline through `O(W^-2)`, but
does not promote the linear-`q` term to `O(W^-2)`.  Here `n_max` is the
effective `harmonic_max`; higher harmonics are omitted from the correction,
and active nonpositive harmonics are rejected when nonzero `q` and
`order >= 1` request it.  With `order = 0` no linear-`q` term is added, while
`q = 0` returns the ordinary baseline exactly without imposing that positive-
harmonic restriction.  No optical supercell or photon bands are introduced.
Coherent beams with distinct wavevectors cannot be represented by this
one-model interface; use separate calls or a future graded/supercell solver
instead.

### Peierls time dependence

Each hopping dressed by `exp[−i a(t)·d_{ijR}]`. Fourier coefficient:

```math
C_n(d) = (1/T) ∫_0^T dt e^{inΩ₀t} exp[−i a(t)·d]
```

Main path: the generalized Bessel backend (exact, no `n_time`); the uniform
time-grid DFT is retained per link as the fallback for `R_α = |a_α·d| > 8` and
as the crate-internal cross-validation reference.

### Sambe Hamiltonian

```math
H^{(n)}_{ij}(k) = Σ_R t_{ij}(R) C_n(d_{ijR}) e^{i2πk·R}
```

```math
[H_F(k)]_{in,jm} = H^{(n−m)}_{ij}(k) + nΩ₀ δ_{nm}δ_{ij}
```

Quasienergies can be folded to `[−Ω₀/2, Ω₀/2)` via `fold_quasienergy`.

### Real-space Bessel backend (main path of `floquet_effective_model`)

van Vleck through `O(W⁻²)`, with `W = omega0_ev = ħΩ₀`, entirely in real space — no `k_mesh`, no `n_time`
(only `R > 8` fallback links touch the time grid):

```math
T_eff(R) = T_0(R) + Σ_{n=1}^{harmonic_max} comm_n(R) / (n W),
\qquad
comm_n(R) = (AB)(R) − (BA)(R),
```

with `A_R = T_n(R)`, `B_R = T_{−n}(R)` and BOTH discrete convolutions
`(AB)(R) = Σ A_{R−R'}B_{R'}` and `(BA)(R) = Σ B_{R−R'}A_{R'}` (the "P − P†"
single-convolution simplification is wrong for non-commuting blocks).
For `order = 2`, the implementation additionally evaluates
`Σ_{m≠0}[H_{−m},[H_0,H_m]]/(2m²W²)` and
`Σ_{m,m'≠0;m'≠m}[H_{−m'},[H_{m'−m},H_m]]/(3mm'W²)`.
The two operands of an inner/outer commutator may have different supports;
`real_space_commutator_with_supports` returns their Minkowski sum without
premature Hermitian symmetrization. The output support is the union through
the double Minkowski sum at order 1 and through the triple sum at order 2,
in lexicographic order — `target_hamR` is rejected. Hermiticity
`T(R) = T(−R)†` is enforced exactly on the final signed sum.

**Commutator-order convention (verified against the literature):** the code
uses `[H^(n), H^(−n)]/(nħΩ)`, opposite to the literature's
`[H^{(−n)}, H^{(n)}]/(nħω)`. Combined with our k-convention
`H(k) = Σ t(R)e^{+i2πk·R}` being the literature's mirror (`H_n(k) = H_n^lit(−k)`),
the two sign flips cancel for the TR-odd first-order terms — the results match
the literature pointwise. Do not "fix" either sign in isolation.

Machinery: `bessel_peierls_coeffs` (recursive one-mode convolutions,
adaptive tail truncation, `cutoff_margin ∈ [0,48]`, all integer arithmetic
checked), `floquet_harmonic_cache` (dedup by distinct `d`, `[u64; DIM]` bit
keys), `real_space_commutator` (two zgemm convolutions via the
`(A·B)^T = B^T·A^T` transpose trick, fp symmetrization),
`floquet_effective_model_legacy` (crate-internal k-space reference,
`pub(crate)` with `FloquetEffectiveOptions::target_hamR`/`with_target_hamR`).

### Graphene circular-light benchmarks (tests in `src/floquet.rs`)

`floquet::tests::graphene_*` pins the implementation against the analytic
derivation (arXiv:1511.00755 conventions, plan §9). Drive `a = α(1, i)` for
right-handed CPL reproduces `a(t)·e_l = α sin(ωt − 2πl/3)`; the single-mode
closed form gives `C_n(e_l) = J_n(α)e^{i2πnl/3}`.

- **A**: `H_n(k)` equals the literature `H_n(−k)` elementwise (1e-12);
  the literature phase uses Cartesian `k`, converted as `k_cart·e_l =
  2π·k_frac·R_int` (the non-orthonormal `lat` must not be dropped).
- **B**: order-0 effective model = `J·J_0(α)` renormalized literature `H_0`.
- **C**: Sambe quasienergy gap at `K = (1/3, 1/3)` converges to
  `Δ_exact = √((ħω)²+4g²) − ħω` (`g = (3/2)|J|α`), measured as the outermost
  folded branch — the minimal folded spacing is wrong (near-zero states from
  higher photon sectors). At K the `n ≡ 0 (mod 3)` harmonics vanish and
  `n ≡ 1 (mod 3)` survive with amplitude `3jJ_n(α)` (nilpotent).
- **D**: order-1 `d_z(k) = 2K_eff Σ_j sin(2πk·b_j)` with
  `K_eff = −(2J²/ħω) Σ J_n²(α)/n·sin(2πn/3)`; `n = 3m` terms vanish exactly;
  a sign-flipped implementation fails the test.

Second-order van Vleck is enabled by `FloquetEffectiveOptions::with_order(2)`
and cross-validated against both the legacy k-space path and the full Sambe
quasienergy spectrum (the residual changes from `O(Ω⁻²)` to `O(Ω⁻³)`).

### Key API types

| Type | Purpose |
|------|---------|
| `LightMode` | One harmonic component: `LightMode::new(harmonic, a_complex)` |
| `FloquetDrive` | `omega0_ev` + `Vec<LightMode>`; builder: `new()`, `with_modes()`, `add_mode()` |
| `FloquetTruncation` | Photon cutoff `n_max` and time-grid `n_time`; `n_sector()` = `2n_max+1` |
| `IncidentBasis` | Transverse polarization basis from incident direction |
| `FloquetEffectiveOptions` | Builder for van Vleck: `with_order(n)`, `with_harmonic_max(n)` (`with_target_hamR(rs)` is crate-internal, legacy path only) |
| `Floquet` trait | `floquet_model`, `floquet_ham_onek`, `floquet_band_onek`, `floquet_quasienergy_onek` |
| `Model::floquet_effective_model` | Inherent method returning same-size effective Model |
| `Model::floquet_effective_q_model` | Same-size coherent weak-field correction through `O(A^2 q/W)` for one common Cartesian `q` |
| `Model::floquet_effective_mode_resolved_model` | `H_0 + sum_alpha (H_eff[a_alpha] - H_0)` for mutually incoherent modes; no weights |

### Two Floquet paths

| Path | Returns | Basis size | When |
|------|---------|------------|------|
| `floquet_model` | Enlarged `Model` | `nsta·(2N+1)` | Exact, any Ω |
| `floquet_effective_model` | Same-size `Model` | `nsta` | Ω ≫ bandwidth; real-space Bessel backend, no `k_mesh` |
| `floquet_effective_q_model` | Same-size `Model` | `nsta` | Coherent weak-field, long-wavelength `O(A^2 q/W)` correction |
| `floquet_effective_model_legacy` | Same-size `Model` | `nsta` | `pub(crate)` k-space reference: cross-validation, custom `target_hamR` |

`floquet_model` encodes photon sectors as additional orbitals. Spinless basis
order: `(photon sector, orbital)`. Spinful: `(spin, photon sector, orbital)`.
It replicates the input atom metadata and orbital projections in the same
sector-major order. `floquet_effective_model` preserves the input metadata
unchanged and determines its own real-space support (Minkowski sum of the
input `hamR`); a custom `target_hamR` (legacy path only) must contain unique
vectors and be closed under `R -> -R`.

---

## Performance Notes

**`zaxpy`** (`src/ndarray_lapack.rs`): safe BLAS `y += alpha * x` for `Complex<f64>` slices.
Preferred over `Zip`/elementwise for direction-weight accumulation. Only call when
source and destination are standard contiguous slices.

**Allocation reduction**: avoid `Array1::from_vec(...)` and `.to_owned()` inside hot loops;
prefer preallocated buffers. For rayon folds over `mu` values, mutate a local
`Array1<f64>` accumulator directly.

**Autovec/SIMD**: simple contiguous slice loops autovectorize better than `ndarray`
indexed/transposed views. Use `RUSTFLAGS="-C target-cpu=native"` for AVX2/AVX512.

**Energy-cut hotspots** (in priority order):

1. `eval_*_at_lam` functions allocate `Vec<f64>` and `Vec<Complex<f64>>` per call.
   Pre-allocating thread-local buffers (or stack arrays for nsta ≤ 32) would
   eliminate allocation overhead entirely.
2. The zero-clone `TrackedSimplex` refactor already removed the old
   `accumulate_tetrahedron_*_kquad` matrix clones; remaining cost is the
   per-call scratch `Vec` allocation in `eval_*_at_lam` and the
   materialized interpolated matrices in the simplex quadrature helpers.
3. Nonlinear 3D lacks a sorted-μ sweep — loops over all μ per (tet, band).
   Same binary-search + range-add optimization as AHC 3D applies.
4. `energy_gradient_3d` recomputes the same gradient for every band at every μ
   within the same tetrahedron. Computing once per (tet, μ) would cut 3D
   intrinsic cost roughly in half.

---

## Refactoring Guidelines

- **No batch sed/Python**: modify files one at a time with proper tooling.
- **Parallel agents for multi-file changes**: core file first manually, then launch
  3-4 agents in parallel for distinct groups of files.
- **Commit after each successful `cargo check`**: prevents data loss from
  `git checkout` discarding uncommitted work.
- **Constructing models**: `Model::<false, 2>::tb_model(lat, orb, None)?` for spinless 2D;
  `Model::<true, 2>::tb_model(lat, orb, None)?` for spinful 2D;
  `Model::<false, 3, HasRMatrix>::from_hr(path, seed, 0.0)?` for 3D with position matrix.
- **Atom/orbital ownership**: `Model` owns the dense orbital arrays. `Atom`
  stores explicit typed `OrbitalId` handles, not a count or pointer. `None` in
  `tb_model(..., None)` is a genuine orbital-only model; it does not fabricate
  H atoms. Crystal symmetry always requires explicit `Atom` metadata.
- **Optional Atom magnetism**: every Atom stores `Option<[f64; 3]>` in Cartesian
  coordinates. Constructors default to `None`; use `set_magnetic_moment` and
  `clear_magnetic_moment`. `magnetic_crystal_symmetry_from_atoms` and the
  matching irreducible-mesh method consume this metadata directly, mapping
  `None` to a zero moment for cryspglib. Explicit `&moments` methods remain
  per-call overrides. Supercells, cuts, serde, and Model validation preserve or
  validate the optional moment.
- **External fields in symmetry analysis**: uniform electric/magnetic fields
  already encoded in the Hamiltonian are supplied as optional Cartesian vectors
  in `SymmetryParameters::external_fields`. They are per-analysis context, not
  `Model` fields. Rustb passes them explicitly into cryspglib. Return both the
  structural operation group and the operation subset that preserves E
  (time-even polar) and B (time-odd axial), including surviving combined
  anti-unitary operations. Effective meshes use this subset; structural
  high-symmetry tables error when the field has reduced the group. Magnetic
  mesh reduction requires explicit moments via `magnetic_irreducible_kmesh`;
  `irreducible_kmesh(..., time_reversal=true, ...)` is a caller assertion about
  the Hamiltonian. Character-table operation headers are canonical database
  operations, while detected structural operations remain in model basis.
- **Hamiltonian symmetry is Atom-based and explicit**: call
  `Model::check_hamiltonian_symmetry(provider, request)`. It always obtains
  structural candidates from Atom positions; an orbital-only model is
  rejected. The default checks the structural grey extension after optional
  E/B filtering. Use `AtomicOrbitalBasis` for strict automatic actions of
  Atom-owned, globally aligned Wannier90 `s/p/d/f` and hybrid projections; it
  supports spinless/spinful and unitary/antiunitary operations, and rejects
  projection subsets not closed under a target operation, duplicate radial
  channels, non-atom-centred functions, and other non-closed subspaces.
  `ScalarSiteBasis` remains the one-`s`-orbital
  special case. General Wannier gauges and local frames must implement
  `BasisSymmetryRepresentation`; missing metadata is `Unresolved`, never
  `Broken`.
- **Exact real-space convention**: for
  `g|b,R> = sum_(s,a) D_s[a,b]|a,W R+s>`, evaluate
  `K_g(R) = sum_(s,t) D_s^dag H(W R+t-s) D_t`. A unitary operation requires
  `H(R)=K_g(R)`; an anti-unitary operation requires `H(R)=K_g(R)*`. Compare the
  union of stored `hamR` and every inverse-transformed support point, treating
  absent blocks as zero. Never replace this proof by sparse k sampling.
- **Residual magnetic-group identification**: survivors must pass cryspglib's
  identity/duplicate/inverse/closure validator. Derive the final group's family
  Hall from its own spatial projection, then identify UNI/BNS. The original
  structural Hall is provenance only and is not a valid family-Hall hint after
  symmetry reduction. A nonclosed threshold result or unresolved action is
  explicitly `FinalMagneticGroup::Inconclusive`; never repair or guess it.
- **Forced symmetrization is strict and non-mutating**: call
  `Model::symmetrize_hamiltonian(target, provider, parameters)`. Validate the
  target group, then recompute its compatibility with the current lattice,
  Atom positions/types, optional Atom moments, and E/B context before invoking
  the provider or averaging H. Any target mismatch is
  `TargetMagneticGroupIncompatible`; never silently use a smaller group. The
  localized actions must form a projective magnetic corepresentation. Average
  the complete symmetry-generated support, restore Hermiticity, keep `R=0` at
  row zero, realign old `rmatrix` blocks by R, zero-fill generated rmatrix rows,
  and postcheck every target covariance equation. Return a new Model.
- **Band coreps diagnose rather than hide broken symmetry**: call
  `Model::calculate_irrep(options)` to detect the Atom-defined/field-effective
  magnetic group, or `calculate_irrep_for_group(target, options)` for an
  explicit target. These band-label entry points always use the atom-centred
  orbital IDs and `Model::orb_projection`; they do not accept a custom basis
  provider. Resolve and validate every localized target action before launching
  any eigensolver; an incomplete orbital shell or ambiguous Wannier gauge is a
  hard `IrrepBasisRepresentation` error; actions that do not compose with the
  canonical factor system for `SPIN` (linear spinless, spin-half double group
  and T-squared-minus-one spinful) are also a hard pre-solve error. Check
  exact real-space covariance under every target operation before any
  eigensolver, using a scale with the scalar onsite energy removed so the
  decision is invariant under `H -> H + C I`. If any operation breaks H,
  retain residuals and raw characters but mark every target label `???`, even
  when a sampled eigenspace happens to close. Once preflight is complete,
  resolve target actions, validate generator-by-operation composition, check
  real-space covariance, and process independent high-symmetry k points in
  parallel while preserving canonical order and deterministic first errors.
  Group consecutive degenerate
  bands with an energy-origin-invariant threshold, compute unitary sewing
  characters and antiunitary closure residuals, and fit formal corep
  multiplicities. Only integer multiplicities, small reconstruction residual,
  and a closed projected subspace may receive a database label. Print every raw
  complex unitary character; mark antiunitary entries `N/A`. A nonprimitive
  cell whose translations alias in the database frame requires unfolding and
  must error rather than receive primitive-cell labels. Keep BLAS
  single-threaded under this outer Rayon parallelism.
- **Current integration status (2026-08-26)**: milestones A–E, Hamiltonian
  certification/MSG identification F1, forced symmetrization F2, and parallel
  high-symmetry band-corep analysis are implemented, together with strict
  automatic global-frame atomic-orbital actions through `f` and all Wannier90
  hybrid families. The focused
  Hamiltonian suite covers pure/hybrid orbital actions, spinless/spinful
  antiunitary operations, cell representatives, and full cubic
  corepresentations. Strict release clippy and full-suite counts must be
  refreshed whenever this section is used as release evidence.
- **Still deferred**: explicit radial-channel metadata, automatic local-frame
  or SOC-entangled Wannier representations, gauge-covariant Peierls
  transformations, supercell band-irrep unfolding, and wiring weighted meshes
  into response solvers remain follow-up work in
  `CRYSPGLIB_INTEGRATION_PLAN.md`.

### Current cryspglib corep boundary used by Rustb (2026-08-30)

- Rustb consumes the strict
  `cryspglib::irrep::magnetic_summary::magnetic_irrep_summary_by_uni` surface.
  Each operation-aligned character is `Option<Complex64>`: `Some(z)` is a
  computed value, including a defined zero, while `None` is reserved for a
  genuinely pending Type-A antiunitary column.  Never convert this distinction
  back into placeholder `f64` zeros.
- Ordinary, spinor, and compound coreps are already merged by cryspglib with
  explicit source provenance and authoritative dimensions.  Rustb must not
  reconstruct source tables, parse compound Miller--Love labels, pair raw
  character arrays with Hall operations, or run a second recovery pass.
- The former scientific blockers are implemented: compound constituent
  orbits, general Type-C partner transport, lattice/centering Bloch phases,
  and general crystallographic SU(2) frame transport.  Type-C evaluates
  `chi(a0^-1 h a0)^*` with exact Seitz reduction, the residual lattice phase,
  and the double-group central sign.  The relevant cryspglib commits are
  `efc7e24`, `9e956c4`, `0851ca9`, `1de71e1`, and `83ca8e8`; Rustb consumes the
  plural compound surface from `da4df70`.
- The release-only `exhaustive_strict_complex_summary_acceptance_census` is a
  hard regression gate.  Its 2026-08-30 result is `summaries=1651`,
  `points=10390`, and `coreps=52793`.  A future strict-summary failure is a
  correctness regression and must not be hidden as `???` or routed through a
  consumer-side fallback.
- The corresponding Rustb release library gate is `261 passed / 2 ignored`;
  the two ignored full-library censuses pass explicitly.  Release doc-tests are
  `22 passed / 2 ignored`, and the `intel-mkl-system,cryspglib` all-target check
  succeeds.
- Type-A antiunitary characters may be
  `TypeAAntiunitaryPending`.  Rustb fits the available unitary characters and
  prints antiunitary entries as `N/A`; it must not present the placeholder
  zeros as computed antiunitary characters.  This does not prevent irrep/corep
  labels because antiunitary traces are basis/gauge dependent and the Wigner
  type plus unitary character row supplies the formal fitting data.
- cryspglib currently supplies fixed-k magnetic-little-group coreps, not
  full-k-star corepresentations.  Rustb's present high-symmetry band labeling
  uses the fixed-k data; any future star/unfolding API must treat this as an
  explicit missing capability.
- The old real-valued cryspglib corep APIs remain only as deprecated
  compatibility surfaces.  New Rustb code must use the strict complex summary;
  reintroducing the legacy `Vec<f64>` path would again erase genuine imaginary
  characters and conflate a pending antiunitary value with a physical zero.
