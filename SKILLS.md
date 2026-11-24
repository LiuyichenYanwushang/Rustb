# Rustb 0.7 — Practical API Guide

This guide follows the const-generic `Model<SPIN, DIM, R>` API in the current
source tree. For mathematical definitions and complete error semantics, use the
generated rustdoc.

Canonical public modules are `Rustb::model::physics` (Bloch Hamiltonians and
DOS) and `Rustb::model::utils` (lattice-vector lookup and array helpers).
Doc-hidden `model_physics`/`model_utils` aliases remain under both
`Rustb::model` and the crate root. The module reorganization does not move
inherent `Model` methods or the root `find_R`/`remove_row`/`remove_col` imports.
Berry phases and Wilson loops belong to `geometry`; Berry-curvature responses
belong to `response`.

`Model::validate()` checks array shapes, finite Hamiltonian/position data,
an invertible lattice, and atom/orbital references. Deserialization and
high-level response/DOS entry points run these checks. It does not certify
Hermiticity or every real-space support convention; directly editing public
model fields still requires the caller to preserve those invariants.

Most snippets below assume:

```rust
use ndarray::{arr1, arr2, array, Array1};
use num_complex::Complex;
use Rustb::*;
```

They are intended to run inside a function returning `Rustb::Result<()>`.

## 1. Model construction

### Model type parameters

| Parameter | Meaning |
|---|---|
| `SPIN: bool` | `false` for spinless; `true` for a spin-up/spin-down basis |
| `DIM: usize` | Real-space dimension, normally 1, 2, or 3 |
| `R: RMatrixData` | `NoRMatrix` by default; `HasRMatrix` stores position matrix elements |

For concrete `HasRMatrix`, `use Rustb::RMatrixData;` (or `use Rustb::*;`) keeps
`.select(axis, indices)` wrapper-preserving; without the trait in scope, deref
calls ndarray's `select`, returning `Array4<Complex<f64>>`. For generic
`R: RMatrixData`, selection returns `R`: Rustb's `HasRMatrix` wraps the selected
`Array4`, while `NoRMatrix` has no storage and returns itself. The shared method
name is intentional.

```rust
let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let orb = arr2(&[[0.0, 0.0], [0.5, 0.5]]);

let spinless = Model::<false, 2>::tb_model(lat.clone(), orb.clone(), None)?;
let spinful = Model::<true, 2>::tb_model(lat, orb, None)?;
```

`lat` is a square `DIM × DIM` matrix whose rows are real-space lattice
vectors. Each row of `orb` is an orbital position in fractional coordinates.

### Hoppings and onsite terms

```rust
let mut model = Model::<false, 2>::tb_model(
    arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    arr2(&[[0.0, 0.0]]),
    None,
)?;

model.set_hop(-1.0, 0, 0, &array![1, 0], None);
model.add_hop(-0.5, 0, 0, &array![0, 1], None);
model.set_onsite(&arr1(&[0.2]), None);
```

`set_hop` replaces a hopping and `add_hop` accumulates it. Both maintain the
Hermitian-conjugate hopping at `-R`.

For a spinful model, `None` means a spin-independent identity term. Use
`SpinDirection::X`, `SpinDirection::Y`, or `SpinDirection::Z` for a Pauli
component:

```rust
let mut spinful = Model::<true, 2>::tb_model(
    arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    arr2(&[[0.0, 0.0]]),
    None,
)?;
spinful.add_hop(0.1, 0, 0, &array![1, 0], SpinDirection::Z);
```

### Spin operators

Spinful models provide the dimensionless operator `S_a / ℏ` directly:

```rust
let sx = spinful.build_spin_matrix(SpinDirection::X);
let sy = spinful.build_spin_matrix(SpinDirection::Y);
let sz = spinful.build_spin_matrix(SpinDirection::Z);
assert_eq!(sz.dim(), (spinful.nsta(), spinful.nsta()));
```

The return type is `Array2<Complex<f64>>`; the argument is a `SpinDirection`,
with no `Option` or `Result`. The matrix is `σ_a ⊗ I_norb / 2` in the basis
`(all ↑ orbitals, all ↓ orbitals)`: `Sy` has `-i/2` in its upper-right
orbital diagonal and `+i/2` in the lower-left. This method is available only
on `Model<true, DIM, R>`, for either position-matrix storage type, without
requiring atoms or projections. A spin current additionally combines this
operator with velocity as `{S_a / ℏ, v} / 2`.

### Orbital projections

`orb_angular` requires orbital projections and an explicit owning atom for
every orbital:

```rust
let mut model = Model::<false, 2>::tb_model(
    arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    arr2(&[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    Some(vec![Atom::with_orbitals(
        array![0.0, 0.0],
        AtomType::C,
        [OrbitalId::new(0), OrbitalId::new(1), OrbitalId::new(2)],
    )]),
)?;
model.set_projection(&vec![OrbProj::px, OrbProj::py, OrbProj::pz]);
let angular_momentum = model.orb_angular()?;
assert_eq!(angular_momentum.dim(), (3, model.nsta(), model.nsta()));
```

The result contains `(Lx, Ly, Lz) / ℏ` in the model basis, even for a 1D or
2D crystal. A spinful model uses `diag(L_a, L_a)` in the basis
`(all ↑ orbitals, all ↓ orbitals)`; it does not include spin angular momentum.
With the row-ket eigenvectors `C[band, basis]` returned by `solve_onek`, use
`C.mapv(|z| z.conj()).dot(&L_a).dot(&C.t())` to obtain a band-basis operator.

This is a local atomic approximation derived from the angular projections;
inter-atom blocks are zero. The projection metadata does not distinguish
radial shells or local coordinate frames. Use an orthonormal angular basis
consistent with the model. A truncated shell returns `P L_a P / ℏ`, which
need not obey the full angular-momentum commutators. This operator is not
the Bloch-state orbital magnetic moment or bulk orbital magnetization.

### Wannier90 import

Use `Wannier90::from_hr` for both model types. It reads three-dimensional
Wannier90 data and requires `_r.dat` only for `HasRMatrix`:

```rust
use Rustb::{HasRMatrix, Model, Wannier90};

let model: Model<false, 3> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;

let model_with_r: Model<false, 3, HasRMatrix> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;
```

The `HasRMatrix` form includes position-matrix contributions in velocity
operators. `_r.dat` and `_hr.dat` must declare the same set of translations;
block order can differ. The importer rejects truncation, invalid indices, zero
weights, and overflowing sizes. `zero_energy` must be finite. Element labels are matched
case-insensitively, so the species token of a `.win` projection line, the
`atoms_cart`/`atoms_frac` labels and the extra-atom labels of `_centres.xyz`
may be spelled `v`/`V` or `cl`/`Cl`; the orbital token after `:` remains
case-sensitive. Orbital labels follow the Wannier90 built-in angular-function
order (`p`: pz, px, py; `d`: dz2, dxz, dyz, dx2-y2, dxy). The base order is the
delivered projection block (each projection line over its matching atoms in atom
order; this matches Wannier90's state order when the `.win` file matches the
run). When `_centres.xyz` stores one atom-centred contiguous block per atom,
whole blocks are matched to the nearest atom, so a block order differing from the
atom list is handled; bond-centred hybrids and ambiguous blocks fall back to the
delivered projection order with a warning, and the labels can then be wrong if
that `.win` file was reordered relative to the run. Optional wsvec files and coordinate unit conversion retain
their existing behavior.

When exported from the same model, `_hr.dat` and `_r.dat` have identical
translation block order: ascending lexicographic `(Rx, Ry, Rz)`, with `Rz`
varying fastest. Only the file block order changes; 1-based matrix indexing
and the physics are unchanged. Neither writer modifies the model or
synthesizes missing conjugate blocks:

```rust
use Rustb::OutPut;

model_with_r.output_hr("out/", "wannier90")?; // 5I5,2F12.6; unit weights
model_with_r.output_r("out/", "wannier90")?;  // 5I5,6F12.6; Cartesian Å, no weights
```

The files round to six decimal places. Use the in-memory model or a full-precision
serialization for smaller symmetry residuals. Non-finite and over-width fields
are rejected before the destination file is opened.

### Aligning orbital positions with their atoms

`shift_to_atom` and its alias `move_to_atom` borrow `&self` and return
`Result<Self>`. Keep the returned model rather than expecting an in-place edit:

```rust
let aligned = model_with_r.move_to_atom()?;
assert_eq!(aligned.ham, model_with_r.ham);
```

Only the fractional coordinates of atom-owned orbitals change, to those of
their parent atoms. The real-space Hamiltonian and position matrix are unchanged;
this is not a gauge transformation. Explicit edits (`set_*`, `add_hop`,
`remove_*`, `reorder_atom`) still act in place.

### Numeric bounds and serialized models

Hopping amplitudes use `Into<Complex64>`; `f64` and `Complex64` callers remain
valid. The custom `HopUse`, `ToFloat`, and `UseFloat` traits were removed.
`gen_kmesh` and `gen_krange` accept `num_traits::Float`, including `f32`/`f64`.
Text output uses standard formatting bounds.

Model deserialization keeps the existing fields and legacy atom representation.
Unknown or duplicate fields are rejected, as are mismatched spin/dimension,
missing required fields and invalid model data. `HasRMatrix` requires `rmatrix`;
`NoRMatrix` still accepts a supplied matrix and discards it after parsing.


### Model inspection

```rust
let dimension = model.dim_r();
let orbitals = model.norb();
let states = model.nsta();
let atoms = model.natom();
let reciprocal_lattice = model.rec_lat()?;

let lattice = &model.lat;
let positions = &model.orb;
let hopping_vectors = &model.hamR;
let hopping_blocks = &model.ham;
```

## 2. k-points, bands, and density of states

### Uniform mesh

```rust
let k_mesh = arr1(&[51usize, 51]);
let k_points = gen_kmesh::<f64>(&k_mesh)?;
let bands = model.solve_band_all_parallel(&k_points)?;
```

`gen_kmesh` returns fractional reciprocal coordinates with shape
`(product(k_mesh), DIM)`.

### High-symmetry path

```rust
let path = arr2(&[
    [0.0, 0.0],
    [2.0 / 3.0, 1.0 / 3.0],
    [0.5, 0.5],
    [0.0, 0.0],
]);
let labels = vec!["Γ", "K", "M", "Γ"];

let (k_points, k_distance, node_distance) = model.k_path(&path, 501)?;
let bands = model.solve_band_all_parallel(&k_points)?;
model.show_band(&path, &labels, 501, "band_output")?;
```

`show_band` writes plotting data and a PDF below the output directory supplied
as its final argument.

### One k-point and Bloch Hamiltonian

```rust
let k = arr1(&[0.25, 0.0]);
let h_atom = model.gen_ham(&k, Gauge::Atom);
let h_lattice = model.gen_ham(&k, Gauge::Lattice);
let band = model.solve_band_onek(&k)?;
```

For several points, construct a batch with a shared Fourier sum:

```rust
let points = arr2(&[[0.25, 0.0], [0.3, 0.1]]);
let hams = model.gen_ham_batch(&points, Gauge::Atom);
// hams.shape() == [2, model.nsta(), model.nsta()]
```

`gen_ham_batch` creates no Rayon tasks. It uses GEMM to reuse hopping data
across points and supports both gauges and non-contiguous input views.
Call it on bounded subsets when the complete H(k) array would be too large.

Both serial and parallel band/eigenvector solvers use this shared batch
constructor. Parallel jobs receive contiguous k-point ranges and process them
in batches sized from a total per-process Fourier buffer allowance `U`:

```
U = detected available memory / local process count
per-worker share = U / N
```

`N` is the current Rayon pool size for both serial and parallel solvers. A
serial solver uses one share, including inside an outer parallel loop; it
does not claim the whole process allowance. The share is fixed for the entire
call, including when there are fewer k-points than workers or only tail jobs
remain; idle workers' shares are never redistributed. For example, total
allowances of 256 GiB and 1.5 TiB with 64 workers give 4 GiB and 24 GiB per
worker, respectively. These are illustrative inputs, not measured machine RAM.

Memory detection targets Linux, taking the minimum of host `MemAvailable`
and each finite cgroup v1/v2 limit minus its current usage, including the
process's named cgroup, its ancestors and any exposed controller-root limit. A fully
readable unlimited hierarchy uses host availability; a physical controller
root need not expose a limit. An unreadable named ancestor or missing usage
for a finite limit makes detection unknown. Unavailable memory data or unknown
hierarchies use a total allowance of 128 MiB. The local process count is the
largest positive count reported by `SLURM_NTASKS_PER_NODE`,
`OMPI_COMM_WORLD_LOCAL_SIZE`, `MV2_COMM_WORLD_LOCAL_SIZE` or `MPI_LOCALNRANKS`,
defaulting to one; no MPI dependency is required.

`U` is resolved once per process, with no 128 MiB floor or 1 GiB cap. Set
`RUSTB_FOURIER_MEMORY_MIB` before the first solve to override the **total** `U`
with a positive integer number of MiB; the override is not divided by the
local process count. Zero, malformed and overflowing values use detection.
Both paths retain at least one byte and cap at the target's `isize::MAX`.
Pin this variable and the Rayon pool size (normally `RAYON_NUM_THREADS`) for
repeatable batching across runs: the machine snapshot can vary, and one-point
`zaxpy` versus multi-point `zgemm` sums can differ in their last bits. Set the
override when launcher counts or container memory limits are hidden.

The share sizes Bloch phase and Hamiltonian buffers, with no fixed point cap.
Every batch permits at least one k-point: a large Hamiltonian is not rejected
solely because it exceeds the share or even total `U`. Concurrency is reduced
when needed to keep such points within `U` where possible. This is a soft
Fourier buffer scheduling target, not a hard per-thread or process memory cap.
LAPACK buffers, orbital phases and returned arrays are additional; the
`nk * nsta^2` eigenvector output of `solve_all_parallel` can dominate memory.

Batching is internal to the existing solver methods and requires no options.
Rayon chooses which workers run the jobs; they are not pinned to CPU cores.
Configure BLAS threading separately (for example `MKL_NUM_THREADS=1` with outer
Rayon parallelism).

### Eigenvalue ordering and eigenvector axes

All eight `Solve` methods return `Result`. Use `?` to propagate invalid-model,
k-point, generated-Hamiltonian, and eigensolver errors. A batch validates the
model once, even when it contains zero k-points. Successful results contain only
finite energies and eigenvector components: finite matrix entries alone do not
prevent spectral overflow. Selected-window solvers check only the returned bands,
so an omitted overflowing band does not invalidate a finite subset. Hermiticity
remains the caller's responsibility. `Berry` Wilson-loop methods also propagate
these errors.

`solve_band_range_onek(&k, (low, high), tolerance)?` selects energies in
`(low, high]`; `solve_range_onek` also returns row-ket eigenvectors. Bounds and
absolute convergence tolerance use the model's energy units; nonpositive
tolerances select LAPACK's default. Nonfinite or unordered bounds and nonfinite
tolerances return errors. No selected bands produces empty energies and, when
requested, an eigenvector matrix of shape `(0, nsta)`.

`solve_onek(&k)?` returns `(energies, evec)` with shapes `(nsta,)` and
`(nsta, nsta)`. Energies are in ascending order, including all spin states in
one ordering. Row `evec.row(n)` contains the **ket coefficients** belonging to
`energies[n]`; eigenvectors are not columns and the row is not a bra.

`solve_all(&points)?` and `solve_all_parallel(&points)?` return shapes
`(nk, nsta)` and `(nk, nsta, nsta)`, with indices `[ik, n]` and
`[ik, n, basis]`. The k axis preserves input order; each k-point is sorted
independently by energy. The basis axis follows the model: spin-up orbitals,
then spin-down orbitals for spinful models. This does not imply spin ordering
of the bands. Neither method tracks band character or aligns eigenvector
phases between k-points.

All three methods return Atom-gauge states. Let `C` be the returned two-dimensional
eigenvector matrix at one k-point, and `*` mean elementwise conjugation:

```text
H C^T = C^T diag(E)
C* C^T = I
O_band = C* O C^T
```

For example, with `op` in the same Atom gauge and basis:

```rust
let (energies, evec) = model.solve_onek(&k)?;
let ket = evec.row(0); // H.dot(&ket) ≈ energies[0] * ket
let op_band = evec.mapv(|z| z.conj()).dot(&op.dot(&evec.t()));
```

All internal full-spectrum calculations use one solver boundary that packs
logical matrices in Fortran layout and returns row kets, including strided
views. This avoids the raw `ndarray-linalg 0.18.1` C-layout complex transpose
behavior. `ndarray_linalg::conjugate()` means conjugate transpose, while
`mapv(|z| z.conj())` only conjugates; use the row-ket formula above.

Eigenvectors have arbitrary overall phases; a degenerate subspace admits
arbitrary orthonormal rotations. Compare residuals, overlaps or subspace
projectors instead of requiring elementwise equality across calls or backends.
The `Solve::solve_onek` and `Solve::solve_all` rustdoc contains executable
complex-H examples checking these conventions.

The low-level `ndarray_lapack::{eigh_x, eigh_r, eigvalsh_x, eigvalsh_r,
eigvalsh_v}` functions return `Result`. They require square matrices, accept
C/F/strided views, and honor the requested triangle of the logical input.
`eigh_x` and `eigh_r` return row kets directly; do not conjugate their result.

### Bloch orbital angular momentum

`orbital_angular::OrbitalAngular` computes the band-space Bloch operator
`L / hbar`, including complex off-diagonal elements, using Busch, Mertig and
Göbel (2023), Eq. (3). This differs from the atomic-orbital matrices returned
by `Model::orb_angular()` and from bulk orbital magnetization.

```rust
use Rustb::orbital_angular::OrbitalAngular;
// Energies in eV; lat and rmatrix in angstroms, as for velocities.
let angular = model.orbital_angular_momentum_onek(&k)?;
// shape: (3, nsta, nsta), axes Lx / Ly / Lz, ascending-energy band basis.
let lz_band0 = angular[[2, 0, 0]].re;
```

This isolated-band formula rejects gaps <= `1e-10` eV, non-Hermitian
operators, and invalid inputs. Band matrix elements transform covariantly
under band rephasing. In 2D only Lz is nonzero.

`k_path` on both `Model` and `SurfGreen` requires at least two finite nodes,
distinct consecutive nodes, and at least as many samples as nodes. It retains
every node, including nodes bordering very short segments. Distances use the
object's reciprocal-lattice metric without `2*pi`; malformed or singular
lattices return an error. Mesh and plane generators reject zero and overflowing sizes.
`phy_0` is the superconducting flux quantum `h/(2e)` in webers.

Intrinsic nonlinear Hall is charge-current only: its signature takes neither a
`spin` nor an `eta_ev` argument, so those requests cannot be expressed at all.
Direct and energy-cut intrinsic kernels both omit gaps <= `1e-10` eV.
Finite-temperature Hall
energy-cut convolution samples only the requested Fermi windows, so its
sample count does not grow as `1/T`.

`FloquetTruncation::n_sector()` and `sectors()` now return `Result`; use `?`
in fallible code. Sambe entry points check allocation arithmetic and reject a
drive whose link needs more samples than the per-link fallback grid can
alias-free, instead of silently aliasing high harmonics. For a k scan, construct `floquet_model` once and use its
ordinary band solvers to reuse the Fourier work.

Examples write to `target/example-output/`; plotting tests write to
`target/test-output/`. Local documentation needs no workspace-specific Cargo
configuration. To include the custom header, run from the repository root:
`RUSTDOCFLAGS="--html-in-header docs-header.html" cargo doc --no-deps --features openblas-system`.

### Density of states

```rust
let (energy, dos) = model.dos(
    &arr1(&[101usize, 101]),
    -4.0,
    4.0,
    801,
    0.02,
)?;
```

The final two arguments are the number of energy points and Gaussian smearing
width.

## 3. Hubbard mean field

`HubbardModel` requires a spinful bare model. Supply either one interaction per
orbital or a uniform value:

```rust
let mut bare = Model::<true, 1>::tb_model(
    array![[1.0]],
    array![[0.0]],
    None,
)?;
bare.add_hop(-1.0, 0, 0, &array![1], None);

let hubbard = HubbardModel::with_uniform_u(bare, 2.0)?;
```

Choose whether self-consistency keeps the chemical potential fixed or keeps
the initial electron filling:

```rust
let constraint = MeanFieldConstraint::FixedInitialFilling {
    reference_mu: 0.0,
};
let occupation = Occupation::FermiSmearing { width: 0.01 };
let mut params = MeanFieldParams::new([200], constraint, occupation);
params.max_iterations = 500;
params.density_tolerance = 1e-10;
params.mixing = 0.2;
params.initial_magnetization = InitialMagnetization::UniformVector {
    moment_per_orbital: [1e-3, 0.0, 0.0],
};

let model = hubbard.solve_hartree_fock(&params)?;
```

For `FixedInitialFilling`, Rustb first evaluates the bare-model filling at
`reference_mu` using direct Fermi occupations on the requested k-mesh. It then
solves for a new chemical potential at every iteration. The returned value is
an ordinary `Model<true, DIM, R>` with the converged chemical potential shifted
to zero. The unrestricted Hartree-Fock iteration uses a complete local `2 × 2`
spin-density matrix, so non-collinear `Sx`/`Sy` order generates the corresponding
complex Fock spin-flip terms.

Spin observables are available directly from any spinful model:

```rust
let spin_by_band = model.spin_expectation_onek(&arr1(&[0.25]))?;
let local_spin = model.local_spin_moment(&[200], 0.0, occupation)?;
let total_spin = model.spin_moment(&[200], 0.0, occupation)?;
let filling = model.electron_filling(&[200], 0.0, occupation)?;
```

`local_spin_moment` has shape `(norb, 3)`. Spin values are in units of `hbar`.
For custom non-collinear seeds, use
`InitialMagnetization::CustomVectors(Array2<f64>)`, whose rows contain
`[p_x, p_y, p_z] = 2<S>/hbar`.

## 4. Velocity, response, and quantum geometry

Brillouin-zone responses share **one** small grid configuration,
`Parameters<DIM>`:

| Field | Meaning |
|-------|---------|
| `conditions.t_kelvin` | Temperature in kelvin; `Sampling::Fixed(0.0)` is the exact zero-temperature step function |
| `conditions.mu_ev` | Chemical potential in eV |
| `conditions.omega_ev` | Photon / perturbation frequency in eV; DC entry points require `Sampling::Fixed(0.0)`, while the occupation-weighted per-k helpers require a fully fixed DC point. The band-resolved per-k methods take no `Conditions` at all |
| `kmesh` | Uniform k-mesh `[usize; DIM]` |
| `integration` | `Integration::Direct` / `Simplex` / `EnergyCut` |

Everything a single method chooses is an argument of that method, never a field
of the shared structure. Nothing is defaulted: `Integration` and
`FieldSymmetry` implement no `Default`, and `eta_ev` is a plain `f64` wherever
a denominator is broadened.

| Method-specific argument | Type | Where it appears |
|--------------------------|------|------------------|
| directions | `[[f64; DIM]; 2]` (rank 2) or `[[f64; DIM]; 3]` (rank 3, `(current, field_1, field_2)`) | every Brillouin-zone entry point |
| `eta_ev` | `f64` | every entry point that broadens a denominator; absent from `intrinsic_nonlinear_hall` |
| `spin` | `Option<SpinDirection>` | `hall_conductivity`, `extrinsic_nonlinear_hall`, `berry_curvature_at`, `occupied_berry_curvature_at/_on` |
| `field_symmetry` | `FieldSymmetry` | `extrinsic_nonlinear_hall` only |

Because a method only accepts the arguments it reads, requesting a spin current
from a charge-only method is a **compile error** rather than a runtime
rejection. `Hall` and `extrinsic_nonlinear_hall` have separate inherent
implementations for `Model<false, DIM, R>` and `Model<true, DIM, R>`; the
spinless one returns `TbError::SpinNotAllowed` for `spin: Some(_)`. Wrappers
generic over `const SPIN: bool` must specialize these calls or supply their own
trait bound for dispatch. Generic Berry-curvature callers can use the existing
bound `Model<SPIN, DIM, R>: BerryCurvature<DIM>`.

Each axis is either `Sampling::Fixed(value)` or `Sampling::Values(series)`.
**At most one axis may be `Values`**: that axis is evaluated from one shared
k-mesh preparation, so eigenstates, velocity kernels and band tracking are
computed once however many samples you ask for. Sampling two axes at once is
rejected before any k-mesh work. Total cost is preparation plus the evaluations
at each sample; temperature scans still repeat the energy cuts, convolutions
or simplex quadrature on the shared vertex data.

```rust
let mu = Array1::linspace(-1.0, 1.0, 101);

// fixed T and omega, sweeping mu
let hall = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(30.0),
        mu_ev: Sampling::Values(mu),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [51, 51],
    integration: Integration::Direct,
};

// Directions, broadening and spin current are method arguments.
let hall_result = model.hall_conductivity(&hall, [[1.0, 0.0], [0.0, 1.0]], 1e-3, None)?;

// The simple cubic lattice of this section is 3D; a 2D model uses [[f64; 2]; 2].
```

`Conditions::fixed(t_kelvin, mu_ev, omega_ev)` pins all three axes at once.
Results carry `axis: ResponseAxis` — `Fixed`, or `Temperature` /
`ChemicalPotential` / `Frequency` with the sampled values — and
`single()` returns a scalar only for `Fixed`.

### Velocity operators

```rust
let k = arr1(&[0.2, 0.3]);
let (velocity, h_k) = model.gen_v(&k, Gauge::Atom);

let directions = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let (projected_velocity, h_k) =
    model.gen_v_projected(&k, Gauge::Atom, &directions);
```

`gen_v` returns an array with shape `(DIM, nsta, nsta)`.
`gen_v_projected` returns one operator for each row of `directions`.

The batch counterparts share phase factors and combine H plus all requested
derivative sums in one GEMM:

```rust
let (velocities, hams) = model.gen_v_batch(&k_points, Gauge::Atom);
let (projected, hams) =
    model.gen_v_projected_batch(&k_points, Gauge::Atom, &directions);
```

The velocity shapes are `(nk, DIM, nsta, nsta)` and
`(nk, directions.nrows(), nsta, nsta)`; H has shape `(nk, nsta, nsta)`.
These constructors create no Rayon jobs, and their output/workspace grows with
the supplied point count. Single-point `gen_v` and `gen_v_projected` use the
same implementation, allowing H and direction derivatives to share hopping
reads even at a single point. Position-matrix commutators and the existing
atom/lattice gauge conventions are retained. Each H(k) is still diagonalized
individually; no block-diagonal Hamiltonian mixing different k-points is built.

### Temperature and occupation

The temperature axis selects the electronic occupation: `0.0` is the exact
zero-temperature step function, a positive value is a Fermi-Dirac
distribution at that temperature.

```rust
let zero_temperature = Sampling::Fixed(0.0);
let physical_temperature = Sampling::Fixed(30.0);
let temperature_series = Sampling::Values(Array1::linspace(0.0, 300.0, 31));
```

Direct Fermi-surface calculations containing `-df/dE` require a positive
thermal energy `k_B T` with a finite peak derivative `0.25 / (k_B T)`. Zero
widths or overflowing peaks reject the whole `Direct` call, including when
they occur inside a `Values` series. Subnormal widths with finite peaks remain
valid.
Energy-cut algorithms represent the exact zero-temperature delta function.

### Berry curvature

```rust
let k = arr1(&[0.2, 0.3]);
let directions = [[1.0, 0.0], [0.0, 1.0]];

// Band-resolved: no temperature, chemical potential, k-mesh or integration.
let bands = model.berry_curvature_at(&k, directions, 1e-3, None)?;

// Occupation-weighted: one fixed DC Conditions, still no k-mesh.
let conditions = Conditions::fixed(300.0, 0.0, 0.0);
let occupied = model.occupied_berry_curvature_at(&k, &conditions, directions, 1e-3, None)?;
```

`bands.berry_curvature` and `bands.energies` contain one value per band. The
occupied variants sum with the occupation selected by the fixed temperature and
chemical potential. For a spin Hall kernel pass the last argument as
`Some(SpinDirection::Z)`. These per-k-point methods evaluate one DC state, so
every axis of the `Conditions` must be `Fixed` and `omega_ev` must be zero. A
sampled axis or a nonzero frequency returns `InvalidResponseParameter`. The
band-resolved form takes no `Conditions` at all, so it cannot be given a swept
one.

### Hall conductivity

```rust
let mu = Array1::linspace(-2.0, 2.0, 101);
let params = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(30.0),
        mu_ev: Sampling::Values(mu),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [51, 51],
    integration: Integration::EnergyCut,
};

let directions = [[1.0, 0.0], [0.0, 1.0]];
let result = model.hall_conductivity(&params, directions, 1e-3, None)?;
let sigma_vs_mu = result.conductivity; // one entry per sampled mu

// A spin Hall kernel states the spin current explicitly; a spinless model
// returns TbError::SpinNotAllowed instead.
let spin_hall = model.hall_conductivity(&params, directions, 1e-3, Some(SpinDirection::Z))?;
```

`result.axis` is `ResponseAxis::ChemicalPotential(mu)`, and `single()` returns
the scalar of a `Conditions::fixed` calculation. `Integration::Direct` performs
a uniform k-point sum, `EnergyCut` a band-tracked cut; both prepare the k-mesh
once and then weight every sample, and `EnergyCut` requires the sampled chemical
potentials to ascend. Sampling the temperature instead is the same call with the
`Values` series moved to `t_kelvin`.

### Nonlinear Hall response

All public rank-three directions are current-first — the three rows are
`(current, field_1, field_2)`:

```rust
let mu = Array1::linspace(-1.0, 1.0, 101);
let conditions = Conditions {
    t_kelvin: Sampling::Fixed(30.0),
    mu_ev: Sampling::Values(mu),
    omega_ev: Sampling::Fixed(0.0),
};
let directions = [
    [1.0, 0.0], // current
    [1.0, 0.0], // field 1
    [0.0, 1.0], // field 2
];

// The intrinsic response is charge-only: no spin, no eta_ev argument exists.
let intrinsic_result = model.intrinsic_nonlinear_hall(
    &Parameters {
        conditions: conditions.clone(),
        kmesh: [51, 51],
        integration: Integration::Direct,
    },
    directions,
)?;

// The extrinsic response states its broadening, spin current and field
// ordering explicitly.
let extrinsic_result = model.extrinsic_nonlinear_hall(
    &Parameters {
        conditions,
        kmesh: [51, 51],
        integration: Integration::EnergyCut,
    },
    directions,
    1e-3,
    None,
    FieldSymmetry::Ordered,
)?;
```

`FieldSymmetry::Symmetrized` averages the two external-field permutations,
`FieldSymmetry::Ordered` returns one raw ordered kernel. Both entry points share
one eigendecomposition per k-point between the two field orderings, and the
energy-cut path shares one band-tracking pass. `intrinsic_nonlinear_hall` takes
neither `eta_ev` nor `field_symmetry` and has no spin-current argument at all,
so those requests are compile errors. Both entry points require
`omega_ev: Sampling::Fixed(0.0)`. `NonlinearHallResult::single()` mirrors
`HallConductivityResult::single()`.

Both nonlinear entry points require a normal, finite, nonzero lattice determinant
and check **every** final conductivity sample. Nonfinite output rejects the
whole call with `InvalidResponseParameter`; no partially valid scan is returned.
Extrinsic squared gaps and broadening must remain representable (with rounding
headroom for energy-cut interpolation). These are ordinary `f64` range checks,
not arbitrary-precision evaluation: a finite exact answer can still be rejected
when the required intermediate representation is unsupported.

Direct integration samples `-df/dE` on k-points, so **every** sample must have
a finite peak derivative `0.25 / (k_B T)`. Zero widths or overflowing peaks
reject the whole call before any k-mesh work. Use
`Integration::EnergyCut` for the exact zero-temperature Fermi surface.

### Quantum geometry

```rust
let mu = Array1::linspace(-1.0, 1.0, 101);
let params = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(0.0),
        mu_ev: Sampling::Values(mu),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [51, 51],
    integration: Integration::Simplex,
};

let directions = [[1.0, 0.0], [0.0, 1.0]];
let result = model.quantum_geometry(&params, directions, 1e-3)?;
let metric = result.metric; // one entry per sample
let berry_curvature = result.berry_curvature;
```

For reusable band-resolved data, use the `QuantumGeometry` trait methods
`quantum_geometry_at` and `quantum_geometry_on`. They take only a k-point (or
list), the two directions and `eta_ev` — no `Conditions`, no k-mesh — so a
swept state cannot be passed to them at all. The at/on/Direct kernel divides
velocities by a scaled gap/broadening norm before multiplying, so uniformly tiny
or large energy scales do not themselves destroy the metric. Singular unbroadened
gaps, lost scaled-velocity components and unrepresentable geometry return
errors instead of successful NaN/Inf or an uncertifiable zero. Simplex
interpolates pair gaps directly, rather than subtracting separately rounded
absolute energies.
Simplex retains unscaled interpolated primitives and rejects overflowing,
underflowed or nonzero-subnormal velocity products, whose relative information
cannot survive interpolation. Denominators need rounding headroom above the
numerical cutoff and below f64::MAX; use Direct for extreme-scale geometry.

### Optical conductivity

Returns the retarded **interband** Kubo conductivity with `e²/hbar` omitted,
normalized by `|det(lat)|`; it excludes Drude terms and exactly degenerate
pairs. Frequencies and broadening are in eV. Use positive `eta` at resonances;
an unbroadened pole returns an error. For `det(lat) > 0`, its insulating DC
antisymmetric part is minus the occupied Berry-curvature integral returned by
`hall_conductivity`, which retains a signed-determinant convention.

Full Cartesian tensors share one eigendecomposition and one band-tracking
pass across components, retaining the Cartesian band velocities. Simplex
frequency scans interpolate energies and kernels once per quadrature point
and reuse them across the full frequency list.

```rust
let params = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(30.0),
        mu_ev: Sampling::Fixed(0.0),
        omega_ev: Sampling::Values(Array1::linspace(0.0, 4.0, 401)),
    },
    kmesh: [51, 51],
    integration: Integration::Simplex,
};

// One projected component.
let result = model.optical_conductivity(&params, [[1.0, 0.0], [0.0, 1.0]], 1e-2)?;
let sigma = result.conductivity; // (components, samples)

// The full ordered Cartesian tensor has its own entry point.
let tensor = model.optical_conductivity_tensor(&params, 1e-2)?;
```

This is the only entry point that may sample `omega_ev` — the four
DC entry points require `Sampling::Fixed(0.0)`. Optical response is
charge-only, so there is no `spin` argument to misstate. It may equally
sample `t_kelvin` or `mu_ev`, in which case the frequency stays fixed and the
columns follow that axis. The full-tensor entry point returns every ordered
Cartesian component `(0,0), (0,1), ..., (DIM-1,DIM-1)` as the rows of
`conductivity`, sharing one diagonalization and band-tracking pass; no empty
or sentinel direction matrix is involved.

## 5. Wilson loops and topology

Closed loops must end at a point differing from the first point by an integer
reciprocal lattice vector within `1e-9` per coordinate. All five `Berry` methods
return `Result` and accept `&[usize]` for the band selection: array slices and
borrowed `Vec`s both work. Select at least one distinct, in-range band; loops
need at least two finite k-points of the model's dimension. The selected band
subspace must be isolated, with sufficiently fine sampling; no gap is certified.

```rust
let occupied = [0usize];
let loop_k = arr2(&[
    [0.0, 0.0],
    [0.25, 0.0],
    [0.5, 0.0],
    [0.75, 0.0],
    [1.0, 0.0],
]);

let phases = model.berry_loop(&loop_k, &occupied)?;
let total_phase = model.berry_loop_det(&loop_k, &occupied)?;

let centres = model.wannier_centre(
    &occupied,
    &arr1(&[0.0, 0.0]),
    &arr1(&[1.0, 0.0]),
    &arr1(&[0.0, 1.0]),
    101,
    101,
)?;
```

`berry_loop` uses SVD-unitarized links and returns unsorted eigenphases;
`berry_loop_det` returns the phase of the raw determinant product. Both use
`-arg(W)` in **radians**. Detected singular overlaps, zero/nonfinite determinants
(including underflow), and solver/SVD failures return errors.

`berry_phase(&occupied, &loops)?` takes `(n_loop, n_k, DIM)` coordinates and
returns `(n_occ, n_loop)`, sorting each column independently without tracking
phase branches. An empty outer batch returns `(n_occ, 0)` after validation.
`wannier_centre` has the same axis order, `(n_occ, nk1)`, and units; divide by
`2*pi` for centres in lattice units. It includes endpoints along both directions,
requires `nk1, nk2 >= 2`, and `dir_2` must close modulo a reciprocal vector.

`berry_flux` takes the same origin and two directions plus `nk1, nk2 >= 1`.
Its shape is `(nk1, nk2, n_occ)` in radians, with plaquette orientation
start → +dir_1 → +dir_1+dir_2 → +dir_2 → start. All generated grids reject
overflowing allocation sizes and propagate individual loop errors.

## 6. Supercells, cuts, and surfaces

### Supercells and finite structures

```rust
let transform = arr2(&[[2.0, 0.0], [0.0, 3.0]]);
let supercell = model.make_supercell(&transform)?;

// Twenty layers along lattice direction 1.
let ribbon = model.cut_piece(20, 1)?;

// Hexagonal finite region; supported shape codes are 3, 4, 6, and 8.
let dot = model.cut_dot(10, 6, None)?;
```

For a 3D `cut_dot`, pass the two in-plane directions through
`Some(vec![dir_1, dir_2])`.

### Surface Green function

```rust
let surface = SurfGreen::from_Model(
    &model,
    0,       // open lattice direction
    1e-3,    // imaginary broadening
    None,    // optional maximum principal-layer range
)?;

let k_parallel = arr1(&[0.25]);
let (right_ldos, left_ldos, bulk_ldos) =
    surface.surf_green_one(&k_parallel, 0.0)?;

let energy = Array1::linspace(-2.0, 2.0, 401);
let (right_curve, left_curve, bulk_curve) =
    surface.surf_green_onek(&k_parallel, &energy)?;
```

The k-vector passed to the surface object has length `DIM - 1`.
`gen_ham_onek`, `surf_green_one`, `surf_green_onek`, `surf_green_path`,
`show_arc_state`, and `show_surf_state` return `Result`; propagate with `?`.
`surf_green_path` returns `(left, right, bulk)`, whereas the single-k methods
retain `(right, left, bulk)`. Path arrays have shape `(nk, nenergy)`.
The path and plotting methods no longer accept a `spin` argument. For example:

```rust
let (left, right, bulk) = surface.surf_green_path(&kpoints, -2.0, 2.0, 401)?;
surface.show_surf_state("surface", &path, &labels, 101, -2.0, 2.0, 401)?;
```

Path plots use logarithmic normalization and require finite positive densities;
a constant positive dataset maps to the middle of the color scale. File and
plotting failures propagate as errors.

## 7. Floquet driven systems

`LightMode::a_complex` is the rescaled vector potential `eA/hbar`, in inverse
lattice-length units. The drive is `a(t) = Re sum_l a_l exp(-ilΩt)`, and
`H(t) = sum_l H^(l) exp(-ilΩt)`. The positive-time basis `exp(+inΩt)` gives
Sambe blocks `H^(m-n) + nW δ_nm`, where `W = omega0_ev = hbar*Ω`. The first
van Vleck correction is `sum_(n>0) [H^(-n), H^(n)]/(nW)` (negative harmonic first).

```rust
let drive = FloquetDrive::with_modes(
    0.8,
    vec![LightMode::new(
        1,
        arr1(&[
            Complex::new(0.12, 0.0),
            Complex::new(0.0, 0.12),
        ]),
    )],
);
let truncation = FloquetTruncation::new(1);
let k = arr1(&[0.2, 0.1]);

let sambe_model = model.floquet_model(&drive, &truncation)?;
// sambe_model.ham and sambe_model.hamR are the real-space Hamiltonian.
let h_floquet = sambe_model.gen_ham(&k, Gauge::Lattice);
let mut quasienergy = sambe_model.solve_band_onek(&k)?;
quasienergy.mapv_inplace(|energy| fold_quasienergy(energy, drive.omega0_ev));
quasienergy.as_slice_mut().unwrap().sort_by(f64::total_cmp);

let effective =
    model.floquet_effective_model(&drive, None)?;

// Mutually incoherent beams: evaluate every nonzero mode independently and
// add its correction around one common static H0. There are no extra weights;
// each LightMode amplitude already fixes that beam's intensity.
let options = FloquetEffectiveOptions::new()
    .with_order(1)
    .with_harmonic_max(2);
let effective_incoherent =
    model.floquet_effective_mode_resolved_model(&drive, Some(&options))?;

// Keep van Vleck terms through O(W^-3), without truncating in field amplitude.
let third_order_options = FloquetEffectiveOptions::new().with_order(3);
let effective_third_order = model.floquet_effective_model(
    &drive,
    Some(&third_order_options),
)?;

// Nonzero q requires NoRMatrix. Use one common Cartesian wavevector in
// inverse-length units to add the same weak-field O(a^2 q/W) correction
// on top of the third-order uniform baseline.
let q_cartesian = arr1(&[2.0e-3, 0.0]);
let effective_linear_q = model.floquet_effective_q_model(
    &drive,
    Some(&third_order_options),
    &q_cartesian,
)?;
```

| API | Basis size | Intended regime |
|---|---:|---|
| `floquet_model` | `nsta * (2*n_max + 1)` | Real-space Sambe model with a photon cutoff |
| `floquet_effective_model` | `nsta` | Coherent off-resonant, high-frequency expansion |
| `floquet_effective_mode_resolved_model` | `nsta` | Mode-diagonal mutually incoherent correction sum |
| `floquet_effective_q_model` | `nsta` | One coherent common `q`, through `O(a² q/W)`; nonzero `q` requires `NoRMatrix` |

All these methods return an ordinary real-space `Model<SPIN, DIM, R>`
with the input `R` type. Build the Sambe model once and reuse `gen_ham`,
`gen_ham_batch` or the ordinary `Solve` methods across a k-path. Use
`fold_quasienergy` on the unfolded energies and sort afterwards if ascending
quasienergies are required.

For a spatially uniform field, the diagonal-position gauge gives the exact
time dependence within the model:

```text
H_ij(R,t) = exp(-i a(t)·d_ijR) [H_ij(R) + hbar da(t)/dt · (X - X_d)_ij(R)]
```

Here `d_ijR = (R + tau_j - tau_i) · lat`; `X_d` is diagonal in the localized
orbital basis, at the centres set by `orb` and the cell translations.
`HasRMatrix` supplies the full projected-position remainder `X - X_d`,
including its nonlocal blocks. For `NoRMatrix` the remainder is absent.
The same link-phase Fourier sampler uses generalized-Bessel coefficients
where resolvable and a self-sized per-link time-grid DFT otherwise. The
position term shifts the sampled phase harmonics by the Fourier labels of
`da/dt`. Neither uniform-drive path expands in field amplitude.

The returned `HasRMatrix` carries the bare projected position embedding,
remapped by lattice translation with new rows zero-filled. Sambe places a
copy on each photon diagonal: the basis order is `(photon, orbital)` for
spinless models and `(spin, photon, orbital)` for spinful models. The
embedding is not a micromotion-dressed observable; applying ordinary
response methods to the result does not establish physical Floquet transport.

`FloquetTruncation::new(N)` controls only the Sambe photon sectors
`-N..=N`. Their differences require Hamiltonian harmonics through `±2N`;
converge `N` for the quasienergies of interest. The effective-model APIs
instead use `FloquetEffectiveOptions` (defaults: `order = 1`,
`harmonic_max = 2`). Order 0 keeps the driven time average; orders 1–3
retain terms through `O(W^-1)`, `O(W^-2)`, and `O(W^-3)`, respectively.
With `h = harmonic_max`, the order-two outer indices `|m|, |m'|` are still
bounded by `h`, and the mixed difference `m'-m` can reach `±2h`. Order 3
adds the full cubic zero-average van Vleck coefficient, using generator
windows `0 < |m| <= h` for `A` and `0 < |m| <= 2h` for `B`. The contractions
read input Hamiltonian harmonics through `±3h`. If the Hamiltonian has
appreciable harmonics outside `±h`, increase `h` and check convergence. The
cutoff does not change the time average. To preserve the outer cutoff of an
old call that used a photon truncation, pass `.with_harmonic_max(2 * old_n_max)`.

Production spatial convolutions use only a zero-aliasing FFT grid, streamed
in slabs, with no user k-mesh or target hopping range. The effective hopping
support is determined from the harmonic support and can reach double
Minkowski sums at order 1, triple sums at order 2, and fourfold sums at order 3
(the cubic inverse-frequency term is degree four in the input Hamiltonian
blocks). With multiple Rayon threads, pin BLAS to one thread
(`OPENBLAS_NUM_THREADS=1` or `MKL_NUM_THREADS=1`) to avoid oversubscription.


For multiple mutually incoherent modes,
`floquet_effective_mode_resolved_model` computes
`H0 + sum_alpha (H_eff[a_alpha] - H0)`. It evaluates each nonzero mode on its
own and unions the hopping supports. This agrees with random-phase averaging
at `O(a²)`; it is not an exact all-orders phase average because higher-order
cross-intensity terms are omitted. Changing the mode order can change
floating-point rounding.

For either `R` type, `floquet_effective_q_model` with `q = 0` is identical to
`floquet_effective_model`. Nonzero `q` is an error for `HasRMatrix`. For
`NoRMatrix`, the linear-`q` correction uses ordinary Cartesian momentum
derivatives, not numerical differences: for a bond
`d = (R + tau_j - tau_i) * lat`, `∂_a H_0` is represented by `i*d_a*t(R)`.
It builds `G_n = a_n·∂ H_0`, forms `{G_n, G_n†}` by zero-aliasing FFT convolution,
and adds `+sum_(n>0) (q·∂){G_n, G_n†}/(8nW)` by multiplying the output
blocks by `i*q·d_out`. Same-harmonic amplitudes are summed coherently. Both
`|a_n·d| << 1` and `|q·d| << 1` are required; distinct coherent wavevectors
need a graded or commensurate-supercell treatment. `order = 0` adds no
linear-`q` term; higher orders include the same `O(a² q/W)` correction.
For this nonzero-`q` correction, the sum runs through `harmonic_max` and
active nonpositive harmonics are rejected.

For three-dimensional illumination, `IncidentBasis::from_direction` constructs
two transverse polarization vectors from a propagation direction.

## 8. Fermi-surface output

```rust
model.show_fermi_surface(
    &arr1(&[101usize, 101]),
    0.0,
    "fermi_surface",
)?;
```

Three-dimensional models can export data for FermiSurfer or XCrySDen:

```rust
model_3d.write_bxsf(&[50, 50, 50], 0.0, "fermi_surface")?;

write_spin_frmsf(
    &spin_up_model,
    &spin_down_model,
    &[50, 50, 50],
    0.0,
    "spin_split",
)?;
```

`show_fermi_surface_plane` extracts a two-dimensional slice of a 3D model.

## 9. Magnetic fields and unfolding

### Uniform magnetic field

```rust
// For a 2D model, mag_dir must be 2 (out of plane).
let magnetic = model.add_magnetic_field(
    2,
    [10, 10],
    1, // total integer flux quanta through the magnetic supercell
)?;
```

For a 3D model, `mag_dir` selects the lattice direction parallel to the field.

### Band unfolding

```rust
let transform = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
let supercell = model.make_supercell(&transform)?;
let path = arr2(&[[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]);

let spectral_weight = supercell.unfold(
    &transform,
    &path,
    401,
    -3.0,
    3.0,
    401,
    1e-2,
    1e-5,
)?;
```

The unfolding path uses primitive-cell fractional reciprocal coordinates.
It follows the same node-preserving sampling rules as `k_path`, with distances
computed from the primitive lattice `transform⁻¹ · supercell.lat`; `nk` must
be at least the number of path nodes.

## 10. Magnon models and exchange I/O

A `Magnon<DIM>` stores TB2J-style real-space exchange tensors: `ham` has shape
`(nR, 3n, 3n)`, the Cartesian components of magnetic site `i` occupy
`3i .. 3i + 3`, and row 0 of `hamR` is `R = 0` (single-ion anisotropy lives in
its diagonal `3x3` blocks). `J` is TB2J's unit-vector coefficient (`|S| = 1`,
ordered-pair double counting, eV); `spin` holds the spin quantum numbers
`S_i = |m_i| / 2` in units of `hbar`, and `axis` the reference moment
directions.

```rust
use ndarray::{array, Array3};
use Rustb::magnon::{Magnon, MagnonReference};

let lat = array![[3.0]];
let orb = array![[0.0]];
let spin = array![1.5];
let axis = array![[0.0, 0.0, 1.0]];
let ham_r = array![[0isize], [1], [-1]];
let mut ham = Array3::<f64>::zeros((3, 3, 3));
for d in 0..3 {
    ham[[1, d, d]] = 0.02; // J(R = +1)
    ham[[2, d, d]] = 0.02; // J(R = -1)
}
let magnon =
    Magnon::<1>::new(lat, orb, spin, axis, ham, ham_r, MagnonReference::collinear())?;
magnon.validate()?;
assert_eq!((magnon.nspin(), magnon.nsta(), magnon.nR()), (1, 3, 3));

// hr round trip: <prefix>_magnon_hr.dat + <prefix>_magnon_meta.toml
magnon.write_hr("magnon_out", "chain")?;
let restored = Magnon::<1>::read("magnon_out", "chain")?;
assert_eq!(restored.ham, magnon.ham);
```

The `_hr` file is a `nsta nR` header followed by `Rx Ry Rz a b J` lines
(1-based indices, real `J`, eV, every entry exactly once). The metadata sidecar
stores the lattice, site positions, spins, reference axes, the `q`/`n`/`uz`
reference structure, and the `eV` / `hbar` / `unit-vector` conventions.
`Magnon::from_hr(dir, prefix, lat, orb, spin, axis, reference)` reads the
matrix alone when the geometry is already known.

`validate()` enforces the invariants the solver and writer rely on: shapes,
finite data, positive spins, unit reference axes, unique translations with
`R = 0` first, the presence of every `-R`, and
`J_ij^ab(R) = J_ji^ba(-R)`. Direct field edits require re-validating.

`axis[i]` fixes the physical longitudinal spin direction. In contrast,
`MagnonReference::with_quantization_axis(&uz)` selects only the transverse
local-frame gauge: any finite unit `uz` must give identical energies (including
single-Q references and anisotropic exchange). Bogoliubov matrices may change
by a diagonal Nambu phase transformation. Parallel/antiparallel and nearly
parallel choices use an orthonormal, right-handed frame; `uz` does not rotate
the material's anisotropy or moment direction.

Spin-wave frequencies follow TB2J: the stored landscape `J` is converted to the
operator exchange `J / (S_i S_j)` before linear spin-wave theory, so a
ferromagnetic chain disperses as `omega(q) = (2 / S) [J(0) - J(q)]`.

```rust
use Rustb::magnon::MagnonSolve;

let q = array![0.25];
let omega = magnon.magnon_band_onek(&q)?;   // (nspin,), ascending, eV
let bands = magnon.magnon_band_all(&points)?; // (nq, nspin)
let h = magnon.gen_bdg(&q)?;                // 2n x 2n Hermitian Bogoliubov matrix
let jq = magnon.gen_jq(&q)?;                // 3n x 3n J(q), eV
let (omega_grid, dos) =
    magnon.magnon_dos(&array![16usize], 0.0, 0.2, 401, 0.002)?;
```

The `MagnonSolve` trait mirrors `Solve`: `magnon_band_onek`,
`magnon_band_all` and `magnon_band_all_parallel`. `Magnon` also implements
`Kpath`, so `magnon.k_path(&path, 101)` samples a band path with the same
node-preserving rule as `Model`. The solver uses Colpa's method through a
Hermitian eigenfactorization (not a Cholesky factor), so positive-semidefinite
Goldstone references do not require a positive-definite matrix. Numerical zero
modes may retain a small roundoff residual; this is not evidence of a physical
gap. A reference whose BdG matrix is indefinite beyond tolerance returns
`InvalidMagnonInvariant { invariant: "positive_semidefinite_bdg", .. }` instead
of a complex frequency.

### TB2J-style exchange extraction from a `Model`

`Magnon::from_model` takes one spinful collinear `Model<true, DIM, R>` and
splits its spin-major blocks; `Magnon::from_collinear` takes two spinless
models, one per channel. Both run the LKAG / TB2J extraction (magnetic force
theorem, local rigid spin rotation, complex-energy Green's functions) and
return a `Magnon` in the same conventions as the hr files.

Shift the input model to `E_F = 0` before extraction. For unshifted Wannier90
files, pass the original Fermi energy as `Model::from_hr`'s third argument
(`zero_energy`, the energy subtracted on import). Use the same shift for both
spin channels, or `0.0` for files already shifted to `E_F = 0`.

```rust
use Rustb::magnon::ExchangeParams;

// 5x5 k-mesh (forced odd), 65 nodes with the default CFR contour.
let kmesh = array![5usize, 5];
let params = ExchangeParams::new(&kmesh).with_nz(65);

// One spinful model: [all up orbitals, all down orbitals].
let magnon = Magnon::<2>::from_model(&model_spinful, &params)?;

// Or two spinless models.
let magnon = Magnon::<2>::from_collinear(&model_up, &model_dn, &params)?;
```

Magnetic sites come from the `Atom` metadata: an atom is magnetic when it
carries a magnetic moment, and its orbitals are the ones it owns
(`Atom::with_orbitals`); `with_magnetic_atoms` selects them explicitly and a
model without atoms falls back to one site per orbital.

Moments do not have to be supplied: Wannier90 files store none, so when neither
`with_spin`/`with_axis` nor the atoms provide them, the extraction derives the
site magnetization `m_i = Tr(rho_up - rho_dn)` from the two channels (TB2J's
`spinat`), with `S_i = |m_i| / 2` in units of `hbar` and the axis along `+z` or
`-z`. A direction supplied by the caller is cross-checked against that
magnetization: a global spin flip is a symmetry and passes, a per-site
contradiction raises `moment_direction_consistent` because it would silently
flip the sign of `J`.

The energy integral uses TB2J's **continued-fraction representation** by
default (`MagnonContour::Cfr { temperature_kelvin: 600.0 }`). CFR uses nodes
on the imaginary axis and does not use the energy window. To use a window set
by `with_energy_window`, select the upper-half-plane semicircle with
`ExchangeParams::with_semicircle_contour()`. Choose a window that covers the
occupied spectrum and avoids band energies at both endpoints. The Simpson
quadrature can converge slowly in metals when the arc ends at `E_F = 0`.

The extraction returns only isotropic exchange: `J_ij(R)` multiplies the
`3 × 3` identity. The sign is normalized by `sign(axis_i · axis_j)` so that
`J > 0` is ferromagnetic. Anisotropic exchange and Dzyaloshinskii–Moriya terms
would require spinor extraction; spin-off-diagonal blocks in a spinful input
model are rejected.

### Floquet superexchange

Start with an undriven `Magnon<DIM>` (for example, one read from exchange files
or extracted with TB2J). `magnon.floquet_superexchange(&drive, u_ev, harmonic_max)`
returns a new `Magnon<DIM>` on the same lattice and exchange support, retaining
its spin lengths and reference directions. The `FloquetDrive` is spatially
uniform; its commensurate modes are summed coherently before the Peierls
exponential. `u_ev` is a single positive interaction energy in eV, not a
quantity inferred from the input exchange tensors.

Each non-onsite bond is dressed independently in parallel. For its Cartesian
bond vector `d = (R + tau_j - tau_i) · lat`, let `C_m(d)` be the `m`th Fourier
coefficient of `exp[-i a(t) · d]` and `W = drive.omega0_ev = hbar*Omega`. The
second-order Aharonov–Bohm exchange correction is

```text
J_iso   = trace(J) / 3
F_N     = 1 + sum_{m=1..N} (|C_m(d)|² + |C_-m(d)|²) (mW)² / (U² - (mW)²)
J_light = J + (F_N - 1) J_iso I
```

`N = harmonic_max` is a nonnegative `isize`. Only the correction is truncated:
Parseval's identity retains the static `1`. Pairing both signs of `m` gives
identical factors for forward and reverse bonds even when `|C_m| ≠ |C_-m|`.
The method leaves the Dzyaloshinskii–Moriya and traceless symmetric parts
unchanged, as well as the entire onsite block (`i = j`, `R = 0`). It cannot
produce helicity-odd chirality or higher-order multispin terms.

The caller must justify treating every isotropic exchange component being
dressed as half-filled, single-orbital superexchange in the off-resonant
strong-coupling regime. This is a dressing of supplied exchange, not a way to
extract hopping, infer `U`, or identify and dress just one channel of a mixed
exchange. Increase `harmonic_max` and compare the exchange tensors and the
observables of interest until they converge. Rejection near a retained pole
is a numerical guard, not a verification of the material's charge gap or
absence of resonances. The method does not compute a quantum state or
establish stability of the retained magnetic reference.

```rust
use Rustb::floquet::{FloquetDrive, LightMode};
use Rustb::magnon::{Magnon, MagnonSolve};
use ndarray::array;
use num_complex::Complex;

let dark = Magnon::<2>::read("exchange", "insulator")?;
let drive = FloquetDrive::with_modes(7.0, vec![
    LightMode::new(1, array![Complex::new(0.08, 0.0), Complex::new(0.0, 0.04)])]);
let dressed = dark.floquet_superexchange(&drive, 4.0, 8)?;
// Check that increasing the cutoff leaves the relevant tensors/bands unchanged
// within your tolerance before using the result.
let higher_cutoff = dark.floquet_superexchange(&drive, 4.0, 12)?;
let q = array![0.25, 0.0];
let bands = dressed.magnon_band_onek(&q)?;
let bands_check = higher_cutoff.magnon_band_onek(&q)?;
```

See the [theory](<src/magnon/driven_theory.md>) for the bond factor and its scope.

### Driven exchange and odd-parity magnons

These are independent, lattice-specific tools, not wrappers around
`Magnon::floquet_superexchange`. The honeycomb closed form is specific to
circularly driven monolayers. The AA-bilayer helper supplies static couplings
and analytic LSWT; it does not generate driven interlayer or ring couplings.

`magnon::floquet::DrivenExchange::honeycomb` implements the Aharonov--Bohm
mechanism of the paper *Odd-Parity Magnons*: a half-filled Hubbard model on the
honeycomb lattice under circularly polarized light is reduced to an effective
spin Hamiltonian `J1`, `J2`, `chi` by a Floquet--Sambe strong-coupling expansion.
The microscopic single-orbital half-filled model has `S = 1/2`. The `S = 1.5`
examples below are phenomenological uses of the resulting spin-polynomial
coefficients, not a microscopic derivation for larger moments. Their easy-axis
anisotropy is an external spin-model input, not generated by the SU(2)-invariant
Hubbard reduction (`(S_z)^2` is constant for spin half).

```rust
use Rustb::magnon::floquet::{DrivenExchange, DrivenHubbardParams};

let params = DrivenHubbardParams {
    hopping_ev: 0.1,
    interaction_ev: 1.0,
    photon_ev: 2.5,   // off-resonant: U, 2U and the photon ladder must not mix
    lambda: 0.4,      // e E a / (hbar Omega)
    helicity: 1.0,    // xi = +-1
    harmonic_max: 12, // Floquet cutoff
};
let exchange = DrivenExchange::honeycomb(&params)?;
let (omega_up, omega_down) = exchange.honeycomb_dispersion(1.5, -0.01, &[1.0 / 3.0, 2.0 / 3.0])?;
```

`J1` and `J2` are even in the helicity while `chi` is odd, so reversing the
light handedness reverses the splitting. The dispersion is the paper's
`sqrt((A_k +- D_k)^2 - |B_k|^2)` with `D_k = 2 chi S^2 sum_nu sin(k.b_nu)`; the
two branches are degenerate without the drive, split with opposite signs at
`K = (1/3, 2/3)` and `K' = (2/3, 1/3)`, and satisfy `omega_up(k) =
omega_down(-k)` (odd parity, `f`-wave near Gamma).

`honeycomb_functions` and `honeycomb_dispersion` return `Result`; propagate
with `?`. They reject non-positive/non-finite spin and non-finite physical
inputs. A genuinely unstable branch returns a structured error rather than
`NaN`; only roundoff at a zero mode is tolerated. `DrivenHubbardParams::lambda`
is a non-negative amplitude (use `helicity` for handedness); a negative value
or non-finite helicity is rejected before Bessel evaluation.

### Worked example: the odd-parity dispersion

`examples/magnon_odd_parity` prints the Gamma-K-M-Gamma dispersion of Eq. (6)
beside the real-space solver, the valley splitting at `K`/`K'`, and the f-wave
pattern on a fixed Cartesian `|k|` circle. Circle points use `k_frac = k_cart B^-1`
with the model's own reciprocal lattice; path distances use the shared `Kpath`
metric (without `2*pi`). Sign counting skips zero samples using a relative
threshold and a floating-point subtraction-noise floor. Nonzero chirality must
produce a peak well above that floor before the six-sign-change assertion;
this guard is not an arbitrary-precision error certificate. Zero chirality
requires zero splitting, not six sign changes. Example unit tests check the
Cartesian roundtrip, sample count and cyclic zero handling.

Its final high-frequency propagator check is a separate phenomenological
exchange-bond drive test, not an additional AB correction or a second material
illumination step. It asserts that test's numerical residual:

```text
cargo run --release --features openblas-system --example magnon_odd_parity -- [lambda] [spin] [K_ev] [helicity]
```

Expected output with the defaults: solver-vs-analytic residual `1e-16`,
`Delta(K) = +3.6e-5`, `Delta(K') = -3.6e-5`, six f-wave sign changes, and
`2e-11` for the Sambe-to-effective residual.

### From the driven-exchange coefficients to a `Magnon`

`DrivenExchange::honeycomb_magnon(spin, anisotropy_ev)` builds a `Magnon` whose
spin-wave solver reproduces `DrivenExchange::honeycomb_dispersion` (the paper's
Eq. 6), which is the consistency check between the analytic coefficients and the
generic real-space tensor solver. The landscape mapping is

```text
J_land   = -(S^2 / 2) J_paper      (J1 and J2)
SIA_zz   = -K S^2                  (R = 0, i = i block)
dmi_land = chi S^3 / 2             (next-nearest-neighbour DMI)
```

The `S^2/2` carries both the spin-operator normalisation and the factor `2` of
TB2J's `C` term; using `J_land = -S^2 J_paper` doubles every frequency. The
anisotropy enters only through the longitudinal `C` term, so it keeps a single
`S^2`. The chirality needs `S^3/2` because the solver's `A1[0,0] - A1[1,1]` is
`(8/S) sum d sin(k.b)`, and our next-nearest list is `{b1, -b2, b3}`, so the
paper's `sum_nu sin(k.b_nu)` needs alternating signs `(+1, -1, +1)`. With this
mapping the solver reproduces Eq. (6) to `1e-9` eV. The reference is the Neel
state, so `K` must be non-positive and `j1 > 0` (antiferromagnetic).

```rust
let exchange = DrivenExchange::honeycomb(&params)?;
let magnon = exchange.honeycomb_magnon(1.5, -0.01)?;
let bands = magnon.magnon_band_onek(&array![1.0 / 3.0, 2.0 / 3.0])?;
```

### AA-stacked bilayer

`magnon::floquet::BilayerExchange::static_limit` gives the bilayer couplings of
the paper's Supplemental Material: the intralayer `j1`, `j2` and `chirality`
plus the vertical pair exchange `jperp`, the cross-layer `jcross` and the three
four-spin coefficients `k1`, `k2`, `k3` on a vertical rectangle.

```rust
use Rustb::magnon::floquet::{BilayerExchange, BilayerParams};

let bilayer = BilayerExchange::static_limit(&BilayerParams {
    hopping_ev: 0.1,
    interlayer_hopping_ev: 0.05,
    interaction_ev: 1.0,
})?;
// t_perp -> 0 reduces to the monolayer: J1 = 4t^2/U - 16t^4/U^3, k1 = k2 = k3 = 0.
```

`BilayerSpinWave::dispersion(q)` diagonalizes the paper's four-band A-type
antiferromagnet: two spin sectors, each a `4 x 4` Bogoliubov–de Gennes problem
transcribed from the Supplemental Material's explicit blocks (`epsilon_A`,
`epsilon_B`, the nearest-neighbour hopping, the interlayer pairing `C(k)` and
the chirality `+-2 chi S1 S2 s_k`). `sector_bands(q)` keeps the spin labels
apart; the sectors are exactly degenerate for equal moments, and a nonzero
`chi` splits them at each valley while swapping them between `K` and `K'`, so
`omega_up(K) = omega_down(K')` (odd parity). This analytic model requires the
exact crossed assignment `spins = [S1, S2, S2, S1]`; four independently chosen
moments are rejected, but either ordering of `S1` and `S2` is allowed. Exchange
coefficients and requested momenta must be finite.

The requested quadratic block must be positive semidefinite within the stated
numerical tolerance. The solver validates Hermiticity and normalizes by the
actual energy scale before reusing the Colpa frequency path. A bilayer-only
preflight rejects negative directions beyond `8*d*EPSILON*max(|lambda(H)|)`
(`d=4` per sector), rather than inheriting a loose absolute energy floor. It
does not accept negative-energy modes merely because their metric norm is
positive. PSD zero
modes are supported, without claiming normalizable Jordan-zero-mode vectors.
Checking one momentum does not certify stability over the entire Brillouin zone.
Net ferromagnetic intralayer coupling is the usual A-type condition; the bare
`J1 - J_D` alone does not include the ring/cross-coupling corrections.

### Floquet–Sambe driving of a magnon model

The existing `MagnonFloquet` API applies a **phenomenological exchange-bond
drive**, not the Aharonov–Bohm superexchange dressing above. It is not a
required second step: do not use both to model the same optical process. It
reuses electronic Floquet parameter types, but the outputs differ:
`MagnonFloquet::floquet_model` returns the propagator-based `MagnonSambe` solver;
`Magnon::floquet_effective_model` returns the time-averaged model (order 0 only).
`fold_quasienergy` folds into `[-W/2, W/2)`, where `W = drive.omega0_ev`. The
phenomenological drive dresses exchange bonds with `exp[-i a(t) · d]`.

```rust
use Rustb::floquet::{FloquetDrive, FloquetTruncation, LightMode};
use Rustb::magnon::{MagnonFloquet, MagnonSolve};

let drive = FloquetDrive {
    omega0_ev: 0.05,
    modes: vec![LightMode::new(1, array![Complex::new(0.0, 0.02), Complex::new(0.02, 0.0)])],
};
let sambe = magnon.floquet_model(&drive, &FloquetTruncation { n_max: 2 })?;
let quasi = sambe.quasienergies_at(&array![0.25])?;      // folded, n_band values
let bands = sambe.quasienergy_bands(&kpoints)?;
let averaged = magnon.floquet_effective_model(&drive, None)?;  // order 0
```

The quasienergies come from the **one-period propagator** of the bosonic
Bogoliubov–de Gennes dynamics `i dPsi/dt = g H(t) Psi`, keeping only
eigenvectors with positive metric norm. That avoids two traps: an ordinary
Hermitian diagonalization returns particle *and* hole continua, and at a
degenerate Goldstone mode the metric-weighted matrix `g H` is defective, so
decomposing `g S` splits the photon replicas of a zero mode spuriously. The
time dependence is applied to the *real-space* exchange tensors, where
`J_ji(-R, t) = conj(J_ij(R, t))` holds by construction, and the BdG matrix is
built once per time step; building a single Fourier harmonic would not be
Hermitian.

With no drive the folded quasienergies reproduce the undriven bands exactly; as
`Omega` exceeds the magnon bandwidth they converge to the `order = 0` effective
model. Higher van Vleck orders mix particle and hole components, so
`floquet_effective_model` rejects them with `magnon_effective_order`. The drive
must satisfy the same amplitude/length checks as the electronic path, and the
current implementation requires a collinear reference (`reference.q = 0`)
because the harmonics do not carry the single-Q rotation.

### Aharonov–Casher drive for collinear magnets

`MagnonSambe::new_aharonov_casher(&magnon, &electric, spin_axis, &truncation)`
adds a different drive: the directed transverse `S_i^- S_j^+` term acquires
`exp[-i (electric(t) × spin_axis) · d_ij]`. Longitudinal and onsite terms are
not driven. The input must conserve spin around a common unit `spin_axis`,
with the cell-periodic reference moments parallel or antiparallel to that axis.
The three components of each `electric` mode are Cartesian and multiplied by
`g μ_B/(ℏ c²)` already, in inverse ångström, even for lower-dimensional models.
This is neither an electron vector potential nor a second dressing of AB exchange.

Use `.with_time_steps(steps)?` to converge the midpoint integration separately
from the harmonic cutoff. `ac_spin_quasienergies_at(&q)` returns the folded
energies in the `(-1, +1)` spin sectors; sorting within a sector does not track
bands or unwrap the Floquet zone. Check the undriven reference for stability
before interpreting driven dispersions. Unstable and unresolved propagator
modes are reported as errors rather than treated as real bands.

## 11. Conventions and build notes

- k-points are fractional reciprocal coordinates.
- The Bloch phase is `exp(2*pi*i*k·R)`.
- Orbital positions are fractional coordinates stored by rows.
- Real-space lattice vectors are rows of `Model::lat`; fractional row
  coordinates convert as `fractional.dot(lat)`.
- `Gauge::Lattice` uses only `R` in the Fourier phase.
- `Gauge::Atom` includes orbital-position phases.
- A spinful basis is ordered as spin-up orbitals followed by spin-down orbitals.
- `None` denotes a spin-independent operator; there is no
  `SpinDirection::None` variant.

### Optional cryspglib symmetry

Enable `cryspglib` together with exactly one BLAS backend. Symmetry analysis is
defined only for 3D models with explicit atoms. Orbital-only models created by
`tb_model(lat, orb, None)` are valid TB models but deliberately return
`MissingAtomicStructure` here.

```rust
let atoms = vec![Atom::with_orbitals(
    array![0.0, 0.0, 0.0],
    AtomType::Si,
    [OrbitalId::new(0)],
)];
let mut model = Model::<false, 3>::tb_model(
    Array2::eye(3),
    array![[0.0, 0.0, 0.0]],
    Some(atoms),
)?;

// Atom moments are optional and default to None.
model.atoms[0].set_magnetic_moment([0.0, 0.0, 1.0])?;
let magnetic = model
    .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;
model.atoms[0].clear_magnetic_moment();

let structural = model.crystal_symmetry(&SymmetryParameters::default())?;
let kpoints = structural.high_symmetry_kpoints()?;
let character_table = structural.character_table_at("GM")?;
let character_columns = structural.character_table_operations()?;

let field_parameters = SymmetryParameters {
    external_fields: ExternalFields {
        electric: Some([0.0, 0.0, 1.0]),
        magnetic: None,
    },
    ..Default::default()
};
let effective = model.crystal_symmetry(&field_parameters)?;
assert!(effective.field_preserving_operations.len() <= effective.operations.len());
```

`operations` describes the atomic lattice. `field_preserving_operations`
describes the effective subset when a Hamiltonian already contains the
explicitly supplied uniform E/B fields. The fields are inputs to this analysis,
not persistent `Model` data. E is treated as a time-even polar vector; B as a
time-odd axial vector. The context is passed into cryspglib. If a field reduces
the group, structural high-symmetry tables return `FieldReducedSymmetryData`;
irreducible meshes instead use the surviving unitary/anti-unitary operations.
Magnetic order is separate. Every Atom has an optional finite Cartesian moment;
`None` is the nonmagnetic default, `set_magnetic_moment` attaches one, and
`clear_magnetic_moment` removes it. Use
`magnetic_crystal_symmetry_from_atoms` and
`magnetic_irreducible_kmesh_from_atoms` for stored moments, or the explicit
`&moments` variants for a per-call override. Character-table column headers
come from `character_table_operations()` in canonical database basis; do not
positionally pair them with `operations`, which stays in model basis. For
mappings onto `gen_kmesh` order, use
`IrreducibleKMesh::rustb_full_to_irreducible`. The mesh methods are always
Gamma-centered, matching `gen_kmesh`; use `cryspglib::stabilized_reciprocal_mesh`
directly for a shifted (Monkhorst-Pack) mesh.

To determine whether the actual TB Hamiltonian preserves those structural
candidates, call the separate exact real-space checker:

```rust
let report = model.check_hamiltonian_symmetry(
    &AtomicOrbitalBasis,
    &HamiltonianSymmetryRequest::default(),
)?;
```

The default candidate set is the structural grey extension `G + G1'`, filtered
by `SymmetryParameters::external_fields` before checking. Each operation is
reported as `Preserved`, `Broken`, or `Unresolved`; use
`report.is_fully_compatible()` for the safe `Option<bool>` summary and inspect
`report.final_group` for an identified residual UNI/BNS group or a structured
inconclusive reason.

`AtomicOrbitalBasis` is the strict automatic choice for Atom-owned, globally
aligned Wannier90 `s/p/d/f`, `sp`, `sp2`, `sp3`, `sp3d`, and `sp3d2`
projections. It verifies atom centres, orthonormality, closure, integer cell
shifts, and the orbital/spin magnetic corepresentation. It rejects repeated
radial channels, projection subsets not closed under a target operation, local
frames, and general Wannier gauges; use a custom
`BasisSymmetryRepresentation` for those. `ScalarSiteBasis` remains valid only
for exactly one atom-centred `s` orbital per Atom. The checker
validates each Laurent action, checks the complete finite `hamR` support,
verifies survivor closure, and lets cryspglib derive the effective family Hall.
Do not replace an `Unresolved`/`Inconclusive` outcome by calling operation-only
magnetic classification or by assuming the structural Hall is the reduced
group's family Hall.

Forced symmetrization is a separate opt-in constructor:

```rust
let target = model
    .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;
let symmetrized = model.symmetrize_hamiltonian(
    &target,
    &AtomicOrbitalBasis,
    &HamiltonianSymmetrizationParameters::default(),
)?;
```

It returns a new Model and leaves `model` unchanged. Before calling the basis
provider or averaging H, Rustb requires every normalized target operation to
be compatible with the current lattice, Atom positions/species, optional Atom
moments, and explicit E/B context. Failure is
`TbError::TargetMagneticGroupIncompatible`, never a best-effort projection onto
a smaller group.

The projection validates one projective magnetic corepresentation, averages
the complete real-space support (including nonsymmorphic shifts and
antiunitary conjugation), adds missing `hamR` partners, enforces Hermiticity,
and postchecks every covariance equation. For `HasRMatrix`, `rmatrix` is
projected as an affine vector operator: the explicit cell offset contributes
through the sector overlap, antiunitary operations conjugate the transformed
block, and the operation translation is subtracted from the onsite diagonal
before applying the inverse Cartesian rotation. Hermiticity is restored.
Newly generated support receives the projected position block rather than an
unconditional zero.

For `HasRMatrix`, `symmetrize_hamiltonian`, `symmetrize_model_and_centres`,
and all three pair entry points require a stored onsite (`R = 0`) position
block in each input. They set the final centres from the real onsite diagonal
of the projected `r`, keeping its cell representative without wrapping. Pair
projection assumes the two Wannier bases already use a common gauge; covariance
checks do not align phases, local frames, or cell representatives between inputs.
For groups with channel swaps, each fractional component of the difference
between corresponding centres must be smaller than half a cell in magnitude.
The centre fit searches the rounded fractional image and its 26 neighbours;
reduce highly skew lattice bases first if that stencil misses nearby images.
Each shared centre in a spinful model is checked against both spin-major
blocks. Rustb re-resolves the localized actions on the final model or pair
and checks real-space covariance of both `H` and `r` for every target
operation. If action resolution, centre consistency, or covariance fails, the
call returns an error. The input models are unchanged on success and error.
Position covariance and centre consistency use `parameters.tolerances.position`
(or `parameters.structural_parameters.symprec` if unset), in Cartesian Å.
`NoRMatrix` models skip the position-matrix checks. Operation equality during closure and projective-corepresentation checks uses
`parameters.tolerances.operation` (fractional translation; default `1e-5`, the
same numerical value as the default detection precision but an independent
setting, so raising `symprec` does not raise `operation`). It is an algebraic
 equality
tolerance only: Hamiltonian covariance still uses the `absolute`/`relative` and
`hermiticity` tolerances, position-matrix covariance uses `position`, and
localized-action matrix equality uses `representation`.

Rayon parallelizes the independent projection tasks: one task per real-space
`R` block for the Reynolds average, the position-matrix projection and the
paired-Hermiticity restoration; one task per target operation for basis-action
resolution, projective-corepresentation checks, post-projection covariance, and
the iterative centre and joint projections; one task per operation pair for
pair action resolution; and one task per `R` block for the Hamiltonian
validation clones, per-key Hermiticity checks and scale partials. Duplicate
detection, support construction and every ordered reduction stay serial. Each
block still accumulates symmetry operations in the original order, and parallel
results are collected in task order before any reduction, so the returned
matrices, the error variant and order, and the max-element acceptance scale are
independent of the Rayon worker count; the reported `relative_frobenius`
diagnostic can differ in the last few ulps. `symmetrize_centres` now requires a
`Sync` basis provider like `symmetrize_hamiltonian`, the pair methods and the
already-`Sync` `symmetrize_model_and_centres`; the built-in providers satisfy
this. Set `RAYON_NUM_THREADS` to control the outer pool and keep BLAS at one
thread (e.g. `MKL_NUM_THREADS=1` or `OPENBLAS_NUM_THREADS=1`) to avoid
oversubscription. Kernels that call BLAS through `symmetry_ordered_map` fall
back to serial with the Netlib features because reference Netlib CBLAS is not
reentrant; the BLAS-free elementwise maps (support hermitization, block copies,
scale partials) remain parallel, so use `RAYON_NUM_THREADS=1` with reference
Netlib when those BLAS-touching kernels must stay serial too. OpenBLAS must
support concurrent calls: if it was built with `USE_THREAD=0`, it needs
`USE_LOCKING=1`
([OpenBLAS FAQ](https://github.com/OpenMathLib/OpenBLAS/blob/v0.3.32/docs/faq.md#multi-threaded)).
For an unlocked build, use `RAYON_NUM_THREADS=1`; setting only
`OPENBLAS_NUM_THREADS=1` does not make the library reentrant.

For a spinless 3D Wannier model whose centres are displaced from their parent
atoms, use `CenterResolvedBasis` instead of `AtomicOrbitalBasis`. Both
`symmetrize_model_and_centres` and `symmetrize_centres` borrow `&self` and
return `Result<Self>` rather than editing in place.
The joint method first projects the centres, then resolves the actions on
those centres before projecting the Hamiltonian and any position matrix:

~~~rust
use Rustb::{CenterResolvedBasis, HamiltonianSymmetrizationParameters, SymmetryParameters};

let target = model.magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;
let basis = CenterResolvedBasis::new();
let model_sym = model.symmetrize_model_and_centres(
    &target,
    &basis,
    &HamiltonianSymmetrizationParameters::default(),
)?;
~~~

Use `model_sym` for calculations on the projected model. The projections do
not return reports; use `check_hamiltonian_symmetry` for separate Hamiltonian
diagnostics.

The provider maps parent atoms and their cell images, selects the source
orbital from the resolved localized action (largest coefficient in the target
row), falling back to an OrbProj label match and then to transformed-centre
distance, and optionally fits the local frame against the onsite Hamiltonian.
When the onsite block is degenerate or the two onsite spectra do not match, it
keeps the global cubic angular rotation; that gauge remains undetermined by H
alone.

For a centre-only projection, use `symmetrize_centres` instead:

~~~rust
let centred = model.symmetrize_centres(&target, &basis, 0.05)?;
~~~

This changes only the returned model's orbital centres, not `H` or `r`. The
supplied `tolerance` in Å must be finite and nonnegative. For `HasRMatrix`,
the proposed centres must match the unchanged onsite `r` diagonal in the
same cell representative within that tolerance; a missing onsite block or a
mismatch is an error. `NoRMatrix` skips the diagonal check.

For two spin channels related by a magnetic group, start with spinless 3D
models `up` and `dn` with matching lattices, atom ordering, orbital projections,
and position-matrix storage types. All three `symmetrize_pair*` methods
take shared references and return `Result<(Self, Self)>` in input order.
Use the returned models instead of the unchanged inputs:

~~~rust
let (up_sym, dn_sym) = Model::symmetrize_pair(
    &up,
    &dn,
    &target,
    &CenterResolvedBasis::new(),
    &HamiltonianSymmetrizationParameters::default(),
)?;
~~~

Unitary target operations project each channel independently; anti-unitary
(spin-flip) operations couple the channels, and both centres and Hamiltonians
are projected jointly. The target operations' `time_reversal` flags must
describe the actual magnetic group; a wrong flag assignment is a user error,
not silently repaired.

For a collinear spin group, prefer the cryspglib-certified entries instead of
hand-building the old magnetic-group flags. `detect_collinear_spin_group`
returns the complete `CollinearSpinGroup`: its unitary half and the `g A0`
antiunitary partners are both present. The unitary entry filters that complete
group internally:

~~~rust
use Rustb::{CollinearMoments, SpinTolerances, detect_collinear_spin_group};

let group = detect_collinear_spin_group(
    &crystal,
    CollinearMoments::Vectors(&moments),
    SpinTolerances::default(),
)?;

let (up_sym, dn_sym) = Model::symmetrize_pair_with_collinear_spin_group(
    &up,
    &dn,
    &group,
    &CenterResolvedBasis::new(),
    &HamiltonianSymmetrizationParameters::default(),
)?;

// Unitary half only: A0 antiunitary partners are filtered out internally.
let (up_unitary, dn_unitary) =
    Model::symmetrize_pair_with_unitary_collinear_spin_group(
        &up,
        &dn,
        &group,
        &CenterResolvedBasis::new(),
        &HamiltonianSymmetrizationParameters::default(),
    )?;
~~~

`Preserve`/`Swap` selects the source spin channel independently of the
unitary/antiunitary flag: a unitary Swap exchanges the channels without
conjugating the orbital Hamiltonian. The full-group average uses `2|G|`
operations; the unitary-subgroup average uses `|G|`.

Band labels use the model's atom-centred orbital metadata directly:

```rust
let report = model.calculate_irrep(None)?;
println!("{report}");

// Or test an explicitly supplied target group:
let target_report = model.calculate_irrep_for_group(
    &target,
    Some(&IrrepCalculationOptions::default()),
)?;
```

This band-irrep entry point is intentionally fixed to the automatic atomic
orbital representation. It does not accept a custom basis provider.

All target operations and their `SPIN`-appropriate factor system are resolved
before diagonalization; the spinful path explicitly enforces the canonical
double-group signs and $\mathcal T^2=-1$. An incomplete/non-closed orbital
basis or inconsistent custom representation is therefore an error. The full
real-space Hamiltonian is then certified against every target operation with
an energy-origin-invariant residual scale: if it is broken, the calculation
retains covariance
residuals and raw complex characters but all target labels become `???`.
High-symmetry k points run in parallel and are collected in canonical order;
use a one-thread BLAS configuration to avoid nested oversubscription. The
printed table contains every unitary complex character. Antiunitary operations
contribute sewing/closure diagnostics and are printed as `N/A`, not as an
ordinary trace character. Folded nonprimitive models require unfolding before
primitive-cell database labels can be assigned.

### BLAS/LAPACK backends

No backend is enabled by default; every build, test, and doc command must
select exactly one backend feature.

`ndarray/blas` is enabled for matrix products, using that same backend for
supported layouts and sizes. Small or incompatible layouts use ndarray's
fallback; no additional backend choice is required.

| Feature | Backend |
|---|---|
| `intel-mkl-static` | Statically linked Intel MKL |
| `intel-mkl-system` | System Intel MKL |
| `openblas-static` | Statically linked OpenBLAS |
| `openblas-system` | System OpenBLAS |
| `netlib-static` | Statically linked reference Netlib |
| `netlib-system` | System Netlib |

### Optional allocators

`mimalloc` and `jemalloc` are optional, default-off, and mutually exclusive:

```bash
cargo build --release --features intel-mkl-system,mimalloc
```

### Validation commands

```bash
cargo fmt --check
cargo check --all-targets --features openblas-system
cargo test --release --features intel-mkl-system
cargo clippy --all-targets --features intel-mkl-system
cargo doc --no-deps --features intel-mkl-system
```

Use release mode for numerical tests. Several integration-style tests invoke
gnuplot and write generated artifacts below `target/test-output/`.
