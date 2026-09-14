# Rustb 0.7 — Practical API Guide

This guide follows the const-generic `Model<SPIN, DIM, R>` API in the current
source tree. For mathematical definitions and complete error semantics, use the
generated rustdoc.

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

`from_hr` reads three-dimensional Wannier90 data. The model type controls
whether `_r.dat` is required:

```rust
let model: Model<false, 3> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;

let model_with_r: Model<false, 3, HasRMatrix> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;
```

The `HasRMatrix` form includes position-matrix contributions in velocity
operators.

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
in bounded batches. The default policy balances the known point/thread counts
under a 128 MiB total budget for Bloch phase matrices and Hamiltonian batches;
it has no fixed 16-point cap. Output arrays, orbital phases and LAPACK workspace
are additional to this budget. A single point is still processed when its
buffers exceed the budget.

Batching is internal to the existing solver methods and requires no options.
When points are fewer than workers, jobs can contain a single point; the policy
does not guarantee an optimal batch size on every machine. Rayon chooses which
workers run the jobs; they are not pinned to CPU cores. Configure BLAS threading
separately (for example `MKL_NUM_THREADS=1` with outer Rayon parallelism).

### Eigenvalue ordering and eigenvector axes

All eight `Solve` methods return `Result`. Use `?` to propagate invalid-model,
k-point, generated-Hamiltonian, and eigensolver errors. A batch validates the
model once, even when it contains zero k-points. Hermiticity remains the caller's
responsibility. Legacy `Berry` Wilson-loop methods still have infallible
signatures and panic if their internal solve fails.

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

The raw `ndarray-linalg 0.18.1` `.eigh()` result for a C-layout complex H uses
a different convention: its internal axis swap makes LAPACK solve `H^T = H*`.
Rustb converts that raw matrix `U` with `ndarray_linalg::conjugate(&U)` into
`C = U†`. Here `conjugate()` means **conjugate transpose**; `.t()` only
transposes, while `mapv(|z| z.conj())` only conjugates. The raw-U formula
`U^T O U*` must not be used with the returned C. Recheck this compensation if
the dependency or input memory layout changes; matching energies alone will
not detect a conjugated-eigenvector error.

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
in fallible code. Sambe entry points check allocation arithmetic and reject
a time grid below the per-link spectral estimate instead of silently aliasing
high harmonics. For a k scan, construct `floquet_model` once and use its
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
swept state cannot be passed to them at all.

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
reciprocal lattice vector.

```rust
let occupied = vec![0usize];
let loop_k = arr2(&[
    [0.0, 0.0],
    [0.25, 0.0],
    [0.5, 0.0],
    [0.75, 0.0],
    [1.0, 0.0],
]);

let phases = model.berry_loop(&loop_k, &occupied);
let total_phase = model.berry_loop_det(&loop_k, &occupied);

let centres = model.wannier_centre(
    &occupied,
    &arr1(&[0.0, 0.0]),
    &arr1(&[1.0, 0.0]),
    &arr1(&[0.0, 1.0]),
    101,
    101,
);
```

`berry_flux` takes the same origin and two directions plus `nk1` and `nk2`.

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
let surface = surf_Green::from_Model(
    &model,
    0,       // open lattice direction
    1e-3,    // imaginary broadening
    None,    // optional maximum principal-layer range
)?;

let k_parallel = arr1(&[0.25]);
let (right_ldos, left_ldos, bulk_ldos) =
    surface.surf_green_one(&k_parallel, 0.0);

let energy = Array1::linspace(-2.0, 2.0, 401);
let (right_curve, left_curve, bulk_curve) =
    surface.surf_green_onek(&k_parallel, &energy);
```

The k-vector passed to the surface object has length `DIM - 1`.

## 7. Floquet driven systems

`LightMode::a_complex` is the rescaled vector potential `eA/hbar`, in inverse
lattice-length units.

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
let truncation = FloquetTruncation::new(1, 128);
let k = arr1(&[0.2, 0.1]);

let sambe_model = model.floquet_model(&drive, &truncation)?;
let h_floquet =
    model.floquet_ham_onek(&k, &drive, &truncation, Gauge::Lattice)?;
let quasienergy =
    model.floquet_quasienergy_onek(&k, &drive, &truncation, Gauge::Lattice)?;

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

// Retain the complete van Vleck correction through O(omega^-2).
let second_order_options = FloquetEffectiveOptions::new().with_order(2);
let effective_second_order = model.floquet_effective_model(
    &drive,
    Some(&second_order_options),
)?;

// One coherent plane wave with a Cartesian wavevector (inverse-length units).
// This returns the same number of states as `model` and adds the analytic
// weak-field O(A^2 q / W) correction to the ordinary q=0 result.
let q_cartesian = arr1(&[2.0e-3, 0.0]);
let effective_linear_q = model.floquet_effective_q_model(
    &drive,
    Some(&second_order_options),
    &q_cartesian,
)?;
```

| API | Basis size | Intended regime |
|---|---:|---|
| `floquet_model` / `floquet_ham_onek` | `nsta * (2*n_max + 1)` | Full truncated Sambe problem |
| `floquet_effective_model` | `nsta` | Off-resonant, high-frequency expansion |
| `floquet_effective_mode_resolved_model` | `nsta` | Mode-diagonal mutually incoherent correction sum |
| `floquet_effective_q_model` | `nsta` | One coherent common `q`, through `O(A^2 q/W)` |

`floquet_effective_model` uses the real-space generalized-Bessel backend:
no `k_mesh` and no `target_hamR` — the effective hopping support is
determined automatically: up to the double Minkowski sum for the default
`order = 1`, and up to the triple Minkowski sum for `order = 2`.
All three effective-model APIs use `FloquetEffectiveOptions`, whose defaults
are `order = 1` and `harmonic_max = 2`. The harmonic cutoff is a concrete
nonnegative integer, independent of the Sambe photon cutoff. These APIs take
no `FloquetTruncation`: out-of-range links fall back to a per-link
time-grid DFT sized from the link's own bandwidth and the requested
harmonic range.

`FloquetTruncation` controls photon sectors and time sampling only for the
full Sambe APIs. To preserve an old effective-model call's implicit harmonic
range during migration, pass `.with_harmonic_max(2 * old_n_max)` explicitly.

For multiple mutually incoherent modes,
`floquet_effective_mode_resolved_model` computes
`H0 + sum_alpha (H_eff[a_alpha] - H0)`.  It never combines the modes inside one
Peierls exponential, skips exact-zero modes, and unions all generated hopping
supports.  This is the leading weak-field random-phase result, not an exact
all-orders phase average: higher-order cross-intensity terms are omitted.

`floquet_effective_q_model` does not use numerical momentum differences.
For a Cartesian bond `d = (R + tau_j - tau_i) * lat`, it represents
`D_a H_0` by the real-space block `i*d_a*t(R)`.  It constructs
`G_n = a_n·D H_0`, forms `{G_n,G_n†}` by real-space convolution, and applies
`q·D` by multiplying every resulting output hopping by `i*q·d_out`.  The
same-harmonic complex amplitudes are summed coherently.  Both
`|a_n·d| << 1` and `|q·d| << 1` are required; distinct coherent wavevectors
need a graded or commensurate-supercell treatment rather than this method.
The correction includes only positive harmonics through the effective
`harmonic_max`; active nonpositive harmonics are rejected for nonzero `q` when
`order >= 1`.  `order = 0` adds no linear-`q` term, and `q = 0` follows the
ordinary `floquet_effective_model` path exactly.

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

## 10. Conventions and build notes

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
and postchecks every covariance equation. For `HasRMatrix`, old position-matrix
blocks stay aligned by lattice vector and newly generated support receives
zeros; `rmatrix` is not incorrectly treated as a scalar Hamiltonian.

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
