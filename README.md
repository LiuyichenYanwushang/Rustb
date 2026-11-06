# Rustb

Rustb is a Rust 2024 library for tight-binding calculations in condensed-matter
physics. It provides model construction, band structures, density of states,
linear and nonlinear response, quantum geometry, Wilson loops, surface Green
functions, Wannier90 import, magnetic fields, band unfolding, Fermi surfaces,
and Floquet calculations.

[![Crates.io](https://img.shields.io/crates/v/Rustb.svg)](https://crates.io/crates/Rustb)

The development API is **0.7.3 (unreleased)** and uses the const-generic model type
`Model<SPIN, DIM, R>`.

Source builds pin the optional `cryspglib` dependency to a public Git commit
because its required 0.2.1 complex-character API is not yet on crates.io.
Clone builds do not need a sibling checkout. Publishing Rustb 0.7.3 requires
publishing cryspglib 0.2.1 first, then replacing its `git`/`rev` dependency
with the registry version. The crates.io installation below currently gives
the released 0.7.1 API.

Every module page of the generated documentation contains a complete,
compiled example; this README gives the short version of each module. See
[Documentation](#documentation) for how to build both.

## Installation

Rustb has no default BLAS/LAPACK backend. Callers must enable exactly one of
the mutually exclusive backend features. The `openblas-system` default was
removed after 0.7.1, so upgrading callers must add a backend feature
explicitly. For system OpenBLAS, use:

```toml
[dependencies]
Rustb = { version = "0.7", default-features = false, features = ["openblas-system"] }
ndarray = "0.17"
num-complex = "0.4"
```

Available backends are `intel-mkl-static`, `intel-mkl-system`,
`openblas-static`, `openblas-system`, `netlib-static`, and `netlib-system`.
Enabling more than one backend (including via `--all-features`) is not
supported and is rejected at compile time. Add the optional `cryspglib`
feature to enable crystallographic and magnetic symmetry analysis without a C
dependency:

```toml
Rustb = { version = "0.7", default-features = false, features = ["intel-mkl-system", "cryspglib"] }
```

The optional `mimalloc` and `jemalloc` allocator features are mutually
exclusive and can be combined with one backend feature. `default-features = false`
matters for the released 0.7.1, whose implicit `openblas-system` default would
otherwise combine with the backend listed here.

Source builds enable `ndarray/blas`, so ndarray matrix products use the selected
backend for supported layouts and sizes, with ndarray's fallback otherwise.

On Debian/Ubuntu, `netlib-src` expects `libcblas.so`, while the distribution
provides its CBLAS symbols inside `libblas.so`. After installing `libblas-dev`,
`liblapack-dev` and `gfortran`, provide a local linker alias when using Netlib:

```sh
mkdir -p target/netlib-link
ln -sf "$(cc -print-file-name=libblas.so)" target/netlib-link/libcblas.so
LIBRARY_PATH="$PWD/target/netlib-link${LIBRARY_PATH:+:$LIBRARY_PATH}" cargo test --release --features netlib-system
```

Rustb requires Rust 1.90 or newer.

## Quick start

This example builds a spinless two-dimensional graphene model, plots its band
structure, and evaluates a Gaussian-broadened density of states:

```rust
use ndarray::{arr1, arr2, array, Array1};
use Rustb::*;

fn main() -> Result<()> {
    // Lattice vectors are stored as rows; orbital positions are rows in
    // fractional lattice coordinates.
    let lat = arr2(&[
        [3.0_f64.sqrt(), -1.0],
        [3.0_f64.sqrt(), 1.0],
    ]);
    let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);

    let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;

    // add_hop/set_hop also insert the Hermitian-conjugate hopping at -R.
    model.add_hop(-2.85, 0, 1, &array![0, 0], None);
    model.add_hop(-2.85, 0, 1, &array![-1, 0], None);
    model.add_hop(-2.85, 0, 1, &array![0, -1], None);

    let path = arr2(&[
        [0.0, 0.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [0.5, 0.5],
        [0.0, 0.0],
    ]);
    let labels = vec!["Γ", "K", "M", "Γ"];
    model.show_band(&path, &labels, 501, "graphene")?;

    let k_mesh = arr1(&[101, 101]);
    let (_energy, _dos) = model.dos(&k_mesh, -4.0, 4.0, 801, 0.02)?;

    Ok(())
}
```

`Model::k_path`, `SurfGreen::k_path`, and `unfold` preserve every path node,
including short segments, and require at least as many samples as nodes.
Unfolding paths use primitive-cell reciprocal coordinates; k-path distances
omit the `2*pi` factor. Invalid paths return an error.

All `Solve` methods return `Result`: use `model.solve_band_onek(&k)?` for
energies or `model.solve_onek(&k)?` for `(energies, row_ket_eigenvectors)`.
Single-point and batched methods report invalid models, invalid k-points and
eigensolver failures through errors. Energy-window methods select `(low, high]`,
with bounds and absolute convergence tolerance in the model's energy units.
All full-spectrum calculations share the row-ket convention `C[band, basis]`:
`H C^T = C^T diag(E)` and `O_band = C* O C^T`.

Surface Hamiltonian, spectral-density, and plotting methods also return `Result`.
Remove the old `spin` argument from `surf_green_path`, `show_arc_state`, and
`show_surf_state`, and propagate failures with `?`. Single-k surface methods
return `(right, left, bulk)`; paths retain `(left, right, bulk)`.
Hopping amplitudes use `Into<Complex64>` and k meshes use `num_traits::Float`;
the custom numeric conversion traits were removed. Serialized model fields stay
the same, with unknown and duplicate fields now rejected.

The five `Berry` Wilson-loop methods also return `Result` and accept occupied
bands as slices, for example `model.berry_loop(&loop_k, &[0])?`. Returned phases
are in radians; batch and Wannier arrays have shape `(n_occ, n_loop)` with
independently sorted columns. Divide by `2*pi` for centres in lattice units.
Invalid band selections, loops and grids, and linear algebra failures return errors.

For a spinful model, use `Model::<true, DIM>`. Spin-independent terms take
`None`; Pauli-matrix terms take `SpinDirection::X`, `Y`, or `Z`:

```rust
let mut model = Model::<true, 2>::tb_model(lat, orb, None)?;
model.set_onsite(&arr1(&[0.5, -0.5]), None);
model.add_hop(0.2, 0, 0, &array![1, 0], SpinDirection::Z);
```

`model.build_spin_matrix(SpinDirection::Z)` returns the spin operator
`S_z / ℏ = σ_z ⊗ I_norb / 2` as an `Array2<Complex<f64>>`, ordered as all
spin-up orbitals followed by all spin-down orbitals. This method is available
on spinful models with either position-matrix storage type.

## Model types

```text
Model<SPIN, DIM, R>
      │     │    └─ NoRMatrix (default) or HasRMatrix
      │     └────── real-space dimension, normally 1, 2, or 3
      └──────────── false: spinless, true: spinful
```

`HasRMatrix` stores Wannier position-matrix elements and enables the associated
commutator contribution in velocity calculations. `NoRMatrix` is a zero-sized
type.

Atoms explicitly reference the dense orbital basis through typed `OrbitalId`
values. The model remains the sole owner of orbital positions, projections, and
Hamiltonian arrays:

```rust
let carbon = Atom::with_orbitals(
    array![0.0, 0.0, 0.0],
    AtomType::C,
    [OrbitalId::new(0), OrbitalId::new(2)],
);
```

`tb_model(lat, orb, None)` creates a genuine orbital-only model and does not
invent atoms or chemical species. Such a model remains valid for tight-binding
calculations, but crystal-symmetry analysis returns `MissingAtomicStructure`.

## Module guide

Each section below describes one public module and shows the shortest useful
call sequence. The snippets share this spinless, gapped two-band model, a
generic k-point, a small uniform mesh, and an atom-based three-dimensional
model for the symmetry sections:

```rust
use ndarray::{arr1, arr2, array, Array1, Array2};
use num_complex::Complex;
use Rustb::*;

// Spinless, gapped two-band model (graphene-like) reused below.
let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
let orb = arr2(&[[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]]);
let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
model.add_hop(-2.7, 0, 1, &array![0, 0], None);
model.add_hop(-2.7, 0, 1, &array![-1, 0], None);
model.add_hop(-2.7, 0, 1, &array![0, -1], None);
model.set_onsite(&arr1(&[0.2, -0.2]), None);

let k = arr1(&[0.25, 0.10]);
let points = gen_kmesh::<f64>(&arr1(&[21usize, 21]))?;

// Atom-based 3D model reused by the symmetry sections.
let mut silicon = {
    let atoms = vec![Atom::with_orbitals(
        array![0.0, 0.0, 0.0],
        AtomType::Si,
        [OrbitalId::new(0)],
    )];
    Model::<false, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], Some(atoms))?
};
```

Every snippet assumes these imports and a surrounding function returning
`Rustb::Result<()>`, so `?` can be used directly. The `crystal_symmetry`,
`hamiltonian_symmetry` and `irrep_analysis` sections require the `cryspglib`
feature.

### `model` — core data structure

`Model<SPIN, DIM, R>` owns the lattice, orbital positions and projections,
the real-space hopping blocks `ham`/`hamR`, optional atom metadata and the
optional position matrix. The module also defines `Gauge` (the Bloch-basis
convention), `Dimension` and `SpinDirection`. Models serialize with serde
through a validated wire format: unknown or duplicate fields and invalid data
are rejected on load. `Model::validate` checks shapes, finite data, an
invertible lattice and atom/orbital references; the high-level solvers and
response entry points run it before doing any work.

```rust
let h_atom = model.gen_ham(&k, Gauge::Atom);
let h_lattice = model.gen_ham(&k, Gauge::Lattice);
assert_eq!(h_atom.dim(), (model.nsta(), model.nsta()));
assert_eq!(model.dim_r(), 2);

let wire = toml::to_string(&model).unwrap();
let restored: Model<false, 2> = toml::from_str(&wire).unwrap();
assert_eq!(restored.ham, model.ham);
```

The `toml` crate is not re-exported by Rustb; add it to your own dependencies
(or use another serde format) to reproduce the serialization snippet.

### Model construction and editing

Build a model with `tb_model`, then fill in hoppings, on-site terms, spin
decoration, supercells and atom/orbital edits. `set_hop` replaces a block and
`add_hop` accumulates it; both maintain the Hermitian-conjugate block at
`-R` automatically. Invalid indices, lengths and non-real on-site terms are
programming errors and are reported as panics, as documented on each method.

```rust
let mut chain = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
chain.set_hop(-1.0, 0, 0, &array![1], None); // also writes the -R conjugate
chain.add_onsite(&arr1(&[0.4]), None);

let doubled = chain.make_supercell(&arr2(&[[2.0]]))?;
assert_eq!(doubled.norb(), 2);
```

### `atom_struct` — atoms and orbital metadata

`Atom` stores a fractional position (lattice-vector units, the same
convention as the rows of `Model::orb`), a chemical species, typed `OrbitalId`
handles for the orbitals it owns, and an optional Cartesian magnetic moment.
`OrbProj` labels the angular character of an orbital. Atom metadata is
optional for tight-binding calculations but required by the symmetry-analysis
APIs and `Model::orb_angular`.

```rust
let site = silicon.atom(AtomId::new(0))?;
let orbital = silicon.orbital(OrbitalId::new(0))?;
assert_eq!(site.orbitals().len(), 1);
assert_eq!(orbital.projection(), &OrbProj::s);

silicon.atoms[0].set_magnetic_moment([0.0, 0.0, 1.0])?;
silicon.atoms[0].clear_magnetic_moment();
```

### `kpoints` — uniform k-meshes

`gen_kmesh` returns fractional reciprocal coordinates in `[0, 1)`, first
axis outermost, with the row order shared by every solver, Fermi-surface
writer and irreducible-mesh mapping. `gen_krange` returns the matching cell
bounds, including their upper endpoints.

```rust
let mesh = arr1(&[21usize, 21]);
let k_points = gen_kmesh::<f64>(&mesh)?;
assert_eq!(k_points.dim(), (441, 2));

let cells = gen_krange::<f64>(&mesh)?;
assert_eq!(cells.dim(), (441, 2, 2));
```

### `kpath` — piecewise-linear paths

`k_path` samples a path given in fractional reciprocal coordinates, retains
every node exactly, and returns `(k-points, cumulative distance, node
distances)`. Distances use the object's reciprocal-lattice metric without the
`2*pi` factor. Invalid coordinates, too few samples and singular lattices
return errors.

```rust
let path = arr2(&[[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]]);
let (k_path, distance, nodes) = model.k_path(&path, 101)?;
assert_eq!(k_path.nrows(), 101);
assert_eq!(nodes.len(), 3);
assert_eq!(distance.len(), 101);
```

### `kplane` — planar k-grids

`gen_kplane` samples a plane through reciprocal space defined by an origin and
two spanning vectors; it is the grid behind `show_fermi_surface_plane` and
manual spectral maps.

```rust
let origin = arr1(&[0.0, 0.0, 0.0]);
let v1 = arr1(&[0.5, 0.0, 0.0]);
let v2 = arr1(&[0.0, 0.5, 0.0]);
let plane = gen_kplane(&origin, &v1, &v2, 11, 11)?;
assert_eq!(plane.dim(), (121, 3));
```

### `solve_ham` — diagonalization

The `Solve` trait provides single-k and batched band/eigenvector solvers.
Batches share one Fourier preparation and are split into jobs sized from a
memory budget; a per-worker share is fixed for the whole call. Energies ascend
independently at each k-point and no band tracking is performed. Eigenvectors
are row kets: `evec.row(n)` belongs to `energies[n]` and satisfies
`H C^T = C^T diag(E)`. Every method returns `Result`.

```rust
let bands = model.solve_band_onek(&k)?;
let (energies, kets) = model.solve_onek(&k)?;
assert_eq!(kets.dim(), (model.nsta(), model.nsta()));

let window = model.solve_band_range_onek(&k, (-1.0, 1.0), 0.0)?;
let mesh_bands = model.solve_band_all_parallel(&points)?;
assert_eq!(mesh_bands.dim(), (points.nrows(), model.nsta()));
```

### `model_physics` — Bloch Hamiltonian and DOS

`gen_ham` builds one `H(k)` in either gauge; `gen_ham_batch` does the same for
a batch with one shared Fourier sum, without spawning Rayon tasks. `dos`
diagonalizes a uniform mesh and convolves the eigenvalues with a Gaussian.

```rust
let h_k = model.gen_ham(&k, Gauge::Atom);
let h_batch = model.gen_ham_batch(&points, Gauge::Atom);
assert_eq!(h_batch.dim(), (points.nrows(), model.nsta(), model.nsta()));

let (energy, dos) = model.dos(&arr1(&[21usize, 21]), -3.0, 3.0, 101, 0.05)?;
assert_eq!(energy.len(), dos.len());
```

### `hubbard` — unrestricted Hartree-Fock

`HubbardModel` adds orbital-resolved on-site interactions to a spinful model.
`solve_hartree_fock` performs non-collinear unrestricted Hartree-Fock with the
complete local `2 x 2` spin-density matrix and returns an ordinary
`Model<true, DIM, R>` with the converged chemical potential shifted to zero.
`MeanFieldConstraint` selects a fixed chemical potential or a filling computed
once from the bare model.

```rust
let mut bare = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
bare.add_hop(-1.0, 0, 0, &array![1], None);

let hubbard = HubbardModel::with_uniform_u(bare, 2.0)?;
let mut params = MeanFieldParams::new(
    [101],
    MeanFieldConstraint::FixedInitialFilling { reference_mu: 0.0 },
    Occupation::FermiSmearing { width: 0.01 },
);
params.initial_magnetization = InitialMagnetization::UniformVector {
    moment_per_orbital: [1e-3, 0.0, 0.0],
};

let hf_model = hubbard.solve_hartree_fock(&params)?;
let total_spin = hf_model.spin_moment(&[101], 0.0, params.occupation)?;
```

### `velocity` — velocity operators

`gen_v` returns the Cartesian velocity operators at one k-point together with
`H(k)`; `gen_v_projected` fuses the direction weights into the hopping and
position-matrix sums. The `_batch` variants share one GEMM per batch. A
position matrix (`HasRMatrix`) adds its commutator contribution. Observables
are gauge invariant even though the blocks are not.

```rust
let (velocity, h_k) = model.gen_v(&k, Gauge::Atom);
assert_eq!(velocity.dim(), (2, model.nsta(), model.nsta()));

let directions = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
let (projected, h_k) = model.gen_v_projected(&k, Gauge::Atom, &directions);
assert_eq!(projected.dim(), (2, model.nsta(), model.nsta()));
```

### `orbital_angular` — Bloch orbital angular momentum

`orbital_angular_momentum_onek` computes the band-space operator `L / hbar` of
Busch, Mertig and Göbel (2023) from the velocity operators. The result has
shape `(3, nsta, nsta)` in ascending-energy band order and is distinct from
the atomic matrices returned by `Model::orb_angular`.

```rust
use Rustb::orbital_angular::OrbitalAngular;

let angular = model.orbital_angular_momentum_onek(&k)?;
assert_eq!(angular.dim(), (3, model.nsta(), model.nsta()));
let lz_band_0 = angular[[2, 0, 0]];
```

### `thermodynamics` — occupations

`Occupation` is shared by the mean-field solver and the spin observables:
`ZeroTemperature` is an exact step function, `FermiDirac` takes a temperature
in kelvin, and `FermiSmearing` takes an energy width in eV. The response layer
instead states temperature directly through `Sampling`.

```rust
let occupation = Occupation::FermiDirac { temperature_kelvin: 300.0 };
let f = occupation.value(-0.1, 0.0)?;
let window = occupation.minus_derivative(-0.1, 0.0)?;
assert!((0.0..=1.0).contains(&f));
assert!(window > 0.0);
```

### `response` — Brillouin-zone response API

The Brillouin-zone response API is built on one shared `Parameters<DIM>` grid
configuration. Its entry points are `hall_conductivity`,
`optical_conductivity`, `optical_conductivity_tensor`,
`extrinsic_nonlinear_hall`, `intrinsic_nonlinear_hall` and
`quantum_geometry` (documented in its own section, with the same parameters).
Direct, simplex and energy-cut algorithms are selected by `Integration`, never
by a different method name. Methods receive only the choices they read:
projection directions, `eta_ev`, and (where a spin current exists) `spin` and
`field_symmetry`. The charge-only entry points (optical conductivity and
tensor, quantum geometry, intrinsic nonlinear Hall) have no `spin` argument, so
a spin-current request is a compile error. `hall_conductivity`,
`extrinsic_nonlinear_hall` and the Berry-curvature helpers do accept `spin` and
return `TbError::SpinNotAllowed` on a spinless model.

```rust
let params: Parameters<2> = Parameters {
    conditions: Conditions::fixed(300.0, 0.0, 0.0),
    kmesh: [21, 21],
    integration: Integration::Direct,
};

let hall = model.hall_conductivity(&params, [[1.0, 0.0], [0.0, 1.0]], 1e-3, None)?;
let geometry = model.quantum_geometry(&params, [[1.0, 0.0], [0.0, 1.0]], 1e-3)?;
assert_eq!(hall.conductivity.len(), 1);
assert_eq!(geometry.metric.len(), 1);
```

### `response::config` — shared grid parameters

`Parameters<DIM>` holds `conditions` (temperature in kelvin, chemical potential
in eV, photon frequency in eV), the uniform `kmesh` and the `integration`
algorithm. Each axis is either `Sampling::Fixed(value)` or
`Sampling::Values(series)`; at most one axis may be sampled. Nothing is
defaulted, and `Integration`/`FieldSymmetry` implement no `Default`.

```rust
let sweep: Parameters<2> = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(300.0),
        mu_ev: Sampling::Values(Array1::linspace(-1.0, 1.0, 21)),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [21, 21],
    integration: Integration::EnergyCut,
};
assert!(!sweep.conditions.mu_ev.is_fixed());
assert!(sweep.conditions.omega_ev.is_fixed());
```

### `response::linear` — Hall conductivity

`hall_conductivity` evaluates the anomalous (and, with `spin`, spin) Hall
response. `Integration::Direct` is a uniform k-sum; `EnergyCut` is a
band-tracked occupancy cut (empty/partial/full simplex classification plus
occupancy quadrature and thermal convolution) and requires ascending sampled
chemical potentials. A spinless model returns `TbError::SpinNotAllowed` for a nonempty
spin request. The result carries `axis: ResponseAxis` and one value per sample.

```rust
let mu = Array1::linspace(-2.0, 2.0, 21);
let params: Parameters<2> = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(30.0),
        mu_ev: Sampling::Values(mu),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [11, 11],
    integration: Integration::Direct,
};

let result = model.hall_conductivity(&params, [[1.0, 0.0], [0.0, 1.0]], 1e-3, None)?;
assert_eq!(result.conductivity.len(), 21);
```

### `response::nonlinear` — nonlinear Hall response

Both nonlinear entry points read the three direction rows as
`(current, field_1, field_2)`. The intrinsic response is charge-only and takes
no broadening; the extrinsic response takes `eta_ev`, an optional spin current
and an explicit `FieldSymmetry` (`Ordered` or `Symmetrized`). DC conditions
are required, and direct integration needs a finite Fermi-window peak at every
sample.

```rust
let directions = [
    [1.0, 0.0], // current
    [1.0, 0.0], // field 1
    [0.0, 1.0], // field 2
];
let params: Parameters<2> = Parameters {
    conditions: Conditions::fixed(30.0, 0.0, 0.0),
    kmesh: [11, 11],
    integration: Integration::Direct,
};

let intrinsic = model.intrinsic_nonlinear_hall(&params, directions)?;
let extrinsic = model.extrinsic_nonlinear_hall(
    &params,
    directions,
    1e-3,
    None,
    FieldSymmetry::Symmetrized,
)?;
assert_eq!(intrinsic.conductivity.len(), extrinsic.conductivity.len());
```

### `response::optical` — optical conductivity

`optical_conductivity` returns one projected interband Kubo component;
`optical_conductivity_tensor` returns the full ordered Cartesian tensor from
one shared diagonalization and band-tracking pass. The response excludes Drude
terms and exactly degenerate pairs; frequencies and broadening are in eV, and
the `e^2/hbar` prefactor is omitted. `omega_ev` may be sampled here, unlike
the DC entry points.

```rust
let params: Parameters<2> = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(300.0),
        mu_ev: Sampling::Fixed(0.0),
        omega_ev: Sampling::Values(Array1::linspace(0.0, 3.0, 31)),
    },
    kmesh: [11, 11],
    integration: Integration::Simplex,
};

let component = model.optical_conductivity(&params, [[1.0, 0.0], [0.0, 1.0]], 1e-2)?;
let tensor = model.optical_conductivity_tensor(&params, 1e-2)?;
assert_eq!(component.conductivity.ncols(), 31);
assert_eq!(tensor.conductivity.nrows(), 4); // 2 x 2 ordered Cartesian components
```

### `response::traits` — per-k Berry curvature

`BerryCurvature` provides band-resolved and occupation-weighted Berry
curvature at individual k-points. The band-resolved form takes no
thermodynamic state at all; the occupation-weighted form takes one fully fixed
DC `Conditions` and never a k-mesh. A spin current is requested by
`Some(SpinDirection::Z)` on a spinful model.

```rust
let directions = [[1.0, 0.0], [0.0, 1.0]];
let bands = model.berry_curvature_at(&k, directions, 1e-3, None)?;
assert_eq!(bands.berry_curvature.len(), model.nsta());

let dc = Conditions::fixed(300.0, 0.0, 0.0);
let occupied = model.occupied_berry_curvature_at(&k, &dc, directions, 1e-3, None)?;
assert!(occupied.is_finite());
```

### `quantum_geometry` — quantum metric and Berry curvature

The `QuantumGeometry` trait returns band-resolved `(metric, Berry curvature,
energies)` at one k-point or on a list, with no thermodynamic state. The
integrated `Model::quantum_geometry` entry point weighs the bands by the
occupation selected by `Parameters`; it lives in this module and shares
`Parameters` with the response entry points. Simplex mode reports small-gap
diagnostics.

```rust
let directions = [[1.0, 0.0], [0.0, 1.0]];
let at_k = model.quantum_geometry_at(&k, directions, 1e-3)?;
assert_eq!(at_k.metric.len(), model.nsta());

let on_mesh = model.quantum_geometry_on(&points, directions, 1e-3)?;
assert_eq!(on_mesh.metric.dim(), (points.nrows(), model.nsta()));
```

### `geometry` — Wilson loops and topology

The `Berry` trait computes Berry phases, Berry flux, hybrid Wannier centres and
Wilson-loop spectra. The selected band subspace must be isolated and the loop
must close modulo an integer reciprocal vector. All five methods return
`Result`; `berry_loop` uses SVD-unitarized links, `berry_loop_det` uses the raw
determinant product, and both report `-arg(W)` in radians.

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
let total = model.berry_loop_det(&loop_k, &occupied)?;

let centres = model.wannier_centre(
    &occupied,
    &arr1(&[0.0, 0.0]),
    &arr1(&[1.0, 0.0]),
    &arr1(&[0.0, 1.0]),
    11,
    11,
)?;
assert_eq!(phases.len(), 1);
assert!(total.is_finite());
assert_eq!(centres.dim(), (1, 11));
```

### `cut` — slabs and dots

`CutModel::cut_piece` truncates one lattice direction into `num` layers,
turning inter-cell hopping into intra-slab hopping. `cut_dot` keeps a shaped
finite region (triangle, square, hexagon or octagon).

```rust
let ribbon = model.cut_piece(8, 1)?;
let dot = model.cut_dot(4, 6, None)?;
assert_eq!(ribbon.norb(), 8 * model.norb());
assert!(dot.norb() > 0);
```

### `surfgreen` — semi-infinite surface Green function

`SurfGreen::from_Model` prepares the principal layers and hopping matrices for
one open direction; `surf_green_one` and `surf_green_onek` evaluate the surface
spectral densities, and the plotting methods write `arc.dat` and PDFs. The
k-vector passed to the surface object has length `DIM - 1`. Single-k methods
return `(right, left, bulk)`; path methods return `(left, right, bulk)`.

```rust
let surface = SurfGreen::from_Model(&model, 0, 1e-3, None)?;
let k_parallel = arr1(&[0.25]);
let (right, left, bulk) = surface.surf_green_one(&k_parallel, 0.0)?;
assert!(right.is_finite());

let energy = Array1::linspace(-2.0, 2.0, 41);
let (right, left, bulk) = surface.surf_green_onek(&k_parallel, &energy)?;
assert_eq!(right.len(), 41);
```

### `fermi_surface` — Fermi-surface output

Three traits cover the Fermi surface: `FermiSurface` (marching squares in 2D,
marching tetrahedra in 3D), `FermiSurfacePlane` (2D slices of a 3D model) and
`BxsfExport` for 3D models. `write_spin_frmsf` exports spin-split surfaces. The Fermi
energy is supplied explicitly: surface plots evaluate the contour/isosurface
at that energy, BXSF records it in the header, and FRMSF band energies are
shifted so that `E_F = 0`.

```rust
model.show_fermi_surface(&arr1(&[41usize, 41]), 0.0, "fermi_surface")?;

let cubic = Model::<false, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None)?;
cubic.write_bxsf(&[11, 11, 11], 0.0, "fermi_surface")?;
```

### `floquet` — periodically driven systems

`FloquetDrive` holds the base photon energy and one or more `LightMode` values
(integer harmonics and complex amplitudes of `eA/hbar`). `floquet_model`
builds the enlarged real-space Sambe Hamiltonian; `floquet_effective_model`
returns a same-size van Vleck model from the real-space Bessel backend.
`floquet_effective_mode_resolved_model` treats modes as mutually incoherent,
and `floquet_effective_q_model` adds the coherent long-wavelength
`O(A^2 q/W)` correction. All results are ordinary `Model` values; fold
eigenvalues with `fold_quasienergy`.

```rust
let drive = FloquetDrive::with_modes(
    0.8,
    vec![LightMode::new(
        1,
        arr1(&[Complex::new(0.12, 0.0), Complex::new(0.0, 0.12)]),
    )],
);

let sambe = model.floquet_model(&drive, &FloquetTruncation::new(1))?;
let mut quasienergies = sambe.solve_band_onek(&k)?;
quasienergies.mapv_inplace(|e| fold_quasienergy(e, drive.omega0_ev));
quasienergies.as_slice_mut().unwrap().sort_by(f64::total_cmp);

let effective = model.floquet_effective_model(&drive, None)?;
let second_order = model.floquet_effective_model(
    &drive,
    Some(&FloquetEffectiveOptions::new().with_order(2)),
)?;
assert_eq!(effective.nsta(), model.nsta());
assert_eq!(second_order.nsta(), model.nsta());
```

### `magnetic_field` — uniform magnetic field

`add_magnetic_field` applies the Peierls substitution on a magnetic supercell:
`mag_dir` selects the field direction (`2` for a 2D model), `expand` gives
the two perpendicular enlargement factors and `phi_total` is the integer number
of flux quanta through the full magnetic supercell.

```rust
let magnetic = model.add_magnetic_field(2, [4, 4], 1)?;
assert_eq!(magnetic.norb(), 16 * model.norb());
let bands = magnetic.solve_band_onek(&arr1(&[0.0, 0.0]))?;
assert_eq!(bands.len(), magnetic.nsta());
```

### `unfold` — band unfolding

The `Unfold` trait projects a supercell spectral function onto the primitive
cell following PRL 104, 216401 (2010). The transformation matrix `U` maps
primitive to supercell lattice vectors; the path uses primitive fractional
reciprocal coordinates and the primitive reciprocal metric.

```rust
let transform = arr2(&[[2.0, 0.0], [0.0, 2.0]]);
let supercell = model.make_supercell(&transform)?;
let path = arr2(&[[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]);
// Spectral weight, shape (n_energy, nk).
let weight = supercell.unfold(&transform, &path, 101, -3.0, 3.0, 101, 1e-2, 1e-5)?;
assert_eq!(weight.dim(), (101, 101));
```

### `wannier90` — Wannier90 import

`Wannier90::from_hr` reads `_hr.dat` and, when present, `_centres.xyz` and
`_wsvec.dat`. `_r.dat` is read only for a `HasRMatrix` model, and is then
required; `NoRMatrix` models never open it. Declared record counts, indices,
weights and sizes are validated before the model is assembled.

```rust
let wannier: Model<false, 3> = Model::from_hr("path/to/seedname/", "wannier90", 0.0)?;
let with_positions: Model<false, 3, HasRMatrix> =
    Model::from_hr("path/to/seedname/", "wannier90", 0.0)?;
assert_eq!(wannier.nsta(), with_positions.nsta());
```

### `crystal_symmetry` — structural and magnetic symmetry

With the `cryspglib` feature, a three-dimensional model with explicit atoms can
query structure and magnetic space groups, high-symmetry k-points, character
tables and irreducible reciprocal meshes. Uniform electric and magnetic fields
already encoded in the Hamiltonian are supplied per call through
`SymmetryParameters::external_fields`; they are analysis inputs, not model
state. Character-table columns use the canonical database basis, while
`operations` stays in the model basis.

```rust
let symmetry = silicon.crystal_symmetry(&SymmetryParameters::default())?;
let points = symmetry.high_symmetry_kpoints()?;
let gamma_table = symmetry.character_table_at("GM")?;
let operations = symmetry.character_table_operations()?;

let mesh = silicon.irreducible_kmesh([8, 8, 8], true, &SymmetryParameters::default())?;
assert!((mesh.weights.sum() - 1.0).abs() < 1e-12);
assert!(!points.is_empty() && !operations.is_empty() && !gamma_table.is_empty());
```

### `hamiltonian_symmetry` — Hamiltonian certification

Structure symmetry is only a candidate symmetry of the tight-binding model.
`check_hamiltonian_symmetry` certifies the actual hoppings and on-site matrices
on the complete finite real-space support, including nonsymmorphic cell shifts,
and identifies the residual magnetic group (UNI/BNS). A missing basis
representation is `Unresolved`, never `Broken`. `symmetrize_hamiltonian`
optionally projects a slightly broken model onto a validated target group and
returns a new model.

```rust
let report = silicon.check_hamiltonian_symmetry(
    &AtomicOrbitalBasis,
    &HamiltonianSymmetryRequest::default(),
)?;
let summary: Option<bool> = report.is_fully_compatible();

let target = silicon.magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;
let symmetrized = silicon.symmetrize_hamiltonian(
    &target,
    &AtomicOrbitalBasis,
    &HamiltonianSymmetrizationParameters::default(),
)?;
assert_eq!(symmetrized.nsta(), silicon.nsta());
assert!(summary.is_some());
```

### `irrep_analysis` — magnetic band irreps and coreps

`calculate_irrep` produces an irvsp-like table of band characters and magnetic
corepresentation labels at every isolated high-symmetry k-point. The report
retains the raw complex unitary characters. Missing orbital metadata is a hard
error before any solver runs; a Hamiltonian that breaks the target group keeps
its residuals and characters but marks every target label `???`. Use one BLAS
thread when relying on the inner Rayon parallelism.

```rust
let report = silicon.calculate_irrep(None)?;
println!("{report}");

let target = silicon.magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;
let for_target = silicon.calculate_irrep_for_group(
    &target,
    Some(&IrrepCalculationOptions::default()),
)?;
println!("{for_target}");
```

### `output` — plots and Wannier90 files

The `OutPut` trait writes band-structure PDFs (`show_band`), Wannier90
`_hr.dat`/`_win`/`_centres.xyz` files and POSCAR structures. `draw_heatmap`
renders a 2D array. Plotting requires the `gnuplot` executable with the
`pdfcairo` terminal.

```rust
let path = arr2(&[[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.0, 0.0]]);
let labels = vec!["Γ", "K", "Γ"];
model.show_band(&path, &labels, 101, "band_output")?;
silicon.output_hr("target/example-output/", "seedname")?;
silicon.output_win("target/example-output/", "seedname")?;
```

### `io` — plain-text output

`write_txt` and `write_txt_1` write one 2D or 1D array as text and return
`std::io::Result`, which converts into `TbError`.

```rust
write_txt(&arr2(&[[1.0, 2.0], [3.0, 4.0]]), "target/example-output/data.dat")?;
write_txt_1(&arr1(&[1.0, 2.0]), "target/example-output/data1.dat")?;
```

### `math` — small numerical helpers

`comm` and `anti_comm` compute matrix commutators and anti-commutators;
`gauss` is the normalized Gaussian kernel used for DOS-style broadening (the
`dos` implementation inlines the same kernel).

```rust
let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
let b = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
let commutator = comm(&a, &b);
let anticommutator = anti_comm(&a, &b);
let kernel = gauss(0.1, 0.2);
assert_eq!(commutator.dim(), (2, 2));
assert_eq!(anticommutator.dim(), (2, 2));
assert!(kernel > 0.0);
```

### `phy_const` — physical constants

`phy_const` collects the elementary charge, `hbar`, the `e^2/hbar` conductivity prefactor,
the electron mass, the Bohr magneton and the superconducting flux quantum
`h/(2e)` in SI units.

```rust
use Rustb::phy_const::{Quantum_conductivity, hbar, mu_B, phy_0};
assert!(Quantum_conductivity > 0.0);
assert!(hbar > 0.0 && mu_B > 0.0 && phy_0 > 0.0);
```

### `generics` — index and dimension conversions

`SpinDirection::try_from_index` maps `0..=3` to `None`/`X`/`Y`/`Z` with a
structured error; `from_index` panics outside that range. `Dimension` converts
to and from `usize` with validation.

```rust
let spin = SpinDirection::try_from_index(3)?;
assert_eq!(spin, Some(SpinDirection::Z));

let dimension: usize = Dimension::two.into();
assert_eq!(dimension, 2);
assert_eq!(usize::from(Dimension::three), 3);
```

### `error` — structured error type

Most fallible public APIs return `Rustb::Result<T>`, an alias for
`std::result::Result<T, TbError>`; the plain-text writers in `io` return
`std::io::Result` and convert with `?`. `TbError` is a `thiserror` enum; match
on the variants you can recover from and propagate the rest.

```rust
let bad = TbError::InvalidDimension { dim: 4, supported: vec![1, 2, 3] };
assert!(matches!(bad, TbError::InvalidDimension { dim: 4, .. }));

let width = Occupation::ZeroTemperature.energy_width()?;
assert_eq!(width, 0.0);
```

### `model_utils` — lattice-vector lookup

`find_R` locates a lattice vector in `hamR`; `remove_row` and `remove_col`
return arrays with one index dropped. Surface-Green construction uses the
row/column removers, while `find_R` also backs real-space lookups in `output`,
`wannier90`, `floquet` and `magnetic_field`.

```rust
let index = find_R(&model.hamR, &array![0, 0]);
assert!(index.is_some());

let fewer_rows = remove_row(model.hamR.clone(), 0);
assert_eq!(fewer_rows.nrows() + 1, model.hamR.nrows());
```

### `ndarray_lapack` — LAPACK eigensolvers

The safe wrappers around LAPACK's Hermitian eigensolvers return `Result`,
accept any ndarray layout and return row kets. `eigh_x`/`eigh_r` compute
selected eigenpairs in a finite energy window; `eigvalsh_x`/`eigvalsh_r`/
`eigvalsh_v` compute eigenvalues only. `zaxpy` is the safe BLAS
`y += alpha * x` used internally for direction-weight accumulation.

```rust
use ndarray_linalg::UPLO;

let h = model.gen_ham(&k, Gauge::Atom);
let (evals, evecs) = ndarray_lapack::eigh_x(&h, (-20.0, 20.0), 0.0, UPLO::Lower)?;
assert_eq!(evecs.ncols(), model.nsta());
assert_eq!(evals.len(), evecs.nrows());
```

`ndarray-linalg` is a direct dependency of Rustb but is not re-exported; add
`ndarray-linalg = "0.18"` to your own dependencies to name `UPLO`.

## Conventions

- k-points are fractional reciprocal coordinates; the Bloch phase is
  `exp(2*pi*i*k·R)`.
- Lattice vectors are rows of `Model::lat`; orbital positions are rows of
  `Model::orb` in fractional coordinates (`cart = frac.dot(lat)`).
- `Gauge::Lattice` uses only `R` in the phase; `Gauge::Atom` also includes
  orbital-position phases. Physical observables are gauge invariant.
- A spinful basis is ordered as all spin-up orbitals followed by all spin-down
  orbitals. `None` denotes a spin-independent operator; there is no
  `SpinDirection::None` variant.
- Full-spectrum solvers return row kets `C[band, basis]` with
  `H C^T = C^T diag(E)`; use `O_band = C* O C^T`.
- Energy-window solvers select `(low, high]`; nonpositive tolerances select the
  LAPACK default.

## Documentation

- Generated rustdoc: every module page carries its full example and the
  mathematical conventions. Build it with
  `cargo doc --no-deps --features openblas-system` (add `cryspglib` for the
  symmetry modules, and `RUSTDOCFLAGS="--html-in-header docs-header.html"` for
  the KaTeX header).
- [`SKILLS.md`](SKILLS.md) is the longer practical cookbook, including batching
  details, numeric bounds and migration notes.
- [`CHANGELOG.md`](CHANGELOG.md) records breaking changes and numerical fixes.
- Executable examples live in `examples/<name>/main.rs`; run one with
  `cargo run --features intel-mkl-system --example <name>`.

## Development

```bash
cargo fmt --check
cargo check --all-targets --features openblas-system
cargo test --release --features openblas-system
cargo clippy --all-targets --features openblas-system
cargo doc --no-deps --features openblas-system
```

Numerical tests should be run in release mode. Some integration-style tests
invoke gnuplot and write generated files below `target/test-output/`.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License
