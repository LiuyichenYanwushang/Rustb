# Rustb 0.8 — Practical API Guide

This guide follows the const-generic `Model<SPIN, DIM, R>` API in the current
source tree. For mathematical definitions and complete error semantics, use the
generated rustdoc.

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

`lat` is a square `DIM × DIM` matrix whose columns are real-space lattice
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

### Orbital projections

Orbital projections are needed by operations such as `orb_angular`:

```rust
model.set_projection(&vec![OrbProj::s]);
let angular_momentum = model.orb_angular()?;
```

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
let bands = model.solve_band_all_parallel(&k_points);
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
let bands = model.solve_band_all_parallel(&k_points);
model.show_band(&path, &labels, 501, "band_output")?;
```

`show_band` writes plotting data and a PDF below the output directory supplied
as its final argument.

### One k-point and Bloch Hamiltonian

```rust
let k = arr1(&[0.25, 0.0]);
let h_atom = model.gen_ham(&k, Gauge::Atom);
let h_lattice = model.gen_ham(&k, Gauge::Lattice);
let band = model.solve_band_onek(&k);
```

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

High-level response calculations follow one pattern:

1. Build a const-generic parameter structure.
2. Choose an `Occupation`, current operator, and integration enum explicitly.
3. Call one model method and read a named result structure.

Directions are `[f64; DIM]`, so a direction/model dimension mismatch is a
compile-time error.

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

### Occupations

```rust
let zero_temperature = Occupation::ZeroTemperature;
let physical_temperature = Occupation::FermiDirac {
    temperature_kelvin: 30.0,
};
let metallic_smearing = Occupation::FermiSmearing { width: 0.01 };
```

Use a finite thermal or smearing width for direct Fermi-surface calculations
that contain `-df/dE`. Energy-cut algorithms can represent the exact
zero-temperature delta function.

### Berry curvature

```rust
let xy = DirectionPair::new([1.0, 0.0], [0.0, 1.0]);
let berry_params = BerryCurvatureParams {
    directions: xy,
    current: CurrentOperator::Charge,
    broadening: 1e-3,
};
let k = arr1(&[0.2, 0.3]);

let bands = model.berry_curvature_at(&k, &berry_params)?;
let occupied = model.occupied_berry_curvature_at(
    &k,
    &berry_params,
    0.0,
    Occupation::ZeroTemperature,
)?;
```

`bands.berry_curvature` and `bands.energies` contain one value per band.
For a spin Hall kernel, set
`current: CurrentOperator::Spin(SpinDirection::Z)`.

### Hall conductivity

```rust
let mu = Array1::linspace(-2.0, 2.0, 101);
let mut params = HallConductivityParams::new([51, 51], xy, mu);
params.occupation = Occupation::FermiDirac {
    temperature_kelvin: 30.0,
};
params.current = CurrentOperator::Charge;
params.broadening = 1e-3;
params.integration = HallIntegration::EnergyCut;

let result = model.hall_conductivity(&params)?;
let sigma_vs_mu = result.conductivity;
```

Use `HallConductivityParams::at_mu` and `result.single()` for a scalar
chemical potential. `HallIntegration::Direct` performs a uniform k-point sum;
`EnergyCut` uses band-tracked simplex integration.

### Nonlinear Hall response

All public rank-three directions are current-first:

```rust
let directions = NonlinearHallDirections::new(
    [1.0, 0.0], // current
    [1.0, 0.0], // field 1
    [0.0, 1.0], // field 2
);
let mu = Array1::linspace(-1.0, 1.0, 101);

let mut intrinsic = IntrinsicNonlinearHallParams::new(
    [51, 51],
    directions,
    mu.clone(),
    Occupation::FermiSmearing { width: 0.01 },
);
intrinsic.integration = NonlinearHallIntegration::Direct;
let intrinsic_result = model.intrinsic_nonlinear_hall(&intrinsic)?;

let mut extrinsic = ExtrinsicNonlinearHallParams::new(
    [51, 51],
    directions,
    mu,
    Occupation::FermiSmearing { width: 0.01 },
);
extrinsic.field_symmetry = FieldSymmetry::Symmetrized;
extrinsic.current = CurrentOperator::Charge;
let extrinsic_result = model.extrinsic_nonlinear_hall(&extrinsic)?;
```

`FieldSymmetry::Ordered` returns one ordered extrinsic kernel;
`Symmetrized` averages the two external-field permutations. Direct
integration requires a finite Fermi window. Energy-cut integration accepts
`Occupation::ZeroTemperature`.

### Quantum geometry

```rust
let mu = Array1::linspace(-1.0, 1.0, 101);
let mut params = QuantumGeometryParams::new([51, 51], xy, mu);
params.occupation = Occupation::ZeroTemperature;
params.broadening = 1e-3;
params.integration = QuantumGeometryIntegration::Simplex;

let result = model.quantum_geometry(&params)?;
let metric = result.metric;
let berry_curvature = result.berry_curvature;
```

For reusable band-resolved data, use the `QuantumGeometry` trait methods
`quantum_geometry_at` and `quantum_geometry_on`.

### Optical conductivity

```rust
let frequencies = Array1::linspace(0.0, 4.0, 401);
let mut params =
    OpticalConductivityParams::new([51, 51], xy, frequencies, 0.0);
params.occupation = Occupation::FermiDirac {
    temperature_kelvin: 30.0,
};
params.broadening = 1e-2;
params.integration = OpticalIntegration::Simplex;
params.directions = OpticalDirections::Cartesian;

let result = model.optical_conductivity(&params)?;
let tensor_components = result.directions;
let sigma = result.conductivity;
```

`OpticalDirections::Pair(xy)` computes one projected component.
`Cartesian` returns all ordered `DIM * DIM` tensor components; rows of
`conductivity` correspond to entries in `result.directions`.

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
    model.floquet_effective_model(&drive, &truncation, [32, 32], None)?;
```

| API | Basis size | Intended regime |
|---|---:|---|
| `floquet_model` / `floquet_ham_onek` | `nsta * (2*n_max + 1)` | Full truncated Sambe problem |
| `floquet_effective_model` | `nsta` | Off-resonant, high-frequency expansion |

For a custom effective hopping range, the target vectors must be unique and
closed under `R -> -R`:

```rust
let options = FloquetEffectiveOptions::new()
    .with_order(1)
    .with_q_max(2)
    .with_target_hamR(array![
        [-1, 0],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, 0],
    ]);

let effective = model.floquet_effective_model(
    &drive,
    &truncation,
    [32, 32],
    Some(&options),
)?;
```

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

## 10. Conventions and build notes

- k-points are fractional reciprocal coordinates.
- The Bloch phase is `exp(2*pi*i*k·R)`.
- Orbital positions are fractional coordinates stored by rows.
- Real-space lattice vectors are columns of `Model::lat`.
- `Gauge::Lattice` uses only `R` in the Fourier phase.
- `Gauge::Atom` includes orbital-position phases.
- A spinful basis is ordered as spin-up orbitals followed by spin-down orbitals.
- `None` denotes a spin-independent operator; there is no
  `SpinDirection::None` variant.

### BLAS/LAPACK backends

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
cargo check --all-targets
cargo test --release --features intel-mkl-system
cargo clippy --all-targets --features intel-mkl-system
cargo doc --no-deps --features intel-mkl-system
```

Use release mode for numerical tests. Several integration-style tests invoke
gnuplot and regenerate tracked artifacts below `tests/`.
