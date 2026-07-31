# Rustb

Rustb is a Rust 2024 library for tight-binding calculations in condensed-matter
physics. It provides model construction, band structures, density of states,
linear and nonlinear response, quantum geometry, Wilson loops, surface Green
functions, Wannier90 import, magnetic fields, band unfolding, Fermi surfaces,
and Floquet calculations.

[![Crates.io](https://img.shields.io/crates/v/Rustb.svg)](https://crates.io/crates/Rustb)

The current API version is **0.8.0** and uses the const-generic model type
`Model<SPIN, DIM, R>`.

## Installation

Choose a BLAS/LAPACK backend suitable for the target system:

```toml
[dependencies]
Rustb = { version = "0.8", features = ["intel-mkl-static"] }
ndarray = "0.17"
num-complex = "0.4"
```

Available backends are `intel-mkl-static`, `intel-mkl-system`,
`openblas-static`, `openblas-system`, `netlib-static`, and `netlib-system`.
The optional `mimalloc` and `jemalloc` allocator features are mutually
exclusive and can be combined with one backend feature.

Rustb requires Rust 1.90 or newer.

## Quick start

This example builds a spinless two-dimensional graphene model, plots its band
structure, and evaluates a Gaussian-broadened density of states:

```rust
use ndarray::{arr1, arr2, array, Array1};
use Rustb::*;

fn main() -> Result<()> {
    // Lattice vectors are stored as columns; orbital positions are rows in
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

For a spinful model, use `Model::<true, DIM>`. Spin-independent terms take
`None`; Pauli-matrix terms take `SpinDirection::X`, `Y`, or `Z`:

```rust
let mut model = Model::<true, 2>::tb_model(lat, orb, None)?;
model.set_onsite(&arr1(&[0.5, -0.5]), None);
model.add_hop(0.2, 0, 0, &array![1, 0], SpinDirection::Z);
```

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

Wannier90 models can be loaded as:

```rust
let model: Model<false, 3> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;

let model_with_r: Model<false, 3, HasRMatrix> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;
```

## Response calculations

High-level response methods use a configuration/result pair. The same
`Occupation` type is shared with the Hubbard solver, and integration algorithms
are selected explicitly instead of by calling a different method name:

```rust
let xy = DirectionPair::new([1.0, 0.0], [0.0, 1.0]);
let mu = Array1::linspace(-1.0, 1.0, 201);

let mut hall = HallConductivityParams::new([101, 101], xy, mu.clone());
hall.occupation = Occupation::FermiDirac {
    temperature_kelvin: 20.0,
};
hall.integration = HallIntegration::EnergyCut;
let hall_result = model.hall_conductivity(&hall)?;

let mut geometry = QuantumGeometryParams::new([101, 101], xy, mu.clone());
geometry.integration = QuantumGeometryIntegration::Simplex;
let geometry_result = model.quantum_geometry(&geometry)?;

let mut optical = OpticalConductivityParams::new(
    [101, 101],
    xy,
    Array1::linspace(0.0, 4.0, 401),
    0.0,
);
optical.integration = OpticalIntegration::Simplex;
let optical_result = model.optical_conductivity(&optical)?;
```

For nonlinear Hall calculations, all tensor indices are current-first and
encoded in one value:

```rust
let directions = NonlinearHallDirections::new(
    [1.0, 0.0], // current
    [1.0, 0.0], // first field
    [0.0, 1.0], // second field
);
let params = IntrinsicNonlinearHallParams::new(
    [101, 101],
    directions,
    mu,
    Occupation::FermiSmearing { width: 0.01 },
);
let nonlinear_result = model.intrinsic_nonlinear_hall(&params)?;
```

Results use named fields such as `conductivity`, `metric`,
`berry_curvature`, `frequencies`, and `diagnostics`; response methods no
longer return positional tuples. Direct integration supports 1D–3D where the
quantity is defined. Simplex and energy-cut paths support their documented 2D
or 3D subsets.

## Hubbard mean field

`HubbardModel` adds orbital-resolved on-site interactions to a spinful model.
Its non-collinear unrestricted Hartree-Fock solver updates the complete local
`2 × 2` spin-density matrix, including Hartree density terms and Fock spin-flip
terms. It can either hold the chemical potential fixed or preserve the filling
calculated from the bare model at a reference Fermi level:

```rust
let mut bare = Model::<true, 1>::tb_model(
    array![[1.0]],
    array![[0.0]],
    None,
)?;
bare.add_hop(-1.0, 0, 0, &array![1], None);

let hubbard = HubbardModel::with_uniform_u(bare, 2.0)?;
let mut params = MeanFieldParams::new(
    [200],
    MeanFieldConstraint::FixedInitialFilling {
        reference_mu: 0.0,
    },
    Occupation::FermiSmearing { width: 0.01 },
);
params.initial_magnetization = InitialMagnetization::UniformVector {
    moment_per_orbital: [1e-3, 0.0, 0.0],
};

let model = hubbard.solve_hartree_fock(&params)?;
let moment = model.spin_moment(&[200], 0.0, params.occupation)?;
```

The result is an ordinary `Model<true, DIM, R>`. Its converged chemical
potential has already been shifted to zero. Direct occupation sums are used
instead of integrating a broadened DOS; `FermiSmearing` is available for
zero-temperature metallic calculations.

## Main capabilities

- Model construction and transformations: `tb_model`, `set_hop`, `add_hop`,
  `set_onsite`, `make_supercell`, `cut_piece`, and `cut_dot`.
- Non-collinear unrestricted Hartree-Fock with orbital-dependent `U`, fixed
  chemical potential or fixed initial filling, metallic smearing, and spin
  observables.
- Solvers and output: `gen_ham`, `solve_band_onek`,
  `solve_band_all_parallel`, `show_band`, and `dos`.
- Response and geometry: anomalous Hall conductivity, nonlinear Hall
  conductivity, optical conductivity, Berry curvature, and quantum geometry.
- Topology: Berry phases, Berry flux, Wilson loops, and hybrid Wannier centres.
- Boundaries and fields: surface Green functions and uniform magnetic fields
  through the Peierls substitution.
- Interfaces: Wannier90 import, BXSF/FRMSF export, and band unfolding.
- Driven systems: Floquet-Sambe Hamiltonians and same-size van Vleck effective
  models.

See [SKILLS.md](SKILLS.md) for current signatures and practical examples.
The generated rustdoc contains the detailed mathematical conventions.

## Development

```bash
cargo fmt --check
cargo check --all-targets
cargo test --release --features intel-mkl-system
cargo clippy --all-targets --features intel-mkl-system
cargo doc --no-deps --features intel-mkl-system
```

Numerical tests should be run in release mode. Some integration-style tests
invoke gnuplot and regenerate files below `tests/`.

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License
