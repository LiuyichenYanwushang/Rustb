# Rustb

Rustb is a Rust 2024 library for tight-binding calculations in condensed-matter
physics. It provides model construction, band structures, density of states,
linear and nonlinear response, quantum geometry, Wilson loops, surface Green
functions, Wannier90 import, magnetic fields, band unfolding, Fermi surfaces,
and Floquet calculations.

[![Crates.io](https://img.shields.io/crates/v/Rustb.svg)](https://crates.io/crates/Rustb)

The current API version is **0.7.0** and uses the const-generic model type
`Model<SPIN, DIM, R>`.

## Installation

Choose a BLAS/LAPACK backend suitable for the target system:

```toml
[dependencies]
Rustb = { version = "0.7", features = ["intel-mkl-static"] }
ndarray = "0.17"
num-complex = "0.4"
```

Available backends are `intel-mkl-static`, `intel-mkl-system`,
`openblas-static`, `openblas-system`, `netlib-static`, and `netlib-system`.
Add the optional `cryspglib` feature to enable crystallographic and magnetic
symmetry analysis without a C dependency:

```toml
Rustb = { version = "0.7", features = ["intel-mkl-system", "cryspglib"] }
```

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

## Optional crystal symmetry

With the `cryspglib` feature, a three-dimensional model with explicit atoms can
query structure and magnetic symmetry, high-symmetry points, complete character
tables, and irreducible reciprocal meshes:

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

let symmetry = model.crystal_symmetry(&SymmetryParameters::default())?;
let points = symmetry.high_symmetry_kpoints()?;
let gamma_table = symmetry.character_table_at("GM")?;
let gamma_columns = symmetry.character_table_operations()?;

let mesh = model.irreducible_kmesh(
    [12, 12, 12],
    [0, 0, 0],
    true,
    &SymmetryParameters::default(),
)?;
assert!((mesh.weights.sum() - 1.0).abs() < 1e-12);
```

Character-table operation columns use cryspglib's canonical database basis;
`character_table_operations()` returns the headers in that exact frame and
order. `symmetry.operations` instead remains in the input model basis.

Uniform electric and magnetic fields already encoded in a Hamiltonian must be
supplied explicitly to the symmetry call because the atomic lattice alone does
not contain this information:

```rust
let parameters = SymmetryParameters {
    external_fields: ExternalFields {
        electric: None,
        magnetic: Some([0.0, 0.0, 1.0]),
    },
    ..Default::default()
};
let symmetry = model.crystal_symmetry(&parameters)?;
```

`symmetry.operations` is the unchanged structural group;
`symmetry.field_preserving_operations` is the effective subset compatible with
the supplied fields. Rustb passes this context into cryspglib; it is not merely
post-processing hidden in the model adapter. If the field reduces the group,
structural-group high-symmetry points and character tables return
`FieldReducedSymmetryData` instead of being mislabelled as effective data. The
irreducible mesh is generated from the effective unitary and anti-unitary
operations. The fields are analysis inputs and are not stored in `Model`.

Each Atom instead carries an optional Cartesian magnetic moment. It defaults
to `None`, so ordinary structures are nonmagnetic until a caller explicitly
attaches a moment:

```rust
assert_eq!(model.atoms[0].magnetic_moment(), None);
model.atoms[0].set_magnetic_moment([0.0, 0.0, 1.0])?;

let magnetic = model
    .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;

model.atoms[0].clear_magnetic_moment();
```

`Some([0.0; 3])` is an explicit zero vector and `None` means no moment was
attached; both contribute zero to crystallographic magnetic-group detection.
The explicit `magnetic_crystal_symmetry(&moments, ...)` and
`magnetic_irreducible_kmesh(&moments, ...)` methods remain available as
per-call overrides. `SPIN=true` alone is never treated as magnetic order. The
boolean `time_reversal` argument of `irreducible_kmesh` is likewise an explicit
Hamiltonian-level assertion, not something inferred from `SPIN`.

### Hamiltonian compatibility and the residual magnetic group

Structure symmetry is only a candidate symmetry of a tight-binding model. To
test the actual hopping and onsite matrices, use the separate, read-only
Hamiltonian certification API:

```rust
let report = model.check_hamiltonian_symmetry(
    &ScalarSiteBasis::default(),
    &HamiltonianSymmetryRequest::default(),
)?;

match &report.final_group {
    FinalMagneticGroup::Identified(group) => {
        println!("residual MSG: UNI {}, BNS {}", group.uni_number, group.bns_number);
    }
    FinalMagneticGroup::Inconclusive { reason } => {
        println!("more basis metadata is required: {reason}");
    }
}
```

The default request tests the Atom-derived grey candidate group `G + G1'` so
that Type-II, Type-III, and Type-IV survivors can be discovered. Optional E/B
fields in `SymmetryParameters` filter those candidates first. The checker then
uses Rustb's exact finite real-space hopping support, including nonsymmorphic
cell shifts; it does not infer a group from a sampled k mesh.

`ScalarSiteBasis` is deliberately limited to one atom-centred `s` orbital per
Atom (with complete orbital ownership). For `p/d/f`, hybrid, local-frame, SOC
entangled, or arbitrary Wannier bases, implement
`BasisSymmetryRepresentation`—closures implementing the same signature are
accepted as well—and return explicit `LocalizedBasisAction` cell-shift
matrices. Missing basis metadata is `Unresolved`/`Inconclusive`, not a false
claim that the Hamiltonian broke the operation.

Every decided operation contains absolute/relative residuals and a worst
`(R, bra, ket)` witness. Validated sewing actions remain in the report for
future little-group and band-irrep work. A final UNI/BNS label is returned only
after cryspglib verifies group closure and derives the survivor's own family
Hall setting; the original structural Hall is provenance only.

### Forced Hamiltonian symmetrization

To project a slightly symmetry-broken Hamiltonian onto a chosen magnetic group,
use the separate opt-in constructor. It returns a new Model and never mutates
the input:

```rust
let target = model
    .magnetic_crystal_symmetry_from_atoms(&SymmetryParameters::default())?;

let symmetrized = model.symmetrize_hamiltonian(
    &target,
    &ScalarSiteBasis,
    &HamiltonianSymmetrizationParameters::default(),
)?;
```

Before resolving basis matrices or averaging any hopping, Rustb recomputes
compatibility against the current lattice, Atom positions and species,
optional Atom moments, and the supplied electric/magnetic field context. A
target from another structure/setting, or one broken by these moments or
fields, returns `TbError::TargetMagneticGroupIncompatible` immediately.

For a valid localized action, the implementation applies the complete
real-space magnetic Reynolds average, including nonsymmorphic cell shifts and
antiunitary conjugation. It validates projective group composition (so
spin-half phases and `T^2=-1` are supported), expands `hamR` to every generated
hopping block, restores Hermiticity, and rechecks every target covariance
equation. Existing `rmatrix` blocks remain aligned by lattice vector; newly
generated support receives zero position-matrix blocks. As with certification,
non-scalar Wannier gauges require an explicit `BasisSymmetryRepresentation`.

Wannier90 models can be loaded as:

```rust
let model: Model<false, 3> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;

let model_with_r: Model<false, 3, HasRMatrix> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;
```

## Response calculations

Every high-level response method shares a single configuration type,
`Parameters<DIM>`, with fields `T` (kelvin; `0.0` = zero temperature), `mu`
(eV), `eta` (broadening), `kmesh`, `omega` (eV), `spin`
(`None` = charge current), `direction` (`Array2<f64>`, shape `(rank, DIM)`),
`integration` (`Integration::Direct`/`Simplex`/`EnergyCut`), and
`field_symmetry` (extrinsic NLH only). Methods ignore the fields they do not
need, and each returns a named result structure:

```rust
let mu = Array1::linspace(-1.0, 1.0, 201);

let mut hall = Parameters::rank2([101, 101], [1.0, 0.0], [0.0, 1.0], mu.clone())
    .with_temperature(20.0);
hall.integration = Integration::EnergyCut;
let hall_result = model.hall_conductivity(&hall)?;

let mut geometry = Parameters::rank2([101, 101], [1.0, 0.0], [0.0, 1.0], mu.clone());
geometry.integration = Integration::Simplex;
let geometry_result = model.quantum_geometry(&geometry)?;

let mut optical = Parameters::rank2([101, 101], [1.0, 0.0], [0.0, 1.0], array![0.0]);
optical.omega = Array1::linspace(0.0, 4.0, 401);
optical.integration = Integration::Simplex;
let optical_result = model.optical_conductivity(&optical)?;
```

For nonlinear Hall calculations, all tensor indices are current-first — row 0
of the direction matrix is the current, rows 1-2 the fields:

```rust
let params = Parameters::rank3(
    [101, 101],
    [1.0, 0.0], // current
    [1.0, 0.0], // first field
    [0.0, 1.0], // second field
    mu,
)
.with_temperature(30.0);
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
