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

## Installation

Rustb has no default BLAS/LAPACK backend. Callers must enable exactly one of
the mutually exclusive backend features. The `openblas-system` default was
removed after 0.7.1, so upgrading callers must add a backend feature
explicitly. For system OpenBLAS, use:

```toml
[dependencies]
Rustb = { version = "0.7", features = ["openblas-system"] }
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
Rustb = { version = "0.7", features = ["intel-mkl-system", "cryspglib"] }
```

The optional `mimalloc` and `jemalloc` allocator features are mutually
exclusive and can be combined with one backend feature.

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
Hopping amplitudes now use `Into<Complex64>` and k meshes use `num_traits::Float`;
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
    &AtomicOrbitalBasis,
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

`AtomicOrbitalBasis` reads each Atom's owned orbital IDs and the corresponding
`orb_projection` labels. It automatically handles globally aligned Wannier90
`s/p/d/f`, `sp`, `sp2`, `sp3`, `sp3d`, and `sp3d2` functions for both
`SPIN=false` and `SPIN=true`; antiunitary operations include the appropriate
complex conjugation and, for spinful models, the spin-1/2 action. The provider
rejects incomplete projection sets whenever they are not closed under a target
operation, along with duplicate/radial channels, non-atom-centred orbitals, and
other non-closed local subspaces, instead of guessing. Local orbital
frames, SOC-entangled orbitals, or arbitrary Wannier gauges must implement
`BasisSymmetryRepresentation` and return explicit `LocalizedBasisAction`
cell-shift matrices. `ScalarSiteBasis` remains as the narrower one-`s`-orbital
special case. Missing basis metadata is `Unresolved`/`Inconclusive`, not a
false claim that the Hamiltonian broke the operation.

Every decided operation contains absolute/relative residuals and a worst
`(R, bra, ket)` witness. Validated sewing actions remain in the report for
little-group and band-irrep work. A final UNI/BNS label is returned only
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
    &AtomicOrbitalBasis,
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

### Magnetic band irreps and coreps

With the `cryspglib` feature, `calculate_irrep` detects the Atom-defined
magnetic group and prints an irvsp-like table at all isolated high-symmetry k
points:

```rust
let irreps = model.calculate_irrep(None)?;
println!("{irreps}");
```

`None` selects `IrrepCalculationOptions::default()`; pass `Some(&options)` to
override tolerances or external fields. The band-irrep entry point currently
supports the atom-centred basis recorded directly by `Atom` orbital IDs and
`Model::orb_projection` (`OrbProj::s`, `px`, `py`, `pz`, and the supported
`d`/`f`/hybrid projections). It does not accept a second basis argument and
does not guess local Wannier frames or general Wannier gauges.

Before diagonalization, the calculation verifies the full localized actions,
their `SPIN`-appropriate magnetic factor system (including
$\mathcal T^2=-1$ for spinful models), and exact covariance of the complete
real-space Hamiltonian. Covariance thresholds are invariant under adding a
constant energy to every band. It then diagonalizes independent k points in
parallel; operation-action preparation, projective composition checks, and
Hamiltonian covariance checks are parallel as well. It restricts each unitary
sewing matrix to consecutive degenerate bands and fits the resulting characters
to the magnetic-corepresentation table. The
printed table includes every raw complex unitary character; antiunitary rows
are marked `N/A` because `Tr(U K)` is not an ordinary character. Output order
remains deterministic. For efficient outer parallelism, use one BLAS thread
(for example `MKL_NUM_THREADS=1` or `OPENBLAS_NUM_THREADS=1`).

This API deliberately distinguishes two failures. If an orbital set is not
closed under a target operation, or a local Wannier representation cannot be
constructed, it returns `TbError::IrrepBasisRepresentation` before solving any
k point. If the representation exists but the Hamiltonian breaks the target
group, exact real-space residuals, raw complex characters, and fitted
multiplicities are retained while every unreliable target label becomes `???`.
Therefore the same call can diagnose an input model and verify the result of
`symmetrize_hamiltonian`.

For a caller-supplied target use:

```rust
let irreps = model.calculate_irrep_for_group(
    &target,
    Some(&IrrepCalculationOptions::default()),
)?;
```

When explicit options are supplied, their electric/magnetic fields are applied
to the target's full operation set before reidentification; with `None`, the
field context stored in `target` is retained. The current API expects a cell
compatible with the magnetic-irrep database setting. If multiple input
translations collapse onto one database operation (a folded, nonprimitive
supercell), a preflight error is returned rather than assigning primitive-cell
labels; unfold that model first.

## Floquet driven systems

`FloquetDrive` represents a commensurate periodic vector potential. Its base
photon energy `omega0_ev` is measured in eV, while each
`LightMode::a_complex` is the complex amplitude of `e A / hbar` in inverse
lattice-length units. The mode's integer harmonic is measured relative to the
base frequency:

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

// Full truncated Sambe problem and folded quasienergies.
let sambe_model = model.floquet_model(&drive, &truncation)?;
let quasienergies = model.floquet_quasienergy_onek(
    &k,
    &drive,
    &truncation,
    Gauge::Lattice,
)?;

// Same-size off-resonant van Vleck model.
let effective = model.floquet_effective_model(&drive, None)?;

// Retain terms through O(omega^-2).
let options = FloquetEffectiveOptions::new().with_order(2);
let effective_second_order =
    model.floquet_effective_model(&drive, Some(&options))?;
```

| API | Returned basis size | Intended use |
|-----|--------------------:|--------------|
| `floquet_model` / `floquet_ham_onek` | `nsta * (2*n_max + 1)` | Full truncated Sambe problem |
| `floquet_quasienergy_onek` | `nsta * (2*n_max + 1)` | Quasienergies folded into one Floquet zone |
| `floquet_effective_model` | `nsta` | Coherent, off-resonant van Vleck expansion |
| `floquet_effective_mode_resolved_model` | `nsta` | Mutually incoherent mode-diagonal correction sum |
| `floquet_effective_q_model` | `nsta` | Coherent long-wavelength correction through `O(A^2 q/W)` |

The real-space effective-model path determines its generated hopping support
automatically. `FloquetEffectiveOptions::harmonic_max` controls the commutator
sums and defaults to `2`; `order` defaults to `1` and may be `0`, `1`, or `2`.
All three effective-model APIs take only `FloquetEffectiveOptions` as numerical
controls. `FloquetTruncation` controls the photon cutoff `n_max` of the full
Sambe calculation; its coefficients no longer take a sampling count. When migrating an effective-model call that relied on
the old cutoff, set `.with_harmonic_max(2 * old_n_max)` explicitly to preserve
its harmonic range.
Use `floquet_effective_mode_resolved_model` when different modes are mutually
incoherent. Modes passed together to `floquet_effective_model` instead
interfere coherently inside the same Peierls phase. See
[`examples/floquet_chain/main.rs`](examples/floquet_chain/main.rs) for a
complete executable example.

Wannier90 models can be loaded as:

```rust
let model: Model<false, 3> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;

let model_with_r: Model<false, 3, HasRMatrix> =
    Model::from_hr("path/to/files/", "wannier90", 0.0)?;
```

## Response calculations

The Brillouin-zone responses share one small grid configuration,
`Parameters<DIM>`, holding exactly three fields:

```rust
pub struct Parameters<const DIM: usize> {
    pub conditions: Conditions,      // T, mu, omega; at most one sampled
    pub kmesh: [usize; DIM],         // uniform k-mesh
    pub integration: Integration,    // Direct / Simplex / EnergyCut
}
```

Everything a single method chooses is an argument of that method — the
projection directions, `eta_ev`, and (where a spin current exists) `spin` and
`field_symmetry`. A caller therefore never states a value the method ignores.
Nothing is defaulted: every argument is required.

`conditions` fixes a thermodynamic point and may sample **exactly one** of its
three axes. Each axis is either `Sampling::Fixed(value)` or
`Sampling::Values(series)`; a sampled axis reuses one k-mesh preparation, so
eigenstates, velocity kernels and band tracking are computed once no matter how
many samples are requested.

Directions are fixed-size arrays whose length is checked at compile time:
`[[f64; DIM]; 2]` for rank-two responses and `[[f64; DIM]; 3]` for rank-three
responses.

DC responses require `omega_ev: Sampling::Fixed(0.0)`. Optical,
quantum-geometry and intrinsic nonlinear Hall are charge-only: no `spin`
argument exists for them. Hall and extrinsic nonlinear Hall have separate
implementations for spinless and spinful models; a spinless model returns
`TbError::SpinNotAllowed` for `spin: Some(_)`. Wrappers generic over
`const SPIN: bool` must specialize their calls.

```rust
// Fixed T and mu, sweeping the chemical potential through the Hall response.
let mu = Array1::linspace(-1.0, 1.0, 201);
let hall = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(20.0),
        mu_ev: Sampling::Values(mu),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [101, 101],
    integration: Integration::EnergyCut,
};
let hall_result = model.hall_conductivity(&hall, [[1.0, 0.0], [0.0, 1.0]], 1e-3, None)?;

// Same grid, quantum geometry instead: there is no spin argument at all, so
// the call cannot misstate one. The trailing value is the denominator width.
let geometry = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(20.0),
        mu_ev: Sampling::Fixed(0.0),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [101, 101],
    integration: Integration::Simplex,
};
let geometry_result = model.quantum_geometry(&geometry, [[1.0, 0.0], [0.0, 1.0]], 1e-3)?;

// Fixed T and mu, sweeping the photon frequency. The full ordered Cartesian
// tensor has its own entry point; no sentinel direction matrix is involved.
let optical = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(20.0),
        mu_ev: Sampling::Fixed(0.0),
        omega_ev: Sampling::Values(Array1::linspace(0.0, 4.0, 401)),
    },
    kmesh: [101, 101],
    integration: Integration::Simplex,
};
let one_component = model.optical_conductivity(&optical, [[1.0, 0.0], [0.0, 1.0]], 1e-2)?;
let full_tensor = model.optical_conductivity_tensor(&optical, 1e-2)?;
```

For nonlinear Hall calculations, all tensor indices are current-first — the
three direction rows are `(current, field_1, field_2)`:

```rust
let params = Parameters {
    conditions: Conditions {
        t_kelvin: Sampling::Fixed(30.0),
        mu_ev: Sampling::Values(mu),
        omega_ev: Sampling::Fixed(0.0),
    },
    kmesh: [101, 101],
    integration: Integration::EnergyCut,
};
let intrinsic = model.intrinsic_nonlinear_hall(
    &params,
    [
        [1.0, 0.0], // current
        [1.0, 0.0], // first field
        [0.0, 1.0], // second field
    ],
)?;

// The extrinsic response takes one spin current and one field-ordering
// convention, both stated explicitly.
let extrinsic = model.extrinsic_nonlinear_hall(
    &params,
    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    1e-3,
    None,
    FieldSymmetry::Symmetrized,
)?;
```

Band-resolved methods take no thermodynamic state at all when they do not need
one. Only the occupation-weighted helpers take a `Conditions`, and never a
k-mesh or an integration algorithm:

```rust
// No temperature, chemical potential or k-mesh is involved.
let bands = model.berry_curvature_at(&k, [[1.0, 0.0], [0.0, 1.0]], 1e-3, None)?;

// Occupation weighting states the one fixed DC point it uses.
let occupied = model.occupied_berry_curvature_on(
    &k_points,
    &Conditions::fixed(300.0, 0.0, 0.0),
    [[1.0, 0.0], [0.0, 1.0]],
    1e-3,
    None,
)?;
```

Results use named fields such as `conductivity`, `metric`,
`berry_curvature` and `diagnostics`, together with `axis: ResponseAxis`
(`Fixed`, or `Temperature`/`ChemicalPotential`/`Frequency` carrying the
sampled values). Direct integration supports 1D–3D where the quantity is
defined. Simplex and energy-cut paths support their documented 2D or 3D
subsets.

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
- Driven systems: Floquet-Sambe Hamiltonians, same-size van Vleck effective
  models, and a coherent long-wavelength `O(A^2 q/W)` effective-model
  correction.

See [SKILLS.md](SKILLS.md) for current signatures and practical examples.
The generated rustdoc contains the detailed mathematical conventions.

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
