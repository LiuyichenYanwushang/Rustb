# Rustb–cryspglib integration and basis-ownership plan

Status: milestones A–E, F1 (exact Hamiltonian-symmetry certification plus
residual magnetic-group identification), and F2 (validated opt-in Hamiltonian
symmetrization), plus strict global-frame atomic-orbital representations, are
implemented for Rustb 0.7 pre-release. Wiring weighted meshes into response
solvers, local-frame/radial-channel representations, gauge-covariant Peierls
actions, and numerical band irrep/corep assignment remain follow-up work.

## 1. Objective and non-objectives

Rustb will use cryspglib as an optional, pure-Rust crystallographic-symmetry
backend. The two crates remain independent projects and the dependency is
strictly one-way:

```text
Rustb Model
  -> checked crystal adapter
  -> cryspglib Crystal / symmetry analysis
  -> Rustb-owned result and query types
```

The structure adapter never assumes that a tight-binding Hamiltonian respects
every detected structure operation. Milestone F1 adds a separate, explicit
Hamiltonian certification call whose basis representation is supplied by the
caller (or by a deliberately strict scalar-site provider). Symmetrization and
numerical band-irrep assignment remain separate opt-in milestones.

The integration must preserve the common Rustb workflow in which a model has
orbitals but no atomic-site metadata.

## 2. Basis and ownership redesign

### 2.1 Ownership rule

`Model` remains the sole owner of the dense orbital arrays and Hamiltonian.
`Atom` does not own, copy, or borrow orbital storage. Instead, it holds typed
model-local orbital identifiers:

```rust
pub struct OrbitalId(usize);
pub struct AtomId(usize);

pub struct Atom {
    position: Array1<f64>,
    name: AtomType,
    orbitals: Vec<OrbitalId>,
    magnetic_moment: Option<[f64; 3]>,
}
```

An `OrbitalId` indexes the same row in `Model::orb` and
`Model::orb_projection`, and the corresponding orbital axes in `ham` and
`rmatrix`. This is a safe reference-like handle, not a raw pointer.

Rust references, raw pointers, and `Rc`/`Weak` are rejected for persistent
storage because they would make `Model` self-referential and fragile under
move, clone, serde, array reallocation, supercell construction, cutting, and
orbital reordering.

Borrowed access is provided at call time:

```rust
let atom = model.atom(AtomId::new(0))?;
for orbital in atom.orbitals() {
    println!("{:?}", orbital.position());
}
```

Here `AtomView<'model>` and `OrbitalRef<'model>` contain ordinary Rust borrows,
so the compiler prevents model mutation while a view is live.

### 2.2 Supported model states

The following are all valid:

- orbitals with no atoms (`atoms.is_empty()`);
- atoms with no selected TB orbitals;
- orbitals not assigned to any atom;
- an atom owning non-contiguous orbital IDs;
- orbital centers different from atomic positions.

Each orbital may belong to at most one atom. The reverse
`orbital -> Option<AtomId>` table is derived by `Model::validate()` and is not
serialized as a second source of truth.

### 2.3 Required invariants

`Model::validate()` checks at least:

- lattice and coordinate dimensions;
- finite lattice/positions and a non-singular lattice where required;
- `orb_projection.len() == norb`;
- every `OrbitalId` is in range;
- an orbital has at most one atomic owner;
- `ham.shape() == (hamR.nrows(), nsta, nsta)`;
- `hamR.ncols() == DIM`;
- `rmatrix`, when present, is aligned with `hamR`, `DIM`, and `nsta`.

Public symmetry entry points validate on every call because Rustb currently
exposes model fields publicly.

### 2.4 Migration of model-changing operations

Every operation that selects or reorders orbitals constructs one explicit map:

```text
old OrbitalId -> Option<new OrbitalId>
```

The same map updates orbital positions, projections, Hamiltonian axes,
position-matrix axes, and every atom's orbital IDs. This rule applies to
orbital removal, atom removal, atom/orbital reordering, supercells, cuts,
Floquet/Sambe models, unfolding, and Wannier90 import.

`remove_atom` keeps its historical cascade behavior during the 0.7 migration,
but clearer APIs are added:

- `remove_atoms_and_orbitals`;
- `remove_atoms_only`, which leaves the former orbitals unassigned;
- `prune_empty_atoms`.

## 3. Dependency and public API boundary

### 3.1 Cargo feature

Rustb adds one optional feature:

```toml
cryspglib = { path = "../cryspglib", version = "0.2.0", optional = true,
              default-features = false }

[features]
cryspglib = ["dep:cryspglib"]
```

cryspglib never depends on Rustb. Rustb does not wildcard-re-export
cryspglib. During local development Cargo uses the sibling path; publication
requires publishing the matching cryspglib version first.

### 3.2 Rustb API

The entry point is an extension trait implemented for 3D models:

```rust
pub trait CrystalSymmetry {
    fn crystal_symmetry(
        &self,
        parameters: &SymmetryParameters,
    ) -> Result<CrystalSymmetryDataset>;

    fn magnetic_crystal_symmetry_from_atoms(
        &self,
        parameters: &SymmetryParameters,
    ) -> Result<MagneticCrystalSymmetry>;

    fn magnetic_crystal_symmetry(
        &self,
        moments: &[[f64; 3]],
        parameters: &SymmetryParameters,
    ) -> Result<MagneticCrystalSymmetry>;
}
```

`SymmetryParameters` contains structural tolerances and the optional field
context. Every Atom has an optional finite Cartesian moment: `None` is the
nonmagnetic default; `Atom::set_magnetic_moment` attaches a moment; and
`Atom::clear_magnetic_moment` removes it. The `_from_atoms` entry points use
that stored metadata directly, while `magnetic_crystal_symmetry` remains an
explicit per-call override. `Model<true, ..>` only describes a spinful basis
and is never treated as proof of magnetic order.

Uniform electric and magnetic fields already encoded in a Hamiltonian are
supplied explicitly as optional Cartesian vectors in `SymmetryParameters` for
that analysis call and forwarded into cryspglib. They are not stored in
`Model`. The result distinguishes
the structural operation group from the effective subset preserving the fields:
electric field is time-even/polar; magnetic field is time-odd/axial. A later
Hamiltonian-representation validator remains necessary for arbitrary
symmetry-breaking terms that cannot be described by uniform E/B metadata.

When fields reduce the operation set, canonical high-symmetry points and
character tables of the structural database group are rejected with
`FieldReducedSymmetryData`; they are not presented as effective-group data.
Irreducible meshes are generated from the field-preserving operations,
including combined spatial×time-reversal operations where applicable.

Rustb owns its public result types so that cryspglib 0.x changes do not become
uncontrolled Rustb API changes. Results retain all setting context required by
later queries: space-group number, Hall number, transformation matrix, origin
shift, standardized lattice, operations, Wyckoff data, and atom mappings.

### 3.3 Conversion contract

Rustb stores real-space lattice vectors as rows and uses
`fractional.dot(model.lat)` for Cartesian coordinates. cryspglib uses
`lattice[cartesian_component][lattice_vector]`, so the adapter transposes:

```text
cryspglib_lattice[i][j] = model.lat[[j, i]]
```

The adapter uses atomic positions, never Wannier/orbital centers. Atomic types
are converted through `AtomType::atomic_number()` rather than enum casts.

The first implementation targets `DIM = 3`. A later explicit 2D adapter may
embed a 2x2 lattice into 3D with a user-provided positive vacuum length and
`AperiodicAxis::Z`. `DIM = 1` is rejected because cryspglib's layer API models
one, not two, aperiodic axes.

## 4. Functional milestones

### Milestone A — model ownership foundation

Implementation:

1. Introduce `AtomId`, `OrbitalId`, `AtomView`, and `OrbitalRef`.
2. Replace the atom orbital count with `Vec<OrbitalId>`.
3. Add explicit atom and orbital-only construction paths.
4. Add `Model::validate`, ownership lookup, and remapping helpers.
5. Migrate active model transformations away from prefix-count assumptions.
6. Keep the typed-ID serde representation validated on load. The implemented
   legacy-count reader maps an all-count file to contiguous IDs in Atom order
   and rejects files that mix legacy counts with typed IDs; the typed format is
   required for non-contiguous or unassigned ownership.

Expected result:

- atoms can safely reference arbitrary orbitals;
- dense orbital/Hamiltonian storage is unchanged;
- orbital-only models remain valid;
- cloning and serialization contain no self-references.

Tests:

- non-contiguous orbital ownership;
- unassigned orbitals and empty atoms;
- duplicate/out-of-range ownership errors;
- removal/reorder remapping;
- spinless/spinful and `NoRMatrix`/`HasRMatrix` validation;
- supercell, cut, Floquet, unfold, and serde regressions.

### Milestone B — optional symmetry adapter

Implementation:

1. Add the `cryspglib` Cargo feature and conditional module/export.
2. Add structured `TbError` variants preserving `cryspglib::SymError` as the
   source.
3. Validate the model and require non-empty, explicitly typed atoms.
4. Transpose the lattice and copy finite fractional site positions and atomic
   numbers into `cryspglib::Crystal`.
5. Convert the returned dataset into Rustb-owned types.

Expected result:

- feature-off Rustb has no cryspglib dependency in its API;
- feature-on 3D models return space group, Hall setting, point group,
  operations, Wyckoff letters, site symmetries, equivalent atoms, and
  primitive/standard cell metadata;
- malformed models return `Result::Err` and never panic.
- `tb_model(..., None)` is orbital-only and symmetry analysis rejects it with
  `MissingAtomicStructure`; no orbital center is promoted to an atom.

Tests:

- feature-off and feature-on compilation;
- a skew P1 lattice that detects accidental transpose;
- simple cubic Pm-3m;
- diamond/Si known group and equivalent atoms;
- atom centers deliberately different from orbital centers;
- missing atoms and invalid model invariants return structured errors.

### Milestone C — magnetic symmetry

Implementation:

1. Add explicit site-moment input and finite/count validation.
2. Call cryspglib magnetic analysis and preserve UNI, BNS, OG, parent Hall,
   time-reversal flags, and complete magnetic operations.
3. Never use operation-only classification where a detected parent Hall is
   available.

Expected result:

- nonmagnetic and magnetic results are distinguishable without sentinel
  values;
- the known ambiguous UNI settings retain sufficient parent/family Hall
  context and do not silently collapse to UNI 0.

Tests:

- `SPIN=true` without moments remains ordinary structural analysis;
- ferro-, antiferro-, and non-collinear fixtures;
- invalid moment length/NaN;
- an ambiguity regression retaining parent Hall context.

### Milestone D — high-symmetry queries and character tables

Implementation:

1. Expose canonical high-symmetry point records with an explicit coordinate
   frame tag.
2. Add a tested reciprocal-basis conversion from the cryspglib data-Hall frame
   to the original Rustb input frame before points may be passed to `k_path`.
3. Distinguish isolated points from parameterized symmetry lines.
4. Return structured character tables as well as formatted Markdown tables.
5. For magnetic groups, query corepresentations with the retained UNI and
   Hall-setting context.
6. Keep character-table operation headers in their canonical database frame
   and expose that exact ordered operation list separately from input-basis
   structural operations.

Expected result:

- users can list high-symmetry points for a detected model, choose a label,
  obtain its complete operation/class-column character table, and build a
  Rustb path in the original reciprocal basis;
- nonstandard Hall settings cannot silently reuse canonical coordinates.

Tests:

- standard and alternate Hall settings of the same group;
- reciprocal-coordinate round trips on a skew cell;
- ordinary character tables and magnetic corep tables;
- UNI 282–284 and other setting-sensitive cases;
- unsupported lines/settings return an explicit error.

### Milestone E — irreducible weighted meshes

Implementation:

1. Convert cryspglib grid addresses and mapping table into the Rustb-owned
   `IrreducibleKMesh` result.
2. Derive normalized multiplicity weights and retain the full-to-irreducible
   map in both cryspglib and Rustb `gen_kmesh` order.
3. Provide separate structural/time-reversal and explicit-moment magnetic mesh
   entry points so magnetic order is never inferred from `SPIN`.
4. Add weighted sampling to direct Brillouin-zone summation (follow-up).
5. Keep simplex and energy-cut algorithms on their full topological mesh until
   they receive a symmetry-aware topology implementation.

Expected result:

- callers can consume normalized irreducible meshes directly without assuming
  cryspglib and `gen_kmesh` share an index order;
- after response integration is added, direct sums can use symmetry reduction
  without changing normalization, while unsupported topological quadratures
  must reject weighted input instead of producing plausible but wrong numbers.

Tests:

- weights sum to one and multiplicities sum to the full mesh size;
- full-grid and weighted direct sums agree for invariant scalar/vector test
  integrands;
- shifted/unshifted meshes and time-reversal toggles;
- index ordering is checked rather than assumed to match `gen_kmesh`.

### Milestone F — Hamiltonian symmetry and band representations

#### F1 — implemented exact certification and residual MSG identification

Implementation flow:

1. `Model::check_hamiltonian_symmetry` validates model shapes, explicit Atom
   ownership of every orbital, finite matrix elements, unique `hamR`, complete
   `R <-> -R` storage, and
   `H(-R) = H(R)^dagger` before testing any operation.
2. Detect the Atom-based structural space group, form its grey extension
   `G + G 1'` by default, and then apply the per-call uniform electric/magnetic
   field context. Testing only the unitary structural group is available as a
   diagnostic mode but cannot return a certified final MSG.
3. Resolve every candidate through `BasisSymmetryRepresentation`.
   `AtomicOrbitalBasis` automatically handles complete atom-centred,
   global-frame Wannier90 `s/p/d/f` and hybrid projection subspaces, including
   orbital cell representatives and the spin action. `ScalarSiteBasis` remains
   the one-`s` special case. Repeated radial channels, incomplete shells, local
   frames, and custom Wannier gauges provide finite `CellShiftAction` matrices
   explicitly.
4. Represent the localized action as

   ```text
   g |b,R> = sum_(s,a) (D_s^g)_(a,b) |a, W R + s>.
   ```

   Validate its Laurent-operator unitarity before using it. Spinful scalar
   sites use the axial `SU(2)` lift of the Cartesian operation and the Rustb
   basis order `[all up orbitals, all down orbitals]`; anti-unitary operations
   include `i sigma_y K`.
5. On the complete finite real-space support, evaluate

   ```text
   K_g(R) = sum_(s,t) D_s^dagger H(W R + t - s) D_t.
   ```

   Require `H(R) = K_g(R)` for unitary operations and
   `H(R) = K_g(R)*` for anti-unitary operations. The comparison domain includes
   inverse images of all stored hopping blocks, so a transformed nonzero term
   cannot hide at an absent (implicitly zero) `R`. This is a proof on the
   finite Laurent polynomial, not a sampled-k heuristic.
6. Return one residual and worst `(R, bra, ket)` witness per operation, retain
   every validated localized action/sewing matrix for later little-group work,
   and distinguish `Preserved`, `Broken`, and `Unresolved`. An unresolved
   orbital action gives a lower bound and never masquerades as physical
   symmetry breaking.
7. Pass the survivor set to cryspglib's
   `ValidatedMagneticOperationSet`, which checks identity, duplicates,
   inverses, and closure. It derives the surviving group's own family Hall
   setting before UNI/BNS identification. The higher structural Hall is stored
   only as provenance, preventing the known Type-IV setting ambiguity from
   being “resolved” with the wrong supergroup Hall.

#### F2 — implemented forced Hamiltonian symmetrization

Public API:

```rust
let symmetrized = model.symmetrize_hamiltonian(
    &target_magnetic_group,
    &basis_representation,
    &HamiltonianSymmetrizationParameters::default(),
)?;
```

Implementation flow:

1. Validate the input Model, finite/Hermitian hopping support, tolerances, and
   the supplied magnetic operation set. cryspglib normalizes fractional
   translations and proves identity, uniqueness, inverses, and closure.
2. Recompute the admissible magnetic operations from the current Model's
   lattice, Atom positions and species, optional Atom moments, and explicit
   electric/magnetic field context. Every target operation must be present in
   this recomputed set. Any mismatch returns
   `TbError::TargetMagneticGroupIncompatible` before the basis provider is
   called and before any hopping matrix is averaged.
3. Identify the normalized target operations and require matching UNI,
   BNS/OG, and magnetic type metadata. The structural-parent SG/Hall fields
   are not confused with the residual MSG family SG/Hall fields.
4. Resolve and validate one localized Laurent action for every normalized
   target operation. Besides individual unitarity and orbital-centre geometry,
   require the actions to form a projective magnetic corepresentation:

   ```text
   sum_(s + W_g t = u) D_s^g (D_t^h)^(conj if g antiunitary)
       = z_(g,h) D_(u-n)^(gh),  |z_(g,h)| = 1.
   ```

   The global phase admits spin-half double groups and `T^2 = -1` without
   admitting unrelated per-operation matrices.
5. Apply the real-linear magnetic Reynolds projection on the complete
   symmetry-generated support:

   ```text
   H_sym(R) = (1 / |M|) sum_(g in M) P_g H(R),
   P_g H(R) = C_theta [sum_(s,t) D_s^dag H(W_g R+t-s) D_t].
   ```

   Close support under `R <-> -R`, enforce Hermiticity pairwise, keep `R=0`
   in `hamR` row zero, retain original support order where possible, and append
   generated blocks deterministically.
6. Return a new `Model`; `self` is unchanged. Existing `HasRMatrix` blocks are
   remapped by lattice vector, while newly generated Hamiltonian-support rows
   receive zero `rmatrix` blocks because position-matrix data is a different
   operator and is not silently symmetrized as a scalar.
7. Revalidate the returned Model and rerun every complete-support covariance
   equation. A failed postcheck returns an error rather than a partially or
   plausibly symmetrized Model.

Expected result:

- slightly symmetry-broken hoppings can be projected onto an explicitly chosen
  compatible MSG;
- lattice/Atom/species/moment/field incompatibility is a hard pre-projection
  error;
- the original Model is never mutated;
- nonsymmorphic cell shifts, antiunitary actions, spin-half projective phases,
  generated hopping support, Hermiticity, and `rmatrix` alignment remain
  explicit and testable.

Implemented tests:

- cubic hoppings `1, 2, 3` average to `2`, the input stays unchanged, and a
  second projection is idempotent;
- spinless grey projection removes imaginary time-reversal-breaking hopping;
- spinful grey projection removes a Zeeman term;
- support expansion creates missing cubic partners while keeping `hamR[0]=0`;
- `HasRMatrix` original rows retain their blocks and generated rows are zero;
- Type-IV staggered order preserves half-translation times time reversal;
- `T^2=-1` and anti-half-translation cell-shift composition are accepted;
- inconsistent projective actions and unsupported orbital metadata are hard
  errors;
- incompatible lattice, Atom type, stored moments, and external fields fail
  before basis resolution.

#### F3 — partially implemented orbital actions; deferred band representations and general gauges

1. Complete `s/p/d/f` shells and Wannier90 hybrid families are now inferred
   from Atom ownership plus `OrbProj` in the global Cartesian frame. Add
   explicit radial-channel and local-frame metadata before extending inference
   to repeated shells. `OrbProj` alone remains insufficient for a general
   Wannier gauge, incomplete shells, or spin-orbit-entangled orbitals.
2. Add gauge-covariant Peierls-field actions where spatial operations require
   position-dependent `U(1)` compensation.
3. Restrict retained sewing matrices to little groups, handle degenerate
   eigenspaces, and assign numerical band irreps/coreps with setting-aware
   cryspglib tables.

Expected result:

- structure symmetry and Hamiltonian symmetry are reported separately;
- unsupported local-frame/radial/user Wannier representations require explicit
  user matrices rather than being guessed;
- a closed reduced operation set is identified as its final UNI/BNS magnetic
  group with family-Hall and coordinate-transform metadata;
- threshold-induced nonclosure and incomplete basis metadata are explicit
  `Inconclusive` results rather than guessed groups;
- no input model is modified; opt-in symmetrization returns a new model.

Tests:

- implemented: cubic scalar model retains all 96 grey operations;
- implemented: anisotropic hopping reduces cubic symmetry to a 16-operation
  orthorhombic grey group;
- implemented: complex directed hopping breaks pure time reversal and yields a
  Type-III group;
- implemented: spinful Zeeman onsite term exercises the axial `SU(2)` and
  anti-unitary paths;
- implemented: half translation produces separate zero/cross-cell action
  sectors and the correct lattice-gauge Bloch phase;
- implemented: uniform magnetic-field context reduces the grey candidate set
  before Hamiltonian checking;
- implemented: non-Hermitian public mutation is rejected and unsupported
  orbital metadata returns `Inconclusive`, not `Broken`;
- implemented: Type-IV staggered Zeeman order, nonsymmorphic complete-support
  covariance, orbital-cell representative gauge, and spinful antiunitary
  sewing conventions;
- implemented: forced symmetrization, incompatible-target rejection,
  projective-corepresentation validation, support expansion, idempotence, and
  `HasRMatrix` alignment;
- deferred with F3: arbitrary Wannier-gauge covariance, complete `p/d/f`
  shells, Peierls gauge compensation, and degenerate-band character/projector
  tests.

## 5. Verification commands

Use one BLAS backend at a time and run numerical tests in release mode:

```bash
cargo fmt --all -- --check
# Feature-off Rustb no longer compiles: no default BLAS backend is selected (intentional E0080).
cargo check -p Rustb --features openblas-system
cargo check -p Rustb --features openblas-system,cryspglib
cargo check -p Rustb --features intel-mkl-system,cryspglib
cargo test -p Rustb --release --features intel-mkl-system,cryspglib --lib
cargo clippy -p Rustb --release --all-targets --features intel-mkl-system,cryspglib -- -D warnings
cargo doc -p Rustb --release --no-deps --features intel-mkl-system,cryspglib
```

Do not use `--all-features`: Rustb's allocator features are mutually exclusive,
and BLAS backends must not be enabled together.

## 6. Delivery and acceptance

Each milestone is separately reviewable and must update its unit tests,
rustdoc, README, SKILLS, and CLAUDE notes with the public API. Existing generated
PDF outputs are user-owned and are never included in these changes.

The current delivery is accepted when milestones A–E, F1, and F2 compile and
pass release tests, and the plan records deferred F3 limitations explicitly. No
API may return a fake nonmagnetic or zero identifier on failure: invalid input
uses structured `Result` errors, while physically incomplete classification
uses an explicit `Inconclusive` report carrying its reason.
