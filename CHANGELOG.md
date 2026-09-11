# Changelog

## 0.7.3 — Unreleased

### Safety and numerical fixes

- Correct the interband optical Kubo kernel: retain both longitudinal absorption
  and the antisymmetric Hall response, with `e²/hbar` omitted. Unbroadened poles
  return errors instead of silently vanishing; Drude terms remain excluded.
- Hopping setters/adders identify onsite terms by `R=0` and orbital indices,
  independent of support row order. All effective Floquet paths place the origin
  first, validate results, and remain safe to edit with `add_onsite`.
- Floquet time-grid estimates no longer apply the Bessel precision margin as a
  minimum bandwidth. Saturated adaptive estimates use an amplitude-dependent
  analytic bound; an arbitrary fixed grid cap is not treated as sufficient.
- Low-level LAPACK eigensolvers return `Result`, reject non-square inputs and
  overflowing dimensions, support empty matrices, and pack logical matrices
  in column-major order. Both eigenvector APIs return row kets; the range
  solver no longer applies an extra conjugation.
- Replace the unfinished Bloch orbital-angular-momentum implementation with
  the complex band-space operator of Busch, Mertig and Göbel (2023), Eq. (3).
  The API takes only `kvec`, using eV and angstroms; the result is `L/hbar`
  with shape `(3, nsta, nsta)`. Invalid operators and degenerate bands return errors.
  This is distinct from atomic orbital matrices and bulk magnetization.
- Use inversion-paired tetrahedralizations throughout 3D response integration.
  Average both triangulations of partially occupied cut prisms to remove
  sensitivity to nearly equal vertex-energy ordering.
- Bound finite-temperature Hall energy-cut convolution by 72 quadrature
  samples per requested chemical potential, eliminating the `1/T`-sized table
  and its endpoint extrapolation.
- Use the same inclusive `1e-10` eV intrinsic interband-gap cutoff in direct
  and energy-cut kernels. Reject unsupported intrinsic spin-current requests.
- Model validation rejects nonfinite operators and lattices without a finite
  inverse. Response and DOS entry points check model data before calculation;
  DOS rejects nonfinite energy bounds. Irrep Seitz matching honors operation
  tolerances below `1e-8` instead of silently relaxing them.
- Full optical tensors share one eigendecomposition and tracking pass per
  mesh; simplex frequency scans reuse interpolated energies and kernels.
  Both extrinsic field orderings share the same band basis and tracking.
  Direct extrinsic integration retains only scalar band kernels after each k.
- Fix one-dimensional k meshes, validate mesh/plane allocation arithmetic,
  reject empty/invalid sampling, and retain every k-path node without NaNs.
  K-path distances retain the existing convention without `2*pi`.
- Correct `phy_0` to the documented superconducting flux quantum `h/(2e)`.
- Floquet sector enumeration returns `Result`; Sambe constructors check array
  dimensions, finite arithmetic and conservative per-link sampling requirements.
  Explicitly sufficient large grids remain allowed; DC modes contribute no
  time bandwidth. Returned real-space models place the origin block first.
  Quasienergy folding avoids overflow when both energy and frequency are finite.

### API changes

- Response input is explicit now. `Parameters<DIM>` holds `conditions`
  (`t_kelvin`, `mu_ev`, `omega_ev`, each `Sampling::Fixed(value)` or
  `Sampling::Values(series)`), `kmesh`, `direction`, `integration`, `spin`,
  `field_symmetry` and `eta_ev`. At most one axis may be `Values`; that axis
  is evaluated from one shared k-mesh preparation, so eigenstates, velocity
  kernels and band tracking are computed once instead of once per sample.
  Sampling several axes is rejected before any k-mesh work.
- Remove the whole `Parameters` constructor surface — `new`, `at_mu`,
  `with_temperature`, `with_spin`, `with_frequency`, `with_integration` —
  and construct with `Parameters::rank2`/`rank3` plus an explicit
  `Conditions` and `ResponseOptions`. `Integration` and `FieldSymmetry` no
  longer implement `Default`, and `eta_ev` is `Option<f64>`: every entry point
  that broadens a denominator requires it, while `intrinsic_nonlinear_hall`
  ignores it.
- A sampled `omega_ev` is rejected by the four frequency-independent entry
  points; `optical_conductivity` is the only response that may sample it.
- Response validation parameters are renamed: `T` becomes `t_kelvin`, `mu`
  becomes `mu_ev`, and a missing `eta_ev` is reported as
  `InvalidResponseParameter { parameter: "eta_ev" }`.
- Results carry `axis: ResponseAxis` (`Fixed`, or `Temperature` /
  `ChemicalPotential` / `Frequency` with the sampled values) instead of a
  copied `chemical_potentials` / `frequencies` grid; `single()` returns a
  scalar only for `Fixed`. `optical_conductivity` may sample `t_kelvin` or
  `mu_ev` as well as `omega_ev`.
- Per-k-point methods (`berry_curvature_at`, `occupied_berry_curvature_at`,
  `occupied_berry_curvature_on`, `quantum_geometry_at`, `quantum_geometry_on`)
  require every axis `Fixed` and reject a sampled
  axis instead of silently ignoring fields. They validate every field, including
  the k-mesh they do not read.
- The nonlinear Hall entry points reject the whole call when any sample reaches
  zero temperature with `Integration::Direct`, before any k-mesh work, instead
  of switching algorithm at an individual sample.
- `FloquetEffectiveOptions.harmonic_max` is `isize`; negative cutoffs are rejected.
  `floquet_effective_q_model` takes `(drive, options, wavevector_cartesian)` and
  no longer takes a Sambe truncation.
- Atomic `Model::orb_angular()` returns `(3, nsta, nsta)`, replacing
  `(dim_r, norb, norb)`, with spin-major duplication of the orbital blocks.
- Add `gen_ham_batch`, `gen_v_batch`, `gen_v_projected_batch`, and parallel/batched
  band solvers. Stored position matrices must span every `hamR` row, consistent
  with `Model::validate`; omitted blocks must be explicitly zero-padded.

### Build and repository changes

- Replace the sibling `cryspglib` path with a fixed public 0.2.1 Git commit.
  Publishing still requires releasing cryspglib 0.2.1 to crates.io first and
  switching this dependency to its registry version.
- Add CI for OpenBLAS/Netlib, optional symmetry, Rust 1.90, formatting, rustdoc,
  and clippy correctness/unused-Result checks. Run both exhaustive symmetry
  censuses on tags, scheduled runs and manual workflow runs.
- Remove workspace-dependent rustdoc configuration and fix broken links.
  Include example/benchmark source and both declared license texts in packages.
- Move example and test output into `target/`, remove tracked band-data
  products, and stop ignoring Rust source under `tests/`. Handle previously
  discarded example plotting errors.

This repair batch does not migrate every infallible library API to `Result`,
privatize `Model` fields, add Floquet q-scan caches, split large modules,
or clear all existing clippy style/complexity warnings. Those are separate
API, performance and maintenance changes. Git history has not been rewritten.

Known test coverage gap: native LAPACK `info != 0` failure paths do not yet
have a backend-independent regression test.
