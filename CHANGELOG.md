# Changelog

## 0.7.3 — Unreleased

### Safety and numerical fixes

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
