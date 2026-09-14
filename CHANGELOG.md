# Changelog

## 0.7.3 — Unreleased

### Safety and numerical fixes

- Model, surface, and unfolding k-paths share validation and node-preserving
  sampling. Surface and unfolding paths now round and reserve segment intervals
  like `Model::k_path`, so their interior sample locations can change. Invalid
  paths return errors; unfolding retains the primitive-cell reciprocal metric
  and k-path distances retain the convention without `2*pi`.
- Direct nonlinear Hall rejects zero thermal widths and overflowing peak
  derivatives `0.25 / (k_B T)`, including inside scans, before preparing the
  k-mesh. Subnormal widths with finite peak derivatives remain valid.
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

- **Breaking:** all five `Berry` Wilson-loop methods return `Result` and take
  band selections as `&[usize]`. Empty, duplicate or out-of-range selections,
  malformed/nonfinite/unclosed loops, invalid grid counts and overflowing array
  sizes return errors. Solver and linear algebra errors propagate through the
  parallel loop/flux/Wannier APIs. Detected singular SVD overlaps and zero or
  nonfinite raw determinants (including underflow) return errors instead of
  undefined phases. No spectral-gap tolerance is added.
  Shared preparation preserves endpoint sewing and the two Wilson algorithms;
  Wannier sampling reuses the batch phase evaluator. Documentation now matches
  the existing radians convention and `(n_occ, n_loop)` batch/Wannier axes;
  numerical signs, units and axis order have not changed.
- **Breaking:** all eight `Solve` methods return `Result`; callers must handle
  errors or propagate with `?`. Invalid models, wrong-dimensional/nonfinite
  k-points, nonfinite generated Hamiltonians, and eigensolver failures return
  errors. Serial and parallel batches validate the model once, including empty
  batches. Numerical gauges, ordering, and row-ket output conventions are
  unchanged. The two energy-window methods consistently document `(low, high]`
  and absolute tolerance in model energy units. Callers propagate solver errors,
  including the `Berry` methods migrated above.
- `Model<true, DIM, R>::build_spin_matrix(SpinDirection)` returns
  `Array2<Complex<f64>>` directly: `S_a / ℏ = σ_a ⊗ I_norb / 2` in
  spin-major order, for either position-matrix storage type. The spin-current
  operator is `{S_a / ℏ, v} / 2`.
- Hall and extrinsic nonlinear Hall use separate inherent implementations for
  `Model<false, DIM, R>` and `Model<true, DIM, R>`. Callers generic over
  `const SPIN: bool` must specialize or supply their own trait bound for
  dispatch. Generic Berry callers need the bound
  `Model<SPIN, DIM, R>: BerryCurvature<DIM>`.
- Response input is explicit now. `Parameters<DIM>` holds exactly the three
  fields every Brillouin-zone response shares: `conditions` (`t_kelvin`,
  `mu_ev`, `omega_ev`, each `Sampling::Fixed(value)` or
  `Sampling::Values(series)`), `kmesh` and `integration`. At most one axis may
  be `Values`; that axis is evaluated from one shared k-mesh preparation, so
  eigenstates, velocity kernels and band tracking are computed once instead of
  once per sample. Sampling several axes is rejected before any k-mesh work.
- `Parameters` no longer carries per-method choices. `direction`, `spin`,
  `field_symmetry` and `eta_ev` moved into the entry-point signatures, so a
  caller never states a value the method ignores. `ResponseOptions`,
  `Parameters::rank2`/`rank3`, and the empty-`direction` sentinel for the full
  optical tensor are removed; a full Cartesian optical tensor is requested
  through the new `optical_conductivity_tensor`, which shares one
  diagonalization and band-tracking pass with the component entry point.
- Directions are now fixed-size arrays: `[[f64; DIM]; 2]` for rank-two
  responses and `[[f64; DIM]; 3]` in `(current, field_1, field_2)` order for
  rank-three responses. Rank and dimension are compile-time constraints, so a
  wrong number of direction rows no longer compiles; component finiteness and
  the zero-row rejection are still runtime errors. `eta_ev` is a plain `f64`
  and is absent from `intrinsic_nonlinear_hall`, which broadens no denominator.
  `Integration` and `FieldSymmetry` still implement no `Default`.
- `hall_conductivity` and `extrinsic_nonlinear_hall` take `spin:
  Option<SpinDirection>` directly and keep the previous rejection behaviour: a
  spinless model returns `TbError::SpinNotAllowed` for `spin: Some(_)`.
- Band-resolved per-k methods take no thermodynamic state at all:
  `quantum_geometry_at` and `quantum_geometry_on` now take only the
  k-point(s), the directions and `eta_ev`; `berry_curvature_at` additionally
  takes `spin: Option<SpinDirection>`. The
  occupation-weighted helpers `occupied_berry_curvature_at`/`_on` take one
  fixed DC `&Conditions` and still reject a sampled axis or nonzero frequency.
  None of these methods takes a k-mesh or an integration algorithm any more.
- `OpticalConductivityResult` no longer carries a `<DIM>` generic, since its
  fields never used it. `NonlinearHallResult` gains `single()`, matching
  `HallConductivityResult::single()`: it returns the scalar of a `Fixed`
  calculation and `None` for any sampled axis, including a one-element series.
- The four Brillouin-zone DC entry points and
  `occupied_berry_curvature_at`/`_on` require
  `omega_ev: Sampling::Fixed(0.0)`; nonzero fixed frequencies and sampled
  frequencies return structured errors. Optical response accepts both.
- Optical and quantum-geometry methods, like intrinsic nonlinear Hall, no
  longer accept a spin request at all: their signatures carry no `spin`
  argument, so the old runtime rejection became a compile error. Hall,
  extrinsic nonlinear Hall and the Berry helpers keep
  `spin: Option<SpinDirection>`, and a spinless model still returns
  `TbError::SpinNotAllowed`.
- Response validation parameters are renamed: `T` becomes `t_kelvin` and `mu`
  becomes `mu_ev`. `eta_ev` is a required `f64` argument wherever a denominator
  is broadened, so there is no "missing `eta_ev`" state; a negative or
  non-finite value is reported as
  `InvalidResponseParameter { parameter: "eta_ev" }`.
- Results carry `axis: ResponseAxis` (`Fixed`, or `Temperature` /
  `ChemicalPotential` / `Frequency` with the sampled values) instead of a
  copied `chemical_potentials` / `frequencies` grid; `single()` returns a
  scalar only for `Fixed`. `optical_conductivity` may sample `t_kelvin` or
  `mu_ev` as well as `omega_ev`.
- Validation is centralised in `Conditions::resolve`, so for a call that is
  invalid in two ways at once the reported parameter can differ from before:
  a negative `t_kelvin` sample combined with two sampled axes now reports
  `t_kelvin` rather than the second sampled axis. The error variant, the
  rejection itself, and the guarantee that it happens before any k-mesh work
  are unchanged.
- The occupation-weighted per-k helpers (`occupied_berry_curvature_at`,
  `occupied_berry_curvature_on`) take one fixed DC `Conditions`, require every
  axis `Fixed`, and reject a sampled axis instead of silently ignoring it. The
  band-resolved per-k methods (`berry_curvature_at`, `quantum_geometry_at`,
  `quantum_geometry_on`) take no thermodynamic state and no k-mesh at all, so
  they can no longer validate fields they never read.
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
