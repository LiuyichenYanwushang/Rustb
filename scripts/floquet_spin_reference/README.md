# Independent Hubbard dimer reference (T20a only)

This is a finite-state NumPy oracle, **not** the production path engine, a Rust
input API, M1 acceptance, or completion of T20. Full T20 still requires T10.
No Rust code, cargo command, added dependency, or production formula is used.

## Reproduce

From the repository root, using the already-installed NumPy:

```sh
mkdir -p target/goal-a-logs/task-8 target/goal-a-archives
OPENBLAS_NUM_THREADS=1 python3 -W error scripts/floquet_spin_reference/dimer_reference.py --self-check > target/goal-a-logs/task-8/check.json 2>&1
status=$?
printf 'self-check exit=%s\n' "$status"
gzip -c target/goal-a-logs/task-8/check.json > target/goal-a-logs/task-8/check.json.gz
gzip -t target/goal-a-logs/task-8/check.json.gz
# Preserve a nonzero self-check exit; inspect JSON before accepting a run.
test "$status" -eq 0
```

The JSON records Python/NumPy versions, source SHA-256, arguments, parameters,
residuals, diagnostics, and exit code. Exceptions preserve partial results and
traceback and exit 1. Checks use explicit exceptions, not removable Python
`assert` statements. Do not suppress warnings or enlarge tolerances to get PASS.
The interface deliberately has no parameter sweep: at most 126 states, M=8/10,
12 continuation steps per t, and DFT grids 2048/4096. This ceiling is for a small
reference, not a general solver's convergence policy.

## Physical contract

- Energies are eV. Uniform U=4, W=3; hopping amplitude t=.16,.08,.04.
  The dimensionless scalar Peierls phase is A cos(theta)+B sin(2 theta), with
  A=.8, B=.45. Static tests separately set A=B=0.
- Four fermion modes are spin-major `(0 up, 1 up, 0 down, 1 down)`.
  The six two-electron Fock states use ascending creation operators. Each
  `c_i^dag c_j` includes both annihilation and creation permutation signs.
  Explicit two-point checks establish the bra/ket direction and sign.
- The matrix element with bra site 0, ket site 1, displacement +1 is
  `t exp[-i phase(theta)]`, multiplying `c_0sigma^dag c_1sigma`; its reverse
  uses the conjugate phase and reversed fermion operator. Both are independently
  quadrature-sampled, not constructed by a Bessel library.
- `C_q=mean(exp(+iq theta) exp(-i phase))`, so `H(theta)=sum_q h_q exp(-iq theta)`.
  In the positive-time basis `exp(+in theta)`, temporal projection gives
  Sambe block `(n,m)=V^(m-n)+delta_nm(U Nd+nW)`, n,m in [-M,M].
  The derivative `-i W d/dtheta` supplies `+nW`; an `h_q` hop changes photon
  `m -> m-q`. Neither the coefficients nor the resolvent `-1/(Nd U+nW)` change.
  All differences up to 2M are sampled: there is no accidental single-hop
  cutoff M inside the finite Sambe matrix.
- P is zero-photon, one electron per site, in tensor spin basis
  `|up up>, |up down>, |down up>, |down down>`. Its Fock embedding contains
  a minus sign for `|down up>` because its site-ordered creators are not
  spin-major ordered. Spins are S=1/2, not Pauli matrices of length one.
- Q excludes only these four states, so R=Q/(0-H0) includes nonzero-photon
  singly occupied states as well as both doublons. All Q denominators must be
  finite and exceed .5 eV in magnitude; all inverse entries must be finite and
  max|H0 R+1|<3e-16 on Q. No eta is introduced.
- Dense products independently evaluate
  `H2=P V R V P` and
  `H4=P V R V R V R V P - {P V R^2 V P,H2}/2`.
  These are unit-t coefficients; physical matrices are `t^2 H2+t^4 H4`.
  This dimer cannot test two simultaneous doublons or non-bipartite H3.
- Exact finite-Sambe eigenvectors are selected by overlap with the preceding
  four-dimensional subspace, starting from P at t=0, never by lowest energy or
  folded proximity to zero. Each step checks principal overlap >.95, P overlap
  minimum singular value >.9, and separation from the complement >.5 eV.
  If `P^dag psi=L s R^dag`, the canonical isometry is `psi R L^dag`;
  its P overlap is positive. Matrix comparison uses this polar gauge, with
  isometry, polar-Hermiticity and eigen-equation residual checks. Boundary
  photon weights are reported; no quasienergy folding is used in selection.

The expansion convention follows the plan's canonical Hermitian Lowdin/SW
contract; that formula alone is not an independent proof. Here exact numerical
block diagonalization and the static analytic gap are separate oracles. There
is no fit to infer H2/H4 from the exact energies. Dense perturbation theory and
finite-Sambe ED are independent of the Rust sparse implementation, **not** an
independent forward-time physical certification: they share the same temporal
projection. The direct time-quadrature projection check rejects the old reversed
index, but is not a propagator. The driven dimer's scalar exchange can be blind
to time reversal; forward evolution must be checked separately (root's electronic
forward-propagation oracle), not inferred from this dimer's unchanged K2/H4.

## Recorded validation and negative controls

Observed environment: Python 3.7.10, NumPy 1.20.3,
`OPENBLAS_NUM_THREADS=1`. Final strict-warning self-check exits 0. Representative
values (small LAPACK/platform-dependent last-bit differences are expected):

| Quantity | Result |
|---|---:|
| Static exact matrix vs `(sqrt(U^2+16t^2)-U)/2` singlet gap, max residual | 4.03e-15 eV |
| Static H2 / H4 coefficients | `4/U`, `-16/U^3` |
| Two-direction K2 (unit t) | 1.1713956489238915 /eV |
| Wrong `4 sum |C_q(+d)|^2/(U-qW)` | 1.4375081861289476 /eV |
| Same wrong formula on reverse direction | .9052831117188354 /eV |
| max difference `|C_n|-|C_-n|` | .1663085000701247 |
| Dense H2 vs two-direction exchange matrix | 2.22e-16 /eV |
| DFT 2048 vs 4096 maximum coefficient difference | 6.94e-17 |
| H4 difference if nonzero-photon singly occupied Q is omitted | .09958561187755954 /eV^3 |
| H4 folded term norm | .8586015510603519 /eV^3 |
| Lowest four raw eigenvectors' total P weight | 2.92e-19 |

The correct exchange is
`K2=2 sum_q (|C_q(+d)|^2+|C_q(-d)|^2)/(U-qW)`.
Equivalently, relabel by destination photon n and use `C_-n` over `U+nW`.
The summed K2 is unchanged by this migration because both link directions are
included; the individually wrong one-direction values exchange places.
Swapping directions leaves it invariant. The wrong expressions average to K2
but individually fail: arbitrary scalar two-frequency phases need both
intermediate doublon directions, not a blind four-times-one-direction rule.
Static fourth order has zero bare chain and nonzero folded contribution, so
omitting folded terms also genuinely fails.

| t | ||H_exact-t^2 H2|| (eV) | ||H_exact-t^2 H2-t^4 H4|| (eV) |
|---|---:|---:|
| .16 | 6.02823e-4 | 2.51349e-5 |
| .08 | 3.88376e-5 | 4.09794e-7 |
| .04 | 2.44649e-6 | 6.47375e-9 |

Halving ratios are 15.52/15.87 and 61.34/63.30, consistent with fourth/sixth
powers, respectively. This scaling supplements, not replaces, static closed
form and independently diagonalized canonical matrices. No fitting is used.
M=8 vs 10 exact **canonical matrix at t=.16** differs by 6.13e-12 eV, compared
with the retained-order residual 2.51e-5 eV; this is a truncation-budget check,
not a claim of absolute 1e-12 accuracy. Unscaled coefficient differences are
9.31e-11 /eV (H2) and 5.90e-9 /eV^3 (H4). M8 boundary weight is 1.45e-11;
M10 at t=.16 is 3.18e-14. Tested bounds are 1e-9 eV for exact matrix change,
1e-8/1e-7 for coefficient changes, and 1e-8 for M8 boundary weight. Smallest
reported sixth-order residual remains well above numerical eigensolve noise
(max 4.39e-14 eV in the positive-time rerun).

## Positive-time migration evidence (task-16)

The old `h_(n-m)+nW` assembly was an internally consistent reversed-time
reference, not a physical forward-time certificate. On 2026-11-18 it was replaced
by the explicit positive-time temporal projection above; no old logs were
rewritten. The strict-warning rerun is retained in
[positive-time JSON](../../target/goal-a-logs/task-16/dimer-positive-time.json)
and its [gzip-tested archive](../../target/goal-a-logs/task-16/dimer-positive-time.json.gz).
Source SHA-256 is
`87d18eb7914be85514b06149d112b3bd3242e8b7f85d857dae14a727cd8bbf8c`.
`OPENBLAS_NUM_THREADS=1 python3 -W error scripts/floquet_spin_reference/dimer_reference.py --self-check`
exits 0. Direct time-quadrature projection residual is `1.62e-15`, while the old
reversed index gives `.47039147`; K2 remains `1.1713956489238915 /eV`.
H2/H24 halving ratios remain `15.52/15.87` and `61.34/63.30`.
This evidence is a refreshed independent implementation check, not an extension
of the dimer's physical coverage to a helicity-sensitive forward propagator.

## Retained failures and handoff

The first run failed on Python 3.7's missing `int.bit_count`; it was replaced
with stdlib binary popcount. The second run passed numerical checks but emitted
a NumPy 1.20.3 divide warning. An isolated probe recorded **98 finite nonzero Q
denominators, min magnitude 1**, and equal finite vector/scalar inverses, yet
vector division warned. The script now computes these tiny inverses with Python
scalar division and checks inverse residuals explicitly; it does not suppress
warnings. The subsequent `python3 -W error` run passes. Failed logs/probes and
all successful JSON remain under `target/goal-a-logs/task-8`, gzip-tested.
They are runtime evidence, not committed golden fixtures.

Source is handed to lead for independent review. No claim of a completed T20,
M1, production generality, or review convergence is made by this script.

---

# Independent Hubbard triangle reference (T20b only)

This is a separate static, finite-state NumPy reference for the triangular
three-site strong-coupling term. It does not alter the dimer oracle, production
Rust, or claim T20/M2 completion.

## Reproduce

```sh
mkdir -p target/goal-a-logs/task-12
OPENBLAS_NUM_THREADS=1 python3 -W error scripts/floquet_spin_reference/triangle_reference.py --self-check > target/goal-a-logs/task-12/self-check-review2.json 2>&1
status=$?
gzip -c target/goal-a-logs/task-12/self-check-review2.json > target/goal-a-logs/task-12/self-check-review2.json.gz
gzip -t target/goal-a-logs/task-12/self-check-review2.json.gz
test "$status" -eq 0
```

The script requires only Python/NumPy, uses the fixed small `U=4`, `Phi=.41`,
`t=.1,.05,.025` workload (20 Fock states, P dimension 8, 11 continuation
steps), and emits parameters, diagnostics, residuals, source hash and software
versions. Failures are retained as traceback JSON; warnings are errors. No
sweeps, broad tolerances, cargo, or fitted coefficients are involved.

## Convention and checks

- Three fermions occupy six **spin-major** modes `(up0,up1,up2,down0,down1,down2)`;
  the complete `C(6,3)=20` basis is explicitly generated and creation/annihilation
  signs are applied at each operator. P consists of one fermion per site. Its
  tensor order is `(s0,s1,s2)` and each Fock sign is computed from creator-order
  inversions, not hard-coded by visual intuition.
- The directed matrix-element labels are `t_01=t_12=t_20=exp(i Phi/3)` and
  reverse links are their conjugates, so the index cycle product
  `t_01 t_12 t_20=exp(i Phi)`. Each `t_ij` multiplies `c_i^dag c_j`, so the
  **electron arrows run oppositely**, `1->0, 2->1, 0->2`. This fixes the flux
  orientation and the reported chirality sign. The dimensionless hopping matrix
  has unit amplitude; t is the external expansion parameter.
- `H0=U*N_d`, `Q=1-P`, and the dense scalar resolvent is `R=Q/(0-H0)`,
  evaluated elementwise with checked finite, nonzero denominators. The only
  virtual states are the 12 non-P states in this canonical N=3 sector.
  Independently evaluated matrix products are `H2=P V R V P` and
  `H3=P V R V R V P`; the physical retained block is `t^2 H2+t^3 H3`.
- An independently assembled 64-element product-Pauli basis uses
  `S=(I,X/2,Y/2,Z/2)`. Every matrix is projected with its diagonal
  Hilbert-Schmidt Gram normalization `Tr(T_a^dag T_b)/8` and reconstructed;
  residuals are reported. Chirality is explicitly
  `chi_012=S0 dot (S1 cross S2)` and its coefficient is independently obtained
  by `Tr(chi H3)/Tr(chi^2)`. Fock-to-tensor signs and the spin-half chi norm are
  checked. The full reported H3 tensor coefficients expose the pair/three-site
  terms, rather than labeling every nonzero term “chirality.”
- The zero-flux cancellation has an independent particle-hole proof, not just
  a numerical assertion. On the fixed `N=L=3` sector, the canonical unitary
  `C c_{i sigma} C^dagger=c^dagger_{i sigma}` maps
  `U n_{i up}n_{i down}` to itself plus `U(1-n_i)`; summed over sites the extra
  term is `U(L-N)=0`. It preserves P and Q. For `i != j`, it maps
  `c_i^dag c_j` to `-c_j^dag c_i`, hence real hopping `V` to `-V`.
  Meanwhile `C S C^dagger=(-Sx,+Sy,-Sz)`, a proper global pi rotation about y.
  Since `R=Q/(0-H0)` is invariant, `C(P V R V R V P)C^dagger=-H3`; but this
  block is SU(2)-invariant, so it is also unchanged by that pi rotation. Thus
  the full `H3` matrix vanishes at Phi=0: no real-cosine cyclic scalar or pair
  term may be retained. The script checks the whole computed matrix norm.
- At nonzero flux, particle-hole maps `V(Phi)` to `-V(-Phi)`, so the
  SU(2)-invariant third-order block is flux-odd. For the stated index product
  and opposite electron-arrow convention, all three starting sites and both
  orientations give the measured chirality coefficient `-24 sin(Phi)/U^2`:
  the two negative resolvent denominators multiply positively and fermionic
  spin-permutation signs fix the remaining sign. The script checks this 24
  factor and sign against the independent Fock trace projection.
- Exact ED diagonalizes the full 20-state `H0+tV`, follows the P8 subspace from
  t=0 by rolling overlap with the previous P8 frame (not repeated P-weight
  selection or lowest-energy labels); it separately reports P-sector overlap.
  It checks minimum continuation overlap and spectral separation, then forms
  `Psi polar(P^dagger Psi)^dagger`. Hermitization first measures and bounds the
  anti-Hermitian residual. The JSON reports isometry, polar positivity/Hermiticity,
  and tracked-subspace eigen residuals. Halving ratios must lie in strict windows
  for `||H_exact-t^2 H2||=O(t^3)` and
  `||H_exact-t^2 H2-t^3 H3||=O(t^4)`; these checks are not a fit.

The exact finite-system test and tensor trace decomposition are separate
validation paths. This is a reference, not by itself production acceptance.

## Disconnected driven dimers: a fourth-order window counterexample

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -W error scripts/floquet_spin_reference/disconnected_reference.py --self-check
```

With strictly band-limited hopping (`H=1`), two disconnected dimers must have
an additive fourth-order spin operator. An intermediate window of `M=1`
incorrectly removes some signed `m=±2` paths without removing the folded term:
`||H4_AB−H4_A⊗I−I⊗H4_B||_F ≈ 0.0522 eV`. A complete window `M=2` reduces the
residual below `6e-17 eV`. Independent time projection anchors the block sign;
no Rust kernel is called. This is a check of window consistency, not of
convergence in the physical drive harmonic cutoff.

## Four-site plaquette: static ED and forward time propagation

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -W error scripts/floquet_spin_reference/square_reference.py --self-check
```

The static check uses NumPy (`--static-only`); SciPy is needed for the driven
ODE. The full 70-state ED tests the fourth-order polar-aligned spin matrix,
including the effects of omitting folded terms or two doublons. The driven
case integrates the time-dependent electron Hamiltonian, aligns the periodic
Floquet modes in the zero-photon gauge, and checks signed chirality. At
`t=0.10 eV`, the positive-time plaquette has `κ012≈+4.967e-6 eV`; reversing
the protocol changes its sign. These values are independent reference results,
not themselves an end-to-end test of the Rust periodic API.

## Noncoplanar loop with a periodic bridge

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -W error scripts/floquet_spin_reference/nonplanar_reference.py
```

The NumPy script recomputes the four-site signed coefficients of the
noncoplanar ring for both signs of the vertical polarization and with
that component absent. The periodic reference adds the y-directed bridge
and its four connected wedges. It derives the labelled finite-q BdG
matrix directly from spinor flip matrix elements of the dense reference
blocks, not from the Rust exchange or magnon reduction.

## Three vertical AA rectangles

```bash
OPENBLAS_NUM_THREADS=1 python3 -B -W error scripts/floquet_spin_reference/aa_rectangle_reference.py
```

The three in-plane bond directions have different projections of the same
3D drive. The script reports each rectangle's two-hop exchanges, the four-hop
diagonal exchange, all three ring coefficients and one oriented chirality.
Explicit cell counting of rectangles, wedges and dimers gives the full
labelled 8×8 BdG matrix directly from local spin flips. The script checks
both time directions at two finite q points and two harmonic cutoffs. The
analytic static AA check is a separate regression.
