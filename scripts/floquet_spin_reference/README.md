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
- `C_n=mean(exp(+in theta) exp(-i phase))`.
  Sambe block `(n,m)=V^(n-m)+delta_nm(U Nd+nW)`, n,m in [-M,M].
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
is no fit to infer H2/H4 from the exact energies.

## Recorded validation and negative controls

Observed environment: Python 3.7.10, NumPy 1.20.3,
`OPENBLAS_NUM_THREADS=1`. Final strict-warning self-check exits 0. Representative
values (small LAPACK/platform-dependent last-bit differences are expected):

| Quantity | Result |
|---|---:|
| Static exact matrix vs `(sqrt(U^2+16t^2)-U)/2` singlet gap, max residual | 4.03e-15 eV |
| Static H2 / H4 coefficients | `4/U`, `-16/U^3` |
| Two-direction K2 (unit t) | 1.1713956489238915 /eV |
| Wrong `4 sum |C_n(+d)|^2/(U+nW)` | .9052831117188355 /eV |
| Same wrong formula on reverse direction | 1.4375081861289474 /eV |
| max difference `|C_n|-|C_-n|` | .1663085000701247 |
| Dense H2 vs two-direction exchange matrix | 2.22e-16 /eV |
| DFT 2048 vs 4096 maximum coefficient difference | 6.94e-17 |
| H4 difference if nonzero-photon singly occupied Q is omitted | .09958561187755954 /eV^3 |
| H4 folded term norm | .8586015510603519 /eV^3 |
| Lowest four raw eigenvectors' total P weight | 2.92e-19 |

The correct exchange is
`K2=2 sum_n (|C_n(+d)|^2+|C_n(-d)|^2)/(U+nW)`.
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
(max 3.98e-14 eV).

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
