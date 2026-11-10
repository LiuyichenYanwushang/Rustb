# Floquet order-2 outer-convolution regrouping

Algorithm commit: `bdc2483`; before source base: `f6498d1`. This is the first optimization after the [baseline](benches/FLOQUET_BASELINE.md), not a new public API.

## Change and numerical contract

[Implementation](src/floquet.rs) groups only the mixed second-order family, with `I={-N,...,-1,1,...,N}`:

```math
Y_p = \sum_{m\in I,\ m\ne p} [H_{p-m},H_m]/m,\qquad
B = W^{-2}\sum_{p\in I}[H_{-p},Y_p]/(3p).
```

- The first nested family, first-order sign, genuine harmonics through ±2N, full S/S²/S³ support and final Hermitian projection are unchanged. No intermediate Hermitian projection or positive-harmonic half-domain assumption is introduced.
- One Y map per live fold is streamed, not all inner terms. For the instrumented ordinary-range grouped-path N=2 fixture, single-thread [regression counters](src/floquet/tests.rs#L5791) observe 36 original kernel entries versus at most 28. This is not a universal bound for fallback/replay inputs. This is not an actual GEMM count, total allocation count or timing result.
- A single active signed pair stays on the old path: mixed N=1 and pure second-carrier N=2 have no outer reuse.
- The old termwise closure remains the fallback and internal test reference. Conservative total-L1 bounds retain normal finite raw/scaled headroom; the smallest original coefficient `inverse_W²/(3N²)` must be normal. Nonzero cache components below `eps⁴=2^-208` select the old arithmetic, **not pruning**. With binary64 component/reciprocal grids, the grouped raw-outer quantum is at least 2^-895, above the normal limit 2^-1022; this blocks underflow before later frequency amplification. Defensive per-p checks also replay termwise if a weighted inner or accumulated Y is unsafe.
- Regrouping changes floating-point operation order. Range guards are **not** a uniform relative-to-final-result error theorem for arbitrarily ill-conditioned cancellation, nor a bitwise determinism promise.

## Validation and environment

- 122 release Floquet tests pass with OpenBLAS and with sequential LP64 MKL: original 114 tests unchanged, plus eight focused algebra/range/structural tests. They cover literal k-space and forced-old references, N=0..3, storage-valid non-Hermitian input, 1/4-thread agreement, rescaled/zero/cancelled fields, Inf/zero reciprocal squares, the amplified-outer-underflow counterexample and exact floor neighbor.
- All-target OpenBLAS check, format check and all 12 benchmark-fixture checks at each thread count pass. Checks are evidence, not a new independent physics proof.
- Timing backend: OpenBLAS 0.3.20 pthread/SkylakeX, BLAS/OMP/MKL thread variables=1, Rayon=1 or4; rustc1.99.1, i9-11900K,8 visible WSL2 CPUs. Timing processes unset host MKL `LD_PRELOAD`; no global configuration change.
- MKL initially linked only the LP64 interface and failed to load `mkl_blas_dgemm`. The successful MKL-only validation process explicitly preloaded the matching core and sequential components; no OpenBLAS feature was combined with MKL. [MKL loader diagnosis](target/floquet-grouping-f6498d1/mkl-linkage.txt.gz), [successful MKL gate](target/floquet-grouping-f6498d1/floquet-mkl-seq.log.gz).

## Main results: adjacent ABBA confirmation

At each Rayon count, fixed old/new executables ran old,new,new,old, serially without overlapping own builds/tests/measurements. Each case in each round used20 iteration-normalized Criterion samples,1s warmup and3s target, including the full public construction **and output destruction**. Setup/validation was untimed but warmed caches/thermal history. Each below value is the geometric mean of its two per-round median estimates, not an individual-call latency or a cross-round confidence interval. Ratio=old/new; percentage is elapsed-time change.

| Case | Rayon | Old ms | New ms | Ratio | Time change |
|---|---:|---:|---:|---:|---:|
| `dense16_r9/mixed_o1_n2` | 1 | 3.687 | 3.645 | 1.01× | -1.2% |
| `dense16_r9/mixed_o2_n2` | 1 | 11.710 | 9.218 | 1.27× | -21.3% |
| `dense16_r9/mixed_o2_n8` | 1 | 127.694 | 53.522 | 2.39× | -58.1% |
| `dense32_r9/mixed_o2_n2` | 1 | 56.975 | 41.562 | 1.37× | -27.1% |
| `dense16_r25/mixed_o2_n2` | 1 | 77.500 | 53.058 | 1.46× | -31.5% |
| `dense16_r9/mixed_o1_n2` | 4 | 1.846 | 1.890 | 0.98× | +2.4% |
| `dense16_r9/mixed_o2_n2` | 4 | 7.080 | 5.615 | 1.26× | -20.7% |
| `dense16_r9/mixed_o2_n8` | 4 | 77.165 | 31.100 | 2.48× | -59.7% |
| `dense32_r9/mixed_o2_n2` | 4 | 22.683 | 17.349 | 1.31× | -23.5% |
| `dense16_r25/mixed_o2_n2` | 4 | 48.942 | 33.690 | 1.45× | -31.2% |

The order-1 control changed −1.2% / +2.4% rather than showing a comparable gain. Host snapshots and ABBA mitigate, not eliminate, external contention/drift. These are synthetic working-tree measurements, not a causal decomposition of the E2 phase or a guarantee for the user’s absent slow model.

## Initial full matrix (secondary evidence)

Old rounds preceded implementation/builds; new rounds came later in reversed thread order. Therefore the adjacent ABBA confirmation above, not these time-separated ratios alone, supports the headline. Values are per-run Criterion median estimates.

| Case | Rayon1 old→new ms | Rayon4 old→new ms |
|---|---:|---:|
| `dense16_r9/mixed_o0_n2` | 2.937 → 2.981 | 1.347 → 1.356 |
| `dense16_r9/mixed_o1_n2` | 3.891 → 4.148 | 1.865 → 1.857 |
| `dense16_r9/mixed_o2_n1` | 5.584 → 5.639 | 3.625 → 3.529 |
| `dense16_r9/mixed_o2_n2` | 12.112 → 9.549 | 7.205 → 5.534 |
| `dense16_r9/mixed_o2_n4` | 35.995 → 20.275 | 21.829 → 11.663 |
| `dense16_r9/mixed_o2_n8` | 128.860 → 55.517 | 82.059 → 30.930 |
| `dense16_r9/fundamental_o2_n2` | 10.049 → 7.286 | 6.742 → 4.978 |
| `dense16_r9/second_o2_n2` | 3.859 → 3.987 | 3.050 → 3.058 |
| `dense16_r9/zero_o2_n8` | 0.134 → 0.133 | 0.135 → 0.137 |
| `dense4_r9/mixed_o2_n2` | 2.862 → 2.112 | 1.422 → 1.200 |
| `dense32_r9/mixed_o2_n2` | 58.532 → 42.648 | 22.297 → 17.574 |
| `dense16_r25/mixed_o2_n2` | 78.908 → 54.811 | 49.530 → 34.285 |

## Memory tradeoff

Three allocation processes and three separately fresh RSS processes per case/thread/version:144 allocation +144 RSS probes. Table shows marginal medians in MiB. Requested bytes are cumulative Rust allocator requests; Rust peak is warm net allocator growth. RSS is the separate whole-process high-water mark, including inputs, pool/backend startup, output and exit. Neither is inferred from or subtracted from the other. Allocation elapsed (instrumented, excludes output destruction) is not a Criterion timing. Full observed min/max ranges and counts are in the [summary](target/floquet-grouping-f6498d1/summary.json.gz).

| Case | Rayon | Requested MiB old→new | Rust peak MiB old→new | RSS MiB old→new |
|---|---:|---:|---:|---:|
| `dense16_r9/mixed_o2_n2` | 1 | 11.957 → 10.729 | 1.017 → 1.018 | 9.188 → 9.348 |
| `dense16_r9/mixed_o2_n2` | 4 | 18.891 → 14.495 | 3.229 → 3.174 | 18.777 → 18.645 |
| `dense16_r9/mixed_o2_n8` | 1 | 90.997 → 46.612 | 1.933 → 1.933 | 10.738 → 10.730 |
| `dense16_r9/mixed_o2_n8` | 4 | 192.989 → 59.883 | 4.229 → 4.016 | 19.520 → 19.625 |

N=8/Rayon4 cumulative requested bytes fell from202,363,475 to62,791,891 (about69%), while measured RSS stayed near20MiB. This is **not** a69% peak-memory reduction. Individual peaks can increase slightly; large-model peak behavior still needs the actual input.

## Provenance and reproduction

A content-hash comparison of 69 archived compiled-input files (production sources, manifests/lock,
and benchmark code) found differences only in the implementation and test files under Floquet; the
added tests are not compiled into the benchmark executables. [Snapshot comparison](target/floquet-grouping-f6498d1/source-comparison.json.gz).
These ratios compare frozen before/after working trees and are not a clean-commit-only reproducibility
guarantee. The controlled source comparison does not eliminate runtime/compiler/layout confounding
or isolate the E2 phase.

- Before executable hashes are the preserved baseline’s: timing `b1b486ed7e5ed8b9fb9e68b79e2ee0cb29e7fd328c20e8263164c75dc7631245`, memory `b9e94496abfec269471007b0bbe732c580e00d0e014c33a27b3eeb1b296f140f`. All four hashes and the principal measurement/validation command chains are in the [summary](target/floquet-grouping-f6498d1/summary.json.gz). Fixed snapshots prevent rebuilds or historical Criterion paths from replacing either side.
- [Measured source snapshot](target/floquet-grouping-f6498d1/after-source.tar.gz) SHA256: `50dfb322338fabb8409a8d1efc6580e92fea089a4a3528976a19924df6e8b62a`; [algorithm patch](target/floquet-grouping-f6498d1/algorithm.patch.gz). The measured tree also contained unrelated uncommitted source changes, so a commit SHA alone is not provenance. Later unrelated [manifest](Cargo.toml) edits (optional cryspglib path) were not submitted here and do not identify the frozen executables. cryspglib was not enabled for these measurements.
- Original [fixture definitions](benches/fixtures/floquet.rs), [Criterion group](benches/bench_main.rs) and [allocation/RSS modes](benches/bench_memory.rs) are unchanged. The models are reproducible dense coupled synthetic examples, not the user’s actual model or generic worst-case link/cache work.
- Raw logs are compressed on every round; all four full-matrix trees (12 cases each) and eight ABBA Criterion trees (five cases each) have separate round archives. Locally retained artifacts are not included in the documentation commit; offline rebuilding also needs matching toolchain/dependency caches and system libraries and is not guaranteed bit-identical.


Example OpenBLAS gate (exact timing/memory invocations and snapshot paths are recorded in the summary):

```sh
env -u LD_PRELOAD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=4 \
  CARGO_TARGET_DIR="$PWD/target/dsh-doc" \
  cargo test --release --offline --features openblas-system --lib floquet::tests:: -- --test-threads=1
```

For MKL validation use only `--features intel-mkl-system`, matching `MKLROOT`/library path and process-local `LD_PRELOAD=$MKLROOT/lib/libmkl_core.so:$MKLROOT/lib/libmkl_sequential.so`; all performance observations above remain OpenBLAS-only.
