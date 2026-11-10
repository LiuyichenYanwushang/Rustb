# Floquet effective-model baseline (source base a40bfc4)

Status: **complete, measured**. This benchmark change makes no production Floquet or public-API changes
(`production_floquet_source_changes: false` in [summary.json](target/floquet-baseline-a40bfc4/summary.json.gz)).
Round archives and representative logs are linked below; per-probe logs follow the documented naming pattern. The results section is
the mechanically generated table text inserted verbatim.

## Scope

- Measure the public effective-model construction, `Model::floquet_effective_model`, on the shared
  synthetic Floquet fixture set: 12 cases, one Criterion benchmark group (`floquet_effective`), two
  Rayon thread counts (1 and 4).
- Record three metric families in separate runs per case: Criterion wall-clock medians (both thread counts),
  Rust allocator-observed allocation counts/bytes, and whole-process peak RSS.
- Also record the runtime fixture/structural gates and benchmark-CLI argument behavior, so a future
  change can tell "slower" from "silently did nothing".
- Out of scope: isolated stage timing, isolated E2-correction cost, GEMM-call counts, real user
  models, and any optimization proposal. This Floquet baseline adds no stage probes: it records and
  reports no isolated stage time, no isolated E2-correction cost and no GEMM call count. This WSL
  baseline did not perform perf/valgrind sampling.

Criterion samples are iteration-normalized batches, not 20 individually timed calls or a
per-call latency distribution. Memory medians are marginal summaries across separate processes;
the three-process min/max values are observed ranges, not confidence intervals or universal maxima.

## Fixtures

Shared code: [benches/fixtures/floquet.rs](benches/fixtures/floquet.rs), used by
[benches/bench_main.rs](benches/bench_main.rs#L14-L15) and
[benches/bench_memory.rs](benches/bench_memory.rs#L39-L40). Case matrix is `CASES` at
[benches/fixtures/floquet.rs#L22-L119](benches/fixtures/floquet.rs#L22-L119); model builder at
[#L122-L177](benches/fixtures/floquet.rs#L122-L177); drive at
[#L179-L195](benches/fixtures/floquet.rs#L179-L195); validation at
[#L206-L408](benches/fixtures/floquet.rs#L206-L408).

Design facts (all synthetic, spinless, `Model<false, 2>`, complex blocks):

- Fully coupled: every orbital pair `i != j` has a nonzero complex block at `R = (0,0)`, and every
  `(i, j)` is filled for every `R` in the compact square. Increasing `nsta` therefore grows one
  coupled model; it never pads independent copies of a small model.
- Non-orthogonal lattice `[[1.1, 0.15], [0.0, 0.95]]` with fractional orbital embeddings
  `(i + 1, (7i + 3) mod (nsta + 1)) / (nsta + 1)`, so Peierls link lengths are non-generic.
- Growing onsite span `0.12 (i - (nsta-1)/2)`; hopping scale `0.3 / (sqrt(nsta) (1 + |x| + |y|))`.
- Input support is the compact square: 9 `R` rows for radius 1 (3x3), 25 rows for radius 2 (5x5).
  The same modular position/link rule is reused across `nsta = 4/16/32`, so those three cases share
  a link structure instead of representing a generic worst case. One representative per `R/-R`
  pair is written; `set_hop` supplies the reversed conjugated block.
- Drive: `omega0_ev = 10` eV; `mixed` = coherent `l=1` + `l=2` modes, `fundamental` = `l=1`
  only, `second` = `l=2` only, `zero` = no modes. Amplitudes come from the fixture source; there
  is no user-supplied real-world model or drive for this baseline.

| Case | nsta | input `R` rows | order | harmonic_max | drive |
|---|---|---|---|---|---|
| `dense16_r9/mixed_o0_n2` | 16 | 9 | 0 | 2 | mixed |
| `dense16_r9/mixed_o1_n2` | 16 | 9 | 1 | 2 | mixed |
| `dense16_r9/mixed_o2_n1` | 16 | 9 | 2 | 1 | mixed |
| `dense16_r9/mixed_o2_n2` | 16 | 9 | 2 | 2 | mixed |
| `dense16_r9/mixed_o2_n4` | 16 | 9 | 2 | 4 | mixed |
| `dense16_r9/mixed_o2_n8` | 16 | 9 | 2 | 8 | mixed |
| `dense16_r9/fundamental_o2_n2` | 16 | 9 | 2 | 2 | fundamental |
| `dense16_r9/second_o2_n2` | 16 | 9 | 2 | 2 | second |
| `dense16_r9/zero_o2_n8` | 16 | 9 | 2 | 8 | zero |
| `dense4_r9/mixed_o2_n2` | 4 | 9 | 2 | 2 | mixed |
| `dense32_r9/mixed_o2_n2` | 32 | 9 | 2 | 2 | mixed |
| `dense16_r25/mixed_o2_n2` | 16 | 25 | 2 | 2 | mixed |

## Environment and provenance

- Toolchain: rustc 1.99.1, LLVM 23.1.1, cargo 1.99.1, x86_64, WSL2, Linux 5.15.146.1, Intel
  i9-11900K, 8 visible CPUs, empty `RUSTFLAGS`.
- Backend: exactly one feature, `openblas-system`. OpenBLAS 0.3.20+ds-1, pthread, SkylakeX.
  `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; `RAYON_NUM_THREADS` 1 and 4.
  Never more than one BLAS backend feature.
- `LD_PRELOAD` globally contained MKL on this host. Every resumed gate and timing process
  explicitly unset `LD_PRELOAD` for that process only; no global configuration was changed. The
  backend isolation record is [backend-isolation.txt](target/floquet-baseline-a40bfc4/backend-isolation.txt.gz).
- Source state: commit `a40bfc4` **plus** the benchmark drafts **and** unrelated uncommitted work
  ([CHANGELOG.md](CHANGELOG.md), [SKILLS.md](SKILLS.md),
  [hamiltonian_symmetry.rs](src/hamiltonian_symmetry.rs), [lib.rs](src/lib.rs)) plus untracked
  [symmetry modules](src/hamiltonian_symmetry/). The measured tree is **not** clean, and the benchmark commit SHA alone is **not**
  provenance. Unrelated tracked changes are captured in
  [unrelated-tracked.patch](target/floquet-baseline-a40bfc4/unrelated-tracked.patch.gz).
- Measured source archive (full package source + `benches` + workspace `Cargo.toml`/`Cargo.lock`):
  [measured-source.tar.gz](target/floquet-baseline-a40bfc4/measured-source.tar.gz), sha256
  `2d470af8d4f08a7675d916ee22dd3e967fd09c118d1a749f9bd4521e58ff7776`.
- Binary sha256: `bench_main` = `b1b486ed7e5ed8b9fb9e68b79e2ee0cb29e7fd328c20e8263164c75dc7631245`;
  `bench_memory` = `b9e94496abfec269471007b0bbe732c580e00d0e014c33a27b3eeb1b296f140f`.
- The source archive was captured after both Flash review fixes and formatting. Its three benchmark
  files ([bench_main.rs](benches/bench_main.rs), [bench_memory.rs](benches/bench_memory.rs),
  [fixtures/floquet.rs](benches/fixtures/floquet.rs)) were independently compared byte-for-byte with
  the measured sources. The binary hashes above stayed unchanged. Benchmark code is committed as
  `581a9ea`; the captured dirty source archive, not that commit alone, identifies the measured build.
  This results report was written later and was not a compiled input.
- Supporting records under [target/floquet-baseline-a40bfc4/](target/floquet-baseline-a40bfc4/):
  [source-snapshot.txt](target/floquet-baseline-a40bfc4/source-snapshot.txt.gz),
  [resumed-environment.txt](target/floquet-baseline-a40bfc4/resumed-environment.txt.gz),
  [openblas-version.txt](target/floquet-baseline-a40bfc4/openblas-version.txt.gz),
  [premeasurement-state.txt](target/floquet-baseline-a40bfc4/premeasurement-state.txt.gz),
  [build-benches.log.gz](target/floquet-baseline-a40bfc4/build-benches.log.gz),
  [build-resumed.log.gz](target/floquet-baseline-a40bfc4/build-resumed.log.gz),
  [check-all-targets.log.gz](target/floquet-baseline-a40bfc4/check-all-targets.log.gz),
  [check-resumed.log.gz](target/floquet-baseline-a40bfc4/check-resumed.log.gz),
  [floquet-regression.log.gz](target/floquet-baseline-a40bfc4/floquet-regression.log.gz).
- Machine-readable aggregate: [summary.json](target/floquet-baseline-a40bfc4/summary.json.gz) (scope,
  environment, gates, both timing rounds, repeats, memory); intermediate records
  [timing-summary.json](target/floquet-baseline-a40bfc4/timing-summary.json.gz) and
  [memory-summary.json](target/floquet-baseline-a40bfc4/memory-summary.json.gz).
- Per-probe logs: 72 allocation and 72 RSS fresh-process runs, 3 per case per thread count, named
  `alloc-r{1,4}-<case-slug>-{1,2,3}.log.gz` and `rss-r{1,4}-<case-slug>-{1,2,3}.log.gz` in the same
  directory, e.g.
  [alloc-r1-dense16_r9_mixed_o2_n2-1.log.gz](target/floquet-baseline-a40bfc4/alloc-r1-dense16_r9_mixed_o2_n2-1.log.gz)
  and
  [rss-r4-dense16_r9_mixed_o2_n2-1.log.gz](target/floquet-baseline-a40bfc4/rss-r4-dense16_r9_mixed_o2_n2-1.log.gz).
  Round archives and round logs are listed under Reproduction.

Artifacts are available in this workspace and the [evidence bundle](target/floquet-baseline-a40bfc4-evidence.tar.gz);
the documentation commit does not distribute the target directory. Offline rebuilding also needs
matching dependency caches and system OpenBLAS. This source capture is not a hermetic or guaranteed
bit-identical rebuild recipe.

## Metrics

**Criterion medians (r1, r4).** Timed region is exactly the whole public construction plus destruction
of the returned model:
`b.iter(|| black_box(model.floquet_effective_model(&light, Some(&options)).unwrap()))` at
[benches/bench_main.rs#L890-L910](benches/bench_main.rs#L890-L910). Model, drive and options are built
once outside the loop. All 12 cases ran at both Rayon thread counts with `group.sample_size(20)` fixed
([#L892](benches/bench_main.rs#L892)), `--noplot`, and a separate `CRITERION_HOME` per round. The
initial rounds used warm-up 1 s and a 3 s measurement target; the three key-case repeats reused the
same binaries and 20 samples with a 5 s target in reversed thread order (4 then 1). Each reported value
is the median point estimate recorded in that round's `estimates.json`: the `mean` and `slope` lines
Criterion prints are different statistics and are not the median, so they are not substituted here.
`validate_case` runs once per case before its benchmark and is **not** timed
([#L897](benches/bench_main.rs#L897)). Each measured case therefore starts after validation has
already exercised the same construction path (allocator/page-cache warm), not from a cold process.
Only whole-construction time is observed; this baseline adds no stage probe splitting out the E2 term,
the harmonic coefficients, or any GEMM work, and no GEMM call counts are recorded in
[summary.json](target/floquet-baseline-a40bfc4/summary.json.gz) or reported here.

**Allocation counts/bytes.** Prebuilt `bench_memory` with one exact case id
([benches/bench_memory.rs#L144-L213](benches/bench_memory.rs#L144-L213)): three independent processes
per case and thread count (72 probes); in each, one warmup construction is dropped, then exactly one
construction is counted. `allocations` and `allocated_bytes` are Rust
allocator requests (a reallocation counts its full new size); `peak_live_bytes` is peak net growth
since counting began. Native BLAS allocations are invisible to this allocator, and the counted output
is not destroyed inside the window, so bytes at the end remain live. Its `time_ms` includes counter
overhead and must **not** be compared with the Criterion medians.

**Peak RSS.** Prebuilt `bench_memory --floquet-rss <id>` under `/usr/bin/time -v`: a fresh process,
one construction, no warmup and no separate fixture validation; three independent processes per case
and thread count (72 probes). The value is whole-process peak RSS, including input build, Rayon pool
priming, backend loading and exit, and is not incremental memory. No "RSS minus allocator" subtraction
is performed, so it cannot be used to infer BLAS memory.

**Host state.** The first round was interrupted before sampling: its preparation (fixture authoring and
compilation) ran while the host was heavily loaded, and it produced **no timing samples**, so nothing from
it is reported as a result (preparation record:
[preparation-paused.tar.gz](target/floquet-baseline-a40bfc4/preparation-paused.tar.gz)). Formal rounds
ran only after the resume, on the quieter host: resumed prechecks measured 77-99% idle CPU, and every
build, timing, memory and repeat command ran with `env -u LD_PRELOAD` for that process only. Load
before/after each round is stored
([load-before-r1.log.gz](target/floquet-baseline-a40bfc4/load-before-r1.log.gz),
[load-after-r1.log.gz](target/floquet-baseline-a40bfc4/load-after-r1.log.gz),
[load-before-r4.log.gz](target/floquet-baseline-a40bfc4/load-before-r4.log.gz),
[load-after-r4.log.gz](target/floquet-baseline-a40bfc4/load-after-r4.log.gz),
[load-repeat-r4-before.log.gz](target/floquet-baseline-a40bfc4/load-repeat-r4-before.log.gz),
[load-repeat-r4-after.log.gz](target/floquet-baseline-a40bfc4/load-repeat-r4-after.log.gz),
[load-repeat-r1-before.log.gz](target/floquet-baseline-a40bfc4/load-repeat-r1-before.log.gz),
[load-repeat-r1-after.log.gz](target/floquet-baseline-a40bfc4/load-repeat-r1-after.log.gz)), so the
rounds are auditable. No isolation is assumed. The r1/r4 medians are two thread-count observations;
they are **not** a parallel-speedup claim, and the measured repeat drift (up to 6.08%) is retained
rather than averaged away.

**Structural gates (not oracles).** All 12 `validate_case` checks passed at Rayon 1 and Rayon 4
(24 checks), and 114 release Floquet tests passed. `validate_case` is a runtime anti-noop/structure
check: input `Model::validate`, state counts, a nonzero block on every interorbital link, the exact
unique input-square `hamR` set and row count, output validate/state count/lattice/orbital preservation,
finiteness, exact Hermiticity `H(R) = H(-R)^dagger` on input and output, exact origin-first
lexicographic Minkowski support, a nonzero order-2 vs order-1 Bloch correction, and the zero-drive /
pure-`l=2` controls. These confirm the fixture and the call did real work; they are **not** an
independent derivation of the physics. Independent oracles remain the existing test suite
([src/floquet/tests.rs](src/floquet/tests.rs)). The Criterion `--bench floquet_effective` filter does
not gate this: the fixture `case.model()`/`case.light()`/`case.options()` calls and
`validate_case(case)` run in the loop before `group.bench_function`
([#L893-L897](benches/bench_main.rs#L893-L897)), so all 12 fixtures are built and validated in every
timing round even when the filter is used, and each timed case carries untimed thermal and page-cache
history from the earlier cases' models and validations in the same process. Per-case round data is
staged in [cases-r1.tsv.gz](target/floquet-baseline-a40bfc4/cases-r1.tsv.gz) and
[cases-r4.tsv.gz](target/floquet-baseline-a40bfc4/cases-r4.tsv.gz); this report records only the gate
result (PASS), not the underlying numbers.

**CLI behavior.** The memory harness strips at most one `--bench`
([benches/bench_memory.rs#L215-L251](benches/bench_memory.rs#L215-L251)), the flag Cargo supplies even
for a `harness = false` target: the parser allows **zero or one** `--bench` and rejects a
**duplicate** one, because the leftover flag matches no mode. The logged invocations used one `--bench`,
which was accepted. Unknown case id, an extra argument, and a duplicate `--bench` each exit 2. Logs:
[cli-cargo-flag.log.gz](target/floquet-baseline-a40bfc4/cli-cargo-flag.log.gz),
[cli-unknown.log.gz](target/floquet-baseline-a40bfc4/cli-unknown.log.gz),
[cli-extra.log.gz](target/floquet-baseline-a40bfc4/cli-extra.log.gz),
[cli-duplicate.log.gz](target/floquet-baseline-a40bfc4/cli-duplicate.log.gz).

## Reproduction

Build once with `env -u LD_PRELOAD`, never inside a timing round (every resumed command, including this
build, ran with `LD_PRELOAD` unset for that process only):

```sh
env -u LD_PRELOAD CARGO_TARGET_DIR=$PWD/target/dsh-doc cargo bench --offline \
  --features openblas-system --bench bench_main --bench bench_memory --no-run --message-format=json
```

Take the executable paths from the `compiler-artifact` messages. Actual prebuilt paths:
`target/dsh-doc/release/deps/bench_main-2d18948e9484e877` and
`target/dsh-doc/release/deps/bench_memory-8642d385734d7e92`.

Criterion round, with a separate `CRITERION_HOME` per round (repeat with `RAYON_NUM_THREADS=4`, `-r4`,
`criterion-r4` for the second round):

```sh
env -u LD_PRELOAD RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CRITERION_HOME=$PWD/target/floquet-baseline-a40bfc4/criterion-r1 \
  target/dsh-doc/release/deps/bench_main-2d18948e9484e877 --bench floquet_effective \
  --save-baseline floquet-before-r1 --warm-up-time 1 --measurement-time 3 --noplot
```

Key-case repeats re-ran only `dense16_r9/mixed_o2_n2`, `dense16_r9/mixed_o2_n8` and
`dense32_r9/mixed_o2_n2` with a 5 s measurement target, Rayon 4 first and Rayon 1 second, using
`CRITERION_HOME=$PWD/target/floquet-baseline-a40bfc4/repeat-r4` and `.../repeat-r1`. Exact invocations
are saved in [commands.json](target/floquet-baseline-a40bfc4/commands.json.gz); outputs are in [repeat-r4.log.gz](target/floquet-baseline-a40bfc4/repeat-r4.log.gz) and
[repeat-r1.log.gz](target/floquet-baseline-a40bfc4/repeat-r1.log.gz).

Allocation mode (exact case id, e.g. `dense16_r9/mixed_o2_n2`):

```sh
env -u LD_PRELOAD RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  target/dsh-doc/release/deps/bench_memory-8642d385734d7e92 --bench \
  --floquet dense16_r9/mixed_o2_n2
```

RSS mode (fresh process, one construction):

```sh
env -u LD_PRELOAD RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/time -v target/dsh-doc/release/deps/bench_memory-8642d385734d7e92 --bench \
  --floquet-rss dense16_r9/mixed_o2_n2
```

Gates: `--floquet-check` for all 12 cases; invalid CLI forms must exit 2. Every round is archived:
[criterion-r1.tar.gz](target/floquet-baseline-a40bfc4/criterion-r1.tar.gz) with
[timing-r1.log.gz](target/floquet-baseline-a40bfc4/timing-r1.log.gz), and
[criterion-r4.tar.gz](target/floquet-baseline-a40bfc4/criterion-r4.tar.gz) with
[timing-r4.log.gz](target/floquet-baseline-a40bfc4/timing-r4.log.gz); the repeats as
[repeat-r4.tar.gz](target/floquet-baseline-a40bfc4/repeat-r4.tar.gz) with
[repeat-r4.log.gz](target/floquet-baseline-a40bfc4/repeat-r4.log.gz) and
[repeat-r1.tar.gz](target/floquet-baseline-a40bfc4/repeat-r1.tar.gz) with
[repeat-r1.log.gz](target/floquet-baseline-a40bfc4/repeat-r1.log.gz). No round remains pending.
The actual Cargo `--floquet-check` dispatch and legacy no-argument response mode also passed;
[Cargo smoke log](target/floquet-baseline-a40bfc4/cli-real-cargo-check.log.gz),
[legacy smoke log](target/floquet-baseline-a40bfc4/cli-legacy-response.log.gz),
[format check](target/floquet-baseline-a40bfc4/format-check.log.gz).

Raw estimate directories can be restored from their round archives before a future comparison:

```sh
tar -xzf target/floquet-baseline-a40bfc4/criterion-r1.tar.gz -C target/floquet-baseline-a40bfc4
# Rebuild the candidate, keep the same environment, then use
# --baseline floquet-before-r1 with CRITERION_HOME pointing at criterion-r1.
```

## Measured results

### Initial Criterion rounds

Milliseconds: median estimate [95% bootstrap CI], 20 samples per case. These are warmed full public construction plus output destruction.

| Case | Rayon 1 (ms) | Rayon 4 (ms) |
|---|---:|---:|
| `dense16_r9/mixed_o0_n2` | 2.977 [2.959, 3.009] | 1.416 [1.386, 1.457] |
| `dense16_r9/mixed_o1_n2` | 3.937 [3.890, 4.002] | 2.066 [1.978, 2.179] |
| `dense16_r9/mixed_o2_n1` | 5.946 [5.897, 6.043] | 3.707 [3.597, 3.862] |
| `dense16_r9/mixed_o2_n2` | 12.833 [12.733, 13.033] | 7.593 [7.469, 7.732] |
| `dense16_r9/mixed_o2_n4` | 38.185 [37.559, 38.665] | 22.423 [22.179, 22.517] |
| `dense16_r9/mixed_o2_n8` | 144.766 [138.447, 149.853] | 79.194 [78.811, 80.518] |
| `dense16_r9/fundamental_o2_n2` | 10.731 [10.408, 10.864] | 6.793 [6.741, 6.859] |
| `dense16_r9/second_o2_n2` | 3.976 [3.948, 4.066] | 3.085 [3.056, 3.111] |
| `dense16_r9/zero_o2_n8` | 0.134 [0.132, 0.137] | 0.139 [0.136, 0.142] |
| `dense4_r9/mixed_o2_n2` | 3.105 [3.004, 3.217] | 1.513 [1.457, 1.537] |
| `dense32_r9/mixed_o2_n2` | 65.540 [63.279, 69.030] | 22.934 [22.744, 23.128] |
| `dense16_r25/mixed_o2_n2` | 81.576 [80.353, 85.174] | 50.054 [49.055, 50.626] |

### Independent key-case repeats

Same binaries, 20 samples, 1 s warmup and 5 s target; thread order reversed (4 then 1). Initial rounds used a 3 s target. The table retains drift rather than replacing initial values with the faster observation.

| Case | Rayon | Initial median (ms) | Repeat median (ms) | Drift |
|---|---:|---:|---:|---:|
| `dense16_r9/mixed_o2_n2` | 4 | 7.593 | 7.635 | +0.55% |
| `dense16_r9/mixed_o2_n8` | 4 | 79.194 | 82.604 | +4.31% |
| `dense32_r9/mixed_o2_n2` | 4 | 22.934 | 24.116 | +5.16% |
| `dense16_r9/mixed_o2_n2` | 1 | 12.833 | 12.654 | -1.39% |
| `dense16_r9/mixed_o2_n8` | 1 | 144.766 | 137.323 | -5.14% |
| `dense32_r9/mixed_o2_n2` | 1 | 65.540 | 61.557 | -6.08% |

Observed repeat drift reached 6.08%; within-run confidence intervals do not capture all between-run variation. Future small performance changes need matched repeated before/after measurements, not comparison against a single old point.

### Allocation and RSS observations

Three independent allocation processes and three independent RSS processes per case/thread setting. Counts and cumulative requested bytes are medians; peak columns show median [min, max], in MiB (2^20 bytes). Rust peak is net allocator growth, while RSS is the whole fresh-process high-water mark: these are different metrics and are not subtracted. The allocation timer is not a Criterion timing result.

| Case | Rayon | Allocations | Requested MiB | Rust peak MiB [min, max] | Process RSS MiB [min, max] |
|---|---:|---:|---:|---:|---:|
| `dense16_r9/mixed_o0_n2` | 1 | 20028 | 3.630 | 0.333 [0.333, 0.333] | 6.621 [6.617, 6.668] |
| `dense16_r9/mixed_o1_n2` | 1 | 20829 | 5.859 | 0.574 [0.574, 0.574] | 8.668 [8.598, 8.703] |
| `dense16_r9/mixed_o2_n1` | 1 | 22854 | 7.481 | 0.877 [0.877, 0.877] | 9.141 [9.129, 9.203] |
| `dense16_r9/mixed_o2_n2` | 1 | 28423 | 11.957 | 1.017 [1.017, 1.017] | 9.352 [9.238, 9.480] |
| `dense16_r9/mixed_o2_n4` | 1 | 50218 | 28.363 | 1.298 [1.298, 1.298] | 10.016 [9.953, 10.020] |
| `dense16_r9/mixed_o2_n8` | 1 | 136431 | 90.997 | 1.933 [1.933, 1.933] | 10.859 [10.809, 10.859] |
| `dense16_r9/fundamental_o2_n2` | 1 | 21619 | 10.207 | 1.017 [1.017, 1.017] | 9.352 [9.242, 9.438] |
| `dense16_r9/second_o2_n2` | 1 | 16050 | 5.931 | 1.017 [1.017, 1.017] | 9.289 [9.281, 9.438] |
| `dense16_r9/zero_o2_n8` | 1 | 829 | 1.902 | 0.564 [0.564, 0.564] | 6.723 [6.652, 6.887] |
| `dense4_r9/mixed_o2_n2` | 1 | 20402 | 6.225 | 0.091 [0.091, 0.091] | 6.512 [6.434, 6.566] |
| `dense32_r9/mixed_o2_n2` | 1 | 71620 | 41.842 | 3.982 [3.982, 3.982] | 13.461 [13.410, 13.520] |
| `dense16_r25/mixed_o2_n2` | 1 | 107220 | 37.173 | 3.276 [3.276, 3.276] | 12.750 [12.723, 12.750] |
| `dense16_r9/mixed_o0_n2` | 4 | 20028 | 3.630 | 0.340 [0.340, 0.340] | 6.797 [6.754, 6.902] |
| `dense16_r9/mixed_o1_n2` | 4 | 20829 | 5.859 | 0.574 [0.574, 0.574] | 8.926 [8.891, 9.008] |
| `dense16_r9/mixed_o2_n1` | 4 | 23371 | 9.264 | 2.071 [2.011, 2.072] | 16.938 [16.828, 17.055] |
| `dense16_r9/mixed_o2_n2` | 4 | 30434 | 18.891 | 3.393 [3.373, 3.694] | 18.711 [18.703, 18.789] |
| `dense16_r9/mixed_o2_n4` | 4 | 57749 | 54.309 | 3.722 [3.719, 3.762] | 18.969 [18.965, 19.191] |
| `dense16_r9/mixed_o2_n8` | 4 | 166042 | 192.989 | 4.214 [4.140, 4.457] | 19.813 [19.598, 21.563] |
| `dense16_r9/fundamental_o2_n2` | 4 | 23630 | 17.141 | 3.398 [3.218, 3.500] | 20.625 [18.785, 20.629] |
| `dense16_r9/second_o2_n2` | 4 | 16567 | 7.715 | 2.250 [2.213, 2.293] | 17.043 [16.945, 17.152] |
| `dense16_r9/zero_o2_n8` | 4 | 829 | 1.902 | 0.564 [0.564, 0.564] | 7.129 [6.906, 7.156] |
| `dense4_r9/mixed_o2_n2` | 4 | 22413 | 6.937 | 0.297 [0.281, 0.300] | 7.148 [7.121, 7.211] |
| `dense32_r9/mixed_o2_n2` | 4 | 73631 | 68.686 | 12.200 [11.917, 12.784] | 31.840 [29.176, 31.918] |
| `dense16_r25/mixed_o2_n2` | 4 | 118577 | 77.318 | 10.777 [10.327, 12.283] | 27.305 [26.488, 28.938] |
