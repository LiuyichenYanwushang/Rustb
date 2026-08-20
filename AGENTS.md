# Repository Guidelines

## Project Structure & Module Organization

Rustb is a Rust 2024 library crate for tight-binding physics calculations. Core source lives in `src/`; `src/lib.rs` re-exports the public API and contains integration-style tests. Key modules include `model.rs` for `Model<SPIN, DIM, R>`, `model_build.rs` for constructors and hopping setup, `model_physics.rs` for DOS/topology routines, `conductivity.rs` and `quantum_geometry.rs` for response calculations, `surfgreen.rs` for surface Green functions, and `wannier90.rs` for Wannier90 input. Examples are declared in `Cargo.toml` and stored under `examples/<name>/main.rs`. Test outputs are under `tests/`; benchmarks are in `benches/`; rustdoc assets are in `docs/` and `docs-header.html`.

## Build, Test, and Development Commands

- `cargo build --features openblas-system`: compile the crate with an explicit BLAS/LAPACK backend.
- `cargo build --features intel-mkl-static` or `cargo build --features openblas-static`: build with a specific BLAS/LAPACK backend.
- `cargo test --features openblas-system`: run unit and integration-style tests. Some tests emit plot/data files through gnuplot.
- `cargo test <name> --features openblas-system`: run a focused test, for example `cargo test gen_kmesh --features openblas-system`.
- `cargo runexample graphene`: use the local Cargo alias for `cargo run --features intel-mkl-system --example graphene` (alias hardcodes `intel-mkl-system`).
- `cargo bench --features openblas-system`: run Criterion benchmarks in `benches/`.
- `cargo mydoc`: build and open crate docs without dependencies (alias hardcodes `--features openblas-system`).

The `mydoc`, `testall`, and `runexample` aliases hardcode one backend. Cargo
merges extra `--features` instead of replacing alias-injected ones, so use the
underlying `cargo doc`/`cargo test`/`cargo run` command directly when a
different backend is needed.

## Coding Style & Naming Conventions

Use `cargo fmt` before committing and keep code idiomatic Rust with four-space indentation. Follow existing public API patterns: const-generic models are written as `Model::<false, 2>` or `Model<SPIN, DIM, HasRMatrix>`, and trait impls generally use `impl<const SPIN: bool, const DIM: usize, R: RMatrixData>`. Prefer `TbError`/`Result` over panics in library code. Keep module names lowercase except for existing compatibility files such as `SKmodel.rs`.

## Testing Guidelines

Place focused unit tests near the implementation in `#[cfg(test)] mod tests`, using names like `test_gen_kmesh` or `test_build_model_with_f_orbital`. Use `cargo test --features openblas-system -- --nocapture` when inspecting numerical or plotting output. If a change affects BLAS/LAPACK behavior, also run `cargo testall` (alias passes `intel-mkl-system`) when an Intel MKL system backend is available.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit prefixes such as `fix:`, `feat:`, `refactor:`, `bench:`, and `docs:`. Keep commit subjects imperative and scoped to one change. Pull requests should describe the physics/API behavior changed, list commands run, note selected backend features, and include plots or regenerated artifacts when output files under `tests/` or `examples/` are intentionally updated.

## Agent-Specific Notes

Do not copy old API patterns from `README.md`; `CLAUDE.md` documents the newer const-generic `Model<SPIN, DIM, R>` design. Avoid broad rewrites of generated test outputs unless the requested change requires refreshing them.
