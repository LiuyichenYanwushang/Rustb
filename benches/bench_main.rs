//! Performance benchmarks for Rustb hot-path functions.
//!
//! Run with: cargo bench --features intel-mkl-system
//! Filter:    cargo bench --features intel-mkl-system -- gen_ham

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::*;


use num_complex::Complex;
use Rustb::*;

// ── Model builders (not timed) ──────────────────────────────────────────────

/// 2-orbital graphene-like model (small baseline)
fn build_small() -> Model {
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::tb_model(2, lat, orb, false, None).unwrap();
    m.set_onsite(&arr1(&[-0.7, 0.7]), SpinDirection::None);
    let r0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.add_hop(Complex::new(-1.0, 0.0), 0, 1, &r.to_owned(), SpinDirection::None);
    }
    m
}

/// 3×3 supercell of small model (18 orbitals, medium)
fn build_medium() -> Model {
    let li: Complex<f64> = Complex::i();
    let t = Complex::new(2.0, 0.0);
    let t2 = Complex::new(-1.0, 0.0);
    let delta = 0.7;
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::tb_model(2, lat, orb, false, None).unwrap();
    m.set_onsite(&arr1(&[-delta, delta]), SpinDirection::None);
    let r0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.add_hop(t, 0, 1, &r.to_owned(), SpinDirection::None);
    }
    let r0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.add_hop(t2 * li, 0, 0, &r.to_owned(), SpinDirection::None);
    }
    let r0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.add_hop(t2 * li, 1, 1, &r.to_owned(), SpinDirection::None);
    }
    #[allow(non_snake_case)]
    let U = array![[3.0, 0.0], [0.0, 3.0]];
    m.make_supercell(&U).unwrap()
}

/// 2-orbital spinful model (4 states, spin-orbit coupling)
fn build_small_spinful() -> Model {
    let li: Complex<f64> = Complex::i();
    let t = -1.0;
    let soc = 0.06 * t;
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::tb_model(2, lat, orb, true, None).unwrap();
    m.set_onsite(&arr1(&[0.0, 0.0]), SpinDirection::None);
    let r0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.set_hop(t, 0, 1, &r.to_owned(), SpinDirection::None);
    }
    let r0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.set_hop(soc * li, 0, 0, &r.to_owned(), SpinDirection::z);
    }
    let r0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.set_hop(soc * li, 1, 1, &r.to_owned(), SpinDirection::z);
    }
    m
}

// ── Benchmarks ──────────────────────────────────────────────────────────────

fn bench_gen_ham(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_ham");
    let models = [
        ("small(2orb)", build_small()),
        ("medium(18orb)", build_medium()),
        ("spinful(4sta)", build_small_spinful()),
    ];
    let kvec = arr1(&[0.3, 0.5]);

    for (name, model) in models.iter() {
        group.bench_with_input(
            BenchmarkId::new("Lattice", name),
            &(model, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_ham(black_box(kv), Gauge::Lattice)),
        );
        group.bench_with_input(
            BenchmarkId::new("Atom", name),
            &(model, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_ham(black_box(kv), Gauge::Atom)),
        );
    }
    group.finish();
}

fn bench_gen_v(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_v");
    let models = [
        ("small(2orb)", build_small()),
        ("medium(18orb)", build_medium()),
        ("spinful(4sta)", build_small_spinful()),
    ];
    let kvec = arr1(&[0.3, 0.5]);

    for (name, model) in models.iter() {
        group.bench_with_input(
            BenchmarkId::new("Atom_gauge", name),
            &(model, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_v(black_box(kv), Gauge::Atom)),
        );
        group.bench_with_input(
            BenchmarkId::new("Lattice_gauge", name),
            &(model, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_v(black_box(kv), Gauge::Lattice)),
        );
    }
    group.finish();
}

fn bench_solve_onek(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_onek");
    let models = [
        ("small(2orb)", build_small()),
        ("medium(18orb)", build_medium()),
        ("spinful(4sta)", build_small_spinful()),
    ];
    let kvec = arr1(&[0.3, 0.5]);

    for (name, model) in models.iter() {
        group.bench_with_input(
            BenchmarkId::new("eigh", name),
            &(model, &kvec),
            |b, (m, kv)| b.iter(|| m.solve_onek(black_box(kv))),
        );
    }
    group.finish();
}

fn bench_solve_band_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_band_all_parallel");
    let model = build_small();
    let nk = 101;
    let kmesh = arr1(&[nk, nk]);
    let kvec = gen_kmesh(&kmesh).unwrap();
    group.bench_function("small_101x101", |b| {
        b.iter(|| model.solve_band_all_parallel(black_box(&kvec)))
    });
    group.finish();
}

fn bench_berry_curvature_onek(c: &mut Criterion) {
    let mut group = c.benchmark_group("berry_curvature_onek");
    let models = [
        ("small(2orb)", build_small()),
        ("spinful(4sta)", build_small_spinful()),
    ];
    let kvec = arr1(&[0.3, 0.5]);
    let dir1 = arr1(&[1.0, 0.0]);
    let dir2 = arr1(&[0.0, 1.0]);

    for (name, model) in models.iter() {
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(model, &kvec, &dir1, &dir2),
            |b, (m, kv, d1, d2)| {
                b.iter(|| m.berry_curvature_onek(black_box(kv), black_box(d1), black_box(d2), 0.0, 0.0, 0, 1e-3))
            },
        );
        if model.spin {
            group.bench_with_input(
                BenchmarkId::new("spin_z", name),
                &(model, &kvec, &dir1, &dir2),
                |b, (m, kv, d1, d2)| {
                    b.iter(|| m.berry_curvature_onek(black_box(kv), black_box(d1), black_box(d2), 0.0, 0.0, 3, 1e-3))
                },
            );
        }
    }
    group.finish();
}

fn bench_hall_conductivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hall_conductivity");
    let model = build_small();
    let nk = 31;
    let kmesh = arr1(&[nk, nk]);
    let dir1 = arr1(&[1.0, 0.0]);
    let dir2 = arr1(&[0.0, 1.0]);

    group.bench_function("small_31x31", |b| {
        b.iter(|| {
            model
                .Hall_conductivity(
                    black_box(&kmesh),
                    black_box(&dir1),
                    black_box(&dir2),
                    0.0,
                    0.0,
                    0,
                    1e-3,
                )
                .unwrap()
        })
    });
    group.finish();
}

fn bench_dos(c: &mut Criterion) {
    let mut group = c.benchmark_group("dos");
    let model = build_small();
    let kmesh = arr1(&[51, 51]);

    group.bench_function("small_51x51", |b| {
        b.iter(|| model.dos(black_box(&kmesh), -3.0, 3.0, 500, 1e-2).unwrap())
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_gen_ham,
    bench_gen_v,
    bench_solve_onek,
    bench_solve_band_parallel,
    bench_berry_curvature_onek,
    bench_hall_conductivity,
    bench_dos,
);
criterion_main!(benches);
