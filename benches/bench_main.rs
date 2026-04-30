//! Performance benchmarks for Rustb hot-path functions.
//!
//! Run with: cargo bench --features intel-mkl-system
//! Filter:    cargo bench --features intel-mkl-system -- gen_ham

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use ndarray::*;

use Rustb::*;
use num_complex::Complex;

// ── Model builders (not timed) ──────────────────────────────────────────────

/// 2-orbital graphene-like model (small baseline)
fn build_small() -> Model<false> {
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::<false>::tb_model(2, lat, orb, None).unwrap();
    m.set_onsite(&arr1(&[-0.7, 0.7]), SpinDirection::None);
    let r0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.add_hop(
            Complex::new(-1.0, 0.0),
            0,
            1,
            &r.to_owned(),
            SpinDirection::None,
        );
    }
    m
}

/// 3×3 supercell of small model (18 orbitals, medium)
fn build_medium() -> Model<false> {
    let li: Complex<f64> = Complex::i();
    let t = Complex::new(2.0, 0.0);
    let t2 = Complex::new(-1.0, 0.0);
    let delta = 0.7;
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::<false>::tb_model(2, lat, orb, None).unwrap();
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
fn build_small_spinful() -> Model<true> {
    let li: Complex<f64> = Complex::i();
    let t = -1.0;
    let soc = 0.06 * t;
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::<true>::tb_model(2, lat, orb, None).unwrap();
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

/// N×N supercell: replicates the 2-orbital graphene model to get nsta = 2*N².
fn build_large(n: usize) -> Model<false> {
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut m = Model::<false>::tb_model(2, lat, orb, None).unwrap();
    m.set_onsite(&arr1(&[-0.7, 0.7]), SpinDirection::None);
    let r0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        m.add_hop(
            Complex::new(-1.0, 0.0),
            0,
            1,
            &r.to_owned(),
            SpinDirection::None,
        );
    }
    let sc = arr2(&[[n as f64, 0.0], [0.0, n as f64]]);
    m.make_supercell(&sc).unwrap()
}

// ── Spinless model lists (all Model<false>) ────────────────────────────────

const SPINLESS_MODELS: &[(&str, fn() -> Model<false>)] = &[
    ("small(2orb)", build_small as fn() -> Model<false>),
    ("medium(18orb)", build_medium as fn() -> Model<false>),
    ("large(128orb)", build_large8 as fn() -> Model<false>),
    ("xlarge(450orb)", build_large15 as fn() -> Model<false>),
];

fn build_large8() -> Model<false> { build_large(8) }
fn build_large15() -> Model<false> { build_large(15) }

const SPINLESS_SMALL: &[(&str, fn() -> Model<false>)] = &[
    ("small(2orb)", build_small as fn() -> Model<false>),
    ("medium(18orb)", build_medium as fn() -> Model<false>),
];

const SPINLESS_CURV: &[(&str, fn() -> Model<false>)] = &[
    ("small(2orb)", build_small as fn() -> Model<false>),
];

// ── Benchmarks ──────────────────────────────────────────────────────────────

fn bench_gen_ham(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_ham");
    let kvec = arr1(&[0.3, 0.5]);

    for (name, build) in SPINLESS_MODELS.iter() {
        let m = build();
        group.bench_with_input(
            BenchmarkId::new("Lattice", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_ham(black_box(kv), Gauge::Lattice)),
        );
        group.bench_with_input(
            BenchmarkId::new("Atom", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_ham(black_box(kv), Gauge::Atom)),
        );
    }
    // Spinful separately
    {
        let m = build_small_spinful();
        let name = "spinful(4sta)";
        group.bench_with_input(
            BenchmarkId::new("Lattice", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_ham(black_box(kv), Gauge::Lattice)),
        );
        group.bench_with_input(
            BenchmarkId::new("Atom", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_ham(black_box(kv), Gauge::Atom)),
        );
    }
    group.finish();
}

fn bench_gen_v(c: &mut Criterion) {
    let mut group = c.benchmark_group("gen_v");
    let kvec = arr1(&[0.3, 0.5]);

    for (name, build) in SPINLESS_MODELS.iter() {
        let m = build();
        group.bench_with_input(
            BenchmarkId::new("Atom_gauge", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_v(black_box(kv), Gauge::Atom)),
        );
        group.bench_with_input(
            BenchmarkId::new("Lattice_gauge", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_v(black_box(kv), Gauge::Lattice)),
        );
    }
    // Spinful separately
    {
        let m = build_small_spinful();
        let name = "spinful(4sta)";
        group.bench_with_input(
            BenchmarkId::new("Atom_gauge", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_v(black_box(kv), Gauge::Atom)),
        );
        group.bench_with_input(
            BenchmarkId::new("Lattice_gauge", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.gen_v(black_box(kv), Gauge::Lattice)),
        );
    }
    group.finish();
}

fn bench_solve_onek(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_onek");
    let kvec = arr1(&[0.3, 0.5]);

    for (name, build) in SPINLESS_SMALL.iter() {
        let m = build();
        group.bench_with_input(
            BenchmarkId::new("eigh", name),
            &(&m, &kvec),
            |b, (m, kv)| b.iter(|| m.solve_onek(black_box(kv))),
        );
    }
    // Spinful separately
    {
        let m = build_small_spinful();
        group.bench_with_input(
            BenchmarkId::new("eigh", "spinful(4sta)"),
            &(&m, &kvec),
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
    let kvec = arr1(&[0.3, 0.5]);
    let dir1 = arr1(&[1.0, 0.0]);
    let dir2 = arr1(&[0.0, 1.0]);

    for (name, build) in SPINLESS_CURV.iter() {
        let m = build();
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(&m, &kvec, &dir1, &dir2),
            |b, (m, kv, d1, d2)| {
                b.iter(|| {
                    m.berry_curvature_onek(
                        black_box(kv),
                        black_box(d1),
                        black_box(d2),
                        0.0,
                        0.0,
                        0,
                        1e-3,
                    )
                })
            },
        );
    }
    // Spinful: scalar + spin_z
    {
        let m = build_small_spinful();
        let name = "spinful(4sta)";
        group.bench_with_input(
            BenchmarkId::new("scalar", name),
            &(&m, &kvec, &dir1, &dir2),
            |b, (m, kv, d1, d2)| {
                b.iter(|| {
                    m.berry_curvature_onek(
                        black_box(kv),
                        black_box(d1),
                        black_box(d2),
                        0.0,
                        0.0,
                        0,
                        1e-3,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("spin_z", name),
            &(&m, &kvec, &dir1, &dir2),
            |b, (m, kv, d1, d2)| {
                b.iter(|| {
                    m.berry_curvature_onek(
                        black_box(kv),
                        black_box(d1),
                        black_box(d2),
                        0.0,
                        0.0,
                        3,
                        1e-3,
                    )
                })
            },
        );
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

/// Compare per-row (dimension-dispatched) vs BLAS dot for phase factor computation.
fn bench_phase_dot_vs_perrow(c: &mut Criterion) {
    let mut group = c.benchmark_group("phase_dot_vs_perrow");
    let kvec = arr1(&[0.3, 0.5]);
    let pi2 = 2.0 * std::f64::consts::PI;

    for n in [8, 15, 30, 50] {
        let m = build_large(n);
        let dim = m.dim_r();

        // hamR: per-row (dimension-dispatched)
        group.bench_function(BenchmarkId::new("hamR_perrow", n), |b| {
            b.iter(|| {
                let us: Vec<Complex<f64>> = m
                    .hamR
                    .outer_iter()
                    .map(|r| {
                        let mut p = 0.0f64;
                        for d in 0..dim {
                            p += r[d] as f64 * kvec[d];
                        }
                        Complex::new(0.0, pi2 * p).exp()
                    })
                    .collect();
                black_box(us)
            })
        });

        // hamR: BLAS dot (matmul GEMV)
        group.bench_function(BenchmarkId::new("hamR_dot", n), |b| {
            b.iter(|| {
                let hamr_f64 = m.hamR.mapv(|x| x as f64);
                let phases = hamr_f64.dot(&kvec);
                let us: Array1<Complex<f64>> = phases.mapv(|p| Complex::new(0.0, pi2 * p).exp());
                black_box(us)
            })
        });

        // orb: per-row
        group.bench_function(BenchmarkId::new("orb_perrow", n), |b| {
            b.iter(|| {
                let op: Vec<Complex<f64>> = m
                    .orb
                    .outer_iter()
                    .map(|tau| {
                        let mut p = 0.0f64;
                        for d in 0..dim {
                            p += tau[d] * kvec[d];
                        }
                        Complex::new(0.0, pi2 * p).exp()
                    })
                    .collect();
                black_box(op)
            })
        });

        // orb: BLAS dot
        group.bench_function(BenchmarkId::new("orb_dot", n), |b| {
            b.iter(|| {
                let phases = m.orb.dot(&kvec);
                let op: Array1<Complex<f64>> = phases.mapv(|p| Complex::new(0.0, pi2 * p).exp());
                black_box(op)
            })
        });
    }

    group.finish();
}

fn bench_surfgreen(c: &mut Criterion) {
    let mut group = c.benchmark_group("surfgreen");
    let m = build_small();

    group.bench_function("from_Model", |b| {
        b.iter(|| {
            let sg = surf_Green::from_Model(black_box(&m), 0, 0.01, None).unwrap();
            black_box(sg)
        })
    });

    let sg = surf_Green::from_Model(&m, 0, 0.01, None).unwrap();
    let kvec_1d_sg = arr1(&[0.3]);
    group.bench_function("gen_ham_onek", |b| {
        b.iter(|| {
            let (h0, hr) = sg.gen_ham_onek(black_box(&kvec_1d_sg));
            black_box((h0, hr))
        })
    });

    let sg = surf_Green::from_Model(&m, 1, 0.01, None).unwrap();
    let kvec_1d = arr1(&[0.3]);
    group.bench_function("surf_green_one", |b| {
        b.iter(|| {
            let r = sg.surf_green_one(black_box(&kvec_1d), 0.0);
            black_box(r)
        })
    });

    let energies = Array1::linspace(-3.0, 3.0, 20);
    group.bench_function("surf_green_onek", |b| {
        b.iter(|| {
            let r = sg.surf_green_onek(black_box(&kvec_1d), black_box(&energies));
            black_box(r)
        })
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
    bench_phase_dot_vs_perrow,
    bench_surfgreen,
);
criterion_main!(benches);
