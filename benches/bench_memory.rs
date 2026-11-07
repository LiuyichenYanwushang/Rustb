//! Allocation-counting benchmark for representative response paths.
//!
//! Phase timing is not instrumented here: the repository has no public stage
//! probe and this WSL image has no usable perf/valgrind backend. End-to-end
//! time plus allocator-observed counts are reported instead; phase timing is
//! intentionally left out rather than guessed.
//!
//! The crate's `mimalloc`/`jemalloc` features install their own global
//! allocator. Under those features the counting allocator is disabled and the
//! allocation columns print `n/a`, so the target still builds and the normal
//! criterion benchmarks still run.
//!
//! `allocated_bytes` is the sum of requested allocation sizes (a reallocation
//! counts its full new size). `peak_live_bytes` is the peak net growth since
//! measurement began, not process peak RSS; frees of pre-existing blocks can
//! make the instantaneous value negative, which the peak ignores.

#[cfg(not(any(feature = "mimalloc", feature = "jemalloc")))]
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use std::time::Instant;

use Rustb::*;
use ndarray::*;
use num_complex::Complex;

#[cfg(not(any(feature = "mimalloc", feature = "jemalloc")))]
struct CountingAllocator;

const ALLOCATION_COUNTING: bool = cfg!(not(any(feature = "mimalloc", feature = "jemalloc")));

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicIsize = AtomicIsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

#[cfg(not(any(feature = "mimalloc", feature = "jemalloc")))]
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            let live = LIVE_BYTES.fetch_add(layout.size() as isize, Ordering::Relaxed)
                + layout.size() as isize;
            PEAK_LIVE_BYTES.fetch_max(live.max(0) as usize, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            LIVE_BYTES.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() && COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
            let delta = new_size as isize - layout.size() as isize;
            let live = LIVE_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
            PEAK_LIVE_BYTES.fetch_max(live.max(0) as usize, Ordering::Relaxed);
        }
        new_ptr
    }
}

#[cfg(not(any(feature = "mimalloc", feature = "jemalloc")))]
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

struct Measurement<R> {
    value: R,
    elapsed: std::time::Duration,
    allocations: usize,
    allocated_bytes: usize,
    peak_live_bytes: usize,
}

fn measure<R>(body: impl FnOnce() -> R) -> Measurement<R> {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let start = Instant::now();
    let value = body();
    let elapsed = start.elapsed();
    COUNTING.store(false, Ordering::Relaxed);
    Measurement {
        value,
        elapsed,
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
    }
}

fn build_medium() -> Model<false, 2> {
    let li: Complex<f64> = Complex::i();
    let t = Complex::new(2.0, 0.0);
    let t2 = Complex::new(-1.0, 0.0);
    let delta = 0.7;
    let lat = arr2(&[[1.0, 0.0], [0.5, 3.0_f64.sqrt() / 2.0]]);
    let orb = arr2(&[[1.0 / 3.0, 1.0 / 3.0], [2.0 / 3.0, 2.0 / 3.0]]);
    let mut model = Model::<false, 2>::tb_model(lat, orb, None).unwrap();
    model.set_onsite(&arr1(&[-delta, delta]), None);
    let r0: Array2<isize> = arr2(&[[0, 0], [-1, 0], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        model.add_hop(t, 0, 1, &r.to_owned(), None);
    }
    let r0: Array2<isize> = arr2(&[[1, 0], [-1, 1], [0, -1]]);
    for r in r0.axis_iter(Axis(0)) {
        model.add_hop(t2 * li, 0, 0, &r.to_owned(), None);
    }
    let r0: Array2<isize> = arr2(&[[-1, 0], [1, -1], [0, 1]]);
    for r in r0.axis_iter(Axis(0)) {
        model.add_hop(t2 * li, 1, 1, &r.to_owned(), None);
    }
    model
        .make_supercell(&array![[3.0, 0.0], [0.0, 3.0]])
        .unwrap()
}

fn main() {
    let model = build_medium();
    let kmesh = [9, 9];
    let eta = 0.05;
    let directions = [[1.0, 0.0], [0.0, 1.0]];
    let directions3 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

    let hall_direct = Parameters {
        conditions: Conditions {
            t_kelvin: Sampling::Fixed(150.0),
            mu_ev: Sampling::Fixed(3.0),
            omega_ev: Sampling::Fixed(0.0),
        },
        kmesh,
        integration: Integration::Direct,
    };
    let hall_cut = Parameters {
        conditions: Conditions {
            t_kelvin: Sampling::Fixed(0.0),
            mu_ev: Sampling::Values(Array1::linspace(2.5, 4.5, 16)),
            omega_ev: Sampling::Fixed(0.0),
        },
        kmesh,
        integration: Integration::EnergyCut,
    };
    let optical = Parameters {
        conditions: Conditions {
            t_kelvin: Sampling::Fixed(150.0),
            mu_ev: Sampling::Fixed(3.0),
            omega_ev: Sampling::Values(Array1::linspace(4.0, 12.0, 16)),
        },
        kmesh,
        integration: Integration::Simplex,
    };
    let geometry = Parameters {
        conditions: Conditions {
            t_kelvin: Sampling::Values(array![0.0, 100.0, 300.0]),
            mu_ev: Sampling::Fixed(3.0),
            omega_ev: Sampling::Fixed(0.0),
        },
        kmesh,
        integration: Integration::Simplex,
    };
    let intrinsic = Parameters {
        conditions: Conditions {
            t_kelvin: Sampling::Fixed(150.0),
            mu_ev: Sampling::Values(Array1::linspace(2.5, 4.5, 16)),
            omega_ev: Sampling::Fixed(0.0),
        },
        kmesh,
        integration: Integration::EnergyCut,
    };

    let _ = model.hall_conductivity(&hall_direct, directions, eta, None);
    let _ = model.hall_conductivity(&hall_cut, directions, eta, None);
    let _ = model.optical_conductivity_tensor(&optical, eta);
    let _ = model.quantum_geometry(&geometry, directions, eta);
    let _ = model.intrinsic_nonlinear_hall(&intrinsic, directions3);

    if !ALLOCATION_COUNTING {
        println!("# allocator counting disabled under mimalloc/jemalloc features");
    }
    println!("path\ttime_ms\tallocs\tallocated_bytes\tpeak_live_bytes");

    macro_rules! report {
        ($name:expr, $body:expr) => {{
            let m = measure(|| $body);
            black_box(m.value.unwrap());
            if ALLOCATION_COUNTING {
                println!(
                    "{}\t{:.3}\t{}\t{}\t{}",
                    $name,
                    m.elapsed.as_secs_f64() * 1e3,
                    m.allocations,
                    m.allocated_bytes,
                    m.peak_live_bytes
                );
            } else {
                println!(
                    "{}\t{:.3}\tn/a\tn/a\tn/a",
                    $name,
                    m.elapsed.as_secs_f64() * 1e3
                );
            }
        }};
    }

    report!(
        "hall_direct_fixed_mu",
        model.hall_conductivity(&hall_direct, directions, eta, None)
    );
    report!(
        "hall_energy_cut_mu16",
        model.hall_conductivity(&hall_cut, directions, eta, None)
    );
    report!(
        "optical_tensor_simplex_omega16",
        model.optical_conductivity_tensor(&optical, eta)
    );
    report!(
        "qgt_simplex_temperature3",
        model.quantum_geometry(&geometry, directions, eta)
    );
    report!(
        "intrinsic_energy_cut_2d_mu16",
        model.intrinsic_nonlinear_hall(&intrinsic, directions3)
    );
}
