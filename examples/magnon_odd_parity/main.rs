//! Driven honeycomb antiferromagnet of "Odd-Parity Magnons".
//!
//! Prints the Gamma-K-M-Gamma dispersion of Eq. (6) together with the generic
//! real-space `Magnon` solver, the valley splitting, the f-wave pattern of the
//! splitting in Cartesian coordinates, and the high-frequency Sambe check.
//!
//! ```text
//! cargo run --release --features openblas-system --example magnon_odd_parity -- \
//!     [lambda] [spin] [K_ev] [helicity]
//! ```

use Rustb::floquet::{FloquetDrive, FloquetTruncation, LightMode, fold_quasienergy};
use Rustb::magnon::floquet::{DrivenExchange, DrivenHubbardParams};
use Rustb::magnon::{MagnonFloquet, MagnonSolve};
use ndarray::array;
use ndarray_linalg::Inverse;
use num_complex::Complex;

/// Lattice used by `DrivenExchange::honeycomb_magnon`.
const A: f64 = 5.0;

fn main() -> Rustb::Result<()> {
    let mut args = std::env::args().skip(1);
    let lambda: f64 = args.next().map_or(0.4, |value| value.parse().unwrap());
    let spin: f64 = args.next().map_or(1.5, |value| value.parse().unwrap());
    let anisotropy: f64 = args.next().map_or(-0.01, |value| value.parse().unwrap());
    let helicity: f64 = args.next().map_or(1.0, |value| value.parse().unwrap());

    let params = DrivenHubbardParams {
        hopping_ev: 0.05,
        interaction_ev: 1.25,
        photon_ev: 4.0,
        lambda,
        helicity,
        // The Bessel order must cover the drive amplitude.
        harmonic_max: 8.max(lambda.abs().ceil() as isize + 4),
    };
    let exchange = DrivenExchange::honeycomb(&params)?;
    println!(
        "# lambda = {lambda}, helicity = {helicity}, S = {spin}, K = {anisotropy} eV\n\
         # J1 = {:e} eV, J2 = {:e} eV, chi = {:e} eV (chi(-helicity) = -chi)",
        exchange.j1, exchange.j2, exchange.chirality
    );

    // Honeycomb BZ path: Gamma -> K -> M -> Gamma with the M adjacent to K.
    let nodes = [
        ("Gamma", [0.0, 0.0]),
        ("K", [1.0 / 3.0, 2.0 / 3.0]),
        ("M", [0.5, 0.5]),
        ("Gamma", [0.0, 0.0]),
    ];
    let per_segment = 24;
    let mut points: Vec<([f64; 2], f64)> = Vec::new();
    let mut distance = 0.0;
    for window in nodes.windows(2) {
        let (_, start) = window[0];
        let (_, end) = window[1];
        let step = [
            (end[0] - start[0]) / per_segment as f64,
            (end[1] - start[1]) / per_segment as f64,
        ];
        for index in 0..per_segment {
            let k = [
                start[0] + index as f64 * step[0],
                start[1] + index as f64 * step[1],
            ];
            points.push((
                k,
                distance + index as f64 * (step[0] * step[0] + step[1] * step[1]).sqrt(),
            ));
        }
        distance += (step[0] * step[0] + step[1] * step[1]).sqrt() * per_segment as f64;
    }
    points.push(([0.0, 0.0], distance));

    let magnon = exchange.honeycomb_magnon(spin, anisotropy)?;
    println!("#       kx         ky   distance     omega_up    omega_down     solver");
    let mut worst: f64 = 0.0;
    for (k, path) in points.iter() {
        let (up, down) = exchange.honeycomb_dispersion(spin, anisotropy, k);
        let bands = magnon.magnon_band_onek(&array![k[0], k[1]])?;
        let mut expected = [up, down];
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for band in 0..2 {
            worst = worst.max((bands[band] - expected[band]).abs());
        }
        println!(
            "{:10.6} {:10.6} {path:9.4} {up:12.6} {down:12.6} {:11.6}",
            k[0], k[1], bands[0]
        );
    }
    println!("# max |solver - sorted analytic| = {worst:+.3e} eV (must stay below 1e-8)");
    assert!(
        worst < 1e-8,
        "the solver no longer reproduces Eq. (6): {worst:e}"
    );

    // Valley splitting at the two inequivalent corners.
    let (up_k, down_k) = exchange.honeycomb_dispersion(spin, anisotropy, &[1.0 / 3.0, 2.0 / 3.0]);
    let (up_p, down_p) = exchange.honeycomb_dispersion(spin, anisotropy, &[2.0 / 3.0, 1.0 / 3.0]);
    println!(
        "\n# Delta(K) = {:+.6e} eV, Delta(K') = {:+.6e} eV, sum = {:.3e}",
        up_k - down_k,
        up_p - down_p,
        (up_k - down_k) + (up_p - down_p)
    );

    // f-wave: sample a circle of *fixed Cartesian* |k| around Gamma and count
    // the sign changes of the splitting (six zeros, three positive lobes).
    let rec_lat = 2.0
        * std::f64::consts::PI
        * array![[A, 0.0], [0.5 * A, 0.5 * 3.0_f64.sqrt() * A]]
            .t()
            .inv()
            .expect("invertible lattice");
    let radius_cart = 0.05;
    // A prime sample count keeps the samples off the zeros.
    let samples = 73;
    let mut splittings = Vec::with_capacity(samples);
    for index in 0..samples {
        let phi = std::f64::consts::TAU * index as f64 / samples as f64;
        let cart = [radius_cart * phi.cos(), radius_cart * phi.sin()];
        // k_frac . rec_lat = k_cart  =>  k_frac = k_cart . rec_lat^{-1}
        let k = [
            cart[0] * rec_lat[[0, 0]] + cart[1] * rec_lat[[1, 0]],
            cart[0] * rec_lat[[0, 1]] + cart[1] * rec_lat[[1, 1]],
        ];
        let (up, down) = exchange.honeycomb_dispersion(spin, anisotropy, &k);
        splittings.push(up - down);
    }
    let mut zeros = 0;
    for index in 0..samples {
        let a = splittings[index];
        let b = splittings[(index + 1) % samples];
        if (a < 0.0) != (b < 0.0) {
            zeros += 1;
        }
    }
    let largest = splittings
        .iter()
        .cloned()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    println!(
        "# f-wave at |k_cart| = {radius_cart} /A: {zeros} sign changes (expect 6), \
         max |Delta| = {largest:.3e} eV"
    );
    assert_eq!(zeros, 6, "the splitting should have six zeros");

    // High-frequency check: the Sambe quasienergies must converge to the
    // order-0 effective model when the photon energy dominates the bandwidth.
    let q = array![0.25, 0.0];
    let drive = FloquetDrive {
        omega0_ev: 50.0,
        modes: vec![LightMode::new(
            1,
            array![Complex::new(0.02, 0.0), Complex::new(0.0, 0.02)],
        )],
    };
    let sambe = magnon.floquet_model(&drive, &FloquetTruncation { n_max: 2 })?;
    let quasi = sambe.quasienergies_at(&q)?;
    let averaged = magnon.floquet_effective_model(&drive, None)?;
    let bands = averaged.magnon_band_onek(&q)?;
    let residual = (0..quasi.len())
        .map(|index| (quasi[index] - fold_quasienergy(bands[index], drive.omega0_ev)).abs())
        .fold(0.0f64, f64::max);
    println!(
        "# Sambe vs order-0 effective at q = (0.25, 0): max residual = {residual:+.3e} eV \
         (must stay below 1e-6)"
    );
    assert!(
        residual < 1e-6,
        "the high-frequency limit drifted: {residual:e}"
    );
    Ok(())
}
