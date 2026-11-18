//! Driven honeycomb antiferromagnet of "Odd-Parity Magnons".
//!
//! Prints the Gamma-K-M-Gamma dispersion of Eq. (6) together with the generic
//! real-space `Magnon` solver, the valley splitting, the f-wave pattern of the
//! splitting in Cartesian coordinates, and an independent phenomenological
//! high-frequency propagator check. The latter is not an additional microscopic
//! AB optical correction to the paper's already-driven exchange coefficients.
//!
//! ```text
//! cargo run --release --features openblas-system --example magnon_odd_parity -- \
//!     [lambda] [spin] [K_ev] [helicity]
//! ```

use Rustb::Kpath;
use Rustb::floquet::{FloquetDrive, FloquetTruncation, LightMode, fold_quasienergy};
use Rustb::magnon::floquet::{DrivenExchange, DrivenHubbardParams};
use Rustb::magnon::{MagnonFloquet, MagnonSolve};
use ndarray::{Array2, array};
use ndarray_linalg::Inverse;
use num_complex::Complex;

/// Fractional samples corresponding to a circle in physical reciprocal space.
fn fractional_circle(
    reciprocal: &Array2<f64>,
    radius: f64,
    samples: usize,
) -> Rustb::Result<Vec<[f64; 2]>> {
    // Row coordinates satisfy k_fractional * B = k_cartesian.
    let transform = reciprocal.inv()?;
    Ok((0..samples)
        .map(|index| {
            let phi = std::f64::consts::TAU * index as f64 / samples as f64;
            let fractional = array![radius * phi.cos(), radius * phi.sin()].dot(&transform);
            [fractional[0], fractional[1]]
        })
        .collect())
}

fn sign_changes(values: &[f64], zero_tolerance: f64) -> usize {
    let signs: Vec<bool> = values
        .iter()
        .filter(|value| value.abs() > zero_tolerance)
        .map(|value| *value > 0.0)
        .collect();
    signs
        .iter()
        .zip(signs.iter().cycle().skip(1))
        .filter(|(a, b)| a != b)
        .count()
}

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

    // Shared node-preserving sampler, with Cartesian reciprocal distances
    // omitting 2*pi, as in every Rustb Kpath implementation.
    let magnon = exchange.honeycomb_magnon(spin, anisotropy)?;
    let nodes = array![[0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0], [0.5, 0.5], [0.0, 0.0]];
    let (points, distances, _) = magnon.k_path(&nodes, 73)?;
    println!("#       kx         ky   distance     omega_up    omega_down     solver");
    let mut worst: f64 = 0.0;
    for (point, path) in points.rows().into_iter().zip(distances.iter()) {
        let k = [point[0], point[1]];
        let (up, down) = exchange.honeycomb_dispersion(spin, anisotropy, &k)?;
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
    let (up_k, down_k) =
        exchange.honeycomb_dispersion(spin, anisotropy, &[1.0 / 3.0, 2.0 / 3.0])?;
    let (up_p, down_p) =
        exchange.honeycomb_dispersion(spin, anisotropy, &[2.0 / 3.0, 1.0 / 3.0])?;
    println!(
        "\n# Delta(K) = {:+.6e} eV, Delta(K') = {:+.6e} eV, sum = {:.3e}",
        up_k - down_k,
        up_p - down_p,
        (up_k - down_k) + (up_p - down_p)
    );

    // f-wave: sample a circle of *fixed Cartesian* |k| around Gamma and count
    // the sign changes of the splitting (six zeros, three positive lobes).
    let rec_lat = magnon.rec_lat()?;
    let radius_cart = 0.05;
    // Zero samples are handled by the counter, not avoided by a special count.
    let samples = 73;
    let mut splittings = Vec::with_capacity(samples);
    let mut subtraction_floor = 0.0_f64;
    for k in fractional_circle(&rec_lat, radius_cart, samples)? {
        let (up, down) = exchange.honeycomb_dispersion(spin, anisotropy, &k)?;
        // A conservative subtraction-noise guard, not an arbitrary-precision
        // error certificate for the complete dispersion calculation.
        subtraction_floor = subtraction_floor.max(32.0 * f64::EPSILON * (up.abs() + down.abs()));
        splittings.push(up - down);
    }
    let largest = splittings
        .iter()
        .cloned()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    let zeros = sign_changes(&splittings, (largest * 1e-8).max(subtraction_floor));
    println!(
        "# f-wave at |k_cart| = {radius_cart} /A: {zeros} sign changes, \
         max |Delta| = {largest:.3e} eV"
    );
    if exchange.chirality == 0.0 {
        assert_eq!(largest, 0.0, "zero chirality must not split the sectors");
    } else {
        assert!(
            largest > 100.0 * subtraction_floor,
            "splitting {largest:e} is too close to subtraction noise {subtraction_floor:e}"
        );
        assert_eq!(zeros, 6, "a resolved f-wave should have six sign changes");
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_samples_roundtrip_to_fixed_cartesian_radius() {
        // A skew reciprocal basis distinguishes multiplying by B from B^-1.
        let reciprocal = array![[1.7, 0.3], [-0.4, 2.2]];
        let radius = 0.05;
        let samples = 73;
        let points = fractional_circle(&reciprocal, radius, samples).unwrap();
        assert_eq!(points.len(), samples);
        for (index, point) in points.iter().enumerate() {
            let phi = std::f64::consts::TAU * index as f64 / samples as f64;
            let cart = array![point[0], point[1]].dot(&reciprocal);
            assert!((cart[0] - radius * phi.cos()).abs() < 1e-14, "{cart:?}");
            assert!((cart[1] - radius * phi.sin()).abs() < 1e-14, "{cart:?}");
            assert!((cart[0].hypot(cart[1]) - radius).abs() < 1e-14);
        }
    }

    #[test]
    fn cyclic_sign_count_ignores_zero_samples_and_respects_scale() {
        assert_eq!(sign_changes(&[], 0.0), 0);
        assert_eq!(sign_changes(&[0.0, -0.0, 0.0], 0.0), 0);
        assert_eq!(sign_changes(&[1.0, 0.0, -1.0, -0.0], 1e-12), 2);
        for phase in [0.0, 0.17] {
            for scale in [1.0, 1e-30] {
                let values: Vec<f64> = (0..360)
                    .map(|i| {
                        scale * (3.0 * (std::f64::consts::TAU * i as f64 / 360.0 + phase)).cos()
                    })
                    .collect();
                assert_eq!(sign_changes(&values, scale * 1e-8), 6);
            }
        }
    }
}
