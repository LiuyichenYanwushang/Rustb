//! Honeycomb benchmark for the electronic Aharonov–Bohm (AB) mechanism.
//!
//! The lattice is the Néel honeycomb of *Odd-Parity Magnons*, arXiv:2605.31411v1.
//! Its published Fig. 2 is Aharonov–Casher, not the AB calculation below.
//! This test uses the repository's AB example parameters and compares the
//! new bondwise method against the second-order term, not the whole figure.
//! AB exchange reference: Quito and Flint, arXiv:2003.05933v2, Eq. (41).
//!
//! At the default drive J1 falls from 8.000000 to 7.312223 meV. The two AB2
//! bands coincide at 53.612499, 62.905001 and 61.941382 meV at Γ, K and M.
//! The repository's specialized higher-order model gives a signed K/K' split
//! of +0.036057/-0.036057 meV; the AB2 calculation has zero split.
//!
//! `cargo test --release --features openblas-system --test magnon_odd_parity_ab -- --nocapture`

use Rustb::Kpath;
use Rustb::floquet::{FloquetDrive, LightMode};
use Rustb::magnon::MagnonSolve;
use Rustb::magnon::floquet::{DrivenExchange, DrivenHubbardParams};
use ndarray::{arr2, array};
use num_complex::Complex;
use std::f64::consts::TAU;

const T: f64 = 0.05;
const U: f64 = 1.25;
const W: f64 = 4.0;
const LAMBDA: f64 = 0.4;
const SPIN: f64 = 1.5;
const ANISOTROPY: f64 = -0.01;
const TOL: f64 = 2e-12; // eV

fn ab_exchange_oracle() -> f64 {
    // Independent time DFT: full denominators, not the production Bessel
    // backend or its Parseval correction. A 256-point grid is ample at λ=0.4.
    let mut exchange = 0.0;
    for m in -12..=12 {
        let coefficient = (0..256)
            .map(|i| {
                let theta = TAU * f64::from(i) / 256.0;
                Complex::from_polar(1.0, f64::from(m) * theta - LAMBDA * theta.cos())
            })
            .sum::<Complex<f64>>()
            / 256.0;
        exchange += 4.0 * T * T * coefficient.norm_sqr() / (U - f64::from(m) * W);
    }
    exchange
}

fn nn_band(j: f64, [qx, qy]: [f64; 2]) -> f64 {
    // Drop the common basis-position phase from the three NN links.
    let gamma =
        1.0 + Complex::from_polar(1.0, TAU * qy) + Complex::from_polar(1.0, TAU * (qy - qx));
    let a = SPIN * (3.0 * j - 2.0 * ANISOTROPY);
    let b = SPIN * j * gamma.norm();
    ((a - b) * (a + b)).sqrt()
}

#[test]
fn honeycomb_ab_recovers_second_order_but_not_odd_parity_splitting() -> Rustb::Result<()> {
    // Start with the second-order dark exchange. Do not dress the fourth-order
    // J2 as if it were a second-order bond, and do not inject chi in the input.
    let j_dark = 4.0 * T * T / U;
    let dark = DrivenExchange {
        j1: j_dark,
        j2: 0.0,
        chirality: 0.0,
    }
    .honeycomb_magnon(SPIN, ANISOTROPY)?;
    let nearest_bond_angstrom = 5.0 / 3.0_f64.sqrt();
    let amplitude = LAMBDA / nearest_bond_angstrom;
    let drive = |helicity: f64| {
        FloquetDrive::with_modes(
            W,
            vec![LightMode::new(
                1,
                array![
                    Complex::new(amplitude, 0.0),
                    Complex::new(0.0, helicity * amplitude),
                ],
            )],
        )
    };
    let right = dark.floquet_superexchange(&drive(1.0), U, 8)?;
    let left = dark.floquet_superexchange(&drive(-1.0), U, 8)?;
    let converged = dark.floquet_superexchange(&drive(1.0), U, 12)?;
    let j_reference = ab_exchange_oracle();
    assert!(
        j_reference < 0.95 * j_dark,
        "benchmark must resolve an exchange change"
    );
    let mut nearest_count = 0;
    for r in 0..dark.nR() {
        for i in 0..2 {
            for j in 0..2 {
                let tensor = dark.jtensor(i, j, r)?;
                let actual = right.jtensor(i, j, r)?;
                if i != j && tensor[0][0] != 0.0 {
                    nearest_count += 1;
                    for (a, row) in actual.iter().enumerate() {
                        for (b, &value) in row.iter().enumerate() {
                            let expected = if a == b {
                                -0.5 * SPIN * SPIN * j_reference
                            } else {
                                0.0
                            };
                            assert!(
                                (value - expected).abs() < TOL,
                                "R={r}, bond={i}->{j}, a={a}, b={b}"
                            );
                        }
                    }
                } else {
                    assert_eq!(
                        actual, tensor,
                        "onsite and absent bonds must stay unchanged"
                    );
                }
            }
        }
    }
    assert_eq!(nearest_count, 6); // 3 NN bonds and their reciprocal partners
    assert!((&right.ham - &left.ham).iter().all(|d| d.abs() < TOL));
    assert!((&right.ham - &converged.ham).iter().all(|d| d.abs() < TOL));

    let gamma = [0.0, 0.0];
    let k = [1.0 / 3.0, 2.0 / 3.0];
    let kp = [2.0 / 3.0, 1.0 / 3.0];
    let m = [0.5, 0.5];
    let (points, _, _) = right.k_path(&arr2(&[gamma, k, m, gamma]), 73)?;
    let bands = right.magnon_band_all(&points)?;
    let mut max_error: f64 = 0.0;
    for (q, energies) in points.rows().into_iter().zip(bands.rows()) {
        let expected = nn_band(j_reference, [q[0], q[1]]);
        for &energy in energies {
            max_error = max_error.max((energy - expected).abs());
        }
    }
    assert!(
        max_error < TOL,
        "AB second-order LSWT: max error {max_error:e} eV"
    );

    // The retained specialized AB implementation includes fourth order. Use it
    // only as a separate comparison, never to obtain the new method's input.
    let params = DrivenHubbardParams {
        hopping_ev: T,
        interaction_ev: U,
        photon_ev: W,
        lambda: LAMBDA,
        helicity: 1.0,
        harmonic_max: 8,
    };
    let full = DrivenExchange::honeycomb(&params)?;
    let split = |q| -> Rustb::Result<f64> {
        let (up, down) = full.honeycomb_dispersion(SPIN, ANISOTROPY, &q)?;
        Ok(up - down)
    };
    let j_nn = -2.0 * right.jtensor(0, 1, 0)?[0][0] / SPIN.powi(2);
    println!(
        "J1(dark)={:.9} meV, J1(AB2)={:.9} meV; full AB J1={:.9} meV, J2={:.9} meV, chi={:.9} meV",
        1e3 * j_dark,
        1e3 * j_nn,
        1e3 * full.j1,
        1e3 * full.j2,
        1e3 * full.chirality
    );
    println!("max second-order analytic spectrum error (73 points): {max_error:.3e} eV");
    println!(
        "point    dark[meV]    AB2 band 1[meV]    AB2 band 2[meV]    full AB signed split[meV]"
    );
    for (label, q) in [("Gamma", gamma), ("K", k), ("M", m), ("K'", kp)] {
        let bare = dark.magnon_band_onek(&array![q[0], q[1]])?;
        let energy = right.magnon_band_onek(&array![q[0], q[1]])?;
        assert!((energy[0] - energy[1]).abs() < TOL);
        println!(
            "{label:5}  {:12.6}  {:17.6}  {:17.6}  {:+.9}",
            1e3 * bare[0],
            1e3 * energy[0],
            1e3 * energy[1],
            1e3 * split(q)?
        );
    }
    assert!(
        split(k)?.abs() > 1e-5,
        "full comparator must split the valley"
    );
    assert!((split(k)? + split(kp)?).abs() < TOL);
    Ok(())
}
