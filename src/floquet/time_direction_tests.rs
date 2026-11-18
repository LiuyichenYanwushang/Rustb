//! Forward Schrödinger propagation independent of any Fourier/Sambe convention.
use super::{Floquet, FloquetDrive, FloquetEffectiveOptions, FloquetTruncation, LightMode};
use crate::ndarray_lapack::eigvalsh_v;
use crate::{Gauge, Model};
use ndarray::{Array2, array};
use ndarray_linalg::UPLO;
use num_complex::Complex;
use std::f64::consts::TAU;
type C = Complex<f64>;

fn propagated_positive_energy(omega: f64, steps: usize, reverse: bool) -> f64 {
    let delta: f64 = 0.23;
    let hopping: f64 = 0.31;
    let step = TAU / steps as f64;
    let norm = delta.hypot(hopping);
    let c = (norm * step / omega).cos();
    let s = (norm * step / omega).sin() / norm;
    let mut evolution = Array2::<C>::eye(2);
    for k in 0..steps {
        let theta = (k as f64 + 0.5) * step;
        let time = if reverse { -theta } else { theta };
        let phase = 0.8 * time.cos() + 0.45 * (2.0 * time).sin();
        let link = C::from_polar(hopping, -phase);
        let h = array![
            [C::new(delta, 0.0), link],
            [link.conj(), C::new(-delta, 0.0)]
        ];
        let increment = Array2::<C>::eye(2) * c + h * C::new(0.0, -s);
        // Later times multiply on the LEFT: i W dU/dtheta = H(theta) U.
        evolution = increment.dot(&evolution);
    }
    let trace = evolution[[0, 0]] + evolution[[1, 1]];
    assert!(trace.im.abs() < 1e-11);
    (trace.re * 0.5).clamp(-1.0, 1.0).acos() * omega / TAU
}

#[test]
fn sambe_matches_forward_time_not_reversed_protocol() {
    let mut model = Model::<false, 1>::tb_model(array![[2.0]], array![[0.0], [0.5]], None).unwrap();
    model.set_hop(0.23, 0, 0, &array![0], None);
    model.set_hop(-0.23, 1, 1, &array![0], None);
    model.set_hop(0.31, 0, 1, &array![0], None);
    let drive = FloquetDrive::with_modes(
        3.0,
        vec![
            LightMode::new(1, array![C::new(0.8, 0.0)]),
            LightMode::new(2, array![C::new(0.0, 0.45)]),
        ],
    );
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(10))
        .unwrap();
    let eigenvalues =
        eigvalsh_v(&sambe.gen_ham(&array![0.0], Gauge::Lattice), UPLO::Lower).unwrap();
    // This specifically gapped high-frequency toy has exactly two UNFOLDED
    // central eigenvalues; this is not a general Floquet state-selection rule.
    let central: Vec<_> = eigenvalues
        .iter()
        .copied()
        .filter(|e| e.abs() < 1.5)
        .collect();
    assert_eq!(central.len(), 2);
    let coarse = propagated_positive_energy(3.0, 4096, false);
    let forward = propagated_positive_energy(3.0, 8192, false);
    let reversed = propagated_positive_energy(3.0, 8192, true);
    assert!((forward - coarse).abs() < 2e-8);
    assert!(
        (forward - reversed).abs() > 1e-4,
        "toy must distinguish protocol direction"
    );
    eprintln!(
        "Sambe={:.15e}, forward={forward:.15e}, reverse={reversed:.15e}",
        central[1]
    );
    assert!(
        (central[1] - forward).abs() < 2e-8,
        "Sambe must describe the declared forward H(t), not H(-t)"
    );
}

#[test]
fn van_vleck_orders_converge_to_forward_propagation() {
    let mut model = Model::<false, 1>::tb_model(array![[2.0]], array![[0.0], [0.5]], None).unwrap();
    model.set_hop(0.23, 0, 0, &array![0], None);
    model.set_hop(-0.23, 1, 1, &array![0], None);
    model.set_hop(0.31, 0, 1, &array![0], None);
    let mut errors = Vec::new();
    // The first-order coefficient is relatively small for this drive. Compare
    // the inverse-frequency slopes in the asymptotic window, not at W=6 eV
    // where the second-order term appreciably changes the order-zero residual.
    for omega in [24.0, 48.0, 96.0] {
        let drive = FloquetDrive::with_modes(
            omega,
            vec![
                LightMode::new(1, array![C::new(0.8, 0.0)]),
                LightMode::new(2, array![C::new(0.0, 0.45)]),
            ],
        );
        let reference = propagated_positive_energy(omega, 8192, false);
        let mut row = [0.0; 3];
        for (order, error) in row.iter_mut().enumerate() {
            let options = FloquetEffectiveOptions::default()
                .with_order(order)
                .with_harmonic_max(20);
            let effective = model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap();
            let levels = eigvalsh_v(
                &effective.gen_ham(&array![0.0], Gauge::Lattice),
                UPLO::Lower,
            )
            .unwrap();
            *error = (levels[1] - reference).abs();
        }
        eprintln!("W={omega}: forward propagation residuals order0/1/2 = {row:?}");
        assert!(
            row[1] < 0.3 * row[0],
            "first-order sign must improve forward dynamics"
        );
        assert!(
            row[2] < 0.3 * row[1],
            "second-order terms must improve the physical first order"
        );
        errors.push(row);
    }
    for pair in errors.windows(2) {
        let ratios = [
            pair[0][0] / pair[1][0],
            pair[0][1] / pair[1][1],
            pair[0][2] / pair[1][2],
        ];
        eprintln!("frequency-doubling residual ratios = {ratios:?}");
        assert!((1.6..2.5).contains(&ratios[0]));
        assert!((3.0..5.5).contains(&ratios[1]));
        assert!((5.0..11.0).contains(&ratios[2]));
    }
}
