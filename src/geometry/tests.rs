use super::Berry;
use crate::{Model, SpinDirection};
use ndarray::{Array2, Array3, array, s};
use num_complex::Complex;
use std::f64::consts::PI;

// Analytic occupied projector for H = [[m, t exp(2 pi i kx)],
// [t exp(-2 pi i kx), -m]], m = 0.5 + 0.2 cos(2 pi ky), t = 0.8.
// Tr(P0 P1 ... Pn) is the cyclic product of single-band overlaps.
// The endpoint repeats P0, so omit it; no eigenvectors or production H(k)
// enter this oracle.
fn projector_phase(points: &[[f64; 2]]) -> f64 {
    let mut product = Array2::<Complex<f64>>::eye(2);
    for &[kx, ky] in &points[..points.len() - 1] {
        let mass = 0.5 + 0.2 * (2.0 * PI * ky).cos();
        let energy = (mass * mass + 0.8_f64.powi(2)).sqrt();
        let off_diagonal = -Complex::from_polar(0.4 / energy, 2.0 * PI * kx);
        let projector = array![
            [Complex::new((1.0 - mass / energy) / 2.0, 0.0), off_diagonal],
            [
                off_diagonal.conj(),
                Complex::new((1.0 + mass / energy) / 2.0, 0.0)
            ],
        ];
        product = product.dot(&projector);
    }
    -(product[[0, 0]] + product[[1, 1]]).arg()
}

#[test]
fn atomic_sewing_preserves_radians_sign_and_spin_major_positions() {
    let mut model =
        Model::<false, 2>::tb_model(Array2::eye(2), array![[0.31, -0.17], [-0.13, 0.23]], None)
            .unwrap();
    model.set_onsite(&array![-1.0, 1.0], None);
    let points = array![[0.12, -0.08], [0.49, -0.08], [1.12, -0.08]];
    // Atomic centres give +2 pi G dot tau. Compare the spectrum without
    // assuming that unsorted Wilson eigenvalues follow the occupied-band order.
    let mut phases = model.berry_loop(&points, &[0, 1]).unwrap().to_vec();
    phases.sort_by(f64::total_cmp);
    assert!((phases[0] + 2.0 * PI * 0.13).abs() < 1e-12);
    assert!((phases[1] - 2.0 * PI * 0.31).abs() < 1e-12);
    assert!((model.berry_loop_det(&points, &[0, 1]).unwrap() - 2.0 * PI * 0.18).abs() < 1e-12);
    let reversed = points.slice(s![..;-1, ..]);
    assert!((model.berry_loop(&reversed, &[0]).unwrap()[0] + 2.0 * PI * 0.31).abs() < 1e-12);

    let mut spinful =
        Model::<true, 2>::tb_model(Array2::eye(2), array![[0.17, 0.29], [-0.21, 0.11]], None)
            .unwrap();
    spinful.set_onsite(&array![-0.2, 0.3], None);
    spinful.add_onsite(&array![-1.1, -0.8], SpinDirection::Z);
    // Energies [-1.3, -0.5, 0.9, 1.1] keep the spin-major basis order.
    for (band, position) in [0.17, -0.21, 0.17, -0.21].into_iter().enumerate() {
        let phase = spinful.berry_loop(&points, &[band]).unwrap()[0];
        assert!((phase - 2.0 * PI * position).abs() < 1e-12);
        assert!((spinful.berry_loop_det(&points, &[band]).unwrap() - phase).abs() < 1e-12);
    }
}

#[test]
fn complex_band_loops_flux_and_wannier_match_analytic_projectors() {
    let mut model =
        Model::<false, 2>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
    model.set_onsite(&array![0.5, -0.5], None);
    model.set_hop(0.1, 0, 0, &array![0, 1], None);
    model.set_hop(-0.1, 1, 1, &array![0, 1], None);
    model.set_hop(0.8, 0, 1, &array![1, 0], None);
    let points = [
        [0.0, 0.13],
        [0.12, 0.13],
        [0.37, 0.13],
        [0.61, 0.13],
        [0.83, 0.13],
        [1.0, 0.13],
    ];
    let expected = projector_phase(&points);
    assert!(expected.abs() > 0.2);
    let loop_k = Array2::from_shape_fn((points.len(), 2), |(i, j)| points[i][j]);
    assert!((model.berry_loop(&loop_k, &[0]).unwrap()[0] - expected).abs() < 1e-12);
    assert!((model.berry_loop_det(&loop_k, &[0]).unwrap() - expected).abs() < 1e-12);
    assert!(
        (model
            .berry_loop(&loop_k.slice(s![..;-1, ..]), &[0])
            .unwrap()[0]
            + expected)
            .abs()
            < 1e-12
    );

    let flux = model
        .berry_flux(
            &[0],
            &array![0.11, 0.08],
            &array![0.3, 0.0],
            &array![0.0, 0.35],
            2,
            3,
        )
        .unwrap();
    let reversed_flux = model
        .berry_flux(
            &[0],
            &array![0.11, 0.08],
            &array![0.0, 0.35],
            &array![0.3, 0.0],
            3,
            2,
        )
        .unwrap();
    assert_eq!(flux.dim(), (2, 3, 1));
    assert_eq!(reversed_flux.dim(), (3, 2, 1));
    assert!(flux.iter().any(|phase| phase.abs() > 1e-3));
    for i in 0..2 {
        for j in 0..3 {
            let x = 0.11 + i as f64 * 0.15;
            let y = 0.08 + j as f64 * 0.35 / 3.0;
            let plaquette = [
                [x, y],
                [x + 0.15, y],
                [x + 0.15, y + 0.35 / 3.0],
                [x, y + 0.35 / 3.0],
                [x, y],
            ];
            let expected = projector_phase(&plaquette);
            assert!((flux[[i, j, 0]] - expected).abs() < 1e-12);
            assert!((reversed_flux[[j, i, 0]] + expected).abs() < 1e-12);
        }
    }

    // Three transverse ky values and seven integration kx values distinguish
    // both axes, the integration direction, and inclusion of both endpoints.
    let batch = Array3::from_shape_fn((3, 7, 2), |(i, j, axis)| {
        if axis == 0 {
            0.07 + j as f64 / 6.0
        } else {
            -0.07 + i as f64 * 0.25
        }
    });
    let phases = model.berry_phase(&[0], &batch).unwrap();
    let centres = model
        .wannier_centre(
            &[0],
            &array![0.07, -0.07],
            &array![0.0, 0.5],
            &array![1.0, 0.0],
            3,
            7,
        )
        .unwrap();
    assert_eq!(phases.dim(), (1, 3));
    assert_eq!(centres.dim(), (1, 3));
    for i in 0..3 {
        let manual = (0..7)
            .map(|j| [0.07 + j as f64 / 6.0, -0.07 + i as f64 * 0.25])
            .collect::<Vec<_>>();
        let expected = projector_phase(&manual);
        assert!((phases[[0, i]] - expected).abs() < 1e-12);
        assert!((centres[[0, i]] - expected).abs() < 1e-12);
    }
    assert!((centres[[0, 0]] - centres[[0, 2]]).abs() > 0.1);
}

#[test]
fn berry_batch_sorts_columns_and_preserves_empty_outer_axis() {
    let mut model =
        Model::<false, 2>::tb_model(Array2::eye(2), array![[0.31, -0.17], [-0.13, 0.23]], None)
            .unwrap();
    model.set_onsite(&array![-1.0, 1.0], None);
    let batch = array![
        [[0.1, 0.2], [0.6, 0.2], [1.1, 0.2]],
        [[0.1, 0.2], [0.1, 0.7], [0.1, 1.2]],
        [[0.1, 0.2], [0.6, 0.7], [1.1, 1.2]],
    ];
    let actual = model.berry_phase(&[0, 1], &batch).unwrap();
    let expected = array![[-0.13, -0.17, 0.10], [0.31, 0.23, 0.14]] * (2.0 * PI);
    assert_eq!(actual.dim(), (2, 3));
    assert!(
        actual
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    let empty = Array3::zeros((0, 3, 2));
    assert_eq!(model.berry_phase(&[0, 1], &empty).unwrap().dim(), (2, 0));
    assert!(model.berry_phase(&[], &empty).is_err());
    assert!(model.berry_phase(&[0], &Array3::zeros((0, 1, 2))).is_err());
    assert!(model.berry_phase(&[0], &Array3::zeros((0, 3, 1))).is_err());
    let mut broken = batch;
    broken[[1, 2, 1]] += 0.2;
    assert!(model.berry_phase(&[0], &broken).is_err());
    model.ham[[0, 0, 0]] = Complex::new(f64::NAN, 0.0);
    assert!(model.berry_phase(&[0], &empty).is_err());
}

#[test]
fn berry_rejects_invalid_loops_bands_models_and_plane_inputs() {
    let model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.2, 0.1]], None).unwrap();
    let valid = array![[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]];
    for invalid in [
        Array2::zeros((0, 2)),
        Array2::zeros((1, 2)),
        Array2::zeros((2, 1)),
        Array2::zeros((2, 3)),
        array![[0.0, 0.0], [f64::NAN, 0.0], [1.0, 0.0]],
        array![[0.0, 0.0], [f64::INFINITY, 0.0], [1.0, 0.0]],
        array![[0.0, 0.0], [0.2, 0.0]],
    ] {
        assert!(model.berry_loop(&invalid, &[0]).is_err());
        assert!(model.berry_loop_det(&invalid, &[0]).is_err());
    }
    for occ in [&[][..], &[0, 0][..], &[1][..], &[usize::MAX][..]] {
        assert!(model.berry_loop(&valid, occ).is_err());
        assert!(model.berry_loop_det(&valid, occ).is_err());
    }
    let within_tolerance = array![[0.0, 0.0], [1.0 + 0.5e-9, 0.0]];
    assert!(model.berry_loop(&within_tolerance, &[0]).is_ok());
    assert!(model.berry_loop_det(&within_tolerance, &[0]).is_ok());
    let outside_tolerance = array![[0.0, 0.0], [1.0 + 2e-9, 0.0]];
    assert!(model.berry_loop(&outside_tolerance, &[0]).is_err());
    assert!(model.berry_loop_det(&outside_tolerance, &[0]).is_err());

    let origin = array![0.0, 0.0];
    let x = array![1.0, 0.0];
    let y = array![0.0, 1.0];
    for bad in [
        array![0.0],
        array![f64::NAN, 0.0],
        array![0.0, f64::INFINITY],
    ] {
        for (start, first, second) in [(&bad, &x, &y), (&origin, &bad, &y), (&origin, &x, &bad)] {
            assert!(model.berry_flux(&[0], start, first, second, 2, 2).is_err());
            assert!(
                model
                    .wannier_centre(&[0], start, first, second, 2, 2)
                    .is_err()
            );
        }
    }
    for (n1, n2) in [(0, 1), (1, 0), (usize::MAX, 2), (2, usize::MAX)] {
        assert!(model.berry_flux(&[0], &origin, &x, &y, n1, n2).is_err());
    }
    for (n1, n2) in [
        (0, 2),
        (1, 2),
        (2, 0),
        (2, 1),
        (usize::MAX, 2),
        (2, usize::MAX),
    ] {
        assert!(model.wannier_centre(&[0], &origin, &x, &y, n1, n2).is_err());
    }
    assert_eq!(
        model.berry_flux(&[0], &origin, &x, &y, 1, 1).unwrap().dim(),
        (1, 1, 1)
    );
    assert!(
        model
            .wannier_centre(&[0], &origin, &x, &array![0.0, 0.5], 2, 2)
            .is_err()
    );
    for occ in [&[][..], &[0, 0][..], &[1][..]] {
        assert!(model.berry_flux(occ, &origin, &x, &y, 1, 1).is_err());
        assert!(model.wannier_centre(occ, &origin, &x, &y, 2, 2).is_err());
    }
    let mut invalid_model = model;
    invalid_model.ham[[0, 0, 0]] = Complex::new(f64::NAN, 0.0);
    assert!(invalid_model.berry_loop(&valid, &[0]).is_err());
    assert!(invalid_model.berry_loop_det(&valid, &[0]).is_err());
    assert!(
        invalid_model
            .berry_flux(&[0], &origin, &x, &y, 1, 1)
            .is_err()
    );
    assert!(
        invalid_model
            .wannier_centre(&[0], &origin, &x, &y, 2, 2)
            .is_err()
    );
}

#[test]
fn berry_propagates_generated_hamiltonian_errors_from_parallel_workers() {
    let mut model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
    model.set_hop(f64::MAX, 0, 0, &array![1, 0], None);
    model.validate().unwrap(); // Stored data are finite; H(0) overflows.
    let bad_loop = array![[0.0, 0.0], [0.0, 1.0]];
    assert!(model.berry_loop(&bad_loop, &[0]).is_err());
    assert!(model.berry_loop_det(&bad_loop, &[0]).is_err());
    let batch = array![[[0.25, 0.0], [0.25, 1.0]], [[0.0, 0.0], [0.0, 1.0]],];
    assert!(model.berry_loop(&batch.slice(s![0, .., ..]), &[0]).is_ok());
    assert!(model.berry_phase(&[0], &batch).is_err());
    assert!(
        model
            .berry_flux(
                &[0],
                &array![0.25, 0.0],
                &array![0.75, 0.0],
                &array![0.0, 1.0],
                2,
                2
            )
            .is_err()
    );
    assert!(
        model
            .wannier_centre(
                &[0],
                &array![0.25, 0.0],
                &array![0.25, 0.0],
                &array![0.0, 1.0],
                2,
                2
            )
            .is_err()
    );
}

#[test]
fn berry_rejects_exactly_orthogonal_adjacent_selected_states() {
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_hop(0.5, 0, 0, &array![1], None);
    model.set_hop(-0.5, 1, 1, &array![1], None);
    // H(k) = diag(cos(2 pi k), -cos(2 pi k)): the lowest band switches
    // between exact coordinate basis vectors, making the overlap exactly 0.
    let points = array![[0.0], [0.5], [1.0]];
    assert!(model.berry_loop(&points, &[0]).is_err());
    assert!(model.berry_loop_det(&points, &[0]).is_err());
    // The complete two-band subspace has nonsingular overlaps despite the
    // crossing, so no artificial spectral-gap threshold should reject it.
    assert!(model.berry_loop(&points, &[0, 1]).is_ok());
    assert!(model.berry_loop_det(&points, &[0, 1]).is_ok());
}

#[test]
fn determinant_underflow_does_not_affect_unitarized_wilson_links() {
    let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_onsite(&array![1e-8, -1e-8], None);
    model.set_hop(1.0, 0, 1, &array![1], None);
    // Opposite points of this gapped two-band loop have overlap magnitude
    // m/sqrt(1+m^2), approximately 1e-8. Every link is nonsingular, but the
    // raw product of 48 links underflows. Backtracking cancels the phase.
    let points = Array2::from_shape_fn((49, 1), |(i, _)| {
        if i == 48 {
            1.0
        } else if i % 2 == 0 {
            0.0
        } else {
            0.5
        }
    });
    assert!(model.berry_loop(&points, &[0]).unwrap()[0].abs() < 1e-12);
    assert!(model.berry_loop_det(&points, &[0]).is_err());
}
