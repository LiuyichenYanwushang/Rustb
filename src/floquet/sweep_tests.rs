use crate::{
    Floquet, FloquetDrive, FloquetEffectiveOptions, FloquetTruncation, Gauge, HasRMatrix,
    LightMode, Model, NoRMatrix, RMatrixData, TbError, find_R,
};
use ndarray::{Array2, ArrayView2, Axis, array, s};
use num_complex::Complex64 as C;
use rayon::ThreadPoolBuilder;

const REQUESTS: [(usize, isize); 11] = [
    (1, 4),
    (2, 2),
    (2, 1),
    (2, 4),
    (3, 2),
    (3, 1),
    (0, 9),
    (3, 0),
    (2, 1),
    (3, 2),
    (0, isize::MAX), // The time average must not inflate the shared window.
];

fn model<R: RMatrixData>() -> Model<false, 2, R> {
    let mut model = Model::tb_model(
        array![[1.0, 0.0], [0.0, 1.0]],
        array![[0.0, 0.0], [0.23, 0.31]],
        None,
    )
    .unwrap();
    model.set_onsite(&array![0.33, -0.27], None);
    model.set_hop(C::new(0.61, 0.17), 0, 1, &array![0, 0], None);
    model.set_hop(C::new(-0.7, 0.21), 0, 1, &array![1, 0], None);
    model.set_hop(C::new(0.46, -0.13), 0, 1, &array![0, 1], None);
    model
}

fn drive() -> FloquetDrive {
    FloquetDrive::with_modes(
        4.7,
        vec![
            LightMode::new(1, array![C::new(0.82, 0.13), C::new(0.19, 0.92)]),
            LightMode::new(2, array![C::new(0.25, -0.35), C::new(-0.44, 0.15)]),
        ],
    )
}

fn assert_mixed_window_changes<R: RMatrixData>(model: &Model<false, 2, R>, drive: &FloquetDrive) {
    // At h=1, both mixed second-order terms require H_{±2}. Check that
    // omitting them changes this fixture, using the public Sambe blocks.
    let sambe = model
        .floquet_model(drive, &FloquetTruncation::new(1))
        .unwrap();
    let h = sambe.gen_ham(&array![0.17, -0.23], Gauge::Lattice);
    let block = |n: usize, m: usize| h.slice(s![2 * n..2 * (n + 1), 2 * m..2 * (m + 1)]);
    let comm = |a: ArrayView2<'_, C>, b: ArrayView2<'_, C>| a.dot(&b) - b.dot(&a);
    let mixed = comm(block(1, 0), comm(block(0, 2), block(1, 0)).view())
        + comm(block(0, 1), comm(block(2, 0), block(0, 1)).view());
    assert!(mixed.iter().any(|z| z.norm() > 1e-3));
}

fn assert_matches_independent<const DIM: usize, R: RMatrixData + Send>(
    model: &Model<false, DIM, R>,
    drive: &FloquetDrive,
    requests: &[(usize, isize)],
) {
    let pools = [1, 4].map(|n| ThreadPoolBuilder::new().num_threads(n).build().unwrap());
    let references = pools[0].install(|| {
        requests
            .iter()
            .map(|&(order, h)| {
                let options = FloquetEffectiveOptions::new()
                    .with_order(order)
                    .with_harmonic_max(h);
                model
                    .floquet_effective_model(drive, Some(&options))
                    .unwrap()
            })
            .collect::<Vec<_>>()
    });

    for pool in pools {
        pool.install(|| {
            let mut seen = Vec::new();
            model
                .floquet_effective_sweep(drive, requests, |key, actual| {
                    assert_eq!(key, requests[seen.len()]);
                    let expected = &references[seen.len()];
                    assert_eq!(actual.hamR, expected.hamR, "support for {key:?}");
                    assert!(actual.hamR.row(0).iter().all(|&r| r == 0));
                    assert_eq!(actual.ham.dim(), expected.ham.dim());
                    for (idx, &want) in expected.ham.indexed_iter() {
                        let got = actual.ham[idx];
                        assert!(
                            (got - want).norm() < 2e-11,
                            "{key:?} {idx:?}: {got} != {want}"
                        );
                    }
                    if R::HAS_RMATRIX {
                        for (row, r) in actual.hamR.rows().into_iter().enumerate() {
                            let position = actual.rmatrix.as_array4().index_axis(Axis(0), row);
                            if let Some(input) = find_R(&model.hamR, &r) {
                                assert_eq!(
                                    position,
                                    model.rmatrix.as_array4().index_axis(Axis(0), input)
                                );
                            } else {
                                assert!(position.iter().all(|&z| z == C::ZERO));
                            }
                        }
                    }
                    seen.push(key);
                    Ok(())
                })
                .unwrap();
            assert_eq!(seen.as_slice(), requests);
        });
    }
}

#[test]
fn sweep_matches_independent_without_position() {
    let model = model::<NoRMatrix>();
    let drive = drive();
    assert_mixed_window_changes(&model, &drive);
    assert_matches_independent(&model, &drive, &REQUESTS);
}

#[test]
fn sweep_matches_independent_with_position_only_rows() {
    let mut model = model::<HasRMatrix>();
    model.set_hop(0.0, 0, 1, &array![1, 1], None);
    let plus = find_R(&model.hamR, &array![1, 1]).unwrap();
    let minus = find_R(&model.hamR, &array![-1, -1]).unwrap();
    let position = C::new(0.19, -0.08);
    model.rmatrix[[plus, 0, 0, 1]] = position;
    model.rmatrix[[minus, 0, 1, 0]] = position.conj();
    assert!(
        model
            .ham
            .index_axis(Axis(0), plus)
            .iter()
            .all(|&z| z == C::ZERO)
    );
    let drive = drive();
    assert_mixed_window_changes(&model, &drive);
    assert_matches_independent(&model, &drive, &REQUESTS);
}

#[test]
fn sweep_batches_cubic_windows_with_high_position_harmonic() {
    let mut model = Model::<false, 3, HasRMatrix>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.23, 0.0, 0.31]],
        None,
    )
    .unwrap();
    model.set_onsite(&array![0.33, -0.27], None);
    model.set_hop(C::new(0.41, -0.03), 0, 1, &array![0, 0, 0], None);
    model.set_hop(C::new(-0.5, 0.22), 0, 1, &array![0, 0, 1], None);
    model.set_hop(0.0, 0, 1, &array![1, 0, 1], None);
    for sign in [1, -1] {
        let row = find_R(&model.hamR, &array![sign, 0, sign]).unwrap();
        let i = usize::from(sign < 0);
        model.rmatrix[[row, 2, i, 1 - i]] = C::new(0.13, sign as f64 * 0.17);
    }
    let drive = FloquetDrive::with_modes(
        4.7,
        vec![
            LightMode::new(2, array![C::new(0.82, 0.13), C::ZERO, C::new(0.19, 0.92)]),
            LightMode::new(6, array![C::ZERO, C::ZERO, C::new(0.22, 0.33)]),
        ],
    );
    // h=2 needs H_6 even though 6 is outside both generator windows.
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(3))
        .unwrap();
    let row = find_R(&sambe.hamR, &array![1, 0, 1]).unwrap();
    assert!(sambe.ham[[row, 0, 13]].norm() > 0.1);
    assert_matches_independent(&model, &drive, &[(3, 4), (3, 2), (3, 3)]);
}

#[test]
fn sweep_keeps_forbidden_low_harmonics_exactly_zero() {
    let mut model = Model::<false, 1>::tb_model(array![[2.0]], array![[0.0], [0.5]], None).unwrap();
    model.set_onsite(&array![1.0, -1.0], None);
    model.set_hop(1.0, 0, 1, &array![0], None);
    let drive = FloquetDrive::with_modes(
        1e-40,
        vec![
            LightMode::new(4, array![C::new(1000.0, 0.0)]),
            LightMode::new(8, array![C::new(1000.0, 0.0)]),
        ],
    );
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(1);
    let independent = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();

    // The link is periodic in one quarter of the drive period. The wide
    // request forces temporal sampling, but leakage into H_{±1} must not be
    // multiplied by W^-2 for the narrow request.
    let mut checked = false;
    model
        .floquet_effective_sweep(&drive, &[(2, 1), (1, 2048)], |key, actual| {
            if key == (2, 1) {
                assert_eq!(actual.hamR, independent.hamR);
                for i in 0..2 {
                    assert_eq!(actual.ham[[0, i, i]], model.ham[[0, i, i]]);
                    assert_eq!(actual.ham[[0, i, i]], independent.ham[[0, i, i]]);
                    assert!(
                        (actual.ham[[0, i, 1 - i]] - independent.ham[[0, i, 1 - i]]).norm() < 2e-11
                    );
                }
                checked = true;
            }
            Ok(())
        })
        .unwrap();
    assert!(checked);
}

#[test]
fn sweep_stops_on_callback_error() {
    let mut seen = Vec::new();
    let message = "stop from sweep callback";
    let result = model::<NoRMatrix>().floquet_effective_sweep(&drive(), &REQUESTS, |key, _| {
        seen.push(key);
        if seen.len() == 2 {
            Err(TbError::Other(message.into()))
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(TbError::Other(s)) if s == message));
    assert_eq!(seen.as_slice(), &REQUESTS[..2]);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn sweep_rejects_oversized_shared_cache_before_callbacks() {
    let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
    let drive = FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![C::ONE])]);
    // Even an all-zero Hamiltonian needs this dense cache shape. It fits
    // address arithmetic but exceeds the sweep memory budget.
    let err = model
        .floquet_effective_sweep(&drive, &[(0, 0), (1, 10_000_000_000)], |_, _| {
            panic!("callback ran despite an oversized shared cache")
        })
        .expect_err("accepted an oversized shared harmonic cache");
    assert!(err.to_string().contains("sweep memory budget"), "{err}");
}

#[test]
fn sweep_rejects_large_static_sumset_before_building_it() {
    let mut model =
        Model::<false, 3>::tb_model(Array2::eye(3), array![[0., 0., 0.]], None).unwrap();
    let mut rows = vec![0_isize; 3];
    for axis in 0..3 {
        for r in -1024..=1024 {
            if r != 0 {
                rows.extend(std::array::from_fn::<_, 3, _>(|i| {
                    if i == axis { r } else { 0 }
                }));
            }
        }
    }
    model.ham = ndarray::Array3::zeros((rows.len() / 3, 1, 1));
    model.hamR = Array2::from_shape_vec((rows.len() / 3, 3), rows).unwrap();
    let err = model
        .floquet_effective_sweep(&FloquetDrive::new(1.), &[(3, 1)], |_, _| {
            panic!("oversized static sweep called the visitor")
        })
        .unwrap_err();
    assert!(err.to_string().contains("static output"), "{err}");
}

#[test]
fn sweep_validates_requests_before_callbacks_and_skips_empty() {
    let model = model::<NoRMatrix>();
    let drive = drive();
    for invalid in [(0, -1), (4, 0), (3, isize::MAX), (1, isize::MAX)] {
        let result = model.floquet_effective_sweep(&drive, &[(0, 0), invalid], |key, _| {
            panic!("callback ran for {key:?} despite invalid request {invalid:?}")
        });
        assert!(result.is_err(), "accepted {invalid:?}");
    }

    model
        .floquet_effective_sweep(&FloquetDrive::new(0.0), &[], |_, _| {
            panic!("empty sweep called the callback")
        })
        .unwrap();
}
