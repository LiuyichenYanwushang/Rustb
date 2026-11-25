use super::{Floquet, FloquetDrive, FloquetEffectiveOptions, FloquetTruncation, LightMode};
use crate::{
    Atom, AtomType, Gauge, HasRMatrix, Model, OrbProj, OrbitalId, RMatrixData, SpinDirection,
    find_R,
};
use ndarray::{Array1, Array2, Array3, Axis, array};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

const CUTOFF: isize = 4;
const QUADRATURE_MAX: isize = 2 * CUTOFF;
const TOL: f64 = 3e-12;

fn spinful_model<R: RMatrixData>() -> Model<true, 2, R> {
    // Paired labels share atoms/projections, not up/down Wannier centres.
    let orb = array![[0.13, 0.21], [0.63, 0.08], [0.19, 0.17], [0.58, 0.16]];
    let mut atoms = vec![
        Atom::with_orbitals(orb.row(0).to_owned(), AtomType::C, [OrbitalId::new(0)]),
        Atom::with_orbitals(orb.row(1).to_owned(), AtomType::O, [OrbitalId::new(1)]),
    ];
    atoms[0].set_magnetic_moment([0.3, -0.1, 0.2]).unwrap();
    let mut model =
        Model::tb_model_with_state_centres(array![[1.6, 0.4], [-0.3, 1.2]], orb, Some(atoms))
            .unwrap();
    model.orb_projection = vec![OrbProj::s, OrbProj::px];
    model.set_onsite(&array![0.6, -0.44], None);
    model.add_onsite(&array![-0.19, 0.12], SpinDirection::Z);
    model.set_hop(C::new(0.41, -0.22), 0, 1, &array![0, 0], None);
    model.set_hop(C::new(-0.27, 0.18), 1, 0, &array![1, 0], SpinDirection::Y);
    model.set_hop(0.23, 1, 0, &array![0, 1], None);
    model.set_hop(0.0, 0, 0, &array![2, -1], None);
    model
}

fn strong_drive() -> FloquetDrive {
    FloquetDrive::with_modes(
        5.3,
        vec![
            LightMode::new(0, array![C::new(0.42, 9.0), C::new(-0.38, -7.0)]),
            LightMode::new(1, array![C::new(1.3, -0.42), C::new(-0.47, 0.75)]),
            LightMode::new(-2, array![C::new(0.62, 0.34), C::new(0.41, -0.58)]),
        ],
    )
}

// theta = Omega t; W da/dtheta = hbar da/dt. Evaluate the signed input
// modes literally, rather than grouping carriers or shifting Fourier bins.
fn field<const DIM: usize>(drive: &FloquetDrive, theta: f64) -> ([f64; DIM], [f64; DIM]) {
    let mut a = [0.0; DIM];
    let mut da = [0.0; DIM];
    for mode in &drive.modes {
        let l = mode.harmonic as f64;
        let phase = C::from_polar(1.0, -l * theta);
        for axis in 0..DIM {
            let z = mode.a_complex[axis] * phase;
            a[axis] += z.re;
            da[axis] += l * z.im;
        }
    }
    (a, da)
}

fn quadrature(model: &Model<true, 2, HasRMatrix>, drive: &FloquetDrive) -> Vec<Array3<C>> {
    let shape = model.ham.dim();
    let centers = model.orb.dot(&model.lat);
    let mut harmonics = vec![Array3::zeros(shape); (2 * QUADRATURE_MAX + 1) as usize];
    let steps = 512;
    for sample in 0..steps {
        let theta = TAU * sample as f64 / steps as f64;
        let (a, da) = field::<2>(drive, theta);
        let instant = Array3::from_shape_fn(shape, |(r, i, j)| {
            let mut phase = 0.0;
            let mut bare = model.ham[[r, i, j]];
            for axis in 0..2 {
                let translation: f64 = (0..2)
                    .map(|f| model.hamR[[r, f]] as f64 * model.lat[[f, axis]])
                    .sum();
                let d = translation + centers[[j, axis]] - centers[[i, axis]];
                phase -= a[axis] * d;
                let mut y = model.rmatrix[[r, axis, i, j]];
                if model.hamR.row(r) == array![0, 0] && i == j {
                    y -= centers[[i, axis]];
                }
                bare += (drive.omega0_ev * da[axis]) * y;
            }
            C::from_polar(1.0, phase) * bare
        });
        for (q, h) in (-QUADRATURE_MAX..=QUADRATURE_MAX).zip(&mut harmonics) {
            h.scaled_add(
                C::from_polar(1.0 / steps as f64, q as f64 * theta),
                &instant,
            );
        }
    }
    harmonics
}

fn bloch_harmonics(
    harmonics: &[Array3<C>],
    support: &Array2<isize>,
    k: &Array1<f64>,
) -> Vec<Array2<C>> {
    harmonics
        .iter()
        .map(|harmonic| {
            let mut out = Array2::zeros((4, 4));
            for (r, block) in support.rows().into_iter().zip(harmonic.outer_iter()) {
                let angle = TAU * (k[0] * r[0] as f64 + k[1] * r[1] as f64);
                out.scaled_add(C::from_polar(1.0, angle), &block);
            }
            out
        })
        .collect()
}

// Evaluate both nested-commutator families independently in Bloch space,
// summing every signed term rather than reusing adjoints or FFT/convolutions.
fn van_vleck(harmonics: &[Array2<C>], w: f64) -> ([Array2<C>; 3], f64) {
    let h = |q: isize| &harmonics[(q + QUADRATURE_MAX) as usize];
    let comm = |a: &Array2<C>, b: &Array2<C>| a.dot(b) - b.dot(a);
    let mut first = h(0).clone();
    for m in 1..=CUTOFF {
        first.scaled_add(C::from(1.0 / (m as f64 * w)), &comm(h(-m), h(m)));
    }
    let mut second = first.clone();
    let mut high_mixed = Array2::zeros(first.raw_dim());
    for m in (-CUTOFF..=CUTOFF).filter(|&m| m != 0) {
        second.scaled_add(
            C::from(0.5 / (m as f64 * w).powi(2)),
            &comm(h(-m), &comm(h(0), h(m))),
        );
        for l in -CUTOFF..=CUTOFF {
            if l != 0 && l != m {
                // Only the two outer indices are cut off. Track the terms
                // with larger inner indices so the fixture must exercise them.
                let target = if (l - m).abs() > CUTOFF {
                    &mut high_mixed
                } else {
                    &mut second
                };
                target.scaled_add(
                    C::from(1.0 / (3.0 * l as f64 * m as f64 * w * w)),
                    &comm(h(-l), &comm(h(l - m), h(m))),
                );
            }
        }
    }
    let high_mixed_size = high_mixed.iter().map(|z| z.norm()).fold(0.0, f64::max);
    ([h(0).clone(), first, second + high_mixed], high_mixed_size)
}

#[track_caller]
fn assert_matrix(actual: &Array2<C>, expected: &Array2<C>) {
    assert_eq!(actual.dim(), expected.dim());
    for (idx, &want) in expected.indexed_iter() {
        assert!(
            (actual[idx] - want).norm() < TOL,
            "{idx:?}: {} != {want}",
            actual[idx]
        );
    }
}

fn assert_bare_position(
    input: &Model<true, 2, HasRMatrix>,
    out: &Model<true, 2, HasRMatrix>,
    sectors: usize,
) {
    out.validate().unwrap();
    assert_eq!(out.lat, input.lat);
    assert_eq!(out.atoms.len(), input.atoms.len() * sectors);
    assert_eq!(out.norb(), input.norb() * sectors);
    assert_eq!(out.nsta(), input.nsta() * sectors);
    for state in 0..out.nsta() {
        let original = 2 * (state / (2 * sectors)) + state % 2;
        assert_eq!(out.orb.row(state), input.orb.row(original));
    }
    for sector in 0..sectors {
        for orbital in 0..2 {
            let output_orb = sector * 2 + orbital;
            assert_eq!(out.orb.row(output_orb), input.orb.row(orbital));
            assert_eq!(
                out.orb_projection[output_orb],
                input.orb_projection[orbital]
            );
            let a = &out.atoms[output_orb];
            let b = &input.atoms[orbital];
            assert_eq!(a.position_ref(), b.position_ref());
            assert_eq!(a.atom_type(), b.atom_type());
            assert_eq!(a.magnetic_moment(), b.magnetic_moment());
            assert_eq!(a.orbitals(), &[OrbitalId::new(output_orb)]);
        }
    }
    for r in input.hamR.rows() {
        assert!(find_R(&out.hamR, &r).is_some());
    }
    for (r, translation) in out.hamR.rows().into_iter().enumerate() {
        let source = find_R(&input.hamR, &translation);
        for ((axis, i, j), &value) in out.rmatrix.index_axis(Axis(0), r).indexed_iter() {
            // Undo the spin-major (spin, photon, orbital) ordering.
            let original = |idx| 2 * (idx / (2 * sectors)) + idx % 2;
            let expected = source
                .filter(|_| (i / 2) % sectors == (j / 2) % sectors)
                .map_or(C::ZERO, |row| {
                    input.rmatrix[[row, axis, original(i), original(j)]]
                });
            assert_eq!(
                value, expected,
                "bare r at R={translation:?}, ({axis},{i},{j})"
            );
        }
    }
}

#[test]
fn unequal_spin_centres_position_sambe_and_effective_match_strong_field_quadrature() {
    let mut model = spinful_model::<HasRMatrix>();
    for (axis, i, j, value) in [
        (0, 0, 1, C::new(0.35, -0.1)),
        (1, 0, 1, C::new(-0.18, 0.28)),
    ] {
        model.rmatrix[[0, axis, i, j]] = value;
        model.rmatrix[[0, axis, j, i]] = value.conj();
    }
    // Not every onsite diagonal component is an orbital centre.
    model.rmatrix[[0, 0, 2, 2]] += 0.15;
    let r = find_R(&model.hamR, &array![2, -1]).unwrap();
    let s = find_R(&model.hamR, &array![-2, 1]).unwrap();
    for (axis, i, j, value) in [
        (0, 0, 3, C::new(0.21, -0.15)),
        (1, 3, 0, C::new(-0.27, 0.11)),
        (0, 1, 1, C::new(0.12, 0.07)),
    ] {
        model.rmatrix[[r, axis, i, j]] = value;
        model.rmatrix[[s, axis, j, i]] = value.conj();
    }
    assert!(
        model
            .ham
            .index_axis(Axis(0), r)
            .iter()
            .all(|&z| z == C::ZERO)
    );
    let drive = strong_drive();
    let harmonics = quadrature(&model, &drive);
    const N: usize = 2;
    const SECTORS: usize = 2 * N + 1;
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(N as isize))
        .unwrap();
    assert_bare_position(&model, &sambe, SECTORS);
    // Even the corner photon blocks q = ±2N are sizeable on the r-only row.
    for q in [-2 * (N as isize), 2 * (N as isize)] {
        assert!(harmonics[(q + QUADRATURE_MAX) as usize][[r, 0, 3]].norm() > 1e-2);
    }
    for (out_r, row) in sambe.hamR.rows().into_iter().enumerate() {
        let input_r = find_R(&model.hamR, &row).expect("Sambe must not grow hopping support");
        let mut expected = Array2::zeros((4 * SECTORS, 4 * SECTORS));
        let index = |sector, state| state / 2 * (2 * SECTORS) + sector * 2 + state % 2;
        for n in 0..SECTORS {
            for m in 0..SECTORS {
                let h = &harmonics[(m as isize - n as isize + QUADRATURE_MAX) as usize];
                for i in 0..4 {
                    for j in 0..4 {
                        expected[[index(n, i), index(m, j)]] = h[[input_r, i, j]];
                    }
                    if n == m && row == array![0, 0] {
                        expected[[index(n, i), index(m, i)]] +=
                            (n as f64 - N as f64) * drive.omega0_ev;
                    }
                }
            }
        }
        assert_matrix(&sambe.ham.index_axis(Axis(0), out_r).to_owned(), &expected);
    }
    let effective = [0, 1, 2].map(|order| {
        let options = FloquetEffectiveOptions::new()
            .with_order(order)
            .with_harmonic_max(CUTOFF);
        let out = model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();
        assert_bare_position(&model, &out, 1);
        out
    });
    assert!(
        find_R(&effective[1].hamR, &array![4, -2]).is_some(),
        "commutator of r-only rows must grow support"
    );
    for k in [array![0.17, -0.33], array![0.41, 0.24]] {
        let (reference, high_mixed) = van_vleck(
            &bloch_harmonics(&harmonics, &model.hamR, &k),
            drive.omega0_ev,
        );
        assert!(
            high_mixed > 1e-5,
            "fixture must exercise |m′−m| beyond the outer cutoff"
        );
        for (out, want) in effective.iter().zip(&reference) {
            assert_matrix(&out.gen_ham(&k, Gauge::Lattice), want);
        }
    }
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(CUTOFF);
    let q0 = model
        .floquet_effective_q_model(&drive, Some(&options), &array![0.0, 0.0])
        .unwrap();
    assert_eq!(q0.hamR, effective[2].hamR);
    assert_eq!(q0.ham, effective[2].ham);
    assert_eq!(q0.rmatrix.0, effective[2].rmatrix.0);
    let error = model
        .floquet_effective_q_model(&drive, Some(&options), &array![1e-3, 0.0])
        .unwrap_err();
    assert!(
        error.to_string().contains("nonzero q with HasRMatrix"),
        "{error}"
    );
}

#[test]
fn unequal_spin_centres_match_all_orders_peierls_in_sambe_and_effective_models() {
    let peierls = spinful_model::<crate::NoRMatrix>();
    let centres = spinful_model::<HasRMatrix>();
    let drive = strong_drive();
    let k = array![0.17, 0.31];
    let trunc = FloquetTruncation::new(2);
    let bare_sambe = peierls.floquet_model(&drive, &trunc).unwrap();
    let centre_sambe = centres.floquet_model(&drive, &trunc).unwrap();
    assert_bare_position(&centres, &centre_sambe, trunc.n_sector().unwrap());
    bare_sambe.validate().unwrap();
    assert_eq!(bare_sambe.orb, centre_sambe.orb);
    assert_eq!(centre_sambe.hamR.nrows(), bare_sambe.hamR.nrows());
    for (row, translation) in centre_sambe.hamR.rows().into_iter().enumerate() {
        let bare_row = find_R(&bare_sambe.hamR, &translation).unwrap();
        assert_matrix(
            &centre_sambe.ham.index_axis(Axis(0), row).to_owned(),
            &bare_sambe.ham.index_axis(Axis(0), bare_row).to_owned(),
        );
    }

    let harmonics = quadrature(&centres, &drive);
    let (reference, high_mixed) = van_vleck(
        &bloch_harmonics(&harmonics, &centres.hamR, &k),
        drive.omega0_ev,
    );
    assert!(
        high_mixed > 1e-8,
        "pure centre fixture must exercise mixed terms beyond the outer cutoff"
    );
    for (order, want) in reference.iter().enumerate() {
        let options = FloquetEffectiveOptions::new()
            .with_order(order)
            .with_harmonic_max(CUTOFF);
        let p = peierls
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();
        let c = centres
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();
        assert_bare_position(&centres, &c, 1);
        p.validate().unwrap();
        assert_eq!(p.orb, peierls.orb);
        let h = c.gen_ham(&k, Gauge::Lattice);
        assert_matrix(&h, &p.gen_ham(&k, Gauge::Lattice));
        assert_matrix(&h, want);
    }
}

#[test]
fn unequal_spin_centres_zero_drive_is_exact_for_both_position_types() {
    fn check<R: RMatrixData>() {
        let mut model = spinful_model::<R>();
        if R::HAS_RMATRIX {
            let r = model.rmatrix.as_array4_mut();
            r[[0, 0, 3, 3]] += 0.07;
            r[[0, 1, 0, 3]] = C::new(0.13, -0.04);
            r[[0, 1, 3, 0]] = C::new(0.13, 0.04);
        }
        let assert_static = |out: &Model<true, 2, R>| {
            out.validate().unwrap();
            assert_eq!(out.orb, model.orb);
            // An effective model chooses its own support, so match blocks by
            // translation and require every extra block to vanish exactly.
            assert!(out.hamR.nrows() >= model.hamR.nrows());
            let mut retained = vec![false; out.hamR.nrows()];
            for (row, translation) in model.hamR.rows().into_iter().enumerate() {
                let out_row = find_R(&out.hamR, &translation).expect("translation lost");
                retained[out_row] = true;
                assert_eq!(
                    out.ham.index_axis(Axis(0), out_row).to_owned(),
                    model.ham.index_axis(Axis(0), row).to_owned()
                );
            }
            for (out_row, keep) in retained.iter().enumerate() {
                if !keep {
                    assert!(
                        out.ham
                            .index_axis(Axis(0), out_row)
                            .iter()
                            .all(|value| value.norm_sqr() == 0.0),
                        "zero drive must not create new hopping blocks"
                    );
                }
            }
            assert_eq!(out.orb_projection, model.orb_projection);
            assert_eq!(out.natom(), model.natom());
            for (a, b) in out.atoms.iter().zip(&model.atoms) {
                assert_eq!(a.orbitals(), b.orbitals());
                assert_eq!(a.position_ref(), b.position_ref());
                assert_eq!(a.atom_type(), b.atom_type());
                assert_eq!(a.magnetic_moment(), b.magnetic_moment());
            }
            if R::HAS_RMATRIX {
                let out_rmatrix = out.rmatrix.as_array4();
                let model_rmatrix = model.rmatrix.as_array4();
                for (row, translation) in model.hamR.rows().into_iter().enumerate() {
                    let out_row = find_R(&out.hamR, &translation).expect("translation lost");
                    assert_eq!(
                        out_rmatrix.index_axis(Axis(0), out_row).to_owned(),
                        model_rmatrix.index_axis(Axis(0), row).to_owned()
                    );
                }
                for (out_row, keep) in retained.iter().enumerate() {
                    if !keep {
                        assert!(
                            out_rmatrix
                                .index_axis(Axis(0), out_row)
                                .iter()
                                .all(|value| value.norm_sqr() == 0.0),
                            "zero drive must not create new position blocks"
                        );
                    }
                }
            }
        };
        for drive in [
            FloquetDrive::new(3.2),
            FloquetDrive::with_modes(3.2, vec![LightMode::new(1, array![C::ZERO, C::ZERO])]),
        ] {
            for order in 0..=3 {
                let options = FloquetEffectiveOptions::new().with_order(order);
                assert_static(
                    &model
                        .floquet_effective_model(&drive, Some(&options))
                        .unwrap(),
                );
                assert_static(
                    &model
                        .floquet_effective_mode_resolved_model(&drive, Some(&options))
                        .unwrap(),
                );
                assert_static(
                    &model
                        .floquet_effective_q_model(&drive, Some(&options), &array![0.0, 0.0])
                        .unwrap(),
                );
            }
            let requests = [(0, 2), (1, 2), (2, 2), (3, 2)];
            let mut visited = 0;
            model
                .floquet_effective_sweep(&drive, &requests, |key, out| {
                    assert_eq!(key, requests[visited]);
                    visited += 1;
                    assert_static(out);
                    Ok(())
                })
                .unwrap();
            assert_eq!(visited, requests.len());
            assert_static(
                &model
                    .floquet_model(&drive, &FloquetTruncation::new(0))
                    .unwrap(),
            );

            let sambe = model
                .floquet_model(&drive, &FloquetTruncation::new(1))
                .unwrap();
            sambe.validate().unwrap();
            assert_eq!((sambe.norb(), sambe.nsta()), (6, 12));
            assert_eq!(sambe.natom(), 3 * model.natom());
            assert_eq!(sambe.orb_projection.len(), 6);
            for state in 0..12 {
                assert_eq!(
                    sambe.orb.row(state),
                    model.orb.row(2 * (state / 6) + state % 2)
                );
            }
            for ((r, i, j), &value) in sambe.ham.indexed_iter() {
                let ni = (i / 2) % 3;
                let nj = (j / 2) % 3;
                let mut expected = if ni == nj {
                    model.ham[[r, 2 * (i / 6) + i % 2, 2 * (j / 6) + j % 2]]
                } else {
                    C::ZERO
                };
                if r == 0 && i == j {
                    expected += (ni as f64 - 1.0) * drive.omega0_ev;
                }
                assert_eq!(value, expected);
            }
        }
    }
    check::<crate::NoRMatrix>();
    check::<HasRMatrix>();
}

#[test]
fn position_sambe_matches_accelerated_frame_evolution() {
    use crate::ndarray_lapack::eigvalsh_v;
    use ndarray_linalg::{EigVals, UPLO};

    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[2.0]], array![[0.12], [0.63]], None)
            .unwrap();
    model.set_onsite(&array![0.53, -0.26], None);
    model.set_hop(C::new(0.32, 0.14), 0, 1, &array![0], None);
    model.set_hop(C::new(-0.19, 0.22), 0, 1, &array![1], None);
    model.rmatrix[[0, 0, 0, 1]] = C::new(0.35, -0.21);
    model.rmatrix[[0, 0, 1, 0]] = C::new(0.35, 0.21);
    let r = find_R(&model.hamR, &array![1]).unwrap();
    let s = find_R(&model.hamR, &array![-1]).unwrap();
    model.rmatrix[[r, 0, 1, 0]] = C::new(0.24, 0.15);
    model.rmatrix[[s, 0, 0, 1]] = C::new(0.24, -0.15);
    let drive = FloquetDrive::with_modes(
        4.0,
        vec![
            LightMode::new(0, array![C::from(0.31)]),
            LightMode::new(1, array![C::new(0.97, 0.17)]),
            LightMode::new(-2, array![C::new(0.38, -0.44)]),
        ],
    );
    let k = array![0.23];
    let w = drive.omega0_ev;

    // In the accelerated frame the fractional momentum follows
    // k' = k - a L/(2pi). Use the FULL lattice-gauge r at k', without any
    // embedding phases or centre subtraction. A periodic diagonal change
    // of frame conjugates U(T), so its eigenphases are frame independent.
    let generator = |theta| {
        let ([a], [da]) = field::<1>(&drive, theta);
        let k_shift = array![k[0] - a * model.lat[[0, 0]] / TAU];
        let mut h = model.gen_ham(&k_shift, Gauge::Lattice) / w;
        for (r, block) in model
            .hamR
            .rows()
            .into_iter()
            .zip(model.rmatrix.outer_iter())
        {
            let phase = C::from_polar(1.0, TAU * k_shift[0] * r[0] as f64);
            h.scaled_add(da * phase, &block.index_axis(Axis(0), 0));
        }
        h * -C::I
    };
    let propagate = |steps: usize| {
        let dt = TAU / steps as f64;
        let mut u = Array2::<C>::eye(2);
        for step in 0..steps {
            let t = step as f64 * dt;
            let midpoint = generator(t + dt * 0.5);
            let p1 = generator(t).dot(&u);
            let p2 = midpoint.dot(&(&u + &p1 * (dt * 0.5)));
            let p3 = midpoint.dot(&(&u + &p2 * (dt * 0.5)));
            let p4 = generator(t + dt).dot(&(&u + &p3 * dt));
            u += &((p1 + p2 * 2.0 + p3 * 2.0 + p4) * (dt / 6.0));
        }
        let unitarity = u.t().mapv(|z| z.conj()).dot(&u) - Array2::<C>::eye(2);
        assert!(unitarity.iter().all(|z| z.norm() < 1e-10));
        let phases = u.eigvals().unwrap().mapv(|z| -w * z.arg() / TAU);
        let mut levels = [phases[0], phases[1]];
        levels.sort_by(f64::total_cmp);
        levels
    };
    let coarse = propagate(4096);
    let want = propagate(8192);
    for (a, b) in coarse.into_iter().zip(want) {
        assert!((a - b).abs() < 1e-10);
    }
    for n_max in [16, 20] {
        let sambe = model
            .floquet_model(&drive, &FloquetTruncation::new(n_max))
            .unwrap();
        let levels = eigvalsh_v(&sambe.gen_ham(&k, Gauge::Lattice), UPLO::Lower).unwrap();
        // This toy has only two levels in the central zone, well away from its
        // edges. Selecting unfolded central levels is not a general prescription.
        let central: Vec<_> = levels
            .iter()
            .copied()
            .filter(|e| e.abs() < w / 2.0)
            .collect();
        assert_eq!(central.len(), 2);
        for (got, expected) in central.iter().zip(want) {
            assert!(
                (got - expected).abs() < 2e-9,
                "N={n_max}, Sambe {central:?}, propagation {want:?}"
            );
        }
    }
}

fn circular_dipoles(delta: f64, xi: f64) -> Model<false, 2, HasRMatrix> {
    let mut model =
        Model::<false, 2, HasRMatrix>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None)
            .unwrap();
    model.set_onsite(&array![delta, -delta], None);
    model.rmatrix[[0, 0, 0, 1]] = C::from(xi);
    model.rmatrix[[0, 0, 1, 0]] = C::from(xi);
    model.rmatrix[[0, 1, 0, 1]] = -C::I * xi;
    model.rmatrix[[0, 1, 1, 0]] = C::I * xi;
    model
}

#[test]
fn circular_noncommuting_dipoles_have_a_signed_van_vleck_correction() {
    let (delta, a, w) = (0.61, 0.42, 3.7);
    let model = circular_dipoles(delta, 1.0);
    for helicity in [1.0, -1.0] {
        let drive = FloquetDrive::with_modes(
            w,
            vec![LightMode::new(1, array![C::from(a), C::I * (helicity * a)])],
        );
        // Coincident centres: H(t) = delta sigma_z + W A(-sin theta
        // sigma_x + helicity cos theta sigma_y), with only ±1 sidebands.
        // The helicity-odd mass comes from [H_-1,H_1], not the average.
        let mass = delta - helicity * w * a * a;
        for (order, expect) in [delta, mass, mass - 2.0 * delta * a * a]
            .into_iter()
            .enumerate()
        {
            let options = FloquetEffectiveOptions::new().with_order(order);
            let out = model
                .floquet_effective_model(&drive, Some(&options))
                .unwrap();
            assert_matrix(
                &out.ham.index_axis(Axis(0), 0).to_owned(),
                &array![[C::from(expect), C::ZERO], [C::ZERO, C::from(-expect)]],
            );
        }
    }
}

#[test]
fn position_mode_resolved_keeps_sub_ulp_circular_shifts() {
    let model = circular_dipoles(1.0, 1.0);
    let a = 2.0_f64.powi(-29);
    let w = 4.0;
    let mode = LightMode::new(1, array![C::from(a), C::I * a]);
    let options = FloquetEffectiveOptions::new()
        .with_order(1)
        .with_harmonic_max(1);
    let drive = FloquetDrive::with_modes(w, vec![mode.clone()]);
    let single = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    assert_eq!(single.ham[[0, 0, 0]], C::ONE);

    // Each mode shift -W A² rounds away when added to the bare mass,
    // but the sum must be rounded only once, after the modes are combined.
    let count = 64;
    let drive = FloquetDrive::with_modes(w, vec![mode; count]);
    let resolved = model
        .floquet_effective_mode_resolved_model(&drive, Some(&options))
        .unwrap();
    assert_eq!(
        resolved.ham[[0, 0, 0]],
        C::from(1.0 - count as f64 * w * a * a)
    );
    assert_eq!(resolved.rmatrix.0, model.rmatrix.0);
}

#[test]
fn position_derivative_sidebands_keep_balanced_extreme_scales() {
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    let trunc = FloquetTruncation::new(1);
    for (y, a, w) in [
        (1e-210, 1e200, 4.0),
        (1e100, 1e-200, 4.0),
        // Rounding W/2 first at this subnormal frequency loses a third.
        (2.0_f64.powi(1020), 1.0, f64::from_bits(3)),
    ] {
        model.rmatrix[[0, 0, 0, 1]] = C::from(y);
        model.rmatrix[[0, 0, 1, 0]] = C::from(y);
        let drive = FloquetDrive::with_modes(w, vec![LightMode::new(1, array![C::from(a)])]);
        let out = model.floquet_model(&drive, &trunc).unwrap();
        let expected = -0.5 * (w * (y * a));
        let got = out.ham[[0, 2, 5]];
        assert_eq!(got.re, 0.0);
        assert!(
            (got.im / expected - 1.0).abs() < 2e-13,
            "a={a:e}, y={y:e}, W={w:e}: got {got}, expected {expected}i"
        );
    }

    // No scalar link phase to overflow here; the physical derivative sideband
    // itself is out of range, and must not reach the Hamiltonian as NaN/Inf.
    model.rmatrix[[0, 0, 0, 1]] = C::from(f64::MAX / 4.0);
    model.rmatrix[[0, 0, 1, 0]] = C::from(f64::MAX / 4.0);
    let drive = FloquetDrive::with_modes(4.0, vec![LightMode::new(1, array![C::from(4.0)])]);
    for result in [
        model.floquet_model(&drive, &trunc),
        model.floquet_effective_model(&drive, None),
    ] {
        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("representable product range"),
            "{error}"
        );
    }
}

#[test]
fn position_keeps_small_peierls_coefficient_amplified_by_a_dipole() {
    let mut model = Model::<false, 2, HasRMatrix>::tb_model(
        Array2::eye(2),
        array![[0.0, 0.0], [0.5, 0.0]],
        None,
    )
    .unwrap();
    model.rmatrix[[0, 1, 0, 1]] = C::ONE;
    model.rmatrix[[0, 1, 1, 0]] = C::ONE;
    let drive = FloquetDrive::with_modes(
        1e18,
        vec![
            LightMode::new(1, array![C::from(2.0), C::ZERO]),
            LightMode::new(16, array![C::ZERO, C::I]),
        ],
    );
    // H_01(theta) = 16 W cos(16 theta) exp(-i cos(theta)). The average
    // is 16 W J_16(1), about 11.5 eV despite J_16(1) being below 1e-18.
    let expected = C::from(11.498_234_538_891_989);
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(0))
        .unwrap();
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    let effective = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let block = array![[C::ZERO, expected], [expected, C::ZERO]];
    assert_matrix(&sambe.ham.index_axis(Axis(0), 0).to_owned(), &block);
    assert_matrix(&effective.ham.index_axis(Axis(0), 0).to_owned(), &block);
}

#[test]
fn position_cartesian_projection_preserves_small_components_in_cancellation() {
    let w = 2.0_f64.powi(60);
    let small = 2.0_f64.powi(-54);
    let mut model = Model::<false, 3, HasRMatrix>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        None,
    )
    .unwrap();
    model.rmatrix[[0, 0, 0, 1]] = C::ONE;
    model.rmatrix[[0, 0, 1, 0]] = C::ONE;
    let drive = FloquetDrive::with_modes(
        w,
        vec![LightMode::new(
            1,
            array![C::new(1.0, 1.0), C::new(small, -1.0), -C::ONE],
        )],
    );
    let zero = model
        .floquet_model(&drive, &FloquetTruncation::new(0))
        .unwrap();
    // J1(2^-54) = 2^-55 to much better than floating-point precision.
    assert!((zero.ham[[0, 0, 1]] - C::new(0.0, -32.0)).norm() < 1e-12);

    let mut model =
        Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), Array2::zeros((2, 3)), None)
            .unwrap();
    for d in 0..3 {
        model.rmatrix[[0, d, 0, 1]] = C::ONE;
        model.rmatrix[[0, d, 1, 0]] = C::ONE;
    }
    let drive = FloquetDrive::with_modes(
        w,
        vec![LightMode::new(1, array![C::ONE, C::from(small), -C::ONE])],
    );
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(1))
        .unwrap();
    assert_eq!(sambe.ham[[0, 0, 3]], C::new(0.0, -32.0));
}

#[test]
fn position_cancelling_cartesian_drives_are_exactly_static() {
    let model = circular_dipoles(0.37, 0.6);
    let w = 1e-200;
    let mode = array![C::new(0.09, 0.04), C::new(0.11, -0.03)];
    let dc = LightMode::new(0, array![C::from(0.23), C::from(-0.11)]);
    let drive = FloquetDrive::with_modes(
        w,
        vec![
            dc.clone(),
            LightMode::new(1, mode.clone()),
            LightMode::new(1, -mode),
        ],
    );
    let static_drive = FloquetDrive::with_modes(w, vec![dc]);
    let trunc = FloquetTruncation::new(1);
    let got = model.floquet_model(&drive, &trunc).unwrap();
    let want = model.floquet_model(&static_drive, &trunc).unwrap();
    assert_eq!(got.hamR, want.hamR);
    assert_eq!(got.ham, want.ham);
}

#[test]
fn position_projection_retains_product_roundoff_in_cancellation() {
    let w = 2.0_f64.powi(60);
    let eps = 2.0_f64.powi(-27);
    let want = C::new(0.0, 32.0);

    let mut dipole =
        Model::<false, 2, HasRMatrix>::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None)
            .unwrap();
    for (d, y) in [1.0 - eps, 1.0].into_iter().enumerate() {
        dipole.rmatrix[[0, d, 0, 1]] = C::from(y);
        dipole.rmatrix[[0, d, 1, 0]] = C::from(y);
    }
    let drive = FloquetDrive::with_modes(
        w,
        vec![LightMode::new(1, array![C::from(1.0 + eps), -C::ONE])],
    );
    let sambe = dipole
        .floquet_model(&drive, &FloquetTruncation::new(1))
        .unwrap();
    // (1+eps)(1-eps)-1 = -eps²; the derivative contributes -iW/2.
    assert_eq!(sambe.ham[[0, 0, 3]], want);

    let mut phase = Model::<false, 3, HasRMatrix>::tb_model(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [1.0 - eps, 1.0, 0.0]],
        None,
    )
    .unwrap();
    phase.rmatrix[[0, 2, 0, 1]] = C::ONE;
    phase.rmatrix[[0, 2, 1, 0]] = C::ONE;
    let drive = FloquetDrive::with_modes(
        w,
        vec![LightMode::new(1, array![C::from(1.0 + eps), -C::ONE, C::I])],
    );
    let sambe = phase
        .floquet_model(&drive, &FloquetTruncation::new(0))
        .unwrap();
    // -i W J1(-eps²) = +32i to floating-point accuracy.
    assert!((sambe.ham[[0, 0, 1]] - want).norm() < 1e-12);
}

#[test]
fn position_average_retains_hopping_when_large_dipole_terms_cancel() {
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
    model.set_hop(1.0, 0, 0, &array![1], None);
    for translation in [-1, 1] {
        let r = find_R(&model.hamR, &array![translation]).unwrap();
        model.rmatrix[[r, 0, 0, 0]] = C::ONE;
    }
    let drive = FloquetDrive::with_modes(2.0_f64.powi(60), vec![LightMode::new(1, array![C::ONE])]);
    let options = FloquetEffectiveOptions::new()
        .with_order(0)
        .with_harmonic_max(0);
    let out = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    // The dipole term is proportional to the time derivative of exp(∓i cos θ)
    // and averages to zero. Its opposite sidebands must not erase J_0(1).
    for translation in [-1, 1] {
        let r = find_R(&out.hamR, &array![translation]).unwrap();
        let got = out.ham[[r, 0, 0]];
        assert!(
            (got - 0.765_197_686_557_966_6).norm() < TOL,
            "R={translation}: {got}"
        );
    }
}

#[test]
fn position_order_two_survives_underflow_of_the_combined_frequency_scale() {
    let delta = 2.0_f64.powi(100);
    let a = 2.0_f64.powi(-310);
    let w = 2.0_f64.powi(560);
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], Array2::zeros((4, 1)), None)
            .unwrap();
    model.set_onsite(&array![0.0, delta, 3.0 * delta, 0.0], None);
    for (i, j, x) in [(0, 1, 1.0), (1, 3, 2.0), (0, 2, 1.0), (2, 3, -1.0)] {
        model.rmatrix[[0, 0, i, j]] = C::from(x);
        model.rmatrix[[0, 0, j, i]] = C::from(x);
    }
    let drive = FloquetDrive::with_modes(w, vec![LightMode::new(1, array![C::from(a)])]);
    // Equal and opposite imaginary harmonics give no order-one correction.
    // The nested term at (0,3) is a²(2Δ - 3Δ)/2. Its raw block and its
    // final value are both normal, even though the isolated 1/W² is zero.
    for (order, expected) in [(0, 0.0), (1, 0.0), (2, -2.0_f64.powi(-521))] {
        let options = FloquetEffectiveOptions::new()
            .with_order(order)
            .with_harmonic_max(1);
        let out = model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();
        assert_eq!(out.ham[[0, 0, 3]], C::from(expected), "order {order}");
        assert_eq!(out.ham[[0, 3, 0]], C::from(expected), "order {order}");
    }
}

#[test]
fn position_sambe_retains_sidebands_between_wide_harmonics() {
    let mut model =
        Model::<false, 2, HasRMatrix>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
    let (a, t, y) = (0.74, 0.39, 0.27);
    model.set_hop(t, 0, 0, &array![1, 0], None);
    let r = find_R(&model.hamR, &array![1, 0]).unwrap();
    let s = find_R(&model.hamR, &array![-1, 0]).unwrap();
    model.rmatrix[[r, 1, 0, 0]] = C::from(y);
    model.rmatrix[[s, 1, 0, 0]] = C::from(y);
    let p = isize::MAX;
    let w = 1.0 / p as f64;
    let drive = FloquetDrive::with_modes(
        w,
        vec![
            LightMode::new(p, array![C::from(a), C::I * a]),
            LightMode::new(isize::MIN, array![C::from(a), C::from(0.31)]),
        ],
    );
    let out = model
        .floquet_model(&drive, &FloquetTruncation::new(1))
        .unwrap();
    let row = find_R(&out.hamR, &array![1, 0]).unwrap();

    // Frequencies p and p+1 have a beat at 1. For small q the resonant
    // two-angle Fourier indices are (-q,q); all other pairs differ by
    // orders O(p) and have negligible Bessel amplitudes at a < 1.
    // The first and second carrier add cosine and sine dipole terms,
    // respectively. Keep their frequency weights distinct on 32-bit too.
    let cosine = w * p as f64 * a * y;
    let sine = -(w * isize::MIN.unsigned_abs() as f64) * 0.31 * y;
    let expected = [-2, -1, 0, 1, 2].map(|q| {
        let q = q as f64;
        let (mut left, mut right, mut first, mut second) = (C::ZERO, C::ZERO, C::ZERO, C::ZERO);
        for k in 0..256 {
            let angle = TAU * k as f64 / 256.0;
            let (sin, cos) = angle.sin_cos();
            let l = C::from_polar(1.0 / 256.0, -a * cos - q * angle);
            let r = C::from_polar(1.0 / 256.0, -a * cos + q * angle);
            left += l;
            right += r;
            first += l * (t + cosine * cos);
            second += r * (sine * sin);
        }
        first * right + left * second
    });
    assert!((expected[1] - expected[3]).norm() > 1e-3);
    for n in 0..3 {
        for m in 0..3 {
            let idx = m + 2 - n;
            assert!(
                (out.ham[[row, n, m]] - expected[idx]).norm() < TOL,
                "q={}",
                idx as isize - 2
            );
        }
    }
}

#[test]
fn position_floquet_public_branches_reject_nonhermitian_blocks_and_unnegatable_translation() {
    use crate::TbError;

    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_hop(0.0, 0, 0, &array![1], None);
    let row = find_R(&model.hamR, &array![1]).unwrap();
    let drive = FloquetDrive::with_modes(4.0, vec![LightMode::new(1, array![C::from(0.1)])]);
    let options = FloquetEffectiveOptions::new().with_order(0);

    for defect in [
        "hermitian_hamiltonian",
        "hermitian_position_matrix",
        "translation",
    ] {
        let mut invalid = model.clone();
        match defect {
            "hermitian_hamiltonian" => invalid.ham[[row, 0, 1]] = C::I,
            "hermitian_position_matrix" => invalid.rmatrix[[row, 0, 0, 1]] = C::I,
            _ => invalid.hamR[[row, 0]] = isize::MIN,
        }
        invalid.validate().unwrap(); // The synchronized storage is still valid.
        for (branch, result) in [
            (
                "Sambe",
                invalid.floquet_model(&drive, &FloquetTruncation::new(1)),
            ),
            (
                "effective",
                invalid.floquet_effective_model(&drive, Some(&options)),
            ),
            (
                "mode-resolved",
                invalid.floquet_effective_mode_resolved_model(&drive, Some(&options)),
            ),
            (
                "zero q",
                invalid.floquet_effective_q_model(&drive, Some(&options), &array![0.0]),
            ),
        ] {
            let error = result.unwrap_err();
            let expected = match &error {
                TbError::InvalidModelInvariant { invariant, .. } => *invariant == defect,
                TbError::Other(message) => {
                    defect == "translation" && message.contains("cannot be negated")
                }
                _ => false,
            };
            assert!(expected, "{branch}, {defect}: {error}");
        }
    }
}
