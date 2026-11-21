use super::kspace::EffectivePath;
use super::{Floquet, FloquetDrive, FloquetEffectiveOptions, FloquetTruncation, LightMode};
use crate::{Gauge, HasRMatrix, Model, SpinDirection, find_R};
use ndarray::{Axis, array};
use num_complex::Complex64 as C;
use std::f64::consts::TAU;

#[track_caller]
fn assert_close(actual: C, expected: C) {
    assert!(
        (actual - expected).norm() < 2e-13,
        "expected {expected}, got {actual}"
    );
}

#[test]
fn onsite_dipole_sambe_and_van_vleck_stark_shift() {
    let (delta, xi, a, w) = (0.63, 0.72, 0.31, 3.8);
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_onsite(&array![delta, -delta], None);
    model.rmatrix[[0, 0, 0, 1]] = C::from(xi);
    model.rmatrix[[0, 0, 1, 0]] = C::from(xi);
    let mode = LightMode::new(1, array![C::from(a)]);
    let drive = FloquetDrive::with_modes(w, vec![mode.clone()]);
    let k = array![0.0];

    // With no k dependence and one position component, the velocity-gauge
    // Hamiltonian is the rotation exp(i a(t) X) H0 exp(-i a(t) X).
    // Through quadratic order this is Δ[(1-2b(t)²) σz + 2b(t) σy],
    // with b(t) = A xi cos(Ωt). Compare all blocks, including ±2 and nW.
    let b = a * xi;
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(1))
        .unwrap();
    let hf = sambe.gen_ham(&k, Gauge::Lattice);
    assert_eq!(hf.dim(), (6, 6));
    for ((row, col), &actual) in hf.indexed_iter() {
        let n = (row / 2) as isize - 1;
        let m = (col / 2) as isize - 1;
        let z = if row % 2 == 0 { 1.0 } else { -1.0 };
        let expected = match ((m - n).abs(), row % 2 == col % 2) {
            (0, true) => C::from(n as f64 * w + z * delta * (1.0 - b * b)),
            (1, false) => C::new(0.0, -z * delta * b),
            (2, true) => C::from(-0.5 * z * delta * b * b),
            _ => C::ZERO,
        };
        assert_close(actual, expected);
    }

    // In the length gauge the off-diagonal field has Fourier magnitude W b/2.
    // The two virtual denominators 2Δ±W give -Δ b²/(1-4Δ²/W²).
    // Retain only the selected inverse-frequency orders and no powers > b².
    for order in 0..=2 {
        let options = FloquetEffectiveOptions::new().with_order(order);
        let effective = model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();
        let h = effective.gen_ham(&k, Gauge::Lattice);
        let shift = -delta
            * b
            * b
            * (1.0
                + if order == 2 {
                    4.0 * delta * delta / (w * w)
                } else {
                    0.0
                });
        assert_close(h[[0, 0]], C::from(delta + shift));
        assert_close(h[[1, 1]], C::from(-delta - shift));
        assert_close(h[[0, 1]], C::ZERO);
        assert_close(h[[1, 0]], C::ZERO);

        let uniform = model
            .floquet_effective_q_model(&drive, Some(&options), &array![0.0])
            .unwrap();
        assert_eq!(uniform.gen_ham(&k, Gauge::Lattice), h);
        assert_eq!(uniform.rmatrix.0, model.rmatrix.0);
        assert!(
            model
                .floquet_effective_q_model(&drive, Some(&options), &array![1e-3])
                .is_err()
        );
    }

    // The same two modes add amplitudes coherently and shifts incoherently.
    let doubled = FloquetDrive::with_modes(w, vec![mode.clone(), mode]);
    let coherent = model.floquet_effective_model(&doubled, None).unwrap();
    let resolved = model
        .floquet_effective_mode_resolved_model(&doubled, None)
        .unwrap();
    assert_close(
        coherent.gen_ham(&k, Gauge::Lattice)[[0, 0]],
        C::from(delta * (1.0 - 4.0 * b * b)),
    );
    assert_close(
        resolved.gen_ham(&k, Gauge::Lattice)[[0, 0]],
        C::from(delta * (1.0 - 2.0 * b * b)),
    );
    assert_eq!(resolved.rmatrix.0, model.rmatrix.0);
}

#[test]
fn circular_onsite_position_curvature_and_van_vleck_signs() {
    let (a, w) = (0.08_f64, 3.7);
    let mut model = Model::<false, 2, HasRMatrix>::tb_model(
        array![[1.0, 0.0], [0.0, 1.0]],
        array![[0.0, 0.0], [0.0, 0.0]],
        None,
    )
    .unwrap();
    model.rmatrix[[0, 0, 0, 1]] = C::ONE;
    model.rmatrix[[0, 0, 1, 0]] = C::ONE;
    model.rmatrix[[0, 1, 0, 1]] = -C::I;
    model.rmatrix[[0, 1, 1, 0]] = C::I;
    let k = array![0.0, 0.0];

    for delta in [0.0, 0.61] {
        model.set_onsite(&array![delta, -delta], None);
        for helicity in [1.0, -1.0] {
            let drive = FloquetDrive::with_modes(
                w,
                vec![LightMode::new(
                    1,
                    array![C::from(a), C::new(0.0, helicity * a)],
                )],
            );
            // a(t)=A(cos θ, h sin θ), X=σx, Y=σy; curvature Fxy=2σz.
            let average = delta * (1.0 - 2.0 * a * a) - helicity * w * a * a;
            if delta == 0.0 {
                let sambe = model
                    .floquet_model(&drive, &FloquetTruncation::new(1))
                    .unwrap();
                let hf = sambe.gen_ham(&k, Gauge::Lattice);
                assert_eq!(hf.dim(), (6, 6));
                for ((row, col), &actual) in hf.indexed_iter() {
                    let expected = if row == col {
                        let z = if row % 2 == 0 { 1.0 } else { -1.0 };
                        (row / 2) as f64 * w - w + z * average
                    } else {
                        0.0
                    };
                    assert_close(actual, C::from(expected));
                }

                // Independent forward-time check: the length-gauge Hamiltonian
                // W A(-sin θ σx + h cos θ σy) becomes W(h A σy - h σz/2)
                // in the rotating frame exp(-i h θ σz/2). The frame is -I after
                // a period, so the branch connected to +σz shifts by
                // -h W (sqrt(1/4+A²)-1/2), here rationalized for accuracy.
                let forward = -helicity * w * a * a / (0.5 + (0.25 + a * a).sqrt());
                assert!((hf[[2, 2]].re - forward).abs() <= w * a.powi(4));
            }

            // For Δ≠0 the fundamental is Δ A(σy ∓ i h σx). Both
            // [H^(-1),H^(1)] and the nested commutator affect the mass.
            let expected = [
                average,
                average - helicity * 4.0 * delta * delta * a * a / w,
                average
                    - helicity * 4.0 * delta * delta * a * a / w
                    - 8.0 * delta.powi(3) * a * a / (w * w),
            ];
            for (order, energy) in expected.into_iter().enumerate() {
                let options = FloquetEffectiveOptions::new().with_order(order);
                let effective = model
                    .floquet_effective_model(&drive, Some(&options))
                    .unwrap();
                let h = effective.gen_ham(&k, Gauge::Lattice);
                assert_close(h[[0, 0]], C::from(energy));
                assert_close(h[[1, 1]], C::from(-energy));
                assert_close(h[[0, 1]], C::ZERO);
                assert_close(h[[1, 0]], C::ZERO);
            }
        }
    }
}

#[test]
fn curvature_curl_generates_two_color_sambe_harmonics() {
    let (a, w, lx, ry) = (0.2, 3.9, 1.6, 0.3);
    let mut model = Model::<false, 2, HasRMatrix>::tb_model(
        array![[lx, 0.4], [0.0, 1.0]],
        array![[0.0, 0.0]],
        None,
    )
    .unwrap();
    model.set_hop(0.0, 0, 0, &array![1, 1], None);
    for r in [-1, 1] {
        let row = find_R(&model.hamR, &array![r, r]).unwrap();
        model.rmatrix[[row, 1, 0, 0]] = C::from(ry);
    }
    let drive = FloquetDrive::with_modes(
        w,
        vec![
            LightMode::new(1, array![C::from(a), C::ZERO]),
            LightMode::new(2, array![C::ZERO, C::from(a)]),
        ],
    );
    // Ax=0, Ay(k)=2 ry cos(k_cart·R_cart), with R_cart,x=lx on this skew lattice.
    // At k_frac=(1/4,0), Fxy = ∂_kx Ay = -2 lx ry. For a(t)=A(cos θ,
    // cos 2θ), -W (ax ay' - ay ax') Fxy/2 = W A² Fxy (3 sin θ + sin 3θ)/4.
    let scale = -2.0 * lx * ry * w * a * a / 8.0;
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(2))
        .unwrap();
    let hf = sambe.gen_ham(&array![0.25, 0.0], Gauge::Lattice);
    assert_eq!(hf.dim(), (5, 5));
    for ((row, col), &actual) in hf.indexed_iter() {
        let expected = match col as isize - row as isize {
            0 => C::from((row as f64 - 2.0) * w),
            q @ (-1 | 1) => C::new(0.0, q as f64 * 3.0 * scale),
            q @ (-3 | 3) => C::new(0.0, q.signum() as f64 * scale),
            _ => C::ZERO,
        };
        assert_close(actual, expected);
    }
}

#[test]
fn center_only_position_reduces_to_quadratic_peierls_coupling() {
    let lat = array![[2.0]];
    let orb = array![[0.125], [0.5]];
    let mut peierls = Model::<false, 1>::tb_model(lat.clone(), orb.clone(), None).unwrap();
    let mut projected = Model::<false, 1, HasRMatrix>::tb_model(lat, orb, None).unwrap();
    let t = C::new(0.4, -0.22);
    peierls.set_hop(t, 0, 1, &array![1], None);
    projected.set_hop(t, 0, 1, &array![1], None);
    let k = array![0.19];
    let bare = t * C::from_polar(1.0, TAU * k[0]);
    let options = FloquetEffectiveOptions::new().with_order(0);
    let a = 0.08;
    let drive = FloquetDrive::with_modes(4.0, vec![LightMode::new(1, array![C::from(a)])]);
    let full = peierls
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let quadratic = projected
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    // d = 2 * (1 + 0.5 - 0.125); J0(Ad) = 1 - (Ad)²/4 + (Ad)^4/64 + …
    let z: f64 = a * 2.75;
    let got = quadratic.gen_ham(&k, Gauge::Lattice)[[0, 1]];
    assert_close(got, bare * (1.0 - z * z / 4.0));
    let remainder = (full.gen_ham(&k, Gauge::Lattice)[[0, 1]] - got) / (bare * (z.powi(4) / 64.0));
    assert!(
        (remainder - C::ONE).norm() < 0.002,
        "J0 remainder ratio {remainder}"
    );
}

#[test]
fn spinful_position_is_bare_and_aligned_on_generated_translations() {
    let (lat, delta, a) = (1.4, 0.57, 0.23);
    let mut model =
        Model::<true, 1, HasRMatrix>::tb_model(array![[lat]], array![[0.23]], None).unwrap();
    model.set_onsite(&array![delta], SpinDirection::Z);
    // Allocate the ±1 rows; only the bare position has nonzero entries there.
    model.set_hop(0.0, 0, 0, &array![1], None);
    let z = C::new(0.19, 0.13);
    for (r, value) in [(1, z), (-1, z.conj())] {
        let row = find_R(&model.hamR, &array![r]).unwrap();
        model.rmatrix[[row, 0, 0, 1]] = value;
        model.rmatrix[[row, 0, 1, 0]] = value;
    }
    let drive = FloquetDrive::with_modes(4.2, vec![LightMode::new(1, array![C::new(0.0, a)])]);
    let options = FloquetEffectiveOptions::new().with_order(0);
    let effective = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(1))
        .unwrap();
    effective.validate().unwrap();
    sambe.validate().unwrap();

    // X(k)=x(k)σx up to a scalar center, with x(k)=2 Re(z exp(i k_cart L)).
    // The time average is Δ(1-A²x²)σz - (Δ A²/2) (∂_k x) σy; the
    // x² contribution requires genuine nonzero output hoppings at ±2.
    let k = array![0.17];
    let phase = z * C::from_polar(1.0, TAU * k[0]);
    let x = 2.0 * phase.re;
    let dx = -2.0 * lat * phase.im;
    let h = effective.gen_ham(&k, Gauge::Lattice);
    assert_close(h[[0, 0]], C::from(delta * (1.0 - a * a * x * x)));
    assert_close(h[[0, 1]], C::new(0.0, delta * a * a * dx / 2.0));

    // The spin-major Sambe basis is (spin, photon, orbital). A sine field
    // has coefficients a^(±1) = ±i A/2; H_F[n,m] uses H^(m−n), not H^(n−m).
    let hf = sambe.gen_ham(&k, Gauge::Lattice);
    assert_close(hf[[1, 5]], C::from(a * delta * x));
    assert_close(hf[[1, 3]], C::from(-a * delta * x));

    for (r, value) in [(2, z), (-2, z.conj())] {
        let row = find_R(&effective.hamR, &array![r]).unwrap();
        assert_close(effective.ham[[row, 0, 0]], -delta * a * a * value * value);
        let row = find_R(&sambe.hamR, &array![r]).unwrap();
        assert_close(sambe.ham[[row, 1, 1]], -delta * a * a * value * value);
    }

    // Output may reorder translations. Copy only original position blocks,
    // into equal-photon blocks for Sambe, and zero-fill generated rows.
    for (support, position, sectors) in [
        (&effective.hamR, &effective.rmatrix, 1),
        (&sambe.hamR, &sambe.rmatrix, 3),
    ] {
        for r in model.hamR.rows() {
            assert!(find_R(support, &r).is_some());
        }
        for (i_r, r) in support.rows().into_iter().enumerate() {
            let source = find_R(&model.hamR, &r);
            for ((_, i, j), &value) in position.index_axis(Axis(0), i_r).indexed_iter() {
                let expected = source
                    .filter(|_| i % sectors == j % sectors)
                    .map_or(C::ZERO, |row| {
                        model.rmatrix[[row, 0, i / sectors, j / sectors]]
                    });
                assert_eq!(value, expected, "position at R={r:?}, indices ({i},{j})");
            }
        }
    }
}

#[test]
fn position_order_two_retains_dyadic_term_after_exact_static_cancellation() {
    let delta = 2.0_f64.powi(330);
    let a = 2.0_f64.powi(-30);
    let w = 2.0_f64.powi(560);
    let mut model = Model::<false, 1, HasRMatrix>::tb_model(
        array![[1.0]],
        array![[0.0], [0.0], [0.0], [0.0]],
        None,
    )
    .unwrap();
    model.set_onsite(&array![0.0, delta, 2.0 * delta, 0.0], None);
    for (i, j, x) in [(0, 1, 1.0), (1, 3, 2.0), (0, 2, 1.0), (2, 3, -1.0)] {
        model.rmatrix[[0, 0, i, j]] = C::from(x);
        model.rmatrix[[0, 0, j, i]] = C::from(x);
    }
    let drive = FloquetDrive::with_modes(w, vec![LightMode::new(1, array![C::from(a)])]);

    // The two paths through j=1,2 give zero static shift at (0,3):
    // sum_j E_j X_0j X_j3 = 2Δ - 2Δ. The linear cosine harmonics are equal,
    // so order 1 is zero. Order 2 instead weights each path by E_j³:
    // a² (2Δ³ - (2Δ)³)/(2W²) = -3 * 2^-190, even though 1/W² underflows.
    for (order, expected) in [(0, 0.0), (1, 0.0), (2, -3.0 * 2.0_f64.powi(-190))] {
        let options = FloquetEffectiveOptions::new().with_order(order);
        let effective = model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();
        assert_eq!(effective.ham[[0, 0, 3]], C::from(expected), "order {order}");
        assert_eq!(effective.ham[[0, 3, 0]], C::from(expected), "order {order}");
    }
}

#[test]
fn position_quadratic_coupling_balances_extreme_amplitude_and_matrix_scales() {
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_onsite(&array![1.0, -1.0], None);
    let options = FloquetEffectiveOptions::new().with_order(0);
    for (s, a) in [(1e-210, 1e200), (1e100, 1e-200)] {
        model.rmatrix[[0, 0, 0, 0]] = C::from(s);
        model.rmatrix[[0, 0, 0, 1]] = C::from(s);
        model.rmatrix[[0, 0, 1, 0]] = C::from(s);
        model.rmatrix[[0, 0, 1, 1]] = C::from(-s);
        let drive = FloquetDrive::with_modes(4.0, vec![LightMode::new(1, array![C::from(a)])]);
        let effective = model
            .floquet_effective_model(&drive, Some(&options))
            .unwrap();

        // For X=s(σx+σz), the quadratic rotation term is
        // 2(s a cos θ)² (σx-σz), whose average has off-diagonal +(s a)².
        let expected = (s * a).powi(2);
        let actual = effective.ham[[0, 0, 1]];
        let relative_error = (actual / expected - C::ONE).norm();
        assert!(
            relative_error < 2e-13,
            "s={s:e}, a={a:e}: expected {expected:e}, got {actual}"
        );
    }
}
#[test]
fn wilson_line_reduces_to_peierls_and_exposes_the_dipole_order() {
    use ndarray::Array2;
    use ndarray_linalg::{Eigh, UPLO};

    let lat = array![[3.0_f64]];
    let build = |dipole: f64| {
        let mut model =
            Model::<false, 1, HasRMatrix>::tb_model(lat.clone(), array![[0.0], [0.3]], None)
                .unwrap();
        model.set_hop(0.42, 0, 1, &array![0], None);
        model.set_hop(0.31, 0, 0, &array![1], None);
        model.rmatrix[[0, 0, 0, 0]] = C::from(0.0);
        model.rmatrix[[0, 0, 1, 1]] = C::from(0.9);
        if dipole != 0.0 {
            model.rmatrix[[0, 0, 0, 1]] = C::from(dipole);
            model.rmatrix[[0, 0, 1, 0]] = C::from(dipole);
        }
        model
    };
    // Fractional reciprocal coordinates: the Bloch phase carries 2*pi.
    let tau = 2.0 * std::f64::consts::PI;
    let hamiltonian = |model: &Model<false, 1, HasRMatrix>, k: f64, shift: f64| {
        let mut block = Array2::<C>::zeros((model.nsta(), model.nsta()));
        for (i_r, row) in model.hamR.outer_iter().enumerate() {
            let phase = C::from_polar(1.0, tau * (k - shift) * row[0] as f64);
            block = block + model.ham.index_axis(Axis(0), i_r).mapv(|z| z * phase);
        }
        block
    };
    let connection = |model: &Model<false, 1, HasRMatrix>, k: f64| {
        let mut block = Array2::<C>::zeros((model.nsta(), model.nsta()));
        for (i_r, row) in model.hamR.outer_iter().enumerate() {
            let phase = C::from_polar(1.0, tau * k * row[0] as f64);
            block = block
                + model
                    .rmatrix
                    .index_axis(Axis(0), i_r)
                    .index_axis(Axis(0), 0)
                    .mapv(|z| z * phase);
        }
        block
    };
    // exp(i A X) for a Hermitian X from its eigenbasis.
    let exponent = |hermitian: &Array2<C>, a: f64| {
        let (values, vectors) = hermitian.eigh(UPLO::Lower).unwrap();
        let n = hermitian.nrows();
        let mut out = Array2::<C>::zeros((n, n));
        for index in 0..n {
            let column = vectors.column(index);
            let phase = C::from_polar(1.0, a * values[index]);
            for i in 0..n {
                for j in 0..n {
                    out[[i, j]] += column[i] * column[j].conj() * phase;
                }
            }
        }
        out
    };
    let wilson = |model: &Model<false, 1, HasRMatrix>, k: f64, a: f64| {
        // A in 1/Angstrom shifts the Cartesian wavevector; convert to the
        // fractional coordinate used by the Bloch phase.
        let shift = a * lat[[0, 0]] / tau;
        let w = exponent(&connection(model, k - 0.5 * shift), a);
        let h = hamiltonian(model, k, shift);
        let wd = w.t().mapv(|z| z.conj());
        w.dot(&h).dot(&wd)
    };
    let peierls = |model: &Model<false, 1, HasRMatrix>, k: f64, a: f64| {
        let mut block = Array2::<C>::zeros((model.nsta(), model.nsta()));
        for (i_r, row) in model.hamR.outer_iter().enumerate() {
            for i in 0..model.nsta() {
                for j in 0..model.nsta() {
                    let d = (row[0] as f64 + model.orb[[j, 0]] - model.orb[[i, 0]]) * lat[[0, 0]];
                    // exp(i k R) times the Peierls link phase exp(-i A d).
                    let phase = C::from_polar(1.0, tau * k * row[0] as f64 - a * d);
                    block[[i, j]] += model.ham.index_axis(Axis(0), i_r)[[i, j]] * phase;
                }
            }
        }
        block
    };

    let centre_only = build(0.0);
    let dipolar = build(0.25);
    let a = 0.17;
    let mut worst = 0.0_f64;
    for k in [0.0_f64, 0.13, 0.37] {
        let w = wilson(&centre_only, k, a);
        let p = peierls(&centre_only, k, a);
        for (x, y) in w.iter().zip(p.iter()) {
            worst = worst.max((x - y).norm());
        }
    }
    println!("Wilson vs Peierls with centre-only r: max|diff| = {worst:.3e}");
    assert!(worst < 1e-12, "centre-only Wilson line must equal Peierls");

    // An off-diagonal dipole is dropped by Peierls, so the gap is first order in A.
    let mut ratios = Vec::new();
    for a in [0.02_f64, 0.04, 0.08] {
        let mut diff = 0.0_f64;
        for k in [0.0_f64, 0.13, 0.37] {
            let w = wilson(&dipolar, k, a);
            let p = peierls(&dipolar, k, a);
            for (x, y) in w.iter().zip(p.iter()) {
                diff = diff.max((x - y).norm());
            }
        }
        ratios.push(diff / a);
        println!(
            "A={a}: max|Wilson-Peierls|={diff:.3e} diff/A={:.3e}",
            diff / a
        );
    }
    assert!(
        (ratios[2] - ratios[0]).abs() < 0.2 * ratios[0],
        "the omitted dipole enters linearly in A: {ratios:?}"
    );
}

#[test]
fn position_h0_family_adjoint_matches_termwise() {
    use super::real_space::REAL_SPACE_SUPPORT_CONSTRUCTIONS;

    let mut model = Model::<false, 2, HasRMatrix>::tb_model(
        array![[1.0, 0.0], [0.0, 1.0]],
        array![[0.0, 0.0], [0.35, 0.1]],
        None,
    )
    .unwrap();
    model.set_onsite(&array![0.4, -0.2], None);
    model.set_hop(0.25, 0, 1, &array![0, 0], None);
    model.set_hop(-0.13, 1, 0, &array![1, 0], None);
    model.rmatrix[[0, 0, 0, 1]] = C::from(0.3);
    model.rmatrix[[0, 0, 1, 0]] = C::from(0.3);
    model.rmatrix[[0, 1, 0, 0]] = C::from(-0.17);
    model.rmatrix[[0, 1, 1, 1]] = C::from(-0.17);
    model.validate().unwrap();

    let drive = FloquetDrive::with_modes(
        4.0,
        vec![
            LightMode::new(1, array![C::from(0.2), C::from(0.05)]),
            LightMode::new(2, array![C::from(-0.07), C::from(0.11)]),
        ],
    );
    let options = FloquetEffectiveOptions::new()
        .with_order(2)
        .with_harmonic_max(2);
    let results = [false, true].map(|adjoint| {
        REAL_SPACE_SUPPORT_CONSTRUCTIONS.with(|count| count.set(0));
        let out = model
            .floquet_effective_model_impl(
                &drive,
                Some(&options),
                false,
                adjoint,
                EffectivePath::RealSpace,
            )
            .unwrap();
        (
            out,
            REAL_SPACE_SUPPORT_CONSTRUCTIONS.with(|count| count.get()),
        )
    });
    assert_eq!(results[0].0.hamR, results[1].0.hamR);
    let scale = results[0]
        .0
        .ham
        .iter()
        .chain(results[1].0.ham.iter())
        .map(|z| z.norm())
        .fold(f64::MIN_POSITIVE, f64::max);
    for (a, b) in results[0].0.ham.iter().zip(results[1].0.ham.iter()) {
        assert!((a - b).norm() <= 2e-11 * scale, "{a} vs {b}");
    }
    println!(
        "position order-two convolutions: termwise={} adjoint={}",
        results[0].1, results[1].1
    );
    assert!(results[1].1 < results[0].1, "adjoint path was not taken");
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

#[test]
fn position_sambe_retains_the_beat_between_wide_harmonics() {
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_onsite(&array![1.0, -1.0], None);
    model.rmatrix[[0, 0, 0, 1]] = C::ONE;
    model.rmatrix[[0, 0, 1, 0]] = C::ONE;
    let a = 2.0_f64.powi(-5);
    let drive = FloquetDrive::with_modes(
        4.0,
        [isize::MAX, isize::MIN]
            .into_iter()
            .map(|h| LightMode::new(h, array![C::from(a)]))
            .collect(),
    );
    let sambe = model
        .floquet_model(&drive, &FloquetTruncation::new(1))
        .unwrap();

    // 2 cos(pθ)cos(qθ) = cos(θ) + cos((p+q)θ). In -2 a(t)² σz
    // the beat is -2 a² cos θ σz, giving -a² on each side of the central
    // sector for orbital 0. No carrier, sum, or second harmonic fits here.
    let beat = C::from(-a * a);
    assert_eq!(sambe.ham[[0, 2, 0]], beat);
    assert_eq!(sambe.ham[[0, 2, 4]], beat);
    assert_eq!(sambe.ham[[0, 0, 4]], C::ZERO);
    let options = FloquetEffectiveOptions::new()
        .with_harmonic_max(0)
        .with_order(2);
    let effective = model
        .floquet_effective_model(&drive, Some(&options))
        .unwrap();
    assert_eq!(effective.ham[[0, 0, 0]], C::from(1.0 - 2.0 * a * a));
    assert_eq!(sambe.ham[[0, 2, 2]], effective.ham[[0, 0, 0]]);
}

#[test]
fn position_mode_resolved_adds_sub_ulp_shifts_before_the_bare_hamiltonian() {
    let mut model =
        Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0], [0.0]], None).unwrap();
    model.set_onsite(&array![1.0, -1.0], None);
    model.rmatrix[[0, 0, 0, 1]] = C::ONE;
    model.rmatrix[[0, 0, 1, 0]] = C::ONE;
    let a = 2.0_f64.powi(-28);
    let mode = LightMode::new(1, array![C::from(a)]);
    let options = FloquetEffectiveOptions::new().with_order(0);
    let single_drive = FloquetDrive::with_modes(4.0, vec![mode.clone()]);
    let single = model
        .floquet_effective_model(&single_drive, Some(&options))
        .unwrap();
    assert_eq!(single.ham[[0, 0, 0]], C::ONE);

    // H=σz and X=σx give -a² σz per cosine mode. Each shift rounds away
    // in isolation, but their sum moves the bare value by eight ulps below 1.
    let count = 64;
    let drive = FloquetDrive::with_modes(4.0, vec![mode; count]);
    let resolved = model
        .floquet_effective_mode_resolved_model(&drive, Some(&options))
        .unwrap();
    assert_eq!(resolved.ham[[0, 0, 0]], C::from(1.0 - count as f64 * a * a));
}

#[test]
fn position_does_not_silently_drop_underflowed_curvature() {
    let mut model = Model::<false, 2, HasRMatrix>::tb_model(
        array![[1.0, 0.0], [0.0, 1.0]],
        array![[0.0, 0.0], [0.0, 0.0]],
        None,
    )
    .unwrap();
    let xi = 2.0_f64.powi(-600);
    model.rmatrix[[0, 0, 0, 1]] = C::from(xi);
    model.rmatrix[[0, 0, 1, 0]] = C::from(xi);
    model.rmatrix[[0, 1, 0, 1]] = -C::I * xi;
    model.rmatrix[[0, 1, 1, 0]] = C::I * xi;
    let drive = FloquetDrive::with_modes(
        2.0_f64.powi(800),
        vec![LightMode::new(1, array![C::from(0.125), C::I * 0.125])],
    );
    // Raw curvature products underflow before multiplying by W, but the
    // exact diagonal correction −2^-406 is finite. Reject, don't return zero.
    let err = model
        .floquet_effective_model(&drive, Some(&FloquetEffectiveOptions::new().with_order(0)))
        .unwrap_err();
    assert!(err.to_string().contains("representable product range"));
}
