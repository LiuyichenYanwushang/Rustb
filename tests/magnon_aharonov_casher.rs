use Rustb::TbError;
use Rustb::floquet::{FloquetDrive, FloquetTruncation, LightMode, fold_quasienergy};
use Rustb::magnon::{Magnon, MagnonReference, MagnonSambe, MagnonSolve};
use ndarray::{Array1, Array2, Array3, array};
use num_complex::Complex;

const W: f64 = 2.3;
const AXIS: [f64; 3] = [0.0, 0.0, 1.0];
const ELECTRIC: [[f64; 2]; 3] = [[0.21, 0.13], [-0.08, 0.19], [0.14, -0.27]];

fn drive(amplitudes: [[f64; 2]; 3]) -> FloquetDrive {
    FloquetDrive::with_modes(
        W,
        vec![LightMode::new(
            1,
            Array1::from_iter(amplitudes.map(|[re, im]| Complex::new(re, im))),
        )],
    )
}

fn instantaneous_model([ex, ey, _]: [f64; 3]) -> Rustb::Result<Magnon<3>> {
    let lat = array![[2.0, 0.3, 0.4], [0.7, 1.8, -0.2], [1.1, -0.8, 2.5]];
    let orb = array![[0.10, 0.12, 0.18], [0.38, 0.33, 0.46]];
    let ham_r = array![
        [0isize, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];
    let mut ham = Array3::zeros((ham_r.nrows(), 6, 6));
    for (i, [xy, z]) in [[0.008, 0.017], [0.004, 0.021]].into_iter().enumerate() {
        ham[[0, 3 * i, 3 * i]] = xy;
        ham[[0, 3 * i + 1, 3 * i + 1]] = xy;
        ham[[0, 3 * i + 2, 3 * i + 2]] = z;
    }
    // The three inter-sublattice links are noncoplanar. The same-sublattice
    // link along the skew third lattice vector also drives the normal block.
    for (r, reverse, i, j, xy, dm, zz) in [
        (0, 0, 0, 1, -0.027, 0.004, -0.043),
        (1, 2, 0, 1, -0.019, -0.003, -0.031),
        (3, 4, 0, 1, -0.014, 0.005, -0.025),
        (5, 6, 0, 0, 0.023, 0.006, 0.034),
    ] {
        let [dx, dy] = std::array::from_fn(|cart| {
            (0..3)
                .map(|a| (ham_r[[r, a]] as f64 + orb[[j, a]] - orb[[i, a]]) * lat[[a, cart]])
                .sum::<f64>()
        });
        let (sin, cos) = (ey * dx - ex * dy).sin_cos(); // (E × z) · d
        let exchange = array![[xy, -dm, 0.0], [dm, xy, 0.0], [0.0, 0.0, zz]];
        let rotation = array![[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]];
        let rotated = exchange.dot(&rotation); // J R_z(-theta), in real Cartesian spin space.
        for ((a, b), &value) in rotated.indexed_iter() {
            ham[[r, 3 * i + a, 3 * j + b]] = value;
            ham[[reverse, 3 * j + b, 3 * i + a]] = value;
        }
    }
    Magnon::new(
        lat,
        orb,
        array![1.25, 1.75],
        array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        ham,
        ham_r,
        MagnonReference::collinear(),
    )
}

fn assert_bdg_eq(actual: &Array2<Complex<f64>>, expected: &Array2<Complex<f64>>, context: &str) {
    assert_eq!(actual.dim(), expected.dim(), "{context}");
    for (ij, &actual) in actual.indexed_iter() {
        let expected = expected[ij];
        assert!(
            (actual - expected).norm() < 2e-12,
            "{context}: BdG{ij:?} = {actual}, expected {expected}"
        );
    }
}

#[test]
fn ac_bdg_matches_instantaneous_real_exchange_in_three_dimensions() -> Rustb::Result<()> {
    let dark = instantaneous_model([0.0; 3])?;
    // The largest bond phase amplitude is below 0.62 radians; harmonics
    // beyond ±12 are negligible at the elementwise tolerance.
    let ac = MagnonSambe::new_aharonov_casher(
        &dark,
        &drive(ELECTRIC),
        AXIS,
        &FloquetTruncation::new(6),
    )?;
    for time in [0.0, 0.19, 0.73, 1.41] {
        let (sin, cos) = (W * time).sin_cos();
        let instant = instantaneous_model(ELECTRIC.map(|[re, im]| re * cos + im * sin))?;
        for q in [array![0.0, 0.0, 0.0], array![0.17, -0.29, 0.38]] {
            assert_bdg_eq(
                &ac.bdg_at_time(&q, time)?,
                &instant.gen_bdg(&q)?,
                &format!("q={q:?}, t={time}"),
            );
        }
    }
    Ok(())
}

#[test]
fn ac_time_ordering_matches_independent_runge_kutta() -> Rustb::Result<()> {
    let dark = instantaneous_model([0.0; 3])?;
    let q = array![0.17, -0.29, 0.38];
    let derivative = |t: f64, u: &Array2<Complex<f64>>| -> Rustb::Result<Array2<Complex<f64>>> {
        let (sin, cos) = (W * t).sin_cos();
        let instant = instantaneous_model(ELECTRIC.map(|[re, im]| re * cos + im * sin))?;
        let mut rhs = instant.gen_bdg(&q)?.dot(u);
        for ((row, _), value) in rhs.indexed_iter_mut() {
            *value *= Complex::new(0.0, if row < 2 { -1.0 } else { 1.0 });
        }
        Ok(rhs)
    };
    let mut reference = Array2::eye(4);
    let step = std::f64::consts::TAU / (W * 256.0);
    for sample in 0..256 {
        let t = f64::from(sample) * step;
        let k1 = derivative(t, &reference)?;
        let k2 = derivative(t + step / 2.0, &(&reference + &k1 * (step / 2.0)))?;
        let k3 = derivative(t + step / 2.0, &(&reference + &k2 * (step / 2.0)))?;
        let k4 = derivative(t + step, &(&reference + &k3 * step))?;
        reference += &((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (step / 6.0));
    }
    let error = |steps| -> Rustb::Result<f64> {
        let model = MagnonSambe::new_aharonov_casher(
            &dark,
            &drive(ELECTRIC),
            AXIS,
            &FloquetTruncation::new(6),
        )?
        .with_time_steps(steps)?;
        let result = model.propagator_at(&q)?;
        Ok((result - &reference)
            .iter()
            .map(|z| z.norm())
            .fold(0.0, f64::max))
    };
    let (coarse, fine) = (error(128)?, error(512)?);
    assert!(
        fine < 1e-7 && coarse > fine * 12.0,
        "midpoint residuals coarse={coarse:e}, fine={fine:e}"
    );
    Ok(())
}

#[test]
fn ac_zero_or_parallel_electric_field_leaves_bdg_unchanged() -> Rustb::Result<()> {
    let dark = instantaneous_model([0.0; 3])?;
    let q = array![-0.11, 0.27, 0.35];
    let expected = dark.gen_bdg(&q)?;
    for (name, electric) in [
        ("zero drive", FloquetDrive::new(W)),
        ("parallel field", drive([[0.0; 2], [0.0; 2], [0.37, -0.42]])),
    ] {
        let ac =
            MagnonSambe::new_aharonov_casher(&dark, &electric, AXIS, &FloquetTruncation::new(2))?;
        for time in [0.0, 0.73] {
            assert_bdg_eq(&ac.bdg_at_time(&q, time)?, &expected, name);
        }
    }
    Ok(())
}

#[test]
fn ac_dark_propagation_recovers_both_spin_sectors() -> Rustb::Result<()> {
    let dark = instantaneous_model([0.0; 3])?;
    let q = array![0.17, -0.29, 0.38];
    let ac = MagnonSambe::new_aharonov_casher(
        &dark,
        &FloquetDrive::new(W),
        AXIS,
        &FloquetTruncation::new(0),
    )?;
    let sectors = ac.ac_spin_quasienergies_at(&q)?;
    assert_eq!(sectors[0].len(), 1);
    assert_eq!(sectors[1].len(), 1);
    let mut actual = [sectors[0][0], sectors[1][0]];
    actual.sort_by(f64::total_cmp);
    let mut expected = dark
        .magnon_band_onek(&q)?
        .mapv(|energy| fold_quasienergy(energy, W))
        .to_vec();
    expected.sort_by(f64::total_cmp);
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-12);
    }
    Ok(())
}

#[test]
fn ac_phase_inactive_ising_bond_does_not_call_the_harmonic_backend() -> Rustb::Result<()> {
    let n = 1.0 / 3.0_f64.sqrt();
    let mut ham = Array3::from_elem((3, 3, 3), 0.01);
    ham.index_axis_mut(ndarray::Axis(0), 0).fill(0.0);
    let dark = Magnon::<1>::new(
        array![[1.0]],
        array![[0.0]],
        array![1.0],
        array![[n, n, n]],
        ham,
        array![[0], [1], [-1]],
        MagnonReference::collinear(),
    )?;
    let electric = drive([[0.0; 2], [40000.0, 0.0], [0.0; 2]]);
    let ac =
        MagnonSambe::new_aharonov_casher(&dark, &electric, [n; 3], &FloquetTruncation::new(0))?;
    let q = array![0.25];
    assert_bdg_eq(
        &ac.bdg_at_time(&q, 0.7)?,
        &dark.gen_bdg(&q)?,
        "longitudinal AC null",
    );
    Ok(())
}

#[test]
fn ac_spin_labels_and_folding_follow_the_moment_orientation() -> Rustb::Result<()> {
    let mut ham = Array3::zeros((1, 6, 6));
    // Independent easy-axis sites: one flip on +z carries -1; one on -z carries +1.
    // For S=1 their HP energies are 2 Jzz = 0.2 W and 0.7 W, respectively.
    ham[[0, 2, 2]] = 0.1 * W;
    ham[[0, 5, 5]] = 0.35 * W;
    let magnon = Magnon::<3>::new(
        Array2::eye(3),
        array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]],
        array![1.0, 1.0],
        array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
        ham,
        array![[0, 0, 0]],
        MagnonReference::collinear(),
    )?;
    let q = array![0.17, 0.09, 0.23];
    let ac = MagnonSambe::new_aharonov_casher(
        &magnon,
        &drive(ELECTRIC),
        AXIS,
        &FloquetTruncation::new(2),
    )?;
    let energies = ac.ac_spin_quasienergies_at(&q)?;
    assert!((energies[0][0] - 0.2 * W).abs() < 2e-12);
    assert!((energies[1][0] + 0.3 * W).abs() < 2e-12);
    assert!(ac.with_time_steps(usize::MAX).is_err());
    Ok(())
}

#[test]
fn ac_finite_large_photon_energy_keeps_dark_quasienergy_finite() -> Rustb::Result<()> {
    let mut ham = Array3::zeros((1, 3, 3));
    ham[[0, 2, 2]] = 1.5e307;
    let magnon = Magnon::<1>::new(
        array![[1.0]],
        array![[0.0]],
        array![1.0],
        array![[0.0, 0.0, 1.0]],
        ham,
        array![[0]],
        MagnonReference::collinear(),
    )?;
    let ac = MagnonSambe::new_aharonov_casher(
        &magnon,
        &FloquetDrive::new(1e308),
        AXIS,
        &FloquetTruncation::new(1),
    )?;
    let modes = ac.ac_spin_quasienergies_at(&array![0.25])?;
    assert!((modes[0][0] / 3e307 - 1.0).abs() < 2e-12);
    Ok(())
}

#[test]
fn ac_averages_reciprocal_roundoff_without_hiding_onsite_anisotropy() -> Rustb::Result<()> {
    let mut ham = Array3::zeros((3, 3, 3));
    for a in 0..3 {
        ham[[1, a, a]] = 0.01 + 9e-7;
        ham[[2, a, a]] = 0.01;
    }
    let mut magnon = Magnon::<1>::new(
        array![[1.0]],
        array![[0.0]],
        array![1.0],
        array![[0.0, 0.0, 1.0]],
        ham,
        array![[0], [1], [-1]],
        MagnonReference::collinear(),
    )?;
    let q = array![0.25];
    let ac = MagnonSambe::new_aharonov_casher(
        &magnon,
        &FloquetDrive::new(W),
        AXIS,
        &FloquetTruncation::new(0),
    )?;
    assert_bdg_eq(
        &ac.bdg_at_time(&q, 0.0)?,
        &magnon.gen_bdg(&q)?,
        "reciprocal roundoff",
    );
    assert_eq!(ac.ac_spin_quasienergies_at(&q)?[0].len(), 1);
    magnon.ham.fill(0.0);
    for (a, value) in [1.0 + 1e-11, 1.0 - 1e-11, 1.0].into_iter().enumerate() {
        magnon.ham[[0, a, a]] = value;
    }
    assert!(matches!(
        MagnonSambe::new_aharonov_casher(
            &magnon,
            &FloquetDrive::new(1e-6),
            AXIS,
            &FloquetTruncation::new(0)
        ),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "ac_spin_conservation",
            ..
        })
    ));
    Ok(())
}

#[test]
fn ac_rejects_exchange_without_axial_u1_symmetry() -> Rustb::Result<()> {
    let mut dark = instantaneous_model([0.0; 3])?;
    dark.ham[[0, 0, 3]] += 0.007;
    dark.ham[[0, 3, 0]] += 0.007; // break J_xx = J_yy but preserve the paired transpose
    dark.validate()?;
    assert!(matches!(
        MagnonSambe::new_aharonov_casher(&dark, &drive(ELECTRIC), AXIS, &FloquetTruncation::new(2)),
        Err(TbError::InvalidMagnonInvariant {
            invariant: "ac_spin_conservation",
            ..
        })
    ));
    Ok(())
}

// MGFE6GE6_MAGNON_DIR=target/mgfe6ge6-ac/static-775-cfr101 \
// cargo test --release --features openblas-system --test magnon_aharonov_casher \
//   probe_mgfe6ge6_ac -- --ignored --nocapture
#[test]
#[ignore = "requires an external TB2J export and low-frequency propagation"]
fn probe_mgfe6ge6_ac() -> Rustb::Result<()> {
    const PHOTON_EV: f64 = 0.001;
    let dir = std::env::var("MGFE6GE6_MAGNON_DIR").expect("provide MGFE6GE6_MAGNON_DIR");
    let dark = Magnon::<3>::read(&dir, "mgfe6ge6_static")?;
    assert_eq!(dark.nspin(), 6);
    // Sensitivity check, not a fit to the unpublished phase in Fig. 3(i).
    let a = 0.01;
    let electric = FloquetDrive::with_modes(
        PHOTON_EV,
        vec![LightMode::new(
            1,
            array![
                Complex::new(a, 0.0),
                Complex::new(0.0, a),
                Complex::new(0.0, 0.0),
            ],
        )],
    );
    let q = array![0.13, 0.29, 0.09];
    println!(
        "static Gamma: {:?}",
        dark.magnon_band_onek(&array![0.0, 0.0, 0.0])
    );
    println!("static q={q:?} in eV: {:?}", dark.magnon_band_onek(&q)?);
    let dark_ac = MagnonSambe::new_aharonov_casher(
        &dark,
        &FloquetDrive::new(PHOTON_EV),
        AXIS,
        &FloquetTruncation::new(0),
    )?;
    println!(
        "zero field / folded eV: {:?}",
        dark_ac.ac_spin_quasienergies_at(&q)
    );
    for steps in [128, 256, 512] {
        let ac =
            MagnonSambe::new_aharonov_casher(&dark, &electric, AXIS, &FloquetTruncation::new(3))?
                .with_time_steps(steps)?;
        let bands = ac.ac_spin_quasienergies_at(&q)?;
        assert_eq!([bands[0].len(), bands[1].len()], [3, 3]);
        println!("a={a} A^-1, steps={steps} / folded eV: {bands:?}");
    }
    Ok(())
}

// This scan avoids a branch assignment: it tests the sum over all positive
// modes in each spin sector, modulo W, at locally stable dark momenta only.
#[test]
#[ignore = "requires an external MgFe6Ge6 TB2J export"]
fn scan_mgfe6ge6_ac_helicity() -> Rustb::Result<()> {
    use rayon::prelude::*;
    use std::f64::consts::TAU;

    const W_MATERIAL: f64 = 0.001;
    const N_PHI: usize = 36;
    const RADIUS: f64 = 0.40; // Å^-1, Cartesian
    const A: f64 = 0.01; // normalized AC field, not a fitted material value
    let dir = std::env::var("MGFE6GE6_MAGNON_DIR").expect("provide MGFE6GE6_MAGNON_DIR");
    let dark = Magnon::<3>::read(&dir, "mgfe6ge6_static")?;
    let build = |sign, cutoff, steps| {
        let amplitudes = array![
            Complex::new(A, 0.0),
            Complex::new(0.0, sign * A),
            Complex::new(0.0, 0.0)
        ];
        let drive = FloquetDrive::with_modes(W_MATERIAL, vec![LightMode::new(1, amplitudes)]);
        MagnonSambe::new_aharonov_casher(&dark, &drive, AXIS, &FloquetTruncation::new(cutoff))?
            .with_time_steps(steps)
    };
    let primary = [build(1.0, 3, 512)?, build(-1.0, 3, 512)?];
    let more_steps = [build(1.0, 3, 1024)?, build(-1.0, 3, 1024)?];
    let more_harmonics = [build(1.0, 5, 512)?, build(-1.0, 5, 512)?];
    let spin_sum = |ac: &MagnonSambe<3>, q| -> Rustb::Result<f64> {
        let [down, up] = ac.ac_spin_quasienergies_at(q)?;
        assert_eq!([down.len(), up.len()], [3, 3]);
        Ok(up.sum() - down.sum())
    };
    let helicity_odd = |pair: &[MagnonSambe<3>; 2], q| -> Rustb::Result<f64> {
        let diff = spin_sum(&pair[0], q)? - spin_sum(&pair[1], q)?;
        Ok(0.5 * fold_quasienergy(diff, W_MATERIAL))
    };
    let mut points = Vec::new();
    // Include kz=0, where the leading k-wave contrast must vanish, and
    // opposite kz values to test the expected reflection-odd sign.
    for qz in [-0.15, 0.0, 0.15] {
        for i in 0..N_PHI {
            let phi = TAU * i as f64 / N_PHI as f64;
            let kc = array![RADIUS * phi.cos(), RADIUS * phi.sin(), 0.0];
            let mut q = dark.lat.dot(&kc) / TAU;
            q[2] = qz; // z is the hexagonal lattice c direction in this export
            points.push((qz, i, phi, q));
        }
    }
    println!(
        "# input={dir}, radius={RADIUS} 1/angstrom, photon={W_MATERIAL} eV, normalized field={A} 1/angstrom"
    );
    let values = points
        .par_iter()
        .map(|(qz, i, phi, q)| -> Rustb::Result<_> {
            dark.magnon_band_onek(q)?;
            let value = helicity_odd(&primary, q)?;
            // Include the extremes of the measured C6 residual (3 and 9).
            let refinement = if matches!(i, 0 | 3 | 9) {
                Some((
                    helicity_odd(&more_steps, q)?,
                    helicity_odd(&more_harmonics, q)?,
                ))
            } else {
                None
            };
            Ok((qz, i, phi, value, refinement))
        })
        .collect::<Vec<_>>();
    println!(
        "# fractional_qz, azimuth_index, phi_rad, helicity_odd_spin_sum_ev, more_steps_ev, more_harmonics_ev"
    );
    let mut failed = Vec::new();
    for ((qz, i, _, q), entry) in points.iter().zip(values) {
        match entry {
            Ok((qz, i, phi, value, Some((time, photons)))) => {
                println!("AC_CSV,{qz},{i},{phi:.15e},{value:.15e},{time:.15e},{photons:.15e}")
            }
            Ok((qz, i, phi, value, None)) => println!("AC_CSV,{qz},{i},{phi:.15e},{value:.15e},,"),
            Err(error) => failed.push(format!("qz={qz}, index={i}, q={q:?}: {error}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} of {} requested points unresolved: {failed:#?}",
        failed.len(),
        points.len()
    );
    Ok(())
}
