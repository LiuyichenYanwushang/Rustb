use Rustb::*;
use ndarray::{arr1, array};
use num_complex::Complex;
use std::collections::HashSet;

fn tetrahedral_a2g_modes_like_recipe(
    harmonic: isize,
    momentum_basis_cartesian: &ndarray::Array2<f64>,
    a_complex_scale: f64,
    a2g_domain: f64,
) -> Result<Vec<LightMode>> {
    if harmonic <= 0 {
        return Err(Rustb::TbError::Other(
            "A2g mode construction expects positive harmonic (q->-q is conjugate)".to_string(),
        ));
    }
    if momentum_basis_cartesian.nrows() != 3 || momentum_basis_cartesian.ncols() != 3 {
        return Err(Rustb::TbError::Other(format!(
            "A2g tetrahedral mode basis must be 3x3, got {:?}x{:?}",
            momentum_basis_cartesian.nrows(),
            momentum_basis_cartesian.ncols()
        )));
    }
    if !a2g_domain.is_finite() || (a2g_domain != 1.0 && a2g_domain != -1.0) {
        return Err(Rustb::TbError::Other(
            "A2g domain parameter should be either +1 or -1".to_string(),
        ));
    }
    if !a_complex_scale.is_finite() {
        return Err(Rustb::TbError::Other(
            "A2g amplitude scale must be a finite number".to_string(),
        ));
    }

    let mut modes = Vec::with_capacity(8);
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
    let sign_triplets = [
        [1_isize, 1_isize, 1_isize],
        [1_isize, 1_isize, -1_isize],
        [1_isize, -1_isize, 1_isize],
        [1_isize, -1_isize, -1_isize],
        [-1_isize, 1_isize, 1_isize],
        [-1_isize, 1_isize, -1_isize],
        [-1_isize, -1_isize, 1_isize],
        [-1_isize, -1_isize, -1_isize],
    ];

    for label in sign_triplets {
        let direction = momentum_basis_cartesian
            .rows()
            .into_iter()
            .zip(label.iter().copied())
            .fold(array![0.0, 0.0, 0.0], |mut acc, (axis, sign)| {
                acc[0] += axis[0] * sign as f64;
                acc[1] += axis[1] * sign as f64;
                acc[2] += axis[2] * sign as f64;
                acc
            });

        let norm = (direction.iter().map(|value| value * value).sum::<f64>()).sqrt();
        if norm <= 0.0 {
            return Err(Rustb::TbError::Other(
                "A2g mode direction has zero norm after basis projection".to_string(),
            ));
        }

        let incident = IncidentBasis::from_direction(&direction)?;
        let helicity = a2g_domain * (label.iter().product::<isize>() as f64);
        let circular = incident.polarization([
            Complex::new(inv_sqrt2, 0.0),
            Complex::new(0.0, helicity * inv_sqrt2),
        ]);
        modes.push(LightMode::new(
            harmonic,
            circular.mapv(|z| a_complex_scale * z),
            label,
        ));
    }

    Ok(modes)
}

#[test]
fn tetrahedral_a2g_modes_builds_eight_body_diagonal_modes_with_expected_signs() {
    let basis = array![[0.05, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.08],];
    let modes = tetrahedral_a2g_modes_like_recipe(1, &basis, 0.123, 1.0).unwrap();

    assert_eq!(modes.len(), 8);
    let mut seen = HashSet::new();
    for mode in modes {
        let label = mode.momentum_label.clone();
        assert_eq!(label.len(), 3);
        for value in label.iter() {
            assert_eq!(value.abs(), 1_isize);
        }
        assert!(seen.insert(label.clone()));
        let sign = (label.iter().product::<isize>() as f64) * mode.harmonic as f64;
        let direction = basis.rows().into_iter().zip(label.iter()).fold(
            array![0.0, 0.0, 0.0],
            |mut direction, (axis, factor)| {
                direction += &(axis.to_owned() * (*factor as f64));
                direction
            },
        );
        let incident = IncidentBasis::from_direction(&direction).unwrap();
        let eta = if sign > 0.0 { 1.0 } else { -1.0 };
        let c1_expected = Complex::new(0.123 / (2.0_f64).sqrt(), 0.0);
        let c2_expected = Complex::new(0.0, eta * 0.123 / (2.0_f64).sqrt());
        let expected = incident.polarization([c1_expected, c2_expected]);
        let diff_norm = (0..3)
            .map(|index| (mode.a_complex[index] - expected[index]).norm_sqr())
            .sum::<f64>()
            .sqrt();
        assert!(
            diff_norm < 1.0e-14,
            "unexpected polarization reconstruction for {label:?}"
        );
        assert_eq!(mode.harmonic, 1);
    }
    assert_eq!(seen.len(), 8);
}

#[test]
fn tetrahedral_a2g_domains_are_exact_helicity_partners() {
    let basis = array![[0.05, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.08],];
    let positive = tetrahedral_a2g_modes_like_recipe(1, &basis, 0.07, 1.0).unwrap();
    let negative = tetrahedral_a2g_modes_like_recipe(1, &basis, 0.07, -1.0).unwrap();

    assert_eq!(positive.len(), negative.len());
    for (mode_plus, mode_minus) in positive.iter().zip(negative.iter()) {
        assert_eq!(mode_plus.momentum_label, mode_minus.momentum_label);
        for index in 0..3 {
            assert!(
                (mode_plus.a_complex[index].re - mode_minus.a_complex[index].re).abs() < 1.0e-14
            );
            assert!(
                (mode_plus.a_complex[index].im + mode_minus.a_complex[index].im).abs() < 1.0e-14
            );
        }
    }
}

#[test]
fn floquet_incident_basis_api_example() {
    let lat = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let orb = array![[0.0, 0.0, 0.0]];
    let mut model = Model::<false, 3>::tb_model(lat, orb, None).unwrap();
    model.set_hop(-1.0_f64, 0, 0, &arr1(&[1isize, 0, 0]), None);
    model.set_hop(-0.8_f64, 0, 0, &arr1(&[0isize, 1, 0]), None);
    model.set_hop(-0.6_f64, 0, 0, &arr1(&[0isize, 0, 1]), None);

    let incident = IncidentBasis::from_direction(&arr1(&[0.0, 0.0, 1.0])).unwrap();
    let circular = incident.polarization([
        Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex::new(0.0, 1.0 / 2.0_f64.sqrt()),
    ]);
    let drive = FloquetDrive::uniform(
        0.8,
        vec![LightMode::uniform(1, circular.mapv(|z| 0.12 * z))],
    );
    let trunc = FloquetTruncation::new(1, 128);
    let k = arr1(&[0.2, 0.1, 0.0]);

    let hf = model
        .floquet_ham_onek(&k, &drive, &trunc, Gauge::Lattice)
        .unwrap();
    assert_eq!(hf.dim(), (3, 3));

    let mut max_diff = 0.0f64;
    for i in 0..hf.nrows() {
        for j in 0..hf.ncols() {
            max_diff = max_diff.max((hf[[i, j]] - hf[[j, i]].conj()).norm());
        }
    }
    assert!(max_diff < 1e-11, "max hermiticity error = {max_diff:e}");

    let qe = model
        .floquet_quasienergy_onek(&k, &drive, &trunc, Gauge::Lattice)
        .unwrap();
    assert_eq!(qe.len(), 3);
    for &x in qe.iter() {
        assert!(x >= -0.5 * drive.omega0_ev - 1e-12 && x < 0.5 * drive.omega0_ev + 1e-12);
    }
}
