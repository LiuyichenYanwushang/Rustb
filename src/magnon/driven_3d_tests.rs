//! Driven checks on a two-layer noncoplanar loop and a periodic simple-cubic lattice, not material benchmarks.
use super::{PeriodicSpinSite, StrongCouplingOptions};
use crate::kpath::Kpath;
use crate::magnon::{MagnonReference, MagnonSolve};
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::{Array2, arr1, array};
use num_complex::Complex64;
use std::f64::consts::TAU;

const T: f64 = 0.1;
const CELLS: [[isize; 3]; 4] = [[0; 3], [0; 3], [0, 0, 1], [0, 0, 1]];
const TRIPLES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
const QUARTICS: [[[usize; 2]; 2]; 3] = [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]];

fn on_ring<const N: usize>(sites: [PeriodicSpinSite<3>; N]) -> bool {
    let origin = sites[0];
    sites.iter().all(|s| {
        (0..3).all(|a| s.cell[a] - origin.cell[a] == CELLS[s.orbital][a] - CELLS[origin.orbital][a])
    })
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 2e-12,
        "{what}: {actual:e} vs {expected:e} eV"
    );
}

#[test]
fn driven_nonplanar_ring_uses_z_polarization_at_fourth_order() {
    // One unwrapped ring has Cartesian vertices (1,0,3.5), (1,1,3.5),
    // (2,1,4.5), (0,0,4.5): scalar triple product of the three relative
    // vectors is -2. Both layers have an in-plane bond (different directions),
    // and the two interlayer bonds actually cross the z-cell boundary.
    let mut model = Model::<true, 3>::tb_model(
        array![[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]],
        array![
            [0.25, 0., 0.875],
            [0.25, 0.25, 0.875],
            [0.5, 0.25, 0.125],
            [0., 0., 0.125]
        ],
        None,
    )
    .unwrap();
    for (i, weight) in [1.0, 0.8, 0.6, 0.9].into_iter().enumerate() {
        let j = (i + 1) % 4;
        let r: [isize; 3] = std::array::from_fn(|a| CELLS[j][a] - CELLS[i][a]);
        model.set_hop(T * weight, i, j, &arr1(&r), None);
    }
    // One y-only bridge (orthogonal to the drive) per ring produces real
    // q_y dispersion; q_z probes the phase of translated intra-ring terms.
    model.set_hop(T * 0.5, 1, 0, &array![0, 1, 0], None);
    let input = HubbardModel::with_uniform_u(model, 4.0).unwrap();
    let opts = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 4,
        intermediate_photon_max: 8,
        ..Default::default()
    };

    // Independent 128-point Peierls DFT and dense Fock/Sambe calculation
    // from scripts/floquet_spin_reference/nonplanar_reference.py
    // (H=4, M=8, min |E_Q|=1 eV), which also checks the H=6 sweep.
    // Values are the signed per-ring H4 coefficients in units of t^4;
    // all four chi slots follow increasing orbital order, not spatial order.
    // At order 4 the cut-edge bridge enters only dimers or two-bond wedges.
    // These have at most one driven leg; a shift of time makes its linear
    // drive time-reversal symmetric, so they cannot add either chi or quartics.
    let sambe_chi = [
        -0.152_363_035_479_024,
        -0.152_363_035_479_024,
        0.088_061_277_023_041,
        0.068_251_330_817_347,
    ];
    for (z_sign, ring, a02) in [
        (
            1.0,
            0.162_351_302_870_054,
            Complex64::new(9.460_309_807_891_381e-7, -9.330_077_974_342_236e-6),
        ),
        (
            0.0,
            0.282_014_334_032_602,
            Complex64::new(7.631_633_406_619_847e-6, -7.557_586_847_121_914e-6),
        ),
        (
            -1.0,
            0.162_351_302_870_054,
            Complex64::new(9.709_120_034_957_458e-6, -1.100_989_608_433_093e-6),
        ),
    ] {
        // Re[a exp(-iθ)] = (0.4 cos θ, 0, 0.35 z_sign sin θ). In
        // particular z_sign=0 drops only the third drive component.
        let drive = FloquetDrive::with_modes(
            5.0,
            vec![LightMode::new(
                1,
                array![
                    Complex64::new(0.4, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.35 * z_sign)
                ],
            )],
        );
        let spin = input.floquet_spin_model(&drive, &opts).unwrap();
        assert_eq!(spin.diagnostics().active_photon_max, 8);
        assert_eq!(spin.diagnostics().nearest_detuning_ev, Some(1.0));

        let mut chi = [None; 4];
        for c in spin.chiralities() {
            let key = c.sites.map(|s| s.orbital);
            if let Some(i) = TRIPLES
                .iter()
                .position(|&ids| ids == key && on_ring(c.sites))
            {
                assert!(chi[i].replace(c.coefficient_ev).is_none());
            } else {
                assert_close(c.coefficient_ev, 0.0, "off-ring chi");
            }
        }
        for (got, want) in chi.into_iter().zip(sambe_chi) {
            assert_close(got.unwrap_or(0.0), z_sign * want * T.powi(4), "ring chi");
        }

        let mut quartic = [None; 3];
        for c in spin.quartics() {
            let [[a, b], [d, e]] = c.pairs;
            let key = c.pairs.map(|pair| pair.map(|s| s.orbital));
            if let Some(i) = QUARTICS
                .iter()
                .position(|&ids| ids == key && on_ring([a, b, d, e]))
            {
                assert!(quartic[i].replace(c.coefficient_ev).is_none());
            } else {
                assert_close(c.coefficient_ev, 0.0, "off-ring quartic");
            }
        }
        for (got, want) in quartic.into_iter().zip([ring, -ring, ring]) {
            assert_close(got.unwrap(), want * T.powi(4), "ring quartic");
        }

        let axes = array![[0., 0., 1.], [0., 0., -1.], [0., 0., 1.], [0., 0., -1.]];
        let magnon = spin.magnon(&axes, MagnonReference::collinear()).unwrap();
        let q = array![0.19, 0.23, 0.37];
        // Independent BdG baseline: linearize the spin-coherent-state
        // expectation of each Fock/Sambe spin matrix using direct flip matrix
        // elements, then Fourier transform with exp(+2πiq·(R_j-R_i))
        // and local transverse axes (x, ±y). The matrix elements keep the
        // sign and translated phase that a sorted band comparison can lose.
        let bdg = magnon.gen_bdg(&q).unwrap();
        assert_close((bdg[[0, 2]] - a02).norm(), 0.0, "signed finite-q BdG A02");

        if z_sign > 0.0 {
            let b12 = Complex64::new(-0.001_364_904_177_041, 0.001_461_532_267_089);
            assert_close((bdg[[1, 6]] - b12).norm(), 0.0, "signed finite-q BdG B12");
            // The same independent Hessian (with bridge dimer and its four
            // connected wedges) has minimum eigenvalues >4e-4 eV here.
            for (qy, expected) in [
                (
                    0.23,
                    [
                        0.001_350_735_559_627,
                        0.001_506_471_209_088,
                        0.006_168_625_739_720,
                        0.006_326_905_107_307,
                    ],
                ),
                (
                    0.41,
                    [
                        0.001_709_665_346_860,
                        0.001_971_876_119_456,
                        0.006_919_000_979_183,
                        0.007_183_732_729_073,
                    ],
                ),
            ] {
                let bands = magnon.magnon_band_onek(&array![q[0], qy, q[2]]).unwrap();
                assert_eq!(bands.len(), 4);
                for (got, want) in bands.into_iter().zip(expected) {
                    assert_close(got, want, "stable finite-q magnon frequency");
                }
            }
        }
    }
}

#[test]
fn driven_simple_cubic_magnon_path_and_dos_match_analytic_band() {
    let (u, photon_ev, amplitude) = (4.0, 5.0, 1.5);
    let hopping = [T, 0.8 * T, 0.6 * T];
    let mut model =
        Model::<true, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None).unwrap();
    for (axis, t) in hopping.into_iter().enumerate() {
        let mut r = [0; 3];
        r[axis] = 1;
        model.set_hop(t, 0, 0, &arr1(&r), None);
    }
    let drive = FloquetDrive::with_modes(
        photon_ev,
        vec![LightMode::new(
            1,
            arr1(&[Complex64::new(amplitude, 0.0); 3]),
        )],
    );
    let cutoff = 4;
    let spin = HubbardModel::with_uniform_u(model, u)
        .unwrap()
        .floquet_spin_model(
            &drive,
            &StrongCouplingOptions {
                max_hopping_order: 2,
                hop_harmonic_max: cutoff,
                intermediate_photon_max: cutoff,
                ..Default::default()
            },
        )
        .unwrap();
    let magnon = spin
        .magnon(&array![[0.0, 0.0, 1.0]], MagnonReference::collinear())
        .unwrap();

    // Independent Peierls quadrature: K_a = 4t_a² Σ_m |C_m|²/(U+mW),
    // with C_m the Fourier coefficient of exp(-i A cos θ) on each unit bond.
    let virtual_sum = (-cutoff..=cutoff)
        .map(|m| {
            let c = (0..64)
                .map(|n| {
                    let theta = TAU * n as f64 / 64.0;
                    Complex64::from_polar(1.0, m as f64 * theta - amplitude * theta.cos())
                })
                .sum::<Complex64>()
                / 64.0;
            c.norm_sqr() / (u + m as f64 * photon_ev)
        })
        .sum::<f64>();
    let exchange = hopping.map(|t| 4.0 * t * t * virtual_sum);
    assert!(exchange.iter().all(|&k| k < 0.0));
    // The S=1/2 ferromagnet's one-spin-flip band is -Σ_a K_a(1 - cos(2πq_a)).
    let frequency = |q: [f64; 3]| -> f64 {
        q.into_iter()
            .zip(exchange)
            .map(|(q, k)| -k * (1.0 - (TAU * q).cos()))
            .sum()
    };

    // Γ→Z changes only q_z; the distinct exchanges also expose axis swaps.
    let path = array![[0., 0., 0.], [0., 0., 0.5], [0.5, 0., 0.5], [0.5, 0.5, 0.5]];
    let (points, distances, nodes) = magnon.k_path(&path, 7).unwrap();
    assert_eq!(
        points,
        array![
            [0., 0., 0.],
            [0., 0., 0.25],
            [0., 0., 0.5],
            [0.25, 0., 0.5],
            [0.5, 0., 0.5],
            [0.5, 0.25, 0.5],
            [0.5, 0.5, 0.5]
        ]
    );
    assert_eq!(distances, array![0., 0.25, 0.5, 0.75, 1., 1.25, 1.5]);
    assert_eq!(nodes, array![0., 0.5, 1., 1.5]);
    let bands = magnon.magnon_band_all(&points).unwrap();
    assert_eq!(bands.dim(), (7, 1));
    for (row, q) in points.rows().into_iter().enumerate() {
        assert_close(bands[[row, 0]], frequency([q[0], q[1], q[2]]), "path band");
    }

    // Enumerate the discrete rectangular mesh independently, retaining every
    // q point (including coincident energies) with weight 1/(nx ny nz).
    let mesh = [2, 3, 4];
    let [nx, ny, nz] = mesh;
    let mut energies = Vec::new();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                energies.push(frequency([
                    x as f64 / nx as f64,
                    y as f64 / ny as f64,
                    z as f64 / nz as f64,
                ]));
            }
        }
    }
    let (sigma, bins) = (0.002, 201);
    let w_min = -8.0 * sigma;
    let w_max = -2.0 * exchange.iter().sum::<f64>() + 8.0 * sigma;
    let dw = (w_max - w_min) / (bins - 1) as f64;
    let (omega, dos) = magnon
        .magnon_dos(&arr1(&mesh), w_min, w_max, bins, sigma)
        .unwrap();
    assert_eq!((omega.len(), dos.len()), (bins, bins));
    let weight = 1.0 / ((nx * ny * nz) as f64 * sigma * TAU.sqrt());
    for (i, (&w, &actual)) in omega.iter().zip(&dos).enumerate() {
        assert_close(w, w_min + i as f64 * dw, "DOS grid");
        let expected = weight
            * energies
                .iter()
                .map(|&e| (-0.5 * ((w - e) / sigma).powi(2)).exp())
                .sum::<f64>();
        assert!(
            (actual - expected).abs() < 1e-9,
            "DOS at {w:e} eV: {actual:e} vs {expected:e}"
        );
    }
    let total = dos.sum() * dw;
    assert!((total - 1.0).abs() < 1e-12, "integrated DOS = {total}");
}
