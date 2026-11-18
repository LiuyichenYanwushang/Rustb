use super::{EffectiveSpinModel, StrongCouplingOptions};
use crate::magnon::floquet::{DrivenExchange, DrivenHubbardParams};
use crate::{FloquetDrive, HubbardModel, LightMode, Model};
use ndarray::array;
use num_complex::Complex;

// A(0) to B(R) in counterclockwise ray order; the three positive NNN
// representatives are delta2-delta3, delta3-delta1, delta1-delta2.
const NN: [[isize; 2]; 3] = [[0, 0], [0, 1], [-1, 1]];
const NNN: [[isize; 2]; 3] = [[1, 0], [-1, 1], [0, -1]];

fn neg([x, y]: [isize; 2]) -> [isize; 2] {
    [-x, -y]
}

// Identity, each NN/NNN bond and each *clockwise* V orbit in the reference
// cell. The residual bound accounts for all terms outside this support.
#[derive(Debug, Default)]
struct CellDensity {
    constant: f64,
    j1: [f64; 3],
    j2: [f64; 6],
    chi: [f64; 6],
    unlisted_bound: f64,
}

fn cell_density(spin: &EffectiveSpinModel<2>) -> CellDensity {
    let mut out = CellDensity {
        constant: spin.constant_ev(),
        ..Default::default()
    };
    for p in spin.pairs() {
        let r = if p.i == 1 && p.j == 0 {
            neg(p.cell)
        } else {
            p.cell
        };
        if p.i != p.j
            && let Some(i) = NN.iter().position(|&x| x == r)
        {
            out.j1[i] += p.exchange_ev;
            continue;
        }
        if p.i == p.j
            && let Some(i) = NNN.iter().position(|&x| x == r || neg(x) == r)
        {
            out.j2[p.i * 3 + i] += p.exchange_ev;
            continue;
        }
        out.unlisted_bound += 0.75 * p.exchange_ev.abs();
    }
    for term in spin.chiralities() {
        let count_a = term.sites.iter().filter(|s| s.orbital == 0).count();
        if count_a == 1 || count_a == 2 {
            // Find the V's middle by sublattice, NOT by production site order.
            let center = count_a - 1;
            let mid = term.sites.iter().position(|s| s.orbital == center).unwrap();
            let [x, y] = term.sites[mid].cell;
            let mut ends = term.sites.iter().filter(|s| s.orbital != center);
            let mut ray = || {
                let [a, b] = ends.next().unwrap().cell;
                let offset = [a - x, b - y];
                let a_to_b = if center == 0 { offset } else { neg(offset) };
                NN.iter().position(|&r| r == a_to_b)
            };
            if let (Some(first), Some(second)) = (ray(), ray())
                && first != second
            {
                // [end_i, center, end_(i+1)] is geometrically clockwise
                // on A and B (B rays are A rays rotated by pi). Moving the
                // center from an outer slot or reversing the ends flips χ.
                let (i, end_sign) = if (first + 1) % 3 == second {
                    (first, 1.0)
                } else {
                    (second, -1.0)
                };
                let mid_sign = if mid == 1 { 1.0 } else { -1.0 };
                out.chi[center * 3 + i] += end_sign * mid_sign * term.coefficient_ev;
                continue;
            }
        }
        out.unlisted_bound += 0.75 * term.coefficient_ev.abs();
    }
    // Honeycomb has no four-cycle. Any four-spin terms are null up to roundoff.
    out.unlisted_bound += spin
        .quartics()
        .iter()
        .map(|q| 0.5625 * q.coefficient_ev.abs())
        .sum::<f64>();
    out
}

impl CellDensity {
    fn closed(j1: f64, j2: f64, chi: f64) -> Self {
        // Pauli blocking: the fully polarized Hubbard state is annihilated by
        // every hop, while each exchange has expectation J/4 and χ vanishes.
        Self {
            constant: -(3.0 * j1 + 6.0 * j2) / 4.0,
            j1: [j1; 3],
            j2: [j2; 6],
            chi: [chi; 6],
            unlisted_bound: 0.0,
        }
    }

    fn assert_matches(&self, reference: &Self, case: &str) {
        // Independent site-major finite-Sambe dimer/V calculation found at
        // most 1.8e-14 eV drift from (H,M)=(6,12) to (8,16) for these cases.
        const TOL: f64 = 5e-13;
        let mut op_bound = (self.constant - reference.constant).abs();
        assert!(
            op_bound <= TOL,
            "{case}: constant: {self:?} vs {reference:?}"
        );
        for (name, got, want) in [
            ("J1", self.j1.as_slice(), reference.j1.as_slice()),
            ("J2", self.j2.as_slice(), reference.j2.as_slice()),
            ("χ (CW)", self.chi.as_slice(), reference.chi.as_slice()),
        ] {
            for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
                let err = (a - b).abs();
                assert!(err <= TOL, "{case}: {name} orbit {i}: {a:e} vs {b:e} eV");
                // ||Si·Sj||=3/4; ||χ||<=6/8 by its six Pauli words.
                op_bound += 0.75 * err;
            }
        }
        let extra = self.unlisted_bound + reference.unlisted_bound;
        assert!(
            extra <= TOL,
            "{case}: unexpected spin operators: bound {extra:e} eV"
        );
        op_bound += extra;
        assert!(
            op_bound <= 5e-12,
            "{case}: full cell operator bound {op_bound:e} eV"
        );
    }
}

#[test]
fn driven_honeycomb_matches_the_signed_closed_spin_operator() {
    let root3 = 3.0_f64.sqrt();
    for (t, u, w) in [(0.1, 1.0, 2.5), (0.05, 1.25, 1.0)] {
        // Unit NN length, two sites per cell at A=(0,0), B=delta1=(0,-1).
        // Cartesian delta2=(sqrt(3)/2,1/2), delta3=(-sqrt(3)/2,1/2).
        let mut electron = Model::<true, 2>::tb_model(
            array![[root3, 0.0], [root3 / 2.0, 1.5]],
            array![[0.0, 0.0], [1.0 / 3.0, -2.0 / 3.0]],
            None,
        )
        .unwrap();
        for r in NN {
            electron.set_hop(t, 0, 1, &array![r[0], r[1]], None);
        }
        let input = HubbardModel::with_uniform_u(electron, u).unwrap();
        for (lambda, xi) in [(0.0, 1.0), (0.4, 1.0), (1.0, 1.0), (0.4, -1.0), (1.0, -1.0)] {
            // Re[a exp(-i theta)] = lambda(sin theta,-xi*cos theta), so
            // C_q(d) = J_q(lambda) exp(+i*xi*q*arg(d)). Relative to the
            // displayed vector potential in floquet.rs this shifts time by
            // pi/Omega; the resulting (-1)^q phases cancel on a closed path.
            let drive = FloquetDrive::with_modes(
                w,
                vec![LightMode::new(
                    1,
                    array![Complex::new(0.0, lambda), Complex::new(-xi * lambda, 0.0),],
                )],
            );
            let run = |order, h| {
                let spin = input
                    .floquet_spin_model(
                        &drive,
                        &StrongCouplingOptions {
                            max_hopping_order: order,
                            hop_harmonic_max: h,
                            intermediate_photon_max: 2 * h,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                let active_m = if lambda == 0.0 {
                    0
                } else if order == 4 {
                    2 * h
                } else {
                    h
                };
                assert_eq!(spin.diagnostics().active_photon_max, active_m);
                cell_density(&spin)
            };
            let closed = |hopping_ev, harmonic_max| {
                DrivenExchange::honeycomb(&DrivenHubbardParams {
                    hopping_ev,
                    interaction_ev: u,
                    photon_ev: w,
                    lambda,
                    helicity: xi,
                    harmonic_max,
                })
                .unwrap()
            };
            let expected = |c: DrivenExchange| CellDensity::closed(c.j1, c.j2, c.chirality);
            let case = format!("t={t} U={u} W={w} lambda={lambda} xi={xi}");

            // H=single-jump cutoff, M=cumulative-window cutoff; the closed
            // oracle instead bounds each intermediate m_i by L and allows
            // jumps up to 2L. Converge each scheme before comparing them.
            let fine = run(4, 8);
            fine.assert_matches(&run(4, 6), &format!("{case}: (H,M)=(6,12) -> (8,16)"));
            let oracle = closed(t, 10);
            expected(oracle).assert_matches(&expected(closed(t, 8)), &format!("{case}: L=8 -> 10"));
            if lambda != 0.0 {
                assert!(xi * oracle.chirality > 5e-5, "{case}: unresolved signed χ");
            }
            // Isolate the closed H2 from its t^2 + t^4 polynomial without
            // reusing the production Bessel cache. The remainder is H4.
            let j1_second = (16.0 * closed(t / 2.0, 10).j1 - oracle.j1) / 3.0;
            run(2, 8).assert_matches(
                &CellDensity::closed(j1_second, 0.0, 0.0),
                &format!("{case}: H2"),
            );
            fine.assert_matches(&expected(oracle), &format!("{case}: H2+H4"));
        }
    }
}
