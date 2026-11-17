//! Floquet-driven exchange and the magnon bands it produces.
//!
//! This module implements the Aharonov–Bohm mechanism of
//! *Odd-Parity Magnons*: a half-filled single-orbital Hubbard model on the
//! honeycomb lattice, driven by circularly polarized light through the Peierls
//! substitution, is reduced to an effective spin Hamiltonian by a
//! Floquet–Sambe strong-coupling (Löwdin) expansion. The result is
//!
//! ```text
//! H = J1 sum_<ij> S_i.S_j + J2 sum_<<ik>> S_i.S_k
//!     + chi sum_(ijk) S_i.(S_j x S_k) + O(t^6),
//! ```
//!
//! with `chi(-xi) = -chi(xi)` while `J1` and `J2` are even in the helicity
//! `xi = +-1`. At quadratic spin-wave order the chirality is a momentum-odd
//! term `D_k = 2 chi S^2 sum_nu sin(k.b_nu)` and the two magnon branches split,
//!
//! ```text
//! hbar omega_{up,down}(k) = sqrt((A_k +- D_k)^2 - |B_k|^2),
//! ```
//!
//! which satisfy `omega_up(k) = omega_down(-k)`: an odd-parity magnon.
//!
//! # Light coupling
//!
//! For `A(t) = A0 (-sin(Omega t), xi cos(Omega t))` the Peierls phase on a bond
//! of length `a` at angle `phi` gives
//!
//! ```text
//! t_ij(t) = sum_m t_ij J_m(lambda) exp(i xi m phi) exp(-i m Omega t),
//! lambda  = e A0 a / hbar = e E a / (hbar Omega).
//! ```
//!
//! The exchange and chirality then follow from the second- and fourth-order
//! Löwdin sums over Floquet harmonics. The `F`, `B` and `G` contributions
//! distinguish the intermediate states (one doublon–holon pair, a nonzero
//! Floquet replica with the model-space return, and the folded normalization),
//! and the lattice multiplicities convert a single three-site `V` cluster into
//! the coefficients below.

use crate::error::{Result, TbError};
use crate::floquet::bessel_j;
use num_complex::Complex;
use std::f64::consts::PI;

/// Physical parameters of the driven half-filled Hubbard model.
#[derive(Clone, Copy, Debug)]
pub struct DrivenHubbardParams {
    /// Hopping amplitude `t` in eV.
    pub hopping_ev: f64,
    /// On-site repulsion `U` in eV.
    pub interaction_ev: f64,
    /// Photon energy `hbar Omega` in eV.
    pub photon_ev: f64,
    /// Dimensionless Peierls amplitude `lambda = e E a / (hbar Omega)`.
    pub lambda: f64,
    /// Optical helicity `xi = +-1`.
    pub helicity: f64,
    /// Truncation of the Floquet harmonic sums.
    pub harmonic_max: isize,
}

impl DrivenHubbardParams {
    /// Validate the parameters and the off-resonant (high-frequency) conditions.
    pub fn validate(&self) -> Result<()> {
        for (name, value) in [
            ("hopping_ev", self.hopping_ev),
            ("interaction_ev", self.interaction_ev),
            ("photon_ev", self.photon_ev),
            ("lambda", self.lambda),
        ] {
            if !value.is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "finite_drive_parameter",
                    message: format!("{name} must be finite, found {value}"),
                });
            }
        }
        if self.interaction_ev <= 0.0 || self.photon_ev <= 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_energy_scale",
                message: "interaction_ev and photon_ev must be positive".into(),
            });
        }
        if (self.helicity.abs() - 1.0).abs() > 1e-12 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "helicity",
                message: format!("helicity must be +1 or -1, found {}", self.helicity),
            });
        }
        if self.harmonic_max < 0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "harmonic_max",
                message: "harmonic_max must be non-negative".into(),
            });
        }
        // The Bessel harmonics must decay inside the retained ladder.
        if self.lambda.abs() > self.harmonic_max as f64 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "harmonic_cutoff",
                message: format!(
                    "|lambda| = {} needs harmonic_max >= that value, found {}",
                    self.lambda.abs(),
                    self.harmonic_max
                ),
            });
        }
        // Off-resonant Floquet expansion: U, 2U and every retained replica away
        // from a charge resonance.
        let cutoff = self.harmonic_max;
        for m in -cutoff..=cutoff {
            if m == 0 {
                continue;
            }
            let detuning = m as f64 * self.photon_ev;
            let scale = detuning.abs().max(self.interaction_ev);
            if (detuning - self.interaction_ev).abs() < 1e-8 * scale
                || (detuning - 2.0 * self.interaction_ev).abs() < 1e-8 * scale
            {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "off_resonant",
                    message: format!("the Floquet replica m = {m} hits a charge resonance"),
                });
            }
        }
        Ok(())
    }
}

/// Effective spin couplings generated by the drive, in eV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrivenExchange {
    /// Nearest-neighbour Heisenberg coupling.
    pub j1: f64,
    /// Next-nearest-neighbour Heisenberg coupling.
    pub j2: f64,
    /// Scalar spin chirality `chi`.
    pub chirality: f64,
}

impl DrivenExchange {
    /// Effective `J1`, `J2` and `chi` of the CPL-driven honeycomb Hubbard model.
    ///
    /// Implements the Floquet–Sambe Löwdin sums of the paper's Supplemental
    /// Material: second order
    /// `J^(2) = 4 t^2 sum_m J_m^2 / (U - m hbar Omega)`, and fourth order
    /// through the `F` (one-doublon), `B` (nonzero-replica) and `G` (folded)
    /// contributions with the honeycomb `V`-cluster multiplicities.
    pub fn honeycomb(params: &DrivenHubbardParams) -> Result<Self> {
        params.validate()?;
        let t = params.hopping_ev;
        let u = params.interaction_ev;
        let omega = params.photon_ev;
        let lambda = params.lambda;
        let xi = params.helicity;
        let m_max = params.harmonic_max;
        let t2 = t * t;
        let t4 = t2 * t2;

        // Second order: J1^(2).
        let mut j1_second = 0.0;
        for m in -m_max..=m_max {
            let jm = bessel_j(m, lambda);
            j1_second += 4.0 * t2 * jm * jm / (u - m as f64 * omega);
        }

        // Fourth order: F, B and G weights.
        let mut j1_f = 0.0;
        let mut j2_f = 0.0;
        let mut chi_f = 0.0;
        let mut j1_b = 0.0;
        let mut j2_b = 0.0;
        let mut chi_b = 0.0;
        let mut g_sum = 0.0;
        // The products below need harmonics up to 2 * m_max.
        let span = 2 * m_max;
        let jm: Vec<f64> = (-span..=span).map(|m| bessel_j(m, lambda)).collect();
        let index = |m: isize| (m + span) as usize;
        for m1 in -m_max..=m_max {
            for m2 in -m_max..=m_max {
                for m3 in -m_max..=m_max {
                    let product = t4
                        * jm[index(m1)]
                        * jm[index(m2 - m1)]
                        * jm[index(m2 - m3)]
                        * jm[index(m3)];
                    let d1 = u - m1 as f64 * omega;
                    let d2 = u - m2 as f64 * omega;
                    let d3 = u - m3 as f64 * omega;
                    let f_weight = product / (d1 * d2 * d3);
                    let phase = PI * (m1 - m2 + m3) as f64 / 3.0;
                    j1_f += 16.0 * f_weight * ((PI * (m1 - m3) as f64 / 3.0).cos() + phase.cos());
                    j2_f += -4.0 * f_weight * phase.cos();
                    chi_f += 8.0 * xi * f_weight * phase.sin();
                    // B: a nonzero even replica returns to the model space.
                    if m2 != 0 && m2 % 2 == 0 {
                        let b_weight =
                            product * sign_pow(m1 - m3) / (d1 * (m2 as f64 * omega) * d3);
                        let b_phase = PI * m2 as f64 / 3.0;
                        j1_b += -16.0 * b_weight * (1.0 + 2.0 * b_phase.cos());
                        j2_b += 8.0 * b_weight * b_phase.cos();
                        chi_b += -16.0 * xi * b_weight * b_phase.sin();
                    }
                }
            }
        }
        for m in -m_max..=m_max {
            for n in -m_max..=m_max {
                let dm = u - m as f64 * omega;
                let dn = u - n as f64 * omega;
                g_sum += t4
                    * jm[index(m)]
                    * jm[index(m)]
                    * jm[index(n)]
                    * jm[index(n)]
                    * (1.0 / (dm * dm * dn) + 1.0 / (dm * dn * dn));
            }
        }
        let j1_g = -24.0 * g_sum;
        let j2_g = 4.0 * g_sum;
        Ok(Self {
            j1: j1_second + j1_f + j1_b + j1_g,
            j2: j2_f + j2_b + j2_g,
            chirality: chi_f + chi_b,
        })
    }

    /// Honeycomb spin-wave functions `A_k`, `B_k`, `D_k` of the paper.
    ///
    /// `q` is in fractional reciprocal coordinates of the hexagonal cell with
    /// the nearest-neighbour vectors of the Supplemental Material, and the
    /// returned `(a_k, |b_k|, d_k)` uses the units of the couplings (eV).
    pub fn honeycomb_functions(
        &self,
        spin: f64,
        anisotropy_ev: f64,
        q: &[f64; 2],
    ) -> (f64, f64, f64) {
        // The Supplemental Material's vectors in fractional coordinates of the
        // hexagonal cell (a1 = delta2 - delta3, a2 = delta2 - delta1):
        // delta1 = a(0,-1), delta2 = a(sqrt3/2, 1/2), delta3 = a(-sqrt3/2, 1/2),
        // b1 = delta2 - delta3, b2 = delta3 - delta1, b3 = delta1 - delta2.
        let delta = [
            [1.0 / 3.0, -2.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [-2.0 / 3.0, 1.0 / 3.0],
        ];
        let bl: [[f64; 2]; 3] = [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]];
        let mut sum_b = Complex::new(0.0, 0.0);
        for d in delta {
            let phase = std::f64::consts::TAU * (q[0] * d[0] + q[1] * d[1]);
            sum_b += Complex::from_polar(1.0, phase);
        }
        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for b in bl {
            let phase = std::f64::consts::TAU * (q[0] * b[0] + q[1] * b[1]);
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }
        let a_k = 3.0 * spin * self.j1 + 2.0 * spin * self.j2 * (sum_cos - 3.0)
            - 2.0 * anisotropy_ev * spin;
        let b_k = spin * self.j1 * sum_b;
        let d_k = 2.0 * self.chirality * spin * spin * sum_sin;
        (a_k, b_k.norm(), d_k)
    }

    /// Magnon energies of the two spin branches at `q`, in eV.
    ///
    /// Returns `(omega_up, omega_down)`; a branch whose bracket
    /// `(A +- D)^2 - |B|^2` is negative is unphysical (unstable reference
    /// state) and reported as `NaN`.
    pub fn honeycomb_dispersion(&self, spin: f64, anisotropy_ev: f64, q: &[f64; 2]) -> (f64, f64) {
        let (a_k, b_abs, d_k) = self.honeycomb_functions(spin, anisotropy_ev, q);
        let branch = |sign: f64| {
            let dressed = (a_k + sign * d_k).powi(2);
            let value = dressed - b_abs * b_abs;
            if value >= 0.0 {
                value.sqrt()
            } else if -value <= 1e-10 * (dressed + b_abs * b_abs).max(f64::MIN_POSITIVE) {
                // At a gapless point the two squares differ by round-off only.
                0.0
            } else {
                f64::NAN
            }
        };
        (branch(1.0), branch(-1.0))
    }
}

fn sign_pow(exponent: isize) -> f64 {
    if exponent.rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(lambda: f64, helicity: f64) -> DrivenHubbardParams {
        DrivenHubbardParams {
            hopping_ev: 0.1,
            interaction_ev: 1.0,
            photon_ev: 2.5,
            lambda,
            helicity,
            harmonic_max: 12,
        }
    }

    #[test]
    fn golden_couplings_match_the_reference_sums() {
        // Independent 50-digit mpmath evaluation of the printed fourth-order
        // sums (t = 0.05, U = 1.25, hbar Omega = 1, xi = -1, cutoff 16).
        let base = DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 1.0,
            lambda: 0.0,
            helicity: -1.0,
            harmonic_max: 16,
        };
        for (lambda, j1, j2, chi) in [
            (
                0.1,
                8.0414311292351087e-3,
                1.1152106530260925e-5,
                -6.1434755970829542e-6,
            ),
            (
                0.4,
                9.3080577517533330e-3,
                -4.9842594659995401e-6,
                -8.9948130318571188e-5,
            ),
            (
                1.0,
                1.3188931012532839e-2,
                9.1315624886234976e-5,
                -3.3449034397063350e-4,
            ),
        ] {
            let params = DrivenHubbardParams { lambda, ..base };
            let exchange = DrivenExchange::honeycomb(&params).unwrap();
            assert!(
                (exchange.j1 - j1).abs() < 1e-15,
                "lambda = {lambda}: J1 = {:e}",
                exchange.j1
            );
            assert!(
                (exchange.j2 - j2).abs() < 1e-15,
                "lambda = {lambda}: J2 = {:e}",
                exchange.j2
            );
            assert!(
                (exchange.chirality - chi).abs() < 1e-15,
                "lambda = {lambda}: chi = {:e}",
                exchange.chirality
            );
        }
    }

    #[test]
    fn harmonic_sums_converge_with_the_cutoff() {
        let base = DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 1.0,
            lambda: 5.0,
            helicity: 1.0,
            harmonic_max: 8,
        };
        let coarse = DrivenExchange::honeycomb(&base).unwrap();
        let fine = DrivenExchange::honeycomb(&DrivenHubbardParams {
            harmonic_max: 16,
            ..base
        })
        .unwrap();
        assert!(
            (coarse.j1 - fine.j1).abs() < 1e-6,
            "{:e} {:e}",
            coarse.j1,
            fine.j1
        );
        assert!((coarse.chirality - fine.chirality).abs() < 1e-6);
    }

    #[test]
    fn dispersion_matches_the_analytic_values_at_gamma_and_k() {
        let params = DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.4,
            helicity: 1.0,
            harmonic_max: 8,
        };
        let exchange = DrivenExchange::honeycomb(&params).unwrap();
        let spin = 1.5;
        let anisotropy = -0.01;
        // Gamma: |B_k| = 3 S J1, D_k = 0, A_k = 3 S J1 - 2 K S.
        let (a_k, b_abs, d_k) = exchange.honeycomb_functions(spin, anisotropy, &[0.0, 0.0]);
        assert!((a_k - (3.0 * spin * exchange.j1 - 2.0 * anisotropy * spin)).abs() < 1e-12);
        assert!((b_abs - 3.0 * spin * exchange.j1).abs() < 1e-12);
        assert!(d_k.abs() < 1e-15);
        // K: |B_k| = 0, D_k = 3 sqrt(3) chi S^2.
        let (a_k, b_abs, d_k) =
            exchange.honeycomb_functions(spin, anisotropy, &[1.0 / 3.0, 2.0 / 3.0]);
        assert!(b_abs.abs() < 1e-12, "|B_K| = {b_abs}");
        assert!(
            (d_k - 3.0 * 3.0f64.sqrt() * exchange.chirality * spin * spin).abs() < 1e-12,
            "D_K = {d_k}"
        );
        // At K, sum_nu cos(k.b_nu) = -3/2, so the J2 term is -9 S J2.
        assert!(
            (a_k - (3.0 * spin * exchange.j1 - 9.0 * spin * exchange.j2 - 2.0 * anisotropy * spin))
                .abs()
                < 1e-12,
            "A_K = {a_k}"
        );
    }

    #[test]
    fn gapless_point_is_not_reported_as_nan() {
        let params = DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.0,
            helicity: 1.0,
            harmonic_max: 8,
        };
        let exchange = DrivenExchange::honeycomb(&params).unwrap();
        for spin in [0.5, 1.5, 2.5] {
            let (up, down) = exchange.honeycomb_dispersion(spin, 0.0, &[0.0, 0.0]);
            assert!(
                up.is_finite() && down.is_finite(),
                "spin {spin}: {up} {down}"
            );
            assert!(
                up.abs() < 1e-12 && down.abs() < 1e-12,
                "spin {spin}: {up} {down}"
            );
        }
    }

    #[test]
    fn validate_rejects_bad_parameters() {
        let base = params(0.4, 1.0);
        // A charge resonance.
        let resonant = DrivenHubbardParams {
            photon_ev: 1.0,
            ..base
        };
        assert!(DrivenExchange::honeycomb(&resonant).is_err());
        // A cutoff too small for the Bessel ladder.
        let truncated = DrivenHubbardParams {
            harmonic_max: 0,
            ..base
        };
        assert!(DrivenExchange::honeycomb(&truncated).is_err());
        // Non-finite and negative scales.
        assert!(
            DrivenExchange::honeycomb(&DrivenHubbardParams {
                interaction_ev: 0.0,
                ..base
            })
            .is_err()
        );
        assert!(
            DrivenExchange::honeycomb(&DrivenHubbardParams {
                helicity: 0.5,
                ..base
            })
            .is_err()
        );
    }

    #[test]
    fn static_limit_reproduces_the_perturbative_exchange() {
        let p = params(0.0, 1.0);
        let exchange = DrivenExchange::honeycomb(&p).unwrap();
        let expected_j1 = 4.0 * 0.01 / 1.0 - 16.0 * 1e-4 / 1.0;
        let expected_j2 = 4.0 * 1e-4 / 1.0;
        assert!(
            (exchange.j1 - expected_j1).abs() < 1e-15,
            "J1 = {} expected {expected_j1}",
            exchange.j1
        );
        assert!(
            (exchange.j2 - expected_j2).abs() < 1e-18,
            "J2 = {} expected {expected_j2}",
            exchange.j2
        );
        assert!(exchange.chirality.abs() < 1e-12);
    }

    #[test]
    fn helicity_flips_the_chirality_only() {
        let light = DrivenExchange::honeycomb(&params(0.4, 1.0)).unwrap();
        let other = DrivenExchange::honeycomb(&params(0.4, -1.0)).unwrap();
        assert!((light.j1 - other.j1).abs() < 1e-14);
        assert!((light.j2 - other.j2).abs() < 1e-14);
        assert!((light.chirality + other.chirality).abs() < 1e-14);
        assert!(light.chirality.abs() > 1e-6);
    }

    #[test]
    fn drive_splits_the_magnon_branches() {
        let undriven = DrivenExchange::honeycomb(&params(0.0, 1.0)).unwrap();
        let driven = DrivenExchange::honeycomb(&params(0.4, 1.0)).unwrap();
        // K = (1/3, 2/3) and K' = (2/3, 1/3) in the reciprocal basis built from
        // b1 = delta2 - delta3, b2 = delta3 - delta1.
        let k = [1.0 / 3.0, 2.0 / 3.0];
        let k_prime = [2.0 / 3.0, 1.0 / 3.0];
        let (up, down) = undriven.honeycomb_dispersion(1.5, -0.01, &k);
        assert!((up - down).abs() < 1e-12, "undriven split {}, {}", up, down);
        let (up, down) = driven.honeycomb_dispersion(1.5, -0.01, &k);
        let split = up - down;
        assert!(split.abs() > 1e-6, "no splitting at K: {split:e}");
        // Opposite signs at the two valleys, and odd parity overall.
        let (up_prime, down_prime) = driven.honeycomb_dispersion(1.5, -0.01, &k_prime);
        let split_prime = up_prime - down_prime;
        assert!(
            (split + split_prime).abs() < 1e-3 * split.abs(),
            "K and K' splits {split:e} {split_prime:e}"
        );
        let (up_neg, down_neg) = driven.honeycomb_dispersion(1.5, -0.01, &[-k[0], -k[1]]);
        assert!(
            ((up - down) + (up_neg - down_neg)).abs() < 1e-9 * split.abs().max(1.0),
            "odd parity violated"
        );
    }
}
