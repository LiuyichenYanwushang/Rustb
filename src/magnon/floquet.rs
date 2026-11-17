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

// ---------------------------------------------------------------------------
// Floquet–Sambe model and quasienergies of a driven magnon Hamiltonian
// ---------------------------------------------------------------------------

use crate::floquet::{
    FloquetDrive, FloquetEffectiveOptions, FloquetTruncation, bessel_peierls_coeffs,
};
use crate::magnon::Magnon as MagnonModel;
use ndarray::{Array1, Array2, Array3, Axis};
use ndarray_linalg::Eig;

/// Floquet–Sambe model of a periodically driven magnon Hamiltonian.
///
/// The drive dresses every exchange bond with the Peierls phase the electronic
/// module uses,
///
/// ```text
/// J_ij(R, t) = J_ij(R) exp[-i a(t).d_ijR],
/// ```
///
/// whose harmonics are the Bessel coefficients `C_m(d)` of the drive. The
/// quasienergies are obtained from the one-period propagator of the bosonic
/// Bogoliubov–de Gennes dynamics,
///
/// ```text
/// i d Psi / dt = g H(t) Psi,   g = diag(1, -1),
/// ```
///
/// rather than from a Hermitian diagonalization: only eigenvectors with
/// positive metric norm are physical, and at a degenerate (Goldstone) mode the
/// metric-weighted matrix `g H` is defective, so a naive eigen-decomposition of
/// `g S` splits the photon replicas of a zero mode spuriously.
pub struct MagnonSambe<const DIM: usize = 3> {
    magnon: MagnonModel<DIM>,
    /// Exchange harmonics `J^(m)(R)`, indexed by `m + harmonic_span`.
    harmonics: Vec<Array3<Complex<f64>>>,
    omega_ev: f64,
    n_max: isize,
    harmonic_span: isize,
    n_time: usize,
}

/// Periodic driving of a magnon model, mirroring the electronic
/// [`crate::floquet::Floquet`] trait.
pub trait MagnonFloquet {
    /// Static Sambe-space model produced by [`MagnonFloquet::floquet_model`].
    type FloquetModel;

    /// Build the Floquet–Sambe model of a periodically driven magnon.
    fn floquet_model(
        &self,
        drive: &FloquetDrive,
        truncation: &FloquetTruncation,
    ) -> Result<Self::FloquetModel>;
}

impl<const DIM: usize> MagnonFloquet for MagnonModel<DIM> {
    type FloquetModel = MagnonSambe<DIM>;

    fn floquet_model(
        &self,
        drive: &FloquetDrive,
        truncation: &FloquetTruncation,
    ) -> Result<Self::FloquetModel> {
        MagnonSambe::new(self, drive, truncation)
    }
}

impl<const DIM: usize> MagnonSambe<DIM> {
    /// Build the Sambe model from an undriven magnon and a drive.
    pub fn new(
        magnon: &MagnonModel<DIM>,
        drive: &FloquetDrive,
        truncation: &FloquetTruncation,
    ) -> Result<Self> {
        magnon.validate()?;
        // The harmonics below rebuild the exchange without the single-Q
        // reference rotation that `gen_bdg` applies.
        if magnon.reference.q.iter().any(|value| value.abs() > 0.0) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "collinear_reference",
                message: "the Floquet drive currently needs a collinear reference \
                          (reference.q = 0)"
                    .into(),
            });
        }
        crate::floquet::validate_floquet_drive::<DIM>(drive)?;
        if !drive.omega0_ev.is_finite() || drive.omega0_ev <= 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "photon_energy",
                message: format!(
                    "the photon energy must be positive, found {}",
                    drive.omega0_ev
                ),
            });
        }
        if truncation.n_max < 0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "photon_cutoff",
                message: "the photon cutoff must be non-negative".into(),
            });
        }
        let sectors = truncation.n_sector()?;
        // The Sambe matrix couples sectors m and n through H^(m-n), so the
        // harmonics have to reach twice the ladder.
        let span = truncation
            .n_max
            .checked_mul(2)
            .ok_or(TbError::InvalidMagnonInvariant {
                invariant: "photon_cutoff",
                message: "2 * n_max overflows".into(),
            })?;
        let harmonic_count = span
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .filter(|&value| value > 0)
            .ok_or(TbError::InvalidMagnonInvariant {
                invariant: "photon_cutoff",
                message: "the harmonic table size overflows".into(),
            })? as usize;
        // Cap the enlarged basis like the electronic Sambe validator does.
        let size = 2usize
            .checked_mul(magnon.nspin())
            .and_then(|value| value.checked_mul(sectors))
            .ok_or(TbError::InvalidMagnonInvariant {
                invariant: "sambe_allocation",
                message: "the Sambe basis size overflows".into(),
            })?;
        let elements = size
            .checked_mul(size)
            .and_then(|value| value.checked_mul(harmonic_count))
            .ok_or(TbError::InvalidMagnonInvariant {
                invariant: "sambe_allocation",
                message: "the Sambe basis size overflows".into(),
            })?;
        if elements > 1 << 28 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "sambe_allocation",
                message: format!("{elements} complex matrix elements exceed the 2^28 limit"),
            });
        }
        let n = magnon.nspin();
        let dim = 3 * n;
        let mut harmonics = Vec::with_capacity(harmonic_count);
        for index in 0..harmonic_count {
            let m = index as isize - span;
            let mut tensor = Array3::<Complex<f64>>::zeros((magnon.nR(), dim, dim));
            for i_r in 0..magnon.nR() {
                for i in 0..n {
                    for j in 0..n {
                        let displacement = bond_displacement(magnon, i_r, i, j);
                        let coefficient = bessel_peierls_coeffs(&displacement, drive, m, m, 0)?[0];
                        for alpha in 0..SC_SPIN {
                            for beta in 0..SC_SPIN {
                                let value = magnon.ham[[i_r, 3 * i + alpha, 3 * j + beta]];
                                tensor[[i_r, 3 * i + alpha, 3 * j + beta]] =
                                    Complex::new(value, 0.0) * coefficient;
                            }
                        }
                    }
                }
            }
            harmonics.push(tensor);
        }
        // The midpoint rule converges as O(n_time^-2); four samples per photon
        // sector with a floor of 64 is ample for the bandwidths treated here.
        let n_time = (4 * sectors).max(64);
        Ok(Self {
            magnon: magnon.clone(),
            harmonics,
            omega_ev: drive.omega0_ev,
            n_max: truncation.n_max,
            harmonic_span: span,
            n_time,
        })
    }

    /// The undriven magnon model.
    pub fn magnon(&self) -> &MagnonModel<DIM> {
        &self.magnon
    }

    /// Photon energy `hbar Omega` in eV.
    pub fn omega_ev(&self) -> f64 {
        self.omega_ev
    }

    /// Number of retained photon sectors, `2 n_max + 1`.
    pub fn n_sector(&self) -> usize {
        (2 * self.n_max + 1) as usize
    }

    /// Number of time samples used for the one-period propagator.
    pub fn n_time(&self) -> usize {
        self.n_time
    }

    /// Time-dependent Bogoliubov–de Gennes matrix `H(q, t)`.
    ///
    /// The time dependence is applied to the *real-space* exchange tensors,
    /// where `J_ji(-R, t) = conj(J_ij(R, t))` holds by construction, and the
    /// BdG matrix is then built once per time step. Building a single Fourier
    /// harmonic instead would not be Hermitian, because only the pair
    /// `H^(m), H^(-m)` is adjoint-related.
    pub fn bdg_at_time(&self, q: &Array1<f64>, time: f64) -> Result<Array2<Complex<f64>>> {
        let point = self.magnon.check_q(q)?;
        if !time.is_finite() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_time",
                message: format!("the time sample must be finite, found {time}"),
            });
        }
        let n = self.magnon.nspin();
        let dim = 3 * n;
        let mut phases = Vec::with_capacity(self.harmonics.len());
        for index in 0..self.harmonics.len() {
            let m = index as isize - self.harmonic_span;
            phases.push(Complex::from_polar(1.0, -(m as f64) * self.omega_ev * time));
        }
        let mut jq = Array2::<Complex<f64>>::zeros((dim, dim));
        let mut j0 = Array2::<Complex<f64>>::zeros((dim, dim));
        let mut block = Array2::<Complex<f64>>::zeros((dim, dim));
        for i_r in 0..self.magnon.nR() {
            block.fill(Complex::new(0.0, 0.0));
            for (index, phase) in phases.iter().enumerate() {
                let tensor = self.harmonics[index].index_axis(Axis(0), i_r);
                for alpha in 0..dim {
                    for beta in 0..dim {
                        block[[alpha, beta]] += tensor[[alpha, beta]] * *phase;
                    }
                }
            }
            let mut phase = 0.0;
            for (axis, coordinate) in point.iter().enumerate().take(DIM) {
                phase -= std::f64::consts::TAU * self.magnon.hamR[[i_r, axis]] as f64 * coordinate;
            }
            let factor = Complex::from_polar(1.0, phase);
            for alpha in 0..dim {
                for beta in 0..dim {
                    jq[[alpha, beta]] += block[[alpha, beta]] * factor;
                    // The C term couples to the real instantaneous exchange.
                    j0[[alpha, beta]] += Complex::new(block[[alpha, beta]].re, 0.0);
                }
            }
        }
        self.magnon.bdg_from_jq(&jq, &j0)
    }

    /// One-period propagator `U(T)` of `i dPsi/dt = g H(t) Psi`.
    pub fn propagator_at(&self, q: &Array1<f64>) -> Result<Array2<Complex<f64>>> {
        let size = 2 * self.magnon.nspin();
        let period = std::f64::consts::TAU / self.omega_ev;
        let step = period / self.n_time as f64;
        let mut metric = Array1::<f64>::ones(size);
        for index in self.magnon.nspin()..size {
            metric[index] = -1.0;
        }
        let mut propagator = Array2::<Complex<f64>>::eye(size);
        for sample in 0..self.n_time {
            let time = (sample as f64 + 0.5) * step;
            let h = self.bdg_at_time(q, time)?;
            let mut generator = Array2::<Complex<f64>>::zeros((size, size));
            for row in 0..size {
                for column in 0..size {
                    generator[[row, column]] =
                        Complex::new(0.0, -step) * h[[row, column]] * metric[row];
                }
            }
            let factor = matrix_exponential(&generator)?;
            propagator = factor.dot(&propagator);
        }
        Ok(propagator)
    }

    /// Quasienergies at one `q`, folded into `[-Omega/2, Omega/2)`, ascending.
    ///
    /// The one-period propagator is diagonalized and only eigenvectors with
    /// positive metric norm are kept, so the result has exactly `n_band`
    /// entries and matches the undriven bands when the drive vanishes.
    pub fn quasienergies_at(&self, q: &Array1<f64>) -> Result<Array1<f64>> {
        let n = self.magnon.nspin();
        let size = 2 * n;
        let propagator = self.propagator_at(q)?;
        let (values, vectors) = propagator.eig()?;
        let mut metric = Array1::<f64>::ones(size);
        for index in n..size {
            metric[index] = -1.0;
        }
        let mut physical: Vec<(f64, f64)> = Vec::new();
        for (column, value) in values.iter().enumerate() {
            let mut norm = 0.0;
            for row in 0..size {
                norm += metric[row] * vectors[[row, column]].norm_sqr();
            }
            if norm <= 1e-9 {
                continue;
            }
            // A dissipative or parametrically unstable drive leaves |lambda| != 1,
            // and then the phase alone no longer defines a real quasienergy.
            if !value.norm().is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "driven_system_unstable",
                    message: "the propagator has a non-finite eigenvalue".into(),
                });
            }
            if (value.norm() - 1.0).abs() > 1e-6 {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "driven_system_unstable",
                    message: format!(
                        "the propagator eigenvalue has |lambda| = {:.6}, not one",
                        value.norm()
                    ),
                });
            }
            let energy = -value.arg() / (std::f64::consts::TAU) * self.omega_ev;
            physical.push((energy, norm));
        }
        if physical.len() != n {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_norm_quasienergies",
                message: format!(
                    "found {} positive-norm quasienergies, expected {n}; a Krein \
                     collision at a degenerate quasienergy can make the selection \
                     ambiguous",
                    physical.len()
                ),
            });
        }
        physical.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Array1::<f64>::zeros(n);
        for (index, entry) in physical.iter().enumerate() {
            out[index] = entry.0;
        }
        Ok(out)
    }

    /// Quasienergies of a list of `q` points, shape `(n_q, n_band)`.
    pub fn quasienergy_bands(&self, qs: &Array2<f64>) -> Result<Array2<f64>> {
        if qs.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "magnon q-points".into(),
                expected: DIM,
                found: qs.ncols(),
            });
        }
        let n = self.magnon.nspin();
        let mut out = Array2::<f64>::zeros((qs.nrows(), n));
        for (row, q) in qs.axis_iter(Axis(0)).enumerate() {
            let values = self.quasienergies_at(&q.to_owned())?;
            for (column, value) in values.iter().enumerate() {
                out[[row, column]] = *value;
            }
        }
        Ok(out)
    }
}

/// Matrix exponential by scaling and squaring with a Taylor series.
pub(crate) fn matrix_exponential(matrix: &Array2<Complex<f64>>) -> Result<Array2<Complex<f64>>> {
    let size = matrix.nrows();
    if matrix.ncols() != size {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "square_matrix",
            message: "the matrix exponential needs a square matrix".into(),
        });
    }
    let mut norm = 0.0f64;
    for row in 0..size {
        let mut row_sum = 0.0;
        for column in 0..size {
            row_sum += matrix[[row, column]].norm();
        }
        norm = norm.max(row_sum);
    }
    if !norm.is_finite() {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "finite_generator",
            message: "the matrix exponential received a non-finite matrix".into(),
        });
    }
    let mut squarings = 0u32;
    while norm > 0.5 {
        norm *= 0.5;
        squarings += 1;
    }
    let scale = 0.5f64.powi(squarings as i32);
    let scaled = matrix * Complex::new(scale, 0.0);
    let mut result = Array2::<Complex<f64>>::eye(size);
    let mut term = Array2::<Complex<f64>>::eye(size);
    for order in 1..=18 {
        term = term.dot(&scaled) * Complex::new(1.0 / order as f64, 0.0);
        result += &term;
    }
    for _ in 0..squarings {
        result = result.dot(&result);
    }
    Ok(result)
}

/// Cartesian displacement `(R + tau_j - tau_i) . L` of the bond `(R, i, j)`.
fn bond_displacement<const DIM: usize>(
    magnon: &MagnonModel<DIM>,
    i_r: usize,
    i: usize,
    j: usize,
) -> Array1<f64> {
    let fractional: Vec<f64> = (0..DIM)
        .map(|axis| magnon.hamR[[i_r, axis]] as f64 + magnon.orb[[j, axis]] - magnon.orb[[i, axis]])
        .collect();
    let mut cartesian = Array1::<f64>::zeros(DIM);
    for (component, value) in cartesian.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (axis, fraction) in fractional.iter().enumerate() {
            sum += fraction * magnon.lat[[axis, component]];
        }
        *value = sum;
    }
    cartesian
}

/// Spin components of a magnon tensor.
const SC_SPIN: usize = 3;

impl<const DIM: usize> MagnonModel<DIM> {
    /// Time-averaged (van Vleck `order = 0`) magnon model of a driven system.
    ///
    /// Every exchange bond acquires the Bessel renormalization `J_m(0)(d)` of
    /// the drive, so the result is an ordinary [`MagnonModel`] with renormalized
    /// couplings: the same interface as the electronic
    /// [`crate::Model::floquet_effective_model`]. Higher orders mix particle and
    /// hole components and are not representable as exchange tensors, so they
    /// are rejected here; the full propagator spectrum is available through
    /// [`MagnonSambe::quasienergies_at`].
    pub fn floquet_effective_model(
        &self,
        drive: &FloquetDrive,
        options: Option<&FloquetEffectiveOptions>,
    ) -> Result<Self> {
        let order = options.map(|value| value.order).unwrap_or(0);
        if order > 0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "magnon_effective_order",
                message: format!(
                    "the magnon effective model is available at order 0 only; order {order} \
                     mixes particle and hole components, use MagnonSambe::quasienergies_at"
                ),
            });
        }
        let sambe = MagnonSambe::<DIM>::new(self, drive, &FloquetTruncation { n_max: 0 })?;
        let harmonic = &sambe.harmonics[0];
        let mut ham = Array3::<f64>::zeros(self.ham.dim());
        let mut largest_imaginary = 0.0f64;
        for i_r in 0..self.nR() {
            for alpha in 0..self.ham.shape()[1] {
                for beta in 0..self.ham.shape()[2] {
                    let value = harmonic[[i_r, alpha, beta]];
                    largest_imaginary = largest_imaginary.max(value.im.abs());
                    ham[[i_r, alpha, beta]] = value.re;
                }
            }
        }
        let largest_real = ham.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if largest_imaginary > 1e-8 * largest_real.max(1e-300) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "complex_time_averaged_exchange",
                message: format!(
                    "the time-averaged exchange keeps an imaginary part \
                     ({largest_imaginary:.3e} eV) that a real exchange tensor cannot \
                     represent; use MagnonSambe::quasienergies_at"
                ),
            });
        }
        MagnonModel::<DIM>::new(
            self.lat.clone(),
            self.orb.clone(),
            self.spin.clone(),
            self.axis.clone(),
            ham,
            self.hamR.clone(),
            self.reference.clone(),
        )
    }
}

#[cfg(test)]
mod sambe_tests {
    use super::*;
    use crate::floquet::{FloquetDrive, FloquetTruncation, LightMode, fold_quasienergy};
    use crate::magnon::MagnonReference;
    use crate::magnon::MagnonSolve;
    use ndarray::{Array3, array};

    fn chain_magnon() -> MagnonModel<1> {
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        for i_r in 1..3 {
            for component in 0..3 {
                ham[[i_r, component, component]] = 0.01;
            }
        }
        MagnonModel::<1>::new(
            array![[1.0]],
            array![[0.0]],
            array![1.5],
            array![[0.0, 0.0, 1.0]],
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    fn linear_drive(omega: f64, amplitude: f64) -> FloquetDrive {
        FloquetDrive {
            omega0_ev: omega,
            modes: vec![LightMode::new(
                1,
                array![Complex::new(amplitude, amplitude)],
            )],
        }
    }

    #[test]
    fn matrix_exponential_matches_diagonal_and_pauli_cases() {
        use ndarray::array;
        // Diagonal: exp(diag(a, b)) = diag(e^a, e^b).
        let diagonal = array![
            [Complex::new(0.3, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(-1.2, 0.0)]
        ];
        let exponential = matrix_exponential(&diagonal).unwrap();
        assert!((exponential[[0, 0]].re - 0.3f64.exp()).abs() < 1e-14);
        assert!((exponential[[1, 1]].re - (-1.2f64).exp()).abs() < 1e-14);
        assert!(exponential[[0, 1]].norm() < 1e-16);
        // exp(i theta sigma_x) = cos theta I + i sin theta sigma_x.
        let theta = 0.7f64;
        let generator = array![
            [Complex::new(0.0, 0.0), Complex::new(0.0, theta)],
            [Complex::new(0.0, theta), Complex::new(0.0, 0.0)]
        ];
        let exponential = matrix_exponential(&generator).unwrap();
        assert!((exponential[[0, 0]].re - theta.cos()).abs() < 1e-14);
        assert!((exponential[[0, 1]].im - theta.sin()).abs() < 1e-14);
        assert!((exponential[[1, 0]].im - theta.sin()).abs() < 1e-14);
    }

    #[test]
    fn propagator_is_metric_unitary() {
        let magnon = chain_magnon();
        let drive = linear_drive(2.0, 0.03);
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 1 })
            .unwrap();
        let q = array![0.25f64];
        let propagator = sambe.propagator_at(&q).unwrap();
        let mut metric = Array2::<Complex<f64>>::eye(2);
        metric[[1, 1]] = Complex::new(-1.0, 0.0);
        // U^dagger g U = g.
        let mut left = propagator.t().mapv(|value| value.conj());
        left = left.dot(&metric).dot(&propagator);
        for row in 0..2 {
            for column in 0..2 {
                assert!(
                    (left[[row, column]] - metric[[row, column]]).norm() < 1e-10,
                    "({row},{column}) = {:?}",
                    left[[row, column]]
                );
            }
        }
    }

    #[test]
    fn quasienergies_are_stable_against_the_time_grid() {
        let magnon = chain_magnon();
        let drive = linear_drive(0.5, 0.02);
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 2 })
            .unwrap();
        let q = array![0.25f64];
        let base = sambe.quasienergies_at(&q).unwrap()[0];
        // The midpoint rule with the default grid must already be converged.
        assert!(sambe.n_time() >= 64);
        let coarse = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 0 })
            .unwrap();
        let coarse_value = coarse.quasienergies_at(&q).unwrap()[0];
        assert!(
            (coarse_value - base).abs() < 5e-3,
            "coarse {coarse_value} vs base {base}"
        );
    }

    #[test]
    fn rejects_spiral_references_mismatched_drives_and_bad_q() {
        // A single-Q reference needs the rotation the harmonics do not apply.
        let mut spiral = chain_magnon();
        spiral.reference.q = array![0.25];
        let drive = FloquetDrive {
            omega0_ev: 1.0,
            modes: Vec::new(),
        };
        let error = match spiral.floquet_model(&drive, &FloquetTruncation { n_max: 1 }) {
            Err(error) => error,
            Ok(_) => panic!("a spiral reference must be rejected"),
        };
        assert!(error.to_string().contains("collinear reference"), "{error}");
        // A drive whose amplitude vector does not match DIM.
        let mismatched = FloquetDrive {
            omega0_ev: 1.0,
            modes: vec![LightMode::new(
                1,
                array![Complex::new(0.01, 0.0), Complex::new(0.0, 0.0)],
            )],
        };
        assert!(
            chain_magnon()
                .floquet_model(&mismatched, &FloquetTruncation { n_max: 1 })
                .is_err()
        );
        // A q point of the wrong length must be reported, not panic.
        let good = chain_magnon()
            .floquet_model(&drive, &FloquetTruncation { n_max: 1 })
            .unwrap();
        assert!(good.quasienergies_at(&array![0.0, 0.0]).is_err());
    }

    #[test]
    fn two_site_complex_drive_keeps_the_hamiltonian_hermitian() {
        // Two inequivalent bonds and a complex link projection exercise the
        // C-term harmonic, which must use (J^m + conj(J^-m))/2.
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 3, 3));
        ham[[1, 0, 0]] = 0.02;
        ham[[1, 1, 1]] = 0.015;
        ham[[1, 2, 2]] = 0.02;
        ham[[2, 0, 0]] = 0.02;
        ham[[2, 1, 1]] = 0.015;
        ham[[2, 2, 2]] = 0.02;
        let magnon = MagnonModel::<1>::new(
            array![[1.0]],
            array![[0.0]],
            array![1.5],
            array![[0.0, 0.0, 1.0]],
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap();
        let drive = FloquetDrive {
            omega0_ev: 0.5,
            modes: vec![LightMode::new(1, array![Complex::new(0.1, 0.1)])],
        };
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 2 })
            .unwrap();
        let q = array![0.25f64];
        // H(t) must be Hermitian at every time.
        for time in [0.0f64, 0.37, 1.11] {
            let h = sambe.bdg_at_time(&q, time).unwrap();
            for row in 0..2 {
                for column in 0..2 {
                    let delta = h[[row, column]] - h[[column, row]].conj();
                    assert!(
                        delta.norm() < 1e-12,
                        "t = {time}: ({row},{column}) {delta:?}"
                    );
                }
            }
        }
        // ... and the propagator stays metric-unitary.
        let propagator = sambe.propagator_at(&q).unwrap();
        let mut metric = Array2::<Complex<f64>>::eye(2);
        metric[[1, 1]] = Complex::new(-1.0, 0.0);
        let left = propagator
            .t()
            .mapv(|value| value.conj())
            .dot(&metric)
            .dot(&propagator);
        for row in 0..2 {
            for column in 0..2 {
                assert!((left[[row, column]] - metric[[row, column]]).norm() < 1e-10);
            }
        }
    }

    /// Two sublattices with distinct positions, so the `tau_j - tau_i` part of
    /// the bond displacement is exercised.
    fn two_site_magnon() -> MagnonModel<1> {
        let ham_r = array![[0isize], [1], [-1]];
        let mut ham = Array3::<f64>::zeros((3, 6, 6));
        // A(0)-B(0) and B(0)-A(1): a two-site chain with S = (1.5, 1.2).
        for component in 0..3 {
            // Antiferromagnetic coupling for the up/down reference.
            ham[[0, component, 3 + component]] = -0.02;
            ham[[0, 3 + component, component]] = -0.02;
            ham[[1, 3 + component, component]] = -0.02;
            ham[[2, component, 3 + component]] = -0.02;
        }
        MagnonModel::<1>::new(
            array![[2.0]],
            array![[0.0], [0.5]],
            array![1.5, 1.2],
            array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ham,
            ham_r,
            MagnonReference::collinear(),
        )
        .unwrap()
    }

    #[test]
    fn two_sublattice_model_reproduces_its_bands_without_a_drive() {
        let magnon = two_site_magnon();
        let drive = FloquetDrive {
            omega0_ev: 2.0,
            modes: Vec::new(),
        };
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 1 })
            .unwrap();
        let points = array![[0.0], [0.13], [0.31], [0.5]];
        let bands = magnon.magnon_band_all(&points).unwrap();
        let quasi = sambe.quasienergy_bands(&points).unwrap();
        assert_eq!(quasi.dim().1, 2);
        for row in 0..4 {
            for band in 0..2 {
                let expected = fold_quasienergy(bands[[row, band]], drive.omega0_ev);
                assert!(
                    (quasi[[row, band]] - expected).abs() < 1e-8,
                    "q = {}: quasi {} vs undriven {expected}",
                    points[[row, 0]],
                    quasi[[row, band]]
                );
            }
        }
    }

    #[test]
    fn zero_amplitude_effective_model_is_the_undriven_model() {
        let magnon = two_site_magnon();
        let drive = FloquetDrive {
            omega0_ev: 1.0,
            modes: vec![LightMode::new(1, array![Complex::new(0.0, 0.0)])],
        };
        let effective = magnon.floquet_effective_model(&drive, None).unwrap();
        let difference = (&effective.ham - &magnon.ham)
            .mapv(f64::abs)
            .iter()
            .copied()
            .fold(0.0, f64::max);
        assert!(difference < 1e-15, "max deviation {difference:e}");
    }

    #[test]
    fn complex_time_average_is_rejected_by_the_effective_model() {
        // A static (harmonic-zero) drive is time independent, but its Peierls
        // phase is complex, so the average cannot be a real exchange tensor.
        let magnon = two_site_magnon();
        let drive = FloquetDrive {
            omega0_ev: 1.0,
            modes: vec![LightMode::new(0, array![Complex::new(0.5, 0.0)])],
        };
        let error = match magnon.floquet_effective_model(&drive, None) {
            Err(error) => error,
            Ok(_) => panic!("a complex time average must be rejected"),
        };
        assert!(error.to_string().contains("imaginary part"), "{error}");
    }

    #[test]
    fn zero_drive_reproduces_the_undriven_bands() {
        let magnon = chain_magnon();
        let drive = FloquetDrive {
            omega0_ev: 1.0,
            modes: Vec::new(),
        };
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 2 })
            .unwrap();
        assert_eq!(sambe.n_sector(), 5);
        for q in [0.0f64, 0.125, 0.25, 0.5] {
            let band = magnon.magnon_band_onek(&array![q]).unwrap()[0];
            let quasi = sambe.quasienergies_at(&array![q]).unwrap();
            assert_eq!(quasi.len(), 1);
            let expected = fold_quasienergy(band, drive.omega0_ev);
            assert!(
                (quasi[0] - expected).abs() < 1e-8,
                "q = {q}: quasi {} vs undriven {expected}",
                quasi[0]
            );
        }
    }

    #[test]
    fn high_frequency_limit_matches_the_effective_model() {
        let magnon = chain_magnon();
        let drive = linear_drive(50.0, 0.05);
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 2 })
            .unwrap();
        let effective = magnon.floquet_effective_model(&drive, None).unwrap();
        for q in [0.125f64, 0.25, 0.375] {
            let quasi = sambe.quasienergies_at(&array![q]).unwrap()[0];
            let band = effective.magnon_band_onek(&array![q]).unwrap()[0];
            let expected = fold_quasienergy(band, drive.omega0_ev);
            assert!(
                (quasi - expected).abs() < 5e-3,
                "q = {q}: quasi {quasi} vs effective {expected}"
            );
        }
    }

    #[test]
    fn quasienergies_are_folded_and_the_effective_model_is_guarded() {
        let magnon = chain_magnon();
        let drive = linear_drive(0.5, 0.02);
        let sambe = magnon
            .floquet_model(&drive, &FloquetTruncation { n_max: 2 })
            .unwrap();
        let quasi = sambe.quasienergies_at(&array![0.25]).unwrap();
        for value in quasi.iter() {
            assert!(*value >= -0.5 * drive.omega0_ev - 1e-9, "{value}");
            assert!(*value < 0.5 * drive.omega0_ev + 1e-9, "{value}");
        }
        let error = magnon
            .floquet_effective_model(
                &drive,
                Some(&FloquetEffectiveOptions {
                    order: 1,
                    ..Default::default()
                }),
            )
            .unwrap_err();
        assert!(error.to_string().contains("order 0 only"), "{error}");
    }
}
