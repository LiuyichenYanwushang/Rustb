//! Floquet-driven exchange and the magnon bands it produces.
//!
//! This module implements the Aharonov–Bohm mechanism of
//! *Odd-Parity Magnons*: a half-filled single-orbital Hubbard model on the
//! honeycomb lattice, driven by circularly polarized light through the Peierls
//! substitution, is reduced to an effective spin Hamiltonian by a
//! Floquet–Sambe strong-coupling (Löwdin) expansion. The result is
//!
//! $$
//! H = J\_1\\!\sum\_{\langle ij\rangle}\mathbf S\_i\cdot\mathbf S\_j
//!   \+ J\_2\\!\sum\_{\langle\\!\langle ik\rangle\\!\rangle}
//!     \mathbf S\_i\cdot\mathbf S\_k
//!   \+ \chi\\!\sum\_{(i,j,k)}\mathbf S\_i\cdot(\mathbf S\_j\times\mathbf S\_k)
//!   \+ O(t^6),
//! $$
//!
//! with $\chi(-\xi) = -\chi(\xi)$ while $J\_1$ and $J\_2$ are even in the
//! helicity $\xi=\pm1$. At quadratic spin-wave order the chirality is a
//! momentum-odd term
//!
//! $$
//! D\_{\mathbf k} = 2\chi S^2\sum\_{\nu}\sin(\mathbf k\cdot\mathbf b\_\nu)
//! $$
//!
//! and the two magnon branches split,
//!
//! $$
//! \hbar\omega\_{\uparrow,\downarrow}(\mathbf k)
//!   = \sqrt{\bigl(A\_{\mathbf k}\pm D\_{\mathbf k}\bigr)^2
//!           \- \lvert B\_{\mathbf k}\rvert^2},
//! $$
//!
//! which satisfy $\omega\_\uparrow(\mathbf k) = \omega\_\downarrow(-\mathbf k)$:
//! an odd-parity magnon.
//!
//! # Light coupling
//!
//! For $\mathbf A(t) = A\_0(-\sin\Omega t,\\ \xi\cos\Omega t)$ the Peierls
//! phase on a bond of length $a$ at angle $\phi$ gives
//!
//! $$
//! \begin{aligned}
//! t\_{ij}(t) &= \sum\_m t\_{ij}\\,\mathcal J\_m(\lambda)\\,
//!   e^{i\xi m\phi}\\,e^{-im\Omega t},\\\\
//! \lambda &= \frac{eA\_0a}{\hbar} = \frac{eEa}{\hbar\Omega}.
//! \end{aligned}
//! $$
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
    /// Hopping amplitude $t$ in eV.
    pub hopping_ev: f64,
    /// On-site repulsion $U$ in eV.
    pub interaction_ev: f64,
    /// Photon energy $\hbar\Omega$ in eV.
    pub photon_ev: f64,
    /// Nonnegative dimensionless Peierls amplitude
    /// $\lambda = eEa/(\hbar\Omega)$; negative values are rejected by
    /// [`Self::validate`].
    pub lambda: f64,
    /// Optical helicity $\xi=\pm1$.
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
            ("helicity", self.helicity),
        ] {
            if !value.is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "finite_drive_parameter",
                    message: format!("{name} must be finite, found {value}"),
                });
            }
        }
        if self.lambda < 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "nonnegative_lambda",
                message: format!("lambda must be non-negative, found {}", self.lambda),
            });
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

/// Effective couplings of the AA-stacked bilayer (paper, Supplemental
/// Material "Bilayer model").
///
/// The bilayer adds a vertical pair exchange `jperp`, a cross-layer diagonal
/// exchange `jcross` and three four-spin coefficients `k1`, `k2`, `k3` on each
/// vertical rectangle, on top of the monolayer `j1`, `j2` and `chirality`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilayerExchange {
    /// Intralayer nearest-neighbour exchange, before subtracting the
    /// phenomenological ferromagnetic direct exchange $J\_D$ of the paper.
    pub j1: f64,
    /// Intralayer next-nearest-neighbour exchange.
    pub j2: f64,
    /// Vertical pair exchange.
    pub jperp: f64,
    /// Cross-layer nearest-neighbour exchange.
    pub jcross: f64,
    /// Four-spin coefficient of the same-layer intralayer bilinears.
    pub k1: f64,
    /// Four-spin coefficient of the vertical-pair bilinears.
    pub k2: f64,
    /// Four-spin coefficient of the diagonal cross-layer bilinears; stored with
    /// the sign printed in the Supplemental Material, i.e. $k\_3=-k\_1$.
    pub k3: f64,
    /// Scalar spin chirality of each layer.
    pub chirality: f64,
}

/// Hopping and interaction parameters of the bilayer (the static limit has no
/// drive).
#[derive(Clone, Copy, Debug)]
pub struct BilayerParams {
    /// In-plane hopping $t$ in eV.
    pub hopping_ev: f64,
    /// Vertical hopping $t\_\perp$ in eV.
    pub interlayer_hopping_ev: f64,
    /// On-site repulsion $U$ in eV.
    pub interaction_ev: f64,
}

impl BilayerExchange {
    /// Decoupled bilayer whose intralayer part is a given monolayer exchange.
    ///
    /// $J\_\perp$, $J\_\times$ and the four-spin coefficients vanish, so the two
    /// layers are independent and each reproduces the monolayer spectrum.
    pub fn decoupled(monolayer: &DrivenExchange) -> Self {
        Self {
            j1: monolayer.j1,
            j2: monolayer.j2,
            jperp: 0.0,
            jcross: 0.0,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            chirality: monolayer.chirality,
        }
    }

    /// Static ($\lambda=0$) couplings of the AA-stacked bilayer.
    ///
    /// Reproduces the closed forms printed in the Supplemental Material:
    ///
    /// $$
    /// \begin{aligned}
    /// J\_1 &= \frac{4t^2}{U} - \frac{16t^4 + 4t^2t\_\perp^2}{U^3},\\\\
    /// J\_2 &= \frac{4t^4}{U^3},\\\\
    /// J\_\perp &= \frac{4t\_\perp^2}{U}
    ///   \- \frac{16t\_\perp^4 + 12t^2t\_\perp^2}{U^3},\\\\
    /// J\_\times &= \frac{4t^2t\_\perp^2}{U^3},\\\\
    /// K^{(1)} = K^{(2)} = -K^{(3)}
    ///   &= \frac{80\\,t^2t\_\perp^2}{U^3},\qquad \chi = 0.
    /// \end{aligned}
    /// $$
    pub fn static_limit(params: &BilayerParams) -> Result<Self> {
        let t = params.hopping_ev;
        let tp = params.interlayer_hopping_ev;
        let u = params.interaction_ev;
        // Every coefficient is even in the hoppings, so their sign is a gauge.
        for (name, value) in [
            ("hopping_ev", t),
            ("interlayer_hopping_ev", tp),
            ("interaction_ev", u),
        ] {
            if !value.is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "finite_bilayer_parameter",
                    message: format!("{name} must be finite, found {value}"),
                });
            }
        }
        if u <= 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_interaction",
                message: "the on-site interaction must be positive".into(),
            });
        }
        let t2 = t * t;
        let tp2 = tp * tp;
        let u3 = u * u * u;
        let bilayer = Self {
            j1: 4.0 * t2 / u - (16.0 * t2 * t2 + 4.0 * t2 * tp2) / u3,
            j2: 4.0 * t2 * t2 / u3,
            jperp: 4.0 * tp2 / u - (16.0 * tp2 * tp2 + 12.0 * t2 * tp2) / u3,
            jcross: 4.0 * t2 * tp2 / u3,
            k1: 80.0 * t2 * tp2 / u3,
            k2: 80.0 * t2 * tp2 / u3,
            k3: -80.0 * t2 * tp2 / u3,
            chirality: 0.0,
        };
        let values = [
            bilayer.j1,
            bilayer.j2,
            bilayer.jperp,
            bilayer.jcross,
            bilayer.k1,
            bilayer.k2,
            bilayer.k3,
        ];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_bilayer_couplings",
                message: "the bilayer couplings overflow for these parameters".into(),
            });
        }
        Ok(bilayer)
    }
}

impl DrivenExchange {
    /// Build a `Magnon` whose spin-wave spectrum reproduces
    /// [`Self::honeycomb_dispersion`].
    ///
    /// The mapping between the paper's convention (spin operators with
    /// $\lvert\mathbf S\rvert = S$)
    ///
    /// $$
    /// H = J\sum\_{\langle ij\rangle}\mathbf S\_i\cdot\mathbf S\_j
    /// $$
    ///
    /// and this crate's landscape convention (unit vectors)
    ///
    /// $$
    /// E = -\sum \hat J\\,\mathbf s\_i\cdot\mathbf s\_j
    /// $$
    ///
    /// is
    ///
    /// $$
    /// \begin{aligned}
    /// J\_{\mathrm{land}} &= -\tfrac{S^2}{2}\\,J\_{\mathrm{paper}},
    ///   &&(J\_1,\\ J\_2)\\\\
    /// \mathrm{SIA}\_{zz} &= -K S^2,
    ///   &&(\text{single-ion anisotropy, }R=0\text{ block})\\\\
    /// d\_{\mathrm{mi}} &= \tfrac{\chi S^3}{2}
    ///   &&(\text{next-nearest-neighbour DMI}).
    /// \end{aligned}
    /// $$
    ///
    /// The $S^2/2$ of the exchange carries both the spin-operator normalisation
    /// and the $2$ of TB2J's $C$ term (the transverse frame contributes a factor
    /// two of its own). The anisotropy enters only through that same $C$ term,
    /// which is why it keeps a single $S^2$ rather than $S^2/2$. The chirality
    /// needs $S^3/2$ because the solver's
    /// $A\_1\lbrack0,0\rbrack-A\_1\lbrack1,1\rbrack$ is
    ///
    /// $$
    /// \bigl(A\_1\lbrack0,0\rbrack-A\_1\lbrack1,1\rbrack\bigr)\_{\mathbf q}
    /// = \frac{8}{S}\sum\_{\nu} d\\,\sin(\mathbf k\cdot\mathbf b\_\nu).
    /// $$
    ///
    /// Our next-nearest list is $\\{\mathbf b\_1,-\mathbf b\_2,\mathbf b\_3\\}$, so
    /// the paper's $\sum\_\nu\sin(\mathbf k\cdot\mathbf b\_\nu)$ needs
    /// alternating signs.
    ///
    /// With this mapping the resulting bands reproduce
    /// [`Self::honeycomb_dispersion`] to `1e-12` eV on a k-mesh, and the two
    /// branches match the spin-labelled `BilayerSpinWave::sector_bands` of the
    /// decoupled bilayer. The reference is the Neel state, so `anisotropy_ev`
    /// must be non-positive (an easy axis) and `self.j1 > 0` (antiferromagnetic
    /// in the paper's convention).
    pub fn honeycomb_magnon(&self, spin: f64, anisotropy_ev: f64) -> Result<MagnonModel<2>> {
        if !spin.is_finite() || spin <= 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_spin",
                message: format!("the spin length must be positive, found {spin}"),
            });
        }
        if !anisotropy_ev.is_finite() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_anisotropy",
                message: "the anisotropy must be finite".into(),
            });
        }
        if ![self.j1, self.j2, self.chirality]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_exchange_coefficients",
                message: "j1, j2 and chirality must be finite".into(),
            });
        }
        if self.j1 <= 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_nearest_exchange",
                message: format!("j1 must be positive, found {}", self.j1),
            });
        }
        if anisotropy_ev > 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "easy_axis_anisotropy",
                message: format!(
                    "the Neel reference needs an easy axis (K <= 0), found {anisotropy_ev}\
                      ; an easy-plane anisotropy leaves the reference unstable"
                ),
            });
        }
        let orb = ndarray::array![[0.0, 0.0], [1.0 / 3.0, -2.0 / 3.0]];
        let translations: Vec<[isize; 2]> =
            vec![[0, 0], [0, 1], [-1, 1], [1, -1], [0, -1], [1, 0], [-1, 0]];
        let mut ham_r = Array2::<isize>::zeros((translations.len(), 2));
        for (row, value) in translations.iter().enumerate() {
            ham_r[[row, 0]] = value[0];
            ham_r[[row, 1]] = value[1];
        }
        let mut ham = Array3::<f64>::zeros((translations.len(), 6, 6));
        let index_of = |r: [isize; 2]| {
            translations
                .iter()
                .position(|value| *value == r)
                .unwrap_or(0)
        };
        let mut add = |i: usize, k: usize, r: [isize; 2], tensor: [[f64; 3]; 3]| {
            let forward = index_of(r);
            let backward = index_of([-r[0], -r[1]]);
            for alpha in 0..3 {
                for beta in 0..3 {
                    ham[[forward, 3 * i + alpha, 3 * k + beta]] += tensor[alpha][beta];
                    ham[[backward, 3 * k + beta, 3 * i + alpha]] += tensor[alpha][beta];
                }
            }
        };
        let j1 = -0.5 * spin * spin * self.j1;
        let j2 = -0.5 * spin * spin * self.j2;
        // The solver's A1[0,0] - A1[1,1] equals (8/S) sum_R d_R sin(2 pi q.R),
        // while Eq. (6) needs 2 D_k = 4 chi S^2 sum sin, hence d = chi S^3 / 2.
        // Our next-nearest list is {b1, -b2, b3}, so the paper's sum over b_nu
        // needs the alternating signs (+1, -1, +1).
        let dmi = 0.5 * self.chirality * spin.powi(3);
        // The anisotropy enters only through the longitudinal C term, which
        // carries a factor 2, so the block must end up holding -K S^2. The
        // on-site block sits at R = 0, where `add` writes the forward and the
        // backward entry into the same place, hence the half here.
        let sia = -0.5 * anisotropy_ev * spin * spin;
        if [j1, j2, dmi, sia].iter().any(|value| !value.is_finite()) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_honeycomb_magnon_coefficients",
                message: "spin-scaled honeycomb coefficients overflowed".into(),
            });
        }
        let mut nearest = [[0.0f64; 3]; 3];
        for (component, entry) in nearest.iter_mut().enumerate() {
            entry[component] = j1;
        }
        for r in [[0, 0], [0, 1], [-1, 1]] {
            add(0, 1, r, nearest);
        }
        for site in 0..2 {
            for (r, sign) in [([1isize, 0], 1.0), ([1, -1], -1.0), ([0, -1], 1.0)] {
                let mut tensor = [[0.0f64; 3]; 3];
                for (component, entry) in tensor.iter_mut().enumerate() {
                    entry[component] = j2;
                }
                tensor[0][1] = -sign * dmi;
                tensor[1][0] = sign * dmi;
                add(site, site, r, tensor);
            }
            let mut tensor = [[0.0f64; 3]; 3];
            tensor[2][2] = sia;
            add(site, site, [0, 0], tensor);
        }
        MagnonModel::<2>::new(
            ndarray::array![[5.0, 0.0], [2.5, 4.330127018922193]],
            orb,
            Array1::from(vec![spin, spin]),
            ndarray::array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            ham,
            ham_r,
            crate::magnon::MagnonReference::collinear(),
        )
    }
}

/// Effective spin couplings generated by the drive, in eV.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrivenExchange {
    /// Nearest-neighbour Heisenberg coupling $J\_1$.
    pub j1: f64,
    /// Next-nearest-neighbour Heisenberg coupling $J\_2$.
    pub j2: f64,
    /// Scalar spin chirality $\chi$.
    pub chirality: f64,
}

impl DrivenExchange {
    /// Effective $J\_1$, $J\_2$ and $\chi$ of the CPL-driven honeycomb Hubbard
    /// model.
    ///
    /// Implements the Floquet–Sambe Löwdin sums of the paper's Supplemental
    /// Material: second order
    ///
    /// $$
    /// J^{(2)} = 4t^2\sum\_m
    /// \frac{\mathcal J\_m^2(\lambda)}{U-m\hbar\Omega},
    /// $$
    ///
    /// and fourth order through the $F$ (one-doublon), $B$ (nonzero-replica) and
    /// $G$ (folded) contributions with the honeycomb $V$-cluster multiplicities.
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
        let result = Self {
            j1: j1_second + j1_f + j1_b + j1_g,
            j2: j2_f + j2_b + j2_g,
            chirality: chi_f + chi_b,
        };
        if [result.j1, result.j2, result.chirality]
            .iter()
            .all(|value| value.is_finite())
        {
            Ok(result)
        } else {
            Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_driven_exchange",
                message: "the driven exchange coefficients overflowed or became nonfinite".into(),
            })
        }
    }

    /// Honeycomb spin-wave functions $A\_{\mathbf k}$, $B\_{\mathbf k}$ and
    /// $D\_{\mathbf k}$ of the paper.
    ///
    /// $\mathbf q$ is in fractional reciprocal coordinates of the hexagonal cell
    /// with the nearest-neighbour vectors of the Supplemental Material, and the
    /// returned $(a\_{\mathbf k},\lvert b\_{\mathbf k}\rvert,d\_{\mathbf k})$
    /// uses the units of the couplings (eV).
    ///
    /// # Errors
    /// Returns [`TbError::InvalidMagnonInvariant`] for nonpositive/nonfinite spin,
    /// nonfinite anisotropy or momentum, invalid exchange coefficients, or a
    /// nonfinite computed result.
    pub fn honeycomb_functions(
        &self,
        spin: f64,
        anisotropy_ev: f64,
        q: &[f64; 2],
    ) -> Result<(f64, f64, f64)> {
        if !spin.is_finite() || spin <= 0.0 {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_finite_spin",
                message: format!("spin must be finite and positive, found {spin}"),
            });
        }
        if !anisotropy_ev.is_finite() || q.iter().any(|value| !value.is_finite()) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_honeycomb_input",
                message: "anisotropy and momentum components must be finite".into(),
            });
        }
        if [self.j1, self.j2, self.chirality]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_exchange_coefficients",
                message: "j1, j2 and chirality must be finite".into(),
            });
        }
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
        let result = (a_k, b_k.norm(), d_k);
        if result.0.is_finite() && result.1.is_finite() && result.2.is_finite() {
            Ok(result)
        } else {
            Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_honeycomb_functions",
                message: "honeycomb spin-wave functions overflowed".into(),
            })
        }
    }

    /// Magnon energies of the two spin branches at $\mathbf q$, in eV.
    ///
    /// Returns $(\omega\_\uparrow,\omega\_\downarrow)$. A branch is stable only
    /// when $A\pm D \ge \lvert B\rvert$ within an energy-scale-relative
    /// roundoff tolerance; a negative radicand beyond roundoff is returned as a
    /// structured error.
    ///
    /// # Errors
    /// Returns [`TbError::InvalidMagnonInvariant`] for invalid inputs, a
    /// nonfinite result, or an unstable branch (including a negative-energy
    /// reference that would otherwise appear stable after squaring).
    pub fn honeycomb_dispersion(
        &self,
        spin: f64,
        anisotropy_ev: f64,
        q: &[f64; 2],
    ) -> Result<(f64, f64)> {
        let (a_k, b_abs, d_k) = self.honeycomb_functions(spin, anisotropy_ev, q)?;
        let branch = |sign: f64| -> Result<f64> {
            let energy = a_k + sign * d_k;
            let scale = energy.abs().max(b_abs);
            if !energy.is_finite() || !scale.is_finite() {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "finite_honeycomb_dispersion",
                    message: "honeycomb branch energy is nonfinite".into(),
                });
            }
            if scale == 0.0 {
                return Ok(0.0);
            }
            // Normalize before comparing or squaring. This avoids overflow at
            // large energy scales and preserves subnormal-scale spectra.
            let e = energy / scale;
            let b = b_abs / scale;
            let delta = e - b;
            let tolerance = 1e-10;
            if delta < -tolerance {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "stable_honeycomb_branch",
                    message: format!("A ± D = {energy} is below |B| = {b_abs}"),
                });
            }
            let radicand_scaled = delta * (e + b);
            if !radicand_scaled.is_finite() || radicand_scaled < -tolerance {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "nonnegative_honeycomb_radicand",
                    message: format!("normalized branch radicand is invalid: {radicand_scaled}"),
                });
            }
            let result = scale * radicand_scaled.max(0.0).sqrt();
            if result.is_finite() {
                Ok(result)
            } else {
                Err(TbError::InvalidMagnonInvariant {
                    invariant: "finite_honeycomb_dispersion",
                    message: "honeycomb dispersion overflowed".into(),
                })
            }
        };
        Ok((branch(1.0)?, branch(-1.0)?))
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
        let (a_k, b_abs, d_k) = exchange
            .honeycomb_functions(spin, anisotropy, &[0.0, 0.0])
            .unwrap();
        assert!((a_k - (3.0 * spin * exchange.j1 - 2.0 * anisotropy * spin)).abs() < 1e-12);
        assert!((b_abs - 3.0 * spin * exchange.j1).abs() < 1e-12);
        assert!(d_k.abs() < 1e-15);
        // K: |B_k| = 0, D_k = 3 sqrt(3) chi S^2.
        let (a_k, b_abs, d_k) = exchange
            .honeycomb_functions(spin, anisotropy, &[1.0 / 3.0, 2.0 / 3.0])
            .unwrap();
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
    fn splitting_has_f_wave_angular_dependence() {
        // Around Gamma the splitting is cubic and C3-invariant: six sign
        // changes on a physical Cartesian circle, not a fractional-coordinate ellipse.
        let params = DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.4,
            helicity: 1.0,
            harmonic_max: 8,
        };
        let exchange = DrivenExchange::honeycomb(&params).unwrap();
        let (spin, anisotropy) = (1.5, -0.01);
        let magnon = exchange.honeycomb_magnon(spin, anisotropy).unwrap();
        // B = 2pi (lat^T)^-1, hence q_frac = k_cart lat^T / 2pi.
        // Read the actual model lattice, including its length scale.
        let fractional = |radius: f64, theta: f64| {
            let cart = [radius * theta.cos(), radius * theta.sin()];
            let q = [
                (cart[0] * magnon.lat[[0, 0]] + cart[1] * magnon.lat[[0, 1]])
                    / std::f64::consts::TAU,
                (cart[0] * magnon.lat[[1, 0]] + cart[1] * magnon.lat[[1, 1]])
                    / std::f64::consts::TAU,
            ];
            // Independent 2x2 inverse recovers Cartesian k and checks the radius.
            let det =
                magnon.lat[[0, 0]] * magnon.lat[[1, 1]] - magnon.lat[[0, 1]] * magnon.lat[[1, 0]];
            let x = std::f64::consts::TAU * (q[0] * magnon.lat[[1, 1]] - q[1] * magnon.lat[[0, 1]])
                / det;
            let y = std::f64::consts::TAU * (q[1] * magnon.lat[[0, 0]] - q[0] * magnon.lat[[1, 0]])
                / det;
            assert!((x.hypot(y) - radius).abs() < 1e-15);
            q
        };
        let sample = |couplings: &DrivenExchange, radius: f64, theta: f64| {
            let q = fractional(radius, theta);
            let (up, down) = couplings
                .honeycomb_dispersion(spin, anisotropy, &q)
                .unwrap();
            assert!(up.is_finite() && down.is_finite());
            up - down
        };
        let radius = 0.03;
        let steps = 720;
        let values: Vec<_> = (0..steps)
            .map(|index| {
                sample(
                    &exchange,
                    radius,
                    std::f64::consts::TAU * index as f64 / steps as f64,
                )
            })
            .collect();
        assert_eq!(values.len(), steps);
        let peak = values.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(peak > 1e-9, "no resolved splitting: {peak:e}");
        // Remove near-zero samples first, then close the sign cycle. A node
        // landing exactly on the mesh must neither add nor lose a crossing.
        let signs: Vec<_> = values
            .iter()
            .filter(|x| x.abs() > peak * 1e-8)
            .map(|x| x.signum())
            .collect();
        assert!(signs.len() > steps - 12);
        let changes = signs
            .iter()
            .zip(signs.iter().cycle().skip(1))
            .take(signs.len())
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(changes, 6);
        for theta in [0.2f64, 1.0, 2.3] {
            let plus = sample(&exchange, radius, theta);
            let opposite = sample(&exchange, radius, theta + PI);
            let rotated = sample(&exchange, radius, theta + std::f64::consts::TAU / 3.0);
            assert!((plus + opposite).abs() < peak * 1e-7);
            assert!((plus - rotated).abs() < peak * 1e-7);
            let half = sample(&exchange, radius / 2.0, theta);
            assert!(
                (plus / half - 8.0).abs() < 0.03,
                "not cubic: {plus:e}/{half:e}"
            );
        }
        let zero_chi = DrivenExchange {
            chirality: 0.0,
            ..exchange
        };
        for index in 0..steps {
            assert_eq!(
                sample(
                    &zero_chi,
                    radius,
                    std::f64::consts::TAU * index as f64 / steps as f64
                ),
                0.0
            );
        }
    }

    #[test]
    fn honeycomb_magnon_reproduces_the_paper_dispersion() {
        use crate::magnon::MagnonSolve;
        use ndarray::array;
        let params = DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.4,
            helicity: 1.0,
            harmonic_max: 8,
        };
        let exchange = DrivenExchange::honeycomb(&params).unwrap();
        let (spin, anisotropy) = (1.5, -0.01);
        let magnon = exchange.honeycomb_magnon(spin, anisotropy).unwrap();
        for k in [
            [0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0],
            [0.5, 0.0],
            [0.2, 0.1],
            [0.13, 0.27],
        ] {
            // Independent SM geometry in Cartesian coordinates, not
            // honeycomb_functions (which discards the phase of B).
            let k_cart = [
                std::f64::consts::TAU * k[0] / 5.0,
                std::f64::consts::TAU * (2.0 * k[1] - k[0]) / (5.0 * 3.0f64.sqrt()),
            ];
            let phase = |d: [f64; 2]| k_cart[0] * d[0] + k_cart[1] * d[1];
            let delta = [
                [0.0, -5.0 / 3.0f64.sqrt()],
                [2.5, 2.5 / 3.0f64.sqrt()],
                [-2.5, 2.5 / 3.0f64.sqrt()],
            ];
            let b_atomic: Complex<f64> = delta
                .into_iter()
                .map(|d| Complex::from_polar(spin * exchange.j1, phase(d)))
                .sum();
            let nnn = [
                [5.0, 0.0],
                [-2.5, 2.5 * 3.0f64.sqrt()],
                [-2.5, -2.5 * 3.0f64.sqrt()],
            ];
            let a = 3.0 * spin * exchange.j1 - 2.0 * spin * anisotropy
                + 2.0
                    * spin
                    * exchange.j2
                    * (nnn.iter().map(|&d| phase(d).cos()).sum::<f64>() - 3.0);
            let d = 2.0
                * exchange.chirality
                * spin
                * spin
                * nnn.iter().map(|&d| phase(d).sin()).sum::<f64>();
            // SM B uses embedded delta bonds; Magnon uses cell-periodic R.
            // Remove exp(i k.tau_B), tau_B=delta_1, without conjugating B.
            let b = b_atomic * Complex::from_polar(1.0, -phase(delta[0]));
            let z = Complex::new(0.0, 0.0);
            // Fixed Nambu order (a_q, b_q, a†_-q, b†_-q):
            // up=(0,3) carries A+D, down=(1,2) carries A-D.
            // No sorted eigenvalue comparison can establish these labels.
            let expected_h = array![
                [Complex::new(a + d, 0.0), z, z, b],
                [z, Complex::new(a - d, 0.0), b.conj(), z],
                [z, b, Complex::new(a - d, 0.0), z],
                [b.conj(), z, z, Complex::new(a + d, 0.0)],
            ];
            let h = magnon.gen_bdg(&array![k[0], k[1]]).unwrap();
            assert_eq!(h.dim(), (4, 4));
            for ((row, col), expected) in expected_h.indexed_iter() {
                assert!(
                    (h[[row, col]] - expected).norm() < 1e-12,
                    "q={k:?} Nambu[{row},{col}]: {} != {expected}",
                    h[[row, col]]
                );
            }
            if k == [0.13, 0.27] {
                assert!(b.im.abs() > 1e-3, "oracle must resolve complex B");
                assert!(d.abs() > 1e-6, "oracle must resolve labelled splitting");
                assert!((h[[0, 0]].re - h[[1, 1]].re - 2.0 * d).abs() < 1e-12);
            }
            let bands = magnon.magnon_band_onek(&array![k[0], k[1]]).unwrap();
            let (up, down) = exchange.honeycomb_dispersion(spin, anisotropy, &k).unwrap();
            let expected = if up < down { [up, down] } else { [down, up] };
            for band in 0..2 {
                assert!(
                    (bands[band] - expected[band]).abs() < 1e-9,
                    "k = {k:?}: solver {bands:?} paper {expected:?}"
                );
            }
        }
        // The valley splitting changes sign between K and K'.
        let at_k = magnon
            .magnon_band_onek(&array![1.0 / 3.0, 2.0 / 3.0])
            .unwrap();
        let at_prime = magnon
            .magnon_band_onek(&array![2.0 / 3.0, 1.0 / 3.0])
            .unwrap();
        // Odd parity swaps the spin branches between the valleys, so the
        // *sorted* spectrum is identical at K and K' while it is split. A
        // global sign flip of `chirality` is not observable in a sorted
        // frequency spectrum (the branches only swap labels). This final
        // spectral check pins |chi|; the labelled 4x4 oracle above pins the
        // mapping's chirality sign as well as the complex pairing phase.
        assert!(at_k[1] - at_k[0] > 1e-6, "{at_k:?}");
        for band in 0..2 {
            assert!(
                (at_k[band] - at_prime[band]).abs() < 1e-12,
                "{at_k:?} {at_prime:?}"
            );
        }
    }

    #[test]
    fn bilayer_static_limit_matches_the_supplemental_forms() {
        use super::BilayerExchange;
        use super::BilayerParams;
        let params = BilayerParams {
            hopping_ev: 0.1,
            interlayer_hopping_ev: 0.05,
            interaction_ev: 1.7,
        };
        let bilayer = BilayerExchange::static_limit(&params).unwrap();
        let (t, tp, u) = (
            params.hopping_ev,
            params.interlayer_hopping_ev,
            params.interaction_ev,
        );
        let u3 = u * u * u;
        let t2 = t * t;
        let tp2 = tp * tp;
        assert!(
            (bilayer.j1 - (4.0 * t2 / u - (16.0 * t2 * t2 + 4.0 * t2 * tp2) / u3)).abs() < 1e-15
        );
        assert!((bilayer.j2 - 4.0 * t2 * t2 / u3).abs() < 1e-18);
        assert!(
            (bilayer.jperp - (4.0 * tp2 / u - (16.0 * tp2 * tp2 + 12.0 * t2 * tp2) / u3)).abs()
                < 1e-15
        );
        assert!((bilayer.jcross - 4.0 * t2 * tp2 / u3).abs() < 1e-18);
        assert!((bilayer.k1 - 80.0 * t2 * tp2 / u3).abs() < 1e-17);
        assert!((bilayer.k2 - bilayer.k1).abs() < 1e-18);
        assert!((bilayer.k3 + bilayer.k1).abs() < 1e-18);
        assert_eq!(bilayer.chirality, 0.0);
        // A U-power mistake is invisible at U = 1 (u^3 == u), so pin U = 2
        // against independently evaluated values.
        let doubled = BilayerExchange::static_limit(&BilayerParams {
            interaction_ev: 2.0,
            ..params
        })
        .unwrap();
        assert!((doubled.j2 - 5.0e-5).abs() < 1e-15, "J2 = {}", doubled.j2);
        assert!((doubled.jcross - 1.25e-5).abs() < 1e-15);
        assert!((doubled.k1 - 2.5e-4).abs() < 1e-14);
        assert!((doubled.j1 - 0.0197875).abs() < 1e-15);
        assert!((doubled.jperp - 0.00495).abs() < 1e-14);
        // The decoupled limit must agree with the driven single-layer sums of
        // the same paper at zero amplitude.
        let monolayer = DrivenExchange::honeycomb(&DrivenHubbardParams {
            hopping_ev: 0.1,
            interaction_ev: 2.0,
            photon_ev: 1.0,
            lambda: 0.0,
            helicity: 1.0,
            harmonic_max: 0,
        })
        .unwrap();
        let decoupled = BilayerExchange::static_limit(&BilayerParams {
            hopping_ev: 0.1,
            interlayer_hopping_ev: 0.0,
            interaction_ev: 2.0,
        })
        .unwrap();
        assert!(
            (decoupled.j1 - monolayer.j1).abs() < 1e-15,
            "{} {}",
            decoupled.j1,
            monolayer.j1
        );
        assert!((decoupled.j2 - monolayer.j2).abs() < 1e-18);
        assert!(decoupled.chirality.abs() < 1e-18);
        // Decoupled layers reduce to the monolayer.
        let single = BilayerExchange::static_limit(&BilayerParams {
            interlayer_hopping_ev: 0.0,
            ..params
        })
        .unwrap();
        let monolayer = params.hopping_ev;
        assert!(
            (single.j1 - (4.0 * monolayer * monolayer / u - 16.0 * monolayer.powi(4) / u3)).abs()
                < 1e-15
        );
        assert_eq!(single.jperp, 0.0);
        assert_eq!(single.jcross, 0.0);
        assert_eq!(single.k1, 0.0);
        assert!(
            BilayerExchange::static_limit(&BilayerParams {
                interaction_ev: 0.0,
                ..params
            })
            .is_err()
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
            let (up, down) = exchange
                .honeycomb_dispersion(spin, 0.0, &[0.0, 0.0])
                .unwrap();
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
        // Negative amplitudes are not accepted at the validated API boundary;
        // otherwise they can reach the Bessel backend's nonnegative-argument contract.
        let negative_lambda = DrivenHubbardParams {
            lambda: -0.4,
            ..base
        };
        assert!(negative_lambda.validate().is_err());
        assert!(DrivenExchange::honeycomb(&negative_lambda).is_err());
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
        for helicity in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(DrivenHubbardParams { helicity, ..base }.validate().is_err());
        }
        let overflowing_hopping = DrivenHubbardParams {
            hopping_ev: 1e100,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.0,
            helicity: 1.0,
            harmonic_max: 2,
        };
        assert!(overflowing_hopping.validate().is_ok());
        assert!(DrivenExchange::honeycomb(&overflowing_hopping).is_err());
    }

    #[test]
    fn honeycomb_public_functions_reject_invalid_values_and_unstable_energy_signs() {
        let exchange = DrivenExchange {
            j1: 1.0,
            j2: 0.0,
            chirality: 0.0,
        };
        assert!(exchange.honeycomb_functions(0.0, 0.0, &[0.0, 0.0]).is_err());
        assert!(
            exchange
                .honeycomb_functions(f64::NAN, 0.0, &[0.0, 0.0])
                .is_err()
        );
        assert!(
            exchange
                .honeycomb_functions(1.0, f64::INFINITY, &[0.0, 0.0])
                .is_err()
        );
        assert!(
            exchange
                .honeycomb_functions(1.0, 0.0, &[f64::NAN, 0.0])
                .is_err()
        );
        // A<0 and |B|>0 gives a positive squared radicand at K=4, but is not
        // a stable positive-energy magnon branch and must not pass by squaring.
        let (negative_a, b_abs, _) = exchange.honeycomb_functions(1.0, 4.0, &[0.0, 0.0]).unwrap();
        assert!(negative_a < -b_abs);
        assert!((negative_a * negative_a - b_abs * b_abs) > 0.0);
        assert!(
            exchange
                .honeycomb_dispersion(1.0, 4.0, &[0.0, 0.0])
                .is_err()
        );
        assert!(exchange.honeycomb_dispersion(1.0, 0.0, &[0.0, 0.0]).is_ok());
        for coefficient_scale in [1e200, 1e-200] {
            let scaled = DrivenExchange {
                j1: coefficient_scale,
                j2: 0.0,
                chirality: 0.0,
            };
            let (omega, _) = scaled
                .honeycomb_dispersion(1.0, -coefficient_scale, &[0.0, 0.0])
                .unwrap();
            assert!(((omega / coefficient_scale) - 4.0).abs() < 1e-12);
        }
        assert!(exchange.honeycomb_magnon(1.0, 0.0).is_ok());
        assert!(
            DrivenExchange {
                j1: 0.0,
                ..exchange
            }
            .honeycomb_magnon(1.0, 0.0)
            .is_err()
        );
        assert!(
            DrivenExchange {
                chirality: f64::NAN,
                ..exchange
            }
            .honeycomb_magnon(1.0, 0.0)
            .is_err()
        );
        assert!(exchange.honeycomb_magnon(f64::MAX, 0.0).is_err());
    }

    #[test]
    fn static_limit_reproduces_the_perturbative_exchange() {
        // U != 1 distinguishes the fourth-order U^-3 law from U^-1.
        // At lambda=0 only m1=m2=m3=0 survives: F gives J2=-4t^4/U^3,
        // folded G gives +8t^4/U^3; their sum is +4t^4/U^3.
        let p = DrivenHubbardParams {
            interaction_ev: 2.0,
            ..params(0.0, 1.0)
        };
        let exchange = DrivenExchange::honeycomb(&p).unwrap();
        let expected_j1 = 0.0198;
        let expected_j2 = 0.00005;
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
        let (up, down) = undriven.honeycomb_dispersion(1.5, -0.01, &k).unwrap();
        assert!((up - down).abs() < 1e-12, "undriven split {}, {}", up, down);
        let (up, down) = driven.honeycomb_dispersion(1.5, -0.01, &k).unwrap();
        let split = up - down;
        assert!(split.abs() > 1e-6, "no splitting at K: {split:e}");
        // Opposite signs at the two valleys, and odd parity overall.
        let (up_prime, down_prime) = driven.honeycomb_dispersion(1.5, -0.01, &k_prime).unwrap();
        let split_prime = up_prime - down_prime;
        assert!(
            (split + split_prime).abs() < 1e-3 * split.abs(),
            "K and K' splits {split:e} {split_prime:e}"
        );
        let (up_neg, down_neg) = driven
            .honeycomb_dispersion(1.5, -0.01, &[-k[0], -k[1]])
            .unwrap();
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
/// $$
/// \mathbf J\_{ij}(\mathbf R,t)
///   = \mathbf J\_{ij}(\mathbf R)\\,
///     e^{-i\\,\mathbf a(t)\cdot\mathbf d\_{ij\mathbf R}},
/// $$
///
/// whose harmonics are the Bessel coefficients $C\_m(\mathbf d)$ of the drive.
/// The quasienergies are obtained from the one-period propagator of the bosonic
/// Bogoliubov–de Gennes dynamics,
///
/// $$
/// i\frac{d\Psi}{dt} = gH(t)\Psi,\qquad g=\operatorname{diag}(1,-1),
/// $$
///
/// rather than from a Hermitian diagonalization: only eigenvectors with
/// positive metric norm are physical, and at a degenerate (Goldstone) mode the
/// metric-weighted matrix $gH$ is defective, so a naive eigen-decomposition of
/// $gS$ splits the photon replicas of a zero mode spuriously.
pub struct MagnonSambe<const DIM: usize = 3> {
    magnon: MagnonModel<DIM>,
    /// Exchange harmonics $\mathbf J^{(m)}(\mathbf R)$, indexed by
    /// `m + harmonic_span`.
    harmonics: Vec<Array3<Complex<f64>>>,
    omega_ev: f64,
    n_max: isize,
    harmonic_span: isize,
    n_time: usize,
    ac_axis: Option<[f64; 3]>,
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
        Self::build(magnon, drive, truncation, None)
    }

    /// Build a transverse Aharonov–Casher drive of a collinear magnet.
    ///
    /// `electric_drive` has three Cartesian components in inverse ångström:
    /// the electric field already multiplied by the magnetic moment factor
    /// `g μ_B/(ℏ c²)`. With the common unit spin axis `n`, the directed
    /// `S_i^- S_j^+` term gets `exp(-i (electric(t) × n) · d_ij)`.
    /// Only transverse intersite exchange is dressed. The static model must
    /// conserve spin about `n`; onsite and longitudinal terms are unchanged.
    /// In dimensions below three the lattice is embedded in the first `DIM`
    /// Cartesian coordinates. No charge hopping or Hubbard U is involved.
    pub fn new_aharonov_casher(
        magnon: &MagnonModel<DIM>,
        electric_drive: &FloquetDrive,
        spin_axis: [f64; 3],
        truncation: &FloquetTruncation,
    ) -> Result<Self> {
        Self::build(magnon, electric_drive, truncation, Some(spin_axis))
    }

    fn build(
        magnon: &MagnonModel<DIM>,
        drive: &FloquetDrive,
        truncation: &FloquetTruncation,
        ac_axis: Option<[f64; 3]>,
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
        let ac_axis = if let Some(axis) = ac_axis {
            crate::floquet::validate_floquet_drive::<3>(drive)?;
            let norm2 = axis.iter().map(|v| v * v).sum::<f64>();
            if !norm2.is_finite() || (norm2 - 1.0).abs() > 1e-8 {
                return Err(ac_error(
                    "ac_spin_axis",
                    "the AC spin axis must be a unit vector",
                ));
            }
            let unit = axis.map(|v| v / norm2.sqrt());
            for moment in magnon.axis.rows() {
                let projection = moment.iter().zip(unit).map(|(v, n)| v * n).sum::<f64>();
                if (0..3).any(|a| (moment[a] - projection * unit[a]).abs() > 1e-8) {
                    return Err(ac_error(
                        "ac_collinear_axis",
                        "all moments must be parallel or antiparallel to the AC spin axis",
                    ));
                }
            }
            Some(unit)
        } else {
            crate::floquet::validate_floquet_drive::<DIM>(drive)?;
            None
        };
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
        let onsite = (0..n).map(|i| magnon.onsite_scalar(i)).collect::<Vec<_>>();
        let harmonics = if let Some(axis) = ac_axis {
            ac_harmonics(magnon, drive, axis, span, harmonic_count, &onsite)?
        } else {
            let mut harmonics = Vec::with_capacity(harmonic_count);
            for index in 0..harmonic_count {
                let m = index as isize - span;
                let mut tensor = Array3::<Complex<f64>>::zeros((magnon.nR(), dim, dim));
                for i_r in 0..magnon.nR() {
                    for i in 0..n {
                        for j in 0..n {
                            let displacement = bond_displacement(magnon, i_r, i, j);
                            let coefficient =
                                bessel_peierls_coeffs(&displacement, drive, m, m, 0)?[0];
                            for alpha in 0..SC_SPIN {
                                for beta in 0..SC_SPIN {
                                    let mut value = magnon.ham[[i_r, 3 * i + alpha, 3 * j + beta]];
                                    if i_r == 0 && i == j && alpha == beta {
                                        // Onsite is Peierls-inactive. Remove the same
                                        // fixed-spin constant as in ordinary BdG.
                                        value -= onsite[i];
                                    }
                                    tensor[[i_r, 3 * i + alpha, 3 * j + beta]] =
                                        Complex::new(value, 0.0) * coefficient;
                                }
                            }
                        }
                    }
                }
                harmonics.push(tensor);
            }
            harmonics
        };
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
            ac_axis,
        })
    }

    /// The undriven magnon model.
    pub fn magnon(&self) -> &MagnonModel<DIM> {
        &self.magnon
    }

    /// Photon energy $\hbar\Omega$ in eV.
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

    /// Set the midpoint integration steps per period, independently of the
    /// harmonic cutoff. Converge both controls for quantitative results.
    pub fn with_time_steps(mut self, steps: usize) -> Result<Self> {
        if !(1..=(1 << 20)).contains(&steps) {
            return Err(ac_error(
                "floquet_time_steps",
                "time steps must be in 1..=2^20",
            ));
        }
        self.n_time = steps;
        Ok(self)
    }

    /// Time-dependent Bogoliubov–de Gennes matrix $H(\mathbf q,t)$.
    ///
    /// The time dependence is applied to the *real-space* exchange tensors,
    /// where
    /// $\mathbf J\_{ji}(-\mathbf R,t) = \mathbf J\_{ij}(\mathbf R,t)^\dagger$
    /// holds by construction, and the BdG matrix is then built once per time
    /// step. Building a single Fourier harmonic instead would not be Hermitian,
    /// because only the pair $H^{(m)}$, $H^{(-m)}$ is adjoint-related.
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
        let angle = self.omega_ev * time;
        let mut phases = Vec::with_capacity(self.harmonics.len());
        for index in 0..self.harmonics.len() {
            let m = index as isize - self.harmonic_span;
            let phase = if m == 0 { 0.0 } else { -(m as f64) * angle };
            if !phase.is_finite() {
                return Err(ac_error(
                    "finite_floquet_phase",
                    "the Floquet harmonic time phase is not finite",
                ));
            }
            phases.push(Complex::from_polar(1.0, phase));
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
            let turns = (0..DIM)
                .map(|axis| self.magnon.hamR[[i_r, axis]] as f64 * point[axis])
                .sum::<f64>();
            let factor = super::physics::cis_turns(-turns);
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

    /// One-period propagator $U(T)$ of $i\\,d\Psi/dt = gH(t)\Psi$.
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

    /// Quasienergies at one $\mathbf q$, folded into
    /// $\lbrack-\hbar\Omega/2,\\ \hbar\Omega/2)$, ascending.
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

    /// Folded AC quasienergies sorted separately by spin `(-1, +1)` along the AC axis.
    ///
    /// The ordering within each sector does not track Floquet bands or unfold
    /// photon replicas. Match the corresponding branches before computing a splitting.
    pub fn ac_spin_quasienergies_at(&self, q: &Array1<f64>) -> Result<[Array1<f64>; 2]> {
        let n_axis = self.ac_axis.ok_or_else(|| {
            ac_error(
                "ac_spin_axis",
                "spin sectors require an Aharonov–Casher model",
            )
        })?;
        let n = self.magnon.nspin();
        let propagator = self.propagator_at(q)?;
        let mut sectors = [Vec::new(), Vec::new()];
        for (i, moment) in self.magnon.axis.rows().into_iter().enumerate() {
            let eta = moment.iter().zip(n_axis).map(|(v, a)| v * a).sum::<f64>();
            // A Holstein–Primakoff creation on a moment +n lowers S_n.
            let particle = usize::from(eta < 0.0);
            sectors[particle].push(i);
            sectors[1 - particle].push(i + n);
        }
        for &a in &sectors[0] {
            for &b in &sectors[1] {
                if [propagator[[a, b]], propagator[[b, a]]]
                    .iter()
                    .any(|z| !z.norm().is_finite() || z.norm() > 1e-8)
                {
                    return Err(ac_error(
                        "ac_sector_leakage",
                        "the propagator does not preserve the AC spin sectors",
                    ));
                }
            }
        }
        let mut out = [Array1::zeros(0), Array1::zeros(0)];
        for (keys, result) in sectors.iter().zip(&mut out) {
            let matrix = Array2::from_shape_fn((keys.len(), keys.len()), |(a, b)| {
                propagator[[keys[a], keys[b]]]
            });
            let (values, vectors) = matrix.eig()?;
            let mut energies = Vec::new();
            for (column, value) in values.iter().enumerate() {
                if !value.norm().is_finite() || (value.norm() - 1.0).abs() > 1e-6 {
                    return Err(ac_error(
                        "driven_system_unstable",
                        format!("AC sector propagator eigenvalue {value:?} is off the unit circle"),
                    ));
                }
                let norm = keys
                    .iter()
                    .enumerate()
                    .map(|(row, &key)| {
                        vectors[[row, column]].norm_sqr() * if key < n { 1.0 } else { -1.0 }
                    })
                    .sum::<f64>();
                if norm > 1e-9 {
                    let energy = (-value.arg() / std::f64::consts::TAU) * self.omega_ev;
                    energies.push(crate::floquet::fold_quasienergy(energy, self.omega_ev));
                }
            }
            if energies.len() != keys.iter().filter(|&&key| key < n).count() {
                return Err(ac_error(
                    "positive_norm_quasienergies",
                    "could not resolve the physical branches of an AC spin sector",
                ));
            }
            energies.sort_by(f64::total_cmp);
            *result = Array1::from_vec(energies);
        }
        Ok(out)
    }

    /// Quasienergies of a list of $\mathbf q$ points, shape `(n_q, n_band)`.
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

fn ac_error(invariant: &'static str, message: impl Into<String>) -> TbError {
    TbError::InvalidMagnonInvariant {
        invariant,
        message: message.into(),
    }
}

fn ac_harmonics<const DIM: usize>(
    magnon: &MagnonModel<DIM>,
    drive: &FloquetDrive,
    n: [f64; 3],
    span: isize,
    count: usize,
    onsite: &[f64],
) -> Result<Vec<Array3<Complex<f64>>>> {
    if magnon
        .ham
        .len()
        .checked_mul(count)
        .is_none_or(|len| len > 1 << 27)
    {
        return Err(ac_error(
            "ac_harmonic_allocation",
            "AC exchange harmonic table exceeds 2^27 complex elements",
        ));
    }
    let cross = [[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]];
    let mut harmonics = vec![Array3::<Complex<f64>>::zeros(magnon.ham.raw_dim()); count];
    let translations: std::collections::HashMap<Vec<isize>, usize> = magnon
        .hamR
        .rows()
        .into_iter()
        .enumerate()
        .map(|(r, row)| (row.to_vec(), r))
        .collect();
    for (r, cell) in magnon.hamR.rows().into_iter().enumerate() {
        let reverse = translations[&cell.iter().map(|&x| -x).collect::<Vec<_>>()];
        if r > reverse {
            continue;
        }
        for (i, &scalar) in onsite.iter().enumerate() {
            for j in 0..magnon.nspin() {
                if r == reverse && i > j {
                    continue;
                }
                let onsite_block = r == 0 && i == j;
                let forward = magnon.jtensor(i, j, r)?;
                let backward = magnon.jtensor(j, i, reverse)?;
                let raw: [[f64; 3]; 3] = std::array::from_fn(|a| {
                    std::array::from_fn(|b| {
                        let mean = f64::midpoint(forward[a][b], backward[b][a]);
                        mean - if onsite_block && a == b { scalar } else { 0.0 }
                    })
                });
                let scale = raw.iter().flatten().map(|x| x.abs()).fold(0.0, f64::max);
                if scale == 0.0 {
                    continue;
                }
                // A rotation of one endpoint preserves real reciprocity only
                // when the original bond commutes with rotations about n.
                for a in 0..3 {
                    for b in 0..3 {
                        let difference: f64 = (0..3)
                            .map(|k| {
                                (raw[a][k] / scale) * cross[k][b]
                                    - cross[a][k] * (raw[k][b] / scale)
                            })
                            .sum();
                        if difference.abs() > 1e-10 {
                            return Err(ac_error(
                                "ac_spin_conservation",
                                format!(
                                    "exchange block (row {r}, sites {i},{j}) does not conserve spin about the AC axis"
                                ),
                            ));
                        }
                    }
                }
                let index = |a, b| [r, 3 * i + a, 3 * j + b];
                if onsite_block {
                    for a in 0..3 {
                        for b in 0..3 {
                            harmonics[span as usize][index(a, b)] = raw[a][b].into();
                        }
                    }
                    continue;
                }
                let generator: [[f64; 3]; 3] = raw
                    .map(|row| std::array::from_fn(|b| (0..3).map(|k| row[k] * cross[k][b]).sum()));
                let longitudinal: [[f64; 3]; 3] = raw.map(|row| {
                    let along = row.iter().zip(n).map(|(v, x)| v * x).sum::<f64>();
                    n.map(|x| along * x)
                });
                // A longitudinal-only bond has no AC phase, even for a
                // field that would exceed the harmonic backend on other bonds.
                if generator.iter().flatten().all(|&value| value == 0.0) {
                    for a in 0..3 {
                        for b in 0..3 {
                            harmonics[span as usize][index(a, b)] = raw[a][b].into();
                            harmonics[span as usize][[reverse, 3 * j + b, 3 * i + a]] =
                                raw[a][b].into();
                        }
                    }
                    continue;
                }
                let d = crate::floquet::link_displacement_cartesian(
                    &magnon.lat,
                    magnon.orb.row(i),
                    magnon.orb.row(j),
                    magnon.hamR.row(r),
                );
                // (electric × n) · d = electric · (n × d), in Cartesian coordinates.
                let projected = Array1::from_iter(
                    cross
                        .iter()
                        .map(|row| row.iter().zip(&d).map(|(v, x)| v * x).sum::<f64>()),
                );
                if projected.iter().any(|x| !x.is_finite()) {
                    return Err(ac_error(
                        "ac_bond_displacement",
                        format!("non-finite AC link (row {r}, sites {i},{j})"),
                    ));
                }
                let coeffs = bessel_peierls_coeffs(&projected, drive, -span, span, 0)?;
                for a in 0..3 {
                    for b in 0..3 {
                        let longitudinal = longitudinal[a][b];
                        let generator = generator[a][b];
                        for h in 0..count {
                            let minus = coeffs[h];
                            let plus = coeffs[count - 1 - h].conj();
                            let cos = 0.5 * (minus + plus);
                            let sin_minus = Complex::new(0.0, -0.5) * (minus - plus);
                            let val = (raw[a][b] - longitudinal) * cos
                                + generator * sin_minus
                                + if h == span as usize {
                                    longitudinal
                                } else {
                                    0.0
                                };
                            if !val.re.is_finite() || !val.im.is_finite() {
                                return Err(ac_error(
                                    "finite_ac_exchange",
                                    "AC exchange harmonic is non-finite",
                                ));
                            }
                            harmonics[h][index(a, b)] = val;
                            harmonics[h][[reverse, 3 * j + b, 3 * i + a]] = val;
                        }
                    }
                }
            }
        }
    }
    Ok(harmonics)
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

/// Cartesian displacement of the bond $(\mathbf R,i,j)$,
///
/// $$
/// \mathbf d\_{ij}(\mathbf R)
///   = (\mathbf R + \boldsymbol\tau\_j - \boldsymbol\tau\_i)\cdot\mathbf L.
/// $$
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
    /// Every exchange bond is multiplied by the drive's time-averaged Peierls
    /// coefficient $C\_0(\mathbf d)$, so the result is an ordinary
    /// [`MagnonModel`] with renormalized couplings: the same interface
    /// as the electronic
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

// ---------------------------------------------------------------------------
// Bilayer linear spin-wave theory (paper, Supplemental Material section II D)
// ---------------------------------------------------------------------------

/// Intralayer blocks $(A\_1,A\_2)$ and the interlayer pairing $C$ at one
/// $\mathbf q$.
type LayerBlocks = (
    Array2<Complex<f64>>,
    Array2<Complex<f64>>,
    Array2<Complex<f64>>,
);

/// Linear spin-wave theory of the AA-stacked bilayer.
///
/// Transcribes the explicit blocks of the Supplemental Material for an A-type
/// antiferromagnet (ferromagnetic inside each layer, antiparallel between the
/// layers) with $(S\_{A1},S\_{B1},S\_{A2},S\_{B2})=(S\_1,S\_2,S\_2,S\_1)$. The two spin
/// sectors are diagonalized separately through the paraunitary problem, so the
/// four bands keep their spin labels.
#[derive(Clone, Copy, Debug)]
pub struct BilayerSpinWave {
    /// Exchange coefficients of the bilayer.
    pub exchange: BilayerExchange,
    /// Local spin lengths $(S\_{A1},S\_{B1},S\_{A2},S\_{B2})$ in units of
    /// $\hbar$.
    pub spins: [f64; 4],
    /// Easy-axis single-ion anisotropy $K$ in eV (negative for an easy axis).
    pub anisotropy_ev: f64,
    /// Phenomenological intralayer ferromagnetic direct exchange $J\_D>0$ in
    /// eV, subtracted from $J\_1$ exactly as in the Supplemental Material.
    pub direct_exchange_ev: f64,
}

impl BilayerSpinWave {
    /// Validate the parameters.
    pub fn validate(&self) -> Result<()> {
        if self
            .spins
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "positive_bilayer_spins",
                message: format!("the local spins must be positive, found {:?}", self.spins),
            });
        }
        if !self.anisotropy_ev.is_finite() || !self.direct_exchange_ev.is_finite() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_bilayer_parameter",
                message: "the anisotropy and the direct exchange must be finite".into(),
            });
        }
        // The analytic blocks use only S1,S2 and assume this exact crossed
        // assignment; silently ignoring the other two supplied spins is invalid.
        if self.spins[2] != self.spins[1] || self.spins[3] != self.spins[0] {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "crossed_bilayer_spins",
                message: "the bilayer requires spins [S1, S2, S2, S1]".into(),
            });
        }
        let j = self.exchange;
        if [j.j1, j.j2, j.jperp, j.jcross, j.k1, j.k2, j.k3, j.chirality]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_bilayer_exchange",
                message: "all bilayer exchange coefficients must be finite".into(),
            });
        }
        Ok(())
    }

    /// Structure factors $(\gamma,c,s)$ at $\mathbf q$ in fractional
    /// coordinates.
    ///
    /// $$
    /// \gamma\_{\mathbf k} = \sum\_\mu e^{i\mathbf k\cdot\boldsymbol\delta\_\mu},
    /// \qquad
    /// c\_{\mathbf k} = \sum\_\nu\cos(\mathbf k\cdot\mathbf b\_\nu),
    /// \qquad
    /// s\_{\mathbf k} = \sum\_\nu\sin(\mathbf k\cdot\mathbf b\_\nu),
    /// $$
    ///
    /// with the vectors of the Supplemental Material.
    pub fn structure_factors(&self, q: &[f64; 2]) -> (Complex<f64>, f64, f64) {
        let delta = [
            [1.0 / 3.0, -2.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [-2.0 / 3.0, 1.0 / 3.0],
        ];
        let bl = [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]];
        let mut gamma = Complex::new(0.0, 0.0);
        for d in delta {
            let phase = std::f64::consts::TAU * (q[0] * d[0] + q[1] * d[1]);
            gamma += Complex::from_polar(1.0, phase);
        }
        let mut c = 0.0;
        let mut s = 0.0;
        for b in bl {
            let phase = std::f64::consts::TAU * (q[0] * b[0] + q[1] * b[1]);
            c += phase.cos();
            s += phase.sin();
        }
        (gamma, c, s)
    }

    fn layer_blocks(&self, q: &[f64; 2]) -> Result<LayerBlocks> {
        self.validate()?;
        if q.iter().any(|value| !value.is_finite()) {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_bilayer_q",
                message: "bilayer momentum components must be finite".into(),
            });
        }
        let s = self.spins;
        let j = self.exchange;
        let (gamma, c, s_sum) = self.structure_factors(q);
        let root = (s[0] * s[1]).sqrt();
        let j1p = j.j1 - self.direct_exchange_ev + j.k1 * s[0] * s[1];
        let vertical = j.jperp - 3.0 * j.k2 * s[0] * s[1];
        let diagonal_a = j.jcross - j.k3 * s[1] * s[1];
        let diagonal_b = j.jcross - j.k3 * s[0] * s[0];
        let eps_a = (-3.0 * j1p + vertical) * s[1]
            + 3.0 * diagonal_a * s[0]
            + 2.0 * j.j2 * s[0] * (c - 3.0)
            - 2.0 * self.anisotropy_ev * s[0];
        let eps_b = (-3.0 * j1p + vertical) * s[0]
            + 3.0 * diagonal_b * s[1]
            + 2.0 * j.j2 * s[1] * (c - 3.0)
            - 2.0 * self.anisotropy_ev * s[1];
        let chirality = 2.0 * j.chirality * s[0] * s[1] * s_sum;
        let hopping = Complex::new(j1p * root, 0.0) * gamma;
        let mut a1 = Array2::<Complex<f64>>::zeros((2, 2));
        a1[[0, 0]] = Complex::new(eps_a - chirality, 0.0);
        a1[[1, 1]] = Complex::new(eps_b + chirality, 0.0);
        a1[[0, 1]] = hopping;
        a1[[1, 0]] = hopping.conj();
        let mut a2 = Array2::<Complex<f64>>::zeros((2, 2));
        a2[[0, 0]] = Complex::new(eps_b - chirality, 0.0);
        a2[[1, 1]] = Complex::new(eps_a + chirality, 0.0);
        a2[[0, 1]] = hopping;
        a2[[1, 0]] = hopping.conj();
        let mut pairing = Array2::<Complex<f64>>::zeros((2, 2));
        pairing[[0, 0]] = Complex::new(vertical * root, 0.0);
        pairing[[1, 1]] = Complex::new(vertical * root, 0.0);
        pairing[[0, 1]] = Complex::new(diagonal_a * s[0], 0.0) * gamma;
        pairing[[1, 0]] = Complex::new(diagonal_b * s[1], 0.0) * gamma.conj();
        Ok((a1, a2, pairing))
    }

    /// Magnon frequencies at $\mathbf q$, ascending (four bands).
    ///
    /// The two spin sectors of the Supplemental Material are diagonalized
    /// through their `4 x 4` Bogoliubov–de Gennes matrices, returning two
    /// nonnegative stable frequencies per sector, including zero modes. For a
    /// defective Goldstone mode only its frequency is returned, not a
    /// normalizable positive-norm eigenvector.
    pub fn dispersion(&self, q: &[f64; 2]) -> Result<Array1<f64>> {
        let (up, down) = self.sector_bands(q)?;
        let mut spectra = [up[0], up[1], down[0], down[1]];
        spectra.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Array1::from(spectra.to_vec()))
    }

    /// Spin-resolved bands (up sector, down sector) at $\mathbf q$, each
    /// ascending.
    /// Each sector contains two nonnegative stable frequencies, including
    /// Goldstone zeros; no normalized zero-mode eigenvectors are promised.
    pub fn sector_bands(&self, q: &[f64; 2]) -> Result<([f64; 2], [f64; 2])> {
        let (up, down) = self.sector_matrices(q)?;
        let mut up_bands = [0.0f64; 2];
        let mut down_bands = [0.0f64; 2];
        for (matrix, target) in [(up, &mut up_bands), (down, &mut down_bands)] {
            let values = paraunitary_spectrum(&matrix, 2)?;
            target[0] = values[0];
            target[1] = values[1];
        }
        Ok((up_bands, down_bands))
    }

    // Keep the production assembly directly testable without exposing raw blocks.
    fn sector_matrices(
        &self,
        q: &[f64; 2],
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        let (a1, a2, pairing) = self.layer_blocks(q)?;
        let minus_q = [-q[0], -q[1]];
        let (a1_neg, a2_neg, pairing_neg) = self.layer_blocks(&minus_q)?;
        let dagger = |matrix: &Array2<Complex<f64>>| matrix.t().mapv(|value| value.conj());
        let transpose = |matrix: &Array2<Complex<f64>>| matrix.t().to_owned();
        // Upper sector: [[A1(k), C(k)], [C^dag(k), A2^T(-k)]].
        let mut up = Array2::<Complex<f64>>::zeros((4, 4));
        for i in 0..2 {
            for k in 0..2 {
                up[[i, k]] = a1[[i, k]];
                up[[i, 2 + k]] = pairing[[i, k]];
                up[[2 + i, k]] = dagger(&pairing)[[i, k]];
                up[[2 + i, 2 + k]] = transpose(&a2_neg)[[i, k]];
            }
        }
        // Lower sector: [[A2(k), C^T(-k)], [C*(-k), A1^T(-k)]].
        let mut down = Array2::<Complex<f64>>::zeros((4, 4));
        for i in 0..2 {
            for k in 0..2 {
                down[[i, k]] = a2[[i, k]];
                down[[i, 2 + k]] = transpose(&pairing_neg)[[i, k]];
                down[[2 + i, k]] = pairing_neg[[i, k]].conj();
                down[[2 + i, 2 + k]] = transpose(&a1_neg)[[i, k]];
            }
        }
        Ok((up, down))
    }
}

/// Stable bosonic frequencies, including PSD Goldstone modes.
///
/// For $H=KK^\dagger \ge 0$, the shared Colpa solver diagonalizes
/// $M=K^\dagger gK$, not $H$. If $Mv=\omega v$ with $\omega>0$, then
/// $y=gKv$ is a $gH$ eigenvector and
/// $y^\dagger gy = \omega\lVert v\rVert^2 > 0$. Zero frequencies need no
/// normalizable Jordan eigenvector.
fn paraunitary_spectrum(matrix: &Array2<Complex<f64>>, n: usize) -> Result<Array1<f64>> {
    let size = matrix.nrows();
    if n == 0 || n.checked_mul(2) != Some(size) || matrix.ncols() != size {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "bdg_shape",
            message: format!(
                "expected a nonempty 2n-square BdG matrix for n={n}, found {:?}",
                matrix.dim()
            ),
        });
    }
    if matrix
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(TbError::InvalidMagnonInvariant {
            invariant: "finite_bilayer_bdg",
            message: "all BdG entries must be finite".into(),
        });
    }
    // Component magnitudes avoid overflow of |z| for finite complex entries.
    // Normalize BEFORE LAPACK: the shared PSD tolerance (including its floor)
    // is then relative to H, never to an arbitrary 1 eV energy unit.
    let scale = matrix
        .iter()
        .map(|z| z.re.abs().max(z.im.abs()))
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(Array1::zeros(n));
    }
    let normalized = matrix.mapv(|z| z / scale);
    let roundoff = 64.0 * f64::EPSILON * size as f64;
    for row in 0..size {
        for column in 0..size {
            if (normalized[[row, column]] - normalized[[column, row]].conj()).norm() > roundoff {
                return Err(TbError::InvalidMagnonInvariant {
                    invariant: "hermitian_bilayer_bdg",
                    message: "the BdG matrix must be Hermitian".into(),
                });
            }
        }
    }
    // The shared Colpa solver also certifies positivity. This entry point
    // receives an already formed matrix, so there is no A-C formation scale.
    let mut frequencies = super::physics::paraunitary_frequencies(&normalized, n, &[])?;
    for value in &mut frequencies {
        *value *= scale;
        if !value.is_finite() {
            return Err(TbError::InvalidMagnonInvariant {
                invariant: "finite_bilayer_frequency",
                message: "a magnon frequency overflowed".into(),
            });
        }
    }
    Ok(frequencies)
}

#[cfg(test)]
mod bilayer_sw_tests {
    use super::*;

    #[test]
    fn solver_rejects_nonhermitian_input() {
        let mut h = Array2::<Complex<f64>>::eye(2);
        h[[0, 1]] = Complex::new(0.2, 0.1);
        for scale in [1e-200, 1e-12, 1.0, 1e200] {
            let scaled = h.mapv(|z| z * scale);
            assert!(
                paraunitary_spectrum(&scaled, 1).is_err(),
                "non-Hermitian H accepted at scale={scale:e}"
            );
        }
    }

    #[test]
    fn solver_rejects_positive_norm_negative_energy() {
        // gH has a negative-energy eigenvector (1,0) with positive g-norm.
        let h = -Array2::<Complex<f64>>::eye(2);
        assert!(
            paraunitary_spectrum(&h, 1).is_err(),
            "negative-energy reference accepted"
        );
    }

    #[test]
    fn solver_rejects_tiny_imaginary_spectrum() {
        let scale = 1e-12;
        let h = ndarray::array![
            [Complex::new(0.0, 0.0), Complex::new(scale, 0.0)],
            [Complex::new(scale, 0.0), Complex::new(0.0, 0.0)]
        ];
        assert!(paraunitary_spectrum(&h, 1).is_err());
    }

    #[test]
    fn solver_rejects_resolved_tiny_negative_direction() {
        for scale in [1e-200, 1e-12, 1.0, 1e200] {
            let h = Array2::from_diag(&ndarray::array![
                Complex::new(-1e-14 * scale, 0.0),
                Complex::new(scale, 0.0)
            ]);
            assert!(
                paraunitary_spectrum(&h, 1).is_err(),
                "scale={scale:e}: resolved negative direction accepted"
            );
        }
    }

    #[test]
    fn solver_scaled_bosonic_oracles_and_instabilities() {
        for scale in [1.0, 1e-3, 1e-12, 1e-200, 1e200] {
            // Independent single-mode oracle: gH has frequencies ±sqrt(A²-|B|²),
            // NOT the ordinary Hermitian eigenvalues A±|B|. Complex B tests phase.
            let h = ndarray::array![
                [Complex::new(5.0, 0.0), Complex::new(0.0, 3.0)],
                [Complex::new(0.0, -3.0), Complex::new(5.0, 0.0)]
            ]
            .mapv(|z| z * scale);
            let omega = paraunitary_spectrum(&h, 1).unwrap()[0];
            assert!(
                (omega / scale - 4.0).abs() < 1e-12,
                "scale={scale:e}, omega={omega:e}"
            );
            // A=0, B=1: exactly imaginary eigenvalues ±i*scale at every scale.
            let unstable = ndarray::array![
                [Complex::new(0.0, 0.0), Complex::new(scale, 0.0)],
                [Complex::new(scale, 0.0), Complex::new(0.0, 0.0)]
            ];
            assert!(
                paraunitary_spectrum(&unstable, 1).is_err(),
                "scale={scale:e}: imaginary spectrum accepted"
            );
            let negative = -Array2::<Complex<f64>>::eye(2) * scale;
            assert!(paraunitary_spectrum(&negative, 1).is_err());
            // A small but resolved negative direction in an otherwise positive
            // H remains unstable; normalization cannot turn it into roundoff.
            let saddle = Array2::from_diag(&ndarray::array![
                Complex::new(-1e-8 * scale, 0.0),
                Complex::new(scale, 0.0)
            ]);
            assert!(paraunitary_spectrum(&saddle, 1).is_err());
            // Close but distinct frequencies must not be replaced by a cluster representative.
            let diagonal = Array2::from_diag(&ndarray::array![
                Complex::new(scale, 0.0),
                Complex::new(scale * (1.0 + 1e-7), 0.0),
                Complex::new(scale, 0.0),
                Complex::new(scale * (1.0 + 1e-7), 0.0)
            ]);
            let omega = paraunitary_spectrum(&diagonal, 2).unwrap();
            assert!((omega[0] / scale - 1.0).abs() < 1e-13);
            assert!(
                (omega[1] / scale - (1.0 + 1e-7)).abs() < 1e-13,
                "scale={scale:e}: {omega:?}"
            );
            // Singular PSD H has a defective gH: its frequencies are zero, not
            // two normalizable finite-g-norm eigenvectors of a Jordan block.
            let goldstone = Array2::from_elem((2, 2), Complex::new(scale, 0.0));
            let zero = paraunitary_spectrum(&goldstone, 1).unwrap()[0];
            assert!(
                zero >= 0.0 && zero / scale < 1e-7,
                "scale={scale:e}: {zero:e}"
            );
        }
    }

    #[test]
    fn solver_validates_shape_finiteness_and_zero_matrix() {
        for shape in [(2, 3), (3, 3), (0, 0)] {
            assert!(paraunitary_spectrum(&Array2::zeros(shape), 1).is_err());
        }
        assert!(paraunitary_spectrum(&Array2::zeros((0, 0)), 0).is_err());
        assert!(paraunitary_spectrum(&Array2::zeros((2, 2)), usize::MAX).is_err());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut h = Array2::<Complex<f64>>::eye(2);
            h[[0, 0]] = Complex::new(value, 0.0);
            assert!(paraunitary_spectrum(&h, 1).is_err());
            h[[0, 0]] = Complex::new(1.0, value);
            assert!(paraunitary_spectrum(&h, 1).is_err());
        }
        assert_eq!(
            paraunitary_spectrum(&Array2::zeros((4, 4)), 2).unwrap(),
            ndarray::array![0.0, 0.0]
        );
    }

    #[test]
    fn bilayer_public_contract_checks_crossed_spins_and_finite_inputs() {
        let wave = coupled_chiral_wave();
        let mut invalid = wave;
        invalid.spins[2] = 1.1;
        assert!(invalid.validate().is_err());
        invalid = wave;
        invalid.spins[3] = 1.6;
        assert!(invalid.validate().is_err());
        // S1 < S2 is valid: this is a crossed assignment, not an ordering rule.
        let mut swapped = wave;
        swapped.spins = [1.0, 1.5, 1.5, 1.0];
        assert!(swapped.sector_bands(&[0.13, 0.27]).is_ok());
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for field in 0..8 {
                let mut invalid = wave;
                let j = &mut invalid.exchange;
                let fields = [
                    &mut j.j1,
                    &mut j.j2,
                    &mut j.jperp,
                    &mut j.jcross,
                    &mut j.k1,
                    &mut j.k2,
                    &mut j.k3,
                    &mut j.chirality,
                ];
                *fields.into_iter().nth(field).unwrap() = value;
                assert!(invalid.validate().is_err(), "exchange field {field}");
            }
            for q in [[value, 0.0], [0.0, value]] {
                assert!(wave.sector_bands(&q).is_err());
            }
        }
    }

    #[test]
    fn bilayer_public_bands_are_energy_rescaling_covariant() {
        let base = coupled_chiral_wave();
        let q = [0.13, 0.27];
        let (up, down) = base.sector_bands(&q).unwrap();
        for scale in [1e-200, 1e-12, 1e-3, 1e200] {
            let mut wave = base;
            let j = &mut wave.exchange;
            for value in [
                &mut j.j1,
                &mut j.j2,
                &mut j.jperp,
                &mut j.jcross,
                &mut j.k1,
                &mut j.k2,
                &mut j.k3,
                &mut j.chirality,
                &mut wave.anisotropy_ev,
                &mut wave.direct_exchange_ev,
            ] {
                *value *= scale;
            }
            let (scaled_up, scaled_down) = wave.sector_bands(&q).unwrap();
            for (expected, actual) in up
                .into_iter()
                .chain(down)
                .zip(scaled_up.into_iter().chain(scaled_down))
            {
                assert!(
                    (actual / scale - expected).abs() < 1e-13,
                    "scale={scale:e}: {actual:e} vs {expected:e}"
                );
            }
        }
    }

    fn monolayer() -> DrivenExchange {
        // A finite `lambda` keeps the chirality non-zero.
        DrivenExchange::honeycomb(&DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.4,
            helicity: 1.0,
            harmonic_max: 8,
        })
        .unwrap()
    }

    fn bilayer(exchange: BilayerExchange, spin: f64) -> BilayerSpinWave {
        // The A-type reference needs a ferromagnetic intralayer coupling, so the
        // phenomenological direct exchange must dominate the Hubbard J1.
        BilayerSpinWave {
            exchange,
            spins: [spin, spin, spin, spin],
            anisotropy_ev: 0.0,
            direct_exchange_ev: 2.0 * exchange.j1,
        }
    }

    fn coupled_chiral_wave() -> BilayerSpinWave {
        BilayerSpinWave {
            exchange: BilayerExchange {
                j1: 0.008,
                j2: 0.00002,
                jperp: 0.0012,
                jcross: 0.00007,
                k1: 0.00011,
                k2: 0.00013,
                k3: -0.00017,
                chirality: 0.00009,
            },
            spins: [1.5, 1.0, 1.0, 1.5],
            anisotropy_ev: -0.002,
            direct_exchange_ev: 0.016,
        }
    }

    // Independent explicit S153 matrices: original SM.tex lines 2476–2515
    // (source attachment 0a3b5777). In particular C is NOT symmetric, and the
    // lower-left down block is C*(-k), not C^dag(-k). No production block or
    // structure-factor helper is used to construct this reference.
    fn s153_matrices(w: &BilayerSpinWave, q: [f64; 2]) -> [Array2<Complex<f64>>; 2] {
        let j = w.exchange;
        let (s1, s2) = (w.spins[0], w.spins[1]);
        let t = std::f64::consts::TAU;
        let g = Complex::from_polar(1.0, t * (q[0] - 2.0 * q[1]) / 3.0)
            + Complex::from_polar(1.0, t * (q[0] + q[1]) / 3.0)
            + Complex::from_polar(1.0, t * (-2.0 * q[0] + q[1]) / 3.0);
        let c = (t * q[0]).cos() + (t * (q[1] - q[0])).cos() + (t * q[1]).cos();
        let sine = (t * q[0]).sin() + (t * (q[1] - q[0])).sin() - (t * q[1]).sin();
        let jp = j.j1 - w.direct_exchange_ev + j.k1 * s1 * s2;
        let v = Complex::new((j.jperp - 3.0 * j.k2 * s1 * s2) * (s1 * s2).sqrt(), 0.0);
        let x = (j.jcross - j.k3 * s2 * s2) * s1 * g;
        let y = (j.jcross - j.k3 * s1 * s1) * s2 * g.conj();
        let h = jp * (s1 * s2).sqrt() * g;
        let ea = (-3.0 * jp + j.jperp - 3.0 * j.k2 * s1 * s2) * s2
            + 3.0 * (j.jcross - j.k3 * s2 * s2) * s1
            + 2.0 * j.j2 * s1 * (c - 3.0)
            - 2.0 * w.anisotropy_ev * s1;
        let eb = (-3.0 * jp + j.jperp - 3.0 * j.k2 * s1 * s2) * s1
            + 3.0 * (j.jcross - j.k3 * s1 * s1) * s2
            + 2.0 * j.j2 * s2 * (c - 3.0)
            - 2.0 * w.anisotropy_ev * s2;
        let d = 2.0 * j.chirality * s1 * s2 * sine;
        let a = Complex::new(ea - d, 0.0);
        let b = Complex::new(eb + d, 0.0);
        let an = Complex::new(ea + d, 0.0);
        let bn = Complex::new(eb - d, 0.0);
        [
            ndarray::array![
                [a, h, v, x],
                [h.conj(), b, y, v],
                [v, y.conj(), b, h],
                [x.conj(), v, h.conj(), a]
            ],
            ndarray::array![
                [bn, h, v, y.conj()],
                [h.conj(), an, x.conj(), v],
                [v, x, an, h],
                [y, v, h.conj(), bn]
            ],
        ]
    }

    #[test]
    fn coupled_chiral_sectors_are_hermitian_and_match_s153() {
        let wave = coupled_chiral_wave();
        for q in [[0.13, 0.27], [0.3, 0.1], [0.05, 0.45], [0.0, 0.0]] {
            let (up, down) = wave.sector_matrices(&q).unwrap();
            for (label, actual, expected) in [
                ("up", up, s153_matrices(&wave, q)[0].clone()),
                ("down", down, s153_matrices(&wave, q)[1].clone()),
            ] {
                let mut hermitian = 0.0_f64;
                let mut oracle = 0.0_f64;
                for i in 0..4 {
                    for k in 0..4 {
                        for z in [actual[[i, k]], expected[[i, k]]] {
                            assert!(
                                z.re.is_finite() && z.im.is_finite(),
                                "{label} at {q:?}: {z:?}"
                            );
                        }
                        hermitian = hermitian.max((actual[[i, k]] - actual[[k, i]].conj()).norm());
                        oracle = oracle.max((actual[[i, k]] - expected[[i, k]]).norm());
                    }
                }
                println!("S153 q={q:?} sector={label} hermitian={hermitian:e} oracle={oracle:e}");
                assert!(hermitian < 1e-14, "{label} at {q:?}: {hermitian:e}");
                assert!(oracle < 1e-14, "{label} at {q:?}: {oracle:e}");
            }
        }
    }

    #[test]
    fn coupled_chiral_general_momenta_keep_spin_labels() {
        let wave = coupled_chiral_wave();
        let mut residual = 0.0_f64;
        let mut splitting = 0.0_f64;
        for q in [[0.13, 0.27], [0.3, 0.1], [0.05, 0.45]] {
            let (up, down) = wave.sector_bands(&q).unwrap();
            let (up_neg, down_neg) = wave.sector_bands(&[-q[0], -q[1]]).unwrap();
            for band in 0..2 {
                residual = residual.max((up[band] - down_neg[band]).abs());
                residual = residual.max((down[band] - up_neg[band]).abs());
                splitting = splitting.max((up[band] - down[band]).abs());
                assert!(up[band] > 0.0 && down[band] > 0.0);
            }
        }
        println!("label residual={residual:e}, splitting={splitting:e}");
        assert!(residual < 1e-12, "{residual:e}");
        assert!(splitting > 1e-6, "{splitting:e}");
    }

    #[test]
    fn paper_parameter_gamma_has_real_dynamical_spectrum() {
        // Paper-parameter assembly regression using static interlayer approximation,
        // NOT a full driven Fig. 3 reproduction. SM.tex 2719–2733 and main.tex
        // 215–222: t=.05, tp=.015, U=1.25, W=1, lambda=.6, xi=-1, M=7,
        // JD=.012, (S1,S2)=(1,.8), K=0. Driven interlayer terms are not generated.
        let mut exchange = BilayerExchange::static_limit(&BilayerParams {
            hopping_ev: 0.05,
            interlayer_hopping_ev: 0.015,
            interaction_ev: 1.25,
        })
        .unwrap();
        let intra = DrivenExchange::honeycomb(&DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 1.0,
            lambda: 0.6,
            helicity: -1.0,
            harmonic_max: 7,
        })
        .unwrap();
        exchange.j1 = intra.j1;
        exchange.j2 = intra.j2;
        exchange.chirality = intra.chirality;
        let wave = BilayerSpinWave {
            exchange,
            spins: [1.0, 0.8, 0.8, 1.0],
            anisotropy_ev: 0.0,
            direct_exchange_ev: 0.012,
        };
        let (up, down) = wave.sector_matrices(&[0.0, 0.0]).unwrap();
        for (label, mut generator) in [("up", up), ("down", down)] {
            for row in 2..4 {
                for column in 0..4 {
                    generator[[row, column]] *= -1.0;
                }
            }
            let (values, _) = generator.eig().unwrap();
            assert!(
                values.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
                "{label}: {values:?}"
            );
            let imaginary = values.iter().map(|z| z.im.abs()).fold(0.0_f64, f64::max);
            println!("paper-parameter Gamma {label}: {values:?}; max_imag={imaginary:e}");
            // Absolute eV threshold, independent of the solver's scale floor.
            assert!(imaginary < 1e-10, "{label}: {values:?}");
        }
        let bands = wave.sector_bands(&[0.0, 0.0]);
        println!("paper-parameter public Gamma sector_bands: {bands:?}");
        let (up, down) =
            bands.expect("public Gamma solver must accept the real dynamical spectrum");
        assert!(up.iter().chain(down.iter()).all(|value| value.is_finite()));
    }

    #[test]
    fn ferromagnetic_layer_has_a_goldstone_mode() {
        // Decoupled layers: each is a ferromagnetic honeycomb.
        let exchange = monolayer();
        let wave = bilayer(BilayerExchange::decoupled(&exchange), 1.5);
        for q in [[0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0], [0.5, 0.0]] {
            let bands = wave.dispersion(&q).unwrap();
            assert_eq!(bands.len(), 4, "q = {q:?}");
            assert!(bands.iter().all(|value| value.is_finite()), "q = {q:?}");
            assert!(bands[0] >= -1e-9, "q = {q:?}: {bands:?}");
        }
        // The acoustic mode is exactly gapless at Gamma.
        let gamma_bands = wave.dispersion(&[0.0, 0.0]).unwrap();
        assert!(gamma_bands[0].abs() < 1e-9, "{gamma_bands:?}");
        // Away from Gamma the flat band is lifted.
        let zone = wave.dispersion(&[0.5, 0.0]).unwrap();
        assert!(zone[0] > 1e-6, "{zone:?}");
    }

    #[test]
    fn coupled_bilayer_is_even_in_momentum_without_chirality() {
        // A static drive turns the chirality off, and without it the two
        // sectors must be even in k; this exercises the C(k), C^T(-k) and
        // A^T(-k) arguments of the sector assembly.
        let static_exchange = DrivenExchange::honeycomb(&DrivenHubbardParams {
            hopping_ev: 0.05,
            interaction_ev: 1.25,
            photon_ev: 4.0,
            lambda: 0.0,
            helicity: 1.0,
            harmonic_max: 8,
        })
        .unwrap();
        let mut exchange = BilayerExchange::static_limit(&BilayerParams {
            hopping_ev: 0.05,
            interlayer_hopping_ev: 0.02,
            interaction_ev: 1.25,
        })
        .unwrap();
        // Keep the intralayer part of the driven monolayer and add the
        // interlayer couplings of the bilayer.
        exchange.j1 = static_exchange.j1;
        exchange.j2 = static_exchange.j2;
        exchange.chirality = 0.0;
        let mut wave = BilayerSpinWave {
            exchange,
            spins: [1.5, 1.0, 1.0, 1.5],
            anisotropy_ev: -0.002,
            direct_exchange_ev: 2.0 * exchange.j1,
        };
        wave.direct_exchange_ev = 2.0 * wave.exchange.j1;
        for q in [[0.13f64, 0.27], [0.3, 0.1], [0.05, 0.45]] {
            let forward = wave.dispersion(&q).unwrap();
            let backward = wave.dispersion(&[-q[0], -q[1]]).unwrap();
            assert!(forward.iter().all(|value| value.is_finite()), "{forward:?}");
            for index in 0..4 {
                assert!(
                    (forward[index] - backward[index]).abs() < 1e-10,
                    "q = {q:?}: {forward:?} vs {backward:?}"
                );
            }
        }
    }

    #[test]
    fn chirality_swaps_the_spin_sectors_between_the_valleys() {
        let exchange = monolayer();
        // The paper's crossed moment assignment needs S1 != S2 for the sectors
        // to differ; equal spins make them exactly degenerate.
        let mut wave = bilayer(BilayerExchange::decoupled(&exchange), 1.5);
        wave.spins = [1.5, 1.0, 1.0, 1.5];
        let k = [1.0 / 3.0, 2.0 / 3.0];
        let k_prime = [2.0 / 3.0, 1.0 / 3.0];
        let (up_k, down_k) = wave.sector_bands(&k).unwrap();
        let (up_prime, down_prime) = wave.sector_bands(&k_prime).unwrap();
        // Odd parity: the up sector at K equals the down sector at K'.
        for band in 0..2 {
            assert!(
                (up_k[band] - down_prime[band]).abs() < 1e-12,
                "{up_k:?} {down_prime:?}"
            );
            assert!(
                (down_k[band] - up_prime[band]).abs() < 1e-12,
                "{down_k:?} {up_prime:?}"
            );
        }
        // ... and the sectors are genuinely split at a given valley.
        assert!((up_k[0] - down_k[0]).abs() > 1e-6, "{up_k:?} {down_k:?}");
    }

    #[test]
    fn zero_chirality_keeps_the_valleys_degenerate() {
        let mut exchange = monolayer();
        exchange.chirality = 0.0;
        let wave = bilayer(BilayerExchange::decoupled(&exchange), 1.5);
        let at_k = wave.dispersion(&[1.0 / 3.0, 2.0 / 3.0]).unwrap();
        let at_prime = wave.dispersion(&[2.0 / 3.0, 1.0 / 3.0]).unwrap();
        for index in 0..4 {
            assert!(
                (at_k[index] - at_prime[index]).abs() < 1e-9,
                "{at_k:?} {at_prime:?}"
            );
        }
    }
}
