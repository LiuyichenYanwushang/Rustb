#![doc = include_str!("driven_theory.md")]

use super::{Magnon, MagnonReference};
use crate::floquet::{FloquetDrive, drive_has_ac_components, validate_floquet_drive};
use crate::{HubbardModel, RMatrixData, Result, TbError};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use std::collections::BTreeMap;

mod aggregate;
mod cluster;
mod merge;
mod operators;
mod periodic;
pub use merge::SpinLengthMapping;
pub use periodic::PeriodicSpinSite;

fn invalid(invariant: &'static str, message: impl Into<String>) -> TbError {
    TbError::InvalidMagnonInvariant {
        invariant,
        message: message.into(),
    }
}

/// Hopping expansion and numerical budgets, not inverse-frequency order.
#[derive(Clone, Debug)]
pub struct StrongCouplingOptions {
    /// Highest retained hopping order (2, 3 or 4).
    pub max_hopping_order: usize,
    /// Single-hop Fourier change $\lvert\Delta m\rvert\le H$.
    pub hop_harmonic_max: isize,
    /// Intermediate total photon label $\lvert m\rvert\le M$, independently
    /// of $H$.
    pub intermediate_photon_max: isize,
    /// Reject an actually coupled virtual charge state closer than this (eV).
    pub min_detuning_ev: f64,
    /// Upper bound on individual
    /// $\lvert t^{(m)}\_{ij}/(U+mW)\rvert$, not an accuracy certificate.
    pub max_hopping_ratio: f64,
    /// Budget for directed bond/photon contributions and higher-order cluster work.
    pub max_work: usize,
    /// Budget for the shared Peierls coefficient cache, in bytes.
    pub max_cache_bytes: usize,
    /// Budget for higher-order cluster numeric payload, excluding the cache and BLAS scratch.
    pub max_cluster_bytes: usize,
}

impl Default for StrongCouplingOptions {
    fn default() -> Self {
        Self {
            max_hopping_order: 2,
            hop_harmonic_max: 8,
            intermediate_photon_max: 8,
            min_detuning_ev: 1e-8,
            max_hopping_ratio: 0.25,
            max_work: 1 << 24,
            max_cache_bytes: 256 << 20,
            max_cluster_bytes: 256 << 20,
        }
    }
}

impl StrongCouplingOptions {
    pub fn validate(&self) -> Result<()> {
        if !(2..=4).contains(&self.max_hopping_order) {
            return Err(invalid(
                "supported_hopping_order",
                "supported hopping orders are 2, 3 and 4",
            ));
        }
        if self.hop_harmonic_max < 0 || self.intermediate_photon_max < 0 {
            return Err(invalid(
                "nonnegative_photon_cutoffs",
                "H and M must be nonnegative",
            ));
        }
        if !self.min_detuning_ev.is_finite() || self.min_detuning_ev <= 0.0 {
            return Err(invalid(
                "positive_detuning_threshold",
                "min_detuning_ev must be finite and positive",
            ));
        }
        if !self.max_hopping_ratio.is_finite()
            || !(0.0..1.0).contains(&self.max_hopping_ratio)
            || self.max_hopping_ratio == 0.0
        {
            return Err(invalid(
                "hopping_ratio_bound",
                "max_hopping_ratio must lie strictly between zero and one",
            ));
        }
        if self.max_work == 0 || self.max_cache_bytes == 0 || self.max_cluster_bytes == 0 {
            return Err(invalid(
                "positive_expansion_budgets",
                "work and cache budgets must be positive",
            ));
        }
        Ok(())
    }
}

/// One undirected quantum interaction
/// $K\\,\mathbf S\_i(\mathbf 0)\cdot\mathbf S\_j(\mathbf R)$, counted once per cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinPair<const DIM: usize> {
    pub i: usize,
    pub j: usize,
    pub cell: [isize; DIM],
    /// Positive K is antiferromagnetic; units are eV.
    pub exchange_ev: f64,
}

/// A scalar chirality $\mathbf S\_i\cdot(\mathbf S\_j\times\mathbf S\_k)$ with the given site order.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinChirality<const DIM: usize> {
    pub sites: [PeriodicSpinSite<DIM>; 3],
    pub coefficient_ev: f64,
}

/// A product $(\mathbf S\_i\cdot\mathbf S\_j)(\mathbf S\_k\cdot\mathbf S\_l)$ on four distinct sites.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinQuartic<const DIM: usize> {
    pub pairs: [[PeriodicSpinSite<DIM>; 2]; 2],
    pub coefficient_ev: f64,
}

/// Observed diagnostics. No cutoff-convergence error is claimed without a sweep.
#[derive(Clone, Debug)]
pub struct StrongCouplingDiagnostics {
    pub hopping_order: usize,
    pub hop_harmonic_max: isize,
    pub intermediate_photon_max: isize,
    /// Effective intermediate photon window; static higher-order expansions use zero.
    pub active_photon_max: isize,
    /// Number of terms examined in the direct second-order pair sum.
    pub virtual_terms_visited: usize,
    pub nonzero_virtual_terms: usize,
    /// Minimum active one-hop detuning; higher-order poles are checked separately.
    pub nearest_detuning_ev: Option<f64>,
    pub largest_hopping_ratio: f64,
    /// Number of nontrivial periodic induced clusters expanded through the requested order.
    pub expanded_clusters: usize,
}

/// Quantum spin-half coefficients; not a second band/geometry solver hierarchy.
///
/// The constant is relative to the bare singly occupied manifold. At second
/// order each bond has constant $-K/4$. Higher orders may also contribute
/// chiral and four-spin operators.
#[derive(Clone, Debug)]
pub struct EffectiveSpinModel<const DIM: usize> {
    lat: Array2<f64>,
    orb: Array2<f64>,
    pairs: Vec<SpinPair<DIM>>,
    chiralities: Vec<SpinChirality<DIM>>,
    quartics: Vec<SpinQuartic<DIM>>,
    constant_ev: f64,
    diagnostics: StrongCouplingDiagnostics,
}

impl<const DIM: usize> EffectiveSpinModel<DIM> {
    pub fn lattice(&self) -> &Array2<f64> {
        &self.lat
    }
    pub fn positions(&self) -> &Array2<f64> {
        &self.orb
    }
    pub fn pairs(&self) -> &[SpinPair<DIM>] {
        &self.pairs
    }
    pub fn chiralities(&self) -> &[SpinChirality<DIM>] {
        &self.chiralities
    }
    pub fn quartics(&self) -> &[SpinQuartic<DIM>] {
        &self.quartics
    }
    pub fn constant_ev(&self) -> f64 {
        self.constant_ev
    }
    pub fn diagnostics(&self) -> &StrongCouplingDiagnostics {
        &self.diagnostics
    }

    /// Convert the reduced spin polynomial to a collinear quadratic `Magnon`.
    ///
    /// Spins remain $S=1/2$; no single-ion gap or material-spin promotion is
    /// added. Axes must be collinear and periodic in the supplied cell
    /// (reference $\mathbf q=\mathbf 0$).
    /// Build a magnetic electronic supercell first when needed. The ordinary
    /// magnon solver, not this conversion, tests stability at each momentum.
    pub fn magnon(
        &self,
        axes: &Array2<f64>,
        reference: MagnonReference<DIM>,
    ) -> Result<Magnon<DIM>> {
        self.magnon_with_spins(axes, Array1::from_elem(self.orb.nrows(), 0.5), reference)
    }

    fn magnon_with_spins(
        &self,
        axes: &Array2<f64>,
        spins: Array1<f64>,
        reference: MagnonReference<DIM>,
    ) -> Result<Magnon<DIM>> {
        let n = self.orb.nrows();
        if spins.len() != n || spins.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(invalid(
                "positive_spin",
                "expected one finite positive spin length per site",
            ));
        }
        let normalized = collinear_axes(axes, n, &reference)?;
        let size = n
            .checked_mul(3)
            .ok_or_else(|| invalid("spin_matrix_size", "spin matrix size overflow"))?;
        let mut blocks = BTreeMap::new();
        blocks.insert([0; DIM], Array2::<f64>::zeros((size, size)));
        // Each undirected operator contributes two transposed landscape blocks.
        let mut add_pair =
            |[i, j]: [PeriodicSpinSite<DIM>; 2], scalar: f64, axial: [f64; 3]| -> Result<()> {
                let mut cell = [0; DIM];
                for (a, value) in cell.iter_mut().enumerate() {
                    *value = j.cell[a].checked_sub(i.cell[a]).ok_or_else(|| {
                        invalid(
                            "translation_overflow",
                            "magnon pair relative cell overflows",
                        )
                    })?;
                }
                let [x, y, z] = axial;
                let tensor = [[scalar, z, -y], [-z, scalar, x], [y, -x, scalar]];
                for (r, transpose) in [(cell, false), (negate(cell)?, true)] {
                    let block = blocks
                        .entry(r)
                        .or_insert_with(|| Array2::zeros((size, size)));
                    for a in 0..3 {
                        for b in 0..3 {
                            let (row, col) = (3 * i.orbital + a, 3 * j.orbital + b);
                            block[if transpose { [col, row] } else { [row, col] }] += tensor[a][b];
                        }
                    }
                }
                Ok(())
            };
        // TB2J uses ordered unit-vector bonds: each direction gets -S_i S_j/2.
        // Spectator lengths multiply this factor. Reject unusable spin products
        // rather than silently losing terms or creating a non-finite Hessian.
        let landscape_coefficient = |sites: &[usize], coefficient: f64| -> Result<f64> {
            if coefficient == 0.0 {
                return Ok(0.0);
            }
            let factor = sites.iter().try_fold(-0.5_f64, |factor, &i| {
                let factor = factor * spins[i];
                if !factor.is_normal() {
                    return Err(invalid(
                        "spin_lifting_scale",
                        "material spin length product is outside normal floating-point range",
                    ));
                }
                Ok(factor)
            })?;
            let value = factor * coefficient;
            if value == 0.0 || !value.is_finite() {
                return Err(invalid(
                    "spin_lifting_scale",
                    "nonzero operator coefficient is not representable at the requested spin lengths",
                ));
            }
            Ok(value)
        };
        for pair in &self.pairs {
            add_pair(
                [
                    PeriodicSpinSite {
                        orbital: pair.i,
                        cell: [0; DIM],
                    },
                    PeriodicSpinSite {
                        orbital: pair.j,
                        cell: pair.cell,
                    },
                ],
                landscape_coefficient(&[pair.i, pair.j], pair.exchange_ev)?,
                [0.0; 3],
            )?;
        }
        for term in &self.chiralities {
            let [i, j, k] = term.sites;
            let scale =
                landscape_coefficient(&[i.orbital, j.orbital, k.orbital], term.coefficient_ev)?;
            for (pair, spectator) in [([i, j], k), ([j, k], i), ([k, i], j)] {
                let axis = normalized.row(spectator.orbital);
                let axial = std::array::from_fn(|a| scale * axis[a]);
                add_pair(pair, 0.0, axial)?;
            }
        }
        for term in &self.quartics {
            let [p, q] = term.pairs;
            let sites = [p[0].orbital, p[1].orbital, q[0].orbital, q[1].orbital];
            let scale = landscape_coefficient(&sites, term.coefficient_ev)?;
            for (pair, [k, l]) in [(p, q), (q, p)] {
                let alignment = normalized.row(k.orbital).dot(&normalized.row(l.orbital));
                add_pair(pair, scale * alignment, [0.0; 3])?;
            }
        }
        let mut keys: Vec<_> = blocks.keys().copied().collect();
        keys.sort_by_key(|r| (*r != [0; DIM], *r));
        let mut ham = Array3::zeros((keys.len(), size, size));
        let mut ham_r = Array2::zeros((keys.len(), DIM));
        for (index, key) in keys.iter().enumerate() {
            ham.index_axis_mut(ndarray::Axis(0), index)
                .assign(&blocks[key]);
            for a in 0..DIM {
                ham_r[[index, a]] = key[a];
            }
        }
        Magnon::new(
            self.lat.clone(),
            self.orb.clone(),
            spins,
            normalized,
            ham,
            ham_r,
            reference,
        )
    }
}

fn collinear_axes<const D: usize>(
    axes: &Array2<f64>,
    n: usize,
    reference: &MagnonReference<D>,
) -> Result<Array2<f64>> {
    if n == 0 || axes.dim() != (n, 3) || axes.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "collinear_spin_axes",
            "expected one finite three-component axis per site",
        ));
    }
    reference.validate()?;
    if reference.q.iter().any(|q| *q != 0.0) {
        return Err(invalid(
            "periodic_collinear_reference",
            "use a magnetic cell with reference q=0",
        ));
    }
    let mut normalized = axes.clone();
    for mut row in normalized.rows_mut() {
        let norm2 = row.iter().map(|x| x * x).sum::<f64>();
        if (norm2 - 1.0).abs() > 1e-6 {
            return Err(invalid(
                "unit_reference_axis",
                "spin axes must be unit vectors",
            ));
        }
        row /= norm2.sqrt();
    }
    let direction = normalized.row(0).to_owned();
    for mut row in normalized.rows_mut() {
        let cross = [
            row[1] * direction[2] - row[2] * direction[1],
            row[2] * direction[0] - row[0] * direction[2],
            row[0] * direction[1] - row[1] * direction[0],
        ];
        if cross.iter().map(|x| x * x).sum::<f64>() > 1e-24 {
            return Err(invalid(
                "collinear_spin_axes",
                "noncollinear LSWT is outside this interface",
            ));
        }
        let sign = if row.dot(&direction) < 0.0 { -1.0 } else { 1.0 };
        // Snap tolerated roundoff tilts to an exactly collinear stationary state.
        row.zip_mut_with(&direction, |v, axis| *v = sign * axis);
    }
    Ok(normalized)
}

fn negate<const DIM: usize>(r: [isize; DIM]) -> Result<[isize; DIM]> {
    let mut result = [0; DIM];
    for (out, value) in result.iter_mut().zip(r) {
        *out = value
            .checked_neg()
            .ok_or_else(|| invalid("translation_overflow", "cannot negate lattice translation"))?;
    }
    Ok(result)
}

struct Bond<const DIM: usize> {
    i: usize,
    j: usize,
    cell: [isize; DIM],
    block: usize,
}

impl<const DIM: usize, R: RMatrixData> HubbardModel<DIM, R> {
    /// Generate half-filled single-orbital quantum spin coefficients in real space.
    ///
    /// This defines $P$ (one electron/site, photon zero), not an HF/filling
    /// check. Uniform $U>0$, a common real onsite, strict spin independence and
    /// Hermitian hopping are required. Provided Atom metadata must not describe multiple
    /// orbitals per atom. Site identity is always the orbital row: empty atoms
    /// add no spins, and unassigned orbitals are independent sites just as in
    /// orbital-only models. Metadata checks cannot certify a material downfolding.
    /// No SOC/Hund or phenomenological $S>1/2$ lifting.
    ///
    /// Orders 3 and 4 use periodic induced clusters and Möbius subtraction.
    /// For time-dependent fourth order, the intermediate window must be at least
    /// twice the single-hop harmonic cutoff to preserve disconnected cancellation.
    /// Requested cutoffs are not a convergence certificate: sweep the harmonic
    /// cutoff and allow the intermediate window to grow with it.
    pub fn floquet_spin_model(
        &self,
        drive: &FloquetDrive,
        options: &StrongCouplingOptions,
    ) -> Result<EffectiveSpinModel<DIM>> {
        options.validate()?;
        let model = self.bare_model();
        model.validate()?;
        validate_floquet_drive::<DIM>(drive)?;
        let n = model.norb();
        let u = self.onsite_u()[0];
        if !u.is_finite() || u <= 0.0 || self.onsite_u().iter().any(|v| *v != u) {
            return Err(invalid(
                "uniform_repulsive_u",
                "strong coupling requires identical positive U on every site",
            ));
        }
        if model.atoms.iter().any(|atom| atom.norb() > 1) {
            return Err(invalid(
                "single_orbital_sites",
                "multi-orbital Atom metadata is outside the single-orbital reduction",
            ));
        }
        let mut rows = BTreeMap::new();
        for (block, row) in model.hamR.rows().into_iter().enumerate() {
            let mut key = [0; DIM];
            for a in 0..DIM {
                key[a] = row[a];
            }
            if rows.insert(key, block).is_some() {
                return Err(invalid(
                    "unique_translations",
                    "duplicate hopping translation",
                ));
            }
        }
        let onsite = rows
            .get(&[0; DIM])
            .map_or(Complex::new(0.0, 0.0), |r| model.ham[[*r, 0, 0]]);
        if onsite.im != 0.0 {
            return Err(invalid("scalar_onsite", "the common onsite must be real"));
        }
        let mut bonds = Vec::new();
        for (cell, block) in &rows {
            let reverse = negate(*cell)?;
            let reverse_block = rows.get(&reverse).copied();
            for i in 0..n {
                for j in 0..n {
                    let hopping = model.ham[[*block, i, j]];
                    if hopping != model.ham[[*block, i + n, j + n]]
                        || model.ham[[*block, i, j + n]] != Complex::new(0.0, 0.0)
                        || model.ham[[*block, i + n, j]] != Complex::new(0.0, 0.0)
                    {
                        return Err(invalid(
                            "spin_independent_hopping",
                            "require identical spin-major blocks and zero spin flips",
                        ));
                    }
                    let partner = reverse_block
                        .map_or(Complex::new(0.0, 0.0), |r| model.ham[[r, j, i]].conj());
                    if hopping != partner {
                        return Err(invalid(
                            "hermitian_hopping",
                            "t_ij(R) must equal conjugate t_ji(-R), without silent symmetrization",
                        ));
                    }
                    if *cell == [0; DIM] && i == j {
                        if hopping != onsite {
                            return Err(invalid(
                                "scalar_onsite",
                                "all sites must have the same real onsite",
                            ));
                        }
                        continue;
                    }
                    if hopping == Complex::new(0.0, 0.0) {
                        continue;
                    }
                    if (i, j, *cell) < (j, i, reverse) {
                        bonds.push(Bond {
                            i,
                            j,
                            cell: *cell,
                            block: *block,
                        });
                    }
                }
            }
        }
        bonds.sort_by_key(|bond| (bond.i, bond.j, bond.cell));
        let cut = options
            .hop_harmonic_max
            .min(options.intermediate_photon_max);
        let mut active = options.clone();
        if !drive_has_ac_components(drive) || bonds.is_empty() {
            active.hop_harmonic_max = 0;
            active.intermediate_photon_max = 0;
        } else if options.max_hopping_order > 2 {
            if options.max_hopping_order == 4 {
                let complete = options.hop_harmonic_max.checked_mul(2).ok_or_else(|| {
                    invalid(
                        "complete_photon_window",
                        "fourth-order photon window overflows",
                    )
                })?;
                if options.intermediate_photon_max < complete {
                    return Err(invalid(
                        "complete_photon_window",
                        format!("driven fourth order needs a photon window M >= 2H = {complete}"),
                    ));
                }
                active.intermediate_photon_max = complete;
            } else {
                // Two interior photon labels can differ by as much as 2M.
                // A third-order path may use (+1,-2,+1) even when M=1.
                active.hop_harmonic_max = options.hop_harmonic_max.min(cut.saturating_mul(2));
                active.intermediate_photon_max = cut;
            }
        } else {
            active.hop_harmonic_max = cut;
            active.intermediate_photon_max = cut;
        }
        let cut = active.hop_harmonic_max.min(active.intermediate_photon_max);
        let mut diagnostics = StrongCouplingDiagnostics {
            hopping_order: options.max_hopping_order,
            hop_harmonic_max: options.hop_harmonic_max,
            intermediate_photon_max: options.intermediate_photon_max,
            active_photon_max: active.intermediate_photon_max,
            virtual_terms_visited: 0,
            nonzero_virtual_terms: 0,
            nearest_detuning_ev: None,
            largest_hopping_ratio: 0.0,
            expanded_clusters: 0,
        };
        let mut result = EffectiveSpinModel {
            lat: model.lat.clone(),
            orb: model.orb.clone(),
            pairs: Vec::new(),
            chiralities: Vec::new(),
            quartics: Vec::new(),
            constant_ev: 0.0,
            diagnostics: diagnostics.clone(),
        };
        if bonds.is_empty() {
            return Ok(result);
        }
        let count = cut
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| invalid("photon_window_size", "photon window exceeds index range"))?;
        let work = bonds
            .len()
            .checked_mul(count)
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| invalid("expansion_work", "virtual contribution count overflows"))?;
        if work > options.max_work {
            return Err(invalid(
                "expansion_work",
                format!(
                    "need {work} directed virtual contributions, budget {}",
                    options.max_work
                ),
            ));
        }
        let cache = model.validated_harmonic_cache(
            drive,
            active.hop_harmonic_max,
            options.max_cache_bytes,
        )?;
        for bond in bonds {
            let mut exchange = 0.0;
            for photon in -cut..=cut {
                // From source sector 0 to target m in the positive-time basis.
                // Construct the reverse by conjugation, exactly as in the cluster kernel.
                let index = [bond.block, bond.i, bond.j];
                for coefficient in [
                    cache.harmonic_blocks(-photon)[index],
                    cache.harmonic_blocks(photon)[index].conj(),
                ] {
                    diagnostics.virtual_terms_visited += 1;
                    let magnitude = coefficient.norm();
                    if !magnitude.is_finite() {
                        return Err(invalid(
                            "finite_harmonic_hopping",
                            "harmonic hopping magnitude is not finite",
                        ));
                    }
                    if magnitude == 0.0 {
                        continue;
                    }
                    diagnostics.nonzero_virtual_terms += 1;
                    let denominator = drive.omega0_ev.mul_add(photon as f64, u);
                    let detuning = denominator.abs();
                    if !denominator.is_finite() || detuning < options.min_detuning_ev {
                        return Err(invalid(
                            "nonresonant_charge_state",
                            format!("coupled one-doublon state m={photon}, |U+mW|={detuning:e} eV"),
                        ));
                    }
                    let ratio = magnitude / denominator.abs();
                    if !ratio.is_finite() || ratio > options.max_hopping_ratio {
                        return Err(invalid(
                            "hopping_detuning_ratio",
                            format!(
                                "virtual hopping ratio {ratio:e} exceeds {}",
                                options.max_hopping_ratio
                            ),
                        ));
                    }
                    diagnostics.largest_hopping_ratio =
                        diagnostics.largest_hopping_ratio.max(ratio);
                    diagnostics.nearest_detuning_ev = Some(
                        diagnostics
                            .nearest_detuning_ev
                            .map_or(detuning, |d| d.min(detuning)),
                    );
                    // Both doublon directions, NOT a blind four-times-forward formula.
                    // Divide by sqrt(|denominator|) before squaring: forming
                    // t*t first can overflow, while (t/U)^2*U can underflow.
                    let scaled = magnitude / denominator.abs().sqrt();
                    exchange += 2.0 * denominator.signum() * scaled * scaled;
                }
            }
            let exchange_ev = exchange;
            if !exchange_ev.is_finite() {
                return Err(invalid(
                    "finite_spin_exchange",
                    "second-order exchange overflowed",
                ));
            }
            result.constant_ev -= exchange_ev / 4.0;
            result.pairs.push(SpinPair {
                i: bond.i,
                j: bond.j,
                cell: bond.cell,
                exchange_ev,
            });
        }
        if options.max_hopping_order > 2 {
            let mut work_used = work;
            let correction = aggregate::aggregate_clusters(
                model,
                &rows,
                &cache,
                u,
                drive.omega0_ev,
                &active,
                options.max_cluster_bytes,
                &mut work_used,
            )?;
            diagnostics.expanded_clusters = correction.visited_clusters;
            result.constant_ev += correction.constant_ev;
            let mut pairs = BTreeMap::new();
            for pair in result.pairs.drain(..).chain(correction.pairs) {
                let sum = pairs.entry((pair.i, pair.j, pair.cell)).or_insert(0.0);
                *sum += pair.exchange_ev;
                if !sum.is_finite() {
                    return Err(invalid(
                        "finite_spin_exchange",
                        "periodic pair exchange overflowed",
                    ));
                }
            }
            result.pairs = pairs
                .into_iter()
                .map(|((i, j, cell), exchange_ev)| SpinPair {
                    i,
                    j,
                    cell,
                    exchange_ev,
                })
                .collect();
            result.chiralities = correction
                .chirality
                .into_iter()
                .map(|(sites, coefficient_ev)| SpinChirality {
                    sites,
                    coefficient_ev,
                })
                .collect();
            result.quartics = correction
                .quartic
                .into_iter()
                .map(|(pairs, coefficient_ev)| SpinQuartic {
                    pairs,
                    coefficient_ev,
                })
                .collect();
        }
        if !result.constant_ev.is_finite() {
            return Err(invalid(
                "finite_spin_constant",
                "per-cell constant overflowed",
            ));
        }
        result.diagnostics = diagnostics;
        Ok(result)
    }
}

#[cfg(test)]
#[path = "driven_honeycomb_tests.rs"]
mod honeycomb_tests;
#[cfg(test)]
#[path = "driven_merge_tests.rs"]
mod merge_tests;
#[cfg(test)]
#[path = "driven_periodic_tests.rs"]
mod periodic_tests;
#[cfg(test)]
#[path = "driven_square_tests.rs"]
mod square_tests;
#[cfg(test)]
#[path = "driven_supercell_tests.rs"]
mod supercell_tests;
#[cfg(test)]
#[path = "driven_tests.rs"]
mod tests;
