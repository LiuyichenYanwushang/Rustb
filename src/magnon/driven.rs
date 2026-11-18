//! Real-space, half-filled single-orbital Floquet–Hubbard spin reduction.
//!
//! The initial implementation retains hopping order two only. It does not use
//! an electronic van Vleck Hamiltonian or infer hopping/U from a static magnon.
//! Every input orbital is one spin-half site. A common scalar onsite energy is
//! removed; reported constants are relative to the singly occupied manifold.

use super::{Magnon, MagnonReference};
use crate::floquet::{FloquetDrive, validate_floquet_drive};
use crate::{HubbardModel, RMatrixData, Result, TbError};
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use std::collections::BTreeMap;

fn invalid(invariant: &'static str, message: impl Into<String>) -> TbError {
    TbError::InvalidMagnonInvariant {
        invariant,
        message: message.into(),
    }
}

/// Hopping expansion and numerical budgets, not inverse-frequency order.
#[derive(Clone, Debug)]
pub struct StrongCouplingOptions {
    /// Highest hopping order. Currently only 2 is implemented; others fail.
    pub max_hopping_order: usize,
    /// Single-hop Fourier change |delta m| <= H.
    pub hop_harmonic_max: isize,
    /// Intermediate total photon label |m| <= M, independently of H.
    pub intermediate_photon_max: isize,
    /// Reject an actually coupled virtual charge state closer than this (eV).
    pub min_detuning_ev: f64,
    /// Upper bound on individual |t_ij^(m)/(U+mW)|, not an accuracy certificate.
    pub max_hopping_ratio: f64,
    /// Budget for directed bond/photon contributions (backend has its own caps).
    pub max_work: usize,
    /// Budget for the shared Peierls coefficient cache, in bytes.
    pub max_cache_bytes: usize,
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
        }
    }
}

impl StrongCouplingOptions {
    pub fn validate(&self) -> Result<()> {
        if self.max_hopping_order != 2 {
            return Err(invalid(
                "supported_hopping_order",
                "only hopping order two is implemented",
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
        if self.max_work == 0 || self.max_cache_bytes == 0 {
            return Err(invalid(
                "positive_expansion_budgets",
                "work and cache budgets must be positive",
            ));
        }
        Ok(())
    }
}

/// One undirected quantum interaction K S_i(0).S_j(R), counted once per cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SpinPair<const DIM: usize> {
    pub i: usize,
    pub j: usize,
    pub cell: [isize; DIM],
    /// Positive K is antiferromagnetic; units are eV.
    pub exchange_ev: f64,
}

/// Observed diagnostics. No cutoff-convergence error is claimed without a sweep.
#[derive(Clone, Debug)]
pub struct StrongCouplingDiagnostics {
    pub hopping_order: usize,
    pub hop_harmonic_max: isize,
    pub intermediate_photon_max: isize,
    /// At second order both cutoffs constrain the same single virtual label.
    pub active_photon_max: isize,
    pub virtual_terms_visited: usize,
    pub nonzero_virtual_terms: usize,
    pub nearest_detuning_ev: Option<f64>,
    pub largest_hopping_ratio: f64,
}

/// Quantum spin-half coefficients; not a second band/geometry solver hierarchy.
///
/// The constant includes -K/4 for each pair, so the isolated triplet has zero
/// shift. Use `magnon` for a periodic collinear LSWT reference in this same cell.
#[derive(Clone, Debug)]
pub struct EffectiveSpinModel<const DIM: usize> {
    lat: Array2<f64>,
    orb: Array2<f64>,
    pairs: Vec<SpinPair<DIM>>,
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
    pub fn constant_ev(&self) -> f64 {
        self.constant_ev
    }
    pub fn diagnostics(&self) -> &StrongCouplingDiagnostics {
        &self.diagnostics
    }

    /// Convert quantum pairs to ordinary unit-vector-landscape `Magnon`.
    ///
    /// Spins remain S=1/2; no single-ion gap or material-spin promotion is added.
    /// Axes must be collinear and periodic in the supplied cell (reference q=0).
    /// Build a magnetic electronic supercell first when needed. The ordinary
    /// magnon solver, not this conversion, tests stability at each momentum.
    pub fn magnon(
        &self,
        axes: &Array2<f64>,
        reference: MagnonReference<DIM>,
    ) -> Result<Magnon<DIM>> {
        let n = self.orb.nrows();
        if axes.dim() != (n, 3) || axes.iter().any(|v| !v.is_finite()) {
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
        let direction = normalized.row(0);
        for row in normalized.rows() {
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
        }
        let size = n
            .checked_mul(3)
            .ok_or_else(|| invalid("spin_matrix_size", "spin matrix size overflow"))?;
        let mut blocks = BTreeMap::new();
        blocks.insert([0; DIM], Array2::<f64>::zeros((size, size)));
        for pair in &self.pairs {
            let reverse = negate(pair.cell)?;
            // Ordered landscape pairs: E=-sum e_i J_ij e_j, S_i=S_j=1/2.
            let value = -pair.exchange_ev / 8.0;
            for (cell, i, j) in [(pair.cell, pair.i, pair.j), (reverse, pair.j, pair.i)] {
                let block = blocks
                    .entry(cell)
                    .or_insert_with(|| Array2::zeros((size, size)));
                for component in 0..3 {
                    block[[3 * i + component, 3 * j + component]] += value;
                }
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
            Array1::from_elem(n, 0.5),
            normalized,
            ham,
            ham_r,
            reference,
        )
    }
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
    reverse_block: usize,
}

impl<const DIM: usize, R: RMatrixData> HubbardModel<DIM, R> {
    /// Generate half-filled single-orbital quantum spin coefficients in real space.
    ///
    /// This defines P (one electron/site, photon zero), not an HF/filling check.
    /// Uniform U>0, a common real onsite, strict spin independence and Hermitian
    /// hopping are required. Provided Atom metadata must not describe multiple
    /// orbitals per atom. Site identity is always the orbital row: empty atoms
    /// add no spins, and unassigned orbitals are independent sites just as in
    /// orbital-only models. Metadata checks cannot certify a material downfolding.
    /// No SOC/Hund or phenomenological S>1/2 lifting.
    ///
    /// Only hopping order 2 is currently supported. The Peierls cache, coherent
    /// modes, Bessel backend and per-link DFT fallback are the existing backend.
    /// Requested cutoffs are not a convergence certificate: sweep H and M.
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
                            reverse_block: reverse_block.ok_or_else(|| {
                                invalid("hermitian_hopping", "missing reverse hopping block")
                            })?,
                        });
                    }
                }
            }
        }
        bonds.sort_by_key(|bond| (bond.i, bond.j, bond.cell));
        let cut = options
            .hop_harmonic_max
            .min(options.intermediate_photon_max);
        let mut diagnostics = StrongCouplingDiagnostics {
            hopping_order: 2,
            hop_harmonic_max: options.hop_harmonic_max,
            intermediate_photon_max: options.intermediate_photon_max,
            active_photon_max: cut,
            virtual_terms_visited: 0,
            nonzero_virtual_terms: 0,
            nearest_detuning_ev: None,
            largest_hopping_ratio: 0.0,
        };
        let mut result = EffectiveSpinModel {
            lat: model.lat.clone(),
            orb: model.orb.clone(),
            pairs: Vec::new(),
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
        let cache = model.validated_harmonic_cache(drive, cut, options.max_cache_bytes)?;
        for bond in bonds {
            let mut exchange = 0.0;
            for photon in -cut..=cut {
                let block = cache.harmonic_blocks(photon);
                for coefficient in [
                    block[[bond.block, bond.i, bond.j]],
                    block[[bond.reverse_block, bond.j, bond.i]],
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
#[path = "driven_tests.rs"]
mod tests;
