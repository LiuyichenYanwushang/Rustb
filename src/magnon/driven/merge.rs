//! Overlay only the microscopic driven-minus-static correction on a TB2J model.

use super::{
    EffectiveSpinModel, PeriodicSpinSite, SpinChirality, SpinPair, SpinQuartic,
    StrongCouplingOptions, invalid,
};
use crate::floquet::FloquetDrive;
use crate::magnon::Magnon;
use crate::{HubbardModel, RMatrixData, Result};
use ndarray::{Array2, Array3, Axis, s};
use std::collections::BTreeMap;

/// How to interpret the spin-half coefficients in the target magnon model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpinLengthMapping {
    /// Require spin length 1/2 on every mapped site.
    SpinHalf,
    /// Keep the reduced quantum operator coefficients and substitute the
    /// target's spin lengths in the classical-energy Hessian. This is a
    /// phenomenological lifting, not a microscopic higher-spin derivation.
    KeepOperatorCoefficients,
}

impl<const D: usize, R: RMatrixData> HubbardModel<D, R> {
    /// Add the microscopic driven-minus-zero-field magnon correction to `baseline`.
    ///
    /// Both expansions use the supplied hopping order, cutoffs and tolerances.
    /// Subtraction is done on the spin polynomial, before collinear reduction.
    /// The baseline must use the same lattice matrix and a cell-periodic collinear
    /// reference (`q=0`). `model_to_baseline[i]` is the target site of orbital `i`;
    /// this must be a bijection. Fractional positions must agree modulo integer
    /// cells (absolute tolerance `1e-9`, coordinates with magnitude below
    /// `2^53`), including for a zero-drive call. Normalized magnetic axes must
    /// have cross-product norm at most `1e-12` against the first axis.
    /// The output keeps the baseline spin lengths, axes and site order. Existing
    /// translations retain their order; any new translations are appended in
    /// lexicographic order. No baseline anisotropy or static exchange is replaced.
    ///
    /// This neither finds a new reference state nor establishes its stability.
    /// Do not also drive the returned model with a phenomenological bond drive for
    /// the same light field.
    pub fn floquet_magnon_delta(
        &self,
        drive: &FloquetDrive,
        options: &StrongCouplingOptions,
        baseline: &Magnon<D>,
        model_to_baseline: &[usize],
        spin_mapping: SpinLengthMapping,
    ) -> Result<Magnon<D>> {
        baseline.validate()?;
        if spin_mapping == SpinLengthMapping::SpinHalf && baseline.spin.iter().any(|&s| s != 0.5) {
            return Err(invalid(
                "spin_half_mapping",
                "target spins must be 1/2 unless fixed operator coefficients are explicitly requested",
            ));
        }
        // Ordinary BdG forms S_i*S_j before taking its square root. Require
        // usable self products as well, even for sites without a correction.
        if baseline.spin.iter().any(|&s| !(s * s).is_normal()) {
            return Err(invalid(
                "spin_lifting_scale",
                "target spin squares must lie in the normal floating-point range",
            ));
        }
        let bare = self.bare_model();
        bare.validate()?;
        if bare.lat != baseline.lat {
            return Err(invalid(
                "magnon_delta_lattice",
                "electronic and magnon lattice matrices must match",
            ));
        }
        let n = baseline.nspin();
        if bare.norb() != n || model_to_baseline.len() != n {
            return Err(invalid(
                "magnon_delta_site_map",
                "expected a bijection between electronic orbitals and magnetic sites",
            ));
        }
        let mut seen = vec![false; n];
        let mut shifts = vec![[0; D]; n];
        for (i, &j) in model_to_baseline.iter().enumerate() {
            if j >= n || std::mem::replace(&mut seen[j], true) {
                return Err(invalid(
                    "magnon_delta_site_map",
                    "expected a bijection without repeated or out-of-range sites",
                ));
            }
            for d in 0..D {
                let source = bare.orb[[i, d]];
                let target = baseline.orb[[j, d]];
                let fraction = source.fract() - target.fract();
                let carry = fraction.round();
                let limit = (1_u64 << 53) as f64;
                // Compare fractions before forming a potentially huge integer shift.
                if source.abs() >= limit || target.abs() >= limit || (fraction - carry).abs() > 1e-9
                {
                    return Err(invalid(
                        "magnon_delta_site_map",
                        format!(
                            "orbital {i} and target {j} must differ by integer fractional translations"
                        ),
                    ));
                }
                shifts[i][d] = source.trunc() as i64 - target.trunc() as i64 + carry as i64;
            }
        }
        let remap = |anchor: PeriodicSpinSite<D>,
                     site: PeriodicSpinSite<D>|
         -> Result<PeriodicSpinSite<D>> {
            let mut cell = [0; D];
            for (d, x) in cell.iter_mut().enumerate() {
                let relative = site.cell[d] as i128 - anchor.cell[d] as i128
                    + i128::from(shifts[site.orbital][d])
                    - i128::from(shifts[anchor.orbital][d]);
                *x = isize::try_from(relative).map_err(|_| {
                    invalid(
                        "translation_overflow",
                        "translated relative spin site overflows",
                    )
                })?;
            }
            Ok(PeriodicSpinSite {
                orbital: model_to_baseline[site.orbital],
                cell,
            })
        };
        let mut delta = self.floquet_spin_model(drive, options)?;
        // A valid zero-field input returns the baseline bit-for-bit, including
        // its translation order. Still validate the requested reference below.
        if drive
            .modes
            .iter()
            .all(|m| m.a_complex.iter().all(|z| z.re == 0.0 && z.im == 0.0))
        {
            super::collinear_axes(&baseline.axis, n, &baseline.reference)?;
            return Ok(baseline.clone());
        } else {
            let zero = self.floquet_spin_model(&FloquetDrive::new(drive.omega0_ev), options)?;
            subtract_and_remap(&mut delta, zero, remap)?;
        }
        delta.orb = baseline.orb.clone();
        let correction = delta.magnon_with_spins(
            &baseline.axis,
            baseline.spin.clone(),
            baseline.reference.clone(),
        )?;
        add_correction(baseline, &correction)
    }
}

fn subtract<K: Ord>(
    driven: impl Iterator<Item = (K, f64)>,
    zero: impl Iterator<Item = (K, f64)>,
) -> impl Iterator<Item = (K, f64)> {
    let mut coefficients: BTreeMap<_, _> = driven.collect();
    for (key, value) in zero {
        *coefficients.entry(key).or_default() -= value;
    }
    coefficients.into_iter().filter(|(_, value)| *value != 0.0)
}

fn subtract_and_remap<const D: usize>(
    driven: &mut EffectiveSpinModel<D>,
    zero: EffectiveSpinModel<D>,
    remap: impl Fn(PeriodicSpinSite<D>, PeriodicSpinSite<D>) -> Result<PeriodicSpinSite<D>>,
) -> Result<()> {
    let pairs = |p: SpinPair<D>| ((p.i, p.j, p.cell), p.exchange_ev);
    let chiral = |p: SpinChirality<D>| (p.sites, p.coefficient_ev);
    let quartic = |p: SpinQuartic<D>| (p.pairs, p.coefficient_ev);
    for ((i, j, cell), exchange_ev) in subtract(
        std::mem::take(&mut driven.pairs).into_iter().map(pairs),
        zero.pairs.into_iter().map(pairs),
    ) {
        let anchor = PeriodicSpinSite {
            orbital: i,
            cell: [0; D],
        };
        let i = remap(anchor, anchor)?;
        let j = remap(anchor, PeriodicSpinSite { orbital: j, cell })?;
        driven.pairs.push(SpinPair {
            i: i.orbital,
            j: j.orbital,
            cell: j.cell,
            exchange_ev,
        });
    }
    for ([i, j, k], coefficient_ev) in subtract(
        std::mem::take(&mut driven.chiralities)
            .into_iter()
            .map(chiral),
        zero.chiralities.into_iter().map(chiral),
    ) {
        // Do not sort the remapped sites: their order fixes the chirality sign.
        driven.chiralities.push(SpinChirality {
            sites: [remap(i, i)?, remap(i, j)?, remap(i, k)?],
            coefficient_ev,
        });
    }
    for ([[i, j], [k, l]], coefficient_ev) in subtract(
        std::mem::take(&mut driven.quartics)
            .into_iter()
            .map(quartic),
        zero.quartics.into_iter().map(quartic),
    ) {
        driven.quartics.push(SpinQuartic {
            // In the cell-periodic reference each spectator depends only on
            // its orbital; only the translation within each pair enters HP.
            pairs: [[remap(i, i)?, remap(i, j)?], [remap(k, k)?, remap(k, l)?]],
            coefficient_ev,
        });
    }
    Ok(())
}

fn add_correction<const D: usize>(
    baseline: &Magnon<D>,
    correction: &Magnon<D>,
) -> Result<Magnon<D>> {
    let active = correction
        .ham
        .outer_iter()
        .enumerate()
        .filter(|(_, block)| block.iter().any(|&x| x != 0.0))
        .map(|(row, _)| row)
        .collect::<Vec<_>>();
    let mut result = baseline.clone();
    if active.is_empty() {
        return Ok(result);
    }
    let key = |rows: &Array2<isize>, row| std::array::from_fn::<_, D, _>(|d| rows[[row, d]]);
    let mut indices: BTreeMap<_, _> = (0..baseline.nR())
        .map(|row| (key(&baseline.hamR, row), row))
        .collect();
    for &row in &active {
        let nr = indices.len();
        indices.entry(key(&correction.hamR, row)).or_insert(nr);
    }
    let n_rows = indices.len();
    if n_rows != baseline.nR() {
        let size = baseline.nsta();
        if n_rows
            .checked_mul(size)
            .and_then(|n| n.checked_mul(size))
            .is_none_or(|n| n > isize::MAX as usize / size_of::<f64>())
        {
            return Err(invalid(
                "spin_matrix_size",
                "merged exchange matrix size overflows",
            ));
        }
        result.ham = Array3::zeros((n_rows, size, size));
        result
            .ham
            .slice_mut(s![..baseline.nR(), .., ..])
            .assign(&baseline.ham);
        result.hamR = Array2::zeros((n_rows, D));
        for (r, &row) in &indices {
            for d in 0..D {
                result.hamR[[row, d]] = r[d];
            }
        }
    }
    for row in active {
        let target = indices[&key(&correction.hamR, row)];
        result.ham.index_axis_mut(Axis(0), target).zip_mut_with(
            &correction.ham.index_axis(Axis(0), row),
            |dst, &x| {
                if x != 0.0 {
                    *dst += x;
                }
            },
        );
    }
    result.validate()?;
    Ok(result)
}
