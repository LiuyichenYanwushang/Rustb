#![doc = include_str!("driven_theory.md")]

use super::{Magnon, SPIN_COMPONENTS};
use crate::error::{Result, TbError};
use crate::floquet::{
    FloquetDrive, LightMode, bessel_peierls_coeffs, link_displacement_cartesian,
    validate_floquet_drive,
};
use num_complex::Complex;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};

impl<const DIM: usize> Magnon<DIM> {
    /// Dress each intersite isotropic exchange as second-order Floquet superexchange.
    ///
    /// Supply finite positive `u_ev` and a symmetric photon cutoff
    /// `0 <= harmonic_max <= 32768`. The input model must already contain the
    /// static exchanges. This method treats their entire intersite isotropic
    /// part as superexchange; it cannot separate other isotropic mechanisms.
    /// Onsite tensors, DMI, and traceless anisotropy are not dressed. No electronic
    /// hopping model is required. Independently evaluates each reciprocal bond
    /// pair in parallel and returns an ordinary `Magnon` for the same reference.
    /// Active bond tensors are averaged with their reciprocal transposes before dressing.
    ///
    /// Rejects coupled poles in the retained window to floating-point accuracy.
    /// The caller must choose a physically off-resonant drive and converge the
    /// harmonic cutoff; neither perturbative validity nor reference stability
    /// can be inferred from this input alone. See the [module theory](self).
    pub fn floquet_superexchange(
        &self,
        drive: &FloquetDrive,
        u_ev: f64,
        harmonic_max: isize,
    ) -> Result<Self> {
        self.validate()?;
        validate_floquet_drive::<DIM>(drive)?;
        if !u_ev.is_finite() || u_ev <= 0.0 {
            return Err(invalid(
                "superexchange_u",
                "U must be finite and positive".into(),
            ));
        }
        if !(0..=32768).contains(&harmonic_max) {
            return Err(invalid(
                "superexchange_harmonics",
                "harmonic_max must be in 0..=32768".into(),
            ));
        }
        // DC Peierls phases cancel on an out-and-back path. A zero cutoff
        // retains no photon-assisted correction.
        if harmonic_max == 0 || !drive.modes.iter().any(|mode| mode.harmonic != 0) {
            return Ok(self.clone());
        }
        let ac = FloquetDrive::with_modes(
            drive.omega0_ev,
            drive
                .modes
                .iter()
                .filter(|mode| mode.harmonic != 0)
                .cloned()
                .collect(),
        );
        let n = self.nspin();
        let translations: HashMap<Vec<isize>, usize> = self
            .hamR
            .rows()
            .into_iter()
            .enumerate()
            .map(|(idx, row)| (row.to_vec(), idx))
            .collect();
        let mut bonds = Vec::new();
        for (r, row) in self.hamR.rows().into_iter().enumerate() {
            // validate() proved that the negation exists and is representable.
            let reverse = translations[&row.iter().map(|&x| -x).collect::<Vec<_>>()];
            if r > reverse {
                continue;
            }
            for i in 0..n {
                for j in 0..n {
                    if r == reverse && i >= j {
                        continue;
                    }
                    let (i, j) = (SPIN_COMPONENTS * i, SPIN_COMPONENTS * j);
                    let forward = self.isotropic_exchange(r, i, j);
                    let backward = self.isotropic_exchange(reverse, j, i);
                    let isotropic = f64::midpoint(forward, backward);
                    if isotropic != 0.0 {
                        bonds.push((r, reverse, i, j, isotropic));
                    }
                }
            }
        }
        let energy_scale = u_ev.max(drive.omega0_ev);
        let u = u_ev / energy_scale;
        let w = drive.omega0_ev / energy_scale;
        let corrections: Vec<_> = bonds
            .par_iter()
            .map(|&(r, reverse, i, j, isotropic)| {
                let d = link_displacement_cartesian(
                    &self.lat,
                    self.orb.row(i / SPIN_COMPONENTS),
                    self.orb.row(j / SPIN_COMPONENTS),
                    self.hamR.row(r),
                );
                if d.iter().any(|d| !d.is_finite()) {
                    return Err(invalid(
                        "superexchange_link",
                        format!(
                            "non-finite displacement on bond ({r}, {}, {})",
                            i / SPIN_COMPONENTS,
                            j / SPIN_COMPONENTS
                        ),
                    ));
                }
                // Fold equal and opposite temporal harmonics before the Bessel
                // expansion: cancelled carriers must not leak into a pole.
                let link_drive = projected_drive(&ac, &d)?;
                if link_drive.modes.is_empty() {
                    return Ok(None);
                }
                let coefficients = bessel_peierls_coeffs(
                    &ndarray::arr1(&[1.0]),
                    &link_drive,
                    -harmonic_max,
                    harmonic_max,
                    0,
                )?;
                let mut delta = 0.0;
                for m in 1..=harmonic_max {
                    let minus = coefficients[(harmonic_max - m) as usize];
                    let plus = coefficients[(harmonic_max + m) as usize];
                    if [minus, plus].iter().all(|z| z.re == 0.0 && z.im == 0.0) {
                        continue;
                    }
                    let photon = m as f64;
                    let x = photon * w;
                    let detuning = (-w).mul_add(photon, u);
                    if detuning.abs() <= 64.0 * f64::EPSILON * u.max(x) {
                        return Err(invalid(
                            "nonresonant_charge_state",
                            format!(
                                "coupled superexchange pole at m={m} on bond ({r}, {}, {})",
                                i / SPIN_COMPONENTS,
                                j / SPIN_COMPONENTS
                            ),
                        ));
                    }
                    // Parseval supplies the static part exactly, including the
                    // discarded terms. Sum only deviations from 1 to avoid loss
                    // of the weak correction through cancellation with the baseline.
                    delta +=
                        ((minus.norm_sqr() + plus.norm_sqr()) * (x / (u + x))) * (x / detuning);
                }
                if delta == 0.0 {
                    return Ok(None);
                }
                // One physical tensor per bond, even for inputs whose
                // reciprocal partners differ within validation tolerance.
                let block = std::array::from_fn::<_, SPIN_COMPONENTS, _>(|a| {
                    std::array::from_fn::<_, SPIN_COMPONENTS, _>(|b| {
                        let mean = f64::midpoint(
                            self.ham[[r, i + a, j + b]],
                            self.ham[[reverse, j + b, i + a]],
                        );
                        if a == b {
                            delta.mul_add(isotropic, mean)
                        } else {
                            mean
                        }
                    })
                });
                Ok(Some(block))
            })
            .collect();
        let mut out = self.clone();
        for ((r, reverse, i, j, _), correction) in bonds.into_iter().zip(corrections) {
            if let Some(block) = correction? {
                for (a, row) in block.into_iter().enumerate() {
                    for (b, value) in row.into_iter().enumerate() {
                        out.ham[[r, i + a, j + b]] = value;
                        out.ham[[reverse, j + b, i + a]] = value;
                    }
                }
            }
        }
        out.validate()?;
        Ok(out)
    }

    fn isotropic_exchange(&self, r: usize, i: usize, j: usize) -> f64 {
        let [x, y, z] = std::array::from_fn(|a| self.ham[[r, i + a, j + a]]);
        if x == y && y == z {
            return x;
        }
        let trace_sum = |scale| {
            let mut sum = (0.0, 0.0);
            for value in [x, y, z] {
                compensated_add(&mut sum, value * scale);
            }
            sum.0 + sum.1
        };
        let trace = trace_sum(1.0);
        if trace.is_finite() {
            trace / 3.0
        } else {
            trace_sum(1.0 / 3.0)
        }
    }
}

fn compensated_add((sum, correction): &mut (f64, f64), value: f64) {
    let next = *sum + value;
    *correction += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

// Build only the oscillating scalar signal seen by this bond. Compensate
// before dropping a zero carrier so a large cancelling pair cannot hide a
// smaller real amplitude and its associated resonance.
fn projected_drive(drive: &FloquetDrive, d: &ndarray::Array1<f64>) -> Result<FloquetDrive> {
    let mut carriers = BTreeMap::<usize, [(f64, f64); 2]>::new();
    for mode in &drive.modes {
        let [real, imag] = carriers.entry(mode.harmonic.unsigned_abs()).or_default();
        let imaginary_sign = if mode.harmonic < 0 { -1.0 } else { 1.0 };
        for (a, x) in mode.a_complex.iter().zip(d) {
            compensated_add(real, a.re * x);
            compensated_add(imag, imaginary_sign * a.im * x);
        }
    }
    let mut modes = Vec::new();
    for (h, [real, imag]) in carriers {
        let amplitude = Complex::new(real.0 + real.1, imag.0 + imag.1);
        if !amplitude.re.is_finite() || !amplitude.im.is_finite() {
            return Err(invalid(
                "superexchange_link",
                format!("non-finite projected carrier at harmonic {h}"),
            ));
        }
        if amplitude.re != 0.0 || amplitude.im != 0.0 {
            let (h, amplitude) = match isize::try_from(h) {
                Ok(h) => (h, amplitude),
                Err(_) => (isize::MIN, amplitude.conj()),
            };
            modes.push(LightMode::new(h, ndarray::arr1(&[amplitude])));
        }
    }
    Ok(FloquetDrive::with_modes(drive.omega0_ev, modes))
}

fn invalid(invariant: &'static str, message: String) -> TbError {
    TbError::InvalidMagnonInvariant { invariant, message }
}

#[cfg(test)]
#[path = "superexchange_tests.rs"]
mod tests;
