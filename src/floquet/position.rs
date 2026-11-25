//! Uniform all-field-orders Floquet harmonics in the diagonal-position gauge.

use super::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::AtomicBool;

// Refuse products outside the representable range before a later frequency
// factor can amplify a component that would otherwise silently round to zero.
fn checked_position_product(a: f64, b: f64) -> Result<f64> {
    let value = a * b;
    if !value.is_finite() || (value == 0.0 && a != 0.0 && b != 0.0) {
        return Err(TbError::Other(
            "Floquet position coupling exceeds the representable product range".into(),
        ));
    }
    Ok(value)
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    pub(super) fn validate_floquet_position(&self) -> Result<()> {
        let indices: BTreeMap<_, _> = self
            .hamR
            .outer_iter()
            .enumerate()
            .map(|(i, r)| (r.to_vec(), i))
            .collect();
        if indices.len() != self.hamR.nrows() {
            return Err(TbError::Other(
                "Floquet position input has duplicate translations".into(),
            ));
        }
        let r = self.rmatrix.as_array4();
        for (row, &ir) in &indices {
            let opposite = row
                .iter()
                .map(|x| x.checked_neg())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    TbError::Other("Floquet position translation cannot be negated".into())
                })?;
            let partner = indices.get(&opposite);
            let mismatch = |z: Complex<f64>, y: Complex<f64>| {
                let scale = z.re.abs().max(z.im.abs()).max(y.re.abs()).max(y.im.abs());
                (z.re - y.re).abs().max((z.im - y.im).abs()) > 1e-12 * scale
            };
            for ((i, j), &z) in self.ham.index_axis(Axis(0), ir).indexed_iter() {
                let y = partner.map_or(Complex::ZERO, |&jr| self.ham[[jr, j, i]].conj());
                if mismatch(z, y) {
                    return Err(TbError::InvalidModelInvariant {
                        invariant: "hermitian_hamiltonian",
                        message: format!(
                            "non-Hermitian hopping at translation {row:?}, state pair ({i}, {j})"
                        ),
                    });
                }
            }
            for ((d, i, j), &z) in r.index_axis(Axis(0), ir).indexed_iter() {
                let y = partner.map_or(Complex::ZERO, |&jr| r[[jr, d, j, i]].conj());
                if mismatch(z, y) {
                    return Err(TbError::InvalidModelInvariant {
                        invariant: "hermitian_position_matrix",
                        message: format!(
                            "non-Hermitian position block at translation {row:?}, component {d}, state pair ({i}, {j})"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Both Sambe and van Vleck use this cache, with every row aligned to input hamR.
    /// Given validated Hermitian input, Fourier-expand
    /// `exp(-i a·d) (T + hbar dot(a)·Y)` with `Y` the position minus each state’s own centre.
    pub(super) fn position_harmonic_cache(
        &self,
        drive: &FloquetDrive,
        harmonic_max: isize,
    ) -> Result<FloquetHarmonicCache> {
        let nr = self.hamR.nrows();
        let nsta = self.nsta();
        let count = harmonic_max
            .checked_mul(2)
            .and_then(|k| k.checked_add(1))
            .and_then(|k| usize::try_from(k).ok())
            .ok_or_else(|| TbError::Other("Floquet position harmonic range overflows".into()))?;
        let size = [count, nr, nsta, nsta]
            .into_iter()
            .try_fold(1_usize, usize::checked_mul)
            .filter(|&n| n <= isize::MAX as usize / size_of::<Complex<f64>>())
            .ok_or_else(|| {
                TbError::Other("Floquet position harmonic cache size overflows".into())
            })?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(size).map_err(|e| {
            TbError::Other(format!(
                "Cannot allocate Floquet position harmonic cache: {e}"
            ))
        })?;
        storage.resize(size, Complex::ZERO);
        let mut blocks = Array4::from_shape_vec((count, nr, nsta, nsta), storage)?;
        if drive.modes.is_empty() {
            blocks
                .index_axis_mut(Axis(0), harmonic_max as usize)
                .assign(&self.ham);
            return Ok(FloquetHarmonicCache {
                harmonic_min: -harmonic_max,
                harmonic_max,
                blocks,
                used_time_grid: false,
            });
        }

        let field = coherent_cartesian_field(drive);
        let has_ac = field.range(1..).next().is_some();
        let k = harmonic_max as i128;
        let r = self.rmatrix.as_array4();
        let centres = has_ac.then(|| self.orb.dot(&self.lat));
        let rows: HashMap<[isize; DIM], _> = self
            .hamR
            .outer_iter()
            .enumerate()
            .map(|(ir, row)| (std::array::from_fn(|a| row[a]), ir))
            .collect();
        let fallback_warned = AtomicBool::new(false);
        let mut projected_drives = HashMap::new();
        let mut cached = HashMap::<_, (f64, Array1<Complex<f64>>)>::new();
        let mut terms = Vec::new();
        let mut residual = Vec::new();
        for (ir, row) in self.hamR.outer_iter().enumerate() {
            let mut opposite = [0; DIM];
            for (dst, &src) in opposite.iter_mut().zip(&row) {
                *dst = src.checked_neg().ok_or_else(|| {
                    TbError::Other("Floquet position translation cannot be negated".into())
                })?;
            }
            let Some(&jr) = rows.get(&opposite) else {
                // Only an exactly zero row can lack a partner after validation.
                continue;
            };
            // Select a translation by value rather than input row index,
            // so row permutations do not change the chosen adjoint.
            if row.iter().cmp(&opposite).is_gt() {
                continue;
            }
            for i in 0..nsta {
                for j in 0..nsta {
                    if ir == jr && i > j {
                        continue;
                    }
                    let hopping = self.ham[[ir, i, j]];
                    let mut y: [Complex<f64>; DIM] = std::array::from_fn(|a| r[[ir, a, i, j]]);
                    let diagonal = ir == jr && i == j;
                    if let Some(centres) = centres.as_ref().filter(|_| diagonal) {
                        for (a, z) in y.iter_mut().enumerate() {
                            z.re -= centres[[i, a]];
                        }
                    }
                    if hopping == Complex::ZERO
                        && (!has_ac || y.iter().all(|&z| z == Complex::ZERO))
                    {
                        continue;
                    }
                    let d = if diagonal {
                        Array1::zeros(DIM)
                    } else {
                        self.link_displacement_cartesian(i, j, &row)
                    };
                    let key: [u64; DIM] =
                        std::array::from_fn(|a| if d[a] == 0.0 { 0 } else { d[a].to_bits() });
                    let (projected_drive, gcd) = match projected_drives.entry(key) {
                        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                        std::collections::hash_map::Entry::Vacant(e) => {
                            let projected = coherent_link_drive(&d, drive, &field)?;
                            let gcd = phase_harmonic_gcd(&projected);
                            e.insert((projected, gcd))
                        }
                    };
                    let period = *gcd as i128;
                    terms.clear();
                    if hopping != Complex::ZERO {
                        terms.push((0, hopping));
                    }
                    for (&p, components) in field.range(1..) {
                        let p = p as i128;
                        // The two shifted windows are mirrors. If neither
                        // reaches the projected carrier lattice, this exact
                        // zero cannot tighten the hopping's accuracy budget.
                        let reachable = if period == 0 {
                            p <= k
                        } else {
                            -(-(p - k)).div_euclid(period) <= (p + k).div_euclid(period)
                        };
                        if !reachable || y.iter().all(|&z| z == Complex::ZERO) {
                            continue;
                        }
                        for label in [p, -p] {
                            let mut dipole = Complex::ZERO;
                            let mut tail = Complex::ZERO;
                            let mut add_product = |x, z: Complex<f64>| -> Result<()> {
                                let product = Complex::new(
                                    checked_position_product(x, z.re)?,
                                    checked_position_product(x, z.im)?,
                                );
                                add_coherent_product(&mut dipole, &mut tail, x, z, product);
                                Ok(())
                            };
                            for (&(hi, lo), &z) in components.iter().zip(&y) {
                                for a in [hi, lo] {
                                    if a == Complex::ZERO {
                                        continue;
                                    }
                                    let a = if label > 0 { a } else { a.conj() };
                                    add_product(a.re, z)?;
                                    add_product(a.im, Complex::new(-z.im, z.re))?;
                                }
                            }
                            dipole += tail;
                            if dipole == Complex::ZERO {
                                continue;
                            }
                            let p_half = (label as f64) * 0.5;
                            let w = drive.omega0_ev;
                            let scale = p_half * w;
                            let weight = |x| {
                                if scale.is_normal() {
                                    checked_position_product(x, scale)
                                } else if (x * w).is_normal() {
                                    checked_position_product(x * w, p_half)
                                } else {
                                    checked_position_product(
                                        checked_position_product(x, p_half)?,
                                        w,
                                    )
                                }
                            };
                            // -ipW (b_p . Y), with the factor 1/2 in p_half.
                            let q = Complex::new(weight(dipole.im)?, -weight(dipole.re)?);
                            terms.push((-label, q));
                        }
                    }

                    // Each p needs [-K-p, K-p]. Merge adjacent/overlapping
                    // intervals, not the gaps between distant carriers, and
                    // request only the windows used by this matrix element.
                    terms.sort_unstable_by_key(|&(shift, _)| shift);
                    let start = if diagonal { harmonic_max as usize } else { 0 };
                    // Bound the combined error from all derivative shifts,
                    // including shifts in separately sampled windows.
                    let l1: f64 = terms
                        .iter()
                        .filter(|(shift, _)| *shift != 0)
                        .map(|(_, weight)| weight.norm())
                        .sum();
                    let error_budget = if l1 == 0.0 {
                        f64::INFINITY
                    } else {
                        (1e-8 / l1).min(f64::MAX)
                    };
                    // Preserve a small hopping or dipole term between much
                    // larger, cancelling derivative sidebands.
                    residual.clear();
                    if terms.len() > 1 {
                        residual.resize(count, Complex::ZERO);
                    }
                    let mut remaining = terms.as_slice();
                    while let Some(&(shift, _)) = remaining.first() {
                        let lo = shift - k;
                        let mut hi = shift + k;
                        let mut len = 1;
                        while let Some(&(next, _)) = remaining.get(len) {
                            if next - k > hi + 1 {
                                break;
                            }
                            hi = next + k;
                            len += 1;
                        }
                        let (group, rest) = remaining.split_at(len);

                        let cache_key = (key, lo, hi);
                        if !cached
                            .get(&cache_key)
                            .is_some_and(|(tol, _)| *tol <= error_budget)
                        {
                            let c = position_peierls_coeffs(
                                projected_drive,
                                *gcd,
                                lo,
                                hi,
                                error_budget,
                                &fallback_warned,
                            )?;
                            cached.insert(cache_key, (error_budget, c));
                        }
                        let c = &cached[&cache_key].1;
                        for &(shift, weight) in group {
                            let offset = (shift - k - lo) as usize; // subrange of checked interval
                            for n in start..count {
                                let sum = &mut blocks[[n, ir, i, j]];
                                let contribution = weight * c[offset + n];
                                if let Some(tail) = residual.get_mut(n) {
                                    add_coherent(sum, tail, contribution);
                                } else {
                                    *sum += contribution;
                                }
                            }
                        }
                        remaining = rest;
                    }
                    for n in start..count {
                        let mut z =
                            blocks[[n, ir, i, j]] + residual.get(n).copied().unwrap_or_default();
                        if !z.is_finite() {
                            return Err(TbError::Other(format!(
                                "Floquet position harmonic is not finite at translation {row}, state pair ({i}, {j})"
                            )));
                        }
                        // H_n(R) = H_-n(-R)^dagger, including the self-pair at R=0,n=0.
                        if diagonal && n == start {
                            z.im = 0.0;
                        }
                        blocks[[n, ir, i, j]] = z;
                        blocks[[count - 1 - n, jr, j, i]] = z.conj();
                    }
                }
            }
        }
        Ok(FloquetHarmonicCache {
            harmonic_min: -harmonic_max,
            harmonic_max,
            blocks,
            used_time_grid: fallback_warned.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    // This is the bare position operator in the returned basis, not an
    // observable transformed by the optical gauge and Floquet micromotion.
    pub(super) fn floquet_output_position(
        &self,
        support: &Array2<isize>,
        total: usize,
        photon_basis: Option<&[Vec<usize>]>,
    ) -> Result<R> {
        if !R::HAS_RMATRIX {
            return Ok(self.rmatrix.clone());
        }
        let nr = support.nrows();
        [nr, DIM, total, total]
            .into_iter()
            .try_fold(size_of::<Complex<f64>>(), usize::checked_mul)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                TbError::Other("Floquet position array size exceeds the addressable range".into())
            })?;
        let mut rmatrix = Array4::zeros((nr, DIM, total, total));
        let input: BTreeMap<_, _> = self
            .hamR
            .outer_iter()
            .enumerate()
            .map(|(ir, row)| (row.to_vec(), ir))
            .collect();
        for (ir, row) in support.outer_iter().enumerate() {
            let Some(&old_ir) = input.get(&row.to_vec()) else {
                continue;
            };
            let source = self.rmatrix.as_array4().index_axis(Axis(0), old_ir);
            if let Some(sectors) = photon_basis {
                for sector in sectors {
                    for ((d, i, j), &value) in source.indexed_iter() {
                        rmatrix[[ir, d, sector[i], sector[j]]] = value;
                    }
                }
            } else {
                rmatrix.index_axis_mut(Axis(0), ir).assign(&source);
            }
        }
        Ok(R::from_array(rmatrix))
    }
}
