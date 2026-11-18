//! Sparse finite-Fock Sambe kernel for half-filled clusters ($L\le4$).
use super::{StrongCouplingOptions, invalid};
use crate::{Result, TbError};
use ndarray::{Array2, Array3};
use num_complex::Complex;
type C = Complex<f64>;

#[derive(Clone, Copy)]
struct Hop {
    row: usize,
    col: usize,
    sign: f64,
    amplitude: C,
}
fn fail(name: &'static str) -> TbError {
    invalid(name, "integer/index or allocation size overflow")
}
fn fermion_hop(state: u16, create: usize, annihilate: usize) -> Option<(u16, f64)> {
    if state & (1 << annihilate) == 0 || state & (1 << create) != 0 {
        return None;
    }
    let parity = |s: u16, mode: usize| (s & ((1u16 << mode) - 1)).count_ones() % 2;
    let a = if parity(state, annihilate) == 0 {
        1.0
    } else {
        -1.0
    };
    let after = state ^ (1 << annihilate);
    let b = if parity(after, create) == 0 {
        1.0
    } else {
        -1.0
    };
    Some((after | (1 << create), a * b))
}
fn bytes(n: usize) -> Result<usize> {
    n.checked_mul(std::mem::size_of::<C>())
        .ok_or_else(|| fail("cluster_workspace_bytes"))
}
fn apply_r(a: &Array2<C>, energies: &[f64], p_rows: &[bool], threshold: f64) -> Result<Array2<C>> {
    let mut out = a.clone();
    for r in 0..out.nrows() {
        if p_rows[r] {
            out.row_mut(r).fill(C::new(0., 0.));
            continue;
        }
        let d = energies[r];
        let active = out.row(r).iter().any(|z| *z != C::new(0., 0.));
        if !d.is_finite() || d.abs() < threshold {
            if active {
                return Err(invalid(
                    "nonresonant_cluster_state",
                    format!("coupled state |Nd U+mW|={:e} eV", d.abs()),
                ));
            }
            continue;
        }
        for z in out.row_mut(r) {
            *z /= -d;
        }
    }
    finite(&out, "finite_cluster_intermediate")?;
    Ok(out)
}
fn finite(a: &Array2<C>, invariant: &'static str) -> Result<()> {
    if a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        Err(invalid(invariant, "non-finite intermediate"))
    } else {
        Ok(())
    }
}
fn gram(a: &Array2<C>, b: &Array2<C>) -> Array2<C> {
    a.t().mapv(|z| z.conj()).dot(b)
}
// Bound each real/imaginary component before cancellation in a†b.
fn gram_scale(a: &Array2<C>, b: &Array2<C>) -> Result<f64> {
    let max_component = |x: &Array2<C>| {
        x.iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0., f64::max)
    };
    let x = max_component(a);
    let y = max_component(b);
    // Multiplying the smaller operand by the integer factor first avoids losing
    // subnormal products before the row sum; overflow is rejected, not saturated.
    let scale = (x.min(y) * (2. * a.nrows() as f64)) * x.max(y);
    if !scale.is_finite() {
        return Err(invalid(
            "finite_cluster_scale",
            "Gram magnitude bound overflow",
        ));
    }
    Ok(scale)
}
fn hermitize(a: &mut Array2<C>, scale: f64) -> Result<()> {
    finite(a, "finite_cluster_coefficients")?;
    if !scale.is_finite() || scale < 0. {
        return Err(invalid(
            "finite_cluster_scale",
            "invalid pre-cancellation scale",
        ));
    }
    let within = |x: f64| x == 0. || (scale > 0. && x.abs() / scale <= 1e-10);
    for i in 0..a.nrows() {
        if !within(a[[i, i]].im) {
            return Err(invalid(
                "hermitian_cluster_coefficient",
                "diagonal imaginary residual is too large",
            ));
        }
        a[[i, i]].im = 0.;
        for j in i + 1..a.ncols() {
            let x = a[[i, j]];
            let y = a[[j, i]].conj();
            if x == y {
                continue; // Preserve equal subnormal entries exactly.
            }
            let defect = if scale > 0. {
                (x.re / scale - y.re / scale).hypot(x.im / scale - y.im / scale)
            } else {
                f64::INFINITY
            };
            if !defect.is_finite() || defect > 1e-10 {
                return Err(invalid(
                    "hermitian_cluster_coefficient",
                    format!("Hermiticity residual {defect:e}"),
                ));
            }
            let midpoint = |a: f64, b: f64| {
                if a == b {
                    a
                } else if a.is_sign_negative() == b.is_sign_negative() {
                    a + (b - a) * 0.5
                } else {
                    (a + b) * 0.5
                }
            };
            let z = C::new(midpoint(x.re, y.re), midpoint(x.im, y.im));
            a[[i, j]] = z;
            a[[j, i]] = z.conj();
        }
    }
    Ok(())
}

/// Expand unit-amplitude hopping coefficients through order 2/3/4.
/// `hopping[n+H,i,j]` multiplies $c^\dagger\_i c\_j$ in the single-hop Sambe
/// block.
pub(super) fn expand_cluster(
    hopping: &Array3<C>,
    u: f64,
    omega: f64,
    options: &StrongCouplingOptions,
    work_used: &mut usize,
    workspace_bytes: usize,
) -> Result<(Vec<Array2<C>>, [f64; 3])> {
    let l = hopping.shape()[1];
    let h = options.hop_harmonic_max;
    let m = options.intermediate_photon_max;
    let hcount = h
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| fail("hop_harmonic_range"))?;
    if !(2..=4).contains(&l) || hopping.shape() != [hcount, l, l] {
        return Err(invalid(
            "cluster_hopping_shape",
            "expected (2H+1,L,L), 2<=L<=4",
        ));
    }
    if !(2..=4).contains(&options.max_hopping_order)
        || h < 0
        || m < 0
        || !u.is_finite()
        || u <= 0.
        || !omega.is_finite()
        || omega <= 0.
        || !options.min_detuning_ev.is_finite()
        || options.min_detuning_ev <= 0.
        || !options.max_hopping_ratio.is_finite()
        || options.max_hopping_ratio <= 0.
        || options.max_hopping_ratio >= 1.
        || options.max_work == 0
    {
        return Err(invalid(
            "cluster_expansion_parameters",
            "invalid order, cutoff, U/W, or budget",
        ));
    }
    let np = m
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| fail("photon_range"))?;
    // Every apply_sparse visits all photon/harmonic bins, even for empty hopping.
    // Reject unavoidable scan work before allocating the large Sambe rectangles.
    let minimum_scan = np
        .checked_mul(hcount)
        .and_then(|x| x.checked_mul(if options.max_hopping_order >= 3 { 2 } else { 1 }))
        .and_then(|x| x.checked_add(*work_used))
        .ok_or_else(|| fail("cluster_work"))?;
    if minimum_scan > options.max_work {
        return Err(invalid(
            "cluster_work_budget",
            "photon/harmonic bin scans exceed remaining work budget",
        ));
    }
    if hopping
        .iter()
        .any(|z| !z.re.is_finite() || !z.im.is_finite())
    {
        return Err(invalid("finite_cluster_hopping", "hopping must be finite"));
    }
    for i in 0..l {
        for j in 0..l {
            for n in -h..=h {
                if hopping[[(n + h) as usize, i, j]] != hopping[[((-n) + h) as usize, j, i]].conj()
                {
                    return Err(invalid(
                        "hermitian_cluster_harmonics",
                        "h_n(i,j) must equal conjugate h_-n(j,i)",
                    ));
                }
                if i == j && hopping[[(n + h) as usize, i, i]] != C::new(0., 0.) {
                    return Err(invalid(
                        "zero_cluster_onsite_hopping",
                        "common onsite removed; diagonal hopping must vanish at every harmonic",
                    ));
                }
            }
        }
    }
    let total = 1usize
        .checked_shl(u32::try_from(2 * l).map_err(|_| fail("fock_modes"))?)
        .ok_or_else(|| fail("fock_dimension"))?;
    let nf = (0..total).filter(|s| s.count_ones() as usize == l).count();
    let dim = nf.checked_mul(np).ok_or_else(|| fail("sambe_dimension"))?;
    let pd = 1usize
        .checked_shl(l as u32)
        .ok_or_else(|| fail("tensor_dimension"))?;
    let rect = bytes(
        dim.checked_mul(pd)
            .ok_or_else(|| fail("cluster_workspace_bytes"))?,
    )?;
    // Requested heap-payload cap: P/X/W/T/RT plus Gram scratch, result/folded tensors,
    // sparse capacities, Fock states/lookup, energies and P flags. Input hopping belongs
    // to the caller. Allocator bookkeeping/rounding and BLAS scratch are not an RSS cap.
    let hops_per_harmonic = nf
        .checked_mul(2)
        .and_then(|x| x.checked_mul(l))
        .and_then(|x| x.checked_mul(l - 1))
        .ok_or_else(|| fail("cluster_workspace_bytes"))?;
    let sparse_bytes = hops_per_harmonic
        .checked_mul(hcount)
        .and_then(|x| x.checked_mul(std::mem::size_of::<Hop>()))
        .and_then(|x| x.checked_add(hcount.checked_mul(std::mem::size_of::<Vec<Hop>>())?))
        .ok_or_else(|| fail("cluster_workspace_bytes"))?;
    let metadata_bytes = dim
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<bool>())
        .and_then(|x| x.checked_add(nf.checked_mul(std::mem::size_of::<u16>())?))
        .and_then(|x| x.checked_add(total.checked_mul(std::mem::size_of::<usize>())?))
        .and_then(|x| {
            x.checked_add(
                (options.max_hopping_order - 1).checked_mul(std::mem::size_of::<Array2<C>>())?,
            )
        })
        .ok_or_else(|| fail("cluster_workspace_bytes"))?;
    let required = rect
        .checked_mul(if options.max_hopping_order == 4 {
            6
        } else if options.max_hopping_order == 3 {
            5
        } else {
            4
        })
        .and_then(|x| x.checked_add(bytes(pd.checked_mul(pd)?.checked_mul(12)?).ok()?))
        .and_then(|x| x.checked_add(sparse_bytes))
        .and_then(|x| x.checked_add(metadata_bytes))
        .ok_or_else(|| fail("cluster_workspace_bytes"))?;
    if required > workspace_bytes {
        return Err(invalid(
            "cluster_workspace_budget",
            format!("estimated numeric payload {required} bytes exceeds cap {workspace_bytes}"),
        ));
    }
    // L<=4 gives at most 256 bit patterns: a fixed-domain table needs no hash
    // allocation or unaccounted growth. Reserve every variable-length payload above.
    let mut states = Vec::with_capacity(nf);
    let mut lookup = vec![usize::MAX; total];
    for s in (0..total).filter(|s| s.count_ones() as usize == l) {
        lookup[s] = states.len();
        states.push(s as u16);
    }
    let mut energies = vec![0.; dim];
    let mut prows = vec![false; dim];
    let mut hops = Vec::with_capacity(hcount);
    for _ in 0..hcount {
        hops.push(Vec::<Hop>::with_capacity(hops_per_harmonic));
    }
    for (si, &s) in states.iter().enumerate() {
        let nd = (0..l)
            .filter(|site| s & (1 << site) != 0 && s & (1 << (site + l)) != 0)
            .count();
        for ph in -m..=m {
            let r = (ph + m) as usize * nf + si;
            energies[r] = omega.mul_add(ph as f64, nd as f64 * u);
            prows[r] = ph == 0 && nd == 0;
        }
        for spin in 0..2 {
            for i in 0..l {
                for j in 0..l {
                    if i == j {
                        continue;
                    }
                    let Some((ns, sgn)) = fermion_hop(s, i + spin * l, j + spin * l) else {
                        continue;
                    };
                    let row = lookup[usize::from(ns)];
                    if row == usize::MAX {
                        return Err(fail("fock_state_lookup"));
                    }
                    for n in -h..=h {
                        let amp = hopping[[(n + h) as usize, i, j]];
                        if amp != C::new(0., 0.) {
                            hops[(n + h) as usize].push(Hop {
                                row,
                                col: si,
                                sign: sgn,
                                amplitude: amp,
                            });
                        }
                    }
                }
            }
        }
    }
    let mut p = Array2::<C>::zeros((dim, pd));
    for bits in 0..pd {
        let mut s = 0u16;
        let mut modes = [0usize; 4];
        for (site, mode) in modes.iter_mut().enumerate().take(l) {
            *mode = site
                + if (bits >> (l - 1 - site)) & 1 != 0 {
                    l
                } else {
                    0
                };
            s |= 1 << *mode;
        }
        let inv = (0..l)
            .flat_map(|i| (i + 1..l).map(move |j| (i, j)))
            .filter(|(i, j)| modes[*i] > modes[*j])
            .count();
        let si = lookup[usize::from(s)];
        if si == usize::MAX {
            return Err(fail("fock_tensor_embedding"));
        }
        p[[m as usize * nf + si, bits]] = C::new(if inv % 2 == 0 { 1. } else { -1. }, 0.);
    }
    let mut work = 0usize;
    let mut x = Array2::<C>::zeros((dim, pd));
    apply_sparse(
        &hops, h, m, nf, pd, &p, &mut x, &energies, &prows, options, true, work_used, &mut work,
    )?;
    let w = apply_r(&x, &energies, &prows, options.min_detuning_ev)?;
    let mut scales = [0.; 3];
    scales[0] = gram_scale(&x, &w)?;
    let mut h2 = gram(&x, &w);
    hermitize(&mut h2, scales[0])?;
    let mut result = Vec::with_capacity(options.max_hopping_order - 1);
    result.push(h2.clone());
    if options.max_hopping_order >= 3 {
        let mut t = Array2::<C>::zeros((dim, pd));
        apply_sparse(
            &hops, h, m, nf, pd, &w, &mut t, &energies, &prows, options, false, work_used,
            &mut work,
        )?;
        scales[1] = gram_scale(&w, &t)?;
        let mut h3 = gram(&w, &t);
        hermitize(&mut h3, scales[1])?;
        result.push(h3);
        if options.max_hopping_order >= 4 {
            let rt = apply_r(&t, &energies, &prows, options.min_detuning_ev)?;
            // The actual two-hop coupling can resolve an intermediate replica
            // that is far outside the one-hop photon window. Check the row
            // norm after coherent path summation and P exclusion, not a bare
            // denominator for a replica that no path couples to.
            for (row, amplitude) in rt.rows().into_iter().enumerate() {
                let ratio = amplitude.iter().fold(0.0_f64, |acc, z| acc.hypot(z.norm()));
                if ratio > options.max_hopping_ratio {
                    return Err(invalid(
                        "cluster_two_hop_ratio",
                        format!(
                            "two-hop resolvent row norm {ratio:e} exceeds {:e} at |Nd U+mW|={:e} eV",
                            options.max_hopping_ratio,
                            energies[row].abs()
                        ),
                    ));
                }
            }
            scales[2] = gram_scale(&t, &rt)?;
            let mut h4 = gram(&t, &rt);
            let norm = gram(&w, &w);
            // Bound the two folded products before they cancel the bare chain.
            // Both square inputs have the same row count, so gram_scale also
            // bounds their ordinary product (conjugation does not change magnitude).
            scales[2] += gram_scale(&norm, &h2)? * 0.5 + gram_scale(&h2, &norm)? * 0.5;
            if !scales[2].is_finite() {
                return Err(invalid(
                    "finite_cluster_scale",
                    "fourth-order magnitude bound overflow",
                ));
            }
            let folded = norm.dot(&h2) + h2.dot(&norm);
            h4 -= &folded.mapv(|z| z * 0.5);
            hermitize(&mut h4, scales[2])?;
            result.push(h4);
        }
    }
    for a in &result {
        finite(a, "finite_cluster_coefficients")?;
    }
    *work_used = work_used
        .checked_add(work)
        .ok_or_else(|| fail("cluster_work"))?;
    Ok((result, scales))
}
fn apply_sparse(
    hops: &[Vec<Hop>],
    h: isize,
    m: isize,
    nf: usize,
    pd: usize,
    input: &Array2<C>,
    out: &mut Array2<C>,
    energies: &[f64],
    prows: &[bool],
    opt: &StrongCouplingOptions,
    check_ratio: bool,
    global: &mut usize,
    local: &mut usize,
) -> Result<()> {
    let budget = opt.max_work.saturating_sub(*global);
    for ph in -m..=m {
        for n in -h..=h {
            *local = local.checked_add(1).ok_or_else(|| fail("cluster_work"))?;
            if *local > budget {
                return Err(invalid(
                    "cluster_work_budget",
                    "photon/harmonic bin scan exceeds remaining work budget",
                ));
            }
            // h_n multiplies exp(-inΩt); on exp(+imΩt), destination is m-n.
            // The derivative contribution and hence Nd U+mW in R are unchanged.
            let dest = ph.checked_sub(n).ok_or_else(|| fail("photon_index"))?;
            if dest < -m || dest > m {
                continue;
            }
            let sb = (ph + m) as usize * nf;
            let db = (dest + m) as usize * nf;
            for hop in &hops[(n + h) as usize] {
                // Candidate scans cost work even when the source row is unoccupied.
                *local = local.checked_add(1).ok_or_else(|| fail("cluster_work"))?;
                if *local > budget {
                    return Err(invalid(
                        "cluster_work_budget",
                        "sparse hopping candidate scans exceed remaining work budget",
                    ));
                }
                let source_active = input.row(sb + hop.col).iter().any(|z| *z != C::new(0., 0.));
                if !source_active {
                    continue;
                }
                let coeff = hop.amplitude * hop.sign;
                if check_ratio && coeff != C::new(0., 0.) && !prows[db + hop.row] {
                    let d = energies[db + hop.row];
                    if !d.is_finite() || d.abs() < opt.min_detuning_ev {
                        return Err(invalid(
                            "nonresonant_cluster_state",
                            format!("coupled state |Nd U+mW|={:e} eV", d.abs()),
                        ));
                    }
                    if check_ratio {
                        let ratio = coeff.norm() / d.abs();
                        if !ratio.is_finite() || ratio > opt.max_hopping_ratio {
                            return Err(invalid(
                                "cluster_hopping_ratio",
                                format!(
                                    "individual hopping ratio {ratio:e} exceeds {}",
                                    opt.max_hopping_ratio
                                ),
                            ));
                        }
                    }
                }
                for c in 0..pd {
                    out[[db + hop.row, c]] += coeff * input[[sb + hop.col, c]];
                }
            }
        }
    }
    finite(out, "finite_cluster_intermediate")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options(order: usize) -> StrongCouplingOptions {
        StrongCouplingOptions {
            max_hopping_order: order,
            hop_harmonic_max: 0,
            intermediate_photon_max: 0,
            min_detuning_ev: 1e-9,
            max_hopping_ratio: 0.99,
            max_work: 1 << 24,
            max_cache_bytes: 1 << 20,
            max_cluster_bytes: 1 << 20,
        }
    }
    fn hop(l: usize, edges: &[(usize, usize, C)]) -> Array3<C> {
        let mut a = Array3::zeros((1, l, l));
        for &(i, j, z) in edges {
            a[[0, i, j]] = z;
            a[[0, j, i]] = z.conj();
        }
        a
    }
    fn close(a: &Array2<C>, b: &Array2<C>, tol: f64) {
        assert_eq!(a.dim(), b.dim());
        assert!((a - b).mapv(|z| z.norm_sqr()).sum().sqrt() < tol);
    }
    fn dense_dimer_reference(hopping: &Array3<C>, u: f64, omega: f64) -> Vec<Array2<C>> {
        let states: Vec<u16> = (0u16..16).filter(|s| s.count_ones() == 2).collect();
        let photons = [-1isize, 0, 1];
        let nf = states.len();
        let dim = nf * photons.len();
        let annihilate = |state: u16, mode: usize| -> Option<(u16, f64)> {
            if state & (1 << mode) == 0 {
                return None;
            }
            let sign = if (state & ((1u16 << mode) - 1)).count_ones() % 2 == 0 {
                1.
            } else {
                -1.
            };
            Some((state ^ (1 << mode), sign))
        };
        let bilinear = |state: u16, create: usize, destroy: usize| -> Option<(u16, f64)> {
            let (state, a) = annihilate(state, destroy)?;
            if state & (1 << create) != 0 {
                return None;
            }
            let b = if (state & ((1u16 << create) - 1)).count_ones() % 2 == 0 {
                1.
            } else {
                -1.
            };
            Some((state | (1 << create), a * b))
        };
        let mut v = Array2::<C>::zeros((dim, dim));
        let mut energy = vec![0.; dim];
        for (pi, &ph) in photons.iter().enumerate() {
            for (si, &state) in states.iter().enumerate() {
                let row = pi * nf + si;
                let nd = (state & 1 != 0 && state & 4 != 0) as usize
                    + (state & 2 != 0 && state & 8 != 0) as usize;
                energy[row] = nd as f64 * u + ph as f64 * omega;
                for (qi, &source_ph) in photons.iter().enumerate() {
                    // Independently project <exp(+i ph t)|H(t)|exp(+i source_ph t)>.
                    // This dense algebra checks sparse assembly, not forward evolution.
                    let delta = source_ph - ph;
                    if !(-1..=1).contains(&delta) {
                        continue;
                    }
                    for (sj, &source) in states.iter().enumerate() {
                        for spin in 0..2 {
                            for i in 0..2 {
                                for j in 0..2 {
                                    let Some((next, sign)) =
                                        bilinear(source, i + 2 * spin, j + 2 * spin)
                                    else {
                                        continue;
                                    };
                                    if next != state {
                                        continue;
                                    }
                                    let col = qi * nf + sj;
                                    let amp = hopping[[(delta + 1) as usize, i, j]] * sign;
                                    v[[row, col]] += amp;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut p = Array2::<C>::zeros((dim, 4));
        for bits in 0..4 {
            let modes = [
                if bits & 2 == 0 { 0 } else { 2 },
                if bits & 1 == 0 { 1 } else { 3 },
            ];
            let state = (1u16 << modes[0]) | (1u16 << modes[1]);
            let sign = if modes[0] < modes[1] { 1. } else { -1. };
            let index = states.iter().position(|&s| s == state).unwrap();
            p[[nf + index, bits]] = C::new(sign, 0.);
        }
        let x = v.dot(&p);
        let mut w = x.clone();
        for r in 0..dim {
            for c in 0..4 {
                w[[r, c]] = if energy[r] == 0. {
                    C::new(0., 0.)
                } else {
                    -x[[r, c]] / energy[r]
                };
            }
        }
        let h2 = gram(&x, &w);
        let t = v.dot(&w);
        let h3 = gram(&w, &t);
        let mut rt = t.clone();
        for r in 0..dim {
            for c in 0..4 {
                rt[[r, c]] = if energy[r] == 0. {
                    C::new(0., 0.)
                } else {
                    -t[[r, c]] / energy[r]
                };
            }
        }
        let mut h4 = gram(&t, &rt);
        let norm = gram(&w, &w);
        h4 -= &(norm.dot(&h2) + h2.dot(&norm)).mapv(|z| z * 0.5);
        vec![h2, h3, h4]
    }
    #[test]
    fn positive_time_basis_harmonic_lowers_destination_photon() {
        // A single h_+1 term times exp(+i*0*t) is exp(-it), not exp(+it).
        // Test assembly directly: dimer exchange alone is time-reversal blind.
        let hops = vec![
            vec![],
            vec![],
            vec![Hop {
                row: 0,
                col: 0,
                sign: 1.,
                amplitude: C::new(0.2, 0.3),
            }],
        ];
        let mut input = Array2::<C>::zeros((3, 1));
        input[[1, 0]] = C::new(1., 0.);
        let mut out = Array2::<C>::zeros((3, 1));
        apply_sparse(
            &hops,
            1,
            1,
            1,
            1,
            &input,
            &mut out,
            &[-3., 0., 3.],
            &[false, true, false],
            &options(2),
            false,
            &mut 0,
            &mut 0,
        )
        .unwrap();
        assert_eq!(out[[0, 0]], C::new(0.2, 0.3));
        assert_eq!(out[[2, 0]], C::new(0., 0.));
    }
    #[test]
    fn cancellation_scales_survive_zero_flux_triangle_and_scale_by_order() {
        let run = |t| {
            let z = C::new(t, 0.);
            let a = hop(3, &[(0, 1, z), (1, 2, z), (2, 0, z)]);
            expand_cluster(&a, 4., 3., &options(4), &mut 0, 1 << 20).unwrap()
        };
        let (large, scales) = run(0.1);
        let (_, small_scales) = run(0.01);
        let residual = large[1]
            .iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0., f64::max);
        assert!(scales[1] > 0. && residual < scales[1] * 1e-12);
        for i in 0..3 {
            assert!(scales[i].is_finite() && scales[i] > 0.);
            assert!((small_scales[i] / scales[i] - 0.1_f64.powi(i as i32 + 2)).abs() < 1e-15);
        }
    }
    #[test]
    fn hermitize_has_no_energy_floor_and_preserves_equal_extreme_entries() {
        for value in [f64::from_bits(1), 1e308] {
            let mut a = Array2::<C>::zeros((2, 2));
            a[[0, 1]] = C::new(value, 0.);
            a[[1, 0]] = C::new(value, 0.);
            hermitize(&mut a, value).unwrap();
            assert_eq!(a[[0, 1]].re, value);
            assert_eq!(a[[1, 0]].re, value);
        }
        let mut tiny = Array2::<C>::zeros((1, 1));
        tiny[[0, 0]] = C::new(1e-100, 1e-105);
        assert!(hermitize(&mut tiny, 1e-100).is_err());
    }
    #[test]
    fn static_dimer_h2_h4_are_singlet_projectors() {
        let a = hop(2, &[(0, 1, C::new(1., 0.))]);
        let (r, _) = expand_cluster(&a, 4., 3., &options(4), &mut 0, 1 << 20).unwrap();
        let singlet = Array2::from_shape_fn((4, 4), |(i, j)| {
            C::new([0., 1., -1., 0.][i] * [0., 1., -1., 0.][j] / 2., 0.)
        });
        close(&r[0], &singlet.mapv(|z| -z), 1e-12);
        close(&r[2], &singlet.mapv(|z| z * 0.25), 1e-12);
    }
    #[test]
    fn driven_dimer_h2_sums_both_photon_directions() {
        let mut a = Array3::<C>::zeros((3, 2, 2));
        let forward_plus = C::new(0.2, 0.1);
        let forward_minus = C::new(-0.1, 0.15);
        a[[2, 0, 1]] = forward_plus;
        a[[0, 0, 1]] = forward_minus;
        a[[0, 1, 0]] = forward_plus.conj();
        a[[2, 1, 0]] = forward_minus.conj();
        let mut o = options(4);
        o.hop_harmonic_max = 1;
        o.intermediate_photon_max = 1;
        let (r, _) = expand_cluster(&a, 4., 3., &o, &mut 0, 1 << 20).unwrap();
        let projector = Array2::from_shape_fn((4, 4), |(i, j)| {
            C::new([0., 1., -1., 0.][i] * [0., 1., -1., 0.][j] / 2., 0.)
        });
        let k = -2. * (forward_plus.norm_sqr() + forward_minus.norm_sqr()) * (1. + 1. / 7.);
        close(&r[0], &projector.mapv(|z| z * k), 1e-12);
        let dense = dense_dimer_reference(&a, 4., 3.);
        for order in 0..3 {
            close(&r[order], &dense[order], 1e-11);
        }
        assert!(r[2].iter().all(|z| z.re.is_finite() && z.im.is_finite()));
    }
    #[test]
    fn triangle_chirality_matches_independent_flux_reference() {
        let phi: f64 = 0.41;
        let z = C::from_polar(1., phi / 3.);
        let a = hop(3, &[(0, 1, z), (1, 2, z), (2, 0, z)]);
        let (plus, _) = expand_cluster(&a, 4., 3., &options(3), &mut 0, 1 << 20).unwrap();
        let b = hop(3, &[(0, 1, z.conj()), (1, 2, z.conj()), (2, 0, z.conj())]);
        let (minus, _) = expand_cluster(&b, 4., 3., &options(3), &mut 0, 1 << 20).unwrap();
        let mut chi = Array2::<C>::zeros((8, 8));
        let s = [
            [
                [C::new(0., 0.), C::new(0.5, 0.)],
                [C::new(0.5, 0.), C::new(0., 0.)],
            ],
            [
                [C::new(0., 0.), C::new(0., -0.5)],
                [C::new(0., 0.5), C::new(0., 0.)],
            ],
            [
                [C::new(0.5, 0.), C::new(0., 0.)],
                [C::new(0., 0.), C::new(-0.5, 0.)],
            ],
        ];
        for (a, b, c, sign) in [
            (0, 1, 2, 1.),
            (1, 2, 0, 1.),
            (2, 0, 1, 1.),
            (0, 2, 1, -1.),
            (2, 1, 0, -1.),
            (1, 0, 2, -1.),
        ] {
            for i in 0..8 {
                for j in 0..8 {
                    let x = [i >> 2 & 1, i >> 1 & 1, i & 1];
                    let y = [j >> 2 & 1, j >> 1 & 1, j & 1];
                    chi[[i, j]] += s[a][x[0]][y[0]] * s[b][x[1]][y[1]] * s[c][x[2]][y[2]] * sign;
                }
            }
        }
        let tr = |m: &Array2<C>| gram(&chi, m).diag().iter().copied().sum::<C>();
        let c = (tr(&plus[1]) / gram(&chi, &chi).diag().iter().copied().sum::<C>()).re;
        let cm = (tr(&minus[1]) / gram(&chi, &chi).diag().iter().copied().sum::<C>()).re;
        assert!((c + 24. * phi.sin() / 16.).abs() < 1e-11);
        assert!((c + cm).abs() < 1e-12);
    }
    #[test]
    fn disconnected_dimers_are_additive_including_folded_fourth_order() {
        // The 70-state L=4 half-filled Fock sector includes two-doublon Q states;
        // additivity forces their bare/folded fourth-order pieces to cancel.
        let a = hop(4, &[(0, 1, C::new(1., 0.)), (2, 3, C::new(1., 0.))]);
        let (r, _) = expand_cluster(&a, 4., 3., &options(4), &mut 0, 1 << 22).unwrap();
        let local = |bits: f64| {
            let mut v = Array2::<C>::zeros((16, 16));
            for i in 0..16 {
                for j in 0..16 {
                    let mut x = 0.;
                    for base in [0, 2] {
                        // The untouched pair is an identity factor, so its tensor index must match.
                        if (base == 0 && (i >> 2) != (j >> 2)) || (base == 2 && (i & 3) != (j & 3))
                        {
                            continue;
                        }
                        let bi = (i >> base) & 3;
                        let bj = (j >> base) & 3;
                        let si = if bi == 1 {
                            1.
                        } else if bi == 2 {
                            -1.
                        } else {
                            0.
                        };
                        let sj = if bj == 1 {
                            1.
                        } else if bj == 2 {
                            -1.
                        } else {
                            0.
                        };
                        x += si * sj * bits / 8.;
                    }
                    v[[i, j]] = C::new(x, 0.);
                }
            }
            v
        };
        let expected_h2 = local(-4.);
        let expected_h4 = local(1.0);
        assert_eq!(expected_h2[[1, 5]], C::new(0., 0.));
        assert_eq!(expected_h4[[1, 5]], C::new(0., 0.));
        close(&r[0], &expected_h2, 1e-12);
        close(&r[2], &expected_h4, 1e-12);
        assert!(r[1].iter().all(|z| z.norm() < 1e-12));
    }
    #[test]
    fn workspace_includes_gram_rectangle_scratch_at_orders_two_and_three() {
        let a = hop(2, &[]);
        for (order, live_rectangles) in [(2, 4), (3, 5)] {
            let mut o = options(order);
            o.intermediate_photon_max = 100;
            let rect = 201 * 6 * 4 * std::mem::size_of::<C>();
            // Even zero hopping still allocates Gram conjugation scratch.
            // Large M prevents the small pd^2 reserve from masking a missing rect.
            let error =
                expand_cluster(&a, 4., 3., &o, &mut 0, live_rectangles * rect - 1).unwrap_err();
            assert!(error.to_string().contains("cluster_workspace_budget"));
        }
    }
    #[test]
    fn fused_denominator_preserves_subtraction_residual() {
        let omega = 0.1_f64;
        let u = omega * 3.;
        let detuning = omega.mul_add(-3., u);
        assert_eq!(u - 3. * omega, 0.);
        assert!(detuning.abs() > 2e-17);
        let mut a = Array3::<C>::zeros((7, 2, 2));
        a[[6, 0, 1]] = C::new(1e-18, 0.);
        a[[0, 1, 0]] = C::new(1e-18, 0.);
        let mut o = options(2);
        o.hop_harmonic_max = 3;
        o.intermediate_photon_max = 3;
        o.min_detuning_ev = 2e-17;
        let (result, _) = expand_cluster(&a, u, omega, &o, &mut 0, 1 << 20).unwrap();
        assert!(result[0].iter().any(|z| z.norm() > 0.));
    }
    #[test]
    fn empty_harmonic_bins_fail_work_budget_before_large_allocations() {
        let a = Array3::<C>::zeros((100001, 2, 2));
        let mut o = options(2);
        o.hop_harmonic_max = 50000;
        o.intermediate_photon_max = 25000;
        o.max_work = 1;
        let error = expand_cluster(&a, 4., 3., &o, &mut 0, 1).unwrap_err();
        assert!(error.to_string().contains("cluster_work_budget"));
    }
    #[test]
    fn inactive_sparse_candidates_still_consume_work() {
        let hops = vec![vec![Hop {
            row: 0,
            col: 0,
            sign: 1.,
            amplitude: C::new(1., 0.),
        }]];
        let input = Array2::<C>::zeros((1, 1));
        let mut out = input.clone();
        let mut o = options(2);
        o.max_work = 1; // One bin fits, its inactive candidate does not.
        let error = apply_sparse(
            &hops,
            0,
            0,
            1,
            1,
            &input,
            &mut out,
            &[1.],
            &[false],
            &o,
            false,
            &mut 0,
            &mut 0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("cluster_work_budget"));
    }
    #[test]
    fn nonzero_photon_singly_occupied_pole_is_checked_only_at_fourth_order() {
        let mut a = Array3::<C>::zeros((3, 2, 2));
        for n in 0..3 {
            a[[n, 0, 1]] = C::new(if n == 1 { 0.03 } else { 0.02 }, 0.);
            a[[n, 1, 0]] = a[[n, 0, 1]];
        }
        let mut o = options(3);
        o.hop_harmonic_max = 1;
        o.intermediate_photon_max = 1;
        o.min_detuning_ev = 0.5;
        assert!(expand_cluster(&a, 4., 0.25, &o, &mut 0, 1 << 20).is_ok());
        o.max_hopping_order = 4;
        let error = expand_cluster(&a, 4., 0.25, &o, &mut 0, 1 << 20).unwrap_err();
        assert!(error.to_string().contains("nonresonant_cluster_state"));
    }
    #[test]
    fn workspace_and_actual_resonance_fail_closed() {
        let a = hop(2, &[(0, 1, C::new(1., 0.))]);
        assert!(expand_cluster(&a, 4., 3., &options(4), &mut 0, 1).is_err());
        let mut pole_hop = Array3::<C>::zeros((3, 2, 2));
        pole_hop[[0, 0, 1]] = C::new(1., 0.);
        pole_hop[[2, 1, 0]] = C::new(1., 0.);
        let mut o = options(2);
        o.hop_harmonic_max = 1;
        o.intermediate_photon_max = 1;
        assert!(expand_cluster(&pole_hop, 3., 3., &o, &mut 0, 1 << 20).is_err());
    }
}
