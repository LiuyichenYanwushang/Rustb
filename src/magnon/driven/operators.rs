//! Quantum spin-half invariant-operator basis and ordinary tensor embedding.
use super::invalid;
use crate::Result;
use ndarray::{Array1, Array2};
use ndarray_linalg::Solve;
use num_complex::Complex;

type C = Complex<f64>;

/// Scalar operators in the spin-rotation-invariant $S=1/2$ site basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpinMonomial {
    Constant,
    Pair([usize; 2]),
    Chirality([usize; 3]),
    Quartic([[usize; 2]; 2]),
}

/// Reusable invariant basis and its Gram matrix for one to four sites.
#[derive(Clone, Debug)]
pub(super) struct SpinBasis {
    nsites: usize,
    terms: Vec<(SpinMonomial, Array2<C>)>,
    gram: Array2<f64>,
}

impl SpinBasis {
    pub(super) fn new(nsites: usize) -> Result<Self> {
        if !(1..=4).contains(&nsites) {
            return Err(invalid(
                "spin_basis_sites",
                "expected one through four sites",
            ));
        }
        let dim = 1usize << nsites;
        let mut terms = vec![(SpinMonomial::Constant, Array2::eye(dim))];
        for i in 0..nsites {
            for j in (i + 1)..nsites {
                terms.push((SpinMonomial::Pair([i, j]), pair_operator(nsites, i, j)));
            }
        }
        for i in 0..nsites {
            for j in (i + 1)..nsites {
                for k in (j + 1)..nsites {
                    terms.push((
                        SpinMonomial::Chirality([i, j, k]),
                        chirality_operator(nsites, i, j, k),
                    ));
                }
            }
        }
        if nsites == 4 {
            for pairing in [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]] {
                let product = pair_operator(nsites, pairing[0][0], pairing[0][1])
                    .dot(&pair_operator(nsites, pairing[1][0], pairing[1][1]));
                terms.push((SpinMonomial::Quartic(pairing), product));
            }
        }
        let count = terms.len();
        let mut gram = Array2::zeros((count, count));
        for a in 0..count {
            for b in 0..count {
                gram[[a, b]] = trace_product(&terms[a].1, &terms[b].1).re / dim as f64;
            }
        }
        Ok(Self {
            nsites,
            terms,
            gram,
        })
    }

    pub(super) fn dimension(&self) -> usize {
        self.terms.len()
    }

    /// Decompose with a relative tolerance on the pre-cancellation energy scale.
    pub(super) fn decompose(
        &self,
        h: &Array2<C>,
        formation_scale_ev: f64,
        relative_tolerance: f64,
    ) -> Result<Vec<(SpinMonomial, f64)>> {
        let dim = 1usize << self.nsites;
        if h.dim() != (dim, dim) {
            return Err(invalid(
                "spin_matrix_shape",
                format!("expected ({dim},{dim})"),
            ));
        }
        if !formation_scale_ev.is_finite()
            || formation_scale_ev < 0.0
            || !relative_tolerance.is_finite()
            || relative_tolerance < 0.0
        {
            return Err(invalid(
                "spin_decomposition_tolerance",
                "formation scale and relative tolerance must be finite and nonnegative",
            ));
        }
        if h.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err(invalid(
                "finite_spin_matrix",
                "matrix contains non-finite values",
            ));
        }
        let hscale = h
            .iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0.0_f64, f64::max);
        if hscale == 0.0 {
            return Ok(self.terms.iter().map(|(term, _)| (*term, 0.0)).collect());
        }
        // Scale very small matrices up by an exact binary power before trace
        // products with Pauli/2. Leave ordinary matrices unscaled to preserve
        // exact zeros from cancellation (and never scale a large matrix down).
        let small_unit = f64::MIN_POSITIVE.sqrt(); // 2^-511 eV
        let unit_ev = if hscale < small_unit { small_unit } else { 1.0 };
        let h = h.mapv(|z| C::new(z.re / unit_ev, z.im / unit_ev));
        let reference = formation_scale_ev.max(hscale);
        let exceeds = |r: f64| !r.is_finite() || r * (unit_ev / reference) > relative_tolerance;
        let hermitian_residual = h
            .iter()
            .zip(h.t().iter())
            .map(|(a, b)| (*a - b.conj()).norm())
            .fold(0.0_f64, f64::max);
        if exceeds(hermitian_residual) {
            return Err(invalid(
                "hermitian_spin_matrix",
                format!("scaled anti-Hermitian residual {hermitian_residual:e} exceeds tolerance"),
            ));
        }
        let mut rhs = Array1::zeros(self.terms.len());
        for (i, (_, basis)) in self.terms.iter().enumerate() {
            let overlap = trace_product(basis, &h) / dim as f64;
            if exceeds(overlap.im.abs()) {
                return Err(invalid(
                    "real_spin_coefficients",
                    format!(
                        "coefficient trace imaginary part {:e} exceeds tolerance",
                        overlap.im.abs()
                    ),
                ));
            }
            rhs[i] = overlap.re;
        }
        let coefficients = self
            .gram
            .clone()
            .solve_into(rhs)
            .map_err(|e| invalid("spin_basis_gram_solve", e.to_string()))?;
        if coefficients.iter().any(|x| !x.is_finite()) {
            return Err(invalid(
                "finite_spin_coefficients",
                "Gram solve produced non-finite coefficients",
            ));
        }
        let mut reconstructed = Array2::<C>::zeros((dim, dim));
        for ((_, basis), &coefficient) in self.terms.iter().zip(coefficients.iter()) {
            reconstructed += &basis.mapv(|z| z * coefficient);
        }
        let residual = h
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (*a - *b).norm())
            .fold(0.0_f64, f64::max);
        if exceeds(residual) {
            return Err(invalid(
                "su2_invariant_spin_matrix",
                format!(
                    "scaled reconstruction residual {residual:e} exceeds tolerance (energy unit {unit_ev:e} eV)"
                ),
            ));
        }
        Ok(self
            .terms
            .iter()
            .zip(coefficients.iter())
            .map(|((term, _), &coefficient)| (*term, coefficient * unit_ev))
            .collect())
    }
}

/// Embed a local operator into a larger ordinary tensor product.
pub(super) fn embed_spin_matrix(
    local: &Array2<C>,
    parent_sites: usize,
    site_map: &[usize],
) -> Result<Array2<C>> {
    if !(1..=4).contains(&parent_sites) || site_map.is_empty() || site_map.len() > parent_sites {
        return Err(invalid(
            "spin_embedding_sites",
            "local and parent site counts must be in 1..=4",
        ));
    }
    let local_dim = 1usize
        .checked_shl(site_map.len() as u32)
        .ok_or_else(|| invalid("spin_embedding_size", "local dimension overflow"))?;
    let parent_dim = 1usize
        .checked_shl(parent_sites as u32)
        .ok_or_else(|| invalid("spin_embedding_size", "parent dimension overflow"))?;
    if local.dim() != (local_dim, local_dim) {
        return Err(invalid(
            "spin_embedding_shape",
            format!("expected ({local_dim},{local_dim}) local matrix"),
        ));
    }
    let mut seen = vec![false; parent_sites];
    for &site in site_map {
        if site >= parent_sites || seen[site] {
            return Err(invalid(
                "spin_embedding_map",
                "site map must be distinct and in range",
            ));
        }
        seen[site] = true;
    }
    if local.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(invalid(
            "finite_spin_embedding",
            "local matrix contains non-finite values",
        ));
    }
    let mut out = Array2::zeros((parent_dim, parent_dim));
    for row in 0..parent_dim {
        for col in 0..parent_dim {
            let mut local_row = 0usize;
            let mut local_col = 0usize;
            let mut compatible = true;
            for (local_site, &site) in site_map.iter().enumerate() {
                let shift = parent_sites - 1 - site;
                let rb = (row >> shift) & 1;
                let cb = (col >> shift) & 1;
                local_row |= rb << (site_map.len() - 1 - local_site);
                local_col |= cb << (site_map.len() - 1 - local_site);
            }
            for site in 0..parent_sites {
                if !seen[site] {
                    let shift = parent_sites - 1 - site;
                    if ((row >> shift) & 1) != ((col >> shift) & 1) {
                        compatible = false;
                        break;
                    }
                }
            }
            if compatible {
                out[[row, col]] = local[[local_row, local_col]];
            }
        }
    }
    Ok(out)
}

fn trace_product(a: &Array2<C>, b: &Array2<C>) -> C {
    let mut trace = C::new(0.0, 0.0);
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            trace += a[[i, j]].conj() * b[[i, j]];
        }
    }
    trace
}

fn spin_component(nsites: usize, site: usize, axis: usize) -> Array2<C> {
    let dim = 1usize << nsites;
    let mut out = Array2::zeros((dim, dim));
    for state in 0..dim {
        let bit = (state >> (nsites - 1 - site)) & 1;
        match axis {
            0 => out[[state ^ (1 << (nsites - 1 - site)), state]] = C::new(0.5, 0.0),
            1 => {
                out[[state ^ (1 << (nsites - 1 - site)), state]] = if bit == 0 {
                    C::new(0.0, 0.5)
                } else {
                    C::new(0.0, -0.5)
                }
            }
            _ => out[[state, state]] = C::new(if bit == 0 { 0.5 } else { -0.5 }, 0.0),
        }
    }
    out
}

fn pair_operator(nsites: usize, i: usize, j: usize) -> Array2<C> {
    let mut out = Array2::zeros((1usize << nsites, 1usize << nsites));
    for axis in 0..3 {
        out += &spin_component(nsites, i, axis).dot(&spin_component(nsites, j, axis));
    }
    out
}

fn chirality_operator(nsites: usize, i: usize, j: usize, k: usize) -> Array2<C> {
    let mut out = Array2::zeros((1usize << nsites, 1usize << nsites));
    // epsilon_{abc} S_i^a S_j^b S_k^c, with an independent Cartesian sum.
    for (a, b, c, sign) in [
        (0, 1, 2, 1.0),
        (1, 2, 0, 1.0),
        (2, 0, 1, 1.0),
        (0, 2, 1, -1.0),
        (2, 1, 0, -1.0),
        (1, 0, 2, -1.0),
    ] {
        out += &(spin_component(nsites, i, a)
            .dot(&spin_component(nsites, j, b))
            .dot(&spin_component(nsites, k, c))
            * C::new(sign, 0.0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(a: &Array2<C>, b: &Array2<C>, tolerance: f64) {
        assert_eq!(a.dim(), b.dim());
        assert!(
            a.iter()
                .zip(b.iter())
                .all(|(x, y)| (*x - *y).norm() <= tolerance)
        );
    }

    #[test]
    fn invariant_gram_ranks_and_roundtrip() {
        for (n, rank) in [(1, 1), (2, 2), (3, 5), (4, 14)] {
            let basis = SpinBasis::new(n).unwrap();
            assert_eq!(basis.dimension(), rank);
            assert!(basis.gram.clone().solve_into(Array1::ones(rank)).is_ok());
            let mut h = Array2::<C>::zeros((1 << n, 1 << n));
            for (idx, (_, op)) in basis.terms.iter().enumerate() {
                h += &op.mapv(|z| z * ((idx as f64 - 3.0) * 0.137));
            }
            let got = basis.decompose(&h, 1.0, 1e-12).unwrap();
            for (idx, (_, value)) in got.iter().enumerate() {
                assert!((*value - (idx as f64 - 3.0) * 0.137).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ordinary_dyadic_operator_has_no_extra_pair_support() {
        let h = (Array2::<C>::eye(8) + pair_operator(3, 0, 1) + pair_operator(3, 0, 2)) * -0.25;
        let coefficients = SpinBasis::new(3)
            .unwrap()
            .decompose(&h, 1.0, 1e-12)
            .unwrap();
        for (term, value) in coefficients {
            let expected = match term {
                SpinMonomial::Constant | SpinMonomial::Pair([0, 1] | [0, 2]) => -0.25,
                _ => 0.0,
            };
            assert_eq!(value, expected, "{term:?}");
        }
    }

    #[test]
    fn chirality_epsilon_sign_is_independent() {
        let chi = chirality_operator(3, 0, 1, 2);
        let mut epsilon = Array2::<C>::zeros((8, 8));
        for (a, b, c, sign) in [
            (0, 1, 2, 1.),
            (1, 2, 0, 1.),
            (2, 0, 1, 1.),
            (0, 2, 1, -1.),
            (2, 1, 0, -1.),
            (1, 0, 2, -1.),
        ] {
            epsilon += &(spin_component(3, 0, a)
                .dot(&spin_component(3, 1, b))
                .dot(&spin_component(3, 2, c))
                * sign);
        }
        close(&chi, &epsilon, 0.0);
        let dij = pair_operator(3, 0, 1);
        let djk = pair_operator(3, 1, 2);
        close(
            &chi,
            &((dij.dot(&djk) - djk.dot(&dij)) * C::new(0.0, 1.0)),
            1e-14,
        );
    }

    #[test]
    fn quantum_repeated_site_pair_identity_and_overlap() {
        let sij = pair_operator(2, 0, 1);
        let expected = Array2::<C>::eye(4) * C::new(3.0 / 16.0, 0.0) - &sij * C::new(0.5, 0.0);
        close(&sij.dot(&sij), &expected, 1e-14);
        let dot = sij.dot(&sij);
        let basis = SpinBasis::new(2).unwrap();
        let coefficients = basis.decompose(&dot, 1.0, 1e-12).unwrap();
        let mut rebuilt = Array2::<C>::zeros((4, 4));
        for ((_, op), (_, x)) in basis.terms.iter().zip(coefficients) {
            rebuilt += &op.mapv(|z| z * x);
        }
        close(&dot, &rebuilt, 1e-12);
    }

    #[test]
    fn rejects_noninvariant_nonhermitian_and_embeds_as_tensor_operator() {
        let basis = SpinBasis::new(2).unwrap();
        let mut bad = Array2::<C>::zeros((4, 4));
        bad[[0, 0]] = C::new(1.0, 0.0);
        assert!(basis.decompose(&bad, 1.0, 1e-12).is_err());
        bad[[0, 1]] = C::new(0.0, 1.0);
        assert!(basis.decompose(&bad, 1.0, 1e-12).is_err());
        let pair = pair_operator(2, 0, 1);
        let embedded = embed_spin_matrix(&pair, 3, &[0, 2]).unwrap();
        assert_eq!(embedded.dim(), (8, 8));
        let chi = chirality_operator(3, 0, 1, 2);
        let odd = embed_spin_matrix(&chirality_operator(3, 0, 1, 2), 4, &[0, 1, 2]).unwrap();
        let permuted = embed_spin_matrix(&chirality_operator(3, 0, 1, 2), 4, &[2, 1, 0]).unwrap();
        // Reversing site order reverses the epsilon chirality sign.
        close(&permuted, &(-&odd), 0.0);
        assert!(embed_spin_matrix(&chi, 2, &[0, 0, 1]).is_err());
    }
}
