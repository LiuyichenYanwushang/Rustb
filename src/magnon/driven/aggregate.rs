//! Translation-orbit Möbius subtraction and aggregation of orders three and four.

use super::cluster::expand_cluster;
use super::operators::{SpinBasis, SpinMonomial, embed_spin_matrix};
use super::periodic::{PeriodicSpinSite, canonical_vertices, vertex_clusters};
use super::{SpinPair, StrongCouplingOptions, invalid, negate};
use crate::floquet::FloquetHarmonicCache;
use crate::{Model, RMatrixData, Result, TbError};
use ndarray::{Array2, Array3};
use num_complex::Complex;
use std::collections::{BTreeMap, btree_map::Entry};
use std::mem::size_of;

type C = Complex<f64>;
type Sites<const D: usize, const N: usize> = [PeriodicSpinSite<D>; N];
type PairKey<const D: usize> = (usize, usize, [isize; D]);

#[derive(Default)]
pub(super) struct Aggregate<const D: usize> {
    pub(super) constant_ev: f64,
    pub(super) pairs: Vec<SpinPair<D>>,
    pub(super) chirality: Vec<(Sites<D, 3>, f64)>,
    pub(super) quartic: Vec<([Sites<D, 2>; 2], f64)>,
    pub(super) visited_clusters: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Key<const D: usize> {
    Constant,
    Pair(PairKey<D>),
    Chirality(Sites<D, 3>),
    Quartic([Sites<D, 2>; 2]),
}

fn pair_direction<const D: usize>(
    a: PeriodicSpinSite<D>,
    b: PeriodicSpinSite<D>,
) -> Result<(PairKey<D>, bool)> {
    let mut cell = [0; D];
    for (k, rel) in cell.iter_mut().enumerate() {
        *rel = b.cell[k]
            .checked_sub(a.cell[k])
            .ok_or_else(|| invalid("translation_overflow", "cluster relative cell overflow"))?;
    }
    let forward = (a.orbital, b.orbital, cell);
    let reverse = (b.orbital, a.orbital, negate(cell)?);
    // The same undirected key and forward orientation used by the order-two path.
    Ok(if forward < reverse {
        (forward, true)
    } else {
        (reverse, false)
    })
}

fn budget_error() -> TbError {
    invalid(
        "cluster_workspace_budget",
        "periodic cluster payload exceeds max_cluster_bytes",
    )
}

fn remaining(limit: usize, bytes: usize) -> Result<usize> {
    limit.checked_sub(bytes).ok_or_else(budget_error)
}

fn scale(a: &Array2<C>) -> f64 {
    a.iter()
        .map(|z| z.re.abs().max(z.im.abs()))
        .fold(0.0, f64::max)
}

/// The cache must have been validated for the model and cover the full active
/// single-hop window in `options`. The caller validates the fourth-order
/// intermediate window and accounts for the cache separately.
#[allow(clippy::too_many_arguments)]
pub(super) fn aggregate_clusters<const D: usize, R: RMatrixData>(
    model: &Model<true, D, R>,
    rows: &BTreeMap<[isize; D], usize>,
    cache: &FloquetHarmonicCache,
    u: f64,
    omega: f64,
    options: &StrongCouplingOptions,
    max_cluster_bytes: usize,
    work_used: &mut usize,
) -> Result<Aggregate<D>> {
    let order = options.max_hopping_order;
    if !(3..=4).contains(&order) {
        return Err(invalid(
            "cluster_hopping_order",
            "expected order three or four",
        ));
    }
    let h = options.hop_harmonic_max;
    let harmonics = h
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid("hop_harmonic_range", "harmonic count overflow"))?;
    let clusters = vertex_clusters(model, order, work_used, options.max_work, max_cluster_bytes)?;

    // Logical payload, not RSS: allocator, map-node and BLAS overhead are
    // excluded. Start from the retained keys now that enumeration is done.
    let mut available = max_cluster_bytes;
    available = remaining(
        available,
        clusters
            .capacity()
            .checked_mul(size_of::<Vec<PeriodicSpinSite<D>>>())
            .ok_or_else(budget_error)?,
    )?;
    for sites in &clusters {
        available = remaining(
            available,
            sites
                .capacity()
                .checked_mul(size_of::<PeriodicSpinSite<D>>())
                .ok_or_else(budget_error)?,
        )?;
    }
    available = remaining(
        available,
        clusters
            .len()
            .checked_mul(size_of::<Vec<Array2<C>>>())
            .ok_or_else(budget_error)?,
    )?;
    let mut weights: Vec<Vec<Array2<C>>> = Vec::with_capacity(clusters.len());
    let mut terms = BTreeMap::new();
    let mut visited_clusters = 0;
    let mut basis = None;
    let mut basis_sites = 0;

    for sites in &clusters {
        let n = sites.len();
        if n == 1 {
            weights.push(Vec::new());
            continue;
        }
        // Fixed size bounds: n <= 4. Reserve basis terms, Gram/solve, and
        // spectral/reconstruction scratch alongside the retained weights.
        let dim = 1usize << n;
        let matrix_bytes = dim * dim * size_of::<C>();
        let count = [0, 1, 2, 5, 14][n];
        let basis_bytes = (count + 4) * matrix_bytes
            + 2 * count * size_of::<(SpinMonomial, Array2<C>)>()
            + (2 * count * count + 2 * count) * size_of::<f64>()
            + count * size_of::<(SpinMonomial, f64)>();
        let hopping_bytes = harmonics
            .checked_mul(n * n * size_of::<C>())
            .ok_or_else(budget_error)?;

        let mut hopping = None;
        for a in 0..n {
            for b in a + 1..n {
                let ((i, j, cell), forward) = pair_direction(sites[a], sites[b])?;
                let Some(&block) = rows.get(&cell) else {
                    continue;
                };
                if model.ham[[block, i, j]] == C::new(0.0, 0.0) {
                    continue;
                }
                if hopping.is_none() {
                    remaining(remaining(available, basis_bytes)?, hopping_bytes)?;
                }
                let local = hopping.get_or_insert_with(|| Array3::zeros((harmonics, n, n)));
                let (a, b) = if forward { (a, b) } else { (b, a) };
                for q in -h..=h {
                    let value = cache.harmonic_blocks(q)[[block, i, j]];
                    local[[(h + q) as usize, a, b]] = value;
                    local[[(h - q) as usize, b, a]] = value.conj();
                }
            }
        }
        let Some(hopping) = hopping else {
            weights.push(Vec::new());
            continue;
        };
        let (mut full, [_, h3_scale, h4_scale]) = expand_cluster(
            &hopping,
            u,
            omega,
            options,
            work_used,
            remaining(remaining(available, basis_bytes)?, hopping_bytes)?,
        )?;
        drop(hopping);
        visited_clusters += 1;
        // H2 is needed inside the kernel for H4's folded term, but is NOT a
        // correction here: the parent already sums that order directly.
        full.remove(0);
        available = remaining(
            available,
            full.len() * matrix_bytes + full.capacity() * size_of::<Array2<C>>(),
        )?;
        // The raw Fock terms may already have canceled before subtraction of
        // proper subsets; only the reconstruction tolerance uses these bounds.
        let mut scales = [h3_scale, h4_scale];
        let scratch = matrix_bytes + n * (4 * size_of::<PeriodicSpinSite<D>>() + size_of::<bool>());
        remaining(remaining(available, basis_bytes)?, scratch)?;

        for mask in 1usize..((1 << n) - 1) {
            if mask.count_ones() < 2 {
                continue;
            }
            let mut map = [0; 4];
            let mut subset = Vec::with_capacity(mask.count_ones() as usize);
            for (site_index, &site) in sites.iter().enumerate() {
                if mask & (1 << site_index) != 0 {
                    map[subset.len()] = site_index;
                    subset.push(site);
                }
            }
            let key = canonical_vertices(&subset)?;
            let index = clusters
                .binary_search_by(|c| c.len().cmp(&key.len()).then_with(|| c.cmp(&key)))
                .map_err(|_| invalid("cluster_subset", "missing periodic proper vertex subset"))?;
            // Do not deduplicate masks by periodic key: two subsets can be
            // translates, yet are different embeddings in this parent tensor.
            for ((weight, scale_ev), subweight) in
                full.iter_mut().zip(&mut scales).zip(&weights[index])
            {
                *scale_ev = scale_ev.max(scale(subweight));
                *weight -= &embed_spin_matrix(subweight, n, &map[..subset.len()])?;
            }
        }

        if n != basis_sites {
            drop(basis.take());
            basis = Some(SpinBasis::new(n)?);
            basis_sites = n;
        }
        let basis = basis.as_ref().expect("basis initialized for current size");
        for (weight, full_scale) in full.iter().zip(scales) {
            // Möbius cancellation may make the residual much smaller than
            // either operand. This tolerance checks reconstruction only.
            for (term, value) in basis.decompose(weight, full_scale, 1e-10)? {
                if value == 0.0 {
                    continue;
                }
                let key = match term {
                    SpinMonomial::Constant => Key::Constant,
                    SpinMonomial::Pair([a, b]) => Key::Pair(pair_direction(sites[a], sites[b])?.0),
                    SpinMonomial::Chirality(abc) => {
                        // Local indices and sites are sorted; one common anchor
                        // shift preserves the order and hence the epsilon sign.
                        let key = canonical_vertices(&abc.map(|a| sites[a]))?;
                        Key::Chirality([key[0], key[1], key[2]])
                    }
                    SpinMonomial::Quartic(pairs) => {
                        // Uses all four vertices of the already canonical parent.
                        // Independent pair shifts would lose their displacement.
                        let mut pairs = pairs.map(|pair| pair.map(|a| sites[a]));
                        pairs.sort();
                        Key::Quartic(pairs)
                    }
                };
                let sum = match terms.entry(key) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => {
                        available = remaining(available, size_of::<(Key<D>, f64)>())?;
                        remaining(remaining(available, basis_bytes)?, scratch)?;
                        e.insert(0.0)
                    }
                };
                *sum += value;
                if !sum.is_finite() {
                    return Err(invalid(
                        "finite_spin_coefficient",
                        "per-cell aggregation overflowed",
                    ));
                }
            }
        }
        weights.push(full);
    }

    drop(weights);
    drop(basis);
    drop(clusters);
    // At most one result entry per accumulated key. Check the temporary overlap
    // of ordered reduction and output buffers before reserving exact capacities.
    remaining(
        max_cluster_bytes,
        terms
            .len()
            .checked_mul(2 * size_of::<(Key<D>, f64)>())
            .ok_or_else(budget_error)?,
    )?;
    let mut result = Aggregate {
        visited_clusters,
        ..Aggregate::default()
    };
    let (mut np, mut nc, mut nq) = (0, 0, 0);
    for (key, &value) in &terms {
        if value == 0.0 {
            continue;
        }
        match key {
            Key::Constant => {}
            Key::Pair(_) => np += 1,
            Key::Chirality(_) => nc += 1,
            Key::Quartic(_) => nq += 1,
        }
    }
    result.pairs.reserve_exact(np);
    result.chirality.reserve_exact(nc);
    result.quartic.reserve_exact(nq);
    for (key, value) in terms {
        if value == 0.0 {
            continue;
        }
        match key {
            Key::Constant => result.constant_ev = value,
            Key::Pair((i, j, cell)) => result.pairs.push(SpinPair {
                i,
                j,
                cell,
                exchange_ev: value,
            }),
            Key::Chirality(sites) => result.chirality.push((sites, value)),
            Key::Quartic(pairs) => result.quartic.push((pairs, value)),
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::floquet::FloquetDrive;
    use ndarray::array;

    #[test]
    fn static_periodic_chain_subtracts_both_translated_bond_embeddings() {
        let mut model = Model::<true, 1>::tb_model(array![[1.]], array![[0.]], None).unwrap();
        let (t, u) = (0.2_f64, 4.0_f64);
        model.set_hop(t, 0, 0, &array![1], None);
        let rows = model
            .hamR
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, row)| ([row[0]], i))
            .collect();
        let cache = model
            .validated_harmonic_cache(&FloquetDrive::new(3.), 0, 1 << 20)
            .unwrap();
        let options = StrongCouplingOptions {
            max_hopping_order: 4,
            hop_harmonic_max: 0,
            intermediate_photon_max: 0,
            ..StrongCouplingOptions::default()
        };
        let got =
            aggregate_clusters(&model, &rows, &cache, u, 3., &options, 1 << 20, &mut 0).unwrap();
        let scale = t.powi(4) / u.powi(3);
        assert_eq!(got.visited_clusters, 2); // one bond orbit and one adjacent triple orbit
        assert_eq!(got.pairs.len(), 2);
        assert_eq!(got.pairs[0].cell, [-2]);
        assert_eq!(got.pairs[1].cell, [-1]);
        assert!((got.pairs[0].exchange_ev - 4. * scale).abs() < 1e-15);
        assert!((got.pairs[1].exchange_ev + 16. * scale).abs() < 1e-15);
        assert!((got.constant_ev - 3. * scale).abs() < 1e-15);
        assert!(got.chirality.is_empty());
        assert!(got.quartic.is_empty());
        assert!(aggregate_clusters(&model, &rows, &cache, u, 3., &options, 1, &mut 0).is_err());
    }
}
