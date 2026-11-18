//! Translation-orbit vertex clusters for occupation-restoring processes.
//!
//! With $p$ offsite hops every visited vertex has equal nonzero incoming/outgoing
//! degree, hence at most $p$ vertices. A connected balanced directed multigraph
//! admits an Euler tour. Closed walks therefore generate candidate supports
//! through $p=4$, but NOT process weights: the Fock kernel still sums all
//! hopping orders, including intermediate disconnected moves and two doublons.

use super::invalid;
use crate::{Model, RMatrixData, Result};
use num_complex::Complex;
use std::{collections::BTreeSet, mem::size_of};

/// A physical spin site: an orbital row and an integer unit-cell translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PeriodicSpinSite<const DIM: usize> {
    pub orbital: usize,
    pub cell: [isize; DIM],
}

fn add<const D: usize>(a: [isize; D], b: [isize; D]) -> Result<[isize; D]> {
    let mut out = [0; D];
    for k in 0..D {
        out[k] = a[k].checked_add(b[k]).ok_or_else(|| {
            invalid(
                "translation_overflow",
                "cluster translation addition overflow",
            )
        })?;
    }
    Ok(out)
}

/// Minimum over every vertex anchor, without point-group/automorphism division.
/// A common translation preserves lexicographic vertex order, so a sorted
/// subset's order also remains its local tensor-spin order after re-anchoring.
pub(super) fn canonical_vertices<const D: usize>(
    sites: &[PeriodicSpinSite<D>],
) -> Result<Vec<PeriodicSpinSite<D>>> {
    let mut unique = sites.to_vec();
    unique.sort();
    unique.dedup();
    let mut best = None;
    for anchor in &unique {
        let mut translated = Vec::with_capacity(unique.len());
        for site in &unique {
            let mut cell = [0; D];
            for (a, value) in cell.iter_mut().enumerate() {
                *value = site.cell[a].checked_sub(anchor.cell[a]).ok_or_else(|| {
                    invalid("translation_overflow", "cluster re-anchoring overflow")
                })?;
            }
            translated.push(PeriodicSpinSite {
                orbital: site.orbital,
                cell,
            });
        }
        if best.as_ref().is_none_or(|previous| &translated < previous) {
            best = Some(translated);
        }
    }
    Ok(best.unwrap_or_default())
}

struct Enumerator<'a, const D: usize> {
    neighbors: Vec<Vec<(usize, [isize; D])>>,
    max_order: usize,
    work: &'a mut usize,
    limit: usize,
    bytes_left: usize,
    clusters: BTreeSet<Vec<PeriodicSpinSite<D>>>,
}

impl<const D: usize> Enumerator<'_, D> {
    fn insert(&mut self, key: Vec<PeriodicSpinSite<D>>) -> Result<()> {
        if !self.clusters.contains(&key) {
            // Reserve one header in the set and one in the sorted output
            // buffer during conversion; the site payload moves without cloning.
            // Both capacities are bounded here, not allocator/map-node overhead.
            let bytes = key.capacity() * size_of::<PeriodicSpinSite<D>>()
                + 2 * size_of::<Vec<PeriodicSpinSite<D>>>();
            self.bytes_left = self.bytes_left.checked_sub(bytes).ok_or_else(|| {
                invalid(
                    "cluster_workspace_budget",
                    "periodic cluster keys exceed max_cluster_bytes",
                )
            })?;
            self.clusters.insert(key);
        }
        Ok(())
    }

    fn register(&mut self, path: &[PeriodicSpinSite<D>]) -> Result<()> {
        let cluster = canonical_vertices(path)?;
        if self.clusters.contains(&cluster) {
            // Any existing key, including one found as a subset of a larger
            // key, already has every nonempty proper subset in the same set.
            return Ok(());
        }
        // At most four vertices. Close under actual vertex subsets, including
        // disconnected ones, before recording the new full key.
        for mask in 1usize..((1usize << cluster.len()) - 1) {
            let subset: Vec<_> = cluster
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, site)| *site)
                .collect();
            self.insert(canonical_vertices(&subset)?)?;
        }
        self.insert(cluster)
    }

    fn visit(&mut self, path: &mut Vec<PeriodicSpinSite<D>>) -> Result<()> {
        if path.len() - 1 == self.max_order {
            return Ok(());
        }
        let current = *path.last().expect("nonempty rooted path");
        for index in 0..self.neighbors[current.orbital].len() {
            *self.work = self
                .work
                .checked_add(1)
                .ok_or_else(|| invalid("expansion_work", "cluster work counter overflow"))?;
            if *self.work > self.limit {
                return Err(invalid(
                    "expansion_work",
                    "periodic cluster enumeration exceeds work budget",
                ));
            }
            let (orbital, offset) = self.neighbors[current.orbital][index];
            let next = PeriodicSpinSite {
                orbital,
                cell: add(current.cell, offset)?,
            };
            path.push(next);
            if next == path[0] && path.len() >= 3 {
                self.register(path)?;
            }
            self.visit(path)?;
            path.pop();
        }
        Ok(())
    }
}

/// Include all nonempty vertex subsets (also disconnected induced subsets).
/// Results ascend by vertex count, then canonical key, for Möbius subtraction.
/// The byte limit covers retained key payload and headers, not adjacency
/// or scratch for a single bounded-size walk and its subsets.
pub(super) fn vertex_clusters<const D: usize, R: RMatrixData>(
    model: &Model<true, D, R>,
    max_order: usize,
    work: &mut usize,
    limit: usize,
    max_bytes: usize,
) -> Result<Vec<Vec<PeriodicSpinSite<D>>>> {
    if !(2..=4).contains(&max_order) {
        return Err(invalid(
            "cluster_hopping_order",
            "vertex support enumeration implements orders two through four",
        ));
    }
    model.validate()?;
    let n = model.norb();
    let mut neighbors = vec![Vec::new(); n];
    for (block, translation) in model.hamR.rows().into_iter().enumerate() {
        let mut cell = [0; D];
        for a in 0..D {
            cell[a] = translation[a];
        }
        for (i, list) in neighbors.iter_mut().enumerate() {
            for j in 0..n {
                if i == j && cell == [0; D] {
                    continue;
                }
                if model.ham[[block, i, j]] != Complex::new(0.0, 0.0) {
                    list.push((j, cell));
                }
            }
        }
    }
    for list in &mut neighbors {
        list.sort();
        list.dedup();
    }
    let mut enumerator = Enumerator {
        neighbors,
        max_order,
        work,
        limit,
        bytes_left: max_bytes,
        clusters: BTreeSet::new(),
    };
    for orbital in 0..n {
        enumerator.visit(&mut vec![PeriodicSpinSite {
            orbital,
            cell: [0; D],
        }])?;
    }
    let mut result = Vec::with_capacity(enumerator.clusters.len());
    result.extend(enumerator.clusters);
    result.sort_unstable_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn canonical_translation_and_anchor_order_do_not_change_keys() {
        let a = PeriodicSpinSite {
            orbital: 1,
            cell: [0, 0],
        };
        let b = PeriodicSpinSite {
            orbital: 0,
            cell: [1, -1],
        };
        let c = PeriodicSpinSite {
            orbital: 0,
            cell: [0, 1],
        };
        let original = canonical_vertices(&[a, b, c]).unwrap();
        let shifted: Vec<_> = [c, a, b]
            .iter()
            .map(|p| PeriodicSpinSite {
                orbital: p.orbital,
                cell: add(p.cell, [7, -3]).unwrap(),
            })
            .collect();
        assert_eq!(original, canonical_vertices(&shifted).unwrap());
    }

    #[test]
    fn square_supports_include_disconnected_subsets_and_no_symmetry_division() {
        let mut model =
            Model::<true, 2>::tb_model(array![[1., 0.], [0., 1.]], array![[0., 0.]], None).unwrap();
        model.set_hop(0.1, 0, 0, &array![1, 0], None);
        model.set_hop(0.1, 0, 0, &array![0, 1], None);
        let clusters = vertex_clusters(&model, 4, &mut 0, 100_000, usize::MAX).unwrap();
        let counts: Vec<_> = (1..=4)
            .map(|n| clusters.iter().filter(|c| c.len() == n).count())
            .collect();
        assert_eq!(counts, vec![1, 6, 6, 1]);
        assert!(clusters.iter().any(|c| c.len() == 2 && {
            let difference = [c[1].cell[0] - c[0].cell[0], c[1].cell[1] - c[0].cell[1]];
            difference[0].abs() == 1 && difference[1].abs() == 1
        }));
        assert!(vertex_clusters(&model, 4, &mut 0, 1, usize::MAX).is_err());
    }

    #[test]
    fn cluster_key_cap_rejects_before_walk_enumeration_finishes() {
        let mut model = Model::<true, 1>::tb_model(array![[1.]], array![[0.]], None).unwrap();
        model.set_hop(0.1, 0, 0, &array![1], None);
        // One singleton and one bond orbit, with one header each in the set
        // and the result. Their backing buffers are moved, not copied.
        let cap = 4 * size_of::<Vec<PeriodicSpinSite<1>>>() + 3 * size_of::<PeriodicSpinSite<1>>();
        let mut full_work = 0;
        let clusters = vertex_clusters(&model, 2, &mut full_work, 100, cap).unwrap();
        assert_eq!(clusters.iter().map(Vec::len).collect::<Vec<_>>(), [1, 2]);
        let mut stopped_work = 0;
        let error = vertex_clusters(&model, 2, &mut stopped_work, 100, cap - 1).unwrap_err();
        assert!(error.to_string().contains("cluster_workspace_budget"));
        assert!(stopped_work < full_work);
    }

    #[test]
    fn every_representable_nonzero_hopping_remains_in_support() {
        let mut model = Model::<true, 1>::tb_model(array![[1.]], array![[0.]], None).unwrap();
        model.set_hop(0.1, 0, 0, &array![1], None);
        for z in &mut model.ham {
            if *z != Complex::new(0.0, 0.0) {
                *z = Complex::new(f64::from_bits(1), 0.0);
            }
        }
        let clusters = vertex_clusters(&model, 2, &mut 0, 100, usize::MAX).unwrap();
        assert_eq!(clusters.iter().filter(|c| c.len() == 2).count(), 1);
    }

    #[test]
    fn molecular_triangle_is_one_vertex_orbit_not_divided_by_three() {
        let mut model =
            Model::<true, 1>::tb_model(array![[3.]], array![[0.], [0.3], [0.6]], None).unwrap();
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            model.set_hop(0.1, i, j, &array![0], None);
        }
        let clusters = vertex_clusters(&model, 3, &mut 0, 100_000, usize::MAX).unwrap();
        assert_eq!(clusters.iter().filter(|c| c.len() == 3).count(), 1);
        assert_eq!(clusters.iter().filter(|c| c.len() == 2).count(), 3);
    }
}
