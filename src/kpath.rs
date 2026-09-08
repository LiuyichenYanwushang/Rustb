use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use ndarray::prelude::*;
use ndarray_linalg::Inverse;

pub trait Kpath {
    //! Generate high symmetry path from high symmetry points, plot band structure
    /// Sample a piecewise-linear path, including every node exactly.
    /// Distances use reciprocal-lattice units without a `2*pi` factor.
    /// Requires at least two distinct consecutive nodes and `nk >= path.nrows()`.
    fn k_path(
        &self,
        path: &Array2<f64>,
        nk: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Kpath for Model<SPIN, DIM, R> {
    fn k_path(
        &self,
        path: &Array2<f64>,
        nk: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        if self.dim_r() == 0 {
            return Err(TbError::ZeroDimKPathError);
        }
        let n_node: usize = path.len_of(Axis(0));
        if self.dim_r() != path.len_of(Axis(1)) {
            return Err(TbError::PathLengthMismatch {
                expected: self.dim_r(),
                actual: path.len_of(Axis(1)),
            });
        }
        if n_node < 2 || nk < n_node {
            return Err(TbError::Other(
                "k_path requires at least two nodes and at least one sample per node".into(),
            ));
        }
        if path.iter().any(|x| !x.is_finite()) {
            return Err(TbError::Other("k_path requires finite coordinates".into()));
        }
        nk.checked_mul(DIM)
            .and_then(|n| n.checked_mul(size_of::<f64>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| {
                TbError::Other("k_path output size exceeds the addressable array size".into())
            })?;
        let k_metric = (self.lat.dot(&self.lat.t()))
            .inv()
            .map_err(TbError::Linalg)?;
        let mut k_node = Array1::<f64>::zeros(n_node);
        for n in 1..n_node {
            //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
            let dk = path.row(n).to_owned() - path.slice(s![n - 1, ..]).to_owned();
            let a = k_metric.dot(&dk);
            let dklen: f64 = dk.dot(&a).sqrt();
            if !dklen.is_finite() || dklen <= 0.0 {
                return Err(TbError::Other(
                    "k_path requires distinct consecutive nodes and a finite reciprocal metric"
                        .into(),
                ));
            }
            k_node[[n]] = k_node[[n - 1]] + dklen;
        }
        if !k_node[n_node - 1].is_finite() {
            return Err(TbError::Other("k_path length is not finite".into()));
        }
        let mut node_index: Vec<usize> = vec![0];
        for n in 1..n_node - 1 {
            let frac = k_node[[n]] / k_node[[n_node - 1]];
            // Reserve an interval for each remaining segment, including short
            // segments whose proportional sample count would round to zero.
            let a = ((frac * (nk - 1) as f64).round() as usize)
                .clamp(node_index[n - 1] + 1, nk - (n_node - n));
            node_index.push(a)
        }
        node_index.push(nk - 1);
        let mut k_dist = Array1::<f64>::zeros(nk);
        let mut k_vec = Array2::<f64>::zeros((nk, self.dim_r()));
        //k_vec.slice_mut(s![0,..]).assign(&path.slice(s![0,..]));
        k_vec.row_mut(0).assign(&path.row(0));
        for n in 1..n_node {
            let n_i = node_index[n - 1];
            let n_f = node_index[n];
            let kd_i = k_node[[n - 1]];
            let kd_f = k_node[[n]];
            let k_i = path.row(n - 1);
            let k_f = path.row(n);
            for j in n_i..n_f + 1 {
                let frac: f64 = ((j - n_i) as f64) / ((n_f - n_i) as f64);
                k_dist[[j]] = kd_i + frac * (kd_f - kd_i);
                k_vec
                    .row_mut(j)
                    .assign(&((1.0 - frac) * k_i.to_owned() + frac * k_f.to_owned()));
            }
        }
        Ok((k_vec, k_dist, k_node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k_path_rejects_invalid_sampling_and_coordinates() {
        let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        for path in [
            Array2::zeros((0, 1)),
            array![[0.0]],
            array![[0.0], [0.0]],
            array![[0.0], [f64::NAN]],
        ] {
            assert!(model.k_path(&path, 10).is_err());
        }
        for nk in [0, 1, usize::MAX] {
            assert!(model.k_path(&array![[0.0], [1.0]], nk).is_err());
        }
    }

    #[test]
    fn k_path_keeps_short_segments_and_reciprocal_distance_convention() {
        let model = Model::<false, 1>::tb_model(array![[2.0]], array![[0.0]], None).unwrap();
        let path = array![[0.0], [1e-9], [1.0], [0.0]];
        for nk in [4, 11] {
            let (points, distances, nodes) = model.k_path(&path, nk).unwrap();
            assert_eq!(points.dim(), (nk, 1));
            assert!(points.iter().chain(distances.iter()).all(|x| x.is_finite()));
            for p in path.column(0) {
                assert!(points.column(0).iter().any(|x| x == p));
            }
            assert!(distances.windows(2).into_iter().all(|w| w[1] > w[0]));
            assert_eq!(nodes[nodes.len() - 1], 1.0);
            assert_eq!(points[[nk - 1, 0]], 0.0);
        }
    }
}
