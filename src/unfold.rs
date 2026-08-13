use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::output::*;
use crate::solve_ham::*;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::{Complex, Complex64};
use rayon::prelude::*;
use std::f64::consts::PI;

/// Fractional-coordinate tolerance used to identify equivalent orbital and
/// atom representatives during unfolding.
///
/// Previously `5e-2`, which merged genuinely distinct primitive orbitals whose
/// centers were closer than 0.05 (e.g. `0.0` and `0.03`) into one
/// representative and made unfold hard-fail with `InvalidAtomConfiguration`.
/// `1e-3` is far above the floating-point noise of the fold (`~1e-15`) and far
/// below the typical distance between distinct Wannier centers.
const REPRESENTATIVE_POSITION_TOLERANCE: f64 = 1e-3;
pub trait Unfold {
    //! Band unfolding algorithm. Computes the unfolded band structure, and can be
    //! used to study alloys, supercells, impurities, defects, and charge density
    //! waves projected onto the primitive cell.
    /// The algorithm follows PRL 104, 216401 (2010).
    /// Representative centers are matched within a fixed fractional-coordinate
    /// tolerance of `1e-3`.
    ///
    /// First, define the supercell Brillouin-zone Hamiltonian $H_{\\bm K}$ and its
    /// Green's function $$G(\og,\bm K)=(\og+i\eta-H_{\bm K})^{-1}$$
    ///
    /// where $H_{\bm K}$ is the supercell Hamiltonian. Its eigenvalues and
    /// eigenvectors are $\ve_{N\bm K}$ and $\bra{\psi_{N\bm K}}$.
    ///
    /// We can then write the Green's function in the eigenbasis as
    /// $$G(\og,\bm K)=\sum_{N}\f{\dyad{\psi_{N\bm K}}}{\og+i\eta-\ve_{N\bm K}}$$
    ///
    /// Using the spectral theorem, $A(\og,\bm K)=-\f{1}{\pi}\Im G(\og,\bm K)$.
    /// Taking the trace of $A$ gives the supercell spectrum.
    ///
    /// However, we want the primitive-cell spectrum, so we need the primitive-cell
    /// basis $\ket{n\bm k}$.
    ///
    /// The unfolded spectral function is
    /// $$A_{nn}(\og,\bm k)=\sum_{N\bm K}\lt\\vert \braket{n\bm k}{\psi_{N\bm K}}\rt\\vert^2 A_{NN}(\og,\bm K)$$
    ///
    ///Next we compute $\braket{n\bm k}{\psi_{N\bm K}}$.
    ///
    ///First, we have $$ \lt\\{
    ///\\begin{aligned}
    ///\ket{N\bm K}&=\f{1}{\sqrt{V}}\sum_{\bm R}e^{-i\bm K\cdot(\bm R+\bm\tau_N)}\ket{N\bm R}\\\\
    ///\ket{n\bm k}&=\f{1}{\sqrt{v}}\sum_{\bm r}e^{-i\bm k\cdot(\bm r+\bm\tau_n)}\ket{n\bm r}\\\\
    ///\\end{aligned}\rt\.$$
    ///
    ///Then, consider a supercell mapping relating the primitive cell $a$ to the
    ///supercell $A$ via $A=Ua$, where $A$ and $a$ are the lattice vectors. From
    ///the relation $b a^T=(2\pi)I$ and $B A^T=(2\pi)I$, we immediately obtain
    ///$b=BU^T$. Here $B$ and $b$ are the reciprocal lattice vectors of the
    ///supercell and primitive cell, respectively.
    ///
    /// $$
    /// \begin{aligned}
    /// \bra{n\bm k}\ket{N\bm K}&=\sum_{J\bm R}\braket{n\bm k}{J\bm R}\braket{J\bm R}{J\bm K}\braket{J\bm K}{N\bm K}\\\\
    /// &=\sum_{J\bm R}\braket{n\bm k}{n'(J)\bm R+\bm r(J)}\braket{J\bm R}{J\bm k}\braket{J\bm K}{N\bm K}\\\\
    /// &=\sqrt{\frac{1}{V}}\sum_{J\bm R}e^{i(\bm K-\bm k)\cdot\bm R-i\bm k\cdot\bm r(J)}\delta_{n,n'(J)}\braket{J\bm k}{N\bm K}.
    /// \end{aligned}
    /// $$
    ///
    /// Clearly, $r(J)$ and $n'(J)$ can be computed. Using $U$, the correspondence
    /// between folded and unfolded states is obtained.
    fn unfold(
        &self,
        U: &Array2<f64>,
        path: &Array2<f64>,
        nk: usize,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        eta: f64,
        precision: f64,
    ) -> Result<Array2<f64>>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Unfold for Model<SPIN, DIM, R> {
    fn unfold(
        &self,
        U: &Array2<f64>,
        path: &Array2<f64>,
        nk: usize,
        E_min: f64,
        E_max: f64,
        E_n: usize,
        eta: f64,
        precision: f64,
    ) -> Result<Array2<f64>> {
        self.validate()?;
        if !self.atoms.is_empty() && self.orbital_owners()?.iter().any(Option::is_none) {
            return Err(TbError::InvalidModelInvariant {
                invariant: "unfold_orbital_ownership",
                message: "band unfolding requires every orbital to belong to an atom".to_string(),
            });
        }
        let li: Complex<f64> = Complex::i();
        let E = Array1::<f64>::linspace(E_min, E_max, E_n);
        let mut A0 = Array2::<f64>::zeros((E_n, nk));
        let inv_U = U.inv().map_err(TbError::Linalg)?;
        let unfold_lat = &inv_U.dot(&self.lat);
        let V = self.lat.det().map_err(TbError::Linalg)?;
        let unfold_V = unfold_lat.det().map_err(TbError::Linalg)?;
        let U_det = U.det().map_err(TbError::Linalg)?;
        if !U_det.is_finite() || U_det <= 1.0 {
            // NaN must not reach the rounding/modulo below: NaN passes the
            // arithmetic guards and NaN.round() as usize saturates to 0,
            // which would panic in `norb % 0`.
            return Err(TbError::InvalidSupercellDet { det: U_det });
        }
        let cell_count = U_det.round();
        if (U_det - cell_count).abs() > precision {
            return Err(TbError::InvalidSupercellDet { det: U_det });
        }
        let cell_count = cell_count as usize;
        if cell_count == 0 || self.norb() % cell_count != 0 || self.natom() % cell_count != 0 {
            return Err(TbError::InvalidAtomConfiguration);
        }
        //我们先根据path计算一下k点
        let (kvec, kdist, knode) = {
            let n_node: usize = path.len_of(Axis(0));
            let k_metric = (&unfold_lat.dot(&unfold_lat.t())).inv().unwrap();
            let mut k_node = Array1::<f64>::zeros(n_node);
            for n in 1..n_node {
                //let dk=path.slice(s![n,..]).to_owned()-path.slice(s![n-1,..]).to_owned();
                let dk = path.row(n).to_owned() - path.slice(s![n - 1, ..]).to_owned();
                let a = k_metric.dot(&dk);
                let dklen = dk.dot(&a).sqrt();
                k_node[[n]] = k_node[[n - 1]] + dklen;
            }
            let mut node_index: Vec<usize> = vec![0];
            for n in 1..n_node - 1 {
                let frac = k_node[[n]] / k_node[[n_node - 1]];
                let a = (frac * ((nk - 1) as f64).round()) as usize;
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
            (k_vec, k_dist, k_node)
        };

        //我们先unfold一下k点
        let fold_k = &kvec.dot(&U.t()); // fold_k 是要求解的本征值和本征态
        let (eval, evec) = self.solve_all_parallel(&fold_k); //开始求解本征态和本征值
        let eval = eval.mapv(|x| Complex::new(x, 0.0));
        let mut G = Array3::<Complex<f64>>::zeros((E_n, nk, self.nsta()));
        //Zip::from(G.outer_iter_mut()).and(E.view()).par_for_each(|mut g,og| {g.assign(&(Complex::new(1.0,0.0)/(*og+li*eta-&eval)));});
        for (e, og) in E.iter().enumerate() {
            for (k, vec) in eval.outer_iter().enumerate() {
                for i in 0..self.nsta() {
                    G[[e, k, i]] = 1.0 / (*og + eta * li - vec[[i]]);
                }
            }
        }
        let mut G = G.mapv(|x| -x.im() / PI);
        G.swap_axes(0, 1);
        //接下来我们计算原胞的原子位置和轨道位置
        let mut unit_orb = Array2::<f64>::zeros((0, self.dim_r()));
        let unfold_orb = &self.orb.dot(U).map(|x| {
            if (x.fract() - 1.0).abs() < precision || x.fract().abs() < precision {
                0.0
            } else if x.fract() < 0.0 {
                x.fract() + 1.0
            } else {
                x.fract()
            }
        });
        let mut match_orb_list = Array1::<usize>::zeros(self.norb());
        if self.atoms.is_empty() {
            // Orbital-only models are first-class: derive primitive-orbital
            // representatives directly from folded Wannier centers.
            for (orbital, folded) in unfold_orb.outer_iter().enumerate() {
                let representative = unit_orb.outer_iter().position(|candidate| {
                    (&folded - &candidate).norm() < REPRESENTATIVE_POSITION_TOLERANCE
                });
                match representative {
                    Some(representative) => match_orb_list[orbital] = representative,
                    None => {
                        unit_orb.push_row(folded);
                        match_orb_list[orbital] = unit_orb.nrows() - 1;
                    }
                }
            }
            if unit_orb.nrows() != self.norb() / cell_count {
                return Err(TbError::InvalidAtomConfiguration);
            }
        } else {
            let mut unit_atom = Array2::<f64>::zeros((0, self.dim_r()));
            let atom_position = self.atom_position();
            let unfold_atom = &atom_position.dot(U).map(|x| {
                if (x.fract() - 1.0).abs() < precision || x.fract().abs() < precision {
                    0.0
                } else if x.fract() < 0.0 {
                    x.fract() + 1.0
                } else {
                    x.fract()
                }
            });
            let mut unit_atom_orbitals = Vec::<Vec<usize>>::new();
            let mut unit_atom_types = Vec::new();
            for (atom_index, folded_atom) in unfold_atom.outer_iter().enumerate() {
                let representative =
                    unit_atom
                        .outer_iter()
                        .enumerate()
                        .position(|(candidate_index, candidate)| {
                            (&folded_atom - &candidate).norm() < REPRESENTATIVE_POSITION_TOLERANCE
                                && unit_atom_types[candidate_index]
                                    == self.atoms[atom_index].atom_type()
                        });
                if let Some(representative) = representative {
                    if unit_atom_orbitals[representative].len() != self.atoms[atom_index].norb() {
                        return Err(TbError::InvalidAtomConfiguration);
                    }
                    for (&orbital, &unit_orbital) in self.atoms[atom_index]
                        .orbitals()
                        .iter()
                        .zip(&unit_atom_orbitals[representative])
                    {
                        match_orb_list[orbital.index()] = unit_orbital;
                    }
                } else {
                    unit_atom.push_row(folded_atom);
                    unit_atom_types.push(self.atoms[atom_index].atom_type());
                    let mut representative_orbitals =
                        Vec::with_capacity(self.atoms[atom_index].norb());
                    for &orbital in self.atoms[atom_index].orbitals() {
                        unit_orb.push_row(unfold_orb.row(orbital.index()));
                        let unit_orbital = unit_orb.nrows() - 1;
                        match_orb_list[orbital.index()] = unit_orbital;
                        representative_orbitals.push(unit_orbital);
                    }
                    unit_atom_orbitals.push(representative_orbitals);
                }
            }
            if unit_atom.nrows() != self.natom() / cell_count {
                return Err(TbError::InvalidAtomConfiguration);
            }
        }
        //好了, 接下来让我们计算权重
        let mut weight = Array2::<Complex<f64>>::zeros((nk, self.nsta()));
        let mut B = Array3::<f64>::zeros((nk, E_n, unit_orb.nrows()));
        if SPIN {
            for k0 in 0..nk {
                let mut r = &self.orb.dot(&fold_k.row(k0)) - &self.orb.dot(U).dot(&kvec.row(k0));
                let r0 = r.clone();
                r.append(Axis(0), r0.view()).unwrap();
                weight
                    .slice_mut(s![k0, ..])
                    .assign(&(-PI * 2.0 * r).mapv(|x| Complex::new(0.0, x).exp()));
            }
            let A = match_orb_list.clone();
            match_orb_list.append(Axis(0), A.view()).unwrap();
            Zip::from(B.outer_iter_mut())
                .and(G.outer_iter())
                .and(weight.outer_iter())
                .and(evec.outer_iter())
                .par_for_each(|mut b, g, w, vec| {
                    for (i0, mut b0) in b.axis_iter_mut(Axis(1)).enumerate() {
                        let mut A = Array1::<Complex<f64>>::zeros(self.nsta());
                        A = match_orb_list
                            .iter()
                            .enumerate()
                            .zip(w.iter().zip(vec.axis_iter(Axis(1))))
                            .fold(A.clone(), |mut acc, ((io, orb_index), (w0, vec0))| {
                                if *orb_index == i0 {
                                    acc + *w0 * &vec0
                                } else {
                                    acc
                                }
                            });
                        let A = A.mapv(|x| x.norm_sqr());
                        b0.assign(&g.dot(&A));
                    }
                });
        } else {
            for k0 in 0..nk {
                let weight1 = (-PI
                    * 2.0
                    * (&self.orb.dot(&fold_k.row(k0)) - &self.orb.dot(U).dot(&kvec.row(k0))))
                    .mapv(|x| Complex::new(0.0, x).exp());
                weight.slice_mut(s![k0, ..]).assign(&weight1);
            }
            Zip::from(B.outer_iter_mut())
                .and(G.outer_iter())
                .and(weight.outer_iter())
                .and(evec.outer_iter())
                .par_for_each(|mut b, g, w, vec| {
                    for (i0, mut b0) in b.axis_iter_mut(Axis(1)).enumerate() {
                        let mut A = Array1::<Complex<f64>>::zeros(self.nsta());
                        A = match_orb_list
                            .iter()
                            .enumerate()
                            .zip(w.iter().zip(vec.axis_iter(Axis(1))))
                            .fold(A.clone(), |mut acc, ((io, orb_index), (w0, vec0))| {
                                if *orb_index == i0 {
                                    acc + *w0 * &vec0
                                } else {
                                    acc
                                }
                            });
                        let A = A.mapv(|x| x.norm_sqr());
                        b0.assign(&g.dot(&A));
                    }
                });
        }
        A0 = B.sum_axis(Axis(2));
        Ok(A0.reversed_axes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpinDirection;
    use crate::draw_heatmap;
    use crate::kpath::*;
    use gnuplot::{
        AutoOption, AxesCommon, Color, Figure, Fix, Font, LineStyle, Major, PointSymbol, Rotate,
        Solid, TextOffset,
    };
    use ndarray::prelude::*;
    use ndarray::*;
    use num_complex::Complex;
    use std::f64::consts::PI;
    use std::time::{Duration, Instant};

    #[test]
    fn unfold_rejects_nan_determinant() {
        // Regression: a NaN determinant used to pass both arithmetic guards
        // (NaN <= 1.0 is false, (NaN - 0).abs() > prec is false), then
        // NaN.round() as usize saturated to 0 and `norb % 0` panicked.
        let model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let result = model.unfold(
            &array![[f64::NAN, 0.0], [0.0, 1.0]],
            &array![[0.0, 0.0], [0.5, 0.0]],
            2,
            -1.0,
            1.0,
            2,
            1e-2,
            1e-3,
        );
        assert!(matches!(result, Err(TbError::InvalidSupercellDet { .. })));
    }

    #[test]
    fn unfold_distinguishes_close_primitive_orbitals() {
        // Regression: with the historical 5e-2 representative tolerance,
        // primitive orbitals at 0.0 and 0.03 both matched the (0,0)
        // representative, unit_orb got 1 row instead of norb/cell_count = 2,
        // and unfold returned Err(InvalidAtomConfiguration) for a valid model.
        let model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0], [0.03, 0.0]], None)
                .unwrap();
        let supercell = model.make_supercell(&array![[2.0, 0.0], [0.0, 1.0]]).unwrap();
        let result = supercell.unfold(
            &array![[2.0, 0.0], [0.0, 1.0]],
            &array![[0.0, 0.0], [0.5, 0.0]],
            2,
            -1.0,
            1.0,
            2,
            1e-2,
            1e-3,
        );
        assert!(result.is_ok(), "unfold failed: {:?}", result.err());
    }

    #[test]
    fn unfold_rejects_nondivisible_orbital_count() {
        let model = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
            None,
        )
        .unwrap();
        let result = model.unfold(
            &array![[2.0, 0.0], [0.0, 1.0]],
            &array![[0.0, 0.0], [0.5, 0.0]],
            2,
            -1.0,
            1.0,
            2,
            1e-2,
            1e-3,
        );
        assert!(matches!(result, Err(TbError::InvalidAtomConfiguration)));
    }

    #[test]
    fn unfold_test() {
        use std::fs::create_dir_all;
        let li: Complex<f64> = 1.0 * Complex::i();
        let t1 = 1.0 + 0.0 * li;
        let t2 = 0.1 + 0.0 * li;
        let norb: usize = 2;
        let lat = arr2(&[[3.0_f64.sqrt(), -1.0], [3.0_f64.sqrt(), 1.0]]);
        let orb = arr2(&[[0., 0.], [1.0 / 3.0, 0.0], [0.0, 1.0 / 3.0]]);
        let mut model = Model::<true, 2>::tb_model(lat, orb, None).unwrap();
        //最近邻hopping
        model.add_hop(t1, 0, 1, &array![0, 0], None);
        model.add_hop(t1, 2, 0, &array![0, 0], None);
        model.add_hop(t1, 1, 2, &array![0, 0], None);
        model.add_hop(t1, 0, 2, &array![0, -1], None);
        model.add_hop(t1, 0, 1, &array![-1, 0], None);
        model.add_hop(t1, 2, 1, &array![-1, 1], None);

        let nk: usize = 101;
        let path = array![[0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0], [0.5, 0.], [0.0, 0.0]];
        let label = vec!["G", "K", "M", "G"];
        let (kvec, kdist, knode) = model.k_path(&path, nk).unwrap();
        let U = array![[2.0, 0.0], [0.0, 2.0]];

        let start = Instant::now(); // 开始计时
        let super_model = model.make_supercell(&U).unwrap();
        let end = Instant::now(); // 结束计时
        let duration = end.duration_since(start); // 计算执行时间
        println!("make_supercell took {} seconds", duration.as_secs_f64()); // 输出执行时间
        let A = super_model
            .unfold(&U, &path, nk, -3.0, 5.0, nk, 1e-2, 1e-3)
            .unwrap();
        let name = "./tests/unfold_test/";
        create_dir_all(&name).expect("can't creat the file");
        draw_heatmap(&A.reversed_axes(), "./tests/unfold_test/unfold_band.pdf");
        super_model.show_band(&path, &label, nk, name).unwrap();
    }
}
