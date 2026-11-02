//! Model-basis operator construction.
//!
//! These methods expose spin and local atomic orbital-angular-momentum
//! operators in the same basis used by the Hamiltonian. They do not depend on
//! eigenstates; band-basis representations are obtained by the caller.

use super::{Model, RMatrixData, SpinDirection};
use crate::error::{Result, TbError};
use ndarray::*;
use num_complex::Complex;

impl<const DIM: usize, R: RMatrixData> Model<true, DIM, R> {
    /// Build the dimensionless spin operator
    /// `S_a / ℏ = σ_a ⊗ I_norb / 2` in the model basis.
    ///
    /// Returns an `(nsta, nsta)` matrix in spin-major order
    /// `(all ↑ orbitals, all ↓ orbitals)`. This is `S_a/ℏ`, not the
    /// dimensionful spin operator. With `v` the velocity operator,
    /// `{S_a / ℏ, v} / 2` is the spin current normalized by `ℏ`; the
    /// conventional dimensionful spin current is
    /// `{S_a, v} / 2 = (ℏ / 2) {S_a / ℏ, v}`. Because `S_a/ℏ` is
    /// dimensionless, the normalized current carries the units of `v`.
    /// No eigenstates or atomic projections are needed.
    ///
    /// ```
    /// use ndarray::{array, Array2};
    /// use num_complex::Complex;
    /// use Rustb::{Model, SpinDirection};
    /// let model = Model::<true, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
    /// let sz: Array2<Complex<f64>> = model.build_spin_matrix(SpinDirection::Z);
    /// assert_eq!(sz.diag(), array![Complex::new(0.5, 0.0), Complex::new(-0.5, 0.0)]);
    /// # Ok::<(), Rustb::TbError>(())
    /// ```
    ///
    /// Spinless models do not have this method:
    ///
    /// ```compile_fail,E0599
    /// use Rustb::{Model, SpinDirection};
    /// fn spinless(model: &Model<false, 2>) {
    ///     model.build_spin_matrix(SpinDirection::Z);
    /// }
    /// ```
    pub fn build_spin_matrix(&self, spin: SpinDirection) -> Array2<Complex<f64>> {
        let norb = self.norb();
        let mut matrix = Array2::<Complex<f64>>::zeros((self.nsta(), self.nsta()));
        let half = Complex::new(0.5, 0.0);
        let i_half = Complex::new(0.0, 0.5);
        for i in 0..norb {
            match spin {
                SpinDirection::X => {
                    matrix[[i, i + norb]] = half;
                    matrix[[i + norb, i]] = half;
                }
                SpinDirection::Y => {
                    matrix[[i, i + norb]] = -i_half;
                    matrix[[i + norb, i]] = i_half;
                }
                SpinDirection::Z => {
                    matrix[[i, i]] = half;
                    matrix[[i + norb, i + norb]] = -half;
                }
            }
        }
        matrix
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    /// 构造局域原子轨道角动量矩阵，数值以 ℏ 为单位，即返回 `L / ℏ`。
    ///
    /// 返回形状始终为 `(3, nsta, nsta)`，第一轴依次是 `Lx, Ly, Lz`，
    /// 对应 [`OrbProj`](crate::atom_struct::OrbProj) 定义的笛卡尔坐标轴，与晶格维数 `DIM` 无关。
    /// 后两轴为模型基底中的 `(bra, ket)`，即
    /// `result[[a, i, j]] = ⟨i|L_a/ℏ|j⟩`，不是能带表象。
    ///
    /// 无自旋时基底按全局轨道编号排列；有自旋时依次为
    /// `(全部轨道 ↑, 全部轨道 ↓)`，每个分量返回 `diag(L_a, L_a)`，
    /// 两个交叉自旋块均为零。这是轨道角动量，不含自旋角动量 `S`。
    /// 若 [`Solve::solve_onek`](crate::solve_ham::Solve::solve_onek) 返回的
    /// 本征矢矩阵为 `C[band, basis]`（每行存一个 ket 的系数），则能带表象为
    /// `C.mapv(|z| z.conj()).dot(&L_a).dot(&C.t())`；不能漏掉左侧共轭。
    ///
    /// 矩阵元由同一原子上轨道的 s/p/d/f 球谐展开（包括杂化轨道）计算，
    /// 不同原子之间置零。这里采用局域原子轨道近似，并未使用实际 Wannier
    /// 波函数的空间积分，也不是 Bloch 态的轨道磁矩或体材料的轨道磁化。
    /// `OrbProj` 不包含径向壳层及局域坐标架信息：调用者应保证同一原子的
    /// 投影构成与模型一致的正交角向基底，不能据此区分如 2p 与 3p 的径向壳层。
    ///
    /// 对不完整的轨道子空间，返回投影后的 `P L_a P / ℏ`；例如仅保留
    /// `(px, py)` 时 `Lx = Ly = 0`，但 `Lz` 非零。这样的截断一般不再满足
    /// 完整角动量代数；完整的单个 l 壳层则满足 `[Lx, Ly] = i Lz` 和
    /// `Lx² + Ly² + Lz² = l(l+1) I`（均指上述无量纲矩阵）。
    ///
    /// 没有原子结构、存在未归属原子的轨道或模型校验失败时返回错误。
    #[inline(always)]
    pub fn orb_angular(&self) -> Result<Array3<Complex<f64>>> {
        if self.atoms.is_empty() {
            return Err(TbError::MissingAtomicStructure);
        }
        self.validate()?;
        // Every orbital must be owned by an atom.  Unowned orbitals (reachable
        // through remove_atoms_only or partial-ownership Atom construction)
        // would silently keep zero angular-momentum matrix elements, which is
        // indistinguishable from a genuine s-orbital zero.
        let owners = self.orbital_owners()?;
        if let Some(unowned) = owners.iter().position(Option::is_none) {
            return Err(TbError::Other(format!(
                "orb_angular requires every orbital to be owned by an atom, \
                 but orbital {unowned} has no owner. \
                 Remove the orbital or attach it to an Atom first."
            )));
        }
        let li = Complex::i() * 1.0;
        let norb = self.norb();
        let mut L = Array3::<Complex<f64>>::zeros((3, self.nsta(), self.nsta()));
        let mut Lz_orig = Array2::<Complex<f64>>::zeros((16, 16));
        Lz_orig
            .slice_mut(s![1..4, 1..4])
            .assign(&Array2::from_diag(&array![-1.0, 0.0, 1.0]).mapv(|x| Complex::new(x, 0.0)));
        Lz_orig.slice_mut(s![4..9, 4..9]).assign(
            &Array2::from_diag(&array![-2.0, -1.0, 0.0, 1.0, 2.0]).mapv(|x| Complex::new(x, 0.0)),
        );
        Lz_orig.slice_mut(s![9..16, 9..16]).assign(
            &Array2::from_diag(&array![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
                .mapv(|x| Complex::new(x, 0.0)),
        );
        let mut Lup_orig = Array2::<Complex<f64>>::zeros((16, 16));
        let mut Ldn_orig = Array2::<Complex<f64>>::zeros((16, 16));
        for l in 0..4 {
            for (m0, m) in (-(l as isize)..=l as isize).enumerate() {
                let i = l * l + m0;
                if m + 1 > l as isize && m - 1 < -(l as isize) {
                    continue;
                } else if m + 1 > l as isize {
                    let l = l as f64;
                    let m = m as f64;
                    Ldn_orig[[i - 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m - 1.0)).sqrt(), 0.0);
                } else if m - 1 < -(l as isize) {
                    let l = l as f64;
                    let m = m as f64;
                    Lup_orig[[i + 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m + 1.0)).sqrt(), 0.0);
                } else {
                    let l = l as f64;
                    let m = m as f64;
                    Ldn_orig[[i - 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m - 1.0)).sqrt(), 0.0);
                    Lup_orig[[i + 1, i]] =
                        Complex::new((l * (l + 1.0) - m * (m + 1.0)).sqrt(), 0.0);
                }
            }
        }
        // In units of hbar, Lx=(L+ + L-)/2 and
        // Ly=(L+ - L-)/(2i).  The one-half is essential: omitting it makes
        // rotations generated about x or y advance by twice the requested
        // angle, while Lz remains normally scaled.
        let Lx_orig = Complex::new(0.5, 0.0) * (&Lup_orig + &Ldn_orig);
        let Ly_orig = (-li * 0.5) * (&Lup_orig - &Ldn_orig);
        for atom0 in self.atoms.iter() {
            for &orbital_i in atom0.orbitals() {
                let i = orbital_i.index();
                let proj_i: Array1<Complex<f64>> = self.orb_projection[i]
                    .to_quantum_number()?
                    .mapv(|x: Complex<f64>| x.conj());
                for &orbital_j in atom0.orbitals() {
                    let j = orbital_j.index();
                    let proj_j = self.orb_projection[j].to_quantum_number()?;
                    L[[0, i, j]] = proj_i.dot(&Lx_orig.dot(&proj_j));
                    L[[1, i, j]] = proj_i.dot(&Ly_orig.dot(&proj_j));
                    L[[2, i, j]] = proj_i.dot(&Lz_orig.dot(&proj_j));
                    if SPIN {
                        // The model stores all spin-up orbitals before all
                        // spin-down orbitals; orbital L acts identically on both.
                        for axis in 0..3 {
                            L[[axis, i + norb, j + norb]] = L[[axis, i, j]];
                        }
                    }
                }
            }
        }
        Ok(L)
    }
}

#[cfg(test)]
mod angular_momentum_tests {
    use super::*;
    use crate::AtomType;
    use crate::atom_struct::{Atom, OrbProj, OrbitalId};
    use crate::{HasRMatrix, NoRMatrix};

    fn atomic<const SPIN: bool, const DIM: usize>(projections: &[OrbProj]) -> Model<SPIN, DIM> {
        let mut model = Model::tb_model(
            Array2::eye(DIM),
            Array2::zeros((projections.len(), DIM)),
            Some(vec![Atom::with_orbitals(
                Array1::zeros(DIM),
                AtomType::C,
                (0..projections.len()).map(OrbitalId::new),
            )]),
        )
        .unwrap();
        model.set_projection(&projections.to_vec());
        model
    }

    fn assert_matrix_close(
        actual: ArrayView2<'_, Complex<f64>>,
        expected: ArrayView2<'_, Complex<f64>>,
    ) {
        assert_eq!(actual.dim(), expected.dim());
        assert!(
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a - b).norm() < 1e-12)
        );
    }

    #[test]
    fn spin_matrices_have_spin_major_half_spin_eigenstates() {
        fn check<R: RMatrixData>() {
            let model =
                Model::<true, 1, R>::tb_model(array![[1.0]], array![[0.1], [0.7]], None).unwrap();
            let zero = Complex::new(0.0, 0.0);
            let one = Complex::new(1.0, 0.0);
            let c = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
            let ic = Complex::<f64>::i() * c;
            // Independent analytic spinors: |±x>, |±y>, and |±z>.
            for (direction, positive, negative) in [
                (SpinDirection::X, [c, c], [c, -c]),
                (SpinDirection::Y, [c, ic], [c, -ic]),
                (SpinDirection::Z, [one, zero], [zero, one]),
            ] {
                let spin = model.build_spin_matrix(direction);
                assert_eq!(spin.dim(), (4, 4));
                for orbital in 0..2 {
                    for (eigenvalue, spinor) in [(0.5, positive), (-0.5, negative)] {
                        let mut ket = Array1::zeros(4);
                        ket[orbital] = spinor[0];
                        ket[orbital + 2] = spinor[1];
                        let residual = spin.dot(&ket) - ket.mapv(|value| value * eigenvalue);
                        assert!(
                            residual.iter().all(|value| value.norm() < 1e-14),
                            "{direction:?}, orbital {orbital}, eigenvalue {eigenvalue}: {residual:?}"
                        );
                    }
                }
            }
        }
        check::<NoRMatrix>();
        check::<HasRMatrix>();
    }

    #[test]
    fn spin_matrices_obey_hermiticity_commutators_and_casimir() {
        fn check<R: RMatrixData>() {
            let model =
                Model::<true, 3, R>::tb_model(Array2::eye(3), Array2::zeros((3, 3)), None).unwrap();
            let spin = [SpinDirection::X, SpinDirection::Y, SpinDirection::Z]
                .map(|direction| model.build_spin_matrix(direction));
            let mut casimir = Array2::<Complex<f64>>::zeros((6, 6));
            for d in 0..3 {
                let (a, b, c) = (&spin[d], &spin[(d + 1) % 3], &spin[(d + 2) % 3]);
                assert_matrix_close(a.view(), a.t().mapv(|value| value.conj()).view());
                assert_matrix_close((a.dot(b) - b.dot(a)).view(), (Complex::i() * c).view());
                casimir += &a.dot(a);
            }
            let expected = Array2::<Complex<f64>>::eye(6) * Complex::new(0.75, 0.0);
            assert_matrix_close(casimir.view(), expected.view());
        }
        check::<NoRMatrix>();
        check::<HasRMatrix>();
    }

    // Independent Cartesian oracle: L = -i r × grad acting on (px, py, pz).
    fn p_generators() -> Array3<Complex<f64>> {
        array![
            [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
        .mapv(|value| Complex::new(0.0, value))
    }

    fn check_p<const DIM: usize>() {
        let angular = atomic::<false, DIM>(&[OrbProj::px, OrbProj::py, OrbProj::pz])
            .orb_angular()
            .unwrap();
        let expected = p_generators();
        assert_eq!(angular.dim(), (3, 3, 3));
        for d in 0..3 {
            assert_matrix_close(
                angular.index_axis(Axis(0), d),
                expected.index_axis(Axis(0), d),
            );
        }
    }

    #[test]
    fn cartesian_p_generators_do_not_depend_on_crystal_dimension() {
        check_p::<1>();
        check_p::<2>();
        check_p::<3>();
    }

    #[test]
    fn complete_shells_obey_hermiticity_commutators_and_casimir() {
        let shells = [
            vec![OrbProj::s],
            vec![OrbProj::px, OrbProj::py, OrbProj::pz],
            vec![
                OrbProj::dxy,
                OrbProj::dyz,
                OrbProj::dxz,
                OrbProj::dz2,
                OrbProj::dx2y2,
            ],
            vec![
                OrbProj::fz3,
                OrbProj::fxz2,
                OrbProj::fyz2,
                OrbProj::fzx2y2,
                OrbProj::fxyz,
                OrbProj::fxx23y2,
                OrbProj::fy3x2y2,
            ],
        ];
        for (l, projections) in shells.iter().enumerate() {
            let angular = atomic::<false, 3>(projections).orb_angular().unwrap();
            let mut casimir = Array2::<Complex<f64>>::zeros((projections.len(), projections.len()));
            for d in 0..3 {
                let a = angular.index_axis(Axis(0), d);
                let b = angular.index_axis(Axis(0), (d + 1) % 3);
                let c = angular.index_axis(Axis(0), (d + 2) % 3);
                assert_matrix_close(a, a.t().mapv(|value| value.conj()).view());
                assert_matrix_close((&a.dot(&b) - &b.dot(&a)).view(), (Complex::i() * &c).view());
                casimir += &a.dot(&a);
            }
            let expected = Array2::<Complex<f64>>::eye(projections.len())
                * Complex::new((l * (l + 1)) as f64, 0.0);
            assert_matrix_close(casimir.view(), expected.view());
        }
    }

    #[test]
    fn spinful_operator_has_two_identical_blocks_and_no_spin_flip() {
        let model = atomic::<true, 3>(&[OrbProj::px, OrbProj::py, OrbProj::pz]);
        let angular = model.orb_angular().unwrap();
        assert_eq!(angular.dim(), (3, model.nsta(), model.nsta()));
        let p = p_generators();
        for d in 0..3 {
            assert_matrix_close(angular.slice(s![d, ..3, ..3]), p.index_axis(Axis(0), d));
            assert_matrix_close(angular.slice(s![d, 3.., 3..]), p.index_axis(Axis(0), d));
            assert!(
                angular
                    .slice(s![d, ..3, 3..])
                    .iter()
                    .all(|z| z.norm() == 0.0)
            );
            assert!(
                angular
                    .slice(s![d, 3.., ..3])
                    .iter()
                    .all(|z| z.norm() == 0.0)
            );
        }
    }

    #[test]
    fn atom_ownership_and_projection_order_determine_matrix_elements() {
        let groups = [[4, 0, 2], [3, 5, 1]]; // (px, py, pz) for each atom
        let mut orbitals = Array2::zeros((6, 3));
        for &i in &groups[1] {
            orbitals[[i, 0]] = 0.4;
        }
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            orbitals,
            Some(vec![
                Atom::with_orbitals(
                    array![0.0, 0.0, 0.0],
                    AtomType::C,
                    groups[0].map(OrbitalId::new),
                ),
                Atom::with_orbitals(
                    array![0.4, 0.0, 0.0],
                    AtomType::C,
                    groups[1].map(OrbitalId::new),
                ),
            ]),
        )
        .unwrap();
        model.set_projection(&vec![
            OrbProj::py,
            OrbProj::pz,
            OrbProj::pz,
            OrbProj::px,
            OrbProj::px,
            OrbProj::py,
        ]);
        let angular = model.orb_angular().unwrap();
        let mut expected = Array3::zeros((3, 6, 6));
        let p = p_generators();
        for group in &groups {
            for d in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        expected[[d, group[i], group[j]]] = p[[d, i, j]];
                    }
                }
            }
        }
        for d in 0..3 {
            assert_matrix_close(
                angular.index_axis(Axis(0), d),
                expected.index_axis(Axis(0), d),
            );
        }
    }

    #[test]
    fn sp3_hybrids_and_incomplete_p_basis_are_projected_operators() {
        let angular = atomic::<false, 3>(&[
            OrbProj::sp3_1,
            OrbProj::sp3_2,
            OrbProj::sp3_3,
            OrbProj::sp3_4,
        ])
        .orb_angular()
        .unwrap();
        let transform = array![
            [1., 1., 1., 1.],
            [1., 1., -1., -1.],
            [1., -1., 1., -1.],
            [1., -1., -1., 1.]
        ]
        .mapv(|value| Complex::new(0.5 * value, 0.0));
        let p = p_generators();
        let partial = atomic::<false, 3>(&[OrbProj::px, OrbProj::py])
            .orb_angular()
            .unwrap();
        for d in 0..3 {
            let mut sp = Array2::zeros((4, 4));
            sp.slice_mut(s![1.., 1..]).assign(&p.index_axis(Axis(0), d));
            let expected = transform.dot(&sp.dot(&transform.t()));
            assert_matrix_close(angular.index_axis(Axis(0), d), expected.view());
            assert_matrix_close(partial.index_axis(Axis(0), d), p.slice(s![d, ..2, ..2]));
        }
        // The truncated (px,py) subspace is not closed under Lx/Ly. It must
        // retain P L P, rather than being forced to satisfy the full algebra.
        assert!(
            partial
                .slice(s![0..2, .., ..])
                .iter()
                .all(|z| z.norm() < 1e-12)
        );
        assert!((partial[[2, 0, 1]] + Complex::i()).norm() < 1e-12);
    }
}
