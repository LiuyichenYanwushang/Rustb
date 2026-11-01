//! Core implementation of tight-binding model operations and Hamiltonian construction.

//!
//! # Examples
//!
//! Build a model, inspect the arrays that define it, validate the invariants
//! every solver relies on, and round-trip it through the serde wire format.
//!
//! ```rust
//! use ndarray::{arr1, arr2, array};
//! use Rustb::*;
//!
//! # fn main() -> Result<()> {
//! let lat = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
//! let orb = arr2(&[[0.0, 0.0], [0.5, 0.5]]);
//! let mut model = Model::<false, 2>::tb_model(lat, orb, None)?;
//! model.add_hop(-1.0, 0, 1, &array![0, 0], None);
//! model.set_onsite(&arr1(&[0.2, -0.2]), None);
//!
//! // Structural invariants checked by the solvers and response entry points.
//! model.validate()?;
//! assert_eq!(model.dim_r(), 2);
//! assert_eq!((model.norb(), model.nsta()), (2, 2));
//! assert_eq!(model.hamR.ncols(), 2);
//!
//! // Reciprocal vectors are rows: B = 2*pi*(A^T)^-1.
//! let reciprocal = model.rec_lat()?;
//! assert_eq!(reciprocal.dim(), (2, 2));
//!
//! // The serde wire format is validated on load: shape mismatches,
//! // non-finite data and unknown fields are rejected instead of trusted.
//! let encoded = toml::to_string(&model).unwrap();
//! let restored: Model<false, 2> = toml::from_str(&encoded).unwrap();
//! assert_eq!(restored.ham, model.ham);
//!
//! // Gauge selects the Bloch-basis phase convention, not the physics.
//! let k = arr1(&[0.25, 0.125]);
//! let h_atom = model.gen_ham(&k, Gauge::Atom);
//! let h_lattice = model.gen_ham(&k, Gauge::Lattice);
//! assert_eq!(h_atom.dim(), h_lattice.dim());
//! # Ok(())
//! # }
//! ```
//!
//!

pub mod types;
pub mod validate;
pub mod rmatrix;
pub mod access;
pub mod model_build;
pub mod model_physics;
pub mod model_utils;

pub use self::types::{Dimension, Gauge, SpinDirection};
pub use self::rmatrix::{HasRMatrix,NoRMatrix,RMatrixData};
pub use self::access::{AtomView, OrbitalRef};
pub use crate::model_utils::{find_R, remove_col, remove_row};

use crate::atom_struct::{Atom, AtomId, AtomWire, OrbProj, OrbitalId, atoms_from_wire};
use crate::error::{Result, TbError};
use ndarray::*;
use ndarray_linalg::Inverse;
use num_complex::Complex;
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};


// ── RMatrix type-level tag ──────────────────────────────────────────────────


// ── Model struct ────────────────────────────────────────────────────────────

/// Maximum allowed distance (fractional coordinates, modulo a lattice vector)
/// between an orbital and its parent atom's position.
///
/// Enforced by [`Model::validate`].  Kept well below 1/2 so supercell image
/// folding is unambiguous, and below the typical nearest-neighbour distance so
/// bond-centered orbitals remain representable.
///
/// # Known limitation
///
/// The check is component-wise on the CURRENT lattice basis and is therefore
/// not invariant under general `GL(d, Z)` basis changes: a model valid in one
/// integer basis may be rejected after an arbitrary integer basis transform.
/// `make_supercell` / `cut_*` re-validate their outputs, so such transforms
/// fail loudly rather than silently.
pub const ORBITAL_ATOM_POSITION_TOLERANCE: f64 = 0.1;

/// Tight-binding model structure.
///
/// Const generic `SPIN`: spinless (false, default) / spinful (true).
/// Const generic `DIM`: spatial dimension 1/2/3 (default 3).
/// Type parameter `R`: [`HasRMatrix`] or [`NoRMatrix`] (default).
#[derive(Clone, Debug)]
pub struct Model<const SPIN: bool = false, const DIM: usize = 3, R: RMatrixData = NoRMatrix> {
    pub lat: Array2<f64>,
    pub orb: Array2<f64>,
    pub orb_projection: Vec<OrbProj>,
    pub atoms: Vec<Atom>,
    pub ham: Array3<Complex<f64>>,
    pub hamR: Array2<isize>,
    pub rmatrix: R,
}


// Manual Serialize
impl<const SPIN: bool, const DIM: usize, R: RMatrixData + Serialize> Serialize
    for Model<SPIN, DIM, R>
{
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let n_fields = if R::HAS_RMATRIX { 9 } else { 8 };
        let mut s = serializer.serialize_struct("Model", n_fields)?;
        s.serialize_field("dim_r", &DIM)?;
        s.serialize_field("spin", &SPIN)?;
        s.serialize_field("lat", &self.lat)?;
        s.serialize_field("orb", &self.orb)?;
        s.serialize_field("orb_projection", &self.orb_projection)?;
        s.serialize_field("atoms", &self.atoms)?;
        s.serialize_field("ham", &self.ham)?;
        s.serialize_field("hamR", &self.hamR)?;
        if R::HAS_RMATRIX {
            s.serialize_field("rmatrix", &self.rmatrix)?;
        }
        s.end()
    }
}

// Keep the on-disk fields independent of the requested const generics/storage.
// Serde rejects unknown and duplicate fields before model validation.
#[derive(Deserialize)]
#[serde(rename = "Model", deny_unknown_fields)]
struct ModelWire {
    dim_r: usize,
    spin: bool,
    lat: Array2<f64>,
    orb: Array2<f64>,
    orb_projection: Vec<OrbProj>,
    atoms: Vec<AtomWire>,
    ham: Array3<Complex<f64>>,
    hamR: Array2<isize>,
    #[serde(default, deserialize_with = "deserialize_rmatrix")]
    rmatrix: Option<HasRMatrix>,
}

// A missing matrix is allowed for NoRMatrix; a present value must still be an
// array (including when the caller will discard it), not a null placeholder.
fn deserialize_rmatrix<'de, De: Deserializer<'de>>(
    deserializer: De,
) -> std::result::Result<Option<HasRMatrix>, De::Error> {
    HasRMatrix::deserialize(deserializer).map(Some)
}

impl ModelWire {
    fn into_model<const SPIN: bool, const DIM: usize, R: RMatrixData, E: de::Error>(
        self,
        rmatrix: R,
    ) -> std::result::Result<Model<SPIN, DIM, R>, E> {
        if self.spin != SPIN {
            return Err(E::custom(format!(
                "spin mismatch: file has spin={}, but Model<{}> was requested",
                self.spin, SPIN
            )));
        }
        if self.dim_r != DIM {
            return Err(E::custom(format!(
                "dimension mismatch: file has dim_r={}, but Model<DIM={}> was requested",
                self.dim_r, DIM
            )));
        }
        let model = Model {
            lat: self.lat,
            orb: self.orb,
            orb_projection: self.orb_projection,
            atoms: atoms_from_wire(self.atoms).map_err(E::custom)?,
            ham: self.ham,
            hamR: self.hamR,
            rmatrix,
        };
        model.validate().map_err(E::custom)?;
        Ok(model)
    }
}

impl<'de, const SPIN: bool, const DIM: usize> Deserialize<'de> for Model<SPIN, DIM, NoRMatrix> {
    fn deserialize<De: Deserializer<'de>>(
        deserializer: De,
    ) -> std::result::Result<Self, De::Error> {
        ModelWire::deserialize(deserializer)?.into_model(NoRMatrix)
    }
}

impl<'de, const SPIN: bool, const DIM: usize> Deserialize<'de> for Model<SPIN, DIM, HasRMatrix> {
    fn deserialize<De: Deserializer<'de>>(
        deserializer: De,
    ) -> std::result::Result<Self, De::Error> {
        let mut wire = ModelWire::deserialize(deserializer)?;
        let rmatrix = wire
            .rmatrix
            .take()
            .ok_or_else(|| de::Error::missing_field("rmatrix"))?;
        wire.into_model(rmatrix)
    }
}


// Include Model implementation from submodules

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    #[inline(always)]
    pub fn has_rmatrix(&self) -> bool {
        R::HAS_RMATRIX
    }


    pub fn dim_r(&self) -> usize {
        DIM
    }
    /// Reciprocal lattice vectors satisfying `B Aᵀ = 2π·I`, where `A` is the
    /// real-space lattice (rows = lattice vectors, stored in [`Model::lat`]).
    ///
    /// Each row of the returned matrix is a reciprocal lattice vector
    /// `bᵢ`.  The inversion can fail for a degenerate real-space lattice.
    pub fn rec_lat(&self) -> Result<Array2<f64>> {
        let inv_t = self
            .lat
            .t()
            .to_owned()
            .inv()
            .map_err(|e| TbError::Other(format!("Failed to invert lattice: {e}")))?;
        Ok(std::f64::consts::TAU * inv_t)
    }
    #[inline(always)]
    pub fn atom_list(&self) -> Vec<usize> {
        let mut atom_list = Vec::new();
        for a in self.atoms.iter() {
            atom_list.push(a.norb());
        }
        atom_list
    }
    #[inline(always)]
    pub fn natom(&self) -> usize {
        self.atoms.len()
    }
    #[inline(always)]
    pub fn norb(&self) -> usize {
        self.orb.nrows()
    }
    #[inline(always)]
    pub fn nsta(&self) -> usize {
        if SPIN { 2 * self.norb() } else { self.norb() }
    }
}

impl<const DIM: usize, R: RMatrixData> Model<true, DIM, R> {
    /// Build `S_a / ℏ = σ_a ⊗ I_norb / 2` in the model basis.
    ///
    /// Returns an `(nsta, nsta)` matrix in spin-major order
    /// `(all ↑ orbitals, all ↓ orbitals)`. This is the spin operator;
    /// the corresponding spin current is `{S_a / ℏ, v} / 2`.
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
    /// 对应 [`OrbProj`] 定义的笛卡尔坐标轴，与晶格维数 `DIM` 无关。
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

#[cfg(test)]
mod ownership_tests {
    use super::*;
    use crate::AtomType;
    use ndarray::array;

    #[test]
    fn validation_rejects_singular_lattices_and_nonfinite_operators() {
        let valid = Model::<false, 2, HasRMatrix>::tb_model(
            array![[1.0, 0.2], [0.3, 1.0]],
            array![[0.0, 0.0]],
            None,
        )
        .unwrap();
        for lattice in [
            array![[0.0, 0.0], [0.0, 0.0]],
            array![[1.0, 2.0], [2.0, 4.0]],
        ] {
            let mut model = valid.clone();
            model.lat = lattice.clone();
            assert!(matches!(
                model.validate(),
                Err(TbError::InvalidModelInvariant {
                    invariant: "invertible_lattice",
                    ..
                })
            ));
            assert!(Model::<false, 2>::tb_model(lattice, array![[0.0, 0.0]], None).is_err());
            let encoded = toml::to_string(&model).unwrap();
            assert!(toml::from_str::<Model<false, 2, HasRMatrix>>(&encoded).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut model = valid.clone();
            model.ham[[0, 0, 0]].re = value;
            assert!(matches!(
                model.validate(),
                Err(TbError::InvalidModelInvariant {
                    invariant: "finite_hamiltonian",
                    ..
                })
            ));
            let encoded = toml::to_string(&model).unwrap();
            assert!(toml::from_str::<Model<false, 2, HasRMatrix>>(&encoded).is_err());
            model.ham[[0, 0, 0]].re = 0.0;
            model.rmatrix[[0, 0, 0, 0]].im = value;
            assert!(matches!(
                model.validate(),
                Err(TbError::InvalidModelInvariant {
                    invariant: "finite_position_matrix",
                    ..
                })
            ));
        }
    }

    #[test]
    fn validation_rejects_nonfinite_atomic_and_orbital_geometry() {
        let valid = Model::<false, 2>::tb_model(
            Array2::eye(2),
            array![[0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut model = valid.clone();
            model.orb[[0, 1]] = value;
            assert!(matches!(
                model.validate(),
                Err(TbError::InvalidModelInvariant {
                    invariant: "finite_geometry",
                    ..
                })
            ));
            model.orb[[0, 1]] = 0.0;
            model.atoms[0].set_position(array![value, 0.0]);
            assert!(matches!(
                model.validate(),
                Err(TbError::InvalidModelInvariant {
                    invariant: "atomic_position",
                    ..
                })
            ));
            // Nonfinite moments are rejected by the public setter before
            // they can enter a model; preserve the previous valid metadata.
            let mut atom = valid.atoms[0].clone();
            atom.set_magnetic_moment([1.0, 2.0, 3.0]).unwrap();
            assert!(atom.set_magnetic_moment([value, 0.0, 0.0]).is_err());
            assert_eq!(atom.magnetic_moment(), Some([1.0, 2.0, 3.0]));
        }
    }

    fn assert_model_round_trip<const SPIN: bool, R>()
    where
        R: RMatrixData + Serialize,
        for<'de> Model<SPIN, 3, R>: Deserialize<'de>,
    {
        let mut model = Model::<SPIN, 3, R>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0), OrbitalId::new(1)],
            )]),
        )
        .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.0]).unwrap();
        model.set_hop(Complex::new(0.3, -0.2), 0, 1, &array![1, 0, 0], None);
        model.add_hop(0.7, 0, 1, &array![1, 0, 0], None);

        let encoded = toml::to_string(&model).unwrap();
        let decoded: Model<SPIN, 3, R> = toml::from_str(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded.norb(), 2);
        assert_eq!(decoded.lat, model.lat);
        assert_eq!(decoded.orb, model.orb);
        assert_eq!(decoded.orb_projection, model.orb_projection);
        assert_eq!(decoded.ham, model.ham);
        assert_eq!(decoded.hamR, model.hamR);
        assert_eq!(
            decoded.atoms[0].orbitals(),
            &[OrbitalId::new(0), OrbitalId::new(1)]
        );
        assert_eq!(decoded.atoms[0].magnetic_moment(), Some([0.0, 0.0, 2.0]));
        assert_eq!(decoded.has_rmatrix(), R::HAS_RMATRIX);
        if R::HAS_RMATRIX {
            assert_eq!(decoded.rmatrix.as_array4(), model.rmatrix.as_array4());
        }
    }

    fn non_contiguous_model() -> Model<false, 3> {
        // Orbitals must sit at their parent atom's position (validate()
        // enforces ORBITAL_ATOM_POSITION_TOLERANCE).
        Model::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(
                    array![0.0, 0.0, 0.0],
                    AtomType::C,
                    [OrbitalId::new(0), OrbitalId::new(2)],
                ),
                Atom::with_orbitals(array![0.2, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn atom_view_borrows_explicit_non_contiguous_orbitals() {
        let model = non_contiguous_model();
        let atom = model.atom(AtomId::new(0)).unwrap();
        assert_eq!(atom.orbitals().len(), 2);
        assert_eq!(atom.orbitals()[0].id(), OrbitalId::new(0));
        assert_eq!(atom.orbitals()[1].id(), OrbitalId::new(2));
        assert_eq!(atom.orbitals()[1].position(), array![0.0, 0.0, 0.0].view());
    }

    #[test]
    fn orbital_only_model_has_no_fabricated_atoms() {
        let model =
            Model::<false, 3>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None).unwrap();
        assert!(model.atoms.is_empty());
        assert_eq!(model.orbital_owners().unwrap(), vec![None]);
        assert!(matches!(
            model.orb_angular(),
            Err(TbError::MissingAtomicStructure)
        ));
    }

    #[test]
    fn rmatrix_diagonal_is_cartesian_orbital_position() {
        // Position matrix elements are Cartesian (matching Wannier90
        // _r.dat); the default diagonal must equal frac · lat, not the raw
        // fractional coordinates.
        let lat = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let orb = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
        let model = Model::<true, 3, HasRMatrix>::tb_model(lat, orb, None).unwrap();
        let rmatrix = model.rmatrix.as_array4();
        let cart = model.orb.dot(&model.lat);
        for i in 0..model.nsta() {
            let s = i % model.norb();
            for axis in 0..3 {
                assert!(
                    (rmatrix[[0, axis, i, i]] - Complex::new(cart[[s, axis]], 0.0)).norm() < 1e-12,
                    "rmatrix diagonal ({i}, {axis}) must equal frac·lat = {}",
                    cart[[s, axis]]
                );
            }
        }
    }

    #[test]
    fn validate_rejects_orbitals_far_from_their_atom() {
        // An orbital displaced beyond ORBITAL_ATOM_POSITION_TOLERANCE from its
        // parent atom must be rejected at construction.
        let ok = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.05, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        );
        assert!(ok.is_ok(), "deviation within tolerance must pass");

        let bad = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.15, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        );
        assert!(
            matches!(
                bad,
                Err(TbError::InvalidModelInvariant {
                    invariant: "orbital_atom_position",
                    ..
                })
            ),
            "deviation beyond tolerance must fail with orbital_atom_position"
        );

        // Periodicity-aware: 0.97 vs 0.03 is 0.06 apart mod 1 (across the
        // boundary) — must pass even though the raw difference is 0.94.
        let across_boundary = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.97, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.03, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        );
        assert!(
            across_boundary.is_ok(),
            "mod-1 distance 0.06 must be allowed"
        );
    }

    #[test]
    fn remove_last_orbital_is_transactional() {
        // Regression: removing every orbital mutated all arrays before the
        // final validate() reported NoOrbitals, leaving the caller's model
        // corrupted. The error must fire before any mutation.
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let norb_before = model.norb();
        assert!(matches!(model.remove_orb(&[0]), Err(TbError::NoOrbitals)));
        assert_eq!(
            model.norb(),
            norb_before,
            "failed removal must not mutate the model"
        );
        assert!(matches!(model.remove_atom(&[0]), Err(TbError::NoOrbitals)));
        assert_eq!(model.norb(), norb_before);
        assert_eq!(
            model.natom(),
            1,
            "failed atom removal must not mutate atoms"
        );
    }

    #[test]
    fn tb_model_rejects_empty_orbital_set() {
        // Regression: tb_model with a zero-row orb matrix used to succeed and
        // produce a model that silently flowed into solve/response entry
        // points. It must now fail with NoOrbitals.
        let result =
            Model::<false, 3>::tb_model(Array2::eye(3), Array2::<f64>::zeros((0, 3)), None);
        assert!(matches!(result, Err(TbError::NoOrbitals)));
    }

    #[test]
    fn orb_angular_rejects_unowned_orbitals() {
        // Regression: an orbital not owned by any atom used to silently keep
        // zero angular-momentum matrix elements (indistinguishable from a
        // genuine s-orbital). It must now produce a clear error.
        let model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let err = model.orb_angular().unwrap_err();
        assert!(
            err.to_string().contains("has no owner"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_orbital_ownership_is_rejected() {
        let result = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::O, [OrbitalId::new(0)]),
            ]),
        );
        assert!(matches!(result, Err(TbError::DuplicateOrbitalOwner { .. })));
    }

    #[test]
    fn removing_an_orbital_remaps_ids_and_keeps_empty_sites() {
        let mut model = non_contiguous_model();
        model.remove_orb(&[1]).unwrap();
        model.validate().unwrap();
        assert_eq!(model.norb(), 2);
        assert_eq!(
            model.atoms[0].orbitals(),
            &[OrbitalId::new(0), OrbitalId::new(1)]
        );
        assert!(model.atoms[1].orbitals().is_empty());
    }

    #[test]
    fn explicit_atom_removal_modes_preserve_their_named_semantics() {
        let mut metadata_only = non_contiguous_model();
        metadata_only.remove_atoms_only(&[0]).unwrap();
        assert_eq!(metadata_only.norb(), 3);
        assert_eq!(metadata_only.natom(), 1);
        assert_eq!(
            metadata_only.orbital_owners().unwrap(),
            vec![None, Some(AtomId::new(0)), None]
        );

        metadata_only.remove_orb(&[1]).unwrap();
        assert_eq!(metadata_only.natom(), 1);
        assert!(metadata_only.atoms[0].orbitals().is_empty());
        metadata_only.prune_empty_atoms().unwrap();
        assert_eq!(metadata_only.natom(), 0);

        let mut cascading = non_contiguous_model();
        cascading.remove_atoms_and_orbitals(&[0]).unwrap();
        assert_eq!(cascading.natom(), 1);
        assert_eq!(cascading.norb(), 1);
        assert_eq!(cascading.atoms[0].orbitals(), &[OrbitalId::new(0)]);
    }

    #[test]
    fn orbital_only_supercell_remains_orbital_only() {
        let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.25]], None).unwrap();
        let supercell = model.make_supercell(&array![[2.0]]).unwrap();
        assert_eq!(supercell.norb(), 2);
        assert!(supercell.atoms.is_empty());
        assert_eq!(supercell.orbital_owners().unwrap(), vec![None, None]);
        supercell.validate().unwrap();
    }

    #[test]
    fn supercell_replicates_optional_atom_moments() {
        let mut model = Model::<false, 1>::tb_model(
            array![[1.0]],
            array![[0.25]],
            Some(vec![Atom::with_orbitals(
                array![0.25],
                AtomType::Fe,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        model.atoms[0].set_magnetic_moment([0.0, 0.0, 2.5]).unwrap();
        let supercell = model.make_supercell(&array![[3.0]]).unwrap();
        assert_eq!(supercell.natom(), 3);
        assert!(
            supercell
                .atoms
                .iter()
                .all(|atom| atom.magnetic_moment() == Some([0.0, 0.0, 2.5]))
        );
        supercell.validate().unwrap();
    }

    #[test]
    fn model_serde_round_trips_all_type_level_storage_combinations() {
        assert_model_round_trip::<false, NoRMatrix>();
        assert_model_round_trip::<true, NoRMatrix>();
        assert_model_round_trip::<false, HasRMatrix>();
        assert_model_round_trip::<true, HasRMatrix>();
    }

    #[test]
    fn model_serde_checks_required_fields_types_and_storage() {
        let model =
            Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None)
                .unwrap();
        let value = toml::Value::try_from(&model).unwrap();
        let encoded = toml::to_string(&value).unwrap();
        let stripped: Model<false, 3> = toml::from_str(&encoded).unwrap();
        assert_eq!(stripped.ham, model.ham);
        assert_eq!(stripped.orb, model.orb);
        assert!(
            toml::from_str::<Model<true, 3>>(&encoded)
                .unwrap_err()
                .to_string()
                .contains("spin mismatch")
        );
        assert!(
            toml::from_str::<Model<false, 2, HasRMatrix>>(&encoded)
                .unwrap_err()
                .to_string()
                .contains("dimension mismatch")
        );
        for field in [
            "dim_r",
            "spin",
            "lat",
            "orb",
            "orb_projection",
            "atoms",
            "ham",
            "hamR",
            "rmatrix",
        ] {
            let mut missing = value.clone();
            missing.as_table_mut().unwrap().remove(field);
            let encoded = toml::to_string(&missing).unwrap();
            let error = toml::from_str::<Model<false, 3, HasRMatrix>>(&encoded).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("missing field `{field}`"))
            );
            if field == "rmatrix" {
                assert!(toml::from_str::<Model<false, 3>>(&encoded).is_ok());
            } else {
                assert!(toml::from_str::<Model<false, 3>>(&encoded).is_err());
            }
        }
        let mut malformed = value;
        malformed["rmatrix"] = toml::Value::Integer(0);
        let encoded = toml::to_string(&malformed).unwrap();
        assert!(toml::from_str::<Model<false, 3>>(&encoded).is_err());
        assert!(toml::from_str::<Model<false, 3, HasRMatrix>>(&encoded).is_err());
    }

    #[test]
    fn model_serde_rejects_unknown_and_duplicate_fields() {
        let model =
            Model::<false, 3, HasRMatrix>::tb_model(Array2::eye(3), array![[0.0, 0.0, 0.0]], None)
                .unwrap();
        let value = toml::Value::try_from(&model).unwrap();
        for field in ["spin", "rmatrix", "unexpected"] {
            let mut fields: Vec<_> = value.as_table().unwrap().clone().into_iter().collect();
            fields.push((
                field.to_string(),
                value.get(field).cloned().unwrap_or(toml::Value::Integer(0)),
            ));
            // Feed entries directly: a TOML parser would reject duplicate keys
            // itself and never exercise the derived Model deserializer.
            let error = Model::<false, 3>::deserialize(de::value::MapDeserializer::<
                _,
                toml::de::Error,
            >::new(
                fields.clone().into_iter()
            ))
            .unwrap_err();
            let expected = if field == "unexpected" {
                "unknown field"
            } else {
                "duplicate field"
            };
            assert!(error.to_string().contains(expected), "{error}");
            let error = Model::<false, 3, HasRMatrix>::deserialize(de::value::MapDeserializer::<
                _,
                toml::de::Error,
            >::new(
                fields.into_iter()
            ))
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn legacy_atom_counts_deserialize_to_contiguous_typed_ids() {
        let mut model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            Some(vec![
                Atom::with_orbitals(array![0.0, 0.0, 0.0], AtomType::C, [OrbitalId::new(0)]),
                Atom::with_orbitals(array![0.5, 0.0, 0.0], AtomType::O, [OrbitalId::new(1)]),
            ]),
        )
        .unwrap();
        model.atoms[1].set_magnetic_moment([0.0, 1.0, 0.0]).unwrap();
        let mut value = toml::Value::try_from(&model).unwrap();
        let atoms = value["atoms"].as_array_mut().unwrap();
        for atom in atoms {
            let table = atom.as_table_mut().unwrap();
            let count = table.remove("orbitals").unwrap().as_array().unwrap().len();
            table.insert("atom_list".to_string(), toml::Value::Integer(count as i64));
            if let Some(moment) = table.remove("magnetic_moment") {
                table.insert("magnetic".to_string(), moment);
            }
        }

        let decoded: Model<false, 3> = toml::from_str(&toml::to_string(&value).unwrap()).unwrap();
        assert_eq!(decoded.atoms[0].orbitals(), &[OrbitalId::new(0)]);
        assert_eq!(decoded.atoms[1].orbitals(), &[OrbitalId::new(1)]);
        assert_eq!(decoded.atoms[0].magnetic_moment(), None);
        assert_eq!(decoded.atoms[1].magnetic_moment(), Some([0.0, 1.0, 0.0]));
    }

    #[test]
    fn standalone_legacy_atom_deserialization_is_rejected() {
        // Regression: deserializing a standalone Atom in the legacy
        // per-atom-count format assigned IDs 0..N to each atom independently,
        // producing overlapping OrbitalIds that surfaced later as a cryptic
        // DuplicateOrbitalOwner at Model attach. It must now be rejected with
        // a migration hint.
        let model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::C,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let mut value = toml::Value::try_from(&model).unwrap();
        let atom = value["atoms"].as_array_mut().unwrap()[0]
            .as_table_mut()
            .unwrap();
        atom.remove("orbitals").unwrap();
        atom.insert("atom_list".to_string(), toml::Value::Integer(1));
        let wire = toml::to_string(&atom).unwrap();
        let err = toml::from_str::<Atom>(&wire).unwrap_err();
        assert!(
            err.to_string().contains("standalone Atom"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn legacy_zero_magnetic_alias_is_distinct_from_a_missing_moment() {
        let model = Model::<false, 3>::tb_model(
            Array2::eye(3),
            array![[0.0, 0.0, 0.0]],
            Some(vec![Atom::with_orbitals(
                array![0.0, 0.0, 0.0],
                AtomType::Fe,
                [OrbitalId::new(0)],
            )]),
        )
        .unwrap();
        let mut value = toml::Value::try_from(&model).unwrap();
        let atom = value["atoms"].as_array_mut().unwrap()[0]
            .as_table_mut()
            .unwrap();
        atom.insert(
            "magnetic".to_string(),
            toml::Value::Array(vec![
                toml::Value::Float(0.0),
                toml::Value::Float(0.0),
                toml::Value::Float(0.0),
            ]),
        );
        let decoded: Model<false, 3> = toml::from_str(&toml::to_string(&value).unwrap()).unwrap();
        assert_eq!(decoded.atoms[0].magnetic_moment(), Some([0.0; 3]));
    }

    #[test]
    fn standalone_atom_deserialization_rejects_nonfinite_moments() {
        let atom = Atom::new(array![0.0, 0.0, 0.0], AtomType::Fe);
        let mut value = toml::Value::try_from(&atom).unwrap();
        value.as_table_mut().unwrap().insert(
            "magnetic_moment".to_string(),
            toml::Value::Array(vec![
                toml::Value::Float(f64::NAN),
                toml::Value::Float(0.0),
                toml::Value::Float(0.0),
            ]),
        );
        let encoded = toml::to_string(&value).unwrap();
        assert!(toml::from_str::<Atom>(&encoded).is_err());
    }
}
