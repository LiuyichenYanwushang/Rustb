//! Magnetic field implementation using Landau gauge and Peierls substitution.
//!
//! This module provides a trait for adding a uniform magnetic field $\mathbf{B}$ to a
//! tight‑binding model. The orbital effect is included via the Peierls substitution,
//! while the spin Zeeman coupling is added when spin degrees of freedom are present.
//!
//! ## Mathematical Background
//!
//! ### 1. Peierls substitution in the Landau gauge
//!
//! In a tight‑binding basis, a magnetic field modifies the hopping amplitude
//! $t_{ij}(\mathbf{R})$ by a phase factor
//!
//! $$
//! t_{ij}(\mathbf{R}) \to t_{ij}(\mathbf{R})
//! \exp\left( -i\frac{e}{\hbar} \int_{\mathbf{r}_j+\mathbf{R}}^{\mathbf{r}_i}
//! \mathbf{A}(\mathbf{r})\cdot d\mathbf{r} \right),
//! $$
//!
//! where $\mathbf{A}$ is the vector potential ($\mathbf{B} = \nabla\times\mathbf{A}$).
//! For a uniform field $\mathbf{B}=B\hat{\mathbf{z}}$ in a 2D system, the Landau gauge
//! $\mathbf{A} = (0, Bx, 0)$ is commonly used. However, for arbitrary (possibly non‑orthogonal)
//! lattices and for 3D systems, it is convenient to work with **fractional coordinates**
//! $u^1, u^2$ along the two directions perpendicular to $\mathbf{B}$.
//!
//! Let the two perpendicular lattice vectors be $\mathbf{a}_1, \mathbf{a}_2$ and let
//! $(u^1, u^2) \in [0,1)^2$ be the fractional coordinates of an orbital.
//! If the supercell repeats $N_1$ times along $\mathbf{a}_1$ and $N_2$ times along
//! $\mathbf{a}_2$, the total magnetic flux through the supercell is
//!
//! $$
//! \Phi = \phi \Phi_0, \qquad \Phi_0 = 2\pi\frac{\hbar}{e},
//! $$
//!
//! where $\phi$ is an integer (the number of flux quanta per supercell). The flux per
//! original unit cell is then $\phi/(N_1 N_2)$.
//!
//! Using the gauge $\mathbf{A} = \frac{\Phi}{N_1 N_2} u^1 \nabla u^2$ (in fractional
//! coordinates) and properly accounting for the periodic boundary conditions of the
//! supercell, the Peierls phase for a hop from orbital $i$ at $(u^1_i, u^2_i)$ to
//! orbital $j$ at $(u^1_j, u^2_j)$ with cell translation $(m_1, m_2)$ becomes
//! [`MagneticField::add_magnetic_field`]
//!
//! $$
//! \theta = 2\pi\phi\left[
//!   \frac{u^1_i + v^1_j}{2}(v^2_j - u^2_i) - m_1 v^2_j
//! \right],
//! $$
//!
//! where $v^1_j = u^1_j + m_1$ and $v^2_j = u^2_j + m_2$ are the absolute fractional
//! coordinates of the target orbital. The second term $-m_1 v^2_j$ restores translation
//! symmetry of the supercell Hamiltonian and is essential for a correct spectrum.
//!
//! ### 2. Zeeman coupling
//!
//! If spin is enabled (`Model.spin = true`), the Zeeman energy is added to the on‑site
//! term of the Hamiltonian:
//!
//! $$
//! H_Z = \frac{g\mu_B}{\hbar}\mathbf{B}\cdot\mathbf{S},
//! $$
//!
//! where $g \approx 2$, $\mu_B$ is the Bohr magneton, and $\mathbf{S} = \frac{\hbar}{2}\bm{\sigma}$.
//! The magnetic field magnitude is calculated from the flux $\phi$ and the area of the
//! supercell face perpendicular to $\mathbf{B}$.
//!
//! ## Usage
//!
//! The trait `MagneticField` provides a single method `add_magnetic_field`.
//! It requires the model to be 2D or 3D.
//!
//! * For a **2D** model, the magnetic field must point along the $z$‑axis
//!   (i.e. `mag_dir = 2`). The `expand` array gives the supercell repetitions
//!   along the two in‑plane lattice vectors (e.g. `[Nx, Ny]`).
//! * For a **3D** model, `mag_dir` can be `0` ($x$), `1` ($y$), or `2` ($z$).
//!   The two perpendicular directions are chosen automatically with a cyclic order
//!   that preserves the right‑hand rule for the cross product.
//!
//! The parameter `phi` is the **total number of flux quanta** $\Phi/\Phi_0$ through the
//! entire supercell. It must be an integer. The physical flux per original unit cell is
//! $\phi / (\text{expand}[0] \cdot \text{expand}[1])$.
//!
//! ## Example
//!
//! ```rust
//! use Rustb::{Model, MagneticField};
//! let model: Model = /* ... a 2D model ... */;
//! // Magnetic field in z‑direction, supercell 9×9, 40 flux quanta
//! let magnetic_model = model.add_magnetic_field(2, [9, 9], 40)?;
//! ```
//!
//! ## References
//!
//! - D. R. Hofstadter, *Energy levels and wave functions of Bloch electrons in rational
//!   and irrational magnetic fields*, Phys. Rev. B **14**, 2239 (1976).
//! - D. Vanderbilt, *Berry Phases in Electronic Structure Theory*, Cambridge (2018),
//!   Chapter 6.

use crate::Model;
use crate::error::{Result, TbError};
use crate::find_R;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use std::f64::consts::PI;

/// 辅助函数：计算最大公约数
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Trait for adding a magnetic field to a tight-binding model.
pub trait MagneticField {
    /// Add a uniform magnetic field in the Landau gauge.
    ///
    /// ## Arguments
    /// * `mag_dir` – 磁场方向 (0, 1, 或 2 代表 x, y, z)。如果是 2D 体系，磁场应该在 z 轴方向，即 `mag_dir = 2`。
    /// * `expand` – 长度为2的数组 `[a, b]`，表示垂直于磁场方向的两个格矢上的扩胞倍数。
    /// * `phi` – 穿过整个超胞 (a*b) 的总磁通量，单位为 $\Phi_0$ (整数)。
    ///           原胞内的物理磁通为 phi / (a*b)。
    fn add_magnetic_field(&self, mag_dir: usize, expand: [usize; 2], phi: isize) -> Result<Self>
    where
        Self: Sized;
}

impl MagneticField for Model {
    fn add_magnetic_field(&self, mag_dir: usize, expand: [usize; 2], phi: isize) -> Result<Self> {
        // 1. 验证输入维度与方向
        if mag_dir >= self.dim_r() && !(self.dim_r() == 2 && mag_dir == 2) {
            return Err(TbError::InvalidDirection {
                index: mag_dir,
                dim: self.dim_r(),
            });
        }
        if expand[0] == 0 || expand[1] == 0 {
            return Err(TbError::InvalidSupercellSize(0));
        }

        let total_area_cells = expand[0] * expand[1];

        // 计算并提示每个原胞的有效磁通量 (处理互质问题)
        if phi != 0 {
            let gcd_val = gcd(phi.abs() as usize, total_area_cells);
            if gcd_val > 1 {
                let reduced_p = phi / (gcd_val as isize);
                let reduced_q = total_area_cells / gcd_val;
                println!(
                    "提示: 每个原胞的磁通量为 {}/{} Φ₀, 互质约化为 {}/{} Φ₀。",
                    phi, total_area_cells, reduced_p, reduced_q
                );
            } else {
                println!("提示: 每个原胞的磁通量为 {}/{} Φ₀。", phi, total_area_cells);
            }
        }

        // 2. 确定与磁场垂直的两个非磁场格矢方向
        let perp_dirs: [usize; 2] = match self.dim_r() {
            3 => match mag_dir {
                0 => [1, 2],
                1 => [2, 0], // 偶置换保证磁场旋度的正负号正确 (z × x = y)
                2 => [0, 1],
                _ => unreachable!(),
            },
            2 => match mag_dir {
                2 => [0, 1], // 2D 体系: 磁场位于 z (2), 非磁场格矢只能是 x(0) 和 y(1)
                _ => {
                    return Err(TbError::InvalidDirection {
                        index: mag_dir,
                        dim: 2,
                    });
                } // 面内磁场无法产生轨道Peierls效应，予以拦截
            },
            _ => {
                return Err(TbError::InvalidDimension {
                    dim: self.dim_r(),
                    supported: vec![2, 3],
                });
            }
        };

        // 3. 一次性构建 a × b 超胞
        let mut u_matrix = Array2::<f64>::eye(self.dim_r());
        u_matrix[[perp_dirs[0], perp_dirs[0]]] = expand[0] as f64;
        u_matrix[[perp_dirs[1], perp_dirs[1]]] = expand[1] as f64;
        let super_model = self.make_supercell(&u_matrix)?;

        let d1 = perp_dirs[0];
        let d2 = perp_dirs[1];
        let norb = super_model.norb();
        let spin = super_model.spin;
        let tot_orb = if spin { norb * 2 } else { norb };

        // 4. Peierls 轨道效应：使用鲁棒的分数坐标规范场 (Fractional Gauge)
        let mut new_ham = super_model.ham.clone();

        if phi != 0 {
            for iR in 0..super_model.hamR.nrows() {
                let R_vec = super_model.hamR.row(iR);
                // m1, m2 是以“超胞格矢”为单位的平移整数
                let m1 = R_vec[d1] as f64;
                let m2 = R_vec[d2] as f64;

                let mut ham_slice = new_ham.slice_mut(s![iR, .., ..]);

                // 统一处理所有轨道（包括自旋翻转项）
                for i in 0..tot_orb {
                    let spatial_i = i % norb;
                    // u1, u2 是归一化到[0, 1) 的分数坐标，天然包容了任何非正交形变
                    let u1_i = super_model.orb[[spatial_i, d1]];
                    let u2_i = super_model.orb[[spatial_i, d2]];

                    for j in 0..tot_orb {
                        let hop = ham_slice[[i, j]];
                        if hop.norm() < 1e-12 {
                            continue;
                        }

                        let spatial_j = j % norb;
                        let u1_j = super_model.orb[[spatial_j, d1]];
                        let u2_j = super_model.orb[[spatial_j, d2]];

                        // 跃迁目标的绝对分数坐标 (包含跨越胞的平移)
                        let v1_j = u1_j + m1;
                        let v2_j = u2_j + m2;

                        // 【核心物理修正】
                        // 计算基于超胞规范群修正的 Peierls 相位。
                        // 第一项是 Landau 规范积分: A = Φ * u1 * ∇u2
                        // 第二项是规范变换相 -m1 * v2_j，用于修复 PBC 边界条件下的哈密顿量平移对称性。
                        let phase_val = 2.0
                            * PI
                            * (phi as f64)
                            * ((u1_i + v1_j) / 2.0 * (v2_j - u2_i) - m1 * v2_j);

                        let peierls = Complex::new(phase_val.cos(), phase_val.sin());
                        ham_slice[[i, j]] = hop * peierls;
                    }
                }
            }
        }

        // 5. Zeeman 效应 (如果包含自旋)
        if spin && phi != 0 {
            // 精确计算垂直于磁场平面的超胞截面积 (适用非正交格矢)
            let area = if self.dim_r() == 3 {
                let L1 = super_model.lat.row(d1);
                let L2 = super_model.lat.row(d2);
                let cross_x = L1[1] * L2[2] - L1[2] * L2[1];
                let cross_y = L1[2] * L2[0] - L1[0] * L2[2];
                let cross_z = L1[0] * L2[1] - L1[1] * L2[0];
                (cross_x.powi(2) + cross_y.powi(2) + cross_z.powi(2)).sqrt()
            } else {
                let L1 = super_model.lat.row(0);
                let L2 = super_model.lat.row(1);
                (L1[0] * L2[1] - L1[1] * L2[0]).abs()
            };

            const PHI0: f64 = 4.135667662e-15; // T·m²
            let area_m2 = area * 1e-20; // 从 Å² 转为 m²
            let B_tesla = (phi as f64) * PHI0 / area_m2;

            const MU_B: f64 = 5.7883818012e-5; // eV/T
            let g = 2.0;
            let zeeman_energy = g * MU_B * B_tesla; // eV

            // 定位 On-site 分块 (R = 0)
            let zero_R = Array1::<isize>::zeros(super_model.dim_r());
            let index0 = find_R(&super_model.hamR, &zero_R).unwrap();
            let mut ham0 = new_ham.slice_mut(s![index0, .., ..]);

            for i in 0..norb {
                match mag_dir {
                    0 => {
                        // σ_x 矩阵
                        ham0[[i, i + norb]] += zeeman_energy / 2.0;
                        ham0[[i + norb, i]] += zeeman_energy / 2.0;
                    }
                    1 => {
                        // σ_y 矩阵：[0, -i; i, 0]
                        ham0[[i, i + norb]] += Complex::new(0.0, -zeeman_energy / 2.0);
                        ham0[[i + norb, i]] += Complex::new(0.0, zeeman_energy / 2.0);
                    }
                    2 => {
                        // σ_z 矩阵
                        ham0[[i, i]] += zeeman_energy / 2.0;
                        ham0[[i + norb, i + norb]] -= zeeman_energy / 2.0;
                    }
                    _ => unreachable!(),
                }
            }
        }

        let mut new_model = super_model;
        new_model.ham = new_ham;
        Ok(new_model)
    }
}
