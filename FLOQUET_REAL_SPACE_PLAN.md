# Floquet Real-Space Bessel Backend — 设计计划书

目标：让 `floquet_effective_model` 对**任意偏振（线/圆/椭圆）、任意多束、任意公度谐波**的驱动，
直接在实空间用广义 Bessel 展开构造一阶 van Vleck 有效模型。消除 `k_mesh` 与 `n_time`
两个人为参数，主路径变为精确的实空间卷积。

状态：设计阶段，未实现。时间网格路径保留为 reference backend。

---

## 1. 通用驱动表示

所有偏振/多束/多频统一归结为一个**复振幅模集合**（现有 `FloquetDrive` 已经是这个形式）：

```math
a(t) = \operatorname{Re}\sum_{\alpha} a_\alpha\, e^{-i l_\alpha \Omega_0 t},
\qquad a_\alpha \in \mathbb C^{DIM},\quad l_\alpha \in \mathbb Z.
```

- 线偏振：`a_α` 为实矢量（一个实振幅沿偏振方向）
- 圆偏振：`a_α = (a_0/\sqrt{2})(\mathbf e_1 \pm i\,\mathbf e_2)`（`IncidentBasis::polarization` 的 Jones 矢量给出）
- 椭圆偏振：一般的复矢量 `a_α`
- 多束相干：每条束一个模（或多模），**全部放进同一个 `FloquetDrive`**
- 多束非相干（phase-averaged）：分别构造各束的模型再相加（不混在一个 drive 里）

**每条 bond 的投影** `d_{ijR} = (R + τ_j − τ_i)L`：

```math
z_\alpha(d) := a_\alpha \cdot d = R_\alpha(d)\, e^{i\delta_\alpha(d)},
\qquad
R_\alpha = |z_\alpha|,\quad \delta_\alpha = \arg z_\alpha,
```

于是单模对 bond 的贡献：

```math
a_\alpha(t)\cdot d = R_\alpha \cos(l_\alpha \Omega_0 t - \delta_\alpha).
```

**关键**：`(R_α, δ_α)` 对每条 bond 是**标量对**——偏振几何的全部信息压缩进去了，后续公式与偏振形式无关。

---

## 2. 广义 Bessel 展开（精确、按模递归卷积）

Jacobi–Anger（配合本库约定 `e^{−i a(t)·d}`）：

```math
e^{-i R \cos\theta} = \sum_{m=-\infty}^{\infty} (-i)^m J_m(R)\, e^{im\theta}.
```

Peierls 指数按模严格因式分解：

```math
e^{-i a(t)\cdot d}
=
\prod_{\alpha} e^{-i R_\alpha \cos(l_\alpha \Omega_0 t - \delta_\alpha)}
=
\prod_{\alpha} \sum_{m_\alpha} (-i)^{m_\alpha} J_{m_\alpha}(R_\alpha)\, e^{-i m_\alpha \delta_\alpha}\, e^{i m_\alpha l_\alpha \Omega_0 t}.
```

Fourier 投影 `C_q(d) = (1/T_0)\int e^{iq\Omega_0 t} e^{-ia(t)\cdot d} dt`（`T_0 = 2π/Ω₀`）。
共振条件 `q + Σ_α l_α m_α = 0`，即选择定则 **`Σ_α l_α m_α = −q`**（已由直接数值积分验证；
等价的另一种写法是 `Σ l m = +q` 配相位 `e^{+imδ}`——两者不可混用）：

```math
\boxed{
C_q(d)
=
\sum_{\{m_\alpha\}\,:\,\sum_\alpha l_\alpha m_\alpha = -q}
\;\prod_\alpha\; (-i)^{m_\alpha}\, J_{m_\alpha}\!\big(R_\alpha(d)\big)\,
e^{-i m_\alpha \delta_\alpha(d)}
}
```

### 2.1 递归卷积算法（不枚举多指标）

定义第 α 个模的**单模序列**（`|m| ≤ M_α` 截断）：

```math
B_\alpha(m) := (-i)^m\, J_m(R_\alpha)\, e^{-i m \delta_\alpha},
\qquad -M_\alpha \le m \le M_\alpha.
```

顺序折叠（离散卷积，注意 **`+l_α m`** 的方向来自共振条件 `q + Σ l m = 0`）：

```math
S^{(0)}_q = \delta_{q,0},
\qquad
S^{(\alpha)}_q = \sum_{m=-M_\alpha}^{M_\alpha} S^{(\alpha-1)}_{q + l_\alpha m}\; B_\alpha(m).
```

全部模折叠完成后 `C_q = S^{(N_mode)}_q`。

**复杂度**：`O(N_mode × N_q × M_avg)`，与 `n_time` 无关。
**截断**：截断误差以尾部 `Σ_{|m|>M_α}|J_m(R_α)|` 衡量（实测：`R=5`、`M=11`（margin 6）→ **1.9e-4**；`R=8`、`M=14` → ~8e-4——单模尾并非 `(R/2)^M/M!` 单一项，而是整条尾之和）。
达到 <1e-12 需 `M = ⌈R⌉ + 16` 量级。**实现采用自适应截断**：逐模增大 `M_α` 直到
`Σ_{|m|>M_α}|J_m(R_α)| ≤ ε/(N_mode·S_α)`（`S_α` 为当前部分和上界，`ε = 1e-12`），
而非固定 margin。`cutoff_margin` 参数仅作最小 margin 与测试钩子保留，取值范围
`[0, 48]`（配合 `R ≤ 8` 保证 `M_α ≤ 64` 的硬上界：`⌈8⌉ + 48 = 56`，该处尾部已 ~1e-64，
增长循环不会触发）。
对超大 `R_α`（强场）退化为渐近展开或回退时间网格（见 §7）。

### 2.2 Bessel 函数求值

`J_m(R)`：包装 `puruspe::Jn`（纯 Rust 特殊函数库，MIT/Apache-2.0；实测本范围最坏相对误差 ~2e-15）。
负阶用 `J_{−m} = (−1)^m J_m`。一个纯函数 `bessel_j(m: isize, r: f64) -> f64`，单元测试对照 NIST 查表值 + 独立的 Miller 下行递推参照。

---

## 3. 实空间一阶 van Vleck（精确、无 k 空间）

记号：`A_R := T_q(R) = t(R)·C_q(d_R)`，`B_R := T_{−q}(R)`（都是 `FloquetHarmonicCache` 的块）。
实空间 Hermiticity 对（已验证）：`B_R = A_{−R}^\dagger`，且 k 空间里 `B(k) = A(k)^\dagger`。

对易子块 = **两个方向的离散卷积**（已验证 `(BA)(R) ≠ (AB)(R)^\dagger`——块一般不交换，
"单卷积 + 共轭"的简化是**错误的**，会让 `H_eff` 产生 O(1) 误差）：

```math
\boxed{
\mathrm{comm}_q(R)
=
(AB)(R) - (BA)(R),
\qquad
(AB)(R) = \sum_{R'} A_{R-R'}\, B_{R'},
\qquad
(BA)(R) = \sum_{R'} B_{R-R'}\, A_{R'}
}
```

Hermiticity 由构造保证（k 空间 `comm(k) = [A, A†]` Hermitian ⇒ 实空间对
`comm(R) = comm(−R)^\dagger`），fp 噪声在后处理中对称化。

一阶有效模型（**完全实空间、精确、自动 Hermitian**）：

```math
\boxed{
T_{\rm eff}(R)
=
T_0(R)
+
\sum_{q=1}^{q_{\max}} \frac{\mathrm{comm}_q(R)}{q\,\hbar\Omega_0}
}
```

- **support 自动确定**：`supp(T_eff) = supp(T_0) ∪ ⋃_q [supp(A) + supp(B)]`（Minkowski 和），
  不再需要 `target_hamR` 参数；
- **无 k_mesh、无混叠、无逆 FT**；
- 复杂度 `O(|supp|² × q_max × nsta³)`，每个乘积 `A_{R−R'}·B_{R'}`、`B_{R−R'}·A_{R'}` 用 `blas::zgemm`
  （两个卷积都要算，"P − P† 半乘法"不成立）。

---

## 4. 函数接口设计（Rust 类型签名草案）

### 4.1 后端选择

```rust
/// 求 C_q(d) 的两种后端。
pub enum PeierlsFourierMethod {
    /// 时间网格数值 DFT（现状；reference/兜底）
    TimeGrid,
    /// 广义 Bessel 解析展开（精确、O(N_mode·N_q·M)）
    Bessel { cutoff_margin: isize },   // 最小 margin 与测试钩子；实际截断自适应（见 §2.1）
}
```

### 4.2 Bessel 系数（纯函数）

```rust
/// d: 实数键位移（Cartesian，DIM 维）
/// drive: 模集合（l_α, a_α）
/// q_range: [q_min, q_max]
/// 返回 C_q(d)，q = q_min..=q_max
fn bessel_peierls_coeffs(
    d: &Array1<f64>,
    drive: &FloquetDrive,
    q_min: isize,
    q_max: isize,
    cutoff_margin: isize,
) -> Result<Array1<Complex<f64>>>
```

内部步骤：
1. 每模算 `z_α = a_α·d` → `R_α, δ_α`（`R_α = 0` 时模退化为 `δ_{m,0}`，跳过）
2. 自适应截断 `M_α`（尾部 `Σ_{|m|>M_α}|J_m(R_α)| ≤ ε`，`ε=1e-12`）；`cutoff_margin` 仅作最小 margin
3. `S` 序列按 §2.1 折叠（长度 `q_max − q_min + 1`，卷积边界 clamp 到范围外=0）
4. 返回 `S`

### 4.3 谐波缓存（复用现有 `FloquetHarmonicCache`）

```
floquet_harmonic_cache(drive, trunc, q_min, q_max, method)
    -> FloquetHarmonicCache
    // blocks[q_index(q), i_r, i, j] = t_ij(R) · C_q(d_ijR)
```

新增：**按 distinct `d` 去重的 `C_q` 缓存**（`HashMap<[u64; DIM] 位模式, Vec<Complex<f64>>>`）——
spin up/down 共用同一 `d`。对称键 `±d` 互查（`C_q(−d) = C_{−q}(d)*`）留作后续优化
（要求 q 范围对称；当前调用点均满足，先保持最小改动）。

### 4.4 实空间对易子

```rust
/// A_R = T_q(R)、B_R = T_{−q}(R)，均与 hamR 一一对应（来自谐波缓存）。
/// 返回 (comm(R) 块序列, 其 R 矢量序列)。
fn real_space_commutator(
    a_blocks: &[Array2<Complex<f64>>],   // T_q(R)，长度 = hamR.nrows()
    b_blocks: &[Array2<Complex<f64>>],   // T_{−q}(R)，长度 = hamR.nrows()
    ham_r: &Array2<isize>,
) -> Result<(Vec<Array2<Complex<f64>>>, Array2<isize>)>
```

步骤：
1. `supp = {R1 + R2 : R1, R2 ∈ hamR}`（Minkowski 和，去重、字典序输出）
2. 对每个 `R ∈ supp`：
   `(AB)(R) = Σ_{R'} A_{R−R'} · zgemm B_{R'}`，
   `(BA)(R) = Σ_{R'} B_{R−R'} · zgemm A_{R'}`（块索引用 `HashMap<R, usize>` 查，
   同一 `R−R'` 查表共享给两个卷积；行主序 `C += α·A·B` 用转置恒等式
   `(A·B)^T = B^T·A^T` 直接喂列主序 `zgemm('N','N')`，无转置拷贝）
3. `comm(R) = (AB)(R) − (BA)(R)`（AB 项 α=+1、BA 项 α=−1，同一累加器）
4. fp 后处理：强制 `comm(R) = comm(−R)†`（复用 `enforce_real_space_hermiticity`
   做 ±R 对平均；构造本身已 Hermitian，这一步只清浮点噪声）
5. support 不闭合于 `R → −R` 时返回 `MissingHermitianConjugateHopping`
   （仅手改 hamR 破坏 Model 不变式时发生）

### 4.5 顶层入口

```rust
/// 实空间 Bessel 有效模型（新主路径）
pub fn floquet_effective_model_bessel(
    &self,
    drive: &FloquetDrive,
    trunc: &FloquetTruncation,
    options: &FloquetEffectiveOptions,     // 只含 order=1、q_max
) -> Result<Model<SPIN, DIM, NoRMatrix>>
```

内部：
1. `cache = floquet_harmonic_cache(drive, trunc, −q_max, q_max, Bessel)`（含 d 去重）
2. `T_0(R)` = cache 的 q=0 块（静态模型重建）
3. 对 `q = 1..q_max`：`comm_q = real_space_commutator(cache.blocks[q], hamR)`，
   `T_eff(R) += comm_q(R)/(q·ħΩ₀)`（按 R 合并块、去重）
4. Hermiticity 安全检查 + `Model` 组装（orb/atoms 照抄静态模型）

### 4.6 旧路径保留

```rust
/// 现状（k 空间 + 逆 FT）改名为 *_legacy，供交叉验证测试调用
pub fn floquet_effective_model_legacy(
    &self, drive, trunc, k_mesh, options
) -> Result<Model<SPIN, DIM, NoRMatrix>>
```

最终公开 API：`floquet_effective_model` 指向 Bessel 路径，`k_mesh` 参数删除（0.7 预发布允许 breaking）。

---

## 5. 通用性论证（逐情形核对）

| 输入情形 | 通用表示 | Bessel 路径处理 |
|---------|---------|----------------|
| 线偏振单束 | 单模，`a_α ∈ ℝ^D` | `R_α, δ_α ∈ {0, π}` |
| 圆偏振单束 | 单模，`a_α = a₀(e₁ ± i e₂)/√2` | 一般复 `z_α` → `R_α, δ_α` |
| 椭圆偏振 | 单模，任意复 `a_α` | 同上 |
| 多束同频相干 | 多模 `l_α = 1` | 递归卷积自动含束间干涉（交叉 `m_α` 项） |
| 多频公度 | 多模 `l_α = p_α` | 选择定则 `Σ l_α m_α = −q` 自动处理 |
| 多束非相干 | **不合并**：逐束构造模型后相加 | 调用方负责（与 DFT 路径语义一致） |
| 不可公度频率 | 超出范围（需多维 Floquet） | 显式返回错误（检测基频不可约） |

---

## 6. 测试计划（每层独立 + 交叉验证）

1. **Bessel 函数**：`bessel_j` 对照查表值（`m=0..20`，`R=0.1..5`），误差 < 1e-12
2. **单模系数**：`C_q = (−i)^q J_q(R)e^{+iqδ}`（复振幅情形的正确相位；`δ ∈ {0,π}` 时退化为实形式）显式公式 vs `bessel_peierls_coeffs`
3. **多模混频**：`l=(1,2)`、`l=(1,3)` 的 `C_q` vs 时间网格 DFT（误差 < 1e-10）
4. **四束 tetrahedral**：`C_q` vs DFT；`T_eff` vs legacy k 空间路径（块级比较，误差 < 1e-10）
5. **规范不变性**：同一物理位点不同代表点 → 相同能带（现有 `supercell_fold_preserves_physics_across_gauge_choices` 模式）
6. **Hermiticity**：`T_eff(R) − T_eff(−R)† = 0`（逐块断言）
7. **截断收敛**：最小 margin 翻倍 → 系数变化 < 1e-12（自适应截断在更小 margin 下即应稳定）
8. **性能 smoke**：100×100 原胞、q_max=2 的 Bessel 路径 vs legacy 路径计时比（目标 >100×）

---

## 7. 兜底与回退规则

- `R_α > R_max`（如 8）或模数 > 8：自动回退 `TimeGrid`（保持通用性）。回退网格的
  `n_time` 按该键的 Nyquist 带宽 `2·Σ_α |l_α|·M_α(R_α) + 4` 自适应放大（`M_α` 与
  Bessel 路径同一自适应截断，margin 48 起步），避免大谐波下 DFT 混叠；超过
  `2^20` 点时截断并 warn-once。
- `bessel_peierls_coeffs` 内所有整数算术 `checked_*`，溢出报错不静默
- 不可公度频率检测：`l_α` 集合的 gcd ≠ 1 且存在非 1 基频 → 显式 `TbError`
- Bessel 与 DFT 的 `C_q` 在测试中交叉验证，发布前默认后端以测试通过为准

---

## 8. 实施顺序（每步可提交）

1. `bessel_j` + 单模 `C_q` + 测试 1–2
2. 递归卷积多模 `bessel_peierls_coeffs` + 测试 3
3. d 去重缓存接入 `floquet_harmonic_cache`
4. `real_space_commutator`（两个卷积 `(AB)(R)`、`(BA)(R)`，均 zgemm）+ 单 q 测试
5. `floquet_effective_model_bessel` 组装 + 测试 4–7
6. legacy 保留与公开 API 切换 + 性能 smoke + SKILLS.md/README 文档更新
   （最终 API 定型后一次性更新）+ spinful 模型交叉验证测试
