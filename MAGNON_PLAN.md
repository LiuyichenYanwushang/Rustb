# MAGNON_PLAN — 在 Rustb 中实现磁振子模块

状态：历史设计与里程碑记录，以下“完成”仅描述当时的局部交付，不代表通用 Floquet 算法已完成。

当前通用微观近似改为 `Magnon::floquet_superexchange`，对已有实空间交换施加二阶逐键修正；
旧 Hubbard 高阶方案及撤换说明见 [MAGNON_DRIVEN_PLAN.md](<MAGNON_DRIVEN_PLAN.md>)。
蜂窝闭式、双层解析 LSWT 和用周期传播子求准能量的唯象 `MagnonSambe` 是独立路径。
下表及附录不作为当前实现或验收的权威说明。

## 1. 目标与范围

在 `src/magnon.rs` + `src/magnon/` 中实现与电子结构模块并列的磁振子功能：

1. `Magnon<DIM>` 结构体：晶胞、磁性位点、自旋量子数、参考自旋取向、参考磁结构
   （单 Q 传播矢量 / 旋转轴 / 量子化轴），以及实空间交换张量 `J_ij(R)`（3×3），
   存成与 `Model::ham` 同构的 `(nR, 3n, 3n)`。
2. 玻色子（paraunitary / Colpa）磁振子能谱、态密度、k 路径求解；复用
   `kpoints::gen_kmesh`、`kpath::interpolate_k_path`、`ndarray_lapack::eigh_x`、
   `TbError`、`Dimension` 与 `Kpath` trait。
3. 自研 TB2J 式交换参数提取：从自旋极化的 `Model` 出发，复能量格林函数 +
   等值线/极点积分 + A 张量提取，直接构造 `Magnon`。
4. "hr 形式"的磁振子哈密顿量文本输出与读回。

不做：自旋动力学、蒙特卡洛、有限温磁振子。

## 2. 物理约定（与 TB2J 对齐，已确认）

自旋哈密顿量（TB2J convention）：

```
E = -Σ_i K_i (S_i·n_i)^2 - Σ_{i,j,R} S_i·J_ij(R)·S_j(R)
```

- `|S_i| = 1`：存的是 TB2J 的**单位矢量景观系数**，单位 eV；
- 求和遍历**有序对**：`(i,j,R)` 与 `(j,i,-R)` 都写入 `ham`，一条键的能量是
  两条之和（没有 1/2 因子）；
- `S_i`（自旋量子数，ħ 单位）= `|m_i|/2`（g=2，轨道矩淬灭）；
- 求解器内部把景观系数换算成算符交换 `J^op = J/(S_iS_j)` 后套线性自旋波，
  与 TB2J 的 `Magnon.Jq`（除 `S_iS_j`、A/B 乘 `√(S_iS_j)`、C 乘 `S_j`）等价。
  铁磁链的校验式：`ω(q) = (2/S)[J(0) - J(q)]`，单键景观耦合 `j`（每方向）时
  `ω(q) = 4j(1-cos q)/S`；
- 交换张量分解：`J = J_iso·I + J_ani(对称) + D_antisym`，
  `D_antisym = [[0,D_z,-D_y],[-D_z,0,D_x],[D_y,-D_x,0]]`；
- 单离子各向异性放在 `(R=0, i=i)` 的对角块里（`k1·e⊗e` 或完整 3×3 张量）。

## 3. 数据结构

```rust
pub struct MagnonReference<const DIM: usize = 3> {
    pub q: Array1<f64>,                // 单 Q 传播矢量（约化坐标），默认 0
    pub rotation_axis: Array1<f64>,    // 参考结构旋转轴（TB2J 的 n），默认 x
    pub quantization_axis: Array1<f64>,// 局域系量子化轴（TB2J 的 uz），默认 z
}

pub struct Magnon<const DIM: usize = 3> {
    pub lat: Array2<f64>,     // (DIM, DIM) 行 = 晶格矢量，Å
    pub orb: Array2<f64>,     // (n, DIM) 磁性位点分数坐标
    pub spin: Array1<f64>,    // (n,) S_i（ħ 单位）
    pub axis: Array2<f64>,    // (n, 3) 参考自旋取向（单位矢量，Cartesian）
    pub ham: Array3<f64>,     // (nR, 3n, 3n) 交换张量，eV
    pub hamR: Array2<isize>,  // (nR, DIM) 格矢，第 0 行为 R = 0
    pub reference: MagnonReference<DIM>,
}
```

- `nsta() = 3 * nspin()`，位点 `i` 的笛卡尔分量占 `3i..3i+3`（TB2J 的 `i3j3`）；
- `R=0` 的 `3×3` 对角块放单离子各向异性；
- 不变式：`J_ij^{ab}(R) = J_ji^{ba}(-R)`，`-R` 必须也在 `hamR` 中，格矢唯一，
  `nR ≥ 1`，格矢分量可求负（排除 `isize::MIN`），晶格有有限逆。

## 4. API

```rust
Magnon::<DIM>::new(lat, orb, spin, axis, ham, hamR, reference) -> Result<Self>
magnon.validate()?;
magnon.rec_lat()?;                        // 与 Model::rec_lat 同公式
magnon.jtensor(i, j, iR) -> Result<[[f64; 3]; 3]>;

// M1
magnon.gen_jq(&q)?;                       // 3n x 3n 复 Hermitian
magnon.gen_bdg(&q)?;                      // 2n x 2n（局域坐标系）
trait MagnonSolve { magnon_band_onek, magnon_band_all }
impl Kpath for Magnon<DIM>;               // 复用 k_path
magnon.dos(kmesh, wmin, wmax, nw, sigma)?;

// M2-M4: inputs share one energy reference, with the common E_F shifted to 0.
let params = ExchangeParams::new(&kmesh);
let magnon = Magnon::from_model(&spinful_model, &params)?; // Model<true, DIM, R>
let pair = CollinearModel::new(up, down)?;
let magnon = Magnon::from_model(&pair, &params)?;         // CollinearModel<DIM, R>

// M0 已完成
magnon.write_hr(dir, prefix)?;            // <prefix>_magnon_hr.dat + <prefix>_magnon_meta.toml
Magnon::<DIM>::read(dir, prefix)?;
Magnon::<DIM>::from_hr(dir, prefix, lat, orb, spin, axis, reference)?;
```

当前转换优先使用目标类型的构造器
`Magnon::from_model(&source, &params) -> Result<Magnon<DIM>>`：
支持 `Model<true, DIM, R>` 和 `CollinearModel<DIM, R>`，均通过共享引用借用输入。
公开 trait `ToMagnon<const DIM: usize>` 提供
`to_magnon(&self, &ExchangeParams) -> Result<Magnon<DIM>>`，每种源类型的数值实现
仅在该 trait 中保留一份。`Magnon::from_model` 以 `ToMagnon<DIM>` 为约束并直接转发；
两个源类型原有的 inherent `to_magnon` 方法也保留为薄封装，通过
`<Self as ToMagnon<DIM>>::to_magnon(self, params)` 转发，不复制数值逻辑。
调用构造器或具体源类型的 `source.to_magnon(&params)` 均无需导入 trait；
公开的 `ToMagnon` 供泛型约束与 UFCS 使用。
迁移时移除 `Magnon::from_collinear`，不保留兼容别名。
`CollinearModel::new(up, down)?` 校验共同晶格、按序原子元数据、轨道数与按序投影；
该配对构造器允许两个通道的实空间支持独立，但 `Magnon::from_model` 仍要求完全相同
且顺序一致的 `hamR`，不一致会被拒绝。两个通道必须使用共同能量参考并共同对齐到
`E_F = 0`，不可各自独立平移能量。提取结果沿用 up 通道几何、共同原子元数据与
`atom_of_site` 映射；此次迁移不改变交换提取算法或通道中心不同的既有警告。

## 5. TB2J 算法映射

共线情形（先把两个电子通道共同对齐到 `E_F = 0`；`Δ_i = H↑_{ii} - H↓_{ii}` 是 on-site 全轨道块）：

```
J_ij(R) = (1/4π) Im ∫_{-∞}^0 dε Tr[ Δ_i G↑_ij(R,ε) Δ_j G↓_ji(-R,ε) ]
G^σ_ij(R,z) = (1/Nk) Σ_k [z I - H^σ(k)]^{-1}_{ij} e^{-2πi k·R}
```

实现步骤：

1. k 网格取**奇数**、Γ 中心；R 盒子 = 与网格共格的 `±(n_i-1)/2`；
2. 每个 k 做一次本征分解，`G(k,z) = Σ_n C_n C_n† / (z - ε_n)`（不需求逆）；
3. 傅里叶变换到 R；
4. 能量积分：
   - `MagnonContour::Cfr { temperature_kelvin }`（默认）：费米函数 CFR 极点求和；
   - `MagnonContour::Semicircle`：上半平面圆弧，使用 `emin`、`emax` 和 `nz` 做 Simpson 求积；
5. 共线入口提取各向同性 J；下文的非共线 J/DMI/Jani 是独立的规划；
6. 写回 `Magnon`（`(i,j,R)`/`(j,i,-R)` 都填）。

非共线（spinor）情形（M4）：
`p_i = (h_i^x σ_x + h_i^y σ_y + h_i^z σ_z)·ê_i`，
`A^{uv} = (1/π) Tr[p_i σ^u G_ij p_j σ^v G_ji]`，
`J_iso = Im(A^{00}-A^{xx}-A^{yy}-A^{zz})`，`J_ani^{uv} = Im(A^{uv}+A^{vu}_{-R})`，
`D_u = Re(A^{0u}-A^{u0})`。

## 6. hr 文件格式（已实现）

`<prefix>_magnon_hr.dat`：纯数值，无元数据，wannier90 风格但不逐字节兼容。

```
# Rustb magnon HR v1: ... exchange_convention=unit-vector ...
<nsta> <nR>
<Rx> <Ry> <Rz> <a> <b> <J>          （1 基；共 nsta*nsta*nR 行，逐条恰好一次）
```

读取规则：忽略 `#` 与空行，接受 CRLF 与制表符；首行必须恰好两个整数；矩阵元必须
有限；`DIM<3` 时越界平面分量必须为 0；重复条目、缺条目、多于 `nR` 个格矢都报错；
首个格矢必须是 `R=0`（`validate` 检查）。

`<prefix>_magnon_meta.toml`：

```toml
format_version = 1
dim = 1
nspin = 1
energy_unit = "eV"
spin_unit = "hbar"
exchange_convention = "unit-vector"
lat = [[3.0]]
orb = [[0.0]]
spin = [1.5]
axis = [[0.0, 0.0, 1.0]]
q = [0.0]
rotation_axis = [1.0, 0.0, 0.0]
quantization_axis = [0.0, 0.0, 1.0]
```

`write_hr` / `read` 互相校验 `format_version`、`dim`、三个约定字符串与 `nspin`；
`prefix` 必须是纯文件名主干（拒绝空、`..`、含路径分隔符）。

## 7. 里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M0 完成** | 结构体 + `validate` + `jtensor`/`rec_lat` + 参考结构 + hr/meta 读写 | 33 单元测试 + 2 doctest；fmt/clippy 干净；all-targets 编译 |
| **M1 完成（共线）** | `gen_jq` / `gen_bdg` / Colpa（本征分解版）/ `MagnonSolve` / `Kpath` / DOS | 一维 FM 解析色散（`4j(1-cos)/S`）、两子格 AFM（`4|j|/S·|sin(pi q)|`）、SIA 能隙（`2K/S`）、120° 环的 3 个精确 Goldstone 零模、DMI 下 BdG Hermitian、串并行一致、DOS 归一 |
| **M1b 完成** | 非共线参考结构的冻结磁振子（精确 Hessian）交叉验证 | 结论：TB2J `Hq` 没有缺项。3 格点 120° 环在 1D 双分链上本就是**鞍点**（真实基态是 Néel），精确 Hessian 在同一 q 处给出同样的负本征值；改用三角格子单 Q（Q=(1/3,1/3)）后 BdG 处处半正定，磁 Γ 点在化学区折叠到 q=0 与 q=Q 两处为零，q=(0.25,0) 的 ω 与 TB2J 逐位一致（0.08432740427116 eV） |
| **M2 完成** | TB2J 共线提取：`Magnon::from_model` 统一接收 `Model<true, DIM, R>`（自旋分块）或 `CollinearModel<DIM, R>`（up/dn 模型对）+ `ExchangeParams`（semicircle 等高线 + Simpson） | 两格点反铁磁链的 J(0)、J(±1) 与独立 numpy 实现逐位一致（1e-12）；两个入口互相一致（1e-14）；提取出的 `Magnon` 在磁 Γ 点零模、区界有隙；非共线输入被拒绝 |
| **延期** | 非共线 spinor 提取（DMI/J_ani，原目标 M4 剩余项）：已完成 `A^{uv}` 机械部分与自旋主序索引修正（`state = spin*n_orb + orbital`，`P_i` 已验证）、归一化关系（`A00-ΣAaa = 4 Tr[P G↑ P G↓]`）与 TB2J 参考基准（`scripts/tb2j_ncl_reference`：`J_iso=0.9450`、`DMI=(0,1.8671,0) meV`）。**未完成**：`A^{uv} → 3×3 张量` 的组装约定——共线极限给出 x/y 与 z 通道不一致（ratio −4/3 vs +1.38），说明 TB2J 的 `Jiso`/`Jani` 并非简单的 `(Jiso/3)δ + Jani`，需要一次专门的推导而不是限定时间的试错 | 96 个 magnon 单测不受影响 |
| **F3c 完成** | 解析系数 ↔ 张量求解器一致性：`DrivenExchange::honeycomb_magnon` 用 `J_land = −(S²/2)J_paper`、`SIA_zz = −KS²`、`dmi_land = χS³/2`（次近邻列表是 `{b1,−b2,b3}`，手征项需 `(+,−,+)` 变号） 构造 `Magnon`，其 `magnon_band_onek` 复现式 (6)（四个 k 点到 1e-9 eV；排序谱在 K/K′ 相同且有分裂，全局 χ 符号在排序谱中不可观测，符号由解析 `D_k` 测试钉住） | 95 个 magnon 单测 |
| **F3b 完成** | 双层磁振子谱（SM §III.D）：`BilayerSpinWave` 逐式转写 `A1(k)/A2(k)/C(k)` 与两个自旋扇区的 4×4 BdG（`sector_bands` 保留自旋标签） | 铁磁层在 Γ 有精确零模且平带被抬起；`χ=0` 时两谷严格简并；`χ≠0` 时 `ω_↑(K)=ω_↓(K′)`（扇区在两谷互换，奇宇称），同谷内分裂 >1e-6。92 个 magnon 单测 |
| **F3 完成（静态）** | 论文双层模型（SM §Bilayer）：`BilayerExchange` + `BilayerParams`，静态极限逐式复现 `J1/J2/J⊥/J×/K^(1..3)/χ`，`t⊥→0` 退化为单层、U=0 拒绝 | 89 个 magnon 单测。待办：双层磁振子谱（SM §III.D）与驱动下的 `J⊥(λ)/J×(λ)` |
| **F2b 完成** | 论文式 (6) 的角依赖验收：`Δω(k)` 绕 Γ 采样得三条零交叉（cos3θ，f-wave）、`k→−k` 严格反号 | 88 个 magnon 单测。遗留：χ/K → `Magnon` 张量的标定（J_land = −(S²/2)J_paper、dmi_land = χS⁴ 已测出，单离子项与 DMI 的正反号写入仍需修正）、论文双层模型（SM §Bilayer） |
| **F2 完成** | 磁振子 Floquet 接口（与电子体系同名同参）：`MagnonFloquet::floquet_model`（Sambe，光子阶梯到 2N）、`MagnonSambe::{quasienergies_at, quasienergy_bands, bdg_at_time, propagator_at}`、`Magnon::floquet_effective_model`（order 0）、`fold_quasienergy` 复用 | 准能带用**单周期传播子**（`i dΨ/dt = g H(t) Ψ`，正范数筛选）而非 Hermitian/Sambe 对角化：零驱动精确复现未驱动能带（0 与 ±Ω,±2Ω 不劈裂）；Ω≫带宽时收敛到 order-0 有效模型（5e-3 内）；折叠范围、光子阶梯间隔 Ω、order≥1 拒绝均有测试。80 个 magnon 单测 |
| **F1 完成** | 驱动交换（论文 Odd-Parity Magnons）：`DrivenHubbardParams` + `DrivenExchange::honeycomb`（Floquet–Sambe Löwdin 展开的二阶 + 四阶 F/B/G 求和）、蜂窝 `A_k/B_k/D_k` 与 `ω_{↑,↓}` | 静态极限复现 `J1=4t²/U−16t⁴/U³`、`J2=4t⁴/U³`、`χ=0`；`J1/J2` 螺旋度偶、`χ` 奇；K/K′ 分裂反号、整体奇宇称。待办：χ→`Magnon` 张量映射（`D_z=Sχ` 的归一化待定）、Sambe 接口（`MagnonFloquet`/`MagnonSambe`/`floquet_effective_model`） |
| **M4 完成（部分）** | 磁矩解析：`Magnon::from_model` 的两种输入未给磁矩时由两通道占据数推 `m_i = Tr(ρ↑−ρ↓)`（TB2J `spinat`），`S_i = |m_i|/2`、轴 `±z`；显式给的轴做一致性检查（全局翻转允许，逐位点矛盾报 `moment_direction_consistent`） | 真实数据：不传磁矩时得到 ±2.7854 μB，与显式传入时 J 完全一致（24.78249 meV）；混合轴测试被拒绝；两格点模型导出的轴与 J 一致 |
| **M3 完成** | CFR 极点积分（`MagnonContour::Cfr`，默认 600 K）+ 半圆可选；`wannier90` 空 projections 时按 Wannier 中心就近分配轨道 | 两格点 gapped 模型：CFR = −2.5774 meV 与 TB2J 逐位一致、精确积分 −2.5771；**真实 MgFe6Ge6（55 WF，1121 R，9×9×5）所有壳层与 TB2J 前 4–5 位一致**（2.631 Å 24 键 +24.78249 对 +24.7825；4.256/4.327/4.556/5.004/5.064/5.261/6.235 全部吻合，5.261 Å 的离散 −0.0087~+0.6901 也一致） |
| M4 | 非共线 spinor 提取（4×4 A） | 共线极限一致；DMI 符号表 |
| M5 | 单 Q 旋转（参考结构进入 `gen_jq`）、能带图、示例 | 螺旋结构 Goldstone；文档 |
| M6 | 对抗式 review 收敛 → 提交 | AGENTS.md 流程 |

## 8. 构建与验证命令（本机沙箱特例）

`~/.cargo` 与父目录 `/home/liuyichen/TB_rs` 对 agent 只读，且父目录是 Cargo
workspace（成员含 `cryspglib`，其 git 源在缓存中缺失）：

```bash
cd /home/liuyichen/TB_rs/Rustb
CARGO_TARGET_DIR=$PWD/target cargo --offline \
  --config 'patch."https://github.com/LiuyichenYanwushang/cryspglib.git".cryspglib.path="/home/liuyichen/TB_rs/cryspglib"' \
  test --features openblas-system --lib magnon
```

`cargo clippy` 会吞掉 `--config`，用
`RUSTC_WORKSPACE_WRAPPER=clippy-driver` + `cargo check` 等价替代。

## 9. 已决事项

- **能标**：按 TB2J（用户确认）：存单位矢量景观系数，求解器除 `S_iS_j`。
  额外提供 `operator_exchange()` 风格的换算入口（M1）。
- **存储**：单一 `Array3<f64>`，形状 `(nR, 3n, 3n)`，与 `Model::ham` 同构。
- **IO**：矩阵文件纯数值 + TOML 元数据旁车，记录单位与约定字符串。
- **求解接口**：新增 `MagnonSolve` trait，`Magnon` 实现 `Kpath`；**不**实现电子
  `Solve`（BdG 只有 n 个物理模式、本征矢是 Bogoliubov 系数，契约冲突）。
- **Q 处理**：`MagnonReference` 现在存字段，旋转在 M5 接入 `gen_jq`；M1 先做
  `q = 0` 与"胞内已含磁结构"的情形。
- **`jtensor` 返回 `[[f64; 3]; 3]`**（值语义，无分配），调用方不接触打包下标。
- **`write_hr_body` / `write_meta` 私有**，公开面只有 `write_hr` / `read` / `from_hr`。

## 10. M0 review 记录

第一轮 review 提出 8 项阻塞项，已全部处理：

1. `validate` 在 `nR = 0` 和 `isize::MIN` 格矢上 panic → 提前拒绝，`checked_neg`；
2. 解析器重复/缺失条目、`DIM<3` 越界分量、`DIM>3` 下标越界 → 逐条位图 + 显式拒绝；
3. `nsta=0` 时的超大分配 → 正数检查 + 尺寸上限（对齐 `wannier90::matrix_size`）；
4. 约定表述矛盾 → 模块文档、计划与 meta 字段统一为 TB2J 单位矢量约定；
5. 自旋单位未标注 → meta 增加 `spin_unit = "hbar"`；
6. 缺 TB2J 参考结构字段 → 新增 `MagnonReference`（`q`/`n`/`uz`）；
7. `jtensor` 返回分配、IO 辅助函数暴露 → 改为定长数组、降为私有；
8. 文档与实现不一致（格式、`from_hr` 签名、模块表、SKILLS.md）→ 同步更新。

其余非阻塞项（原子写、BufWriter、前缀校验以外的大文件优化等）留待后续里程碑。

### 第二轮 review（收敛）

独立复核确认 8 项全部 FIXED，无新增阻塞问题，且用数值把约定钉死：

- 逐条核对 TB2J commit `6a4f9e`（2026-11-17）与 release 0.9.22 的 `magnon3.py`：
  `Jq` 含 `1/(S_iS_j)` 与 `e^{-2πi q·R}`，`Hq` 的 A/B 乘 `√(S_iS_j)`、C 乘 `S_j`，
  默认 `n = x`、`uz = z`、`Q = 0`。本实现的约定与两者一致。
- 复现 TB2J `Magnon.Hq` 于 1-site FM 链（S=1.5, j=0.01）：q=0.25 得
  `2.6666666667e-2`、q=0.5 得 `5.3333333333e-2`，与 `4j(1-cos q)/S` 及
  `(2/S)[J(0)-J(q)]` 到机器精度一致；q=0 为 3.5e-18。
- 两格点亚铁磁（S=[1,2]、各向异性键、易轴 SIA）：TB2J `Hq` 与独立 LL 线性化
  结果两支在 8 个 k 点上一致到 1.6e-11。
- 推导核对：`E=-Σ u_i·J·u_j` 代入 `S_i=S_i u_i` 得 `J^op=J/(S_iS_j)`；HP 展开给
  `ω(q)=2S[J^op(0)-J^op(q)]`，均匀 S 时即 `(2/S)[J(0)-J(q)]`。

按 review 意见补充的非阻塞修复：`check_prefix` 不再误拒含 `..` 的普通文件名；
解析器把"首格矢非 R=0"报为 `FileParse` 并给出普通向量；`validate` 消息不再打印
ndarray 调试串；`jtensor` 文档限定"对已校验模型"；补充 6 个测试（无效模型不落盘、
超限头部、多于 nR 个格矢、R=0 非首、DIM=3 往返、meta 中坏参考结构）。

测试计数：**39 单元测试 + 2 doctest**。

### M1 review（收敛）

第二轮评审（M1 求解器）提出 1 个阻塞项 + 7 条非阻塞意见：

1. **阻塞（已修）**：单 Q 参考结构下，`jq_of` 对每个块只做右乘旋转；张量若与旋转不对易，
   `(i,j,R)` 与 `(j,i,-R)` 两个块不再互为转置，`J(q)` 非厄米，下游 `eigh_full` 只读一个
   三角会静默给出错误频率。修复：傅里叶求和后取厄米部分
   （实二次型只感知对称部分），并新增各向异性张量 + Q≠0 的厄米性测试。
2. 正定判据改为相对容差 `1e-10·max|λ| + 1e-13`（纯 Goldstone 点全零矩阵需要绝对下限）。
3. 局域系退化回退的文档改正：TB2J 的回退是 `R = I; R[2] = v`（对 +z 与本实现一致，
   对 −z 是左手系/非正交的），本实现用 Gram-Schmidt 正交标架，谱一致、Bogoliubov 矢量差相位；
   新增非坐标量子化轴的回退测试。
4. 补充测试：非等自旋铁磁链（S=(1,2)，与 TB2J 参考值逐位一致）、`gen_jq` 与手算傅里叶和、
   DOS 参数错误路径。
5. 评审确认：本征分解版 Colpa 与 Cholesky 版对半正定 `H` 等价（相似变换 `w = gKu`），
   `1/√(S_iS_j)` 与 `2/S_i` 因子正确，`A2 = U†conj(Jq)U` 正确。
6. 遗留非阻塞：`bdg_of` 每个 q 重算 `J(0)` 与局域标架（可预计算）、DOS 重复 validate、
   并行路径的一次性拷贝。

