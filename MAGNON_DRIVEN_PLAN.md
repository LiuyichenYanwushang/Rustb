# 通用实空间 Floquet–Hubbard → Magnon：Goal 与执行计划（方案 A）

> 创建：2026-11-18。用户已要求按本计划逐步实现。
> 本文件是新方案 A 的执行入口；MAGNON_PLAN.md 是旧模块历史记录，不是新目标的完成证明。
> 状态：规划与 M0.1 执行中。M1–M6 尚未实现，不得沿用旧目标的 complete 状态。
> 持久目标 ID：`goal-c6dd9882-7e8e-4c0a-af53-a8ddc7ea8c8b`；恢复时先 get_goal
> 读取最新 revision/activation，不把文档中的状态当成执行授权。

## 1. 可直接交给 lead 的 Goal

在 Rustb 中实现一个维度通用、完全在实空间生成有效相互作用的周期驱动单轨道
Hubbard 强耦合算法。输入已有电子 Model/HubbardModel 和 FloquetDrive，在半满、
自旋无关 hopping、统一正 U 的前提下，计算 hopping 的二、三、四阶 Hermitian
有效自旋哈密顿量，包括二体交换、有向三自旋标量手性和四自旋项。自动处理周期
晶格路径、费米符号、双 doublon、非零光子单占据中间态以及折叠/归一化修正，
禁止按 honeycomb/square/bilayer 分类硬编码生产公式。

对指定的共线磁参考态做通用实空间 LSWT，尽可能返回普通 Magnon<DIM>，复用其
BdG、能带、k-path 与 DOS。支持 DIM=1/2/3，复用已有任意相干多模偏振的 Peierls
谐波后端。提供与静态 TB2J Magnon 的差分合并，未驱动交换只计一次。

完成必须有：独立二聚体/三角簇/四点环 oracle、蜂窝解析系数与带标签矩阵回归、
方格子与驱动双层、3D 与超胞等价、光子截断收敛、结构化错误、release 测试、
fmt/clippy、rustdoc/SKILLS/示例同步和收敛的对抗式 review。阶段失败不可用放宽
容差、排序后谱、移除难例或声明整个目标完成代替解决。

不在本期：AC、多轨道 Hund/SOC、任意自旋模型的高阶 van Vleck、拓扑/热输运、
边缘态、非共线 spinor TB2J 提取。保留已有功能，但不把它们作为本目标验收门槛。

## 2. 必须钉死的物理边界

1. 只有静态 Magnon/J 不能唯一反推电子 AB 修正；电子 hopping 和 U 是必需输入。
2. 任意晶格不保证非零 odd-parity。对称性禁止的 chi 或分裂必须允许为零。
3. 单轨道半满的微观自旋是 S=1/2。默认生成和 LSWT 均采用该自旋长度。
   S>1/2 材料应用是另一个显式的唯象算符提升契约：先冻结 J/chi/四自旋的
   多项式代表元、其系数单位/标定以及使用者提供的材料自旋，再做 LSWT；
   S=1/2 算符恒等式下等价的表达式在 S>1/2 不必等价，不能自动随意选代表元。
   此提升不能宣称是原单轨道微观推导；未冻结/未验收时拒绝 S>1/2 合并。
4. 多 Wannier 轨道属于同一个磁原子的真实数据不等于“一轨道一磁位点”。
   本期不自动降维，不宣称适用任意多轨道第一性原理 Hamiltonian。
5. 首期只支持统一 U>0、半满、共同标量 onsite（可去掉整体能量零点）、
   hopping=标量×自旋单位矩阵。不同 onsite、不同 U、SOC、自旋依赖 hopping
   必须在入口拒绝；不修改 HubbardModel 现有更宽的 Hartree–Fock 契约。
6. 允许自旋无关复 hopping；静态磁通可使 chi(0) 不为零，合并时也要扣除。
7. hopping 阶数 2/3/4 与电子 van Vleck 的 1/Omega 阶数不同，不混用选项语义。
8. 主生成器不接受 kmesh，不对周期无限晶格做全系统 Fock 空间指数展开。

## 3. 权威证据与优先阅读

- AGENTS.md：提交前 review、模型约定、后端与 release 测试、公共 API 文档同步。
- src/Hubbard.rs：HubbardModel::bare_model/onsite_u/with_uniform_u；优先复用。
- src/floquet.rs 与 src/floquet/coefficients.rs：FloquetDrive、LightMode、
  bessel_peierls_coeffs 的可见性、缓存、截断尾估计与符号。
- src/model.rs、src/model/build.rs：spin-major 状态排序与 set_hop 自动厄米伴随。
- src/magnon.rs、src/magnon/physics.rs：landscape 有序键约定、局域标架、BdG。
- src/magnon/floquet.rs：仅作现有解析 oracle 和待修基线，不抄其晶格特判进引擎。
- 论文原始 main/SM 附件：先用 glob/附件工具定位并实际读取；记录物理页或 TeX
  行号与方程，不把对话中的方程编号当成已核对来源。找不到时明确报告，不能编造。
  本轮已读原始 SM.tex（附件 `0a3b5777`）2476–2515 行：C 块、up/down Nambu 基、
  down 下左 C*(-k) 无转置，以及物理模式必须正能且正范数；可作 T01 来源。
- MAGNON_PLAN.md 历史含过时结论（例如三次 f-wave 过零、Sambe 完成范围、
  spinor 归一化猜测）；引用前重验。历史测试数量不是当前测试证据。

## 4. 约定与数据契约（G0，后续任务共享）

### 4.1 空间/时间/自旋

- 矩阵元 t_ij(R)=<i,0|H|j,R>；Cartesian 行向量 d=(R+tau_j-tau_i)·lat。
- exp(+i 2pi k_frac·R)；电子 spin-major：state=spin*norb+orbital。
- Peierls exp[-i a(t)·d]；C_m(d)=average(exp[+im Omega t] exp[-i a(t)·d])。
- 每条矩阵元的 bra/ket 与费米 c†c 的实际跳跃方向要用二点数值例验证，
  不凭“正向键”文字推断；核对 t_ji^(-m)(-R)=conj(t_ij^(m)(R))。
- 本期 E、U、W=hbar*Omega、t、J、chi、四自旋系数均为 eV；a(t) 为晶格长度逆单位。
- 自旋始终是三分量 Cartesian，DIM 仅表示空间周期维度。
- 光场模式相干相加再取指数；多模不能分别算修正后相加冒充相干驱动。
- M1 二阶公式必须从两个 doublon 方向推导。任意多模下 |C_m(d)| 不一定与
  |C_-m(d)| 相同，不能把圆偏振的单方向4倍公式直接推广；T12须加入同一无向键
  正反取向和双频相对相位破坏m偶对称的回归。

### 4.2 有效算符与 Magnon 的转换

- 微观输出约定：每条无向键一次，H_spin=E0+sum K_ij S_i·S_j
  +sum chi_ijk S_i·(S_j×S_k)+四自旋算符。
- Magnon 存的是有序单位矢量 landscape，E=-sum e_i·J_ij(R)·e_j。
  普通两体转换 J_land=-K_ij*S_i*S_j/2（每个方向都写）。
- 转换三自旋/四自旋时从该 Hamiltonian 的二次展开推导，禁止套蜂窝 S 幂次
  经验式。onsite 与反向键写入不能无意加倍。
- **量子约简先于经典化**：重复位点的算符积先在完整 S=1/2 算符代数中约简，
  再冻结独立基，最后才做 HP/Hessian。例如 (Si·Sj)^2=3/16-(Si·Sj)/2；
  两种等价输入必须产生同一 LSWT，不能对未约简表达式直接做经典 Hessian。
- 真正四点项采用“明确点对乘积的线性组合”作为公开系数含义，循环置换算符
  必须先转换到该约定（含常数/二体项），不能只输出未规范化的 ring 系数。
- 三元组带有序方向；反交换端点导致 chi 变号。3D 不用 z 投影面积挑选
  “正三角形”，使用确定的顶点排序/置换奇偶表示方向。
- canonical key 用整数晶胞平移和单胞位点；不能仅按物理距离、叉积或浮点坐标
  合并路径。周期同一子格不同 R 的位点是不同物理位点。

### 4.3 规范选择与完整四阶表达式

先写推导与可运行小簇参照，再写生产路径引擎。固定一种 Hermitian canonical
Löwdin/SW 约定；说明与论文低能空间内部幺正变换的关系。

令 P 是零光子、单占据模型空间；Q=1-P；R=Q/(E0-H0)，H0 含 U 与光子对角项。
本期入口确保 PVP=0（共同 onsite 已去掉）。候选展开需从原始来源核对：

    H2 = P V R V P
    H3 = P V R V R V P
    H4 = P V R V R V R V P
         - 1/2 { P V R^2 V P, P V R V P }

R 的分母号由本仓库 Fourier/Sambe 约定推导，不与论文 U-mW 的标签机械混用。
Q 必须包含 Nd=0 但光子数非零的态；仅排除回到 P 的中间态。R^2 指算符平方，
不是把最后积分结果再平方。折叠项是矩阵反对易子，不能用单条闭路规则替代。

来源已在本轮实际核对：原始 SM.tex（附件摘要前缀 `0a3b5777`）第 91–158 行，
`eq:AB-Lowdin` 第 135–142 行与上式 H2/H4 一致；第 111–125 行明确区分
非共振/强耦合条件和数值 photon 截断收敛。该段限定二分图，H3 不为此段所证明，
需在 G1 对一般 PVP=0 的展开单独推导/核验。实际附件路径可用 glob `**/SM.tex`
定位，全文路径不作为跨机器可移植契约。

原文一致不等于实现验证；G1 必须通过二聚体精确展开、独立有限 Sambe
block-diagonalization/低能拟合才冻结实现。

### 4.4 M1 通用二阶目标公式（推导已核，数值实现仍 pending）

独立 second-order-design 审计与 lead 的两格点费米符号核验一致：在本仓库
Sambe 对角 +nW、统一 U、共同 onsite 已去掉、自旋无关 hopping 下，一条无向键
的两个 doublon 方向都需累加：

    K_ij = 2 sum_n (|t_ij^(n)|^2 + |t_ji^(n)|^2) / (U+nW)
         = 2 sum_n |t_ij^(n)|^2 [1/(U+nW) + 1/(U-nW)]
    H_ij^(2) = K_ij (Si·Sj - 1/4)

其中 reverse-link 包含反向 R；换元要求 photon 窗口对称。triplet 修正为0、singlet
为-K，因此输出普通 K Si·Sj 时常数须为-K/4。静态退化为4|t|^2/U与常数-|t|^2/U。
只有 |t_ij^(n)|=|t_ij^(-n)| 时才可简化成圆偏振常见的单方向4倍式。
T12 必须以双频相位打破该偶性（例如 a(t)·d=A cos(t)+B sin(2t)）并通过独立
小簇 oracle 及反向键 invariance 验收；本段推导不代替该数值门槛。

## 5. 模块与 API 原则

拟新增生产入口 src/magnon/driven.rs。内部确有独立职责后再按需拆为
bonds/expansion/operators；不先生成空模块或插件/trait 框架。

- 复用 HubbardModel<DIM,R>，不再创造拥有同一电子 Model 的输入容器。
- 拟议入口职责：HubbardModel + FloquetDrive + 强耦合选项 → 有效自旋模型。
- 一个选项类型只拥有实际读到的 hopping 阶数、截断、误差/资源策略。
- 一个有效自旋结果拥有几何与 J/chi/四自旋项及必要的收敛诊断，不复制 bands/DOS。
- LSWT 转换返回普通 Magnon；若表示不了必须报告具体项，而非无声丢失它。
- 公共命名/字段在 G0 API 决策中冻结；以上是职责草案，不是已存在接口或可编译示例。
- 现有 honeycomb 和 BilayerSpinWave 先保留独立 oracle。不要过早改成调用新引擎，
  否则回归测试会变成同源自证；待验收后再决定薄包装迁移。
- 现有 MagnonSambe 是另一条唯象交换键驱动/传播子路径，不是本 AB 微观算法；
  文档明确分开，禁止默认叠加造成两次驱动。

### 5.1 已选择的周期计数路线（T30 先提交设计证明）

采用**顶点诱导有限簇完整 H_eff + 顶点子集 inclusion–exclusion**，不混用
“按过程支撑直接累加已连通权重”。对候选四阶闭合路径的物理顶点集合 C：

    W(C) = H_eff(C) - sum_{proper nonempty vertex subsets D of C} embed[W(D)]

H_eff(C) 使用原无限 hopping 图在 C 上的全部诱导边，所有阶数/折叠项先完整相加；
子集含断开和孤立情形，定义其贡献并用断开两二聚体验证 W=0，不能仅凭预期丢弃。
重复的顶点集合只算一次；路径枚举只负责发现候选 C，不再次累加其结果。
算符分解只在得到 W(C) 后做，且较小算符支撑（例如四点簇生成的二体修正）
仍属于该簇贡献，不能按算符支撑再扣一次。

用所有顶点之一作为平移锚点，枚举将该锚点 R 移到零的表示，排序 (site,R) 后取
字典序最小 canonical key；每个平移轨道每胞只累加一次。不同旋转/空间嵌入不按
抽象图同构合并，不除以顶点数/automorphism 阶数，不额外按 root 再加权。
T30 必须证明这种选择覆盖完整占据恢复过程的支撑，包括多电子置换、回退路径、
四跳非简单环以及诱导弦边，而不是只枚举单电子简单闭环；并与原胞/双倍超胞逐键
还原一致。若此证明失败，先更新设计和 review，不能暗改成第二种计数法。

### 5.2 已选择的材料延拓契约（非微观结论）

先独立完成 S=1/2 输出。材料路径显式选择**保留量子自旋算符多项式系数**：
在 T23 冻结的不同位点 J/chi/四点积代表元上，保留各系数（eV），将自旋大小
替换为用户明确提供的材料幅度，再做半经典 HP；不同时声称保留单位矢量能量泛函。
此选择是唯象 ansatz，不是从单轨道推导出真实多轨道材料的唯一映射。

没有显式接受该契约的调用，S 与 1/2 不匹配则结构化拒绝。材料自旋列表的 shape
校验不算物理映射验收。T43 要用非零驱动、非 1/2、非等自旋，分别验证二体
sqrt(Si Sj)/纵向 Sj、chi 的 Si sqrt(Sj Sk)、四体 spectator Si Sj 权重与独立
Hessian 一致；零驱动合并恒等只能验证防重计数，不能验证该自旋延拓。
材料外加各向异性与静态 TB2J 保持独立，不伪装成 SU(2) 单轨道 Hubbard 的输出。

## 6. 任务 DAG 与验收（只认下面的阶段状态）

状态枚举：pending / in_progress / review / verified / committed。
verified 必须附命令、退出码、测试数量/残差、review 处理和具体 commit（提交后）。

| ID | 依赖 | 任务与最小交付 | 必须通过的 gate |
|---|---|---|---|
| T00 | 无 | 本计划、证据索引、旧完成范围纠正、当前基线 | 干净树起点、独立记录旧缺陷；不宣称已实现 A |
| T01 | T00 | M0.1 双层 down 下左块共轭修复 | 修复前 red；各扇区 Hermitian；非等自旋耦合/chi；独立 S153 矩阵；一般 k 的标签对称性 |
| T02 | T00 | 负 lambda 与 public spin 参数的结构化校验 | 负值按明确语义取模/拒绝，无 panic；NaN/Inf/零自旋；所有共用调用点覆盖 |
| T03 | T00 | 任意 quantization_axis 的局域标架 | 量子化轴不改变物理能谱；V_i=实际磁矩方向；易轴单点/多点/复 DMI 残差 |
| T04 | T01,T02,T03,T05 | 清理虚假测试/范围说明 | 删无断言 probe；测试与示例使用真实Cartesian圆/逆变换、稳健六过零；非 U=1 静态测试；不夸大 Sambe/AC/双层驱动范围 |
| T05 | T01 | 双层求解器数值安全 | shape/有限/Hermitian输入；相对实际能标的虚部/聚类阈值；正范数且正能（Goldstone单独处理）；负正范数能量/微小meV复谱/简并反例；优先复用已有Colpa不增平行solver |
| T10 | T00 | G0 输入契约/API 与实空间键提取 | 审计已有类型后写设计决策；维度/自旋独立/U/onsite 拒绝路径；入口定义半满 P，不从 hopping 推断 filling；不新增多余模型层级 |
| T11 | T10 | 键谐波缓存与周期顶点 canonical key | 共享已验证缓存和 per-link DFT fallback 的最小 crate-private 访问面；reverse-Hermitian；相干模式干涉；椭圆/3D；有限截断和资源算术检查 |
| T12 | T04,T11 | M1 通用二阶交换和 S=1/2 Magnon | 静态 4|t|²/U；有限光场独立 dimer；2D/3D 的双子格 AFM 稳定参考（允许 Goldstone）；不以单离子 SIA 人为开隙 |
| T20a | T00 | 独立二聚体 ED/有限 Sambe 参照脚本（可与旧基线修复并行，不依赖新Rust API） | 静态精确gap；从P追踪而非最低谱；dense H2/H4含folded；双频非偶谐波二阶反例；截断和t缩放证据；不调用生产公式 |
| T20 | T10,T20a | G1 canonical 展开公式及独立小簇参照 | 推导来源；二聚体 singlet-triplet 4t²/U-16t⁴/U³；从 P 追踪有限驱动 Sambe（含按最低值误选副本的反例）与 t 缩放误差；先于生产 H4 |
| T21 | T11,T20 | 小簇 Fock 费米运算和 R/R² | 正则反对易、粒子数、SU(2)、Pauli 符号；Nd=0,m!=0 与 Nd=2；P 排除正确 |
| T22 | T21 | H2/H3/H4 与折叠矩阵项 | 单独对照小簇矩阵（能量零点/模型空间 gauge 对齐）；Hermitian；triangle odd order；square double-doublon；断开簇可加性 |
| T23 | T22 | 算符分解 | 重复位点先做完整 S=1/2 量子约简；固定独立基/秩；J/chi/明确四点积；重构残差；等价表达同约简矩阵；端点置换/静态复磁通；禁止丢弃余项 |
| T30 | T11,T23 | 周期连通簇、嵌入多重数、子簇扣除 | 原胞/超胞、边排序/平移原点不变；完整四阶后断开项消去；无全晶格 Fock 指数构造 |
| T31 | T30 | M2 蜂窝硬门槛 | lambda=0,0.4,1 及第二组 t,U,W；J1/J2/chi、方向和总局域算符；光子截断先收敛；不只比较排序谱 |
| T32 | T31 | M3 方格子/椭圆/非二分图 | 方向分辨 J；square J1/J2/J3/ring 与原文；三角三阶；线偏/圆偏/椭圆；相反 helicity 条件测试 |
| T40 | T03,T23 | M4 通用多自旋实空间 HP 二次展开 | 量子等价表达经约简后同LSWT；finite-difference Hessian/不同位点相干态参照；一阶驻点残差；二体/chi/ring 单项；正确 landscape 因子 |
| T41 | T12,T31,T40 | LSWT → Magnon 与标签诊断 | 蜂窝逐元素 up/down 块；复配对/异自旋；不以排序验证 chi 全局符号；零模与不稳定区分 |
| T42 | T30,T40,T41 | 周期双层和 3D 端到端 | 实际驱动层间/混合簇由统一引擎产生；耦合双层逐块对照；3D 立方/金刚石/非共面键；能带路径/DOS |
| T43 | T41 | TB2J 静态差分合并 | 明确位点/晶胞映射；显式接受§5.2的固定算符系数延拓，否则拒绝非1/2；非零驱动非等自旋权重；原基底不变；零驱动恒等；chi(0)!=0；不重复静态环项 |
| T50 | T32,T42,T43 | M5 收敛/资源/质量验收 | photon M→M+delta、hopping 缩放阶数、密度/配位成本、线程确定性、非共振与弱耦合诊断 |
| T51 | T50 | M6 发布与最终 review | release 全库/示例/rustdoc；fmt/clippy；SKILLS；相同原始公式的独立 oracle 仍存在；review 全挑战收敛 |

### 当前台账

- T00：committed，`6f23dbb`。plan-review 的 R1–R5 已 ACCEPT；plan-rereview
  确认规划 blocker 已解决；plan-final-review 确认 converged、DAG 无循环。
  文档验证：cargo fmt --check、git diff --check 均 exit 0；不代表算法实现。
- T01：committed，`ddc0aa8`；由 bilayer-impl 执行，lead 读取 red/green 原日志复核。
  真实 red：q=(.13,.27) 的 down Hermitian/S153 残差 2.1489297416323814e-4 eV。
  修复后 release focused `bilayer_sw_tests -- --nocapture` exit0，7/7；
  Hermitian=0、oracle最大1.7347e-18 eV、标签残差1.3878e-16 eV、split1.3291e-4 eV。
  Gamma raw最大虚部1.11394e-16 eV，公开sector_bands返回Ok；仍有约4e-11 eV
  的Goldstone微负值，严格正能/零模语义留T05，不宣称已修复。
  bilayer-review 的 finite 加固建议 ACCEPT 后，fresh bilayer-rereview-luna 确认
  converged，无 solver 容差改动。归档：
  `target/goal-a-archives/task-1-reviewed-green-20261118T1427.tar.gz`（tar/gzip校验exit0）。
- T02：committed，`8d3f62d`，validation-impl 执行，lead复核最终diff并集成验证。真实red：
  负lambda=-0.4的validate()未返回Err，`validate_rejects_bad_parameters` exit101；
  早先未匹配测试的过滤器不算证据。两个蜂窝辅助函数已改Result，例子/指南同迁移；
  非有限helicity、自旋/动量/系数、生成输出溢出均结构化拒绝。
  首审/root接受的平方overflow/underflow用归一化差乘和修复，新增1e±200回归；
  负能反例明确A<-|B|但平方radicand>0，不能用平方把不稳定参考伪装成实谱。
  validation-rereview确认converged，F/B/G求和与双层solver容差未变。
  focused release Floquet31/31；lead最终release magnon104/104、example check、
  fmt/diff-check、clippy --lib --tests均exit0。全库warning仍为96/206（含重复），
  不声称全库lint干净。压缩证据：target/goal-a-logs/task-3/T02-negative-red.log.gz、
  T02-final-focused.log.gz、T02-final-example-check.log.gz、integration-*.log.gz，
  均gzip校验通过；未运行仍待T04几何修复的示例本体。
- T03：committed，`4adc8ee`，由 lead 接手；frames-impl 两次在领取任务前失败，无代码产物。
  tilted-uz 的 SIA 虚假不稳定已真实 red，初版修复后 release magnon 102/102；
  首审 frames-review 发现近平行叉积舍入放大，追加triad测试真实 red（纵向正交
  断言失败），ACCEPT 后改为归一化前重投影；并补single-Q+XXZ+DMI 的逐元素
  Nambu规范协变与独立手推色散。最终精确版本release magnon 103/103 exit0，
  fmt/diff-check exit0，clippy --release --lib --tests exit0，改动文件无lint定位；
  全库仍有96个lib/206个lib-test warnings（含重复），未宣称全库lint干净。
  frames-rereview 对投影/手推谱确认无问题，其“0..2未覆盖2x2”的挑战经明确REBUT
  后已撤回；fresh frames-final-review 确认 converged。归档：
  `target/goal-a-archives/task-4-final-magnon.tar.gz`、`task-4-final-clippy.tar.gz`，
  均gzip校验exit0；两个red也分别保留压缩证据。
- T04a：in_progress（task-7，lead，独立示例子任务，不代表整个T04完成）。
  真实red：旧cart·B导致roundtrip首点[.1385,.0585]而非[.05,0]；逆矩阵修复后
  两个example helper tests通过。首审example-geometry-review接受几何/周期计数，
  提出的样本长度、subtraction-noise guard、prime计数注释挑战已ACCEPT并修改。
  精确版复测及default/zero/reversed-helicity CLI待T05库源冻结；fresh review中。
- T05：in_progress，solver-safety启动失败后由solver-impl-sol接手（task-6已claim）。
  已审计并接受归一化后复用现有PSD Colpa路线，非零mode证明用y=gKv、只承诺
  Goldstone频率不承诺Jordan零模规范本征矢；先red，不能扩大容差/裁负频掩盖失稳。
- T20a：verified（task-8，独立参照，不代表T20/G1整体或M1完成）。dimer-reference
  实现6态spin-major Fock/有限Sambe/P追踪与polar规范、dense H2/H4 folded。
  dimer-review-luna独立静态审查无阻断，确认固定12步/固定光场/M8与10的边界；
  lead逐段读源并独立OPENBLAS_NUM_THREADS=1 python3 -W error运行exit0，脚本SHA
  ec9a376fba8cbdd997a129342584b3a31d59c9504524dcb9c494a73d39ab3f9e。
  静态残差约4.03e-15eV；双方向K2=1.17139565/eV，错单方向0.90528311/eV；
  H2/H24误差减半比趋向16/64；M8/10在t=.16的canonical矩阵差6.13e-12eV，
  仅为相对2.51e-5eV阶数误差的截断预算核验，不宣称1e-12绝对精度。
  NumPy1.20.3标量逆数规避已保留probe的向量warning，不suppress；全部Q逆数校验。
  lead证据target/goal-a-archives/task-8-lead-check.tar.gz gzip校验通过。
- T10a：只读审计完成（input-design，task-5 completed），不代表T10已实现。
  确认HubbardModel借用接口可复用、Model::validate不保证厄米/支撑、缓存必须走
  validated facade并保留DFT fallback；具体输入冻结仍是T10 gate。
- 其余：pending。实际状态变化在本节逐项补证据，不靠聊天总结覆盖。

## 7. 数值与性能策略（不可凭“性能不是问题”跳过）

1. 四阶闭合占据过程的连通支撑位点上限先证明，再设计 Fock bitset 容量。
   不机械枚举四步可达的所有五点随机游走；回到模型空间需要净占据恢复。
2. 与最大配位、hopping 支撑、光子范围、模式数相关的预算必须量化。
   三重 photon 求和可很大，不能用“10^4–10^5 路径”代替成本估计。
3. 区分单跳 Fourier cutoff、累积中间 photon cutoff、折叠项各自的窗口。
   m2-m1 的谐波可能超出中间 photon cutoff；禁止无声置零或提前裁剪。
4. 非共振检查基于实际可达、非零权的中间态；二阶不误拒不可达的 2U 共振。
   高阶需要同时检查 Nd U + photon*W 的距离和相关 hopping/交换尺度。
5. 强场/近共振/阶数不收敛返回结构化错误或明确诊断；不加任意 eta 隐藏共振。
6. 谐波误差与微扰误差分开。M>=|lambda| 不是收敛证明；固定大 M 也不是证明。
7. oracle 相比用 atol+rtol；接近零的 chi 不用相对误差；正负零不当成角向过零。
8. exact-ED/Sambe oracle 必须在 t→0 从 P 的 2^N 维子空间出发，以 P 投影权重
   和连续子空间追踪选择目标零副本分支，检查秩/投影奇异值；禁止挑“最靠近零的
   2^N 个数”或先折叠所有 eigenvalue 再选。副本截断扩大时验证边界权重与能隙
   收敛，失去可分离子空间时返回错误。对比能隙/谱，matrix 对比需用 polar/overlap
   明确定义同一 canonical 低能空间规范；两种 SW gauge 的矩阵差异不自动等于错误。
9. 对纯实、零驱动、时间反演对称输入才要求 chi(0)=0；复磁通例外显式测试。
10. 有对称性才断言 helicity 翻转与 up(k)=down(-k)；不能强加给任意材料/磁态。

## 8. 子 agent 的通用执行协议

### 8.1 启动模板（lead 填写 ID、范围、依赖）

    你执行 MAGNON_DRIVEN_PLAN.md 的任务 <ID>。
    先读取 AGENTS.md、本计划、相关源码及全部调用方；读 team_task_get 最新 revision。
    确认所有依赖 verified/committed 后 claim，否则报告缺失，不自行绕过。
    输入/输出：<明确类型与上游产物>。
    写范围：<最少文件/区域>。共享目录已有其他 agent；不写范围外，不重置 git。
    验收：<任务行的 gate + 指定 oracle>。先写 red regression，再最小修复。
    不允许：替换 oracle 为生产函数、用排序掩盖标签错误、增容差、改范围、提交。
    若需要更改公共契约，先 send_message 给 lead；由 lead 统一修改 facade/文档。
    运行 release focused tests，记录命令、退出码、数值残差与日志路径。
    完成后回报证据、风险、下游接口；标记 task complete 不代表整个 goal 完成。

### 8.2 必须返回的交接内容

- task ID、依赖版本/commit、实际修改文件、公共签名是否改变。
- 推导与原始来源定位、采用的号/度量/规范/计数约定。
- red/green 的完整命令、退出码、断言失败/通过的数值证据。
- 新增/删除测试清单；oracle 是否调用生产路径（如是则不算独立）。
- 未解决问题与下阶段的确切输入，不写“基本完成/应该正确”。
- 日志/结果归档路径；无凭据/原始大数据写入文档。

### 8.3 并行与合并

- 初期最多两个实现分支：T01/T02 都碰 floquet.rs 时必须串行或约定不交叉区域。
- T03 可与 T01 并行；后续 oracle/推导可与输入层并行，H4 必须等 G1。
- lead 控制公共 facade、error.rs、lib.rs、SKILLS.md 和本计划；指定 owner 后不抢写。
- reviewer 只读 diff 与独立证据，不充当同一实现者的“自我批准”。
- 有 accepted 修改必须换 fresh reviewer 再审；每条挑战写 ACCEPT/REBUT 与理由。
- review 未收敛不提交；“cargo check 后提交”不能覆盖 AGENTS 的强制 review 条件。

## 9. 固定 review checklist

- [ ] 单位、自旋 S、无向/有向键双计数、onsite 写入次数。
- [ ] Fourier/Sambe photon 标签、bra/ket 与实际跳跃方向、helicity 号。
- [ ] H_eff Hermitian、SU(2)、Fock 费米符号与占据恢复。
- [ ] Q 包含非零 photon 的单占据态；H4 折叠项和双 doublon 无遗漏。
- [ ] 周期 canonical key、三元组奇偶、四点项基的独立性、子簇扣除。
- [ ] BdG 每个自旋扇区单独 Hermitian，Delta(k)=Delta^T(-k)。
- [ ] 带标签矩阵/分支比较；退化空间不凭排序或任意 eigensolver 基矢贴标签。
- [ ] 非驻点/能量不稳定/动力学不稳定/Goldstone 区分，正范数不等于正能量。
- [ ] 误差与能标相适应，meV 系统不使用 max(scale,1 eV) 掩盖虚部。
- [ ] DIM 通用且无 honeycomb/square/bilayer 生产分支；3D 不假定平面法向为 z。
- [ ] 反例真实红过；独立参照；无仅打印测试、同源公式自证或固定零输出假通过。
- [ ] 文档精确陈述支持范围；无“任意 Magnon 必然 odd-parity”等过度承诺。

## 10. 验证命令与产物

以实际 workspace 和权限为准；不能复制旧聊天中的成功输出。每个 bash 检查退出码，
有管道用 pipefail；不要 grep/tail 过滤 cargo 错误。常规命令：

    cargo fmt --check
    CARGO_TARGET_DIR="$PWD/target" cargo --offline test --release --features openblas-system --lib <filter>
    CARGO_TARGET_DIR="$PWD/target" cargo --offline clippy --release --features openblas-system --all-targets
    CARGO_TARGET_DIR="$PWD/target" cargo --offline test --release --features openblas-system
    CARGO_TARGET_DIR="$PWD/target" cargo --offline test --doc --release --features openblas-system

本机 git cryspglib 若离线缺失，使用已经确认的本地 patch 配置（见旧计划命令）；
父 Cargo.lock 不可写时请求 exact-command 许可，不修改依赖或复制 checkout 绕过拒绝。
MKL 可用则按 AGENTS 增跑对应后端，不能无证据声称可用。

每轮测试后立即将本轮日志、残差表、绘图归档压缩，并校验归档可读。
建议 target/goal-a-logs/<task-id>/ 与 target/goal-a-archives/<task-id>.tar.gz。
只处理本轮明确生成的产物；不未经确认删除 target、旧数据或用户文件。
外部 Python/ED oracle 如参与发布验收，最小脚本与输入应纳入仓库，不能只留 /tmp。

## 11. 规划 review 决策记录（2026-11-18）

首轮 reviewer：plan-review；fresh reviewer：plan-rereview（只读，无 cargo）。
后者确认 R1–R5 规划 blocker 已解决、DAG 无循环，但并未验收尚未实现的算法。
其新增指南警示建议已 ACCEPT：SKILLS 示例明确 S>1/2 为唯象、SIA 为外加输入；
T30 补充完整占据恢复过程不等于单电子简单环。上述局部补充随最终 diff 再审。

- R1 ACCEPT：重复位点量子积先在 S=1/2 约简；T23 验矩阵等价，T40 验同LSWT，
  避免两 gate 互相依赖；ring 定为明确四点积组合。
- R2 ACCEPT：§5.2 明确采用固定算符多项式系数的材料延拓 ansatz，非默认微观结论；
  T43 加非零驱动、非等自旋权重验收，无显式接受则拒绝自旋不匹配。
- R3 ACCEPT：§7 指定 P 投影子空间追踪/秩与 canonical 对齐，G1 增加“最低本征值
  误选 photon 副本”的反例，不拿折叠后排序作 oracle。
- R4 ACCEPT：T12 移除单离子 SIA 稳定化；S=1/2 的 (Sz)^2 是常数，允许Goldstone。
- R5 ACCEPT：§5.1 固定顶点诱导簇完整 H_eff + inclusion–exclusion，写明平移
  canonical key、每胞一次和不除 automorphism；设计证明仍是 T30 gate。
- 执行建议 ACCEPT：复用 HubbardModel；T11 不能绕过共享缓存的 DFT fallback；
  入口定义半满投影而不是从 bare hopping 检查掺杂；T10/T20 可不等旧测试清理。
- 归档建议说明：日志/数值结果压缩仍按用户要求执行，但不将压缩视作物理正确性证明；
  不要求无关新增绘图或清理用户旧数据。
- reviewer 已撤回“当前缺 T40/T43 节点”和“原始来源未核对”，两者不作为未解挑战。

## 12. 完成判据与禁止自动降级

- T00–T51 所有硬门槛满足，有独立证据和收敛 review 后才能 complete goal。
- 每轮只报告该轮实际完成的任务；read/get_goal 后才允许最终 complete。
- 遇到公式、计数或 oracle 不收敛：保留最小失败例与未通过状态，推进独立任务，
  不凭“尝试三轮”把功能删掉并宣称目标完成。正式减范围需要用户同意。
- 原始论文 λ_c≈0.527 是综合参照，不用不完整驱动系数硬凑 0.4% 精度；
  先逐块/逐项验收，实际采用相同模型参数与截断后才对临界点定量比较。
- 历史错误推测（例如 CFR 不依赖窗口时猜 emin/emax 导致零结果）不当事实传递。
