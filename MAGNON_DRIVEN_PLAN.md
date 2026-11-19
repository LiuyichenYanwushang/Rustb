# 光驱动交换：当前范围与历史

旧方案 A 的通用 Floquet–Hubbard 二、三、四阶 hopping 生成器及 TB2J 明暗差分接口已撤换。
旧推导、任务 DAG 和当时的验收记录见 `git show c06c6c0:MAGNON_DRIVEN_PLAN.md`；它们不是当前 API 或数值验证的依据。

当前入口是 `Magnon::floquet_superexchange(&drive, u_ev, harmonic_max) -> Result<Magnon<DIM>>`。
它在已有静态 `Magnon` 上并行处理各条实空间非 onsite 键，只对 `trace(J)/3` 乘以二阶
Aharonov–Bohm 超交换因子。DMI、无迹各向异性、onsite 块和磁参考结构保持不变。
正负谐波成对求和，不产生旋向奇变手性或多自旋项。

调用者提供统一的正 `U`（eV），并确认待修正的各向同性交换均可用半满单轨道超交换近似；
程序不从静态交换反推 `U`、hopping 或多轨道机制。逐步增大 `harmonic_max`，检查交换张量及
所关注的可观测量是否收敛。截断收敛不证明材料有电荷能隙、处于非共振强耦合区或参考磁态稳定；
本方法也不求量子波函数。

现有圆偏振蜂窝闭式、AA 双层静态系数和解析线性自旋波工具各自保留。直接唯象交换键驱动
仍走原有 `MagnonFloquet` / `MagnonSambe` 路径；本入口不求 Sambe 准能谱，不应为同一
光驱动过程叠加两种修正。见[用法](<SKILLS.md#floquet-superexchange>)与[理论](<src/magnon/driven_theory.md>)。
