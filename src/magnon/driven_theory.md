# 从静态交换得到二阶 Floquet 超交换

[`crate::magnon::Magnon::floquet_superexchange`] 接收一个已有的静态 `Magnon`、
均匀周期 Peierls 驱动、有限正的有效 $U$（eV）和对称谐波截断。它不需要电子 hopping
文件，不从静态谱反推 $U$，也不直接给磁振子施加电子 Peierls 相位。

## 逐键推导

把静态交换的一部分解释为半满单轨道的二阶超交换。电子在实际笛卡尔键
$\mathbf d=(\mathbf R+\tau\_j-\tau\_i)\cdot\mathrm{lat}$ 上的 hopping 含周期相位
$e^{-i\mathbf a(t)\cdot\mathbf d}$，其 Fourier 系数为 $C\_m(\mathbf d)$。
令 $W=\hbar\Omega$。虚跃迁有两个中间 doublon 方向，反向键满足
$C\_m(-\mathbf d)=C\_{-m}(\mathbf d)^*$。以无光交换 $4|t|^2/U$ 归一化，
光照交换只需要一个无量纲键因子：

$$
F = \frac12\sum\_m |C\_m|^2
      \left(\frac{U}{U+mW}+\frac{U}{U-mW}\right).
$$

单色光有 $|C\_m|^2=\mathcal J\_m(|\mathbf a\_0\cdot\mathbf d|)^2$。
多色相干驱动通常不能假定两个谐波符号的权重相等。Parseval 给出
$\sum\_m|C\_m|^2=1$，所以在 $M$ 阶对称窗口内计算

$$
F-1=\sum\_{m=1}^{M} (|C\_m|^2+|C\_{-m}|^2)\frac{(mW)^2}{U^2-(mW)^2}.
$$

截断之外的权重只贡献静态分母；要增大 $M$ 检查收敛。这样无光、只有静态相位，
或光场垂直于键时不改交换；把所有分母都近似成 $U$ 也没有二阶光致变化。

## 作用在静态交换上

对每个非 onsite 张量块 $J$，只更新各向同性投影
$J\mapsto J+(F-1)\,\operatorname{tr}(J)I/3$；两个互易方向共用同一 $F$。
若输入的互易块在校验容差内略有差别，实际修正的键先对正向块与反向转置取平均；
非各向同性分量不受光场重标定。最终输出重新通过 `Magnon::validate`。
相同磁性位点在不同晶胞仍是两个不同物理自旋。真正的同位点 onsite、DMI、无迹各向异性、
晶格和参考态都不变。该模型不能从静态 $J$ 区分直接交换和超交换；使用者须确认此近似。

```rust
use ndarray::array;
use num_complex::Complex;
use Rustb::{FloquetDrive, LightMode, Magnon, MagnonReference};
use Rustb::magnon::MagnonSolve;

# fn main() -> Rustb::Result<()> {
// TB2J's negative J is antiferromagnetic; the magnetic cell has two sites.
let j = -0.02;
let mut exchange = ndarray::Array3::zeros((3, 6, 6));
for a in 0..3 {
    for (row, from, to) in [(0, a, 3 + a), (0, 3 + a, a), (1, 3 + a, a), (2, a, 3 + a)] {
        exchange[[row, from, to]] = j;
    }
}
let baseline = Magnon::<1>::new(
    array![[2.0]], array![[0.0], [0.5]], array![0.5, 0.5],
    array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    exchange, array![[0], [1], [-1]], MagnonReference::collinear(),
)?;
let drive = FloquetDrive::with_modes(1.0, vec![LightMode::new(1, array![Complex::new(0.4, 0.0)])]);
let dressed = baseline.floquet_superexchange(&drive, 4.2, 4)?;
let q = array![0.5];
assert!(dressed.magnon_band_onek(&q)?[0] > baseline.magnon_band_onek(&q)?[0]);
# Ok(())
# }
```

公式要求电荷虚跃迁的混合足够弱，并避开有权重的 $U\pm mW$ 共振。程序拒绝
已保留且系数后端解析出非零权重、数值上命中极点的谐波，但不能仅从静态自旋模型保证物理非共振。
本接口复用有资源上限的贝塞尔谐波后端，超限时返回错误，不用粗时间网格替代。输出是有效静态交换，
不包含主动的磁振子共振、加热、光致手性回路或旋向反号的二阶项。用普通求谱器时也须检查
所选参考态是否仍稳定。独立的唯象磁振子驱动入口不应再叠加同一物理机制。
