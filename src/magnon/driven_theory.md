# Floquet–Hubbard 强耦合自旋理论：公式、规范与实现边界

本页供逐式核对。[`HubbardModel::floquet_spin_model`] 接受 hopping 二至四阶；
高阶使用周期簇 Möbius 聚合，并在量子算符归约后支持固定自旋半的共线 LSWT。
[`HubbardModel::floquet_magnon_delta`] 将光场减零场的修正叠加到静态 TB2J 模型；
材料自旋延拓需显式选择，见第 11 节。

> **正向时间迁移：保持输入 Fourier/Peierls 定义，修正 Sambe 投影。**
> 当前采用正时间基 $e^{+in\Omega t}$ 与 $H^{(m-n)}+nW\delta\_{nm}$。
> 旧装配 $H^{(n-m)}+nW\delta\_{nm}$ 实际对应 $H(-t)$；第 3 节保留发现经过。
> 空间 $\mathbf k\mapsto-\mathbf k$ 不是该时间投影错误的物理证明或补偿。
> 独立双频二态 Peierls 正向传播用例已检出旧错误并验证迁移后的 Sambe；
> **这不代表完整测试套件或动态高阶手性、周期聚合、高阶 HP 已通过端到端验收。**

## 1. 电子模型与假设

定义物理站点 $x=(i,\mathbf R)$：$i$ 是输入 orbital 行，$\mathbf R$ 是整数晶胞。
同一个 orbital 在不同晶胞是不同自旋，不能把它们约化成同一个 $\mathbf S\_i$。
每个站点只有一个空间轨道、两个自旋态，电子数固定为站点数（半填充）。

$$
H(t)=H\_U+V(t),\qquad
H\_U=U\sum\_x n\_{x\uparrow}n\_{x\downarrow},\qquad U>0,
$$

$$
V(t)=\sum\_{x,y,\sigma}t\_{xy}(t)c^\dagger\_{x\sigma}c\_{y\sigma},
\qquad t\_{yx}(t)=t\_{xy}(t)^\*.
$$

这里的 hopping 求和包含正反方向。输入为自旋无关 hopping、统一 $U$，
不含 SOC、Hund、多轨道相互作用或站点势能差。
共同实 onsite $\epsilon$ 在固定粒子数空间中只贡献 $N\_e\epsilon$，已作为能量原点移去。
因此 $PVP=0$，但这不等于 $PVQ=0$。

半填充是投影空间的定义，不由 HF 或给定化学势自动推断。
Atom metadata 若明确包含每原子多轨道会被拒绝；空 Atom 不添加自旋，
未归属 Atom 的 orbital 仍按独立站点处理。这不构成对材料 downfolding 的认证。

## 2. 实空间、Peierls 与 Fourier 约定

采用电子矩阵元约定

$$
t\_{ij}(\mathbf R)=\langle i,\mathbf0|H|j,\mathbf R\rangle.
$$

所以该矩阵元对应电子从 $j,\mathbf R$ 跳到 $i,\mathbf0$。
实空间晶格向量按行存储；若 $L$ 为实空间晶格矩阵、$\tau\_i$ 为分数坐标，则

$$
\mathbf d\_{ij\mathbf R}=(\mathbf R+\tau\_j-\tau\_i)L,
\qquad
H\_{ij}(\mathbf k)=\sum\_{\mathbf R}t\_{ij}(\mathbf R)
 e^{+2\pi i\mathbf k\cdot\mathbf R}.
$$

驱动输入定义为

$$
\mathbf a(t)=\operatorname{Re}\sum\_\alpha
\mathbf a\_\alpha e^{-i\ell\_\alpha\Omega t},\qquad
W=\hbar\Omega=\texttt{omega0\\\_ev}>0.
$$

$W$ 的单位为 eV，$\Omega$ 是角频率，二者不能混写。
$\mathbf a$ 的单位为逆长度，电荷/矢势的符号须与下列**明确规定的相位**一起使用：

$$
t\_{ij}(\mathbf R,t)=t\_{ij}(\mathbf R)
\exp\lbrack-i\mathbf a(t)\cdot\mathbf d\_{ij\mathbf R}\rbrack,
$$

$$
C\_q(\mathbf d)=\frac1{T\_\Omega}\int\_0^{T\_\Omega}
 e^{+iq\Omega t}e^{-i\mathbf a(t)\cdot\mathbf d}\\,dt,
\qquad
h^{(q)}\_{ij}(\mathbf R)=t\_{ij}(\mathbf R)C\_q(\mathbf d).
$$

因此反变换是负号：

$$
H(t)=\sum\_q H^{(q)}e^{-iq\Omega t},\qquad
h^{(q)}\_{ji}(-\mathbf R)=h^{(-q)}\_{ij}(\mathbf R)^\*.
$$

所有 modes 是同一个相干经典场：先相加，再取 Peierls 指数。
同谐波会干涉，不同谐波会产生混频，不能把各 mode 的最终交换简单相加。
使用现有 Bessel 系数后端及每条 link 的 DFT fallback，不另造光场层次。
一般有 $|C\_q|\ne|C\_{-q}|$，例如标量相位 $A\cos\theta+B\sin2\theta$。

## 3. 必须核对的 Sambe 时间方向

### 3.1 标准物理 Floquet 方程

由 $i\hbar\partial\_t\psi=H(t)\psi$ 和
$\psi=e^{-i\varepsilon t/\hbar}u(t)$ 得

$$
\lbrack H(t)-i\hbar\partial\_t\rbrack u(t)=\varepsilon u(t).
$$

若采用 $u(t)=\sum\_m u\_m e^{+im\Omega t}$，逐项投影给出

$$
\mathcal K^{\mathrm{phys},+}\_{nm}
 =H^{(m-n)}+nW\delta\_{nm}.
$$

若改用 $u(t)=\sum\_m v\_m e^{-im\Omega t}$，则应同时改变对角项：

$$
\mathcal K^{\mathrm{phys},-}\_{nm}
 =H^{(n-m)}-nW\delta\_{nm}.
$$

这两种正确表示可用 $n\mapsto-n$ 联系；不能只拼接其中的非对角和另一种的对角项。

### 3.2 迁移前后与独立物理边界

旧电子 Sambe 和旧私有小簇核拼接了
$\mathcal K^{\mathrm{old}}\_{nm}=H^{(n-m)}+nW\delta\_{nm}$。
在第 2 节的 Fourier 定义下，它严格等于

$$
\mathcal K^{\mathrm{old}}
 =\operatorname{Sambe}\_{e^{+in\Omega t}}
       \lbrack H(-t)-i\hbar\partial\_t\rbrack.
$$

这叫**协议倒序**，不是完整反幺正时间反演（后者还涉及自旋、磁矩和复共轭等）。
对这里的实驱动，协议倒序等效于输入复振幅共轭；圆/椭圆偏振旋向随之反转。
迁移不改变 mode 振幅、Peierls 相位或 Fourier 定义，只将装配统一到第 3.1 节的正时间基。
因此旧 CPL 的 helicity-sensitive 质量/手性输出需要重新计算；
不是把全部准能量统一乘负号。历史上“k 镜像的两个负号抵消”的说法不是物理证明。

一个能检出方向的独立模型是

$$
H(t)=\Delta\sigma\_z+g(\sigma\_x\cos\Omega t+\sigma\_y\sin\Omega t).
$$

其 $H^{(1)}=g\sigma\_+$、$H^{(-1)}=g\sigma\_-$。
正向协议的一阶高频质量为 $-g^2\sigma\_z/W$，旧装配给 $+g^2\sigma\_z/W$。
所以现行 van Vleck 一阶必须是 $\sum\_{n>0}\lbrack H^{(-n)},H^{(n)}\rbrack/(nW)$；
两族 $W^{-2}$ 双对易项不因本次迁移整体翻号。
同一正向约定下，共同波矢的弱场线性 q 项为

$$
\delta\_q H\_{\mathrm{eff}}=+\sum\_{n>0}\frac{q\_a a\_{ni}a^\*\_{nj}}{8nW}
D\_a\\{D\_iH\_0,D\_jH\_0\\}.
$$

正号来自 $H^{(n)}:k\to k+q$ 与 $H^{(-n)}:k\to k-q$ 两条路径的 midpoint 展开，
不是仅凭对易子翻号猜出的结果；其精度仍仅为 $O(A^2q/W)$。

独立双频二态 Peierls 用例直接按 $i\hbar\partial\_t U=H(t)U$ 正向传播，
后时刻增量左乘；它区分了 forward 与 reversed 协议，迁移后的 Sambe 已通过该用例。
无符号能隙或内部 dense/sparse 一致性不足以认证一般旋向；
该用例也不等于动态高阶自旋手性或完整公开工作流的验收。
以下第 4–7 节采用现行正时间基；谐波 q 的 hopping 将总 photon 标签从 $m$ 送到 $m-q$。

## 4. 投影空间、费米符号与虚态分母

对一个含 $L\_c\le4$ 个物理站点的小簇，固定电子数 $N\_e=L\_c$。
局域 Fock 模按 spin-major 排列

$$
(0\uparrow,\ldots,(L\_c-1)\uparrow,
 0\downarrow,\ldots,(L\_c-1)\downarrow).
$$

完整 Fock 维数是 $\binom{2L\_c}{L\_c}$：二、三、四站点分别为 6、20、70。
只有**一个属于整个多电子系统的 photon 标签**，不是每个电子一个独立 photon orbital。

$$
\mathcal H\_0|\alpha,m\rangle=(N\_d(\alpha)U+mW)|\alpha,m\rangle,
$$

$$
P=\\{\text{每站点恰一个电子},\\ m=0\\},\qquad Q=1-P,\qquad E\_0=0,
$$

$$
\mathcal R=Q(E\_0-Q\mathcal H\_0Q)^{-1}\_{Q}Q,
\qquad
\mathcal R|\alpha,m\rangle
 =-\frac{|\alpha,m\rangle}{N\_d(\alpha)U+mW}\quad(\alpha,m\in Q).
$$

$\mathcal R$ 在 $P$ 上为零。$Q$ 包括 $N\_d=0,m\ne0$ 的自旋光子副本，
不只是 doublon–holon 态；四阶还必须允许 $N\_d=2$。
$m$ 是正时间基 $e^{+im\Omega t}$ 的总扇区标签；谐波 q 的 hopping 满足 $m\_{\mathrm{out}}=m\_{\mathrm{in}}-q$。
不应把标签 m 未加解释地称作“吸收 m 个光子”。

严格地说，写作整个 Q 上的逆时必须假设该逆存在。若要排除精确共振的暗态，
须先证明它们与所保留过程解耦，再把 Hilbert 空间限制到相关不变子空间。
当前程序仅对实际非零输入行检查分母；这**不证明整个 Q 有谱隙**，
也不自动解决精确简并的暗态/更高阶混合问题。
共振若参与相关过程，应扩大低能空间或另做简并处理，不能加任意 $i\eta$ 掩盖。

对 $c\_p^\dagger c\_q$，先湮灭 $q$ 再产生 $p$，分别计较低模式的占据奇偶。
自旋 tensor basis 按 site 0 最高位、$\uparrow=0$/$\downarrow=1$ 排列。
它嵌入 Fock 空间时还须包含逐站点 creator 列表到 spin-major 列表的排列符号；
例如二站点 $|\downarrow\uparrow\rangle$ 有负号。
这一步不能省略，否则交换和手性矩阵的非对角符号会错误。

## 5. Canonical Hermitian hopping 展开

这里按 hopping 次数展开，不按 $1/W$ 展开；$U$ 和 $W$ 都保留在分母中。
它不是“先对电子做 van Vleck，再把有效 hopping 塞进静态 Hubbard 公式”。
两种截断一般不交换。

在 $PVP=0$ 下，固定 canonical 正交归一化规范：

$$
H\_{\mathrm{spin}}=H\_2+H\_3+H\_4+O(V^5),
$$

$$
H\_2=PV\mathcal RVP,\qquad
H\_3=PV\mathcal RV\mathcal RVP,
$$

$$
H\_4=PV\mathcal RV\mathcal RV\mathcal RVP
\-\frac12\left\\{PV\mathcal R^2VP,\\ PV\mathcal RVP\right\\}.
$$

所有 $V$ 在这里都是扩展 Sambe 空间中的多电子 hopping 算符。
$\mathcal R$ 已含 Q 投影，因此中途返回 photon-zero 单占据空间的过程
不进入 bare chain；它们的归一化效应不能因此丢弃。

### 5.1 Folded 项从何而来

令 Bloch wave operator 为 $\Omega\_B=P+X$、$X=QXP$，其非正交有效块为 $H\_B$。
由解耦方程

$$
X=\mathcal RVP+\mathcal RVX-\mathcal R XH\_B
$$

可得 $X\_1=\mathcal RVP$，并在四阶得到

$$
H\_{B,4}=PV\mathcal RV\mathcal RV\mathcal RVP-N\_2H\_2,
\qquad N\_2=X\_1^\dagger X\_1=PV\mathcal R^2VP.
$$

规范化度量 $G=P+X^\dagger X$，canonical isometry 为
$\Omega\_B G^{-1/2}$，所以

$$
H\_{\mathrm{can}}=G^{1/2}H\_BG^{-1/2},\qquad
H\_{\mathrm{can},4}=H\_{B,4}+\frac12\lbrack N\_2,H\_2\rbrack.
$$

合并即给出 $-\\{N\_2,H\_2\\}/2$。它不是任选的防重复计数项。
$\mathcal R$ 可有正负号，但 $N\_2$ 半正定。

### 5.2 数值内核的等价写法

为避免显式算 $\mathcal R^2$ 及巨大/极小数的中间溢出，代码使用矩阵

$$
\mathsf X=VP,\qquad \mathsf W=\mathcal R\mathsf X,\qquad
\mathsf T=V\mathsf W.
$$

这里粗体 $\mathsf W$ 不是光子能量 $W$。

$$
H\_2=\mathsf X^\dagger\mathsf W,\qquad
H\_3=\mathsf W^\dagger\mathsf T,
$$

$$
H\_4=\mathsf T^\dagger\mathcal R\mathsf T
\-\frac12\\{\mathsf W^\dagger\mathsf W,H\_2\\}.
$$

$\mathsf T$ 不必预先删去 P 分量：$\mathsf W^\dagger\mathsf T$ 自动排除它，
第四阶则由 $\mathcal R\mathsf T$ 排除。R 的作用直接除物理分母，而不是先存其倒数。

这是固定 P、photon 约定、tensor 嵌入和 canonical 规范的有效算符。
P 内另做酉变换可改变高阶系数而保持相应阶数的谱；观测量也须同步变换。
它没有同时给出完整 kick/micromotion，不应直接称为任意起始时刻的裸自旋
stroboscopic Hamiltonian。

## 6. 二阶：两个 doublon 方向及常数

每条无向 bond 只计一次。将两个 doublon 方向一起求和并重标 m 后，现行正时间基给出

$$
K\_{ij}(\mathbf R)=2\sum\_m
\frac{|h^{(m)}\_{ij}(\mathbf R)|^2+|h^{(m)}\_{ji}(-\mathbf R)|^2}{U+mW},
$$

$$
H\_{2,ij\mathbf R}=K\_{ij}(\mathbf R)
\left(\mathbf S\_{(i,\mathbf0)}\cdot\mathbf S\_{(j,\mathbf R)}-\frac14\right).
$$

特别地，$i=j,\mathbf R\ne0$ 仍是两个不同站点，不是 onsite Casimir。
后续压缩记号 $\mathbf S\_i,\mathbf S\_j,\ldots$ 均表示完整物理站点，而非仅 orbital 行。
利用反向 link 的 Hermiticity，可改写为

$$
K\_{ij}=2\sum\_m |h^{(m)}\_{ij}|^2
\left(\frac1{U+mW}+\frac1{U-mW}\right).
$$

这解释了两个重点：

1. 静态极限为 $K=4|t|^2/U$；常数 $-K/4$ 使二阶 triplet shift 为零。
2. 不能一般地把它写成四倍单方向求和；只有额外满足对称权重时才能这样简化。

在对称的 photon 截断下，两个方向合起来的权重对 $m\mapsto-m$ 为偶函数，
所以此二阶 pair 结果对第 3 节的协议倒序不敏感。
**这正是二阶数值通过不能为动态高阶手性旋向背书的原因。**

## 7. 三/四阶的独立静态锚点

二聚体精确 singlet–triplet gap 为

$$
\Delta\_{\mathrm dimer}
=\frac{\sqrt{U^2+16|t|^2}-U}{2}
=\frac{4|t|^2}{U}-\frac{16|t|^4}{U^3}+O(t^6/U^5).
$$

静态二聚体的四阶 bare chain 为零而 folded 项非零，能直接杀掉漏 folded 的实现。
这个精确量子 gap 是对交换系数的参照，**不等于有限 AFM dimer 的 LSWT gap**。

对三角形，定义 index-cycle product

$$
t\_{01}t\_{12}t\_{20}=t^3e^{i\Phi},\qquad
\chi\_{012}=\mathbf S\_0\cdot(\mathbf S\_1\times\mathbf S\_2).
$$

电子 hopping 箭头是 $1\to0,2\to1,0\to2$，不是 index 列表的正向箭头。
在所列正号 hopping Hamiltonian 与此 $\chi\_{012}$ 定义下，静态三阶为

$$
H\_3=\kappa\_{012}^{\lbrack3\rbrack}\chi\_{012},\qquad
\kappa\_{012}^{\lbrack3\rbrack}=-\frac{24\\,\operatorname{Im}(t\_{01}t\_{12}t\_{20})}{U^2}.
$$

实 hopping 的完整三阶块为零，不只是手性为零。证明：粒子–空穴变换
$c\_{i\sigma}\mapsto c\_{i\sigma}^\dagger$ 使 $t\mapsto-t^\*$，
$H\_U\mapsto H\_U+U(L\_c-N\_e)=H\_U$；P/Q 均不变。
在 P 上自旋变为 $(-S\_x,S\_y,-S\_z)$，是 proper 的统一 $\pi$-y 旋转。
SU(2) 不变的 $H\_3$ 对该旋转不变，但对实 hopping 的三次齐次项必须变号，故 $H\_3=0$。
不能只保留某条三循环的实部而漏掉其余 fermionic 过程。

## 8. 周期簇、vertex Möbius 与计数

对诱导站点簇 C，先用相同 U、drive、H/M 和规范算完整 $H\_p(C)$，
再按每个 hopping 阶数分别递推

$$
\mathcal W\_p(C)=H\_p(C)-
\sum\_{\varnothing\ne S\subsetneq C}
\iota\_{S\hookrightarrow C}\mathcal W\_p(S).
$$

求和包含父簇里每一个实际 subset/embedding；不同嵌入即使具有相同 canonical key
也必须保留重数。包含 disconnected induced subsets，常数项也一起扣除。
嵌入是在自旋 tensor Hilbert 空间补 spectator identity，不再加一次费米符号。
不能先做算符分解，再因某项是二自旋便把它错误归给二站点母簇。

$$
H\_p^{\mathrm{lat}}=\sum\_{\lbrack C\rbrack\_{\mathrm{translation}}}
\sum\_{\mathbf R}\tau\_{\mathbf R}\mathcal W\_p(C).
$$

每个平移轨道只计一次；不除 $|C|$，不除 automorphism 数，也不自动合并点群等价簇。
每个 vertex anchor 都尝试共同平移、排序，取 lexicographic 最小 key。

四个 offsite hops 的占据恢复过程，每个访问顶点入度等于出度且至少为一，
因此最多访问四个顶点。Connected balanced 图有 Euler tour，故闭合游走能生成
候选 vertex supports；**过程权重仍须由完整 Fock 核计算**，不能只沿一条电子 loop。
中间两次运动可断开、可产生两个 doublon，不能据此删去路径。

**只枚举 connected supports 还需要 disconnected cancellation 的前提。**
四跳也可由两条互不相连的 bond 各往返一次；它们不能由一次连通闭合游走生成，
任意遥远两条 bond 的并集也不会由 proper-subset 补全自动补出。
只有完整的 bare/folded 配对及一致的 photon 路径集合使这些不连通过程抵消后，
才能舍去其 vertex weight。静态断连双 dimer 的测试仅认证该静态抵消，
不能代替驱动下、任意有限 photon 窗口的证明。

对于每跳 $|\Delta m|\le H$、总 photon 返回零的 p 跳过程，

$$
\lvert m\_s\rvert\le\min(s,p-s)H,\qquad
M\ge\lfloor p/2\rfloor H
$$

是完整容纳这些闭合 photon 轨迹的充分条件。
特别四阶可要求 $M\ge2H$。若 $H=M=1$，两条独立 bond 的同号 photon 跳跃
可能需要 $m=2$ 的中间态；硬窗口删去其 bare 路径却保留相应 folded 项，
会破坏不连通抵消，产生不能当作物理相互作用的有限窗口伪项。
公开四阶入口在存在非零裸键、且输入含非零非静态模式时强制 $M\ge2H$，只计算必要的 $2H$ 窗口。
这是保守验证：不会在这一步推断相干抵消或偏振与所有键正交；未输入非零非静态模式时取 $H=M=0$。
独立带限断连双 dimer 的参照中，$H=1,M=1$ 留下 $5.22\times10^{-2}$ eV 的四阶不加性，
$M=2$ 后降到 $5.92\times10^{-17}$ eV。它验证截断抵消，不是对完整 Peierls 谐波收敛的证明。

## 9. 必须先做量子 spin-half 归约

目标自旋多项式写为

$$
H\_{\mathrm{spin}}=E\_c+
\sum K\_{ij}\\,\mathbf S\_i\cdot\mathbf S\_j+
\sum\kappa\_{ijk}\\,\chi\_{ijk}+
\sum Q\_{ij;kl}(\mathbf S\_i\cdot\mathbf S\_j)(\mathbf S\_k\cdot\mathbf S\_l),
$$

自旋算符取 $\hbar=1$（Pauli/2），故 $E\_c,K,\kappa,Q$ 都是能量，单位 eV。
四自旋基使用四个不同物理站点、三种不相交 pair 配对。
在替换经典矢量或 HP 之前，必须保留

$$
S\_i^aS\_i^b=\frac{\delta\_{ab}}4+
\frac{i}{2}\epsilon\_{abc}S\_i^c,
$$

$$
(\mathbf S\_i\cdot\mathbf S\_j)^2=\frac3{16}-\frac12\mathbf S\_i\cdot\mathbf S\_j,
$$

$$
(\mathbf S\_i\cdot\mathbf S\_j)(\mathbf S\_j\cdot\mathbf S\_k)
=\frac14\mathbf S\_i\cdot\mathbf S\_k-\frac{i}{2}\chi\_{ijk}.
$$

最后一个乘积反序时虚部反号。$\mathbf S\_i^2=3/4$，不等于经典矢量长度平方 $1/4$。

对 1/2/3/4 个 spin-half，SU(2)-不变 Hermitian 算符的**实维数**分别为
1/2/5/14，不是 Hilbert 空间维数。基为 identity、所有 pairs、所有三点 chirality，
以及四点的三种 disjoint-pair products。基不必正交，系数由

$$
G\_{\mu\nu}=\operatorname{Tr}(O\_\mu^\dagger O\_\nu),\qquad
b\_\mu=\operatorname{Tr}(O\_\mu^\dagger H),\qquad Gc=b
$$

求解并重构核对。非 SU(2) 残差超过数值界限时拒绝，不能把错误矩阵默默投影。

## 10. 共线 LSWT 与普通 Magnon

取局域右手标架 $(\mathbf e\_{i1},\mathbf e\_{i2},\mathbf m\_i)$，
$\mathbf U\_i=\mathbf e\_{i1}+i\mathbf e\_{i2}$。二次 HP 展开为

$$
\mathbf S\_i=S\_i\mathbf m\_i-\mathbf m\_i b\_i^\dagger b\_i
\+\sqrt{\frac{S\_i}{2}}
(\mathbf U\_i^\* b\_i+\mathbf U\_i b\_i^\dagger)+O(b^3).
$$

LSWT 还要求线性 HP 项为零，即参考态先满足力矩条件
$\mathbf m\_i\times\partial E\_{\mathrm{cl}}/\partial\mathbf m\_i=0$。
只有在驻点上检查二次矩阵 PSD 才能认证局域稳定性；有线性项时不能仅报告 Hessian 能带。
纯 SU(2) 标量相互作用的共线参考满足该驻点条件，外加各向异性/材料 tensor 则需另核。
LSWT 是额外的半经典近似；微观 $t/U$ 展开的收敛不等于 $1/S$ 展开收敛，
尤其 $S=1/2$ 时不保证强量子涨落体系的定量 magnon 谱。

当前公开转换固定 $S\_i=1/2$、共线 axes、同一磁胞内 periodic reference（$\mathbf q=\mathbf 0$）。
需要更大磁胞时先对电子输入建 supercell。它不寻找磁基态，不支持当前接口下的
120 度等非共线参考态。

Rustb landscape 的存储约定为正反方向都包含的 ordered sum：

$$
E\_{\mathrm{land}}=-\sum\_{ij\mathbf R}
\mathbf e\_i^{\mathsf T}J^{\mathrm{land}}\_{ij}(\mathbf R)\mathbf e\_j.
$$

所以量子 pair 的每个 ordered tensor 应取

$$
J^{\mathrm{land}}\_{ij}=-\frac12K\_{ij}S\_iS\_j I\_3
=-\frac{K\_{ij}}8 I\_3\quad(S\_i=S\_j=1/2).
$$

这个 $-K/8$ 与量子 Hamiltonian 的常数 $-K/4$ 是两件不同的事。
$(S\_i^z)^2=1/4$ 对 spin-half 只是常数；本转换不会凭空加 single-ion gap。

对高阶转换，令 $\delta\mathbf S\_i^{\lbrack1\rbrack}$ 是线性 HP 项。
三点项的二次部分含三个 cyclic spectator 替换，不能只选一个 spectator：

$$
\lbrack\kappa\chi\_{ijk}\rbrack\_{\mathrm{HP},2}
=\kappa\lbrack
 S\_i\mathbf m\_i\cdot(\delta\mathbf S\_j^{\lbrack1\rbrack}\times\delta\mathbf S\_k^{\lbrack1\rbrack})
\+S\_j\mathbf m\_j\cdot(\delta\mathbf S\_k^{\lbrack1\rbrack}\times\delta\mathbf S\_i^{\lbrack1\rbrack})
\+S\_k\mathbf m\_k\cdot(\delta\mathbf S\_i^{\lbrack1\rbrack}\times\delta\mathbf S\_j^{\lbrack1\rbrack})\rbrack.
$$

在二次阶上，它等效于三个 bilinear DMI，
$\mathbf D\_{ij}^{(k)}=\kappa S\_k\mathbf m\_k$（及 cyclic 项）。
其每个 ordered landscape block 为

$$
(J^{\mathrm{land}}\_{ij})\_{ab}
=-\frac{S\_iS\_j}{2}\sum\_c\epsilon\_{cab}(D\_{ij}^{(k)})\_c
=-\frac{\kappa}{16}\sum\_c\epsilon\_{cab}(m\_k)\_c
\quad(S\_i=S\_j=S\_k=1/2).
$$

这只是在该参考态的二次阶等价，不是将三自旋量子算符恒等替换成 DMI。
四个不同站点的 pair-product 在共线参考下给出

$$
Q(\mathbf S\_i\cdot\mathbf S\_j)(\mathbf S\_k\cdot\mathbf S\_l)
\\ \leadsto\\ Q\langle\mathbf S\_k\cdot\mathbf S\_l\rangle\_0
             (\mathbf S\_i\cdot\mathbf S\_j)^{\lbrack2\rbrack}
\+Q\langle\mathbf S\_i\cdot\mathbf S\_j\rangle\_0
             (\mathbf S\_k\cdot\mathbf S\_l)^{\lbrack2\rbrack}.
$$

这里 $\lbrack2\rbrack$ 指 HP 二次部分，不是 hopping 二阶；零阶能量另行保留。
共线时写 $\mathbf m\_a=\eta\_a\mathbf n$，$\eta\_a=\pm1$，参考值为
$\langle\mathbf S\_k\cdot\mathbf S\_l\rangle\_0=S\_kS\_l\eta\_k\eta\_l$，
这是局域有序参考积态的值，不是精确纠缠基态的相关函数。因此

$$
\delta K\_{ij}=Q\_{ij;kl}S\_kS\_l\eta\_k\eta\_l,\qquad
\delta J^{\mathrm{land}}\_{ij}
=-\frac{Q\_{ij;kl}}{32}\eta\_k\eta\_l I\_3\quad(S\_a=1/2),
$$

并加上交换两个 pair 后的贡献。
公开 `magnon` 用这些参考态相关的二次系数构造普通 `Magnon`，不改变原量子算符结果。

Nambu 向量及玻色激发能满足（代码返回 eV，不是每秒）

$$
\Psi\_{\mathbf k}=(b\_{1\mathbf k},\ldots,b\_{N\mathbf k},
 b^\dagger\_{1,-\mathbf k},\ldots,b^\dagger\_{N,-\mathbf k})^{\mathsf T},
\quad H^{\lbrack2\rbrack}=\frac12\sum\_{\mathbf k}\Psi\_{\mathbf k}^\dagger
\mathcal H\_B(\mathbf k)\Psi\_{\mathbf k}+\mathrm{const},
$$

$$
g=\operatorname{diag}(I\_N,-I\_N),\qquad
g\mathcal H\_B y=\varepsilon\_{\mathrm{mag}} y,
\qquad \omega\_{\mathrm{mag}}^{\mathrm{angular}}=\varepsilon\_{\mathrm{mag}}/\hbar.
$$

实的 $g\mathcal H\_B$ 本征值本身不足以证明参考态是局域能量极小值。
稳定参考要求二次能量矩阵 PSD（容许数值误差内 Goldstone 零模）。
若 $\mathcal H\_B=KK^\dagger$，则可用 Hermitian 矩阵

$$
M=K^\dagger gK,\qquad Mv=\varepsilon\_{\mathrm{mag}} v,\qquad
 y=gKv,\qquad y^\dagger gy=\varepsilon\_{\mathrm{mag}}\\|v\\|^2
$$

求非负稳定激发能；角频率再除以 $\hbar$。注意是 $gKv$，不是 $Kv$。
Jordan 型零模这里只承诺零激发能，不承诺可归一化本征矢；小正 roundoff 不代表真实 gap。
一次 q 的稳定检查也不等于全 BZ 参考态稳定性的证明。

## 11. 材料自旋与 TB2J 差分

微观模型严格给出 spin-half。把量子归约后的系数固定，再把自旋长度替换成
材料 $S\_i$，是一种**须明确接受的现象学 lifting**，不是唯一的高自旋 Hubbard 推导。
外加的材料 SIA 也不能冒充单轨道 spin-half 微观生成结果。

TB2J 合并在相同规范、同一站点映射与相同自旋映射下使用

$$
H\_{\mathrm{material}}(a)=H\_{\mathrm{TB2J}}^{\mathrm{static}}
\+\lbrack H\_{\mathrm{micro}}(a)-H\_{\mathrm{micro}}(0)\rbrack\_{\mathrm{mapped}}.
$$

`SpinLengthMapping::SpinHalf` 要求目标自旋严格为 1/2；
`KeepOperatorCoefficients` 采用上面的延拓，二体、三体、四体的单位矢量曲率分别
乘参与站点的两个、三个、四个自旋长度。若这些乘积超出可用浮点范围，返回错误。
必须提供 orbital 到磁站点的双射，且晶格矩阵相同。若输入第 i 个中心等于目标中心
加整数移位 $s_i$，则键从 $R$ 变成 $R+s_j-s_i$；映射保持手性的有序站点顺序。
先对量子自旋多项式做差，再按基线自旋大小和同一胞内共线参考做 HP 归约。
算法不调整磁矩轴、保留原有各向异性，并将新平移行追加到原行之后。
明场与零场分别受同一组数值预算约束；零场只用零光子副本。
接近共线的数值舍入按基线首个轴判定，修正沿该共线参考计算，不修改输出基线的轴。
调用者仍须检查合并参考的驻点与稳定性。

不能把完整微观静态交换再加一遍，也不能对同一 AB 光学过程同时应用本修正和
旧的 phenomenological magnon bond-drive，从而二次计数。

## 12. 收敛与验收清单

- hopping 阶数 $p$、场幅阶数、$1/W$ 阶数是不同的展开，不能互称“二阶”。
- 单跳截断 $|\Delta m|\le H$ 与累计 $|m\_s|\le M$ 独立；二阶取 min，
  三/四阶必须逐步保留中间 photon 轨迹，不能把缺失 cache bin 当零。
- 四阶累计 photon 可达 $2H$；有限 M 是空间投影，不是自动的收敛证明。
- 除一跳 $|t^{(m)}/(U+mW)|$ 之外，四阶核对相干求和后的第二级中间量 $RT$：
  对每个可达 Q 态，从 P 的全部列取 Euclidean 行范数并检查 `max_hopping_ratio`。
  这能拒绝一跳分母都正常、但两跳强烈混入近共振副本的情况；仍不是整体误差界。
- 代码不按振幅阈值删去小系数，但受 `f64` 运算舍入限制。单次路径乘积在累加前下溢、
  或同一算符块中不同能标跨度过大时，即使理论系数最终可表示，也不保证相对精度。
- 必须分别扫 H/M、比较 hopping 各阶、检查活跃分母及必要的中间耦合尺度。
- 精确对角化必须从 $t=0$ 的 P 子空间连续追踪并做 canonical polar 对齐，
  不能取 Floquet 矩阵最低若干本征值，也不能取折叠后最近零的若干值。
- 内部稀疏核与相同公式的 dense Sambe 对照，只验证实现；二聚体精确 gap、
  三角静态通量和正向时间传播才提供不同层面的独立物理约束。
- 正向 Sambe 迁移由独立电子传播确认；自旋侧另有方环的正向传播和逐系数对照、
  蜂窝/AA 双层/非共面 3D 的带标签 BdG 对照，以及原胞/双倍超胞逐算符检查。
  这些固定夹具不证明任意驱动、晶格或能标的收敛性。
- 以上理论不证明长期无加热、真实系统必占据该低能分支，或材料单轨道近似成立。

## 13. 可运行的四阶共线例子

以下是两站点反铁磁链。能量用 eV，`LightMode` 的振幅用晶格长度的倒数。
先保留量子多项式，再选反平行参考轴；这里没有人为加入单离子各向异性。

```rust
use ndarray::array;
use num_complex::Complex64;
use Rustb::{FloquetDrive, HubbardModel, LightMode, Model};
use Rustb::magnon::{MagnonReference, MagnonSolve};
use Rustb::magnon::driven::StrongCouplingOptions;

fn main() -> Rustb::Result<()> {
    let mut electron = Model::<true, 1>::tb_model(array![[2.0]], array![[0.0], [0.5]], None)?;
    electron.set_hop(0.1, 0, 1, &array![0isize], None);
    electron.set_hop(0.1, 0, 1, &array![-1isize], None);
    let hubbard = HubbardModel::with_uniform_u(electron, 4.0)?;
    let light = FloquetDrive::with_modes(5.0, vec![LightMode::new(1, array![Complex64::new(0.3, 0.0)])]);
    let options = StrongCouplingOptions {
        max_hopping_order: 4,
        hop_harmonic_max: 4,
        intermediate_photon_max: 8,
        ..Default::default()
    };
    let spin = hubbard.floquet_spin_model(&light, &options)?;
    let axes = array![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let magnon = spin.magnon(&axes, MagnonReference::collinear())?;
    let energies = magnon.magnon_band_onek(&array![0.25])?;
    assert_eq!(energies.len(), 2);
    assert!(energies.iter().all(|e| e.is_finite() && *e > 0.0));
    Ok(())
}
```
