//! Eigenvalue solver for tight-binding Hamiltonians.
use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::ndarray_lapack::{eigh_r, eigvalsh_r, eigvalsh_v};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;

// Total Bloch phase matrix/Hamiltonian batch budget across jobs. Output arrays,
// orbital phases and LAPACK workspace are additional; process even one oversized k-point.
const FOURIER_MEMORY_BUDGET: usize = 128 * 1024 * 1024;

// Return (k-points per coarse job, k-points per GEMM). Each coarse job streams
// its assigned range in bounded batches; Rayon can schedule jobs dynamically.
fn batch_plan(nk: usize, nr: usize, nsta: usize, threads: usize) -> (usize, usize) {
    if nk == 0 {
        return (1, 1);
    }
    let bytes_per_k = nsta
        .checked_mul(nsta)
        .and_then(|width| width.checked_add(nr))
        .and_then(|width| width.checked_mul(size_of::<Complex<f64>>()))
        .expect("batch buffer size overflow");
    let live_points = (FOURIER_MEMORY_BUDGET / bytes_per_k.max(1)).max(1);
    let workers = threads.max(1).min(nk).min(live_points);
    let job_size = nk.div_ceil(workers);
    (job_size, job_size.min(live_points / workers))
}

fn diagonalize_with_vectors<S: Data<Elem = Complex<f64>>>(
    ham: &ArrayBase<S, Ix2>,
) -> (Array1<f64>, Array2<Complex<f64>>) {
    // 布局/共轭约定（此处所有调用者均传入 gen_ham[_batch] 生成的 C 布局 H）：
    // 本次核对的 ndarray-linalg 0.18.1 的 eigh_inplace 对 C 布局只做 swap_axes(0, 1)，
    // 并没有共轭数据，所以 LAPACK 实际读到 H^T = H*（H 必须是 Hermitian）。
    // 原始返回矩阵 U（下面的 vectors）按列存放 H* 的本征矢：H* U = U D。
    // 因而 H 的列 ket 矩阵为 U*，而 Rustb 要返回按行存放 ket 系数的 C：
    //     C = (U*)^T = U†，C[n, alpha] = <alpha|psi_n>。
    // 注意 ndarray_linalg::conjugate(&U) 是“共轭转置”，不是仅逐元素共轭！
    // 若改成 mapv(|z| z.conj()) 就会丢掉转置，使 band/basis 两个轴颠倒。
    // 对 Rustb 返回的 C：H C^T = C^T D，算符变换为 C* O C^T。
    // 对原始 eigh 返回的 U：算符变换才是 U^T O U*。两套公式不能混用。
    // 这是当前依赖和输入布局对应的补偿，不适用于任意 F 布局的 eigh 结果。
    // 升级 ndarray-linalg/lax 或改变 H 的布局时，必须重跑复数 H 的残差检查；
    // H 与 H* 的能量相同，仅比较本征值不足以发现这个错误。
    let (energies, vectors) = ham
        .eigh(UPLO::Lower)
        .expect("Hermitian eigendecomposition failed");
    (
        energies,
        conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>>(&vectors),
    )
}

/// Solve the tight-binding Hamiltonian H(k).
///
/// 本征值按能量升序排列；Rustb 返回的本征矢以“能带为行、基底分量为列”。
/// 完整的数组轴、共轭、规范及基底变换约定见 [`Solve::solve_onek`] 和
/// [`Solve::solve_all`]，不能直接套用原始 `ndarray_linalg::Eigh::eigh` 的返回约定。
pub trait Solve {
    /// Solve energy bands at a single k-point.
    ///
    /// 返回长度为 `nsta` 的实数数组，按能量非递减排列：`E[0] <= E[1] <= ...`。
    /// 这是所有态的统一能量排序，不是轨道顺序、自旋分块顺序或按能量绝对值排序。
    fn solve_band_onek<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>) -> Array1<f64>;
    /// Solve energy bands at a single k-point with a range
    fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> Array1<f64>;
    /// Solve energy bands at all given k-points.
    ///
    /// 返回形状 `(nk, nsta)`，`bands[[ik, n]]` 对应输入第 `ik` 个 k 点的
    /// 第 `n` 小能量。每个 k 点独立升序排序，不进行跨 k 点的能带追踪。
    fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Array2<f64>;
    /// Solve energy bands in parallel, using bounded batches of Fourier sums.
    /// Each job processes a contiguous k-point range and diagonalizes individual
    /// H(k) matrices before building its next batch. Returns eigenvalues in
    /// input k-point order. Configure BLAS threading separately from Rayon.
    /// Shape and energy ordering are identical to [`Solve::solve_band_all`].
    fn solve_band_all_parallel<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>)
    -> Array2<f64>;
    /// Solve energies and eigenvectors at one k-point in [`Gauge::Atom`].
    ///
    /// # 本征值排序与本征矢的轴
    ///
    /// 返回 `(energies, evec)`，令 `N = self.nsta()`：
    ///
    /// | 返回值 | 形状/索引 | 含义 |
    /// |---|---|---|
    /// | `energies` | `(N,)`，`energies[n]` | 第 `n` 小的能量，单位与模型一致（通常 eV） |
    /// | `evec` | `(N, N)`，`evec[[n, alpha]]` | 第 `n` 个本征态在第 `alpha` 个基底态上的 ket 系数 |
    ///
    /// `energies[0] <= energies[1] <= ... <= energies[N-1]`，允许相等。
    /// 内部 `.eigh(UPLO::Lower)` 的 `Lower` 指读取哪个三角，**与本征值升降序无关**。
    /// 排序由底层 [LAPACK ZHEEV](https://www.netlib.org/lapack/explore-html/d8/d1c/group__heev_gadbb2b87ce42e51fdaac3228a857a58c8.html)
    /// 保证。要求模型的完整 H 为 Hermitian；求解器只使用一个三角，不会验证另一侧是否一致。
    /// 同一索引 `n` 始终配对 `energies[n]` 与 `evec.row(n)`；**一个本征矢是一行，
    /// 不是一列**。不要单独重排能量而不同时重排本征矢的行。
    ///
    /// `alpha` 沿用模型的基底顺序：无自旋时是轨道顺序；有自旋时是
    /// `[orbital_0_up, ..., orbital_(norb-1)_up, orbital_0_down, ..., orbital_(norb-1)_down]`。
    /// 这是基底轴的顺序，**能带轴 `n` 不按自旋分块**；有自旋混合时也没有固定的自旋标签。
    ///
    /// # 行中存的是 ket 系数，不是 bra
    ///
    /// 令 `C = evec`、`H = self.gen_ham(kvec, Gauge::Atom)`、`D = diag(energies)`。
    /// `C[n, alpha] = <alpha | psi_n>`，所以逐元素满足
    /// `sum_beta H[alpha,beta] * C[n,beta] = energies[n] * C[n,alpha]`。
    /// 虽然系数存在二维数组的一行，取出的 `C.row(n)` 是一维数组，可以直接传给
    /// `H.dot(&C.row(n))`；**不需要再对这一行取共轭**。
    ///
    /// 用 `*` 表示逐元素复共轭、`T` 表示转置、`†` 表示共轭转置：
    ///
    /// ```text
    /// H C^T       = C^T D          // 本征方程；C^T 的每一列是一个 ket
    /// C* C^T      = I              // 本征态正交归一
    /// C* H C^T    = D              // Hamiltonian 的能带表象
    /// O_band      = C* O C^T       // 同一 k 点、同一 Atom 规范/基底下的任意算符
    /// H           = C^T D C*       // 由完整本征系重建 H
    /// ```
    ///
    /// 对应 Rust 写法是 `evec.mapv(|z| z.conj()).dot(&op.dot(&evec.t()))`。
    /// `ndarray` 的 `.t()` 只转置、不共轭；`mapv(|z| z.conj())` 只共轭、不转置；
    /// `ndarray_linalg::conjugate()` 则同时共轭和转置，三者不能互换。
    ///
    /// # 与直接调用 ndarray-linalg 的区别
    ///
    /// 本次核对的 `ndarray-linalg 0.18.1` 对 Rustb 的 C 布局复数 H 调用 `.eigh()`
    /// 时，内部交换轴，实际求解的是 `H^T = H*`。原始返回矩阵 U 按列存放 `H*`
    /// 的本征矢，因此原始 U 的算符变换为 `U^T O U*`。
    /// 本方法已经用共轭转置将它转换为 `C = U†`，再按上述 Rustb 约定返回。
    /// **不要把原始 U 的变换公式或“本征矢在列”的习惯套到这里。**
    /// 此说明针对当前依赖及 C 布局调用路径；F 布局直接调用或依赖升级应重新核对。
    /// `H` 与 `H*` 本征值相同，单看能量正确、或只测实数模型，都不足以检查本征矢约定。
    ///
    /// # 相位、简并与能带连续性
    ///
    /// 各本征矢已归一化，但整体复相位任意。简并子空间内可以返回任意正交基，
    /// 简并态之间没有固定排序；本方法也不按轨道成分、自旋或相邻 k 点的重叠追踪能带。
    /// 比较不同求解器或调用的结果时，应检查本征方程、态的重叠或简并子空间投影，
    /// 不应要求本征矢逐元素一致。需要连续本征矢或数值 k 导数时，应自行处理相位/
    /// 子空间对齐；Berry 相位和 Wilson loop 也可以用规范不变的重叠方法计算，
    /// 不要求先将每个态的相位调成连续。
    ///
    /// # Example: 检查复数 Hamiltonian 的返回约定
    ///
    /// ```
    /// use Rustb::{Gauge, Model, Solve};
    /// use ndarray::{array, Array2};
    /// use num_complex::Complex;
    ///
    /// let mut model = Model::<false, 1>::tb_model(
    ///     array![[1.0]], array![[0.0], [0.2], [0.4]], None,
    /// ).unwrap();
    /// model.set_onsite(&array![-0.8, 0.3, 1.2], None);
    /// model.add_hop(Complex::new(0.4, 0.7), 0, 1, &array![0], None);
    /// model.add_hop(Complex::new(-0.2, 0.5), 1, 2, &array![1], None);
    /// model.add_hop(0.17, 0, 0, &array![1], None); // 使能量随 k 改变
    /// let k = array![0.23];
    /// let h = model.gen_ham(&k, Gauge::Atom);
    /// let (energies, evec) = model.solve_onek(&k);
    ///
    /// for n in 0..energies.len() {
    ///     if n > 0 { assert!(energies[n - 1] <= energies[n]); }
    ///     let ket = evec.row(n); // 第 n 行就是第 n 个态的 ket 系数
    ///     let residual = h.dot(&ket) - &ket * energies[n];
    ///     assert!(residual.iter().all(|z| z.norm() < 1e-12));
    /// }
    /// let bra = evec.mapv(|z| z.conj());
    /// let h_band = bra.dot(&h.dot(&evec.t()));
    /// let diagonal = Array2::from_diag(&energies.mapv(|e| Complex::new(e, 0.0)));
    /// assert!(h_band.iter().zip(&diagonal).all(|(a, b)| (a - b).norm() < 1e-12));
    /// let overlap = bra.dot(&evec.t());
    /// let identity = Array2::<Complex<f64>>::eye(model.nsta());
    /// assert!(overlap.iter().zip(&identity).all(|(a, b)| (a - b).norm() < 1e-12));
    /// ```
    fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> (Array1<f64>, Array2<Complex<f64>>);
    /// Solve eigenvalues in the half-open energy window `(range.0, range.1]`.
    ///
    /// Returns ascending energies and row ket coefficients with shape
    /// `(number_of_selected_bands, nsta)`, in the same gauge as [`Self::solve_onek`].
    /// `epsilon` is LAPACK's absolute convergence tolerance; nonpositive values
    /// select its default tolerance.
    ///
    /// # Panics
    /// Panics for a k-vector with the wrong dimension, nonfinite or unordered
    /// range bounds, a nonfinite tolerance, or an eigensolver failure.
    fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> (Array1<f64>, Array2<Complex<f64>>);
    /// Solve energies and eigenvectors for all input k-points, serially.
    ///
    /// # 数组排列与逐点排序
    ///
    /// 输入 `kvec` 的形状为 `(nk, DIM)`，每一行是一个分数倒空间坐标。
    /// 返回 `(energies, evec)`，令 `N = self.nsta()`：
    ///
    /// - `energies.shape() == [nk, N]`，索引为 `energies[[ik, n]]`；
    /// - `evec.shape() == [nk, N, N]`，索引为 `evec[[ik, n, alpha]]`；
    /// - 轴 0 是**输入 k 点索引**，轴 1 是**能带索引**，轴 2 是**模型基底索引**。
    ///   `evec.slice(s![ik, n, ..])` 才是一个完整本征矢。
    ///
    /// k 点顺序与输入完全一致，不按坐标或能量重排 k 点；内部如何分批不改变这一点。
    /// 在每个 `ik` 内，`energies[[ik, 0]] <= ... <= energies[[ik, N-1]]`，
    /// 同一个 `n` 与 `evec[[ik, n, ..]]` 配对，所有自旋/轨道态一起按能量排序。
    /// 各 k 点**独立排序**，没有跨 k 点的能带追踪：能带交叉处，同一个 `n` 不保证
    /// 代表连续的轨道或自旋特征。
    ///
    /// # 本征矢约定与单点接口一致
    ///
    /// 使用 [`Gauge::Atom`]；每个 `evec.index_axis(Axis(0), ik)` 都遵循
    /// [`Solve::solve_onek`] 的“行存 ket 系数”约定。令该二维切片为 C，则：
    ///
    /// ```text
    /// H(k_ik) C^T = C^T diag(energies[ik, :])
    /// O_band(k_ik) = C* O(k_ik) C^T
    /// ```
    ///
    /// 应先取出二维的第 `ik` 个切片，再对它调用 `.t()`；不要对整个三维 `evec`
    /// 转置/反转轴来替代逐 k 点的矩阵转置。共轭、基底顺序、简并和整体相位的细节
    /// 见 [`Solve::solve_onek`]。与逐点或并行求解比较时，允许浮点舍入差异以及
    /// 本征矢的整体相位/简并子空间旋转，不保证逐元素或逐 bit 相同。
    ///
    /// # Example: 按 `[k, band, basis]` 取态
    ///
    /// ```
    /// use Rustb::{Gauge, Model, Solve};
    /// use ndarray::{array, s};
    /// use num_complex::Complex;
    /// # let mut model = Model::<false, 1>::tb_model(
    /// #     array![[1.0]], array![[0.0], [0.2], [0.4]], None,
    /// # ).unwrap();
    /// # model.set_onsite(&array![-0.8, 0.3, 1.2], None);
    /// # model.add_hop(Complex::new(0.4, 0.7), 0, 1, &array![0], None);
    /// # model.add_hop(Complex::new(-0.2, 0.5), 1, 2, &array![1], None);
    /// # model.add_hop(0.17, 0, 0, &array![1], None);
    /// let points = array![[0.31], [0.07], [0.22]]; // 保留这个输入顺序
    /// let (energies, evec) = model.solve_all(&points);
    /// assert_eq!(energies.dim(), (3, model.nsta()));
    /// assert_eq!(evec.dim(), (3, model.nsta(), model.nsta()));
    /// for ik in 0..points.nrows() {
    ///     let k = points.row(ik);
    ///     let h = model.gen_ham(&k, Gauge::Atom);
    ///     let (one_energy, _) = model.solve_onek(&k);
    ///     for n in 0..model.nsta() {
    ///         assert!((energies[[ik, n]] - one_energy[n]).abs() < 1e-12);
    ///         let ket = evec.slice(s![ik, n, ..]);
    ///         let residual = h.dot(&ket) - &ket * energies[[ik, n]];
    ///         assert!(residual.iter().all(|z| z.norm() < 1e-12));
    ///     }
    /// }
    /// let (parallel_energy, _) = model.solve_all_parallel(&points);
    /// assert!(parallel_energy.iter().zip(&energies).all(|(a, b)| (a - b).abs() < 1e-12));
    /// ```
    fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>);
    /// Parallel counterpart of [`Solve::solve_all`], with the same return convention.
    ///
    /// `energies[[ik, n]]` 配对 `evec[[ik, n, alpha]]`，能量在每个 k 点内升序，
    /// k 点保留输入顺序，本征矢使用 Atom 规范且按行存放 ket 系数。
    /// 并行完成顺序不改变输出轴顺序；不进行能带追踪或相位对齐。
    /// 数组形状、变换公式及简并态比较方法详见 [`Solve::solve_all`]、[`Solve::solve_onek`]。
    fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>);
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Solve for Model<SPIN, DIM, R> {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn solve_band_onek<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>) -> Array1<f64> {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(kvec, Gauge::Atom);
        let eval = eigvalsh_v(&hamk, UPLO::Upper).expect("Hermitian eigendecomposition failed");
        eval
    }

    fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> Array1<f64> {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        let eval = eigvalsh_r(&hamk, range, epsilon, UPLO::Upper)
            .expect("Hermitian range eigendecomposition failed");
        eval
    }
    fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Array2<f64> {
        self.solve_band_batches(kvec, false)
    }
    #[allow(non_snake_case)]
    fn solve_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Array2<f64> {
        self.solve_band_batches(kvec, true)
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> (Array1<f64>, Array2<Complex<f64>>) {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        diagonalize_with_vectors(&hamk)
    }
    fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        range: (f64, f64),
        epsilon: f64,
    ) -> (Array1<f64>, Array2<Complex<f64>>) {
        assert_eq!(
            kvec.len(),
            self.dim_r(),
            "Wrong, the k-vector's length:k_len={} must equal to the dimension of model:{}.",
            kvec.len(),
            self.dim_r()
        );
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        let (eval, evec) = eigh_r(&hamk, range, epsilon, UPLO::Upper)
            .expect("Hermitian range eigendecomposition failed");
        (eval, evec)
    }

    #[allow(non_snake_case)]
    fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        self.solve_eigenvector_batches(kvec, false)
    }
    #[allow(non_snake_case)]
    fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        self.solve_eigenvector_batches(kvec, true)
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    fn solve_band_batches<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        parallel: bool,
    ) -> Array2<f64> {
        let threads = if parallel {
            rayon::current_num_threads()
        } else {
            1
        };
        let (job_size, batch_size) =
            batch_plan(points.nrows(), self.hamR.nrows(), self.nsta(), threads);
        let mut bands = Array2::zeros((points.nrows(), self.nsta()));
        let solve = |(points, mut energies): (ArrayView2<'_, f64>, ArrayViewMut2<'_, f64>)| {
            for (batch, mut output) in points
                .axis_chunks_iter(Axis(0), batch_size)
                .zip(energies.axis_chunks_iter_mut(Axis(0), batch_size))
            {
                let hams = self.gen_ham_batch(&batch, Gauge::Lattice);
                for (ham, mut row) in hams.outer_iter().zip(output.outer_iter_mut()) {
                    row.assign(
                        &eigvalsh_v(&ham, UPLO::Upper)
                            .expect("Hermitian eigendecomposition failed"),
                    );
                }
            }
        };
        if parallel {
            points
                .axis_chunks_iter(Axis(0), job_size)
                .into_par_iter()
                .zip(
                    bands
                        .axis_chunks_iter_mut(Axis(0), job_size)
                        .into_par_iter(),
                )
                .for_each(solve);
        } else {
            points
                .axis_chunks_iter(Axis(0), job_size)
                .zip(bands.axis_chunks_iter_mut(Axis(0), job_size))
                .for_each(solve);
        }
        bands
    }

    fn solve_eigenvector_batches<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        parallel: bool,
    ) -> (Array2<f64>, Array3<Complex<f64>>) {
        let threads = if parallel {
            rayon::current_num_threads()
        } else {
            1
        };
        let (job_size, batch_size) =
            batch_plan(points.nrows(), self.hamR.nrows(), self.nsta(), threads);
        let mut bands = Array2::zeros((points.nrows(), self.nsta()));
        let mut vectors = Array3::zeros((points.nrows(), self.nsta(), self.nsta()));
        let solve = |((points, mut energies), mut vectors): (
            (ArrayView2<'_, f64>, ArrayViewMut2<'_, f64>),
            ArrayViewMut3<'_, Complex<f64>>,
        )| {
            for ((batch, mut output), mut states) in points
                .axis_chunks_iter(Axis(0), batch_size)
                .zip(energies.axis_chunks_iter_mut(Axis(0), batch_size))
                .zip(vectors.axis_chunks_iter_mut(Axis(0), batch_size))
            {
                let hams = self.gen_ham_batch(&batch, Gauge::Atom);
                for ((ham, mut row), mut state) in hams
                    .outer_iter()
                    .zip(output.outer_iter_mut())
                    .zip(states.outer_iter_mut())
                {
                    let (energies, eigenvectors) = diagonalize_with_vectors(&ham);
                    row.assign(&energies);
                    state.assign(&eigenvectors);
                }
            }
        };
        if parallel {
            points
                .axis_chunks_iter(Axis(0), job_size)
                .into_par_iter()
                .zip(
                    bands
                        .axis_chunks_iter_mut(Axis(0), job_size)
                        .into_par_iter(),
                )
                .zip(
                    vectors
                        .axis_chunks_iter_mut(Axis(0), job_size)
                        .into_par_iter(),
                )
                .for_each(solve);
        } else {
            points
                .axis_chunks_iter(Axis(0), job_size)
                .zip(bands.axis_chunks_iter_mut(Axis(0), job_size))
                .zip(vectors.axis_chunks_iter_mut(Axis(0), job_size))
                .for_each(solve);
        }
        (bands, vectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HasRMatrix, Velocity};
    use std::f64::consts::TAU;

    #[test]
    fn range_solver_preserves_complex_ket_convention() {
        let model = complex_model::<true, 3>();
        let k = array![0.17, 0.31, 0.07];
        let ham = model.gen_ham(&k, Gauge::Atom);
        let full = model.solve_band_onek(&k);
        let lower = (full[0] + full[1]) * 0.5;
        let (energies, vectors) = model.solve_range_onek(&k, (lower, 10.0), 0.0);
        assert_eq!(energies.len(), model.nsta() - 1);
        for (n, ket) in vectors.outer_iter().enumerate() {
            assert!((energies[n] - full[n + 1]).abs() < 1e-12);
            let residual = ham.dot(&ket) - &ket * energies[n];
            assert!(residual.iter().all(|z| z.norm() < 1e-12));
        }
    }

    #[test]
    fn batch_plan_bounds_memory_and_job_count() {
        for nk in [0, 1, 128, 129, 4096, 29791] {
            for threads in [1, 3, 8, 256] {
                for (nr, nsta) in [(0, 0), (7, 4), (841, 200), (1, 4096)] {
                    let per_k = (nr + nsta * nsta) * size_of::<Complex<f64>>();
                    let (job_size, batch_size) = batch_plan(nk, nr, nsta, threads);
                    let jobs = nk.div_ceil(job_size);
                    assert!(jobs <= threads);
                    assert!(
                        jobs * batch_size * per_k <= FOURIER_MEMORY_BUDGET
                            || (jobs <= 1 && batch_size == 1 && FOURIER_MEMORY_BUDGET < per_k)
                    );
                    assert!(batch_size > 0 && job_size > 0);
                }
            }
        }
        // Automatic batches are not capped at the previous hard-coded 16.
        assert!(batch_plan(4096, 841, 200, 8).1 > 16);
    }

    // Direct scalar sums, independent of the new GEMM and phase helpers.
    fn operator_reference<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        k: ArrayView1<'_, f64>,
        gauge: Gauge,
    ) -> (Array3<Complex<f64>>, Array2<Complex<f64>>) {
        let n = model.nsta();
        let mut ham = Array2::<Complex<f64>>::zeros((n, n));
        let mut velocity = Array3::<Complex<f64>>::zeros((DIM, n, n));
        let mut position = Array3::<Complex<f64>>::zeros((DIM, n, n));
        for (ir, r) in model.hamR.outer_iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    let delta: Vec<f64> = (0..DIM)
                        .map(|axis| {
                            r[axis] as f64
                                + if matches!(gauge, Gauge::Atom) {
                                    model.orb[[j % model.norb(), axis]]
                                        - model.orb[[i % model.norb(), axis]]
                                } else {
                                    0.0
                                }
                        })
                        .collect();
                    let phase = Complex::new(
                        0.0,
                        TAU * delta.iter().zip(k.iter()).map(|(r, k)| r * k).sum::<f64>(),
                    )
                    .exp();
                    let hopping = phase * model.ham[[ir, i, j]];
                    ham[[i, j]] += hopping;
                    for d in 0..DIM {
                        let displacement = (0..DIM)
                            .map(|axis| delta[axis] * model.lat[[axis, d]])
                            .sum::<f64>();
                        velocity[[d, i, j]] += Complex::new(0.0, displacement) * hopping;
                        if R::HAS_RMATRIX && ir < model.rmatrix.as_array4().len_of(Axis(0)) {
                            position[[d, i, j]] += phase * model.rmatrix.as_array4()[[ir, d, i, j]];
                        }
                    }
                }
            }
        }
        if R::HAS_RMATRIX {
            for d in 0..DIM {
                if matches!(gauge, Gauge::Atom) {
                    for i in 0..n {
                        position[[d, i, i]] = Complex::new(0.0, 0.0);
                    }
                }
                for i in 0..n {
                    for j in 0..n {
                        for l in 0..n {
                            velocity[[d, i, j]] += Complex::<f64>::i()
                                * (ham[[i, l]] * position[[d, l, j]]
                                    - position[[d, i, l]] * ham[[l, j]]);
                        }
                    }
                }
            }
        }
        (velocity, ham)
    }

    fn compare_operators<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
    ) {
        let storage = Array2::from_shape_fn((22, 2 * DIM), |(ik, d)| 0.019 * (ik + 3 * d) as f64);
        let directions = Array2::from_shape_fn((2, DIM), |(p, d)| 0.3 * (p + d + 1) as f64 - 0.7);
        for nk in [0, 1, 11] {
            let points = storage.slice(s![..2*nk;2, ..;2]);
            for gauge in [Gauge::Atom, Gauge::Lattice] {
                let hams = model.gen_ham_batch(&points, gauge);
                let (velocities, vhams) = model.gen_v_batch(&points, gauge);
                let (projected, phams) = model.gen_v_projected_batch(&points, gauge, &directions);
                assert_eq!(hams.dim(), (nk, model.nsta(), model.nsta()));
                assert_eq!(velocities.dim(), (nk, DIM, model.nsta(), model.nsta()));
                for (ik, k) in points.outer_iter().enumerate() {
                    let (vref, href) = operator_reference(model, k, gauge);
                    if nk == 1 {
                        let (single_velocity, single_ham) = model.gen_v(&k, gauge);
                        assert!(
                            single_velocity
                                .iter()
                                .zip(vref.iter())
                                .all(|(a, b)| (a - b).norm() < 2e-12)
                        );
                        assert!(
                            single_ham
                                .iter()
                                .zip(href.iter())
                                .all(|(a, b)| (a - b).norm() < 2e-12)
                        );
                    }
                    for h in [
                        hams.index_axis(Axis(0), ik),
                        vhams.index_axis(Axis(0), ik),
                        phams.index_axis(Axis(0), ik),
                    ] {
                        assert!(
                            h.iter()
                                .zip(href.iter())
                                .all(|(a, b)| (a - b).norm() < 2e-12)
                        );
                    }
                    assert!(
                        velocities
                            .index_axis(Axis(0), ik)
                            .iter()
                            .zip(vref.iter())
                            .all(|(a, b)| (a - b).norm() < 2e-12)
                    );
                    for p in 0..directions.nrows() {
                        let mut expected = Array2::<Complex<f64>>::zeros(href.dim());
                        for d in 0..DIM {
                            expected.scaled_add(
                                Complex::new(directions[[p, d]], 0.0),
                                &vref.index_axis(Axis(0), d),
                            );
                        }
                        assert!(
                            projected
                                .slice(s![ik, p, .., ..])
                                .iter()
                                .zip(expected.iter())
                                .all(|(a, b)| (a - b).norm() < 2e-12)
                        );
                    }
                }
            }
        }
    }

    fn operator_case<const SPIN: bool, const DIM: usize>() {
        let mut bare = complex_model::<SPIN, DIM>();
        compare_operators(&bare);
        let mut position =
            Model::<SPIN, DIM, HasRMatrix>::tb_model(bare.lat.clone(), bare.orb.clone(), None)
                .unwrap();
        position.ham = bare.ham.clone();
        position.hamR = bare.hamR.clone();
        position.rmatrix = HasRMatrix(Array4::from_shape_fn(
            (bare.hamR.nrows(), DIM, bare.nsta(), bare.nsta()),
            |(_, d, i, j)| {
                Complex::new(
                    0.017 * (1 + d + i + j) as f64,
                    0.013 * (i as f64 - j as f64),
                )
            },
        ));
        position.validate().unwrap();
        compare_operators(&position);
        // All array layouts are supported without copying the full R arrays.
        bare.ham.invert_axis(Axis(0));
        bare.hamR.invert_axis(Axis(0));
        compare_operators(&bare);
        bare.ham.swap_axes(1, 2);
        compare_operators(&bare);
        position.ham.invert_axis(Axis(0));
        position.hamR.invert_axis(Axis(0));
        position.rmatrix.0.invert_axis(Axis(0));
        compare_operators(&position);
        position.ham.swap_axes(1, 2);
        position.rmatrix.0.swap_axes(2, 3);
        compare_operators(&position);
        position
            .rmatrix
            .0
            .slice_mut(s![1.., .., .., ..])
            .fill(Complex::new(0.0, 0.0));
        compare_operators(&position);
        bare.hamR = Array2::zeros((0, DIM));
        bare.ham = Array3::zeros((0, bare.nsta(), bare.nsta()));
        compare_operators(&bare);
    }

    #[test]
    fn batched_hamiltonians_and_velocities_match_direct_sums() {
        operator_case::<false, 1>();
        operator_case::<false, 2>();
        operator_case::<false, 3>();
        operator_case::<true, 1>();
        operator_case::<true, 2>();
        operator_case::<true, 3>();
    }

    #[test]
    fn velocity_rejects_position_support_prefix_like_model_validation() {
        let mut model =
            Model::<false, 1, HasRMatrix>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        model.add_hop(1.0, 0, 0, &array![1], None);
        model.rmatrix.0 = model.rmatrix.0.slice(s![..1, .., .., ..]).to_owned();
        assert!(model.validate().is_err());
        for nproj in [0, 1] {
            assert!(
                std::panic::catch_unwind(|| model.gen_v_projected_batch(
                    &Array2::<f64>::zeros((0, 1)),
                    Gauge::Atom,
                    &Array2::zeros((nproj, 1)),
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn batched_eigenvectors_keep_atom_gauge_and_band_order() {
        let model = complex_model::<true, 3>();
        let points = Array2::from_shape_fn((17, 3), |(i, d)| 0.027 * (i + d) as f64);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap();
        let (bands, vectors) = pool.install(|| model.solve_all_parallel(&points));
        let (serial_bands, _) = model.solve_all(&points);
        assert!(
            bands
                .iter()
                .zip(serial_bands.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        for (ik, k) in points.outer_iter().enumerate() {
            let ham = operator_reference(&model, k, Gauge::Atom).1;
            for band in 0..model.nsta() {
                let vector = vectors.slice(s![ik, band, ..]);
                let residual = ham.dot(&vector) - &vector * bands[[ik, band]];
                assert!(residual.iter().all(|value| value.norm() < 2e-12));
            }
        }
    }

    fn complex_model<const SPIN: bool, const DIM: usize>() -> Model<SPIN, DIM> {
        let mut lat = Array2::eye(DIM);
        if DIM > 1 {
            lat[[0, 1]] = 0.37;
        }
        let orb = Array2::from_shape_fn((2, DIM), |(i, axis)| {
            0.13 + 0.17 * i as f64 + 0.11 * axis as f64
        });
        let mut model = Model::tb_model(lat, orb, None).unwrap();
        let nsta = model.nsta();
        model.hamR = Array2::zeros((1 + 2 * DIM, DIM));
        model.ham = Array3::zeros((1 + 2 * DIM, nsta, nsta));
        for i in 0..nsta {
            model.ham[[0, i, i]] = Complex::new(0.2 * i as f64, 0.0);
        }
        for axis in 0..DIM {
            let positive = 1 + 2 * axis;
            let negative = positive + 1;
            model.hamR[[positive, axis]] = 1;
            model.hamR[[negative, axis]] = -1;
            for i in 0..nsta {
                for j in 0..nsta {
                    // Complex hopping, including spin mixing, breaks k -> -k
                    // symmetry so a Fourier-sign error changes the spectrum.
                    let value = Complex::new(
                        0.03 * (1 + axis + i + 2 * j) as f64,
                        0.02 * (2 + 2 * axis + 3 * i + j) as f64,
                    );
                    model.ham[[positive, i, j]] = value;
                    model.ham[[negative, j, i]] = value.conj();
                }
            }
        }
        model.validate().unwrap();
        model
    }

    fn compare_bands<const SPIN: bool, const DIM: usize>(model: &Model<SPIN, DIM>) {
        for threads in [1, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for nk in [0, 1, 7, 17, 129] {
                let storage = Array2::from_shape_fn((2 * nk, 2 * DIM), |(i, axis)| {
                    (0.137 * i as f64 + 0.071 * axis as f64) % 1.0
                });
                // Exercise non-contiguous k-points and incomplete final batches.
                let points = storage.slice(s![..;2, ..;2]);
                let mut expected = Array2::zeros((nk, model.nsta()));
                for (k, mut row) in points.outer_iter().zip(expected.outer_iter_mut()) {
                    row.assign(&model.solve_band_onek(&k));
                }
                let serial = model.solve_band_all(&points);
                let actual = pool.install(|| model.solve_band_all_parallel(&points));
                assert_eq!(actual.dim(), (nk, model.nsta()));
                for result in [&serial, &actual] {
                    for (actual, expected) in result.iter().zip(expected.iter()) {
                        assert!(
                            (actual - expected).abs() < 1e-11,
                            "spin={SPIN} dim={DIM} threads={threads} nk={nk}: {actual} vs {expected}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn batched_bands_match_pointwise_atom_gauge() {
        compare_bands(&complex_model::<false, 1>());
        compare_bands(&complex_model::<false, 2>());
        compare_bands(&complex_model::<false, 3>());
        compare_bands(&complex_model::<true, 1>());
        compare_bands(&complex_model::<true, 2>());
        compare_bands(&complex_model::<true, 3>());
    }

    #[test]
    fn batched_bands_support_noncontiguous_hopping_blocks() {
        let mut model = complex_model::<true, 3>();
        model.ham.invert_axis(Axis(0));
        model.hamR.invert_axis(Axis(0));
        assert!(model.ham.as_slice().is_none());
        compare_bands(&model);
    }

    #[test]
    #[should_panic(expected = "hopping shape must match the model")]
    fn batched_bands_reject_mismatched_hopping_shape_before_blas() {
        let mut model = complex_model::<false, 3>();
        model.ham = Array3::zeros((1, model.nsta(), model.nsta()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| model.solve_band_all_parallel(&Array2::zeros((17, 3))));
    }
}
