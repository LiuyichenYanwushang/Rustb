//! Eigenvalue solver for tight-binding Hamiltonians.
use crate::Gauge;
use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::ndarray_lapack::{eigh_r, eigvalsh_r, eigvalsh_v};
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::sync::OnceLock;

// Total Bloch phase matrix/Hamiltonian batch budget across jobs. Output arrays,
// orbital phases and LAPACK workspace are additional; process even one oversized k-point.
//
// A fixed budget shrinks the per-GEMM batch on large nodes, because the batch is
// `budget / (configured_threads * bytes_per_k)`. The budget is derived once per
// process from cgroup headroom and the host's available memory, and split between
// the processes sharing the node. It bounds the
// Fourier buffers only: the process peak is roughly the budget plus one LAPACK
// copy of H(k) per worker plus the returned arrays.
const FOURIER_BUDGET_FALLBACK: usize = 128 * 1024 * 1024;

// Parse a cgroup memory limit: bytes, or any unlimited spelling. The v2 `max`
// token and the v1 `PAGE_COUNTER_MAX * PAGE_SIZE` value (about 2^63) mean
// unlimited (u64::MAX), distinct from an unreadable or malformed value (None).
fn parse_cgroup_limit(text: &str) -> Option<u64> {
    if text.trim() == "max" {
        return Some(u64::MAX);
    }
    let bytes: u64 = text.trim().parse().ok()?;
    Some(if bytes < (1u64 << 62) {
        bytes
    } else {
        u64::MAX
    })
}

// Parse `MemAvailable` from /proc/meminfo, which is reported in kB.
fn parse_meminfo_available(text: &str) -> Option<u64> {
    let line = text
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

// Positive decimal environment variable. Missing, zero and malformed values all
// read as "not set", so a typo never silently disables parallelism.
fn positive_env(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    let parsed = value.trim().parse::<u64>().ok()?;
    (parsed > 0).then_some(parsed)
}

// cgroup directories named by /proc/self/cgroup: the unified v2 line, or the v1
// line that owns the memory controller.
fn parse_self_cgroup(text: &str) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    for line in text.lines() {
        let mut fields = line.splitn(3, ':');
        let (Some(hierarchy), Some(controllers), Some(path)) =
            (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        if (hierarchy == "0" || controllers.split(',').any(|name| name == "memory"))
            && !paths.iter().any(|known| known == path)
        {
            paths.push(path.to_string());
        }
    }
    paths
}

// "/a/b" -> ["/", "/a", "/a/b"]: a cgroup limit applies to every descendant.
fn cgroup_ancestors(path: &str) -> Vec<String> {
    let mut prefixes = vec![String::from("/")];
    let mut current = String::new();
    for component in path.split('/').filter(|part| !part.is_empty()) {
        current.push('/');
        current.push_str(component);
        prefixes.push(current.clone());
    }
    prefixes
}

// Read Linux memory data; the injected reader keeps tests independent of
// mutable host state without changing process-wide environment variables.
fn detected_memory_ceiling() -> Option<u64> {
    detected_memory_ceiling_with(|path| std::fs::read_to_string(path).ok())
}

// Each finite ancestor contributes its remaining allowance, not its total
// limit. A fully readable unlimited hierarchy can use MemAvailable. An
// unlimited root alone cannot establish that an inaccessible child is unlimited.
fn detected_memory_ceiling_with(read: impl Fn(&str) -> Option<String>) -> Option<u64> {
    let mut directories = vec![String::from("/")];
    let mut named_cgroup = false;
    if let Some(text) = read("/proc/self/cgroup") {
        for path in parse_self_cgroup(&text) {
            named_cgroup |= path != "/";
            for ancestor in cgroup_ancestors(&path) {
                if !directories.contains(&ancestor) {
                    directories.push(ancestor);
                }
            }
        }
    }
    let mut complete = true;
    let mut missing_usage = false;
    let mut headroom: Option<u64> = None;
    for directory in directories {
        let directory = directory.trim_end_matches('/');
        let remaining = [
            (
                format!("/sys/fs/cgroup{directory}"),
                "memory.max",
                "memory.current",
            ),
            (
                format!("/sys/fs/cgroup/memory{directory}"),
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            ),
        ]
        .into_iter()
        .filter_map(|(base, limit_file, usage_file)| {
            let limit = parse_cgroup_limit(&read(&format!("{base}/{limit_file}"))?)?;
            if limit == u64::MAX {
                return Some(limit);
            }
            match read(&format!("{base}/{usage_file}"))
                .and_then(|text| text.trim().parse::<u64>().ok())
            {
                Some(used) => Some(limit.saturating_sub(used)),
                None => {
                    missing_usage = true;
                    None
                }
            }
        })
        .min();
        // Physical cgroup-v2 roots have no memory.max/current files. A
        // namespace root may expose them; use those bounds when present.
        complete &= directory.is_empty() || remaining.is_some();
        if let Some(remaining) = remaining {
            headroom = Some(headroom.map_or(remaining, |known| known.min(remaining)));
        }
    }
    let available = read("/proc/meminfo").and_then(|text| parse_meminfo_available(&text));
    if missing_usage || (named_cgroup && !complete) {
        return None;
    }
    match headroom {
        Some(u64::MAX) | None => available,
        Some(remaining) => Some(available.map_or(remaining, |host| host.min(remaining))),
    }
}

// Resolve one budget from an explicit override (bytes), the detected ceiling and
// the number of co-resident processes. No policy cap/floor applies to detected
// memory. Keep at least one byte and respect the target's addressable size.
fn resolve_fourier_budget(
    override_bytes: Option<u64>,
    detected: Option<u64>,
    local_processes: u64,
) -> usize {
    let bytes = override_bytes
        .filter(|bytes| *bytes > 0)
        .unwrap_or_else(|| {
            detected.map_or(FOURIER_BUDGET_FALLBACK as u64, |ceiling| {
                ceiling / local_processes.max(1)
            })
        });
    bytes.clamp(1, isize::MAX as u64) as usize
}

// Per-node process count exported by common launchers: SLURM, Open MPI,
// MVAPICH2 and Intel MPI. Every rank measures the same node, so the share has to
// be divided between them rather than granted once per rank. A launcher that
// reports nothing leaves the default of one process; threads are not ranks.
// Conflicting launchers use the largest count, so Slurm cannot hide MPI ranks.
fn local_process_count(read: impl Fn(&str) -> Option<u64>) -> u64 {
    [
        "SLURM_NTASKS_PER_NODE",
        "OMPI_COMM_WORLD_LOCAL_SIZE",
        "MV2_COMM_WORLD_LOCAL_SIZE",
        "MPI_LOCALNRANKS",
    ]
    .into_iter()
    .filter_map(read)
    .max()
    .unwrap_or(1)
}

// Budget for Bloch phase matrices and H(k) batches, resolved once per process:
// one run keeps identical batching end to end. The inputs are machine state, so
// two runs with different visible memory may still batch differently, and a
// one-point batch sums with zaxpy instead of zgemm; pin
// `RUSTB_FOURIER_MEMORY_MIB` (and `RAYON_NUM_THREADS`, since the batch also
// divides by the worker count) when identical last bits matter.
fn fourier_memory_budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        // Exact per-process budget in MiB; the escape hatch for callers that know
        // their share better than any detection can.
        let override_bytes =
            positive_env("RUSTB_FOURIER_MEMORY_MIB").and_then(|mib| mib.checked_mul(1024 * 1024));
        resolve_fourier_budget(
            override_bytes,
            detected_memory_ceiling(),
            local_process_count(positive_env),
        )
    })
}

// Return (k-points per coarse job, k-points per GEMM). Each worker gets budget/N
// for configured N, even when fewer jobs exist or only the tail remains. Never
// redistribute idle workers' shares. One oversized point is still attempted;
// reduce concurrency when necessary to fit those points in the total budget.
fn batch_plan(
    nk: usize,
    nr: usize,
    nsta: usize,
    threads: usize,
    budget: usize,
) -> Result<(usize, usize)> {
    if nk == 0 {
        return Ok((1, 1));
    }
    let bytes_per_k = nsta
        .checked_mul(nsta)
        .and_then(|width| width.checked_add(nr))
        .and_then(|width| width.checked_mul(size_of::<Complex<f64>>()))
        .ok_or_else(|| TbError::Other("batch buffer size overflow".into()))?;
    let live_points = (budget / bytes_per_k.max(1)).max(1);
    let workers = threads.max(1).min(nk).min(live_points);
    let job_size = nk.div_ceil(workers);
    let batch_size = (budget / threads.max(1) / bytes_per_k.max(1)).max(1);
    Ok((job_size, job_size.min(batch_size)))
}

// Finite model data can still overflow during the Fourier sum or gauge transform.
fn check_finite_hamiltonian<S: Data<Elem = Complex<f64>>>(ham: &ArrayBase<S, Ix2>) -> Result<()> {
    if ham.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
        return Err(TbError::Other(
            "H(k) contains nonfinite matrix elements".into(),
        ));
    }
    Ok(())
}

#[inline(always)]
fn diagonalize_with_vectors<S: Data<Elem = Complex<f64>>>(
    ham: &ArrayBase<S, Ix2>,
) -> Result<(Array1<f64>, Array2<Complex<f64>>)> {
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
    check_finite_hamiltonian(ham)?;
    let (energies, vectors) = ham.eigh(UPLO::Lower)?;
    Ok((
        energies,
        conjugate::<Complex<f64>, OwnedRepr<Complex<f64>>>(&vectors),
    ))
}

/// Solve the tight-binding Hamiltonian H(k).
///
/// 本征值按能量升序排列；Rustb 返回的本征矢以“能带为行、基底分量为列”。
/// 完整的数组轴、共轭、规范及基底变换约定见 [`Solve::solve_onek`] 和
/// [`Solve::solve_all`]。
///
/// All methods return errors for invalid models, incorrectly sized or nonfinite
/// k-points, nonfinite generated Hamiltonians, and eigensolver failures.
/// Models must pass [`Model::validate`] and have Hermitian H(k); Hermiticity
/// remains the caller's responsibility. Batched methods validate the model once
/// per call and preserve input k-point order, including for empty batches.
pub trait Solve {
    /// Solve energy bands at a single k-point.
    ///
    /// 返回长度为 `nsta` 的实数数组，按能量非递减排列：`E[0] <= E[1] <= ...`。
    /// 这是所有态的统一能量排序，不是轨道顺序、自旋分块顺序或按能量绝对值排序。
    /// Errors follow the shared [`Solve`] contract.
    fn solve_band_onek<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix1>)
    -> Result<Array1<f64>>;
    /// Solve only energies in the half-open energy window `(low, high]`.
    ///
    /// `energy_window` bounds use the model's energy units (usually eV).
    /// `tolerance` is LAPACK's absolute energy convergence tolerance in those
    /// same units; nonpositive values select LAPACK's default tolerance.
    /// Returns ascending energies, or an empty array if no band is selected.
    ///
    /// # Errors
    /// In addition to the shared [`Solve`] errors, rejects nonfinite or unordered
    /// window bounds and a nonfinite tolerance. See [`Self::solve_range_onek`]
    /// for the counterpart that also returns eigenvectors.
    fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy_window: (f64, f64),
        tolerance: f64,
    ) -> Result<Array1<f64>>;
    /// Solve energy bands at all given k-points.
    ///
    /// 返回形状 `(nk, nsta)`，`bands[[ik, n]]` 对应输入第 `ik` 个 k 点的
    /// 第 `n` 小能量。每个 k 点独立升序排序，不进行跨 k 点的能带追踪。
    /// Batches use the same construction and adaptive memory budget as
    /// [`Solve::solve_band_all_parallel`]. A serial call uses one worker's share
    /// from the current Rayon pool, including inside an outer parallel loop.
    /// Errors follow the shared [`Solve`] contract.
    fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>;
    /// Solve energy bands in parallel, using bounded batches of Fourier sums.
    /// Each job processes a contiguous k-point range and diagonalizes individual
    /// H(k) matrices before building its next batch. Returns eigenvalues in
    /// input k-point order. Configure BLAS threading separately from Rayon.
    ///
    /// Each worker's Fourier buffer target is `U / N`, where `N` is the current
    /// Rayon pool size, fixed throughout the call. Serial solvers use one share.
    /// Idle workers' shares are never redistributed, including at the tail.
    /// `U` is the process's available-memory allowance: the minimum of Linux
    /// `MemAvailable` and readable cgroup `limit - usage`, divided by the largest
    /// reported launcher-local process count (default one). Fully readable
    /// unlimited cgroups use host availability. Detection targets Linux;
    /// unavailable memory data or unknown named hierarchies fall back to a
    /// total `U` of 128 MiB. There is no fixed cap or detected-memory floor.
    ///
    /// `U` is cached once per process; `N` is taken from each call's thread pool.
    /// Set `RUSTB_FOURIER_MEMORY_MIB` before the first solve to override total
    /// `U`, not the per-worker share. Positive MiB values are used up to the
    /// target's addressable size; zero, malformed or overflowing values are
    /// ignored. Pin both this override and `RAYON_NUM_THREADS` for repeatable
    /// batching across runs. Hidden container limits require an explicit budget.
    ///
    /// This is a scheduling target, not a hard memory limit. One oversized
    /// k-point is still attempted, with fewer concurrent jobs if needed; models
    /// are not rejected solely for exceeding the target. Returned arrays,
    /// orbital phases and LAPACK workspace are additional, so allocating the
    /// Fourier buffers does not guarantee the complete solve fits in memory.
    /// Shape and energy ordering are identical to [`Solve::solve_band_all`].
    fn solve_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<f64>>;
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
    /// 要求模型的完整 H 为 Hermitian；求解器只使用一个三角，不会验证另一侧是否一致。
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
    /// # 相位、简并与能带连续性
    ///
    /// 各本征矢已归一化，但整体复相位任意。简并子空间内可以返回任意正交基，
    /// 简并态之间没有固定排序；本方法也不按轨道成分、自旋或相邻 k 点的重叠追踪能带。
    /// 比较不同求解器或调用的结果时，应检查本征方程、态的重叠或简并子空间投影，
    /// 不应要求本征矢逐元素一致。需要连续本征矢或数值 k 导数时，应自行处理相位/
    /// 子空间对齐；Berry 相位和 Wilson loop 也可以用规范不变的重叠方法计算，
    /// 不要求先将每个态的相位调成连续。
    ///
    /// # Errors
    /// Returns the shared [`Solve`] errors.
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
    /// let (energies, evec) = model.solve_onek(&k)?;
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
    /// # Ok::<(), Rustb::error::TbError>(())
    /// ```
    fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> Result<(Array1<f64>, Array2<Complex<f64>>)>;
    /// Solve energies and eigenvectors in the half-open energy window `(low, high]`.
    ///
    /// Returns ascending energies and row ket coefficients with shape
    /// `(number_of_selected_bands, nsta)`, in the same gauge as [`Self::solve_onek`].
    /// `energy_window` and `tolerance` use the model's energy units (usually eV).
    /// `tolerance` is LAPACK's absolute convergence tolerance; nonpositive values
    /// select its default. No selected bands gives empty energies and a
    /// `(0, nsta)` eigenvector matrix. See [`Self::solve_band_range_onek`] for
    /// the energy-only counterpart.
    ///
    /// # Errors
    /// In addition to the shared [`Solve`] errors, rejects nonfinite or unordered
    /// window bounds and a nonfinite tolerance.
    fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy_window: (f64, f64),
        tolerance: f64,
    ) -> Result<(Array1<f64>, Array2<Complex<f64>>)>;
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
    /// let (energies, evec) = model.solve_all(&points)?;
    /// assert_eq!(energies.dim(), (3, model.nsta()));
    /// assert_eq!(evec.dim(), (3, model.nsta(), model.nsta()));
    /// for ik in 0..points.nrows() {
    ///     let k = points.row(ik);
    ///     let h = model.gen_ham(&k, Gauge::Atom);
    ///     let (one_energy, _) = model.solve_onek(&k)?;
    ///     for n in 0..model.nsta() {
    ///         assert!((energies[[ik, n]] - one_energy[n]).abs() < 1e-12);
    ///         let ket = evec.slice(s![ik, n, ..]);
    ///         let residual = h.dot(&ket) - &ket * energies[[ik, n]];
    ///         assert!(residual.iter().all(|z| z.norm() < 1e-12));
    ///     }
    /// }
    /// let (parallel_energy, _) = model.solve_all_parallel(&points)?;
    /// assert!(parallel_energy.iter().zip(&energies).all(|(a, b)| (a - b).abs() < 1e-12));
    /// # Ok::<(), Rustb::error::TbError>(())
    /// ```
    fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array3<Complex<f64>>)>;
    /// Parallel counterpart of [`Solve::solve_all`], with the same return convention.
    ///
    /// `energies[[ik, n]]` 配对 `evec[[ik, n, alpha]]`，能量在每个 k 点内升序，
    /// k 点保留输入顺序，本征矢使用 Atom 规范且按行存放 ket 系数。
    /// 并行完成顺序不改变输出轴顺序；不进行能带追踪或相位对齐。
    /// 数组形状、变换公式及简并态比较方法详见 [`Solve::solve_all`]、[`Solve::solve_onek`]。
    ///
    /// Batching follows [`Solve::solve_band_all_parallel`], including its adaptive
    /// `RUSTB_FOURIER_MEMORY_MIB` budget; the returned eigenvectors are an
    /// additional `nk * nsta²` allocation outside that budget and usually dominate
    /// the peak for a large basis. The serial [`Solve::solve_all`] shares the same
    /// policy and uses one worker's fixed share from the current Rayon pool.
    fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array3<Complex<f64>>)>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Solve for Model<SPIN, DIM, R> {
    #[allow(non_snake_case)]
    #[inline(always)]
    fn solve_band_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> Result<Array1<f64>> {
        self.validate_solver_input(kvec.view().insert_axis(Axis(0)))?;
        let hamk = self.gen_ham(kvec, Gauge::Atom);
        check_finite_hamiltonian(&hamk)?;
        eigvalsh_v(&hamk, UPLO::Upper)
    }

    fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy_window: (f64, f64),
        tolerance: f64,
    ) -> Result<Array1<f64>> {
        self.validate_solver_input(kvec.view().insert_axis(Axis(0)))?;
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        check_finite_hamiltonian(&hamk)?;
        eigvalsh_r(&hamk, energy_window, tolerance, UPLO::Upper)
    }
    fn solve_band_all<S: Data<Elem = f64>>(&self, kvec: &ArrayBase<S, Ix2>) -> Result<Array2<f64>> {
        self.solve_band_batches(kvec, false)
    }
    #[allow(non_snake_case)]
    fn solve_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<Array2<f64>> {
        self.solve_band_batches(kvec, true)
    }
    #[allow(non_snake_case)]
    #[inline(always)]
    fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> Result<(Array1<f64>, Array2<Complex<f64>>)> {
        self.validate_solver_input(kvec.view().insert_axis(Axis(0)))?;
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        diagonalize_with_vectors(&hamk)
    }
    fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy_window: (f64, f64),
        tolerance: f64,
    ) -> Result<(Array1<f64>, Array2<Complex<f64>>)> {
        self.validate_solver_input(kvec.view().insert_axis(Axis(0)))?;
        let hamk = self.gen_ham(&kvec, Gauge::Atom);
        check_finite_hamiltonian(&hamk)?;
        eigh_r(&hamk, energy_window, tolerance, UPLO::Upper)
    }

    #[allow(non_snake_case)]
    fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array3<Complex<f64>>)> {
        self.solve_eigenvector_batches(kvec, false)
    }
    #[allow(non_snake_case)]
    fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array3<Complex<f64>>)> {
        self.solve_eigenvector_batches(kvec, true)
    }
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> Model<SPIN, DIM, R> {
    fn validate_solver_input(&self, points: ArrayView2<'_, f64>) -> Result<()> {
        if points.ncols() != DIM {
            return Err(TbError::DimensionMismatch {
                context: "solver k-point dimension".into(),
                expected: DIM,
                found: points.ncols(),
            });
        }
        if points.iter().any(|value| !value.is_finite()) {
            return Err(TbError::Other("solver k-points must be finite".into()));
        }
        self.validate()
    }

    fn solve_band_batches<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        parallel: bool,
    ) -> Result<Array2<f64>> {
        self.validate_solver_input(points.view())?;
        let (job_size, batch_size) = self.solver_batch_plan(points.nrows(), parallel)?;
        let mut bands = Array2::zeros((points.nrows(), self.nsta()));
        let solve =
            |(points, mut energies): (ArrayView2<'_, f64>, ArrayViewMut2<'_, f64>)| -> Result<()> {
                for (batch, mut output) in points
                    .axis_chunks_iter(Axis(0), batch_size)
                    .zip(energies.axis_chunks_iter_mut(Axis(0), batch_size))
                {
                    let hams = self.gen_ham_batch(&batch, Gauge::Lattice);
                    for (ham, mut row) in hams.outer_iter().zip(output.outer_iter_mut()) {
                        check_finite_hamiltonian(&ham)?;
                        row.assign(&eigvalsh_v(&ham, UPLO::Upper)?);
                    }
                }
                Ok(())
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
                .try_for_each(solve)?;
        } else {
            points
                .axis_chunks_iter(Axis(0), job_size)
                .zip(bands.axis_chunks_iter_mut(Axis(0), job_size))
                .try_for_each(solve)?;
        }
        Ok(bands)
    }

    fn solve_eigenvector_batches<S: Data<Elem = f64>>(
        &self,
        points: &ArrayBase<S, Ix2>,
        parallel: bool,
    ) -> Result<(Array2<f64>, Array3<Complex<f64>>)> {
        self.validate_solver_input(points.view())?;
        let (job_size, batch_size) = self.solver_batch_plan(points.nrows(), parallel)?;
        let mut bands = Array2::zeros((points.nrows(), self.nsta()));
        let mut vectors = Array3::zeros((points.nrows(), self.nsta(), self.nsta()));
        let solve = |((points, mut energies), mut vectors): (
            (ArrayView2<'_, f64>, ArrayViewMut2<'_, f64>),
            ArrayViewMut3<'_, Complex<f64>>,
        )|
         -> Result<()> {
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
                    let (energies, eigenvectors) = diagonalize_with_vectors(&ham)?;
                    row.assign(&energies);
                    state.assign(&eigenvectors);
                }
            }
            Ok(())
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
                .try_for_each(solve)?;
        } else {
            points
                .axis_chunks_iter(Axis(0), job_size)
                .zip(bands.axis_chunks_iter_mut(Axis(0), job_size))
                .zip(vectors.axis_chunks_iter_mut(Axis(0), job_size))
                .try_for_each(solve)?;
        }
        Ok((bands, vectors))
    }

    // Serial solves may run inside an outer Rayon loop (e.g. Berry loops).
    // They receive one U/N share, not an entire process allowance each.
    fn solver_batch_plan(&self, nk: usize, parallel: bool) -> Result<(usize, usize)> {
        let configured_threads = rayon::current_num_threads();
        let total = fourier_memory_budget();
        let (threads, budget) = if parallel {
            (configured_threads, total)
        } else {
            (1, (total / configured_threads).max(1))
        };
        batch_plan(nk, self.hamR.nrows(), self.nsta(), threads, budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HasRMatrix, Velocity};
    use std::f64::consts::TAU;

    fn solve_at_every_entry(model: &Model<false, 1>, k: &Array1<f64>) -> [Result<()>; 8] {
        let points = k.view().insert_axis(Axis(0));
        [
            model.solve_band_onek(k).map(|_| ()),
            model.solve_band_range_onek(k, (-1.0, 1.0), 0.0).map(|_| ()),
            model.solve_onek(k).map(|_| ()),
            model.solve_range_onek(k, (-1.0, 1.0), 0.0).map(|_| ()),
            model.solve_band_all(&points).map(|_| ()),
            model.solve_band_all_parallel(&points).map(|_| ()),
            model.solve_all(&points).map(|_| ()),
            model.solve_all_parallel(&points).map(|_| ()),
        ]
    }

    #[test]
    fn solvers_return_errors_for_invalid_points_and_models() {
        let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        for k in [array![], array![0.0, 0.0]] {
            for result in solve_at_every_entry(&model, &k) {
                assert!(matches!(result, Err(TbError::DimensionMismatch {
                    expected: 1, found, ..
                }) if found == k.len()));
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for result in solve_at_every_entry(&model, &array![value]) {
                assert!(
                    matches!(result, Err(TbError::Other(message)) if message.contains("k-points must be finite"))
                );
            }
        }
        model.ham = Array3::zeros((2, 1, 1));
        for result in solve_at_every_entry(&model, &array![0.0]) {
            assert!(matches!(
                result,
                Err(TbError::InvalidModelInvariant {
                    invariant: "hamiltonian_shape",
                    ..
                })
            ));
        }
        model.ham = Array3::from_elem((1, 1, 1), Complex::new(f64::NAN, 0.0));
        for result in solve_at_every_entry(&model, &array![0.0]) {
            assert!(matches!(
                result,
                Err(TbError::InvalidModelInvariant {
                    invariant: "finite_hamiltonian",
                    ..
                })
            ));
        }
    }

    #[test]
    fn empty_solver_batches_keep_shapes_and_validate_inputs() {
        let model = complex_model::<true, 2>();
        let points = Array2::<f64>::zeros((0, 2));
        for bands in [
            model.solve_band_all(&points),
            model.solve_band_all_parallel(&points),
        ] {
            assert_eq!(bands.unwrap().dim(), (0, model.nsta()));
        }
        for result in [model.solve_all(&points), model.solve_all_parallel(&points)] {
            let (energies, vectors) = result.unwrap();
            assert_eq!(energies.dim(), (0, model.nsta()));
            assert_eq!(vectors.dim(), (0, model.nsta(), model.nsta()));
        }
        let wrong_shape = Array2::<f64>::zeros((0, 3));
        for result in [
            model.solve_band_all(&wrong_shape).map(|_| ()),
            model.solve_band_all_parallel(&wrong_shape).map(|_| ()),
            model.solve_all(&wrong_shape).map(|_| ()),
            model.solve_all_parallel(&wrong_shape).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(TbError::DimensionMismatch {
                    expected: 2,
                    found: 3,
                    ..
                })
            ));
        }
        let mut invalid_model = model;
        invalid_model.ham = Array3::zeros((1, 4, 4));
        for result in [
            invalid_model.solve_band_all(&points).map(|_| ()),
            invalid_model.solve_band_all_parallel(&points).map(|_| ()),
            invalid_model.solve_all(&points).map(|_| ()),
            invalid_model.solve_all_parallel(&points).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(TbError::InvalidModelInvariant {
                    invariant: "hamiltonian_shape",
                    ..
                })
            ));
        }
    }

    #[test]
    fn range_solvers_use_energy_window_and_tolerance_contract() {
        let mut model =
            Model::<false, 1>::tb_model(array![[1.0]], array![[0.0], [0.0], [0.0]], None).unwrap();
        model.set_onsite(&array![-1.0, 0.0, 2.0], None);
        let k = array![0.0];
        for tolerance in [-1.0, 0.0, 1e-12] {
            assert_eq!(
                model
                    .solve_band_range_onek(&k, (-1.0, 2.0), tolerance)
                    .unwrap(),
                array![0.0, 2.0]
            );
            let (energies, vectors) = model.solve_range_onek(&k, (-1.0, 2.0), tolerance).unwrap();
            assert_eq!(energies, array![0.0, 2.0]);
            assert_eq!(vectors.dim(), (2, 3));
            assert!(
                model
                    .solve_band_range_onek(&k, (3.0, 4.0), tolerance)
                    .unwrap()
                    .is_empty()
            );
            let (empty, vectors) = model.solve_range_onek(&k, (3.0, 4.0), tolerance).unwrap();
            assert!(empty.is_empty());
            assert_eq!(vectors.dim(), (0, 3));
        }
        for (window, tolerance) in [
            ((1.0, 1.0), 0.0),
            ((2.0, -1.0), 0.0),
            ((f64::NAN, 2.0), 0.0),
            ((-1.0, f64::INFINITY), 0.0),
            ((-1.0, 2.0), f64::NAN),
            ((-1.0, 2.0), f64::INFINITY),
        ] {
            assert!(matches!(
                model.solve_band_range_onek(&k, window, tolerance),
                Err(TbError::Other(_))
            ));
            assert!(matches!(
                model.solve_range_onek(&k, window, tolerance),
                Err(TbError::Other(_))
            ));
        }
    }

    #[test]
    fn solvers_propagate_fourier_overflow_from_worker_jobs() {
        let mut model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        model.set_hop(1e308, 0, 0, &array![1], None);
        model.validate().unwrap(); // finite R blocks; H(0) = 2e308 overflows
        for result in solve_at_every_entry(&model, &array![0.0]) {
            assert!(
                matches!(result, Err(TbError::Other(message)) if message.contains("H(k) contains nonfinite"))
            );
        }
        let points = array![[0.25], [0.25], [0.25], [0.0], [0.25]];
        for threads in [1, 3] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for result in pool.install(|| {
                [
                    model.solve_band_all(&points).map(|_| ()),
                    model.solve_band_all_parallel(&points).map(|_| ()),
                    model.solve_all(&points).map(|_| ()),
                    model.solve_all_parallel(&points).map(|_| ()),
                ]
            }) {
                assert!(
                    matches!(result, Err(TbError::Other(message)) if message.contains("H(k) contains nonfinite"))
                );
            }
        }
    }

    #[test]
    fn range_solver_preserves_complex_ket_convention() {
        let model = complex_model::<true, 3>();
        let k = array![0.17, 0.31, 0.07];
        let ham = model.gen_ham(&k, Gauge::Atom);
        let full = model.solve_band_onek(&k).unwrap();
        let lower = (full[0] + full[1]) * 0.5;
        let (energies, vectors) = model.solve_range_onek(&k, (lower, 10.0), 0.0).unwrap();
        assert_eq!(energies.len(), model.nsta() - 1);
        for (n, ket) in vectors.outer_iter().enumerate() {
            assert!((energies[n] - full[n + 1]).abs() < 1e-12);
            let residual = ham.dot(&ket) - &ket * energies[n];
            assert!(residual.iter().all(|z| z.norm() < 1e-12));
        }
    }

    #[test]
    fn batch_plan_bounds_memory_and_job_count() {
        for budget in [FOURIER_BUDGET_FALLBACK, 1 << 30, 7, 1] {
            for nk in [0, 1, 128, 129, 4096, 29791] {
                for threads in [1, 3, 8, 256] {
                    // (0, 0) only pins "no panic without buffers": its per-point
                    // cost is zero, so the budget bound is vacuous there.
                    for (nr, nsta) in [(0, 0), (7, 4), (841, 200), (1, 4096)] {
                        let per_k = (nr + nsta * nsta) * size_of::<Complex<f64>>();
                        let (job_size, batch_size) =
                            batch_plan(nk, nr, nsta, threads, budget).unwrap();
                        let jobs = nk.div_ceil(job_size);
                        assert!(jobs <= threads);
                        assert!(
                            jobs * batch_size * per_k <= budget
                                || (jobs <= 1 && batch_size == 1 && budget < per_k)
                        );
                        assert!(batch_size > 0 && job_size > 0);
                        assert!(batch_size * per_k <= budget / threads || batch_size == 1);
                    }
                }
            }
        }
        // Automatic batches are not capped at the previous hard-coded 16.
        assert!(batch_plan(4096, 841, 200, 8, 1 << 30).unwrap().1 > 16);
        assert!(batch_plan(1, 1, usize::MAX, 1, FOURIER_BUDGET_FALLBACK).is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn worker_shares_scale_with_memory_and_configured_threads() {
        // Planning only: 4096^2 complex numbers are 256 MiB per point.
        // No large matrices are allocated. The node examples are fixed inputs.
        let small = resolve_fourier_budget(None, Some(256 << 30), 1);
        let large = resolve_fourier_budget(None, Some(1536 << 30), 1);
        assert_eq!(small / 64, 4 << 30);
        assert_eq!(large / 64, 24 << 30);
        for (threads, small_batch, large_batch) in
            [(64, 16, 96), (96, 10, 64), (128, 8, 48), (256, 4, 24)]
        {
            assert_eq!(
                batch_plan(65536, 0, 4096, threads, small).unwrap().1,
                small_batch
            );
            assert_eq!(
                batch_plan(65536, 0, 4096, threads, large).unwrap().1,
                large_batch
            );
            // Fewer points never increase the per-worker share. A tail uses
            // the original plan; only the final chunk's length shrinks.
            for nk in [1, 2, threads - 1, threads + 1, 65537] {
                let (job, batch) = batch_plan(nk, 0, 4096, threads, small).unwrap();
                assert!(batch <= small_batch);
                assert!(nk.div_ceil(job) <= threads);
            }
        }
        // A point beyond even the total allowance still runs alone; the
        // planner does not impose a model-size rejection at the fallback.
        assert_eq!(
            batch_plan(100, 0, 4096, 64, FOURIER_BUDGET_FALLBACK).unwrap(),
            (100, 1)
        );
    }

    #[test]
    fn serial_and_parallel_plans_use_the_calling_pools_worker_share() {
        let model = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        let total = fourier_memory_budget();
        // Plan only; this ensures job length does not truncate the batch.
        let nk = usize::MAX / 16;
        for threads in [1, 3] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let expected = (total / threads / 32).max(1); // H plus one Bloch phase
                    let (serial_job, serial_batch) = model.solver_batch_plan(nk, false).unwrap();
                    let (_, parallel_batch) = model.solver_batch_plan(nk, true).unwrap();
                    assert_eq!(serial_job, nk);
                    assert_eq!(serial_batch, expected);
                    assert_eq!(parallel_batch, expected);
                    assert_eq!(fourier_memory_budget(), total);
                });
        }
    }

    #[test]
    fn memory_ceiling_parsers_distinguish_unlimited_and_convert_kib() {
        assert_eq!(parse_cgroup_limit("1073741824\n"), Some(1 << 30));
        assert_eq!(parse_cgroup_limit("max\n"), Some(u64::MAX));
        assert_eq!(parse_cgroup_limit("9223372036854771712\n"), Some(u64::MAX));
        assert_eq!(parse_cgroup_limit(""), None);
        assert_eq!(parse_cgroup_limit("not a number\n"), None);

        let meminfo = "MemTotal:       65800000 kB\nMemAvailable:   63000000 kB\n";
        assert_eq!(parse_meminfo_available(meminfo), Some(63000000 * 1024));
        assert_eq!(parse_meminfo_available("MemTotal: 65800000 kB\n"), None);
        assert_eq!(parse_meminfo_available("MemAvailable: huge kB\n"), None);
        assert_eq!(
            parse_meminfo_available("MemAvailable: 18446744073709551615 kB\n"),
            None // checked_mul overflow, not a wrapped ceiling
        );
    }

    #[test]
    fn self_cgroup_paths_cover_unified_and_memory_controller_lines() {
        assert_eq!(
            parse_self_cgroup("0::/user.slice/session-1.scope\n"),
            ["/user.slice/session-1.scope"]
        );
        assert_eq!(
            parse_self_cgroup("11:memory:/docker/abc\n10:cpu:/docker/abc\n"),
            ["/docker/abc"]
        );
        // A line whose controllers do not include `memory` names no memory
        // cgroup, and two qualifying lines for the same path collapse to one.
        assert!(parse_self_cgroup("10:cpu:/only-cpu\n").is_empty());
        assert_eq!(
            parse_self_cgroup("0::/shared\n11:memory,hugetlb:/shared\n"),
            ["/shared"]
        );
        assert_eq!(parse_self_cgroup("0::/\n"), ["/"]);
        assert!(parse_self_cgroup("no colons here\n").is_empty());
        assert_eq!(
            cgroup_ancestors("/user.slice/session-1.scope"),
            ["/", "/user.slice", "/user.slice/session-1.scope"]
        );
        assert_eq!(cgroup_ancestors("/"), ["/"]);
    }

    #[test]
    fn cgroup_memory_ceiling_uses_visible_remaining_allowance() {
        let mut files = std::collections::BTreeMap::from([
            ("/proc/self/cgroup", "0::/job/step\n"),
            ("/proc/meminfo", "MemAvailable: 16 kB\n"),
            ("/sys/fs/cgroup/memory.max", "max\n"),
            ("/sys/fs/cgroup/job/memory.max", "max\n"),
            ("/sys/fs/cgroup/job/step/memory.max", "max\n"),
        ]);
        let detect = |files: &std::collections::BTreeMap<&str, &str>| {
            detected_memory_ceiling_with(|path| files.get(path).map(|text| text.to_string()))
        };
        // Only a fully readable unlimited hierarchy can borrow MemAvailable.
        assert_eq!(detect(&files), Some(16 * 1024));
        // The physical v2 root has no memory.max or memory.current files.
        files.remove("/sys/fs/cgroup/memory.max");
        assert_eq!(detect(&files), Some(16 * 1024));
        files.insert("/sys/fs/cgroup/memory.max", "max");
        files.remove("/sys/fs/cgroup/job/step/memory.max");
        assert_eq!(detect(&files), None);
        files.insert("/sys/fs/cgroup/job/step/memory.max", "malformed");
        assert_eq!(detect(&files), None);
        files.insert("/sys/fs/cgroup/job/step/memory.max", "max");

        // The finite parent constrains the unlimited child by its headroom.
        files.insert("/sys/fs/cgroup/job/memory.max", "32768");
        files.insert("/sys/fs/cgroup/job/memory.current", "24576");
        assert_eq!(detect(&files), Some(8192));
        // An inaccessible child may impose a tighter bound than its parent.
        files.remove("/sys/fs/cgroup/job/step/memory.max");
        assert_eq!(detect(&files), None);
        files.insert("/sys/fs/cgroup/job/step/memory.max", "max");
        files.insert("/proc/meminfo", "MemAvailable: 4 kB\n");
        assert_eq!(detect(&files), Some(4096));
        files.remove("/proc/meminfo");
        assert_eq!(detect(&files), Some(8192));

        files.insert("/sys/fs/cgroup/job/step/memory.max", "4096");
        files.insert("/sys/fs/cgroup/job/step/memory.current", "1024");
        assert_eq!(detect(&files), Some(3072));
        files.insert("/sys/fs/cgroup/job/memory.current", "32769");
        assert_eq!(detect(&files), Some(0));

        files.insert("/sys/fs/cgroup/job/step/memory.max", "max");
        files.insert("/proc/meminfo", "MemAvailable: 16 kB\n");
        files.remove("/sys/fs/cgroup/job/memory.current");
        assert_eq!(detect(&files), None);
        // A readable child must not erase a tighter parent with unknown usage.
        files.insert("/sys/fs/cgroup/job/memory.max", "2048");
        files.insert("/sys/fs/cgroup/job/step/memory.max", "4096");
        assert_eq!(detect(&files), None);
    }

    #[test]
    fn cgroup_v1_memory_ceiling_requires_usage_for_finite_limits() {
        for (usage, expected) in [
            (Some("4096\n"), Some(12288)),
            (None, None),
            (Some("malformed\n"), None),
        ] {
            assert_eq!(
                detected_memory_ceiling_with(|path| {
                    match path {
                        "/proc/self/cgroup" => Some("11:memory:/job\n"),
                        "/proc/meminfo" => Some("MemAvailable: 16 kB\n"),
                        "/sys/fs/cgroup/memory/memory.limit_in_bytes" => {
                            Some("9223372036854771712\n")
                        }
                        "/sys/fs/cgroup/memory/job/memory.limit_in_bytes" => Some("16384\n"),
                        "/sys/fs/cgroup/memory/job/memory.usage_in_bytes" => usage,
                        _ => None,
                    }
                    .map(str::to_owned)
                }),
                expected
            );
        }
    }

    #[test]
    fn unknown_root_cgroup_can_use_host_available_memory() {
        for membership in [None, Some("0::/\n")] {
            assert_eq!(
                detected_memory_ceiling_with(|path| {
                    match path {
                        "/proc/self/cgroup" => membership,
                        "/proc/meminfo" => Some("MemAvailable: 16 kB\n"),
                        _ => None,
                    }
                    .map(str::to_owned)
                }),
                Some(16 * 1024)
            );
        }
        assert_eq!(detected_memory_ceiling_with(|_| None), None);
    }

    #[test]
    fn finite_root_cgroup_with_unknown_usage_cannot_use_host_memory() {
        for usage in [None, Some("malformed\n")] {
            assert_eq!(
                detected_memory_ceiling_with(|path| {
                    match path {
                        "/proc/self/cgroup" => Some("0::/\n"),
                        "/proc/meminfo" => Some("MemAvailable: 16 kB\n"),
                        "/sys/fs/cgroup/memory.max" => Some("4096\n"),
                        "/sys/fs/cgroup/memory.current" => usage,
                        _ => None,
                    }
                    .map(str::to_owned)
                }),
                None
            );
        }
    }

    #[test]
    fn local_process_count_uses_largest_launcher_count() {
        for (counts, expected) in [
            ([None, None, None, None], 1),
            ([Some(1), Some(16), None, None], 16),
            ([Some(16), Some(1), None, None], 16),
            ([Some(1), Some(2), Some(8), Some(4)], 8),
            ([Some(1), Some(2), Some(4), Some(8)], 8),
        ] {
            assert_eq!(
                local_process_count(|name| match name {
                    "SLURM_NTASKS_PER_NODE" => counts[0],
                    "OMPI_COMM_WORLD_LOCAL_SIZE" => counts[1],
                    "MV2_COMM_WORLD_LOCAL_SIZE" => counts[2],
                    "MPI_LOCALNRANKS" => counts[3],
                    _ => None,
                }),
                expected,
                "launcher counts: {counts:?}"
            );
        }
    }

    #[test]
    fn fourier_budget_preserves_available_memory_and_explicit_overrides() {
        assert_eq!(resolve_fourier_budget(None, Some(1 << 30), 1), 1 << 30);
        assert_eq!(resolve_fourier_budget(None, Some(32 << 30), 32), 1 << 30);
        // Detected low memory is not inflated to the historical fallback.
        assert_eq!(resolve_fourier_budget(None, Some(64 << 20), 1), 64 << 20);
        assert_eq!(
            resolve_fourier_budget(None, None, 1),
            FOURIER_BUDGET_FALLBACK
        );
        assert_eq!(resolve_fourier_budget(None, Some(512 << 20), 0), 512 << 20);
        assert_eq!(resolve_fourier_budget(None, Some(0), 1), 1);
        assert_eq!(resolve_fourier_budget(None, Some(64 << 20), u64::MAX), 1);
        assert_eq!(resolve_fourier_budget(Some(7), None, 16), 7);
        assert_eq!(
            resolve_fourier_budget(Some(512 << 20), Some(64 << 20), 16),
            512 << 20
        );
        assert_eq!(resolve_fourier_budget(Some(0), Some(64 << 20), 1), 64 << 20);
        assert_eq!(
            resolve_fourier_budget(Some(u64::MAX), None, 1),
            isize::MAX as usize
        );

        // Cache stability must not reread MemAvailable or mutate shared env.
        let budget = fourier_memory_budget();
        assert!(budget > 0);
        assert_eq!(fourier_memory_budget(), budget);
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
        let (bands, vectors) = pool.install(|| model.solve_all_parallel(&points)).unwrap();
        let (serial_bands, _) = model.solve_all(&points).unwrap();
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
                    row.assign(&model.solve_band_onek(&k).unwrap());
                }
                let serial = model.solve_band_all(&points).unwrap();
                let actual = pool
                    .install(|| model.solve_band_all_parallel(&points))
                    .unwrap();
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
    fn batched_bands_reject_mismatched_hopping_shape_before_blas() {
        let mut model = complex_model::<false, 3>();
        model.ham = Array3::zeros((1, model.nsta(), model.nsta()));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        assert!(matches!(
            pool.install(|| model.solve_band_all_parallel(&Array2::zeros((17, 3)))),
            Err(TbError::InvalidModelInvariant {
                invariant: "hamiltonian_shape",
                ..
            })
        ));
    }
}
