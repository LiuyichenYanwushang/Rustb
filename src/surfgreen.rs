//! Surface Green functions for semi-infinite tight-binding systems.
//!
//! This module implements the iterative principal-layer decimation method of
//! López Sancho, López Sancho, and Rubio. It computes the Green functions of the
//! two possible terminations of a periodic bulk, together with the bulk Green
//! function, without constructing a thick finite slab.
//!
//! # Principal-layer Hamiltonian
//!
//! At fixed surface-parallel momentum $\mathbf{k}_\parallel$, a sufficiently
//! large principal layer makes the Hamiltonian block tridiagonal along the
//! surface-normal direction:
//!
//! ```math
//! \mathcal{H}(\mathbf{k}_\parallel)=
//! \begin{pmatrix}
//! H_0 & \alpha_0 & 0 & \cdots \\
//! \beta_0 & H_0 & \alpha_0 & \ddots \\
//! 0 & \beta_0 & H_0 & \ddots \\
//! \vdots & \ddots & \ddots & \ddots
//! \end{pmatrix},
//! \qquad
//! \beta_0=\alpha_0^\dagger.
//! ```
//!
//! [`SurfGreen::from_Model`] obtains such a layer by enlarging the unit cell
//! along the requested surface normal to cover the model's normal hopping
//! range. [`SurfGreen::gen_ham_onek`] then Fourier transforms the remaining
//! in-plane translations:
//!
//! ```math
//! H_0(\mathbf{k}_\parallel)
//! =\sum_{\mathbf{R}_\parallel}
//! H_0(\mathbf{R}_\parallel)
//! e^{2\pi i\mathbf{k}_\parallel\cdot\mathbf{R}_\parallel},
//! \qquad
//! \alpha_0(\mathbf{k}_\parallel)
//! =\sum_{\mathbf{R}_\parallel}
//! H_{01}(\mathbf{R}_\parallel)
//! e^{2\pi i\mathbf{k}_\parallel\cdot\mathbf{R}_\parallel}.
//! ```
//!
//! Both blocks are returned in the atom-position gauge. For orbital positions
//! $\boldsymbol{\tau}_m$, the applied gauge transformation is
//!
//! ```math
//! H_{mn}\longmapsto
//! e^{-2\pi i\boldsymbol{\tau}_m\cdot\mathbf{k}_\parallel}
//! H_{mn}
//! e^{ 2\pi i\boldsymbol{\tau}_n\cdot\mathbf{k}_\parallel}.
//! ```
//!
//! # López-Sancho decimation
//!
//! Define the retarded energy $z=E+i\eta$ with $\eta>0$, and initialize
//!
//! ```math
//! \epsilon_0=\epsilon_0^L=\epsilon_0^R=H_0,
//! \qquad
//! \alpha_0=H_{01},
//! \qquad
//! \beta_0=H_{01}^\dagger.
//! ```
//!
//! One decimation step first forms
//!
//! ```math
//! g_n=(zI-\epsilon_n)^{-1},
//! ```
//!
//! and then updates the effective bulk layer, the two surface layers, and the
//! residual inter-layer couplings:
//!
//! ```math
//! \begin{aligned}
//! \epsilon_{n+1}
//! &=\epsilon_n
//!   +\alpha_n g_n\beta_n
//!   +\beta_n g_n\alpha_n,\\
//! \epsilon_{n+1}^{L}
//! &=\epsilon_n^{L}+\alpha_n g_n\beta_n,\\
//! \epsilon_{n+1}^{R}
//! &=\epsilon_n^{R}+\beta_n g_n\alpha_n,\\
//! \alpha_{n+1}&=\alpha_n g_n\alpha_n,\\
//! \beta_{n+1}&=\beta_n g_n\beta_n.
//! \end{aligned}
//! ```
//!
//! Each iteration doubles the number of original layers represented by one
//! effective layer. Once the residual couplings have decayed, the required
//! Green functions are
//!
//! ```math
//! G_L^s=(zI-\epsilon_\infty^L)^{-1},
//! \qquad
//! G_R^s=(zI-\epsilon_\infty^R)^{-1},
//! \qquad
//! G_B=(zI-\epsilon_\infty)^{-1}.
//! ```
//!
//! Equivalently, the left surface Green function obeys the nonlinear Dyson
//! equation
//!
//! ```math
//! G_L^s=\left[zI-H_0-H_{01}G_L^sH_{01}^\dagger\right]^{-1},
//! ```
//!
//! with the coupling order reversed for the right termination.
//!
//! # Returned spectral densities
//!
//! The public evaluation routines return the trace spectral density of a whole
//! principal layer, rather than the Green matrices themselves:
//!
//! ```math
//! \rho_X(E,\mathbf{k}_\parallel)
//! =-\frac{1}{\pi}\operatorname{Im}\operatorname{Tr}G_X,
//! \qquad X\in\{L,R,B\}.
//! ```
//!
//! A positive [`SurfGreen::eta`] gives retarded broadening. The scalar-energy
//! routine uses at most 10 decimation steps and stops when
//! $\sum_{ij}|(\alpha_n)_{ij}|<10^{-8}$; the vector-energy routine uses the
//! same limit with a threshold of $10^{-6}$.
//!
//! # References
//!
//! - M. P. López Sancho, J. M. López Sancho, and J. Rubio,
//!   “Quick iterative scheme for the calculation of transfer matrices:
//!   application to Mo (100),” *Journal of Physics F: Metal Physics* **14**,
//!   1205–1215 (1984),
//!   [doi:10.1088/0305-4608/14/5/016](https://doi.org/10.1088/0305-4608/14/5/016).
//! - M. P. López Sancho, J. M. López Sancho, and J. Rubio,
//!   “Highly convergent schemes for the calculation of bulk and surface Green
//!   functions,” *Journal of Physics F: Metal Physics* **15**, 851–858 (1985),
//!   [doi:10.1088/0305-4608/15/4/009](https://doi.org/10.1088/0305-4608/15/4/009).
use crate::Model;
use crate::RMatrixData;
use crate::error::{Result, TbError};
use crate::kpath::*;
use crate::kpoints::gen_kmesh;
pub use crate::model_utils::{remove_col, remove_row};
use gnuplot::Major;
use gnuplot::{Auto, AutoOption::Fix, AxesCommon, Custom, Figure, Font};
use ndarray::concatenate;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_linalg::conjugate;
use ndarray_linalg::*;
use num_complex::Complex;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Write};
use std::process::{Command, Stdio};

/// Basic building block for surface Green's function calculations.
#[derive(Clone, Debug)]
pub struct SurfGreen {
    /// - The real space dimension of the model.
    pub dim_r: usize,
    /// - The number of orbitals in the model.
    pub norb: usize,
    /// - The number of states in the model. If spin is enabled, nsta=norb$\times$2
    pub nsta: usize,
    /// - The number of atoms in the model. The atom and atom_list at the back are used to store the positions of the atoms, and the number of orbitals corresponding to each atom.
    pub natom: usize,
    /// - Whether this surface Green's function was built from a spinful model.
    ///   Derived from `Model<SPIN>` via [`from_Model`](SurfGreen::from_Model).
    pub spin: bool,
    /// - The lattice vector of the model, a dim_r$\times$dim_r matrix, the axis0 direction stores a 1$\times$dim_r lattice vector.
    pub lat: Array2<f64>,
    /// - The position of the orbitals in the model. We use fractional coordinates uniformly.
    pub orb: Array2<f64>,
    /// - The position of the atoms in the model, also in fractional coordinates.
    pub atom: Array2<f64>,
    /// - The number of orbitals in the atoms, in the same order as the atom positions.
    pub atom_list: Vec<usize>,
    /// - The bulk Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub eta: f64,
    pub ham_bulk: Array3<Complex<f64>>,
    /// - The distance between the unit cell hoppings, i.e. R in $\bra{m0}\hat H\ket{nR}$.
    pub ham_bulkR: Array2<isize>,
    /// - The bulk Hamiltonian of the model, $\bra{m0}\hat H\ket{nR}$, a three-dimensional complex tensor of size n_R$\times$nsta$\times$ nsta, where the first nsta*nsta matrix corresponds to hopping within the unit cell, i.e. <m0|H|n0>, and the subsequent matrices correspond to hopping within hamR.
    pub ham_hop: Array3<Complex<f64>>,
    pub ham_hopR: Array2<isize>,
}

impl Kpath for SurfGreen {
    fn k_path(
        &self,
        path: &Array2<f64>,
        nk: usize,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        //! Generate a k-path from high-symmetry points for band structure plotting.
        interpolate_k_path(&self.lat, self.dim_r, path, nk)
    }
}

impl SurfGreen {
    /// Construct a `SurfGreen` from a [`Model`].
    ///
    /// This method is generic over the const parameter `SPIN`, which
    /// determines whether spin is enabled. The `SurfGreen::spin` field is
    /// set from this const generic (`spin = SPIN`), and the internal
    /// Hamiltonian construction (`orb_phase` doubling, etc.) uses it at
    /// runtime.
    ///
    /// `dir` specifies the surface normal direction.
    ///
    /// `eta` is the small imaginary part for the Green's function.
    ///
    /// With `Np = None`, the principal-layer thickness is the largest absolute
    /// hopping range along `dir`. `Some(np)` caps that thickness at `np`; use a
    /// cap smaller than the true hopping range only when that approximation is
    /// intentional.
    ///
    /// For directions not aligned with a lattice vector, use [`Model::make_supercell`] first.
    pub fn from_Model<const SPIN: bool, const DIM: usize, R: RMatrixData>(
        model: &Model<SPIN, DIM, R>,
        dir: usize,
        eta: f64,
        Np: Option<usize>,
    ) -> Result<SurfGreen> {
        if dir >= model.dim_r() {
            return Err(TbError::InvalidDirection {
                index: dir,
                dim: model.dim_r(),
            });
        }
        model.validate()?;
        if !eta.is_finite() || eta <= 0.0 {
            return Err(surface_error("eta", "must be finite and positive"));
        }
        if Np == Some(0) {
            return Err(surface_error("Np", "must be positive"));
        }
        let mut R_max: usize = 1;
        for R0 in model.hamR.rows() {
            if R_max < R0[[dir]].unsigned_abs() {
                R_max = R0[[dir]].unsigned_abs();
            }
        }
        let R_max = match Np {
            Some(np) => {
                if R_max > np {
                    np
                } else {
                    R_max
                }
            }
            None => R_max,
        };

        let mut U = Array2::<f64>::eye(model.dim_r());
        U[[dir, dir]] = R_max as f64;
        let model = model.make_supercell(&U)?;
        let mut ham0 = Array3::<Complex<f64>>::zeros((0, model.nsta(), model.nsta()));
        let mut hamR0 = Array2::<isize>::zeros((0, model.dim_r()));
        let mut hamR = Array3::<Complex<f64>>::zeros((0, model.nsta(), model.nsta()));
        let mut hamRR = Array2::<isize>::zeros((0, model.dim_r()));
        let use_hamR = model.hamR.rows();
        let use_ham = model.ham.outer_iter();
        // No clone needed: push/push_row copy from the view directly
        for (ham, R) in use_ham.zip(use_hamR) {
            if R[[dir]] == 0 {
                ham0.push(Axis(0), ham)?;
                hamR0.push_row(R)?;
            } else if R[[dir]] > 0 {
                hamR.push(Axis(0), ham)?;
                hamRR.push_row(R)?;
            }
        }
        let new_lat = remove_row(model.lat.clone(), dir);
        let new_lat = remove_col(new_lat.clone(), dir);
        let new_orb = remove_col(model.orb.clone(), dir);
        let new_atom = remove_col(model.atom_position(), dir);
        let new_hamR0 = remove_col(hamR0, dir);
        let new_hamRR = remove_col(hamRR, dir);
        let green: SurfGreen = SurfGreen {
            dim_r: model.dim_r() - 1,
            norb: model.norb(),
            nsta: model.nsta(),
            natom: model.natom(),
            spin: SPIN,
            lat: new_lat,
            orb: new_orb,
            atom: new_atom,
            atom_list: model.atom_list(),
            ham_bulk: ham0,
            ham_bulkR: new_hamR0,
            ham_hop: hamR,
            ham_hopR: new_hamRR,
            eta,
        };
        Ok(green)
    }

    /// Construct the principal-layer blocks at one surface momentum.
    ///
    /// The returned pair is `(H_0(k_parallel), H_01(k_parallel))`, in the
    /// atom-position gauge described in the [module-level documentation](self).
    /// Both matrices have shape `nsta × nsta`.
    ///
    /// Returns an error for malformed surface storage or nonfinite/wrong-length momentum.
    #[inline(always)]
    pub fn gen_ham_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>)> {
        self.validate()?;
        if kvec.len() != self.dim_r {
            return Err(TbError::KVectorLengthMismatch {
                expected: self.dim_r,
                actual: kvec.len(),
            });
        }
        if kvec.iter().any(|x| !x.is_finite()) {
            return Err(surface_error("kvec", "components must be finite"));
        }
        // Orbital gauge phases: exp(i 2π τ·k)
        let orb_phase: Array1<Complex<f64>> = self
            .orb
            .dot(kvec)
            .mapv(|p| Complex::new(0.0, 2.0 * PI * p).exp());
        let orb_phase = if self.spin {
            concatenate![Axis(0), orb_phase, orb_phase]
        } else {
            orb_phase
        };
        // R-space phases for ham_bulk and ham_hop
        let U0 = (self.ham_bulkR.map(|x| *x as f64))
            .dot(kvec)
            .map(|x| Complex::<f64>::new(0.0, *x * 2.0 * PI).exp());
        let UR = (self.ham_hopR.map(|x| *x as f64))
            .dot(kvec)
            .map(|x| Complex::<f64>::new(0.0, *x * 2.0 * PI).exp());
        // Build H(k) with scaled_add: zero-allocation per R
        let mut ham0k = Array2::<Complex<f64>>::zeros((self.nsta, self.nsta));
        Zip::from(self.ham_bulk.outer_iter())
            .and(&U0)
            .for_each(|ham, &u| ham0k.scaled_add(u, &ham));
        let mut hamRk = Array2::<Complex<f64>>::zeros((self.nsta, self.nsta));
        Zip::from(self.ham_hop.outer_iter())
            .and(&UR)
            .for_each(|ham, &u| hamRk.scaled_add(u, &ham));
        // Gauge transform: H'[m,n] = conj(φ[m]) * H[m,n] * φ[n]
        // In-place O(nsta²) replaces two O(nsta³) matrix multiplies with diagonal U.
        for ham in [&mut ham0k, &mut hamRk] {
            for m in 0..self.nsta {
                let mut row = ham.slice_mut(s![m, ..]);
                let conj_pm = orb_phase[m].conj();
                Zip::from(&mut row)
                    .and(&orb_phase)
                    .for_each(|h, &pn| *h *= conj_pm * pn);
            }
        }
        if ham0k
            .iter()
            .chain(hamRk.iter())
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(surface_error(
                "Hamiltonian",
                "Fourier transform must be finite",
            ));
        }
        Ok((ham0k, hamRk))
    }
    fn validate(&self) -> Result<()> {
        if !self.eta.is_finite() || self.eta <= 0.0 {
            return Err(surface_error("eta", "must be finite and positive"));
        }
        if self.norb == 0
            || self.norb.checked_mul(if self.spin { 2 } else { 1 }) != Some(self.nsta)
            || self.orb.dim() != (self.norb, self.dim_r)
            || self.lat.dim() != (self.dim_r, self.dim_r)
            || self.ham_bulk.dim() != (self.ham_bulkR.nrows(), self.nsta, self.nsta)
            || self.ham_hop.dim() != (self.ham_hopR.nrows(), self.nsta, self.nsta)
            || self.ham_bulkR.ncols() != self.dim_r
            || self.ham_hopR.ncols() != self.dim_r
        {
            return Err(surface_error(
                "storage",
                "inconsistent surface model dimensions",
            ));
        }
        if self
            .orb
            .iter()
            .chain(self.lat.iter())
            .any(|x| !x.is_finite())
            || self
                .ham_bulk
                .iter()
                .chain(self.ham_hop.iter())
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(surface_error(
                "storage",
                "surface model data must be finite",
            ));
        }
        Ok(())
    }

    /// Evaluate spectral densities at one energy, returning `(right, left, bulk)`.
    ///
    /// Each density is $-\operatorname{Im}\operatorname{Tr}(G)/\pi$ for one
    /// principal layer. Uses at most 10 decimation steps and a residual-coupling
    /// threshold of `1e-8`. Invalid input and matrix inversion errors propagate.
    pub fn surf_green_one<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy: f64,
    ) -> Result<(f64, f64, f64)> {
        if !energy.is_finite() {
            return Err(surface_error("energy", "must be finite"));
        }
        let (ham, hop) = self.gen_ham_onek(kvec)?;
        decimate(&ham, &hop, energy, self.eta, 1e-8)
    }

    /// Evaluate an energy grid, returning `(right, left, bulk)` in energy order.
    ///
    /// Uses at most 10 decimation steps per energy and a residual-coupling
    /// threshold of `1e-6`. The energy grid must be nonempty and finite.
    pub fn surf_green_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        if energy.is_empty() || energy.iter().any(|e| !e.is_finite()) {
            return Err(surface_error("energy", "grid must be nonempty and finite"));
        }
        let (ham, hop) = self.gen_ham_onek(kvec)?;
        let mut right = Array1::zeros(energy.len());
        let mut left = Array1::zeros(energy.len());
        let mut bulk = Array1::zeros(energy.len());
        for (i, &e) in energy.iter().enumerate() {
            (right[i], left[i], bulk[i]) = decimate(&ham, &hop, e, self.eta, 1e-6)?;
        }
        Ok((right, left, bulk))
    }

    /// Evaluate a momentum path, returning `(left, right, bulk)`.
    ///
    /// Arrays have shape `(kvec.nrows(), E_n)`. This path-level order reverses
    /// the first two elements of [`Self::surf_green_one`] and
    /// [`Self::surf_green_onek`]. Energies span the inclusive range `[E_min, E_max]`.
    pub fn surf_green_path(
        &self,
        kvec: &Array2<f64>,
        E_min: f64,
        E_max: f64,
        E_n: usize,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        validate_energy_range(E_min, E_max, E_n)?;
        self.validate()?;
        if kvec.nrows() == 0 || kvec.ncols() != self.dim_r || kvec.iter().any(|x| !x.is_finite()) {
            return Err(surface_error(
                "kvec",
                "path must be nonempty, finite and have dim_r columns",
            ));
        }
        kvec.nrows()
            .checked_mul(E_n)
            .and_then(|n| n.checked_mul(size_of::<f64>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| surface_error("grid", "array size overflows"))?;
        let energy = Array1::linspace(E_min, E_max, E_n);
        // Collect in indexed momentum order before propagating errors, so the
        // first failing momentum is deterministic across Rayon schedules.
        let results: Vec<_> = kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.surf_green_onek(&k, &energy))
            .collect();
        let mut right = Array2::zeros((kvec.nrows(), E_n));
        let mut left = right.clone();
        let mut bulk = right.clone();
        for (i, result) in results.into_iter().enumerate() {
            let (r, l, b) = result?;
            right.row_mut(i).assign(&r);
            left.row_mut(i).assign(&l);
            bulk.row_mut(i).assign(&b);
        }
        Ok((left, right, bulk))
    }

    /// Write `arc.dat` and left/right/bulk PDFs for a two-dimensional surface.
    ///
    /// The data file uses Cartesian reciprocal coordinates (including `2π`);
    /// image axes retain fractional coordinates. Spectral densities are unscaled.
    /// Invalid input, file errors and gnuplot failures are returned to the caller.
    pub fn show_arc_state(&self, name: &str, kmesh: &Array1<usize>, energy: f64) -> Result<()> {
        self.validate()?;
        if self.dim_r != 2 || kmesh.len() != 2 {
            return Err(TbError::InvalidKmeshDimensions(kmesh.to_owned()));
        }
        if !energy.is_finite() {
            return Err(surface_error("energy", "must be finite"));
        }
        let kvec = gen_kmesh::<f64>(kmesh)?;
        let reciprocal = 2.0 * PI * self.lat.inv()?.reversed_axes();
        let kvec_real = kvec.dot(&reciprocal);
        if kvec_real.iter().any(|x| !x.is_finite()) {
            return Err(surface_error(
                "coordinates",
                "reciprocal coordinates must be finite",
            ));
        }
        let results: Vec<_> = kvec
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|k| self.surf_green_one(&k, energy))
            .collect();
        let mut left = Array2::zeros((kmesh[0], kmesh[1]));
        let mut right = left.clone();
        let mut bulk = left.clone();
        for (i, result) in results.into_iter().enumerate() {
            let (r, l, b) = result?;
            let index = [i / kmesh[1], i % kmesh[1]];
            left[index] = l;
            right[index] = r;
            bulk[index] = b;
        }
        create_dir_all(name)?;
        let mut writer = BufWriter::new(File::create(format!("{name}/arc.dat"))?);
        writeln!(writer, "# nk1, nk2, N_L, N_R, N_B")?;
        for (i, k) in kvec_real.rows().into_iter().enumerate() {
            let index = [i / kmesh[1], i % kmesh[1]];
            writeln!(
                writer,
                "{:.6}    {:.6}    {:.6}    {:.6}    {:.6}",
                k[0], k[1], left[index], right[index], bulk[index]
            )?;
        }
        writer.flush()?;
        for (suffix, data) in [("l", &left), ("r", &right), ("b", &bulk)] {
            let mut figure = surface_figure(data, (0.0, 0.0, 1.0, 1.0), &[]);
            figure.set_terminal("pdfcairo", &format!("{name}/surf_state_{suffix}.pdf"));
            render_surface(&figure, &mut Command::new("gnuplot"))?;
        }
        Ok(())
    }

    /// Write path spectra (`dos.surf_l`, `dos.surf_r`, `dos.surf_bulk`) and PDFs.
    ///
    /// Each positive density is log-normalized independently to `[-10, 10]`;
    /// constant positive data maps to zero. Data files contain this same color
    /// scale. Path distances omit `2π`, as in [`Kpath::k_path`].
    /// Invalid input, nonpositive densities, file and gnuplot errors propagate.
    pub fn show_surf_state(
        &self,
        name: &str,
        kpath: &Array2<f64>,
        label: &[&str],
        nk: usize,
        E_min: f64,
        E_max: f64,
        E_n: usize,
    ) -> Result<()> {
        validate_energy_range(E_min, E_max, E_n)?;
        if E_min == E_max || E_n < 2 {
            return Err(surface_error(
                "energy",
                "a heatmap needs at least two distinct energies",
            ));
        }
        if label.len() != kpath.nrows() {
            return Err(TbError::PathLengthMismatch {
                expected: kpath.nrows(),
                actual: label.len(),
            });
        }
        let (kvec, kdist, knode) = self.k_path(kpath, nk)?;
        let energy = Array1::linspace(E_min, E_max, E_n);
        let (left, right, bulk) = self.surf_green_path(&kvec, E_min, E_max, E_n)?;
        // Validate all three datasets before creating any output.
        let surfaces = [
            ("l", "l", log_normalize(&left)?),
            ("r", "r", log_normalize(&right)?),
            ("bulk", "b", log_normalize(&bulk)?),
        ];
        let ticks: Vec<_> = knode.iter().copied().zip(label.iter().copied()).collect();
        create_dir_all(name)?;
        for (data_suffix, pdf_suffix, data) in surfaces {
            let mut writer =
                BufWriter::new(File::create(format!("{name}/dos.surf_{data_suffix}"))?);
            for (i, &distance) in kdist.iter().enumerate() {
                for (j, &e) in energy.iter().enumerate() {
                    writeln!(writer, "{distance:.6}    {e:.6}    {:.6}", data[[i, j]])?;
                }
                writeln!(writer)?;
            }
            writer.flush()?;
            let mut figure = surface_figure(&data, (kdist[0], E_min, kdist[nk - 1], E_max), &ticks);
            figure.set_terminal("pdfcairo", &format!("{name}/surf_state_{pdf_suffix}.pdf"));
            render_surface(&figure, &mut Command::new("gnuplot"))?;
        }
        Ok(())
    }
}

fn surface_error(parameter: &'static str, message: &str) -> TbError {
    TbError::InvalidSurfaceParameter {
        parameter,
        message: message.into(),
    }
}

fn validate_energy_range(min: f64, max: f64, count: usize) -> Result<()> {
    if !min.is_finite() || !max.is_finite() || min > max || !(max - min).is_finite() {
        return Err(TbError::InvalidEnergyRange { min, max });
    }
    if count == 0 || count > isize::MAX as usize / size_of::<f64>() {
        return Err(surface_error(
            "energy",
            "grid size must be positive and fit in memory indexing",
        ));
    }
    Ok(())
}

// The scalar and vector APIs intentionally retain their distinct stopping thresholds.
fn decimate(
    ham: &Array2<Complex<f64>>,
    hop: &Array2<Complex<f64>>,
    energy: f64,
    eta: f64,
    tolerance: f64,
) -> Result<(f64, f64, f64)> {
    let epsilon = Complex::new(energy, eta) * Array2::<Complex<f64>>::eye(ham.nrows());
    let mut bulk = ham.clone();
    let mut left = ham.clone();
    let mut right = ham.clone();
    let mut alpha = hop.clone();
    let mut beta = conjugate(hop);
    for _ in 0..10 {
        let g = (&epsilon - &bulk).inv()?;
        let ag = alpha.dot(&g);
        let bg = beta.dot(&g);
        let agb = ag.dot(&beta);
        let bga = bg.dot(&alpha);
        bulk += &agb;
        bulk += &bga;
        left += &agb;
        right += &bga;
        alpha = ag.dot(&alpha);
        beta = bg.dot(&beta);
        if alpha.iter().map(|z| z.norm()).sum::<f64>() < tolerance {
            break;
        }
    }
    let density = |block: Array2<Complex<f64>>| -> Result<f64> {
        let rho = -(&epsilon - block).inv()?.into_diag().sum().im / PI;
        if !rho.is_finite() {
            return Err(surface_error(
                "density",
                "decimation produced a nonfinite spectral density",
            ));
        }
        Ok(rho)
    };
    Ok((density(right)?, density(left)?, density(bulk)?))
}

fn log_normalize(data: &Array2<f64>) -> Result<Array2<f64>> {
    if data.is_empty() || data.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(surface_error(
            "density",
            "log normalization requires finite positive densities",
        ));
    }
    let mut logarithms = data.mapv(f64::ln);
    let min = logarithms.iter().copied().fold(f64::INFINITY, f64::min);
    let max = logarithms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        logarithms.fill(0.0);
    } else {
        logarithms.mapv_inplace(|x| (x - min) / (max - min) * 20.0 - 10.0);
    }
    Ok(logarithms)
}

fn surface_figure(
    data: &Array2<f64>,
    bounds: (f64, f64, f64, f64),
    ticks: &[(f64, &str)],
) -> Figure {
    let mut figure = Figure::new();
    let axes = figure.axes2d();
    axes.set_palette(Custom(&[
        (-1.0, 0.0, 0.0, 0.0),
        (-0.9, 65.0 / 255.0, 9.0 / 255.0, 103.0 / 255.0),
        (0.0, 147.0 / 255.0, 37.0 / 255.0, 103.0 / 255.0),
        (0.2, 220.0 / 255.0, 80.0 / 255.0, 57.0 / 255.0),
        (1.0, 252.0 / 255.0, 254.0 / 255.0, 164.0 / 255.0),
    ]));
    // Model grids are (x, y), whereas gnuplot consumes rows of fixed y.
    axes.image(
        data.t().iter(),
        data.ncols(),
        data.nrows(),
        Some(bounds),
        &[],
    );
    axes.set_x_range(Fix(bounds.0), Fix(bounds.2));
    axes.set_y_range(Fix(bounds.1), Fix(bounds.3));
    axes.set_aspect_ratio(Fix(1.0));
    if ticks.is_empty() {
        axes.set_x_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
    } else {
        axes.set_x_ticks_custom(
            ticks.iter().map(|&(x, label)| Major(x, Fix(label))),
            &[],
            &[Font("Times New Roman", 24.0)],
        );
    }
    axes.set_y_ticks(Some((Auto, 0)), &[], &[Font("Times New Roman", 24.0)]);
    axes.set_cb_ticks_custom(
        [
            Major(-10.0, Fix("low")),
            Major(0.0, Fix("0")),
            Major(10.0, Fix("high")),
        ],
        &[],
        &[Font("Times New Roman", 24.0)],
    );
    figure
}

fn render_surface(figure: &Figure, command: &mut Command) -> Result<()> {
    // Figure::show internally panics on spawn errors and discards write/status
    // failures. Echo to an infallible memory buffer and own the process instead.
    let mut script = Vec::new();
    figure.echo(&mut script);
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("gnuplot stdin is unavailable"))?;
    // Drain diagnostics while feeding the image: either pipe can exceed the
    // OS buffer, so writing all input before reading stderr can deadlock.
    let (written, output) = std::thread::scope(|scope| {
        let writer = scope.spawn(move || stdin.write_all(&script));
        let output = child.wait_with_output();
        let written = writer
            .join()
            .unwrap_or_else(|_| Err(std::io::Error::other("gnuplot input writer panicked")));
        (written, output)
    });
    let output = output?;
    if !output.status.success() {
        return Err(TbError::Other(format!(
            "gnuplot exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    written?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plotting_drains_large_diagnostics_while_writing_large_images() {
        let figure = surface_figure(&Array2::ones((200, 200)), (0.0, 0.0, 1.0, 1.0), &[]);
        let error = render_surface(
            &figure,
            Command::new("sh").args(["-c", "head -c 131072 /dev/zero >&2; cat >/dev/null; exit 1"]),
        )
        .unwrap_err();
        assert!(matches!(error, TbError::Other(message) if message.contains("gnuplot exited")));
    }

    fn asymmetric_surface() -> SurfGreen {
        let mut model =
            Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0], [0.0, 0.0]], None)
                .unwrap();
        model.set_hop(0.4, 0, 0, &array![0, 0], None);
        model.set_hop(-0.7, 1, 1, &array![0, 0], None);
        model.set_hop(Complex::new(0.6, 0.2), 0, 1, &array![1, 0], None);
        SurfGreen::from_Model(&model, 0, 0.05, None).unwrap()
    }

    #[test]
    fn point_grid_and_path_preserve_surface_order() {
        let surface = asymmetric_surface();
        let energies = array![-0.5, 0.0, 0.5];
        let k = array![0.27];
        let (right, left, bulk) = surface.surf_green_onek(&k, &energies).unwrap();
        let (path_left, path_right, path_bulk) = surface
            .surf_green_path(&array![[0.27], [0.61]], -0.5, 0.5, 3)
            .unwrap();
        assert!((right[1] - left[1]).abs() > 0.01);
        for (i, &energy) in energies.iter().enumerate() {
            // This model consists of independent inter-layer dimers. The left
            // edge leaves orbital 1 dangling, and the right leaves orbital 0.
            // This is an analytic oracle independent of the decimation loop.
            let z = Complex::new(energy, surface.eta);
            let determinant = (z - 0.4) * (z + 0.7) - 0.4;
            let expected_left = -((z + 0.7) / determinant + 1.0 / (z + 0.7)).im / PI;
            let expected_right = -(1.0 / (z - 0.4) + (z - 0.4) / determinant).im / PI;
            let expected_bulk = -((2.0 * z + 0.3) / determinant).im / PI;
            let point = surface.surf_green_one(&k, energy).unwrap();
            for (actual, expected) in [
                (point.0, expected_right),
                (point.1, expected_left),
                (point.2, expected_bulk),
                (right[i], expected_right),
                (left[i], expected_left),
                (bulk[i], expected_bulk),
                (path_right[[0, i]], expected_right),
                (path_left[[0, i]], expected_left),
                (path_bulk[[0, i]], expected_bulk),
            ] {
                assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
            }
        }
        assert_eq!(path_left.row(0), path_left.row(1));
        assert_eq!(path_right.row(0), path_right.row(1));
    }

    #[test]
    fn surface_inputs_and_inversion_failures_are_errors() {
        let mut surface = asymmetric_surface();
        assert!(surface.gen_ham_onek(&array![0.0, 0.0]).is_err());
        assert!(surface.gen_ham_onek(&array![f64::NAN]).is_err());
        assert!(surface.surf_green_one(&array![0.0], f64::INFINITY).is_err());
        assert!(surface.surf_green_onek(&array![0.0], &array![]).is_err());
        assert!(
            surface
                .surf_green_onek(&array![0.0], &array![f64::NAN])
                .is_err()
        );
        for (min, max, count) in [
            (1.0, 0.0, 2),
            (0.0, 1.0, 0),
            (0.0, f64::INFINITY, 2),
            (0.0, 1.0, usize::MAX),
        ] {
            assert!(
                surface
                    .surf_green_path(&array![[0.0]], min, max, count)
                    .is_err()
            );
        }
        assert!(
            surface
                .surf_green_path(&Array2::zeros((0, 1)), 0.0, 1.0, 2)
                .is_err()
        );
        assert!(
            surface
                .surf_green_path(&array![[0.0, 0.0]], 0.0, 1.0, 2)
                .is_err()
        );
        surface.eta = 0.0;
        assert!(surface.surf_green_one(&array![0.0], 0.0).is_err());
        surface.eta = 0.05;
        surface.ham_bulk.fill(Complex::new(0.0, 0.0));
        surface.ham_bulk[[0, 0, 0]] = Complex::new(0.0, surface.eta);
        surface.ham_hop.fill(Complex::new(0.0, 0.0));
        assert!(matches!(
            surface.surf_green_one(&array![0.0], 0.0),
            Err(TbError::Linalg(_))
        ));
        assert!(matches!(
            surface.surf_green_onek(&array![0.0], &array![0.0]),
            Err(TbError::Linalg(_))
        ));
        assert!(matches!(
            surface.surf_green_path(&array![[0.0]], 0.0, 0.0, 1),
            Err(TbError::Linalg(_))
        ));
        surface.orb = Array2::zeros((1, 1));
        assert!(surface.gen_ham_onek(&array![0.0]).is_err());
    }

    #[test]
    fn isolated_layers_and_constructor_validation() {
        let model = Model::<false, 2>::tb_model(Array2::eye(2), array![[0.0, 0.0]], None).unwrap();
        let surface = SurfGreen::from_Model(&model, 0, 0.2, None).unwrap();
        let (right, left, bulk) = surface.surf_green_one(&array![0.0], 0.0).unwrap();
        assert!((bulk - 1.0 / (PI * 0.2)).abs() < 1e-12);
        assert_eq!(right, left);
        assert_eq!(left, bulk);
        for eta in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(SurfGreen::from_Model(&model, 0, eta, None).is_err());
        }
        assert!(SurfGreen::from_Model(&model, 0, 0.2, Some(0)).is_err());
    }

    #[test]
    fn log_normalization_rejects_invalid_data_and_centers_constants() {
        assert_eq!(
            log_normalize(&array![[2.0, 2.0]]).unwrap(),
            array![[0.0, 0.0]]
        );
        assert_eq!(
            log_normalize(&array![[1.0, 10.0, 100.0]]).unwrap(),
            array![[-10.0, 0.0, 10.0]]
        );
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(log_normalize(&array![[value]]).is_err());
        }
        assert!(log_normalize(&Array2::zeros((0, 0))).is_err());
        assert!(
            log_normalize(&array![[f64::from_bits(1), f64::MAX]])
                .unwrap()
                .iter()
                .all(|x| x.is_finite())
        );
    }

    #[test]
    fn nonsquare_heatmap_preserves_x_y_order() {
        let data = array![[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]];
        let figure = surface_figure(&data, (0.0, 0.0, 1.0, 1.0), &[]);
        let mut script = Vec::new();
        figure.echo(&mut script);
        assert!(String::from_utf8_lossy(&script).contains("array=(2,3)"));
        let expected: Vec<u8> = [11.0_f64, 21.0, 12.0, 22.0, 13.0, 23.0]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
        assert!(
            script
                .windows(expected.len())
                .any(|bytes| bytes == expected)
        );
    }

    #[test]
    fn plotting_validates_before_files_and_propagates_io_errors() {
        let surface = asymmetric_surface();
        let path =
            std::env::temp_dir().join(format!("rustb-surface-errors-{}", std::process::id()));
        let name = path.to_str().unwrap();
        let kpath = array![[0.0], [1.0]];
        assert!(
            surface
                .show_surf_state(name, &kpath, &["G"], 2, -1.0, 1.0, 2)
                .is_err()
        );
        assert!(
            surface
                .show_surf_state(name, &kpath, &["G", "G"], 2, 0.0, 0.0, 2)
                .is_err()
        );
        let mut invalid_density = surface.clone();
        invalid_density.ham_hop.fill(Complex::new(0.0, 0.0));
        invalid_density.ham_bulk.fill(Complex::new(0.0, 0.0));
        for band in 0..invalid_density.nsta {
            invalid_density.ham_bulk[[0, band, band]] =
                Complex::new(0.0, 2.0 * invalid_density.eta);
        }
        assert!(
            invalid_density
                .show_surf_state(name, &kpath, &["G", "G"], 2, -1.0, 1.0, 2)
                .is_err()
        );
        assert!(!path.exists());
        File::create(&path).unwrap();
        assert!(matches!(
            surface.show_surf_state(name, &kpath, &["G", "G"], 2, -1.0, 1.0, 2),
            Err(TbError::Io(_))
        ));
        let surface2d = surface_for_k_path();
        assert!(matches!(
            surface2d.show_arc_state(name, &array![2, 3], 0.0),
            Err(TbError::Io(_))
        ));
        assert!(surface2d.show_arc_state(name, &array![2], 0.0).is_err());
        assert!(surface2d.show_arc_state(name, &array![2, 0], 0.0).is_err());
        std::fs::remove_file(&path).unwrap();
        // An existing directory at the data filename exercises File::create,
        // independently of the create_dir_all error above.
        create_dir_all(path.join("dos.surf_l")).unwrap();
        assert!(matches!(
            surface.show_surf_state(name, &kpath, &["G", "G"], 2, -1.0, 1.0, 2),
            Err(TbError::Io(_))
        ));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn plotting_reports_spawn_and_exit_errors() {
        let figure = surface_figure(&array![[1.0, 2.0], [3.0, 4.0]], (0.0, 0.0, 1.0, 1.0), &[]);
        assert!(matches!(
            render_surface(&figure, &mut Command::new("/missing/rustb-gnuplot")),
            Err(TbError::Io(_))
        ));
        let error = render_surface(
            &figure,
            Command::new("sh")
                .arg("-c")
                .arg("cat >/dev/null; echo deliberate-plot-error >&2; exit 7"),
        )
        .unwrap_err();
        assert!(error.to_string().contains("deliberate-plot-error"));
    }

    fn surface_for_k_path() -> SurfGreen {
        let mut model = Model::<false, 3>::tb_model(
            array![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
            array![[0.0, 0.0, 0.0]],
            None,
        )
        .unwrap();
        model.set_hop(1.0, 0, 0, &array![0, 0, 1], None);
        SurfGreen::from_Model(&model, 2, 1e-3, None).unwrap()
    }

    #[test]
    fn k_path_retains_short_first_and_last_segments() {
        let surface = surface_for_k_path();
        let path = array![
            [0.25, 0.25],
            [0.250000001, 0.250000001],
            [1.0, 1.0],
            [1.000000001, 1.000000001]
        ];
        for nk in [4, 11] {
            let (points, distances, nodes) = surface.k_path(&path, nk).unwrap();
            assert_eq!(points.dim(), (nk, 2));
            assert!(points.iter().chain(distances.iter()).all(|x| x.is_finite()));
            assert!(distances.windows(2).into_iter().all(|w| w[1] > w[0]));
            assert_eq!(points.row(0), path.row(0));
            assert_eq!(points.row(nk - 1), path.row(3));
            for (i, node) in path.rows().into_iter().enumerate() {
                let index = points.rows().into_iter().position(|p| p == node).unwrap();
                assert_eq!(distances[index], nodes[i]);
                // Along (t, t), this surface's reciprocal metric gives |dt|.
                assert!((nodes[i] - (node[0] - 0.25)).abs() < 1e-14);
            }
            if nk == 4 {
                assert_eq!(points, path);
            }
        }
    }

    #[test]
    fn k_path_uses_reduced_lattice_metric_and_rounded_node_allocation() {
        let surface = surface_for_k_path();
        assert_eq!(surface.lat, array![[1.0, 0.0], [1.0, 1.0]]);
        // (lat * lat^T)^-1 = [[2, -1], [-1, 1]], without a 2*pi factor.
        // The two segment lengths are 1 and 2. Five intervals put the middle
        // node at round(5/3) = 2, then split the second segment in thirds.
        let path = array![[0.0, 0.0], [1.0, 1.0], [1.0, 3.0]];
        let (points, distances, nodes) = surface.k_path(&path, 6).unwrap();
        let expected_points = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 5.0 / 3.0],
            [1.0, 7.0 / 3.0],
            [1.0, 3.0]
        ];
        let expected_distances = array![0.0, 0.5, 1.0, 5.0 / 3.0, 7.0 / 3.0, 3.0];
        assert_eq!(points.dim(), expected_points.dim());
        assert_eq!(distances.len(), expected_distances.len());
        assert_eq!(nodes.len(), 3);
        for (actual, expected) in points.iter().zip(expected_points.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        for (actual, expected) in distances.iter().zip(expected_distances.iter()) {
            assert!((actual - expected).abs() < 1e-14);
        }
        for (actual, expected) in nodes.iter().zip([0.0, 1.0, 3.0]) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn k_path_rejects_empty_single_repeated_and_nonfinite_nodes() {
        let surface = surface_for_k_path();
        for path in [
            Array2::zeros((0, 2)),
            array![[0.0, 0.0]],
            array![[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            array![[0.0, 0.0], [f64::NAN, 1.0]],
            array![[0.0, 0.0], [1.0, f64::INFINITY]],
            array![[0.0, 0.0], [f64::NEG_INFINITY, 1.0]],
        ] {
            assert!(surface.k_path(&path, 5).is_err(), "path: {path:?}");
        }
    }

    #[test]
    fn k_path_rejects_undersampling() {
        let surface = surface_for_k_path();
        let path = array![[0.0, 0.0], [1.0, 1.0], [1.0, 3.0]];
        for nk in [0, 1, 2] {
            assert!(surface.k_path(&path, nk).is_err(), "nk: {nk}");
        }
    }

    #[test]
    fn k_path_rejects_singular_mutated_lattice() {
        let mut surface = surface_for_k_path();
        surface.lat = array![[1.0, 1.0], [2.0, 2.0]];
        assert!(matches!(
            surface.k_path(&array![[0.0, 0.0], [1.0, 1.0]], 3),
            Err(TbError::Linalg(_))
        ));
    }

    #[test]
    fn k_path_rejects_nonfinite_and_malformed_mutated_lattice() {
        let mut surface = surface_for_k_path();
        let path = array![[0.0, 0.0], [1.0, 1.0]];
        for lat in [
            array![[f64::NAN, 0.0], [0.0, 1.0]],
            array![[1.0, 0.0], [0.0, f64::INFINITY]],
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            Array2::eye(3),
        ] {
            surface.lat = lat;
            assert!(surface.k_path(&path, 3).is_err(), "lat: {:?}", surface.lat);
        }
    }
}
