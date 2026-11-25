use super::{Model, NoRMatrix, RMatrixData};
use crate::error::{Result, TbError};
use crate::floquet::{FloquetDrive, FloquetEffectiveOptions, drive_has_ac_components};
use crate::solve_ham::Solve;
use ndarray::{Array1, Array2, Array3, ArrayBase, Data, Ix1, Ix2};
use num_complex::Complex;

/// Two uncoupled spin channels of one collinear tight-binding system.
///
/// The supplied models have exactly equal lattice and atomic metadata. Each
/// model retains its storage; orbital centres, hopping supports, and position
/// matrices remain independent. The inherent solvers mirror [`Solve`] with
/// results in `(up, down)` order. No spin-flip terms are represented; use a
/// spinor [`Model`] if the channels are coupled.
///
/// ```
/// use Rustb::{CollinearModel, Model};
/// use ndarray::array;
///
/// let mut up = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None)?;
/// let mut down = up.clone();
/// up.set_onsite(&array![-1.0], None);
/// down.set_onsite(&array![1.0], None);
/// let model = CollinearModel::new(up, down)?;
/// let (e_up, e_down) = model.solve_band_onek(&array![0.0])?;
/// assert_eq!((e_up[0], e_down[0]), (-1.0, 1.0));
/// # Ok::<(), Rustb::TbError>(())
/// ```
#[derive(Clone, Debug)]
pub struct CollinearModel<const DIM: usize = 3, R: RMatrixData = NoRMatrix> {
    up: Model<false, DIM, R>,
    down: Model<false, DIM, R>,
}

impl<const DIM: usize, R: RMatrixData> CollinearModel<DIM, R> {
    /// Take ownership of the spin-up and spin-down models without copying them.
    ///
    /// Both models must validate and have exactly equal lattice entries and
    /// atom lists, in order, including species, fractional positions, orbital
    /// IDs and optional magnetic moments. No tolerance, wrapping or reordering
    /// is applied. The caller must use a common energy reference. Orbital
    /// bases are not matched between the two channels.
    pub fn new(up: Model<false, DIM, R>, down: Model<false, DIM, R>) -> Result<Self> {
        up.validate()?;
        down.validate()?;
        if up.lat != down.lat {
            return Err(TbError::InvalidModelInvariant {
                invariant: "collinear_lattice",
                message: "up and down must have exactly the same lattice".into(),
            });
        }
        if up.atoms.len() != down.atoms.len() {
            return Err(TbError::InvalidModelInvariant {
                invariant: "collinear_atoms",
                message: format!(
                    "up and down have different atom counts ({} and {})",
                    up.atoms.len(),
                    down.atoms.len()
                ),
            });
        }
        for (i, (a, b)) in up.atoms.iter().zip(&down.atoms).enumerate() {
            if a.atom_type() != b.atom_type()
                || a.position_ref() != b.position_ref()
                || a.orbitals() != b.orbitals()
                || a.magnetic_moment() != b.magnetic_moment()
            {
                return Err(TbError::InvalidModelInvariant {
                    invariant: "collinear_atoms",
                    message: format!("up and down atom {i} must have identical metadata"),
                });
            }
        }
        Ok(Self { up, down })
    }

    /// Borrow the common lattice matrix (real-space lattice vectors as rows).
    pub fn lat(&self) -> &Array2<f64> {
        &self.up.lat
    }

    /// Borrow the common atomic metadata in model order.
    pub fn atoms(&self) -> &[crate::Atom] {
        &self.up.atoms
    }

    /// Borrow the spin-up model.
    pub fn up(&self) -> &Model<false, DIM, R> {
        &self.up
    }

    /// Borrow the spin-down model.
    pub fn down(&self) -> &Model<false, DIM, R> {
        &self.down
    }

    /// Recover the owned models in `up, down` order.
    pub fn into_channels(self) -> (Model<false, DIM, R>, Model<false, DIM, R>) {
        (self.up, self.down)
    }

    /// Return the two independently sorted energy arrays in `up, down` order.
    pub fn solve_band_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        Ok((
            self.up.solve_band_onek(kvec)?,
            self.down.solve_band_onek(kvec)?,
        ))
    }

    /// Return independently sorted energies in the same window for both channels.
    ///
    /// Uses the half-open window `(low, high]` and absolute energy tolerance of
    /// [`Solve::solve_band_range_onek`]. Nonpositive tolerance selects LAPACK's
    /// default. The two arrays may have different lengths, including zero.
    pub fn solve_band_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy_window: (f64, f64),
        tolerance: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        Ok((
            self.up
                .solve_band_range_onek(kvec, energy_window, tolerance)?,
            self.down
                .solve_band_range_onek(kvec, energy_window, tolerance)?,
        ))
    }

    /// Serial counterpart of [`Self::solve_band_all_parallel`].
    ///
    /// Returns `(up, down)`, each shaped `(nk, nband_channel)` in input k order.
    pub fn solve_band_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        Ok((
            self.up.solve_band_all(kvec)?,
            self.down.solve_band_all(kvec)?,
        ))
    }

    /// Return `up, down` energy arrays, each with shape `(nk, nband)`.
    ///
    /// The channels run sequentially, with k-points parallelized inside each
    /// solve. Each channel keeps the input k-point order. Energies are sorted
    /// independently at each k-point; matching bands between channels is a
    /// separate calculation.
    pub fn solve_band_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        Ok((
            self.up.solve_band_all_parallel(kvec)?,
            self.down.solve_band_all_parallel(kvec)?,
        ))
    }

    /// Return `((energies_up, vectors_up), (energies_down, vectors_down))`.
    ///
    /// Each channel uses its own [`super::Gauge::Atom`] embedding and the row-ket
    /// convention of [`Solve::solve_onek`]: `vectors[band, basis]`. No phase,
    /// basis or band matching is performed between channels or k-points.
    pub fn solve_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
    ) -> Result<(
        (Array1<f64>, Array2<Complex<f64>>),
        (Array1<f64>, Array2<Complex<f64>>),
    )> {
        Ok((self.up.solve_onek(kvec)?, self.down.solve_onek(kvec)?))
    }

    /// Windowed counterpart of [`Self::solve_onek`], in `(up, down)` order.
    ///
    /// Both channels use the same `(low, high]` window and absolute energy
    /// tolerance; nonpositive tolerance selects LAPACK's default. Each vector
    /// array is `(nselected_channel, nband_channel)`, possibly with zero rows.
    pub fn solve_range_onek<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix1>,
        energy_window: (f64, f64),
        tolerance: f64,
    ) -> Result<(
        (Array1<f64>, Array2<Complex<f64>>),
        (Array1<f64>, Array2<Complex<f64>>),
    )> {
        Ok((
            self.up.solve_range_onek(kvec, energy_window, tolerance)?,
            self.down.solve_range_onek(kvec, energy_window, tolerance)?,
        ))
    }

    /// Solve both channels serially, returning `((E_up, C_up), (E_down, C_down))`.
    ///
    /// Each energy array is `(nk, nband_channel)`; vectors are
    /// `(nk, nband_channel, nband_channel)` with row ket coefficients in that
    /// channel's Atom gauge. Input k order is preserved, with independent
    /// ascending energy order at each point. See [`Solve::solve_all`].
    pub fn solve_all<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(
        (Array2<f64>, Array3<Complex<f64>>),
        (Array2<f64>, Array3<Complex<f64>>),
    )> {
        Ok((self.up.solve_all(kvec)?, self.down.solve_all(kvec)?))
    }

    /// Parallel-k counterpart of [`Self::solve_all`], with identical axes.
    ///
    /// Channels run sequentially. Returned eigenvector arrays for both channels
    /// are retained and are outside the per-channel Fourier workspace budget;
    /// pass k-point chunks when the full `nk * nband_channel²` arrays are too large.
    pub fn solve_all_parallel<S: Data<Elem = f64>>(
        &self,
        kvec: &ArrayBase<S, Ix2>,
    ) -> Result<(
        (Array2<f64>, Array3<Complex<f64>>),
        (Array2<f64>, Array3<Complex<f64>>),
    )> {
        Ok((
            self.up.solve_all_parallel(kvec)?,
            self.down.solve_all_parallel(kvec)?,
        ))
    }

    /// Apply the same spin-conserving uniform drive to both channels.
    pub fn floquet_effective_model(
        &self,
        drive: &FloquetDrive,
        options: Option<&FloquetEffectiveOptions>,
    ) -> Result<Self> {
        Ok(Self {
            up: self.up.floquet_effective_model(drive, options)?,
            down: self.down.floquet_effective_model(drive, options)?,
        })
    }

    /// Visit both effective channels together for each `(order, harmonic_max)`.
    ///
    /// Callbacks run in input order, including duplicate requests, and stop
    /// on the first callback error. Invalid cutoff/order pairs and oversized
    /// harmonic-cache layouts are rejected before callbacks. Both channels
    /// prepare each batch before emitting its results; a failure preparing a
    /// batch can suppress its callbacks. An empty request list does nothing.
    ///
    /// Runs consecutive batches of two requests and prepares channel harmonics
    /// separately per batch. At most two completed up-channel models are held
    /// while solving the down channel. The FFT workspace budget applies to each
    /// channel separately, not to the retained models or callback work.
    pub fn floquet_effective_sweep(
        &self,
        drive: &FloquetDrive,
        requests: &[(usize, isize)],
        mut visit: impl FnMut((usize, isize), &Self) -> Result<()>,
    ) -> Result<()> {
        if requests.is_empty() {
            return Ok(());
        }
        let has_ac = drive_has_ac_components(drive);
        self.up.effective_sweep_cache_max(requests, has_ac)?;
        self.down.effective_sweep_cache_max(requests, has_ac)?;
        for batch in requests.chunks(2) {
            let mut up_models = Vec::with_capacity(batch.len());
            let up_result = self
                .up
                .floquet_effective_sweep_owned(drive, batch, |key, up| {
                    up_models.push((key, up));
                    Ok(())
                });
            // Pair the completed up prefix even if a later up solve failed.
            let ready = &batch[..up_models.len()];
            let mut up_models = up_models.into_iter();
            self.down
                .floquet_effective_sweep_owned(drive, ready, |key, down| {
                    let (_, up) = up_models
                        .next()
                        .filter(|(up_key, _)| *up_key == key)
                        .ok_or_else(|| {
                            TbError::Other(
                                "collinear Floquet sweeps produced different requests".into(),
                            )
                        })?;
                    visit(key, &Self { up, down })
                })?;
            up_result?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
