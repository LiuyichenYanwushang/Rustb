//! Bloch-electron orbital angular momentum from band energies and velocities.
//!
//! This is the band-space operator of Eq. (3) in
//! [Busch, Mertig and Göbel (2023)](https://doi.org/10.1103/PhysRevResearch.5.043052),
//! including complex off-diagonal matrix elements. It is distinct from the
//! local atomic-orbital approximation [`Model::orb_angular`] and is not the
//! Brillouin-zone integral for bulk orbital magnetization.

use crate::error::{Result, TbError};
use crate::phy_const::{Element_charge, hbar, mass_charge};
use crate::velocity::Velocity;
use crate::{Gauge, Model, RMatrixData};
use ndarray::prelude::*;
use ndarray_linalg::{Eigh, UPLO};
use num_complex::Complex;

/// Band-space Bloch-electron orbital angular momentum.
pub trait OrbitalAngular: Velocity {
    /// Return `L / hbar` with shape `(3, nsta, nsta)` in ascending-energy band order.
    /// The first axis is always `(Lx, Ly, Lz)`; a two-dimensional model lies in
    /// the xy plane and only `Lz` can be nonzero. A one-dimensional model returns zero.
    ///
    /// `kvec` is fractional reciprocal momentum. Hamiltonian energies are eV;
    /// lattice vectors and stored position matrices use angstroms, following
    /// the velocity convention. The orbital g-factor is the electron value 1.
    ///
    /// With `D^a` the Cartesian derivatives returned by [`Velocity::gen_v`],
    /// transformed to the band basis, the convention is
    /// ```math
    /// (L_a/\hbar)_{mn} = -\frac{i m_e e u^2}{2\hbar^2}
    /// \sum_{l\notin\{m,n\}}\left(\frac{1}{E_l-E_m}+\frac{1}{E_l-E_n}\right)
    /// (D^b_{ml}D^c_{ln}-D^c_{ml}D^b_{ln}),
    /// ```
    /// where `(a,b,c)` is cyclic, `u = 1e-10` m, and `e` converts eV
    /// to joules. The sum also applies to `m == n`. Matrix elements transform
    /// covariantly under band rephasing; they are not individually gauge invariant.
    ///
    /// # Errors
    /// Rejects invalid geometry, non-finite/non-Hermitian operators,
    /// and band gaps at or below `1e-10` eV. The quoted single-band formula does
    /// not define a prescription inside degenerate subspaces; no gap is silently
    /// replaced by a cutoff or discarded.
    fn orbital_angular_momentum_onek(&self, kvec: &Array1<f64>) -> Result<Array3<Complex<f64>>>;
}

impl<const SPIN: bool, const DIM: usize, R: RMatrixData> OrbitalAngular for Model<SPIN, DIM, R> {
    fn orbital_angular_momentum_onek(&self, kvec: &Array1<f64>) -> Result<Array3<Complex<f64>>> {
        if !(1..=3).contains(&DIM) {
            return Err(TbError::InvalidDimension {
                dim: DIM,
                supported: vec![1, 2, 3],
            });
        }
        if kvec.len() != DIM {
            return Err(TbError::KVectorLengthMismatch {
                expected: DIM,
                actual: kvec.len(),
            });
        }
        let factor = mass_charge * Element_charge / (2.0 * hbar * hbar) * 1e-20;
        if kvec.iter().any(|x| !x.is_finite()) {
            return Err(TbError::Other(
                "orbital angular momentum requires finite k coordinates".into(),
            ));
        }
        self.validate()?;
        let (velocity, ham) = self.gen_v(kvec, Gauge::Atom);
        if ham
            .iter()
            .chain(velocity.iter())
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(TbError::Other(
                "orbital angular momentum requires finite Hamiltonian and velocity matrices".into(),
            ));
        }
        let scale = ham.iter().map(|z| z.norm()).fold(1.0_f64, f64::max);
        if ham
            .indexed_iter()
            .any(|((i, j), z)| (*z - ham[[j, i]].conj()).norm() > 1e-10 * scale)
        {
            return Err(TbError::Other(
                "orbital angular momentum requires a Hermitian Hamiltonian".into(),
            ));
        }
        for component in velocity.axis_iter(Axis(0)) {
            let scale = component.iter().map(|z| z.norm()).fold(1.0_f64, f64::max);
            if component
                .indexed_iter()
                .any(|((i, j), z)| (*z - component[[j, i]].conj()).norm() > 1e-10 * scale)
            {
                return Err(TbError::Other(
                    "orbital angular momentum requires Hermitian velocity matrices".into(),
                ));
            }
        }
        // An explicit Fortran layout makes ndarray-linalg return columns of H's
        // eigenvectors, without the C-layout transpose/conjugation compensation.
        let mut column_major = Array2::zeros((self.nsta(), self.nsta()).f());
        column_major.assign(&ham);
        let (energies, ket) = column_major.eigh(UPLO::Lower)?;
        if energies
            .windows(2)
            .into_iter()
            .any(|pair| pair[1] - pair[0] <= 1e-10)
        {
            return Err(TbError::Other("Bloch orbital angular momentum is undefined by the single-band formula for gaps <= 1e-10 eV".into()));
        }
        let bra = ket.t().mapv(|z| z.conj());
        let mut band_velocity = Array3::zeros((3, self.nsta(), self.nsta()));
        for axis in 0..DIM {
            band_velocity
                .index_axis_mut(Axis(0), axis)
                .assign(&bra.dot(&velocity.index_axis(Axis(0), axis).dot(&ket)));
        }
        Ok(orbital_from_band(&energies, &band_velocity, factor))
    }
}

fn orbital_from_band(
    energies: &Array1<f64>,
    velocity: &Array3<Complex<f64>>,
    factor: f64,
) -> Array3<Complex<f64>> {
    let count = energies.len();
    let mut angular = Array3::zeros((3, count, count));
    for m in 0..count {
        for n in 0..count {
            for l in 0..count {
                if l == m || l == n {
                    continue;
                }
                let weight = Complex::new(0.0, -factor)
                    * (1.0 / (energies[l] - energies[m]) + 1.0 / (energies[l] - energies[n]));
                for a in 0..3 {
                    let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                    angular[[a, m, n]] += weight
                        * (velocity[[b, m, l]] * velocity[[c, l, n]]
                            - velocity[[c, m, l]] * velocity[[b, l, n]]);
                }
            }
        }
    }
    angular
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    // H = sin(qx) sigma_x + sin(qy) sigma_y + mass sigma_z, q=2pi*k.
    fn dirac_model(mass: f64) -> Model<false, 2> {
        let mut model = Model::tb_model(Array2::eye(2), Array2::zeros((2, 2)), None).unwrap();
        model.set_onsite(&array![mass, -mass], None);
        model.add_hop(Complex::new(0.0, -0.5), 0, 1, &array![1, 0], None);
        model.add_hop(Complex::new(0.0, 0.5), 0, 1, &array![-1, 0], None);
        model.add_hop(-0.5, 0, 1, &array![0, 1], None);
        model.add_hop(0.5, 0, 1, &array![0, -1], None);
        model
    }

    #[test]
    fn bloch_angular_matches_two_band_analytic_result() {
        let mass = 0.7;
        let model = dirac_model(mass);
        let unit = 1e-10;
        for k in [array![0.0, 0.0], array![0.13, 0.07], array![0.31, 0.1]] {
            let angular = model.orbital_angular_momentum_onek(&k).unwrap();
            let (x, y) = (TAU * k[0], TAU * k[1]);
            let expected = -mass_charge * Element_charge * unit * unit / (hbar * hbar)
                * mass
                * x.cos()
                * y.cos()
                / (mass * mass + x.sin().powi(2) + y.sin().powi(2));
            assert_eq!(angular.dim(), (3, 2, 2));
            for n in 0..2 {
                assert!((angular[[2, n, n]].re - expected).abs() < 1e-12);
                assert!(angular[[2, n, n]].im.abs() < 1e-12);
            }
            assert!(
                angular
                    .slice(s![..2, .., ..])
                    .iter()
                    .all(|z| z.norm() < 1e-12)
            );
            assert!(angular[[2, 0, 1]].norm() < 1e-12);
        }
    }

    #[test]
    fn bloch_angular_scales_with_lattice_and_preserves_energy_origin() {
        let mut model = dirac_model(0.7);
        let k = array![0.13, 0.07];
        let reference = model.orbital_angular_momentum_onek(&k).unwrap();
        for n in 0..2 {
            model.ham[[0, n, n]] += 17.0;
        }
        let shifted = model.orbital_angular_momentum_onek(&k).unwrap();
        assert!(
            reference
                .iter()
                .zip(shifted.iter())
                .all(|(a, b)| (a - b).norm() < 1e-12)
        );
        // Lengths stay in angstroms: shrinking the physical lattice by ten
        // scales the two velocity factors, and hence L/hbar, by 1/100.
        model.lat *= 0.1;
        let scaled = model.orbital_angular_momentum_onek(&k).unwrap();
        assert!(
            reference
                .iter()
                .zip(&scaled)
                .all(|(a, b)| (a * 0.01 - b).norm() < 1e-12)
        );
    }

    #[test]
    fn multiband_angular_is_hermitian_and_band_phase_covariant() {
        let energies = array![-1.0, 0.4, 2.0];
        let velocity = Array3::from_shape_fn((3, 3, 3), |(a, m, n)| {
            Complex::new(
                (a + m + n + 1) as f64 * 0.2,
                (m as f64 - n as f64) * (a + 1) as f64 * 0.13,
            )
        });
        let angular = orbital_from_band(&energies, &velocity, 0.7);
        let phases = [
            Complex::from_polar(1.0, 0.31),
            Complex::from_polar(1.0, -0.72),
            Complex::from_polar(1.0, 1.19),
        ];
        let rephased = Array3::from_shape_fn((3, 3, 3), |(a, m, n)| {
            phases[m].conj() * velocity[[a, m, n]] * phases[n]
        });
        let transformed = orbital_from_band(&energies, &rephased, 0.7);
        assert!(angular[[2, 0, 1]].norm() > 1e-3);
        for ((a, m, n), value) in angular.indexed_iter() {
            assert!((*value - angular[[a, n, m]].conj()).norm() < 1e-12);
            assert!((transformed[[a, m, n]] - phases[m].conj() * value * phases[n]).norm() < 1e-12);
        }
    }

    #[test]
    fn bloch_angular_rejects_undefined_inputs() {
        let model = dirac_model(0.7);
        assert!(model.orbital_angular_momentum_onek(&array![0.0]).is_err());
        assert!(
            model
                .orbital_angular_momentum_onek(&array![f64::NAN, 0.0])
                .is_err()
        );
        assert!(
            dirac_model(0.0)
                .orbital_angular_momentum_onek(&array![0.0, 0.0])
                .is_err()
        );
        let chain = Model::<false, 1>::tb_model(array![[1.0]], array![[0.0]], None).unwrap();
        assert_eq!(
            chain.orbital_angular_momentum_onek(&array![0.2]).unwrap(),
            Array3::zeros((3, 1, 1))
        );

        // A non-Hermitian position operator produces non-Hermitian velocities,
        // even when the Hamiltonian and all array shapes are valid.
        let mut invalid = Model::<false, 2, crate::HasRMatrix>::tb_model(
            Array2::eye(2),
            Array2::zeros((2, 2)),
            None,
        )
        .unwrap();
        invalid.set_onsite(&array![-1.0, 1.0], None);
        invalid.rmatrix[[0, 0, 0, 1]] = Complex::new(1.0, 0.0);
        invalid.rmatrix[[0, 1, 0, 1]] = Complex::new(1.0, 0.0);
        invalid.rmatrix[[0, 1, 1, 0]] = Complex::new(1.0, 0.0);
        assert!(
            invalid
                .orbital_angular_momentum_onek(&array![0.0, 0.0])
                .is_err()
        );
    }
}
