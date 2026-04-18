//! Uniform magnetic field for a tight-binding model.
//!
//! This module exposes only one public interface:
//!
//! $$
//! \texttt{Model::add\_magnetic\_field(mag\_dir, expand, phi\_total)}.
//! $$
//!
//! Everything else is kept internal on purpose.
//!
//! # Mathematical convention
//!
//! We work with a tight-binding Hamiltonian stored in real space as
//!
//! $$
//! H_{ij}(\mathbf R)=\langle i,\mathbf 0|\hat H|j,\mathbf R\rangle,
//! $$
//!
//! where `Model.lat` stores the lattice vectors as **rows** and `Model.orb`
//! stores orbital positions in **fractional coordinates**.
//!
//! For a uniform magnetic field, the orbital coupling is introduced through the
//! Peierls substitution
//!
//! $$
//! H_{ij}(\mathbf R)\;\to\;H_{ij}(\mathbf R)\exp\bigl(i\theta_{ij}(\mathbf R)\bigr),
//! $$
//!
//! with the electron-sign convention
//!
//! $$
//! \theta_{ij}(\mathbf R)
//! =-\frac{e}{\hbar}\int \mathbf A(\mathbf r)\cdot d\mathbf l
//! =-2\pi\frac{1}{\Phi_0}\int \mathbf A(\mathbf r)\cdot d\mathbf l,
//! \qquad \Phi_0=\frac{h}{e}.
//! $$
//!
//! ## Magnetic supercell and gauge
//!
//! Let `expand = [m, n]`. In 3D, `mag_dir = k` means that the magnetic field is
//! chosen parallel to the **lattice direction** $\mathbf a_k$, while the flux is
//! threaded through the plaquette spanned by the other two lattice directions.
//!
//! In 2D, only `mag_dir = 2` is allowed, corresponding to the usual out-of-plane
//! orbital magnetic field.
//!
//! After building the magnetic supercell, we work in its fractional coordinates
//! $(U_1,U_2)$ and use the periodic Landau gauge
//!
//! $$
//! \mathbf A(\mathbf r)=N_\phi\,\Phi_0\,U_1\,\nabla U_2,
//! $$
//!
//! where `phi_total = N_\phi` is the **integer number of flux quanta through the
//! full magnetic supercell**.
//!
//! For a hop from orbital $j$ in translated cell $(R_1,R_2)$ to orbital $i$ in the
//! home cell, the Peierls phase used here is
//!
//! $$
//! \theta_{ij}
//! =-2\pi N_\phi\left[
//! \frac{U_{1,i}+V_{1,j}}{2}\,(V_{2,j}-U_{2,i})-R_1V_{2,j}
//! \right],
//! $$
//!
//! with
//!
//! $$
//! V_{1,j}=U_{1,j}+R_1,
//! \qquad
//! V_{2,j}=U_{2,j}+R_2.
//! $$
//!
//! The last term is the boundary gauge patch required by magnetic periodic
//! boundary conditions.
//!
//! ## What happens to `rmatrix`
//!
//! The library stores
//!
//! $$
//! r^{\alpha}_{ij}(\mathbf R)=\langle i,\mathbf 0|\hat r_\alpha|j,\mathbf R\rangle.
//! $$
//!
//! Under the same magnetic gauge dressing of the localized basis, every nonlocal
//! one-body matrix element acquires the same link phase. Therefore, if `rmatrix`
//! contains nonlocal entries, they must transform as
//!
//! $$
//! r^{\alpha}_{ij}(\mathbf R)
//! \to
//! e^{i\theta_{ij}(\mathbf R)}r^{\alpha}_{ij}(\mathbf R).
//! $$
//!
//! This implementation therefore applies the same Peierls phase both to `ham` and
//! to `rmatrix`. For purely onsite diagonal `rmatrix`, this changes nothing because
//! the onsite phase is exactly unity.
//!
//! ## Zeeman coupling
//!
//! If spin is enabled, the onsite Zeeman term is added in Cartesian spin space:
//!
//! $$
//! H_Z=\frac{g\mu_B}{2}\,\mathbf B\cdot\boldsymbol\sigma.
//! $$
//!
//! In 3D the actual magnetic field is taken as
//!
//! $$
//! \mathbf B=\beta\,\mathbf a_{\mathrm{mag}},
//! $$
//!
//! where $\mathbf a_{\mathrm{mag}}$ is the lattice vector selected by `mag_dir`, and
//! $\beta$ is fixed by the flux condition through the primitive plaquette. This is
//! the correct non-orthogonal-lattice generalization of the usual Landau-gauge
//! construction.

use crate::error::{Result, TbError};
use crate::find_R;
use crate::Model;
use ndarray::prelude::*;
use ndarray::*;
use num_complex::Complex;
use std::f64::consts::TAU;

const FLUX_QUANTUM_T_M2: f64 = 4.135_667_696e-15_f64;
const MU_B_EV_PER_T: f64 = 5.788_381_8060e-5_f64;

/// Public interface for adding a uniform magnetic field to a tight-binding model.
pub trait MagneticField {
    /// Add a uniform magnetic field to the model.
    ///
    /// The interface is intentionally minimal:
    ///
    /// $$
    /// \texttt{mag\_dir}
    /// $$
    /// selects the lattice direction of the field in 3D, or must be `2` in 2D.
    ///
    /// $$
    /// \texttt{expand=[m,n]}
    /// $$
    /// gives the magnetic-supercell enlargement factors along the two directions
    /// perpendicular to the field.
    ///
    /// $$
    /// \texttt{phi\_total}=N_\phi
    /// $$
    /// is the integer number of flux quanta threading the full magnetic supercell.
    ///
    /// Equivalently, the flux per primitive plaquette is
    ///
    /// $$
    /// \frac{\Phi_{\mathrm{primitive}}}{\Phi_0}=\frac{N_\phi}{mn}.
    /// $$
    fn add_magnetic_field(&self, mag_dir: usize, expand: [usize; 2], phi_total: isize) -> Result<Self>
    where
        Self: Sized;
}

impl MagneticField for Model {
    fn add_magnetic_field(&self, mag_dir: usize, expand: [usize; 2], phi_total: isize) -> Result<Self> {
        validate_dimensions(self.dim_r(), mag_dir, expand)?;
        let perp = perpendicular_dirs(self.dim_r(), mag_dir)?;

        let mut u = Array2::<f64>::eye(self.dim_r());
        u[[perp[0], perp[0]]] = expand[0] as f64;
        u[[perp[1], perp[1]]] = expand[1] as f64;
        let super_model = self.make_supercell(&u)?;

        let d1 = perp[0];
        let d2 = perp[1];
        let norb = super_model.norb();
        let total_basis = super_model.nsta();
        let dim_r = super_model.dim_r();

        let mut new_ham = super_model.ham.clone();
        let mut new_rmatrix = super_model.rmatrix.clone();

        if phi_total != 0 {
            for i_r in 0..super_model.hamR.nrows() {
                let r_vec = super_model.hamR.row(i_r);
                let r1 = r_vec[d1];
                let r2 = r_vec[d2];

                let mut ham_slice = new_ham.slice_mut(s![i_r, .., ..]);
                let mut r_slice = new_rmatrix.slice_mut(s![i_r, .., .., ..]);

                for i in 0..total_basis {
                    let orb_i = i % norb;
                    let u1_i = super_model.orb[[orb_i, d1]];
                    let u2_i = super_model.orb[[orb_i, d2]];

                    for j in 0..total_basis {
                        let orb_j = j % norb;
                        let u1_j = super_model.orb[[orb_j, d1]];
                        let u2_j = super_model.orb[[orb_j, d2]];

                        let phase = peierls_phase_periodic_landau(
                            phi_total,
                            u1_i,
                            u2_i,
                            u1_j,
                            u2_j,
                            r1,
                            r2,
                        );
                        let peierls = Complex::new(phase.cos(), phase.sin());

                        ham_slice[[i, j]] *= peierls;
                        for alpha in 0..dim_r {
                            r_slice[[alpha, i, j]] *= peierls;
                        }
                    }
                }
            }
        }

        if super_model.spin && phi_total != 0 {
            let b_cart_tesla = magnetic_field_cartesian(&self.lat, self.dim_r(), mag_dir, expand, phi_total)?;
            let zeeman = zeeman_block_cartesian(b_cart_tesla, 2.0);

            let zero_r = Array1::<isize>::zeros(super_model.dim_r());
            let onsite_index = find_R(&super_model.hamR, &zero_r)
                .ok_or_else(|| TbError::Other("R = 0 block not found in magnetic supercell".to_string()))?;
            let mut ham0 = new_ham.slice_mut(s![onsite_index, .., ..]);
            add_zeeman_term(&mut ham0, norb, zeeman);
        }

        let mut out = super_model;
        out.ham = new_ham;
        out.rmatrix = new_rmatrix;
        Ok(out)
    }
}

fn peierls_phase_periodic_landau(
    phi_total: isize,
    u1_i: f64,
    u2_i: f64,
    u1_j: f64,
    u2_j: f64,
    r1: isize,
    r2: isize,
) -> f64 {
    let v1_j = u1_j + r1 as f64;
    let v2_j = u2_j + r2 as f64;
    let reduced_line_integral_in_flux_quanta =
        0.5 * (u1_i + v1_j) * (v2_j - u2_i) - (r1 as f64) * v2_j;
    -TAU * (phi_total as f64) * reduced_line_integral_in_flux_quanta
}

fn add_zeeman_term(
    ham0: &mut ArrayViewMut2<'_, Complex<f64>>,
    norb: usize,
    z: [[Complex<f64>; 2]; 2],
) {
    for orb in 0..norb {
        let up = orb;
        let dn = orb + norb;
        ham0[[up, up]] += z[0][0];
        ham0[[up, dn]] += z[0][1];
        ham0[[dn, up]] += z[1][0];
        ham0[[dn, dn]] += z[1][1];
    }
}

fn zeeman_block_cartesian(b_cart_tesla: [f64; 3], g_factor: f64) -> [[Complex<f64>; 2]; 2] {
    let pref = 0.5 * g_factor * MU_B_EV_PER_T;
    let bx = pref * b_cart_tesla[0];
    let by = pref * b_cart_tesla[1];
    let bz = pref * b_cart_tesla[2];

    [
        [Complex::new(bz, 0.0), Complex::new(bx, -by)],
        [Complex::new(bx, by), Complex::new(-bz, 0.0)],
    ]
}

fn magnetic_field_cartesian(
    lat: &Array2<f64>,
    dim_r: usize,
    mag_dir: usize,
    expand: [usize; 2],
    phi_total: isize,
) -> Result<[f64; 3]> {
    let perp = perpendicular_dirs(dim_r, mag_dir)?;
    let flux_per_primitive = phi_total as f64 / (expand[0] as f64 * expand[1] as f64);

    if dim_r == 2 {
        let a1 = pad_to_3(lat.row(perp[0]));
        let a2 = pad_to_3(lat.row(perp[1]));
        let area_vec = cross3(a1, a2);
        let area_a2 = norm3(area_vec);
        if area_a2.abs() < 1e-14 {
            return Err(TbError::Other("2D lattice area is numerically zero".to_string()));
        }
        let b_mag = flux_per_primitive * FLUX_QUANTUM_T_M2 / (area_a2 * 1e-20);
        Ok([0.0, 0.0, b_mag])
    } else {
        let a_mag = pad_to_3(lat.row(mag_dir));
        let a_p1 = pad_to_3(lat.row(perp[0]));
        let a_p2 = pad_to_3(lat.row(perp[1]));
        let omega = dot3(a_mag, cross3(a_p1, a_p2));
        if omega.abs() < 1e-14 {
            return Err(TbError::Other(
                "oriented primitive volume is numerically zero".to_string(),
            ));
        }
        let beta = flux_per_primitive * FLUX_QUANTUM_T_M2 / (omega * 1e-20);
        Ok([beta * a_mag[0], beta * a_mag[1], beta * a_mag[2]])
    }
}

fn validate_dimensions(dim_r: usize, mag_dir: usize, expand: [usize; 2]) -> Result<()> {
    if expand[0] == 0 || expand[1] == 0 {
        return Err(TbError::InvalidSupercellSize(0));
    }
    match dim_r {
        2 => {
            if mag_dir != 2 {
                return Err(TbError::InvalidDirection { index: mag_dir, dim: dim_r });
            }
        }
        3 => {
            if mag_dir >= 3 {
                return Err(TbError::InvalidDirection { index: mag_dir, dim: dim_r });
            }
        }
        _ => {
            return Err(TbError::InvalidDimension {
                dim: dim_r,
                supported: vec![2, 3],
            });
        }
    }
    Ok(())
}

fn perpendicular_dirs(dim_r: usize, mag_dir: usize) -> Result<[usize; 2]> {
    match dim_r {
        2 => {
            if mag_dir == 2 {
                Ok([0, 1])
            } else {
                Err(TbError::InvalidDirection { index: mag_dir, dim: dim_r })
            }
        }
        3 => match mag_dir {
            0 => Ok([1, 2]),
            1 => Ok([2, 0]),
            2 => Ok([0, 1]),
            _ => Err(TbError::InvalidDirection { index: mag_dir, dim: dim_r }),
        },
        _ => Err(TbError::InvalidDimension {
            dim: dim_r,
            supported: vec![2, 3],
        }),
    }
}

fn pad_to_3(v: ArrayView1<'_, f64>) -> [f64; 3] {
    match v.len() {
        0 => [0.0, 0.0, 0.0],
        1 => [v[0], 0.0, 0.0],
        2 => [v[0], v[1], 0.0],
        _ => [v[0], v[1], v[2]],
    }
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(v: [f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}
