//! Optical polarization helpers.
//!
//! This module provides geometry utilities for building transverse polarization
//! vectors for a given wave propagation direction.

use crate::error::{Result, TbError};
use ndarray::Array1;
use ndarray::prelude::*;
use num_complex::Complex;

#[derive(Clone, Debug)]
pub struct IncidentBasis {
    /// Normalized incident-light direction.
    pub k_hat: Array1<f64>,
    /// First transverse unit vector.
    pub e1: Array1<f64>,
    /// Second transverse unit vector.
    pub e2: Array1<f64>,
}

impl IncidentBasis {
    /// Build a right-handed transverse basis from an incident wave-vector
    /// direction in Cartesian coordinates.
    pub fn from_direction(k_hat_cart: &Array1<f64>) -> Result<Self> {
        let k_hat = normalize3(k_hat_cart)?;
        let reference = if k_hat[2].abs() < 0.9 {
            arr1(&[0.0, 0.0, 1.0])
        } else {
            arr1(&[1.0, 0.0, 0.0])
        };
        Self::from_direction_and_reference(&k_hat, &reference)
    }

    /// Build a right-handed transverse basis from an incident wave-vector
    /// direction and an explicit reference vector.
    ///
    /// The reference vector must be linearly independent of `k_hat_cart` and
    /// finite.
    pub fn from_direction_and_reference(
        k_hat_cart: &Array1<f64>,
        reference: &Array1<f64>,
    ) -> Result<Self> {
        if k_hat_cart.len() != 3 {
            return Err(TbError::DimensionMismatch {
                context: "IncidentBasis::from_direction_and_reference".to_string(),
                expected: 3,
                found: k_hat_cart.len(),
            });
        }
        if reference.len() != 3 {
            return Err(TbError::DimensionMismatch {
                context: "IncidentBasis::from_direction_and_reference: reference".to_string(),
                expected: 3,
                found: reference.len(),
            });
        }
        let k_hat = normalize3(k_hat_cart)?;
        let reference_norm = normalize3(reference)?;

        // e1 is orthogonal to both k_hat and the reference direction, and should
        // not be numerically tiny.
        let mut e1 = cross3(&reference_norm, &k_hat);
        if e1.iter().all(|x| x.abs() < f64::EPSILON) {
            return Err(TbError::Other(
                "IncidentBasis reference direction is parallel to propagation direction"
                    .to_string(),
            ));
        }
        e1 = normalize3(&e1)?;

        let e2 = normalize3(&cross3(&k_hat, &e1))?;
        Ok(Self { k_hat, e1, e2 })
    }

    /// Return `jones[0] * e1 + jones[1] * e2`.
    pub fn polarization(&self, jones: [Complex<f64>; 2]) -> Array1<Complex<f64>> {
        let mut out = Array1::<Complex<f64>>::zeros(3);
        for i in 0..3 {
            out[i] = jones[0] * self.e1[i] + jones[1] * self.e2[i];
        }
        out
    }
}

fn normalize3(v: &Array1<f64>) -> Result<Array1<f64>> {
    let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(TbError::Other(
            "IncidentBasis::normalize expects a finite non-zero vector".to_string(),
        ));
    }
    Ok(v / norm)
}

fn cross3(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}
