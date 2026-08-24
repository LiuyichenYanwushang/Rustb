//! Optical polarization helpers.
//!
//! This module provides geometry utilities for building transverse polarization
//! vectors for a given wave propagation direction.
//!
//! [`IncidentBasis::from_direction`] chooses a numerically convenient reference
//! axis independently for each direction.  That convention is useful for a
//! single beam, but its transverse frame is not globally phase-continuous over
//! the sphere.  Coherent multi-beam constructions should instead use
//! [`IncidentBasis::from_direction_and_reference`] with an explicitly chosen
//! laboratory-frame reference, or provide their Cartesian complex amplitudes
//! directly.

use crate::error::{Result, TbError};
use ndarray::Array1;
use ndarray::prelude::*;
use num_complex::Complex;

/// A right-handed orthonormal frame `(e1, e2, k_hat)` for transverse light.
///
/// With the time convention `Re[a exp(-i omega t)]`, the Jones vector
/// `(1, +i) / sqrt(2)` denotes the corresponding complex Cartesian amplitude
/// in this frame.  Changing the transverse reference rotates that amplitude by
/// a beam-dependent phase, which matters for coherent multi-beam interference.
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
    ///
    /// The automatically selected reference axis changes between `z` and `x`
    /// for numerical stability.  Use [`Self::from_direction_and_reference`] if
    /// relative optical phases between several beams must be fixed.
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

        // e1 is orthogonal to both k_hat and the reference direction.  Since
        // both inputs are normalized, its norm is sin(angle); reject a nearly
        // parallel reference before normalization amplifies roundoff.
        let e1_raw = cross3(&reference_norm, &k_hat);
        let transverse_norm = norm3(&e1_raw);
        if !transverse_norm.is_finite() || transverse_norm <= 1.0e-12 {
            return Err(TbError::Other(
                "IncidentBasis reference direction is parallel or numerically too close to the propagation direction"
                    .to_string(),
            ));
        }
        let e1 = e1_raw / transverse_norm;

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
    if v.len() != 3 {
        return Err(TbError::DimensionMismatch {
            context: "IncidentBasis direction".to_string(),
            expected: 3,
            found: v.len(),
        });
    }
    let norm = norm3(v);
    if !norm.is_finite() || norm <= 0.0 {
        return Err(TbError::Other(
            "IncidentBasis::normalize expects a finite non-zero vector".to_string(),
        ));
    }
    Ok(v / norm)
}

fn norm3(v: &Array1<f64>) -> f64 {
    v[0].hypot(v[1]).hypot(v[2])
}

fn cross3(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn from_direction_rejects_wrong_dimensions_without_panicking() {
        for direction in [arr1(&[1.0]), arr1(&[1.0, 0.0]), arr1(&[1.0, 0.0, 0.0, 0.0])] {
            assert!(matches!(
                IncidentBasis::from_direction(&direction),
                Err(TbError::DimensionMismatch {
                    expected: 3,
                    found: _,
                    ..
                })
            ));
        }
    }

    #[test]
    fn from_direction_rejects_zero_and_non_finite_vectors() {
        for direction in [
            arr1(&[0.0, 0.0, 0.0]),
            arr1(&[f64::NAN, 0.0, 1.0]),
            arr1(&[f64::INFINITY, 0.0, 1.0]),
        ] {
            assert!(IncidentBasis::from_direction(&direction).is_err());
        }
    }

    #[test]
    fn explicit_reference_builds_right_handed_orthonormal_frame() {
        let basis = IncidentBasis::from_direction_and_reference(
            &arr1(&[1.0, 2.0, 3.0]),
            &arr1(&[0.0, 0.0, 1.0]),
        )
        .unwrap();

        let dot = |a: &Array1<f64>, b: &Array1<f64>| {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>()
        };
        assert!((dot(&basis.k_hat, &basis.k_hat) - 1.0).abs() < 1.0e-14);
        assert!((dot(&basis.e1, &basis.e1) - 1.0).abs() < 1.0e-14);
        assert!((dot(&basis.e2, &basis.e2) - 1.0).abs() < 1.0e-14);
        assert!(dot(&basis.k_hat, &basis.e1).abs() < 1.0e-14);
        assert!(dot(&basis.k_hat, &basis.e2).abs() < 1.0e-14);
        assert!(dot(&basis.e1, &basis.e2).abs() < 1.0e-14);
        assert!(
            cross3(&basis.e1, &basis.e2)
                .iter()
                .zip(basis.k_hat.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f64, f64::max)
                < 1.0e-14
        );
    }

    #[test]
    fn explicit_reference_rejects_parallel_and_nearly_parallel_axes() {
        let direction = arr1(&[0.0, 0.0, 1.0]);
        assert!(
            IncidentBasis::from_direction_and_reference(&direction, &arr1(&[0.0, 0.0, 2.0]))
                .is_err()
        );
        assert!(
            IncidentBasis::from_direction_and_reference(&direction, &arr1(&[1.0e-14, 0.0, 1.0]))
                .is_err()
        );
    }

    #[test]
    fn stable_norm_accepts_large_and_small_finite_directions() {
        for direction in [
            arr1(&[f64::MAX / 4.0, f64::MAX / 4.0, 0.0]),
            arr1(&[f64::MIN_POSITIVE, 0.0, 0.0]),
        ] {
            let basis = IncidentBasis::from_direction(&direction).unwrap();
            assert!(basis.k_hat.iter().all(|value| value.is_finite()));
        }
    }
}
