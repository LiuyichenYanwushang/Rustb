//! Generic trait definitions for numeric type flexibility and hopping parameter conversions.
use crate::SpinDirection;
use crate::TbError;
use crate::model::Dimension;
use num_complex::Complex64;
use num_traits::identities::Zero;

pub trait ToFloat {
    fn to_float(self) -> f64;
}
impl ToFloat for usize {
    fn to_float(self) -> f64 {
        self as f64
    }
}

impl ToFloat for isize {
    fn to_float(self) -> f64 {
        self as f64
    }
}

impl ToFloat for f32 {
    fn to_float(self) -> f64 {
        self as f64
    }
}

impl ToFloat for f64 {
    fn to_float(self) -> f64 {
        self
    }
}

pub trait usefloat: Copy + Clone + Zero + std::fmt::Display + PartialOrd {
    fn from<T: ToFloat>(n: T) -> Self;
}
impl usefloat for f32 {
    fn from<T: ToFloat>(n: T) -> Self {
        n.to_float() as f32
    }
}

impl usefloat for f64 {
    fn from<T: ToFloat>(n: T) -> Self {
        n.to_float()
    }
}

//这里的trait是为了让set_hop 可以同时满足 f64 和 Complex64 的
pub trait hop_use: Copy + Clone + Zero {
    fn to_complex(&self) -> Complex64;
}
impl hop_use for f64 {
    fn to_complex(&self) -> Complex64 {
        Complex64::new(*self, 0.0)
    }
}

impl hop_use for Complex64 {
    fn to_complex(&self) -> Complex64 {
        *self
    }
}

// Conversion from integer types to Option<SpinDirection>.
// Note: we cannot impl From<usize> for Option<SpinDirection> due to orphan rules
// (both From and Option are foreign). Use SpinDirection::from_usize() instead.
impl SpinDirection {
    /// Convert a `usize` (0=I, 1=X, 2=Y, 3=Z) to `Option<SpinDirection>`.
    /// Returns `None` for spin index 0 (identity), `Some(SpinDirection::X/Y/Z)` for 1/2/3.
    pub fn from_index(index: usize) -> Option<SpinDirection> {
        match index {
            0 => None,
            1 => Some(SpinDirection::X),
            2 => Some(SpinDirection::Y),
            3 => Some(SpinDirection::Z),
            _ => panic!("Invalid spin index: {}. Valid values are 0 (I), 1 (X), 2 (Y), 3 (Z).", index),
        }
    }
}

impl From<Dimension> for usize {
    fn from(d: Dimension) -> Self {
        d as usize
    }
}

impl TryFrom<usize> for Dimension {
    type Error = TbError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Dimension::one),
            2 => Ok(Dimension::two),
            3 => Ok(Dimension::three),
            _ => Err(TbError::InvalidDimension {
                dim: value,
                supported: vec![1, 2, 3],
            }),
        }
    }
}
