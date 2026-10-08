//! Checked conversions for spin directions and spatial dimensions.
use crate::SpinDirection;
use crate::TbError;
use crate::model::Dimension;

// Conversion from integer types to Option<SpinDirection>.
// Note: we cannot impl From<usize> for Option<SpinDirection> due to orphan rules
// (both From and Option are foreign). Use SpinDirection::from_usize() instead.
impl SpinDirection {
    /// Convert a `usize` (0=I, 1=X, 2=Y, 3=Z) to `Option<SpinDirection>`.
    /// Returns `None` for spin index 0 (identity), `Some(SpinDirection::X/Y/Z)` for 1/2/3.
    ///
    /// # Panics
    ///
    /// Panics for indices outside 0..=3. Prefer [`SpinDirection::try_from_index`]
    /// for a non-panicking conversion.
    pub fn from_index(index: usize) -> Option<SpinDirection> {
        match Self::try_from_index(index) {
            Ok(value) => value,
            Err(error) => panic!("{error}"),
        }
    }

    /// Fallible conversion from a `usize` spin index.
    ///
    /// `0` maps to `Ok(None)` (identity), `1..=3` map to
    /// `Ok(Some(SpinDirection::X/Y/Z))`, and anything else returns
    /// [`TbError::InvalidSpinValue`].
    pub fn try_from_index(index: usize) -> crate::error::Result<Option<SpinDirection>> {
        match index {
            0 => Ok(None),
            1 => Ok(Some(SpinDirection::X)),
            2 => Ok(Some(SpinDirection::Y)),
            3 => Ok(Some(SpinDirection::Z)),
            _ => Err(TbError::InvalidSpinValue {
                spin: index,
                supported: vec![0, 1, 2, 3],
            }),
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
