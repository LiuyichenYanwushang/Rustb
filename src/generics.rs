//!这个是给程序提供泛型支持的模块
use crate::SpinDirection;
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

// 实现从u8到SpinDirection的转换
impl From<u8> for SpinDirection {
    fn from(value: u8) -> Self {
        match value {
            0 => SpinDirection::None,
            1 => SpinDirection::x,
            2 => SpinDirection::y,
            3 => SpinDirection::z,
            _ => panic!("Invalid value for SpinDirection"),
        }
    }
}

// 实现从usize到SpinDirection的转换
impl From<usize> for SpinDirection {
    fn from(value: usize) -> Self {
        match value {
            0 => SpinDirection::None,
            1 => SpinDirection::x,
            2 => SpinDirection::y,
            3 => SpinDirection::z,
            _ => panic!("Invalid value for SpinDirection"),
        }
    }
}

// 实现从i32到SpinDirection的转换
impl From<i32> for SpinDirection {
    fn from(value: i32) -> Self {
        match value {
            0 => SpinDirection::None,
            1 => SpinDirection::x,
            2 => SpinDirection::y,
            3 => SpinDirection::z,
            _ => panic!("Invalid value for SpinDirection"),
        }
    }
}
