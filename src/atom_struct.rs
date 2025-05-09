use ndarray::Array1;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::fmt;
///This is the orbital projection
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum OrbProj {
    /// $$\ket{s}=\ket{0,0}$$
    s,
    /// $$\ket{p_x}=\frac{1}{\sqrt{2}}\lt(\ket{1,-1}-\ket{1,1}\rt)$$
    px,
    /// $$\ket{p_y}=\frac{i}{\sqrt{2}}\lt(\ket{1,-1}+\ket{1,1}\rt)$$
    py,
    /// $$\ket{p_z}=\ket{1,0}$$
    pz,
    /// $$\ket{d_{xy}}=-\f{i}{\sqrt{2}}\lt(\ket{2,2}-\ket{2,-2}\rt)$$
    dxy,
    /// $$\ket{d_{yz}}=-\f{i}{\sqrt{2}}\lt(\ket{2,1}+\ket{2,-1}\rt)$$
    dyz,
    /// $$\ket{d_{xz}}=-\f{1}{\sqrt{2}}\lt(\ket{2,1}-\ket{2,-1}\rt)$$
    dxz,
    /// $$\ket{d_{z^2}}=\ket{2,0}$$
    dz2,
    /// $$\ket{d_{x^2-y^2}}=\f{1}{\sqrt{2}}\lt(\ket{2,2}+\ket{2,-2}\rt)$$
    dx2y2,
    /// $$\ket{f_{z^3}}=\ket{3,0}$$
    fz3,
    /// $$\ket{f_{xz^2}}=\f{1}{\sqrt{2}}\lt(\ket{3,1}-\ket{3,-1}\rt)$$
    fxz2,
    /// $$\ket{f_{yz^2}}=-\f{i}{\sqrt{2}}\lt(\ket{3,1}+\ket{3,-1}\rt)$$
    fyz2,
    /// $$\ket{f_{z(x^2-y^2)}}=\f{1}{\sqrt{2}}\lt(\ket{3,2}+\ket{3,-2}\rt)$$
    fzx2y2,
    /// $$\ket{f_{xyz}}=-\f{i}{\sqrt{2}}\lt(\ket{3,2}-\ket{3,-2}\rt)$$
    fxyz,
    /// $$\ket{f_{x(x^2-3y^2)}}=\f{1}{\sqrt{2}}\lt(\ket{3,3}-\ket{3,-3}\rt)$$
    fxx23y2,
    /// $$\ket{f_{y(3x^2-y^2)}}=-\f{i}{\sqrt{2}}\lt(\ket{3,3}+\ket{3,-3}\rt)$$
    fy3x2y2,
    /// $$\ket{sp_{1}}=\frac{1}{\sqrt{2}}\lt(\ket{s}+\ket{p}\rt)$$
    sp_1,
    /// $$\ket{sp_{2}}=\frac{1}{\sqrt{2}}\lt(\ket{s}-\ket{p}\rt)$$
    sp_2,
    /// $$\ket{sp^2_{1}}=\f{1}{\sqrt{3}}\ket{s}-\f{1}{\sqrt{6}}\ket{p_x}+\f{1}{\sqrt{2}}\ket{p_y}$$
    sp2_1,
    /// $$\ket{sp^2_{1}}=\f{1}{\sqrt{3}}\ket{s}-\f{1}{\sqrt{6}}\ket{p_x}-\f{1}{\sqrt{2}}\ket{p_y}$$
    sp2_2,
    /// $$\ket{sp^2_{1}}=\f{1}{\sqrt{3}}\ket{s}+\f{2}{\sqrt{6}}\ket{p_x}$$
    sp2_3,
    /// $$\ket{sp^3_{1}}=\frac{1}{2}\lt(\ket{s}+\ket{p_x}+\ket{p_y}+\ket{p_z}\rt)$$
    sp3_1,
    /// $$\ket{sp^3_{2}}=\frac{1}{2}\lt(\ket{s}+\ket{p_x}-\ket{p_y}-\ket{p_z}\rt)$$
    sp3_2,
    /// $$\ket{sp^3_{3}}=\frac{1}{2}\lt(\ket{s}-\ket{p_x}+\ket{p_y}-\ket{p_z}\rt)$$
    sp3_3,
    /// $$\ket{sp^3_{4}}=\frac{1}{2}\lt(\ket{s}-\ket{p_x}-\ket{p_y}+\ket{p_z}\rt)$$
    sp3_4,
    /// $$\ket{sp^3d_{1}}=\f{1}{\sqrt{3}}\ket{s}-\f{1}{\sqrt{6}}\ket{p_x}+\f{1}{\sqrt{2}}\ket{p_y}$$
    sp3d_1,
    /// $$\ket{sp^3d_{2}}=\f{1}{\sqrt{3}}\ket{s}-\f{1}{\sqrt{6}}\ket{p_x}-\f{1}{\sqrt{2}}\ket{p_y}$$
    sp3d_2,
    /// $$\ket{sp^3d_{3}}=\f{1}{\sqrt{3}}\ket{s}+\f{2}{\sqrt{6}}\ket{p_x}$$
    sp3d_3,
    /// $$\ket{sp^3d_{4}}=\f{1}{\sqrt{2}}\lt(\ket{p_z}+\ket{d_{z^2}}\rt)$$
    sp3d_4,
    /// $$\ket{sp^3d_{5}}=-\f{1}{\sqrt{2}}\lt(\ket{p_z}-\ket{d_{z^2}}\rt)$$
    sp3d_5,
    /// $$\ket{sp^3d^2_{1}}=\frac{1}{\sqrt{6}}\ket{s}-\f{1}{\sqrt{2}}\ket{p_x}-\f{1}{\sqrt{12}}\ket{d_{z^2}}+\f{1}{2}\ket{d_{x^2-y^2}}$$
    sp3d2_1,
    /// $$\ket{sp^3d^2_{2}}=\frac{1}{\sqrt{6}}\ket{s}+\f{1}{\sqrt{2}}\ket{p_x}-\f{1}{\sqrt{12}}\ket{d_{z^2}}+\f{1}{2}\ket{d_{x^2-y^2}}$$
    sp3d2_2,
    /// $$\ket{sp^3d^2_{3}}=\frac{1}{\sqrt{6}}\ket{s}-\f{1}{\sqrt{2}}\ket{p_x}-\f{1}{\sqrt{12}}\ket{d_{z^2}}-\f{1}{2}\ket{d_{x^2-y^2}}$$
    sp3d2_3,
    /// $$\ket{sp^3d^2_{4}}=\frac{1}{\sqrt{6}}\ket{s}+\f{1}{\sqrt{2}}\ket{p_x}-\f{1}{\sqrt{12}}\ket{d_{z^2}}-\f{1}{2}\ket{d_{x^2-y^2}}$$
    sp3d2_4,
    /// $$\ket{sp^3d^2_{5}}=\frac{1}{\sqrt{6}}\ket{s}-\f{1}{\sqrt{2}}\ket{p_z}+\f{1}{\sqrt{3}}\ket{d_{z^2}}$$
    sp3d2_5,
    /// $$\ket{sp^3d^2_{6}}=\frac{1}{\sqrt{6}}\ket{s}+\f{1}{\sqrt{2}}\ket{p_z}+\f{1}{\sqrt{3}}\ket{d_{z^2}}$$
    sp3d2_6,
}

impl OrbProj {
    pub fn from_str(s: &str) -> Self {
        match s {
            "s" => OrbProj::s,
            "px" => OrbProj::px,
            "py" => OrbProj::py,
            "pz" => OrbProj::pz,
            "dxy" => OrbProj::dxy,
            "dxz" => OrbProj::dxz,
            "dyz" => OrbProj::dyz,
            "dz2" => OrbProj::dz2,
            "dx2-y2" => OrbProj::dx2y2,
            "fz3" => OrbProj::fz3,
            "fxz2" => OrbProj::fxz2,
            "fyz2" => OrbProj::fyz2,
            "fzx2y2" => OrbProj::fzx2y2,
            "fxyz" => OrbProj::fxyz,
            "fxx2-3y2" => OrbProj::fxx23y2,
            "fy3x2-y2" => OrbProj::fy3x2y2,
            "sp-1" => OrbProj::sp_1,
            "sp-2" => OrbProj::sp_2,
            "sp2-1" => OrbProj::sp2_1,
            "sp2-2" => OrbProj::sp2_2,
            "sp2-3" => OrbProj::sp2_3,
            "sp3-1" => OrbProj::sp3_1,
            "sp3-2" => OrbProj::sp3_2,
            "sp3-3" => OrbProj::sp3_3,
            "sp3-4" => OrbProj::sp3_4,
            "sp3d-1" => OrbProj::sp3d_1,
            "sp3d-2" => OrbProj::sp3d_2,
            "sp3d-3" => OrbProj::sp3d_3,
            "sp3d-4" => OrbProj::sp3d_4,
            "sp3d-5" => OrbProj::sp3d_5,
            "sp3d2-1" => OrbProj::sp3d2_1,
            "sp3d2-2" => OrbProj::sp3d2_2,
            "sp3d2-3" => OrbProj::sp3d2_3,
            "sp3d2-4" => OrbProj::sp3d2_4,
            "sp3d2-5" => OrbProj::sp3d2_5,
            "sp3d2-6" => OrbProj::sp3d2_6,
            _ => panic!("Wrong, unrecognised projections {}", s),
        }
    }
    /// 这个函数是将 \ket{px},\ket{py},\ket{pz} 等原子轨道基转化为以 l,m 为基的函数的.
    /// 它输入一个原子轨道比如 $\ket{px}$, 输出一个 array![Complex<f64>;16], 表示
    /// $$[\ket{0,0},\ket{1,-1},\ket{1,0},\ket{1,1},\ket{2,-2},\cdots,\ket{3,3}]$$
    pub fn to_quantum_number(&self) -> Array1<Complex<f64>> {
        let s = match self {
            OrbProj::s => [Complex::new(0.0, 0.0); 16],
            OrbProj::px => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[1] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s[3] = Complex::new(-1.0 / 2_f64.sqrt(), 0.0);
                s
            }
            OrbProj::py => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[1] = Complex::new(0.0, 1.0 / 2_f64.sqrt());
                s[3] = Complex::new(0.0, 1.0 / 2_f64.sqrt());
                s
            }
            OrbProj::pz => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[2] = Complex::new(1.0, 0.0);
                s
            }
            OrbProj::dxy => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[4] = Complex::new(0.0, 1.0 / 2_f64.sqrt());
                s[8] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s
            }
            OrbProj::dyz => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[5] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s[7] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s
            }
            OrbProj::dxz => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[5] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s[7] = Complex::new(-1.0 / 2_f64.sqrt(), 0.0);
                s
            }
            OrbProj::dz2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[6] = Complex::new(1.0, 0.0);
                s
            }
            OrbProj::dx2y2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[4] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s[8] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s
            }
            OrbProj::fz3 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[12] = Complex::new(1.0, 0.0);
                s
            }
            OrbProj::fxz2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[11] = Complex::new(-1.0 / 2_f64.sqrt(), 0.0);
                s[13] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s
            }
            OrbProj::fyz2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[11] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s[13] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s
            }
            OrbProj::fzx2y2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[10] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s[14] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s
            }
            OrbProj::fxyz => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[10] = Complex::new(0.0, 1.0 / 2_f64.sqrt());
                s[14] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s
            }
            OrbProj::fxx23y2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[9] = Complex::new(-1.0 / 2_f64.sqrt(), 0.0);
                s[15] = Complex::new(1.0 / 2_f64.sqrt(), 0.0);
                s
            }
            OrbProj::fy3x2y2 => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[9] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s[15] = Complex::new(0.0, -1.0 / 2_f64.sqrt());
                s
            }
            _ => panic!("for sp,sp2,sp3 et.al is now not consideredZ"),
        };
        Array1::from(s.to_vec())
    }
}

impl fmt::Display for OrbProj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            OrbProj::s => "s",
            OrbProj::px => "px",
            OrbProj::py => "py",
            OrbProj::pz => "pz",
            OrbProj::dxy => "dxy",
            OrbProj::dxz => "dxz",
            OrbProj::dyz => "dyz",
            OrbProj::dz2 => "dz2",
            OrbProj::dx2y2 => "dx2-y2",
            OrbProj::fz3 => "fz3",
            OrbProj::fxz2 => "fxz2",
            OrbProj::fyz2 => "fyz2",
            OrbProj::fzx2y2 => "fzx2y2",
            OrbProj::fxyz => "fxyz",
            OrbProj::fxx23y2 => "fxx2-3y2",
            OrbProj::fy3x2y2 => "fy3x2-y2",
            OrbProj::sp_1 => "sp-1",
            OrbProj::sp_2 => "sp-2",
            OrbProj::sp2_1 => "sp2-1",
            OrbProj::sp2_2 => "sp2-2",
            OrbProj::sp2_3 => "sp2-3",
            OrbProj::sp3_1 => "sp3-1",
            OrbProj::sp3_2 => "sp3-2",
            OrbProj::sp3_3 => "sp3-3",
            OrbProj::sp3_4 => "sp3-4",
            OrbProj::sp3d_1 => "sp3d-1",
            OrbProj::sp3d_2 => "sp3d-2",
            OrbProj::sp3d_3 => "sp3d-3",
            OrbProj::sp3d_4 => "sp3d-4",
            OrbProj::sp3d_5 => "sp3d-5",
            OrbProj::sp3d2_1 => "sp3d2-1",
            OrbProj::sp3d2_2 => "sp3d2-2",
            OrbProj::sp3d2_3 => "sp3d2-3",
            OrbProj::sp3d2_4 => "sp3d2-4",
            OrbProj::sp3d2_5 => "sp3d2-5",
            OrbProj::sp3d2_6 => "sp3d2-6",
        };
        write!(f, "{}", s)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub enum AtomType {
    /// This is the type of the Atom
    H,
    He,
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    K,
    Ca,
    Sc,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    Zn,
    Ga,
    Ge,
    As,
    Se,
    Br,
    Kr,
    Rb,
    Sr,
    Y,
    Zr,
    Nb,
    Mo,
    Tc,
    Ru,
    Rh,
    Pd,
    Ag,
    Cd,
    In,
    Sn,
    Sb,
    Te,
    I,
    Xe,
    Cs,
    Ba,
    La,
    Ce,
    Pr,
    Nd,
    Pm,
    Sm,
    Eu,
    Gd,
    Tb,
    Dy,
    Ho,
    Er,
    Tm,
    Yb,
    Lu,
    Hf,
    Ta,
    W,
    Re,
    Os,
    Ir,
    Pt,
    Au,
    Hg,
    Tl,
    Pb,
    Bi,
    Po,
    At,
    Rn,
    Fr,
    Ra,
}

impl AtomType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "H" => AtomType::H,
            "He" => AtomType::He,
            "Li" => AtomType::Li,
            "Be" => AtomType::Be,
            "B" => AtomType::B,
            "C" => AtomType::C,
            "N" => AtomType::N,
            "O" => AtomType::O,
            "F" => AtomType::F,
            "Ne" => AtomType::Ne,
            "Na" => AtomType::Na,
            "Mg" => AtomType::Mg,
            "Al" => AtomType::Al,
            "Si" => AtomType::Si,
            "P" => AtomType::P,
            "S" => AtomType::S,
            "Cl" => AtomType::Cl,
            "Ar" => AtomType::Ar,
            "K" => AtomType::K,
            "Ca" => AtomType::Ca,
            "Sc" => AtomType::Sc,
            "Ti" => AtomType::Ti,
            "V" => AtomType::V,
            "Cr" => AtomType::Cr,
            "Mn" => AtomType::Mn,
            "Fe" => AtomType::Fe,
            "Co" => AtomType::Co,
            "Ni" => AtomType::Ni,
            "Cu" => AtomType::Cu,
            "Zn" => AtomType::Zn,
            "Ga" => AtomType::Ga,
            "Ge" => AtomType::Ge,
            "As" => AtomType::As,
            "Se" => AtomType::Se,
            "Br" => AtomType::Br,
            "Kr" => AtomType::Kr,
            "Rb" => AtomType::Rb,
            "Sr" => AtomType::Sr,
            "Y" => AtomType::Y,
            "Zr" => AtomType::Zr,
            "Nb" => AtomType::Nb,
            "Mo" => AtomType::Mo,
            "Tc" => AtomType::Tc,
            "Ru" => AtomType::Ru,
            "Rh" => AtomType::Rh,
            "Pd" => AtomType::Pd,
            "Ag" => AtomType::Ag,
            "Cd" => AtomType::Cd,
            "In" => AtomType::In,
            "Sn" => AtomType::Sn,
            "Sb" => AtomType::Sb,
            "Te" => AtomType::Te,
            "I" => AtomType::I,
            "Xe" => AtomType::Xe,
            "Cs" => AtomType::Cs,
            "Ba" => AtomType::Ba,
            "La" => AtomType::La,
            "Ce" => AtomType::Ce,
            "Pr" => AtomType::Pr,
            "Nd" => AtomType::Nd,
            "Pm" => AtomType::Pm,
            "Sm" => AtomType::Sm,
            "Eu" => AtomType::Eu,
            "Gd" => AtomType::Gd,
            "Tb" => AtomType::Tb,
            "Dy" => AtomType::Dy,
            "Ho" => AtomType::Ho,
            "Er" => AtomType::Er,
            "Tm" => AtomType::Tm,
            "Yb" => AtomType::Yb,
            "Lu" => AtomType::Lu,
            "Hf" => AtomType::Hf,
            "Ta" => AtomType::Ta,
            "W" => AtomType::W,
            "Re" => AtomType::Re,
            "Os" => AtomType::Os,
            "Ir" => AtomType::Ir,
            "Pt" => AtomType::Pt,
            "Au" => AtomType::Au,
            "Hg" => AtomType::Hg,
            "Tl" => AtomType::Tl,
            "Pb" => AtomType::Pb,
            "Bi" => AtomType::Bi,
            "Po" => AtomType::Po,
            "At" => AtomType::At,
            "Rn" => AtomType::Rn,
            "Fr" => AtomType::Fr,
            "Ra" => AtomType::Ra,
            _ => panic!("unrecognized atom {}", s),
        }
    }
    pub fn to_str(&self) -> &str {
        let symbol = match self {
            AtomType::H => "H",
            AtomType::He => "He",
            AtomType::Li => "Li",
            AtomType::Be => "Be",
            AtomType::B => "B",
            AtomType::C => "C",
            AtomType::N => "N",
            AtomType::O => "O",
            AtomType::F => "F",
            AtomType::Ne => "Ne",
            AtomType::Na => "Na",
            AtomType::Mg => "Mg",
            AtomType::Al => "Al",
            AtomType::Si => "Si",
            AtomType::P => "P",
            AtomType::S => "S",
            AtomType::Cl => "Cl",
            AtomType::Ar => "Ar",
            AtomType::K => "K",
            AtomType::Ca => "Ca",
            AtomType::Sc => "Sc",
            AtomType::Ti => "Ti",
            AtomType::V => "V",
            AtomType::Cr => "Cr",
            AtomType::Mn => "Mn",
            AtomType::Fe => "Fe",
            AtomType::Co => "Co",
            AtomType::Ni => "Ni",
            AtomType::Cu => "Cu",
            AtomType::Zn => "Zn",
            AtomType::Ga => "Ga",
            AtomType::Ge => "Ge",
            AtomType::As => "As",
            AtomType::Se => "Se",
            AtomType::Br => "Br",
            AtomType::Kr => "Kr",
            AtomType::Rb => "Rb",
            AtomType::Sr => "Sr",
            AtomType::Y => "Y",
            AtomType::Zr => "Zr",
            AtomType::Nb => "Nb",
            AtomType::Mo => "Mo",
            AtomType::Tc => "Tc",
            AtomType::Ru => "Ru",
            AtomType::Rh => "Rh",
            AtomType::Pd => "Pd",
            AtomType::Ag => "Ag",
            AtomType::Cd => "Cd",
            AtomType::In => "In",
            AtomType::Sn => "Sn",
            AtomType::Sb => "Sb",
            AtomType::Te => "Te",
            AtomType::I => "I",
            AtomType::Xe => "Xe",
            AtomType::Cs => "Cs",
            AtomType::Ba => "Ba",
            AtomType::La => "La",
            AtomType::Ce => "Ce",
            AtomType::Pr => "Pr",
            AtomType::Nd => "Nd",
            AtomType::Pm => "Pm",
            AtomType::Sm => "Sm",
            AtomType::Eu => "Eu",
            AtomType::Gd => "Gd",
            AtomType::Tb => "Tb",
            AtomType::Dy => "Dy",
            AtomType::Ho => "Ho",
            AtomType::Er => "Er",
            AtomType::Tm => "Tm",
            AtomType::Yb => "Yb",
            AtomType::Lu => "Lu",
            AtomType::Hf => "Hf",
            AtomType::Ta => "Ta",
            AtomType::W => "W",
            AtomType::Re => "Re",
            AtomType::Os => "Os",
            AtomType::Ir => "Ir",
            AtomType::Pt => "Pt",
            AtomType::Au => "Au",
            AtomType::Hg => "Hg",
            AtomType::Tl => "Tl",
            AtomType::Pb => "Pb",
            AtomType::Bi => "Bi",
            AtomType::Po => "Po",
            AtomType::At => "At",
            AtomType::Rn => "Rn",
            AtomType::Fr => "Fr",
            AtomType::Ra => "Ra",
        };
        symbol
    }
}

impl fmt::Display for AtomType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            AtomType::H => "H ",
            AtomType::He => "He",
            AtomType::Li => "Li",
            AtomType::Be => "Be",
            AtomType::B => "B ",
            AtomType::C => "C ",
            AtomType::N => "N ",
            AtomType::O => "O ",
            AtomType::F => "F ",
            AtomType::Ne => "Ne",
            AtomType::Na => "Na",
            AtomType::Mg => "Mg",
            AtomType::Al => "Al",
            AtomType::Si => "Si",
            AtomType::P => "P ",
            AtomType::S => "S ",
            AtomType::Cl => "Cl",
            AtomType::Ar => "Ar",
            AtomType::K => "K ",
            AtomType::Ca => "Ca",
            AtomType::Sc => "Sc",
            AtomType::Ti => "Ti",
            AtomType::V => "V ",
            AtomType::Cr => "Cr",
            AtomType::Mn => "Mn",
            AtomType::Fe => "Fe",
            AtomType::Co => "Co",
            AtomType::Ni => "Ni",
            AtomType::Cu => "Cu",
            AtomType::Zn => "Zn",
            AtomType::Ga => "Ga",
            AtomType::Ge => "Ge",
            AtomType::As => "As",
            AtomType::Se => "Se",
            AtomType::Br => "Br",
            AtomType::Kr => "Kr",
            AtomType::Rb => "Rb",
            AtomType::Sr => "Sr",
            AtomType::Y => "Y ",
            AtomType::Zr => "Zr",
            AtomType::Nb => "Nb",
            AtomType::Mo => "Mo",
            AtomType::Tc => "Tc",
            AtomType::Ru => "Ru",
            AtomType::Rh => "Rh",
            AtomType::Pd => "Pd",
            AtomType::Ag => "Ag",
            AtomType::Cd => "Cd",
            AtomType::In => "In",
            AtomType::Sn => "Sn",
            AtomType::Sb => "Sb",
            AtomType::Te => "Te",
            AtomType::I => "I ",
            AtomType::Xe => "Xe",
            AtomType::Cs => "Cs",
            AtomType::Ba => "Ba",
            AtomType::La => "La",
            AtomType::Ce => "Ce",
            AtomType::Pr => "Pr",
            AtomType::Nd => "Nd",
            AtomType::Pm => "Pm",
            AtomType::Sm => "Sm",
            AtomType::Eu => "Eu",
            AtomType::Gd => "Gd",
            AtomType::Tb => "Tb",
            AtomType::Dy => "Dy",
            AtomType::Ho => "Ho",
            AtomType::Er => "Er",
            AtomType::Tm => "Tm",
            AtomType::Yb => "Yb",
            AtomType::Lu => "Lu",
            AtomType::Hf => "Hf",
            AtomType::Ta => "Ta",
            AtomType::W => "W ",
            AtomType::Re => "Re",
            AtomType::Os => "Os",
            AtomType::Ir => "Ir",
            AtomType::Pt => "Pt",
            AtomType::Au => "Au",
            AtomType::Hg => "Hg",
            AtomType::Tl => "Tl",
            AtomType::Pb => "Pb",
            AtomType::Bi => "Bi",
            AtomType::Po => "Po",
            AtomType::At => "At",
            AtomType::Rn => "Rn",
            AtomType::Fr => "Fr",
            AtomType::Ra => "Ra",
        };
        write!(f, "{}", symbol)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Atom {
    position: Array1<f64>,
    name: AtomType,
    atom_list: usize,
    magnetic: [f64; 3],
}

impl Atom {
    pub fn position(&self) -> Array1<f64> {
        self.position.clone()
    }
    pub fn norb(&self) -> usize {
        self.atom_list
    }
    pub fn atom_type(&self) -> AtomType {
        self.name
    }
    pub fn push_orb(&mut self) {
        self.atom_list += 1;
    }
    pub fn remove_orb(&mut self) {
        self.atom_list -= 1;
    }
    pub fn change_type(&mut self, new_type: AtomType) {
        self.name = new_type;
    }
    pub fn new(position: Array1<f64>, atom_list: usize, name: AtomType) -> Atom {
        Atom {
            position,
            atom_list,
            name,
            magnetic: [0.0, 0.0, 0.0],
        }
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Atom {{ name: {}, position: {:?}, atom_list: {}, magnetic moment:{:?}}}",
            self.name, self.position, self.atom_list, self.magnetic
        )
    }
}
