use crate::{Result, TbError};
use ndarray::Array1;
use num_complex::Complex;
use serde::{Deserialize, Serialize};
use std::fmt;
///This is the orbital projection
///
/// Variant names follow the physical orbital labels (`s`, `px`, `d_{z^2}`,
/// etc.) rather than Rust's CamelCase convention, so non_camel_case_types
/// is allowed intentionally here.
#[repr(u8)]
#[allow(non_camel_case_types)]
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
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "s" => Ok(OrbProj::s),
            "px" => Ok(OrbProj::px),
            "py" => Ok(OrbProj::py),
            "pz" => Ok(OrbProj::pz),
            "dxy" => Ok(OrbProj::dxy),
            "dxz" => Ok(OrbProj::dxz),
            "dyz" => Ok(OrbProj::dyz),
            "dz2" => Ok(OrbProj::dz2),
            "dx2-y2" => Ok(OrbProj::dx2y2),
            "fz3" => Ok(OrbProj::fz3),
            "fxz2" => Ok(OrbProj::fxz2),
            "fyz2" => Ok(OrbProj::fyz2),
            "fzx2y2" => Ok(OrbProj::fzx2y2),
            "fxyz" => Ok(OrbProj::fxyz),
            "fxx2-3y2" => Ok(OrbProj::fxx23y2),
            "fy3x2-y2" => Ok(OrbProj::fy3x2y2),
            "sp-1" => Ok(OrbProj::sp_1),
            "sp-2" => Ok(OrbProj::sp_2),
            "sp2-1" => Ok(OrbProj::sp2_1),
            "sp2-2" => Ok(OrbProj::sp2_2),
            "sp2-3" => Ok(OrbProj::sp2_3),
            "sp3-1" => Ok(OrbProj::sp3_1),
            "sp3-2" => Ok(OrbProj::sp3_2),
            "sp3-3" => Ok(OrbProj::sp3_3),
            "sp3-4" => Ok(OrbProj::sp3_4),
            "sp3d-1" => Ok(OrbProj::sp3d_1),
            "sp3d-2" => Ok(OrbProj::sp3d_2),
            "sp3d-3" => Ok(OrbProj::sp3d_3),
            "sp3d-4" => Ok(OrbProj::sp3d_4),
            "sp3d-5" => Ok(OrbProj::sp3d_5),
            "sp3d2-1" => Ok(OrbProj::sp3d2_1),
            "sp3d2-2" => Ok(OrbProj::sp3d2_2),
            "sp3d2-3" => Ok(OrbProj::sp3d2_3),
            "sp3d2-4" => Ok(OrbProj::sp3d2_4),
            "sp3d2-5" => Ok(OrbProj::sp3d2_5),
            "sp3d2-6" => Ok(OrbProj::sp3d2_6),
            //_ => panic!("Wrong, unrecognised projections {}", s),
            _ => Err(TbError::InvalidOrbitalProjection(s.to_string())),
        }
    }
    /// Converts atomic orbital basis functions ($\ket{p_x}$, $\ket{p_y}$, $\ket{p_z}$, etc.) into the $(l, m)$ quantum-number basis.
    /// Takes an atomic orbital such as $\ket{p_x}$ as input and returns a 16-element `Array1<Complex<f64>>` representing the expansion in:
    /// $$[\ket{0,0},\ket{1,-1},\ket{1,0},\ket{1,1},\ket{2,-2},\cdots,\ket{3,3}]$$
    pub fn to_quantum_number(&self) -> Result<Array1<Complex<f64>>> {
        let s = match self {
            OrbProj::s => {
                let mut s = [Complex::new(0.0, 0.0); 16];
                s[0] = Complex::new(1.0, 0.0);
                s
            }
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
            _ => return Err(TbError::HybridOrbitalNotSupported("sp,sp2,sp3".to_string())),
        };
        Ok(Array1::from(s.to_vec()))
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize, Ord, PartialOrd)]
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
    /// Atomic number associated with this element.
    ///
    /// Keeping this conversion in one place avoids leaking the declaration
    /// order of [`AtomType`] into crystallographic and file-format adapters.
    #[inline]
    pub const fn atomic_number(self) -> i32 {
        self as i32 + 1
    }
}

impl AtomType {
    /// Parse a chemical element symbol (e.g. `"C"`, `"Fe"`) into an [`AtomType`].
    ///
    /// Returns [`TbError::InvalidAtomType`] for unrecognized symbols.
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "H" => Ok(AtomType::H),
            "He" => Ok(AtomType::He),
            "Li" => Ok(AtomType::Li),
            "Be" => Ok(AtomType::Be),
            "B" => Ok(AtomType::B),
            "C" => Ok(AtomType::C),
            "N" => Ok(AtomType::N),
            "O" => Ok(AtomType::O),
            "F" => Ok(AtomType::F),
            "Ne" => Ok(AtomType::Ne),
            "Na" => Ok(AtomType::Na),
            "Mg" => Ok(AtomType::Mg),
            "Al" => Ok(AtomType::Al),
            "Si" => Ok(AtomType::Si),
            "P" => Ok(AtomType::P),
            "S" => Ok(AtomType::S),
            "Cl" => Ok(AtomType::Cl),
            "Ar" => Ok(AtomType::Ar),
            "K" => Ok(AtomType::K),
            "Ca" => Ok(AtomType::Ca),
            "Sc" => Ok(AtomType::Sc),
            "Ti" => Ok(AtomType::Ti),
            "V" => Ok(AtomType::V),
            "Cr" => Ok(AtomType::Cr),
            "Mn" => Ok(AtomType::Mn),
            "Fe" => Ok(AtomType::Fe),
            "Co" => Ok(AtomType::Co),
            "Ni" => Ok(AtomType::Ni),
            "Cu" => Ok(AtomType::Cu),
            "Zn" => Ok(AtomType::Zn),
            "Ga" => Ok(AtomType::Ga),
            "Ge" => Ok(AtomType::Ge),
            "As" => Ok(AtomType::As),
            "Se" => Ok(AtomType::Se),
            "Br" => Ok(AtomType::Br),
            "Kr" => Ok(AtomType::Kr),
            "Rb" => Ok(AtomType::Rb),
            "Sr" => Ok(AtomType::Sr),
            "Y" => Ok(AtomType::Y),
            "Zr" => Ok(AtomType::Zr),
            "Nb" => Ok(AtomType::Nb),
            "Mo" => Ok(AtomType::Mo),
            "Tc" => Ok(AtomType::Tc),
            "Ru" => Ok(AtomType::Ru),
            "Rh" => Ok(AtomType::Rh),
            "Pd" => Ok(AtomType::Pd),
            "Ag" => Ok(AtomType::Ag),
            "Cd" => Ok(AtomType::Cd),
            "In" => Ok(AtomType::In),
            "Sn" => Ok(AtomType::Sn),
            "Sb" => Ok(AtomType::Sb),
            "Te" => Ok(AtomType::Te),
            "I" => Ok(AtomType::I),
            "Xe" => Ok(AtomType::Xe),
            "Cs" => Ok(AtomType::Cs),
            "Ba" => Ok(AtomType::Ba),
            "La" => Ok(AtomType::La),
            "Ce" => Ok(AtomType::Ce),
            "Pr" => Ok(AtomType::Pr),
            "Nd" => Ok(AtomType::Nd),
            "Pm" => Ok(AtomType::Pm),
            "Sm" => Ok(AtomType::Sm),
            "Eu" => Ok(AtomType::Eu),
            "Gd" => Ok(AtomType::Gd),
            "Tb" => Ok(AtomType::Tb),
            "Dy" => Ok(AtomType::Dy),
            "Ho" => Ok(AtomType::Ho),
            "Er" => Ok(AtomType::Er),
            "Tm" => Ok(AtomType::Tm),
            "Yb" => Ok(AtomType::Yb),
            "Lu" => Ok(AtomType::Lu),
            "Hf" => Ok(AtomType::Hf),
            "Ta" => Ok(AtomType::Ta),
            "W" => Ok(AtomType::W),
            "Re" => Ok(AtomType::Re),
            "Os" => Ok(AtomType::Os),
            "Ir" => Ok(AtomType::Ir),
            "Pt" => Ok(AtomType::Pt),
            "Au" => Ok(AtomType::Au),
            "Hg" => Ok(AtomType::Hg),
            "Tl" => Ok(AtomType::Tl),
            "Pb" => Ok(AtomType::Pb),
            "Bi" => Ok(AtomType::Bi),
            "Po" => Ok(AtomType::Po),
            "At" => Ok(AtomType::At),
            "Rn" => Ok(AtomType::Rn),
            "Fr" => Ok(AtomType::Fr),
            "Ra" => Ok(AtomType::Ra),
            _ => Err(TbError::InvalidAtomType(s.to_string())),
        }
    }
    /// Standard chemical element symbol of this [`AtomType`] (e.g. `"Fe"`).
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

/// Strongly typed index of an atom in a [`Model`](crate::Model).
///
/// IDs are local to one model. They remain valid while the atom ordering is
/// unchanged.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct AtomId(usize);

impl AtomId {
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for AtomId {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

/// Strongly typed index of a physical (not spin-doubled) orbital in a model.
///
/// An `OrbitalId` addresses the same row in `Model::orb` and
/// `Model::orb_projection`, and the corresponding orbital axes of the
/// Hamiltonian. It is a safe model-local handle, not a pointer.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize, Serialize)]
pub struct OrbitalId(usize);

impl OrbitalId {
    #[inline]
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl From<usize> for OrbitalId {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

/// Atomic-site metadata and its explicit references to the model's orbitals.
///
/// The model owns the dense orbital arrays. An atom stores only typed orbital
/// IDs, which keeps the structure movable, cloneable, and serializable while
/// allowing non-contiguous orbital assignments.
#[derive(Debug, Clone, Serialize)]
pub struct Atom {
    position: Array1<f64>,
    name: AtomType,
    #[serde(default)]
    orbitals: Vec<OrbitalId>,
    /// Optional Cartesian magnetic moment of this site.
    ///
    /// `None` is the default and means that no magnetic moment is attached to
    /// the atom. `Some([0.0; 3])` is a distinct, explicit zero-moment input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    magnetic_moment: Option<[f64; 3]>,
}

/// Deserialization wire format supporting both the typed-ID representation
/// and Rustb's pre-0.7 per-atom orbital counts.
#[derive(Deserialize)]
pub(crate) struct AtomWire {
    position: Array1<f64>,
    name: AtomType,
    #[serde(default)]
    orbitals: Option<Vec<OrbitalId>>,
    #[serde(default)]
    atom_list: Option<usize>,
    #[serde(default, alias = "magnetic")]
    magnetic_moment: Option<[f64; 3]>,
}

impl AtomWire {
    fn into_atom(self, legacy_start: usize) -> std::result::Result<(Atom, usize), String> {
        if self
            .magnetic_moment
            .is_some_and(|moment| moment.iter().any(|component| !component.is_finite()))
        {
            return Err("atomic magnetic moment components must be finite".to_string());
        }
        let (orbitals, next_legacy) = match (self.orbitals, self.atom_list) {
            (Some(orbitals), None) => (orbitals, legacy_start),
            (None, Some(count)) => {
                let end = legacy_start
                    .checked_add(count)
                    .ok_or_else(|| "legacy atom orbital count overflows usize".to_string())?;
                ((legacy_start..end).map(OrbitalId::new).collect(), end)
            }
            (None, None) => (Vec::new(), legacy_start),
            (Some(_), Some(_)) => {
                return Err(
                    "atom contains both typed 'orbitals' and legacy 'atom_list' fields".to_string(),
                );
            }
        };
        Ok((
            Atom {
                position: self.position,
                name: self.name,
                orbitals,
                magnetic_moment: self.magnetic_moment,
            },
            next_legacy,
        ))
    }
}

pub(crate) fn atoms_from_wire(wires: Vec<AtomWire>) -> std::result::Result<Vec<Atom>, String> {
    let has_legacy = wires.iter().any(|wire| wire.atom_list.is_some());
    let has_typed = wires.iter().any(|wire| wire.orbitals.is_some());
    if has_legacy && has_typed {
        return Err("cannot mix legacy atom orbital counts with typed orbital IDs".to_string());
    }
    let mut next_legacy = 0;
    wires
        .into_iter()
        .map(|wire| {
            let (atom, next) = wire.into_atom(next_legacy)?;
            next_legacy = next;
            Ok(atom)
        })
        .collect()
}

impl<'de> Deserialize<'de> for Atom {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let wire = AtomWire::deserialize(deserializer)?;
        // The legacy per-atom orbital-count format cannot be decoded for a
        // standalone Atom: each atom would independently assign IDs starting
        // at 0, producing overlapping OrbitalIds that fail later at Model
        // attach with a cryptic DuplicateOrbitalOwner. The Model-level
        // deserializer assigns IDs sequentially instead.
        if wire.atom_list.is_some() {
            return Err(serde::de::Error::custom(
                "legacy per-atom orbital counts cannot be deserialized as a \
                 standalone Atom; deserialize the full Model (which assigns \
                 orbital IDs sequentially) or migrate the data to typed \
                 orbital IDs",
            ));
        }
        wire.into_atom(0)
            .map(|(atom, _)| atom)
            .map_err(serde::de::Error::custom)
    }
}

impl Atom {
    /// Fractional position of this atomic site.
    pub fn position(&self) -> Array1<f64> {
        self.position.clone()
    }

    /// Borrow the fractional position without allocating.
    pub fn position_ref(&self) -> &Array1<f64> {
        &self.position
    }

    /// Overwrite the fractional position (crate-internal normalization).
    pub(crate) fn set_position(&mut self, position: Array1<f64>) {
        self.position = position;
    }

    /// Number of model orbitals explicitly assigned to this atom.
    pub fn norb(&self) -> usize {
        self.orbitals.len()
    }

    /// Chemical species of this atomic site.
    pub fn atom_type(&self) -> AtomType {
        self.name
    }

    /// Model-local orbital IDs assigned to this atom.
    pub fn orbitals(&self) -> &[OrbitalId] {
        &self.orbitals
    }

    /// Optional Cartesian magnetic moment attached to the atomic site.
    pub fn magnetic_moment(&self) -> Option<[f64; 3]> {
        self.magnetic_moment
    }

    /// Attach a finite Cartesian magnetic moment to this atom.
    ///
    /// The moment is site metadata. Uniform external electric/magnetic fields
    /// remain per-analysis inputs in `SymmetryParameters` and are not stored
    /// here.
    pub fn set_magnetic_moment(&mut self, moment: [f64; 3]) -> Result<()> {
        if moment.iter().any(|component| !component.is_finite()) {
            return Err(TbError::InvalidAtomicMagneticMoment { moment });
        }
        self.magnetic_moment = Some(moment);
        Ok(())
    }

    /// Remove the magnetic moment, restoring the default nonmagnetic Atom.
    pub fn clear_magnetic_moment(&mut self) {
        self.magnetic_moment = None;
    }

    /// Whether an explicit magnetic moment is attached to this atom.
    pub fn has_magnetic_moment(&self) -> bool {
        self.magnetic_moment.is_some()
    }

    /// Change the chemical species of this atom.
    ///
    /// Only the species label changes; position, owned orbitals, and the
    /// optional magnetic moment are left untouched.  Symmetry analyses and
    /// `from_hr` species matching use the current species, so call this
    /// before running them.
    pub fn change_type(&mut self, new_type: AtomType) {
        self.name = new_type;
    }

    /// Construct an atom that currently owns no tight-binding orbitals.
    pub fn new(position: Array1<f64>, name: AtomType) -> Atom {
        Atom {
            position,
            name,
            orbitals: Vec::new(),
            magnetic_moment: None,
        }
    }

    /// Construct an atom with explicit model-local orbital IDs.
    pub fn with_orbitals(
        position: Array1<f64>,
        name: AtomType,
        orbitals: impl IntoIterator<Item = OrbitalId>,
    ) -> Atom {
        Atom {
            position,
            name,
            orbitals: orbitals.into_iter().collect(),
            magnetic_moment: None,
        }
    }

    pub(crate) fn set_orbitals(&mut self, orbitals: Vec<OrbitalId>) {
        self.orbitals = orbitals;
    }
}

impl fmt::Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Atom {{ name: {}, position: {:?}, orbitals: {:?}, magnetic moment:{:?}}}",
            self.name, self.position, self.orbitals, self.magnetic_moment
        )
    }
}
