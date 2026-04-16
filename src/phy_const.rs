use std::f64::consts::PI;
///元电荷 $e$, 单位 C
pub const Element_charge: f64 = 1.602176487e-19;
///约化普朗克常数 $\hbar$, 单位 $\text{J}\cdot \text{s}$
pub const hbar: f64 = 1.054571628e-34;
///量子霍尔电导 $e^2/\hbar$, 单位$\Omega^{-1}$
pub const Quantum_conductivity: f64 = Element_charge * Element_charge / hbar;
///电子质量 $m_e$, 单位 Kg
pub const mass_charge: f64 = 9.10938215e-31;
///波尔磁子 $e\hbar/m_e$, 单位 J/T
pub const mu_B: f64 = Element_charge * hbar / mass_charge / 2.0;
///磁通量子 $\hbar/e$, 单位 $\text{T}\cdot\text{m}^2$
pub const phy_0: f64 = hbar / Element_charge;
