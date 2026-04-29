use std::f64::consts::PI;
/// Elementary charge $e$, in Coulombs (C).
pub const Element_charge: f64 = 1.602176487e-19;
/// Reduced Planck constant $\hbar$, in $\text{J}\cdot\text{s}$.
pub const hbar: f64 = 1.054571628e-34;
/// Quantum of conductance $e^2/\hbar$, in $\Omega^{-1}$.
pub const Quantum_conductivity: f64 = Element_charge * Element_charge / hbar;
/// Electron rest mass $m_e$, in kg.
pub const mass_charge: f64 = 9.10938215e-31;
/// Bohr magneton $\mu_B = e\hbar/(2m_e)$, in J/T.
pub const mu_B: f64 = Element_charge * hbar / mass_charge / 2.0;
/// Magnetic flux quantum $\Phi_0 = h/(2e)$, in $\text{T}\cdot\text{m}^2$.
pub const phy_0: f64 = hbar / Element_charge;
