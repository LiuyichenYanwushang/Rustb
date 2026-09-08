//! Physical constants.
//!
//! Names follow common physics notation (`hbar`, `mu_B`, ...) rather than
//! Rust's UPPER_CASE convention, so `non_upper_case_globals` is allowed here.
#![allow(non_upper_case_globals)]

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
pub const phy_0: f64 = std::f64::consts::PI * hbar / Element_charge;

#[cfg(test)]
mod tests {
    #[test]
    fn superconducting_flux_quantum_has_si_magnitude() {
        // h/(2e) ~= 2.067833848e-15 Wb. Existing constants use older CODATA
        // values, so this checks the physical scale without redefining them.
        assert!((super::phy_0 / 2.067833848e-15 - 1.0).abs() < 1e-6);
    }
}
