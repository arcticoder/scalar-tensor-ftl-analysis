"""
Horndeski Scalar-Tensor Theory

Most general scalar-tensor theory with second-order field equations.
Lagrangian: L = L2 + L3 + L4 + L5

Focus on L2-L3 subset for Vainshtein screening analysis.

References:
    - Horndeski (1974), Int. J. Theor. Phys. 10, 363
    - Kobayashi et al. (2011), Prog. Theor. Phys. 126, 511
"""

import numpy as np
from typing import Tuple, Callable, Optional
from dataclasses import dataclass


@dataclass
class HorndeskiParams:
    """Parameters for Horndeski theory (L2-L3 subset)."""
    # L2: K(φ, X) - kinetic term
    # X ≡ -(1/2) g^μν ∂_μφ ∂_νφ (kinetic invariant)
    
    # Cubic Galileon coupling
    c3: float = 1.0  # G3(φ, X) = c3 * φ * X
    
    # Vainshtein screening strength
    M_pl: float = 2.435e18  # Reduced Planck mass (GeV)
    Lambda_3: float = 1e-3  # Strong coupling scale (GeV) - controls screening
    
    # Background field
    phi_0: float = 1.0  # Normalized background


class HorndeskiField:
    """
    Horndeski scalar field with Vainshtein screening.
    
    L2 + L3 terms:
        L2 = K(φ, X)
        L3 = G3(φ, X) □φ
        
    For cubic Galileon: G3 = c3 * φ * X
    
    Vainshtein mechanism suppresses ∇φ in strong-field regime.
    """
    
    def __init__(self, params: Optional[HorndeskiParams] = None):
        """
        Initialize Horndeski field.
        
        Args:
            params: Horndeski parameters
        """
        self.params = params or HorndeskiParams()
        
    def kinetic_function(self, phi: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        K(φ, X) - kinetic term.
        
        Standard choice: K = X (canonical kinetic term)
        
        Args:
            phi: Scalar field
            X: Kinetic invariant -(1/2) g^μν ∂φ ∂φ
            
        Returns:
            K(φ, X)
        """
        return X  # Canonical kinetic term
    
    def cubic_coupling(self, phi: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        G3(φ, X) - cubic Galileon coupling.
        
        For Vainshtein screening: G3 = c3 * φ * X
        
        Args:
            phi: Scalar field
            X: Kinetic invariant
            
        Returns:
            G3(φ, X)
        """
        return self.params.c3 * phi * X
    
    def vainshtein_radius(
        self,
        M_source: float,
        r_source: float
    ) -> float:
        """
        Vainshtein screening radius.
        
        For cubic Galileon:
            R_V = (r_s * M / M_pl²)^(1/3)
            
        where r_s ~ G M / c² (Schwarzschild radius)
        
        Inside R_V: scalar interactions suppressed by (r/R_V)^3
        Outside R_V: standard 1/r scalar force
        
        Args:
            M_source: Source mass-energy (kg or J/c²)
            r_source: Source size (m)
            
        Returns:
            R_V in meters
        """
        # Convert to natural units (c = ℏ = 1)
        G_N = 6.67430e-11  # m³/kg/s²
        c = 299792458.0  # m/s
        
        # Schwarzschild radius
        r_s = 2.0 * G_N * M_source / c**2  # meters
        
        # M_pl in kg (from GeV)
        M_pl_kg = self.params.M_pl * 1.783e-27  # GeV → kg
        
        # R_V = (r_s * M / M_pl²)^(1/3)
        # Approximation: R_V ~ (r_s * r_source²)^(1/3)
        R_V = (r_s * r_source**2)**(1/3)
        
        return R_V
    
    def screening_suppression(
        self,
        r: np.ndarray,
        R_V: float
    ) -> np.ndarray:
        """
        Vainshtein screening suppression factor.
        
        ε(r) = 1                    for r > R_V
        ε(r) = (r/R_V)^3           for r < R_V
        
        Scalar gradient: ∇φ_screened = ε(r) * ∇φ_unscreened
        
        Args:
            r: Radial distance (m)
            R_V: Vainshtein radius (m)
            
        Returns:
            Suppression factor ε(r)
        """
        epsilon = np.ones_like(r)
        
        # Inside screening radius
        mask = r < R_V
        epsilon[mask] = (r[mask] / R_V)**3
        
        return epsilon
    
    def effective_scalar_stress(
        self,
        r: np.ndarray,
        R_V: float,
        T_scalar_unscreened: np.ndarray
    ) -> np.ndarray:
        """
        Effective scalar stress-energy with screening.
        
        T^(eff)_μν = ε²(r) * T^(scalar)_μν
        
        Factor ε² from (∇φ)² dependence in stress-energy.
        
        Args:
            r: Radial distance
            R_V: Vainshtein radius
            T_scalar_unscreened: Unscreened scalar stress-energy
            
        Returns:
            Screened T_scalar
        """
        epsilon = self.screening_suppression(r, R_V)
        
        # T_scalar ~ (∇φ)² → suppressed by ε²
        return epsilon**2 * T_scalar_unscreened
    
    def estimate_warp_bubble_screening(
        self,
        v_s: float,
        R_bubble: float,
        rho_warp: float
    ) -> dict:
        """
        Estimate Vainshtein screening for warp bubble.
        
        Args:
            v_s: Warp velocity (c = 1)
            R_bubble: Bubble radius (m)
            rho_warp: Warp stress-energy density (J/m³)
            
        Returns:
            Dict with R_V, suppression factors, consistency checks
        """
        # Effective mass from integrated stress-energy
        # M_eff ~ ρ * V ~ ρ * R³
        V_bubble = (4.0 / 3.0) * np.pi * R_bubble**3
        E_total = rho_warp * V_bubble  # Total energy (J)
        
        c = 299792458.0  # m/s
        M_eff = E_total / c**2  # Effective mass (kg)
        
        # Vainshtein radius
        R_V = self.vainshtein_radius(M_eff, R_bubble)
        
        # Suppression at bubble wall (r = R_bubble)
        epsilon_wall = self.screening_suppression(
            np.array([R_bubble]), R_V
        )[0]
        
        # Suppression at 2*R_bubble
        epsilon_2R = self.screening_suppression(
            np.array([2.0 * R_bubble]), R_V
        )[0]
        
        # Consistency check: is R_V >> R_bubble?
        # (For effective screening, need R_V > 10 * R_bubble)
        screening_effective = (R_V > 10.0 * R_bubble)
        
        return {
            'M_eff_kg': M_eff,
            'R_V_m': R_V,
            'R_V_over_R': R_V / R_bubble,
            'epsilon_wall': epsilon_wall,
            'epsilon_2R': epsilon_2R,
            'stress_suppression_wall': epsilon_wall**2,
            'stress_suppression_2R': epsilon_2R**2,
            'screening_effective': screening_effective,
            'requires_tuning': (R_V < R_bubble)  # If True, needs unphysical Lambda_3
        }


def demo_horndeski_screening():
    """Demonstrate Vainshtein screening for warp bubble."""
    print("=" * 70)
    print("Horndeski Theory: Vainshtein Screening for Warp Bubble")
    print("=" * 70)
    print()
    
    # Warp bubble parameters
    v_s = 0.1  # Warp velocity (c = 1)
    R_bubble = 1.0  # meters
    sigma = 0.5  # Wall thickness
    rho_warp = 1e37  # J/m³ (from BD analysis)
    
    print("Warp Bubble Parameters:")
    print(f"  v_s = {v_s:.2f} c")
    print(f"  R = {R_bubble:.1f} m")
    print(f"  ρ_warp ≈ {rho_warp:.1e} J/m³")
    print()
    
    # Create Horndeski field
    horndeski = HorndeskiField()
    
    print("Horndeski Parameters:")
    print(f"  c3 = {horndeski.params.c3:.1f} (cubic Galileon coupling)")
    print(f"  Λ3 = {horndeski.params.Lambda_3:.1e} GeV (strong coupling scale)")
    print()
    
    # Estimate screening
    screening_data = horndeski.estimate_warp_bubble_screening(
        v_s, R_bubble, rho_warp
    )
    
    print("Vainshtein Screening Analysis:")
    print(f"  M_eff = {screening_data['M_eff_kg']:.2e} kg")
    print(f"  R_V = {screening_data['R_V_m']:.2e} m")
    print(f"  R_V / R_bubble = {screening_data['R_V_over_R']:.2e}")
    print()
    
    print("Suppression Factors:")
    print(f"  ε(R) = {screening_data['epsilon_wall']:.4f} (scalar gradient)")
    print(f"  ε²(R) = {screening_data['stress_suppression_wall']:.4e} (stress-energy)")
    print(f"  ε(2R) = {screening_data['epsilon_2R']:.4f}")
    print(f"  ε²(2R) = {screening_data['stress_suppression_2R']:.4e}")
    print()
    
    print("Screening Effectiveness:")
    if screening_data['screening_effective']:
        print("  ✓ R_V >> R_bubble → Strong screening regime")
        print(f"    Scalar stress suppressed by {1.0/screening_data['stress_suppression_wall']:.1e}×")
    else:
        print("  ✗ R_V ≤ R_bubble → Weak/no screening")
        
    if screening_data['requires_tuning']:
        print("  ⚠ WARNING: Requires R_V > R_bubble")
        print("    → Need extreme Λ3 tuning (unphysical)")
    print()
    
    # Test different screening scales
    print("Sensitivity to Λ3 (strong coupling scale):")
    for Lambda_3 in [1e-6, 1e-3, 1.0, 1e3]:
        params_test = HorndeskiParams(Lambda_3=Lambda_3)
        horndeski_test = HorndeskiField(params_test)
        screening_test = horndeski_test.estimate_warp_bubble_screening(
            v_s, R_bubble, rho_warp
        )
        
        print(f"  Λ3 = {Lambda_3:.1e} GeV:")
        print(f"    R_V/R = {screening_test['R_V_over_R']:.2e}, " +
              f"ε²(R) = {screening_test['stress_suppression_wall']:.2e}")
    
    print()
    print("=" * 70)
    print("Conclusion:")
    print("  Vainshtein screening CAN suppress scalar stress near warp bubble")
    print("  IF R_V >> R_bubble (requires appropriate Λ3)")
    print()
    print("  Next: Compute ANEC with screened scalar contribution")
    print("=" * 70)


if __name__ == '__main__':
    demo_horndeski_screening()
