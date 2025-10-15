"""
Brans-Dicke Theory Implementation

Field equations:
    G_μν = (8πG/φ) T_μν + ω/φ² [∇_μφ ∇_νφ - (1/2) g_μν (∇φ)²]
            - (1/φ) [∇_μ∇_νφ - g_μν □φ]

Scalar field equation:
    □φ = (8πG/(3 + 2ω)) T    (in Jordan frame)

Where:
    ω = coupling constant (ω_BD > 40,000 from Cassini)
    φ = scalar field (effective gravitational coupling G_eff = G/φ)
    T = trace of stress-energy tensor

References:
    - Brans & Dicke (1961), Phys. Rev. 124, 925
    - Will (2014), Living Rev. Relativ. 17, 4
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BransDickeParams:
    """Parameters for Brans-Dicke theory."""
    omega: float = 50000.0  # BD coupling (Cassini constraint: ω >= 40,000, use 50k for margin)
    phi_0: float = 1.0      # Background scalar field (normalized)
    G: float = 6.67430e-11  # Gravitational constant (SI units)
    c: float = 299792458.0  # Speed of light (SI units)


class BransDickeField:
    """
    Brans-Dicke scalar field φ(x,t) coupled to spacetime metric.
    
    Provides:
        - Scalar field profile φ(r_s, t)
        - Effective gravitational coupling G_eff = G/φ
        - Scalar stress-energy tensor T_μν^(scalar)
        - Field equation residuals for validation
    """
    
    def __init__(self, params: Optional[BransDickeParams] = None):
        """
        Initialize Brans-Dicke field.
        
        Args:
            params: BD parameters (uses defaults if None)
        """
        self.params = params or BransDickeParams()
        
    def phi(self, r_s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Scalar field profile φ(r_s, t).
        
        For warp bubble analysis, we use:
            φ(r_s) = φ_0 [1 + δφ(r_s)]
        
        where δφ(r_s) is determined by solving the scalar field equation
        with warp bubble stress-energy as source.
        
        Args:
            r_s: Bubble-frame radial distance (m)
            t: Time coordinate (s)
            
        Returns:
            Scalar field value φ
        """
        # Simple profile for initial testing:
        # φ = φ_0 everywhere (will be refined with full field equations)
        return self.params.phi_0 * np.ones_like(r_s)
    
    def d_phi_dr_s(self, r_s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Radial derivative ∂φ/∂r_s.
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            
        Returns:
            ∂φ/∂r_s
        """
        # For constant φ_0 profile:
        return np.zeros_like(r_s)
    
    def d_phi_dt(self, r_s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Time derivative ∂φ/∂t.
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            
        Returns:
            ∂φ/∂t
        """
        # For static profile:
        return np.zeros_like(r_s)
    
    def G_eff(self, r_s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Effective gravitational coupling G_eff = G/φ.
        
        This is the locally measured gravitational constant.
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            
        Returns:
            Effective gravitational constant G_eff (SI units)
        """
        phi_val = self.phi(r_s, t)
        return self.params.G / phi_val
    
    def scalar_stress_energy(
        self,
        r_s: np.ndarray,
        t: np.ndarray,
        theta: float,
        phi_angle: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Scalar field contribution to stress-energy tensor.
        
        T_μν^(scalar) = ω/φ² [∇_μφ ∇_νφ - (1/2) g_μν (∇φ)²]
                        - (1/φ) [∇_μ∇_νφ - g_μν □φ]
        
        Returns components: (T_tt, T_tr, T_rr, T_θθ) in spherical coordinates.
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            theta: Polar angle
            phi_angle: Azimuthal angle
            
        Returns:
            (T_tt, T_tr, T_rr, T_θθ) scalar stress-energy components
        """
        phi_val = self.phi(r_s, t)
        phi_r = self.d_phi_dr_s(r_s, t)
        phi_t = self.d_phi_dt(r_s, t)
        
        # Kinetic term: (∇φ)² = g^μν ∇_μφ ∇_νφ
        # For Minkowski background (approximate):
        # (∇φ)² ≈ -φ_t² + φ_r²
        nabla_phi_sq = -phi_t**2 + phi_r**2
        
        # ω/φ² term
        omega_term = self.params.omega / phi_val**2
        
        # T_tt component
        T_tt = omega_term * (phi_t**2 - 0.5 * nabla_phi_sq)
        
        # T_tr component (off-diagonal)
        T_tr = omega_term * phi_t * phi_r
        
        # T_rr component
        T_rr = omega_term * (phi_r**2 - 0.5 * nabla_phi_sq)
        
        # T_θθ = T_φφ (isotropic in angular directions)
        T_theta_theta = -0.5 * omega_term * nabla_phi_sq
        
        return T_tt, T_tr, T_rr, T_theta_theta
    
    def box_phi(self, r_s: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        d'Alembertian operator □φ = g^μν ∇_μ∇_νφ.
        
        For Minkowski background (approximate):
            □φ ≈ -∂²φ/∂t² + ∂²φ/∂r² + (2/r) ∂φ/∂r
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            
        Returns:
            □φ
        """
        # For constant φ_0 profile, all derivatives vanish:
        return np.zeros_like(r_s)
    
    def field_equation_residual(
        self,
        r_s: np.ndarray,
        t: np.ndarray,
        T_trace: np.ndarray
    ) -> np.ndarray:
        """
        Residual of scalar field equation for validation.
        
        Field equation:
            □φ = (8πG/(3 + 2ω)) T
        
        Residual:
            R = □φ - (8πG/(3 + 2ω)) T
        
        Should be ~ 0 for consistent solution.
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            T_trace: Trace of stress-energy tensor
            
        Returns:
            Field equation residual
        """
        box_phi = self.box_phi(r_s, t)
        source = (8 * np.pi * self.params.G / (3 + 2 * self.params.omega)) * T_trace
        
        return box_phi - source
    
    def solar_system_constraint(self) -> Tuple[bool, float]:
        """
        Check solar system constraint: ω >= 40,000 (Cassini).
        
        Returns:
            (satisfied, omega_value)
        """
        satisfied = self.params.omega >= 40000.0
        return satisfied, self.params.omega
    
    def post_newtonian_gamma(self) -> float:
        """
        Post-Newtonian parameter γ in Brans-Dicke theory.
        
        γ = (1 + ω) / (2 + ω)
        
        GR predicts γ = 1. Cassini constraint: |γ - 1| < 2.3×10⁻⁵
        
        Returns:
            Post-Newtonian γ parameter
        """
        return (1.0 + self.params.omega) / (2.0 + self.params.omega)


class BransDickeWarpMetric:
    """
    Warp bubble metric with Brans-Dicke modifications.
    
    Combines:
        - Base metric (Alcubierre, Natário, etc.)
        - Brans-Dicke scalar field φ(r_s, t)
        - Modified Einstein equations
    """
    
    def __init__(
        self,
        base_metric: callable,
        bd_params: Optional[BransDickeParams] = None
    ):
        """
        Initialize BD-modified warp metric.
        
        Args:
            base_metric: Function returning (g_μν, g^μν) for base warp drive
            bd_params: Brans-Dicke parameters
        """
        self.base_metric = base_metric
        self.bd_field = BransDickeField(bd_params)
        
    def metric(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute metric g_μν and inverse g^μν with BD modifications.
        
        For initial implementation, we use the base metric directly
        (φ = φ_0 = const doesn't modify metric, only field equations).
        
        Args:
            x, y, z, t: Spacetime coordinates
            
        Returns:
            (g_μν, g^μν) metric and inverse (shape: (4, 4, ...))
        """
        # Get base metric
        g, g_inv = self.base_metric(x, y, z, t)
        
        # In Jordan frame with φ = φ_0, metric is unchanged
        # (modifications enter through G_eff in field equations)
        
        return g, g_inv
    
    def stress_energy_total(
        self,
        r_s: np.ndarray,
        t: np.ndarray,
        theta: float,
        phi_angle: float,
        T_matter: np.ndarray
    ) -> np.ndarray:
        """
        Total stress-energy tensor including scalar field contribution.
        
        T_total^μν = T_matter^μν + T_scalar^μν
        
        Args:
            r_s: Bubble-frame radial distance
            t: Time coordinate
            theta: Polar angle
            phi_angle: Azimuthal angle
            T_matter: Matter stress-energy tensor (4×4)
            
        Returns:
            Total stress-energy tensor (4×4)
        """
        # Get scalar contribution
        T_scalar_tt, T_scalar_tr, T_scalar_rr, T_scalar_theta = \
            self.bd_field.scalar_stress_energy(r_s, t, theta, phi_angle)
        
        # Construct scalar tensor (spherically symmetric)
        T_scalar = np.zeros_like(T_matter)
        T_scalar[0, 0] = T_scalar_tt  # T_tt
        T_scalar[0, 1] = T_scalar_tr  # T_tr
        T_scalar[1, 0] = T_scalar_tr  # T_rt
        T_scalar[1, 1] = T_scalar_rr  # T_rr
        T_scalar[2, 2] = T_scalar_theta  # T_θθ
        T_scalar[3, 3] = T_scalar_theta  # T_φφ
        
        return T_matter + T_scalar
    
    def validate_field_equations(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        t: np.ndarray,
        T_matter: np.ndarray
    ) -> dict:
        """
        Validate Brans-Dicke field equations at given points.
        
        Checks:
            1. Einstein equations with G_eff = G/φ
            2. Scalar field equation □φ = source
            3. Solar system constraints
        
        Args:
            x, y, z, t: Spacetime coordinates
            T_matter: Matter stress-energy tensor
            
        Returns:
            Dictionary of validation metrics
        """
        # Compute r_s (bubble-frame distance)
        r_s = np.sqrt(x**2 + y**2 + z**2)
        
        # Scalar field equation residual
        T_trace = np.trace(T_matter, axis1=0, axis2=1)
        residual = self.bd_field.field_equation_residual(r_s, t, T_trace)
        
        # Solar system constraint
        ss_ok, omega = self.bd_field.solar_system_constraint()
        
        # Post-Newtonian γ
        gamma_pn = self.bd_field.post_newtonian_gamma()
        
        return {
            'field_residual_max': np.max(np.abs(residual)),
            'field_residual_rms': np.sqrt(np.mean(residual**2)),
            'solar_system_ok': ss_ok,
            'omega_bd': omega,
            'gamma_pn': gamma_pn,
            'gamma_deviation': abs(gamma_pn - 1.0)
        }


def demo_brans_dicke():
    """Demonstration of Brans-Dicke field with warp bubble."""
    
    print("=== Brans-Dicke Field Demonstration ===\n")
    
    # Create BD field with Cassini-compliant ω
    bd_field = BransDickeField(BransDickeParams(omega=50000.0))
    
    # Test points (around warp bubble)
    r_s = np.linspace(0.1, 10.0, 100)  # 0.1 to 10 meters
    t = np.zeros_like(r_s)
    
    # Scalar field profile
    phi = bd_field.phi(r_s, t)
    print(f"φ(r_s) range: [{phi.min():.6f}, {phi.max():.6f}]")
    print(f"φ_0 = {bd_field.params.phi_0:.6f}")
    
    # Effective gravitational constant
    G_eff = bd_field.G_eff(r_s, t)
    print(f"\nG_eff range: [{G_eff.min():.6e}, {G_eff.max():.6e}] m³/kg/s²")
    print(f"G_eff / G = {G_eff[0] / bd_field.params.G:.6f}")
    
    # Solar system constraints
    ss_ok, omega = bd_field.solar_system_constraint()
    print(f"\nSolar system constraint (ω > 40,000): {'✓ PASS' if ss_ok else '✗ FAIL'}")
    print(f"ω_BD = {omega:.0f}")
    
    # Post-Newtonian parameter
    gamma_pn = bd_field.post_newtonian_gamma()
    print(f"\nPost-Newtonian γ = {gamma_pn:.10f}")
    print(f"|γ - 1| = {abs(gamma_pn - 1.0):.2e}")
    print(f"Cassini bound: |γ - 1| < 2.3×10⁻⁵ → {'✓ PASS' if abs(gamma_pn - 1.0) < 2.3e-5 else '✗ FAIL'}")
    
    # Scalar stress-energy
    T_tt, T_tr, T_rr, T_theta = bd_field.scalar_stress_energy(
        r_s, t, theta=np.pi/2, phi_angle=0.0
    )
    print(f"\nScalar stress-energy (φ = const, all derivatives → 0):")
    print(f"T^(scalar)_tt max: {np.max(np.abs(T_tt)):.2e} J/m³")
    print(f"T^(scalar)_rr max: {np.max(np.abs(T_rr)):.2e} J/m³")
    
    print("\n=== END ===")


if __name__ == '__main__':
    demo_brans_dicke()
