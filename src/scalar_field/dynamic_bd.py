"""
Dynamic Brans-Dicke field solver for warp bubble.

Solves scalar field equation:
    □φ = (8πG/(3 + 2ω)) T

where T = trace of stress-energy tensor from warp bubble.

For Alcubierre metric, we have significant T near bubble wall,
so φ will develop spatial profile φ(r_s).
"""

import numpy as np
from scipy.integrate import solve_bvp
from typing import Tuple, Callable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scalar_field.brans_dicke import BransDickeParams


class DynamicBDField:
    """
    Solve for BD scalar field profile in warp bubble background.
    
    Uses spherically symmetric approximation:
        □φ ≈ -∂²φ/∂t² + ∂²φ/∂r² + (2/r) ∂φ/∂r
        
    For static case (∂_t = 0):
        ∂²φ/∂r² + (2/r) ∂φ/∂r = (8πG/(3 + 2ω)) T(r)
    """
    
    def __init__(
        self,
        params: BransDickeParams,
        T_trace_func: Callable[[np.ndarray], np.ndarray]
    ):
        """
        Initialize dynamic BD field solver.
        
        Args:
            params: BD parameters
            T_trace_func: Function T(r_s) giving stress-energy trace
        """
        self.params = params
        self.T_trace = T_trace_func
        
        # Source coefficient
        self.alpha = (8.0 * np.pi * params.G) / (3.0 + 2.0 * params.omega)
        
        # Solution storage
        self.r_grid = None
        self.phi_solution = None
        self.dphi_dr_solution = None
        
    def solve_static_profile(
        self,
        r_min: float = 0.01,
        r_max: float = 10.0,
        n_points: int = 200,
        phi_boundary: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve static BD field equation using perturbative approach.
        
        For ω >> 1, the coupling α = 8πG/(3+2ω) is very small.
        Use φ = φ_0 + δφ with δφ << φ_0.
        
        Linearized equation:
            δφ'' + (2/r) δφ' ≈ α T(r)
            
        This is just Poisson equation in spherical symmetry.
        
        Args:
            r_min: Minimum radius (m)
            r_max: Maximum radius (m)
            n_points: Number of grid points
            phi_boundary: Background φ₀ value
            
        Returns:
            (r_grid, phi, dphi_dr)
        """
        # Grid
        r = np.linspace(r_min, r_max, n_points)
        
        # Perturbative solution: Green's function method
        # For equation: ∇²δφ = α T(r) in spherical symmetry
        # Solution: δφ(r) = -α/(4π) ∫ T(r')/|r-r'| d³r'
        #
        # For spherically symmetric T(r'):
        # δφ(r) = -α ∫₀^∞ r'² T(r') [min(r,r')/max(r,r')] dr'
        #       = -α [1/r ∫₀ʳ r'² T(r') dr' + ∫ᵣ^∞ r' T(r') dr']
        
        T = self.T_trace(r)
        
        # For simplicity, use finite domain approximation:
        # Assume T → 0 for r > r_max
        
        delta_phi = np.zeros_like(r)
        
        for i in range(len(r)):
            r_i = r[i]
            
            # Part 1: ∫₀^rᵢ r'² T(r')/rᵢ dr'
            if i > 0:
                r_inner = r[:i+1]
                T_inner = T[:i+1]
                integrand_inner = r_inner**2 * T_inner
                integral_inner = np.trapz(integrand_inner, r_inner) / r_i
            else:
                integral_inner = 0.0
            
            # Part 2: ∫_rᵢ^∞ r' T(r') dr'
            if i < len(r) - 1:
                r_outer = r[i:]
                T_outer = T[i:]
                integrand_outer = r_outer * T_outer
                integral_outer = np.trapz(integrand_outer, r_outer)
            else:
                integral_outer = 0.0
            
            delta_phi[i] = -self.alpha * (integral_inner + integral_outer)
        
        # Full solution
        phi = phi_boundary + delta_phi
        
        # Derivative: numerical gradient
        dphi_dr = np.gradient(phi, r)
        
        # Store solution
        self.r_grid = r
        self.phi_solution = phi
        self.dphi_dr_solution = dphi_dr
        
        return self.r_grid, self.phi_solution, self.dphi_dr_solution
    
    def evaluate(self, r_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate solved φ(r_s) and φ'(r_s).
        
        Args:
            r_s: Radial coordinates
            
        Returns:
            (φ, dφ/dr)
        """
        if self.phi_solution is None:
            raise RuntimeError("Must call solve_static_profile() first")
        
        # Interpolate solution
        phi = np.interp(r_s, self.r_grid, self.phi_solution)
        dphi_dr = np.interp(r_s, self.r_grid, self.dphi_dr_solution)
        
        return phi, dphi_dr
    
    def compute_residual(self, r_s: np.ndarray) -> np.ndarray:
        """
        Compute field equation residual for validation.
        
        R = φ'' + (2/r) φ' - α T(r)
        
        Should be ~ 0 for correct solution.
        
        Args:
            r_s: Radial coordinates
            
        Returns:
            Residual array
        """
        if self.dphi_dr_solution is None:
            raise RuntimeError("Must solve first")
        
        # Interpolate φ'
        dphi_dr = np.interp(r_s, self.r_grid, self.dphi_dr_solution)
        
        # Numerical derivative φ'' from φ'
        # Use finite differences on φ'
        dr = np.diff(self.r_grid)
        dphi_prime = np.diff(self.dphi_dr_solution)
        d2phi_dr2 = dphi_prime / dr
        
        # Extend to match grid size
        d2phi_dr2 = np.concatenate([[d2phi_dr2[0]], d2phi_dr2])
        
        # Interpolate to r_s
        d2phi = np.interp(r_s, self.r_grid, d2phi_dr2)
        
        # Source term
        T = self.T_trace(r_s)
        
        # Residual
        r_safe = np.where(r_s < 1e-10, 1e-10, r_s)
        residual = d2phi + (2.0 / r_safe) * dphi_dr - self.alpha * T
        
        return residual


def demo_dynamic_bd_field():
    """Demonstrate dynamic BD field solving."""
    print("=== Dynamic Brans-Dicke Field Solver ===\n")
    
    # Mock stress-energy trace (realistic scale for v_s ~ 0.1c warp)
    def T_trace_gaussian(r):
        """
        Mock T(r) with realistic amplitude.
        
        For Alcubierre at v_s = 0.1c:
            ρ ~ 10³⁷-10³⁸ J/m³ (from Phase A)
            
        Use conservative estimate.
        """
        R = 1.0  # Bubble radius (m)
        sigma = 0.5  # Width (m)
        amplitude = 1e37  # J/m³ (reduced from 1e40)
        
        return amplitude * np.exp(-((r - R) ** 2) / (2 * sigma**2))
    
    # Create solver
    params = BransDickeParams(omega=50000.0)
    solver = DynamicBDField(params, T_trace_gaussian)
    
    print(f"Parameters:")
    print(f"  ω_BD = {params.omega:.0f}")
    print(f"  α = 8πG/(3+2ω) = {solver.alpha:.6e}")
    print()
    
    # Solve
    print("Solving static BD field equation...")
    r, phi, dphi_dr = solver.solve_static_profile(
        r_min=0.01,
        r_max=5.0,
        n_points=200,
        phi_boundary=1.0
    )
    print(f"  Grid: {len(r)} points from {r[0]:.2f} to {r[-1]:.2f} m")
    print()
    
    # Solution statistics
    print("Solution:")
    print(f"  φ(0.01 m) = {phi[0]:.6f}")
    print(f"  φ(R=1 m)  = {phi[np.argmin(np.abs(r - 1.0))]:.6f}")
    print(f"  φ(5 m)    = {phi[-1]:.6f}")
    print(f"  φ range: [{phi.min():.6f}, {phi.max():.6f}]")
    print(f"  φ' range: [{dphi_dr.min():.6e}, {dphi_dr.max():.6e}]")
    print()
    
    # Validation: residual
    r_test = np.linspace(0.1, 4.0, 50)
    residual = solver.compute_residual(r_test)
    
    print("Field equation residual:")
    print(f"  max|R| = {np.max(np.abs(residual)):.2e}")
    print(f"  rms(R) = {np.sqrt(np.mean(residual**2)):.2e}")
    print(f"  Status: {'✓ PASS' if np.max(np.abs(residual)) < 1e-6 else '✗ FAIL'}")
    print()
    
    # G_eff variation
    G_eff = params.G / phi
    print("Effective gravitational constant:")
    print(f"  G_eff(center) = {G_eff[0]:.6e} m³/kg/s²")
    print(f"  G_eff(R)      = {G_eff[np.argmin(np.abs(r - 1.0))]:.6e} m³/kg/s²")
    print(f"  G_eff(∞)      = {G_eff[-1]:.6e} m³/kg/s²")
    print(f"  Variation: {(G_eff.max() - G_eff.min()) / params.G * 100:.3f}%")
    print()
    
    print("=== END ===")


if __name__ == '__main__':
    demo_dynamic_bd_field()
