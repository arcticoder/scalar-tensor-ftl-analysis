"""
Demonstration: Brans-Dicke field coupled to Alcubierre metric.

Shows how scalar field φ(r_s) modifies effective gravitational coupling
near the warp bubble.
"""

import numpy as np
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scalar_field.brans_dicke import BransDickeField, BransDickeParams
from src.metrics.alcubierre_simple import alcubierre_metric, alcubierre_shape_function


def main():
    print("=" * 70)
    print("Brans-Dicke + Alcubierre Warp Bubble")
    print("=" * 70)
    print()
    
    # Parameters
    v_s = 0.1      # Warp velocity (c = 1)
    R = 1.0        # Bubble radius (m)
    sigma = 0.5    # Wall thickness (m)
    omega_bd = 50000.0  # BD coupling
    
    print("Parameters:")
    print(f"  Warp velocity: v_s = {v_s:.2f} c")
    print(f"  Bubble radius: R = {R:.1f} m")
    print(f"  Wall thickness: σ = {sigma:.1f} m")
    print(f"  BD coupling: ω = {omega_bd:.0f}")
    print()
    
    # Create BD field
    bd_field = BransDickeField(BransDickeParams(omega=omega_bd))
    
    # Radial profile (0 to 3R)
    r_s = np.linspace(0.01, 3.0 * R, 100)
    t = np.zeros_like(r_s)
    
    # Alcubierre shape function
    f, df, d2f = alcubierre_shape_function(r_s, R, sigma)
    
    print("Alcubierre Shape Function:")
    print(f"  f(0)     = {f[0]:.6f}")
    print(f"  f(R)     = {f[r_s.size // 3]:.6f}")
    print(f"  f(2R)    = {f[2 * r_s.size // 3]:.6f}")
    print(f"  f(3R)    = {f[-1]:.6f}")
    print()
    
    # BD scalar field (currently constant φ₀)
    phi = bd_field.phi(r_s, t)
    G_eff = bd_field.G_eff(r_s, t)
    
    print("Brans-Dicke Field:")
    print(f"  φ₀ = {bd_field.params.phi_0:.6f}")
    print(f"  φ(r_s) = const = {phi[0]:.6f}")
    print(f"  G_eff = G/φ = {G_eff[0]:.6e} m³/kg/s²")
    print(f"  G_eff/G = {G_eff[0] / bd_field.params.G:.6f}")
    print()
    
    # Post-Newtonian parameter
    gamma_pn = bd_field.post_newtonian_gamma()
    print("Solar System Constraints:")
    print(f"  γ_PPN = {gamma_pn:.10f}")
    print(f"  |γ - 1| = {abs(gamma_pn - 1.0):.2e}")
    print(f"  Cassini bound: |γ - 1| < 2.3×10⁻⁵")
    print(f"  Status: {'✓ PASS' if abs(gamma_pn - 1.0) < 2.3e-5 else '✗ FAIL'}")
    print()
    
    # Scalar stress-energy (should vanish for constant φ)
    T_scalar_tt, T_scalar_tr, T_scalar_rr, T_scalar_theta = \
        bd_field.scalar_stress_energy(r_s, t, theta=np.pi/2, phi_angle=0.0)
    
    print("Scalar Stress-Energy (φ = const):")
    print(f"  max|T^(scalar)_tt| = {np.max(np.abs(T_scalar_tt)):.2e} J/m³")
    print(f"  max|T^(scalar)_rr| = {np.max(np.abs(T_scalar_rr)):.2e} J/m³")
    print(f"  (All should be ~ 0 for constant field)")
    print()
    
    # Metric validation
    x = np.array([0.0, R, 2*R])
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t_pts = np.zeros_like(x)
    
    g, g_inv = alcubierre_metric(x, y, z, t_pts, v_s=v_s, R=R, sigma=sigma)
    
    print("Alcubierre Metric Validation:")
    for i in range(len(x)):
        product = g[:, :, i] @ g_inv[:, :, i]
        error = np.max(np.abs(product - np.eye(4)))
        print(f"  r_s = {x[i]:.1f} m: g·g⁻¹ error = {error:.2e}")
    print()
    
    print("=" * 70)
    print("Next Step: Compute ANEC with BD modifications")
    print("=" * 70)
    print()
    print("Expected result:")
    print("  - For φ = const, ANEC should match pure GR Alcubierre")
    print("  - Phase A result: Alcubierre has 0% ANEC violations")
    print("  - BD modification enters through G_eff in field equations")
    print("  - But constant φ doesn't change metric → same ANEC")
    print()
    print("To test BD effects, need dynamic φ(r_s) profile:")
    print("  - Solve □φ = source(T_matter)")
    print("  - Coupling to warp bubble stress-energy")
    print("  - Re-compute ANEC with modified field equations")


if __name__ == '__main__':
    main()
