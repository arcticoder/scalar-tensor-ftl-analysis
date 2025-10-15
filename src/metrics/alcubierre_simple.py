"""
Simplified Alcubierre metric for BD-ANEC testing.

Adapted from lqg-macroscopic-coherence Phase A implementation.
"""

import numpy as np
from typing import Tuple


def alcubierre_shape_function(
    r_s: np.ndarray,
    R: float = 1.0,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Alcubierre shape function with derivatives.
    
    f(r_s) = 0.5 * [tanh((R + σ - r_s)/σ) - tanh((R - σ - r_s)/σ)]
    
    Args:
        r_s: Distance from bubble center (m)
        R: Bubble radius (m)
        sigma: Wall thickness (m)
        
    Returns:
        (f, df/dr_s, d²f/dr_s²)
    """
    arg_p = (R + sigma - r_s) / sigma
    arg_m = (R - sigma - r_s) / sigma
    
    tanh_p = np.tanh(arg_p)
    tanh_m = np.tanh(arg_m)
    f = 0.5 * (tanh_p - tanh_m)
    
    sech2_p = 1.0 - tanh_p**2
    sech2_m = 1.0 - tanh_m**2
    df = (-1.0 / (2.0 * sigma)) * (sech2_p - sech2_m)
    
    dsech2_p = (2.0 / sigma) * sech2_p * tanh_p
    dsech2_m = (2.0 / sigma) * sech2_m * tanh_m
    d2f = (-1.0 / (2.0 * sigma)) * (dsech2_p - dsech2_m)
    
    return f, df, d2f


def alcubierre_metric(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    v_s: float = 0.1,
    R: float = 1.0,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alcubierre warp metric and inverse.
    
    Metric:
        ds² = -dt² + [dx - v_s f(r_s) dt]² + dy² + dz²
        
    Components:
        g_tt = -1 + (v_s f)²
        g_tx = -v_s f
        g_xx = g_yy = g_zz = 1
        
    Args:
        x, y, z, t: Spacetime coordinates (m, s)
        v_s: Warp velocity (c = 1)
        R: Bubble radius (m)
        sigma: Wall thickness (m)
        
    Returns:
        (g_μν, g^μν) with shape (4, 4, *broadcast_shape)
    """
    # Broadcast shapes
    shape = np.broadcast_shapes(
        np.asarray(x).shape,
        np.asarray(y).shape,
        np.asarray(z).shape,
        np.asarray(t).shape
    )
    
    # Convert to arrays
    x = np.broadcast_to(np.asarray(x), shape)
    y = np.broadcast_to(np.asarray(y), shape)
    z = np.broadcast_to(np.asarray(z), shape)
    
    # Distance from bubble center (at origin)
    r_s = np.sqrt(x**2 + y**2 + z**2)
    
    # Shape function
    f, _, _ = alcubierre_shape_function(r_s, R, sigma)
    v_f = v_s * f
    
    # Metric tensor
    g = np.zeros((4, 4) + shape)
    g[0, 0] = -1.0 + v_f**2
    g[0, 1] = g[1, 0] = -v_f
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    # Inverse metric (analytic)
    g_inv = np.zeros((4, 4) + shape)
    g_inv[0, 0] = -1.0
    g_inv[0, 1] = g_inv[1, 0] = -v_f
    g_inv[1, 1] = 1.0 - v_f**2
    g_inv[2, 2] = 1.0
    g_inv[3, 3] = 1.0
    
    return g, g_inv


def validate_alcubierre_metric():
    """Validate Alcubierre metric implementation."""
    print("=== Alcubierre Metric Validation ===\n")
    
    # Test points
    x = np.array([0.0, 0.5, 1.0, 2.0])
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = np.zeros_like(x)
    
    # Compute metric
    g, g_inv = alcubierre_metric(x, y, z, t, v_s=0.1, R=1.0, sigma=0.5)
    
    # Check g · g⁻¹ = I
    identity_errors = []
    for i in range(len(x)):
        g_i = g[:, :, i]
        g_inv_i = g_inv[:, :, i]
        product = g_i @ g_inv_i
        identity = np.eye(4)
        error = np.max(np.abs(product - identity))
        identity_errors.append(error)
        
    print(f"g · g⁻¹ = I errors:")
    for i, (x_i, err) in enumerate(zip(x, identity_errors)):
        print(f"  x = {x_i:.1f} m: {err:.2e}")
    
    max_error = max(identity_errors)
    print(f"\nMax error: {max_error:.2e}")
    
    if max_error < 1e-14:
        print("✓ PASS: Inverse metric is correct\n")
    else:
        print("✗ FAIL: Inverse metric has errors\n")
    
    # Check shape function at r_s = 0, R, 2R
    r_test = np.array([0.0, 1.0, 2.0])
    f, df, d2f = alcubierre_shape_function(r_test, R=1.0, sigma=0.5)
    
    print("Shape function values:")
    print(f"  f(0)  = {f[0]:.6f} (expect ~1.0)")
    print(f"  f(R)  = {f[1]:.6f} (expect ~0.5)")
    print(f"  f(2R) = {f[2]:.6f} (expect ~0.0)")
    
    print("\n=== END ===")


if __name__ == '__main__':
    validate_alcubierre_metric()
