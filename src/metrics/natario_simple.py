"""
Simplified Natário metric for ANEC testing.

Flow-based warp drive with zero expansion.
"""

import numpy as np
from typing import Tuple


def natario_shape_function(
    r_s: np.ndarray,
    R: float = 1.0,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Natário shape function (same as Alcubierre).
    
    f(r_s) = 0.5 * [tanh((R + σ - r_s)/σ) - tanh((R - σ - r_s)/σ)]
    
    Args:
        r_s: Distance from bubble center (m)
        R: Bubble radius (m)
        sigma: Wall thickness (m)
        
    Returns:
        (f, df/dr_s)
    """
    arg_p = (R + sigma - r_s) / sigma
    arg_m = (R - sigma - r_s) / sigma
    
    tanh_p = np.tanh(arg_p)
    tanh_m = np.tanh(arg_m)
    f = 0.5 * (tanh_p - tanh_m)
    
    sech2_p = 1.0 - tanh_p**2
    sech2_m = 1.0 - tanh_m**2
    df = (-1.0 / (2.0 * sigma)) * (sech2_p - sech2_m)
    
    return f, df


def natario_metric(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    v_s: float = 0.1,
    R: float = 1.0,
    sigma: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Natário warp metric (flow-based, zero expansion).
    
    Metric:
        ds² = -dt² + (dx^i + v^i dt)(dx_i + v_i dt)
        
    Shift vector:
        v^i = -v_s f(r_s) n^i
        
    where n^i = x^i / r_s (radial unit vector)
    
    Components:
        g_tt = -1 + v²
        g_ti = v_i
        g_ij = δ_ij
        
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
    
    # Distance from bubble center
    r_s = np.sqrt(x**2 + y**2 + z**2)
    r_s_safe = np.where(r_s < 1e-10, 1e-10, r_s)
    
    # Shape function
    f, _ = natario_shape_function(r_s, R, sigma)
    
    # Shift vector: v^i = -v_s f n^i
    # In Cartesian: v^x = -v_s f (x/r_s), etc.
    v_x = -v_s * f * (x / r_s_safe)
    v_y = -v_s * f * (y / r_s_safe)
    v_z = -v_s * f * (z / r_s_safe)
    
    # v² = v_x² + v_y² + v_z²
    v_squared = v_x**2 + v_y**2 + v_z**2
    
    # Metric tensor
    g = np.zeros((4, 4) + shape)
    g[0, 0] = -1.0 + v_squared
    g[0, 1] = g[1, 0] = v_x
    g[0, 2] = g[2, 0] = v_y
    g[0, 3] = g[3, 0] = v_z
    g[1, 1] = 1.0
    g[2, 2] = 1.0
    g[3, 3] = 1.0
    
    # Inverse metric
    # For Natário: g^tt = -1, g^ti = v^i, g^ij = δ^ij - v^i v^j
    g_inv = np.zeros((4, 4) + shape)
    g_inv[0, 0] = -1.0
    g_inv[0, 1] = g_inv[1, 0] = v_x
    g_inv[0, 2] = g_inv[2, 0] = v_y
    g_inv[0, 3] = g_inv[3, 0] = v_z
    g_inv[1, 1] = 1.0 - v_x**2
    g_inv[2, 2] = 1.0 - v_y**2
    g_inv[3, 3] = 1.0 - v_z**2
    g_inv[1, 2] = g_inv[2, 1] = -v_x * v_y
    g_inv[1, 3] = g_inv[3, 1] = -v_x * v_z
    g_inv[2, 3] = g_inv[3, 2] = -v_y * v_z
    
    return g, g_inv


def validate_natario_metric():
    """Validate Natário metric implementation."""
    print("=== Natário Metric Validation ===\n")
    
    # Test points
    x = np.array([0.0, 0.5, 1.0, 2.0])
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    t = np.zeros_like(x)
    
    # Compute metric
    g, g_inv = natario_metric(x, y, z, t, v_s=0.1, R=1.0, sigma=0.5)
    
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
    
    print("=== END ===")


if __name__ == '__main__':
    validate_natario_metric()
