"""
Minimal ANEC (Averaged Null Energy Condition) Runner

Computes ∫ T_μν k^μ k^ν dλ along null geodesics.

For warp metric comparison:
    - Pure GR: T_μν from metric
    - Horndeski: T_μν + T^(screened)_scalar

Simplified from lqg-anec-framework Phase A implementation.
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Callable, Dict, List
import json


def compute_null_geodesic(
    metric_func: Callable,
    x0: np.ndarray,
    k0: np.ndarray,
    lambda_max: float = 10.0,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate null geodesic: dx^μ/dλ = k^μ.
    
    Simplified version (ignores Christoffel symbols for now).
    
    Args:
        metric_func: Function (x, y, z, t) → (g, g_inv)
        x0: Initial position [t, x, y, z]
        k0: Initial 4-momentum [k_t, k_x, k_y, k_z]
        lambda_max: Affine parameter range
        n_points: Number of integration points
        
    Returns:
        (lambda_grid, x_path) where x_path.shape = (n_points, 4)
    """
    # Simple straight-line approximation for initial test
    # (Full geodesic requires Christoffel symbols)
    lambda_grid = np.linspace(0, lambda_max, n_points)
    
    x_path = np.zeros((n_points, 4))
    for i, lam in enumerate(lambda_grid):
        x_path[i] = x0 + k0 * lam
    
    return lambda_grid, x_path


def compute_anec_integral(
    lambda_grid: np.ndarray,
    x_path: np.ndarray,
    k_path: np.ndarray,
    rho_func: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
) -> float:
    """
    Compute ANEC integral: ∫ T_μν k^μ k^ν dλ.
    
    For diagonal stress-energy: T_μν k^μ k^ν ≈ ρ k^t k^t
    
    Args:
        lambda_grid: Affine parameter values
        x_path: Geodesic path (n_points, 4)
        k_path: 4-momentum along path (n_points, 4)
        rho_func: Function (t, x, y, z) → ρ energy density
        
    Returns:
        ANEC integral value
    """
    # Extract coordinates
    t_vals = x_path[:, 0]
    x_vals = x_path[:, 1]
    y_vals = x_path[:, 2]
    z_vals = x_path[:, 3]
    
    # Compute energy density along path
    rho_vals = rho_func(t_vals, x_vals, y_vals, z_vals)
    
    # Simple approximation: T_μν k^μ k^ν ≈ ρ (for null vectors)
    # (Full calculation requires full stress-energy tensor)
    integrand = rho_vals
    
    # Integrate using trapezoidal rule
    anec = np.trapz(integrand, lambda_grid)
    
    return anec


def run_multimetric_anec_sweep(
    metrics: Dict[str, Callable],
    rho_funcs: Dict[str, Callable],
    rays: List[Tuple[np.ndarray, np.ndarray]],
    lambda_max: float = 10.0,
    n_points: int = 100
) -> Dict:
    """
    Run ANEC computation for multiple metrics and rays.
    
    Args:
        metrics: Dict of metric_name → metric_func(x,y,z,t)
        rho_funcs: Dict of metric_name → rho_func(t,x,y,z)
        rays: List of (x0, k0) initial conditions
        lambda_max: Geodesic affine parameter range
        n_points: Integration points
        
    Returns:
        Results dict with per-metric, per-ray ANEC values
    """
    results = {
        'config': {
            'n_rays': len(rays),
            'lambda_max': lambda_max,
            'n_points': n_points
        },
        'metrics': {}
    }
    
    for metric_name, metric_func in metrics.items():
        print(f"Processing {metric_name}...")
        
        rho_func = rho_funcs.get(metric_name)
        if rho_func is None:
            print(f"  Warning: No rho_func for {metric_name}, skipping")
            continue
        
        ray_results = []
        anec_values = []
        
        for ray_idx, (x0, k0) in enumerate(rays):
            # Integrate geodesic
            lambda_grid, x_path = compute_null_geodesic(
                metric_func, x0, k0, lambda_max, n_points
            )
            
            # k constant along affine parameter (simplified)
            k_path = np.tile(k0, (n_points, 1))
            
            # Compute ANEC
            anec = compute_anec_integral(lambda_grid, x_path, k_path, rho_func)
            
            ray_results.append({
                'ray_idx': ray_idx,
                'x0': x0.tolist(),
                'k0': k0.tolist(),
                'anec': float(anec)
            })
            anec_values.append(anec)
        
        # Statistics
        anec_array = np.array(anec_values)
        negative_count = np.sum(anec_array < 0)
        negative_fraction = negative_count / len(anec_array)
        
        results['metrics'][metric_name] = {
            'rays': ray_results,
            'anec_values': anec_values,
            'negative_count': int(negative_count),
            'negative_fraction': float(negative_fraction),
            'anec_min': float(np.min(anec_array)),
            'anec_max': float(np.max(anec_array)),
            'anec_median': float(np.median(anec_array))
        }
        
        print(f"  {metric_name}: {negative_count}/{len(rays)} negative " +
              f"({negative_fraction*100:.1f}%)")
    
    return results


def create_test_rays(n_rays: int = 9) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create test null rays passing through warp bubble.
    
    Args:
        n_rays: Number of rays
        
    Returns:
        List of (x0, k0) tuples
    """
    rays = []
    
    # Impact parameters (distance from center in y-direction)
    b_values = np.linspace(-2.0, 2.0, n_rays)
    
    for b in b_values:
        # Initial position: far left, offset by b in y
        x0 = np.array([0.0, -5.0, b, 0.0])  # [t, x, y, z]
        
        # Null momentum: k^μ = (1, 1, 0, 0) (rightward lightlike)
        # Normalized to k·k = 0 in Minkowski
        k0 = np.array([1.0, 1.0, 0.0, 0.0])
        
        rays.append((x0, k0))
    
    return rays


def demo_anec_runner():
    """Demonstrate minimal ANEC runner."""
    print("=" * 70)
    print("Minimal ANEC Runner Demo")
    print("=" * 70)
    print()
    
    # Import metric wrappers
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.metrics.alcubierre_simple import alcubierre_metric
    
    # Mock energy density function
    def rho_alcubierre(t, x, y, z):
        """Mock ρ ~ constant for now."""
        return np.ones_like(x) * 1e37  # J/m³
    
    # Create rays
    rays = create_test_rays(n_rays=9)
    print(f"Created {len(rays)} null rays")
    print()
    
    # Run ANEC sweep
    metrics = {
        'alcubierre_gr': alcubierre_metric
    }
    
    rho_funcs = {
        'alcubierre_gr': rho_alcubierre
    }
    
    results = run_multimetric_anec_sweep(
        metrics, rho_funcs, rays,
        lambda_max=10.0, n_points=50
    )
    
    # Display results
    print()
    print("Results Summary:")
    for metric_name, data in results['metrics'].items():
        print(f"\n{metric_name}:")
        print(f"  Negative fraction: {data['negative_fraction']*100:.1f}%")
        print(f"  ANEC range: [{data['anec_min']:.2e}, {data['anec_max']:.2e}]")
        print(f"  ANEC median: {data['anec_median']:.2e}")
    
    print()
    print("=" * 70)
    print("Note: This is a simplified runner (straight-line geodesics)")
    print("Full implementation requires Christoffel symbols + stress-energy")
    print("=" * 70)


if __name__ == '__main__':
    demo_anec_runner()
