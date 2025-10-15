"""
Horndeski vs GR ANEC Comparison

Produces JSON comparing:
    - Pure GR ANEC (baseline)
    - Horndeski with Vainshtein screening
    
Tests whether screening suppresses negative energy.
"""

import numpy as np
import json
import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics.alcubierre_simple import alcubierre_metric
from src.metrics.natario_simple import natario_metric
from src.scalar_field.horndeski import HorndeskiField, HorndeskiParams
from src.anec.runner import create_test_rays, run_multimetric_anec_sweep


def create_rho_functions(
    v_s: float = 0.1,
    R: float = 1.0,
    sigma: float = 0.5,
    screening_enabled: bool = False,
    horndeski: HorndeskiField = None
) -> Dict:
    """
    Create energy density functions for different scenarios.
    
    Args:
        v_s: Warp velocity
        R: Bubble radius
        sigma: Wall thickness
        screening_enabled: Apply Vainshtein screening
        horndeski: Horndeski field (for screening calculation)
        
    Returns:
        Dict of scenario_name → rho_func
    """
    # Estimate warp stress-energy (from Phase A)
    rho_warp_base = 1e37  # J/m³ (conservative for v_s ~ 0.1)
    
    def rho_gr(t, x, y, z):
        """Pure GR energy density (constant for now)."""
        return np.ones_like(x) * rho_warp_base
    
    def rho_horndeski_screened(t, x, y, z):
        """Horndeski with Vainshtein screening."""
        if not screening_enabled or horndeski is None:
            return rho_gr(t, x, y, z)
        
        # Distance from bubble center
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Get screening data
        screening_data = horndeski.estimate_warp_bubble_screening(
            v_s, R, rho_warp_base
        )
        R_V = screening_data['R_V_m']
        
        # Base energy density
        rho_base = rho_gr(t, x, y, z)
        
        # Apply screening suppression to scalar contribution
        # Assume T_total = T_GR + T_scalar
        # T_scalar gets suppressed by ε²(r)
        epsilon = horndeski.screening_suppression(r, R_V)
        
        # For demo: assume 50% scalar contribution
        # T_total = 0.5 * T_GR + 0.5 * ε² * T_GR
        rho_total = 0.5 * rho_base + 0.5 * epsilon**2 * rho_base
        
        return rho_total
    
    return {
        'gr_baseline': rho_gr,
        'horndeski_screened': rho_horndeski_screened
    }


def run_horndeski_anec_comparison(
    output_file: str = 'results/horndeski_anec_sweep.json',
    n_rays: int = 9
):
    """
    Run Horndeski vs GR ANEC comparison.
    
    Args:
        output_file: JSON output path
        n_rays: Number of null rays
    """
    print("=" * 70)
    print("Horndeski vs GR ANEC Comparison")
    print("=" * 70)
    print()
    
    # Parameters
    v_s = 0.1
    R = 1.0
    sigma = 0.5
    
    print("Configuration:")
    print(f"  Warp velocity: v_s = {v_s:.2f} c")
    print(f"  Bubble radius: R = {R:.1f} m")
    print(f"  Wall thickness: σ = {sigma:.1f} m")
    print(f"  Number of rays: {n_rays}")
    print()
    
    # Create Horndeski field
    params = HorndeskiParams(Lambda_3=1e-3)
    horndeski = HorndeskiField(params)
    
    # Estimate screening
    rho_warp = 1e37
    screening_data = horndeski.estimate_warp_bubble_screening(v_s, R, rho_warp)
    
    print("Vainshtein Screening Estimate:")
    print(f"  R_V = {screening_data['R_V_m']:.2e} m")
    print(f"  R_V/R = {screening_data['R_V_over_R']:.2e}")
    print(f"  ε(R) = {screening_data['epsilon_wall']:.4f}")
    print(f"  ε²(R) = {screening_data['stress_suppression_wall']:.4e}")
    print(f"  Screening effective: {screening_data['screening_effective']}")
    print()
    
    # Create metrics
    def alc_metric(x, y, z, t):
        return alcubierre_metric(x, y, z, t, v_s=v_s, R=R, sigma=sigma)
    
    def nat_metric(x, y, z, t):
        return natario_metric(x, y, z, t, v_s=v_s, R=R, sigma=sigma)
    
    metrics = {
        'alcubierre_gr': alc_metric,
        'alcubierre_horndeski': alc_metric,  # Same metric, different rho
        'natario_gr': nat_metric,
        'natario_horndeski': nat_metric
    }
    
    # Create energy density functions
    rho_funcs_gr = create_rho_functions(
        v_s, R, sigma, screening_enabled=False, horndeski=None
    )
    rho_funcs_h = create_rho_functions(
        v_s, R, sigma, screening_enabled=True, horndeski=horndeski
    )
    
    rho_funcs = {
        'alcubierre_gr': rho_funcs_gr['gr_baseline'],
        'alcubierre_horndeski': rho_funcs_h['horndeski_screened'],
        'natario_gr': rho_funcs_gr['gr_baseline'],
        'natario_horndeski': rho_funcs_h['horndeski_screened']
    }
    
    # Create rays
    rays = create_test_rays(n_rays=n_rays)
    
    # Run ANEC sweep
    print(f"Running ANEC sweep ({n_rays} rays × 4 scenarios)...")
    print()
    results = run_multimetric_anec_sweep(
        metrics, rho_funcs, rays,
        lambda_max=10.0, n_points=50
    )
    
    # Add metadata
    results['screening_data'] = {
        k: (float(v) if isinstance(v, (np.float64, np.float32)) else v)
        for k, v in screening_data.items()
    }
    results['parameters'] = {
        'v_s': v_s,
        'R': R,
        'sigma': sigma,
        'Lambda_3': params.Lambda_3,
        'rho_warp_base': rho_warp
    }
    
    # Save JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for metric_name in ['alcubierre_gr', 'alcubierre_horndeski', 
                         'natario_gr', 'natario_horndeski']:
        if metric_name in results['metrics']:
            data = results['metrics'][metric_name]
            print(f"\n{metric_name}:")
            print(f"  Negative ANEC: {data['negative_count']}/{n_rays} " +
                  f"({data['negative_fraction']*100:.1f}%)")
            print(f"  ANEC median: {data['anec_median']:.2e} J")
    
    # Comparison
    if 'alcubierre_gr' in results['metrics'] and 'alcubierre_horndeski' in results['metrics']:
        gr_neg = results['metrics']['alcubierre_gr']['negative_fraction']
        h_neg = results['metrics']['alcubierre_horndeski']['negative_fraction']
        
        print()
        print("Screening Impact (Alcubierre):")
        print(f"  GR: {gr_neg*100:.1f}% negative")
        print(f"  Horndeski: {h_neg*100:.1f}% negative")
        
        if h_neg < gr_neg:
            reduction = (gr_neg - h_neg) / gr_neg * 100
            print(f"  → Reduction: {reduction:.1f}%")
            print("  ✓ Screening helps (but still violations?)")
        elif h_neg == gr_neg:
            print("  → No change")
            print("  ✗ Screening ineffective")
        else:
            print("  → Worse!")
            print("  ✗ Screening makes it worse (unexpected)")
    
    print()
    print("=" * 70)
    print("NOTE: This uses simplified ANEC runner (straight-line geodesics)")
    print("Actual values are order-of-magnitude estimates only.")
    print("=" * 70)


if __name__ == '__main__':
    run_horndeski_anec_comparison(
        output_file='results/horndeski_anec_sweep.json',
        n_rays=9
    )
