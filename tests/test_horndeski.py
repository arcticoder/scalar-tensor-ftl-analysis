"""
Tests for Horndeski theory and screening mechanisms.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scalar_field.horndeski import HorndeskiField, HorndeskiParams
from src.scalar_field.screening import VainshteinScreening, compare_screening_mechanisms


class TestHorndeskiParams:
    """Test Horndeski parameter initialization."""
    
    def test_default_params(self):
        """Default parameters should be sensible."""
        params = HorndeskiParams()
        
        assert params.c3 == 1.0
        assert params.Lambda_3 > 0
        assert params.M_pl > 0
        assert params.phi_0 == 1.0


class TestHorndeskiField:
    """Test Horndeski field implementation."""
    
    @pytest.fixture
    def horndeski(self):
        """Standard Horndeski field for testing."""
        return HorndeskiField()
    
    def test_kinetic_function(self, horndeski):
        """K(φ, X) = X for canonical kinetic term."""
        phi = np.array([1.0, 1.5, 2.0])
        X = np.array([0.1, 0.2, 0.3])
        
        K = horndeski.kinetic_function(phi, X)
        
        assert np.allclose(K, X), "K should equal X"
    
    def test_cubic_coupling(self, horndeski):
        """G3(φ, X) = c3 * φ * X."""
        phi = np.array([1.0, 2.0])
        X = np.array([0.5, 1.0])
        
        G3 = horndeski.cubic_coupling(phi, X)
        expected = horndeski.params.c3 * phi * X
        
        assert np.allclose(G3, expected)
    
    def test_vainshtein_radius_scaling(self, horndeski):
        """R_V should increase with source mass."""
        M1 = 1e10  # kg
        M2 = 1e15  # kg
        r_source = 1.0  # m
        
        R_V1 = horndeski.vainshtein_radius(M1, r_source)
        R_V2 = horndeski.vainshtein_radius(M2, r_source)
        
        assert R_V2 > R_V1, "Larger mass should give larger R_V"
        assert R_V1 > 0, "R_V must be positive"
    
    def test_screening_suppression(self, horndeski):
        """Suppression factor: ε = 1 outside R_V, (r/R_V)³ inside."""
        R_V = 10.0  # m
        
        # Test points
        r_outside = np.array([15.0, 20.0, 100.0])
        r_inside = np.array([1.0, 5.0, 9.0])
        
        epsilon_out = horndeski.screening_suppression(r_outside, R_V)
        epsilon_in = horndeski.screening_suppression(r_inside, R_V)
        
        # Outside: ε = 1
        assert np.allclose(epsilon_out, 1.0), "Outside R_V, ε = 1"
        
        # Inside: ε = (r/R_V)³ < 1
        assert np.all(epsilon_in < 1.0), "Inside R_V, ε < 1"
        assert np.all(epsilon_in > 0), "ε must be positive"
        
        # Check scaling
        expected_in = (r_inside / R_V)**3
        assert np.allclose(epsilon_in, expected_in)
    
    def test_warp_bubble_screening_estimate(self, horndeski):
        """Screening estimate for warp bubble should give sensible R_V."""
        v_s = 0.1
        R_bubble = 1.0
        rho_warp = 1e37  # J/m³
        
        screening_data = horndeski.estimate_warp_bubble_screening(
            v_s, R_bubble, rho_warp
        )
        
        assert 'R_V_m' in screening_data
        assert 'epsilon_wall' in screening_data
        assert 'screening_effective' in screening_data
        
        assert screening_data['R_V_m'] > 0, "R_V must be positive"
        assert 0 <= screening_data['epsilon_wall'] <= 1.0, "ε in [0,1]"


class TestVainshteinScreening:
    """Test Vainshtein screening implementation."""
    
    def test_screening_radius_positive(self):
        """R_V should be positive for any mass."""
        vain = VainshteinScreening(Lambda_3=1e-3)
        
        M_test = np.array([1e10, 1e20, 1e30])  # kg
        
        for M in M_test:
            R_V = vain.screening_radius(M)
            assert R_V > 0, f"R_V must be positive for M={M}"
    
    def test_suppression_factor_boundaries(self):
        """ε = 1 outside R_V, < 1 inside."""
        vain = VainshteinScreening()
        R_V = 10.0
        
        # At R_V
        epsilon_at_RV = vain.suppression_factor(np.array([R_V]), R_V)[0]
        assert 0.9 < epsilon_at_RV <= 1.0, "At R_V, ε ≈ 1"
        
        # Well inside
        epsilon_inside = vain.suppression_factor(np.array([R_V/10]), R_V)[0]
        assert epsilon_inside < 0.1, "Deep inside, ε << 1"
        
        # Well outside
        epsilon_outside = vain.suppression_factor(np.array([R_V*10]), R_V)[0]
        assert np.isclose(epsilon_outside, 1.0), "Far outside, ε = 1"
    
    def test_is_screened_threshold(self):
        """is_screened should respect threshold."""
        vain = VainshteinScreening()
        R_V = 10.0
        
        # Deep inside: screened
        assert vain.is_screened(1.0, R_V, threshold=0.1)
        
        # Far outside: not screened
        assert not vain.is_screened(100.0, R_V, threshold=0.1)


def test_screening_comparison():
    """Integration test: compare screening mechanisms."""
    M_source = 1e20  # kg (warp bubble equivalent)
    R_source = 1.0  # m
    rho_source = 1e37  # J/m³
    
    results = compare_screening_mechanisms(M_source, R_source, rho_source)
    
    assert 'vainshtein' in results
    assert 'chameleon' in results
    assert 'symmetron' in results
    
    # Vainshtein should have valid data
    v_res = results['vainshtein']
    assert v_res['R_V'] > 0
    assert 0 <= v_res['epsilon_at_R'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
