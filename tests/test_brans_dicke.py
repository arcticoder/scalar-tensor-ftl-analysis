"""
Tests for Brans-Dicke field implementation.

Validates:
    - Solar system constraints (ω > 40,000)
    - Post-Newtonian parameter γ
    - Scalar field equation residuals
    - Stress-energy tensor properties
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.scalar_field.brans_dicke import (
    BransDickeField,
    BransDickeParams,
    BransDickeWarpMetric
)


class TestBransDickeParams:
    """Test parameter validation."""
    
    def test_default_params(self):
        """Default parameters should satisfy Cassini constraint."""
        params = BransDickeParams()
        
        assert params.omega >= 40000.0, "Default ω violates Cassini bound"
        assert params.phi_0 == 1.0, "Background field should be normalized"
        assert params.G == pytest.approx(6.67430e-11, rel=1e-6)
        assert params.c == 299792458.0
    
    def test_custom_params(self):
        """Custom parameters should be settable."""
        params = BransDickeParams(omega=100000.0, phi_0=1.5)
        
        assert params.omega == 100000.0
        assert params.phi_0 == 1.5


class TestBransDickeField:
    """Test scalar field implementation."""
    
    @pytest.fixture
    def bd_field(self):
        """Standard BD field for testing."""
        return BransDickeField(BransDickeParams(omega=50000.0))
    
    def test_constant_field_profile(self, bd_field):
        """For initial implementation, φ = φ_0 everywhere."""
        r_s = np.linspace(0.1, 10.0, 100)
        t = np.zeros_like(r_s)
        
        phi = bd_field.phi(r_s, t)
        
        assert np.allclose(phi, bd_field.params.phi_0), "φ should be constant"
        assert phi.shape == r_s.shape, "Shape mismatch"
    
    def test_field_derivatives_zero(self, bd_field):
        """For constant profile, all derivatives should vanish."""
        r_s = np.linspace(0.1, 10.0, 100)
        t = np.zeros_like(r_s)
        
        dphi_dr = bd_field.d_phi_dr_s(r_s, t)
        dphi_dt = bd_field.d_phi_dt(r_s, t)
        
        assert np.allclose(dphi_dr, 0.0), "∂φ/∂r should be zero"
        assert np.allclose(dphi_dt, 0.0), "∂φ/∂t should be zero"
    
    def test_effective_G(self, bd_field):
        """G_eff = G/φ should equal G for φ = 1."""
        r_s = np.array([1.0, 5.0, 10.0])
        t = np.zeros_like(r_s)
        
        G_eff = bd_field.G_eff(r_s, t)
        
        expected = bd_field.params.G / bd_field.params.phi_0
        assert np.allclose(G_eff, expected), "G_eff = G/φ_0"
    
    def test_scalar_stress_energy_vanishes(self, bd_field):
        """For constant φ, scalar stress-energy should vanish."""
        r_s = np.array([1.0, 5.0, 10.0])
        t = np.zeros_like(r_s)
        
        T_tt, T_tr, T_rr, T_theta = bd_field.scalar_stress_energy(
            r_s, t, theta=np.pi/2, phi_angle=0.0
        )
        
        # All derivatives are zero → all components vanish
        assert np.allclose(T_tt, 0.0, atol=1e-20), "T_tt should vanish"
        assert np.allclose(T_tr, 0.0, atol=1e-20), "T_tr should vanish"
        assert np.allclose(T_rr, 0.0, atol=1e-20), "T_rr should vanish"
        assert np.allclose(T_theta, 0.0, atol=1e-20), "T_θθ should vanish"
    
    def test_box_phi_zero(self, bd_field):
        """□φ should be zero for constant field."""
        r_s = np.linspace(0.1, 10.0, 50)
        t = np.zeros_like(r_s)
        
        box_phi = bd_field.box_phi(r_s, t)
        
        assert np.allclose(box_phi, 0.0), "□φ = 0 for φ = const"
    
    def test_field_equation_residual_zero_matter(self, bd_field):
        """For T = 0 and φ = const, residual should be zero."""
        r_s = np.linspace(0.1, 10.0, 50)
        t = np.zeros_like(r_s)
        T_trace = np.zeros_like(r_s)
        
        residual = bd_field.field_equation_residual(r_s, t, T_trace)
        
        assert np.allclose(residual, 0.0), "Residual should vanish"
    
    def test_solar_system_constraint_pass(self):
        """ω = 50,000 should pass Cassini bound."""
        bd_field = BransDickeField(BransDickeParams(omega=50000.0))
        
        satisfied, omega = bd_field.solar_system_constraint()
        
        assert satisfied, "Should satisfy ω > 40,000"
        assert omega == 50000.0
    
    def test_solar_system_constraint_fail(self):
        """ω = 10,000 should fail Cassini bound."""
        bd_field = BransDickeField(BransDickeParams(omega=10000.0))
        
        satisfied, omega = bd_field.solar_system_constraint()
        
        assert not satisfied, "Should violate ω > 40,000"
        assert omega == 10000.0
    
    def test_post_newtonian_gamma(self):
        """γ should approach 1 for large ω."""
        # GR limit: ω → ∞, γ → 1
        bd_field_large = BransDickeField(BransDickeParams(omega=1e6))
        gamma_large = bd_field_large.post_newtonian_gamma()
        
        assert abs(gamma_large - 1.0) < 1e-6, "γ → 1 for ω → ∞"
        
        # ω = 50,000: γ = 50,001/50,002 ≈ 0.99998
        bd_field = BransDickeField(BransDickeParams(omega=50000.0))
        gamma = bd_field.post_newtonian_gamma()
        
        expected = 50001.0 / 50002.0
        assert abs(gamma - expected) < 1e-10, "γ = (1+ω)/(2+ω)"
        
        # Cassini constraint: |γ - 1| < 2.3×10⁻⁵
        assert abs(gamma - 1.0) < 2.3e-5, "Should satisfy Cassini γ bound"


class TestBransDickeWarpMetric:
    """Test BD-modified warp metric."""
    
    @pytest.fixture
    def mock_base_metric(self):
        """Mock Minkowski metric for testing."""
        def metric_func(x, y, z, t):
            # Return Minkowski η_μν
            shape = np.broadcast_shapes(
                np.asarray(x).shape,
                np.asarray(y).shape,
                np.asarray(z).shape,
                np.asarray(t).shape
            )
            
            g = np.zeros((4, 4) + shape)
            g_inv = np.zeros((4, 4) + shape)
            
            # η_μν = diag(-1, 1, 1, 1)
            g[0, 0] = -1.0
            g[1, 1] = 1.0
            g[2, 2] = 1.0
            g[3, 3] = 1.0
            
            g_inv[0, 0] = -1.0
            g_inv[1, 1] = 1.0
            g_inv[2, 2] = 1.0
            g_inv[3, 3] = 1.0
            
            return g, g_inv
        
        return metric_func
    
    def test_initialization(self, mock_base_metric):
        """BD warp metric should initialize correctly."""
        bd_metric = BransDickeWarpMetric(
            mock_base_metric,
            BransDickeParams(omega=50000.0)
        )
        
        assert bd_metric.bd_field.params.omega == 50000.0
    
    def test_metric_unchanged_constant_phi(self, mock_base_metric):
        """For φ = const, metric should equal base metric."""
        bd_metric = BransDickeWarpMetric(mock_base_metric)
        
        x, y, z, t = 1.0, 0.0, 0.0, 0.0
        g, g_inv = bd_metric.metric(x, y, z, t)
        g_base, g_inv_base = mock_base_metric(x, y, z, t)
        
        assert np.allclose(g, g_base), "Metric should match base"
        assert np.allclose(g_inv, g_inv_base), "Inverse should match base"
    
    def test_stress_energy_total(self, mock_base_metric):
        """Total stress-energy should include scalar contribution."""
        bd_metric = BransDickeWarpMetric(mock_base_metric)
        
        r_s = np.array([1.0])
        t = np.array([0.0])
        T_matter = np.zeros((4, 4, 1))
        T_matter[0, 0, 0] = 1e40  # Energy density
        
        T_total = bd_metric.stress_energy_total(
            r_s, t, theta=np.pi/2, phi_angle=0.0, T_matter=T_matter
        )
        
        # For constant φ, scalar contribution vanishes → T_total = T_matter
        assert np.allclose(T_total, T_matter), "T_total = T_matter + 0"
    
    def test_validation_constant_phi(self, mock_base_metric):
        """Validation should pass for constant φ with T = 0."""
        bd_metric = BransDickeWarpMetric(mock_base_metric)
        
        x, y, z, t = np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
        T_matter = np.zeros((4, 4, 1))
        
        validation = bd_metric.validate_field_equations(x, y, z, t, T_matter)
        
        assert validation['field_residual_max'] < 1e-15, "Residual should be ~ 0"
        assert validation['solar_system_ok'], "Should satisfy ω > 40,000"
        assert abs(validation['gamma_pn'] - 1.0) < 2.3e-5, "Should satisfy γ bound"


def test_integration():
    """Integration test: BD field + mock metric."""
    # Create BD field
    bd_field = BransDickeField(BransDickeParams(omega=50000.0))
    
    # Test points
    r_s = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    t = np.zeros_like(r_s)
    
    # Verify all properties
    phi = bd_field.phi(r_s, t)
    G_eff = bd_field.G_eff(r_s, t)
    gamma = bd_field.post_newtonian_gamma()
    
    assert np.all(phi > 0), "φ must be positive"
    assert np.all(G_eff > 0), "G_eff must be positive"
    assert abs(gamma - 1.0) < 2.3e-5, "γ within Cassini bounds"
    
    print("Integration test PASSED")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
