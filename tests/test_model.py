"""Tests for nonlinear_information_system.model."""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_information_system.model import resonance, gamma_eff, f, jacobian

CFG = {
    "LAMBDA": 1.0,
    "V": 1.0,
    "V_C": 0.30,
    "MU": 1.8,
    "DELTA": 0.40,
    "ETA": 1.6,
    "THETA": 0.35,
    "GAMMA_0": 1.2,
    "XI": 0.25,
    "SIGMA": 15.0,
    "KAPPA": 5.0,
}


class TestResonance:
    @pytest.mark.parametrize("phi", [-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0])
    def test_range(self, phi):
        r = resonance(phi, CFG)
        assert 0.0 <= r <= 1.0, f"resonance({phi}) = {r} out of [0,1]"

    def test_symmetric(self):
        for phi in [0.1, 0.5, 1.5]:
            assert resonance(phi, CFG) == pytest.approx(resonance(-phi, CFG))

    def test_above_vc_closer_to_one(self):
        assert resonance(1.0, CFG) > 0.5

    def test_below_vc_closer_to_zero(self):
        assert resonance(0.0, CFG) < 0.5


class TestGammaEff:
    @pytest.mark.parametrize("A,B", [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (0.0, 1.0)])
    def test_positive(self, A, B):
        g = gamma_eff(A, B, CFG)
        assert g > 0.0, f"gamma_eff({A},{B}) = {g} not positive"

    def test_decreasing_with_AB(self):
        g0 = gamma_eff(0.0, 0.0, CFG)
        g1 = gamma_eff(0.5, 0.5, CFG)
        assert g1 < g0

    def test_max_at_zero(self):
        assert gamma_eff(0.0, 0.0, CFG) == pytest.approx(CFG["GAMMA_0"])


class TestRHS:
    def test_output_shape(self):
        y = np.array([0.5, 0.1, 0.3, 0.3])
        dy = f(0.0, y, CFG)
        assert dy.shape == (4,)

    def test_first_component_is_dphi(self):
        y = np.array([0.5, 0.1, 0.3, 0.3])
        dy = f(0.0, y, CFG)
        assert dy[0] == pytest.approx(y[1])
