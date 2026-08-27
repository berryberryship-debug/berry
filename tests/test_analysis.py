"""Tests for nonlinear_information_system.analysis."""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_information_system.model import f
from nonlinear_information_system.analysis import (
    find_fixed_point,
    stability_report,
    eigenvalue_summary,
    configurational_potential,
    extractable_energy,
    cumulative_dissipation,
    topological_orientation,
)
from nonlinear_information_system.simulation import simulate

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


class TestFindFixedPoint:
    def test_residual_small(self):
        x0 = np.array([0.05, 0.0, 0.02, 0.02])
        result = find_fixed_point(x0, CFG)
        residual = np.linalg.norm(f(0.0, result.x, CFG))
        assert residual < 1e-8, f"|f(x*)| = {residual}"


class TestStabilityReport:
    def test_return_types(self):
        y = np.array([1.0, 0.0, 0.5, 0.5])
        J, eig, max_real, is_stable = stability_report(y, CFG)
        assert J.shape == (4, 4)
        assert len(eig) == 4
        assert isinstance(max_real, float)
        assert isinstance(is_stable, (bool, np.bool_))

    def test_max_real_consistency(self):
        y = np.array([1.0, 0.0, 0.5, 0.5])
        _, eig, max_real, is_stable = stability_report(y, CFG)
        assert max_real == pytest.approx(float(np.max(np.real(eig))))
        assert is_stable == (max_real < 0.0)


class TestEigenvalueSummary:
    def test_keys(self):
        eig = np.array([-1.0 + 0.5j, -0.3 - 0.2j, -2.0 + 0j, -0.1 + 0j])
        summary = eigenvalue_summary(eig)
        assert set(summary.keys()) == {"real_parts", "imag_parts", "max_real", "is_stable"}

    def test_stable_system(self):
        eig = np.array([-1.0, -2.0, -0.5, -0.1])
        summary = eigenvalue_summary(eig)
        assert summary["is_stable"] is True
        assert summary["max_real"] == pytest.approx(-0.1)

    def test_unstable_system(self):
        eig = np.array([-1.0, 0.01, -0.5, -0.1])
        summary = eigenvalue_summary(eig)
        assert summary["is_stable"] is False


CFG_SIM = {**CFG, "t0": 0.0, "t1": 10.0, "n_samples": 200, "y0": [0.05, 0.0, 0.02, 0.02]}


class TestConfigurationalPotential:
    def test_returns_float(self):
        y = np.array([1.0, 0.0, 0.5, 0.5])
        P = configurational_potential(y, CFG)
        assert isinstance(P, float)

    def test_zero_field(self):
        y = np.array([0.0, 0.0, 0.5, 0.5])
        P = configurational_potential(y, CFG)
        assert P == pytest.approx(CFG["LAMBDA"] / 4.0 * CFG["V"] ** 4)

    def test_at_minimum_lower_than_at_zero(self):
        y_min = np.array([CFG["V"], 0.0, 0.0, 0.0])
        y_zero = np.array([0.0, 0.0, 0.0, 0.0])
        assert configurational_potential(y_min, CFG) < configurational_potential(y_zero, CFG)


class TestExtractableEnergy:
    def test_non_negative(self):
        y = np.array([1.0, 0.0, 0.5, 0.5])
        ref = np.array([0.0, 0.0, 0.0, 0.0])
        assert extractable_energy(y, ref, CFG) >= 0.0

    def test_lower_potential_yields_positive(self):
        ref = np.array([0.0, 0.0, 0.0, 0.0])
        y_low = np.array([CFG["V"], 0.0, 0.0, 0.0])
        assert extractable_energy(y_low, ref, CFG) > 0.0

    def test_higher_potential_yields_zero(self):
        y_high = np.array([0.0, 0.0, 0.0, 0.0])
        ref = np.array([CFG["V"], 0.0, 0.0, 0.0])
        assert extractable_energy(y_high, ref, CFG) == 0.0


class TestCumulativeDissipation:
    def test_non_negative(self):
        sol = simulate(CFG_SIM["y0"], CFG_SIM)
        D = cumulative_dissipation(sol, CFG_SIM)
        assert D >= 0.0

    def test_increases_with_integration_time(self):
        cfg_short = {**CFG_SIM, "t1": 5.0, "n_samples": 100}
        cfg_long = {**CFG_SIM, "t1": 10.0, "n_samples": 200}
        D_short = cumulative_dissipation(simulate(cfg_short["y0"], cfg_short), cfg_short)
        D_long = cumulative_dissipation(simulate(cfg_long["y0"], cfg_long), cfg_long)
        assert D_long >= D_short


class TestTopologicalOrientation:
    @pytest.mark.parametrize("phi,expected", [(1.0, 1), (-1.0, -1), (0.0, 0)])
    def test_sign(self, phi, expected):
        y = np.array([phi, 0.0, 0.5, 0.5])
        assert topological_orientation(y) == expected

    def test_settled_trajectory(self):
        sol = simulate(CFG_SIM["y0"], CFG_SIM)
        yT = sol.y[:, -1]
        orient = topological_orientation(yT)
        assert orient in {-1, 0, 1}
