"""Tests for nonlinear_information_system.analysis."""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_information_system.model import f
from nonlinear_information_system.analysis import (
    find_fixed_point,
    stability_report,
    eigenvalue_summary,
)

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
