"""Tests for nonlinear_information_system.model — Jacobian accuracy."""

from __future__ import annotations

import numpy as np
import pytest

from nonlinear_information_system.model import f, jacobian

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

_POINTS = [
    np.array([0.5, 0.1, 0.3, 0.3]),
    np.array([1.0, 0.0, 0.5, 0.5]),
    np.array([-0.8, -0.2, 0.1, 0.9]),
]


def _numerical_jacobian(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(0.0, y, CFG)
    for j in range(n):
        yp = y.copy()
        yp[j] += eps
        J[:, j] = (f(0.0, yp, CFG) - f0) / eps
    return J


@pytest.mark.parametrize("y", _POINTS)
def test_jacobian_vs_numerical(y):
    J_analytic = jacobian(y, CFG)
    J_numeric = _numerical_jacobian(y)
    np.testing.assert_allclose(
        J_analytic,
        J_numeric,
        rtol=1e-4,
        atol=1e-6,
        err_msg=f"Jacobian mismatch at y={y}",
    )
