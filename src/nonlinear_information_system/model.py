"""
model.py
--------
Core mathematical model: resonance function, effective friction, right-hand
side of the ODE system, and its analytical Jacobian.

State vector:  y = [phi, dphi, A, B]
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Resonance
# ---------------------------------------------------------------------------

def resonance(phi: float, cfg: dict[str, Any]) -> float:
    """Continuous phase-transition function around the critical threshold V_C.

    Parameters
    ----------
    phi:
        Current value of the field variable.
    cfg:
        Flat configuration dictionary (must contain ``SIGMA`` and ``V_C``).

    Returns
    -------
    float
        Value in ``[0, 1]``.
    """
    return 0.5 * (1.0 + np.tanh(cfg["SIGMA"] * (abs(phi) - cfg["V_C"])))


def dr_dphi(phi: float, cfg: dict[str, Any]) -> float:
    """Derivative of the resonance function with respect to *phi*.

    ``abs(phi)`` is not differentiable at ``phi = 0``; the sign convention
    ``np.sign(0) == 0`` is used there.

    Parameters
    ----------
    phi:
        Current value of the field variable.
    cfg:
        Flat configuration dictionary (must contain ``SIGMA`` and ``V_C``).

    Returns
    -------
    float
        Derivative value.
    """
    z = cfg["SIGMA"] * (abs(phi) - cfg["V_C"])
    if abs(z) > 350.0:
        sech2 = 0.0
    else:
        sech2 = 1.0 / np.cosh(z) ** 2
    return 0.5 * cfg["SIGMA"] * sech2 * np.sign(phi)


# ---------------------------------------------------------------------------
# Effective friction
# ---------------------------------------------------------------------------

def gamma_eff(A: float, B: float, cfg: dict[str, Any]) -> float:
    """Effective friction, decreasing with the structuration ``A * B``.

    Parameters
    ----------
    A, B:
        Structuration variables.
    cfg:
        Flat configuration dictionary (must contain ``GAMMA_0`` and ``KAPPA``).

    Returns
    -------
    float
        Positive friction coefficient.
    """
    return cfg["GAMMA_0"] * np.exp(-cfg["KAPPA"] * A * B)


def dgamma_dA(A: float, B: float, cfg: dict[str, Any]) -> float:
    """Partial derivative of :func:`gamma_eff` with respect to *A*."""
    return -cfg["KAPPA"] * B * gamma_eff(A, B, cfg)


def dgamma_dB(A: float, B: float, cfg: dict[str, Any]) -> float:
    """Partial derivative of :func:`gamma_eff` with respect to *B*."""
    return -cfg["KAPPA"] * A * gamma_eff(A, B, cfg)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def f(t: float, y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Right-hand side of the ODE system.

    The state vector is ``y = [phi, dphi, A, B]``.

    Parameters
    ----------
    t:
        Current time (unused; included for ``solve_ivp`` compatibility).
    y:
        State vector of length 4.
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    numpy.ndarray
        Time derivative ``[dphi, ddphi, dA, dB]``.
    """
    phi, dphi, A, B = y

    R = resonance(phi, cfg)
    g = gamma_eff(A, B, cfg)

    force = -cfg["LAMBDA"] * phi * (phi ** 2 - cfg["V"] ** 2) + cfg["XI"] * A * phi
    ddphi = force - g * dphi

    dA = cfg["MU"] * B * R * (1.0 - A) - cfg["DELTA"] * A
    dB = cfg["ETA"] * A * (1.0 - B) - cfg["THETA"] * B

    return np.array([dphi, ddphi, dA, dB], dtype=float)


# ---------------------------------------------------------------------------
# Analytical Jacobian
# ---------------------------------------------------------------------------

def jacobian(y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Analytical Jacobian of :func:`f` evaluated at *y*.

    Parameters
    ----------
    y:
        State vector ``[phi, dphi, A, B]``.
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    numpy.ndarray
        4×4 Jacobian matrix.
    """
    phi, dphi, A, B = y

    R = resonance(phi, cfg)
    Rphi = dr_dphi(phi, cfg)
    g = gamma_eff(A, B, cfg)
    gA = dgamma_dA(A, B, cfg)
    gB = dgamma_dB(A, B, cfg)

    J = np.zeros((4, 4), dtype=float)

    J[0, 1] = 1.0

    J[1, 0] = -cfg["LAMBDA"] * (3.0 * phi ** 2 - cfg["V"] ** 2) + cfg["XI"] * A
    J[1, 1] = -g
    J[1, 2] = cfg["XI"] * phi - gA * dphi
    J[1, 3] = -gB * dphi

    J[2, 0] = cfg["MU"] * B * (1.0 - A) * Rphi
    J[2, 2] = -cfg["MU"] * B * R - cfg["DELTA"]
    J[2, 3] = cfg["MU"] * R * (1.0 - A)

    J[3, 2] = cfg["ETA"] * (1.0 - B)
    J[3, 3] = -cfg["ETA"] * A - cfg["THETA"]

    return J
