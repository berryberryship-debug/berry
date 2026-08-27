"""
analysis.py
-----------
Fixed-point search, stability analysis, and eigenvalue summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import root

from .model import f, jacobian


# ---------------------------------------------------------------------------
# Fixed-point search
# ---------------------------------------------------------------------------

def find_fixed_point(x0: np.ndarray, cfg: dict[str, Any]):
    """Find a fixed point of the system by solving ``f(x) = 0``.

    Parameters
    ----------
    x0:
        Initial guess for the solver (length-4 array).
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Result object; the fixed point is in ``result.x`` when
        ``result.success`` is ``True``.
    """
    return root(lambda x: f(0.0, x, cfg), x0, method="hybr")


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def stability_report(
    x: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Compute local stability at a given state.

    Parameters
    ----------
    x:
        State vector ``[phi, dphi, A, B]``.
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    J : numpy.ndarray
        4×4 Jacobian matrix at *x*.
    eig : numpy.ndarray
        Eigenvalues of *J*.
    max_real : float
        Maximum real part of the eigenvalues.
    is_stable : bool
        ``True`` when all eigenvalues have strictly negative real parts.
    """
    J = jacobian(x, cfg)
    eig = np.linalg.eigvals(J)
    max_real = float(np.max(np.real(eig)))
    return J, eig, max_real, max_real < 0.0


def eigenvalue_summary(eig: np.ndarray) -> dict[str, Any]:
    """Build a human-readable summary of an eigenvalue array.

    Parameters
    ----------
    eig:
        Complex eigenvalue array.

    Returns
    -------
    dict
        Keys: ``real_parts``, ``imag_parts``, ``max_real``, ``is_stable``.
    """
    real = np.real(eig).tolist()
    imag = np.imag(eig).tolist()
    max_real = float(max(real))
    return {
        "real_parts": real,
        "imag_parts": imag,
        "max_real": max_real,
        "is_stable": max_real < 0.0,
    }
