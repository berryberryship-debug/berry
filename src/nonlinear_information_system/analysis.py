"""
analysis.py
-----------
Fixed-point search, stability analysis, eigenvalue summaries, and derived
thermodynamic-like quantities (extractable energy, cumulative dissipation,
topological orientation).

Notation
--------
All quantities are in the model's internal units.  They are derived directly
from the ODE equations — no external physical calibration is implied.

* Configurational potential::

    P(φ, A) = (LAMBDA/4)(φ² − V²)² − (XI·A/2)φ²

  This is the antiderivative of the conservative force term in the φ̈ equation
  (i.e. ``force = −∂P/∂φ``).

* Extractable energy (exergy analogue)::

    E_ext(y, ref) = max(0, P(ref) − P(y))

  Positive part of the configurational surplus over a reference state.
  Inspired by the thermodynamic exergy concept (useful work relative to a
  reference); *not* a literal energy in joules.

* Cumulative dissipation (Landauer/Prigogine analogue)::

    D = ∫ γ_eff(A(t), B(t)) · φ̇(t)² dt

  Work exported by the damping term along the trajectory.  Computed by
  trapezoidal integration over the discrete time grid.

* Topological orientation::

    τ = sign(φ_∞)

  Discrete label (+1 / −1 / 0) indicating which basin of the double-well
  φ settled into.  For a 0-dimensional oscillator this is a basin-choice
  variable, not a true spatial topological invariant.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import root

from .model import f, jacobian, gamma_eff


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


# ---------------------------------------------------------------------------
# Derived thermodynamic-like quantities
# ---------------------------------------------------------------------------

def configurational_potential(y: np.ndarray, cfg: dict[str, Any]) -> float:
    """Configurational potential at state *y*.

    Defined as the antiderivative of the conservative force in the φ̈ equation::

        P(φ, A) = (LAMBDA/4)(φ² − V²)² − (XI·A/2)φ²

    so that ``−∂P/∂φ = −LAMBDA·φ·(φ²−V²) + XI·A·φ``.

    Parameters
    ----------
    y:
        State vector ``[phi, dphi, A, B]``.
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    float
        Potential value in model-internal units.
    """
    phi, _dphi, A, _B = y
    return (
        cfg["LAMBDA"] / 4.0 * (phi ** 2 - cfg["V"] ** 2) ** 2
        - cfg["XI"] * A / 2.0 * phi ** 2
    )


def extractable_energy(
    y: np.ndarray,
    ref: np.ndarray,
    cfg: dict[str, Any],
) -> float:
    """Extractable (exergy-like) energy of state *y* relative to reference *ref*.

    Returns the positive part of the configurational surplus::

        E_ext = max(0, P(ref) − P(y))

    A positive value means *y* sits lower in the potential than *ref*, so the
    transition from *ref* to *y* could theoretically export work.  In model
    units only — no physical energy calibration.

    Parameters
    ----------
    y:
        Current state vector ``[phi, dphi, A, B]``.
    ref:
        Reference state vector (e.g. initial condition or fixed point).
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    float
        Non-negative extractable energy.
    """
    return max(0.0, configurational_potential(ref, cfg) - configurational_potential(y, cfg))


def cumulative_dissipation(sol: Any, cfg: dict[str, Any]) -> float:
    """Cumulative dissipation exported by friction along the trajectory.

    Computes::

        D = ∫ γ_eff(A(t), B(t)) · φ̇(t)² dt

    via trapezoidal integration on the discrete time grid stored in *sol*.
    Inspired by Landauer's principle (cost of irreversible operations) and
    Prigogine's dissipative structures; here it quantifies the total
    configurational work exported by the effective damping term.

    Parameters
    ----------
    sol:
        ODE solution object as returned by :func:`simulation.simulate`.
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    float
        Cumulative dissipation in model-internal units.
    """
    phi, dphi, A, B = sol.y  # each shape (n_samples,)
    integrand = np.array(
        [gamma_eff(a, b, cfg) * dph ** 2 for a, b, dph in zip(A, B, dphi)]
    )
    return float(np.trapezoid(integrand, sol.t))


def topological_orientation(y: np.ndarray) -> int:
    """Discrete basin label for the final state.

    Returns ``sign(φ)``: ``+1`` if φ > 0, ``−1`` if φ < 0, ``0`` if φ = 0.

    This indicates which well of the double-well potential the oscillator
    settled into.  It is a basin-choice variable for this 0-dimensional model,
    not a spatial topological invariant.

    Parameters
    ----------
    y:
        State vector ``[phi, dphi, A, B]``.

    Returns
    -------
    int
        +1, −1, or 0.
    """
    phi = y[0]
    return int(np.sign(phi))
