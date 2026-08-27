"""
simulation.py
-------------
Numerical integration of the ODE system using ``scipy.integrate.solve_ivp``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .model import f


def simulate(
    y0: list[float] | np.ndarray,
    cfg: dict[str, Any],
) -> "scipy.integrate.OdeResult":  # noqa: F821
    """Integrate the ODE system from ``t0`` to ``t1``.

    Parameters
    ----------
    y0:
        Initial state ``[phi, dphi, A, B]``.
    cfg:
        Flat configuration dictionary.  Must contain ``t0``, ``t1``,
        ``n_samples``, and all model parameters.

    Returns
    -------
    scipy.integrate.OdeResult
        Solution object.  Raises ``RuntimeError`` on integration failure.
    """
    t_eval = np.linspace(cfg["t0"], cfg["t1"], int(cfg["n_samples"]))
    sol = solve_ivp(
        lambda t, y: f(t, y, cfg),
        (cfg["t0"], cfg["t1"]),
        np.array(y0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol
