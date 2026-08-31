#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nonlinear_system_all_in_one.py
==============================
Single self-contained file that combines every module of the
nonlinear_information_system package.

No package installation required.  Run directly::

    python nonlinear_system_all_in_one.py
    python nonlinear_system_all_in_one.py --config path/to/config.yaml

Dependencies (install once)::

    pip install numpy scipy pandas matplotlib PyYAML

Sections
--------
1. CONFIGURATION     – default parameters + YAML loader
2. MODEL             – resonance, gamma_eff, ODE right-hand side, Jacobian
3. ANALYSIS          – fixed-point, stability, extractable energy,
                       cumulative dissipation, topological orientation
4. SIMULATION        – solve_ivp wrapper
5. EXPORT            – CSV trajectory + JSON metadata
6. PLOTTING          – phase portraits + time series
7. CLI / MAIN        – orchestration + argparse entry point
"""

from __future__ import annotations

# ============================================================================
# Standard-library & third-party imports
# ============================================================================

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import root


# ============================================================================
# 1.  CONFIGURATION
# ============================================================================

# Default parameters — identical to config/default.yaml.
# Override any value by passing a YAML file via --config.
DEFAULT_CFG: dict[str, Any] = {
    # --- model ---
    "LAMBDA":  1.0,
    "V":       1.0,
    "V_C":     0.30,
    "MU":      1.8,
    "DELTA":   0.40,
    "ETA":     1.6,
    "THETA":   0.35,
    "GAMMA_0": 1.2,
    "XI":      0.25,
    "SIGMA":   15.0,
    "KAPPA":   5.0,
    # --- simulation ---
    "t0":       0.0,
    "t1":       100.0,
    "n_samples": 4000,
    "y0":       [0.05, 0.0, 0.02, 0.02],
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and return a flat dictionary.

    If *path* is None the built-in :data:`DEFAULT_CFG` is returned directly
    (no file I/O needed).

    Parameters
    ----------
    path:
        Path to a YAML file with ``model:`` and ``simulation:`` sections,
        or None to use the built-in defaults.

    Returns
    -------
    dict
        Flat configuration dictionary.
    """
    if path is None:
        return dict(DEFAULT_CFG)

    try:
        import yaml  # optional — only needed when a YAML file is provided
    except ImportError as exc:
        raise ImportError("PyYAML is required to load a config file: pip install PyYAML") from exc

    with Path(path).open(encoding="utf-8") as fh:
        nested = yaml.safe_load(fh)

    flat: dict[str, Any] = {}
    for section in nested.values():
        if isinstance(section, dict):
            flat.update(section)
    return flat


# ============================================================================
# 2.  MODEL
# ============================================================================
# State vector:  y = [phi, dphi, A, B]
#
# ODE system:
#   φ̈  = -Λ·φ·(φ²-V²) + Ξ·A·φ  -  γ_eff(A,B)·φ̇
#   Ȧ  = μ·B·R(φ)·(1-A)  -  δ·A
#   Ḃ  = η·A·(1-B)  -  θ·B


def resonance(phi: float, cfg: dict[str, Any]) -> float:
    """Continuous phase-transition function R(φ) ∈ [0, 1].

    R(φ) = 0.5 · (1 + tanh(SIGMA · (|φ| − V_C)))
    """
    return 0.5 * (1.0 + np.tanh(cfg["SIGMA"] * (abs(phi) - cfg["V_C"])))


def dr_dphi(phi: float, cfg: dict[str, Any]) -> float:
    """Derivative dR/dφ.

    Uses np.sign(phi) at φ=0 (sub-derivative convention).
    """
    z = cfg["SIGMA"] * (abs(phi) - cfg["V_C"])
    sech2 = 0.0 if abs(z) > 350.0 else 1.0 / np.cosh(z) ** 2
    return 0.5 * cfg["SIGMA"] * sech2 * np.sign(phi)


def gamma_eff(A: float, B: float, cfg: dict[str, Any]) -> float:
    """Effective friction: γ_eff = GAMMA_0 · exp(−KAPPA·A·B).

    Decreases as structuration A·B grows.  Always strictly positive.
    """
    return cfg["GAMMA_0"] * np.exp(-cfg["KAPPA"] * A * B)


def dgamma_dA(A: float, B: float, cfg: dict[str, Any]) -> float:
    """∂γ_eff/∂A = −KAPPA · B · γ_eff."""
    return -cfg["KAPPA"] * B * gamma_eff(A, B, cfg)


def dgamma_dB(A: float, B: float, cfg: dict[str, Any]) -> float:
    """∂γ_eff/∂B = −KAPPA · A · γ_eff."""
    return -cfg["KAPPA"] * A * gamma_eff(A, B, cfg)


def f(t: float, y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Right-hand side of the ODE system.

    Parameters
    ----------
    t:
        Current time (unused; present for solve_ivp compatibility).
    y:
        State vector [phi, dphi, A, B].
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    numpy.ndarray
        [dphi, ddphi, dA, dB]
    """
    phi, dphi, A, B = y

    R = resonance(phi, cfg)
    g = gamma_eff(A, B, cfg)

    force = -cfg["LAMBDA"] * phi * (phi ** 2 - cfg["V"] ** 2) + cfg["XI"] * A * phi
    ddphi = force - g * dphi

    dA = cfg["MU"] * B * R * (1.0 - A) - cfg["DELTA"] * A
    dB = cfg["ETA"] * A * (1.0 - B) - cfg["THETA"] * B

    return np.array([dphi, ddphi, dA, dB], dtype=float)


def jacobian(y: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Analytical 4×4 Jacobian of f evaluated at state y.

    Parameters
    ----------
    y:
        State vector [phi, dphi, A, B].
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    numpy.ndarray
        4×4 Jacobian matrix.
    """
    phi, dphi, A, B = y

    R   = resonance(phi, cfg)
    Rphi = dr_dphi(phi, cfg)
    g   = gamma_eff(A, B, cfg)
    gA  = dgamma_dA(A, B, cfg)
    gB  = dgamma_dB(A, B, cfg)

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


# ============================================================================
# 3.  ANALYSIS
# ============================================================================
# Configurational potential:
#   P(φ, A) = (Λ/4)(φ²−V²)² − (Ξ·A/2)·φ²
#   ⟹  −∂P/∂φ = −Λ·φ·(φ²−V²) + Ξ·A·φ   (= conservative force)
#
# Extractable energy (exergy analogue):
#   E_ext(y, ref) = max(0, P(ref) − P(y))
#
# Cumulative dissipation (Landauer/Prigogine analogue):
#   D = ∫ γ_eff(A,B) · φ̇² dt
#
# Topological orientation:
#   τ = sign(φ_∞)   →  +1 / −1 / 0  (basin label)


def find_fixed_point(x0: np.ndarray, cfg: dict[str, Any]):
    """Find a fixed point by solving f(x) = 0 with a hybrid Newton method.

    Parameters
    ----------
    x0:
        Initial guess (length-4 array).
    cfg:
        Flat configuration dictionary.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``result.x`` contains the solution; check ``result.success``.
    """
    return root(lambda x: f(0.0, x, cfg), x0, method="hybr")


def stability_report(
    x: np.ndarray, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Local stability at state x.

    Returns
    -------
    J         : 4×4 Jacobian at x
    eig       : eigenvalues of J
    max_real  : maximum real part of the eigenvalues
    is_stable : True iff all Re(λ) < 0
    """
    J = jacobian(x, cfg)
    eig = np.linalg.eigvals(J)
    max_real = float(np.max(np.real(eig)))
    return J, eig, max_real, max_real < 0.0


def eigenvalue_summary(eig: np.ndarray) -> dict[str, Any]:
    """Human-readable dict summary of an eigenvalue array."""
    real = np.real(eig).tolist()
    imag = np.imag(eig).tolist()
    max_real = float(max(real))
    return {
        "real_parts": real,
        "imag_parts": imag,
        "max_real": max_real,
        "is_stable": max_real < 0.0,
    }


def configurational_potential(y: np.ndarray, cfg: dict[str, Any]) -> float:
    """P(φ, A) = (Λ/4)(φ²−V²)² − (Ξ·A/2)·φ²   [model-internal units]."""
    phi, _dphi, A, _B = y
    return (
        cfg["LAMBDA"] / 4.0 * (phi ** 2 - cfg["V"] ** 2) ** 2
        - cfg["XI"] * A / 2.0 * phi ** 2
    )


def extractable_energy(
    y: np.ndarray, ref: np.ndarray, cfg: dict[str, Any]
) -> float:
    """E_ext = max(0, P(ref) − P(y))   [exergy analogue, model units]."""
    return max(
        0.0,
        configurational_potential(ref, cfg) - configurational_potential(y, cfg),
    )


def cumulative_dissipation(sol: Any, cfg: dict[str, Any]) -> float:
    """D = ∫ γ_eff(A,B)·φ̇² dt   [trapezoidal, model units]."""
    _phi, dphi, A, B = sol.y
    integrand = np.array(
        [gamma_eff(a, b, cfg) * dph ** 2 for a, b, dph in zip(A, B, dphi)]
    )
    return float(np.trapezoid(integrand, sol.t))


def topological_orientation(y: np.ndarray) -> int:
    """τ = sign(φ_∞) : +1 / −1 / 0 — which basin φ settled into."""
    return int(np.sign(y[0]))


# ============================================================================
# 4.  SIMULATION
# ============================================================================

def simulate(y0: list[float] | np.ndarray, cfg: dict[str, Any]):
    """Integrate the ODE system from t0 to t1 with RK45.

    Parameters
    ----------
    y0:
        Initial state [phi, dphi, A, B].
    cfg:
        Flat configuration dictionary (must contain t0, t1, n_samples).

    Returns
    -------
    scipy.integrate.OdeResult
        Raises RuntimeError on integration failure.
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


# ============================================================================
# 5.  EXPORT
# ============================================================================

def export_trajectory_csv(
    run_dir: Path,
    sol: Any,
    cfg: dict[str, Any],
    filename: str = "trajectory.csv",
) -> pd.DataFrame:
    """Save the trajectory to a CSV file and return the DataFrame.

    Columns: t, phi, dphi, A, B, R, gamma_eff
    """
    phi, dphi, A, B = sol.y
    df = pd.DataFrame(
        {
            "t":         sol.t,
            "phi":       phi,
            "dphi":      dphi,
            "A":         A,
            "B":         B,
            "R":         [resonance(x, cfg) for x in phi],
            "gamma_eff": [gamma_eff(a, b, cfg) for a, b in zip(A, B)],
        }
    )
    df.to_csv(Path(run_dir) / filename, index=False, encoding="utf-8")
    return df


def export_metadata_json(
    run_dir: Path,
    cfg: dict[str, Any],
    x_star: np.ndarray,
    eig_star: np.ndarray,
    yT: np.ndarray,
    eig_T: np.ndarray,
    filename: str = "metadata.json",
    *,
    e_ext_star: float | None = None,
    e_ext_T: float | None = None,
    dissipation: float | None = None,
    orientation: int | None = None,
) -> None:
    """Save run metadata (config + eigenvalues + derived quantities) to JSON."""
    meta: dict[str, Any] = {
        "config":       {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()},
        "x_star":       x_star.tolist(),
        "eig_star_real": np.real(eig_star).tolist(),
        "eig_star_imag": np.imag(eig_star).tolist(),
        "yT":           yT.tolist(),
        "eig_T_real":   np.real(eig_T).tolist(),
        "eig_T_imag":   np.imag(eig_T).tolist(),
    }
    if e_ext_star is not None:
        meta["extractable_energy_x_star"] = e_ext_star
    if e_ext_T is not None:
        meta["extractable_energy_yT"] = e_ext_T
    if dissipation is not None:
        meta["cumulative_dissipation"] = dissipation
    if orientation is not None:
        meta["topological_orientation"] = orientation
    (Path(run_dir) / filename).write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ============================================================================
# 6.  PLOTTING
# ============================================================================

def plot_phase_portrait(
    run_dir: Path, sol: Any, filename: str = "phase_portrait.png"
) -> None:
    """Save side-by-side phase portraits (phi, dphi) and (A, B)."""
    phi, dphi, A, B = sol.y

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)

    axes[0].plot(phi, dphi, lw=1.2)
    axes[0].set_xlabel("phi")
    axes[0].set_ylabel("dphi")
    axes[0].set_title("Phase portrait (phi, dphi)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(A, B, lw=1.2, color="tab:orange")
    axes[1].set_xlabel("A")
    axes[1].set_ylabel("B")
    axes[1].set_title("Phase portrait (A, B)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(run_dir) / filename, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(
    run_dir: Path, sol: Any, filename: str = "timeseries.png"
) -> None:
    """Save a four-panel time-series plot (phi, dphi, A, B vs t)."""
    phi, dphi, A, B = sol.y
    t = sol.t
    labels = ["phi", "dphi", "A", "B"]
    data   = [phi, dphi, A, B]

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), dpi=150, sharex=True)
    for ax, label, values in zip(axes, labels, data):
        ax.plot(t, values, lw=1.0)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t")
    fig.suptitle("Time series of state variables")
    fig.tight_layout()
    fig.savefig(Path(run_dir) / filename, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 7.  CLI / MAIN
# ============================================================================

def _make_run_dir(base: Path = Path("results/runs")) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: dict[str, Any]) -> None:
    """Execute a complete analysis run with the given flat configuration."""
    run_dir = _make_run_dir()

    y0 = np.array(cfg["y0"], dtype=float)

    # --- Fixed point & stability at x* ---
    fp = find_fixed_point(y0, cfg)
    x_star = fp.x
    _, eig_star, max_real_star, stable_star = stability_report(x_star, cfg)

    # --- Numerical integration ---
    sol = simulate(y0, cfg)
    yT  = sol.y[:, -1]
    _, eig_T, max_real_T, stable_T = stability_report(yT, cfg)

    # --- Derived quantities ---
    e_ext_star  = extractable_energy(x_star, y0, cfg)
    e_ext_T     = extractable_energy(yT, y0, cfg)
    dissipation = cumulative_dissipation(sol, cfg)
    orientation = topological_orientation(yT)

    # --- Save outputs ---
    export_trajectory_csv(run_dir, sol, cfg)
    export_metadata_json(
        run_dir, cfg, x_star, eig_star, yT, eig_T,
        e_ext_star=e_ext_star, e_ext_T=e_ext_T,
        dissipation=dissipation, orientation=orientation,
    )
    plot_phase_portrait(run_dir, sol)
    plot_timeseries(run_dir, sol)

    # --- Console report ---
    np.set_printoptions(precision=6, suppress=True)
    sep = "=" * 72
    print(sep)
    print("NONLINEAR INFORMATION SYSTEM — ANALYSIS RUN")
    print(sep)
    print(f"Run directory           : {run_dir}")
    print(f"Fixed point found       : {fp.success}  {fp.message}")
    print(f"x*                      : {x_star}")
    print(f"max Re(λ) x*            : {max_real_star:.6e}  ({'stable' if stable_star else 'unstable'})")
    print(f"yT                      : {yT}")
    print(f"max Re(λ) yT            : {max_real_T:.6e}  ({'stable' if stable_T else 'unstable'})")
    print()
    print(f"Topological orientation : {orientation:+d}")
    print(f"Extractable energy x*   : {e_ext_star:.5f}")
    print(f"Extractable energy yT   : {e_ext_T:.5f}")
    print(f"Cumulative dissipation  : {dissipation:.5f}")
    print(sep)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Nonlinear information-system — complete analysis run."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file (default: built-in parameters).",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
