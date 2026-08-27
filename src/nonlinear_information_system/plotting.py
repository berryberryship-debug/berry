"""
plotting.py
-----------
Phase portraits and time-series visualisations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def plot_phase_portrait(
    run_dir: Path,
    sol: Any,
    filename: str = "phase_portrait.png",
) -> None:
    """Save phase-portrait figures ``(phi, dphi)`` and ``(A, B)``.

    Parameters
    ----------
    run_dir:
        Directory in which the PNG file will be written.
    sol:
        ODE solution object as returned by :func:`simulation.simulate`.
    filename:
        Name of the output file (default: ``phase_portrait.png``).
    """
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
    run_dir: Path,
    sol: Any,
    filename: str = "timeseries.png",
) -> None:
    """Save a four-panel time-series plot of all state variables.

    Parameters
    ----------
    run_dir:
        Directory in which the PNG file will be written.
    sol:
        ODE solution object as returned by :func:`simulation.simulate`.
    filename:
        Name of the output file (default: ``timeseries.png``).
    """
    phi, dphi, A, B = sol.y
    t = sol.t
    labels = ["phi", "dphi", "A", "B"]
    data = [phi, dphi, A, B]

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
