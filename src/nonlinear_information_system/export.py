"""
export.py
---------
Save trajectory data (CSV) and run metadata (JSON) to disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .model import resonance, gamma_eff


def export_trajectory_csv(
    run_dir: Path,
    sol: Any,
    cfg: dict[str, Any],
    filename: str = "trajectory.csv",
) -> pd.DataFrame:
    """Export the simulation trajectory to a CSV file.

    Parameters
    ----------
    run_dir:
        Directory in which the file will be written.
    sol:
        ODE solution object as returned by :func:`simulation.simulate`.
    cfg:
        Flat configuration dictionary.
    filename:
        Name of the output file (default: ``trajectory.csv``).

    Returns
    -------
    pandas.DataFrame
        The exported data frame.
    """
    phi, dphi, A, B = sol.y
    df = pd.DataFrame(
        {
            "t": sol.t,
            "phi": phi,
            "dphi": dphi,
            "A": A,
            "B": B,
            "R": [resonance(x, cfg) for x in phi],
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
    """Export run metadata (config + eigenvalue summaries + derived quantities) to a JSON file.

    Parameters
    ----------
    run_dir:
        Directory in which the file will be written.
    cfg:
        Flat configuration dictionary.
    x_star:
        Fixed-point state vector.
    eig_star:
        Eigenvalues at the fixed point.
    yT:
        Final state vector from the simulation.
    eig_T:
        Eigenvalues at the final state.
    filename:
        Name of the output file (default: ``metadata.json``).
    e_ext_star:
        Extractable energy at the fixed point (optional).
    e_ext_T:
        Extractable energy at the final state (optional).
    dissipation:
        Cumulative dissipation along the trajectory (optional).
    orientation:
        Topological orientation of the final state (optional).
    """
    meta: dict[str, Any] = {
        "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()},
        "x_star": x_star.tolist(),
        "eig_star_real": np.real(eig_star).tolist(),
        "eig_star_imag": np.imag(eig_star).tolist(),
        "yT": yT.tolist(),
        "eig_T_real": np.real(eig_T).tolist(),
        "eig_T_imag": np.imag(eig_T).tolist(),
    }
    if e_ext_star is not None:
        meta["extractable_energy_x_star"] = e_ext_star
    if e_ext_T is not None:
        meta["extractable_energy_yT"] = e_ext_T
    if dissipation is not None:
        meta["cumulative_dissipation"] = dissipation
    if orientation is not None:
        meta["topological_orientation"] = orientation
    (Path(run_dir) / filename).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
