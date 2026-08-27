"""
cli.py
------
Command-line entry point for a complete analysis run.

Usage::

    python -m nonlinear_information_system.cli
    python -m nonlinear_information_system.cli --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import load_config, flatten_config
from .analysis import find_fixed_point, stability_report, extractable_energy, cumulative_dissipation, topological_orientation
from .simulation import simulate
from .export import export_trajectory_csv, export_metadata_json
from .plotting import plot_phase_portrait, plot_timeseries


def _make_run_dir(base: Path = Path("results/runs")) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg: dict) -> None:
    """Execute a complete analysis run with the given flat configuration."""
    run_dir = _make_run_dir()

    y0 = np.array(cfg["y0"], dtype=float)

    fp = find_fixed_point(y0, cfg)
    x_star = fp.x
    _, eig_star, max_real_star, stable_star = stability_report(x_star, cfg)

    sol = simulate(y0, cfg)
    yT = sol.y[:, -1]
    _, eig_T, max_real_T, stable_T = stability_report(yT, cfg)

    e_ext_star = extractable_energy(x_star, y0, cfg)
    e_ext_T = extractable_energy(yT, y0, cfg)
    dissipation = cumulative_dissipation(sol, cfg)
    orientation = topological_orientation(yT)

    export_trajectory_csv(run_dir, sol, cfg)
    export_metadata_json(run_dir, cfg, x_star, eig_star, yT, eig_T,
                         e_ext_star=e_ext_star, e_ext_T=e_ext_T,
                         dissipation=dissipation, orientation=orientation)
    plot_phase_portrait(run_dir, sol)
    plot_timeseries(run_dir, sol)

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
        description="Run a nonlinear information-system analysis."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML configuration file (default: config/default.yaml).",
    )
    args = parser.parse_args(argv)

    nested_cfg = load_config(args.config)
    cfg = flatten_config(nested_cfg)
    run(cfg)


if __name__ == "__main__":
    main()
