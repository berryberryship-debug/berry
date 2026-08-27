"""Tests for nonlinear_information_system.export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nonlinear_information_system.simulation import simulate
from nonlinear_information_system.export import export_trajectory_csv, export_metadata_json
from nonlinear_information_system.analysis import stability_report

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
    "t0": 0.0,
    "t1": 10.0,
    "n_samples": 200,
    "y0": [0.05, 0.0, 0.02, 0.02],
}


@pytest.fixture()
def run_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def solution():
    return simulate(CFG["y0"], CFG)


class TestExportTrajectoryCSV:
    def test_file_created(self, run_dir, solution):
        export_trajectory_csv(run_dir, solution, CFG)
        assert (run_dir / "trajectory.csv").exists()

    def test_expected_columns(self, run_dir, solution):
        import pandas as pd

        df = export_trajectory_csv(run_dir, solution, CFG)
        assert set(df.columns) == {"t", "phi", "dphi", "A", "B", "R", "gamma_eff"}

    def test_row_count(self, run_dir, solution):
        df = export_trajectory_csv(run_dir, solution, CFG)
        assert len(df) == CFG["n_samples"]


class TestExportMetadataJSON:
    def test_file_created(self, run_dir, solution):
        yT = solution.y[:, -1]
        _, eig_T, _, _ = stability_report(yT, CFG)
        x_star = yT.copy()
        eig_star = eig_T.copy()
        export_metadata_json(run_dir, CFG, x_star, eig_star, yT, eig_T)
        assert (run_dir / "metadata.json").exists()

    def test_json_structure(self, run_dir, solution):
        yT = solution.y[:, -1]
        _, eig_T, _, _ = stability_report(yT, CFG)
        x_star = yT.copy()
        eig_star = eig_T.copy()
        export_metadata_json(run_dir, CFG, x_star, eig_star, yT, eig_T)
        data = json.loads((run_dir / "metadata.json").read_text())
        for key in ("config", "x_star", "eig_star_real", "eig_star_imag", "yT", "eig_T_real", "eig_T_imag"):
            assert key in data, f"Key '{key}' missing from metadata.json"
