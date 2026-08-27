"""
nonlinear_information_system
============================
Nonlinear information-theoretic dynamical system with Jacobian analysis.
"""

from .config import load_config, flatten_config
from .model import resonance, gamma_eff, f, jacobian
from .analysis import (
    find_fixed_point,
    stability_report,
    eigenvalue_summary,
    configurational_potential,
    extractable_energy,
    cumulative_dissipation,
    topological_orientation,
)
from .simulation import simulate
from .export import export_trajectory_csv, export_metadata_json

__all__ = [
    "load_config",
    "flatten_config",
    "resonance",
    "gamma_eff",
    "f",
    "jacobian",
    "find_fixed_point",
    "stability_report",
    "eigenvalue_summary",
    "configurational_potential",
    "extractable_energy",
    "cumulative_dissipation",
    "topological_orientation",
    "simulate",
    "export_trajectory_csv",
    "export_metadata_json",
]
