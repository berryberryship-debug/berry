#!/usr/bin/env python3
"""
main.py — Point d'entrée racine
================================
Ce fichier regroupe tous les modules du package nonlinear_information_system
et lance un run d'analyse complet.

Usage::

    python main.py
    python main.py --config config/default.yaml

Il est l'équivalent de::

    python -m nonlinear_information_system

Modules importés
----------------
config      : chargement et aplatissement du fichier YAML de configuration
model       : résonance, friction effective, second membre ODE, Jacobien analytique
analysis    : point fixe, stabilité, énergie extractible, dissipation, orientation
simulation  : intégration numérique solve_ivp
export      : sauvegarde CSV (trajectoire) et JSON (métadonnées)
plotting    : portraits de phase et séries temporelles
cli         : orchestration complète d'un run
"""

# ---------------------------------------------------------------------------
# Imports — tous les modules du package
# ---------------------------------------------------------------------------

from nonlinear_information_system.config import load_config, flatten_config

from nonlinear_information_system.model import (
    resonance,
    dr_dphi,
    gamma_eff,
    dgamma_dA,
    dgamma_dB,
    f,
    jacobian,
)

from nonlinear_information_system.analysis import (
    find_fixed_point,
    stability_report,
    eigenvalue_summary,
    configurational_potential,
    extractable_energy,
    cumulative_dissipation,
    topological_orientation,
)

from nonlinear_information_system.simulation import simulate

from nonlinear_information_system.export import (
    export_trajectory_csv,
    export_metadata_json,
)

from nonlinear_information_system.plotting import (
    plot_phase_portrait,
    plot_timeseries,
)

from nonlinear_information_system.cli import run, main

# ---------------------------------------------------------------------------
# Lancement
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
