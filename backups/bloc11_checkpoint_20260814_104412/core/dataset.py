#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Metadata:
    device_id: str
    date: str
    temperature_mK: float
    notes: str = ""

    def __post_init__(self):
        if self.temperature_mK < 0.0:
            raise ValueError("La température (mK) ne peut pas être négative.")


@dataclass
class Observables:
    target_state: np.ndarray
    measured_fidelity: float
    measured_leakage: float
    fidelity_uncertainty: float = 0.001
    leakage_uncertainty: float = 0.001
    timeseries: np.ndarray | None = None

    def __post_init__(self):
        self.target_state = np.asarray(self.target_state, dtype=complex)
        if self.target_state.ndim != 1 or self.target_state.size == 0:
            raise ValueError("target_state doit être un tableau 1D non vide.")
        
        norm = float(np.linalg.norm(self.target_state))
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise ValueError(f"L'état cible doit être normalisé à 1 (norme : {norm:.6f}).")

        if not (0.0 <= self.measured_fidelity <= 1.0):
            raise ValueError(f"La fidélité mesurée doit être dans [0, 1] (valeur : {self.measured_fidelity}).")

        if not (0.0 <= self.measured_leakage <= 1.0):
            raise ValueError(f"Le leakage mesuré doit être dans [0, 1] (valeur : {self.measured_leakage}).")

        if self.fidelity_uncertainty < 0.0 or self.leakage_uncertainty < 0.0:
            raise ValueError("Les incertitudes de mesure doivent être positives ou nulles.")

        if self.timeseries is not None:
            self.timeseries = np.asarray(self.timeseries)
            if self.timeseries.ndim not in (1, 2):
                raise ValueError("timeseries doit être un tableau 1D ou 2D.")


@dataclass
class Targets:
    target_fidelity: float
    target_leakage_max: float

    def __post_init__(self):
        if not (0.0 <= self.target_fidelity <= 1.0):
            raise ValueError("La fidélité cible doit être comprise dans [0, 1].")
        if not (0.0 <= self.target_leakage_max <= 1.0):
            raise ValueError("Le leakage maximal cible doit être compris dans [0, 1].")


@dataclass
class ExperimentalDataset:
    metadata: Metadata
    observables: Observables
    targets: Targets

    def as_dict(self) -> dict:
        """Exporte le dataset sous forme de dictionnaire structuré pour audit ou export."""
        return {
            "metadata": {
                "device_id": self.metadata.device_id,
                "date": self.metadata.date,
                "temperature_mK": self.metadata.temperature_mK,
                "notes": self.metadata.notes,
            },
            "observables": {
                "measured_fidelity": self.observables.measured_fidelity,
                "measured_leakage": self.observables.measured_leakage,
                "fidelity_uncertainty": self.observables.fidelity_uncertainty,
                "leakage_uncertainty": self.observables.leakage_uncertainty,
                "has_timeseries": self.observables.timeseries is not None,
            },
            "targets": {
                "target_fidelity": self.targets.target_fidelity,
                "target_leakage_max": self.targets.target_leakage_max,
            },
        }
