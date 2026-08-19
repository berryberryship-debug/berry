#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core.dataset import ExperimentalDataset


@dataclass
class PipelineConfig:
    omega_q: float = 5.0
    anharmonicity: float = -0.3
    n_levels: int = 3
    logical_levels: tuple[int, int] = (0, 1)
    t_duration: float = 20.0
    n_steps: int = 1000
    seed: int = 20260813

    def __post_init__(self):
        if any(i < 0 or i >= self.n_levels for i in self.logical_levels):
            raise ValueError("logical_levels doit être inclus dans [0, n_levels-1].")


class QuantumPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.t = np.linspace(0.0, config.t_duration, config.n_steps)
        self.dt = self.t[1] - self.t[0]

    def step_1_baseline(self, system, pulse, noise):
        return system.simulate(self.t, pulse, noise=noise)

    def step_2_incremental_coupling(
        self, system, pulse, noise, moire_field, xg, yg, x0, y0, sigma_device, lambda_val
    ):
        from core.moire import effective_field
        m_eff_traj = np.array([
            effective_field(moire_field, xg, yg, t=ti, x0=x0, y0=y0, sigma_device=sigma_device)
            for ti in self.t
        ])
        detuning = lambda_val * m_eff_traj
        return system.simulate_with_custom_detuning(self.t, pulse, noise=noise, detuning=detuning)

    def step_3_metrology(self, trajectory: np.ndarray, target_state: np.ndarray):
        """
        Métrologie étendue sur la trajectoire complète :
        - Fidélité finale
        - Leakage final (hors sous-espace logique)
        - Leakage maximal transitoire sur toute la trajectoire
        - Populations finales par niveau
        """
        from core.metrics import state_fidelity

        if trajectory.ndim != 2 or trajectory.shape[0] != self.config.n_levels:
            raise ValueError("Format de trajectoire incompatible avec n_levels.")

        final_state = trajectory[:, -1]
        probs_final = np.abs(final_state) ** 2
        probs_traj = np.abs(trajectory) ** 2

        fidelity_final = float(state_fidelity(target_state, final_state))
        
        logical_indices = list(self.config.logical_levels)
        leakage_final = float(max(0.0, 1.0 - np.sum(probs_final[logical_indices])))
        
        # Pic de fuite au cours de l'évolution temporelle
        logical_pop_traj = np.sum(probs_traj[logical_indices, :], axis=0)
        leakage_max = float(np.max(np.maximum(0.0, 1.0 - logical_pop_traj)))

        return {
            "fidelity_final": fidelity_final,
            "leakage_final": leakage_final,
            "leakage_max": leakage_max,
            "populations_final": probs_final,
        }

    def step_4_calibration_objective(
        self, theta_params, dataset: ExperimentalDataset, system_factory, pulse, noise
    ):
        system = system_factory(theta_params)
        trajectory = self.step_1_baseline(system, pulse, noise)
        
        metrics = self.step_3_metrology(trajectory, dataset.observables.target_state)

        sigma_fid = max(dataset.observables.fidelity_uncertainty, 1e-6)
        sigma_leak = max(dataset.observables.leakage_uncertainty, 1e-6)

        fid_err = ((metrics["fidelity_final"] - dataset.targets.target_fidelity) / sigma_fid) ** 2
        leak_err = ((metrics["leakage_final"] - dataset.targets.target_leakage_max) / sigma_leak) ** 2

        return fid_err + leak_err
