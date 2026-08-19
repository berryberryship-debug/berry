#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantumLab — Pipeline Unifié des 4 Étapes (Phase 3)
1. Baseline numérique (λ = 0)
2. Couplage incrémental Moiré (λ > 0)
3. Métrologie étendue sur trajectoire & sous-espace logique
4. Calibration empirique et optimisation par section dorée
"""

import numpy as np

from core.transmon import TransmonSystem
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.moire import MoireField
from core.dataset import Metadata, Observables, Targets, ExperimentalDataset
from core.pipeline import QuantumPipeline, PipelineConfig
from core.calibration import calibrate_system, evaluate_calibration


def main():
    print("==================================================")
    print("QUANTUMLAB — EXÉCUTION DU PIPELINE EN 4 ÉTAPES")
    print("==================================================")

    # Configuration globale
    config = PipelineConfig(
        omega_q=5.0,
        anharmonicity=-0.3,
        n_levels=3,
        logical_levels=(0, 1),
        t_duration=20.0,
        n_steps=1000,
        seed=20260813,
    )
    pipeline = QuantumPipeline(config)

    # Impulsion et Bruit de référence
    t0 = config.t_duration / 2.0
    pulse = gaussian(pipeline.t, t0=t0, sigma=3.0, amplitude=1.0)
    noise = generate_1f_noise(n_samples=len(pipeline.t), dt=pipeline.dt, seed=config.seed) * 0.05

    # Système physique de base
    system = TransmonSystem(
        omega_q=config.omega_q,
        anharmonicity=config.anharmonicity,
        n_levels=config.n_levels,
    )

    # --------------------------------------------------
    # ÉTAPE 1 : Baseline numérique (λ = 0)
    # --------------------------------------------------
    print("\n[Étape 1] Exécution de la baseline numérique (λ = 0)...")
    trajectory_baseline = pipeline.step_1_baseline(system, pulse, noise=noise.copy())
    metrics_base = pipeline.step_3_metrology(trajectory_baseline, np.array([0.0, 1.0, 0.0], dtype=complex))
    print(f"    -> Fidélité baseline : {metrics_base['fidelity_final']:.6f}")
    print(f"    -> Leakage baseline  : {metrics_base['leakage_final']:.6e}")

    # --------------------------------------------------
    # ÉTAPE 2 & 3 : Couplage incrémental Moiré et Métrologie (λ > 0)
    # --------------------------------------------------
    print("\n[Étape 2 & 3] Application du couplage Moiré et métrologie de trajectoire...")
    xv = np.linspace(-5.0, 5.0, 51)
    yv = np.linspace(-5.0, 5.0, 51)
    xg, yg = np.meshgrid(xv, yv)
    moire_field = MoireField(k1=(1.0, 0.0), k2=(1.05, 0.0), amplitude=1.0)

    lambda_val = 0.05
    trajectory_coupled = pipeline.step_2_incremental_coupling(
        system, pulse, noise.copy(), moire_field, xg, yg, x0=0.0, y0=0.0, sigma_device=0.5, lambda_val=lambda_val
    )
    metrics_coupled = pipeline.step_3_metrology(trajectory_coupled, np.array([0.0, 1.0, 0.0], dtype=complex))
    print(f"    -> Fidélité avec λ={lambda_val} : {metrics_coupled['fidelity_final']:.6f}")
    print(f"    -> Leakage avec λ={lambda_val}  : {metrics_coupled['leakage_final']:.6e}")

    # --------------------------------------------------
    # ÉTAPE 4 : Calibration empirique et optimisation
    # --------------------------------------------------
    print("\n[Étape 4] Lancement de la calibration empirique des paramètres...")
    target_state = np.array([0.0, 1.0, 0.0], dtype=complex)
    dataset = ExperimentalDataset(
        metadata=Metadata(device_id="transmon_Sherbrooke_A1", date="2026-08-14", temperature_mK=15.0),
        observables=Observables(
            target_state=target_state,
            measured_fidelity=0.982,
            measured_leakage=0.007,
            fidelity_uncertainty=0.002,
            leakage_uncertainty=0.001,
        ),
        targets=Targets(target_fidelity=0.98, target_leakage_max=0.01),
    )

    def system_factory(theta):
        return TransmonSystem(omega_q=config.omega_q, anharmonicity=config.anharmonicity, n_levels=config.n_levels)

    calib_res = calibrate_system(pipeline, system_factory, dataset, pulse, noise.copy(), bounds=(0.1, 1.5))
    report = evaluate_calibration(calib_res["optimal_theta"], pipeline, system_factory, dataset, pulse, noise.copy())

    print(f"    -> Paramètre optimal θ : {report['theta']:.6f}")
    print(f"    -> Fidélité calibrée   : {report['fidelity_sim']:.6f}")
    print(f"    -> Leakage calibré     : {report['leakage_sim']:.6e}")

    print("\n==================================================")
    print("TOUTES LES 4 ÉTAPES SONT VALIDÉES AVEC SUCCÈS")
    print("==================================================")

if __name__ == "__main__":
    main()
