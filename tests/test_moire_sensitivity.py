#!/usr/init/env python3
# -*- coding: utf-8 -*-

"""
QuantumLab — Test de sensibilité Moiré (Phase 3)
Balayage de λ avec métrologie complète sur trajectoire.
"""

import numpy as np

from core.transmon import TransmonSystem
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.moire import MoireField
from core.pipeline import QuantumPipeline, PipelineConfig


def main():
    print("==============================================")
    print("QUANTUMLAB — SENSIBILITÉ MOIRÉ (PHASE 3)")
    print("==============================================")

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

    pulse_amplitude = 0.5
    pulse_sigma = 3.0
    t0 = config.t_duration / 2.0

    pulse = gaussian(
        pipeline.t,
        t0=t0,
        sigma=pulse_sigma,
        amplitude=pulse_amplitude,
    )

    noise = (
        generate_1f_noise(
            n_samples=len(pipeline.t),
            dt=pipeline.dt,
            seed=config.seed,
        )
        * 0.05
    )

    system = TransmonSystem(
        omega_q=config.omega_q,
        anharmonicity=config.anharmonicity,
        n_levels=config.n_levels,
    )

    target_state = np.array([0.0, 1.0, 0.0], dtype=complex)

    # Grille spatiale et champ Moiré
    xv = np.linspace(-5.0, 5.0, 101)
    yv = np.linspace(-5.0, 5.0, 101)
    xg, yg = np.meshgrid(xv, yv)

    moire_field = MoireField(
        k1=(1.0, 0.0),
        k2=(1.05, 0.0),
        amplitude=1.0,
    )

    lambda_values = [0.0, 0.01, 0.05, 0.10]
    baseline_metrics = None

    for lam in lambda_values:
        if lam == 0.0:
            trajectory = pipeline.step_1_baseline(system, pulse, noise)
        else:
            trajectory = pipeline.step_2_incremental_coupling(
                system,
                pulse,
                noise,
                moire_field,
                xg,
                yg,
                x0=0.0,
                y0=0.0,
                sigma_device=0.5,
                lambda_val=lam,
            )

        metrics = pipeline.step_3_metrology(trajectory, target_state)

        if lam == 0.0:
            baseline_metrics = metrics
            print(f"\n[λ = {lam:.2f}] (Baseline)")
            print(f"    Fidélité finale : {metrics['fidelity_final']:.6f}")
            print(f"    Leakage final   : {metrics['leakage_final']:.6e}")
            print(f"    Leakage max     : {metrics['leakage_max']:.6e}")
            print(f"    Populations     : {np.round(metrics['populations_final'], 4)}")
        else:
            diff_fid = metrics['fidelity_final'] - baseline_metrics['fidelity_final']
            print(f"\n[λ = {lam:.2f}]")
            print(f"    Fidélité finale : {metrics['fidelity_final']:.6f} | ΔF : {diff_fid:+.6e}")
            print(f"    Leakage final   : {metrics['leakage_final']:.6e}")
            print(f"    Leakage max     : {metrics['leakage_max']:.6e}")
            print(f"    Populations     : {np.round(metrics['populations_final'], 4)}")

    print("\n==============================================")
    print("BALAYAGE DE SENSIBILITÉ TERMINÉ")
    print("==============================================")


if __name__ == "__main__":
    main()
