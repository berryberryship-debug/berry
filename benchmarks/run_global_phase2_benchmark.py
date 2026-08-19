#!/usr/init/env python3
# -*- coding: utf-8 -*-
"""
QuantumLab — Global Benchmark Phase 2 (Monte-Carlo Multi-Runs)
"""

import os
import json
import csv
import time
import numpy as np

from core.transmon import TransmonSystem
from core.noise import generate_1f_noise
from core.pulses import gaussian, drag
from core.metrics import state_fidelity

def run_phase2_benchmark():
    config = {
        "base_seed": 20260813,
        "monte_carlo_runs": 5,
        "omega_q": 5.0,
        "anharmonicity": -0.3,
        "n_levels": 3,
        "t_duration": 20.0,
        "n_steps": 1000,
        "pulse_sigma": 3.0,
        "pulse_amplitude": 0.5
    }

    t = np.linspace(0, config["t_duration"], config["n_steps"])
    dt = t[1] - t[0]

    system = TransmonSystem(
        omega_q=config["omega_q"],
        anharmonicity=config["anharmonicity"],
        n_levels=config["n_levels"]
    )

    target_state = np.array([0.0, 1.0, 0.0], dtype=complex)
    t0 = config["t_duration"] / 2.0
    pulse_gauss = gaussian(t, t0=t0, sigma=config["pulse_sigma"], amplitude=config["pulse_amplitude"])
    pulse_drag = drag(t, t0=t0, sigma=config["pulse_sigma"], amplitude=config["pulse_amplitude"], beta=0.2)

    results_summary = []
    start_time = time.time()

    seeds = [config["base_seed"] + i for i in range(config["monte_carlo_runs"])]

    for run_idx, seed in enumerate(seeds):
        noise = generate_1f_noise(n_samples=len(t), dt=dt, seed=seed) * 0.05

        # Scénario A : Idéal (sans bruit)
        states_ideal = system.simulate(t, pulse_gauss, noise=np.zeros_like(t))
        final_ideal = states_ideal[:, -1]
        fid_ideal = state_fidelity(target_state, final_ideal)
        leak_ideal = float(np.abs(final_ideal[2])**2)

        # Scénario B : Gaussien avec Bruit 1/f
        states_noisy_gauss = system.simulate(t, pulse_gauss, noise=noise)
        final_noisy_gauss = states_noisy_gauss[:, -1]
        fid_noisy_gauss = state_fidelity(target_state, final_noisy_gauss)
        leak_noisy_gauss = float(np.abs(final_noisy_gauss[2])**2)

        # Scénario C : DRAG avec Bruit 1/f
        states_noisy_drag = system.simulate(t, pulse_drag, noise=noise)
        final_noisy_drag = states_noisy_drag[:, -1]
        fid_noisy_drag = state_fidelity(target_state, final_noisy_drag)
        leak_noisy_drag = float(np.abs(final_noisy_drag[2])**2)

        run_data = {
            "run_index": run_idx,
            "seed": seed,
            "ideal_gaussian": {
                "fidelity": float(fid_ideal),
                "infidelity": float(1.0 - fid_ideal),
                "leakage_level_2": leak_ideal
            },
            "noisy_gaussian": {
                "fidelity": float(fid_noisy_gauss),
                "infidelity": float(1.0 - fid_noisy_gauss),
                "leakage_level_2": leak_noisy_gauss
            },
            "noisy_drag": {
                "fidelity": float(fid_noisy_drag),
                "infidelity": float(1.0 - fid_noisy_drag),
                "leakage_level_2": leak_noisy_drag
            }
        }
        results_summary.append(run_data)

    execution_time = time.time() - start_time

    def get_stats(scenario, metric):
        vals = [r[scenario][metric] for r in results_summary]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }

    aggregated = {
        "config": config,
        "execution_time_seconds": round(execution_time, 4),
        "monte_carlo_aggregated": {
            "ideal_gaussian": {
                "fidelity": get_stats("ideal_gaussian", "fidelity"),
                "leakage_level_2": get_stats("ideal_gaussian", "leakage_level_2")
            },
            "noisy_gaussian": {
                "fidelity": get_stats("noisy_gaussian", "fidelity"),
                "leakage_level_2": get_stats("noisy_gaussian", "leakage_level_2")
            },
            "noisy_drag": {
                "fidelity": get_stats("noisy_drag", "fidelity"),
                "leakage_level_2": get_stats("noisy_drag", "leakage_level_2")
            }
        },
        "runs_detail": results_summary
    }

    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/benchmark_phase2_result.json"
    csv_path = "outputs/benchmark_phase2_result.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=4)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_index", "seed", "scenario", "fidelity", "infidelity", "leakage_level_2"])
        for r in results_summary:
            for sc in ["ideal_gaussian", "noisy_gaussian", "noisy_drag"]:
                writer.writerow([
                    r["run_index"],
                    r["seed"],
                    sc,
                    r[sc]["fidelity"],
                    r[sc]["infidelity"],
                    r[sc]["leakage_level_2"]
                ])

    print(f"==> [PHASE 2] Benchmark terminé. {len(results_summary)} runs enregistrés.")
    print(f"    Seeds utilisées : {[r['seed'] for r in results_summary]}")

if __name__ == "__main__":
    run_phase2_benchmark()
