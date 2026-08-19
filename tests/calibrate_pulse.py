#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantumLab — Script de calibration de l'amplitude d'impulsion (Phase 3)
Version autonome (NumPy pur, sans dépendance SciPy).
"""

import numpy as np

from core.transmon import TransmonSystem
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.pipeline import QuantumPipeline, PipelineConfig


def objective(
    pulse_amplitude,
    pipeline,
    system,
    noise,
    target_state,
    w_leak=10.0,
    w_fidelity=1.0,
):
    t0 = pipeline.config.t_duration / 2.0
    
    pulse = gaussian(
        pipeline.t,
        t0=t0,
        sigma=3.0,
        amplitude=pulse_amplitude,
    )

    trajectory = pipeline.step_1_baseline(
        system,
        pulse,
        noise=noise.copy(),
    )

    metrics = pipeline.step_3_metrology(trajectory, target_state)

    fidelity = metrics["fidelity_final"]
    
    populations = metrics["populations_final"]
    leakage_direct = float(populations[2])
    leakage_pipeline = float(metrics["leakage_final"])

    if not np.isclose(leakage_direct, leakage_pipeline, atol=1e-10):
        raise ValueError(
            "Incohérence entre leakage_final et populations_final[2]"
        )

    leakage = leakage_direct

    cost = (
        w_leak * leakage
        + w_fidelity * (1.0 - fidelity) ** 2
    )

    return float(cost)


def evaluate_amplitude(
    pulse_amplitude,
    pipeline,
    system,
    noise,
    target_state,
):
    t0 = pipeline.config.t_duration / 2.0

    pulse = gaussian(
        pipeline.t,
        t0=t0,
        sigma=3.0,
        amplitude=pulse_amplitude,
    )

    trajectory = pipeline.step_1_baseline(
        system,
        pulse,
        noise=noise.copy(),
    )

    return pipeline.step_3_metrology(
        trajectory,
        target_state,
    )


def minimize_scalar_golden(func, bounds, args=(), tol=1e-5, maxiter=50):
    a, b = bounds
    invphi = (np.sqrt(5.0) - 1.0) / 2.0

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    fc = func(c, *args)
    fd = func(d, *args)

    for _ in range(maxiter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - invphi * (b - a)
            fc = func(c, *args)
        else:
            a = c
            c = d
            fc = fd
            d = a + invphi * (b - a)
            fd = func(d, *args)

    xmin = (a + b) / 2.0
    fmin = func(xmin, *args)

    class Result:
        pass

    res = Result()
    res.x = xmin
    res.fun = fmin
    res.success = True
    res.message = "Optimisation terminée avec succès (méthode de la section dorée)."
    return res


def calibrate_amplitude(
    pipeline,
    system,
    noise,
    target_state,
    amplitude_min=0.0,
    amplitude_max=2.0,
):
    result = minimize_scalar_golden(
        objective,
        bounds=(amplitude_min, amplitude_max),
        args=(pipeline, system, noise, target_state),
    )

    return {
        "optimal_amplitude": result.x,
        "objective_value": result.fun,
        "success": result.success,
        "message": result.message,
    }


def main():
    print("==============================================")
    print("CALIBRATION DE L'AMPLITUDE D'IMPULSION (PHASE 3)")
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

    system = TransmonSystem(
        omega_q=config.omega_q,
        anharmonicity=config.anharmonicity,
        n_levels=config.n_levels,
    )

    target_state = np.array([0.0, 1.0, 0.0], dtype=complex)

    noise = (
        generate_1f_noise(
            n_samples=len(pipeline.t),
            dt=pipeline.dt,
            seed=config.seed,
        )
        * 0.05
    )

    print("[*] Lancement de l'optimisation de pulse_amplitude...")
    calib_result = calibrate_amplitude(
        pipeline=pipeline,
        system=system,
        noise=noise,
        target_state=target_state,
        amplitude_min=0.05,
        amplitude_max=1.5,
    )

    metrics_opt = evaluate_amplitude(
        pulse_amplitude=calib_result["optimal_amplitude"],
        pipeline=pipeline,
        system=system,
        noise=noise,
        target_state=target_state,
    )

    print()
    print("[*] Métriques à l'amplitude optimale")
    print(f"    Amplitude       : {calib_result['optimal_amplitude']:.6f}")
    print(f"    Fidélité finale : {metrics_opt['fidelity_final']:.6f}")
    print(f"    Fuite finale    : {metrics_opt['leakage_final']:.6e}")
    print(f"    Population P2   : {metrics_opt['populations_final'][2]:.6e}")
    print(f"    Succès opt.     : {calib_result['success']}")
    print("==============================================")


if __name__ == "__main__":
    main()
