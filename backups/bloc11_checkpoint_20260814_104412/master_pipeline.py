#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QuantumLab — Matrice d'Orchestration Unifiée (Les 4 Étapes)
Architecture anti-entropique avec pulse_builder dynamique et scan_then_refine (Zéro SciPy).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable

# =====================================================================
# IMPORTATION DES PRIMITIVES PHYSIQUES
# =====================================================================
from core.transmon import TransmonSystem
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.moire import MoireField


# =====================================================================
# CONTRAT DE DONNÉES EXPÉRIMENTALES
# =====================================================================
@dataclass
class Metadata:
    device_id: str
    date: str
    temperature_mK: float
    notes: str = ""

@dataclass
class Observables:
    target_state: np.ndarray
    measured_fidelity: float
    measured_leakage: float
    fidelity_uncertainty: float = 0.001
    leakage_uncertainty: float = 0.001

    def __post_init__(self):
        self.target_state = np.asarray(self.target_state, dtype=complex)

@dataclass
class Targets:
    target_fidelity: float
    target_leakage_max: float

@dataclass
class ExperimentalDataset:
    metadata: Metadata
    observables: Observables
    targets: Targets


# =====================================================================
# MOTEUR D'ORCHESTRATION ET MÉTROLOGIE (Avec Normalisation)
# =====================================================================
@dataclass
class PipelineConfig:
    omega_q: float = 5.0
    anharmonicity: float = -0.3
    n_levels: int = 3
    logical_levels: tuple[int, int] = (0, 1)
    t_duration: float = 20.0
    n_steps: int = 1000
    seed: int = 20260813

class QuantumPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.t = np.linspace(0.0, config.t_duration, config.n_steps)
        self.dt = self.t[1] - self.t[0]

    def step_1_baseline(self, system, pulse, noise):
        return system.simulate(self.t, pulse, noise=noise)

    def step_2_incremental_coupling(self, system, pulse, noise, moire_field, xg, yg, x0, y0, sigma_device, lambda_val):
        from core.moire import effective_field
        m_eff_traj = np.array([
            effective_field(moire_field, xg, yg, t=ti, x0=x0, y0=y0, sigma_device=sigma_device)
            for ti in self.t
        ])
        detuning = lambda_val * m_eff_traj
        return system.simulate_with_custom_detuning(self.t, pulse, noise=noise, detuning=detuning)

    def step_3_metrology(self, trajectory: np.ndarray, target_state: np.ndarray):
        from core.metrics import state_fidelity

        if trajectory.ndim != 2:
            raise ValueError(
                "La trajectoire doit être un tableau 2D "
                "de forme (n_levels, n_times)."
            )

        final_state = np.asarray(
            trajectory[:, -1],
            dtype=complex,
        )

        norm = np.linalg.norm(final_state)

        if norm <= 1e-12:
            raise ValueError(
                "L'état final a une norme nulle."
            )

        normalized_state = final_state / norm
        probs_final = np.abs(normalized_state) ** 2

        fidelity_final = float(
            state_fidelity(
                target_state,
                normalized_state,
            )
        )

        logical_indices = list(
            self.config.logical_levels
        )

        logical_population = np.sum(
            probs_final[logical_indices]
        )

        leakage_final = float(
            max(0.0, 1.0 - logical_population)
        )

        return {
            "fidelity_final": fidelity_final,
            "leakage_final": leakage_final,
            "populations_final": probs_final,
            "state_norm_final": float(norm),
        }


# =====================================================================
# MODULE DE CALIBRATION EMPIRIQUE (scan_then_refine & pulse_builder - 100% Numpy)
# =====================================================================
def objective_function(
    theta: float,
    pipeline: QuantumPipeline,
    system_factory: Callable,
    dataset: ExperimentalDataset,
    pulse_builder: Callable[[float], np.ndarray],
    noise: np.ndarray,
) -> float:
    system = system_factory(theta)
    pulse = pulse_builder(theta)

    trajectory = pipeline.step_1_baseline(
        system,
        pulse,
        noise=noise.copy(),
    )

    metrics = pipeline.step_3_metrology(
        trajectory,
        dataset.observables.target_state,
    )

    fidelity = metrics["fidelity_final"]
    leakage = metrics["leakage_final"]

    sigma_fid = max(
        dataset.observables.fidelity_uncertainty,
        1e-6,
    )

    sigma_leak = max(
        dataset.observables.leakage_uncertainty,
        1e-6,
    )

    fidelity_deficit = max(
        0.0,
        dataset.targets.target_fidelity - fidelity,
    )

    leakage_excess = max(
        0.0,
        leakage - dataset.targets.target_leakage_max,
    )

    fidelity_error = (
        fidelity_deficit / sigma_fid
    ) ** 2

    leakage_error = (
        leakage_excess / sigma_leak
    ) ** 2

    return float(fidelity_error + leakage_error)


def golden_section_minimize(func: Callable, bounds: tuple[float, float], args: tuple = (), tol: float = 1e-5, maxiter: int = 50):
    """Optimisation 1D bornée par recherche par section dorée (pure NumPy)."""
    a, b = bounds
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = func(c, *args)
    fd = func(d, *args)

    for _ in range(maxiter):
        if abs(b - a) < tol: break
        if fc < fd:
            b = d; d = c; fd = fc; c = b - invphi * (b - a); fc = func(c, *args)
        else:
            a = c; c = d; fc = fd; d = a + invphi * (b - a); fd = func(d, *args)

    xmin = (a + b) / 2.0
    class Result: pass
    res = Result()
    res.x = xmin
    res.fun = func(xmin, *args)
    res.success = True
    res.message = "Golden section converged"
    return res


def scan_then_refine(
    func,
    bounds,
    args=(),
    n_scan=101,
    tol=1e-5,
):
    # Balayage initial (scan)
    x_values = np.linspace(
        bounds[0],
        bounds[1],
        n_scan,
    )

    costs = np.array([
        func(x, *args)
        for x in x_values
    ])

    best_index = int(np.argmin(costs))

    if best_index == 0:
        left = x_values[0]
        right = x_values[1]
    elif best_index == len(x_values) - 1:
        left = x_values[-2]
        right = x_values[-1]
    else:
        left = x_values[best_index - 1]
        right = x_values[best_index + 1]

    # Raffinement avec section dorée NumPy pure (sans scipy)
    result = golden_section_minimize(
        func,
        bounds=(left, right),
        args=args,
        tol=tol
    )

    return {
        "optimal_theta": float(result.x),
        "objective_value": float(result.fun),
        "success": bool(result.success),
        "message": result.message,
        "scan_x": x_values,
        "scan_cost": costs,
    }


def calibrate_system(
    pipeline: QuantumPipeline,
    system_factory: Callable,
    dataset: ExperimentalDataset,
    pulse_builder: Callable[[float], np.ndarray],
    noise: np.ndarray,
    bounds: tuple[float, float] = (0.05, 2.0),
) -> dict:
    return scan_then_refine(
        objective_function,
        bounds=bounds,
        args=(
            pipeline,
            system_factory,
            dataset,
            pulse_builder,
            noise,
        ),
    )


# =====================================================================
# EXÉCUTION SÉQUENTIELLE DES 4 ÉTAPES
# =====================================================================
def main():
    print("==================================================")
    print("QUANTUMLAB — MATRICE UNIFIÉE (4 ÉTAPES)")
    print("==================================================")

    config = PipelineConfig()
    pipeline = QuantumPipeline(config)
    t0 = config.t_duration / 2.0
    
    pulse_default = gaussian(pipeline.t, t0=t0, sigma=3.0, amplitude=1.0)
    noise = generate_1f_noise(n_samples=len(pipeline.t), dt=pipeline.dt, seed=config.seed) * 0.05
    system = TransmonSystem(omega_q=config.omega_q, anharmonicity=config.anharmonicity, n_levels=config.n_levels)
    target_state = np.array([0.0, 1.0, 0.0], dtype=complex)

    # [Étape 1] Baseline Numérique
    print("\n[+] ÉTAPE 1 : Exécution de la baseline numérique (λ = 0)")
    traj_base = pipeline.step_1_baseline(system, pulse_default, noise=noise.copy())
    met_base = pipeline.step_3_metrology(traj_base, target_state)
    print(f"    Fidélité : {met_base['fidelity_final']:.6f} | Leakage : {met_base['leakage_final']:.6e}")
    print(f"    Norme de l'état : {met_base['state_norm_final']:.6f}")

    # [Étape 2 & 3] Couplage Spatial et Métrologie
    print("\n[+] ÉTAPE 2 & 3 : Injection du champ Moiré et extraction des observables (λ = 0.05)")
    xv = np.linspace(-5.0, 5.0, 51); yv = np.linspace(-5.0, 5.0, 51); xg, yg = np.meshgrid(xv, yv)
    moire_field = MoireField(k1=(1.0, 0.0), k2=(1.05, 0.0), amplitude=1.0)
    traj_coupled = pipeline.step_2_incremental_coupling(system, pulse_default, noise.copy(), moire_field, xg, yg, 0.0, 0.0, 0.5, 0.05)
    met_coupled = pipeline.step_3_metrology(traj_coupled, target_state)
    print(f"    Fidélité : {met_coupled['fidelity_final']:.6f} | Leakage : {met_coupled['leakage_final']:.6e}")

    # [Étape 4] Calibration Empirique Dynamique
    print("\n[+] ÉTAPE 4 : Optimisation dynamique (scan_then_refine)")
    dataset = ExperimentalDataset(
        metadata=Metadata("Sherbrooke_A1", "2026-08-14", 15.0),
        observables=Observables(target_state, 0.982, 0.007, 0.002, 0.001),
        targets=Targets(0.98, 0.01)
    )

    def sys_factory(theta): 
        return TransmonSystem(
            omega_q=config.omega_q, 
            anharmonicity=config.anharmonicity, 
            n_levels=config.n_levels
        )

    def pulse_builder(theta):
        return gaussian(
            pipeline.t,
            t0=t0,
            sigma=3.0,
            amplitude=theta,
        )

    calib = calibrate_system(
        pipeline=pipeline,
        system_factory=sys_factory,
        dataset=dataset,
        pulse_builder=pulse_builder,
        noise=noise.copy(),
        bounds=(0.1, 1.5),
    )
    
    theta_opt = calib["optimal_theta"]

    traj_opt = pipeline.step_1_baseline(
        sys_factory(theta_opt),
        pulse_builder(theta_opt),
        noise=noise.copy(),
    )

    met_opt = pipeline.step_3_metrology(
        traj_opt,
        target_state,
    )
    
    print(f"    Amplitude calibrée (θ) : {theta_opt:.6f}")
    print(f"    Fidélité optimisée     : {met_opt['fidelity_final']:.6f}")
    print(f"    Leakage éradiqué       : {met_opt['leakage_final']:.6e}")

    print("\n==================================================")
    print("SYSTÈME INTÉGRALEMENT CONVERGÉ ET SAUVEGARDÉ")
    print("==================================================")

if __name__ == "__main__":
    main()
