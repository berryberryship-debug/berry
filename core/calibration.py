#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from typing import Callable
from core.dataset import ExperimentalDataset
from core.pipeline import QuantumPipeline

def objective_function(theta: float, pipeline: QuantumPipeline, system_factory: Callable[[float], object], dataset: ExperimentalDataset, pulse: np.ndarray, noise: np.ndarray) -> float:
    system = system_factory(theta)
    trajectory = pipeline.step_1_baseline(system, pulse, noise=noise.copy())
    metrics = pipeline.step_3_metrology(trajectory, dataset.observables.target_state)

    sigma_fid = max(dataset.observables.fidelity_uncertainty, 1e-6)
    sigma_leak = max(dataset.observables.leakage_uncertainty, 1e-6)

    fid_err = ((metrics["fidelity_final"] - dataset.targets.target_fidelity) / sigma_fid) ** 2
    leak_err = ((metrics["leakage_final"] - dataset.targets.target_leakage_max) / sigma_leak) ** 2

    return float(fid_err + leak_err)

def golden_section_minimize(func: Callable, bounds: tuple[float, float], args: tuple = (), tol: float = 1e-5, maxiter: int = 50) -> dict:
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

    return {
        "optimal_theta": xmin,
        "objective_value": fmin,
        "success": True,
        "message": "Calibration convergée."
    }

def calibrate_system(pipeline: QuantumPipeline, system_factory: Callable[[float], object], dataset: ExperimentalDataset, pulse: np.ndarray, noise: np.ndarray, bounds: tuple[float, float] = (0.05, 2.0)) -> dict:
    return golden_section_minimize(objective_function, bounds=bounds, args=(pipeline, system_factory, dataset, pulse, noise))

def evaluate_calibration(optimal_theta: float, pipeline: QuantumPipeline, system_factory: Callable[[float], object], dataset: ExperimentalDataset, pulse: np.ndarray, noise: np.ndarray) -> dict:
    system = system_factory(optimal_theta)
    trajectory = pipeline.step_1_baseline(system, pulse, noise=noise.copy())
    metrics = pipeline.step_3_metrology(trajectory, dataset.observables.target_state)

    return {
        "theta": optimal_theta,
        "fidelity_sim": metrics["fidelity_final"],
        "fidelity_target": dataset.targets.target_fidelity,
        "leakage_sim": metrics["leakage_final"],
        "leakage_target_max": dataset.targets.target_leakage_max,
        "populations": metrics["populations_final"].tolist()
    }
