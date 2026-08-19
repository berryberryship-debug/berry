#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bloc 11 : Optimisation Quantique
Fonction de coût basée strictement sur la fidélité et le leakage.
"""

def pulse_builder_factory(pipeline, t0):
    def build(theta):
        return gaussian(pipeline.t, t0=t0, sigma=3.0, amplitude=theta)
    return build

def objective_function(theta, pipeline, system_factory, dataset, pulse_builder, noise):
    system = system_factory(theta)
    pulse = pulse_builder(theta)
    trajectory = pipeline.step_1_baseline(system, pulse, noise=noise.copy())
    metrics = pipeline.step_3_metrology(trajectory, dataset.observables.target_state)

    fidelity = metrics["fidelity_final"]
    leakage = metrics["leakage_final"]

    sigma_fid = max(dataset.observables.fidelity_uncertainty, 1e-6)
    sigma_leak = max(dataset.observables.leakage_uncertainty, 1e-6)

    fidelity_deficit = max(0.0, dataset.targets.target_fidelity - fidelity)
    leakage_excess = max(0.0, leakage - dataset.targets.target_leakage_max)

    return float((fidelity_deficit / sigma_fid) ** 2 + (leakage_excess / sigma_leak) ** 2)
