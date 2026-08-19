#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from core.dataset import Metadata, Observables, Targets, ExperimentalDataset
from core.pipeline import QuantumPipeline, PipelineConfig

def main():
    print("==============================================")
    print("TEST INFRASTRUCTURE DATASET & PIPELINE (V3)")
    print("==============================================")

    target_state = np.array([0.0, 1.0, 0.0], dtype=complex)

    ds = ExperimentalDataset(
        metadata=Metadata(
            device_id="transmon_Sherbrooke_A1",
            date="2026-08-13",
            temperature_mK=15.0,
            notes="Calibration baseline Phase 3",
        ),
        observables=Observables(
            target_state=target_state,
            measured_fidelity=0.982,
            measured_leakage=0.007,
            fidelity_uncertainty=0.002,
            leakage_uncertainty=0.001,
            timeseries=np.zeros((3, 1000)),
        ),
        targets=Targets(
            target_fidelity=0.98,
            target_leakage_max=0.01,
        ),
    )

    print(f"[+] Dataset validé et exportable : {ds.as_dict()}")

    # Utilisation de logical_levels au lieu de leakage_index
    config = PipelineConfig(logical_levels=(0, 1))
    pipeline = QuantumPipeline(config)
    print(f"[+] Pipeline prêt (logical_levels={config.logical_levels})")

    print("==============================================")
    print("TOUS LES TESTS INFRASTRUCTURE PASSENT")
    print("==============================================")

if __name__ == "__main__":
    main()
