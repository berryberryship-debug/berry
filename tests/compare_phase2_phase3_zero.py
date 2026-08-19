#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from core.transmon import TransmonSystem
from core.pulses import gaussian
from core.noise import generate_1f_noise
from core.metrics import state_fidelity


def main():

    print("==============================================")
    print("COMPARAISON PHASE 2 / PHASE 3 — λ = 0")
    print("==============================================")

    omega_q = 5.0
    anharmonicity = -0.3
    n_levels = 3

    t_duration = 20.0
    n_steps = 1000

    pulse_sigma = 3.0
    pulse_amplitude = 0.5

    seed = 20260813

    t = np.linspace(
        0.0,
        t_duration,
        n_steps,
    )

    dt = t[1] - t[0]
    t0 = t_duration / 2.0

    pulse = gaussian(
        t,
        t0=t0,
        sigma=pulse_sigma,
        amplitude=pulse_amplitude,
    )

    noise = (
        generate_1f_noise(
            n_samples=len(t),
            dt=dt,
            seed=seed,
        )
        * 0.05
    )

    system = TransmonSystem(
        omega_q=omega_q,
        anharmonicity=anharmonicity,
        n_levels=n_levels,
    )

    target = np.array(
        [0.0, 1.0, 0.0],
        dtype=complex,
    )

    # --------------------------------------------------
    # PHASE 2
    # --------------------------------------------------

    states_p2 = system.simulate(
        t,
        pulse,
        noise=noise,
    )

    psi_p2 = states_p2[:, -1]

    fid_p2 = float(
        state_fidelity(
            target,
            psi_p2,
        )
    )

    leak_p2 = float(
        abs(psi_p2[2]) ** 2
    )

    # --------------------------------------------------
    # PHASE 3 — λ = 0
    # --------------------------------------------------

    zero_detuning = np.zeros_like(t)

    states_p3 = system.simulate_with_custom_detuning(
        t,
        pulse,
        noise=noise,
        detuning=zero_detuning,
    )

    psi_p3 = states_p3[:, -1]

    fid_p3 = float(
        state_fidelity(
            target,
            psi_p3,
        )
    )

    leak_p3 = float(
        abs(psi_p3[2]) ** 2
    )

    # --------------------------------------------------
    # Comparaison
    # --------------------------------------------------

    state_difference = np.linalg.norm(
        psi_p2 - psi_p3
    )

    print()
    print("PHASE 2 — simulate()")
    print(f"    Fidélité : {fid_p2:.12f}")
    print(f"    Leakage  : {leak_p2:.12f}")

    print()
    print("PHASE 3 — simulate_with_custom_detuning(), λ=0")
    print(f"    Fidélité : {fid_p3:.12f}")
    print(f"    Leakage  : {leak_p3:.12f}")

    print()
    print("DIFFÉRENCE")
    print(f"    ΔF       : {fid_p3 - fid_p2:+.12e}")
    print(f"    ΔLeakage : {leak_p3 - leak_p2:+.12e}")
    print(f"    ||ψ2-ψ3||: {state_difference:.12e}")

    print()

    if np.allclose(
        psi_p2,
        psi_p3,
        rtol=1e-12,
        atol=1e-12,
    ):
        print("PASS — λ=0 est numériquement identique.")
    else:
        print("FAIL — λ=0 n'est PAS identique à Phase 2.")

    print("==============================================")


if __name__ == "__main__":
    main()
