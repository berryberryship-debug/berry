#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QuantumLab — Modèle Transmon et Évolution Temporelle
"""

import numpy as np

class TransmonSystem:
    def __init__(self, omega_q: float = 5.0, anharmonicity: float = -0.3, n_levels: int = 3):
        self.omega_q = omega_q
        self.anharmonicity = anharmonicity
        self.n_levels = n_levels
        
        # Opérateurs de base (Hamiltonien non perturbé)
        self.H0 = np.diag([n * self.omega_q + 0.5 * n * (n - 1) * self.anharmonicity for n in range(n_levels)])
        self.n_op = np.diag([np.sqrt(n) for n in range(1, n_levels)], k=1) + np.diag([np.sqrt(n) for n in range(1, n_levels)], k=-1)

    def hamiltonian(self, control_field: complex, noise_field: float = 0.0) -> np.ndarray:
        omega_eff = self.omega_q + noise_field
        H_sys = np.diag([n * omega_eff + 0.5 * n * (n - 1) * self.anharmonicity for n in range(self.n_levels)])
        H_drive = 0.5 * control_field * self.n_op + 0.5 * np.conj(control_field) * self.n_op
        return H_sys + H_drive

    def simulate(self, t: np.ndarray, pulse: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
        dt = t[1] - t[0]
        n_steps = len(t)
        psi = np.zeros(self.n_levels, dtype=complex)
        psi[0] = 1.0  # état fondamental |0>
        
        if noise is None:
            noise = np.zeros_like(t)
            
        states = np.zeros((self.n_levels, n_steps), dtype=complex)
        states[:, 0] = psi
        
        for i in range(1, n_steps):
            H = self.hamiltonian(pulse[i-1], noise[i-1])
            # Évolution unitaire par approximation de Cayley ou Runge-Kutta ordre 2
            dpsi = -1j * np.dot(H, psi)
            psi = psi + dt * dpsi
            psi = psi / np.linalg.norm(psi)  # Normalisation
            states[:, i] = psi
            
        return states


    def simulate_with_custom_detuning(
        self,
        t,
        pulse,
        noise=None,
        detuning=None,
    ):
        """
        Évolution avec une modulation de détuning externe.

        IMPORTANT :
        Cette méthode est une extension de Phase 3.
        La méthode simulate() existante reste inchangée.

        detuning[i] représente une modulation supplémentaire
        de fréquence à l'instant t[i].
        """

        import numpy as np

        t = np.asarray(t, dtype=float)
        pulse = np.asarray(pulse, dtype=float)

        if t.ndim != 1:
            raise ValueError("t doit être un vecteur 1D.")

        if pulse.shape != t.shape:
            raise ValueError(
                "pulse et t doivent avoir la même longueur."
            )

        if noise is None:
            noise = np.zeros_like(t)
        else:
            noise = np.asarray(noise, dtype=float)

        if noise.shape != t.shape:
            raise ValueError(
                "noise et t doivent avoir la même longueur."
            )

        if detuning is None:
            detuning = np.zeros_like(t)
        else:
            detuning = np.asarray(detuning, dtype=float)

        if detuning.shape != t.shape:
            raise ValueError(
                "detuning et t doivent avoir la même longueur."
            )

        # Cette méthode dépend de l'implémentation interne existante.
        # On vérifie d'abord si le système expose une fonction
        # d'évolution personnalisable.

        # Opérateur nombre :
        #
        #     N = diag(0, 1, 2, ...)
        #
        # Une modulation de fréquence du transmon agit
        # diagonalement sur les niveaux.
        number_operator = np.diag(
            np.arange(self.n_levels, dtype=float)
        )

        states = np.zeros(
            (self.n_levels, len(t)),
            dtype=complex,
        )

        psi = np.zeros(
            self.n_levels,
            dtype=complex,
        )
        psi[0] = 1.0

        states[:, 0] = psi

        for i in range(1, len(t)):

            # Hamiltonien de base : exactement la même
            # construction que dans simulate().
            H = self.hamiltonian(
                pulse[i - 1],
                noise[i - 1],
            )

            # Terme de modulation externe :
            #
            #     H_Moire = detuning(t) * N
            #
            # Ici detuning(t) = lambda * M_eff(t).
            H = H + detuning[i - 1] * number_operator

            # Même schéma temporel explicite que simulate().
            dpsi = -1j * np.dot(H, psi)

            psi = psi + (t[i] - t[i - 1]) * dpsi

            # Conservation numérique de la norme.
            norm = np.linalg.norm(psi)

            if norm == 0.0:
                raise FloatingPointError(
                    "Norme d'état nulle pendant l'évolution."
                )

            psi = psi / norm

            states[:, i] = psi

        return states
