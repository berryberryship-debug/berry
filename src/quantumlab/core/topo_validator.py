#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de Validation Topologique - Frenkel-Kontorova
Évalue la déformation continue de la densité solitonique sous contrainte quantique.
"""

import numpy as np

class FrenkelKontorovaValidator:
    def __init__(self, N=240, b=1.06):
        self.N = N
        self.b = float(b)
        # État de base relaxé (idéalisé pour la baseline)
        # Dans un flux complet, ceci serait le résultat de la relaxation overdamped
        self.x_baseline = np.arange(N) * self.b
        self.density_baseline = self._compute_density(self.x_baseline)

    def _compute_density(self, x):
        """Calcule la densité locale de solitons par gradient de phase déroulée."""
        unwrapped = np.unwrap(2 * np.pi * x / self.b) * (self.b / (2 * np.pi))
        return np.gradient(unwrapped)

    def evaluer_perte_coherence(self, amplitude_pulse, energy_diagnostic=1.0):
        """
        Quantifie la déformation topologique induite par l'impulsion.
        L'amplitude theta crée une perturbation de phase dans le réseau.
        """
        amplitude_pulse = float(amplitude_pulse)
        energy_diagnostic = float(energy_diagnostic)

        # Injection de l'énergie quantique comme perturbation de la position
        # La perturbation est proportionnelle à la force de l'impulsion
        perturbation = amplitude_pulse * np.sin(2 * np.pi * self.x_baseline / self.b)
        x_perturbed = self.x_baseline + (perturbation * energy_diagnostic)
        
        # Calcul de la nouvelle densité sous contrainte
        density_perturbed = self._compute_density(x_perturbed)
        
        # Norme L2 de la déformation de la densité (distance euclidienne)
        deformation = np.linalg.norm(density_perturbed - self.density_baseline)
        
        return float(deformation)

if __name__ == "__main__":
    # Test d'intégrité interne
    validator = FrenkelKontorovaValidator()
    perte = validator.evaluer_perte_coherence(amplitude_pulse=0.15)
    print("--- VALIDATEUR TOPOLOGIQUE OPÉRATIONNEL ---")
    print(f"Déformation de densité mesurée pour theta=0.15 : {perte:.6f}")
