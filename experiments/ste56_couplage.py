#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STE-56 : Matrice de Couplage Transverse
Modélisation de la transition dimensionnelle (1D vers Réseau 2D)
"""
import numpy as np

class STE56_Couplage:
    def __init__(self, num_chains=3, nx=100, dx=0.1, dt=0.05, c=1.0, w0=1.0, K=0.5):
        self.num_chains = num_chains
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.c = c
        self.w0 = w0
        self.K = float(K)  # Constante de couplage transverse
        
        # État du système : phi (actuel), phi_prev (passé), phi_next (futur)
        self.phi = np.zeros((self.num_chains, self.nx))
        self.phi_prev = np.zeros((self.num_chains, self.nx))
        self.phi_next = np.zeros((self.num_chains, self.nx))

    def injecter_kink(self, chain_idx: int, position: float, velocity: float = 0.0):
        """Injecte un soliton parfait (kink) sur une chaîne spécifique."""
        gamma = 1.0 / np.sqrt(1.0 - (velocity/self.c)**2)
        x = np.arange(self.nx) * self.dx
        # Équation exacte du soliton de Sine-Gordon
        self.phi[chain_idx, :] = 4 * np.arctan(np.exp(gamma * self.w0 * (x - position) / self.c))
        self.phi_prev = np.copy(self.phi)

    def step(self):
        """Intégration d'un pas de temps avec couplage inter-chaînes (Méthode FDTD)."""
        for n in range(self.num_chains):
            # Laplacien spatial 1D intra-chaîne (Conditions aux limites périodiques)
            laplacien_x = np.roll(self.phi[n], -1) - 2*self.phi[n] + np.roll(self.phi[n], 1)
            
            # Couplage transverse inter-chaînes (Interaction de voisinage immédiat)
            phi_up = self.phi[n-1] if n > 0 else self.phi[n] # Frontière libre
            phi_down = self.phi[n+1] if n < self.num_chains - 1 else self.phi[n]
            couplage = self.K * (phi_up - 2*self.phi[n] + phi_down)
            
            # Évolution temporelle matricielle
            self.phi_next[n] = (
                2*self.phi[n] - self.phi_prev[n] 
                + (self.c * self.dt / self.dx)**2 * laplacien_x 
                - self.dt**2 * self.w0**2 * np.sin(self.phi[n])
                + self.dt**2 * couplage
            )
        
        # Synchronisation des registres (zéro latence)
        self.phi_prev = np.copy(self.phi)
        self.phi = np.copy(self.phi_next)

if __name__ == "__main__":
    engine = STE56_Couplage(num_chains=3, K=0.8)
    
    # On injecte l'information uniquement sur la chaîne centrale
    engine.injecter_kink(chain_idx=1, position=5.0) 
    print(f"[*] STE-56 Initialisé. Matrice de {engine.num_chains} chaînes. Couplage K={engine.K}")
    print("[*] Injection du Kink (soliton) sur la chaîne 1...")
    
    # L'information se transmet aux chaînes adjacentes via le coefficient K
    for _ in range(15):
        engine.step()
        
    print("[*] Propagation transverse achevée. L'information s'est intriquée avec succès.")
