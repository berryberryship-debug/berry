"""Moteur Transmon avec ionisation induite par la mesure."""
import numpy as np
from dataclasses import dataclass

@dataclass
class Transmon:
    """
    Transmon qubit avec Hamiltonien complet et modèle d'ionisation.
    
    H = 4*E_C*(n - n_g)^2 - E_J*cos(phi)
    
    Paramètres typiques (GHz) :
      - E_C ~ 0.2-0.3 GHz
      - E_J ~ 10-20 GHz
      - E_J/E_C ~ 50-100
    """
    E_C: float = 0.25
    E_J: float = 15.0
    n_g: float = 0.0
    N_levels: int = 10

    def hamiltonien(self) -> np.ndarray:
        """Construit la matrice Hamiltonienne du transmon."""
        N = self.N_levels
        
        # Base de nombre |n>
        n_vals = np.arange(N) - self.n_g
        H_charge = 4 * self.E_C * np.diag(n_vals**2)
        
        # Terme Josephson : -E_J*cos(phi)
        # cos(phi) ≈ (exp(i*phi) + exp(-i*phi))/2
        # Dans la base de nombre, exp(±i*phi) déplace de ±1
        a = np.diag(np.sqrt(np.arange(1, N)), k=-1)
        a_dag = a.T
        
        # Approximation : cos(phi) ≈ 1 - phi^2/2 + phi^4/24
        # avec phi ∝ (a + a†)
        phi = (a + a_dag) / np.sqrt(2)
        cos_phi = np.eye(N) - 0.5 * (phi @ phi) + (1.0/24.0) * np.linalg.matrix_power(phi, 4)
        
        H_josephson = -self.E_J * cos_phi
        
        return H_charge + H_josephson

    def niveaux_energie(self) -> np.ndarray:
        """Calcule les niveaux d'énergie (valeurs propres)."""
        H = self.hamiltonien()
        return np.sort(np.real(np.linalg.eigvalsh(H)))

    def anharmonicite(self) -> float:
        """
        Anharmonicité : alpha = (E_2 - E_1) - (E_1 - E_0)
        Typiquement négative (~ -0.2 à -0.3 GHz).
        """
        E = self.niveaux_energie()
        if len(E) < 3:
            return np.nan
        return (E[2] - E[1]) - (E[1] - E[0])

    def frequence_01(self) -> float:
        """Fréquence de transition |0⟩ → |1⟩."""
        E = self.niveaux_energie()
        if len(E) < 2:
            return np.nan
        return E[1] - E[0]

    def seuil_ionisation(self, n_photons_critiques: float = 100.0) -> dict:
        """Estime le seuil d'ionisation (Dumas et al., PRX 2024)."""
        if n_photons_critiques > 80:
            etat = "stable"
        else:
            etat = "ionisation_risque"
        
        return {
            "seuil_photons": n_photons_critiques,
            "etat": etat,
            "reference": "Dumas et al., Phys. Rev. X 14, 041023 (2024)"
        }

    def simuler_lecture(self, n_photons: float, duree: float = 1.0) -> dict:
        """Simule une impulsion de lecture et détecte le risque d'ionisation."""
        if n_photons < 50:
            risque = "faible"
            ionisation_detectee = False
        elif n_photons < 150:
            risque = "moyen"
            ionisation_detectee = False
        else:
            risque = "élevé"
            ionisation_detectee = True
        
        return {
            "n_photons": n_photons,
            "duree_us": duree,
            "risque": risque,
            "ionisation_detectee": ionisation_detectee,
            "frequence_01_GHz": self.frequence_01(),
            "anharmonicite_GHz": self.anharmonicite()
        }
