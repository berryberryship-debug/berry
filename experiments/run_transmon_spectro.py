"""Spectroscopie et ionisation du transmon."""
import sys
from pathlib import Path

# Franchissement de la frontière topologique
racine_labo = Path(__file__).resolve().parent.parent
sys.path.append(str(racine_labo))

from core.transmon import Transmon

def run_transmon_experiment():
    print("[*] Initialisation du transmon (E_C=0.25 GHz, E_J=15 GHz)...")
    
    transmon = Transmon(E_C=0.25, E_J=15.0, n_g=0.0, N_levels=10)
    
    # Spectre d'énergie
    E = transmon.niveaux_energie()
    print("--- Niveaux d'energie (GHz) ---")
    for i, e in enumerate(E[:5]):
        print(f"|{i}> : {e:.4f}")
        
    print(f"Frequence 0->1 : {transmon.frequence_01():.4f} GHz")
    print(f"Anharmonicite   : {transmon.anharmonicite():.4f} GHz")
    
    # Simulation de lecture et risque d'ionisation
    print("--- Simulation de lecture ---")
    for n_ph in [30, 80, 200]:
        resultat = transmon.simuler_lecture(n_photons=n_ph, duree=1.0)
        print(f"n_photons={n_ph:3d} | risque={resultat['risque']:6s} | ionisation={resultat['ionisation_detectee']}")

if __name__ == "__main__":
    run_transmon_experiment()
