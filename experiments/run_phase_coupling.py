"""Expérience de couplage de phase (résonance transitoire)."""
import sys
from pathlib import Path

racine_labo = Path(__file__).resolve().parent.parent
sys.path.append(str(racine_labo))

from core.phase_coupling import Oscillateur, MatriceCouplage

def run_coupling_experiment():
    print("[*] Lancement de l'expérience de résonance depuis le sas expérimental...")
    
    diapason_A = Oscillateur(frequence=1.0)
    diapason_B = Oscillateur(frequence=1.10) 
    moteur = MatriceCouplage(osc_1=diapason_A, osc_2=diapason_B)
    
    historique = []
    for step in moteur.lancer_simulation(duree_max=1.0, dt=0.2):
        historique.append(step)
        
    return historique

if __name__ == "__main__":
    resultats = run_coupling_experiment()
    print("-" * 50)
    for res in resultats:
        print(f"ΔPhase: {res['delta_phase']:<8.4f} | ΔS: {res['entropie']:<8.4f} | {res['etat']}")
