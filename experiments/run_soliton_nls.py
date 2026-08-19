"""Expérience d'isolation et de propagation du Soliton."""
import sys
from pathlib import Path

# Franchissement de la frontière topologique
racine_labo = Path(__file__).resolve().parent.parent
sys.path.append(str(racine_labo))

from core.soliton_nls import SolitonNLSE

def run_soliton_experiment():
    print("[*] Initialisation du soliton fondamental (NLSE 1D)...")
    
    # Injection des paramètres avec la terminologie scientifique (eta, v, x0, theta0)
    soliton = SolitonNLSE(eta=1.0, v=0.5, x0=0.0, theta0=0.0)
    
    historique = []
    
    # On utilise la méthode lancer_simulation définie dans le noyau
    for step in soliton.lancer_simulation(duree_max=2.0, dt=0.5):
        historique.append(step)
        
    return historique

if __name__ == "__main__":
    resultats = run_soliton_experiment()
    print("-" * 60)
    for i, res in enumerate(resultats):
        print(f"Pas {i}: énergie={res['energie']:.4f} | masse={res['masse']:.4f} | état={res['etat']}")
