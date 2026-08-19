"""Visualisation de la propagation du soliton NLSE."""
import sys
from pathlib import Path

racine_labo = Path(__file__).resolve().parent.parent
sys.path.append(str(racine_labo))

from core.soliton_nls import SolitonNLSE

def collecter_trajectoire(duree_max: float = 2.0, dt: float = 0.1):
    soliton = SolitonNLSE(eta=1.0, v=0.5, x0=0.0, theta0=0.0)
    temps, positions, energies = [], [], []
    
    for step in soliton.lancer_simulation(duree_max=duree_max, dt=dt):
        temps.append(step['temps'])
        positions.append(soliton.x0)
        energies.append(step['energie'])
    
    return temps, positions, energies

if __name__ == "__main__":
    t, x, e = collecter_trajectoire()
    print("Trajectoire collectée :")
    for i in range(len(t)):
        print(f"t={t[i]:.1f} | x={x[i]:.2f} | E={e[i]:.4f}")
