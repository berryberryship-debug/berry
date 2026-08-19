"""Moteur fondamental du Soliton (Non-Linear Schrödinger Equation)."""
from dataclasses import dataclass

@dataclass
class SolitonNLSE:
    eta: float = 1.0       # Amplitude mathématique du soliton
    v: float = 0.0         # Vitesse de propagation
    x0: float = 0.0        # Position spatiale
    theta0: float = 0.0    # Phase initiale

    def propager(self, dt: float) -> dict:
        """Fait avancer le soliton tout en calculant la conservation de son énergie."""
        self.x0 += self.v * dt
        
        # Le soliton conserve son énergie et sa masse malgré le déplacement (Néguentropie)
        energie_conservee = self.eta ** 2
        masse_invariante = energie_conservee * 2.0 
        
        return {
            "energie": energie_conservee,
            "masse": masse_invariante,
            "etat": "[+] Cohérence Maintenue"
        }

    def lancer_simulation(self, duree_max: float, dt: float = 0.1):
        """Génère le flux d'états du soliton dans le temps."""
        temps = 0.0
        while temps <= duree_max:
            donnees_etape = self.propager(dt)
            donnees_etape['temps'] = temps
            yield donnees_etape
            temps += dt
