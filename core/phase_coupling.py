from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator


@dataclass
class Oscillateur:
    frequence: float
    phase: float = 0.0
    amplitude: float = 1.0

    def propager(self, dt: float) -> float:
        if dt <= 0:
            raise ValueError("dt doit être positif.")
        omega = 2.0 * math.pi * self.frequence
        self.phase = (self.phase + omega * dt) % (2.0 * math.pi)
        return self.amplitude * math.sin(self.phase)


@dataclass
class MatriceCouplage:
    osc_1: Oscillateur
    osc_2: Oscillateur
    entropie_systeme: float = 0.0
    seuil_resonance: float = 0.05

    def evaluer_fenetre(self, dt: float) -> dict:
        self.osc_1.propager(dt)
        self.osc_2.propager(dt)

        delta_phase = abs(
            math.sin(self.osc_1.phase - self.osc_2.phase)
        )

        if delta_phase <= self.seuil_resonance:
            etat = "COINCIDENCE MAXIMALE : SAUT TOPOLOGIQUE"
            self.entropie_systeme = max(
                0.0,
                self.entropie_systeme - 0.5 * dt,
            )
            saut = True
        else:
            etat = "DECALAGE DE PHASE : DISSIPATION"
            self.entropie_systeme += 0.2 * dt
            saut = False

        return {
            "delta_phase": delta_phase,
            "entropie": self.entropie_systeme,
            "etat": etat,
            "saut_actif": saut,
        }

    def lancer_simulation(
        self,
        duree_max: float,
        dt: float = 0.1,
    ) -> Iterator[dict]:
        if duree_max < 0 or dt <= 0:
            raise ValueError("duree_max doit être >= 0 et dt > 0.")

        temps = 0.0
        while temps < duree_max:
            observation = self.evaluer_fenetre(dt)
            observation["temps"] = temps
            yield observation
            temps += dt


if __name__ == "__main__":
    print("[*] Initialisation du couplage de phases")

    moteur = MatriceCouplage(
        osc_1=Oscillateur(frequence=1.0),
        osc_2=Oscillateur(frequence=1.05),
    )

    print("-" * 75)
    print(
        f"{'Temps':<10} | {'Delta phase':<15} | "
        f"{'Entropie':<12} | Statut"
    )
    print("-" * 75)

    for observation in moteur.lancer_simulation(
        duree_max=4.0,
        dt=0.1,
    ):
        print(
            f"{observation['temps']:<10.2f} | "
            f"{observation['delta_phase']:<15.4f} | "
            f"{observation['entropie']:<12.4f} | "
            f"{observation['etat']}"
        )
