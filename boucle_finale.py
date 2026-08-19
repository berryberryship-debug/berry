from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import numpy as np


# ============================================================
# LANGAGE INTERNE
# ============================================================

@dataclass
class Description:
    predicate: str
    args: tuple[str, ...] = ()
    meta: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"


@dataclass
class EtatLinguistique:
    descriptions: list[Description] = field(
        default_factory=list
    )

    invariants: set[str] = field(
        default_factory=set
    )

    def ajouter(self, description: Description) -> None:
        self.descriptions.append(description)

        # Mémoire limitée pour éviter l'explosion
        if len(self.descriptions) > 200:
            self.descriptions = self.descriptions[-200:]

        compteurs = Counter(
            d.predicate
            for d in self.descriptions
        )

        self.invariants = {
            predicat
            for predicat, nombre in compteurs.items()
            if nombre >= 3
        }

    def realite(self) -> list[str]:
        return sorted(self.invariants)


def traduire_entree(entree: str) -> Description:
    mots = (
        entree.lower()
        .replace(",", "")
        .split()
    )

    if not mots:
        return Description("vide")

    return Description(
        predicate=mots[0],
        args=tuple(mots[1:]),
        meta={"source": "observation"}
    )


# ============================================================
# CONTROLEUR DE RAISON D'ÊTRE
# ============================================================

@dataclass
class Finalite:
    """
    Finalité opérationnelle du système.

    Le système cherche à :
    - maintenir sa cohérence ;
    - préserver son intégrité ;
    - limiter sa dispersion ;
    - récupérer après un choc.
    """

    coherence_cible: float = 0.30
    dispersion_max: float = 0.80
    integrite_min: float = 0.70

    def evaluer(
        self,
        coherence: float,
        dispersion: float,
        integrite: float
    ) -> float:

        erreur_coherence = abs(
            coherence - self.coherence_cible
        )

        penalite_dispersion = max(
            0.0,
            dispersion - self.dispersion_max
        )

        penalite_integrite = max(
            0.0,
            self.integrite_min - integrite
        )

        score = (
            1.0
            - erreur_coherence
            - penalite_dispersion
            - penalite_integrite
        )

        return max(0.0, min(1.0, score))

    def produire_action(
        self,
        coherence: float,
        dispersion: float,
        integrite: float
    ) -> np.ndarray:

        # Action douce : retour vers une zone viable
        correction = np.zeros(2)

        if dispersion > self.dispersion_max:
            correction -= 0.08

        if integrite < self.integrite_min:
            correction -= 0.04

        if coherence < self.coherence_cible:
            correction += 0.03

        return correction


# ============================================================
# RESEAU DYNAMIQUE
# ============================================================

class Systeme:

    def __init__(
        self,
        nombre_noeuds: int = 32,
        duree: int = 600,
        choc: int = 300
    ):
        self.N = nombre_noeuds
        self.T = duree
        self.choc = choc

        self.rng = np.random.default_rng(7)

        self.x = self.rng.normal(
            0,
            1,
            (self.N, 2)
        )

        self.W = self.rng.uniform(
            0.0,
            0.30,
            (self.N, self.N)
        )

        np.fill_diagonal(self.W, 0.0)

        self.memoire = np.zeros(
            (self.N, self.N)
        )

        self.langage = EtatLinguistique()
        self.finalite = Finalite()

        self.historique = []

    def mesurer(self) -> tuple[float, float, float]:

        distances = np.linalg.norm(
            self.x[:, None, :] - self.x[None, :, :],
            axis=2
        )

        coherence = float(
            np.mean(np.exp(-distances))
        )

        dispersion = float(
            np.mean(np.linalg.norm(self.x, axis=1))
        )

        integrite = float(
            np.mean(self.W > 0.05)
        )

        return coherence, dispersion, integrite

    def observer(self, instant: int) -> None:

        coherence, dispersion, integrite = (
            self.mesurer()
        )

        entrees = []

        if coherence > 0.25:
            entrees.append(
                "coherence reseau"
            )

        if dispersion > 0.80:
            entrees.append(
                "dispersion reseau"
            )

        if integrite > 0.20:
            entrees.append(
                "integrite liens"
            )

        if instant == self.choc:
            entrees.append(
                "choc reseau"
            )

        if not entrees:
            entrees.append(
                "etat indetermine"
            )

        for entree in entrees:
            self.langage.ajouter(
                traduire_entree(entree)
            )

    def appliquer_finalite(self) -> None:

        coherence, dispersion, integrite = (
            self.mesurer()
        )

        action = self.finalite.produire_action(
            coherence,
            dispersion,
            integrite
        )

        self.x += action

    def etape(self, instant: int) -> None:

        # Choc local
        if instant == self.choc:
            bloc = [2, 3, 4, 7, 11]

            self.x[bloc] += self.rng.normal(
                0,
                2.5,
                (len(bloc), 2)
            )

        degre = (
            self.W.sum(axis=1, keepdims=True)
            + 1e-9
        )

        voisinage = (
            self.W @ self.x
        ) / degre

        dx = (
            -0.20 * self.x
            + 0.35 * (voisinage - self.x)
        )

        bruit = self.rng.normal(
            0,
            0.025,
            self.x.shape
        )

        self.x += 0.04 * dx + bruit

        # Mémoire relationnelle
        distances = np.linalg.norm(
            self.x[:, None, :] - self.x[None, :, :],
            axis=2
        )

        correlations = np.exp(-distances)

        self.memoire = (
            0.995 * self.memoire
            + 0.005 * correlations
        )

        # Topologie adaptative
        self.W += 0.04 * (
            0.35 * self.memoire
            - 0.15 * self.W
        )

        self.W = np.clip(
            self.W,
            0.0,
            1.0
        )

        # Fermeture langage → état → action
        if instant % 10 == 0 or instant == self.choc:
            self.observer(instant)
            self.appliquer_finalite()

        coherence, dispersion, integrite = (
            self.mesurer()
        )

        score = self.finalite.evaluer(
            coherence,
            dispersion,
            integrite
        )

        self.historique.append({
            "temps": instant,
            "coherence": coherence,
            "dispersion": dispersion,
            "integrite": integrite,
            "score": score,
            "invariants": len(
                self.langage.invariants
            )
        })

    def executer(self) -> None:

        for instant in range(self.T):
            self.etape(instant)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    systeme = Systeme()
    systeme.executer()

    dernier = systeme.historique[-1]

    print()
    print("===== BOUCLE FERMÉE =====")
    print()

    print(
        "Raison d'être :",
        "maintenir cohérence, intégrité et viabilité"
    )

    print()
    print("État final :")
    print(
        " - cohérence :",
        round(dernier["coherence"], 5)
    )
    print(
        " - dispersion :",
        round(dernier["dispersion"], 5)
    )
    print(
        " - intégrité :",
        round(dernier["integrite"], 5)
    )
    print(
        " - score de viabilité :",
        round(dernier["score"], 5)
    )
    print(
        " - invariants linguistiques :",
        sorted(systeme.langage.invariants)
    )
    print(
        " - descriptions mémorisées :",
        len(systeme.langage.descriptions)
    )

    print()
    print("Dernières observations :")

    for observation in systeme.langage.descriptions[-8:]:
        print(" -", observation)

    print()
    print("Test terminé.")
