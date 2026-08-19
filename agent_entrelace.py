from __future__ import annotations

from dataclasses import dataclass, field
import math
import random


# ============================================================
# MILIEU
# ============================================================

@dataclass
class Milieu:
    lumiere: float = 0.5
    mouvement: float = 0.0
    danger: float = 0.0
    ressource: float = 0.5
    distance_obstacle: float = 1.0

    def perturber(self, instant: int) -> None:
        """
        Introduit des événements sensoriels dans le milieu.
        """

        if instant == 20:
            self.lumiere = 0.9
            self.ressource = 0.8

        if instant == 45:
            self.mouvement = 0.9
            self.danger = 1.0
            self.distance_obstacle = 0.2

        if instant == 70:
            self.mouvement = 0.1
            self.danger = 0.0
            self.distance_obstacle = 0.9

        if instant == 95:
            self.lumiere = 0.4
            self.ressource = 0.3


# ============================================================
# CAPTEURS
# ============================================================

@dataclass
class EtatSensoriel:
    lumiere: float
    mouvement: float
    danger: float
    ressource: float
    distance_obstacle: float


def capter(milieu: Milieu) -> EtatSensoriel:
    """
    Transforme le milieu en signaux sensoriels.
    Un léger bruit représente l'imperfection des capteurs.
    """

    def bruit(valeur: float, intensite: float = 0.02) -> float:
        return max(
            0.0,
            min(
                1.0,
                valeur + random.uniform(
                    -intensite,
                    intensite
                )
            )
        )

    return EtatSensoriel(
        lumiere=bruit(milieu.lumiere),
        mouvement=bruit(milieu.mouvement),
        danger=bruit(milieu.danger),
        ressource=bruit(milieu.ressource),
        distance_obstacle=bruit(
            milieu.distance_obstacle
        ),
    )


# ============================================================
# VALENCE INTERNE
# ============================================================

@dataclass
class EtatInterne:
    valence: float = 0.0
    energie: float = 0.7
    securite: float = 1.0
    tension: float = 0.0
    memoire: list[str] = field(
        default_factory=list
    )


def evaluer_sensorialite(
    sensoriel: EtatSensoriel,
    interne: EtatInterne
) -> None:
    """
    Transforme les perceptions en valence interne.
    """

    plaisir = (
        0.35 * sensoriel.ressource
        + 0.20 * sensoriel.lumiere
        + 0.20 * sensoriel.distance_obstacle
    )

    menace = (
        0.60 * sensoriel.danger
        + 0.25 * sensoriel.mouvement
        + 0.30 * (1.0 - sensoriel.distance_obstacle)
    )

    interne.valence = max(
        -1.0,
        min(
            1.0,
            plaisir - menace
        )
    )

    interne.tension = max(
        0.0,
        min(
            1.0,
            menace
        )
    )

    interne.securite = max(
        0.0,
        min(
            1.0,
            1.0 - menace
        )
    )

    interne.energie += (
        0.03 * sensoriel.ressource
        - 0.04 * menace
    )

    interne.energie = max(
        0.0,
        min(
            1.0,
            interne.energie
        )
    )


# ============================================================
# LANGAGE INTERNE
# ============================================================

def produire_descriptions(
    sensoriel: EtatSensoriel,
    interne: EtatInterne
) -> list[str]:

    descriptions = []

    if sensoriel.lumiere > 0.7:
        descriptions.append("lumiere forte")
    elif sensoriel.lumiere < 0.3:
        descriptions.append("lumiere faible")

    if sensoriel.mouvement > 0.7:
        descriptions.append("mouvement rapide")

    if sensoriel.danger > 0.7:
        descriptions.append("danger proche")

    if sensoriel.ressource > 0.7:
        descriptions.append("ressource accessible")

    if sensoriel.distance_obstacle < 0.3:
        descriptions.append("obstacle proche")

    if interne.valence > 0.25:
        descriptions.append("etat favorable")

    if interne.valence < -0.25:
        descriptions.append("etat defavorable")

    if interne.tension > 0.7:
        descriptions.append("tension elevee")

    if interne.securite > 0.7:
        descriptions.append("securite retrouvee")

    if not descriptions:
        descriptions.append("etat neutre")

    return descriptions


# ============================================================
# ACTION
# ============================================================

def choisir_action(
    sensoriel: EtatSensoriel,
    interne: EtatInterne
) -> str:

    if sensoriel.danger > 0.7:
        return "fuir"

    if sensoriel.distance_obstacle < 0.3:
        return "eviter_obstacle"


def appliquer_action(
    action: str,
    milieu: Milieu,
    interne: EtatInterne
) -> None:

    if action == "fuir":
        interne.energie -= 0.03
        milieu.danger *= 0.75
        milieu.mouvement *= 0.70

    elif action == "eviter_obstacle":
        interne.energie -= 0.01
        milieu.distance_obstacle = min(
            1.0,
            milieu.distance_obstacle + 0.15
        )

    elif action == "explorer_ressource":
        interne.energie += 0.06
        milieu.ressource *= 0.90

    elif action == "chercher_ressource":
        milieu.ressource = min(
            1.0,
            milieu.ressource + 0.10
        )

    elif action == "maintenir_direction":
        interne.energie -= 0.005

    interne.energie = max(
        0.0,
        min(
            1.0,
            interne.energie
        )
    )


if __name__ == "__main__":

    random.seed(7)

    milieu = Milieu()
    interne = EtatInterne()

    print()
    print("===== AGENT SENSORIEL =====")
    print()

    for instant in range(120):

        milieu.perturber(instant)

        sensoriel = capter(milieu)

        evaluer_sensorialite(
            sensoriel,
            interne
        )

        descriptions = produire_descriptions(
            sensoriel,
            interne
        )

        action = choisir_action(
            sensoriel,
            interne
        )

        appliquer_action(
            action,
            milieu,
            interne
        )

        interne.memoire.extend(
            descriptions
        )

        if len(interne.memoire) > 100:
            interne.memoire = interne.memoire[-100:]

        if instant % 5 == 0 or descriptions:

            print(
                "t =",
                instant,
                "| valence =",
                round(interne.valence, 3),
                "| énergie =",
                round(interne.energie, 3),
                "| sécurité =",
                round(interne.securite, 3),
                "| action =",
                action
            )

            print(
                "  perception :",
                ", ".join(descriptions)
            )

    print()
    print("===== ÉTAT FINAL =====")
    print(
        "Valence :",
        round(interne.valence, 3)
    )
    print(
        "Énergie :",
        round(interne.energie, 3)
    )
    print(
        "Sécurité :",
        round(interne.securite, 3)
    )
    print(
        "Descriptions mémorisées :",
        len(interne.memoire)
    )
    print()
    print("Test sensoriel terminé.")
