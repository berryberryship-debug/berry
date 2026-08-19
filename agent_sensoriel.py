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
        return "eviter_ob
cd ~/quantumlab
rm -f agent_entrelace.py

cat > agent_entrelace.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import random


# ============================================================
# LANGAGE INTERNE
# ============================================================

@dataclass
class Description:
    predicate: str
    args: tuple[str, ...] = ()
    meta: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        contenu = ", ".join(self.args)
        return f"{self.predicate}({contenu})"


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


# ============================================================
# RESEAU RELATIONNEL
# ============================================================

@dataclass
class ReseauRelationnel:
    nombre_noeuds: int = 8
    liens: dict[tuple[int, int], float] = field(
        default_factory=dict
    )

    def cle(self, a: int, b: int) -> tuple[int, int]:
        return tuple(sorted((a, b)))

    def modifier(
        self,
        a: int,
        b: int,
        variation: float
    ) -> None:

        if a == b:
            return

        if not (
            0 <= a < self.nombre_noeuds
            and 0 <= b < self.nombre_noeuds
        ):
            return

        identifiant = self.cle(a, b)

        ancienne_valeur = self.liens.get(
            identifiant,
            0.0
        )

        nouvelle_valeur = (
            ancienne_valeur + variation
        )

        self.liens[identifiant] = max(
            0.0,
            min(1.0, nouvelle_valeur)
        )

    def total(self) -> float:
        return sum(self.liens.values())

    def afficher(self) -> None:
        actifs = [
            (lien, poids)
            for lien, poids in self.liens.items()
            if poids > 0.0
        ]

        if not actifs:
            print(" - aucun lien actif")
            return

        for (a, b), poids in sorted(actifs):
            print(
                f" - lien {a}-{b} :",
                round(poids, 3)
            )


# ============================================================
# MILIEU
# ============================================================

@dataclass
class Milieu:
    lumiere: float = 0.5
    mouvement: float = 0.0
    danger: float = 0.0
    ressource: float = 0.5
    obstacle: float = 0.8

    def evoluer(self, instant: int) -> None:

        if instant == 20:
            self.lumiere = 0.9
            self.ressource = 0.9

        if instant == 45:
            self.mouvement = 0.9
            self.danger = 1.0
            self.obstacle = 0.2

        if instant == 70:
            self.mouvement = 0.1
            self.danger = 0.0
            self.obstacle = 0.8

        if instant == 95:
            self.lumiere = 0.4
            self.ressource = 0.3


# ============================================================
# CAPTEURS
# ============================================================

@dataclass
class Perception:
    lumiere: float
    mouvement: float
    danger: float
    ressource: float
    obstacle: float


def capter(milieu: Milieu) -> Perception:

    def bruit(valeur: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                valeur + random.uniform(-0.02, 0.02)
            )
        )

    return Perception(
        lumiere=bruit(milieu.lumiere),
        mouvement=bruit(milieu.mouvement),
        danger=bruit(milieu.danger),
        ressource=bruit(milieu.ressource),
        obstacle=bruit(milieu.obstacle),
    )


# ============================================================
# PERCEPTION → DESCRIPTION
# ============================================================

def decrire(
    perception: Perception
) -> list[Description]:

    descriptions = []

    if perception.lumiere > 0.7:
        descriptions.append(
            Description(
                "lumiere_forte",
                ("milieu",)
            )
        )

    if perception.mouvement > 0.7:
        descriptions.append(
            Description(
                "mouvement_rapide",
                ("milieu",)
            )
        )

    if perception.danger > 0.7:
        descriptions.append(
            Description(
                "danger",
                ("milieu",)
            )
        )

    if perception.ressource > 0.7:
        descriptions.append(
            Description(
                "ressource",
                ("milieu",)
            )
        )

    if perception.obstacle < 0.3:
        descriptions.append(
            Description(
                "obstacle",
                ("proche",)
            )
        )

    if not descriptions:
        descriptions.append(
            Description(
                "etat_neutre",
                ("milieu",)
            )
        )

    return descriptions


# ============================================================
# DESCRIPTION → TOPOLOGIE
# ============================================================

def appliquer_description(
    description: Description,
    reseau: ReseauRelationnel
) -> None:

    if description.predicate == "ressource":
        reseau.modifier(0, 1, 0.08)
        reseau.modifier(0, 2, 0.05)

    elif description.predicate == "lumiere_forte":
        reseau.modifier(0, 1, 0.03)

    elif description.predicate == "danger":
        reseau.modifier(0, 1, -0.12)
        reseau.modifier(0, 2, -0.08)

    elif description.predicate == "obstacle":
        reseau.modifier(0, 3, -0.06)

    elif description.predicate == "mouvement_rapide":
        reseau.modifier(1, 2, -0.04)

    elif description.predicate == "etat_neutre":
        reseau.modifier(0, 1, 0.01)


# ============================================================
# ÉVALUATION INTERNE
# ============================================================

@dataclass
class EtatInterne:
    energie: float = 0.7
    securite: float = 1.0
    val
cd ~/quantumlab
rm -f agent_entrelace.py

cat > agent_entrelace.py <<'PY'
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import random


# ============================================================
# LANGAGE INTERNE
# ============================================================

@dataclass
class Description:
    predicate: str
    args: tuple[str, ...] = ()
    meta: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        contenu = ", ".join(self.args)
        return f"{self.predicate}({contenu})"


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


# ============================================================
# RESEAU RELATIONNEL
# ============================================================

@dataclass
class ReseauRelationnel:
    nombre_noeuds: int = 8
    liens: dict[tuple[int, int], float] = field(
        default_factory=dict
    )

    def cle(self, a: int, b: int) -> tuple[int, int]:
        return tuple(sorted((a, b)))

    def modifier(
        self,
        a: int,
        b: int,
        variation: float
    ) -> None:

        if a == b:
            return

        if not (
            0 <= a < self.nombre_noeuds
            and 0 <= b < self.nombre_noeuds
        ):
            return

        identifiant = self.cle(a, b)

        ancienne_valeur = self.liens.get(
            identifiant,
            0.0
        )

        nouvelle_valeur = (
            ancienne_valeur + variation
        )

        self.liens[identifiant] = max(
            0.0,
            min(1.0, nouvelle_valeur)
        )

    def total(self) -> float:
        return sum(self.liens.values())

    def afficher(self) -> None:
        actifs = [
            (lien, poids)
            for lien, poids in self.liens.items()
            if poids > 0.0
        ]

        if not actifs:
            print(" - aucun lien actif")
            return

        for (a, b), poids in sorted(actifs):
            print(
                f" - lien {a}-{b} :",
                round(poids, 3)
            )


# ============================================================
# MILIEU
# ============================================================

@dataclass
class Milieu:
    lumiere: float = 0.5
    mouvement: float = 0.0
    danger: float = 0.0
    ressource: float = 0.5
    obstacle: float = 0.8

    def evoluer(self, instant: int) -> None:

        if instant == 20:
            self.lumiere = 0.9
            self.ressource = 0.9

        if instant == 45:
            self.mouvement = 0.9
            self.danger = 1.0
            self.obstacle = 0.2

        if instant == 70:
            self.mouvement = 0.1
            self.danger = 0.0
            self.obstacle = 0.8

        if instant == 95:
            self.lumiere = 0.4
            self.ressource = 0.3


# ============================================================
# CAPTEURS
# ============================================================

@dataclass
class Perception:
    lumiere: float
    mouvement: float
    danger: float
    ressource: float
    obstacle: float


def capter(milieu: Milieu) -> Perception:

    def bruit(valeur: float) -> float:
        return max(
            0.0,
            min(
                1.0,
                valeur + random.uniform(-0.02, 0.02)
            )
        )

    return Perception(
        lumiere=bruit(milieu.lumiere),
        mouvement=bruit(milieu.mouvement),
        danger=bruit(milieu.danger),
        ressource=bruit(milieu.ressource),
        obstacle=bruit(milieu.obstacle),
    )


# ============================================================
# PERCEPTION → DESCRIPTION
# ============================================================

def decrire(
    perception: Perception
) -> list[Description]:

    descriptions = []

    if perception.lumiere > 0.7:
        descriptions.append(
            Description(
                "lumiere_forte",
                ("milieu",)
            )
        )

    if perception.mouvement > 0.7:
        descriptions.append(
            Description(
                "mouvement_rapide",
                ("milieu",)
            )
        )

    if perception.danger > 0.7:
        descriptions.append(
            Description(
                "danger",
                ("milieu",)
            )
        )

    if perception.ressource > 0.7:
        descriptions.append(
            Description(
                "ressource",
                ("milieu",)
            )
        )

    if perception.obstacle < 0.3:
        descriptions.append(
            Description(
                "obstacle",
                ("proche",)
            )
        )

    if not descriptions:
        descriptions.append(
            Description(
                "etat_neutre",
                ("milieu",)
            )
        )

    return descriptions


# ============================================================
# DESCRIPTION → TOPOLOGIE
# ============================================================

def appliquer_description(
    description: Description,
    reseau: ReseauRelationnel
) -> None:

    if description.predicate == "ressource":
        reseau.modifier(0, 1, 0.08)
        reseau.modifier(0, 2, 0.05)

    elif description.predicate == "lumiere_forte":
        reseau.modifier(0, 1, 0.03)

    elif description.predicate == "danger":
        reseau.modifier(0, 1, -0.12)
        reseau.modifier(0, 2, -0.08)

    elif description.predicate == "obstacle":
        reseau.modifier(0, 3, -0.06)

    elif description.predicate == "mouvement_rapide":
        reseau.modifier(1, 2, -0.04)

    elif description.predicate == "etat_neutre":
        reseau.modifier(0, 1, 0.01)


# ============================================================
# ÉVALUATION INTERNE
# ============================================================

@dataclass
class EtatInterne:
    energie: float = 0.7
    securite: float = 1.0
    valence: float = 0.0
    tension: float = 0.0


def evaluer(
    perception: Perception,
    interne: EtatInterne
) -> None:

    plaisir = (
        0.40 * perception.ressource
        + 0.20 * perception.lumiere
        + 0.20 * perception.obstacle
    )

    menace = (
        0.60 * perception.danger
        + 0.25 * perception.mouvement
        + 0.30 * (1.0 - perception.obstacle)
    )

    interne.valence = max(
        -1.0,
        min(1.0, plaisir - menace)
    )

    interne.tension = max(
        0.0,
        min(1.0, menace)
    )

    interne.securite = max(
        0.0,
        min(1.0, 1.0 - menace)
    )

    interne.energie += (
        0.03 * perception.ressource
        - 0.04 * menace
    )

    interne.energie = max(
        0.0,
        min(1.0, interne.energie)
    )


# ============================================================
# FINALITE → ACTION
# ============================================================

def choisir_action(
    perception: Perception,
    interne: EtatInterne
) -> str:

    if perception.danger > 0.7:
        return "fuir"

    if perception.obstacle < 0.3:
        return "eviter"

    if perception.ressource > 0.7:
        return "explorer"

    if interne.energie < 0.25:
        return "chercher"

    if interne.valence > 0.25:
        return "maintenir"

    return "observer"


def appliquer_action(
    action: str,
    milieu: Milieu,
    interne: EtatInterne
) -> None:

    if action == "fuir":
        interne.energie -= 0.04
        milieu.danger *= 0.70
        milieu.mouvement *= 0.70

    elif action == "eviter":
        interne.energie -= 0.01
        milieu.obstacle = min(
            1.0,
            milieu.obstacle + 0.15
        )

    elif action == "explorer":
        interne.energie += 0.05
        milieu.ressource *= 0.90

    elif action == "chercher":
        milieu.ressource = min(
            1.0,
            milieu.ressource + 0.10
        )

    elif action == "maintenir":
        interne.energie -= 0.005

    interne.energie = max(
        0.0,
        min(1.0, interne.energie)
    )


# ============================================================
# BOUCLE FERMÉE
# ============================================================

def executer(duree: int = 120) -> None:

    milieu = Milieu()
    perception = None

    reseau = ReseauRelationnel(
        nombre_noeuds=8
    )

    langage = EtatLinguistique()
    interne = EtatInterne()

    for instant in range(duree):

        # 1. Le milieu évolue
        milieu.evoluer(instant)

        # 2. Le système perçoit
        perception = capter(milieu)

        # 3. Le système évalue
        evaluer(
            perception,
            interne
        )

        # 4. La perception devient langage
        descriptions = decrire(
            perception
        )

        # 5. Le langage modifie le réseau
        for description in descriptions:
            langage.ajouter(description)
            appliquer_description(
                description,
                reseau
            )

        # 6. Le système choisit une action
        action = choisir_action(
            perception,
            interne
        )

        # 7. L'action modifie le milieu
        appliquer_action(
            action,
            milieu,
            interne
        )

        if instant % 5 == 0 or action == "fuir":

            mots = [
                str(description)
                for description in descriptions
            ]

            print(
                "t =",
                instant,
                "| action =",
                action,
                "| valence =",
                round(interne.valence, 3),
                "| énergie =",
                round(interne.energie, 3),
                "| sécurité =",
                round(interne.securite, 3)
            )

            print(
                "  descriptions :",
                ", ".join(mots)
            )

    print()
    print("===== BILAN FINAL =====")
    print()

    print(
        "Invariants :",
        sorted(langage.invariants)
    )

    print(
        "Descriptions mémorisées :",
        len(langage.descriptions)
    )

    print(
        "Énergie finale :",
        round(interne.energie, 4)
    )

    print(
        "Sécurité finale :",
        round(interne.securite, 4)
    )

    print(
        "Valence finale :",
        round(interne.valence, 4)
    )

    print()
    print("Liens du réseau :")
    reseau.afficher()

    print()
    print("Test terminé.")


if __name__ == "__main__":
    random.seed(7)
    executer()
