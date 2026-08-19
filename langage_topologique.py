from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Iterable


# ============================================================
# COUCHE SÉMIOTIQUE
# ============================================================

@dataclass
class Signe:
    signifiant: str
    contexte: tuple[str, ...] = ()
    signifié: str = ""
    meta: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        contenu = ", ".join(self.contexte)
        return f"{self.signifiant}({contenu})"


def construire_signe(entree: str) -> Signe:
    tokens = (
        entree.lower()
        .replace(",", "")
        .split()
    )

    if not tokens:
        return Signe(
            signifiant="vide",
            signifié="absence de signal"
        )

    signifiant = tokens[0]
    contexte = tuple(tokens[1:])

    significations = {
        "objet": "entité observée",
        "relation": "connexion entre entités",
        "proche": "distance faible",
        "stable": "persistance temporelle",
        "choc": "perturbation",
        "securite": "intégrité du flux",
    }

    signifié = significations.get(
        signifiant,
        "catégorie inconnue"
    )

    return Signe(
        signifiant=signifiant,
        contexte=contexte,
        signifié=signifié,
        meta={"source": "flux"}
    )


# ============================================================
# ÉTAT LINGUISTIQUE
# ============================================================

@dataclass
class EtatLinguistique:
    signes: list[Signe] = field(
        default_factory=list
    )

    invariants: set[str] = field(
        default_factory=set
    )

    def ajouter(self, signe: Signe) -> None:
        self.signes.append(signe)

        compteurs = Counter(
            s.signifiant
            for s in self.signes
        )

        self.invariants = {
            mot
            for mot, nombre in compteurs.items()
            if nombre >= 3
        }


# ============================================================
# TOPOLOGIE DU RÉSEAU
# ============================================================

@dataclass
class ReseauRelationnel:
    nombre_noeuds: int
    liens: dict[tuple[int, int], float] = field(
        default_factory=dict
    )

    def normaliser_lien(
        self,
        a: int,
        b: int
    ) -> tuple[int, int]:
        if a > b:
            return b, a
        return a, b

    def obtenir(
        self,
        a: int,
        b: int
    ) -> float:
        cle = self.normaliser_lien(a, b)
        return self.liens.get(cle, 0.0)

    def modifier(
        self,
        a: int,
        b: int,
        variation: float
    ) -> None:
        if a == b:
            return

        cle = self.normaliser_lien(a, b)

        ancienne_valeur = self.liens.get(
            cle,
            0.0
        )

        nouvelle_valeur = (
            ancienne_valeur + variation
        )

        self.liens[cle] = max(
            0.0,
            min(1.0, nouvelle_valeur)
        )

    def afficher(self) -> None:
        for (a, b), poids in sorted(
            self.liens.items()
        ):
            if poids > 0:
                print(
                    f" - lien {a}-{b} :",
                    round(poids, 3)
                )


# ============================================================
# EXTRACTION DES NŒUDS
# ============================================================

def extraire_entiers(
    mots: tuple[str, ...]
) -> list[int]:
    resultats = []

    for mot in mots:
        propre = mot.strip(".,;:()[]")

        try:
            resultats.append(int(propre))
        except ValueError:
            continue

    return resultats


# ============================================================
# LANGAGE → TOPOLOGIE
# ============================================================

def appliquer_signe(
    signe: Signe,
    reseau: ReseauRelationnel
) -> None:
    """
    Transforme un signe en action topologique.

    stable  : renforce le lien de 0.15
    proche  : renforce le lien de 0.10
    relation: crée ou renforce le lien de 0.05
    choc    : affaiblit le lien de 0.20
    """

    noeuds = extraire_entiers(
        signe.contexte
    )

    if len(noeuds) < 2:
        return

    a, b = noeuds[0], noeuds[1]

    if not (
        0 <= a < reseau.nombre_noeuds
        and 0 <= b < reseau.nombre_noeuds
    ):
        return

    variations = {
        "stable": 0.15,
        "proche": 0.10,
        "relation": 0.05,
        "choc": -0.20,
    }

    variation = variations.get(
        signe.signifiant
    )

    if variation is not None:
        reseau.modifier(
            a,
            b,
            variation
        )


def traiter_flux(
    flux: Iterable[str],
    nombre_noeuds: int = 32
) -> tuple[
    EtatLinguistique,
    ReseauRelationnel
]:
    etat = EtatLinguistique()

    reseau = ReseauRelationnel(
        nombre_noeuds=nombre_noeuds
    )

    for entree in flux:

        signe = construire_signe(entree)

        etat.ajouter(signe)

        appliquer_signe(
            signe,
            reseau
        )

    return etat, reseau


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    flux = [
        "relation 2 3",
        "proche 2 3",
        "stable 2 3",
        "stable 2 3",
        "relation 7 11",
        "proche 7 11",
        "stable 7 11",
        "choc 2 3",
        "securite flux authentique",
    ]

    etat, reseau = traiter_flux(flux)

    print()
    print("===== LANGAGE → TOPOLOGIE =====")
    print()

    print("Invariants linguistiques :")
    for invariant in sorted(etat.invariants):
        print(" -", invariant)

    print()
    print("Liens modifiés par le langage :")
    reseau.afficher()

    print()
    print("Test terminé.")
