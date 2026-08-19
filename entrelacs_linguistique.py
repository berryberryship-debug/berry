from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Iterable


# ============================================================
# SAUSSURE : SIGNIFIANT, SIGNIFIÉ ET VALEUR RELATIONNELLE
# ============================================================

@dataclass
class Signe:
    signifiant: str
    signifié: str
    contexte: tuple[str, ...] = ()
    meta: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.signifiant}"
            f" -> "
            f"{self.signifié}"
        )


def construire_signe(entree: str) -> Signe:
    tokens = (
        entree.lower()
        .replace(",", "")
        .split()
    )

    if not tokens:
        return Signe(
            "vide",
            "absence de signal"
        )

    signifiant = tokens[0]
    contexte = tuple(tokens[1:])

    significations = {
        "objet": "entité observée",
        "relation": "connexion entre entités",
        "position": "configuration spatiale",
        "stable": "persistance temporelle",
        "proche": "faible distance relationnelle",
        "choc": "perturbation externe",
        "securite": "intégrité du flux",
    }

    signifié = significations.get(
        signifiant,
        "catégorie non interprétée"
    )

    return Signe(
        signifiant=signifiant,
        signifié=signifié,
        contexte=contexte,
        meta={"source": "flux"}
    )


@dataclass
class SystemeSemiotique:
    signes: list[Signe] = field(
        default_factory=list
    )

    relations: dict[str, set[str]] = field(
        default_factory=dict
    )

    def ajouter(self, signe: Signe) -> None:
        anciens = list(self.signes)

        self.signes.append(signe)

        if signe.signifiant not in self.relations:
            self.relations[signe.signifiant] = set()

        for autre in anciens:
            if autre.signifiant != signe.signifiant:

                self.relations[
                    signe.signifiant
                ].add(autre.signifiant)

                if autre.signifiant not in self.relations:
                    self.relations[
                        autre.signifiant
                    ] = set()

                self.relations[
                    autre.signifiant
                ].add(signe.signifiant)

    def frequence(self, signifiant: str) -> int:
        return sum(
            1
            for signe in self.signes
            if signe.signifiant == signifiant
        )

    def valeur(self, signifiant: str) -> float:
        fréquence = self.frequence(signifiant)

        relations = len(
            self.relations.get(signifiant, set())
        )

        return fréquence + 0.25 * relations

    def vocabulaire(self) -> list[str]:
        return sorted(self.relations.keys())

    def realite(self) -> list[str]:
        """
        Invariants sémiotiques :
        signes présents au moins trois fois.
        """

        return sorted(
            signifiant
            for signifiant in self.vocabulaire()
            if self.frequence(signifiant) >= 3
        )


# ============================================================
# LANGAGE INTERNE
# ============================================================

@dataclass
class Description:
    predicate: str
    args: tuple[str, ...]
    meta: dict[str, str] = field(
        default_factory=dict
    )

    def __str__(self) -> str:
        return (
            f"{self.predicate}"
            f"({', '.join(self.args)})"
        )


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

        compteurs = Counter(
            desc.predicate
            for desc in self.descriptions
        )

        self.invariants = {
            predicat
            for predicat, compteur in compteurs.items()
            if compteur >= 3
        }

    def realite(self) -> list[str]:
        return sorted(self.invariants)


def description_depuis_signe(
    signe: Signe
) -> Description:
    return Description(
        predicate=signe.signifiant,
        args=signe.contexte,
        meta={
            "signifié": signe.signifié,
            **signe.meta
        }
    )


# ============================================================
# COUPLAGE SAUSSURE + AUTOPOÏÈSE
# ============================================================

def traiter_flux(
    flux: Iterable[str]
) -> tuple[SystemeSemiotique, EtatLinguistique]:

    systeme = SystemeSemiotique()
    etat = EtatLinguistique()

    for entree in flux:

        # 1. Entrée brute vers signe
        signe = construire_signe(entree)

        # 2. Ajout au système différentiel
        systeme.ajouter(signe)

        # 3. Traduction en description interne
        description = description_depuis_signe(signe)
        etat.ajouter(description)

        # 4. Méta-description contrôlée
        if etat.invariants:

            meta = Description(
                "je_constate",
                (
                    "invariants",
                    ", ".join(etat.realite())
                ),
                {"niveau": "meta"}
            )

            etat.ajouter(meta)

    return systeme, etat


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    flux = [
        "objet A position 10",
        "objet B position 20",
        "objet A position 12",
        "relation A B proche",
        "objet A position 11",
        "relation A B stable",
        "objet C position 5",
        "relation A B stable",
        "objet A position 13",
        "relation A B stable",
        "securite flux authentique",
    ]

    systeme, etat = traiter_flux(flux)

    print()
    print("===== COUPLAGE SAUSSURE + LANGAGE INTERNE =====")
    print()

    print("Vocabulaire :")
    for mot in systeme.vocabulaire():
        print(" -", mot)

    print()
    print("Valeurs différentielles :")

    for mot in systeme.vocabulaire():
        print(
            " -",
            mot,
            ":",
            round(systeme.valeur(mot), 3)
        )

    print()
    print(
        "Réalité sémiotique :",
        systeme.vocabulaire()
    )

    print(
        "Invariants linguistiques :",
        etat.realite()
    )

    print(
        "Nombre de descriptions :",
        len(etat.descriptions)
    )

    print()
    print("Dernières descriptions :")

    for description in etat.descriptions[-8:]:
        print(" -", description)

    print()
    print("Test terminé.")
