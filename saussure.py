from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class Signe:
    """
    Un signe composé d'un signifiant et d'un signifié.
    """

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


@dataclass
class SystemeSemiotique:
    """
    Système de signes organisé par différences et relations.
    """

    signes: list[Signe] = field(
        default_factory=list
    )

    relations: dict[str, set[str]] = field(
        default_factory=dict
    )

    def ajouter(self, signe: Signe) -> None:
        self.signes.append(signe)

        if signe.signifiant not in self.relations:
            self.relations[signe.signifiant] = set()

        for autre in self.signes:

            if autre.signifiant != signe.signifiant:
                self.relations[
                    signe.signifiant
                ].add(autre.signifiant)

    def frequence(self, signifiant: str) -> int:
        return sum(
            1
            for signe in self.signes
            if signe.signifiant == signifiant
        )

    def valeur(self, signifiant: str) -> float:
        """
        Valeur différentielle simplifiée.

        La valeur dépend de la fréquence du signe
        et du nombre de signes auxquels il est relié.
        """

        frequence = self.frequence(signifiant)

        nombre_relations = len(
            self.relations.get(signifiant, set())
        )

        return float(
            frequence + 0.25 * nombre_relations
        )

    def vocabulaire(self) -> list[str]:
        return sorted(self.relations.keys())

    def relations_du_signe(
        self,
        signifiant: str
    ) -> list[str]:
        return sorted(
            self.relations.get(signifiant, set())
        )


def construire_signe(entree: str) -> Signe:
    """
    Transforme une entrée en signe.

    Le premier mot devient le signifiant.
    Les mots suivants forment le signifié.
    """

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
    arguments = tuple(tokens[1:])

    if signifiant == "objet":
        signifié = "entité observée"

    elif signifiant == "relation":
        signifié = "connexion entre entités"

    elif signifiant == "position":
        signifié = "configuration spatiale"

    elif signifiant == "stable":
        signifié = "persistance temporelle"

    else:
        signifié = "catégorie non interprétée"

    return Signe(
        signifiant=signifiant,
        signifié=signifié,
        contexte=arguments,
        meta={"source": "entrée"}
    )


def analyser_flux(
    flux: list[str]
) -> SystemeSemiotique:

    systeme = SystemeSemiotique()

    for entree in flux:
        signe = construire_signe(entree)
        systeme.ajouter(signe)

    return systeme


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
    ]

    systeme = analyser_flux(flux)

    print()
    print("===== SYSTÈME SÉMIOTIQUE DE SAUSSURE =====")
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
    print("Relations entre signes :")

    for mot in systeme.vocabulaire():
        relations = systeme.relations_du_signe(mot)

        print(
            " -",
            mot,
            "->",
            relations
        )

    print()
    print("Derniers signes :")

    for signe in systeme.signes[-5:]:
        print(" -", signe)

    print()
    print("Test sémiotique terminé.")
