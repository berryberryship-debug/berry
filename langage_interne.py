from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Description:
    predicate: str
    args: tuple[str, ...]
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

        compteurs: dict[str, int] = {}

        for desc in self.descriptions:
            compteurs[desc.predicate] = (
                compteurs.get(desc.predicate, 0) + 1
            )

        self.invariants = {
            predicat
            for predicat, compteur in compteurs.items()
            if compteur >= 3
        }

    def realite(self) -> list[str]:
        return sorted(self.invariants)


def traduire_entree(brut: str) -> Description:
    tokens = (
        brut.lower()
        .replace(",", "")
        .split()
    )

    if not tokens:
        return Description("vide", ())

    return Description(
        tokens[0],
        tuple(tokens[1:]),
        {"source": "brut"}
    )


def boucle_autopoietique(
    flux_entrees: Iterable[str],
    etat_initial: EtatLinguistique | None = None,
    max_iter: int | None = None,
) -> EtatLinguistique:

    etat = (
        etat_initial
        if etat_initial is not None
        else EtatLinguistique()
    )

    for i, brut in enumerate(flux_entrees):

        if max_iter is not None and i >= max_iter:
            break

        description = traduire_entree(brut)
        etat.ajouter(description)

        if etat.invariants and i % 3 == 0:
            meta = Description(
                "je_constate",
                (
                    "invariants",
                    ", ".join(etat.realite())
                ),
                {"niveau": "meta"}
            )
            etat.ajouter(meta)

    return etat


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

    etat_exemple = boucle_autopoietique(
        flux,
        max_iter=10
    )

    print()
    print("===== LANGAGE INTERNE =====")
    print(
        "Réel du système :",
        etat_exemple.realite()
    )
    print(
        "Nombre de descriptions :",
        len(etat_exemple.descriptions)
    )

    for description in etat_exemple.descriptions[-5:]:
        print(" -", description)
