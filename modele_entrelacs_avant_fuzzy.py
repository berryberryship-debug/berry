from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LANGAGE INTERNE AUTOPOIETIQUE
# ============================================================

@dataclass
class Description:
    """Énoncé élémentaire dans le langage interne du système."""
    predicate: str
    args: tuple[str, ...]
    meta: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})"


@dataclass
class EtatLinguistique:
    """État interne : réseau de descriptions et invariants."""

    descriptions: list[Description] = field(
        default_factory=list
    )

    invariants: set[str] = field(
        default_factory=set
    )

    def ajouter(self, d: Description) -> None:
        self.descriptions.append(d)

        counts: dict[str, int] = {}

        for desc in self.descriptions:
            counts[desc.predicate] = (
                counts.get(desc.predicate, 0) + 1
            )

        self.invariants = {
            predicate
            for predicate, count in counts.items()
            if count >= 3
        }

    def realite(self) -> list[str]:
        """Invariants descriptifs du système."""
        return sorted(self.invariants)


def traduire_entree(brut: str) -> Description:
    """Transforme une entrée brute en description interne."""

    tokens = (
        brut.lower()
        .replace(",", "")
        .split()
    )

    if not tokens:
        return Description("vide", ())

    predicate = tokens[0]
    args = tuple(tokens[1:])

    return Description(
        predicate,
        args,
        {"source": "brut"}
    )


def boucle_autopoietique(
    flux_entrees: Iterable[str],
    etat_initial: EtatLinguistique | None = None,
    max_iter: int | None = None,
) -> EtatLinguistique:
    """Boucle de traduction, intégration et méta-description."""

    etat = (
        etat_initial
        if etat_initial is not None
        else EtatLinguistique()
    )

    i = 0

    for brut in flux_entrees:

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

        i += 1

    return etat


# ============================================================
# PARAMÈTRES DU RÉSEAU
# ============================================================

rng = np.random.default_rng(7)

N = 32
T = 900
DT = 0.04
CHOC = 450
K = 5

BLOCS = [
    [2, 3, 4, 7, 11, 16, 18, 24, 26, 28],
    [5, 8, 9, 10, 13, 15, 27, 30, 31],
    [1, 6, 17, 19, 21, 22, 25, 29],
    [0, 12, 14, 20, 23],
]


def garder_meilleurs_liens(W: np.ndarray) -> np.ndarray:
    """Conserve les K liens les plus forts de chaque nœud."""

    resultat = np.zeros_like(W)

    for i in range(N):
        indices = np.argsort(W[i])[-K:]
        resultat[i, indices] = W[i, indices]

    np.fill_diagonal(resultat, 0.0)

    return resultat


def creer_reseau() -> tuple[np.ndarray, np.ndarray]:
    """Crée les états initiaux et la topologie initiale."""

    x = rng.normal(0, 1, (N, 2))

    masque = rng.random((N, N)) < 0.12

    W = np.where(
        masque,
        rng.uniform(0.05, 0.35, (N, N)),
        0.0
    )

    np.fill_diagonal(W, 0.0)

    return x, W


def creer_flux_linguistique(
    x: np.ndarray,
    t: int
) -> list[str]:
    """Transforme l'état numérique du réseau en entrées linguistiques."""

    energie = float(np.mean(np.sum(x ** 2, axis=1)))
    moyenne_x = float(np.mean(x[:, 0]))
    moyenne_y = float(np.mean(x[:, 1]))

    flux = [
        f"objet reseau energie {energie:.3f}",
        f"position moyenne x {moyenne_x:.3f}",
        f"position moyenne y {moyenne_y:.3f}",
    ]

    if t >= CHOC:
        flux.append("perturbation active")

    if energie < 0.5:
        flux.append("stabilite forte")
    else:
        flux.append("stabilite faible")

    return flux


def simuler(
    etat_initial: np.ndarray,
    reseau_initial: np.ndarray,
    bruits: np.ndarray,
    champ_global: bool = False,
    memoire_distribuee: bool = False,
    auto_maintien: bool = False,
    memoire_longue: bool = False,
    entrelacs: bool = False,
    perturbe: bool = False
) -> dict:

    x = etat_initial.copy()
    W = reseau_initial.copy()

    memoire_courte = np.zeros((N, N))
    memoire_forme = np.zeros((N, N))
    memoire_modes = np.zeros(N)
    milieu = np.zeros(2)

    etat_linguistique = EtatLinguistique()

    historique_viabilite = []
    historique_integration = []
    historique_invariants = []

    for t in range(T):

        if perturbe and t == CHOC:
            x[BLOCS[0]] += rng.normal(
                0,
                3,
                (len(BLOCS[0]), 2)
            )

        degre = W.sum(axis=1, keepdims=True) + 1e-9
        voisinage = (W @ x) / degre

        dx = (
            -0.55 * x
            + 0.80 * (voisinage - x)
            - 0.22 * x
        )

        # Bohm : cohérence globale
        if champ_global:
            centre_global = x.mean(axis=0)
            dx += 0.18 * (centre_global - x)

        # Varela : auto-maintien
        if auto_maintien:
            energie = np.mean(np.sum(x ** 2, axis=1))

            if energie > 0.8:
                dx += -0.25 * x

        # Merleau-Ponty : système et milieu
        if entrelacs:
            perception = milieu - x.mean(axis=0)
            dx += 0.18 * perception

        x = x + DT * dx
        x = x + bruits[t]

        if entrelacs:
            action = 0.12 * x.mean(axis=0)

            milieu = (
                0.995 * milieu
                + 0.005 * action
                + 0.01 * bruits[t].mean(axis=0)
            )

        distance = np.linalg.norm(
            x[:, None, :] - x[None, :, :],
            axis=2
        )

        correlation = np.exp(-distance)

        # Mémoire courte
        memoire_courte = (
            0.992 * memoire_courte
            + 0.008 * correlation
        )

        # Sheldrake : mémoire historique formelle
        if memoire_longue:
            memoire_forme = (
                0.999 * memoire_forme
                + 0.001 * correlation
            )

        # Pribram : mémoire distribuée fréquentielle
        if memoire_distribuee:

            signal = x[:, 0] + 1j * x[:, 1]

            modes = np.fft.fft(signal)
            amplitudes = np.abs(modes)
            phases = np.angle(modes)

            memoire_modes = (
                0.995 * memoire_modes
                + 0.005 * amplitudes
            )

            modes_reconstruits = (
                modes
                + 0.12
                * memoire_modes
                * np.exp(1j * phases)
            )

            signal_reconstruit = np.fft.ifft(
                modes_reconstruits
            )

            x[:, 0] += 0.08 * (
                signal_reconstruit.real - x[:, 0]
            )

            x[:, 1] += 0.08 * (
                signal_reconstruit.imag - x[:, 1]
            )

        # Mise à jour de la topologie
        memoire_utilisee = memoire_courte.copy()

        if memoire_longue:
            memoire_utilisee += 0.35 * memoire_forme

        W += DT * (
            0.80 * memoire_utilisee
            - 0.35 * W
        )

        W = np.clip(W, 0.0, 1.0)
        W = garder_meilleurs_liens(W)

        # Entrées dans le langage interne
        flux = creer_flux_linguistique(x, t)

        for entree in flux:
            etat_linguistique.ajouter(
                traduire_entree(entree)
            )

        # Mesure d'intégration linguistique
        nombre_descriptions = len(
            etat_linguistique.descriptions
        )

        nombre_invariants = len(
            etat_linguistique.invariants
        )

        integration = (
            nombre_invariants
            / max(1, nombre_descriptions)
        )

        energie = np.mean(np.sum(x ** 2, axis=1))

        historique_viabilite.append(energie)
        historique_integration.append(integration)
        historique_invariants.append(nombre_invariants)

    return {
        "viabilite": np.array(historique_viabilite),
        "integration": np.array(historique_integration),
        "invariants": np.array(historique_invariants),
        "etat_linguistique": etat_linguistique,
    }


# ============================================================
# COMPARAISON DES MODÈLES
# ============================================================

etat_initial, reseau_initial = creer_reseau()

bruits = rng.normal(
    0,
    0.035,
    (T, N, 2)
)


modeles = {
    "Base": {},

    "Bohm": {
        "champ_global": True
    },

    "Pribram": {
        "memoire_distribuee": True
    },

    "Varela": {
        "auto_maintien": True
    },

    "Sheldrake": {
        "memoire_longue": True
    },

    "Merleau-Ponty": {
        "entrelacs": True
    },

    "Complet": {
        "champ_global": True,
        "memoire_distribuee": True,
        "auto_maintien": True,
        "memoire_longue": True,
        "entrelacs": True
    }
}


resultats = {}

for nom, options in modeles.items():

    normal = simuler(
        etat_initial,
        reseau_initial,
        bruits,
        **options,
        perturbe=False
    )

    perturbe = simuler(
        etat_initial,
        reseau_initial,
        bruits,
        **options,
        perturbe=True
    )

    difference = np.abs(
        perturbe["viabilite"]
        - normal["viabilite"]
    )

    apres_choc = difference[CHOC:]

    pic = np.max(apres_choc)
    final = difference[-1]
    aire = np.sum(apres_choc) * DT

    seuil = 0.01 * pic
    indices = np.where(apres_choc <= seuil)[0]

    if len(indices) > 0:
        recuperation = indices[0] * DT
    else:
        recuperation = np.nan

    integration_finale = float(
        perturbe["integration"][-1]
    )

    invariants_finaux = int(
        perturbe["invariants"][-1]
    )

    resultats[nom] = {
        "normal": normal,
        "perturbe": perturbe,
        "difference": difference,
        "pic": pic,
        "final": final,
        "aire": aire,
        "recuperation": recuperation,
        "integration": integration_finale,
        "invariants": invariants_finaux,
    }


# ============================================================
# RÉSULTATS
# ============================================================

print()
print("===== COMPARAISON DES MODÈLES =====")
print()

for nom, resultat in resultats.items():

    print(nom)
    print(
        "  Écart maximal après choc :",
        round(resultat["pic"], 6)
    )
    print(
        "  Écart final              :",
        round(resultat["final"], 8)
    )
    print(
        "  Aire de perturbation     :",
        round(resultat["aire"], 6)
    )
    print(
        "  Temps de récupération    :",
        round(resultat["recuperation"], 4)
    )
    print(
        "  Intégration linguistique :",
        round(resultat["integration"], 6)
    )
    print(
        "  Invariants finaux        :",
        resultat["invariants"]
    )
    print()


# ============================================================
# GRAPHIQUES
# ============================================================

plt.style.use("dark_background")

fig, axes = plt.subplots(
    4,
    1,
    figsize=(13, 14)
)

couleurs = {
    "Base": "#94a3b8",
    "Bohm": "#22d3ee",
    "Pribram": "#a78bfa",
    "Varela": "#f59e0b",
    "Sheldrake": "#fb7185",
    "Merleau-Ponty": "#34d399",
    "Complet": "#ffffff"
}


for nom, resultat in resultats.items():

    axes[0].plot(
        resultat["difference"],
        label=nom,
        color=couleurs[nom],
        linewidth=2
    )

    axes[1].plot(
        resultat["perturbe"]["viabilite"],
        label=nom,
        color=couleurs[nom],
        linewidth=2
    )

    axes[2].plot(
        resultat["perturbe"]["integration"],
        label=nom,
        color=couleurs[nom],
        linewidth=2
    )


noms = list(resultats.keys())

aires = [
    resultats[nom]["aire"]
    for nom in noms
]

axes[3].bar(
    noms,
    aires,
    color=[
        couleurs[nom]
        for nom in noms
    ]
)


for axe in axes[:3]:

    axe.axvline(
        CHOC,
        color="white",
        linestyle="--",
        alpha=0.7
    )

    axe.grid(alpha=0.15)
    axe.legend(fontsize=8)


axes[0].set_title(
    "Écart entre trajectoire normale et perturbée"
)
axes[0].set_ylabel("Écart")


axes[1].set_title(
    "Viabilité de la trajectoire perturbée"
)
axes[1].set_ylabel("Viabilité")


axes[2].set_title(
    "Intégration linguistique"
)
axes[2].set_ylabel("Invariants / descriptions")


axes[3].set_title(
    "Aire totale de la perturbation"
)
axes[3].set_ylabel("Aire")
axes[3].tick_params(axis="x", rotation=30)
axes[3].grid(axis="y", alpha=0.15)


fig.suptitle(
    "Entrelacs, mémoire, autopoïèse et langage interne",
    fontsize=16
)

fig.tight_layout()

plt.savefig(
    "comparaison_entrelacs.png",
    dpi=180,
    bbox_inches="tight"
)

print("Image créée : comparaison_entrelacs.png")


# ============================================================
# EXEMPLE LINGUISTIQUE FINAL
# ============================================================

etat_exemple = boucle_autopoietique(
    [
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
    ],
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
