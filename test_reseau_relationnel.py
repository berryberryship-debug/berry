import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Réseau relationnel avec mémoire et topologie adaptative
# --------------------------------------------------

rng = np.random.default_rng(7)

N = 32
T = 900
DT = 0.04
CHOC = 450

# Paramètres de la mémoire et de la plasticité
MEMORY = 0.992
RENFORCEMENT = 0.80
OUBLI = 0.35
COMPETITION = 0.55

# Chaque unité conserve au maximum K connexions
K = 5


def initialiser_reseau():
    x = rng.normal(0, 1, (N, 2))

    masque = rng.random((N, N)) < 0.12
    W = np.where(
        masque,
        rng.uniform(0.05, 0.35, (N, N)),
        0.0
    )

    np.fill_diagonal(W, 0.0)

    memoire = np.zeros((N, N))

    return x, W, memoire


def conserver_meilleurs_liens(W, k=K):
    """
    Chaque nœud conserve uniquement ses k connexions
    les plus fortes. Cela empêche la connexion totale.
    """
    resultat = np.zeros_like(W)

    for i in range(N):
        indices = np.argsort(W[i])[-k:]
        resultat[i, indices] = W[i, indices]

    np.fill_diagonal(resultat, 0.0)

    return resultat


def simuler(memoire_active, topologie_adaptative):
    x, W, memoire = initialiser_reseau()

    cible = np.zeros((N, 2))

    viabilite = []
    densite = []
    recuperation = []
    erreur_prediction = []
    nombre_liens = []

    etat_precedent = x.copy()

    for t in range(T):

        # Perturbation externe
        if t == CHOC:
            x += rng.normal(0, 2, (N, 2))

        # Prévision locale très simple
        prediction = x.copy()

        # Influence du voisinage
        degre = W.sum(axis=1, keepdims=True) + 1e-9
        voisinage = (W @ x) / degre

        # Dynamique propre + interaction + retour vers la zone viable
        dx = (
            -0.55 * x
            + 0.80 * (voisinage - x)
            + 0.22 * (cible - x)
        )

        x = x + DT * dx
        x = x + rng.normal(0, 0.035, x.shape)

        # Corrélations entre unités
        distance = np.linalg.norm(
            x[:, None, :] - x[None, :, :],
            axis=2
        )

        correlation = np.exp(-distance)

        if memoire_active:
            memoire = (
                MEMORY * memoire
                + (1 - MEMORY) * correlation
            )

        if topologie_adaptative:
            moyenne_locale = memoire.mean(axis=1, keepdims=True)

            # Renforcement + oubli + compétition
            variation_W = (
                RENFORCEMENT * memoire
                - OUBLI * W
                - COMPETITION * moyenne_locale
            )

            W = W + DT * variation_W
            W = np.clip(W, 0.0, 1.0)

            # Sélection locale : seuls les meilleurs liens restent
            W = conserver_meilleurs_liens(W)

        # Mesures
        ecart = np.mean(np.sum((x - cible) ** 2, axis=1))

        variation = np.mean(
            np.sum((x - etat_precedent) ** 2, axis=1)
        )

        erreur = np.mean(
            np.sum((x - prediction) ** 2, axis=1)
        )

        viabilite.append(ecart)
        densite.append(W.sum() / (N * (N - 1)))
        recuperation.append(variation)
        erreur_prediction.append(erreur)
        nombre_liens.append(np.count_nonzero(W))

        etat_precedent = x.copy()

    return {
        "viabilite": np.array(viabilite),
        "densite": np.array(densite),
        "recuperation": np.array(recuperation),
        "erreur": np.array(erreur_prediction),
        "liens": np.array(nombre_liens),
    }


# --------------------------------------------------
# Trois modèles comparés
# --------------------------------------------------

resultats = {
    "Sans mémoire / fixe":
        simuler(
            memoire_active=False,
            topologie_adaptative=False
        ),

    "Mémoire / fixe":
        simuler(
            memoire_active=True,
            topologie_adaptative=False
        ),

    "Mémoire / adaptatif":
        simuler(
            memoire_active=True,
            topologie_adaptative=True
        ),
}


# --------------------------------------------------
# Visualisation
# --------------------------------------------------

plt.style.use("dark_background")

fig, axes = plt.subplots(
    4,
    1,
    figsize=(11, 10),
    sharex=True
)

couleurs = [
    "#94a3b8",
    "#f59e0b",
    "#22d3ee"
]

for (nom, donnees), couleur in zip(
    resultats.items(),
    couleurs
):
    axes[0].plot(
        donnees["viabilite"],
        color=couleur,
        label=nom,
        linewidth=1.5
    )

    axes[1].plot(
        donnees["densite"],
        color=couleur,
        linewidth=1.5
    )

    axes[2].plot(
        donnees["recuperation"],
        color=couleur,
        linewidth=1.5
    )

    axes[3].plot(
        donnees["liens"],
        color=couleur,
        linewidth=1.5
    )

for axe in axes:
    axe.axvline(
        CHOC,
        color="white",
        linestyle="--",
        alpha=0.5
    )

axes[0].set_ylabel("Viabilité")
axes[1].set_ylabel("Densité")
axes[2].set_ylabel("Variation")
axes[3].set_ylabel("Nombre de liens")
axes[3].set_xlabel("Temps")

axes[0].set_title(
    "Mémoire → sélection des liens → réorganisation topologique"
)

axes[0].legend(fontsize=8)

fig.tight_layout()

plt.savefig(
    "resultat_reseau_relationnel_corrige.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# Résultats numériques
# --------------------------------------------------

print()
print("===== RÉSULTATS =====")
print()

for nom, donnees in resultats.items():

    viabilite_finale = donnees["viabilite"][-1]
    densite_finale = donnees["densite"][-1]
    liens_finaux = donnees["liens"][-1]

    # Moyenne après la perturbation
    erreur_post_choc = donnees["erreur"][CHOC:].mean()

    print(nom)
    print(
        "  Viabilité finale :",
        round(viabilite_finale, 5)
    )
    print(
        "  Densité finale   :",
        round(densite_finale, 5)
    )
    print(
        "  Liens finaux     :",
        int(liens_finaux)
    )
    print(
        "  Erreur post-choc :",
        round(erreur_post_choc, 5)
    )
    print()
