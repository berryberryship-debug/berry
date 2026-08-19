import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Courbe temporelle de propagation et de récupération
# ============================================================

rng = np.random.default_rng(7)

N = 32
T = 900
DT = 0.04
CHOC = 450

K = 5
MEMORY = 0.992
RENFORCEMENT = 0.80
OUBLI = 0.35
COMPETITION = 0.55

BLOCS = [
    [2, 3, 4, 7, 11, 16, 18, 24, 26, 28],
    [5, 8, 9, 10, 13, 15, 27, 30, 31],
    [1, 6, 17, 19, 21, 22, 25, 29],
    [0, 12, 14, 20, 23],
]


def garder_meilleurs_liens(W):
    resultat = np.zeros_like(W)

    for i in range(N):
        indices = np.argsort(W[i])[-K:]
        resultat[i, indices] = W[i, indices]

    np.fill_diagonal(resultat, 0.0)

    return resultat


def creer_etat_initial():
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


def simuler(etat_initial, reseau_initial, memoire_initiale,
            bruits, perturbation=False):

    x = etat_initial.copy()
    W = reseau_initial.copy()
    memoire = memoire_initiale.copy()

    cible = np.zeros((N, 2))

    historique_x = []
    historique_W = []

    for t in range(T):

        # Choc uniquement dans l'expérience perturbée
        if perturbation and t == CHOC:
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
            + 0.22 * (cible - x)
        )

        x = x + DT * dx
        x = x + bruits[t]

        distance = np.linalg.norm(
            x[:, None, :] - x[None, :, :],
            axis=2
        )

        correlation = np.exp(-distance)

        memoire = (
            MEMORY * memoire
            + (1.0 - MEMORY) * correlation
        )

        moyenne_locale = memoire.mean(
            axis=1,
            keepdims=True
        )

        W += DT * (
            RENFORCEMENT * memoire
            - OUBLI * W
            - COMPETITION * moyenne_locale
        )

        W = np.clip(W, 0.0, 1.0)
        W = garder_meilleurs_liens(W)

        historique_x.append(x.copy())
        historique_W.append(W.copy())

    return {
        "x": np.array(historique_x),
        "W": np.array(historique_W),
    }


def cohesion(positions, bloc):
    points = positions[bloc]
    centre = points.mean(axis=0)
    distances = np.linalg.norm(points - centre, axis=1)
    return np.mean(distances)


def viabilite(positions):
    return np.mean(np.sum(positions ** 2, axis=1))


# ============================================================
# Même départ, même bruit, un seul choc
# ============================================================

etat_initial, reseau_initial, memoire_initiale = (
    creer_etat_initial()
)

bruits = rng.normal(
    0,
    0.035,
    (T, N, 2)
)

normal = simuler(
    etat_initial,
    reseau_initial,
    memoire_initiale,
    bruits,
    perturbation=False
)

perturbe = simuler(
    etat_initial,
    reseau_initial,
    memoire_initiale,
    bruits,
    perturbation=True
)


# ============================================================
# Calcul des courbes
# ============================================================

# Cohésion de chaque bloc dans l'expérience normale
cohesions_normales = np.zeros((4, T))
cohesions_perturbees = np.zeros((4, T))

for t in range(T):

    for b, bloc in enumerate(BLOCS):

        cohesions_normales[b, t] = cohesion(
            normal["x"][t],
            bloc
        )

        cohesions_perturbees[b, t] = cohesion(
            perturbe["x"][t],
            bloc
        )

# Écart entre la trajectoire normale et perturbée
ecarts_blocs = np.zeros((4, T))

for b in range(4):

    ecarts_blocs[b] = np.abs(
        cohesions_perturbees[b]
        - cohesions_normales[b]
    )

viabilite_normale = np.array([
    viabilite(x)
    for x in normal["x"]
])

viabilite_perturbee = np.array([
    viabilite(x)
    for x in perturbe["x"]
])

ecart_global = np.abs(
    viabilite_perturbee
    - viabilite_normale
)


# ============================================================
# Affichage
# ============================================================

plt.style.use("dark_background")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True
)

couleurs = [
    "#22d3ee",
    "#f59e0b",
    "#a78bfa",
    "#fb7185",
]


# Courbes de cohésion du bloc perturbé
axes[0].plot(
    cohesions_normales[0],
    color="#94a3b8",
    linewidth=2,
    label="Bloc 1 — trajectoire normale"
)

axes[0].plot(
    cohesions_perturbees[0],
    color="#22d3ee",
    linewidth=2,
    label="Bloc 1 — trajectoire perturbée"
)

axes[0].axvline(
    CHOC,
    color="white",
    linestyle="--",
    alpha=0.7,
    label="Choc"
)

axes[0].set_ylabel("Cohésion du bloc 1")
axes[0].set_title(
    "Réaction du bloc directement perturbé"
)
axes[0].legend(fontsize=8)


# Propagation de l'écart dans chaque bloc
for b in range(4):

    axes[1].plot(
        ecarts_blocs[b],
        color=couleurs[b],
        linewidth=1.8,
        label=f"Bloc {b + 1}"
    )

axes[1].axvline(
    CHOC,
    color="white",
    linestyle="--",
    alpha=0.7
)

axes[1].set_ylabel("Écart à la référence")
axes[1].set_title(
    "Propagation de la perturbation entre les blocs"
)
axes[1].legend(fontsize=8)


# Récupération globale
axes[2].plot(
    ecart_global,
    color="#f97316",
    linewidth=2,
    label="Écart global"
)

axes[2].axvline(
    CHOC,
    color="white",
    linestyle="--",
    alpha=0.7
)

axes[2].set_xlabel("Temps")
axes[2].set_ylabel("Écart global")
axes[2].set_title(
    "Retour du réseau vers sa trajectoire de référence"
)
axes[2].legend(fontsize=8)

for axe in axes:
    axe.grid(
        True,
        alpha=0.12
    )

fig.suptitle(
    "Propagation, mémoire et récupération topologique",
    fontsize=16
)

fig.tight_layout()

plt.savefig(
    "courbe_recuperation.png",
    dpi=180,
    bbox_inches="tight"
)

print()
print("===== COURBE DE RÉCUPÉRATION =====")
print()
print("Image créée : courbe_recuperation.png")
print()
print(
    "Écart global au moment du choc :",
    round(ecart_global[CHOC], 5)
)
print(
    "Écart global final :",
    round(ecart_global[-1], 7)
)
print(
    "Écart maximal du bloc 1 :",
    round(ecarts_blocs[0].max(), 5)
)
print(
    "Écart final du bloc 1 :",
    round(ecarts_blocs[0, -1], 7)
)
print()
