import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

rng = np.random.default_rng(7)

N = 32
T = 900
DT = 0.04
K = 5

MEMORY = 0.992
RENFORCEMENT = 0.80
OUBLI = 0.35
COMPETITION = 0.55


def conserver_meilleurs_liens(W, k=K):
    resultat = np.zeros_like(W)

    for i in range(N):
        indices = np.argsort(W[i])[-k:]
        resultat[i, indices] = W[i, indices]

    np.fill_diagonal(resultat, 0.0)
    return resultat


def simuler():
    x = rng.normal(0, 1, (N, 2))

    masque = rng.random((N, N)) < 0.12
    W = np.where(
        masque,
        rng.uniform(0.05, 0.35, (N, N)),
        0.0
    )

    np.fill_diagonal(W, 0.0)

    memoire = np.zeros((N, N))
    cible = np.zeros((N, 2))

    for t in range(T):

        # Dynamique relationnelle
        degre = W.sum(axis=1, keepdims=True) + 1e-9
        voisinage = (W @ x) / degre

        dx = (
            -0.55 * x
            + 0.80 * (voisinage - x)
            + 0.22 * (cible - x)
        )

        x += DT * dx
        x += rng.normal(0, 0.035, x.shape)

        # Mémoire des interactions
        distance = np.linalg.norm(
            x[:, None, :] - x[None, :, :],
            axis=2
        )

        correlation = np.exp(-distance)

        memoire = (
            MEMORY * memoire
            + (1 - MEMORY) * correlation
        )

        # Plasticité topologique
        moyenne_locale = memoire.mean(axis=1, keepdims=True)

        variation = (
            RENFORCEMENT * memoire
            - OUBLI * W
            - COMPETITION * moyenne_locale
        )

        W += DT * variation
        W = np.clip(W, 0.0, 1.0)

        # Sélection des relations dominantes
        W = conserver_meilleurs_liens(W)

    return x, W


def construire_graphe(W):
    G = nx.Graph()

    for i in range(N):
        G.add_node(i)

    for i in range(N):
        for j in range(i + 1, N):

            poids = max(W[i, j], W[j, i])

            if poids > 0.04:
                G.add_edge(i, j, weight=poids)

    return G


# --------------------------------------------------
# Simulation finale
# --------------------------------------------------

positions, W = simuler()
G = construire_graphe(W)

# Détection des communautés
communautes = list(nx.community.greedy_modularity_communities(G))

couleur_noeud = {}

for numero, communaute in enumerate(communautes):
    for noeud in communaute:
        couleur_noeud[noeud] = numero


# Position graphique des blocs
position_graphique = nx.spring_layout(
    G,
    seed=12,
    weight="weight",
    iterations=300
)

# Degré de chaque nœud
degres = dict(G.degree())
tailles = [
    100 + 45 * degres[i]
    for i in G.nodes()
]

couleurs = [
    couleur_noeud[i]
    for i in G.nodes()
]

# Affichage
plt.style.use("dark_background")

fig, ax = plt.subplots(
    figsize=(11, 9),
    dpi=150
)

fig.patch.set_facecolor("#070b16")
ax.set_facecolor("#070b16")

# Liens internes et liens entre blocs
liens_internes = []
liens_ponts = []

for i, j in G.edges():

    if couleur_noeud[i] == couleur_noeud[j]:
        liens_internes.append((i, j))
    else:
        liens_ponts.append((i, j))

# Liens internes : plus faibles et plus nombreux
nx.draw_networkx_edges(
    G,
    position_graphique,
    edgelist=liens_internes,
    edge_color="#64748b",
    alpha=0.45,
    width=1.0,
    ax=ax
)

# Ponts entre blocs : couleur vive
nx.draw_networkx_edges(
    G,
    position_graphique,
    edgelist=liens_ponts,
    edge_color="#f59e0b",
    alpha=0.95,
    width=2.5,
    ax=ax
)

nx.draw_networkx_nodes(
    G,
    position_graphique,
    node_color=couleurs,
    node_size=tailles,
    cmap=plt.cm.hsv,
    alpha=0.95,
    edgecolors="white",
    linewidths=0.7,
    ax=ax
)

nx.draw_networkx_labels(
    G,
    position_graphique,
    font_size=7,
    font_color="white",
    ax=ax
)

ax.set_title(
    "Blocs, jonctions et ponts dans un réseau à mémoire",
    fontsize=15,
    color="#e2e8f0",
    pad=18
)

ax.text(
    0.02,
    0.02,
    "liens gris : cohésion interne   |   liens orange : jonctions entre blocs",
    transform=ax.transAxes,
    color="#94a3b8",
    fontsize=9
)

ax.axis("off")

plt.tight_layout()

plt.savefig(
    "blocs_jonctions_reseau.png",
    dpi=180,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

# --------------------------------------------------
# Mesures topologiques
# --------------------------------------------------

nombre_blocs = len(communautes)
nombre_noeuds = G.number_of_nodes()
nombre_liens = G.number_of_edges()

composantes = list(nx.connected_components(G))

if nombre_liens > 0:
    clustering = nx.average_clustering(G)
else:
    clustering = 0.0

print()
print("===== STRUCTURE ÉMERGENTE =====")
print()
print("Nombre de nœuds :", nombre_noeuds)
print("Nombre de liens :", nombre_liens)
print("Nombre de blocs :", nombre_blocs)
print("Composantes connexes :", len(composantes))
print("Clustering moyen :", round(clustering, 4))
print("Ponts entre blocs :", len(liens_ponts))
print()

for numero, communaute in enumerate(communautes):
    print(
        "Bloc",
        numero + 1,
        ":",
        len(communaute),
        "nœuds ->",
        sorted(communaute)
    )

print()
print("Image créée : blocs_jonctions_reseau.png")
