import numpy as np
import networkx as nx

# ============================================================
# ANALYSE D'UN RÉSEAU RELATIONNEL À MÉMOIRE
# Même état initial, même bruit, un seul choc expérimental
# ============================================================

rng = np.random.default_rng(7)

# Paramètres généraux
N = 32
T = 900
DT = 0.04
CHOC = 450

# Topologie et mémoire
K = 5
MEMORY = 0.992
RENFORCEMENT = 0.80
OUBLI = 0.35
COMPETITION = 0.55

# Bloc volontairement perturbé
BLOC_PERTURBE = [2, 3, 4, 7, 11, 16, 18, 24, 26, 28]


# ============================================================
# Création du réseau initial
# ============================================================

def creer_etat_initial():
    """
    Crée un état initial identique pour les deux expériences.
    """

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


# ============================================================
# Limitation du nombre de connexions
# ============================================================

def conserver_meilleurs_liens(W):
    """
    Chaque nœud garde seulement ses K liens les plus forts.
    Cela empêche la saturation du réseau.
    """

    resultat = np.zeros_like(W)

    for i in range(N):

        indices = np.argsort(W[i])[-K:]

        resultat[i, indices] = W[i, indices]

    np.fill_diagonal(resultat, 0.0)

    return resultat


# ============================================================
# Simulation
# ============================================================

def simuler(
    etat_initial,
    reseau_initial,
    memoire_initiale,
    bruits,
    perturbation=False
):
    """
    Simule le réseau.

    Les deux expériences utilisent exactement :
    - le même état initial ;
    - le même réseau initial ;
    - la même mémoire initiale ;
    - les mêmes bruits.

    Seule la perturbation change.
    """

    x = etat_initial.copy()
    W = reseau_initial.copy()
    memoire = memoire_initiale.copy()

    cible = np.zeros((N, 2))

    # Historique complet
    historique_x = []
    historique_W = []
    historique_cohesion = []
    historique_viabilite = []
    historique_densite = []

    for t in range(T):

        # ----------------------------------------------------
        # Choc expérimental appliqué à un seul modèle
        # ----------------------------------------------------
        if perturbation and t == CHOC:

            x[BLOC_PERTURBE] += rng.normal(
                0,
                3,
                (len(BLOC_PERTURBE), 2)
            )

        # ----------------------------------------------------
        # Interaction avec le voisinage
        # ----------------------------------------------------
        degre = W.sum(axis=1, keepdims=True) + 1e-9

        voisinage = (W @ x) / degre

        # ----------------------------------------------------
        # Dynamique locale
        # ----------------------------------------------------
        dx = (
            -0.55 * x
            + 0.80 * (voisinage - x)
            + 0.22 * (cible - x)
        )

        x = x + DT * dx

        # Même bruit dans les deux expériences
        x = x + bruits[t]

        # ----------------------------------------------------
        # Mémoire relationnelle
        # ----------------------------------------------------
        distances = np.linalg.norm(
            x[:, None, :] - x[None, :, :],
            axis=2
        )

        correlations = np.exp(-distances)

        memoire = (
            MEMORY * memoire
            + (1.0 - MEMORY) * correlations
        )

        # ----------------------------------------------------
        # Réorganisation topologique
        # ----------------------------------------------------
        moyenne_locale = memoire.mean(
            axis=1,
            keepdims=True
        )

        variation_W = (
            RENFORCEMENT * memoire
            - OUBLI * W
            - COMPETITION * moyenne_locale
        )

        W = W + DT * variation_W

        W = np.clip(W, 0.0, 1.0)

        W = conserver_meilleurs_liens(W)

        # ----------------------------------------------------
        # Mesures
        # ----------------------------------------------------
        viabilite = np.mean(
            np.sum((x - cible) ** 2, axis=1)
        )

        densite = W.sum() / (N * (N - 1))

        historique_x.append(x.copy())
        historique_W.append(W.copy())
        historique_viabilite.append(viabilite)
        historique_densite.append(densite)

    return {
        "x": np.array(historique_x),
        "W": np.array(historique_W),
        "viabilite": np.array(historique_viabilite),
        "densite": np.array(historique_densite),
    }


# ============================================================
# Construction d'un graphe final
# ============================================================

def construire_graphe(W):
    """
    Transforme la matrice de connexions en graphe non orienté.
    """

    G = nx.Graph()

    for i in range(N):
        G.add_node(i)

    for i in range(N):

        for j in range(i + 1, N):

            poids = max(W[i, j], W[j, i])

            if poids > 0.04:
                G.add_edge(
                    i,
                    j,
                    weight=poids
                )

    return G


# ============================================================
# Détection des blocs
# ============================================================

def detecter_blocs(G):
    """
    Détecte des communautés avec modularité gloutonne.
    """

    blocs = list(
        nx.community.greedy_modularity_communities(G)
    )

    return blocs


# ============================================================
# Cohésion d'un bloc
# ============================================================

def cohesion_bloc(positions, bloc):
    """
    Mesure la distance moyenne des nœuds au centre du bloc.
    """

    indices = list(bloc)

    points = positions[indices]

    centre = points.mean(axis=0)

    distance = np.linalg.norm(
        points - centre,
        axis=1
    )

    return np.mean(distance)


# ============================================================
# Analyse des liens internes et externes
# ============================================================

def analyser_liens(W, blocs):
    """
    Compte les liens internes et externes.
    """

    appartenance = {}

    for numero, bloc in enumerate(blocs):

        for noeud in bloc:
            appartenance[noeud] = numero

    liens_internes = 0
    liens_externes = 0

    for i in range(N):

        for j in range(i + 1, N):

            poids = max(W[i, j], W[j, i])

            if poids <= 0.04:
                continue

            if appartenance[i] == appartenance[j]:
                liens_internes += 1
            else:
                liens_externes += 1

    total_possible_interne = 0

    for bloc in blocs:

        taille = len(bloc)

        total_possible_interne += (
            taille * (taille - 1) / 2
        )

    total_possible_externe = 0

    for a in range(len(blocs)):

        for b in range(a + 1, len(blocs)):

            total_possible_externe += (
                len(blocs[a]) * len(blocs[b])
            )

    densite_interne = (
        liens_internes
        / max(1, total_possible_interne)
    )

    densite_externe = (
        liens_externes
        / max(1, total_possible_externe)
    )

    rapport = (
        densite_interne
        / max(densite_externe, 1e-9)
    )

    return {
        "internes": liens_internes,
        "externes": liens_externes,
        "densite_interne": densite_interne,
        "densite_externe": densite_externe,
        "rapport": rapport,
    }


# ============================================================
# Programme principal
# ============================================================

print()
print("Initialisation du réseau...")

etat_initial, reseau_initial, memoire_initiale = (
    creer_etat_initial()
)

# Bruit commun aux deux expériences
bruits = rng.normal(
    0,
    0.035,
    (T, N, 2)
)

print("Simulation normale...")
normal = simuler(
    etat_initial,
    reseau_initial,
    memoire_initiale,
    bruits,
    perturbation=False
)

print("Simulation perturbée...")
perturbe = simuler(
    etat_initial,
    reseau_initial,
    memoire_initiale,
    bruits,
    perturbation=True
)

# Réseau final de référence
positions_finales = normal["x"][-1]
W_final = normal["W"][-1]

G = construire_graphe(W_final)
blocs = detecter_blocs(G)

analyse = analyser_liens(
    W_final,
    blocs
)


# ============================================================
# Résultats de structure
# ============================================================

print()
print("===== STRUCTURE ÉMERGENTE =====")
print()

print("Nombre de nœuds :", G.number_of_nodes())
print("Nombre de liens :", G.number_of_edges())
print("Nombre de blocs :", len(blocs))
print(
    "Composantes connexes :",
    nx.number_connected_components(G)
)

if G.number_of_edges() > 0:
    clustering = nx.average_clustering(G)
else:
    clustering = 0.0

print(
    "Clustering moyen :",
    round(clustering, 4)
)

print()
print("Liens internes :", analyse["internes"])
print("Liens externes :", analyse["externes"])
print(
    "Densité interne :",
    round(analyse["densite_interne"], 4)
)
print(
    "Densité externe :",
    round(analyse["densite_externe"], 4)
)
print(
    "Rapport interne/externe :",
    round(analyse["rapport"], 4)
)

print()

for numero, bloc in enumerate(blocs):

    print(
        "Bloc",
        numero + 1,
        ":",
        len(bloc),
        "nœuds ->",
        sorted(bloc)
    )


# ============================================================
# Résilience des blocs
# ============================================================

print()
print("===== RÉSILIENCE DES BLOCS =====")
print()

for numero, bloc in enumerate(blocs):

    coh_normal_avant = cohesion_bloc(
        normal["x"][CHOC - 1],
        bloc
    )

    coh_perturbe_avant = cohesion_bloc(
        perturbe["x"][CHOC - 1],
        bloc
    )

    coh_perturbe_pic = cohesion_bloc(
        perturbe["x"][CHOC],
        bloc
    )

    coh_normal_finale = cohesion_bloc(
        normal["x"][-1],
        bloc
    )

    coh_perturbe_finale = cohesion_bloc(
        perturbe["x"][-1],
        bloc
    )

    ecart_avant = abs(
        coh_normal_avant
        - coh_perturbe_avant
    )

    ecart_pic = abs(
        coh_normal_avant
        - coh_perturbe_pic
    )

    ecart_final = abs(
        coh_normal_finale
        - coh_perturbe_finale
    )

    print("Bloc", numero + 1)
    print(
        "  Cohésion avant choc :",
        round(coh_normal_avant, 5)
    )
    print(
        "  Écart au pic        :",
        round(ecart_pic, 5)
    )
    print(
        "  Écart final         :",
        round(ecart_final, 5)
    )
    print()


# ============================================================
# Récupération globale
# ============================================================

viabilite_normale = normal["viabilite"]
viabilite_perturbee = perturbe["viabilite"]

difference_avant = np.mean(
    np.abs(
        viabilite_normale[:CHOC]
        - viabilite_perturbee[:CHOC]
    )
)

difference_apres = np.mean(
    np.abs(
        viabilite_normale[CHOC:]
        - viabilite_perturbee[CHOC:]
    )
)

print("===== RÉCUPÉRATION GLOBALE =====")
print()

print(
    "Différence moyenne avant choc :",
    round(difference_avant, 6)
)

print(
    "Différence moyenne après choc :",
    round(difference_apres, 6)
)

print(
    "Viabilité normale finale :",
    round(viabilite_normale[-1], 6)
)

print(
    "Viabilité perturbée finale :",
    round(viabilite_perturbee[-1], 6)
)

print()
