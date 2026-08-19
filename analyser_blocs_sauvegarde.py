import numpy as np
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


def meilleurs_liens(W):
    resultat = np.zeros_like(W)

    for i in range(N):
        indices = np.argsort(W[i])[-K:]
        resultat[i, indices] = W[i, indices]

    np.fill_diagonal(resultat, 0.0)
    return resultat


def simuler(perturbation=False):
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

    mesures = []

    for t in range(T):

        # Perturbation locale du premier bloc
        if perturbation and t == 450:
            bloc_perturbe = [2, 3, 4, 7, 11, 16, 18, 24, 26, 28]
            x[bloc_perturbe] += rng.normal(
                0,
                3,
                (len(bloc_perturbe), 2)
            )

        degre = W.sum(axis=1, keepdims=True) + 1e-9
        voisinage = (W @ x) / degre

        dx = (
            -0.55 * x
            + 0.80 * (voisinage - x)
            + 0.22 * (cible - x)
        )

        x += DT * dx
        x += rng.normal(0, 0.035, x.shape)

        distance = np.linalg.norm(
            x[:, None, :] - x[None, :, :],
            axis=2
        )

        correlation = np.exp(-distance)

        memoire = (
            MEMORY * memoire
            + (1 - MEMORY) * correlation
        )

        moyenne_locale = memoire.mean(axis=1, keepdims=True)

        W += DT * (
            RENFORCEMENT * memoire
            - OUBLI * W
            - COMPETITION * moyenne_locale
        )

        W = np.clip(W, 0.0, 1.0)
        W = meilleurs_liens(W)

        mesures.append((x.copy(), W.copy()))

    return mesures


def graphe_final(W):
    G = nx.Graph()

    for i in range(N):
        G.add_node(i)

    for i in range(N):
        for j in range(i + 1, N):
            poids = max(W[i, j], W[j, i])

            if poids > 0.04:
                G.add_edge(i, j, weight=poids)

    return G


def analyser_structure(W, communautes):
    internes = 0
    externes = 0

    for i in range(N):
        for j in range(i + 1, N):

            poids = max(W[i, j], W[j, i])

            if poids <= 0.04:
                continue

            meme_bloc = False

            for bloc in communautes:
                if i in bloc and j in bloc:
                    meme_bloc = True
                    break

            if meme_bloc:
                internes += 1
            else:
                externes += 1

    densite_interne = internes / max(1, sum(
        len(bloc) * (len(bloc) - 1) / 2
        for bloc in communautes
    ))

    total_possible_externe = 0

    for a in range(len(communautes)):
        for b in range(a + 1, len(communautes)):
            total_possible_externe += (
                len(communautes[a]) * len(communautes[b])
            )

    densite_externe = externes / max(1, total_possible_externe)

    rapport = densite_interne / max(densite_externe, 1e-9)

    return internes, externes, densite_interne, densite_externe, rapport


def distance_moyenne(x, bloc):
    valeurs = x[bloc]
    centre = valeurs.mean(axis=0)
    return np.mean(np.linalg.norm(valeurs - centre, axis=1))


# --------------------------------------------------
# Simulation sans puis avec perturbation
# --------------------------------------------------

normal = simuler(perturbation=False)
perturbe = simuler(perturbation=True)

x_normal, W_normal = normal[-1]
x_perturbe, W_perturbe = perturbe[-1]

G = graphe_final(W_normal)

communautes = list(
    nx.community.greedy_modularity_communities(G)
)

print()
print("===== ANALYSE DES BLOCS =====")
print()

for numero, bloc in enumerate(communautes):
    print(
        "Bloc",
        numero + 1,
        ":",
        len(bloc),
        "nœuds ->",
        sorted(bloc)
    )

internes, externes, densite_interne, densite_externe, rapport = (
    analyser_structure(W_normal, communautes)
)

print()
print("Liens internes :", internes)
print("Liens externes :", externes)
print("Densité interne :", round(densite_interne, 4))
print("Densité externe :", round(densite_externe, 4))
print("Rapport interne/externe :", round(rapport, 4))

print()
print("===== EFFET DE LA PERTURBATION =====")
print()

for numero, bloc in enumerate(communautes):

    avant = distance_moyenne(x_normal, list(bloc))
    apres = distance_moyenne(x_perturbe, list(bloc))

    variation = apres - avant

    print("Bloc", numero + 1)
    print("  Cohésion avant :", round(avant, 4))
    print("  Cohésion après :", round(apres, 4))
    print("  Variation      :", round(variation, 4))
    print()

print("Interprétation :")
print()
print("Un rapport interne/externe supérieur à 1 indique")
print("une cohésion interne plus forte que la cohésion externe.")
print()
print("Une faible variation après perturbation indique")
print("une bonne résilience du bloc.")
