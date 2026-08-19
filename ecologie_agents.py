import random
import math
import matplotlib.pyplot as plt

random.seed(7)

DUREE = 300
MAX_AGENTS = 60


class Agent:
    def __init__(self, identifiant, x, y, adulte=True):
        self.id = identifiant
        self.x = x
        self.y = y
        self.energie = 0.7
        self.blessure = 0.0
        self.fatigue = 0.0
        self.age = 0
        self.adulte = adulte
        self.vivant = True
        self.parent_a = None
        self.parent_b = None


class Monde:
    def __init__(self):
        self.agents = []
        self.ressources = []
        self.morts = []
        self.prochain_id = 0
        self.naissances = 0
        self.soins = 0
        self.reproductions = 0

        for _ in range(15):
            self.nouvel_agent(
                random.random(),
                random.random()
            )

        for _ in range(10):
            self.ressources.append([
                random.random(),
                random.random(),
                1.0
            ])

    def nouvel_agent(self, x, y, adulte=True):
        if len(self.agents) >= MAX_AGENTS:
            return None

        agent = Agent(
            self.prochain_id,
            x,
            y,
            adulte
        )

        self.prochain_id += 1
        self.agents.append(agent)
        return agent

    def vivants(self):
        return [
            a for a in self.agents
            if a.vivant
        ]

    def adultes(self):
        return [
            a for a in self.vivants()
            if a.adulte
        ]

    def petits(self):
        return [
            a for a in self.vivants()
            if not a.adulte
        ]

    def distance(self, a, x, y):
        return math.hypot(
            a.x - x,
            a.y - y
        )

    def danger(self, t, agent):
        danger = 0.0

        if 80 <= t <= 130:
            danger += 0.85

        if 210 <= t <= 240:
            danger += 0.55

        for x, y, age in self.morts:
            d = self.distance(agent, x, y)
            danger += (
                0.30
                * math.exp(-8.0 * d)
                * math.exp(-age / 100.0)
            )

        return min(1.0, danger)

    def ressource_proche(self, agent):
        disponibles = [
            r for r in self.ressources
            if r[2] > 0.05
        ]

        if not disponibles:
            return None

        return min(
            disponibles,
            key=lambda r: self.distance(
                agent,
                r[0],
                r[1]
            )
        )

    def voisin(self, agent, adulte=False):
        candidats = [
            a for a in self.vivants()
            if a.id != agent.id
            and (not adulte or a.adulte)
        ]

        if not candidats:
            return None

        return min(
            candidats,
            key=lambda a: math.hypot(
                agent.x - a.x,
                agent.y - a.y
            )
        )

    def aller_vers(self, agent, x, y, vitesse):
        dx = x - agent.x
        dy = y - agent.y
        norme = math.hypot(dx, dy) + 1e-9

        agent.x += vitesse * dx / norme
        agent.y += vitesse * dy / norme

        agent.x = max(0.0, min(1.0, agent.x))
        agent.y = max(0.0, min(1.0, agent.y))

    def fuir(self, agent, danger):
        if danger <= 0.35:
            return

        voisins = [
            a for a in self.vivants()
            if a.id != agent.id
            and self.distance(
                agent,
                a.x,
                a.y
            ) < 0.25
        ]

        if voisins:
            cx = sum(a.x for a in voisins) / len(voisins)
            cy = sum(a.y for a in voisins) / len(voisins)

            dx = agent.x - cx
            dy = agent.y - cy
            norme = math.hypot(dx, dy) + 1e-9

            agent.x += 0.025 * dx / norme
            agent.y += 0.025 * dy / norme
        else:
            agent.x += random.uniform(-0.025, 0.025)
            agent.y += random.uniform(-0.025, 0.025)

        agent.x = max(0.0, min(1.0, agent.x))
        agent.y = max(0.0, min(1.0, agent.y))

        agent.energie -= 0.008
        agent.fatigue += 0.006

    def manger(self, agent):
        ressource = self.ressource_proche(agent)

        if ressource is None:
            return

        if self.distance(
            agent,
            ressource[0],
            ressource[1]
        ) < 0.06:

            prise = min(0.20, ressource[2])
            agent.energie = min(
                1.0,
                agent.energie + prise
            )
            ressource[2] -= prise
            agent.fatigue = max(
                0.0,
                agent.fatigue - 0.02
            )

    def soigner(self, agent):
        if agent.blessure <= 0.1:
            return

        soigneur = self.voisin(
            agent,
            adulte=True
        )

        if soigneur is None:
            return

        d = math.hypot(
            agent.x - soigneur.x,
            agent.y - soigneur.y
        )

        if d < 0.08 and soigneur.energie > 0.4:
            agent.blessure = max(
                0.0,
                agent.blessure - 0.08
            )
            soigneur.energie -= 0.025
            self.soins += 1

    def nourrir_petit(self, petit):
        if petit.adulte:
            return

        parent = None

        candidats = [
            a for a in self.vivants()
            if a.id in (
                petit.parent_a,
                petit.parent_b
            )
        ]

        if candidats:
            parent = min(
                candidats,
                key=lambda a: math.hypot(
                    petit.x - a.x,
                    petit.y - a.y
                )
            )

        if parent is None:
            parent = self.voisin(
                petit,
                adulte=True
            )

        if parent is None:
            return

        d = math.hypot(
            petit.x - parent.x,
            petit.y - parent.y
        )

        if d < 0.10 and parent.energie > 0.45:
            parent.energie -= 0.04
            petit.energie = min(
                1.0,
                petit.energie + 0.04
            )

    def reproduire(self, agent):
        if not agent.adulte:
            return

        if agent.energie < 0.55:
            return

        partenaire = self.voisin(
            agent,
            adulte=True
        )

        if partenaire is None:
            return

        if partenaire.energie < 0.78:
            return

        d = math.hypot(
            agent.x - partenaire.x,
            agent.y - partenaire.y
        )

        if d > 0.10 or random.random() > 0.06:
            return

        enfant = self.nouvel_agent(
            (agent.x + partenaire.x) / 2,
            (agent.y + partenaire.y) / 2,
            adulte=False
        )

        if enfant is not None:
            enfant.parent_a = agent.id
            enfant.parent_b = partenaire.id

            agent.energie -= 0.18
            partenaire.energie -= 0.18

            self.naissances += 1
            self.reproductions += 1

    def mourir(self, agent):
        agent.vivant = False
        self.morts.append([
            agent.x,
            agent.y,
            0
        ])

    def renouveler_ressources(self):
        for ressource in self.ressources:
            ressource[2] = min(
                1.0,
                ressource[2] + 0.012
            )

        if len(self.ressources) < 18:
            self.ressources.append([
                random.random(),
                random.random(),
                0.8
            ])

    def etape(self, t):
        self.renouveler_ressources()

        for mort in self.morts:
            mort[2] += 1

        self.morts = [
            m for m in self.morts
            if m[2] < 180
        ]

        for agent in list(self.vivants()):
            danger = self.danger(t, agent)

            if danger > 0.35:
                self.fuir(agent, danger)

            elif agent.energie < 0.70:
                ressource = self.ressource_proche(agent)

                if ressource is not None:
                    self.aller_vers(
                        agent,
                        ressource[0],
                        ressource[1],
                        0.018
                    )

                self.manger(agent)

            else:
                ressource = self.ressource_proche(agent)

                if ressource is not None:
                    self.aller_vers(
                        agent,
                        ressource[0],
                        ressource[1],
                        0.008
                    )

            self.soigner(agent)

            for petit in self.petits():
                if math.hypot(
                    agent.x - petit.x,
                    agent.y - petit.y
                ) < 0.15:
                    self.nourrir_petit(petit)

            self.reproduire(agent)

            agent.age += 1
            agent.energie -= 0.0008
            agent.fatigue += 0.0008

            if danger < 0.2:
                agent.blessure = max(
                    0.0,
                    agent.blessure - 0.006
                )
                agent.fatigue = max(
                    0.0,
                    agent.fatigue - 0.004
                )

            if not agent.adulte and agent.age > 70:
                agent.adulte = True

            if danger > 0.6 and random.random() < 0.002:
                agent.blessure = min(
                    1.0,
                    agent.blessure + 0.35
                )

            if (
                agent.energie < 0.03
                or agent.blessure > 0.95
            ):
                self.mourir(agent)

            agent.x = max(
                0.0,
                min(1.0, agent.x)
            )
            agent.y = max(
                0.0,
                min(1.0, agent.y)
            )

    def statistiques(self):
        vivants = self.vivants()

        if not vivants:
            return 0, 0.0, 0, 0

        energie = sum(
            a.energie for a in vivants
        ) / len(vivants)

        blesses = sum(
            a.blessure > 0.2
            for a in vivants
        )

        petits = sum(
            not a.adulte
            for a in vivants
        )

        return (
            len(vivants),
            energie,
            blesses,
            petits
        )


monde = Monde()

populations = []
energies = []
blesses = []
petits = []
traces_morts = []

for t in range(DUREE):
    monde.etape(t)

    population, energie, blesses_t, petits_t = (
        monde.statistiques()
    )

    populations.append(population)
    energies.append(energie)
    blesses.append(blesses_t)
    petits.append(petits_t)
    traces_morts.append(len(monde.morts))

    if t % 50 == 0:
        print(
            "t =",
            t,
            "| population =",
            population,
            "| énergie =",
            round(energie, 3),
            "| blessés =",
            blesses_t,
            "| petits =",
            petits_t
        )

print()
print("===== ÉCOLOGIE DES AGENTS =====")
print()
print("Population finale :", populations[-1])
print("Énergie finale :", round(energies[-1], 4))
print("Blessés finaux :", blesses[-1])
print("Petits finaux :", petits[-1])
print("Traces de morts :", traces_morts[-1])
print("Naissances :", monde.naissances)
print("Soins réalisés :", monde.soins)
print("Reproductions :", monde.reproductions)

plt.style.use("dark_background")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True
)

temps = range(DUREE)

axes[0].plot(
    temps,
    populations,
    label="Population",
    color="#22d3ee"
)

axes[0].plot(
    temps,
    petits,
    label="Petits",
    color="#f59e0b"
)

axes[1].plot(
    temps,
    energies,
    label="Énergie moyenne",
    color="#34d399"
)

axes[1].plot(
    temps,
    blesses,
    label="Blessés",
    color="#fb7185"
)

axes[2].plot(
    temps,
    traces_morts,
    label="Traces de morts",
    color="#a78bfa"
)

for axe in axes:
    axe.grid(alpha=0.2)
    axe.legend()

axes[0].set_ylabel("Nombre")
axes[0].set_title("Population et petits")

axes[1].set_ylabel("Niveau")
axes[1].set_title("Énergie et blessures")

axes[2].set_ylabel("Traces")
axes[2].set_xlabel("Temps")
axes[2].set_title("Mémoire des morts")

axes[0].axvspan(
    80,
    130,
    color="red",
    alpha=0.12
)

axes[1].axvspan(
    80,
    130,
    color="red",
    alpha=0.12
)

axes[2].axvspan(
    80,
    130,
    color="red",
    alpha=0.12
)

fig.suptitle(
    "Écologie relationnelle des agents",
    fontsize=16
)

fig.tight_layout()

plt.savefig(
    "ecologie_agents.png",
    dpi=180,
    bbox_inches="tight"
)

print()
print("Image créée : ecologie_agents.png")
