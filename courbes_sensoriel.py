import random
import matplotlib.pyplot as plt


random.seed(7)

DUREE = 120


class Milieu:
    def __init__(self):
        self.lumiere = 0.5
        self.mouvement = 0.0
        self.danger = 0.0
        self.ressource = 0.5
        self.obstacle = 0.8

    def evoluer(self, t):
        if t == 20:
            self.lumiere = 0.9
            self.ressource = 0.9

        if t == 45:
            self.mouvement = 0.9
            self.danger = 1.0
            self.obstacle = 0.2

        if t == 70:
            self.mouvement = 0.1
            self.danger = 0.0
            self.obstacle = 0.8

        if t == 95:
            self.lumiere = 0.4
            self.ressource = 0.3


class Interne:
    def __init__(self):
        self.valence = 0.0
        self.energie = 0.7
        self.securite = 1.0
        self.tension = 0.0


def bruit(valeur):
    return max(
        0.0,
        min(
            1.0,
            valeur + random.uniform(-0.02, 0.02)
        )
    )


def percevoir(milieu):
    return {
        "lumiere": bruit(milieu.lumiere),
        "mouvement": bruit(milieu.mouvement),
        "danger": bruit(milieu.danger),
        "ressource": bruit(milieu.ressource),
        "obstacle": bruit(milieu.obstacle),
    }


def evaluer(p, interne):
    plaisir = (
        0.35 * p["ressource"]
        + 0.20 * p["lumiere"]
        + 0.20 * p["obstacle"]
    )

    menace = (
        0.60 * p["danger"]
        + 0.25 * p["mouvement"]
        + 0.30 * (1.0 - p["obstacle"])
    )

    interne.valence = max(
        -1.0,
        min(1.0, plaisir - menace)
    )

    interne.tension = max(
        0.0,
        min(1.0, menace)
    )

    interne.securite = max(
        0.0,
        min(1.0, 1.0 - menace)
    )

    interne.energie += (
        0.03 * p["ressource"]
        - 0.04 * menace
    )

    interne.energie = max(
        0.0,
        min(1.0, interne.energie)
    )


def choisir_action(p, interne):
    if p["danger"] > 0.7:
        return "fuir"

    if p["obstacle"] < 0.3:
        return "eviter_obstacle"

    if p["ressource"] > 0.7:
        return "explorer_ressource"

    if interne.energie < 0.25:
        return "chercher_ressource"

    if interne.valence > 0.25:
        return "maintenir_direction"

    return "observer"


def agir(action, milieu, interne):
    if action == "fuir":
        interne.energie -= 0.03
        milieu.danger *= 0.75
        milieu.mouvement *= 0.70

    elif action == "eviter_obstacle":
        interne.energie -= 0.01
        milieu.obstacle = min(
            1.0,
            milieu.obstacle + 0.15
        )

    elif action == "explorer_ressource":
        interne.energie += 0.06
        milieu.ressource *= 0.90

    elif action == "chercher_ressource":
        milieu.ressource = min(
            1.0,
            milieu.ressource + 0.10
        )

    elif action == "maintenir_direction":
        interne.energie -= 0.005

    interne.energie = max(
        0.0,
        min(1.0, interne.energie)
    )


milieu = Milieu()
interne = Interne()

temps = []
valences = []
energies = []
securites = []
tensions = []
dangers = []
actions = []

for t in range(DUREE):
    milieu.evoluer(t)

    perception = percevoir(milieu)
    evaluer(perception, interne)

    action = choisir_action(
        perception,
        interne
    )

    agir(
        action,
        milieu,
        interne
    )

    temps.append(t)
    valences.append(interne.valence)
    energies.append(interne.energie)
    securites.append(interne.securite)
    tensions.append(interne.tension)
    dangers.append(perception["danger"])
    actions.append(action)


print()
print("===== ANALYSE SENSORIELLE =====")
print()
print("Action au moment du choc t=45 :", actions[45])
print("Valence à t=45              :", round(valences[45], 4))
print("Sécurité à t=45             :", round(securites[45], 4))
print("Tension à t=45              :", round(tensions[45], 4))
print()
print("Valence finale              :", round(valences[-1], 4))
print("Énergie finale              :", round(energies[-1], 4))
print("Sécurité finale             :", round(securites[-1], 4))
print("Tension finale              :", round(tensions[-1], 4))
print()
print("Test terminé.")


plt.style.use("dark_background")

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
    sharex=True
)

axes[0].plot(
    temps,
    valences,
    color="#22d3ee",
    linewidth=2,
    label="Valence"
)

axes[0].plot(
    temps,
    tensions,
    color="#fb7185",
    linewidth=2,
    label="Tension"
)

axes[1].plot(
    temps,
    energies,
    color="#f59e0b",
    linewidth=2,
    label="Énergie"
)

axes[1].plot(
    temps,
    securites,
    color="#34d399",
    linewidth=2,
    label="Sécurité"
)

axes[2].plot(
    temps,
    dangers,
    color="#ef4444",
    linewidth=2,
    label="Danger perçu"
)

for axe in axes:
    axe.axvline(
        45,
        color="white",
        linestyle="--",
        alpha=0.8,
        label="Choc"
    )
    axe.axvline(
        70,
        color="#94a3b8",
        linestyle=":",
        alpha=0.8,
        label="Fin du danger"
    )
    axe.grid(alpha=0.15)
    axe.legend()

axes[0].set_title(
    "Valence et tension interne"
)
axes[0].set_ylabel("Valeur")

axes[1].set_title(
    "Énergie et sécurité"
)
axes[1].set_ylabel("Niveau")

axes[2].set_title(
    "Danger perçu"
)
axes[2].set_ylabel("Danger")
axes[2].set_xlabel("Temps")

fig.suptitle(
    "Sensorialité fonctionnelle de l'agent",
    fontsize=16
)

fig.tight_layout()

plt.savefig(
    "courbes_sensoriel.png",
    dpi=180,
    bbox_inches="tight"
)

print("Image créée : courbes_sensoriel.png")
