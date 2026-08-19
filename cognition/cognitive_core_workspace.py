from episodic_memory import (
    Episode,
    EpisodicMemory
)
from predictive_model import PredictiveModel
from self_model import SelfModel
from metacognition import Metacognition
from workspace import GlobalWorkspace


def choisir_action(
    energie: float,
    danger: float,
    idee: str | None
) -> str:

    if idee == "danger_critique":
        return "fuir"

    if idee == "energie_faible":
        return "chercher"

    if danger > 0.6:
        return "fuir"

    if energie < 0.3:
        return "chercher"

    return "maintenir"


def executer() -> None:
    memory = EpisodicMemory()
    predictor = PredictiveModel()
    self_model = SelfModel()
    metacognition = Metacognition()
    workspace = GlobalWorkspace()

    energie = 0.7
    danger = 0.0

    for temps in range(100):

        if temps == 35:
            danger = 0.95

        if temps == 65:
            danger = 0.05

        # Compétition des contenus
        workspace.soumettre(
            "danger_critique",
            danger,
            "perception",
            {"danger": danger}
        )

        workspace.soumettre(
            "energie_faible",
            1.0 - energie,
            "modele_de_soi",
            {"energie": energie}
        )

        idee = workspace.diffuser()

        nom_idee = (
            idee.nom
            if idee is not None
            else None
        )

        action = choisir_action(
            energie,
            danger,
            nom_idee
        )

        prediction = predictor.predict(
            energie,
            danger,
            action
        )

        if action == "fuir":
            energie -= 0.08
            danger *= 0.88

        elif action == "chercher":
            energie += 0.05

        else:
            energie -= 0.01

        energie = max(
            0.0,
            min(1.0, energie)
        )

        erreur = predictor.error(
            prediction,
            energie,
            danger
        )

        episode = Episode(
            time=temps,
            observation={
                "energie": energie,
                "danger": danger
            },
            action=action,
            consequence={
                "energie": energie,
                "danger": danger
            },
            surprise=erreur,
            valence=-1.0 if danger > 0.6 else 0.5,
            causes=["perception", "workspace"]
        )

        memory.remember(episode)

        self_model.update(
            energy=energie,
            danger=danger,
            injury=0.0,
            fatigue=max(0.0, 1.0 - energie)
        )

        if nom_idee is not None:
            metacognition.revise(
                f"{action} répond à {nom_idee}",
                erreur
            )
        else:
            metacognition.revise(
                f"{action} en l'absence d'alerte globale",
                erreur
            )

        if temps % 10 == 0 or idee is not None:
            print(
                "t =",
                temps,
                "| idée diffusée =",
                nom_idee,
                "| action =",
                action,
                "| énergie =",
                round(energie, 3),
                "| danger =",
                round(danger, 3)
            )

    print()
    print("===== GLOBAL WORKSPACE =====")
    print("Épisodes :", len(memory.episodes))
    print("Broadcasts :", len(workspace.historique))
    print("Modèle de soi :", self_model.status())
    print("Croyances :")

    for croyance in metacognition.explain():
        print(" -", croyance)


if __name__ == "__main__":
    executer()
