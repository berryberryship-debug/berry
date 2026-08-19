from dataclasses import dataclass


@dataclass
class Prediction:
    expected_energy: float
    expected_danger: float
    confidence: float


class PredictiveModel:
    def predict(
        self,
        energy: float,
        danger: float,
        action: str
    ) -> Prediction:

        if action == "fuir":
            expected_energy = max(
                0.0,
                energy - 0.08
            )

            expected_danger = max(
                0.0,
                danger - 0.25
            )

        elif action == "chercher":
            expected_energy = min(
                1.0,
                energy + 0.12
            )

            expected_danger = danger

        else:
            expected_energy = max(
                0.0,
                energy - 0.01
            )

            expected_danger = danger

        return Prediction(
            expected_energy=expected_energy,
            expected_danger=expected_danger,
            confidence=0.5
        )

    def error(
        self,
        prediction: Prediction,
        energy: float,
        danger: float
    ) -> float:

        erreur_energie = abs(
            prediction.expected_energy - energy
        )

        erreur_danger = abs(
            prediction.expected_danger - danger
        )

        return (
            erreur_energie + erreur_danger
        ) / 2.0
