from dataclasses import dataclass


def limiter(valeur, minimum, maximum):
    return max(minimum, min(maximum, valeur))


@dataclass
class ControleurFlou:
    gain_erreur: float = 0.35
    gain_variation: float = 0.12
    correction_max: float = 0.25
    zone_morte: float = 0.02
    erreur_precedente: float = 0.0

    def calculer(self, cible, valeur):
        erreur = cible - valeur
        variation = erreur - self.erreur_precedente

        self.erreur_precedente = erreur

        if abs(erreur) < self.zone_morte:
            return 0.0

        correction = (
            self.gain_erreur * erreur
            + self.gain_variation * variation
        )

        return limiter(
            correction,
            -self.correction_max,
            self.correction_max
        )


if __name__ == "__main__":
    controleur = ControleurFlou()

    cible = 0.0
    valeur = 2.0

    print()
    print("===== TEST DE LOGIQUE FLOUE =====")
    print()

    for instant in range(20):
        correction = controleur.calculer(
            cible,
            valeur
        )

        valeur += correction

        print(
            "t =",
            instant,
            "| valeur =",
            round(valeur, 5),
            "| correction =",
            round(correction, 5)
        )

    print()
    print("Test terminé.")
