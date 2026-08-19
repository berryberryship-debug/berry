from dataclasses import dataclass, field


@dataclass
class SelfModel:
    energy: float = 0.7
    danger: float = 0.0
    injury: float = 0.0
    fatigue: float = 0.0
    social_bonds: dict[int, float] = field(
        default_factory=dict
    )
    identity: str = "organisme_simule"

    def update(
        self,
        energy: float,
        danger: float,
        injury: float,
        fatigue: float
    ) -> None:
        self.energy = energy
        self.danger = danger
        self.injury = injury
        self.fatigue = fatigue

    def status(self) -> dict:
        return {
            "energy": round(self.energy, 4),
            "danger": round(self.danger, 4),
            "injury": round(self.injury, 4),
            "fatigue": round(self.fatigue, 4),
            "identity": self.identity,
        }
