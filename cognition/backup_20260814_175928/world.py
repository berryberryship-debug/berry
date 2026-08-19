from dataclasses import dataclass


@dataclass
class World:
    danger: float = 0.0
    light_level: float = 0.5
    temperature: float = 0.5
    food_density: float = 0.5

    def step(self, time: int) -> None:
        if time == 10:
            self.danger = 0.9

        if time == 25:
            self.danger = 0.1

        self.danger = max(
            0.0,
            min(1.0, self.danger)
        )
