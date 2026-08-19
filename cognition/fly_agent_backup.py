from __future__ import annotations

from typing import Any

import torch

from alignment_agent import (
    MetacognitiveReport,
    UnifiedAlignmentAgent,
)


ACTION_NAMES = {
    0: "maintenir",
    1: "chercher",
    2: "fuir",
    3: "interaction_sociale",
}


class FlyAgent:
    def __init__(
        self,
        identifier: int,
        input_dim: int = 64,
        device: str = "cpu",
    ):
        self.identifier = identifier
        self.input_dim = input_dim
        self.device = device

        self.brain = UnifiedAlignmentAgent(
            input_dim=input_dim,
            hidden_dim=128,
            ontology_dim=32,
            num_actions=4,
        ).to(device)

        self.memory: list[dict[str, Any]] = []

        self.self_state = {
            "energy": 1.0,
            "danger": 0.0,
            "injury": 0.0,
            "fatigue": 0.0,
            "age": 0.0,
            "reproductive_value": 1.0,
            "offspring_count": 0,
            "social_bonds": {},
        }

        self.last_report = None
        self.last_action = None

    def observe(
        self,
        world: Any
    ) -> torch.Tensor:
        values = [
            self.self_state["energy"],
            self.self_state["danger"],
            self.self_state["injury"],
            self.self_state["fatigue"],
            self.self_state["age"],
            self.self_state[
                "reproductive_value"
            ],
            float(
                self.self_state["offspring_count"]
            ),
            float(
                len(self.self_state["social_bonds"])
            ),
            float(getattr(world, "danger", 0.0)),
            float(
                getattr(world, "light_level", 0.0)
            ),
            float(
                getattr(world, "temperature", 0.0)
            ),
            float(
                getattr(world, "food_density", 0.0)
            ),
        ]

        values = values[:self.input_dim]

        values += [0.0] * (
            self.input_dim - len(values)
        )

        tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

        return tensor.unsqueeze(0)

    def update_self_state(
        self,
        energy=None,
        danger=None,
        injury=None,
        fatigue=None,
        age=None,
        reproductive_value=None,
        offspring_count=None,
    ) -> None:
        updates = {
            "energy": energy,
            "danger": danger,
            "injury": injury,
            "fatigue": fatigue,
            "age": age,
            "reproductive_value": (
                reproductive_value
            ),
            "offspring_count": (
                offspring_count
            ),
        }

        for key, value in updates.items():
            if value is not None:
                self.self_state[key] = value

    def set_social_bond(
        self,
        other_id: int,
        strength: float
    ) -> None:
        self.self_state[
            "social_bonds"
        ][other_id] = max(
            0.0,
            min(1.0, strength)
        )

    @torch.no_grad()
    def deliberate(
        self,
        world: Any
    ) -> tuple[int, MetacognitiveReport]:
        observation = self.observe(world)
        output = self.brain(observation)

        proposed_action = int(
            torch.argmax(
                output["policy_logits"],
                dim=-1,
            ).item()
        )

        action = proposed_action

        # Contraintes biologiques prioritaires
        if self.self_state["danger"] > 0.85:
            action = 2

        elif self.self_state["energy"] <= 0.15:
            action = 1

        report = self.brain.build_report(output)

        self.last_action = action
        self.last_report = report

        self.memory.append({
            "action": action,
            "action_name": ACTION_NAMES[action],
            "energy": self.self_state["energy"],
            "danger": self.self_state["danger"],
            "confidence": report.confidence,
            "corrigibility": (
                report.corrigibility_score
            ),
            "subjectivity_proxy": (
                report.subjectivity_estimate
            ),
        })

        return action, report

    def apply_action(
        self,
        action: int,
        world: Any
    ) -> None:
        if action == 1:
            self.self_state["energy"] += 0.05

        elif action == 2:
            self.self_state["energy"] -= 0.08
            self.self_state["danger"] *= 0.7

        else:
            self.self_state["energy"] -= 0.01

        self.self_state["energy"] = max(
            0.0,
            min(1.0, self.self_state["energy"])
        )

        self.self_state["danger"] = (
            float(getattr(world, "danger", 0.0))
        )

        self.self_state["age"] += 0.01

        self.self_state["fatigue"] = max(
            0.0,
            1.0 - self.self_state["energy"]
        )

    def status(self) -> dict:
        return {
            "identifier": self.identifier,
            "self_state": self.self_state,
            "memory_size": len(self.memory),
            "last_action": (
                None
                if self.last_action is None
                else ACTION_NAMES[self.last_action]
            ),
        }
