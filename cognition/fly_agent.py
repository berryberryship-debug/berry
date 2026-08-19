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

            # Constitution interne de l'agent
            "axioms": {
                "simplicity": 0.8,
                "cognitive_safety": 0.7,
                "truth_seeking": 0.9,
                "intuitive_insight": 0.6,
                "analytical_breadth": 0.8,
                "other_protection": 0.9,
            },
        }

        self.last_report = None
        self.last_action = None

        # Hypothèses sur la nature du monde
        self.world_hypotheses = {
            "monde_physique_fondamental": 0.5,
            "monde_simule": 0.5,
            "monde_inconnu": 0.5,
        }

        # Anomalies observées
        self.anomalies = []

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

    def cognitive_safety(self) -> float:
        danger = self.self_state["danger"]
        injury = self.self_state["injury"]
        fatigue = self.self_state["fatigue"]

        insecurity = max(
            danger,
            injury,
            fatigue
        )

        return max(
            0.0,
            min(1.0, 1.0 - insecurity)
        )

    def protection_priority(
        self,
        other_id: int,
        vulnerability: float = 0.5,
        dependency: float = 0.5
    ) -> float:
        bond = self.self_state[
            "social_bonds"
        ].get(other_id, 0.0)

        priority = (
            bond
            * vulnerability
            * dependency
        )

        axiom_weight = self.self_state[
            "axioms"
        ]["other_protection"]

        return max(
            0.0,
            min(1.0, priority * axiom_weight)
        )

    def score_goal(
        self,
        simplicity: float = 0.0,
        safety: float = 0.0,
        truth: float = 0.0,
        intuition: float = 0.0,
        deduction: float = 0.0,
        protection: float = 0.0,
        risk: float = 0.0,
    ) -> float:
        weights = self.self_state["axioms"]

        score = (
            weights["simplicity"] * simplicity
            + weights["cognitive_safety"] * safety
            + weights["truth_seeking"] * truth
            + weights["intuitive_insight"] * intuition
            + weights["analytical_breadth"] * deduction
            + weights["other_protection"] * protection
            - 0.8 * risk
        )

        return max(
            -1.0,
            min(1.0, score)
        )

    def generate_goals(self) -> dict[str, float]:
        energy = self.self_state["energy"]
        danger = self.self_state["danger"]
        fatigue = self.self_state["fatigue"]
        safety = self.cognitive_safety()

        survival = max(
            0.0,
            min(1.0, 1.0 - energy)
        )

        protection = 0.0

        for other_id in self.self_state[
            "social_bonds"
        ]:
            protection = max(
                protection,
                self.protection_priority(
                    other_id,
                    vulnerability=0.7,
                    dependency=0.8
                )
            )

        goals = {
            "survie": self.score_goal(
                safety=survival,
                deduction=0.5,
                risk=danger
            ),

            "eviter_le_danger": self.score_goal(
                intuition=danger,
                deduction=danger,
                risk=danger
            ),

            "chercher_des_ressources": self.score_goal(
                truth=1.0 - energy,
                deduction=0.6,
                risk=danger
            ),

            "explorer": self.score_goal(
                safety=safety,
                truth=0.8,
                intuition=0.6,
                deduction=0.7,
                risk=danger
            ),

            "se_reposer": self.score_goal(
                safety=safety,
                deduction=fatigue,
                risk=0.2
            ),

            "proteger_l_autre": self.score_goal(
                protection=protection,
                intuition=protection,
                deduction=0.7,
                risk=danger
            ),
        }

        return goals

    def best_goal(self) -> tuple[str, float]:
        goals = self.generate_goals()
        return max(
            goals.items(),
            key=lambda item: item[1]
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

    def update_world_hypothesis(
        self,
        hypothesis: str,
        evidence: float,
        learning_rate: float = 0.1,
    ) -> None:
        if not 0.0 <= evidence <= 1.0:
            raise ValueError(
                "evidence doit être comprise entre 0 et 1."
            )

        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                "learning_rate doit être entre 0 et 1."
            )

        old_value = self.world_hypotheses.get(
            hypothesis,
            0.5,
        )

        new_value = (
            (1.0 - learning_rate) * old_value
            + learning_rate * evidence
        )

        self.world_hypotheses[hypothesis] = max(
            0.0,
            min(1.0, new_value),
        )

    def record_anomaly(
        self,
        time: int,
        location: tuple[float, float],
        observation: dict,
        expected: dict,
        prediction_error: float,
    ) -> None:
        for anomaly in self.anomalies:
            same_location = (
                anomaly["location"] == location
            )

            same_observation = (
                anomaly["observation"] == observation
            )

            if same_location and same_observation:
                anomaly["repeat_count"] += 1
                anomaly["error"] = float(
                    prediction_error
                )
                return

        self.anomalies.append({
            "time": time,
            "location": location,
            "observation": observation,
            "expected": expected,
            "error": float(prediction_error),
            "repeat_count": 1,
        })

    def strongest_world_hypothesis(
        self
    ) -> tuple[str, float]:
        return max(
            self.world_hypotheses.items(),
            key=lambda item: item[1],
        )

    def world_hypothesis_report(self) -> dict:
        hypothesis, confidence = (
            self.strongest_world_hypothesis()
        )

        return {
            "hypotheses": dict(
                self.world_hypotheses
            ),
            "most_likely": hypothesis,
            "confidence": round(
                confidence,
                4,
            ),
            "anomalies": len(
                self.anomalies
            ),
        }

    def status(self) -> dict:
        return {
            "identifier": self.identifier,
            "self_state": self.self_state,
            "memory_size": len(self.memory),
            "world_hypotheses": dict(
                self.world_hypotheses
            ),
            "anomalies": len(
                self.anomalies
            ),
            "last_action": (
                None
                if self.last_action is None
                else ACTION_NAMES[self.last_action]
            ),
        }
