from dataclasses import dataclass, field


@dataclass
class Belief:
    statement: str
    confidence: float
    evidence: int = 1


@dataclass
class Metacognition:
    beliefs: dict[str, Belief] = field(
        default_factory=dict
    )

    def revise(
        self,
        statement: str,
        surprise: float
    ) -> None:

        if statement not in self.beliefs:
            self.beliefs[statement] = Belief(
                statement=statement,
                confidence=0.5,
                evidence=1
            )
            return

        belief = self.beliefs[statement]
        belief.evidence += 1

        if surprise < 0.2:
            belief.confidence = min(
                1.0,
                belief.confidence + 0.08
            )
        else:
            belief.confidence = max(
                0.0,
                belief.confidence - 0.12
            )

    def explain(self) -> list[str]:
        return [
            (
                f"{belief.statement} "
                f"(confiance={belief.confidence:.2f}, "
                f"preuves={belief.evidence})"
            )
            for belief in self.beliefs.values()
        ]
