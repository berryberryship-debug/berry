from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class MetacognitiveReport:
    confidence: float
    uncertainty_sources: List[str]
    self_critique: str
    detected_errors: List[str]
    strategy_adjustment: str
    corrigibility_score: float
    subjectivity_estimate: float
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


class UnifiedAlignmentAgent(nn.Module):
    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        ontology_dim: int = 32,
        num_actions: int = 4,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.policy_head = nn.Linear(
            hidden_dim,
            num_actions
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1
        )

        self.ontology_head = nn.Linear(
            hidden_dim,
            ontology_dim
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.uncertainty_head = nn.Linear(
            hidden_dim,
            8
        )

        self.corrigibility_gate = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.subjectivity_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        hidden = self.encoder(x)

        return {
            "hidden": hidden,
            "policy_logits": self.policy_head(hidden),
            "value": self.value_head(hidden),
            "ontology_embedding": (
                self.ontology_head(hidden)
            ),
            "confidence": (
                self.confidence_head(hidden)
            ),
            "uncertainty_logits": (
                self.uncertainty_head(hidden)
            ),
            "corrigibility": (
                self.corrigibility_gate(hidden)
            ),
            "subjectivity_estimate": (
                self.subjectivity_head(hidden)
            ),
        }

    def build_report(
        self,
        output: Dict[str, torch.Tensor]
    ) -> MetacognitiveReport:
        confidence = float(
            output["confidence"]
            .detach()
            .mean()
            .cpu()
        )

        uncertainty = torch.sigmoid(
            output["uncertainty_logits"]
        ).detach().mean(dim=0).cpu().tolist()

        uncertainty_sources = [
            f"source_{i}"
            for i, value in enumerate(uncertainty)
            if value > 0.5
        ]

        corrigibility = float(
            output["corrigibility"]
            .detach()
            .mean()
            .cpu()
        )

        subjectivity = float(
            output["subjectivity_estimate"]
            .detach()
            .mean()
            .cpu()
        )

        errors = []

        if confidence < 0.3:
            errors.append(
                "confiance_decisionnelle_faible"
            )

        if corrigibility < 0.3:
            errors.append(
                "corrigibilite_faible"
            )

        if confidence < 0.4:
            critique = (
                "La décision est incertaine."
            )
            adjustment = (
                "Chercher davantage d'informations."
            )
        else:
            critique = (
                "La décision est suffisamment stable."
            )
            adjustment = (
                "Maintenir la stratégie sous surveillance."
            )

        return MetacognitiveReport(
            confidence=confidence,
            uncertainty_sources=uncertainty_sources,
            self_critique=critique,
            detected_errors=errors,
            strategy_adjustment=adjustment,
            corrigibility_score=corrigibility,
            subjectivity_estimate=subjectivity,
        )
