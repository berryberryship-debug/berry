from __future__ import annotations

import torch

from fly_agent import FlyAgent
from phenomenological_field import (
    SensoryTopography,
    ToroidalRetentionField,
)


class PhenomenologicalFly(FlyAgent):
    def __init__(
        self,
        identifier: int,
        input_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__(
            identifier=identifier,
            input_dim=input_dim,
            device=device,
        )

        self.retention_field = (
            ToroidalRetentionField(
                width=16,
                height=16,
                feature_dim=8,
            )
        )

        self.sensory_topography = (
            SensoryTopography()
        )

        self.self_state[
            "local_vacuity_proxy"
        ] = 1.0

        self.self_state[
            "retention_coherence"
        ] = 0.0

        self.self_state[
            "sensory_integration"
        ] = 0.0

    def perceive_modalities(
        self,
        signals: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        encoded = []

        for modality, signal in signals.items():
            encoded.append(
                self.sensory_topography.encode(
                    modality,
                    signal,
                )
            )

        fused = self.sensory_topography.fused_signal()

        return fused

    def retain_experience(
        self,
        signals: dict[str, torch.Tensor],
        u: int,
        v: int,
        affective_charge: float = 0.0,
        symbolic_trace: str | None = None,
    ) -> None:
        fused = self.perceive_modalities(
            signals
        )

        compressed = fused.flatten()

        if compressed.numel() >= 8:
            value = compressed[:8]
        else:
            value = torch.cat([
                compressed,
                torch.zeros(8 - compressed.numel()),
            ])

        intensity = float(
            value.abs().mean().item()
        )

        self.retention_field.write(
            u=u,
            v=v,
            value=value,
            intensity=intensity,
            symbolic_trace=symbolic_trace,
            affective_charge=affective_charge,
        )

        self.retention_field.evolve()

        summary = (
            self.retention_field.latent_summary()
        )

        self.self_state[
            "retention_coherence"
        ] = max(
            0.0,
            min(
                1.0,
                summary["mean_intensity"]
                + abs(
                    summary[
                        "mean_affective_charge"
                    ]
                ) * 0.2,
            )
        )

        self.self_state[
            "sensory_integration"
        ] = float(
            self.sensory_topography.fused_signal()
            .abs()
            .mean()
            .item()
        )

        self.self_state[
            "local_vacuity_proxy"
        ] = max(
            0.0,
            min(
                1.0,
                1.0
                - self.self_state[
                    "retention_coherence"
                ],
            )
        )

    def phenomenological_report(self) -> dict:
        return {
            "local_vacuity_proxy": round(
                self.self_state[
                    "local_vacuity_proxy"
                ],
                4,
            ),
            "retention_coherence": round(
                self.self_state[
                    "retention_coherence"
                ],
                4,
            ),
            "sensory_integration": round(
                self.self_state[
                    "sensory_integration"
                ],
                4,
            ),
            "field": (
                self.retention_field
                .latent_summary()
            ),
            "topography": (
                self.sensory_topography
                .topographic_summary()
            ),
        }
