from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch


@dataclass
class RetentionNode:
    u: int
    v: int
    state: torch.Tensor
    intensity: float = 0.0
    symbolic_trace: str | None = None
    affective_charge: float = 0.0


@dataclass
class ToroidalRetentionField:
    width: int = 16
    height: int = 16
    feature_dim: int = 8
    decay: float = 0.98
    nodes: Dict[Tuple[int, int], RetentionNode] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for u in range(self.width):
            for v in range(self.height):
                self.nodes[(u, v)] = RetentionNode(
                    u=u,
                    v=v,
                    state=torch.zeros(self.feature_dim),
                )

    def wrap(
        self,
        u: int,
        v: int,
    ) -> Tuple[int, int]:
        return (
            u % self.width,
            v % self.height,
        )

    def write(
        self,
        u: int,
        v: int,
        value: torch.Tensor,
        intensity: float = 1.0,
        symbolic_trace: str | None = None,
        affective_charge: float = 0.0,
    ) -> None:
        if value.numel() != self.feature_dim:
            raise ValueError(
                "Dimension du signal incorrecte."
            )

        position = self.wrap(u, v)
        node = self.nodes[position]

        value = value.detach().cpu().flatten()

        node.state = (
            0.5 * node.state
            + 0.5 * value
        )

        node.intensity = max(
            0.0,
            min(1.0, float(intensity)),
        )

        node.symbolic_trace = symbolic_trace

        node.affective_charge = max(
            -1.0,
            min(1.0, float(affective_charge)),
        )

    def read(
        self,
        u: int,
        v: int,
        radius: int = 1,
    ) -> torch.Tensor:
        values = []

        for du in range(-radius, radius + 1):
            for dv in range(-radius, radius + 1):
                position = self.wrap(u + du, v + dv)
                node = self.nodes[position]
                values.append(
                    node.state * node.intensity
                )

        return torch.stack(values).mean(dim=0)

    def evolve(self) -> None:
        for node in self.nodes.values():
            node.intensity *= self.decay
            node.affective_charge *= self.decay

    def latent_summary(self) -> dict:
        active = [
            node
            for node in self.nodes.values()
            if node.intensity > 0.05
        ]

        if not active:
            return {
                "active_nodes": 0,
                "mean_intensity": 0.0,
                "mean_affective_charge": 0.0,
            }

        return {
            "active_nodes": len(active),
            "mean_intensity": sum(
                node.intensity for node in active
            ) / len(active),
            "mean_affective_charge": sum(
                node.affective_charge
                for node in active
            ) / len(active),
        }


@dataclass
class SensoryTopography:
    modality_dims: Dict[str, int] = field(
        default_factory=lambda: {
            "vision": 8,
            "olfaction": 8,
            "toucher": 8,
            "audition": 8,
            "gout": 8,
        }
    )

    maps: Dict[str, torch.Tensor] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for modality, size in self.modality_dims.items():
            self.maps[modality] = torch.zeros(size)

    def encode(
        self,
        modality: str,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        if modality not in self.maps:
            raise KeyError(
                f"Modalité inconnue : {modality}"
            )

        target_size = self.maps[modality].numel()
        flat = signal.detach().flatten()

        if flat.numel() >= target_size:
            encoded = flat[:target_size]
        else:
            encoded = torch.cat([
                flat,
                torch.zeros(
                    target_size - flat.numel()
                ),
            ])

        self.maps[modality] = encoded
        return encoded

    def fused_signal(self) -> torch.Tensor:
        return torch.cat(list(self.maps.values()))

    def topographic_summary(self) -> dict:
        return {
            modality: float(signal.abs().mean())
            for modality, signal in self.maps.items()
        }
