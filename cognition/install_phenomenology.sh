#!/data/data/com.termux/files/usr/bin/bash

set -Eeuo pipefail

PROJECT="$HOME/quantumlab"
COGNITION="$PROJECT/cognition"
BACKUP="$COGNITION/backup_$(date +%Y%m%d_%H%M%S)"

red='\033[0;31m'
green='\033[0;32m'
yellow='\033[1;33m'
reset='\033[0m'

fail() {
    echo -e "${red}ERREUR : $1${reset}" >&2
    exit 1
}

trap 'fail "Erreur à la ligne $LINENO."' ERR

echo "===== INSTALLATION ROBUSTE ====="

[ -d "$PROJECT" ] || fail "Dossier introuvable : $PROJECT"
[ -d "$COGNITION" ] || fail "Dossier introuvable : $COGNITION"

cd "$COGNITION"

command -v python >/dev/null 2>&1 \
    || fail "Python n'est pas installé."

python - <<'PY'
try:
    import torch
except ImportError:
    raise SystemExit(
        "PyTorch est absent. Installez-le avec : pip install torch"
    )
print("PyTorch disponible.")
PY

mkdir -p "$BACKUP"

for file in \
    alignment_agent.py \
    fly_agent.py \
    world.py \
    main.py \
    phenomenological_field.py \
    phenomenological_fly.py \
    test_phenomenology.py
do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP/$file"
    fi
done

echo -e "${green}Sauvegarde créée : $BACKUP${reset}"

cat > phenomenological_field.py <<'PY'
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
PY

cat > phenomenological_fly.py <<'PY'
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

        self.self_state.update({
            "local_vacuity_proxy": 1.0,
            "retention_coherence": 0.0,
            "sensory_integration": 0.0,
        })

    def retain_experience(
        self,
        signals: dict[str, torch.Tensor],
        u: int,
        v: int,
        affective_charge: float = 0.0,
        symbolic_trace: str | None = None,
    ) -> None:
        for modality, signal in signals.items():
            self.sensory_topography.encode(
                modality,
                signal,
            )

        fused = self.sensory_topography.fused_signal()
        value = fused[:8]

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

        summary = (
            self.retention_field.latent_summary()
        )

        coherence = (
            summary["mean_intensity"]
            + 0.2 * abs(
                summary["mean_affective_charge"]
            )
        )

        self.self_state[
            "retention_coherence"
        ] = max(0.0, min(1.0, coherence))

        self.self_state[
            "sensory_integration"
        ] = float(fused.abs().mean().item())

        self.self_state[
            "local_vacuity_proxy"
        ] = max(
            0.0,
            min(
                1.0,
                1.0 - self.self_state[
                    "retention_coherence"
                ],
            ),
        )

        self.retention_field.evolve()

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
PY

cat > test_phenomenology.py <<'PY'
import torch

from phenomenological_fly import (
    PhenomenologicalFly,
)


def main() -> None:
    fly = PhenomenologicalFly(
        identifier=0
    )

    signals = {
        "vision": torch.rand(8),
        "olfaction": torch.rand(8),
        "toucher": torch.rand(8),
        "audition": torch.rand(8),
        "gout": torch.rand(8),
    }

    fly.retain_experience(
        signals=signals,
        u=3,
        v=5,
        affective_charge=0.7,
        symbolic_trace="presence_autrui",
    )

    print("===== TEST PHÉNOMÉNOLOGIQUE =====")
    print(fly.phenomenological_report())


if __name__ == "__main__":
    main()
PY

echo "===== COMPILATION ====="

python -m py_compile \
    phenomenological_field.py \
    phenomenological_fly.py \
    test_phenomenology.py

echo -e "${green}Compilation réussie.${reset}"

echo "===== TEST ====="

python test_phenomenology.py

echo
echo -e "${green}Installation terminée avec succès.${reset}"
echo "Sauvegarde : $BACKUP"

