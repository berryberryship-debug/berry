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
