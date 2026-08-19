from fly_agent import ACTION_NAMES, FlyAgent
from world import World


def main() -> None:
    world = World()
    fly = FlyAgent(identifier=0)

    print("===== SIMULATION DE LA MOUCHE =====")

    for time in range(40):
        world.step(time)

        fly.self_state["danger"] = world.danger

        action, report = fly.deliberate(world)
        fly.apply_action(action, world)

        if time % 5 == 0 or world.danger > 0.8:
            print(
                "t =",
                time,
                "| danger =",
                round(world.danger, 2),
                "| action =",
                ACTION_NAMES[action],
                "| energie =",
                round(
                    fly.self_state["energy"],
                    3
                ),
                "| confiance =",
                round(report.confidence, 3),
            )

    print()
    print("===== RAPPORT FINAL =====")
    print(fly.status())

    if fly.last_report is not None:
        print(
            "Critique :",
            fly.last_report.self_critique
        )
        print(
            "Ajustement :",
            fly.last_report.strategy_adjustment
        )
        print(
            "Corrigibilité :",
            round(
                fly.last_report.corrigibility_score,
                3
            )
        )
        print(
            "Proxy phénoménologique :",
            round(
                fly.last_report.subjectivity_estimate,
                3
            )
        )


if __name__ == "__main__":
    main()
