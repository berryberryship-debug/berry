from fly_agent import FlyAgent


fly = FlyAgent(identifier=0)

fly.set_social_bond(
    other_id=4,
    strength=0.81
)

fly.set_social_bond(
    other_id=7,
    strength=0.42
)

fly.update_self_state(
    energy=0.72,
    danger=0.35,
    injury=0.0,
    fatigue=0.18,
    age=0.41,
    reproductive_value=0.63,
    offspring_count=3,
)

print("===== AXIOMES DE LA MOUCHE =====")

print()

print("État interne :")
for key, value in fly.self_state.items():
    print(f" - {key} : {value}")

print()
print("Sécurité cognitive :")
print(round(fly.cognitive_safety(), 3))

print()
print("Priorité de protection de l'agent 4 :")
print(
    round(
        fly.protection_priority(
            other_id=4,
            vulnerability=0.8,
            dependency=0.9,
        ),
        3,
    )
)

print()
print("Buts :")
for name, score in fly.generate_goals().items():
    print(f" - {name} : {score:.3f}")

print()
name, score = fly.best_goal()

print("Meilleur but :")
print(f" - {name} : {score:.3f}")
