from fly_agent import FlyAgent


fly = FlyAgent(identifier=0)

fly.record_anomaly(
    time=12,
    location=(0.5, 0.5),
    observation={"temperature": 0.9},
    expected={"temperature": 0.5},
    prediction_error=0.4,
)

fly.record_anomaly(
    time=20,
    location=(0.5, 0.5),
    observation={"temperature": 0.9},
    expected={"temperature": 0.5},
    prediction_error=0.35,
)

fly.update_world_hypothesis(
    "monde_simule",
    evidence=0.8,
)

print("===== HYPOTHÈSES DU MONDE =====")
print(fly.world_hypothesis_report())

print()
print("===== ÉTAT FINAL =====")
print(fly.status())
