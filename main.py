import os
from pathlib import Path
from ray import tune
from functions.sumo import Runner

ROOT = Path(__file__).parent
os.environ["ROOT"] = str(ROOT)

config = "./config/config.yaml"

runner = Runner(config={"OurConfig": config})

def sumo_train(config):
    runner.setup(config)
    try:
        score = runner.step(ROOT)
    except AssertionError:
        score = float("inf")
    return {"error": score}

search_space = {
    "acceleration": tune.uniform(2.0,4.0),
    "deceleration": tune.uniform(3.0,5.0),
    "tau": tune.uniform(0.1, 2.0),
    "speedFactor": 1,
    "speedMode": 31
}

analysis = tune.run(
    sumo_train,
    config=search_space,
    num_samples=1000,
    metric="error",
    mode="min"
)

print("Best config: ", analysis.get_best_config(metric="error", mode="min"))
