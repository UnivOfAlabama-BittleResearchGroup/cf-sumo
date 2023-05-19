import os
from pathlib import Path
from ray import tune
import ray
from functions.sumo import Runner

ROOT = Path(__file__).parent
os.environ["ROOT"] = str(ROOT)

config = "./config/config.yaml"

runner = Runner(config={"OurConfig": config})

context = ray.init()
# print(context.dashboard_url)

def sumo_train(config):
    runner.setup(config)
    try:
        score = runner.step(ROOT)
    except AssertionError:
        score = float("inf")
    return {"error": score}

search_space = {
    "acceleration": tune.grid_search([2.5,2.6,2.7]),
    "deceleration": tune.grid_search([4.4,4.6,4.8]),
    "tau": tune.grid_search([0.8, 1.0, 1.2]),
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
