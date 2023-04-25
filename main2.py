import os
from pathlib import Path
from ray import tune
from ray.tune import (
    run_experiments,
    grid_search,
    register_env,
)

from functions.sumo import Runner


ROOT = Path(__file__).parent
os.environ["ROOT"] = str(ROOT)

config = "./config/config.yaml"


# this should open the leader/follower file 
r = Runner(config={'OurConfig': config})


# this should start sumo IF it is not already running
# TODO: it should create a copy of the config file & write the passed CF parameters to it
r.setup({
    'acceleration': 1.0,
    'deceleration': 1.0,
    'tau': 1.0,
})

# run the simulation
# process error
# write error to config file
# save config file
# return error as a float
err = r.step()
print(err)


# SUMO should not close during this step
r.setup({
    'acceleration': 2.0,
    'deceleration': 6.0,
    'tau': .5,
})

err = r.step()
print(err)


r.cleanup()




# # TODO: RAY TUNE
# search_space = {
#     "acceleration": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
#     "deceleration": tune.choice([1, 2, 3]),
#     "tau": tune.uniform(0.1, 1.0),
# }

# tuner = tune.Tuner(err, param_space=search_space)
# results = tuner.fit()
# print(results.get_best_result(metric="error", mode="min").config)