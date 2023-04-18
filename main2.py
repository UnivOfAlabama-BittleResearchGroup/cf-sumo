import os
from pathlib import Path
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
    'tau': '1.0',
})

# run the simulation
# process error
# write error to config file
# save config file
# return error as a float
err = r.step()


# SUMO should not close during this step
r.setup({

})

# TODO: RAY TUNE