# main file, v1 attempt at combining simulation.py and analysis.py
from dataclasses import dataclass
import os
import sys
from pathlib import Path
import optparse
import random
import pandas as pd
import numpy as np
import yaml
import itertools
from omegaconf import OmegaConf
import time
import sumolib
from genetic_algorithm import genetic_algorithm

# first, need to load ROOT
ROOT = Path(__file__).parent
os.environ['ROOT'] = str(ROOT)

# set config directory
config_dir = ROOT / "sumo-xml" / "output" / "configs"

# set config file
config = OmegaConf.load(ROOT / "config" / "config.yaml")

# check if SUMO_HOME is in environment variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# import required modules for sumo sim
from sumolib import checkBinary
import traci

# define cf_param_functions
cf_param_functions = {
    "acceleration": lambda traci_, vehID, accel: traci_.vehicle.setAccel(vehID, accel),
    "tau": lambda traci_, vehID, tau: traci_.vehicle.setTau(vehID, tau),
    "speedFactor": lambda traci_, vehID, speedFactor: traci_.vehicle.setSpeedFactor(vehID, speedFactor),
    "deceleration": lambda traci_, vehID, decel: traci_.vehicle.setDecel(vehID, decel),
    "SpeedMode": lambda traci_, vehID, SpeedMode: traci_.vehicle.setSpeedMode(vehID, 32 if vehID == "leader" else 31),
}

# define dataclasses
@dataclass
class CFParameters:
    acceleration: float
    tau: float
    speedFactor: float
    deceleration: float
    speedFactor: float
    SpeedMode: int

@dataclass
class Config:
    leader_file: str
    run_id: int
    output_path: str
    sumo_gui: bool

@dataclass
class SUMOParameters:
    step: float
    delay: float

@dataclass
class YAMLConfig:
    Config: Config
    CFParameters: CFParameters
    SUMOParameters: SUMOParameters

# define config create function
def config_create(run_id, cf_parameters):
    output_path = "${oc.env:ROOT}/sumo-xml/output"
    output_file_path = ROOT / "sumo-xml" / "output" / "configs"
    config_file_path = ROOT / "config" / "config.yaml"

    config_data = {
        'Config': {
            'leader_file': '${oc.env:ROOT}/data/test.csv',
            'run_id': run_id,
            'output_path': output_path,
            'sumo_gui': False
        },
        'CFParameters': {
            'tau': cf_parameters[0],
            'acceleration': cf_parameters[1],
            'deceleration': cf_parameters[2],
            'speedFactor': cf_parameters[3],
            'SpeedMode': 31
        },
        'SUMOParameters': {
            'step': 0.1,
            'delay': 3
        }
    }

    config_copy = f'{run_id}_config.yaml'
    copy_path = f'{output_file_path}/{config_copy}'
    with open(copy_path, 'w') as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"Config #{run_id} created successfully")
    return OmegaConf.create(config_data)

# sumo functions
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def add_vehicle(name, start_speed, cf_parameters):
    traci.vehicle.add(name, "trip", departSpeed=start_speed)
    #traci.vehicle.setActionStepLength(name, 1)
    for param, value in cf_parameters.items():
        cf_param_functions[param](traci, name, value)

def input_data(leader_file):
    leader_df = pd.read_csv(leader_file)
    leader_dict = leader_df.to_dict("records")
    return leader_dict

def run(step, delay, leader_dict, cf_parameters):
    """execute the TraCI control loop"""  # dataframe version
    add_flag = False
    symtime = 0
    start_speed = leader_dict[0]['leadvelocity']
    traci.route.add("trip", ["E2", "E2"])
    add_vehicle("follower" if symtime > 0 else "leader", start_speed, cf_parameters)
    for row in leader_dict:
        if (symtime >= delay) and not add_flag:
            add_vehicle("follower" if symtime > 0 else "leader", start_speed, cf_parameters)
            add_flag = True
        traci.vehicle.setSpeed("leader", row['leadvelocity'])
        traci.simulationStep()
        symtime += step
    traci.close()
    sys.stdout.flush()

def xml_to_df(xml_path):
        rows = []
        xml_path = str(xml_path)

        for r in sumolib.xml.parse_fast_nested(xml_path, "timestep", ["time"], "vehicle", ["id", "speed", "pos", "acceleration"]):
            rows.append(
                {
                    **r[0]._asdict(),
                    **r[1]._asdict()
                }
            )

        return pd.DataFrame(rows)

def sumo_sim(run_id, config: YAMLConfig):
    if __name__ == "__main__":
        options = get_options()
        # this script has been called from the command line. It will start sumo as a
        # server, then connect and run
        if config.Config.sumo_gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        # this is the normal way of using traci. sumo is started as a
        # subprocess and then the python script connects and runs
        traci.start([sumoBinary, "-n", f"{ROOT / 'sumo-xml' / 'net.net.xml'}", "--step-length", f"{config.SUMOParameters.step}",
                    "--no-step-log", "true", "--fcd-output", f"{ROOT / 'sumo-xml' / 'output' / f'{run_id}_fcd.xml'}", "--fcd-output.acceleration"])
        run(config.SUMOParameters.step, config.SUMOParameters.delay, input_data(config.Config.leader_file), config.CFParameters)
        sim_df = xml_to_df(f"{ROOT / 'sumo-xml' / 'output' / f'{run_id}_fcd.xml'}")
        sim_df = sim_df.query("id == 'follower'")
        print(f"Simulation #{run_id} finished")

        return sim_df

# define error metrics
def rmsn(run_id, sim_df, error_df):
        # root mean square error normalized
        n = len(error_df)
        sim_df = sim_df.reset_index(drop=True)
        # print(error_df['followposition'] - sim_df['pos'].astype(float))
        pos_observed_sum = np.sum(error_df['followposition'].astype(float))
        # print(error_df)
        # print(sim_df)

        # e1 = np.sum(error_df["followposition"].astype(float))
        # e2 = np.sum(sim_df["pos"].astype(float))
        # print(e1, e2)

        rmsn_numerator = np.sqrt(n * np.sum(np.square(error_df['followposition'] - sim_df['pos'].astype(float))))
        rmsn_val = rmsn_numerator / (pos_observed_sum)

        config_file = config_dir / f"{run_id}_config.yaml"
        config = OmegaConf.load(config_file)
        config.CFParameters.rmsn = float(rmsn_val)
        OmegaConf.save(config, config_file)

        return rmsn_val

def rmspe(run_id, sim_df, error_df):
        # root mean square percentage error
        n = len(error_df)
        sim_df = sim_df.reset_index(drop=True)
        dev = np.square((sim_df['pos'].astype(float) - error_df['followposition']) / error_df['followposition'])
        rmspe_val = np.sqrt(np.sum(dev) / n)

        config_file = config_dir / f"{run_id}_config.yaml"
        config = OmegaConf.load(config_file)
        config.CFParameters.rmspe = float(rmspe_val)
        OmegaConf.save(config, config_file)

        return rmspe_val

def mpe(run_id, sim_df, error_df):
    # mean percentage error
    n = len(error_df)
    sim_df = sim_df.reset_index(drop=True)
    mean = np.sum((sim_df['pos'].astype(float) - error_df['followposition']) / error_df['followposition'])
    mpe_val = mean / n

    config_file = config_dir / f"{run_id}_config.yaml"
    config = OmegaConf.load(config_file)
    config.CFParameters.mpe = float(mpe_val)
    OmegaConf.save(config, config_file)

    return mpe_val

# define our error function
def error_metric(run_id, sim_df):
    actual_df = pd.read_csv(ROOT / "data" / "test.csv")
    error_df = actual_df.copy()
    error_df = error_df.dropna()
    error_df = error_df.drop(columns=['epoch_time', 'leadvelocity', 'leadposition', 'leadacceleration'])
    error_df = error_df.reset_index(drop=True)

    # can change with any error function
    rmsn_val = rmsn(run_id, sim_df, error_df)
    rmspe_val = rmspe(run_id, sim_df, error_df)
    mpe_val = mpe(run_id, sim_df, error_df)

    print(f"RMSN = {rmsn_val}")
    return rmsn_val

# parameter_sim now takes params as a dictionary and returns error
def parameter_sim(run_counter, params):
    run_id = run_counter["count"]
    run_counter["count"] += 1

    cf_parameters = [params['tau'], params['acceleration'], params['deceleration'], params['speedFactor']]
    config = config_create(run_id, cf_parameters)
    sim_df = sumo_sim(run_id, config)
    error = error_metric(run_id, sim_df)
    return error

# def parameter_sim(run_counter, cf_parameters):
#     run_id = run_counter["count"]
#     run_counter["count"] += 1

#     config = config_create(run_id, cf_parameters)
#     sim_df = sumo_sim(run_id, config)
#     error_metric(run_id, sim_df)  # returns error

# # define our parameter simulation function
# def parameter_sim(run_id, cf_parameters):
#     config = config_create(run_id, cf_parameters)
#     sim_df = sumo_sim(run_id, config)
#     error_metric(run_id, sim_df)

if __name__ == "__main__":
    # Create a run_counter dictionary
    run_counter = {"count": 0}

    # Define the parameter ranges for the genetic algorithm
    param_ranges = {
        'tau': (0.1, 2.0), # default 1
        'acceleration': (1.6, 5.0), # default 2.6
        'deceleration': (1.5, 7.0), # default 4.5
        'speedFactor': (0.3, 2.0) # default 1
    }

    # Run the genetic algorithm
    population_size = 50
    num_generations = 100
    num_parents = 10
    best_params = genetic_algorithm(population_size, num_generations, num_parents, param_ranges, parameter_sim)

    print("Best parameters found by the genetic algorithm:", best_params)

    # Run the simulation with the best parameters
    parameter_sim(run_counter, best_params)
