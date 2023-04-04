from dataclasses import dataclass
import os
import sys
from pathlib import Path
import optparse
import random
import pandas as pd
import yaml
import itertools
from omegaconf import OmegaConf
import time

ROOT = Path(__file__).parent
os.environ['ROOT'] = str(ROOT)

config = OmegaConf.load(ROOT / "config" / "config.yaml")

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

cf_param_functions = {
    "acceleration": lambda traci_, vehID, accel: traci_.vehicle.setAccel(vehID, accel),
    "tau": lambda traci_, vehID, tau: traci_.vehicle.setTau(vehID, tau),
    "speedFactor": lambda traci_, vehID, speedFactor: traci_.vehicle.setSpeedFactor(vehID, speedFactor),
    "deceleration": lambda traci_, vehID, decel: traci_.vehicle.setDecel(vehID, decel),
    "SpeedMode": lambda traci_, vehID, SpeedMode: traci_.vehicle.setSpeedMode(vehID, SpeedMode if vehID == "leader" else 31),
}

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
    
    with open(config_file_path, 'w') as yaml_file:
        yaml.dump(config_data, yaml_file, default_flow_style=False, sort_keys=False)

    print(f"Config # '{run_id}' created successfully!")


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

# def run(step, delay):
#     """execute the TraCI control loop"""
#     add_flag = False
#     symtime = 0
#     start_speed = 5 # from parsed file
#     traci.simulationStep()
#     traci.route.add("trip", ["E2", "E2"])
#     add_vehicle("follower" if symtime > 0 else "leader", start_speed)
#     while symtime < 3600:
#         if (symtime >= delay) and not add_flag:
#             add_vehicle("follower" if symtime > 0 else "leader", start_speed)
#             add_flag = True
#         traci.simulationStep()
#         symtime += step
#     traci.close()
#     sys.stdout.flush()


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
        print("Simulation finished")

def main(config: YAMLConfig):
    tau = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    acceleration = [2.0, 2.6, 3.2]
    deceleration = [4.0, 4.5, 5.0]
    speed_factor = [0.07, 0.1, 0.13]
    all_combinations = list(itertools.product(tau, acceleration, deceleration, speed_factor))
    run_ids = len(all_combinations)
    for run_id in range(run_ids):
        config_create(run_id, all_combinations[run_id])
        sumo_sim(run_id, config)
    
    return run_ids

if __name__ == "__main__":
    start_time = time.time()
    run_ids = main(config)
    print(f"{run_ids} simulations finished --- %s seconds ---" % (time.time() - start_time))