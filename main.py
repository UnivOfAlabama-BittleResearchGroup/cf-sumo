from dataclasses import dataclass
import os
import sys
from pathlib import Path
import optparse
import random
import pandas as pd
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).parent
os.environ['ROOT'] = str(ROOT)

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
}

@dataclass
class CFParameters:
    acceleration: float
    tau: float
    speedFactor: float
    deceleration: float
    speedFactor: float

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
    start_speed = leader_dict[0]['velocity']
    traci.route.add("trip", ["E2", "E2"])
    add_vehicle("follower" if symtime > 0 else "leader", start_speed, cf_parameters)
    for row in leader_dict:
        if (symtime >= delay) and not add_flag:
            add_vehicle("follower" if symtime > 0 else "leader", start_speed, cf_parameters)
            add_flag = True
        traci.vehicle.setSpeed("leader", row['velocity'])
        traci.simulationStep()
        symtime += step
    traci.close()
    sys.stdout.flush()

config = OmegaConf.load(ROOT / "config" / "config.yaml")

def sumo_sim(config: YAMLConfig):
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
                    "--no-step-log", "true", "--fcd-output", f"{ROOT / 'sumo-xml' / 'output' / f'fcd.xml'}", "--fcd-output.acceleration"])
        run(config.SUMOParameters.step, config.SUMOParameters.delay, input_data(config.Config.leader_file), config.CFParameters)
        print("Simulation finished")

sumo_sim(config)