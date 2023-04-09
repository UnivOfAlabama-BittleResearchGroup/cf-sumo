# main file, v1 attempt at combining simulation.py and analysis.py
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
from simulation import *
from analysis import *

# first, need to load ROOT
ROOT = Path(__file__).parent
os.environ['ROOT'] = str(ROOT)

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
    "SpeedMode": lambda traci_, vehID, SpeedMode: traci_.vehicle.setSpeedMode(vehID, SpeedMode if vehID == "leader" else 31),
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

