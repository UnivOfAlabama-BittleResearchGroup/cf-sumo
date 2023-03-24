import os
import sys
from pathlib import Path
import optparse
import random
import pandas as pd

ROOT = Path(__file__).parent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


def add_vehicle(name, start_speed):
    traci.vehicle.add(name, "trip", departSpeed=start_speed)
    traci.vehicle.setSpeedMode(name, 32)

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


def run(step, delay, leader_dict):
    """execute the TraCI control loop"""  # dataframe version
    add_flag = False
    symtime = 0
    start_speed = leader_dict[0]['velocity']
    traci.route.add("trip", ["E2", "E2"])
    add_vehicle("follower" if symtime > 0 else "leader", start_speed)
    for row in leader_dict:
        if (symtime >= delay) and not add_flag:
            add_vehicle("follower" if symtime > 0 else "leader", start_speed)
            add_flag = True
        traci.vehicle.setSpeed("leader", row['velocity'])
        traci.simulationStep()
        symtime += step
    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    options = get_options()
    step = 0.1
    delay = 3  # seconds
    leader_file = ROOT / "data" / "test.csv"

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # this is the normal way of using traci. sumo is started as a
    # subprocess and then the python script connects and runs
    traci.start([sumoBinary, "-n", f"{ROOT / 'sumo-xml' / 'net.net.xml'}", "--step-length", f"{step}",
                "--no-step-log", "true", "--fcd-output", f"{ROOT / 'sumo-xml' / 'output' / 'fcd.xml'}", "--fcd-output.acceleration"])
    run(step, delay, input_data(leader_file))
