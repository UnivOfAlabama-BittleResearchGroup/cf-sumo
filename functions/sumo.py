from copy import deepcopy
from ray.tune import Trainable
from omegaconf import OmegaConf
from sumolib import checkBinary
import traci
import traci.constants as tc
from typing import Dict, Optional, Callable, Any
import pandas as pd
from logging import Logger


from .error_metrics import error_metrics


# define cf_param_functions
cf_param_functions = {
    "acceleration": lambda traci_, vehID, accel: traci_.vehicle.setAccel(vehID, accel),
    "tau": lambda traci_, vehID, tau: traci_.vehicle.setTau(vehID, tau),
    "speedFactor": lambda traci_, vehID, speedFactor: traci_.vehicle.setSpeedFactor(
        vehID, speedFactor
    ),
    "deceleration": lambda traci_, vehID, decel: traci_.vehicle.setDecel(vehID, decel),
    "speedMode": lambda traci_, vehID, val: traci_.vehicle.setSpeedMode(
        vehID, val
    ),
}




class Runner(Trainable):

    CONNECTION_NUM = 0

    def __init__(self, 
                 config: dict,
                 logger_creator: Callable[[Dict[str, Any]], Logger] = None,
                 remote_checkpoint_dir: Optional[str] = None,
                 sync_function_tpl: Optional[str] = None,
                 **kwargs):

        try:
            self._config = config.pop('OurConfig')
            # if not instance of OmegaConf, then it is a dict, convert to OmegaConf
            if not isinstance(self._config, OmegaConf):
                self._config = OmegaConf.load(self._config)
        except KeyError:
            raise KeyError("You must pass in a config object")

        self._traci: traci.connection = None

        self._sim_time = 0
        self._sim_step = self._config.SUMOParameters.step

        self._step_counter = 0

        self._cf_parameters: dict = None

        self._rw_df: pd.DataFrame = pd.read_csv(self._config.Config.leader_file,) # real world file
        self._rw_array = self._rw_df[['leadvelocity', 'followvelocity']].to_numpy() # real world file

        # NEED to get time offset from config file
        self._follower_offset = self._rw_df.loc[self._rw_df['followvelocity'].notna(), 'seconds'].iloc[0]

        # IMPORTANT: This probably shouldn't change
        super().__init__(config, logger_creator, remote_checkpoint_dir, sync_function_tpl, **kwargs)


    def setup(self, config: Dict[str, float]):
        # TODO: Add your setup code here
        self._cf_parameters = config

        if self._sim_time > 1e6:
            self._sim_time = 0

        if self._traci is None:
            self._start_sumo()
        

        return super().setup(config)

    def step(self):
        # TODO: Add your training code here
        # this is where the simulation will run
        
        # create a copy of the config file
        run_config = deepcopy(self._config)

        speed_list = self.run()

        error_metrics(speed_list, run_config, self._rw_array)

        #TODO: save run_config to file. How do we name the file?
        
    
        self._step_counter += 1    
        return run_config.Error.val

    def cleanup(self):
        # TODO: Add your cleanup code here
        self._cleanup_traci()
        return super().cleanup()
    
    def _cleanup_traci(self, ):
        self._traci.close()
        self._traci = None
        self._sim_time = 0


    def _start_sumo(self):
        # Start SUMO
        if self._config.Config.sumo_gui:
            sumoBinary = checkBinary("sumo-gui")
        else:
            sumoBinary = checkBinary("sumo")

        traci.start(
            [
                sumoBinary,
                *map(str, self._config.SUMOParameters.cmd_line)
            ],
            label=str(Runner.CONNECTION_NUM)
        )

        self._traci = traci.getConnection(
            str(Runner.CONNECTION_NUM)
        )

        print(f"Starting SUMO with connection number {Runner.CONNECTION_NUM}")

        Runner.CONNECTION_NUM += 1

    def run(self, ):
        add_flag = False
        start_speed = self._rw_array[0][0] # TODO follower and leader start speed
        
        if 'trip' not in self._traci.route.getIDList():
            self._traci.route.add("trip", ["E2", "E2"])


        leader_name = f"leader_{int(self._sim_time)}"
        follower_name = f"follower_{int(self._sim_time)}"

        self._add_vehicle(leader_name, start_speed, {'speedMode': 32})


        start_time = self._traci.simulation.getTime()

        pos_list = []

        for row in self._rw_array:
            if ((self._sim_time - start_time) >= self._follower_offset) and not add_flag:
                self._add_vehicle(
                    follower_name, start_speed, self._cf_parameters
                )
                traci.vehicle.subscribe(follower_name, (tc.VAR_SPEED, tc.VAR_DISTANCE))
                traci.vehicle.subscribe(leader_name, (tc.VAR_SPEED, tc.VAR_DISTANCE))

                add_flag = True
            self._traci.vehicle.setSpeed(leader_name, row[0])
            self._traci.simulationStep()
            # get subscription results
            positions = self._traci.vehicle.getAllSubscriptionResults()
            if add_flag:
                pos_list.append([
                    positions[follower_name][tc.VAR_DISTANCE],
                    positions[leader_name][tc.VAR_DISTANCE],
                ])
                print(pos_list[-1])

            self._sim_time += self._sim_step
        return pos_list

    def _add_vehicle(self, name: str, start_speed: float, cf_parameters: dict):
        self._traci.vehicle.add(name, "trip", departSpeed=start_speed)
        self._traci.vehicle.moveTo(name, "E2_0", 0, tc.MOVE_AUTOMATIC)
        for param, value in cf_parameters.items():
            cf_param_functions[param](self._traci, name, value)


