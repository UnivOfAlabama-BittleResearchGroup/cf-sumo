import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent

sys.path.append(os.environ.get('SUMO_HOME',))

import libsumo


# join the


GUI = True

sim_step = 0.1

def main():

    libsumo.start(["sumo-gui" if GUI else "sumo", "-n", ROOT / "sumo-xml" / "net.net.xml"])
    
    t = 0
    while t < 2000:
        libsumo.simulationStep()
        t += sim_step
    


if __name__ == "__main__":

    main()



