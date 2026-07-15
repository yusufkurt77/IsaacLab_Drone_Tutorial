import argparse
from isaaclab.app import AppLauncher

# Parse the arguments
parser = argparse.ArgumentParser(description="This is an empty simulation")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This must be initialized before importng the lbrary
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---After Launch---
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG

def create_scene():

    # The Ground plane
    ground_config = sim_utils.GroundPlaneCfg()
    ground_config.func("/World/ground", ground_config)

    # The Lignt
    light_config = sim_utils.DomeLightCfg(intensity=3000.0)
    light_config.func("/World/Light", light_config)

def drone_object():

    # The Drone
    drone_config = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    drone_config.init_state.pos = (0.0, 0.0, 0.5)
    drone = Articulation(drone_config)
    return drone

def main():

    # The Simulation
    simulation_config = SimulationCfg(dt=0.01)
    simulation = SimulationContext(simulation_config)
    simulation.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.3])

    create_scene()
    drone = drone_object()
    simulation.reset()
    print("Simulation Started")

    # The Simulation will keep running
    while simulation_app.is_running():
        simulation.step()
        drone.update(simulation_config.dt)

if __name__=="__main__":
    main()
    simulation_app.close()