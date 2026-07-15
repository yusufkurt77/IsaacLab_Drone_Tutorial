import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fly the Drone")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
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

    body_id = drone.find_bodies("body")[0]
    robot_mass = drone.root_physx_view.get_masses().sum().item()
    print(f"Mass of the drone with id {body_id}:{robot_mass}")

    # g = 9.81
    robot_weight = robot_mass * 9.81

    forces = torch.zeros(1, 1, 3, device=drone.device)
    torques = torch.zeros(1, 1, 3, device=drone.device)

    target_z = 1.0
    kp = 6.0 # Proportional Gain
    kd = 3.0 # Derivative Gain

    count = 0
    while simulation_app.is_running():

        # Observation
        position = drone.data.root_pos_w[0]
        velocity = drone.data.root_lin_vel_w[0]

        # Decision
        height_error = target_z - position[2]
        thrust_z = robot_weight + robot_mass * (kp * height_error - kd * velocity[2])
        thrust_z = max(thrust_z, 0.0)
        forces[0, 0, 2] = thrust_z

        # Action
        drone.set_external_force_and_torque(forces, torques, body_ids=body_id)
        drone.write_data_to_sim()
        simulation.step()
        drone.update(simulation_config.dt)

        if count % 50 == 0:
            print(f"step {count:4d} | pos z = {position[2]:.3f} m | thrust = {thrust_z:.3f} N")
        count += 1

if __name__=="__main__":
    main()
    simulation_app.close()