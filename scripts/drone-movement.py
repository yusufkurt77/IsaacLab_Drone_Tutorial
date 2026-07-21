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
    simulation.set_camera_view(eye=[0.6, 0.6, 0.9], target=[0.0, 0.0, 0.8])
    create_scene()
    drone = drone_object()
    simulation.reset()
    print("Simulation Started")
    body_id = drone.find_bodies("body")[0]
    robot_mass = drone.root_physx_view.get_masses().sum().item()
    print(f"Mass of the drone with id {body_id}:{robot_mass}")
    # g = 9.81
    robot_weight = robot_mass * 9.81

    # Propeller joints, spun cosmetically below - purely visual, no effect on
    # the actual flight physics above.
    prop_joint_ids, prop_joint_names = drone.find_joints(".*rotor.*|.*prop.*|m[1-4]_joint")
    print(f"Found propeller joints: {prop_joint_names}")
    print(f"All joint names on this asset: {drone.joint_names}")

    forces = torch.zeros(1, 1, 3, device=drone.device)
    torques = torch.zeros(1, 1, 3, device=drone.device)
    target_z = 1.0
    kp = 6.0 # Proportional Gain
    kd = 3.0 # Derivative Gain

    base_spin = 300.0
    spin_per_newton = 150.0
    # Manually tracked propeller angle - written directly to the sim each step
    # instead of using set_joint_velocity_target, since that has to be "chased"
    # by the joint's own actuator (stiffness/damping/torque limits), and a weak
    # or position-holding actuator can silently fail to reach it. Writing joint
    # state directly guarantees the visual spin speed regardless of whatever
    # actuator config this asset happens to have.
    prop_angle = torch.zeros(1, len(prop_joint_ids), device=drone.device) if prop_joint_ids else None

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

        # Cosmetic
        if prop_joint_ids:
            spin_speed = base_spin + spin_per_newton * thrust_z
            prop_angle = (prop_angle + spin_speed * simulation_config.dt) % (2 * torch.pi)
            prop_vel = torch.full((1, len(prop_joint_ids)), spin_speed, device=drone.device)
            drone.write_joint_state_to_sim(prop_angle, prop_vel, joint_ids=prop_joint_ids)

        drone.write_data_to_sim()
        simulation.step()
        drone.update(simulation_config.dt)
        if count % 50 == 0:
            print(f"step {count:4d} | pos z = {position[2]:.3f} m | thrust = {thrust_z:.3f} N")
        count += 1
if __name__=="__main__":
    main()
    simulation_app.close()
