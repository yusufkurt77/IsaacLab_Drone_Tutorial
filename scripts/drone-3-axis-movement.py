import argparse
from isaaclab.app import AppLauncher

#Standard IsaacLab arguments
parser = argparse.ArgumentParser(description = "Moving the Drone in 3 Axis")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_assets import CRAZYFLIE_CFG

# Gravitational Constant
g = 9.81

def create_scene():

    # The Ground Plane
    ground_config = sim_utils.GroundPlaneCfg()
    ground_config.func("/World/ground", ground_config)

    # The Light
    light_config = sim_utils.DomeLightCfg(intensity = 3000.0)
    light_config.func("/World/Light", light_config)

def drone_object():

    # The Drone
    drone_config = CRAZYFLIE_CFG.replace(prim_path = "/World/Crazyflie")
    drone_config.init_state.pos = (0.0, 0.0, 0.5)
    drone = Articulation(drone_config)
    return drone

def main():

    # The Simulation
    simulation_config = SimulationCfg(dt = 0.01)
    simulation = SimulationContext(simulation_config)
    simulation.set_camera_view(eye = [1.2, 1.2, 1.05], target = [0.5, 0.5, 0.85])

    create_scene()
    drone = drone_object()
    simulation.reset()
    print("Simulation Started")

    body_id = drone.find_bodies("body")[0]
    robot_mass = drone.root_physx_view.get_masses().sum().item()
    robot_weight = robot_mass * g

    inertia = drone.root_physx_view.get_inertias()[0, body_id[0]].reshape(3, 3)
    I_roll = inertia[0, 0].item()
    I_pitch = inertia[1, 1].item()

    prop_joint_ids, prop_joint_names = drone.find_joints(".*rotor.*|.*prop.*|m[1-4]_joint")
    print(f"Found propeller joints: {prop_joint_names}")

    forces = torch.zeros(1, 1, 3, device = drone.device)
    torques = torch.zeros(1, 1, 3, device = drone.device)

    target = torch.tensor([1.0, 1.0, 1.0], device = drone.device)

    kp_xy, kd_xy = 1.5, 1.5
    kp_z, kd_z = 6.0, 3.0

    kp_att, kd_att = 400.0, 40.0

    max_torque = 0.01

    max_tilt = 0.3

    base_spin = 60.0
    spin_per_newton = 40.0

    count = 0
    while simulation_app.is_running():

        # Observation
        position = drone.data.root_pos_w[0]
        lin_vel = drone.data.root_lin_vel_w[0]
        quat = drone.data.root_quat_w[0] # (w, x, y, z)
        ang_vel = drone.data.root_ang_vel_b[0] # roll_rate, pitch_rate, yaw_rate
        
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat.unsqueeze(0))
        roll, pitch = roll.item(), pitch.item()
        roll_rate, pitch_rate = ang_vel[0].item(), ang_vel[1].item()

        # Decision 
        error_x = target[0] - position[0]
        error_y = target[1] - position[1]

        accel_x_des = kp_xy * error_x -kd_xy * lin_vel[0]
        accel_y_des = kp_xy * error_y -kd_xy * lin_vel[1]

        desired_pitch = torch.clamp(torch.tensor(accel_x_des / g), -max_tilt, max_tilt).item()
        desired_roll = torch.clamp(torch.tensor(-accel_y_des / g), -max_tilt, max_tilt).item()

        error_z = target[2] - position[2]
        thrust_z = robot_weight + robot_mass * (kp_z * error_z - kd_z * lin_vel[2])
        thrust_z = max(thrust_z, 0.0)        

        pitch_accel_des = kp_att * (desired_pitch - pitch) - kd_att * pitch_rate
        roll_accel_des = kp_att * (desired_roll - roll) - kd_att * roll_rate
        torque_pitch = I_pitch * pitch_accel_des
        torque_roll = I_roll * roll_accel_des

        torque_pitch = max(min(torque_pitch, max_torque), -max_torque)
        torque_roll = max(min(torque_roll, max_torque), -max_torque)

        # Action
        forces[0, 0, 2] = thrust_z
        torques[0, 0, 0] = torque_roll
        torques[0, 0, 1] = torque_pitch

        drone.set_external_force_and_torque(forces, torques, body_ids = body_id)

        # Cosmetic
        if prop_joint_ids:
            spin_speed = base_spin + spin_per_newton * thrust_z
            prop_vel = torch.full((1, len(prop_joint_ids)), spin_speed, device=drone.device)
            drone.set_joint_velocity_target(prop_vel, joint_ids=prop_joint_ids)
        
        drone.write_data_to_sim()
        simulation.step()
        drone.update(simulation_config.dt)

        if count % 50 == 0:
            print(f"step {count:4d} | pos = ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) "
                f"| roll={roll:+.3f} pitch={pitch:+.3f} | thrust={thrust_z:.3f} N")
        
        count += 1
    
if __name__ == "__main__":
    main()
    simulation_app.close()