"""Run the single-joint actuator fault-response trials.

Day 2 scope: baseline (uncontrolled) mode only. Rate-limited / sensor-validated
mode is added in Day 3 as a follow-up commit to this same file.

Usage:
    ./isaaclab.sh -p scripts/run_trials.py --mode baseline --num_envs 64
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-joint actuator fault-response trials.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel randomized trials.")
parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "rate_limited"])
parser.add_argument("--duration", type=float, default=5.0, help="Sim seconds per trial.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- isaaclab modules are only importable after the app starts."""

import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

ASSET_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "single_joint_actuator.usda")

# Fault torque range (N*m) -- see README for reasoning.
# Max gravity restoring torque on this arm is m*g*(L/2) = 0.5*9.81*0.15 ~ 0.74 N*m.
# Keeping the fault range comfortably under that so a real equilibrium exists
# for every sampled trial (equilibrium angle = arcsin(tau / 0.74)):
#   0.2 N*m -> ~16 deg,  0.6 N*m -> ~55 deg
FAULT_TORQUE_MIN = 0.2
FAULT_TORQUE_MAX = 0.6

# Actuator config -- the single source of truth for stiffness/damping/limits.
# stiffness=0.0 -> pure torque/effort control, no implicit position drive.
JOINT_DAMPING = 0.05  # N*m*s/rad, small bearing friction

SINGLE_JOINT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ASSET_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            sleep_threshold=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.0,
            stabilization_threshold=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # clear of the ground -- arm hangs 0.3 m below the pivot
        joint_pos={"PivotJoint": 0.0},
    ),
    actuators={
        "pivot": ImplicitActuatorCfg(
            joint_names_expr=["PivotJoint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=20.0,
            stiffness=0.0,
            damping=JOINT_DAMPING,
        ),
    },
)


@configclass
class SingleJointSceneCfg(InteractiveSceneCfg):
    """A ground plane, a light, and n copies of the single-joint arm."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )
    robot: ArticulationCfg = SINGLE_JOINT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_baseline(sim: SimulationContext, scene: InteractiveScene, fault_torque: torch.Tensor, num_steps: int, sim_dt: float):
    """No rate limiting, no sensor check -- the fault torque is commanded as an instant step."""
    robot = scene["robot"]
    num_envs = scene.num_envs

    joint_pos_log = torch.zeros(num_envs, num_steps)
    joint_vel_log = torch.zeros(num_envs, num_steps)
    torque_log = torch.zeros(num_envs, num_steps)

    # reset to the hanging-down equilibrium
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    scene.reset()

    commanded = fault_torque.to(robot.data.joint_pos.device).unsqueeze(-1)

    for step in range(num_steps):
        robot.set_joint_effort_target(commanded)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        joint_pos_log[:, step] = robot.data.joint_pos[:, 0].cpu()
        joint_vel_log[:, step] = robot.data.joint_vel[:, 0].cpu()
        torque_log[:, step] = commanded[:, 0].cpu()

    return joint_pos_log, joint_vel_log, torque_log


def main():
    torch.manual_seed(args_cli.seed)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.35])

    scene_cfg = SingleJointSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    fault_torque = torch.empty(args_cli.num_envs).uniform_(FAULT_TORQUE_MIN, FAULT_TORQUE_MAX)

    sim_dt = sim.get_physics_dt()
    num_steps = int(args_cli.duration / sim_dt)

    if args_cli.mode == "baseline":
        joint_pos, joint_vel, torque = run_baseline(sim, scene, fault_torque, num_steps, sim_dt)
    else:
        raise NotImplementedError("rate_limited mode lands in Day 3.")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args_cli.mode}_trials.npz")
    np.savez(
        out_path,
        fault_torque=fault_torque.numpy(),
        joint_pos=joint_pos.numpy(),
        joint_vel=joint_vel.numpy(),
        commanded_torque=torque.numpy(),
        sim_dt=sim_dt,
        seed=args_cli.seed,
    )
    print(f"[INFO]: Saved {joint_pos.shape[0]} trials x {joint_pos.shape[1]} steps to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()