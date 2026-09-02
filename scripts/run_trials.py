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

# Day 3 rate limiter -- derived, not measured, from the Day 2 physics:
#   Thin-rod inertia about the pivot: I = (1/3) m L^2 = (1/3)(0.5)(0.3^2) = 0.015 kg*m^2
#   Linearized natural frequency near equilibrium: w_n = sqrt(m*g*(L/2)*cos(theta_eq) / I)
#     -> w_n ~= 5.3-7.0 rad/s across the theta_eq ~= 16-55 deg range -> period ~= 0.9-1.2 s
#   Damping ratio zeta = JOINT_DAMPING / (2*I*w_n) ~= 0.24-0.32 -- underdamped, multiple
#     visible oscillations before settling, consistent with the 5.0s default --duration.
# MAX_TORQUE_RATE is chosen so ramping to FAULT_TORQUE_MAX takes ~0.3s, i.e. a fraction of
# one natural period -- fast enough to still be a real step-like event, slow enough to
# meaningfully change the transient relative to baseline's single-step command.
# This is a stated engineering choice, not a measured constant -- worth revisiting against
# the actual settling-time numbers once Day 4 stats are in.
MAX_TORQUE_RATE = 2.0  # N*m/s

# Sensor validation threshold. Because stiffness=0 and joint_vel_target defaults to 0,
# ImplicitActuator's applied-torque estimate reduces to
#     applied_torque ~= commanded_torque - JOINT_DAMPING * joint_vel
# (see isaaclab.actuators.actuator_pd.ImplicitActuator.compute) -- it is NOT an
# independent PhysX torque sensor, it's a software estimate using the real joint_vel
# readback. So "divergence" here tracks how fast the arm is actually swinging, not an
# injected fault. At a plausible peak swing speed of a few rad/s, JOINT_DAMPING * joint_vel
# is on the order of 0.1-0.2 N*m -- this threshold is set to make holds possible during
# fast portions of the swing without being trivially always-triggered. TUNE THIS against
# the real max |joint_vel| in results/raw/baseline_trials.npz before trusting it; it may
# legitimately turn out to rarely or never fire, which is a fine, honest result to report.
SENSOR_VALIDATION_THRESHOLD = 0.1  # N*m

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


def run_rate_limited(sim: SimulationContext, scene: InteractiveScene, fault_torque: torch.Tensor, num_steps: int, sim_dt: float):
    """Ramp the commanded torque toward fault_torque at MAX_TORQUE_RATE, validating each
    step's ramped command against the actuator's applied_torque estimate. If they diverge
    beyond SENSOR_VALIDATION_THRESHOLD, hold the last accepted value instead of applying
    the new one.

    Uses two write_data_to_sim() calls on a held step: one to get the actuator model's
    applied_torque estimate for the candidate (ramped) command, and -- only if that
    candidate is rejected -- a second to re-write the held value so both the sim buffers
    and the logged applied_torque reflect what was actually applied this step.
    """
    robot = scene["robot"]
    num_envs = scene.num_envs
    device = robot.data.joint_pos.device

    joint_pos_log = torch.zeros(num_envs, num_steps)
    joint_vel_log = torch.zeros(num_envs, num_steps)
    torque_log = torch.zeros(num_envs, num_steps)  # ramped torque actually accepted this step
    applied_torque_log = torch.zeros(num_envs, num_steps)  # actuator's applied_torque estimate
    held_log = torch.zeros(num_envs, num_steps, dtype=torch.bool)  # True where validation held

    # reset to the hanging-down equilibrium
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    scene.reset()

    target = fault_torque.to(device).unsqueeze(-1)
    delta_max = MAX_TORQUE_RATE * sim_dt
    last_accepted = torch.zeros_like(target)  # no fault torque commanded yet at t=0

    for step in range(num_steps):
        # rate-limit the ramp toward the target, clamped to +-delta_max this step
        ramped = last_accepted + torch.clamp(target - last_accepted, -delta_max, delta_max)

        robot.set_joint_effort_target(ramped)
        scene.write_data_to_sim()
        applied = robot.data.applied_torque[:, :1].clone()

        diverged = (ramped - applied).abs() > SENSOR_VALIDATION_THRESHOLD
        if diverged.any():
            accepted = torch.where(diverged, last_accepted, ramped)
            robot.set_joint_effort_target(accepted)
            scene.write_data_to_sim()
            applied = robot.data.applied_torque[:, :1].clone()
        else:
            accepted = ramped

        sim.step()
        scene.update(sim_dt)

        last_accepted = accepted

        joint_pos_log[:, step] = robot.data.joint_pos[:, 0].cpu()
        joint_vel_log[:, step] = robot.data.joint_vel[:, 0].cpu()
        torque_log[:, step] = accepted[:, 0].cpu()
        applied_torque_log[:, step] = applied[:, 0].cpu()
        held_log[:, step] = diverged[:, 0].cpu()

    total_held = held_log.sum().item()
    total_steps = held_log.numel()
    print(f"[INFO]: Sensor validation held {total_held}/{total_steps} step-env pairs ({100 * total_held / total_steps:.3f}%)")

    return joint_pos_log, joint_vel_log, torque_log, applied_torque_log, held_log


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

    extra = {}
    if args_cli.mode == "baseline":
        joint_pos, joint_vel, torque = run_baseline(sim, scene, fault_torque, num_steps, sim_dt)
    else:
        joint_pos, joint_vel, torque, applied_torque, held = run_rate_limited(
            sim, scene, fault_torque, num_steps, sim_dt
        )
        extra = {"applied_torque": applied_torque.numpy(), "held": held.numpy()}

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
        **extra,
    )
    print(f"[INFO]: Saved {joint_pos.shape[0]} trials x {joint_pos.shape[1]} steps to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()