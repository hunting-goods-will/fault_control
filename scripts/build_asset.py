"""Author the single-joint actuator asset as a standalone USD file.

This is a one-off script, run once to generate assets/single_joint_actuator.usd.
It does NOT launch the Isaac Sim app -- it uses the pxr USD Physics API directly
to author a stage and export it, which is faster and doesn't need a GPU context.

Model (documented invented parameters -- see README):
    - Base: fixed link, pinned to the world via a FixedJoint carrying the
      ArticulationRootAPI (standard fixed-base articulation pattern).
    - Arm: a single slender capsule, length 0.3 m, radius 0.02 m, mass 0.5 kg,
      hanging straight down from the pivot under gravity (joint_pos = 0 is the
      arm's stable equilibrium with no torque applied).
    - Joint: revolute, Y-axis (swings in the X-Z plane, the plane containing
      gravity), connecting Base -> Arm.

No actuator drive (stiffness/damping/effort limits) is authored here on purpose.
All of that is configured in Python via ImplicitActuatorCfg in run_trials.py, so
there's exactly one place the actuation numbers live, not two.

Mass is specified directly (not density); the physics engine derives the
inertia tensor from the mass + collision geometry rather than us hand-deriving
an inertia matrix -- simpler and just as physically valid for this purpose.

This launches the Isaac Sim app headlessly -- in a pip-installed Isaac Sim
environment, `pxr` is only importable after the app has started, since that's
what wires the compiled Kit extensions onto the Python path. No physics
stepping happens; it just authors the stage, exports it, and closes.

Usage:
    ./isaaclab.sh -p scripts/build_asset.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Author the single-joint actuator USD asset.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # this script only writes a file, never needs a window

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows -- pxr is only importable after the app has started."""

import os

from pxr import Usd, UsdGeom, UsdPhysics, Gf

ARM_LENGTH = 0.3  # m
ARM_RADIUS = 0.02  # m
ARM_MASS = 0.5  # kg
BASE_MASS = 1.0  # kg -- fixed body, magnitude doesn't matter dynamically

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "single_joint_actuator.usda")


def build() -> None:
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)  # Usd.Stage.CreateNew refuses to overwrite an existing layer

    stage = Usd.Stage.CreateNew(OUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    # --- Base link: sits at the origin, pinned to the world ---
    base_path = "/World/Base"
    base_xform = UsdGeom.Xform.Define(stage, base_path)
    base_geom = UsdGeom.Cylinder.Define(stage, base_path + "/geom")
    base_geom.CreateRadiusAttr(0.03)
    base_geom.CreateHeightAttr(0.05)
    base_geom.CreateAxisAttr("Y")
    UsdPhysics.CollisionAPI.Apply(base_geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(base_xform.GetPrim())
    UsdPhysics.MassAPI.Apply(base_xform.GetPrim()).CreateMassAttr(BASE_MASS)

    # --- Arm link: hangs straight down from the pivot at the origin ---
    arm_path = "/World/Arm"
    arm_xform = UsdGeom.Xform.Define(stage, arm_path)
    arm_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -ARM_LENGTH / 2.0))
    arm_geom = UsdGeom.Capsule.Define(stage, arm_path + "/geom")
    arm_geom.CreateRadiusAttr(ARM_RADIUS)
    arm_geom.CreateHeightAttr(ARM_LENGTH)
    arm_geom.CreateAxisAttr("Z")
    UsdPhysics.CollisionAPI.Apply(arm_geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(arm_xform.GetPrim())
    UsdPhysics.MassAPI.Apply(arm_xform.GetPrim()).CreateMassAttr(ARM_MASS)

    # --- Fixed joint anchoring the base to the world; marks the articulation root ---
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/BaseFixed")
    fixed_joint.CreateBody1Rel().SetTargets([base_path])
    UsdPhysics.ArticulationRootAPI.Apply(fixed_joint.GetPrim())

    # --- Revolute joint: Base -> Arm, Y axis, located at the pivot (world origin) ---
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/PivotJoint")
    joint.CreateAxisAttr("Y")
    joint.CreateBody0Rel().SetTargets([base_path])
    joint.CreateBody1Rel().SetTargets([arm_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, ARM_LENGTH / 2.0))

    # Placeholder drive so PhysX/Isaac Lab recognizes this joint as actuated at
    # all -- an ImplicitActuatorCfg at spawn time appears to *configure* an
    # existing drive rather than create one from scratch. Values here don't
    # matter; run_trials.py's ImplicitActuatorCfg overrides them.
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)
    drive.CreateMaxForceAttr(1.0e6)

    stage.GetRootLayer().Save()
    print(f"[INFO] Wrote asset to {OUT_PATH}")


if __name__ == "__main__":
    build()
    simulation_app.close()
