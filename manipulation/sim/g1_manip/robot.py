"""Unitree G1 model loading and joint metadata.

The G1 MJCF comes from MuJoCo Menagerie (``scripts/fetch_g1.sh``). Actuators in
that model are position servos named after the joint they drive, in the same
order as the joints, so actuator ``i`` drives hinge joint ``i``.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(os.environ.get("G1_MODEL_DIR", Path(__file__).parent / "assets" / "unitree_g1"))
G1_XML = ASSET_DIR / "g1_with_hands.xml"

FREE_JOINT = "floating_base_joint"
STAND_KEY = "stand"
PELVIS_HEIGHT = 0.80

LEGS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
]
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


def arm(side: str) -> list[str]:
    return [
        f"{side}_{j}_joint"
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]


def hand(side: str) -> list[str]:
    return [
        f"{side}_hand_{j}_joint"
        for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
    ]


# Named sets of joints the policy can control. Everything else is held at the
# standing pose by its position servo.
JOINT_GROUPS: dict[str, list[str]] = {
    "right_arm": arm("right") + hand("right"),
    "right_arm_waist": WAIST + arm("right") + hand("right"),
    "both_arms": arm("left") + hand("left") + arm("right") + hand("right"),
    "upper_body": WAIST + arm("left") + hand("left") + arm("right") + hand("right"),
    "whole_body": LEGS + WAIST + arm("left") + hand("left") + arm("right") + hand("right"),
}

# End-effector sites added to the model (body, local position).
EE_SITES = {
    "right_index_tip": ("right_hand_index_1_link", (0.045, 0.0, 0.0)),
    "left_index_tip": ("left_hand_index_1_link", (0.045, 0.0, 0.0)),
    "right_palm": ("right_wrist_yaw_link", (0.10, -0.01, 0.0)),
    "left_palm": ("left_wrist_yaw_link", (0.10, 0.01, 0.0)),
}


def load_g1_spec(fixed_base: bool) -> tuple[mujoco.MjSpec, np.ndarray]:
    """Loads the G1 as an editable MjSpec and returns it with the standing joint pose.

    With ``fixed_base`` the floating joint is removed so the pelvis is welded in
    place at standing height, which is the usual starting point for training
    upper-body manipulation without needing a balance controller.
    """
    if not G1_XML.exists():
        raise FileNotFoundError(f"G1 model not found at {G1_XML}. Run scripts/fetch_g1.sh first.")

    spec = mujoco.MjSpec.from_file(str(G1_XML))
    # Keyframe qpos = [free joint (7), hinge joints (nu)].
    stand_qpos = np.array(spec.key(STAND_KEY).qpos)[7:]
    spec.delete(spec.key(STAND_KEY))

    pelvis = spec.body("pelvis")
    pelvis.pos = [0.0, 0.0, PELVIS_HEIGHT]
    if fixed_base:
        spec.delete(spec.joint(FREE_JOINT))

    for name, (body, pos) in EE_SITES.items():
        spec.body(body).add_site(name=name, pos=pos, size=[0.008, 0, 0], rgba=[1, 0, 0, 1], group=5)

    return spec, stand_qpos


def add_ground_and_lights(spec: mujoco.MjSpec, base: tuple[float, float, float, float] = (0, 0, 0, 0)) -> None:
    """Adds a ground plane, a light and an over-the-shoulder camera for a robot at base=(x, y, floor_z, yaw)."""
    x, y, z, yaw = base
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.25, 0.3, 0.35, 1],
    )
    world.add_light(pos=[x, y, z + 3], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7])
    cam_pos = np.array([x, y, z]) + rot @ np.array([-0.6, -1.3, 1.6])
    xyaxes = np.concatenate([rot @ np.array([0.788, -0.616, 0]), rot @ np.array([0.251, 0.32, 0.914])])
    world.add_camera(name="front", pos=cam_pos, xyaxes=xyaxes)
