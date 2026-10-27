"""Scripted button press for ``G1ButtonPressEnv`` using IK.

Two phases, each with a minimum-jerk time profile:

1. approach: IK solves the arm pose that holds the index fingertip ``standoff``
   in front of the button face, pointing into the board, and the arm moves
   there in joint space (a smooth, collision-free swing; a Cartesian straight
   line from the hanging arm drags the arm through awkward configurations);
2. press: the fingertip moves in a straight line along the board normal to
   ``press_depth`` behind the face. This phase is solved with IK every control
   step at the robot's *current* base pose (the base is free-floating and sways),
   warm-started from the previous solution.

Joint angles go through the env's normal action interface. The hand holds a
pointing gesture throughout.

The controller reads the button pose from the simulator (privileged state); on
the robot it would come from perception.
"""

from __future__ import annotations

import numpy as np

from g1_loco import robot
from g1_loco.envs.button_press import BUTTON_TRAVEL, G1ButtonPressEnv
from g1_loco.ik import SiteIK

# Right arm posture the IK null space pulls toward: upper arm raised forward, elbow bent,
# which is where the arm ends up for buttons in front of the chest.
RIGHT_ARM_REST = np.array([-1.0, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])
IK_LIMITS = {
    # Shoulder roll above this swings the upper arm into the torso for buttons near the centerline.
    "right_shoulder_roll_joint": (-np.inf, -0.15),
    # With joint_group "right_arm_waist" the torso turns toward close, centered buttons (otherwise the
    # wrist runs into the chest); keep its lean small so the free-floating base stays balanced.
    "waist_yaw_joint": (-0.6, 0.6),
    "waist_roll_joint": (-0.1, 0.1),
    "waist_pitch_joint": (-0.1, 0.2),
}

# Pointing gesture: index straight, middle finger curled out of the way, thumb at its stand pose.
POINTING_HAND = {
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 1.5,
    "right_hand_middle_1_joint": 1.7,
}


def min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s**3 * (10 - 15 * s + 6 * s**2)


class ButtonPressIK:
    def __init__(
        self,
        env: G1ButtonPressEnv,
        standoff: float = 0.06,
        press_depth: float = BUTTON_TRAVEL + 0.004,
        approach_time: float = 1.5,
        press_time: float = 0.6,
    ):
        """
        Args:
            env: the unwrapped env, with joint_group "right_arm_waist" (recommended) or "right_arm".
                Its ``action_scale`` must be large enough to reach the IK solutions (about 2.5 rad from
                stand), see ``ik_press.py``.
            standoff: distance in front of the button face where the approach ends (m).
            press_depth: how far behind the resting button face the fingertip is commanded (m).
                Past the full stroke so the position servos still push after tracking error.
        """
        self.env = env
        self.standoff = standoff
        self.press_depth = press_depth
        self.approach_time = approach_time
        self.press_time = press_time

        model = env.model
        controlled = [model.actuator(i).name for i in env.ctrl_ids]
        if controlled == robot.JOINT_GROUPS["right_arm_waist"]:
            ik_joints, rest = robot.WAIST + robot.arm("right"), np.concatenate([np.zeros(3), RIGHT_ARM_REST])
        elif controlled == robot.JOINT_GROUPS["right_arm"]:
            ik_joints, rest = robot.arm("right"), RIGHT_ARM_REST
        else:
            raise ValueError("ButtonPressIK expects an env with joint_group 'right_arm_waist' or 'right_arm'")
        limits = {n: lim for n, lim in IK_LIMITS.items() if n in ik_joints}
        self.ik = SiteIK(model, env.fingertip, ik_joints, rest=rest, limits=limits)
        self._ik_slots = np.array([controlled.index(n) for n in ik_joints])
        self._hand_slots = np.array([controlled.index(n) for n in POINTING_HAND])
        self._hand_targets = np.array(list(POINTING_HAND.values()))
        self.reset()

    def reset(self) -> None:
        """Plans the motion from the current state. Call after ``env.reset()``."""
        env, data = self.env, self.env.data
        self.t = 0.0
        board = env.model.body("button_board").id
        self.normal = data.xmat[board].reshape(3, 3)[:, 0].copy()  # into the board
        self.face = data.site_xpos[env.model.site("button_face").id].copy()
        self.pre_press = self.face - self.standoff * self.normal
        self.press = self.face + self.press_depth * self.normal

        self.q_start = data.qpos[self.ik.qpos_ids].copy()
        # Seeding from the rest posture (not the hanging arm) picks the natural elbow-down branch.
        self.approach_result = self.ik.solve(data.qpos, self.pre_press, self.normal, q_init=self.ik.rest)
        self.q_pre = self.approach_result.q
        self._q = self.q_pre.copy()
        self.last_result = self.approach_result

    def tip_target(self, t: float) -> np.ndarray:
        """Fingertip position on the press line at time t (the approach phase is in joint space)."""
        s = min_jerk((t - self.approach_time) / self.press_time)
        return (1 - s) * self.pre_press + s * self.press

    def act(self) -> np.ndarray:
        """Next action for ``env.step``."""
        self.t += self.env.dt
        if self.t < self.approach_time:
            s = min_jerk(self.t / self.approach_time)
            q = (1 - s) * self.q_start + s * self.q_pre
        else:
            self.last_result = self.ik.solve(self.env.data.qpos, self.tip_target(self.t), self.normal, q_init=self._q)
            q = self._q = self.last_result.q

        joint_targets = self.env.default_ctrl[self.env.ctrl_ids].copy()
        joint_targets[self._ik_slots] = q
        joint_targets[self._hand_slots] = self._hand_targets
        return self.env.action_for(joint_targets)
