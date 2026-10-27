"""Button press task (competition task 5, and the keypad's building block).

A spring-loaded push button sits on a board in front of the robot at a random
position. The right index fingertip must push it in past a threshold.
"""

from __future__ import annotations

import mujoco
import numpy as np

from g1_loco.envs.base import G1Env

BUTTON_TRAVEL = 0.012  # m, full stroke
PRESS_DEPTH = 0.008  # m, counts as pressed
BOARD_THICKNESS = 0.02


class G1ButtonPressEnv(G1Env):
    def __init__(
        self,
        target_range: tuple[tuple[float, float], ...] = ((0.28, 0.38), (-0.35, -0.05), (0.80, 1.10)),
        fingertip: str = "right_index_tip",
        **kwargs,
    ):
        # (x, y, z) ranges for the button face center in the world frame. The
        # robot stands at the origin facing +x; negative y is its right side.
        self.target_range = np.array(target_range)
        self.fingertip = fingertip
        super().__init__(**kwargs)

    def _build_task(self, spec: mujoco.MjSpec) -> None:
        board = spec.worldbody.add_body(name="button_board", pos=[0.45, -0.2, 1.0])
        board.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[BOARD_THICKNESS / 2, 0.12, 0.12],
            pos=[BOARD_THICKNESS / 2, 0, 0],
            rgba=[0.8, 0.65, 0.45, 1],
        )
        button = board.add_body(name="button", pos=[0, 0, 0])
        button.add_joint(
            name="button_joint",
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[1, 0, 0],
            range=[0, BUTTON_TRAVEL],
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=300.0,
            damping=2.0,
            # Direct (stiffness, damping) end-stop so the light button can't be driven past its travel.
            solref_limit=[-20000.0, -50.0],
        )
        button.add_geom(
            name="button_cap",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.03, 0.008, 0],
            pos=[-0.008, 0, 0],
            quat=[0.7071068, 0, 0.7071068, 0],
            rgba=[0.95, 0.85, 0.1, 1],
            mass=0.02,
        )
        button.add_site(name="button_face", pos=[-0.016, 0, 0], size=[0.005, 0, 0])
        # The board is welded to the world, so parent-child filtering doesn't apply.
        spec.add_exclude(bodyname1="button_board", bodyname2="button")

    def _reset_task(self) -> None:
        # The board is a static body; moving its model position moves the button.
        board_id = self.model.body("button_board").id
        lo, hi = self.target_range.T
        self.model.body_pos[board_id] = self.np_random.uniform(lo, hi)
        self._button_qpos = self.model.jnt_qposadr[self.model.joint("button_joint").id]
        self._button_site = self.model.site("button_face").id

    def button_depth(self) -> float:
        return float(self.data.qpos[self._button_qpos])

    def _task_obs(self) -> np.ndarray:
        tip = self.site_pos(self.fingertip)
        target = self.data.site_xpos[self._button_site]
        return np.concatenate(
            [
                self.to_pelvis_frame(tip - self.data.xpos[self.pelvis_id]),
                self.to_pelvis_frame(target - tip),
                [self.button_depth() / BUTTON_TRAVEL],
            ]
        )

    def _task_reward(self, action):
        dist = float(np.linalg.norm(self.data.site_xpos[self._button_site] - self.site_pos(self.fingertip)))
        depth = self.button_depth()
        pressed = depth >= PRESS_DEPTH

        reward = 1.0 - np.tanh(5.0 * dist)
        reward += 2.0 * depth / BUTTON_TRAVEL
        if pressed:
            reward += 10.0
        return reward, pressed, {"distance": dist, "button_depth": depth, "is_success": pressed}
