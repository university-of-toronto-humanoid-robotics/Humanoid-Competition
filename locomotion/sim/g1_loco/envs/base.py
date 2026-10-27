"""Gymnasium base environment for Unitree G1 tasks in MuJoCo.

Subclasses add task geometry in ``_build_task`` and implement the task hooks
(``_reset_task``, ``_task_obs``, ``_task_reward``). The base class handles the
robot, the action mapping, proprioceptive observations and rendering.

Same interface as ``g1_manip.envs.base.G1ManipEnv`` (manipulation/sim), on a
flat floor without the competition arena.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from g1_loco import robot


class G1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        joint_group: str = "right_arm",
        fixed_base: bool = False,
        frame_skip: int = 10,
        action_scale: float = 0.75,
        reset_noise: float = 0.05,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (480, 640),
    ):
        """
        Args:
            action_scale: joint targets are ``stand + action_scale * action`` (rad), clipped to the
                actuator ranges.
        """
        self.fixed_base = fixed_base
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.reset_noise = reset_noise
        self.render_mode = render_mode
        self.render_size = render_size

        spec, self._stand_qpos = robot.load_g1_spec(fixed_base)
        self._base_pos = np.array([0.0, 0.0, robot.PELVIS_HEIGHT])
        self._base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        robot.add_ground_and_lights(spec)
        self._build_task(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.metadata["render_fps"] = int(round(1.0 / self.dt))

        names = robot.JOINT_GROUPS[joint_group]
        self.ctrl_ids = np.array([self.model.actuator(n).id for n in names])
        self.qpos_ids = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in names])
        self.qvel_ids = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in names])
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange[self.ctrl_ids].T
        self.default_ctrl = self._stand_qpos.copy()
        self.pelvis_id = self.model.body("pelvis").id
        self.site_ids = {n: self.model.site(n).id for n in robot.EE_SITES}

        n_act = len(names)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self._prev_action = np.zeros(n_act)
        self.reset()
        obs_dim = self._get_obs().shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._viewer = None
        self._renderer = None

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    # ----- task hooks -------------------------------------------------------

    def _build_task(self, spec: mujoco.MjSpec) -> None:
        """Adds task bodies/geoms to the scene before compilation."""

    def _reset_task(self) -> None:
        """Randomizes task state. Called after the robot is reset, before mj_forward."""

    def _task_obs(self) -> np.ndarray:
        return np.zeros(0)

    def _task_reward(self, action: np.ndarray) -> tuple[float, bool, dict[str, Any]]:
        """Returns (reward, terminated, info)."""
        return 0.0, False, {}

    # ----- helpers ----------------------------------------------------------

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.site_ids[name]].copy()

    def to_pelvis_frame(self, vec_world: np.ndarray) -> np.ndarray:
        """Rotates a world-frame vector into the pelvis frame."""
        return self.data.xmat[self.pelvis_id].reshape(3, 3).T @ vec_world

    def pelvis_height(self) -> float:
        return float(self.data.xpos[self.pelvis_id][2])

    def action_for(self, joint_targets: np.ndarray) -> np.ndarray:
        """Inverse of the action mapping: the action that commands ``joint_targets`` for the controlled joints."""
        return np.clip((joint_targets - self.default_ctrl[self.ctrl_ids]) / self.action_scale, -1.0, 1.0)

    # ----- gym API ----------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        hinge_qpos = self._stand_qpos.copy()
        noise = self.np_random.uniform(-self.reset_noise, self.reset_noise, size=len(self.ctrl_ids))
        hinge_qpos[self.ctrl_ids] += noise
        hinge_qpos = np.clip(hinge_qpos, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        offset = 0 if self.fixed_base else 7
        if not self.fixed_base:
            self.data.qpos[:3] = self._base_pos
            self.data.qpos[3:7] = self._base_quat
        self.data.qpos[offset : offset + self.model.nu] = hinge_qpos
        self.data.ctrl[:] = hinge_qpos
        self._prev_action[:] = 0.0

        self._reset_task()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.default_ctrl[self.ctrl_ids] + self.action_scale * action
        self.data.ctrl[self.ctrl_ids] = np.clip(target, self.ctrl_low, self.ctrl_high)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        reward, terminated, info = self._task_reward(action)
        reward -= 0.01 * float(np.sum(np.square(action - self._prev_action)))
        if not self.fixed_base and self.pelvis_height() - self._base_pos[2] < 0.5 - robot.PELVIS_HEIGHT:
            terminated = True
            info["fell"] = True
            reward -= 10.0
        self._prev_action = action

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        parts = [
            self.data.qpos[self.qpos_ids] - self._stand_qpos[self.ctrl_ids],
            0.1 * self.data.qvel[self.qvel_ids],
            self._prev_action,
        ]
        if not self.fixed_base:
            gravity = self.to_pelvis_frame(np.array([0.0, 0.0, -1.0]))
            parts += [gravity, 0.25 * self.data.qvel[3:6]]
        parts.append(self._task_obs())
        return np.concatenate(parts).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                # On macOS the passive viewer requires launching the script with `mjpython`.
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, *self.render_size)
            self._renderer.update_scene(self.data, camera="front")
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
