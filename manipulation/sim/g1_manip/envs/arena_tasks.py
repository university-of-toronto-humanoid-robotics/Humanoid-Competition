"""Tasks on the real competition fixtures (arena built from the official CAD).

The robot is placed at a station in front of the fixtures (see arena.STATION_SPECS).
Fixture state comes from ArenaMechanisms, so pressed buttons light up and the
door latch works the same way in every task.
"""

from __future__ import annotations

import numpy as np

from g1_manip.envs.base import G1ManipEnv


class G1ArenaButtonsEnv(G1ManipEnv):
    """Press a random sequence of the station's buttons in order.

    With station="keypad" and sequence_length=5 this is the keypad task (the code
    is given to the policy as the current target; the CAD's display shows a fixed
    code). With station="button_wall" and sequence_length=1 it is task 5.
    """

    def __init__(
        self,
        station: str = "button_wall",
        sequence_length: int = 1,
        fingertip: str = "right_index_tip",
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.fingertip = fingertip
        super().__init__(station=station, **kwargs)

    def _reset_task(self) -> None:
        self.buttons = list(self.station.fixtures)
        self._sites = {b: self.model.site(f"{b}_site").id for b in self.buttons}
        replace = self.sequence_length > len(self.buttons)
        self.sequence = list(self.np_random.choice(self.buttons, self.sequence_length, replace=replace))
        self.progress = 0
        self._was_pressed = {b: False for b in self.buttons}

    @property
    def target(self) -> str:
        return self.sequence[min(self.progress, len(self.sequence) - 1)]

    def _task_obs(self) -> np.ndarray:
        tip = self.site_pos(self.fingertip)
        target = self.data.site_xpos[self._sites[self.target]]
        return np.concatenate(
            [
                self.to_pelvis_frame(tip - self.data.xpos[self.pelvis_id]),
                self.to_pelvis_frame(target - tip),
                [self.mechanisms.button_depth(self.data, self.target), self.progress / self.sequence_length],
            ]
        )

    def _task_reward(self, action):
        target = self.target
        dist = float(np.linalg.norm(self.data.site_xpos[self._sites[target]] - self.site_pos(self.fingertip)))
        reward = 1.0 - np.tanh(5.0 * dist) + 2.0 * min(self.mechanisms.button_depth(self.data, target), 1.0)

        for b in self.buttons:
            pressed = self.mechanisms.is_pressed(self.data, b)
            if pressed and not self._was_pressed[b]:
                if b == target:
                    reward += 5.0
                    self.progress += 1
                else:
                    reward -= 1.0
            self._was_pressed[b] = pressed

        success = self.progress >= self.sequence_length
        if success:
            reward += 10.0
        info = {"distance": dist, "progress": self.progress, "target": target, "is_success": success}
        return reward, success, info


class G1ArenaValveEnv(G1ManipEnv):
    """Turn one of the station's ball valves from perpendicular to parallel (90 deg)."""

    def __init__(self, station: str = "valve_wall", hand_site: str = "right_palm", success_fraction: float = 0.9, **kwargs):
        self.hand_site = hand_site
        self.success_fraction = success_fraction
        super().__init__(station=station, **kwargs)

    def _reset_task(self) -> None:
        self.valves = list(self.station.fixtures)
        self.target = str(self.np_random.choice(self.valves))
        self._grip = self.model.site(f"{self.target}_grip").id
        self._range = float(self.model.jnt_range[self.model.joint(f"{self.target}_joint").id, 1])

    def _turned(self) -> float:
        return abs(self.mechanisms.valve_angle(self.data, self.target)) / self._range

    def _task_obs(self) -> np.ndarray:
        hand = self.site_pos(self.hand_site)
        grip = self.data.site_xpos[self._grip]
        angle = self.mechanisms.valve_angle(self.data, self.target)
        return np.concatenate(
            [
                self.to_pelvis_frame(hand - self.data.xpos[self.pelvis_id]),
                self.to_pelvis_frame(grip - hand),
                [np.sin(angle), np.cos(angle)],
            ]
        )

    def _task_reward(self, action):
        dist = float(np.linalg.norm(self.data.site_xpos[self._grip] - self.site_pos(self.hand_site)))
        turned = self._turned()
        success = turned >= self.success_fraction
        reward = 1.0 - np.tanh(5.0 * dist) + 3.0 * turned + (10.0 if success else 0.0)
        return reward, success, {"distance": dist, "turned": turned, "target": self.target, "is_success": success}
