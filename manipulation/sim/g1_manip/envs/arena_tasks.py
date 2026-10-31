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


class G1ArenaBulbEnv(G1ManipEnv):
    """Tighten one of the station's light bulbs, unscrewed by a random number of turns at reset."""

    def __init__(
        self,
        station: str = "bulb_wall",
        hand_site: str = "right_palm",
        loosen: tuple[float, float] = (0.5, 1.5),
        tight_turns: float = 0.05,
        **kwargs,
    ):
        """
        Args:
            loosen: range of turns the target bulb is unscrewed by.
            tight_turns: success once fewer turns than this remain.
        """
        self.hand_site = hand_site
        self.loosen = loosen
        self.tight_turns = tight_turns
        super().__init__(station=station, **kwargs)

    def _reset_task(self) -> None:
        self.bulbs = list(self.station.fixtures)
        self.target = str(self.np_random.choice(self.bulbs))
        self.start_turns = float(self.np_random.uniform(*self.loosen))
        self.mechanisms.unscrew_bulb(self.data, self.target, self.start_turns)
        self._tip = self.model.site(f"{self.target}_tip").id

    def turns_left(self) -> float:
        return self.mechanisms.bulb_turns(self.data, self.target)

    def _task_obs(self) -> np.ndarray:
        hand = self.site_pos(self.hand_site)
        tip = self.data.site_xpos[self._tip]
        return np.concatenate(
            [
                self.to_pelvis_frame(hand - self.data.xpos[self.pelvis_id]),
                self.to_pelvis_frame(tip - hand),
                [self.turns_left() / self.start_turns],
            ]
        )

    def _task_reward(self, action):
        dist = float(np.linalg.norm(self.data.site_xpos[self._tip] - self.site_pos(self.hand_site)))
        turns = self.turns_left()
        success = turns < self.tight_turns
        reward = 1.0 - np.tanh(5.0 * dist) + 3.0 * (1.0 - turns / self.start_turns) + (10.0 if success else 0.0)
        return reward, success, {"distance": dist, "turns_left": turns, "target": self.target, "is_success": success}


class G1ArenaDoorEnv(G1ManipEnv):
    """Turn the lever handle, push the door open, then let the springs close it again.

    Success once the door has been opened to ``open_angle`` and is closed and latched again.
    Walking through the open door is left to the locomotion controller.
    """

    def __init__(self, station: str = "door", hand_site: str = "right_index_tip", open_angle: float = 0.2, **kwargs):
        self.hand_site = hand_site
        self.open_angle = open_angle
        super().__init__(station=station, **kwargs)

    def _reset_task(self) -> None:
        self._grip = self.model.site("door_handle_start_grip").id
        self._handle_q = self.model.jnt_qposadr[self.model.joint("door_handle_start_joint").id]
        self.max_angle = 0.0

    def handle_angle(self) -> float:
        return float(self.data.qpos[self._handle_q])

    def _task_obs(self) -> np.ndarray:
        hand = self.site_pos(self.hand_site)
        grip = self.data.site_xpos[self._grip]
        return np.concatenate(
            [
                self.to_pelvis_frame(hand - self.data.xpos[self.pelvis_id]),
                self.to_pelvis_frame(grip - hand),
                [self.handle_angle(), self.mechanisms.door_angle(self.data), float(self.mechanisms.door_latched(self.data))],
            ]
        )

    def _task_reward(self, action):
        dist = float(np.linalg.norm(self.data.site_xpos[self._grip] - self.site_pos(self.hand_site)))
        angle = self.mechanisms.door_angle(self.data)
        self.max_angle = max(self.max_angle, angle)
        opened = self.max_angle >= self.open_angle
        success = opened and self.mechanisms.door_latched(self.data)
        reward = 1.0 - np.tanh(5.0 * dist) + self.handle_angle() + 5.0 * min(self.max_angle, self.open_angle)
        reward += 10.0 if success else 0.0
        info = {"distance": dist, "door_angle": angle, "opened": opened, "is_success": success}
        return reward, success, info
