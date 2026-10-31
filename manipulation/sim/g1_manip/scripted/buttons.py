"""Keypad (task 1) and button board (task 5): press buttons in order with the index fingertip.

Action sequence, for ``G1ArenaButtonsEnv`` (keypad, button_wall, button_floor):

1. pointing gesture: index straight, middle finger curled away;
2. reach the first button's pre-press point, ``standoff`` in front of the cap, finger along the
   button axis: an arm swing at the wall, a balanced crouch at floor level;
3. per button: press in a straight line along the axis toward ``press_depth`` behind the cap face
   (past the 4 mm that light it, just short of the 6 mm stop: the robot is free-floating and
   pushing on the stop with stiff arm servos shoves it over). The press stops as soon as the
   button lights, and it is force-guarded: above ``max_force`` the finger backs off instead of
   pushing harder. Then pull back to the pre-press point. If the button did not light, press
   again 1 mm deeper;
4. hop to the next button: back off along the axis, swing to the next button's canonical arm
   pose, come in along its axis;
5. back to standing.

Joint limits that matter: the right shoulder roll is kept below -0.15 rad so the upper arm stays
off the torso for buttons near the centerline. The waist may turn ±0.6 rad but lean only slightly.
The 6 mm button travel and 4 mm light threshold set the press depth window.
"""

from __future__ import annotations

import numpy as np

from g1_manip.scripted.base import POINTING_HAND, ScriptedIK


class ButtonSequenceIK(ScriptedIK):
    def __init__(
        self,
        env,
        standoff: float = 0.05,
        press_depth: float = 0.0055,
        contact_speed: float = 0.03,
        transfer_speed: float = 0.25,
        retries: int = 2,
        max_force: float = 15.0,
    ):
        """
        Args:
            env: unwrapped ``G1ArenaButtonsEnv``.
            standoff: distance in front of the cap face where presses start and end (m).
            press_depth: fingertip target behind the resting cap face (m). Between the light threshold
                (4 mm) and the stop (6 mm).
            contact_speed: fingertip speed from 1 cm before the cap to the press depth (m/s). Slow,
                so the finger doesn't hit the cap with the arm's momentum.
            transfer_speed: average fingertip speed between buttons (m/s).
            retries: extra presses per button if it doesn't light.
            max_force: contact force limit while pressing (N). The spring needs 2.4 N to light the
                button and 3.6 N at the stop; the margin covers the finger's servo tracking error.
        """
        self.max_force = max_force
        self.standoff = standoff
        self.press_depth = press_depth
        self.contact_speed = contact_speed
        self.transfer_speed = transfer_speed
        self.retries = retries
        super().__init__(env, site=env.fingertip)

    def button_frame(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(cap face center, axis pointing into the board) of a button at rest."""
        m, d = self.env.model, self.env.data
        face = d.site_xpos[m.site(f"{name}_site").id].copy()
        face -= d.qpos[m.jnt_qposadr[m.joint(f"{name}_joint").id]] * d.xmat[m.body(name).id].reshape(3, 3)[:, 2]
        return face, d.xmat[m.body(name).id].reshape(3, 3)[:, 2].copy()

    def _script(self):
        env = self.env
        self.phase = "approach"
        yield from self.move_joints(POINTING_HAND, 0.3)
        prev = None
        for name in env.sequence:
            face, axis = self.button_frame(name)
            pre = face - self.standoff * axis
            if prev is None:
                yield from self.reach(pre, axis)
            elif np.linalg.norm(pre - prev[0]) > 1e-6:
                self.phase = f"transfer to {name}"
                duration = max(0.6, np.linalg.norm(pre - prev[0]) / self.transfer_speed)
                yield from self.hop(*prev, pre, axis, duration)
            prev = (pre, axis)

            progress = env.progress
            for attempt in range(1 + self.retries):
                self.phase = f"press {name}" + (f" (retry {attempt})" if attempt else "")
                press = face + (self.press_depth + 0.001 * attempt) * axis
                near = face - 0.01 * axis
                lit = lambda: env.progress > progress  # noqa: E731  (on the robot: the light, seen by a camera)
                yield from self.line(pre, near, axis, 0.35)
                duration = 2 * np.linalg.norm(press - near) / self.contact_speed  # min-jerk: 2x the average
                yield from self.line(near, press, axis, duration, max_force=self.max_force, until=lit)
                yield from self.line(self.ik().site_pose(self.env.data.qpos)[0], pre, axis, 0.4)
                if lit():
                    break
        self.phase = "retract"
        yield from self.retract()
