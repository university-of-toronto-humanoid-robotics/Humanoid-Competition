"""Valve turning (task 4): push a quarter-turn ball valve's lever through 90 degrees.

The lever is pushed rather than grasped: the index finger points into the valve along its hinge
axis, its side against the side of the lever, and the fingertip moves on an arc around the axis.
The lever needs about 9 N at the contact point (0.4 N·m of friction at 4.5 cm), well within the
arm's reach of force, and pushing needs no grasp to fail.

Action sequence, for ``G1ArenaValveEnv`` (valve_wall, valve_floor):

1. pointing gesture;
2. choose the turning direction. The sim valve turns ±90° (both end parallel to the pipe). On the
   apparatus only one direction is free, and that would be fixed per valve when pre-mapping. Here
   the direction is the one whose arc the hand follows without hitting the valve's mounting
   (collision-checked IK solutions along the arc) and that the arm reaches best;
3. reach a point ``hover`` above the start of the arc, behind the lever (on the side it is
   pushed from), coming in along the hinge axis: an arm swing at the wall, a crouch at floor level;
4. lower the fingertip along the axis to lever height;
5. sweep the arc, force-guarded, until the lever is at 90° (it stops at its end stop; the guard
   keeps the finger from shoving against the stop);
6. lift off along the axis and go back to standing.

Joint limits that matter: the arc spans ~2 rad around the valve axis, and for the valves whose axis
points sideways the forearm has to turn with it. The wrist yaw/pitch (kept within ±1.3 rad to stop
the wrist links colliding) and shoulder yaw take that, which is what decides the direction.
"""

from __future__ import annotations

import numpy as np

from g1_manip.scripted.base import POINTING_HAND, ScriptedIK

CONTACT_RADIUS = 0.045  # m from the hinge axis; the lever is ~6 cm long
START_GAP = 0.45  # rad behind the lever where the finger comes down, clear of it
# Fingertip height relative to the lever's middle, along the axis. The lever is only 1.3 cm tall and the
# fingertip rides up on it under load; aimed at the bottom edge, the finger's side covers the whole lever.
CONTACT_DEPTH = -0.008


class ValveTurnIK(ScriptedIK):
    def __init__(self, env, hover: float = 0.06, sweep_speed: float = 0.6, max_force: float = 30.0):
        """
        Args:
            env: unwrapped ``G1ArenaValveEnv``.
            hover: height above the lever (along the axis) where the finger comes down (m).
            sweep_speed: average angular speed of the push (rad/s).
            max_force: contact force limit while pushing (N); the lever needs ~9 N.
        """
        self.hover = hover
        self.sweep_speed = sweep_speed
        self.max_force = max_force
        super().__init__(env, site="right_index_tip")

    def valve_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """(hinge center, axis toward the robot, lever direction at 0 rad, a x u, lever height) of the target."""
        m, d, name = self.env.model, self.env.data, self.env.target
        body = m.body(name).id
        rot, center = d.xmat[body].reshape(3, 3), d.xpos[body].copy()
        angle = self.env.mechanisms.valve_angle(d, name)
        axis = rot[:, 2].copy()
        grip = d.site_xpos[m.site(f"{name}_grip").id] - center
        height = float(grip @ axis)
        lever = grip - height * axis
        lever /= np.linalg.norm(lever)
        # Undo the current angle so u is the lever direction at 0 rad.
        side = np.cross(axis, lever)
        u = np.cos(angle) * lever - np.sin(angle) * side
        return center, axis, u, np.cross(axis, u), height

    def arc_point(self, phi: float, lift: float = 0.0) -> np.ndarray:
        """Fingertip target at angle ``phi`` on the push arc, ``lift`` above contact height."""
        center, axis, u, v, height = self.frame
        return center + (height + CONTACT_DEPTH + lift) * axis + CONTACT_RADIUS * (np.cos(phi) * u + np.sin(phi) * v)

    def plan_direction(self) -> float:
        """+1 or -1: the turning direction whose fingertip arc the arm reaches best without the hand
        running into the valve's mounting (some valves have a block right beside the lever's arc)."""
        ik, axis = self.ik(), self.frame[1]
        theta0 = self.env.mechanisms.valve_angle(self.env.data, self.env.target)
        costs = {}
        for sign in (1.0, -1.0):
            cost, q = 0.0, ik.rest
            for phi in theta0 + sign * np.linspace(-START_GAP, np.pi / 2, 10):
                r = ik.solve(self.env.data.qpos, self.arc_point(phi), -axis, q_init=q)
                cost += r.pos_err + 0.02 * r.axis_err + 0.1 * self.hand_collides(ik.joints, r.q, ignore=(self.env.target,))
                q = r.q
            costs[sign] = cost
        return min(costs, key=costs.get)

    def _script(self):
        env = self.env
        self.phase = "approach"
        yield from self.move_joints(POINTING_HAND, 0.3)
        self.frame = self.valve_frame()
        axis = self.frame[1]
        sign = self.plan_direction()
        theta0 = env.mechanisms.valve_angle(env.data, env.target)
        start = theta0 - sign * START_GAP
        # The finger runs on to where the lever's far end stop is: the finger trails the lever by more than
        # its contact width once it pushes on the lever's rounded edge, and the stop plus the force guard
        # keep that from shoving the lever.
        end = sign * (np.pi / 2 + 0.25)

        # Short clearance: the top valves face up under the board frame's top rail.
        yield from self.reach(self.arc_point(start, self.hover), -axis, clearance=0.1)
        self.phase = "lower onto lever"
        yield from self.line(self.arc_point(start, self.hover), self.arc_point(start), -axis, 0.6)

        self.phase = "turn"
        done = lambda: abs(env.mechanisms.valve_angle(env.data, env.target)) >= 0.97 * np.pi / 2  # noqa: E731
        duration = abs(end - start) / self.sweep_speed
        yield from self.path(
            lambda s: (self.arc_point((1 - s) * start + s * end), -axis), duration, max_force=self.max_force, until=done
        )
        tip = self.site_pose()[0]
        self.phase = "lift off"
        yield from self.line(tip, tip + self.hover * axis, -axis, 0.5)
        self.phase = "retract"
        yield from self.retract()
