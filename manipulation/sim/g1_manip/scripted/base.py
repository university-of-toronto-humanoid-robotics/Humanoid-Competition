"""Shared machinery for the scripted IK controllers.

A controller's task is a generator of control steps (``_script``). The motion primitives
below are generators too, so a task reads as its action sequence:

    yield from self.reach(pre_press, normal)
    yield from self.line(pre_press, press, normal, 0.5)

Each step a primitive writes joint targets into ``self.q_cmd`` (one per controlled joint, in
the env's order) and yields; ``act`` turns them into an action through the env's normal
action interface.

The env should use joint_group "whole_body". The controller then runs a balance loop: the
ankles pitch to keep the center of mass over the middle of the feet (``balance_offsets``).
Without it the robot only stands on stiff position servos. Its CoM sits 3 cm behind the middle
of the feet, near the heel edge, and reaching forward or a push from a fixture rocks it over.

Wall-height fixtures: the arm IK moves the waist and right arm at the robot's current
(free-floating, swaying) base pose, like the locomotion button press.

Floor-level fixtures: ``crouch`` first plans a statically balanced crouch with ``WholeBodyIK``
(feet planted, CoM over the feet) that brings the hand near the fixture, and the legs' servos
carry it out. After that the arm IK moves only the right arm, so the waist and legs, and with
them the balance, stay as planned. ``stand_up`` replays the crouch backwards.

Joint limits: both IK solvers clip to the model's joint ranges and lock joints that reach
them. ``ARM_LIMITS`` and ``CROUCH_LIMITS`` tighten some ranges further where the full range
leads to collisions (the IK does not model them) or loses balance.

Fixture poses are read from the simulator (privileged state); on the robot they would come
from perception and the pre-mapped apparatus.
"""

from __future__ import annotations

from typing import Iterator

import mujoco
import numpy as np

from g1_manip import robot
from g1_manip.ik import IKResult, KeepOut, SiteIK, WholeBodyIK

# Right arm posture the IK null space pulls toward: upper arm raised forward, elbow bent.
RIGHT_ARM_REST = np.array([-1.0, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])

# Intermediate arm pose for the first approach at the wall: upper arm slightly back, elbow at 90
# degrees, forearm forward. The hand stays within 0.32 m of the pelvis on the way there.
READY_ARM = dict(zip(robot.arm("right"), [0.4, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0]))

# At the ends of their ranges (1.61 rad) the wrist pitch and yaw fold the wrist links into each other.
WRIST_LIMITS = {"right_wrist_pitch_joint": (-1.3, 1.3), "right_wrist_yaw_joint": (-1.3, 1.3)}

ARM_LIMITS = WRIST_LIMITS | {
    # Shoulder roll above this swings the upper arm into the torso for targets near the centerline.
    "right_shoulder_roll_joint": (-np.inf, -0.15),
    # The waist turns toward close, centered targets (otherwise the wrist runs into the chest);
    # keep its lean small so the free-floating base stays balanced.
    "waist_yaw_joint": (-0.6, 0.6),
    "waist_roll_joint": (-0.1, 0.1),
    "waist_pitch_joint": (-0.1, 0.2),
}
# The waist leans only a little: moving waist roll and pitch costs 10x an arm joint (weighted least
# norm), otherwise they take up most of every move (longest lever), end at their limits and swing
# the torso's mass off the feet. Yaw barely moves the CoM and turns the chest out of the way.
ARM_WEIGHTS = {"waist_yaw_joint": 2.0, "waist_roll_joint": 10.0, "waist_pitch_joint": 10.0}

# The chest (front face ~8 cm ahead of the torso origin, 11 cm to each side) plus the wrist links
# radius: reaching across the body, the IK would otherwise put the wrist inside the chest.
CHEST = KeepOut(
    frame="torso_link",
    lower=(-np.inf, -0.15, -0.05),
    upper=(0.15, 0.15, 0.5),
    points=(
        ("right_wrist_roll_link", (0.0, 0.0, 0.0)),
        ("right_wrist_yaw_link", (0.0, 0.0, 0.0)),
        ("right_wrist_yaw_link", (0.08, 0.0, 0.0)),  # palm
    ),
)

# Crouch posture the whole-body null space pulls toward: knees bent, hips and ankles flexed so
# the shins and torso stay parallel, the torso leaning forward on the waist.
CROUCH_REST = dict.fromkeys(robot.LEGS, 0.0) | {
    f"{side}_{j}_joint": v for side in ("left", "right") for j, v in (("hip_pitch", -0.6), ("knee", 1.2), ("ankle_pitch", -0.6))
}
CROUCH_LIMITS = WRIST_LIMITS | {
    "right_shoulder_roll_joint": (-np.inf, -0.15),
    "waist_yaw_joint": (-0.4, 0.4),  # more twist swings the left shoulder into the fixture frames
    "waist_roll_joint": (-0.1, 0.1),
    # Deeper hip flexion drives the thigh into the pelvis (self-collision), which tips the robot.
    "left_hip_pitch_joint": (-1.2, np.inf),
    "right_hip_pitch_joint": (-1.2, np.inf),
}
# For the crouch goal only: bend the knees instead of jackknifing at the hips with the knees
# locked straight at their (hyperextension) stop. The path to the goal starts from straight knees.
CROUCH_GOAL_LIMITS = CROUCH_LIMITS | {"left_knee_joint": (0.3, np.inf), "right_knee_joint": (0.3, np.inf)}

# Left arm bent at 90 degrees with the forearm forward while crouching (mirror of READY_ARM), so the
# hanging hand doesn't swing into the floor, the knees or the fixture frames as the torso pitches.
LEFT_ARM_TUCK = dict(zip(robot.arm("left"), [0.4, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]))

# Index finger straight, middle finger curled out of the way, thumb folded against the palm (at its
# stand pose it reaches 15 cm along the hand, 6 cm short of the index tip, and clips the boards).
POINTING_HAND = {
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.7,
    "right_hand_thumb_2_joint": -1.7,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 1.5,
    "right_hand_middle_1_joint": 1.7,
}


def min_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return s**3 * (10 - 15 * s + 6 * s**2)


class ScriptedIK:
    """Base class: runs ``_script`` one control step per ``act`` call."""

    def __init__(self, env, site: str = "right_index_tip", balance_gains: tuple[float, float] = (2.0, 0.3)):
        """
        Args:
            env: the unwrapped arena env, with joint_group "whole_body" ("right_arm_waist" works at the
                wall, without the balance loop). Its ``action_scale`` must cover the IK solutions (3.0
                rad covers every joint range used here).
            site: the end-effector site the task places.
            balance_gains: (rad/m, rad s/m) ankle pitch per CoM offset ahead of the middle of the feet,
                and per its rate.
        """
        self.env = env
        self.site = site
        model = env.model
        self.controlled = [model.actuator(i).name for i in env.ctrl_ids]
        self._slot = {n: i for i, n in enumerate(self.controlled)}
        if not set(robot.WAIST + robot.arm("right")) <= set(self.controlled):
            raise ValueError("expected an env with joint_group 'whole_body' or 'right_arm_waist'")
        self.balance = set(robot.LEGS) <= set(self.controlled)
        self.floor = any("floor" in f for f in env.station.fixtures)
        if self.floor and not self.balance:
            raise ValueError("floor fixtures need joint_group 'whole_body' (the robot crouches)")
        if self.floor:
            arm_joints, rest, limits = robot.arm("right"), RIGHT_ARM_REST, {k: ARM_LIMITS[k] for k in ARM_LIMITS if "waist" not in k}
        else:
            arm_joints, rest = robot.WAIST + robot.arm("right"), np.concatenate([np.zeros(3), RIGHT_ARM_REST])
            limits = ARM_LIMITS
        self._arm = (arm_joints, rest, limits)
        self._iks: dict[str, SiteIK] = {}
        self.balance_gains = balance_gains
        self.max_joint_speed = 2.5  # rad/s, cap on commanded joint speed in Cartesian moves
        feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self._feet = feet
        self._soles = np.flatnonzero(np.isin(model.geom_bodyid, feet) & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
        self._ankles = [self._slot[f"{side}_ankle_pitch_joint"] for side in ("left", "right")] if self.balance else []
        robot_root = model.body_rootid[env.pelvis_id]
        self._robot_bodies = set(np.flatnonzero(model.body_rootid == robot_root))
        self._hand_bodies = {i for i in range(model.nbody) if model.body(i).name.startswith(("right_hand", "right_wrist"))}
        self._force = np.zeros(6)
        self.reset()

    # ----- interface ------------------------------------------------------------

    def reset(self) -> None:
        """Plans from the current state. Call after ``env.reset()``."""
        self.q_cmd = self.env.data.qpos[self.env.qpos_ids].copy()
        self.phase = ""
        self.done = False
        self.last_result: IKResult | None = None
        self.crouch_plan: list[np.ndarray] | None = None
        self._com_err, self._com_rate, self._com_ref = None, 0.0, None
        self.peak_force = 0.0  # largest contact force between the right hand and the world so far (N)
        self._steps = self._script()

    def act(self) -> np.ndarray:
        """Next action for ``env.step``."""
        if not self.done:
            try:
                next(self._steps)
            except StopIteration:
                self.done = True
        q = self.q_cmd.copy()
        if self.balance:
            q[self._ankles] += self.balance_offset()
        return self.env.action_for(q)

    def hand_force(self) -> float:
        """Total normal contact force between the right hand (incl. wrist) and anything not on the robot (N).

        On the robot this would come from the wrist force/torque sensor or the fingers' joint torques.
        """
        m, d = self.env.model, self.env.data
        total = 0.0
        for i in range(d.ncon):
            b1, b2 = m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]
            if (b1 in self._hand_bodies and b2 not in self._robot_bodies) or (
                b2 in self._hand_bodies and b1 not in self._robot_bodies
            ):
                mujoco.mj_contactForce(m, d, i, self._force)
                total += self._force[0]
        self.peak_force = max(self.peak_force, total)
        return total

    def hand_collides(self, joints: list[str], q: np.ndarray, ignore: tuple[str, ...] = ()) -> bool:
        """Whether the right hand would penetrate the static arena with ``joints`` at ``q`` (planning check).

        Uses the robot's current pose for everything else. Bodies named in ``ignore`` (the fixture being
        manipulated) don't count.
        """
        m = self.env.model
        if not hasattr(self, "_probe"):
            self._probe = mujoco.MjData(m)
        probe = self._probe
        probe.qpos[:] = self.env.data.qpos
        probe.qpos[[m.jnt_qposadr[m.joint(j).id] for j in joints]] = q
        mujoco.mj_kinematics(m, probe)
        mujoco.mj_collision(m, probe)
        ignored = {m.body(b).id for b in ignore}
        for i in range(probe.ncon):
            b1, b2 = m.geom_bodyid[probe.contact[i].geom1], m.geom_bodyid[probe.contact[i].geom2]
            other = b2 if b1 in self._hand_bodies else b1 if b2 in self._hand_bodies else None
            if other is not None and other not in self._robot_bodies and other not in ignored:
                return True
        return False

    def com_error(self) -> float:
        """How far the CoM is ahead of the middle of the feet (m), along the feet's forward axis."""
        data = self.env.data
        forward = np.mean([data.xmat[f].reshape(3, 3)[:, 0] for f in self._feet], axis=0)
        offset = data.subtree_com[self.env.pelvis_id] - data.geom_xpos[self._soles].mean(axis=0)
        return float(offset[:2] @ forward[:2] / np.linalg.norm(forward[:2]))

    def balance_offset(self) -> float:
        """Ankle pitch correction (rad): pitching the ankles up tilts the body back and moves the CoM back.

        The CoM setpoint starts where the CoM is at reset and eases to the middle of the feet over about
        a second, so the loop doesn't kick the robot when it switches on.
        """
        err = self.com_error()
        if self._com_err is None:
            self._com_ref = err
        else:
            self._com_rate = 0.7 * self._com_rate + 0.3 * (err - self._com_err) / self.env.dt
            self._com_ref *= np.exp(-self.env.dt / 0.5)
        self._com_err = err
        kp, kd = self.balance_gains
        return float(np.clip(kp * (err - self._com_ref) + kd * self._com_rate, -0.25, 0.25))

    def _script(self) -> Iterator[None]:
        raise NotImplementedError

    # ----- helpers --------------------------------------------------------------

    def ik(self, site: str | None = None) -> SiteIK:
        """Arm IK for a site: waist + right arm at the wall, right arm only when crouched."""
        site = site or self.site
        if site not in self._iks:
            joints, rest, limits = self._arm
            self._iks[site] = SiteIK(
                self.env.model, site, joints, rest=rest, limits=limits, weights=ARM_WEIGHTS, keep_out=[CHEST]
            )
        return self._iks[site]

    def steps(self, duration: float) -> int:
        return max(1, int(round(duration / self.env.dt)))

    def get(self, joints: list[str]) -> np.ndarray:
        return self.q_cmd[[self._slot[n] for n in joints]]

    def set(self, joints: list[str], q: np.ndarray) -> None:
        self.q_cmd[[self._slot[n] for n in joints]] = q

    def site_pose(self, site: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Current (position, x axis) of a site in the simulation."""
        sid = self.env.model.site(site or self.site).id
        return self.env.data.site_xpos[sid].copy(), self.env.data.site_xmat[sid].reshape(3, 3)[:, 0].copy()

    # ----- motion primitives ------------------------------------------------------

    def move_joints(self, targets: dict[str, float], duration: float) -> Iterator[None]:
        """Minimum-jerk joint-space move of the named joints; the others hold."""
        names = list(targets)
        start, goal = self.get(names), np.array([targets[n] for n in names])
        n = self.steps(duration)
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            self.set(names, (1 - s) * start + s * goal)
            yield

    def move_to(
        self, pos: np.ndarray, direction: np.ndarray | None, duration: float, extra: dict[str, float] | None = None
    ) -> Iterator[None]:
        """Joint-space move to the arm pose that puts the site at ``pos`` (pointing along ``direction``).

        Used for the first approach: a smooth swing, where a Cartesian straight line from the hanging
        arm would drag the arm through awkward configurations. Seeded from the rest posture, which picks
        the natural elbow-down branch.
        """
        ik = self.ik()
        self.last_result = ik.solve(self.env.data.qpos, pos, direction, q_init=ik.rest)
        yield from self.move_joints(dict(zip(ik.joints, self.last_result.q)) | (extra or {}), duration)

    def path(
        self, fn, duration: float, site: str | None = None, max_force: float | None = None, until=None
    ) -> Iterator[None]:
        """Cartesian path ``fn(s) -> (pos, direction)`` for s in [0, 1] (minimum-jerk timing).

        IK is solved every control step at the robot's *current* base pose, warm-started from the
        previous solution, so the path holds while the free-floating base sways.

        Args:
            max_force: guarded move for paths that push on a fixture. While the hand's contact force is
                above this (N), the path stops advancing and backs off a step, so the position servos
                (stiff, ~5 kN/m at the hand) can't crush the fixture when it is stiffer or further in than
                expected. Backs off a tenth of the path at a time (a step near the end of the
                minimum-jerk profile is too small to relieve anything). Gives up after three times the
                planned duration.
            until: callable; the path stops as soon as it returns True (e.g. the button lit up).
        """
        ik = self.ik(site)
        n = self.steps(duration)
        k = 0
        for _ in range(3 * n if max_force is not None else n):
            if until is not None and until():
                break
            if max_force is None or self.hand_force() < max_force:
                k += 1
            else:
                k = max(k - max(1, n // 10), 0)
            pos, direction = fn(min_jerk(k / n))
            q = self.get(ik.joints)
            self.last_result = ik.solve(self.env.data.qpos, pos, direction, q_init=q)
            # Rate limit: if the IK flips to another branch (it can near the joint limits), the arm
            # moves over there slowly instead of lunging into whatever is in front of the hand.
            step = self.max_joint_speed * self.env.dt
            self.set(ik.joints, q + np.clip(self.last_result.q - q, -step, step))
            yield
            if k >= n:
                break

    def line(self, p0, p1, direction, duration: float, site=None, max_force=None, until=None):
        p0, p1 = np.asarray(p0), np.asarray(p1)
        yield from self.path(lambda s: ((1 - s) * p0 + s * p1, direction), duration, site, max_force, until)

    def hold(self, duration: float) -> Iterator[None]:
        for _ in range(self.steps(duration)):
            self.hand_force()  # keep the peak force up to date
            yield

    def reach(
        self, pos: np.ndarray, axis: np.ndarray, duration: float = 1.5, clearance: float = 0.2, direction=None
    ) -> Iterator[None]:
        """First approach to a fixture: the arm to ``READY_ARM``, then a swing at the wall or a crouch at
        floor level (raising the arm first keeps it from sweeping down past the thigh).

        Both end ``clearance`` back from ``pos`` along ``axis`` (the fixture axis, pointing into the
        board), and the last stretch is a straight line along the axis. At the wall the arm first bends
        into ``READY_ARM`` (forearm forward, hand ~0.3 m ahead of the pelvis), then swings to the arm
        pose for the clearance point. A single joint-space swing from the hanging arm straightens the
        elbow halfway and sweeps the hand through the board; a straight Cartesian rise runs the hand
        up the front of the torso.

        Args:
            direction: site pointing direction, defaults to ``axis``.
        """
        direction = axis if direction is None else direction
        yield from self.move_joints(READY_ARM, duration / 2)
        if self.floor:
            # Crouched, only the arm moves: a long last stretch would need the legs again.
            far = pos - min(clearance, 0.06) * np.asarray(axis)
            yield from self.crouch(far, direction, 2 * duration)
        else:
            far = pos - clearance * np.asarray(axis)
            yield from self.move_to(far, direction, duration / 2)
        yield from self.line(far, pos, direction, 0.8)

    def hop(
        self, p_from, axis_from, p_to, axis_to, duration: float = 1.0, clearance: float = 0.08, direction=None
    ) -> Iterator[None]:
        """From one fixture to the next: back off along ``axis_from``, swing, come in along ``axis_to``.

        The swing goes to the arm pose solved from the rest posture, not warm-started from the current
        one. Tracking a long path warm-started drifts into contorted poses at the joint limits (elbow
        at its stop, wrists at their ends) and then flips to another branch in one step.
        """
        dir_from = axis_from if direction is None else direction
        dir_to = axis_to if direction is None else direction
        back, far = p_from - clearance * np.asarray(axis_from), p_to - clearance * np.asarray(axis_to)
        yield from self.line(p_from, back, dir_from, 0.4)
        yield from self.move_to(far, dir_to, duration)
        yield from self.line(far, p_to, dir_to, 0.4)

    def retract(self, duration: float = 1.5) -> Iterator[None]:
        """Back to the standing pose (standing up first at floor level)."""
        if self.floor and self.crouch_plan is not None:
            yield from self.stand_up(duration * 2)
        joints = [n for n in robot.WAIST + robot.arm("right") + robot.arm("left") if n in self._slot]
        yield from self.move_joints({n: self.env.default_ctrl[self.env.model.actuator(n).id] for n in joints}, duration)

    # ----- crouching (floor fixtures) ---------------------------------------------

    def _crouch_ik(self, goal: bool) -> WholeBodyIK:
        key = f"crouch_goal:{self.site}" if goal else f"crouch:{self.site}"
        if key not in self._iks:
            joints = robot.LEGS + robot.WAIST + robot.arm("right")
            rest = np.array([CROUCH_REST.get(n, 0.0) for n in joints])
            rest[-7:] = RIGHT_ARM_REST
            limits = CROUCH_GOAL_LIMITS if goal else CROUCH_LIMITS
            self._iks[key] = WholeBodyIK(self.env.model, self.site, joints, rest=rest, limits=limits, keep_out=[CHEST])
        return self._iks[key]

    def _plan_joints(self) -> list[str]:
        return robot.LEGS + robot.WAIST + robot.arm("right") + robot.arm("left")

    def crouch(self, pos: np.ndarray, direction: np.ndarray | None, duration: float = 3.0) -> Iterator[None]:
        """Crouches so the site ends up at ``pos``, balanced the whole way down.

        The goal posture is one whole-body IK solve. The path to it is the joint-space interpolation
        from the current posture, with every waypoint projected back onto the constraints (feet where
        they stand, CoM over the support polygon) while keeping the hand where the interpolation puts
        it. Waypoints are solved from the previous *planned* configuration, not the measured one:
        feeding back the sagging, lagging base makes the legs chase it down.
        """
        data = self.env.data
        stance = data.qpos.copy()
        left = robot.arm("left")
        left_ids = [self.env.model.jnt_qposadr[self.env.model.joint(n).id] for n in left]
        left0, tuck = stance[left_ids].copy(), np.array(list(LEFT_ARM_TUCK.values()))
        guess = stance.copy()
        guess[left_ids] = tuck

        goal_ik, ik = self._crouch_ik(goal=True), self._crouch_ik(goal=False)
        goal = goal_ik.solve(guess, pos, direction, q_init=goal_ik.rest, stance=stance)
        self.last_result = goal
        q0 = stance[ik.qpos_ids].copy()
        n = self.steps(duration)
        plan, qplan = [], stance.copy()
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            qplan[ik.qpos_ids] = (1 - s) * q0 + s * goal.q
            qplan[left_ids] = (1 - s) * left0 + s * tuck
            hand, _ = ik.site_pose(qplan)
            qplan = ik.solve(qplan, hand, None, stance=stance, max_iters=50).qpos
            plan.append(qplan.copy())
        self.crouch_plan = plan

        joints = self._plan_joints()
        ids = [self.env.model.jnt_qposadr[self.env.model.joint(j).id] for j in joints]
        for qp in plan:
            self.set(joints, qp[ids])
            yield

    def stand_up(self, duration: float = 3.0) -> Iterator[None]:
        """Returns the arm to where the crouch left it, then replays the crouch backwards."""
        joints = self._plan_joints()
        ids = [self.env.model.jnt_qposadr[self.env.model.joint(j).id] for j in joints]
        end = self.crouch_plan[-1][ids]
        arm = robot.arm("right")
        yield from self.move_joints({j: end[joints.index(j)] for j in arm}, duration / 3)
        plan = self.crouch_plan[::-1]
        stride = max(1, int(round(len(plan) / self.steps(duration))))
        for qp in plan[::stride]:
            self.set(joints, qp[ids])
            yield
        self.crouch_plan = None
