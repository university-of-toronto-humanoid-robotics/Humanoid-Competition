"""Damped least-squares inverse kinematics for the G1.

Two solvers share one task-priority step:

- ``SiteIK`` (same as ``locomotion/sim/g1_loco/ik.py``): puts a site at a world position, optionally
  with one of its local axes along a world direction, by moving a set of hinge joints. Everything
  else, including the floating base, stays where it is, so the solution is valid for the robot's
  current base pose.
- ``WholeBodyIK``: the same site task, but the floating base is free too and the legs are used.
  Both feet stay where they are and the center of mass stays over the support polygon (top
  priority), so it finds statically balanced postures, e.g. a crouch for floor-level fixtures.

Joints that hit their limits are locked out of the step. A null-space term pulls redundant DoFs
toward a rest posture. Optional ``KeepOut`` boxes keep chosen points of the robot out of a region of
another body (crude self-collision avoidance, e.g. the wrist out of the chest). The solvers work on their own scratch ``MjData``; the simulation is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class IKResult:
    q: np.ndarray  # solved angles of the IK joints, in the order they were given
    pos_err: float  # m
    axis_err: float  # |sin| of the angle between the site axis and the target direction
    iters: int
    converged: bool  # position within tolerance; pointing is best effort, see axis_err


@dataclass(frozen=True)
class KeepOut:
    """A box in a body's frame that some points on the robot must stay out of.

    While a point is inside, the solver pushes it out through the nearest face, ahead of the site task.
    """

    frame: str  # body the box is fixed to, e.g. "torso_link"
    lower: tuple[float, float, float]  # box corners in that body's frame (+-inf for open sides)
    upper: tuple[float, float, float]
    points: tuple[tuple[str, tuple[float, float, float]], ...]  # (body, offset in that body's frame)


@dataclass
class WholeBodyResult(IKResult):
    qpos: np.ndarray = None  # full solved configuration, incl. the base
    com_err: float = 0.0  # m, horizontal
    feet_err: float = 0.0  # m (+ rad), largest foot pose error


def prioritized_step(tasks, q, lower, upper, posture, weights=None) -> np.ndarray:
    """One task-priority step: each task in the null space of the ones before it, then posture.

    Args:
        tasks: [(jacobian, error, damping)], highest priority first.
        q, lower, upper: current values and bounds of the variables (+-inf for unbounded ones).
        posture: desired change per variable, applied in the remaining null space.
        weights: cost of moving each variable (weighted least norm); a heavier joint takes a smaller
            share of the motion. None weighs all equally.

    Variables at a limit that the step would push further are locked (removed from the
    Jacobians) and the step recomputed, so clipping can't undo the projections.
    """
    n = len(q)
    # Solve in scaled variables dy = sqrt(w) dq, where the plain least-norm step is the weighted one.
    scale = np.ones(n) if weights is None else 1.0 / np.sqrt(weights)
    free = np.ones(n, dtype=bool)
    for _ in range(n):
        dy = np.zeros(n)
        null = np.eye(free.sum())
        for jac, err, damping in tasks:
            jac = jac[:, free] * scale[free]
            jac_null = jac @ null
            # Damped pseudo-inverse for the step; exact projector so lower priorities don't leak upward.
            pinv_damped = jac_null.T @ np.linalg.inv(jac_null @ jac_null.T + damping * np.eye(len(err)))
            dy[free] += pinv_damped @ (err - jac @ dy[free])
            null = null - np.linalg.pinv(jac_null, rcond=1e-4) @ jac_null
        dy[free] += null @ (posture[free] / scale[free])
        dq = dy * scale

        pushing = free & (((q <= lower + 1e-9) & (dq < 0)) | ((q >= upper - 1e-9) & (dq > 0)))
        if not pushing.any():
            break
        free &= ~pushing
    return dq


def _joint_bounds(model, joints, limits):
    jnt_ids = np.array([model.joint(n).id for n in joints])
    if np.any(model.jnt_type[jnt_ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise ValueError("only hinge joints can be solved for")
    lower, upper = model.jnt_range[jnt_ids].T.copy()
    for name, (lo, hi) in (limits or {}).items():
        i = joints.index(name)
        lower[i], upper[i] = max(lo, lower[i]), min(hi, upper[i])
    return model.jnt_qposadr[jnt_ids], model.jnt_dofadr[jnt_ids], lower, upper


class SiteIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: list[str],
        rest: np.ndarray | None = None,
        limits: dict[str, tuple[float, float]] | None = None,
        weights: dict[str, float] | None = None,
        keep_out: list[KeepOut] | None = None,
        site_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_gain: float = 0.5,
        null_gain: float = 0.05,
        damping: float = 1e-3,
        axis_damping: float = 0.05,
        max_step: float = 0.1,
    ):
        """
        Args:
            site: site to place, e.g. "right_index_tip".
            joints: hinge joints the solver may move.
            rest: posture the null-space term pulls toward (same order as ``joints``); defaults to mid-range.
            limits: per-joint (lower, upper) overrides, tighter than the model's joint ranges, e.g. to keep
                the arm off the torso (the solver does not check collisions).
            weights: per-joint motion cost (default 1), e.g. > 1 on the waist so the arm does most of
                the work: the waist has the longest lever and would otherwise take up most of every move.
            keep_out: regions some robot points must stay out of, see ``KeepOut``.
            site_axis: the site-frame axis aligned with ``target_dir`` in ``solve``.
            axis_gain: fraction of the pointing error corrected per iteration. Pointing is a secondary
                task: it only uses the joint motion left over after the position task.
            null_gain: fraction of the rest-posture error corrected per iteration, in the null space.
            damping: Levenberg-Marquardt damping, trades accuracy near singularities for smoothness.
            axis_damping: damping of the pointing task, larger so it doesn't take big joint steps that
                knock the site off target through the nonlinearity.
            max_step: per-iteration cap on any joint change (rad).
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        self.joints = list(joints)
        self.qpos_ids, self.dof_ids, self.lower, self.upper = _joint_bounds(model, self.joints, limits)
        self.rest = 0.5 * (self.lower + self.upper) if rest is None else np.asarray(rest, dtype=float)
        self.weights = np.array([(weights or {}).get(n, 1.0) for n in self.joints])
        self.site_axis = np.asarray(site_axis, dtype=float)
        self.axis_gain = axis_gain
        self.null_gain = null_gain
        self.damping = damping
        self.axis_damping = axis_damping
        self.max_step = max_step
        self.keep_out = [
            (model.body(box.frame).id, np.array(box.lower, float), np.array(box.upper, float),
             [(model.body(b).id, np.array(off, float)) for b, off in box.points])
            for box in keep_out or []
        ]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def site_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: (site position, site_axis in world) for a full qpos."""
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self._current_pose()

    def _current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        xmat = self.data.site_xmat[self.site_id].reshape(3, 3)
        return self.data.site_xpos[self.site_id].copy(), xmat @ self.site_axis

    def _site_tasks(self, target_pos, target_dir, dof_ids):
        """(position task, pointing task or None, pos_err, axis_err) at the current scratch state."""
        pos, axis = self._current_pose()
        pos_task = target_pos - pos
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        axis_err, axis_task = 0.0, None
        if target_dir is not None:
            # Rotation vector that turns the current axis onto the target: d(axis)/dt = w x axis.
            axis_rot = np.cross(axis, target_dir)
            axis_err = float(np.linalg.norm(axis_rot))
            axis_task = (self._jacr[:, dof_ids], self.axis_gain * axis_rot, self.axis_damping)
        pos_task = (self._jacp[:, dof_ids], pos_task, self.damping)
        return pos_task, axis_task, float(np.linalg.norm(pos_task[1])), axis_err

    def _keep_out_task(self, dof_ids):
        """Task pushing every point that is inside a keep-out box out through the nearest face, or None."""
        m, d = self.model, self.data
        rows, errs = [], []
        jac_point, jac_frame = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for frame, lower, upper, points in self.keep_out:
            rot, origin = d.xmat[frame].reshape(3, 3), d.xpos[frame]
            for body, offset in points:
                point = d.xpos[body] + d.xmat[body].reshape(3, 3) @ offset
                local = rot.T @ (point - origin)
                if not np.all((local > lower) & (local < upper)):
                    continue
                depths = np.concatenate([local - lower, upper - local])  # to each face, -x -y -z +x +y +z
                face = int(np.argmin(depths))
                normal = rot[:, face % 3] * (-1.0 if face < 3 else 1.0)
                # Motion of the point relative to the box: both bodies may move with the solved joints.
                mujoco.mj_jac(m, d, jac_point, None, point, body)
                mujoco.mj_jac(m, d, jac_frame, None, point, frame)
                rows.append(normal @ (jac_point - jac_frame)[:, dof_ids])
                errs.append(depths[face] + 0.005)
        return (np.array(rows), np.array(errs), self.damping) if rows else None

    def solve(
        self,
        qpos: np.ndarray,
        target_pos: np.ndarray,
        target_dir: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        max_iters: int = 300,
        pos_tol: float = 1e-4,
        axis_tol: float = 1e-3,
    ) -> IKResult:
        """
        Args:
            qpos: full configuration that fixes every joint not being solved for (incl. the base).
            target_pos: world position for the site.
            target_dir: world direction for ``site_axis``; None solves for position only.
            q_init: initial guess for the IK joints (warm start); defaults to their values in ``qpos``.
        """
        self.data.qpos[:] = qpos
        if q_init is not None:
            self.data.qpos[self.qpos_ids] = q_init
        target_pos = np.asarray(target_pos, dtype=float)
        if target_dir is not None:
            target_dir = np.asarray(target_dir, dtype=float) / np.linalg.norm(target_dir)

        pos_err = axis_err = np.inf
        for it in range(max_iters):
            # mj_jacSite needs the body poses (mj_kinematics) and the motion dofs cdof (mj_comPos).
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            pos_task, axis_task, pos_err, axis_err = self._site_tasks(target_pos, target_dir, self.dof_ids)
            keep_out = self._keep_out_task(self.dof_ids)
            if pos_err < pos_tol and axis_err < axis_tol and keep_out is None:
                break

            q = self.data.qpos[self.qpos_ids]
            tasks = ([keep_out] if keep_out else []) + [pos_task] + ([axis_task] if axis_task is not None else [])
            dq = prioritized_step(tasks, q, self.lower, self.upper, self.null_gain * (self.rest - q), self.weights)
            dq *= min(1.0, self.max_step / max(np.abs(dq).max(), 1e-12))
            if np.abs(dq).max() < 1e-7:  # stalled: pointing is as good as it gets at this position
                break
            self.data.qpos[self.qpos_ids] = np.clip(q + dq, self.lower, self.upper)

        return IKResult(
            q=self.data.qpos[self.qpos_ids].copy(),
            pos_err=pos_err,
            axis_err=axis_err,
            iters=it,
            converged=pos_err < pos_tol,
        )


class WholeBodyIK(SiteIK):
    """``SiteIK`` with a free floating base, planted feet and a balanced center of mass.

    Tasks, highest priority first: both feet keep their pose and the CoM stays (horizontally)
    over the middle of the support polygon; the site position; the site pointing; the rest posture.
    The base pose comes out of the solution. On the robot, the legs' position servos produce it.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: list[str],
        feet: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
        com_offset: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            joints: hinge joints the solver may move, normally legs + waist + one arm.
            feet: bodies kept in place (6D).
            com_offset: CoM target ahead of the support center (m, along the feet's forward axis).
            **kwargs: as for ``SiteIK``.
        """
        super().__init__(model, site, joints, **kwargs)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) == 0:
            raise ValueError("WholeBodyIK needs a floating base")
        base_dof = model.jnt_dofadr[free[0]]
        self.base_dofs = np.arange(base_dof, base_dof + 6)
        self.var_dofs = np.concatenate([self.base_dofs, self.dof_ids])
        self.feet = [model.body(f).id for f in feet]
        # Contact points of the soles; their centroid is the middle of the support polygon.
        self.sole_geoms = np.flatnonzero(np.isin(model.geom_bodyid, self.feet) & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
        self.root = model.body("pelvis").id
        self.com_offset = com_offset
        self._jacc = np.zeros((3, model.nv))

    def support_center(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        return self._support_center()

    def _support_center(self) -> np.ndarray:
        center = self.data.geom_xpos[self.sole_geoms].mean(axis=0)
        forward = np.mean([self.data.xmat[f].reshape(3, 3)[:, 0] for f in self.feet], axis=0)
        return center + self.com_offset * forward

    def com(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.subtree_com[self.root].copy()

    def solve(
        self,
        qpos: np.ndarray,
        target_pos: np.ndarray,
        target_dir: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        stance: np.ndarray | None = None,
        max_iters: int = 500,
        pos_tol: float = 1e-3,
        axis_tol: float = 1e-2,
        com_tol: float = 2e-3,
    ) -> WholeBodyResult:
        """
        Args:
            qpos: initial guess for the full configuration, incl. the base.
            q_init: warm start for the IK joints, overriding their values in ``qpos``.
            stance: full configuration whose foot poses (and support polygon) are kept; defaults to
                ``qpos``. Pass the configuration the robot is actually standing in when ``qpos`` is a
                guess whose feet have moved.
        """
        m, d = self.model, self.data
        d.qpos[:] = qpos if stance is None else stance
        mujoco.mj_kinematics(m, d)
        feet_pos = [d.xpos[f].copy() for f in self.feet]
        feet_quat = [d.xquat[f].copy() for f in self.feet]
        com_target = self._support_center()[:2]
        d.qpos[:] = qpos
        if q_init is not None:
            d.qpos[self.qpos_ids] = q_init
        target_pos = np.asarray(target_pos, dtype=float)
        if target_dir is not None:
            target_dir = np.asarray(target_dir, dtype=float) / np.linalg.norm(target_dir)

        n_base = len(self.base_dofs)
        lower = np.concatenate([np.full(n_base, -np.inf), self.lower])
        upper = np.concatenate([np.full(n_base, np.inf), self.upper])
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        qvel = np.zeros(m.nv)
        pos_err = axis_err = com_err = feet_err = np.inf
        for it in range(max_iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

            # Balance task: feet (6 each) + CoM xy (2).
            rows, errs, feet_err = [], [], 0.0
            for f, p, quat in zip(self.feet, feet_pos, feet_quat):
                mujoco.mj_jacBody(m, d, jacp, jacr, f)
                rot = np.zeros(3)
                mujoco.mju_subQuat(rot, quat, d.xquat[f])  # in the foot frame
                rot = d.xmat[f].reshape(3, 3) @ rot
                rows += [jacp[:, self.var_dofs].copy(), jacr[:, self.var_dofs].copy()]
                errs += [p - d.xpos[f], rot]
                feet_err = max(feet_err, float(np.linalg.norm(p - d.xpos[f])), float(np.linalg.norm(rot)))
            mujoco.mj_jacSubtreeCom(m, d, self._jacc, self.root)
            com = com_target - d.subtree_com[self.root][:2]
            com_err = float(np.linalg.norm(com))
            rows.append(self._jacc[:2, self.var_dofs])
            errs.append(com)
            balance = (np.vstack(rows), np.concatenate(errs), 1e-4)

            pos_task, axis_task, pos_err, axis_err = self._site_tasks(target_pos, target_dir, self.var_dofs)
            keep_out = self._keep_out_task(self.var_dofs)
            if pos_err < pos_tol and axis_err < axis_tol and com_err < com_tol and feet_err < 1e-3 and keep_out is None:
                break

            q = np.concatenate([np.zeros(n_base), d.qpos[self.qpos_ids]])
            posture = np.concatenate([np.zeros(n_base), self.null_gain * (self.rest - q[n_base:])])
            tasks = [balance] + ([keep_out] if keep_out else []) + [pos_task] + ([axis_task] if axis_task is not None else [])
            dq = prioritized_step(tasks, q, lower, upper, posture, np.concatenate([np.ones(n_base), self.weights]))
            dq *= min(1.0, self.max_step / max(np.abs(dq).max(), 1e-12))
            if np.abs(dq).max() < 1e-7:
                break
            qvel[:] = 0.0
            qvel[self.var_dofs] = dq
            mujoco.mj_integratePos(m, d.qpos, qvel, 1.0)
            d.qpos[self.qpos_ids] = np.clip(d.qpos[self.qpos_ids], self.lower, self.upper)

        return WholeBodyResult(
            q=d.qpos[self.qpos_ids].copy(),
            pos_err=pos_err,
            axis_err=axis_err,
            iters=it,
            converged=pos_err < pos_tol and com_err < com_tol and feet_err < 1e-3,
            qpos=d.qpos.copy(),
            com_err=com_err,
            feet_err=feet_err,
        )
