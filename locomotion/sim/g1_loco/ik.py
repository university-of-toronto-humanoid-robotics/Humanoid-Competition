"""Damped least-squares inverse kinematics for a site on the G1.

Solves for a subset of hinge joints (e.g. one arm) that puts a site at a world
position, optionally with one of the site's local axes pointing along a world
direction as a lower-priority task (it never costs position accuracy). All other joints, including the floating base, stay at the
configuration passed in, so the solution is valid for the robot's current base
pose. A null-space term pulls the redundant DoFs toward a rest posture, which
keeps the 7-DoF arm out of awkward local minima.

The solver works on its own scratch ``MjData``; the simulation is never touched.
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


class SiteIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site: str,
        joints: list[str],
        rest: np.ndarray | None = None,
        limits: dict[str, tuple[float, float]] | None = None,
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
            site_axis: the site-frame axis aligned with ``target_dir`` in ``solve``.
            axis_gain: fraction of the pointing error corrected per iteration. Pointing is a secondary
                task: it only uses the joint motion left over after the position task.
            null_gain: fraction of the rest-posture error corrected per iteration, in the null space.
            damping: Levenberg-Marquardt damping, trades accuracy near singularities for smoothness.
            axis_damping: damping of the pointing task, larger so it doesn't take big joint steps that
                knock the fingertip off target through the nonlinearity.
            max_step: per-iteration cap on any joint change (rad).
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site).id
        jnt_ids = np.array([model.joint(n).id for n in joints])
        if np.any(model.jnt_type[jnt_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("SiteIK only supports hinge joints")
        self.qpos_ids = model.jnt_qposadr[jnt_ids]
        self.dof_ids = model.jnt_dofadr[jnt_ids]
        self.lower, self.upper = model.jnt_range[jnt_ids].T.copy()
        for name, (lo, hi) in (limits or {}).items():
            i = joints.index(name)
            self.lower[i], self.upper[i] = max(lo, self.lower[i]), min(hi, self.upper[i])
        self.rest = 0.5 * (self.lower + self.upper) if rest is None else np.asarray(rest, dtype=float)
        self.site_axis = np.asarray(site_axis, dtype=float)
        self.axis_gain = axis_gain
        self.null_gain = null_gain
        self.damping = damping
        self.axis_damping = axis_damping
        self.max_step = max_step
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
            pos, axis = self._current_pose()
            pos_task = target_pos - pos
            pos_err = float(np.linalg.norm(pos_task))
            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
            axis_err, axis_task = 0.0, None
            if target_dir is not None:
                # Rotation vector that turns the current axis onto the target: d(axis)/dt = w x axis.
                axis_rot = np.cross(axis, target_dir)
                axis_err = float(np.linalg.norm(axis_rot))
                axis_task = (self._jacr[:, self.dof_ids], self.axis_gain * axis_rot)
            if pos_err < pos_tol and axis_err < axis_tol:
                break

            q = self.data.qpos[self.qpos_ids]
            dq = self._step(q, (self._jacp[:, self.dof_ids], pos_task), axis_task)
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

    def _step(self, q, pos_task, axis_task) -> np.ndarray:
        """One task-priority step: position, then pointing in its null space, then posture.

        Joints at a limit that the step would push further are locked (removed from
        the Jacobians) and the step recomputed, so clipping can't undo the projections.
        """
        n = len(q)
        free = np.ones(n, dtype=bool)
        for _ in range(n):
            dq = np.zeros(n)
            null = np.eye(free.sum())
            for task in [pos_task] + ([axis_task] if axis_task is not None else []):
                jac, err = task
                jac = jac[:, free]
                jac_null = jac @ null
                # Damped pseudo-inverse for the step; exact projector so lower priorities don't leak upward.
                damping = self.damping if task is pos_task else self.axis_damping
                pinv_damped = jac_null.T @ np.linalg.inv(jac_null @ jac_null.T + damping * np.eye(len(err)))
                dq[free] += pinv_damped @ (err - jac @ dq[free])
                null = null - np.linalg.pinv(jac_null, rcond=1e-4) @ jac_null
            dq[free] += null @ (self.null_gain * (self.rest[free] - q[free]))

            pushing = free & (((q <= self.lower + 1e-9) & (dq < 0)) | ((q >= self.upper - 1e-9) & (dq > 0)))
            if not pushing.any():
                break
            free &= ~pushing
        return dq
