import itertools

import gymnasium as gym
import numpy as np
import pytest

import g1_loco  # noqa: F401
from g1_loco import robot
from g1_loco.button_ik import RIGHT_ARM_REST, ButtonPressIK
from g1_loco.ik import SiteIK

ENV_ID = "G1ButtonPressFloating-v0"


@pytest.fixture(scope="module")
def env():
    env = gym.make(ENV_ID, joint_group="right_arm_waist", action_scale=3.0)
    env.reset(seed=0)
    yield env.unwrapped
    env.close()


def button_grid(env):
    """Corners and center of the env's button range, as pre-press fingertip targets."""
    lo, hi = env.target_range.T
    points = [np.array(c) for c in itertools.product(*env.target_range)] + [0.5 * (lo + hi)]
    return [p - np.array([0.076, 0.0, 0.0]) for p in points]


def test_ik_reaches_button_range_with_finger_pointing_in(env):
    ik = SiteIK(env.model, "right_index_tip", robot.arm("right"), rest=RIGHT_ARM_REST)
    for target in button_grid(env):
        result = ik.solve(env.data.qpos, target, target_dir=[1.0, 0.0, 0.0], q_init=ik.rest)
        assert result.converged, target
        assert np.all(result.q >= ik.lower) and np.all(result.q <= ik.upper)
        qpos = env.data.qpos.copy()
        qpos[ik.qpos_ids] = result.q
        pos, axis = ik.site_pose(qpos)
        np.testing.assert_allclose(pos, target, atol=1e-4)
        assert axis[0] > np.cos(np.radians(20))  # pointing is best effort, but close


def test_ik_respects_limit_overrides(env):
    roll = "right_shoulder_roll_joint"
    ik = SiteIK(env.model, "right_index_tip", robot.arm("right"), rest=RIGHT_ARM_REST, limits={roll: (-np.inf, -0.3)})
    i = robot.arm("right").index(roll)
    for target in button_grid(env):
        assert ik.solve(env.data.qpos, target, q_init=ik.rest).q[i] <= -0.3 + 1e-9


def test_ik_does_not_touch_simulation(env):
    qpos = env.data.qpos.copy()
    SiteIK(env.model, "right_index_tip", robot.arm("right")).solve(qpos, [0.3, -0.2, 1.0])
    np.testing.assert_array_equal(env.data.qpos, qpos)


@pytest.mark.parametrize("joint_group", ["right_arm_waist", "right_arm"])
def test_controller_presses_button(joint_group):
    env = gym.make(ENV_ID, joint_group=joint_group, action_scale=3.0)
    controller = None
    for seed in range(5):
        env.reset(seed=seed)
        if controller is None:
            controller = ButtonPressIK(env.unwrapped)
        else:
            controller.reset()
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(controller.act())
            done = terminated or truncated
        assert info["is_success"] and not info.get("fell", False), seed
    env.close()
