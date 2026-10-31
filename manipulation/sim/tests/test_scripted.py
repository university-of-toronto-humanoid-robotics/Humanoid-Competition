import gymnasium as gym
import numpy as np
import pytest

import g1_manip  # noqa: F401
from g1_manip import arena, robot
from g1_manip.ik import KeepOut, SiteIK, WholeBodyIK
from g1_manip.scripted import TASKS

pytestmark = pytest.mark.skipif(not arena.ARENA_XML.exists(), reason="arena not built (scripts/build_arena.py)")


@pytest.fixture(scope="module")
def keypad_env():
    env = gym.make("G1ArenaKeypad-v0", joint_group="whole_body", action_scale=3.0)
    env.reset(seed=0)
    yield env.unwrapped
    env.close()


def test_weights_shift_motion_off_heavy_joints(keypad_env):
    env = keypad_env
    joints = robot.WAIST + robot.arm("right")
    face = env.data.site_xpos[env.model.site("keypad_5_site").id] - np.array([0.0, 0.0, 0.0])
    target = face + np.array([0.05, 0.0, 0.0])  # station faces -x: 5 cm in front of the key
    light = SiteIK(env.model, "right_index_tip", joints).solve(env.data.qpos, target)
    heavy = SiteIK(env.model, "right_index_tip", joints, weights=dict.fromkeys(robot.WAIST, 10.0)).solve(env.data.qpos, target)
    assert light.converged and heavy.converged
    assert np.abs(heavy.q[:3]).sum() < np.abs(light.q[:3]).sum()


def test_keep_out_box_is_respected(keypad_env):
    env = keypad_env
    box = KeepOut("torso_link", (-np.inf, -0.15, -0.05), (0.15, 0.15, 0.5), (("right_wrist_yaw_link", (0.0, 0.0, 0.0)),))
    ik = SiteIK(env.model, "right_index_tip", robot.arm("right"), keep_out=[box])
    # A target right in front of the chest, which the plain solver reaches with the wrist inside it.
    torso = env.model.body("torso_link").id
    rot, origin = env.data.xmat[torso].reshape(3, 3), env.data.xpos[torso]
    result = ik.solve(env.data.qpos, origin + rot @ np.array([0.2, 0.0, 0.15]))
    qpos = env.data.qpos.copy()
    qpos[ik.qpos_ids] = result.q
    ik.site_pose(qpos)
    wrist = rot.T @ (ik.data.xpos[env.model.body("right_wrist_yaw_link").id] - origin)
    assert not (wrist[0] < 0.15 - 0.01 and abs(wrist[1]) < 0.15 - 0.01)


def test_whole_body_ik_keeps_feet_and_balance():
    env = gym.make("G1Arena-v0", station="button_floor", joint_group="whole_body").unwrapped
    env.reset(seed=0)
    joints = robot.LEGS + robot.WAIST + robot.arm("right")
    ik = WholeBodyIK(env.model, "right_index_tip", joints)
    target = env.data.site_xpos[env.model.site("board_floor_2_site").id] + np.array([-0.05, 0.0, 0.05])
    result = ik.solve(env.data.qpos, target, q_init=None)
    assert result.converged and result.feet_err < 1e-3 and result.com_err < 2e-3


@pytest.mark.parametrize("task", ["keypad", "button_wall"])
def test_button_controllers_press_gently(task):
    env_id, kwargs, controller_cls = TASKS[task]
    env = gym.make(env_id, action_scale=3.0, **kwargs)
    controller = None
    for seed in range(2):
        env.reset(seed=seed)
        controller = controller_cls(env.unwrapped) if controller is None else controller
        controller.reset()
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(controller.act())
            done = terminated or truncated
        assert info["is_success"] and not info.get("fell", False), seed
        assert controller.peak_force < 15.0, controller.peak_force
    env.close()
