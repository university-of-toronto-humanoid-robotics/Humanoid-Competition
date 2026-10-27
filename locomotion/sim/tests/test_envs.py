import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

import g1_loco  # noqa: F401

ENV_ID = "G1ButtonPressFloating-v0"


def test_env_api():
    env = gym.make(ENV_ID)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


def test_rollout_is_finite_and_robot_stays_up():
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            break
    assert not info.get("fell", False)
    env.close()


def test_button_presses_and_springs_back():
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=0)
    button = env.model.body("button").id
    env.data.xfrc_applied[button] = [10.0, 0, 0, 0, 0, 0]
    mujoco.mj_step(env.model, env.data, 200)
    assert env.button_depth() >= 0.008
    env.data.xfrc_applied[button] = 0.0
    mujoco.mj_step(env.model, env.data, 200)
    assert env.button_depth() < 0.001


def test_action_for_inverts_action_mapping():
    env = gym.make(ENV_ID, action_scale=3.0).unwrapped
    env.reset(seed=0)
    targets = env.default_ctrl[env.ctrl_ids] + 0.3
    env.step(env.action_for(targets))
    np.testing.assert_allclose(env.data.ctrl[env.ctrl_ids], np.clip(targets, env.ctrl_low, env.ctrl_high))
