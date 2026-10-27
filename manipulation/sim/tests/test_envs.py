import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import g1_manip  # noqa: F401

ENV_IDS = ["G1ButtonPress-v0", "G1ButtonPressFloating-v0"]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_api(env_id):
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_rollout_is_finite_and_robot_stays_up(env_id):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if terminated or truncated:
            break
    assert not info.get("fell", False)
    env.close()


def test_button_presses_and_springs_back():
    env = gym.make("G1ButtonPress-v0").unwrapped
    env.reset(seed=0)
    button = env.model.body("button").id
    env.data.xfrc_applied[button] = [10.0, 0, 0, 0, 0, 0]
    mujoco.mj_step(env.model, env.data, 200)
    assert env.button_depth() >= 0.008
    env.data.xfrc_applied[button] = 0.0
    mujoco.mj_step(env.model, env.data, 200)
    assert env.button_depth() < 0.001


def test_rgb_render():
    env = gym.make("G1ButtonPress-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (480, 640, 3) and frame.std() > 0
    env.close()
