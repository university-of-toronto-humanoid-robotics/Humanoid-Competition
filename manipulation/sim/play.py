"""Roll out a trained policy (or random actions) and report success rate.

    python play.py --env G1ButtonPress-v0 --run runs/G1ButtonPress-v0
    mjpython play.py --env G1ButtonPress-v0 --run runs/G1ButtonPress-v0 --render   # macOS viewer
"""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import g1_manip  # noqa: F401  (registers envs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="G1ButtonPress-v0")
    parser.add_argument("--run", type=Path, default=None, help="run dir from train.py; omit for random actions")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="open the MuJoCo viewer (use mjpython on macOS)")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    venv = DummyVecEnv([lambda: gym.make(args.env, render_mode=render_mode)])
    policy = None
    if args.run is not None:
        venv = VecNormalize.load(str(args.run / "vecnormalize.pkl"), venv)
        venv.training = False
        venv.norm_reward = False
        policy = PPO.load(args.run / "policy", device="cpu")

    dt = venv.get_attr("dt")[0]  # control step in sim seconds
    successes, returns = [], []
    obs = venv.reset()
    next_step = time.perf_counter()
    for _ in range(args.episodes):
        done, ep_return, info = False, 0.0, {}
        while not done:
            if policy is None:
                action = np.array([venv.action_space.sample()])
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            if args.render:  # pace to wall-clock time
                next_step += dt
                while time.perf_counter() < next_step:  # poll; sleep() oversleeps and lags the viewer
                    pass
            ep_return += float(reward[0])
            done, info = bool(dones[0]), infos[0]
        successes.append(bool(info.get("is_success", False)))
        returns.append(ep_return)
        print(f"return={ep_return:7.2f}  success={successes[-1]}")
    print(f"success rate: {np.mean(successes):.0%}  mean return: {np.mean(returns):.2f}")
    venv.close()


if __name__ == "__main__":
    main()
