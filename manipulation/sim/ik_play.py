"""Run a scripted IK controller on a manipulation task and report its success rate.

    python ik_play.py --task keypad --episodes 10
    python ik_play.py --task all                  # every task
    mjpython ik_play.py --task valve_floor --render  # macOS viewer
"""

import argparse
import time

import gymnasium as gym
import numpy as np

import g1_manip  # noqa: F401  (registers envs)
from g1_manip.scripted import TASKS


def run(task: str, episodes: int, seed: int, render: bool, max_seconds: float | None) -> float:
    env_id, kwargs, controller_cls = TASKS[task]
    # The RL default (0.75 rad around stand) can't reach the IK solutions; 3 rad covers the joint ranges used.
    kwargs = {"action_scale": 3.0, **kwargs}
    if max_seconds is not None:
        kwargs["max_episode_steps"] = int(max_seconds * 50)
    env = gym.make(env_id, render_mode="human" if render else None, **kwargs)
    controller, successes, times = None, [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        if controller is None:
            controller = controller_cls(env.unwrapped)
        else:
            controller.reset()
        dt = env.unwrapped.dt
        next_step = time.perf_counter()
        done, info, steps, ik_err = False, {}, 0, 0.0
        while not done:
            _, _, terminated, truncated, info = env.step(controller.act())
            if controller.last_result is not None:
                ik_err = max(ik_err, controller.last_result.pos_err)
            steps += 1
            done = terminated or truncated
            if render:  # pace to wall-clock time
                next_step += dt
                while time.perf_counter() < next_step:  # poll; sleep() oversleeps and lags the viewer
                    pass
        success = bool(info.get("is_success", False))
        successes.append(success)
        if success:
            times.append(steps * dt)
        extra = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items() if k not in ("is_success", "distance")}
        print(
            f"{task:13s} episode {ep:3d}  success={success!s:5s}  t={steps * dt:5.2f}s  phase={controller.phase!r}  "
            f"max_ik_err={ik_err * 1000:.1f}mm  peak_hand_force={controller.peak_force:.0f}N  {extra}"
        )
    print(f"{task}: success rate {np.mean(successes):.0%}  mean time {np.mean(times) if times else np.nan:.1f}s")
    env.close()
    return float(np.mean(successes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="keypad", choices=[*TASKS, "all"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=None, help="override the task's episode time limit")
    parser.add_argument("--render", action="store_true", help="open the MuJoCo viewer (use mjpython on macOS)")
    args = parser.parse_args()
    tasks = list(TASKS) if args.task == "all" else [args.task]
    rates = {t: run(t, args.episodes, args.seed, args.render, args.max_seconds) for t in tasks}
    if len(rates) > 1:
        print("  ".join(f"{t}={r:.0%}" for t, r in rates.items()))


if __name__ == "__main__":
    main()
