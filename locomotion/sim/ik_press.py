"""Press the button in G1ButtonPressFloating-v0 with the IK controller and report success rate.

    python ik_press.py --episodes 20
    mjpython ik_press.py --render          # macOS viewer
"""

import argparse
import time

import gymnasium as gym
import numpy as np

import g1_loco  # noqa: F401  (registers envs)
from g1_loco.button_ik import ButtonPressIK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="G1ButtonPressFloating-v0")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    # The RL default (0.75 rad around stand) can't reach chest-height buttons; IK needs the full joint range.
    parser.add_argument("--action-scale", type=float, default=3.0)
    parser.add_argument("--joint-group", default="right_arm_waist", choices=["right_arm_waist", "right_arm"])
    parser.add_argument("--render", action="store_true", help="open the MuJoCo viewer (use mjpython on macOS)")
    args = parser.parse_args()

    env = gym.make(
        args.env,
        joint_group=args.joint_group,
        action_scale=args.action_scale,
        render_mode="human" if args.render else None,
    )
    controller = None
    successes, press_times = [], []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        if controller is None:
            controller = ButtonPressIK(env.unwrapped)
        else:
            controller.reset()
        dt = env.unwrapped.dt
        next_step = time.perf_counter()
        done, info, steps, ik_err = False, {}, 0, 0.0
        while not done:
            _, _, terminated, truncated, info = env.step(controller.act())
            ik_err = max(ik_err, controller.last_result.pos_err)  # approach solve, then each press step
            steps += 1
            done = terminated or truncated
            if args.render:  # pace to wall-clock time
                next_step += dt
                while time.perf_counter() < next_step:  # poll; sleep() oversleeps and lags the viewer
                    pass
        success = bool(info.get("is_success", False))
        successes.append(success)
        if success:
            press_times.append(steps * dt)
        print(
            f"episode {ep:3d}  button={np.round(controller.face, 3)}  success={success}  t={steps * dt:4.2f}s  "
            f"depth={info['button_depth'] * 1000:4.1f}mm  max_ik_err={ik_err * 1000:.2f}mm  fell={info.get('fell', False)}"
        )
    print(f"success rate: {np.mean(successes):.0%}  mean time to press: {np.mean(press_times) if press_times else np.nan:.2f}s")
    env.close()


if __name__ == "__main__":
    main()
