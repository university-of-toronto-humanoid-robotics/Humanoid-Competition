"""Train a PPO policy on a G1 manipulation env with Stable-Baselines3.

    python train.py --env G1ButtonPress-v0 --n-envs 8 --timesteps 2000000
    tensorboard --logdir runs
"""

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import g1_manip  # noqa: F401  (registers envs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="G1ButtonPress-v0")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="PPO with MLP policies is usually fastest on CPU")
    args = parser.parse_args()

    run_dir = Path("runs") / (args.run_name or args.env)
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(args.env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=512,
        batch_size=1024,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]), log_std_init=-1.0),
        tensorboard_log=str(run_dir),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    checkpoint = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(run_dir / "policy")
    venv.save(str(run_dir / "vecnormalize.pkl"))
    print(f"Saved policy and normalization stats to {run_dir}")


if __name__ == "__main__":
    main()
