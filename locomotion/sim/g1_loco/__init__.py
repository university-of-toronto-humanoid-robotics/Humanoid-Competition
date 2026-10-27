"""MuJoCo + Gymnasium environments for the Unitree G1 (locomotion trials), with IK-based controllers."""

from gymnasium.envs.registration import register

# Same task as g1_manip's G1ButtonPressFloating-v0: free-floating G1 pressing a button
# (the mobility course marks each checkpoint with one).
register(
    id="G1ButtonPressFloating-v0",
    entry_point="g1_loco.envs:G1ButtonPressEnv",
    max_episode_steps=250,  # 5 s at 50 Hz control
    kwargs={"fixed_base": False},
)
