"""MuJoCo + Gymnasium environments for training Unitree G1 manipulation policies."""

from gymnasium.envs.registration import register

register(
    id="G1ButtonPress-v0",
    entry_point="g1_manip.envs:G1ButtonPressEnv",
    max_episode_steps=250,  # 5 s at 50 Hz control
)
register(
    id="G1ButtonPressFloating-v0",
    entry_point="g1_manip.envs:G1ButtonPressEnv",
    max_episode_steps=250,
    kwargs={"fixed_base": False},
)

# Competition arena (official CAD, built by scripts/build_arena.py).
register(
    id="G1Arena-v0",  # no task reward; pass station=... to stand at any fixture
    entry_point="g1_manip.envs:G1ManipEnv",
    max_episode_steps=500,
    kwargs={"station": "keypad"},
)
register(
    id="G1ArenaKeypad-v0",
    entry_point="g1_manip.envs:G1ArenaButtonsEnv",
    max_episode_steps=750,
    kwargs={"station": "keypad", "sequence_length": 5},
)
register(
    id="G1ArenaButtonPress-v0",
    entry_point="g1_manip.envs:G1ArenaButtonsEnv",
    max_episode_steps=250,
    kwargs={"station": "button_wall", "sequence_length": 1},
)
register(
    id="G1ArenaValve-v0",
    entry_point="g1_manip.envs:G1ArenaValveEnv",
    max_episode_steps=300,
    kwargs={"station": "valve_wall"},
)
