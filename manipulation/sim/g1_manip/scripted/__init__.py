"""Scripted IK controllers for the manipulation-trial tasks (see ``base.py`` for the shared machinery)."""

from g1_manip.scripted.base import ScriptedIK
from g1_manip.scripted.buttons import ButtonSequenceIK
from g1_manip.scripted.valve import ValveTurnIK

# Wall fixtures: stand further back than the RL stations. With the fixtures ~0.3 m in front of the
# pelvis, a finger pointing straight in needs the arm folded to its elbow stop, and the swing to the
# first fixture sweeps the hand through the board.
WALL = {"joint_group": "whole_body", "station_offset": (-0.12, 0.0)}
# The scripted motions are slower than the RL episode limits allow.
SECONDS = 50  # control steps per second

# task name -> (env id, env kwargs, controller class)
TASKS = {
    "keypad": ("G1ArenaKeypad-v0", {**WALL, "max_episode_steps": 25 * SECONDS}, ButtonSequenceIK),
    # 4 cm right of the RL station: the hanging left hand otherwise touches the board frame's side rail.
    "button_wall": ("G1ArenaButtonPress-v0", {**WALL, "station_offset": (-0.12, -0.04), "sequence_length": 5, "max_episode_steps": 25 * SECONDS}, ButtonSequenceIK),
    "button_floor": ("G1ArenaButtonPressFloor-v0", {"sequence_length": 5, "max_episode_steps": 40 * SECONDS}, ButtonSequenceIK),
    "valve_wall": ("G1ArenaValve-v0", {**WALL, "max_episode_steps": 20 * SECONDS}, ValveTurnIK),
    "valve_floor": ("G1ArenaValveFloor-v0", {"max_episode_steps": 25 * SECONDS}, ValveTurnIK),
}

__all__ = ["ScriptedIK", "ButtonSequenceIK", "ValveTurnIK", "TASKS"]
