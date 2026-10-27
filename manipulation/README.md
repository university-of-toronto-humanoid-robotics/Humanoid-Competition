# Manipulation

Work for the **Manipulation Trials** of the 2026 IEEE Humanoids Loco-Manipulation Challenge (Santa Clara, Dec 6–9, 2026), organized by NIST.

## Trial format

- Obstacles and rough terrain are removed. The robot only walks on flat ground (wheeled humanoids are fine).
- The robot walks from the **starting location** through tasks 1 → 5 in order to the **ending location**, then does them again in reverse (5 → 1) back to the start. That is **one cycle**. Cycles repeat until time runs out.
- Trials last about **10–20 min**. The goal is reliability: as many clean repetitions as possible, not speed.
- Each completed task scores points. Doing **all** tasks in order earns bonus points. Skipping a task, or failing it and moving on, loses the bonus.
- The lightbulb, valve, and button boards are mounted **on the wall and at floor level**, so the robot has to bend down. All mounts must be done for full points.
- Higher autonomy scores higher. Declare the autonomy level for each task. Full autonomy means no operator input from start to end. The apparatus can be pre-mapped on prep day.

## Tasks and strategies

Fill in the strategy fields for each task.

### 1. Keypad (manipulation with visual input)

**Rules:** A 5-digit code appears on a screen. Press the matching buttons on the keypad (digits 0–9) in order. A button lights up when it has been pressed hard enough.

**High-level strategy**

- Approach / base placement:
- Perception (reading the code, locating the keys):
- Arm / hand motion (finger pose, press depth, retract):
- Verification (light feedback, retry logic):
- Autonomy level:
- Known failure modes and mitigations:
- Sim / training plan:

### 2. Door traversal

**Rules:** The door is 3 ft × 7 ft with a standard lever handle. Turn the handle, push the door open, and hold it open while walking through. Torsion springs in the hinges pull the door shut, and the robot must let it close safely.

**High-level strategy**

- Approach / base placement:
- Perception (handle pose, door state):
- Handle turn + push sequence:
- Holding the door while walking through (which arm / body part):
- Safe closing:
- Autonomy level:
- Known failure modes and mitigations:
- Sim / training plan:

### 3. Lightbulb

**Rules:** Ordinary light fixtures are mounted on a board, with 3D-printed bulbs. Some bulbs only need tightening in place. Others have to be picked up from a box and screwed in. The board has wall-height and floor-level positions.

**High-level strategy**

- Approach / base placement (wall vs. floor fixtures):
- Perception (bulb / socket pose):
- Grasp (finger configuration on the bulb):
- Screwing motion (wrist roll range, regrasp cycle, done detection):
- Pick-from-box sequence:
- Autonomy level:
- Known failure modes and mitigations:
- Sim / training plan:

### 4. Valve turning

**Rules:** Standard 90° lever-handle ball valves are mounted in several orientations, at wall height and at floor level. Turn each one from perpendicular to parallel.

**High-level strategy**

- Approach / base placement (wall vs. floor valves):
- Perception (handle orientation, current state):
- Contact strategy (grasp vs. push the lever):
- Handling the different mount orientations:
- Autonomy level:
- Known failure modes and mitigations:
- Sim / training plan:

### 5. Button pressing

**Rules:** The same push buttons as the keypad are mounted on a board, at wall height and at floor level. Press them.

**High-level strategy**

- Approach / base placement (wall vs. floor buttons):
- Perception (button locations):
- Press motion (finger pose, press depth):
- Bending down for floor buttons (waist / knee strategy, balance):
- Autonomy level:
- Known failure modes and mitigations:
- Sim / training plan: start from `G1ButtonPress-v0` (see below)

### Loco-manipulation (phase 3, shared with [locomotion](../locomotion/README.md))

**Rules:** Obstacles and terrain are back in. Order: keypad → door → first mobility button → lightbulb → valve → button board → pick up a crate from the counter → climb the stairs and press the landing button → come down and walk back across the terrain to the start **while carrying the crate**, then put it on the ground. That is one cycle.

**High-level strategy**

- Crate grasp / carry posture:
- Pressing buttons while carrying the crate:
- Integration with locomotion policy:
- Autonomy level:
- Known failure modes and mitigations:

## Simulation: [`sim/`](sim/)

MuJoCo + Gymnasium environments for training Unitree G1 (29-DoF body, Dex3 3-finger hands) manipulation policies with RL. The G1 model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1).

### Setup

```bash
cd manipulation/sim
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[train,dev]"
./scripts/fetch_g1.sh        # downloads the G1 model (~34 MB, gitignored)
uv pip install -e ".[cad]" && python scripts/build_arena.py   # competition arena from the official CAD (~4 min)
python -m pytest tests                 # sanity checks (arena tests skip if the arena isn't built)
```

### Train / evaluate

```bash
python train.py --env G1ButtonPress-v0 --n-envs 8 --timesteps 2000000
tensorboard --logdir runs
python play.py --env G1ButtonPress-v0 --run runs/G1ButtonPress-v0                   # success rate
mjpython play.py --env G1ButtonPress-v0 --run runs/G1ButtonPress-v0 --render        # viewer (macOS needs mjpython)
python play.py --env G1ButtonPress-v0                                                # random actions, no policy
```

### Environments

| ID | Base | Description |
|---|---|---|
| `G1ButtonPress-v0` | fixed (pelvis welded at standing height) | Press a spring-loaded button at a random spot in front of the robot with the right index finger. |
| `G1ButtonPressFloating-v0` | floating | Same task, but the robot is free and legs hold the standing pose with PD. Episode ends if the pelvis drops below 0.5 m. |

- **Control:** 50 Hz policy (physics at 500 Hz, `frame_skip=10`). Actions in `[-1, 1]` are offsets from the standing pose (`action_scale` rad) and are sent to the G1's position servos. Joints outside the controlled group hold the standing pose.
- **Joint groups** (`joint_group=`, defined in [robot.py](sim/g1_manip/robot.py)): `right_arm` (default, 14), `right_arm_waist`, `both_arms`, `upper_body`, `whole_body`.
- **Observation:** controlled joint positions and velocities, the previous action, and (floating base only) gravity in the pelvis frame plus base angular velocity. Task terms come last: fingertip position, fingertip-to-target vector, and button depth.
- **Reward (button press):** reach shaping `1 - tanh(5·d)`, plus press depth, plus +10 on success (terminates). Minus an action-rate penalty.
- Override any constructor arg: `gym.make("G1ButtonPress-v0", joint_group="right_arm_waist", target_range=((0.28, 0.38), (-0.35, -0.05), (0.3, 0.6)))`.

### Competition arena (official CAD)

[`scripts/build_arena.py`](sim/scripts/build_arena.py) converts the organizers' testbed model ([`CAD-Files/CompetitionTestbed.dae`](https://github.com/peterso/ieee-humanoids-2026-loco-manipulation-challenge/tree/main/CAD-Files), pinned commit) into `g1_manip/assets/arena/arena.xml`. Building takes about 4 minutes, and the output is gitignored.

```bash
uv pip install -e ".[cad]"
python scripts/build_arena.py
python -m mujoco.viewer --mjcf g1_manip/assets/arena/arena.xml    # browse the arena (mjpython on macOS)
```

The CAD stays in its own frame (meters). The start location is at the +x end (x ≈ 11.6 m). Platforms are 0.13 m above the ground.

**What moves:**

| Fixture | Bodies | Mechanism |
|---|---|---|
| Keypad (task 1) | `keypad_0` … `keypad_9` | Slide joint, 6 mm travel, about 3.6 N spring. A button lights up past 2/3 travel. |
| Door (task 2) | `door`, `door_handle_start`, `door_handle_far` | Sprung hinge with 0–100° of swing. Latched until either lever handle turns about 27°. Re-latches when the door closes. |
| Lightbulbs (task 3) | `bulb_wall_0..4`, `bulb_floor_0..4` | Hinge coupled to a slide (E26 thread pitch). `ArenaMechanisms.unscrew_bulb()` sets the "needs tightening" state. |
| Valves (task 4) | `valve_wall_0..4`, `valve_floor_0..4` | Lever on a ±90° hinge with 0.4 N·m friction. |
| Buttons (task 5) | `board_wall_0..4`, `board_floor_0..4` | Same mechanism as the keypad. |
| Mobility buttons | `mobility_1`, `_2`, `_3`, `_5` | Same mechanism (button 4 is missing from this CAD revision). |

Everything else is static, with collision shapes that are boxes, convex hulls, or CoACD decompositions. The ramps and 4×4 stepover beams go in a separate `terrain` body that is removed by default (`include_terrain=True` keeps them). The gantry, human figure, and spare crates are left out.

**Stations:** `station=` places the robot facing a fixture group: `keypad`, `door`, `bulb_wall`, `bulb_floor`, `valve_wall`, `valve_floor`, `button_wall`, `button_floor`. With the fixed base, the right hand can reach every wall-height fixture. Floor-level boards are 0.3 m or more out of reach and need `fixed_base=False` with a whole-body joint group (squatting).

| ID | Task |
|---|---|
| `G1Arena-v0` | No reward; stand at any station (`gym.make("G1Arena-v0", station="door")`). Use it to build new tasks or for teleop. |
| `G1ArenaKeypad-v0` | Press a random 5-digit code in order. The policy is given the current target digit. The display mesh always reads 13592. |
| `G1ArenaButtonPress-v0` | Press a random button on the wall board. |
| `G1ArenaValve-v0` | Turn a random wall valve 90°. |

**Known gaps:**
- The keypad display is static geometry.
- The organizers say the CAD is not final and has no real friction or mass data.
- The CAD shows the door open toward the start side. I made it swing away from the start, because the rules say the door is pushed open. Flip the hinge range in `build_arena.py` if the final apparatus differs.

### Adding a task

Subclass [`G1ManipEnv`](sim/g1_manip/envs/base.py) and implement:

- `_build_task(spec)`: add bodies, joints, and sites to the `MjSpec` before it compiles.
- `_reset_task()`: randomize the task.
- `_task_obs()`: return task observations.
- `_task_reward(action)`: return `(reward, terminated, info)`.

Then register the env in [`g1_manip/__init__.py`](sim/g1_manip/__init__.py). [`button_press.py`](sim/g1_manip/envs/button_press.py) is the reference example for a flat-scene task, and [`arena_tasks.py`](sim/g1_manip/envs/arena_tasks.py) for an arena task, where `self.station` and `self.mechanisms` give the fixtures and their state.
