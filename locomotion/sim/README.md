# g1_loco

MuJoCo sim for the mobility-course buttons. `G1ButtonPressFloating-v0` is the same task as in
`manipulation/sim` (free-floating G1, spring-loaded button at a random spot in front of the robot),
pressed here by an IK controller instead of a learned policy.

## Setup

```bash
cd locomotion/sim
uv venv && uv pip install -e ".[dev]"
scripts/fetch_g1.sh          # G1 model from MuJoCo Menagerie
```

## Run

```bash
python ik_press.py --episodes 20
mjpython ik_press.py --render        # macOS viewer
pytest
```

## How it works

- `g1_loco/ik.py`: `SiteIK`, damped least-squares IK for a site over a set of hinge joints. Position is
  the primary task. Pointing a site axis along a direction is secondary, so it never costs position
  accuracy. A rest posture fills the remaining null space. Joints that hit their limits are locked
  out of the step. The solver runs on its own `MjData` at the robot's current base pose.
- `g1_loco/button_ik.py`: `ButtonPressIK`. It solves once for the pose that puts the index fingertip
  6 cm in front of the button, pointing into the board, and swings the arm there in joint space.
  Then it presses straight in along the board normal, re-solving IK every step against the swaying
  floating base. Joint targets go through the env's normal action interface (`env.action_for`).
- The button pose is read from the simulator. On the robot it would come from perception.

Notes:

- `ik_press.py` builds the env with `action_scale=3.0`. The RL default (±0.75 rad around stand)
  can't reach chest-height buttons.
- The default `joint_group="right_arm_waist"` lets the torso turn and lean slightly. It succeeds on
  198/200 seeds. `right_arm` alone succeeds on 97/100.
- The IK doesn't model collisions. The remaining failures are buttons that are both close and near
  the centerline (x ≈ 0.27 m, |y| < 0.08 m), where the wrist ends up against the torso.
