import gymnasium as gym
import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import g1_manip  # noqa: F401
from g1_manip import arena

pytestmark = pytest.mark.skipif(not arena.ARENA_XML.exists(), reason="arena not built (scripts/build_arena.py)")

ARENA_ENVS = ["G1Arena-v0", "G1ArenaKeypad-v0", "G1ArenaButtonPress-v0", "G1ArenaValve-v0"]


@pytest.fixture(scope="module")
def arena_model():
    model = mujoco.MjModel.from_xml_path(str(arena.ARENA_XML))
    return model, mujoco.MjData(model)


def test_arena_has_all_fixtures(arena_model):
    model, _ = arena_model
    mech = arena.ArenaMechanisms(model)
    assert sorted(b for b in mech.buttons if b.startswith("keypad_")) == sorted(f"keypad_{d}" for d in "0123456789")
    assert len([b for b in mech.buttons if b.startswith("board_")]) == 10
    assert len(mech.valves) == 10
    assert len(mech.bulbs) == 10
    assert mech.has_door


def test_keypad_digit_layout(arena_model):
    """1-2-3 on the top row left to right as seen from the start side, 0 alone at the bottom."""
    model, data = arena_model
    mujoco.mj_forward(model, data)
    pos = {d: data.xpos[model.body(f"keypad_{d}").id] for d in "0123456789"}
    assert pos["1"][2] > pos["4"][2] > pos["7"][2] > pos["0"][2]
    # Facing the keypad from +x, the viewer's right is +y.
    assert pos["1"][1] < pos["2"][1] < pos["3"][1]


def test_stations_stand_on_platform():
    for name, st in arena.stations().items():
        assert 0.05 < st.floor_z < 0.25, name


def test_door_passage_is_clear(arena_model):
    """With the door open, a ray through the doorway at chest height hits nothing."""
    model, data = arena_model
    data.qpos[model.jnt_qposadr[model.joint("door_hinge").id]] = np.deg2rad(95)
    mujoco.mj_forward(model, data)
    hinge = np.array(arena.load_meta()["door"]["hinge_pos"])
    geomid = np.zeros(1, np.int32)
    for z in (0.5, 1.0, 1.5):
        start = np.array([hinge[0] + 0.6, hinge[1] + 0.5, z])
        dist = mujoco.mj_ray(model, data, start, np.array([-1.0, 0, 0]), np.array([0, 0, 0, 1, 0, 0], np.uint8), 1, -1, geomid)
        assert dist < 0 or dist > 1.2, f"doorway blocked at z={z} by geom {geomid[0]}"


@pytest.mark.parametrize("env_id", ARENA_ENVS)
def test_arena_env_api(env_id):
    env = gym.make(env_id)
    check_env(env.unwrapped, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    env.close()


def test_button_press_lights_up():
    env = gym.make("G1ArenaButtonPress-v0").unwrapped
    env.reset(seed=0)
    target = env.target
    body = env.model.body(target).id
    axis = env.data.xmat[body].reshape(3, 3)[:, 2]
    env.data.xfrc_applied[body, :3] = 8.0 * axis
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
    assert env.mechanisms.is_pressed(env.data, target)
    assert info["is_success"] and terminated
    lit = env.mechanisms._cap_geoms[target]
    assert np.allclose(env.model.geom_rgba[lit[0]], arena.LIT_RGBA)


def test_valve_turns_and_stops_at_90_degrees():
    env = gym.make("G1ArenaValve-v0").unwrapped
    env.reset(seed=0)
    body = env.model.body(env.target).id
    axis = env.data.xmat[body].reshape(3, 3)[:, 2]
    for _ in range(50):
        env.data.xfrc_applied[body, 3:] = 2.0 * axis
        mujoco.mj_step(env.model, env.data, 10)
    angle = env.mechanisms.valve_angle(env.data, env.target)
    assert np.isclose(angle, np.pi / 2, atol=0.05)


def test_bulb_thread_couples_turn_and_depth():
    env = gym.make("G1Arena-v0", station="bulb_wall").unwrapped
    env.reset(seed=0)
    bulb = env.station.fixtures[0]
    env.mechanisms.unscrew_bulb(env.data, bulb, 2.0)
    mujoco.mj_forward(env.model, env.data)
    body = env.model.body(bulb).id
    axis = env.data.xmat[body].reshape(3, 3)[:, 2]
    for _ in range(200):
        env.data.xfrc_applied[body, 3:] = 0.3 * axis  # tighten
        mujoco.mj_step(env.model, env.data, 10)
    assert env.mechanisms.bulb_turns(env.data, bulb) < 0.05
    slide = env.data.qpos[env.model.jnt_qposadr[env.model.joint(f"{bulb}_slide").id]]
    assert abs(slide) < 1e-3


def test_door_latch_requires_handle():
    env = gym.make("G1Arena-v0", station="door").unwrapped
    env.reset(seed=0)
    m, d = env.model, env.data
    door = m.body("door").id
    push = np.array([-40.0, 0, 0])
    for _ in range(25):
        d.xfrc_applied[door, :3] = push
        mujoco.mj_step(m, d, 10)
        env.mechanisms.update(d)
    assert env.mechanisms.door_angle(d) < 0.03 and env.mechanisms.door_latched(d)

    handle = m.body("door_handle_start").id
    h_axis = d.xmat[handle].reshape(3, 3)[:, 2]
    for _ in range(100):
        d.xfrc_applied[handle, 3:] = 1.5 * h_axis
        d.xfrc_applied[door, :3] = push
        mujoco.mj_step(m, d, 10)
        env.mechanisms.update(d)
    assert env.mechanisms.door_angle(d) > np.deg2rad(30)

    d.xfrc_applied[:] = 0
    for _ in range(400):
        mujoco.mj_step(m, d, 10)
        env.mechanisms.update(d)
    assert env.mechanisms.door_angle(d) < 0.05, "spring should close the door"
