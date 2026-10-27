"""The competition testbed (converted from the official CAD) as a MuJoCo scene.

``scripts/build_arena.py`` generates ``assets/arena/arena.xml``. This module
attaches it to a robot spec, places the robot at a task station, and runs the
arena's discrete mechanisms (button lights, door latch) during simulation.

Arena frame: the CAD's own frame in meters. The start location is at the +x end
(x ~ 11.6 m); the floor platforms' top surface is ~0.13 m above the ground plane.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

ARENA_DIR = Path(os.environ.get("G1_ARENA_DIR", Path(__file__).parent / "assets" / "arena"))
ARENA_XML = ARENA_DIR / "arena.xml"
ARENA_META = ARENA_DIR / "arena_meta.json"

PRESS_FRACTION = 0.66  # of full travel, counts as pressed (lights up)
LIT_RGBA = np.array([1.0, 0.25, 0.1, 1.0])


@dataclass(frozen=True)
class Station:
    """Robot base pose (pelvis xy, floor height, yaw) facing a group of fixtures."""

    x: float
    y: float
    floor_z: float
    yaw: float
    fixtures: tuple[str, ...]


# Fixture groups and how the robot stands in front of them. `standoff` is the
# horizontal pelvis-to-fixture distance; `lateral` shifts the robot to its left
# so the fixtures sit in front of the right arm.
STATION_SPECS = {
    "keypad": dict(prefix="keypad_", standoff=0.30, lateral=0.18),
    "door": dict(prefix="door_handle_start", standoff=0.42, lateral=0.15),
    "bulb_wall": dict(prefix="bulb_wall_", standoff=0.30, lateral=0.18),
    "bulb_floor": dict(prefix="bulb_floor_", standoff=0.40, lateral=0.18),
    "valve_wall": dict(prefix="valve_wall_", standoff=0.30, lateral=0.18),
    "valve_floor": dict(prefix="valve_floor_", standoff=0.40, lateral=0.18),
    "button_wall": dict(prefix="board_wall_", standoff=0.28, lateral=0.18),
    "button_floor": dict(prefix="board_floor_", standoff=0.40, lateral=0.18),
}


def require_arena() -> None:
    if not ARENA_XML.exists():
        raise FileNotFoundError(f"Arena not found at {ARENA_XML}. Run scripts/build_arena.py first.")


def load_meta() -> dict:
    require_arena()
    return json.loads(ARENA_META.read_text())


def load_arena_spec(include_terrain: bool = False) -> mujoco.MjSpec:
    require_arena()
    spec = mujoco.MjSpec.from_file(str(ARENA_XML))
    # Absolute mesh paths so the spec can be attached to a model loaded from elsewhere.
    for mesh in spec.meshes:
        mesh.file = str(ARENA_DIR / "meshes" / Path(mesh.file).name)
    spec.meshdir = ""
    if not include_terrain:
        spec.delete(spec.body("terrain"))
    return spec


def attach_arena(spec: mujoco.MjSpec, include_terrain: bool = False) -> None:
    spec.attach(load_arena_spec(include_terrain), prefix="", frame=spec.worldbody.add_frame())


def outward_normal(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    """Direction a fixture faces (toward the robot), from its body frame convention."""
    z = data.xmat[model.body(body).id].reshape(3, 3)[:, 2]
    # Buttons and bulbs have z pointing into the board; valves and door handles point outward.
    return z if body.startswith("valve_") else -z


def compute_stations(model: mujoco.MjModel) -> dict[str, Station]:
    """Derives robot standing poses from a compiled arena model."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bodies = [model.body(i).name for i in range(model.nbody)]
    stations = {}
    for name, cfg in STATION_SPECS.items():
        fixtures = tuple(b for b in bodies if b.startswith(cfg["prefix"]))
        if not fixtures:
            continue
        center = np.mean([data.xpos[model.body(b).id] for b in fixtures], axis=0)
        if name == "door":
            normal = np.array([1.0, 0.0, 0.0])  # handle side faces the start location
        else:
            normal = np.mean([outward_normal(model, data, b) for b in fixtures], axis=0)
        n = np.array([normal[0], normal[1], 0.0])
        n /= np.linalg.norm(n)
        yaw = float(np.arctan2(-n[1], -n[0]))
        left = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        xy = center + n * cfg["standoff"] + left * cfg["lateral"]
        stations[name] = Station(float(xy[0]), float(xy[1]), floor_height(model, data, xy), yaw, fixtures)
    return stations


def floor_height(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray) -> float:
    """Height of the platform under xy, ignoring mobility terrain (absent in manipulation trials)."""
    geomid = np.zeros(1, np.int32)
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
    dist = mujoco.mj_ray(model, data, np.array([xy[0], xy[1], 3.0]), np.array([0.0, 0.0, -1.0]), None, 1, terrain, geomid)
    return 3.0 - dist if dist >= 0 else 0.0


def stations() -> dict[str, Station]:
    meta = load_meta()
    return {k: Station(**{**v, "fixtures": tuple(v["fixtures"])}) for k, v in meta["stations"].items()}


def disable_far_collisions(model: mujoco.MjModel, data: mujoco.MjData, center: np.ndarray, radius: float) -> None:
    """Turns off static arena collision geoms farther than `radius` from `center` (speed)."""
    mujoco.mj_forward(model, data)
    for g in range(model.ngeom):
        if model.geom_group[g] != 3 or model.body_weldid[model.geom_bodyid[g]] != 0:
            continue
        aabb_center = data.geom_xpos[g] + data.geom_xmat[g].reshape(3, 3) @ model.geom_aabb[g, :3]
        reach = np.linalg.norm(model.geom_aabb[g, 3:])
        if np.linalg.norm(aabb_center[:2] - center[:2]) - reach > radius:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0


class ArenaMechanisms:
    """State and discrete logic of the arena fixtures for one MjModel/MjData pair."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        names = [model.body(i).name for i in range(model.nbody)]
        self.buttons = [n for n in names if n.startswith(("keypad_", "board_", "mobility_"))]
        self.valves = [n for n in names if n.startswith("valve_")]
        self.bulbs = [n for n in names if n.startswith("bulb_")]

        self._qadr = {}
        for b in self.buttons + self.valves:
            self._qadr[b] = model.jnt_qposadr[model.joint(f"{b}_joint").id]
        for b in self.bulbs:
            self._qadr[b] = model.jnt_qposadr[model.joint(f"{b}_turn").id]
            self._qadr[f"{b}_slide"] = model.jnt_qposadr[model.joint(f"{b}_slide").id]
        self._travel = {b: model.jnt_range[model.joint(f"{b}_joint").id, 1] for b in self.buttons}
        self._cap_geoms = {
            b: np.flatnonzero((model.geom_bodyid == model.body(b).id) & (model.geom_group == 2)) for b in self.buttons
        }
        self._cap_rgba = {b: model.geom_rgba[g].copy() for b, g in self._cap_geoms.items()}

        self.has_door = "door" in names
        if self.has_door:
            self._door_q = model.jnt_qposadr[model.joint("door_hinge").id]
            self._latch = model.equality("door_latch").id
            self._handle_q = [
                model.jnt_qposadr[model.joint(f"{h}_joint").id]
                for h in ("door_handle_start", "door_handle_far")
                if f"{h}_joint" in [model.joint(j).name for j in range(model.njnt)]
            ]
            self._handle_release = 0.6 * model.jnt_range[model.joint("door_handle_start_joint").id, 1]

    # --- queries ------------------------------------------------------------

    def button_depth(self, data: mujoco.MjData, name: str) -> float:
        return float(data.qpos[self._qadr[name]] / self._travel[name])

    def is_pressed(self, data: mujoco.MjData, name: str) -> bool:
        return self.button_depth(data, name) >= PRESS_FRACTION

    def valve_angle(self, data: mujoco.MjData, name: str) -> float:
        return float(data.qpos[self._qadr[name]])

    def bulb_turns(self, data: mujoco.MjData, name: str) -> float:
        """Turns remaining until tight (0 = fully screwed in)."""
        return float(-data.qpos[self._qadr[name]] / (2 * np.pi))

    def door_angle(self, data: mujoco.MjData) -> float:
        return float(data.qpos[self._door_q])

    def door_latched(self, data: mujoco.MjData) -> bool:
        return bool(data.eq_active[self._latch])

    # --- state changes ------------------------------------------------------

    def unscrew_bulb(self, data: mujoco.MjData, name: str, turns: float) -> None:
        pitch = self.model.eq_data[self.model.equality(f"{name}_thread").id, 1] * 2 * np.pi
        data.qpos[self._qadr[name]] = -turns * 2 * np.pi
        data.qpos[self._qadr[f"{name}_slide"]] = -turns * pitch

    def update(self, data: mujoco.MjData) -> None:
        """Call after each physics step: lights pressed buttons, operates the door latch."""
        for b in self.buttons:
            rgba = LIT_RGBA if self.is_pressed(data, b) else None
            for g in self._cap_geoms[b]:
                self.model.geom_rgba[g] = rgba if rgba is not None else self._cap_rgba[b][list(self._cap_geoms[b]).index(g)]
        if self.has_door:
            handle_turned = any(data.qpos[q] > self._handle_release for q in self._handle_q)
            closed = abs(data.qpos[self._door_q]) < 0.02
            if handle_turned:
                data.eq_active[self._latch] = 0
            elif closed:
                data.eq_active[self._latch] = 1
