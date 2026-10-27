"""Converts the official competition testbed CAD into a MuJoCo arena.

Source: CAD-Files/CompetitionTestbed.dae from
https://github.com/peterso/ieee-humanoids-2026-loco-manipulation-challenge
(SketchUp export, inches, Z-up), pinned to CAD_COMMIT below.

Output (g1_manip/assets/arena/, gitignored):
    arena.xml        standalone MJCF; open with `python -m mujoco.viewer --mjcf ...`
    meshes/*.stl     merged visual meshes and convex collision parts
    arena_meta.json  fixture names/axes used by g1_manip.arena

The CAD is kept in its own world frame (meters). The robot's start location is
at the +x end of the arena (x ~ 11.6 m), the stairs at the -x end.

What gets articulated:
    keypad + button-board + mobility push buttons  slide joint with return spring
    valve levers                                     hinge, +-90 deg, friction
    light bulbs                                      hinge coupled to a slide (screw thread)
    door leaf                                        sprung hinge, latched until a lever handle turns

Everything else in the arena is static. Pieces outside the arena (gantry,
human figure, spare crates) are dropped. Mobility terrain (ramps and 4x4
stepover beams) goes in a separate `terrain` body so loaders can remove it for
the manipulation trials.

    python scripts/build_arena.py            # needs the `cad` extra: pip install -e ".[cad]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import collada
import numpy as np
import trimesh
from collada import scene as cs
from scipy.spatial import ConvexHull

CAD_COMMIT = "10bd00d636e8fc593c7a430ed712e4f65b3ee5e9"
CAD_URL = (
    "https://raw.githubusercontent.com/peterso/ieee-humanoids-2026-loco-manipulation-challenge/"
    f"{CAD_COMMIT}/CAD-Files/CompetitionTestbed.dae"
)
INCH = 0.0254

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "g1_manip" / "assets" / "arena"

# Top-level SketchUp groups (at CAD_COMMIT) that are not part of the arena.
EXCLUDED_GROUPS = {"SketchUp_Instance_189", "SketchUp_Instance_205", "group_73", "SketchUp_Instance_209"}
# Pinwheel ramps and the 4x4 stepover beams; removed during manipulation trials.
TERRAIN_GROUPS = {"SketchUp_Instance_194", "SketchUp_Instance_199", "SketchUp_Instance_204", "group_142", "group_137"}
DOOR_LEAF_INSTANCE = "SketchUp/group_111/SketchUp_Instance_94"

# Fixture group prefixes -> semantic names.
KEYPAD = "SketchUp/group_38"
BUTTON_BOARD_WALL = "SketchUp/group_63/group_71"
BUTTON_BOARD_FLOOR = "SketchUp/group_63/group_64"
VALVE_WALL = "SketchUp/group_84/group_86"
VALVE_FLOOR = "SketchUp/group_84/group_89"
BULB_WALL = "SketchUp/group_94/group_95"
BULB_FLOOR = "SketchUp/group_94/group_103"
# Mobility-course buttons, numbered as in the rules (button 4 is not in this CAD revision).
MOBILITY_BUTTONS = {
    "SketchUp/SketchUp_Instance_192": "mobility_1",
    "SketchUp/SketchUp_Instance_191": "mobility_2",
    "SketchUp/SketchUp_Instance_193": "mobility_3",
    "SketchUp/SketchUp_Instance_188": "mobility_5",
}

# Textured SketchUp materials have no flat color; approximate them by name.
MATERIAL_COLORS = [
    (r"osb", (0.80, 0.66, 0.46)),
    (r"bamboo", (0.86, 0.74, 0.54)),
    (r"veneer", (0.78, 0.62, 0.42)),
    (r"maple", (0.88, 0.78, 0.62)),
    (r"cherry", (0.62, 0.36, 0.24)),
    (r"oak|willow", (0.70, 0.52, 0.36)),
    (r"cork", (0.72, 0.56, 0.40)),
    (r"marble", (0.90, 0.90, 0.88)),
    (r"metal|fencing|brick", (0.62, 0.63, 0.65)),
    (r"plastic|denim", (0.25, 0.55, 0.30)),
]
DEFAULT_COLOR = (0.75, 0.65, 0.50)

# Mechanism parameters.
BUTTON_TRAVEL = 0.006  # m
BUTTON_STIFFNESS = 600.0  # N/m  (~3.6 N at full travel)
VALVE_RANGE = np.pi / 2
VALVE_FRICTION = 0.4  # N*m
BULB_PITCH = 1.0 / 7.0 * INCH  # E26 thread, m per turn
BULB_TURNS = 4.0  # max unscrew travel in turns
DOOR_STIFFNESS = 8.0  # N*m/rad torsion spring
DOOR_DAMPING = 4.0
DOOR_RANGE = np.deg2rad(100)
HANDLE_RANGE = np.deg2rad(45)
HANDLE_STIFFNESS = 0.6


# ----------------------------------------------------------------------------
# Loading


def download(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"CompetitionTestbed_{CAD_COMMIT[:7]}.dae"
    if not path.exists():
        print(f"Downloading {CAD_URL}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(CAD_URL, tmp)
        tmp.rename(path)
    return path


def load_pieces(dae_path: Path) -> list[dict]:
    """Flattens the Collada scene into world-space triangle pieces in meters."""
    doc = collada.Collada(
        str(dae_path), ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError]
    )
    pieces = []

    def walk(node, M, path):
        if isinstance(node, cs.Node):
            M = M @ node.matrix
            path = path + [node.xmlnode.get("name") or node.id or ""]
            for c in node.children:
                walk(c, M, path)
        elif isinstance(node, cs.NodeNode):
            walk(node.node, M, path)
        elif isinstance(node, cs.GeometryNode):
            matmap = {m.symbol: m.target for m in node.materials}
            for prim in node.geometry.primitives:
                if "Line" in type(prim).__name__:
                    continue  # SketchUp edge lines
                tris = prim.triangleset() if hasattr(prim, "triangleset") else prim
                if tris.vertex_index is None or len(tris.vertex_index) == 0:
                    continue
                v = (tris.vertex @ M[:3, :3].T + M[:3, 3]) * INCH
                mat = matmap.get(tris.material)
                pieces.append(
                    dict(
                        path="/".join(path),
                        mat=mat.name if mat is not None else "",
                        rgba=_material_rgba(mat),
                        verts=v.astype(np.float64),
                        faces=tris.vertex_index.astype(np.int64),
                    )
                )

    for n in doc.scene.nodes:
        walk(n, np.eye(4), [])
    return pieces


def _material_rgba(mat) -> tuple[float, ...]:
    if mat is not None:
        diffuse = getattr(mat.effect, "diffuse", None)
        if isinstance(diffuse, tuple):
            return tuple(float(c) for c in diffuse)
        name = mat.name.lower()
        for pattern, color in MATERIAL_COLORS:
            if re.search(pattern, name):
                return (*color, 1.0)
    return (*DEFAULT_COLOR, 1.0)


def top_group(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


# ----------------------------------------------------------------------------
# Geometry helpers


def mesh_of(pieces: list[dict]) -> trimesh.Trimesh:
    verts, faces, off = [], [], 0
    for p in pieces:
        verts.append(p["verts"])
        faces.append(p["faces"] + off)
        off += len(p["verts"])
    return trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces), process=True)


def excluding(pieces: list[dict], drop: list[dict]) -> list[dict]:
    ids = {id(p) for p in drop}
    return [p for p in pieces if id(p) not in ids]


def min_axis(points: np.ndarray) -> np.ndarray:
    """Direction of least variance (normal of a plate, axis of a disc)."""
    _, _, vt = np.linalg.svd(points - points.mean(0), full_matrices=False)
    return vt[2]


def max_axis(points: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(points - points.mean(0), full_matrices=False)
    return vt[0]


def quat_from_mat(R: np.ndarray) -> list[float]:
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return (q / np.linalg.norm(q)).tolist()


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in np.atleast_1d(v))


def _coacd_worker(verts: np.ndarray, faces: np.ndarray):
    import coacd

    coacd.set_log_level("error")
    return coacd.run_coacd(coacd.Mesh(verts, faces), threshold=0.05, max_convex_hull=24)


class CollisionBuilder:
    """Chooses a collision representation per piece: box, convex hull, or CoACD parts."""

    def __init__(self, mesh_dir: Path, tol: float = 0.003, min_size: float = 0.01, decompose_size: float = 0.15):
        self.mesh_dir = mesh_dir
        self.tol = tol
        self.min_size = min_size
        self.decompose_size = decompose_size  # smaller non-convex pieces just use their hull
        self.stats = defaultdict(int)
        self._pool = None

    def shapes(self, mesh: trimesh.Trimesh, name: str, frame_T: np.ndarray | None = None, split: bool = True) -> list[dict]:
        """Returns geoms (in the frame given by world->local transform frame_T)."""
        if frame_T is not None:
            mesh = mesh.copy()
            mesh.apply_transform(frame_T)
        if mesh.extents.max() < self.min_size or len(mesh.faces) == 0:
            self.stats["skipped"] += 1
            return []
        # SketchUp often stores several disjoint boards as one primitive. Curved
        # surfaces come with unshared vertices, so weld them first and only split
        # into a handful of genuine parts.
        mesh = mesh.copy()
        mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=5)
        parts = mesh.split(only_watertight=False) if split else []
        if 1 < len(parts) <= 12:
            return [g for i, part in enumerate(parts) for g in self.shapes(part, f"{name}_c{i}", split=False)]
        samples, _ = trimesh.sample.sample_surface(mesh, 600, seed=0)
        samples = np.concatenate([samples, mesh.vertices])

        try:
            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        except Exception:  # degenerate (collinear) sliver: axis-aligned box
            to_origin = trimesh.transformations.translation_matrix(-mesh.bounds.mean(0))
            extents = mesh.extents
        local = trimesh.transform_points(samples, to_origin)
        half = np.maximum(extents / 2, 0.0)
        gap = np.min(half[None] - np.abs(local), axis=1)
        if np.mean(gap < self.tol) > 0.98:
            self.stats["box"] += 1
            T = np.linalg.inv(to_origin)
            return [dict(type="box", size=np.maximum(half, 0.003), pos=T[:3, 3], quat=quat_from_mat(T[:3, :3]))]

        try:
            hull = ConvexHull(mesh.vertices)
            dist = np.min(-(samples @ hull.equations[:, :3].T + hull.equations[:, 3]), axis=1)
            convex = np.mean(dist < self.tol) > 0.98
        except Exception:
            convex = True  # degenerate (flat) -> hull of a sheet, fine
        convex = convex or mesh.extents.max() < self.decompose_size
        if convex:
            self.stats["hull"] += 1
            return [self._mesh_geom(mesh.vertices, mesh.faces, name)]

        parts = self._decompose(mesh)
        if parts is None:
            self.stats["hull_fallback"] += 1
            print(f"  warning: convex decomposition failed for {name} (extent {mesh.extents.round(2)}), using its hull")
            return [self._mesh_geom(mesh.vertices, mesh.faces, name)]
        self.stats["coacd"] += 1
        return [self._mesh_geom(v, f, f"{name}_p{i}") for i, (v, f) in enumerate(parts)]

    def _decompose(self, mesh: trimesh.Trimesh):
        # CoACD can abort the process on degenerate input, so run it in a worker.
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
        try:
            return self._pool.submit(_coacd_worker, mesh.vertices, mesh.faces).result(timeout=120)
        except Exception:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            return None

    def _mesh_geom(self, verts, faces, name) -> dict:
        trimesh.Trimesh(verts, faces).export(self.mesh_dir / f"{name}.stl")
        return dict(type="mesh", mesh=name)


# ----------------------------------------------------------------------------
# MJCF writing


class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.mesh_dir = out / "meshes"
        self.root = ET.Element("mujoco", model="competition_arena")
        ET.SubElement(self.root, "compiler", angle="radian", meshdir="meshes")
        ET.SubElement(self.root, "option", timestep="0.002", integrator="implicitfast")
        default = ET.SubElement(self.root, "default")
        arena = ET.SubElement(default, "default", {"class": "arena"})
        ET.SubElement(arena, "mesh", inertia="shell")
        vis = ET.SubElement(arena, "default", {"class": "arena_visual"})
        ET.SubElement(vis, "geom", group="2", contype="0", conaffinity="0", density="0")
        col = ET.SubElement(arena, "default", {"class": "arena_collision"})
        ET.SubElement(col, "geom", group="3", rgba="0.5 0.3 0.8 0.4", friction="0.8 0.02 0.001", density="0")
        self.asset = ET.SubElement(self.root, "asset")
        self.world = ET.SubElement(self.root, "worldbody")
        self.equality = ET.SubElement(self.root, "equality")
        self.contact = ET.SubElement(self.root, "contact")
        self.n_mesh = 0

    def add_mesh_asset(self, name: str) -> None:
        ET.SubElement(self.asset, "mesh", name=name, file=f"{name}.stl", **{"class": "arena"})

    def visual(self, parent, pieces: list[dict], prefix: str, frame_T: np.ndarray | None = None) -> None:
        """Adds visual meshes for pieces, merged by color."""
        by_color = defaultdict(list)
        for p in pieces:
            by_color[tuple(np.round(p["rgba"], 3))].append(p)
        for i, (rgba, group) in enumerate(by_color.items()):
            mesh = mesh_of(group)
            if frame_T is not None:
                mesh.apply_transform(frame_T)
            if len(mesh.faces) == 0:
                continue
            name = f"{prefix}_vis{i}"
            mesh.export(self.mesh_dir / f"{name}.stl")
            self.add_mesh_asset(name)
            ET.SubElement(parent, "geom", mesh=name, type="mesh", rgba=fmt(rgba), **{"class": "arena_visual"})

    def collision(self, parent, shapes: list[dict], name: str | None = None) -> list[str]:
        names = []
        for i, s in enumerate(shapes):
            attrs = {"class": "arena_collision"}
            if name is not None:
                attrs["name"] = f"{name}_col{i}"
                names.append(attrs["name"])
            if s["type"] == "box":
                attrs.update(type="box", size=fmt(s["size"]), pos=fmt(s["pos"]), quat=fmt(s["quat"]))
            else:
                self.add_mesh_asset(s["mesh"])
                attrs.update(type="mesh", mesh=s["mesh"])
            ET.SubElement(parent, "geom", attrs)
        return names

    def save(self) -> None:
        ET.indent(self.root)
        ET.ElementTree(self.root).write(self.out / "arena.xml")


def body_frame(origin: np.ndarray, z_axis: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Body frame at origin with (optionally) z aligned to z_axis. Returns (world->local T, pos, quat)."""
    R = np.eye(3)
    if z_axis is not None:
        z = z_axis / np.linalg.norm(z_axis)
        x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.9 else np.cross([0, 1, 0], z)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z], axis=1)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ origin
    return T, origin, quat_from_mat(R)


# ----------------------------------------------------------------------------
# Main


def build(dae_path: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out / "meshes", ignore_errors=True)
    (out / "meshes").mkdir(parents=True, exist_ok=True)

    pieces = [p for p in load_pieces(dae_path) if top_group(p["path"]) not in EXCLUDED_GROUPS]
    print(f"{len(pieces)} arena pieces")
    w = Writer(out)
    colb = CollisionBuilder(w.mesh_dir)
    meta = {"cad_commit": CAD_COMMIT, "buttons": {}, "valves": {}, "bulbs": {}, "door": {}}
    used: set[int] = set()

    def take(pred) -> list[dict]:
        sel = [p for i, p in enumerate(pieces) if i not in used and pred(p)]
        used.update(i for i, p in enumerate(pieces) if pred(p))
        return sel

    # --- push buttons -------------------------------------------------------
    button_roots: dict[str, str] = {}
    for p in pieces:
        if p["mat"] == "Metal_Brushed_2":
            root = "/".join(p["path"].split("/")[:-2])
            bezel_leaf = p["path"]
            button_roots[root] = bezel_leaf

    def button_kind(root: str) -> str:
        if root in MOBILITY_BUTTONS:
            return MOBILITY_BUTTONS[root]
        if root.startswith(KEYPAD):
            return "keypad"
        if root.startswith(BUTTON_BOARD_WALL):
            return "board_wall"
        if root.startswith(BUTTON_BOARD_FLOOR):
            return "board_floor"
        return "button"

    raw_buttons = []
    for root, bezel_leaf in button_roots.items():
        group = take(lambda p, r=root: p["path"].startswith(r + "/") or p["path"] == r)
        bezel = [p for p in group if p["path"] == bezel_leaf]
        cap = [p for p in group if p["path"] != bezel_leaf]
        cap_v = np.concatenate([p["verts"] for p in cap])
        bezel_v = np.concatenate([p["verts"] for p in bezel])
        axis = min_axis(bezel_v)
        if np.dot(axis, bezel_v.mean(0) - cap_v.mean(0)) < 0:
            axis = -axis  # point into the board
        raw_buttons.append((button_kind(root), root, cap, bezel, cap_v, axis))

    # Keypad digits: 3x3 grid + 0 below, read left-to-right as seen from the front.
    keypad = [b for b in raw_buttons if b[0] == "keypad"]
    if keypad:
        out_n = -keypad[0][5]  # direction the keypad faces
        viewer_right = np.cross(-out_n, [0, 0, 1])  # right-hand side for someone facing the keypad
        centers = np.array([b[4].mean(0) for b in keypad])
        rows = np.round(centers[:, 2], 2)
        order = np.lexsort((centers @ viewer_right, -rows))
        digits = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
        for d, idx in zip(digits, order):
            keypad[idx] = (f"keypad_{d}",) + keypad[idx][1:]
    counters = defaultdict(int)
    buttons = []
    for b in keypad + [b for b in raw_buttons if b[0] != "keypad"]:
        name = b[0]
        if name.startswith("board_"):
            name = f"{name}_{counters[name]}"
            counters[b[0]] += 1
        buttons.append((name,) + b[1:])

    for name, root, cap, bezel, cap_v, axis in buttons:
        w.visual(w.world, bezel, f"{name}_bezel")
        w.collision(w.world, colb.shapes(mesh_of(bezel), f"{name}_bezel"))
        center = cap_v.mean(0)
        T, pos, quat = body_frame(center, axis)
        body = ET.SubElement(w.world, "body", name=name, pos=fmt(pos), quat=fmt(quat))
        ET.SubElement(body, "inertial", pos="0 0 0", mass="0.03", diaginertia="1e-5 1e-5 1e-5")
        ET.SubElement(
            body, "joint", name=f"{name}_joint", type="slide", axis="0 0 1", range=f"0 {BUTTON_TRAVEL}",
            limited="true", stiffness=str(BUTTON_STIFFNESS), damping="1", solreflimit="-20000 -50",
        )
        w.visual(body, cap, name, T)
        col_names = w.collision(body, colb.shapes(mesh_of(cap), f"{name}_cap", T), name=f"{name}_cap")
        front = -np.max(-(trimesh.transform_points(cap_v, T))[:, 2])  # most outward point along -z
        ET.SubElement(body, "site", name=f"{name}_site", pos=f"0 0 {front:.5f}", size="0.006", rgba="1 0 0 1", group="5")
        # The cap slides into its bezel, whose collision is a filled hull; the joint limit is the stop.
        ET.SubElement(w.contact, "exclude", body1="world", body2=name)
        meta["buttons"][name] = dict(
            joint=f"{name}_joint", site=f"{name}_site", geoms=col_names, travel=BUTTON_TRAVEL,
            normal=(-axis).tolist(), center=center.tolist(),
        )

    # --- valves -------------------------------------------------------------
    valve_roots = sorted({p["path"].split("/ID4137/")[0] for p in pieces if "/ID4137/" in p["path"]})
    for board, prefix in ((VALVE_WALL, "valve_wall"), (VALVE_FLOOR, "valve_floor")):
        roots = [r for r in valve_roots if r.startswith(board)]
        for k, root in enumerate(roots):
            name = f"{prefix}_{k}"
            group = take(lambda p, r=root: p["path"].startswith(r + "/"))
            lever = [p for p in group if "/ID4137/SketchUp_Instance_79/" in p["path"]]
            stem = [p for p in group if "/ID4137/SketchUp_Instance_82/" in p["path"]]
            body_pieces = excluding(group, lever)
            stem_v = np.concatenate([p["verts"] for p in stem])
            axis = min_axis(stem_v)
            valve_body_v = np.concatenate([p["verts"] for p in excluding(body_pieces, stem) if p["mat"] == "Color_D08"])
            if np.dot(axis, stem_v.mean(0) - valve_body_v.mean(0)) < 0:
                axis = -axis  # outward: from the valve body through the stem
            w.visual(w.world, body_pieces, f"{name}_body")
            w.collision(w.world, colb.shapes(mesh_of(body_pieces), f"{name}_body"))
            T, pos, quat = body_frame(stem_v.mean(0), axis)
            body = ET.SubElement(w.world, "body", name=name, pos=fmt(pos), quat=fmt(quat))
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.1", diaginertia="2e-4 2e-4 2e-4")
            ET.SubElement(
                body, "joint", name=f"{name}_joint", type="hinge", axis="0 0 1",
                range=f"{-VALVE_RANGE:.5f} {VALVE_RANGE:.5f}", limited="true",
                frictionloss=str(VALVE_FRICTION), damping="0.05", solreflimit="0.004 1", solimplimit="0.99 0.999 0.001",
            )
            ET.SubElement(w.contact, "exclude", body1="world", body2=name)  # lever wraps the static stem
            w.visual(body, lever, name, T)
            col_names = w.collision(body, colb.shapes(mesh_of(lever), f"{name}_lever", T), name=f"{name}_lever")
            grip = [p for p in lever if p["mat"] == "Color_A11"]
            grip_c = trimesh.transform_points(np.concatenate([p["verts"] for p in grip]).mean(0)[None], T)[0]
            ET.SubElement(body, "site", name=f"{name}_grip", pos=fmt(grip_c), size="0.008", rgba="1 0 0 1", group="5")
            meta["valves"][name] = dict(joint=f"{name}_joint", site=f"{name}_grip", geoms=col_names, range=VALVE_RANGE)

    # --- light bulbs --------------------------------------------------------
    glass_all = [p for p in pieces if p["mat"] == "Translucent_Glass_Blue"]
    for board, prefix in ((BULB_WALL, "bulb_wall"), (BULB_FLOOR, "bulb_floor")):
        # Each bulb's leaf group holds its glass and its fixture.
        roots = sorted({p["path"] for p in glass_all if p["path"].startswith(board)})
        for k, root in enumerate(roots):
            name = f"{prefix}_{k}"
            glass = take(lambda p, r=root: p["path"] == r and p["mat"] == "Translucent_Glass_Blue")
            fixture = [p for p in pieces if p["path"] == root and p["mat"] != "Translucent_Glass_Blue"]
            glass_v = np.concatenate([p["verts"] for p in glass])
            fixture_v = np.concatenate([p["verts"] for p in fixture])
            axis = max_axis(glass_v)
            if np.dot(axis, fixture_v.mean(0) - glass_v.mean(0)) < 0:
                axis = -axis  # into the socket
            # Body frame at the glass center with z into the socket; tightening = +rotation (right-hand thread).
            T, pos, quat = body_frame(glass_v.mean(0), axis)
            body = ET.SubElement(w.world, "body", name=name, pos=fmt(pos), quat=fmt(quat))
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.04", diaginertia="3e-5 3e-5 3e-5")
            max_travel = BULB_TURNS * BULB_PITCH
            ET.SubElement(
                body, "joint", name=f"{name}_slide", type="slide", axis="0 0 1",
                range=f"{-max_travel:.5f} 0", limited="true", damping="1",
            )
            ET.SubElement(
                body, "joint", name=f"{name}_turn", type="hinge", axis="0 0 1",
                range=f"{-BULB_TURNS * 2 * np.pi:.5f} 0", limited="true", frictionloss="0.05", damping="0.002",
            )
            ET.SubElement(
                w.equality, "joint", name=f"{name}_thread", joint1=f"{name}_slide", joint2=f"{name}_turn",
                polycoef=f"0 {BULB_PITCH / (2 * np.pi):.6g} 0 0 0", solref="0.005 1",
            )
            w.visual(body, glass, name, T)
            col_names = w.collision(body, colb.shapes(mesh_of(glass), f"{name}_glass", T), name=f"{name}_glass")
            tip = np.min(trimesh.transform_points(glass_v, T)[:, 2])
            ET.SubElement(body, "site", name=f"{name}_tip", pos=f"0 0 {tip:.5f}", size="0.008", rgba="1 0 0 1", group="5")
            meta["bulbs"][name] = dict(
                slide=f"{name}_slide", turn=f"{name}_turn", site=f"{name}_tip", geoms=col_names,
                pitch=BULB_PITCH, max_turns=BULB_TURNS,
            )
            # Bulb and its own fixture always overlap at the threads.
            ET.SubElement(w.contact, "exclude", body1="world", body2=name)

    # --- door ---------------------------------------------------------------
    leaf_all = take(lambda p: p["path"].startswith(DOOR_LEAF_INSTANCE))
    jamb = [p for p in leaf_all if abs(min_axis(p["verts"])[0]) > 0.95 and len(p["faces"]) == 52]
    moving = excluding(leaf_all, jamb)
    panel = max((p for p in moving if p["mat"] != "Sophie_Shirt_Gray"), key=lambda p: np.ptp(p["verts"], 0).sum())
    handles = [p for p in moving if p["mat"] == "Sophie_Shirt_Gray"]
    trim = excluding(moving, [panel] + handles)
    jamb_v = np.concatenate([p["verts"] for p in jamb])
    jamb_x = jamb_v[:, 0].mean()
    opening_lo = jamb_v[:, 1].min()

    # The CAD shows the leaf open; rotate it back into the frame.
    pv = panel["verts"]
    xy = pv[:, :2]
    _, _, vt = np.linalg.svd(xy - xy.mean(0))
    d = vt[0] if np.dot(vt[0], [0, 1]) > 0 else -vt[0]
    hinge_end = xy[np.argmin((xy - xy.mean(0)) @ d)]
    theta = np.arctan2(d[0], d[1])  # rotate d onto +y
    R = rot_z(theta)
    close_T = np.eye(4)
    close_T[:3, :3] = R
    close_T[:3, 3] = np.array([*hinge_end, 0]) - R @ np.array([*hinge_end, 0])
    door_pieces = [panel] + trim + handles
    closed = [dict(p, verts=trimesh.transform_points(p["verts"], close_T)) for p in door_pieces]
    panel_c = closed[0]["verts"]
    shift = np.array([jamb_x - panel_c[:, 0].mean(), opening_lo + 0.004 - panel_c[:, 1].min(), 0.0])
    closed = [dict(p, verts=p["verts"] + shift) for p in closed]
    panel_c = closed[0]["verts"]
    # Rules: the robot pushes the door open from the start side (+x), so it swings toward -x.
    hinge_pt = np.array([panel_c[:, 0].min(), panel_c[:, 1].min(), panel_c[:, 2].min()])
    T, pos, quat = body_frame(hinge_pt)
    w.visual(w.world, jamb, "door_jamb")
    for i, p in enumerate(jamb):
        w.collision(w.world, colb.shapes(mesh_of([p]), f"door_jamb{i}"))
    leaf = ET.SubElement(w.world, "body", name="door", pos=fmt(pos))
    size = np.ptp(panel_c, 0)
    ET.SubElement(
        leaf, "inertial", pos=fmt(panel_c.mean(0) - hinge_pt), mass="15",
        diaginertia=fmt([15 * (size[1] ** 2 + size[2] ** 2) / 12, 15 * (size[0] ** 2 + size[2] ** 2) / 12, 15 * size[1] ** 2 / 3]),
    )
    ET.SubElement(
        leaf, "joint", name="door_hinge", type="hinge", axis="0 0 1", range=f"0 {DOOR_RANGE:.5f}",
        limited="true", stiffness=str(DOOR_STIFFNESS), damping=str(DOOR_DAMPING), springref="0",
    )
    leaf_static = [p for p in closed if p["mat"] != "Sophie_Shirt_Gray"]
    w.visual(leaf, leaf_static, "door_leaf", T)
    door_cols = w.collision(leaf, colb.shapes(mesh_of(leaf_static[:1]), "door_panel", T), name="door_panel")
    # Lever handles: the two big pieces are the levers, the rest are roses on the leaf.
    closed_handles = [p for p in closed if p["mat"] == "Sophie_Shirt_Gray"]
    levers = sorted(closed_handles, key=lambda p: -len(p["faces"]))[:2]
    roses = excluding(closed_handles, levers)
    w.visual(leaf, roses, "door_roses", T)
    handle_names = []
    for k, lev in enumerate(sorted(levers, key=lambda p: p["verts"][:, 0].mean(), reverse=True)):
        side = "start" if k == 0 else "far"
        lv = lev["verts"]
        rose_v = np.concatenate([p["verts"] for p in roses])
        # Pivot: the rose on the same face of the leaf as this lever.
        same = rose_v[np.sign(rose_v[:, 0] - panel_c[:, 0].mean()) == np.sign(lv[:, 0].mean() - panel_c[:, 0].mean())]
        pivot = same.mean(0) if len(same) else lv[np.argmin(np.abs(lv[:, 1] - lv[:, 1].mean()))]
        pivot[2] = lv[:, 2].mean()
        lever_dir = lv.mean(0) - pivot
        lever_dir[0] = lever_dir[2] = 0
        lever_dir /= np.linalg.norm(lever_dir) + 1e-9
        normal = np.array([1.0, 0, 0]) if side == "start" else np.array([-1.0, 0, 0])
        axis = np.cross(lever_dir, [0, 0, -1])  # positive rotation pushes the lever end down
        Th, hpos, hquat = body_frame(pivot, axis)
        hname = f"door_handle_{side}"
        hb = ET.SubElement(leaf, "body", name=hname, pos=fmt(hpos - hinge_pt), quat=fmt(hquat))
        ET.SubElement(hb, "inertial", pos="0 0 0", mass="0.2", diaginertia="1e-4 1e-4 1e-4")
        ET.SubElement(
            hb, "joint", name=f"{hname}_joint", type="hinge", axis="0 0 1", range=f"0 {HANDLE_RANGE:.5f}",
            limited="true", stiffness=str(HANDLE_STIFFNESS), damping="0.02",
        )
        w.visual(hb, [lev], hname, Th)
        hcols = w.collision(hb, colb.shapes(mesh_of([lev]), hname, Th), name=hname)
        grip = trimesh.transform_points((pivot + lever_dir * 0.08)[None], Th)[0]
        ET.SubElement(hb, "site", name=f"{hname}_grip", pos=fmt(grip), size="0.01", rgba="1 0 0 1", group="5")
        handle_names.append(dict(joint=f"{hname}_joint", site=f"{hname}_grip", geoms=hcols, normal=normal.tolist()))
        ET.SubElement(w.contact, "exclude", body1="door", body2=hname)
    ET.SubElement(w.equality, "joint", name="door_latch", joint1="door_hinge", polycoef="0 0 0 0 0", active="true")
    meta["door"] = dict(
        hinge="door_hinge", latch="door_latch", geoms=door_cols, handles=handle_names,
        range=DOOR_RANGE, latch_release=float(HANDLE_RANGE * 0.6), hinge_pos=hinge_pt.tolist(),
    )
    ET.SubElement(w.contact, "exclude", body1="world", body2="door")

    # --- terrain + static arena --------------------------------------------
    terrain = take(lambda p: top_group(p["path"]) in TERRAIN_GROUPS)
    tb = ET.SubElement(w.world, "body", name="terrain")
    w.visual(tb, terrain, "terrain")
    for i, p in enumerate(terrain):
        w.collision(tb, colb.shapes(mesh_of([p]), f"terrain{i}"))

    static = [p for i, p in enumerate(pieces) if i not in used]
    w.visual(w.world, static, "static")
    by_group = defaultdict(list)
    for p in static:
        by_group[p["path"]].append(p)
    for i, (path, group) in enumerate(by_group.items()):
        for j, p in enumerate(group):
            w.collision(w.world, colb.shapes(mesh_of([p]), f"static{i}_{j}"))

    w.save()
    (out / "arena_meta.json").write_text(json.dumps(meta, indent=1))
    write_stations(out)
    n_mesh = len(list((out / "meshes").glob("*.stl")))
    size_mb = sum(f.stat().st_size for f in (out / "meshes").glob("*.stl")) / 1e6
    print(f"collision shapes: {dict(colb.stats)}; {n_mesh} mesh files ({size_mb:.1f} MB)")
    print(f"buttons={len(meta['buttons'])} valves={len(meta['valves'])} bulbs={len(meta['bulbs'])} door handles={len(handle_names)}")
    print(f"wrote {out / 'arena.xml'}")


def write_stations(out: Path) -> None:
    """Compiles the arena and stores robot standing poses for each station in arena_meta.json."""
    import dataclasses

    import sys

    import mujoco

    sys.path.insert(0, str(ROOT))
    from g1_manip import arena

    model = mujoco.MjModel.from_xml_path(str(out / "arena.xml"))
    meta = json.loads((out / "arena_meta.json").read_text())
    meta["stations"] = {k: dataclasses.asdict(v) for k, v in arena.compute_stations(model).items()}
    (out / "arena_meta.json").write_text(json.dumps(meta, indent=1))
    for k, v in meta["stations"].items():
        print(f"  station {k:13s} x={v['x']:.3f} y={v['y']:.3f} floor={v['floor_z']:.3f} yaw={np.degrees(v['yaw']):.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dae", type=Path, default=None, help="use a local CompetitionTestbed.dae instead of downloading")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stations-only", action="store_true", help="only recompute station poses")
    args = parser.parse_args()
    if args.stations_only:
        write_stations(args.out)
        return
    dae = args.dae or download(args.out.parent / "_cache")
    print(f"CAD: {dae} (sha1 {hashlib.sha1(dae.read_bytes()).hexdigest()[:12]})")
    build(dae, args.out)


if __name__ == "__main__":
    main()
