"""
Ground-truth object models: load them, and derive the numbers the experiments use.

Each object lives in ``objects/<name>/``:

- ``<name>.yaml``: the authored record (mass, CoM, mesh, rest pose, tip edges,
  provenance). This is the only file a person edits.
- ``<name>.urdf`` and ``meshes/<name>.stl``: generated from the YAML and the CAD
  mesh by ``ros2 run irb120_models object_models generate``, for RViz, MoveIt
  and MuJoCo. Never hand-edit them; ``object_models check`` fails when they are
  stale.

Why not put everything in the URDF: it has no slot for uncertainty, provenance,
friction, how the object rests on the table, or which edge it tips about, and
the old ``object_params.json`` showed that numbers without a source drift. The
YAML carries those, and the URDF stays a plain interchange file any tool loads.

Frames. The *body frame* is the mesh's CAD frame. The *support frame* has the
table plane at z = 0, z up; ``rest_pose`` maps body to support for the pose the
experiments use. Tip edges are lines in the support frame. ``com_x`` (horizontal
distance from the tip edge to the CoM) and ``com_z`` (CoM height) are derived
per edge, because they are properties of the edge as much as of the object; the
estimator takes ``com_x`` as known, so getting the edge right matters as much as
the CoM.

Ground truth is the *measured* mass and CoM. Most everyday objects have no CAD
model, so nothing here depends on one: the mesh is optional, and theta* is derived
from the measured CoM about the tip edge (sharp-edge atan(com_x / com_z)). Where a
CAD model exists, its values go in the ``cad`` block, which is informational only.
When the CoM was measured against a tip edge rather than in a body frame, the
``recorded`` block holds it and ``ground_truth()`` returns it with
``source='recorded'``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import yaml

STATUSES = ('complete', 'provisional')
BODY_FRAMES = ('cad', 'tip_edge')


def objects_dir() -> Path:
    """Return ``objects/``: the source tree if available, else the installed share."""
    source = Path(__file__).resolve().parents[1] / 'objects'
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('irb120_models')) / 'objects'


def list_objects() -> list[str]:
    return sorted(p.parent.name for p in objects_dir().glob('*/*.yaml')
                  if p.stem == p.parent.name)


@dataclass(frozen=True)
class Measured:
    """A value plus where it came from. ``value`` is None until it is measured."""

    value: object = None
    sigma: object = None
    source: str | None = None


@dataclass(frozen=True)
class TipEdge:
    point: np.ndarray       # on the edge, support frame, m
    direction: np.ndarray   # along the edge, support frame


@dataclass(frozen=True)
class ObjectModel:
    name: str
    label: str
    status: str
    directory: Path
    body_frame: str
    mesh: Path | None
    rest_xyz: np.ndarray
    rest_rpy: np.ndarray
    mass: Measured
    com: Measured
    inertia: Measured
    tip_edges: dict[str, TipEdge]
    friction: dict
    recorded: dict | None
    cad: dict | None
    notes: list[str] = field(default_factory=list)

    @property
    def urdf(self) -> Path | None:
        path = self.directory / f'{self.name}.urdf'
        return path if path.exists() else None

    @property
    def stl(self) -> Path | None:
        path = self.directory / 'meshes' / f'{self.name}.stl'
        return path if path.exists() else None

    def rest_rotation(self) -> np.ndarray:
        return rpy_to_matrix(self.rest_rpy)

    def com_in_support(self) -> np.ndarray | None:
        if self.com.value is None:
            return None
        return self.rest_rotation() @ np.asarray(self.com.value, float) + self.rest_xyz

    def ground_truth(self, edge: str = 'default') -> dict:
        """
        Return mass, com_x, com_z and theta* for tipping about ``edge``.

        ``source`` says whether the CoM came from the body frame ('body_frame') or
        from the edge-relative measurement ('recorded'). theta* is derived from that
        measured CoM: the sharp-edge balance angle atan(com_x / com_z). A CAD theta*,
        where one exists, is returned separately as ``theta_star_cad_deg``.
        """
        rec = self.recorded or {}
        out = {'mass': self.mass.value, 'edge': edge,
               'theta_star_cad_deg': (self.cad or {}).get('theta_star_deg')}
        p = self.com_in_support()
        if p is not None and edge in self.tip_edges:
            com_x, com_z = edge_offsets(p, self.tip_edges[edge])
            out.update(com_x=com_x, com_z=com_z, source='body_frame')
        elif rec.get('com_from_edge') is not None and rec.get('edge', 'default') == edge:
            com_x, _, com_z = (float(v) for v in rec['com_from_edge'])
            out.update(com_x=com_x, com_z=com_z, source='recorded')
        else:
            out.update(com_x=None, com_z=None, source=None)
        out['theta_star_deg'] = (math.degrees(math.atan2(out['com_x'], out['com_z']))
                                 if out['com_x'] is not None else None)
        return out


def load_object(name: str) -> ObjectModel:
    path = objects_dir() / name / f'{name}.yaml'
    if not path.exists():
        raise ValueError(f"Unknown object '{name}'. Known: {list_objects()}")
    return parse_object(path)


def parse_object(path: Path) -> ObjectModel:
    raw = yaml.safe_load(path.read_text()) or {}
    where = str(path)
    name = raw.get('name')
    if name != path.stem or path.parent.name != name:
        raise ValueError(f"{where}: 'name' must match the file and folder name ({path.stem})")
    status = raw.get('status')
    if status not in STATUSES:
        raise ValueError(f'{where}: status must be one of {STATUSES}, got {status!r}')
    body_frame = raw.get('body_frame', 'cad')
    if body_frame not in BODY_FRAMES:
        raise ValueError(f'{where}: body_frame must be one of {BODY_FRAMES}')
    mesh = raw.get('mesh')
    mesh_path = path.parent / mesh if mesh else None
    if mesh_path is not None and not mesh_path.exists():
        raise ValueError(f'{where}: mesh {mesh_path} does not exist')
    rest = raw.get('rest_pose') or {}
    edges = {key: TipEdge(_vec(e['point'], where), _unit(_vec(e['direction'], where), where))
             for key, e in (raw.get('tip_edges') or {}).items()}
    for key, edge in edges.items():
        if abs(edge.direction[2]) > 1e-6:
            raise ValueError(f"{where}: tip edge '{key}' must be horizontal in the support frame")

    model = ObjectModel(
        name=name,
        label=raw.get('label', name),
        status=status,
        directory=path.parent,
        body_frame=body_frame,
        mesh=mesh_path,
        rest_xyz=_vec(rest.get('xyz', [0, 0, 0]), where),
        rest_rpy=_vec(rest.get('rpy', [0, 0, 0]), where),
        mass=_measured(raw.get('mass'), where, 'mass', shape=()),
        com=_measured(raw.get('com'), where, 'com', shape=(3,)),
        inertia=_measured(raw.get('inertia'), where, 'inertia', shape=(3, 3)),
        tip_edges=edges,
        friction=raw.get('friction') or {},
        recorded=raw.get('recorded'),
        cad=raw.get('cad'),
        notes=list(raw.get('notes') or []),
    )
    if body_frame == 'tip_edge' and mesh_path is not None:
        raise ValueError(f"{where}: body_frame 'tip_edge' is for objects without a CAD mesh")
    if status == 'complete':  # a mesh is never required: most objects have no CAD
        has_com = model.com.value is not None or (model.recorded or {}).get('com_from_edge')
        missing = [k for k, v in (('mass', model.mass.value), ('com', has_com),
                                  ('tip_edges.default', edges.get('default')))
                   if v is None]
        if missing:
            raise ValueError(f"{where}: status 'complete' but missing {missing}")
    return model


def edge_offsets(p_support: np.ndarray, edge: TipEdge) -> tuple[float, float]:
    """(horizontal distance from the edge line to the point, height above the edge)."""
    d = np.array([edge.direction[0], edge.direction[1]])
    d /= np.linalg.norm(d)
    v = p_support - edge.point
    return float(abs(v[0] * d[1] - v[1] * d[0])), float(v[2])


def rpy_to_matrix(rpy) -> np.ndarray:
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), \
        math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _measured(block, where: str, key: str, shape: tuple) -> Measured:
    if block is None:
        return Measured()
    if not isinstance(block, dict):
        raise ValueError(f"{where}: '{key}' must be a mapping with value/sigma/source")
    value = block.get('value')
    if value is not None:
        arr = np.asarray(value, dtype=float)
        if arr.shape != shape:
            raise ValueError(f"{where}: '{key}.value' must have shape {shape}, got {arr.shape}")
        value = float(arr) if shape == () else arr
    return Measured(value, block.get('sigma'), block.get('source'))


def _vec(value, where: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f'{where}: expected a 3-vector, got {value!r}')
    return arr


def _unit(v: np.ndarray, where: str) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError(f'{where}: tip edge direction is zero')
    return v / n
