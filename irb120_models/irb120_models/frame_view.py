"""
Check an object's experiment frame and CoM against a perceived snapshot (``object_models frame``).

    object_models frame SNAPSHOT [--name NAME] [--com X Y Z] [--origin press|cloud]
                                 [--shift DX DY DZ] [--yaw DEG]
                                 [--mesh PATH] [--mesh-pose X Y Z R P Y] [--png OUT]

SNAPSHOT is a perceive folder (or an episode folder: its first perceive). NAME defaults to the
episode's object label and is looked up in the configurations (com_obj, base_w,
support_distances, mesh); an unknown name just shows the snapshot.

The frame is near_outer_base_edge_x_along_push_z_up, placed automatically: x along the
snapshot's push direction, z = 0 on the table, y on the footprint's lateral mid-line (the object
cloud's lowest BASE_BAND_M). Origin x (--origin): 'press' (default) = the press-pull pivot the
robot used (camera-reviewed; for a sharp base it is the near outer edge, for a filleted base it is
the flat support, so shift by the fillet); 'cloud' = the near extreme of the fused cloud's
footprint (registration between cameras can put that 1-2 cm off; check against the pivots).
--shift (mm, frame axes) and --yaw (deg) move it by hand; the CoM, edges and mesh move with it.
The clouds and the pivots the robot used (press-pull magenta, forward cyan) stay put, so the
question the view answers is: does the frame I measured my CoM in sit where the robot tipped?

All lengths typed or shown are millimetres; the YAML snippet (Enter) is in metres.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from irb120_models import mesh_tools
from irb120_models.configurations import configuration_mesh, load_configuration

BASE_BAND_M = .02        # footprint = object cloud points this close to its lowest point
EXTREME_PERCENTILE = 2   # robust extremes of the footprint (rejects stray voxels)
TABLE_Z_DEFAULT = -.021  # table height in base_link when the snapshot does not record it
SCENE_PAD_M = .06        # scene-cloud crop around the object
TABLE_CLEARANCE_M = .003  # scene points this close to the scene's own table are not drawn
DATA = Path.home() / 'Documents' / 'irb120_data'
DERIVED = DATA / 'derived'   # scene clouds rebuilt for archived snapshots (analysis/backfill_scene_clouds.py)

KEYS = """CONTROLS
m          what moves: frame / CoM / far edge
arrows     x (left/right), y (up/down)
PgUp PgDn  z
[ ]        yaw the frame
+ -        step: 0.1 / 1 / 10 mm
v          view: side, top, front, 3D
o  g       orange / grey cloud on/off
h          hide the numbers panel
s          save a screenshot (current dir)
Enter / p  print the YAML entry here
r          reset camera     q  quit
mouse      rotate, Shift pan, wheel zoom"""

LEGEND_TEXT = (('red arrow', 'frame x: push direction, away from the robot'),
               ('green arrow', "frame y: robot's left"),
               ('blue arrow', 'frame z: up from the table'),
               ('red line', 'near edge, x = 0 (frame origin)'),
               ('pink line', 'far edge, x = base_w'),
               ('yellow', 'CoM (vertical line when z is unknown)'),
               ('magenta', 'press-pull pivot the robot used'),
               ('cyan', 'forward pivot the robot used'),
               ('white arrow', 'toward the robot (near side)'),
               ('orange dots', 'object cloud (fused, 5 mm voxels)'),
               ('grey dots', 'scene cloud (per camera, table removed)'),
               ('corner axes', 'robot base_link x/y/z'))

# ---------------------------------------------------------------- snapshot and frame (numpy only)

def resolve_snapshot(path: Path) -> tuple[Path, dict]:
    """Perceive folder and its episode.json (empty dict if none)."""
    path = Path(path).expanduser().resolve()
    if not (path / 'object_cloud.npz').exists():
        found = sorted(p.parent for p in path.glob('*_perceive/object_cloud.npz'))
        if not found:
            raise FileNotFoundError(f'{path}: no object_cloud.npz here or in */*_perceive')
        path = found[0]
    episode = path.parent / 'episode.json'
    return path, (json.loads(episode.read_text()) if episode.exists() else {})


def scene_folder(folder: Path) -> Path | None:
    """Where support_scene_cloud.npz is: the snapshot, or its mirror under DERIVED."""
    if (folder / 'support_scene_cloud.npz').exists():
        return folder
    try:
        mirror = DERIVED / folder.relative_to(DATA)
    except ValueError:
        return None
    return mirror if (mirror / 'support_scene_cloud.npz').exists() else None


def load_snapshot(folder: Path) -> dict:
    """Object cloud, optional scene cloud, table z, push direction and the robot's pivots (base_link)."""
    cloud = np.load(folder / 'object_cloud.npz')['xyz'].astype(float)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    table_z, scene, scene_table, where = TABLE_Z_DEFAULT, None, None, scene_folder(folder)
    if where is not None and (where / 'support_scene_cloud.json').exists():
        table_z = json.loads((where / 'support_scene_cloud.json').read_text()).get('table_z_reference_m', TABLE_Z_DEFAULT)
    if where is not None:
        data = np.load(where / 'support_scene_cloud.npz')
        scene = data['xyz'].astype(float)
        camera = data['camera_id'] if 'camera_id' in data else np.zeros(len(scene), int)
        lo, hi = cloud.min(0) - SCENE_PAD_M, cloud.max(0) + SCENE_PAD_M
        keep = np.isfinite(scene).all(axis=1) & np.all((scene >= lo) & (scene <= hi), axis=1)
        scene, camera, scene_table = scene[keep], camera[keep], {}
        on_table = np.zeros(len(scene), bool)
        for cam in np.unique(camera):  # each camera sees the table at its own height
            z = scene_table_z(scene[camera == cam], table_z)
            if z is not None:
                scene_table[f'cam{cam}'] = z
                on_table |= (camera == cam) & (scene[:, 2] <= z + TABLE_CLEARANCE_M)
        scene = scene[~on_table]  # the grid shows the table
    contacts = json.loads((folder / 'contacts.json').read_text()) if (folder / 'contacts.json').exists() else {}
    pivots, direction = {}, None
    for mode in ('press', 'forward_tip'):
        geom = (contacts.get(mode) or {}).get('geometry') or {}
        if geom.get('pivot') is not None:
            pivots[mode] = dict(point=np.asarray(geom['pivot'], float),
                                axis=np.asarray(geom.get('axis') or [0., 1., 0.], float),
                                source=geom.get('pivot_source'))
        if mode == 'forward_tip' and geom.get('direction') is not None:
            direction = np.asarray(geom['direction'], float)
    if direction is None and 'press' in pivots:
        direction = -np.asarray(contacts['press']['geometry'].get('direction') or [-1., 0., 0.], float)
    if direction is None:
        direction = np.array([1., 0., 0.])
    return dict(cloud=cloud, scene=scene, table_z=float(table_z), scene_table_z=scene_table,
                direction=direction, pivots=pivots)


def scene_table_z(scene, table_z, reach=.015, bin_m=.001) -> float | None:
    """Densest 1 mm height bin within +-reach of the configured table: where the cameras put it."""
    z = scene[np.abs(scene[:, 2] - table_z) < reach, 2]
    if len(z) < 100:
        return None
    counts, edges = np.histogram(z, bins=np.arange(table_z - reach, table_z + reach + bin_m, bin_m))
    return float(edges[np.argmax(counts)] + bin_m / 2)


def horizontal_unit(v) -> np.ndarray:
    v = np.array([v[0], v[1], 0.], float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError('push direction has no horizontal component')
    return v / n


def rotation(direction, yaw_deg: float = 0.) -> np.ndarray:
    """Columns: frame x (push, yawed), y (left of it), z (up), in base_link."""
    u = horizontal_unit(direction)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], 0.])
    return np.column_stack([x, [-x[1], x[0], 0.], [0., 0., 1.]])


def footprint(cloud, R, table_z) -> dict:
    """Footprint extremes of the cloud's lowest band, in the axes of R (origin: base_link)."""
    base = cloud[cloud[:, 2] <= cloud[:, 2].min() + BASE_BAND_M]
    s, t = base @ R[:, 0], base @ R[:, 1]
    q = EXTREME_PERCENTILE
    near, far = np.percentile(s, [q, 100 - q])
    right, left = np.percentile(t, [q, 100 - q])
    return dict(near=float(near), far=float(far), right=float(right), left=float(left),
                gap_above_table_m=float(base[:, 2].min() - table_z), points=len(base))


def auto_origin(snap, origin='press') -> tuple[np.ndarray, str]:
    """Frame origin at table height on the footprint's lateral mid-line; x from the press-pull
    pivot ('press', when the snapshot has one) or the near extreme of the cloud footprint."""
    R = rotation(snap['direction'])
    f = footprint(snap['cloud'], R, snap['table_z'])
    if origin == 'press' and 'press' in snap['pivots']:
        x, how = float(snap['pivots']['press']['point'] @ R[:, 0]), 'press-pull pivot'
    else:
        x, how = f['near'], 'near extreme of the cloud footprint'
    return x * R[:, 0] + (f['left'] + f['right']) / 2 * R[:, 1] + np.array([0., 0., snap['table_z']]), how


class Frame:
    """Experiment frame in base_link: the automatic placement, plus a hand shift and yaw."""

    def __init__(self, direction, auto, shift_mm=(0., 0., 0.), yaw_deg=0.):
        self.direction, self.auto = np.asarray(direction, float), np.asarray(auto, float)
        self.shift_mm, self.yaw_deg = np.array(shift_mm, float), float(yaw_deg)

    @property
    def R(self) -> np.ndarray:
        return rotation(self.direction, self.yaw_deg)

    @property
    def origin(self) -> np.ndarray:
        return self.auto + self.R @ (self.shift_mm / 1000)

    def to_frame(self, pts) -> np.ndarray:
        return (np.atleast_2d(pts) - self.origin) @ self.R

    def to_base(self, pts) -> np.ndarray:
        return np.atleast_2d(pts) @ self.R.T + self.origin

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.origin
        return T


def euler_matrix(r_deg, p_deg, y_deg) -> np.ndarray:
    """URDF-style rpy (fixed axes x, y, z) as a rotation matrix."""
    r, p, y = np.radians([r_deg, p_deg, y_deg])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def record_for(name: str | None) -> dict | None:
    if not name:
        return None
    try:
        return load_configuration(name)
    except ValueError:
        return None


def record_com_mm(record) -> list:
    """Record com_obj as [x, y, z] in mm (None for unknown; legacy [x, z] gets y = 0)."""
    if record is None:
        return [None, None, None]
    com = list(record['com_obj'])
    if len(com) == 2:
        com = [com[0], 0., com[1]]
    return [None if v is None else 1000 * v for v in com]


def report(state) -> str:
    """Numbers in the current frame, for the terminal and the overlay."""
    f, snap, rec = state['frame'], state['snap'], state['record']
    cloud = f.to_frame(snap['cloud'])
    fp = footprint(snap['cloud'], f.R, snap['table_z'])
    o = f.origin @ f.R
    lines = [f"{state['name'] or '(no name)'}   {state['snapshot'].parent.parent.name}/{state['snapshot'].name}",
             f"frame: origin at the {state['origin']}, shift [{', '.join(f'{v:.1f}' for v in f.shift_mm)}] mm, "
             f"yaw {f.yaw_deg:.1f} deg   moving: {state['target']}, step {state['step']:g} mm",
             f"footprint x {1000 * (fp['near'] - o[0]):.1f} .. {1000 * (fp['far'] - o[0]):.1f} mm "
             f"(length {1000 * (fp['far'] - fp['near']):.1f}), "
             f"y {1000 * (fp['right'] - o[1]):.1f} .. {1000 * (fp['left'] - o[1]):.1f} mm, "
             f"top z {1000 * cloud[:, 2].max():.0f} mm",
             f"cloud starts {1000 * fp['gap_above_table_m']:.0f} mm above the table (the footprint is that band)"
             + ('' if not snap['scene_table_z'] else '; table seen at ' + ', '.join(
                 f'{cam} {1000 * (z - snap["table_z"]):+.1f}' for cam, z in snap['scene_table_z'].items()) + ' mm')]
    for mode, label in (('press', 'press-pull pivot'), ('forward_tip', 'forward pivot')):
        if mode in snap['pivots']:
            p = f.to_frame(snap['pivots'][mode]['point'])[0] * 1000
            extra = ''
            if mode == 'forward_tip' and state['base_w_mm'] is not None:
                extra = f", {p[0] - state['base_w_mm']:+.1f} mm from the far edge"
            lines.append(f"{label}: x {p[0]:.1f} mm, y {p[1]:.1f} mm{extra}  ({snap['pivots'][mode]['source']})")
    com = state['com_mm']
    lines.append('CoM: [' + ', '.join('?' if v is None else f'{v:.1f}' for v in com) + '] mm'
                 + ('  (z unknown: drawn as a vertical line)' if com[2] is None else ''))
    bw_mm = state['base_w_mm']
    if bw_mm is not None and com[0] is not None:
        lines.append(f"base_w {bw_mm:.1f} mm -> support_distances: press_pull {com[0]:.1f}, "
                     f"forward {bw_mm - com[0]:.1f} mm")
    if rec is not None:
        d = rec.get('support_distances') or {}
        bw = rec.get('base_w')
        lines.append(f"record now: base_w {'?' if bw is None else f'{1000 * bw:.1f}'} mm, support_distances "
                     f"press_pull {'?' if d.get('press_pull') is None else f'{1000 * d['press_pull']:.1f}'}, "
                     f"forward {'?' if d.get('forward') is None else f'{1000 * d['forward']:.1f}'} mm")
    return '\n'.join(lines)


def snippet(state) -> str:
    f, com = state['frame'], state['com_mm']
    fp = footprint(state['snap']['cloud'], f.R, state['snap']['table_z'])
    m = lambda v: 'null' if v is None else f'{v / 1000:.4f}'  # noqa: E731
    args = [f"object_models frame {state['snapshot']}"]
    if state['origin_arg'] != 'press':
        args.append(f"--origin {state['origin_arg']}")
    if state['name']:
        args.append(f"--name {state['name']}")
    if np.any(f.shift_mm):
        args.append('--shift ' + ' '.join(f'{v:g}' for v in f.shift_mm))
    if f.yaw_deg:
        args.append(f'--yaw {f.yaw_deg:g}')
    if any(v is not None for v in com):
        args.append('--com ' + ' '.join('nan' if v is None else f'{v:g}' for v in com))
    bw = state['base_w_mm']
    if bw is not None:
        args.append(f'--base-w {bw:g}')
    forward = None if bw is None or com[0] is None else bw - com[0]
    return '\n'.join([
        '# Paste under `configurations:` in objects/everyday/configurations.yaml (replace the {} if empty).',
        '# Fill placement, uncertainty, measurement and source; leave mass null until estimates are frozen.',
        f"  {state['name'] or 'NAME'}:",
        '    placement: "TODO: which face points at the robot (marked on the object)"',
        '    mass: null',
        f"    com_obj: [{', '.join(m(v) for v in com)}]",
        f"    base_w: {m(bw)}" + ('' if bw is not None else
                                  f"  # cloud footprint here: {1000 * (fp['far'] - fp['near']):.0f} mm; use calipers"),
        f"    support_distances: {{press_pull: {m(com[0])}, forward: {m(forward)}}}",
        '    uncertainty:',
        '      support_distances: {press_pull: null, forward: null}',
        '      com_obj: [null, null, null]',
        f'    measurement: "frame checked with: {" ".join(args)}"',
        '    source: null'])


# ---------------------------------------------------------------- VTK view

def _vtk():
    import vtk  # noqa: F401  (system python3-vtk; imported lazily so the package does not need it)
    from vtk.util import numpy_support
    return vtk, numpy_support


def _points_actor(pts, color, size):
    vtk, ns = _vtk()
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(np.ascontiguousarray(pts, dtype=np.float64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    glyph = vtk.vtkVertexGlyphFilter()
    glyph.SetInputData(poly)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetPointSize(size)
    return actor


def _lines_actor(segments, color, width=3, opacity=1.):
    vtk, _ = _vtk()
    points, cells = vtk.vtkPoints(), vtk.vtkCellArray()
    for a, b in segments:
        i = points.InsertNextPoint(*a)
        j = points.InsertNextPoint(*b)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, j)
        cells.InsertNextCell(line)
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetLineWidth(width)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def _sphere_actor(center, radius, color):
    vtk, _ = _vtk()
    src = vtk.vtkSphereSource()
    src.SetCenter(*center)
    src.SetRadius(radius)
    src.SetThetaResolution(24)
    src.SetPhiResolution(24)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(src.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    return actor


def _mesh_actor(triangles, color=(.7, .7, .9), opacity=.2):
    vtk, ns = _vtk()
    tri = np.asarray(triangles, float).reshape(-1, 3)
    points = vtk.vtkPoints()
    points.SetData(ns.numpy_to_vtk(np.ascontiguousarray(tri), deep=True))
    ids = np.column_stack([np.full(len(tri) // 3, 3), np.arange(len(tri)).reshape(-1, 3)]).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(len(tri) // 3, ns.numpy_to_vtkIdTypeArray(ids.astype(np.int64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def _label(position, text, color, size=16):
    vtk, _ = _vtk()
    actor = vtk.vtkBillboardTextActor3D()
    actor.SetInput(text)
    actor.SetPosition(*position)
    actor.GetTextProperty().SetFontSize(size)
    actor.GetTextProperty().SetColor(*color)
    actor.GetTextProperty().SetBold(True)
    return actor


def _arrow_segments(origin, axis, length, head=.008):
    tip = origin + axis * length
    side = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < .9 else np.cross(axis, [1, 0, 0])
    side = side / np.linalg.norm(side)
    return [(origin, tip), (tip, tip - axis * head + side * head / 2), (tip, tip - axis * head - side * head / 2)]


def fixed_actors(state) -> list:
    """Things that do not move with the frame: clouds and the robot's pivots (base_link)."""
    snap, actors = state['snap'], []
    if snap['scene'] is not None:
        actors.append(_points_actor(snap['scene'], (.55, .55, .55), 1.5))
    actors.append(_points_actor(snap['cloud'], (1., .55, .1), 4))
    for mode, color in (('press', (.9, .1, .9)), ('forward_tip', (.1, .85, .9))):
        if mode in snap['pivots']:
            p, a = snap['pivots'][mode]['point'], horizontal_unit(snap['pivots'][mode]['axis'])
            actors += [_lines_actor([(p - .04 * a, p + .04 * a)], color, 5), _sphere_actor(p, .003, color)]
    return actors


def frame_actors(state) -> list:
    """Frame axes, table patch, near/far edges, CoM and mesh, built in base_link from the frame."""
    f, snap, rec = state['frame'], state['snap'], state['record']
    o, R = f.origin, f.R
    fp = footprint(snap['cloud'], R, snap['table_z'])
    oo = o @ R
    y0, y1 = fp['right'] - oo[1] - .01, fp['left'] - oo[1] + .01
    far = None if state['base_w_mm'] is None else state['base_w_mm'] / 1000
    at = lambda x, y, z: o + R @ np.array([x, y, z])  # noqa: E731
    actors = []
    for k, (color, label) in enumerate((((1, .15, .15), 'x'), ((.15, .8, .15), 'y'), ((.25, .4, 1), 'z'))):
        actors.append(_lines_actor(_arrow_segments(o, R[:, k], .05), color, 4))
        actors.append(_label(o + R[:, k] * .057, label, color))
    # Toward the robot: the near side. Drawn just outside the near edge, on the table.
    start = at(-.015, 0, .002)
    actors.append(_lines_actor(_arrow_segments(start, -R[:, 0], .05, .012), (1, 1, 1), 4))
    actors.append(_label(at(-.09, 0, .02), 'ROBOT (near side)', (1, 1, 1)))
    actors.append(_label(at((far or fp['far'] - oo[0]) + .02, 0, .01), 'far side', (1, .6, .6)))
    x_lo, x_hi = min(-.02, fp['near'] - oo[0] - .02), max(far or 0, fp['far'] - oo[0]) + .02
    grid = [(at(x, y0 - .01, 0), at(x, y1 + .01, 0)) for x in np.arange(x_lo, x_hi + 1e-9, .01)]
    grid += [(at(x_lo, y, 0), at(x_hi, y, 0)) for y in np.arange(y0 - .01, y1 + .01 + 1e-9, .01)]
    actors.append(_lines_actor(grid, (.35, .35, .35), 1, .5))
    # Edges run along y; the 15 mm ticks make them visible in the side view too.
    actors.append(_lines_actor([(at(0, y0, 0), at(0, y1, 0)), (at(0, y0, 0), at(0, y0, .015))], (1, .15, .15), 3))
    if far:
        actors.append(_lines_actor([(at(far, y0, 0), at(far, y1, 0)), (at(far, y0, 0), at(far, y0, .015))],
                                   (1, .6, .6), 3))
    top = float(f.to_frame(snap['cloud'])[:, 2].max()) + .01
    x, y, z = [None if v is None else v / 1000 for v in state['com_mm']]
    if x is not None:
        y = y or 0.
        if z is None:
            actors.append(_lines_actor([(at(x, y, 0), at(x, y, top))], (1, 1, 0), 4))
        else:
            actors += [_sphere_actor(at(x, y, z), .005, (1, 1, 0)), _lines_actor([(at(x, y, 0), at(x, y, z))], (1, 1, 0), 1)]
        actors.append(_sphere_actor(at(x, y, 0), .003, (1, 1, 0)))
    d = (rec or {}).get('support_distances') or {}
    if d.get('press_pull') is not None:  # table-edge balance from the near side
        actors.append(_lines_actor([(at(d['press_pull'], y0, 0), at(d['press_pull'], y0, top))], (.9, .1, .9), 2, .7))
    if d.get('forward') is not None and far:
        xf = far - d['forward']
        actors.append(_lines_actor([(at(xf, y1, 0), at(xf, y1, top))], (.1, .85, .9), 2, .7))
    if state['mesh'] is not None:
        pose = state['mesh_pose']
        Rm = euler_matrix(*pose[3:])
        tri = state['mesh'].reshape(-1, 3) @ Rm.T + np.asarray(pose[:3]) / 1000
        actors.append(_mesh_actor(f.to_base(tri)))
    return actors


VIEWS = ('side', 'top', 'front', '3D')


def set_view(renderer, state, view):
    f = state['frame']
    c = f.to_base(np.median(f.to_frame(state['snap']['cloud']), axis=0))[0]
    x, y, z = f.R[:, 0], f.R[:, 1], f.R[:, 2]
    eye, up = {'side': (-y, z), 'top': (z, y), 'front': (-x, z),
               '3D': ((-y - .8 * x + .6 * z) / np.linalg.norm(-y - .8 * x + .6 * z), z)}[view]
    cam = renderer.GetActiveCamera()
    cam.SetFocalPoint(*c)
    cam.SetPosition(*(c + .6 * eye))
    cam.SetViewUp(*up)
    cam.SetParallelProjection(view != '3D')
    renderer.ResetCamera()
    renderer.ResetCameraClippingRange()


def build_state(snapshot, name=None, com_mm=None, shift_mm=(0., 0., 0.), yaw_deg=0., mesh=None,
                mesh_pose=(0., 0., 0., 0., 0., 0.), origin='press', base_w_mm=None):
    folder, episode = resolve_snapshot(snapshot)
    snap = load_snapshot(folder)
    name = name or episode.get('object')
    record = record_for(name)
    if com_mm is None:
        com_mm = record_com_mm(record)
    com_mm = [None if v is None or not math.isfinite(v) else float(v) for v in com_mm]
    if mesh is None and record is not None and record.get('mesh'):
        mesh = configuration_mesh(name)
    if base_w_mm is None and record is not None and record.get('base_w'):
        base_w_mm = 1000 * record['base_w']
    auto, how = auto_origin(snap, origin)
    frame = Frame(snap['direction'], auto, shift_mm, yaw_deg)
    return dict(snapshot=folder, name=name, record=record, snap=snap, frame=frame, com_mm=com_mm,
                origin=how, origin_arg=origin, base_w_mm=base_w_mm,
                mesh=mesh_tools.load_triangles(mesh) if mesh else None, mesh_path=mesh,
                mesh_pose=list(mesh_pose), target='frame', step=1.)


def show(state, png=None):
    vtk, _ = _vtk()
    window = vtk.vtkRenderWindow()
    window.SetWindowName('object frame check')
    text = vtk.vtkTextActor()
    text.GetTextProperty().SetFontSize(15)
    text.GetTextProperty().SetColor(1, 1, 1)
    text.SetPosition(10, 10)
    renderers = []
    viewports = ((0, 0, .5, 1), (.5, 0, 1, 1)) if png else ((0, 0, 1, 1),)
    for vp in viewports:
        r = vtk.vtkRenderer()
        r.SetViewport(*vp)
        r.SetBackground(.12, .12, .14)
        window.AddRenderer(r)
        renderers.append(r)
    clouds = {}
    for r in renderers:
        acts = fixed_actors(state)
        for a in acts:
            r.AddActor(a)
        for key, a in zip(('g', 'o') if state['snap']['scene'] is not None else ('o',), acts):
            clouds.setdefault(key, []).append(a)
    renderers[0].AddActor2D(text)
    live = []

    def refresh():
        for r, acts in live:
            for a in acts:
                r.RemoveActor(a)
        live.clear()
        for r in renderers:
            acts = frame_actors(state)
            for a in acts:
                r.AddActor(a)
            live.append((r, acts))
        text.SetInput(report(state))
        window.Render()

    print(report(state))
    if png:
        window.SetOffScreenRendering(1)
        window.SetSize(1800, 900)
        refresh()
        set_view(renderers[0], state, 'side')
        set_view(renderers[1], state, 'top')
        window.Render()
        grab = vtk.vtkWindowToImageFilter()
        grab.SetInput(window)
        grab.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(png))
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        print(f'wrote {png} (left: side view, right: top view)')
        return
    window.SetSize(1500, 950)
    interactor = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    interactor.SetRenderWindow(window)
    # Robot base_link axes in the lower-right corner; they turn with the camera.
    axes = vtk.vtkAxesActor()
    axes.SetXAxisLabelText('robot x')
    axes.SetYAxisLabelText('robot y')
    axes.SetZAxisLabelText('robot z')
    marker = vtk.vtkOrientationMarkerWidget()
    marker.SetOrientationMarker(axes)
    marker.SetInteractor(interactor)
    marker.SetViewport(.84, 0, 1, .2)
    marker.SetEnabled(1)
    marker.InteractiveOff()
    view = {'i': 0}
    moves = {'Left': (0, -1), 'Right': (0, 1), 'Up': (1, 1), 'Down': (1, -1), 'Prior': (2, 1), 'Next': (2, -1)}

    def on_key(obj, _event):
        key = interactor.GetKeySym()
        f = state['frame']
        if key in moves:
            axis, sign = moves[key]
            if state['target'] == 'frame':
                f.shift_mm[axis] += sign * state['step']
            elif state['target'] == 'CoM':
                com = state['com_mm']
                com[axis] = (com[axis] or 0.) + sign * state['step']
            elif axis == 0:
                fp = footprint(state['snap']['cloud'], f.R, state['snap']['table_z'])
                start = state['base_w_mm'] or 1000 * (fp['far'] - fp['near'])
                state['base_w_mm'] = max(start + sign * state['step'], 1.)
        elif key in ('bracketleft', 'bracketright') and state['target'] == 'frame':
            f.yaw_deg += (1 if key == 'bracketright' else -1) * min(state['step'], 1.)
        elif key == 'm':
            order = ('frame', 'CoM', 'far edge')
            state['target'] = order[(order.index(state['target']) + 1) % len(order)]
        elif key in clouds:
            for a in clouds[key]:
                a.SetVisibility(not a.GetVisibility())
        elif key == 'h':
            text.SetVisibility(not text.GetVisibility())
        elif key == 's':
            from datetime import datetime
            out = Path.cwd() / f"{state['name'] or 'object'}_frame_{datetime.now():%Y%m%d_%H%M%S}.png"
            grab = vtk.vtkWindowToImageFilter()
            grab.SetInput(window)
            grab.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(str(out))
            writer.SetInputConnection(grab.GetOutputPort())
            writer.Write()
            print(f'saved {out}')
        elif key in ('plus', 'equal', 'KP_Add'):
            state['step'] = min(state['step'] * 10, 10.)
        elif key in ('minus', 'KP_Subtract'):
            state['step'] = max(state['step'] / 10, .1)
        elif key == 'v':
            view['i'] = (view['i'] + 1) % len(VIEWS)
            set_view(renderers[0], state, VIEWS[view['i']])
        elif key == 'r':
            renderers[0].ResetCamera()
        elif key in ('Return', 'KP_Enter', 'p') or interactor.GetKeyCode() in ('\r', '\n'):
            print('\n' + report(state) + '\n\n' + snippet(state) + '\n', flush=True)
        elif key in ('q', 'Escape'):
            window.Finalize()
            interactor.TerminateApp()
            return
        else:
            return
        refresh()

    style.AddObserver('KeyPressEvent', on_key)
    style.AddObserver('CharEvent', lambda *_: None)  # replaces VTK's default letter keys
    print('\nLEGEND\n' + '\n'.join(f'{name:<14s} {label}' for name, label in LEGEND_TEXT) + '\n\n' + KEYS, flush=True)
    refresh()
    set_view(renderers[0], state, 'side')
    window.Render()
    interactor.Start()
    print('\n' + snippet(state))
