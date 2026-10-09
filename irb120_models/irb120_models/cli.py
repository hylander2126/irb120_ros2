"""
Create, regenerate and inspect object models (``ros2 run irb120_models object_models``).

    new NAME --mesh PATH [--label TEXT]   start objects/NAME/ from a CAD export
    generate [NAME ...]                  rewrite the URDF and STL from the YAML + mesh
    check [NAME ...]                     exit 1 if any generated file is stale
    show [NAME ...]                      ground truth, plus what the mesh alone implies

``show`` is the sanity check for hand-measured values: it prints the uniform-density
centroid and the com_x / theta* that centroid would give, next to the measured ones.
A hollow or weighted object will legitimately differ, but a sign flip or a
centimetre-scale gap usually means the CoM was entered in the wrong frame.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape

from irb120_models import mesh_tools
from irb120_models.objects import edge_offsets, list_objects, load_object, ObjectModel, \
    objects_dir
import numpy as np

STL_HEADER = 'irb120_models: generated from {src}; do not edit'


def render_urdf(model: ObjectModel) -> str | None:
    """URDF text for ``model``, or None if there is nothing to put in it yet."""
    stl_uri = (f'package://irb120_models/objects/{model.name}/meshes/{model.name}.stl'
               if model.mesh is not None else None)
    inertial = _inertial(model)
    if stl_uri is None and inertial is None:
        return None
    lines = [
        '<?xml version="1.0"?>',
        f'<!-- GENERATED from {model.name}.yaml by `object_models generate`. Do not edit. -->',
        f'<!-- status: {model.status}. Body frame = CAD frame of {model.name}.yaml. -->',
        f'<robot name="{escape(model.name)}">',
        f'  <link name="{escape(model.name)}">',
    ]
    if inertial is not None:
        lines += inertial
    if stl_uri is not None:
        for tag in ('visual', 'collision'):
            lines += [f'    <{tag}>', '      <geometry>',
                      f'        <mesh filename="{stl_uri}"/>', '      </geometry>',
                      f'    </{tag}>']
    lines += ['  </link>', '</robot>', '']
    return '\n'.join(lines)


def _inertial(model: ObjectModel) -> list[str] | None:
    if model.mass.value is None or model.com.value is None:
        return None
    if model.inertia.value is not None:
        inertia, how = np.asarray(model.inertia.value), 'from the YAML'
    elif model.mesh is not None:
        props = mesh_tools.mass_properties(mesh_tools.load_triangles(model.mesh),
                                           model.mass.value)
        inertia, how = props['inertia'], 'uniform-density estimate from the mesh'
    else:
        return None
    x, y, z = model.com.value
    return [
        f'    <!-- inertia: {how} -->',
        '    <inertial>',
        f'      <origin xyz="{x:.6g} {y:.6g} {z:.6g}" rpy="0 0 0"/>',
        f'      <mass value="{model.mass.value:.6g}"/>',
        '      <inertia ixx="{:.6g}" ixy="{:.6g}" ixz="{:.6g}" iyy="{:.6g}" iyz="{:.6g}" '
        'izz="{:.6g}"/>'.format(inertia[0, 0], inertia[0, 1], inertia[0, 2],
                                inertia[1, 1], inertia[1, 2], inertia[2, 2]),
        '    </inertial>',
    ]


def generated_files(model: ObjectModel) -> dict[Path, bytes | None]:
    """Return what ``generate`` would write; None means the file should not exist."""
    out: dict[Path, bytes | None] = {}
    urdf = render_urdf(model)
    out[model.directory / f'{model.name}.urdf'] = urdf.encode() if urdf is not None else None
    stl_path = model.directory / 'meshes' / f'{model.name}.stl'
    if model.mesh is None:
        out[stl_path] = None
    elif model.mesh != stl_path:  # an STL source that *is* meshes/<name>.stl needs nothing
        out[stl_path] = mesh_tools.stl_bytes(mesh_tools.load_triangles(model.mesh),
                                             STL_HEADER.format(src=model.mesh.name))
    return out


def cmd_generate(names: list[str]) -> int:
    for name in names:
        model = load_object(name)
        for path, data in generated_files(model).items():
            if data is None:
                path.unlink(missing_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f'{name}: generated ({model.status})')
    return 0


def cmd_check(names: list[str]) -> int:
    stale = []
    for name in names:
        model = load_object(name)
        for path, data in generated_files(model).items():
            current = path.read_bytes() if path.exists() else None
            if current != data:
                stale.append(str(path))
    for path in stale:
        print(f'stale: {path}')
    if stale:
        print('Run: ros2 run irb120_models object_models generate')
    return 1 if stale else 0


def cmd_new(name: str, mesh: Path, label: str | None) -> int:
    folder = objects_dir() / name
    if folder.exists():
        print(f'{folder} already exists', file=sys.stderr)
        return 1
    if mesh.suffix.lower() not in ('.gltf', '.stl'):
        print('Mesh must be .gltf or .stl, exported in metres', file=sys.stderr)
        return 1
    props = mesh_tools.mass_properties(mesh_tools.load_triangles(mesh))
    (folder / 'meshes').mkdir(parents=True)
    shutil.copy2(mesh, folder / 'meshes' / f'{name}{mesh.suffix.lower()}')
    lo, hi = props['bounds']
    c = props['centroid']
    (folder / f'{name}.yaml').write_text(NEW_TEMPLATE.format(
        name=name, label=label or name, ext=mesh.suffix.lower(), src=mesh.name,
        lo=_fmt(lo), hi=_fmt(hi), vol=props['volume'] * 1e6, c=_fmt(c)))
    print(f'Created {folder}. Fill in mass, com and tip_edges, then run `generate`.')
    return 0


def cmd_show(names: list[str]) -> int:
    for name in names:
        model = load_object(name)
        print(f'\n{name} ({model.label}) status={model.status} frame={model.body_frame}')
        print(f'  mass   {_val(model.mass.value)} kg   ({model.mass.source or "no source"})')
        print(f'  com    {_val(model.com.value)} m (body)   ({model.com.source or "no source"})')
        for edge in model.tip_edges or {'default': None}:
            gt = model.ground_truth(edge)
            print(f'  edge {edge!r}: com_x={_val(gt["com_x"])} com_z={_val(gt["com_z"])} '
                  f'theta*={_val(gt["theta_star_deg"], 2)} deg  [source: {gt["source"]}]'
                  + (f'  CAD theta*={gt["theta_star_cad_deg"]}'
                     if gt['theta_star_cad_deg'] is not None else ''))
        if model.mesh is None:
            print('  mesh   none')
            continue
        props = mesh_tools.mass_properties(mesh_tools.load_triangles(model.mesh))
        c = model.rest_rotation() @ props['centroid'] + model.rest_xyz
        lo, hi = props['bounds']
        print(f'  mesh   {model.mesh.name}: volume {props["volume"] * 1e6:.1f} cm^3, '
              f'bounds {_fmt(lo)} .. {_fmt(hi)}')
        print(f'  uniform-density centroid (support frame) {_fmt(c)}')
        for edge, e in model.tip_edges.items():
            cx, cz = edge_offsets(c, e)
            print(f'    edge {edge!r} if uniform: com_x={cx:.4f} com_z={cz:.4f} '
                  f'theta*={math.degrees(math.atan2(cx, cz)):.2f} deg')
    return 0


def _fmt(v) -> str:
    return '[' + ', '.join(f'{x:.4f}' for x in v) + ']'


def _val(v, nd: int = 4) -> str:
    if v is None:
        return '—'
    if np.ndim(v):
        return _fmt(v)
    return f'{v:.{nd}f}'


NEW_TEMPLATE = """\
# Ground truth for {label}. This file is the source of truth; {name}.urdf and
# meshes/{name}.stl are generated from it (`ros2 run irb120_models object_models generate`).
name: {name}
label: {label}
status: provisional          # 'complete' once mass, com and tip_edges.default are measured
body_frame: cad              # the CAD frame of the mesh below
mesh: meshes/{name}{ext}     # copied from {src}
# Mesh as imported: bounds {lo} .. {hi} m, volume {vol:.1f} cm^3,
# uniform-density centroid {c} m. Use `object_models show {name}` to compare.
rest_pose:                   # body -> support frame (table z = 0, z up), as it rests
  xyz: [0.0, 0.0, 0.0]
  rpy: [0.0, 0.0, 0.0]
mass:
  value: null                # kg
  sigma: null
  source: null               # e.g. "kitchen scale, 2026-10-09"
com:
  value: null                # [x, y, z] in the body frame, m
  sigma: null
  source: null
inertia:
  value: null                # 3x3 about the CoM, body frame; null = uniform-density estimate
  source: null
tip_edges:                   # lines in the support frame the object pivots about
  default:
    point: [0.0, 0.0, 0.0]
    direction: [0.0, 1.0, 0.0]
friction:
  table_mu: null
  source: null
recorded: null
cad: null                    # CAD values if a CAD model exists; informational only
notes: []
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='object_models', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for cmd in ('generate', 'check', 'show'):
        p = sub.add_parser(cmd)
        p.add_argument('names', nargs='*', help='default: all objects')
    p = sub.add_parser('new')
    p.add_argument('name')
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--label')
    args = parser.parse_args(argv)
    if args.command == 'new':
        return cmd_new(args.name, args.mesh, args.label)
    names = args.names or list_objects()
    return {'generate': cmd_generate, 'check': cmd_check, 'show': cmd_show}[args.command](names)


if __name__ == '__main__':
    sys.exit(main())
