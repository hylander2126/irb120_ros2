"""Experimental variants; SI values and optional meshes in family YAML files.

These records use a declared experiment frame, independently of CAD registration.
Missing measurements stay None. Family model placeholders are never inherited.
"""
from copy import deepcopy
from pathlib import Path
import math

import yaml

from .objects import objects_dir

NEAR_EDGE_FRAME = 'near_outer_base_edge_x_along_push_z_up'
CENTER_FRAME = 'bottom_center_x_along_push_z_up'


def list_configurations(family=None, *, include_temporary=False):
    """List experiment identifiers, optionally for one family."""
    files = ([objects_dir()/family/'configurations.yaml'] if family else
             sorted(objects_dir().glob('*/configurations.yaml')))
    names = []
    for path in files:
        data = yaml.safe_load(path.read_text())
        names.extend(name for name, record in data['configurations'].items()
                     if include_temporary or not record.get('temporary', False))
    return sorted(names)


def load_configuration(name):
    """Return a validated variant dict, including its family and source YAML.

    mass: kg; com_obj: [x,y,z] or legacy [x,z] m; base_w and fillet_radius: m. The declared frame
    places the origin at the near outer base envelope or at bottom center.
    Neither origin is an inferred rolling support.
    support_distances are optional independently specified active-support-to-CoM
    horizontal distances, keyed by forward/press_pull. Mesh paths are relative
    to the family folder; use configuration_mesh() to obtain an absolute Path.
    """
    for path in sorted(objects_dir().glob('*/configurations.yaml')):
        data = yaml.safe_load(path.read_text())
        if name not in data['configurations']:
            continue
        record = deepcopy(data.get('defaults') or {})
        # Nested defaults (e.g. uncertainties) may be overridden field by field.
        for key, value in data['configurations'][name].items():
            if isinstance(value, dict) and isinstance(record.get(key), dict):
                record[key].update(deepcopy(value))
            else:
                record[key] = deepcopy(value)
        record.update(name=name, family=path.parent.name, source_yaml=str(path))
        _validate(record)
        record['initial_support_distances'] = initial_support_distances(record)
        return record
    raise ValueError(f'Unknown experimental configuration {name!r}')


def _number(value, label, *, positive=False, signed=False):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label}: expected a finite number or null')
    if not signed and (value < 0 or (positive and value == 0)):
        raise ValueError(f'{label}: must be {"positive" if positive else "nonnegative"}')


def _validate(record):
    name = record['name']
    frame = record.get('frame', NEAR_EDGE_FRAME)
    if frame not in (NEAR_EDGE_FRAME, CENTER_FRAME):
        raise ValueError(f'{name}: unknown coordinate frame {frame!r}')
    for key in ('mass', 'base_w', 'fillet_radius'):
        _number(record.get(key), f'{name}.{key}', positive=key!='fillet_radius')
    model = record.get('support_model')
    if model not in (None, 'circular_base_fillet'):
        raise ValueError(f'{name}: unknown support_model {model!r}')
    if model == 'circular_base_fillet':
        width, radius = record.get('base_w'), record.get('fillet_radius')
        if width is not None and radius is not None and radius > width/2:
            raise ValueError(f'{name}: fillet_radius exceeds outer radius base_w/2')
    com = record.get('com_obj')
    if not isinstance(com, list) or len(com) not in (2, 3):
        raise ValueError(f'{name}.com_obj: expected [x,y,z] or [x,z], with null for unknown values')
    axes = ('x', 'z') if len(com)==2 else ('x', 'y', 'z')
    for axis, value in zip(axes, com):
        _number(value, f'{name}.com_obj.{axis}', positive=axis=='z',
                signed=axis=='y' or (axis=='x' and frame==CENTER_FRAME))
    if com[0] is not None and record.get('base_w') is not None:
        outside = (abs(com[0]) > record['base_w']/2 if frame==CENTER_FRAME
                   else com[0] > record['base_w'])
        if outside:
            raise ValueError(f'{name}: com_obj.x is outside the base envelope')
    for mode, value in (record.get('support_distances') or {}).items():
        if mode not in ('forward', 'press_pull'):
            raise ValueError(f'{name}: unknown support distance mode {mode!r}')
        _number(value, f'{name}.support_distances.{mode}')
    uncertainty = record.get('uncertainty') or {}
    for key, value in uncertainty.items():
        if isinstance(value, dict):
            for mode, item in value.items():
                _number(item, f'{name}.uncertainty.{key}.{mode}')
        elif isinstance(value, list):
            if key == 'com_obj' and len(value) not in (2,3):
                raise ValueError(f'{name}.uncertainty.com_obj: expected [sigma_x,sigma_y,sigma_z] or [sigma_x,sigma_z]')
            for item in value:
                _number(item, f'{name}.uncertainty.{key}')
        else:
            _number(value, f'{name}.uncertainty.{key}')
    if record.get('mesh'):
        _mesh_path(record)


def support_closure(record):
    """press_pull + forward - base_w (m), when all three are known; else None.

    Two independent table-edge measurements plus the footprint length should close; the
    residual is a direct check on the planar CoM measurement.
    """
    d = record.get('support_distances') or {}
    if None in (d.get('press_pull'), d.get('forward'), record.get('base_w')):
        return None
    return d['press_pull'] + d['forward'] - record['base_w']


def planar_com(record):
    """Return [x,z] for planar estimation, preserving the stored 2D/3D record."""
    com = record['com_obj']
    return [com[0], com[-1]]


def com_from_near_edge(record):
    """Translate stored [x,z] into the near-envelope frame without modifying it.

    Bottom-center x needs base_w to translate. Unknown dimensions yield an
    unknown x, never a fabricated pivot distance. z is unchanged because both
    origins are at the same table plane.
    """
    x, z = planar_com(record)
    if record.get('frame', NEAR_EDGE_FRAME) == CENTER_FRAME:
        width = record.get('base_w')
        x = x + width/2 if x is not None and width is not None else None
    return [x, z]


def initial_support_distances(record):
    """Horizontal CoM distances from the upright support boundary, in metres.

    For a circular bottom fillet the flat support radius is outer_radius - fillet.
    This is an initial fixed-pivot reference, not a model of subsequent rolling.
    Explicit per-mode measurements override the geometric reference.
    """
    result = dict(forward=None, press_pull=None)
    width, radius = record.get('base_w'), record.get('fillet_radius')
    x = record['com_obj'][0]
    if record.get('support_model') == 'circular_base_fillet':
        if width is not None and radius is not None and x is not None:
            center_x = x if record.get('frame', NEAR_EDGE_FRAME)==CENTER_FRAME else x-width/2
            flat_radius = width/2-radius
            result.update(forward=flat_radius-center_x, press_pull=flat_radius+center_x)
    elif radius == 0:
        near_x = com_from_near_edge(record)[0]
        if near_x is not None:
            result['press_pull'] = near_x
            if width is not None:
                result['forward'] = width-near_x
    for mode, value in (record.get('support_distances') or {}).items():
        if value is not None:
            result[mode] = value
    for mode, value in result.items():
        if value is not None and value < -1e-12:
            raise ValueError(f"{record['name']}: CoM is outside the initial {mode} support")
        if value is not None:
            result[mode] = max(0., value)
    return result


def _mesh_path(record):
    folder = Path(record['source_yaml']).parent.resolve()
    path = (folder/record['mesh']).resolve()
    if not path.is_relative_to(folder):
        raise ValueError(f"{record['name']}: mesh must be inside its family folder")
    if not path.is_file():
        raise ValueError(f"{record['name']}: mesh does not exist: {path}")
    return path


def configuration_mesh(name):
    """Return an optional existing mesh; no geometry is fabricated or required."""
    record = load_configuration(name)
    return _mesh_path(record) if record.get('mesh') else None
