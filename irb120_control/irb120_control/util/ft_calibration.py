"""Offline, rank-checked F/T calibration and versioned calibration files.

Model, per static pose with tool orientation R (sensor axes -> world):

    raw_f = electrical_zero + R^T g_world
    raw_t = torque_zero + tool_com x (R^T g_world)

The force solve gives the frame-fixed bias and the tool's weight vector. The torque
solve then reuses that weight: the same captures identify the torque bias and the
tool's centre of mass (in sensor axes, about the NetFT's reported moment origin), so
the finger's own weight is removed at any orientation instead of being left for the
session tare to absorb at one orientation only.

Schema 1 files (force only) still load, with tool_com = 0.
"""

import json
from pathlib import Path

from irb120_control.util.ft_frames import WRENCH_AXES_FRAME
import numpy as np


def solve_force_calibration(poses):
    """Solve frame-fixed bias and world-frame gravity; reject ambiguous poses."""
    if len(poses) < 3:
        raise ValueError('Need at least 3 distinct orientations spanning multiple axes')
    a = np.vstack([np.hstack([np.eye(3), np.asarray(p['R']).T]) for p in poses])
    b = np.concatenate([p['raw_f'] for p in poses])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite calibration samples')
    x, _, rank, sv = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6 or sv[0] / sv[-1] > 100:
        raise ValueError('Calibration orientations are rank-deficient or ill-conditioned; '
                         'vary multiple axes')
    return x[:3], x[3:], sv, (b - a @ x).reshape(-1, 3)


def solve_torque_calibration(poses, g_world):
    """Solve the torque bias and tool CoM, given the weight from the force solve."""
    if len(poses) < 3:
        raise ValueError('Need at least 3 distinct orientations spanning multiple axes')
    g_world = np.asarray(g_world, dtype=float)
    a = np.vstack([np.hstack([np.eye(3), -_skew(np.asarray(p['R']).T @ g_world)]) for p in poses])
    b = np.concatenate([p['raw_t'] for p in poses])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite calibration samples')
    x, _, rank, sv = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6 or sv[0] / sv[-1] > 100:
        raise ValueError('Torque calibration is rank-deficient or ill-conditioned; '
                         'vary multiple axes')
    return x[:3], x[3:], sv, (b - a @ x).reshape(-1, 3)


def default_calibration_path() -> Path:
    """Return the reviewed, active calibration: irb120_control/config/ft_calibration.json."""
    source = Path(__file__).resolve().parents[2] / 'config' / 'ft_calibration.json'
    if source.parent.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('irb120_control')) / 'config' / 'ft_calibration.json'


def load_calibration(path):
    """Return (electrical_zero, g_world, torque_zero, tool_com); never silently fall back."""
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') not in (1, 2) or data.get('axes_frame') != WRENCH_AXES_FRAME:
        raise ValueError('Unsupported calibration schema or axes frame')
    keys = ['electrical_zero', 'g_world', 'torque_zero']
    if data['schema_version'] == 2:
        keys.append('tool_com_m')
    vectors = []
    for key in keys:
        vector = np.asarray(data[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f'Invalid calibration vector: {key}')
        vectors.append(vector)
    if len(vectors) == 3:
        vectors.append(np.zeros(3))
    return tuple(vectors)


def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
