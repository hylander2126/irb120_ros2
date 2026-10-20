"""Circular-transition rolling support helpers (from the cylinder fillet audit)."""
import numpy as np

import _paths  # noqa: F401
from irb120_control.estimation.tip_core import pivot_wrench, tilt_noslip


def rolling_ball(phi, r0, radius, finger_radius):
    """Ball trajectory under flat-top rolling on a circular base.

    Coordinates are relative to the initial ground support. Ry(-phi) rotation; the
    circular base centre translates -R*phi along x and stays at z = R.
    """
    x = r0[0] + finger_radius * phi; z = r0[2] - radius
    return np.c_[-radius * phi + x * np.cos(phi) - z * np.sin(phi),
                 radius + x * np.sin(phi) + z * np.cos(phi)]


def rolling_angle(log, pivot, radius, finger_radius):
    k = int(np.argmax(log.contact)); r0 = log.p_ball[k] - pivot
    observed = log.p_ball[:, [0, 2]] - pivot[[0, 2]]
    phi = tilt_noslip(log, pivot).rotvec @ log.tilt_dir
    for _ in range(12):
        x = r0[0] + finger_radius * phi; z = r0[2] - radius
        predicted = rolling_ball(phi, r0, radius, finger_radius)
        dx = -radius + finger_radius * np.cos(phi) - x * np.sin(phi) - z * np.cos(phi)
        dz = finger_radius * np.sin(phi) + x * np.cos(phi) - z * np.sin(phi)
        derivative = np.c_[dx, dz]
        step = np.sum(derivative * (predicted - observed), axis=1) / np.sum(derivative ** 2, axis=1)
        phi -= np.clip(step, -.1, .1)
    residual = np.linalg.norm(rolling_ball(phi, r0, radius, finger_radius) - observed, axis=1)
    return phi, residual


def tau_at(log, pivot):
    """Measured moment about `pivot` (per-sample allowed) along the tilt axis."""
    return pivot_wrench(log, np.zeros_like(log.p_ball), pivot)[:, :3] @ log.tilt_dir
