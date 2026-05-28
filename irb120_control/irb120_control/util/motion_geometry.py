"""Small geometry helpers shared by motion controllers."""

from __future__ import annotations

import math


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def quat_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract Y-axis pitch from a quaternion, in radians."""
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    return math.asin(clamp(sin_pitch, 1.0))


def arc_angle_xz(x: float, z: float, center_x: float, center_z: float = 0.0) -> float:
    """Angle from +Z toward +X for an XZ-plane arc."""
    return math.atan2(x - center_x, z - center_z)


def arc_velocity_xz(
    theta: float,
    tangential_speed: float,
    radial_speed: float,
) -> tuple[float, float]:
    """Return (vx, vz) for an XZ arc.

    Positive tangential speed follows decreasing theta: toward -X for theta
    near zero. Positive radial speed moves outward from the arc center.
    """
    tangent_x = -math.cos(theta)
    tangent_z = math.sin(theta)
    radial_x = math.sin(theta)
    radial_z = math.cos(theta)
    vx = tangential_speed * tangent_x + radial_speed * radial_x
    vz = tangential_speed * tangent_z + radial_speed * radial_z
    return vx, vz


def radial_force_xz(theta: float, force_x: float, force_z: float) -> float:
    """Magnitude of force projected onto an XZ arc's outward radial direction."""
    return abs(force_x * math.sin(theta) + force_z * math.cos(theta))
