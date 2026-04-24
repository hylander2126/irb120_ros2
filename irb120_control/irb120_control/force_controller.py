"""PI force controller for normal-direction contact regulation.

Computes a Z velocity command that drives measured force toward a reference,
with integral wind-up clamping. Designed to be instantiated once and reused
across multiple motion phases (e.g. PULL and UNPULL).
"""


class PIForceController:
    def __init__(
        self,
        kp: float,
        ki: float,
        force_ref_n: float,
        max_normal_speed: float,
        control_hz: float,
        integral_limit: float = 2.0,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._force_ref = force_ref_n
        self._max_speed = max_normal_speed
        self._dt = 1.0 / control_hz
        self._integral_limit = integral_limit
        self._integral = 0.0

    def reset(self) -> None:
        """Clear integrator state between phases."""
        self._integral = 0.0

    def update(self, force_z: float) -> float:
        """Return a Z velocity command (m/s) given the current normal force.

        Positive output means move toward the surface (increase contact force).
        Caller negates if the surface is in the -Z direction.
        """
        error = self._force_ref - force_z
        self._integral = _clamp(self._integral + error * self._dt, self._integral_limit)
        cmd = self._kp * error + self._ki * self._integral
        return _clamp(cmd, self._max_speed)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
