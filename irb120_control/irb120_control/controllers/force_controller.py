"""PID force controller for normal-direction contact regulation.

Computes a velocity command that drives measured force toward a reference,
with integral wind-up clamping and derivative damping on force error.
Designed to be instantiated once and reused across multiple motion phases.
"""


class PIDForceController:
    def __init__(
        self,
        kp: float,
        ki: float,
        force_ref_n: float,
        max_normal_speed: float,
        control_hz: float,
        integral_limit: float = 2.0,
        kd: float = 0.0,
        deadband_n: float = 0.0,
        measurement_filter_alpha: float = 1.0,
        output_slew_rate: float | None = None,
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._force_ref = force_ref_n
        self._max_speed = max_normal_speed
        self._dt = 1.0 / control_hz
        self._integral_limit = integral_limit
        self._deadband_n = deadband_n
        self._measurement_filter_alpha = measurement_filter_alpha
        self._output_slew_step = output_slew_rate * self._dt if output_slew_rate is not None else None
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_force: float | None = None
        self._prev_output = 0.0

    def reset(self) -> None:
        """Clear integrator and derivative state between phases."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_force = None
        self._prev_output = 0.0

    def set_reference(self, force_ref_n: float) -> None:
        """Update the force setpoint without disturbing the integrator."""
        self._force_ref = force_ref_n

    @property
    def reference(self) -> float:
        return self._force_ref

    def update(self, force: float) -> float:
        """Return a velocity command (m/s) given the current normal force.

        Positive output means move toward the surface (increase contact force).
        Caller negates if the surface is in the outward direction.
        """
        if self._filtered_force is None:
            self._filtered_force = force
        else:
            alpha = max(0.0, min(1.0, self._measurement_filter_alpha))
            self._filtered_force += alpha * (force - self._filtered_force)

        error = self._force_ref - self._filtered_force
        if abs(error) < self._deadband_n:
            error = 0.0
            self._integral = 0.0
        self._integral = _clamp(self._integral + error * self._dt, self._integral_limit)
        derivative = (error - self._prev_error) / self._dt
        self._prev_error = error
        cmd = self._kp * error + self._ki * self._integral + self._kd * derivative
        cmd = _clamp(cmd, self._max_speed)
        if self._output_slew_step is not None:
            cmd = self._prev_output + _clamp(cmd - self._prev_output, self._output_slew_step)
        self._prev_output = cmd
        return cmd


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
