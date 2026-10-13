"""Pure geometry and bounded forward/return motion for the forward-tip primitive."""

from dataclasses import dataclass
import math

import numpy as np

from irb120_control.util.push_geometry import (
    APPROACH_CLEARANCE, BALL_RADIUS, OBJECT_PADDING,
)


TIP_STANDOFF = .02  # ball-centre standoff; tip_targets raises it if needed to clear the padded box


def tip_targets(contact, xyz, standoff=0.04):
    """Return ball, edge-normal direction, pivot and a box-clearing pre-tip point.

    All inputs/outputs are in base_link, metres. The saved geometry direction
    chooses the sign of the horizontal normal to the pivot axis, not the local
    face normal (which may be sloped). Clear the entire padded approach box.
    """
    if not contact.get('available'):
        raise ValueError(f"No forward-tip contact: {contact.get('reason')}")
    geometry = contact.get('geometry') or {}
    ball = np.asarray(contact.get('ball_center'), dtype=float)
    axis = np.asarray(geometry.get('axis'), dtype=float)
    preferred = np.asarray(geometry.get('direction'), dtype=float)
    pivot = np.asarray(geometry.get('pivot'), dtype=float)
    normal = np.asarray(contact.get('normal'), dtype=float)
    cloud = np.asarray(xyz, dtype=float)
    if any(v.shape != (3,) or not np.isfinite(v).all()
           for v in (ball, axis, preferred, pivot, normal)):
        raise ValueError('Forward-tip contact needs finite ball, normal, pivot, axis and direction')
    if (cloud.ndim != 2 or cloud.shape[1] != 3 or not len(cloud)
            or not np.isfinite(cloud).all()):
        raise ValueError('Forward-tip approach needs a finite nonempty Nx3 cloud')
    if (np.linalg.norm(axis) < 1e-9 or np.linalg.norm(preferred) < 1e-9
            or np.linalg.norm(normal) < 1e-9 or not math.isfinite(standoff) or standoff <= 0):
        raise ValueError('Invalid forward-tip direction, normal or standoff')
    axis = axis / np.linalg.norm(axis)
    preferred = preferred / np.linalg.norm(preferred)
    if abs(axis[2]) > 1e-3 or abs(preferred[2]) > 1e-3:
        raise ValueError('Forward-tip pivot axis and direction must be horizontal')
    direction = np.cross(axis, [0., 0., 1.])
    direction /= np.linalg.norm(direction)
    if direction @ preferred < 0:
        direction = -direction
    if direction @ preferred < .99:
        raise ValueError('Saved direction is inconsistent with the pivot edge')
    if direction @ (normal / np.linalg.norm(normal)) >= -.2:
        raise ValueError('Edge-normal motion does not push into the selected face')
    if (pivot - ball) @ direction <= 0:
        raise ValueError('Forward-tip contact must be behind its tipping edge')
    lo = cloud.min(axis=0) - OBJECT_PADDING
    hi = cloud.max(axis=0) + OBJECT_PADDING
    # Minimum projection of the actual axis-aligned collision box, not just
    # the cloud. This also handles diagonally oriented tipping edges.
    box_min = np.where(direction >= 0, lo, hi) @ direction
    clearance = (ball @ direction - box_min) + BALL_RADIUS + APPROACH_CLEARANCE
    pre = ball - max(standoff, float(clearance)) * direction
    return ball, direction, pivot, pre


@dataclass(frozen=True)
class TipConfig:
    speed: float = .005
    max_travel: float = .080       # from first confirmed contact
    reverse_fraction: float = .10
    force_limit: float = 20.
    contact_force: float = .5     # arms the relative threshold above tare noise
    stable_samples: int = 5       # fresh F/T samples, not repeated control ticks
    min_tip_travel: float = .002
    contact_overrun: float = .010
    position_tolerance: float = .001
    phase_timeout: float = 90.

    def __post_init__(self):
        values = (self.speed, self.max_travel, self.force_limit, self.contact_force,
                  self.min_tip_travel, self.contact_overrun,
                  self.position_tolerance, self.phase_timeout)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Forward-tip limits and speed must be finite and positive')
        if not 0 < self.reverse_fraction < 1:
            raise ValueError('reverse_fraction must be between zero and one')
        if self.force_limit <= self.contact_force or self.max_travel <= self.min_tip_travel:
            raise ValueError('Force/travel limits must exceed their arming thresholds')
        if not isinstance(self.stable_samples, int) or self.stable_samples < 1:
            raise ValueError('stable_samples must be a positive integer')


class TipMotion:
    """One-dimensional seek, tip, reverse to start; no ROS or force regulation.

    Progress is measured from the actual pre-tip pose along the edge normal.
    A completed cycle means the finger returned after a force-drop reversal;
    object restoration is not certified by this position/force-only controller.
    """

    def __init__(self, config, expected_contact, now):
        if not math.isfinite(expected_contact) or expected_contact <= 0:
            raise ValueError('Expected contact must be ahead of the pre-tip pose')
        self.config = config
        self.expected_contact = expected_contact
        self.state = 'SEEK'
        self.state_started = now
        self.contact_progress = None
        self.peak_force = 0.
        self.contact_count = 0
        self.low_count = 0
        self.reversal_reason = None
        self.failure_reason = None
        self.returned = False
        self.completed = False

    @property
    def done(self):
        return self.state in ('DONE', 'STOPPED')

    def stop(self, reason):
        self.failure_reason = reason
        self.state = 'STOPPED'

    def reverse(self, now, reason, success=False):
        self.reversal_reason = reason
        if not success:
            self.failure_reason = reason
        self.state = 'RETURN'
        self.state_started = now

    def step(self, now, progress, force, force_norm, fresh=True):
        """Return signed speed; caller enforces fresh sensors and pose tolerances."""
        c = self.config
        if self.done:
            return 0.
        if not all(math.isfinite(v) for v in (now, progress, force, force_norm)):
            self.stop('nonfinite_feedback')
            return 0.
        if force_norm >= c.force_limit:
            self.stop('force_limit')
            return 0.
        if now - self.state_started >= c.phase_timeout:
            if self.state == 'RETURN':
                self.stop('return_timeout')
            else:
                self.reverse(now, 'forward_timeout')
            return 0.
        if self.state == 'RETURN':
            if progress <= c.position_tolerance:
                self.returned = True
                self.completed = self.failure_reason is None
                self.state = 'DONE'
                return 0.
            return -min(c.speed, 2. * progress)
        if self.state == 'SEEK':
            if fresh:
                if abs(force) >= c.contact_force:
                    if self.contact_count == 0:
                        self.contact_progress = progress
                    self.contact_count += 1
                    self.peak_force = max(self.peak_force, abs(force))
                else:
                    self.contact_count = 0
                    self.peak_force = 0.
                    self.contact_progress = None
                if self.contact_count >= c.stable_samples:
                    self.state = 'TIP'
                    self.state_started = now
            if self.state == 'SEEK' and progress >= self.expected_contact + c.contact_overrun:
                self.reverse(now, 'contact_not_found')
                return 0.
        if self.state == 'TIP':
            travel = progress - self.contact_progress
            if fresh:
                self.peak_force = max(self.peak_force, abs(force))
                low = abs(force) <= c.reverse_fraction * self.peak_force
                self.low_count = self.low_count + 1 if low and travel >= c.min_tip_travel else 0
                if self.low_count >= c.stable_samples:
                    self.reverse(now, 'force_drop', success=True)
                    return 0.
            if travel >= c.max_travel:
                self.reverse(now, 'travel_limit')
                return 0.
        return c.speed
