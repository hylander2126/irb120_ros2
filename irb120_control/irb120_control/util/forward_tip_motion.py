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
    contact_force: float = .05    # touch: median of stable_samples fresh samples (noise sd ~.012 N)
    tip_force: float = .30        # weakest tip whose force drop stands clear of the noise
    load_travel: float = .0015    # travel after touch to reach tip_force, else too easy (or a false touch)
    stable_samples: int = 5       # fresh F/T samples, not repeated control ticks
    min_tip_travel: float = .002
    contact_overrun: float = .010
    position_tolerance: float = .001
    phase_timeout: float = 90.
    probe_travel: float = .010    # tip travel at which the finger backs off once to tell tip from slide
    probe_distance: float = .002
    follow_fraction: float = .30  # a tilted object keeps this much of its force on the finger backing off

    def __post_init__(self):
        values = (self.speed, self.max_travel, self.force_limit, self.contact_force,
                  self.tip_force, self.load_travel, self.min_tip_travel, self.contact_overrun,
                  self.position_tolerance, self.phase_timeout, self.probe_travel,
                  self.probe_distance)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('Forward-tip limits and speed must be finite and positive')
        if not 0 < self.reverse_fraction < 1 or not 0 < self.follow_fraction < 1:
            raise ValueError('reverse_fraction and follow_fraction must be between zero and one')
        if (self.force_limit <= self.tip_force or self.tip_force < self.contact_force
                or self.max_travel <= self.min_tip_travel):
            raise ValueError('Force/travel limits must exceed their arming thresholds')
        if not self.probe_distance < self.probe_travel < self.max_travel:
            raise ValueError('The slide probe must back off less than it travelled, within max_travel')
        if not isinstance(self.stable_samples, int) or self.stable_samples < 1:
            raise ValueError('stable_samples must be a positive integer')


class TipMotion:
    """One-dimensional seek, tip, reverse to start; no ROS or force regulation.

    Progress is measured from the actual pre-tip pose along the edge normal.
    A completed cycle means the finger returned after a force-drop reversal;
    object restoration is not certified by this position/force-only controller.

    What the push does to the object is told from force against travel:
      tip       the force falls as the object rotates toward its balance point
                (reverse at reverse_fraction of the peak: 'force_drop', success);
      too easy  contact made, but the force stays under tip_force: the balance point
                is too close (or the object too light) to see the force drop before
                it topples, so back off at once ('too_easy');
      slide     at probe_travel the finger backs off probe_distance. A tilted object
                rests on its edge and follows the finger back under gravity; a slid
                one stays where it is and the force vanishes ('slide'). The slide is
                kept as a friction sample (slide_force, slide_travel).
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
        self.low_count = 0
        self.peak_signal = 0.
        self.reversal_reason = None
        self.failure_reason = None
        self.returned = False
        self.completed = False
        self.contact_force = None  # median force while loading after touch, N
        self.probe = None          # dict(start, before, held, followed) once the probe has run
        self.slide_force = None    # median force while sliding, N
        self.slide_travel = None
        self._window = []          # last stable_samples fresh |force|
        self._load = []            # |force| since touch
        self._tip = []             # (travel, |force|) in TIP before the probe
        self._probe_forces = []

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

    def _median_window(self):
        n = self.config.stable_samples
        return float(np.median(self._window[-n:])) if len(self._window) >= n else 0.

    def step(self, now, progress, force, force_norm, fresh=True, reversal_signal=None):
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
        if fresh:
            self._window = self._window[-(c.stable_samples - 1):] + [abs(force)]
        if self.state == 'SEEK':
            if fresh and self._median_window() >= c.contact_force:
                self.state = 'LOAD'
                self.contact_progress = progress
                self._load = []
            elif progress >= self.expected_contact + c.contact_overrun:
                self.reverse(now, 'contact_not_found')
                return 0.
        if self.state == 'LOAD':
            if fresh:
                self._load.append(abs(force))
                if self._median_window() >= c.tip_force:
                    self.state = 'TIP'
                    self.state_started = now
                    self.contact_force = self.peak_force = self._median_window()
                    if reversal_signal is not None:
                        self.peak_signal = abs(reversal_signal)
                    return c.speed
            if progress - self.contact_progress >= c.load_travel:
                held = float(np.median(self._load)) if self._load else 0.
                if held >= c.contact_force:
                    self.contact_force = held
                    self.reverse(now, 'too_easy')
                    return 0.
                self.state, self.contact_progress, self._window = 'SEEK', None, []  # false touch
        if self.state == 'TIP':
            travel = progress - self.contact_progress
            if fresh:
                self.peak_force = max(self.peak_force, abs(force))
                if self.probe is None:
                    self._tip.append((travel, abs(force)))
                signal = abs(force) if reversal_signal is None else abs(reversal_signal)
                if not math.isfinite(signal):
                    self.stop('nonfinite_feedback')
                    return 0.
                self.peak_signal = (self.peak_force if reversal_signal is None
                                    else max(self.peak_signal, signal))
                low = self.peak_signal > 0 and signal <= c.reverse_fraction * self.peak_signal
                self.low_count = self.low_count + 1 if low and travel >= c.min_tip_travel else 0
                if self.low_count >= c.stable_samples:
                    self.reverse(now, 'force_drop' if reversal_signal is None else 'torque_drop', success=True)
                    return 0.
            if travel >= c.max_travel:
                self.reverse(now, 'travel_limit')
                return 0.
            if self.probe is None and travel >= c.probe_travel:
                recent = [f for t, f in self._tip if t >= travel - c.probe_distance]
                self.probe = dict(start=progress, before=float(np.median(recent)) if recent else 0.,
                                  held=None, followed=None)
                self._probe_forces = []
                self.state = 'PROBE'
                return -c.speed
        if self.state == 'PROBE':
            p = self.probe
            if fresh and progress <= p['start'] - c.probe_distance / 2:
                self._probe_forces.append(abs(force))
            if progress > p['start'] - c.probe_distance:
                return -c.speed
            p['held'] = float(np.median(self._probe_forces)) if self._probe_forces else 0.
            p['followed'] = p['held'] >= max(c.follow_fraction * p['before'], c.contact_force)
            if p['followed']:
                self.state, self.low_count = 'TIP', 0
                return c.speed
            sliding = [f for t, f in self._tip if t >= c.min_tip_travel] or [f for _, f in self._tip]
            self.slide_force = float(np.median(sliding)) if sliding else None
            self.slide_travel = p['start'] - self.contact_progress
            self.reverse(now, 'slide')
            return 0.
        return c.speed
