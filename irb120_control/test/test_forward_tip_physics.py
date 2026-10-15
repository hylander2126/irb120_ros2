"""Forward tip in closed loop with a quasi-static object: tip, slide or near-topple.

The object stands on its pivot edge; the finger pushes its near face at height H
above the pivot, the face L in front of it (tipT geometry). Quasi-static: the
finger force balances either table friction (slide, while the object is flat) or
gravity about the pivot (tip). Past the balance angle the object falls over.
Numbers are from tipT_c (tip, 0.7 N), tipT_d (slide, 0.55 N), tipT_e (topples
under 0.1 N) and tipT_a (slides, then tips).
"""

import math

import numpy as np
import pytest

from irb120_control.util.forward_tip_motion import TipConfig, TipMotion

G = 9.81
L, H = .121, .175        # face distance in front of the pivot, finger height above it
STIFFNESS = 2000.        # N/m, finger-object contact
CONTACT_AT = .025        # pre-tip pose to the face
DT, LAG_TICKS, SPEED_GAIN = .01, 3, .6  # 100 Hz; Servo lags and runs ~60% of command
NOISE = .012             # N, transformed force noise


class Object:
    def __init__(self, mass, d, z, mu, mu_after=None, sticky_after=None):
        self.mass, self.d, self.z = mass, d, z
        self.mu, self.mu_after, self.sticky_after = mu, mu_after, sticky_after
        self.x = self.theta = 0.
        self.fallen = False
        self.balance = math.atan2(d, z)

    def friction(self):
        late = self.sticky_after is not None and self.x >= self.sticky_after
        return (self.mu_after if late else self.mu) * self.mass * G

    def tip_force(self, theta):
        """Horizontal finger force that holds the object at theta (frictionless face)."""
        return self.mass * G * (self.d * math.cos(theta) - self.z * math.sin(theta)) / L * math.cos(theta)

    def face(self, theta):
        return self.x + L * (1 - 1 / math.cos(theta)) + H * math.tan(theta)

    def push(self, s):
        """Finger at s (from the face's starting position); returns the force on it."""
        if self.fallen:
            return 0.
        force = STIFFNESS * max(0., s - self.face(self.theta))
        if self.theta == 0. and force <= min(self.friction(), self.tip_force(0.)):
            return force
        if self.theta == 0. and self.friction() < self.tip_force(0.):
            self.x = s - self.friction() / STIFFNESS
            return self.friction()

        def excess(theta):  # finger force at theta minus what gravity holds there
            return STIFFNESS * (s - self.face(theta)) - self.tip_force(theta)
        if excess(self.balance) > 0:
            self.fallen = True
            return 0.
        if excess(0.) <= 0:
            self.theta = 0.
            return max(0., STIFFNESS * (s - self.face(0.)))
        lo, hi = 0., self.balance
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
        self.theta = lo
        return self.tip_force(lo)


def run(obj, config=TipConfig(), seed=0):
    rng = np.random.default_rng(seed)
    motion = TipMotion(config, CONTACT_AT, 0.)
    progress, commands, t = 0., [0.] * LAG_TICKS, 0.
    while not motion.done and t < 300:
        force = obj.push(progress - CONTACT_AT) + rng.normal(0, NOISE)
        commands.append(motion.step(t, progress, -force, abs(force)))
        progress += SPEED_GAIN * commands.pop(0) * DT
        t += DT
    return motion


def tipT_c():
    return Object(mass=.5, d=.0167, z=.078, mu=.3)


def test_tip_completes_and_the_probe_sees_the_object_follow():
    obj = tipT_c()
    motion = run(obj)
    assert motion.completed and motion.reversal_reason == 'force_drop'
    assert motion.probe['followed']
    assert not obj.fallen and obj.theta == 0. and obj.x == 0.


def test_slide_is_caught_at_the_probe_and_kept_as_a_friction_sample():
    obj = Object(mass=.5, d=.03, z=.08, mu=.11)  # tipping needs 1.2 N; slides at 0.54 N
    motion = run(obj)
    assert motion.failure_reason == 'slide' and motion.returned and not motion.completed
    assert motion.slide_force == pytest.approx(obj.friction(), abs=.02)
    assert motion.slide_travel == pytest.approx(TipConfig().probe_travel, abs=.001)
    assert not obj.fallen and obj.theta == 0. and obj.x < .012  # tipT_d slid 67 mm


def test_object_close_to_toppling_is_released_before_it_falls():
    obj = Object(mass=.5, d=.0021, z=.08, mu=.3)  # 0.085 N, balance 1.5 deg away
    motion = run(obj)
    assert motion.failure_reason == 'too_easy'
    assert motion.contact_force == pytest.approx(obj.tip_force(0.), abs=.02)
    assert not obj.fallen and obj.theta == 0.


def test_heavy_object_close_to_toppling_tips_by_force_drop_without_falling():
    obj = Object(mass=3., d=.0021, z=.08, mu=.3)  # 0.51 N: enough signal for the force drop
    motion = run(obj)
    assert motion.completed and not obj.fallen


def test_slow_tip_is_not_mistaken_for_a_slide():
    obj = Object(mass=.2, d=.04, z=.1, mu=.6)  # balance 22 deg, ~60 mm of travel
    motion = run(obj)
    assert motion.probe['followed'] and motion.completed and not obj.fallen


def test_slide_then_tip_stops_at_the_slide():
    obj = Object(mass=.5, d=.02, z=.08, mu=.12, mu_after=.3, sticky_after=.035)  # tipT_a
    motion = run(obj)
    assert motion.failure_reason == 'slide' and not obj.fallen


@pytest.mark.parametrize('seed', range(5))
def test_outcomes_hold_across_noise(seed):
    assert run(tipT_c(), seed=seed).completed
    assert run(Object(mass=.5, d=.03, z=.08, mu=.11), seed=seed).failure_reason == 'slide'
    obj = Object(mass=.5, d=.0021, z=.08, mu=.3)
    assert run(obj, seed=seed).failure_reason == 'too_easy' and not obj.fallen


def test_nothing_there_reports_no_contact():
    motion = run(Object(mass=.5, d=.02, z=.08, mu=.3), TipConfig())
    assert motion.contact_progress is not None
    obj = Object(mass=.5, d=.02, z=.08, mu=.3)
    obj.x = 1.  # far away
    motion = run(obj)
    assert motion.reversal_reason == 'contact_not_found' and motion.returned
