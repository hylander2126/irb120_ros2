"""Forward-tip direction, threshold arming and bounded return without hardware."""

import math

import numpy as np
import pytest

from irb120_control.util.forward_tip_motion import TipConfig, TipMotion, tip_targets
from irb120_control.util.push_geometry import APPROACH_CLEARANCE, BALL_RADIUS, OBJECT_PADDING


def contact(angle=0.):
    direction = np.array([math.cos(angle), math.sin(angle), 0.])
    return dict(available=True, ball_center=[.4, .1, .2],
                normal=(-direction + [0., 0., .2]).tolist(),
                geometry=dict(direction=direction.tolist(),
                              axis=np.cross([0., 0., 1.], direction).tolist(),
                              pivot=(np.array([.4, .1, 0.]) + .1 * direction).tolist()))


@pytest.mark.parametrize('angle', [0., .6, 2., -1.])
def test_oblique_edge_not_sloped_face_sets_horizontal_line_and_box_clearance(angle):
    cloud = np.array([[.3, -.1, 0.], [.6, .3, .3]])
    saved = contact(angle)
    ball, direction, pivot, pre = tip_targets(saved, cloud)
    assert direction @ saved['geometry']['axis'] == pytest.approx(0.)
    assert np.linalg.norm(direction) == pytest.approx(1.)
    assert direction[2] == 0.
    assert pre[2] == ball[2]
    assert (pivot - ball) @ direction > 0
    lo, hi = cloud.min(axis=0) - OBJECT_PADDING, cloud.max(axis=0) + OBJECT_PADDING
    assert pre @ direction <= np.where(direction >= 0, lo, hi) @ direction - BALL_RADIUS - APPROACH_CLEARANCE + 1e-12


@pytest.mark.parametrize('field,value', [('axis', [0., 0., 0.]),
                                       ('direction', [0., 0., 1.]),
                                       ('pivot', [float('nan'), 0., 0.])])
def test_invalid_geometry_rejected(field, value):
    saved = contact()
    saved['geometry'][field] = value
    with pytest.raises(ValueError):
        tip_targets(saved, [[.3, 0., 0.], [.6, .2, .3]])


def armed(force=10.):
    motion = TipMotion(TipConfig(), .04, 0.)
    for i in range(5):
        motion.step(.01 * i, .04 + .0001 * i, force, abs(force))
    assert motion.state == 'TIP'
    return motion


def test_zero_before_contact_does_not_reverse():
    motion = TipMotion(TipConfig(), .04, 0.)
    for i in range(20):
        assert motion.step(i * .01, .001 * i, 0., 0.) > 0
    assert motion.state == 'SEEK'
    assert motion.peak_force == 0.


@pytest.mark.parametrize('sign', [-1., 1.])
def test_ten_percent_of_peak_reverses_then_returns_at_fixed_start(sign):
    motion = armed(sign * 10.)
    for i in range(4):
        assert motion.step(.1 + .01 * i, .05, sign * 1., 1.) > 0
    assert motion.step(.15, .05, sign * 1., 1.) == 0.
    assert motion.reversal_reason == 'force_drop'
    assert motion.step(.16, .05, 2., 2.) < 0
    assert motion.step(.3, .0005, 0., 0.) == 0
    assert motion.completed and motion.returned and motion.done


def test_peak_persists_and_threshold_uses_new_peak():
    motion = armed(5.)
    motion.step(.1, .045, 12., 12.)
    for i in range(5):
        motion.step(.2 + .01 * i, .05, 1.1, 1.1)
    assert motion.state == 'RETURN'
    assert motion.peak_force == 12.


def test_repeated_sensor_sample_does_not_count_as_five_low_samples():
    motion = armed()
    motion.step(.1, .05, .5, .5)
    for i in range(10):
        motion.step(.2 + .01 * i, .05, .5, .5, fresh=False)
    assert motion.state == 'TIP'
    assert motion.low_count == 1
    motion.step(.4, .05, 2., 2.)
    assert motion.low_count == 0


def test_noise_spike_cannot_arm_contact_or_pollute_peak():
    motion = TipMotion(TipConfig(), .04, 0.)
    motion.step(.01, .02, 15., 15.)
    motion.step(.02, .02, 0., 0.)
    for i in range(5):
        motion.step(.1 + i * .01, .04, 2., 2.)
    assert motion.peak_force == 2.


def test_initial_force_relaxation_requires_minimum_tip_travel():
    motion = armed()
    for i in range(8):
        motion.step(.1 + i * .01, .041, .5, .5)
    assert motion.state == 'TIP'


@pytest.mark.parametrize('cause', ['travel_limit', 'forward_timeout', 'contact_not_found'])
def test_limits_reverse_but_do_not_report_success(cause):
    motion = TipMotion(TipConfig(), .04, 0.) if cause == 'contact_not_found' else armed()
    if cause == 'travel_limit':
        motion.step(.1, .13, 10., 10.)
    elif cause == 'forward_timeout':
        motion.step(100., .045, 10., 10.)
    else:
        motion.step(.1, .051, 0., 0.)
    assert motion.state == 'RETURN'
    assert motion.failure_reason == cause
    motion.step(100.1 if cause == 'forward_timeout' else .2, 0., 0., 0.)
    assert motion.returned and not motion.completed


def test_resultant_force_limit_stops_even_when_axial_force_is_small():
    motion = armed()
    assert motion.step(.1, .05, .2, 21.) == 0
    assert motion.state == 'STOPPED'
    assert motion.failure_reason == 'force_limit'
    assert not motion.returned


def test_return_timeout_and_nonfinite_feedback_stop():
    motion = armed()
    motion.reverse(.1, 'force_drop', success=True)
    assert motion.step(100., .02, 0., 0.) == 0
    assert motion.failure_reason == 'return_timeout'
    motion = armed()
    assert motion.step(.1, float('nan'), 0., 0.) == 0
    assert motion.failure_reason == 'nonfinite_feedback'


@pytest.mark.parametrize('kwargs', [dict(reverse_fraction=0.), dict(reverse_fraction=1.),
                                   dict(speed=-1.), dict(speed=float('nan')),
                                   dict(force_limit=.1), dict(max_travel=.001)])
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TipConfig(**kwargs)
