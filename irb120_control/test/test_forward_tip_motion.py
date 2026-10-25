"""Forward-tip direction, threshold arming and bounded return without hardware."""

import math

import numpy as np
import pytest

from irb120_control.util.forward_tip_motion import TipConfig, TipMotion, slab_boxes, tip_targets
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
    ball, direction, pivot, pre = tip_targets(saved, cloud, clearance='box')
    assert direction @ saved['geometry']['axis'] == pytest.approx(0.)
    assert np.linalg.norm(direction) == pytest.approx(1.)
    assert direction[2] == 0.
    assert pre[2] == ball[2]
    assert (pivot - ball) @ direction > 0
    lo, hi = cloud.min(axis=0) - OBJECT_PADDING, cloud.max(axis=0) + OBJECT_PADDING
    assert pre @ direction <= np.where(direction >= 0, lo, hi) @ direction - BALL_RADIUS - APPROACH_CLEARANCE + 1e-12


def test_shared_cloud_prevents_outliers_in_both_pre_tip_and_collision_slabs():
    """A couple of stray points (sensor noise, a stereo mismatch) far from the real surface
    must not blow out the padded box the pre-tip standoff has to clear. Real surface points are
    never isolated like this -- dense grid here stands in for one, well above the robust-bounds
    neighbor threshold so none of it gets excluded either."""
    grid = np.stack(np.meshgrid(np.arange(.35, .45, .003), np.arange(.05, .15, .003),
                                 [.2]), axis=-1).reshape(-1, 3)
    outliers = np.array([[.05, .1, .2], [.08, .11, .19]])  # isolated, ~300-350 mm from the grid
    saved = contact()
    clean = tip_targets(saved, grid)[3]
    from irb120_perception.object_cloud import prepare_object_cloud
    raw = np.vstack([grid, outliers])
    canonical, audit = prepare_object_cloud(raw, np.full(len(raw), 3))
    with_outliers = tip_targets(saved, canonical['xyz'])[3]
    assert audit['removed_points'] == 2
    assert np.allclose(clean, with_outliers)
    assert min(lo[0] for lo, hi in slab_boxes(canonical['xyz'])) > .3
    # Sanity check the outliers really would have moved it, proving the filter is load-bearing
    # here and not just inactive (e.g. the grid alone already clearing by coincidence).
    lo = outliers.min(axis=0)[0] - OBJECT_PADDING
    naive_clearance = (saved['ball_center'][0] - lo) + BALL_RADIUS + APPROACH_CLEARANCE
    assert naive_clearance > 2 * np.linalg.norm(clean - saved['ball_center'])


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
                                   dict(force_limit=.1), dict(max_travel=.001),
                                   dict(tip_force=.01), dict(probe_travel=.1),
                                   dict(probe_distance=.02), dict(follow_fraction=1.)])
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TipConfig(**kwargs)


def seek(motion, forces, start=.0395, step=.0001, t0=0.):
    for i, force in enumerate(forces):
        motion.step(t0 + .01 * i, start + step * i, force, abs(force))


def test_light_contact_backs_off_as_too_easy():
    motion = TipMotion(TipConfig(), .04, 0.)
    seek(motion, [.09] * 30)  # 3 mm at under tip_force
    assert motion.state == 'RETURN' and motion.failure_reason == 'too_easy'
    assert motion.contact_force == pytest.approx(.09)
    assert motion.contact_progress == pytest.approx(.0399)


def test_brief_touch_under_noise_is_not_too_easy():
    motion = TipMotion(TipConfig(), .04, 0.)
    seek(motion, [.06] * 5 + [0.] * 25)
    assert motion.state == 'SEEK' and motion.failure_reason is None
    seek(motion, [2.] * 5, start=.043, t0=1.)
    assert motion.state == 'TIP' and motion.contact_progress == pytest.approx(.0432)  # median: 3rd of 5


@pytest.mark.parametrize('held,outcome', [(5., 'TIP'), (.01, 'RETURN')])
def test_probe_backs_off_once_and_judges_by_the_force_left(held, outcome):
    motion = armed()
    progress = .0404 + .010
    assert motion.step(.1, progress, 9., 9.) < 0 and motion.state == 'PROBE'
    for i in range(30):
        progress -= .0001
        speed = motion.step(.2 + .01 * i, progress, held, held)
        if motion.state != 'PROBE':
            break
    assert motion.state == outcome and motion.probe['followed'] == (outcome == 'TIP')
    if outcome == 'TIP':
        assert speed > 0
        assert motion.step(1., .0404 + .02, 9., 9.) > 0 and motion.state == 'TIP'  # only once
    else:
        assert motion.failure_reason == 'slide'
        assert motion.slide_force == pytest.approx(9.) and motion.slide_travel == pytest.approx(.010)


def test_seek_runs_faster_than_the_tip_itself():
    config = TipConfig()
    assert config.seek_speed == pytest.approx(1.5 * config.speed)
    motion = TipMotion(config, .04, 0.)
    assert motion.step(0., .0, 0., 0.) == config.seek_speed   # free air
    for i in range(config.stable_samples):                     # touch: load at the tip speed
        speed = motion.step(.01 * (i + 1), .03, .4, .4)
    assert motion.state in ('LOAD', 'TIP') and speed == config.speed


def drill_like():
    """Dense head at the contact height, plus a grip reaching 60 mm further back lower down."""
    def block(x0, x1, z0, z1):
        g = np.mgrid[x0:x1:.004, .07:.13:.004, z0:z1:.004]
        return g.reshape(3, -1).T
    return np.vstack([block(.405, .45, -.02, .23),    # body/head, front face just past the ball
                      block(.345, .405, .10, .15)])   # grip, well below the ball (z .2)


def test_slab_clearance_ignores_a_wide_part_below_the_approach():
    cloud, saved = drill_like(), contact()
    ball = np.array(saved['ball_center'])
    close = tip_targets(saved, cloud, .02)[3]
    whole = tip_targets(saved, cloud, .02, clearance='box')[3]
    # The whole box puts the pre-tip behind the grip; the slices only behind the head.
    assert (ball - whole)[0] > .07
    assert (ball - close)[0] == pytest.approx(ball[0] - (.405 - OBJECT_PADDING) + BALL_RADIUS
                                              + APPROACH_CLEARANCE, abs=.0045)
    # The close pre-tip ball is outside every slab box its corridor crosses.
    for lo, hi in slab_boxes(cloud):
        if lo[2] < ball[2] + BALL_RADIUS and hi[2] > ball[2] - BALL_RADIUS:
            assert close[0] + BALL_RADIUS <= lo[0] + 1e-9


def test_slab_boxes_cover_the_cloud_down_to_the_table():
    cloud = drill_like()
    boxes = slab_boxes(cloud)
    assert boxes[0][0][2] == pytest.approx(-.021)
    for p in cloud[::50]:
        assert any(np.all(p >= lo) and np.all(p <= hi) for lo, hi in boxes)
    grip_slab = [b for b in boxes if b[0][2] < .12 < b[1][2]][0]
    head_slab = [b for b in boxes if b[0][2] < .2 < b[1][2]][0]
    assert grip_slab[0][0] < head_slab[0][0] - .05
