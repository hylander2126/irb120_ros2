"""Regression for a recessed contact whose fixed standoff nearly touches the box."""
import numpy as np
import pytest
from irb120_control.util.push_geometry import push_approach, BALL_RADIUS, OBJECT_PADDING


def test_recessed_contact_clears_box_preserving_endpoint():
    ball = np.array([0.649025147, 0.01433493, 0.06046466])
    xyz = np.array([[0.63262463, -0.043, -0.009], [0.70468605, 0.014, 0.177]])
    pre, travel = push_approach(ball, xyz, .04, .08)
    assert np.isclose(xyz[:, 0].min() - OBJECT_PADDING - pre[0] - BALL_RADIUS, .01)
    assert np.isclose(pre[0] + travel, ball[0] - .04 + .08)
    np.testing.assert_allclose(pre[1:], ball[1:])


def test_already_clear_approach_unchanged():
    pre, travel = push_approach([.6, 0, .06], [[.63, 0, 0]], .04, .08)
    np.testing.assert_allclose(pre, [.56, 0, .06])
    assert np.isclose(travel, .08)


def test_push_pitch_clears_table_at_lowest_ball():
    from irb120_control.util.push_geometry import (
        BALL_TO_STACK, STACK_RADIUS, min_push_pitch)
    h, c = .01825, .02
    th = min_push_pitch(h, c)
    assert np.isclose(h + BALL_TO_STACK * np.sin(th) - STACK_RADIUS * np.cos(th), c)
    assert 25 < np.degrees(th) < 32
    assert min_push_pitch(.2, .02) == 0.0   # ball high enough: horizontal is fine


def test_oblique_approach_preserves_line_endpoint_and_box_clearance():
    from irb120_control.util.push_geometry import horizontal_push_direction
    d = horizontal_push_direction([-.96, .26, .1])
    ball = np.array([.65, -.02, .02])
    xyz = np.array([[.60, -.05, 0], [.69, .04, .15]])
    pre, travel = push_approach(ball, xyz, .04, .08, d)
    np.testing.assert_allclose(np.array(pre) + travel*d, ball + .04*d)
    assert pre[0] + BALL_RADIUS <= xyz[:, 0].min() - OBJECT_PADDING - .01 + 1e-12
    assert pre[2] == ball[2]
    np.testing.assert_allclose(np.cross(ball-np.array(pre), d), 0., atol=1e-12)


def test_tool_yaw_tracks_vector_without_changing_pitch_clearance():
    from scipy.spatial.transform import Rotation
    from irb120_control.util.push_geometry import push_orientation
    d = np.array([np.cos(.25), -np.sin(.25), 0.])
    axis = Rotation.from_quat(push_orientation(d, 31.)).apply([1., 0., 0.])
    np.testing.assert_allclose(axis[:2] / np.linalg.norm(axis[:2]), d[:2])
    assert axis[2] == pytest.approx(-np.sin(np.radians(31.)))


@pytest.mark.parametrize('normal', [[0, 0, 0], [-.5, .8, 0], [np.nan, 0, 0], [1, 0, 0]])
def test_invalid_or_unreachable_push_normal_is_rejected(normal):
    from irb120_control.util.push_geometry import horizontal_push_direction
    with pytest.raises(ValueError):
        horizontal_push_direction(normal)


@pytest.mark.parametrize('force,status', [
    ([-1., 0., 0.], 'validated'), ([1., 0., 0.], 'rejected'),
    ([0., 1., 0.], 'rejected'), ([-.1, 0., 0.], 'waiting'),
    ([-21., 0., 0.], 'rejected'), ([np.nan, 0., 0.], 'rejected'),
])
def test_force_check_uses_opposing_reaction_not_force_magnitude_alone(force, status):
    from irb120_control.util.push_geometry import force_push_check
    assert force_push_check(force, [1., 0., 0.])[0] == status


def test_force_check_for_oblique_push_ignores_vertical_direction():
    from irb120_control.util.push_geometry import force_push_check
    d = np.array([.96, -.28, 0.])
    assert force_push_check(-d + [0, 0, 2], d)[0] == 'validated'
