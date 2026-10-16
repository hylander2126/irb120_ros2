"""Regression for a recessed contact whose fixed standoff nearly touches the box."""
import numpy as np
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
