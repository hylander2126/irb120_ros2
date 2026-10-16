"""Verify push clearance settings without robot motion."""
import pytest

from irb120_control import push
from irb120_control.util.object_obstacle import TABLE_Z


def test_ball_clearance_is_ten_mm_and_original_wrist_clearance_is_preserved():
    assert push.PUSH_MIN_BALL_CENTER_Z - .01325 - TABLE_Z == pytest.approx(.010)
    assert push.PUSH_TABLE_CLEARANCE == .025
