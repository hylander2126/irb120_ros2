"""Press approach: horizontal finger first, pitched down only when the approach does not plan."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

arc_static = pytest.importorskip('irb120_control.arc_static')


def node(results):
    """results: per pitch, (planned_and_moved, execution_attempted)."""
    calls = []

    def approach(pitch, action):
        calls.append(pitch)
        ok, executed = results[pitch]
        if executed:
            action['execution_success'] = ok
        return ok
    fake = SimpleNamespace(_approach_at_pitch=approach, _press_pitch_deg=0.0, get_logger=lambda: Mock())
    return fake, calls


def test_horizontal_first():
    fake, calls = node({0.0: (True, True)})
    assert arc_static.ArcStatic.move_to_pre_squash(fake)
    assert calls == [0.0] and fake._press_pitch_deg == 0.0


def test_pitches_down_until_a_plan_succeeds():
    fake, calls = node({0.0: (False, False), 10.0: (False, False), 20.0: (True, True)})
    assert arc_static.ArcStatic.move_to_pre_squash(fake)
    assert calls == [0.0, 10.0, 20.0] and fake._press_pitch_deg == 20.0


def test_failed_execution_does_not_try_another_pitch():
    fake, calls = node({0.0: (False, True)})
    assert not arc_static.ArcStatic.move_to_pre_squash(fake)
    assert calls == [0.0]


def test_gives_up_after_30_degrees():
    fake, calls = node({p: (False, False) for p in arc_static.PRESS_PITCH_CANDIDATES_DEG})
    assert not arc_static.ArcStatic.move_to_pre_squash(fake)
    assert calls == [0.0, 10.0, 20.0, 30.0]


@pytest.mark.parametrize('pitch', [0.0, 10.0, 30.0])
def test_pitch_tilts_finger_down_and_forces_rotate_to_world(pitch):
    R = Rotation.from_quat(arc_static.press_orientation(pitch))
    finger = R.apply([1., 0., 0.])          # tool +X is the finger axis
    assert finger[2] == pytest.approx(-math.sin(math.radians(pitch)))
    world_force = np.array([1.5, 0.2, -4.0])
    tool = R.inv().apply(world_force)
    fake = SimpleNamespace(_press_pitch_deg=pitch, _now_s=lambda: 0.0, _stall_guard=Mock(),
                           _ft_capture=Mock(), _ft_transformed_log=[])
    msg = SimpleNamespace(wrench=SimpleNamespace(force=SimpleNamespace(x=tool[0], y=tool[1], z=tool[2]),
                                                 torque=SimpleNamespace(x=0., y=0., z=0.)),
                          header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0)))
    try:
        arc_static.ArcStatic._on_wrench(fake, msg)
    except AttributeError:
        pass  # later bookkeeping needs a real node; the force fields are set first
    assert [fake._force_x, fake._force_y, fake._force_z_signed] == pytest.approx(world_force)
