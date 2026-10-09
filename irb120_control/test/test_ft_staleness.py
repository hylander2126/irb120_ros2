"""The FSM must not act on a frozen F/T reading: a stale stream retracts without retry."""
from types import SimpleNamespace
from unittest.mock import Mock

from irb120_control.arc_static import ArcStatic, FT_STALE_SEC
from irb120_control.util.stall_guard import StallGuard
import pytest


def fake(state, last_wrench_s, now=100.0, late=False):
    n = SimpleNamespace(
        _done=False, _state=state, _last_wrench_s=last_wrench_s, _force_z=0.4,
        _retry_requested=True, _abort_reason=None, _servo_cmd=Mock(),
        _transition=Mock(), _lookup_pose=Mock(return_value=None),
        _now_s=lambda: now, get_logger=Mock(return_value=Mock()),
        _stall_guard=SimpleNamespace(tick=lambda: late))
    n._wrench_stale = lambda: ArcStatic._wrench_stale(n)
    return n


@pytest.mark.parametrize('state', ['SQUASH', 'LULL', 'ARC', 'UNARC'])
@pytest.mark.parametrize('last', [None, 100.0 - FT_STALE_SEC - 0.01])
def test_stale_wrench_retracts_without_retry(state, last):
    n = fake(state, last)
    ArcStatic._tick(n)
    n._transition.assert_called_once_with('RETRACT')
    n._servo_cmd.publish_zero.assert_called_once()
    n._servo_cmd.publish_twist.assert_not_called()
    n._lookup_pose.assert_not_called()
    assert n._retry_requested is False and n._abort_reason == 'ft_stale'


def test_fresh_wrench_proceeds_to_the_state_machine():
    n = fake('ARC', 100.0 - FT_STALE_SEC / 2)
    ArcStatic._tick(n)
    n._lookup_pose.assert_called_once()  # got past the guard
    n._transition.assert_not_called()


def test_retract_runs_even_when_stale():
    n = fake('RETRACT', None)
    ArcStatic._tick(n)
    n._lookup_pose.assert_called_once()
    assert n._abort_reason is None


def test_late_tick_holds_instead_of_retracting():
    """2026-10-09: arc_static froze 100 ms; the tick ran before the queued wrenches."""
    n = fake('UNARC', 100.0 - 0.104, late=True)
    ArcStatic._tick(n)
    n._transition.assert_not_called()
    n._servo_cmd.publish_zero.assert_called_once()
    n._lookup_pose.assert_not_called()
    assert n._abort_reason is None


def test_on_time_tick_after_a_late_one_still_retracts():
    n = fake('UNARC', 100.0 - 0.104)
    ArcStatic._tick(n)
    n._transition.assert_called_once_with('RETRACT')


def test_late_tick_retracts_on_wrench_older_than_the_defer_cap():
    n = fake('ARC', 100.0 - 0.6, late=True)
    ArcStatic._tick(n)
    n._transition.assert_called_once_with('RETRACT')


def test_stall_guard_flags_only_a_late_tick(monkeypatch):
    clock = iter([0.0, 0.010, 0.120, 0.130])
    monkeypatch.setattr('irb120_control.util.stall_guard.time.monotonic', lambda: next(clock))
    guard = StallGuard()
    assert [guard.tick() for _ in range(4)] == [False, False, True, False]
