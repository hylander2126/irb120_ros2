import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from irb120_control.util import perception_snapshot as module


def _msg(stamp_s):
    sec = int(stamp_s)
    return SimpleNamespace(header=SimpleNamespace(
        stamp=SimpleNamespace(sec=sec, nanosec=round((stamp_s - sec) * 1e9))))


class _SpinAborted(Exception):
    """Raised once all subscriptions are set up, before take_snapshot would block on real data."""


class _FakeNode:
    def __init__(self):
        self.callbacks = []

    def create_subscription(self, msg_type, topic, callback, qos):
        # Exactly what rclpy.Subscription.__init__ does; raises the same ValueError
        # as the bug if `callback` is a bound container method like deque.append.
        inspect.signature(callback).bind(object())
        self.callbacks.append(callback)
        return MagicMock()

    def destroy_subscription(self, sub):
        pass

    def get_logger(self):
        return MagicMock()


def test_snapshot_subscriptions_use_inspectable_callbacks(monkeypatch, tmp_path):
    node = _FakeNode()
    stage = MagicMock()
    stage.path = tmp_path
    episode = MagicMock()
    episode.begin_stage.return_value = stage

    monkeypatch.setattr(module, "_set_detector_active", lambda node, active: None)
    monkeypatch.setattr(module.rclpy, "spin_once", MagicMock(side_effect=_SpinAborted))

    with pytest.raises(_SpinAborted):
        module.take_snapshot(node, episode)

    # DBSCAN/cull + SAM mask/color/metadata + cam1 depth = at least 7 subscriptions,
    # each of which must have survived rclpy's own signature check above.
    assert len(node.callbacks) >= 7


def test_closest_depth_frame_prefers_settle_snapshot_over_stale_deque():
    """Regression for the ~1 s RGB/depth mismatch seen in practice: by the time
    the SAM cull + diagnostics wait finishes, the rolling cam1_depths deque may
    only hold frames far from the SAM color frame's timestamp, while the frame
    captured at settle time (images['cam1']['depth']) is actually closer."""
    deque_frames = [_msg(10.9), _msg(11.9), _msg(12.9)]
    settle_frame = _msg(10.02)
    target = 10.0

    closest = module._closest_depth_frame([*deque_frames, settle_frame], target)
    assert closest is settle_frame


def test_closest_depth_frame_uses_deque_when_it_is_the_better_match():
    deque_frames = [_msg(10.01), _msg(11.0)]
    settle_frame = _msg(8.0)
    closest = module._closest_depth_frame([*deque_frames, settle_frame], 10.0)
    assert closest is deque_frames[0]


def test_closest_depth_frame_empty_or_all_none_returns_none():
    assert module._closest_depth_frame([], 10.0) is None
    assert module._closest_depth_frame([None, None], 10.0) is None
