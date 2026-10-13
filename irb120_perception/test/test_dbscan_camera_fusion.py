"""The DBSCAN detector fuses only when every camera is fresh, and removes each
camera's table with its own fixed plane. No ROS graph or cameras needed."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import PointCloud2
from irb120_perception import object_detector_dbscan as module


def _node(now):
    status = Mock()
    node = SimpleNamespace(
        _active=True, base_frame='base_link', tf_buffer=Mock(), get_logger=lambda: Mock(),
        roi={'x': (0., 1.), 'y': (-1., 1.), 'z': (-.05, 1.)},
        _cameras=['cam1', 'cam2', 'cam3'], _camera_clouds={}, max_camera_age=1.,
        _table_planes={'cam1': np.array([0., 0., -.021]), 'cam2': np.array([0., 0., -.012]),
                       'cam3': np.array([0., 0., -.018])},
        table_clearance=.006, _activated_at=now - 10., _status_pub=status,
        _accumulator=None, min_pts=1, _segment_dbscan=Mock(return_value=[]),
        _reset_smoothing=Mock(), _publish_empty=Mock(), _publish_results=Mock())
    node._not_fusing = lambda reason: module.DBSCANObjectDetector._not_fusing(node, reason)
    return node


def _cloud(frame):
    msg = PointCloud2()
    msg.header.frame_id = frame
    return msg


@pytest.fixture
def clock(monkeypatch):
    now = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module, 'apply_tf', lambda pts, tf: pts)
    return now


def test_missing_camera_blocks_fusion_and_is_reported(monkeypatch, clock):
    monkeypatch.setattr(module, 'pointcloud2_to_xyz', lambda msg: np.array([[.5, 0., .05]]))
    node = _node(clock[0])
    module.DBSCANObjectDetector._cloud_cb(node, _cloud('c1'), 'cam1')
    module.DBSCANObjectDetector._cloud_cb(node, _cloud('c2'), 'cam2')
    node._segment_dbscan.assert_not_called()
    assert 'cam3: no cloud since activation' in node._status_pub.publish.call_args.args[0].data

    clock[0] += 2.  # cam1/cam2 now stale when cam3 arrives
    module.DBSCANObjectDetector._cloud_cb(node, _cloud('c3'), 'cam3')
    node._segment_dbscan.assert_not_called()
    reason = node._status_pub.publish.call_args.args[0].data
    assert 'cam1: last cloud 2.0 s old' in reason and 'cam2: last cloud 2.0 s old' in reason


def test_tf_failure_drops_that_camera(monkeypatch, clock):
    monkeypatch.setattr(module, 'pointcloud2_to_xyz', lambda msg: np.array([[.5, 0., .05]]))
    node = _node(clock[0])
    for cam in ('cam1', 'cam2', 'cam3'):
        module.DBSCANObjectDetector._cloud_cb(node, _cloud(cam), cam)
    assert node._segment_dbscan.call_count == 1
    node.tf_buffer.lookup_transform.side_effect = RuntimeError('no realsense2_link')
    module.DBSCANObjectDetector._cloud_cb(node, _cloud('cam2_optical'), 'cam2')
    assert 'cam2' not in node._camera_clouds
    assert 'no realsense2_link' in node._status_pub.publish.call_args.args[0].data
    node.tf_buffer.lookup_transform.side_effect = None
    module.DBSCANObjectDetector._cloud_cb(node, _cloud('c1'), 'cam1')
    assert node._segment_dbscan.call_count == 1  # still not fusing without cam2


def test_each_camera_loses_its_own_table(monkeypatch, clock):
    # Every camera sees the same points: table-ish ones at -16 mm and -8 mm, object at +50 mm.
    pts = np.array([[.5, 0., -.016], [.5, 0., -.008], [.5, 0., .05]])
    monkeypatch.setattr(module, 'pointcloud2_to_xyz', lambda msg: pts)
    node = _node(clock[0])
    for cam in ('cam1', 'cam2', 'cam3'):
        module.DBSCANObjectDetector._cloud_cb(node, _cloud(cam), cam)
    kept = {cam: node._camera_clouds[cam][1][:, 2].tolist() for cam in node._cameras}
    assert kept['cam1'] == [-.008, .05]  # table -21 mm: removed up to -15 mm
    assert kept['cam2'] == [.05]         # table -12 mm: removed up to -6 mm
    assert kept['cam3'] == [-.008, .05]  # table -18 mm: removed up to -12 mm
    fused = node._segment_dbscan.call_args.args[0]
    assert len(fused) == 5  # fused only once the third camera arrived
