"""Exercise the real culler callback without cameras, a ROS graph, or SAM weights."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip('rclpy')
from sensor_msgs.msg import Image, PointCloud2
from irb120_perception import object_detector_dbscan_sam_cull as module


def test_diagnostics_keep_exact_rgb_and_cloud_association(monkeypatch):
    source = np.array([[0., 0, 1], [.01, 0, 1], [.02, 0, 1]])
    monkeypatch.setattr(module, 'pointcloud2_to_xyz', lambda msg: source)
    color = Image(height=2, width=3, encoding='rgb8', step=9)
    color.data = bytes(range(18))
    color.header.frame_id = 'cam_optical'
    color.header.stamp.sec = 12
    color.header.stamp.nanosec = 345
    cloud = PointCloud2()
    cloud.header.frame_id = 'base_link'
    cloud.header.stamp.sec = 15
    cloud.header.stamp.nanosec = 678
    mask = np.array([[True, False, True], [False, True, False]])
    node = SimpleNamespace(_active=True, _predictor=object(), _latest_color=color,
                           _info=object(), _last_run=None, min_interval=0,
                           base_frame='base_link', tf_buffer=Mock(),
                           _mask_pub=Mock(), _color_pub=Mock(), _metadata_pub=Mock(),
                           _publish_results=Mock(), get_logger=lambda: Mock())

    def cull(points, rgb, info, tf):
        assert rgb is color
        node._diagnostic_mask = mask
        return points

    node._cull = cull
    module.DBSCANSAMCuller._cloud_cb(node, cloud)
    saved_mask = node._mask_pub.publish.call_args.args[0]
    saved_rgb = node._color_pub.publish.call_args.args[0]
    metadata = json.loads(node._metadata_pub.publish.call_args.args[0].data)
    assert saved_mask.header.stamp == cloud.header.stamp == saved_rgb.header.stamp
    assert saved_rgb.data == color.data
    assert color.header.stamp.sec == 12  # original acquisition header not mutated
    assert metadata['color_stamp'] == 12+345e-9
    assert metadata['cloud_stamp'] == [15, 678]
    np.testing.assert_array_equal(np.frombuffer(saved_mask.data, np.uint8).reshape(2, 3), mask*255)
    np.testing.assert_array_equal(node._publish_results.call_args.args[1][0], source)
