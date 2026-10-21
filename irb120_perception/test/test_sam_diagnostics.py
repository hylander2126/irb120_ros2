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
                           _latest_depth=None,
                           _info=object(), _last_run=None, min_interval=0,
                           base_frame='base_link', tf_buffer=Mock(),
                           _mask_pub=Mock(), _color_pub=Mock(), _depth_pub=Mock(),
                           _metadata_pub=Mock(),
                           _publish_results=Mock(), get_logger=lambda: Mock(),
                           _review_inputs=lambda: None)

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
    assert metadata['depth_paired_at_source'] is False
    assert 'depth_stamp' not in metadata
    node._depth_pub.publish.assert_not_called()


def test_depth_paired_at_the_same_moment_as_color_not_after_the_sam_call(monkeypatch):
    """depth_msg must be read before _cull(), so its skew from color_msg reflects
    two back-to-back topic reads, not however long the SAM inference call takes."""
    import irb120_perception.object_detector_dbscan_sam_cull as module
    source = np.array([[0., 0, 1], [.01, 0, 1], [.02, 0, 1]])
    monkeypatch.setattr(module, 'pointcloud2_to_xyz', lambda msg: source)
    color = Image(height=2, width=3, encoding='rgb8', step=9)
    color.data = bytes(range(18))
    color.header.frame_id = 'cam_optical'
    color.header.stamp.sec, color.header.stamp.nanosec = 12, 345
    depth = Image(height=2, width=3, encoding='16UC1', step=6)
    depth.data = bytes(range(12))
    depth.header.stamp.sec, depth.header.stamp.nanosec = 12, 100_000_345
    cloud = PointCloud2()
    cloud.header.frame_id = 'base_link'
    cloud.header.stamp.sec, cloud.header.stamp.nanosec = 15, 678
    mask = np.array([[True, False, True], [False, True, False]])
    node = SimpleNamespace(_active=True, _predictor=object(), _latest_color=color,
                           _latest_depth=depth,
                           _info=object(), _last_run=None, min_interval=0,
                           base_frame='base_link', tf_buffer=Mock(),
                           _mask_pub=Mock(), _color_pub=Mock(), _depth_pub=Mock(),
                           _metadata_pub=Mock(),
                           _publish_results=Mock(), get_logger=lambda: Mock(),
                           _review_inputs=lambda: None)

    def cull(points, rgb, info, tf):
        node._latest_depth = None  # simulate a stall/drop during the costly SAM call
        node._diagnostic_mask = mask
        return points

    node._cull = cull
    module.DBSCANSAMCuller._cloud_cb(node, cloud)
    saved_depth = node._depth_pub.publish.call_args.args[0]
    metadata = json.loads(node._metadata_pub.publish.call_args.args[0].data)
    assert saved_depth.header.stamp == cloud.header.stamp
    assert saved_depth.data == depth.data
    assert metadata['depth_paired_at_source'] is True
    assert metadata['depth_stamp'] == 12 + 100_000_345e-9
    assert metadata['rgb_depth_delta_s'] == pytest.approx(0.1)


def test_review_mask_from_dbscan_cloud_with_its_own_predictor(monkeypatch):
    """The cam3 review mask is prompted by the DBSCAN cloud's base (not the culled cloud), with
    the review camera's own predictor, alongside the cam1 cull; it carries the cloud
    stamp and cam1's metadata announces it so the snapshot waits."""
    source = np.array([[0., 0, 1], [.01, 0, 1], [.02, 0, 1], [.5, 0, 1]])
    monkeypatch.setattr(module, 'pointcloud2_to_xyz', lambda msg: source)
    color = Image(height=2, width=3, encoding='rgb8', step=9)
    color.data = bytes(range(18))
    review_color = Image(height=2, width=3, encoding='rgb8', step=9)
    review_color.data = bytes(range(1, 19))
    review_color.header.frame_id = 'cam3_optical'
    review_color.header.stamp.sec = 13
    cloud = PointCloud2()
    cloud.header.frame_id = 'base_link'
    cloud.header.stamp.sec, cloud.header.stamp.nanosec = 15, 678
    mask = np.array([[True, False, True], [False, True, False]])
    review = dict(name='cam3', color=review_color, depth=None, info=object(),
                  mask_pub=Mock(), color_pub=Mock(), depth_pub=Mock(), metadata_pub=Mock())
    review_predictor = object()
    order = []
    node = SimpleNamespace(_active=True, _predictor=object(), _latest_color=color, _latest_depth=None,
                           _info=object(), _last_run=None, min_interval=0, _review=review,
                           _review_predictor=review_predictor,
                           base_frame='base_link', tf_buffer=Mock(),
                           _mask_pub=Mock(), _color_pub=Mock(), _depth_pub=Mock(),
                           _metadata_pub=Mock(), get_logger=lambda: Mock(),
                           _publish_results=lambda *a: order.append('cull'))

    def cull(points, rgb, info, tf):
        node._diagnostic_mask = mask
        return points[:3]

    def box_mask(points, rgb, info, tf, predictor=None):
        assert rgb is review_color and predictor is review_predictor
        # A box around the DBSCAN cloud's base (not the culled cloud, not the whole object).
        np.testing.assert_array_equal(points, module.review_prompt_points(source))
        return mask, None, None, None

    node._cull = cull
    node._box_mask = box_mask
    node._review_inputs = lambda: module.DBSCANSAMCuller._review_inputs(node)
    node._publish_review_mask = lambda *a: (order.append('review'),
                                            module.DBSCANSAMCuller._publish_review_mask(node, *a))
    module.DBSCANSAMCuller._cloud_cb(node, cloud)
    assert order == ['cull', 'review']
    assert json.loads(node._metadata_pub.publish.call_args.args[0].data)['review_camera'] == 'cam3'
    saved_mask = review['mask_pub'].publish.call_args.args[0]
    saved_rgb = review['color_pub'].publish.call_args.args[0]
    metadata = json.loads(review['metadata_pub'].publish.call_args.args[0].data)
    assert saved_mask.header.stamp == cloud.header.stamp == saved_rgb.header.stamp
    assert saved_rgb.data == review_color.data and review_color.header.stamp.sec == 13
    assert metadata['camera'] == 'cam3' and metadata['cloud_stamp'] == [15, 678]
    np.testing.assert_array_equal(np.frombuffer(saved_mask.data, np.uint8).reshape(2, 3), mask*255)
    review['depth_pub'].publish.assert_not_called()


def test_review_prompt_box_covers_only_the_base_padded_down_to_the_table():
    tall = np.array([[.6, -.2, -.01], [.75, .2, -.01], [.68, 0., .3], [.65, .1, .02]])
    corners = module.review_prompt_points(tall)
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners[:, 2].min(), -.025)   # 15 mm below the lowest point
    np.testing.assert_allclose(corners[:, 2].max(), .03)     # 4 cm band: the 30 cm top is left out
    np.testing.assert_allclose(corners[:, 0].min(), .57)     # padded 3 cm
    np.testing.assert_allclose(corners[:, 0].max(), .78)
