import numpy as np
import pytest
from irb120_perception.press_targets import press_targets, pre_press_center, build_press_markers


def press():
    return dict(available=True, point=[.56, -.005, .153], ball_center=[.56, -.005, .16625],
                geometry=dict(pivot=[.614, -.0035, -.021], axis=[0., -1., 0.]))


def test_preview_and_world_motion_targets_agree_with_base_mount():
    saved = press()
    preview = press_targets(saved)
    motion = press_targets(saved, np.eye(3), [0, 0, .021])
    np.testing.assert_allclose(preview['pre_press'], pre_press_center(saved['ball_center']))
    for key in ['contact', 'ball', 'pivot', 'pre_press']:
        np.testing.assert_allclose(motion[key], preview[key]+[0, 0, .021])
    np.testing.assert_allclose(motion['pre_press'], [.56, -.005, .21725])
    assert saved == press()


def test_no_contact_cannot_be_previewed_or_executed():
    with pytest.raises(ValueError):
        press_targets(dict(available=False))


def test_rviz_spheres_match_saved_contact_and_commanded_prepress():
    pytest.importorskip('visualization_msgs.msg')
    from builtin_interfaces.msg import Time
    targets = press_targets(press())
    markers = build_press_markers(targets, 'base_link', Time())
    for marker, key in zip(markers.markers, ['contact', 'ball', 'pre_press']):
        p = marker.pose.position
        np.testing.assert_allclose([p.x, p.y, p.z], targets[key])
        assert marker.header.frame_id == 'base_link'
    assert markers.markers[2].scale.x == pytest.approx(.0265)
