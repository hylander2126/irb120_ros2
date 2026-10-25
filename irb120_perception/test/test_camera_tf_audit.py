"""Prevent stale calibration and missing/double cam2 correction in real captures."""
import numpy as np
import pytest

from irb120_perception.camera_tf_audit import audit_camera_pose, resolve_cam2_z_offset


REFERENCE = dict(path='cam2_calibration.launch.py', parent='base_link', child='realsense2_link',
                 translation=[.63, .32, .797], quaternion=[0., 0., 0., 1.])


@pytest.mark.parametrize('captured_mm', [0., 3., 6., 9.])
def test_total_correction_is_applied_exactly_once(captured_mm):
    t = np.array(REFERENCE['translation']) + [0., 0., captured_mm/1000]
    audit = audit_camera_pose('cam2', np.eye(3), t, REFERENCE)
    extra, report = resolve_cam2_z_offset(dict(tf_audit=audit))
    assert extra == pytest.approx((4-captured_mm)/1000)
    assert report['effective_total_offset_m'] == pytest.approx(.004)


def test_old_xy_calibration_or_conflicting_publisher_is_rejected():
    old_t = np.array(REFERENCE['translation']) + [.0014, -.0005, -.0023]
    with pytest.raises(ValueError, match='configured calibration'):
        audit_camera_pose('cam2', np.eye(3), old_t, REFERENCE)
    shifted_t = np.array(REFERENCE['translation']) + [0., 0., .006]
    old_static = dict(parent='base_link', translation=REFERENCE['translation'],
                      quaternion=REFERENCE['quaternion'])
    with pytest.raises(ValueError, match='conflicting static TF'):
        audit_camera_pose('cam2', np.eye(3), shifted_t, REFERENCE, [old_static])


def test_legacy_replay_requires_explicit_offset():
    with pytest.raises(ValueError, match='legacy replay'):
        resolve_cam2_z_offset({})
    extra, report = resolve_cam2_z_offset({}, .006)
    assert extra == .006
    assert report['effective_total_offset_m'] is None
    with pytest.raises(ValueError, match='finite'):
        resolve_cam2_z_offset({}, float('nan'))


def test_other_cameras_cannot_hide_z_calibration_changes():
    with pytest.raises(ValueError, match='configured calibration'):
        audit_camera_pose('cam1', np.eye(3), [.63, .32, .803], REFERENCE)
