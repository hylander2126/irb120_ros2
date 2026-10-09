"""Check log compatibility, calibration identifiability and actual frame availability."""

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import xacro

from irb120_control.util import runtime_log_dir as logs
from irb120_control.util.ft_calibration import load_calibration, solve_force_calibration
from irb120_control.util.ft_capture import FTCapture
from irb120_control.util.ft_frames import WRENCH_AXES_FRAME
from irb120_control.monitoring.netft_preprocessor import NetFTPreprocessor


def test_old_and_new_log_shapes_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(logs, 'runtime_log_dir', lambda _: tmp_path)
    wrench = [1., 2., 3., 4., 5., 6., 7.]
    pose = [1., .6, 0., .3, 0., 0., 0., 1.]
    logs.save_ft_pose_log([wrench], [pose], 'unused', 'old', timestamp='1')
    with np.load(tmp_path / 'old_1.npz') as old:
        assert old['fx'].tolist() == [2.]
        assert old['ft_msg_stamp_s'].size == 0
        assert old['ft_px'].size == 0
    # Raw stamps must never be mistaken for an old raw-stream sensor pose.
    new_wrench = wrench + [.4, 0., .3, 0., 0., 0., 1., .99, .98]
    new_pose = pose + [0., 0., 3., .97]
    raw = [1.01, *wrench[1:], .99]
    logs.save_ft_pose_log([new_wrench], [new_pose], 'unused', 'new',
                         ft_raw_log=[raw], timestamp='1')
    with np.load(tmp_path / 'new_1.npz') as new:
        assert new['ft_msg_stamp_s'].tolist() == [.99]
        assert new['ft_tf_stamp_s'].tolist() == [.98]
        assert new['pose_tf_stamp_s'].tolist() == [.97]
        assert new['ft_raw_msg_stamp_s'].tolist() == [.99]
        assert new['fx_raw'].tolist() == [2.]
        assert new['ft_px'].tolist() == [.4]
    logs.save_ft_pose_log([], [], 'unused', 'raw', ft_raw_log=[raw], timestamp='1')
    with np.load(tmp_path / 'raw_1.npz') as saved:
        assert saved['fx_raw'].tolist() == [2.]
        assert saved['ft_px'].size == 0
    logs.save_ft_pose_log([wrench], [pose], 'unused', 'legacy_raw',
                         ft_raw_log=[new_wrench[:14]], timestamp='1')
    with np.load(tmp_path / 'legacy_raw_1.npz') as saved:
        assert saved['ft_px'].tolist() == [.4]
        assert saved['ft_raw_msg_stamp_s'].size == 0


def test_calibration_recovers_known_bias_and_rejects_unobservable_poses(tmp_path):
    bias, gravity = np.array([.2, -.1, .4]), np.array([0., 0., -1.5])
    rotations = Rotation.from_euler('xyz', [[0, 0, 0], [80, 0, 0], [0, 80, 0]], degrees=True).as_matrix()
    poses = [{'R': r, 'raw_f': bias + r.T @ gravity} for r in rotations]
    ez, g, _, residual = solve_force_calibration(poses)
    np.testing.assert_allclose(ez, bias, atol=1e-12)
    np.testing.assert_allclose(g, gravity, atol=1e-12)
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    for invalid in [poses[:2], [poses[0]] * 3]:
        with pytest.raises(ValueError):
            solve_force_calibration(invalid)
    path = tmp_path / 'calibration.json'
    data = dict(schema_version=1, axes_frame=WRENCH_AXES_FRAME,
                electrical_zero=ez.tolist(), g_world=g.tolist(), torque_zero=[0, 0, 0])
    path.write_text(json.dumps(data))
    np.testing.assert_allclose(load_calibration(path)[0], bias)
    data['axes_frame'] = 'root_finger'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='axes frame'):
        load_calibration(path)


def test_axes_frame_exists_in_both_generated_robot_models():
    path = Path(__file__).parents[2] / 'irb120_models/urdf/irb120_with_finger.xacro'
    for mount in ('false', 'true'):
        root = ET.fromstring(xacro.process_file(str(path), mappings={'use_charuco_mount': mount}).toxml())
        assert WRENCH_AXES_FRAME in {link.attrib['name'] for link in root.findall('link')}


def test_gravity_poll_uses_tool_axes_and_rotates_gravity():
    q = Rotation.from_euler('y', 90, degrees=True).as_quat()
    tf = NS(header=NS(stamp=NS(sec=10, nanosec=0)),
            transform=NS(rotation=NS(x=q[0], y=q[1], z=q[2], w=q[3])))
    node = NS(_tf_buffer=Mock(), _safe_now=lambda: 10.1, _warn_throttled=Mock(),
              _g_world=np.array([0., 0., -2.]), _last_tf_success=None,
              _tool_com=np.array([0., 0., .05]),
              _quat_to_rot=NetFTPreprocessor._quat_to_rot)
    node._tf_buffer.lookup_transform.return_value = tf
    NetFTPreprocessor._tf_poll_cb(node)
    assert node._tf_buffer.lookup_transform.call_args.args[1] == WRENCH_AXES_FRAME
    np.testing.assert_allclose(node._g_world_cur_S, [2., 0., 0.], atol=1e-12)
    # Tool weight moment rotates with the tool: c x g_S = (0,0,.05) x (2,0,0).
    np.testing.assert_allclose(node._tool_torque_S, [0., .1, 0.], atol=1e-12)
    assert node._last_tf_success == 10.1
    node._safe_now = lambda: 11.
    NetFTPreprocessor._tf_poll_cb(node)
    assert node._last_tf_success == 10.1
    node._warn_throttled.assert_called_once()


def test_capture_reset_keeps_processing_state_and_clears_raw():
    capture = FTCapture(Mock(), lambda: 5.)
    capture.on_state(NS(data=json.dumps({'tare_active': True})))
    wrench = NS(force=NS(x=1, y=2, z=3), torque=NS(x=4, y=5, z=6))
    capture.on_raw(NS(wrench=wrench, header=NS(frame_id='ft_link', stamp=NS(sec=4, nanosec=500000000))))
    assert capture.raw == [[5., 1, 2, 3, 4, 5, 6, 4.5]]
    capture.reset()
    assert capture.raw == []
    assert capture.metadata()['ft_processing_states'] == [{'tare_active': True}]


def test_torque_calibration_recovers_bias_and_tool_com(tmp_path):
    from irb120_control.util.ft_calibration import solve_torque_calibration
    bias_f, gravity = np.array([.2, -.1, .4]), np.array([0., 0., -1.5])
    t0, com = np.array([.003, -.002, .001]), np.array([.001, -.002, .045])
    rotations = Rotation.from_euler('xyz', [[0, 0, 0], [80, 0, 0], [0, 80, 0]], degrees=True).as_matrix()
    poses = [{'R': r, 'raw_f': bias_f + r.T @ gravity, 'raw_t': t0 + np.cross(com, r.T @ gravity)}
             for r in rotations]
    _, g, _, _ = solve_force_calibration(poses)
    tz, c, _, residual = solve_torque_calibration(poses, g)
    np.testing.assert_allclose(tz, t0, atol=1e-12)
    np.testing.assert_allclose(c, com, atol=1e-12)
    np.testing.assert_allclose(residual, 0., atol=1e-12)
    # Schema 2 carries tool_com; schema 1 files still load with tool_com = 0.
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(dict(schema_version=2, axes_frame=WRENCH_AXES_FRAME,
                                    electrical_zero=bias_f.tolist(), g_world=g.tolist(),
                                    torque_zero=tz.tolist(), tool_com_m=c.tolist())))
    np.testing.assert_allclose(load_calibration(path)[3], com)
    path.write_text(json.dumps(dict(schema_version=1, axes_frame=WRENCH_AXES_FRAME,
                                    electrical_zero=[0, 0, 0], g_world=[0, 0, -1],
                                    torque_zero=[0, 0, 0])))
    np.testing.assert_allclose(load_calibration(path)[3], 0.)
