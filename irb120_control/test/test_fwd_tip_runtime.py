"""Exercise actual ROS-node methods with fake I/O; never connect to the robot."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from irb120_control.util.ft_frames import WRENCH_AXES_FRAME, WRENCH_HEADER_FRAME
import numpy as np
import pytest

from irb120_control.util.forward_tip_motion import TipConfig, TipMotion

SOURCE = Path(__file__).parents[1] / 'irb120_control' / 'fwd_tip.py'


def methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ForwardTip')
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rotate']
    keep += [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ('_tick', '_on_wrench')]
    env = dict(np=np, math=math, time=SimpleNamespace(monotonic=lambda: 1.),
               TransformException=LookupError, WrenchStamped=object,
               BASE_FRAME='world', EE_LINK='finger_ball_center', SENSOR_TIMEOUT_SEC=.3,
               LINE_TOLERANCE_M=.003, ORIENTATION_TOLERANCE_RAD=math.radians(3.),
               WRENCH_AXES_FRAME=WRENCH_AXES_FRAME, WRENCH_HEADER_FRAME=WRENCH_HEADER_FRAME)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SOURCE), 'exec'), env)
    return env


def fake_node():
    node = Mock()
    node.config = TipConfig()
    node.motion = TipMotion(node.config, .04, 0.)
    node._force_received = .99
    node._start_pose = np.array([.4, 0., .2, 0., 0., 0., 1.])
    node._transform.return_value = node._start_pose.copy()
    node._direction = np.array([.6, .8, 0.])
    node._world_from_servo = np.eye(3)
    node._force = np.array([0., 0., 0.])
    node._force_serial = 1
    node._used_serial = 0
    node._last_report = 0.
    node._pose_log = []
    node._motion_log = []
    node._now_s.return_value = 1.
    return node


def test_tick_commands_edge_normal_with_zero_height_and_rotation():
    node = fake_node()
    methods()['_tick'](node)
    args = node._servo_cmd.publish_twist.call_args.args
    np.testing.assert_allclose(args, [.003, .004, 0.])
    assert 'wy' not in node._servo_cmd.publish_twist.call_args.kwargs
    node._stop.assert_not_called()


@pytest.mark.parametrize('failure', ['force', 'tf', 'height', 'orientation'])
def test_tick_stops_on_stale_data_or_pose_drift(failure):
    node = fake_node()
    if failure == 'force':
        node._force_received = .1
    elif failure == 'tf':
        node._transform.side_effect = LookupError('missing')
    elif failure == 'height':
        node._transform.return_value[2] += .004
    else:
        node._transform.return_value[3:] = [0., math.sin(.1), 0., math.cos(.1)]
    methods()['_tick'](node)
    node._stop.assert_called_once()
    node._servo_cmd.publish_twist.assert_not_called()


def test_wrench_is_rotated_from_its_declared_frame_before_projection():
    node = fake_node()
    node._transform.return_value = np.array([0., 0., 0., 0., 0., math.sqrt(.5), math.sqrt(.5)])
    node._recording = True
    node._ft_log = []
    vec = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    msg = SimpleNamespace(
        header=SimpleNamespace(frame_id='ft_link', stamp=SimpleNamespace(sec=1, nanosec=0)),
        wrench=SimpleNamespace(force=vec(3., 0., 0.), torque=vec(0., 2., 0.)))
    methods()['_on_wrench'](node, msg)
    np.testing.assert_allclose(node._force, [0., 3., 0.], atol=1e-12)
    np.testing.assert_allclose(node._ft_log[0][4:7], [-2., 0., 0.], atol=1e-12)
    # The 'ft_link' header label must resolve to the real axes frame, which exists in TF.
    assert [c.args[0] for c in node._transform.call_args_list] == ['tool0']
    assert node._force_serial == 2


@pytest.mark.parametrize('returned,paused,home_result', [(True, True, True), (False, True, True),
                                                       (True, False, True), (True, True, False)])
def test_cleanup_requires_verified_return_and_pause_before_home(returned, paused, home_result):
    tree = ast.parse(SOURCE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    outer = next(n for n in main.body if isinstance(n, ast.Try) and n.finalbody)
    cleanup = outer.finalbody[0]  # Stop, gate home, always deactivate EGM and close publisher.
    fn = ast.parse('def cleanup():\n    pass').body[0]
    fn.body = [ast.parse('home_ok = False').body[0], cleanup,
               ast.Return(value=ast.Name(id='home_ok', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    node = Mock()
    node.motion.returned = returned
    node.pre_ready = True
    node.motion_started = True
    node.egm_requested = True
    node.set_servo_paused.return_value = paused
    node.home.return_value = home_result
    stop_egm = Mock()
    env = dict(node=node, rclpy=SimpleNamespace(ok=lambda: True), home_ok=False,
               deactivate_egm=stop_egm)
    exec(compile(module, str(SOURCE), 'exec'), env)
    assert env['cleanup']() == (returned and paused and home_result)
    assert node.home.call_count == int(returned and paused)
    stop_egm.assert_called_once_with(node)
    node._servo_cmd.close.assert_called_once()
