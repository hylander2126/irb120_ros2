"""Geometry, torque reversal and actual arc control tick with fake robot I/O."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

from irb120_control.util.forward_arc_geometry import PivotArc, pivot_torque
from irb120_control.util.forward_tip_motion import TipConfig, TipMotion
from test_fwd_tip_runtime import fake_node, methods


def test_arbitrary_edge_axis_has_constant_radius_and_correct_plane():
    axis = np.array([.6, .8, 0.])
    direction = np.cross(axis, [0., 0., 1.])
    pivot = np.array([.4, -.1, .03])
    ball = pivot - .04 * direction + [0., 0., .18]
    arc = PivotArc(pivot, -axis, ball, direction)
    for angle in np.linspace(0, .6, 30):
        point = arc.point(angle)
        assert np.linalg.norm(point - arc.center) == pytest.approx(arc.radius)
        assert (point - ball) @ axis == pytest.approx(0., abs=1e-12)
        assert arc.angle(point) == pytest.approx(angle)
        assert arc.tangent(angle) @ (point - arc.center) == pytest.approx(0., abs=1e-12)
    assert arc.tangent(0) @ direction > 0


def test_full_wrench_shift_preserves_contact_moment():
    force = np.array([2., 0., -.3])
    sensor = np.array([-.1, 0., .3])
    contact = np.array([-.04, 0., .2])
    moment = np.cross(contact - sensor, force) + [0., .012, 0.]
    actual = pivot_torque(force, moment, sensor, np.zeros(3), np.array([0., 1., 0.]))
    assert actual == pytest.approx((np.cross(contact, force) + [0., .012, 0.])[1])


def test_torque_drop_reverses_with_high_force_and_only_fresh_samples():
    motion = TipMotion(TipConfig(), .02, 0.)
    motion.state = 'TIP'
    motion.contact_progress = .02
    motion.probe = {'already': 'tested'}
    motion.step(.1, .025, 2., 2., reversal_signal=.4)
    for i in range(4):
        motion.step(.2 + i * .01, .03, 2., 2., reversal_signal=.04)
    for i in range(10):
        motion.step(.3 + i * .01, .03, 2., 2., fresh=False, reversal_signal=.04)
    assert motion.state == 'TIP'
    motion.step(.5, .03, 2., 2., reversal_signal=.04)
    assert motion.state == 'RETURN'
    assert motion.reversal_reason == 'torque_drop'
    assert motion.step(.6, .01, 2., 2., reversal_signal=.04) < 0
    motion.step(.7, 0., 0., 0., reversal_signal=0.)
    assert motion.completed and motion.returned


def arc_tick():
    source = Path(__file__).parents[1] / 'irb120_control/fwd_arc.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_tick')
    env = methods()
    env.update(PivotArc=PivotArc, pivot_torque=pivot_torque)
    exec(compile(ast.Module(body=[tick], type_ignores=[]), str(source), 'exec'), env)
    return env['_tick']


@pytest.mark.parametrize('returning', [False, True])
def test_runtime_commands_arc_tangent_and_retraces_it(returning):
    node = fake_node()
    node._pivot = np.zeros(3)
    node._direction = np.array([1., 0., 0.])
    node._arc = node._nominal_arc = PivotArc(node._pivot, [0., 1., 0.], [-.04, 0., .18], node._direction)
    node._arc_start = .04
    node._start_pose[:3] = [-.08, 0., .18]
    node._transform.return_value[:3] = node._arc.point(.2)
    node._force = np.array([2., 0., 0.])
    node._torque = np.zeros(3)
    node._wrench_origin = np.array([0., 0., .2])
    node.max_angle = math.radians(30.)
    node.trajectory_metadata = {}
    node.motion.state = 'RETURN' if returning else 'TIP'
    node.motion.contact_progress = .04
    node.motion.probe = {'already': 'tested'}
    arc_tick()(node)
    command = np.array(node._servo_cmd.publish_twist.call_args.args)
    np.testing.assert_allclose(command, node._arc.tangent(.2) * node.config.speed * (-1 if returning else 1), atol=1e-12)
    assert abs(command[2]) > 0
    node._stop.assert_not_called()


def test_runtime_angle_limit_returns_before_pushing_further():
    node = fake_node()
    node._pivot = np.zeros(3)
    node._direction = np.array([1., 0., 0.])
    node._arc = node._nominal_arc = PivotArc(node._pivot, [0., 1., 0.], [-.04, 0., .18], node._direction)
    node._arc_start = .04
    node._transform.return_value[:3] = node._arc.point(.3)
    node.max_angle = .25
    node._torque = node._wrench_origin = np.zeros(3)
    node.trajectory_metadata = {}
    node.motion.state = 'TIP'
    node.motion.contact_progress = .04
    arc_tick()(node)
    assert node.motion.state == 'RETURN'
    assert node.motion.reversal_reason == 'angle_limit'
