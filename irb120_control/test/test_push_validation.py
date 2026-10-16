"""Contact validation during continuous motion, without ROS or robot hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from irb120_control.util.push_geometry import force_push_check

SOURCE = Path(__file__).parents[1] / 'irb120_control' / 'push.py'


def method(name, **extra):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Push')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in {
                     'PUSH_CONTACT_MARGIN', 'PUSH_CONTACT_FORCE_N', 'PUSH_FORCE_WINDOW_SEC',
                     'PUSH_FORCE_ANGLE_DEG', 'PUSH_FORCE_LIMIT_N', 'PUSH_VELOCITY_SCALE',
                     'APPROACH_VELOCITY_SCALE', 'PUSH_APPROACH_LIFT', 'PUSH_DESCENT_SPEED'} for t in n.targets)]
    env = dict(np=np, BASE_FRAME='base_link', force_push_check=force_push_check, **extra)
    exec(compile(ast.Module(body=[*constants, fn], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[name]


def moving_node(force, *, offset=0., stale=False):
    d = np.array([.96, -.28, 0.])
    ball = np.array([.6, .01, .04])
    p = ball + offset*d
    samples = [(10.+i*.01, np.asarray(force)) for i in range(9)] if not stale else [(9., force)]
    node = NS(_push_direction=d, _force_samples=samples, _force_check={'status': 'waiting'},
              _lookup_ee_pose=lambda: NS(position=NS(x=p[0], y=p[1], z=p[2])),
              get_logger=Mock(return_value=Mock()))
    return node, ball


@pytest.mark.parametrize('force,status', [
    ([-.48, .14, 0.], 'validated'), ([.48, -.14, 0.], 'rejected'),
    ([0., 1., 0.], 'rejected'), ([-21., 0., 0.], 'rejected'),
])
def test_force_direction_checked_while_moving(force, status):
    node, ball = moving_node(force)
    d = node._push_direction.copy()
    ok = method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
    assert ok == (status == 'validated')
    assert node._force_check['status'] == status
    np.testing.assert_array_equal(node._push_direction, d)


def test_no_contact_stops_after_bounded_distance_and_stale_force_rejects():
    for stale, offset in [(False, .006), (True, 0.)]:
        node, ball = moving_node([0., 0., 0.], offset=offset, stale=stale)
        assert not method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
        assert node._force_check['status'] == 'rejected'


def test_unvalidated_contact_may_continue_before_contact_margin():
    node, ball = moving_node([0., 0., 0.], offset=-.01)
    assert method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
    assert node._force_check['status'] == 'waiting'


def test_validation_is_latched_and_does_not_steer():
    node, ball = moving_node([0., 1., 0.], offset=.02)
    node._force_check = dict(status='validated')
    assert method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
    assert node._force_check['status'] == 'validated'


def test_one_continuous_push_uses_execution_check_and_original_endpoint():
    node, ball = moving_node([-.48, .14, 0.])
    node._pre_push_pos = ball - .04*node._push_direction
    node._mean_contact_force = lambda: np.zeros(3)
    node._check_moving_contact = lambda b: method('_check_moving_contact',
                                               time=NS(monotonic=lambda: 10.09))(node, b)
    moves = []
    def move(*args, **kwargs):
        moves.append(kwargs)
        return kwargs['execution_check']()
    node._cartesian_move = move
    assert method('_validated_push')(node, ball, .08)
    assert len(moves) == 1
    np.testing.assert_allclose(moves[0]['target_position'], ball + .04*node._push_direction)


def test_no_full_motion_without_fresh_unloaded_force_or_validated_contact():
    for force in (None, np.array([1., 0., 0.])):
        node = NS(_mean_contact_force=lambda: force, _force_check={}, _cartesian_move=Mock())
        assert not method('_validated_push')(node, [.6, 0, .04], .08)
        node._cartesian_move.assert_not_called()
    node, ball = moving_node([0., 0., 0.])
    node._pre_push_pos = ball - .04*node._push_direction
    node._mean_contact_force = lambda: np.zeros(3)
    node._cartesian_move = Mock(return_value=True)
    assert not method('_validated_push')(node, ball, .08)
    assert node._force_check['status'] == 'rejected'


@pytest.mark.parametrize('fresh', [False, True])
def test_force_average_rejects_old_samples_and_accepts_fresh_window(fresh):
    clock = NS(now=10.)
    node = NS(_force_samples=[(9., np.array([99., 0., 0.]))], _force_check={})
    def spin(node, timeout_sec):
        clock.now += timeout_sec
        if fresh:
            node._force_samples.append((clock.now, np.array([-1., 0., 0.])))
    result = method('_mean_contact_force', time=NS(monotonic=lambda: clock.now),
                    rclpy=NS(ok=lambda: True, spin_once=spin))(node)
    if fresh:
        np.testing.assert_allclose(result, [-1., 0., 0.])
    else:
        assert result is None
        assert node._force_check['status'] == 'rejected'


def test_approach_plans_high_then_descends_slowly_and_checks_arrival():
    from contextlib import nullcontext
    calls = []
    pre = np.array([.55, 0., .04])
    node = NS(_pre_push_pos=pre, _object_xyz=np.zeros((2,3)), _move_group_client=None,
              _push_orientation=[0., 0., 0., 1.], _approach_diag={},
              _arrived=lambda *a: calls.append(('arrived', a)) or True,
              _cartesian_move=lambda *a, **kw: calls.append(('descend', kw)) or True)
    def plan(*a, **kw):
        calls.append(('plan', kw))
        return True
    assert method('move_to_pre_push', object_obstacle=lambda *a: nullcontext(),
                  plan_and_execute_pose_goal=plan)(node)
    assert [c[0] for c in calls] == ['plan', 'arrived', 'descend', 'arrived']
    np.testing.assert_allclose(calls[0][1]['target_position'], pre + [0.,0.,.08])
    np.testing.assert_allclose(calls[2][1]['target_position'], pre)
    assert calls[2][1]['max_speed'] == .005
