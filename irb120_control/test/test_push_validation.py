"""Contact validation during continuous motion, without ROS or robot hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from collections import deque
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
                     'PUSH_CONTACT_MARGIN', 'PUSH_DISTANCE', 'PUSH_CONTACT_FORCE_N', 'PUSH_FORCE_WINDOW_SEC',
                     'PUSH_FORCE_ANGLE_DEG', 'PUSH_FORCE_LIMIT_N', 'PUSH_VELOCITY_SCALE',
                     'PUSH_SPEED', 'PUSH_CONTROL_HZ', 'PUSH_LINE_GAIN', 'PUSH_SETTLE_SEC',
                     'PUSH_TRACK_GAIN', 'PUSH_MAX_CMD_SPEED',
                     'APPROACH_VELOCITY_SCALE', 'PUSH_APPROACH_LIFT', 'PUSH_DESCENT_SPEED'} for t in n.targets)]
    env = dict(np=np, BASE_FRAME='base_link', force_push_check=force_push_check, **extra)
    exec(compile(ast.Module(body=[*constants, fn], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env[name]


def moving_node(force, *, offset=0., stale=False):
    d = np.array([.96, -.28, 0.])
    ball = np.array([.6, .01, .04])
    p = ball + offset*d
    samples = [(10.+i*.01, np.asarray(force)) for i in range(9)] if not stale else [(9., force)]
    node = NS(_push_direction=d, _force_samples=deque(samples), _force_check={'status': 'waiting'},
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
    for stale, offset in [(False, .051), (True, 0.)]:
        node, ball = moving_node([0., 0., 0.], offset=offset, stale=stale)
        assert not method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
        assert node._force_check['status'] == 'rejected'


def test_unvalidated_contact_may_continue_before_contact_margin():
    node, ball = moving_node([0., 0., 0.], offset=-.01)
    assert method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
    assert node._force_check['status'] == 'waiting'


def test_validation_is_latched_and_does_not_steer():
    node, ball = moving_node([0., 1., 0.], offset=.02)
    node._force_check = dict(status='validated', contact_pos=ball.tolist())
    assert method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))(node, ball)
    assert node._force_check['status'] == 'validated'


def test_contact_felt_late_is_accepted_and_push_stops_50mm_after_it():
    check = method('_check_moving_contact', time=NS(monotonic=lambda: 10.09))
    node, ball = moving_node([-.48, .14, 0.], offset=.012)  # felt 12 mm past the prediction
    assert check(node, ball) and node._force_check['status'] == 'validated'
    assert node._force_check['contact_offset_m'] == pytest.approx(.012)
    contact = np.asarray(node._force_check['contact_pos'])
    for pushed, going in [(.03, True), (.051, False)]:
        p = contact + pushed*node._push_direction
        node._lookup_ee_pose = lambda p=p: NS(position=NS(x=p[0], y=p[1], z=p[2]))
        assert check(node, ball) == going
    assert node._force_check['pushed_m'] == pytest.approx(.051)
    # The motion was cancelled on purpose: _validated_push reports success.
    node._pre_push_pos = ball - .04*node._push_direction
    node._mean_contact_force = lambda: np.zeros(3)
    node._cartesian_move = lambda *a, **kw: kw['plan_only']
    def servo(b, travel):
        node._force_check.update(status='validated', pushed_m=.05)
        return True
    node._servo_push = servo
    assert method('_validated_push')(node, ball, .065)


def test_push_line_is_only_planned_then_servoed_to_a_stop_50mm_after_contact():
    node, ball = moving_node([-.48, .14, 0.])
    node._pre_push_pos = ball - .04*node._push_direction
    node._mean_contact_force = lambda: np.zeros(3)
    moves = []
    node._cartesian_move = lambda *a, **kw: moves.append(kw) or True
    # Fake Servo: each tick moves the finger by the commanded twist and stamps a fresh
    # wrench (reaction once the ball passes the predicted contact).
    clock = NS(now=10.)
    state = NS(pos=np.asarray(node._pre_push_pos, dtype=float), twists=[], paused=[])
    def spin_once(n, timeout_sec):
        clock.now += timeout_sec
    def publish_twist(vx, vy, vz, state=''):
        v = np.array([vx, vy, vz])
        state_.twists.append(v)
        state_.pos = state_.pos + .57 * v / 50.  # this rig's Servo: ~57% of the command
        touching = (state_.pos - ball) @ node._push_direction > -.005
        node._force_samples.append((clock.now, np.array([-.48, .14, 0.]) if touching else np.zeros(3)))
        return True
    state_ = state
    node._servo_cmd = NS(publish_twist=publish_twist, publish_zero=lambda *a: True)
    node._lookup_ee_pose = lambda: NS(position=NS(x=state_.pos[0], y=state_.pos[1], z=state_.pos[2]))
    node._set_servo_paused = lambda paused: state_.paused.append(paused) or True
    node._motion_uncertain = False
    env = dict(time=NS(monotonic=lambda: clock.now), rclpy=NS(ok=lambda: True, spin_once=spin_once))
    node._check_moving_contact = lambda b: method('_check_moving_contact', **env)(node, b)
    node._servo_push = lambda b, travel: method('_servo_push', **env)(node, b, travel)
    assert method('_validated_push')(node, ball, .2)
    assert len(moves) == 1 and moves[0]['plan_only']
    np.testing.assert_allclose(moves[0]['target_position'], ball + .16*node._push_direction)
    assert state_.paused == [False, True]
    assert node._force_check['pushed_m'] == pytest.approx(.05, abs=.001)
    pushed = (state_.pos - np.asarray(node._force_check['contact_pos'])) @ node._push_direction
    assert pushed == pytest.approx(.05, abs=.001)  # stopped there, not at the planned end
    speeds = [.57 * np.linalg.norm(v) for v in state_.twists[-50:]]
    assert np.mean(speeds) == pytest.approx(.009, rel=.05)  # tracks PUSH_SPEED despite the 57%


def test_no_full_motion_without_fresh_unloaded_force_or_validated_contact():
    for force in (None, np.array([1., 0., 0.])):
        node = NS(_mean_contact_force=lambda: force, _force_check={}, _cartesian_move=Mock())
        assert not method('_validated_push')(node, [.6, 0, .04], .08)
        node._cartesian_move.assert_not_called()
    node, ball = moving_node([0., 0., 0.])
    node._pre_push_pos = ball - .04*node._push_direction
    node._mean_contact_force = lambda: np.zeros(3)
    node._cartesian_move = Mock(return_value=True)
    node._servo_push = Mock(return_value=True)
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
