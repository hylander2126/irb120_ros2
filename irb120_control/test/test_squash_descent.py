"""Exercise the actual SQUASH branch without ROS or hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).parents[1] / 'irb120_control' / 'arc_static.py'


def squash_tick():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ArcStatic')
    tick = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_tick')
    branch = next(n for n in tick.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "self._state == 'SQUASH'")
    fn = ast.parse('def tick(self, pz):\n    pass').body[0]
    fn.body = branch.body
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    env = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in (
                'SQUASH_STANDOFF', 'SQUASH_EXTRA_DESCENT_M', 'SQUASH_TIMEOUT_SEC',
                'CONTACT_STABLE_SAMPLES', 'DESCEND_SPEED') for t in n.targets):
            exec(compile(ast.Module(body=[n], type_ignores=[]), str(SOURCE), 'exec'), env)
    exec(compile(module, str(SOURCE), 'exec'), env)
    return env['tick']


def node(force=0.5):
    return SimpleNamespace(_state='SQUASH', _pre_squash_pos=(.6, 0., .23),
        _have_force=True, _force_z=force, _force_y=0., _contact_felt=True,
        _force_ctrl=SimpleNamespace(reference=5.), _contact_count=0,
        _retry_requested=False, _servo_cmd=Mock(), _transition=Mock(),
        _check_timeout=Mock(return_value=False), get_logger=Mock(return_value=Mock()))


def test_missed_contact_retracts_without_downward_command_or_retry():
    n = node()
    squash_tick()(n, .194)
    n._transition.assert_called_once_with('RETRACT')
    n._servo_cmd.publish_zero.assert_called_once()
    n._servo_cmd.publish_twist.assert_not_called()
    assert not n._retry_requested


def test_above_floor_continues_descent():
    n = node()
    squash_tick()(n, .201)
    n._servo_cmd.publish_twist.assert_called_once()
    n._transition.assert_not_called()


def test_expected_force_stops_descent_even_at_floor():
    n = node(force=5.1)
    squash_tick()(n, .194)
    n._transition.assert_called_once_with('LULL')
    n._servo_cmd.publish_twist.assert_not_called()
