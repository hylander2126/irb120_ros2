"""Cancellation must finish before the caller can retreat; test without ROS."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


def execute_method(env):
    path = Path(__file__).parents[1] / 'irb120_control/controllers/cartesian_move.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_execute_trajectory')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    return env['_execute_trajectory']


@pytest.mark.parametrize('stops', [True, False])
def test_rejected_check_cancels_and_requires_terminal_result(stops):
    state = NS(done=False)
    terminal = NS(result=NS(error_code=NS(val=1)))
    result = NS(done=lambda: state.done, result=lambda: terminal if state.done else None)
    cancel = object()
    handle = NS(accepted=True, get_result_async=lambda: result,
                cancel_goal_async=Mock(return_value=cancel))
    accepted = NS(done=lambda: True, result=lambda: handle)
    client = NS(wait_for_server=lambda **kw: True, send_goal_async=lambda g: accepted)
    def spin_until(node, future, **kw):
        if future is result:
            state.done = stops
    env = dict(rclpy=NS(spin_until_future_complete=spin_until, spin_once=lambda *a, **kw: None),
               time=NS(monotonic=lambda: 0.), ExecuteTrajectory=NS(Goal=NS),
               MoveItErrorCodes=NS(SUCCESS=1))
    node = NS(get_logger=lambda: Mock())
    if stops:
        assert execute_method(env)(node, client, object(), 1., execution_check=lambda: False) is False
    else:
        with pytest.raises(RuntimeError, match='Cannot confirm'):
            execute_method(env)(node, client, object(), 1., execution_check=lambda: False)
    handle.cancel_goal_async.assert_called_once()


def test_default_execution_without_check_still_succeeds():
    terminal = NS(result=NS(error_code=NS(val=1)))
    result = NS(done=lambda: True, result=lambda: terminal)
    handle = NS(accepted=True, get_result_async=lambda: result)
    accepted = NS(done=lambda: True, result=lambda: handle)
    client = NS(wait_for_server=lambda **kw: True, send_goal_async=lambda g: accepted)
    env = dict(rclpy=NS(spin_until_future_complete=lambda *a, **kw: None),
               ExecuteTrajectory=NS(Goal=NS), MoveItErrorCodes=NS(SUCCESS=1))
    assert execute_method(env)(NS(get_logger=lambda: Mock()), client, object(), 1.)
