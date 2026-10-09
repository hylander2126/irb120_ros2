"""Exercise action-success verification in the actual helper without ROS motion."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
from irb120_control.controllers import moveit_single_shot as module


def _client(*codes):
    """MoveGroup client whose successive plans return the given error codes."""
    sent = []
    for code in codes:
        result = NS(status=4, result=NS(error_code=NS(val=code), planned_trajectory=Mock(),
                                       trajectory_start=Mock(), executed_trajectory=Mock()))
        completed = Mock(); completed.done.return_value = True; completed.result.return_value = result
        handle = Mock(accepted=True); handle.get_result_async.return_value = completed
        future = Mock(); future.done.return_value = True; future.result.return_value = handle
        sent.append(future)
    client = Mock(); client.send_goal_async.side_effect = sent
    return client


def _no_ros(monkeypatch, verified=True):
    monkeypatch.setattr(module.rclpy, 'spin_until_future_complete', lambda *a, **k: None)
    monkeypatch.setattr(module, '_wait_for_joint_states', lambda *a, **k: True)
    execute = Mock(return_value=True)
    monkeypatch.setattr(module, 'execute_checked_plan', execute)
    verifier = Mock(return_value=verified)
    monkeypatch.setattr(module, 'verify_joint_goal', verifier)
    return execute, verifier


def test_action_success_is_insufficient(monkeypatch):
    _, verifier = _no_ros(monkeypatch, verified=False)
    SUCCESS = module.MoveItErrorCodes.SUCCESS
    assert not module.plan_and_execute_joint_goal(Mock(), _client(SUCCESS), joint_positions=[0.]*6)
    verifier.assert_called_once()
    verifier.return_value = True
    assert module.plan_and_execute_joint_goal(Mock(), _client(SUCCESS), joint_positions=[0.]*6)


def test_failed_plan_is_replanned_before_executing(monkeypatch):
    execute, _ = _no_ros(monkeypatch)
    FAIL, SUCCESS = module.MoveItErrorCodes.INVALID_MOTION_PLAN, module.MoveItErrorCodes.SUCCESS
    client = _client(FAIL, SUCCESS)
    assert module.plan_and_execute_joint_goal(Mock(), client, joint_positions=[0.]*6)
    assert client.send_goal_async.call_count == 2
    execute.assert_called_once()


def test_gives_up_after_plan_retries_without_moving(monkeypatch):
    execute, _ = _no_ros(monkeypatch)
    FAIL = module.MoveItErrorCodes.INVALID_MOTION_PLAN
    client = _client(FAIL, FAIL)
    assert not module.plan_and_execute_joint_goal(Mock(), client, joint_positions=[0.]*6, plan_retries=2)
    assert client.send_goal_async.call_count == 2
    execute.assert_not_called()


def test_pose_goal_replans_and_records_final_result(monkeypatch):
    execute, _ = _no_ros(monkeypatch)
    FAIL, SUCCESS = module.MoveItErrorCodes.INVALID_MOTION_PLAN, module.MoveItErrorCodes.SUCCESS
    client = _client(FAIL, SUCCESS)
    monkeypatch.setattr('rosidl_runtime_py.convert.message_to_ordereddict', lambda msg: {})
    diagnostics = {}
    assert module.plan_and_execute_pose_goal(
        Mock(), client, target_position=(.5, 0., .1), target_orientation=(0., 0., 0., 1.),
        diagnostics=diagnostics)
    assert client.send_goal_async.call_count == 2
    assert diagnostics['moveit_error_code'] == SUCCESS and diagnostics['execution_success']
    execute.assert_called_once()


def test_pose_goal_gives_up_without_moving(monkeypatch):
    execute, _ = _no_ros(monkeypatch)
    FAIL = module.MoveItErrorCodes.INVALID_MOTION_PLAN
    monkeypatch.setattr('rosidl_runtime_py.convert.message_to_ordereddict', lambda msg: {})
    client = _client(FAIL, FAIL, FAIL)
    assert not module.plan_and_execute_pose_goal(
        Mock(), client, target_position=(.5, 0., .1), target_orientation=(0., 0., 0., 1.))
    assert client.send_goal_async.call_count == 3
    execute.assert_not_called()
