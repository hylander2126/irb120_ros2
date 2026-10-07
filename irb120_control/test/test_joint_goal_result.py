"""Exercise action-success verification in the actual helper without ROS motion."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
from irb120_control.controllers import moveit_single_shot as module


def test_action_success_is_insufficient(monkeypatch):
    monkeypatch.setattr(module.rclpy,'spin_until_future_complete',lambda *a,**k:None)
    monkeypatch.setattr(module,'_wait_for_joint_states',lambda *a,**k:True)
    verifier=Mock(return_value=False)
    monkeypatch.setattr(module,'verify_joint_goal',verifier)
    result=NS(result=NS(error_code=NS(val=module.MoveItErrorCodes.SUCCESS)))
    completed=Mock();completed.done.return_value=True;completed.result.return_value=result
    handle=Mock(accepted=True);handle.get_result_async.return_value=completed
    sent=Mock();sent.done.return_value=True;sent.result.return_value=handle
    client=Mock();client.send_goal_async.return_value=sent
    node=Mock()
    assert not module.plan_and_execute_joint_goal(node,client,joint_positions=[0.]*6)
    verifier.assert_called_once()
    verifier.return_value=True
    assert module.plan_and_execute_joint_goal(node,client,joint_positions=[0.]*6)
