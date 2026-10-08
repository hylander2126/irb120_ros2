import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from irb120_control.util.approach_diagnostics import ApproachDiagnostics, position_error_mm


def test_signed_error_explains_direction_of_missed_target():
    result = position_error_mm([.5085, -.0044, .1647], [.5596, -.0054, .2176])
    np.testing.assert_allclose(result['actual_minus_target_mm'], [-51.1, 1., -52.9])
    assert result['distance_mm'] > 70


def test_missing_tf_is_saved_with_joints_and_invalid_numbers_are_explicit(tmp_path):
    node = SimpleNamespace(_now_s=lambda: 10., _tf_buffer=Mock(),
        _tracking_log=[dict(source='joint_states', time=9.9, names=['joint_1'],
                            positions=[.1], velocities=[float('nan')])])
    node._tf_buffer.lookup_transform.side_effect = RuntimeError('missing finger transform')
    path = tmp_path/'approach.jsonl'
    log = ApproachDiagnostics(node, path, [0, 0, 0], 'world', 'finger_ball_center')
    log.sample('after_approach')
    row = json.loads(path.read_text())
    assert row['tf']['available'] is False
    assert 'missing finger' in row['tf']['reason']
    assert row['joint_states']['positions'] == [.1]
    assert row['joint_states']['velocities'] == [None]
    assert row['controller_state'] is None


def test_tf_logs_age_and_error_without_claiming_action_success(tmp_path):
    tf = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=9,nanosec=500000000)),
        transform=SimpleNamespace(translation=SimpleNamespace(x=.1,y=.2,z=.3),
                                  rotation=SimpleNamespace(x=0.,y=0.,z=0.,w=1.)))
    node=SimpleNamespace(_now_s=lambda:10., _tf_buffer=Mock(), _tracking_log=[])
    node._tf_buffer.lookup_transform.return_value=tf
    path=tmp_path/'approach.jsonl'
    ApproachDiagnostics(node,path,[.11,.2,.3],'world','finger_ball_center').sample()
    row=json.loads(path.read_text())
    assert row['tf']['age_s']==.5
    np.testing.assert_allclose(row['tf']['actual_minus_target_mm'],[-10,0,0])
    assert 'moveit_reported_success' not in row


def test_fk_uses_measured_joints_and_records_endpoint_error(tmp_path, monkeypatch):
    import rclpy
    from moveit_msgs.srv import GetPositionFK
    from geometry_msgs.msg import PoseStamped
    result=GetPositionFK.Response()
    result.error_code.val=1
    pose=PoseStamped()
    pose.header.frame_id='world'
    pose.pose.position.x=.5
    pose.pose.orientation.w=1.
    result.pose_stamped=[pose]
    future=Mock()
    future.done.return_value=True
    future.result.return_value=result
    client=Mock()
    client.wait_for_service.return_value=True
    client.call_async.return_value=future
    node=SimpleNamespace(_now_s=lambda:10.,
        _tracking_log=[dict(source='joint_states',time=9.9,names=['joint_1'],positions=[.2])],
        create_client=Mock(return_value=client),destroy_client=Mock(),get_logger=Mock())
    monkeypatch.setattr(rclpy,'spin_until_future_complete',lambda *a,**kw:None)
    path=tmp_path/'approach.jsonl'
    ApproachDiagnostics(node,path,[.55,0,0],'world','finger_ball_center').measured_fk('after_approach_fk')
    request=client.call_async.call_args.args[0]
    assert list(request.robot_state.joint_state.position)==[.2]
    row=json.loads(path.read_text())
    assert row['available']
    np.testing.assert_allclose(row['actual_minus_target_mm'],[-50,0,0])
    node.destroy_client.assert_called_once_with(client)
