"""Exercise the real ROS message adapter against fake service responses."""
import pytest

pytest.importorskip('moveit_msgs')
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from trajectory_msgs.msg import JointTrajectoryPoint
from irb120_control.shove import MoveItChecker
from test_shove_check import request


@pytest.mark.parametrize('fraction,error,cable,expected', [
    (1., 1, 0., True), (.9999, 1, 0., False),
    (1., -1, 0., False), (1., 1, 1.1, False),
])
def test_adapter(fraction, error, cable, expected):
    checker = object.__new__(MoveItChecker)
    checker.ik, checker.cartesian = object(), object()
    calls = []

    def call(client, req):
        calls.append(req)
        if client is checker.ik:
            result = GetPositionIK.Response()
            result.error_code.val = 1
            result.solution.joint_state.name = ['joint_6']
            result.solution.joint_state.position = [0.]
            return result
        result = GetCartesianPath.Response()
        result.error_code.val = error
        result.fraction = fraction
        result.solution.joint_trajectory.joint_names = ['joint_6']
        result.solution.joint_trajectory.points = [JointTrajectoryPoint(positions=[cable])]
        return result

    checker.call = call
    data = request()
    result = checker(data, data['candidates'][0])
    assert result['geometric_path_found'] is expected
    assert calls[0].ik_request.avoid_collisions
    path = calls[1]
    assert path.avoid_collisions and path.header.frame_id == 'base_link'
    assert len(path.waypoints) == 3
    assert list(path.start_state.joint_state.position) == [0.]
    assert path.path_constraints.joint_constraints[0].joint_name == 'joint_6'
