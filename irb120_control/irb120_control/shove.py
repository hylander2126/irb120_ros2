"""Read-only MoveIt feasibility check for externally calculated shove candidates."""
import argparse
import json
from pathlib import Path

from .util.shove_check import check_candidates, validate_request


class MoveItChecker:
    """Solve contact IK, then a collision-checked Cartesian path through braking."""

    def __init__(self, node, timeout=15.):
        from moveit_msgs.srv import GetCartesianPath, GetPositionIK
        self.node, self.timeout = node, timeout
        self.ik = node.create_client(GetPositionIK, '/compute_ik')
        self.cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path')

    def call(self, client, request):
        import rclpy
        if not client.wait_for_service(timeout_sec=self.timeout):
            raise RuntimeError(f'Service unavailable: {client.srv_name}')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout)
        if not future.done():
            future.cancel()
            raise RuntimeError(f'Service timed out: {client.srv_name}')
        response = future.result()
        if response is None:
            raise RuntimeError(f'No response: {client.srv_name}')
        return response

    def __call__(self, request, candidate):
        from geometry_msgs.msg import Pose
        from moveit_msgs.srv import GetCartesianPath, GetPositionIK
        from rosidl_runtime_py.convert import message_to_ordereddict
        from .controllers.moveit_single_shot import cable_path_constraints, validate_cable_trajectory
        poses = []
        for sample in candidate['samples']:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, sample['position'])
            q = sample['orientation_xyzw']
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, q)
            poses.append(pose)
        ik = GetPositionIK.Request()
        ik.ik_request.group_name = 'manipulator'
        ik.ik_request.ik_link_name = 'finger_ball_center'
        ik.ik_request.pose_stamped.header.frame_id = 'base_link'
        ik.ik_request.pose_stamped.pose = poses[0]
        ik.ik_request.robot_state.joint_state.name = request['joint_names']
        ik.ik_request.robot_state.joint_state.position = list(map(float, request['initial_joints']))
        ik.ik_request.avoid_collisions = True
        ik.ik_request.constraints = cable_path_constraints()
        ik.ik_request.timeout.sec = 2
        try:
            contact = self.call(self.ik, ik)
            if contact.error_code.val != 1:
                return dict(geometric_path_found=False, reason='contact_ik_failed',
                            moveit_error_code=contact.error_code.val)
            path = GetCartesianPath.Request()
            path.header.frame_id = 'base_link'
            path.start_state = contact.solution
            path.group_name = 'manipulator'
            path.link_name = 'finger_ball_center'
            path.waypoints = poses
            path.max_step = .001
            path.revolute_jump_threshold = .1
            path.avoid_collisions = True
            path.path_constraints = cable_path_constraints()
            path.max_velocity_scaling_factor = .1
            path.max_acceleration_scaling_factor = .1
            response = self.call(self.cartesian, path)
            complete = (response.error_code.val == 1 and math_is_complete(response.fraction)
                        and validate_cable_trajectory(response.solution))
            return dict(geometric_path_found=complete, fraction=response.fraction,
                        moveit_error_code=response.error_code.val,
                        reason='complete_path' if complete else 'partial_or_invalid_path',
                        contact_state=message_to_ordereddict(contact.solution),
                        moveit_trajectory=message_to_ordereddict(response.solution))
        except RuntimeError as exc:
            return dict(geometric_path_found=False, reason='service_failure', detail=str(exc))


def math_is_complete(fraction):
    import math
    return math.isfinite(fraction) and fraction >= 1.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='Query MoveIt only; never execute motion')
    check.add_argument('request', type=Path)
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    request = validate_request(json.loads(args.request.read_text()))
    import rclpy
    rclpy.init()
    node = rclpy.create_node('shove_feasibility_check')
    try:
        report = check_candidates(request, MoveItChecker(node))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return 0 if report['selected_candidate'] is not None else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
