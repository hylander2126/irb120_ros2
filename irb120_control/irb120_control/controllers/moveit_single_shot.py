#!/usr/bin/env python3
"""Single-shot MoveIt planning helpers for IRB120 control scripts."""

from __future__ import annotations

from dataclasses import dataclass
import time
import math
import copy
from typing import Sequence

import rclpy
from geometry_msgs.msg import Pose as GeometryPose
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive
from .joint_verification import JointGoalVerifier


# Fixed cable-neutral window. Never wrap angles or recenter on the current pose.
CABLE_LIMIT = math.pi / 3


def cable_path_constraints():
    constraints = Constraints()
    constraints.joint_constraints = [JointConstraint(
        joint_name='joint_6', position=0., tolerance_above=CABLE_LIMIT,
        tolerance_below=CABLE_LIMIT, weight=1.)]
    return constraints


def validate_cable_trajectory(trajectory):
    joints = trajectory.joint_trajectory
    if 'joint_6' not in joints.joint_names or not joints.points:
        return False
    index = joints.joint_names.index('joint_6')
    previous = None
    for point in joints.points:
        if len(point.positions) != len(joints.joint_names):
            return False
        angle = point.positions[index]
        if not math.isfinite(angle) or abs(angle) > CABLE_LIMIT:
            return False
        if previous is not None and abs(angle-previous) > CABLE_LIMIT:
            return False
        previous = angle
    return True


def execute_checked_plan(node, trajectory, timeout_sec):
    if not validate_cable_trajectory(trajectory):
        node.get_logger().error('Trajectory violates joint 6 cable limits (+/-60 deg); not executing')
        return False
    client = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
    try:
        if not client.wait_for_server(timeout_sec=5.):
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=5.)
        if not future.done() or future.result() is None or not future.result().accepted:
            return False
        handle = future.result()
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result, timeout_sec=timeout_sec)
        if not result.done():
            handle.cancel_goal_async()
            node.get_logger().error('Trajectory execution timed out; cancellation requested')
            return False
        return (result.result() is not None and result.result().status == 4
                and result.result().result.error_code.val == MoveItErrorCodes.SUCCESS)
    finally:
        client.destroy()


def verify_joint_goal(node, joint_names, joint_positions, tolerance, timeout_sec=5.0):
    verifier = JointGoalVerifier(joint_names, joint_positions,
        node.get_clock().now().nanoseconds * 1e-9, tolerance=tolerance)

    def callback(msg):
        verifier.observe(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            node.get_clock().now().nanoseconds * 1e-9, msg.name, msg.position)

    sub = node.create_subscription(JointState, '/joint_states', callback, 1)
    deadline = time.monotonic() + timeout_sec
    try:
        while rclpy.ok() and time.monotonic() < deadline and not verifier.report['verified']:
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        node.destroy_subscription(sub)
    node._last_joint_goal_verification = dict(verifier.report)
    if not verifier.report['verified']:
        node.get_logger().error(f'Measured joint goal verification failed: {verifier.report}')
    return verifier.report['verified']


DEFAULT_GROUP_NAME = "manipulator"
DEFAULT_BASE_FRAME = "world"
DEFAULT_EE_LINK = "finger_ball_center"
DEFAULT_POSITION_TOLERANCE = 0.0005 # 0.002
DEFAULT_ORIENTATION_TOLERANCE = 0.001 # 0.01


@dataclass(frozen=True)
class PoseGoalDefaults:
    """Default values for a constrained pose goal."""

    position_tolerance: float = DEFAULT_POSITION_TOLERANCE
    orientation_tolerance: float = DEFAULT_ORIENTATION_TOLERANCE
    planning_attempts: int = 5
    allowed_planning_time: float = 10.0
    velocity_scaling_factor: float = 0.1
    acceleration_scaling_factor: float = 0.1
    workspace_min_corner: Sequence[float] = (-1.5, -1.5, -0.5)
    workspace_max_corner: Sequence[float] = (1.5, 1.5, 2.0)


def _wait_for_joint_states(node, timeout_sec: float = 10.0) -> bool:
    """Block until /joint_states publishes at least one message.

    move_group returns INVALID_MOTION_PLAN if it hasn't received a robot state
    yet. This happens in the window right after EGM connects and the joint state
    broadcaster starts flowing. Returns True once a message is received.
    """
    received = [False]

    def _cb(_msg: JointState) -> None:
        if 'joint_6' not in _msg.name:
            return
        index = list(_msg.name).index('joint_6')
        if index >= len(_msg.position):
            return
        angle = _msg.position[index]
        received[0] = math.isfinite(angle) and abs(angle) <= CABLE_LIMIT

    sub = node.create_subscription(JointState, "/joint_states", _cb, 1)
    end = node.get_clock().now().nanoseconds * 1e-9 + timeout_sec
    while not received[0] and node.get_clock().now().nanoseconds * 1e-9 < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)

    if not received[0]:
        node.get_logger().warn("No measured joint 6 inside fixed +/-60 deg cable window; not planning")
        return False
    return True


def plan_and_execute_pose_goal(
    node,
    move_group_client,
    *,
    group_name: str = DEFAULT_GROUP_NAME,
    base_frame: str = DEFAULT_BASE_FRAME,
    ee_link: str = DEFAULT_EE_LINK,
    target_position: Sequence[float],
    target_orientation: Sequence[float],
    defaults: PoseGoalDefaults | None = None,
    velocity_scale: float | None = None,
    acceleration_scale: float | None = None,
    timeout_server_sec: float = 10.0,
    timeout_goal_send_sec: float = 15.0,
    timeout_result_sec: float = 30.0,
    diagnostics: dict | None = None,
) -> bool:
    """Plan and execute a constrained pose goal using MoveGroup.

    velocity_scale and acceleration_scale override the values in defaults when
    provided, without requiring a full PoseGoalDefaults instance.
    """

    if diagnostics is not None:
        diagnostics.update(frame=base_frame, link=ee_link, target_position=list(target_position),
                           target_orientation_xyzw=list(target_orientation), status='started')
    defaults = defaults or PoseGoalDefaults()
    if velocity_scale is not None:
        defaults = PoseGoalDefaults(
            position_tolerance=defaults.position_tolerance,
            orientation_tolerance=defaults.orientation_tolerance,
            planning_attempts=defaults.planning_attempts,
            allowed_planning_time=defaults.allowed_planning_time,
            velocity_scaling_factor=velocity_scale,
            acceleration_scaling_factor=acceleration_scale if acceleration_scale is not None else velocity_scale,
            workspace_min_corner=defaults.workspace_min_corner,
            workspace_max_corner=defaults.workspace_max_corner,
        )
    elif acceleration_scale is not None:
        defaults = PoseGoalDefaults(
            position_tolerance=defaults.position_tolerance,
            orientation_tolerance=defaults.orientation_tolerance,
            planning_attempts=defaults.planning_attempts,
            allowed_planning_time=defaults.allowed_planning_time,
            velocity_scaling_factor=defaults.velocity_scaling_factor,
            acceleration_scaling_factor=acceleration_scale,
            workspace_min_corner=defaults.workspace_min_corner,
            workspace_max_corner=defaults.workspace_max_corner,
        )
    px, py, pz = target_position
    qx, qy, qz, qw = target_orientation

    node.get_logger().info(f"Moving to pose target: ({px}, {py}, {pz})")

    if not move_group_client.wait_for_server(timeout_sec=timeout_server_sec):
        if diagnostics is not None:
            diagnostics["status"] = "server_unavailable"
        node.get_logger().error("MoveGroup action server not available")
        return False

    # Wait for at least one valid joint state before planning. move_group will
    # return INVALID_MOTION_PLAN (-10) if it has no current robot state yet,
    # which happens in the ~1-2s window after EGM connects and /joint_states
    # starts flowing.
    if not _wait_for_joint_states(node, timeout_sec=10.0):
        return False

    pos_constraint = PositionConstraint()
    pos_constraint.header.frame_id = base_frame
    pos_constraint.link_name = ee_link
    pos_constraint.target_point_offset.x = 0.0
    pos_constraint.target_point_offset.y = 0.0
    pos_constraint.target_point_offset.z = 0.0

    bounding = BoundingVolume()
    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [defaults.position_tolerance]
    bounding.primitives = [sphere]

    center = GeometryPose()
    center.position.x = px
    center.position.y = py
    center.position.z = pz
    center.orientation.w = 1.0
    bounding.primitive_poses = [center]
    pos_constraint.constraint_region = bounding
    pos_constraint.weight = 1.0

    ori_constraint = OrientationConstraint()
    ori_constraint.header.frame_id = base_frame
    ori_constraint.link_name = ee_link
    ori_constraint.orientation.x = qx
    ori_constraint.orientation.y = qy
    ori_constraint.orientation.z = qz
    ori_constraint.orientation.w = qw
    ori_constraint.absolute_x_axis_tolerance = defaults.orientation_tolerance
    ori_constraint.absolute_y_axis_tolerance = defaults.orientation_tolerance
    ori_constraint.absolute_z_axis_tolerance = defaults.orientation_tolerance
    ori_constraint.weight = 1.0

    goal_constraints = Constraints()
    goal_constraints.position_constraints = [pos_constraint]
    goal_constraints.orientation_constraints = [ori_constraint]

    request = MotionPlanRequest()
    request.group_name = group_name
    request.goal_constraints = [goal_constraints]
    request.path_constraints = cable_path_constraints()
    # Keep the tool near its requested radial-out orientation during approach,
    # not only at the destination. Joint 6 separately limits cable winding.
    path_orientation = copy.deepcopy(ori_constraint)
    path_orientation.parameterization = OrientationConstraint.ROTATION_VECTOR
    path_orientation.absolute_x_axis_tolerance = CABLE_LIMIT
    path_orientation.absolute_y_axis_tolerance = CABLE_LIMIT
    path_orientation.absolute_z_axis_tolerance = CABLE_LIMIT
    request.path_constraints.orientation_constraints = [path_orientation]

    request.num_planning_attempts = defaults.planning_attempts
    request.allowed_planning_time = defaults.allowed_planning_time
    request.max_velocity_scaling_factor = defaults.velocity_scaling_factor
    request.max_acceleration_scaling_factor = defaults.acceleration_scaling_factor

    ws = WorkspaceParameters()
    ws.header.frame_id = base_frame
    ws.min_corner.x = defaults.workspace_min_corner[0]
    ws.min_corner.y = defaults.workspace_min_corner[1]
    ws.min_corner.z = defaults.workspace_min_corner[2]
    ws.max_corner.x = defaults.workspace_max_corner[0]
    ws.max_corner.y = defaults.workspace_max_corner[1]
    ws.max_corner.z = defaults.workspace_max_corner[2]
    request.workspace_parameters = ws

    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options.plan_only = True

    future = move_group_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_goal_send_sec)
    if not future.done() or future.result() is None:
        if diagnostics is not None:
            diagnostics["status"] = "send_timeout_or_failure"
        node.get_logger().error("Failed to send MoveGroup goal")
        return False

    handle = future.result()
    if not handle.accepted:
        if diagnostics is not None:
            diagnostics["status"] = "goal_rejected"
        node.get_logger().error("MoveGroup goal rejected")
        return False

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_result_sec)
    if not result_future.done() or result_future.result() is None:
        if diagnostics is not None:
            diagnostics["status"] = "result_timeout_or_failure"
        node.get_logger().error("MoveGroup result not received")
        return False

    result = result_future.result().result
    if diagnostics is not None:
        from rosidl_runtime_py.convert import message_to_ordereddict
        diagnostics.update(status='result_received', action_status=result_future.result().status,
                           moveit_error_code=result.error_code.val,
                           planned_trajectory=message_to_ordereddict(result.planned_trajectory),
                           trajectory_start=message_to_ordereddict(result.trajectory_start),
                           executed_trajectory=message_to_ordereddict(result.executed_trajectory))
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
        node.get_logger().error(f"MoveGroup failed: error_code={result.error_code.val}")
        return False

    success = execute_checked_plan(node, result.planned_trajectory, timeout_result_sec)
    if diagnostics is not None:
        diagnostics['execution_success'] = success
    return success


# Joint names for the IRB120 manipulator group (order matches robot URDF/SRDF)
_IRB120_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
]

DEFAULT_JOINT_TOLERANCE = 0.01  # rad


def plan_and_execute_joint_goal(
    node,
    move_group_client,
    *,
    joint_positions: Sequence[float],
    group_name: str = DEFAULT_GROUP_NAME,
    joint_names: Sequence[str] = _IRB120_JOINT_NAMES,
    joint_tolerance: float = DEFAULT_JOINT_TOLERANCE,
    velocity_scaling_factor: float = 0.1,
    acceleration_scaling_factor: float = 0.1,
    planning_attempts: int = 5,
    allowed_planning_time: float = 10.0,
    timeout_server_sec: float = 10.0,
    timeout_goal_send_sec: float = 15.0,
    timeout_result_sec: float = 30.0,
) -> bool:
    """Plan and execute a joint-space goal using MoveGroup.

    joint_positions: 6-element sequence of target joint angles in radians,
                     ordered to match joint_names.
    """
    node._last_joint_goal_verification = dict(verified=False, reason='joint action not completed')
    if len(joint_positions) != len(joint_names):
        node.get_logger().error(
            f"joint_positions length {len(joint_positions)} != joint_names length {len(joint_names)}"
        )
        return False

    node.get_logger().info(
        f"Moving to joint target: {[round(v, 4) for v in joint_positions]}"
    )

    if not move_group_client.wait_for_server(timeout_sec=timeout_server_sec):
        node.get_logger().error("MoveGroup action server not available")
        return False

    if not _wait_for_joint_states(node, timeout_sec=10.0):
        return False

    goal_constraints = Constraints()
    for name, position in zip(joint_names, joint_positions):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = position
        jc.tolerance_above = joint_tolerance
        jc.tolerance_below = joint_tolerance
        jc.weight = 1.0
        goal_constraints.joint_constraints.append(jc)

    request = MotionPlanRequest()
    request.group_name = group_name
    request.goal_constraints = [goal_constraints]
    request.path_constraints = cable_path_constraints()
    request.num_planning_attempts = planning_attempts
    request.allowed_planning_time = allowed_planning_time
    request.max_velocity_scaling_factor = velocity_scaling_factor
    request.max_acceleration_scaling_factor = acceleration_scaling_factor

    goal = MoveGroup.Goal()
    goal.request = request
    goal.planning_options.plan_only = True

    future = move_group_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_goal_send_sec)
    if not future.done() or future.result() is None:
        node.get_logger().error("Failed to send MoveGroup joint goal")
        return False

    handle = future.result()
    if not handle.accepted:
        node.get_logger().error("MoveGroup joint goal rejected")
        return False

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_result_sec)
    if not result_future.done() or result_future.result() is None:
        node.get_logger().error("MoveGroup joint result not received")
        return False

    result = result_future.result().result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
        node.get_logger().error(f"MoveGroup joint goal failed: error_code={result.error_code.val}")
        return False

    if not execute_checked_plan(node, result.planned_trajectory, timeout_result_sec):
        return False
    if not verify_joint_goal(node, joint_names, joint_positions, joint_tolerance):
        return False
    node.get_logger().info("Reached joint target; measured joints verified.")
    return True
