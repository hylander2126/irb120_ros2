#!/usr/bin/env python3
"""Single-shot MoveIt planning helpers for IRB120 control scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import rclpy
from geometry_msgs.msg import Pose as GeometryPose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive


DEFAULT_GROUP_NAME = "manipulator"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_EE_LINK = "finger_ball_center"
DEFAULT_POSITION_TOLERANCE = 0.002
DEFAULT_ORIENTATION_TOLERANCE = 0.01


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
    timeout_server_sec: float = 10.0,
    timeout_goal_send_sec: float = 15.0,
    timeout_result_sec: float = 30.0,
) -> bool:
    """Plan and execute a constrained pose goal using MoveGroup.

    The helper sends one goal, waits for the plan to be accepted, and then waits
    for the result before returning.
    """

    defaults = defaults or PoseGoalDefaults()
    px, py, pz = target_position
    qx, qy, qz, qw = target_orientation

    node.get_logger().info(f"Moving to pose target: ({px}, {py}, {pz})")

    if not move_group_client.wait_for_server(timeout_sec=timeout_server_sec):
        node.get_logger().error("MoveGroup action server not available")
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

    future = move_group_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_goal_send_sec)
    if not future.done() or future.result() is None:
        node.get_logger().error("Failed to send MoveGroup goal")
        return False

    handle = future.result()
    if not handle.accepted:
        node.get_logger().error("MoveGroup goal rejected")
        return False

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_result_sec)
    if not result_future.done() or result_future.result() is None:
        node.get_logger().error("MoveGroup result not received")
        return False

    result = result_future.result().result
    if result.error_code.val != MoveItErrorCodes.SUCCESS:
        node.get_logger().error(f"MoveGroup failed: error_code={result.error_code.val}")
        return False

    node.get_logger().info("Reached pose target.")
    return True