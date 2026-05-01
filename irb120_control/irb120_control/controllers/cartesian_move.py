#!/usr/bin/env python3
"""Cartesian straight-line move helper for IRB120 control scripts.

Mirrors the interface of moveit_single_shot.plan_and_execute_pose_goal:
takes a node + pre-created action/service clients, plans a straight-line
Cartesian path through the given waypoints, and executes it.
"""

from __future__ import annotations

from typing import Sequence

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from moveit_msgs.srv import GetCartesianPath


DEFAULT_GROUP_NAME        = "manipulator"
DEFAULT_BASE_FRAME        = "base_link"
DEFAULT_EE_LINK           = "finger_ball_center"
DEFAULT_MAX_STEP          = 0.001   # m — IK resolution along the path
DEFAULT_JUMP_THRESHOLD    = 0.0     # disabled
DEFAULT_VELOCITY_SCALE    = 0.05
DEFAULT_ACCEL_SCALE       = 0.025
DEFAULT_MAX_CARTESIAN_SPEED = 0.0   # 0 = no ceiling (rely on velocity_scale)
DEFAULT_MIN_FRACTION      = 0.99    # abort if path coverage falls below this
DEFAULT_AVOID_COLLISIONS  = False


def plan_and_execute_cartesian(
    node,
    cartesian_client,
    execute_client,
    *,
    waypoints: Sequence[Pose],
    group_name: str = DEFAULT_GROUP_NAME,
    base_frame: str = DEFAULT_BASE_FRAME,
    ee_link: str = DEFAULT_EE_LINK,
    velocity_scale: float = DEFAULT_VELOCITY_SCALE,
    acceleration_scale: float | None = None,
    max_cartesian_speed: float = DEFAULT_MAX_CARTESIAN_SPEED,
    max_step: float = DEFAULT_MAX_STEP,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
    avoid_collisions: bool = DEFAULT_AVOID_COLLISIONS,
    min_fraction: float = DEFAULT_MIN_FRACTION,
    execute_timeout_sec: float = 60.0,
) -> bool:
    """Plan a Cartesian straight-line path and execute it.

    Args:
        node:              ROS node (used for logging and spinning).
        cartesian_client:  Service client for /compute_cartesian_path.
        execute_client:    Action client for /execute_trajectory.
        waypoints:         Ordered list of Pose targets in base_frame.
        velocity_scale:    Fraction of joint velocity limits (primary speed knob).
        acceleration_scale: Fraction of joint accel limits. Defaults to velocity_scale / 2.
        max_cartesian_speed: Hard EE speed ceiling in m/s (0 = disabled).
        min_fraction:      Minimum acceptable path coverage [0, 1].

    Returns:
        True on success, False on any planning or execution failure.
    """
    if acceleration_scale is None:
        acceleration_scale = velocity_scale / 2.0

    if not cartesian_client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("compute_cartesian_path service not available")
        return False

    req = GetCartesianPath.Request()
    req.header.frame_id                 = base_frame
    req.header.stamp                    = node.get_clock().now().to_msg()
    req.group_name                      = group_name
    req.link_name                       = ee_link
    req.waypoints                       = list(waypoints)
    req.max_step                        = max_step
    req.jump_threshold                  = jump_threshold
    req.avoid_collisions                = avoid_collisions
    req.max_velocity_scaling_factor     = velocity_scale
    req.max_acceleration_scaling_factor = acceleration_scale
    req.max_cartesian_speed             = max_cartesian_speed
    req.cartesian_speed_limited_link    = ee_link if max_cartesian_speed > 0.0 else ""
    req.start_state                     = RobotState()  # current state

    future = cartesian_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=15.0)
    if not future.done() or future.result() is None:
        node.get_logger().error("compute_cartesian_path did not respond")
        return False

    resp = future.result()
    node.get_logger().info(f"Cartesian path coverage: {resp.fraction * 100:.1f}%")
    if resp.fraction < min_fraction:
        node.get_logger().error(
            f"Cartesian path only {resp.fraction * 100:.1f}% complete "
            f"(required {min_fraction * 100:.0f}%) — aborting"
        )
        return False

    return _execute_trajectory(node, execute_client, resp.solution, execute_timeout_sec)


def _execute_trajectory(node, execute_client, trajectory, timeout_sec: float) -> bool:
    if not execute_client.wait_for_server(timeout_sec=5.0):
        node.get_logger().error("execute_trajectory action server not available")
        return False

    goal = ExecuteTrajectory.Goal()
    goal.trajectory = trajectory

    future = execute_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done() or future.result() is None:
        node.get_logger().error("execute_trajectory goal not accepted")
        return False

    handle = future.result()
    if not handle.accepted:
        node.get_logger().error("execute_trajectory goal rejected")
        return False

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    if not result_future.done() or result_future.result() is None:
        node.get_logger().error("execute_trajectory result not received")
        return False

    ec = result_future.result().result.error_code.val
    if ec != MoveItErrorCodes.SUCCESS:
        node.get_logger().error(f"execute_trajectory failed: error_code={ec}")
        return False

    return True
