#!/usr/bin/env python3
"""Minimal MoveIt pose-goal tester for IRB120 kinematics checks."""

from __future__ import annotations

import math

import rclpy
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm

from irb120_control.controllers.moveit_single_shot import (
    DEFAULT_BASE_FRAME,
    DEFAULT_EE_LINK,
    DEFAULT_GROUP_NAME,
    PoseGoalDefaults,
    plan_and_execute_pose_goal,
)


TARGET_POSITION = (0.55, 0.0, 0.314)#0.400 + 0.1038, -0.082, 0.100)
TARGET_RPY = (0.0, 0.0, 0.0)


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert roll/pitch/yaw to a quaternion in x, y, z, w order."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def main() -> int:
    rclpy.init()
    node = rclpy.create_node("pose_goal_accuracy_tester")
    move_group_client = ActionClient(node, MoveGroup, "/move_action")

    if not ensure_egm_active(node):
        node.get_logger().error("EGM is not active. Please start the EGM server on the robot and try again.")
        return 1
    

    target_position = TARGET_POSITION
    target_orientation = rpy_to_quat(*TARGET_RPY)
    defaults = PoseGoalDefaults(
        position_tolerance=0.0005, #0.001,
        orientation_tolerance=0.001, # 0.005
        planning_attempts=8,
        allowed_planning_time=10.0,
        velocity_scaling_factor=0.05,
        acceleration_scaling_factor=0.05,
    )

    node.get_logger().info(
        "Testing pose goal: "
        f"pos=({TARGET_POSITION[0]:.3f}, {TARGET_POSITION[1]:.3f}, {TARGET_POSITION[2]:.3f}) m, "
        f"rpy=({TARGET_RPY[0]:.3f}, {TARGET_RPY[1]:.3f}, {TARGET_RPY[2]:.3f}) rad, "
        f"tolerances=({defaults.position_tolerance:.4f} m, {defaults.orientation_tolerance:.4f} rad)"
    )

    try:
        success = plan_and_execute_pose_goal(
            node,
            move_group_client,
            group_name=DEFAULT_GROUP_NAME,
            base_frame="world", #DEFAULT_BASE_FRAME,
            ee_link=DEFAULT_EE_LINK,
            target_position=target_position,
            target_orientation=target_orientation,
            defaults=defaults,
            velocity_scale=0.05,
            acceleration_scale=0.05,
            timeout_server_sec=10.0,
            timeout_goal_send_sec=15.0,
            timeout_result_sec=30.0,
        )
    finally:
        deactivate_egm(node)
        node.destroy_node()
        rclpy.shutdown()

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())