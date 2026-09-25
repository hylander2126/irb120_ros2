#!/usr/bin/env python3
"""Run saved hand-eye calibration joint poses with manual sample gating.

Every move is planned by move_group (collision-checked against the robot
model, including the table and bracket) and executed through it; nothing is
sent straight to the trajectory controller. Requires bringup_stack's
move_group.

Usage examples:
  ros2 run irb120_handeye run_calibration_poses
  ros2 run irb120_handeye run_calibration_poses --pose-file joints_8_32mm.yaml
  ros2 run irb120_handeye run_calibration_poses --velocity-scaling 0.1 --settle-time 2.0
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml

PLANNING_GROUP = "manipulator"
PLANNING_TIME_SEC = 5.0
PLANNING_ATTEMPTS = 3
GOAL_JOINT_TOLERANCE_RAD = 1e-3


def _status_name(status: int) -> str:
    table = {
        0: "UNKNOWN",
        1: "ACCEPTED",
        2: "EXECUTING",
        3: "CANCELING",
        4: "SUCCEEDED",
        5: "CANCELED",
        6: "ABORTED",
    }
    return table.get(status, str(status))


def _resolve_pose_path(pose_path: Optional[str], pose_file: str) -> str:
    if pose_path:
        return os.path.abspath(pose_path)
    return os.path.join(
        get_package_share_directory("irb120_handeye"),
        "calibrations",
        pose_file,
    )


def _load_pose_yaml(path: str) -> Tuple[List[str], List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    joint_names = data.get("joint_names", [])
    joint_values = data.get("joint_values", [])

    if not joint_names or not joint_values:
        raise RuntimeError(f"Invalid pose file: {path} (missing joint_names or joint_values)")

    expected = len(joint_names)
    for i, row in enumerate(joint_values):
        if len(row) != expected:
            raise RuntimeError(
                f"Invalid pose row {i} in {path}: expected {expected} values, got {len(row)}"
            )

    return [str(j) for j in joint_names], [[float(v) for v in row] for row in joint_values]


class HandeyePoseRunner(Node):
    def __init__(self, joint_names: List[str]):
        super().__init__("handeye_pose_runner")
        self.joint_names = joint_names
        self._joint_map: Dict[str, float] = {}

        self._joint_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_state, 20)
        self._plan_client = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        self._execute_client = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

    def _on_joint_state(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._joint_map[name] = float(pos)

    def wait_for_joint_states(self, timeout_sec: float = 10.0) -> bool:
        end = time.time() + timeout_sec
        while rclpy.ok() and time.time() < end:
            if all(j in self._joint_map for j in self.joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _current_positions(self) -> List[float]:
        return [self._joint_map[name] for name in self.joint_names]

    def wait_for_move_group(self, timeout_sec: float = 10.0) -> bool:
        return (self._plan_client.wait_for_service(timeout_sec=timeout_sec)
                and self._execute_client.wait_for_server(timeout_sec=timeout_sec))

    def plan(self, target_positions: List[float], velocity_scaling: float, acceleration_scaling: float,
             start_positions: Optional[List[float]] = None) -> Tuple[Optional[RobotTrajectory], str]:
        """Collision-checked joint-space plan to target_positions.

        Plans from start_positions if given (used to check a whole pose
        sequence before moving), else from move_group's current robot state.
        Returns (trajectory, "") or (None, reason).
        """
        req = GetMotionPlan.Request()
        mpr = req.motion_plan_request
        mpr.group_name = PLANNING_GROUP
        mpr.num_planning_attempts = PLANNING_ATTEMPTS
        mpr.allowed_planning_time = PLANNING_TIME_SEC
        mpr.max_velocity_scaling_factor = velocity_scaling
        mpr.max_acceleration_scaling_factor = acceleration_scaling
        if start_positions is None:
            mpr.start_state.is_diff = True
        else:
            mpr.start_state.joint_state.name = list(self.joint_names)
            mpr.start_state.joint_state.position = [float(v) for v in start_positions]
        mpr.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=name, position=float(pos), tolerance_above=GOAL_JOINT_TOLERANCE_RAD,
                            tolerance_below=GOAL_JOINT_TOLERANCE_RAD, weight=1.0)
            for name, pos in zip(self.joint_names, target_positions)])]

        future = self._plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=PLANNING_TIME_SEC + 10.0)
        if not future.done() or future.result() is None:
            return None, "no response from move_group /plan_kinematic_path"
        response = future.result().motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f"planning failed (MoveIt error code {response.error_code.val})"
        return response.trajectory, ""

    def execute(self, trajectory: RobotTrajectory) -> bool:
        """Execute a planned trajectory through move_group (which checks the
        robot is still at the trajectory's start before moving)."""
        send_future = self._execute_client.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory))
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Failed to send execute_trajectory goal")
            return False

        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("execute_trajectory goal rejected")
            return False

        result_future = handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("execute_trajectory result not received")
            return False

        wrapped = result_future.result()
        if wrapped.status != 4 or wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Execution finished with status={_status_name(wrapped.status)}, "
                                    f"MoveIt error code {wrapped.result.error_code.val}")
            return False
        return True


def trajectory_duration(trajectory: RobotTrajectory) -> float:
    points = trajectory.joint_trajectory.points
    if not points:
        return 0.0
    t = points[-1].time_from_start
    return t.sec + t.nanosec * 1e-9


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run saved hand-eye calibration poses with manual sampling.")
    p.add_argument(
        "--pose-file",
        default="joints_5_6mm.yaml",
        help="Pose YAML filename under share/irb120_handeye/calibrations/",
    )
    p.add_argument(
        "--pose-path",
        default=None,
        help="Absolute or relative path to a pose YAML (overrides --pose-file).",
    )
    p.add_argument("--velocity-scaling", type=float, default=0.15, help="MoveIt max velocity scaling factor.")
    p.add_argument("--settle-time", type=float, default=1.5, help="Seconds to wait after each move.")
    p.add_argument(
        "--auto-continue",
        action="store_true",
        help="Do not wait for Enter between poses.",
    )
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()

    pose_path = _resolve_pose_path(args.pose_path, args.pose_file)
    if not os.path.isfile(pose_path):
        print(f"Pose file not found: {pose_path}")
        return 1

    try:
        joint_names, joint_values = _load_pose_yaml(pose_path)
    except Exception as exc:
        print(str(exc))
        return 1

    rclpy.init()
    node = HandeyePoseRunner(joint_names)

    try:
        node.get_logger().info(f"Loaded {len(joint_values)} poses from: {pose_path}")
        node.get_logger().info(f"Joint order: {joint_names}")

        if not node.wait_for_joint_states(timeout_sec=10.0):
            node.get_logger().error("No complete joint state received. Is bringup running?")
            return 2
        if not node.wait_for_move_group(timeout_sec=10.0):
            node.get_logger().error("move_group not available. Is bringup_stack running?")
            return 2

        if not args.auto_continue:
            input(
                "Ready to move through calibration poses. In RViz HandEye, use Take sample manually.\n"
                "Press Enter to start..."
            )

        total = len(joint_values)
        for i, target in enumerate(joint_values, start=1):
            node.get_logger().info(f"Moving to pose {i}/{total}")
            trajectory, reason = node.plan(target, args.velocity_scaling, args.velocity_scaling)
            if trajectory is None:
                node.get_logger().error(f"Pose {i}: {reason}")
            ok = trajectory is not None and node.execute(trajectory)
            if not ok:
                if args.auto_continue:
                    continue
                resp = input("Move failed. Press Enter to continue to next pose, or Ctrl+C to abort.")
                _ = resp
                continue

            settle_end = time.time() + max(0.0, args.settle_time)
            while rclpy.ok() and time.time() < settle_end:
                rclpy.spin_once(node, timeout_sec=0.1)

            if not args.auto_continue:
                input(
                    f"Pose {i}/{total} reached and settled.\n"
                    "Take sample in RViz, then press Enter for next pose..."
                )

        node.get_logger().info("Completed all requested calibration poses.")
        return 0
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
