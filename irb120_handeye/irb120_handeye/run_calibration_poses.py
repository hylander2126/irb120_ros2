#!/usr/bin/env python3
"""Run saved hand-eye calibration joint poses with manual sample gating.

Usage examples:
  ros2 run irb120_handeye run_calibration_poses
  ros2 run irb120_handeye run_calibration_poses --pose-file joints_8_32mm.yaml
  ros2 run irb120_handeye run_calibration_poses --move-time 5.0 --settle-time 2.0
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import yaml


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
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

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

    def move_to(self, target_positions: List[float], move_time_sec: float) -> bool:
        if not self._traj_client.wait_for_server(timeout_sec=8.0):
            self.get_logger().error("Trajectory action server not available")
            return False

        current = self._current_positions()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points = [
            JointTrajectoryPoint(
                positions=current,
                time_from_start=Duration(sec=0, nanosec=200_000_000),
            ),
            JointTrajectoryPoint(
                positions=target_positions,
                time_from_start=Duration(sec=int(move_time_sec), nanosec=int((move_time_sec % 1.0) * 1e9)),
            ),
        ]

        send_future = self._traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Failed to send trajectory goal")
            return False

        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False

        result_future = handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("Trajectory result not received")
            return False

        status = result_future.result().status
        if status != 4:
            self.get_logger().error(f"Trajectory finished with status={_status_name(status)}")
            return False
        return True


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run saved hand-eye calibration poses with manual sampling.")
    p.add_argument(
        "--pose-file",
        default="joints_20_14mm.yaml",
        help="Pose YAML filename under share/irb120_handeye/calibrations/",
    )
    p.add_argument(
        "--pose-path",
        default=None,
        help="Absolute or relative path to a pose YAML (overrides --pose-file).",
    )
    p.add_argument("--move-time", type=float, default=4.0, help="Seconds per move.")
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

        if not args.auto_continue:
            input(
                "Ready to move through calibration poses. In RViz HandEye, use Take sample manually.\n"
                "Press Enter to start..."
            )

        total = len(joint_values)
        for i, target in enumerate(joint_values, start=1):
            node.get_logger().info(f"Moving to pose {i}/{total}")
            ok = node.move_to(target, move_time_sec=max(0.5, args.move_time))
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
