#!/usr/bin/env python3
"""
Small ABB IRB120 motion smoke-test.

Flow: start EGM -> tiny joint_6 pulse via trajectory action -> stop EGM.
"""

import sys
import time
from typing import Dict, List, Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from abb_robot_msgs.srv import TriggerWithResultCode
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES: List[str] = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]
BASE_FRAME = "base_link"
EE_FRAME = "finger_ball_center"


def call_trigger(node: Node, client, name: str, timeout: float = 8.0) -> bool:
    if not client.wait_for_service(timeout_sec=timeout):
        node.get_logger().error(f"Service not available: {name}")
        return False
    future = client.call_async(TriggerWithResultCode.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        node.get_logger().error(f"Failed trigger call: {name}")
        return False
    res = future.result()
    node.get_logger().info(f"{name} -> code={res.result_code}, msg='{res.message}'")
    return res.result_code == 1


def wait_for_joint_state(node: Node, timeout: float = 6.0) -> Optional[List[float]]:
    latest: Dict[str, float] = {}

    def cb(msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in JOINT_NAMES:
                latest[name] = float(pos)

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)
    end = time.time() + timeout
    while time.time() < end and rclpy.ok():
        if all(name in latest for name in JOINT_NAMES):
            node.destroy_subscription(sub)
            return [latest[name] for name in JOINT_NAMES]
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    node.get_logger().error("Did not receive complete /joint_states in time")
    return None


def wait_for_ee_position(
    node: Node,
    base_frame: str = BASE_FRAME,
    ee_frame: str = EE_FRAME,
    timeout: float = 2.0,
) -> Optional[List[float]]:
    latest: Optional[List[float]] = None

    def cb(msg: TFMessage) -> None:
        nonlocal latest
        for tf in msg.transforms:
            if tf.header.frame_id == base_frame and tf.child_frame_id == ee_frame:
                t = tf.transform.translation
                latest = [float(t.x), float(t.y), float(t.z)]
                return

    sub = node.create_subscription(TFMessage, "/tf", cb, 10)
    end = time.time() + timeout
    while time.time() < end and rclpy.ok():
        if latest is not None:
            node.destroy_subscription(sub)
            return latest
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return None


def send_trajectory(
    node: Node,
    action: ActionClient,
    points: List[JointTrajectoryPoint],
) -> bool:
    if not action.wait_for_server(timeout_sec=8.0):
        node.get_logger().error("Trajectory action server not available")
        return False

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = JOINT_NAMES
    goal.trajectory.points = points
    send_future = action.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future, timeout_sec=8.0)
    if not send_future.done() or send_future.result() is None:
        node.get_logger().error("Failed to send trajectory goal")
        return False

    handle = send_future.result()
    if not handle.accepted:
        node.get_logger().error("Trajectory goal was rejected")
        return False

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=20.0)
    if not result_future.done() or result_future.result() is None:
        node.get_logger().error("No trajectory result received")
        return False
    status = result_future.result().status
    ok = status == 4
    if not ok:
        node.get_logger().error(f"Trajectory goal failed with status={status}")
    return ok


def main() -> int:
    rclpy.init()
    node = Node("motion_smoke_tester")
    start_egm = node.create_client(TriggerWithResultCode, "/rws_client/start_egm_joint")
    stop_egm = node.create_client(TriggerWithResultCode, "/rws_client/stop_egm")
    traj = ActionClient(
        node,
        FollowJointTrajectory,
        "/joint_trajectory_controller/follow_joint_trajectory",
    )
    exit_code = 1
    try:
        if not call_trigger(node, start_egm, "/rws_client/start_egm_joint"):
            return 2

        current = wait_for_joint_state(node, timeout=6.0)
        if current is None:
            return 3
        ee_before = wait_for_ee_position(node)

        print("Before motion:")
        print(f"  joints: {current}")
        if ee_before is not None:
            print(f"  ee xyz ({BASE_FRAME}->{EE_FRAME}): {ee_before}")
        else:
            print(f"  ee xyz ({BASE_FRAME}->{EE_FRAME}): unavailable")

        delta = 0.2
        up = current.copy()
        up[5] += delta

        p1 = JointTrajectoryPoint(positions=current, time_from_start=Duration(sec=2))
        p2 = JointTrajectoryPoint(positions=up, time_from_start=Duration(sec=5))
        p3 = JointTrajectoryPoint(positions=current, time_from_start=Duration(sec=8))

        if not send_trajectory(node, traj, [p1, p2, p3]):
            return 4

        after = wait_for_joint_state(node, timeout=2.0)
        ee_after = wait_for_ee_position(node)

        if after is not None:
            joint_delta = [a - b for a, b in zip(after, current)]
            print("After motion:")
            print(f"  joints: {after}")
            print(f"  joint delta: {joint_delta}")
        else:
            print("After motion:\n  joints: unavailable")

        if ee_before is not None and ee_after is not None:
            ee_delta = [a - b for a, b in zip(ee_after, ee_before)]
            print(f"  ee xyz ({BASE_FRAME}->{EE_FRAME}): {ee_after}")
            print(f"  ee delta xyz: {ee_delta}")
        elif ee_after is not None:
            print(f"  ee xyz ({BASE_FRAME}->{EE_FRAME}): {ee_after}")
            print("  ee delta xyz: unavailable (missing before pose)")
        else:
            print(f"  ee xyz ({BASE_FRAME}->{EE_FRAME}): unavailable")

        node.get_logger().info("Motion smoke-test passed")
        exit_code = 0
    finally:
        call_trigger(node, stop_egm, "/rws_client/stop_egm", timeout=4.0)
        node.destroy_node()
        rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
