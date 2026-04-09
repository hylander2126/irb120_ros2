#!/usr/bin/env python3
"""
Temporary pipeline smoke-test for ABB IRB120 ROS2 stack.

Flow:
1) pp_to_main -> start_rapid -> start_egm_joint
2) switch to joint_trajectory_controller
3) read current joint state
4) send no-motion trajectory point
5) send tiny joint_6 pulse (+delta then back)
6) stop_egm on exit

Use only in safe conditions (manual mode, low speed override, clear workspace).
"""

import sys
import time
from typing import Dict, List

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from abb_robot_msgs.srv import TriggerWithResultCode
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


class MotionSmokeTester(Node):
    def __init__(self) -> None:
        super().__init__("motion_smoke_tester")

        self.joint_names: List[str] = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.latest_joint_positions: Dict[str, float] = {}

        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

        self.stop_rapid_cli = self.create_client(
            TriggerWithResultCode, "/rws_client/stop_rapid"
        )
        self.pp_to_main_cli = self.create_client(
            TriggerWithResultCode, "/rws_client/pp_to_main"
        )
        self.start_rapid_cli = self.create_client(
            TriggerWithResultCode, "/rws_client/start_rapid"
        )
        self.start_egm_cli = self.create_client(
            TriggerWithResultCode, "/rws_client/start_egm_joint"
        )
        self.stop_egm_cli = self.create_client(
            TriggerWithResultCode, "/rws_client/stop_egm"
        )
        self.switch_cli = self.create_client(
            SwitchController, "/controller_manager/switch_controller"
        )

        self.traj_action = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                self.latest_joint_positions[name] = float(pos)

    def _wait_for_service(self, client, name: str, timeout: float = 10.0) -> bool:
        end = time.time() + timeout
        while time.time() < end and rclpy.ok():
            if client.wait_for_service(timeout_sec=0.2):
                return True
        self.get_logger().error(f"Service not available: {name}")
        return False

    def _call_trigger(self, client, name: str, timeout: float = 8.0) -> bool:
        if not self._wait_for_service(client, name):
            return False
        fut = client.call_async(TriggerWithResultCode.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        if not fut.done() or fut.result() is None:
            self.get_logger().error(f"Failed trigger call: {name}")
            return False

        res = fut.result()
        self.get_logger().info(
            f"{name} -> code={res.result_code}, msg='{res.message}'"
        )
        return res.result_code == 1

    def _switch_to_trajectory(self) -> bool:
        if not self._wait_for_service(self.switch_cli, "/controller_manager/switch_controller"):
            return False

        req = SwitchController.Request()
        req.activate_controllers = ["joint_trajectory_controller"]
        req.deactivate_controllers = []
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = False
        req.timeout = Duration(sec=1, nanosec=0)

        fut = self.switch_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=8.0)
        if not fut.done() or fut.result() is None:
            self.get_logger().error("switch_controller call failed")
            return False

        ok = bool(fut.result().ok)
        if ok:
            self.get_logger().info("Switched to joint_trajectory_controller")
        else:
            self.get_logger().error(f"Controller switch rejected: {fut.result().message}")
        return ok

    def _wait_for_joint_state(self, timeout: float = 5.0) -> bool:
        end = time.time() + timeout
        while time.time() < end and rclpy.ok():
            if all(name in self.latest_joint_positions for name in self.joint_names):
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error("Did not receive complete /joint_states in time")
        return False

    def _send_trajectory(self, points: List[JointTrajectoryPoint]) -> bool:
        if not self.traj_action.wait_for_server(timeout_sec=8.0):
            self.get_logger().error("Trajectory action server not available")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points = points

        send_future = self.traj_action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=8.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Failed to send trajectory goal")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("No trajectory result received")
            return False

        status = result_future.result().status
        if status == 4:
            self.get_logger().info("Trajectory goal succeeded")
            return True

        self.get_logger().error(f"Trajectory goal failed with status={status}")
        return False

    def run(self) -> int:
        self._call_trigger(self.stop_rapid_cli, "/rws_client/stop_rapid")
        if not self._call_trigger(self.pp_to_main_cli, "/rws_client/pp_to_main"):
            return 2
        if not self._call_trigger(self.start_rapid_cli, "/rws_client/start_rapid"):
            return 3
        if not self._call_trigger(self.start_egm_cli, "/rws_client/start_egm_joint"):
            return 4

        if not self._switch_to_trajectory():
            return 5

        if not self._wait_for_joint_state(timeout=6.0):
            return 6

        current = [self.latest_joint_positions[j] for j in self.joint_names]
        self.get_logger().info(f"Current joints: {current}")

        p0 = JointTrajectoryPoint()
        p0.positions = current
        p0.time_from_start = Duration(sec=3, nanosec=0)

        if not self._send_trajectory([p0]):
            return 7

        delta = 0.005
        up = current.copy()
        up[5] += delta

        p1 = JointTrajectoryPoint()
        p1.positions = current
        p1.time_from_start = Duration(sec=2, nanosec=0)

        p2 = JointTrajectoryPoint()
        p2.positions = up
        p2.time_from_start = Duration(sec=5, nanosec=0)

        p3 = JointTrajectoryPoint()
        p3.positions = current
        p3.time_from_start = Duration(sec=8, nanosec=0)

        if not self._send_trajectory([p1, p2, p3]):
            return 8

        self.get_logger().info("Motion smoke-test passed")
        return 0

    def cleanup(self) -> None:
        self._call_trigger(self.stop_egm_cli, "/rws_client/stop_egm", timeout=4.0)


def main() -> int:
    rclpy.init()
    node = MotionSmokeTester()
    code = 1
    try:
        code = node.run()
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
