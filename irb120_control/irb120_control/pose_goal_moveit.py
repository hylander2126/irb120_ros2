#!/usr/bin/env python3
"""Send a single EE pose goal to MoveIt move_group via the MoveGroup action.

By default this works in relative end-effector terms, not absolute Cartesian
targets. Set `use_relative_goal:=false` to use an explicit pose instead.
"""

import math
import sys
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import BoundingVolume
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import MotionPlanRequest
from moveit_msgs.msg import OrientationConstraint
from moveit_msgs.msg import PlanningOptions
from moveit_msgs.msg import PositionConstraint
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener


class PoseGoalMoveIt(Node):
    def __init__(self) -> None:
        super().__init__("pose_goal_moveit")

        self.declare_parameter("group_name", "manipulator")
        self.declare_parameter("ee_link", "tool0")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("use_relative_goal", True)

        self.declare_parameter("target_x", 0.35)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", 0.40)
        self.declare_parameter("target_qx", 0.0)
        self.declare_parameter("target_qy", 1.0)
        self.declare_parameter("target_qz", 0.0)
        self.declare_parameter("target_qw", 0.0)

        self.declare_parameter("relative_dx", 0.0)
        self.declare_parameter("relative_dy", 0.0)
        self.declare_parameter("relative_dz", 0.0)
        self.declare_parameter("relative_droll", 0.0)
        self.declare_parameter("relative_dpitch", 0.0)
        self.declare_parameter("relative_dyaw", 0.0)

        self.declare_parameter("pos_tolerance", 0.005)
        self.declare_parameter("orient_tolerance", 0.03)
        self.declare_parameter("allowed_planning_time", 5.0)
        self.declare_parameter("num_planning_attempts", 5)
        self.declare_parameter("max_velocity_scaling", 0.2)
        self.declare_parameter("max_acceleration_scaling", 0.2)

        self.move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _param(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _normalize_quaternion(q: Quaternion) -> Quaternion:
        n = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if n < 1e-9:
            return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return Quaternion(x=q.x / n, y=q.y / n, z=q.z / n, w=q.w / n)

    @staticmethod
    def _quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
        return Quaternion(
            x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        )

    @staticmethod
    def _quat_conjugate(q: Quaternion) -> Quaternion:
        return Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)

    @classmethod
    def _rotate_vector(cls, q: Quaternion, vector: tuple[float, float, float]) -> tuple[float, float, float]:
        vq = Quaternion(x=vector[0], y=vector[1], z=vector[2], w=0.0)
        rq = cls._quat_multiply(cls._quat_multiply(q, vq), cls._quat_conjugate(q))
        return (rq.x, rq.y, rq.z)

    @staticmethod
    def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy,
        )

    def _get_current_pose(self) -> PoseStamped:
        target_frame = str(self._param("target_frame"))
        ee_link = str(self._param("ee_link"))
        timeout_sec = 5.0
        end_time = time.time() + timeout_sec
        transform = None
        while rclpy.ok() and time.time() < end_time:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    ee_link,
                    rclpy.time.Time(),
                )
                break
            except TransformException:
                rclpy.spin_once(self, timeout_sec=0.1)

        if transform is None:
            raise RuntimeError(
                f"Timed out waiting for TF {target_frame} -> {ee_link}. "
                "Start the bringup launch and wait for robot_state_publisher / TF to come up."
            )

        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def _build_target_pose(self) -> PoseStamped:
        if bool(self._param("use_relative_goal")):
            current = self._get_current_pose()
            current_q = self._normalize_quaternion(current.pose.orientation)

            local_delta = (
                float(self._param("relative_dx")),
                float(self._param("relative_dy")),
                float(self._param("relative_dz")),
            )
            world_dx, world_dy, world_dz = self._rotate_vector(current_q, local_delta)

            target = PoseStamped()
            target.header.frame_id = current.header.frame_id
            target.pose.position.x = current.pose.position.x + world_dx
            target.pose.position.y = current.pose.position.y + world_dy
            target.pose.position.z = current.pose.position.z + world_dz

            delta_q = self._normalize_quaternion(
                self._quat_from_rpy(
                    float(self._param("relative_droll")),
                    float(self._param("relative_dpitch")),
                    float(self._param("relative_dyaw")),
                )
            )
            target.pose.orientation = self._normalize_quaternion(
                self._quat_multiply(current_q, delta_q)
            )
            return target

        q = Quaternion(
            x=float(self._param("target_qx")),
            y=float(self._param("target_qy")),
            z=float(self._param("target_qz")),
            w=float(self._param("target_qw")),
        )
        q = self._normalize_quaternion(q)

        pose = Pose()
        pose.position.x = float(self._param("target_x"))
        pose.position.y = float(self._param("target_y"))
        pose.position.z = float(self._param("target_z"))
        pose.orientation = q

        target = PoseStamped()
        target.header.frame_id = str(self._param("target_frame"))
        target.pose = pose
        return target

    def _build_goal_constraints(self, target: PoseStamped) -> Constraints:
        pos_tol = float(self._param("pos_tolerance"))
        orient_tol = float(self._param("orient_tolerance"))
        ee_link = str(self._param("ee_link"))

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [pos_tol, pos_tol, pos_tol]

        region = BoundingVolume()
        region.primitives = [primitive]
        region.primitive_poses = [target.pose]

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = target.header.frame_id
        pos_constraint.link_name = ee_link
        pos_constraint.constraint_region = region
        pos_constraint.weight = 1.0

        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = target.header.frame_id
        orient_constraint.link_name = ee_link
        orient_constraint.orientation = target.pose.orientation
        orient_constraint.absolute_x_axis_tolerance = orient_tol
        orient_constraint.absolute_y_axis_tolerance = orient_tol
        orient_constraint.absolute_z_axis_tolerance = orient_tol
        orient_constraint.weight = 1.0

        c = Constraints()
        c.position_constraints = [pos_constraint]
        c.orientation_constraints = [orient_constraint]
        return c

    def send_goal(self) -> bool:
        if not self.move_group_client.wait_for_server(timeout_sec=8.0):
            self.get_logger().error("MoveGroup action server /move_action not available")
            return False

        target = self._build_target_pose()
        goal_constraints = self._build_goal_constraints(target)

        req = MotionPlanRequest()
        req.group_name = str(self._param("group_name"))
        req.goal_constraints = [goal_constraints]
        req.num_planning_attempts = int(self._param("num_planning_attempts"))
        req.allowed_planning_time = float(self._param("allowed_planning_time"))
        req.max_velocity_scaling_factor = float(self._param("max_velocity_scaling"))
        req.max_acceleration_scaling_factor = float(self._param("max_acceleration_scaling"))

        planning = PlanningOptions()
        planning.plan_only = False

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = planning

        self.get_logger().info(
            "Sending pose goal in frame %s to link %s: [%.3f %.3f %.3f]"
            % (
                target.header.frame_id,
                str(self._param("ee_link")),
                target.pose.position.x,
                target.pose.position.y,
                target.pose.position.z,
            )
        )
        if bool(self._param("use_relative_goal")):
            self.get_logger().info(
                "Relative EE goal: dxyz=[%.3f %.3f %.3f], drpy=[%.3f %.3f %.3f]"
                % (
                    float(self._param("relative_dx")),
                    float(self._param("relative_dy")),
                    float(self._param("relative_dz")),
                    float(self._param("relative_droll")),
                    float(self._param("relative_dpitch")),
                    float(self._param("relative_dyaw")),
                )
            )

        send_future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done() or send_future.result() is None:
            self.get_logger().error("Failed to send goal to move_group")
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal was rejected")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("No MoveGroup result received")
            return False

        action_result = result_future.result()
        error_code = action_result.result.error_code.val
        if error_code == 1:
            self.get_logger().info("MoveGroup execution succeeded")
            return True

        self.get_logger().error(f"MoveGroup failed with error_code={error_code}")
        return False


def main() -> int:
    rclpy.init()
    node: Optional[PoseGoalMoveIt] = None
    rc = 1
    try:
        node = PoseGoalMoveIt()
        rc = 0 if node.send_goal() else 2
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
