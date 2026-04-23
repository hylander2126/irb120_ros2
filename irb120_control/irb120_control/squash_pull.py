#!/usr/bin/env python3
"""Simple squash-and-pull Cartesian force controller for the IRB120.

Approach: MoveIt plans to the pre-squash pose (position + orientation).
Squash/Pull/Retract: MoveIt Servo twist commands on /servo_node/delta_twist_cmds.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import WrenchStamped
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
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformException, TransformListener


BASE_FRAME = "base_link"
EE_LINK = "finger_ball_center"
PLANNING_GROUP = "manipulator"
SERVO_TOPIC = "/servo_node/delta_twist_cmds"
WRENCH_TOPIC = "/netft_data_transformed"

# Pre-squash pose — position and orientation the EE must reach before descending.
# Tune quaternion to match your desired squash orientation.
PRE_SQUASH_X = 0.545
PRE_SQUASH_Y = 0.00
PRE_SQUASH_Z = 0.214
PRE_SQUASH_QX = 0.0
PRE_SQUASH_QY = 0.0
PRE_SQUASH_QZ = 0.0
PRE_SQUASH_QW = 1.0

SPHERE_POS_CONSTRAINT = 0.002 # 2mm radius for pre-squash position constraint
RAD_ORIENT_CONSTRAINT = 0.01 # ~0.5 degrees for pre-squash orientation constraint

FORCE_REF_N = 2.0
FORCE_HARD_LIMIT_N = 10.0
CONTACT_STABLE_SAMPLES = 1 # how many consecutive ref_n samples needed to stop squashing?

# MoveIt Servo linear scale factor (servo.yaml: scale.linear).
# All speeds below are in m/s; they are divided by this before publishing.
SERVO_LINEAR_SCALE = 0.07

DESCEND_SPEED = 0.005    # m/s
PULL_SPEED = 0.008       # m/s
RETRACT_SPEED = 0.010    # m/s
RETRACT_CLEARANCE = 0.020
PULL_DISTANCE = 0.030

SQUASH_TIMEOUT_SEC = 8.0
PULL_TIMEOUT_SEC = 8.0

KP_FORCE = 0.0015
KI_FORCE = 0.00015
MAX_NORMAL_SPEED = 0.010

FORCE_LP_ALPHA = 0.50 # higher = responsive, noisier
CONTROL_HZ = 100.0
REQUIRE_OPERATOR_CONFIRM = True


class SquashPull(Node):
    def __init__(self) -> None:
        super().__init__("squash_pull")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._twist_pub = self.create_publisher(TwistStamped, SERVO_TOPIC, 10)
        self._wrench_sub = self.create_subscription(WrenchStamped, WRENCH_TOPIC, self._on_wrench, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._timer = None  # Created in main() after approach completes
        
        # Open log file for command/force correlation during squash phase.
        workspace_root = Path(__file__).resolve().parents[4]
        log_dir = workspace_root / "runtime_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = log_dir / f"squash_pull_log_{timestamp}.csv"
        self._log_file = self._log_path.open("w")
        self._log_file.write('timestamp_s,state,filtered_force_N,cmd_vx,cmd_vy,cmd_vz\n')
        self._log_file.flush()
        self._log_start_time = None

        self._state = "SQUASH"
        self._done = False
        self._integral = 0.0
        self._contact_count = 0
        self._pull_start_x = None
        self._filtered_force = 0.0
        self._have_force = False
        self._last_nonfinite_warn = 0.0
        self._last_tf_warn_time = 0.0
        self._state_start_time = 0.0
        self._contact_felt = False
        self._lull_prompted = False

    def move_to_pre_squash(self) -> bool:
        """Blocking MoveIt call to reach PRE_SQUASH pose. Returns True on success."""
        self.get_logger().info(
            f"Moving to pre-squash pose: ({PRE_SQUASH_X}, {PRE_SQUASH_Y}, {PRE_SQUASH_Z})"
        )

        # Wait until MoveIt is ready to accept a planning request.
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroup action server not available")
            return False

        # Constrain the end effector position to a small sphere around the target.
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = BASE_FRAME
        pos_constraint.link_name = EE_LINK
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0
        bounding = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [SPHERE_POS_CONSTRAINT]
        bounding.primitives = [sphere]
        from geometry_msgs.msg import Pose as _Pose
        center = _Pose()
        center.position.x = PRE_SQUASH_X
        center.position.y = PRE_SQUASH_Y
        center.position.z = PRE_SQUASH_Z
        center.orientation.w = 1.0
        bounding.primitive_poses = [center]
        pos_constraint.constraint_region = bounding
        pos_constraint.weight = 1.0

        # Constrain the end effector orientation to the target quaternion.
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = BASE_FRAME
        ori_constraint.link_name = EE_LINK
        ori_constraint.orientation.x = PRE_SQUASH_QX
        ori_constraint.orientation.y = PRE_SQUASH_QY
        ori_constraint.orientation.z = PRE_SQUASH_QZ
        ori_constraint.orientation.w = PRE_SQUASH_QW
        ori_constraint.absolute_x_axis_tolerance = RAD_ORIENT_CONSTRAINT # radians, ~0.5 degrees
        ori_constraint.absolute_y_axis_tolerance = RAD_ORIENT_CONSTRAINT # radians, ~0.5 degrees (critical axis)
        ori_constraint.absolute_z_axis_tolerance = RAD_ORIENT_CONSTRAINT
        ori_constraint.weight = 1.0

        goal_constraints = Constraints()
        goal_constraints.position_constraints = [pos_constraint]
        goal_constraints.orientation_constraints = [ori_constraint]

        # Build the MoveIt planning request for the manipulator group.
        request = MotionPlanRequest()
        request.group_name = PLANNING_GROUP
        request.goal_constraints = [goal_constraints]
        request.num_planning_attempts = 5
        request.allowed_planning_time = 10.0
        request.max_velocity_scaling_factor = 0.1
        request.max_acceleration_scaling_factor = 0.1

        # Keep planning bounded to the robot's reachable work area.
        ws = WorkspaceParameters()
        ws.header.frame_id = BASE_FRAME
        ws.min_corner.x = -1.5
        ws.min_corner.y = -1.5
        ws.min_corner.z = -0.5
        ws.max_corner.x = 1.5
        ws.max_corner.y = 1.5
        ws.max_corner.z = 2.0
        request.workspace_parameters = ws

        # Wrap the request in the MoveGroup action goal.
        goal = MoveGroup.Goal()
        goal.request = request

        # Send the goal and wait for MoveIt to accept it.
        future = self._move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Failed to send MoveGroup goal")
            return False

        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("MoveGroup goal rejected")
            return False

        # Wait for the planned motion to finish and check the result code.
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveGroup result not received")
            return False

        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveGroup failed: error_code={result.error_code.val}")
            return False

        self.get_logger().info("Reached pre-squash pose.")
        return True

    def _on_wrench(self, msg: WrenchStamped) -> None:
        force_z = abs(msg.wrench.force.z)
        if not math.isfinite(force_z):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_nonfinite_warn > 1.0:
                self._last_nonfinite_warn = now
                self.get_logger().error("Received non-finite z force sample (NaN/Inf). Holding last valid value.")
            return
        if not self._have_force:
            self._filtered_force = force_z
            self._have_force = True
        else:
            self._filtered_force = (1.0 - FORCE_LP_ALPHA) * self._filtered_force + FORCE_LP_ALPHA * force_z

        if self._filtered_force > 0.25:
            print(f"\r  z_force: {self._filtered_force:6.2f} N  state: {self._state:<8}", end="", flush=True)

    def _lookup_pose(self) -> PoseStamped | None:
        try:
            transform = self._tf_buffer.lookup_transform(BASE_FRAME, EE_LINK, rclpy.time.Time())
        except TransformException as exc:
            self._warn_throttled(f"Waiting for TF {BASE_FRAME} -> {EE_LINK}: {exc}")
            return None

        pose = PoseStamped()
        pose.header.frame_id = BASE_FRAME
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    def _warn_throttled(self, message: str, throttle_hz: float = 0.2) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        min_interval = 1.0 / throttle_hz
        if now - self._last_tf_warn_time > min_interval:
            self._last_tf_warn_time = now
            self.get_logger().warn(message)

    def _publish_twist(self, vx: float, vy: float, vz: float) -> None:
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_nonfinite_warn > 1.0:
                self._last_nonfinite_warn = now
                self.get_logger().error(
                    "Refusing to publish non-finite twist command (NaN/Inf). Sending zero instead."
                )
            vx, vy, vz = 0.0, 0.0, 0.0

        # Log command and current force for diagnostics
        now_ns = self.get_clock().now().nanoseconds
        if self._log_start_time is None:
            self._log_start_time = now_ns
        elapsed_s = (now_ns - self._log_start_time) * 1e-9
        self._log_file.write(f'{elapsed_s:.6f},{self._state},{self._filtered_force:.4f},{vx:.6f},{vy:.6f},{vz:.6f}\n')
        self._log_file.flush()

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.linear.x = vx / SERVO_LINEAR_SCALE
        msg.twist.linear.y = vy / SERVO_LINEAR_SCALE
        msg.twist.linear.z = vz / SERVO_LINEAR_SCALE
        self._twist_pub.publish(msg)

    def _publish_zero(self) -> None:
        self._publish_twist(0.0, 0.0, 0.0)

    def _transition(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f"{self._state} -> {state}")
            self._state = state
            self._state_start_time = self.get_clock().now().nanoseconds * 1e-9

    def _operator_confirm(self, message: str) -> bool:
        if not REQUIRE_OPERATOR_CONFIRM:
            return True

        self._publish_zero()
        self.get_logger().warn(message)
        try:
            response = input("Press Enter to continue, or type 'q' to abort: ").strip().lower()
        except EOFError:
            response = "q"

        if response == "q":
            self._done = True
            self.get_logger().warn("Operator aborted squash-pull sequence")
            return False
        return True

    def _tick(self) -> None:
        if self._done:
            self._publish_zero()
            return

        pose = self._lookup_pose()
        if pose is None:
            self._publish_zero()
            return

        x = pose.pose.position.x
        z = pose.pose.position.z

        if self._have_force and self._filtered_force > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(f"Hard z force limit exceeded: {self._filtered_force:.2f} N — retracting")
            self._transition("RETRACT")

        if self._have_force and not math.isfinite(self._filtered_force):
            self.get_logger().error("Filtered force became non-finite. Aborting.")
            self._publish_zero()
            self._done = True
            return

        if self._state == "SQUASH":
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._state_start_time > SQUASH_TIMEOUT_SEC:
                self.get_logger().error(f"SQUASH timed out after {SQUASH_TIMEOUT_SEC:.0f}s — no contact. Retracting.")
                self._transition("RETRACT")
                return
            if self._have_force and self._filtered_force > 0.25 and not self._contact_felt:
                self._contact_felt = True
                self.get_logger().info("Contact felt — halving descend speed")
            descend = (DESCEND_SPEED * 0.5) if self._contact_felt else DESCEND_SPEED
            self._publish_twist(0.0, 0.0, -descend)
            if self._have_force and self._filtered_force >= FORCE_REF_N:
                self._contact_count += 1
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._transition("LULL")
            else:
                self._contact_count = 0
            return

        if self._state == "LULL":
            self._publish_zero()
            if self._lull_prompted:
                return

            self._lull_prompted = True
            if not self._operator_confirm(
                "Squash phase complete. Hold position, then press Enter to start the pull phase."
            ):
                return

            self._pull_start_x = x
            self._integral = 0.0
            self._transition("PULL")
            return

        if self._state == "PULL":
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._state_start_time > PULL_TIMEOUT_SEC:
                self.get_logger().error(f"PULL timed out after {PULL_TIMEOUT_SEC:.0f}s. Retracting.")
                self._transition("RETRACT")
                return
            error = FORCE_REF_N - self._filtered_force
            self._integral = self._clamp(self._integral + error * (1.0 / CONTROL_HZ), 2.0)
            normal_cmd = KP_FORCE * error + KI_FORCE * self._integral
            normal_cmd = self._clamp(normal_cmd, MAX_NORMAL_SPEED)

            pull_cmd = -PULL_SPEED
            normal_z = -normal_cmd
            self._publish_twist(pull_cmd, 0.0, normal_z)

            if self._pull_start_x is not None and abs(x - self._pull_start_x) >= PULL_DISTANCE:
                self._transition("RETRACT")
            return

        if self._state == "RETRACT":
            self._publish_twist(0.0, 0.0, RETRACT_SPEED)
            if z >= PRE_SQUASH_Z + RETRACT_CLEARANCE:
                self._publish_zero()
                self._done = True
                self.get_logger().info("Squash-pull sequence complete")
            return


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SquashPull()
    try:
        # Phase 1: MoveIt approach to pre-squash pose (blocking, with correct orientation)
        if not node.move_to_pre_squash():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        if not node._operator_confirm(
            "At pre-squash pose. Confirm clear contact conditions before descending."
        ):
            return 0

        # Phase 2: Servo-based squash/pull/retract
        node._state_start_time = node.get_clock().now().nanoseconds * 1e-9
        node._timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        if hasattr(node, '_log_file'):
            node._log_file.close()
            print(f"\nSquash-pull log written to {node._log_path}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
