#!/usr/bin/env python3
"""Simple squash-and-pull Cartesian force controller for the IRB120.

This node assumes MoveIt Servo is already running and listens on
`/servo_node/delta_twist_cmds`. Edit the hardcoded pose constants below to
match the object you want to test against.
"""

import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


BASE_FRAME = "base_link"
EE_LINK = "finger_ball_center"
SERVO_TOPIC = "/servo_node/delta_twist_cmds"
WRENCH_TOPIC = "/netft_data_transformed"

# Hardcoded pre-squash waypoint. Edit these values directly for your test cell.
PRE_SQUASH_X = 0.534
PRE_SQUASH_Y = 0.00
PRE_SQUASH_Z = 0.226

APPROACH_TOL = 0.010
APPROACH_MAX_SPEED = 0.030

FORCE_REF_N = 2.0
FORCE_HARD_LIMIT_N = 10.0
CONTACT_STABLE_SAMPLES = 6

PRESS_AXIS_SIGN = -1.0    # negative base_link z presses down
PULL_AXIS_SIGN = -1.0     # negative base_link x pulls toward the robot

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

FORCE_LP_ALPHA = 0.20
CONTROL_HZ = 50.0
REQUIRE_OPERATOR_CONFIRM = True


class SquashPull(Node):
    def __init__(self) -> None:
        super().__init__("squash_pull")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._twist_pub = self.create_publisher(TwistStamped, SERVO_TOPIC, 10)
        self._wrench_sub = self.create_subscription(WrenchStamped, WRENCH_TOPIC, self._on_wrench, 10)
        self._timer = None  # Created in main() after TF readiness gate

        self._state = "APPROACH"
        self._done = False
        self._integral = 0.0
        self._contact_count = 0
        self._pull_start_x = None
        self._filtered_force = 0.0
        self._have_force = False
        self._warned_no_servo = False
        self._last_tf_warn = 0.0
        self._awaiting_confirm = False
        self._last_nonfinite_warn = 0.0
        self._last_tf_warn_time = 0.0
        self._state_start_time = 0.0

        self._target = (PRE_SQUASH_X, PRE_SQUASH_Y, PRE_SQUASH_Z)
        self.get_logger().info(
            "Squash-pull armed. Hardcoded waypoint: (%.3f, %.3f, %.3f) in %s"
            % (*self._target, BASE_FRAME)
        )

    def _on_wrench(self, msg: WrenchStamped) -> None:
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        force = math.sqrt(fx * fx + fy * fy + fz * fz)
        if not math.isfinite(force):
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_nonfinite_warn > 1.0:
                self._last_nonfinite_warn = now
                self.get_logger().error("Received non-finite force sample (NaN/Inf). Holding last valid value.")
            return
        if not self._have_force:
            self._filtered_force = force
            self._have_force = True
        else:
            self._filtered_force = (1.0 - FORCE_LP_ALPHA) * self._filtered_force + FORCE_LP_ALPHA * force

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
        """Throttle WARNING messages (max once per ~5 seconds at 0.2 Hz)."""
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
        if self._awaiting_confirm:
            return False

        self._awaiting_confirm = True
        self._publish_zero()
        self.get_logger().warn(message)
        try:
            response = input("Press Enter to continue, or type 'q' to abort: ").strip().lower()
        except EOFError:
            response = "q"
        finally:
            self._awaiting_confirm = False

        if response == "q":
            self._done = True
            self.get_logger().warn("Operator aborted squash-pull sequence")
            return False
        return True

    def _tick(self) -> None:
        if self._done:
            self._publish_zero()
            return

        if self._twist_pub.get_subscription_count() == 0 and not self._warned_no_servo:
            self.get_logger().warn(
                "No subscriber on /servo_node/delta_twist_cmds yet. Start MoveIt Servo first."
            )
            self._warned_no_servo = True

        pose = self._lookup_pose()
        if pose is None:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_tf_warn > 5.0:
                self._last_tf_warn = now
                self.get_logger().warn(f"Waiting for TF {BASE_FRAME} -> {EE_LINK}")
            self._publish_zero()
            return

        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z

        if self._have_force and abs(self._filtered_force) > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(f"Hard force limit exceeded: {self._filtered_force:.2f} N — retracting")
            self._transition("RETRACT")

        if self._have_force and not math.isfinite(self._filtered_force):
            self.get_logger().error("Filtered force became non-finite. Aborting sequence for safety.")
            self._publish_zero()
            self._done = True
            return

        if self._state == "APPROACH":
            dx = PRE_SQUASH_X - x
            dy = PRE_SQUASH_Y - y
            dz = PRE_SQUASH_Z - z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= APPROACH_TOL:
                if not self._operator_confirm(
                    "Reached pre-squash waypoint. Confirm clear contact conditions before descending."
                ):
                    return
                self._transition("SQUASH")
                self._contact_count = 0
                self._publish_zero()
                return

            scale = min(1.0, APPROACH_MAX_SPEED / max(dist, 1e-6))
            self._publish_twist(dx * scale, dy * scale, dz * scale)
            return

        if self._state == "SQUASH":
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._state_start_time > SQUASH_TIMEOUT_SEC:
                self.get_logger().error(f"SQUASH timed out after {SQUASH_TIMEOUT_SEC:.0f}s — no contact detected. Retracting.")
                self._transition("RETRACT")
                return
            self._publish_twist(0.0, 0.0, PRESS_AXIS_SIGN * DESCEND_SPEED)
            if self._have_force and self._filtered_force >= FORCE_REF_N:
                self._contact_count += 1
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._pull_start_x = x
                    self._integral = 0.0
                    self._transition("PULL")
            else:
                self._contact_count = 0
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

            pull_cmd = PULL_AXIS_SIGN * PULL_SPEED
            normal_z = PRESS_AXIS_SIGN * normal_cmd
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
        node._timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())