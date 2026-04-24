#!/usr/bin/env python3
"""Simple squash-and-pull Cartesian force controller for the IRB120.

Approach: MoveIt plans to the pre-squash pose (position + orientation).
Squash/Pull/Retract: MoveIt Servo twist commands on /servo_node/delta_twist_cmds.
"""

import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.moveit_single_shot import plan_and_execute_pose_goal
from irb120_control.servo_command_publisher import ServoCommandPublisher


BASE_FRAME = "base_link"
EE_LINK = "finger_ball_center"
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

FORCE_REF_N = 2.0
FORCE_HARD_LIMIT_N = 10.0
CONTACT_STABLE_SAMPLES = 1 # how many consecutive ref_n samples needed to stop squashing?

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

CONTROL_HZ = 100.0
REQUIRE_OPERATOR_CONFIRM = True


class SquashPull(Node):
    def __init__(self) -> None:
        super().__init__("squash_pull")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._servo = ServoCommandPublisher(self, SERVO_TOPIC, BASE_FRAME)
        self._wrench_sub = self.create_subscription(WrenchStamped, WRENCH_TOPIC, self._on_wrench, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._timer = None  # Created in main() after approach completes

        self._state = "SQUASH"
        self._done = False
        self._integral = 0.0
        self._contact_count = 0
        self._pull_start_x = None
        self._force_z = 0.0
        self._have_force = False
        self._last_tf_warn_time = 0.0
        self._state_start_time = 0.0
        self._contact_felt = False
        self._lull_prompted = False


    def move_to_pre_squash(self) -> bool:
        """Blocking MoveIt call to reach PRE_SQUASH pose. Returns True on success."""
        return plan_and_execute_pose_goal(
            self,
            self._move_group_client,
            target_position=(PRE_SQUASH_X, PRE_SQUASH_Y, PRE_SQUASH_Z),
            target_orientation=(PRE_SQUASH_QX, PRE_SQUASH_QY, PRE_SQUASH_QZ, PRE_SQUASH_QW),
        )

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # Non-finite samples are filtered upstream in netft_preprocessor.
        self._force_z = abs(msg.wrench.force.z)
        self._have_force = True

        if self._force_z > 0.25:
            print(f"\r  z_force: {self._force_z:6.2f} N  state: {self._state:<8}", end="", flush=True)

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
        self._servo.publish_twist(vx, vy, vz, state=self._state, force_z=self._force_z)

    def _publish_zero(self) -> None:
        self._servo.publish_zero(state=self._state, force_z=self._force_z)

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
        # Periodic control callback (scheduled by a ROS timer in main).
        # This advances the state machine and publishes servo commands.
        if self._done:
            self._publish_zero()
            return

        pose = self._lookup_pose()
        if pose is None:
            self._publish_zero()
            return

        x = pose.pose.position.x
        z = pose.pose.position.z

        if self._have_force and self._force_z > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(f"Hard z force limit exceeded: {self._force_z:.2f} N — retracting")
            self._transition("RETRACT")

        if self._state == "SQUASH":
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._state_start_time > SQUASH_TIMEOUT_SEC:
                self.get_logger().error(f"SQUASH timed out after {SQUASH_TIMEOUT_SEC:.0f}s — no contact. Retracting.")
                self._transition("RETRACT")
                return
            if self._have_force and self._force_z > 0.25 and not self._contact_felt:
                self._contact_felt = True
                self.get_logger().info("Contact felt — halving descend speed")
            descend = (DESCEND_SPEED * 0.5) if self._contact_felt else DESCEND_SPEED
            self._publish_twist(0.0, 0.0, -descend)
            if self._have_force and self._force_z >= FORCE_REF_N:
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
            error = FORCE_REF_N - self._force_z
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
        # Timer drives the closed-loop state machine at CONTROL_HZ while spin_once
        # services subscriptions, TF updates, and timer callbacks.
        node._state_start_time = node.get_clock().now().nanoseconds * 1e-9
        node._timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_zero()
        if hasattr(node, '_servo'):
            node._servo.close()
            print(f"\nSquash-pull log written to {node._servo.log_path}")
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    # Propagate main() return code to the shell so launch/scripts can detect success/failure.
    sys.exit(main())
