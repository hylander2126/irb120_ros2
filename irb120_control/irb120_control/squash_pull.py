#!/usr/bin/env python3
"""Simple squash-and-pull Cartesian force controller for the IRB120.

Approach: MoveIt plans to the pre-squash pose (position + orientation).
Squash/Pull/Retract: MoveIt Servo twist commands on /servo_node/delta_twist_cmds.
"""

import sys

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from abb_robot_msgs.srv import TriggerWithResultCode
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.controllers.force_controller import PIForceController
from irb120_control.controllers.moveit_single_shot import plan_and_execute_pose_goal
from irb120_control.util.runtime_log_dir import set_recorder_output_dir, save_ft_log, save_ft_pose_log


BASE_FRAME = "base_link"
EE_LINK = "finger_ball_center"

# Pre-squash pose — position and orientation the EE must reach before descending.
# Tune quaternion to match your desired squash orientation.
PRE_SQUASH_X = 0.545
PRE_SQUASH_Y = 0.00
PRE_SQUASH_Z = 0.308 # 0.214 for heart
PRE_SQUASH_QX = 0.0
PRE_SQUASH_QY = 0.0
PRE_SQUASH_QZ = 0.0
PRE_SQUASH_QW = 1.0

FORCE_REF_N = 4.0 # 2.0 for heart
FORCE_HARD_LIMIT_N = 10.0
CONTACT_STABLE_SAMPLES = 1 # how many consecutive ref_n samples needed to stop squashing?

DESCEND_SPEED = 0.005    # m/s
PULL_SPEED = 0.008       # m/s
RETRACT_SPEED = 0.010    # m/s
RETRACT_CLEARANCE = 0.020
PULL_DISTANCE = 0.06 # 0.030 # m

SQUASH_TIMEOUT_SEC = 30.0 # 12.0
PULL_TIMEOUT_SEC = 30.0 # 16.0
UNPULL_TIMEOUT_SEC = 30.0 # 16.0
RETRACT_TIMEOUT_SEC = 30.0 #12.0
LULL_WAIT_SEC = 1.0

KP_FORCE = 0.0015
KI_FORCE = 0.00015
MAX_NORMAL_SPEED = 0.010
UNPULL_FORCE_BIAS_N = 0.5   # extra normal force added during UNPULL to counter object toppling

CONTROL_HZ = 100.0
REQUIRE_OPERATOR_CONFIRM = True


class SquashPull(Node):
    def __init__(self) -> None:
        super().__init__("squash_pull")
        self._tf_buffer         = Buffer()
        self._tf_listener       = TransformListener(self._tf_buffer, self)
        self._twist_pub         = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self._wrench_sub        = self.create_subscription(WrenchStamped, "/netft_data_transformed", self._on_wrench, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._timer             = None  # Created in main() after approach completes

        self._force_ctrl = PIForceController(
            kp                  =KP_FORCE,
            ki                  =KI_FORCE,
            force_ref_n         =FORCE_REF_N,
            max_normal_speed    =MAX_NORMAL_SPEED,
            control_hz          =CONTROL_HZ,
        )
        self._state = "SQUASH"
        self._done = False
        self._contact_count = 0
        self._pull_start_x = None
        self._pull_end_x = None  # recorded at PULL→UNPULL transition; UNPULL returns here
        self._force_z = 0.0
        self._have_force = False
        self._last_tf_warn_time = 0.0
        self._state_start_time = 0.0
        self._contact_felt = False
        self._lull_next: str = "PULL"  # state to enter after the LULL hold expires
        # Logging buffers
        self._ft_log: list = []    # rows: [time_s, fx, fy, fz, tx, ty, tz]
        self._pose_log: list = []  # rows: [time_s, x, y, z, qx, qy, qz, qw]
        self._last_force_log_time = 0.0
        self._saved_force_ref = FORCE_REF_N

    def _restart_egm(self) -> None:
        """Call start_egm_joint to ensure EGM hasn't gone dark."""
        self.get_logger().info("Ensuring EGM is active before entering servo phase...")
        client = self.create_client(TriggerWithResultCode, "/rws_client/start_egm_joint")
        if client.wait_for_service(timeout_sec=2.0):
            future = client.call_async(TriggerWithResultCode.Request())
            # In a timer callback we wouldn't spin_until_future_complete, but we only call this from main()
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            res = future.result()
            if res:
                self.get_logger().info(f"start_egm_joint -> code={res.result_code}, msg='{res.message}'")
            else:
                self.get_logger().warn("start_egm_joint returned no result")
        else:
            self.get_logger().warn("start_egm_joint service not available, skipping.")


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
        # Append F/T sample (timestamped)
        t = self._now_s()
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z
        self._ft_log.append([t, fx, fy, fz, tx, ty, tz])

        # Light-weight console feedback throttled to 5 Hz
        if self._force_z > 0.25:
            now = t
            if now - self._last_force_log_time > 0.2:
                self._last_force_log_time = now
                self.get_logger().info(f"z_force: {self._force_z:.2f} N  state: {self._state}")

    ## This TF buffer lives here for now, but if we use it in other places, we should create a new helper class/node
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

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _check_timeout(self, timeout_sec: float, label: str) -> bool:
        """Return True and trigger RETRACT if the current state has exceeded timeout_sec."""
        if self._now_s() - self._state_start_time > timeout_sec:
            self.get_logger().error(f"{label} timed out after {timeout_sec:.0f}s. Retracting.")
            self._transition("RETRACT")
            return True
        return False

    def _warn_throttled(self, message: str, throttle_hz: float = 0.2) -> None:
        now = self._now_s()
        min_interval = 1.0 / throttle_hz
        if now - self._last_tf_warn_time > min_interval:
            self._last_tf_warn_time = now
            self.get_logger().warn(message)

    def _publish_twist(self, vx: float, vy: float, vz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        # Safety: stop sequence if servo dies
        if self.count_subscribers("/servo_node/delta_twist_cmds") == 0:
            self.get_logger().error("No servo subscribers — stopping motion")
            self._done = True
            return
        self._twist_pub.publish(msg)

    def _publish_zero(self) -> None:
        self._publish_twist(0.0, 0.0, 0.0)

    def _wait_for_servo_ready(self, timeout_sec: float = 5.0) -> bool:
        end_time = self._now_s() + timeout_sec
        while rclpy.ok() and self._now_s() < end_time:
            if self.count_subscribers("/servo_node/delta_twist_cmds") > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _transition(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f"{self._state} -> {state}")
            self._state = state
            self._state_start_time = self._now_s()

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

        # Log EE pose sample
        t = self._now_s()
        px = pose.pose.position.x
        py = pose.pose.position.y
        pz = pose.pose.position.z
        q = pose.pose.orientation
        self._pose_log.append([t, px, py, pz, q.x, q.y, q.z, q.w])

        x = pose.pose.position.x
        z = pose.pose.position.z

        if self._have_force and self._force_z > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(f"Hard z force limit exceeded: {self._force_z:.2f} N — retracting")
            self._transition("RETRACT")

        if self._state == "SQUASH":
            if self._check_timeout(SQUASH_TIMEOUT_SEC, "SQUASH — no contact"): return
            if self._have_force and self._force_z > 0.25 and not self._contact_felt:
                self._contact_felt = True
                self.get_logger().info(f"Contact felt!")
            self._publish_twist(0.0, 0.0, -DESCEND_SPEED)
            if self._have_force and self._force_z >= FORCE_REF_N:
                self._contact_count += 1
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._transition("LULL")
            else:
                self._contact_count = 0
            return

        if self._state == "LULL":
            # Continue maintaining vertical Z force, but don't move in X/Y.
            normal_z = -self._force_ctrl.update(self._force_z)
            self._publish_twist(0.0, 0.0, normal_z)

            if self._now_s() - self._state_start_time < LULL_WAIT_SEC:
                return
            if self._lull_next == "PULL":
                self._force_ctrl.reset()
                self._force_ctrl.set_reference(FORCE_REF_N)
                self._saved_force_ref = FORCE_REF_N
                self._pull_start_x = x
            elif self._lull_next == "UNPULL":
                # Restore the pre-PULL squash setpoint so the object remains loaded
                # during the UNPULL return stroke.
                restore_ref = self._saved_force_ref + UNPULL_FORCE_BIAS_N
                self._force_ctrl.set_reference(restore_ref)
                self.get_logger().info(
                    f"UNPULL setpoint restored: {restore_ref:.2f} N "
                    f"(saved={self._saved_force_ref:.2f} N + bias={UNPULL_FORCE_BIAS_N:.2f} N)"
                )
            self._transition(self._lull_next)
            return

        if self._state == "PULL":
            if self._check_timeout(PULL_TIMEOUT_SEC, "PULL"): return
            normal_z = -self._force_ctrl.update(self._force_z)
            self._publish_twist(-PULL_SPEED, 0.0, normal_z)

            if self._pull_start_x is not None and abs(x - self._pull_start_x) >= PULL_DISTANCE:
                self._pull_end_x = x
                self._lull_next = "UNPULL"
                self._transition("LULL")
            return

        if self._state == "UNPULL":
            # Reverse the pull: move back toward _pull_start_x at the same speed,
            # running the same PI force loop to keep FORCE_REF_N against the surface.
            if self._check_timeout(UNPULL_TIMEOUT_SEC, "UNPULL"): return
            normal_z = -self._force_ctrl.update(self._force_z)
            self._publish_twist(+PULL_SPEED, 0.0, normal_z)

            # Done when we've returned to the x position where PULL began
            if self._pull_start_x is not None and x >= self._pull_start_x:
                self._force_ctrl.set_reference(FORCE_REF_N)  # restore default for future phases
                self._transition("RETRACT")
            return

        if self._state == "RETRACT":
            if self._now_s() - self._state_start_time > RETRACT_TIMEOUT_SEC:
                self._publish_zero()
                self._done = True
                self.get_logger().error(f"RETRACT timed out after {RETRACT_TIMEOUT_SEC:.0f}s. Aborting sequence.")
                return
            self._publish_twist(0.0, 0.0, RETRACT_SPEED)
            if z >= PRE_SQUASH_Z + RETRACT_CLEARANCE:
                self._publish_zero()
                self._done = True
                self.get_logger().info("Squash-pull sequence complete")
            return


def main(args=None) -> int:
    rclpy.init(args=args)
    node = SquashPull()
    recorder_client = node.create_client(SetBool, "/camera_hull_recorder/set_recording")
    try:
        set_recorder_output_dir(node, "squash_pull")
        if recorder_client.wait_for_service(timeout_sec=5.0):
            future = recorder_client.call_async(SetBool.Request(data=True))
            rclpy.spin_until_future_complete(node, future)
            result = future.result()
            if result is None or not result.success:
                node.get_logger().error(f"Start-recording failed: {result.message if result else 'no response'} — aborting")
                return 1
            node.get_logger().info("Recording started")
        else:
            node.get_logger().error("Recorder service not available after 5 s — aborting")
            return 1

        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error(
                "MoveIt Servo is not ready (/servo_node/delta_twist_cmds has no subscribers). "
                "Launch stack with start_servo:=true before running squash_pull."
            )
            return 1

        # Phase 1: MoveIt approach to pre-squash pose (blocking, with correct orientation)
        if not node.move_to_pre_squash():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        if not node._operator_confirm(
            "At pre-squash pose. Confirm clear contact conditions before descending."
        ):
            return 0

        # Phase 2: Servo-based squash/pull/retract
        # Ensure EGM didn't timeout during the operator confirmation phase
        node._restart_egm()

        # Timer drives the closed-loop state machine at CONTROL_HZ while spin_once
        # services subscriptions, TF updates, and timer callbacks.
        node._state_start_time = node._now_s()
        node._timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder_client.wait_for_service(timeout_sec=2.0):
            if rclpy.ok():
                future = recorder_client.call_async(SetBool.Request(data=False))
                rclpy.spin_until_future_complete(node, future)
                result = future.result()
                if result is None or not result.success:
                    node.get_logger().error(f"Stop-recording failed: {result.message if result else 'no response'}")
                else:
                    node.get_logger().info("Recording stopped")
            else:
                node.get_logger().warn("rclpy already shut down — stop-recording call skipped")
        # Save logs (F/T + EE pose) to runtime_logs/squash_pull
        try:
            save_ft_pose_log(node._ft_log, node._pose_log, "squash_pull", "squash_pull")
        except Exception as exc:
            node.get_logger().error(f"Failed to save F/T+pose log: {exc}")

        node._publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    # Propagate main() return code to the shell so launch/scripts can detect success/failure.
    sys.exit(main())
