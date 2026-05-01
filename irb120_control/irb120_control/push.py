#!/usr/bin/env python3
"""Straight-ahead Cartesian push controller for the IRB120.

All motion is planned with MoveIt's ComputeCartesianPath, which enforces a
straight-line EE path in Cartesian space — no Z drift.

Sequence:
  1. MoveIt approach to PRE_PUSH pose.
  2. Operator confirmation.
  3. Cartesian push: straight +X for PUSH_DISTANCE at PUSH_MAX_CARTESIAN_SPEED.
  4. MoveIt return to PRE_PUSH pose.

F/T data collected during step 3 is saved to a .npz file.
"""

import sys

import rclpy
from geometry_msgs.msg import Pose, WrenchStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import GetCartesianPath  # client created here, passed to cartesian_move
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.controllers.cartesian_move import plan_and_execute_cartesian
from irb120_control.controllers.moveit_single_shot import plan_and_execute_pose_goal
from irb120_control.util.runtime_log_dir import save_ft_pose_log, set_recorder_output_dir


BASE_FRAME = "base_link"
EE_LINK    = "finger_ball_center"
GROUP_NAME = "manipulator"

PRE_PUSH_X  = 0.488
PRE_PUSH_Y  = 0.00
PRE_PUSH_Z  = 0.141
PRE_PUSH_QX = 0.0
PRE_PUSH_QY = 0.0
PRE_PUSH_QZ = 0.0
PRE_PUSH_QW = 1.0

PUSH_DISTANCE            = 0.080   # m
PUSH_VELOCITY_SCALE      = 0.01    # fraction of joint limits (~5 mm/s at this pose)
PUSH_MAX_CARTESIAN_SPEED = 0.010   # m/s hard ceiling — secondary safety cap
CARTESIAN_MAX_STEP       = 0.001   # m — IK resolution along the path
CARTESIAN_JUMP_THRESH    = 0.0     # disabled

APPROACH_VELOCITY_SCALE  = 0.1     # speed for MoveIt approach and return moves
RETURN_VELOCITY_SCALE    = 0.05    # slower return after push

REQUIRE_OPERATOR_CONFIRM = True
DEBUG = True


class Push(Node):
    def __init__(self) -> None:
        super().__init__("push")
        self._tf_buffer         = Buffer()
        self._tf_listener       = TransformListener(self._tf_buffer, self)
        self._wrench_sub        = self.create_subscription(
            WrenchStamped, "/netft_data_transformed", self._on_wrench, 10
        )
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._cartesian_client  = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._execute_client    = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        self._wrench            = (0.0,) * 6
        self._have_force        = False
        self._ft_log: list[list[float]] = []
        self._pose_log: list[list[float]] = []
        self._log_start_s: float | None = None
        self._recording_ft      = False
        self._push_started      = False
        self._debug_timer       = None
        self._pose_log_timer    = self.create_timer(0.01, self._log_pose_cb)

    # ------------------------------------------------------------------ logging

    def _log_pose_cb(self) -> None:
        if not self._recording_ft:
            return
        pose = self._lookup_ee_pose()
        if pose is None:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._log_start_s is None:
            self._log_start_s = now_s
        self._pose_log.append([
            now_s - self._log_start_s,
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        ])

    # ------------------------------------------------------------------ wrench

    def _on_wrench(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        t = msg.wrench.torque
        self._wrench = (f.x, f.y, f.z, t.x, t.y, t.z)
        self._have_force = True
        if self._recording_ft:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self._log_start_s is None:
                self._log_start_s = now_s
            self._ft_log.append([now_s - self._log_start_s, f.x, f.y, f.z, t.x, t.y, t.z])

    # ------------------------------------------------------------------ debug

    def _debug_print(self) -> None:
        pose = self._lookup_ee_pose()
        if pose is None:
            return
        fx, fy, fz = self._wrench[0], self._wrench[1], self._wrench[2]
        print(
            f"[EE] x={pose.position.x:.4f}  y={pose.position.y:.4f}  z={pose.position.z:.4f}  "
            f"F=[{fx:.2f},{fy:.2f},{fz:.2f}]N",
            flush=True,
        )

    def _start_debug_timer(self) -> None:
        if DEBUG:
            self._debug_timer = self.create_timer(1.0, self._debug_print)

    def _stop_debug_timer(self) -> None:
        if self._debug_timer is not None:
            self._debug_timer.cancel()
            self._debug_timer = None

    # ------------------------------------------------------------------ helpers

    def _lookup_ee_pose(self) -> Pose | None:
        try:
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, EE_LINK, rclpy.time.Time())
        except TransformException:
            return None
        p = Pose()
        p.position.x    = tf.transform.translation.x
        p.position.y    = tf.transform.translation.y
        p.position.z    = tf.transform.translation.z
        p.orientation   = tf.transform.rotation
        return p

    def _operator_confirm(self, message: str) -> bool:
        if not REQUIRE_OPERATOR_CONFIRM:
            return True
        self.get_logger().warn(message)
        try:
            response = input("Press Enter to continue, or type 'q' to abort: ").strip().lower()
        except EOFError:
            response = "q"
        if response == "q":
            self.get_logger().warn("Operator aborted push sequence")
            return False
        return True

    # ------------------------------------------------------------------ MoveIt helpers

    def move_to_pre_push(self, velocity_scale: float = APPROACH_VELOCITY_SCALE) -> bool:
        return plan_and_execute_pose_goal(
            self,
            self._move_group_client,
            target_position=(PRE_PUSH_X, PRE_PUSH_Y, PRE_PUSH_Z),
            target_orientation=(PRE_PUSH_QX, PRE_PUSH_QY, PRE_PUSH_QZ, PRE_PUSH_QW),
            velocity_scale=velocity_scale,
        )

    def _cartesian_push(self) -> bool:
        start_pose = self._lookup_ee_pose()
        if start_pose is None:
            self.get_logger().error("Cannot look up EE pose — aborting push")
            return False

        target = Pose()
        target.position.x  = start_pose.position.x + PUSH_DISTANCE
        target.position.y  = start_pose.position.y
        target.position.z  = start_pose.position.z
        target.orientation = start_pose.orientation

        self.get_logger().info(
            f"Cartesian push: x {start_pose.position.x:.4f} -> {target.position.x:.4f}  "
            f"z stays at {start_pose.position.z:.4f}"
        )

        self._start_debug_timer()
        ok = plan_and_execute_cartesian(
            self,
            self._cartesian_client,
            self._execute_client,
            waypoints=[target],
            velocity_scale=PUSH_VELOCITY_SCALE,
            max_cartesian_speed=PUSH_MAX_CARTESIAN_SPEED,
            max_step=CARTESIAN_MAX_STEP,
            jump_threshold=CARTESIAN_JUMP_THRESH,
            execute_timeout_sec=PUSH_DISTANCE / PUSH_MAX_CARTESIAN_SPEED + 30.0,
        )
        self._stop_debug_timer()
        return ok



def main(args=None) -> int:
    rclpy.init(args=args)
    node = Push()
    recorder_client = node.create_client(SetBool, "/camera_hull_recorder/set_recording")
    try:
        set_recorder_output_dir(node, "push")
        if recorder_client.wait_for_service(timeout_sec=5.0):
            future = recorder_client.call_async(SetBool.Request(data=True))
            rclpy.spin_until_future_complete(node, future)
            result = future.result()
            if result is None or not result.success:
                node.get_logger().error(
                    f"Start-recording failed: {result.message if result else 'no response'} — aborting"
                )
                return 1
            node.get_logger().info("Recording started")
        else:
            node.get_logger().error("Recorder service not available after 5 s — aborting")
            return 1

        # Phase 1: approach
        if not node.move_to_pre_push():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        if not node._operator_confirm(
            "At pre-push pose. Confirm object is in position before pushing."
        ):
            return 0

        # Phase 2: Cartesian push (F/T recorded during execution)
        node._push_started = True
        node._recording_ft = True
        node.get_logger().info(
            f"Pushing {PUSH_DISTANCE*1000:.0f}mm in +X  "
            f"velocity_scale={PUSH_VELOCITY_SCALE}  "
            f"cartesian_ceiling={PUSH_MAX_CARTESIAN_SPEED*1000:.1f}mm/s"
        )
        push_ok = node._cartesian_push()
        node._recording_ft = False

        if not push_ok:
            node.get_logger().error("Push failed.")

        # Phase 3: return to pre-push pose
        node.get_logger().info("Returning to pre-push pose...")
        if not node.move_to_pre_push(velocity_scale=RETURN_VELOCITY_SCALE):
            node.get_logger().error("Return to pre-push pose failed.")

    except KeyboardInterrupt:
        pass
    finally:
        if recorder_client.wait_for_service(timeout_sec=2.0):
            if rclpy.ok():
                future = recorder_client.call_async(SetBool.Request(data=False))
                rclpy.spin_until_future_complete(node, future)
                result = future.result()
                if result is None or not result.success:
                    node.get_logger().error(
                        f"Stop-recording failed: {result.message if result else 'no response'}"
                    )
                else:
                    node.get_logger().info("Recording stopped")
            else:
                node.get_logger().warn("rclpy already shut down — stop-recording call skipped")
        if node._push_started:
            save_ft_pose_log(node._ft_log, node._pose_log, subdir="push", prefix="push_ft_pose")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
