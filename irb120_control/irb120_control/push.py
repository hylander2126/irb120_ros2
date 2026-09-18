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
from datetime import datetime

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
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.runtime_log_dir import (
    load_object_params,
    module_constants,
    save_ft_pose_log,
    save_run_metadata,
    start_recording,
    stop_recording,
    VALID_OBJECTS,
)


BASE_FRAME = "base_link"
EE_LINK    = "finger_ball_center"
GROUP_NAME = "manipulator"

PUSH_DISTANCE            = 0.080   # m
PUSH_VELOCITY_SCALE      = 0.01    # fraction of joint limits (~5 mm/s at this pose)
PUSH_MAX_CARTESIAN_SPEED = 0.010   # m/s hard ceiling — secondary safety cap
CARTESIAN_MAX_STEP       = 0.001   # m — IK resolution along the path
CARTESIAN_JUMP_THRESH    = 0.0     # disabled

APPROACH_VELOCITY_SCALE  = 0.1     # speed for MoveIt approach and return moves
RETURN_VELOCITY_SCALE    = 0.05    # slower return after push

REQUIRE_OPERATOR_CONFIRM = True
DEBUG = True

# Session F/T tare, run automatically at the pre-push pose (stationary, unloaded,
# confirmed-but-not-yet-touching) right before each push — see _tare_ft_sensor().
TARE_WAIT_SEC = 3.0   # netft_preprocessor accumulates TARE_SAMPLES=200 asynchronously
                       # after the service call returns; this is a generous fixed wait
                       # rather than a completion signal (none is exposed).

PUSH_VIDEO_QUALITY = "h264"   # camera_hull_recorder's "h264" mode now encodes via a
                               # direct ffmpeg/libx264 CRF pipe (see _FfmpegH264Writer),
                               # not cv2.VideoWriter's own ineffective h264 backend —
                               # good quality at a small fraction of "lossless"'s size.


class Push(Node):
    def __init__(self) -> None:
        super().__init__("push")
        self.declare_parameter("object", "")
        obj = self.get_parameter("object").get_parameter_value().string_value
        if obj not in VALID_OBJECTS:
            raise ValueError(
                f"Required parameter 'object' must be one of {sorted(VALID_OBJECTS)}, "
                f"got: '{obj}'. Pass it with: --ros-args -p object:=box"
            )
        self._object = obj
        self._log_subdir = f"{obj}/push"
        params = load_object_params(obj)
        pp = params["pre_push"]
        self._pre_push_pos = (pp["x"], pp["y"], pp["z"])
        self._pre_push_ori = (pp["qx"], pp["qy"], pp["qz"], pp["qw"])
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
            try:
                tf = self._tf_buffer.lookup_transform(BASE_FRAME, "ft_link", rclpy.time.Time())
                tr = tf.transform.translation
                ro = tf.transform.rotation
                ft_px, ft_py, ft_pz = tr.x, tr.y, tr.z
                ft_qx, ft_qy, ft_qz, ft_qw = ro.x, ro.y, ro.z, ro.w
            except TransformException:
                ft_px = ft_py = ft_pz = 0.0
                ft_qx = ft_qy = ft_qz = 0.0
                ft_qw = 1.0
            self._ft_log.append([
                now_s - self._log_start_s,
                f.x, f.y, f.z, t.x, t.y, t.z,
                ft_px, ft_py, ft_pz, ft_qx, ft_qy, ft_qz, ft_qw,
            ])

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
            target_position=self._pre_push_pos,
            target_orientation=self._pre_push_ori,
            velocity_scale=velocity_scale,
        )

    def _cartesian_move(self, x_distance: float, velocity_scale: float) -> bool:
        """Move the EE straight in X by x_distance (signed) at the given velocity scale."""
        start_pose = self._lookup_ee_pose()
        if start_pose is None:
            self.get_logger().error("Cannot look up EE pose — aborting move")
            return False

        target = Pose()
        target.position.x  = start_pose.position.x + x_distance
        target.position.y  = start_pose.position.y
        target.position.z  = start_pose.position.z
        target.orientation = start_pose.orientation

        direction = "+X" if x_distance >= 0 else "-X"
        self.get_logger().info(
            f"Cartesian {direction} {abs(x_distance)*1000:.1f}mm: "
            f"x {start_pose.position.x:.4f} -> {target.position.x:.4f}  "
            f"z stays at {start_pose.position.z:.4f}"
        )

        self._start_debug_timer()
        ok = plan_and_execute_cartesian(
            self,
            self._cartesian_client,
            self._execute_client,
            waypoints=[target],
            velocity_scale=velocity_scale,
            max_cartesian_speed=PUSH_MAX_CARTESIAN_SPEED,
            max_step=CARTESIAN_MAX_STEP,
            jump_threshold=CARTESIAN_JUMP_THRESH,
            execute_timeout_sec=abs(x_distance) / PUSH_MAX_CARTESIAN_SPEED + 30.0,
        )
        self._stop_debug_timer()
        return ok



def _tare_ft_sensor(node: Push, wait_sec: float = TARE_WAIT_SEC) -> None:
    """Zero the F/T sensor's session residual at the current pose. Call ONLY when
    the tool is stationary and definitely not in contact with anything — e.g. right
    after the operator confirms the object is positioned but before the push starts.

    netft_preprocessor's /set_tare service returns as soon as it *starts*
    accumulating TARE_SAMPLES; the actual capture finishes asynchronously as more
    wrench samples arrive. There's no completion signal exposed, so this just
    blocks for a generous fixed window afterward.
    """
    client = node.create_client(SetBool, "/netft_preprocessor/set_tare")
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().warn("netft_preprocessor tare service not available — skipping session tare")
        return
    future = client.call_async(SetBool.Request(data=True))
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
    result = future.result()
    if result is None or not result.success:
        node.get_logger().warn(
            f"Session tare request failed: {result.message if result else 'no response'} — proceeding untared"
        )
        return
    node.get_logger().info(f"{result.message} Waiting {wait_sec:.1f}s for capture to finish...")
    deadline = node.get_clock().now().nanoseconds * 1e-9 + wait_sec
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def main(args=None) -> int:
    rclpy.init(args=args)
    node = Push()
    try:
        if not ensure_egm_active(node):
            return 1

        # show_hull=False: push videos don't need the object-hull overlay —
        # see camera_hull_recorder's show_hull param (on by default via
        # bringup_stack.launch.py for arc recordings).
        if not start_recording(node, node._log_subdir, quality=PUSH_VIDEO_QUALITY, show_hull=False):
            node.get_logger().error("Recording failed to start on one or more cameras — aborting")
            return 1

        # Phase 1: approach
        if not node.move_to_pre_push():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        if not node._operator_confirm(
            "At pre-push pose. Confirm object is in position before pushing."
        ):
            return 0

        # Stationary, confirmed-in-position, not yet touching — zero the F/T
        # sensor's session residual right here before any contact happens.
        _tare_ft_sensor(node)

        # Phase 2: Cartesian push (F/T recorded during execution)
        node._push_started = True
        push_ok = None
        return_ok = None
        node._recording_ft = True
        node.get_logger().info(
            f"Pushing {PUSH_DISTANCE*1000:.0f}mm in +X  "
            f"velocity_scale={PUSH_VELOCITY_SCALE}  "
            f"cartesian_ceiling={PUSH_MAX_CARTESIAN_SPEED*1000:.1f}mm/s"
        )
        push_ok = node._cartesian_move(+PUSH_DISTANCE, PUSH_VELOCITY_SCALE)
        node._recording_ft = False

        if not push_ok:
            node.get_logger().error("Push failed.")

        # Phase 3: Cartesian return — retrace in -X to stay in the same IK
        # solution branch (joint-space return can rotate joint 6 by 360°).
        node.get_logger().info("Returning via Cartesian path...")
        return_ok = node._cartesian_move(-PUSH_DISTANCE, RETURN_VELOCITY_SCALE)
        if not return_ok:
            node.get_logger().error("Cartesian return failed — attempting joint-space fallback.")
            node.move_to_pre_push(velocity_scale=RETURN_VELOCITY_SCALE)

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording(node)
        deactivate_egm(node)
        if node._push_started:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_ft_pose_log(
                    node._ft_log, node._pose_log, subdir=node._log_subdir, prefix="push_ft_pose", timestamp=ts
                )
                save_run_metadata(
                    node._log_subdir,
                    "push_ft_pose",
                    ts,
                    {
                        **module_constants(globals()),
                        "object": node._object,
                        "push_ok": push_ok,
                        "return_ok": return_ok,
                    },
                )
            except Exception as exc:
                node.get_logger().error(f"Failed to save push F/T+pose log: {exc}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
