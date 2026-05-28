#!/usr/bin/env python3
"""No-load arc exercise for debugging motion and F/T signals."""

import math
import sys

import rclpy
from geometry_msgs.msg import WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from irb120_control.controllers.moveit_single_shot import plan_and_execute_pose_goal
from irb120_control.controllers.servo_command_publisher import ServoCommandPublisher
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.motion_geometry import arc_angle_xz, arc_velocity_xz, clamp, quat_to_pitch
from irb120_control.util.runtime_log_dir import load_object_params, save_ft_pose_log, set_recorder_output_dir

BASE_FRAME = "world"
SERVO_FRAME = "base_link"
EE_LINK = "finger_ball_center"

STATE_IDS = {
    "SQUASH": 1,
    "LULL": 2,
    "ARC": 3,
    "UNARC": 4,
    "RETRACT": 5,
}

OBJECT = "box"
LOG_SUBDIR = f"{OBJECT}/arc_squash"

ARC_MAX_ANGLE_DEG = -20.0
ARC_CENTER_X_OFFSET = 0.005
ARC_TANGENTIAL_SPEED = 0.008
KP_ORIENT = 1.0
MAX_ORIENT_SPEED = 0.5

CONTROL_HZ = 100.0
SQUASH_HOLD_SEC = 2.5
LULL_WAIT_SEC = 1.0
ARC_TIMEOUT_SEC = 45.0
UNARC_TIMEOUT_SEC = 45.0
RETRACT_SPEED = 0.008
RETRACT_DURATION_SEC = 1.0


class ArcTest(Node):
    """Run the arc state machine in free space with no force-dependent exits."""

    def __init__(self) -> None:
        super().__init__("arc_test")
        params = load_object_params(OBJECT)
        ps = params["pre_squash"]
        self._pre_squash_pos = (ps["x"], ps["y"], ps["z"])
        self._pre_squash_ori = (ps["qx"], ps["qy"], ps["qz"], ps["qw"])

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._servo_cmd = ServoCommandPublisher(self, "/servo_node/delta_twist_cmds", SERVO_FRAME)
        self._wrench_sub = self.create_subscription(WrenchStamped, "/netft_data_transformed", self._on_wrench, 10)
        self._det_sub = self.create_subscription(Detection3DArray, "/object_detector/detections", self._on_detection, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._pause_servo_client = self.create_client(SetBool, "/servo_node/pause_servo")
        self._timer = None

        self._state = "SQUASH"
        self._done = False
        self._state_start_time = 0.0
        self._last_tf_warn_time = 0.0
        self._last_arc_log_time = 0.0

        self._force_x = 0.0
        self._force_y = 0.0
        self._force_z = 0.0

        self._arc_center_x: float | None = None
        self._arc_start_angle: float | None = None
        self._arc_end_angle = math.radians(ARC_MAX_ANGLE_DEG)

        self._ft_transformed_log: list = []
        self._pose_log: list = []
        self._obj_pose_log: list = []

    def move_to_pre_squash(self) -> bool:
        return plan_and_execute_pose_goal(
            self,
            self._move_group_client,
            target_position=self._pre_squash_pos,
            target_orientation=self._pre_squash_ori,
            velocity_scale=0.1,
            acceleration_scale=0.1,
        )

    def _set_servo_paused(self, paused: bool) -> None:
        if not self._pause_servo_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("pause_servo service not available")
            return
        future = self._pause_servo_client.call_async(SetBool.Request(data=paused))
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def pause_servo(self) -> None:
        self._set_servo_paused(True)

    def resume_servo(self) -> None:
        self._set_servo_paused(False)

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self._force_x = msg.wrench.force.x
        self._force_y = msg.wrench.force.y
        self._force_z = msg.wrench.force.z
        t = self._now_s()
        try:
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, "ft_link", rclpy.time.Time())
            tr = tf.transform.translation
            ro = tf.transform.rotation
            ft_px, ft_py, ft_pz = tr.x, tr.y, tr.z
            ft_qx, ft_qy, ft_qz, ft_qw = ro.x, ro.y, ro.z, ro.w
        except TransformException:
            ft_px = ft_py = ft_pz = float("nan")
            ft_qx = ft_qy = ft_qz = ft_qw = float("nan")
        self._ft_transformed_log.append([
            t,
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ft_px, ft_py, ft_pz, ft_qx, ft_qy, ft_qz, ft_qw,
        ])

    def _on_detection(self, msg: Detection3DArray) -> None:
        if not msg.detections:
            return
        hyp = msg.detections[0].results[0] if msg.detections[0].results else None
        if hyp is None:
            return
        t = self._now_s()
        p = hyp.pose.pose.position
        q = hyp.pose.pose.orientation
        self._obj_pose_log.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w, quat_to_pitch(q.x, q.y, q.z, q.w)])

    def _lookup_pose(self) -> tuple[float, float, float, float, float, float, float] | None:
        try:
            transform = self._tf_buffer.lookup_transform(BASE_FRAME, EE_LINK, rclpy.time.Time())
        except TransformException as exc:
            self._warn_throttled(f"Waiting for TF {BASE_FRAME} -> {EE_LINK}: {exc}")
            return None
        tr = transform.transform.translation
        q = transform.transform.rotation
        return tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttled(self, message: str, throttle_hz: float = 0.2) -> None:
        now = self._now_s()
        if now - self._last_tf_warn_time > 1.0 / throttle_hz:
            self._last_tf_warn_time = now
            self.get_logger().warn(message)

    def _wait_for_servo_ready(self, timeout_sec: float = 5.0) -> bool:
        end_time = self._now_s() + timeout_sec
        while rclpy.ok() and self._now_s() < end_time:
            if self._servo_cmd.has_subscribers():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _transition(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f"{self._state} -> {state}")
            self._state = state
            self._state_start_time = self._now_s()

    def _init_arc(self, x_contact: float, y_contact: float, z_contact: float) -> None:
        self._arc_center_x = x_contact + ARC_CENTER_X_OFFSET
        self._arc_start_angle = arc_angle_xz(x_contact, z_contact, self._arc_center_x)
        self.get_logger().info(
            f"Arc test init: center=({self._arc_center_x:.4f}, {y_contact:.4f}, 0)  "
            f"r={z_contact:.4f} m  start={math.degrees(self._arc_start_angle):.1f} deg  "
            f"target={ARC_MAX_ANGLE_DEG:.1f} deg"
        )

    def _current_arc_angle(self, x: float, z: float) -> float:
        return arc_angle_xz(x, z, self._arc_center_x)

    def _publish_arc_step(
        self,
        x: float,
        z: float,
        pitch: float,
        tangential_speed: float,
    ) -> tuple[float, bool]:
        angle = self._current_arc_angle(x, z)
        vx, vz = arc_velocity_xz(angle, tangential_speed, 0.0)
        pitch_err = angle - pitch
        wy = clamp(KP_ORIENT * pitch_err, MAX_ORIENT_SPEED)
        ok = self._servo_cmd.publish_twist(vx, 0.0, vz, wy, self._state, self._force_z)
        return angle, ok

    def _tick(self) -> None:
        if self._done:
            self._servo_cmd.publish_zero(self._state, self._force_z)
            return

        pose_row = self._lookup_pose()
        if pose_row is None:
            self._servo_cmd.publish_zero(self._state, self._force_z)
            return

        t = self._now_s()
        px, py, pz, qx, qy, qz, qw = pose_row
        current_pitch = quat_to_pitch(qx, qy, qz, qw)
        arc_angle = self._current_arc_angle(px, pz) if self._arc_center_x is not None else float("nan")
        state_id = STATE_IDS.get(self._state, 0)
        self._pose_log.append([t, px, py, pz, qx, qy, qz, qw, arc_angle, current_pitch, state_id])

        elapsed = t - self._state_start_time

        if self._state == "SQUASH":
            self._servo_cmd.publish_zero(self._state, self._force_z)
            if elapsed >= SQUASH_HOLD_SEC:
                self._transition("LULL")
            return

        if self._state == "LULL":
            self._servo_cmd.publish_zero(self._state, self._force_z)
            if elapsed < LULL_WAIT_SEC:
                return
            if self._arc_center_x is None:
                self._init_arc(px, py, pz)
                self._transition("ARC")
            else:
                self._transition("UNARC")
            return

        if self._state == "ARC":
            if elapsed > ARC_TIMEOUT_SEC:
                self.get_logger().warn("ARC timeout in no-load test; moving to LULL")
                self._transition("LULL")
                return
            angle, ok = self._publish_arc_step(px, pz, current_pitch, ARC_TANGENTIAL_SPEED)
            if not ok:
                self._done = True
                return
            if t - self._last_arc_log_time > 0.2:
                self._last_arc_log_time = t
                self.get_logger().info(
                    f"arc_test: {math.degrees(angle):.1f} / {ARC_MAX_ANGLE_DEG:.1f} deg  "
                    f"pitch: {math.degrees(current_pitch):.1f} deg  "
                    f"ft=({self._force_x:.2f}, {self._force_y:.2f}, {self._force_z:.2f}) N"
                )
            if angle <= self._arc_end_angle:
                self._transition("LULL")
            return

        if self._state == "UNARC":
            if elapsed > UNARC_TIMEOUT_SEC:
                self.get_logger().warn("UNARC timeout in no-load test; retracting")
                self._transition("RETRACT")
                return
            angle, ok = self._publish_arc_step(px, pz, current_pitch, -ARC_TANGENTIAL_SPEED)
            if not ok:
                self._done = True
                return
            if self._arc_start_angle is not None and angle >= self._arc_start_angle - math.radians(1.0):
                self._transition("RETRACT")
            return

        if self._state == "RETRACT":
            if elapsed < RETRACT_DURATION_SEC:
                if not self._servo_cmd.publish_twist(0.0, 0.0, RETRACT_SPEED, state=self._state, force_z=self._force_z):
                    self._done = True
            else:
                self._servo_cmd.publish_zero(self._state, self._force_z)
                self._done = True


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ArcTest()
    recorder_client = node.create_client(SetBool, "/camera_hull_recorder/set_recording")
    try:
        if not tare_netft(node):
            return 1

        set_recorder_output_dir(node, LOG_SUBDIR)
        if recorder_client.wait_for_service(timeout_sec=5.0):
            future = recorder_client.call_async(SetBool.Request(data=True))
            rclpy.spin_until_future_complete(node, future)
            node.get_logger().info("Recording started")
        else:
            node.get_logger().warn("Recorder service not available; continuing no-load arc test without video")

        if not ensure_egm_active(node):
            return 1
        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error("MoveIt Servo is not ready")
            return 1
        if not node.move_to_pre_squash():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        node.get_logger().warn("Starting no-load arc test using box pre-squash pose; force checks are disabled.")
        node.resume_servo()
        node._state_start_time = node._now_s()
        node._timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)

        if rclpy.ok():
            node._servo_cmd.publish_zero(node._state, node._force_z)
            node.pause_servo()
            node.get_logger().info("Returning to box pre-squash pose via MoveIt...")
            node.move_to_pre_squash()
    except KeyboardInterrupt:
        pass
    finally:
        if recorder_client.wait_for_service(timeout_sec=2.0) and rclpy.ok():
            future = recorder_client.call_async(SetBool.Request(data=False))
            rclpy.spin_until_future_complete(node, future)
            node.get_logger().info("Recording stopped")
        try:
            save_ft_pose_log(node._ft_transformed_log, node._pose_log, LOG_SUBDIR, "arc_test", node._obj_pose_log)
        except Exception as exc:
            node.get_logger().error(f"Failed to save no-load arc test log: {exc}")
        node._servo_cmd.publish_zero(node._state, node._force_z)
        node._servo_cmd.close()
        deactivate_egm(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
