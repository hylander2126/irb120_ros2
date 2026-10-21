#!/usr/bin/env python3
"""Normal-directed Cartesian push controller for the IRB120.

All motion is planned with MoveIt's ComputeCartesianPath, which enforces a
straight-line EE path in Cartesian space — no Z drift.

Runs as the 'push' stage of an episode (util/episode.py):
  1. Perception snapshot (skipped if the episode's last stage already is one).
  2. MoveIt approach to the pre-push pose: the detected planar_push contact's
     finger-ball center, PUSH_STANDOFF back along the horizontal inward normal.
  3. Operator confirmation, F/T tare.
  4. Continuous horizontal push: the straight path is planned (collision/IK check only),
     then driven by MoveIt Servo at PUSH_SPEED, so it can stop on any tick. Contact must
     be felt within PUSH_CONTACT_MARGIN past the predicted contact, then the push goes
     PUSH_DISTANCE from where it was felt. Needs the stack launched with start_servo:=true.
  5. Cartesian return, then home, then a second snapshot of the pushed object.

F/T + pose from step 4, metadata and videos go into the stage folder.

Usage:
    ros2 run irb120_control push flashlight                      # new episode
    ros2 run irb120_control push flashlight --episode last
"""

import argparse
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, WrenchStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.srv import GetCartesianPath  # client created here, passed to cartesian_move
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.controllers.cartesian_move import plan_and_execute_cartesian
from irb120_control.controllers.servo_command_publisher import ServoCommandPublisher
from irb120_control.controllers.moveit_single_shot import plan_and_execute_joint_goal, plan_and_execute_pose_goal
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.episode import Episode, write_json_atomic
from irb120_control.util.object_obstacle import TABLE_Z, object_obstacle
from irb120_control.util.push_geometry import (
    min_push_pitch, push_approach, horizontal_push_direction, push_orientation, force_push_check)
from irb120_control.util.perception_snapshot import (
    PUSH_MIN_BALL_CENTER_Z, begin_motion_stage, latest_object_cloud, take_snapshot)
from irb120_control.util.ft_capture import FTCapture, stamp_s
from irb120_control.util.ft_frames import ESTIMATOR_POSE_FRAME
from irb120_control.util.runtime_log_dir import (
    module_constants,
    save_ft_pose_log,
    save_run_metadata,
    start_recording,
    stop_recording,
)


BASE_FRAME = "base_link"
EE_LINK    = "finger_ball_center"
GROUP_NAME = "manipulator"

PUSH_DISTANCE            = 0.050   # m, measured from where contact is felt
PUSH_STANDOFF            = 0.040   # m, pre-push ball center sits this far behind the detected contact (~14 mm clear of the object's collision box)
# Finger pitched tip-down about base +Y. Horizontal (0 deg), the 90 mm F/T stack hangs 45 mm
# below the shaft and hits the table, which forced ball centres >= 81 mm above it
# (perception_snapshot.MIN_BALL_CENTER_Z). The pitch is the smallest that keeps the stack and
# wrist PUSH_TABLE_CLEARANCE above the table with the ball at its lowest allowed centre
# (PUSH_MIN_BALL_CENTER_Z), from the URDF geometry (push_geometry.min_push_pitch): ~28.5 deg.
# A low ball makes toppling during the planar push less likely. The ball is a sphere, so the
# contact point on the face is unchanged; the shaft leans back-and-up, away from the object.
# The wrench is logged in tool0 axes with tool0's pose per sample (ft_q*); friction
# estimators rotate it into base_link (util/ft_frames.forces_in_base).
PUSH_TABLE_CLEARANCE     = 0.025   # m, stack/wrist above the table top at the lowest push
PUSH_PITCH_DEG           = float(np.degrees(min_push_pitch(PUSH_MIN_BALL_CENTER_Z - TABLE_Z,
                                                           PUSH_TABLE_CLEARANCE)))
PUSH_ORIENTATION         = (0.0, float(np.sin(np.radians(PUSH_PITCH_DEG) / 2)), 0.0,
                            float(np.cos(np.radians(PUSH_PITCH_DEG) / 2)))  # EE orientation (xyzw)
PUSH_VELOCITY_SCALE      = 0.01    # fraction of joint limits (~5 mm/s at this pose)
PUSH_MAX_CARTESIAN_SPEED = 0.010   # m/s hard ceiling — secondary safety cap
CARTESIAN_MAX_STEP       = 0.001   # m — IK resolution along the path
CARTESIAN_JUMP_THRESH    = 0.0     # disabled

APPROACH_VELOCITY_SCALE  = 0.05     # speed for MoveIt approach and return moves
RETURN_VELOCITY_SCALE    = 0.03    # slower return after push

PUSH_APPROACH_LIFT = 0.080  # m: finish free-space planning above the low pre-push pose
PUSH_DESCENT_SPEED = 0.005  # m/s: straight, collision-checked descent outside the object

HOME_JOINT_POSITIONS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # retreat here before the post-push snapshot

ARRIVAL_TOL_MM = 5.0       # measured EE vs pre-push target after the approach
ARRIVAL_TOL_DEG = 3.0

REQUIRE_OPERATOR_CONFIRM = True
DEBUG = True

# Session F/T tare, run automatically at the pre-push pose (stationary, unloaded,
# confirmed-but-not-yet-touching) right before each push — see _tare_ft_sensor().
TARE_WAIT_SEC = 3.0   # netft_preprocessor accumulates TARE_SAMPLES=200 asynchronously
                       # after the service call returns; this is a generous fixed wait
                       # rather than a completion signal (none is exposed).

# Validate contact during continuous motion, without changing the push direction.
# Stop if no reaction is confirmed by PUSH_CONTACT_MARGIN past the predicted contact.
PUSH_CONTACT_FORCE_N = 0.3
PUSH_FORCE_ANGLE_DEG = 45.0
PUSH_FORCE_LIMIT_N = 20.0
PUSH_CONTACT_MARGIN = 0.050  # generous: the cloud has put the face ~30 mm early (orange_robot)
PUSH_FORCE_WINDOW_SEC = 0.10
# Servo push: the planned path is only a gut check; Servo drives it and can stop any tick.
PUSH_SPEED = 0.009          # m/s along the push vector (what the old planned push actually ran)
PUSH_CONTROL_HZ = 50.0
PUSH_LINE_GAIN = 1.0        # 1/s: pulls the finger back onto the push line
# Servo on this rig moves at ~57% of the commanded speed (press-pull arcs too), so the push
# tracks a target advancing at PUSH_SPEED instead of trusting the command.
PUSH_TRACK_GAIN = 2.0       # 1/s on the lag behind that target
PUSH_MAX_CMD_SPEED = 0.025  # m/s cap on the command while catching up
PUSH_SETTLE_SEC = 0.5       # zero twists after the push before pausing Servo
SERVO_FRAME = "base_link"   # MoveIt Servo requires base_link as the twist command frame

PUSH_VIDEO_QUALITY = "h264"   # camera_hull_recorder's "h264" mode now encodes via a
                               # direct ffmpeg/libx264 CRF pipe (see _FfmpegH264Writer),
                               # not cv2.VideoWriter's own ineffective h264 backend —
                               # good quality at a small fraction of "lossless"'s size.


class Push(Node):
    def __init__(self, object_name: str) -> None:
        super().__init__("push")
        self._object = object_name
        self._pre_push_pos = None  # set from the detected planar_push contact in main()
        self._push_direction = np.array([1., 0., 0.])
        self._push_orientation = np.asarray(PUSH_ORIENTATION)
        self._force_samples = deque(maxlen=1000)
        self._force_check = {"status": "not_checked"}
        self._motion_uncertain = False
        self._approach_diag = {}   # MoveIt request/result of the approach, saved even on abort
        self._object_xyz = None    # detected object cloud (base_link), avoided on the approach
        self._tf_buffer         = Buffer()
        self._tf_listener       = TransformListener(self._tf_buffer, self)
        self._wrench_sub        = self.create_subscription(
            WrenchStamped, "/netft_data_transformed", self._on_wrench, 10
        )
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._cartesian_client  = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._execute_client    = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self._servo_cmd         = ServoCommandPublisher(self, "/servo_node/delta_twist_cmds", SERVO_FRAME)
        self._pause_servo_client = self.create_client(SetBool, "/servo_node/pause_servo")

        self._wrench            = (0.0,) * 6
        self._have_force        = False
        self._ft_log: list[list[float]] = []
        self._pose_log: list[list[float]] = []
        self._log_start_s: float | None = None
        self._recording_ft      = False
        self._pose_tf_stamp_s = float("nan")
        self._ft_capture = FTCapture(self, self._log_time_s, lambda: self._recording_ft)
        self._push_started      = False
        self._debug_timer       = None
        self._pose_log_timer    = self.create_timer(0.01, self._log_pose_cb)

    def _log_time_s(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self._log_start_s is None:
            self._log_start_s = now
        return now - self._log_start_s

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
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
            float("nan"), float("nan"), 0, self._pose_tf_stamp_s,
        ])

    # ------------------------------------------------------------------ wrench

    def _on_wrench(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        t = msg.wrench.torque
        self._wrench = (f.x, f.y, f.z, t.x, t.y, t.z)
        self._have_force = True
        # The historical message header says ft_link; processed forces use tool0 axes.
        try:
            from scipy.spatial.transform import Rotation
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, ESTIMATOR_POSE_FRAME, rclpy.time.Time())
            q = tf.transform.rotation
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp_s(msg.header.stamp)
            tf_age = self.get_clock().now().nanoseconds * 1e-9 - stamp_s(tf.header.stamp)
            if 0 <= age <= .5 and 0 <= tf_age <= .5:
                base_force = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply([f.x, f.y, f.z])
                if np.isfinite(base_force).all():
                    self._force_samples.append((time.monotonic(), base_force))
        except (TransformException, ValueError):
            pass
        if self._recording_ft:
            now_s = self.get_clock().now().nanoseconds * 1e-9
            if self._log_start_s is None:
                self._log_start_s = now_s
            try:
                tf = self._tf_buffer.lookup_transform(BASE_FRAME, ESTIMATOR_POSE_FRAME, rclpy.time.Time())
                tr = tf.transform.translation
                ro = tf.transform.rotation
                ft_px, ft_py, ft_pz = tr.x, tr.y, tr.z
                ft_qx, ft_qy, ft_qz, ft_qw = ro.x, ro.y, ro.z, ro.w
                tf_stamp = stamp_s(tf.header.stamp)
            except TransformException:
                ft_px = ft_py = ft_pz = float("nan")
                ft_qx = ft_qy = ft_qz = ft_qw = float("nan")
                tf_stamp = float("nan")
            self._ft_log.append([
                now_s - self._log_start_s,
                f.x, f.y, f.z, t.x, t.y, t.z,
                ft_px, ft_py, ft_pz, ft_qx, ft_qy, ft_qz, ft_qw,
                stamp_s(msg.header.stamp), tf_stamp,
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
            self._pose_tf_stamp_s = stamp_s(tf.header.stamp)
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
        """Plan to a raised pre-pose, then descend slowly outside the object box."""
        high = np.asarray(self._pre_push_pos) + [0., 0., PUSH_APPROACH_LIFT]
        try:
            with object_obstacle(self, self._object_xyz):
                ok = plan_and_execute_pose_goal(
                    self, self._move_group_client, base_frame=BASE_FRAME,
                    target_position=high, target_orientation=self._push_orientation,
                    velocity_scale=velocity_scale, diagnostics=self._approach_diag)
                if not ok or not self._arrived(high):
                    return False
                if not self._cartesian_move(0., velocity_scale,
                        target_position=self._pre_push_pos, max_speed=PUSH_DESCENT_SPEED):
                    return False
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return False
        return self._arrived()

    def _arrived(self, target_position=None) -> bool:
        """MoveIt can report a goal reached without the arm getting there (2026-10-16: a
        'successful' approach that never moved). Check the measured EE pose itself."""
        for _ in range(20):  # let TF catch up with the last joint states
            rclpy.spin_once(self, timeout_sec=0.05)
        pose = self._lookup_ee_pose()
        if pose is None:
            self.get_logger().error("Cannot look up EE pose after the approach")
            return False
        target_position = self._pre_push_pos if target_position is None else target_position
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        err_mm = 1000 * float(np.linalg.norm(np.array([pose.position.x, pose.position.y, pose.position.z])
                                             - np.asarray(target_position)))
        err_deg = float(np.degrees(2 * np.arccos(np.clip(abs(q @ self._push_orientation), 0., 1.))))
        self._approach_diag["arrival"] = dict(position_error_mm=err_mm, orientation_error_deg=err_deg)
        if err_mm > ARRIVAL_TOL_MM or err_deg > ARRIVAL_TOL_DEG:
            self.get_logger().error(
                f"Approach reported success but the EE is {err_mm:.1f} mm / {err_deg:.1f} deg from the "
                f"pre-push pose (limits {ARRIVAL_TOL_MM} mm / {ARRIVAL_TOL_DEG} deg). Not pushing. "
                "MoveIt request/result saved as push_approach_diagnostics.json.")
            return False
        return True

    def _set_servo_paused(self, paused: bool) -> bool:
        if not self._pause_servo_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("pause_servo service not available (launch with start_servo:=true)")
            return False
        future = self._pause_servo_client.call_async(SetBool.Request(data=paused))
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return bool(future.done() and future.result() is not None and future.result().success)

    def _servo_push(self, ball, travel) -> bool:
        """Drive the push line with MoveIt Servo until _check_moving_contact says stop, or
        the finger reaches the planned end (`travel` from the pre-push pose). Stopping is a
        zero twist on the next tick, not a trajectory cancel. True if it ran to a stop
        cleanly; the caller reads _force_check for why."""
        start, direction = np.asarray(self._pre_push_pos, dtype=float), self._push_direction
        deadline = time.monotonic() + travel / PUSH_SPEED + 10.
        if not self._set_servo_paused(False):
            self._force_check = dict(status="rejected", reason="could not resume MoveIt Servo")
            return False
        ok = True
        try:
            next_tick = t0 = time.monotonic()
            while rclpy.ok():
                next_tick += 1. / PUSH_CONTROL_HZ
                while time.monotonic() < next_tick:
                    rclpy.spin_once(self, timeout_sec=.002)
                if not self._check_moving_contact(ball):
                    break
                pose = self._lookup_ee_pose()
                if pose is None:
                    self._force_check = dict(status="rejected", reason="missing pose during servo push")
                    break
                offset = np.array([pose.position.x, pose.position.y, pose.position.z]) - start
                along = float(offset @ direction)
                if along >= travel:
                    break  # end of the planned (gut-checked) path
                if time.monotonic() > deadline:
                    self._force_check = dict(self._force_check, status="rejected", reason="servo push timed out")
                    break
                lag = PUSH_SPEED * (time.monotonic() - t0) - along
                speed = min(PUSH_MAX_CMD_SPEED, max(0., PUSH_SPEED + PUSH_TRACK_GAIN * lag))
                v = speed * direction - PUSH_LINE_GAIN * (offset - along * direction)
                if not self._servo_cmd.publish_twist(*v, state="PUSH"):
                    self._force_check = dict(status="rejected", reason="servo command not accepted")
                    ok = False
                    break
        finally:
            settle = time.monotonic() + PUSH_SETTLE_SEC
            while time.monotonic() < settle:
                self._servo_cmd.publish_zero("PUSH")
                rclpy.spin_once(self, timeout_sec=.02)
            if not self._set_servo_paused(True):
                self.get_logger().error("Servo pause unconfirmed; not commanding further motion")
                self._motion_uncertain = True
        return ok

    def _cartesian_move(self, distance: float, velocity_scale: float, *, target_position=None,
                        max_speed=PUSH_MAX_CARTESIAN_SPEED, execution_check=None, plan_only=False) -> bool:
        """Move along the fixed push vector, or retrace to an explicit position."""
        start_pose = self._lookup_ee_pose()
        if start_pose is None:
            self.get_logger().error("Cannot look up EE pose — aborting move")
            return False

        start = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z])
        end = start + distance*self._push_direction if target_position is None else np.asarray(target_position)
        target = Pose()
        target.position.x, target.position.y, target.position.z = end.tolist()
        target.orientation = start_pose.orientation
        length = float(np.linalg.norm(end - start))
        self.get_logger().info(f"Cartesian push-line move {length*1000:.1f}mm to {end.tolist()}")

        self._start_debug_timer()
        try:
            return plan_and_execute_cartesian(
                self, self._cartesian_client, self._execute_client,
                base_frame=BASE_FRAME, waypoints=[target], velocity_scale=velocity_scale,
                max_cartesian_speed=max_speed, max_step=CARTESIAN_MAX_STEP,
                jump_threshold=CARTESIAN_JUMP_THRESH, avoid_collisions=True,
                execute_timeout_sec=length/max_speed + 30., execution_check=execution_check,
                plan_only=plan_only)
        except RuntimeError as exc:
            self._motion_uncertain = True
            self.get_logger().error(str(exc))
            return False
        finally:
            self._stop_debug_timer()


    def _mean_contact_force(self):
        """Check the unloaded baseline using fresh stationary samples."""
        started = time.monotonic()
        while rclpy.ok() and time.monotonic() - started < PUSH_FORCE_WINDOW_SEC:
            rclpy.spin_once(self, timeout_sec=.02)
        samples = [(t, f) for t, f in self._force_samples if t >= started]
        if len(samples) < 5 or time.monotonic() - samples[-1][0] > .1:
            self._force_check = dict(status="rejected", reason="missing or stale force/TF samples")
            return None
        return np.mean([f for _, f in samples], axis=0)

    def _check_moving_contact(self, ball):
        """Validate initial reaction from a rolling window while the finger moves."""
        if self._force_check.get("status") == "validated":
            # Never steer or keep changing the direction; only stop after PUSH_DISTANCE.
            pose = self._lookup_ee_pose()
            if pose is None:
                return True  # the trajectory end bounds the push anyway
            pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            pushed = float((pos - np.asarray(self._force_check["contact_pos"])) @ self._push_direction)
            self._force_check["pushed_m"] = pushed
            samples = [f for t, f in self._force_samples if time.monotonic() - t <= PUSH_FORCE_WINDOW_SEC]
            if samples and np.linalg.norm(np.mean(samples, axis=0)) > PUSH_FORCE_LIMIT_N:
                self._force_check.update(status="rejected", reason=f"force over {PUSH_FORCE_LIMIT_N:.0f} N during push")
                return False
            return pushed < PUSH_DISTANCE
        now = time.monotonic()
        samples = [(t, f) for t, f in self._force_samples if now-t <= PUSH_FORCE_WINDOW_SEC]
        if not samples or now-samples[-1][0] > .1:
            self._force_check = dict(status="rejected", reason="missing or stale force/TF during push")
            return False
        if len(samples) >= 5 and samples[-1][0]-samples[0][0] >= .06:
            force = np.mean([f for _, f in samples], axis=0)
            status, reason = force_push_check(force, self._push_direction,
                PUSH_CONTACT_FORCE_N, PUSH_FORCE_ANGLE_DEG, PUSH_FORCE_LIMIT_N)
            self._force_check = dict(status=status, reason=reason, force_base_n=force.tolist())
            if status == "rejected":
                return False
        pose = self._lookup_ee_pose()
        if pose is None:
            self._force_check = dict(status="rejected", reason="missing pose during contact check")
            return False
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        offset = float((pos-np.asarray(ball)) @ self._push_direction)
        if self._force_check.get("status") == "validated":
            # Where contact was felt: the push distance counts from here.
            self._force_check.update(contact_pos=pos.tolist(), contact_offset_m=offset, pushed_m=0.)
            self.get_logger().info(f"Push vector validated during motion: {self._force_check['reason']}; "
                                   f"contact {offset * 1000:+.1f} mm from predicted; direction stays fixed")
            return True
        if offset > PUSH_CONTACT_MARGIN:
            self._force_check = dict(status="rejected", probe_offset_m=offset,
                                     reason=f"no contact within {PUSH_CONTACT_MARGIN * 1000:.0f} mm of predicted contact")
            return False
        return True

    def _validated_push(self, ball, travel):
        """One continuous fixed-vector push, checking contact before allowing travel past it."""
        force = self._mean_contact_force()
        if force is None:
            return False
        if np.linalg.norm(force) >= PUSH_CONTACT_FORCE_N:
            self._force_check = dict(status="rejected", reason="force not zero at unloaded pre-push pose")
            return False
        self._force_check = dict(status="waiting", reason="approaching contact")
        endpoint = np.asarray(self._pre_push_pos) + travel*self._push_direction
        # Gut check: the whole line must plan collision-free; Servo then drives it.
        if not self._cartesian_move(0., PUSH_VELOCITY_SCALE, target_position=endpoint, plan_only=True):
            self._force_check = dict(status="rejected", reason="push line does not plan")
            return False
        ok = self._servo_push(ball, travel)
        if self._force_check["status"] == "validated" and self._force_check.get("pushed_m", 0.) >= PUSH_DISTANCE:
            return True  # stopped on purpose, PUSH_DISTANCE past the felt contact
        if ok and self._force_check["status"] != "validated":
            self._force_check = dict(status="rejected", reason="motion ended without validated contact")
            return False
        return ok



def _tare_ft_sensor(node: Push, wait_sec: float = TARE_WAIT_SEC) -> bool:
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
        node.get_logger().warn("netft_preprocessor tare service not available — aborting push")
        return False
    future = client.call_async(SetBool.Request(data=True))
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)
    result = future.result()
    if result is None or not result.success:
        node.get_logger().warn(
            f"Session tare request failed: {result.message if result else 'no response'} — aborting push"
        )
        return False
    node.get_logger().info(f"{result.message} Waiting {wait_sec:.1f}s for capture to finish...")
    deadline = node.get_clock().now().nanoseconds * 1e-9 + wait_sec
    while rclpy.ok() and node.get_clock().now().nanoseconds * 1e-9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    return True


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    parser = argparse.ArgumentParser(description="Straight-ahead Cartesian push at the detected planar_push contact")
    parser.add_argument("object", help="Object/episode label; push uses the saved contact, not object-specific parameters")
    parser.add_argument("--episode", default="new", metavar="new|last|PATH",
                        help="Episode to record into (default: a new one). See util/episode.py.")
    parsed = parser.parse_args(raw_args[:ros_args_index])

    rclpy.init(args=raw_args[ros_args_index:])
    node = Push(parsed.object)
    episode = Episode.resolve(parsed.episode, parsed.object, node, allow_unconfigured_object=True)
    stage = None
    push_ok = return_ok = home_ok = None
    try:
        stage, contacts = begin_motion_stage(node, episode, "push")
        if stage is None:
            node.get_logger().error("No usable perception snapshot -- not moving.")
            return 1
        contact = contacts["planar_push"]
        if not contact["available"]:
            node.get_logger().error(f"No planar_push contact: {contact['reason']} -- not moving.")
            return 1
        try:
            # Derive from the normal even for older saved +X-only contacts.
            node._push_direction = horizontal_push_direction(contact.get("normal", []))
            node._push_orientation = push_orientation(node._push_direction, PUSH_PITCH_DEG)
        except ValueError as exc:
            node.get_logger().error(f"{exc}. Take a fresh snapshot -- not moving.")
            return 1
        ball = contact["ball_center"]
        if (not np.isfinite(ball).all() or ball[2] < PUSH_MIN_BALL_CENTER_Z):
            node.get_logger().error("Saved push is below the new table-clearance floor; take a fresh snapshot.")
            return 1
        node._object_xyz = latest_object_cloud(episode)
        node._pre_push_pos, push_travel = push_approach(
            ball, node._object_xyz, PUSH_STANDOFF, PUSH_CONTACT_MARGIN + PUSH_DISTANCE, node._push_direction)
        node.get_logger().info(f"planar_push contact ball center {ball}; pre-push {node._pre_push_pos}")

        if not ensure_egm_active(node):
            return 1
        if not node._servo_cmd.has_subscribers() or not node._set_servo_paused(True):
            node.get_logger().error("MoveIt Servo not running (launch with start_servo:=true) -- not moving.")
            return 1

        # show_hull=False: push videos don't need the object-hull overlay.
        if not start_recording(node, str(stage.path), quality=PUSH_VIDEO_QUALITY, show_hull=False):
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
        if not _tare_ft_sensor(node):
            node._force_check = dict(status="rejected", reason="session tare failed")
            return 1

        # Phase 2: Cartesian push (F/T recorded during execution)
        node._push_started = True
        node._recording_ft = True
        node.get_logger().info(
            f"Pushing {push_travel*1000:.0f}mm along {node._push_direction.tolist()}  "
            f"by MoveIt Servo at {PUSH_SPEED*1000:.1f} mm/s"
        )
        push_ok = node._validated_push(ball, push_travel)
        node._recording_ft = False

        if not push_ok:
            node.get_logger().error(f"Push failed: {node._force_check.get('reason', 'motion failed')}")

        if node._motion_uncertain:
            return 1  # EGM is stopped in finally; do not plan another motion.

        # Phase 3: Cartesian return — retrace to the measured start to stay in the same IK
        # solution branch (joint-space return can rotate joint 6 by 360°).
        node.get_logger().info("Returning via Cartesian path...")
        return_ok = node._cartesian_move(0., RETURN_VELOCITY_SCALE,
                                         target_position=node._pre_push_pos)
        if not return_ok and not node._motion_uncertain:
            node.get_logger().error("Cartesian return failed — attempting joint-space fallback.")
            node._object_xyz = np.vstack((node._object_xyz,
                node._object_xyz + PUSH_DISTANCE*node._push_direction))
            node.move_to_pre_push(velocity_scale=RETURN_VELOCITY_SCALE)

    except KeyboardInterrupt:
        pass
    finally:
        stop_recording(node)
        if node._push_started and not node._motion_uncertain and rclpy.ok():
            # Home, so the post-push snapshot sees the object without the arm in the way. The pushed
            # object may have moved along the push vector: bound the swept volume.
            try:
                swept_xyz = np.vstack((node._object_xyz,
                    node._object_xyz + PUSH_DISTANCE*node._push_direction))
                with object_obstacle(node, swept_xyz):
                    home_ok = plan_and_execute_joint_goal(
                        node, node._move_group_client, joint_positions=HOME_JOINT_POSITIONS,
                        velocity_scaling_factor=APPROACH_VELOCITY_SCALE,
                        acceleration_scaling_factor=APPROACH_VELOCITY_SCALE)
            except RuntimeError as exc:
                node.get_logger().error(str(exc))
                home_ok = False
            if not home_ok:
                node.get_logger().error("Return home failed -- skipping the post-push snapshot.")
        node._servo_cmd.close()
        deactivate_egm(node)
        if stage is not None and node._approach_diag:
            try:
                write_json_atomic(stage.path / "push_approach_diagnostics.json", node._approach_diag)
            except Exception as exc:
                node.get_logger().error(f"Failed to save approach diagnostics: {exc}")
        if stage is not None:
            if node._push_started:
                try:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_ft_pose_log(node._ft_log, node._pose_log, subdir=str(stage.path),
                                     prefix="push_ft_pose", timestamp=ts, ft_raw_log=node._ft_capture.raw)
                    save_run_metadata(str(stage.path), "push_ft_pose", ts, {
                        **module_constants(globals()),
                        **node._ft_capture.metadata(),
                        "receive_time_origin_ros_s": node._log_start_s,
                        "object": node._object,
                        "pre_push": node._pre_push_pos,
                        "push_direction": node._push_direction.tolist(),
                        "push_orientation": node._push_orientation.tolist(),
                        "force_check": node._force_check,
                        "push_ok": push_ok,
                        "return_ok": return_ok,
                    })
                except Exception as exc:
                    node.get_logger().error(f"Failed to save push F/T+pose log: {exc}")
            status = "aborted" if not node._push_started else "ok" if push_ok and return_ok else "failed"
            stage.end(status, push_ok=push_ok, return_ok=return_ok, home_ok=home_ok, pre_push=node._pre_push_pos,
                      push_direction=node._push_direction.tolist(), force_check=node._force_check)
            if home_ok and rclpy.ok():
                take_snapshot(node, episode, label="after push")
        print(episode.summary())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if push_ok and return_ok and home_ok else 1


if __name__ == "__main__":
    sys.exit(main())
