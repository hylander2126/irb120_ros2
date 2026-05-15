#!/usr/bin/env python3
"""Force-informed arc-following squash-and-pull for the IRB120.

Squash phase: identical to squash_pull — descend until force reference is met.
Arc-follow phase: EE follows a circular arc whose:
  - center = (x_contact, y_contact, 0.0)  — directly below post-squash point at z=0
  - radius = z_contact                     — the post-squash EE height

At each tick the desired arc position is computed from the current angle, and
the tangential velocity drives the arc while a PI force controller adjusts the
radial component to maintain the target normal force against the surface.

UNPULL reverses the arc back to the squash angle.
"""

import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from abb_robot_msgs.srv import TriggerWithResultCode
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from irb120_control.controllers.force_controller import PIDForceController
from irb120_control.controllers.moveit_single_shot import plan_and_execute_joint_goal
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.runtime_log_dir import load_object_params, set_recorder_output_dir, save_ft_log, save_ft_pose_log, VALID_OBJECTS

_EGM_START_SRV = "/rws_client/start_egm_joint"

BASE_FRAME = "world"       # fixed world frame — used for TF lookups and arc geometry (z=0 is table plane)
SERVO_FRAME = "base_link"  # MoveIt Servo requires base_link as the twist command frame
EE_LINK = "finger_ball_center"

FORCE_HARD_LIMIT_N = 15.0
CONTACT_STABLE_SAMPLES = 1

DESCEND_SPEED = 0.005       # m/s
ARC_TANGENTIAL_SPEED = 0.008  # m/s along the arc
ARC_ANGLE_DEG = -20.0        # total arc to sweep (degrees)
ARC_CENTER_X_OFFSET = 0.005  # m — shifts arc center in +x so EE starts at a negative angle, giving an immediate -z tangential component

SQUASH_TIMEOUT_SEC = 30.0
ARC_TIMEOUT_SEC = 30.0
UNARC_TIMEOUT_SEC = 30.0
LULL_WAIT_SEC = 1.0
LULL_SETTLE_N = 0.5          # force must be within this many N of reference before leaving LULL
LULL_SETTLE_TIMEOUT_SEC = 8.0 # bail to ARC anyway after this long even if not settled
RETRACT_SPEED = 0.008       # m/s
RETRACT_DURATION_SEC = 3.0

KP_FORCE = 0.001
KI_FORCE = 0.0001
KD_FORCE = 0.0
MAX_NORMAL_SPEED = 0.020

KP_ORIENT = 1.0         # rad/s per rad of pitch error — tune up if sluggish
MAX_ORIENT_SPEED = 0.5  # rad/s clamp

KP_Y_FORCE = 0#.0025      # m/s per N of Y force error
KI_Y_FORCE = 0#.0005      # m/s per N·s of accumulated Y force error
MAX_Y_SPEED = 0.015     # m/s clamp on Y correction output
UNPULL_FORCE_BIAS_N = 0.5

CONTROL_HZ = 100.0
REQUIRE_OPERATOR_CONFIRM = True

LOST_CONTACT_FORCE_THRESH_N = 0.5
LOST_CONTACT_STEPS = 20

ARC_N_SAFETY = 0.15  # N — stop arc early if |fx| drops at or below this threshold


class ArcSquashPull(Node):
    def __init__(self) -> None:
        super().__init__("arc_squash_pull")
        self.declare_parameter("object", "")
        obj = self.get_parameter("object").get_parameter_value().string_value
        if obj not in VALID_OBJECTS:
            raise ValueError(
                f"Required parameter 'object' must be one of {sorted(VALID_OBJECTS)}, "
                f"got: '{obj}'. Pass it with: --ros-args -p object:=box"
            )
        self._object = obj
        self._log_subdir = f"{obj}/arc_squash"
        params = load_object_params(obj)
        ps = params["pre_squash"]
        self._pre_squash_pos = (ps["x"], ps["y"], ps["z"])
        self._pre_squash_ori = (ps["qx"], ps["qy"], ps["qz"], ps["qw"])
        self._pre_squash_q = params["pre_squash_q"]

        self._tf_buffer       = Buffer()
        self._tf_listener     = TransformListener(self._tf_buffer, self)
        self._twist_pub       = self.create_publisher(TwistStamped, "/servo_node/delta_twist_cmds", 10)
        self._wrench_sub      = self.create_subscription(WrenchStamped, "/netft_data", self._on_wrench, 10)
        self._wrench_ctrl_sub = self.create_subscription(WrenchStamped, "/netft_data_transformed", self._on_wrench_ctrl, 10)
        self._det_sub         = self.create_subscription(Detection3DArray, "/object_detector/detections", self._on_detection, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._timer = None

        if "force_ref_n" not in params:
            raise ValueError(f"Object '{obj}' is missing required 'force_ref_n' in object_params.json")
        force_ref = float(params["force_ref_n"])
        self.get_logger().info(f"Object: {obj}  force_ref={force_ref:.1f}N  hard_limit={FORCE_HARD_LIMIT_N:.1f}N")
        self._force_ctrl = PIDForceController(
            kp=KP_FORCE,
            ki=KI_FORCE,
            kd=KD_FORCE,
            force_ref_n=force_ref,
            max_normal_speed=MAX_NORMAL_SPEED,
            control_hz=CONTROL_HZ,
        )

        self._state = "SQUASH"
        self._done = False
        self._contact_count = 0
        self._force_x = 0.0
        self._force_z = 0.0
        self._have_force = False
        self._contact_felt = False
        self._last_tf_warn_time = 0.0
        self._state_start_time = 0.0
        self._lost_contact_count = 0
        self._lull_next: str = "ARC"
        self._saved_force_ref = force_ref
        self._force_y = 0.0                   # signed world-frame Y force from transformed wrench
        self._force_y_ref: float | None = None # Y force at squash contact — maintained throughout arc
        self._vy_integral: float = 0.0        # PI integrator for Y force controller

        # Arc geometry — set when SQUASH completes
        self._arc_center_x: float | None = None    # x of post-squash EE = center x
        self._arc_center_y: float | None = None    # y of post-squash EE = center y
        self._arc_radius: float | None = None      # z_contact (height above z=0 plane)
        self._arc_start_angle: float | None = None # angle at squash contact (radians, in XZ plane)
        self._arc_end_angle: float | None = None   # target angle after full sweep

        self._egm_keepalive_client = self.create_client(TriggerWithResultCode, _EGM_START_SRV)
        self._pause_servo_client   = self.create_client(SetBool, "/servo_node/pause_servo")

        self._ft_transformed_log: list = []
        self._ft_raw_log: list = []
        self._pose_log: list = []
        self._obj_pose_log: list = []
        self._last_force_log_time = 0.0

    # ------------------------------------------------------------------ #
    #  Servo pause / resume
    # ------------------------------------------------------------------ #

    def _set_servo_paused(self, paused: bool) -> None:
        if not self._pause_servo_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("pause_servo service not available")
            return
        future = self._pause_servo_client.call_async(SetBool.Request(data=paused))
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done() and future.result() is not None:
            state = "paused" if paused else "resumed"
            self.get_logger().info(f"Servo {state}: {future.result().message}")
        else:
            self.get_logger().warn("pause_servo call did not complete")

    def pause_servo(self) -> None:
        self._set_servo_paused(True)

    def resume_servo(self) -> None:
        self._set_servo_paused(False)

    def ensure_egm_for_moveit(self) -> bool:
        """Pause servo, re-enable EGM, and wait for JTC to be ready for MoveIt.

        Call this before any MoveIt joint goal when EGM may have timed out
        mid-sequence (e.g. after a long arc or retract phase).
        """
        self.pause_servo()
        self.get_logger().info("Re-enabling EGM before MoveIt return move...")
        return ensure_egm_active(self)

    # ------------------------------------------------------------------ #
    #  MoveIt approach
    # ------------------------------------------------------------------ #

    def move_to_pre_squash(self) -> bool:
        return plan_and_execute_joint_goal(
            self,
            self._move_group_client,
            joint_positions=self._pre_squash_q,
            joint_tolerance=0.003
        )

    # ------------------------------------------------------------------ #
    #  Subscribers
    # ------------------------------------------------------------------ #

    def _on_wrench_ctrl(self, msg: WrenchStamped) -> None:
        # Store signed world-frame components; radial projection computed per-tick once angle is known
        self._force_x = msg.wrench.force.x
        self._force_y = msg.wrench.force.y
        self._force_z = abs(msg.wrench.force.z)
        self._have_force = True
        t = self._now_s()
        self._ft_transformed_log.append([
            t,
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
        ])
        if self._force_z > 0.25 and t - self._last_force_log_time > 0.2:
            self._last_force_log_time = t
            self.get_logger().info(f"z_force: {self._force_z:.2f} N  state: {self._state}")

    def _on_wrench(self, msg: WrenchStamped) -> None:
        t = self._now_s()
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        tx = msg.wrench.torque.x
        ty = msg.wrench.torque.y
        tz = msg.wrench.torque.z
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
        self._ft_raw_log.append([t, fx, fy, fz, tx, ty, tz, ft_px, ft_py, ft_pz, ft_qx, ft_qy, ft_qz, ft_qw])

    def _on_detection(self, msg: Detection3DArray) -> None:
        if not msg.detections:
            return
        hyp = msg.detections[0].results[0] if msg.detections[0].results else None
        if hyp is None:
            return
        t = self._now_s()
        p = hyp.pose.pose.position
        q = hyp.pose.pose.orientation
        self._obj_pose_log.append([t, p.x, p.y, p.z, q.x, q.y, q.z, q.w])

    # ------------------------------------------------------------------ #
    #  TF / helpers
    # ------------------------------------------------------------------ #

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

    def _warn_throttled(self, message: str, throttle_hz: float = 0.2) -> None:
        now = self._now_s()
        if now - self._last_tf_warn_time > 1.0 / throttle_hz:
            self._last_tf_warn_time = now
            self.get_logger().warn(message)

    def _check_timeout(self, timeout_sec: float, label: str) -> bool:
        if self._now_s() - self._state_start_time > timeout_sec:
            self.get_logger().error(f"{label} timed out after {timeout_sec:.0f}s — retracting")
            self._transition("RETRACT")
            return True
        return False

    def _check_lost_contact(self, force: float | None = None) -> bool:
        if (force if force is not None else self._force_z) < LOST_CONTACT_FORCE_THRESH_N:
            self._lost_contact_count += 1
            if self._lost_contact_count >= LOST_CONTACT_STEPS:
                self.get_logger().error(
                    f"Lost contact in {self._state} ({self._lost_contact_count} samples "
                    f"below {LOST_CONTACT_FORCE_THRESH_N:.1f} N) — retracting"
                )
                self._transition("RETRACT")
                return True
        else:
            self._lost_contact_count = 0
        return False

    def _transition(self, state: str) -> None:
        if state != self._state:
            self.get_logger().info(f"{self._state} -> {state}")
            self._state = state
            self._state_start_time = self._now_s()
            self._lost_contact_count = 0

    def _publish_twist(self, vx: float, vy: float, vz: float, wy: float = 0.0) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = SERVO_FRAME
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.y = wy
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
            self.get_logger().warn("Operator aborted arc squash-pull sequence")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Arc geometry helpers
    # ------------------------------------------------------------------ #

    def _init_arc(self, x_contact: float, y_contact: float, z_contact: float) -> None:
        """Compute arc parameters from the post-squash EE position.

        Arc lives in the XZ plane at y = y_contact.
        Center: (x_contact, y_contact, 0)
        Radius: z_contact  (height of EE above the z=0 plane)

        The start angle is the angle of the EE from the center, measured
        from the +Z axis toward +X:  theta = atan2(dx, dz)
        where dx = x_contact - cx = 0 and dz = z_contact - 0 = z_contact,
        so start_angle = 0.
        """
        self._arc_center_x = x_contact + ARC_CENTER_X_OFFSET  # +x shift puts EE at negative start angle → immediate -z tangential component
        self._arc_center_y = y_contact
        self._arc_radius = z_contact
        self._arc_start_angle = math.atan2(x_contact - self._arc_center_x, z_contact)
        self._arc_end_angle = math.radians(ARC_ANGLE_DEG)
        self.get_logger().info(
            f"Arc init: center=({self._arc_center_x:.4f}, {y_contact:.4f}, 0)  "
            f"r={z_contact:.4f} m  start={math.degrees(self._arc_start_angle):.1f} deg"
        )

    def _arc_velocity(self, x: float, z: float, tangential_speed: float, radial_correction: float) -> tuple[float, float]:
        """Return (vx, vz) for arc-following with force feedback.

        tangential_speed: signed speed along the arc (positive = increasing angle = -x direction)
        radial_correction: signed outward radial velocity
                           positive = away from center (reduce contact force)
                           negative = toward center (increase contact force)
                           Caller is responsible for negating PIForceController output before passing here.

        The arc is in the XZ plane. The unit tangent at angle theta (from +Z toward +X):
          tangent = (-cos(theta), sin(theta))  in (x, z)
        The unit radial (outward from center) at angle theta:
          radial  = (sin(theta), cos(theta))   in (x, z)

        theta is recovered from the current EE position.
        """
        cx = self._arc_center_x
        dx = x - cx
        dz = z - 0.0  # center is at z=0
        theta = math.atan2(dx, dz)  # angle from +Z toward +X

        # Unit tangent (in direction of increasing theta, i.e. toward +X side)
        tx = -math.cos(theta)
        tz = math.sin(theta)

        # Unit radial (outward)
        rx = math.sin(theta)
        rz = math.cos(theta)

        vx = tangential_speed * tx + radial_correction * rx
        vz = tangential_speed * tz + radial_correction * rz
        return vx, vz

    def _current_arc_angle(self, x: float, z: float) -> float:
        """Angle of EE from center, measured from +Z toward +X (radians)."""
        return math.atan2(x - self._arc_center_x, z)

    def _radial_force(self, theta: float) -> float:
        """Project world-frame force onto the outward arc normal at angle theta.

        The outward radial unit vector in the XZ plane is (sin(theta), cos(theta)),
        so F_radial = fx*sin(theta) + fz*cos(theta).  At theta=0 this is pure Fz;
        at theta=-90 deg it is pure -Fx (inward becomes Fx pointing toward robot base).
        We return the magnitude so the force controller always sees a positive value
        representing contact force, matching its expectation from the squash phase.
        """
        return abs(self._force_x * math.sin(theta) + self._force_z * math.cos(theta))

    def _vy_force(self) -> float:
        """PI controller that maintains the Y force captured at squash contact."""
        if self._force_y_ref is None:
            return 0.0
        err = self._force_y_ref - self._force_y
        self._vy_integral += err / CONTROL_HZ
        output = KP_Y_FORCE * err + KI_Y_FORCE * self._vy_integral
        return max(-MAX_Y_SPEED, min(MAX_Y_SPEED, output))

    @staticmethod
    def _quat_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
        """Extract Y-axis (pitch) rotation from quaternion, in radians."""
        # Standard ZYX Euler: pitch = asin(2*(qw*qy - qz*qx))
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        return math.asin(max(-1.0, min(1.0, sin_pitch)))

    # ------------------------------------------------------------------ #
    #  Main control tick
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        if self._done:
            self._publish_zero()
            return

        pose = self._lookup_pose()
        if pose is None:
            self._publish_zero()
            return

        t = self._now_s()
        px = pose.pose.position.x
        py = pose.pose.position.y
        pz = pose.pose.position.z
        q = pose.pose.orientation
        self._pose_log.append([t, px, py, pz, q.x, q.y, q.z, q.w])

        if self._have_force and self._force_z > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(f"Hard force limit exceeded: {self._force_z:.2f} N — retracting")
            self._transition("RETRACT")

        # -------- SQUASH --------
        if self._state == "SQUASH":
            if self._check_timeout(SQUASH_TIMEOUT_SEC, "SQUASH"): return
            if self._have_force and self._force_z > 0.25 and not self._contact_felt:
                self._contact_felt = True
                self._force_y_ref = self._force_y  # capture lateral force at first contact
                self.get_logger().info(f"Contact felt! Y force ref set: {self._force_y_ref:.3f} N")
            self._publish_twist(0.0, 0.0, -DESCEND_SPEED)
            if self._have_force and self._force_z >= self._force_ctrl.reference:
                self._contact_count += 1
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._transition("LULL")
            else:
                self._contact_count = 0
            return

        # -------- LULL --------
        if self._state == "LULL":
            normal_z = -self._force_ctrl.update(self._force_z)
            self._publish_twist(0.0, 0.0, normal_z)
            elapsed = self._now_s() - self._state_start_time
            if elapsed < LULL_WAIT_SEC:
                return
            # Wait until force is close to reference, or timeout
            force_error = abs(self._force_z - self._force_ctrl.reference)
            settled = force_error < LULL_SETTLE_N
            timed_out = elapsed > LULL_SETTLE_TIMEOUT_SEC
            if not settled and not timed_out:
                return
            if timed_out and not settled:
                self.get_logger().warn(
                    f"LULL settle timeout — proceeding with force error {force_error:.2f} N"
                )
            if self._lull_next == "ARC":
                self._force_ctrl.reset()
                # Cap at the object's configured force_ref so we don't enter ARC near the hard limit
                pull_ref = min(self._force_z, self._force_ctrl.reference)
                self._force_ctrl.set_reference(pull_ref)
                self._saved_force_ref = pull_ref
                self.get_logger().info(f"ARC setpoint: {pull_ref:.2f} N (measured={self._force_z:.2f} N)")
                # Reset Y PI state
                self._vy_integral = 0.0
                # Capture post-squash position and build arc geometry
                self._init_arc(px, py, pz)
            elif self._lull_next == "UNARC":
                pass  # hold whatever setpoint was active during ARC; no force modulation here
            self._transition(self._lull_next)
            return

        # -------- ARC --------
        if self._state == "ARC":
            if self._check_timeout(ARC_TIMEOUT_SEC, "ARC"): return
            angle = self._current_arc_angle(px, pz)
            f_radial = self._radial_force(angle)
            if self._check_lost_contact(f_radial): return

            # ARC motion with continuous PI Y correction
            radial_corr = -self._force_ctrl.update(f_radial)
            vx, vz = self._arc_velocity(px, pz, ARC_TANGENTIAL_SPEED, radial_corr)
            vy = self._vy_force()
            current_pitch = self._quat_to_pitch(q.x, q.y, q.z, q.w)
            pitch_err = angle - current_pitch
            wy = max(-MAX_ORIENT_SPEED, min(MAX_ORIENT_SPEED, KP_ORIENT * pitch_err))
            self._publish_twist(vx, vy, vz, wy)

            angle_deg = math.degrees(angle)
            if t - self._last_force_log_time > 0.2:
                self._last_force_log_time = t
                self.get_logger().info(
                    f"arc_angle: {angle_deg:.1f} / {ARC_ANGLE_DEG:.1f} deg  "
                    f"pitch: {math.degrees(current_pitch):.1f} deg  err: {math.degrees(pitch_err):.1f} deg  "
                    f"f_radial: {f_radial:.2f} N  fy: {self._force_y:.2f} N  vy: {vy:.4f} m/s"
                    f"  (fx={self._force_x:.2f} fz={self._force_z:.2f})"
                )
            if angle <= self._arc_end_angle:
                self._lull_next = "UNARC"
                self._transition("LULL")
                return
            # n_safety: stop arc early if the pulling x-force has decayed to near zero
            # skip the first 5 degrees to avoid triggering on transients at arc start
            if abs(self._arc_start_angle - angle) >= math.radians(5.0) and abs(self._force_x) <= ARC_N_SAFETY:
                self.get_logger().info(
                    f"n_safety stop: |fx|={abs(self._force_x):.2f} N <= {ARC_N_SAFETY:.2f} N "
                    f"at arc_angle={math.degrees(angle):.1f} deg"
                )
                self._lull_next = "UNARC"
                self._transition("LULL")
            return

        # -------- UNARC --------
        if self._state == "UNARC":
            if self._check_timeout(UNARC_TIMEOUT_SEC, "UNARC"): return
            angle = self._current_arc_angle(px, pz)
            f_radial = self._radial_force(angle)
            if self._check_lost_contact(f_radial): return
            radial_corr = -self._force_ctrl.update(f_radial)
            # Reverse: negative tangential_speed moves back toward start angle
            vx, vz = self._arc_velocity(px, pz, -ARC_TANGENTIAL_SPEED, radial_corr)
            vy = self._vy_force()
            current_pitch = self._quat_to_pitch(q.x, q.y, q.z, q.w)
            pitch_err = angle - current_pitch
            wy = max(-MAX_ORIENT_SPEED, min(MAX_ORIENT_SPEED, KP_ORIENT * pitch_err))
            self._publish_twist(vx, vy, vz, wy)

            angle = self._current_arc_angle(px, pz)
            if angle >= self._arc_start_angle - math.radians(1.0):  # 1 deg tolerance
                self._force_ctrl.set_reference(self._force_ctrl.reference)
                self._transition("RETRACT")
            return

        # -------- RETRACT --------
        if self._state == "RETRACT":
            elapsed = self._now_s() - self._state_start_time
            if elapsed < RETRACT_DURATION_SEC:
                # Publish zero-force twist upward; if servo has no subscribers
                # the robot won't move but we still complete the timer and hand
                # off to the MoveIt return move which will lift it correctly.
                msg = TwistStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = SERVO_FRAME
                msg.twist.linear.z = RETRACT_SPEED
                self._twist_pub.publish(msg)
            else:
                self._publish_zero()
                self._done = True
            return


def main(args=None) -> int:
    rclpy.init(args=args)
    node = ArcSquashPull()
    recorder_client = node.create_client(SetBool, "/camera_hull_recorder/set_recording")
    try:
        set_recorder_output_dir(node, node._log_subdir)
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

        if not ensure_egm_active(node):
            return 1

        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error(
                "MoveIt Servo is not ready (/servo_node/delta_twist_cmds has no subscribers). "
                "Launch stack with start_servo:=true before running arc_squash_pull."
            )
            return 1

        if not node.move_to_pre_squash():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        node.get_logger().info(
            f"[PARAMS] object={node._object}  "
            f"force_ref={node._force_ctrl.reference:.2f}N  "
            f"hard_limit={FORCE_HARD_LIMIT_N:.1f}N  "
            f"pre_squash_pos={node._pre_squash_pos}  "
            f"kp={KP_FORCE}  ki={KI_FORCE}  "
            f"descend_speed={DESCEND_SPEED}m/s  arc_sweep={ARC_ANGLE_DEG}deg"
        )

        if not node._operator_confirm(
            "At pre-squash pose. Confirm clear contact conditions before descending."
        ):
            return 0

        node.resume_servo()
        node._state_start_time = node._now_s()
        node._timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.05)

        if rclpy.ok():
            node._publish_zero()
            # Re-enable EGM before handing off to MoveIt — it may have timed out
            # during a long arc sequence or after the servo-based retract phase.
            if not node.ensure_egm_for_moveit():
                node.get_logger().error("EGM re-enable failed — cannot execute MoveIt return move.")
            else:
                node.get_logger().info("Returning to pre-squash pose via MoveIt...")
                if not node.move_to_pre_squash():
                    node.get_logger().error("MoveIt return to pre-squash failed.")
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
        try:
            save_ft_pose_log(node._ft_transformed_log, node._pose_log, node._log_subdir, "arc_squash_pull", node._obj_pose_log, ft_raw_log=node._ft_raw_log)
        except Exception as exc:
            node.get_logger().error(f"Failed to save F/T+pose log: {exc}")
        node._publish_zero()
        deactivate_egm(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
