#!/usr/bin/env python3
"""Static-wrist force-informed arc-following squash-and-pull for the IRB120.

Squash phase: identical to squash_pull — descend until force reference is met.
Arc-follow phase: EE follows a circular arc whose:
  - center = (x_contact + offset, y_contact, 0.0)
  - radius = z_contact                     — the post-squash EE height

At each tick the desired arc position is computed from the measured EE angle.
The finger orientation is held fixed while a PI force controller adjusts the
radial component computed in the actual XZ arc frame.

UNARC reverses the arc back to the squash angle.  If contact is lost during
ARC or UNARC, the complete attempt is re-run with a higher press force.
"""

import argparse
import math
import sys
from datetime import datetime

import rclpy
from geometry_msgs.msg import WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from irb120_control.controllers.force_controller import PIDForceController
from irb120_control.controllers.moveit_single_shot import plan_and_execute_joint_goal, plan_and_execute_pose_goal
from irb120_control.controllers.servo_command_publisher import ServoCommandPublisher
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.press_point_check import check_press_point
from irb120_control.util.motion_geometry import (
    arc_angle_xz,
    arc_velocity_xz,
    clamp,
    quat_to_pitch,
    radial_force_xz,
)
from irb120_control.util.runtime_log_dir import (
    load_object_params,
    module_constants,
    save_ft_pose_log,
    save_run_metadata,
    start_recording,
    stop_recording,
    VALID_OBJECTS,
)

BASE_FRAME = "world"       # fixed world frame — used for TF lookups and arc geometry (z=0 is table plane)
SERVO_FRAME = "base_link"  # MoveIt Servo requires base_link as the twist command frame
EE_LINK = "finger_ball_center"

HOME_JOINT_POSITIONS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # all-zero joint configuration

STATE_IDS = {
    "SQUASH": 1,
    "LULL": 2,
    "ARC": 3,
    "UNARC": 4,
    "RETRACT": 5,
}

FORCE_HARD_LIMIT_N = 20.0  # was 15.0 — moveit_servo's smoothing filter (servo.yaml,
# use_smoothing: true) lets commanded velocity decay gradually rather than
# stopping instantly, so SQUASH/LULL routinely overshoots ~5N past the squash
# target before the arm actually stops (observed peaks ~15.1-15.4N against a
# 10N target on the monitor). 20N clears that with margin.
CONTACT_STABLE_SAMPLES = 1

DESCEND_SPEED = 0.005       # m/s
ARC_TANGENTIAL_SPEED = 0.008  # m/s along the arc
ARC_TANGENTIAL_RAMP_SEC = 2.0 # seconds to ramp tangential speed at ARC/UNARC onset
ARC_MAX_ANGLE_DEG = -23.0     # safety cap; ARC exits earlier once fx flips sign
ARC_CENTER = (0.61, 0.0, 0.0) # world-frame pivot edge / arc center (x, y, z) # IF BOX AND HEART FAIL, IT'S CAUSE OF THIS, SORTOF. WE USED THE OLD ARC CALCULATION FOR THOSE
ARC_FX_SIGN_DEADBAND_N = 0.08
ARC_FX_SIGN_MIN_SWEEP_DEG = 5.0
ARC_FX_SIGN_MIN_SAMPLES = 20
ARC_FX_FLIP_STABLE_SAMPLES = 5
# ARC exits when the tangent force decays toward zero — that IS the tipping angle
# theta*, so this threshold decides how close to theta* the sweep actually gets, and the
# parameter fit then extrapolates the rest of the way.
#
# A single ABSOLUTE threshold cannot serve objects whose tangential force differs by
# ~24x (measured torque RMS: heart 0.075 N·m vs monitor 1.79 N·m). At 0.1 N the heart
# tripped 3.4-7.6 deg short of theta* while the box got within 0.8 deg — and the box had
# 0.65% z_c error against the heart's 15.6%. Hence a RELATIVE threshold: stop at a
# fraction of the peak tangent force this trial actually produced, so every object stops
# at the same point on its own decay curve. ARC_FX_LOW_FLOOR_N keeps it above sensor
# noise for the lightest objects.
ARC_FX_LOW_FRACTION = 0.08   # of this trial's peak tangent force
ARC_FX_LOW_FLOOR_N  = 0.1    # absolute floor — the old fixed threshold, now a lower bound
ARC_FX_LOW_STABLE_SAMPLES = 5    # number of consecutive ticks below threshold required

SQUASH_TIMEOUT_SEC = 30.0
ARC_TIMEOUT_SEC = 30.0
UNARC_TIMEOUT_SEC = 30.0
LULL_WAIT_SEC = 1.0
LULL_SETTLE_N = 0.5          # force must be within this many N of reference before leaving LULL
LULL_SETTLE_TIMEOUT_SEC = 8.0 # bail to ARC anyway after this long even if not settled
RETRACT_SPEED = 0.008       # m/s
RETRACT_DURATION_SEC = 3.0

KP_FORCE = 0.00035
KI_FORCE = 0.000005
KD_FORCE = 0.0
MAX_NORMAL_SPEED = 0.006
FORCE_DEADBAND_N = 0.25
FORCE_FILTER_ALPHA = 0.12
FORCE_OUTPUT_SLEW_RATE = 0.02  # m/s^2
UNARC_FORCE_AUGMENT_SPEED = 0.004  # m/s inward bias at large force deficit, still clamped by MAX_NORMAL_SPEED
UNARC_FORCE_AUGMENT_SOFTNESS_N = 0.75  # larger = smoother/slower augmentation onset

KP_Y_FORCE = 0#.0025      # m/s per N of Y force error
KI_Y_FORCE = 0#.0005      # m/s per N·s of accumulated Y force error
MAX_Y_SPEED = 0.015     # m/s clamp on Y correction output

CONTROL_HZ = 100.0
REQUIRE_OPERATOR_CONFIRM = True

LOST_CONTACT_FORCE_THRESH_N = 0.3
LOST_CONTACT_STEPS = 20

# Retry policy.  The configured per-object force is always the first attempt.
ADAPTIVE_FORCE_SCALE_FACTOR = 1.25
ADAPTIVE_FORCE_MAX_N = 13.0

def _quat_rotate(q, v):
    """Rotate vector v by quaternion q=(x,y,z,w). Plain math — this is a control node,
    it should not pull in the estimation stack just to move one point between frames."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 * (q_vec x v);  v' = v + w*t + q_vec x t
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def _quat_mul(a, b):
    """Hamilton product a*b, both (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


class ArcStatic(Node):
    def __init__(self, object_name: str | None = None) -> None:
        super().__init__("arc_static")
        self.declare_parameter("object", "")
        obj = object_name or self.get_parameter("object").get_parameter_value().string_value
        if obj not in VALID_OBJECTS:
            raise ValueError(
                f"Required parameter 'object' must be one of {sorted(VALID_OBJECTS)}, "
                f"got: '{obj}'. Pass it as: arc_static box"
            )
        self._object = obj
        self._log_subdir = f"{obj}/arc_squash"
        params = load_object_params(obj)
        ps = params["pre_squash"]
        self._pre_squash_pos = (ps["x"], ps["y"], ps["z"])
        self._pre_squash_ori = (ps["qx"], ps["qy"], ps["qz"], ps["qw"])

        self._tf_buffer       = Buffer()
        self._tf_listener     = TransformListener(self._tf_buffer, self)
        self._servo_cmd       = ServoCommandPublisher(self, "/servo_node/delta_twist_cmds", SERVO_FRAME)
        self._wrench_sub      = self.create_subscription(WrenchStamped, "/netft_data_transformed", self._on_wrench, 10)
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
            deadband_n=FORCE_DEADBAND_N,
            measurement_filter_alpha=FORCE_FILTER_ALPHA,
            output_slew_rate=FORCE_OUTPUT_SLEW_RATE,
        )

        self._state = "SQUASH"
        self._done = False
        self._completed = False
        self._retry_requested = False
        self._contact_count = 0
        self._force_x = 0.0
        self._force_z = 0.0
        self._force_z_signed = 0.0
        self._have_force = False
        self._contact_felt = False
        self._last_tf_warn_time = 0.0
        self._state_start_time = 0.0
        self._lost_contact_count = 0
        self._lull_next: str = "ARC"
        self._force_y = 0.0                   # signed world-frame Y force from transformed wrench
        self._force_y_ref: float | None = None # Y force at squash contact — maintained throughout arc
        self._vy_integral: float = 0.0        # PI integrator for Y force controller

        # Arc geometry — set when SQUASH completes
        self._arc_center_x: float | None = None    # fixed world-frame arc center x
        self._arc_center_z: float | None = None    # fixed world-frame arc center z
        self._arc_start_angle: float | None = None # angle at squash contact (radians, in XZ plane)
        self._arc_end_angle: float | None = None   # target angle after full sweep
        self._arc_fx_pos_count = 0
        self._arc_fx_neg_count = 0
        self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign: int | None = None
        self._arc_fx_low_count = 0
        self._arc_peak_tangent = 0.0

        self._pause_servo_client   = self.create_client(SetBool, "/servo_node/pause_servo")

        self._ft_transformed_log: list = []
        self._pose_log: list = []     # rows: [time_s, x, y, z, qx, qy, qz, qw, arc_angle_rad, wrist_pitch_rad, state_id]
        self._obj_pose_log: list = [] # rows: [time_s, x, y, z, qx, qy, qz, qw, obj_pitch_rad]
        self._command_log: list = []  # rows: [time_s, state_id, angle, f_radial, radial_corr, vx, vy, vz, wy, tangential_speed]
        self._last_wrench_log_time = 0.0
        self._last_arc_log_time = 0.0

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

    # ------------------------------------------------------------------ #
    #  MoveIt approach
    # ------------------------------------------------------------------ #

    def move_to_pre_squash(self) -> bool:
        return plan_and_execute_pose_goal(
            self,
            self._move_group_client,
            target_position=self._pre_squash_pos,
            target_orientation=self._pre_squash_ori,
            velocity_scale=0.1,
            acceleration_scale=0.1,
        )

    def move_to_home(self) -> bool:
        """Return to the robot's all-zero joint configuration, gently — same
        velocity/acceleration scale as the pre-squash approach."""
        return plan_and_execute_joint_goal(
            self,
            self._move_group_client,
            joint_positions=HOME_JOINT_POSITIONS,
            velocity_scaling_factor=0.1,
            acceleration_scaling_factor=0.1,
        )

    # ------------------------------------------------------------------ #
    #  Subscribers
    # ------------------------------------------------------------------ #

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # Keep the signed fixed-orientation XZ force for actual-angle radial/tangent projections.
        self._force_x = msg.wrench.force.x
        self._force_y = msg.wrench.force.y
        self._force_z_signed = msg.wrench.force.z
        self._force_z = abs(self._force_z_signed)
        self._have_force = True
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
        if self._force_z > 0.25 and t - self._last_wrench_log_time > 0.2:
            self._last_wrench_log_time = t

    def _on_detection(self, msg: Detection3DArray) -> None:
        if not msg.detections:
            return
        hyp = msg.detections[0].results[0] if msg.detections[0].results else None
        if hyp is None:
            return
        t = self._now_s()
        p = hyp.pose.pose.position
        q = hyp.pose.pose.orientation

        # The perception stack publishes in 'base_link' (see perception.launch.py) but
        # every other stream in this log — EE pose, ft pose, the arc centre — is in
        # BASE_FRAME ('world'). Those differ by the 21 mm bracket under the robot base,
        # so logging the detection raw would put the object 21 mm low in z relative to
        # everything the estimator compares it against.
        src = msg.header.frame_id
        if src and src != BASE_FRAME:
            try:
                tf = self._tf_buffer.lookup_transform(BASE_FRAME, src, rclpy.time.Time())
            except TransformException as exc:
                self._warn_throttled(f"Dropping detection: no TF {src} -> {BASE_FRAME}: {exc}")
                return
            tr, ro = tf.transform.translation, tf.transform.rotation
            rq = (ro.x, ro.y, ro.z, ro.w)
            rx, ry, rz = _quat_rotate(rq, (p.x, p.y, p.z))
            px, py, pz = rx + tr.x, ry + tr.y, rz + tr.z
            qx, qy, qz, qw = _quat_mul(rq, (q.x, q.y, q.z, q.w))
        else:
            px, py, pz = p.x, p.y, p.z
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

        obj_pitch = quat_to_pitch(qx, qy, qz, qw)
        self._obj_pose_log.append([t, px, py, pz, qx, qy, qz, qw, obj_pitch])

    # ------------------------------------------------------------------ #
    #  TF / helpers
    # ------------------------------------------------------------------ #

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
                self._retry_requested = True
                self.get_logger().warn(
                    f"Lost contact in {self._state} ({self._lost_contact_count} samples "
                    f"below {LOST_CONTACT_FORCE_THRESH_N:.1f} N) — retracting before retry"
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

    def _wait_for_servo_ready(self, timeout_sec: float = 5.0) -> bool:
        end_time = self._now_s() + timeout_sec
        while rclpy.ok() and self._now_s() < end_time:
            if self._servo_cmd.has_subscribers():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _operator_confirm(self, message: str) -> bool:
        if not REQUIRE_OPERATOR_CONFIRM:
            return True
        if not self._servo_cmd.publish_zero(self._state, self._force_z):
            self._done = True
            return False
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

        Arc lives in the XZ plane around fixed world-frame ARC_CENTER.
        Radius is the current EE distance from that fixed pivot edge.

        The start angle is the angle of the EE from the center, measured
        from the +Z axis toward +X:  theta = atan2(dx, dz)
        """
        center_x, center_y, center_z = ARC_CENTER
        self._arc_center_x = center_x
        self._arc_center_z = center_z
        self._arc_start_angle = arc_angle_xz(x_contact, z_contact, self._arc_center_x, self._arc_center_z)
        self._arc_end_angle = math.radians(ARC_MAX_ANGLE_DEG)
        radius = math.hypot(x_contact - center_x, z_contact - center_z)
        self._arc_fx_pos_count = 0
        self._arc_fx_neg_count = 0
        self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign = None
        self._arc_fx_low_count = 0
        self._arc_peak_tangent = 0.0
        self.get_logger().info(
            f"Arc init: center=({center_x:.4f}, {center_y:.4f}, {center_z:.4f})  "
            f"r={radius:.4f} m  start={math.degrees(self._arc_start_angle):.1f} deg"
        )

    def _current_arc_angle(self, x: float, z: float) -> float:
        return arc_angle_xz(x, z, self._arc_center_x, self._arc_center_z)

    def _radial_force(self, theta: float) -> float:
        return radial_force_xz(theta, self._force_x, self._force_z_signed)

    def _tangent_force(self, theta: float) -> float:
        return self._force_x * math.cos(theta) - self._force_z_signed * math.sin(theta)

    def _controlled_contact_force(self, x: float, z: float) -> float:
        """Return the force component currently used for force-dependent safety."""
        if self._state in ("ARC", "UNARC") and self._arc_center_x is not None:
            return self._radial_force(self._current_arc_angle(x, z))
        return self._force_z

    def _vy_force(self) -> float:
        """PI controller that maintains the Y force captured at squash contact."""
        if self._force_y_ref is None:
            return 0.0
        err = self._force_y_ref - self._force_y
        self._vy_integral += err / CONTROL_HZ
        output = KP_Y_FORCE * err + KI_Y_FORCE * self._vy_integral
        return clamp(output, MAX_Y_SPEED)

    def _arc_fx_flipped(self, angle: float) -> bool:
        """Return True once actual-angle tangent force crosses the dominant ARC sign."""
        if self._arc_start_angle is None:
            return False
        swept = abs(self._arc_start_angle - angle)
        if swept < math.radians(ARC_FX_SIGN_MIN_SWEEP_DEG):
            return False
        tangent_force = self._tangent_force(angle)
        if abs(tangent_force) < ARC_FX_SIGN_DEADBAND_N:
            self._arc_fx_flip_count = 0
            return False

        sign = 1 if tangent_force > 0.0 else -1
        if sign > 0:
            self._arc_fx_pos_count += 1
        else:
            self._arc_fx_neg_count += 1

        total = self._arc_fx_pos_count + self._arc_fx_neg_count
        if self._arc_fx_majority_sign is None and total >= ARC_FX_SIGN_MIN_SAMPLES:
            self._arc_fx_majority_sign = 1 if self._arc_fx_pos_count >= self._arc_fx_neg_count else -1
            self.get_logger().info(
                f"ARC tangent-force majority sign locked: {'+' if self._arc_fx_majority_sign > 0 else '-'} "
                f"(pos={self._arc_fx_pos_count}, neg={self._arc_fx_neg_count})"
            )

        if self._arc_fx_majority_sign is None or sign == self._arc_fx_majority_sign:
            self._arc_fx_flip_count = 0
            return False

        self._arc_fx_flip_count += 1
        return self._arc_fx_flip_count >= ARC_FX_FLIP_STABLE_SAMPLES

    def _publish_arc_step(
        self,
        x: float,
        z: float,
        _pitch: float,
        tangential_speed: float,
    ) -> tuple[float, float, float, bool]:
        angle = self._current_arc_angle(x, z)
        f_radial = self._radial_force(angle)

        if self._check_lost_contact(f_radial):
            return angle, f_radial, 0.0, False

        radial_corr = -self._force_ctrl.update(f_radial)
        # Arc-only experiment option: disable radial force regulation by restoring:
        # radial_corr = 0.0
        if self._state == "UNARC":
            force_deficit = max(0.0, self._force_ctrl.reference - f_radial)
            augment = force_deficit / (force_deficit + UNARC_FORCE_AUGMENT_SOFTNESS_N) if force_deficit > 0.0 else 0.0
            radial_corr = clamp(
                radial_corr - UNARC_FORCE_AUGMENT_SPEED * augment,
                MAX_NORMAL_SPEED,
            )

        ramp = min(1.0, max(0.0, (self._now_s() - self._state_start_time) / ARC_TANGENTIAL_RAMP_SEC))
        tangential_cmd = tangential_speed * ramp
        vx, vz = arc_velocity_xz(angle, tangential_cmd, radial_corr)
        vy = self._vy_force()
        wy = 0.0
        self._command_log.append([
            self._now_s(),
            STATE_IDS.get(self._state, 0),
            angle,
            f_radial,
            radial_corr,
            vx,
            vy,
            vz,
            wy,
            tangential_cmd,
        ])
        if not self._servo_cmd.publish_twist(vx, vy, vz, wy, self._state, f_radial):
            self._done = True
            return angle, f_radial, vy, False
        return angle, f_radial, vy, True

    # ------------------------------------------------------------------ #
    #  Main control tick
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        if self._done:
            self._servo_cmd.publish_zero(self._state, self._force_z)
            return

        pose_row = self._lookup_pose()
        if pose_row is None:
            if not self._servo_cmd.publish_zero(self._state, self._force_z):
                self._done = True
            return

        t = self._now_s()
        px, py, pz, qx, qy, qz, qw = pose_row
        current_pitch = quat_to_pitch(qx, qy, qz, qw)
        arc_angle = (
            self._current_arc_angle(px, pz)
            if self._arc_center_x is not None
            else float("nan")
        )
        state_id = STATE_IDS.get(self._state, 0)
        self._pose_log.append([t, px, py, pz, qx, qy, qz, qw, arc_angle, current_pitch, state_id])

        contact_force = self._controlled_contact_force(px, pz) if self._have_force else 0.0
        if self._have_force and contact_force > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(
                f"Hard force limit exceeded: {contact_force:.2f} N "
                f"({self._state} controlled contact force) — retracting"
            )
            self._transition("RETRACT")

        # -------- SQUASH --------
        if self._state == "SQUASH":
            if self._check_timeout(SQUASH_TIMEOUT_SEC, "SQUASH"): return
            if self._have_force and self._force_z > 0.25 and not self._contact_felt:
                self._contact_felt = True
                self._force_y_ref = self._force_y  # capture lateral force at first contact
                self.get_logger().info(f"Contact felt! Y force ref set: {self._force_y_ref:.3f} N")
            if not self._servo_cmd.publish_twist(0.0, 0.0, -DESCEND_SPEED, state=self._state, force_z=self._force_z):
                self._done = True
                return
            if self._have_force and self._force_z >= self._force_ctrl.reference:
                self._contact_count += 1
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._transition("LULL")
            else:
                self._contact_count = 0
            return

        # -------- LULL --------
        if self._state == "LULL":
            if not self._servo_cmd.publish_zero(self._state, self._force_z):
                self._done = True
                return
            elapsed = self._now_s() - self._state_start_time
            if elapsed < LULL_WAIT_SEC:
                return

            # Previous behavior: use LULL as a force-settling phase.
            # Leaving this here intentionally in case we want to re-enable it.
            # normal_z = -self._force_ctrl.update(self._force_z)
            # if not self._servo_cmd.publish_twist(0.0, 0.0, normal_z, state=self._state, force_z=self._force_z):
            #     self._done = True
            #     return
            # force_error = abs(self._force_z - self._force_ctrl.reference)
            # settled = force_error < LULL_SETTLE_N
            # timed_out = elapsed > LULL_SETTLE_TIMEOUT_SEC
            # if not settled and not timed_out:
            #     return
            # if timed_out and not settled:
            #     self.get_logger().warn(
            #         f"LULL settle timeout — proceeding with force error {force_error:.2f} N"
            #     )

            if self._lull_next == "ARC":
                self._force_ctrl.reset()
                # Cap at the object's configured force_ref so we don't enter ARC near the hard limit
                pull_ref = min(self._force_z, self._force_ctrl.reference)
                self._force_ctrl.set_reference(pull_ref)
                self.get_logger().info(f"ARC setpoint: {pull_ref:.2f} N (measured={self._force_z:.2f} N)")
                # Reset Y PI state
                self._vy_integral = 0.0
                # Capture post-squash position and build arc geometry
                self._init_arc(px, py, pz)
            self._transition(self._lull_next)
            return

        # -------- ARC --------
        if self._state == "ARC":
            if self._check_timeout(ARC_TIMEOUT_SEC, "ARC"): return
            angle, f_radial, vy, ok = self._publish_arc_step(px, pz, current_pitch, ARC_TANGENTIAL_SPEED)
            if not ok: return

            angle_deg = math.degrees(angle)
            f_tangent = self._tangent_force(angle)
            if t - self._last_arc_log_time > 0.2:
                self._last_arc_log_time = t
                self.get_logger().info(
                    f"{angle_deg:.1f} / {ARC_MAX_ANGLE_DEG:.1f} deg  "
                    f"pitch(static): {math.degrees(current_pitch):.1f} deg  "
                    f"f_tangent: {f_tangent:.2f} N  f_radial: {f_radial:.2f} N  "
                    f"fx_low_count: {self._arc_fx_low_count}"
                )
            swept = abs(self._arc_start_angle - angle) if self._arc_start_angle is not None else 0.0
            self._arc_peak_tangent = max(self._arc_peak_tangent, abs(f_tangent))
            fx_low_thresh = max(ARC_FX_LOW_FLOOR_N, ARC_FX_LOW_FRACTION * self._arc_peak_tangent)
            if swept >= math.radians(ARC_FX_SIGN_MIN_SWEEP_DEG) and f_tangent < fx_low_thresh:
                self._arc_fx_low_count += 1
                if self._arc_fx_low_count >= ARC_FX_LOW_STABLE_SAMPLES:
                    self._lull_next = "UNARC"
                    self.get_logger().info(
                        f"tangent force below threshold ({fx_low_thresh:.2f} N, "
                        f"{ARC_FX_LOW_FRACTION:.0%} of peak {self._arc_peak_tangent:.2f} N) for "
                        f"{ARC_FX_LOW_STABLE_SAMPLES} ticks: f_tangent={f_tangent:.2f} N "
                        f"at arc_angle={math.degrees(angle):.1f} deg; entering LULL"
                    )
                    self._transition("LULL")
                    return
            else:
                # Only the consecutive-tick counter resets here; the running peak must
                # persist across the whole ARC or the relative threshold would collapse.
                self._arc_fx_low_count = 0

            if self._arc_fx_flipped(angle):
                self._lull_next = "UNARC"
                self.get_logger().info(
                    f"tangent force sign flip reached: f_tangent={self._tangent_force(angle):.2f} N "
                    f"at arc_angle={math.degrees(angle):.1f} deg; entering LULL"
                )
                self._transition("LULL")
                return
            if angle <= self._arc_end_angle:
                self._lull_next = "UNARC"
                self.get_logger().warn(
                    f"Reached max arc angle {ARC_MAX_ANGLE_DEG:.1f} deg before fx sign flip "
                    f"(f_tangent={self._tangent_force(angle):.2f} N); entering LULL as safety fallback"
                )
                self._transition("LULL")
            return

        # -------- UNARC --------
        if self._state == "UNARC":
            if self._check_timeout(UNARC_TIMEOUT_SEC, "UNARC"): return
            angle, f_radial, vy, ok = self._publish_arc_step(px, pz, current_pitch, -ARC_TANGENTIAL_SPEED)
            if not ok: return

            if t - self._last_arc_log_time > 0.2:
                self._last_arc_log_time = t
                f_tangent = self._tangent_force(angle)
                self.get_logger().info(
                    f"unarc {math.degrees(angle):.1f} / {math.degrees(self._arc_start_angle):.1f} deg  "
                    f"pitch(static): {math.degrees(current_pitch):.1f} deg  "
                    f"f_tangent: {f_tangent:.2f} N  f_radial: {f_radial:.2f} N"
                )

            if angle >= self._arc_start_angle - math.radians(1.0):  # 1 deg tolerance
                self._completed = True
                self._transition("RETRACT")
            return

        # -------- RETRACT --------
        if self._state == "RETRACT":
            elapsed = self._now_s() - self._state_start_time
            if elapsed < RETRACT_DURATION_SEC:
                if not self._servo_cmd.publish_twist(0.0, 0.0, RETRACT_SPEED, state=self._state, force_z=self._force_z):
                    self._done = True
                    return
            else:
                self._servo_cmd.publish_zero(self._state, self._force_z)
                self._done = True
            return

    def reset_attempt(self, force_ref: float) -> None:
        """Reset transient FSM/controller state before another full attempt."""
        self._state = "SQUASH"
        self._done = False
        self._completed = False
        self._retry_requested = False
        self._contact_count = 0
        self._contact_felt = False
        self._lost_contact_count = 0
        self._lull_next = "ARC"
        self._force_y_ref = None
        self._vy_integral = 0.0
        self._arc_center_x = None
        self._arc_center_z = None
        self._arc_start_angle = None
        self._arc_end_angle = None
        self._arc_fx_pos_count = 0
        self._arc_fx_neg_count = 0
        self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign = None
        self._arc_fx_low_count = 0
        self._arc_peak_tangent = 0.0
        self._force_ctrl.reset()
        self._force_ctrl.set_reference(force_ref)
        self._state_start_time = self._now_s()

    def run_attempt(self) -> None:
        """Run the FSM until its retract finishes."""
        timer = self.create_timer(1.0 / CONTROL_HZ, self._tick)
        try:
            while rclpy.ok() and not self._done:
                rclpy.spin_once(self, timeout_sec=0.05)
        finally:
            timer.cancel()
            self.destroy_timer(timer)
            self._servo_cmd.publish_zero(self._state, self._force_z)


def run_adaptive_press(node: "ArcStatic", force_ref: float, log_prefix: str = "Attempt") -> tuple[bool, float, int]:
    """Run one standard arc attempt, escalating the squash force by
    ADAPTIVE_FORCE_SCALE_FACTOR and retrying on each lost-contact slip, up to
    ADAPTIVE_FORCE_MAX_N. Stops as soon as an attempt completes cleanly, no
    retry was requested (some other kind of failure), or the ceiling would be
    exceeded.

    Shared by arc_static's own single-run main() and arc_static_batch, so the
    two can't silently drift apart on retry semantics again (see the
    force-collapse bug arc_static_batch had before this was factored out).

    Returns (completed, force_ref_used, attempts) — force_ref_used is whatever
    force the last attempt actually ran at, so a caller doing repeated trials
    of the same object (e.g. arc_static_batch) can seed the NEXT trial's
    starting force from this instead of re-discovering it from scratch.
    """
    attempt = 1
    while rclpy.ok():
        node.get_logger().info(f"=== {log_prefix} {attempt}: press force {force_ref:.2f} N ===")
        node.reset_attempt(force_ref)
        node.run_attempt()

        if node._completed or not node._retry_requested:
            return node._completed, force_ref, attempt

        next_force = force_ref * ADAPTIVE_FORCE_SCALE_FACTOR
        if next_force > ADAPTIVE_FORCE_MAX_N:
            node.get_logger().error(
                f"Retry would require {next_force:.2f} N, above the "
                f"{ADAPTIVE_FORCE_MAX_N:.2f} N adaptive ceiling — aborting"
            )
            return False, force_ref, attempt

        force_ref = next_force
        attempt += 1
        node.pause_servo()
        node.get_logger().info(f"Returning to pre-squash for adaptive retry at {force_ref:.2f} N")
        if not node.move_to_pre_squash():
            node.get_logger().error("MoveIt return for adaptive retry failed — aborting")
            return False, force_ref, attempt
        node.resume_servo()
    return False, force_ref, attempt


def main(args=None) -> int:
    raw_args = list(sys.argv[1:] if args is None else args)
    ros_args_index = raw_args.index("--ros-args") if "--ros-args" in raw_args else len(raw_args)
    app_args = raw_args[:ros_args_index]
    ros_args = raw_args[ros_args_index:]

    parser = argparse.ArgumentParser(description="Run the adaptive arc-static press FSM")
    parser.add_argument("object", nargs="?", choices=sorted(VALID_OBJECTS))
    parser.add_argument(
        "--quality", choices=["h264", "lossless"], default="h264",
        help="Video quality for both cameras this run (default: h264 — everyday runs). "
             "Use 'lossless' for a deliberate one-off high-quality take, e.g. one hero "
             "trial per object for a paper/video.",
    )
    parser.add_argument(
        "--ignore-press-sanity", action="store_true",
        help="Continue even if the press-point sanity check fails (perception disagrees "
             "with the calibrated pre_squash pose, or nothing was detected at all). This "
             "check exists to catch a badly-positioned/undetected object before any "
             "motion — only skip it when you already know why it's failing.",
    )
    parsed = parser.parse_args(app_args)

    rclpy.init(args=ros_args)
    node = ArcStatic(object_name=parsed.object)
    # Defaults in case we abort (or an exception fires) before the attempt loop is reached —
    # the F/T subscriber is already live at this point, so there may be a log to save either way.
    attempt = 0
    force_ref = node._force_ctrl.reference
    try:
        if not tare_netft(node):
            return 1

        if not start_recording(node, node._log_subdir, quality=parsed.quality):
            node.get_logger().error("One or more cameras failed to start recording — aborting")
            return 1

        if not ensure_egm_active(node):
            return 1

        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error(
                "MoveIt Servo is not ready (/servo_node/delta_twist_cmds has no subscribers). "
                "Launch stack with start_servo:=true before running arc_static."
            )
            return 1

        if not check_press_point(node, node._pre_squash_pos, label=f"arc_static/{node._object}"):
            msg = (
                "Press-point sanity check failed — perception disagrees with the calibrated "
                "pre_squash pose (or no detection at all)."
            )
            if parsed.ignore_press_sanity:
                node.get_logger().warn(f"{msg} Continuing anyway (--ignore-press-sanity).")
            else:
                node.get_logger().error(f"{msg} Aborting before any motion.")
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
            f"force_deadband={FORCE_DEADBAND_N}N  "
            f"force_filter_alpha={FORCE_FILTER_ALPHA}  "
            f"max_normal_speed={MAX_NORMAL_SPEED}m/s  "
            f"descend_speed={DESCEND_SPEED}m/s  arc_max={ARC_MAX_ANGLE_DEG}deg  "
            f"static_wrist=true  radial_force_control=true  tangential_ramp={ARC_TANGENTIAL_RAMP_SEC:.1f}s  "
            f"unarc_force_augment={UNARC_FORCE_AUGMENT_SPEED:.4f}m/s softness={UNARC_FORCE_AUGMENT_SOFTNESS_N:.2f}N"
        )

        if not node._operator_confirm(
            "At pre-squash pose. Confirm clear contact conditions before descending."
        ):
            return 0

        node.resume_servo()
        _, force_ref, attempt = run_adaptive_press(node, node._force_ctrl.reference)
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording(node)
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_ft_pose_log(
                node._ft_transformed_log,
                node._pose_log,
                node._log_subdir,
                "arc_static",
                node._obj_pose_log,
                command_log=node._command_log,
                timestamp=ts,
            )
            save_run_metadata(
                node._log_subdir,
                "arc_static",
                ts,
                {
                    **module_constants(globals()),
                    "object": node._object,
                    "final_force_ref_n": force_ref,
                    "attempts": attempt,
                    "completed": node._completed,
                    "press_point_check": getattr(node, "_press_point_check_result", None),
                },
            )
        except Exception as exc:
            node.get_logger().error(f"Failed to save F/T+pose log: {exc}")
        node._servo_cmd.publish_zero(node._state, node._force_z)
        if rclpy.ok():
            node.pause_servo()
            node.get_logger().info(
                "Returning to home position (all joints zero) via MoveIt on the existing EGM session..."
            )
            if not node.move_to_home():
                node.get_logger().error(
                    "MoveIt return to home failed. EGM may have ended; "
                    "not retrying automatically to avoid an unsafe snap."
                )
        node._servo_cmd.close()
        deactivate_egm(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
