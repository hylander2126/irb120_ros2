#!/usr/bin/env python3
"""Adaptive press-and-pull for the monitor.

Identical to arc_static, but if the finger slips off the object at any point
during the ARC phase (lost-contact detected), the squash force reference is
bumped by FORCE_SCALE_FACTOR and the whole sequence retries from the
pre-squash pose.  Retries continue until either:
  - the UNARC finishes cleanly  →  done, or
  - force_ref would exceed FORCE_REF_MAX_N  →  abort.

No F/T or pose logs are saved.  Video recording still runs.
"""

import math
import sys

import rclpy
from geometry_msgs.msg import WrenchStamped
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.controllers.force_controller import PIDForceController
from irb120_control.controllers.moveit_single_shot import plan_and_execute_pose_goal
from irb120_control.controllers.servo_command_publisher import ServoCommandPublisher
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.motion_geometry import (
    arc_angle_xz,
    arc_velocity_xz,
    clamp,
    quat_to_pitch,
    radial_force_xz,
)
from irb120_control.util.runtime_log_dir import load_object_params, set_recorder_output_dir

OBJECT = "monitor"

BASE_FRAME  = "world"
SERVO_FRAME = "base_link"
EE_LINK     = "finger_ball_center"

# ── adaptive press parameters ──────────────────────────────────────────────────
FORCE_REF_INITIAL_N  = 5.0    # starting squash force (N)
FORCE_SCALE_FACTOR   = 1.25   # multiply force_ref by this on each slip
FORCE_REF_MAX_N      = 13.0   # ceiling — abort if we'd exceed this

# ── motion / force constants (mirror arc_static) ──────────────────────────────
FORCE_HARD_LIMIT_N         = 15.0
CONTACT_STABLE_SAMPLES     = 1
DESCEND_SPEED              = 0.005
ARC_TANGENTIAL_SPEED       = 0.008
ARC_TANGENTIAL_RAMP_SEC    = 2.0
ARC_MAX_ANGLE_DEG          = -23.0
ARC_CENTER                 = (0.61, 0.0, 0.0)
ARC_FX_SIGN_DEADBAND_N     = 0.08
ARC_FX_SIGN_MIN_SWEEP_DEG  = 5.0
ARC_FX_SIGN_MIN_SAMPLES    = 20
ARC_FX_FLIP_STABLE_SAMPLES = 5
ARC_FX_LOW_THRESH_N        = 0.5
ARC_FX_LOW_STABLE_SAMPLES  = 5

SQUASH_TIMEOUT_SEC  = 30.0
ARC_TIMEOUT_SEC     = 30.0
UNARC_TIMEOUT_SEC   = 30.0
LULL_WAIT_SEC       = 1.0
RETRACT_SPEED       = 0.008
RETRACT_DURATION_SEC = 3.0

KP_FORCE                    = 0.00035
KI_FORCE                    = 0.000005
KD_FORCE                    = 0.0
MAX_NORMAL_SPEED            = 0.006
FORCE_DEADBAND_N            = 0.25
FORCE_FILTER_ALPHA          = 0.12
FORCE_OUTPUT_SLEW_RATE      = 0.02
UNARC_FORCE_AUGMENT_SPEED   = 0.004
UNARC_FORCE_AUGMENT_SOFTNESS_N = 0.75

CONTROL_HZ = 100.0
REQUIRE_OPERATOR_CONFIRM = True

LOST_CONTACT_FORCE_THRESH_N = 0.3
LOST_CONTACT_STEPS          = 20


class AdaptivePress(Node):
    def __init__(self, force_ref: float) -> None:
        super().__init__("adaptive_press")

        params = load_object_params(OBJECT)
        ps = params["pre_squash"]
        self._pre_squash_pos = (ps["x"], ps["y"], ps["z"])
        self._pre_squash_ori = (ps["qx"], ps["qy"], ps["qz"], ps["qw"])

        self._tf_buffer         = Buffer()
        self._tf_listener       = TransformListener(self._tf_buffer, self)
        self._servo_cmd         = ServoCommandPublisher(self, "/servo_node/delta_twist_cmds", SERVO_FRAME)
        self._wrench_sub        = self.create_subscription(WrenchStamped, "/netft_data_transformed", self._on_wrench, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._pause_servo_client = self.create_client(SetBool, "/servo_node/pause_servo")
        self._timer = None

        self._force_ctrl = PIDForceController(
            kp=KP_FORCE, ki=KI_FORCE, kd=KD_FORCE,
            force_ref_n=force_ref,
            max_normal_speed=MAX_NORMAL_SPEED,
            control_hz=CONTROL_HZ,
            deadband_n=FORCE_DEADBAND_N,
            measurement_filter_alpha=FORCE_FILTER_ALPHA,
            output_slew_rate=FORCE_OUTPUT_SLEW_RATE,
        )

        self._state            = "SQUASH"
        self._done             = False
        self._completed        = False   # set True only when UNARC finishes cleanly
        self._contact_count    = 0
        self._force_x          = 0.0
        self._force_z          = 0.0
        self._force_z_signed   = 0.0
        self._force_y          = 0.0
        self._have_force       = False
        self._contact_felt     = False
        self._last_tf_warn_time = 0.0
        self._state_start_time  = 0.0
        self._lost_contact_count = 0
        self._lull_next        = "ARC"
        self._force_y_ref      = None
        self._vy_integral      = 0.0

        self._arc_center_x      = None
        self._arc_center_z      = None
        self._arc_start_angle   = None
        self._arc_end_angle     = None
        self._arc_fx_pos_count  = 0
        self._arc_fx_neg_count  = 0
        self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign = None
        self._arc_fx_low_count  = 0

        self._last_arc_log_time = 0.0

    # ── subscribers ────────────────────────────────────────────────────────────

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self._force_x        = msg.wrench.force.x
        self._force_y        = msg.wrench.force.y
        self._force_z_signed = msg.wrench.force.z
        self._force_z        = abs(self._force_z_signed)
        self._have_force     = True

    # ── servo pause/resume ─────────────────────────────────────────────────────

    def _set_servo_paused(self, paused: bool) -> None:
        if not self._pause_servo_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("pause_servo service not available")
            return
        future = self._pause_servo_client.call_async(SetBool.Request(data=paused))
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    def pause_servo(self)  -> None: self._set_servo_paused(True)
    def resume_servo(self) -> None: self._set_servo_paused(False)

    # ── MoveIt ─────────────────────────────────────────────────────────────────

    def move_to_pre_squash(self) -> bool:
        return plan_and_execute_pose_goal(
            self, self._move_group_client,
            target_position=self._pre_squash_pos,
            target_orientation=self._pre_squash_ori,
            velocity_scale=0.1, acceleration_scale=0.1,
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _lookup_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, EE_LINK, rclpy.time.Time())
        except TransformException as exc:
            self._warn_throttled(f"Waiting for TF {BASE_FRAME} -> {EE_LINK}: {exc}")
            return None
        t  = tf.transform.translation
        q  = tf.transform.rotation
        return t.x, t.y, t.z, q.x, q.y, q.z, q.w

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttled(self, msg: str, throttle_hz: float = 0.2) -> None:
        now = self._now_s()
        if now - self._last_tf_warn_time > 1.0 / throttle_hz:
            self._last_tf_warn_time = now
            self.get_logger().warn(msg)

    def _check_timeout(self, timeout_sec: float, label: str) -> bool:
        if self._now_s() - self._state_start_time > timeout_sec:
            self.get_logger().error(f"{label} timed out — retracting")
            self._transition("RETRACT")
            return True
        return False

    def _check_lost_contact(self, force: float | None = None) -> bool:
        val = force if force is not None else self._force_z
        if val < LOST_CONTACT_FORCE_THRESH_N:
            self._lost_contact_count += 1
            if self._lost_contact_count >= LOST_CONTACT_STEPS:
                arc_elapsed = self._now_s() - self._state_start_time
                self.get_logger().warn(
                    f"Slip detected at {arc_elapsed:.1f}s into ARC "
                    f"(force_ref={self._force_ctrl.reference:.2f}N) — will retry with higher force"
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
        end = self._now_s() + timeout_sec
        while rclpy.ok() and self._now_s() < end:
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
            return False
        return True

    # ── arc geometry ───────────────────────────────────────────────────────────

    def _init_arc(self, x: float, y: float, z: float) -> None:
        cx, _cy, cz = ARC_CENTER
        self._arc_center_x     = cx
        self._arc_center_z     = cz
        self._arc_start_angle  = arc_angle_xz(x, z, cx, cz)
        self._arc_end_angle    = math.radians(ARC_MAX_ANGLE_DEG)
        self._arc_fx_pos_count = self._arc_fx_neg_count = self._arc_fx_flip_count = 0
        self._arc_fx_majority_sign = None
        self._arc_fx_low_count = 0
        r = math.hypot(x - cx, z - cz)
        self.get_logger().info(
            f"Arc init: center=({cx:.3f}, {cz:.3f})  r={r:.3f}m  "
            f"start={math.degrees(self._arc_start_angle):.1f}deg  "
            f"force_ref={self._force_ctrl.reference:.2f}N"
        )

    def _current_arc_angle(self, x: float, z: float) -> float:
        return arc_angle_xz(x, z, self._arc_center_x, self._arc_center_z)

    def _radial_force(self, theta: float) -> float:
        return radial_force_xz(theta, self._force_x, self._force_z_signed)

    def _tangent_force(self, theta: float) -> float:
        return self._force_x * math.cos(theta) - self._force_z_signed * math.sin(theta)

    def _vy_force(self) -> float:
        if self._force_y_ref is None:
            return 0.0
        err = self._force_y_ref - self._force_y
        self._vy_integral += err / CONTROL_HZ
        return clamp(0.0 * err + 0.0 * self._vy_integral, 0.015)  # Y ctrl disabled for now

    def _arc_fx_flipped(self, angle: float) -> bool:
        if self._arc_start_angle is None:
            return False
        swept = abs(self._arc_start_angle - angle)
        if swept < math.radians(ARC_FX_SIGN_MIN_SWEEP_DEG):
            return False
        ft = self._tangent_force(angle)
        if abs(ft) < ARC_FX_SIGN_DEADBAND_N:
            self._arc_fx_flip_count = 0
            return False
        sign = 1 if ft > 0.0 else -1
        if sign > 0: self._arc_fx_pos_count += 1
        else:        self._arc_fx_neg_count += 1
        total = self._arc_fx_pos_count + self._arc_fx_neg_count
        if self._arc_fx_majority_sign is None and total >= ARC_FX_SIGN_MIN_SAMPLES:
            self._arc_fx_majority_sign = 1 if self._arc_fx_pos_count >= self._arc_fx_neg_count else -1
        if self._arc_fx_majority_sign is None or sign == self._arc_fx_majority_sign:
            self._arc_fx_flip_count = 0
            return False
        self._arc_fx_flip_count += 1
        return self._arc_fx_flip_count >= ARC_FX_FLIP_STABLE_SAMPLES

    def _publish_arc_step(self, x, z, _pitch, tangential_speed):
        angle   = self._current_arc_angle(x, z)
        f_rad   = self._radial_force(angle)
        if self._check_lost_contact(f_rad):
            return angle, f_rad, False
        radial_corr = -self._force_ctrl.update(f_rad)
        if self._state == "UNARC":
            deficit = max(0.0, self._force_ctrl.reference - f_rad)
            aug = deficit / (deficit + UNARC_FORCE_AUGMENT_SOFTNESS_N) if deficit > 0.0 else 0.0
            radial_corr = clamp(radial_corr - UNARC_FORCE_AUGMENT_SPEED * aug, MAX_NORMAL_SPEED)
        ramp = min(1.0, max(0.0, (self._now_s() - self._state_start_time) / ARC_TANGENTIAL_RAMP_SEC))
        vx, vz = arc_velocity_xz(angle, tangential_speed * ramp, radial_corr)
        vy = self._vy_force()
        if not self._servo_cmd.publish_twist(vx, vy, vz, 0.0, self._state, f_rad):
            self._done = True
            return angle, f_rad, False
        return angle, f_rad, True

    # ── main tick ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if self._done:
            self._servo_cmd.publish_zero(self._state, self._force_z)
            return

        pose = self._lookup_pose()
        if pose is None:
            if not self._servo_cmd.publish_zero(self._state, self._force_z):
                self._done = True
            return

        t = self._now_s()
        px, py, pz, qx, qy, qz, qw = pose
        current_pitch = quat_to_pitch(qx, qy, qz, qw)

        contact_force = (
            self._radial_force(self._current_arc_angle(px, pz))
            if self._arc_center_x is not None and self._have_force
            else self._force_z
        )
        if self._have_force and contact_force > FORCE_HARD_LIMIT_N and self._state != "RETRACT":
            self.get_logger().error(f"Hard limit {contact_force:.2f}N — retracting")
            self._transition("RETRACT")

        # ── SQUASH ──
        if self._state == "SQUASH":
            if self._check_timeout(SQUASH_TIMEOUT_SEC, "SQUASH"): return
            if self._have_force and self._force_z > 0.25 and not self._contact_felt:
                self._contact_felt = True
                self._force_y_ref  = self._force_y
            if not self._servo_cmd.publish_twist(0.0, 0.0, -DESCEND_SPEED, state=self._state, force_z=self._force_z):
                self._done = True; return
            if self._have_force and self._force_z >= self._force_ctrl.reference:
                self._contact_count += 1
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._transition("LULL")
            else:
                self._contact_count = 0
            return

        # ── LULL ──
        if self._state == "LULL":
            if not self._servo_cmd.publish_zero(self._state, self._force_z):
                self._done = True; return
            if self._now_s() - self._state_start_time < LULL_WAIT_SEC:
                return
            if self._lull_next == "ARC":
                self._force_ctrl.reset()
                pull_ref = min(self._force_z, self._force_ctrl.reference)
                self._force_ctrl.set_reference(pull_ref)
                self._vy_integral = 0.0
                self._init_arc(px, py, pz)
            self._transition(self._lull_next)
            return

        # ── ARC ──
        if self._state == "ARC":
            if self._check_timeout(ARC_TIMEOUT_SEC, "ARC"): return
            angle, f_rad, ok = self._publish_arc_step(px, pz, current_pitch, ARC_TANGENTIAL_SPEED)
            if not ok: return

            f_tangent = self._tangent_force(angle)
            if t - self._last_arc_log_time > 0.2:
                self._last_arc_log_time = t
                self.get_logger().info(
                    f"ARC {math.degrees(angle):.1f}/{ARC_MAX_ANGLE_DEG:.1f}deg  "
                    f"f_tan={f_tangent:.2f}N  f_rad={f_rad:.2f}N  "
                    f"force_ref={self._force_ctrl.reference:.2f}N  "
                    f"fx_low={self._arc_fx_low_count}"
                )

            swept = abs(self._arc_start_angle - angle) if self._arc_start_angle is not None else 0.0
            if swept >= math.radians(ARC_FX_SIGN_MIN_SWEEP_DEG) and f_tangent < ARC_FX_LOW_THRESH_N:
                self._arc_fx_low_count += 1
                if self._arc_fx_low_count >= ARC_FX_LOW_STABLE_SAMPLES:
                    self.get_logger().info(
                        f"Tangent force < {ARC_FX_LOW_THRESH_N:.2f}N for {ARC_FX_LOW_STABLE_SAMPLES} ticks "
                        f"at {math.degrees(angle):.1f}deg — entering LULL→UNARC"
                    )
                    self._lull_next = "UNARC"
                    self._transition("LULL")
                    return
            else:
                self._arc_fx_low_count = 0

            if self._arc_fx_flipped(angle):
                self.get_logger().info(f"Tangent sign flip at {math.degrees(angle):.1f}deg — LULL→UNARC")
                self._lull_next = "UNARC"
                self._transition("LULL")
                return
            if angle <= self._arc_end_angle:
                self.get_logger().warn(f"Max angle reached — LULL→UNARC")
                self._lull_next = "UNARC"
                self._transition("LULL")
            return

        # ── UNARC ──
        if self._state == "UNARC":
            if self._check_timeout(UNARC_TIMEOUT_SEC, "UNARC"): return
            angle, f_rad, ok = self._publish_arc_step(px, pz, current_pitch, -ARC_TANGENTIAL_SPEED)
            if not ok: return
            if t - self._last_arc_log_time > 0.2:
                self._last_arc_log_time = t
                self.get_logger().info(
                    f"UNARC {math.degrees(angle):.1f}/{math.degrees(self._arc_start_angle):.1f}deg  "
                    f"f_rad={f_rad:.2f}N"
                )
            if angle >= self._arc_start_angle - math.radians(1.0):
                self._completed = True
                self._transition("RETRACT")
            return

        # ── RETRACT ──
        if self._state == "RETRACT":
            elapsed = self._now_s() - self._state_start_time
            if elapsed < RETRACT_DURATION_SEC:
                if not self._servo_cmd.publish_twist(0.0, 0.0, RETRACT_SPEED, state=self._state, force_z=self._force_z):
                    self._done = True; return
            else:
                self._servo_cmd.publish_zero(self._state, self._force_z)
                self._done = True
            return


def _run_attempt(node: AdaptivePress) -> None:
    """Run one full squash→arc→unarc→retract attempt on an existing node."""
    node._state            = "SQUASH"
    node._done             = False
    node._completed        = False
    node._contact_count    = 0
    node._contact_felt     = False
    node._lost_contact_count = 0
    node._lull_next        = "ARC"
    node._force_y_ref      = None
    node._vy_integral      = 0.0
    node._arc_center_x     = None
    node._arc_center_z     = None
    node._arc_start_angle  = None
    node._arc_fx_low_count = 0
    node._force_ctrl.reset()
    node._state_start_time = node._now_s()

    timer = node.create_timer(1.0 / CONTROL_HZ, node._tick)
    while rclpy.ok() and not node._done:
        rclpy.spin_once(node, timeout_sec=0.05)
    timer.cancel()
    node._servo_cmd.publish_zero(node._state, node._force_z)


def main(args=None) -> int:
    rclpy.init(args=args)

    force_ref = FORCE_REF_INITIAL_N
    node = AdaptivePress(force_ref)

    recorder_client = node.create_client(SetBool, "/camera_hull_recorder/set_recording")

    try:
        if not tare_netft(node):
            return 1

        set_recorder_output_dir(node, "monitor/adaptive_press")
        if recorder_client.wait_for_service(timeout_sec=5.0):
            future = recorder_client.call_async(SetBool.Request(data=True))
            rclpy.spin_until_future_complete(node, future)
            result = future.result()
            if result is None or not result.success:
                node.get_logger().error(f"Start-recording failed — aborting")
                return 1
            node.get_logger().info("Recording started")
        else:
            node.get_logger().error("Recorder service unavailable — aborting")
            return 1

        if not ensure_egm_active(node):
            return 1

        if not node._wait_for_servo_ready(timeout_sec=5.0):
            node.get_logger().error("MoveIt Servo not ready — aborting")
            return 1

        if not node.move_to_pre_squash():
            node.get_logger().error("Approach failed — aborting")
            return 1

        if not node._operator_confirm(
            "At pre-squash pose. Confirm clear contact conditions before descending."
        ):
            return 0

        node.resume_servo()

        attempt = 1
        while rclpy.ok():
            node.get_logger().info(
                f"=== Attempt {attempt}  force_ref={force_ref:.3f}N ==="
            )
            node._force_ctrl.set_reference(force_ref)
            _run_attempt(node)

            if not rclpy.ok():
                break

            if node._completed:
                node.get_logger().info("ARC completed without slip — done.")
                break

            # Slip occurred — bump force and retry
            next_force = force_ref * FORCE_SCALE_FACTOR
            if next_force > FORCE_REF_MAX_N:
                node.get_logger().error(
                    f"Next force {next_force:.2f}N would exceed ceiling {FORCE_REF_MAX_N:.1f}N — aborting"
                )
                break

            force_ref = next_force
            attempt  += 1
            node.get_logger().info(f"Returning to pre-squash for retry with force_ref={force_ref:.3f}N")
            node.pause_servo()
            if not node.move_to_pre_squash():
                node.get_logger().error("Return to pre-squash failed — aborting")
                break
            node.resume_servo()

    except KeyboardInterrupt:
        pass
    finally:
        node.pause_servo()
        if recorder_client.wait_for_service(timeout_sec=2.0) and rclpy.ok():
            future = recorder_client.call_async(SetBool.Request(data=False))
            rclpy.spin_until_future_complete(node, future)
        node._servo_cmd.publish_zero(node._state, node._force_z)
        node._servo_cmd.close()
        deactivate_egm(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
