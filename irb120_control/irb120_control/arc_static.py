#!/usr/bin/env python3
"""Static-wrist force-informed arc-following press-pull tip for the IRB120.

Squash phase: descend until the press force reference is met.
Arc-follow phase: EE follows a circular arc whose:
  - center = the detected tipping pivot (press contact's pivot, world frame)
  - radius = the post-squash EE distance from that pivot

At each tick the desired arc position is computed from the measured EE angle.
The finger orientation is held fixed while a PI force controller adjusts the
radial component computed in the actual XZ arc frame.

UNARC reverses the arc back to the squash angle.  If contact is lost during
ARC or UNARC, the complete attempt is re-run with a higher press force.

Runs as the 'press_pull_tip' stage of an episode (util/episode.py). The
targets come from the episode's perception snapshot (taken first unless the
last stage already is one): the approach is SQUASH_STANDOFF above the detected
press contact's finger-ball center, and the arc center is that contact's
tipping pivot. After returning home, a second snapshot is taken.

Usage:
    ros2 run irb120_control arc_static flashlight                   # new episode
    ros2 run irb120_control arc_static flashlight --episode last    # e.g. after push
"""

import argparse
import json
import math
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from rclpy.node import Node
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from irb120_control.controllers.force_controller import PIDForceController
from irb120_perception.press_targets import press_targets, PRESS_STANDOFF
from irb120_control.controllers.moveit_single_shot import plan_and_execute_joint_goal, plan_and_execute_pose_goal
from irb120_control.controllers.servo_command_publisher import ServoCommandPublisher
from irb120_control.util.egm_client import ensure_egm_active, deactivate_egm
from irb120_control.util.episode import Episode
from irb120_control.util.ft_tare import tare_netft
from irb120_control.util.ft_capture import FTCapture, stamp_s
from irb120_control.util.ft_frames import ESTIMATOR_POSE_FRAME
from irb120_control.util.object_obstacle import object_obstacle
from irb120_control.util.perception_snapshot import begin_motion_stage, latest_object_cloud, lookup, take_snapshot
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
SQUASH_EXTRA_DESCENT_M = 0.005  # allow 5 mm below predicted ball-center contact height
SQUASH_STANDOFF = PRESS_STANDOFF       # m, pre-squash ball center sits this far above the detected press contact (~14 mm clear of the object's collision box)
PRE_SQUASH_ORIENTATION = (0.0, 0.0, 0.0, 1.0)  # EE orientation for the approach (xyzw)
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
UNARC_FORCE_AUGMENT_SPEED = 0.0  # disabled: bypassed PID slew limiting and caused return-arc hunting
UNARC_FORCE_AUGMENT_SOFTNESS_N = 0.75  # larger = smoother/slower augmentation onset

KP_Y_FORCE = 0#.0025      # m/s per N of Y force error
KI_Y_FORCE = 0#.0005      # m/s per N·s of accumulated Y force error
MAX_Y_SPEED = 0.015     # m/s clamp on Y correction output

CONTROL_HZ = 100.0
# The FSM acts on the latest /netft_data_transformed sample. If that stream stops (the
# preprocessor withholds output when its gravity-comp TF is older than 0.5 s, or the
# sensor drops out), the last value would otherwise be reused forever: SQUASH would keep
# descending on a stale "no contact" reading, ARC would regulate against a frozen force.
# At ~470 Hz, 0.1 s is ~47 missing samples and 10 control ticks.
FT_STALE_SEC = 0.1
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
    def __init__(self, object_name: str, force_ref_override: float | None = None) -> None:
        super().__init__("arc_static")
        obj = object_name
        self._object = obj
        try:
            params = load_object_params(obj)
        except ValueError:
            if force_ref_override is None:
                raise ValueError(
                    f"Object '{obj}' has no model in irb120_models "
                    f"(known: {sorted(VALID_OBJECTS)}). Add objects/<name>/<name>.yaml there, or "
                    "pass --force-ref N for an ad hoc test object."
                ) from None
            params = {}
        # Both set from the detected press contact by set_targets_from_contacts().
        self._pre_squash_pos = None  # world frame
        self._arc_center = None      # world frame (x, y, z)
        self._object_xyz = None      # detected object cloud (base_link), avoided on the approach

        self._tf_buffer       = Buffer()
        self._tf_listener     = TransformListener(self._tf_buffer, self)
        self._servo_cmd       = ServoCommandPublisher(self, "/servo_node/delta_twist_cmds", SERVO_FRAME)
        self._wrench_sub      = self.create_subscription(WrenchStamped, "/netft_data_transformed", self._on_wrench, 10)
        self._det_sub         = self.create_subscription(Detection3DArray, "/object_detector/detections", self._on_detection, 10)
        self._move_group_client = ActionClient(self, MoveGroup, "/move_action")
        self._timer = None

        if force_ref_override is not None:
            force_ref = float(force_ref_override)
        elif "force_ref_n" not in params:
            raise ValueError(f"Object '{obj}' has no 'force_ref_n' in irb120_control/config/object_control.yaml")
        else:
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
        self._last_wrench_s: float | None = None  # node-clock receipt time of the latest sample
        self._abort_reason: str | None = None
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

        self._tare_events: list = []
        self._ft_capture = FTCapture(self, self._now_s)
        self._pose_tf_stamp_s = float("nan")
        self._ft_transformed_log: list = []
        self._pose_log: list = []     # rows: [time_s, x, y, z, qx, qy, qz, qw, arc_angle_rad, wrist_pitch_rad, state_id]
        self._obj_pose_log: list = [] # rows: [time_s, x, y, z, qx, qy, qz, qw, obj_pitch_rad]
        self._command_log: list = []  # rows: [time_s, state_id, angle, f_radial, radial_corr, vx, vy, vz, wy, tangential_speed]
        self._last_wrench_log_time = 0.0
        self._last_arc_log_time = 0.0
        self._tracking_log = []
        self._force_diagnostics = []
        self._motion_phase = 'approach'
        self._joint_tracking_sub = self.create_subscription(
            JointState, '/joint_states', self._record_joints, 10)
        self._controller_tracking_sub = self.create_subscription(
            JointTrajectoryControllerState, '/joint_trajectory_controller/controller_state',
            self._record_controller, 10)

    def _record_joints(self, msg):
        self._tracking_log.append(dict(time=self._now_s(), source='joint_states',
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            phase=self._motion_phase, state=self._state, names=list(msg.name),
            positions=list(msg.position), velocities=list(msg.velocity)))

    def _record_controller(self, msg):
        row = dict(time=self._now_s(), source='controller_state',
            stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            phase=self._motion_phase, state=self._state, names=list(msg.joint_names))
        for name in ('reference', 'feedback', 'error', 'output'):
            value = getattr(msg, name)
            row[name] = dict(positions=list(value.positions), velocities=list(value.velocities))
        self._tracking_log.append(row)

    # ------------------------------------------------------------------ #
    #  Servo pause / resume
    # ------------------------------------------------------------------ #

    def _set_servo_paused(self, paused: bool) -> bool:
        if not self._pause_servo_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("pause_servo service not available")
            return False
        future = self._pause_servo_client.call_async(SetBool.Request(data=paused))
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done() and future.result() is not None:
            state = "paused" if paused else "resumed"
            self.get_logger().info(f"Servo {state}: {future.result().message}")
            return bool(future.result().success)
        else:
            self.get_logger().warn("pause_servo call did not complete")
            return False

    def pause_servo(self) -> bool:
        return self._set_servo_paused(True)

    def resume_servo(self) -> None:
        self._set_servo_paused(False)

    # ------------------------------------------------------------------ #
    #  MoveIt approach
    # ------------------------------------------------------------------ #

    def move_to_pre_squash(self) -> bool:
        """Plan around the detected object (a collision box only for this move). Also used for
        retries and between batch trials, from the RETRACT pose ~24 mm above the object."""
        diagnostic = None
        action = {}
        success = False
        if getattr(self, '_approach_diagnostic_path', None):
            from irb120_control.util.approach_diagnostics import ApproachDiagnostics
            diagnostic = ApproachDiagnostics(self, self._approach_diagnostic_path,
                                             self._pre_squash_pos, BASE_FRAME, EE_LINK)
            diagnostic.write('approach_request', target=self._pre_squash_pos,
                             orientation_xyzw=PRE_SQUASH_ORIENTATION, frame=BASE_FRAME,
                             link=EE_LINK, provenance=self._approach_provenance)
            diagnostic.start()
        try:
            with object_obstacle(self, self._object_xyz):
                success = plan_and_execute_pose_goal(
                    self, self._move_group_client,
                    target_position=self._pre_squash_pos,
                    target_orientation=PRE_SQUASH_ORIENTATION,
                    base_frame=BASE_FRAME, ee_link=EE_LINK,
                    velocity_scale=0.1, acceleration_scale=0.1, diagnostics=action)
            return success
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            action['exception'] = str(exc)
            return False
        finally:
            if diagnostic is not None:
                diagnostic.finish(success, action)

    def set_targets_from_contacts(self, contacts: dict, object_xyz) -> bool:
        """Pre-squash position and arc center from the detected press contact
        (selector output is in base_link; this node works in world), and the
        object cloud to plan the approach around."""
        self._object_xyz = object_xyz
        press = contacts["press"]
        pivot = (press.get("geometry") or {}).get("pivot")
        if not press["available"] or pivot is None:
            self.get_logger().error(f"No press contact/pivot: {press.get('reason')} -- not moving.")
            return False
        tf = lookup(self, BASE_FRAME, "base_link")
        if tf is None:
            self.get_logger().error(f"No TF {BASE_FRAME} <- base_link -- not moving.")
            return False
        R, t = tf
        targets = press_targets(press, R, t)
        self._approach_provenance = dict(saved_press=press,
            world_from_base_link=dict(R=R.tolist(), t=t.tolist()),
            standoff_m=SQUASH_STANDOFF,
            contacts_file=getattr(self, '_approach_contacts_file', None),
            artifact_sha256=getattr(self, '_approach_artifact_hashes', {}))
        self._pre_squash_pos = tuple(float(v) for v in targets["pre_press"])
        self._arc_center = tuple(float(v) for v in targets["pivot"])
        # Display the exact transformed targets supplied to MoveIt, not a new selection.
        from visualization_msgs.msg import MarkerArray
        from irb120_perception.press_targets import build_press_markers
        self._target_marker_pub = self.create_publisher(
            MarkerArray, '/contact_point_selector/markers', 1)
        def publish_targets():
            self._target_marker_pub.publish(build_press_markers(
                targets, BASE_FRAME, self.get_clock().now().to_msg(), 'Arc command targets'))
        self._target_marker_timer = self.create_timer(.5, publish_targets)
        publish_targets()
        self.get_logger().info(f"Press contact -> pre-squash {self._pre_squash_pos}, arc center {self._arc_center}")
        return True

    def move_to_home(self) -> bool:
        """Return to the robot's all-zero joint configuration, gently — same
        velocity/acceleration scale as the pre-squash approach — planning around
        the object (back at rest after the tip; RETRACT leaves the finger above its box)."""
        try:
            with object_obstacle(self, self._object_xyz):
                return plan_and_execute_joint_goal(
                    self,
                    self._move_group_client,
                    joint_positions=HOME_JOINT_POSITIONS,
                    velocity_scaling_factor=0.1,
                    acceleration_scaling_factor=0.1,
                )
        except RuntimeError as exc:
            self.get_logger().error(str(exc))
            return False

    # ------------------------------------------------------------------ #
    #  Subscribers
    # ------------------------------------------------------------------ #

    def tare_with_metadata(self) -> bool:
        """Record the latest unloaded mean and tare timing without changing tare_netft."""
        requested = self._now_s()
        samples = [row[1:7] for row in self._ft_transformed_log
                   if 0.0 <= requested - row[0] <= 0.3]
        event = {
            "requested_time_s": requested,
            "pre_tare_mean_wrench": np.mean(samples, axis=0).tolist() if samples else None,
            "pre_tare_sample_count": len(samples),
            "wrench_order": "fx fy fz tx ty tz; N and N*m",
        }
        self._tare_events.append(event)
        success = tare_netft(self)
        event.update(completed_time_s=self._now_s(), success=success)
        return success

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # Keep the signed fixed-orientation XZ force for actual-angle radial/tangent projections.
        self._force_x = msg.wrench.force.x
        self._force_y = msg.wrench.force.y
        self._force_z_signed = msg.wrench.force.z
        self._force_z = abs(self._force_z_signed)
        self._have_force = True
        t = self._now_s()
        self._last_wrench_s = t
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
        self._ft_transformed_log.append([
            t,
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ft_px, ft_py, ft_pz, ft_qx, ft_qy, ft_qz, ft_qw,
            stamp_s(msg.header.stamp), tf_stamp,
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
        self._pose_tf_stamp_s = stamp_s(transform.header.stamp)
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
            self.get_logger().warn("Operator aborted arc press-pull sequence")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Arc geometry helpers
    # ------------------------------------------------------------------ #

    def _init_arc(self, x_contact: float, y_contact: float, z_contact: float) -> None:
        """Compute arc parameters from the post-squash EE position.

        Arc lives in the XZ plane around the detected pivot (self._arc_center, world frame).
        Radius is the current EE distance from that fixed pivot edge.

        The start angle is the angle of the EE from the center, measured
        from the +Z axis toward +X:  theta = atan2(dx, dz)
        """
        center_x, center_y, center_z = self._arc_center
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
        self._force_diagnostics.append(dict(time=self._now_s(), state=self._state,
            radial_force=f_radial, **self._force_ctrl.diagnostics))
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

    def _wrench_stale(self) -> bool:
        return self._last_wrench_s is None or self._now_s() - self._last_wrench_s > FT_STALE_SEC

    def _tick(self) -> None:
        if self._done:
            self._servo_cmd.publish_zero(self._state, self._force_z)
            return

        # RETRACT is open-loop upward motion and needs no force, so it is the safe exit.
        if self._state != "RETRACT" and self._wrench_stale():
            age = (float("inf") if self._last_wrench_s is None
                   else self._now_s() - self._last_wrench_s)
            self.get_logger().error(
                f"F/T stream stale in {self._state}: no wrench for {age:.2f} s "
                f"(limit {FT_STALE_SEC:.2f} s) -- retracting without retry")
            self._abort_reason = "ft_stale"
            self._retry_requested = False
            self._servo_cmd.publish_zero(self._state, self._force_z)
            self._transition("RETRACT")
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
        self._pose_log.append([t, px, py, pz, qx, qy, qz, qw, arc_angle, current_pitch, state_id,
                               self._pose_tf_stamp_s])

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
            contact_z = self._pre_squash_pos[2] - SQUASH_STANDOFF
            if (pz <= contact_z - SQUASH_EXTRA_DESCENT_M
                    and not (self._have_force and self._force_z >= self._force_ctrl.reference)):
                self.get_logger().error(
                    f"No expected press force within {SQUASH_EXTRA_DESCENT_M*1000:.0f} mm "
                    "below predicted contact height -- retracting without retry")
                self._retry_requested = False
                self._servo_cmd.publish_zero(self._state, self._force_z)
                self._transition("RETRACT")
                return
            if self._have_force and self._force_z >= self._force_ctrl.reference:
                self._contact_count += 1
                self._servo_cmd.publish_zero(self._state, self._force_z)
                if self._contact_count >= CONTACT_STABLE_SAMPLES:
                    self._transition("LULL")
            else:
                self._contact_count = 0
                if not self._servo_cmd.publish_twist(
                        0.0, 0.0, -DESCEND_SPEED, state=self._state, force_z=self._force_z):
                    self._done = True
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
        self._abort_reason = None
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
        self._motion_phase = 'interaction'
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
    parser = argparse.ArgumentParser(description="Press-pull tip at the detected press contact")
    parser.add_argument(
        "object",
        help=f"Object name. Registered ({sorted(VALID_OBJECTS)}) objects pick up their "
             "force_ref_n from config/object_control.yaml automatically; any other "
             "snapshot/episode label works too if you pass --force-ref.",
    )
    parser.add_argument(
        "--force-ref", type=float, default=None, metavar="N",
        help="Press force target in Newtons. Required for an object without a force_ref_n in "
             "config/object_control.yaml; overrides the registered value otherwise.",
    )
    parser.add_argument(
        "--quality", choices=["h264", "lossless"], default="h264",
        help="Video quality this run (default: h264 — everyday runs). Use 'lossless' for a "
             "deliberate one-off high-quality take, e.g. one hero trial per object for a paper/video.",
    )
    parser.add_argument("--episode", default="new", metavar="new|last|PATH",
                        help="Episode to record into (default: a new one). See util/episode.py.")
    parser.add_argument(
        "--pivot", choices=["cloud", "rgb"], default="cloud",
        help="Pivot variant from the perception stage (default: cloud). 'rgb' requires a "
             "reviewable contacts_rgb_pivot.json and changes only the arc center.",
    )
    parser.add_argument("--tare-at-start", action="store_true",
                        help="Legacy protocol: tare once before the initial approach.")
    parsed = parser.parse_args(raw_args[:ros_args_index])

    rclpy.init(args=raw_args[ros_args_index:])
    node = ArcStatic(parsed.object, force_ref_override=parsed.force_ref)
    episode = Episode.resolve(parsed.episode, parsed.object, node,
                              allow_unconfigured_object=parsed.force_ref is not None)
    stage = None
    attempt = 0
    force_ref = node._force_ctrl.reference
    motion_started = False
    home_ok = None
    try:
        contacts_filename = "contacts_rgb_pivot.json" if parsed.pivot == "rgb" else "contacts.json"
        stage, contacts = begin_motion_stage(
            node, episode, "press_pull_tip", contacts_filename=contacts_filename)
        if stage is None:
            node.get_logger().error(
                f"No usable {parsed.pivot} perception contacts -- not moving. "
                "For RGB, review the preceding pivot_boundary_review.json first.")
            return 1
        node.get_logger().info(f"Using {parsed.pivot} pivot from {contacts_filename}")
        perceive_stage = next(s for s in reversed(episode.load()['stages'])
                              if s['kind'] == 'perceive' and s['status'] == 'ok')
        node._approach_contacts_file = str(episode.path / perceive_stage['dir'] / contacts_filename)
        import hashlib
        snapshot_dir = episode.path / perceive_stage['dir']
        node._approach_artifact_hashes = {
            name: hashlib.sha256((snapshot_dir / name).read_bytes()).hexdigest()
            for name in (contacts_filename, 'final_selection.jpg', 'cam1_camera_info.json',
                         'cam2_camera_info.json', 'cam3_camera_info.json')
            if (snapshot_dir / name).exists()}
        node._approach_diagnostic_path = stage.path / 'approach_diagnostics.jsonl'
        if not node.set_targets_from_contacts(contacts, latest_object_cloud(episode)):
            return 1

        if parsed.tare_at_start and not node.tare_with_metadata():
            return 1

        if not start_recording(node, str(stage.path), quality=parsed.quality):
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

        motion_started = True
        if not node.move_to_pre_squash():
            node.get_logger().error("Approach failed. Aborting.")
            return 1

        if not parsed.tare_at_start:
            if not node.pause_servo() or not node.tare_with_metadata():
                return 1

        node.get_logger().info(
            f"[PARAMS] object={node._object}  "
            f"force_ref={node._force_ctrl.reference:.2f}N  "
            f"hard_limit={FORCE_HARD_LIMIT_N:.1f}N  "
            f"pre_squash_pos={node._pre_squash_pos}  arc_center={node._arc_center}  "
            f"kp={KP_FORCE}  ki={KI_FORCE}  "
            f"force_deadband={FORCE_DEADBAND_N}N  "
            f"force_filter_alpha={FORCE_FILTER_ALPHA}  "
            f"max_normal_speed={MAX_NORMAL_SPEED}m/s  "
            f"descend_speed={DESCEND_SPEED}m/s  arc_max={ARC_MAX_ANGLE_DEG}deg  "
            f"static_wrist=true  radial_force_control=true  tangential_ramp={ARC_TANGENTIAL_RAMP_SEC:.1f}s  "
            f"unarc_force_augment={UNARC_FORCE_AUGMENT_SPEED:.4f}m/s softness={UNARC_FORCE_AUGMENT_SOFTNESS_N:.2f}N"
        )

        if not node._operator_confirm(
            "Approach action finished. Check the actual finger against the preview before descending."
        ):
            return 0

        node.resume_servo()
        _, force_ref, attempt = run_adaptive_press(node, node._force_ctrl.reference)
    except KeyboardInterrupt:
        pass
    finally:
        node._servo_cmd.publish_zero(node._state, node._force_z)
        if motion_started and rclpy.ok():
            node._motion_phase = 'home'
            if node.pause_servo():
                node.get_logger().info(
                    "Returning home; action success will be checked against measured joints."
                )
                home_ok = node.move_to_home()
            else:
                home_ok = False
                node.get_logger().error("Servo pause unconfirmed; home motion not requested.")
            if not home_ok:
                node.get_logger().error(
                    "Home not verified; skipping perception. No automatic EGM restart or motion retry."
                )
        stop_recording(node)
        if stage is not None:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_ft_pose_log(
                    node._ft_transformed_log,
                    node._pose_log,
                    str(stage.path),
                    "arc_static",
                    node._obj_pose_log,
                    ft_raw_log=node._ft_capture.raw,
                    command_log=node._command_log,
                    timestamp=ts,
                    commanded_pivot=node._arc_center,
                )
                (Path(stage.path) / f"arc_diagnostics_{ts}.json").write_text(
                    json.dumps(dict(tracking=node._tracking_log, force_controller=node._force_diagnostics),
                               separators=(',', ':')) + '\n')
                save_run_metadata(str(stage.path), "arc_static", ts, {
                    **module_constants(globals()),
                    **node._ft_capture.metadata(),
                    "object": node._object,
                    "pre_squash": node._pre_squash_pos,
                    "arc_center": node._arc_center,
                    "tare_at_start": parsed.tare_at_start,
                    "tare_events": node._tare_events,
                    "final_force_ref_n": force_ref,
                    "attempts": attempt,
                    "completed": node._completed,
                    "first_attempt_completed": node._completed and attempt == 1,
                    "abort_reason": node._abort_reason,
                    "home_ok": home_ok,
                    "home_verification": getattr(node, "_last_joint_goal_verification", None),
                })
            except Exception as exc:
                node.get_logger().error(f"Failed to save F/T+pose log: {exc}")
        node._servo_cmd.close()
        deactivate_egm(node)
        if stage is not None:
            stage.end("ok" if node._completed and home_ok else "failed" if motion_started else "aborted",
                      completed=node._completed, first_attempt_completed=node._completed and attempt == 1,
                      attempts=attempt, final_force_ref_n=force_ref, home_ok=home_ok,
                      home_verification=getattr(node, "_last_joint_goal_verification", None),
                      pre_squash=node._pre_squash_pos, arc_center=node._arc_center)
            if home_ok and rclpy.ok():
                take_snapshot(node, episode, label="after tip")
        print(episode.summary())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
