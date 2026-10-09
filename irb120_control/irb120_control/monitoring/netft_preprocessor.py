#!/usr/bin/env python3
"""Preprocess NetFT wrench data in ROS 2.

Pipeline (per sample, hot path):
  1. Sanity-check for NaN/Inf.
  2. Subtract calibrated electrical zero (frame-fixed, from FT calibration node).
  3. Subtract calibrated gravity-aligned load (rotated into current sensor frame).
  4. Optional: subtract session tare (small residual offset captured at runtime).
  5. Apply EMA low-pass filter.
  6. Publish.

Calibration constants
---------------------
ELECTRICAL_ZERO and G_WORLD are produced by `calibrate_ft_sensor` (3-pose solve).
They are baked-in constants here — re-run calibration and paste new values when:
  - the finger / distal hardware changes,
  - the FT sensor is remounted,
  - the cell's electrical zero drifts significantly (rare, weeks/months).

Session tare (/netft_preprocessor/set_tare, SetBool)
----------------------------------------------------
  data=true  — at the current pose with no contact, capture the small residual
               offset left after the calibrated comp. Useful at session start
               to absorb drift since the last calibration. Robot must be
               stationary and unloaded.
  data=false — clear the session tare (calibration constants remain).
"""

import json
import hashlib
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool
from std_msgs.msg import String
from irb120_control.util.ft_frames import WRENCH_AXES_FRAME, WRENCH_HEADER_FRAME
from irb120_control.util.ft_capture import state_qos
from irb120_control.util.ft_calibration import default_calibration_path, load_calibration
from tf2_ros import Buffer, TransformException, TransformListener


# ====================================================================
# FT CALIBRATION CONSTANTS — paste from `calibrate_ft_sensor` solve output.
# See README. Re-run calibration if finger/mounting changes.
# ====================================================================
ELECTRICAL_ZERO  = np.array([-0.197409, -0.086334, +0.898847])   # N, frame-fixed
G_WORLD          = np.array([-0.004915, +0.081938, -1.547863])   # N, in world frame
# Torque comp is treated as a constant offset (CoM moment is small; see notes).
TORQUE_ZERO      = np.array([+0.000000, +0.000000, +0.000000])   # N·m, frame-fixed

# ====================================================================

INPUT_TOPIC   = "/netft_data"
OUTPUT_TOPIC  = "/netft_data_transformed"
MONITOR_TOPIC = "/netft_data_monitor"
BASE_FRAME    = "world"
OUTPUT_FRAME  = WRENCH_HEADER_FRAME
TF_FRAME = WRENCH_AXES_FRAME

ENABLE_EMA = True
EMA_ALPHA  = 0.20

TARE_SAMPLES = 200       # samples used for the optional session tare
TF_POLL_HZ   = 50.0      # rate at which R_cur is refreshed from TF
MONITOR_HZ   = 25.0      # rate of /netft_data_monitor for live plotting

# IMPORTANT: NetFT internal tool transform is set so reported axes match world/robot frame
# (the old ft_link was identical to tool0). Keep axes separate from the legacy header label.


class NetFTPreprocessor(Node):
    def __init__(self):
        super().__init__("netft_preprocessor")

        # An explicit calibration_file wins; otherwise the reviewed config/ft_calibration.json
        # if one has been installed; otherwise the legacy constants below (logged loudly).
        self._calibration_file = str(self.declare_parameter('calibration_file', '').value)
        self._calibration_source = 'file'
        if not self._calibration_file and default_calibration_path().exists():
            self._calibration_file = str(default_calibration_path())
            self._calibration_source = 'config/ft_calibration.json'
        self._electrical_zero = ELECTRICAL_ZERO.copy()
        self._g_world = G_WORLD.copy()
        self._torque_zero = TORQUE_ZERO.copy()
        self._tool_com = np.zeros(3)
        self._calibration_sha256 = None
        if self._calibration_file:
            (self._electrical_zero, self._g_world, self._torque_zero,
             self._tool_com) = load_calibration(self._calibration_file)
            self._calibration_sha256 = hashlib.sha256(Path(self._calibration_file).read_bytes()).hexdigest()
            self.get_logger().info(f"F/T calibration: {self._calibration_file}")
        else:
            self._calibration_source = 'legacy constants; recalibration pending'
            self.get_logger().warn("F/T calibration: legacy constants (no calibration file). Recalibrate.")
        self._state_pub = self.create_publisher(String, '/netft_preprocessor/state', state_qos())
        self._last_tf_success = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._pub         = self.create_publisher(WrenchStamped, OUTPUT_TOPIC, 50)
        self._monitor_pub = self.create_publisher(WrenchStamped, MONITOR_TOPIC, 10)
        self._sub         = self.create_subscription(WrenchStamped, INPUT_TOPIC, self._on_wrench, qos)
        self._tare_srv    = self.create_service(SetBool, "/netft_preprocessor/set_tare", self._on_set_tare)

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # --- comp state ---
        # G_WORLD rotated into current sensor frame; recomputed by TF poll.
        self._R_cur          = np.eye(3)
        self._g_world_cur_S  = self._g_world.copy()  # = R_cur^T @ G_WORLD
        self._tool_torque_S  = np.cross(self._tool_com, self._g_world_cur_S)  # tool weight moment

        # --- session tare state ---
        # Captured ONCE per `set_tare data=true` request, then applied each sample.
        self._tare_f       = np.zeros(3)
        self._tare_t       = np.zeros(3)
        self._tare_active  = False
        self._taring       = False
        self._tare_count   = 0
        self._tare_acc_f   = np.zeros(3)
        self._tare_acc_t   = np.zeros(3)

        # --- EMA state ---
        self._ema_1ma  = 1.0 - max(0.0, min(1.0, EMA_ALPHA))
        self._ema_f    = np.zeros(3)
        self._ema_t    = np.zeros(3)
        self._ema_init = False

        # --- misc ---
        self._warn_t_last      = 0.0
        self._last_monitor_pub = 0.0
        self._monitor_interval = 1.0 / MONITOR_HZ
        self._first_sample     = False

        self._watchdog_timer = self.create_timer(10.0, self._watchdog_cb)
        self.create_timer(1.0 / TF_POLL_HZ, self._tf_poll_cb)

        self._publish_state()
        self.get_logger().info(
            f"NetFT preprocessor: {INPUT_TOPIC} -> {OUTPUT_TOPIC} + {MONITOR_TOPIC} @ {MONITOR_HZ:.0f} Hz"
        )
        self.get_logger().info(
            f"Comp constants:  ELECTRICAL_ZERO=({self._electrical_zero[0]:+.3f},{self._electrical_zero[1]:+.3f},{self._electrical_zero[2]:+.3f}) N  "
            f"G_WORLD=({self._g_world[0]:+.3f},{self._g_world[1]:+.3f},{self._g_world[2]:+.3f}) N "
            f"(|G|={np.linalg.norm(self._g_world):.3f} N, mass={np.linalg.norm(self._g_world)/9.81:.4f} kg)"
        )
        self.get_logger().info(
            "Session tare:    ros2 service call /netft_preprocessor/set_tare std_srvs/srv/SetBool '{data: true}'"
        )

    # ------------------------------------------------------------------
    # TF poll — keeps R_cur and rotated G_WORLD fresh for the hot path
    # ------------------------------------------------------------------

    def _tf_poll_cb(self):
        try:
            tf = self._tf_buffer.lookup_transform(BASE_FRAME, TF_FRAME, rclpy.time.Time())
        except TransformException:
            self._warn_throttled(f"Waiting for gravity-compensation TF {BASE_FRAME} -> {TF_FRAME}")
            return
        now = self._safe_now()
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if stamp == 0 or not -0.1 <= now - stamp <= 0.5:
            self._warn_throttled("Stale gravity-compensation TF; waiting for current robot pose")
            return
        self._last_tf_success = now
        q = tf.transform.rotation
        self._R_cur = self._quat_to_rot(q.x, q.y, q.z, q.w)
        # Pre-compute G_WORLD in current sensor frame so the hot path is a single subtract.
        self._g_world_cur_S = self._R_cur.T @ self._g_world
        self._tool_torque_S = np.cross(self._tool_com, self._g_world_cur_S)

    # ------------------------------------------------------------------
    # Session tare service
    # ------------------------------------------------------------------

    def _on_set_tare(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        if req.data and (self._last_tf_success is None or self._safe_now() - self._last_tf_success > 0.5):
            res.success = False
            res.message = "Cannot tare without a current gravity-compensation TF"
            return res
        if req.data:
            # Capture residual offset at the current pose (after calibration comp).
            self._taring     = True
            self._tare_count = 0
            self._tare_acc_f = np.zeros(3)
            self._tare_acc_t = np.zeros(3)
            res.success = True
            res.message = f"Session tare started ({TARE_SAMPLES} samples). Hold robot stationary."
        else:
            self._tare_f      = np.zeros(3)
            self._tare_t      = np.zeros(3)
            self._tare_active = False
            self._taring      = False
            res.success = True
            res.message = "Session tare cleared (calibration constants unchanged)."
        self._publish_state()
        self.get_logger().info(res.message)
        return res

    def _finalise_tare(self):
        inv = 1.0 / self._tare_count
        self._tare_f      = self._tare_acc_f * inv
        self._tare_t      = self._tare_acc_t * inv
        self._tare_active = True
        self._taring      = False
        self._publish_state()
        self.get_logger().info(
            f"Session tare captured. residual_f=({self._tare_f[0]:+.4f},{self._tare_f[1]:+.4f},{self._tare_f[2]:+.4f}) "
            f"residual_t=({self._tare_t[0]:+.4f},{self._tare_t[1]:+.4f},{self._tare_t[2]:+.4f})"
        )

    def _publish_state(self):
        state = {
            'schema_version': 1, 'time_s': self._safe_now(),
            'axes_frame': TF_FRAME, 'header_frame': OUTPUT_FRAME,
            'calibration_file': self._calibration_file or None,
            'calibration_sha256': self._calibration_sha256,
            'calibration_source': self._calibration_source,
            'electrical_zero': self._electrical_zero.tolist(),
            'g_world': self._g_world.tolist(), 'torque_zero': self._torque_zero.tolist(),
            'tool_com_m': self._tool_com.tolist(),
            'ema_enabled': ENABLE_EMA, 'ema_alpha': EMA_ALPHA,
            'tare_active': self._tare_active, 'taring': self._taring,
            'tare_samples': self._tare_count,
            'tare_force': self._tare_f.tolist(), 'tare_torque': self._tare_t.tolist(),
        }
        self._state_pub.publish(String(data=json.dumps(state)))

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def _watchdog_cb(self):
        if not self._first_sample:
            self.get_logger().fatal(
                f"No data on {INPUT_TOPIC} after 10 s -- sensor not connected. Shutting down."
            )
            raise SystemExit(1)
        self._watchdog_timer.cancel()

    # ------------------------------------------------------------------
    # Hot path
    # ------------------------------------------------------------------

    def _on_wrench(self, msg: WrenchStamped):
        self._first_sample = True

        f = np.array([msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z])
        t = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])

        if not (np.isfinite(f).all() and np.isfinite(t).all()):
            self._warn_throttled("Non-finite NetFT sample (NaN/Inf) -- dropping")
            self._publish(np.zeros(3), np.zeros(3), msg.header.stamp)
            return

        if self._last_tf_success is None or self._safe_now() - self._last_tf_success > 0.5:
            self._warn_throttled("No current gravity-compensation TF; withholding compensated wrench")
            return

        # Apply calibration comp.
        f = f - self._electrical_zero - self._g_world_cur_S
        t = t - self._torque_zero - self._tool_torque_S

        # Session tare: accumulate raw post-comp residual, then subtract once captured.
        if self._taring:
            self._tare_acc_f += f
            self._tare_acc_t += t
            self._tare_count += 1
            if self._tare_count >= TARE_SAMPLES:
                self._finalise_tare()

        if self._tare_active:
            f = f - self._tare_f
            t = t - self._tare_t

        # EMA low-pass filter.
        if ENABLE_EMA:
            if not self._ema_init:
                self._ema_f    = f.copy()
                self._ema_t    = t.copy()
                self._ema_init = True
            else:
                self._ema_f = self._ema_1ma * self._ema_f + EMA_ALPHA * f
                self._ema_t = self._ema_1ma * self._ema_t + EMA_ALPHA * t
            f = self._ema_f
            t = self._ema_t

        self._publish(f, t, msg.header.stamp)

    # ------------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------------

    def _publish(self, f, t, stamp):
        out = WrenchStamped()
        out.header.stamp    = stamp
        out.header.frame_id = OUTPUT_FRAME
        out.wrench.force.x  = float(f[0]);  out.wrench.force.y  = float(f[1]);  out.wrench.force.z  = float(f[2])
        out.wrench.torque.x = float(t[0]);  out.wrench.torque.y = float(t[1]);  out.wrench.torque.z = float(t[2])
        self._pub.publish(out)

        now = self._safe_now()
        if now - self._last_monitor_pub >= self._monitor_interval:
            self._last_monitor_pub = now
            self._monitor_pub.publish(out)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _safe_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttled(self, msg, period=2.0):
        now = self._safe_now()
        if now - self._warn_t_last > period:
            self._warn_t_last = now
            self.get_logger().warn(msg)

    @staticmethod
    def _quat_to_rot(x, y, z, w) -> np.ndarray:
        """Quaternion -> 3x3 rotation matrix. R takes sensor-frame vectors to world."""
        x2, y2, z2 = x*x, y*y, z*z
        return np.array([
            [1 - 2*(y2+z2),   2*(x*y - w*z),  2*(x*z + w*y)],
            [2*(x*y + w*z),   1 - 2*(x2+z2),  2*(y*z - w*x)],
            [2*(x*z - w*y),   2*(y*z + w*x),  1 - 2*(x2+y2)],
        ])


def main(args=None):
    rclpy.init(args=args)
    node = NetFTPreprocessor()
    exit_code = 0
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        exit_code = int(e.code) if e.code is not None else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    main()