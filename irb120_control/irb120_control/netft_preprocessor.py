#!/usr/bin/env python3
"""Filter and re-orient NetFT wrench data in ROS 2.

Settings are intentionally hardcoded for this workspace.
"""

import math

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


INPUT_TOPIC = "/netft_data"
OUTPUT_TOPIC = "/netft_data_transformed"
MONITOR_TOPIC = "/netft_data_monitor"
BASE_FRAME = "base_link"
DEFAULT_SENSOR_FRAME = "tool0"
OUTPUT_FRAME = "base_link"

# Sensor axes expressed in the robot/world frame:
# +x_s = -y_w, +y_s = -z_w, +z_s = +x_w
# Therefore v_w = [v_s.z, -v_s.x, -v_s.y]

ENABLE_EMA = False # Was true, testing latency
EMA_ALPHA = 0.20

ENABLE_BIAS = True
BIAS_SAMPLES = 150

MONITOR_HZ = 25.0


def rotate_sensor_to_world(v):
    return (v[2], -v[0], -v[1])


class NetFTPreprocessor(Node):
    def __init__(self):
        super().__init__("netft_preprocessor")

        self.input_topic = INPUT_TOPIC
        self.output_topic = OUTPUT_TOPIC
        self.monitor_topic = MONITOR_TOPIC
        self.default_sensor_frame = DEFAULT_SENSOR_FRAME
        self.output_frame = OUTPUT_FRAME
        self.enable_ema = ENABLE_EMA
        self.ema_alpha = EMA_ALPHA
        self.enable_bias = ENABLE_BIAS
        self.bias_samples = BIAS_SAMPLES

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self._pub = self.create_publisher(WrenchStamped, self.output_topic, 50)
        self._monitor_pub = self.create_publisher(WrenchStamped, self.monitor_topic, 10)
        self._sub = self.create_subscription(WrenchStamped, self.input_topic, self._on_wrench, sensor_qos)

        self._warn_t_last = 0.0

        self._biasing = self.enable_bias and self.bias_samples > 0
        self._bias_count = 0
        self._bias_f = [0.0, 0.0, 0.0]
        self._bias_t = [0.0, 0.0, 0.0]
        self._acc_f = [0.0, 0.0, 0.0]
        self._acc_t = [0.0, 0.0, 0.0]

        self._ema_f = None
        self._ema_t = None
        self._last_monitor_pub = 0.0

        self.get_logger().info(
            f"NetFT preprocessor: {self.input_topic} -> {self.output_topic} + {self.monitor_topic}, "
            f"sensor mount fixed to world/base axes, output={self.output_frame}, monitor={MONITOR_HZ:.1f}Hz"
        )

    def _safe_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _warn_throttled(self, msg, period=2.0):
        now = self._safe_now()
        if now - self._warn_t_last > period:
            self._warn_t_last = now
            self.get_logger().warn(msg)

    def _publish_monitor(self, f_out, t_out, stamp):
        now = self._safe_now()
        if now - self._last_monitor_pub < (1.0 / MONITOR_HZ):
            return
        self._last_monitor_pub = now

        out = WrenchStamped()
        out.header.stamp = stamp
        out.header.frame_id = self.output_frame
        out.wrench.force.x = f_out[0]
        out.wrench.force.y = f_out[1]
        out.wrench.force.z = f_out[2]
        out.wrench.torque.x = t_out[0]
        out.wrench.torque.y = t_out[1]
        out.wrench.torque.z = t_out[2]
        self._monitor_pub.publish(out)

    def _on_wrench(self, msg: WrenchStamped):
        f_raw = (msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z)
        t_raw = (msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z)

        if not all(math.isfinite(v) for v in (*f_raw, *t_raw)):
            self._warn_throttled("Dropped non-finite NetFT sample (NaN/Inf)", period=1.0)
            return

        f_rot = rotate_sensor_to_world(f_raw)
        t_rot = rotate_sensor_to_world(t_raw)

        if self._biasing:
            self._acc_f[0] += f_rot[0]
            self._acc_f[1] += f_rot[1]
            self._acc_f[2] += f_rot[2]
            self._acc_t[0] += t_rot[0]
            self._acc_t[1] += t_rot[1]
            self._acc_t[2] += t_rot[2]
            self._bias_count += 1
            if self._bias_count >= self.bias_samples:
                inv = 1.0 / float(self._bias_count)
                self._bias_f = [self._acc_f[0] * inv, self._acc_f[1] * inv, self._acc_f[2] * inv]
                self._bias_t = [self._acc_t[0] * inv, self._acc_t[1] * inv, self._acc_t[2] * inv]
                self._biasing = False
                self.get_logger().info(
                    "Bias complete: "
                    f"F=({self._bias_f[0]:.3f},{self._bias_f[1]:.3f},{self._bias_f[2]:.3f}) "
                    f"T=({self._bias_t[0]:.3f},{self._bias_t[1]:.3f},{self._bias_t[2]:.3f})"
                )

        f = [f_rot[0] - self._bias_f[0], f_rot[1] - self._bias_f[1], f_rot[2] - self._bias_f[2]]
        t = [t_rot[0] - self._bias_t[0], t_rot[1] - self._bias_t[1], t_rot[2] - self._bias_t[2]]

        if self.enable_ema:
            a = max(0.0, min(1.0, self.ema_alpha))
            if self._ema_f is None:
                self._ema_f = f
                self._ema_t = t
            else:
                self._ema_f = [(1.0 - a) * self._ema_f[i] + a * f[i] for i in range(3)]
                self._ema_t = [(1.0 - a) * self._ema_t[i] + a * t[i] for i in range(3)]
            f_out = self._ema_f
            t_out = self._ema_t
        else:
            f_out = f
            t_out = t

        out = WrenchStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame
        out.wrench.force.x = f_out[0]
        out.wrench.force.y = f_out[1]
        out.wrench.force.z = f_out[2]
        out.wrench.torque.x = t_out[0]
        out.wrench.torque.y = t_out[1]
        out.wrench.torque.z = t_out[2]
        self._pub.publish(out)
        self._publish_monitor(f_out, t_out, msg.header.stamp)


def main(args=None):
    rclpy.init(args=args)
    node = NetFTPreprocessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
