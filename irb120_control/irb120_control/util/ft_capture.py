"""Lightweight raw wrench and occasional processing-state capture; no TF work."""

import json

from geometry_msgs.msg import WrenchStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from irb120_control.util.ft_frames import ESTIMATOR_POSE_FRAME, WRENCH_AXES_FRAME


def stamp_s(stamp):
    """Convert a ROS stamp, preserving zero as an explicit unknown stamp."""
    return stamp.sec + stamp.nanosec * 1e-9


def state_qos():
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                      reliability=ReliabilityPolicy.RELIABLE)


class FTCapture:
    """Record raw samples and latched calibration/tare state for one node."""

    def __init__(self, node, time_s, enabled=lambda: True):
        self.node = node
        self.time_s = time_s
        self.enabled = enabled
        self.raw = []
        self.states = []
        self.latest_state = None
        self.raw_frame = None
        self.raw_sub = node.create_subscription(
            WrenchStamped, '/netft_data', self.on_raw,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.state_sub = node.create_subscription(
            String, '/netft_preprocessor/state', self.on_state, state_qos())

    def on_raw(self, msg):
        if not self.enabled():
            return
        f, t = msg.wrench.force, msg.wrench.torque
        self.raw_frame = msg.header.frame_id
        self.raw.append([self.time_s(), f.x, f.y, f.z, t.x, t.y, t.z,
                         stamp_s(msg.header.stamp)])

    def on_state(self, msg):
        try:
            state = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self.latest_state = state
        self.states.append(state)

    def reset(self):
        self.raw.clear()
        self.states = [self.latest_state] if self.latest_state is not None else []

    def metadata(self):
        return {'log_schema_version': 2, 'ft_pose_reference_frame': ESTIMATOR_POSE_FRAME,
                'wrench_axes_frame': WRENCH_AXES_FRAME, 'raw_header_frame': self.raw_frame,
                'source_stamps': 'ROS seconds; zero means unspecified; TF is latest, not time-matched',
                'ft_processing_states': self.states,
                'ft_processing_state_received': self.latest_state is not None}
