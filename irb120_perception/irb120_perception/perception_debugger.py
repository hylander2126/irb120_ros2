"""
Perception Debugger Node
========================
Companion to the object_detector node. Subscribes to the on-demand debug
topics that object_detector publishes when triggered, and logs a formatted
point-count / extent report to the console.

Trigger a snapshot while this node is running:
  ros2 topic pub --once /object_detector/debug_snapshot std_msgs/msg/Empty '{}'

Topics consumed (all latched depth=1, published once per trigger):
  /object_detector/debug/mask_overlay    sensor_msgs/Image
  /object_detector/debug/pts_camera      sensor_msgs/PointCloud2
  /object_detector/debug/pts_after_roi   sensor_msgs/PointCloud2
  /object_detector/debug/pts_after_clean sensor_msgs/PointCloud2

This node does NOT forward or re-publish these topics — they are consumed
directly by RViz displays configured in moveit_debug_perception.rviz.
"""

import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, PointCloud2


def _unpack_pointcloud2(msg: PointCloud2) -> np.ndarray:
    """Unpack a PointCloud2 message into an (N, 3) float32 array."""
    step = msg.point_step
    data = msg.data
    n = msg.width * msg.height
    xyz = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        b = i * step
        xyz[i, 0] = struct.unpack_from('f', data, b + 0)[0]
        xyz[i, 1] = struct.unpack_from('f', data, b + 4)[0]
        xyz[i, 2] = struct.unpack_from('f', data, b + 8)[0]
    return xyz


def _extents(pts: np.ndarray) -> str:
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    span = mx - mn
    return (
        f'x=[{mn[0]:+.3f}, {mx[0]:+.3f}] span={span[0]:.3f}m  '
        f'y=[{mn[1]:+.3f}, {mx[1]:+.3f}] span={span[1]:.3f}m  '
        f'z=[{mn[2]:+.3f}, {mx[2]:+.3f}] span={span[2]:.3f}m'
    )


class PerceptionDebugger(Node):

    def __init__(self):
        super().__init__('perception_debugger')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Latest snapshot buffers — populated as each topic arrives
        self._pts_cam:   np.ndarray | None = None
        self._pts_roi:   np.ndarray | None = None
        self._pts_clean: np.ndarray | None = None
        self._got_image  = False

        self.create_subscription(
            Image, '/object_detector/debug/mask_overlay',
            self._mask_cb, qos)
        self.create_subscription(
            PointCloud2, '/object_detector/debug/pts_camera',
            self._pts_cam_cb, qos)
        self.create_subscription(
            PointCloud2, '/object_detector/debug/pts_after_roi',
            self._pts_roi_cb, qos)
        self.create_subscription(
            PointCloud2, '/object_detector/debug/pts_after_clean',
            self._pts_clean_cb, qos)

        self.get_logger().info(
            'perception_debugger ready.\n'
            'Trigger a snapshot with:\n'
            '  ros2 topic pub --once /object_detector/debug_snapshot '
            'std_msgs/msg/Empty \'{}\''
        )

    # -------------------------------------------------------------------------

    def _mask_cb(self, _: Image):
        self._got_image = True
        self._try_report()

    def _pts_cam_cb(self, msg: PointCloud2):
        self._pts_cam = _unpack_pointcloud2(msg)
        self._try_report()

    def _pts_roi_cb(self, msg: PointCloud2):
        self._pts_roi = _unpack_pointcloud2(msg)
        self._try_report()

    def _pts_clean_cb(self, msg: PointCloud2):
        self._pts_clean = _unpack_pointcloud2(msg)
        self._try_report()

    def _try_report(self):
        """Print the report once all four topics have arrived for this snapshot."""
        if not (self._got_image
                and self._pts_cam   is not None
                and self._pts_roi   is not None
                and self._pts_clean is not None):
            return

        cam   = self._pts_cam
        roi   = self._pts_roi
        clean = self._pts_clean

        # Clear buffers so next trigger produces a fresh report
        self._pts_cam = self._pts_roi = self._pts_clean = None
        self._got_image = False

        self.get_logger().info(
            '\n'
            '╔══════════════════════════════════════════════════════════╗\n'
            '║              SAM PIPELINE DEBUG SNAPSHOT                ║\n'
            '╠══════════════════════════════════════════════════════════╣\n'
            f'║  Stage              Points   \n'
            f'║  ─────────────────────────────────────────────────────  \n'
            f'║  pts_camera         {len(cam):6d}   {_extents(cam)}\n'
            f'║  pts_after_roi      {len(roi):6d}   {_extents(roi)}\n'
            f'║  pts_after_clean    {len(clean):6d}   {_extents(clean)}\n'
            f'║  \n'
            f'║  Hull input Z-span: {clean[:,2].max() - clean[:,2].min():.4f} m  '
            f'(< 5mm → likely flat/degenerate hull)\n'
            '╚══════════════════════════════════════════════════════════╝\n'
            'RViz topics:\n'
            '  mask overlay  → /object_detector/debug/mask_overlay\n'
            '  raw pts (cam) → /object_detector/debug/pts_camera\n'
            '  ROI pts       → /object_detector/debug/pts_after_roi\n'
            '  clean pts     → /object_detector/debug/pts_after_clean\n'
        )


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionDebugger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
