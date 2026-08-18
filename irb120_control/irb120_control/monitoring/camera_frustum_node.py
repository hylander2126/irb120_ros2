#!/usr/bin/env python3

# UNUSED CURRENTLY 

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import numpy as np


class CameraFrustumNode(Node):
    def __init__(self):
        super().__init__('camera_frustum_node')

        self.declare_parameter('near', 0.1)
        self.declare_parameter('far', 2.15)
        self.declare_parameter('r', 0.2)
        self.declare_parameter('g', 0.8)
        self.declare_parameter('b', 1.0)
        self.declare_parameter('alpha', 0.6)
        self.declare_parameter('line_width', 0.005)
        self.declare_parameter('camera_info_topic', '/realsense/depth/camera_info')

        self._near = self.get_parameter('near').value
        self._far = self.get_parameter('far').value
        self._color = ColorRGBA(
            r=self.get_parameter('r').value,
            g=self.get_parameter('g').value,
            b=self.get_parameter('b').value,
            a=self.get_parameter('alpha').value,
        )
        self._line_width = self.get_parameter('line_width').value

        self._pub = self.create_publisher(Marker, '/realsense/fov_marker', 10)

        topic = self.get_parameter('camera_info_topic').value
        self._sub = self.create_subscription(
            CameraInfo, topic, self._camera_info_cb, qos_profile_sensor_data
        )
        self.get_logger().info(f'Camera frustum node started, listening on {topic}')

    def _camera_info_cb(self, msg: CameraInfo):
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        w = msg.width
        h = msg.height

        # Pixel corners → unit-ray directions (camera optical frame: +Z forward)
        corners_px = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
        rays = [
            np.array([(px - cx) / fx, (py - cy) / fy, 1.0])
            for px, py in corners_px
        ]

        near_pts = [r * self._near for r in rays]
        far_pts = [r * self._far for r in rays]

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = msg.header.frame_id  # realsense_depth_optical_frame
        marker.ns = 'camera_fov'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = self._line_width
        marker.color = self._color

        def pt(v):
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            return p

        # Near rectangle
        for i in range(4):
            marker.points.append(pt(near_pts[i]))
            marker.points.append(pt(near_pts[(i + 1) % 4]))

        # Far rectangle
        for i in range(4):
            marker.points.append(pt(far_pts[i]))
            marker.points.append(pt(far_pts[(i + 1) % 4]))

        # Connecting edges (origin → far corners to show apex)
        origin = np.zeros(3)
        for fp in far_pts:
            marker.points.append(pt(origin))
            marker.points.append(pt(fp))

        self._pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CameraFrustumNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
