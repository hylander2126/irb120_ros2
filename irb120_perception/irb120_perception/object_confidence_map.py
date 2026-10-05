#!/usr/bin/env python3
"""CPU-light persistent voxel confidence map for a stationary segmented object."""

import struct

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import SetBool


class ObjectConfidenceMap(Node):
    def __init__(self):
        super().__init__('object_confidence_map')
        self.declare_parameter('input_cloud', '/object_detector_dbscan_sam_cull/object_points')
        self.declare_parameter('output_cloud', '/object_digital_twin/confidence_points')
        self.declare_parameter('voxel_size_m', 0.005)
        self.declare_parameter('min_observations', 3)
        self.declare_parameter('motion_reset_m', 0.035)
        self.declare_parameter('max_age_sec', 20.0)
        self._voxel = float(self.get_parameter('voxel_size_m').value)
        self._cells, self._last_center, self._frame = {}, None, ''
        self._publisher = self.create_publisher(PointCloud2, self.get_parameter('output_cloud').value, 1)
        self.create_subscription(PointCloud2, self.get_parameter('input_cloud').value, self._cloud, 10)
        self.create_service(SetBool, '/object_confidence_map/reset', self._reset)
        self.create_timer(0.5, self._publish)

    def _reset(self, request, response):
        if request.data:
            self._clear('service request')
            response.success, response.message = True, 'confidence map cleared'
        else:
            response.success, response.message = False, 'set data=true to clear the confidence map'
        return response

    def _clear(self, why):
        self._cells, self._last_center = {}, None
        self.get_logger().info(f'Confidence map reset ({why})')

    def _cloud(self, msg):
        points = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1)
        if len(points) < 3:
            return
        xyz = np.column_stack([points['x'], points['y'], points['z']]).astype(float)
        center = np.median(xyz, axis=0)
        if self._last_center is not None and np.linalg.norm(center - self._last_center) > self.get_parameter('motion_reset_m').value:
            self._clear(f'object centroid moved {np.linalg.norm(center - self._last_center) * 1000:.0f} mm')
        self._last_center, self._frame = center, msg.header.frame_id or 'base_link'
        keys = np.floor(xyz / self._voxel).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        now = self.get_clock().now().nanoseconds * 1e-9
        for i, key in enumerate(unique):
            mean = xyz[inverse == i].mean(axis=0)
            cell = self._cells.get(tuple(key))
            if cell is None:
                self._cells[tuple(key)] = [mean[0], mean[1], mean[2], 1, now]
            else:
                # Bounded running average: stable geometry converges, but can still adapt
                # modestly to a corrected calibration rather than becoming permanently stale.
                weight = min(cell[3], 100)
                cell[:3] = ((np.asarray(cell[:3]) * weight + mean) / (weight + 1)).tolist()
                cell[3], cell[4] = cell[3] + 1, now

    def _publish(self):
        if not self._cells:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        max_age = float(self.get_parameter('max_age_sec').value)
        self._cells = {k: v for k, v in self._cells.items() if now - v[4] <= max_age}
        minimum = int(self.get_parameter('min_observations').value)
        stable = [v for v in self._cells.values() if v[3] >= minimum]
        if not stable:
            return
        maximum = max(v[3] for v in stable)
        packed = []
        for x, y, z, hits, _ in stable:
            c = min(1.0, hits / maximum)
            # Blue (new/uncertain) -> green -> yellow (well-supported).
            r, g, b = int(255 * c), int(255 * min(1., 2 * c)), int(255 * (1 - c))
            packed.append((x, y, z, struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]))
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self._frame)
        fields = [PointField(name=n, offset=o, datatype=t, count=1) for n, o, t in (
            ('x', 0, PointField.FLOAT32), ('y', 4, PointField.FLOAT32),
            ('z', 8, PointField.FLOAT32), ('rgb', 12, PointField.UINT32))]
        self._publisher.publish(point_cloud2.create_cloud(header, fields, packed))


def main(args=None):
    rclpy.init(args=args)
    node = ObjectConfidenceMap()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
