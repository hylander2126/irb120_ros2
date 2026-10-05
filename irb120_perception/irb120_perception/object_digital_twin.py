#!/usr/bin/env python3
"""Publish an editable, perception-aligned object twin for RViz."""

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


class ObjectDigitalTwin(Node):
    """A detected cloud drives pose; a JSON file drives editable appearance."""
    def __init__(self):
        super().__init__('object_digital_twin')
        self.declare_parameter('input_cloud', '/object_detector_dbscan_sam_cull/object_points')
        self.declare_parameter('marker_topic', '/object_digital_twin/markers')
        self.declare_parameter('model_file', '')
        self.declare_parameter('frame_id', 'base_link')
        self._model_path = Path(self.get_parameter('model_file').value).expanduser()
        self._model_mtime, self._last_cloud = None, None
        self._model = self._defaults()
        self._publisher = self.create_publisher(MarkerArray, self.get_parameter('marker_topic').value, 1)
        self.create_subscription(PointCloud2, self.get_parameter('input_cloud').value, self._cloud, 10)
        self.create_timer(0.5, self._publish)
        self._reload_model(force=True)

    @staticmethod
    def _defaults():
        return {'name': 'detected object', 'shape': 'box', 'mesh_resource': '', 'dimensions_m': None,
                'use_detected_dimensions': True, 'offset_m': [0., 0., 0.],
                'color_rgba': [0.1, 0.75, 1., 0.45], 'show_detected_bounds': True}

    def _reload_model(self, force=False):
        if not self._model_path:
            return
        try:
            mtime = self._model_path.stat().st_mtime_ns
            if not force and mtime == self._model_mtime:
                return
            with self._model_path.open() as f:
                update = json.load(f)
            if not isinstance(update, dict):
                raise ValueError('top level must be an object')
            self._model, self._model_mtime = {**self._defaults(), **update}, mtime
            self.get_logger().info(f'Reloaded editable twin model: {self._model_path}')
        except FileNotFoundError:
            if force:
                self.get_logger().warn(f'Twin model file not found: {self._model_path}; using detected bounds')
        except (json.JSONDecodeError, ValueError) as exc:
            self.get_logger().error(f'Keeping previous twin model; cannot read {self._model_path}: {exc}')

    def _cloud(self, msg):
        p = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1)
        if len(p) >= 3:
            self._last_cloud = (msg.header.frame_id or self.get_parameter('frame_id').value,
                                np.column_stack([p['x'], p['y'], p['z']]))

    @staticmethod
    def _marker(frame, stamp, marker_id, kind, pos, scale, color, text=''):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = frame, stamp
        marker.ns, marker.id, marker.type, marker.action = 'object_digital_twin', marker_id, kind, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, pos)
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, scale)
        marker.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=float(color[3]))
        marker.text = text
        return marker

    def _publish(self):
        self._reload_model()
        if self._last_cloud is None:
            return
        frame, points = self._last_cloud
        lo, hi = points.min(axis=0), points.max(axis=0)
        observed_size, center = np.maximum(hi - lo, 0.002), (lo + hi) / 2
        dims = self._model['dimensions_m']
        size = np.asarray(dims if dims and not self._model['use_detected_dimensions'] else observed_size, dtype=float)
        if size.shape != (3,) or np.any(size <= 0):
            self.get_logger().error('dimensions_m must contain three positive metre values; using detected bounds')
            size = observed_size
        offset = np.asarray(self._model['offset_m'], dtype=float)
        center += offset if offset.shape == (3,) else 0
        color = self._model['color_rgba'] if len(self._model['color_rgba']) == 4 else self._defaults()['color_rgba']
        stamp = self.get_clock().now().to_msg()
        shape = Marker.CYLINDER if self._model['shape'] == 'cylinder' else Marker.CUBE
        if self._model['mesh_resource']:
            shape = Marker.MESH_RESOURCE
        model_marker = self._marker(frame, stamp, 0, shape, center, size, color)
        if shape == Marker.MESH_RESOURCE:
            # Supports package://... paths, file:// paths, and RViz-supported mesh formats.
            model_marker.mesh_resource = self._model['mesh_resource']
            model_marker.mesh_use_embedded_materials = True
        markers = [model_marker]
        if self._model['show_detected_bounds']:
            markers.append(self._marker(frame, stamp, 1, Marker.CUBE, (lo + hi) / 2, observed_size,
                                        [1., 1., 1., .16]))
        label = f"{self._model['name']}  {size[0]*1000:.0f} x {size[1]*1000:.0f} x {size[2]*1000:.0f} mm"
        markers.append(self._marker(frame, stamp, 2, Marker.TEXT_VIEW_FACING,
                                    center + [0, 0, size[2] / 2 + .03], [0, 0, .035], [1, 1, 1, .95], label))
        self._publisher.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDigitalTwin()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
