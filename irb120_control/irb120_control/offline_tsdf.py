#!/usr/bin/env python3
"""Capture a short, coloured TSDF scan directly from the running ROS stack.

This intentionally has no bag or episode dependency.  Start it while the
object is stationary and the normal perception/camera stack is running; it
uses the active SAM-culled cloud as an object mask and live RGB-D frames for a
few seconds, then writes one coloured PLY and one easy-to-view STL.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

from irb120_control.tsdf_mesh import clean_mesh

CAMERAS = {'cam1': 'realsense', 'cam2': 'realsense2', 'cam3': 'realsense3'}
OBJECT_TOPIC = '/object_detector_dbscan_sam_cull/object_points'


def _image(message):
    """Decode standard RealSense images directly; no cv_bridge ABI dependency."""
    encoding = message.encoding.lower()
    if encoding in ('16uc1', 'mono16'):
        dtype, channels = np.uint16, 1
    elif encoding == '32fc1':
        dtype, channels = np.float32, 1
    elif encoding in ('rgb8', 'bgr8'):
        dtype, channels = np.uint8, 3
    else:
        raise ValueError(f'Unsupported image encoding {message.encoding}')
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, :message.width * channels * np.dtype(dtype).itemsize].view(dtype)
    image = image.reshape(message.height, message.width, channels) if channels == 3 else image.reshape(message.height, message.width)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if encoding == 'bgr8' else image


class LiveTsdf:
    def __init__(self, node, voxel_m, dilation_px, max_frames):
        self.node, self.dilation_px, self.max_frames = node, dilation_px, max_frames
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_m, sdf_trunc=4 * voxel_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.xyz, self.frame = None, 'base_link'
        self.latest = {cam: {} for cam in CAMERAS}
        self.counts = {cam: 0 for cam in CAMERAS}
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        node.create_subscription(PointCloud2, OBJECT_TOPIC, self._cloud, qos_profile_sensor_data)
        for cam, namespace in CAMERAS.items():
            node.create_subscription(Image, f'/{namespace}/color/image_raw',
                                     lambda msg, c=cam: self._set(c, 'color', msg), qos_profile_sensor_data)
            node.create_subscription(CameraInfo, f'/{namespace}/color/camera_info',
                                     lambda msg, c=cam: self._set(c, 'info', msg), qos_profile_sensor_data)
            node.create_subscription(Image, f'/{namespace}/aligned_depth_to_color/image_raw',
                                     lambda msg, c=cam: self._depth(c, msg), qos_profile_sensor_data)

    def _set(self, camera, kind, message):
        self.latest[camera][kind] = message

    def _cloud(self, msg):
        points = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True).reshape(-1)
        if len(points) >= 3:
            self.xyz = np.column_stack([points['x'], points['y'], points['z']]).astype(float)
            self.frame = msg.header.frame_id or 'base_link'

    def _depth(self, camera, depth_message):
        if self.xyz is None or self.counts[camera] >= self.max_frames:
            return
        latest = self.latest[camera]
        if 'color' not in latest or 'info' not in latest:
            return
        try:
            depth, color = _image(depth_message), _image(latest['color'])
            if depth.dtype != np.uint16:
                depth = np.nan_to_num(depth * 1000.).clip(0, 65535).astype(np.uint16)
            if color.shape[:2] != depth.shape:
                return
            info = latest['info']
            K, D = np.asarray(info.k).reshape(3, 3), np.asarray(info.d)
            tf = self.tf.lookup_transform(info.header.frame_id, self.frame, rclpy.time.Time())
            q, tr = tf.transform.rotation, tf.transform.translation
            R = np.array([[1 - 2*(q.y*q.y + q.z*q.z), 2*(q.x*q.y - q.z*q.w), 2*(q.x*q.z + q.y*q.w)],
                          [2*(q.x*q.y + q.z*q.w), 1 - 2*(q.x*q.x + q.z*q.z), 2*(q.y*q.z - q.x*q.w)],
                          [2*(q.x*q.z - q.y*q.w), 2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x*q.x + q.y*q.y)]])
            t = np.array([tr.x, tr.y, tr.z])
        except (TransformException, ValueError):
            return
        optical = self.xyz @ R.T + t
        optical = optical[optical[:, 2] > .05]
        uv, _ = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), K, D)
        uv = np.rint(uv.reshape(-1, 2)).astype(int)
        mask = np.zeros(depth.shape, np.uint8)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
        uv = uv[inside]
        mask[uv[:, 1], uv[:, 0]] = 255
        mask = cv2.dilate(mask, np.ones((self.dilation_px, self.dilation_px), np.uint8))
        masked_depth = np.where(mask > 0, depth, 0).astype(np.uint16)
        if not np.any(masked_depth):
            return
        color = color.copy()
        color[mask == 0] = 0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(depth.shape[1], depth.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        extrinsic = np.eye(4)
        extrinsic[:3, :3], extrinsic[:3, 3] = R, t
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(masked_depth), depth_scale=1000., depth_trunc=2.,
            convert_rgb_to_intensity=False)
        self.volume.integrate(rgbd, intrinsic, extrinsic)
        self.counts[camera] += 1

    @property
    def frames(self):
        return sum(self.counts.values())

    @property
    def full(self):
        return all(n >= self.max_frames for n in self.counts.values())


def main(args=None):
    raw = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=3.0, help='maximum capture time in seconds')
    parser.add_argument('--max-frames-per-camera', type=int, default=12)
    parser.add_argument('--voxel-m', type=float, default=.005)
    parser.add_argument('--dilation-px', type=int, default=9)
    parser.add_argument('--completion', choices=('convex-hull', 'none'), default='convex-hull',
                        help='convex-hull fills all concavities; none retains observed geometry')
    parser.add_argument('--smooth-iterations', type=int, default=10)
    parser.add_argument('--output-dir', help='directory for mesh files; default: runtime_logs/tsdf/<timestamp>')
    parsed, ros_args = parser.parse_known_args(raw)
    if parsed.voxel_m <= 0 or parsed.smooth_iterations < 0:
        parser.error('--voxel-m must be positive and --smooth-iterations nonnegative')
    rclpy.init(args=ros_args)
    node = rclpy.create_node('offline_tsdf_capture')
    capture = LiveTsdf(node, parsed.voxel_m, parsed.dilation_px, parsed.max_frames_per_camera)
    deadline = time.monotonic() + parsed.duration
    try:
        while rclpy.ok() and time.monotonic() < deadline and not capture.full:
            rclpy.spin_once(node, timeout_sec=.05)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not capture.frames:
        raise RuntimeError(f'No frames fused. Is the active stack publishing {OBJECT_TOPIC} and all cameras?')
    out = Path(parsed.output_dir) if parsed.output_dir else Path.cwd() / 'runtime_logs' / 'tsdf' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    mesh, cloud = capture.volume.extract_triangle_mesh(), capture.volume.extract_point_cloud()
    o3d.io.write_triangle_mesh(str(out / 'object_mesh_raw.ply'), mesh)
    mesh, cleanup = clean_mesh(mesh, parsed.voxel_m, parsed.completion, parsed.smooth_iterations)
    o3d.io.write_triangle_mesh(str(out / 'object_mesh_colored.ply'), mesh)
    o3d.io.write_triangle_mesh(str(out / 'object_mesh.stl'), mesh)  # geometry-only, broadly supported
    o3d.io.write_point_cloud(str(out / 'object_surface_colored.ply'), cloud)
    report = {'source': 'live ROS topics', 'voxel_m': parsed.voxel_m, 'frames_integrated': capture.frames,
              'frames_by_camera': capture.counts, 'mesh_vertices': len(mesh.vertices),
              'mesh_triangles': len(mesh.triangles), 'cleanup': cleanup, 'output': str(out)}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
