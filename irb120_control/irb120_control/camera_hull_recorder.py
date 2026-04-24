#!/usr/bin/env python3
"""Overlay object hull markers on camera frames and optionally record to mp4."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class CameraHullRecorder(Node):
    def __init__(self) -> None:
        super().__init__("camera_hull_recorder")

        self.declare_parameter("image_topic", "/realsense/color/image_raw")
        self.declare_parameter("camera_info_topic", "/realsense/color/camera_info")
        self.declare_parameter("marker_topic", "/object_detector/markers")
        self.declare_parameter("annotated_image_topic", "~/annotated_image")
        self.declare_parameter("recording_service", "~/set_recording")
        self.declare_parameter("output_dir", "")
        self.declare_parameter("output_fps", 30.0)
        self.declare_parameter("line_thickness", 1)
        self.declare_parameter("line_b", 0)
        self.declare_parameter("line_g", 255)
        self.declare_parameter("line_r", 0)
        self.declare_parameter("object_id_whitelist", [])
        self.declare_parameter("auto_start_recording", False)

        self._image_topic = str(self.get_parameter("image_topic").value)
        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._marker_topic = str(self.get_parameter("marker_topic").value)
        self._annotated_image_topic = str(self.get_parameter("annotated_image_topic").value)
        self._recording_service_name = str(self.get_parameter("recording_service").value)
        self._output_dir_param = str(self.get_parameter("output_dir").value)
        self._output_fps = max(1.0, float(self.get_parameter("output_fps").value))
        self._line_thickness = max(1, int(self.get_parameter("line_thickness").value))
        self._line_color = (
            int(self.get_parameter("line_b").value),
            int(self.get_parameter("line_g").value),
            int(self.get_parameter("line_r").value),
        )
        self._whitelist = {str(x) for x in self.get_parameter("object_id_whitelist").value}
        self._auto_start = bool(self.get_parameter("auto_start_recording").value)

        self._bridge = CvBridge()
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._camera_frame = ""
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None
        self._camera_info_ready = False

        self._latest_markers: list[Marker] = []
        self._writer: cv2.VideoWriter | None = None
        self._recording_active = False
        self._pending_recording_start = self._auto_start
        self._output_path: Path | None = None

        self._image_sub = self.create_subscription(Image, self._image_topic, self._on_image, 10)
        self._camera_info_sub = self.create_subscription(CameraInfo, self._camera_info_topic, self._on_camera_info, 10)
        self._marker_sub = self.create_subscription(MarkerArray, self._marker_topic, self._on_markers, 10)
        self._annotated_pub = self.create_publisher(Image, self._annotated_image_topic, 10)
        self._recording_srv = self.create_service(SetBool, self._recording_service_name, self._on_set_recording)

        self.get_logger().info(
            "Camera hull recorder ready: "
            f"image={self._image_topic}, camera_info={self._camera_info_topic}, markers={self._marker_topic}, "
            f"service={self._recording_service_name}"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        if len(msg.k) < 9:
            return
        self._camera_frame = msg.header.frame_id
        self._fx = float(msg.k[0])
        self._fy = float(msg.k[4])
        self._cx = float(msg.k[2])
        self._cy = float(msg.k[5])
        self._camera_info_ready = True

    def _on_markers(self, msg: MarkerArray) -> None:
        kept: list[Marker] = []
        for marker in msg.markers:
            if marker.action in (Marker.DELETE, Marker.DELETEALL):
                continue
            if not self._whitelist:
                kept.append(marker)
                continue
            marker_key = f"{marker.ns}:{marker.id}"
            if (
                marker.ns in self._whitelist
                or marker.text in self._whitelist
                or str(marker.id) in self._whitelist
                or marker_key in self._whitelist
            ):
                kept.append(marker)
        self._latest_markers = kept

    def _on_set_recording(self, req: SetBool.Request, res: SetBool.Response) -> SetBool.Response:
        if req.data:
            if self._recording_active:
                res.success = True
                res.message = f"Already recording: {self._output_path}"
                return res
            self._pending_recording_start = True
            res.success = True
            res.message = "Recording will start on next frame"
            return res

        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._recording_active = False
        self._pending_recording_start = False
        res.success = True
        res.message = "Recording stopped"
        return res

    def _on_image(self, msg: Image) -> None:
        if not self._camera_info_ready:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to convert image: {exc}")
            return

        annotated = frame.copy()
        self._draw_marker_hulls(annotated, msg.header.frame_id, msg.header.stamp, frame.shape[1], frame.shape[0])

        try:
            self._annotated_pub.publish(self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to publish annotated image: {exc}")

        if self._pending_recording_start and not self._recording_active:
            self._start_writer(frame.shape[1], frame.shape[0])

        if self._recording_active and self._writer is not None:
            self._writer.write(annotated)

    def _start_writer(self, width: int, height: int) -> None:
        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = output_dir / f"camera_hull_overlay_{ts}.mp4"
        self._writer = cv2.VideoWriter(
            str(self._output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._output_fps,
            (width, height),
        )
        if not self._writer.isOpened():
            self.get_logger().error(f"Failed to open writer: {self._output_path}")
            self._writer = None
            self._recording_active = False
            self._pending_recording_start = False
            return
        self._recording_active = True
        self._pending_recording_start = False
        self.get_logger().info(f"Recording started: {self._output_path}")

    def _resolve_output_dir(self) -> Path:
        if self._output_dir_param:
            return Path(self._output_dir_param)
        workspace_root = Path(__file__).resolve().parents[4]
        return workspace_root / "runtime_logs"

    def _draw_marker_hulls(self, img: np.ndarray, image_frame: str, stamp, width: int, height: int) -> None:
        for marker in self._latest_markers:
            tf = self._lookup_transform(marker.header.frame_id, image_frame, stamp)
            if tf is None:
                continue

            drew_wireframe = self._draw_marker_wireframe(img, marker, tf, width, height)

            # Fallback for marker types without explicit line topology.
            if not drew_wireframe:
                points_marker = self._marker_points(marker)
                if len(points_marker) < 3:
                    continue
                pixels = self._project_points_with_tf(points_marker, tf, width, height)
                if len(pixels) < 3:
                    continue
                hull = cv2.convexHull(np.array(pixels, dtype=np.float32)).astype(np.int32)
                cv2.polylines(img, [hull], isClosed=True, color=self._line_color, thickness=self._line_thickness)

            label = marker.text if marker.text else f"{marker.ns}:{marker.id}"
            if label:
                anchor = self._marker_anchor_pixel(marker, tf, width, height)
                if anchor is not None:
                    cv2.putText(img, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._line_color, 1, cv2.LINE_AA)

    def _draw_marker_wireframe(
        self,
        img: np.ndarray,
        marker: Marker,
        tf: TransformStamped,
        width: int,
        height: int,
    ) -> bool:
        if not marker.points:
            return False

        if marker.type == Marker.LINE_LIST:
            drew = False
            for i in range(0, len(marker.points) - 1, 2):
                p0 = self._marker_point_to_pixel(marker, marker.points[i], tf, width, height)
                p1 = self._marker_point_to_pixel(marker, marker.points[i + 1], tf, width, height)
                if p0 is None or p1 is None:
                    continue
                cv2.line(img, p0, p1, self._line_color, self._line_thickness, cv2.LINE_AA)
                drew = True
            return drew

        if marker.type == Marker.LINE_STRIP:
            pixels = []
            for p in marker.points:
                px = self._marker_point_to_pixel(marker, p, tf, width, height)
                if px is not None:
                    pixels.append(px)
            if len(pixels) >= 2:
                cv2.polylines(img, [np.array(pixels, dtype=np.int32)], isClosed=False, color=self._line_color, thickness=self._line_thickness)
                return True
            return False

        if marker.type == Marker.TRIANGLE_LIST:
            drew = False
            for i in range(0, len(marker.points) - 2, 3):
                tri = [
                    self._marker_point_to_pixel(marker, marker.points[i + j], tf, width, height)
                    for j in range(3)
                ]
                if any(p is None for p in tri):
                    continue
                p0, p1, p2 = tri
                cv2.line(img, p0, p1, self._line_color, self._line_thickness, cv2.LINE_AA)
                cv2.line(img, p1, p2, self._line_color, self._line_thickness, cv2.LINE_AA)
                cv2.line(img, p2, p0, self._line_color, self._line_thickness, cv2.LINE_AA)
                drew = True
            return drew

        return False

    def _marker_anchor_pixel(self, marker: Marker, tf: TransformStamped, width: int, height: int) -> tuple[int, int] | None:
        if marker.points:
            for p in marker.points:
                px = self._marker_point_to_pixel(marker, p, tf, width, height)
                if px is not None:
                    return px
        center = self._transform_point(np.array([0.0, 0.0, 0.0], dtype=np.float64), tf)
        return self._project_point(center, width, height)

    def _marker_point_to_pixel(self, marker: Marker, point, tf: TransformStamped, width: int, height: int) -> tuple[int, int] | None:
        p_local = np.array([point.x, point.y, point.z], dtype=np.float64)
        p_marker = self._apply_pose(p_local, marker.pose.position, marker.pose.orientation)
        p_cam = self._transform_point(p_marker, tf)
        return self._project_point(p_cam, width, height)

    def _lookup_transform(self, source_frame: str, target_frame: str, stamp) -> TransformStamped | None:
        try:
            return self._tf_buffer.lookup_transform(target_frame, source_frame, stamp)
        except TransformException:
            try:
                return self._tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            except TransformException:
                return None

    def _project_point(self, p_cam: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
        z = p_cam[2]
        if z <= 0.0:
            return None
        u = (self._fx * p_cam[0] / z) + self._cx
        v = (self._fy * p_cam[1] / z) + self._cy
        if not (0.0 <= u < width and 0.0 <= v < height):
            return None
        return int(round(u)), int(round(v))

    def _project_points_with_tf(self, points: list[np.ndarray], tf: TransformStamped, width: int, height: int) -> list[tuple[float, float]]:
        pixels: list[tuple[float, float]] = []
        for p in points:
            p_cam = self._transform_point(p, tf)
            z = p_cam[2]
            if z <= 0.0:
                continue
            u = (self._fx * p_cam[0] / z) + self._cx
            v = (self._fy * p_cam[1] / z) + self._cy
            if 0.0 <= u < width and 0.0 <= v < height:
                pixels.append((u, v))
        return pixels

    def _marker_points(self, marker: Marker) -> list[np.ndarray]:
        points: list[np.ndarray] = []

        if marker.points:
            for p in marker.points:
                p_local = np.array([p.x, p.y, p.z], dtype=np.float64)
                points.append(self._apply_pose(p_local, marker.pose.position, marker.pose.orientation))
            return points

        if marker.type in (Marker.CUBE, Marker.SPHERE, Marker.CYLINDER):
            for p_local in self._primitive_points_from_marker(marker):
                points.append(self._apply_pose(p_local, marker.pose.position, marker.pose.orientation))

        return points

    def _primitive_points_from_marker(self, marker: Marker) -> list[np.ndarray]:
        if marker.type == Marker.CUBE:
            hx, hy, hz = marker.scale.x * 0.5, marker.scale.y * 0.5, marker.scale.z * 0.5
            return [
                np.array([sx * hx, sy * hy, sz * hz], dtype=np.float64)
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ]

        if marker.type == Marker.SPHERE:
            r = marker.scale.x * 0.5
            return [
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                np.array([r, 0.0, 0.0], dtype=np.float64),
                np.array([-r, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, r, 0.0], dtype=np.float64),
                np.array([0.0, -r, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, r], dtype=np.float64),
                np.array([0.0, 0.0, -r], dtype=np.float64),
            ]

        if marker.type == Marker.CYLINDER:
            h, r = marker.scale.z, marker.scale.x * 0.5
            hz = h * 0.5
            ring = [
                np.array([r * math.cos(a), r * math.sin(a), z], dtype=np.float64)
                for z in (-hz, hz)
                for a in np.linspace(0.0, 2.0 * math.pi, num=16, endpoint=False)
            ]
            return ring

        return []

    def _project_points(
        self,
        points: list[np.ndarray],
        source_frame: str,
        target_frame: str,
        stamp,
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        tf = self._lookup_transform(source_frame, target_frame, stamp)
        if tf is None:
            return []
        return self._project_points_with_tf(points, tf, width, height)

    @staticmethod
    def _transform_point(p: np.ndarray, tf: TransformStamped) -> np.ndarray:
        qx = tf.transform.rotation.x
        qy = tf.transform.rotation.y
        qz = tf.transform.rotation.z
        qw = tf.transform.rotation.w
        tx = tf.transform.translation.x
        ty = tf.transform.translation.y
        tz = tf.transform.translation.z

        rot = CameraHullRecorder._quat_to_rot(qx, qy, qz, qw)
        return rot @ p + np.array([tx, ty, tz], dtype=np.float64)

    @staticmethod
    def _apply_pose(p: np.ndarray, pos, ori) -> np.ndarray:
        rot = CameraHullRecorder._quat_to_rot(ori.x, ori.y, ori.z, ori.w)
        trans = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        return rot @ p + trans

    @staticmethod
    def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def destroy_node(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return super().destroy_node()


def main(args=None) -> int:
    rclpy.init(args=args)
    node = CameraHullRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
