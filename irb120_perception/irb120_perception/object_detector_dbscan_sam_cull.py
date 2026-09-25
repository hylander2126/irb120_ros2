"""DBSCAN -> SAM cull: use one prompted SAM mask to clean DBSCAN output.

This produces the object cloud the pipeline uses (launched by
perception.launch.py; irb120_control's perception snapshot reads its
~/object_points). It subscribes to
DBSCAN's published object cloud, projects that coarse 3-D result into cam1,
asks MobileSAM for one box-prompted mask, and publishes only source points
whose image pixels lie inside that mask.  Thus SAM is a boundary cull, not an
expensive segment-everything detector.
"""

import time

import numpy as np
import rclpy
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from irb120_perception.perception_common import ObjectDetectorBase, apply_tf, pointcloud2_to_xyz

try:
    from mobile_sam import SamPredictor, sam_model_registry
    _SAM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on optional venv
    SamPredictor = None
    sam_model_registry = None
    _SAM_IMPORT_ERROR = exc


def _image_to_rgb(msg: Image) -> np.ndarray:
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f"unsupported color encoding '{msg.encoding}'")
    row_stride = msg.step // 3
    image = buf.reshape(msg.height, row_stride, 3)[:, :msg.width, :]
    if msg.encoding == 'bgr8':
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


class DBSCANSAMCuller(ObjectDetectorBase):
    def __init__(self):
        super().__init__('object_detector_dbscan_sam_cull')
        self.declare_parameter('input_cloud', '/object_detector/object_points')
        self.declare_parameter('color_topic', '/realsense/color/image_raw')
        self.declare_parameter('camera_info_topic', '/realsense/color/camera_info')
        self.declare_parameter('sam_checkpoint', '')
        self.declare_parameter('sam_model_type', 'vit_t')
        self.declare_parameter('sam_device', 'cpu')
        self.declare_parameter('box_padding_px', 12)
        self.declare_parameter('min_cull_interval_sec', 2.0)

        p = self.get_parameter
        self.box_padding_px = int(p('box_padding_px').value)
        self.min_interval = float(p('min_cull_interval_sec').value)
        self._latest_color = None
        self._info = None
        self._last_run = None
        self._predictor = None

        checkpoint = p('sam_checkpoint').value
        if _SAM_IMPORT_ERROR is not None:
            self.get_logger().error(f'mobile_sam unavailable: {_SAM_IMPORT_ERROR}')
        elif not checkpoint:
            self.get_logger().error("'sam_checkpoint' is required for SAM culling")
        else:
            model = sam_model_registry[p('sam_model_type').value](checkpoint=checkpoint)
            model.to(device=p('sam_device').value)
            self._predictor = SamPredictor(model)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, p('color_topic').value, self._color_cb, qos)
        self.create_subscription(CameraInfo, p('camera_info_topic').value, self._info_cb, qos)
        self.create_subscription(PointCloud2, p('input_cloud').value, self._cloud_cb, qos)
        self.get_logger().info('DBSCAN SAM culler ready' if self._predictor else
                               'DBSCAN SAM culler ready (model not loaded)')

    def _color_cb(self, msg):
        self._latest_color = msg

    def _info_cb(self, msg):
        self._info = msg

    def _cloud_cb(self, msg):
        if not self._active or self._predictor is None or self._latest_color is None or self._info is None:
            return
        now = time.monotonic()
        if self._last_run is not None and now - self._last_run < self.min_interval:
            return
        self._last_run = now
        t0 = now
        source = pointcloud2_to_xyz(msg)
        if len(source) < 3:
            self._publish_empty(msg.header)
            return
        try:
            # DBSCAN normally publishes in base_link; retain support for any
            # compatible source cloud frame for standalone use.
            if msg.header.frame_id != self.base_frame:
                source_tf = self.tf_buffer.lookup_transform(
                    self.base_frame, msg.header.frame_id, rclpy.time.Time())
                source = apply_tf(source, source_tf)
            cam_tf = self.tf_buffer.lookup_transform(
                self.base_frame, self._latest_color.header.frame_id,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.3))
        except Exception as exc:
            self.get_logger().warn(f'TF lookup failed: {exc}')
            return
        kept = self._cull(source, self._latest_color, self._info, cam_tf)
        header = msg.header
        header.frame_id = self.base_frame
        if len(kept) < 3:
            self._reset_smoothing()
            self._publish_empty(header)
        else:
            self._publish_results(header, [kept])
        self.get_logger().info(
            f'DBSCAN→SAM cull: {len(source)} -> {len(kept)} points in '
            f'{(time.monotonic() - t0) * 1000:.0f} ms')

    def _cull(self, points_base, color_msg, info, cam_tf):
        color = _image_to_rgb(color_msg)
        t = cam_tf.transform.translation
        q = cam_tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ])
        # cam_tf maps camera -> base.  Invert it to project base points.
        points_cam = (points_base - np.array([t.x, t.y, t.z])) @ R
        valid_z = points_cam[:, 2] > 0.01
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        u = np.rint(points_cam[:, 0] * fx / points_cam[:, 2] + cx).astype(np.int32)
        v = np.rint(points_cam[:, 1] * fy / points_cam[:, 2] + cy).astype(np.int32)
        inside = valid_z & (u >= 0) & (u < color.shape[1]) & (v >= 0) & (v < color.shape[0])
        if inside.sum() < 3:
            return np.empty((0, 3), dtype=np.float32)
        pad = self.box_padding_px
        box = np.array([max(0, u[inside].min() - pad), max(0, v[inside].min() - pad),
                        min(color.shape[1] - 1, u[inside].max() + pad),
                        min(color.shape[0] - 1, v[inside].max() + pad)])
        self._predictor.set_image(color)
        masks, _, _ = self._predictor.predict(box=box, multimask_output=False)
        mask = masks[0]
        retain = inside.copy()
        indices = np.flatnonzero(inside)
        retain[indices] = mask[v[indices], u[indices]]
        return points_base[retain]


def main(args=None):
    rclpy.init(args=args)
    node = DBSCANSAMCuller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
