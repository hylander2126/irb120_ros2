"""DBSCAN -> SAM cull: use one prompted SAM mask to clean DBSCAN output.

This produces the object cloud the pipeline uses (launched by
perception.launch.py; irb120_control's perception snapshot reads its
~/object_points). It subscribes to
DBSCAN's published object cloud, projects that coarse 3-D result into cam1,
asks MobileSAM for one box-prompted mask, and publishes only source points
whose image pixels lie inside that mask.  Thus SAM is a boundary cull, not an
expensive segment-everything detector.

Optionally (review_* parameters, cam3 in perception.launch.py) the same DBSCAN
cloud also box-prompts a mask in a review camera, at the same time as the cam1
mask (each camera has its own predictor; the model weights are shared). That
mask culls nothing; it is published on ~/review_sam_* with the same cloud stamp
for the perception snapshot's forward-tip pivot fit (the far edge only cam3 sees).
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import time

import numpy as np
import rclpy
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import String

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
        self.declare_parameter('depth_topic', '/realsense/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/realsense/color/camera_info')
        self.declare_parameter('sam_checkpoint', '')
        self.declare_parameter('sam_model_type', 'vit_t')
        self.declare_parameter('sam_device', 'cpu')
        self.declare_parameter('box_padding_px', 12)
        self.declare_parameter('min_cull_interval_sec', 2.0)
        # Review camera: mask only, never culls. Empty color topic disables it.
        self.declare_parameter('review_camera', 'cam3')
        self.declare_parameter('review_color_topic', '')
        self.declare_parameter('review_depth_topic', '')
        self.declare_parameter('review_camera_info_topic', '')

        p = self.get_parameter
        self.box_padding_px = int(p('box_padding_px').value)
        self.min_interval = float(p('min_cull_interval_sec').value)
        self._latest_color = None
        self._latest_depth = None
        self._info = None
        self._last_run = None
        self._predictor = None
        self._review_predictor = None
        self._diagnostic_mask = None
        self._mask_pub = self.create_publisher(Image, "~/sam_mask", 10)
        self._color_pub = self.create_publisher(Image, "~/sam_color", 10)
        self._depth_pub = self.create_publisher(Image, "~/sam_depth", 10)
        self._metadata_pub = self.create_publisher(String, "~/sam_metadata", 10)
        self._review = None
        if p('review_color_topic').value:
            self._review = dict(name=p('review_camera').value, color=None, depth=None, info=None,
                                mask_pub=self.create_publisher(Image, "~/review_sam_mask", 10),
                                color_pub=self.create_publisher(Image, "~/review_sam_color", 10),
                                depth_pub=self.create_publisher(Image, "~/review_sam_depth", 10),
                                metadata_pub=self.create_publisher(String, "~/review_sam_metadata", 10))

        checkpoint = p('sam_checkpoint').value
        if _SAM_IMPORT_ERROR is not None:
            self.get_logger().error(f'mobile_sam unavailable: {_SAM_IMPORT_ERROR}')
        elif not checkpoint:
            self.get_logger().error("'sam_checkpoint' is required for SAM culling")
        else:
            model = sam_model_registry[p('sam_model_type').value](checkpoint=checkpoint)
            model.to(device=p('sam_device').value)
            self._predictor = SamPredictor(model)
            if self._review is not None:
                self._review_predictor = SamPredictor(model)

        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, p('color_topic').value, self._color_cb, qos)
        self.create_subscription(Image, p('depth_topic').value, self._depth_cb, qos)
        self.create_subscription(CameraInfo, p('camera_info_topic').value, self._info_cb, qos)
        self.create_subscription(PointCloud2, p('input_cloud').value, self._cloud_cb, qos)
        if self._review is not None:
            for key, msg_type, topic in (('color', Image, 'review_color_topic'),
                                         ('depth', Image, 'review_depth_topic'),
                                         ('info', CameraInfo, 'review_camera_info_topic')):
                if p(topic).value:
                    self.create_subscription(msg_type, p(topic).value,
                                             lambda m, k=key: self._review.__setitem__(k, m), qos)
        self.get_logger().info('DBSCAN SAM culler ready' if self._predictor else
                               'DBSCAN SAM culler ready (model not loaded)')

    def _color_cb(self, msg):
        self._latest_color = msg

    def _depth_cb(self, msg):
        self._latest_depth = msg

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
            self.get_logger().error(f'TF lookup failed, no cull: {exc}')
            return
        color_msg = self._latest_color
        # Grabbed here, before the ~seconds-long SAM call below, so its skew from
        # color_msg reflects two back-to-back reads of live topics, not whatever a
        # downstream consumer can still find in its own buffer once cull() returns.
        depth_msg = self._latest_depth
        self._diagnostic_mask = None
        review = self._review_inputs()
        with ThreadPoolExecutor(max_workers=2) as pool:
            cam1_job = pool.submit(self._cull, source, color_msg, self._info, cam_tf)
            review_job = None if review is None else pool.submit(
                self._box_mask, source, review[0], review[2], review[3], self._review_predictor)
            kept = cam1_job.result()
            review_boxed = None
            if review_job is not None:
                try:
                    review_boxed = review_job.result()
                    if review_boxed is None:
                        self.get_logger().error(f"{self._review['name']} review mask: cloud not in view")
                except Exception as exc:  # the cull stands without the review mask
                    self.get_logger().error(f"{self._review['name']} review mask failed: {exc}")
        # Pair diagnostic images by the input CLOUD stamp, just like object_points.
        # sam_color is the exact RGB input; its header stamp is an association ID.
        if self._diagnostic_mask is not None:
            mask_msg = Image()
            mask_msg.header = copy.deepcopy(color_msg.header)
            mask_msg.header.stamp = copy.deepcopy(msg.header.stamp)
            mask_msg.height, mask_msg.width = self._diagnostic_mask.shape
            mask_msg.encoding = 'mono8'
            mask_msg.step = mask_msg.width
            mask_msg.data = (self._diagnostic_mask.astype(np.uint8) * 255).tobytes()
            rgb_msg = copy.deepcopy(color_msg)
            rgb_msg.header.stamp = copy.deepcopy(msg.header.stamp)
            self._mask_pub.publish(mask_msg)
            self._color_pub.publish(rgb_msg)
            metadata = {
                'cloud_stamp': [msg.header.stamp.sec, msg.header.stamp.nanosec],
                'color_stamp': color_msg.header.stamp.sec + color_msg.header.stamp.nanosec*1e-9,
                'mask_source': 'live MobileSAM culler exact RGB input',
                'depth_paired_at_source': depth_msg is not None,
            }
            if review_boxed is not None:
                # Tells the snapshot a review mask for this cloud follows the cull output.
                metadata['review_camera'] = self._review['name']
            if depth_msg is not None:
                depth_stamp = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec*1e-9
                metadata['depth_stamp'] = depth_stamp
                metadata['rgb_depth_delta_s'] = abs(depth_stamp - metadata['color_stamp'])
                depth_out = copy.deepcopy(depth_msg)
                depth_out.header.stamp = copy.deepcopy(msg.header.stamp)
                self._depth_pub.publish(depth_out)
            self._metadata_pub.publish(String(data=json.dumps(metadata)))
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
        if review_boxed is not None:
            self._publish_review_mask(msg.header.stamp, review_boxed[0], review[0], review[1])

    def _review_inputs(self):
        """(color, depth, info, cam_tf) for the review camera, or None (logged) if unavailable."""
        review = getattr(self, '_review', None)
        if review is None or self._review_predictor is None:
            return None
        if review['color'] is None or review['info'] is None:
            self.get_logger().error(f"{review['name']} review mask: no color/camera_info received")
            return None
        color_msg = review['color']
        try:
            cam_tf = self.tf_buffer.lookup_transform(
                self.base_frame, color_msg.header.frame_id,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.3))
        except Exception as exc:
            self.get_logger().error(f"{review['name']} review mask: TF lookup failed: {exc}")
            return None
        return color_msg, review['depth'], review['info'], cam_tf

    def _publish_review_mask(self, cloud_stamp, mask, color_msg, depth_msg):
        """Publish the review-camera mask (from the DBSCAN cloud) under the cloud stamp."""
        review = self._review
        mask_msg = Image()
        mask_msg.header = copy.deepcopy(color_msg.header)
        mask_msg.header.stamp = copy.deepcopy(cloud_stamp)
        mask_msg.height, mask_msg.width = mask.shape
        mask_msg.encoding = 'mono8'
        mask_msg.step = mask_msg.width
        mask_msg.data = (mask.astype(np.uint8) * 255).tobytes()
        rgb_msg = copy.deepcopy(color_msg)
        rgb_msg.header.stamp = copy.deepcopy(cloud_stamp)
        metadata = {
            'camera': review['name'],
            'cloud_stamp': [cloud_stamp.sec, cloud_stamp.nanosec],
            'color_stamp': color_msg.header.stamp.sec + color_msg.header.stamp.nanosec*1e-9,
            'mask_source': 'live MobileSAM review mask, box from the DBSCAN cloud',
            'depth_paired_at_source': depth_msg is not None,
        }
        if depth_msg is not None:
            metadata['depth_stamp'] = depth_msg.header.stamp.sec + depth_msg.header.stamp.nanosec*1e-9
            metadata['rgb_depth_delta_s'] = abs(metadata['depth_stamp'] - metadata['color_stamp'])
            depth_out = copy.deepcopy(depth_msg)
            depth_out.header.stamp = copy.deepcopy(cloud_stamp)
            review['depth_pub'].publish(depth_out)
        review['mask_pub'].publish(mask_msg)
        review['color_pub'].publish(rgb_msg)
        review['metadata_pub'].publish(String(data=json.dumps(metadata)))
        self.get_logger().info(f"{review['name']} review mask published")

    def _cull(self, points_base, color_msg, info, cam_tf):
        boxed = self._box_mask(points_base, color_msg, info, cam_tf)
        if boxed is None:
            return np.empty((0, 3), dtype=np.float32)
        mask, u, v, inside = boxed
        self._diagnostic_mask = mask
        retain = inside.copy()
        indices = np.flatnonzero(inside)
        retain[indices] = mask[v[indices], u[indices]]
        return points_base[retain]

    def _box_mask(self, points_base, color_msg, info, cam_tf, predictor=None):
        """One SAM mask prompted by the points' padded image box: (mask, u, v, inside) or None.
        `predictor` defaults to cam1's; each camera needs its own (set_image keeps state)."""
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
            return None
        pad = self.box_padding_px
        box = np.array([max(0, u[inside].min() - pad), max(0, v[inside].min() - pad),
                        min(color.shape[1] - 1, u[inside].max() + pad),
                        min(color.shape[0] - 1, v[inside].max() + pad)])
        predictor = self._predictor if predictor is None else predictor
        predictor.set_image(color)
        masks, _, _ = predictor.predict(box=box, multimask_output=False)
        return masks[0], u, v, inside


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
