"""
IRB120 Workspace Object Detector — SAM 2 backend
=================================================
Subscribes to the RealSense RGB image and aligned depth image, runs SAM 2 on
the RGB frame to segment objects visually, back-projects each mask into 3D
using the depth image and camera intrinsics, crops to the robot workspace,
then for each object computes:

  - 3D convex hull  (vertices + triangular faces)
  - Centroid        (geometry_msgs/Point in base_link)
  - Orientation     (PCA principal axes → quaternion, X = longest axis)

GPU required (~30-80 ms/frame on RTX 4070). Handles touching/adjacent objects
and complex shapes that the pure-geometry DBSCAN backend
(`object_detector_dbscan.py`) cannot separate.

Single camera only — unlike the DBSCAN backend, this node does not fuse
multiple camera viewpoints. SAM segments a single 2D RGB image; combining
masks from two independent camera views would require cross-view detection
association (matching the same object's mask in each view) rather than the
simple point-cloud concatenation the DBSCAN backend uses. Not implemented yet.

Table removal: handled by roi_z_min (set to known table height + margin).
No RANSAC needed since the table height is fixed.

Publishes:
  ~/detections     vision_msgs/Detection3DArray
  ~/markers        visualization_msgs/MarkerArray
  ~/object_points  sensor_msgs/PointCloud2 (x,y,z,label)

Dependencies:
  system pip: numpy, scipy
  pip (venv): sam2, torch (CUDA), opencv-python
  ROS:        sensor_msgs, vision_msgs, visualization_msgs, tf2_ros, cv_bridge

Launch under the venv python: ~/.venvs/.venv_torch_SAM/bin/python3
"""

import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Empty

import cv2
from cv_bridge import CvBridge

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM_OK = True
except ImportError:
    SAM_OK = False

from irb120_perception.perception_common import (
    ObjectDetectorBase, apply_tf, xyz_to_pointcloud2, voxel_downsample, remove_outliers,
)


class SAMObjectDetector(ObjectDetectorBase):

    def __init__(self):
        super().__init__('object_detector')

        # ---- SAM-specific parameters -------------------------------------------
        self.declare_parameter('input_cloud',        '/realsense/aligned_depth_to_color/image_raw')
        self.declare_parameter('input_image',        '/realsense/color/image_raw')
        self.declare_parameter('camera_info',        '/realsense/color/camera_info')
        self.declare_parameter('sam_weights',          '')
        self.declare_parameter('sam_config',           'configs/sam2.1/sam2.1_hiera_t.yaml')
        self.declare_parameter('sam_points_per_side',  16)
        self.declare_parameter('sam_iou_thresh',       0.80)
        self.declare_parameter('sam_min_mask_area',    500)   # pixels
        self.declare_parameter('sam_min_cluster_pts',  30)
        self.declare_parameter('sam_prominent_only',   True)
        # Stability / denoising params
        self.declare_parameter('depth_median_ksize',   5)     # depth blur kernel (0=off, odd int)
        self.declare_parameter('outlier_std_ratio',    2.0)   # statistical outlier removal threshold

        p = self.get_parameter
        self.sam_min_pts        = p('sam_min_cluster_pts').value
        self.sam_prominent_only = p('sam_prominent_only').value
        self.depth_median_ksize  = p('depth_median_ksize').value
        self.outlier_std_ratio   = p('outlier_std_ratio').value
        # EMA state for SAM path (centroid shift smoothing)
        self._smooth_centroid: np.ndarray | None = None
        self._smooth_verts:    np.ndarray | None = None

        # ---- Debug snapshot (on-demand, consumed by perception_debugger node) --
        # Trigger: ros2 topic pub --once /object_detector/sam_debug_snapshot std_msgs/Empty '{}'
        self._debug_requested = False
        self.create_subscription(Empty, '~/sam_debug_snapshot', self._debug_trigger_cb, 10)
        self._pub_dbg_mask_img  = self.create_publisher(Image,       '~/debug/sam_mask_overlay',    1)
        self._pub_dbg_pts_cam   = self.create_publisher(PointCloud2, '~/debug/sam_pts_camera',      1)
        self._pub_dbg_pts_roi   = self.create_publisher(PointCloud2, '~/debug/sam_pts_after_roi',   1)
        self._pub_dbg_pts_clean = self.create_publisher(PointCloud2, '~/debug/sam_pts_after_clean', 1)

        self._init_sam(p)
        self._latest_image    = None
        self._latest_depth    = None
        self._latest_cam_info = None
        self._bridge = CvBridge()

        # ---- QoS --------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            CameraInfo, p('camera_info').value, self._cam_info_cb, sensor_qos)
        self.create_subscription(
            Image, p('input_image').value, self._image_cb, sensor_qos)
        self.create_subscription(
            Image, p('input_cloud').value, self._depth_cb, sensor_qos)
        self.get_logger().info('object_detector ready [SAM 2]')

    # -------------------------------------------------------------------------
    # SAM initialisation
    # -------------------------------------------------------------------------

    def _init_sam(self, p):
        if not SAM_OK:
            self.get_logger().fatal(
                'torch/sam2 not available in this Python. '
                'Launch with the venv python: ~/.venvs/.venv_torch_SAM/bin/python3')
            raise RuntimeError('sam2 missing')

        weights = p('sam_weights').value
        cfg     = p('sam_config').value
        # Use GPU if available; automatic mask generator will be ~10× slower on CPU
        device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Loading SAM 2 on {device} …')

        model = build_sam2(cfg, weights, device=device)
        # SAM2AutomaticMaskGenerator runs a grid of prompts over the full image
        # and merges/filters the resulting masks by IoU and area thresholds.
        self._sam = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=p('sam_points_per_side').value,
            pred_iou_thresh=p('sam_iou_thresh').value,
            min_mask_region_area=p('sam_min_mask_area').value,
        )
        self.get_logger().info('SAM 2 loaded.')

    # -------------------------------------------------------------------------
    # Subscribers
    # -------------------------------------------------------------------------

    def _debug_trigger_cb(self, _: Empty):
        # Flag is checked on the next SAM frame and then cleared after publishing
        self._debug_requested = True
        self.get_logger().info('Debug snapshot requested — will publish on next SAM frame.')

    def _image_cb(self, msg: Image):
        # Store latest RGB frame; used by SAM on the next depth callback
        self._latest_image = msg

    def _depth_cb(self, msg: Image):
        # Store depth frame for reference, then immediately kick off SAM processing
        self._latest_depth = msg
        self._depth_cloud_cb(msg)

    def _cam_info_cb(self, msg: CameraInfo):
        # Intrinsics are stable after camera startup; stored once and reused every frame
        self._latest_cam_info = msg

    def _depth_cloud_cb(self, msg: Image):
        """Receives aligned depth image, runs SAM+back-projection."""
        t0 = time.monotonic()

        # Both RGB and camera_info must have arrived at least once before we can proceed
        if self._latest_image is None or self._latest_cam_info is None:
            self.get_logger().warn('Waiting for RGB image and camera_info',
                                   throttle_duration_sec=5.0)
            return

        # Look up depth_optical_frame→base_link at the latest available time
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return

        clusters = self._segment_sam_depth(msg, tf)

        if not clusters:
            # No objects found — reset all EMA state to avoid ghost smoothing
            self._smooth_centroid = None
            self._smooth_verts    = None
            self._reset_smoothing()
            self._publish_empty(msg.header)
            return

        # Apply per-frame centroid EMA smoothing before publishing
        clusters = self._smooth_clusters(clusters)
        self._publish_results(msg.header, clusters)
        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(f'{len(clusters)} object(s) [sam] in {dt:.1f} ms',
                               throttle_duration_sec=2.0)

    # -------------------------------------------------------------------------
    # Segmentation
    # -------------------------------------------------------------------------

    def _smooth_clusters(self, clusters: list) -> list:
        """
        EMA smooth the prominent cluster's centroid and point cloud across frames.
        Keeps the hull stable when SAM masks jitter slightly between frames.
        Only smooths the first (prominent) cluster; others pass through unchanged.
        """
        a = self.smooth_alpha
        pts = clusters[0]
        new_centroid = pts.mean(axis=0)

        if self._smooth_centroid is None:
            # First detection — initialise EMA with the raw values
            self._smooth_centroid = new_centroid
            self._smooth_verts    = pts
        else:
            # Blend the new centroid with the running EMA (alpha=1 → fully raw)
            self._smooth_centroid = a * new_centroid + (1 - a) * self._smooth_centroid

            # Smooth the point cloud by shifting it so its centroid matches the EMA centroid.
            # This damps positional drift without changing the hull shape.
            shift = self._smooth_centroid - new_centroid
            smoothed_pts = pts + shift

            # EMA on individual points is ill-defined across frames (different N),
            # so we store the shifted cloud directly — centroid is already smoothed.
            self._smooth_verts = smoothed_pts

        return [self._smooth_verts] + clusters[1:]

    def _segment_sam_depth(self, depth_msg: Image, tf) -> list:
        """
        SAM 2 on the latest RGB image.
        Each mask → back-project masked depth pixels to 3D using pinhole model
        → transform to base_link → ROI filter → cluster list.
        """
        # Convert RGB → SAM input
        bgr = self._bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Depth image: 16UC1 in mm, passthrough to preserve raw integer values
        depth_img = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        # Median blur kills salt-and-pepper depth noise without blurring edges
        k = self.depth_median_ksize
        if k > 1:
            depth_img = cv2.medianBlur(depth_img.astype(np.uint16), k).astype(np.float32)

        # Camera intrinsics (same for every mask in this frame)
        cam_k = self._latest_cam_info.k
        fx, fy = cam_k[0], cam_k[4]   # focal lengths in pixels
        cx, cy = cam_k[2], cam_k[5]   # principal point in pixels

        # Pixel index grids — computed once and reused for every mask
        h_img, w_img = depth_img.shape
        u, v = np.meshgrid(np.arange(w_img), np.arange(h_img))

        # Run SAM on RGB (no_grad avoids storing gradients, ~10-20% faster)
        with torch.no_grad():
            masks_data = self._sam.generate(rgb)
        self.get_logger().info(f'SAM generated {len(masks_data)} masks',
                               throttle_duration_sec=2.0)
        if not masks_data:
            return []

        # Back-projection constants: convert depth (mm→m) and unproject using
        # the standard pinhole model  X = (u-cx)*Z/fx,  Y = (v-cy)*Z/fy
        Z_full = depth_img / 1000.0
        X_full = (u - cx) * Z_full / fx
        Y_full = (v - cy) * Z_full / fy
        # Pixels with depth=0 are invalid (no return from sensor)
        valid_depth = depth_img > 0

        debug = self._debug_requested
        stamp = depth_msg.header.stamp
        cam_frame = depth_msg.header.frame_id

        m = self.roi
        clusters = []
        for md in masks_data:
            this_mask = md['segmentation']  # H×W bool

            # Combine the SAM binary mask with the valid-depth mask so we only
            # back-project pixels that both belong to the object and have depth
            valid = this_mask & valid_depth
            pts_cam = np.stack([X_full[valid], Y_full[valid], Z_full[valid]], axis=1).astype(np.float32)

            # Skip masks that don't cover enough depth pixels to form a cluster
            if len(pts_cam) < self.sam_min_pts:
                continue

            # Rotate and translate from camera frame into robot base_link
            pts_base = apply_tf(pts_cam, tf)

            # Crop to workspace bounding box — removes background and table surface
            roi_mask = (
                (pts_base[:,0] >= m['x'][0]) & (pts_base[:,0] <= m['x'][1]) &
                (pts_base[:,1] >= m['y'][0]) & (pts_base[:,1] <= m['y'][1]) &
                (pts_base[:,2] >= m['z'][0]) & (pts_base[:,2] <= m['z'][1])
            )
            pts_roi = pts_base[roi_mask]

            # After ROI crop the mask might now be too sparse to be a real object
            if len(pts_roi) < self.sam_min_pts:
                continue

            # Reduce point density then remove statistical outliers for a cleaner hull
            pts_vox  = voxel_downsample(pts_roi, self.voxel_size)
            pts_clean = remove_outliers(pts_vox, self.outlier_std_ratio)

            if len(pts_clean) >= self.sam_min_pts:
                # Store extra fields alongside the cleaned points for prominence
                # selection and debug publishing below
                clusters.append((pts_clean, md['area'], pts_cam, pts_roi, this_mask))

        self.get_logger().info(f'SAM clusters passing filter: {len(clusters)}',
                               throttle_duration_sec=2.0)

        if not clusters:
            return []

        # Pick the "prominent" object: highest mean Z in base_link.
        # Objects sitting on the table have a higher Z centroid than the table surface itself,
        # making this robust against SAM selecting the table/floor as the largest mask.
        if self.sam_prominent_only:
            clusters = [max(clusters, key=lambda c: c[0][:, 2].mean())]

        # Publish debug snapshot for the prominent cluster if requested.
        # The perception_debugger node subscribes to these topics and handles display/logging.
        if debug and clusters:
            self._debug_requested = False
            pts_clean, _, pts_cam_dbg, pts_roi_dbg, mask_2d = clusters[0]
            # Blend the SAM mask orange over the RGB image for visual confirmation
            overlay = rgb.copy()
            overlay[mask_2d] = (
                overlay[mask_2d] * 0.4 + np.array([255, 80, 0]) * 0.6
            ).clip(0, 255).astype(np.uint8)
            self._pub_dbg_mask_img.publish(
                self._bridge.cv2_to_imgmsg(
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), encoding='bgr8'))
            # Publish intermediate point clouds at each processing stage
            self._pub_dbg_pts_cam.publish(
                xyz_to_pointcloud2(pts_cam_dbg, cam_frame, stamp))
            self._pub_dbg_pts_roi.publish(
                xyz_to_pointcloud2(pts_roi_dbg, self.base_frame, stamp))
            self._pub_dbg_pts_clean.publish(
                xyz_to_pointcloud2(pts_clean, self.base_frame, stamp))

        # Return only the cleaned point arrays; metadata was only needed above
        return [c[0] for c in clusters]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = SAMObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
