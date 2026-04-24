"""
IRB120 Workspace Object Detector
=================================
Subscribes to the RealSense pointcloud and (optionally) RGB image,
crops to the robot workspace, segments objects, then for each object
computes:

  - 3D convex hull  (vertices + triangular faces)
  - Centroid        (geometry_msgs/Point in base_link)
  - Orientation     (PCA principal axes → quaternion, X = longest axis)

Two segmentation backends, selected by the 'segmentation_method' param:

  'dbscan'  — pure geometry, fast, no GPU needed
              Cluster remaining points spatially with DBSCAN.
              Works well when objects are separated by a gap.
              Fails when objects touch or have similar depth.

  'sam'     — SAM 2 on the RGB image, GPU required
              Segment objects visually, back-project masks into 3D.
              Handles touching/adjacent objects and complex shapes.
              ~30-80 ms/frame on RTX 4070.

Table removal: handled by roi_z_min (set to known table height + margin).
No RANSAC needed since the table height is fixed.

Publishes:
  ~/detections   vision_msgs/Detection3DArray
  ~/markers      visualization_msgs/MarkerArray

Dependencies:
  system pip: numpy, scikit-learn, scipy
  pip (venv): sam2, torch (CUDA), opencv-python
  ROS:        sensor_msgs, vision_msgs, visualization_msgs, tf2_ros, cv_bridge
"""

import struct
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import ColorRGBA, Empty
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

try:
    import cv2
    from cv_bridge import CvBridge
    CV_OK = True
except ImportError:
    CV_OK = False

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM_OK = True
except ImportError:
    SAM_OK = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def pointcloud2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract (N,3) float32 XYZ from a PointCloud2 message."""
    # Build a lookup from field name → field descriptor so we can find x/y/z byte offsets
    fields = {f.name: f for f in msg.fields}
    ox, oy, oz = fields['x'].offset, fields['y'].offset, fields['z'].offset
    step = msg.point_step   # bytes per point
    n = msg.width * msg.height
    endian = '>' if msg.is_bigendian else '<'
    # Fast path: data is contiguous and fields are large enough to read safely
    contiguous = msg.row_step == step * msg.width

    if contiguous and step >= max(ox, oy, oz) + 4:
        # Build a structured dtype that maps directly onto the raw byte buffer,
        # letting numpy extract x/y/z columns without any Python loop.
        dtype = np.dtype({
            'names': ['x', 'y', 'z'],
            'formats': [endian + 'f4', endian + 'f4', endian + 'f4'],
            'offsets': [ox, oy, oz],
            'itemsize': step,
        })
        view = np.frombuffer(msg.data, dtype=dtype, count=n)
        xyz = np.empty((n, 3), dtype=np.float32)
        xyz[:, 0] = view['x']
        xyz[:, 1] = view['y']
        xyz[:, 2] = view['z']
    else:
        # Slow path: non-contiguous or unusual layout — unpack point by point
        data = msg.data
        xyz = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            b = i * step
            xyz[i, 0] = struct.unpack_from('f', data, b + ox)[0]
            xyz[i, 1] = struct.unpack_from('f', data, b + oy)[0]
            xyz[i, 2] = struct.unpack_from('f', data, b + oz)[0]
    # Drop NaN/Inf points (invalid depth returns from the sensor)
    return xyz[np.isfinite(xyz).all(axis=1)]


def apply_tf(pts: np.ndarray, tf) -> np.ndarray:
    """Apply a TransformStamped to (N,3) array."""
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    # Convert quaternion to 3×3 rotation matrix
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    # Rotate all points, then translate: p_out = R·p + t
    return (R @ pts.T).T + np.array([t.x, t.y, t.z])


def rotation_to_quaternion(R: np.ndarray):
    """3×3 rotation matrix → (x,y,z,w) quaternion."""
    # Shepperd's method: branch on the largest diagonal element to avoid
    # division by near-zero when the corresponding component is small.
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s


def pca_orientation(pts: np.ndarray, prev_axes: np.ndarray | None = None):
    """PCA → (centroid, (qx,qy,qz,qw), axes_3x3) where X aligns with longest axis.

    If prev_axes (3×3, columns = previous frame's principal axes) is provided,
    each axis is sign-flipped to be consistent with the previous frame rather
    than pinned to world directions. This eliminates jitter flips while still
    tracking genuine object reorientations caused by robot interaction.
    """
    centroid = pts.mean(axis=0)
    # SVD of the mean-centred cloud; right singular vectors (rows of Vt) are
    # the principal axes sorted by descending variance.
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    R = Vt.T  # columns are principal axes, descending variance

    if prev_axes is not None:
        # Flip each axis independently to match the previous frame's direction.
        # A genuine reorientation (e.g. robot tilts the object) still registers
        # because the dot product only resolves the 180° sign ambiguity, not the
        # actual angle between frames.
        for i in range(3):
            if np.dot(R[:, i], prev_axes[:, i]) < 0:
                R[:, i] *= -1
        # Re-enforce right-handedness after independent per-axis flips
        R[:, 2] = np.cross(R[:, 0], R[:, 1])

    # Ensure det(R) = +1 (proper rotation, not a reflection)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    return centroid, rotation_to_quaternion(R), R


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two (x,y,z,w) quaternions."""
    # Negate q1 if needed so we always interpolate along the shorter arc
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    # When quaternions are nearly identical, fall back to normalised lerp
    # to avoid division by sin(~0)
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta = np.arccos(dot)
    return (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)


def convex_hull_scipy(pts: np.ndarray):
    """Scipy convex hull → (vertices Nx3, triangles Mx3) or (None, None)."""
    try:
        hull = ConvexHull(pts.astype(np.float64))
        # hull.vertices: indices into pts of hull boundary points only
        verts = pts[hull.vertices]
        # hull.simplices indexes into pts; remap to the compacted verts array
        idx_map = {old: new for new, old in enumerate(hull.vertices)}
        tris = np.array([[idx_map[i] for i in tri] for tri in hull.simplices])
        return verts, tris
    except Exception:
        # ConvexHull raises if pts are degenerate (coplanar, < 4 points, etc.)
        return None, None


def xyz_to_pointcloud2(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Pack an (N,3) float32 array into a PointCloud2 message."""
    pts = pts.astype(np.float32)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(pts)
    msg.is_dense = False
    msg.is_bigendian = False
    msg.point_step = 12  # 3 × float32
    msg.row_step = msg.point_step * len(pts)
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = pts.tobytes()
    return msg


def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """Reduce point density: keep one point per voxel cell."""
    # Assign each point to a voxel by flooring its coordinates
    idx = np.floor(pts / voxel_size).astype(np.int32)
    # np.unique on rows gives one representative index per unique voxel
    _, unique = np.unique(idx, axis=0, return_index=True)
    return pts[unique]


def remove_outliers(pts: np.ndarray, std_ratio: float) -> np.ndarray:
    """Remove points further than std_ratio * std from the centroid."""
    if len(pts) < 4:
        return pts
    dists = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    # Keep only points within mean + N*std of the centroid distance distribution
    return pts[dists < dists.mean() + std_ratio * dists.std()]


def label_color(idx: int) -> ColorRGBA:
    # Fixed palette cycles across detected objects for consistent RViz colours
    palette = [
        (0.92, 0.26, 0.21), (0.13, 0.59, 0.95), (0.30, 0.69, 0.31),
        (1.00, 0.76, 0.03), (0.61, 0.15, 0.69), (0.01, 0.74, 0.83),
    ]
    r, g, b = palette[idx % len(palette)]
    return ColorRGBA(r=r, g=g, b=b, a=0.6)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        # ---- Parameters -------------------------------------------------------
        # self.declare_parameter('input_cloud',        '/realsense/depth/color/points')
        self.declare_parameter('input_cloud',        '/realsense/aligned_depth_to_color/image_raw') # Depth image over pointcloud (align to rgb)
        self.declare_parameter('input_image',        '/realsense/color/image_raw')
        self.declare_parameter('camera_info',        '/realsense/color/camera_info')
        self.declare_parameter('base_frame',         'base_link')
        self.declare_parameter('segmentation_method', 'dbscan')   # 'dbscan' | 'sam'
        self.declare_parameter('roi_x_min',   0.15)
        self.declare_parameter('roi_x_max',   0.80)
        self.declare_parameter('roi_y_min',  -0.25)
        self.declare_parameter('roi_y_max',   0.25)
        self.declare_parameter('roi_z_min',  -0.02)
        self.declare_parameter('roi_z_max',   0.50)
        self.declare_parameter('voxel_size',  0.005)
        # DBSCAN params
        self.declare_parameter('dbscan_eps',      0.02)
        self.declare_parameter('dbscan_min_pts',  20)
        self.declare_parameter('min_cluster_pts', 30)
        self.declare_parameter('max_cluster_pts', 50000)
        # SAM params
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
        self.declare_parameter('smooth_alpha',         0.3)   # EMA weight for new frame (0=frozen, 1=raw)
        # Robot self-masking: exclude points near these TF link frames
        self.declare_parameter('robot_mask_links',  [
            'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6',
            'flange', 'ft_link', 'finger_link', 'finger_ball_center',
        ])
        self.declare_parameter('robot_mask_padding', 0.08)    # exclusion sphere radius (m)

        p = self.get_parameter
        self.base_frame  = p('base_frame').value
        self.method      = p('segmentation_method').value
        self.roi = dict(
            x=(p('roi_x_min').value, p('roi_x_max').value),
            y=(p('roi_y_min').value, p('roi_y_max').value),
            z=(p('roi_z_min').value, p('roi_z_max').value),
        )
        self.voxel_size       = p('voxel_size').value
        self.dbscan_eps       = p('dbscan_eps').value
        self.dbscan_min_pts   = p('dbscan_min_pts').value
        self.min_pts          = p('min_cluster_pts').value
        self.max_pts          = p('max_cluster_pts').value
        self.sam_min_pts        = p('sam_min_cluster_pts').value
        self.sam_prominent_only = p('sam_prominent_only').value
        self.depth_median_ksize  = p('depth_median_ksize').value
        self.outlier_std_ratio   = p('outlier_std_ratio').value
        self.smooth_alpha        = p('smooth_alpha').value
        self.robot_mask_links    = p('robot_mask_links').value
        self.robot_mask_padding  = p('robot_mask_padding').value
        # EMA state for SAM path (centroid shift smoothing)
        self._smooth_centroid: np.ndarray | None = None
        self._smooth_verts:    np.ndarray | None = None
        # EMA state for orientation + centroid smoothing (both backends)
        # Keyed by obj_id; cleared when detection is absent
        self._smooth_q:    dict[int, np.ndarray] = {}  # quaternion (x,y,z,w)
        self._smooth_pos:  dict[int, np.ndarray] = {}  # centroid (3,)
        self._smooth_axes: dict[int, np.ndarray] = {}  # 3×3 principal axes (cols)

        # ---- TF ---------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Publishers -------------------------------------------------------
        self.pub_det = self.create_publisher(Detection3DArray, '~/detections', 10)
        self.pub_mk  = self.create_publisher(MarkerArray,      '~/markers',    10)

        # ---- Debug snapshot (on-demand, consumed by perception_debugger node) --
        # Trigger: ros2 topic pub --once /object_detector/debug_snapshot std_msgs/Empty '{}'
        self._debug_requested = False
        self.create_subscription(Empty, '~/debug_snapshot', self._debug_trigger_cb, 10)
        self._pub_dbg_mask_img  = self.create_publisher(Image,       '~/debug/mask_overlay',    1)
        self._pub_dbg_pts_cam   = self.create_publisher(PointCloud2, '~/debug/pts_camera',      1)
        self._pub_dbg_pts_roi   = self.create_publisher(PointCloud2, '~/debug/pts_after_roi',   1)
        self._pub_dbg_pts_clean = self.create_publisher(PointCloud2, '~/debug/pts_after_clean', 1)

        # ---- QoS --------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- Backend setup ----------------------------------------------------
        if self.method == 'sam':
            self._init_sam(p)
            self._latest_image    = None
            self._latest_depth    = None
            self._latest_cam_info = None
            self._bridge = CvBridge()
            self.create_subscription(
                CameraInfo, p('camera_info').value, self._cam_info_cb, sensor_qos)
            self.create_subscription(
                Image, p('input_image').value, self._image_cb, sensor_qos)
            self.create_subscription(
                Image, p('input_cloud').value, self._depth_cb, sensor_qos)
            self.get_logger().info('object_detector ready [SAM 2]')
        else:
            self.declare_parameter('input_cloud_pc', '/realsense/depth/color/points')
            self.create_subscription(
                PointCloud2, p('input_cloud_pc').value, self._cloud_cb, sensor_qos)
            self.get_logger().info('object_detector ready [DBSCAN]')

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

    def _cloud_cb(self, msg: PointCloud2):
        """DBSCAN path: receives a PointCloud2, transforms to base_link, segments."""
        t0 = time.monotonic()

        pts_cam = pointcloud2_to_xyz(msg)
        if pts_cam.shape[0] == 0:
            return

        # Look up camera→base_link transform at the latest available time
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return

        # Rotate and translate all points into the robot base_link frame
        pts_base = apply_tf(pts_cam, tf)

        # Crop to the configured workspace bounding box (removes table, walls, etc.)
        m = self.roi
        mask = (
            (pts_base[:,0] >= m['x'][0]) & (pts_base[:,0] <= m['x'][1]) &
            (pts_base[:,1] >= m['y'][0]) & (pts_base[:,1] <= m['y'][1]) &
            (pts_base[:,2] >= m['z'][0]) & (pts_base[:,2] <= m['z'][1])
        )
        pts_roi = pts_base[mask]

        # Not enough points to form even one cluster — publish empty and bail
        if pts_roi.shape[0] < self.min_pts:
            self._publish_empty(msg.header)
            return

        clusters = self._segment_dbscan(pts_roi)

        if not clusters:
            # No clusters found — reset EMA state so stale smoothing doesn't
            # carry over to the next detection
            self._smooth_q.clear()
            self._smooth_pos.clear()
            self._smooth_axes.clear()
            self._publish_empty(msg.header)
            return

        self._publish_results(msg.header, clusters)
        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(f'{len(clusters)} object(s) [dbscan] in {dt:.1f} ms',
                               throttle_duration_sec=2.0)

    def _depth_cloud_cb(self, msg: Image):
        """SAM path: receives aligned depth image, runs SAM+back-projection."""
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
            self._smooth_q.clear()
            self._smooth_pos.clear()
            self._smooth_axes.clear()
            self._publish_empty(msg.header)
            return

        # Apply per-frame centroid EMA smoothing before publishing
        clusters = self._smooth_clusters(clusters)
        self._publish_results(msg.header, clusters)
        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(f'{len(clusters)} object(s) [sam] in {dt:.1f} ms',
                               throttle_duration_sec=2.0)

    # -------------------------------------------------------------------------
    # Segmentation backends
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

    def _segment_dbscan(self, pts_roi: np.ndarray):
        """DBSCAN on the ROI pointcloud. Returns list of (N,3) arrays."""
        # Voxel downsample: keep one point per grid cell to reduce density and
        # make DBSCAN runtime independent of sensor resolution
        voxel_idx = np.floor(pts_roi / self.voxel_size).astype(np.int32)
        _, unique = np.unique(voxel_idx, axis=0, return_index=True)
        pts_down = pts_roi[unique]

        if len(pts_down) < self.min_pts:
            return []

        # DBSCAN groups nearby points into clusters; label=-1 means noise/outlier
        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_pts,
        ).fit_predict(pts_down)

        # Collect each cluster, filtering by size to exclude noise blobs and
        # degenerate single-point hits
        clusters = []
        for lbl in set(labels) - {-1}:
            c = pts_down[labels == lbl]
            if self.min_pts <= len(c) <= self.max_pts:
                clusters.append(c)
        return clusters

    def _segment_sam_points(self, cloud_msg: PointCloud2,
                     pts_cam: np.ndarray,
                     pts_roi: np.ndarray,
                     idx_roi: np.ndarray):
        """
        SAM 2 on the latest RGB image.
        Each mask → select matching 3D points from ROI → cluster.

        ROI is "Region of Interest"

        The pointcloud and image share the same pixel grid:
          point index i  ↔  pixel (row = i // width, col = i % width)
        """
        if self._latest_image is None:
            self.get_logger().warn('No RGB image received yet', throttle_duration_sec=5.0)
            return []

        # Convert ROS image → numpy BGR → RGB
        bgr = self._bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Run SAM — returns a list of mask dicts, each with 'segmentation' (H×W bool)
        masks_data = self._sam.generate(rgb)
        self.get_logger().info(f'SAM generated {len(masks_data)} masks, ROI pts: {len(pts_roi)}',
                               throttle_duration_sec=2.0)
        if not masks_data:
            return []

        w = cloud_msg.width
        # Derive pixel (row, col) for each ROI point from its linear index in the cloud.
        # This works because the aligned depth cloud has the same pixel layout as the image.
        rows = idx_roi // w
        cols = idx_roi % w

        clusters = []
        for md in masks_data:
            mask_2d = md['segmentation']  # H×W bool array
            h_img, w_img = mask_2d.shape

            # Guard against the rare case where cloud and image resolutions differ
            valid = (rows < h_img) & (cols < w_img)
            in_mask = np.zeros(len(idx_roi), dtype=bool)
            # Select the ROI points whose pixel falls inside this SAM mask
            in_mask[valid] = mask_2d[rows[valid], cols[valid]]

            cluster = pts_roi[in_mask]
            if len(cluster) >= self.sam_min_pts:
                clusters.append(cluster)

        self.get_logger().info(f'SAM clusters passing filter: {len(clusters)}',
                               throttle_duration_sec=2.0)
        return clusters


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

    # -------------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------------

    def _publish_results(self, header, clusters):
        detections = Detection3DArray()
        detections.header.stamp    = header.stamp
        detections.header.frame_id = self.base_frame
        markers = MarkerArray()

        # Delete all previous markers before adding new ones so stale hulls don't linger
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for obj_id, pts in enumerate(clusters):
            # Compute PCA orientation, passing previous axes to resolve sign ambiguity
            prev_axes = self._smooth_axes.get(obj_id, None)
            centroid, (qx, qy, qz, qw), axes = pca_orientation(pts, prev_axes)
            self._smooth_axes[obj_id] = axes

            # EMA smooth centroid position and orientation (both backends).
            # Uses SLERP for quaternion so it stays normalized and takes the
            # shortest arc — prevents the 360° spin that lerp can cause.
            a = self.smooth_alpha
            q_raw = np.array([qx, qy, qz, qw], dtype=np.float64)
            if obj_id in self._smooth_q:
                centroid = a * centroid + (1 - a) * self._smooth_pos[obj_id]
                q_raw    = slerp_quaternion(self._smooth_q[obj_id], q_raw, a)
            self._smooth_pos[obj_id] = centroid
            self._smooth_q[obj_id]   = q_raw
            qx, qy, qz, qw = q_raw

            # Axis-aligned bounding box size from the raw (non-hull) point cloud
            mins, maxs = pts.min(axis=0), pts.max(axis=0)
            size = maxs - mins
            # Convex hull for wireframe visualisation
            verts, tris = convex_hull_scipy(pts)
            color = label_color(obj_id)
            stamp = header.stamp
            frame = self.base_frame

            # --- Detection3D message ---
            det = Detection3D()
            det.header = detections.header
            det.id = str(obj_id)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'object'
            hyp.hypothesis.score    = 1.0
            # Pose carries both position and orientation in base_link
            hyp.pose.pose.position.x = float(centroid[0])
            hyp.pose.pose.position.y = float(centroid[1])
            hyp.pose.pose.position.z = float(centroid[2])
            hyp.pose.pose.orientation.x = float(qx)
            hyp.pose.pose.orientation.y = float(qy)
            hyp.pose.pose.orientation.z = float(qz)
            hyp.pose.pose.orientation.w = float(qw)
            det.results.append(hyp)
            # Bounding box duplicates pose + AABB size for consumers that use bbox directly
            det.bbox.center.position.x = float(centroid[0])
            det.bbox.center.position.y = float(centroid[1])
            det.bbox.center.position.z = float(centroid[2])
            det.bbox.center.orientation.x = float(qx)
            det.bbox.center.orientation.y = float(qy)
            det.bbox.center.orientation.z = float(qz)
            det.bbox.center.orientation.w = float(qw)
            det.bbox.size.x = float(size[0])
            det.bbox.size.y = float(size[1])
            det.bbox.size.z = float(size[2])
            detections.detections.append(det)

            # --- RViz markers ---

            # Hull wireframe: each triangle edge emitted as a LINE_LIST pair
            if verts is not None:
                markers.markers.append(
                    self._mk_hull(obj_id, stamp, frame, verts, tris, color))

            # Centroid sphere
            markers.markers.append(
                self._mk_centroid(obj_id, stamp, frame, centroid, color))

            # PCA axes arrows (R=X longest, G=Y, B=Z shortest)
            R_mat = self._quat_to_mat(qx, qy, qz, qw)
            axis_colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9),
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9),
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9),
            ]
            for ai, ac in enumerate(axis_colors):
                # Scale each arrow to half the object extent along that axis
                markers.markers.append(
                    self._mk_axis(obj_id*10+ai+100, stamp, frame,
                                  centroid, R_mat[:,ai], float(size[ai])*0.5, ac))

        self.pub_det.publish(detections)
        self.pub_mk.publish(markers)

    def _publish_empty(self, header):
        # Publish zero-detection array and a DELETEALL marker to clear RViz
        d = Detection3DArray()
        d.header.stamp    = header.stamp
        d.header.frame_id = self.base_frame
        self.pub_det.publish(d)
        mk = MarkerArray()
        clr = Marker()
        clr.action = Marker.DELETEALL
        mk.markers.append(clr)
        self.pub_mk.publish(mk)

    # -------------------------------------------------------------------------
    # Marker builders
    # -------------------------------------------------------------------------

    def _mk_hull(self, obj_id, stamp, frame, verts, tris, color):
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame
        m.ns = 'hull'; m.id = obj_id
        m.type = Marker.LINE_LIST; m.action = Marker.ADD
        m.scale.x = 0.002
        m.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.8)
        m.pose.orientation.w = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=3.0).to_msg()
        # Each triangle contributes 3 edges; each edge is a start+end point pair
        for tri in tris:
            for i in range(3):
                a, b = verts[tri[i]], verts[tri[(i+1)%3]]
                m.points.append(Point(x=a[0], y=a[1], z=a[2]))
                m.points.append(Point(x=b[0], y=b[1], z=b[2]))
        return m

    def _mk_centroid(self, obj_id, stamp, frame, centroid, color):
        m = Marker()
        m.header.stamp = stamp; m.header.frame_id = frame
        m.ns = 'centroid'; m.id = obj_id
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = float(centroid[0])
        m.pose.position.y = float(centroid[1])
        m.pose.position.z = float(centroid[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.02
        m.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=1.0)
        m.lifetime = rclpy.duration.Duration(seconds=3.0).to_msg()
        return m

    def _mk_axis(self, marker_id, stamp, frame, origin, axis, scale, color):
        m = Marker()
        m.header.stamp = stamp; m.header.frame_id = frame
        m.ns = 'axes'; m.id = marker_id
        m.type = Marker.ARROW; m.action = Marker.ADD
        # scale.x = shaft diameter, scale.y = head diameter, scale.z = head length
        m.scale.x = 0.005; m.scale.y = 0.010; m.scale.z = 0.015
        m.color = color
        m.lifetime = rclpy.duration.Duration(seconds=3.0).to_msg()
        # ARROW with two points: tail at origin, tip at origin + axis*scale
        m.points = [
            Point(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
            Point(x=float(origin[0]+axis[0]*scale),
                  y=float(origin[1]+axis[1]*scale),
                  z=float(origin[2]+axis[2]*scale)),
        ]
        m.pose.orientation.w = 1.0
        return m

    @staticmethod
    def _quat_to_mat(qx, qy, qz, qw):
        # Standard quaternion-to-rotation-matrix formula
        x, y, z, w = qx, qy, qz, qw
        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
