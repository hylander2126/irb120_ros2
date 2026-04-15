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
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import ColorRGBA
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
    fields = {f.name: f for f in msg.fields}
    ox, oy, oz = fields['x'].offset, fields['y'].offset, fields['z'].offset
    step = msg.point_step
    data = msg.data
    n = msg.width * msg.height
    xyz = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        b = i * step
        xyz[i, 0] = struct.unpack_from('f', data, b + ox)[0]
        xyz[i, 1] = struct.unpack_from('f', data, b + oy)[0]
        xyz[i, 2] = struct.unpack_from('f', data, b + oz)[0]
    return xyz[np.isfinite(xyz).all(axis=1)]


def apply_tf(pts: np.ndarray, tf) -> np.ndarray:
    """Apply a TransformStamped to (N,3) array."""
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    return (R @ pts.T).T + np.array([t.x, t.y, t.z])


def rotation_to_quaternion(R: np.ndarray):
    """3×3 rotation matrix → (x,y,z,w) quaternion."""
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


def pca_orientation(pts: np.ndarray):
    """PCA → (centroid, (qx,qy,qz,qw)) where X aligns with longest axis."""
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    R = Vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return centroid, rotation_to_quaternion(R)


def convex_hull_scipy(pts: np.ndarray):
    """Scipy convex hull → (vertices Nx3, triangles Mx3) or (None, None)."""
    try:
        hull = ConvexHull(pts.astype(np.float64))
        return pts[hull.vertices], hull.simplices
    except Exception:
        return None, None


def label_color(idx: int) -> ColorRGBA:
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
        self.declare_parameter('input_cloud',        '/realsense/depth/color/points')
        self.declare_parameter('input_image',        '/realsense/color/image_raw')
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

        p = self.get_parameter
        self.base_frame  = p('base_frame').value
        self.method      = p('segmentation_method').value
        self.roi = dict(
            x=(p('roi_x_min').value, p('roi_x_max').value),
            y=(p('roi_y_min').value, p('roi_y_max').value),
            z=(p('roi_z_min').value, p('roi_z_max').value),
        )
        self.voxel_size     = p('voxel_size').value
        self.dbscan_eps     = p('dbscan_eps').value
        self.dbscan_min_pts = p('dbscan_min_pts').value
        self.min_pts        = p('min_cluster_pts').value
        self.max_pts        = p('max_cluster_pts').value
        self.sam_min_pts    = p('sam_min_cluster_pts').value

        # ---- TF ---------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Publishers -------------------------------------------------------
        self.pub_det = self.create_publisher(Detection3DArray, '~/detections', 10)
        self.pub_mk  = self.create_publisher(MarkerArray,      '~/markers',    10)

        # ---- QoS --------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- Backend setup ----------------------------------------------------
        if self.method == 'sam':
            self._init_sam(p)
            # Need both cloud and image — sync via latest-stamp cache
            self._latest_image = None
            self._bridge = CvBridge()
            self.create_subscription(
                Image, p('input_image').value, self._image_cb, sensor_qos)
            self.create_subscription(
                PointCloud2, p('input_cloud').value, self._cloud_cb, sensor_qos)
            self.get_logger().info('object_detector ready [SAM 2]')
        else:
            self.create_subscription(
                PointCloud2, p('input_cloud').value, self._cloud_cb, sensor_qos)
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
        device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Loading SAM 2 on {device} …')

        model = build_sam2(cfg, weights, device=device)
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

    def _image_cb(self, msg: Image):
        """Cache the latest RGB image for SAM."""
        self._latest_image = msg

    def _cloud_cb(self, msg: PointCloud2):
        t0 = time.monotonic()

        # 1. Parse + transform to base_link
        pts_cam = pointcloud2_to_xyz(msg)
        if pts_cam.shape[0] == 0:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}',
                                   throttle_duration_sec=5.0)
            return

        pts_base = apply_tf(pts_cam, tf)

        # 2. ROI crop
        m = self.roi
        mask = (
            (pts_base[:,0] >= m['x'][0]) & (pts_base[:,0] <= m['x'][1]) &
            (pts_base[:,1] >= m['y'][0]) & (pts_base[:,1] <= m['y'][1]) &
            (pts_base[:,2] >= m['z'][0]) & (pts_base[:,2] <= m['z'][1])
        )
        pts_roi = pts_base[mask]
        # Keep camera-frame ROI indices for SAM back-projection
        idx_roi = np.where(mask)[0]

        if pts_roi.shape[0] < self.min_pts:
            self._publish_empty(msg.header)
            return

        # 3. Segment
        if self.method == 'sam':
            clusters = self._segment_sam(msg, pts_cam, pts_roi, idx_roi)
        else:
            clusters = self._segment_dbscan(pts_roi)

        if not clusters:
            self._publish_empty(msg.header)
            return

        # 4. Per-cluster: hull + pose + publish
        self._publish_results(msg.header, clusters)

        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(
            f'{len(clusters)} object(s) [{self.method}] in {dt:.1f} ms',
            throttle_duration_sec=2.0)

    # -------------------------------------------------------------------------
    # Segmentation backends
    # -------------------------------------------------------------------------

    def _segment_dbscan(self, pts_roi: np.ndarray):
        """DBSCAN on the ROI pointcloud. Returns list of (N,3) arrays."""
        # Voxel downsample: keep one point per grid cell
        voxel_idx = np.floor(pts_roi / self.voxel_size).astype(np.int32)
        _, unique = np.unique(voxel_idx, axis=0, return_index=True)
        pts_down = pts_roi[unique]

        if len(pts_down) < self.min_pts:
            return []

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_pts,
        ).fit_predict(pts_down)

        clusters = []
        for lbl in set(labels) - {-1}:
            c = pts_down[labels == lbl]
            if self.min_pts <= len(c) <= self.max_pts:
                clusters.append(c)
        return clusters

    def _segment_sam(self, cloud_msg: PointCloud2,
                     pts_cam: np.ndarray,
                     pts_roi: np.ndarray,
                     idx_roi: np.ndarray):
        """
        SAM 2 on the latest RGB image.
        Each mask → select matching 3D points from ROI → cluster.

        The pointcloud and image share the same pixel grid:
          point index i  ↔  pixel (row = i // width, col = i % width)
        """
        if self._latest_image is None:
            self.get_logger().warn('No RGB image received yet', throttle_duration_sec=5.0)
            return []

        # Convert ROS image → numpy BGR → RGB
        bgr = self._bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Run SAM
        masks_data = self._sam.generate(rgb)
        if not masks_data:
            return []

        w = cloud_msg.width
        clusters = []

        for md in masks_data:
            mask_2d = md['segmentation']  # H×W bool array

            # Map ROI point indices → pixel coords → check mask
            rows = idx_roi // w
            cols = idx_roi % w

            # Bounds check (image and cloud should match but be safe)
            h_img, w_img = mask_2d.shape
            valid = (rows < h_img) & (cols < w_img)
            in_mask = valid & mask_2d[rows * valid, cols * valid]

            cluster = pts_roi[in_mask]
            if len(cluster) >= self.sam_min_pts:
                clusters.append(cluster)

        return clusters

    # -------------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------------

    def _publish_results(self, header, clusters):
        detections = Detection3DArray()
        detections.header.stamp    = header.stamp
        detections.header.frame_id = self.base_frame
        markers = MarkerArray()

        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        for obj_id, pts in enumerate(clusters):
            centroid, (qx, qy, qz, qw) = pca_orientation(pts)
            mins, maxs = pts.min(axis=0), pts.max(axis=0)
            size = maxs - mins
            verts, tris = convex_hull_scipy(pts)
            color = label_color(obj_id)
            stamp = header.stamp
            frame = self.base_frame

            # Detection3D
            det = Detection3D()
            det.header = detections.header
            det.id = str(obj_id)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'object'
            hyp.hypothesis.score    = 1.0
            hyp.pose.pose.position.x = float(centroid[0])
            hyp.pose.pose.position.y = float(centroid[1])
            hyp.pose.pose.position.z = float(centroid[2])
            hyp.pose.pose.orientation.x = float(qx)
            hyp.pose.pose.orientation.y = float(qy)
            hyp.pose.pose.orientation.z = float(qz)
            hyp.pose.pose.orientation.w = float(qw)
            det.results.append(hyp)
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

            # Hull wireframe
            if verts is not None:
                markers.markers.append(
                    self._mk_hull(obj_id, stamp, frame, verts, tris, color))

            # Centroid sphere
            markers.markers.append(
                self._mk_centroid(obj_id, stamp, frame, centroid, color))

            # PCA axes (R=X longest, G=Y, B=Z)
            R_mat = self._quat_to_mat(qx, qy, qz, qw)
            axis_colors = [
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.9),
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9),
                ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.9),
            ]
            for ai, ac in enumerate(axis_colors):
                markers.markers.append(
                    self._mk_axis(obj_id*10+ai+100, stamp, frame,
                                  centroid, R_mat[:,ai], float(size[ai])*0.5, ac))

        self.pub_det.publish(detections)
        self.pub_mk.publish(markers)

    def _publish_empty(self, header):
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
        m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
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
        m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
        return m

    def _mk_axis(self, marker_id, stamp, frame, origin, axis, scale, color):
        m = Marker()
        m.header.stamp = stamp; m.header.frame_id = frame
        m.ns = 'axes'; m.id = marker_id
        m.type = Marker.ARROW; m.action = Marker.ADD
        m.scale.x = 0.005; m.scale.y = 0.010; m.scale.z = 0.015
        m.color = color
        m.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
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
