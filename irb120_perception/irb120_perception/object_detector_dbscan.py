"""
IRB120 Workspace Object Detector — DBSCAN backend
==================================================
Subscribes to a PointCloud2 (by default, the robot-masked, multi-camera-fused
cloud published by `robot_mask_filter`), crops to the robot workspace, clusters
remaining points spatially with DBSCAN, then for each object computes:

  - 3D convex hull  (vertices + triangular faces)
  - Centroid        (geometry_msgs/Point in base_link)
  - Orientation     (PCA principal axes → quaternion, X = longest axis)

Pure geometry, fast, no GPU needed. Works well when objects are separated by a
gap; fails when objects touch or have similar depth (use the SAM backend,
`object_detector_sam.py`, for that case instead).

Single-object workspaces (`single_object_mode`): DBSCAN naturally reports one
cluster per disconnected point group, so a rigid object with a real 3D gap
between its own parts — e.g. a monitor whose base connects to the screen only
at a rear joint neither camera can see — comes back as multiple separate
"objects". When `single_object_mode` is on, every cluster that survives the
normal size filtering (still rejects noise/stray blobs) gets unioned into one
combined object instead of published separately. Off by default; only turn
it on if the workspace is scoped to a single physical item per detection
cycle — if two genuinely separate objects can share the ROI, this will
wrongly fuse them into one.

Multi-camera note: this node itself has no camera-count awareness. Point-cloud
fusion across cameras happens upstream in `robot_mask_filter`, which transforms
each camera's points into `base_frame` before masking/publishing — this node
just clusters whatever arrives on `input_cloud_pc` as one cloud. See that
node's docstring for the fusion details.

Table removal: handled by roi_z_min (set to known table height + margin).
No RANSAC needed since the table height is fixed.

Publishes:
  ~/detections     vision_msgs/Detection3DArray
  ~/markers        visualization_msgs/MarkerArray
  ~/object_points  sensor_msgs/PointCloud2 (x,y,z,label)

Dependencies:
  system pip: numpy, scikit-learn, scipy
  ROS:        sensor_msgs, vision_msgs, visualization_msgs, tf2_ros
"""

import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2

from sklearn.cluster import DBSCAN

from irb120_perception.perception_common import (
    ObjectDetectorBase, apply_tf, pointcloud2_to_xyz, voxel_downsample,
)


class DBSCANObjectDetector(ObjectDetectorBase):

    def __init__(self):
        super().__init__('object_detector')

        # ---- DBSCAN-specific parameters ---------------------------------------
        self.declare_parameter('input_cloud_pc', '/realsense/depth/color/points')
        self.declare_parameter('dbscan_eps',      0.02)
        self.declare_parameter('dbscan_min_pts',  20)
        self.declare_parameter('min_cluster_pts', 30)
        self.declare_parameter('max_cluster_pts', 50000)
        self.declare_parameter('single_object_mode', True)

        p = self.get_parameter
        self.dbscan_eps     = p('dbscan_eps').value
        self.dbscan_min_pts = p('dbscan_min_pts').value
        self.min_pts        = p('min_cluster_pts').value
        self.max_pts         = p('max_cluster_pts').value
        self.single_object_mode = p('single_object_mode').value

        # ---- QoS --------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            PointCloud2, p('input_cloud_pc').value, self._cloud_cb, sensor_qos)
        self.get_logger().info('object_detector ready [DBSCAN]')

    # -------------------------------------------------------------------------

    def _cloud_cb(self, msg: PointCloud2):
        """Receives a PointCloud2 (possibly multi-camera fused), transforms to
        base_link if needed, crops to the workspace ROI, and segments."""
        t0 = time.monotonic()

        pts_cam = pointcloud2_to_xyz(msg)
        if pts_cam.shape[0] == 0:
            return

        # Look up cloud-frame→base_link transform at the latest available time.
        # When robot_mask_filter has already fused/published in base_frame this
        # resolves to identity — tf2 special-cases source == target frame.
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
            self._reset_smoothing()
            self._publish_empty(msg.header)
            return

        self._publish_results(msg.header, clusters)
        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(f'{len(clusters)} object(s) [dbscan] in {dt:.1f} ms',
                               throttle_duration_sec=2.0)

    def _segment_dbscan(self, pts_roi: np.ndarray):
        """DBSCAN on the ROI pointcloud. Returns list of (N,3) arrays.

        Voxel downsampling here also absorbs cross-camera overlap: when two
        cameras both see the same surface, their transformed points land in
        (or very near) the same voxel cells and collapse to one representative
        point, rather than just doubling density everywhere.
        """
        pts_down = voxel_downsample(pts_roi, self.voxel_size)

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

        # Single-object workspace: a rigid object can still come back as multiple
        # disconnected clusters (e.g. a monitor's base and screen, joined only at
        # a rear seam no camera can see). Union everything that survived the size
        # filter above into one object rather than publishing them separately.
        if self.single_object_mode and len(clusters) > 1:
            clusters = [np.concatenate(clusters, axis=0)]

        return clusters


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DBSCANObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
