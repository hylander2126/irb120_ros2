"""
IRB120 Workspace Object Detector — DBSCAN backend
==================================================
Subscribes to one to three raw camera PointCloud2 streams, transforms and
fuses their latest clouds in `base_link`, crops to the workspace, then clusters
the remaining points spatially with DBSCAN. For each object it computes:

  - 3D convex hull  (vertices + triangular faces)
  - Centroid        (geometry_msgs/Point in base_link)
  - Orientation     (PCA principal axes → quaternion, X = longest axis)

Pure geometry, fast, no GPU needed. Works well when objects are separated by a
gap; fails when objects touch or have similar depth, since their points merge
into a single cluster with no spatial gap to split on.

Before clustering, `remove_sparse_outliers` strips locally-sparse points (e.g.
a depth-camera "flying pixel" noise trail bleeding off an object edge) that
would otherwise chain-link onto a real cluster via DBSCAN's single-linkage
behaviour — see that function's docstring in `perception_common.py`.

Temporal accumulation (`accum_frames`): since the robot is stationary and out
of frame while this runs, single-frame noise can be distinguished from real
geometry by whether it *recurs* across frames, not just by local density
within one frame. When `accum_frames > 1`, each incoming ROI-cropped frame is
pushed into a `FrameAccumulator` (see its docstring) and segmentation only
runs on the fused, persistence-filtered cloud once the sliding window fills.
The node keeps the original single-frame default for standalone use; the
offline launch enables a 20-observation / 12-hit consensus window.

Budget for this: the naive estimate is `accum_frames / rate` seconds of
latency before the first usable detection, using the fused-cloud publish
rate. That estimate is only as good as your assumption about `rate` —
With direct camera input, budget roughly `accum_frames / observation-rate`
before the first result. While the window is
filling, this node publishes nothing at all (not even an empty detection) on
`~/object_points`/`~/detections`, so a consumer that reads "the first message
after activation" (e.g. irb120_control's perception snapshot) never mistakes
a warm-up frame for "no objects detected" — but that same silence is
indistinguishable from a hang if the window never fills in a reasonable time.

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

Multi-camera note: optional `input_cloud_pc2` and `input_cloud_pc3` streams
are individually transformed and cached here; every arrival is clustered with
the latest cloud from every enabled camera. This targets a static offline
scene and intentionally avoids a hardware-sync requirement.

Table removal: the ROI makes a coarse crop, then a dominant near-horizontal
plane is fitted and removed using `table_plane_distance`.

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
    FrameAccumulator, ObjectDetectorBase, apply_tf, pointcloud2_to_xyz,
    fit_dominant_horizontal_plane, remove_plane, remove_sparse_outliers, voxel_downsample,
)


class DBSCANObjectDetector(ObjectDetectorBase):

    def __init__(self):
        super().__init__('object_detector')

        # ---- DBSCAN-specific parameters ---------------------------------------
        self.declare_parameter('input_cloud_pc', '/realsense/depth/color/points')
        # Optional raw cameras.  Unlike the historical masked input, these are
        # fused here after their individual TF transforms; no robot masking is
        # performed.  Keeping this in the detector makes the mask filter an
        # optional, separate safety tool rather than a pipeline dependency.
        self.declare_parameter('input_cloud_pc2', '')
        self.declare_parameter('input_cloud_pc3', '')
        self.declare_parameter('dbscan_eps',      0.02)
        self.declare_parameter('dbscan_min_pts',  20)
        self.declare_parameter('min_cluster_pts', 30)
        self.declare_parameter('max_cluster_pts', 50000)
        self.declare_parameter('single_object_mode', True)
        self.declare_parameter('outlier_k',         8)    # neighbours sampled per point for local-density check
        self.declare_parameter('outlier_std_ratio', 2.0)  # 0 disables the check
        self.declare_parameter('table_plane_distance', 0.008)
        self.declare_parameter('accum_frames',   1)  # sliding-window length; 1 = off (single-frame, original behaviour)
        self.declare_parameter('accum_min_hits', 0)  # min distinct frames a voxel must appear in; 0 = auto (~60% of accum_frames)

        p = self.get_parameter
        self.dbscan_eps     = p('dbscan_eps').value
        self.dbscan_min_pts = p('dbscan_min_pts').value
        self.min_pts        = p('min_cluster_pts').value
        self.max_pts         = p('max_cluster_pts').value
        self.single_object_mode = p('single_object_mode').value
        self.outlier_k          = p('outlier_k').value
        self.outlier_std_ratio  = p('outlier_std_ratio').value
        self.table_plane_distance = p('table_plane_distance').value

        accum_frames = int(p('accum_frames').value)
        if accum_frames > 1:
            accum_min_hits = int(p('accum_min_hits').value)
            if accum_min_hits <= 0:
                accum_min_hits = max(2, int(np.ceil(0.6 * accum_frames)))
            self._accumulator = FrameAccumulator(accum_frames, self.voxel_size, accum_min_hits)
            self.get_logger().info(
                f'Temporal accumulation on: {accum_frames} frames, min_hits={accum_min_hits}')
        else:
            self._accumulator = None

        # ---- QoS --------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._camera_clouds = {}
        self.create_subscription(
            PointCloud2, p('input_cloud_pc').value,
            lambda msg: self._cloud_cb(msg, 'cam1'), sensor_qos)
        for slot, parameter in (('cam2', 'input_cloud_pc2'), ('cam3', 'input_cloud_pc3')):
            topic = p(parameter).value
            if topic:
                self.create_subscription(
                    PointCloud2, topic,
                    lambda msg, s=slot: self._cloud_cb(msg, s), sensor_qos)
                self.get_logger().info(f'Raw multi-camera fusion enabled — {slot}: {topic}')
        self.get_logger().info('object_detector ready [DBSCAN]')

    # -------------------------------------------------------------------------

    def _on_activate(self):
        # Fresh activation window -> fresh temporal window. Otherwise the
        # first fused cloud of a new check could blend in frames buffered
        # from whatever the scene looked like the *previous* time this node
        # was active (a different object, or the same object before it moved).
        if self._accumulator is not None:
            self._accumulator.reset()
        self._camera_clouds.clear()

    def _cloud_cb(self, msg: PointCloud2, slot='cam1'):
        """Receives a PointCloud2 (possibly multi-camera fused), transforms to
        base_link if needed, crops to the workspace ROI, and segments."""
        if not self._active:
            return
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
        self._camera_clouds[slot] = pts_base[mask]
        # The scene is static during offline perception, so reusing the most
        # recent cloud from each camera avoids a hardware-sync requirement.
        pts_roi = np.concatenate(list(self._camera_clouds.values()), axis=0)

        # Plane rejection happens before downsampling/DBSCAN, so table points
        # cannot chain a flying-pixel island into a real object cluster.
        plane = fit_dominant_horizontal_plane(
            pts_roi, distance=self.table_plane_distance)
        if plane is not None:
            pts_roi = remove_plane(pts_roi, plane, self.table_plane_distance)

        if self._accumulator is not None:
            fused = self._accumulator.add(pts_roi)
            if fused is None:
                # Sliding window still warming up — publish nothing at all
                # (not even empty) so a consumer waiting for "the first
                # message" after activation doesn't grab a warm-up frame.
                # See this module's docstring, "Temporal accumulation".
                return
            pts_for_seg = fused
        else:
            pts_for_seg = pts_roi

        # Not enough points to form even one cluster — publish empty and bail
        if pts_for_seg.shape[0] < self.min_pts:
            self._publish_empty(msg.header)
            return

        clusters = self._segment_dbscan(pts_for_seg)

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

        When temporal accumulation is on, `pts_roi` here is already the fused,
        one-point-per-surviving-voxel output of `FrameAccumulator.fuse()` at
        this same `voxel_size` — this call is then a cheap near-no-op (each
        point already occupies its own voxel) rather than a real downsample,
        kept only so this function's contract doesn't depend on whether the
        caller accumulated first.
        """
        pts_down = voxel_downsample(pts_roi, self.voxel_size)

        # Strip locally-sparse points (e.g. a depth-camera "flying pixel" noise
        # trail bleeding off an object edge) *before* clustering. DBSCAN's
        # single-linkage chaining would otherwise happily absorb a sparse trail
        # like that into a real object's cluster one point-hop at a time, or
        # bridge two genuinely separate objects into one. See
        # `remove_sparse_outliers`'s docstring.
        if self.outlier_std_ratio > 0:
            pts_down = remove_sparse_outliers(pts_down, self.outlier_k, self.outlier_std_ratio)

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
