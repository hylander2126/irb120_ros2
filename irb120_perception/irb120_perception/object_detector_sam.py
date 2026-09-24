"""
IRB120 Workspace Object Detector — SAM backend
================================================
Alternative segmentation backend to `object_detector_dbscan.py`, added to
compare against it, not to replace it — both can run at once (see
`perception_sam.launch.py`), publishing on separate topic namespaces
(`object_detector` vs `object_detector_sam`).

DBSCAN clusters by spatial gap, so it cannot split two objects that touch or
share a depth profile (see object_detector_dbscan.py's docstring). This
backend instead segments the color image with a promptable vision model
(Segment Anything, in "segment everything" mode) and back-projects each mask
through the aligned, robot-masked depth image into 3D. It splits on visual
appearance/boundary, so touching objects with a visible seam are not a
special case for it the way they are for DBSCAN.

Model: MobileSAM (`sam_model_type='vit_t'`) by default — the same
`sam_model_registry`/`SamAutomaticMaskGenerator` API as Meta's original
Segment Anything, but with a ~40M-param TinyViT image encoder in place of
ViT-H's 632M params, because this machine is CPU-only and full SAM is
impractically slow there (the image encoder alone is tens of seconds to
minutes per frame on CPU; MobileSAM is low single-digit seconds). A full SAM
checkpoint ('vit_b'/'vit_l'/'vit_h') also loads through this same class if
you ever run this on a GPU box and want the accuracy back.

No realtime requirement (robot at rest, out of frame): SAM re-segments at
most once every `min_reseg_interval_sec` (default 5s) while active, not on
every ~30 Hz depth-image arrival — re-running a multi-second CPU inference
that often would be pure waste for a scene that isn't changing. This is a
throttle, not a one-shot: an earlier version ran exactly once per activation
and then silently never again for the rest of that activation (a real bug —
toggling a topic's display in RViz doesn't call any service, so it looked
like "only works the first time"). Explicitly deactivating and reactivating
via `~/set_active` immediately allows a fresh run regardless of the throttle
window, same as before.

Status / limitations:
  - Single camera (camera 1) only, matching the current scope of
    `robot_mask_filter`'s `~/depth_masked_sam` output — see that node's
    docstring ("The SAM depth-image path is untouched by this — single
    camera only"). Multi-camera fusion for this backend (matching DBSCAN's
    three-camera fusion) is not implemented; it would need either 2D mask
    correspondence across viewpoints or per-camera SAM + 3D mask fusion.
  - Masks are class-agnostic (SAM has no notion of "object" vs "table" vs
    "background") — the ROI crop and `min_cluster_pts`/`max_cluster_pts`
    filtering do the same job DBSCAN's cluster-size filter does, rejecting
    masks that are too small (noise) or too large (a robot-masked hole,
    the table, or the whole scene) after back-projection.
  - `max_depth_gap_ratio` drops a mask whose pixels are mostly invalid
    depth (zero) — most commonly a mask that mostly covers the robot-masked
    hole in the middle of the frame, which produces a plausible-looking 2D
    mask with almost no usable 3D geometry behind it.

Speed tuning — profiled directly (script timings, not guesses) on this
machine against a synthetic 1280x720 test scene with 6 objects, MobileSAM,
`min_mask_region_area=400`:

  Where the time actually goes: the image encoder is cheap (~2.7s) and
  roughly resolution-independent — SAM/MobileSAM resize the input to a fixed
  size internally before the encoder runs, so input image resolution is NOT
  a lever (downscaling the image yourself before calling `generate()` does
  not help — it only loses detail before the model's own resize). The real
  cost is the point-grid mask-decode + post-processing loop, which scales
  close to linearly with `points_per_side^2`:
    points_per_side= 8 (64 pts):  ~12s
    points_per_side=12 (144 pts): ~23s
    points_per_side=16 (256 pts): ~39s  (this node's shipped default)

  `points_per_side` below ~8 hits a real recall cliff on the test scene —
  not just slower convergence, but objects silently missing entirely
  (pps=4 found 3/6 objects, pps=6 found 2/6, pps=8 and pps=10 both found
  5/6 with no further gain above 8). Do not go below 8 without re-verifying
  recall on your actual scene; this node's default is intentionally left at
  16, above that floor, since accuracy matters more than speed here — lower
  it yourself only after checking recall doesn't regress for your objects.

  Thread count is a separate, effectively free lever: at points_per_side=8,
  OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1 (this node's
  previous default, set defensively after the EGM incident below) took
  28s; =8 took 11s; =20 (all cores) took 12s — *slightly slower* than 8, so
  there's no accuracy/completeness trade-off here at all, just diminishing
  returns and oversubscription overhead past ~8 threads. `perception_sam.
  launch.py`'s `sam_num_threads` argument now defaults to 8 instead of 1.
  Reasoning for why this is still safe near the EGM control loop, unlike
  the incident that motivated capping to 1 in the first place: that incident
  was `robot_mask_filter` running *continuously* (every ~11ms, forever,
  while active) with *unbounded* thread count (OpenBLAS defaulted to
  MAX_THREADS=64, no cap at all) — sustained, unbounded oversubscription.
  This node instead runs *once* per activation (a single bounded burst), and
  8 threads leaves 12+ of this machine's 20 cores free throughout that
  burst, so the EGM/controller_manager loop is never starved of a core to
  run on. Different risk profile, not the same mistake repeated — but if you
  want the older, more conservative single-threaded behavior back, pass
  `sam_num_threads:=1` at launch.

  Bottom line: full "segment everything" MobileSAM cannot reach ~1-2s on
  this CPU without giving up real recall (see the pps cliff above) — that's
  inherent to decoding a whole point grid, not a tuning artifact. The
  thread-count fix alone (zero accuracy cost) cuts the shipped
  points_per_side=16 default from the ~85s originally observed live to
  roughly ~40s. Getting meaningfully below that needs a different approach
  entirely, e.g. seeding a handful of targeted point/box prompts from
  DBSCAN's cluster centroids instead of a dense full-image point grid — not
  implemented here.

Depends on packages not in package.xml (pip, not a rosdep-known package —
install into this workspace's Python environment, not declared as a ROS
exec_depend):

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git

and a MobileSAM checkpoint (see the `sam_checkpoint` parameter — this node
logs an error and never produces detections until it's set and the file
exists).

Running this node: use `perception_sam.launch.py`, not a bare `ros2 run` —
irb120_perception itself stays built with the workspace's normal system
Python (robot_mask_filter and object_detector_dbscan run near a live,
250 Hz-real-time EGM control loop and must not depend on whichever Python a
pip-heavy venv happens to be), so a plain `ros2 run irb120_perception
object_detector_sam` executes under system Python and fails the torch import
immediately. The launch file instead runs this one node's process through
the venv's interpreter via `prefix`, and caps it to single-threaded
BLAS/torch (`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS=1`) —
see that launch file's docstring for why: building the whole package under
the venv's Python once caused robot_mask_filter to spin up a 20-thread
OpenBLAS pool on every point-cloud callback and starve that same EGM loop on
live hardware. If you ever do need to run this file directly (e.g. outside
any launch file, against a bag), invoke it explicitly with the venv's
python3 and the same env vars — never rely on its installed shebang.

Publishes (identical contract to object_detector_dbscan, so RViz/consumers
can point at either):
  ~/detections     vision_msgs/Detection3DArray
  ~/markers        visualization_msgs/MarkerArray
  ~/object_points  sensor_msgs/PointCloud2 (x,y,z,label)

Dependencies:
  system pip: numpy
  ROS:        sensor_msgs, vision_msgs, visualization_msgs, tf2_ros
  extra pip:  torch (CPU build), mobile_sam — see above
"""

import time

import numpy as np
import rclpy
import message_filters
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo

from irb120_perception.perception_common import (
    ObjectDetectorBase, apply_tf, fit_dominant_horizontal_plane, remove_plane,
)

try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
    _SAM_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - exercised only without the extra deps installed
    sam_model_registry = None
    SamAutomaticMaskGenerator = None
    _SAM_IMPORT_ERROR = exc


def _image_to_rgb(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image (rgb8 or bgr8) to an HxWx3 uint8 array.

    No cv_bridge dependency, matching how the rest of this package (and
    robot_mask_filter's own depth handling) unpacks Image messages by hand.
    """
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding not in ('rgb8', 'bgr8'):
        raise ValueError(f"unsupported color encoding '{msg.encoding}' (expected rgb8/bgr8)")
    row_stride = msg.step // 3
    img = buf.reshape(msg.height, row_stride, 3)[:, :msg.width, :]
    if msg.encoding == 'bgr8':
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def _image_to_depth_m(msg: Image) -> np.ndarray:
    """Decode a 16UC1 depth image (millimetres, 0 = invalid) to metres."""
    depth_mm = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
    return depth_mm.astype(np.float32) / 1000.0


class SAMObjectDetector(ObjectDetectorBase):

    def __init__(self):
        super().__init__('object_detector_sam')

        # ---- SAM-specific parameters -------------------------------------
        self.declare_parameter('color_topic', '/realsense/color/image_raw')
        self.declare_parameter('depth_topic', '/realsense/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/realsense/color/camera_info')
        self.declare_parameter('color_topic2', '')
        self.declare_parameter('depth_topic2', '')
        self.declare_parameter('camera_info_topic2', '')
        self.declare_parameter('color_topic3', '')
        self.declare_parameter('depth_topic3', '')
        self.declare_parameter('camera_info_topic3', '')
        self.declare_parameter('sync_slop_sec', 0.08)
        self.declare_parameter('sam_model_type', 'vit_t')  # 'vit_t' = MobileSAM; 'vit_b'/'vit_l'/'vit_h' = full SAM (too slow on CPU)
        self.declare_parameter('sam_checkpoint', '')        # path to .pt/.pth checkpoint; required
        self.declare_parameter('sam_device', 'cpu')
        self.declare_parameter('points_per_side', 16)        # SAM default is 32; halved for CPU speed, at some recall cost
        self.declare_parameter('points_per_batch', 64)        # decoder batch size — see "Speed tuning" below; mask-decode cost scales ~linearly with points_per_side^2, not with this
        self.declare_parameter('pred_iou_thresh', 0.86)
        self.declare_parameter('stability_score_thresh', 0.92)
        self.declare_parameter('min_mask_region_px', 400)     # drop tiny 2D masks before any 3D work
        self.declare_parameter('min_cluster_pts', 30)          # post-back-projection — same meaning as DBSCAN's
        self.declare_parameter('max_cluster_pts', 50000)
        self.declare_parameter('max_depth_gap_ratio', 0.3)    # drop a mask if more than this fraction of its pixels have invalid/masked depth
        self.declare_parameter('table_plane_distance', 0.008) # remove table pixels retained by a 2-D SAM mask
        self.declare_parameter('min_reseg_interval_sec', 5.0)  # throttle, not a one-shot — see this module's docstring

        p = self.get_parameter
        self.color_topic       = p('color_topic').value
        self.depth_topic       = p('depth_topic').value
        self.camera_info_topic = p('camera_info_topic').value
        self.sam_model_type    = p('sam_model_type').value
        self.sam_checkpoint    = p('sam_checkpoint').value
        self.sam_device        = p('sam_device').value
        self.points_per_side          = p('points_per_side').value
        self.points_per_batch         = p('points_per_batch').value
        self.pred_iou_thresh          = p('pred_iou_thresh').value
        self.stability_score_thresh   = p('stability_score_thresh').value
        self.min_mask_region_px       = p('min_mask_region_px').value
        self.min_pts             = p('min_cluster_pts').value
        self.max_pts             = p('max_cluster_pts').value
        self.max_depth_gap_ratio = p('max_depth_gap_ratio').value
        self.table_plane_distance = p('table_plane_distance').value
        self.min_reseg_interval_sec = p('min_reseg_interval_sec').value

        self._mask_generator = None
        self._cameras = {}
        self._last_run_end_time = None  # time.monotonic() of the last completed run, or None if never run

        if _SAM_IMPORT_ERROR is not None:
            self.get_logger().error(
                f"mobile_sam/torch not importable ({_SAM_IMPORT_ERROR}); this node will "
                f"subscribe but never produce detections until the dependency is installed "
                f"— see this module's docstring for the pip install commands.")
        elif not self.sam_checkpoint:
            self.get_logger().error(
                "'sam_checkpoint' parameter not set — see this module's docstring for the "
                "checkpoint download step. This node will subscribe but never produce detections.")
        else:
            self._load_model()

        # ---- QoS / subscriptions -----------------------------------------
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._add_camera('cam1', self.color_topic, self.depth_topic,
                         self.camera_info_topic, sensor_qos)
        for slot, suffix in (('cam2', '2'), ('cam3', '3')):
            color = p(f'color_topic{suffix}').value
            depth = p(f'depth_topic{suffix}').value
            info = p(f'camera_info_topic{suffix}').value
            if any((color, depth, info)):
                if not all((color, depth, info)):
                    raise ValueError(f'{slot} requires color, depth, and camera-info topics together')
                self._add_camera(slot, color, depth, info, sensor_qos)
        self.get_logger().info(
            'object_detector_sam ready [SAM]' +
            ('' if self._mask_generator is not None else ' (model NOT loaded — see error above)'))

    # -------------------------------------------------------------------------

    def _on_activate(self):
        # Explicit reactivation always gets an immediate fresh run, bypassing
        # the min_reseg_interval_sec throttle — otherwise a deliberate
        # "turn it on for this check" could still be stuck waiting out a
        # throttle window left over from a previous activation.
        self._last_run_end_time = None

    def _load_model(self):
        model = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint)
        model.to(device=self.sam_device)
        self._mask_generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_px,
        )
        self.get_logger().info(
            f'Loaded SAM model type={self.sam_model_type} device={self.sam_device} '
            f'checkpoint={self.sam_checkpoint}')

    def _add_camera(self, slot, color_topic, depth_topic, info_topic, sensor_qos):
        """Register a camera with colour/depth timestamp synchronization."""
        state = {'info': None, 'pair': None}
        self._cameras[slot] = state
        self.create_subscription(
            CameraInfo, info_topic,
            lambda msg, s=slot: self._camera_info_cb(s, msg), sensor_qos)
        color_sub = message_filters.Subscriber(
            self, Image, color_topic, qos_profile=sensor_qos)
        depth_sub = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=sensor_qos)
        sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=3,
            slop=float(self.get_parameter('sync_slop_sec').value))
        sync.registerCallback(lambda color, depth, s=slot: self._image_pair_cb(s, color, depth))
        # Keep strong references: otherwise message_filters subscriptions can
        # be garbage-collected even though their callbacks were registered.
        state['subscriptions'] = (color_sub, depth_sub, sync)
        self.get_logger().info(f'SAM camera enabled — {slot}: {color_topic}')

    def _camera_info_cb(self, slot, msg):
        self._cameras[slot]['info'] = msg

    def _image_pair_cb(self, slot, color, depth):
        self._cameras[slot]['pair'] = (color, depth)
        if not self._active:
            return
        now = time.monotonic()
        if self._last_run_end_time is not None and (now - self._last_run_end_time) < self.min_reseg_interval_sec:
            return
        if self._mask_generator is None:
            return
        snapshots = [(name, state['pair'][0], state['pair'][1], state['info'])
                     for name, state in self._cameras.items()
                     if state['pair'] is not None and state['info'] is not None]
        if not snapshots:
            return
        # Claim this throttle window immediately, even if segmentation below
        # fails partway — otherwise a transient TF failure would retry a
        # multi-second SAM pass on every subsequent ~33ms depth frame instead
        # of waiting out min_reseg_interval_sec like everything else does.
        self._last_run_end_time = now
        self._segment_and_publish(snapshots)

    # -------------------------------------------------------------------------

    def _segment_and_publish(self, snapshots):
        t0 = time.monotonic()

        clusters = []
        raw_mask_count = 0
        header = snapshots[0][2].header
        for slot, color_msg, depth_msg, info_msg in snapshots:
            try:
                color = _image_to_rgb(color_msg)
            except ValueError as e:
                self.get_logger().error(f'{slot}: {e}')
                continue
            depth = _image_to_depth_m(depth_msg)
            if color.shape[:2] != depth.shape:
                self.get_logger().warn(f'{slot}: color/depth dimensions differ; skipping pair')
                continue
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, depth_msg.header.frame_id,
                    rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
            except Exception as e:
                self.get_logger().warn(f'{slot}: TF lookup failed: {e}')
                continue
            masks = self._mask_generator.generate(color)
            raw_mask_count += len(masks)
            clusters.extend(self._masks_to_clusters(masks, depth, info_msg, tf))

        # SAM masks are 2-D regions and commonly include the table touching an
        # object.  Estimate the shared horizontal support plane in 3-D, then
        # remove it before instance association.  This is intentionally after
        # SAM: a mask can still trace the object's visual boundary precisely.
        plane_samples = (np.concatenate(clusters, axis=0) if clusters else
                         np.empty((0, 3), dtype=np.float32))
        plane = fit_dominant_horizontal_plane(
            plane_samples, distance=self.table_plane_distance)
        if plane is not None:
            clusters = [remove_plane(c, plane, self.table_plane_distance)
                        for c in clusters]
            clusters = [c for c in clusters if self.min_pts <= len(c) <= self.max_pts]
        clusters = self._fuse_camera_instances(clusters)
        header.frame_id = self.base_frame

        if not clusters:
            self._reset_smoothing()
            self._publish_empty(header)
        else:
            self._publish_results(header, clusters)

        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(
            f'{len(clusters)} object(s) [SAM] in {dt:.0f} ms '
            f'({raw_mask_count} raw masks, {len(snapshots)} camera(s))')

    def _masks_to_clusters(self, masks, depth, info_msg, tf):
        """Back-project one camera's SAM masks into base_link."""
        fx, fy, cx, cy = info_msg.k[0], info_msg.k[4], info_msg.k[2], info_msg.k[5]
        m = self.roi
        clusters = []
        for mask in masks:
            seg = mask['segmentation']
            vs, us = np.nonzero(seg)
            if len(us) < self.min_mask_region_px:
                continue

            z = depth[vs, us]
            valid = z > 0
            # Most of the mask has no usable depth (e.g. it's mostly the
            # robot-masked hole) -> its 3D geometry can't be trusted.
            if valid.size == 0 or (1.0 - valid.mean()) > self.max_depth_gap_ratio:
                continue

            us_v, vs_v, z_v = us[valid], vs[valid], z[valid]
            x = (us_v - cx) * z_v / fx
            y = (vs_v - cy) * z_v / fy
            pts_cam = np.stack([x, y, z_v], axis=1).astype(np.float32)
            pts_base = apply_tf(pts_cam, tf)

            in_roi = (
                (pts_base[:, 0] >= m['x'][0]) & (pts_base[:, 0] <= m['x'][1]) &
                (pts_base[:, 1] >= m['y'][0]) & (pts_base[:, 1] <= m['y'][1]) &
                (pts_base[:, 2] >= m['z'][0]) & (pts_base[:, 2] <= m['z'][1])
            )
            pts_base = pts_base[in_roi]
            if self.min_pts <= len(pts_base) <= self.max_pts:
                clusters.append(pts_base)
        return clusters

    @staticmethod
    def _fuse_camera_instances(clusters):
        """Fuse only instances whose 3-D AABBs materially overlap.

        This deliberately does not use DBSCAN: spatial clustering at this
        stage would again join two touching SAM instances.  AABB IoU is a
        conservative cross-view identity test for the static tabletop scene.
        """
        fused = []
        for cluster in clusters:
            lo, hi = cluster.min(axis=0), cluster.max(axis=0)
            for i, existing in enumerate(fused):
                e_lo, e_hi = existing.min(axis=0), existing.max(axis=0)
                inter = np.maximum(0.0, np.minimum(hi, e_hi) - np.maximum(lo, e_lo))
                inter_vol = float(np.prod(inter))
                union = float(np.prod(hi - lo) + np.prod(e_hi - e_lo) - inter_vol)
                if union > 0.0 and inter_vol / union >= 0.15:
                    fused[i] = np.concatenate((existing, cluster), axis=0)
                    break
            else:
                fused.append(cluster)
        return fused


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
