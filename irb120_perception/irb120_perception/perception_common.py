"""
Shared geometry helpers and publishing base class for the object detector backend(s).
=======================================================================================
`object_detector_dbscan.py` subclasses `ObjectDetectorBase(Node)` here for:

  - PointCloud2 <-> numpy conversion, TF application, PCA orientation, convex hulls
  - The Detection3DArray / MarkerArray publishing logic and EMA pose smoothing

That shared surface lives here as free functions plus one `ObjectDetectorBase(Node)`,
split out from the DBSCAN-specific segmentation code so a future alternate backend
(e.g. a different segmentation method) can subclass it without duplicating this
plumbing.
"""

import struct
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401

from scipy.spatial import ConvexHull, cKDTree


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


def fit_dominant_horizontal_plane(pts: np.ndarray, distance: float = 0.008,
                                  min_inliers: int = 200):
    """Fit the dominant near-horizontal plane with deterministic RANSAC.

    Returns ``(normal, offset)`` for ``normal.dot(point) + offset == 0``, or
    ``None`` when no credible tabletop is present.  Constraining the normal to
    base-frame Z prevents a large vertical object face from being mistaken for
    the table.
    """
    if len(pts) < 3:
        return None
    sample = pts
    if len(sample) > 20000:
        sample = sample[np.random.default_rng(0).choice(len(sample), 20000, replace=False)]
    rng = np.random.default_rng(1)
    best = None
    best_count = 0
    for _ in range(128):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if abs(normal[2]) < 0.85:
            continue
        offset = -float(normal @ a)
        inliers = np.abs(sample @ normal + offset) <= distance
        count = int(inliers.sum())
        if count > best_count:
            best_count, best = count, inliers
    if best is None or best_count < min_inliers:
        return None
    inlier_pts = sample[best]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ centroid)


def remove_plane(pts: np.ndarray, plane, distance: float) -> np.ndarray:
    """Return points farther than ``distance`` from a fitted plane."""
    if plane is None or len(pts) == 0:
        return pts
    normal, offset = plane
    return pts[np.abs(pts @ normal + offset) > distance]


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


def xyzl_to_pointcloud2(clusters: list, frame_id: str, stamp) -> PointCloud2:
    """Pack a list of per-object (Ni,3) float arrays into one labeled PointCloud2.

    Fields: x, y, z, label (int32 = index into `clusters`, i.e. the same
    obj_id used for Detection3D.id). Lets downstream nodes (e.g.
    contact_point_selector) recover per-object raw points from a single topic
    without redoing segmentation.
    """
    if clusters:
        pts = np.concatenate([c.astype(np.float32) for c in clusters], axis=0)
        labels = np.concatenate([
            np.full(len(c), i, dtype=np.int32) for i, c in enumerate(clusters)
        ])
    else:
        pts = np.zeros((0, 3), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)

    n = len(pts)
    dtype = np.dtype({
        'names': ['x', 'y', 'z', 'label'],
        'formats': ['<f4', '<f4', '<f4', '<i4'],
        'offsets': [0, 4, 8, 12],
        'itemsize': 16,
    })
    buf = np.empty((n,), dtype=dtype)
    buf['x'], buf['y'], buf['z'], buf['label'] = pts[:, 0], pts[:, 1], pts[:, 2], labels

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = n
    msg.is_dense = False
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.fields = [
        PointField(name='x',     offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',     offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',     offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='label', offset=12, datatype=PointField.INT32,   count=1),
    ]
    msg.data = buf.tobytes()
    return msg


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


def remove_sparse_outliers(pts: np.ndarray, k: int, std_ratio: float) -> np.ndarray:
    """Remove points sitting in a locally sparse region — e.g. a RealSense
    "flying pixel" noise trail bleeding off a depth edge, which DBSCAN's
    single-linkage chaining can otherwise merge onto a real dense object one
    point-to-point hop at a time, even though the trail as a whole sits far
    from the object.

    Unlike a centroid-distance filter, this looks at *local* density (mean
    distance to each point's k nearest neighbours), so it doesn't penalise
    points that are legitimately far from the centroid but still embedded in
    the dense body — e.g. either end of a tall/elongated object.
    """
    if len(pts) <= k:
        return pts
    tree = cKDTree(pts)
    # k+1 because a point's own nearest "neighbour" (distance 0) is itself
    dists, _ = tree.query(pts, k=k + 1)
    mean_knn_dist = dists[:, 1:].mean(axis=1)
    thresh = mean_knn_dist.mean() + std_ratio * mean_knn_dist.std()
    return pts[mean_knn_dist <= thresh]


class FrameAccumulator:
    """Fuses a sliding window of single-frame point clouds from a static scene
    into one denoised cloud, using voxel occupancy *persistence* rather than
    single-frame density.

    Rationale: with the robot at rest and out of frame, the scene genuinely
    isn't changing, so any difference between consecutive frames is sensor
    noise (RealSense "flying pixels", specular dropouts, per-frame depth
    jitter) rather than signal. A real surface point lands in the same voxel
    cell on nearly every frame; noise doesn't — it's spatially transient even
    when, within a single frame, it's locally dense enough to survive
    `remove_sparse_outliers`. Requiring a voxel to be hit by at least
    `min_hits` of the last `n_frames` frames catches exactly the noise that a
    single-frame filter structurally cannot: a noise cluster that looks dense
    *within one frame* but doesn't recur across frames.

    This replaces the single-frame `voxel_downsample` step (its output is
    already one point per surviving voxel), not `remove_sparse_outliers`,
    which is still worth running afterward to catch any residual noise voxel
    that happens to be spatially isolated but temporally persistent (e.g. a
    reflective speck that fools the depth sensor the same way every frame).
    """

    def __init__(self, n_frames: int, voxel_size: float, min_hits: int):
        self.n_frames = max(1, int(n_frames))
        self.voxel_size = voxel_size
        self.min_hits = max(1, min(int(min_hits), self.n_frames))
        self._frames: deque[np.ndarray] = deque(maxlen=self.n_frames)

    def reset(self):
        """Drop all buffered frames — call when a new activation window starts
        so a stale frame from before the object moved/was reset doesn't blend
        into the first fused cloud of the new window."""
        self._frames.clear()

    def add(self, pts: np.ndarray) -> np.ndarray | None:
        """Push one frame's (already ROI-cropped, base-frame) points.

        Returns the fused cloud once `n_frames` frames have been buffered
        (a full sliding window thereafter, refused on every call), or None
        while still warming up — callers must treat None as "no result yet",
        not "zero detections", so a warm-up frame is never mistaken for an
        empty scene.
        """
        self._frames.append(pts)
        if len(self._frames) < self.n_frames:
            return None
        return self.fuse()

    def fuse(self) -> np.ndarray:
        """Voxel-occupancy-consensus fusion of every currently buffered frame."""
        frames = [f for f in self._frames if len(f)]
        if not frames:
            return np.zeros((0, 3), dtype=np.float32)

        # Step 1: collapse each frame to one representative point per voxel it
        # touches (mirrors voxel_downsample, done per-frame so a dense cluster
        # within a single frame can't inflate that frame's "vote").
        idx_fv, pts_fv = [], []
        for fi, pts in enumerate(frames):
            idx = np.floor(pts / self.voxel_size).astype(np.int64)
            _, unique = np.unique(idx, axis=0, return_index=True)
            idx_fv.append(idx[unique])
            pts_fv.append(pts[unique])
        idx_fv = np.concatenate(idx_fv, axis=0)
        pts_fv = np.concatenate(pts_fv, axis=0)

        # Step 2: group those per-frame representatives by voxel (now ignoring
        # which frame each came from) and count *distinct frames* per voxel —
        # each frame contributed at most one point per voxel above, so this
        # count is exactly the persistence count, not a raw point count.
        voxel_ids, inverse = np.unique(idx_fv, axis=0, return_inverse=True)
        hit_counts = np.bincount(inverse, minlength=len(voxel_ids))

        # Represent each surviving voxel by the mean of its per-frame points —
        # averages down inter-frame positional jitter as a side benefit.
        sums = np.zeros((len(voxel_ids), 3), dtype=np.float64)
        np.add.at(sums, inverse, pts_fv.astype(np.float64))
        means = sums / hit_counts[:, np.newaxis]

        return means[hit_counts >= self.min_hits].astype(np.float32)


def label_color(idx: int) -> ColorRGBA:
    # Fixed palette cycles across detected objects for consistent RViz colours
    palette = [
        (0.92, 0.26, 0.21), (0.13, 0.59, 0.95), (0.30, 0.69, 0.31),
        (1.00, 0.76, 0.03), (0.61, 0.15, 0.69), (0.01, 0.74, 0.83),
    ]
    r, g, b = palette[idx % len(palette)]
    return ColorRGBA(r=r, g=g, b=b, a=0.6)


# ---------------------------------------------------------------------------
# Shared node base: params, TF, publishers, EMA-smoothed publishing
# ---------------------------------------------------------------------------

class ObjectDetectorBase(Node):
    """
    Common plumbing for object detector backend(s): ROI/voxel/smoothing params,
    TF, the Detection3D/MarkerArray/object_points publishers, per-object EMA
    pose smoothing, and marker construction.

    Subclasses declare their own backend-specific params/subscriptions and call
    `self._publish_results(header, clusters)` / `self._publish_empty(header)`
    with a list of per-object (Ni,3) point arrays in `self.base_frame`.

    On/off gate: this is compute-heavy (point cloud math every frame) but is
    only actually needed briefly — e.g. for irb120_control's perception
    snapshot, while the robot is out of the way of the object. `active` (declared param, default True — matches
    the historical always-on behaviour) gates whether subclasses' data
    callbacks do any work at all; toggle at runtime via:

        ros2 service call /object_detector/set_active std_srvs/srv/SetBool "{data: false}"

    Subclasses must add `if not self._active: return` at the top of whatever
    callback triggers segmentation (this base class has no opinion on which
    callback that is).
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # ---- Shared parameters -------------------------------------------
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('active', True)
        self.declare_parameter('roi_x_min',   0.15)
        self.declare_parameter('roi_x_max',   0.80)
        self.declare_parameter('roi_y_min',  -0.25)
        self.declare_parameter('roi_y_max',   0.25)
        self.declare_parameter('roi_z_min',  -0.02)
        self.declare_parameter('roi_z_max',   0.50)
        self.declare_parameter('voxel_size',  0.005)
        self.declare_parameter('smooth_alpha', 0.3)   # EMA weight for new frame (0=frozen, 1=raw)

        p = self.get_parameter
        self.base_frame = p('base_frame').value
        self.roi = dict(
            x=(p('roi_x_min').value, p('roi_x_max').value),
            y=(p('roi_y_min').value, p('roi_y_max').value),
            z=(p('roi_z_min').value, p('roi_z_max').value),
        )
        self.voxel_size   = p('voxel_size').value
        self.smooth_alpha = p('smooth_alpha').value
        self._active      = bool(p('active').value)

        # ---- TF -------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Publishers -------------------------------------------------------
        self.pub_det = self.create_publisher(Detection3DArray, '~/detections',    10)
        self.pub_mk  = self.create_publisher(MarkerArray,      '~/markers',      10)
        self.pub_pts = self.create_publisher(PointCloud2,      '~/object_points', 10)

        self._active_srv = self.create_service(SetBool, '~/set_active', self._on_set_active)

        # EMA state for orientation + centroid smoothing, keyed by obj_id;
        # cleared when detection is absent
        self._smooth_q:    dict[int, np.ndarray] = {}  # quaternion (x,y,z,w)
        self._smooth_pos:  dict[int, np.ndarray] = {}  # centroid (3,)
        self._smooth_axes: dict[int, np.ndarray] = {}  # 3×3 principal axes (cols)

    def _reset_smoothing(self):
        """Clear all EMA state — call when a frame yields zero detections so
        stale smoothing doesn't carry over into the next real detection."""
        self._smooth_q.clear()
        self._smooth_pos.clear()
        self._smooth_axes.clear()

    def _on_set_active(self, req, res):
        was_active = self._active
        self._active = bool(req.data)
        if self._active and not was_active:
            self._on_activate()
        if not self._active:
            # Don't leave a stale detection/hull hanging around once we stop
            # updating it — clear immediately rather than freezing in place.
            self._reset_smoothing()
            hdr = Header()
            hdr.stamp = self.get_clock().now().to_msg()
            hdr.frame_id = self.base_frame
            self._publish_empty(hdr)
        self.get_logger().info(f"active={self._active}")
        res.success = True
        res.message = f"active={self._active}"
        return res

    def _on_activate(self):
        """Hook for subclasses: called on the inactive -> active transition,
        before any new data callback runs. Override to reset any per-activation
        state (e.g. a temporal accumulator) that must not carry over from a
        previous activation window."""
        pass

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
        self.pub_pts.publish(xyzl_to_pointcloud2(clusters, self.base_frame, header.stamp))

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
        self.pub_pts.publish(xyzl_to_pointcloud2([], self.base_frame, header.stamp))

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
                # rosidl_generator_py accepts built-in Python floats here,
                # not NumPy scalar types (which otherwise abort the process
                # in geometry_msgs__msg__point__convert_from_py).
                m.points.append(Point(x=float(a[0]), y=float(a[1]), z=float(a[2])))
                m.points.append(Point(x=float(b[0]), y=float(b[1]), z=float(b[2])))
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
