"""
Shared geometry helpers and publishing base class for the object detector backends.
====================================================================================
`object_detector_dbscan.py` and `object_detector_sam.py` are separate nodes (different
runtime deps — DBSCAN needs only numpy/scipy/sklearn, SAM needs torch+cv2+sam2 in a
GPU venv) but they share:

  - PointCloud2 <-> numpy conversion, TF application, PCA orientation, convex hulls
  - The Detection3DArray / MarkerArray publishing logic and EMA pose smoothing

That shared surface lives here as free functions plus one `ObjectDetectorBase(Node)`
that each backend subclasses. Backend-specific segmentation stays in the subclass.
"""

import struct

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401

from scipy.spatial import ConvexHull


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


def xyzl_to_pointcloud2(clusters: list, frame_id: str, stamp) -> PointCloud2:
    """Pack a list of per-object (Ni,3) float arrays into one labeled PointCloud2.

    Fields: x, y, z, label (int32 = index into `clusters`, i.e. the same
    obj_id used for Detection3D.id). Lets downstream nodes (e.g.
    press_point_selector) recover per-object raw points from a single topic
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
# Shared node base: params, TF, publishers, EMA-smoothed publishing
# ---------------------------------------------------------------------------

class ObjectDetectorBase(Node):
    """
    Common plumbing for both segmentation backends: ROI/voxel/smoothing params,
    TF, the Detection3D/MarkerArray/object_points publishers, per-object EMA
    pose smoothing, and marker construction.

    Subclasses declare their own backend-specific params/subscriptions and call
    `self._publish_results(header, clusters)` / `self._publish_empty(header)`
    with a list of per-object (Ni,3) point arrays in `self.base_frame`.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # ---- Shared parameters -------------------------------------------
        self.declare_parameter('base_frame', 'base_link')
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

        # ---- TF -------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Publishers -------------------------------------------------------
        self.pub_det = self.create_publisher(Detection3DArray, '~/detections',    10)
        self.pub_mk  = self.create_publisher(MarkerArray,      '~/markers',      10)
        self.pub_pts = self.create_publisher(PointCloud2,      '~/object_points', 10)

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
