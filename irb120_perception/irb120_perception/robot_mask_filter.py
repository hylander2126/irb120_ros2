"""
Robot Mask Filter Node
======================
Sits between the RealSense and any downstream perception node. Removes depth
measurements that fall inside the robot's body so the arm is never mistaken
for a detected object.

Two masking primitives are combined:

  Mesh half-space test  — for IRB120 links base_link..link_6.
    Each collision STL (already convex-simplified, ~100 triangles) is loaded
    once at startup. Each frame the mesh vertices are transformed to base_link
    via TF, face normals are recomputed in that frame, and a point is masked
    if it is on the interior side of ALL face planes (i.e. inside the convex
    hull).  With padding>0 each plane is shifted outward by that amount.

  Capsule test  — for ft_link and the finger (no simplified mesh available).
    A point is masked if its distance to the line segment between the two TF
    origins is less than the capsule radius.

Operates on both streams in parallel:

  PointCloud2 path  (for DBSCAN):
    in:  /realsense/depth/color/points  (+ optionally /realsense2/depth/color/points)
    out: ~/points_masked_dbscan

  Aligned depth image path  (for SAM):
    in:  /realsense/aligned_depth_to_color/image_raw  +  color/camera_info
    out: ~/depth_masked_sam   (16UC1, masked pixels set to 0)

Two-camera fusion (PointCloud2 path only):
  When `input_cloud2` is set (non-empty), this node is also where the two
  cameras' point clouds get fused for the DBSCAN backend. Each camera's cloud
  is transformed into `base_frame` independently (accurate extrinsics assumed
  — no ICP/registration refinement is done here), robot-masked using the same
  camera-agnostic mesh/capsule test, then concatenated and published as one
  cloud on ~/points_masked_dbscan — already in `base_frame`, not the original
  camera-optical frame. `object_detector_dbscan` downstream is completely
  unaware there were two cameras; it just clusters whatever cloud arrives.

  The two camera topics are not hardware-synced, so rather than requiring a
  matched pair (message_filters ApproximateTimeSynchronizer), each camera's
  latest processed cloud is cached and the merged cloud is republished on
  every new arrival from either camera, reusing the other camera's most
  recent cache entry (up to ~1 frame stale, ~33ms at 30Hz). For a static or
  slow-moving tabletop scene this is preferable to dropping frames waiting
  for an exact pair match. The SAM depth-image path is untouched by this —
  single camera only.

  Leave `input_cloud2` empty to disable fusion and run single-camera as before
  (output is now always in `base_frame` though, even with fusion disabled).

Tune radius live — no rebuild:
  ros2 param set /robot_mask_filter robot_mask_padding 0.10
"""

import os
import struct
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from tf2_ros import Buffer, TransformListener


# ---------------------------------------------------------------------------
# STL loader
# ---------------------------------------------------------------------------

def _load_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a binary STL → (face_normals N×3, vertices N×3×3), float64."""
    with open(path, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        normals = np.empty((n, 3), dtype=np.float64)
        verts   = np.empty((n, 3, 3), dtype=np.float64)
        for i in range(n):
            normals[i] = struct.unpack('<3f', f.read(12))
            verts[i, 0] = struct.unpack('<3f', f.read(12))
            verts[i, 1] = struct.unpack('<3f', f.read(12))
            verts[i, 2] = struct.unpack('<3f', f.read(12))
            f.read(2)  # attribute byte count
    return normals, verts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _apply_tf_to_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ pts.T).T + t


def _get_tf(tf_buffer, base_frame: str, link: str):
    """Look up link→base_frame transform. Returns (R 3×3, t 3) or None."""
    try:
        tf = tf_buffer.lookup_transform(
            base_frame, link,
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.02))
        tr = tf.transform.translation
        q  = tf.transform.rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ], dtype=np.float64)
        t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        return R, t
    except Exception:
        return None


def _mesh_inside_mask(pts: np.ndarray,
                      verts_local: np.ndarray,
                      R: np.ndarray, t: np.ndarray,
                      padding: float) -> np.ndarray:
    """Return boolean mask: True where pts are INSIDE the transformed convex mesh.

    verts_local: (N_faces, 3, 3) mesh vertices in link-local frame.
    R, t: rotation and translation from link frame to the pts frame (base_link).
    padding: outward shift applied to each face plane (metres).
    """
    flat_w = (verts_local.reshape(-1, 3).astype(np.float32) @ R.T.astype(np.float32)) + t.astype(np.float32)
    verts_w = flat_w.reshape(verts_local.shape[0], 3, 3)

    e1 = verts_w[:, 1] - verts_w[:, 0]   # (F, 3)
    e2 = verts_w[:, 2] - verts_w[:, 0]
    normals_w = np.cross(e1, e2)          # (F, 3)
    nlen = np.linalg.norm(normals_w, axis=1, keepdims=True)
    valid = (nlen[:, 0] > 1e-10)
    normals_w[valid] /= nlen[valid]
    normals_w = normals_w[valid].astype(np.float32)   # (F', 3)
    anchors   = verts_w[valid, 0]                     # (F', 3)

    # sd[i, f] = dot(pts[i] - anchors[f], normals_w[f])
    # Vectorised: pts @ normals_w.T − (anchors * normals_w).sum(axis=1)
    p = pts.astype(np.float32)
    sd = p @ normals_w.T - (anchors * normals_w).sum(axis=1)  # (M, F')
    return sd.max(axis=1) <= padding


def _capsule_inside_mask(pts: np.ndarray,
                         A: np.ndarray, B: np.ndarray,
                         radius: float) -> np.ndarray:
    """Return boolean mask: True where pts are within radius of segment A→B."""
    AB = B - A
    len2 = float(np.dot(AB, AB))
    p = pts.astype(np.float64)

    if len2 < 1e-9:
        diff = p - A
        return (diff * diff).sum(axis=1) <= radius ** 2

    t = np.clip((p - A) @ AB / len2, 0.0, 1.0)
    closest = A + t[:, np.newaxis] * AB
    diff = p - closest
    return (diff * diff).sum(axis=1) <= radius ** 2


# ---------------------------------------------------------------------------
# PointCloud2 helpers
# ---------------------------------------------------------------------------

def _unpack_pc2(msg: PointCloud2) -> np.ndarray:
    fields = {f.name: f for f in msg.fields}
    ox, oy, oz = fields['x'].offset, fields['y'].offset, fields['z'].offset
    step = msg.point_step
    n = msg.width * msg.height
    endian = '>' if msg.is_bigendian else '<'
    contiguous = msg.row_step == step * msg.width

    if contiguous and step >= max(ox, oy, oz) + 4:
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
        data = msg.data
        xyz = np.empty((n, 3), dtype=np.float32)
        for i in range(n):
            b = i * step
            xyz[i, 0] = struct.unpack_from('f', data, b + ox)[0]
            xyz[i, 1] = struct.unpack_from('f', data, b + oy)[0]
            xyz[i, 2] = struct.unpack_from('f', data, b + oz)[0]
    return xyz


def _pack_pc2(pts: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    pts = pts.astype(np.float32)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(pts)
    msg.is_dense = False
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(pts)
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = pts.tobytes()
    return msg


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class RobotMaskFilter(Node):

    # Links that have collision STL meshes in the irb120_control package
    MESH_LINKS = ['base_link', 'link_1', 'link_2', 'link_3', 'link_4', 'link_5', 'link_6']

    # Capsule segments for end-effector (no simplified mesh available)
    # Each tuple is (parent_link, child_link)
    CAPSULE_SEGMENTS = [
        ('link_6',      'ft_link'),
        ('ft_link',     'finger_link'),
        ('finger_link', 'finger_ball_center'),
    ]

    def __init__(self):
        super().__init__('robot_mask_filter')

        self.declare_parameter('base_frame',          'base_link')
        self.declare_parameter('robot_mask_padding',  0.04)   # metres outward expansion
        self.declare_parameter('capsule_radius',      0.05)   # capsule radius for EE links
        self.declare_parameter('input_cloud',  '/realsense/depth/color/points')
        self.declare_parameter('input_cloud2', '')  # second camera's cloud; '' = fusion disabled
        self.declare_parameter('input_depth',  '/realsense/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info',  '/realsense/color/camera_info')
        self.declare_parameter('tf_cache_rate_hz', 20.0)

        p = self.get_parameter
        self.base_frame     = p('base_frame').value
        self.mesh_padding   = p('robot_mask_padding').value
        self.capsule_radius = p('capsule_radius').value

        # Transforms barely change between one depth frame and the next, so
        # look them up on a slow timer instead of ~14x per point-cloud
        # callback (7 mesh links + 3 capsule segments x2 endpoints + camera
        # frame) — that serial TF round-tripping was the dominant cost, not
        # the (already-vectorised) mask math itself.
        self._tf_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cloud_frame: str | None = None
        self._cloud2_frame: str | None = None
        self._depth_frame: str | None = None
        self._capsule_links = sorted({link for pair in self.CAPSULE_SEGMENTS for link in pair})

        # Per-camera cache of the latest masked, base_frame-transformed cloud —
        # merged and republished whenever either camera's cloud arrives (see
        # module docstring "Two-camera fusion"). 'cam2' stays empty/unused
        # when input_cloud2 is not set.
        self._cam_pts_base: dict[str, np.ndarray] = {}

        # Load all collision meshes at startup
        mesh_dir = os.path.join(
            get_package_share_directory('irb120_control'),
            'meshes', 'irb120_3_58', 'collision')

        self._meshes: dict[str, np.ndarray] = {}  # link → verts (N,3,3)
        for link in self.MESH_LINKS:
            path = os.path.join(mesh_dir, f'{link}.stl')
            if os.path.exists(path):
                _, verts = _load_stl(path)
                self._meshes[link] = verts
                self.get_logger().info(
                    f'Loaded collision mesh: {link}.stl ({len(verts)} faces)')
            else:
                self.get_logger().warn(f'Collision mesh not found: {path}')

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._cam_info: CameraInfo | None = None

        # Best Effort matches the RealSense driver's own output — required to
        # subscribe to it at all without deliberately overriding QoS.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Reliable so RViz's default PointCloud2 display (which requests
        # Reliable unless you override it) picks these up with no extra
        # per-display QoS fiddling. Safe here: downstream consumers
        # (object_detector) already request Best Effort, and a Best-Effort
        # subscriber can always read a Reliable publisher.
        output_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PointCloud2, p('input_cloud').value, self._cloud_cb, sensor_qos)
        cloud2_topic = p('input_cloud2').value
        if cloud2_topic:
            self.create_subscription(
                PointCloud2, cloud2_topic, self._cloud2_cb, sensor_qos)
            self.get_logger().info(f'Two-camera fusion enabled — cloud2: {cloud2_topic}')
        self.create_subscription(
            CameraInfo,  p('camera_info').value, self._cam_info_cb, sensor_qos)
        self.create_subscription(
            Image, p('input_depth').value, self._depth_cb, sensor_qos)

        self._pub_cloud = self.create_publisher(PointCloud2, '~/points_masked_dbscan', output_qos)
        self._pub_depth = self.create_publisher(Image,       '~/depth_masked_sam',     output_qos)

        cache_period = 1.0 / max(1.0, float(p('tf_cache_rate_hz').value))
        self.create_timer(cache_period, self._refresh_tf_cache)

        self.get_logger().info(
            f'robot_mask_filter ready — '
            f'{len(self._meshes)} mesh link(s), '
            f'{len(self.CAPSULE_SEGMENTS)} capsule segment(s), '
            f'mesh_padding={self.mesh_padding:.3f} m, '
            f'capsule_radius={self.capsule_radius:.3f} m'
        )

    # -------------------------------------------------------------------------

    def _cam_info_cb(self, msg: CameraInfo):
        self._cam_info = msg

    def _refresh_tf_cache(self):
        """Look up every link/frame transform we need, once, on a slow timer.

        Runs off the point-cloud/depth hot path entirely. A failed lookup
        just leaves the previous cached value in place (better than flapping
        back to fail-open every tick if TF hiccups for one cycle).
        """
        links = self.MESH_LINKS + self._capsule_links
        if self._cloud_frame is not None:
            links = links + [self._cloud_frame]
        if self._cloud2_frame is not None:
            links = links + [self._cloud2_frame]
        if self._depth_frame is not None:
            links = links + [self._depth_frame]
        for link in set(links):
            tf = _get_tf(self.tf_buffer, self.base_frame, link)
            if tf is not None:
                self._tf_cache[link] = tf

    def _build_robot_mask(self, pts: np.ndarray) -> np.ndarray:
        """Return boolean keep-mask (True = not robot) for (N,3) points in base_link."""
        mask_out = np.zeros(len(pts), dtype=bool)  # True = masked (robot)

        # --- Mesh half-space tests ---
        for link, verts in self._meshes.items():
            candidates = ~mask_out
            if not candidates.any():
                break
            tf = self._tf_cache.get(link)
            if tf is None:
                continue
            R, t = tf
            inside = _mesh_inside_mask(pts[candidates], verts, R, t, self.mesh_padding)
            mask_out[candidates] |= inside

        # --- Capsule tests for end-effector ---
        for parent, child in self.CAPSULE_SEGMENTS:
            candidates = ~mask_out
            if not candidates.any():
                break
            tf_a = self._tf_cache.get(parent)
            tf_b = self._tf_cache.get(child)
            if tf_a is None or tf_b is None:
                continue
            A, B = tf_a[1], tf_b[1]  # origin = translation only
            inside = _capsule_inside_mask(pts[candidates], A, B, self.capsule_radius)
            mask_out[candidates] |= inside

        return ~mask_out  # keep = not masked

    def _cloud_cb(self, msg: PointCloud2):
        if self._cloud_frame != msg.header.frame_id:
            self._cloud_frame = msg.header.frame_id
            self._refresh_tf_cache()  # warm the cache immediately for a new frame_id
        self._process_and_publish(msg, 'cam1')

    def _cloud2_cb(self, msg: PointCloud2):
        if self._cloud2_frame != msg.header.frame_id:
            self._cloud2_frame = msg.header.frame_id
            self._refresh_tf_cache()  # warm the cache immediately for a new frame_id
        self._process_and_publish(msg, 'cam2')

    def _process_and_publish(self, msg: PointCloud2, slot: str):
        """Transform+mask one camera's cloud into base_frame, cache it under
        `slot`, then republish the concatenation of every cached camera's
        latest cloud (see module docstring "Two-camera fusion")."""
        t0 = time.monotonic()
        xyz = _unpack_pc2(msg)
        finite = np.isfinite(xyz).all(axis=1)

        tf = self._tf_cache.get(msg.header.frame_id)
        if tf is None:
            # TF unavailable — unlike the old single-camera behaviour, we can't
            # fail open here by passing points through unmasked: the output is
            # now always tagged base_frame (see module docstring), so publishing
            # this camera's points without the base_frame transform would mislabel
            # camera-frame coordinates as base_frame ones. Safer to just skip this
            # camera's update and keep republishing the other camera's/previous
            # cached data until TF recovers (self-heals via the periodic TF cache
            # refresh timer).
            self.get_logger().warn(f'TF lookup failed for {msg.header.frame_id}, '
                                   f'skipping this {slot} frame', throttle_duration_sec=5.0)
            if slot not in self._cam_pts_base:
                return
        else:
            R, t = tf
            pts_base = _apply_tf_to_points(
                xyz[finite].astype(np.float32), R.astype(np.float32), t.astype(np.float32))
            keep = self._build_robot_mask(pts_base)
            self._cam_pts_base[slot] = pts_base[keep]

        merged = (np.concatenate(list(self._cam_pts_base.values()), axis=0)
                  if self._cam_pts_base else np.zeros((0, 3), dtype=np.float32))
        self._pub_cloud.publish(_pack_pc2(merged, self.base_frame, msg.header.stamp))

        dt = (time.monotonic() - t0) * 1000
        counts = ', '.join(f'{k}={len(v)}' for k, v in self._cam_pts_base.items())
        self.get_logger().info(
            f'points_masked in {dt:.1f} ms ({counts}, merged={len(merged)})',
            throttle_duration_sec=2.0,
        )

    def _depth_cb(self, msg: Image):
        if self._cam_info is None:
            self._pub_depth.publish(msg)
            return
        if self._depth_frame != msg.header.frame_id:
            self._depth_frame = msg.header.frame_id
            self._refresh_tf_cache()  # warm the cache immediately for a new frame_id

        depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
            msg.height, msg.width).copy()

        k = self._cam_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        u, v = np.meshgrid(np.arange(msg.width), np.arange(msg.height))
        Z = depth.astype(np.float32) / 1000.0
        pts_cam = np.stack(
            [(u.ravel() - cx) * Z.ravel() / fx,
             (v.ravel() - cy) * Z.ravel() / fy,
             Z.ravel()], axis=1)

        # Transform camera points to base_link
        tf = self._tf_cache.get(msg.header.frame_id)
        if tf is None:
            self.get_logger().warn('TF lookup failed for depth mask',
                                   throttle_duration_sec=5.0)
            self._pub_depth.publish(msg)
            return
        R, t = tf
        pts_base = _apply_tf_to_points(pts_cam, R.astype(np.float32), t.astype(np.float32))

        valid = (depth.ravel() > 0)
        keep = np.ones(msg.height * msg.width, dtype=bool)
        if valid.any():
            keep[valid] = self._build_robot_mask(pts_base[valid])

        depth.ravel()[~keep] = 0

        out = Image()
        out.header       = msg.header
        out.height       = msg.height
        out.width        = msg.width
        out.encoding     = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step         = msg.step
        out.data         = depth.tobytes()
        self._pub_depth.publish(out)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = RobotMaskFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
