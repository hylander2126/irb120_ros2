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
    in:  /realsense/depth/color/points
    out: ~/points_masked

  Aligned depth image path  (for SAM):
    in:  /realsense/aligned_depth_to_color/image_raw  +  color/camera_info
    out: ~/depth_masked   (16UC1, masked pixels set to 0)

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


def _tf_origin(tf_buffer, base_frame: str, link: str) -> np.ndarray | None:
    result = _get_tf(tf_buffer, base_frame, link)
    if result is None:
        return None
    _, t = result
    return t  # origin = translation only


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
        self.declare_parameter('input_depth',  '/realsense/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info',  '/realsense/color/camera_info')

        p = self.get_parameter
        self.base_frame     = p('base_frame').value
        self.mesh_padding   = p('robot_mask_padding').value
        self.capsule_radius = p('capsule_radius').value

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

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PointCloud2, p('input_cloud').value, self._cloud_cb, sensor_qos)
        self.create_subscription(
            CameraInfo,  p('camera_info').value, self._cam_info_cb, sensor_qos)
        self.create_subscription(
            Image, p('input_depth').value, self._depth_cb, sensor_qos)

        self._pub_cloud = self.create_publisher(PointCloud2, '~/points_masked', sensor_qos)
        self._pub_depth = self.create_publisher(Image,       '~/depth_masked',  sensor_qos)

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

    def _build_robot_mask(self, pts: np.ndarray) -> np.ndarray:
        """Return boolean keep-mask (True = not robot) for (N,3) points in base_link."""
        mask_out = np.zeros(len(pts), dtype=bool)  # True = masked (robot)

        # --- Mesh half-space tests ---
        for link, verts in self._meshes.items():
            candidates = ~mask_out
            if not candidates.any():
                break
            tf = _get_tf(self.tf_buffer, self.base_frame, link)
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
            A = _tf_origin(self.tf_buffer, self.base_frame, parent)
            B = _tf_origin(self.tf_buffer, self.base_frame, child)
            if A is None or B is None:
                continue
            inside = _capsule_inside_mask(pts[candidates], A, B, self.capsule_radius)
            mask_out[candidates] |= inside

        return ~mask_out  # keep = not masked

    def _cloud_cb(self, msg: PointCloud2):
        t0 = time.monotonic()
        xyz = _unpack_pc2(msg)
        finite = np.isfinite(xyz).all(axis=1)
        keep = finite.copy()
        if finite.any():
            tf = _get_tf(self.tf_buffer, self.base_frame, msg.header.frame_id)
            if tf is not None:
                R, t = tf
                pts_base = _apply_tf_to_points(xyz[finite].astype(np.float32), R.astype(np.float32), t.astype(np.float32))
                keep[finite] = self._build_robot_mask(pts_base)
            # if TF unavailable, keep all finite points (fail-open)
        self._pub_cloud.publish(_pack_pc2(xyz[keep], msg.header.frame_id, msg.header.stamp))
        dt = (time.monotonic() - t0) * 1000
        self.get_logger().info(
            f'points_masked in {dt:.1f} ms (in={len(xyz)}, keep={int(keep.sum())})',
            throttle_duration_sec=2.0,
        )

    def _depth_cb(self, msg: Image):
        if self._cam_info is None:
            self._pub_depth.publish(msg)
            return

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
        tf = _get_tf(self.tf_buffer, self.base_frame, msg.header.frame_id)
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
