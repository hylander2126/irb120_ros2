"""
Press Point Selector
=====================
Given the labeled per-object point cloud published by object_detector
(~/object_points), picks a single 3D "press point" on the target object:
the highest, closest-to-camera point, biased toward the lateral middle of
the object rather than an extreme corner.

Heuristic (per point, higher = better):

  score = w_height * height_score      (higher Z = better)
        + w_camera * camera_score      (closer to the camera = better)
        + w_center * center_score      (closer to the object's lateral
                                         centroid, measured in the plane
                                         perpendicular to the camera's
                                         view direction = better)

The final press point is a medoid of the top `top_fraction` scoring points:
the mean of that region is computed for noise-robustness, then snapped to
the nearest *actual* sensed point. This avoids the mean itself, which sinks
inside a curved/rounded object (the chord between two points on a convex
surface always bows inward) — the medoid instead guarantees the result
lies exactly on the real surface, at an extremity.

On-demand only: this node does NOT recompute on every incoming frame. It
caches the latest ~/object_points message and only runs the heuristic when
the ~/compute_press_point service (std_srvs/Trigger) is called — so the
target doesn't drift while the arm is mid-approach. The last computed
result is re-published at a low fixed rate (pose / marker / TF) purely for
a stable RViz preview between triggers; that republish does not touch the
point cloud or recompute anything.

Call it:
  ros2 service call /press_point_selector/compute_press_point std_srvs/srv/Trigger {}

Publishes:
  ~/press_pose    geometry_msgs/PoseStamped   (Z axis = press/approach direction)
  ~/press_marker  visualization_msgs/MarkerArray (sphere at the point + arrow along approach)
  TF frame        base_frame -> press_frame_id
"""

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformBroadcaster


# ---------------------------------------------------------------------------
# Geometry helpers (kept local/self-contained, matching this package's
# convention of not sharing helpers across node modules)
# ---------------------------------------------------------------------------

def unpack_labeled_pointcloud2(msg: PointCloud2):
    """Unpack object_detector's labeled PointCloud2 -> (xyz Nx3 float32, labels N int32)."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    dtype = np.dtype({
        'names': ['x', 'y', 'z', 'label'],
        'formats': ['<f4', '<f4', '<f4', '<i4'],
        'offsets': [0, 4, 8, 12],
        'itemsize': msg.point_step,
    })
    view = np.frombuffer(msg.data, dtype=dtype, count=n)
    xyz = np.empty((n, 3), dtype=np.float32)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = view['x'], view['y'], view['z']
    labels = view['label'].astype(np.int32).copy()
    return xyz, labels


def rotation_to_quaternion(R: np.ndarray):
    """3x3 rotation matrix -> (x,y,z,w) quaternion (Shepperd's method)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return (R[2, 1]-R[1, 2])*s, (R[0, 2]-R[2, 0])*s, (R[1, 0]-R[0, 1])*s, 0.25/s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return 0.25*s, (R[0, 1]+R[1, 0])/s, (R[0, 2]+R[2, 0])/s, (R[2, 1]-R[1, 2])/s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return (R[0, 1]+R[1, 0])/s, 0.25*s, (R[1, 2]+R[2, 1])/s, (R[0, 2]-R[2, 0])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return (R[0, 2]+R[2, 0])/s, (R[1, 2]+R[2, 1])/s, 0.25*s, (R[1, 0]-R[0, 1])/s


def frame_from_approach_axis(z_axis: np.ndarray):
    """Build a right-handed rotation matrix whose Z column is z_axis (normalized).

    X/Y are derived via Gram-Schmidt against a reference vector, falling
    back to a different reference if z_axis is nearly parallel to the
    default one (avoids a degenerate near-zero cross product).
    """
    z = z_axis / np.linalg.norm(z_axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class PressPointSelector(Node):

    def __init__(self):
        super().__init__('press_point_selector')

        self.declare_parameter('input_points',    '/object_detector/object_points')
        self.declare_parameter('base_frame',      'base_link')
        self.declare_parameter('camera_frame',    'realsense_color_optical_frame')
        self.declare_parameter('press_frame_id',  'press_point')
        # target_object_id < 0 => auto-select the object with the highest mean Z
        # (the "prominent" object, matching object_detector's own convention)
        self.declare_parameter('target_object_id', -1)
        self.declare_parameter('top_fraction',     0.12)   # fraction of best-scoring points averaged
        self.declare_parameter('min_top_points',   5)       # floor on how many points to average
        self.declare_parameter('w_height',         1.0)     # weight: higher Z is better
        self.declare_parameter('w_camera',         1.0)     # weight: closer to camera is better
        self.declare_parameter('w_center',         0.5)     # weight: closer to lateral centroid is better
        self.declare_parameter('publish_rate_hz',  5.0)     # re-publish cached result at this rate

        p = self.get_parameter
        self.base_frame       = p('base_frame').value
        self.camera_frame     = p('camera_frame').value
        self.press_frame_id   = p('press_frame_id').value
        self.target_object_id = int(p('target_object_id').value)
        self.top_fraction     = float(p('top_fraction').value)
        self.min_top_points   = int(p('min_top_points').value)
        self.w_height         = float(p('w_height').value)
        self.w_camera         = float(p('w_camera').value)
        self.w_center         = float(p('w_center').value)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._latest_cloud: PointCloud2 | None = None
        self._last_result: dict | None = None   # cached: point, quat, obj_id

        self.create_subscription(
            PointCloud2, p('input_points').value, self._cloud_cb, 10)

        self.pub_pose = self.create_publisher(PoseStamped, '~/press_pose', 10)
        self.pub_mk   = self.create_publisher(MarkerArray, '~/press_marker', 10)

        self.create_service(Trigger, '~/compute_press_point', self._compute_cb)

        rate = max(0.5, float(p('publish_rate_hz').value))
        self.create_timer(1.0 / rate, self._republish_cb)

        self.get_logger().info(
            'press_point_selector ready — call '
            'ros2 service call ~/compute_press_point std_srvs/srv/Trigger {} to select a point.')

    # -------------------------------------------------------------------------

    def _cloud_cb(self, msg: PointCloud2):
        # Cheap cache only — no computation happens here (on-demand design)
        self._latest_cloud = msg

    def _republish_cb(self):
        # Keeps the last computed result visible/fresh in RViz between triggers,
        # without recomputing anything.
        if self._last_result is not None:
            self._publish_result(self._last_result, self.get_clock().now().to_msg())

    # -------------------------------------------------------------------------

    def _compute_cb(self, request, response):
        if self._latest_cloud is None:
            response.success = False
            response.message = 'No point cloud received yet on input_points topic.'
            return response

        xyz, labels = unpack_labeled_pointcloud2(self._latest_cloud)
        if len(xyz) == 0:
            response.success = False
            response.message = 'No objects currently detected.'
            return response

        present = np.unique(labels)
        if self.target_object_id >= 0:
            if self.target_object_id not in present:
                response.success = False
                response.message = (
                    f'target_object_id={self.target_object_id} not among '
                    f'currently detected ids {present.tolist()}.')
                return response
            obj_id = self.target_object_id
        else:
            # Auto: pick the object with the highest mean Z (prominent object)
            obj_id = max(present, key=lambda lbl: xyz[labels == lbl, 2].mean())

        pts = xyz[labels == obj_id].astype(np.float64)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except Exception as e:
            response.success = False
            response.message = f'TF lookup {self.base_frame} <- {self.camera_frame} failed: {e}'
            return response
        t = tf.transform.translation
        camera_pos = np.array([t.x, t.y, t.z], dtype=np.float64)

        point, view_dir = self._select_press_point(pts, camera_pos)

        R = frame_from_approach_axis(view_dir)
        qx, qy, qz, qw = rotation_to_quaternion(R)

        result = dict(point=point, quat=(qx, qy, qz, qw), obj_id=int(obj_id), n_pts=len(pts))
        self._last_result = result

        stamp = self.get_clock().now().to_msg()
        self._publish_result(result, stamp)

        response.success = True
        response.message = (
            f'Press point for object {obj_id} ({len(pts)} pts): '
            f'({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}) in {self.base_frame}'
        )
        self.get_logger().info(response.message)
        return response

    # -------------------------------------------------------------------------

    def _select_press_point(self, pts: np.ndarray, camera_pos: np.ndarray):
        """Score every point of the target object, then pick a real point near
        the top scorers' mean (a medoid), not the mean itself.

        A plain mean of several nearby surface points sinks *inside* the
        object whenever the surface is curved/rounded — the chord between
        two points on a convex surface always bows inward. Snapping to the
        nearest actual sensed point keeps noise-robustness (the region is
        still chosen from several top-scoring points) while guaranteeing the
        result lies exactly on the real surface, at an extremity.

        Returns (point (3,), view_dir (3,) from camera toward the point).
        """
        centroid = pts.mean(axis=0)
        view_dir = normalize(centroid - camera_pos)

        # --- height score: higher Z is better ---
        z = pts[:, 2]
        z_range = z.max() - z.min()
        height_score = (z - z.min()) / z_range if z_range > 1e-9 else np.zeros_like(z)

        # --- camera score: closer to the camera is better ---
        dist = np.linalg.norm(pts - camera_pos, axis=1)
        d_range = dist.max() - dist.min()
        camera_score = (dist.max() - dist) / d_range if d_range > 1e-9 else np.zeros_like(dist)

        # --- center score: closer to the lateral centroid is better, measured
        # in the plane perpendicular to the camera's view direction (i.e. the
        # "middle of the object" as seen from the camera, not just XY middle) ---
        ref = np.array([0.0, 0.0, 1.0]) if abs(view_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        right = normalize(np.cross(ref, view_dir))
        up    = np.cross(view_dir, right)
        rel = pts - centroid
        u = rel @ right
        v = rel @ up
        lateral = np.hypot(u, v)
        l_range = lateral.max()
        center_score = 1.0 - (lateral / l_range) if l_range > 1e-9 else np.ones_like(lateral)

        score = (self.w_height * height_score
                 + self.w_camera * camera_score
                 + self.w_center * center_score)

        k = max(self.min_top_points, int(np.ceil(self.top_fraction * len(pts))))
        k = min(k, len(pts))
        top_idx = np.argpartition(score, -k)[-k:]
        top_pts = pts[top_idx]

        # Medoid: the actual sensed point nearest the top-scorers' mean.
        # Using the mean directly would sink inside a curved surface.
        mean_pt = top_pts.mean(axis=0)
        medoid = top_pts[np.argmin(np.linalg.norm(top_pts - mean_pt, axis=1))]
        return medoid, view_dir

    # -------------------------------------------------------------------------
    # Publish
    # -------------------------------------------------------------------------

    def _publish_result(self, result: dict, stamp):
        point = result['point']
        qx, qy, qz, qw = result['quat']

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        self.pub_pose.publish(pose)

        tf_msg = TransformStamped()
        tf_msg.header.frame_id = self.base_frame
        tf_msg.header.stamp = stamp
        tf_msg.child_frame_id = self.press_frame_id
        tf_msg.transform.translation.x = float(point[0])
        tf_msg.transform.translation.y = float(point[1])
        tf_msg.transform.translation.z = float(point[2])
        tf_msg.transform.rotation.x = float(qx)
        tf_msg.transform.rotation.y = float(qy)
        tf_msg.transform.rotation.z = float(qz)
        tf_msg.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(tf_msg)

        markers = MarkerArray()

        sphere = Marker()
        sphere.header.frame_id = self.base_frame
        sphere.header.stamp = stamp
        sphere.ns = 'press_point'; sphere.id = 0
        sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
        sphere.pose = pose.pose
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.015
        sphere.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.95)
        markers.markers.append(sphere)

        # Arrow visualizes the press/approach direction: tail offset back
        # along the approach axis (away from the object), tip at the point.
        z_axis = np.array([
            2*(qx*qz + qy*qw),
            2*(qy*qz - qx*qw),
            1 - 2*(qx*qx + qy*qy),
        ])
        tail = point - z_axis * 0.05
        arrow = Marker()
        arrow.header.frame_id = self.base_frame
        arrow.header.stamp = stamp
        arrow.ns = 'press_point'; arrow.id = 1
        arrow.type = Marker.ARROW; arrow.action = Marker.ADD
        arrow.scale.x = 0.004; arrow.scale.y = 0.008; arrow.scale.z = 0.01
        arrow.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9)
        arrow.pose.orientation.w = 1.0
        arrow.points = [
            Point(x=float(tail[0]), y=float(tail[1]), z=float(tail[2])),
            Point(x=float(point[0]), y=float(point[1]), z=float(point[2])),
        ]
        markers.markers.append(arrow)

        self.pub_mk.publish(markers)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PressPointSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
