"""
Press Point Selector
=====================
Given the labeled per-object point cloud published by object_detector
(~/object_points), deterministically picks a single 3D "press-and-pull"
contact point on the target object.

This is deliberately a physically-motivated selection, not just an
aesthetic "looks centered on top" pick — it targets a press-then-pull
tipping strategy: press straight down to pin the object's base against the
table (increasing friction at its near-robot pivot edge) without inducing
premature/adversarial tip torque, then pull from up high for maximum
leverage about that same pivot. That physical goal is what fixes the
priority order below — it is not free to reshuffle per object shape the
way a purely visual "looks right" heuristic would be:

  a) Top-most:  keep points within `top_z_tol` (or `top_z_tol_frac` * the
                object's Z-range, whichever is larger) of the maximum Z —
                maximizes the pull phase's lever arm about the pivot.
  b) Nearest:   within that top band, keep points within `min_x_tol` (or
                `min_x_tol_frac` * the band's own X-range) of ITS minimum
                X. Smaller X = closer to the robot base = closer to the
                pivot edge, which is what keeps the press phase's torque
                pinning the pivot down rather than fighting it. This is
                NOT trying to reach some absolute pivot X value — it's the
                closest the object's own top-face geometry gets, which for
                a non-overhanging object (nearly all of them: their top
                doesn't extend past their own base footprint) is generally
                a bit past the pivot, and that's expected, not a shortfall
                to fix. No pivot coordinate needs to be known or estimated
                for this — "min available X on the top band" already *is*
                the closest approach the geometry allows.
  c) Mid-depth: tie-break among the survivors of (a)+(b) — closest to the
                midpoint of the *whole object's* Y extent. Objects are
                expected to be placed roughly centered on the robot's Y=0
                line, so this mostly resolves near-ties, not a real
                competing objective.

  (Earlier versions of this file tried finding "front" as a Y-centered or
  whole-object X-band *before* top-most, chasing a purely visual "looks
  centered, looks front-facing" result — that fought the physical goal
  above on tilted/reclined/curved objects in three different ways across
  several reorders. Restoring top-most as the dominant criterion, with
  nearest-X as a real secondary step (not just a rare tie-break) rather
  than reshuffled ahead of it, is what actually lines up with the press-
  then-pull physics. Git history has the visual-only attempts if useful:
  `git log -- irb120_perception/press_point_selector.py`.)

The result is always one of the actual sensed points (never an average/
interpolation) — no risk of the point sinking inside a curved surface the
way a plain mean of several points would.

Points arrive in `base_frame` already (object_detector publishes
~/object_points there), so no TF lookup is needed to compute the point
itself — only to broadcast/republish it.

Orientation is a fixed parameter (default identity), not computed from a
view direction: this matches the rest of the control stack, where the
pre-squash pose orientation is fixed per calibration in
`irb120_control/object_params.json` (only x/y/z vary per object) and the
squash/pull motion itself runs along a fixed axis.

ONE-SHOT DESIGN
---------------
This node does exactly one thing and exits — no service, no standing
subscription loop, no "trigger it from a second terminal" dance:

  ros2 run irb120_perception press_point_selector

waits (briefly) for the next ~/object_points message, computes the point
once from whatever data that message holds, PRINTS the result to the
console, PUBLISHES it (pose/marker/TF, repeated for a short grace window
so RViz/late subscribers have time to catch it), then exits.

Callable from other Python files two ways:

1. Zero-ROS-node reuse — if you already have the object's points as a
   plain (N,3) array (e.g. from your own subscription), just call the pure
   functions directly, no node/spin involved:

     from irb120_perception.press_point_selector import (
         select_press_point, build_press_pose_msg, build_press_markers, build_press_tf_msg)
     point = select_press_point(object_points_xyz)
     pose_msg = build_press_pose_msg(point, quat, base_frame, stamp)

2. Full one-shot node reuse — from another node's setup or the top of its
   own control loop, construct this node once and call `.run_once(...)`
   whenever you want a fresh point (it blocks briefly waiting for a cloud
   message, then computes/prints/publishes, same as the CLI):

     from irb120_perception.press_point_selector import PressPointSelector
     pp_node = PressPointSelector()
     ...
     result = pp_node.run_once(timeout_sec=2.0)   # call this at loop start
     if result is not None:
         point = result['point']

   Note: `run_once()` calls `rclpy.spin_once()` internally to wait for
   data, so only call it from plain (non-callback) code or a single-
   threaded executor context — not from inside an already-executing
   callback of a MultiThreadedExecutor.

Publishes:
  ~/press_pose    geometry_msgs/PoseStamped   (position = press point; orientation = fixed, see above)
  ~/press_marker  visualization_msgs/MarkerArray (sphere at the point + arrow along approach)
  TF frame        base_frame -> press_frame_id
"""

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformBroadcaster


# ---------------------------------------------------------------------------
# Geometry helpers (kept local/self-contained, matching this package's
# convention of not sharing helpers across node modules). Pure numpy — no
# ROS types — so they're importable and testable without a running node.
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


def select_press_point(pts: np.ndarray,
                        top_z_tol: float = 0.003,
                        top_z_tol_frac: float = 0.05,
                        min_x_tol: float = 0.003,
                        min_x_tol_frac: float = 0.05,
                        y_tie_tol: float = 1e-6) -> np.ndarray:
    """Deterministically pick one point of an object's cloud as the
    press-and-pull contact point. See module docstring for the physical
    (press-then-pull tipping) reasoning behind the a) top-most, b) nearest,
    c) mid-depth order this applies.

    pts: (N,3) array in base_frame (X = reach/forward from the robot base —
         smaller X is closer to the robot/the tipping pivot edge; Y =
         lateral/depth; Z = height). Must be non-empty.

    top_z_tol / top_z_tol_frac and min_x_tol / min_x_tol_frac are both
    `max(absolute, fraction * range)` bands: the first around the max Z of
    the whole object, the second around the min X within that top band —
    but sized as a fraction of the *whole object's* X-span, not the top
    band's own (a near-flat top edge can have a tiny X-span of its own —
    any fraction of that collapses to the single most extreme point,
    regardless of where it lands on Y).

    Returns the selected point, (3,) float64 — always one of the actual
    input points, never an average.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] == 0:
        raise ValueError('select_press_point: empty point set')
    if pts.shape[0] == 1:
        return pts[0].copy()

    # Whole-object X-span, used below as the reference for the "nearest"
    # tolerance — NOT the top band's own X-span, which can be a sliver a
    # few mm wide (a near-flat top edge) and would make any fraction of it
    # collapse to almost nothing, landing on whichever single point happens
    # to be the literal minimum regardless of how it sits on Y.
    obj_x_span = float(pts[:, 0].max() - pts[:, 0].min())

    # a) top-most: tight band around the whole object's max Z — maximizes
    # the pull phase's lever arm about the pivot.
    z = pts[:, 2]
    z_span = float(z.max() - z.min())
    z_tol = max(top_z_tol, top_z_tol_frac * z_span)
    top = pts[z >= (z.max() - z_tol)]

    if top.shape[0] == 1:
        return top[0].copy()

    # b) nearest: within that top band, band around ITS min X — sized off
    # the whole object's own X-span (see above), not the top band's own.
    # Not trying to reach any absolute pivot coordinate — "min available X
    # near the top" already is the closest the object's own geometry gets,
    # which for a non-overhanging object is expected to land a bit past the
    # true pivot edge, not exactly on or before it.
    x_tol = max(min_x_tol, min_x_tol_frac * obj_x_span)
    near = top[top[:, 0] <= (top[:, 0].min() + x_tol)]

    if near.shape[0] == 1:
        return near[0].copy()

    # c) mid-depth tie-break: closest to the *whole object's* Y midpoint
    mid_y = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    y_dist = np.abs(near[:, 1] - mid_y)
    tied = near[y_dist <= y_dist.min() + y_tie_tol]

    if tied.shape[0] == 1:
        return tied[0].copy()

    # exceedingly rare final tie-break: prefer the nearer point
    idx = int(np.argmin(tied[:, 0]))
    return tied[idx].copy()


# ---------------------------------------------------------------------------
# Message builders — pure functions (given a point + quaternion, produce the
# ROS messages). No Node/publisher dependency, so any node can call these
# directly with its own publishers/broadcaster.
# ---------------------------------------------------------------------------

def build_press_pose_msg(point: np.ndarray, quat, frame_id: str, stamp) -> PoseStamped:
    qx, qy, qz, qw = quat
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position.x = float(point[0])
    pose.pose.position.y = float(point[1])
    pose.pose.position.z = float(point[2])
    pose.pose.orientation.x = float(qx)
    pose.pose.orientation.y = float(qy)
    pose.pose.orientation.z = float(qz)
    pose.pose.orientation.w = float(qw)
    return pose


def build_press_tf_msg(point: np.ndarray, quat, parent_frame: str, child_frame: str, stamp) -> TransformStamped:
    qx, qy, qz, qw = quat
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = parent_frame
    tf_msg.header.stamp = stamp
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = float(point[0])
    tf_msg.transform.translation.y = float(point[1])
    tf_msg.transform.translation.z = float(point[2])
    tf_msg.transform.rotation.x = float(qx)
    tf_msg.transform.rotation.y = float(qy)
    tf_msg.transform.rotation.z = float(qz)
    tf_msg.transform.rotation.w = float(qw)
    return tf_msg


def build_press_markers(point: np.ndarray, quat, frame_id: str, stamp) -> MarkerArray:
    qx, qy, qz, qw = quat
    markers = MarkerArray()

    sphere = Marker()
    sphere.header.frame_id = frame_id
    sphere.header.stamp = stamp
    sphere.ns = 'press_point'; sphere.id = 0
    sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
    sphere.pose.position.x = float(point[0])
    sphere.pose.position.y = float(point[1])
    sphere.pose.position.z = float(point[2])
    sphere.pose.orientation.x = float(qx)
    sphere.pose.orientation.y = float(qy)
    sphere.pose.orientation.z = float(qz)
    sphere.pose.orientation.w = float(qw)
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.015
    sphere.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.95)
    markers.markers.append(sphere)

    # Arrow visualizes the (fixed) approach direction: tail offset back
    # along the approach axis (away from the object), tip at the point.
    z_axis = np.array([
        2*(qx*qz + qy*qw),
        2*(qy*qz - qx*qw),
        1 - 2*(qx*qx + qy*qy),
    ])
    tail = np.asarray(point, dtype=np.float64) - z_axis * 0.05
    arrow = Marker()
    arrow.header.frame_id = frame_id
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

    return markers


# ---------------------------------------------------------------------------
# Node — one-shot: subscribes, waits for one message, computes, prints,
# publishes for a short grace window, done. No service, no standing timer.
# ---------------------------------------------------------------------------

class PressPointSelector(Node):

    def __init__(self):
        super().__init__('press_point_selector')

        self.declare_parameter('input_points',    '/object_detector/object_points')
        self.declare_parameter('base_frame',      'base_link')
        self.declare_parameter('press_frame_id',  'press_point')
        # target_object_id < 0 => auto-select the object with the highest mean Z
        # (the "prominent" object, matching object_detector's own convention)
        self.declare_parameter('target_object_id', -1)
        self.declare_parameter('top_z_tol',      0.003)  # m, absolute floor for the "top-most" band
        self.declare_parameter('top_z_tol_frac', 0.05)   # fraction of object's Z-range, whichever is larger
        self.declare_parameter('min_x_tol',      0.003)  # m, absolute floor for the "nearest" band
        self.declare_parameter('min_x_tol_frac', 0.05)   # fraction of the top band's own X-range, whichever is larger
        # Fixed contact orientation — matches the rest of the stack, where
        # pre-squash orientation is calibrated once and only position varies.
        self.declare_parameter('orientation_qx',   0.0)
        self.declare_parameter('orientation_qy',   0.0)
        self.declare_parameter('orientation_qz',   0.0)
        self.declare_parameter('orientation_qw',   1.0)

        p = self.get_parameter
        self.input_points_topic = p('input_points').value
        self.base_frame       = p('base_frame').value
        self.press_frame_id   = p('press_frame_id').value
        self.target_object_id = int(p('target_object_id').value)
        self.top_z_tol        = float(p('top_z_tol').value)
        self.top_z_tol_frac   = float(p('top_z_tol_frac').value)
        self.min_x_tol        = float(p('min_x_tol').value)
        self.min_x_tol_frac   = float(p('min_x_tol_frac').value)

        quat = np.array([p('orientation_qx').value, p('orientation_qy').value,
                          p('orientation_qz').value, p('orientation_qw').value], dtype=np.float64)
        qn = np.linalg.norm(quat)
        self.orientation = tuple(quat / qn) if qn > 1e-9 else (0.0, 0.0, 0.0, 1.0)

        self.tf_broadcaster = TransformBroadcaster(self)

        self._latest_cloud: PointCloud2 | None = None

        self.create_subscription(
            PointCloud2, self.input_points_topic, self._cloud_cb, 10)

        self.pub_pose = self.create_publisher(PoseStamped, '~/press_pose', 10)
        self.pub_mk   = self.create_publisher(MarkerArray, '~/press_marker', 10)

    # -------------------------------------------------------------------------

    def _cloud_cb(self, msg: PointCloud2):
        self._latest_cloud = msg

    # -------------------------------------------------------------------------

    def run_once(self, timeout_sec: float = 5.0,
                 publish_grace_sec: float = 1.0,
                 publish_hz: float = 10.0) -> dict | None:
        """Block until one ~/object_points message arrives (or `timeout_sec`
        elapses), compute the press point exactly once from that message,
        print the result to the console, then publish it (pose/marker/TF)
        repeatedly for `publish_grace_sec` — long enough for RViz or any
        other already-open subscriber to actually receive it before this
        call returns (a single publish can otherwise race a process exit
        and be silently dropped by a Volatile-QoS subscriber).

        Returns the result dict (`point`, `quat`, `obj_id`, `n_pts`), or
        None if no cloud arrived in time, or no object was found in it.
        """
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
        while self._latest_cloud is None and self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._latest_cloud is None:
            self._report_failure(
                f'No point cloud received on {self.input_points_topic} within {timeout_sec:.1f}s.')
            return None

        xyz, labels = unpack_labeled_pointcloud2(self._latest_cloud)
        if len(xyz) == 0:
            self._report_failure('No objects currently detected.')
            return None

        present = np.unique(labels)
        if self.target_object_id >= 0:
            if self.target_object_id not in present:
                self._report_failure(
                    f'target_object_id={self.target_object_id} not among '
                    f'currently detected ids {present.tolist()}.')
                return None
            obj_id = self.target_object_id
        else:
            # Auto: pick the object with the highest mean Z (prominent object)
            obj_id = max(present, key=lambda lbl: xyz[labels == lbl, 2].mean())

        pts = xyz[labels == obj_id].astype(np.float64)

        point = select_press_point(
            pts,
            top_z_tol=self.top_z_tol, top_z_tol_frac=self.top_z_tol_frac,
            min_x_tol=self.min_x_tol, min_x_tol_frac=self.min_x_tol_frac,
        )

        result = dict(point=point, quat=self.orientation, obj_id=int(obj_id), n_pts=len(pts))

        summary = (
            f'Press point for object {obj_id} ({len(pts)} pts): '
            f'({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}) in {self.base_frame}'
        )
        print(f'[press_point_selector] {summary}')
        self.get_logger().info(summary)

        # Publish repeatedly for a short grace window so a subscriber that
        # was already listening (RViz, another node) reliably gets it even
        # though this node is about to exit.
        n_pub = max(1, int(publish_grace_sec * publish_hz))
        period = 1.0 / publish_hz
        for _ in range(n_pub):
            self._publish_result(result, self.get_clock().now().to_msg())
            time.sleep(period)

        return result

    def _report_failure(self, message: str) -> None:
        print(f'[press_point_selector] {message}')
        self.get_logger().error(message)

    # -------------------------------------------------------------------------

    def _publish_result(self, result: dict, stamp) -> None:
        point, quat = result['point'], result['quat']
        self.pub_pose.publish(build_press_pose_msg(point, quat, self.base_frame, stamp))
        self.tf_broadcaster.sendTransform(
            build_press_tf_msg(point, quat, self.base_frame, self.press_frame_id, stamp))
        self.pub_mk.publish(build_press_markers(point, quat, self.base_frame, stamp))


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = PressPointSelector()
    try:
        result = node.run_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if result is not None else 1)
