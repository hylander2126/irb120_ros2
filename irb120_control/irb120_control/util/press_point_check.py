"""
Press-point sanity check
=========================
Safety gate for a first real-hardware rollout of perception-driven press-
point selection: compute the point live from the camera, compare it against
the calibrated `pre_squash` pose already known to work for this object, and
refuse to move if they disagree by more than `tolerance` — a large mismatch
means either the object was detected wrong or isn't where it's supposed to
be, and descending anyway is exactly the failure mode this exists to catch.

This does NOT change what pose actually drives the robot: callers still use
their own calibrated `pre_squash` pose for the real motion (see
`irb120_control/object_params.json`). This only decides whether it's safe to
proceed at all, and logs the comparison either way — which doubles as a
running validation dataset for the perception pipeline itself ("agreed with
calibrated ground truth in N/M trials, mean deviation X cm").

Usage (from any Node, right before `move_to_pre_squash()`):

    from irb120_control.util.press_point_check import check_press_point

    if not check_press_point(node, node._pre_squash_pos, label=f"arc_static/{node._object}"):
        node.get_logger().error("Press-point sanity check failed — aborting before any motion.")
        return 1

Logs every check (pass or fail) to `runtime_logs/press_point_check.csv`, and
also stashes the same result as a dict on `node._press_point_check_result` for
callers to fold into their own per-trial metadata (see arc_static.py).

Also switches robot_mask_filter + object_detector into their 'active' state
for the duration of the check and back to idle immediately afterward (pass or
fail) — see each node's module docstring for the on/off gate itself. This is
the intended trigger point for that gate: the robot is still clear of the
object here, and neither node is needed again until the next check.

Also publishes a Marker at the computed point (see PRESS_POINT_MARKER_TOPIC)
whenever one was actually computed — camera_hull_recorder subscribes to this
and burns it into the recorded video. Cleared in the same finally block that
deactivates perception, so it disappears from the video at the same moment
the hull does.
"""

import csv
from datetime import datetime

import numpy as np
import rclpy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker

from irb120_perception.press_point_selector import select_press_point, unpack_labeled_pointcloud2
from irb120_control.util.runtime_log_dir import runtime_log_dir

DEFAULT_TOPIC   = '/object_detector/object_points'
DEFAULT_STANDOFF  = 0.05   # m, above the computed press point — matches the ~5cm scale of the tolerance itself
DEFAULT_TOLERANCE = 0.05   # m, full 3D distance between computed and hardcoded pre-press positions
DEFAULT_TIMEOUT    = 5.0   # s, how long to wait for one ~/object_points message

# Published once per check (whenever a point was actually computed, pass or
# fail) so it can be burned into the recorded video — see
# camera_hull_recorder.py's press_point_marker_topic. Distinct from the object
# hull's per-object palette (label_color() in perception_common.py) so it never
# gets confused with a detected object in a figure: bright magenta, nothing
# else in this pipeline uses that color.
PRESS_POINT_MARKER_TOPIC = '/press_point_marker'
_PRESS_POINT_COLOR = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)

# robot_mask_filter + object_detector are compute-heavy (point cloud math every
# frame, or a full SAM pass) but only actually needed for the moment it takes
# to run this check — the robot is still clear of the object at this point in
# every caller's sequence. Activate for the duration of the check, then idle
# again immediately after, pass or fail.
_PERCEPTION_ACTIVE_SERVICES = ('/robot_mask_filter/set_active', '/object_detector/set_active')


def _set_perception_active(node, active: bool, timeout_sec: float = 3.0) -> None:
    """Best-effort toggle of robot_mask_filter + object_detector's on/off gate.

    Never raises — a missing or slow service just logs a warning and moves on;
    this is a compute-saving optimization, not something the check's safety
    guarantee depends on (both nodes default to active if never toggled).
    """
    for service in _PERCEPTION_ACTIVE_SERVICES:
        client = node.create_client(SetBool, service)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            node.get_logger().warn(f'{service} not available — leaving as-is')
            continue
        future = client.call_async(SetBool.Request(data=active))
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if not future.done() or future.result() is None:
            node.get_logger().warn(f'{service} call timed out')


def _publish_press_point_marker(node, computed: np.ndarray, frame_id: str) -> None:
    """Publish a one-shot marker at the computed press point, for the video
    overlay (camera_hull_recorder) and/or RViz. Reuses a publisher cached on
    `node` across calls rather than creating a new one every check."""
    if not hasattr(node, '_press_point_marker_pub'):
        node._press_point_marker_pub = node.create_publisher(Marker, PRESS_POINT_MARKER_TOPIC, 10)
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = 'press_point'
    m.id = 0
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(computed[0])
    m.pose.position.y = float(computed[1])
    m.pose.position.z = float(computed[2])
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = 0.015
    m.color = _PRESS_POINT_COLOR
    node._press_point_marker_pub.publish(m)


def _clear_press_point_marker(node) -> None:
    """Clear the press-point marker — called in lockstep with deactivating
    perception so it disappears from the video at the same moment the hull
    does, rather than lingering over the scene once the robot is up against
    the object."""
    if not hasattr(node, '_press_point_marker_pub'):
        node._press_point_marker_pub = node.create_publisher(Marker, PRESS_POINT_MARKER_TOPIC, 10)
    m = Marker()
    m.header.stamp = node.get_clock().now().to_msg()
    m.ns = 'press_point'
    m.id = 0
    m.action = Marker.DELETE
    node._press_point_marker_pub.publish(m)


def check_press_point(node,
                       hardcoded_pos,
                       *,
                       object_points_topic: str = DEFAULT_TOPIC,
                       standoff: float = DEFAULT_STANDOFF,
                       tolerance: float = DEFAULT_TOLERANCE,
                       timeout_sec: float = DEFAULT_TIMEOUT,
                       label: str = '') -> bool:
    """Compute the live press point, compare its pre-press position against
    `hardcoded_pos`, log the result, and return True iff it's safe to proceed.

    node: any rclpy Node — a temporary subscription is added to it and left
          in place afterward (harmless; it just keeps caching the latest
          cloud, same pattern as press_point_selector.PressPointSelector).
    hardcoded_pos: (x, y, z) — the calibrated pre_squash position to compare
                   against. This function does NOT return a pose to move to;
                   the caller keeps using this same value for the real motion.
    label: free-text tag for the log line/CSV row (e.g. "arc_static/monitor").

    Fails closed: no cloud received in time, or no object detected, counts
    as a failed check (returns False) — same as an excessive deviation.
    """
    hardcoded = np.asarray(hardcoded_pos, dtype=np.float64)

    _set_perception_active(node, True)
    try:
        latest = {}
        sub = node.create_subscription(
            PointCloud2, object_points_topic, lambda msg: latest.setdefault('msg', msg), 10)

        deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
        while 'msg' not in latest and node.get_clock().now() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_subscription(sub)

        if 'msg' not in latest:
            _report(node, label, False, None, hardcoded, None,
                    f'no cloud received on {object_points_topic} within {timeout_sec:.1f}s')
            return False

        xyz, labels = unpack_labeled_pointcloud2(latest['msg'])
        if len(xyz) == 0:
            _report(node, label, False, None, hardcoded, None, 'no objects currently detected')
            return False

        present = np.unique(labels)
        obj_id = max(present, key=lambda lbl: xyz[labels == lbl, 2].mean())  # prominent object, same convention as press_point_selector
        pts = xyz[labels == obj_id].astype(np.float64)

        point = select_press_point(pts)
        computed = np.array([point[0], point[1], point[2] + standoff], dtype=np.float64)
        _publish_press_point_marker(node, computed, latest['msg'].header.frame_id)

        dist = float(np.linalg.norm(computed - hardcoded))
        ok = dist <= tolerance
        _report(node, label, ok, computed, hardcoded, dist, 'OK' if ok else f'EXCEEDS {tolerance:.3f}m tolerance')

        # Stashed on the node (not just the CSV) so callers can fold it into their
        # own per-trial metadata sidecar — see save_run_metadata() in arc_static.py.
        node._press_point_check_result = {
            'label': label,
            'frame_id': latest['msg'].header.frame_id,
            'computed_xyz': computed.tolist(),
            'hardcoded_xyz': hardcoded.tolist(),
            'dist_m': dist,
            'tolerance_m': tolerance,
            'ok': ok,
        }
        return ok
    finally:
        _set_perception_active(node, False)
        _clear_press_point_marker(node)


def _report(node, label, ok: bool, computed, hardcoded, dist, note: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    computed_s  = f'({computed[0]:.4f}, {computed[1]:.4f}, {computed[2]:.4f})' if computed is not None else '(n/a)'
    hardcoded_s = f'({hardcoded[0]:.4f}, {hardcoded[1]:.4f}, {hardcoded[2]:.4f})'
    dist_s = f'{dist:.4f}m' if dist is not None else 'n/a'
    line = (f'[press_point_check] {label}: computed={computed_s} hardcoded={hardcoded_s} '
            f'dist={dist_s} -> {note}')
    print(line)
    if ok:
        node.get_logger().info(line)
    else:
        node.get_logger().warn(line)

    csv_path = runtime_log_dir('.') / 'press_point_check.csv'
    is_new = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(['timestamp', 'label', 'computed_x', 'computed_y', 'computed_z',
                        'hardcoded_x', 'hardcoded_y', 'hardcoded_z', 'dist_m', 'note'])
        w.writerow([
            ts, label,
            *(computed.tolist() if computed is not None else ['', '', '']),
            *hardcoded.tolist(),
            dist if dist is not None else '',
            note,
        ])
