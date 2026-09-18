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

"""

import csv
from datetime import datetime

import numpy as np
import rclpy
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import SetBool

from sensor_msgs_py import point_cloud2
from irb120_perception.contact_point_selector import select_contact_points
from irb120_control.util.runtime_log_dir import runtime_log_dir

DEFAULT_TOPIC   = '/object_detector/object_points'
DEFAULT_STANDOFF  = 0.05   # m, above the computed press point — matches the ~5cm scale of the tolerance itself
DEFAULT_TOLERANCE = 0.05   # m, full 3D distance between computed and hardcoded pre-press positions
DEFAULT_TIMEOUT    = 5.0   # s, how long to wait for one ~/object_points message

# Frame the comparison is done in. The perception stack publishes detections in
# 'base_link' (see perception.launch.py), while every pose the controller plans and
# logs against is in 'world' (arc_static.BASE_FRAME, moveit_single_shot.DEFAULT_BASE_FRAME).
# The two differ by the 21 mm bracket the robot base sits on, so the press point MUST be
# transformed before it is compared with a world-frame hardcoded position.
COMPARE_FRAME = 'world'

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


def _to_frame(node, point: np.ndarray, src_frame: str, dst_frame: str):
    """Transform a point between TF frames. Returns (point, note); point is None on failure.

    A no-op when the frames already match, so this stays correct if the perception
    stack is ever reconfigured to publish directly in the controller's frame.
    """
    if not src_frame:
        return None, 'missing source frame'
    if src_frame == dst_frame:
        return point, 'no transform needed'
    buf = getattr(node, '_tf_buffer', None)
    if buf is None:
        return None, 'node has no _tf_buffer'
    try:
        tf = buf.lookup_transform(dst_frame, src_frame, rclpy.time.Time())
    except Exception as exc:                     # TransformException and friends
        return None, str(exc)
    t = tf.transform.translation
    q = tf.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return np.asarray(point, dtype=np.float64) @ R.T + np.array([t.x, t.y, t.z]), 'ok'


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
          cloud).
    hardcoded_pos: (x, y, z) — the calibrated pre_squash position to compare
                   against. This function does NOT return a pose to move to;
                   the caller keeps using this same value for the real motion.
    label: free-text tag for the log line/CSV row (e.g. "arc_static/monitor").

    Fails closed: no cloud received in time, or no object detected, counts
    as a failed check (returns False) — same as an excessive deviation.
    """
    hardcoded = np.asarray(hardcoded_pos, dtype=np.float64)
    node._press_point_check_result = {
        'selector': 'contact_point_selector', 'ok': False, 'label': label,
    }

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

        try:
            data = point_cloud2.read_points(
                latest['msg'], field_names=('x', 'y', 'z', 'label'), skip_nans=True).reshape(-1)
            xyz = np.column_stack([data[field] for field in ('x', 'y', 'z')])
            labels = data['label']
            finite = np.isfinite(xyz).all(axis=1)
            xyz, labels = xyz[finite], labels[finite]
        except (ValueError, AssertionError, KeyError) as exc:
            _report(node, label, False, None, hardcoded, None, f'invalid object cloud: {exc}')
            return False
        if len(xyz) == 0:
            _report(node, label, False, None, hardcoded, None, 'no objects currently detected')
            return False

        present = np.unique(labels)
        obj_id = max(present, key=lambda lbl: xyz[labels == lbl, 2].mean())  # preserve prominent-object policy
        pts = xyz[labels == obj_id].astype(np.float64)

        src_frame = latest['msg'].header.frame_id
        # Selection needs Z-up geometry and the table height, not just a final
        # point transform. The calibrated table is z=0 in world.
        pts_world, conv_note = _to_frame(node, pts, src_frame, COMPARE_FRAME)
        if pts_world is None:
            _report(node, label, False, None, hardcoded, None,
                    f'cannot transform object cloud {src_frame}->{COMPARE_FRAME}: {conv_note}')
            return False
        try:
            press = select_contact_points(pts_world, table_z=0.0)['press']
        except ValueError as exc:
            _report(node, label, False, None, hardcoded, None, f'contact selector failed: {exc}')
            return False
        node._press_point_check_result = {
            'selector': 'contact_point_selector', 'ok': False,
            'reason': press.get('reason', ''),
            'candidate_counts': press.get('candidate_counts', {}),
        }
        if not press['available']:
            _report(node, label, False, None, hardcoded, None,
                    f"press unavailable: {press['reason']}; counts={press.get('candidate_counts', {})}")
            return False

        # Preserve the calibration-check convention: surface contact + vertical
        # standoff, NOT ball_center + standoff (which would add a new radius offset).
        computed = press['point'] + np.array([0., 0., standoff])
        dist = float(np.linalg.norm(computed - hardcoded))
        ok = dist <= tolerance
        _report(node, label, ok, computed, hardcoded, dist, 'OK' if ok else f'EXCEEDS {tolerance:.3f}m tolerance')

        # Stashed on the node (not just the CSV) so callers can fold it into their
        # own per-trial metadata sidecar — see save_run_metadata() in arc_static.py.
        node._press_point_check_result = {
            'selector': 'contact_point_selector',
            'candidate_counts': press.get('candidate_counts', {}),
            'contact_xyz': press['point'].tolist(),
            'label': label,
            'frame_id': COMPARE_FRAME,          # frame the comparison was actually done in
            'source_frame_id': src_frame,       # frame perception published in
            'computed_xyz': computed.tolist(),
            'hardcoded_xyz': hardcoded.tolist(),
            'dist_m': dist,
            'tolerance_m': tolerance,
            'ok': ok,
        }
        return ok
    finally:
        _set_perception_active(node, False)


def _report(node, label, ok: bool, computed, hardcoded, dist, note: str) -> None:
    node._press_point_check_result.update(ok=bool(ok), reason=note)
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
