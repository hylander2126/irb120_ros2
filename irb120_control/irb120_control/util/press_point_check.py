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

Logs every check (pass or fail) to `runtime_logs/press_point_check.csv`.
"""

import csv
from datetime import datetime

import numpy as np
import rclpy
from sensor_msgs.msg import PointCloud2

from irb120_perception.press_point_selector import select_press_point, unpack_labeled_pointcloud2
from irb120_control.util.runtime_log_dir import runtime_log_dir

DEFAULT_TOPIC   = '/object_detector/object_points'
DEFAULT_STANDOFF  = 0.05   # m, above the computed press point — matches the ~5cm scale of the tolerance itself
DEFAULT_TOLERANCE = 0.05   # m, full 3D distance between computed and hardcoded pre-press positions
DEFAULT_TIMEOUT    = 5.0   # s, how long to wait for one ~/object_points message


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
    latest = {}
    sub = node.create_subscription(
        PointCloud2, object_points_topic, lambda msg: latest.setdefault('msg', msg), 10)

    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while 'msg' not in latest and node.get_clock().now() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)

    hardcoded = np.asarray(hardcoded_pos, dtype=np.float64)

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

    dist = float(np.linalg.norm(computed - hardcoded))
    ok = dist <= tolerance
    _report(node, label, ok, computed, hardcoded, dist, 'OK' if ok else f'EXCEEDS {tolerance:.3f}m tolerance')
    return ok


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
