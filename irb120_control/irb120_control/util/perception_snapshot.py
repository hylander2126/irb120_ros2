"""Perception snapshot: one episode 'perceive' stage, and the contacts motion stages use.

take_snapshot() saves into <episode>/NN_perceive/:

  object_cloud.npz         the object cloud the contacts come from (DBSCAN -> SAM cull), base_link
  object_cloud_dbscan.npz  the raw DBSCAN cloud it was culled from, for comparison
  contacts.json            the contacts motion steps use (base_link). Selected once:
                           forward_tip + planar_push by select_contact_points(), press by
                           experimental_view_check's top-support rule, both against the
                           cam1 SAM press pivot when reviewable (pivot_boundary_review.json),
                           then the forward pivot moved to the cam3 SAM fit when reviewable
  experimental_view_check.json  press candidates and why each passed or failed
  camN_color.png           one color frame per camera, taken with the cloud
  Aligned depth is used in memory and retained in the bag, not separate images.
  camN_camera_info.json    intrinsics + the camera's pose in base_link
  camN_overlay.png         color frame with the cloud (cyan), the points SAM culled (red)
                           and the contacts projected in
  camN_contact_review.png  the press contact: pivot, pre-squash and contact fingertip spheres
  camN_forward_contact_review.png  the same for the forward-tip contact (pre-tip sphere)
  cam3_sam_mask.png        the culler's cam3 review mask (+ cam3_sam_color, cam3_sam_depth)
  cam3_forward_pivot_review.png, forward_pivot_review.json
                           the forward (+X, far-edge) pivot fit from that mask, and whether
                           the pivot is straight or curved (contacts forward_tip.pivot_shape)
  bag/                     short rosbag of raw camera streams, TF, joint states, detector outputs
  snapshot.json            summary of the above, including tip_choice: which tip
                           (press_pull_tip / forward_tip) this snapshot calls for, and why

The bag is ~40-70 MB (BAG_SECONDS counts from when the recorder has subscribed
to every topic). The DBSCAN input clouds aren't bagged
(far too big); they can be regenerated from aligned depth + camera_info.

The DBSCAN detector is switched off -> on first so its temporal accumulator
starts empty (a snapshot after a push must not fuse frames from before it),
and off again afterwards. The arm must be out of the cameras' view.

All three cameras must deliver color, aligned depth, camera info and TF; if any
is missing the snapshot fails and no contacts are selected.
"""

from collections import deque

import json
import os
import signal
import subprocess
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_srvs.srv import SetBool
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from irb120_control.util.episode import write_json_atomic
from irb120_control.util.tip_choice import choose_tip
from irb120_perception.contact_point_selector import select_contact_points

CAMERAS = {"cam1": "realsense", "cam2": "realsense2", "cam3": "realsense3"}
BASE_FRAME = "base_link"
DBSCAN_TOPIC = "/object_detector/object_points"
CULL_TOPIC = "/object_detector_dbscan_sam_cull/object_points"
SAM_MASK_TOPIC = "/object_detector_dbscan_sam_cull/sam_mask"
SAM_COLOR_TOPIC = "/object_detector_dbscan_sam_cull/sam_color"
SAM_DEPTH_TOPIC = "/object_detector_dbscan_sam_cull/sam_depth"
SAM_METADATA_TOPIC = "/object_detector_dbscan_sam_cull/sam_metadata"
REVIEW_TOPICS = {k: f"/object_detector_dbscan_sam_cull/review_sam_{k}" for k in ("mask", "color", "depth", "metadata")}
DETECTOR_ACTIVE_SERVICE = "/object_detector/set_active"
DETECTOR_STATUS_TOPIC = "/object_detector/camera_status"  # why the detector is not fusing, if it isn't

SETTLE_CLOUDS = 5          # non-empty DBSCAN clouds to wait for after the accumulator reset
DBSCAN_TIMEOUT_SEC = 10.0
CULL_TIMEOUT_SEC = 15.0    # the culler takes ~2 s per pass, with >= 2 s between passes
REVIEW_TIMEOUT_SEC = 8.0   # cam3 review mask: one more SAM call, after the cull is published
BAG_SECONDS = 1.0          # 0 = no bag
BAG_TOPICS = [f"/{ns}/{t}" for ns in CAMERAS.values()
              for t in ("color/image_raw", "color/camera_info",
                        "aligned_depth_to_color/image_raw", "aligned_depth_to_color/camera_info")] + [
    "/tf", "/tf_static", "/joint_states", DBSCAN_TOPIC, CULL_TOPIC, "/netft_data_transformed"]

# Lowest finger-ball center the selector may pick (base_link; table top is z=-0.021). Below
# this the horizontal push puts the wrist/sensor into the table: at ~18 mm above the table
# MoveIt found no valid pose. 0.06 (81 mm above the table) is the lowest the old
# hand-tuned pushes used. Also applies to the press/forward contacts, which sit far higher.
MIN_BALL_CENTER_Z = 0.06

CONTACT_COLORS = {"planar_push": (0, 200, 0), "forward_tip": (13, 140, 255), "press": (255, 0, 255)}  # BGR


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def _closest_depth_frame(candidates, target_stamp):
    """The candidate Image message closest in time to target_stamp, or None.

    Searching only the rolling cam1_depths deque can miss a better match: by
    the time the SAM cull + diagnostics wait finishes, an early depth frame
    near the settle-time snapshot (closer to target_stamp) may still beat
    everything currently in the deque.
    """
    valid = [m for m in candidates if m is not None]
    return min(valid, key=lambda m: abs(_stamp(m) - target_stamp)) if valid else None


def _json_or_empty(msg) -> dict:
    try:
        value = json.loads(msg.data)
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _review_match(review, cull_msg):
    """(mask, color, depth or None, metadata) of the review-camera mask for this cull, else None."""
    stamp = cull_msg.header.stamp
    found = [next((m for m in review[k] if m.header.stamp == stamp), None) for k in ("mask", "color", "depth")]
    metadata = next((m for m in map(_json_or_empty, review["metadata"])
                     if m.get("cloud_stamp") == [stamp.sec, stamp.nanosec]), None)
    if found[0] is None or found[1] is None or metadata is None:
        return None
    return (*found, metadata)


def _xyz(msg: PointCloud2) -> np.ndarray:
    data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True).reshape(-1)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _quat_matrix(x, y, z, w) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def lookup(node, target: str, source: str, timeout_sec: float = 2.0):
    """(R, t) mapping points in `source` into `target`, or None. Spins the node while waiting.
    Uses the node's _tf_buffer (the motion nodes have one), creating it if needed."""
    if getattr(node, "_tf_buffer", None) is None:
        node._tf_buffer = Buffer()
        node._tf_listener = TransformListener(node._tf_buffer, node)
    start = time.monotonic()
    next_report = start + 0.5  # only speak up if the wait is noticeable
    while True:
        try:
            tf = node._tf_buffer.lookup_transform(target, source, rclpy.time.Time())
            t, q = tf.transform.translation, tf.transform.rotation
            if next_report > start + 0.5:
                node.get_logger().info(f"TF {target} <- {source} received after {time.monotonic() - start:.1f}s")
            return _quat_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        except Exception:
            now = time.monotonic()
            if now - start > timeout_sec:
                node.get_logger().error(f"TF {target} <- {source} not available after {timeout_sec:.1f}s")
                return None
            if now >= next_report:
                node.get_logger().info(
                    f"Waiting for TF {target} <- {source} ({now - start:.1f}/{timeout_sec:.0f}s)...")
                next_report = now + 1.0
            rclpy.spin_once(node, timeout_sec=0.05)


def _set_detector_active(node, active: bool) -> None:
    client = node.create_client(SetBool, DETECTOR_ACTIVE_SERVICE)
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().warn(f"{DETECTOR_ACTIVE_SERVICE} not available")
        return
    future = client.call_async(SetBool.Request(data=active))
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)


def _record_bag(node, out, seconds: float) -> float:
    """Record BAG_TOPICS for `seconds` once the recorder has subscribed to all of them
    (discovery can take over a second per camera; counting from startup left cam2/cam3
    with a few frames or none). Returns the bag size in MB."""
    name = f"snapshot_bag_{os.getpid()}"
    proc = subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(out / "bag"), "-s", "mcap", "--storage-preset-profile",
         "zstd_fast", "--node-name", name, "--disable-keyboard-controls", "--topics", *BAG_TOPICS],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and proc.poll() is None:
        if name in node.get_node_names():
            subscribed = {topic for topic, _ in node.get_subscriber_names_and_types_by_node(name, "/")}
            if subscribed >= set(BAG_TOPICS):
                break
        rclpy.spin_once(node, timeout_sec=0.1)
    else:
        node.get_logger().warn("[snapshot] bag recorder did not subscribe to every topic within 10 s")
    end = time.monotonic() + seconds
    while time.monotonic() < end and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return sum(p.stat().st_size for p in (out / "bag").rglob("*") if p.is_file()) / 1e6 \
        if (out / "bag").exists() else 0.0


def _overlay(color, K, D, R, t, cloud, culled, contacts, depth=None):
    """Draw base-frame points/contacts on one camera image. (R, t) map base -> optical."""
    img = color.copy()

    def px(points):
        cam = np.atleast_2d(points) @ R.T + t
        cam = cam[cam[:, 2] > 0.05]
        if not len(cam):
            return []
        uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), K, D)
        return [tuple(int(round(c)) for c in p) for p in uv.reshape(-1, 2)]

    for points, bgr in ((culled, (0, 0, 255)), (cloud, (255, 255, 0))):
        if len(points):
            for p in px(points[:: max(1, len(points) // 4000)]):
                cv2.circle(img, p, 1, bgr, -1)
    from irb120_perception.contact_graphics import DepthDrawing
    drawing = DepthDrawing(img, depth, dict(K=K, D=D, base_to_optical=dict(R=R, t=t)), occlusion=False)
    img = drawing.image
    for mode, bgr in CONTACT_COLORS.items():
        c = contacts.get(mode) or {}
        geometry = c.get("geometry") or {}
        boundary = geometry.get("support_boundary")
        if boundary is not None:
            for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
                drawing.line(np.asarray(a), np.asarray(b), bgr)
        if geometry.get("pivot") is not None:
            pivot, axis = np.asarray(geometry["pivot"]), np.asarray(geometry["axis"])
            drawing.marker(pivot, bgr)
            direction = np.asarray(geometry["direction"])
            drawing.line(pivot, pivot + 0.04 * direction, bgr, arrow=True)
            drawing.line(pivot - 0.04 * axis, pivot + 0.04 * axis, bgr)
        if c.get("available"):
            p = px(np.asarray(c["point"]))
            if p:
                drawing.marker(np.asarray(c['point']), bgr)
                cv2.putText(img, mode, (p[0][0] + 10, p[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1,
                            cv2.LINE_AA)
    return img


def _crop_panel(img, point, info, half=110, size=320, label=""):
    """Zoomed, labeled single-camera panel centered on `point`'s projection.

    Falls back to the full (downsized) frame when `point` is None or projects
    outside/behind the camera -- still useful to show "not visible from here".
    """
    panel = img
    if point is not None:
        tf = info["base_to_optical"]
        cam = np.asarray(point) @ np.asarray(tf["R"]).T + np.asarray(tf["t"])
        if cam[2] > 0.05:
            uv, _ = cv2.projectPoints(cam.reshape(1, 3), np.zeros(3), np.zeros(3),
                                      np.asarray(info["K"], float), np.asarray(info.get("D", [])))
            x, y = uv.reshape(2)
            h, w = img.shape[:2]
            x0 = int(max(0, min(w - 2 * half, x - half)))
            y0 = int(max(0, min(h - 2 * half, y - half)))
            region = img[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if region.size:
                panel = cv2.resize(region, (size, size))
    else:
        panel = cv2.resize(img, (size, int(size * img.shape[0] / img.shape[1])))
    panel = panel.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 20), (20, 20, 20), -1)
    cv2.putText(panel, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _final_selection_montage(review_frames, contacts, pivot_record):
    """Full camera views of the final contacts, keeping the pivot in context."""
    from irb120_perception.contact_graphics import contact_graphic, DepthDrawing
    press = contacts.get("press") or {}
    if not press.get("available"):
        return None
    panels = []
    for cam in ("cam1", "cam2", "cam3"):
        if cam not in review_frames:
            continue
        color, depth, info = review_frames[cam]
        img = contact_graphic(color, depth, info, press, press.get("geometry"))
        drawing = DepthDrawing(img, depth, info, occlusion=False)
        for mode, label, shade in [("planar_push", "PUSH", (0, 230, 0)),
                                   ("forward_tip", "FORWARD TIP", (0, 140, 255))]:
            selected = contacts.get(mode) or {}
            if not selected.get("available"):
                continue
            point = np.asarray(selected["point"])
            drawing.marker(point, shade)
            geometry = selected.get("geometry") or {}
            if mode == "forward_tip" and geometry.get("pivot") is not None:
                pivot, axis = np.asarray(geometry["pivot"]), np.asarray(geometry["axis"])
                drawing.line(pivot - .04*axis, pivot + .04*axis, shade)
                drawing.marker(pivot, shade)
            direction = (selected.get("geometry") or {}).get("direction", selected.get("direction"))
            if direction is not None:
                drawing.line(point, point+.04*np.asarray(direction), shade, arrow=True)
            uv, z = drawing.project([point])
            if z[0] > .05 and np.isfinite(uv).all():
                offset = [8, 20] if mode == "forward_tip" else [8, -6]
                cv2.putText(drawing.image, label, tuple(np.rint(uv[0]+offset).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, shade, 1, cv2.LINE_AA)
        panels.append(drawing.image)
    if not panels:
        return None
    height = min(p.shape[0] for p in panels)
    return np.hstack([cv2.resize(p, (int(p.shape[1]*height/p.shape[0]), height)) for p in panels])


def take_snapshot(node, episode, label: str = "") -> dict:
    """Run one perceive stage in `episode`; returns its snapshot.json record (ok=True on success)."""
    stage = episode.begin_stage("perceive", label=label)
    out = stage.path
    record = {"ok": False, "label": label, "min_ball_center_z": MIN_BALL_CENTER_Z}
    frames = {cam: {} for cam in CAMERAS}
    dbscan, cull = [], []
    sam_masks, sam_colors, sam_depths = deque(maxlen=16), deque(maxlen=16), deque(maxlen=16)
    sam_metadata = deque(maxlen=16)
    review = {k: deque(maxlen=16) for k in REVIEW_TOPICS}
    cam1_depths = deque(maxlen=90)
    camera_status = deque(maxlen=1)
    subs = [node.create_subscription(PointCloud2, DBSCAN_TOPIC, dbscan.append, 10),
            node.create_subscription(PointCloud2, CULL_TOPIC, cull.append, 10),
            node.create_subscription(String, DETECTOR_STATUS_TOPIC,
                                     lambda msg: camera_status.append(msg.data), 10)]
    # rclpy validates callback signatures. deque.append is a builtin whose
    # signature inspect.signature() cannot read, so use ordinary callables.
    subs.extend([node.create_subscription(Image, SAM_MASK_TOPIC,
                                          lambda msg: sam_masks.append(msg), 10),
                 node.create_subscription(Image, SAM_COLOR_TOPIC,
                                          lambda msg: sam_colors.append(msg), 10),
                 # Directly paired by the culler at the same moment as sam_color, before its
                 # ~seconds-long SAM call -- preferred over the cam1_depths rolling-buffer
                 # search below, which can miss by seconds once that call runs long.
                 node.create_subscription(Image, SAM_DEPTH_TOPIC,
                                          lambda msg: sam_depths.append(msg), 10),
                 node.create_subscription(String, SAM_METADATA_TOPIC,
                                          lambda msg: sam_metadata.append(msg), 10),
                 node.create_subscription(Image, "/realsense/aligned_depth_to_color/image_raw",
                                          lambda msg: cam1_depths.append(msg),
                                          qos_profile_sensor_data)])
    subs.extend(node.create_subscription(String if k == "metadata" else Image, topic,
                                         lambda msg, q=review[k]: q.append(msg), 10)
                for k, topic in REVIEW_TOPICS.items())
    for cam, ns in CAMERAS.items():
        for key, msg_type, topic in (("color", Image, f"/{ns}/color/image_raw"),
                                     ("depth", Image, f"/{ns}/aligned_depth_to_color/image_raw"),
                                     ("info", CameraInfo, f"/{ns}/color/camera_info")):
            subs.append(node.create_subscription(
                msg_type, topic, lambda m, c=cam, k=key: frames[c].__setitem__(k, m), qos_profile_sensor_data))
    cloud_msg = cull_msg = None
    try:
        # 1. Fresh DBSCAN cloud: reset the accumulator, then let it settle.
        _set_detector_active(node, False)
        _set_detector_active(node, True)
        dbscan.clear()
        n, t0 = 0, time.monotonic()
        while n < SETTLE_CLOUDS and time.monotonic() - t0 < DBSCAN_TIMEOUT_SEC:
            rclpy.spin_once(node, timeout_sec=0.05)
            for msg in dbscan:
                if msg.width * msg.height:
                    cloud_msg, n = msg, n + 1
            dbscan.clear()
        images = {cam: dict(m) for cam, m in frames.items()}  # taken together with the cloud

        # 2. The SAM cull of that cloud (the culler republishes with its input's stamp).
        t1 = time.monotonic()
        while cloud_msg is not None and cull_msg is None and time.monotonic() - t1 < CULL_TIMEOUT_SEC:
            rclpy.spin_once(node, timeout_sec=0.05)
            cull_msg = next((m for m in cull if _stamp(m) >= _stamp(cloud_msg) - 1e-6), None)
        # Diagnostics are optional: older cullers must still support snapshots.
        if cull_msg is not None:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                diagnostic_pair = all(any(m.header.stamp == cull_msg.header.stamp for m in queue)
                                      for queue in (sam_masks, sam_colors))
                metadata_match = False
                for message in sam_metadata:
                    try:
                        metadata_match |= (json.loads(message.data).get('cloud_stamp') ==
                                           [cull_msg.header.stamp.sec,
                                            cull_msg.header.stamp.nanosec])
                    except (ValueError, TypeError):
                        pass
                if diagnostic_pair and metadata_match:
                    break
                rclpy.spin_once(node, timeout_sec=0.05)
            # The culler announces a review-camera mask in its metadata, then sends it after the cull.
            review_camera = next((m.get('review_camera') for m in map(_json_or_empty, sam_metadata)
                                  if m.get('cloud_stamp') == [cull_msg.header.stamp.sec,
                                                              cull_msg.header.stamp.nanosec]), None)
            deadline = time.monotonic() + REVIEW_TIMEOUT_SEC
            while review_camera and time.monotonic() < deadline and _review_match(review, cull_msg) is None:
                rclpy.spin_once(node, timeout_sec=0.05)
        record["wait_s"] = round(time.monotonic() - t0, 1)

        if BAG_SECONDS > 0:
            record["bag_mb"] = round(_record_bag(node, out, BAG_SECONDS), 1)
    finally:
        for sub in subs:
            node.destroy_subscription(sub)
        _set_detector_active(node, False)

    # 3. Clouds.
    raw = _xyz(cloud_msg) if cloud_msg is not None else np.empty((0, 3))
    obj = _xyz(cull_msg) if cull_msg is not None else np.empty((0, 3))
    np.savez_compressed(out / "object_cloud_dbscan.npz", xyz=raw)
    np.savez_compressed(out / "object_cloud.npz", xyz=obj)
    record["dbscan_points"], record["points"] = len(raw), len(obj)
    contacts = {}
    if cloud_msg is None:
        record["error"] = (f"no object in the DBSCAN cloud within {DBSCAN_TIMEOUT_SEC:.0f} s"
                           + (f" -- detector: {camera_status[-1]}" if camera_status else ""))
    elif cull_msg is None:
        record["error"] = f"no SAM cull output within {CULL_TIMEOUT_SEC:.0f} s -- is the culler running?"
    elif len(obj) < 3:
        record["error"] = "SAM cull removed the whole object"

    # 4. Per-camera images, intrinsics and pose. All three cameras must be up, with TF:
    # no contacts are selected from a snapshot missing one.
    kept = {tuple(p) for p in np.round(obj, 5)}
    culled = np.array([p for p in raw if tuple(np.round(p, 5)) not in kept]).reshape(-1, 3)
    bridge = CvBridge()
    sam_images = None
    record['sam_mask'] = {'available': False, 'reason': 'no matching mask/color pair'}
    if cull_msg is not None:
        pair = [next((m for m in queue if m.header.stamp == cull_msg.header.stamp), None)
                for queue in (sam_masks, sam_colors)]
        if all(m is not None for m in pair):
            mask, rgb = pair
            sam_images = (bridge.imgmsg_to_cv2(mask, desired_encoding='mono8'),
                          bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8'))
            record['sam_mask'] = {'available': True, 'storage': 'in_memory_only',
                                  'cloud_stamp': _stamp(cull_msg),
                                  'stamp_semantics': 'cloud association, not RGB acquisition'}
            for message in sam_metadata:
                try:
                    metadata = json.loads(message.data)
                except (ValueError, TypeError):
                    continue
                if metadata.get('cloud_stamp') == [cull_msg.header.stamp.sec, cull_msg.header.stamp.nanosec]:
                    # Prefer the culler's own depth frame: it grabbed this at the same
                    # moment as sam_color, before its SAM call, so the two are back-to-back
                    # topic reads -- not however long that call then took.
                    paired = next((m for m in sam_depths if m.header.stamp == cull_msg.header.stamp), None)
                    closest = paired if paired is not None else _closest_depth_frame(
                        [*cam1_depths, images.get('cam1', {}).get('depth')], metadata['color_stamp'])
                    if closest is not None:
                        sam_depth = bridge.imgmsg_to_cv2(closest, desired_encoding="passthrough")
                        if sam_depth.dtype != np.uint16:
                            sam_depth = np.nan_to_num(sam_depth * 1000.0).clip(0, 65535).astype(np.uint16)
                        np.savez_compressed(out / 'cam1_sam_depth.npz', depth_mm=sam_depth,
                                            stamp=_stamp(closest), encoding="16UC1")
                    break
    record['review_mask'] = {'available': False, 'reason': 'no review-camera mask for this cull'}
    matched = _review_match(review, cull_msg) if cull_msg is not None else None
    if matched is not None:
        mask, rgb, depth_msg, metadata = matched
        cam = metadata.get('camera', 'cam3')
        cv2.imwrite(str(out / f'{cam}_sam_mask.png'), bridge.imgmsg_to_cv2(mask, desired_encoding='mono8'))
        cv2.imwrite(str(out / f'{cam}_sam_color.png'), bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8'))
        if depth_msg is not None:
            sam_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if sam_depth.dtype != np.uint16:
                sam_depth = np.nan_to_num(sam_depth * 1000.0).clip(0, 65535).astype(np.uint16)
            np.savez_compressed(out / f'{cam}_sam_depth.npz', depth_mm=sam_depth,
                                stamp=_stamp(depth_msg), encoding="16UC1")
        record['review_mask'] = {'available': True, 'camera': cam, 'file': f'{cam}_sam_mask.png'}
    review_frames = {}
    camera_errors = []
    for cam in CAMERAS:
        m = images[cam]
        missing = [k for k in ("color", "depth", "info") if k not in m]
        if missing:
            camera_errors.append(f"{cam}: no {'/'.join(missing)} received")
            continue
        K, D = np.array(m["info"].k).reshape(3, 3), np.array(m["info"].d)
        color = bridge.imgmsg_to_cv2(m["color"], desired_encoding="bgr8")
        cv2.imwrite(str(out / f"{cam}_color.png"), color)
        depth = bridge.imgmsg_to_cv2(m["depth"], desired_encoding="passthrough")
        if depth.dtype != np.uint16:  # 32FC1 metres -> mm
            depth = np.nan_to_num(depth * 1000.0).clip(0, 65535).astype(np.uint16)
        # Preserve the exact aligned metric frame used by contact validation.
        np.savez_compressed(out / f"{cam}_aligned_depth.npz", depth_mm=depth,
                            stamp=_stamp(m["depth"]), encoding="16UC1")
        info = {"frame_id": m["color"].header.frame_id, "K": K.tolist(), "D": D.tolist(),
                "width": m["info"].width, "height": m["info"].height}
        tf = lookup(node, m["color"].header.frame_id, BASE_FRAME)
        if tf is None:
            camera_errors.append(f"{cam}: no TF {BASE_FRAME} -> {m['color'].header.frame_id}")
        else:
            R, t = tf
            info["position_in_base"] = (-R.T @ t).tolist()
            info["base_to_optical"] = {"R": R.tolist(), "t": t.tolist()}
            review_frames[cam] = (color, depth, info)
        write_json_atomic(out / f"{cam}_camera_info.json", info)
    if camera_errors:
        record["camera_errors"] = camera_errors
        record.setdefault("error", "camera(s) not available: " + "; ".join(camera_errors))

    def refresh_overlays(contacts):
        """Draw every camera's overlay.png against `contacts`."""
        for cam, (color, depth, info) in review_frames.items():
            tf = info["base_to_optical"]
            cv2.imwrite(str(out / f"{cam}_overlay.png"), _overlay(
                color, np.asarray(info["K"]), np.asarray(info["D"]),
                np.asarray(tf["R"]), np.asarray(tf["t"]), obj, culled, contacts, depth))

    if "error" in record:
        write_json_atomic(out / "contacts.json", contacts)
        record["available"] = {m: False for m in CONTACT_COLORS}
        refresh_overlays(contacts)
        write_json_atomic(out / "snapshot.json", record)
        stage.end("failed", **{k: record[k] for k in ("points", "dbscan_points", "available", "bag_mb", "error")
                               if k in record})
        node.get_logger().error(f"[snapshot] FAILED, no contacts selected: {record['error']}")
        return record

    # 5. Press pivot: where the cam1 SAM mask meets the table (pivot_boundary_review).
    # Press selection uses it whenever the fit is reviewable, else the cloud estimate.
    from irb120_perception.pivot_boundary_review import evaluate as evaluate_press_pivot
    pivot_report = evaluate_press_pivot(out, sam_images=sam_images)
    press_pivot = None
    if pivot_report.get("candidate_status") == "reviewable":
        press_pivot = (pivot_report.get("geometry") or {}).get("candidate_pivot_base_m")
    record["rgb_pivot"] = {"applied": press_pivot is not None,
                           "candidate_status": pivot_report.get("candidate_status"),
                           "reason": None if press_pivot is not None else pivot_report.get("fallback_reason")}

    # 6. Contacts, selected once: forward_tip and planar_push by select_contact_points,
    # press by the top-support rule (experimental_view_check), against that press pivot.
    from irb120_perception.experimental_view_check import evaluate as evaluate_press
    cameras = {cam: (depth, info) for cam, (_, depth, info) in review_frames.items()}
    try:
        contacts = _jsonable(select_contact_points(obj, min_ball_center_z=MIN_BALL_CENTER_Z,
                                                   press_pivot_override=press_pivot))
        contacts, evidence = evaluate_press(out, contacts, cameras=cameras, rule="top_support_v1",
                                            press_pivot_override=press_pivot)
        write_json_atomic(out / "experimental_view_check.json", evidence)
        record["experimental_view_check"] = {"decision": evidence["decision"]}
        record["ok"] = True
    except ValueError as exc:
        record["error"] = f"contact selection failed: {exc}"

    # 7. Forward-tip pivot from the cam3 mask: records whether it is straight or curved
    # (choose_tip reads it) and, when reviewable, moves the forward pivot to that fit.
    if record["ok"]:
        from irb120_perception.forward_pivot_review import (
            contacts_with_forward_pivot, evaluate as evaluate_forward_pivot)
        forward_report = evaluate_forward_pivot(out, contacts)
        contacts, applied, reason = contacts_with_forward_pivot(contacts, forward_report)
        record["forward_pivot"] = {"shape": forward_report["shape"],
                                   "candidate_status": forward_report["candidate_status"],
                                   "applied": applied, "reason": reason}
    write_json_atomic(out / "contacts.json", contacts)
    record["available"] = {m: bool((contacts.get(m) or {}).get("available")) for m in CONTACT_COLORS}
    refresh_overlays(contacts)

    # 8. Which tip this snapshot calls for, and why (the same rule run_pipeline uses
    # for its first tip): forward tip about a straight pivot edge, else press-pull.
    if record["ok"]:
        tip, reason = choose_tip(contacts)
        record["tip_choice"] = {"tip": tip, "reason": reason}
        node.get_logger().info(f"[snapshot] tip choice: {tip} ({reason})")

    # Cheap visual byproduct only: draw the final selected press and forward contacts, not a new search.
    from irb120_perception.contact_graphics import contact_graphic
    graphics_start = time.monotonic()
    graphics = {"files": [], "style": "flat projection"}
    press = contacts.get("press") or {}
    for cam, (color, depth, info) in review_frames.items():
        name = f"{cam}_contact_review.png"
        try:
            img = contact_graphic(color, depth, info, press, press.get("geometry"))
            if not cv2.imwrite(str(out / name), img):
                raise OSError(f"could not write {name}")
            graphics["files"].append(name)
        except (ValueError, KeyError, OSError, cv2.error) as exc:
            graphics.setdefault("errors", {})[cam] = str(exc)
            node.get_logger().warn(f"[snapshot] contact graphics: {exc}")
    forward = contacts.get("forward_tip") or {}
    if forward.get("available"):
        from irb120_control.util.forward_tip_motion import TIP_STANDOFF, tip_targets
        try:
            pre = tip_targets(forward, obj, TIP_STANDOFF)[3]
        except ValueError as exc:  # fwd_tip would refuse this contact too; draw without a pre-tip
            pre = np.asarray(forward["ball_center"])
            graphics.setdefault("errors", {})["forward_pre_tip"] = str(exc)
        for cam, (color, depth, info) in review_frames.items():
            name = f"{cam}_forward_contact_review.png"
            try:
                img = contact_graphic(color, depth, info, forward, forward.get("geometry"),
                                      pre=pre, reference_arrow=False)
                if not cv2.imwrite(str(out / name), img):
                    raise OSError(f"could not write {name}")
                graphics["files"].append(name)
            except (ValueError, KeyError, OSError, cv2.error) as exc:
                graphics.setdefault("errors", {})[f"{cam}_forward"] = str(exc)
                node.get_logger().warn(f"[snapshot] forward contact graphics: {exc}")
    graphics["elapsed_seconds"] = time.monotonic()-graphics_start
    record["contact_graphics"] = graphics

    # One multi-camera strip for the combination actually used (SAM pivot +
    # its press point when applied, else the cloud-pivot fallback) -- the
    # same three-camera view for every contact.
    try:
        montage = _final_selection_montage(review_frames, contacts, record.get("rgb_pivot") or {})
        if montage is not None and cv2.imwrite(str(out / "final_selection.jpg"), montage):
            record["final_selection"] = {"file": "final_selection.jpg"}
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        node.get_logger().warn(f"[snapshot] final selection montage: {exc}")

    write_json_atomic(out / "snapshot.json", record)
    stage.end("ok" if record["ok"] else "failed",
              **{k: record[k] for k in ("points", "dbscan_points", "available", "bag_mb", "error") if k in record},
              **({"tip": record["tip_choice"]["tip"], "tip_reason": record["tip_choice"]["reason"]}
                 if "tip_choice" in record else {}))
    if not record["ok"]:
        node.get_logger().error(f"[snapshot] {record['error']}")
    return record


def latest_contacts(episode, filename="contacts.json"):
    """Contacts variant from the last successful perceive stage, else None -- also None if
    the robot has moved since (aborted motion stages don't count; see Episode.last_scene_stage)."""
    stage = episode.last_scene_stage()
    if stage is None or stage["kind"] != "perceive":
        return None
    path = episode.path / stage["dir"] / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def latest_object_cloud(episode) -> np.ndarray:
    """The object cloud (base_link) from the episode's most recent successful perceive stage."""
    stage = next(s for s in reversed(episode.load()["stages"]) if s["kind"] == "perceive" and s["status"] == "ok")
    return np.load(episode.path / stage["dir"] / "object_cloud.npz")["xyz"]


def begin_motion_stage(node, episode, kind: str, contacts_filename="contacts.json"):
    """Snapshot first (unless the last stage already is one), then start the motion stage.
    Returns (stage, contacts), or (None, None) if perception failed -- don't move then.
    Call while the arm is out of the cameras' view."""
    if episode.last_stage_kind() != "perceive":
        take_snapshot(node, episode)
    contacts = latest_contacts(episode, contacts_filename)
    if contacts is None:
        return None, None
    return episode.begin_stage(kind), contacts
