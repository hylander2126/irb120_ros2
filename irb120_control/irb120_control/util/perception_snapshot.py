"""Perception snapshot: one episode 'perceive' stage, and the contacts motion stages use.

take_snapshot() saves into <episode>/NN_perceive/:

  object_cloud.npz         the object cloud the contacts come from (DBSCAN -> SAM cull), base_link
  object_cloud_dbscan.npz  the raw DBSCAN cloud it was culled from, for comparison
  contacts.json            full select_contact_points() output for that cloud (base_link)
  camN_color.png           one color frame per camera, taken with the cloud
  camN_depth.png           the matching aligned depth, 16-bit PNG in mm
  camN_camera_info.json    intrinsics + the camera's pose in base_link
  camN_overlay.png         color frame with the cloud (cyan), the points SAM culled (red)
                           and the contacts projected in
  bag/                     short rosbag of raw camera streams, TF, joint states, detector outputs
  snapshot.json            summary of the above

The bag is ~70 MB (BAG_SECONDS counts from when the recorder is up; it also
records during its own ~2 s startup). The DBSCAN input clouds aren't bagged
(far too big); they can be regenerated from aligned depth + camera_info.

The DBSCAN detector is switched off -> on first so its temporal accumulator
starts empty (a snapshot after a push must not fuse frames from before it),
and off again afterwards. The arm must be out of the cameras' view.
"""

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
from tf2_ros import Buffer, TransformListener

from irb120_control.util.episode import write_json_atomic
from irb120_perception.contact_point_selector import select_contact_points

CAMERAS = {"cam1": "realsense", "cam2": "realsense2", "cam3": "realsense3"}
BASE_FRAME = "base_link"
DBSCAN_TOPIC = "/object_detector/object_points"
CULL_TOPIC = "/object_detector_dbscan_sam_cull/object_points"
DETECTOR_ACTIVE_SERVICE = "/object_detector/set_active"

SETTLE_CLOUDS = 5          # non-empty DBSCAN clouds to wait for after the accumulator reset
DBSCAN_TIMEOUT_SEC = 10.0
CULL_TIMEOUT_SEC = 15.0    # the culler takes ~2 s per pass, with >= 2 s between passes
BAG_SECONDS = 1.0          # 0 = no bag
BAG_TOPICS = [f"/{ns}/{t}" for ns in CAMERAS.values()
              for t in ("color/image_raw", "color/camera_info",
                        "aligned_depth_to_color/image_raw", "aligned_depth_to_color/camera_info")] + [
    "/tf", "/tf_static", "/joint_states", DBSCAN_TOPIC, CULL_TOPIC, "/netft_data_transformed"]

CONTACT_COLORS = {"planar_push": (0, 200, 0), "forward_tip": (13, 140, 255), "press": (255, 0, 255)}  # BGR


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


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
    deadline = time.monotonic() + timeout_sec
    while True:
        try:
            tf = node._tf_buffer.lookup_transform(target, source, rclpy.time.Time())
            t, q = tf.transform.translation, tf.transform.rotation
            return _quat_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
        except Exception:
            if time.monotonic() > deadline:
                return None
            rclpy.spin_once(node, timeout_sec=0.05)


def _set_detector_active(node, active: bool) -> None:
    client = node.create_client(SetBool, DETECTOR_ACTIVE_SERVICE)
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().warn(f"{DETECTOR_ACTIVE_SERVICE} not available")
        return
    future = client.call_async(SetBool.Request(data=active))
    rclpy.spin_until_future_complete(node, future, timeout_sec=3.0)


def _record_bag(node, out, seconds: float) -> float:
    """Record BAG_TOPICS for `seconds` after the recorder is up. Returns the bag size in MB."""
    name = f"snapshot_bag_{os.getpid()}"
    with open(out / "bag_record.log", "w") as log:
        proc = subprocess.Popen(
            ["ros2", "bag", "record", "-o", str(out / "bag"), "-s", "mcap", "--storage-preset-profile",
             "zstd_fast", "--node-name", name, "--disable-keyboard-controls", "--topics", *BAG_TOPICS],
            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 10.0
    while name not in node.get_node_names() and time.monotonic() < deadline and proc.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    end = time.monotonic() + 0.5 + seconds  # 0.5 s for topic discovery
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


def _overlay(color, K, D, R, t, cloud, culled, contacts):
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
    for mode, bgr in CONTACT_COLORS.items():
        c = contacts.get(mode) or {}
        geometry = c.get("geometry") or {}
        if geometry.get("pivot") is not None:
            pivot, axis = np.asarray(geometry["pivot"]), np.asarray(geometry["axis"])
            ends = px(np.array([pivot - 0.04 * axis, pivot + 0.04 * axis]))
            if len(ends) == 2:
                cv2.line(img, ends[0], ends[1], bgr, 2, cv2.LINE_AA)
        if c.get("available"):
            p = px(np.asarray(c["point"]))
            if p:
                cv2.circle(img, p[0], 7, bgr, 2, cv2.LINE_AA)
                cv2.putText(img, mode, (p[0][0] + 10, p[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1,
                            cv2.LINE_AA)
    return img


def take_snapshot(node, episode, label: str = "") -> dict:
    """Run one perceive stage in `episode`; returns its snapshot.json record (ok=True on success)."""
    stage = episode.begin_stage("perceive", label=label)
    out = stage.path
    record = {"ok": False, "label": label}
    frames = {cam: {} for cam in CAMERAS}
    dbscan, cull = [], []
    subs = [node.create_subscription(PointCloud2, DBSCAN_TOPIC, dbscan.append, 10),
            node.create_subscription(PointCloud2, CULL_TOPIC, cull.append, 10)]
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
        record["wait_s"] = round(time.monotonic() - t0, 1)

        if BAG_SECONDS > 0:
            record["bag_mb"] = round(_record_bag(node, out, BAG_SECONDS), 1)
    finally:
        for sub in subs:
            node.destroy_subscription(sub)
        _set_detector_active(node, False)

    # 3. Clouds + contacts.
    raw = _xyz(cloud_msg) if cloud_msg is not None else np.empty((0, 3))
    obj = _xyz(cull_msg) if cull_msg is not None else np.empty((0, 3))
    np.savez_compressed(out / "object_cloud_dbscan.npz", xyz=raw)
    np.savez_compressed(out / "object_cloud.npz", xyz=obj)
    record["dbscan_points"], record["points"] = len(raw), len(obj)
    contacts = {}
    if cloud_msg is None:
        record["error"] = f"no object in the DBSCAN cloud within {DBSCAN_TIMEOUT_SEC:.0f} s"
    elif cull_msg is None:
        record["error"] = f"no SAM cull output within {CULL_TIMEOUT_SEC:.0f} s -- is the culler running?"
    elif len(obj) < 3:
        record["error"] = "SAM cull removed the whole object"
    else:
        try:
            contacts = _jsonable(select_contact_points(obj))
            record["ok"] = True
        except ValueError as exc:
            record["error"] = f"contact selection failed: {exc}"
    write_json_atomic(out / "contacts.json", contacts)
    record["available"] = {m: (contacts.get(m) or {}).get("available", False) for m in CONTACT_COLORS}

    # 4. Per-camera images, intrinsics, pose, overlay.
    kept = {tuple(p) for p in np.round(obj, 5)}
    culled = np.array([p for p in raw if tuple(np.round(p, 5)) not in kept]).reshape(-1, 3)
    bridge = CvBridge()
    for cam, m in images.items():
        if "color" not in m or "info" not in m:
            record.setdefault("missing_cameras", []).append(cam)
            continue
        K, D = np.array(m["info"].k).reshape(3, 3), np.array(m["info"].d)
        color = bridge.imgmsg_to_cv2(m["color"], desired_encoding="bgr8")
        cv2.imwrite(str(out / f"{cam}_color.png"), color)
        if "depth" in m:
            depth = bridge.imgmsg_to_cv2(m["depth"], desired_encoding="passthrough")
            if depth.dtype != np.uint16:  # 32FC1 metres -> mm
                depth = np.nan_to_num(depth * 1000.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(out / f"{cam}_depth.png"), depth)
        info = {"frame_id": m["color"].header.frame_id, "K": K.tolist(), "D": D.tolist(),
                "width": m["info"].width, "height": m["info"].height}
        tf = lookup(node, m["color"].header.frame_id, BASE_FRAME)
        if tf is not None:
            R, t = tf
            info["position_in_base"] = (-R.T @ t).tolist()
            info["base_to_optical"] = {"R": R.tolist(), "t": t.tolist()}
            cv2.imwrite(str(out / f"{cam}_overlay.png"), _overlay(color, K, D, R, t, obj, culled, contacts))
        write_json_atomic(out / f"{cam}_camera_info.json", info)

    write_json_atomic(out / "snapshot.json", record)
    stage.end("ok" if record["ok"] else "failed",
              **{k: record[k] for k in ("points", "dbscan_points", "available", "bag_mb", "error") if k in record})
    if not record["ok"]:
        node.get_logger().error(f"[snapshot] {record['error']}")
    return record


def latest_contacts(episode):
    """Contacts from the episode's last stage if it is a successful perceive, else None."""
    stages = episode.load()["stages"]
    if not stages or stages[-1]["kind"] != "perceive" or stages[-1]["status"] != "ok":
        return None
    with open(episode.path / stages[-1]["dir"] / "contacts.json") as f:
        return json.load(f)


def begin_motion_stage(node, episode, kind: str):
    """Snapshot first (unless the last stage already is one), then start the motion stage.
    Returns (stage, contacts), or (None, None) if perception failed -- don't move then.
    Call while the arm is out of the cameras' view."""
    if episode.last_stage_kind() != "perceive":
        take_snapshot(node, episode)
    contacts = latest_contacts(episode)
    if contacts is None:
        return None, None
    return episode.begin_stage(kind), contacts
