"""Perception snapshot: one episode 'perceive' stage, and the contacts motion stages use.

take_snapshot() saves into <episode>/NN_perceive/:

  object_cloud.npz         the object cloud the contacts come from (DBSCAN -> SAM cull), base_link
  object_cloud_dbscan.npz  the raw DBSCAN cloud it was culled from, for comparison
  contacts.json            full select_contact_points() output for that cloud (base_link)
  cam1_sam_mask.png        optional SAM mask matched to the selected culled cloud
  cam1_sam_color.png       exact RGB input for that mask
  camN_color.png           one color frame per camera, taken with the cloud
  Aligned depth is used in memory and retained in the bag, not separate images.
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
from irb120_perception.contact_point_selector import select_contact_points

CAMERAS = {"cam1": "realsense", "cam2": "realsense2", "cam3": "realsense3"}
BASE_FRAME = "base_link"
DBSCAN_TOPIC = "/object_detector/object_points"
CULL_TOPIC = "/object_detector_dbscan_sam_cull/object_points"
SAM_MASK_TOPIC = "/object_detector_dbscan_sam_cull/sam_mask"
SAM_COLOR_TOPIC = "/object_detector_dbscan_sam_cull/sam_color"
SAM_METADATA_TOPIC = "/object_detector_dbscan_sam_cull/sam_metadata"
DETECTOR_ACTIVE_SERVICE = "/object_detector/set_active"

SETTLE_CLOUDS = 5          # non-empty DBSCAN clouds to wait for after the accumulator reset
DBSCAN_TIMEOUT_SEC = 10.0
CULL_TIMEOUT_SEC = 15.0    # the culler takes ~2 s per pass, with >= 2 s between passes
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


def take_snapshot(node, episode, label: str = "") -> dict:
    """Run one perceive stage in `episode`; returns its snapshot.json record (ok=True on success)."""
    stage = episode.begin_stage("perceive", label=label)
    out = stage.path
    record = {"ok": False, "label": label, "min_ball_center_z": MIN_BALL_CENTER_Z}
    frames = {cam: {} for cam in CAMERAS}
    dbscan, cull = [], []
    sam_masks, sam_colors = deque(maxlen=16), deque(maxlen=16)
    sam_metadata = deque(maxlen=16)
    cam1_depths = deque(maxlen=90)
    subs = [node.create_subscription(PointCloud2, DBSCAN_TOPIC, dbscan.append, 10),
            node.create_subscription(PointCloud2, CULL_TOPIC, cull.append, 10)]
    # rclpy validates callback signatures. deque.append is a builtin whose
    # signature inspect.signature() cannot read, so use ordinary callables.
    subs.extend([node.create_subscription(Image, SAM_MASK_TOPIC,
                                          lambda msg: sam_masks.append(msg), 10),
                 node.create_subscription(Image, SAM_COLOR_TOPIC,
                                          lambda msg: sam_colors.append(msg), 10),
                 node.create_subscription(String, SAM_METADATA_TOPIC,
                                          lambda msg: sam_metadata.append(msg), 10),
                 node.create_subscription(Image, "/realsense/aligned_depth_to_color/image_raw",
                                          lambda msg: cam1_depths.append(msg),
                                          qos_profile_sensor_data)])
    for cam, ns in CAMERAS.items():
        for key, msg_type, topic in (("color", Image, f"/{ns}/color/image_raw"),
                                     ("depth", Image, f"/{ns}/aligned_depth_to_color/image_raw"),
                                     ("infra1", Image, f"/{ns}/infra1/image_rect_raw"),
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
            contacts = _jsonable(select_contact_points(obj, min_ball_center_z=MIN_BALL_CENTER_Z))
            record["ok"] = True
        except ValueError as exc:
            record["error"] = f"contact selection failed: {exc}"
    write_json_atomic(out / "contacts.json", contacts)
    record["available"] = {m: (contacts.get(m) or {}).get("available", False) for m in CONTACT_COLORS}

    # 4. Per-camera images, intrinsics, pose, overlay.
    kept = {tuple(p) for p in np.round(obj, 5)}
    culled = np.array([p for p in raw if tuple(np.round(p, 5)) not in kept]).reshape(-1, 3)
    bridge = CvBridge()
    record['sam_mask'] = {'available': False, 'reason': 'no matching mask/color pair'}
    if cull_msg is not None:
        pair = [next((m for m in queue if m.header.stamp == cull_msg.header.stamp), None)
                for queue in (sam_masks, sam_colors)]
        if all(m is not None for m in pair):
            mask, rgb = pair
            ok_mask = cv2.imwrite(str(out / 'cam1_sam_mask.png'),
                                 bridge.imgmsg_to_cv2(mask, desired_encoding='mono8'))
            ok_rgb = cv2.imwrite(str(out / 'cam1_sam_color.png'),
                                bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8'))
            record['sam_mask'] = {'available': bool(ok_mask and ok_rgb),
                                  'cloud_stamp': _stamp(cull_msg),
                                  'stamp_semantics': 'cloud association, not RGB acquisition'}
            for message in sam_metadata:
                try:
                    metadata = json.loads(message.data)
                except (ValueError, TypeError):
                    continue
                if metadata.get('cloud_stamp') == [cull_msg.header.stamp.sec, cull_msg.header.stamp.nanosec]:
                    closest = _closest_depth_frame(
                        [*cam1_depths, images.get('cam1', {}).get('depth')], metadata['color_stamp'])
                    if closest is not None:
                        sam_depth = bridge.imgmsg_to_cv2(closest, desired_encoding="passthrough")
                        if sam_depth.dtype != np.uint16:
                            sam_depth = np.nan_to_num(sam_depth * 1000.0).clip(0, 65535).astype(np.uint16)
                        np.savez_compressed(out / 'cam1_sam_depth.npz', depth_mm=sam_depth,
                                            stamp=_stamp(closest), encoding="16UC1")
                        metadata['depth_stamp'] = _stamp(closest)
                        metadata['rgb_depth_delta_s'] = abs(metadata['depth_stamp']-metadata['color_stamp'])
                    write_json_atomic(out / 'cam1_sam_metadata.json', metadata)
                    break
    review_frames = {}
    for cam, m in images.items():
        if "color" not in m or "info" not in m:
            record.setdefault("missing_cameras", []).append(cam)
            continue
        K, D = np.array(m["info"].k).reshape(3, 3), np.array(m["info"].d)
        color = bridge.imgmsg_to_cv2(m["color"], desired_encoding="bgr8")
        depth = None
        cv2.imwrite(str(out / f"{cam}_color.png"), color)
        if "infra1" in m:
            infrared = bridge.imgmsg_to_cv2(m["infra1"], desired_encoding="passthrough")
            cv2.imwrite(str(out / f"{cam}_infra1.png"), infrared)
        if "depth" in m:
            depth = bridge.imgmsg_to_cv2(m["depth"], desired_encoding="passthrough")
            if depth.dtype != np.uint16:  # 32FC1 metres -> mm
                depth = np.nan_to_num(depth * 1000.0).clip(0, 65535).astype(np.uint16)
            # Preserve the exact aligned metric frame used by contact validation.
            # The auxiliary bag may miss late-discovered camera topics if a
            # driver stalls or restarts during its short recording window.
            np.savez_compressed(out / f"{cam}_aligned_depth.npz", depth_mm=depth,
                                stamp=_stamp(m["depth"]), encoding="16UC1")
            record.setdefault("depth_frames", []).append(cam)
        info = {"frame_id": m["color"].header.frame_id, "K": K.tolist(), "D": D.tolist(),
                "width": m["info"].width, "height": m["info"].height}
        tf = lookup(node, m["color"].header.frame_id, BASE_FRAME)
        if tf is not None:
            R, t = tf
            info["position_in_base"] = (-R.T @ t).tolist()
            info["base_to_optical"] = {"R": R.tolist(), "t": t.tolist()}
            cv2.imwrite(str(out / f"{cam}_overlay.png"), _overlay(color, K, D, R, t, obj, culled, contacts, depth))
        write_json_atomic(out / f"{cam}_camera_info.json", info)
        if tf is not None:
            review_frames[cam] = (color, depth, info)

    def refresh_overlays(contacts):
        """Redraw every camera's overlay.png against a newly-applied `contacts`."""
        for cam in CAMERAS:
            info_path = out / f"{cam}_camera_info.json"
            if not info_path.exists():
                continue
            info = json.loads(info_path.read_text())
            tf = info.get("base_to_optical")
            if tf is None:
                continue
            color = cv2.imread(str(out / f"{cam}_color.png"))
            cv2.imwrite(str(out / f"{cam}_overlay.png"), _overlay(
                color, np.asarray(info["K"]), np.asarray(info["D"]),
                np.asarray(tf["R"]), np.asarray(tf["t"]), obj, culled, contacts,
                review_frames.get(cam, (None, None, None))[1]))

    # The camera-validated top-support selector is the default press path.
    # Environment overrides remain as an explicit rollback/debug mechanism:
    # `off` restores baseline selection and `shadow` records without applying.
    experiment = os.environ.get("IRB120_EXPERIMENTAL_VIEW_CHECK", "apply")
    if experiment not in ("off", "shadow", "apply"):
        raise ValueError("IRB120_EXPERIMENTAL_VIEW_CHECK must be off, shadow, or apply")
    if experiment != "off" and record["ok"]:
        from irb120_perception.experimental_view_check import evaluate
        rule = os.environ.get("IRB120_EXPERIMENTAL_PRESS_RULE", "top_support_v1")
        proposed, evidence = evaluate(out, contacts,
            cameras={cam: (depth, info) for cam, (_, depth, info) in review_frames.items()
                     if depth is not None}, rule=rule)
        write_json_atomic(out / "contacts_baseline.json", contacts)
        write_json_atomic(out / "contacts_experimental_view_check.json", proposed)
        write_json_atomic(out / "experimental_view_check.json", evidence)
        record["experimental_view_check"] = {"mode": experiment, "decision": evidence["decision"]}
        if experiment == "apply":
            contacts = proposed
            write_json_atomic(out / "contacts.json", contacts)
            record["available"] = {m: bool((contacts.get(m) or {}).get("available")) for m in CONTACT_COLORS}
            refresh_overlays(contacts)

    # RGB pivot runs alongside the production cloud pivot by default. `shadow`
    # records a candidate without using it; `apply` re-seeds select_contact_points'
    # press pivot with a reviewable SAM/RGB boundary fit once it clears the gates
    # in pivot_boundary_review.evaluate(), moving the selected press point itself
    # (not just the arc's rotation center) -- set IRB120_RGB_PIVOT=apply to use it.
    pivot_mode = os.environ.get("IRB120_RGB_PIVOT", "shadow")
    if pivot_mode not in ("off", "shadow", "apply"):
        raise ValueError("IRB120_RGB_PIVOT must be off, shadow, or apply")
    if pivot_mode != "off" and record["ok"]:
        from irb120_perception.pivot_boundary_review import (
            contacts_with_candidate_pivot, evaluate as evaluate_pivot_boundary)
        pivot_report = evaluate_pivot_boundary(out)
        pivot_record = {"mode": pivot_mode,
                        "candidate_status": pivot_report.get("candidate_status"),
                        "reason": pivot_report.get("fallback_reason"),
                        "applied": False}
        try:
            rgb_contacts = contacts_with_candidate_pivot(contacts, pivot_report)
            write_json_atomic(out / "contacts_rgb_pivot.json", rgb_contacts)
        except ValueError as exc:
            pivot_record["reason"] = str(exc)
        if pivot_mode == "apply" and pivot_report.get("candidate_status") == "reviewable":
            pivot = (pivot_report.get("geometry") or {}).get("candidate_pivot_base_m")
            try:
                reseeded = _jsonable(select_contact_points(
                    obj, min_ball_center_z=MIN_BALL_CENTER_Z, press_pivot_override=pivot))
                if not (reseeded.get("press") or {}).get("available"):
                    raise ValueError("press contact unavailable after RGB pivot reseed")
                write_json_atomic(out / "contacts_pre_rgb_pivot.json", contacts)
                contacts = reseeded
                write_json_atomic(out / "contacts.json", contacts)
                record["available"] = {m: bool((contacts.get(m) or {}).get("available")) for m in CONTACT_COLORS}
                refresh_overlays(contacts)
                pivot_record["applied"] = True
            except ValueError as exc:
                pivot_record["reason"] = str(exc)
        record["rgb_pivot"] = pivot_record

    # Cheap visual byproduct only: draw the final selected press, not a new search.
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
    graphics["elapsed_seconds"] = time.monotonic()-graphics_start
    record["contact_graphics"] = graphics
    write_json_atomic(out / "snapshot.json", record)
    stage.end("ok" if record["ok"] else "failed",
              **{k: record[k] for k in ("points", "dbscan_points", "available", "bag_mb", "error") if k in record})
    if not record["ok"]:
        node.get_logger().error(f"[snapshot] {record['error']}")
    return record


def latest_contacts(episode, filename="contacts.json"):
    """Contacts variant from the last successful perceive stage, else None."""
    stages = episode.load()["stages"]
    if not stages or stages[-1]["kind"] != "perceive" or stages[-1]["status"] != "ok":
        return None
    path = episode.path / stages[-1]["dir"] / filename
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
