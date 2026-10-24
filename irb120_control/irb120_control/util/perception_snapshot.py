"""Reviewed offline FFS episode handoff and shared motion snapshot accessors.

Live DBSCAN/SAM snapshots were retired. take_snapshot captures a fresh FFS run;
import_ffs_snapshot also accepts an explicitly reviewed unchanged scene.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import subprocess
import os
import signal

import numpy as np
import rclpy
from tf2_ros import Buffer, TransformListener

from .episode import write_json_atomic

MIN_BALL_CENTER_Z = 0.06
PUSH_MIN_BALL_CENTER_Z = -0.021 + 0.01325 + 0.010

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


def _run_ffs_command(node, command, log_path, timeout=900.):
    """Keep ROS callbacks alive while a non-motion perception subprocess runs."""
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        deadline = time.monotonic()+timeout
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'FFS step timed out; see {log_path}')
                rclpy.spin_once(node, timeout_sec=.1)
            if process.returncode:
                raise RuntimeError(f'FFS step exited {process.returncode}; see {log_path}')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def take_snapshot(node, episode, label: str = "", seed: bool = True) -> dict:
    """Fresh FFS capture -> reconstruction -> report -> contacts -> episode stage.

    Call only with a stationary object and the arm clear, as for the old snapshot
    API. The retained seed argument is ignored: FFS always reacquires. The existing
    primitive/operator/MoveIt gates still own motion approval.
    """
    from datetime import datetime
    from irb120_perception.stereo_offline_bridge import WORKSPACE, VENV_PYTHON
    run = episode.path/'ffs_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    capture, recon, contacts = (run/name for name in ('capture', 'recon', 'contacts'))
    commands = [
        ('capture', 'capture_stereo_snapshot.py', [str(capture), '--stationary-arm-clear']),
        ('reconstruct', 'reconstruct_stereo_offline.py', [str(capture), '--output', str(recon)]),
        ('report', 'report_stereo_offline.py', [str(recon)]),
        ('contacts', 'select_stereo_contacts.py', [str(recon), '--output', str(contacts)]),
    ]
    stage = episode.begin_stage('perceive', backend='offline_ffs', label=label)
    try:
        for name, script, args in commands:
            node.get_logger().info(f'[snapshot] FFS {name}: {run}')
            _run_ffs_command(node, [str(VENV_PYTHON), str(WORKSPACE/'scripts'/script), *args],
                             run/f'{name}.log')
        record = _import_ffs_snapshot(episode, run, review_status='pending_motion_gate', label=label, stage=stage)
        return record
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as exc:
        reason = str(exc)
        node.get_logger().error(f'[snapshot] FFS failed: {reason}')
        record = dict(ok=False, backend='offline_ffs', reason=reason, source_run=str(run))
        write_json_atomic(stage.path/'snapshot.json', record)
        if episode.load()['stages'][stage.index]['status'] == 'running':
            stage.end('failed', reason=reason)
        return record
    except KeyboardInterrupt:
        stage.end('aborted', reason='FFS capture/reconstruction interrupted')
        raise


def import_ffs_snapshot(episode, run_dir, *, reviewed_unchanged=False):
    """Copy a reviewed run into an episode; caller attests the scene is unchanged.

    This approval records review/scene continuity, not trajectory feasibility.
    No live services or robot commands are used. Reimporting a capture after
    intervening motion is rejected, even with a repeated approval flag.
    """
    if not reviewed_unchanged:
        raise ValueError("Review contacts and confirm the scene is unchanged with --reviewed-unchanged")
    return _import_ffs_snapshot(episode, run_dir, review_status='reviewed_unchanged')


def _import_ffs_snapshot(episode, run_dir, *, review_status, label='reviewed import', stage=None):
    run = Path(run_dir).expanduser().resolve()
    contacts_dir, recon, capture = run/'contacts', run/'recon', run/'capture'
    handoff = json.loads((contacts_dir/'handoff.json').read_text())
    metadata = json.loads((capture/'capture.json').read_text())
    contacts = json.loads((contacts_dir/'contacts.json').read_text())
    if handoff.get('frame') != 'base_link' or metadata.get('base_frame') != 'base_link':
        raise ValueError('FFS handoff must use base_link')
    if not metadata.get('stationary_arm_clear'):
        raise ValueError('Capture lacks stationary-arm-clear declaration')
    if Path(handoff['source']).resolve() != recon:
        raise ValueError('Contacts are not derived from this run/recon')
    if set(metadata.get('cameras', {})) != {'cam1', 'cam2', 'cam3'}:
        raise ValueError('Complete three-camera capture required')
    for mode in ('planar_push', 'forward_tip', 'press'):
        contact = contacts.get(mode)
        if not isinstance(contact, dict) or not isinstance(contact.get('available'), bool):
            raise ValueError(f'Incomplete contact selection: {mode}')
        if contact['available']:
            for name in ('point', 'ball_center'):
                value = np.asarray(contact[name], dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {mode} {name}')
    for cam in ('cam1', 'cam2', 'cam3'):
        for suffix in ('camera_info.json', 'color.png', 'aligned_depth.npz'):
            if not (recon/f'{cam}_{suffix}').is_file():
                raise ValueError(f'Missing calibrated view evidence: {cam}_{suffix}')
    xyz = np.load(contacts_dir/'object_cloud.npz')['xyz']
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Invalid observed object cloud')
    # Identity includes calibrated capture and raw observations, not just path/name.
    digest = hashlib.sha256()
    for name in ('capture.json', 'cam1.npz', 'cam2.npz', 'cam3.npz'):
        with (capture/name).open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
    capture_id = digest.hexdigest()
    stages = episode.load()['stages']
    if any(s['status'] == 'running' for i, s in enumerate(stages) if stage is None or i != stage.index):
        raise ValueError('Cannot import during an unfinished stage')
    for index, previous in enumerate(stages):
        if previous.get('results', {}).get('capture_id') != capture_id:
            continue
        if any(s['kind'] != 'perceive' and s['status'] != 'aborted' for s in stages[index+1:]):
            raise ValueError('This capture predates motion; acquire a fresh FFS run')
    if stage is None:
        stage = episode.begin_stage('perceive', backend='offline_ffs', label=label)
    try:
        shutil.copytree(contacts_dir, stage.path, dirs_exist_ok=True)
        shutil.copy2(capture/'capture.json', stage.path/'capture.json')
        for pattern in ('cam*_camera_info.json', 'cam*_color.png', 'cam*_aligned_depth.npz',
                        'pivot_boundary_review.json', 'forward_pivot_review.json'):
            for source in recon.glob(pattern):
                shutil.copy2(source, stage.path/source.name)
        for cam in ('cam1', 'cam2', 'cam3'):
            if not (stage.path/f'{cam}_camera_info.json').exists():
                raise ValueError(f'Missing {cam} calibration in reconstruction')
        if (stage.path/'contacts.png').exists():
            import cv2
            image = cv2.imread(str(stage.path/'contacts.png'))
            if image is None or not cv2.imwrite(str(stage.path/'final_selection.jpg'), image):
                raise ValueError('Invalid contact review image')
        from .tip_choice import choose_tip
        tip, reason = choose_tip(contacts)
        record = dict(ok=True, backend='offline_ffs', source_run=str(run), capture_id=capture_id,
                      reviewed_unchanged=review_status == 'reviewed_unchanged', review_status=review_status,
                      trajectory_checked=False,
                      tip_choice={'tip': tip, 'reason': reason},
                      available={m: contacts[m]['available'] for m in ('planar_push', 'forward_tip', 'press')})
        write_json_atomic(stage.path/'snapshot.json', record)
        from .support_cloud_recording import record_support_scene
        # Existing depth/FK consumers use the copied calibrated camera files.
        # Support-cloud rebuild is optional evidence, never a fabricated pass.
        try:
            record['support_scene'] = record_support_scene(stage.path)
        except (OSError, ValueError, KeyError) as exc:
            record['support_scene'] = {'status': 'unavailable', 'reason': str(exc)}
        write_json_atomic(stage.path/'snapshot.json', record)
        stage.end('ok', backend='offline_ffs', capture_id=capture_id, points=len(xyz))
        return record
    except Exception as exc:
        stage.end('failed', reason=str(exc))
        raise

def latest_contacts(episode, filename="contacts.json"):
    """Contacts variant from the last successful perceive stage, else None -- also None if
    the robot has moved since (aborted motion stages don't count; see Episode.last_scene_stage)."""
    stages = episode.load()['stages']
    if stages and stages[-1]['kind'] == 'perceive' and stages[-1]['status'] != 'ok':
        return None
    stage = episode.last_scene_stage()
    if stage is None or stage["kind"] != "perceive" or stage.get("backend") != "offline_ffs":
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
    if episode.last_stage_kind() != "perceive" or latest_contacts(episode, contacts_filename) is None:
        if not take_snapshot(node, episode)['ok']:
            return None, None
    contacts = latest_contacts(episode, contacts_filename)
    if contacts is None:
        return None, None
    _publish_contact_markers(node, contacts)
    return episode.begin_stage(kind), contacts


def _publish_contact_markers(node, contacts):
    """Frozen target review for existing RViz/video subscribers, never commands."""
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import MarkerArray
    from irb120_perception.contact_point_selector import build_contact_markers
    if not hasattr(node, '_ffs_contact_marker_pub'):
        node._ffs_contact_marker_pub = node.create_publisher(
            MarkerArray, '/contact_point_selector/markers',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    node._ffs_contact_marker_pub.publish(
        build_contact_markers(contacts, 'base_link', node.get_clock().now().to_msg()))
