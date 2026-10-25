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


FFS_WORKER_REQUEST = '/ffs_worker/request'
FFS_WORKER_STATUS = '/ffs_worker/status'
FFS_WORKER_DISCOVERY_SEC = 1.5


def _ffs_worker(node):
    """(publisher, status messages) for the warm FFS worker, or None if none is ready.
    The worker subscribes to the request topic only once its models are loaded."""
    from std_msgs.msg import String
    if getattr(node, '_ffs_worker_pub', None) is None:
        node._ffs_worker_status = []
        node._ffs_worker_pub = node.create_publisher(String, FFS_WORKER_REQUEST, 10)
        node._ffs_worker_sub = node.create_subscription(
            String, FFS_WORKER_STATUS, lambda m: node._ffs_worker_status.append(m.data), 50)
    deadline = time.monotonic()+FFS_WORKER_DISCOVERY_SEC
    while node._ffs_worker_pub.get_subscription_count() == 0:
        if time.monotonic() >= deadline:
            return None
        rclpy.spin_once(node, timeout_sec=.05)
    return node._ffs_worker_pub, node._ffs_worker_status


def _run_ffs_worker(node, worker, request, timeout=900.):
    """Send one request to the warm worker and spin until its done/error status."""
    import uuid
    from std_msgs.msg import String
    publisher, statuses = worker
    request = dict(request, id=uuid.uuid4().hex)
    statuses.clear()
    publisher.publish(String(data=json.dumps(request)))
    deadline = time.monotonic()+timeout
    seen = 0
    while True:
        while seen < len(statuses):
            status = json.loads(statuses[seen])
            seen += 1
            if status.get('id') != request['id']:
                continue
            if status['event'] in ('error', 'busy'):
                raise RuntimeError(f"FFS worker: {status.get('error', status['event'])}")
            if status['event'] == 'done':
                return status
            node.get_logger().info(f"[snapshot] FFS worker: {status['event']}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FFS worker gave no result in {timeout:.0f} s; see {request['logs']}")
        rclpy.spin_once(node, timeout_sec=.1)


def take_snapshot(node, episode, label: str = "", seed: bool = True) -> dict:
    """Fresh FFS capture -> reconstruction -> report -> contacts -> episode stage.

    Call only with a stationary object and the arm clear, as for the old snapshot
    API. The retained seed argument is ignored: FFS always reacquires. The existing
    primitive/operator/MoveIt gates still own motion approval.

    Uses the warm FFS worker (stereo_offline_worker.py, started by bringup_stack) when it
    is running, else one fresh subprocess per step. Either way the same steps write the
    same files; snapshot.json records which ran (ffs_runner).
    """
    from irb120_perception.stereo_offline_bridge import WORKSPACE, VENV_PYTHON
    # The FFS run is written straight into the stage (capture/, recon/, contact selection
    # at the stage root; see irb120_perception.snapshot_layout), so nothing is copied.
    # The episode's first perceive also builds the review mesh, report and HTML viewers;
    # later ones skip them for speed.
    review = not any(s['kind'] == 'perceive' for s in episode.load()['stages'])
    stage = episode.begin_stage('perceive', backend='offline_ffs', label=label)
    run = stage.path
    capture, recon, logs = run/'capture', run/'recon', run/'logs'
    logs.mkdir()
    commands = [
        ('capture', 'capture_stereo_snapshot.py', [str(capture), '--stationary-arm-clear']),
        ('reconstruct', 'reconstruct_stereo_offline.py',
         [str(capture), '--output', str(recon)] + (['--mesh'] if review else [])),
        *([('report', 'report_stereo_offline.py', [str(recon)])] if review else []),
        ('contacts', 'select_stereo_contacts.py',
         [str(recon), '--output', str(run)] + (['--review-mesh'] if review else [])),
    ]
    try:
        started = time.monotonic()
        worker = _ffs_worker(node)
        if worker is not None:
            node.get_logger().info(f'[snapshot] FFS via warm worker: {run}')
            status = _run_ffs_worker(node, worker, dict(capture=str(capture), recon=str(recon),
                                                        contacts=str(run), logs=str(logs), report=review))
            runner = dict(runner='warm_worker', **{k: v for k, v in status.items() if k.endswith('_seconds')})
        else:
            node.get_logger().warn('[snapshot] FFS worker not running (bringup_stack ffs_worker:=true '
                                   'starts it); fresh subprocesses reload the models each step')
            for name, script, args in commands:
                node.get_logger().info(f'[snapshot] FFS {name}: {run}')
                _run_ffs_command(node, [str(VENV_PYTHON), str(WORKSPACE/'scripts'/script), *args],
                                 logs/f'{name}.log')
            runner = dict(runner='subprocess')
        runner['wall_seconds'] = time.monotonic()-started
        runner['review'] = review
        record = _import_ffs_snapshot(episode, run, review_status='pending_motion_gate', label=label,
                                      stage=stage, ffs_runner=runner)
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
    """Copy a reviewed standalone run (capture/, recon/, contacts/) into a new perceive
    stage; caller attests the scene is unchanged.

    This approval records review/scene continuity, not trajectory feasibility.
    No live services or robot commands are used. Reimporting a capture after
    intervening motion is rejected, even with a repeated approval flag.
    """
    if not reviewed_unchanged:
        raise ValueError("Review contacts and confirm the scene is unchanged with --reviewed-unchanged")
    return _import_ffs_snapshot(episode, run_dir, review_status='reviewed_unchanged')


def _import_ffs_snapshot(episode, run_dir, *, review_status, label='reviewed import', stage=None,
                         ffs_runner=None):
    run = Path(run_dir).expanduser().resolve()
    recon, capture = run/'recon', run/'capture'
    # A standalone FFS run keeps contact selection in contacts/; an episode stage at its root.
    contacts_dir = run/'contacts' if (run/'contacts'/'handoff.json').exists() else run
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
    from irb120_perception.object_cloud import load_object_cloud
    xyz = load_object_cloud(contacts_dir/'object_cloud.npz')['xyz']
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
        if stage.path.resolve() != run:
            # Reviewed import of a standalone run: bring it into the stage layout once.
            shutil.copytree(capture, stage.path/'capture')
            shutil.copytree(recon, stage.path/'recon')
            shutil.copytree(contacts_dir, stage.path, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('capture', 'recon', 'contacts',
                                                         'object_cloud.npz', 'object_cloud.json'))
            from irb120_perception.object_cloud import publish_object_cloud
            publish_object_cloud(stage.path/'recon', stage.path)
        from .tip_choice import choose_tip
        tip, reason = choose_tip(contacts)
        record = dict(ok=True, backend='offline_ffs', source_run=str(run), capture_id=capture_id,
                      reviewed_unchanged=review_status == 'reviewed_unchanged', review_status=review_status,
                      trajectory_checked=False,
                      tip_choice={'tip': tip, 'reason': reason},
                      available={m: contacts[m]['available'] for m in ('planar_push', 'forward_tip', 'press')})
        if ffs_runner is not None:
            record['ffs_runner'] = ffs_runner
        write_json_atomic(stage.path/'snapshot.json', record)
        from .support_cloud_recording import record_support_scene
        # Support-cloud rebuild is optional evidence, never a fabricated pass.
        try:
            record['support_scene'] = record_support_scene(stage.path)
        except (OSError, ValueError, KeyError) as exc:
            record['support_scene'] = {'status': 'unavailable', 'reason': str(exc)}
        write_json_atomic(stage.path/'snapshot.json', record)
        stage.end('ok', backend='offline_ffs', capture_id=capture_id, points=len(xyz),
                  **({'ffs_runner': ffs_runner['runner']} if ffs_runner else {}))
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
    # Older snapshots must be reacquired, rather than cleaned differently by each motion.
    from irb120_perception.object_cloud import load_object_cloud
    try:
        load_object_cloud(path.parent/'object_cloud.npz')
    except (OSError, ValueError, KeyError):
        return None
    with open(path) as f:
        return json.load(f)


def latest_object_cloud(episode) -> np.ndarray:
    """The object cloud (base_link) from the episode's most recent successful perceive stage."""
    stage = next(s for s in reversed(episode.load()["stages"]) if s["kind"] == "perceive" and s["status"] == "ok")
    from irb120_perception.object_cloud import load_object_cloud
    return load_object_cloud(episode.path / stage["dir"] / "object_cloud.npz")["xyz"]


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
