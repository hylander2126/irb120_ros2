"""Fresh FFS acquisition and frozen handoff gates without a ROS graph/robot."""
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from irb120_control.util.episode import Episode
from irb120_control.util import perception_snapshot as module


def episode_at(path):
    path.mkdir()
    (path/'episode.json').write_text(json.dumps(dict(stages=[])))
    return Episode(path)


def make_run(run):
    for name in ('capture', 'recon', 'contacts'):
        (run/name).mkdir(parents=True)
    capture, recon, contacts = (run/name for name in ('capture', 'recon', 'contacts'))
    (capture/'capture.json').write_text(json.dumps(dict(base_frame='base_link',
        stationary_arm_clear=True, cameras={c: {} for c in ('cam1', 'cam2', 'cam3')})))
    for cam in ('cam1', 'cam2', 'cam3'):
        (capture/f'{cam}.npz').write_bytes(cam.encode())
        (recon/f'{cam}_camera_info.json').write_text(json.dumps(dict(K=np.eye(3).tolist(),
            base_to_optical=dict(R=np.eye(3).tolist(), t=[0., 0., 0.]))))
        cv2.imwrite(str(recon/f'{cam}_color.png'), np.zeros((4, 4, 3), np.uint8))
        np.savez(recon/f'{cam}_aligned_depth.npz', depth_mm=np.full((4, 4), 100, np.uint16))
    (contacts/'handoff.json').write_text(json.dumps(dict(frame='base_link', source=str(recon.resolve()))))
    (contacts/'contacts.json').write_text(json.dumps({mode: {'available': False}
        for mode in ('press', 'forward_tip', 'planar_push')}))
    np.savez(contacts/'object_cloud.npz', xyz=np.array([[.5, 0., .1], [.51, .01, .1], [.5, .02, .12]]))
    return run


def test_import_requires_review_and_source_match(tmp_path):
    episode = episode_at(tmp_path/'episode')
    run = make_run(tmp_path/'run')
    with pytest.raises(ValueError, match='Review contacts'):
        module.import_ffs_snapshot(episode, run)
    handoff = run/'contacts/handoff.json'
    handoff.write_text(json.dumps(dict(frame='base_link', source='/wrong/recon')))
    with pytest.raises(ValueError, match='not derived'):
        module.import_ffs_snapshot(episode, run, reviewed_unchanged=True)
    assert episode.load()['stages'] == []


def test_motion_invalidates_contacts_and_blocks_capture_reuse(tmp_path):
    episode = episode_at(tmp_path/'episode')
    run = make_run(tmp_path/'run')
    record = module.import_ffs_snapshot(episode, run, reviewed_unchanged=True)
    assert record['reviewed_unchanged'] and not record['trajectory_checked']
    assert module.latest_contacts(episode) is not None
    original = (episode.path/'00_perceive/contacts.json').read_bytes()
    (run/'contacts/contacts.json').write_text('{}')
    assert (episode.path/'00_perceive/contacts.json').read_bytes() == original
    (run/'contacts/contacts.json').write_bytes(original)
    motion = episode.begin_stage('forward_arc')
    motion.end('failed')  # failed motion can still have moved the object
    assert module.latest_contacts(episode) is None
    with pytest.raises(ValueError, match='predates motion'):
        module.import_ffs_snapshot(episode, run, reviewed_unchanged=True)


def test_failed_snapshot_does_not_fall_back_to_earlier_contacts(tmp_path, monkeypatch):
    episode = episode_at(tmp_path/'episode')
    module.import_ffs_snapshot(episode, make_run(tmp_path/'run'), reviewed_unchanged=True)
    monkeypatch.setattr(module, '_run_ffs_command', Mock(side_effect=RuntimeError('capture failed')))
    result = module.take_snapshot(Mock(), episode)
    assert not result['ok']
    assert episode.load()['stages'][-1]['status'] == 'failed'
    assert module.latest_contacts(episode) is None


def test_automatic_capture_runs_all_four_fresh_steps(tmp_path, monkeypatch):
    episode = episode_at(tmp_path/'episode')
    calls = []

    def run_step(node, command, log):
        calls.append(Path(command[1]).name)
        run = log.parent
        if calls[-1] == 'capture_stereo_snapshot.py':
            make_run(run)
        log.write_text('fake step succeeded')

    monkeypatch.setattr(module, '_run_ffs_command', run_step)
    first = module.take_snapshot(Mock(), episode, label='initial')
    second = module.take_snapshot(Mock(), episode, label='retake')
    assert first['ok'] and second['ok']
    assert first['source_run'] != second['source_run']
    assert first['review_status'] == 'pending_motion_gate'
    assert not first['reviewed_unchanged']
    assert calls == ['capture_stereo_snapshot.py', 'reconstruct_stereo_offline.py',
                     'report_stereo_offline.py', 'select_stereo_contacts.py']*2


def test_interrupted_capture_terminates_its_process_group(tmp_path, monkeypatch):
    process = Mock(pid=12345)
    process.poll.return_value = None
    monkeypatch.setattr(module.subprocess, 'Popen', Mock(return_value=process))
    terminate = Mock()
    monkeypatch.setattr(module.os, 'killpg', terminate)
    monkeypatch.setattr(module.rclpy, 'spin_once', Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        module._run_ffs_command(Mock(), ['fake-perception'], tmp_path/'capture.log')
    terminate.assert_called_once_with(12345, module.signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=5.)


def test_missing_view_evidence_is_not_a_complete_snapshot(tmp_path):
    episode = episode_at(tmp_path/'episode')
    run = make_run(tmp_path/'run')
    (run/'recon/cam3_aligned_depth.npz').unlink()
    with pytest.raises(ValueError, match='Missing calibrated view'):
        module.import_ffs_snapshot(episode, run, reviewed_unchanged=True)
    assert episode.load()['stages'] == []


def test_contact_markers_accept_json_arrays_from_ffs():
    from builtin_interfaces.msg import Time
    from irb120_perception.contact_point_selector import build_contact_markers
    selected = dict(available=True, point=[.5, 0., .1], direction=[1., 0., 0.],
                    geometry=dict(pivot=[.5, 0., -.021], axis=[0., 1., 0.],
                                  direction=[1., 0., 0.], edge=None))
    contacts = {m: selected for m in ('press', 'forward_tip', 'planar_push')}
    markers = build_contact_markers(contacts, 'base_link', Time()).markers
    assert len(markers) == 15
    assert all(m.header.frame_id == 'base_link' for m in markers)


def test_legacy_snapshot_cannot_become_a_new_motion_target(tmp_path, monkeypatch):
    episode = episode_at(tmp_path/'episode')
    stage = episode.begin_stage('perceive')  # historical DBSCAN stage lacks FFS backend identity
    (stage.path/'contacts.json').write_text('{}')
    stage.end('ok')
    assert module.latest_contacts(episode) is None
    fresh = Mock(return_value={'ok': False})
    monkeypatch.setattr(module, 'take_snapshot', fresh)
    assert module.begin_motion_stage(Mock(), episode, 'push') == (None, None)
    fresh.assert_called_once()
    assert len(episode.load()['stages']) == 1
