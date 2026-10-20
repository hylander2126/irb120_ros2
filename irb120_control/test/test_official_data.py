"""Official-data mode, log root override, camera agreement check, video frame stamps."""
import json

import numpy as np
import pytest

from irb120_control.util import official_run
from irb120_control.util.registration_check import TABLE_Z_BASE_LINK, check_folder
from irb120_control.util.runtime_log_dir import LOG_ROOT_ENV, log_root, runtime_log_dir
from irb120_control.util.video_stamps import frame_stamps_path, load_frame_stamps

CLEAN = {"git_commit": "abc", "git_dirty": False}


def test_log_root_follows_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path / "raw"))
    assert log_root() == tmp_path / "raw"
    assert runtime_log_dir("obj/episodes").is_dir()
    absolute = tmp_path / "elsewhere"
    assert runtime_log_dir(str(absolute)) == absolute       # stage paths are absolute


def test_official_refuses_without_protocol_or_clean_tree(tmp_path, monkeypatch):
    monkeypatch.delenv(LOG_ROOT_ENV, raising=False)
    with pytest.raises(official_run.OfficialRunError, match="PROTOCOL.md"):
        official_run.prepare(str(tmp_path), git_info=CLEAN)
    (tmp_path / "PROTOCOL.md").write_text("protocol")
    (tmp_path / "MANIFEST.json").write_text("{}")
    with pytest.raises(official_run.OfficialRunError, match="uncommitted"):
        official_run.prepare(str(tmp_path), git_info={"git_commit": "abc", "git_dirty": True})


def test_official_sets_log_root_and_records_hashes(tmp_path, monkeypatch):
    monkeypatch.delenv(LOG_ROOT_ENV, raising=False)
    (tmp_path / "PROTOCOL.md").write_text("protocol")
    (tmp_path / "MANIFEST.json").write_text("{}")
    info = official_run.prepare(str(tmp_path), git_info=CLEAN)
    assert log_root() == tmp_path / "raw"
    assert info["protocol_sha256"] and info["manifest_sha256"] and info["ft_calibration_sha256"]
    json.dumps(info)


def _scene(folder, heights_mm):
    rng = np.random.default_rng(0)
    obj = np.c_[rng.normal(0.65, 0.01, (500, 2)), rng.uniform(0, 0.1, 500)]
    xyz, cam = [], []
    for c, dz in enumerate(heights_mm, start=1):
        a = rng.uniform(0, 2 * np.pi, 4000); r = rng.uniform(0.06, 0.13, 4000)
        z = TABLE_Z_BASE_LINK + dz / 1000 + rng.normal(0, 0.0005, 4000)
        xyz.append(np.c_[0.65 + r * np.cos(a), 0.65 + r * np.sin(a), z]); cam.append(np.full(4000, c))
    obj[:, 1] = rng.normal(0.65, 0.01, 500)
    np.savez(folder / "support_scene_cloud.npz", xyz=np.vstack(xyz), camera_id=np.concatenate(cam).astype(np.uint8))
    np.savez(folder / "object_cloud.npz", xyz=obj)


def test_camera_agreement_pass_and_fail(tmp_path):
    good, bad = tmp_path / "good", tmp_path / "bad"
    good.mkdir(); bad.mkdir()
    _scene(good, [0.0, 0.5, -0.4])
    _scene(bad, [0.0, 8.0, 3.0])
    assert check_folder(good)["status"] == "pass"
    r = check_folder(bad)
    assert r["status"] == "fail" and abs(r["spread_m"] - 0.008) < 0.0007
    assert check_folder(tmp_path)["status"] == "unavailable"


def test_frame_stamps_round_trip(tmp_path):
    video = tmp_path / "camera_hull_overlay_cam2_20261020_120000.mp4"
    assert load_frame_stamps(video) is None
    frame_stamps_path(video).write_text("frame,stamp_s,receive_s\n0,10.0,10.01\n1,10.033,10.04\n")
    idx, t = load_frame_stamps(video)
    assert idx.tolist() == [0, 1] and np.allclose(t, [10.0, 10.033])


def test_pipeline_camera_gate_only_stops_official_episodes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from irb120_control import orchestrator
    from irb120_control.util.episode import Episode

    monkeypatch.setenv(LOG_ROOT_ENV, str(tmp_path))
    episode = Episode.create("gate_test", note="test", allow_unconfigured_object=True)
    stage = episode.begin_stage("perceive")
    _scene(stage.path, [0.0, 8.0, 3.0])
    stage.end("ok")
    dev = SimpleNamespace(official_info=None, auto=True)
    official = SimpleNamespace(official_info={"data_root": str(tmp_path)}, auto=True)
    assert orchestrator.registration_ok(episode, dev)
    assert not orchestrator.registration_ok(episode, official)
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert orchestrator.registration_ok(episode, SimpleNamespace(official_info={"x": 1}, auto=False))
    assert any("camera agreement: FAIL" in e["text"] for e in episode.load()["events"])
