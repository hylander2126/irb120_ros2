"""Perceive-stage and episode lookup across the current and pre-2026-10-24 layouts."""
from irb120_perception.snapshot_layout import episode_jsons, snapshot_file


def test_snapshot_file_prefers_stage_root_then_recon_then_capture(tmp_path):
    (tmp_path/'recon').mkdir()
    (tmp_path/'capture').mkdir()
    (tmp_path/'recon/cam1_camera_info.json').write_text('{}')
    (tmp_path/'capture/capture.json').write_text('{}')
    assert snapshot_file(tmp_path, 'cam1_camera_info.json') == tmp_path/'recon/cam1_camera_info.json'
    assert snapshot_file(tmp_path, 'capture.json') == tmp_path/'capture/capture.json'
    (tmp_path/'cam1_camera_info.json').write_text('{}')  # old layout: copy at the root
    assert snapshot_file(tmp_path, 'cam1_camera_info.json') == tmp_path/'cam1_camera_info.json'
    assert snapshot_file(tmp_path, 'missing.npz') == tmp_path/'missing.npz'


def test_episode_jsons_finds_both_layouts(tmp_path):
    new = tmp_path/'drill/20261025_090000/episode.json'
    old = tmp_path/'drill/episodes/20261020_090000/episode.json'
    other = tmp_path/'heart_a/20261021_090000/episode.json'
    for p in (new, old, other):
        p.parent.mkdir(parents=True)
        p.write_text('{}')
    assert episode_jsons(tmp_path, 'drill') == [old, new]
    assert set(episode_jsons(tmp_path)) == {new, old, other}
    assert set(episode_jsons(tmp_path, recursive=True)) == {new, old, other}
