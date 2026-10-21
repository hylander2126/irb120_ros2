"""The frame check's placement and transforms (numpy only; no VTK window)."""
import json

import numpy as np
import pytest

from irb120_models import frame_view


def box_cloud(near=.60, length=.08, width=.05, height=.10, table_z=-.021):
    g = np.mgrid[0:1:21j, 0:1:11j, 0:1:26j].reshape(3, -1).T
    return np.array([near, -width / 2, table_z + .014]) + g * [length, width, height]


def test_frame_round_trip_shift_and_yaw():
    f = frame_view.Frame([1., 0., 0.], [.6, 0., -.021], shift_mm=(10., -5., 0.), yaw_deg=90.)
    np.testing.assert_allclose(f.R[:, 0], [0., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(f.origin, [.605, .01, -.021], atol=1e-12)
    pts = np.random.default_rng(0).normal(size=(5, 3))
    np.testing.assert_allclose(f.to_base(f.to_frame(pts)), pts, atol=1e-12)
    np.testing.assert_allclose(f.matrix()[:3, :3] @ [1, 0, 0] + f.matrix()[:3, 3], f.to_base([1., 0., 0.])[0])


def test_cloud_origin_is_near_footprint_edge_on_mid_line():
    snap = dict(cloud=box_cloud(), direction=np.array([1., 0., 0.]), table_z=-.021, pivots={})
    origin, how = frame_view.auto_origin(snap, 'press')  # no press pivot: falls back to the cloud
    assert 'cloud' in how
    np.testing.assert_allclose(origin, [.6, 0., -.021], atol=2e-3)
    f = frame_view.Frame(snap['direction'], origin)
    fp = frame_view.footprint(snap['cloud'], f.R, snap['table_z'])
    assert fp['far'] - fp['near'] == pytest.approx(.08, abs=4e-3)


def test_press_pivot_sets_origin_x_and_push_direction_is_respected():
    snap = dict(cloud=box_cloud(), direction=np.array([0., -1., 0.]), table_z=-.021,
                pivots=dict(press=dict(point=np.array([.62, .03, -.021]), axis=np.array([1., 0., 0.]), source='t')))
    origin, how = frame_view.auto_origin(snap)
    assert how == 'press-pull pivot'
    assert origin @ np.array([0., -1., 0.]) == pytest.approx(-.03)
    assert origin[2] == pytest.approx(-.021)


def test_record_com_and_scene_table():
    assert frame_view.record_com_mm(None) == [None, None, None]
    assert frame_view.record_com_mm(dict(com_obj=[.03, .1])) == pytest.approx([30., 0., 100.])
    rng = np.random.default_rng(1)
    table = np.column_stack([rng.uniform(size=(500, 2)), -.021 + .004 + rng.normal(0, 3e-4, 500)])
    assert frame_view.scene_table_z(table, -.021) == pytest.approx(-.017, abs=1e-3)


def test_resolve_snapshot_from_episode(tmp_path):
    p = tmp_path / 'episodes' / 'x' / '00_perceive'
    p.mkdir(parents=True)
    np.savez(p / 'object_cloud.npz', xyz=box_cloud())
    (p.parent / 'episode.json').write_text(json.dumps(dict(object='mug')))
    folder, episode = frame_view.resolve_snapshot(p.parent)
    assert folder == p and episode['object'] == 'mug'
