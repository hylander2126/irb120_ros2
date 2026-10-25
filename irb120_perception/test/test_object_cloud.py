"""The perception boundary publishes one supported cloud without changing observations."""
import json

import numpy as np
import pytest

from irb120_perception.object_cloud import (
    load_object_cloud, prepare_object_cloud, publish_object_cloud,
)


def test_sampling_filter_and_camera_provenance():
    raw = np.array([[.6, 0., .1], [.6001, 0., .1], [.606, 0., .1],
                    [.6, 0., .1], [.179, -.07, .135]])
    cameras = np.array([1, 1, 1, 2, 3])
    cloud, audit = prepare_object_cloud(raw, cameras)
    ids = cloud['source_index']
    np.testing.assert_array_equal(cloud['xyz'], raw[ids])
    np.testing.assert_array_equal(cloud['camera_id'], cameras[ids])
    assert set(ids) == {0, 2, 3}  # actual samples; same voxel in cam2 retains provenance
    assert audit['removed_points'] == 1
    assert audit['removed_source_index'] == [4]
    assert audit['removed_camera_id'] == [3]


def test_supported_disconnected_thin_features_are_preserved():
    raw = np.array([[.3, 0., .02], [.306, 0., .02],
                    [.65, 0., .2], [.656, 0., .2]])
    cloud, audit = prepare_object_cloud(raw, np.ones(len(raw), int))
    assert len(cloud['xyz']) == len(raw) and audit['removed_points'] == 0


@pytest.mark.parametrize('raw', [[], [[.5, 0., .1]],
                               [[.5, 0., .1], [.7, 0., .1]], [[np.nan, 0., 0.]]])
def test_no_supported_surface_fails_closed(raw):
    with pytest.raises(ValueError):
        prepare_object_cloud(raw, np.ones(len(raw), int))


def test_handoff_uses_same_file_and_load_does_not_reclean(tmp_path):
    recon, handoff = tmp_path/'recon', tmp_path/'contacts'
    recon.mkdir(); handoff.mkdir()
    cloud, audit = prepare_object_cloud([[.6, 0., .1], [.606, 0., .1]], [1, 1])
    np.savez_compressed(recon/'object_cloud.npz', **cloud)
    (recon/'object_cloud.json').write_text(json.dumps(audit))
    published = publish_object_cloud(recon, handoff)
    assert (recon/'object_cloud.npz').samefile(handoff/'object_cloud.npz')
    assert (recon/'object_cloud.json').samefile(handoff/'object_cloud.json')
    np.testing.assert_array_equal(published['xyz'], cloud['xyz'])
    np.testing.assert_array_equal(load_object_cloud(handoff/'object_cloud.npz')['xyz'], cloud['xyz'])


def test_legacy_cloud_requires_fresh_perception(tmp_path):
    np.savez(tmp_path/'object_cloud.npz', xyz=[[.6, 0., .1]])
    with pytest.raises(ValueError, match='fresh snapshot'):
        load_object_cloud(tmp_path/'object_cloud.npz')
