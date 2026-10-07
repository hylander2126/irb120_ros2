"""Physical score equivalence and offline evidence interpretation."""
import numpy as np

from irb120_perception.contact_validation import (
    approach_evidence, changed_cameras, cloud_trials, torque,
)


def test_signed_torque_matches_old_terms_for_rotated_pivot():
    direction = np.array([-.8, .6, 0.])
    geometry = dict(direction=direction, axis=np.cross([0, 0, 1], direction),
                    pivot=np.array([.3, -.1, -.02]))
    points = np.array([[.32, .01, .18], [.34, .02, .17]])
    r = points-geometry['pivot']
    for ratio in (0, .25, 1, 2):
        old = (np.cross(r, direction) @ geometry['axis']
               + ratio * (np.cross(r, [0, 0, -1]) @ geometry['axis']))
        np.testing.assert_allclose(torque(points, geometry, ratio), old)


def test_torque_penalizes_inboard_downward_press():
    geometry = dict(direction=np.array([-1., 0, 0]), axis=np.array([0., -1, 0]),
                    pivot=np.zeros(3))
    np.testing.assert_allclose(torque([[.02, 0, .1], [.04, 0, .1]], geometry), [.08, .06])


def test_local_support_flags_ball_center_beyond_observed_patch():
    x, y = np.meshgrid(np.linspace(-.005, .005, 11), np.linspace(-.005, .005, 11))
    points = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    flat = approach_evidence(points, np.zeros(3), np.array([0., 0, 1]))
    tilted = approach_evidence(points, np.zeros(3), np.array([.6, 0, .8]))
    assert flat['center_inside_local_xy_hull']
    assert not tilted['center_inside_local_xy_hull']
    assert tilted['local_xy_hull_margin_mm'] < 0


def test_missing_patch_evidence_does_not_claim_support():
    points = np.array([[0., 0, 0], [.001, 0, 0]])
    evidence = approach_evidence(points, points[0], np.array([0., 0, 1]))
    assert evidence['local_xy_hull_margin_mm'] is None
    assert not evidence['center_inside_local_xy_hull']


def test_depth_bias_preserves_invalid_pixels_and_original_inputs():
    depth = np.array([[0, 1, 500, 65535]], dtype=np.uint16)
    cameras = {'cam2': (depth, {'base_to_optical': {'R': np.eye(3).tolist(), 't': [0., 0, 0]}})}
    altered = changed_cameras(cameras, 'depth_-4')['cam2'][0]
    np.testing.assert_array_equal(altered, [[0, 1, 496, 65531]])
    np.testing.assert_array_equal(depth, [[0, 1, 500, 65535]])
    translated = changed_cameras(cameras, 'cam2_tx_2')
    assert translated['cam2'][1]['base_to_optical']['t'][0] == .002
    assert cameras['cam2'][1]['base_to_optical']['t'][0] == 0


def test_cloud_trials_are_seeded_and_recompute_geometry():
    from test_contact_point_selector import box
    points, _ = box()
    points[:, 2] -= .021  # Match the saved-scene table plane used by the harness.
    first = cloud_trials(points, trials=1, seed=42)
    second = cloud_trials(points, trials=1, seed=42)
    assert len(first['records']) == 18
    for a, b in zip(first['records'], second['records']):
        assert a['available'] == b['available']
        if a['available']:
            np.testing.assert_array_equal(a['point'], b['point'])
    available = [r for r in first['records'] if 'pivot_shift_mm' in r]
    assert available
    assert any(r['pivot_shift_mm'] > 0 for r in available)
