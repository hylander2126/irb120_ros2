"""Check contact geometry and failure cases without ROS or robot hardware."""

import numpy as np
import pytest

from irb120_perception.contact_point_selector import (
    estimate_pivot, select_contact_points,
)


def box():
    points, normals = [], []
    axes = [np.linspace(0.5, 0.6, 21), np.linspace(-0.05, 0.05, 21),
            np.linspace(0., 0.2, 41)]
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        grid = np.array(np.meshgrid(*[axes[i][1:-1] for i in others])).reshape(2, -1).T
        for end, sign in [(axes[axis][0], -1), (axes[axis][-1], 1)]:
            if axis == 2 and sign == -1:  # unseen underside
                continue
            face = np.zeros((len(grid), 3))
            face[:, axis] = end
            face[:, others] = grid
            normal = np.zeros_like(face)
            normal[:, axis] = sign
            points.append(face)
            normals.append(normal)
    return np.concatenate(points), np.concatenate(normals)


def test_box_contacts_and_signed_scores():
    points, normals = box()
    result = select_contact_points(points, normals=normals, table_z=0.)
    push, tip, press = [result[key] for key in ('planar_push', 'forward_tip', 'press')]
    assert push['available'] and tip['available'] and press['available']
    np.testing.assert_allclose(push['point'], [0.5, 0., 0.02])
    np.testing.assert_allclose(tip['point'], [0.5, 0., 0.195])
    # The horizontal pull moment ties across the flat top.  The press stage
    # therefore chooses the point nearest the -X support edge, minimizing the
    # opposing moment from the downward contact-establishing press.
    np.testing.assert_allclose(press['point'], [0.505, 0., 0.2], atol=1e-10)
    assert press['press_score'] == pytest.approx(-0.005)
    np.testing.assert_allclose(press['edge_midpoint'], [0.5, 0., 0.], atol=1e-10)
    assert press['along_edge_offset'] == pytest.approx(0.)
    for selected, pivot_x, direction_x in [(tip, 0.6, 1.), (press, 0.5, -1.)]:
        geometry = selected['geometry']
        assert geometry['kind'] == 'extended_edge'
        np.testing.assert_allclose(geometry['pivot'], [pivot_x, 0., 0.], atol=1e-10)
        np.testing.assert_allclose(geometry['direction'], [direction_x, 0., 0.])
        assert selected['score'] == pytest.approx(selected['point'][2])
        assert np.any(np.all(points == selected['point'], axis=1))
    assert push['ball_center'][2] - 0.01325 >= 0.003


def test_normals_estimated_from_cloud_and_fused_overlap():
    points, _ = box()
    a = select_contact_points(points, table_z=0.)
    b = select_contact_points([points[::-1], points[::2]], table_z=0.)
    for mode in ('planar_push', 'forward_tip', 'press'):
        assert a[mode]['available']
        np.testing.assert_allclose(a[mode]['point'], b[mode]['point'])
    assert a['planar_push']['normal'][0] < -0.9
    assert a['press']['normal'][2] > 0.9


def test_oblique_box_does_not_rotate_task_direction():
    points, _ = box()
    theta = np.deg2rad(25)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    points = (points - [0.55, 0., 0.]) @ rotation.T + [0.55, 0., 0.]
    geometry = estimate_pivot(points, [1., 0., 0.], 0.)
    assert geometry['kind'] == 'isolated_point'
    np.testing.assert_allclose(geometry['direction'], [1,0,0], atol=1e-10)
    np.testing.assert_allclose(geometry['axis'], [0,1,0], atol=1e-10)
    assert geometry['preferred_flat_alternative'] is None


def test_curved_support_uses_isolated_pivot():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = np.column_stack((0.55 + 0.05 * np.cos(theta),
                              0.05 * np.sin(theta), np.full(128, 0.005)))
    geometry = estimate_pivot(points, [1., 0., 0.], 0.)
    assert geometry['kind'] == 'isolated_point'
    # The robust extreme gives up less than 1 mm on a clean circular footprint
    # rather than allowing one raw-depth fringe sample to own the pivot.
    np.testing.assert_allclose(geometry['pivot'], [0.6, 0., 0.], atol=1e-3)


def test_isolated_pivot_rejects_single_directional_flying_point():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = np.column_stack((0.55 + 0.05 * np.cos(theta),
                              0.05 * np.sin(theta), np.full(128, 0.005)))
    points = np.vstack((points, [0.63, 0., 0.005]))
    geometry = estimate_pivot(points, [1., 0., 0.], 0.)
    assert geometry['kind'] == 'isolated_point'
    assert geometry['pivot'][0] < 0.602
    assert geometry['position_evidence']['extreme_trim_mm'] > 25.


def test_missing_support_does_not_use_overhang_as_base():
    points, normals = box()
    keep = points[:, 2] > 0.03
    result = select_contact_points(points[keep], normals=normals[keep], table_z=0.)
    assert result['planar_push']['available']
    for mode in ('forward_tip', 'press'):
        assert not result[mode]['available']
        assert 'support' in result[mode]['reason']


def test_height_floor_and_absent_centerline_contacts():
    points, normals = box()
    result = select_contact_points(points, normals=normals, table_z=0., min_ball_center_z=0.06)
    assert result['planar_push']['point'][2] >= 0.06
    points[:, 1] += 0.2
    result = select_contact_points(points, normals=normals, table_z=0.)
    assert not result['planar_push']['available']


def test_press_requires_observed_inset_top():
    points, normals = box()
    side = normals[:, 2] == 0
    result = select_contact_points(points[side], normals=normals[side], table_z=0.)
    assert not result['press']['available']
    assert result['forward_tip']['available']
    result = select_contact_points(points, normals=normals, table_z=0., press_inset=0.1)
    assert not result['press']['available']


def test_base_link_and_world_height_agree():
    points, normals = box()
    world = select_contact_points(points, normals=normals, table_z=0.)
    base = select_contact_points(points - [0., 0., 0.021], normals=normals)
    for mode in ('planar_push', 'forward_tip', 'press'):
        np.testing.assert_allclose(base[mode]['point'] + [0., 0., 0.021], world[mode]['point'])


def test_invalid_cloud_and_normal_handling():
    with pytest.raises(ValueError, match='six distinct'):
        select_contact_points(np.zeros((10, 3)))
    points, normals = box()
    with pytest.raises(ValueError, match='Normals must match'):
        select_contact_points(points, normals=normals[:5])
    result = select_contact_points(points, normals=np.zeros_like(normals), table_z=0.)
    assert all(not result[mode]['available'] for mode in ('planar_push', 'forward_tip', 'press'))


def test_press_insets_face_before_scoring_sloped_top():
    points, normals = box()
    top = normals[:, 2] > 0.7
    points[top, 2] += 0.8 * (points[top, 1] + 0.05)
    normals[top] = [0., -0.8, 1.]
    press = select_contact_points(points, normals=normals, table_z=0., press_inset=0.005)['press']
    assert press['available']
    assert press['point'][1] <= 0.04 + 1e-10
    assert press['candidate_counts']['inset'] > press['candidate_counts']['best_score_band']


def test_press_prefers_edgeward_contact_despite_flat_top_height_noise():
    points, normals = box()
    top = normals[:, 2] > 0.7
    # A 2 mm depth-noise ridge at the centre is still within the 3 mm pull-score
    # tolerance.  It must not displace the press point from the pivot-side edge.
    noisy_ridge = top & np.isclose(points[:, 0], 0.55)
    points[noisy_ridge, 2] += 0.002
    press = select_contact_points(points, normals=normals, table_z=0.)['press']
    assert press['available']
    assert press['point'][0] == pytest.approx(0.505)
    assert press['point'][1] == pytest.approx(0.)
    assert press['score'] == pytest.approx(0.2)
    assert press['press_score'] == pytest.approx(-0.005)
    assert press['along_edge_offset'] == pytest.approx(0.)


def test_press_double_scoring_favors_edge_over_noisy_interior_high_point():
    """A depth-noise bump *larger* than score_tolerance, sitting away from the
    pivot edge, must not steal the press point from the true near-edge
    contact. A height-only filter (the old sequential behaviour) would
    exclude every near-edge point once it falls more than score_tolerance
    below the bump, leaving only the bump itself as a candidate -- exactly
    the real-world failure this double-scoring heuristic fixes: the pull
    moment-arm term alone can't tell a noisy interior high point from a
    real one, but the pivot-proximity term decisively prefers the edge."""
    points, normals = box()
    top = normals[:, 2] > 0.7
    # 1 cm bump at the top's centre (X=0.55) -- more than 3x score_tolerance,
    # would fully exclude the X=0.505 edge points under height-only banding.
    noisy_bump = top & np.isclose(points[:, 0], 0.55)
    points[noisy_bump, 2] += 0.01
    press = select_contact_points(points, normals=normals, table_z=0.)['press']
    assert press['available']
    assert press['point'][0] == pytest.approx(0.505)
    assert press['point'][1] == pytest.approx(0.)
    assert press['score'] == pytest.approx(0.2)
    assert press['press_score'] == pytest.approx(-0.005)


def test_press_pivot_weight_zero_reduces_to_pull_moment_only():
    """press_pivot_weight=0 disables the pivot-proximity term entirely, so a
    tall-enough noisy interior bump *does* win -- confirms the double-score
    is actually doing the work in the test above, not some other filter."""
    points, normals = box()
    top = normals[:, 2] > 0.7
    noisy_bump = top & np.isclose(points[:, 0], 0.55)
    points[noisy_bump, 2] += 0.01
    press = select_contact_points(
        points, normals=normals, table_z=0., press_pivot_weight=0.)['press']
    assert press['available']
    assert press['point'][0] == pytest.approx(0.55)


def test_press_pivot_override_shifts_selected_point_and_spares_other_modes():
    """A validated RGB/SAM pivot must re-seed WHERE press centers along the
    edge (not just a downstream rotation center), while forward_tip/planar_push
    keep the cloud estimate untouched."""
    points, normals = box()
    baseline = select_contact_points(points, normals=normals, table_z=0.)
    assert baseline['press']['geometry']['pivot_source'] == 'cloud_estimate'

    overridden = select_contact_points(
        points, normals=normals, table_z=0., press_pivot_override=[0.5, 0.03, 0.])
    press = overridden['press']
    assert press['available']
    assert press['geometry']['pivot_source'] == 'rgb_sam_override'
    np.testing.assert_allclose(press['geometry']['pivot'], [0.5, 0.03, 0.])
    # Centered along the overridden edge, not the baseline's y=0.
    np.testing.assert_allclose(press['point'], [0.505, 0.03, 0.2], atol=1e-10)
    assert press['point'][1] != baseline['press']['point'][1]

    tip = overridden['forward_tip']
    assert tip['geometry']['pivot_source'] == 'cloud_estimate'
    np.testing.assert_allclose(tip['geometry']['pivot'], baseline['forward_tip']['geometry']['pivot'])
    np.testing.assert_allclose(tip['point'], baseline['forward_tip']['point'])


def test_press_pivot_override_rejects_bad_input():
    points, normals = box()
    with pytest.raises(ValueError, match='press_pivot_override'):
        select_contact_points(points, normals=normals, table_z=0., press_pivot_override=[0.5, 0.])
    with pytest.raises(ValueError, match='press_pivot_override'):
        select_contact_points(points, normals=normals, table_z=0., press_pivot_override=[0.5, np.nan, 0.])


def test_press_failure_names_rejecting_filter():
    points, normals = box()
    side = normals[:, 2] == 0
    press = select_contact_points(points[side], normals=normals[side], table_z=0.)['press']
    assert press['reason'] == 'No candidates after surface filter'
    assert press['candidate_counts']['surface'] == 0
    press = select_contact_points(points, normals=normals, table_z=0., press_inset=0.1)['press']
    assert press['reason'] == 'No candidates after inset filter'


def test_press_relaxes_pull_tolerance_for_rounded_top():
    points, normals = box()
    top = normals[:, 2] > 0.7
    normals[top] = [-0.43, 0., 0.9]  # dome tilted toward the -x pull direction
    strict = select_contact_points(points, normals=normals, table_z=0., pull_normal_epsilon=0.1)['press']
    assert strict['available'] and strict['relaxed_pull_epsilon'] == 0.5
    assert select_contact_points(points, normals=normals, table_z=0., pull_normal_epsilon=0.5)['press']['available']


def test_push_rejects_oblique_normals():
    points, normals = box()
    normals[normals[:, 0] < -0.9] = [-0.8078, 0.5893, 0.0127]
    push = select_contact_points(points, normals=normals, table_z=0.)['planar_push']
    assert not push['available']
    assert '20 deg' in push['reason']


def test_push_prefers_alignment_in_low_height_band():
    points, normals = box()
    side = normals[:, 0] < -0.9
    normals[side] = [-np.cos(np.radians(15)), np.sin(np.radians(15)), 0]
    aligned = np.flatnonzero(side & (np.abs(points[:, 1]) < 1e-9)
                             & (np.abs(points[:, 2] - .025) < 1e-9))[0]
    normals[aligned] = [-1, 0, 0]
    push = select_contact_points(points, normals=normals, table_z=0.,
                                 score_tolerance=.006)['planar_push']
    np.testing.assert_allclose(push['point'], points[aligned])
    assert push['normal_angle_deg'] == pytest.approx(0)


def test_curved_pivot_tangent_is_centered_and_normal_to_pull():
    theta = np.linspace(0, 2*np.pi, 128, endpoint=False)
    points = np.column_stack((.55+.05*np.cos(theta), .03*np.sin(theta), np.full(128,.005)))
    g = estimate_pivot(points, [1.,0.,0.], 0.)
    np.testing.assert_allclose(g['tangent'].mean(axis=0), g['pivot'])
    assert abs(np.dot(g['tangent'][1]-g['tangent'][0],g['direction'])) < 1e-12
    assert g['edge'] is None
    assert len(g['support_boundary']) == 128


def test_experimental_filter_precedes_scoring_and_cannot_fallback():
    points, normals = box()
    seen = []
    def keep_inboard(candidates, direction):
        seen.append(len(candidates))
        return candidates[:, 0] > .55
    result = select_contact_points(points, normals=normals, table_z=0.,
                                   experimental_press_filter=keep_inboard)
    assert result['press']['available']
    assert result['press']['point'][0] > .55
    assert seen[0] > 1
    rejected = select_contact_points(points, normals=normals, table_z=0.,
        experimental_press_filter=lambda p,d: np.zeros(len(p),dtype=bool))
    assert not rejected['press']['available']
    assert 'camera_agreement' in rejected['press']['reason']
