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
    np.testing.assert_allclose(press['point'], [0.55, 0., 0.2], atol=1e-10)
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


def test_rotated_box_axis_is_inferred():
    points, _ = box()
    theta = np.deg2rad(25)
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    points = (points - [0.55, 0., 0.]) @ rotation.T + [0.55, 0., 0.]
    geometry = estimate_pivot(points, [1., 0., 0.], 0.)
    assert geometry['kind'] == 'extended_edge'
    np.testing.assert_allclose(geometry['direction'], rotation[:, 0], atol=1e-10)
    np.testing.assert_allclose(geometry['axis'], rotation[:, 1], atol=1e-10)


def test_curved_support_uses_isolated_pivot():
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    points = np.column_stack((0.55 + 0.05 * np.cos(theta),
                              0.05 * np.sin(theta), np.full(128, 0.005)))
    geometry = estimate_pivot(points, [1., 0., 0.], 0.)
    assert geometry['kind'] == 'isolated_point'
    np.testing.assert_allclose(geometry['pivot'], [0.6, 0., 0.], atol=1e-10)


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
