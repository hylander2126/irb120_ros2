import numpy as np
import pytest

from irb120_perception.pivot_boundary_review import intersect_table, lower_boundary


def camera():
    # Downward-looking camera, with roll: image horizontal != world horizontal.
    angle = .4
    R = np.array([[np.cos(angle), np.sin(angle), 0],
                  [np.sin(angle), -np.cos(angle), 0], [0, 0, -1.]])
    return dict(K=[[100., 0, 50], [0, 100., 50], [0, 0, 1]], D=[],
                base_to_optical=dict(R=R, t=[0, 0, 1.]))


def test_ray_intersection_tilt_and_roll():
    info = camera()
    pixels = np.array([[30., 70.], [70., 70.]])
    points, valid = intersect_table(pixels, info, [.1, -.05, .02])
    assert valid.all()
    np.testing.assert_allclose(points[:, 2], .1*points[:, 0]-.05*points[:, 1]+.02)
    tf = info['base_to_optical']
    cam = points @ tf['R'].T + tf['t']
    projected = cam @ np.asarray(info['K']).T
    np.testing.assert_allclose(projected[:, :2]/projected[:, 2:], pixels)
    assert abs(points[1, 1]-points[0, 1]) > .1


def test_component_association_ignores_larger_disconnected_mask():
    info = camera()
    mask = np.zeros((100, 100), np.uint8)
    mask[10:40, 10:40] = 255
    mask[60:70, 60:70] = 255
    cloud, _ = intersect_table([[62, 62], [64, 64], [66, 66]], info, [0, 0, 0])
    pixels, votes = lower_boundary(mask, cloud, info)
    assert votes == 3
    assert (pixels[:, 1] == 69).all()
    assert (pixels[:, 0] >= 60).all()


def test_parallel_and_behind_camera_rays_rejected():
    info = camera()
    _, valid = intersect_table([[50, 50]], info, [0, 0, 2])
    assert not valid.any()


def test_missing_mask_retains_baseline_and_contacts(tmp_path):
    import json
    from irb120_perception.pivot_boundary_review import evaluate
    contacts = {'press': {'geometry': {'pivot': [.6, 0, -.021]}}}
    path = tmp_path/'contacts.json'
    original = json.dumps(contacts)
    path.write_text(original)
    result = evaluate(tmp_path)
    assert result['physics']['selected_pivot_base_m'] == [.6, 0, -.021]
    assert result['physics']['confidence'] == 0
    assert 'missing saved SAM mask' in result['fallback_reason']
    assert path.read_text() == original


def test_flat_and_rounded_support_with_outliers():
    from irb120_perception.pivot_boundary_review import fit_support
    y = np.linspace(-.03, .03, 81)
    plane = [.02, -.01, -.021]
    for rounded in (False, True):
        x = .6 + (12*y*y if rounded else .05*y)
        z = plane[0]*x+plane[1]*y+plane[2]
        points = np.c_[x, y, z]
        points[::15, 0] += .025
        result = fit_support(points, plane)
        assert result['support_type'] == ('rounded_tangent' if rounded else 'flat_edge')
        np.testing.assert_allclose(result['pivot'][0], .6, atol=.001)
        axis = np.asarray(result['axis'])
        assert abs(axis @ np.array([-plane[0], -plane[1], 1])) < 1e-10


def test_depth_checks_reject_missing_depth_and_table_bleed():
    from irb120_perception.pivot_boundary_review import boundary_depth_evidence
    info = camera()
    pixels = np.array([[50, 50], [60, 50], [70, 50]])
    depth = np.full((100, 100), 1000, np.uint16)
    # Camera points down from 1 m: inside object is 10 mm above table.
    depth[45:48, 50] = 990
    depth[45:48, 60] = 0
    # Third candidate is table on both sides, as for a shadow or mask bleed.
    ok, evidence = boundary_depth_evidence(pixels, depth, info, [0, 0, 0])
    assert ok.tolist() == [True, False, False]
    assert evidence['valid_neighborhood'] == [True, False, True]


def test_support_rejects_short_span():
    import pytest
    from irb120_perception.pivot_boundary_review import fit_support
    with pytest.raises(ValueError, match='span'):
        fit_support(np.c_[np.full(30, .6), np.linspace(0, .005, 30), np.zeros(30)], [0, 0, 0])


def test_forward_fit_mirrors_the_press_fit():
    from irb120_perception.pivot_boundary_review import fit_support
    y = np.linspace(-.03, .03, 81)
    plane = [.02, -.01, -.021]
    for rounded in (False, True):
        # Far (+X) edge: the support bulges toward +X.
        x = .7 - (12*y*y if rounded else -.05*y)
        z = plane[0]*x+plane[1]*y+plane[2]
        points = np.c_[x, y, z]
        points[::15, 0] -= .025  # inward outliers
        result = fit_support(points, plane, outward=1.)
        assert result['support_type'] == ('rounded_tangent' if rounded else 'flat_edge')
        np.testing.assert_allclose(result['pivot'][0], .7, atol=.001)
        np.testing.assert_allclose(result['pivot'][2], plane[0]*result['pivot'][0]+plane[1]*result['pivot'][1]+plane[2])
        axis = np.asarray(result['axis'])
        assert abs(axis @ np.array([-plane[0], -plane[1], 1])) < 1e-10
        if not rounded:
            assert axis[0]/axis[1] == pytest.approx(.05, abs=1e-3)


def test_forward_pivot_shape_rules():
    from irb120_perception.forward_pivot_review import pivot_shape
    base = dict(support_type='flat_edge', line_rms_mm=1., quadratic_rms_mm=.8)
    assert pivot_shape(dict(base, line_span_mm=55, band_span_mm=58))[0] == 'straight'
    # A short chord of a curved base fits a line well but is a point-like contact.
    assert pivot_shape(dict(base, line_span_mm=25, band_span_mm=25))[0] == 'curved'
    # A line through only part of the support band.
    assert pivot_shape(dict(base, line_span_mm=45, band_span_mm=80))[0] == 'curved'
    assert pivot_shape(dict(base, support_type='rounded_tangent', line_span_mm=55, band_span_mm=55))[0] == 'curved'


def forward_contacts():
    return {'forward_tip': {'available': True, 'point': [.68, 0, .1], 'ball_center': [.667, 0, .1],
                            'geometry': {'pivot': [.70, 0, -.021], 'axis': [0, 1, 0], 'direction': [1, 0, 0]}}}


def forward_report(status='reviewable', pivot=(.695, .003, -.02), shape='straight'):
    return {'candidate_status': status, 'shape': shape, 'shape_reason': 'test', 'source': 'test',
            'fallback_reason': 'x', 'geometry': {'candidate_pivot_base_m': list(pivot)},
            'evidence': {'support_fit': {'support_type': 'flat_edge'}}}


def test_forward_pivot_applied_with_shape_and_original_untouched():
    from irb120_perception.forward_pivot_review import contacts_with_forward_pivot
    original = forward_contacts()
    result, applied, reason = contacts_with_forward_pivot(original, forward_report())
    assert applied and reason is None
    forward = result['forward_tip']
    assert forward['geometry']['pivot'] == [.695, .003, -.02]
    assert forward['geometry']['axis'] == [0, 1, 0] and forward['point'] == [.68, 0, .1]
    assert forward['pivot_shape']['shape'] == 'straight'
    assert forward['rgb_pivot']['replaced_cloud_pivot_base_m'] == [.70, 0, -.021]
    assert original['forward_tip']['geometry']['pivot'] == [.70, 0, -.021]


def test_forward_pivot_not_applied_but_shape_recorded():
    from irb120_perception.forward_pivot_review import contacts_with_forward_pivot
    for report, apply, why in [(forward_report(), False, 'shadow'),
                               (forward_report(status='rejected', shape='unknown'), True, 'not reviewable'),
                               (forward_report(pivot=(.66, 0, -.02)), True, 'not beyond')]:
        result, applied, reason = contacts_with_forward_pivot(forward_contacts(), report, apply=apply)
        assert not applied and why in reason
        assert result['forward_tip']['geometry']['pivot'] == [.70, 0, -.021]
        assert result['forward_tip']['pivot_shape']['shape'] == report['shape']


def test_forward_review_without_mask_is_unknown(tmp_path):
    import json
    from irb120_perception.forward_pivot_review import evaluate
    (tmp_path/'contacts.json').write_text(json.dumps(forward_contacts()))
    report = evaluate(tmp_path)
    assert report['shape'] == 'unknown' and 'missing saved cam3 SAM mask' in report['fallback_reason']
    assert (tmp_path/'forward_pivot_review.json').exists()
