"""Temporary camera agreement experiment checks; no ROS required."""
import numpy as np
import pytest
from irb120_perception.experimental_view_check import check_view, decide, evaluate


def test_visible_plane_and_occluded_view():
    info = dict(K=[[500,0,50],[0,500,50],[0,0,1]],D=[0]*5,
                base_to_optical=dict(R=np.eye(3).tolist(),t=[0,0,0]))
    depth = np.full((101,101),500,dtype=np.uint16)
    point = np.array([0.,0.,.5])
    view = check_view(point,np.array([1.,0.,0.]),depth,info)
    assert view['usable']
    np.testing.assert_allclose(view['normal'],[0,0,-1],atol=1e-10)
    assert not check_view(point,np.array([1.,0.,0.]),depth-50,info)['usable']


def test_table_plane_cancels_shared_depth_offset():
    info = dict(K=[[500,0,50],[0,500,50],[0,0,1]],D=[0]*5,
                base_to_optical=dict(R=np.eye(3).tolist(),t=[0,0,0]))
    depth = np.full((101,101),506,dtype=np.uint16)
    point = np.array([0.,0.,.5])
    assert not check_view(point,np.array([1.,0.,0.]),depth,info)['usable']
    corrected = check_view(point,np.array([1.,0.,0.]),depth,info,
                           table_plane=np.array([0.,0.,.006]),table_z=0.)
    assert corrected['usable']
    assert abs(corrected['depth_residual_mm']) < 1e-9
    assert corrected['table_offset_mm'] == 6


def test_agreement_abstention_and_direction_veto():
    a=dict(usable=True,normal=[0,0,1],normal_dot_pull=0.)
    assert decide({'a':a,'hidden':dict(usable=False)})[0]=='pass'
    assert decide({'a':a,'b':a})[0]=='pass'
    assert decide({'a':a,'b':dict(usable=True,normal=[1,0,0],normal_dot_pull=-1)})[0]=='uncertain'
    assert decide({'a':dict(a,normal_dot_pull=.01)})[0]=='reject'
    assert decide({'hidden':dict(usable=False)})[0]=='uncertain'


def test_small_top_tilt_uses_same_tolerance_as_selector():
    # A 1.5-degree top slope is valid under the selector's existing 0.1 gate.
    gentle = dict(usable=True, normal=[-.026, 0., .99966], normal_dot_pull=.026)
    assert decide({'cam2': gentle}, max_normal_dot_pull=.1)[0] == 'pass'
    assert decide({'cam2': dict(gentle, normal_dot_pull=.2)}, max_normal_dot_pull=.1)[0] == 'reject'


def test_missing_views_veto_copy_only(tmp_path):
    baseline=dict(press=dict(available=True,point=[0,0,.5],geometry=dict(direction=[1,0,0])),
                  planar_push=dict(available=True))
    from test_contact_point_selector import box
    points, _ = box()
    np.savez(tmp_path/'object_cloud.npz', xyz=points)
    result,report=evaluate(tmp_path,baseline,rule='view_check_v2')
    assert report['decision']=='unavailable'
    assert not result['press']['available']
    assert baseline['press']['available']
    assert result['planar_push']==baseline['planar_push']


def _full_image_support(point, depth, info, table_offset=0.):
    """Reference for check_view's 12 mm neighborhood, computed over the WHOLE
    image (the pre-crop implementation), as (count, unit normal up to sign)."""
    import cv2
    K = np.asarray(info['K']); R = np.asarray(info['base_to_optical']['R'])
    t = np.asarray(info['base_to_optical']['t'])
    q = R@point+t
    uv = np.rint((K@q)[:2]/q[2]).astype(int)
    measured = depth[uv[1], uv[0]]/1000.
    v, u = np.indices(depth.shape); z = depth.astype(float)/1000.
    base = (np.stack([(u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z], axis=-1)-t)@R
    base[:, :, 2] -= table_offset
    distances = np.linalg.norm(base-point, axis=-1)
    mask = ((depth > 0) & (distances <= .012) & (np.abs(z-measured) <= .008)).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask, connectivity=4)
    local = base[labels == labels[uv[1], uv[0]]]
    local = local-local.mean(0)
    return len(local), np.linalg.eigh(local.T@local)[1][:, 0]


def test_cropped_neighborhood_matches_full_image():
    """check_view crops to a pixel window around the projection for speed; the
    support it finds must be exactly the full-image one, including candidates
    at the image border, on a depth step, and with a table-plane correction."""
    c, s = np.cos(np.radians(35)), np.sin(np.radians(35))
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # oblique view of the scene
    info = dict(K=[[610, 0, 320], [0, 610, 240], [0, 0, 1]], D=[0]*5,
                base_to_optical=dict(R=R.tolist(), t=[0., 0., 0.]))
    v, u = np.indices((480, 640))
    depth = (450+.08*u+.15*v).astype(np.uint16)       # sloped surface
    depth[200:260, 300:380] -= 25                      # raised block: depth step
    K = np.asarray(info['K'])
    checked = 0
    for (pu, pv) in [(0, 0), (639, 479), (5, 240), (320, 230), (300, 200), (379, 259),
                     (340, 262), (600, 30), (150, 400)]:
        z = depth[pv, pu]/1000.
        optical = np.array([(pu-K[0, 2])*z/K[0, 0], (pv-K[1, 2])*z/K[1, 1], z])
        for plane, offset in ((None, 0.), (np.array([0., 0., -.023]), -.003)):
            point = R.T@optical+[0, 0, offset]
            view = check_view(point, np.array([1., 0., 0.]), depth, info,
                              table_plane=plane, table_z=-.02)
            if 'count' not in view:  # exited before the neighborhood step
                continue
            count, normal = _full_image_support(point, depth, info, offset)
            assert view['count'] == count
            checked += 1
            if view['usable']:
                assert abs(abs(np.dot(view['normal'], normal))-1) < 1e-9
    assert checked >= 10


def test_no_support_points_near_the_table_does_not_crash(tmp_path):
    """The box sits above the hard-coded table (z=-0.021), so the cloud pivot estimate
    fails: no press without a SAM pivot, and the SAM pivot is used when given."""
    from test_contact_point_selector import box
    points, _ = box()
    np.savez(tmp_path/'object_cloud.npz', xyz=points)
    baseline = dict(press=dict(available=False), planar_push=dict(available=True))
    result, report = evaluate(tmp_path, baseline, cameras={})
    assert not result['press']['available'] and 'cloud_pivot_error' in report
    result, report = evaluate(tmp_path, baseline, cameras={},
                              press_pivot_override=[.5, 0., 0.])
    assert report['geometry']['pivot'] == [.5, 0., 0.]
    assert report['geometry']['pivot_source'] == 'rgb_sam_override'


def test_press_alternatives_need_more_torque_little_sideways_and_spacing():
    from irb120_perception.experimental_view_check import press_alternatives
    press = dict(point=[.6, 0., .1], ball_center=[.6, 0., .11],
                 geometry=dict(axis=[0., -1., 0.], pivot=[.65, 0., 0.]))
    rows = [dict(point=[.6, 0., .1], decision='pass', torque_per_pull_N_m=.10),     # the press itself
            dict(point=[.59, 0., .1], decision='pass', torque_per_pull_N_m=.12),    # kept, best
            dict(point=[.585, 0., .1], decision='pass', torque_per_pull_N_m=.115),  # too close to it
            dict(point=[.57, .005, .1], decision='pass', torque_per_pull_N_m=.11),  # kept
            dict(point=[.59, .03, .1], decision='pass', torque_per_pull_N_m=.13),   # 30 mm sideways
            dict(point=[.55, 0., .1], decision='reject', torque_per_pull_N_m=.2),   # failed the check
            dict(point=[.61, 0., .1], decision='pass', torque_per_pull_N_m=.09)]    # less torque
    alternatives = press_alternatives(rows, press, lambda p: [0., 0., 1.])
    assert [a['point'].tolist() for a in alternatives] == [[.59, 0., .1], [.57, .005, .1]]
    assert [a['rank'] for a in alternatives] == [1, 2]
    np.testing.assert_allclose(alternatives[0]['ball_center'], [.59, 0., .11])
    assert alternatives[0]['torque_gain'] == pytest.approx(.02)
    assert alternatives[1]['lateral_offset_m'] == pytest.approx(-.005)


def test_press_backups_are_passing_contacts_most_centered_first():
    from irb120_perception.experimental_view_check import press_backups
    press = dict(point=[.6, .015, .1], ball_center=[.6, .015, .11],
                 geometry=dict(axis=[0., -1., 0.], pivot=[.57, 0., 0.]))
    rows = [dict(point=[.6, .015, .1], decision='pass', torque_per_pull_N_m=.10),   # the press itself
            dict(point=[.62, .001, .1], decision='pass', torque_per_pull_N_m=.08),  # B1: 1 mm off center
            dict(point=[.625, .004, .1], decision='pass'),                          # too close to B1
            dict(point=[.61, 0., .1], decision='reject'),                           # failed the check
            dict(point=[.6, -.02, .1], decision='pass', torque_per_pull_N_m=.09)]   # B2: 20 mm off
    backups = press_backups(rows, press, lambda p: [0., 0., 1.])
    assert [b['point'].tolist() for b in backups] == [[.62, .001, .1], [.6, -.02, .1]]
    assert backups[0]['along_edge_offset'] == pytest.approx(-.001)
    np.testing.assert_allclose(backups[0]['ball_center'], [.62, .001, .11])


def test_cam2_press_uses_observed_coordinates_and_ignores_other_view_veto(tmp_path, monkeypatch):
    from irb120_perception import experimental_view_check as module
    from irb120_perception.contact_point_selector import select_contact_points
    from test_contact_point_selector import box
    points, normals = box()
    points[:, 2] -= .021
    camera_id = np.where(points[:, 0] >= .55, 2, 1)
    np.savez(tmp_path/'object_cloud.npz', xyz=points, camera_id=camera_id)
    baseline = select_contact_points(points, normals=normals)
    calls = []
    monkeypatch.setattr(module, 'estimate_table_plane', lambda *a: np.array([0., 0., -.008]))

    def check(point, direction, depth, info, table_plane=None):
        calls.append(table_plane)
        # A side camera's positive normal must remain diagnostic for cam2 mode.
        return dict(usable=True, normal=[0., 0., 1.], normal_dot_pull=info['dot'])

    monkeypatch.setattr(module, 'check_view', check)
    cameras = {cam: (np.ones((2, 2), dtype=np.uint16), dict(dot=dot))
               for cam, dot in [('cam1', 1.), ('cam2', 0.)]}
    selected, report = evaluate(tmp_path, baseline, cameras=cameras,
                                rule='view_check_v2', press_camera='cam2',
                                press_pivot_override=[.5, 0., -.021])
    assert selected['press']['available']
    assert selected['press']['point'][0] >= .55
    assert calls and all(p is None for p in calls)
    assert report['press_camera'] == 'cam2'
    for mode in ('planar_push', 'forward_tip'):
        np.testing.assert_array_equal(selected[mode]['point'], baseline[mode]['point'])
    np.savez(tmp_path/'object_cloud.npz', xyz=points)
    with pytest.raises(ValueError, match='provenance'):
        evaluate(tmp_path, baseline, cameras=cameras, rule='view_check_v2', press_camera='cam2')
