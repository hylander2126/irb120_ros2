import numpy as np
from irb120_perception.contact_point_selector import estimate_pivot


def mixed_boundary(angle=0):
    theta=np.linspace(-np.pi/2,np.pi/2,100)
    xy=np.vstack([np.c_[.04*np.cos(theta),.04*np.sin(theta)],
                  np.c_[np.zeros(41),np.linspace(-.04,.04,41)]])
    a=np.radians(angle);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    return np.c_[xy@rot.T+[.55,0],np.full(len(xy),.005)]


def test_observed_aligned_flat_wins_on_mixed_curved_flat_boundary():
    g=estimate_pivot(mixed_boundary(),[-1,0,0],0)
    assert g['kind']=='extended_edge'
    assert g['preferred_flat_alternative']['accepted']
    np.testing.assert_allclose(g['pivot'],[.55,0,0],atol=1e-10)


def test_near_aligned_flat_is_diagnostic_for_fixed_plane_controller():
    g=estimate_pivot(mixed_boundary(10),[-1,0,0],0)
    assert g['preferred_flat_alternative']['accepted']
    assert abs(g['preferred_flat_alternative']['angle_deg']-10)<1e-6
    assert g['kind']=='isolated_point'
    np.testing.assert_array_equal(g['direction'],[-1,0,0])


def test_perpendicular_flat_cannot_override_front_curved_tangent():
    g=estimate_pivot(mixed_boundary(90),[-1,0,0],0)
    assert g['preferred_flat_alternative'] is None
    np.testing.assert_array_equal(g['axis'],[0,-1,0])


def test_missing_circular_arc_is_not_evidence_of_flat_edge():
    theta=np.linspace(np.radians(-145),np.radians(145),100)
    p=np.c_[.55+.04*np.cos(theta),.04*np.sin(theta),np.full(len(theta),.005)]
    g=estimate_pivot(p,[-1,0,0],0)
    assert g['kind']=='isolated_point'
    assert g['preferred_flat_alternative'] is None
    assert g['flat_edge_candidates']
    assert not any(e['accepted'] for e in g['flat_edge_candidates'])


def test_sparse_noisy_circles_keep_commanded_direction():
    rng=np.random.default_rng(7)
    for _ in range(50):
        theta=rng.uniform(0,2*np.pi,16)
        xy=np.c_[.55+.03*np.cos(theta),.03*np.sin(theta)]+rng.normal(0,.001,(16,2))
        g=estimate_pivot(np.c_[xy,np.full(16,.005)],[-1,0,0],0)
        np.testing.assert_array_equal(g['direction'],[-1,0,0])
        np.testing.assert_array_equal(g['axis'],[0,-1,0])


def test_isolated_pivot_lateral_position_uses_robust_footprint_center():
    theta=np.linspace(0,2*np.pi,64,endpoint=False)
    xy=np.c_[.55+.03*np.cos(theta),.012+.03*np.sin(theta)]
    # A duplicated, biased extreme must not set the tangent's lateral center.
    xy=np.vstack((xy,[[.52,.035],[.52,.036]]))
    g=estimate_pivot(np.c_[xy,np.full(len(xy),.005)],[-1,0,0],0)
    assert g['kind']=='isolated_point'
    assert abs(g['pivot'][1]-.012)<.003
