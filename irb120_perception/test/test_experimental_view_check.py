"""Temporary camera agreement experiment checks; no ROS required."""
import numpy as np
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


def test_agreement_abstention_and_direction_veto():
    a=dict(usable=True,normal=[0,0,1],normal_dot_pull=0.)
    assert decide({'a':a,'hidden':dict(usable=False)})[0]=='pass'
    assert decide({'a':a,'b':a})[0]=='pass'
    assert decide({'a':a,'b':dict(usable=True,normal=[1,0,0],normal_dot_pull=-1)})[0]=='uncertain'
    assert decide({'a':dict(a,normal_dot_pull=.01)})[0]=='reject'
    assert decide({'hidden':dict(usable=False)})[0]=='uncertain'


def test_missing_views_veto_copy_only(tmp_path):
    baseline=dict(press=dict(available=True,point=[0,0,.5],geometry=dict(direction=[1,0,0])),
                  planar_push=dict(available=True))
    from test_contact_point_selector import box
    points, _ = box()
    np.savez(tmp_path/'object_cloud.npz', xyz=points)
    result,report=evaluate(tmp_path,baseline)
    assert report['decision']=='unavailable'
    assert not result['press']['available']
    assert baseline['press']['available']
    assert result['planar_push']==baseline['planar_push']
