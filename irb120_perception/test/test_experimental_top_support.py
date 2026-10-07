import numpy as np
from irb120_perception.experimental_top_support import (
    assess, has_upward_upper_surface, infer_horizontal_top, tipping_torque_scores,
)
from irb120_perception.contact_point_selector import _press_band


def patch():
    x,y=np.meshgrid(np.linspace(-.01,.01,11),np.linspace(-.01,.01,11))
    return np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])


def view(normal, dot):
    return {'cam':dict(usable=True,normal=normal,normal_dot_pull=dot)}


def test_near_tangent_top_passes_but_large_pull_slope_rejects():
    p=patch();n=np.array([0.,0.,1.])
    assert assess(np.zeros(3),n,p,view([.2,0,np.sqrt(.96)],.2))[0]=='pass'
    assert assess(np.zeros(3),n,p,view([.3,0,np.sqrt(.91)],.3))[0]=='reject'


def test_side_fused_normal_and_unsupported_center_reject():
    p=patch()
    assert assess(np.zeros(3),np.array([.8,0,.6]),p,{})[0]=='reject'
    assert assess(np.array([.01,0,0]),np.array([.4,0,np.sqrt(.84)]),p,{})[0]=='reject'


def test_disagreeing_camera_and_missing_evidence_abstain():
    n=np.array([0.,0.,1.]);p=patch()
    assert assess(np.zeros(3),n,p,view([1,0,0],-1))[0]=='uncertain'
    assert assess(np.zeros(3),n,p,{})[0]=='uncertain'


def test_3d_wrench_rewards_height_and_penalizes_inset():
    g=dict(pivot=np.zeros(3),direction=np.array([-1.,0,0]),axis=np.array([0.,-1,0]))
    scores=tipping_torque_scores(np.array([[0,0,.1],[.02,0,.1],[0,0,.12]]),g)
    np.testing.assert_allclose(scores,[.1,.08,.12])


def test_relative_torque_band_preserves_candidates_for_edge_centering():
    points=np.array([[0.,0.,.2],[.008,.02,.2],[.012,0,.2]])
    ids=np.arange(3)
    geometry=dict(pivot=np.zeros(3),axis=np.array([0.,-1.,0.]))
    pull_scores=np.full(3,.2)
    absolute=_press_band(points,ids,pull_scores,geometry,1.,.003,0.)
    relative=_press_band(points,ids,pull_scores,geometry,1.,.003,.05)
    assert list(absolute)==[True,False,False]
    assert list(relative)==[True,True,False]


def test_horizontal_top_is_inferred_from_supported_upper_rim():
    z=np.linspace(0.,.1,21)
    walls=[]
    for x in (-.04,.04):
        walls.append(np.c_[np.full(len(z),x),np.zeros(len(z)),z])
    for y in (-.03,.03):
        walls.append(np.c_[np.zeros(len(z)),np.full(len(z),y),z])
    candidates,report=infer_horizontal_top(np.vstack(walls),spacing=.005)
    assert report['available']
    assert len(candidates)>20
    np.testing.assert_allclose(candidates[:,2],report['top_z'])
    assert np.min(np.linalg.norm(candidates[:,:2],axis=1))<.005


def test_low_noisy_upward_normal_does_not_hide_missing_top():
    points=np.array([[0.,0.,z] for z in np.linspace(0.,.2,30)])
    normals=np.tile([1.,0.,0.],(len(points),1))
    normals[2]=[0.,0.,1.]
    assert not has_upward_upper_surface(points,normals)
    normals[-1]=[0.,0.,1.]
    assert has_upward_upper_surface(points,normals)


def test_inferred_top_still_requires_camera_evidence():
    n=np.array([0.,0.,1.]);p=patch()
    assert assess(np.zeros(3),n,p,{},require_local_support=False)[0]=='uncertain'
    assert assess(np.zeros(3),n,p,view([0,0,1],0),require_local_support=False)[0]=='pass'


def test_in_memory_camera_path_needs_no_depth_png(tmp_path, monkeypatch):
    from test_contact_point_selector import box
    from irb120_perception import experimental_view_check as module
    xyz,_=box();xyz[:,2]-=.021
    np.savez(tmp_path/'object_cloud.npz',xyz=xyz)
    calls=[]
    def camera_check(point,direction,depth,info,**kwargs):
        calls.append(depth)
        return dict(usable=True,normal=[0.,0.,1.],normal_dot_pull=0.)
    monkeypatch.setattr(module,'check_view',camera_check)
    baseline={'press':{'available':False}}
    depth=np.ones((10,10),dtype=np.uint16)
    result,report=module.evaluate(tmp_path,baseline,cameras={'cam':(depth,{})},
                                  rule='top_support_v1',render_overlays=False)
    assert calls and all(d is depth for d in calls)
    assert result['press']['available']
    assert not baseline['press']['available']
    assert not list(tmp_path.glob('*.png'))


def test_t_shape_with_missing_top_samples_uses_camera_validated_inference(tmp_path, monkeypatch):
    from irb120_perception import experimental_view_check as module

    def wall_x(x, ys, zs):
        y,z=np.meshgrid(ys,zs)
        return np.c_[np.full(y.size,x),y.ravel(),z.ravel()]

    def wall_y(y, xs, zs):
        x,z=np.meshgrid(xs,zs)
        return np.c_[x.ravel(),np.full(x.size,y),z.ravel()]

    zs=np.linspace(-.01,.15,33)
    # A wide cap on a narrower stem, with vertical faces but deliberately no
    # sampled horizontal top face—the failure mode in the recorded T object.
    xyz=np.vstack([
        wall_x(.55,np.linspace(-.03,.03,13),np.linspace(.10,.15,11)),
        wall_x(.73,np.linspace(-.03,.03,13),np.linspace(.10,.15,11)),
        wall_y(-.03,np.linspace(.55,.73,37),np.linspace(.10,.15,11)),
        wall_y(.03,np.linspace(.55,.73,37),np.linspace(.10,.15,11)),
        wall_x(.60,np.linspace(-.02,.02,9),zs),
        wall_x(.68,np.linspace(-.02,.02,9),zs),
        wall_y(-.02,np.linspace(.60,.68,17),zs),
        wall_y(.02,np.linspace(.60,.68,17),zs),
    ])
    np.savez(tmp_path/'object_cloud.npz',xyz=xyz)
    monkeypatch.setattr(module,'check_view',lambda *args,**kwargs:
        dict(usable=True,normal=[0.,0.,1.],normal_dot_pull=0.,count=100,depth_residual_mm=0.))
    result,report=module.evaluate(tmp_path,{'press':{'available':False}},
        cameras={'cam':(np.ones((2,2),dtype=np.uint16),{})},
        rule='top_support_v1',render_overlays=False)
    assert report['horizontal_top_inference']['available']
    assert result['press']['available']
    assert result['press']['normal']==[0.,0.,1.]
    assert abs(result['press']['along_edge_offset'])<.006
