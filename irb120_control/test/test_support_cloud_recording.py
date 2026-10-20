"""The analysis recording must preserve lower points without changing contacts."""
import hashlib
import json
import numpy as np
import pytest
from irb120_control.util.support_cloud_recording import record_support_scene,reconstruct_view


def scene(tmp_path):
    obj=np.array([[-.01,-.01,0],[.01,.01,.10],[0,0,.15]])
    np.savez_compressed(tmp_path/'object_cloud.npz',xyz=obj)
    info=dict(K=[[100.,0,2],[0,100.,2],[0,0,1]],D=[0]*5,
              base_to_optical=dict(R=[[1,0,0],[0,-1,0],[0,0,-1]],t=[0,0,.5]))
    (tmp_path/'cam1_camera_info.json').write_text(json.dumps(info))
    depth=np.full((5,5),521,np.uint16);depth[0,0]=0
    np.savez_compressed(tmp_path/'cam1_aligned_depth.npz',depth_mm=depth,stamp=123.)
    return obj,info,depth


def test_retains_table_and_preserves_all_original_files(tmp_path):
    obj,info,depth=scene(tmp_path)
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    result=record_support_scene(tmp_path)
    for name,digest in before.items():assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==digest
    assert result['below_legacy_roi_points']==24
    with np.load(tmp_path/'support_scene_cloud.npz') as saved:
        np.testing.assert_allclose(saved['xyz'][:,2],-.021,atol=1e-8)
        assert np.all(saved['camera_id']==1)
        assert saved['pixel_uv'].shape==(24,2)
        uv=saved['pixel_uv'];xyz=saved['xyz']
        optical=xyz@np.array(info['base_to_optical']['R']).T+info['base_to_optical']['t']
        np.testing.assert_allclose(optical[:,:2]/optical[:,2,None]*100+2,uv,atol=1e-6)
    metadata=json.loads((tmp_path/'support_scene_cloud.json').read_text())
    assert metadata['views'][0]['depth_stamp']==123
    assert metadata['views'][1]['status']=='unavailable'


def test_no_cloud_fails_without_creating_artifact(tmp_path):
    scene(tmp_path)
    np.savez(tmp_path/'object_cloud.npz',xyz=np.empty((0,3)))
    with pytest.raises(ValueError,match='nonempty'):record_support_scene(tmp_path)
    assert not (tmp_path/'support_scene_cloud.npz').exists()


def test_recording_cannot_change_selector_result(tmp_path):
    from irb120_perception.contact_point_selector import select_contact_points
    # Realistic visible five-face box. Selector gets the same original segmented cloud.
    coords=[np.linspace(-.025,.025,9),np.linspace(-.025,.025,9),np.linspace(0,.15,16)]
    points=[]
    for axis in range(3):
        for edge in ([coords[axis][-1]] if axis==2 else [coords[axis][0],coords[axis][-1]]):
            other=[i for i in range(3) if i!=axis]
            grid=np.array(np.meshgrid(coords[other[0]],coords[other[1]])).reshape(2,-1).T
            face=np.zeros((len(grid),3));face[:,axis]=edge;face[:,other]=grid;points.append(face)
    obj=np.concatenate(points)
    scene(tmp_path);np.savez(tmp_path/'object_cloud.npz',xyz=obj)
    before=select_contact_points(obj,table_z=0)
    record_support_scene(tmp_path)
    after=select_contact_points(np.load(tmp_path/'object_cloud.npz')['xyz'],table_z=0)
    encode=lambda value:json.dumps(value,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
    assert encode(before)==encode(after)
