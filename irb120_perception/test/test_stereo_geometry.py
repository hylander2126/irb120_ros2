import numpy as np
import pytest
from irb120_perception.stereo_geometry import disparity_depth, project_depth


def test_metric_scale_principal_offset_and_invalid_disparity():
    d = disparity_depth(np.array([52., 2., np.nan, -1., 502.]), 1000., .05, 12., 10.)
    np.testing.assert_allclose(d, [1., 0., 0., 0., 0.])
    with pytest.raises(ValueError):
        disparity_depth(np.array([10.]), 100., -.05)


def test_identity_preserves_depth_and_holes():
    depth=np.array([[.5,0],[1.,.75]])
    K=np.array([[100.,0,0],[0,100.,0],[0,0,1]])
    aligned, ids, visible=project_depth(depth,K,np.eye(3),np.zeros(3),K,depth.shape)
    np.testing.assert_allclose(aligned,depth)
    np.testing.assert_array_equal(visible,depth>0)
    np.testing.assert_array_equal(ids[visible],np.arange(4).reshape(2,2)[visible])


def test_occluded_background_is_not_assigned_foreground_rgb_mask():
    K=np.eye(3); C=np.diag([.1,1.,1.])
    aligned,ids,visible=project_depth(np.array([[1.,2.]]),K,np.eye(3),np.zeros(3),C,(1,1))
    np.testing.assert_allclose(aligned,[[1.]])
    np.testing.assert_array_equal(visible,[[True,False]])


def test_optical_translation_uses_color_z_not_left_z():
    aligned,_,visible=project_depth(np.array([[1.]]),np.eye(3),np.eye(3),np.array([0,0,.2]),np.eye(3),(1,1))
    np.testing.assert_allclose(aligned,[[1.2]])
    assert visible.item()


def test_pixel_footprint_resampling_does_not_fill_missing_source_depth():
    depth=np.array([[1.,0.,1.]])
    K=np.eye(3); C=np.array([[2.,0,.5],[0,2.,.5],[0,0,1.]])
    aligned,_,_=project_depth(depth,K,np.eye(3),np.zeros(3),C,(2,6),rasterize=True)
    np.testing.assert_allclose(aligned,[[1,1,0,0,1,1],[1,1,0,0,1,1]])
