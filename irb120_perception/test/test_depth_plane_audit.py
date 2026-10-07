import numpy as np

from irb120_perception.depth_plane_audit import fit_plane


def test_robust_plane_fit_rejects_outliers_and_reports_tilt():
    x,y=np.meshgrid(np.linspace(-.2,.2,30),np.linspace(-.1,.1,20))
    z=.02*x-.01*y+.12
    points=np.c_[x.ravel(),y.ravel(),z.ravel()]
    points=np.vstack((points,[[0,0,1],[1,1,-1]]))
    result=fit_plane(points)
    assert result['available']
    np.testing.assert_allclose(result['equation_z_ax_by_c'],[.02,-.01,.12],atol=1e-10)
    assert result['rms_mm']<1e-6
    assert 1.2<result['tilt_deg']<1.4


def test_plane_fit_abstains_on_too_few_points():
    assert not fit_plane(np.zeros((5,3)))['available']
