import numpy as np
import pytest
from scipy.integrate import solve_ivp
from cloud_geometry import infer_profile
from motion_contact import reconstruct,origin_x

@pytest.mark.parametrize('radius',[0.,.008,.016,.024,.030])
def test_cloud_profile_without_configuration(radius):
    z=np.repeat(np.linspace(.0001,.15,140),32);az=np.tile(np.linspace(0,2*np.pi,32,endpoint=False),140)
    zz=np.minimum(z,radius);rho=.04-radius+np.sqrt(np.maximum(0,radius**2-(zz-radius)**2))
    points=np.c_[.65+rho*np.cos(az),.003+rho*np.sin(az),z]
    q=infer_profile(points)
    assert q['outer_radius_m']==pytest.approx(.04,abs=1e-6)
    assert q['fillet_radius_m']==pytest.approx(radius,abs=1e-5)
    np.testing.assert_allclose(q['center_xy'],[.65,.003],atol=1e-6)

@pytest.mark.parametrize('variable_curvature',[False,True])
def test_contact_reconstruction_without_shape_prior(variable_curvature):
    # Manufacture smooth planar rolling with prescribed instantaneous support.
    t=np.linspace(0,10,2001);theta=.04*t;rf=.01325
    def px(th):return .61-(.02*th+.01*th*th if variable_curvature else 0)
    def rhs(th,b):return [-b[1]+rf*np.cos(th),b[0]-px(th)+rf*np.sin(th)]
    sol=solve_ivp(rhs,(0,.4),[.615,.16325],t_eval=theta,rtol=1e-12,atol=1e-13)
    ball=np.c_[sol.y[0],np.zeros(len(t)),sol.y[1]]
    out=reconstruct(t,ball,0,window_s=.3)
    use=(t>1)&(t<9)
    np.testing.assert_allclose(out['theta'][use],theta[use],atol=1e-6)
    np.testing.assert_allclose(out['pivot'][use,0],px(theta[use]),atol=2e-6)


def test_missing_bottom_is_not_radius_evidence():
    # Sharp and 8 mm fillet share the exact same visible wall above 12 mm.
    z=np.repeat(np.linspace(.012,.15,100),32);az=np.tile(np.linspace(0,2*np.pi,32,endpoint=False),100)
    points=np.c_[.65+.04*np.cos(az),.04*np.sin(az),z]
    q=infer_profile(points)
    assert q['near_optimal_radius_range_m'][0]==0
    assert q['near_optimal_radius_range_m'][1]>.008
