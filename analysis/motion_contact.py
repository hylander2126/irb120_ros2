"""Planar instantaneous contact from ball motion and a horizontal table.

No object dimensions or shape profile. Requires nonrotating spherical finger,
rolling on a flat object top, no slip at ground, upright angle zero.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def reconstruct(time,ball,reference_index,window_s=.6,ball_radius=.01325,table_z=0.):
    t=np.asarray(time)-time[reference_index];b=np.asarray(ball)
    dt=float(np.median(np.diff(t)));n=max(7,int(round(window_s/dt))|1)
    n=min(n,len(t)//2*2-1)
    smooth=savgol_filter(b,n,3,axis=0)
    vel=savgol_filter(b,n,3,deriv=1,delta=dt,axis=0)
    def rhs(tt,state):
        z=np.interp(tt,t,smooth[:,2]);vx=np.interp(tt,t,vel[:,0])
        return [vx/(table_z-z+ball_radius*np.cos(state[0]))]
    start=reference_index
    sol=solve_ivp(rhs,(t[start],t[-1]),[0.],t_eval=t[start:],rtol=1e-9,atol=1e-10,max_step=.05)
    theta=np.full(len(t),np.nan);theta[start:]=sol.y[0]
    omega=vel[:,0]/(table_z-smooth[:,2]+ball_radius*np.cos(theta))
    valid=np.isfinite(theta)&(np.abs(omega)>.005)
    pivot=np.c_[np.full(len(t),np.nan),np.full(len(t),b[start,1]),np.full(len(t),table_z)]
    pivot[valid,0]=smooth[valid,0]+ball_radius*np.sin(theta[valid])-vel[valid,2]/omega[valid]
    return dict(theta=theta,pivot=pivot,valid=valid,omega=omega,smooth=smooth,window_samples=n)


def gravity_fit(theta,tau,arm,height,mask):
    use=mask&np.isfinite(theta)&np.isfinite(tau)&np.isfinite(arm)
    if use.sum()<10:raise ValueError('Too few moving samples')
    x=arm[use];s=np.sin(theta[use]);y=tau[use]
    coeff=np.linalg.lstsq(np.c_[x,-s],y/9.81,rcond=None)[0]
    m=coeff[0];h=coeff[1]/m if abs(m)>1e-12 else height/2
    sol=least_squares(lambda p:9.81*p[0]*(x-p[1]*s)-y,
        [max(m,.001),np.clip(h,0,height)],bounds=([1e-8,0],[np.inf,height]),
        ftol=1e-12,xtol=1e-12,gtol=1e-12)
    return dict(m=float(sol.x[0]),z=float(sol.x[1]),rms_mNm=float(np.sqrt(np.mean(sol.fun**2))*1000),
                n=int(use.sum()),active_bounds=sol.active_mask.tolist(),condition=float(np.linalg.cond(np.c_[x,-s])))


def origin_x(ball,theta,reference_index,center_x,ball_radius=.01325):
    # Object body origin initially at (cloud axis x, table z=0).
    # Top rolling advances the ball center by r*theta in object coordinates.
    b0=ball[reference_index]
    return ball[:,0]-((b0[0]-center_x+ball_radius*theta)*np.cos(theta)-b0[2]*np.sin(theta))

def regularize_support(reconstruction,ball,branch,degree=3,ball_radius=.01325):
    """Robust branchwise z(theta) derivative avoids dividing two small velocities.

    Cubic is a fixed low-order smoother, not a bottom-profile or GT prior.
    """
    r=reconstruction;theta=r['theta'];pivot=r['pivot'].copy();valid=r['valid'].copy()
    coefficients={}
    for name in ['tip','return']:
        use=(branch==name)&valid&(theta>np.deg2rad(1))
        scale=max(float(np.nanmax(theta[use])),.01)
        v=theta[use]/scale
        X=np.polynomial.polynomial.polyvander(v,degree)
        initial=np.linalg.lstsq(X,ball[use,2],rcond=None)[0]
        sol=least_squares(lambda c:X@c-ball[use,2],initial,loss='soft_l1',f_scale=.0002,
                          xtol=1e-12,ftol=1e-12,gtol=1e-12)
        slope=np.polynomial.polynomial.polyval(v,np.polynomial.polynomial.polyder(sol.x))/scale
        pivot[use,0]=ball[use,0]+ball_radius*np.sin(theta[use])-slope
        coefficients[name]=sol.x.tolist()
        valid[(branch==name)&~use]=False
    valid&=np.isin(branch,['tip','return'])
    return dict(**{k:v for k,v in r.items() if k not in ['pivot','valid']},pivot=pivot,valid=valid,
                derivative_coefficients=coefficients,degree=degree)
