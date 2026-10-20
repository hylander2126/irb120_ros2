"""Perception-only axisymmetric profile inference. No configuration/CAD/GT imports.

Input points must be in a frame with horizontal table z=0. Assumes upright,
axisymmetric body with a straight upper wall and circular lower transition.
This is a fitted shape family, not an arbitrary-shape reconstruction.
"""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def infer_profile(points):
    p=np.asarray(points,float);p=p[np.isfinite(p).all(1)]
    if len(p)<100:raise ValueError('Insufficient object points')
    height=float(np.quantile(p[:,2],.99))
    wall=p[(p[:,2]>.35*height)&(p[:,2]<.85*height)]
    xy=wall[:,:2];center=np.median(xy,axis=0)
    initial=np.r_[center,np.median(np.linalg.norm(xy-center,axis=1))]
    sol=least_squares(lambda q: np.linalg.norm(xy-q[:2],axis=1)-q[2],initial,
        bounds=([-np.inf,-np.inf,.001],[np.inf,np.inf,height]),loss='soft_l1',f_scale=.001)
    cx,cy,outer=sol.x
    lower=p[(p[:,2]>0)&(p[:,2]<.35*height)]
    rho=np.linalg.norm(lower[:,:2]-sol.x[:2],axis=1);z=lower[:,2]
    def residual(radius):
        zz=np.minimum(np.maximum(z,0),radius)
        pred=outer-radius+np.sqrt(np.maximum(0,radius**2-(zz-radius)**2))
        return rho-pred
    def cost(radius):
        r=residual(radius)/.001
        return float(np.mean(2*(np.sqrt(1+r*r)-1)))
    grid=np.linspace(0,outer*.98,201);costs=np.array([cost(v) for v in grid]);j=np.argmin(costs)
    lo=grid[max(j-1,0)];hi=grid[min(j+1,len(grid)-1)]
    best=minimize_scalar(cost,bounds=(lo,hi),method='bounded',options={'xatol':1e-10}).x
    if cost(0)<=cost(best)+1e-10:best=0.
    # A loss profile, not a calibrated confidence interval.
    near=grid[costs<=cost(best)+.05]
    return dict(center_xy=sol.x[:2].tolist(),outer_radius_m=float(outer),fillet_radius_m=float(best),
        height_m=height,lowest_observed_z_m=float(p[:,2].min()),
        wall_rms_mm=float(np.sqrt(np.mean(sol.fun**2))*1000),
        loss_radius_grid_m=grid.tolist(),loss_profile=costs.tolist(),
        near_optimal_radius_range_m=[float(near.min()),float(near.max())],
        lower_count=len(lower),wall_count=len(wall))
