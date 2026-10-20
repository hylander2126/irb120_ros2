"""Physical fit bounds from the latest pre-action object cloud, never CAD/GT."""
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

MIN_MASS_KG = 1e-8  # 0.00001 gram

def cloud_height_bound(log):
    stage = Path(log.path).parent
    episode = Path(log.episode_dir) if log.episode_dir else stage.parent
    try:
        number = int(stage.name.split('_')[0])
    except ValueError as exc:
        raise ValueError('Physical height bound requires an episode stage and pre-action object cloud') from exc
    folders = sorted(p for p in episode.glob('*_perceive')
                     if int(p.name.split('_')[0]) < number and (p/'object_cloud.npz').exists())
    if not folders:
        raise ValueError('Physical height bound requires a pre-action object_cloud.npz')
    path = folders[-1]/'object_cloud.npz'
    xyz = np.load(path)['xyz']; z = xyz[np.isfinite(xyz).all(axis=1), 2]
    # Saved perception clouds use base_link; table is world z=0, base_link z=21 mm.
    height = float(z.max()+.021) if len(z) else float('nan')
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Invalid cloud-derived height bound')
    return height, str(path)

def bounded_gravity_fit(phi, tau, distance, radius, height):
    if not (np.isfinite(height) and height > 0 and distance > 0):
        raise ValueError('Positive distance and finite positive cloud height required')
    X=np.c_[np.cos(phi),-np.sin(phi)]
    raw=np.linalg.lstsq(X,tau/9.81,rcond=None)[0]
    m=raw[0]/distance
    h=radius+raw[1]/m if abs(m)>1e-15 else height/2
    def residual(p):
        return 9.81*p[0]*(distance*np.cos(phi)-(p[1]-radius)*np.sin(phi))-tau
    sol=least_squares(residual,[max(m,1e-4),np.clip(h,0,height)],
        bounds=([MIN_MASS_KG,0],[np.inf,height]),xtol=1e-12,ftol=1e-12,gtol=1e-12)
    if not sol.success: raise ValueError('Bounded gravity fit failed: '+sol.message)
    m,h=sol.x
    active=[]
    if m<=MIN_MASS_KG+1e-7:active.append('mass_lower')
    if h<1e-7:active.append('height_lower')
    if height-h<1e-7:active.append('height_upper')
    return float(m),float(h),active
