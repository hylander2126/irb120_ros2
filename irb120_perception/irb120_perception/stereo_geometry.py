"""ROS-free calibrated stereo projection. All lengths are metres.

Reconstruction stays on the rectified left grid. RGB is rectified separately;
z buffering supplies mask correspondence; optional source-pixel footprint
rasterization changes sampling resolution without filling missing source depth.
"""
import numpy as np


def disparity_depth(disparity, fx, baseline, cx_left=0., cx_right=0.,
                    min_depth=.2, max_depth=1.4, min_disparity=.5):
    if not (np.isfinite(fx) and np.isfinite(baseline) and fx > 0 and baseline > 0):
        raise ValueError('Positive finite focal length and calibrated baseline required')
    denominator = np.asarray(disparity) + cx_right - cx_left
    depth = np.divide(fx*baseline, denominator, out=np.zeros_like(denominator, dtype=float),
                      where=np.isfinite(denominator) & (denominator > min_disparity))
    depth[~np.isfinite(depth) | (depth < min_depth) | (depth > max_depth)] = 0
    return depth.astype(np.float32)


def project_depth(depth, K, R, t, color_K, color_shape, tolerance=.003, rasterize=False):
    """Return color-grid depth, native->color indices and native visibility mask.

    R,t map left-optical points into rectified RGB optical coordinates. Only
    the nearest surface at each target pixel is visible (3 mm tolerance).
    """
    y, x = np.indices(depth.shape)
    xyz = np.stack(((x-K[0,2])*depth/K[0,0],
                    (y-K[1,2])*depth/K[1,1], depth), -1)
    points = xyz @ R.T + t
    z = points[...,2]
    safe = np.where(np.isfinite(z) & (z > 0), z, 1.)
    u = np.rint(points[...,0]/safe*color_K[0,0]+color_K[0,2]).astype(int)
    v = np.rint(points[...,1]/safe*color_K[1,1]+color_K[1,2]).astype(int)
    h,w = color_shape
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(z) & (z > 0)
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    indices = np.clip(v,0,h-1)*w+np.clip(u,0,w-1)
    zbuf = np.full(h*w, np.inf)
    np.minimum.at(zbuf, indices[valid], z[valid])
    if rasterize:
        # Rasterize each measured source pixel's projected footprint. This is
        # resolution conversion, not interpolation across missing depth pixels.
        # Native geometry is never changed. Constant source Z inside one pixel.
        corners=[]
        for du,dv in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5)):
            offset=np.stack((du*depth/K[0,0],dv*depth/K[1,1],np.zeros_like(depth)),-1)
            q=points+offset@R.T
            qz=np.maximum(q[...,2],1e-8)
            corners.append(np.stack((q[...,0]/qz*color_K[0,0]+color_K[0,2],
                                     q[...,1]/qz*color_K[1,1]+color_K[1,2]),-1))
        lower=np.ceil(np.min(corners,axis=0)).astype(int)
        upper=np.floor(np.max(corners,axis=0)).astype(int)
        # Protect against pathological calibration/near-plane footprints.
        finite=(depth>0)&np.isfinite(depth)&(z>0)
        width=np.clip(upper[...,0]-lower[...,0]+1,0,8)
        height=np.clip(upper[...,1]-lower[...,1]+1,0,8)
        for dy in range(int(height[finite].max(initial=0))):
            for dx in range(int(width[finite].max(initial=0))):
                xx,yy=lower[...,0]+dx,lower[...,1]+dy
                take=finite&(dx<width)&(dy<height)&(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
                np.minimum.at(zbuf,(yy[take]*w+xx[take]),z[take])
    visible = valid & (z <= zbuf[indices]+tolerance)
    zbuf[~np.isfinite(zbuf)] = 0
    return zbuf.reshape(h,w).astype(np.float32), indices, visible
